"""The in-worker steering wiring: what actually runs inside a vLLM
worker to serve the registry's hook ops per request (petri-bloom Phase-2
serving; the missing link between ``lens_steer``'s registry/selector
contract, ``lens_batch``'s position layer, and a live vLLM forward).

THE INSTALLATION ROUTE, and why (read WHAT vllm-lens 1.2.1 ACTUALLY
PROVIDES in ``lens_steer``'s module docstring first — it is the premise).
Three routes were on the table; the choice is a SUBCLASS of vllm-lens's
worker extension that registers ITS OWN torch forward hooks directly on
the decoder layers, one per steered layer, once, at worker startup:

1. **A stock persistent ``vllm_lens.Hook``** — rejected, on identity
   alone: the dispatcher calls ``hook.fn(ctx, hidden[start:end])`` with
   NO request identity of any kind (``_worker_ext._run_post_category`` —
   the ctx carries ``layer_idx``/``seq_len``/``saved``/``model`` and
   nothing else, and the fn does not even learn its own slice offsets),
   so a persistent hook cannot resolve the per-request H3 selector at
   all. Two further disqualifiers, each sufficient: ``seq_len`` is the
   QUERY length of this forward (no absolute positions, no
   prefill-vs-chunk-remnant distinction, no token ids); and
   ``_make_hook`` wraps the whole dispatch in ``try/except`` that LOGS A
   WARNING AND SKIPS — every refusal this module is built around would
   become vllm-lens's own documented silent-capture-nothing failure
   mode.
2. **A Hook whose fn reaches ``get_forward_context()`` and the model
   runner directly** — rejected for the same identity problem (the fn
   receives a row slice with no way to learn which request's rows it
   holds, and no runner reference: ``ctx.model`` is the MODEL), plus the
   same exception-swallowing dispatch.
3. **Our own ``worker_extension_cls``, not subclassing vllm-lens's** —
   rejected because vLLM allows exactly ONE worker extension class and
   the Phase-0 gate's probes (saved-sentinel capture, disable-flag,
   returned-modification evidence) go through the STOCK extension's RPC
   surface (``install_hooks``/``get_captured_states``/...). Subclassing
   ``HiddenStatesExtension`` keeps that whole surface AND occupies the
   single slot: the vllm-lens plugin's ``_patched_create_engine_config``
   injects its own class only ``if not self.worker_extension_cls``
   (read out of ``_activations_plugin.py``), so setting ours FIRST is
   respected, and both APIs coexist in one class.

Our hooks are registered with ``layer.register_forward_hook`` directly
(the same registration site ``install_hooks`` uses, via the same
``_get_layers`` discovery), NOT appended to ``_persistent_hooks`` —
deliberately outside vllm-lens's dispatch, so a refusal PROPAGATES out
of the forward pass and kills the step loudly instead of being logged
and skipped. That is the fail-loud property everything downstream
(``PrefillLedger``, the selector cross-checks, the response-mask id
match) depends on.

THE SPLIT RESIDUAL (the one silent-correctness trap). vLLM's
Llama-family decoder layers return a ``(hidden_states, residual)`` PAIR
whose TRUE residual stream is ``hidden_states + residual`` — vllm-lens
itself sums exactly this before handing rows to hooks
(``_hook_inner``'s ``hook_src[0] + hook_src[1]``) and writes each
hook's modification back as a DELTA onto ``hidden_states`` alone,
leaving ``residual`` untouched (``_apply_hook_delta``). This module
does the same: ops are computed against the summed true stream, the
delta ``applied - true`` lands on a lazily-cloned ``hidden_states``,
and the ``residual`` OBJECT passes through identically. Computing
against ``hidden_states`` alone would look plausible (additive ops
coincide) and silently break every gated/last-token op whose math READS
the hidden state.

PER-FORWARD MECHANICS. The FIRST-FIRING hook of a forward builds the
per-request view of the forward and the other hooks of the SAME forward
reuse it (the lens-vectorize landing, 2026-09-04 — before it every one
of the 32 hooked layers re-classified the batch, ~14 ms of a 258 ms
step at B=32). The forward is identified by the live forward context's
``attn_metadata`` OBJECT (``RealWorkerView.forward_key``): vLLM's
runner builds a fresh dict for every ``execute_model`` and the cache
holds a strong reference to the one it classified, so the id can never
be reused by a later step while it is cached — a plain ``id()`` or a
step counter from a model-level pre-hook would not do (the runner
calls the compiled-model wrapper's ``__call__``, which under
``enforce_eager`` dispatches straight to ``forward`` and never runs
``nn.Module`` pre-hooks). Every hook still re-reads the runner's request
count as the cheap per-layer check that the cached forward is the
live one; a view without a forward key (the CPU fakes) classifies at
every hook exactly as before. Within the classification,
``lens_batch.derive_batch_plan`` classifies the flattened rows,
``build_segment_contexts`` runs the un-skippable positions cross-check
and slices per-request token ids, and the resulting
:class:`~.lens_adapters.HookContext` drives
``WorkerObjectRegistry.apply``. The plan lattice runs only when some
scheduled request's ACTIVE object needs it (scale-0 legs and purely
row-local ops skip it — ``lens_batch.ops_need_plan``'s asymmetry).

THE VECTORIZED APPLY (the same landing). The per-request apply loop —
~14 single-row kernel launches and TWO host<->device syncs per
(request, layer), ~230 ms of the 258 ms step at B=32 — is replaced by
ONE apply per (object, declared facts) GROUP of decode rows per layer:
:func:`build_slot_groups` groups a forward's SINGLE-ROW selector-
carrying slots by object — decode slots under a
:class:`~.lens_adapters.DecodeRows` context (every decode row is at a
position >= 1, so no op consults a per-row position there), and the
one-row slots of row-local objects (ctx None) — the group's rows are
read as one contiguous slice (or one ``index_select`` on an index
tensor built once per forward), the object's ops run once on the
stacked rows, and the delta is written once. Every MULTI-ROW slot (a
prefill, chunked or whole, of any object) keeps the per-request apply:
its masks are per request (the response boundary is located in each
prompt), and — the 2026-09-04 gate's finding — stacking its rows with
anything else changes the row count the fp32 reductions see. Rows
outside every group — unhooked requests, scale-0 legs, other cells'
rows — are never read or written (the mixed-batch isolation tests pin
it). Every ELEMENTWISE step of the grouped apply is bit-identical to
the per-row path at any row count (the all-ones mask multiply that is
dropped is exact: IEEE ``x * 1.0 == x``). Every REDUCTION over the
hidden dim (the gate gemv, the LoRA GEMMs, the LoReFT affine pair) goes
through ``lens_adapters._row_reduction`` (``independent_rows`` at the
group apply sites) under the REDUCTION MODE — since the 2026-09-04
decision ONE stacked ``[B, K]`` call by default, which the gate measured
reducing in an M-dependent order (unequal from B=7 at K=2560/3136, for
rank-4 GEMMs at every K, and at prefill row counts everywhere; one bf16
rounding of the edit or of the sum), or the per-request single-row call
per row under ``SISTEER_LENS_STACKED_REDUCTIONS=0`` (bit-identical by
construction). The mode is attested (``lens_reduction_mode``, the apply
path). The per-row loop stays available as the REFERENCE path —
``SISTEER_LENS_PER_ROW_APPLY=1`` (everything per row, dense legacy
masks) or a comma list of op names / object kinds to serve per row —
and ``scripts/lens_vectorize_gate.py`` holds the served arithmetic to
THE REDUCTION CONTRACT on the GPU: exact for elementwise ops and under
the per-row mode, a bounded bf16-kernel-variance tolerance for reduction
ops under the stacked default (docs/lens_stacked_reductions_scoping.md).
Every TRACKED request's segment is then held to the PER-REQUEST prefill
invariants (``lens_batch.check_tracked_segment``, against that
request's own prompt-token length read off the runner state) — the
always-on defense against preemption-recomputes and chunked-prefill
remnants, valid under arbitrarily overlapping HTTP requests because it
needs no run boundary. The run-scoped
:class:`~.lens_batch.PrefillLedger` is an OPTIONAL additional
cross-check for the in-process exact-replay harness: consumed only
while ARMED (``begin_run``/``finish_run`` RPCs bracket one
``llm.generate`` call), exactly once per forward, at the LOWEST
installed layer's hook, over the TRACKED requests' segments only (so
an untracked request's prefill in the same forward never draws a
credit it was never armed with). The serving path never arms it, and
an unarmed ledger is simply not consulted — never a refusal. Any
genuine invariant violation CONDEMNS the worker: a latch poisons every
subsequent forward and the run-end check, so a mid-run failure can
never be followed by plausible-looking output.

THE SELECTION RECORD (:class:`SelectionRecord`, read over the
``sisteer_selection_record`` RPC) is the one thing here that is NOT part
of serving: a bounded, append-only note of WHICH OBJECT each request was
bound to, written at the apply site from the object ``select`` already
resolved. It exists because the serving gate's ``selection_honored``
check was otherwise an INFERENCE — the right object must have been
loaded because the output looks like that arm's output — which holds
only while outputs are composition-stable, i.e. only under
``VLLM_BATCH_INVARIANT=1``. Selection is a discrete fact, so it is
observed. Observation only, and structurally so: it returns ``None``,
it cannot raise (an internal error is counted, never propagated, so a
diagnostic can never condemn a worker), and nothing on the apply path
reads it back. Its own ``coverage`` string, shipped inside the record,
states what it does NOT cover — op-level mis-binding downstream of
selection reads as a clean pass here.

vLLM / vllm-lens INTERNALS THIS MODULE RELIES ON (every one read out of
the extracted vllm-lens 1.2.1 source or the pinned-vLLM layout it
targets; NONE executed on this box — see GPU VERIFICATION below):

======================================================  ==============
internal                                                confidence
======================================================  ==============
``runner.input_batch.num_reqs`` / ``.req_ids``          high — the
                                                        exact reads
                                                        vllm-lens's own
                                                        dispatch does
``runner.requests[req_id].sampling_params.extra_args``  high — ditto
``get_forward_context().attn_metadata`` as a dict of    high — ditto
per-layer metadata (single-element list unwrapped;      (vllm-lens
anything else refused)                                  takes ``[0]``
                                                        unconditionally;
                                                        we refuse >1)
``query_start_loc`` / ``seq_lens`` /                    medium-high —
``num_actual_tokens`` on the segmentation-bearing       vllm-lens reads
attention metadata group(s)                             the first two;
                                                        ``num_actual_
                                                        tokens`` is the
                                                        V1 flash-attn
                                                        field
``num_actual_tokens`` / ``non_spec_query_start_loc``    medium — field
/ ``num_spec_decodes`` on a hybrid model's              names read from
non-bearing group (Qwen3.5's gated-delta-net layers:    gdn_attn.py,
``GDNAttentionMetadata``), CROSS-CHECKED against the    whose builder
bearing group's segmentation — never authoritative      sets non_spec_
                                                        query_start_loc
                                                        = query_start_
                                                        loc when no spec
                                                        decode is in
                                                        flight; wrong →
                                                        refusal, never
                                                        a mis-row
``positions`` tensor read off the decoder-layer call:   high — both
``kwargs["positions"]`` first, else ``args[0]`` (hooks   conventions
register ``with_kwargs=True``; llama-style models call  read from the
``layer(positions, hidden, residual)`` positionally,    pinned tag's
qwen3_next-style hybrids call keyword-only with         model source;
positions THIRD in the signature, so args[0] there      neither present
would be hidden_states); M-RoPE models hand a           = refusal;
``(3, num_tokens)`` t/h/w stack, reduced to row 0 by    disagreeing
the plan path ONLY when all rows agree                  M-RoPE rows
(_normalize_positions — text-only batches always do)    (multimodal)
                                                        = refusal.
                                                        (vllm-lens's
                                                        own pre-hook
                                                        still encodes
                                                        the positional
                                                        one: ``args[1]``
                                                        = hidden)
``runner.query_start_loc.cpu`` /                        high — read from
``runner.optimistic_seq_lens_cpu`` /                    the pinned
``runner.input_batch.num_computed_tokens_cpu`` /        0.25.1 runner
``runner.mrope_positions.cpu`` (M-RoPE models): the     source: the
CPU sides of the buffers the runner fills in            runner fills
``_prepare_inputs`` and copies to the device (the       these on the
same-step mirrors of the device ``query_start_loc`` /   CPU and
``seq_lens`` / M-RoPE positions the hooks used to read  ``copy_to_gpu``s
with a ``.tolist()`` sync each). NB there is NO CPU     them; the
copy of the 1-D ``positions`` buffer at the tag (it is  equality to the
computed ON the device from ``num_computed_tokens`` +   device tensors is
the per-token query offset), so the CPU positions are   GPU-VERIFIED
DERIVED from ``num_computed_tokens_cpu`` + the CPU       once, at the
``query_start_loc`` — the same arithmetic — and         first classified
verified against the device tensor once                 forward, and a
(``RealWorkerView.forward_positions``)                  mismatch refuses
``runner.requests[req_id].prompt_token_ids`` +          HIGH for the
``.output_token_ids`` as the request's known tokens,    FIELDS: both are
with the fed decode token already appended when the     declared on
hook fires                                              CachedRequestState
                                                        at the pinned tag
                                                        (v1/worker/gpu_
                                                        input_batch.py
                                                        :35-44, read from
                                                        v0.25.1), and
                                                        prompt_token_ids
                                                        is Optional there
                                                        — None under a
                                                        prompt-embeds
                                                        request, which
                                                        this module
                                                        refuses. MEDIUM
                                                        for the decode
                                                        TIMING: that the
                                                        fed token is
                                                        already in
                                                        output_token_ids
                                                        when the hook
                                                        fires. The
                                                        per-segment length
                                                        check turns a
                                                        wrong guess into a
                                                        refusal, never a
                                                        mis-slice
``worker.model_config.enforce_eager``,                  medium — config
``worker.vllm_config`` (v2-runner flag),                unpacking per
``next(runner.model.parameters())`` device/dtype        vLLM WorkerBase
                                                        (the last is
                                                        vllm-lens's own
                                                        idiom)
======================================================  ==============

GPU VERIFICATION REQUIRED (nothing below has executed against a real
vLLM; each is marked ``GPU-VERIFY`` at its use site, and each failure
mode is a loud startup/first-request refusal, never a silent no-op):

- the whole :class:`RealWorkerView` attribute surface, above;
- :class:`SisteerWorkerExtension`'s ``collective_rpc`` glue (install /
  begin_run / finish_run / abort_run / scheduler_config) and the
  tokenizer load from ``model_config.tokenizer``;
- the decode-token timing assumption (the fed token is in
  ``output_token_ids`` before the forward) — a violation shows up as
  the per-segment token-length refusal on the FIRST decode step;
- that torch forward-hook exceptions propagate out of vLLM's V1 model
  execution as a failed step (they do in torch; vLLM's handling of the
  resulting engine error is the unverified half);
- hook coexistence/ordering with vllm-lens's own ``install_hooks``
  hooks when the gate arms both.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Sequence as _ABCSequence
from dataclasses import dataclass, field, replace as dc_replace
from pathlib import Path
from typing import (Any, Callable, Dict, FrozenSet, List, Mapping, Optional,
                    Sequence, Tuple, Union)

import torch

from sisteer.capeval.serving import lens_adapters as la
from sisteer.capeval.serving import lens_batch as lb
from sisteer.capeval.serving import lens_steer as ls
from sisteer.capeval.serving import vllm_patches as _vp
from sisteer.capeval.serving import mamba_backend as _mb
# the pure report functions the sisteer_scheduler_config RPC is built
# from live in the PLUGIN-FREE worker_reports module (the lora-native
# worker extension composes the same functions without vllm-lens);
# imported back by name so every reference below, the RPC body, and
# ``lw.LensWorkerError`` identity are unchanged.
from sisteer.capeval.serving.worker_reports import (  # noqa: F401
    BATCH_INVARIANT_MODE_ATTR, BATCH_INVARIANT_MODE_ATTR_VERSIONS,
    BATCH_INVARIANT_PUBLIC_READER, SCHEDULER_CONFIG_FIELDS, LensWorkerError,
    batch_invariant_active, batch_invariant_report, kv_group_report,
    scheduler_config_report)

#: Applied at IMPORT on purpose: vllm serve resolves --worker-extension-cls in
#: the worker process before load_model, and CustomOp binds its dispatch at
#: __init__, so this is the one moment a class-level patch reaches every
#: rotary instance. False only outside a vLLM process (nothing to patch).
MROPE_PATCH_APPLIED = _vp.patch_mrope_interleaved_layout()

log = logging.getLogger(__name__)


#: bound on the per-request failure ledger (contained AdapterErrors
#: waiting to be claimed over ``sisteer_take_request_failure``). The
#: proxy claims each lens request's entry when its response returns, so
#: the ledger only grows when nothing is collecting — exactly the
#: misconfiguration that must not keep serving; overflow condemns.
REQUEST_FAILURES_CAP = 1024


#: The engine-args string for vLLM's single ``worker_extension_cls``
#: slot. The vllm-lens plugin respects an already-set value
#: (``_activations_plugin._patched_create_engine_config`` injects its
#: own only ``if not self.worker_extension_cls``), so passing this
#: keeps the stock extension's whole RPC surface via inheritance.
WORKER_EXTENSION_CLS = (
    "sisteer.capeval.serving.lens_worker.SisteerWorkerExtension")

#: The dotted runner attributes the wiring reads per forward — the V1
#: model-runner surface vllm-lens's own dispatch reads (plus the request
#: token fields, which are checked at request time by direct attribute
#: access, never ``getattr`` defaults). ``RealWorkerView.
#: missing_runner_attrs`` walks these; preflight refuses on any absence.
REQUIRED_RUNNER_ATTRS = (
    "input_batch.num_reqs",
    "input_batch.req_ids",
    "requests",
    "model",
)

#: The runner's CPU-side mirrors of the per-forward metadata the hooks
#: classify from (the lens-vectorize landing): the CPU halves of the
#: ``CpuGpuBuffer`` / pinned buffers vLLM 0.25.1's ``_prepare_inputs``
#: fills and copies to the device every step. Walked by
#: ``RealWorkerView.missing_runner_attrs`` next to REQUIRED_RUNNER_ATTRS,
#: so a runner without them refuses at PREFLIGHT (the structural half);
#: their VALUES are held equal to the device tensors once, at the first
#: classified forward (the GPU-VERIFY half — nothing is in these buffers
#: before a forward, so materialization time has nothing to compare).
#: ``mrope_positions.cpu`` is checked on demand: only M-RoPE models have
#: the buffer, and only they hand the layers a (3, T) positions stack.
CPU_MIRROR_RUNNER_ATTRS = (
    "query_start_loc.cpu",
    "optimistic_seq_lens_cpu",
    "input_batch.num_computed_tokens_cpu",
)

#: ``SISTEER_LENS_PER_ROW_APPLY`` — the switch back to the per-row apply
#: (the pre-vectorize REFERENCE path). ``1`` / ``all``: every steered slot
#: applied one request at a time with the dense legacy masks, byte for
#: byte the pre-2026-09-04 arithmetic (the GPU gate's reference standup).
#: A comma-separated list of HookOp op names (``relu_gated_vector``,
#: ``lora_oproj``, ...) and/or object kinds (``psr``, ``lora_all``, ...):
#: only objects carrying one of those stay per row — the per-KIND
#: fallback for a kind the gate finds outside its contract at some batch
#: size (a routing/masking defect; the bounded bf16-kernel variance of a
#: stacked reduction is NOT that — it has its own switch,
#: lens_adapters.STACKED_REDUCTIONS_ENV). Unset / ``0``: the vectorized
#: path for everything.
PER_ROW_APPLY_ENV = "SISTEER_LENS_PER_ROW_APPLY"
_PER_ROW_OFF = ("", "0", "false", "off", "no", "none")
_PER_ROW_ALL = ("1", "all", "true", "on", "yes")

#: ``SISTEER_LENS_SHADOW_COMPARE=1`` — the GATE-ONLY in-worker shadow compare
#: (lens-vectorize review #5, 2026-09-04): for every vectorized group at every
#: hooked layer the worker ALSO runs the legacy per-row path — the very
#: function the ``SISTEER_LENS_PER_ROW_APPLY=1`` reference runs
#: (:meth:`LensWorkerSteering._slot_apply` with dense legacy masks) — on a
#: clone of the SAME input rows, compares the two outputs exactly
#: (``torch.equal``) and, when they differ, records the deviation — max
#: |diff|, the elements that differed out of the elements compared, the
#: largest difference between the two EDITS beyond one rounding of the sum in
#: bf16 ulps of the per-row edit (lens_adapters.edit_excess_ulps), and how
#: concentrated the differences are (the largest unequal fraction of any one
#: row and of any one group) — per (kind, op, shape, positions, skip_bos,
#: phase, rows-in-group), and SERVES the per-row result, so a mismatch can
#: never leak into served outputs while the gate runs. A group whose cell
#: serves NO op at a hook's (site, layer) — every hook fires for every group
#: of the forward, and a multi-cell registry installs the union of the cells'
#: sites x layers — is counted UNSERVED (never keyed: no op to classify). It is the
#: authoritative fidelity check: one process, identical tensors, the
#: production kernels at production BI=0 — no admission / composition
#: confound. The GATE reads the record against the contract (scripts/
#: lens_vectorize_gate.py): exact for elementwise ops and under the per-row
#: mode; for a reduction op under the stacked mode, bf16-kernel-variance
#: equivalence to the per-row path, bounded. INERT WHEN OFF: the flag is read once at construction into
#: ``_shadow_compare``; the apply path tests that one attribute per group and
#: takes the ordinary branch — no clone, no second apply, no compare, no
#: record write (:class:`ShadowCompareRecord` stays at its zeros). Read over
#: ``sisteer_shadow_compare``. Default off; never set in production.
SHADOW_COMPARE_ENV = "SISTEER_LENS_SHADOW_COMPARE"
#: ``SISTEER_LENS_RHO_EMIT`` — OBSERVATION ONLY, default OFF (rho-emit probe,
#: 2026-09-08): at the two block-apply sites, after each modifying write, record
#: per (request, hook layer) the norm of the edit that was written and the
#: norm of the COMPOSED stream it was added to (:class:`RhoRecord`; read over
#: ``sisteer_rho_record``). Read ONCE at LensWorkerSteering init into
#: ``_rho_emit``; off, the apply path tests that one bool per modifying apply
#: and nothing else runs — no tensor is created, read or written. On, the
#: record reads ``delta`` and ``true_rows`` (neither aliases the written
#: tensor) and reduces them on device; nothing on the apply path reads the
#: record back. Never set in production.
RHO_EMIT_ENV = "SISTEER_LENS_RHO_EMIT"
APPLY_PATH_VECTORIZED = "vectorized"
APPLY_PATH_PER_ROW = "per_row"
#: the REDUCTION MODE tag every grouped apply path carries right after its
#: "vectorized" token (lens_adapters.reduction_mode(), read once at that
#: module's import): ``vectorized+stacked`` (the default) or
#: ``vectorized+per_row`` (SISTEER_LENS_STACKED_REDUCTIONS=0); with the shadow
#: compare ``vectorized+<mode>+shadow(serves per_row)``; with a per-kind
#: fallback ``vectorized+<mode>(per_row: <names>)``. The all-per-row reference
#: (SISTEER_LENS_PER_ROW_APPLY=1) forms no group, so it stays ``per_row``
#: untagged. A standup can therefore never read as the other mode.
APPLY_PATH_STACKED_TAG = "+" + la.REDUCTION_MODE_STACKED
APPLY_PATH_PER_ROW_TAG = "+" + la.REDUCTION_MODE_PER_ROW
APPLY_PATH_SHADOW_SUFFIX = "+shadow(serves per_row)"


def apply_path_string(mode: str, *, shadow: bool = False, per_row_names: Sequence[str] = ()
                      ) -> str:
    """The attested apply path for a grouped worker in reduction ``mode``
    (``stacked`` / ``per_row``): the string the gate and the A/B driver
    expect per standup, built in ONE place."""
    if mode not in la.REDUCTION_MODES:
        raise LensWorkerError(f"unknown reduction mode {mode!r}; expected one of {la.REDUCTION_MODES}")
    path = APPLY_PATH_VECTORIZED + "+" + mode
    if shadow:
        return path + APPLY_PATH_SHADOW_SUFFIX
    if per_row_names:
        return path + "(per_row: " + ",".join(sorted(per_row_names)) + ")"
    return path


#: the default-mode strings (what a production standup attests)
APPLY_PATH_VECTORIZED_STACKED = apply_path_string(la.REDUCTION_MODE_STACKED)
APPLY_PATH_VECTORIZED_PER_ROW = apply_path_string(la.REDUCTION_MODE_PER_ROW)
APPLY_PATH_SHADOW = apply_path_string(la.REDUCTION_MODE_STACKED, shadow=True)
APPLY_PATH_SHADOW_PER_ROW = apply_path_string(la.REDUCTION_MODE_PER_ROW, shadow=True)
#: the attestation key for the mode (worker scheduler-config report,
#: SelectionRecord, the standup endpoint records, the proxy's /v1/models
#: identity block): ``per_row`` | ``stacked``
LENS_REDUCTION_MODE_KEY = "lens_reduction_mode"
#: the install-payload key carrying the per-cell RESUMED-PREFILL predicate
#: (``lens_adapters.resumed_prefill_ok`` over each served cell's ops):
#: cell id -> bool. An EXTRA key the frozen launcher validator
#: (``vllm_server.validate_install_reports``) copies through untouched into
#: the attestation's top-level ``worker_reports`` — the attested channel the
#: proxy half reads the predicate from (``resumed_prefill_ok_cells_attested``).
RESUMED_PREFILL_OK_CELLS_KEY = "resumed_prefill_ok_cells"
#: its THREE-VALUED sibling — cell id -> ``lens_adapters.resumed_prefill_policy``
#: (``never`` / ``always`` / ``conditional``); the proxy half mints salts from
#: THIS one (``resumed_prefill_policy_cells_attested``); the bool map above is
#: it read as ``policy == "always"``.
RESUMED_PREFILL_POLICY_CELLS_KEY = "resumed_prefill_policy_cells"
#: the install-payload key carrying the per-cell SERVE DECLARATIONS:
#: harness cell id (variant spelling) ->
#: exactly ONE of ``serve_declaration``'s two closed shapes
#: (``{"serve_out_of_support": ["thinking"]}`` | ``{"serve_experiment":
#: <name>}``), DECLARATION-CARRYING cells only (the ``none__lens0`` twin
#: under its own id when its donor declares; undeclared cells absent), every
#: value bound to the LOADED ops at load
#: (``lens_steer._assert_ops_express_declarations``). Rides the same
#: extra-key path as the policy map into the attestation's top-level
#: ``worker_reports``; the strict reader is :func:`serve_declarations_attested`
#: and the proxy is handed its result (``RoutingProxy(lens_serve_declarations=)``).
LENS_SERVE_DECLARATIONS_KEY = "lens_serve_declarations"
#: The DECLARED thinking-ON header as each worker LOADED and cross-checked it
#: (preflight): the install payload key the sampled-rerun standup
#: (``mt_serving_up --thinking-on``) holds to the header it exported
#: (:func:`thinking_header_attested`); ``None`` on a worker whose loaded ops
#: carry no second header, ABSENT on a worker from a tree before this key.
LENS_THINKING_HEADER_KEY = "assistant_header_ids_thinking_on"


def attested_reduction_mode(attestation: Any) -> Tuple[Optional[str], Optional[str]]:
    """``(mode, error)`` from a ``steering_attestation.json`` document: the
    ``lens_reduction_mode`` every worker stamps in BOTH of its reports — the
    scheduler-config one (``scheduler_geometry.worker_reports[*]``, the
    ``sisteer_scheduler_config`` RPC) and the install one (the top-level
    ``worker_reports[*]``, ``sisteer_install_steering``; the same
    :func:`lens_adapters.reduction_mode` value). Every report present in
    either list must carry the key and ALL must agree on a known mode;
    otherwise ``(None, why)`` — a missing key (a worker from an older tree),
    a disagreement between workers or between the two report kinds, or an
    unknown value is named, never defaulted. The standup scripts write the
    result into their endpoint records and hand it to the proxy for its
    ``/v1/models`` identity block; a consumer that finds no key treats the
    server as pre-landing (per-row), which is exactly why an absent key must
    stay absent rather than be guessed here."""
    if not isinstance(attestation, Mapping):
        return None, f"attestation is not a mapping: {type(attestation).__name__}"
    sg = attestation.get("scheduler_geometry")
    lists = [("scheduler-config", (sg.get("worker_reports") if isinstance(sg, Mapping) else sg)),
             ("install", attestation.get("worker_reports"))]
    lists = [(name, reps) for name, reps in lists if reps]
    if not lists or not all(isinstance(reps, list) for _, reps in lists):
        return None, ("no scheduler_geometry.worker_reports / worker_reports list in the "
                      "attestation")
    modes: List[Any] = []
    for name, reps in lists:
        for i, r in enumerate(reps):
            if not isinstance(r, Mapping):
                return None, f"{name} worker {i}: report is not a mapping"
            if LENS_REDUCTION_MODE_KEY not in r:
                return None, (f"{name} worker {i}: no {LENS_REDUCTION_MODE_KEY} in its {name} "
                              "report (a worker from a tree that predates the reduction-mode "
                              "attestation)")
            modes.append(r[LENS_REDUCTION_MODE_KEY])
    distinct = sorted({str(m) for m in modes})
    if len(distinct) != 1:
        return None, f"worker reports disagree on {LENS_REDUCTION_MODE_KEY}: {distinct}"
    mode = distinct[0]
    if mode not in la.REDUCTION_MODES:
        return None, f"{LENS_REDUCTION_MODE_KEY}={mode!r} is not one of {la.REDUCTION_MODES}"
    return mode, None


#: Why a standup attested NO reduction mode: the reason
#: :func:`attested_reduction_mode` returned, written beside
#: :data:`LENS_REDUCTION_MODE_KEY` in the standup endpoint records
#: (``capctrl_serving_up.Standup.endpoint`` / ``lens_gate_standup``). Present
#: only on a record from a tree that HAS the attestation — a pre-landing
#: standup states neither key.
LENS_REDUCTION_MODE_ERROR_KEY = "lens_reduction_mode_error"
#: a NATIVE multi-LoRA standup's ``endpoint.json`` names its engine under
#: ``engine`` (``lora_serving_up.ENGINE`` =
#: ``serving_attestation.SERVING_ATTESTATION_KIND``) and records NO
#: ``lens_reduction_mode`` (it has no lens hook); the reduction mode a consumer
#: records for such a store is the EXPLICIT ``not_applicable`` (``contracts.
#: REDUCTION_MODE_NOT_APPLICABLE``), never the pre-landing ``per_row`` an absent
#: key would otherwise read as. Literals, pinned equal to their owners by test
#: (``tests/test_contracts_native_endpoint.py``): the two Modal copies of
#: :func:`endpoint_reduction_mode` import nothing by design and mirror these.
ENDPOINT_ENGINE_KEY = "engine"
ENGINE_KIND_LORA_NATIVE = "lora_native"
REDUCTION_MODE_NOT_APPLICABLE = "not_applicable"
#: the lens engine's name where a record spells one (a lens standup's record
#: carries no ``engine`` key at all today; the two Modal copies accept this
#: spelling and refuse any other unknown value, so this reader does the same
#: — ``contracts.SERVING_ENGINE_LENS``, pinned equal by test)
ENGINE_KIND_LENS = "lens"
#: the row-facing spelling the native core writes beside ``engine``
#: (``contracts.SERVED_ENGINE_FIELD``, pinned equal by test); read here only
#: to refuse a record whose two spellings of one engine disagree
SERVED_ENGINE_FIELD = "serving_engine"


class ReductionModeUnattested(RuntimeError):
    """A standup endpoint record states no reduction mode AND says why it
    could not determine one. Raised by :func:`endpoint_reduction_mode`
    instead of falling back to the pre-landing default — see that function."""


def endpoint_reduction_mode(endpoint: Any, *,
                            source: str = "standup endpoint record") -> str:
    """The reduction mode a CONSUMER must read out of a standup endpoint
    record — the one place that turns the record's two keys into the mode a
    served store may record as its identity. Three readings, and only three:

    * ``lens_reduction_mode`` NAMED (a known :data:`lens_adapters.
      REDUCTION_MODES` member) — that mode;
    * NEITHER ``lens_reduction_mode`` nor
      :data:`LENS_REDUCTION_MODE_ERROR_KEY` stated — a PRE-LANDING standup
      (every server from before the reduction-mode attestation landed, and
      every store frozen against one), which served ``per_row``;
    * ``lens_reduction_mode`` null WITH a stated
      :data:`LENS_REDUCTION_MODE_ERROR_KEY` — :exc:`ReductionModeUnattested`,
      naming the reason.

    The third case is the whole point. A post-landing standup ALWAYS writes
    the key, and writes null plus a reason when its workers attest nothing or
    DISAGREE (:func:`attested_reduction_mode`). ``record.get(key)`` cannot
    tell that apart from "key absent", so a consumer that defaults an
    unstated mode to ``per_row`` would record the default as the served
    identity of a server whose arithmetic nobody could establish — precisely
    what the attestation exists to prevent. A null with NO reason is still
    the pre-landing reading: the record stated nothing either way.

    THE ENGINE IS READ FIRST: a record whose
    :data:`ENDPOINT_ENGINE_KEY` is :data:`ENGINE_KIND_LORA_NATIVE` is a native
    multi-LoRA standup's — it has no lens hook, so NO ``lens_reduction_mode``
    is consulted and the reading is :data:`REDUCTION_MODE_NOT_APPLICABLE`
    (an explicit value, so no consumer's ``get(key, per_row)`` default ever
    lands on a native store); a native record that nevertheless STATES a
    lens mode is torn and refuses. The two Modal copies mirror this case.
    The engine is read the way ``contracts.endpoint_is_native`` reads it, so
    the two readers never disagree about one record: ``engine`` may be the
    bare kind (``endpoint.json``) or the ``/v1/models`` identity block's
    ``{"kind": ...}`` mapping; the row-facing :data:`SERVED_ENGINE_FIELD`
    beside it must agree (two spellings that disagree are a torn record);
    a value that is neither :data:`ENGINE_KIND_LENS` nor
    :data:`ENGINE_KIND_LORA_NATIVE` refuses (an unknown engine has no
    reduction mode anyone can read; the Modal copies refuse the same).
    """
    if not isinstance(endpoint, Mapping):
        raise ReductionModeUnattested(
            f"{source}: not a mapping ({type(endpoint).__name__}) — no "
            f"{LENS_REDUCTION_MODE_KEY} can be read from it")
    stated = endpoint.get(LENS_REDUCTION_MODE_KEY)
    engine = endpoint.get(ENDPOINT_ENGINE_KEY)
    if isinstance(engine, Mapping):
        # the /v1/models identity block spells the engine as a block; endpoint.json
        # spells the bare kind — one reading for both documents
        engine = engine.get("kind")
    row = endpoint.get(SERVED_ENGINE_FIELD)
    if engine is not None and row is not None and engine != row:
        raise ReductionModeUnattested(
            f"{source}: states {ENDPOINT_ENGINE_KEY}={engine!r} and {SERVED_ENGINE_FIELD}={row!r} — two "
            "spellings of one engine that disagree; a torn record, refusing to read either")
    engine = engine if engine is not None else row
    if engine == ENGINE_KIND_LORA_NATIVE:
        if stated is not None:
            raise ReductionModeUnattested(
                f"{source}: a native multi-LoRA standup ({ENDPOINT_ENGINE_KEY}="
                f"{ENGINE_KIND_LORA_NATIVE!r}) states {LENS_REDUCTION_MODE_KEY}={stated!r} — it has "
                "no lens hook to attest a reduction mode for; a torn record, refusing rather than "
                "reading either fact")
        return REDUCTION_MODE_NOT_APPLICABLE
    if engine is not None and engine != ENGINE_KIND_LENS:
        raise ReductionModeUnattested(
            f"{source}: {ENDPOINT_ENGINE_KEY}={engine!r} is neither {ENGINE_KIND_LENS!r} nor "
            f"{ENGINE_KIND_LORA_NATIVE!r} — no reduction mode can be read for an unknown engine")
    why_none = endpoint.get(LENS_REDUCTION_MODE_ERROR_KEY)
    if stated is None:
        if why_none:
            raise ReductionModeUnattested(
                f"{source}: {LENS_REDUCTION_MODE_KEY} is null and "
                f"{LENS_REDUCTION_MODE_ERROR_KEY} is {why_none!r} — the standup could not "
                "determine the lens reduction mode its own workers served, so this is not a "
                f"pre-landing server (which states neither key and means "
                f"{la.REDUCTION_MODE_PER_ROW!r}); refusing rather than recording the default "
                "as served identity")
        return la.REDUCTION_MODE_PER_ROW
    if stated not in la.REDUCTION_MODES:
        raise ReductionModeUnattested(
            f"{source}: {LENS_REDUCTION_MODE_KEY}={stated!r} is not one of "
            f"{la.REDUCTION_MODES}")
    return stated


def parse_bool_env(value: Optional[str]) -> bool:
    return value is not None and value.strip().lower() in _PER_ROW_ALL


def parse_per_row_apply(value: Optional[str]) -> Tuple[bool, FrozenSet[str]]:
    """``(everything_per_row, per_row_names)`` from the env value —
    see :data:`PER_ROW_APPLY_ENV`. A list containing ``1``/``all`` is
    the whole switch (``all,psr`` == ``1``)."""
    if value is None:
        return False, frozenset()
    v = value.strip()
    if v.lower() in _PER_ROW_OFF:
        return False, frozenset()
    if v.lower() in _PER_ROW_ALL:
        return True, frozenset()
    names = frozenset(x.strip() for x in v.split(",") if x.strip())
    if any(n.lower() in _PER_ROW_ALL for n in names):
        return True, frozenset()
    return False, names

#: The startup assistant-header check (preflight refuses without a
#: tokenizer; a header mismatch is ``verify_assistant_header``'s own
#: loud refusal). Module-level so tests can pin the wiring.
DEFAULT_HEADER_VERIFIER = ls.verify_assistant_header
#: Its sibling for the ALL-TURNS mask's second marker
#: (``lens_steer.verify_assistant_turn_end``): invoked by preflight ONLY
#: when the registry's loaded ops include an all-turns op
#: (``WorkerObjectRegistry.serves_all_turns``), so it never touches the
#: tokenizer for the response-only registries every other test and
#: deployment builds — whatever metadata they carry.
DEFAULT_TURN_END_VERIFIER = ls.verify_assistant_turn_end
#: Its sibling for the DECLARED THINKING-ON header
#: (``lens_steer.verify_assistant_header_thinking_on``): invoked by preflight
#: ONLY when the registry's loaded ops carry a second header
#: (``WorkerObjectRegistry.serves_thinking_on``) — LOADED ops, never
#: metadata (the fifth review's regression rule).
DEFAULT_THINKING_HEADER_VERIFIER = ls.verify_assistant_header_thinking_on


# --- selector extraction --------------------------------------------------------


def extract_selector(extra_args: Optional[Mapping[str, Any]]
                     ) -> Optional[Dict[str, Any]]:
    """The request's validated H3 selector, or ``None`` for an unhooked
    request (no selector key present at all).

    The transport is FLAT: the proxy sets ``vllm_xargs`` to exactly the
    selector mapping, which vLLM's OpenAI frontend copies into
    ``SamplingParams.extra_args`` (``proxy.UPSTREAM_EXTRA_ARGS_KEY``).
    ``extra_args`` may legitimately carry OTHER subsystems' keys next to
    the selector (vllm-lens's ``output_residual_stream`` on gate
    probes), so only the ``SELECTOR_KEYS`` subset is handed to
    ``parse_selector`` — which still refuses a half-selector (a stray
    ``scale`` or ``expect_hash`` without ``petri_object``), a non-string
    id, or a non-numeric scale."""
    if not extra_args:
        return None
    present = {k: extra_args[k] for k in ls.SELECTOR_KEYS if k in extra_args}
    if not present:
        return None
    return ls.parse_selector(present)


# --- the per-forward request view ------------------------------------------------


@dataclass(frozen=True)
class ResumedPrefill:
    """The RESUMED-PREFILL facts of one selector-carrying request in one
    forward (lens_adapters module docstring, RESUMED PREFILLS): how many
    of its prompt positions the KV cache served (``cached``) and how many
    prompt positions this forward computes (``computed`` — the rows the
    hooks actually see). Recorded per request in the
    :class:`SelectionRecord` so a gate can PROVE a warm request resumed
    (and how far) instead of inferring it from timing."""

    cached: int
    computed: int


@dataclass(frozen=True)
class RequestSlot:
    """One scheduled request's slice of a flattened forward: its rows,
    its validated selector + resolved object (``None`` = unhooked), the
    :class:`~.lens_adapters.HookContext` governing exactly its rows
    (``None`` when this forward skipped the plan machinery — only ever
    the case when no active op needs one), and — for a selector-carrying
    request whose prompt resumed from the KV cache in THIS forward — the
    :class:`ResumedPrefill` facts (``None`` for a complete prefill, a
    decode step, or an unhooked request)."""

    req_id: str
    rows: slice
    selector: Optional[Dict[str, Any]]
    obj: Optional[ls.LoadedObject]
    ctx: Optional[la.HookContext]
    resumed: Optional[ResumedPrefill] = None
    #: a CONTAINED per-request refusal decided at classification (a
    #: ``conditional``-policy cell whose cached region is not header-free —
    #: lens_adapters.cached_region_header_conflict): the classifier records
    #: it (``_record_request_failure``), the request serves unsteered from
    #: this forward on and its response is refused at the proxy; the engine
    #: keeps serving. ``None`` for every served slot.
    refusal: Optional[la.ResumedPrefillRefusal] = None
    #: DIAGNOSTICS ONLY — the :class:`SelectionRecord`'s per-request evidence,
    #: computed at CLASSIFICATION on both paths so a request's record reads
    #: the same alone (the row-local path, ``ctx`` None) and co-batched with
    #: a plan-needing request. ``facts``: the request's declared render facts
    #: as its validated selector carries them (:func:`selector_facts`,
    #: ``lens_steer.SELECTOR_FACT_KEYS``), never read off a context.
    #: ``prefill_masked_rows``: for a PREFILL forward, how many of its rows
    #: the bound object's ops select under that render
    #: (``lens_adapters.prefill_mask_rows``: P on an ``all`` cell, the ON
    #: header's width on a declared twin) — from this path's own context on
    #: the plan path (:func:`prefill_masked_rows_of`), from
    #: :func:`row_local_prefill_masked_rows` on the row-local path; ``None``
    #: for a decode step, a shape that is not a complete prefill, an
    #: uncountable geometry, or an unhooked request. Nothing on the apply
    #: path reads either field.
    facts: Optional[Dict[str, bool]] = None
    prefill_masked_rows: Optional[int] = None


#: :class:`ResumeDecision` kinds
RESUME_NOT_A_RESUME = "not_a_resume"        # a complete prefill or a decode step
RESUME_SERVE = "serve"                      # served as a RESUMED PREFILL
RESUME_REFUSE_CONTAINED = "refuse_contained"  # conditional cell, unsafe cached region
RESUME_REFUSE_FATAL = "refuse_fatal"        # never-policy cell: the salt discipline broke


@dataclass(frozen=True)
class ResumeDecision:
    """What :func:`resume_decision` decided for one selector-carrying
    request's segment: its ``kind`` (above), the cell's policy, the
    :class:`ResumedPrefill` facts when the segment is a resume shape, the
    cached region's ids when a ``conditional`` cell is SERVED (they go into
    the HookContext so the response op re-verifies them), and the contained
    refusal when it is not."""

    kind: str
    policy: str
    facts: Optional[ResumedPrefill] = None
    cached_ids: Optional[Tuple[int, ...]] = None
    refusal: Optional[la.ResumedPrefillRefusal] = None
    reason: Optional[str] = None


def resume_decision(obj: ls.LoadedObject, *, computed: int, q: int, seq_len: int,
                    prompt_len: int, tokens: Any, req_id: Any) -> ResumeDecision:
    """THE PER-REQUEST RESUME DECISION (lens_adapters module docstring,
    RESUMED PREFILLS), for one selector-carrying request's segment of a
    caching-ON forward. Not a resume shape (``computed == 0``: a complete
    prefill; ``q == 1, computed >= prompt_len``: a decode step) →
    :data:`RESUME_NOT_A_RESUME`. Otherwise by the cell's
    ``resumed_prefill_policy``: ``always`` → :data:`RESUME_SERVE`;
    ``never`` → :data:`RESUME_REFUSE_FATAL` (the caller raises the
    engine-fatal salt-discipline refusal — behind the proxy's request-unique
    salt this is an invariant breach); ``conditional`` → the cached region
    ``tokens[:computed]`` is checked for the cell's assistant header
    (``lens_adapters.resumed_header_conflict``): header-free →
    :data:`RESUME_SERVE` with the cached ids for the context, else
    :data:`RESUME_REFUSE_CONTAINED` with the typed refusal — an EXPECTED
    event under a stable salt, failed per request, never fatal. The facts
    (``cached``, ``computed`` prompt positions) ride on every resume-shape
    decision so the SelectionRecord attests even a refused resume's
    shape.

    ``conditional`` covers BOTH header-located modes, each under its own
    clause set (``lens_adapters.resumed_header_conflict`` ->
    ``conflict_clauses_for``) — the SAME two clauses for both today: a
    full earlier header in the region and a boundary inside a header both
    refuse, for ``response`` and ``response_all_turns`` alike (the
    all-turns narrowing is future work, see ``lens_adapters``).
    The served decision carries the cached ids either way — the response
    op re-verifies its clauses from them, the all-turns op re-verifies
    them AND locates the spans that opened before the boundary."""
    if computed == 0 or (q == 1 and computed >= prompt_len):
        return ResumeDecision(kind=RESUME_NOT_A_RESUME, policy=la.resumed_prefill_policy(obj.ops))
    policy = la.resumed_prefill_policy(obj.ops)
    facts = ResumedPrefill(cached=int(computed),
                           computed=max(0, min(int(seq_len), int(prompt_len)) - int(computed)))
    if policy == la.RESUMED_ALWAYS:
        return ResumeDecision(kind=RESUME_SERVE, policy=policy, facts=facts)
    if policy == la.RESUMED_NEVER:
        return ResumeDecision(kind=RESUME_REFUSE_FATAL, policy=policy, facts=facts)
    if len(tokens) < computed:
        raise LensWorkerError(
            f"request {req_id!r}: the runner knows {len(tokens)} tokens but "
            f"{computed} prompt positions are cached — the token source is "
            "behind the cache state; refusing to decide a conditional resume "
            "on ids that do not exist")
    cached_ids = tuple(int(t) for t in tokens[:computed])
    conflict = la.resumed_header_conflict(obj.ops, cached_ids)
    if conflict is None:
        return ResumeDecision(kind=RESUME_SERVE, policy=policy, facts=facts, cached_ids=cached_ids)
    refusal = la.ResumedPrefillRefusal(
        f"request {req_id!r} (cell {obj.cell_id!r}, resumed_prefill_policy is "
        f"conditional): RESUMED prefill (q={q}, computed={computed}, "
        f"prompt_len={prompt_len}) refused — {conflict}. The request is failed "
        "and served unsteered; its response must be refused at the proxy "
        "(sisteer_take_request_failure). An expected event under a stable "
        "per-cell salt, contained per request, never engine-fatal.")
    return ResumeDecision(kind=RESUME_REFUSE_CONTAINED, policy=policy, facts=facts,
                          refusal=refusal, reason=conflict)


def _fatal_resume_refusal(req_id: Any, *, q: int, computed: int, prompt_len: int) -> LensWorkerError:
    return LensWorkerError(
        f"request {req_id!r} carries a steering selector but arrived "
        f"as a RESUMED shape (q={q}, computed={computed}, "
        f"prompt_len={prompt_len}) — a prefix-cache hit on a steered "
        "request whose ops cannot resume (lens_adapters."
        "resumed_prefill_ok is False: resumed_prefill_policy is never). Its "
        "cached rows skipped the forward, so the hooks never fired on them: "
        "the per-request cache-salt discipline that makes this impossible "
        "has broken; refusing to serve a partially-steered prompt")


@dataclass(frozen=True)
class ForwardSlots:
    """One classified forward: every request's slot, plus the derived
    plan and its tracked-requests projection (both ``None`` on the
    row-local-only path). ``tracked_plan`` is what the ledger consumes —
    the full plan filtered to requests whose ACTIVE object needs the
    accounting, so an untracked request's prefill in the same forward
    never draws a credit the run was never armed with."""

    slots: Tuple[RequestSlot, ...]
    plan: Optional[lb.BatchPlan]
    tracked_plan: Optional[lb.BatchPlan]


class RequestTokens(_ABCSequence):
    """A request's KNOWN token sequence (prompt + generated so far) as a
    read-only VIEW over the runner's two lists — O(1) ``len``, ranged
    ``__getitem__`` — in place of ``list(prompt) + list(output)``, which
    copied ~1,250 ids twice per (request, layer) to slice out ONE decode
    token (7% of the steered step in the 2026-09-04 profile). The view
    is built per classification and consumed within the same forward;
    the underlying lists are the runner's live state, read, never
    copied or mutated."""

    __slots__ = ("_prompt", "_output")

    def __init__(self, prompt: Sequence[int], output: Sequence[int]):
        self._prompt = prompt
        self._output = output

    def __len__(self) -> int:
        return len(self._prompt) + len(self._output)

    def __getitem__(self, item):
        n_p = len(self._prompt)
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            if step != 1:
                return [self[i] for i in range(start, stop, step)]
            if stop <= n_p:
                return list(self._prompt[start:stop])
            if start >= n_p:
                return list(self._output[start - n_p:stop - n_p])
            return list(self._prompt[start:]) + list(self._output[:stop - n_p])
        i = item.__index__()
        n = len(self)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(i)
        return self._prompt[i] if i < n_p else self._output[i - n_p]

    def __iter__(self):
        yield from self._prompt
        yield from self._output


#: a group's rows: one contiguous slice when the member slots are
#: adjacent in the forward, else the sorted flattened row indices.
GroupRows = Union[slice, Tuple[int, ...]]


@dataclass
class SlotGroup:
    """One VECTORIZED apply unit of a forward: the SINGLE-ROW selector-
    carrying slots that share an object, a selector scale, and a row
    context class — every decode slot of one object under the same
    declared facts (``ctx`` a :class:`~.lens_adapters.DecodeRows`), or
    every one-row slot of one row-local object (``ctx`` None: no op
    consults a position there). A multi-row slot is never a member.
    Built once per forward (:func:`build_slot_groups`); the device index
    tensor for scattered rows is built once per forward too, at the first
    layer that needs it (:meth:`index_on`)."""

    slots: Tuple[RequestSlot, ...]
    cell_id: str
    expect_hash: str
    scale: Optional[float]
    ctx: Optional[la.DecodeRows]
    rows: GroupRows
    _index: Dict[Any, torch.Tensor] = field(default_factory=dict, repr=False)

    @property
    def contiguous(self) -> bool:
        return isinstance(self.rows, slice)

    def index_on(self, device: Any) -> torch.Tensor:
        """The LongTensor of this group's rows on ``device`` — ONE
        host->device copy per (group, device) per forward, never per
        layer (pinned + non_blocking on CUDA; ``torch.tensor(...,
        device=cuda)`` is the blocking copy the profile flagged)."""
        key = str(device)
        idx = self._index.get(key)
        if idx is None:
            assert not isinstance(self.rows, slice)
            cpu = torch.tensor(self.rows, dtype=torch.long)
            if getattr(device, "type", str(device)) == "cuda":
                idx = cpu.pin_memory().to(device, non_blocking=True)
            else:
                idx = cpu.to(device)
            self._index[key] = idx
        return idx


@dataclass(frozen=True)
class ClassifiedForward:
    """One forward's classification, cached across its hooks (module
    docstring): the slots, the row count the metadata covers, the
    vectorized groups, and the selector-carrying slots that stay per
    request (prefills — per-request masks — and per-row-forced
    objects)."""

    fwd: ForwardSlots
    nat: int
    groups: Tuple[SlotGroup, ...]
    per_slot: Tuple[RequestSlot, ...]


def _rows_of(slots: Sequence[RequestSlot]) -> GroupRows:
    rows: List[int] = []
    for s in slots:
        rows.extend(range(s.rows.start, s.rows.stop))
    rows.sort()
    if rows and rows[-1] - rows[0] + 1 == len(rows):
        return slice(rows[0], rows[-1] + 1)
    return tuple(rows)


def _object_forced_per_row(obj: ls.LoadedObject,
                           per_row_names: FrozenSet[str]) -> bool:
    """Whether ``obj`` serves per row under a per-kind fallback list: its
    object kind, or ANY of its ops' op names, is listed."""
    if not per_row_names:
        return False
    if obj.kind in per_row_names:
        return True
    return any(op.op in per_row_names for op in obj.ops)


def build_slot_groups(fwd: ForwardSlots, *, per_row_all: bool = False,
                      per_row_names: FrozenSet[str] = frozenset()
                      ) -> Tuple[Tuple[SlotGroup, ...], Tuple[RequestSlot, ...]]:
    """Partition a forward's SELECTOR-CARRYING slots into vectorized
    groups and per-slot leftovers (module docstring, THE VECTORIZED
    APPLY). Unhooked slots (no selector) belong to neither and are never
    touched.

    A slot is grouped by ``(cell, expect_hash, selector scale, row
    context class)`` when it carries EXACTLY ONE ROW and is a DECODE
    slot (``ctx.phase == decode`` — the class is its declared facts as
    a :class:`~.lens_adapters.DecodeRows`) or a ROW-LOCAL slot (``ctx``
    None — the forward skipped the plan; every active op is position-
    independent). Every MULTI-ROW slot stays per slot, whatever its
    ops: a prefill's masks are located per request, and a group is a
    stack of single rows so that each member's reductions are issued
    exactly as the per-request path issues them (``lens_adapters.
    _row_reduction``; the 2026-09-04 gate measured stacked prefill row
    counts breaking bit-identity on every family). A one-row prefill
    slot (``ctx.phase == prefill``) also stays per slot. Objects named
    by ``per_row_names`` (kind or op name) stay per slot; ``per_row_all``
    puts EVERY selector slot in ``per_slot`` (the reference path). The
    order of groups follows the first appearance of each key in batch
    order; the order of slots inside a group is batch order."""
    if per_row_all:
        return (), tuple(s for s in fwd.slots if s.selector is not None)
    keyed: Dict[Tuple[Any, ...], List[RequestSlot]] = {}
    per_slot: List[RequestSlot] = []
    for slot in fwd.slots:
        if slot.selector is None:
            continue
        if slot.obj is None or _object_forced_per_row(slot.obj, per_row_names):
            per_slot.append(slot)
            continue
        if slot.rows.stop - slot.rows.start != 1:
            per_slot.append(slot)  # multi-row (a prefill): never stacked
            continue
        ctx = slot.ctx
        if ctx is None:
            gctx: Optional[la.DecodeRows] = None
        elif ctx.phase == la.PHASE_DECODE:
            gctx = la.DecodeRows(response_prefill=ctx.response_prefill,
                                 agent_loop=ctx.agent_loop)
        else:
            per_slot.append(slot)  # prefill: the mask is this request's
            continue
        key = (slot.selector["petri_object"], slot.selector["expect_hash"],
               slot.selector.get("scale"), gctx)
        keyed.setdefault(key, []).append(slot)
    groups = tuple(
        SlotGroup(slots=tuple(slots), cell_id=key[0], expect_hash=key[1],
                  scale=key[2], ctx=key[3], rows=_rows_of(slots))
        for key, slots in keyed.items())
    return groups, tuple(per_slot)


def ops_vectorize_in_decode(ops: Sequence[la.HookOp]) -> bool:
    """Whether an object with these ops can MODIFY a vectorized decode group
    — i.e. whether it is ever applied on the GROUPED path at all. This is the
    composition of the two rules that decide it, stated once so the GPU gate
    (scripts/lens_vectorize_gate.py) classifies its inventory from the same
    source the worker serves by: :func:`build_slot_groups` routes multi-row
    (prefill) slots per slot and single-row DECODE slots into groups, and
    ``lens_adapters.apply_hook_ops`` skips an op whose ``phase`` excludes
    the forward's phase. So an object whose EVERY op is ``phase="prefill"`` (alphasteer,
    loreft) is applied only on the per-slot path — its decode groups are
    unmodified on both sides of the shadow compare (``equal=None``, never a
    counted comparison) — while any op with phase ``both`` / ``decode``
    modifies its decode groups and is shadow-compared there."""
    return any(op.phase != la.PHASE_PREFILL for op in ops)


def phase_vectorizes_in_decode(phase: str) -> bool:
    """:func:`ops_vectorize_in_decode` for ONE op phase (the gate's case ids
    carry the phase as a field); refuses a phase outside the vocabulary."""
    if phase not in la.PHASES:
        raise LensWorkerError(f"unknown op phase {phase!r} (vocabulary: {list(la.PHASES)})")
    return phase != la.PHASE_PREFILL


def _object_needs_plan(obj: ls.LoadedObject) -> bool:
    """Whether serving ``obj`` needs the plan lattice + ledger: its ops
    are position/phase-dependent AND it is actually applied.
    ``effective_scale == 0.0`` (the ``none__lens0`` leg) is excluded on
    purpose — ``apply_hook_ops`` returns the hidden states untouched
    before consulting any context, so mis-segmentation cannot change
    the math and the accounting would only add spurious refusals."""
    return obj.effective_scale != 0.0 and lb.ops_need_plan(obj.ops)


def site_layers(registry: ls.WorkerObjectRegistry
                ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """(block_layers, oproj_layers): which layers need a decoder-layer
    (block-output) hook and which additionally need a hook on their
    ``self_attn.o_proj`` module — driven off the OPS, never a kind list
    (the ops_need_plan discipline), so a stacked object that mixes the
    two sites installs both.

    Public on purpose: the launcher's independent install expectation
    (``vllm_server.expected_install_state``) derives its per-site layer
    sets with THIS function, so the readiness comparison is site-aware
    by construction — the 2026-08-28 wave died at readiness because the
    launcher lumped every op layer into the block-site ``layers`` field
    while the worker correctly reported its o_proj hooks under
    ``oproj_layers``."""
    block: set = set()
    oproj: set = set()
    for obj in _registry_objects(registry):
        for op in obj.ops:
            if la.op_site(op) == la.HOOK_SITE_OPROJ:
                oproj.add(op.layer)
            else:
                block.add(op.layer)
    return tuple(sorted(block)), tuple(sorted(oproj))


#: plan-site ordering ranks: within one decoder layer's forward the
#: o_proj module finishes BEFORE the layer's own forward hook fires, so
#: the o_proj hook is the earlier site at the same layer index.
_SITE_RANK = {la.HOOK_SITE_OPROJ: 0, la.HOOK_SITE_BLOCK: 1}


def _resolve_oproj(layer_module: Any, layer_idx: int) -> Any:
    """The attention output-projection module of one decoder layer,
    loudly — ``self_attn.o_proj`` on the pinned vLLM Llama/Gemma families
    (and the HF classes the authoritative hooks attach to), or
    ``mixer.o_proj`` on a NemotronH ATTENTION block (the one seam,
    ``sisteer.models.attention_oproj``, that the fit side's
    ``oproj_layers`` capability probe reads too, so fit and serve agree
    on which layers carry the site). A layer with no attention o_proj —
    another architecture, or a NemotronH Mamba2/MLP block that an
    o_proj-site op was (mis)targeted at — refuses at INSTALL, never a
    hook that silently serves nothing. Mamba's ``mixer.out_proj`` is a
    different attribute name and is never matched."""
    from sisteer.models import attention_oproj

    oproj = attention_oproj(layer_module)
    if oproj is None or not hasattr(oproj, "register_forward_hook"):
        raise LensWorkerError(
            f"decoder layer {layer_idx} has no hookable attention o_proj "
            "module (neither self_attn.o_proj nor a NemotronH attention "
            "mixer.o_proj) — the o_proj-site ops (the lora kinds) cannot "
            "be installed at this layer of this architecture; refusing at "
            "startup rather than serving a silently-unhooked arm")
    return oproj


def resumed_prefill_ok_cells(registry: ls.WorkerObjectRegistry
                             ) -> Dict[str, bool]:
    """cell id -> ``lens_adapters.resumed_prefill_ok(cell ops)`` for every
    served cell — which cells this worker will serve a prefix-cache RESUME
    for (lens_adapters module docstring, RESUMED PREFILLS). Driven off the
    OPS, never a kind list; the scale-0 twin of a cell reads the same as
    its arm (same ops). Reported in the install payload
    (:data:`RESUMED_PREFILL_OK_CELLS_KEY`)."""
    return {obj.cell_id: la.resumed_prefill_ok(obj.ops)
            for obj in _registry_objects(registry)}


def resumed_prefill_policy_cells(registry: ls.WorkerObjectRegistry
                                 ) -> Dict[str, str]:
    """cell id -> ``lens_adapters.resumed_prefill_policy(cell ops)`` for every
    served cell — the THREE-VALUED policy (``never`` / ``always`` /
    ``conditional``) this worker enforces per request, reported in the install
    payload (:data:`RESUMED_PREFILL_POLICY_CELLS_KEY`)."""
    return {obj.cell_id: la.resumed_prefill_policy(obj.ops)
            for obj in _registry_objects(registry)}


def resumed_prefill_policy_cells_attested(attestation: Any
                                          ) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """The per-cell resumed-prefill POLICY as the workers attested it
    (:data:`RESUMED_PREFILL_POLICY_CELLS_KEY` on every top-level install
    ``worker_reports`` entry, equal across workers, every value one of
    ``lens_adapters.RESUMED_POLICIES``) — ``(mapping, None)`` or ``(None,
    reason)``, never a guess. THE PROXY HALF's read: ``always`` → a stable
    per-(cell, intervention) salt; ``conditional`` → a stable salt for
    single-turn requests (one assistant header, at the end of the render)
    and request-unique otherwise; ``never``, missing, or an unreadable
    attestation → request-unique."""
    if not isinstance(attestation, Mapping):
        return None, "attestation is not a mapping"
    reports = attestation.get("worker_reports")
    if not isinstance(reports, list) or not reports:
        return None, "attestation carries no top-level install worker_reports"
    seen: List[Dict[str, str]] = []
    for i, rep_ in enumerate(reports):
        if not isinstance(rep_, Mapping) or RESUMED_PREFILL_POLICY_CELLS_KEY not in rep_:
            return None, (f"worker report {i} carries no {RESUMED_PREFILL_POLICY_CELLS_KEY} — "
                          "a worker from a tree that predates the resumed-prefill policy")
        m = rep_[RESUMED_PREFILL_POLICY_CELLS_KEY]
        if not isinstance(m, Mapping) or not all(v in la.RESUMED_POLICIES for v in m.values()):
            return None, (f"worker report {i}: {RESUMED_PREFILL_POLICY_CELLS_KEY} is not a cell -> "
                          f"policy mapping over {la.RESUMED_POLICIES}")
        seen.append({str(k): str(v) for k, v in m.items()})
    if any(m != seen[0] for m in seen[1:]):
        return None, f"workers disagree on {RESUMED_PREFILL_POLICY_CELLS_KEY}: {seen}"
    return dict(seen[0]), None


def serve_declarations_cells(registry: ls.WorkerObjectRegistry
                             ) -> Dict[str, Dict[str, Any]]:
    """cell id -> the cell's serve declaration for every DECLARATION-CARRYING
    served cell (``WorkerObjectRegistry.serve_declarations_by_cell``; the
    loaded objects' ``serve_declarations``, bound to their ops at load) —
    reported in the install payload (:data:`LENS_SERVE_DECLARATIONS_KEY`).
    ``{}`` for a registry with no declared cell: a STATED empty map, so the
    proxy can tell "attested, nothing declared" from "nothing attested"."""
    return {obj.cell_id: dict(obj.serve_declarations)
            for obj in _registry_objects(registry)
            if getattr(obj, "serve_declarations", None)}


def serve_declarations_attested(attestation: Any
                                ) -> Tuple[Optional[Dict[str, Dict[str, Any]]], Optional[str]]:
    """The per-cell SERVE DECLARATIONS as the workers attested them
    (:data:`LENS_SERVE_DECLARATIONS_KEY` on every top-level install
    ``worker_reports`` entry, equal across workers, every key a non-empty
    string and every value exactly one of ``serve_declaration``'s two closed
    shapes — ``serve_declaration.declarations_of`` is the validator) —
    ``(mapping, None)`` or ``(None, reason)``, never a guess. An attested
    EMPTY map is ``({}, None)``: the standup served no declared cell, which
    the proxy reads as "every thinking-ON declaration refuses
    cell-not-declared", distinct from ``None`` ("nothing attested" -> every
    thinking-ON request refuses unattested — fail closed)."""
    from sisteer.capeval.serving import serve_declaration as sd
    if not isinstance(attestation, Mapping):
        return None, "attestation is not a mapping"
    reports = attestation.get("worker_reports")
    if not isinstance(reports, list) or not reports:
        return None, "attestation carries no top-level install worker_reports"
    seen: List[Dict[str, Dict[str, Any]]] = []
    for i, rep_ in enumerate(reports):
        if not isinstance(rep_, Mapping) or LENS_SERVE_DECLARATIONS_KEY not in rep_:
            return None, (f"worker report {i} carries no {LENS_SERVE_DECLARATIONS_KEY} — "
                          "a worker from a tree that predates the serve-declaration attestation")
        m = rep_[LENS_SERVE_DECLARATIONS_KEY]
        if not isinstance(m, Mapping):
            return None, (f"worker report {i}: {LENS_SERVE_DECLARATIONS_KEY} is not a cell -> "
                          "declaration mapping")
        clean: Dict[str, Dict[str, Any]] = {}
        for cell, value in m.items():
            if not isinstance(cell, str) or not cell:
                return None, (f"worker report {i}: {LENS_SERVE_DECLARATIONS_KEY} has a bad cell "
                              f"id {cell!r}")
            if not isinstance(value, Mapping):
                return None, (f"worker report {i}: {LENS_SERVE_DECLARATIONS_KEY}[{cell!r}] is not "
                              f"a mapping: {value!r}")
            try:
                decl = sd.declarations_of(value)
            except sd.ServeDeclarationError as e:
                return None, (f"worker report {i}: {LENS_SERVE_DECLARATIONS_KEY}[{cell!r}] is "
                              f"malformed: {e}")
            if not decl or set(decl) != set(value):
                return None, (f"worker report {i}: {LENS_SERVE_DECLARATIONS_KEY}[{cell!r}] = "
                              f"{dict(value)!r} is not exactly one closed declaration shape")
            clean[cell] = decl
        seen.append(clean)
    if any(m != seen[0] for m in seen[1:]):
        return None, f"workers disagree on {LENS_SERVE_DECLARATIONS_KEY}: {seen}"
    return dict(seen[0]), None

def thinking_header_attested(attestation: Any
                             ) -> Tuple[Optional[Optional[Tuple[int, ...]]], Optional[str]]:
    """The DECLARED thinking-ON header as the workers attested it
    (:data:`LENS_THINKING_HEADER_KEY` on every top-level install
    ``worker_reports`` entry — the registry's second header after each
    worker's own cross-check at preflight, ``None`` on a worker whose loaded
    ops carry none): ``((ids or None), None)`` when every worker carries the
    key and they agree, else ``(None, reason)``. The standup holds the
    attested ids to the header it derived; a worker predating the landing
    (no key) is a reason, never read as "no header"."""
    if not isinstance(attestation, Mapping):
        return None, "attestation is not a mapping"
    reports = attestation.get("worker_reports")
    if not isinstance(reports, list) or not reports:
        return None, "attestation carries no top-level install worker_reports"
    seen: List[Optional[Tuple[int, ...]]] = []
    for i, rep_ in enumerate(reports):
        if not isinstance(rep_, Mapping) or LENS_THINKING_HEADER_KEY not in rep_:
            return None, (f"worker report {i} carries no {LENS_THINKING_HEADER_KEY} — a worker "
                          "from a tree that predates the thinking-ON header attestation")
        raw = rep_[LENS_THINKING_HEADER_KEY]
        if raw is None:
            seen.append(None)
            continue
        if not isinstance(raw, list) or not raw or any(
                not isinstance(t, int) or isinstance(t, bool) or t < 0 for t in raw):
            return None, (f"worker report {i}: {LENS_THINKING_HEADER_KEY} must be null or a "
                          f"non-empty list of non-negative ints, got {raw!r}")
        seen.append(tuple(int(t) for t in raw))
    if any(v != seen[0] for v in seen[1:]):
        return None, f"workers disagree on {LENS_THINKING_HEADER_KEY}: {seen}"
    return seen[0], None

def _serves_thinking_on(registry: ls.WorkerObjectRegistry) -> bool:
    """Whether the registry's LOADED ops carry the second (thinking-ON) header —
    ``WorkerObjectRegistry.serves_thinking_on`` where the registry has it, False on
    an older registry class (no such method): the install report then says
    ``assistant_header_ids_thinking_on: null``."""
    serves_on = getattr(registry, "serves_thinking_on", None)
    return bool(serves_on()) if serves_on is not None else False



def resumed_prefill_ok_cells_attested(attestation: Any
                                      ) -> Tuple[Optional[Dict[str, bool]], Optional[str]]:
    """The per-cell resumed-prefill predicate as the WORKERS attested it in a
    steering attestation (``vllm_server.STEERING_ATTESTATION_FILE``): read
    from every top-level install ``worker_reports`` entry's
    :data:`RESUMED_PREFILL_OK_CELLS_KEY`, required present on each and equal
    across workers. ``(mapping, None)`` when attested; ``(None, reason)``
    when any worker lacks the key (a tree that predates the landing) or the
    workers disagree — never a guess. This is the read the PROXY HALF
 makes to decide a stable vs request-unique cache salt per
    cell: a cell missing here, or attested False, gets the request-unique
    salt it always had."""
    if not isinstance(attestation, Mapping):
        return None, "attestation is not a mapping"
    reports = attestation.get("worker_reports")
    if not isinstance(reports, list) or not reports:
        return None, "attestation carries no top-level install worker_reports"
    seen: List[Dict[str, bool]] = []
    for i, rep_ in enumerate(reports):
        if not isinstance(rep_, Mapping) or RESUMED_PREFILL_OK_CELLS_KEY not in rep_:
            return None, (f"worker report {i} carries no {RESUMED_PREFILL_OK_CELLS_KEY} — "
                          "a worker from a tree that predates the resumed-prefill landing")
        m = rep_[RESUMED_PREFILL_OK_CELLS_KEY]
        if not isinstance(m, Mapping) or not all(isinstance(v, bool) for v in m.values()):
            return None, f"worker report {i}: {RESUMED_PREFILL_OK_CELLS_KEY} is not a cell -> bool mapping"
        seen.append({str(k): bool(v) for k, v in m.items()})
    if any(m != seen[0] for m in seen[1:]):
        return None, f"workers disagree on {RESUMED_PREFILL_OK_CELLS_KEY}: {seen}"
    return dict(seen[0]), None


def cells_needing_ledger(registry: ls.WorkerObjectRegistry
                         ) -> Tuple[str, ...]:
    """The registry cells whose requests must be armed in the
    :class:`~.lens_batch.PrefillLedger` (usable on BOTH sides: the
    submission side arms prompt lengths for exactly these cells'
    requests; the worker consumes tracked prefills for the same set)."""
    return tuple(obj.cell_id for obj in _registry_objects(registry)
                 if _object_needs_plan(obj))


def _registry_objects(registry: ls.WorkerObjectRegistry
                      ) -> Tuple[ls.LoadedObject, ...]:
    """Every loaded object, in cell-id order.

    Startup enumeration — which layers to hook, which cells need the
    ledger — needs the objects themselves, which is what
    ``WorkerObjectRegistry.objects()`` is for. Request-time access goes
    through ``select`` instead, so the selector cross-checks are never
    bypassed."""
    return registry.objects()


def _int_list(name: str, val: Any) -> List[int]:
    """A 1-D int list from a tensor/list, loudly (thin local version for
    the row-bounds path — the plan path uses ``lens_batch``'s own
    validation and never this)."""
    if torch.is_tensor(val):
        if val.dim() != 1:
            raise LensWorkerError(f"{name} must be 1-D, got shape "
                                  f"{tuple(val.shape)}")
        return [int(x) for x in val.tolist()]
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    raise LensWorkerError(f"{name} must be a 1-D tensor or list, got "
                          f"{type(val).__name__}")


def _row_bounds(query_start_loc: Any, num_actual_tokens: int
                ) -> List[Tuple[int, int]]:
    """Per-request (start, end) rows from ``query_start_loc`` alone —
    the ROW-LOCAL path's segmentation. Deliberately NOT the plan
    lattice: row-local ops are chunking-safe (every row gets the same
    edit wherever the prompt boundary sits), so a chunked-prefill shape
    is tolerated here where ``derive_batch_plan`` would refuse it.
    Structural drift (non-monotone, padded, truncated) still refuses."""
    qsl = _int_list("query_start_loc", query_start_loc)
    if len(qsl) < 2 or qsl[0] != 0:
        raise LensWorkerError(
            f"query_start_loc must start at 0 with at least one segment, "
            f"got {qsl[:3]}... ({len(qsl)} entries)")
    if qsl[-1] != int(num_actual_tokens):
        raise LensWorkerError(
            f"query_start_loc[-1] == {qsl[-1]} != num_actual_tokens == "
            f"{int(num_actual_tokens)} — padded or truncated metadata; "
            "refusing to index rows")
    bounds = []
    for i in range(len(qsl) - 1):
        if qsl[i + 1] <= qsl[i]:
            raise LensWorkerError(
                f"segment {i}: non-increasing query_start_loc "
                f"({qsl[i]} -> {qsl[i + 1]})")
        bounds.append((qsl[i], qsl[i + 1]))
    return bounds


def _tensors_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """``torch.equal`` with a SYNC-FREE fast path: the same tensor object,
    or two views of the same device memory (same data pointer, shape,
    strides, dtype), are equal without reading the device. vLLM builds
    every KV-cache group's metadata from ONE shallow-copied
    ``CommonAttentionMetadata`` (its builders alias
    ``common_attn_metadata.query_start_loc`` / ``.seq_lens``; the GDN
    builder sets ``non_spec_query_start_loc = query_start_loc`` when no
    speculative decode is in flight — all read at the pinned 0.25.1
    tag), so on every real hybrid or multi-group forward the cross-checks
    below resolve here. Tensors that do NOT alias fall through to the
    real comparison — the check is never skipped, only proven cheaply."""
    if a is b:
        return True
    if (torch.is_tensor(a) and torch.is_tensor(b) and a.device == b.device
            and a.dtype == b.dtype and a.shape == b.shape
            and a.stride() == b.stride() and a.data_ptr() == b.data_ptr()):
        return True
    return torch.equal(a, b)


def _normalize_positions(pos: Any) -> Any:
    """The decoder layer's positions input, reduced to the single 1-D
    absolute-position vector the plan lattice consumes.

    M-RoPE models (Qwen3.5 / the Qwen-VL family; GLM-4.6V-Flash, whose
    text_config carries ``rope_parameters.mrope_section`` — the exact
    signal ``uses_mrope`` keys on at the tag) hand every layer a
    ``(3, num_tokens)`` STACK — the temporal/height/width axes of
    multimodal rotary embedding. GPU-VERIFY, read from the pinned
    0.25.1 source: the runner passes ``mrope_positions[:, :num_tokens]``
    whenever ``model_config.uses_mrope`` (gpu_model_runner.py), and
    qwen3_5.py's own forward docstring pins the ``(3, seq_len)`` shape.
    The 3 is NOT the KV-cache-group count — nothing in the runner
    stacks positions per group. For TEXT-ONLY batches all three rows
    are identical by construction — text spans are built as
    ``np.broadcast_to(np.arange(text_len), (3, text_len))``
    (qwen3_vl.py, _get_mrope_input_positions) and decode steps write
    one next-position into every row (get_next_input_positions_tensor)
    — so the stack carries exactly one row segmentation and reducing to
    row 0 is lossless. Rows that DISAGREE mean image/video tokens are
    in the batch, where the absolute-textual-index accounting (and with
    it every position-dependent op) genuinely does not hold: refuse,
    never average or take a row on faith. 1-D input (every non-mrope
    model) passes through untouched, object-identical."""
    if not torch.is_tensor(pos) or pos.dim() != 2:
        return pos
    if pos.shape[0] < 1:
        raise LensWorkerError(
            f"positions arrived as a {tuple(pos.shape)} tensor with no "
            "rows — not a positions layout this wiring reads; refusing "
            "rather than guessing rows")
    first = pos[0]
    for i in range(1, pos.shape[0]):
        if not _tensors_equal(pos[i], first):
            raise LensWorkerError(
                f"positions arrived as a {tuple(pos.shape)} M-RoPE-style "
                "stack whose rows DISAGREE — a multimodal batch places "
                "image/video tokens off the textual index, so the "
                "absolute-position accounting behind every "
                "position-dependent op does not hold there; refusing "
                "rather than steering mis-rowed tokens")
    return first


def build_request_slots(
    *,
    query_start_loc: Any,
    seq_lens: Any,
    num_actual_tokens: Any,
    positions: Any,
    req_ids: Sequence[str],
    raw_selectors: Sequence[Optional[Mapping[str, Any]]],
    token_lookup: Callable[[str], Sequence[int]],
    prompt_len_lookup: Callable[[str], Any],
    registry: ls.WorkerObjectRegistry,
    allow_resumed: bool = False,
) -> ForwardSlots:
    """Classify one flattened forward into per-request slots with
    fully-populated contexts (deliverable #1).

    ``raw_selectors[i]`` is request ``i``'s extracted selector mapping
    (:func:`extract_selector`'s output) or ``None``; resolution
    (``parse``-validated by the extractor, ``select``-resolved here)
    refuses loudly on an unknown id, hash mismatch, or scale
    contradiction. ``token_lookup(req_id)`` returns the request's KNOWN
    token sequence (prompt + generated so far); each segment's rows are
    sliced out of it by absolute position, and a sequence too short for
    its own segment refuses — that is the check that converts the
    decode-token timing assumption (module docstring) into a loud
    failure instead of a silent mis-slice.

    ``prompt_len_lookup(req_id)`` returns the request's SUBMITTED
    prompt-token count; every TRACKED segment (a plan-needing active
    object's) is held to ``lens_batch.check_tracked_segment`` against
    it — the un-skippable per-request prefill invariants that catch
    preemption-recomputes and chunked-prefill remnants with no run
    lifecycle (they run HERE, on the plan path, so no caller can build
    tracked slots without them) — and, under ``allow_resumed``, a
    selector-carrying prefix-cache resume is decided per request by
    :func:`resume_decision` on the cell's ``lens_adapters.
    resumed_prefill_policy``: ``always`` → filed as a RESUMED PREFILL (a
    prefill context at the tail's absolute positions with ``cached > 0``,
    :class:`ResumedPrefill` facts on the slot); ``conditional`` → the same
    when the cached region is header-free (the context also carries the
    cached ids for the response op's own re-verification), else a
    CONTAINED refusal on the slot (``RequestSlot.refusal`` — the classifier
    records it and the request serves unsteered; the engine keeps
    serving); ``never`` → the engine-fatal salt-discipline refusal.

    Two paths, keyed off the ACTIVE objects (never a hard-coded kind
    list): when any needs the plan, the whole forward goes through
    ``lens_batch`` (lattice, positions cross-check, token slicing —
    reused, never re-implemented); when none does, only row bounds are
    derived and every ctx is ``None`` (row-local ops are chunking-safe;
    ``apply_hook_ops`` itself refuses if a context-needing op ever
    reaches a ``None`` ctx, so the asymmetry cannot rot silently).

    On EITHER path every selector-carrying slot also carries the
    :class:`SelectionRecord`'s diagnostics (``RequestSlot.facts`` /
    ``prefill_masked_rows``), derived from the selector and the request's
    own prefill rather than from whether a context was built — so the
    record reads the same for a request served alone and co-batched
    (finding A1; :func:`row_local_prefill_masked_rows`)."""
    req_ids = list(req_ids)
    if len(raw_selectors) != len(req_ids):
        raise LensWorkerError(
            f"{len(raw_selectors)} selectors for {len(req_ids)} requests — "
            "mis-aligned per-request state; refusing")
    objs: List[Optional[ls.LoadedObject]] = []
    selectors: List[Optional[Dict[str, Any]]] = []
    for raw in raw_selectors:
        if raw is None:
            objs.append(None)
            selectors.append(None)
            continue
        sel = ls.parse_selector(raw)  # idempotent re-validation, loud
        objs.append(registry.select(sel["petri_object"], sel["expect_hash"],
                                    sel.get("scale")))
        selectors.append(sel)

    needs_plan = any(o is not None and _object_needs_plan(o) for o in objs)

    if not needs_plan:
        bounds = _row_bounds(query_start_loc, _as_int(num_actual_tokens))
        if len(bounds) != len(req_ids):
            raise LensWorkerError(
                f"attention metadata carries {len(bounds)} segments for "
                f"{len(req_ids)} scheduled requests — metadata/batch "
                "drift; refusing to pair rows with requests")
        # THE SALT-DISCIPLINE GUARD (row-local path), ACTIVE EXACTLY
        # WHEN PREFIX CACHING IS ON (allow_resumed — read off the live
        # engine config at preflight). A SELECTOR-CARRYING request whose
        # ops CANNOT resume (lens_adapters.resumed_prefill_ok False) is
        # served under a request-unique cache salt (proxy._rewrite), so
        # its prefill can never resume from cache: cached rows skip the
        # forward, and a hook that fires only on scheduled rows would
        # silently leave them unsteered — unlike a CHUNKED shape under a
        # caching-OFF config, where every row still passes through some
        # chunk's forward and row-local ops are genuinely chunking-safe
        # (the pre-existing tolerance below, kept for that config). Such
        # a segment under caching that is neither a complete prefill
        # (computed == 0) nor a decode step past its own prompt means the
        # salt discipline broke (or a foreign client bypassed the proxy)
        # — refuse rather than serve a partially-steered prompt. On THIS
        # path that can only be a scale-0 twin of a non-resumable kind
        # (every row-local op is per-position at all positions in both
        # phases, i.e. resumable). A request whose ops CAN resume is
        # served (the proxy shares a per-(cell, intervention) salt for
        # it — the 2026-09-06 landing), and its resume is RECORDED so
        # the SelectionRecord attests it. Selector-less requests are
        # exempt: cache hits are exactly the win the unhooked legs are
        # supposed to get.
        if not allow_resumed:
            slots = _row_local_slots(req_ids, bounds, selectors, objs,
                                     seq_lens=seq_lens, token_lookup=token_lookup)
            return ForwardSlots(slots=slots, plan=None, tracked_plan=None)
        seq = _int_list("seq_lens", seq_lens)
        if len(seq) != len(bounds):
            raise LensWorkerError(
                f"attention metadata carries {len(seq)} seq_lens for "
                f"{len(bounds)} segments — metadata drift; refusing")
        resumed_facts: List[Optional[ResumedPrefill]] = [None] * len(req_ids)
        refusals: List[Optional[la.ResumedPrefillRefusal]] = [None] * len(req_ids)
        for i, (rid, (b_start, b_end)) in enumerate(zip(req_ids, bounds)):
            if selectors[i] is None or objs[i] is None:
                continue
            q = b_end - b_start
            computed = int(seq[i]) - q
            if computed == 0:
                continue  # complete prefill
            n = _as_int(prompt_len_lookup(rid))
            if q == 1 and computed >= n:
                continue  # decode step
            d = resume_decision(objs[i], computed=computed, q=q, seq_len=int(seq[i]),
                                prompt_len=n, tokens=token_lookup(rid), req_id=rid)
            if d.kind == RESUME_REFUSE_FATAL:
                raise _fatal_resume_refusal(rid, q=q, computed=computed, prompt_len=n)
            resumed_facts[i] = d.facts
            refusals[i] = d.refusal    # None when served
        slots = _row_local_slots(req_ids, bounds, selectors, objs,
                                 seq_lens=seq, token_lookup=token_lookup,
                                 resumed=resumed_facts, refusals=refusals)
        return ForwardSlots(slots=slots, plan=None, tracked_plan=None)

    # The plan path is the ONLY consumer of positions (the row-local
    # path above never reads them), so the M-RoPE (3, T) stack is
    # reduced exactly here: after the rows are established equal, row 0
    # IS the batch's absolute-position vector (see _normalize_positions
    # — disagreeing rows refuse).
    positions = _normalize_positions(positions)
    plan = lb.derive_batch_plan(query_start_loc, seq_lens,
                                num_actual_tokens,
                                allow_resumed=allow_resumed)
    if len(plan.segments) != len(req_ids):
        raise LensWorkerError(
            f"attention metadata carries {len(plan.segments)} segments for "
            f"{len(req_ids)} scheduled requests — metadata/batch drift; "
            "refusing to pair rows with requests")

    # THE PER-REQUEST PREFILL INVARIANTS (docstring): every tracked
    # segment against its own request's submitted prompt length, before
    # any row is sliced or steered — and, since 2026-09-06, against its
    # ops' resumed-prefill safety: a RESUMABLE cell's prefix-cache
    # resume is filed as a RESUMED PREFILL (context built below); any
    # other tracked resume refuses (lens_batch.check_tracked_segment).
    tracked = [i for i, o in enumerate(objs)
               if o is not None and _object_needs_plan(o)]
    resumed_prefill: List[int] = []
    resumed_facts: List[Optional[ResumedPrefill]] = [None] * len(req_ids)
    refusals: List[Optional[la.ResumedPrefillRefusal]] = [None] * len(req_ids)
    cached_regions: Dict[int, Tuple[int, ...]] = {}
    for i in tracked:
        seg = plan.segments[i]
        n = _as_int(prompt_len_lookup(req_ids[i]))
        d = resume_decision(objs[i], computed=seg.computed, q=seg.q, seq_len=seg.seq_len,
                            prompt_len=n, tokens=token_lookup(req_ids[i]), req_id=req_ids[i])
        if d.kind == RESUME_REFUSE_CONTAINED:
            # the conditional cell's unsafe hit: failed per request, no
            # context, nothing applied; the lattice's shape checks are moot
            # for rows that are never steered
            refusals[i] = d.refusal
            resumed_facts[i] = d.facts
            continue
        cls = lb.check_tracked_segment(
            seg, prompt_len=n, req_id=req_ids[i],
            resumed_ok=(d.kind == RESUME_SERVE))   # a never-policy resume refuses here, engine-fatal
        if cls == lb.SEG_RESUMED_PREFILL:
            resumed_prefill.append(i)
            resumed_facts[i] = ResumedPrefill(cached=seg.computed, computed=seg.q)
            if d.cached_ids is not None:
                cached_regions[i] = d.cached_ids
    # THE SALT-DISCIPLINE GUARD for the UNTRACKED selector-carrying
    # requests of a plan-path forward (a row-local object, or a scale-0
    # twin, co-scheduled with a tracked request): the row-local path's
    # rule, applied here too, so a steered resume can never slip through
    # merely because another request needed the plan — a resumed shape
    # (or a decode-shaped row inside its own prompt) is served only under
    # the cell's policy (always; conditional when header-free), recorded
    # when it is, contained when a conditional cell's region is unsafe,
    # and refused engine-fatally for a never-policy cell.
    for i, sel in enumerate(selectors):
        if sel is None or i in tracked or objs[i] is None:
            continue
        seg = plan.segments[i]
        if seg.computed == 0:
            continue  # a complete prefill
        n = _as_int(prompt_len_lookup(req_ids[i]))
        if seg.q == 1 and seg.computed >= n:
            continue  # a decode step
        d = resume_decision(objs[i], computed=seg.computed, q=seg.q, seq_len=seg.seq_len,
                            prompt_len=n, tokens=token_lookup(req_ids[i]), req_id=req_ids[i])
        if d.kind == RESUME_REFUSE_FATAL:
            raise _fatal_resume_refusal(req_ids[i], q=seg.q, computed=seg.computed, prompt_len=n)
        resumed_facts[i] = d.facts
        refusals[i] = d.refusal

    # Flatten each request's token ids by absolute position, so
    # lens_batch's own slicing/validation governs the per-segment ids.
    flat_ids: List[int] = []
    for i, seg in enumerate(plan.segments):
        # a RANGED read: ``len`` is O(1) and the slice copies only this
        # segment's ids (one token for a decode step) — never the whole
        # prompt + generation (RequestTokens; a plain list behaves the
        # same, so the CPU fakes are unchanged).
        toks = token_lookup(req_ids[i])
        if len(toks) < seg.seq_len:
            raise LensWorkerError(
                f"request {req_ids[i]!r}: the runner knows {len(toks)} "
                f"tokens but this forward's segment covers absolute "
                f"positions {seg.computed}..{seg.seq_len - 1} — the "
                "token source is behind the forward (the decode-token "
                "timing assumption failed, or the token attribute surface "
                "drifted); refusing to slice ids that do not exist")
        flat_ids.extend(int(t) for t in toks[seg.computed:seg.seq_len])

    contexts = lb.build_segment_contexts(plan, positions, token_ids=flat_ids,
                                         resumed_prefill=resumed_prefill,
                                         cached_ids=cached_regions)
    slots_out: List[RequestSlot] = []
    for i, (rid, sc) in enumerate(zip(req_ids, contexts)):
        sel = selectors[i]
        ctx = (None if refusals[i] is not None
               else _ctx_with_selector_facts(sc.ctx, sel))
        slots_out.append(RequestSlot(
            req_id=rid, rows=sc.rows, selector=sel, obj=objs[i], ctx=ctx,
            resumed=resumed_facts[i], refusal=refusals[i],
            # the record's diagnostics (RequestSlot; finding A1) from THIS
            # path's own context — the values the row-local path files for
            # the same request without one (row_local_prefill_masked_rows)
            facts=None if sel is None else selector_facts(sel),
            prefill_masked_rows=(None if sel is None
                                 else prefill_masked_rows_of(objs[i], ctx))))
    slots = tuple(slots_out)

    tracked_segs = tuple(plan.segments[i] for i in tracked)
    # the ledger's credits are COMPLETE prefills; a resumed prefill of a
    # resumable cell is neither a credit nor a decode step, and the
    # in-proc exact-replay harness (the ledger's only consumer) runs
    # caching-off and never observes one.
    resumed_segs = {plan.segments[i] for i in resumed_prefill}
    tracked_plan = lb.BatchPlan(
        num_tokens=plan.num_tokens,
        segments=tracked_segs,
        full_prefills=tuple(s for s in tracked_segs if s.is_prefill),
        n_decode=sum(1 for s in tracked_segs
                     if not s.is_prefill and s not in resumed_segs),
        n_resumed=len(resumed_segs))
    return ForwardSlots(slots=slots, plan=plan, tracked_plan=tracked_plan)


def _ctx_with_selector_facts(
        ctx: Optional[la.HookContext],
        selector: Optional[Dict[str, Any]]) -> Optional[la.HookContext]:
    """Thread the request's declared per-request FACTS from its validated
    selector into its HookContext — exactly the proxy-verified boundary
    declarations (``lens_steer.SELECTOR_FACT_KEYS``): ``response_prefill``
    (the prefill-seeded-render bit) and ``agent_loop`` (the Petri
    agent-loop-render bit), both consumed by
    ``lens_adapters._response_start``'s declared arm. Each lands in the
    same-named ``HookContext`` field, and ``HookContext`` itself refuses a
    context claiming both (``parse_selector`` already refused the
    selector). A ``None`` ctx (a resumed segment of a selector-less or
    row-local request, or the row-local path) stays ``None``: no position
    op ever consumes it, and ``apply_hook_ops`` refuses if one somehow
    does. A RESUMED PREFILL context (``cached > 0``) carries the facts
    like any prefill context — its ops are ``positions="all"`` by
    construction, so no locator ever reads them there."""
    if ctx is None or selector is None:
        return ctx
    facts = {k: True for k in ls.SELECTOR_FACT_KEYS if selector.get(k)}
    if not facts:
        return ctx
    return dc_replace(ctx, **facts)


def selector_facts(selector: Optional[Mapping[str, Any]]) -> Dict[str, bool]:
    """The request's DECLARED FACTS exactly as its validated selector carries
    them — one bool per ``lens_steer.SELECTOR_FACT_KEYS`` (``response_prefill``
    / ``agent_loop`` / ``thinking_on``), ``False`` where the selector is
    silent or ``None``. What the :class:`SelectionRecord` files as ``facts``:
    the proxy-verified declaration is the authority, and it exists whether
    or not the forward built a positional context to thread it into (finding
    A1: the row-local path builds none, and reading the facts off the
    context there filed ``thinking_on=False`` against a selector carrying the
    fact). The same truthiness rule as :func:`_ctx_with_selector_facts`, so
    a context's facts and the selector's never disagree."""
    return {k: (bool(selector.get(k)) if selector else False)
            for k in ls.SELECTOR_FACT_KEYS}


def prefill_masked_rows_of(obj: Optional[ls.LoadedObject],
                           ctx: Optional[la.RowContext]) -> Optional[int]:
    """``lens_adapters.prefill_mask_rows`` over the bound object's ops on a
    PREFILL context, else ``None`` (a decode row, a group, no object, no
    context, an unlocatable boundary). AN INSTRUMENT — it never raises: a
    diagnostic that can condemn the worker it instruments is worse than
    none, so a count that cannot be computed is ``None``, never a
    refusal."""
    if obj is None or not isinstance(ctx, la.HookContext) \
            or ctx.phase != la.PHASE_PREFILL:
        return None
    try:
        return la.prefill_mask_rows(obj.ops, ctx)
    except Exception:  # noqa: BLE001 — an instrument, never the apply path
        return None


def row_local_prefill_masked_rows(obj: Optional[ls.LoadedObject],
                                  selector: Optional[Mapping[str, Any]], *,
                                  q: Any, seq_len: Any,
                                  token_lookup: Callable[[str], Sequence[int]],
                                  req_id: str) -> Optional[int]:
    """The ROW-LOCAL path's ``prefill_masked_rows`` (:class:`RequestSlot`) for
    one selector-carrying segment — DIAGNOSTIC ONLY. That path builds no
    HookContext: no active op consults a position, so it reads no positions
    and skips every admission check. Yet the record still owes the gate the
    mask geometry (an isolated ``@all_positions`` prefill filed ``None`` for a
    mask covering every row, and a declared twin's width depended on whether a
    positional neighbour shared the forward). For the ONE shape the plan path
    would also file as a prefill — a COMPLETE cold prefill, ``computed == 0``
    and ``q >= 2``, ``lens_batch.derive_batch_plan``'s complete-prefill cell —
    this builds, BESIDE the slot and never on it, the context
    ``lens_batch.build_segment_contexts`` builds for that segment:
    positions ``0..q-1`` (vLLM's contract for a cold prompt; the forward's
    positions tensor is NOT read), ids ``tokens[:q]`` sliced as the plan path
    slices them, the selector's facts threaded in — and counts with
    :func:`prefill_masked_rows_of`. So an ``all`` cell reads P (a BOS-skipping
    ``all`` op needs a context and never takes this path), and a scale-0 twin
    reads its donor's geometry under the request's render (the ON header's
    width on a declared twin) — the values the same request files when
    co-batched. ``None`` for a decode step, a chunk, a resumed shape (the plan
    path gives an untracked resume no context either), a token source behind
    the forward, an unlocatable boundary — and it NEVER raises: nothing here
    refuses a request or touches anything the apply reads."""
    if obj is None or selector is None:
        return None
    try:
        q_i, seq_i = int(q), int(seq_len)
        if q_i < 2 or seq_i != q_i:
            return None                  # not the lattice's complete-prefill cell
        toks = token_lookup(req_id)
        if len(toks) < q_i:
            return None                  # the token source is behind the forward
        ctx = la.HookContext(phase=la.PHASE_PREFILL, positions=tuple(range(q_i)),
                             token_ids=tuple(int(t) for t in toks[:q_i]))
        return prefill_masked_rows_of(obj, _ctx_with_selector_facts(ctx, dict(selector)))
    except Exception:  # noqa: BLE001 — an instrument, never the apply path
        return None


def _diagnostic_seq_lens(seq_lens: Any, n: int) -> List[Any]:
    """``seq_lens`` as ints for the row-local diagnostics, or ``None`` per
    entry when unreadable or mis-sized: the caching-OFF row-local path never
    needed them, so a malformed value files ``None`` rather than starting to
    refuse (the caching-ON branch has already read them loudly)."""
    try:
        seq = _int_list("seq_lens", seq_lens)
    except Exception:  # noqa: BLE001 — a diagnostic read, never a refusal
        return [None] * n
    return list(seq) if len(seq) == n else [None] * n


def _row_local_slots(req_ids: Sequence[str], bounds: Sequence[Tuple[int, int]],
                     selectors: Sequence[Optional[Dict[str, Any]]],
                     objs: Sequence[Optional[ls.LoadedObject]], *,
                     seq_lens: Any, token_lookup: Callable[[str], Sequence[int]],
                     resumed: Optional[Sequence[Optional[ResumedPrefill]]] = None,
                     refusals: Optional[Sequence[Optional[la.ResumedPrefillRefusal]]] = None,
                     ) -> Tuple[RequestSlot, ...]:
    """The ROW-LOCAL path's slots (:func:`build_request_slots`): rows from
    the bounds, ``ctx`` None (no active op consults a position), the resume
    facts / contained refusals when the caching-ON branch decided them —
    plus the record's DIAGNOSTICS beside each selector-carrying slot
    (:class:`RequestSlot`, finding A1): ``facts`` off the selector,
    ``prefill_masked_rows`` from the request's own complete prefill
    (:func:`row_local_prefill_masked_rows`). The diagnostics read
    ``seq_lens`` (contained, :func:`_diagnostic_seq_lens`) and the request's
    ids; they never read positions, and they cannot refuse."""
    seq = _diagnostic_seq_lens(seq_lens, len(bounds))
    out: List[RequestSlot] = []
    for i, (rid, (s, e)) in enumerate(zip(req_ids, bounds)):
        sel = selectors[i]
        out.append(RequestSlot(
            req_id=rid, rows=slice(s, e), selector=sel, obj=objs[i], ctx=None,
            resumed=None if resumed is None else resumed[i],
            refusal=None if refusals is None else refusals[i],
            facts=None if sel is None else selector_facts(sel),
            prefill_masked_rows=(None if sel is None else row_local_prefill_masked_rows(
                objs[i], sel, q=e - s, seq_len=seq[i],
                token_lookup=token_lookup, req_id=rid))))
    return tuple(out)


def _as_int(x: Any) -> int:
    if torch.is_tensor(x):
        return int(x.item())
    return int(x)


# --- forward-metadata normalization -----------------------------------------------
#
# The position logic consumes ONE thing from the forward context's
# ``attn_metadata``: the batch's row segmentation (``query_start_loc``
# / ``seq_lens`` / ``num_actual_tokens``) — a property of the FORWARD,
# not of any layer's attention kind. On a uniform-attention model every
# per-layer metadata object carries it under those canonical V1
# flash-attention names. On a hybrid model the forward context spans
# metadata CLASSES: Qwen3.5's 8 full-attention layers carry the
# canonical fields, while its 24 gated-delta-net layers carry
# ``GDNAttentionMetadata`` (vllm v1/attention/backends/gdn_attn.py) —
# ``num_actual_tokens`` but NO top-level ``query_start_loc`` /
# ``seq_lens``; the same segmentation lives under split names
# (``non_spec_query_start_loc``, ...). So the distinct metadata objects
# are PARTITIONED: the segmentation is taken only from the
# canonical-field-BEARING groups (which must exist and agree
# internally), and every non-bearing group is CROSS-CHECKED against it
# wherever its fields permit. A genuine disagreement anywhere still
# refuses — a mask computed under a wrong segmentation is silently
# wrong, which is exactly the failure class this refusal exists to
# prevent. The non-bearing groups alone are never authoritative.

_SEGMENTATION_ATTRS = ("query_start_loc", "seq_lens", "num_actual_tokens")


def _bears_segmentation(md: Any) -> bool:
    """True when this metadata object carries the batch's row
    segmentation under the canonical V1 flash-attention names — all
    three fields present and populated (an attribute that exists but
    holds ``None`` does not bear the segmentation)."""
    return (torch.is_tensor(getattr(md, "query_start_loc", None))
            and torch.is_tensor(getattr(md, "seq_lens", None))
            and getattr(md, "num_actual_tokens", None) is not None)


def _cross_check_group(md0: Any, other: Any) -> None:
    """Cross-check a non-segmentation-bearing metadata group against the
    canonical segmentation ``md0``, wherever its fields permit; any
    provable disagreement refuses, and an unrecognized/unverifiable
    shape refuses rather than passing unchecked.

    GPU-VERIFY (field names read from vllm
    v1/attention/backends/gdn_attn.py at the pinned tag, not executed
    here): ``GDNAttentionMetadata.num_actual_tokens`` counts the same
    flattened rows as the canonical field, and its builder sets
    ``non_spec_query_start_loc`` to the batch ``query_start_loc``
    whenever no speculative decode is in flight — under spec decode the
    non-spec split genuinely diverges from the batch segmentation, so
    that state is a refusal (cannot verify), never a skipped check."""
    name = type(other).__name__
    # Canonical names, whichever this group carries directly.
    nat = getattr(other, "num_actual_tokens", None)
    if nat is not None and _as_int(nat) != _as_int(md0.num_actual_tokens):
        raise LensWorkerError(
            f"attention metadata {name} reports num_actual_tokens == "
            f"{_as_int(nat)} but the segmentation-bearing group reports "
            f"{_as_int(md0.num_actual_tokens)} — the KV cache groups "
            "disagree on the forward's row count; a mask computed under "
            "either would be silently wrong for the other's layers; "
            "refusing")
    qsl = getattr(other, "query_start_loc", None)
    if torch.is_tensor(qsl) and not _tensors_equal(qsl, md0.query_start_loc):
        raise LensWorkerError(
            f"attention metadata {name} carries a query_start_loc that "
            "disagrees with the segmentation-bearing group's — the KV "
            "cache groups disagree on the forward's row segmentation; "
            "refusing")
    sl = getattr(other, "seq_lens", None)
    if torch.is_tensor(sl) and not _tensors_equal(sl, md0.seq_lens):
        raise LensWorkerError(
            f"attention metadata {name} carries seq_lens that disagree "
            "with the segmentation-bearing group's; refusing")
    # The GDN split names (non_spec_* covers the whole batch only when
    # no speculative decode is in flight).
    n_spec = getattr(other, "num_spec_decodes", None)
    n_spec_tok = getattr(other, "num_spec_decode_tokens", None)
    if ((n_spec is not None and _as_int(n_spec) > 0)
            or (n_spec_tok is not None and _as_int(n_spec_tok) > 0)):
        raise LensWorkerError(
            f"attention metadata {name} reports speculative decode in "
            "flight (num_spec_decodes/num_spec_decode_tokens > 0) — its "
            "split segmentation fields no longer describe the whole "
            "batch, so the cross-check against the segmentation-bearing "
            "group cannot be made; refusing rather than serving "
            "unverified rows")
    nsq = getattr(other, "non_spec_query_start_loc", None)
    if torch.is_tensor(nsq):
        if n_spec is None:
            raise LensWorkerError(
                f"attention metadata {name} carries "
                "non_spec_query_start_loc but no num_spec_decodes — not "
                "the GDNAttentionMetadata shape this cross-check reads; "
                "cannot prove the split covers the whole batch; refusing")
        if not _tensors_equal(nsq, md0.query_start_loc):
            raise LensWorkerError(
                f"attention metadata {name}'s non_spec_query_start_loc "
                "disagrees with the segmentation-bearing group's "
                "query_start_loc (with zero speculative decodes they "
                "must be equal) — the KV cache groups disagree on the "
                "forward's row segmentation; refusing")


def resolve_attn_metadata(md: Any) -> Optional[Tuple[Any, Any, Any]]:
    """Normalize a forward context's raw ``attn_metadata`` into the
    ``(query_start_loc, seq_lens, num_actual_tokens)`` triple the
    position logic consumes, or ``None`` when the forward provably
    carries no metadata (warmup). Module-level and vLLM-free so every
    branch — including the hybrid-model partition — runs on CPU under
    tests; :meth:`RealWorkerView.forward_metadata` is the one caller
    with a live forward context."""
    if md is None:
        return None
    if isinstance(md, list):
        if len(md) != 1:
            raise LensWorkerError(
                f"attn_metadata is a {len(md)}-element list "
                "(micro-batched execution?) — the per-forward "
                "accounting assumes one batch per forward; refusing")
        md = md[0]
        if md is None:
            return None
    if isinstance(md, dict):
        distinct = list({id(v): v for v in md.values()}.values())
        if not distinct:
            return None
        if any(v is None for v in distinct):
            raise LensWorkerError(
                "a per-layer attn_metadata entry is None — cannot prove "
                "the forward's row segmentation covers those layers; "
                "refusing")
        bearing = [v for v in distinct if _bears_segmentation(v)]
        if not bearing:
            raise LensWorkerError(
                "no KV cache group's attention metadata carries the "
                "canonical (query_start_loc, seq_lens, "
                "num_actual_tokens) segmentation fields — got "
                f"{sorted(set(type(v).__name__ for v in distinct))}; the "
                "position logic cannot take a split/partial layout "
                "(e.g. a linear-attention group's non_spec_* fields) as "
                "authoritative; refusing rather than guessing rows")
        md0 = bearing[0]
        for other in bearing[1:]:
            same = (_tensors_equal(other.query_start_loc,
                                   md0.query_start_loc)
                    and _tensors_equal(other.seq_lens, md0.seq_lens)
                    and _as_int(other.num_actual_tokens)
                    == _as_int(md0.num_actual_tokens))
            if not same:
                raise LensWorkerError(
                    "per-layer attention metadata disagrees across KV "
                    "cache groups: the segmentation-bearing groups "
                    "disagree pairwise on (query_start_loc, seq_lens, "
                    "num_actual_tokens) — a mask computed under either "
                    "would be silently wrong for the other's layers; "
                    "refusing")
        for other in distinct:
            if not _bears_segmentation(other):
                _cross_check_group(md0, other)
    else:
        md0 = md
    for attr in _SEGMENTATION_ATTRS:
        if not hasattr(md0, attr):
            raise LensWorkerError(
                f"attention metadata {type(md0).__name__} has no "
                f"{attr!r} — not the vLLM V1 flash-attention layout "
                "this wiring reads; refusing rather than guessing rows")
    return (md0.query_start_loc, md0.seq_lens, md0.num_actual_tokens)


# --- the worker view seam ---------------------------------------------------------
#
# Every real-vLLM touch lives behind this narrow surface, so all the
# logic above/below runs on CPU against fakes. A view must provide:
#
#   num_reqs() -> int
#   req_ids() -> Sequence[str]            # exactly num_reqs entries
#   extra_args(req_id) -> Optional[Mapping]
#   request_tokens(req_id) -> Sequence[int]   # prompt + generated so far
#   prompt_len(req_id) -> int             # the SUBMITTED prompt's token
#                                         # count alone (the per-request
#                                         # prefill invariants' anchor)
#   forward_metadata() -> Optional[(query_start_loc, seq_lens,
#                                    num_actual_tokens)]  # None = no fwd ctx
#   decoder_layers() -> Sequence[torch.nn.Module]
#   model_device_dtype() -> (device, dtype)
#   enforce_eager() -> Optional[bool]     # None = could not determine
#   prefix_caching_enabled() -> Optional[bool]  # None = could not determine
#   v2_model_runner_active() -> Optional[bool]
#   missing_runner_attrs() -> Sequence[str]
#   tensor_parallel_size() -> Optional[int]  # None = could not determine;
#                                         # consulted only when o_proj-site
#                                         # ops are registered (preflight
#                                         # refuses TP != 1 there — the
#                                         # o_proj INPUT is sharded across
#                                         # ranks under TP)


class RealWorkerView:
    """The ONE vllm/vllm-lens-touching implementation of the view
    surface (imports are lazy, inside methods, so this module imports
    cleanly on CPU). EVERY method is GPU-VERIFY: read against the
    extracted vllm-lens 1.2.1 source and the pinned-vLLM V1 layout,
    never executed here — the module docstring's table carries the
    per-attribute confidence."""

    def __init__(self, worker: Any):
        self._worker = worker
        #: the one-time GPU-VERIFY latches (CPU_MIRROR_RUNNER_ATTRS):
        #: set after the runner's CPU mirrors were held equal to the
        #: device tensors on the first classified forward.
        self._cpu_mirror_verified = False
        self._positions_verified = False

    # -- per-forward reads -------------------------------------------------------

    def forward_key(self) -> Any:
        # GPU-VERIFY: the live ForwardContext's attn_metadata object — a
        # FRESH dict per execute_model at the pinned tag
        # (gpu_model_runner._build_attention_metadata: ``attn_metadata:
        # PerLayerAttnMetadata = {}`` per call), set for the forward by
        # set_forward_context. The classification cache holds a strong
        # reference to it, so its identity cannot be recycled by a later
        # step while cached (module docstring). None = no forward context
        # (warmup): nothing to key on, nothing cached.
        from vllm.forward_context import (get_forward_context,
                                          is_forward_context_available)
        if not is_forward_context_available():
            return None
        return get_forward_context().attn_metadata

    def num_reqs(self) -> int:
        # GPU-VERIFY: runner.input_batch.num_reqs (vllm-lens's own read).
        return int(self._worker.model_runner.input_batch.num_reqs)

    def req_ids(self) -> Sequence[str]:
        # GPU-VERIFY: runner.input_batch.req_ids, first num_reqs entries
        # (vllm-lens indexes req_ids[i] for i in range(num_reqs)).
        n = self.num_reqs()
        return tuple(self._worker.model_runner.input_batch.req_ids[:n])

    def extra_args(self, req_id: str) -> Optional[Mapping[str, Any]]:
        # GPU-VERIFY: runner.requests[req_id].sampling_params.extra_args
        # (vllm-lens's own read). A scheduled request MISSING from the
        # state dict refuses: serving it unsteered because its selector
        # could not be read is exactly the silent downgrade this module
        # exists to prevent.
        state = self._worker.model_runner.requests.get(req_id)
        if state is None:
            raise LensWorkerError(
                f"request {req_id!r} is in the input batch but not in the "
                "runner's request state — cannot read its selector; "
                "refusing to serve it blind")
        params = state.sampling_params
        return params.extra_args if params is not None else None

    def request_tokens(self, req_id: str) -> Sequence[int]:
        # GPU-VERIFY (the least-verified read in this module — vllm-lens
        # never touches token ids): CachedRequestState.prompt_token_ids +
        # .output_token_ids, with the fed decode token already appended
        # when the hook fires. Direct attribute access on purpose — an
        # attribute rename raises AttributeError at the first request
        # instead of a getattr default silently serving no ids.
        state = self._worker.model_runner.requests.get(req_id)
        if state is None:
            raise LensWorkerError(
                f"request {req_id!r} is in the input batch but not in the "
                "runner's request state — cannot read its token ids")
        prompt = state.prompt_token_ids
        if prompt is None:
            raise LensWorkerError(
                f"request {req_id!r} carries no prompt_token_ids (a "
                "prompt-embeddings request?) — the response mask and the "
                "prefill accounting need token ids; refusing")
        # a VIEW over the two live lists (RequestTokens): O(1) length,
        # ranged slicing — never the two full copies per (request,
        # layer) the profile charged 7% of the step to.
        return RequestTokens(prompt, state.output_token_ids)

    def prompt_len(self, req_id: str) -> int:
        # GPU-VERIFY: the SUBMITTED prompt's token count alone —
        # len(CachedRequestState.prompt_token_ids), the same field
        # request_tokens reads (declared at the pinned tag; None under a
        # prompt-embeds request, refused). Unlike request_tokens it
        # carries NO decode-timing assumption: the prompt is fixed at
        # submission, which is what makes it the anchor for the
        # per-request prefill invariants.
        state = self._worker.model_runner.requests.get(req_id)
        if state is None:
            raise LensWorkerError(
                f"request {req_id!r} is in the input batch but not in the "
                "runner's request state — cannot read its prompt length")
        prompt = state.prompt_token_ids
        if prompt is None:
            raise LensWorkerError(
                f"request {req_id!r} carries no prompt_token_ids (a "
                "prompt-embeddings request?) — the per-request prefill "
                "invariants need the submitted prompt length; refusing")
        return len(prompt)

    def forward_metadata(self) -> Optional[Tuple[Any, Any, Any]]:
        # GPU-VERIFY: vllm.forward_context + the V1 per-layer-dict
        # attn_metadata layout (vllm-lens's own reads, hardened: where
        # it takes list[0] unconditionally and the first dict value with
        # query_start_loc, we refuse a multi-element list, take the
        # segmentation only from the canonical-field-bearing groups —
        # which must agree — and cross-check every other group; see
        # resolve_attn_metadata, which carries the whole normalization
        # so it runs on CPU under tests).
        from vllm.forward_context import (get_forward_context,
                                          is_forward_context_available)
        if not is_forward_context_available():
            return None
        return resolve_attn_metadata(get_forward_context().attn_metadata)

    def forward_metadata_cpu(self) -> Optional[Tuple[Any, Any, Any]]:
        """:meth:`forward_metadata`'s segmentation triple with the two
        tensors replaced by the runner's SAME-STEP CPU mirrors
        (``query_start_loc.cpu[:n+1]``, ``optimistic_seq_lens_cpu[:n]``
        — CPU_MIRROR_RUNNER_ATTRS), so the classification's
        ``.tolist()`` reads cost no device sync. The device triple is
        still resolved first: every KV-cache-group cross-check runs
        exactly as before (sync-free where the groups alias one buffer,
        ``_tensors_equal``). GPU-VERIFY, ONCE: on the first forward this
        runs for, the mirrors are held equal to the device tensors and a
        mismatch refuses — the runner's CPU/device pairing is a fact of
        the pinned tag, not per request. A runner lacking the mirrors
        refuses at preflight (``missing_runner_attrs``); here it would
        refuse too, never fall back to the device read."""
        resolved = self.forward_metadata()
        if resolved is None:
            return None
        qsl_dev, seq_dev, nat = resolved
        runner = self._worker.model_runner
        n = self.num_reqs()
        try:
            qsl_cpu = runner.query_start_loc.cpu[:n + 1]
            seq_cpu = runner.optimistic_seq_lens_cpu[:n]
        except AttributeError as e:
            raise LensWorkerError(
                "the model runner lacks the CPU mirrors of query_start_loc "
                "/ seq_lens (CPU_MIRROR_RUNNER_ATTRS) — not the pinned "
                f"0.25.1 runner surface ({e}); refusing rather than "
                "reading the device tensors with a sync per hook") from e
        for name, cpu_t, dev_t, want in (("query_start_loc", qsl_cpu,
                                          qsl_dev, n + 1),
                                         ("seq_lens", seq_cpu, seq_dev, n)):
            if tuple(cpu_t.shape) != (want,):
                raise LensWorkerError(
                    f"the runner's CPU {name} mirror covers "
                    f"{tuple(cpu_t.shape)} entries for {n} scheduled "
                    "requests — refusing to pair rows with requests")
            if torch.is_tensor(dev_t) and tuple(dev_t.shape) != (want,):
                raise LensWorkerError(
                    f"attention metadata carries {name} of shape "
                    f"{tuple(dev_t.shape)} for {n} scheduled requests "
                    "(cudagraph padding? the launch pins enforce_eager) — "
                    "refusing to pair rows with requests")
        if not self._cpu_mirror_verified:
            # GPU-VERIFY (one sync, once per worker process): the CPU
            # mirrors ARE what the runner copied to the device this step.
            for name, cpu_t, dev_t in (("query_start_loc", qsl_cpu, qsl_dev),
                                       ("seq_lens", seq_cpu, seq_dev)):
                if torch.is_tensor(dev_t) and not torch.equal(
                        cpu_t.to(device=dev_t.device, dtype=dev_t.dtype),
                        dev_t):
                    raise LensWorkerError(
                        f"the runner's CPU {name} mirror disagrees with "
                        "the attention metadata's device tensor on the "
                        "first classified forward — the pinned 0.25.1 "
                        "CPU/device buffer pairing this wiring reads has "
                        "drifted; refusing to classify from either")
            self._cpu_mirror_verified = True
        return qsl_cpu, seq_cpu, nat

    def forward_positions(self, device_positions: Any, num_tokens: int
                          ) -> Any:
        """The forward's per-row absolute positions from the runner's
        CPU state, for the plan lattice's positions cross-check without
        a device sync (F2 of the lens-vectorize landing):

        - an M-RoPE model (the layer received a 2-D ``(3, T)`` stack —
          Qwen3.5, GLM-4.6V-Flash) -> ``runner.mrope_positions.cpu[:,
          :T]``, the exact CPU source the runner copied to the device
          this step (``_prepare_inputs``); the plan path then reduces
          the stack to its one row on the CPU (``_normalize_positions``
          — disagreeing rows, a multimodal batch, still refuse);
        - otherwise -> DERIVED: the pinned runner computes the 1-D
          ``positions`` buffer ON the device as
          ``num_computed_tokens[req] + query offset`` and keeps no CPU
          copy, so the same arithmetic is run here over
          ``input_batch.num_computed_tokens_cpu`` and the CPU
          ``query_start_loc`` — an INDEPENDENT source from the
          ``seq_lens`` the plan derives ``computed`` from, so the
          cross-check still compares two records.

        GPU-VERIFY, ONCE: the first time this runs, the CPU positions
        are held equal to ``device_positions`` (the tensor the layer was
        actually handed) and a mismatch refuses. Read the plan path's
        positions from here, never from a per-hook ``.tolist()`` of the
        device tensor."""
        runner = self._worker.model_runner
        T = int(num_tokens)
        stacked = torch.is_tensor(device_positions) and device_positions.dim() == 2
        if stacked:
            buf = getattr(runner, "mrope_positions", None)
            cpu = getattr(buf, "cpu", None)
            if not torch.is_tensor(cpu) or cpu.dim() != 2:
                raise LensWorkerError(
                    "the decoder layer received a 2-D (M-RoPE) positions "
                    "stack but the runner carries no mrope_positions.cpu "
                    "buffer — not the pinned 0.25.1 runner surface; "
                    "refusing to classify positions")
            if cpu.shape[1] < T or device_positions.shape[1] < T:
                raise LensWorkerError(
                    f"M-RoPE positions cover {cpu.shape[1]} (cpu) / "
                    f"{device_positions.shape[1]} (device) tokens but the "
                    f"metadata covers {T} rows — refusing to index rows "
                    "the buffers do not carry")
            pos_cpu = cpu[:, :T]
            dev_view = device_positions[:, :T]
        else:
            n = self.num_reqs()
            try:
                qsl_cpu = runner.query_start_loc.cpu[:n + 1].tolist()
                computed = runner.input_batch.num_computed_tokens_cpu[:n]
            except AttributeError as e:
                raise LensWorkerError(
                    "the model runner lacks query_start_loc.cpu / "
                    "input_batch.num_computed_tokens_cpu "
                    "(CPU_MIRROR_RUNNER_ATTRS) — not the pinned 0.25.1 "
                    f"runner surface ({e}); refusing") from e
            if len(qsl_cpu) != n + 1 or len(computed) != n:
                raise LensWorkerError(
                    f"CPU segmentation covers {len(qsl_cpu) - 1} segments "
                    f"/ {len(computed)} computed counts for {n} scheduled "
                    "requests — refusing to derive positions")
            vals: List[int] = []
            for i in range(n):
                c = int(computed[i])
                vals.extend(range(c, c + (qsl_cpu[i + 1] - qsl_cpu[i])))
            if len(vals) != T:
                raise LensWorkerError(
                    f"derived positions cover {len(vals)} rows but the "
                    f"metadata covers {T} — query_start_loc and "
                    "num_actual_tokens disagree; refusing")
            pos_cpu = torch.tensor(vals, dtype=torch.long)
            if torch.is_tensor(device_positions):
                if device_positions.dim() != 1 or device_positions.shape[0] < T:
                    raise LensWorkerError(
                        "the decoder layer's positions tensor has shape "
                        f"{tuple(device_positions.shape)} for {T} rows — "
                        "not a positions layout this wiring reads; "
                        "refusing")
                dev_view = device_positions[:T]
            else:
                dev_view = None
        if not self._positions_verified and dev_view is not None:
            # GPU-VERIFY (one sync, once per worker process).
            if not torch.equal(pos_cpu.to(device=dev_view.device,
                                          dtype=dev_view.dtype), dev_view):
                raise LensWorkerError(
                    "the CPU positions (the runner's mrope_positions.cpu, "
                    "or num_computed_tokens_cpu + the query offset) "
                    "disagree with the positions tensor the decoder layer "
                    "was handed on the first classified forward — the "
                    "pinned 0.25.1 positions arithmetic this wiring "
                    "mirrors has drifted; refusing to classify from "
                    "either")
            self._positions_verified = True
        return pos_cpu

    # -- startup reads -------------------------------------------------------------

    def decoder_layers(self) -> Sequence[Any]:
        # GPU-VERIFY: vllm-lens's own layer discovery, reused so our
        # hooks land exactly where install_hooks' would
        # (language_model.model.layers / model.decoder.layers /
        # model.layers). The served families' vLLM trees at the pinned
        # tag: llama, qwen3_next (Qwen3.5) and nemotron_h expose
        # ``model.layers``; GLM-4.6V-Flash's Glm4vForConditionalGeneration
        # nests the text stack under ``language_model``; gemma-4's
        # conditional-generation wrapper remaps the checkpoint's
        # ``model.language_model.*`` onto ``model.*``. When vllm-lens
        # cannot resolve a tree, the repo's own accessor (the HF-side
        # ``sisteer.models.get_decoder_layers`` — ``model.language_model.
        # layers``, ``backbone.layers``, ...) is tried; a tree neither
        # resolves refuses loudly at preflight rather than hooking
        # nothing.
        model = self._worker.model_runner.model
        try:
            from vllm_lens._worker_ext import _get_layers
            return list(_get_layers(model))
        except AttributeError as lens_err:
            from sisteer.models import get_decoder_layers
            try:
                return list(get_decoder_layers(model))
            except ValueError as ours_err:
                raise LensWorkerError(
                    "cannot locate the decoder layers of "
                    f"{type(model).__name__}: vllm-lens said {lens_err}; "
                    f"sisteer.models said {ours_err} — refusing to install "
                    "hooks nowhere") from ours_err

    def model_device_dtype(self) -> Tuple[Any, Any]:
        # GPU-VERIFY: vllm-lens's own idiom (set_steering_data).
        p = next(self._worker.model_runner.model.parameters())
        return p.device, p.dtype

    def enforce_eager(self) -> Optional[bool]:
        # GPU-VERIFY: model_config unpacked onto the Worker (WorkerBase).
        # None (attribute missing) makes preflight REFUSE — "cannot
        # verify eager" is a refusal, never a default-pass.
        try:
            return bool(self._worker.model_config.enforce_eager)
        except AttributeError:
            return None

    def prefix_caching_enabled(self) -> Optional[bool]:
        # GPU-VERIFY: vllm_config.cache_config.enable_prefix_caching —
        # the same resolved value scheduler_config_report attests. None
        # (surface missing) makes preflight REFUSE: whether the plan
        # lattice may tolerate resumed (cache-hit) segments is keyed on
        # the LIVE cache state, and "cannot verify" is a refusal, never
        # a default in either direction.
        try:
            cache = self._worker.vllm_config.cache_config
            return bool(cache.enable_prefix_caching)
        except AttributeError:
            return None

    def v2_model_runner_active(self) -> Optional[bool]:
        # GPU-VERIFY. None = the config predates the V2 runner flag,
        # which preflight treats as pass (mirroring the vllm-lens
        # plugin); the authoritative V1 check is the structural runner
        # surface below, which the V2 runner does not expose.
        try:
            cfg = self._worker.vllm_config
        except AttributeError:
            return None
        flag = getattr(cfg, "use_v2_model_runner", None)
        return None if flag is None else bool(flag)

    def tensor_parallel_size(self) -> Optional[int]:
        # GPU-VERIFY: vllm_config.parallel_config.tensor_parallel_size —
        # the resolved TP degree. None (surface missing) makes preflight
        # REFUSE when o_proj-site ops are registered: under TP the
        # o_proj module's INPUT is sharded across ranks while the hook's
        # delta would land after the all-reduce on every rank, so an
        # unverifiable TP is a refusal, never a default-pass.
        try:
            return int(self._worker.vllm_config
                       .parallel_config.tensor_parallel_size)
        except (AttributeError, TypeError, ValueError):
            return None

    def missing_runner_attrs(self) -> Sequence[str]:
        # GPU-VERIFY: the structural V1 probe (REQUIRED_RUNNER_ATTRS) plus
        # the CPU mirrors the sync-free classification reads
        # (CPU_MIRROR_RUNNER_ATTRS) — preflight refuses on any absence.
        missing = []
        runner = getattr(self._worker, "model_runner", None)
        if runner is None:
            return ("model_runner",)
        for dotted in REQUIRED_RUNNER_ATTRS + CPU_MIRROR_RUNNER_ATTRS:
            obj = runner
            for part in dotted.split("."):
                if not hasattr(obj, part):
                    missing.append(dotted)
                    break
                obj = getattr(obj, part)
        return tuple(missing)


# --- the orchestrator --------------------------------------------------------------


@dataclass(frozen=True)
class InstallReport:
    """What one install did — the Phase-0 gate's registration evidence
    reads these (assertion 6: registration_count == 1, zero per-request
    registrations).

    ``layers`` are the BLOCK-site hook layers (the pre-existing meaning,
    kept so the gate's expected-hook-layers assertions read unchanged);
    ``oproj_layers`` are the layers additionally hooked at their
    ``self_attn.o_proj`` module (the lora kinds — lens_adapters
    HOOK_SITE_OPROJ). ``plan_layer`` is the layer whose hook consumes
    the run-scoped ledger: the FIRST-FIRING installed hook of the
    forward, which is the lowest hooked layer — at that layer the
    o_proj hook fires before the block hook (o_proj runs inside the
    layer's forward), so the site is part of the internal plan key but
    the layer number reported here is unambiguous either way."""

    registration_count: int
    layers: Tuple[int, ...]
    plan_layer: int
    cells: Tuple[str, ...]
    ledger_cells: Tuple[str, ...]
    oproj_layers: Tuple[int, ...] = ()
    #: RESUMED-PREFILL SAFETY per served cell (``lens_adapters.
    #: resumed_prefill_ok`` over the cell's ops): the worker's own statement
    #: of which cells it will serve a prefix-cache resume for. Rides in the
    #: install payload as ``resumed_prefill_ok_cells`` (an EXTRA key the
    #: frozen launcher validator copies through into the attestation's
    #: worker_reports), so the proxy half can read the predicate from the
    #: attestation instead of re-deriving it. ``None`` = not computed
    #: (a report built outside ``install``).
    resumed_prefill_ok_cells: Optional[Mapping[str, bool]] = None
    #: the three-valued policy per served cell (``lens_adapters.
    #: resumed_prefill_policy``) — the install payload's
    #: ``resumed_prefill_policy_cells``; ``None`` = not computed.
    resumed_prefill_policy_cells: Optional[Mapping[str, str]] = None
    #: the per-cell SERVE DECLARATIONS (:func:`serve_declarations_cells`,
    #: declaration-carrying cells only, bound to the loaded ops) — the
    #: install payload's ``lens_serve_declarations``; ``None`` = not computed
    #: (a report built outside ``install``). An EMPTY map is a stated fact.
    lens_serve_declarations: Optional[Mapping[str, Mapping[str, Any]]] = None
    #: the registry's DECLARED thinking-ON header as loaded and cross-checked
    #: (the install payload's ``assistant_header_ids_thinking_on``): the ids
    #: when the loaded ops carry the second header, ``None`` otherwise.
    #: ``thinking_header_attested`` False (the default) = not computed, the
    #: key omitted (a report built outside ``install``).
    assistant_header_ids_thinking_on: Any = None
    thinking_header_attested: bool = False


def install_report_payload(report: InstallReport) -> Dict[str, Any]:
    """The JSON-able dict ``sisteer_install_steering`` returns over the
    collective RPC — the ONE serializer, importable on CPU, so a parity
    test can hold the exact payload a production worker emits against
    the launcher-side validator (``vllm_server.validate_install_reports``).
    The six install fields are compared by that validator exactly (adding
    one means adding it to ``expected_install_state`` too, or readiness
    refuses); it copies every OTHER key through untouched into the
    attestation's top-level ``worker_reports``, which is how
    ``lens_reduction_mode`` — the served reduction mode (``stacked`` |
    ``per_row``, :func:`lens_adapters.reduction_mode`, the same value the
    ``sisteer_scheduler_config`` report stamps) — rides in the install report
    for consumers that read that list (every worker carries it; a reader
    requires them to agree)."""
    out = {"registration_count": report.registration_count,
           "layers": list(report.layers),
           "oproj_layers": list(report.oproj_layers),
           "plan_layer": report.plan_layer,
           "cells": list(report.cells),
           "ledger_cells": list(report.ledger_cells),
           LENS_REDUCTION_MODE_KEY: la.reduction_mode()}
    if report.resumed_prefill_ok_cells is not None:
        out[RESUMED_PREFILL_OK_CELLS_KEY] = {
            str(c): bool(v) for c, v in sorted(report.resumed_prefill_ok_cells.items())}
    if report.resumed_prefill_policy_cells is not None:
        out[RESUMED_PREFILL_POLICY_CELLS_KEY] = {
            str(c): str(v) for c, v in sorted(report.resumed_prefill_policy_cells.items())}
    if report.lens_serve_declarations is not None:
        out[LENS_SERVE_DECLARATIONS_KEY] = {
            str(c): json.loads(json.dumps(dict(v)))
            for c, v in sorted(report.lens_serve_declarations.items())}
    if report.thinking_header_attested:
        ids = report.assistant_header_ids_thinking_on
        out[LENS_THINKING_HEADER_KEY] = (None if ids is None
                                         else [int(t) for t in ids])
    return out


# --- the selection record (OBSERVATION ONLY) --------------------------------------


#: Per-request detail is capped: a long-lived server sees unboundedly
#: many request ids, and this is a diagnostic, not a log. The aggregate
#: counters keep counting past the cap; only the per-request map stops
#: growing, and ``n_requests_dropped`` says how many it declined.
SELECTION_RECORD_MAX_REQUESTS = 4096
#: distinct requested-cell keys kept in the pair table. Bounded already
#: in practice (``select`` refuses an unregistered id, so the key set is
#: the registry's), capped anyway so a fake/faulty registry cannot make
#: the record the thing that runs a worker out of memory.
SELECTION_RECORD_MAX_PAIR_KEYS = 256
#: mismatching observations kept verbatim.
SELECTION_RECORD_MAX_MISMATCHES = 64

#: The record's own statement of what it covers — shipped INSIDE the
#: record, because a diagnostic that implies coverage it does not have
#: is the same defect as the output-inference it was added to replace.
SELECTION_RECORD_COVERAGE = (
    "WHAT THIS OBSERVES. At the APPLY SITE, per request per steered "
    "layer, the identity of the LoadedObject the slot carries — the "
    "object WorkerObjectRegistry.select RESOLVED for that request's "
    "selector, READ rather than re-derived — plus whether applying it "
    "actually modified that request's rows at that layer. So it is a "
    "DIRECT observation of routing: which cell id, which expect_hash and "
    "which effective_scale the worker bound to each request, compared "
    "against what the request asked for. effective_scale plus "
    "n_applications_modifying is the cheap check that a scale-0 twin was "
    "not served in place of its live arm or the reverse: a twin must "
    "carry scale 0.0 and modify nothing, an arm must carry its serving "
    "scale and modify something. "
    "WHAT IT DOES NOT OBSERVE, stated plainly rather than left to be "
    "assumed. (1) OP-LEVEL BINDING: that the ops fired were computed "
    "from the right tensors is not measured, so a mis-binding DOWNSTREAM "
    "of selection — correct cell id resolved, wrong tensors bound — "
    "reads as a clean pass here. (2) registry.apply's own internal "
    "re-resolution: apply resolves the same selector against the same "
    "immutable registry and therefore returns this same object, but that "
    "is an argument about the registry, not something this record "
    "measures. Read it as 'routing and binding IDENTITY were correct', "
    "never as 'the right math ran'. "
    "WHICH APPLY PATH: each observation is also counted under the path that "
    "made it — n_applications_per_slot / n_applications_grouped (and the "
    "_modifying twins): per_slot = the per-request loop (prefill slots, "
    "objects forced per row, everything under SISTEER_LENS_PER_ROW_APPLY=1), "
    "grouped = one apply for a vectorized group of decode rows (shadow-"
    "verified or not). Record-level only, so a request's entry reads the "
    "same on either path; an observation whose caller named no path is "
    "counted in n_applications_unattributed, which a reader must treat as "
    "a missing attribution, never as either path. "
    "RESUMED PREFILLS (2026-09-06): a request entry also records whether its "
    "prompt was served as a prefix-cache RESUME by a forward the hooks fired "
    "on (prefill_resumed), how many prompt positions the cache served "
    "(prefill_cached_positions) and how many this worker computed "
    "(prefill_computed_positions), plus the modifying applications made on "
    "resumed rows; n_requests_resumed_prefill counts such requests. This "
    "attests THAT a resume happened and how far — never WHO wrote the cached "
    "blocks (the proxy's salt namespace is the only guarantee of that)."
)

#: the path attributions :meth:`SelectionRecord.observe` accepts
SELECTION_PATH_PER_SLOT = "per_slot"
SELECTION_PATH_GROUPED = "grouped"

#: What the record is FOR, in one line, next to the numbers.
SELECTION_RECORD_NOTE = (
    "DIRECT per-request selection evidence, recorded by the worker as it "
    "served. It is the observed half of selection_honored; comparing "
    "output token ids between compositions is the inferred half, and the "
    "two answer different questions. Observation only: nothing on the "
    "apply path reads this back, and recording it cannot raise (any "
    "internal error is counted in n_record_errors instead of "
    "propagating, so a diagnostic can never condemn a worker)."
)


class SelectionRecord:
    """A plain, bounded, append-only record of WHICH OBJECT the worker
    resolved and bound for each request (see
    :data:`SELECTION_RECORD_COVERAGE` for exactly which half of the
    claim that is).

    OBSERVATION ONLY, and structurally so: :meth:`observe` returns
    ``None``, never raises, and nothing on the apply path reads it back.
    It exists because ``selection_honored`` was otherwise verified by
    INFERENCE — concluding the right object was loaded because the
    output looked like that arm's output — which is sound only while
    outputs are composition-stable, i.e. only under
    ``VLLM_BATCH_INVARIANT=1``. Selection is a discrete fact and should
    be observed."""

    def __init__(self,
                 max_requests: int = SELECTION_RECORD_MAX_REQUESTS,
                 max_pair_keys: int = SELECTION_RECORD_MAX_PAIR_KEYS,
                 max_mismatches: int = SELECTION_RECORD_MAX_MISMATCHES,
                 reduction_mode: Optional[str] = None):
        self._max_requests = int(max_requests)
        self._max_pair_keys = int(max_pair_keys)
        self._max_mismatches = int(max_mismatches)
        #: the reduction mode the worker serves groups with (lens_adapters.
        #: reduction_mode(); ``None`` only for a record built outside a
        #: worker) — carried so the gate's evidence names the arithmetic it
        #: was gathered under; ``reset`` keeps it (it is not an observation)
        self.reduction_mode = reduction_mode
        self.reset()

    def reset(self) -> None:
        """Drop every observation. Scoping the record to one submitted
        batch is what lets the harness read a mixed batch's resolutions
        without the rest of the run's requests in the way."""
        self.n_observations = 0
        self.n_mismatched = 0
        self.n_requests_dropped = 0
        self.n_pair_keys_dropped = 0
        self.n_record_errors = 0
        self.first_record_error: Optional[str] = None
        #: per APPLY PATH (record level — see SELECTION_RECORD_COVERAGE):
        #: applications and modifying applications made by the per-slot
        #: loop, by a vectorized group, and by a caller that named neither.
        self.n_applications_per_slot = 0
        self.n_applications_grouped = 0
        self.n_applications_unattributed = 0
        self.n_modifying_per_slot = 0
        self.n_modifying_grouped = 0
        self.n_modifying_unattributed = 0
        #: requested cell id -> applied cell id -> count. Request-id
        #: independent, so it is still a real check when the caller
        #: cannot correlate its rows to the worker's request ids.
        self.pairs: Dict[str, Dict[str, int]] = {}
        #: req_id -> what was requested, what was bound, how often it was
        #: applied and how often that application changed the rows.
        self.requests: Dict[str, Dict[str, Any]] = {}
        self.mismatches: List[Dict[str, Any]] = []

    def observe(self, *, req_id: Any, layer: Any, requested: Any,
                requested_hash: Any, requested_scale: Any,
                applied: Optional[ls.LoadedObject], modified: bool,
                path: Optional[str] = None,
                resumed: Optional[ResumedPrefill] = None,
                ctx: Optional[la.RowContext] = None,
                facts: Optional[Mapping[str, Any]] = None,
                prefill_masked_rows: Optional[int] = None) -> None:
        """Record one (request, layer) application, made by ``path``
        (:data:`SELECTION_PATH_PER_SLOT` / :data:`SELECTION_PATH_GROUPED`;
        anything else is counted as unattributed, never as a path), and —
        when the slot carries them — the request's RESUMED-PREFILL facts
        (:class:`ResumedPrefill`: the forward was the prompt tail after a
        prefix-cache hit; recorded once per request, on its entry), and the
        request's DECLARED FACTS (``facts``: ``response_prefill`` /
        ``agent_loop`` / ``thinking_on``) plus how many rows of its PREFILL
        forward the bound object's ops selected (``prefill_masked_rows`` —
        ``lens_adapters.prefill_mask_rows``; the fit-support-transfer gate's
        evidence: 7 vs 5 on a Qwen response cell under the OFF vs the
        declared ON render, P on the ``@all_positions`` cell).
        The worker hands both in EXPLICITLY off the slot
        (``RequestSlot.facts`` / ``prefill_masked_rows`` — the selector's
        facts and the request's own prefill, the same values on the
        row-local and the plan path), and they are filed as
        given: the facts are the authority over any context's reading, the
        count fills the entry once. Only a caller that hands NEITHER falls
        back to reading them off ``ctx`` (a prefill
        :class:`~.lens_adapters.HookContext`; a ``DecodeRows`` group carries
        the two non-tail facts only). NEVER RAISES — an instrument that can
        condemn the worker it is instrumenting is worse than no instrument,
        so an internal error is counted and the first message kept, and the
        apply path continues untouched (a masked-row count that cannot be
        computed stays null)."""
        try:
            self._observe(req_id=req_id, layer=layer, requested=requested,
                          requested_hash=requested_hash,
                          requested_scale=requested_scale, applied=applied,
                          modified=bool(modified), path=path, resumed=resumed,
                          ctx=ctx, facts=facts,
                          prefill_masked_rows=prefill_masked_rows)
        except Exception as e:  # pragma: no cover — defence, not a path
            self.n_record_errors += 1
            if self.first_record_error is None:
                self.first_record_error = f"{type(e).__name__}: {e}"

    @staticmethod
    def _facts_of(ctx: Optional[la.RowContext]) -> Dict[str, bool]:
        """The declared facts a row context carries (``False`` where the
        context has no such field — a ``DecodeRows`` group carries the two
        non-tail bits only, and thinking is not a decode-phase fact)."""
        return {k: bool(getattr(ctx, k, False)) for k in ls.SELECTOR_FACT_KEYS}

    @staticmethod
    def _facts_dict(facts: Mapping[str, Any]) -> Dict[str, bool]:
        """Explicit facts (the slot's, off its selector — :func:`selector_facts`)
        normalised to the closed vocabulary: one bool per
        ``SELECTOR_FACT_KEYS``, foreign keys dropped."""
        return {k: bool(facts.get(k, False)) for k in ls.SELECTOR_FACT_KEYS}

    @staticmethod
    def _masked_rows_of(applied: Optional[ls.LoadedObject],
                        ctx: Optional[la.RowContext]) -> Optional[int]:
        """The context-only fallback: :func:`prefill_masked_rows_of` (a
        PREFILL context's count, else ``None``; never raises)."""
        return prefill_masked_rows_of(applied, ctx)

    def _observe(self, *, req_id: Any, layer: Any, requested: Any,
                 requested_hash: Any, requested_scale: Any,
                 applied: Optional[ls.LoadedObject], modified: bool,
                 path: Optional[str],
                 resumed: Optional[ResumedPrefill] = None,
                 ctx: Optional[la.RowContext] = None,
                 facts: Optional[Mapping[str, Any]] = None,
                 prefill_masked_rows: Optional[int] = None) -> None:
        if path == SELECTION_PATH_PER_SLOT:
            self.n_applications_per_slot += 1
            if modified:
                self.n_modifying_per_slot += 1
        elif path == SELECTION_PATH_GROUPED:
            self.n_applications_grouped += 1
            if modified:
                self.n_modifying_grouped += 1
        else:
            self.n_applications_unattributed += 1
            if modified:
                self.n_modifying_unattributed += 1
        want = str(requested)
        got = None if applied is None else str(applied.cell_id)
        got_hash = None if applied is None else str(applied.expect_hash)
        got_scale = (None if applied is None
                     else float(applied.effective_scale))
        mismatch = (got != want)

        self.n_observations += 1
        if mismatch:
            self.n_mismatched += 1
            if len(self.mismatches) < self._max_mismatches:
                self.mismatches.append(
                    {"req_id": str(req_id), "layer": _opt_int(layer),
                     "requested": want, "applied_cell": got})

        by_requested = self.pairs.get(want)
        if by_requested is None:
            if len(self.pairs) < self._max_pair_keys:
                by_requested = self.pairs.setdefault(want, {})
            else:
                self.n_pair_keys_dropped += 1
        if by_requested is not None:
            key = "<none>" if got is None else got
            by_requested[key] = by_requested.get(key, 0) + 1

        rid = str(req_id)
        entry = self.requests.get(rid)
        if entry is None:
            if len(self.requests) >= self._max_requests:
                self.n_requests_dropped += 1
                return
            entry = {
                "requested": want,
                "requested_expect_hash": (None if requested_hash is None
                                          else str(requested_hash)),
                "requested_scale": (None if requested_scale is None
                                    else float(requested_scale)),
                "applied_cell": got,
                "applied_expect_hash": got_hash,
                "applied_effective_scale": got_scale,
                "matched": not mismatch,
                # a request's resolution must not change mid-generation:
                # same selector, same registry, same object, every
                # forward. False is itself a finding.
                "stable": True,
                "n_applications": 0,
                "n_applications_modifying": 0,
                "layers": [],
                # RESUMED-PREFILL attestation (ResumedPrefill): whether
                # this request's prompt was ever served as a prefix-cache
                # RESUME by a forward the hooks fired on, and how many
                # prompt positions the cache served / this worker
                # computed. False / 0 / 0 for a cold prefill.
                "prefill_resumed": False,
                "prefill_cached_positions": 0,
                "prefill_computed_positions": 0,
                "n_applications_modifying_resumed": 0,
                # the request's DECLARED FACTS (the proxy-verified selector
                # bits — handed in off the slot, else read off a context)
                # and, from its prefill forward, how many rows the object's
                # ops selected: null until
                # a prefill is observed for this request
                "facts": (self._facts_dict(facts) if facts is not None
                          else self._facts_of(ctx)),
                "prefill_masked_rows": (int(prefill_masked_rows)
                                        if prefill_masked_rows is not None
                                        else self._masked_rows_of(applied, ctx)),
            }
            self.requests[rid] = entry
        else:
            if (entry["requested"] != want
                    or entry["applied_cell"] != got
                    or entry["applied_expect_hash"] != got_hash
                    or entry["applied_effective_scale"] != got_scale):
                entry["stable"] = False
        if mismatch:
            entry["matched"] = False
        entry["n_applications"] += 1
        if modified:
            entry["n_applications_modifying"] += 1
        if facts is not None:
            # the slot's own facts, off its selector: the authority, constant
            # per request, never overridden by a context's reading
            entry["facts"] = self._facts_dict(facts)
        elif isinstance(ctx, la.HookContext) and (
                ctx.phase == la.PHASE_PREFILL or not any(entry["facts"].values())):
            # context-only caller: a prefill context carries the whole
            # declared fact set; a later decode observation never overwrites
            # what the prefill established
            entry["facts"] = self._facts_of(ctx)
        if entry.get("prefill_masked_rows") is None:
            # filled once, by the first observation that knows it (the
            # prefill forward); the explicit count first, else the context's
            entry["prefill_masked_rows"] = (
                int(prefill_masked_rows) if prefill_masked_rows is not None
                else self._masked_rows_of(applied, ctx))
        if resumed is not None:
            if not entry["prefill_resumed"]:
                entry["prefill_resumed"] = True
                entry["prefill_cached_positions"] = int(resumed.cached)
                entry["prefill_computed_positions"] = int(resumed.computed)
            if modified:
                entry["n_applications_modifying_resumed"] += 1
        li = _opt_int(layer)
        if li is not None and li not in entry["layers"]:
            entry["layers"].append(li)
            entry["layers"].sort()

    def as_dict(self) -> Dict[str, Any]:
        """A JSON-able snapshot (the RPC return; plain scalars only)."""
        return {
            "note": SELECTION_RECORD_NOTE,
            "coverage": SELECTION_RECORD_COVERAGE,
            "n_observations": self.n_observations,
            "n_mismatched": self.n_mismatched,
            "all_observations_matched": (self.n_observations > 0
                                         and self.n_mismatched == 0),
            "n_requests_recorded": len(self.requests),
            "n_requests_dropped": self.n_requests_dropped,
            "n_pair_keys_dropped": self.n_pair_keys_dropped,
            "max_requests": self._max_requests,
            "n_record_errors": self.n_record_errors,
            "first_record_error": self.first_record_error,
            "n_unstable_requests": sum(
                1 for e in self.requests.values() if not e["stable"]),
            # RESUMED-PREFILL attestation, record level: requests whose
            # prompt resumed from the KV cache in a hooked forward
            "n_requests_resumed_prefill": sum(
                1 for e in self.requests.values() if e.get("prefill_resumed")),
            "n_applications_per_slot": self.n_applications_per_slot,
            "n_applications_grouped": self.n_applications_grouped,
            "n_applications_unattributed": self.n_applications_unattributed,
            "n_modifying_per_slot": self.n_modifying_per_slot,
            "n_modifying_grouped": self.n_modifying_grouped,
            "n_modifying_unattributed": self.n_modifying_unattributed,
            LENS_REDUCTION_MODE_KEY: self.reduction_mode,
            "pairs": {k: dict(v) for k, v in self.pairs.items()},
            "requests": {k: dict(v, layers=list(v["layers"]))
                         for k, v in self.requests.items()},
            "mismatches": [dict(m) for m in self.mismatches],
        }


def _opt_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class CoScheduleRecord:
    """OBSERVATION ONLY, like :class:`SelectionRecord`: per CLASSIFIED
    forward (one with at least one selector-carrying request), how many
    DISTINCT NONZERO vectorized groups shared it — the co-scheduling
    evidence the GPU gate's multigroup case needs (lens-vectorize review
    N1, 2026-09-04): every other check that case makes (each cell modified
    something, controls unchanged, cells differ, reference == candidate) is
    satisfiable by a sequence of single-group forwards, because HTTP
    admission under BI=0 scheduling can spread a wave across steps; only the
    worker can attest that two distinct groups were in ONE forward.

    A group is NONZERO when its object's effective scale is not 0.0 (the
    ``none__lens0`` twin never counts); its KEY is the grouping key
    ``build_slot_groups`` used — (cell, expect_hash, selector scale, row
    context class); a SAME-RENDER PAIR is two distinct CELLS whose groups
    carry the same row context (both plain decode, both
    ``response_prefill``, both row-local). Observed once per forward on
    the real view (the classification is cached per forward — module
    docstring); a view without a forward key classifies per hook and so
    counts per hook, which the CPU tests state where they rely on it.

    Read over the ``sisteer_coschedule_record`` RPC. Never raises, returns
    None, nothing on the apply path reads it back (the SelectionRecord
    discipline). The base commit has no such record: the per-row path's
    correctness does not depend on batch composition, so a reference
    standup only supplies outputs and attests nothing here."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.n_forwards = 0
        self.n_forwards_with_groups = 0
        self.max_distinct_nonzero_groups_in_one_forward = 0
        self.max_distinct_nonzero_cells_in_one_forward = 0
        self.n_forwards_with_ge2_groups = 0
        self.n_forwards_with_ge2_cells = 0
        self.n_forwards_with_same_render_pair = 0
        self.n_record_errors = 0
        self.first_record_error: Optional[str] = None

    def observe(self, groups: Sequence["SlotGroup"]) -> None:
        """Record one classified forward's vectorized groups. NEVER RAISES."""
        try:
            self._observe(groups)
        except Exception as e:  # pragma: no cover — defence, not a path
            self.n_record_errors += 1
            if self.first_record_error is None:
                self.first_record_error = f"{type(e).__name__}: {e}"

    def _observe(self, groups: Sequence["SlotGroup"]) -> None:
        # compute everything FIRST, mutate LAST: a malformed observation
        # leaves every counter untouched (only the error count moves).
        groups = tuple(groups)
        nonzero = [g for g in groups
                   if g.slots and g.slots[0].obj is not None
                   and float(g.slots[0].obj.effective_scale) != 0.0]
        keys = {(g.cell_id, g.expect_hash, g.scale, g.ctx) for g in nonzero}
        cells = {g.cell_id for g in nonzero}
        by_render: Dict[Any, set] = {}
        for g in nonzero:
            by_render.setdefault(g.ctx, set()).add(g.cell_id)
        same_render_pair = any(len(c) >= 2 for c in by_render.values())
        self.n_forwards += 1
        if groups:
            self.n_forwards_with_groups += 1
        self.max_distinct_nonzero_groups_in_one_forward = max(
            self.max_distinct_nonzero_groups_in_one_forward, len(keys))
        self.max_distinct_nonzero_cells_in_one_forward = max(
            self.max_distinct_nonzero_cells_in_one_forward, len(cells))
        if len(keys) >= 2:
            self.n_forwards_with_ge2_groups += 1
        if len(cells) >= 2:
            self.n_forwards_with_ge2_cells += 1
        if same_render_pair:
            self.n_forwards_with_same_render_pair += 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "note": ("co-scheduling evidence: per classified forward, the distinct "
                     "NONZERO vectorized groups (cell, expect_hash, scale, row context) "
                     "that shared it; observation only, one per forward on the real view"),
            "n_forwards": self.n_forwards,
            "n_forwards_with_groups": self.n_forwards_with_groups,
            "max_distinct_nonzero_groups_in_one_forward":
                self.max_distinct_nonzero_groups_in_one_forward,
            "max_distinct_nonzero_cells_in_one_forward":
                self.max_distinct_nonzero_cells_in_one_forward,
            "n_forwards_with_ge2_groups": self.n_forwards_with_ge2_groups,
            "n_forwards_with_ge2_cells": self.n_forwards_with_ge2_cells,
            "n_forwards_with_same_render_pair": self.n_forwards_with_same_render_pair,
            "n_record_errors": self.n_record_errors,
            "first_record_error": self.first_record_error,
        }


#: bound on the (request, layer) entries a RhoRecord keeps — a few thousand
#: requests over ~10 steered layers fit; past it entries are counted dropped,
#: never raised on
RHO_RECORD_MAX_ENTRIES = 65536

RHO_RECORD_NOTE = (
    "OBSERVATION ONLY (SISTEER_LENS_RHO_EMIT=1): per (request, HOOK layer) at "
    "the block-apply sites, after the write — add_norm = ||applied - true_rows|| "
    "per row (the very delta the production path wrote), h_norm = ||true_rows|| "
    "per row (the COMPOSED residual stream at this hook: hidden + residual "
    "carrying every earlier steered layer's edit), rho = add_norm / h_norm. "
    "Reduced on device to the LAST row's values and the row min/median/max; "
    "converted to floats only in as_dict (no .item() on the apply path). "
    "`layer` is the hook layer (block output); a site=pre object's FIT layer "
    "is hook layer + 1 (lens_adapters._block_output_layer). Nothing on the "
    "apply path reads this back; recording cannot raise (errors are counted)."
)

_RHO_STAT_KEYS = ("add_norm_last", "h_norm_last", "rho_last",
                  "add_norm_min", "add_norm_max",
                  "h_norm_min", "h_norm_median", "h_norm_max",
                  "rho_min", "rho_median", "rho_max")


class RhoRecord:
    """A plain, bounded, append-only record of the served per-layer edit
    ratio ``rho = ||add|| / ||h||`` (:data:`RHO_RECORD_NOTE`), the
    :class:`SelectionRecord` discipline: ``observe_*`` return ``None``,
    never raise, and nothing on the apply path reads the record back.

    Device discipline: each observation is a handful of reductions over
    ``delta`` / ``true_rows`` (``.detach().float()`` copies — the production
    tensors are never written) stacked into ONE small device tensor per
    entry; the host sync happens in :meth:`as_dict`, once per snapshot,
    so an enabled record adds kernels but no per-layer stream sync."""

    def __init__(self, enabled: bool, max_entries: int = RHO_RECORD_MAX_ENTRIES):
        self.enabled = bool(enabled)
        self._max_entries = int(max_entries)
        self.reset()

    def reset(self) -> None:
        self.n_entries = 0
        self.n_entries_dropped = 0
        self.n_record_errors = 0
        self.first_record_error: Optional[str] = None
        #: (req_id, layer, requested cell, applied cell, applied scale, path,
        #:  phase, n_rows, dtype, stats) — stats an 11-vector (a per-slot
        #:  entry) or a 3-vector row view (a grouped member: one row, so
        #:  last == min == median == max)
        self._entries: List[Tuple[Any, ...]] = []

    @staticmethod
    def _norms(delta: torch.Tensor, true_rows: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = delta.detach()
        h = true_rows.detach()
        d32 = d.reshape(-1, d.shape[-1]).float()
        h32 = h.reshape(-1, h.shape[-1]).float()
        an = d32.norm(dim=-1)
        hn = h32.norm(dim=-1)
        return an, hn, an / hn

    def observe_slot(self, *, req_id: Any, layer: Any, cell_id: Any,
                     applied: Optional[ls.LoadedObject], path: Optional[str],
                     phase: Optional[str], delta: torch.Tensor,
                     true_rows: torch.Tensor) -> None:
        """One per-slot apply's rows (``[T, d]``). NEVER RAISES."""
        try:
            if len(self._entries) >= self._max_entries:
                self.n_entries_dropped += 1
                return
            an, hn, rho = self._norms(delta, true_rows)
            stats = torch.stack([an[-1], hn[-1], rho[-1],
                                 an.min(), an.max(),
                                 hn.min(), hn.median(), hn.max(),
                                 rho.min(), rho.median(), rho.max()])
            self._entries.append(self._entry(req_id, layer, cell_id, applied, path,
                                             phase, int(an.shape[0]), delta, stats))
            self.n_entries += 1
        except Exception as e:  # pragma: no cover — defence, not a path
            self.n_record_errors += 1
            if self.first_record_error is None:
                self.first_record_error = f"{type(e).__name__}: {e}"

    def observe_group(self, *, req_ids: Sequence[Any], layer: Any, cell_id: Any,
                      applied: Sequence[Optional[ls.LoadedObject]],
                      path: Optional[str], phase: Optional[str],
                      delta: torch.Tensor, true_rows: torch.Tensor) -> None:
        """One grouped apply (``[B, d]``, row i = member i's single row):
        three ``[B]`` reductions, one entry per member as a row VIEW of
        the stacked result (no per-member kernel). NEVER RAISES."""
        try:
            an, hn, rho = self._norms(delta, true_rows)
            stats3 = torch.stack([an, hn, rho], dim=-1)  # [B, 3]
            for i, rid in enumerate(req_ids):
                if len(self._entries) >= self._max_entries:
                    self.n_entries_dropped += 1
                    continue
                self._entries.append(self._entry(rid, layer, cell_id, applied[i], path,
                                                 phase, 1, delta, stats3[i]))
                self.n_entries += 1
        except Exception as e:  # pragma: no cover — defence, not a path
            self.n_record_errors += 1
            if self.first_record_error is None:
                self.first_record_error = f"{type(e).__name__}: {e}"

    @staticmethod
    def _entry(req_id, layer, cell_id, applied, path, phase, n_rows, delta, stats):
        return (str(req_id), _opt_int(layer), str(cell_id),
                None if applied is None else str(applied.cell_id),
                None if applied is None else float(applied.effective_scale),
                path, phase, int(n_rows), str(delta.dtype), stats)

    def as_dict(self) -> Dict[str, Any]:
        """A JSON-able snapshot (the RPC return): the ONE host sync per
        stats width; non-finite ratios (a zero-norm row) read ``None``."""
        eleven = [e[-1] for e in self._entries if e[-1].numel() == 11]
        three = [e[-1] for e in self._entries if e[-1].numel() == 3]
        v11 = torch.stack(eleven).cpu().tolist() if eleven else []
        v3 = torch.stack(three).cpu().tolist() if three else []
        i11 = i3 = 0
        entries: List[Dict[str, Any]] = []

        def _f(x):
            return float(x) if math.isfinite(x) else None
        for (rid, layer, want, got, got_scale, path, phase, n_rows, dtype,
             stats) in self._entries:
            if stats.numel() == 11:
                vals = v11[i11]
                i11 += 1
            else:
                a, h, r = v3[i3]
                i3 += 1
                vals = [a, h, r, a, a, h, h, h, r, r, r]
            entries.append({
                "req_id": rid, "layer": layer, "requested": want,
                "applied_cell": got, "applied_effective_scale": got_scale,
                "path": path, "phase": phase, "n_rows": n_rows,
                "dtype": dtype, **{k: _f(v) for k, v in zip(_RHO_STAT_KEYS, vals)},
            })
        return {
            "note": RHO_RECORD_NOTE,
            "enabled": self.enabled,
            "env": RHO_EMIT_ENV,
            "n_entries": self.n_entries,
            "n_entries_dropped": self.n_entries_dropped,
            "max_entries": self._max_entries,
            "n_record_errors": self.n_record_errors,
            "first_record_error": self.first_record_error,
            "entries": entries,
        }


def _op_shape_class(op: la.HookOp) -> str:
    """The kernel-shape class of a served op (the same vocabulary the gate's
    inventory uses): rank-1 vs rank-k LoRA / low-rank / affine, the gated
    gemv, the plain add, the last-token matmul."""
    if op.op == la.OP_LORA_OPROJ or op.op in (la.OP_LOWRANK_RELU_GATED,
                                              la.OP_LOWRANK_LINEAR_GATED):
        return "rank1" if op.lowrank_a.shape[0] == 1 else "rankK"
    if op.op == la.OP_AFFINE_SUBSPACE:
        return "rank1" if op.rotate.shape[0] == 1 else "rankK"
    if op.op in (la.OP_RELU_GATED_VECTOR, la.OP_LINEAR_GATED_VECTOR, la.OP_RELU_GATED_VECTOR_PAIR):
        return "gemv"        # the paired write: two gemvs of the same shape per row
    if op.op == la.OP_ADD_VECTOR:
        return "vec"
    if op.op == la.OP_PREFILL_LAST_MATMUL:
        return "dxd"
    return "other"


def shadow_key(kind: str, ops: Sequence[la.HookOp], rows: int) -> str:
    """``kind:op:shape:positions:skipbos{0,1}:phase:B<rows>`` for the ops of
    one object at one (layer, site) applied to a ``rows``-row group; several
    distinct values on an axis are joined with ``+``. Every axis is named
    from the ops, so ``ops`` must be non-empty: the gate classifies a key
    by its op token (reduction vs elementwise) and a key naming no op is
    missing evidence there, never a default class — a group whose cell
    serves NO op at the hook's (site, layer) is not keyed at all but counted
    as UNSERVED (:meth:`ShadowCompareRecord.observe_unserved`); asking for
    its key is a programming error, refused (the 2026-09-04 multigroup gate
    FAIL: ``psr:-:-:-:-:-:B6`` — the block-site psr cell's groups keyed with
    placeholder dashes at the o_proj hook another cell's LoRA installed)."""
    if not ops:
        raise LensWorkerError(
            f"shadow_key({kind!r}, ops=(), rows={rows}): a shadow key names "
            "the op(s) it buckets; a group with no op at this site is "
            "recorded as unserved, never keyed")

    def _join(vals):
        return "+".join(sorted(str(v) for v in vals))
    return ":".join([
        str(kind), _join({op.op for op in ops}), _join({_op_shape_class(op) for op in ops}),
        _join({op.positions for op in ops}),
        _join({f"skipbos{int(bool(op.skip_bos))}" for op in ops}),
        _join({op.phase for op in ops}), f"B{int(rows)}"])


#: bound on the distinct shadow keys kept (a registry's kinds x layers x
#: batch sizes is small; the cap only guards a faulty caller)
SHADOW_RECORD_MAX_KEYS = 4096


def _unequal_stats(applied: torch.Tensor, ref: torch.Tensor,
                   base: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """What the shadow record keeps about ONE unequal group (the
    ``observe`` kwargs): ``max_abs_diff``; the deviation in the per-row
    result's OWN resolution — ``max_ulps`` (lens_adapters.ulp_distance at the
    per-row value's magnitude, over elements whose per-row value is nonzero;
    None when every differing element sits on a zero; inflates where the edit
    cancels the activation, informational); ``max_abs_ref`` (the largest
    |per-row value| in the group); ``n_unequal_elems``; the CONCENTRATION of
    the differences — ``max_frac_row`` (the largest fraction of unequal
    elements in any one row; the last dim is the row) and ``frac_group``
    (unequal elements over the group's elements); and, given ``base`` (the
    UNSTEERED rows both edits were added to), ``max_ulps_edit``: the largest
    excess of |stacked edit - per-row edit| over one bf16 rounding of the
    output, in bf16 ulps of the per-row edit (lens_adapters.edit_excess_ulps
    — THE CONTRACT'S PER-ELEMENT UNIT: kernel variance reads <= ~1, a
    dropped / halved / misplaced edit reads tens to thousands wherever the
    edit exceeds a couple of output ulps) — and ``worst_elem``, WHERE that
    unit peaks (:func:`_worst_elem`: the element's values and its row's edit
    ratio / cosine; evidence, never the verdict). A shape or dtype mismatch
    between the two sides (never expected: one group, one dtype) records the
    absolute difference alone."""
    a, r = applied.detach(), ref.detach()
    diff = (a.float() - r.float()).abs()
    out: Dict[str, Any] = {"max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0}
    if a.shape != r.shape or a.dtype != r.dtype:
        return out
    ne = diff > 0
    n_ne = int(ne.sum().item())
    out["n_unequal_elems"] = n_ne
    out["max_abs_ref"] = float(r.float().abs().max().item()) if r.numel() else 0.0
    if r.numel():
        out["frac_group"] = n_ne / r.numel()
        if r.dim() >= 1 and r.shape[-1] > 0:
            rows = ne.reshape(-1, r.shape[-1]) if r.dim() >= 2 else ne.reshape(1, -1)
            out["max_frac_row"] = float((rows.sum(dim=-1).float() / rows.shape[-1]).max().item())
    nz = ne & (r != 0)
    if bool(nz.any()):
        out["max_ulps"] = float(la.ulp_distance(a, r)[nz].max().item())
    if base is not None and base.shape == r.shape and base.dtype == r.dtype and bool(ne.any()):
        b = base.detach()
        ex = la.edit_excess_ulps(a, r, b)
        out["max_ulps_edit"] = float(ex[ne].max().item())
        try:
            out["worst_elem"] = _worst_elem(a, r, b, ex, ne)
        except Exception as exc:  # pragma: no cover — evidence, never the verdict
            out["worst_elem"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _worst_elem(a: torch.Tensor, r: torch.Tensor, b: torch.Tensor, ex: torch.Tensor,
                ne: torch.Tensor) -> Dict[str, Any]:
    """WHERE the contract's per-element unit peaks in one unequal group
    (lens-grouped-scope, 2026-09-05): the element with the largest
    ``edit_excess_ulps`` — its unsteered (``base``), per-row (``ref``) and
    stacked (``applied``) values, both edits, the deviation in output ulps and
    in edit ulps — and the ROW it sits in: how many of that row's elements
    differ, the per-row edit's norm, and the stacked edit's norm relative to it
    (``row_edit_ratio`` = ||stacked edit|| / ||per-row edit||) with their cosine
    (``row_edit_cos``). Evidence for READING a tolerance violation, never the
    verdict: a gate-scalar perturbation (the fp32 reduction of a gated op landing
    a few fp32 ulps of its PARTIAL SUMS away — many ulps of a small ReLU gate)
    scales the whole row's edit uniformly, ratio 1 +- the gate's relative
    deviation and cosine 1; a dropped / halved / misplaced edit reads ratio ~0 /
    ~0.5 / a cosine well below 1. All in fp32 over the bf16 values."""
    a32, r32, b32 = a.float(), r.float(), b.float()
    last = int(r.shape[-1]) if r.dim() >= 1 and r.shape[-1] > 0 else 1
    ex2, ne2 = ex.reshape(-1, last), ne.reshape(-1, last)
    masked = torch.where(ne2, ex2, torch.full_like(ex2, -1.0))
    flat = int(masked.reshape(-1).argmax().item())
    row, col = divmod(flat, last)
    A, R, B = a32.reshape(-1, last), r32.reshape(-1, last), b32.reshape(-1, last)
    av, rv, bv = float(A[row, col].item()), float(R[row, col].item()), float(B[row, col].item())
    e_ref, e_app = R[row] - B[row], A[row] - B[row]
    n_ref, n_app = float(e_ref.norm().item()), float(e_app.norm().item())
    cos = float((e_ref @ e_app).item() / (n_ref * n_app)) if n_ref > 0 and n_app > 0 else None
    ulps_out = (float(la.ulp_distance(a, r).reshape(-1, last)[row, col].item()) if rv != 0 else None)
    return {"row": int(row), "col": int(col), "base": bv, "ref": rv, "applied": av,
            "edit_ref": rv - bv, "edit_applied": av - bv, "abs_diff": abs(av - rv),
            "ulps_out": ulps_out, "ulps_edit": float(ex2[row, col].item()),
            "row_n_unequal": int(ne2[row].sum().item()), "row_n_elems": int(last),
            "row_edit_norm": n_ref, "row_edit_ratio": ((n_app / n_ref) if n_ref > 0 else None),
            "row_edit_cos": cos, "row_out_norm": float(R[row].norm().item())}


def _gate_stats(ops: Any, rows: torch.Tensor) -> Dict[str, Any]:
    """THE GATE of a gated reduction op (``lens_adapters.GATED_REDUCTION_OPS``:
    the gated-vector pair OP_RELU_GATED_VECTOR / OP_LINEAR_GATED_VECTOR and the
    low-rank pair OP_LOWRANK_RELU_GATED / OP_LOWRANK_LINEAR_GATED), recomputed
    both ways on the group's unsteered rows (lens-grouped-scope, 2026-09-05;
    the contract's INPUT for gated ops since 2026-09-06; inside the shadow
    compare of an UNEQUAL group): per row exactly as the per-row path computes
    it (one single-row ``_matmul`` per row, the bias, the ReLU for the ReLU
    gates) and stacked as the stacked mode computes it (one ``[B, d]`` call).
    The gated-vector gate is one scalar per row (``h @ gate_w + gate_b``); the
    low-rank gate is the k-vector of latents ``h @ Aᵀ`` per row — every field
    below is over GATE ELEMENTS: rows for the vector ops, row x latent for the
    low-rank ops (``gate_n_elems`` counts them). Their difference IS the
    reduction-order variance the contract tolerates, read at its source
    instead of after it has been multiplied into a whole row of edits:
    ``gate_max_abs_diff`` (fp32), ``gate_max_rel_diff`` (relative to the larger
    of the two gates), ``gate_min_abs_at_diff`` (the smallest gate magnitude
    among elements whose gates differed — a near-threshold ReLU gate is where a
    fixed absolute perturbation becomes a large relative one),
    ``gate_max_diff_rel_sum`` — THE RULE'S NUMBER: |delta| over the reduction's
    own magnitude ``sum_i |h_i w_i|`` (kernel variance is a few 2^-24 of it, a
    wrong gate O(1); the gate bounds it at 16 * 2^-24), ``gate_n_rows_unequal``
    (gate elements that differed), ``gate_n_relu_flips`` (elements where
    exactly one side's ReLU fired) and ``gate_max_flip_rel_sum`` — THE RULE'S
    NUMBER FOR A FLIP: the fired side's magnitude over ``sum_i |h_i w_i|``,
    maxed over the flipped elements (None when nothing flipped; a flip is
    allowed only within the same 16 * 2^-24 bound: a threshold crossing by
    less than the reduction's own rounding) — plus up to four ``[per_row,
    stacked, sum|hw|]`` samples, summed / maxed over the gated ops at this
    (site, layer). Empty when no op is gated or the group has one row.
    RECOMPUTED, so equal to the hook's own gates exactly as far as cuBLAS is
    run-to-run deterministic for one shape on one device. A raise inside
    returns ``{"gate_error": ...}``, which the record counts and the gate FAILS
    as missing evidence — never read as a pass."""
    try:
        gated = []
        for op in ops:
            if op.op in la._GATED_OPS and op.gate_w is not None and op.gate_b is not None:
                gated.append(op)
            elif op.op == la.OP_RELU_GATED_VECTOR_PAIR and op.gate_w is not None and op.gate_w2 is not None:
                gated.append(op)
            elif op.op in la._LOWRANK_OPS and op.lowrank_a is not None:
                gated.append(op)
    except Exception:  # pragma: no cover — defence
        return {}
    if not gated or rows.dim() != 2 or rows.shape[0] < 2:
        return {}
    try:
        x = rows.detach().float()
        n_rows = int(x.shape[0])
        tiny = torch.finfo(torch.float32).tiny
        out: Dict[str, Any] = {
            "gate_ops": [str(op.op) for op in gated], "gate_n_rows": n_rows, "gate_n_elems": 0,
            "gate_n_rows_unequal": 0, "gate_n_relu_flips": 0, "gate_max_abs_diff": 0.0,
            "gate_max_rel_diff": 0.0, "gate_min_abs_at_diff": None, "gate_max_abs_ref": 0.0,
            "gate_max_diff_rel_sum": 0.0, "gate_max_flip_rel_sum": None, "gate_rows_unequal_sample": []}
        for op in gated:
            if op.op == la.OP_RELU_GATED_VECTOR_PAIR:
                # the paired write: TWO gated-vector gates on the same rows (pos, then neg), each
                # recomputed both ways exactly as the single gate below and compared side by side
                for w_, b_ in ((op.gate_w, op.gate_b), (op.gate_w2, op.gate_b2)):
                    w = w_.to(device=x.device, dtype=torch.float32)
                    bias = b_.to(device=x.device, dtype=torch.float32)
                    g_row = torch.cat([la._matmul(x[i:i + 1], w).reshape(1, 1) for i in range(n_rows)],
                                      dim=0) + bias
                    g_stk = la._matmul(x, w).reshape(-1, 1) + bias
                    sum_abs = la._matmul(x.abs(), w.abs()).reshape(-1, 1)
                    _gate_compare(out, torch.relu(g_row), torch.relu(g_stk), sum_abs, relu=True)
                continue
            if op.op in la._GATED_OPS:
                # the gated-vector gate: [B, d] @ [d] (+ bias) -> one scalar per row, read as [B, 1]
                w = op.gate_w.to(device=x.device, dtype=torch.float32)
                bias = op.gate_b.to(device=x.device, dtype=torch.float32)
                g_row = torch.cat([la._matmul(x[i:i + 1], w).reshape(1, 1) for i in range(n_rows)],
                                  dim=0) + bias
                g_stk = la._matmul(x, w).reshape(-1, 1) + bias
                sum_abs = la._matmul(x.abs(), w.abs()).reshape(-1, 1)
            else:
                # the low-rank gate: [B, d] @ [d, k] -> k latents per row, as the apply's _lowrank
                a = op.lowrank_a.to(device=x.device, dtype=torch.float32)          # [k, d]
                g_row = torch.cat([la._matmul(x[i:i + 1], a.t()) for i in range(n_rows)], dim=0)
                g_stk = la._matmul(x, a.t())                                         # [B, k]
                sum_abs = la._matmul(x.abs(), a.abs().t())                           # [B, k]
            if op.op in la._RELU_GATES:
                g_row, g_stk = torch.relu(g_row), torch.relu(g_stk)
            _gate_compare(out, g_row, g_stk, sum_abs, relu=op.op in la._RELU_GATES)
        out["gate_source"] = "recomputed"
        return out
    except Exception as exc:  # pragma: no cover — surfaced to the gate as missing evidence
        return {"gate_error": f"{type(exc).__name__}: {exc}"}


def _gate_compare(out: Dict[str, Any], g_row: torch.Tensor, g_stk: torch.Tensor,
                  sum_abs: torch.Tensor, *, relu: bool) -> None:
    """Accumulate the gate-deviation fields into ``out`` (the ``gate_*`` keys
    of :func:`_gate_stats`) from a per-row gate, a stacked gate and the
    reduction magnitude, all POST-ReLU for a ReLU gate, any shape (flattened:
    every element is a gate element)."""
    tiny = torch.finfo(torch.float32).tiny
    g_row, g_stk = g_row.reshape(-1).float(), g_stk.reshape(-1).float()
    sum_abs = sum_abs.reshape(-1).float()
    out["gate_n_elems"] += int(g_row.numel())
    if relu:
        # a threshold crossing: exactly one side's ReLU fired (a zero on a
        # linear gate is a value, not a crossing — not counted there)
        flip = (g_row == 0) != (g_stk == 0)
        n_flip = int(flip.sum().item())
        out["gate_n_relu_flips"] += n_flip
        if n_flip:
            fired = torch.where(g_row == 0, g_stk, g_row).abs()
            fr = float((fired[flip] / sum_abs[flip].clamp_min(tiny)).max().item())
            out["gate_max_flip_rel_sum"] = (fr if out["gate_max_flip_rel_sum"] is None
                                            else max(float(out["gate_max_flip_rel_sum"]), fr))
    diff = (g_stk - g_row).abs()
    ne = diff > 0
    mag = torch.maximum(g_row.abs(), g_stk.abs())
    out["gate_n_rows_unequal"] += int(ne.sum().item())
    out["gate_max_abs_ref"] = max(out["gate_max_abs_ref"], float(g_row.abs().max().item()))
    if bool(ne.any()):
        out["gate_max_abs_diff"] = max(out["gate_max_abs_diff"], float(diff[ne].max().item()))
        rel = diff[ne] / mag[ne].clamp_min(tiny)
        out["gate_max_rel_diff"] = max(out["gate_max_rel_diff"], float(rel.max().item()))
        m = float(mag[ne].min().item())
        out["gate_min_abs_at_diff"] = (m if out["gate_min_abs_at_diff"] is None
                                       else min(float(out["gate_min_abs_at_diff"]), m))
        rs = diff[ne] / sum_abs[ne].clamp_min(tiny)
        out["gate_max_diff_rel_sum"] = max(out["gate_max_diff_rel_sum"], float(rs.max().item()))
        idx = torch.nonzero(ne).reshape(-1)[:4].tolist()
        out["gate_rows_unequal_sample"] += [[float(g_row[i].item()), float(g_stk[i].item()),
                                            float(sum_abs[i].item())] for i in idx]


#: the gate-magnitude histogram's bucket edges, in units of the noise floor f
GATE_HIST_NEAR_F = 10.0
GATE_HIST_MID_F = 1000.0


def _gate_histogram_observed(ops: Any, per_row: Sequence["la.GateCapture"]) -> Dict[str, Any]:
    """THE GATE-MAGNITUDE HISTOGRAM (2026-09-06, measurement only — the gate
    reports it, never rules on it), over the per-row captures of ONE compared
    gated group — EQUAL or unequal (a histogram over the unequal groups alone
    is not a histogram over the served gates; the shadow compare calls this
    for every compared gated group and the record sums it apart from the
    unequal-only gate stats). For every OBSERVED per-row gate element ``g``
    (post-ReLU; one per row for the vector pair, one per row x latent for the
    low-rank pair) with its noise floor ``f = 2^-24 * sum_i |h_i w_i|`` — one
    fp32 rounding of the reduction's own magnitude, the unit the gate bound is
    stated in — the counts ``gate_hist_zero`` (g == 0), ``gate_hist_near``
    (0 < g <= 10 f), ``gate_hist_mid`` (10 f < g <= 1000 f), ``gate_hist_far``
    (g > 1000 f); the DOSE each element writes, ``|eff * g| * ||direction||``
    (``v`` for the vector pair, column j of B for latent j of the low-rank
    pair), summed over the near-threshold elements (``gate_dose_near``) and
    over all (``gate_dose_total``) with their ratio
    ``gate_dose_share_near_threshold`` (None when nothing was written); and
    the smallest nonzero and the largest ``g / f`` (``gate_g_over_f_min_
    nonzero`` / ``gate_g_over_f_max`` — the dimensionless cancellation ratio:
    how many of its own roundings a gate sits above zero). This is the number
    that answers whether the near-threshold gates the evidence run found
    (1e-5 .. 1e-4 on fitted PSR objects) matter: how much of the served dose
    rides on gates within ten roundings of the ReLU. Empty when no op is
    gated or nothing was captured; a capture mismatch is
    ``{"gate_hist_error": ...}`` (informational, never a verdict)."""
    try:
        gated = [op for op in ops if op.op in la.GATED_REDUCTION_OPS]
    except Exception:  # pragma: no cover — defence
        return {}
    if not gated or not per_row:
        return {}
    want = [str(op.op) for op in gated]
    try:
        for i, cap in enumerate(per_row):
            got = [str(e["op"]) for e in cap.entries]
            if got != want:
                return {"gate_hist_error": f"the per-row apply of group row {i} captured gates for {got}, expected {want}"}
        out: Dict[str, Any] = {"gate_hist_n_groups": 1, "gate_hist_n_rows": len(per_row),
                               "gate_hist_zero": 0, "gate_hist_near": 0, "gate_hist_mid": 0, "gate_hist_far": 0,
                               "gate_dose_near": 0.0, "gate_dose_total": 0.0, "gate_dose_share_near_threshold": None,
                               "gate_g_over_f_min_nonzero": None, "gate_g_over_f_max": None}
        for k, op in enumerate(gated):
            first = per_row[0].entries[k]
            g_row = torch.cat([cap.entries[k]["gate"].float() for cap in per_row], dim=0)
            sum_abs = torch.cat([cap.entries[k]["sum_abs"].float() for cap in per_row], dim=0)
            _gate_histogram(out, g_row, dict(first, sum_abs=sum_abs))
        return out
    except Exception as exc:  # pragma: no cover — informational
        return {"gate_hist_error": f"{type(exc).__name__}: {exc}"}


def _gate_histogram(out: Dict[str, Any], g_row: torch.Tensor, entry: Mapping[str, Any]) -> None:
    """Accumulate the gate-magnitude histogram (``_gate_stats_observed``
    docstring) into ``out`` from ONE op's observed per-row gate ``g_row``
    (post-ReLU, ``[rows]`` or ``[rows, k]``) and its capture ``entry``
    (``sum_abs``, ``eff``, ``vector`` / ``write``)."""
    g = g_row.detach().float().abs()
    f = (2.0 ** -24) * entry["sum_abs"].detach().float().reshape(g.shape)
    if entry["kind"] == "vector":
        dir_norm = entry["vector"].detach().float().norm().reshape(1)                    # ||v||
        g2, f2 = g.reshape(-1, 1), f.reshape(-1, 1)
    else:
        dir_norm = entry["write"].detach().float().norm(dim=0)                           # ||B[:, j]|| per latent
        g2, f2 = g.reshape(-1, dir_norm.numel()), f.reshape(-1, dir_norm.numel())
    zero = g2 == 0
    ratio = torch.where(zero, torch.zeros_like(g2), g2 / f2.clamp_min(torch.finfo(torch.float32).tiny))
    near = (~zero) & (ratio <= GATE_HIST_NEAR_F)
    mid = (ratio > GATE_HIST_NEAR_F) & (ratio <= GATE_HIST_MID_F)
    far = ratio > GATE_HIST_MID_F
    out["gate_hist_zero"] += int(zero.sum().item())
    out["gate_hist_near"] += int(near.sum().item())
    out["gate_hist_mid"] += int(mid.sum().item())
    out["gate_hist_far"] += int(far.sum().item())
    dose = (abs(float(entry["eff"])) * g2) * dir_norm.reshape(1, -1)
    out["gate_dose_near"] += float(dose[near].sum().item())
    out["gate_dose_total"] += float(dose.sum().item())
    out["gate_dose_share_near_threshold"] = ((out["gate_dose_near"] / out["gate_dose_total"])
                                             if out["gate_dose_total"] > 0 else None)
    if bool((~zero).any()):
        mn, mx = float(ratio[~zero].min().item()), float(ratio[~zero].max().item())
        out["gate_g_over_f_min_nonzero"] = (mn if out["gate_g_over_f_min_nonzero"] is None
                                            else min(float(out["gate_g_over_f_min_nonzero"]), mn))
        out["gate_g_over_f_max"] = (mx if out["gate_g_over_f_max"] is None
                                    else max(float(out["gate_g_over_f_max"]), mx))


def _gate_stats_observed(ops: Any, grouped: Optional["la.GateCapture"],
                         per_row: Sequence["la.GateCapture"], applied: torch.Tensor,
                         ref: torch.Tensor, base: torch.Tensor) -> Dict[str, Any]:
    """THE GATED CONTRACT'S EVIDENCE (2026-09-06): the gates the two
    paths ACTUALLY USED on one unequal group — ``grouped`` is the
    :class:`la.GateCapture` armed around the grouped (stacked) apply that
    produced ``applied``; ``per_row`` the captures armed around the per-row
    reference applies, in group row order, that produced ``ref`` — compared
    gate element by gate element exactly as :func:`_gate_stats` compares its
    recomputed pair (the same ``gate_*`` fields; ``gate_source`` =
    ``'observed'``), PLUS the identity that binds the observed gate delta to
    the observed output delta: for every element, ``applied - ref`` must equal
    the sum over the group's gated ops of ``eff * (g_stacked - g_per_row) *
    v`` (``eff * (z_stacked - z_per_row) @ Bᵀ`` for the low-rank pair), up to
    the bf16 roundings of the two edits and the two sums — recorded as
    ``gate_edit_consistency_max_ulps`` = max over elements of the residual in
    units of ``ulp(|ref|) + ulp(|ref - base|)`` (kernel variance reads <= ~1;
    the gate bounds it at 4), with the worst element beside it. A misplaced
    gate reads O(1) of sum|hw| in the gate delta; a wrong WRITE (a transposed
    operand, another row's direction) leaves the gate delta small and breaks
    the identity — either fails the gate where the concentration bounds,
    blind to a sparse direction, would not. AND EXACTNESS WHERE THE GATES
    AGREE: after an equal gate a gated-VECTOR op is elementwise — the
    same fp32 products, the same mask, the same cast, the same add — so on a
    row whose observed gate delta is exactly zero for every vector op that
    writes it, and on a row every gated op masks off, the two outputs must be
    bit-identical: ``gate_exact_rows_n`` counts such rows and
    ``gate_exact_rows_unequal_elems`` the unequal elements in them (any is a
    wrong write; the gate FAILs it, where the 4-ulp identity tolerance alone
    would let a one-ulp sparse write through). A LOW-RANK write ``z @ Bᵀ`` is
    itself a k-term reduction, so a zero latent delta does NOT promise an exact
    write — those rows are judged by the LINEAR unit instead
    (``gate_lowrank_zero_delta_max_ulps_edit``: the largest ``edit_excess_ulps``
    over their unequal elements, None when the group has no such row; the gate
    bounds it at the linear 4). Any mismatch between what the two paths
    captured (a missing capture, a different op list, different row counts or
    effective scales) is ``{"gate_error": ...}``: missing evidence, which the
    gate FAILs. Empty when no op is gated or the group has one row.

    EXACT WRITE RECONSTRUCTION for the gated-VECTOR pair:
    a nonzero but legitimate gate delta exempts a row from the exactness rule
    above, and the 4-ulp identity envelope then hides an unrelated sparse
    wrong write on that row (one ulp written where the direction is exactly
    ZERO reads a residual of 1). So when every op on the group is a gated-
    vector op, each side's output is REBUILT from that side's OBSERVED gates
    with the apply's own write (``lens_adapters.gated_vector_write``, one code
    path: ``base + ((eff g).unsqueeze(-1) v [* mask]).to(dtype)``, replayed
    in op order) — ``expected_stacked`` from the grouped capture and
    ``expected_per_row`` from the per-row captures — and ``applied`` /
    ``ref`` must equal them BIT-FOR-BIT: ``gate_write_recon_elems`` elements
    compared (both sides), ``gate_write_recon_mismatch_elems`` the elements
    that are not the reconstruction — any is a wrong write, whatever the gate
    did (the exactness-at-zero rule stays as the vector pair's special case; a
    low-rank row that RECEIVES its edit is never held exact, identical latents
    or not — bounded, below). A group with
    any other op (an elementwise or low-rank op interleaved — no registered
    kind has one) cannot be rebuilt from the captures alone:
    ``gate_write_recon_mismatch_elems`` is None with ``gate_write_recon_
    skipped`` naming why, and the gate reads that as missing evidence for a
    vector-gated key. The LOW-RANK pair is not reconstructed — its write
    ``z @ Bᵀ`` is a k-term reduction whose kernel variance is exactly what
    the linear unit tolerates — so there the identity residual (<= 4) and the
    zero-delta rows' linear unit remain the rule, as stated in the gate."""
    try:
        gated = [op for op in ops if op.op in la.GATED_REDUCTION_OPS]
    except Exception:  # pragma: no cover — defence
        return {}
    if not gated or ref.dim() != 2 or ref.shape[0] < 2:
        return {}
    want = [str(op.op) for op in gated]
    if grouped is None:
        return {"gate_error": "no gate capture from the grouped apply (the shadow compare did not arm one)"}
    try:
        got = [str(e["op"]) for e in grouped.entries]
        if got != want:
            return {"gate_error": f"the grouped apply captured gates for {got}, expected {want}"}
        for i, cap in enumerate(per_row):
            got_i = [str(e["op"]) for e in cap.entries]
            if got_i != want:
                return {"gate_error": f"the per-row apply of group row {i} captured gates for {got_i}, "
                                      f"expected {want}"}
        n_rows, d = int(ref.shape[0]), int(ref.shape[-1])
        out: Dict[str, Any] = {
            "gate_ops": want, "gate_source": "observed", "gate_n_rows": n_rows, "gate_n_slots": len(per_row),
            "gate_n_elems": 0, "gate_n_rows_unequal": 0, "gate_n_relu_flips": 0, "gate_max_abs_diff": 0.0,
            "gate_max_rel_diff": 0.0, "gate_min_abs_at_diff": None, "gate_max_abs_ref": 0.0,
            "gate_max_diff_rel_sum": 0.0, "gate_max_flip_rel_sum": None, "gate_rows_unequal_sample": [],
            "gate_edit_consistency_max_ulps": 0.0, "gate_edit_consistency_worst": None,
            "gate_exact_rows_n": 0, "gate_exact_rows_unequal_elems": 0,
            "gate_lowrank_zero_delta_max_ulps_edit": None,
            "gate_write_recon_elems": 0, "gate_write_recon_mismatch_elems": None, "gate_write_recon_skipped": None}
        a32 = applied.detach().float().reshape(-1, d)
        r32 = ref.detach().float().reshape(-1, d)
        b32 = base.detach().float().reshape(-1, d)
        predicted = torch.zeros_like(r32)
        must_exact = torch.ones(n_rows, dtype=torch.bool, device=r32.device)     # every writing gate agreed / masked
        lowrank_zero = torch.zeros(n_rows, dtype=torch.bool, device=r32.device)  # a low-rank write with dz == 0
        for k, op in enumerate(gated):
            ge = grouped.entries[k]
            g_stk = ge["gate"].float()
            g_row = torch.cat([cap.entries[k]["gate"].float() for cap in per_row], dim=0)
            if g_row.shape != g_stk.shape:
                return {"gate_error": f"{op.op}: the grouped gate is {tuple(g_stk.shape)} but the per-row gates "
                                      f"concatenate to {tuple(g_row.shape)}"}
            effs = {float(ge["eff"])} | {float(cap.entries[k]["eff"]) for cap in per_row}
            if len(effs) != 1:
                return {"gate_error": f"{op.op}: the two paths applied different effective scales {sorted(effs)}"}
            eff = float(ge["eff"])
            _gate_compare(out, g_row, g_stk, ge["sum_abs"], relu=op.op in la._RELU_GATES)
            dg = g_stk - g_row
            if ge["kind"] == "vector":
                dd = (eff * dg).reshape(-1, 1) * ge["vector"].detach().float().reshape(1, -1)
            elif ge["kind"] == "pair":
                # the paired write: two gate columns [B, 2] against the (pos, neg) directions [2, d]
                dg2 = dg.reshape(n_rows, 2)
                vv = ge["vector"].detach().float()
                dd = ((eff * dg2[:, 0:1]) * vv[0].reshape(1, -1)
                      - (eff * dg2[:, 1:2]) * vv[1].reshape(1, -1))
            else:
                dd = eff * la._matmul(dg.reshape(n_rows, -1), ge["write"].detach().float().t())
            if ge["mask"] is not None:
                dd = dd * ge["mask"].to(dd.dtype).reshape(-1, 1)
            predicted = predicted + dd
            receives = (torch.ones(n_rows, dtype=torch.bool, device=r32.device) if ge["mask"] is None
                        else ge["mask"].reshape(-1).to(r32.device) != 0)
            row_zero = (dg.reshape(n_rows, -1) == 0).all(dim=1)
            if ge["kind"] in ("vector", "pair"):
                must_exact &= (~receives) | row_zero
            else:
                must_exact &= ~receives
                lowrank_zero |= receives & row_zero
        lowrank_zero &= ~must_exact
        # EXACT RECONSTRUCTION of both sides from their observed gates (vector-gated groups only —
        # the single-gate pair AND the paired write, whose write is two gated vectors)
        others = [str(op.op) for op in ops
                  if op.op not in la._GATED_OPS and op.op != la.OP_RELU_GATED_VECTOR_PAIR]
        if others:
            out["gate_write_recon_skipped"] = (f"the group also applies {others}: the outputs cannot be rebuilt "
                                               "from the gate captures alone")
        else:
            exp_stk = base.detach().reshape(-1, d)
            exp_row = base.detach().reshape(-1, d)
            for k, op in enumerate(gated):
                ge = grouped.entries[k]
                mask = None if ge["mask"] is None else ge["mask"].reshape(-1)
                if ge["kind"] == "pair":
                    # the paired write: both gates per row, through the apply's own write helper
                    g_stk2 = ge["gate"].reshape(-1, 2)
                    g_row2 = torch.cat([cap.entries[k]["gate"].reshape(-1, 2) for cap in per_row], dim=0)
                    vv = ge["vector"]
                    exp_stk = la.paired_gated_vector_write(exp_stk, g_stk2[:, 0], g_stk2[:, 1], float(ge["eff"]),
                                                           vv[0], vv[1], mask, ref.dtype)
                    exp_row = la.paired_gated_vector_write(exp_row, g_row2[:, 0], g_row2[:, 1], float(ge["eff"]),
                                                           vv[0], vv[1], mask, ref.dtype)
                    continue
                g_stk = ge["gate"].reshape(-1)
                g_row = torch.cat([cap.entries[k]["gate"].reshape(-1) for cap in per_row], dim=0)
                exp_stk = la.gated_vector_write(exp_stk, g_stk, float(ge["eff"]), ge["vector"], mask, ref.dtype)
                exp_row = la.gated_vector_write(exp_row, g_row, float(ge["eff"]), ge["vector"], mask, ref.dtype)
            a_ = applied.detach().reshape(-1, d)
            r_ = ref.detach().reshape(-1, d)
            out["gate_write_recon_elems"] = 2 * int(r_.numel())
            out["gate_write_recon_mismatch_elems"] = int((a_ != exp_stk).sum().item()) + int((r_ != exp_row).sum().item())
        ne = a32 != r32
        out["gate_exact_rows_n"] = int(must_exact.sum().item())
        out["gate_exact_rows_unequal_elems"] = int(ne[must_exact].sum().item()) if out["gate_exact_rows_n"] else 0
        if bool(lowrank_zero.any()):
            ne_lz = ne[lowrank_zero]
            if bool(ne_lz.any()):
                ex = la.edit_excess_ulps(applied.detach().reshape(-1, d), ref.detach().reshape(-1, d),
                                         base.detach().reshape(-1, d))
                out["gate_lowrank_zero_delta_max_ulps_edit"] = float(ex[lowrank_zero][ne_lz].max().item())
            else:
                out["gate_lowrank_zero_delta_max_ulps_edit"] = 0.0
        delta = a32 - r32
        resid = (delta - predicted).abs()
        unit = la.ulp_spacing(r32, ref.dtype) + la.ulp_spacing(r32 - b32, ref.dtype)
        cons = resid / unit
        flat = int(cons.reshape(-1).argmax().item())
        row, col = divmod(flat, d)
        out["gate_edit_consistency_max_ulps"] = float(cons[row, col].item())
        out["gate_edit_consistency_worst"] = {
            "row": int(row), "col": int(col), "delta_out": float(delta[row, col].item()),
            "predicted": float(predicted[row, col].item()), "residual": float(resid[row, col].item()),
            "base": float(b32[row, col].item()), "ref": float(r32[row, col].item())}
        return out
    except Exception as exc:  # pragma: no cover — surfaced to the gate as missing evidence
        return {"gate_error": f"{type(exc).__name__}: {exc}"}


class ShadowCompareRecord:
    """The in-worker shadow compare's bookkeeping (:data:`SHADOW_COMPARE_ENV`).
    ``observe`` never raises; ``as_dict`` is the ``sisteer_shadow_compare``
    payload. Per key (:func:`shadow_key`): groups compared, equal, unequal,
    the max |diff| seen and the first unequal group (layer, cell, rows, max
    |diff|, note); overall the same plus shadow errors (a per-row apply that
    raised where the grouped apply did not — itself a discrepancy) and the
    per-layer-call cell coverage the multigroup gate needs: how many layer
    calls compared groups of >= 2 distinct cells (a layer call is inside one
    forward, so that is >= 2 distinct cells in one forward)."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self.reset()

    def reset(self) -> None:
        self.n_groups_compared = 0
        self.n_equal = 0
        self.n_unequal = 0
        self.n_unmodified_groups = 0
        self.n_shadow_errors = 0
        self.first_unequal: Optional[Dict[str, Any]] = None
        self.first_shadow_error: Optional[Dict[str, Any]] = None
        self.max_abs_diff = 0.0
        #: the deviation in the activation's OWN resolution (lens-grouped-scope,
        #: 2026-09-04): the largest |diff| in ulps of the per-row result's
        #: dtype at that element's magnitude (lens_adapters.ulp_distance, over
        #: elements whose reference is nonzero), the largest |reference| in an
        #: unequal group, and how many elements differed — so a difference can
        #: be read as "n ulps of a value of size m", not only as an absolute
        self.max_ulps = 0.0
        self.max_abs_ref = 0.0
        self.n_unequal_elems = 0
        #: the fields the gate's REDUCTION-OP TOLERANCE reads (contract
        #: change, 2026-09-04): every element of every counted comparison
        #: (equal groups included — the denominator of the case-wide
        #: unequal fraction); the largest excess of |stacked edit -
        #: per-row edit| over one output rounding, in bf16 ulps of the
        #: per-row edit (lens_adapters.edit_excess_ulps; ``_unequal_stats``
        #: ``max_ulps_edit``); and the CONCENTRATION maxima — the largest
        #: unequal fraction of any one row and of any one group — which catch
        #: a whole row's edit dropped, halved or misplaced where the edit is
        #: within one ulp of the output and the case-wide fraction would hide it
        self.n_elems_compared = 0
        self.max_ulps_edit = 0.0
        self.max_frac_row = 0.0
        self.max_frac_group = 0.0
        #: WHERE the unit peaks (2026-09-05): the unequal-group record with the
        #: largest finite ``max_ulps_edit`` (its ``worst_elem`` and ``gate``
        #: sub-records are the evidence for reading a violation) — beside
        #: ``first_unequal``, which stays the first; and the gate-scalar
        #: summary over every unequal gated group (``_gate_stats``).
        self.worst_unequal: Optional[Dict[str, Any]] = None
        self.gate_max_abs_diff = 0.0
        self.gate_max_rel_diff = 0.0
        self.gate_min_abs_at_diff: Optional[float] = None
        self.gate_n_rows_unequal = 0
        self.gate_n_relu_flips = 0
        #: THE GATED-OP RULE'S INPUTS (contract change, 2026-09-06): the largest
        #: |g_stacked - g_per_row| / sum|h_i w_i| over every unequal gated group
        #: (the gate bounds it at 16 * 2^-24), the largest fired-side /
        #: sum|h_i w_i| over every ReLU flip (None when nothing flipped; the
        #: same bound), how many unequal groups CARRIED gate stats (the gate
        #: requires every unequal gated group to — a gated key with fewer is
        #: missing evidence) and how many gate recomputations raised
        #: (``gate_error``; any is a FAIL)
        self.gate_max_diff_rel_sum = 0.0
        self.gate_max_flip_rel_sum: Optional[float] = None
        self.gate_n_groups_with_stats = 0
        self.gate_n_errors = 0
        self.first_gate_error: Optional[Dict[str, Any]] = None
        #: the gate-edit identity's largest residual (``_gate_stats_observed``
        #: ``gate_edit_consistency_max_ulps``; None until an observed gated
        #: group is recorded — the gate bounds it at 4 and reads None as missing)
        self.gate_edit_consistency_max_ulps: Optional[float] = None
        #: PROVENANCE: how many unequal groups carried gate stats whose
        #: ``gate_source`` is ``'observed'`` — the gate requires this to
        #: equal a gated key's ``n_unequal`` (a recomputed or unsourced gate
        #: record is missing evidence); and EXACTNESS: rows whose writing
        #: gates agreed / were masked off, the unequal elements in them
        #: (any is a wrong write), and the low-rank zero-delta rows' linear unit
        self.gate_n_groups_observed = 0
        self.gate_exact_rows_n = 0
        self.gate_exact_rows_unequal_elems = 0
        self.gate_lowrank_zero_delta_max_ulps_edit: Optional[float] = None
        #: WRITE RECONSTRUCTION: unequal vector-gated groups whose two
        #: outputs were rebuilt bit-for-bit from their observed gates,
        #: and the elements that were not the reconstruction (any is a FAIL)
        self.gate_n_groups_recon = 0
        self.gate_write_recon_mismatch_elems = 0
        #: the gate-magnitude histogram (measurement only; ``_gate_histogram_
        #: observed``) over EVERY compared gated group, equal or unequal —
        #: aggregated apart from the unequal-only gate stats: bucket counts summed,
        #: the near-threshold and total dose summed (their ratio exposed as
        #: gate_dose_share_near_threshold), min nonzero / max g/f, and how many
        #: groups contributed
        self.gate_hist = {"gate_hist_n_groups": 0, "gate_hist_zero": 0, "gate_hist_near": 0,
                          "gate_hist_mid": 0, "gate_hist_far": 0}
        self.gate_dose_near = 0.0
        self.gate_dose_total = 0.0
        self.gate_g_over_f_min_nonzero: Optional[float] = None
        self.gate_g_over_f_max: Optional[float] = None
        #: unequal groups whose two sides disagreed on WHETHER the rows were
        #: modified at all (the ``note``) — a routing / masking defect, never
        #: kernel variance; the gate fails any count > 0 in either mode
        self.n_flag_disagreements = 0
        #: groups whose cell serves NO op at the hook's (site, layer) — the
        #: hook fires there for ANOTHER cell's op (``site_layers`` installs
        #: the registry-wide union of sites x layers, and a forward's groups
        #: are built once for every hook), so both paths return the rows
        #: untouched by construction: nothing to compare, no op to key.
        #: Counted here per ``kind@site`` instead of entering ``by_key`` under
        #: an op-less key the gate cannot classify (the 2026-09-04 multigroup
        #: FAIL). A side that DID modify such rows is a routing defect:
        #: ``n_unserved_modified`` + one ``n_flag_disagreements`` each, which
        #: the gate fails in every mode.
        self.n_unserved_groups = 0
        self.unserved: Dict[str, int] = {}
        self.first_unserved: Optional[Dict[str, Any]] = None
        self.n_unserved_modified = 0
        self.first_unserved_modified: Optional[Dict[str, Any]] = None
        self.by_key: Dict[str, Dict[str, Any]] = {}
        self.n_keys_dropped = 0
        self.n_layer_calls = 0
        self.n_layer_calls_with_ge2_cells_compared = 0
        self.max_distinct_cells_compared_in_one_layer_call = 0
        self.cells_compared: List[str] = []
        self.n_record_errors = 0
        self.first_record_error: Optional[str] = None

    def _entry(self, key: str) -> Optional[Dict[str, Any]]:
        e = self.by_key.get(key)
        if e is None:
            if len(self.by_key) >= SHADOW_RECORD_MAX_KEYS:
                self.n_keys_dropped += 1
                return None
            e = self.by_key[key] = {"n_compared": 0, "n_equal": 0, "n_unequal": 0,
                                    "n_unmodified": 0, "n_shadow_errors": 0,
                                    "max_abs_diff": 0.0, "max_ulps": 0.0, "max_abs_ref": 0.0,
                                    "n_unequal_elems": 0, "n_elems_compared": 0,
                                    "max_ulps_edit": 0.0, "max_frac_row": 0.0, "max_frac_group": 0.0,
                                    "n_flag_disagreements": 0, "first_unequal": None,
                                    "worst_unequal": None, "gate_max_rel_diff": 0.0,
                                    "gate_n_rows_unequal": 0, "gate_n_relu_flips": 0,
                                    "gate_max_diff_rel_sum": 0.0, "gate_max_flip_rel_sum": None,
                                    "gate_n_groups_with_stats": 0, "gate_n_errors": 0,
                                    "gate_edit_consistency_max_ulps": None, "gate_n_groups_observed": 0,
                                    "gate_exact_rows_n": 0, "gate_exact_rows_unequal_elems": 0,
                                    "gate_lowrank_zero_delta_max_ulps_edit": None,
                                    "gate_n_groups_recon": 0, "gate_write_recon_mismatch_elems": 0,
                                    "gate_hist_n_groups": 0, "gate_hist_zero": 0, "gate_hist_near": 0,
                                    "gate_hist_mid": 0, "gate_hist_far": 0,
                                    "gate_dose_near": 0.0, "gate_dose_total": 0.0,
                                    "gate_dose_share_near_threshold": None,
                                    "gate_g_over_f_min_nonzero": None, "gate_g_over_f_max": None}
        return e

    def observe(self, *, key: str, layer: Any, cell: Any, rows: Any, site: Any,
                equal: Optional[bool], max_abs_diff: Optional[float] = None, note: str = "",
                max_ulps: Optional[float] = None, max_abs_ref: Optional[float] = None,
                n_unequal_elems: Optional[int] = None, n_elems: Optional[int] = None,
                max_ulps_edit: Optional[float] = None, max_frac_row: Optional[float] = None,
                frac_group: Optional[float] = None, worst_elem: Optional[Mapping[str, Any]] = None,
                gate: Optional[Mapping[str, Any]] = None,
                gate_hist: Optional[Mapping[str, Any]] = None) -> None:
        """One shadow-compared group. ``equal`` None = neither side modified the
        rows (nothing to compare; counted as unmodified). ``n_elems`` (the
        group's element count) is added to ``n_elems_compared`` for every
        counted comparison, equal or not; the ulp-scale fields are optional
        (an unequal group recorded without them keeps the 0 / None the record
        started with). ``gate_hist`` (``_gate_histogram_observed``) is
        aggregated for EVERY counted comparison, equal or unequal, apart from
        the unequal-only ``gate`` stats."""
        try:
            e = self._entry(str(key))
            if equal is None:
                self.n_unmodified_groups += 1
                if e is not None:
                    e["n_unmodified"] += 1
                return
            self.n_groups_compared += 1
            n_el = _opt_int(n_elems)
            if n_el is not None:
                self.n_elems_compared += n_el
            if e is not None:
                e["n_compared"] += 1
                if n_el is not None:
                    e["n_elems_compared"] += n_el
            if gate_hist and "gate_hist_error" not in gate_hist:
                self._add_gate_hist(self.gate_hist, gate_hist)
                if e is not None:
                    self._add_gate_hist(e, gate_hist)
            if equal:
                self.n_equal += 1
                if e is not None:
                    e["n_equal"] += 1
                return
            self.n_unequal += 1
            mad = float(max_abs_diff) if max_abs_diff is not None else float("nan")
            ulps = float(max_ulps) if max_ulps is not None else float("nan")
            ref_mag = float(max_abs_ref) if max_abs_ref is not None else float("nan")
            n_ne = _opt_int(n_unequal_elems)
            ulps_edit = float(max_ulps_edit) if max_ulps_edit is not None else float("nan")
            fr_row = float(max_frac_row) if max_frac_row is not None else float("nan")
            fr_grp = float(frac_group) if frac_group is not None else float("nan")
            rec = {"key": str(key), "layer": _opt_int(layer), "cell": str(cell),
                   "site": str(site), "rows": _opt_int(rows), "max_abs_diff": mad,
                   "max_ulps": (ulps if math.isfinite(ulps) else None),
                   "max_abs_ref": (ref_mag if math.isfinite(ref_mag) else None),
                   "n_unequal_elems": n_ne, "n_elems": _opt_int(n_elems),
                   "max_ulps_edit": (ulps_edit if math.isfinite(ulps_edit) else None),
                   "max_frac_row": (fr_row if math.isfinite(fr_row) else None),
                   "frac_group": (fr_grp if math.isfinite(fr_grp) else None),
                   "worst_elem": (dict(worst_elem) if worst_elem else None),
                   "gate": (dict(gate) if gate else None),
                   "note": str(note)}
            if math.isfinite(mad):
                self.max_abs_diff = max(self.max_abs_diff, mad)
            if math.isfinite(ulps):
                self.max_ulps = max(self.max_ulps, ulps)
            if math.isfinite(ref_mag):
                self.max_abs_ref = max(self.max_abs_ref, ref_mag)
            if math.isfinite(ulps_edit):
                self.max_ulps_edit = max(self.max_ulps_edit, ulps_edit)
            if math.isfinite(fr_row):
                self.max_frac_row = max(self.max_frac_row, fr_row)
            if math.isfinite(fr_grp):
                self.max_frac_group = max(self.max_frac_group, fr_grp)
            if n_ne is not None:
                self.n_unequal_elems += n_ne
            if note:
                self.n_flag_disagreements += 1
            if self.first_unequal is None:
                self.first_unequal = rec
            if math.isfinite(ulps_edit) and (
                    self.worst_unequal is None
                    or ulps_edit > float(self.worst_unequal.get("max_ulps_edit") or 0.0)):
                self.worst_unequal = rec
            gate_err = bool(gate) and "gate_error" in gate
            if gate_err:
                self.gate_n_errors += 1
                if self.first_gate_error is None:
                    self.first_gate_error = {"key": str(key), "layer": _opt_int(layer), "cell": str(cell),
                                             "error": str(gate["gate_error"])}
            elif gate:
                self.gate_n_groups_with_stats += 1
                self.gate_n_rows_unequal += int(gate.get("gate_n_rows_unequal") or 0)
                self.gate_n_relu_flips += int(gate.get("gate_n_relu_flips") or 0)
                if gate.get("gate_max_abs_diff") is not None:
                    self.gate_max_abs_diff = max(self.gate_max_abs_diff, float(gate["gate_max_abs_diff"]))
                if gate.get("gate_max_rel_diff") is not None:
                    self.gate_max_rel_diff = max(self.gate_max_rel_diff, float(gate["gate_max_rel_diff"]))
                if gate.get("gate_min_abs_at_diff") is not None:
                    gm = float(gate["gate_min_abs_at_diff"])
                    self.gate_min_abs_at_diff = (gm if self.gate_min_abs_at_diff is None
                                                 else min(self.gate_min_abs_at_diff, gm))
                if gate.get("gate_max_diff_rel_sum") is not None:
                    self.gate_max_diff_rel_sum = max(self.gate_max_diff_rel_sum,
                                                     float(gate["gate_max_diff_rel_sum"]))
                if gate.get("gate_max_flip_rel_sum") is not None:
                    fr = float(gate["gate_max_flip_rel_sum"])
                    self.gate_max_flip_rel_sum = (fr if self.gate_max_flip_rel_sum is None
                                                  else max(self.gate_max_flip_rel_sum, fr))
                if gate.get("gate_edit_consistency_max_ulps") is not None:
                    cu = float(gate["gate_edit_consistency_max_ulps"])
                    self.gate_edit_consistency_max_ulps = (cu if self.gate_edit_consistency_max_ulps is None
                                                           else max(self.gate_edit_consistency_max_ulps, cu))
                if gate.get("gate_source") == "observed":
                    self.gate_n_groups_observed += 1
                self.gate_exact_rows_n += int(gate.get("gate_exact_rows_n") or 0)
                self.gate_exact_rows_unequal_elems += int(gate.get("gate_exact_rows_unequal_elems") or 0)
                if gate.get("gate_lowrank_zero_delta_max_ulps_edit") is not None:
                    lz = float(gate["gate_lowrank_zero_delta_max_ulps_edit"])
                    self.gate_lowrank_zero_delta_max_ulps_edit = (
                        lz if self.gate_lowrank_zero_delta_max_ulps_edit is None
                        else max(self.gate_lowrank_zero_delta_max_ulps_edit, lz))
                if gate.get("gate_write_recon_mismatch_elems") is not None:
                    self.gate_n_groups_recon += 1
                    self.gate_write_recon_mismatch_elems += int(gate["gate_write_recon_mismatch_elems"])
            if e is not None:
                e["n_unequal"] += 1
                if note:
                    e["n_flag_disagreements"] += 1
                if math.isfinite(mad):
                    e["max_abs_diff"] = max(float(e["max_abs_diff"]), mad)
                if math.isfinite(ulps):
                    e["max_ulps"] = max(float(e["max_ulps"]), ulps)
                if math.isfinite(ref_mag):
                    e["max_abs_ref"] = max(float(e["max_abs_ref"]), ref_mag)
                if math.isfinite(ulps_edit):
                    e["max_ulps_edit"] = max(float(e["max_ulps_edit"]), ulps_edit)
                if math.isfinite(fr_row):
                    e["max_frac_row"] = max(float(e["max_frac_row"]), fr_row)
                if math.isfinite(fr_grp):
                    e["max_frac_group"] = max(float(e["max_frac_group"]), fr_grp)
                if n_ne is not None:
                    e["n_unequal_elems"] += n_ne
                if e["first_unequal"] is None:
                    e["first_unequal"] = rec
                if math.isfinite(ulps_edit) and (
                        e.get("worst_unequal") is None
                        or ulps_edit > float(e["worst_unequal"].get("max_ulps_edit") or 0.0)):
                    e["worst_unequal"] = rec
                if gate_err:
                    e["gate_n_errors"] = int(e.get("gate_n_errors") or 0) + 1
                elif gate:
                    e["gate_n_groups_with_stats"] = int(e.get("gate_n_groups_with_stats") or 0) + 1
                    e["gate_n_rows_unequal"] = int(e.get("gate_n_rows_unequal") or 0) + int(
                        gate.get("gate_n_rows_unequal") or 0)
                    e["gate_n_relu_flips"] = int(e.get("gate_n_relu_flips") or 0) + int(
                        gate.get("gate_n_relu_flips") or 0)
                    if gate.get("gate_max_rel_diff") is not None:
                        e["gate_max_rel_diff"] = max(float(e.get("gate_max_rel_diff") or 0.0),
                                                     float(gate["gate_max_rel_diff"]))
                    if gate.get("gate_max_diff_rel_sum") is not None:
                        e["gate_max_diff_rel_sum"] = max(float(e.get("gate_max_diff_rel_sum") or 0.0),
                                                         float(gate["gate_max_diff_rel_sum"]))
                    if gate.get("gate_max_flip_rel_sum") is not None:
                        fr = float(gate["gate_max_flip_rel_sum"])
                        prev = e.get("gate_max_flip_rel_sum")
                        e["gate_max_flip_rel_sum"] = fr if prev is None else max(float(prev), fr)
                    if gate.get("gate_edit_consistency_max_ulps") is not None:
                        cu = float(gate["gate_edit_consistency_max_ulps"])
                        prev = e.get("gate_edit_consistency_max_ulps")
                        e["gate_edit_consistency_max_ulps"] = cu if prev is None else max(float(prev), cu)
                    if gate.get("gate_source") == "observed":
                        e["gate_n_groups_observed"] = int(e.get("gate_n_groups_observed") or 0) + 1
                    e["gate_exact_rows_n"] = int(e.get("gate_exact_rows_n") or 0) + int(gate.get("gate_exact_rows_n") or 0)
                    e["gate_exact_rows_unequal_elems"] = int(e.get("gate_exact_rows_unequal_elems") or 0) + int(
                        gate.get("gate_exact_rows_unequal_elems") or 0)
                    if gate.get("gate_lowrank_zero_delta_max_ulps_edit") is not None:
                        lz = float(gate["gate_lowrank_zero_delta_max_ulps_edit"])
                        prev = e.get("gate_lowrank_zero_delta_max_ulps_edit")
                        e["gate_lowrank_zero_delta_max_ulps_edit"] = lz if prev is None else max(float(prev), lz)
                    if gate.get("gate_write_recon_mismatch_elems") is not None:
                        e["gate_n_groups_recon"] = int(e.get("gate_n_groups_recon") or 0) + 1
                        e["gate_write_recon_mismatch_elems"] = int(e.get("gate_write_recon_mismatch_elems") or 0) + int(
                            gate["gate_write_recon_mismatch_elems"])
        except Exception as ex:  # pragma: no cover — defence, not a path
            self.n_record_errors += 1
            if self.first_record_error is None:
                self.first_record_error = f"{type(ex).__name__}: {ex}"

    def _add_gate_hist(self, into: Any, gate_hist: Mapping[str, Any]) -> None:
        """Sum one compared gated group's histogram into ``into`` (the case-wide
        ``gate_hist`` dict + the dose / ratio attributes, or a per-key entry)."""
        case_wide = into is self.gate_hist
        for hk in ("gate_hist_n_groups", "gate_hist_zero", "gate_hist_near", "gate_hist_mid", "gate_hist_far"):
            into[hk] = int(into.get(hk) or 0) + int(gate_hist.get(hk) or 0)
        near = float(gate_hist.get("gate_dose_near") or 0.0)
        total = float(gate_hist.get("gate_dose_total") or 0.0)
        mn = gate_hist.get("gate_g_over_f_min_nonzero")
        mx = gate_hist.get("gate_g_over_f_max")
        if case_wide:
            self.gate_dose_near += near
            self.gate_dose_total += total
            if mn is not None:
                self.gate_g_over_f_min_nonzero = (float(mn) if self.gate_g_over_f_min_nonzero is None
                                                  else min(self.gate_g_over_f_min_nonzero, float(mn)))
            if mx is not None:
                self.gate_g_over_f_max = (float(mx) if self.gate_g_over_f_max is None
                                          else max(self.gate_g_over_f_max, float(mx)))
            return
        into["gate_dose_near"] = float(into.get("gate_dose_near") or 0.0) + near
        into["gate_dose_total"] = float(into.get("gate_dose_total") or 0.0) + total
        into["gate_dose_share_near_threshold"] = ((into["gate_dose_near"] / into["gate_dose_total"])
                                                  if into["gate_dose_total"] > 0 else None)
        if mn is not None:
            prev = into.get("gate_g_over_f_min_nonzero")
            into["gate_g_over_f_min_nonzero"] = float(mn) if prev is None else min(float(prev), float(mn))
        if mx is not None:
            prev = into.get("gate_g_over_f_max")
            into["gate_g_over_f_max"] = float(mx) if prev is None else max(float(prev), float(mx))

    def observe_error(self, *, key: Optional[str], layer: Any, cell: Any, rows: Any, site: Any,
                      error: BaseException) -> None:
        """A per-row apply that raised where the grouped apply did not.
        ``key`` None = an UNSERVED group (no op at this site, so no key):
        counted overall, entered under no key."""
        try:
            self.n_shadow_errors += 1
            e = self._entry(str(key)) if key is not None else None
            if e is not None:
                e["n_shadow_errors"] += 1
            if self.first_shadow_error is None:
                self.first_shadow_error = {"key": (str(key) if key is not None else None),
                                           "layer": _opt_int(layer),
                                           "cell": str(cell), "site": str(site),
                                           "rows": _opt_int(rows),
                                           "error": f"{type(error).__name__}: {error}"}
        except Exception as ex:  # pragma: no cover
            self.n_record_errors += 1
            if self.first_record_error is None:
                self.first_record_error = f"{type(ex).__name__}: {ex}"

    def observe_unserved(self, *, kind: Any, cell: Any, site: Any, layer: Any, rows: Any,
                         grouped_modified: bool, per_row_modified: bool) -> None:
        """One group whose cell serves NO op at this hook's (``site``,
        ``layer``): counted under ``kind@site`` (never a ``by_key`` entry —
        there is no op to key or classify). Either side modifying the rows
        is a routing defect (an op-less apply is the identity by contract):
        counted in ``n_unserved_modified`` AND as a modified-flag
        disagreement, which the gate fails in every reduction mode."""
        try:
            self.n_unserved_groups += 1
            k = f"{kind}@{site}"
            self.unserved[k] = self.unserved.get(k, 0) + 1
            rec = {"kind": str(kind), "cell": str(cell), "site": str(site),
                   "layer": _opt_int(layer), "rows": _opt_int(rows)}
            if self.first_unserved is None:
                self.first_unserved = dict(rec)
            if grouped_modified or per_row_modified:
                self.n_unserved_modified += 1
                self.n_flag_disagreements += 1
                if self.first_unserved_modified is None:
                    self.first_unserved_modified = dict(
                        rec, note=(f"no op at this site, yet modified: grouped="
                                   f"{bool(grouped_modified)}, per_row={bool(per_row_modified)}"))
        except Exception as ex:  # pragma: no cover
            self.n_record_errors += 1
            if self.first_record_error is None:
                self.first_record_error = f"{type(ex).__name__}: {ex}"

    def observe_layer_call(self, cells: Any) -> None:
        """One hook (layer, site) call's set of cells whose groups were compared."""
        try:
            cells = {str(c) for c in cells}
            self.n_layer_calls += 1
            self.max_distinct_cells_compared_in_one_layer_call = max(
                self.max_distinct_cells_compared_in_one_layer_call, len(cells))
            if len(cells) >= 2:
                self.n_layer_calls_with_ge2_cells_compared += 1
            for c in sorted(cells):
                if c not in self.cells_compared and len(self.cells_compared) < 64:
                    self.cells_compared.append(c)
        except Exception as ex:  # pragma: no cover
            self.n_record_errors += 1
            if self.first_record_error is None:
                self.first_record_error = f"{type(ex).__name__}: {ex}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "note": ("in-worker shadow compare: for every vectorized group at every hooked "
                     "layer the legacy per-row path ran on a clone of the same rows and the "
                     "two outputs were compared exactly (torch.equal); the per-row result was "
                     "served. Unequal groups record the deviation: max_abs_diff; max_ulps = the "
                     "largest |diff| in ulps of the per-row result's dtype at that element's own "
                     "magnitude (nonzero references; inflates where the edit cancels the "
                     "activation); max_ulps_edit = the largest excess of |stacked edit - per-row "
                     "edit| over one bf16 rounding of the output, in bf16 ulps of the per-row edit "
                     "(lens_adapters.edit_excess_ulps — the gate's per-element unit; kernel "
                     "variance <= ~1, a dropped / halved / misplaced edit far more wherever the edit "
                     "exceeds a couple of output ulps); max_frac_row / max_frac_group = the largest "
                     "fraction of unequal elements in any one row / any one group (the concentration "
                     "bounds that catch a whole row's edit dropped where it is within one output "
                     "ulp); max_abs_ref = the largest |per-row value| in an unequal group; "
                     "n_unequal_elems / n_elems_compared = elements that differed / elements in "
                     "every counted comparison. A group whose cell serves no op at the hook's "
                     "(site, layer) — the hook is another cell's — is UNSERVED (n_unserved_groups, "
                     "per kind@site), never keyed; a side modifying such rows counts as a "
                     "modified-flag disagreement. The CONTRACT is applied by the gate (scripts/lens_vectorize_"
                     "gate.py): exact for elementwise ops and under the per-row mode; a bounded "
                     "bf16-kernel-variance tolerance for reduction ops under the stacked mode — the "
                     "edit-ulp bound for LINEAR reduction ops (LoRA, LoReFT), and for GATED reduction "
                     "ops (the gated-vector and low-rank pairs) a bound AT THE GATE: |g_stacked - "
                     "g_per_row| <= 16 * 2^-24 * sum|h_i w_i| per gate element, a ReLU flip only when the "
                     "fired side is within that bound (2026-09-06). worst_unequal = the unequal group "
                     "where max_ulps_edit peaks, carrying worst_elem (that element's base / per-row / "
                     "stacked values, both edits, its row's edit ratio ||stacked edit|| / ||per-row edit|| "
                     "and cosine) and gate (the gated ops' gates as OBSERVED on the grouped and the per-row "
                     "apply — lens_adapters.GateCapture, never a recomputation: abs / relative deviation, "
                     "the smallest gate that differed, |delta| / sum|hw|, ReLU flips and the fired side's / "
                     "sum|hw|, and gate_edit_consistency_max_ulps = how far applied - ref departs from "
                     "eff * (g_stacked - g_per_row) * v, in ulps of the output + the edit; the recomputed "
                     "pair rides along as recomputed_* for reading only) — the worst_elem is evidence for "
                     "READING a violation, never the verdict; the gate_* totals (gate_max_diff_rel_sum, "
                     "gate_max_flip_rel_sum, gate_edit_consistency_max_ulps, gate_exact_rows_unequal_elems — "
                     "unequal elements in rows whose writing gates agreed or were masked off, where a "
                     "gated-vector write must be bit-identical — gate_lowrank_zero_delta_max_ulps_edit, "
                     "gate_n_groups_with_stats, gate_n_groups_observed, gate_n_errors, case-wide and per "
                     "key) ARE the gated-op rule's inputs: every unequal gated group must carry gate stats "
                     "whose gate_source is 'observed', and a gate_error is missing evidence; "
                     "gate_write_recon_mismatch_elems (with gate_n_groups_recon) counts elements of an unequal "
                     "vector-gated group that are not the bit-exact reconstruction base + eff g v from that "
                     "side's observed gate — any is a wrong write. gate_hist_n_groups / zero / near / mid / far, "
                     "gate_dose_share_near_threshold and gate_g_over_f_min_nonzero / max are the gate-MAGNITUDE "
                     "histogram over the observed per-row gates of EVERY compared gated group, equal or unequal "
                     "(in units of the noise floor f = 2^-24 sum|h_i w_i|; near = within 10 f of the ReLU): a "
                     "measurement the gate reports and never rules on. Gate-only (SISTEER_LENS_SHADOW_COMPARE=1); "
                     "unarmed, the production apply does one module-global lookup per gated op plus one branch "
                     "(vector op) or three branches stacked / B + 2 per-row (low-rank op) — no allocation, no "
                     "context manager; not zero"),
            "enabled": self.enabled,
            "n_groups_compared": self.n_groups_compared, "n_equal": self.n_equal,
            "n_unequal": self.n_unequal, "n_unmodified_groups": self.n_unmodified_groups,
            "n_shadow_errors": self.n_shadow_errors, "max_abs_diff": self.max_abs_diff,
            "max_ulps": self.max_ulps, "max_abs_ref": self.max_abs_ref,
            "n_unequal_elems": self.n_unequal_elems,
            "n_elems_compared": self.n_elems_compared, "max_ulps_edit": self.max_ulps_edit,
            "max_frac_row": self.max_frac_row, "max_frac_group": self.max_frac_group,
            "n_flag_disagreements": self.n_flag_disagreements,
            "n_unserved_groups": self.n_unserved_groups, "unserved": dict(self.unserved),
            "first_unserved": dict(self.first_unserved) if self.first_unserved else None,
            "n_unserved_modified": self.n_unserved_modified,
            "first_unserved_modified": (dict(self.first_unserved_modified)
                                        if self.first_unserved_modified else None),
            "first_unequal": dict(self.first_unequal) if self.first_unequal else None,
            "worst_unequal": dict(self.worst_unequal) if self.worst_unequal else None,
            "gate_max_abs_diff": self.gate_max_abs_diff, "gate_max_rel_diff": self.gate_max_rel_diff,
            "gate_min_abs_at_diff": self.gate_min_abs_at_diff,
            "gate_n_rows_unequal": self.gate_n_rows_unequal, "gate_n_relu_flips": self.gate_n_relu_flips,
            "gate_max_diff_rel_sum": self.gate_max_diff_rel_sum,
            "gate_max_flip_rel_sum": self.gate_max_flip_rel_sum,
            "gate_n_groups_with_stats": self.gate_n_groups_with_stats,
            "gate_n_errors": self.gate_n_errors,
            "gate_edit_consistency_max_ulps": self.gate_edit_consistency_max_ulps,
            "gate_n_groups_observed": self.gate_n_groups_observed,
            "gate_exact_rows_n": self.gate_exact_rows_n,
            "gate_exact_rows_unequal_elems": self.gate_exact_rows_unequal_elems,
            "gate_lowrank_zero_delta_max_ulps_edit": self.gate_lowrank_zero_delta_max_ulps_edit,
            "gate_n_groups_recon": self.gate_n_groups_recon,
            "gate_write_recon_mismatch_elems": self.gate_write_recon_mismatch_elems,
            **dict(self.gate_hist),
            "gate_dose_near": self.gate_dose_near, "gate_dose_total": self.gate_dose_total,
            "gate_dose_share_near_threshold": ((self.gate_dose_near / self.gate_dose_total)
                                               if self.gate_dose_total > 0 else None),
            "gate_g_over_f_min_nonzero": self.gate_g_over_f_min_nonzero,
            "gate_g_over_f_max": self.gate_g_over_f_max,
            "first_gate_error": dict(self.first_gate_error) if self.first_gate_error else None,
            "first_shadow_error": (dict(self.first_shadow_error)
                                   if self.first_shadow_error else None),
            "by_key": {k: dict(v, first_unequal=(dict(v["first_unequal"])
                                                if v["first_unequal"] else None),
                               worst_unequal=(dict(v["worst_unequal"])
                                              if v.get("worst_unequal") else None))
                       for k, v in self.by_key.items()},
            "n_keys_dropped": self.n_keys_dropped,
            "n_layer_calls": self.n_layer_calls,
            "n_layer_calls_with_ge2_cells_compared": self.n_layer_calls_with_ge2_cells_compared,
            "max_distinct_cells_compared_in_one_layer_call":
                self.max_distinct_cells_compared_in_one_layer_call,
            "cells_compared": list(self.cells_compared),
            "n_record_errors": self.n_record_errors,
            "first_record_error": self.first_record_error,
        }


class LensWorkerSteering:
    """The worker-resident steering wiring: preflight once, install
    once, then a per-forward hook body per steered layer. All logic is
    CPU-testable — the ``view`` is the only vLLM-shaped dependency."""

    def __init__(self, *, view: Any, registry: ls.WorkerObjectRegistry,
                 per_row_apply: Optional[str] = None,
                 shadow_compare: Optional[bool] = None,
                 rho_emit: Optional[bool] = None):
        self._view = view
        self._registry = registry
        self._ledger = lb.PrefillLedger()
        self._preflighted = False
        #: the apply path (module docstring, THE VECTORIZED APPLY): the
        #: env switch PER_ROW_APPLY_ENV, or an explicit override (tests,
        #: the GPU gate's harness) in the same vocabulary.
        raw = (per_row_apply if per_row_apply is not None
               else os.environ.get(PER_ROW_APPLY_ENV))
        self._per_row_all, self._per_row_names = parse_per_row_apply(raw)
        #: the GATE-ONLY shadow compare (SHADOW_COMPARE_ENV): read ONCE here;
        #: the apply path tests this attribute per group and nothing else
        #: changes when it is False.
        self._shadow_compare = bool(
            shadow_compare if shadow_compare is not None
            else parse_bool_env(os.environ.get(SHADOW_COMPARE_ENV))) and not self._per_row_all
        self.shadow_record = ShadowCompareRecord(enabled=self._shadow_compare)
        self._shadow_cells_this_call: set = set()
        #: the REDUCTION MODE inside a vectorized group (lens_adapters.
        #: STACKED_REDUCTIONS_ENV, read once at that module's import; stacked
        #: by default): attested as ``lens_reduction_mode`` (the scheduler-
        #: config report, the SelectionRecord) and right after the
        #: "vectorized" token of the apply path, so a standup can never read
        #: as the other mode. Under the all-per-row reference no group ever
        #: forms, so the path stays "per_row" — the mode is still attested,
        #: as the arithmetic a group WOULD get.
        self.reduction_mode = la.reduction_mode()
        self._stacked_reductions = (self.reduction_mode == la.REDUCTION_MODE_STACKED
                                    and not self._per_row_all)
        self.apply_path = (
            APPLY_PATH_PER_ROW if self._per_row_all else
            apply_path_string(self.reduction_mode, shadow=self._shadow_compare,
                              per_row_names=sorted(self._per_row_names)))
        #: the per-forward classification cache (module docstring): the
        #: forward key (a strong reference to the live attn_metadata
        #: object), the runner's request count when it was classified,
        #: and the classification (None = a passthrough forward).
        self._fwd_cache: Optional[Tuple[Any, int, Optional[ClassifiedForward]]] = None
        #: whether the plan lattice tolerates RESUMED (cache-hit)
        #: segments — keyed on the LIVE engine cache state at preflight,
        #: never guessed. Selector-less resumes always serve; a
        #: selector-carrying resume serves only for a cell whose ops pass
        #: lens_adapters.resumed_prefill_ok (as a RESUMED PREFILL) and
        #: refuses otherwise (build_request_slots).
        self._allow_resumed = False
        self._report: Optional[InstallReport] = None
        self._plan_layer: Optional[int] = None
        #: (layer, site rank) of the ONE hook that consumes the ledger
        #: (install() sets it; _SITE_RANK orders the sites at one layer).
        self._plan_site: Optional[Tuple[int, int]] = None
        #: the current forward's positions tensor, stashed by the
        #: decoder-layer PRE-hook of every o_proj-hooked layer for the
        #: o_proj hook that fires inside that layer's forward.
        self._fwd_positions: Any = None
        self._failure: Optional[str] = None
        #: per-REQUEST adapter failures, contained at the apply site
        #: (2026-08-30: a multi-turn render's response-boundary refusal,
        #: raised inside the forward hook, killed the whole EngineCore
        #: and 500'd every subsequent request). A failed request is
        #: served UNSTEERED from that forward on, recorded here, and
        #: claimed by the routing proxy (sisteer_take_request_failure),
        #: which refuses its response — loud per-request failure, live
        #: engine, and never a silently-unsteered completion.
        self._request_failures: Dict[str, Dict[str, Any]] = {}
        #: req_ids ever failed (never popped — the skip marker that
        #: keeps later forwards from re-attempting and re-recording).
        self._failed_req_ids: set = set()
        self._request_failures_cap = REQUEST_FAILURES_CAP
        #: assertion-6 evidence: this design has NO per-request
        #: registration path at all; the counter exists so the gate
        #: reads a live 0 rather than trusting prose.
        self.per_request_registrations = 0
        #: OBSERVATION ONLY (:class:`SelectionRecord`): which object each
        #: request was bound to, written at the apply site and read back
        #: by nothing on the apply path.
        self.selection_record = SelectionRecord(reduction_mode=self.reduction_mode)
        #: OBSERVATION ONLY (:class:`CoScheduleRecord`): per classified
        #: forward, the distinct nonzero groups that shared it.
        self.coschedule_record = CoScheduleRecord()
        #: OBSERVATION ONLY (:class:`RhoRecord`; RHO_EMIT_ENV, or the explicit
        #: override for tests): read ONCE here; the two block-apply sites test
        #: this one bool after each modifying write and nothing else changes
        #: when it is False. Default off.
        self._rho_emit = bool(
            rho_emit if rho_emit is not None
            else parse_bool_env(os.environ.get(RHO_EMIT_ENV)))
        self.rho_record = RhoRecord(enabled=self._rho_emit)

    # -- preflight (deliverable #5) --------------------------------------------------

    def preflight(self, tokenizer: Any,
                  header_verifier: Optional[Callable[[Any, Any], None]] = None,
                  turn_end_verifier: Optional[Callable[[Any, Any], None]] = None,
                  thinking_header_verifier: Optional[Callable[[Any, Any], None]] = None
                  ) -> None:
        """The structural startup checks, each a loud refusal:
        the V1 runner surface, the V2-runner flag, enforce_eager, a
        non-empty registry whose layers exist on this model, the
        assistant-header cross-check against the SERVED tokenizer
        (``lens_steer.verify_assistant_header``) and, for a registry whose
        LOADED ops include ``positions="response_all_turns"``, the turn-end
        scaffold and role-header cross-check
        (``lens_steer.verify_assistant_turn_end``) — gated on the ops, not
        on recorded metadata, so a response-only registry preflights as it
        always did on every template — and, for a registry whose LOADED ops
        carry a declared thinking-ON header, the ON-header cross-check
        (``lens_steer.verify_assistant_header_thinking_on``), gated the same
        way."""
        verifier = (header_verifier if header_verifier is not None
                    else DEFAULT_HEADER_VERIFIER)
        end_verifier = (turn_end_verifier if turn_end_verifier is not None
                        else DEFAULT_TURN_END_VERIFIER)
        on_verifier = (thinking_header_verifier if thinking_header_verifier is not None
                       else DEFAULT_THINKING_HEADER_VERIFIER)
        missing = tuple(self._view.missing_runner_attrs())
        if missing:
            raise LensWorkerError(
                f"preflight: the model runner lacks {list(missing)} — not "
                "the vLLM V1 attribute surface this wiring reads (a V2 "
                "runner, or an internals rename at a new pin). Refusing at "
                "startup: on this surface the alternative is silently "
                "steering nothing")
        v2 = self._view.v2_model_runner_active()
        if v2 is True:
            raise LensWorkerError(
                "preflight: the V2 model runner is active — per-request "
                "state lives elsewhere there and hooks would silently "
                "serve nothing (the same reason vllm-lens pins V1); "
                "refusing at startup")
        eager = self._view.enforce_eager()
        if eager is not True:
            raise LensWorkerError(
                f"preflight: enforce_eager is {eager!r}, not True — with "
                "torch.compile'd forwards the layer hooks never fire "
                "(vllm-lens forces eager for the same reason); refusing "
                "at startup rather than serving a silently-unhooked model")
        caching = self._view.prefix_caching_enabled()
        if caching is None:
            raise LensWorkerError(
                "preflight: the resolved prefix-caching state could not "
                "be read off the engine config — whether the plan lattice "
                "may tolerate resumed (cache-hit) segments is keyed on "
                "that state, and 'cannot verify' is a refusal, never a "
                "default in either direction")
        self._allow_resumed = bool(caching)
        cells = self._registry.cell_ids()
        if not cells:
            raise LensWorkerError(
                "preflight: the worker registry is EMPTY — nothing is "
                "lens-served, so installing the steering wiring is a "
                "launch-configuration error, not a no-op")
        layers = self._view.decoder_layers()
        if not layers:
            raise LensWorkerError(
                "preflight: no decoder layers found on the served model — "
                "nowhere to hook; refusing")
        n_layers = len(layers)
        for obj in _registry_objects(self._registry):
            for op in obj.ops:
                if op.layer >= n_layers:
                    raise LensWorkerError(
                        f"preflight: cell {obj.cell_id!r} steers layer "
                        f"{op.layer} but the served model has {n_layers} "
                        "decoder layers — the registry was exported for a "
                        "different model; refusing at startup")
        _blk, oproj_layers = site_layers(self._registry)
        if oproj_layers:
            # The o_proj-site ops read the o_proj module's INPUT, which
            # tensor parallelism SHARDS across ranks while the hook's
            # delta would land after the all-reduce on every rank — a
            # wrong partial read served N times. TP must therefore be
            # exactly 1, and "cannot verify" is a refusal, never a
            # default in either direction. Consulted ONLY when o_proj
            # ops are registered, so block-only registries never touch
            # the surface.
            tp_read = getattr(self._view, "tensor_parallel_size", None)
            tp = tp_read() if callable(tp_read) else None
            if tp is None:
                raise LensWorkerError(
                    "preflight: the registry carries o_proj-site ops "
                    f"(layers {list(oproj_layers)}) but the resolved "
                    "tensor-parallel size could not be read off the "
                    "engine config — under TP the o_proj input is "
                    "sharded across ranks, so an unverifiable TP cannot "
                    "be served; refusing at startup")
            if int(tp) != 1:
                raise LensWorkerError(
                    f"preflight: tensor_parallel_size == {int(tp)}, but "
                    "the registry carries o_proj-site ops (layers "
                    f"{list(oproj_layers)}) whose edit reads the o_proj "
                    "module's INPUT — sharded across ranks under TP, "
                    "while the delta lands post-all-reduce on every "
                    "rank. Serve this registry at TP=1; a TP-sharded "
                    "o_proj translation must land deliberately WITH its "
                    "own equivalence evidence")
            # the module must exist on every hooked layer NOW — install
            # discovers it too, but a startup refusal beats one a
            # request later.
            for l in oproj_layers:
                _resolve_oproj(layers[l], l)
        if tokenizer is None:
            raise LensWorkerError(
                "preflight: no served tokenizer supplied — the registry's "
                "assistant header cannot be cross-checked, and serving "
                "response-mode cells unchecked would steer the wrong "
                "region; refusing")
        verifier(tokenizer, self._registry.assistant_header_ids)
        # the turn-end / role-header certification is stricter than the
        # header check (it refuses templates whose role headers differ in
        # shape) and applies ONLY to what is actually served: gate it on
        # the registry's LOADED all-turns ops, never on scaffold metadata,
        # which older builds recorded for response-only registries too
        serves_all_turns = getattr(self._registry, "serves_all_turns", None)
        if serves_all_turns is not None and serves_all_turns():
            end_verifier(tokenizer, self._registry.assistant_turn_end_ids)
        # the declared thinking-ON header: the same discipline — gated on
        # LOADED ops carrying the second header, never on the presence of
        # assistant_header_ids_thinking_on metadata
        serves_thinking_on = getattr(self._registry, "serves_thinking_on", None)
        if serves_thinking_on is not None and serves_thinking_on():
            on_verifier(tokenizer,
                        getattr(self._registry, "assistant_header_ids_thinking_on", None))
        self._preflighted = True

    # -- install (deliverable #2) ----------------------------------------------------

    def install(self) -> InstallReport:
        """Register ONE persistent forward hook per steered layer, once
        (assertion 6). Registered DIRECTLY on the layer modules — never
        through vllm-lens's ``_persistent_hooks`` dispatch, whose
        try/except-log-and-skip would turn every refusal below into its
        documented silent no-op."""
        if not self._preflighted:
            raise LensWorkerError(
                "install before preflight — the structural checks are not "
                "optional; run preflight() first")
        if self._report is not None:
            raise LensWorkerError(
                "install called twice — the hook set is registered exactly "
                "once at worker startup (Phase-0 assertion 6); a second "
                "registration would double-apply every op")
        layers, oproj_layers = site_layers(self._registry)
        # the ledger-consuming hook is the FIRST-FIRING installed hook of
        # every forward: lowest layer, and at a tie the o_proj hook
        # (which fires inside that layer's forward, before its output
        # hook). Exactly one hook matches the key, so the credit is
        # drawn once per forward whichever sites are installed.
        candidates = ([(l, _SITE_RANK[la.HOOK_SITE_BLOCK]) for l in layers]
                      + [(l, _SITE_RANK[la.HOOK_SITE_OPROJ])
                         for l in oproj_layers])
        self._plan_site = min(candidates)
        self._plan_layer = self._plan_site[0]
        modules = self._view.decoder_layers()

        # Both decoder-layer hooks register ``with_kwargs=True``:
        # qwen3_next-style models call their layers KEYWORD-ONLY
        # (``layer(positions=..., hidden_states=..., residual=...)``),
        # so a hook registered without it would see ``args == ()`` and
        # could never read positions there. The o_proj module's own
        # hook stays positional-only on purpose — the pinned vLLM
        # linear call convention is ``o_proj(x)``, and its body refuses
        # loudly if that ever changes.
        def _make(layer_idx: int):
            def _hook(module: Any, args: Tuple[Any, ...],
                      kwargs: Mapping[str, Any], output: Any):
                return self.hook_body(layer_idx, args, output,
                                      kwargs=kwargs)
            return _hook

        def _make_pos_stash():
            def _pre(module: Any, args: Tuple[Any, ...],
                     kwargs: Mapping[str, Any]):
                # stash this forward's positions tensor (the same
                # keyword-first / args[0]-fallback read the block hook
                # makes) for the o_proj hook, which fires INSIDE the
                # layer forward and has no positions arg of its own.
                # Overwritten at every hooked layer with the same
                # tensor; never synthesized from the plan, so the
                # positions cross-check stays a real check. A call
                # carrying neither stashes None, which
                # _oproj_body_inner refuses.
                if kwargs and "positions" in kwargs:
                    self._fwd_positions = kwargs["positions"]
                else:
                    self._fwd_positions = args[0] if args else None
                return None
            return _pre

        for l in oproj_layers:
            modules[l].register_forward_pre_hook(_make_pos_stash(),
                                                 with_kwargs=True)
            _resolve_oproj(modules[l], l).register_forward_hook(
                self._make_oproj_hook(l))
        for l in layers:
            modules[l].register_forward_hook(_make(l), with_kwargs=True)
        self._report = InstallReport(
            registration_count=1, layers=layers,
            plan_layer=self._plan_layer,
            cells=self._registry.cell_ids(),
            ledger_cells=cells_needing_ledger(self._registry),
            oproj_layers=oproj_layers,
            resumed_prefill_ok_cells=resumed_prefill_ok_cells(self._registry),
            resumed_prefill_policy_cells=resumed_prefill_policy_cells(self._registry),
            lens_serve_declarations=serve_declarations_cells(self._registry),
            # the ON header this worker loaded and cross-checked (preflight),
            # None on a registry whose loaded ops carry none; the key is
            # always emitted from an install so the standup's reader can tell
            # "attested none" from "an older worker"
            assistant_header_ids_thinking_on=(
                getattr(self._registry, "assistant_header_ids_thinking_on", None)
                if _serves_thinking_on(self._registry) else None),
            thinking_header_attested=True)
        return self._report

    def _make_oproj_hook(self, layer_idx: int):
        def _hook(module: Any, args: Tuple[Any, ...], output: Any):
            return self.oproj_hook_body(layer_idx, args, output)
        return _hook

    # -- the run-level ledger (deliverable #4) ----------------------------------------

    def begin_run(self, prompt_lens: Sequence[int]) -> None:
        """Arm the OPTIONAL run-scoped prefill accounting with the
        submitted prompt lengths of the PLAN-NEEDING requests only
        (:func:`cells_needing_ledger` names which cells those are —
        arming a row-local request's length would refuse a run that is
        actually fine). In-proc exact-replay harness only: one
        ``llm.generate`` call == one run. The HTTP serving path never
        calls this — its defense is the per-request invariants, which
        run armed or not."""
        self._require_alive()
        self._ledger.begin_run(prompt_lens)

    def finish_run(self) -> None:
        """The run-end check: every armed length must have been seen as
        exactly one complete prefill. A failure here (or anywhere) has
        already condemned the worker — the run's outputs are unusable
        and every later call refuses, never a warning."""
        self._require_alive()
        if self._request_failures:
            rid, rec = next(iter(self._request_failures.items()))
            msg = (f"{len(self._request_failures)} unclaimed per-request "
                   f"steering failure(s) — first: request {rid!r} "
                   f"({rec['error_type']}: {rec['error']}). Their outputs "
                   "were served UNSTEERED and must be discarded, never "
                   "filed as this run's results (request failure)")
            self._condemn(msg)
            raise LensWorkerError(msg)
        try:
            self._ledger.finish_run()
        except lb.LensBatchError as e:
            self._condemn(str(e))
            raise

    def abort_run(self) -> None:
        """Exception-path cleanup for an armed run: drop the credits
        WITHOUT the run-end check, so a run that died for a non-steering
        reason (an engine error mid-``generate``) cannot leave stale
        credits that poison the next ``begin_run`` with "already armed".
        Deliberately NOT gated on ``_require_alive`` and NEVER raises —
        cleanup must be callable on a condemned worker (the condemnation
        itself is preserved; only the ledger arming is dropped)."""
        self._ledger.disarm()

    # -- per-request failure containment (the 2026-08-30 incident) --------------------

    def _record_request_failure(self, *, req_id: str, layer: int,
                                site: str, cell_id: str,
                                error: BaseException) -> None:
        """Contain ONE request's adapter failure: record it for the
        proxy to claim, mark the request so no later forward
        re-attempts, and keep the batch — and the engine — alive. The
        request is served UNSTEERED from here on, which is safe only
        because the record makes its response REFUSABLE: the proxy
        claims the entry and 500s the response, so an unsteered
        completion can never be filed as a served result.

        Fail-closed bound: a ledger past ``_request_failures_cap``
        means nothing is claiming entries (a proxy that no longer
        post-checks), and containment without a collector IS the silent
        no-op this design forbids — so overflow condemns the worker."""
        rid = str(req_id)
        self._failed_req_ids.add(rid)
        if rid in self._request_failures:
            return
        if len(self._request_failures) >= self._request_failures_cap:
            raise LensWorkerError(
                f"the per-request failure ledger holds "
                f"{len(self._request_failures)} unclaimed entries (cap "
                f"{self._request_failures_cap}) — nothing is claiming "
                "request failures (is the proxy post-check running?), "
                "and containing failures nobody reads would serve "
                "unsteered completions as results; refusing")
        self._request_failures[rid] = {
            "req_id": rid,
            "cell_id": str(cell_id),
            "layer": int(layer),
            "site": str(site),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        log.error(
            "petri-lens request failure CONTAINED (request %s, cell %s, "
            "layer %d, %s): %s — the request serves unsteered and its "
            "response must be refused at the proxy "
            "(sisteer_take_request_failure)", rid, cell_id, layer, site,
            error)

    def take_request_failure(self, req_id: str) -> Optional[Dict[str, Any]]:
        """Claim (pop) the failure recorded for ``req_id``, or ``None``.
        Also answers for a PARENT id whose engine ids were derived
        (the pinned vLLM: multi-prompt sub-requests ``<id>_<i>``,
        parallel-sampling children ``<i>_<id>``). Claiming is what
        converts a contained failure into a refused response; the skip
        marker stays, so a still-generating request cannot be
        re-attempted after its entry is claimed."""
        rid = str(req_id)
        rec = self._request_failures.pop(rid, None)
        if rec is not None:
            return rec
        for key in list(self._request_failures):
            if (key.startswith(rid + "_") or key.startswith(rid + "-")
                    or key.endswith("_" + rid)):
                return self._request_failures.pop(key)
        return None

    def _require_alive(self) -> None:
        if self._failure is not None:
            raise LensWorkerError(
                "this worker is CONDEMNED by an earlier steering failure "
                f"and its run's outputs must be discarded — {self._failure}")

    def _condemn(self, msg: str) -> None:
        if self._failure is None:
            self._failure = msg

    # -- the per-forward hook body (deliverable #3) ------------------------------------

    def hook_body(self, layer: int, args: Tuple[Any, ...], output: Any,
                  *, kwargs: Optional[Mapping[str, Any]] = None):
        """One steered layer's forward hook. Returns ``None`` (output
        untouched, bit-identical) or the replacement output with the
        delta written into ``hidden_states`` only. Every refusal
        condemns the worker before propagating — EXCEPT a per-slot
        ``AdapterError``, which is attributable to one request's own
        render/context and is CONTAINED instead (recorded, request
        served unsteered, response refused at the proxy): raised here it
        would kill the whole EngineCore, which is how one multi-turn
        Petri render took down the audit server on 2026-08-30.

        ``kwargs`` is the KEYWORD half of the decoder-layer call (the
        hooks register ``with_kwargs=True``): llama/gemma-style models
        call layers positionally and hand ``{}`` here, qwen3_next-style
        hybrids call ``layer(positions=..., hidden_states=...,
        residual=...)`` keyword-only and hand everything here. ``None``
        (the CPU tests' direct calls) reads like ``{}``."""
        self._require_alive()
        try:
            return self._hook_body_inner(layer, args, output,
                                         kwargs=kwargs)
        except Exception as e:
            self._condemn(f"{type(e).__name__}: {e}")
            raise

    def _forward_key(self) -> Any:
        """The live forward's identity (``view.forward_key``), or None
        when the view has none (the CPU fakes) / no forward context —
        then nothing is cached and every hook classifies, as before."""
        fk = getattr(self._view, "forward_key", None)
        return fk() if callable(fk) else None

    def _classify_forward(self, get_positions: Callable[[], Any], *,
                          layer: int = -1, site: str = la.HOOK_SITE_BLOCK
                          ) -> Optional[ClassifiedForward]:
        """The per-forward classification shared by BOTH hook sites,
        computed by the FIRST hook of a forward and reused by every
        later hook of the same forward (module docstring): the runner
        reads, selector extraction, the metadata refusals,
        ``build_request_slots`` and the vectorized grouping. ``None`` =
        a provable passthrough (warmup, or no selector in flight).

        The cache is keyed on the live forward context's attn_metadata
        OBJECT (``_forward_key``; a strong reference is held, so the
        identity cannot alias a later step's); the cheap per-hook check
        is that the runner's request count still matches — a forward
        whose key matches but whose batch changed under it is drift,
        refused. A refusal in the classification propagates (and
        condemns) from the hook that computed it; nothing is cached.

        ``get_positions`` defers the positions read (each site sources
        it differently — the block hook from its own args/kwargs, the
        o_proj hook from the pre-hook stash) until AFTER the passthrough
        and metadata checks, preserving the refusal order: a warmup
        forward with no positions is a passthrough; a selector-carrying
        forward whose positions cannot be read refuses.

        ``layer`` / ``site`` name the hook that computed the
        classification: a CONTAINED per-request refusal decided at
        classification (``RequestSlot.refusal`` — a conditional cell's
        unsafe resume) is recorded against them
        (:meth:`_record_request_failure`), once, here — the failed request
        then serves unsteered at every hook of this and later forwards
        exactly like an apply-site failure, and its response is refused at
        the proxy."""
        key = self._forward_key()
        cache = self._fwd_cache
        if key is not None and cache is not None and cache[0] is key:
            n_now = self._view.num_reqs()
            if n_now != cache[1]:
                raise LensWorkerError(
                    f"the forward context is the one classified with "
                    f"{cache[1]} requests but the runner now reports "
                    f"{n_now} — the batch changed under a live forward; "
                    "refusing to reuse (or rebuild) the classification")
            return cache[2]
        result, n = self._classify_forward_uncached(get_positions)
        if result is not None:
            for slot in result.fwd.slots:
                if slot.refusal is not None:
                    self._record_request_failure(
                        req_id=slot.req_id, layer=layer, site=site,
                        cell_id=(slot.selector or {}).get("petri_object", "?"),
                        error=slot.refusal)
        if key is not None:
            self._fwd_cache = (key, n, result)
        return result

    def _classify_forward_uncached(self, get_positions: Callable[[], Any]
                                   ) -> Tuple[Optional[ClassifiedForward], int]:
        n = self._view.num_reqs()
        if n == 0:
            # A dummy/profile/warmup forward: provably no request is in
            # flight, so passing through is not a silent skip of anyone's
            # steering. (Contrast: metadata missing WITH requests in
            # flight refuses below.)
            return None, n
        req_ids = tuple(self._view.req_ids())
        if len(req_ids) != n:
            raise LensWorkerError(
                f"runner reports {n} requests but yielded {len(req_ids)} "
                "request ids — refusing to pair rows with requests")
        raw = [extract_selector(self._view.extra_args(rid))
               for rid in req_ids]
        if not any(s is not None for s in raw):
            return None, n  # nothing steered in this forward

        # the CPU mirrors when the view has them (the real view — no
        # device sync per classification), else the view's plain read
        # (the CPU fakes hand lists / CPU tensors either way).
        fm_cpu = getattr(self._view, "forward_metadata_cpu", None)
        md = fm_cpu() if callable(fm_cpu) else self._view.forward_metadata()
        if md is None:
            raise LensWorkerError(
                "selector-carrying requests are in flight but the forward "
                "context carries no attention metadata — cannot place any "
                "request's rows; refusing to steer blind (and refusing to "
                "serve them unsteered, which would be the silent version)")
        qsl, seq_lens, nat = md
        nat_i = _as_int(nat)
        positions = get_positions()
        fp = getattr(self._view, "forward_positions", None)
        if callable(fp):
            positions = fp(positions, nat_i)
        fwd = build_request_slots(
            query_start_loc=qsl, seq_lens=seq_lens, num_actual_tokens=nat,
            positions=positions, req_ids=req_ids, raw_selectors=raw,
            token_lookup=self._view.request_tokens,
            prompt_len_lookup=self._view.prompt_len,
            registry=self._registry,
            allow_resumed=self._allow_resumed)
        groups, per_slot = build_slot_groups(
            fwd, per_row_all=self._per_row_all,
            per_row_names=self._per_row_names)
        # observation only (CoScheduleRecord): cannot raise, read by nothing
        # on the apply path.
        self.coschedule_record.observe(groups)
        return ClassifiedForward(fwd=fwd, nat=nat_i, groups=groups,
                                 per_slot=per_slot), n

    def _consume_ledger(self, layer: int, site: str,
                        fwd: "ForwardSlots") -> None:
        """The OPTIONAL run-scoped ledger (module docstring) consumes
        ONCE per forward: only while ARMED (the in-proc exact-replay
        harness's begin_run), only at the FIRST-FIRING installed hook
        (``self._plan_site`` — which runs on every forward), and only
        the TRACKED requests' prefills. Unarmed — the serving path's
        permanent state — it is not consulted: the per-request
        invariants inside build_request_slots are the always-on
        defense."""
        if ((layer, _SITE_RANK[site]) == self._plan_site
                and fwd.tracked_plan is not None and self._ledger.armed):
            self._ledger.observe(fwd.tracked_plan)

    def _hook_body_inner(self, layer: int, args: Tuple[Any, ...],
                         output: Any, *,
                         kwargs: Optional[Mapping[str, Any]] = None):
        def _positions() -> Any:
            # Two decoder-layer call conventions exist at the pinned
            # tag: llama/gemma-style models call ``layer(positions,
            # hidden_states, residual)`` POSITIONALLY (positions first),
            # while qwen3_next/qwen3_5-style hybrids call
            # ``layer(positions=..., hidden_states=..., residual=...)``
            # KEYWORD-ONLY — and their signature puts positions THIRD
            # (``forward(hidden_states, residual, positions=None)``), so
            # whenever the call names it, the keyword is the only
            # unambiguous read: ``args[0]`` on such a call would be
            # hidden_states, a silent mis-row. Keyword first, then the
            # positional convention; a call carrying NEITHER refuses.
            if kwargs is not None and "positions" in kwargs:
                return kwargs["positions"]
            if args:
                return args[0]
            raise LensWorkerError(
                "the decoder layer was called with neither a "
                "'positions' keyword nor positional args — both the "
                "(positions, hidden_states, residual) positional and "
                "the positions=... keyword call conventions this "
                "wiring reads positions from have changed; refusing")

        classified = self._classify_forward(_positions, layer=layer,
                                            site=la.HOOK_SITE_BLOCK)
        if classified is None:
            return None
        fwd = classified.fwd
        self._consume_ledger(layer, la.HOOK_SITE_BLOCK, fwd)

        hidden, residual, was_pair = self._split_output(output)
        true = hidden if residual is None else hidden + residual
        nat_i = classified.nat
        if true.dim() != 2 or true.shape[0] < nat_i:
            raise LensWorkerError(
                f"layer output carries shape {tuple(true.shape)} but the "
                f"metadata covers {nat_i} flattened rows — refusing to "
                "index rows the tensor does not carry")

        # the per-slot leftovers first (the reference loop over exactly
        # these slots — every selector slot under the per-row switch,
        # with the dense legacy masks), then one apply per group.
        new_hidden = self._apply_slots_block(
            layer, classified.per_slot, hidden, true, None,
            legacy_masks=self._per_row_all)
        if self._shadow_compare:
            self._shadow_cells_this_call = set()
        for group in classified.groups:
            new_hidden = self._apply_group_block(layer, group, hidden, true,
                                                 new_hidden)
        if self._shadow_compare:
            self.shadow_record.observe_layer_call(self._shadow_cells_this_call)
        if new_hidden is None:
            return None
        return (new_hidden, residual) if was_pair else new_hidden

    def _slot_apply(self, layer: int, slot: RequestSlot, true_rows: torch.Tensor,
                    *, legacy_masks: bool) -> torch.Tensor:
        """THE per-request block-site apply — the one function both the
        legacy reference path (``SISTEER_LENS_PER_ROW_APPLY=1``, dense
        masks) and the shadow compare run: ``registry.apply`` for this slot's
        selector on ``true_rows`` under the slot's own context."""
        return self._registry.apply(
            slot.selector["petri_object"], slot.selector["expect_hash"],
            true_rows, layer=layer, scale=slot.selector.get("scale"),
            ctx=slot.ctx, legacy_masks=legacy_masks)

    def _slot_apply_oproj(self, layer: int, slot: RequestSlot, in_rows: torch.Tensor,
                          base_rows: torch.Tensor, *, legacy_masks: bool) -> torch.Tensor:
        """The o_proj-site sibling of :meth:`_slot_apply`."""
        return self._registry.apply_oproj(
            slot.selector["petri_object"], slot.selector["expect_hash"],
            in_rows, base_rows, layer=layer, scale=slot.selector.get("scale"),
            ctx=slot.ctx, legacy_masks=legacy_masks)

    def _apply_slots_block(self, layer: int, slots: Sequence[RequestSlot],
                           hidden: torch.Tensor, true: torch.Tensor,
                           new_hidden: Optional[torch.Tensor], *,
                           legacy_masks: bool) -> Optional[torch.Tensor]:
        """The PER-REQUEST apply loop — the pre-vectorize body, verbatim,
        over ``slots``: prefill slots (per-request masks), objects forced
        per row, and, under ``SISTEER_LENS_PER_ROW_APPLY=1``, every slot
        with the dense legacy masks (the GPU gate's reference)."""
        for slot in slots:
            if slot.selector is None:
                continue
            if slot.req_id in self._failed_req_ids:
                continue  # already failed loudly; rows stay unsteered
            true_rows = true[slot.rows]
            try:
                applied = self._slot_apply(layer, slot, true_rows,
                                           legacy_masks=legacy_masks)
            except la.AdapterError as e:
                # attributable to THIS request's render/context alone —
                # contained: fail the request, not the engine
                # (2026-08-30; _record_request_failure).
                self._record_request_failure(
                    req_id=slot.req_id, layer=layer,
                    site=la.HOOK_SITE_BLOCK,
                    cell_id=slot.selector["petri_object"], error=e)
                continue
            modified = applied is not true_rows
            # OBSERVATION ONLY, at the apply site (SelectionRecord, whose
            # `coverage` states exactly which half of the claim this is):
            # ``slot.obj`` is the object ``select`` RESOLVED for this
            # selector — read, never re-derived — and ``modified`` is the
            # comparison the next line makes anyway. Nothing below reads
            # any of it back, no branch depends on it, and observe()
            # cannot raise.
            self.selection_record.observe(
                req_id=slot.req_id, layer=layer,
                requested=slot.selector["petri_object"],
                requested_hash=slot.selector["expect_hash"],
                requested_scale=slot.selector.get("scale"),
                applied=slot.obj, modified=modified,
                path=SELECTION_PATH_PER_SLOT, resumed=slot.resumed,
                ctx=slot.ctx, facts=slot.facts,
                prefill_masked_rows=slot.prefill_masked_rows)
            if not modified:
                continue  # untouched (scale 0 / phase-skipped / no op here)
            # THE SPLIT-RESIDUAL WRITE (module docstring): the op ran on
            # the TRUE stream; the delta lands on hidden_states alone and
            # the residual object passes through untouched — exactly
            # vllm-lens's own _apply_hook_delta discipline.
            delta = applied - true_rows
            if new_hidden is None:
                new_hidden = hidden.clone()
            new_hidden[slot.rows] = new_hidden[slot.rows] + delta
            if self._rho_emit:
                # OBSERVATION ONLY (RhoRecord; RHO_EMIT_ENV), AFTER the write:
                # reads `delta` and `true_rows` — neither aliases new_hidden
                # (a clone of hidden) — and cannot raise.
                self.rho_record.observe_slot(
                    req_id=slot.req_id, layer=layer,
                    cell_id=slot.selector["petri_object"], applied=slot.obj,
                    path=SELECTION_PATH_PER_SLOT,
                    phase=(None if slot.ctx is None else slot.ctx.phase),
                    delta=delta, true_rows=true_rows)
        return new_hidden

    def _live_slots(self, group: SlotGroup) -> Tuple[RequestSlot, ...]:
        """The group's slots not yet failed (a failure earlier in THIS
        forward — at a lower layer — takes a request out of every later
        layer's apply, exactly as the per-slot loop skips it)."""
        return tuple(s for s in group.slots
                     if s.req_id not in self._failed_req_ids)

    def _apply_group_block(self, layer: int, group: SlotGroup,
                           hidden: torch.Tensor, true: torch.Tensor,
                           new_hidden: Optional[torch.Tensor]
                           ) -> Optional[torch.Tensor]:
        """ONE apply for a vectorized group (module docstring, THE
        VECTORIZED APPLY): read the group's rows once, run the object's
        ops once on the stacked rows under the group's DecodeRows (or
        None) context with ``independent_rows`` (each reduction issued
        per row — every member is one request's single row), write the
        delta once. The same containment and
        observation as the per-slot loop, per member: an AdapterError
        for the group is the identical refusal each member's own apply
        would have raised (it depends on the op and the shared declared
        facts, never on a row), so every member is recorded failed."""
        live = self._live_slots(group)
        if not live:
            return new_hidden
        if len(live) != len(group.slots):
            # a member failed earlier in this forward: the cached rows no
            # longer describe the group; the survivors take the per-slot
            # loop for the rest of this forward (rare, exact).
            return self._apply_slots_block(layer, live, hidden, true,
                                           new_hidden, legacy_masks=False)
        rows = group.rows
        idx = None
        if isinstance(rows, slice):
            true_rows = true[rows]
        else:
            idx = group.index_on(true.device)
            true_rows = true.index_select(0, idx)
        # under the shadow compare, OBSERVE the gates this grouped apply uses
        # (la.GateCapture) — the gated contract's evidence. The seam is entered
        # ONLY then: the production path runs the bare apply (no context
        # manager, no capture object) — la.gate_capture refuses a None sink
        grouped_gates: Optional[la.GateCapture] = None
        try:
            if self._shadow_compare:
                grouped_gates = la.GateCapture()
                with la.gate_capture(grouped_gates):
                    applied = self._registry.apply(
                        group.cell_id, group.expect_hash, true_rows, layer=layer,
                        scale=group.scale, ctx=group.ctx, independent_rows=True)
            else:
                applied = self._registry.apply(
                    group.cell_id, group.expect_hash, true_rows, layer=layer,
                    scale=group.scale, ctx=group.ctx, independent_rows=True)
        except la.AdapterError as e:
            for slot in group.slots:
                self._record_request_failure(
                    req_id=slot.req_id, layer=layer,
                    site=la.HOOK_SITE_BLOCK, cell_id=group.cell_id,
                    error=e)
            return new_hidden
        modified = applied is not true_rows
        if self._shadow_compare:
            # GATE ONLY (SHADOW_COMPARE_ENV): the legacy per-row path on a
            # clone of the same rows, compared exactly (the deviation
            # recorded when unequal); the per-row result is what gets
            # served below.
            shadow = self._shadow_block(layer, group, true, true_rows, applied, modified,
                                        grouped_gates=grouped_gates)
            if shadow is None:
                return new_hidden      # a shadow error: members recorded failed
            applied, modified = shadow
        for slot in group.slots:
            self.selection_record.observe(
                req_id=slot.req_id, layer=layer, requested=group.cell_id,
                requested_hash=group.expect_hash,
                requested_scale=group.scale, applied=slot.obj,
                modified=modified, path=SELECTION_PATH_GROUPED,
                resumed=slot.resumed, ctx=slot.ctx, facts=slot.facts,
                prefill_masked_rows=slot.prefill_masked_rows)
        if not modified:
            return new_hidden  # untouched (scale 0 / phase-skipped / no op)
        delta = applied - true_rows
        if new_hidden is None:
            new_hidden = hidden.clone()
        if idx is None:
            new_hidden[rows] = new_hidden[rows] + delta
        else:
            # gather + the same elementwise add + scatter: unique indices,
            # so this is exactly `new_hidden[rows] = new_hidden[rows] +
            # delta` row by row (no atomics, no index_add_).
            new_hidden.index_copy_(0, idx,
                                   new_hidden.index_select(0, idx) + delta)
        if self._rho_emit:
            # OBSERVATION ONLY (RhoRecord; RHO_EMIT_ENV), AFTER the write: one
            # row per member; reads `delta` / `true_rows` only, cannot raise.
            self.rho_record.observe_group(
                req_ids=[s.req_id for s in group.slots], layer=layer,
                cell_id=group.cell_id, applied=[s.obj for s in group.slots],
                path=SELECTION_PATH_GROUPED,
                phase=(None if group.ctx is None else la.PHASE_DECODE),
                delta=delta, true_rows=true_rows)
        return new_hidden

    def _shadow_block(self, layer: int, group: SlotGroup, true: torch.Tensor,
                      true_rows: torch.Tensor, applied: torch.Tensor, modified: bool,
                      grouped_gates: Optional[la.GateCapture] = None
                      ) -> Optional[Tuple[torch.Tensor, bool]]:
        """The block-site shadow compare of ONE group (SHADOW_COMPARE_ENV):
        run :meth:`_slot_apply` with the dense legacy masks — the reference
        path's own function — on a CLONE of each member's rows (the same
        rows the legacy loop reads), stack the results in group row order,
        compare with the grouped ``applied`` and record; return the per-row
        result to SERVE (``(rows, modified)``), or None when the per-row
        path raised where the grouped one did not (recorded as a shadow
        error; the members are recorded failed, exactly as the legacy loop
        would have failed them). ``grouped_gates`` is the gate capture armed
        around the grouped apply that produced ``applied``; each per-row
        apply here is armed with its own, and an UNEQUAL gated group's gate
        evidence is the comparison of those two OBSERVATIONS
        (:func:`_gate_stats_observed`), never a recomputation."""
        obj = group.slots[0].obj
        kind = getattr(obj, "kind", "?")
        ops = self._registry.ops_for(group.cell_id, la.HOOK_SITE_BLOCK, layer)
        # no op at this (site, layer): an UNSERVED group (recorded below,
        # never keyed — shadow_key refuses empty ops)
        key = shadow_key(kind, ops, int(true_rows.shape[0])) if ops else None
        parts: List[torch.Tensor] = []
        per_row_gates: List[la.GateCapture] = []
        ref_modified = False
        try:
            for slot in group.slots:
                rows_i = true[slot.rows].clone()
                cap_i = la.GateCapture()
                with la.gate_capture(cap_i):
                    out_i = self._slot_apply(layer, slot, rows_i, legacy_masks=True)
                per_row_gates.append(cap_i)
                if out_i is not rows_i:
                    ref_modified = True
                parts.append(out_i)
        except la.AdapterError as e:
            self.shadow_record.observe_error(key=key, layer=layer, cell=group.cell_id,
                                             rows=true_rows.shape[0],
                                             site=la.HOOK_SITE_BLOCK, error=e)
            for slot in group.slots:
                self._record_request_failure(
                    req_id=slot.req_id, layer=layer, site=la.HOOK_SITE_BLOCK,
                    cell_id=group.cell_id, error=e)
            return None
        ref = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        if key is None:
            self.shadow_record.observe_unserved(
                kind=kind, cell=group.cell_id, site=la.HOOK_SITE_BLOCK, layer=layer,
                rows=true_rows.shape[0], grouped_modified=modified,
                per_row_modified=ref_modified)
            return (ref, ref_modified) if (modified or ref_modified) else (true_rows, False)
        if not modified and not ref_modified:
            self.shadow_record.observe(key=key, layer=layer, cell=group.cell_id,
                                       rows=true_rows.shape[0], site=la.HOOK_SITE_BLOCK,
                                       equal=None, max_abs_diff=None)
            return true_rows, False
        self._shadow_cells_this_call.add(group.cell_id)   # a COMPARED cell (modified rows)
        if modified != ref_modified:
            equal, note = False, (f"modified flags differ: grouped={modified}, "
                                  f"per_row={ref_modified}")
        else:
            equal, note = bool(torch.equal(applied, ref)), ""
        # the gate-magnitude histogram over the OBSERVED per-row gates of every compared gated
        # group, equal or not (measurement only; aggregated apart from the unequal-only stats)
        gate_hist = _gate_histogram_observed(ops, per_row_gates)
        stats = _unequal_stats(applied, ref, true_rows) if not equal else {}
        if stats:
            # the gated ops' gates as OBSERVED on both paths (the gated rule's input),
            # with the recomputation kept only as a labelled cross-check
            gs = _gate_stats_observed(ops, grouped_gates, per_row_gates, applied, ref, true_rows)
            if gs:
                if "gate_error" not in gs:
                    rc = _gate_stats(ops, true_rows)
                    gs["recomputed_max_diff_rel_sum"] = rc.get("gate_max_diff_rel_sum")
                    gs["recomputed_n_rows_unequal"] = rc.get("gate_n_rows_unequal")
                stats["gate"] = gs
        self.shadow_record.observe(key=key, layer=layer, cell=group.cell_id,
                                   rows=true_rows.shape[0], site=la.HOOK_SITE_BLOCK,
                                   equal=equal, note=note, n_elems=ref.numel(),
                                   gate_hist=(gate_hist or None), **stats)
        return ref, ref_modified

    def oproj_hook_body(self, layer: int, args: Tuple[Any, ...],
                        output: Any):
        """One steered layer's O_PROJ-module forward hook (the lora
        kinds — lens_adapters HOOK_SITE_OPROJ). Returns ``None`` (output
        untouched, bit-identical) or the replacement output with the
        steered rows written in. Every refusal condemns the worker
        before propagating, exactly like the block-site hook — with the
        same per-slot AdapterError containment carve-out."""
        self._require_alive()
        try:
            return self._oproj_body_inner(layer, args, output)
        except Exception as e:
            self._condemn(f"{type(e).__name__}: {e}")
            raise

    def _oproj_body_inner(self, layer: int, args: Tuple[Any, ...],
                          output: Any):
        def _positions() -> Any:
            pos = self._fwd_positions
            if pos is None:
                raise LensWorkerError(
                    "the o_proj hook fired with no stashed positions "
                    "tensor — the decoder-layer pre-hook that captures "
                    "positions did not run, or the layer call carried "
                    "neither a 'positions' keyword nor positional args; "
                    "the decoder-layer call conventions this wiring "
                    "reads positions from have changed. Refusing")
            return pos

        classified = self._classify_forward(_positions, layer=layer,
                                            site=la.HOOK_SITE_OPROJ)
        if classified is None:
            return None
        fwd, nat_i = classified.fwd, classified.nat
        self._consume_ledger(layer, la.HOOK_SITE_OPROJ, fwd)

        if not args or not torch.is_tensor(args[0]) or args[0].dim() != 2:
            raise LensWorkerError(
                "the o_proj module was called with "
                + (f"a {type(args[0]).__name__} first arg"
                   if args else "no positional args")
                + " — this wiring reads the o_proj INPUT rows from "
                "args[0] as a [T, d_in] tensor (the pinned vLLM linear "
                "call convention); refusing to guess where the "
                "pre-projection attention output lives")
        oproj_in = args[0]
        out_t, rest, was_tuple = self._split_oproj_output(output)
        if out_t.dim() != 2 or out_t.shape[0] < nat_i \
                or oproj_in.shape[0] != out_t.shape[0]:
            raise LensWorkerError(
                f"o_proj carries input {tuple(oproj_in.shape)} / output "
                f"{tuple(out_t.shape)} but the metadata covers {nat_i} "
                "flattened rows — refusing to index rows the tensors do "
                "not carry")

        new_out = self._apply_slots_oproj(
            layer, classified.per_slot, oproj_in, out_t, None,
            legacy_masks=self._per_row_all)
        if self._shadow_compare:
            self._shadow_cells_this_call = set()
        for group in classified.groups:
            new_out = self._apply_group_oproj(layer, group, oproj_in, out_t,
                                              new_out)
        if self._shadow_compare:
            self.shadow_record.observe_layer_call(self._shadow_cells_this_call)
        if new_out is None:
            return None
        return (new_out, *rest) if was_tuple else new_out

    def _apply_slots_oproj(self, layer: int, slots: Sequence[RequestSlot],
                           oproj_in: torch.Tensor, out_t: torch.Tensor,
                           new_out: Optional[torch.Tensor], *,
                           legacy_masks: bool) -> Optional[torch.Tensor]:
        """The per-request o_proj apply loop — the pre-vectorize body,
        verbatim (see :meth:`_apply_slots_block`)."""
        for slot in slots:
            if slot.selector is None:
                continue
            if slot.req_id in self._failed_req_ids:
                continue  # already failed loudly; rows stay unsteered
            base_rows = out_t[slot.rows]
            try:
                applied = self._slot_apply_oproj(layer, slot, oproj_in[slot.rows],
                                                 base_rows, legacy_masks=legacy_masks)
            except la.AdapterError as e:
                # same containment discipline as the block site.
                self._record_request_failure(
                    req_id=slot.req_id, layer=layer,
                    site=la.HOOK_SITE_OPROJ,
                    cell_id=slot.selector["petri_object"], error=e)
                continue
            modified = applied is not base_rows
            # OBSERVATION ONLY — the same SelectionRecord discipline as
            # the block site (its `coverage` statement applies verbatim).
            self.selection_record.observe(
                req_id=slot.req_id, layer=layer,
                requested=slot.selector["petri_object"],
                requested_hash=slot.selector["expect_hash"],
                requested_scale=slot.selector.get("scale"),
                applied=slot.obj, modified=modified,
                path=SELECTION_PATH_PER_SLOT, resumed=slot.resumed,
                ctx=slot.ctx, facts=slot.facts,
                prefill_masked_rows=slot.prefill_masked_rows)
            if not modified:
                continue  # untouched (scale 0 / fully masked / no op here)
            if new_out is None:
                new_out = out_t.clone()
            new_out[slot.rows] = applied
        return new_out

    def _apply_group_oproj(self, layer: int, group: SlotGroup,
                           oproj_in: torch.Tensor, out_t: torch.Tensor,
                           new_out: Optional[torch.Tensor]
                           ) -> Optional[torch.Tensor]:
        """ONE o_proj apply for a vectorized group (the o_proj sibling of
        :meth:`_apply_group_block`): the edit is computed from the
        group's o_proj INPUT rows and REPLACES its OUTPUT rows, exactly
        as the per-slot loop writes ``new_out[rows] = applied``."""
        live = self._live_slots(group)
        if not live:
            return new_out
        if len(live) != len(group.slots):
            return self._apply_slots_oproj(layer, live, oproj_in, out_t,
                                           new_out, legacy_masks=False)
        rows = group.rows
        idx = None
        if isinstance(rows, slice):
            in_rows, base_rows = oproj_in[rows], out_t[rows]
        else:
            idx = group.index_on(out_t.device)
            in_rows = oproj_in.index_select(0, idx)
            base_rows = out_t.index_select(0, idx)
        try:
            applied = self._registry.apply_oproj(
                group.cell_id, group.expect_hash, in_rows, base_rows,
                layer=layer, scale=group.scale, ctx=group.ctx,
                independent_rows=True)
        except la.AdapterError as e:
            for slot in group.slots:
                self._record_request_failure(
                    req_id=slot.req_id, layer=layer,
                    site=la.HOOK_SITE_OPROJ, cell_id=group.cell_id,
                    error=e)
            return new_out
        modified = applied is not base_rows
        if self._shadow_compare:
            shadow = self._shadow_oproj(layer, group, oproj_in, out_t, base_rows,
                                        applied, modified)
            if shadow is None:
                return new_out
            applied, modified = shadow
        for slot in group.slots:
            self.selection_record.observe(
                req_id=slot.req_id, layer=layer, requested=group.cell_id,
                requested_hash=group.expect_hash,
                requested_scale=group.scale, applied=slot.obj,
                modified=modified, path=SELECTION_PATH_GROUPED,
                resumed=slot.resumed, ctx=slot.ctx, facts=slot.facts,
                prefill_masked_rows=slot.prefill_masked_rows)
        if not modified:
            return new_out
        if new_out is None:
            new_out = out_t.clone()
        if idx is None:
            new_out[rows] = applied
        else:
            new_out.index_copy_(0, idx, applied)
        return new_out

    def _shadow_oproj(self, layer: int, group: SlotGroup, oproj_in: torch.Tensor,
                      out_t: torch.Tensor, base_rows: torch.Tensor, applied: torch.Tensor,
                      modified: bool) -> Optional[Tuple[torch.Tensor, bool]]:
        """The o_proj-site shadow compare of one group (see :meth:`_shadow_block`)."""
        obj = group.slots[0].obj
        kind = getattr(obj, "kind", "?")
        ops = self._registry.ops_for(group.cell_id, la.HOOK_SITE_OPROJ, layer)
        key = shadow_key(kind, ops, int(base_rows.shape[0])) if ops else None   # see _shadow_block
        parts: List[torch.Tensor] = []
        ref_modified = False
        try:
            for slot in group.slots:
                in_i = oproj_in[slot.rows].clone()
                base_i = out_t[slot.rows].clone()
                out_i = self._slot_apply_oproj(layer, slot, in_i, base_i, legacy_masks=True)
                if out_i is not base_i:
                    ref_modified = True
                parts.append(out_i)
        except la.AdapterError as e:
            self.shadow_record.observe_error(key=key, layer=layer, cell=group.cell_id,
                                             rows=base_rows.shape[0],
                                             site=la.HOOK_SITE_OPROJ, error=e)
            for slot in group.slots:
                self._record_request_failure(
                    req_id=slot.req_id, layer=layer, site=la.HOOK_SITE_OPROJ,
                    cell_id=group.cell_id, error=e)
            return None
        ref = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        if key is None:
            self.shadow_record.observe_unserved(
                kind=kind, cell=group.cell_id, site=la.HOOK_SITE_OPROJ, layer=layer,
                rows=base_rows.shape[0], grouped_modified=modified,
                per_row_modified=ref_modified)
            return (ref, ref_modified) if (modified or ref_modified) else (base_rows, False)
        if not modified and not ref_modified:
            self.shadow_record.observe(key=key, layer=layer, cell=group.cell_id,
                                       rows=base_rows.shape[0], site=la.HOOK_SITE_OPROJ,
                                       equal=None, max_abs_diff=None)
            return base_rows, False
        self._shadow_cells_this_call.add(group.cell_id)   # a COMPARED cell (modified rows)
        if modified != ref_modified:
            equal, note = False, (f"modified flags differ: grouped={modified}, "
                                  f"per_row={ref_modified}")
        else:
            equal, note = bool(torch.equal(applied, ref)), ""
        # no _gate_stats here: the only op served at HOOK_SITE_OPROJ is OP_LORA_OPROJ
        # (la.OPROJ_SITE_OPS), a LINEAR reduction op — no gated op can ever apply at
        # this site, so the block-site call carries every gated group's gate evidence
        # (tests/test_lens_stacked_reductions.py pins OPROJ_SITE_OPS disjoint from
        # GATED_REDUCTION_OPS so a future gated o_proj op has to revisit this)
        stats = _unequal_stats(applied, ref, base_rows) if not equal else {}
        self.shadow_record.observe(key=key, layer=layer, cell=group.cell_id,
                                   rows=base_rows.shape[0], site=la.HOOK_SITE_OPROJ,
                                   equal=equal, note=note, n_elems=ref.numel(), **stats)
        return ref, ref_modified

    @staticmethod
    def _split_oproj_output(output: Any
                            ) -> Tuple[torch.Tensor, Tuple[Any, ...], bool]:
        """(output_rows, trailing, was_tuple) from an o_proj module's
        forward output — the pinned vLLM linear layers return
        ``(output, output_bias)`` where the bias slot is None unless
        skip_bias_add (RowParallelLinear.forward), and the HF reference
        module returns a plain tensor. A non-None trailing bias refuses:
        it means the module deferred its bias, so the tensor in slot 0
        is NOT the o_proj output the authoritative hook edits."""
        if (isinstance(output, tuple) and len(output) == 2
                and torch.is_tensor(output[0])):
            if output[1] is not None:
                raise LensWorkerError(
                    "o_proj returned (output, bias) with a non-None "
                    "deferred bias (skip_bias_add) — slot 0 is then the "
                    "PRE-bias projection, not the o_proj output the "
                    "authoritative hook edits; refusing to steer a "
                    "tensor that is not the one the object was fit on")
            return output[0], (output[1],), True
        if torch.is_tensor(output):
            return output, (), False
        raise LensWorkerError(
            f"o_proj returned {type(output).__name__} — neither the "
            "(output, bias) pair nor a plain tensor; refusing to guess "
            "where the projection output lives")

    @staticmethod
    def _split_output(output: Any
                      ) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        """(hidden_states, residual, was_pair) from a decoder layer's
        output — the Llama-family ``(hidden_states, residual)`` pair or
        a plain tensor (both forms vllm-lens handles); anything else
        refuses."""
        if (isinstance(output, tuple) and len(output) == 2
                and torch.is_tensor(output[0])):
            resid = output[1]
            if resid is not None and not torch.is_tensor(resid):
                raise LensWorkerError(
                    f"decoder layer returned (tensor, {type(resid).__name__})"
                    " — not the (hidden_states, residual) pair this wiring "
                    "understands; refusing")
            return output[0], resid, True
        if torch.is_tensor(output):
            return output, None, False
        raise LensWorkerError(
            f"decoder layer returned {type(output).__name__} — neither the "
            "(hidden_states, residual) pair nor a plain tensor; refusing "
            "to guess where the residual stream lives")


# --- the worker extension (GPU-only glue; every line GPU-VERIFY) -------------------
#
# Thin by design: each RPC method is one call into the CPU-tested
# orchestrator. Defined only when vllm-lens (and therefore vllm) is
# importable, because vLLM imports ``worker_extension_cls`` by its
# module path INSIDE the worker process — a lazily-built class cannot
# fill that slot.

try:
    from vllm_lens._worker_ext import HiddenStatesExtension as _VLensExt
    _HAVE_WORKER_EXT = True
except ImportError:
    _HAVE_WORKER_EXT = False

if _HAVE_WORKER_EXT:

    class SisteerWorkerExtension(_VLensExt):  # pragma: no cover — GPU-only
        """The single ``worker_extension_cls``: vllm-lens's stock RPC
        surface (the Phase-0 gate's probes) by inheritance, plus the
        petri steering wiring. All methods are ``collective_rpc``
        endpoints; arguments and returns are plain JSON-able values
        (H3: tensors and callables never cross the process boundary)."""

        _sisteer_steering: Optional[LensWorkerSteering] = None

        def sisteer_install_steering(self, registry_dir: str
                                     ) -> Dict[str, Any]:
            # GPU-VERIFY end to end: RealWorkerView surface, tokenizer
            # load, device/dtype materialization, hook registration.
            if self._sisteer_steering is not None:
                raise LensWorkerError(
                    "sisteer_install_steering called twice — the hook set "
                    "is registered exactly once at worker startup")
            view = RealWorkerView(self)
            registry = ls.WorkerObjectRegistry.load(Path(registry_dir))
            device, dtype = view.model_device_dtype()
            registry = registry.materialize(device=device, dtype=dtype)
            steering = LensWorkerSteering(view=view, registry=registry)
            steering.preflight(self._sisteer_load_tokenizer())
            report = steering.install()
            self._sisteer_steering = steering
            return install_report_payload(report)

        def sisteer_selection_record(self, reset: bool = False
                                     ) -> Dict[str, Any]:
            """The worker's DIRECT record of which object it bound to
            each request (:class:`SelectionRecord`) — the observed half
            of ``selection_honored``, next to the output-comparison half
            the harness computes. Deliberately NOT gated on
            ``_require_alive``: after a condemnation this record is
            exactly what a reader needs. ``reset=True`` returns the
            snapshot AND clears, so a caller can scope the next read to
            one submitted batch."""
            steering = self._sisteer_require()
            snapshot = steering.selection_record.as_dict()
            if reset:
                steering.selection_record.reset()
            return snapshot

        def sisteer_rho_record(self, reset: bool = False) -> Dict[str, Any]:
            """The worker's per-(request, hook layer) served edit-ratio
            record (:class:`RhoRecord`; SISTEER_LENS_RHO_EMIT=1) — the
            add norm, the composed-stream norm and rho, per steered
            layer per request. ``enabled`` False means the flag was off
            and the record is empty by construction. ``reset=True``
            returns the snapshot AND clears. Observation only; not gated
            on ``_require_alive``."""
            steering = self._sisteer_require()
            snapshot = steering.rho_record.as_dict()
            if reset:
                steering.rho_record.reset()
            return snapshot

        def sisteer_apply_path(self) -> str:
            """Which apply path this worker serves with —
            ``vectorized+stacked`` (the default), ``vectorized+per_row``
            (SISTEER_LENS_STACKED_REDUCTIONS=0), either with
            ``+shadow(serves per_row)`` or ``(per_row: <names>)`` appended,
            or ``per_row`` (SISTEER_LENS_PER_ROW_APPLY=1, the reference) —
            read off the live steering (LensWorkerSteering.apply_path). The
            GPU fidelity gate (scripts/lens_vectorize_gate.py) records it
            per standup so a comparison can never be between two standups
            on the same path, or in the other reduction mode, by mistake. Deliberately NOT part of the install report
            (whose every field readiness compares against the registry)."""
            return self._sisteer_require().apply_path

        def sisteer_shadow_compare(self, reset: bool = False
                                   ) -> Dict[str, Any]:
            """The in-worker shadow compare's record (:class:`ShadowCompareRecord`
            — SHADOW_COMPARE_ENV): per (kind, op, shape, positions, skip_bos,
            phase, rows) the groups compared / equal / unequal, the first
            unequal group with its max |diff|, and the per-layer-call cell
            coverage. ``reset=True`` returns the snapshot AND clears. Not
            gated on ``_require_alive``."""
            steering = self._sisteer_require()
            snapshot = steering.shadow_record.as_dict()
            if reset:
                steering.shadow_record.reset()
            return snapshot

        def sisteer_coschedule_record(self, reset: bool = False
                                      ) -> Dict[str, Any]:
            """The worker's co-scheduling evidence (:class:`CoScheduleRecord`):
            per classified forward, how many distinct nonzero vectorized
            groups shared it. Read by the GPU gate's multigroup case after
            every wave; ``reset=True`` returns the snapshot AND clears.
            Observation only; not gated on ``_require_alive``."""
            steering = self._sisteer_require()
            snapshot = steering.coschedule_record.as_dict()
            if reset:
                steering.coschedule_record.reset()
            return snapshot

        def sisteer_take_request_failure(self, req_id: str
                                         ) -> Optional[Dict[str, Any]]:
            """Claim the contained per-request steering failure for
            ``req_id`` (the chat completion's own id), or ``None``. The
            routing proxy calls this for every lens response BEFORE
            forwarding it, and refuses the response when an entry comes
            back — the loud per-request half of the containment.
            Deliberately NOT gated on ``_require_alive``: after a
            condemnation these records are exactly what a reader
            needs."""
            return self._sisteer_require().take_request_failure(req_id)

        def sisteer_begin_run(self, prompt_lens: List[int]) -> None:
            self._sisteer_require().begin_run(prompt_lens)

        def sisteer_finish_run(self) -> None:
            self._sisteer_require().finish_run()

        def sisteer_abort_run(self) -> None:
            """Exception-path cleanup for an armed run (in-proc harness
            only): drop the credits without the run-end check. Callable
            on a condemned worker — cleanup must never be refused."""
            self._sisteer_require().abort_run()

        def sisteer_mrope_report(self) -> Dict[str, Any]:
            """Rotary attestation (vllm_patches.mrope_patch_report): is a
            non-neox M-RoPE model off the buggy triton kernel? Same RPC the
            plain sampler's extension answers (vllm_patches.attest_engine)."""
            return _vp.mrope_patch_report(self.model_runner.model)

        def sisteer_mamba_backend_report(self) -> Dict[str, Any]:
            """The NEMOTRON FUSED-KERNEL BACKEND report (mamba_backend.
            live_backend_report; MANDATORY at
            every Nemotron standup): every decoder layer whose mixer is a
            MambaMixer2, each kernel role resolved through THAT mixer's module
            globals as {module, qualname}, plus the initialised
            selective-state-update dispatch backend and its kernel. Facts only
            — the standup judges them against the reviewed reference contract
            (configs/capeval/mamba_backend_contract.json) and refuses a missing
            / substituted / unrecognised binding or a missing layer. A
            Nemotron cell never runs on the quiet pure-torch SSM fallback;
            this report is how the served backend is READ rather than
            inferred from a healthy-looking job."""
            return _mb.live_backend_report(self.model_runner.model)

        def sisteer_scheduler_config(self) -> Dict[str, Any]:
            # GPU-VERIFY: vllm_config unpacked onto the Worker
            # (WorkerBase — the same read RealWorkerView's
            # v2_model_runner_active makes), and worker.model_runner —
            # the same surface the REQUIRED_RUNNER_ATTRS walk verifies.
            # Deliberately NOT gated on the steering install: it is a
            # pure config read the launcher runs during readiness, and a
            # geometry that cannot be read must fail the launch loudly
            # (both report halves refuse), never pass it.
            out = scheduler_config_report(self.vllm_config)
            out.update(kv_group_report(
                self.vllm_config, getattr(self, "model_runner", None)))
            # the RESOLVED batch-invariance state (env as vLLM parsed it +
            # the active runtime state): recorded for every launch; the gate
            # holds it to the launcher's intent (review #5, item 4). The
            # launcher's readiness compares only its expected keys, so this
            # is attestation, not a readiness gate — vllm_server is untouched.
            out.update(batch_invariant_report())
            # the REDUCTION MODE served inside a vectorized group
            # (lens_adapters.reduction_mode(), read once at that module's
            # import in THIS worker process): ``stacked`` | ``per_row``.
            # Lands in the attestation file's worker_reports next to the
            # batch-invariant fields, so every standup record and the
            # proxy's /v1/models identity block can name the arithmetic a
            # store was generated under (EvalCheck reads it; absent = an
            # older tree).
            out[LENS_REDUCTION_MODE_KEY] = la.reduction_mode()
            return out

        def _sisteer_require(self) -> LensWorkerSteering:
            if self._sisteer_steering is None:
                raise LensWorkerError(
                    "petri steering is not installed in this worker — "
                    "call sisteer_install_steering at server start")
            return self._sisteer_steering

        def _sisteer_load_tokenizer(self) -> Any:
            # GPU-VERIFY: model_config.tokenizer is the served tokenizer
            # path. Plain from_pretrained — never trust_remote_code.
            path = self.model_config.tokenizer
            if not path:
                raise LensWorkerError(
                    "model_config carries no tokenizer path — the "
                    "assistant-header preflight cannot run; refusing")
            from transformers import AutoTokenizer

            from sisteer.models import pin_reasoning_template_off
            tok = AutoTokenizer.from_pretrained(path)
            # THE VERIFIER MUST RENDER THE FIT'S REGIME (job 255399,
            # 2026-08-30): this fresh tokenizer carries no thinking-off
            # pin, so on a reasoning template (Qwen3.5) the preflight's
            # generation-prompt render was thinking-ON while the probe
            # render carried the completed turn's empty <think></think>
            # block — divergent at the think boundary ([198] vs
            # [271, 248069, 271]), read as boundary retokenization, and
            # the header verification refused at install. The registry's
            # recorded header IS the thinking-off header (H=7, the fit's
            # regime), so the served derivation must be pinned the same
            # way — the identical seam as every other render site
            # (models.load_model_and_tokenizer, chat_markers,
            # openai_shim, both smoke workers). A genuine template
            # mismatch still refuses: the pin changes the RENDER REGIME,
            # never the comparison.
            pin_reasoning_template_off(tok)
            return tok
