"""The lora-native worker extension: vllm-lens-free, read-only.

A lora-native launch keeps the vllm-lens plugin OUT of the server
process (``VLLM_PLUGINS=""`` + ``VLLM_LENS_DISABLE=1`` in the launch
env) so CUDA graphs stay on — which means ``lens_worker.
SisteerWorkerExtension`` (a SUBCLASS of vllm-lens's extension) cannot be
the ``--worker-extension-cls`` there. This class fills vLLM's single
extension slot instead and exposes ONE readiness ``collective_rpc``
method, ``sisteer_scheduler_config``, whose payload the launcher holds
to the composed launch at readiness (``vllm_server.
_verify_native_lora_state``): the no-chunking scheduler geometry and KV
coverage (as for lens), the batch-invariance state, the RESOLVED
eager/graph state — including the dispatcher's FULL capture surface off
``model_runner.cudagraph_dispatcher`` (2026-09-16: FULL decode graphs stop
at ``max_num_seqs``, so readiness holds their largest size to it) — the LoRA
config, and every loaded adapter's fingerprint (``worker_reports``). No install RPC, no hooks: the native
adapters are applied by vLLM itself, selected per request by the
``model`` name the routing proxy sets.

THE LORA ROUTING RECORDER (gemma canary job 489308). Three more
``sisteer_``-prefixed RPCs, used ONLY by the standup's / gate's
``selection_honored`` control and never by a production launch:
``sisteer_arm_lora_routing_recorder(epoch)`` adopts the CLIENT's EPOCH id
(every worker the same one): it wraps the model runner's per-step
``execute_model`` and records, off the
``scheduler_output`` both GPU model runners consume (V1 and V2 — canary #4
ran V2, which has no ``input_batch``), every admitted request's engine id
-> the LoRA id/name the engine bound it to, plus each step's scheduled
request set, cross-checked against the runner's own state by class
(``worker_reports.LoraRoutingRecorder``); ``sisteer_lora_routing_
observations(epoch)`` returns the whole epoch (the engine ids wear the
``-<8 hex>`` randomisation suffix — the check joins them to the responses'
ids); ``sisteer_disarm_lora_routing_recorder(epoch)`` restores
``execute_model``; both refuse another epoch. Every EXPECTED refusal
(``already_armed`` / ``epoch_mismatch`` / ``bad_epoch`` / ``not_armed``) is a
PAYLOAD the client reader refuses on, never an exception out of the method:
the pinned executor reads the per-worker reply queues sequentially and raises
on the first failure without draining the rest, so an exception on a TP>1
launch desyncs the owner's later calls. The recorder lives on the Worker
INSTANCE (``self.sisteer_lora_routing_recorder``, set at arm time) — the
injection asserts no CLASS attribute collides, and an instance attribute
under the same prefix cannot.

THE NEMOTRON BACKEND REPORT (2026-09-13).
A FIFTH ``sisteer_``-prefixed RPC, ``sisteer_mamba_backend_report``, PORTED
from the lens worker (``lens_worker.SisteerWorkerExtension`` answers the same
name): the same pure ``mamba_backend.live_backend_report`` over the native
model runner's model — every decoder layer whose mixer is a ``MambaMixer2``,
each kernel role resolved through that mixer's module globals, plus the
initialised selective-state-update dispatch backend. Facts only; the native
standup core (``native_standup.bring_up_native``) asks it through
``mamba_backend.live_attest`` (``RPC_METHOD``, the contract's own name) and
judges it against the reviewed contract with ``mamba_backend.live_problems``
— a Nemotron standup without a passing report REFUSES (the rule: a
Nemotron cell never runs on the quiet pure-torch SSM fallback). The
cross-workstream constants pin (``tests/test_native_lora_constants_agree.py``)
names five methods on this class since this landing.

THE RESIDUAL SLOT (2026-09-14). Importing
this module imports ``native_residual``, whose IMPORT is the pre-load hook the
slot needs (fact 6: vLLM imports the extension's module in ``init_worker``
before ``load_model``): when the launch env carries
``SISTEER_NATIVE_RESIDUAL_SLOT=1`` (the composer sets it iff a served adapter
carries a ``petri_residual_slot`` payload) it registers
``torch.ops.vllm.sisteer_residual_slot`` and installs the decoder-layer /
LoRA-manager / punica / runner-mixin patches; with the flag unset it touches
nothing and a LoRA-only standup's graph is byte-for-byte the certified one;
with the flag set and vllm absent the import RAISES (a worker asked to arm the
slot never comes up unarmed). Two more ``sisteer_``-prefixed RPCs:
``sisteer_residual_slot_report`` (``worker_reports.residual_slot_report`` —
armed, the op name, the patched classes, the hooked layer count, the bank, the
per-slot payload facts, the runner binding, the ledger state; facts only, the
launcher judges them) and ``sisteer_take_request_failure(req_id)`` — the lens's
contract (``lens_worker.SisteerWorkerExtension.sisteer_take_request_failure``):
claim-and-clear the per-request failure ledger entry, ``None`` when none, and a
FAULT PAYLOAD for every claim once the ledger latched (fact 15) — payloads,
never exceptions (the TP>1 reply-queue rule below). The cross-workstream
constants pin names seven methods on this class since this landing.

THE COMPILE-CACHE ENV FACTOR (tinycap-shared-standup, warm-cache-miss diagnosis
2026-09-29, fix F1). vLLM 0.25.1 keys its torch.compile cache directory on
``sha256([env_hash, config_hash, code_hash, compiler_hash])[:10]``
(``vllm/compilation/backends.py`` 1026–1064), and ``env_hash`` is
``hash_factors(envs.compile_factors())`` — the VALUE of every known vLLM env
var minus an ignore list that, at the tag, does NOT contain ``VLLM_API_KEY``
(``envs.py`` 752 reads it; 2050–2119 never drops it; upstream ``main`` has
since added it with the comment "Credential; never affects compiled artifacts
and must not be persisted in cache_key_factors.json"). The launcher mints a
fresh random token per launch and injects it as ``VLLM_API_KEY`` into the
server process (``vllm_server.launch_server``; ``vllm_auth.TOKEN_ENV``), the
engine/worker child inherits it, so every launch hashed a different
``env_hash`` and no restored compile cache could ever hit (probe
``485efd7135`` vs row ``71450296d1`` with every config factor equal). This
module is imported in EVERY worker before ``init_device`` / ``load_model``
(``worker_base.py`` 261–263: the extension class is resolved by qualname in
``init_worker``) — the harness's one hook in the compiling process before the
first compile — so importing it POPS the variable from that process's
environment (:func:`drop_api_key_from_compile_env`). The API-server process
keeps its copy: nothing it imports pops it (``vllm_entry``, ``vllm_auth``, the
clamp shim; a test pins this by AST), and both guards — vLLM's stock
``--api-key`` middleware and ``SisteerTokenAuthMiddleware`` — read it there.
ONE exception: under ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` the engine core runs
INSIDE the API-server process (the lens gate's / smoke's in-process worker),
and ``api_server.py`` @0.25.1 builds the engine (``run_server_worker`` 712)
before ``build_app`` reads the token (252, 285) — a pop there would silently
drop the stock guard and make ours refuse to start — so the pop is gated on
the engine running out of process. Not an ``--api-key`` argv (the /proc leak
rule in ``vllm_server``) and not a per-lane deterministic token (weakens the
random-per-launch guarantee): the token stays exactly as minted, and only the
process that compiles stops seeing it.

THE ROUTED-EXPERT LoRA BUFFER PATCH. Importing this module
also imports ``native_moe_lora``, whose IMPORT is the second pre-load hook (the
same fact 6): when the launch env carries ``SISTEER_MOE_LORA_RANK=<R>`` — the
composer sets it iff the adapted class wraps the routed experts, from the
registered adapters' own routed rank — it patches ``FusedMoEWithLoRA`` /
``FusedMoE3DWithLoRA``'s buffer allocation to the routed R instead of the
server-wide ``max_lora_rank`` (the cap-to-routed ratio smaller), refuses by name
an adapter whose routed factors exceed R, and clamps the warmup dummies' rank;
with the flag unset it touches nothing. ``sisteer_scheduler_config`` folds
``native_moe_lora.report()`` into its payload (armed, the composed R, the
patched classes) and the launcher holds every MoE wrapper's buffer R to the
composed value (``vllm_server.validate_moe_lora_rank``). None of the paper
models routes experts; the hook stays as an option for a checkpoint that does.

IMPORT-TIME CONTRACT. Importable with neither vllm nor vllm-lens
present: every vllm access happens inside the method, off ``self`` (the
Worker this class is injected into). vLLM injects a worker extension by
appending the class to the Worker's ``__bases__`` after asserting that
NO non-dunder attribute of the extension already exists on the Worker
(``vllm/v1/worker/worker_base.py`` @0.25.1), so this class defines ONLY
``sisteer_``-prefixed methods — it READS worker state
(``self.vllm_config``, ``self.model_runner``) and overrides nothing
(``add_lora`` and friends stay vLLM's); the routing recorder wraps the
model runner's ``execute_model`` on the INSTANCE, and only while armed.
"""

from __future__ import annotations

import os
from typing import Any, Dict, MutableMapping, Optional

from sisteer.capeval.serving import native_moe_lora as _native_moe_lora  # import-time arming (module docstring)
from sisteer.capeval.serving import native_residual as _native_residual  # noqa: F401 — import-time arming (module docstring)
from sisteer.capeval.serving import vllm_auth as _vllm_auth
from sisteer.capeval.serving import worker_reports as wr

#: the per-launch token's variable — ``vllm_auth.TOKEN_ENV``, the one the launcher injects and both auth
#: guards read (a cross-module test pins the three spellings equal)
TOKEN_ENV = _vllm_auth.TOKEN_ENV
#: vLLM's switch for an IN-PROCESS engine core (``envs.py`` 1293: ``bool(int(os.getenv(..., "1")))``): with
#: it at ``0`` the worker shares the API server's interpreter and the token must stay
ENGINE_IN_PROCESS_ENV = "VLLM_ENABLE_V1_MULTIPROCESSING"


def drop_api_key_from_compile_env(environ: MutableMapping[str, str]) -> str:
    """Remove ``VLLM_API_KEY`` from ``environ`` so ``envs.compile_factors()`` — which reads it off
    ``os.environ`` at the first compile and hashes its VALUE into the cache directory's ``env_hash`` (vLLM
    0.25.1; the module docstring) — sees ``None`` on every launch. Returns what happened: ``dropped`` (it
    was there and is gone), ``absent`` (nothing to drop), ``kept_in_process_engine`` (``VLLM_ENABLE_V1_
    MULTIPROCESSING=0``: this interpreter IS the API server, whose guards read the token AFTER the engine is
    built — untouched)."""
    if str(environ.get(ENGINE_IN_PROCESS_ENV, "1")).strip() == "0":
        return "kept_in_process_engine"
    return "dropped" if environ.pop(TOKEN_ENV, None) is not None else "absent"


#: the import-time call on THIS process's environment — vLLM imports this module in every worker before
#: ``init_device`` / ``load_model`` (``worker_base.py`` 261–263 @0.25.1), i.e. before the first compile
COMPILE_ENV_SCRUB = drop_api_key_from_compile_env(os.environ)

#: the engine-args string for vLLM's single ``worker_extension_cls`` slot
#: on a lora-native launch (``vllm_server.NATIVE_WORKER_EXTENSION_CLS``
#: equals this; a cross-workstream test pins them equal).
WORKER_EXTENSION_CLS = (
    "sisteer.capeval.serving.native_worker.NativeWorkerExtension")


class NativeWorkerExtension:
    """The plugin-free ``worker_extension_cls``. One readiness RPC, the
    three routing-recorder RPCs and the Nemotron backend report; plain
    JSON-able returns; every refusal inside the reports raises, which the dev
    RPC route surfaces as a 5xx and the launcher turns into a refusal naming
    the reason."""

    def sisteer_scheduler_config(self) -> Dict[str, Any]:
        # GPU-VERIFY: vllm_config unpacked onto the Worker (WorkerBase) and
        # worker.model_runner — the same two surfaces the lens extension's
        # RPC of the same name reads. Never gated on anything: it is a pure
        # config read the launcher runs during readiness, and a state that
        # cannot be read must fail the launch loudly.
        out = wr.scheduler_config_report(self.vllm_config)
        out.update(wr.kv_group_report(
            self.vllm_config, getattr(self, "model_runner", None)))
        out.update(wr.batch_invariant_report())
        out.update(wr.engine_mode_report(self.vllm_config))
        # THE FULL CAPTURE SURFACE (2026-09-16): the runner's own FULL graph
        # bookkeeping — V1's dispatcher keys or V2's manager graphs, whichever
        # runner this worker constructed (gemma / llama run V2, the hybrids V1:
        # Slurm 521717) — the config's capture list is the piecewise surface;
        # FULL decode graphs stop at max_num_seqs, and readiness holds their
        # largest size to it (the S=51 collapse)
        out.update(wr.cudagraph_capture_coverage_report(
            getattr(self, "model_runner", None)))
        out.update(wr.lora_config_report(self.vllm_config))
        out.update(wr.loaded_adapters_report(
            getattr(self, "model_runner", None)))
        # the EXECUTION side of the adapters (which model modules carry a
        # LoRA wrapper): the launcher holds every fingerprinted module key
        # to this set, so a --lora-target-modules restriction — which
        # filters the wrappers but not the CPU-side factors the fingerprint
        # hashes — cannot pass readiness as the full intervention.
        out.update(wr.lora_execution_modules_report(
            getattr(self, "model_runner", None)))
        # THE ROUTED-EXPERT LoRA BUFFER PATCH: armed in THIS process, the composed
        # routed rank, the patched classes — the launcher holds every MoE wrapper's buffer R
        # (lora_execution_moe_wrappers, above) to the composed value
        out.update(_native_moe_lora.report())
        # WHICH GPU model runner this worker constructed (V1 input_batch / V2
        # req_states+lora_state; gpu_worker.py:384-398 @0.25.1): the launcher
        # refuses an unpinned class and the attestation records the kind
        out.update(wr.model_runner_report(getattr(self, "model_runner", None)))
        return out

    def sisteer_mamba_backend_report(self) -> Dict[str, Any]:
        """The NEMOTRON FUSED-KERNEL BACKEND report (``mamba_backend.
        live_backend_report``), PORTED from the lens worker: every decoder
        layer whose mixer is a MambaMixer2,
        each kernel role resolved through THAT mixer's module globals as
        {module, qualname}, plus the initialised selective-state-update
        dispatch backend and its kernel. Facts only — the native standup core
        judges them against the reviewed reference contract
        (configs/capeval/mamba_backend_contract.json) and refuses a missing /
        substituted / unrecognised binding or a missing layer. GPU-VERIFY:
        ``self.model_runner.model`` is the loaded model on both GPU runners
        (V1 / V2). The import is lazy and stdlib-free of vllm: the report
        walks whatever model object the runner holds."""
        from sisteer.capeval.serving import mamba_backend as mb
        return mb.live_backend_report(getattr(self, "model_runner").model)

    def sisteer_residual_slot_report(self) -> Dict[str, Any]:
        """THE RESIDUAL SLOT report (``worker_reports.residual_slot_report``):
        whether the slot is armed in this worker,
        the op's fully qualified name, every patched decoder class with whether
        its forward is wrapped or inherited, the number of distinct decoder
        blocks that constructed with a boundary, the bank's geometry and
        counters, every loaded slot's payload facts (kind, op, layers,
        boundaries, mode, header width, skip_bos, coeff, fingerprint) keyed by
        ``lora_int_id``, whether the model runner is the one the bank bound at
        ``load_lora_model``, and the ledger's size and fault latch. Facts only —
        ``serving_attestation.residual_slot_problems`` judges them (step 3)."""
        return wr.residual_slot_report(getattr(self, "model_runner", None))

    def sisteer_take_request_failure(self, req_id: Any = None) -> Optional[Dict[str, Any]]:
        """Claim (pop) the residual slot's contained per-request steering
        failure(s) for ``req_id`` (the chat completion's own id — the EXTERNAL id;
        the ledger holds them under the scheduler's ENGINE ids, that id plus the
        pinned ``-<8 hex>`` suffix and a ``<i>_`` prefix per parallel-sampling
        child, and pops every one that belongs to it), or ``None`` —
        the lens worker's RPC of the same name, so the proxy's post-check
        (``_refuse_poisoned_lens_response``, extended to restricted-mode
        native routes in stage 2) reads one contract on both engines. Once the
        ledger has LATCHED (``REQUEST_FAILURES_CAP`` unclaimed entries, fact
        15) every claim returns ``{"fault": "ledger_overflow", ...}`` whatever
        the id, and an armed worker with no bank answers a fault too: a claim
        that cannot be judged must never read as "no failure". A payload, never
        an exception (the TP>1 reply-queue rule below)."""
        from sisteer.capeval.serving import native_residual as nr
        return nr.take_request_failure(req_id)

    # -- the LoRA routing recorder -------------------------------------------
    #
    # REFUSALS ARE PAYLOADS, NEVER EXCEPTIONS (transport): the
    # pinned MultiprocExecutor.collective_rpc reads the per-worker response queues
    # sequentially and raises on the FIRST worker FAILURE without draining the rest,
    # and the queues carry no RPC identifier — a refusal raised here on a TP>1 launch
    # leaves the other workers' replies queued and the owner's next valid call reads a
    # stale reply as its own. So the three methods return
    # {"ok": False, "refusal": <LORA_ROUTING_REFUSALS>, "detail", "live_epoch", ...}
    # for every EXPECTED refusal and the CLIENT reader refuses after collecting every
    # worker's payload. Unexpected internal errors — a surface drift, a bug — still
    # raise: those are not operational refusals and must fail loudly.

    def sisteer_arm_lora_routing_recorder(self, epoch: Any = None) -> Dict[str, Any]:
        """Arm the per-request LoRA routing recorder over THIS worker's model
        runner for the CLIENT's epoch ``epoch`` (the one string arg; every
        worker of the process group adopts the same id, so the launcher's
        per-worker reader sees agreeing payloads): idempotent for the live
        epoch; ``already_armed`` payload while a different one is armed;
        ``bad_epoch`` payload for an empty or malformed epoch. GPU-VERIFY:
        ``self.model_runner`` is the V1 or V2 GPU model runner
        (``gpu_worker.py:384-398`` @0.25.1); the recorder pins the surfaces it
        reads in its own docstring and records the runner's class."""
        # the recorder is INSTANCE state under the sisteer_ prefix (the class carries only
        # its four RPC methods: vLLM's injection asserts no attribute collides)
        rec = getattr(self, "sisteer_lora_routing_recorder", None)
        if rec is None:
            rec = wr.LoraRoutingRecorder(getattr(self, "model_runner", None))
            self.sisteer_lora_routing_recorder = rec
        try:
            return rec.arm(wr.decode_epoch_arg(epoch))
        except wr.LoraRoutingRefusal as e:
            return e.payload(runner_kind=rec.runner_kind, runner_class=rec.runner_class)

    def sisteer_lora_routing_observations(self, epoch: Any = None) -> Dict[str, Any]:
        """THE WHOLE EPOCH's recorded ``engine_request_id -> {lora_int_id,
        lora_name}`` map, the ``inconsistent`` bindings, the per-step scheduled
        request sets, the counters and ``overflow`` — for ``epoch`` (the id
        the control minted; a string over the RPC route). A worker that never
        armed answers ``not_armed``, another epoch ``epoch_mismatch``, a
        malformed epoch ``bad_epoch`` — as payloads the client refuses on, so
        a control can never read silence — or a neighbour's epoch — as its own
        evidence."""
        rec = getattr(self, "sisteer_lora_routing_recorder", None)
        try:
            ep = wr.decode_epoch_arg(epoch)
            if rec is None:
                raise wr.LoraRoutingRefusal(
                    "not_armed",
                    f"observations asked for epoch {ep!r} but this worker never armed a LoRA "
                    "routing recorder")
            return rec.observations_payload(ep)
        except wr.LoraRoutingRefusal as e:
            return e.payload(runner_kind=getattr(rec, "runner_kind", None),
                             runner_class=getattr(rec, "runner_class", None))

    def sisteer_disarm_lora_routing_recorder(self, epoch: Any = None) -> Dict[str, Any]:
        """Restore ``execute_model`` for ``epoch`` (``epoch_mismatch`` /
        ``not_armed`` / ``bad_epoch`` as payloads) and return the final
        summary."""
        rec = getattr(self, "sisteer_lora_routing_recorder", None)
        try:
            ep = wr.decode_epoch_arg(epoch)
            if rec is None:
                raise wr.LoraRoutingRefusal(
                    "not_armed",
                    f"disarm asked for epoch {ep!r} but this worker never armed a LoRA routing "
                    "recorder")
            return rec.disarm(ep)
        except wr.LoraRoutingRefusal as e:
            return e.payload(runner_kind=getattr(rec, "runner_kind", None),
                             runner_class=getattr(rec, "runner_class", None))
