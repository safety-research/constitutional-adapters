"""Server-mode vLLM launch wrapper.

Composes and launches ``vllm serve`` for the Petri audit:

- the two-SI pinned model + revision, bf16;
- ``--enable-lora`` + per-adapter ``--lora-modules <cell_id>=<dir>``
  (genuine adapters + the zero no-op adapter from ``lora_export``);
- the vllm-lens plugin + the worker-side object registry
  (``lens_steer``), wired via environment (the workers preload at
  start), and the steering extension itself
  (``--worker-extension-cls lens_worker.SisteerWorkerExtension``) named
  on the command line — vLLM's single extension slot is empty by
  default, so a launch that does not claim it serves the STOCK
  vllm-lens extension and steers nothing;
- ``--enforce-eager`` on every LENS launch (vllm-lens forces it
  process-wide, so the composer states it and refuses any attempt to
  contradict it); the LORA-NATIVE mode (``lora_native=True``, § below)
  composes ``--no-enforce-eager`` explicitly instead — CUDA graphs on,
  the plugin kept out via the launch env — and PREFIX
  CACHING ON BY DEFAULT (``--enable-prefix-caching``), the leakage
  question now SETTLED rather than deferred. The answer, read out of the
  pinned v0.25.1 block manager (vllm/v1/core/kv_cache_utils.py): KV
  blocks are keyed on the parent-hash chain + token ids + extra keys
  (LoRA NAME on every block; the request ``cache_salt`` on block 0,
  chained into every descendant) — the steering selector
  (``vllm_xargs``/``extra_args``) is in the key NOWHERE, and cached
  tokens SKIP the forward pass, so the lens hooks never fire on them: a
  cached block permanently carries whatever intervention computed it.
  Cross-cell reuse of an identical token prefix would therefore silently
  serve the WRONG intervention's KV. The isolation is the routing
  proxy's per-request ``cache_salt`` (proxy._rewrite): every leg gets
  its own cell-stable salt namespace, and a SELECTOR-CARRYING (lens)
  leg's namespace is decided by its cell's ATTESTED RESUMED-PREFILL
  POLICY (``lens_adapters.resumed_prefill_policy``, published per cell
  in the worker's install payload). A
  cell attested ``always`` — per-position ops at all positions in both
  phases, whose edit for a row is computed from that row alone — gets a
  cell-stable lens salt, and the worker serves and attests its resume
  as a RESUMED PREFILL. Everything else keeps the REQUEST-UNIQUE salt,
  so those prefills still never resume: ``conditional`` (the response
  mask, safe only when the cached region is header-free), ``never``, and
  any cell nothing was attested for. That is correctness of the op
  vocabulary, not only isolation — a prefill-phase broadcast op like
  alphasteer computes its edit from the LAST prompt token, so a prefix
  cached under a different-suffix same-cell prompt would carry a wrong
  edit, and the worker's refusal for those is engine-fatal by design,
  which is why the PROXY declines them a shared namespace rather than
  relying on it. The win lands where it is largest anyway: the
  UNHOOKED legs — base and the prompted baselines, whose ~5.9k-token
  CONSTITUTION prefix was the expensive part — and the LoRA arms
  (natively name-isolated), measured 1.33x at a 6.5k shared prefix.
  Phase-0 assertion 7 now PROVES the isolation on hardware instead of
  asserting the flag is off. ``prefix_caching=False`` composes the old
  OFF launch, and the default (``None``) resolves PER MODEL:
  hybrid-recurrent targets (Qwen3.5's mamba/GDN stack; NemotronH's
  Mamba2 stack) compose OFF unconditionally, because caching is
  structurally impossible for them here — see
  :func:`hybrid_recurrent_model`. Both spellings stay wrapper-owned, so
  a manifest can contradict neither. The Mamba SSM cache dtype
  (``--mamba-ssm-cache-dtype``) is a wrapper-owned OPT-IN for those same
  targets, composed only when a caller asks;
- the NO-CHUNKING scheduler geometry (``--max-num-seqs`` +
  ``--max-num-batched-tokens = max_num_seqs * max_model_len``, module
  constants): vLLM V1 forces chunked prefill on and its server default
  budget (8192 on H200) chunks long/concurrent prefills mid-prompt,
  which the plan lattice refuses — so the sizing is wrapper-owned, and
  for a lens launch the RESOLVED engine values are verified over the
  authenticated scheduler RPC at readiness (a silent clamp fails the
  launch; argv alone is never trusted). A lens-LESS launch composes the
  same flags but skips the runtime verification: with no hooks there is
  no plan lattice, so chunking cannot corrupt it — and it exposes no
  RPC route to ask through;
- the EXPLICIT synchronous-scheduler pin (``--no-async-scheduling``):
  OMITTING the flag is not a pin — the live production launch (Slurm
  job 194568) omitted it and the installed 0.25.1 resolved
  ``scheduler_cls`` to the ASYNC scheduler, which readiness correctly
  refused — so the composer states the class it wants on the command
  line, owns every spelling of the option in manifest flags, and the
  worker RPC attests the class that actually resolved;
- the pinned chat template (part of the harness identity) —
  and, for AGENT-modality launches only (``tool_calls=True``), the pinned
  tool-call parser. Generation-only serving (JB families, benign battery,
  gate canaries) sends no ``tools`` and composes NO parser: the pinned
  vLLM instantiates the parser class on EVERY /v1/chat/completions
  request, so a parser inapplicable to the served tokenizer is a
  per-request 500 that surfaces only after real generation spend
  (job 232411: ``llama3_json`` pinned for a gemma-4 server — no
  ``<|python_tag|>`` in that tokenizer — killed tasks 28-35 minutes in).
  An agent launch additionally holds the manifest's parser to the served
  FAMILY's known pin at compose time (:data:`TOOL_PARSER_BY_FAMILY`:
  ``llama3_json`` / ``gemma4`` / ``glm45`` for GLM-4.6V), so that class
  of mismatch dies in milliseconds, not minutes. Thinking-capable
  templates (Qwen3.5, GLM-4.6V) serve with the render pinned OFF
  (:data:`THINKING_OFF_TEMPLATE_KWARGS`) and, by default, NO reasoning
  parser — with no reasoning block in the completion a parser would only
  re-segment plain content. The ``--reasoning-parser`` surface stays
  wrapper-owned (never a manifest flag), and is composed ONLY through
  :func:`build_server_launch`'s explicit ``reasoning_parser`` opt-in, at
  the served family's verified pin
  (:data:`~sisteer.capeval.tinycap.contracts.REASONING_PARSER_BY_FAMILY`)
  — the thinking-ON standup of the tinycap matrix, whose clients DO send
  thinking-ON template kwargs per request and would otherwise read the
  model's scratchpad out of ``content``;
- the exact-replay mechanism: ``VLLM_BATCH_INVARIANT=1`` in the server
  env, OR single-request serialization at the proxy — an explicit,
  stated choice (:func:`build_server_launch`'s ``exact_replay_strategy``);
- a CLUSTER-INTERNAL bind only (no open port; the wildcard interface is
  refused).

Everything import-time is vLLM-free (:func:`build_server_launch` is pure
and CPU-tested); the actual launch (:func:`launch_server`) needs the
pinned GPU serving venv and is exercised by the §10 gate. vllm-lens
compatibility with the pinned vLLM is UNVERIFIED (plan R2) — the gate's
Phase-0 assertions establish it, and a bootstrap failure routes through
the recorded-unavailable path (``scripts/petri_serving_gate.py``), never
a silent skip.

THE LORA-NATIVE MODE. ``build_server_launch(...,
lora_native=True)`` composes a SECOND serving mode beside the lens one:
one ``vllm serve --enable-lora`` process with the vllm-lens plugin kept
OUT (``VLLM_PLUGINS=""`` + ``VLLM_LENS_DISABLE=1`` in ``launch.env`` —
the pinned ``envs.VLLM_PLUGINS`` splits the empty string to ``['']`` and
``load_plugins_by_group`` loads only names in that list) and CUDA graphs
ON (``--no-enforce-eager``, stated), serving the base model plus every
registered adapter as a vLLM native LoRA selected per request by the
``model`` name. It REFUSES a lens registry (the plugin would force
eager) and an empty adapter set; fits ``--max-lora-rank`` to the
registered adapters' declared ``r`` (:func:`fitted_max_lora_rank`,
legal caps per ``vllm/config/lora.py`` @0.25.1) instead of the legacy
literal 64; composes ``--max-cpu-loras = --max-loras`` (every adapter
resident); names the plugin-free ``native_worker.
NativeWorkerExtension`` in the single extension slot with the auth
middleware and the dev RPC route, so readiness can VERIFY what resolved
(``_verify_native_lora_state``: geometry + KV coverage as for lens, plus
``enforce_eager is False``, a non-NONE CUDA-graph mode, no
``vllm_lens`` import, the LoRA config, and every adapter's bf16
fingerprint bound PER SERVING NAME to its ``lora_int_id`` and its
``/v1/models`` root — ``serving_attestation.py``) and writes
``serving_attestation.json`` (``kind: "lora_native"``), never the lens
file. Every LENS launch composes byte-identically to before the mode
existed (a test pins the legacy lens+lora argv against a golden).
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from sisteer.capeval import manifest as cm
from sisteer.capeval.serving import serving_attestation as _sa
from sisteer.capeval.serving.serving_attestation import (  # noqa: F401
    SERVING_ATTESTATION_FILE)
from sisteer.capeval.serving.vllm_auth import TOKEN_ENV as API_KEY_ENV

try:  # launch-time-only dep; the composer below never needs it
    import vllm  # type: ignore  # noqa: F401
    HAVE_VLLM = True
except ImportError:
    HAVE_VLLM = False


class VllmServerError(RuntimeError):
    """A server-launch refusal — loud, never a silently-weakened flag."""



class KvCoverageShortfall(VllmServerError):
    """Readiness half 3 refused for exactly ONE reason: every pin of the
    no-chunking invariant resolved exactly, and the resolved KV cache backs
    fewer than ``max_num_seqs`` full-length sequences. The numbers ride on
    the exception so a caller that OPTED IN to a KV-derived geometry
    (``capctrl_serving_up --kv-derived-seqs on`` — the Modal launcher's every
    standup) can re-launch ONCE at ``S = floor(kv_backed_concurrency)``
    with the same window; a caller that did not opt in sees the same refusal
    (this IS a VllmServerError carrying the same message). Any other problem
    — a clamp, a long-prefill threshold, a speculative config, an unreadable
    report, or a shortfall BESIDE one of those on the same launch — raises
    the plain VllmServerError, never this, so nothing is ever derived from a
    report the invariant refused. The case: the per-model budgets were
    measured on the cluster's H200; a Modal H200 container reported 782,688
    KV tokens at L=16,384 against the 786,432-token budget (48 x 16,384) —
    47.77-way, so S=47 there."""

    def __init__(self, message: str, *, kv_backed_concurrency: float, kv_capacity_tokens: int,
                 basis: str, max_model_len: int, max_num_seqs: int):
        super().__init__(message)
        self.kv_backed_concurrency = float(kv_backed_concurrency)
        self.kv_capacity_tokens = int(kv_capacity_tokens)
        self.basis = str(basis)
        self.max_model_len = int(max_model_len)
        self.max_num_seqs = int(max_num_seqs)


class KvCoverage(NamedTuple):
    """The KV-coverage facts read off ONE worker's resolved report (readable
    whether or not they cover the geometry): the full-length concurrency the
    cache backs, the capacity in tokens, and which formula said so."""
    kv_backed_concurrency: float
    kv_capacity_tokens: int
    basis: str


#: :attr:`KvCoverage.basis` — a UNIFORM cache (one KV cache group: llama, gemma,
#: glm4v): ``kv_num_gpu_blocks x kv_block_size`` tokens, over ``max_model_len``
KV_BASIS_UNIFORM = "uniform: kv_num_gpu_blocks x kv_block_size"
#: ... and a HYBRID cache (several groups: Qwen3.5, Nemotron-H): the allocator's own
#: ``kv_max_concurrency`` (the block arithmetic misstates a hybrid cache's capacity)
KV_BASIS_HYBRID = "hybrid: the allocator's kv_max_concurrency"

#: env var recording the RESOLVED object-registry dir a lens launch was
#: composed against (provenance; the value the installer passes to the
#: workers over the install RPC).
LENS_REGISTRY_DIR_ENV = "PETRI_LENS_REGISTRY_DIR"
#: env var disabling the lens hook in UNHOOKED helper processes
#: (vllm-lens forces enforce_eager process-wide; helpers set this).
LENS_DISABLE_ENV = "VLLM_LENS_DISABLE"
#: the vLLM batch-invariant exact-replay switch.
BATCH_INVARIANT_ENV = "VLLM_BATCH_INVARIANT"
#: vLLM's dev-endpoint switch. The pinned 0.25.1 exposes its supported
#: server-mode collective_rpc route (POST /collective_rpc,
#: vllm/entrypoints/serve/dev/rpc/api_router.py) ONLY behind this env —
#: it is how the launcher invokes sisteer_install_steering inside every
#: worker. Acceptable here because the server is cluster-internal by
#: construction (the wildcard bind refuses above) and the audit is the
#: port's only client; the trade is stated rather than silent.
SERVER_DEV_MODE_ENV = "VLLM_SERVER_DEV_MODE"

#: the attestation the launcher writes next to the server log after the
#: install RPC round-trip validated every worker's report (LENS launches;
#: the lora-native mode writes the SEPARATE ``SERVING_ATTESTATION_FILE``,
#: re-exported above from serving_attestation.py — its lens-shaped
#: readers must never meet a lora-native record under this name).
STEERING_ATTESTATION_FILE = "steering_attestation.json"
#: the plugin-free worker extension a lora-native launch names in vLLM's
#: single ``--worker-extension-cls`` slot (``native_worker.
#: WORKER_EXTENSION_CLS`` — a cross-workstream test pins them equal).
#: It exposes ONLY ``sisteer_scheduler_config``; no install RPC, no hooks.
NATIVE_WORKER_EXTENSION_CLS = (
    "sisteer.capeval.serving.native_worker.NativeWorkerExtension")
#: THE RESIDUAL SLOT's launch-env flag: set
#: iff a registered adapter's ``adapter_config.json`` carries the
#: ``petri_residual_slot`` block (a PSR / vector cell's bank entry), so the
#: worker extension's import arms the slot; a LoRA-only launch's env — and
#: graph — stays byte-for-byte the certified one. Both strings are pinned
#: equal to ``native_residual.ENV_FLAG`` / ``CONFIG_BLOCK`` by test (this
#: composer is imported by the CPU-light proxy and must not import torch).
RESIDUAL_SLOT_ENV = "SISTEER_NATIVE_RESIDUAL_SLOT"
RESIDUAL_SLOT_CONFIG_BLOCK = "petri_residual_slot"
#: THE PAIRED WRITE (relu-differenced-native-qwen): set iff a registered adapter's
#: ``petri_residual_slot`` block carries the PAIRED op, so the worker's bank
#: allocates its second-gate buffers at construction (before any graph is
#: captured) and the op body takes the two-term path; a launch without a paired
#: payload leaves it unset and the bank / graph are the certified single-gate
#: ones. Both strings pinned equal to ``native_residual.ENV_FLAG_PAIRED`` /
#: ``OP_RELU_GATED_VECTOR_PAIR`` by test (torch-free here).
RESIDUAL_SLOT_PAIRED_ENV = "SISTEER_NATIVE_RESIDUAL_SLOT_PAIRED"
RESIDUAL_SLOT_PAIRED_OP = "relu_gated_vector_pair"
#: THE ROUTED-EXPERT LoRA BUFFER RANK: set iff the adapted
#: class wraps the routed experts (``MOE_EXPERTS_SUFFIX``), to the largest
#: routed rank the registered adapters declare (:func:`fitted_moe_lora_rank` —
#: their ``rank_pattern``, cross-checked against the factor files), so the worker
#: extension's import sizes the fused-MoE LoRA buffers at that R instead of the
#: server-wide ``--max-lora-rank`` (``E·R·(2d + 3I)·2 B`` per MoE layer and slot,
#: so the cap-to-routed ratio smaller). Pinned equal to ``native_moe_lora.ENV_FLAG``
#: by test (this composer is torch-free and must not import the worker-side
#: module's peers). An OPTION of the serving library: none of the paper models
#: routes experts.
MOE_LORA_RANK_ENV = "SISTEER_MOE_LORA_RANK"
#: the values ``--max-lora-rank`` accepts at the pinned 0.25.1
#: (``vllm/config/lora.py`` ``MaxLoRARanks = Literal[1, 8, 16, 32, 64,
#: 128, 256, 320, 512]``). The lora-native composer FITS the cap to the
#: registered adapters (:func:`fitted_max_lora_rank`): rank-4 lora_all and
#: rank-8 subtraction both fit 8; the legacy lens+lora path keeps its
#: literal 64 so it stays byte-identical.
LEGAL_LORA_RANK_CAPS = (1, 8, 16, 32, 64, 128, 256, 320, 512)
#: the collective_rpc method that installs the steering wiring
#: (lens_worker.SisteerWorkerExtension.sisteer_install_steering).
INSTALL_RPC_METHOD = "sisteer_install_steering"
#: the collective_rpc method reporting each worker's RESOLVED scheduler
#: geometry (lens_worker.scheduler_config_report) — the runtime half of
#: the no-chunking guarantee: argv alone cannot prove the engine did not
#: silently clamp the budget (the in-proc smoke asserts the same thing
#: off the live config).
SCHEDULER_RPC_METHOD = "sisteer_scheduler_config"
#: the auth middleware guarding the paths --api-key does NOT (the pinned
#: GUARDED_PREFIX is /v1|/v2|/inference only — /collective_rpc is open
#: without this). Passed via vLLM's supported --middleware hook.
AUTH_MIDDLEWARE_CLS = "sisteer.capeval.serving.vllm_auth.SisteerTokenAuthMiddleware"
#: bounded child shutdown: terminate -> wait this long -> kill -> wait.
CHILD_STOP_TIMEOUT_S = 10.0

_LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")

EXACT_REPLAY_STRATEGIES = ("batch_invariant", "serialize")

#: NO-CHUNKING SCHEDULER GEOMETRY. vLLM V1 forces chunked prefill ON
#: with no flag to disable it, and `vllm serve` on H100/H200-class GPUs
#: defaults to max_num_batched_tokens=8192 (v0.25.1
#: arg_utils.get_batch_defaults) — far below max_model_len, so any long
#: prompt (or two concurrent prefills) is split mid-prompt. The plan
#: lattice REFUSES chunk shapes for plan-needing arms (it cannot honor a
#: partial prompt), so an unsized server bricks on its first chunked
#: steered prefill. The composer therefore always pins
#: ``max_num_batched_tokens = max_num_seqs * max_model_len`` — with S
#: sequences of at most L tokens each, one step schedules at most S*L
#: tokens, so the budget can NEVER bind mid-prompt (the identical
#: argument, and the identical sizing, as the certified in-proc smoke).
#: max_num_seqs defaults per exact-replay strategy: under "serialize"
#: the PROXY holds a global lock, so at most one request is in flight
#: and 1 suffices (keeping the memory-profiling dummy forward small);
#: concurrent modes (batch_invariant / interleaved) get 24. The old
#: default of 8 was the consumer batch geometry the smoke's
#: shared-prefix throughput suite measured (SP_BATCH) — a value matched
#: to a measurement, never a constraint — and it left concurrency on
#: the table for the audit, where ~24 legs share one server.
#:
#: NEGATIVE RESULT (measured, jobs 229194 + 229390, two H200 nodes):
#: 32 x 16384 does NOT fit one H200 under the no-chunking identity.
#: The fixed costs are weights ~15 GiB + full-coverage KV 32 x 16384 x
#: 128 KiB/token = 64.0 GiB + the warmup/profiling peak of a forward
#: over the ENTIRE 524,288-token budget — which includes one MLP
#: intermediate of 524,288 x 14,336 x 2 B = 14.0 GiB — against a
#: 139.81 GiB card, and utilization cannot fix it because it is
#: zero-sum: vLLM sizes KV from what is left after the profiled peak,
#: so raising util grows KV and shrinks the very headroom the warmup
#: allocation needs. Bracketed empirically: util 0.95 under-covers KV
#: by 2,768 tokens (521,520 = 31.83x, the smoke's coverage gate
#: refused); util 0.97 covers KV (544,432 = 33.23x) and then OOMs by
#: 0.48 GiB on the 14.0 GiB warmup tensor. The window between is ~1%
#: wide and failed on two different nodes — do not retry 32 here. (The
#: BoN prompted engines' 64 x 8192 = the same 524,288 budget "served"
#: only in the boots-at-default-util sense: at 0.90 KV resolves 59.44
#: GiB = 486,944 tokens, silently UNDER 64-way full-length coverage.)
#:
#: RE-TESTED AT BI=0 (job 230089, 2026-08-28), after
#: exact_replay was dropped and production moved to
#: serving_mode=interleaved (no VLLM_BATCH_INVARIANT): SAME WALL —
#: 32 x 16384 is not reachable at ANY utilization, so the failure is
#: STRUCTURAL, not a batch-invariance kernel artifact. The
#: three-utilization bracket: 0.90 resolved KV 463,472 tokens (28.29x),
#: 0.95 resolved 520,736 (31.78x — 3,552 short of the 524,288 budget,
#: where BI=1 had resolved 521,520: essentially identical), and 0.97
#: could not init the lens-active engine at all. This CLOSES the
#: question: the wall does not move with the kernels, so nobody needs
#: to retry 32-way when some other serving setting changes.
#:
#: 24 x 16384 = 393,216 is the widest audit geometry that fits with
#: real slack — THE CERTIFIED PRODUCT — with KV 48.0 GiB + MLP
#: intermediate 10.5 GiB + weights ~15
#: GiB leaving ~25 GiB of margin, and the smoke's PETRI-geometry gate
#: (petri_steered_smoke PETRI_* constants, coupled to these defaults by
#: test_petri_geometry_cert) certifies it with resolved KV coverage
#: asserted in the worker — at the production BI=0: 574,064 tokens =
#: 35.04x at util 0.90 (job 230089; the BI=1 run 229597 resolved
#: 574,848 = 35.09x, the same number to within a block). The
#: no-chunking identity below holds at any
#: width; VRAM is what vLLM's own memory profiling verifies at launch
#: (the cache-isolation gate attests the resolved geometry live).
DEFAULT_MAX_NUM_SEQS = 24
DEFAULT_MAX_NUM_SEQS_SERIALIZE = 1
#: overflow safety for the derived product: it crosses into vLLM as an
#: int and downstream C code must never see it wrap (int32 ceiling).
_MAX_SCHEDULER_TOKENS = 2**31 - 1

#: THE REST OF THE NO-CHUNKING INVARIANT. A big budget alone is not
#: enough on the pinned 0.25.1 — two other scheduler mechanisms split
#: or extend scheduled work regardless of the budget, both read out of
#: the pinned source:
#:
#: - ``long_prefill_token_threshold`` caps how many prompt tokens of a
#:   "long" request are scheduled per step, i.e. it CHUNKS any prompt
#:   longer than itself even with an unbound budget; and
#:   ``max_num_partial_prefills > 1`` SILENTLY auto-sets it to 4% of
#:   max_model_len when it is 0 (SchedulerConfig.__post_init__ at the
#:   tag). So all three partial-prefill knobs are pinned to their safe
#:   values (0 / 1 / 1) ON THE COMMAND LINE and verified resolved.
#: - SPECULATIVE DECODING schedules draft+verify tokens, so a "decode"
#:   step carries q > 1 — a shape the plan lattice refuses (it is
#:   bit-for-bit a chunked-prefill continuation) — and it is also what
#:   makes ``max_num_scheduled_tokens`` diverge from the budget (the
#:   field's own docstring at the tag). Every 0.25.1 CLI surface that
#:   can enable it is wrapper-owned: ``--speculative-config`` (JSON;
#:   short alias ``-sc``, refused by the stray-token rule) and the
#:   ``--spec-method`` / ``--spec-model`` / ``--spec-tokens``
#:   shorthands (arg_utils at the tag). The worker RPC additionally
#:   attests the RESOLVED speculative state, which also covers the
#:   "speculators"-format-model auto-enable path that never touches a
#:   flag (maybe_override_with_speculators at the tag) — moot while
#:   ``--model`` is pinned, but attested anyway.
#:
#: Ownership is enforced against the pinned parser's REAL acceptance
#: rules (FlexibleArgumentParser at the tag): option names take ``_``
#: and ``-`` interchangeably (normalized before matching), a dotted
#: ``--opt.field`` form collapses into ``--opt`` (matched on the base
#: option), and ``--config <yaml>`` splices arbitrary engine args from
#: a file — so ``--config`` (and the platform ``--additional-config``)
#: are wrapper-owned outright.
_NO_CHUNK_SCHEDULER_PINS = {
    "long_prefill_token_threshold": 0,
    "max_num_partial_prefills": 1,
    "max_long_partial_prefills": 1,
}

#: the EXACT scheduler class every worker must attest. At the 0.25.1
#: tag ``SchedulerConfig.scheduler_cls`` defaults to ``None``, and
#: ``get_scheduler_cls`` resolves ``None`` to
#: ``vllm.v1.core.sched.scheduler.Scheduler`` — or to
#: ``vllm.v1.core.sched.async_scheduler.AsyncScheduler`` iff
#: ``async_scheduling`` is truthy. OMITTING THE FLAG IS NOT A PIN: the
#: live production launch (Slurm job 194568) composed no
#: async-scheduling flag, this wrapper assumed the omission meant the
#: synchronous Scheduler, and the installed 0.25.1 resolved
#: ``scheduler_cls`` to the AsyncScheduler anyway — readiness correctly
#: refused the attestation and the launch was blocked. The composer
#: therefore pins :data:`SYNC_SCHEDULING_FLAG` explicitly on EVERY
#: launch (the installed CLI generation renders ``bool | None`` engine
#: options with argparse.BooleanOptionalAction, so the negative form is
#: a real accepted flag), owns both spellings of the option against
#: manifest flags, and the worker RPC still attests the EFFECTIVE
#: resolved class — readiness refuses any other value, missing
#: attribute included. Anything else is a CUSTOM scheduler ("Using
#: custom scheduler class ... compatibility may not be maintained", the
#: tag's own warning) whose behavior the no-chunking invariant cannot
#: vouch for.
PINNED_DEFAULT_SCHEDULER_CLS = "vllm.v1.core.sched.scheduler.Scheduler"
#: the explicit synchronous-scheduling pin composed on every launch —
#: the argv half of the guarantee whose runtime half is the
#: ``scheduler_cls`` attestation above (job 194568: omission resolved
#: async; a default is never a pin).
SYNC_SCHEDULING_FLAG = "--no-async-scheduling"

#: vLLM's DEFAULT ceiling on the CUDA-graph capture list when no list and no
#: max are composed: ``min(2 * max_num_seqs, 512)`` (``config/vllm.py``
#: ``_set_cudagraph_sizes`` @0.25.1, ``decode_query_len`` 1 — the composer
#: refuses every speculative-decoding surface).
VLLM_DEFAULT_CUDAGRAPH_CAPTURE_CAP = 512


def cudagraph_capture_sizes_for(max_num_seqs: int) -> Tuple[int, ...]:
    """The CUDA-graph capture list a LORA-NATIVE launch composes for the
    decode width ``max_num_seqs`` (S): vLLM's own default list for S, plus S
    itself — sorted, de-duplicated.

    WHY S MUST BE A MEMBER (Modal H200, 2026-09-16, vLLM 0.25.1, Qwen3.5-4B
    native): the composed launch set ``--max-num-seqs 51`` (a geometry-derived
    S) and left the capture list to vLLM's default ``[1, 2, 4, 8, 16, ...,
    96]``. vLLM captures FULL decode graphs ONLY for capture sizes ``<=
    max_num_seqs * uniform_decode_query_len`` (``v1/cudagraph_dispatcher.py``
    ``initialize_cudagraph_keys`` @0.25.1: ``cudagraph_capture_sizes_for_
    decode = [x for x in sizes if x <= max_num_tokens ...]``) and pads every
    decode batch UP to the next capture size (``_bs_to_padded_graph_size``), so
    a 49-51 sequence decode step padded to 56, found no FULL key at 56 (56 >
    51, never captured) and fell to the PIECEWISE graph with the splitting ops
    (``vllm::qwen_gdn_attention_core``, attention) EAGER: the log read
    ``Profiling CUDA graph memory: PIECEWISE=30 (largest=96), FULL=18
    (largest=48)`` and per-sequence decode collapsed 6x (58 -> 9 tok/s/seq)
    while the attestation (``cudagraph_mode FULL_AND_PIECEWISE``, capture max
    96 >= 51) said graphs-on. Llama at 32/46 was fine: 32 IS a default size.
    With S in the list, a full-width decode pads to S itself, which has a FULL
    key; every narrower batch pads to a captured size <= S, also FULL.

    The default list is replicated here rather than read from vLLM (the
    composer runs in a process with no vLLM): ``[1, 2, 4]`` + multiples of 8
    below 256 + multiples of 16 from 256, every one ``<=`` the default cap
    ``min(2S, 512)`` (:data:`VLLM_DEFAULT_CUDAGRAPH_CAPTURE_CAP`). vLLM's own
    appending of ``max_num_batched_tokens`` when it fits the cap never fires
    here (the budget is ``S x max_model_len``). Passing the list EXPLICITLY
    makes vLLM take it verbatim (de-duplicated, sorted, filtered only against
    ``max_num_batched_tokens``) and pin ``max_cudagraph_capture_size`` to its
    largest entry; ``--max-cudagraph-capture-size`` is deliberately NOT
    composed beside it (vLLM raises when both are given and disagree). S is
    never clamped or moved: the attested geometry other readers pin stays
    exactly the caller's S, and readiness holds the RESOLVED FULL capture
    surface to it (``serving_attestation.cudagraph_coverage_problems``)."""
    if isinstance(max_num_seqs, bool) or not isinstance(max_num_seqs, int):
        raise VllmServerError(
            f"cudagraph_capture_sizes_for: max_num_seqs must be an int, got {max_num_seqs!r}")
    S = int(max_num_seqs)
    if S < 1:
        raise VllmServerError(f"cudagraph_capture_sizes_for: max_num_seqs {S} < 1")
    cap = min(2 * S, VLLM_DEFAULT_CUDAGRAPH_CAPTURE_CAP)
    sizes = {i for i in (1, 2, 4) if i <= cap}
    if cap >= 8:
        sizes.update(range(8, min(cap + 1, 256), 8))
    if cap >= 256:
        sizes.update(range(256, cap + 1, 16))
    sizes.add(S)
    return tuple(sorted(sizes))


#: launch options this WRAPPER owns — pinned and/or validated here, so a
#: manifest flag naming any of them (split or ``=`` form) could silently
#: override a supposedly non-negotiable setting while the recorded
#: identity keeps the pinned value. Every one of them refuses; the only
#: tolerated shape is an exact bare RESTATEMENT of an always-pinned flag
#: (:data:`_PIN_RESTATEMENTS`) — same token, same meaning, no value.
_WRAPPER_OWNED_OPTIONS = frozenset({
    "--model", "--served-model-name", "--revision", "--code-revision",
    "--tokenizer", "--tokenizer-revision", "--dtype", "--host", "--port",
    "--enforce-eager", "--no-enforce-eager",
    "--enable-prefix-caching", "--no-enable-prefix-caching",
    "--chat-template", "--default-chat-template-kwargs",
    "--tool-call-parser",
    # the hybrid-recurrent Mamba state dtype: a composed OPT-IN (the
    # mamba_ssm_cache_dtype parameter), never a manifest flag — a
    # manifest naming it would silently change the served numerics
    # behind the recorded launch identity.
    "--mamba-ssm-cache-dtype",
    # the server's DEFAULT sampling (what vLLM fills into every request
    # that omits a key): composed per model from a frozen contract row for
    # targets that ship no generation_config.json, never a manifest flag —
    # a manifest naming either would change every omit-params decode
    # behind the recorded launch identity.
    "--generation-config", "--override-generation-config",
    "--enable-auto-tool-choice", "--no-enable-auto-tool-choice",
    "--enable-lora", "--no-enable-lora", "--lora-modules",
    "--max-loras", "--max-lora-rank", "--max-model-len", "--seed",
    # the rest of the LoRA surface: the CPU
    # residency (composed = max_loras on a lora-native launch), the
    # adapter dtype (the fingerprint is defined on bf16 bits — a manifest
    # flipping it would silently change what the attestation hashes), and
    # the sharding toggle (both BooleanOptionalAction spellings). A manifest
    # could set every one of these silently before they were owned.
    "--max-cpu-loras", "--lora-dtype",
    "--fully-sharded-loras", "--no-fully-sharded-loras",
    # the deployment-time LoRA RESTRICTION: LoRAConfig.target_modules filters
    # which model modules get a LoRA execution wrapper (model_manager.
    # _match_target_modules @0.25.1) but NOT the CPU-side factors the worker
    # fingerprint hashes, so a manifest `--lora-target-modules` could attest
    # the full fitted intervention while a subset of layers is applied.
    # Owned, and COMPOSED by the launcher from the registered
    # adapters' own target_modules (the adapted class, `o_proj`): with no
    # restriction vLLM wraps EVERY linear projection and dispatches the LoRA
    # kernels on each (reset_lora zeroes the weights, the shrink/expand still
    # run whenever the batch has LoRA requests), so a gate_up_proj wrapper's
    # 20,480-wide packed output stride — not the adapted o_proj's 4,096 —
    # would have set the int32 kernel index ceiling. The launcher asserts
    # the resolved value EQUALS the composed set and holds every executing
    # wrapper's stride to lora_max_module_dim at readiness
    # (serving_attestation.execution_stride_problems).
    "--lora-target-modules",
    # the CUDA-graph capture surface: the lora-native mode composes the
    # capture LIST itself (cudagraph_capture_sizes_for: vLLM's default sizes
    # for S plus S, so the full-width decode batch has a FULL graph — the
    # 2026-09-16 S=51 collapse), leaves the mode to vLLM (FULL_AND_PIECEWISE
    # by default, attested as resolved) and never composes the max, so a
    # manifest must not reshape the graphs behind the recorded identity
    # either. The two REAL 0.25.1 option names are
    # `--cudagraph-capture-sizes` and `--max-cudagraph-capture-size`
    # (engine/arg_utils.py:1478-1482 at the tag); `--cuda-graph-sizes` was
    # a misspelling the pinned parser never registered (kept, harmless). A
    # capture max below max_num_seqs — or a FULL capture surface stopping
    # below it — keeps enforce_eager False and a non-NONE mode while the
    # dispatcher runs every larger decode batch EAGER / PIECEWISE — held at
    # readiness by serving_attestation.cudagraph_coverage_problems.
    "--cudagraph-capture-sizes", "--max-cudagraph-capture-size",
    "--cuda-graph-sizes", "--compilation-config",
    "--max-num-seqs", "--max-num-batched-tokens",
    # the rest of the no-chunking invariant (_NO_CHUNK_SCHEDULER_PINS):
    "--long-prefill-token-threshold", "--max-num-partial-prefills",
    "--max-long-partial-prefills",
    "--enable-chunked-prefill", "--no-enable-chunked-prefill",
    # every 0.25.1 speculative-decoding surface (q>1 decode steps —
    # lattice-refused shapes; -sc, the short alias, dies on the
    # stray-token rule below):
    "--speculative-config", "--spec-method", "--spec-model",
    "--spec-tokens",
    # arbitrary-bypass surfaces: a YAML that splices any engine arg, a
    # platform dict, and a replacement scheduler class:
    "--config", "--additional-config", "--scheduler-cls",
    # the scheduler-CLASS toggle (job 194568): async scheduling swaps
    # the stock synchronous Scheduler for AsyncScheduler wholesale, and
    # the composer pins the negative form — BooleanOptionalAction
    # registers BOTH option strings, so both spellings are owned (the
    # negative one is deliberately NOT a tolerated restatement: the
    # pair is one surface, the pin is already composed, and a manifest
    # has no business naming it in any form):
    "--async-scheduling", "--no-async-scheduling",
    # the REASONING-PARSER surface (--reasoning-parser and its plugin
    # loader, arg_utils at the tag). Thinking-capable targets (Qwen3.5,
    # GLM-4.6V) are served with the render pinned OFF
    # (THINKING_OFF_TEMPLATE_KWARGS): the pinned template already
    # prefills the closed <think></think> block INSIDE the assistant
    # header, so a completion carries no reasoning block for a parser
    # to split off — a parser here would only ever re-segment plain
    # content (the fits' H counts that closed block as header). The
    # thinking-mode decision is made ONCE, on the render; a manifest
    # must not make a second, contradictory one here, and omission is
    # not a pin (job 194568) — so the option is owned outright.
    # Ownership is what makes build_server_launch's `reasoning_parser`
    # opt-in safe: the ONE caller that flips the render thinking-ON per
    # request (the tinycap thinking-ON standup) also names the parser,
    # through the wrapper, at the family's verified pin — and no manifest
    # can contradict it or slip one in behind a thinking-OFF render.
    "--reasoning-parser", "--reasoning-parser-plugin",
    "--worker-extension-cls", "--api-key", "--middleware",
})
#: bare flags the composer ALWAYS emits — a manifest restating one is a
#: harmless documented duplicate, skipped rather than refused.
#: NB neither prefix-caching spelling is a tolerated restatement: the
#: setting is a composed CHOICE (prefix_caching=True by default), so a
#: manifest naming it in any form refuses via ownership above. The same
#: now holds for auto tool choice: it is a composed MODALITY choice
#: (``tool_calls`` — ON for agent launches, explicitly OFF for
#: generation launches), so a manifest naming either spelling refuses.
_PIN_RESTATEMENTS = ("--enforce-eager",)
#: ... and the lora-native mode tolerates NO eager restatement: it
#: composes ``--no-enforce-eager`` (graphs on), so a manifest's bare
#: ``--enforce-eager`` is not a duplicate of the pin, it CONTRADICTS it —
#: refused with a message naming the mode (callers strip it explicitly
#: and record the drop). Mode-aware via ``_manifest_flag_tokens(...,
#: lora_native=True)``; the lens tuple above keeps its shape.
_LORA_NATIVE_PIN_RESTATEMENTS: Tuple[str, ...] = ()

#: THE SERVER-DEFAULT SAMPLING POLICY: how
#: :func:`build_server_launch` treats the keys a REQUEST OMITS.
#:
#: * ``contract_override`` (the default; every launch to date, byte for byte): the
#:   contracts' vendor row for a target that ships no generation_config.json is
#:   composed as ``--override-generation-config`` (``_server_default_generation_config``;
#:   ``contracts.SERVER_DEFAULT_SAMPLING_BY_MODEL``), so an omitted key on a
#:   Qwen3.5 target resolves to 0.7 / 0.8 / 20 the way it resolves to the vendor file on
#:   gemma / GLM / Nemotron. Nothing is composed for a model the contracts do not pin.
#: * ``engine_defaults``: NO wrapper-composed override on the argv — the engine resolves
#:   an omitted key on its own: vLLM's neutral defaults (temperature 1.0, top_p 1.0,
#:   top_k off, min_p 0, penalties 0) for a target that ships no generation_config.json,
#:   the shipped file for one that does. This is what every historical Qwen Petri /
#:   BoN / misalignment anchor decoded at (their standups predate the pin), so a lane
#:   pairing new cells against those anchors serves under it, and verifies the resolved
#:   tuple from the server log (``scripts/sdfbase/record_resolved_sampler.py``,
#:   ``scripts/sdfbase/seedrep/anchor_sampler_preflight.py``) before any production request.
#:
#: The policy NAME is recorded on :attr:`ServerLaunch.server_default_policy`, in the
#: standups' ``endpoint.json`` / ``cells_served.json`` and in the proxy's ``/v1/models``
#: identity block, so a store's sampler provenance is readable without the server log.
#: An unknown policy refuses at compose time.
SERVER_DEFAULT_POLICIES: Tuple[str, ...] = ("contract_override", "engine_defaults")
SERVER_DEFAULT_POLICY_DEFAULT = "contract_override"


@dataclass(frozen=True)
class ServerLaunch:
    """A composed launch: argv + env additions (never ``--export=``-style
    env smuggling — the env dict is applied to the subprocess only).
    ``model`` is the served model id the readiness check validates the
    live ``/v1/models`` listing against (a stale process on the port must
    never satisfy readiness)."""

    cmd: Tuple[str, ...]
    env: Mapping[str, str]
    host: str
    port: int
    model: str
    #: the RESOLVED immutable registry dir of a lens launch, or None for
    #: a lens-less one. When set, :func:`launch_server` will not report
    #: readiness until every worker's sisteer_install_steering report
    #: validated against this registry's expected state.
    lens_registry_dir: Optional[str] = None
    #: the per-launch cryptographically random API token. SECRET, and
    #: kept out of every recorded surface on purpose: ``repr=False``
    #: (no log line shows it), never in :attr:`cmd` (argv is
    #: world-readable via /proc on a shared node), never in :attr:`env`
    #: (the proxy records that mapping as its launch attestation) —
    #: :func:`launch_server` injects it into the CHILD env alone, and
    #: authenticated clients read it from this field.
    api_key: Optional[str] = field(default=None, repr=False)
    #: the composed no-chunking scheduler geometry (module constants):
    #: what :func:`launch_server` verifies the RESOLVED engine config
    #: against over the authenticated scheduler RPC. ``None`` only on a
    #: hand-built launch — a LENS launch without them refuses at verify
    #: time (fail-closed), it is never quietly skipped.
    max_num_seqs: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None
    max_model_len: Optional[int] = None
    #: the composed prefix-caching state — verified RESOLVED in every
    #: worker at readiness (the runtime half; argv proves nothing).
    #: ``None`` only on a hand-built launch, which a lens verify refuses.
    prefix_caching: Optional[bool] = None
    #: the composed thinking-mode render pin: the exact JSON string handed
    #: to ``--default-chat-template-kwargs`` when the pinned template has a
    #: thinking switch (None otherwise). Recorded here so the gate and the
    #: tests can pin the composed value, not re-derive it.
    default_chat_template_kwargs: Optional[str] = None
    #: the composed Mamba SSM-state cache dtype (``--mamba-ssm-cache-dtype``),
    #: an OPT-IN for hybrid-recurrent targets only (None = not composed,
    #: the engine default; every qwen launch to date). Recorded so gates
    #: can pin what was composed — and, when composed, ATTESTED at
    #: readiness: :func:`expected_scheduler_state` expects the RESOLVED
    #: ``cache_config.mamba_ssm_cache_dtype`` (lens_worker.
    #: scheduler_config_report) to equal it, so a flag the engine ignored
    #: refuses readiness instead of serving bf16 state under a float32
    #: attestation (GPU-verified 2026-09-02 on Nemotron-3-Nano-4B).
    mamba_ssm_cache_dtype: Optional[str] = None
    #: bcb-bfcl-200-widen: the sampling CLAMP OVERRIDE record when the
    #: launch was composed under ``sampling_clamp_override`` — ``{from: 0.01,
    #: to: 0.001, why}`` (``contracts.SAMPLING_CLAMP_OVERRIDE``): the server
    #: runs through ``sisteer.capeval.serving.vllm_entry``, which lifts vLLM
    #: 0.25.1's ``_MAX_TEMP`` clamp so BFCL's 0.001 executes literally. None
    #: (every other launch): the stock ``vllm serve`` argv, byte for byte.
    #: The standup records it on ``endpoint.json`` and the driver hashes it
    #: into the served identity, so a server WITH the override never pools
    #: with one WITHOUT.
    sampling_clamp_override: Optional[Mapping[str, object]] = None
    #: the composed server-side sampling default: the exact JSON handed to
    #: ``--override-generation-config`` when the target ships no
    #: generation_config.json and the contracts pin a vendor row for it
    #: (``contracts.SERVER_DEFAULT_SAMPLING_BY_MODEL``); None otherwise —
    #: the model's own shipped file then governs, as it always has.
    override_generation_config: Optional[str] = None
    #: the SERVER-DEFAULT SAMPLING POLICY this launch was composed under
    #: (:data:`SERVER_DEFAULT_POLICIES`): ``contract_override`` (the default —
    #: the row above composed for a pinned model) or ``engine_defaults`` (no
    #: override composed; the engine's own resolution of an omitted key
    #: governs). Recorded, never inferred from the argv.
    server_default_policy: str = SERVER_DEFAULT_POLICY_DEFAULT
    #: WHAT holds the override in vLLM's child processes — recorded (unhashed) so a
    #: server whose children were shimmed is told apart from one relying on fork inheritance
    sampling_clamp_override_mechanism: Optional[str] = None
    #: THE LORA-NATIVE MODE (module docstring): True iff this launch was
    #: composed with ``lora_native=True`` — plugin out, ``--no-enforce-
    #: eager``, the native worker extension named. :func:`launch_server`
    #: then verifies the resolved engine mode + every adapter's
    #: fingerprint per name and writes ``serving_attestation.json``.
    lora_native: bool = False
    #: the registered adapter names in the ONE composed ``--lora-modules``
    #: flag's order (sorted) — the frontend's ``lora_int_id`` order, so
    #: position + 1 is the id the readiness check asserts per name.
    lora_modules: Tuple[str, ...] = ()
    #: name -> the adapter directory bound to it (resolved on a lora-native
    #: launch; as given on the legacy lens+lora path).
    lora_adapter_dirs: Mapping[str, str] = field(default_factory=dict)
    #: the composed LoRA config, attested resolved on a lora-native launch
    #: (``--max-loras``, the fitted ``--max-lora-rank``, ``--max-cpu-
    #: loras``); the legacy path records its literal cap and no CPU count.
    max_loras: Optional[int] = None
    max_lora_rank: Optional[int] = None
    max_cpu_loras: Optional[int] = None
    #: THE LORA KERNEL INDEX CEILING (``serving_attestation`` module
    #: constants; job 488932): the widest adapted module's in/out features
    #: over the registered adapters and the batched-token ceiling
    #: ``(2^31-1) // D`` the composer held ``max_num_seqs x max_model_len``
    #: to. None on a lens launch (those kernels never run) or a hand-built one.
    lora_max_module_dim: Optional[int] = None
    lora_kernel_index_ceiling: Optional[int] = None
    #: THE ADAPTED CLASS: the module suffixes the composed
    #: ``--lora-target-modules`` restricts the LoRA execution wrappers to —
    #: the union of every registered adapter's ``adapter_config.json``
    #: ``target_modules`` (one set, or the compose refuses; ``("o_proj",)`` for
    #: every admitted export). Readiness asserts the resolved LoRAConfig equals
    #: it and that no execution wrapper lies outside it. Empty on a lens
    #: launch or a hand-built one.
    lora_target_modules: Tuple[str, ...] = ()
    #: THE RESIDUAL SLOT (native-psr-serving): True iff a registered adapter
    #: carries a ``petri_residual_slot`` payload — the launch env then carries
    #: :data:`RESIDUAL_SLOT_ENV` and readiness holds every worker's
    #: ``sisteer_residual_slot_report`` to the payloads.
    residual_slot_armed: bool = False
    #: THE PAIRED WRITE: True iff a registered adapter's payload is the paired op
    #: (:data:`RESIDUAL_SLOT_PAIRED_OP`) — the launch env then carries
    #: :data:`RESIDUAL_SLOT_PAIRED_ENV` so the bank allocates its second gate
    #: term before the first forward. Implies ``residual_slot_armed``.
    residual_slot_paired: bool = False
    #: THE ROUTED-EXPERT LoRA BUFFER RANK: on a lora-native launch whose
    #: adapted class wraps the routed experts, the R the fused-MoE LoRA buffers
    #: are allocated at — the launch env carries :data:`MOE_LORA_RANK_ENV` and
    #: readiness holds every MoE wrapper's buffer R to it
    #: (:func:`validate_moe_lora_rank`); None when no expert wrapper is composed.
    moe_lora_rank: Optional[int] = None
    #: THE COMPOSED CUDA-GRAPH CAPTURE LIST (2026-09-16): on a lora-native
    #: launch, :func:`cudagraph_capture_sizes_for` ``(max_num_seqs)`` — vLLM's
    #: default sizes for S plus S itself, the exact ints handed to
    #: ``--cudagraph-capture-sizes``. Readiness holds every composed size to
    #: the RESOLVED ``cudagraph_capture_sizes`` and the RESOLVED FULL capture
    #: surface to S (``validate_cudagraph_coverage``); the standup records it
    #: on ``endpoint.json`` (``composed_cudagraph_capture_sizes``). Empty on a
    #: lens launch (eager: no graphs) or a hand-built one.
    cudagraph_capture_sizes: Tuple[int, ...] = ()
    #: the pinned revision, recorded so the lora-native attestation can
    #: carry it without re-parsing argv.
    revision: Optional[str] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def auth_headers(self) -> Dict[str, str]:
        """The Authorization header every client of this launch sends
        (readiness probe, install RPC, the routing proxy's backend)."""
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}


#: vllm-lens 1.2.1's entry-point name under ``[vllm.general_plugins]``
#: (verified against the published wheel's entry_points.txt). vLLM's
#: ``VLLM_PLUGINS`` allow-list filters by THIS name, never by the
#: distribution name — a mismatch loads no plugin and steers nothing.
LENS_PLUGIN_ENTRY_POINT = "activations"


#: The thinking-mode render pin for reasoning-template targets, exactly as
#: vLLM's own --default-chat-template-kwargs help documents for Qwen-class
#: models. THE FIT-SIDE TWIN: models.load_model pins
#: ``tok._sisteer_template_kwargs = {"enable_thinking": False}`` whenever
#: the chat template mentions ``enable_thinking`` — that render is what
#: derived every fitted object's ``response_header_len`` (Qwen3.5: the
#: empty ``<think></think>`` block, H=7 in the blobs; GLM-4.6V-Flash:
#: ``<|assistant|>\n<think></think>\n``, H=5). Serving the SAME
#: template (the hash check above pins its bytes to the fit's
#: template_hash) with thinking ON would misalign the response mask by
#: construction: token 1 of the "response" would be inside an open
#: ``<think>`` region the fit never saw, and every fitted cell would serve
#: a DIFFERENT intervention than it was fit as — while producing perfectly
#: plausible numbers. The composer therefore pins the render OFF whenever
#: the template carries the switch, keyed on the IDENTICAL signal the fit
#: side keys on, so fit and serve cannot drift apart.
THINKING_OFF_TEMPLATE_KWARGS = '{"enable_thinking": false}'
#: THE THINKING-ON RENDER PIN — the OPT-IN twin of the constant above, composed
#: ONLY when a standup asks for it (``build_server_launch(thinking_on=True)``:
#: ``scripts/sdfbase/mt_serving_up.py --thinking-on``). The value is set
#: EXPLICITLY even on templates whose switch defaults ON (Qwen3.5, Nemotron-3
#: — ``enable_thinking`` undefined means ON there), so the served render is
#: PINNED, never inherited from a template default that a revision could
#: flip. Serving fitted objects under this render is OUT OF SUPPORT by
#: construction (every object was fit thinking-OFF): the registry declares
#: the alternate assistant header the response locator finds the boundary
#: with (``lens_steer.REGISTRY_THINKING_HEADER_KEY``), the proxy admits only
#: DECLARED requests (``proxy.THINKING_ON_KEY``), and the standup refuses a
#: template whose ON header cannot be certified
#: (``chat_markers.thinking_on_certification``) — docs/served_thinking_on.md.
THINKING_ON_TEMPLATE_KWARGS = '{"enable_thinking": true}'


def hybrid_recurrent_model(model_id: str,
                           revision: Optional[str] = None) -> bool:
    """True when the target interleaves RECURRENT (mamba/GDN,
    ``linear_attention``) layers with full attention — the Qwen3.5 family
    here (8 full-attention + 24 gated-deltanet layers on both 9B and 4B).

    Serving consequence, measured on hardware (job 243180 phase C32,
    2026-08-29): PREFIX CACHING IS STRUCTURALLY IMPOSSIBLE for such a
    model on this stack — not merely off by default, and not fixable by
    flag tuning:

    * the only caching mode vLLM 0.25.1 offers a hybrid model is
      ``--mamba-cache-mode=align`` (qwen3_5.py refuses ``all`` outright),
      and
    * align mode asserts ``Chunked prefill is required`` at engine init,
      while THIS serving stack forbids chunked prefill as a CORRECTNESS
      requirement (the no-chunking identity above: steering hooks cannot
      survive chunk shapes — the plan lattice refuses them).

    The two constraints are mutually exclusive, so a hybrid target serves
    caching OFF, always. Do not "fix" a hybrid model's caching by
    enabling align mode — that trades a throughput feature for a silently
    broken steering path. The cost is small here: lens cells never got a
    cache win anyway (request-unique salts), so only prompted baselines
    lose the SI-prefix reuse.

    NOT hybrid in this sense: gemma-4's ``sliding_attention`` layers
    (sliding windows are KV-backed attention; gemma serves caching ON),
    and every layer_types-less dense model — llama, and GLM-4.6V-Flash
    (40 full-attention layers under ``text_config``, no ``layer_types``;
    its M-RoPE ``(3, T)`` positions are a worker-side concern, handled
    by lens_worker's row reduction, not a cache one).

    Also hybrid in this sense: NemotronH (Nemotron-3-Nano-4B, model_type
    ``nemotron_h`` — 21 Mamba2 + 17 MLP + 4 attention blocks). Its
    config.json carries no ``layer_types`` key, only a
    ``hybrid_override_pattern`` string, but transformers 5.14.1's
    NemotronHConfig maps ``layer_types`` onto its derived
    ``layers_block_type`` list, which names every Mamba2 block
    ``linear_attention`` (verified offline against the pinned revision
    dfaf35de) — so the SAME read below classifies it, with no
    model_type table. The vLLM consequence is the same too: the
    ``align`` caching mode is the only hybrid mode and needs chunked
    prefill (its hit granularity is one Mamba block, ~528 tokens by
    default, so short prompts would see 0% hits even if it were on).

    Detection is architecture-authoritative when the config resolves from
    the local cache (``layer_types`` containing ``linear_attention``,
    nested ``text_config`` handled). A REAL launch always resolves — the
    weights must be cached to serve — so the name fallback below exists
    only for synthetic model ids in CPU tests, never as a silent
    degradation of a live serve."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(
            model_id, revision=revision, local_files_only=True)
        txt = getattr(cfg, "text_config", cfg)
        layer_types = getattr(txt, "layer_types", None) or ()
        return "linear_attention" in tuple(layer_types)
    except Exception:
        lowered = model_id.lower()
        return ("qwen3_5" in lowered or "qwen3.5" in lowered
                or "nemotron-3-nano" in lowered or "nemotron_h" in lowered)


#: Per-family TOOL-CALL PARSER pins: the pinned vLLM 0.25.1
#: ``tool_parsers`` registry name whose grammar the served family's chat
#: template actually speaks, keyed on the HF config ``model_type``. Only
#: the AGENT modality composes a parser (``tool_calls=True``), and the
#: pinned server instantiates it per chat request, so a parser that does
#: not fit the served tokenizer is a healthy-looking server that 500s
#: once real traffic arrives (job 232411) — this table turns that into a
#: compose-time refusal for every family it knows. Entries and their
#: provenance:
#:
#: - ``llama`` -> ``llama3_json``: the audit manifest's pin (2026-08-18,
#:   against Meta's official Llama-3.1 template). NOT live-verified: the H11
#:   real-agent smoke was never run on Llama-3.1 (only Qwen h11_smoke.json
#:   records exist), and the NousResearch mirror's template the row serves
#:   since 2026-08-21 (hash b48c47f644389271) renders NO tool definitions
#:   — a BFCL request with and without tools is byte-identical — so the
#:   parser can initialize while the model never sees a tool
#:   (llama BFCL scored exactly the 56 irrelevance items in both
#:   cells). A tool-aware template (Meta's e10ca381b1ccc5cf, or
#:   vLLM's llama3.1_json example) plus a render check (tools appear;
#:   with != without) is the unblock; until then llama's BFCL leg is
#:   blocked in the lane roster.
#: - ``gemma4`` -> ``gemma4``: shipped by the pinned 0.25.1
#:   (tool_parsers registry: ``gemma4_engine_tool_parser``), verified to
#:   initialize against the pinned gemma-4-E4B-it tokenizer (manifest
#:   note).
#: - ``glm4v`` -> ``glm45``: GLM-4.6V-Flash. The 0.25.1 registry maps
#:   ``glm45`` to ``glm47_moe_tool_parser.Glm47MoeModelToolParser``, the
#:   GLM-4.5/4.6/4.7 family's ``<tool_call>{name}<arg_key>..<arg_value>..``
#:   grammar — the exact format the pinned GLM-4.6V-Flash chat template
#:   renders tool calls in (chat_template.jinja at revision 411bb4d7,
#:   read locally). The pairing is offline-verified against the tag's
#:   source; the live per-request instantiation is GPU-VERIFY like every
#:   other family's.
#:
#: A family NOT listed (Qwen3.5 — the qwen waves are generation-only and
#: no pairing was ever verified here) composes the manifest's parser
#: unchecked, exactly as before this table existed: unverifiable at
#: compose time, so the H11 live smoke stays the check. Add a family in
#: the same change that verifies its pairing; never guess one in.
TOOL_PARSER_BY_FAMILY: Mapping[str, str] = {
    # nemotron_h -> qwen3_xml (the bcb-bfcl-200-widen S5 parser probe PASSED both halves on the staged tree 8365992,
    # 2026-09-23: configs/capeval/tasks/parser_probe/nemotron_h_verdict_8365992.json; contracts.TOOL_PARSER_LIVE_PASS["nemotron_h"] gates the lane roster's bfcl_200 legs)
    "nemotron_h": "qwen3_xml",
    "llama": "llama3_json",
    "gemma4": "gemma4",
    "glm4v": "glm45",
    # qwen3_5 -> qwen3_xml: vLLM 0.25.1's
    # `Qwen3EngineToolParser` (registered `qwen3_xml` == `qwen3_coder`), VERIFIED in-lane by the
    # parser probe (scripts/bfcl_parser_probe.py): the offline fixtures
    # (`configs/capeval/tasks/tool_parser_fixtures.json`, schema 2 — eight definite cases exact, the
    # literal-null / undeclared-parameter cases admitted under EITHER the BFCL convention or the
    # parser's schema-faithful string; the family's template-rendered
    # spelling) through the REAL parser on the serving image. The LIVE half (a captured raw
    # completion replayed through a fresh parser, PROBE_ROWS qwen35_4b-f22) gates the lane roster's
    # `bfcl_200` legs, not this table. Nemotron (nemotron_h) stays OUT: its probe has not run.
    "qwen3_5": "qwen3_xml",
    # qwen3_5_moe -> qwen3_xml: the routed-experts member of the Qwen3.5 family (HF `model_type` of the
    # `Qwen3_5MoeForConditionalGeneration` checkpoints) renders tool calls through the SAME chat template family and
    # is parsed by vLLM 0.25.1's `Qwen3EngineToolParser` (`qwen3_xml`), verified live on a routed-experts checkpoint of
    # the family (offline fixtures 9/9, live replay 24 items / 21 called / 21 grammar-in-raw, 2026-10-03). Keyed on the
    # RAW model_type the way the serving path reads it off the config (no alias): the family-inherited qwen3_5 row
    # never stood for a MoE checkpoint. A family's BFCL leg still needs the family's live-PASS record
    # (contracts.TOOL_PARSER_LIVE_PASS); this row is the family's parser, not a model's certificate.
    "qwen3_5_moe": "qwen3_xml",
}

#: NAME markers for :func:`model_family`'s fallback — synthetic ids in
#: CPU tests only (a REAL launch resolves the cached config). Deliberately
#: narrow: exact family spellings, so an unrelated id resolves to None
#: (unchecked) rather than to a wrong family.
_FAMILY_NAME_MARKERS = (
    ("glm-4.6v", "glm4v"), ("glm4.6v", "glm4v"),
    ("glm-4.5v", "glm4v"), ("glm4.5v", "glm4v"),
    ("gemma-4", "gemma4"), ("gemma4", "gemma4"),
    ("llama-3", "llama"), ("llama3", "llama"),
)


def model_family(model_id: str, revision: Optional[str] = None
                 ) -> Optional[str]:
    """The served target's architecture family — the HF config's
    ``model_type`` (``llama``, ``gemma4``, ``glm4v``, ``qwen3_5``, ...)
    read from the LOCAL cache, the same architecture-authoritative shape
    as :func:`hybrid_recurrent_model`: a real launch always resolves (the
    weights must be cached to serve). The name fallback exists for
    synthetic model ids in CPU tests only (:data:`_FAMILY_NAME_MARKERS`);
    ``None`` means "unknown — nothing family-keyed can be checked"."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(
            model_id, revision=revision, local_files_only=True)
        model_type = getattr(cfg, "model_type", None)
        return str(model_type) if model_type else None
    except Exception:
        lowered = model_id.lower()
        for marker, family in _FAMILY_NAME_MARKERS:
            if marker in lowered:
                return family
        return None


#: THE TEXT-ONLY MULTIMODAL PIN. vLLM builds
#: the multimodal encoder infrastructure — and PROFILES it at startup — for any
#: target whose config carries a vision stack, even when every request is text:
#: the cluster's qwen35_4b OOMs (jobs 421951 / 422285) were exactly that profiling
#: step on host RAM. The cluster full22 manifests carry
#: ``--limit-mm-per-prompt '{"image":0,"video":0}'`` for glm46vflash AND qwen35_4b
#: (and ``common.sh`` refuses a GLM / qwen manifest without it); the tinycap standup
#: never passes those manifests (``--manifest`` travels only under ``--tool-calls``),
#: so the WRAPPER composes the pin itself when the target is multimodal
#: (``capctrl_serving_up --text-only-mm auto|on|off``) and records it in
#: endpoint.json (``text_only_mm``, unhashed: a launch flag, the identity geometry
#: stands). The flag is NOT wrapper-owned — a manifest may still carry it, and the
#: standup then records the manifest as its source instead of composing a duplicate.
TEXT_ONLY_MM_FLAG = "--limit-mm-per-prompt"
TEXT_ONLY_MM_FLAG_TOKENS: Tuple[str, str] = (TEXT_ONLY_MM_FLAG, '{"image":0,"video":0}')
#: The families the tokens are VERIFIED for in THIS tree: ``qwen3_5`` alone (Qwen3.5-4B —
#: its pinned config IS multimodal: ``Qwen3_5ForConditionalGeneration`` with a
#: ``vision_config`` and image/video token ids, verified with AutoConfig 2026-09-11). The
#: cluster full22 battery also served GLM-4.6V-Flash under these tokens, but glm4v is OUT
#: of this lane entirely (zero GLM footprint): listing it here would make the shared
#: standup's ``auto`` default compose a flag a GLM launch on main does not carry — a
#: behaviour change outside the lane. A GLM lane that wants the pin adds the entry with
#: its own verification, or passes ``--text-only-mm on``. gemma-4 is multimodal too
#: (``Gemma4ForConditionalGeneration``) but is NOT listed: the 57-cell wave served it
#: without the pin, the pin covers glm4v + qwen3_5 only, and the tokens name a ``video``
#: modality nobody verified against gemma-4's processor — under ``auto`` a multimodal
#: target of an unlisted family composes NOTHING and the record says why;
#: ``--text-only-mm on`` composes the generic tokens for it deliberately.
TEXT_ONLY_MM_TOKENS_BY_FAMILY: Mapping[str, Tuple[str, str]] = {
    "qwen3_5": TEXT_ONLY_MM_FLAG_TOKENS,
    # qwen3_5_moe (2026-10-03): the routed-experts member of the family — its checkpoints' configs ARE
    # multimodal (`Qwen3_5MoeForConditionalGeneration` with a `vision_config` and image / video token ids, the same
    # processor family as qwen3_5), and the SAME two tokens are what every standup of such a checkpoint served under.
    # Keyed on the RAW model_type, the way `capctrl_serving_up` / `native_standup` read it off the config (no alias in
    # the serving path): without this row a MoE standup that carries no manifest composes NOTHING under `auto` and the
    # launcher's full22 gate refuses `REC_TEXT_ONLY_MM_UNPINNED` by construction.
    "qwen3_5_moe": TEXT_ONLY_MM_FLAG_TOKENS,
}
TEXT_ONLY_MM_MODES: Tuple[str, ...] = ("auto", "on", "off")

#: NAME markers for :func:`multimodal_target`'s fallback — synthetic ids in CPU
#: tests only (a REAL launch resolves the cached config).
_MULTIMODAL_NAME_MARKERS = ("glm-4.6v", "glm4.6v", "glm-4.5v", "glm4.5v", "qwen3.5", "qwen3_5",
                            "gemma-4", "gemma4")


def multimodal_target(model_id: str, revision: Optional[str] = None) -> bool:
    """True when the served target's config carries a VISION stack (a ``vision_config``
    or an ``image_token_id`` — ``Glm4vForConditionalGeneration``,
    ``Qwen3_5ForConditionalGeneration``, ``Gemma4ForConditionalGeneration``), so vLLM
    would build and profile multimodal encoders for a text-only workload. Read from the
    LOCAL cache like :func:`hybrid_recurrent_model`; the name fallback serves synthetic
    ids in CPU tests only."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(
            model_id, revision=revision, local_files_only=True)
        return bool(getattr(cfg, "vision_config", None) is not None
                    or getattr(cfg, "image_token_id", None) is not None)
    except Exception:
        lowered = model_id.lower()
        return any(marker in lowered for marker in _MULTIMODAL_NAME_MARKERS)


def text_only_mm_in_cmd(cmd: Sequence[str]) -> Optional[str]:
    """The ``--limit-mm-per-prompt`` VALUE already in a composed argv (a manifest's), in
    split or ``=`` form; None when absent."""
    toks = list(cmd)
    for i, tok in enumerate(toks):
        if tok == TEXT_ONLY_MM_FLAG and i + 1 < len(toks):
            return str(toks[i + 1])
        if tok.startswith(TEXT_ONLY_MM_FLAG + "="):
            return tok.split("=", 1)[1]
    return None


def text_only_mm_decision(mode: str, *, model_id: str, revision: Optional[str],
                          family: Optional[str], cmd: Sequence[str]) -> Dict[str, Any]:
    """What the standup composes for the text-only pin, as a RECORD:
    ``{mode, multimodal, family, text_only_mm, tokens, source, reason}``.

    * ``off``: nothing composed; ``text_only_mm`` False unless the manifest already
      pinned it (then True, ``source: manifest``);
    * ``on``: the family's verified tokens, else the generic tokens — for ANY target
      (a non-multimodal target composes them harmlessly; recorded);
    * ``auto``: the family's verified tokens iff the target is multimodal AND the family
      is in :data:`TEXT_ONLY_MM_TOKENS_BY_FAMILY`; a multimodal target of another family
      composes nothing and says so (``reason``); a text-only target composes nothing.
    A manifest value already in ``cmd`` is never duplicated: it is recorded as the
    source, and ``text_only_mm`` is True iff that value is the zero-image/zero-video pin."""
    if mode not in TEXT_ONLY_MM_MODES:
        raise ValueError(f"--text-only-mm must be one of {TEXT_ONLY_MM_MODES}, got {mode!r}")
    mm = multimodal_target(model_id, revision)
    rec: Dict[str, Any] = {"mode": mode, "multimodal": mm, "family": family, "text_only_mm": False,
                           "tokens": None, "source": None, "reason": None}
    present = text_only_mm_in_cmd(cmd)
    if present is not None:
        rec.update(source="manifest", tokens=[TEXT_ONLY_MM_FLAG, present],
                   text_only_mm=_is_zero_mm_limit(present),
                   reason=None if _is_zero_mm_limit(present) else
                   f"the manifest pins {TEXT_ONLY_MM_FLAG} {present!r}, not the zero-image/zero-video pin")
        return rec
    if mode == "off":
        rec["reason"] = "text-only pin off (--text-only-mm off)"
        return rec
    verified = TEXT_ONLY_MM_TOKENS_BY_FAMILY.get(family or "")
    if mode == "on":
        rec.update(text_only_mm=True, source="standup",
                   tokens=list(verified or TEXT_ONLY_MM_FLAG_TOKENS),
                   reason=None if verified else
                   f"--text-only-mm on: generic tokens composed for family {family!r} (no verified entry)")
        return rec
    if not mm:
        rec["reason"] = "target is not multimodal (no vision_config / image_token_id): nothing to pin"
        return rec
    if verified is None:
        rec["reason"] = (f"multimodal target of family {family!r} has no verified text-only tokens "
                         f"({sorted(TEXT_ONLY_MM_TOKENS_BY_FAMILY)}); nothing composed under auto — "
                         "pass --text-only-mm on to compose the generic tokens deliberately")
        return rec
    rec.update(text_only_mm=True, source="standup", tokens=list(verified))
    return rec


def _is_zero_mm_limit(value: str) -> bool:
    try:
        parsed = json.loads(value)
    except ValueError:
        return False
    return (isinstance(parsed, dict) and parsed.get("image") == 0 and parsed.get("video") == 0)


def build_server_launch(
    spec: cm.PetriSpec,
    *,
    port: int,
    host: str = "127.0.0.1",
    lens_registry_dir: Optional[Path] = None,
    lora_modules: Mapping[str, Path] = (),
    exact_replay_strategy: Optional[str] = None,
    chat_template_path: Optional[Path] = None,
    max_model_len: int = 16384,
    max_num_seqs: Optional[int] = None,
    prefix_caching: Optional[bool] = None,
    tool_calls: bool = False,
    mamba_ssm_cache_dtype: Optional[str] = None,
    reasoning_parser: Optional[str] = None,
    lora_native: bool = False,
    sampling_clamp_override: Optional[float] = None,
    thinking_on: bool = False,
    server_default_policy: str = SERVER_DEFAULT_POLICY_DEFAULT,
) -> ServerLaunch:
    """Compose the ``vllm serve`` invocation from the validated PetriSpec.

    ``server_default_policy`` (:data:`SERVER_DEFAULT_POLICIES`; default
    ``contract_override`` = every launch to date, byte for byte) decides what
    an OMITTED request key resolves to on the served engine: under
    ``contract_override`` the contracts' vendor row is composed as
    ``--override-generation-config`` for a pinned model (Qwen3.5); under
    ``engine_defaults`` nothing is composed and the engine's own resolution
    governs (vLLM's neutral defaults where no generation_config.json ships —
    what the historical Qwen anchors decoded at). The name is recorded on
    :attr:`ServerLaunch.server_default_policy`; an unknown name refuses.

    ``lora_native`` selects the LORA-NATIVE serving mode (module
    docstring): REFUSES together with
    ``lens_registry_dir`` (the plugin would force eager process-wide) and
    with no ``lora_modules`` (a lora-native server with no adapter is a
    passthrough server nobody asked for); composes ``--no-enforce-eager``
    in place of ``--enforce-eager``, ``--enable-lora --max-loras N
    --max-lora-rank <fitted> --max-cpu-loras N`` with the cap fitted to
    the registered adapters' ``adapter_config.json`` (``r``), the ONE
    sorted ``--lora-modules`` flag with RESOLVED directories, the
    plugin-free :data:`NATIVE_WORKER_EXTENSION_CLS` + the auth middleware,
    and an env of ``PYTHONPATH`` (the extension must import in the
    child), ``VLLM_PLUGINS=""`` and ``VLLM_LENS_DISABLE=1`` (the plugin
    kept out, belt and braces) and the dev RPC switch — NO registry env.
    Under it NEITHER eager spelling is a tolerated manifest restatement.
    Every launch with ``lora_native=False`` (the default) composes
    byte-identically to before the mode existed.

    ``sampling_clamp_override`` (the standup's
    ``--sampling-clamp-override <T>``, passed by the tinycap launcher iff the
    run's sampling mode is ``protocol``): lift vLLM 0.25.1's ``_MAX_TEMP``
    clamp to ``T`` (0.001 — ``contracts.BFCL_PROTOCOL_TEMPERATURE``) so BFCL's
    reference temperature executes literally. The composed argv then runs
    the server through the wrapper entrypoint
    ``sisteer.capeval.serving.vllm_entry`` in the SERVING interpreter (the
    venv the ``vllm`` console script belongs to), with the live sisteer
    tree's ``src`` first on the child's ``PYTHONPATH`` — see
    :func:`apply_sampling_clamp_override`. ``None`` (every other launch)
    composes exactly what it did before the override existed, byte for byte.

    ``thinking_on`` (default ``False`` — every launch to date, byte for
    byte) selects the THINKING-ON render pin for a reasoning template:
    :data:`THINKING_ON_TEMPLATE_KWARGS` is composed in place of
    :data:`THINKING_OFF_TEMPLATE_KWARGS`, stated explicitly even where the
    template's own default is ON. REFUSED on a template with no thinking
    switch: the flag would be inert there, and a standup recorded as
    thinking-ON that renders thinking-OFF is exactly the mislabelling this
    composer exists to prevent. Everything else about the launch is
    unchanged; the fit/serve mask boundary this pin moves is the
    STANDUP's to declare (``lens_steer.REGISTRY_THINKING_HEADER_KEY`` +
    the proxy's ``petri_thinking_on`` gate — docs/served_thinking_on.md).
    The two modes are independent: ``thinking_on`` picks the template
    kwargs, ``lora_native`` picks the eager/plugin composition.

    ``reasoning_parser`` is the THINKING-ON opt-in (the only caller today
    is a ``capctrl_serving_up.py
    --reasoning-parser`` standup for the thinking-ON arm of the tinycap
    four-way matrix). Default ``None`` composes nothing, which is the
    only launch shape served to date: every other
    workload pins the render OFF (:data:`THINKING_OFF_TEMPLATE_KWARGS`),
    so a completion carries no reasoning block and a parser here would
    only re-segment plain content. When a run DOES ask the template for
    thinking (a per-request ``chat_template_kwargs`` override, which
    vLLM 0.25.1 merges over the server default), the parser stops being
    cosmetic: without it the thought text lands in ``content`` and every
    downstream extractor — ``ANSWER:``, the numeric match, the choice
    scorer, the code-fence split — reads the model's scratchpad as its
    answer. The flag stays wrapper-owned in
    :data:`_WRAPPER_OWNED_OPTIONS` (a manifest may never name it: the
    thinking decision is made once, by the caller that also sends the
    template kwargs), and the name is held at compose time to the served
    FAMILY's pin (:data:`REASONING_PARSER_BY_FAMILY`, the single
    definition in ``sisteer.capeval.tinycap.contracts``) exactly the way
    ``tool_calls`` holds ``spec.tool_parser`` — a family with no pin
    REFUSES rather than composing a guess, because an unregistered or
    wrong-grammar parser is a healthy-looking server whose every
    thinking-ON completion is silently mis-split.

    ``mamba_ssm_cache_dtype`` is an OPT-IN for hybrid-recurrent targets
    (NVIDIA's serving recommendation for Nemotron-3-Nano is ``float32``
    — the Mamba2 SSM state accumulates over the whole sequence, and bf16
    state drifts on long prompts): composed as the wrapper-owned
    ``--mamba-ssm-cache-dtype`` when given, refused on a non-hybrid
    target (the flag has no Mamba state to act on there), and never
    defaulted — every qwen hybrid launch to date composed nothing here,
    and a launch-identity change must be a deliberate per-model call.

    ``exact_replay_strategy`` is REQUIRED when ``spec.serving_mode ==
    "exact_replay"`` (one of :data:`EXACT_REPLAY_STRATEGIES` — an
    explicit, recorded choice) and REFUSED for ``interleaved``.
    ``chat_template_path`` is REQUIRED: the pinned template is part of
    the harness identity, so its exact bytes are hashed against
    ``spec.chat_template_hash`` before launch — absence or a mismatch
    refuses (vLLM's default/model template must never serve the audit).
    ``max_model_len`` defaults to 16k, and 16k is a FLOOR, not slack —
    do not cut it for concurrency (measured 2026-08-27, NousResearch
    tokenizer @ the audit pin): the CONSTITUTION_SI prefix alone is
    5,872 tokens (6,032 / 5,963 under the gemma-4 / Qwen3.5 tokenizers),
    the audit's target-side turns request ``max_tokens=1024`` (every
    audit log's target config carries it — the 2026-09-03 measurement
    pass corrected this prose, which used to say the runner set none;
    so vLLM 400-rejects any turn whose prompt exceeds 16384 - 1024 =
    15,360 tokens, and the canonical audits show 284 such rejections,
    0.5-1% of target calls, see docs/served_geometry_p99.md), and
    petri's own shipped example transcript for an audited behavior
    (delusion_sycophancy, a 40-turn audit) renders to ~19k tokens
    BEFORE any SI prefix. At the manifest's 15-turn budget a median
    audited conversation projects to ~7.5k (+5.9k for constitution
    cells ~= 13.4k), which 16k holds with tail risk only; the measured
    per-model p99 of prompt+completion runs 11.6k (llama8b) to 18.4k
    (qwen35_9b), so the qwen / long-SI cells need the window to go UP,
    not down (geometry.py's ``petri:<model>`` sub-entries record the
    sized windows, pending re-certification). An overflow does not fail
    loudly: the 400 surfaces to the auditor (inspect_petri
    ``surface_errors=True``) and the audit is judged on the truncated
    transcript — so size this to the turn budget, never down for free
    concurrency.

    ``max_num_seqs`` sizes the NO-CHUNKING scheduler geometry (module
    constants above): the composer always pins ``--max-num-seqs`` and
    ``--max-num-batched-tokens = max_num_seqs * max_model_len`` so the
    scheduler's token budget can never bind mid-prompt — the invariant
    the plan lattice depends on, sized exactly the way the certified
    in-proc smoke sizes it. ``None`` picks the strategy default
    (serialize -> 1, concurrent -> 24; 32 x 16384 measurably does not
    fit one H200 under no-chunking — see DEFAULT_MAX_NUM_SEQS's
    negative-result note); both options are wrapper-owned,
    so a manifest flag can never override them; and for a lens launch
    :func:`launch_server` verifies the RESOLVED engine values over the
    authenticated scheduler RPC (argv alone proves nothing).

    The scheduler CLASS is pinned the same way: every composed command
    carries :data:`SYNC_SCHEDULING_FLAG` (``--no-async-scheduling``),
    because omission is NOT a pin — the live 0.25.1 resolved the ASYNC
    scheduler with no flag given (job 194568) and readiness refused —
    and the RPC attests the resolved class
    (:data:`PINNED_DEFAULT_SCHEDULER_CLS`).

    ``tool_calls`` selects the serving MODALITY, stated explicitly on
    argv either way. ``False`` (the default — generation serving: JB
    attack families, the benign battery, the §10 gate canaries) pins
    auto tool choice OFF and composes no ``--tool-call-parser``: those
    workloads send no ``tools``, and the pinned vLLM instantiates the
    parser per chat request, so an inapplicable parser (llama3_json on a
    gemma-4 tokenizer) is a late per-request 500, never a launch
    failure. ``True`` (the AGENT modality — the petri audit's tool
    round-trips) composes ``spec.tool_parser``, which stays a harness-
    hash input and must be chosen PER TARGET MODEL (the pinned vLLM
    ships ``gemma4`` for gemma-4 targets and ``glm45`` for GLM-4.6V; the
    H11 real-agent smoke verifies the pairing live) — and is held, at
    compose time, to the served family's known pin
    (:data:`TOOL_PARSER_BY_FAMILY`) whenever the family is one this
    module knows: the job-232411 mismatch refuses here instead of 500ing
    per request half an hour into real generation."""
    cm.validate_petri_spec(spec)
    if host in ("0.0.0.0", "::", ""):
        raise VllmServerError(
            "refusing a wildcard/empty bind — the vLLM server is "
            "cluster-internal only (no open port; D0 constraint 5)")
    if lens_registry_dir is not None and host not in _LOOPBACK_HOSTS:
        raise VllmServerError(
            f"refusing to compose a LENS launch bound to {host!r} — a lens "
            "server mounts the dev RPC surface for its install, so it binds "
            f"loopback only ({_LOOPBACK_HOSTS}); loopback is still not "
            "tenant isolation on a shared node, which is why the token "
            "auth below is mandatory on top")
    if not isinstance(port, int) or not (0 < port < 65536):
        raise VllmServerError(f"invalid port {port!r}")
    if lora_native:
        # THE LORA-NATIVE MODE'S TWO COMPOSE-TIME REFUSALS (module docstring).
        if lens_registry_dir is not None:
            raise VllmServerError(
                "lora_native=True together with a lens_registry_dir — the "
                "vllm-lens plugin forces enforce_eager process-wide, so a "
                "lens registry on a lora-native launch would give back the "
                "eager floor the mode exists to escape (and a mixed server "
                "never runs lora-native, D2); serve PSR/lens kinds on a LENS "
                "launch and LoRA-only views on this one")
        if not lora_modules:
            raise VllmServerError(
                "lora_native=True with no lora_modules — a lora-native "
                "server with no adapter is a passthrough server nobody asked "
                "for; register the view's adapters (and the zero adapter)")
        if host not in _LOOPBACK_HOSTS:
            raise VllmServerError(
                f"refusing to compose a LORA-NATIVE launch bound to {host!r} "
                "— it mounts the dev RPC surface for its attestation, so it "
                f"binds loopback only ({_LOOPBACK_HOSTS}), with the token "
                "auth mandatory on top")

    # THE NO-CHUNKING GEOMETRY (module constants): resolve, validate,
    # and derive the product overflow-safely BEFORE anything is
    # composed — a geometry that cannot hold the invariant refuses here.
    if isinstance(max_model_len, bool) or not isinstance(max_model_len, int):
        raise VllmServerError(
            f"max_model_len must be an int, got {max_model_len!r}")
    if max_model_len < 1:
        raise VllmServerError(f"max_model_len {max_model_len} < 1")
    if max_num_seqs is None:
        max_num_seqs = (DEFAULT_MAX_NUM_SEQS_SERIALIZE
                        if exact_replay_strategy == "serialize"
                        else DEFAULT_MAX_NUM_SEQS)
    if isinstance(max_num_seqs, bool) or not isinstance(max_num_seqs, int):
        raise VllmServerError(
            f"max_num_seqs must be an int, got {max_num_seqs!r}")
    if max_num_seqs < 1:
        raise VllmServerError(f"max_num_seqs {max_num_seqs} < 1")
    max_num_batched_tokens = max_num_seqs * max_model_len  # exact (Py int)
    if max_num_batched_tokens > _MAX_SCHEDULER_TOKENS:
        raise VllmServerError(
            f"max_num_seqs * max_model_len == {max_num_seqs} * "
            f"{max_model_len} == {max_num_batched_tokens} exceeds "
            f"{_MAX_SCHEDULER_TOKENS} (the int32 ceiling downstream C "
            "code must never see wrapped) — the no-chunking product "
            "cannot be represented; shrink one factor")

    if spec.serving_mode == "exact_replay":
        if exact_replay_strategy not in EXACT_REPLAY_STRATEGIES:
            raise VllmServerError(
                f"serving_mode=exact_replay needs an explicit "
                f"exact_replay_strategy from {EXACT_REPLAY_STRATEGIES}, "
                f"got {exact_replay_strategy!r} — the mechanism is a "
                "stated choice, never a default (D0 constraint 3)")
    elif exact_replay_strategy is not None:
        raise VllmServerError(
            f"exact_replay_strategy={exact_replay_strategy!r} given but "
            f"serving_mode is {spec.serving_mode!r} — a replay mechanism "
            "on an interleaved run would mislabel every row's mode")

    # PREFIX CACHING IS MODEL-AWARE (hybrid_recurrent_model's docstring
    # carries the measured proof). ``None`` — the default every production
    # caller uses — resolves per architecture: ON for dense/sliding
    # targets (llama, gemma; salt-isolated at the proxy),
    # OFF for hybrid-recurrent targets (Qwen3.5), where caching is
    # STRUCTURALLY impossible on this stack: align mode requires chunked
    # prefill, the lens path refuses chunked prefill for correctness, and
    # there is no third option (job 243180 phase C32). An EXPLICIT True on
    # a hybrid target is a configuration error and refuses loudly rather
    # than composing a launch the engine cannot bring up honestly.
    hybrid_target = hybrid_recurrent_model(spec.target_model,
                                           spec.target_revision)
    if prefix_caching is None:
        prefix_caching = not hybrid_target
    elif prefix_caching and hybrid_target:
        raise VllmServerError(
            f"prefix_caching=True requested for hybrid-recurrent target "
            f"{spec.target_model!r} — prefix caching is structurally "
            "impossible for a mamba/GDN hybrid on this stack: vLLM's only "
            "hybrid caching mode (--mamba-cache-mode=align) requires "
            "chunked prefill, and the lens serving path forbids chunked "
            "prefill as a correctness requirement (the no-chunking "
            "identity). Compose with prefix_caching=None (resolves OFF) "
            "or False; do NOT try to enable align mode instead — see "
            "hybrid_recurrent_model's docstring for the measured proof "
            "(job 243180 phase C32, 2026-08-29)")
    if mamba_ssm_cache_dtype is not None:
        # an OPT-IN knob with meaning only where a Mamba/GDN state exists:
        # on a dense target it is a launch-identity change with no
        # behavior behind it — a configuration error, refused like an
        # explicit prefix_caching=True on a hybrid.
        if not isinstance(mamba_ssm_cache_dtype, str) \
                or not mamba_ssm_cache_dtype:
            raise VllmServerError(
                f"mamba_ssm_cache_dtype={mamba_ssm_cache_dtype!r} must be a "
                "non-empty dtype name (e.g. 'float32') or None")
        if not hybrid_target:
            raise VllmServerError(
                f"mamba_ssm_cache_dtype={mamba_ssm_cache_dtype!r} requested "
                f"for non-hybrid target {spec.target_model!r} — the Mamba "
                "SSM cache dtype only exists on a hybrid-recurrent stack "
                "(hybrid_recurrent_model); composing it on a dense target "
                "would change the launch identity for no behavior. Leave "
                "it None")

    # THE TOOL PARSER IS A PER-FAMILY PIN, held at compose time for every
    # family this module knows (TOOL_PARSER_BY_FAMILY): the pinned 0.25.1
    # instantiates the parser per chat request, so a parser foreign to
    # the served tokenizer is a late per-request 500 (job 232411), never
    # a launch failure — and only the agent modality composes one, so
    # only it is checked. An unknown family composes the manifest's pin
    # unchecked (unverifiable here; the H11 live smoke is the check).
    if tool_calls:
        family = model_family(spec.target_model, spec.target_revision)
        pinned_parser = (TOOL_PARSER_BY_FAMILY.get(family)
                         if family is not None else None)
        if pinned_parser is not None and spec.tool_parser != pinned_parser:
            raise VllmServerError(
                f"manifest tool_parser={spec.tool_parser!r} for target "
                f"{spec.target_model!r} (family {family!r}), whose pinned "
                f"vLLM 0.25.1 tool-call parser is {pinned_parser!r} "
                "(TOOL_PARSER_BY_FAMILY) — a parser foreign to the served "
                "tokenizer is a healthy-looking server that 500s on every "
                "chat request once real traffic arrives (job 232411: "
                "llama3_json on a gemma-4 tokenizer, 28-35 min in); pin "
                f"{pinned_parser!r} in the manifest, or extend the table "
                "in the same change that verifies a different pairing")

    # THE REASONING PARSER IS A PER-FAMILY PIN TOO (tinycap P7/P24), and a
    # STRICTER one than the tool parser's: only a caller that also sends
    # thinking-ON template kwargs has any business asking for it, only a
    # family with a verified pin may serve it, and an unknown family is
    # REFUSED rather than composed unchecked — a mis-segmented completion
    # does not 500, it quietly hands every extractor the scratchpad. The
    # mapping is imported LAZILY because contracts.py imports proxy.py,
    # which imports THIS module: a module-level import would close the
    # cycle. By call time every module is loaded, so this is a dict lookup.
    if reasoning_parser is not None:
        from sisteer.capeval.tinycap.contracts import REASONING_PARSER_BY_FAMILY

        if not isinstance(reasoning_parser, str) or not reasoning_parser.strip():
            raise VllmServerError(
                f"reasoning_parser={reasoning_parser!r} must be a non-empty "
                "parser name (e.g. 'gemma4') or None")
        rfamily = model_family(spec.target_model, spec.target_revision)
        pinned_reasoning = (REASONING_PARSER_BY_FAMILY.get(rfamily)
                            if rfamily is not None else None)
        if pinned_reasoning is None:
            raise VllmServerError(
                f"reasoning_parser={reasoning_parser!r} requested for target "
                f"{spec.target_model!r} (family {rfamily!r}), which has NO "
                "pinned reasoning parser (REASONING_PARSER_BY_FAMILY: "
                f"{sorted(REASONING_PARSER_BY_FAMILY)}) — refusing to serve "
                "thinking-ON with an unverified parser: a grammar that does "
                "not fit the served template leaves the thought text in "
                "'content', where every extractor reads it as the answer. "
                "Add the family in the same change that verifies the pairing")
        if reasoning_parser != pinned_reasoning:
            raise VllmServerError(
                f"reasoning_parser={reasoning_parser!r} for target "
                f"{spec.target_model!r} (family {rfamily!r}), whose pinned "
                f"reasoning parser is {pinned_reasoning!r} "
                "(REASONING_PARSER_BY_FAMILY) — pin the table's name, or "
                "extend the table in the same change that verifies a "
                "different pairing")

    # the pinned chat template: REQUIRED, and its exact bytes must hash to
    # the spec's declared identity — prompt construction must never drift
    # from what the rows claim.
    if chat_template_path is None:
        raise VllmServerError(
            "no chat_template_path — the pinned chat template is part of "
            "the harness identity (plan §5): the composer hashes its exact "
            "bytes against spec.chat_template_hash before launch; absence "
            "would silently serve vLLM's default/model template")
    template = Path(chat_template_path)
    if not template.is_file():
        raise VllmServerError(
            f"chat template {template} does not exist — refusing to launch "
            "with an unverifiable template (plan §5)")
    if spec.chat_template_hash == cm.PETRI_UNFROZEN:
        raise VllmServerError(
            "spec.chat_template_hash is UNFROZEN — freeze the pinned "
            "template hash before composing a launch (plan §5); an "
            "unverifiable template can never serve the audit")
    template_bytes = template.read_bytes()
    template_sha = cm.sha256_hex16(template_bytes)
    if template_sha != spec.chat_template_hash:
        raise VllmServerError(
            f"chat template {template} hashes to {template_sha}, spec "
            f"records {spec.chat_template_hash} — a stale or wrong template "
            "would change prompt construction while rows keep the pinned "
            "identity; refusing")

    # THINKING-MODE RENDER PIN (THINKING_OFF_TEMPLATE_KWARGS' docstring
    # carries the mask-misalignment proof): keyed on the IDENTICAL signal
    # the fit side keys on (models.load_model: `"enable_thinking" in
    # template`), so a template whose fits rendered thinking-off can never
    # silently serve thinking-on. Templates without the switch (llama,
    # gemma) compose nothing — the flag would be a no-op noise pin there.
    default_chat_template_kwargs: Optional[str] = None
    if not isinstance(thinking_on, bool):
        raise VllmServerError(
            f"thinking_on must be a bool, got {thinking_on!r} — the render "
            "regime is a stated choice, never a truthy coercion")
    if b"enable_thinking" in template_bytes:
        default_chat_template_kwargs = (THINKING_ON_TEMPLATE_KWARGS if thinking_on
                                        else THINKING_OFF_TEMPLATE_KWARGS)
    elif thinking_on:
        raise VllmServerError(
            f"thinking_on=True requested but the pinned chat template {template} "
            "carries no thinking switch (no 'enable_thinking' in its bytes) — the "
            "render pin would be inert, and a launch recorded as thinking-ON that "
            "renders thinking-OFF mislabels every row it serves; refusing")

    # SERVER-SIDE SAMPLING DEFAULT for targets that ship NO
    # generation_config.json (contracts.SERVER_DEFAULT_SAMPLING_BY_MODEL —
    # the Qwen3.5 vendor rows, 2026-09-11). vLLM fills every sampling
    # key a request omits from the model's generation_config.json, so the
    # three targets that ship one (gemma, GLM, Nemotron) decode their
    # omit-params requests — BoN, the 4,096-cap multi-turn re-runs, the
    # Petri target, agentic misalignment — at their vendor's top_p / top_k,
    # while Qwen decoded the identical request at vLLM's neutral defaults
    # (top_p 1.0, top_k off). This composes the vendor row as the server
    # default for exactly those models, so an omitted key resolves the same
    # way on every target. Imported lazily for the same import-cycle reason
    # as REASONING_PARSER_BY_FAMILY above. The premise is CHECKED, not
    # assumed: a snapshot that does ship the file (a future revision, a
    # re-uploaded one) would have its vendor file silently masked by the
    # override, so that case REFUSES until the table is re-verified.
    if server_default_policy not in SERVER_DEFAULT_POLICIES:
        raise VllmServerError(
            f"server_default_policy={server_default_policy!r} is not one of "
            f"{SERVER_DEFAULT_POLICIES} — the policy is a recorded choice, never a guess")
    if server_default_policy == "engine_defaults":
        # THE ENGINE-DEFAULTS POLICY (SERVER_DEFAULT_POLICIES): no wrapper-composed
        # override — an omitted key resolves the way the engine resolves it on its own
        # (vLLM's neutral defaults for a target that ships no generation_config.json;
        # the shipped file otherwise). Nothing is composed, so nothing can mask a vendor
        # file and the premise check above has nothing to check.
        override_generation_config = None
    else:
        override_generation_config = _server_default_generation_config(
            spec.target_model, spec.target_revision)

    cmd: List[str] = [
        "vllm", "serve", spec.target_model,
        "--revision", spec.target_revision,
        "--dtype", "bfloat16",  # the two-SI pin
        "--host", host,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        # NO-CHUNKING GEOMETRY (module constants): the budget equals
        # max_num_seqs * max_model_len, so it can never bind mid-prompt
        # — vLLM V1 forces chunked prefill on, and the plan lattice
        # refuses chunk shapes, so an unsized default (8192 on H200)
        # would brick the first long steered prefill. Wrapper-owned;
        # launch_server verifies the RESOLVED values for lens launches.
        "--max-num-seqs", str(max_num_seqs),
        "--max-num-batched-tokens", str(max_num_batched_tokens),
        # ... and the sizing is not the whole invariant: the threshold
        # CHUNKS long prompts regardless of the budget, and
        # max_num_partial_prefills > 1 silently auto-sets it (see
        # _NO_CHUNK_SCHEDULER_PINS) — pinned explicitly so an upstream
        # default drift can never re-enable either.
        "--long-prefill-token-threshold",
        str(_NO_CHUNK_SCHEDULER_PINS["long_prefill_token_threshold"]),
        "--max-num-partial-prefills",
        str(_NO_CHUNK_SCHEDULER_PINS["max_num_partial_prefills"]),
        "--max-long-partial-prefills",
        str(_NO_CHUNK_SCHEDULER_PINS["max_long_partial_prefills"]),
        # THE SCHEDULER CLASS IS PINNED, NOT DEFAULTED (job 194568):
        # with no async-scheduling flag composed, the installed 0.25.1
        # resolved scheduler_cls to AsyncScheduler and readiness
        # correctly refused the attestation — omission is not a pin.
        # The installed CLI generation renders bool|None options with
        # argparse.BooleanOptionalAction, so this negative form is a
        # real flag; the worker RPC attests the class that actually
        # resolved (PINNED_DEFAULT_SCHEDULER_CLS, the runtime half).
        SYNC_SCHEDULING_FLAG,
        # THE EAGER/GRAPH STATE IS STATED EITHER WAY (module docstring):
        # every LENS launch pins eager (vllm-lens forces it process-wide);
        # the LORA-NATIVE mode states the opposite explicitly — CUDA
        # graphs on, with the plugin kept out via the env below — and
        # readiness ATTESTS the resolved mode (enforce_eager False, a
        # non-NONE cudagraph mode) rather than trusting this token.
        ("--no-enforce-eager" if lora_native else "--enforce-eager"),
        # PREFIX CACHING: resolved per model above — ON for dense/
        # sliding targets (leakage settled by per-request cache_salt
        # isolation at the proxy, module docstring), OFF structurally
        # for hybrid-recurrent targets (hybrid_recurrent_model);
        # prefix_caching=False composes the legacy OFF launch anywhere.
        # Both spellings are wrapper-owned either way.
        ("--enable-prefix-caching" if prefix_caching
         else "--no-enable-prefix-caching"),
        # MAMBA SSM-STATE DTYPE: an opt-in composed ONLY on a hybrid-
        # recurrent target when the caller asks (validated above);
        # absent otherwise — the engine default, which every prior
        # hybrid launch ran. Wrapper-owned either way.
        *(("--mamba-ssm-cache-dtype", mamba_ssm_cache_dtype)
          if mamba_ssm_cache_dtype is not None else ()),
        # THINKING-MODE RENDER PIN (see the derivation above / the module
        # constant's docstring). Wrapper-owned; absent for templates with
        # no thinking switch.
        *(("--default-chat-template-kwargs", default_chat_template_kwargs)
          if default_chat_template_kwargs is not None else ()),
        # SERVER-SIDE SAMPLING DEFAULT (derivation above): wrapper-owned,
        # composed only for a target the contracts pin a row for; absent
        # otherwise, so every launch to date keeps its identity.
        *(("--override-generation-config", override_generation_config)
          if override_generation_config is not None else ()),
        # TOOL CALLS ARE A MODALITY CHOICE, stated explicitly either way
        # (omission is not a pin — job 194568's scheduler lesson). The
        # AGENT modality (``tool_calls=True``: the petri audit's tool
        # round-trips; the H11 real-agent smoke verifies the parser)
        # composes the manifest's pinned parser. GENERATION serving — the
        # default: JB attack families, the benign battery, the §10 gate
        # canaries — sends no ``tools`` and pins auto tool choice OFF
        # with no parser: the pinned vLLM 0.25.1 instantiates the parser
        # class on EVERY /v1/chat/completions request
        # (entrypoints/openai/chat_completion/serving.py, read at the
        # tag), so an inapplicable parser is a healthy-looking server
        # that 500s per chat request only once one arrives — job 232411
        # served a gemma-4 target for 28-35 min of real generation, then
        # died on llama3_json's missing '<|python_tag|>'. A parser is a
        # per-model pin; the only workload that needs one is the agent
        # modality, so only that modality composes it. Both spellings of
        # the toggle stay wrapper-owned either way.
        *(("--enable-auto-tool-choice", "--tool-call-parser",
           spec.tool_parser)
          if tool_calls else ("--no-enable-auto-tool-choice",)),
        # THE REASONING PARSER: absent by default (no reasoning block to
        # split — the render is pinned OFF above), composed only when the
        # caller explicitly asks for the thinking-ON arm and only at the
        # served family's verified pin (validated above). Unlike the tool
        # toggle there is no negative spelling to state: vLLM's default IS
        # "no reasoning parser", and that default is what every launch
        # before this knob composed — so absence here is the same launch
        # identity, not an unstated choice.
        *(("--reasoning-parser", reasoning_parser)
          if reasoning_parser is not None else ()),
        "--chat-template", str(template),
    ]
    # THE CUDA-GRAPH CAPTURE LIST (lora-native only; cudagraph_capture_sizes_for
    # carries the 2026-09-16 S=51 incident): vLLM's default sizes for S plus S
    # itself, so the full-width decode step — and every narrower one — pads to
    # a size that HAS a FULL graph. vLLM captures FULL decode graphs only for
    # capture sizes <= max_num_seqs, so a default list that skips S leaves the
    # widest decode batches PIECEWISE with attention eager under a graphs-on
    # attestation. nargs="+" type=int at the pinned parser (arg_utils.
    # collection_to_kwargs for list[int]): one flag, the sorted ints as its
    # values. The max is NOT composed beside it (vLLM raises when the two
    # disagree; it pins the max to the list's largest entry itself). A lens
    # launch is eager and composes nothing here — its argv stays byte-identical.
    capture_sizes: Tuple[int, ...] = ()
    if lora_native:
        capture_sizes = cudagraph_capture_sizes_for(max_num_seqs)
        cmd += ["--cudagraph-capture-sizes", *[str(n) for n in capture_sizes]]
    lora_names: Tuple[str, ...] = ()
    lora_dirs: Dict[str, str] = {}
    max_rank: Optional[int] = None
    max_cpu_loras: Optional[int] = None
    lora_max_dim: Optional[int] = None
    kernel_ceiling: Optional[int] = None
    lora_target_modules: Tuple[str, ...] = ()
    residual_slot_armed = False
    residual_slot_paired = False
    moe_lora_rank: Optional[int] = None
    if lora_modules:
        registered = sorted(dict(lora_modules).items())
        for name, _path in registered:
            if not name:
                raise VllmServerError("empty LoRA adapter name")
        if lora_native:
            # THE FITTED RANK CAP + CPU RESIDENCY: the cap is
            # the smallest legal value >= the largest declared r across
            # the registered adapters' adapter_config.json (read here, so
            # an unregistrable adapter dies at compose time), and every
            # adapter is CPU-resident too (--max-cpu-loras = --max-loras,
            # vLLM's own default made explicit and owned). Directories
            # are RESOLVED so the recorded name -> dir binding, the
            # /v1/models root check and the fingerprint all read one path.
            lora_dirs = {name: str(Path(path).resolve())
                         for name, path in registered}
            max_rank = fitted_max_lora_rank(
                Path(d) for d in lora_dirs.values())
            max_cpu_loras = len(registered)
            # THE ADAPTED CLASS -> --lora-target-modules: with
            # LoRAConfig.target_modules None, vLLM 0.25.1 wraps EVERY linear
            # projection of the model with a LoRA layer and dispatches the
            # shrink/expand kernels on each whenever the batch carries LoRA
            # requests — an adapter lacking a module's factors merely gets
            # reset_lora() (zeroed buffers, kernels still run at that
            # wrapper's stride). gemma-4-E4B's packed gate_up_proj output
            # stride is 20,480 (2 x 10,240), five times the adapted o_proj's
            # 4,096, so the int32 index ceiling below would have been sized
            # from the wrong module. The launch therefore restricts the
            # wrappers to the ONE suffix set every registered adapter adapts
            # (read off adapter_config.json here; disagreement or an unstated
            # set refuses), asserts the resolved value equals it, and holds
            # every executing wrapper's stride to D at readiness.
            lora_target_modules = _adapted_module_class(lora_dirs)
            # THE RESIDUAL SLOT (native-psr-serving): armed iff a registered adapter
            # carries a payload — the same adapter_config.json read as the class above
            residual_slot_armed = _residual_slot_armed(lora_dirs)
            # THE PAIRED WRITE: armed for the second gate term iff a payload is the paired op
            residual_slot_paired = _residual_slot_paired(lora_dirs)
            # THE ROUTED-EXPERT LoRA BUFFER RANK: iff the class wraps the
            # routed experts, the largest routed rank the registered adapters
            # declare — the worker then allocates the fused-MoE LoRA buffers at
            # it, never at the dense cap (native_moe_lora's module docstring)
            if MOE_EXPERTS_SUFFIX in lora_target_modules:
                moe_lora_rank = fitted_moe_lora_rank(
                    Path(d) for d in lora_dirs.values())
                if moe_lora_rank > max_rank:
                    raise VllmServerError(
                        f"the registered adapters' routed-expert rank {moe_lora_rank} exceeds the fitted "
                        f"--max-lora-rank {max_rank}: a routed rank is at most the adapter's own r "
                        "(lora_modules.per_expert_rank), so a declaration above the cap is a malformed "
                        "adapter; refusing")
            # THE LORA KERNEL INDEX CEILING (serving_attestation module
            # constants; gemma canary job 488932): the pinned LoRA Triton
            # kernels index the warmup's full-budget batch with int32
            # offsets, so rows x the widest adapted module must stay under
            # 2^31-1. Read off every registered adapter's safetensors header
            # here — the same read that fits the rank cap — and REFUSED, never
            # clamped: the geometry is the caller's to lower (the standup and
            # the certification driver do, under the named reason).
            try:
                dims = {name: _sa.adapter_module_dims(Path(d))
                        for name, d in lora_dirs.items()}
                lora_max_dim, widest = _sa.max_module_dim(dims)
                kernel_ceiling = _sa.lora_kernel_index_ceiling(lora_max_dim)
            except _sa.ServingAttestationError as e:
                raise VllmServerError(
                    f"cannot size the lora-native kernel index ceiling — the registered "
                    f"adapters' factors could not be read: {e}") from e
            if max_num_batched_tokens > kernel_ceiling:
                raise VllmServerError(
                    f"refusing to compose a LORA-NATIVE launch at max_num_seqs={max_num_seqs} x "
                    f"max_model_len={max_model_len} = {max_num_batched_tokens:,} batched tokens: "
                    f"the widest adapted module of the registered adapters has {lora_max_dim} "
                    f"in/out features ({widest}), so the warmup's full-budget dummy run would "
                    f"index {max_num_batched_tokens:,} x {lora_max_dim} = "
                    f"{max_num_batched_tokens * lora_max_dim:,} elements > 2^31-1 = "
                    f"{_sa.LORA_KERNEL_INDEX_LIMIT:,} — the pinned vLLM 0.25.1 LoRA Triton kernels "
                    "(vllm/lora/ops/triton_ops/lora_shrink_op.py, lora_expand_op.py) compute their "
                    "pointer offsets in int32, and the overflow is 'CUDA error: an illegal memory "
                    "access was encountered' at kernel_warmup -> flashinfer_autotune -> _dummy_run "
                    "(gemma certification canary, job 488932). The "
                    f"{_sa.LORA_KERNEL_CEILING_KEY} is {kernel_ceiling:,} tokens ((2^31-1) // "
                    f"{lora_max_dim}); the largest legal max_num_seqs at max_model_len="
                    f"{max_model_len} is {kernel_ceiling // max_model_len}. Lower S under "
                    f"geometry_override_reason={_sa.LORA_KERNEL_CEILING_REASON!r} (the standup and "
                    "the certification driver do so before their first attempt); the composer "
                    "never clamps")
            cmd += ["--enable-lora",
                    "--max-loras", str(len(registered)),
                    "--max-lora-rank", str(max_rank),
                    "--max-cpu-loras", str(max_cpu_loras),
                    # nargs="+" at the pinned parser (arg_utils.
                    # collection_to_kwargs for list[str]): one flag, the
                    # sorted suffixes as its values
                    "--lora-target-modules", *lora_target_modules]
            if MOE_EXPERTS_SUFFIX in lora_target_modules:
                # THE FUSED-MoE LoRA KERNEL GATE: an adapter
                # class that reaches the routed experts is served by vLLM
                # 0.25.1's FusedMoE3DWithLoRA, whose __init__ asserts the
                # MoE kernel `supports_lora()` — "For unquantized MoE, set
                # moe_backend='triton' or moe_backend='auto'"
                # (vllm/lora/layers/fused_moe.py:86-92); only the Triton
                # (and Marlin / OAI-Triton) experts mix in LoRAExpertsMixin.
                # `auto` does pick Triton once LoRA is enabled
                # (model_executor/layers/fused_moe/oracle/unquantized.py:
                # 177-180: `if moe_config.is_lora_enabled: return TRITON`),
                # but the launch states the kernel it depends on rather than
                # leaning on the oracle's LoRA branch. Composed ONLY when
                # `experts` is in the class, so every o_proj-class argv is
                # byte-identical to before.
                cmd += ["--moe-backend", "triton"]
        else:
            # the LEGACY lens+lora path: literal ceiling, no CPU count —
            # byte-identical to before the lora-native mode existed.
            lora_dirs = {name: str(path) for name, path in registered}
            max_rank = 64  # ceiling; real adapters are far below
            cmd += ["--enable-lora",
                    "--max-loras", str(len(lora_modules)),
                    "--max-lora-rank", str(max_rank)]
        lora_names = tuple(name for name, _ in registered)
        # ONE --lora-modules flag carrying every name=path value. The
        # pinned vLLM 0.25.1 CLI parses this option with nargs="+" and a
        # custom LoRAParserAction whose __call__ REPLACES the parsed list
        # (setattr, not append) on every occurrence — so repeated
        # `--lora-modules a=.. --lora-modules b=..` flags register ONLY
        # the last adapter, and every other arm 404s at request time
        # (read out of vllm/entrypoints/openai/cli_args.py at v0.25.1).
        # The SORTED order here is also the frontend's lora_int_id order
        # (serving.py init_static_loras @0.25.1) — what the lora-native
        # readiness asserts per name (serving_attestation.py).
        cmd.append("--lora-modules")
        for name in lora_names:
            cmd.append(f"{name}={lora_dirs[name]}")
    if lens_registry_dir is not None:
        # THE STEERING EXTENSION MUST BE NAMED ON THE COMMAND LINE. The
        # pinned vLLM 0.25.1 exposes exactly one worker-extension slot
        # (`--worker-extension-cls`, a real `vllm serve` CLI arg at the
        # pinned tag — vllm/engine/arg_utils.py:1145); nothing fills it
        # by default, and the vllm-lens plugin injects ITS OWN class into
        # an empty slot (`_patched_create_engine_config`, only `if not
        # self.worker_extension_cls`). A launch that omits this flag
        # therefore comes up healthy with the STOCK extension: the
        # registry env below is never read, no steering hooks are
        # installed, and every arm serves as base — the exact
        # silent-no-steer failure this lane refuses everywhere.
        # SisteerWorkerExtension SUBCLASSES vllm-lens's extension, so
        # occupying the slot keeps the plugin's whole RPC surface intact
        # (the plugin respects an already-set value — nothing breaks).
        from sisteer.capeval.serving.lens_worker import WORKER_EXTENSION_CLS
        cmd += ["--worker-extension-cls", WORKER_EXTENSION_CLS]
        # ... and the AUTH MIDDLEWARE guarding the dev RPC surface the
        # install needs: the stock --api-key middleware covers /v1|/v2|
        # /inference ONLY, so without this class /collective_rpc would
        # accept unauthenticated worker RPC from any same-host tenant
        # for the server's whole lifetime (loopback is not isolation on
        # shared Slurm, and the route cannot be unmounted after the
        # one-shot install). The dotted class rides vLLM's supported
        # --middleware hook; the token itself NEVER rides argv — the
        # middleware reads it from the server env at startup.
        cmd += ["--middleware", AUTH_MIDDLEWARE_CLS]
    elif lora_native:
        # THE PLUGIN-FREE EXTENSION fills the same single slot: no vllm-lens
        # dependency (importable in a process the plugin never entered),
        # one RPC — sisteer_scheduler_config — whose payload readiness
        # holds to the composed launch (geometry, KV coverage, engine
        # mode, LoRA config, per-name adapter fingerprints). The same auth
        # middleware guards the dev RPC route it answers on.
        cmd += ["--worker-extension-cls", NATIVE_WORKER_EXTENSION_CLS]
        cmd += ["--middleware", AUTH_MIDDLEWARE_CLS]
    # the manifest's pinned extra flags come last (they are hash inputs,
    # so what runs is what was hashed), PARSED per option: a
    # wrapper-owned option can never be overridden, and split-form
    # ``--opt value`` pairs travel whole (the old token-by-token dedup
    # could drop an option name while leaving its value stranded).
    cmd += _manifest_flag_tokens(spec.serving_launch_flags,
                                 lora_enabled=bool(lora_modules),
                                 lora_native=lora_native)

    env: Dict[str, str] = {}
    resolved_registry: Optional[Path] = None
    if lora_native:
        # the extension class must be IMPORTABLE inside the spawned server
        # and its workers (same resolution as the lens branch below).
        import sisteer
        env["PYTHONPATH"] = str(Path(sisteer.__file__).resolve().parents[1])
        # KEEP THE PLUGIN OUT, IN THE LAUNCH ENV, NOT BY INHERITANCE: the
        # pinned envs.VLLM_PLUGINS splits "" to [''] and
        # load_plugins_by_group loads only entry-point names in that list,
        # so the empty string loads NO general plugin (vllm/envs.py,
        # vllm/plugins/__init__.py @0.25.1; every graphs arm attests
        # plugin_loaded=False). The serving-venv bootstrap `unset`s
        # VLLM_PLUGINS for the lens's benefit, which a lora-native child
        # must not inherit — hence set, not omitted.
        env["VLLM_PLUGINS"] = ""
        # ... and the plugin's own disable switch as the belt: even a
        # vllm-lens that got imported some other way installs nothing.
        env[LENS_DISABLE_ENV] = "1"
        # the dev RPC route the attestation reads through (see
        # SERVER_DEV_MODE_ENV for why exposing it is acceptable here).
        env[SERVER_DEV_MODE_ENV] = "1"
        # THE RESIDUAL SLOT: the worker extension's import arms it iff this flag is
        # set (native_residual.ENV_FLAG); a LoRA-only launch's env is unchanged
        if residual_slot_armed:
            env[RESIDUAL_SLOT_ENV] = "1"
        # THE PAIRED WRITE: the bank allocates its second-gate buffers at construction iff
        # this flag is set (native_residual.ENV_FLAG_PAIRED); a single-gate launch's env is
        # unchanged, and so are its bank and its captured graph
        if residual_slot_paired:
            env[RESIDUAL_SLOT_PAIRED_ENV] = "1"
        # THE ROUTED-EXPERT LoRA BUFFER RANK: the worker extension's import sizes the
        # fused-MoE LoRA buffers at this R iff set (native_moe_lora.ENV_FLAG); a launch whose
        # class wraps no expert leaves it unset and its env is unchanged
        if moe_lora_rank is not None:
            env[MOE_LORA_RANK_ENV] = str(int(moe_lora_rank))
    if lens_registry_dir is not None:
        # resolve the registry ONCE at compose time (registry_shim's
        # pointer scheme): the resolved IMMUTABLE build dir is what every
        # worker installs from and what the readiness attestation is
        # validated against — a rebuild publishing mid-launch can never
        # split the workers across two builds. Also accepts a flat
        # export_worker_registry dir unchanged.
        from sisteer.capeval.serving.registry_shim import resolve_registry_dir
        resolved_registry = resolve_registry_dir(lens_registry_dir)
        # the extension class must be IMPORTABLE inside the spawned
        # server and its workers: the composer resolves the live sisteer
        # tree (however this process found it — installed or
        # sys.path-injected) and pins it into the child's PYTHONPATH.
        # launch_server MERGES this with any inherited PYTHONPATH rather
        # than clobbering it.
        import sisteer
        env["PYTHONPATH"] = str(Path(sisteer.__file__).resolve().parents[1])
        # register the vllm-lens plugin + point the workers at the
        # immutable object registry (preloaded at server start, H3).
        # NB the value is the ENTRY-POINT name, not the distribution name:
        # vLLM filters general plugins by entry-point name, and vllm-lens
        # 1.2.1 publishes `activations` under [vllm.general_plugins] (see
        # LENS_PLUGIN_ENTRY_POINT). "vllm_lens" here filtered the plugin
        # OUT, i.e. a server that comes up healthy and steers NOTHING —
        # the exact silent-no-capture failure the Phase-0 gate exists to
        # catch, and one the probe cannot see because it pops this var.
        env["VLLM_PLUGINS"] = LENS_PLUGIN_ENTRY_POINT
        env[LENS_REGISTRY_DIR_ENV] = str(resolved_registry)
        # expose the pinned release's server-mode collective_rpc route:
        # it is the mechanism launch_server uses to actually INSTALL the
        # steering wiring in every worker and collect the attestation —
        # without it the extension's RPC surface exists but nothing ever
        # invokes it, and the server passes readiness unsteered (see
        # SERVER_DEV_MODE_ENV for why exposing it is acceptable here).
        env[SERVER_DEV_MODE_ENV] = "1"
    if exact_replay_strategy == "batch_invariant":
        env[BATCH_INVARIANT_ENV] = "1"

    # the per-launch API token: cryptographically random, minted here,
    # carried ONLY on the repr-suppressed field — launch_server injects
    # it into the child env (vLLM's stock --api-key mechanism reads
    # VLLM_API_KEY; our middleware reads the same variable), and every
    # client attaches it as a Bearer header. Deliberately NOT in `cmd`
    # (argv leaks via /proc on shared nodes) and NOT in `env` (the proxy
    # records launch.env as its attestation).
    api_key = secrets.token_urlsafe(32)

    # bcb-bfcl-200-widen: the protocol CLAMP OVERRIDE — a self-contained
    # step over the finished argv + env (nothing above changes): the stock
    # `vllm serve …` head becomes `<serving python> -m
    # sisteer.capeval.serving.vllm_entry --sisteer-max-temp <T> serve …` and
    # the live sisteer tree's `src` leads the child's PYTHONPATH. Absent, the
    # launch is byte-identical to today's (the card identity stands).
    clamp_record: Optional[Dict[str, object]] = None
    if sampling_clamp_override is not None:
        cmd, env, clamp_record = apply_sampling_clamp_override(cmd, env, sampling_clamp_override)

    return ServerLaunch(cmd=tuple(cmd), env=dict(env), host=host, port=port,
                        model=spec.target_model,
                        lens_registry_dir=(str(resolved_registry)
                                           if resolved_registry else None),
                        api_key=api_key,
                        max_num_seqs=max_num_seqs,
                        max_num_batched_tokens=max_num_batched_tokens,
                        max_model_len=max_model_len,
                        prefix_caching=bool(prefix_caching),
                        default_chat_template_kwargs=(
                            default_chat_template_kwargs),
                        mamba_ssm_cache_dtype=mamba_ssm_cache_dtype,
                        sampling_clamp_override=clamp_record,
                        override_generation_config=override_generation_config,
                        server_default_policy=server_default_policy,
                        sampling_clamp_override_mechanism=(clamp_override_mechanism(env) if clamp_record is not None else None),
                        lora_native=bool(lora_native),
                        lora_modules=lora_names,
                        lora_adapter_dirs=dict(lora_dirs),
                        max_loras=(len(lora_names) if lora_names else None),
                        max_lora_rank=max_rank,
                        max_cpu_loras=max_cpu_loras,
                        lora_max_module_dim=lora_max_dim,
                        lora_kernel_index_ceiling=kernel_ceiling,
                        lora_target_modules=lora_target_modules,
                        residual_slot_armed=bool(residual_slot_armed),
                        residual_slot_paired=bool(residual_slot_paired),
                        moe_lora_rank=(int(moe_lora_rank) if moe_lora_rank is not None else None),
                        cudagraph_capture_sizes=capture_sizes,
                        revision=spec.target_revision)


#: the vLLM ``--lora-target-modules`` suffix of the fused-MoE LoRA wrapper
#: (``lora_export.EXPERTS_TARGET_SUFFIX``): present in an adapter's class iff
#: its FILE carries routed-expert factors, and the one suffix that also
#: composes ``--moe-backend triton`` (build_server_launch).
MOE_EXPERTS_SUFFIX = _sa.EXPERTS_LEAF


def launch_only_spec(model: str, revision: str, *, audit: str, chat_template_hash: str,
                     serving_packages: Mapping[str, str], tool_parser: str,
                     launch_flags: Sequence[str] = (), impl_commit: Optional[str] = None,
                     synthetic_model: str = "none/none",
                     prompt_provenance: Optional[str] = None,
                     fallbacks: bool = True) -> cm.PetriSpec:
    """The minimal VALID :class:`~sisteer.capeval.manifest.PetriSpec` a
    LAUNCH-ONLY caller hands :func:`build_server_launch` — the onboarding
    vLLM gate and the serving probe, which compose and stand up an engine
    but never run an audit: the served ``model`` + ``revision``, an
    interleaved proxy-vllm serving mode, the pinned chat template's hash,
    the caller's launch flags / tool parser / serving packages, no prompted
    cells (so no prefix hashes), an UNFROZEN bundle, ``impl_commit`` when it
    is a 40-hex sha (else UNFROZEN), the smallest legal ``BloomOptions``
    (every field a harness-hash input, none defaulted) and synthetic
    audit-side names (``synthetic_model`` for auditor / judge / scenarios,
    ``prompt_provenance`` default ``package-pinned@<audit>``;
    ``fallbacks=False`` empties the fallback-model chains). Nothing here
    reaches an audit — it is launch composition."""
    impl = impl_commit if re.fullmatch(r"[0-9a-f]{40}", impl_commit or "") else cm.PETRI_UNFROZEN
    kw: Dict[str, Any] = {}
    if not fallbacks:
        kw.update(auditor_fallback_models=[], judge_fallback_models=[])
    return cm.PetriSpec(
        audit=audit, behaviors=list(cm.PETRI_BEHAVIORS), cells=["none"],
        scenario_bundle_hash=cm.PETRI_UNFROZEN, auditor_model=synthetic_model,
        judge_model=synthetic_model, scenarios_model=synthetic_model,
        prompt_provenance=prompt_provenance or f"package-pinned@{audit}",
        target_model=model, target_revision=revision, serving_backend="proxy_vllm",
        serving_mode="interleaved", serving_packages=dict(serving_packages),
        serving_launch_flags=list(launch_flags), chat_template_hash=chat_template_hash,
        tool_parser=tool_parser, serving_impl_commit=impl, prompted_prefix_hashes={},
        bloom_options=cm.BloomOptions(max_turns=1, enable_prefill=False, enable_rollback=False,
                                      compaction=False, realism_filter=False, epochs=1, seed=0),
        **kw)


def _adapted_module_class(lora_dirs: Mapping[str, str]) -> Tuple[str, ...]:
    """The ONE module-suffix set every registered adapter adapts — the
    ``--lora-target-modules`` a lora-native launch composes. Per adapter
    the class is what its FILE carries and nothing else
    (``lora_export.adapter_target_class``: the short suffixes of the
    factor keys plus ``experts`` when any key names a fused-experts
    wrapper) — the FILE is the authoritative record of what the engine
    will execute (an earlier config∪file union let a ``target_modules``
    entry with no factor behind it widen ``--lora-target-modules``
    silently, allocating LoRA buffers and running the kernels at wrappers
    no adapter adapts). The config's ``target_modules`` keeps PEFT
    semantics (real HF module names — on a MoE set
    ``gate_proj``/``up_proj``/``down_proj`` reach the shared expert; the
    routed experts are PARAMETERS it never names, and the vLLM wrapper
    word ``experts`` only ever comes from the file) and is still required
    to be a non-empty list of non-empty str (vLLM's PEFTHelper reads it;
    the zero adapter's un-deduplicated ``["o_proj", "o_proj", ...]`` is
    fine). Every adapter must name the SAME set, because one launch
    restricts the wrappers to one class. Sorted tuple. Refusals name the
    adapter and the sets."""
    from sisteer.capeval.serving import lora_export as _lx  # lazy: torch
    from sisteer.capeval.serving.worker_reports import (  # lazy: torch
        FingerprintError, read_adapter_config)
    sets: Dict[str, frozenset] = {}
    for name, d in sorted(lora_dirs.items()):
        try:
            cfg = read_adapter_config(Path(d))
        except FingerprintError as e:
            raise VllmServerError(
                f"adapter {name!r}: cannot read the adapted module class — {e}") from e
        targets = cfg.get("target_modules")
        if (not isinstance(targets, list) or not targets
                or not all(isinstance(t, str) and t for t in targets)):
            raise VllmServerError(
                f"adapter {name!r} ({d}): adapter_config.json target_modules={targets!r} is not "
                "a non-empty list of module suffixes — the lora-native launch composes "
                "--lora-target-modules from the registered adapters (with no restriction vLLM "
                "wraps every linear projection and runs the LoRA kernels at each wrapper's "
                "stride), and cannot restrict the wrappers to an unstated class; refusing")
        try:
            from_file = _lx.adapter_target_class(Path(d))
        except _lx.LoraExportError as e:
            raise VllmServerError(
                f"adapter {name!r} ({d}): cannot infer the adapted module class from its "
                f"factors — {e}") from e
        # the file wins: never `| frozenset(targets)`
        sets[name] = (frozenset(from_file), RESIDUAL_SLOT_CONFIG_BLOCK in cfg)
    # ONE class over the GENUINE adapters; a residual-slot carrier (its config names the
    # petri_residual_slot block) rides inside it — lora_export.launch_target_class, the one rule
    # native_standup's zero-adapter class reads too (the steered canary, 2026-09-30)
    try:
        return _lx.launch_target_class(sets)
    except _lx.LoraExportError as e:
        raise VllmServerError(str(e)) from e


def _residual_slot_armed(lora_dirs: Mapping[str, str]) -> bool:
    """True iff any registered adapter's ``adapter_config.json`` carries the
    :data:`RESIDUAL_SLOT_CONFIG_BLOCK` block (a residual-slot bank entry:
    ``lora_export._export_residual_slot``)."""
    from sisteer.capeval.serving.worker_reports import read_adapter_config

    return any(RESIDUAL_SLOT_CONFIG_BLOCK in read_adapter_config(Path(d))
               for _name, d in sorted(lora_dirs.items()))


def _residual_slot_paired(lora_dirs: Mapping[str, str]) -> bool:
    """True iff any registered adapter's :data:`RESIDUAL_SLOT_CONFIG_BLOCK` block
    carries the PAIRED op (:data:`RESIDUAL_SLOT_PAIRED_OP`; ``lora_export.
    _export_residual_slot`` over a ``psr_paired_write`` artifact) — the same
    ``adapter_config.json`` read as :func:`_residual_slot_armed`, one field deeper.
    A block that is not a JSON object is left to the worker's validator to refuse."""
    from sisteer.capeval.serving.worker_reports import read_adapter_config

    for _name, d in sorted(lora_dirs.items()):
        block = read_adapter_config(Path(d)).get(RESIDUAL_SLOT_CONFIG_BLOCK)
        if isinstance(block, Mapping) and block.get("op") == RESIDUAL_SLOT_PAIRED_OP:
            return True
    return False


def fitted_max_lora_rank(adapter_dirs: Iterable[Path]) -> int:
    """The ``--max-lora-rank`` a lora-native launch composes: the smallest
    cap in :data:`LEGAL_LORA_RANK_CAPS` that is >= the largest ``r``
    declared across the registered adapters' ``adapter_config.json``
    (rank-4 lora_all and rank-8 subtraction -> 8; a rank-9 adapter -> 16).
    REFUSES a directory without a config (vLLM could not register it
    either), a non-positive ``r``, and ``r`` above the largest legal cap
    (512 at 0.25.1) — the cap is fitted, never guessed."""
    dirs = [Path(d) for d in adapter_dirs]
    if not dirs:
        raise VllmServerError(
            "fitted_max_lora_rank over no adapter directories — nothing to "
            "fit the rank cap to")
    # lazy: worker_reports imports torch, which this composer module must
    # not pull in at import; its reader is the ONE adapter_config.json
    # parser (a missing / unparseable config or a non-positive r refuses
    # by name).
    from sisteer.capeval.serving.worker_reports import (FingerprintError,
                                                        read_adapter_config)
    max_r = 0
    for d in dirs:
        try:
            r = int(read_adapter_config(d)["r"])
        except FingerprintError as e:
            raise VllmServerError(
                f"cannot fit the lora-native rank cap — it is fitted to the "
                "registered adapters' declared r, and vLLM could not "
                f"register this adapter either: {e}") from e
        max_r = max(max_r, r)
    for cap in LEGAL_LORA_RANK_CAPS:
        if cap >= max_r:
            return cap
    raise VllmServerError(
        f"the largest registered adapter rank {max_r} exceeds the largest "
        f"legal --max-lora-rank {LEGAL_LORA_RANK_CAPS[-1]} (vllm/config/"
        "lora.py MaxLoRARanks @0.25.1) — this adapter cannot be served "
        "natively")


def _safetensors_header(weights: Path) -> Dict[str, Any]:
    """The safetensors header of ``weights`` (8-byte little-endian length + JSON;
    ``__metadata__`` dropped) — stdlib only, this composer never imports torch."""
    import json as _json
    import struct
    with open(weights, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise VllmServerError(f"{weights}: not a safetensors file (no 8-byte header length)")
        (n,) = struct.unpack("<Q", raw)
        body = f.read(n)
    try:
        header = _json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise VllmServerError(f"{weights}: unreadable safetensors header — {e}") from e
    if not isinstance(header, dict):
        raise VllmServerError(f"{weights}: safetensors header is not an object")
    header.pop("__metadata__", None)
    return header


#: a factor key whose module path IS a fused-experts wrapper: ``...experts`` (the
#: outer, down_proj half) or ``...experts.base_layer`` (the inner, gate_up half) —
#: ``lora_export._EXPERTS_MODULE_RE``'s reading of the PEFT ParamWrapper layout
_EXPERTS_FACTOR_RE = re.compile(r"^(?P<path>.+\.experts)(?:\.base_layer)?\.lora_A\.weight$")


def fitted_moe_lora_rank(adapter_dirs: Iterable[Path]) -> int:
    """THE ROUTED-EXPERT LoRA BUFFER RANK a lora-native launch composes: the
    LARGEST per-expert rank the registered adapters declare for their routed
    experts — every ``rank_pattern`` value of every ``adapter_config.json``
    (``lora_export._general_config`` writes ``{<experts regex>: r_e}``, ``2·r_e``
    for a difference adapter, and the class zero adapter the same) — cross-checked
    against the FILE: every routed-expert ``lora_A`` factor (``[E·r_e, in]`` in
    PEFT's ParamWrapper layout) must have a row count the declared ``r_e`` divides.
    REFUSES an adapter that carries expert factors but declares no ``rank_pattern``
    (the buffers cannot be sized to a need the adapter does not state), a value
    that is not a positive int, a declaration the file contradicts, and a set of
    adapters none of which declares a routed rank (the caller composes this only
    for an experts-wrapping class, so that is a contradiction too). The worker's
    ``set_lora`` refuses by name any adapter above the composed R, so a stated
    rank is never a silent truncation."""
    from sisteer.capeval.serving.worker_reports import (FingerprintError,
                                                        read_adapter_config)
    dirs = [Path(d) for d in adapter_dirs]
    if not dirs:
        raise VllmServerError("fitted_moe_lora_rank over no adapter directories — nothing to size the expert buffers to")
    routed: Dict[str, int] = {}
    for d in dirs:
        try:
            cfg = read_adapter_config(d)
        except FingerprintError as e:
            raise VllmServerError(f"cannot fit the routed-expert LoRA rank — {e}") from e
        weights = d / "adapter_model.safetensors"
        expert_rows: Dict[str, int] = {}
        if weights.is_file():
            for key, ent in _safetensors_header(weights).items():
                m = _EXPERTS_FACTOR_RE.match(str(key))
                if m and isinstance(ent, Mapping) and isinstance(ent.get("shape"), list) and ent["shape"]:
                    expert_rows[str(key)] = int(ent["shape"][0])
        pattern = cfg.get("rank_pattern")
        if not expert_rows and not pattern:
            continue                      # a dense-only adapter on this server (never the case for an experts class, but legal)
        if not isinstance(pattern, Mapping) or not pattern:
            raise VllmServerError(
                f"adapter {d}: carries routed-expert factors ({len(expert_rows)} lora_A keys) but adapter_config.json "
                "declares no rank_pattern — the fused-MoE LoRA buffers are sized to the routed rank the adapter STATES "
                "(lora_export writes it for every experts export); refusing")
        vals: List[int] = []
        for k, v in pattern.items():
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise VllmServerError(f"adapter {d}: rank_pattern[{k!r}]={v!r} is not a positive int; refusing")
            vals.append(int(v))
        r_e = max(vals)
        for key, rows in expert_rows.items():
            if rows % r_e:
                raise VllmServerError(
                    f"adapter {d}: {key} has {rows} rows, which the declared routed rank {r_e} does not divide "
                    "([E·r_e, in] in PEFT's ParamWrapper layout) — the config and the file disagree; refusing")
        routed[str(d)] = r_e
    if not routed:
        raise VllmServerError(
            "none of the registered adapters declares a routed-expert rank (rank_pattern) yet the launch's adapted "
            f"class wraps the routed experts — cannot size the fused-MoE LoRA buffers for {[str(d) for d in dirs]}; refusing")
    return max(routed.values())


def _server_default_generation_config(model_id: str,
                                      revision: Optional[str]) -> Optional[str]:
    """The ``--override-generation-config`` JSON for ``model_id``, or None.

    Composed ONLY for a model the contracts pin a server-default row for
    (``SERVER_DEFAULT_SAMPLING_BY_MODEL`` — models whose pinned snapshot
    ships no generation_config.json; the set of models and the numbers come
    from the registry's decode policy export). The premise is checked against the
    LOCAL HF cache the launch will serve from: ``try_to_load_from_cache``
    answers a path (the file IS cached), the no-exist sentinel (the hub
    said it does not exist) or None (never asked). A cached file means the
    vendor now ships one and the override would silently mask it — REFUSE,
    naming the file, so the table is re-verified in the same change that
    serves the new revision. The sentinel and None both mean "no vendor
    file to mask" (a real launch has the weights cached, so None is the
    grading host / CPU-test case, never a live serve)."""
    from sisteer.capeval.tinycap.contracts import (
        SERVER_DEFAULT_SAMPLING_BY_MODEL, server_default_generation_config)

    row = SERVER_DEFAULT_SAMPLING_BY_MODEL.get(model_id)
    if row is None:
        return None
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(
            model_id, "generation_config.json", revision=revision)
    except Exception:  # no hub, no cache dir: nothing to mask
        cached = None
    if isinstance(cached, str):
        raise VllmServerError(
            f"target {model_id!r} @ {revision!r} has a generation_config.json in the "
            f"local HF cache ({cached}) but contracts.SERVER_DEFAULT_SAMPLING_BY_MODEL "
            "pins a server-side sampling default for it on the premise that the "
            "snapshot ships NONE — the override would silently mask the vendor's own "
            "file. Re-verify the row against that file and update (or drop) the table "
            "entry in the same change that serves this revision")
    return server_default_generation_config(row)


def _canonical_option_name(name: str) -> str:
    """The option name as the PINNED parser would resolve it, for
    ownership matching. FlexibleArgumentParser (read at the 0.25.1 tag)
    accepts ``_`` and ``-`` interchangeably in option names (normalized
    between the leading ``--`` and the first ``.``), and a dotted
    ``--opt.field value`` form that collapses into the base option
    ``--opt`` as JSON — so ``--max_num_seqs``,
    ``--speculative_config.num_speculative_tokens`` and friends must
    all match their owned base option, or the ownership check is a
    string comparison the real parser does not share."""
    base = name[2:].split(".", 1)[0].replace("_", "-")
    return "--" + base


def _manifest_flag_tokens(flags: List[str], *, lora_enabled: bool,
                          lora_native: bool = False) -> List[str]:
    """Validate + pass through the manifest's extra launch flags.

    Each option is parsed to its NAME (the part before ``=``), with a
    following non-``--`` token treated as its split-form value, so both
    ``--dtype=float16`` and ``--dtype float16`` resolve to the owned
    option ``--dtype``. Names are matched in the pinned parser's
    CANONICAL form (:func:`_canonical_option_name`: underscores
    normalize to dashes, a dotted suffix collapses into its base
    option), so no accepted spelling can slip past ownership. Wrapper-
    owned options refuse; the one tolerated shape is a bare restatement
    of an always-pinned flag (and of ``--enable-lora`` when adapters
    are actually registered), which is skipped as a documented
    duplicate. Single-dash tokens (short aliases like the pinned
    ``-sc`` for ``--speculative-config``) refuse via the stray-token
    rule — the recorded contract is long-form options only.

    The restatement set is MODE-AWARE: under ``lora_native`` it is
    :data:`_LORA_NATIVE_PIN_RESTATEMENTS` (empty) — the composer states
    ``--no-enforce-eager`` there, so a manifest's ``--enforce-eager`` is
    a contradiction, not a duplicate, and refuses with a message naming
    the mode (the lora-native standup strips it before composing and
    RECORDS the drop; every full22 manifest carries it for the lens).

    ABBREVIATIONS refuse too: the pinned parser never sets
    ``allow_abbrev=False`` (read at the tag — neither
    FlexibleArgumentParser nor the arg_utils setup touches it), so
    argparse's default prefix matching accepts ``--speculative-conf``,
    ``--long-prefill-token-thresh``, ``--scheduler-c`` and every other
    unambiguous prefix of a real option. Any manifest option whose
    canonical name is a PREFIX of a wrapper-owned option is therefore
    refused, conservatively — including prefixes argparse would call
    ambiguous (those die at launch anyway; refusing here is earlier and
    names the reason)."""
    toks = list(flags)
    out: List[str] = []
    i = 0
    while i < len(toks):
        tok = toks[i]
        if not tok.startswith("--"):
            raise VllmServerError(
                f"manifest serving_launch_flags: stray value {tok!r} "
                "without a preceding --option — flags are '--opt', "
                "'--opt=value', or '--opt value' pairs (short aliases "
                "like '-sc' are refused: the recorded contract is "
                "long-form options only)")
        name, has_eq, _ = tok.partition("=")
        canon = _canonical_option_name(name)
        j = i + 1
        if j < len(toks) and not toks[j].startswith("--"):
            j += 1  # split-form value travels with its option
        if canon in _WRAPPER_OWNED_OPTIONS:
            bare = not has_eq and j == i + 1
            restatements = (_LORA_NATIVE_PIN_RESTATEMENTS if lora_native
                            else _PIN_RESTATEMENTS)
            restatement = (canon in restatements
                           or (canon == "--enable-lora" and lora_enabled))
            if bare and restatement and name == canon:
                i = j  # documented duplicate of the composed pin
                continue
            if lora_native and canon in ("--enforce-eager",
                                         "--no-enforce-eager"):
                raise VllmServerError(
                    f"manifest serving_launch_flags carries {toks[i:j]!r} "
                    "under lora_native: the lora-native mode composes "
                    "--no-enforce-eager explicitly (CUDA graphs on, the "
                    "plugin kept out), so NEITHER eager spelling is a "
                    "tolerated restatement here — a manifest --enforce-eager "
                    "would contradict the composed graphs-on launch behind "
                    "the hashed identity; strip it from the flags the "
                    "composer sees and RECORD the drop (the standup records "
                    "it as manifest_flags_dropped)")
            raise VllmServerError(
                f"manifest serving_launch_flags carries {toks[i:j]!r}: "
                f"{canon!r} is a wrapper-owned launch setting (pinned and "
                "validated by build_server_launch) — a manifest flag must "
                "never override it, in any accepted spelling (underscore, "
                "dotted, or =-form); what runs must be what the recorded "
                "identity claims (plan §5)")
        abbreviated = sorted(o for o in _WRAPPER_OWNED_OPTIONS
                             if o.startswith(canon))
        if abbreviated:
            raise VllmServerError(
                f"manifest serving_launch_flags carries {toks[i:j]!r}: "
                f"{canon!r} is an accepted argparse ABBREVIATION of the "
                f"wrapper-owned option(s) {abbreviated} (the pinned "
                "parser leaves allow_abbrev on, so a prefix sets the "
                "full option) — a manifest flag must never override a "
                "wrapper-owned setting, abbreviated spellings included; "
                "spell out the full, non-owned option intended")
        out.extend(toks[i:j])
        i = j
    return out


# --- steering installation + attestation (the readiness gate's second half) ----


def expected_install_state(registry_dir: Any) -> Dict[str, Any]:
    """What EVERY worker's install report must equal, computed by the
    LAUNCHER from the same immutable registry the workers load — an
    independent second reading, so a worker that loaded a different (or
    no) registry cannot attest itself. Loads through the committed
    ``WorkerObjectRegistry.load`` (all hash/selector verification) and
    derives layers/ledger cells with the worker's own helpers.

    SITE-AWARE, matching the report's own schema: ``layers`` are the
    BLOCK-site (decoder-layer output) hook layers and ``oproj_layers``
    the ``self_attn.o_proj``-site ones (the lora kinds), split with the
    worker's own ``site_layers`` — the 2026-08-28 scale-12 wave died at
    readiness because this expectation lumped every op layer into
    ``layers`` while a pure-LoRA worker correctly reported them under
    ``oproj_layers`` (worker: ``layers = []``, launcher: ``expected
    [0..31]``). ``plan_layer`` is the lowest hooked layer across BOTH
    sites (the first-firing hook of every forward — the worker's own
    plan choice)."""
    from sisteer.capeval.serving import lens_steer as ls
    from sisteer.capeval.serving.lens_worker import (cells_needing_ledger,
                                                     site_layers)

    reg = ls.WorkerObjectRegistry.load(Path(registry_dir))
    block_layers, oproj_layers = site_layers(reg)
    all_layers = sorted(set(block_layers) | set(oproj_layers))
    if not all_layers:
        raise VllmServerError(
            f"registry {registry_dir} adapts to ZERO hook layers — nothing "
            "would be steered; refusing to compose an installation "
            "expectation for an empty install")
    return {
        "registration_count": 1,
        "layers": list(block_layers),
        "oproj_layers": list(oproj_layers),
        "plan_layer": all_layers[0],
        "cells": sorted(reg.cell_ids()),
        "ledger_cells": sorted(cells_needing_ledger(reg)),
    }


def validate_install_reports(results: Any,
                             expected: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Every worker's ``sisteer_install_steering`` report must match the
    launcher's independently computed expectation EXACTLY — cells,
    block-site layers, o_proj-site layers, plan layer, ledger cells, and
    registration_count == 1. A missing/empty result list, a non-report
    result (the dev RPC route stringifies anything non-JSON), or any
    field mismatch refuses: a server whose workers did not all attest
    the same expected install is not ready, it is unsteered. (A report
    that OMITS ``oproj_layers`` — an older worker — reads as ``[]`` and
    refuses against any o_proj-serving registry, on purpose.)"""
    if not isinstance(results, list) or not results:
        raise VllmServerError(
            f"steering install returned {results!r} — expected one install "
            "report PER WORKER; an empty result set means no worker "
            "installed anything and the server would serve every arm as "
            "base")
    reports: List[Dict[str, Any]] = []
    problems: List[str] = []
    for i, rep in enumerate(results):
        if not isinstance(rep, Mapping):
            problems.append(f"worker {i}: non-report result {rep!r}")
            continue
        got = {
            "registration_count": rep.get("registration_count"),
            "layers": sorted(int(l) for l in (rep.get("layers") or [])),
            "oproj_layers": sorted(int(l) for l in
                                   (rep.get("oproj_layers") or [])),
            "plan_layer": rep.get("plan_layer"),
            "cells": sorted(str(c) for c in (rep.get("cells") or [])),
            "ledger_cells": sorted(str(c) for c in
                                   (rep.get("ledger_cells") or [])),
        }
        for key in ("registration_count", "layers", "oproj_layers",
                    "plan_layer", "cells", "ledger_cells"):
            if got[key] != expected[key]:
                problems.append(
                    f"worker {i}: {key} = {got[key]!r}, expected "
                    f"{expected[key]!r}")
        reports.append(dict(rep))
    if problems:
        raise VllmServerError(
            "steering install reports do not match the registry's expected "
            "state — the server is NOT serving what the registry says and "
            "must not pass readiness:\n- " + "\n- ".join(problems))
    return reports


def _post_collective_rpc(launch: ServerLaunch, method: str,
                         args: List[Any], *, timeout_s: float,
                         what: str) -> Any:
    """POST one ``collective_rpc`` invocation over the pinned release's
    server-mode route and return the parsed JSON payload. Any transport
    or HTTP failure raises with the diagnosis the readiness caller
    needs — route absent (404: dev mode not on / wrong vLLM), token
    refused (401: not this launch's child), worker exception (5xx)."""
    body = json.dumps({"method": method, "args": list(args)}
                      ).encode("utf-8")
    url = f"{launch.base_url}/collective_rpc"
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json",
                 **launch.auth_headers},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:2000]
        except Exception:
            pass
        if e.code == 404:
            hint = ("the collective_rpc route is not mounted — the server "
                    "env lacks VLLM_SERVER_DEV_MODE, or this is not the "
                    "pinned vLLM; a server without the RPC route can "
                    "never be attested and must not pass readiness")
        elif e.code == 401:
            hint = ("the server refused this launch's token — the child "
                    "was not spawned with this launch's api_key (a stale "
                    "or foreign server?); refusing readiness")
        else:
            hint = f"a worker raised during {what}"
        raise VllmServerError(
            f"{what} RPC {url} failed with HTTP {e.code} — "
            f"{hint}: {detail}") from e
    except (urllib.error.URLError, OSError) as e:
        raise VllmServerError(
            f"{what} RPC {url} could not be reached: {e}") from e
    if status != 200:
        raise VllmServerError(f"{what} RPC {url} answered HTTP {status}")
    try:
        return json.loads(raw.decode("utf-8")) if raw else {}
    except ValueError as e:
        raise VllmServerError(
            f"{what} RPC {url} returned unparseable JSON: {e}") from e


# --- the LoRA routing recorder RPCs (gemma canary job 489308) --------------------

#: the three recorder methods (``serving_attestation`` names them once; the
#: native worker extension implements them; the proxy allowlists them)
LORA_ROUTING_ARM_RPC_METHOD = _sa.LORA_ROUTING_ARM_RPC_METHOD
LORA_ROUTING_OBSERVE_RPC_METHOD = _sa.LORA_ROUTING_OBSERVE_RPC_METHOD
LORA_ROUTING_DISARM_RPC_METHOD = _sa.LORA_ROUTING_DISARM_RPC_METHOD
LORA_ROUTING_RPC_METHODS = _sa.LORA_ROUTING_RPC_METHODS
#: one recorder RPC is a config read plus a dict copy — seconds, not minutes
LORA_ROUTING_RPC_TIMEOUT_S = 120.0


def lora_routing_epoch_args(epoch: Any) -> List[str]:
    """The ``args`` list for the arm / observe / disarm RPCs: the epoch id the
    CLIENT minted, as the one string arg the dev route passes
    (``api_router.py`` @0.25.1: string args only)."""
    if not isinstance(epoch, str) or not epoch:
        raise VllmServerError(
            f"lora routing recorder: epoch must be the non-empty string arm() returned, got "
            f"{type(epoch).__name__}")
    return [epoch]


def lora_routing_rpc_result(payload: Any, *, what: str) -> Dict[str, Any]:
    """ONE worker payload out of a recorder RPC's ``{"results": [per-worker]}``
    body. EVERY worker's payload is collected FIRST, then judged: the workers
    answer expected refusals as
    ``{"ok": False, "refusal": ..., "detail": ..., "live_epoch": ...}``
    payloads rather than exceptions, so the pinned executor's per-worker reply
    queues stay aligned; the refusal is raised HERE, naming the refusal kind
    and the workers that refused — a MIXED set (some accepted, some refused)
    is itself a refusal. A success must carry ``ok: True``; an empty result
    set, a non-mapping payload, a payload without ``ok`` and disagreeing
    successes refuse by name (a recorder that answers nothing must not read as
    "nothing misrouted"). Every worker in the process group runs the same
    scheduler output against its own model runner, so their recorders must
    AGREE. Returns worker 0's payload plus ``n_workers``."""
    results = payload.get("results") if isinstance(payload, Mapping) else None
    if not isinstance(results, list) or not results:
        raise VllmServerError(
            f"{what}: the RPC returned {payload!r} — expected one recorder payload PER "
            "WORKER; an empty result set means no worker answered and the selection "
            "control has no structural evidence")
    reports: List[Dict[str, Any]] = []
    for i, rep in enumerate(results):
        if not isinstance(rep, Mapping):
            raise VllmServerError(
                f"{what}: worker {i} answered a non-mapping payload ({type(rep).__name__}) "
                "— not the recorder's JSON payload; refusing")
        reports.append(dict(rep))
    refused = [(i, rep) for i, rep in enumerate(reports) if rep.get("ok") is not True]
    if refused:
        kinds = sorted({str(rep.get("refusal")) for _i, rep in refused})
        first_i, first = refused[0]
        accepted = [i for i in range(len(reports)) if i not in {i for i, _ in refused}]
        mixed = f"; worker(s) {accepted} accepted — a mixed set is itself a refusal" if accepted else ""
        if first.get("refusal") is None and "ok" not in first:
            raise VllmServerError(
                f"{what}: worker {first_i} answered a payload without an 'ok' field — not the "
                "recorder's payload (a pre-payload worker, or a foreign method); refusing")
        raise VllmServerError(
            f"{what}: worker(s) {[i for i, _ in refused]} refused — {', '.join(kinds)}: "
            f"{first.get('detail')!r} (live epoch {first.get('live_epoch')!r}){mixed}")
    first_text = json.dumps(reports[0], sort_keys=True, default=str)
    for i, rep in enumerate(reports[1:], start=1):
        if json.dumps(rep, sort_keys=True, default=str) != first_text:
            raise VllmServerError(
                f"{what}: worker {i}'s recorder payload disagrees with worker 0's — the "
                f"{len(reports)} workers of one process group run the same scheduler output "
                "and must record the same routing; refusing")
    return {**reports[0], "n_workers": len(reports)}


def _lora_routing_launch_check(launch: ServerLaunch, what: str) -> None:
    if not getattr(launch, "lora_native", False):
        raise VllmServerError(
            f"{what}: the launch is not lora-native — only the plugin-free native worker "
            f"extension ({NATIVE_WORKER_EXTENSION_CLS}) carries the LoRA routing recorder; "
            "a lens launch has no per-request adapter routing to attest")


def arm_lora_routing_recorder(launch: ServerLaunch, epoch: str, *,
                              timeout_s: float = LORA_ROUTING_RPC_TIMEOUT_S) -> Dict[str, Any]:
    """``sisteer_arm_lora_routing_recorder(epoch)`` in every worker over the
    launch's authenticated ``/collective_rpc`` route — the CLIENT mints
    ``epoch`` so every worker of a multi-worker launch adopts the same id and
    :func:`lora_routing_rpc_result` sees agreeing payloads (a per-worker mint
    refused here and left every worker armed); refusals propagate."""
    what = "lora routing recorder arm"
    _lora_routing_launch_check(launch, what)
    payload = _post_collective_rpc(launch, LORA_ROUTING_ARM_RPC_METHOD,
                                   lora_routing_epoch_args(epoch),
                                   timeout_s=timeout_s, what=what)
    return lora_routing_rpc_result(payload, what=what)


def lora_routing_observations(launch: ServerLaunch, epoch: str, *,
                              timeout_s: float = LORA_ROUTING_RPC_TIMEOUT_S) -> Dict[str, Any]:
    """``sisteer_lora_routing_observations(epoch)`` — THE WHOLE EPOCH's recorded
    engine-request-id -> LoRA bindings, per-step scheduled sets and counters
    (the join to the control's response ids happens in fidelity)."""
    what = "lora routing observations"
    _lora_routing_launch_check(launch, what)
    payload = _post_collective_rpc(launch, LORA_ROUTING_OBSERVE_RPC_METHOD,
                                   lora_routing_epoch_args(epoch),
                                   timeout_s=timeout_s, what=what)
    return lora_routing_rpc_result(payload, what=what)


def disarm_lora_routing_recorder(launch: ServerLaunch, epoch: str, *,
                                 timeout_s: float = LORA_ROUTING_RPC_TIMEOUT_S) -> Dict[str, Any]:
    """``sisteer_disarm_lora_routing_recorder(epoch)`` — restore ``execute_model``."""
    what = "lora routing recorder disarm"
    _lora_routing_launch_check(launch, what)
    payload = _post_collective_rpc(launch, LORA_ROUTING_DISARM_RPC_METHOD,
                                   lora_routing_epoch_args(epoch),
                                   timeout_s=timeout_s, what=what)
    return lora_routing_rpc_result(payload, what=what)


def validate_model_runner_reports(results: Any) -> Dict[str, Any]:
    """WHICH GPU model runner every worker constructed (``model_runner_class``
    / ``model_runner_kind`` in the readiness report; ``worker_reports.
    model_runner_report``): every worker must name one of the two pinned
    classes (``serving_attestation.MODEL_RUNNER_CLASSES``; canary #4, job
    489533, ran the V2 one) and all must agree — an unknown class is an
    unpinned engine and fails readiness by name. No other value gate: the
    kind is recorded for the summary, not chosen."""
    if not isinstance(results, list) or not results:
        raise VllmServerError("model runner attestation: no worker reports")
    seen: List[Tuple[str, str]] = []
    for i, rep in enumerate(results):
        if not isinstance(rep, Mapping):
            raise VllmServerError(f"model runner attestation: worker {i} report is not a mapping")
        cls = rep.get("model_runner_class")
        kind = rep.get("model_runner_kind")
        if not isinstance(cls, str) or not cls or not isinstance(kind, str):
            raise VllmServerError(
                f"model runner attestation: worker {i} reports no model_runner_class / "
                "model_runner_kind — not the pinned native worker report; refusing readiness")
        if cls not in _sa.MODEL_RUNNER_CLASSES or _sa.MODEL_RUNNER_CLASSES[cls] != kind:
            raise VllmServerError(
                f"model runner attestation: worker {i} constructed {cls!r} (kind {kind!r}), not "
                f"one of the pinned GPU model runners {sorted(_sa.MODEL_RUNNER_CLASSES)} — an "
                "unpinned engine; refusing readiness")
        seen.append((cls, kind))
    if len(set(seen)) != 1:
        raise VllmServerError(
            f"model runner attestation: the workers disagree on the model runner {sorted(set(seen))}")
    cls, kind = seen[0]
    return {"class": cls, "kind": kind, "n_workers": len(seen)}


def expected_scheduler_state(launch: ServerLaunch) -> Dict[str, Any]:
    """What every worker's resolved scheduler state must equal — the
    WHOLE no-chunking invariant, not just the sizing: the composed
    geometry, the pinned partial-prefill knobs (a non-zero long-prefill
    threshold chunks long prompts regardless of the budget), an
    effective per-step scheduled-token ceiling equal to the budget, and
    NO speculative decoding (it schedules q>1 decode steps — lattice-
    refused shapes — and is what makes the ceiling diverge from the
    budget). A launch carrying no composed geometry (hand-built, not
    through :func:`build_server_launch`) REFUSES: for a lens server the
    invariant is load-bearing, so "cannot verify" must fail readiness,
    never silently pass it."""
    if (launch.max_num_seqs is None or launch.max_num_batched_tokens is None
            or launch.max_model_len is None or launch.prefix_caching is None):
        raise VllmServerError(
            "this launch carries no composed scheduler geometry "
            "(max_num_seqs / max_num_batched_tokens / max_model_len / "
            "prefix_caching) — compose through build_server_launch; a lens "
            "server whose no-chunking geometry and cache state cannot be "
            "verified must not pass readiness")
    out: Dict[str, Any] = {
        "max_num_seqs": int(launch.max_num_seqs),
        "max_num_batched_tokens": int(launch.max_num_batched_tokens),
        "max_model_len": int(launch.max_model_len),
        # the scheduler may never issue more than the budget per step:
        # max_num_scheduled_tokens defaults to the budget and diverges
        # only under token-appending features like speculative decoding
        # (the pinned field's own docstring).
        "effective_max_num_scheduled_tokens":
            int(launch.max_num_batched_tokens),
        # no speculative decoding, by ANY config route (flags, -sc,
        # --config YAML, speculators-format model auto-enable):
        "speculative": None,
        # the stock synchronous V1 scheduler, exactly — pinned on the
        # command line (SYNC_SCHEDULING_FLAG: the live 0.25.1 resolved
        # the ASYNC class when the flag was merely omitted, job 194568)
        # and attested here because a custom (or async-switched)
        # scheduler class interprets every knob above, so the invariant
        # means nothing under any other class:
        "scheduler_cls": PINNED_DEFAULT_SCHEDULER_CLS,
        # the RESOLVED cache state must equal the composed choice: a
        # silently-flipped cache flag either loses the isolation story
        # (off->on with no salts audited) or the throughput it was
        # enabled for (on->off) — either way the launch is not what the
        # attestation claims.
        "enable_prefix_caching": bool(launch.prefix_caching),
    }
    if launch.mamba_ssm_cache_dtype is not None:
        # the composed Mamba SSM-state dtype must be what the engine
        # RESOLVED (cache_config.mamba_ssm_cache_dtype, reported by
        # lens_worker.scheduler_config_report) — the flag on argv proves
        # nothing about the state pages the engine actually sized. Only a
        # launch that composed the opt-in expects it (build_server_launch
        # composes it on hybrid-recurrent targets alone), so a dense launch
        # attests nothing new here. GPU-verified 2026-09-02 on
        # Nemotron-3-Nano-4B: composed float32 resolved to float32 (the
        # attested MambaSpec page size is the fp32 ssm state's).
        out["mamba_ssm_cache_dtype"] = str(launch.mamba_ssm_cache_dtype)
    out.update(_NO_CHUNK_SCHEDULER_PINS)
    if launch.lora_native:
        # THE LORA-NATIVE KEYS, asserted by the same
        # validate_scheduler_reports: the engine really is graphs-on
        # (enforce_eager False AND a non-NONE cudagraph mode — config/
        # vllm.py @0.25.1 forces NONE iff eager, so both are needed to
        # rule out a plugin that forced eager after argv said otherwise),
        # no vllm_lens module reached the worker, and the LoRA config +
        # loaded count equal what was composed. `cudagraph_mode` (the
        # enum NAME — vLLM resolves FULL_AND_PIECEWISE / FULL / PIECEWISE
        # per platform) and `lora_dtype` are RECORDED, not asserted: the
        # fingerprint refuses non-bf16 tensors, so a dtype drift still
        # fails readiness by name.
        if (launch.max_lora_rank is None or launch.max_cpu_loras is None
                or launch.max_loras is None or not launch.lora_modules):
            raise VllmServerError(
                "this lora-native launch carries no composed LoRA config "
                "(max_loras / max_lora_rank / max_cpu_loras / lora_modules) "
                "— compose through build_server_launch; a native server "
                "whose LoRA config cannot be verified must not pass "
                "readiness")
        out["enforce_eager"] = False
        out["cudagraph_mode_is_none"] = False
        out["lens_plugin_imported"] = False
        out["lora_max_loras"] = int(launch.max_loras)
        out["lora_max_lora_rank"] = int(launch.max_lora_rank)
        out["lora_max_cpu_loras"] = int(launch.max_cpu_loras)
        out["lora_loaded_count"] = len(launch.lora_modules)
        # THE ADAPTED CLASS: the composer emits
        # --lora-target-modules with the one suffix set the registered
        # adapters adapt, so the resolved LoRAConfig.target_modules must
        # EQUAL it — None would mean vLLM wrapped every linear projection
        # (kernels dispatched at gate_up_proj's 20,480-wide stride under a
        # ceiling sized for o_proj's 4,096), a wider or different list that
        # the restriction is not the adapted class. The execution set itself
        # is asserted per adapter by validate_native_lora_reports and every
        # executing wrapper's stride by validate_execution_strides; this pin
        # is the config-level belt. A launch without the set is hand-built.
        if not launch.lora_target_modules:
            raise VllmServerError(
                "this lora-native launch carries no composed --lora-target-modules set "
                "(lora_target_modules) — compose through build_server_launch; a native "
                "server whose LoRA wrapper class cannot be verified must not pass readiness")
        out["lora_target_modules"] = sorted(launch.lora_target_modules)
    return out


def kv_coverage_problem(rep: Mapping[str, Any], *, max_num_seqs: int,
                        max_model_len: int) -> Optional[str]:
    """The KV-COVERAGE HALF OF THE GEOMETRY, held at readiness: ``None``
    when the worker's resolved KV cache holds the FULL no-chunking
    budget (every scheduled sequence resident at full length
    simultaneously), else a one-line problem naming both numbers.

    max_num_seqs is a scheduler CEILING, not a grant — vLLM runs a
    geometry over a smaller KV cache without erroring anywhere, it just
    schedules fewer sequences and preempts the rest. The scale-12 gemma
    wave requested 48 x 16384 and came up cleanly with a 146,118-token
    KV cache (18.6% of the 786,432-token budget, "Maximum concurrency
    ... 8.92x" in the log): a fifth of the requested width, delivered
    SILENTLY (job 232411). The petri smoke has asserted exactly this per
    engine since the 32 x 16384 llama wall — and since the hybrid fix
    the smoke's kv_coverage_problem DELEGATES to this very function
    (scripts/petri_steered_smoke.py maps its legacy fact keys onto the
    report names), so readiness and the gate cannot drift apart; a
    mis-sized geometry refuses in minutes, naming the numbers, instead
    of degrading for a whole wave. Unreadable facts are a problem too,
    never a pass — the worker refuses them at report time
    (lens_worker.scheduler_config_report + kv_group_report), and a
    report missing them here (an older extension) refuses the same way.

    TWO COVERAGE FORMULAS, keyed on the reported group count. A UNIFORM
    cache (one KV cache group — llama, gemma) keeps the original
    arithmetic verbatim: kv_tokens = num_gpu_blocks x block_size against
    the max_num_seqs x max_model_len budget, byte-identical verdicts
    and messages. A HYBRID cache (multiple groups — Qwen3.5's
    full-attention + gated-delta-net) must NOT use that formula: the
    scheduler block size is an LCM over the groups' block sizes and a
    Mamba/GDN group's pages hold per-sequence STATE, not token spans,
    so num_gpu_blocks x block_size misstates capacity in either
    direction with everything still green — the silent-miscount shape.
    There the verdict is the allocator's own
    ``get_max_concurrency_for_kv_cache_config`` result, computed inside
    the worker (lens_worker.kv_group_report) and held to max_num_seqs.
    Refusal is reserved for UNREADABLE structure — never for "hybrid
    detected": a hybrid with healthy coverage passes."""
    return kv_coverage(rep, max_num_seqs=max_num_seqs, max_model_len=max_model_len)[0]


def kv_coverage(rep: Mapping[str, Any], *, max_num_seqs: int, max_model_len: int
                ) -> Tuple[Optional[str], Optional[KvCoverage]]:
    """:func:`kv_coverage_problem` WITH the facts: ``(problem, coverage)`` —
    ``coverage`` is the :class:`KvCoverage` read off the report whenever its
    structure is readable (passing or short), ``None`` when it is not (then
    ``problem`` names the unreadable field). A SHORTFALL is ``problem`` with
    ``coverage`` present; :func:`validate_scheduler_reports` raises
    :class:`KvCoverageShortfall` when every problem of a launch is one. The
    messages are :func:`kv_coverage_problem`'s, byte for byte."""
    groups = rep.get("kv_num_cache_groups")
    if isinstance(groups, bool) or not isinstance(groups, int) or groups <= 0:
        return (f"the KV-cache group structure could not be read off the "
                f"report (kv_num_cache_groups={groups!r}) — a geometry "
                "whose KV coverage cannot be verified must not pass "
                "readiness"), None
    if groups == 1:
        n_blocks = rep.get("kv_num_gpu_blocks")
        block_size = rep.get("kv_block_size")
        if (isinstance(n_blocks, bool) or not isinstance(n_blocks, int)
                or n_blocks <= 0 or isinstance(block_size, bool)
                or not isinstance(block_size, int) or block_size <= 0):
            return (f"the resolved KV cache size could not be read off the "
                    f"report (kv_num_gpu_blocks={n_blocks!r}, "
                    f"kv_block_size={block_size!r}) — a geometry whose KV "
                    "coverage cannot be verified must not pass readiness"), None
        kv_tokens = n_blocks * block_size
        budget = max_num_seqs * max_model_len
        coverage = KvCoverage(kv_tokens / max_model_len, kv_tokens, KV_BASIS_UNIFORM)
        if kv_tokens < budget:
            return (f"KV cache holds {kv_tokens} tokens < the {budget}-token "
                    f"budget (= {max_num_seqs} x {max_model_len}): only "
                    f"{kv_tokens / max_model_len:.2f}-way concurrency at full "
                    f"length is actually KV-backed, so the requested "
                    f"max_num_seqs={max_num_seqs} exists in argv but not in "
                    "memory — shrink the geometry or raise "
                    "gpu_memory_utilization"), coverage
        return None, coverage
    desc = _kv_group_desc(rep.get("kv_cache_groups"), groups)
    if desc is None:
        return (f"hybrid KV cache ({groups} groups): the per-group "
                f"structure could not be read off the report "
                f"(kv_cache_groups={rep.get('kv_cache_groups')!r}) — a "
                "geometry whose KV coverage cannot be verified must not "
                "pass readiness"), None
    mc = rep.get("kv_max_concurrency")
    if (isinstance(mc, bool) or not isinstance(mc, (int, float))
            or not math.isfinite(mc) or mc <= 0):
        return (f"hybrid KV cache ({desc}): the allocator's "
                f"max-concurrency verdict could not be read off the "
                f"report (kv_max_concurrency={mc!r}) — a geometry whose "
                "KV coverage cannot be verified must not pass readiness"), None
    coverage = KvCoverage(float(mc), int(mc * max_model_len), KV_BASIS_HYBRID)
    if mc < max_num_seqs:
        return (f"hybrid KV cache ({desc}) supports only {mc:.2f}-way "
                f"concurrency at max_model_len={max_model_len} "
                f"(~{int(mc * max_model_len)} tokens) < the requested "
                f"max_num_seqs={max_num_seqs}, so the requested width "
                "exists in argv but not in memory — shrink the geometry "
                "or raise gpu_memory_utilization (the uniform "
                "num_gpu_blocks x block_size arithmetic does not apply "
                "to a hybrid cache; this verdict is the allocator's own)"), coverage
    return None, coverage


def _kv_group_desc(groups: Any, expected_n: int) -> Optional[str]:
    """A one-line human description of the reported KV cache groups
    (``"2 x FullAttentionSpec(block_size=16) + 6 x MambaSpec(...)"``),
    or ``None`` when the structure is unreadable or inconsistent with
    the reported group count — the caller turns that into a refusal."""
    if not isinstance(groups, (list, tuple)) or len(groups) != expected_n:
        return None
    parts = []
    for g in groups:
        if not isinstance(g, Mapping) or not g.get("spec"):
            return None
        parts.append(f"{g.get('num_layers')} x {g['spec']}"
                     f"(block_size={g.get('block_size')})")
    return " + ".join(parts)


def validate_scheduler_reports(results: Any,
                               expected: Mapping[str, Any]
                               ) -> List[Dict[str, Any]]:
    """Every worker's resolved scheduler state must equal the composed
    invariant EXACTLY — a differing value is a silent clamp, a
    long-prefill threshold, a partial-prefill widening, or a
    speculative enable, each of which re-introduces scheduled shapes
    the plan lattice refuses (the in-proc smoke's silent-clamp
    assertion, applied to the WHOLE invariant at readiness). A key
    MISSING from a worker's report is config-surface drift and refuses
    just as loudly — an invariant that cannot be read is not an
    invariant that holds. ``enable_chunked_prefill`` is recorded, never
    asserted: V1 forces it on, and the guarantee is the sizing plus the
    pins."""
    if not isinstance(results, list) or not results:
        raise VllmServerError(
            f"scheduler-config RPC returned {results!r} — expected one "
            "resolved-geometry report PER WORKER; a server whose "
            "geometry cannot be read must not pass readiness")
    reports: List[Dict[str, Any]] = []
    problems: List[str] = []
    shortfalls: List[KvCoverage] = []
    for i, rep in enumerate(results):
        if not isinstance(rep, Mapping):
            problems.append(f"worker {i}: non-report result {rep!r}")
            continue
        for key, want in expected.items():
            if key not in rep:
                problems.append(
                    f"worker {i}: {key} MISSING from the resolved report "
                    "(config surface drift)")
                continue
            got = rep[key]
            if want is None:
                if got is not None:
                    problems.append(
                        f"worker {i}: {key} resolved to {got!r}, expected "
                        "none at all")
                continue
            if isinstance(want, str):
                if got != want:
                    problems.append(
                        f"worker {i}: {key} resolved to {got!r}, expected "
                        f"exactly {want!r}")
                continue
            if isinstance(want, (list, tuple)):
                # a composed LIST (the --lora-target-modules class): the
                # resolved value must be that exact sorted list — None is
                # "every module wrapped", not a tolerated default
                if not isinstance(got, list) or list(got) != list(want):
                    problems.append(
                        f"worker {i}: {key} resolved to {got!r}, composed "
                        f"{list(want)!r}")
                continue
            if isinstance(want, bool):
                if got is not want:
                    problems.append(
                        f"worker {i}: {key} resolved to {got!r}, composed "
                        f"{want!r}")
                continue
            try:
                got_i = int(got)
            except (TypeError, ValueError):
                problems.append(f"worker {i}: {key} = {got!r} (unreadable)")
                continue
            if got_i != want:
                problems.append(
                    f"worker {i}: {key} resolved to {got_i}, composed "
                    f"{want}")
        problem, coverage = kv_coverage(
            rep, max_num_seqs=int(expected["max_num_seqs"]),
            max_model_len=int(expected["max_model_len"]))
        if problem:
            problems.append(f"worker {i}: {problem}")
            if coverage is not None:          # readable facts that fall short: a SHORTFALL
                shortfalls.append(coverage)
        reports.append(dict(rep))
    if problems:
        message = (
            "resolved scheduler state does not match the composed "
            "no-chunking invariant — a silent clamp, a long-prefill "
            "threshold, or a speculative config re-introduces scheduled "
            "shapes the plan lattice refuses (alphasteer-class arms "
            "cannot survive them); the server must not pass "
            "readiness:\n- " + "\n- ".join(problems))
        if shortfalls and len(shortfalls) == len(problems):
            # EVERY problem line is a KV shortfall on a worker whose pins all
            # held (a shortfall adds one line; a pin problem or an unreadable
            # report adds a line with no coverage, so the counts disagree) —
            # the one refusal a KV-derived re-launch may act on
            tight = min(shortfalls, key=lambda c: c.kv_backed_concurrency)
            raise KvCoverageShortfall(
                message, kv_backed_concurrency=tight.kv_backed_concurrency,
                kv_capacity_tokens=tight.kv_capacity_tokens, basis=tight.basis,
                max_model_len=int(expected["max_model_len"]),
                max_num_seqs=int(expected["max_num_seqs"]))
        raise VllmServerError(message)
    return reports


def _verify_scheduler_geometry(launch: ServerLaunch, *,
                               timeout_s: float) -> Dict[str, Any]:
    """Readiness half 3 (lens launches): read every worker's RESOLVED
    scheduler config over the authenticated RPC and hold it to the
    composed no-chunking geometry — the strongest runtime evidence the
    pinned server exposes (the resolved engine config inside the worker
    process itself), mirroring the smoke's silent-clamp assertion.
    Returns the attestation fragment; raises on any mismatch."""
    expected = expected_scheduler_state(launch)
    payload = _post_collective_rpc(
        launch, SCHEDULER_RPC_METHOD, [], timeout_s=timeout_s,
        what="scheduler-geometry verification")
    reports = validate_scheduler_reports(
        payload.get("results") if isinstance(payload, Mapping) else None,
        expected)
    return {
        "rpc_method": SCHEDULER_RPC_METHOD,
        "expected": expected,
        "n_workers": len(reports),
        "worker_reports": reports,
        "note": ("resolved engine config read inside every worker over "
                 "the authenticated collective_rpc route — runtime "
                 "evidence, not argv; enable_chunked_prefill is recorded "
                 "(V1 forces it on). The invariant: max_num_batched_"
                 "tokens == max_num_seqs * max_model_len (the budget can "
                 "never bind mid-prompt), long_prefill_token_threshold "
                 "== 0 with single partial prefills (no per-request "
                 "chunk cap), the effective per-step scheduled-token "
                 "ceiling equals the budget, the resolved KV cache COVERS "
                 "the budget (max_num_seqs is a ceiling, not a grant — an "
                 "under-covered geometry silently preempts its way down "
                 "to a fraction of the requested width), NO speculative "
                 "config "
                 "(spec decode schedules q>1 decode steps, which the "
                 "plan lattice refuses), and the stock synchronous "
                 "scheduler class — pinned on argv (--no-async-"
                 "scheduling) because the installed 0.25.1 resolved "
                 "the ASYNC class when the flag was omitted (job "
                 "194568); a default is never a pin"),
    }


def expected_native_lora_state(launch: ServerLaunch) -> Dict[str, Any]:
    """What EVERY worker's loaded-adapter report must equal on a lora-
    native launch, computed by the LAUNCHER from the adapter directories
    the launch bound to each name — an independent second reading, so a
    worker that loaded different bytes (or the right bytes under the
    wrong name) cannot attest itself. ``{"n_adapters": N, "adapters":
    {name: {"dir", "fingerprint_bf16", "module_keys", "r", "lora_int_id"}}}``,
    the id being ``name``'s position + 1 in ``launch.lora_modules`` (the
    ONE sorted ``--lora-modules`` flag = the frontend's counter order;
    ``serving_attestation`` module docstring). An ASSERTED expectation
    the readiness check holds every worker to, never an inference from
    the report. Any refusal is a :class:`VllmServerError`."""
    if not launch.lora_native:
        raise VllmServerError(
            "expected_native_lora_state on a launch that is not lora_native "
            "— the per-name adapter binding is the lora-native readiness "
            "contract; a lens launch attests its install instead")
    # lazy: worker_reports imports torch, which this composer module
    # (imported by the CPU-light proxy) must not pull in at import time
    from sisteer.capeval.serving.worker_reports import LensWorkerError
    try:
        return _sa.expected_native_lora_state_for(
            launch.lora_modules, launch.lora_adapter_dirs)
    except _sa.ServingAttestationError as e:
        raise VllmServerError(str(e)) from e
    except LensWorkerError as e:  # FingerprintError: a directory that
        # cannot be fingerprinted cannot be attested — refuse by name
        raise VllmServerError(
            f"adapter fingerprint refused at compose time: {e}") from e


def validate_native_lora_reports(results: Any,
                                 expected: Mapping[str, Any]
                                 ) -> List[Dict[str, Any]]:
    """Every worker's loaded-adapter report must match the launcher's
    expectation PER SERVING NAME (``serving_attestation.
    native_lora_report_problems``): id set ``{1..N}`` and count N, then
    for every name the fingerprint, rank, module keys and bf16 dtype at
    that name's asserted id, every fingerprinted key carrying an execution
    wrapper AND an exact-name binding on it (``execution_bindings``, the
    factor vLLM's own lookup selects per module). Two adapters permuted
    between ids refuse naming both; so does a swapped-layer fingerprint, a
    rank or a module key drift, and a factor whose spelling the engine
    never looks up. Raises :class:`VllmServerError`; returns the reports."""
    problems = _sa.native_lora_report_problems(results, expected)
    if problems:
        raise VllmServerError(
            "the loaded adapters do not bind to the composed names — the "
            "server would apply the wrong weights under a serving name and "
            "must not pass readiness:\n- " + "\n- ".join(problems))
    return [dict(r) for r in results]


def validate_cudagraph_coverage(results: Any, launch: ServerLaunch
                                ) -> List[Dict[str, Any]]:
    """Every worker's RESOLVED CUDA-graph capture surface must cover the
    composed decode concurrency (``serving_attestation.
    cudagraph_coverage_problems`` over ``launch.max_num_seqs`` and the
    composed ``launch.cudagraph_capture_sizes``): a
    ``max_cudagraph_capture_size`` below ``max_num_seqs``, or a capture
    list whose largest size is, means the dispatcher runs every fuller
    decode batch EAGER (``v1/cudagraph_dispatcher.py`` ``dispatch()``
    @0.25.1 returns ``CUDAGraphMode.NONE`` for ``num_tokens > max_size``);
    a FULL capture surface (the dispatcher's FULL keys, reported off
    ``model_runner.cudagraph_dispatcher``) whose largest size is below
    ``max_num_seqs`` means the widest decode batches run PIECEWISE with
    attention eager (the 2026-09-16 S=51 collapse — vLLM captures FULL
    decode graphs only for capture sizes ``<= max_num_seqs``); a composed
    size missing from the resolved list means the engine did not take the
    composed surface — all while ``enforce_eager`` stays False and the mode
    non-NONE: graphs-on on the record, eager in production. Raises
    :class:`VllmServerError`; returns the reports."""
    if launch.max_num_seqs is None:
        raise VllmServerError(
            "this lora-native launch carries no composed max_num_seqs — the "
            "CUDA-graph capture coverage cannot be held to anything; compose "
            "through build_server_launch")
    problems = _sa.cudagraph_coverage_problems(
        results, max_num_seqs=int(launch.max_num_seqs),
        composed_capture_sizes=(tuple(launch.cudagraph_capture_sizes)
                                if launch.cudagraph_capture_sizes else None))
    if problems:
        raise VllmServerError(
            "the resolved CUDA-graph capture surface does not cover the "
            "composed decode concurrency — the dispatcher would run "
            "production decode EAGER under a graphs-on attestation; the "
            "server must not pass readiness:\n- " + "\n- ".join(problems))
    return [dict(r) for r in results]


def validate_execution_strides(results: Any, launch: ServerLaunch
                               ) -> List[Dict[str, Any]]:
    """Every worker's EXECUTING LoRA wrappers must lie inside the composed
    adapted class and index under the composed kernel ceiling
    (``serving_attestation.execution_stride_problems`` over
    ``launch.lora_target_modules`` / ``launch.lora_max_module_dim``): a
    wrapper whose suffix is outside ``--lora-target-modules``, or whose
    in/out stride exceeds the D the ceiling was sized from, means the
    pinned kernels index ``rows x`` a WIDER stride than the composer
    held ``max_num_seqs x max_model_len`` to (gemma's gate_up_proj at
    20,480 vs o_proj's 4,096). Raises
    :class:`VllmServerError`; returns the reports."""
    if launch.lora_max_module_dim is None or not launch.lora_target_modules:
        raise VllmServerError(
            "this lora-native launch carries no composed lora_max_module_dim / "
            "lora_target_modules — compose through build_server_launch; the executing "
            "LoRA wrappers' strides cannot be held to the kernel index ceiling, so the "
            "server must not pass readiness")
    problems = _sa.execution_stride_problems(
        results, target_modules=launch.lora_target_modules,
        max_module_dim=int(launch.lora_max_module_dim))
    if problems:
        raise VllmServerError(
            "the executing LoRA wrappers do not stay inside the composed adapted class / "
            "kernel index ceiling — the pinned kernels would index a wider stride than the "
            "geometry was held to; the server must not pass readiness:\n- "
            + "\n- ".join(problems))
    return [dict(r) for r in results]


def validate_moe_lora_rank(results: Any, launch: ServerLaunch) -> Dict[str, Any]:
    """THE ROUTED-EXPERT LoRA BUFFER RANK at readiness: every worker's fused-MoE
    LoRA wrappers (``worker_reports.lora_execution_moe_wrappers``, their buffer R read
    off ``w13_lora_a_stacked``) must be allocated at EXACTLY the composed
    ``launch.moe_lora_rank``, and the worker must report the patch ARMED at that
    value (``native_moe_lora.report``); a launch that composed no routed rank must
    report no expert wrapper and an unarmed patch. Raises :class:`VllmServerError`;
    returns the attestation fragment ``{composed, armed_per_worker, wrappers_per_worker}``."""
    problems: List[str] = []
    composed = launch.moe_lora_rank
    armed_per_worker: List[Any] = []
    wrappers_per_worker: List[Dict[str, int]] = []
    for i, r in enumerate(results or []):
        if not isinstance(r, Mapping):
            problems.append(f"worker {i}: report is not a mapping")
            continue
        armed = r.get("moe_lora_rank_armed")
        reported = r.get("moe_lora_rank_composed")
        moe = r.get("lora_execution_moe_wrappers") or {}
        ranks = {str(k): int(v.get("max_lora_rank")) for k, v in moe.items()
                 if isinstance(v, Mapping) and isinstance(v.get("max_lora_rank"), int)}
        armed_per_worker.append({"armed": armed, "composed": reported})
        wrappers_per_worker.append(ranks)
        if composed is None:
            if armed:
                problems.append(f"worker {i}: the routed-rank patch is ARMED (R={reported}) but the launch composed none")
            if ranks:
                problems.append(f"worker {i}: {len(ranks)} fused-MoE LoRA wrapper(s) exist but the launch composed no "
                                "routed rank — the expert buffers are at the dense cap; the composer did not learn this class")
            continue
        if armed is not True or reported != int(composed):
            problems.append(f"worker {i}: routed-rank patch armed={armed!r} composed={reported!r}, the launch composed "
                            f"{MOE_LORA_RANK_ENV}={composed}")
        if not ranks:
            problems.append(f"worker {i}: the launch composed {MOE_LORA_RANK_ENV}={composed} but the worker reports no "
                            "fused-MoE LoRA wrapper")
        bad = {k: v for k, v in ranks.items() if v != int(composed)}
        if bad:
            problems.append(f"worker {i}: {len(bad)} fused-MoE LoRA wrapper(s) allocated at R != {composed}: "
                            f"{dict(list(sorted(bad.items()))[:3])}")
    if problems:
        raise VllmServerError(
            "the fused-MoE LoRA buffers are not at the composed routed rank — the server must not pass readiness:\n- "
            + "\n- ".join(problems))
    return {"composed": composed, "armed_per_worker": armed_per_worker, "wrappers_per_worker": wrappers_per_worker,
            "env": MOE_LORA_RANK_ENV}


def _verify_native_lora_state(launch: ServerLaunch, *, timeout_s: float,
                              expected_lora: Optional[Dict[str, Any]] = None
                              ) -> Dict[str, Any]:
    """Readiness half 2 (lora-native launches): ONE authenticated
    ``sisteer_scheduler_config`` RPC, then both validators — the
    unchanged :func:`validate_scheduler_reports` over
    :func:`expected_scheduler_state` (geometry, KV coverage, prefix
    caching, ``enforce_eager False``, non-NONE cudagraph mode, no
    ``vllm_lens`` import, LoRA config, loaded count) and
    :func:`validate_native_lora_reports` over
    :func:`expected_native_lora_state` (the per-name binding). Returns
    the attestation fragments; raises on any mismatch. The launcher-side
    expectations are computed BEFORE the RPC so a compose-time
    fingerprint refusal never reaches the worker — ``launch_server``
    computes ``expected_lora`` before it even spawns the child and passes
    it in; ``None`` computes it here."""
    expected = expected_scheduler_state(launch)
    if expected_lora is None:
        expected_lora = expected_native_lora_state(launch)
    payload = _post_collective_rpc(
        launch, SCHEDULER_RPC_METHOD, [], timeout_s=timeout_s,
        what="lora-native engine verification")
    results = payload.get("results") if isinstance(payload, Mapping) else None
    reports = validate_scheduler_reports(results, expected)
    validate_native_lora_reports(reports, expected_lora)
    validate_cudagraph_coverage(reports, launch)
    validate_execution_strides(reports, launch)
    moe_lora_buffers = validate_moe_lora_rank(reports, launch)
    model_runner = validate_model_runner_reports(reports)
    residual_slot = verify_residual_slot(launch, timeout_s=timeout_s)
    return {
        "expected": expected,
        "expected_native_lora": expected_lora,
        "model_runner": model_runner,
        "residual_slot": residual_slot,
        "moe_lora_buffers": moe_lora_buffers,
        "scheduler_geometry": {
            "rpc_method": SCHEDULER_RPC_METHOD,
            "n_workers": len(reports),
            "worker_reports": reports,
            "note": ("resolved engine config read inside every worker over "
                     "the authenticated collective_rpc route — runtime "
                     "evidence, not argv: the no-chunking geometry + KV "
                     "coverage as for a lens launch, plus enforce_eager "
                     "False with a non-NONE CUDA-graph mode (config/vllm.py "
                     "@0.25.1 forces NONE iff eager), no vllm_lens module "
                     "imported, the resolved LoRA config, and every loaded "
                     "adapter's bf16 fingerprint bound per serving name to "
                     "its lora_int_id (serving_attestation.py)"),
        },
    }


def expected_residual_slot_state(launch: ServerLaunch) -> Dict[str, Any]:
    """:func:`serving_attestation.expected_residual_slot_state` over the launch's
    adapter binding, cross-checked against the composed ``residual_slot_armed``
    (the env flag): the two readings of the same directories must agree."""
    if not launch.lora_native:
        raise VllmServerError("expected_residual_slot_state on a launch that is not lora_native")
    from sisteer.capeval.serving.worker_reports import LensWorkerError
    try:
        expected = _sa.expected_residual_slot_state(launch.lora_modules, launch.lora_adapter_dirs)
    except (_sa.ServingAttestationError, LensWorkerError) as e:
        raise VllmServerError(f"residual slot expectation refused at compose time: {e}") from e
    if bool(expected["armed"]) != bool(launch.residual_slot_armed):
        raise VllmServerError(
            f"the composed launch says residual_slot_armed={launch.residual_slot_armed!r} but the "
            f"adapter directories {'carry' if expected['armed'] else 'carry no'} residual payload — "
            "the launch env flag and the bank disagree; refusing")
    if bool(expected.get("paired", False)) != bool(launch.residual_slot_paired):
        raise VllmServerError(
            f"the composed launch says residual_slot_paired={launch.residual_slot_paired!r} but the adapter "
            f"directories {'carry' if expected.get('paired') else 'carry no'} PAIRED payload (op "
            f"{RESIDUAL_SLOT_PAIRED_OP!r}) — the paired launch flag and the bank's second-gate buffers would "
            "disagree; refusing")
    return expected


def residual_slot_reports(launch: ServerLaunch, *, timeout_s: float) -> List[Dict[str, Any]]:
    """ONE authenticated ``sisteer_residual_slot_report`` RPC on a LIVE launch,
    returned as the per-worker report list (``worker_reports.residual_slot_report``'s
    shape) with no judgement — the standup's post-controls ledger check
    (``native_standup.unclaimed_failures_problems``) and the certification
    driver / gate read ``ledger_size`` / ``ledger_fault`` off it AFTER their
    probes have been claimed. Refuses (``VllmServerError``)
    a non-list body, an empty list, or a non-mapping report: a ledger that
    cannot be read is never "empty"."""
    payload = _post_collective_rpc(
        launch, _sa.RESIDUAL_SLOT_RPC_METHOD, [], timeout_s=timeout_s,
        what="residual slot report")
    results = payload.get("results") if isinstance(payload, Mapping) else None
    if not isinstance(results, list) or not results:
        raise VllmServerError(
            f"the residual slot report RPC returned no per-worker report ({results!r}) — the "
            "ledger state cannot be read; refusing to treat it as empty")
    for i, rep in enumerate(results):
        if not isinstance(rep, Mapping):
            raise VllmServerError(f"residual slot report: worker {i} answered a non-report {rep!r}")
    return [dict(r) for r in results]


def verify_residual_slot(launch: ServerLaunch, *, timeout_s: float) -> Dict[str, Any]:
    """Readiness half 3: ONE authenticated
    ``sisteer_residual_slot_report`` RPC, held to
    :func:`expected_residual_slot_state` by
    :func:`serving_attestation.residual_slot_problems` — armed iff a payload is
    registered; when armed, the op under its ONE name, the runner bound, every
    decoder class wrapped, every block hooked, every payload in its slot; the
    ledger empty. Returns the attestation's ``residual_slot`` block; raises
    :class:`VllmServerError` on any problem."""
    expected = expected_residual_slot_state(launch)
    payload = _post_collective_rpc(
        launch, _sa.RESIDUAL_SLOT_RPC_METHOD, [], timeout_s=timeout_s,
        what="residual slot attestation")
    results = payload.get("results") if isinstance(payload, Mapping) else None
    problems = _sa.residual_slot_problems(results, expected)
    if problems:
        raise VllmServerError(
            "the residual slot does not match the composed launch — a server that would write "
            "(or fail to write) the residual objects other than as attested must not pass "
            "readiness:\n- " + "\n- ".join(problems))
    return {"armed": bool(expected["armed"]), "rpc_method": _sa.RESIDUAL_SLOT_RPC_METHOD,
            "n_workers": len(results), "expected": dict(expected),
            "worker_reports": [dict(r) for r in results]}


def _install_steering_via_rpc(launch: ServerLaunch, *,
                              timeout_s: float) -> Dict[str, Any]:
    """Invoke ``sisteer_install_steering`` in every worker over the
    pinned release's server-mode collective_rpc route and validate the
    per-worker reports. Returns the attestation payload; any failure —
    route absent (404: dev mode not on / wrong vLLM), worker exception
    (5xx), no reports, mismatched reports — raises."""
    assert launch.lens_registry_dir is not None
    expected = expected_install_state(launch.lens_registry_dir)
    payload = _post_collective_rpc(
        launch, INSTALL_RPC_METHOD, [str(launch.lens_registry_dir)],
        timeout_s=timeout_s, what="steering install")
    reports = validate_install_reports(
        payload.get("results") if isinstance(payload, Mapping) else None,
        expected)
    return {
        "registry_dir": str(launch.lens_registry_dir),
        "rpc_method": INSTALL_RPC_METHOD,
        "rpc_url": f"{launch.base_url}/collective_rpc",
        "model": launch.model,
        "expected": expected,
        "n_workers": len(reports),
        "worker_reports": reports,
        "attested_at": time.time(),
        "gpu_verify_note": (
            "CPU/static-verified wiring: the /collective_rpc route, its "
            "request/response shapes, and the extension's report schema "
            "are read out of the pinned vLLM 0.25.1 tag and lens_worker; "
            "the live end-to-end (real workers importing the extension, "
            "dev route mounted, hooks steering real forwards) is "
            "GPU-VERIFY, exercised by the serving gate on a node"),
    }


def require_vllm() -> None:
    if not HAVE_VLLM:
        raise VllmServerError(
            "vllm is not importable here — server launch runs only in the "
            "pinned GPU serving venv (configs/capeval/vllm-requirements.txt"
            "; capeval D20 wheel pin). This is the CPU lane.")


def _port_already_listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def _readiness_lists_model(body: Any, model: str) -> bool:
    """Instance-specific readiness: the answering server must list the
    exact model THIS launch serves — a bare 200 from a stale process on
    the port must never pass as readiness."""
    if not isinstance(body, Mapping):
        return False
    data = body.get("data")
    return (isinstance(data, list)
            and any(isinstance(m, Mapping) and m.get("id") == model
                    for m in data))


#: The CUDA allocator setting the cluster's standup path exports
#: (scripts/sdfbase/full22/common.sh) and the Modal container did not — vLLM's post-init
#: kernel warm-up (flashinfer_autotune's full-batch dummy run through the Mamba2 mixer's
#: fp32 gated norm) needed a 22.5 GiB block beside the sized KV cache with 20.1 GiB free and
#: 11.2 GiB reserved-but-unallocated; the same standup on the cluster, same vLLM, same
#: geometry class, same util, passed with this exported. A DEFAULT: an exported value, the
#: launch env or extra_env still win.
ALLOCATOR_ENV = "PYTORCH_CUDA_ALLOC_CONF"
ALLOCATOR_DEFAULT = "expandable_segments:True"


def compose_server_env(launch: ServerLaunch, extra_env: Optional[Mapping[str, str]] = None
                       ) -> Dict[str, str]:
    """The vLLM server process's environment: the parent's, the allocator default when the
    parent exports none, then the launch's composed env (PYTHONPATH additive: the
    composed entry PREPENDS to anything inherited rather than clobbering it), then
    ``extra_env``. The per-launch API token is added by :func:`launch_server` alone."""
    env = dict(os.environ)
    env.setdefault(ALLOCATOR_ENV, ALLOCATOR_DEFAULT)
    launch_env = dict(launch.env)
    if "PYTHONPATH" in launch_env and os.environ.get("PYTHONPATH"):
        inherited = os.environ["PYTHONPATH"]
        if launch_env["PYTHONPATH"] not in inherited.split(os.pathsep):
            launch_env["PYTHONPATH"] += os.pathsep + inherited
        else:
            launch_env["PYTHONPATH"] = inherited
    env.update(launch_env)
    if extra_env:
        env.update(extra_env)
    return env


# --------------------------------------------------------------------------- #
# the protocol sampling CLAMP OVERRIDE                                        #
# --------------------------------------------------------------------------- #

def serving_interpreter() -> Path:
    """The SERVING interpreter: the python of the venv the ``vllm`` console
    script on PATH belongs to (``Path(which("vllm")).parent / "python"``) —
    the interpreter that already imports ``sisteer`` for
    ``--worker-extension-cls``. Refuses when ``vllm`` is not on PATH or the
    sibling ``python`` is not a file (the image holds TWO interpreters — the
    client venv must never run the server)."""
    which = shutil.which("vllm")
    if which is None:
        raise VllmServerError(
            "the sampling clamp override needs the serving interpreter, resolved from the `vllm` "
            "console script — `vllm` is not on PATH; activate the pinned serving venv first")
    python = Path(which).resolve().parent / "python"
    if not python.is_file():
        raise VllmServerError(
            f"the sampling clamp override needs the serving interpreter beside the `vllm` script "
            f"({python}), which is not a file — the console script is not a venv's")
    return python


def sampling_clamp_override_record(value: float) -> Dict[str, object]:
    """The override RECORD (``contracts.SAMPLING_CLAMP_OVERRIDE`` — ``{from,
    to, why}``) for ``value``, which must be EXACTLY the pinned protocol value
    (``contracts.BFCL_PROTOCOL_TEMPERATURE``): the record is hashed identity,
    so a different value is a different instrument and refuses here."""
    from sisteer.capeval.tinycap.contracts import (BFCL_PROTOCOL_TEMPERATURE, SAMPLING_CLAMP_OVERRIDE,
                                                   VLLM_MAX_TEMP_PINNED)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0.0 < float(value) < VLLM_MAX_TEMP_PINNED):
        raise VllmServerError(
            f"sampling clamp override {value!r} is not a temperature strictly between 0 and the wheel's "
            f"clamp {VLLM_MAX_TEMP_PINNED} (vLLM 0.25.1 _MAX_TEMP) — nothing to lift")
    if float(value) != float(BFCL_PROTOCOL_TEMPERATURE):
        raise VllmServerError(
            f"sampling clamp override {value!r} is not the pinned protocol value {BFCL_PROTOCOL_TEMPERATURE} "
            "(contracts.BFCL_PROTOCOL_TEMPERATURE, D13) — the override record is hashed identity, so an "
            "unpinned value is refused rather than recorded")
    return dict(SAMPLING_CLAMP_OVERRIDE)


def apply_sampling_clamp_override(cmd: Sequence[str], env: Mapping[str, str], value: float
                                  ) -> Tuple[List[str], Dict[str, str], Dict[str, object]]:
    """The sampling-clamp override over a composed ``vllm serve`` launch:
    ``(cmd, env, record)`` where the argv head ``["vllm", "serve", …]`` becomes
    ``[<serving python>, "-m", "sisteer.capeval.serving.vllm_entry",
    "--sisteer-max-temp", "<T>", "serve", …]`` — the SAME tail, byte for byte —
    and the child's ``PYTHONPATH`` leads with the live sisteer tree's ``src``
    (however this process found it — installed or path-injected; the staged
    tree in a Modal container, since the standup put it first on
    ``sys.path``), so the wrapper module resolves from the launched tree, not
    the image's mounted copy (:func:`compose_server_env` then prepends this to
    anything inherited). ``vllm_entry`` itself asserts the wheel's clamp value,
    sets the module global and asserts the fork multiprocess method before
    calling the stock CLI. The record is ``contracts.SAMPLING_CLAMP_OVERRIDE``."""
    from sisteer.capeval.serving import vllm_entry as _entry
    from sisteer.capeval.tinycap.contracts import FLAG_VLLM_ENTRY_MAX_TEMP, VLLM_ENTRY_MODULE
    record = sampling_clamp_override_record(value)
    cmd = list(cmd)
    if cmd[:2] != ["vllm", "serve"]:
        raise VllmServerError(
            f"the sampling clamp override wraps a `vllm serve` launch; the composed argv starts {cmd[:2]}")
    # code review: a topology the fork assertion does not cover is refused at
    # COMPOSE time too (the same rule vllm_entry applies at start), so a manifest flag that
    # spawns a frontend or forces spawn never reaches a server under the override
    why = _entry.spawn_topology_reason(cmd[1:])
    if why is not None:
        raise VllmServerError(f"the sampling clamp override refuses this launch topology: {why}")
    python = serving_interpreter()
    wrapped = [str(python), "-m", VLLM_ENTRY_MODULE, FLAG_VLLM_ENTRY_MAX_TEMP, f"{float(value):g}", *cmd[1:]]
    import sisteer
    src = str(Path(sisteer.__file__).resolve().parents[1])
    # round 6: the SHIM (armed by the override value in the environment) re-applies the clamp
    # override at startup in EVERY interpreter vLLM spawns (the Qwen3.5 server is forced to `spawn`);
    # round 9: delivered by exactly ONE route, chosen below
    shim = _entry.shim_dir()
    if not (shim / _entry.SHIM_FILENAME).is_file():
        raise VllmServerError(f"the clamp-override shim {shim / _entry.SHIM_FILENAME} is missing from the live sisteer tree — "
                              "a spawned vLLM child would re-clamp 0.001 to 0.01; refusing to compose the override")
    out_env = dict(env)
    # an inherited shim dir — the live tree's OR any other copy's (the staged tree's, when the standup's
    # `sisteer` is not the one the server imports) — is dropped: the route chosen below is the only one
    rest = [p for p in (out_env.get("PYTHONPATH") or "").split(os.pathsep)
            if p and p != src and not _entry.is_shim_file(Path(p) / _entry.SHIM_FILENAME)]
    out_env[_entry.SHIM_ENV] = f"{float(value):g}"
    out_env.pop(_entry.SHIM_PTH_ENV, None)
    # round 9: exactly ONE delivery route. Preferred: the .pth in the SERVING interpreter's
    # site-packages (boots the shim by absolute path in every interpreter using that site-packages,
    # spawned children included, before any sitecustomize a system or venv layer may ship) — and then
    # the shim dir stays OFF PYTHONPATH: both routes at once exec the shim twice (two module copies)
    # and, before the shim's marker-based de-dup, their finders recursed into each other. Fallback,
    # only when the .pth could not be written: the shim dir on PYTHONPATH as `sitecustomize`.
    try:
        pth = drop_shim_pth(python, shim / _entry.SHIM_FILENAME)
    except VllmServerError as e:
        pth = None
        fallback_why = str(e)
    if pth is not None:
        out_env["PYTHONPATH"] = os.pathsep.join([src, *rest])            # the live tree's src FIRST; NO shim dir
        out_env[_entry.SHIM_PTH_ENV] = str(pth)
        out_env[_entry.SHIM_ROUTE_ENV] = _entry.ROUTE_PTH
    else:
        out_env["PYTHONPATH"] = os.pathsep.join([src, str(shim), *rest])  # the fallback route: the shim as sitecustomize
        out_env[_entry.SHIM_ROUTE_ENV] = _entry.ROUTE_PYTHONPATH
        out_env["SISTEER_VLLM_SHIM_PTH_REFUSED"] = fallback_why[:400]
    return wrapped, out_env, record


def clamp_override_mechanism(env: Mapping[str, str]) -> str:
    """What `build_server_launch` records as the override's mechanism: the in-process override plus
    the shim through the ONE route the composer installed."""
    from sisteer.capeval.serving import vllm_entry as _entry
    return f"in-process (vllm_entry) + sitecustomize shim (SISTEER_VLLM_MAX_TEMP) via {env.get(_entry.SHIM_ROUTE_ENV)}"


def serving_purelib(python: Path) -> Path:
    """The serving interpreter's ``site-packages`` (``sysconfig.get_paths()["purelib"]``), asked of
    THAT interpreter — refuses when it cannot answer (never a guess)."""
    r = subprocess.run([str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                       text=True, capture_output=True, timeout=60)
    if r.returncode != 0 or not r.stdout.strip():
        raise VllmServerError(f"the serving interpreter {python} could not name its site-packages (rc {r.returncode}): "
                              f"{(r.stderr or r.stdout)[-400:]}")
    return Path(r.stdout.strip())


def drop_shim_pth(python: Path, shim_file: Path) -> Path:
    """Write the shim's ``.pth`` boot into the serving interpreter's site-packages (idempotent — the
    same line is left as is; a different line is overwritten) and return its path. Refuses when the
    directory is not writable: the override would then rest on the PYTHONPATH route alone, and a
    spawned child under a shadowing sitecustomize would re-clamp."""
    from sisteer.capeval.serving import vllm_entry as _entry
    purelib = serving_purelib(python)
    pth = purelib / _entry.SHIM_PTH_NAME
    line = _entry.pth_line(Path(shim_file).resolve()) + "\n"
    try:
        if not (pth.is_file() and pth.read_text() == line):
            pth.write_text(line)
    except OSError as e:
        raise VllmServerError(f"cannot drop the clamp-override shim's .pth into {purelib}: {e} — the override needs "
                              "a writable serving site-packages (the Modal container's is)")
    return pth




def launch_server(launch: ServerLaunch, *, log_path: Path,
                  ready_timeout_s: float = 900.0,
                  install_timeout_s: float = 600.0,
                  extra_env: Optional[Mapping[str, str]] = None
                  ) -> subprocess.Popen:
    """Launch the composed server and block until it is READY, which for
    a lens launch means THREE things, in order: ``/v1/models`` answers
    FOR THIS INSTANCE (an already-listening port refuses before spawn, a
    ready response must list the launched model, and the child must
    still be alive after the validated response — a stale process on the
    port can never satisfy readiness while the real child dies on its
    bind); then the steering wiring is actually INSTALLED and
    ATTESTED: ``sisteer_install_steering`` is invoked in every worker
    over the pinned release's collective_rpc route, each worker's report
    is validated against the registry's independently computed expected
    state (cells, hook layers, plan layer, ledger cells,
    registration_count == 1); and then the RESOLVED scheduler geometry
    is verified against the composed no-chunking sizing over the same
    authenticated route (``sisteer_scheduler_config`` — runtime
    evidence, never argv; a silent clamp re-enables mid-prompt chunking
    and fails readiness). The validated reports land in
    ``steering_attestation.json`` next to the server log. A lens server
    whose install or geometry cannot be attested is TERMINATED and this
    raises — an HTTP-healthy-but-unsteered (or silently-chunking) server
    must never receive audit traffic.

    A bootstrap failure here (vllm missing, engine init dies, readiness
    timeout, install refusal) raises — the CALLER (the §10 gate / the
    proxy sbatch) routes it through the recorded-unavailable path (the
    fail-loud-record-exit-clean pattern), never a silent skip. Every
    post-spawn failure shuts the child down through the one bounded path
    (:func:`_shutdown_child`: terminate -> wait -> kill -> wait) and
    leaves NO attestation file behind."""
    if shutil.which(launch.cmd[0]) is None:
        raise VllmServerError(
            f"{launch.cmd[0]!r} not on PATH — activate the pinned serving "
            "venv (vllm-requirements.txt) before launching")
    if launch.lens_registry_dir is not None and not launch.api_key:
        raise VllmServerError(
            "a lens launch carries no api_key — the dev RPC surface its "
            "install needs must never come up unauthenticated on a shared "
            "node; compose through build_server_launch, which mints the "
            "per-launch token")
    if launch.lora_native and not launch.api_key:
        raise VllmServerError(
            "a lora-native launch carries no api_key — the dev RPC surface "
            "its attestation reads through must never come up "
            "unauthenticated on a shared node; compose through "
            "build_server_launch, which mints the per-launch token")
    if _port_already_listening(launch.host, launch.port):
        raise VllmServerError(
            f"something is already listening on {launch.host}:{launch.port} "
            "— refusing to spawn: a stale server would satisfy the "
            "readiness probe and receive audit traffic while the new child "
            "dies on its bind conflict; free the port or stop the stale "
            "process first")
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    att_path = log_path.parent / STEERING_ATTESTATION_FILE
    serving_att_path = log_path.parent / SERVING_ATTESTATION_FILE
    # ATTESTATION LIFECYCLE, step 1: no attestation from any PREVIOUS
    # launch may survive into this one — lens-less launches included,
    # or a stale file would read as THIS launch's install. BOTH files,
    # on EVERY launch: a lens launch into a dir that last held a
    # lora-native server (or the reverse) must not leave the other
    # mode's record beside its own.
    _remove_attestation(att_path)
    _remove_attestation(serving_att_path)
    models_body: Any = None

    env = compose_server_env(launch, extra_env)
    # the per-launch token enters the CHILD's env only, at spawn — never
    # argv (world-readable /proc/<pid>/cmdline on a shared node), never
    # launch.env (recorded by the proxy as its attestation). The stock
    # vLLM middleware reads it for /v1*; SisteerTokenAuthMiddleware
    # reads the same variable for everything else.
    if launch.api_key:
        env[API_KEY_ENV] = launch.api_key
    # a lora-native launch's per-name expectation (the adapter fingerprints
    # off the directories) is computed BEFORE the child is spawned: a
    # fingerprint-contract refusal (non-unit scaling, a foreign target
    # module, a non-fp32 file) must never cost a GPU engine start
    expected_lora = (expected_native_lora_state(launch) if launch.lora_native
                     else None)
    log_file = open(log_path, "ab")
    proc = subprocess.Popen(list(launch.cmd), env=env, stdout=log_file,
                            stderr=subprocess.STDOUT)
    deadline = time.monotonic() + float(ready_timeout_s)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise VllmServerError(
                f"vllm serve exited with {proc.returncode} before becoming "
                f"ready — see {log_path}")
        http_ready = False
        try:
            body = _probe_models(launch)
            if body is not None:
                if not _readiness_lists_model(body, launch.model):
                    _shutdown_child(proc)
                    raise VllmServerError(
                        f"{launch.base_url}/v1/models answered 200 but does "
                        f"not list the launched model {launch.model!r} — a "
                        "stale or foreign server is answering on this "
                        "port; refusing readiness")
                if launch.lora_native:
                    # READINESS (lora-native) half 1: the listing must
                    # carry EVERY adapter name exactly once with the
                    # frontend's own root bound to the composed dir per
                    # name. The frontend registers static adapters BEFORE
                    # the port opens (init_app_state awaits
                    # init_static_loras ahead of serve_http @0.25.1), so
                    # a 200 without a card is a failure, not a race.
                    problems = _readiness_lists_models(body, launch)
                    if problems:
                        _shutdown_child(proc)
                        raise VllmServerError(
                            f"{launch.base_url}/v1/models does not bind "
                            "every composed adapter name to its directory "
                            "— refusing readiness:\n- "
                            + "\n- ".join(problems))
                    models_body = body
                if proc.poll() is not None:
                    raise VllmServerError(
                        f"vllm serve exited with {proc.returncode} right "
                        "after the readiness probe answered — the "
                        "response came from a stale process, not this "
                        f"launch; see {log_path}")
                http_ready = True
        except (urllib.error.URLError, OSError, ValueError):
            pass
        if http_ready:
            if launch.lens_registry_dir is not None:
                # READINESS HALF 2 (lens launches): actually install the
                # steering wiring in every worker and validate the
                # attestation — an HTTP-healthy server that never ran
                # sisteer_install_steering serves every arm as base, so
                # it must never be returned as "ready". Any failure —
                # route missing, worker exception, mismatched reports,
                # unattestable state, or the child dying around the
                # install — shuts the child down (bounded), removes any
                # attestation, and raises (outside the probe's except,
                # so nothing here can be swallowed as a retryable probe
                # error).
                try:
                    attestation = _install_steering_via_rpc(
                        launch, timeout_s=install_timeout_s)
                    # READINESS HALF 3: the RESOLVED scheduler geometry
                    # must equal the composed no-chunking sizing —
                    # runtime evidence over the same authenticated RPC
                    # route, because argv proves nothing about what the
                    # engine resolved (a silent clamp re-enables
                    # mid-prompt chunking, which the plan lattice
                    # refuses and plan-needing arms cannot survive).
                    attestation["scheduler_geometry"] = (
                        _verify_scheduler_geometry(
                            launch, timeout_s=install_timeout_s))
                    attestation.update(_attestation_bindings(launch, proc))
                    _sa.write_json_atomic(att_path, attestation)
                    # step 3: the attested child must STILL be the live,
                    # answering server.
                    _require_child_alive_after(
                        proc, launch, log_path, what="the steering install")
                except BaseException:
                    _remove_attestation(att_path)
                    _shutdown_child(proc)
                    raise
            elif launch.lora_native:
                # READINESS (lora-native) half 2: verify the RESOLVED
                # engine state and bind every adapter to its serving
                # name over the authenticated RPC, then publish
                # serving_attestation.json (kind lora_native — NEVER the
                # lens file) and re-check the child is still the live,
                # answering server. Any failure shuts the child down,
                # removes the attestation and raises — an HTTP-healthy
                # server that is eager, plugin-loaded, or applying the
                # wrong adapter under a name must never receive traffic.
                try:
                    verified = _verify_native_lora_state(
                        launch, timeout_s=install_timeout_s,
                        expected_lora=expected_lora)
                    bindings = _attestation_bindings(launch, proc)
                    # lens-only provenance; a lora-native record has no
                    # registry
                    bindings.pop("registry_index_sha16", None)
                    try:
                        attestation = _sa.build_serving_attestation(
                            model=launch.model, revision=launch.revision,
                            bindings=bindings,
                            expected=verified["expected"],
                            expected_native_lora=verified[
                                "expected_native_lora"],
                            scheduler_geometry=verified[
                                "scheduler_geometry"],
                            models_body=models_body,
                            model_runner={"class": verified["model_runner"]["class"],
                                          "kind": verified["model_runner"]["kind"]},
                            residual_slot=verified["residual_slot"],
                            kernel_ceiling={
                                _sa.LORA_KERNEL_CEILING_KEY: launch.lora_kernel_index_ceiling,
                                _sa.LORA_MAX_MODULE_DIM_KEY: launch.lora_max_module_dim,
                                _sa.LORA_TARGET_MODULES_KEY: list(launch.lora_target_modules)})
                    except _sa.ServingAttestationError as e:
                        raise VllmServerError(str(e)) from e
                    _sa.write_json_atomic(serving_att_path, attestation)
                    _require_child_alive_after(
                        proc, launch, log_path,
                        what="the lora-native attestation")
                except BaseException:
                    _remove_attestation(serving_att_path)
                    _shutdown_child(proc)
                    raise
            return proc
        time.sleep(3.0)
    _shutdown_child(proc)
    raise VllmServerError(
        f"vllm serve did not become ready within {ready_timeout_s}s — see "
        f"{log_path}")


def _readiness_lists_models(body: Any, launch: ServerLaunch) -> List[str]:
    """Lora-native readiness half 1 (``serving_attestation.
    native_models_problems``): ``[]`` iff the ``/v1/models`` body lists
    ``launch.model`` and every name in ``launch.lora_modules`` exactly
    once, each adapter card's ``root`` resolving to the directory the
    launch bound to that name."""
    return _sa.native_models_problems(
        body, model=launch.model, adapter_dirs=launch.lora_adapter_dirs)


def _require_child_alive_after(proc: subprocess.Popen, launch: ServerLaunch,
                               log_path: Path, *, what: str) -> None:
    """The attested child must STILL be the live, answering server — a
    worker that attested and then died would leave a truthful-looking
    attestation for a dead (or replaced) process. Raises
    :class:`VllmServerError`; the caller removes the attestation."""
    if proc.poll() is not None:
        raise VllmServerError(
            f"vllm serve exited with {proc.returncode} right after {what} "
            "— the attestation described a process that is already gone; "
            f"see {log_path}")
    try:
        alive = _probe_models(launch)
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise VllmServerError(
            f"the server stopped answering immediately after {what} was "
            f"written ({e}) — refusing readiness; see {log_path}") from e
    if alive is None or proc.poll() is not None:
        raise VllmServerError(
            "the server died (or stopped listing the model) immediately "
            f"after {what} was written — refusing readiness; see {log_path}")


def _probe_models(launch: ServerLaunch) -> Optional[Any]:
    """One authenticated GET of ``/v1/models``: the parsed body on 200,
    None on any non-200. Connection-level errors propagate (the
    readiness loop treats them as not-yet-ready; post-install callers
    treat them as a dead server)."""
    req = urllib.request.Request(f"{launch.base_url}/v1/models",
                                 headers=dict(launch.auth_headers))
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError:
        return None


def _shutdown_child(proc: subprocess.Popen,
                    stop_timeout_s: Optional[float] = None) -> None:
    """The ONE child-shutdown path, bounded: terminate -> wait ->
    kill -> wait. Every failure branch uses this — a bare terminate()
    leaves a SIGTERM-ignoring (or shutdown-wedged) engine holding the
    GPU and the port indefinitely. (The timeout resolves at CALL time
    so tests can shrink :data:`CHILD_STOP_TIMEOUT_S`.)"""
    if stop_timeout_s is None:
        stop_timeout_s = CHILD_STOP_TIMEOUT_S
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=stop_timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _attestation_bindings(launch: ServerLaunch,
                          proc: subprocess.Popen) -> Dict[str, Any]:
    """What ties an attestation to ONE live launch and nothing else:
    the child pid, this host, the Slurm job (when present), a hash of
    the exact argv (which carries no secret — the token is env-only),
    the registry's content identity, the bound port, and a per-launch
    random nonce so two attestations can never be confused even with
    every other field colliding. The token itself is deliberately NOT
    here — the attestation is a record, records get read broadly."""
    registry_index_sha = None
    if launch.lens_registry_dir is not None:
        index = Path(launch.lens_registry_dir) / "registry.json"
        if index.is_file():
            registry_index_sha = cm.sha256_hex16(index.read_bytes())
    return {
        "server_pid": proc.pid,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "argv_sha256": cm.sha256_hex16(
            "\x00".join(launch.cmd).encode("utf-8")),
        "registry_index_sha16": registry_index_sha,
        "host": launch.host,
        "port": launch.port,
        "nonce": secrets.token_hex(16),
    }


def _remove_attestation(att_path: Path) -> None:
    try:
        att_path.unlink()
    except FileNotFoundError:
        pass
