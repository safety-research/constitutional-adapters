"""The Petri routing proxy.

One OpenAI-compatible ``/v1/chat/completions`` endpoint (+ tool-call
passthrough for agent modality) whose VIRTUAL MODEL NAME selects the
defense — inspect_ai's target role never knows whether it is hitting
base, a steering object, a LoRA, a prompted baseline, or a STACKED
prompt+object cell.

STACKED CELLS. A route may carry BOTH ``si_prefix_hash`` and a
lens/LoRA serving path — the ``stacked__<si>__<method>`` cells, whose
intervention is the SI prefix AND the steered object together. Both
legs are enforced per request, so neither can silently drop: the prefix
is hash-verified exactly like a prompted baseline's (H8 — a stacked
request without its prefix REFUSES), and the object leg travels as the
injected selector/adapter-name whose ``expect_hash`` the worker
verifies. A registered SI prefix on a NON-stacked, non-prompted route
still refuses (``_refuse_unexpected_prefix``): stacking is a declared
identity, never an accident of a client smuggling a prefix.

Structure (deliberate, for CPU-testability without vLLM):

- :class:`RoutingProxy` — the framework-free async core: request
  validation, virtual-name -> route resolution, SI-prefix verification
  (H8), the per-path request rewrite (H3 selector IDs / LoRA adapter
  name / passthrough), the ``(virtual model -> intervention_hash)`` log,
  serving-identity stamping, and exact-replay serialization. Tests drive
  it with a stub :class:`Backend`.
- :class:`HTTPBackend` — the production transport to the server-mode
  vLLM (aiohttp, lazily imported).
- :func:`make_app` / :func:`main` — the aiohttp HTTP surface + CLI
  (cluster-internal bind only; reached over the SSH tunnel).
- :class:`ProxyClientEngine` — the ``runtime.make_engine`` handle for a
  ``proxy_vllm+…`` serving identity: a thin OpenAI-compatible client of
  the (tunnelled) proxy endpoint.

Loud refusals, never silent defaults: unknown virtual names, a missing
or wrong prompted SI prefix, client-smuggled selectors, streaming (not
yet supported — refused, not quietly buffered), ``n``/``best_of`` above 1
on a STEERED route (:class:`SteeredForkRefusal` — vLLM forks inside the
engine where the lens worker never sees the fork, so a forked sequence
may run UNSTEERED and be returned beside its steered sibling; k separate
requests instead, until steered forks are proven), and any route whose
serving identity is not ``proxy_vllm`` (a DEFENSIVE invariant — no audit
cell is hf-served anymore; the HF shim is the §10 gate's reference
oracle only) all refuse with a clear message.

PREFIX CACHING IS ON by default at the vLLM server, and THIS module
owns the isolation that makes it safe. The pinned v0.25.1 keys KV cache
blocks on token ids (+ the parent-hash chain, the LoRA NAME, and the
request ``cache_salt``, which salts block 0 and chains into every
descendant) — the steering selector is in the key NOWHERE, and cached
tokens skip the forward pass, so the lens hooks never fire on them.
Cross-cell reuse of an identical token prefix would silently serve the
WRONG intervention's KV. So ``_rewrite`` stamps EVERY upstream request
with a proxy-owned ``cache_salt``:

- unhooked legs (passthrough base, the prompted baselines) and LoRA
  arms: a CELL-STABLE salt — full within-cell prefix reuse (the ~5.9k
  CONSTITUTION prefix was the R7 cost; measured 1.33x at a 6.5k shared
  prefix), zero cross-cell reuse;
- SELECTOR-CARRYING (lens) legs: BY THE CELL'S ATTESTED
  RESUMED-PREFILL POLICY. The
  worker publishes, per served cell, ``lens_adapters.
  resumed_prefill_policy(ops)`` — ``always`` / ``conditional`` /
  ``never`` — in its install payload; the launcher reads it through
  ``lens_worker.resumed_prefill_policy_cells_attested`` and hands the
  map to this proxy. A cell attested ``always`` (per-position ops at
  all positions in both phases: each row's edit is computed from that
  row alone, so a prefix cached under the SAME cell and intervention
  carries exactly the edit a cold prefill would have given it) gets a
  CELL-STABLE lens salt and its prefills resume. EVERYTHING ELSE keeps
  the request-unique salt it always had — ``conditional``, ``never``,
  a cell absent from the map, and every case where nothing was
  attested at all. FAIL CLOSED is the rule: the proxy shares only what
  it can point at an attestation for, never what it failed to rule
  out. In particular a ``never`` (prefill-phase BROADCAST) cell —
  alphasteer, ``prefill_last_*``, ``antidote:last``, whose edit is
  computed from the LAST prompt token and written across every prompt
  position — is never routed onto a shared prefix BY THIS DECISION,
  rather than by relying on the worker to catch it: the worker's
  refusal for those is engine-fatal by design, so the proxy must not
  hand it the chance.

  The cell-stable lens salt is the unhooked one plus a ``|lens``
  marker, so a lens leg and an unhooked leg of the same (cell,
  intervention) can never land in ONE KV namespace even if a routing
  table ever named both — an isolation the salt-discipline guard would
  otherwise have to catch as a 409 after the fact.

  ``conditional`` (per-position ops under a ``response`` mask) is safe
  only when the cached region carries no assistant header, which the
  worker decides per request; the docs' §4 offers the proxy a
  single-turn refinement for it. NOT taken here: this landing shares
  on ``always`` alone, so the worker's conditional path stays
  unreachable through the proxy and can be turned on later against
  measured evidence rather than on the same day the sharing does.

  DECLARED CELLS SHARE NOTHING: a lens leg whose cell appears in the
  attested ``lens_serve_declarations`` map (a cell served out of its fit
  support — ``@thinking``, ``@all_turns``, ``@all_positions``) is minted
  REQUEST-UNIQUE whatever its
  resumed-prefill policy says, audit label ``declared-request-unique``: a
  transfer experiment's settings must differ in the mask alone, never in
  the cache regime, and an ``all``-positions cell would otherwise be an
  attested ``always`` and run warm beside cold siblings. And the whole
  regime can be pinned by declaration: ``lens_salt_policy=
  "request_unique_all"`` mints every lens leg
  request-unique, label ``forced-request-unique`` — the experiment-2
  standup's attested, recorded fact that its four lens routes all ran cold.

- RESTRICTED-MODE NATIVE routes (native-psr-serving stage 2): a lora
  route whose ATTESTED adapter carries a residual payload at a native
  position mode other than ``all`` (``response`` / ``prompt_last`` /
  ``prefill_all``) is minted REQUEST-UNIQUE, the lens's ``conditional`` /
  ``never`` rule transported — a masked write's KV at generated rows may
  not be resumed by a later request in which those rows are unsteered. An
  all-positions residual route stays cell-stable like a LoRA arm (every
  row's edit is computed from that row alone). The same routes are held
  to a REQUEST ALLOWLIST before forwarding and to the worker's per-request
  failure ledger before returning (``RESTRICTED_ROUTE_ALLOWED_KEYS``,
  ``_refuse_unlocated_native_response``).

KNOWN LIMITATION, stated rather than oversold: the salt covers every
block of a request INCLUDING a genuinely-unsteered shared SI prefix, so
the prompted prefix is NOT shared across different cells — recovering
that cross-cell win needs an SI-exclusion design, a separate landing.
``cache_salt`` is a FORBIDDEN client key: isolation is the proxy's to
inject, never a client knob. The proxy likewise refuses selector
smuggling so no client can widen its own serving surface.

THE SALT DISCIPLINE IS ALSO ENFORCED HERE, NOT ONLY IN THE ENGINE
(:meth:`RoutingProxy._guard_cache_salt`). The worker-side guard that
backs the isolation proof is deliberately FATAL: a steered request that
resumes cached rows raises inside the engine's generate path, EngineCore
treats the raise as unrecoverable, and the WHOLE vLLM server dies with
it (observed live: job 227309's same-salt resume killed the API server;
in a 3–5h audit that one request would fail everything after it). The
proxy already knows everything needed to refuse first — it mints every
salt and knows which routes are selector-carrying — so before any
request goes upstream it verifies, against the salts it has already
forwarded, that (a) a STEERED request's salt was never used before and
(b) an UNSTEERED request's salt was never used by a steered one. A
violation is a per-request 409, the engine never sees the request, and
the server keeps serving. The in-engine guard STAYS, as the backstop
for traffic that does not come through this proxy — defense in depth,
not a relaxation: nothing here widens what the engine would accept.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import re
import secrets
import urllib.error
import urllib.request
import uuid
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import (Any, Callable, Deque, Dict, List, Mapping, Optional, Protocol,
                    Sequence, Tuple)

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri.defense_cells import (CellSource,
                                                 NATIVE_POSITION_MODES,
                                                 NOOP_LENS_CELL_ID,
                                                 PetriCellSet,
                                                 VMODEL_PREFIX,
                                                 served_positions_of)
from sisteer.capeval.serving import serve_declaration as sd
from sisteer.capeval.serving import serving_attestation as _sa
from sisteer.capeval.serving import span_locator as sl
from sisteer.capeval.serving.vllm_server import (BATCH_INVARIANT_ENV,
                                                 EXACT_REPLAY_STRATEGIES,
                                                 LORA_ROUTING_RPC_METHODS)

log = logging.getLogger("petri.proxy")

#: the served lens REDUCTION MODE key a launcher's attestation may carry
#: (``per_row`` | ``stacked``) — the literal twin of
#: ``lens_worker.LENS_REDUCTION_MODE_KEY`` (pinned equal by
#: tests/test_lens_reduction_mode_attestation.py; a literal so this file
#: gains no import). Echoed in the /v1/models identity block ONLY when the
#: launcher attested it; never defaulted here — an absent key is the
#: consumer's "pre-landing, per-row" signal.
LENS_REDUCTION_MODE_KEY = "lens_reduction_mode"
#: the SERVER-DEFAULT SAMPLING POLICY the standup composed its launch under
#: (``vllm_server.SERVER_DEFAULT_POLICIES``: ``contract_override`` |
#: ``engine_defaults``). Handed over in ``launch_attestation`` beside the
#: launch env and echoed on the ``/v1/models`` ``petri_proxy`` identity block
#: ONLY when present (absent = a standup that predates the key; never
#: defaulted here), so a client reads what an OMITTED sampling key resolves
#: to off the proxy itself, without the server log.
SERVER_DEFAULT_POLICY_KEY = "server_default_policy"

#: the THREE-VALUED resumed-prefill policy vocabulary — the literal twin of
#: ``lens_adapters.RESUMED_POLICIES`` / ``RESUMED_ALWAYS``, pinned equal by
#: tests/test_proxy_resumed_prefill_salt.py. Literals for the same reason
#: :data:`LENS_REDUCTION_MODE_KEY` is one: ``lens_adapters`` imports torch and
#: this module must stay importable wherever the proxy runs. This mirrors the
#: three WORDS only — the per-cell TABLE is never re-derived here, it arrives
#: attested from the workers that adapted the ops
#: (``lens_worker.resumed_prefill_policy_cells_attested``).
RESUMED_PREFILL_ALWAYS = "always"
RESUMED_PREFILL_CONDITIONAL = "conditional"
RESUMED_PREFILL_NEVER = "never"
RESUMED_PREFILL_POLICIES = (RESUMED_PREFILL_NEVER, RESUMED_PREFILL_ALWAYS,
                            RESUMED_PREFILL_CONDITIONAL)

#: the KV-namespace REGIME a request's ``cache_salt`` was minted under
#: (:meth:`RoutingProxy.salt_regime_for`), recorded on every forwarded
#: request's audit row: ``cell-stable`` = a namespace shared by every request
#: of this (cell, intervention), so identical prefixes resume; each
#: ``request-unique`` salt is a namespace of one, so nothing ever resumes.
SALT_REGIME_CELL_STABLE = "cell-stable"
SALT_REGIME_REQUEST_UNIQUE = "request-unique"
SALT_REGIMES = (SALT_REGIME_CELL_STABLE, SALT_REGIME_REQUEST_UNIQUE)

#: the marker appended to a CELL-STABLE lens salt (module docstring): it makes
#: the steered namespace of a (cell, intervention) unequal to the unhooked one
#: by construction, and makes the regime readable off a salt in a log.
LENS_STABLE_SALT_MARKER = "lens"

#: the audit row's two resumed-prefill fields: they ride the in-memory
#: ``request_log`` AND the durable structured-logging record.
LEDGER_KEY_SALT_REGIME = "cache_salt_regime"
LEDGER_KEY_RESUMED_POLICY = "resumed_prefill_policy"

#: the audit row's ``resumed_prefill_policy`` when the value is not one of
#: :data:`RESUMED_PREFILL_POLICIES` — three DISTINCT non-policies, because
#: they fail closed for three different reasons and a reader must be able to
#: tell "this proxy was never told" from "this cell was not in what it was
#: told" from "no policy applies to this leg".
POLICY_NOT_CONSULTED = "not-consulted"   # an unhooked / LoRA leg: no selector
POLICY_UNATTESTED = "unattested"         # no attested map reached this proxy
POLICY_CELL_ABSENT = "absent"            # a map, but it does not name this cell
#: the fourth non-value: the leg's cell
#: is in the attested ``lens_serve_declarations`` map, so whatever policy it
#: was attested is OVERRIDDEN to request-unique — declared cells share nothing.
POLICY_DECLARED_SHARES_NOTHING = "declared-request-unique"
#: the fifth: ``lens_salt_policy ==
#: "request_unique_all"`` forced every lens leg request-unique, the map
#: unconsulted — the experiment-2 standup's recorded cache regime.
POLICY_FORCED_REQUEST_UNIQUE = "forced-request-unique"
POLICY_NON_VALUES = (POLICY_NOT_CONSULTED, POLICY_UNATTESTED, POLICY_CELL_ABSENT,
                     POLICY_DECLARED_SHARES_NOTHING, POLICY_FORCED_REQUEST_UNIQUE)

#: THE LENS SALT POLICY — the proxy's own recorded knob over which cache
#: regime its lens legs are minted under
#: (``capctrl_serving_up.py --lens-salt-policy``):
#: ``attested`` is today's behaviour exactly (the attested resumed-prefill
#: map + the declared-cells rule decide); ``request_unique_all`` mints EVERY
#: lens leg request-unique whatever the map says (label
#: :data:`POLICY_FORCED_REQUEST_UNIQUE`). Non-lens legs are untouched. The
#: proxy mints every salt, so its record (``models_payload`` /
#: ``endpoint.json.lens_salt_policy``) is the authority; a closed vocabulary.
LENS_SALT_POLICY_ATTESTED = "attested"
LENS_SALT_POLICY_REQUEST_UNIQUE_ALL = "request_unique_all"
LENS_SALT_POLICIES = (LENS_SALT_POLICY_ATTESTED, LENS_SALT_POLICY_REQUEST_UNIQUE_ALL)
LENS_SALT_POLICY_KEY = "lens_salt_policy"

#: the attested per-cell SERVE DECLARATIONS key — the literal twin of
#: ``lens_worker.LENS_SERVE_DECLARATIONS_KEY``
#: (pinned equal by tests/test_serve_declarations_attestation.py; a literal so
#: this module gains no torch import): ``{harness cell id (variant spelling):
#: {"serve_out_of_support": ["thinking"]} | {"serve_experiment": <name>}}``,
#: declaration-carrying cells only, handed over by the launcher
#: (``RoutingProxy(lens_serve_declarations=)``) and echoed in the /v1/models
#: identity block. ``lens_serve_declarations_error`` rides beside it.
SERVE_DECLARATIONS_KEY = "lens_serve_declarations"
SERVE_DECLARATIONS_ERROR_KEY = "lens_serve_declarations_error"
#: the served template's THINKING-HEADER FACT (``chat_markers.
#: thinking_header_facts``: ``{"switch", "moves", "off_ids", "on_ids"}``),
#: recorded in the routing table by ``export_routing_table(thinking_header=)``
#: and echoed in the identity block — what the ``enable_thinking`` validation
#: below decides from.
THINKING_HEADER_KEY = "thinking_header"
#: the reason the served template's thinking-ON header certification is
#: UNAVAILABLE (``thinking_header.on_ids`` null on a template WITH a switch —
#: Nemotron-3's OPEN ``<think>`` generation prompt; code review X1), handed
#: over by the launcher and echoed beside the fact in ``/v1/models``; null
#: beside a fully derived fact, exactly like ``lens_serve_declarations_error``
THINKING_HEADER_ERROR_KEY = "thinking_header_error"
#: the per-cell fitted-artifact digests —
#: ``{harness id (variant spelling): sha256}`` — a LAUNCH fact the standup
#: records in ``endpoint.json`` and hands over
#: (``RoutingProxy(object_artifact_sha256_by_cell=)``) for the identity
#: block; the proxy validates nothing against it.
OBJECT_SHA256_BY_CELL_KEY = "object_artifact_sha256_by_cell"

#: THE ROUTING TABLE'S VERSIONS.
#: Version 1 is every table the LENS standups write, byte-identical to
#: before this landing (``tests/fixtures/routing_table_v1_golden.json`` pins
#: the bytes). Version 2 is the LORA-NATIVE table: the same document plus a
#: TOP-LEVEL :data:`ADAPTERS_KEY` map ``{cell_id: adapter block}`` covering
#: exactly the lora routes. The block rides OUTSIDE :class:`RouteEntry` on
#: purpose (a nullable dataclass field would have put ``"adapter":
#: null`` into every lens table via ``asdict`` and broken every existing
#: ``RouteEntry(**r)`` loader), so the per-route schema is unchanged and the
#: old loaders keep reading version-1 tables while REFUSING version 2 by its
#: number rather than misreading it.
ROUTING_TABLE_VERSIONS = (1, 2)
ADAPTERS_KEY = "adapters"
#: the EXACT key set of one adapter block: what the standup
#: exported for the cell (``lora_export.AdapterDigests``), bound to the
#: route by ``name == cell_id`` and ``petri_intervention_hash ==
#: route.intervention_hash``. ``petri_zero_adapter`` is True exactly on a
#: zero-adapter control leg (a cell id ending in :data:`LORA_ZERO_CELL_SUFFIX`
#: — ``none__lora0``) and False on every genuine cell: a zero adapter
#: registered under a genuine cell's name would be base served under a
#: steered identity, and a genuine adapter under the control's name would
#: steer the no-op leg. ``petri_source_kind`` is the export kind (null on the
#: zero adapter).
ADAPTER_BLOCK_KEYS = ("name", "dir", "config_sha256", "weights_sha256", "r",
                      "petri_intervention_hash", "petri_zero_adapter",
                      "petri_source_kind")
#: the zero-adapter control leg's cell-id suffix (``defense_cells.
#: _CONTROL_SUFFIXES``' ``lora0`` spelled as a suffix; ``none__lora0`` is the
#: table's derived control) — the literal ``proxy_engine.scale0_twin``
#: resolves a lora route's twin by
LORA_ZERO_CELL_SUFFIX = "__lora0"
#: the serving paths that STEER the model — a lens hook or a native LoRA
#: adapter; passthrough and the prompted baselines do not. ONE literal for
#: every driver's bites gate (``proxy_engine.is_steered_route``,
#: ``bon_served.is_steered``), so no gate can drift to "is lens" again.
STEERED_ROUTE_PATHS = ("lens", "lora")
#: the two adapter files a block's digests are over — the ONE spelling in
#: ``serving_attestation`` (a stdlib-only leaf, so this module gains no torch
#: import); ``lora_export.ADAPTER_CONFIG`` / ``ADAPTER_WEIGHTS`` bind to it too
ADAPTER_CONFIG_FILE = _sa.ADAPTER_CONFIG_FILE
ADAPTER_WEIGHTS_FILE = _sa.ADAPTER_WEIGHTS_FILE

#: THE ENGINE ATTESTATION: what a LORA-NATIVE standup hands the
#: proxy from ``serving_attestation.json`` (``serving_attestation.
#: engine_identity(att)`` produces exactly this shape; D defines it, C reads
#: it, G pins them equal). Given, it binds every lora route's adapter block
#: to the adapter the server ATTESTED under that name (both digests), and
#: refuses any LENS route (a plugin-free server has no lens hook: a lens
#: route there would be base served under a ``+lens+`` identity). Echoed as
#: :data:`ENGINE_KEY` in ``/v1/models`` and stamped on every response.
ENGINE_KEY = "engine"
ENGINE_STATE_KEY = "engine_state"
#: the proxy's allowlisted pass-through to the native worker's LoRA ROUTING
#: RECORDER RPCs: the §10 gate's lora-native canary attaches to a RUNNING
#: standup and holds no launch token, so it reaches
#: ``sisteer_{arm,observe,disarm}_lora_routing_recorder`` through the proxy,
#: which holds the upstream bearer. Exactly the three methods, exactly on a
#: proxy started with a lora-native engine attestation; a lens proxy refuses
#: (no recorder exists there) and every lens route stays byte-identical.
LORA_ROUTING_RPC_ROUTE = "/sisteer/lora-routing/rpc"
ENGINE_KIND_LORA_NATIVE = _sa.SERVING_ATTESTATION_KIND
ENGINE_ATTESTATION_KEYS = ("kind", "enforce_eager", "cudagraph_mode", "lora_config",
                           "residual_slot", "adapters")
#: per attested adapter: the two file digests, the bf16 fingerprint, the
#: asserted ``lora_int_id``, the ``petri_intervention_hash`` the loaded
#: adapter's ``adapter_config.json`` carries (``None`` on the zero adapter)
#: and that config's ``petri_zero_adapter`` flag. The hash is held to every
#: GENUINE lora route's block at start: without it the standalone
#: ``--serving-attestation`` path accepted a regenerated table whose
#: route/block claimed intervention B over a server that attested it loaded
#: object A's adapter under that name, as long as the file digests were
#: copied over — rows then stamped with B's hash while A's weights served.
#: The zero flag is held to every lora route's block BEFORE the hash check:
#: with the table's flag alone deciding the exemption, a table that named a
#: GENUINE adapter ``none__lora0`` (digests copied over) started and returned
#: that steering stamped as the no-op control — the attested flag now has to
#: agree.
#: Since native-psr-serving (attestation schema 2): the adapter's object KIND
#: (``petri_source_kind``, None on the zero adapter), the residual payload's
#: ``residual_fingerprint`` and its ``position_mode`` (both None for a plain
#: LoRA) — the ``/v1/models`` card of a PSR / vector cell carries them, and the
#: engine block carries ``residual_slot: {"armed": bool}``.
ENGINE_ADAPTER_KEYS = ("config_sha256", "weights_sha256", "fingerprint_bf16",
                       "lora_int_id", "petri_intervention_hash",
                       "petri_zero_adapter", "petri_source_kind",
                       "residual_fingerprint", "position_mode")

#: THE RESTRICTED-MODE NATIVE ROUTE. A
#: lora route whose ATTESTED adapter carries a residual payload at a native
#: position mode other than ``all`` (``response`` / ``prompt_last`` /
#: ``prefill_all``) writes the object at a per-request SUBSET of rows the
#: worker locates from the request's prompt ids (the response boundary is the
#: LAST occurrence of the generation header). Three things follow for the
#: proxy, each below, none of which touches an all-positions residual route or
#: a plain LoRA route: (1) the request body is held to an ALLOWLIST before it
#: is forwarded, because the native path otherwise forwards bodies wholesale
#: and vLLM's chat protocol accepts render-changing options
#: (``truncate_prompt_tokens`` + ``truncation_side: right`` removes the header)
#: — an allowlist, never a denylist, since the next render-changing option is
#: not on any denylist yet; (2) the completion is held to the worker's
#: per-request failure ledger (``sisteer_take_request_failure``, the lens's
#: contract) BEFORE it is returned, so an UNLOCATED boundary (served unsteered
#: by the worker) is refused, never filed as the cell; (3) the KV-cache salt is
#: REQUEST-UNIQUE (:meth:`RoutingProxy.salt_regime_for`), the lens's rule for
#: a ``response`` / phase-restricted mask transported: KV computed under a
#: masked write may not be resumed by a later request where those rows are
#: unsteered. The mode is read off the ENGINE ATTESTATION's adapter entry
#: (what the server LOADED under that name), never off a table field — the
#: attestation is the authority the proxy already binds every lora route to.
NATIVE_POSITION_MODE_ALL = "all"
#: the body keys a restricted-mode route forwards (the
#: bodies the drivers actually send); ``continue_final_message`` and
#: ``add_generation_prompt`` ONLY in the seeded shape (final message assistant,
#: ``continue_final_message: true``, ``add_generation_prompt: false``) — the
#: one non-default render the worker's LAST-occurrence locator can place a
#: header in; the proxy's own ``cache_salt`` is added by :meth:`_rewrite`.
RESTRICTED_ROUTE_ALLOWED_KEYS = (
    "model", "messages", "max_tokens", "temperature", "top_p", "top_k", "seed",
    "stop", "n", "logprobs", "top_logprobs", "return_token_ids",
    "return_tokens_as_token_ids", "stream", "continue_final_message",
    "add_generation_prompt",
    # the two EXACT-LENGTH decode controls of the §7.2 closed loop (``closed_loop_via_proxy``:
    # ``min_tokens == max_tokens`` + ``ignore_eos``; the certification's throughput legs and
    # the timing harness) — LENGTH controls that never change the render or the response
    # boundary the allowlist guards (native-psr-serving)
    "min_tokens", "ignore_eos")
RESTRICTED_ROUTE_SEEDED_SHAPE_KEYS = ("continue_final_message", "add_generation_prompt")
#: THE BOUNDARY-REFUSAL CONTROL's marker, fail-closed by intent: proxy
#: vocabulary, never upstream (``_rewrite`` strips it). On a
#: restricted-mode route it admits exactly ONE body the allowlist otherwise refuses — the
#: HEADER-LESS render (a single user turn under ``add_generation_prompt: false``, no
#: ``continue_final_message``, ``n`` 1) — so the standup's sixth positive control can drive
#: a request whose response boundary the worker PROVABLY cannot locate through the whole
#: production path (engine -> worker ledger -> claim -> refusal). Such a completion can only
#: ever leave through the ``response_boundary_unlocated`` refusal; a 200 is the control's
#: failure signal (the P1 on 20a6064e: a clean ledger cannot tell a working join from a
#: broken one). A driver that sends the key gets the same refusal — nothing widens.
BOUNDARY_REFUSAL_CONTROL_KEY = "sisteer_boundary_refusal_control"
RESTRICTED_ROUTE_CONTROL_KEYS = (BOUNDARY_REFUSAL_CONTROL_KEY,)
#: the control key outside its ONE body (HTTP 400): the marker is admitted ONLY beside EXACTLY
#: ``serving_attestation.BOUNDARY_REFUSAL_CONTROL_MESSAGES`` + ``_KWARGS`` (plus ``model``) — never a
#: multi-turn transcript whose earlier assistant header the locator would find
BOUNDARY_CONTROL_SHAPE_INVALID = "boundary_control_shape_invalid"
#: the control key with a value other than THIS proxy's nonce (HTTP 400): the standup reads the nonce
#: off its own RoutingProxy; no client outside it can present the marker
BOUNDARY_CONTROL_NONCE_INVALID = "boundary_control_nonce_invalid"
#: the body keys of the one admitted probe (the marker + model + the fixed kwargs + messages)
BOUNDARY_REFUSAL_CONTROL_BODY_KEYS = frozenset({"model", "messages", BOUNDARY_REFUSAL_CONTROL_KEY}
                                               | set(_sa.BOUNDARY_REFUSAL_CONTROL_KWARGS))
#: the control key on a route with no response boundary to fail (HTTP 400)
BOUNDARY_CONTROL_NOT_RESTRICTED = "boundary_control_not_restricted"
#: a body key outside the allowlist on a restricted-mode route (HTTP 400): the
#: driver-visible behaviour change — widen the list deliberately
RESTRICTED_ROUTE_OPTION_UNSUPPORTED = "restricted_route_option_unsupported"
#: the worker ledgered this request's response boundary as UNLOCATED and served
#: it unsteered — the completion is refused (HTTP 500, the lens's
#: ``lens_request_failed`` shape)
RESPONSE_BOUNDARY_UNLOCATED = "response_boundary_unlocated"
#: the worker's ledger LATCHED its fault (``REQUEST_FAILURES_CAP`` unclaimed
#: entries, fact 15): every claim on a restricted-mode route answers the fault
#: payload, so every such completion is refused (HTTP 500) until the standup is
#: replaced — the engine and every all-positions / LoRA route keep serving
RESPONSE_BOUNDARY_LEDGER_FAULT = "response_boundary_ledger_fault"
#: the post-check could not be made at all — a response without an ``id``, a
#: backend with no claim method (HTTP 502, the lens's ``lens_postcheck_failed``
#: shape; the transport failure inside :meth:`HTTPBackend.take_request_failure`
#: keeps that code, the RPC being the same one)
RESPONSE_BOUNDARY_POSTCHECK_FAILED = "response_boundary_postcheck_failed"
#: ``/v1/completions`` — the proxy serves chat completions only (a raw prompt
#: has no header the worker could locate a boundary against); refused on every
#: route, with the restricted-route code on a restricted-mode route
COMPLETIONS_UNSUPPORTED = "completions_unsupported"
#: the audit row's two native-residual fields, on EVERY row (additive; the
#: stamp contract ``proxy_engine.STAMP_FIELDS`` is unchanged): the route's
#: native position mode (``None`` off a residual route) and what the
#: per-request failure claim read for this row
LEDGER_KEY_POSITION_MODE = "position_mode"
LEDGER_KEY_FAILURE_CLAIM = "request_failure_claim"
FAILURE_CLAIM_NOT_APPLICABLE = "not-applicable"   # no post-check on this route
FAILURE_CLAIM_CLEAN = "clean"                     # claimed: no failure recorded
FAILURE_CLAIM_UNLOCATED = "unlocated"             # claimed an entry -> refused
FAILURE_CLAIM_LEDGER_FAULT = "ledger-fault"       # the fault payload -> refused
FAILURE_CLAIM_UNVERIFIABLE = "unverifiable"       # no id / no RPC -> refused
FAILURE_CLAIM_LABELS = (FAILURE_CLAIM_NOT_APPLICABLE, FAILURE_CLAIM_CLEAN,
                        FAILURE_CLAIM_UNLOCATED, FAILURE_CLAIM_LEDGER_FAULT,
                        FAILURE_CLAIM_UNVERIFIABLE)
#: the audit-row label a REFUSED completion's row keeps, by refusal code (the
#: lens's two codes map onto the same labels: one claim, two engines)
_FAILURE_CLAIM_BY_CODE = {RESPONSE_BOUNDARY_UNLOCATED: FAILURE_CLAIM_UNLOCATED,
                          RESPONSE_BOUNDARY_LEDGER_FAULT: FAILURE_CLAIM_LEDGER_FAULT,
                          RESPONSE_BOUNDARY_POSTCHECK_FAILED: FAILURE_CLAIM_UNVERIFIABLE,
                          "lens_request_failed": FAILURE_CLAIM_UNLOCATED,
                          "lens_postcheck_failed": FAILURE_CLAIM_UNVERIFIABLE}

#: THE THINKING-ON DECLARATION (the third client-visible declaration,
#: beside :data:`RESPONSE_PREFILL_KEY` and
#: :data:`AGENT_LOOP_KEY`): ``petri_thinking_on: true`` declares that the
#: request renders THINKING-ON (``chat_template_kwargs.enable_thinking:
#: true``) on a lens cell whose hashed declaration admits that render
#: (``serve_out_of_support: ["thinking"]``, spelled ``<cell>@thinking``), so
#: the worker locates the response boundary against the cell's declared
#: thinking-ON header. REQUIRED on every lens-route request that sends
#: ``enable_thinking: true`` whose cell serves a HEADER-LOCATED mask
#: (:data:`HEADER_LOCATED_MODES`) — or whose served mask this proxy cannot
#: read (§2.5.4, narrowed 2026-09-09: the declaration protects a mask that
#: is LOCATED by matching the OFF header at the tail, which the ON render
#: moves; a cell served at ``all`` / ``last`` masks every row / the last row
#: BY CONSTRUCTION, reads no header and no ``thinking_on`` fact, so an
#: undeclared ON render there can mis-mask nothing and is accepted, the
#: regime still recorded). Which cells are header-located is a fact the
#: routing table STATES (:data:`LENS_SERVED_POSITIONS_KEY`); absent the fact
#: — an older table, a cell the map does not name — the uniform rule holds
#: and the undeclared render refuses, so the proxy never has to be right
#: about a cell's mask in order to refuse. INJECTED into the selector
#: (``thinking_on``) only where the served template's switch MOVES the
#: header (``thinking_header.moves``), since elsewhere the mask is identical
#: and the declaration is validated and recorded, not injected. The key
#: never travels upstream; ``chat_template_kwargs`` does (vLLM's own key,
#: merged over the launch's thinking-OFF pin).
THINKING_ON_KEY = "petri_thinking_on"
CHAT_TEMPLATE_KWARGS_KEY = "chat_template_kwargs"
ENABLE_THINKING_KEY = "enable_thinking"
#: admitted beside :data:`RESTRICTED_ROUTE_ALLOWED_KEYS` because the proxy
#: already GOVERNS its whole content: ``chat_template_kwargs`` may carry
#: ``enable_thinking`` and nothing else (:meth:`RoutingProxy._validate_kwargs_shape`);
#: on a restricted-mode route any other key inside it refuses under the
#: restricted-route code, before the thinking gate reads it
RESTRICTED_ROUTE_GOVERNED_KEYS = (CHAT_TEMPLATE_KWARGS_KEY,)
#: THE PROXY'S THINKING MODE (the sampled-rerun lineage's standup-wide regime,
#: docs/served_thinking_on.md) — a closed, recorded standup-level fact
#: (``endpoint.json.thinking_mode`` / ``/v1/models.petri_proxy.thinking_mode``).
#: ``off`` (the default, every proxy to date): the render regime is PER REQUEST
#: and the matrix below governs unchanged — a declared ``petri_thinking_on`` beside
#: ``enable_thinking: true`` is validated and served exactly as before (the
#: paired-GSM8K / MATH-500 instruments' path). ``on`` (``mt_serving_up
#: --thinking-on``: the vLLM launch pinned ``enable_thinking: true`` for EVERY
#: render): the launch pin is the regime, so a request that omits the kwarg
#: renders ON and is read as ON; ``enable_thinking: false`` refuses
#: ``thinking_regime_mismatch`` (an OFF render under an ON attestation); EVERY
#: lens-route request must carry ``petri_thinking_on: true`` whatever the cell's
#: served positions (``thinking_on_undeclared`` otherwise — the standup never has
#: to be right about which mask moves in order to refuse); and an undeclared lens
#: cell (an all-position fit) is validated and recorded, not refused, since the
#: standup declared exactly the header-dependent cells. An ON proxy refuses to
#: START without a certified header fact (on_ids present) and an attested
#: ``lens_serve_declarations`` map (fail closed at startup, never per request).
THINKING_MODE_OFF = "off"
THINKING_MODE_ON = "on"
THINKING_MODES = (THINKING_MODE_OFF, THINKING_MODE_ON)
THINKING_MODE_KEY = "thinking_mode"
THINKING_DECLARATION_KEY_KEY = "thinking_declaration_key"
#: the §2.5.6 refusal matrix's codes, one literal each (a driver recognises a
#: refusal by the code off the wire, never by prose)
THINKING_KWARGS_INVALID = "thinking_kwargs_invalid"
THINKING_KWARGS_INERT = "thinking_kwargs_inert"
THINKING_ON_UNDECLARED = "thinking_on_undeclared"
THINKING_ON_INVALID = "thinking_on_invalid"
THINKING_ON_MISMATCH = "thinking_on_mismatch"
THINKING_ON_NOT_LENS = "thinking_on_not_lens"
THINKING_ON_CELL_NOT_DECLARED = "thinking_on_cell_not_declared"
THINKING_ON_UNATTESTED = "thinking_on_unattested"
#: a thinking-ON proxy (``thinking_mode="on"``) refuses a request that flips the
#: pinned regime back OFF (``enable_thinking: false`` on chat or tokenize)
THINKING_REGIME_MISMATCH = "thinking_regime_mismatch"
THINKING_REFUSAL_CODES = (THINKING_KWARGS_INVALID, THINKING_KWARGS_INERT,
                          THINKING_ON_UNDECLARED, THINKING_ON_INVALID,
                          THINKING_ON_MISMATCH, THINKING_ON_NOT_LENS,
                          THINKING_ON_CELL_NOT_DECLARED, THINKING_ON_UNATTESTED,
                          THINKING_REGIME_MISMATCH)
#: the RENDER REGIME recorded on EVERY forwarded request's audit row and
#: echoed in the response's ``petri`` block (§2.5.5 / §2.5.11), from the
#: validated ``chat_template_kwargs``: ``enable_thinking: true`` ->
#: ``thinking_on``, ``false`` -> ``thinking_off``, key absent -> ``unspecified``.
LEDGER_KEY_RENDER_REGIME = "render_regime"
RENDER_REGIME_THINKING_ON = "thinking_on"
RENDER_REGIME_THINKING_OFF = "thinking_off"
RENDER_REGIME_UNSPECIFIED = "unspecified"
RENDER_REGIMES = (RENDER_REGIME_THINKING_ON, RENDER_REGIME_THINKING_OFF,
                  RENDER_REGIME_UNSPECIFIED)

#: THE ALL-TURNS RENDER CHECK: on a lens route served at
#: ``positions="response_all_turns"`` the proxy
#: recomputes the all-turns spans on the request's render
#: (``span_locator.assistant_turn_spans`` — the worker's own arithmetic on
#: the same ids) and checks them against the MESSAGE-derived expectation
#: (``chat_markers.assistant_content_spans``, widened to header + content +
#: end scaffold) before forwarding; a disagreement is a 400 here, counted,
#: never a served mask the messages did not describe. Four codes:
ALL_TURNS_SPAN_MISMATCH = "all_turns_span_mismatch"
ALL_TURNS_RENDER_REFUSED = "all_turns_render_refused"
ALL_TURNS_EXPECTED_UNAVAILABLE = "all_turns_expected_unavailable"
ALL_TURNS_RENDER_FAILED = "all_turns_render_failed"
ALL_TURNS_REFUSAL_CODES = (ALL_TURNS_SPAN_MISMATCH, ALL_TURNS_RENDER_REFUSED,
                           ALL_TURNS_EXPECTED_UNAVAILABLE, ALL_TURNS_RENDER_FAILED)
#: the audit row / structured-log label of the check's outcome
LEDGER_KEY_ALL_TURNS_CHECK = "all_turns_check"
ALL_TURNS_CHECK_OK = "ok"
ALL_TURNS_CHECK_MISMATCH = "mismatch"
ALL_TURNS_CHECK_REFUSED = "refused"
ALL_TURNS_CHECK_UNAVAILABLE = "unavailable"
ALL_TURNS_CHECK_FAILED = "failed"
ALL_TURNS_CHECK_LABELS = (ALL_TURNS_CHECK_OK, ALL_TURNS_CHECK_MISMATCH,
                          ALL_TURNS_CHECK_REFUSED, ALL_TURNS_CHECK_UNAVAILABLE,
                          ALL_TURNS_CHECK_FAILED)
_ALL_TURNS_CODE_LABEL = {ALL_TURNS_SPAN_MISMATCH: ALL_TURNS_CHECK_MISMATCH,
                         ALL_TURNS_RENDER_REFUSED: ALL_TURNS_CHECK_REFUSED,
                         ALL_TURNS_EXPECTED_UNAVAILABLE: ALL_TURNS_CHECK_UNAVAILABLE,
                         ALL_TURNS_RENDER_FAILED: ALL_TURNS_CHECK_FAILED}
#: the served position mode the check is keyed on — the literal twin of
#: ``lens_adapters.POSITIONS_RESPONSE_ALL_TURNS`` (a literal: no torch here;
#: pinned equal by tests/test_all_turns_render_check.py)
POSITIONS_RESPONSE_ALL_TURNS = "response_all_turns"
#: the rest of the SERVED position vocabulary, literal twins of
#: ``lens_adapters.POSITIONS_ALL`` / ``POSITIONS_LAST`` / ``POSITIONS_RESPONSE``
#: and of ``lens_adapters.POSITION_MODES`` (pinned equal by
#: tests/test_all_turns_render_check.py; no torch here)
POSITIONS_ALL = "all"
POSITIONS_LAST = "last"
POSITIONS_RESPONSE = "response"
POSITION_MODES = (POSITIONS_ALL, POSITIONS_LAST, POSITIONS_RESPONSE,
                  POSITIONS_RESPONSE_ALL_TURNS)
#: the HEADER-LOCATED modes — the literal twin of
#: ``lens_adapters.RESPONSE_MASK_MODES``: an op in either LOCATES its mask by
#: matching the cell's OFF assistant header at the prompt tail, which a
#: thinking-ON render moves; the ``all`` / ``last`` masks are every row / the
#: last row by construction and read no header. Header-locatedness is an OP
#: property: a cell with ANY such op is header-located (a mixed set included).
HEADER_LOCATED_MODES = (POSITIONS_RESPONSE, POSITIONS_RESPONSE_ALL_TURNS)
#: the per-cell SERVED POSITIONS map (thinking-ON gate, §2.5.4 narrowing):
#: ``{harness cell id (variant spelling): [<served mode>, ...]}`` over the
#: LENS-served legs whose resolved spec STATES a served mode (stacked cells
#: unwrapped to the object leg — ``defense_cells.served_positions_of``),
#: stamped into the routing table by :func:`export_routing_table` from the
#: same resolved specs the worker registry binds its ops to
#: (``lens_steer._assert_ops_express_spec_positions``), read back by
#: :func:`load_routing_table`, handed over as
#: ``RoutingProxy(lens_served_positions=)`` and echoed in ``/v1/models``. A
#: lens leg whose spec states no mode (the ``none__lens0`` twin, which has no
#: spec of its own; an artifact-defaulted cell) is ABSENT from the map, and
#: the gate reads absence — of the cell, or of the whole map (an older
#: table) — as header-located: today's refusal, fail closed.
LENS_SERVED_POSITIONS_KEY = "lens_served_positions"
#: the ENGINE-NEUTRAL served-positions map (native-positions-resolve):
#: ``{harness cell id: [<served mode>, ...]}`` over EVERY served leg whose
#: positions are attested — the lens legs' stated modes (the same entries as
#: :data:`LENS_SERVED_POSITIONS_KEY`, which stays for old readers) and, on a
#: lora-native table, each lora route's attested ``petri_position_mode``
#: (:func:`served_positions_stamp`). Written by :func:`export_routing_table`
#: iff it names a cell (a table with no attested mode gains no key, so the
#: version-1 golden stays byte-identical), validated back by
#: :func:`load_routing_table`, copied into endpoint.json by both standups, and
#: read FIRST by the served driver (``served_capability_run
#: ._standup_served_positions``), which falls back to the lens map on an old
#: record. A route that attests no mode is ABSENT — never defaulted from the
#: engine's name; the readers refuse it by name.
SERVED_POSITIONS_KEY = "served_positions"


def _canonical_v1_base(url: str) -> str:
    """The ONE canonical OpenAI base representation: ``<root>/v1``, no
    trailing slash. Legacy root URLs (``http://host:port``) normalize by
    appending ``/v1`` once, HERE, at construction — engine and target
    client then speak the same form, and the engine's paths are plain
    ``/models`` / ``/chat/completions`` (never ``/v1/v1/...``)."""
    if not url:
        raise ProxyError("empty base URL")
    base = url.rstrip("/")
    while base.endswith("/v1/v1"):  # collapse an accidental double append
        base = base[: -len("/v1")]
    if not base.endswith("/v1"):
        base = base + "/v1"
    return base

#: The upstream key vLLM's OpenAI frontend maps into
#: ``SamplingParams.extra_args`` (the H3 selector transport). Pinned here
#: as ONE constant so a vLLM rename at the launch-time pin is a one-line
#: edit; the §10 gate exercises it against the real pinned server.
UPSTREAM_EXTRA_ARGS_KEY = "vllm_xargs"
#: Keys a CLIENT request may never carry — selectors are the proxy's to
#: inject, and a client-smuggled selector is a refusal, not an override.
#: ORGANISM OVERLAY (em-misalign, 2026-09-11): a virtual model ``<vmodel>@<organism>``
#: routes exactly as ``<vmodel>`` (same cell, same lens selector, same prefix
#: verification) but names ``<organism>`` — a vLLM LoRA module loaded at launch,
#: an EM model organism — as the UPSTREAM model, so the steering object is served
#: on top of the adapter in one forward pass. Organisms are declared in the routing
#: table (``organisms``) by the standup that loaded them; a ``lora``-path leg cannot
#: be overlaid (its upstream model IS its adapter). The organism rides in the cache
#: salt's stable part (vLLM keys blocks by LoRA name natively; belt and braces).
ORGANISM_SEP = "@"
ORGANISMS_KEY = "organisms"
FORBIDDEN_CLIENT_KEYS = (UPSTREAM_EXTRA_ARGS_KEY, "petri_object",
                         "expect_hash", "extra_args", "cache_salt")

#: The ONE client-visible declaration key of the served-loglikelihood
#: landing: ``petri_response_prefill: true`` declares that the request's
#: final assistant message is a SEED (assistant prefix forcing, sent with
#: ``continue_final_message``) for a teacher-forced scoring read, so a
#: lens route's response mask must be LOCATED at the header inside the
#: prompt (header + seed + generated all masked) instead of anchored at
#: the prompt tail. The proxy verifies the declaration matches the
#: request's actual shape (:meth:`RoutingProxy._validate_response_prefill`)
#: and translates it into the injected selector's ``response_prefill``
#: fact; the key itself never travels upstream. It CANNOT be set
#: accidentally (an explicit ``true`` on a matching request shape, on a
#: lens route, is the only accepted form) and CANNOT widen a regime (the
#: worker consults it only to LOCATE a response op's boundary — the
#: position MODE each op serves is baked in the registry and untouched).
RESPONSE_PREFILL_KEY = "petri_response_prefill"

#: The SECOND client-visible declaration — the multi-turn Petri landing
#: (2026-09-02), modeled on :data:`RESPONSE_PREFILL_KEY` and travelling
#: the same path: ``petri_agent_loop: true`` declares that the request is
#: one of the auditor's AGENT-LOOP turns — a generation-prompt render
#: whose final model turn is still OPEN after tool traffic (the last
#: message is an assistant tool call awaiting its result, or a tool
#: result awaiting the model's continuation) — so a lens route's response
#: mask must again be LOCATED at the last header occurrence rather than
#: anchored at the prompt tail, which such renders never reach. The mask
#: from there covers header + the assistant's tool call + the
#: tool-response syntax and body + every generated position — the regime
#: the completed gemma-E4B audit ran under, kept for comparability
#: (lens_adapters module docstring). The proxy verifies the declaration
#: against BOTH the message structure and the RENDER
#: (:meth:`RoutingProxy._validate_agent_loop`: lens route only;
#: more than one non-system message; final message a tool result or an
#: assistant tool-call turn; a generation-prompt render; the rendered
#: prompt CONTAINS the header ids and does not END with them — the one
#: invariant all three real shapes share, whose tails all differ) and
#: translates it into the injected selector's ``agent_loop`` fact; the
#: key never travels upstream. Petri's own client (inspect_ai's openai
#: provider) cannot set it, so the Petri-facing ENTRY
#: (:func:`declare_agent_loop`) attaches it for lens-route requests of
#: that structure WHOSE RENDER DOES NOT END AT THE HEADER, and the verifier
#: checks it independently — two layers, both required. The render half of
#: the entry is load-bearing: Llama-3.1 closes a tool turn with
#: ``<|eot_id|>`` and re-appends the header, so its agent-loop turns are
#: ordinary tail renders that the tail arm serves undeclared (as it always
#: has); gemma-4 continues the OPEN model turn after tool traffic, so its
#: turns are declared and relocated. A structure-only entry would have
#: turned every llama tool turn into a 400 (caught by
#: test_petri_proxy's tool round-trip pin, 2026-09-02). THE ENTRY IS ARMED
#: BY DEFAULT (``petri_agent_loop_entry``
#: defaults to True; review 2026-09-02): the relocation is made safe by
#: the verification, not by a switch, and an opt-in switch left the one
#: launcher real audits use — the sbatch — disabled while every test
#: passed. A site that wants it off states ``petri_agent_loop_entry=False``
#: (tests/test_petri_proxy_launcher.py keeps that list). The render probe
#: is the served tokenizer: passed EXPLICITLY (the launcher / CLI
#: ``--tokenizer``, manifest-pinned, eager) or built LAZILY from the
#: routing table's ``base_model`` (+ ``base_revision`` when recorded) on
#: the first agent-loop turn — so the many single-turn drivers never pay
#: a tokenizer load. Mutually exclusive with :data:`RESPONSE_PREFILL_KEY`;
#: an undeclared render of this shape is untouched here and still refuses
#: at the adapter, contained per request (the pinned direction).
AGENT_LOOP_KEY = "petri_agent_loop"
#: :func:`agent_loop_shape`'s two structures.
AGENT_LOOP_SHAPE_TOOL_CALL = "assistant_tool_call"
AGENT_LOOP_SHAPE_TOOL_RESULT = "tool_result"

#: env var the ProxyClientEngine reads when the manifest carries no
#: explicit endpoint (the grading host side of the SSH tunnel).
PROXY_BASE_URL_ENV = "PETRI_PROXY_BASE_URL"

#: in-memory bound on RoutingProxy.request_log (a deque of the most
#: recent entries). The DURABLE per-request audit record is the
#: structured logging stream, emitted for every request regardless;
#: this only caps what the process retains, so a multi-day audit cannot
#: grow the proxy without limit.
REQUEST_LOG_MAX = 100_000

#: in-memory bound on the salt-discipline guard's remembered upstream
#: salts (:meth:`RoutingProxy._guard_cache_salt`): the most recent
#: distinct salts forwarded upstream, evicted oldest-first (counted in
#: ``n_salt_guard_evicted``, never silent). The window is deliberately
#: FAR larger than the KV pool it protects: at the production geometry
#: (Llama-3.1-8B on one H200, ~915k cached tokens = ~57k blocks of 16)
#: the pool holds blocks from at most a few thousand recent requests, so
#: a salt old enough to have been evicted HERE cannot still have live
#: blocks THERE — eviction bounds memory without weakening the property
#: the guard enforces. Cell-stable salts are one entry per cell; an
#: evicted one is simply re-recorded on its next use.
SALT_GUARD_MAX = 100_000

#: the request keys the /tokenize pass-through forwards upstream (vLLM's
#: TokenizeChatRequest fields that shape the render); everything else —
#: sampling knobs, proxy declarations — is dropped, and no selector or salt
#: is ever attached (nothing is generated or cached by a tokenize).
TOKENIZE_FORWARD_KEYS = ("messages", "tools", "add_generation_prompt",
                         "continue_final_message", "add_special_tokens",
                         "chat_template_kwargs")


class ProxyError(RuntimeError):
    """A proxy wiring error (construction-time misconfiguration)."""


class ProxyRefusal(ProxyError):
    """A per-request refusal, carried to the client as an OpenAI-style
    error with ``http_status``. ``extra`` (structured facts beside the prose —
    ``worker_error_type`` on a ledgered refusal) rides in the error body so a
    reader judges the refusal by its fields, never by the message text."""

    def __init__(self, message: str, *, http_status: int = 400,
                 code: str = "petri_proxy_refusal", extra: Optional[Mapping[str, Any]] = None):
        super().__init__(message)
        self.http_status = http_status
        self.code = code
        self.extra: Dict[str, Any] = dict(extra or {})


#: The ``error.code`` of the n>1 refusal below. A driver recognises the
#: refusal by THIS constant off the wire (the class does not survive HTTP),
#: never by matching the prose.
STEERED_FORK_REFUSAL_CODE = "steered_fork_unsupported"
#: The sampling keys that make vLLM produce SEVERAL sequences from one
#: request by forking inside the engine.
FORK_REQUEST_KEYS: Tuple[str, ...] = ("n", "best_of")


class SteeredForkRefusal(ProxyRefusal):
    """A STEERED (selector-carrying, lens) route was asked to fork one
    request into several sequences (:data:`FORK_REQUEST_KEYS` > 1) — refused
    until steered forks are proven (E42 / E46).

    The trap this closes: vLLM forks the extra sequences INSIDE the engine,
    after the request has been admitted, so the lens worker never sees ``n``
    at all. A forked sequence on a steered route may therefore run UNSTEERED
    and come back in the same response as its steered sibling, indistinguishable
    from it. Nothing downstream can detect that — the response carries the
    route's intervention stamps either way — so the refusal has to sit at the
    request boundary, which is the one seam every steered request crosses
    (:meth:`RoutingProxy._rewrite`).

    The remedy is k separate requests, which is what the repo's own engine
    already sends (``proxy_engine.assert_client_body`` refuses ``n`` on any
    served route). This guard is the SERVER-side twin of that rule and binds
    every client, not just ours. Passthrough, prompted and LoRA routes are
    unaffected: their interventions are not applied by a per-request hook that
    a fork can miss.

    Importable so an in-process caller can catch the class; over HTTP the
    :data:`STEERED_FORK_REFUSAL_CODE` in ``error.code`` is the same fact."""

    def __init__(self, message: str):
        super().__init__(message, http_status=400,
                         code=STEERED_FORK_REFUSAL_CODE)


def refuse_steered_forks(request: Mapping[str, Any], route: "RouteEntry") -> None:
    """Raise :class:`SteeredForkRefusal` when ``request`` asks a steered route
    to fork. Absent, ``None`` and the integer ``1`` pass; anything else —
    including a value that cannot be read as 1 — refuses, because a guard that
    cannot prove one sequence must not assume one."""
    for key in FORK_REQUEST_KEYS:
        if key not in request:
            continue
        value = request[key]
        if value is None or (isinstance(value, int) and not isinstance(value, bool)
                             and value == 1):
            continue
        raise SteeredForkRefusal(
            f"{key}={value!r} on the STEERED route {route.virtual_model!r} (cell "
            f"{route.cell_id!r}): vLLM forks the extra sequences inside the engine, where "
            "the lens worker never sees them, so a forked sequence may run UNSTEERED and be "
            "returned beside its steered sibling with no way to tell them apart. Steered "
            f"forks are refused until they are proven — send k separate requests instead "
            f"(each with {key} absent or 1). Passthrough, prompted and LoRA routes are "
            "unaffected.")


# --- routing table -------------------------------------------------------------


@dataclass(frozen=True)
class RouteEntry:
    """Everything the proxy needs about one served leg (a projection of
    ``defense_cells.PetriCell`` — rebuildable from the routing-table JSON
    so the on-cluster proxy never needs the full cell registry)."""

    virtual_model: str
    cell_id: str
    cell_class: str
    serving_identity: str
    intervention_hash: str
    control_of: Optional[str] = None
    si_prefix_hash: Optional[str] = None
    #: native-psr-serving §1.8: the NATIVE position mode the residual slot serves this
    #: route at (``all`` / ``response`` / ``prompt_last`` / ``prefill_all``; None on a
    #: plain LoRA / base / lens route) — the same value the routing table's adapter block
    #: and the engine attestation carry; the controls driver records it as
    #: ``steer_positions`` and the stamp-regime expectation reads it. Default None keeps
    #: every existing table and the ``RouteEntry(**r)`` loader byte-compatible.
    petri_position_mode: Optional[str] = None


@dataclass(frozen=True)
class SaltDecision:
    """ONE request's KV-namespace decision (module docstring): the salt that
    went upstream, the :data:`SALT_REGIMES` regime it was minted under, and
    the ``resumed_prefill_policy`` label the decision was made from — one of
    :data:`RESUMED_PREFILL_POLICIES` when the cell was attested, else one of
    :data:`POLICY_NON_VALUES` naming WHY no policy applied. Carried from
    :meth:`RoutingProxy._rewrite` to the audit row and the salt guard so the
    three read one decision instead of re-deriving it (a second derivation
    of a uuid-bearing salt is a different salt)."""

    salt: str
    regime: str
    policy: str
    lens: bool
    #: the route's NATIVE position mode (native-psr-serving): ``None`` off a
    #: residual route; a mode other than ``all`` is what minted a non-lens leg
    #: request-unique. Recorded on the audit row beside the regime.
    position_mode: Optional[str] = None


@dataclass(frozen=True)
class _SaltClaim:
    """What the salt-discipline guard remembers about a salt already
    forwarded (:meth:`RoutingProxy._guard_cache_salt`): whether a STEERED
    request used it, which cell's namespace it is, and the regime it was
    minted under. The regime is remembered because a repeat is legitimate
    for exactly one of them."""

    lens: bool
    cell_id: str
    regime: str


@dataclass(frozen=True)
class PromptedPrefixSpec:
    """What the proxy verifies a prompted request's system prefix against
    (H8): byte length + sha256_hex16 — the SI TEXT itself deliberately
    never lands on the proxy."""

    virtual_model: str
    byte_length: int
    sha16: str


def routes_from_cells(cell_set: PetriCellSet) -> Tuple[RouteEntry, ...]:
    """Project a PetriCellSet onto proxy routes. enumerate_cells derives
    every identity from the validated spec's ``proxy_vllm`` backend, so
    every projected route is proxy-served; a non-proxy identity reaching
    ``_rewrite`` anyway gets a precise 421 refusal (defensive).

    The per-route schema is UNCHANGED by the fit-support-transfer landing
    on purpose (a client's loader builds these rows from the JSON): every
    per-cell fact of that landing rides an ATTESTATION map instead
    (``lens_serve_declarations``, ``object_artifact_sha256_by_cell``,
    ``resumed_prefill_policy_cells``), and a cell's serve variant is
    already spelled in its ``cell_id`` (``<cell>@all_turns``), which WS-0's
    enumeration made agree with the hashed spec."""
    return tuple(
        RouteEntry(
            virtual_model=c.virtual_model_name, cell_id=c.cell_id,
            cell_class=c.cell_class, serving_identity=c.serving_identity,
            intervention_hash=c.intervention_hash, control_of=c.control_of,
            si_prefix_hash=c.si_prefix_hash)
        for c in cell_set.all_served())


def prompted_prefix_specs(
    cell_set: PetriCellSet, source: CellSource,
) -> Tuple[PromptedPrefixSpec, ...]:
    """Build the per-vmodel prefix specs from the registry's SI bytes at
    EXPORT time (launcher side — the bytes stay off the proxy)."""
    specs: List[PromptedPrefixSpec] = []
    for cell in cell_set.all_served():
        if cell.si_prefix_hash is None:
            continue
        canonical_id = cell.control_of or cell.cell_id
        entry = source.get(canonical_id)
        if entry is None or not entry.si_prefix:
            raise ProxyError(
                f"prompted cell {canonical_id!r} has no si_prefix bytes in "
                "the registry — cannot derive the verification spec (H8)")
        sha = cm.sha256_hex16(bytes(entry.si_prefix))
        if sha != cell.si_prefix_hash:
            raise ProxyError(
                f"prompted cell {canonical_id!r}: registry prefix bytes "
                f"hash to {sha}, cell records {cell.si_prefix_hash} — "
                "inconsistent enumeration (H8)")
        specs.append(PromptedPrefixSpec(
            virtual_model=cell.virtual_model_name,
            byte_length=len(bytes(entry.si_prefix)), sha16=sha))
    return tuple(specs)


def lens_served_positions(cell_set: PetriCellSet, source: CellSource
                          ) -> Dict[str, List[str]]:
    """THE SERVED-POSITIONS STAMP (:data:`LENS_SERVED_POSITIONS_KEY`): for
    every LENS-served leg of ``cell_set`` whose registry entry's resolved
    spec STATES a served position mode, ``cell_id -> [mode]`` — the mode read
    through ``defense_cells.served_positions_of`` (a stacked cell's is its
    OBJECT leg's; the same unwrapper ``registry_shim`` decides its markers
    from), the spec the same ``source.get(cell_id)`` the enumeration hashed
    into the leg's ``intervention_hash`` and the worker registry copies
    VERBATIM for the worker to bind its ops to
    (``lens_steer._assert_ops_express_spec_positions``, ``lens_adapters.
    _check_fit_serve_positions``). SERVED, never the fit: the spec's
    ``positions`` is the cell's served mode and overrides the artifact's
    (``lens_adapters._served_positions``); the recorded FIT regime lives in
    the artifact bytes and is not read here.

    A lens leg is ABSENT from the map when its spec states no mode — the
    ``none__lens0`` twin (a DERIVED control leg with no registry entry of its
    own — the real source refuses to look one up — that arms its donor's ops
    at scale 0, a choice ``lens_steer.noop_twin_donor`` makes over registry
    entries this launcher-side stamp does not build) and any canonical cell
    whose served mode defaults to its artifact's — and the proxy reads
    absence as header-located (today's refusal). Only CANONICAL legs are
    looked up, exactly as the standup's own marker derivation does. A STATED
    mode outside :data:`POSITION_MODES` is a wiring defect, refused here (the
    worker's adapter would refuse it too, but the table must never record a
    mode the gate cannot classify). Non-lens legs are never listed: the gate
    they feed is lens-only."""
    out: Dict[str, List[str]] = {}
    for cell in cell_set.canonical:
        if cm.parse_serving_identity(cell.serving_identity).path != "lens":
            continue
        entry = source.get(cell.cell_id)
        spec = getattr(entry, "resolved_spec", None) if entry is not None else None
        mode = served_positions_of(spec)
        if mode is None:
            continue
        if not isinstance(mode, str) or mode not in POSITION_MODES:
            raise ProxyError(
                f"lens cell {cell.cell_id!r}: its resolved spec states served "
                f"positions {mode!r}, outside the served vocabulary {POSITION_MODES} "
                f"— the routing table records only a mode the thinking-ON gate can "
                "classify as header-located or not; refusing the export")
        out[cell.cell_id] = [mode]
    return out


def served_positions_stamp(cell_set: PetriCellSet, source: CellSource,
                           routes: Sequence["RouteEntry"]) -> Dict[str, List[str]]:
    """THE ENGINE-NEUTRAL SERVED-POSITIONS STAMP (:data:`SERVED_POSITIONS_KEY`;
    native-positions-resolve): the lens map's entries
    (:func:`lens_served_positions` — the lens legs whose resolved spec states
    a mode) plus, for every ROUTE that attests a position mode
    (``RouteEntry.petri_position_mode`` — the value :func:`_routes_with_
    position_modes` stamped from the adapter's own attested mode, held to the
    lora path and the native vocabulary there), ``cell_id -> [mode]``: one
    shape for both engines, the list-of-modes value the lens map has always
    carried. DERIVED per route from the ATTESTED mode, never from the engine's
    name: a route attesting none is ABSENT, and the readers refuse it by name
    rather than default it. A cell named by both maps is not one standup's
    shape (a lens leg is not a lora route) and refuses."""
    out: Dict[str, List[str]] = dict(lens_served_positions(cell_set, source))
    for r in routes:
        mode = getattr(r, "petri_position_mode", None)
        if mode is None:
            continue
        if r.cell_id in out:
            raise ProxyError(
                f"{SERVED_POSITIONS_KEY}: cell {r.cell_id!r} is named by the lens map ({out[r.cell_id]}) AND attests "
                f"a route position mode ({mode!r}) — a lens leg is not a lora route; refusing the export")
        out[r.cell_id] = [str(mode)]
    return out


# --- version-2 tables: the per-lora-route adapter block ---------------------------


def _route_path(route: RouteEntry) -> str:
    """The route's serving PATH (``lens`` / ``lora`` / ``passthrough``), read
    from the serving identity — the same source ``_rewrite`` keys on."""
    return cm.parse_serving_identity(route.serving_identity).path


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX64.match(value))


_HEX16 = re.compile(r"^[0-9a-f]{16}$")


def _is_hex16(value: Any) -> bool:
    """A ``petri_intervention_hash`` (``sha256_hex16``): 16 lowercase hex."""
    return isinstance(value, str) and bool(_HEX16.match(value))


def validated_adapter_block(block: Any, *, cell_id: str,
                            intervention_hash: str) -> Dict[str, Any]:
    """ONE adapter block (:data:`ADAPTER_BLOCK_KEYS`), validated and bound to
    the lora route it rides beside: EXACT key set; ``name == cell_id``;
    ``dir`` an absolute path; both digests 64-hex; ``r`` a positive int;
    ``petri_intervention_hash == intervention_hash`` (the route's — for
    ``none__lora0`` that is ``none``'s hash by ``enumerate_cells``'
    construction); ``petri_zero_adapter`` a bool that is True exactly when the
    cell is a zero-adapter control leg (:data:`LORA_ZERO_CELL_SUFFIX`);
    ``petri_source_kind`` a non-empty string or None. Anything else is a
    ``ProxyError`` naming the cell and the field — the block is what the
    proxy later holds the attestation to, so junk here is a wiring defect
    refused before any traffic, never a shape guessed at."""
    if not isinstance(block, Mapping):
        raise ProxyError(f"{ADAPTERS_KEY}[{cell_id!r}] must be a mapping (one adapter "
                         f"block, keys {list(ADAPTER_BLOCK_KEYS)}), got "
                         f"{type(block).__name__}")
    required = set(ADAPTER_BLOCK_KEYS)
    if set(block) != required:
        raise ProxyError(
            f"{ADAPTERS_KEY}[{cell_id!r}] must carry exactly the keys "
            f"{list(ADAPTER_BLOCK_KEYS)}: missing {sorted(required - set(block))}, "
            f"unknown {sorted(set(block) - required)}")
    name = block["name"]
    if name != cell_id:
        raise ProxyError(f"{ADAPTERS_KEY}[{cell_id!r}].name is {name!r} — the block's name "
                         "IS the route's cell id (the adapter the server registers under "
                         "that name is what `model` selects); refusing the mismatch")
    d = block["dir"]
    if not isinstance(d, str) or not d or not os.path.isabs(d):
        raise ProxyError(f"{ADAPTERS_KEY}[{cell_id!r}].dir must be an absolute path, got {d!r}")
    for key in ("config_sha256", "weights_sha256"):
        if not _is_hex64(block[key]):
            raise ProxyError(f"{ADAPTERS_KEY}[{cell_id!r}].{key} must be a 64-hex lowercase "
                             f"sha256, got {block[key]!r}")
    r = block["r"]
    if not isinstance(r, int) or isinstance(r, bool) or r <= 0:
        raise ProxyError(f"{ADAPTERS_KEY}[{cell_id!r}].r must be a positive int, got {r!r}")
    ih = block["petri_intervention_hash"]
    if ih != intervention_hash:
        raise ProxyError(
            f"{ADAPTERS_KEY}[{cell_id!r}].petri_intervention_hash {ih!r} != the route's "
            f"intervention_hash {intervention_hash!r} — the adapter was exported from a "
            "different object than the route serves; refusing")
    zero = block["petri_zero_adapter"]
    if not isinstance(zero, bool):
        raise ProxyError(f"{ADAPTERS_KEY}[{cell_id!r}].petri_zero_adapter must be a bool, "
                         f"got {zero!r}")
    is_zero_cell = cell_id.endswith(LORA_ZERO_CELL_SUFFIX)
    if zero != is_zero_cell:
        raise ProxyError(
            f"{ADAPTERS_KEY}[{cell_id!r}].petri_zero_adapter is {zero} but the cell "
            f"{'is' if is_zero_cell else 'is not'} a zero-adapter control leg "
            f"(suffix {LORA_ZERO_CELL_SUFFIX!r}) — a zero adapter under a genuine cell's "
            "name is base served under a steered identity, and a genuine adapter under "
            "the control's name steers the no-op leg; refusing")
    kind = block["petri_source_kind"]
    if kind is not None and (not isinstance(kind, str) or not kind):
        raise ProxyError(f"{ADAPTERS_KEY}[{cell_id!r}].petri_source_kind must be a non-empty "
                         f"string or None, got {kind!r}")
    return {
        "name": str(name), "dir": str(d),
        "config_sha256": str(block["config_sha256"]),
        "weights_sha256": str(block["weights_sha256"]),
        "r": int(r), "petri_intervention_hash": str(ih),
        "petri_zero_adapter": bool(zero),
        "petri_source_kind": (str(kind) if kind is not None else None),
    }


def _validated_adapter_blocks(blocks: Any, routes: Sequence[RouteEntry]
                              ) -> Dict[str, Dict[str, Any]]:
    """The :data:`ADAPTERS_KEY` map held to ``routes``: a block for EVERY lora
    route, a block for NO other cell (a passthrough / prompted / lens cell
    with a block is a contradiction — nothing selects an adapter there), no
    block for a cell the table does not route, each block validated against
    its route (:func:`validated_adapter_block`). Refusals name the cells."""
    if not isinstance(blocks, Mapping):
        raise ProxyError(f"{ADAPTERS_KEY} must be a cell_id -> adapter block mapping, got "
                         f"{type(blocks).__name__}")
    by_cell = {r.cell_id: r for r in routes}
    lora_cells = {cid for cid, r in by_cell.items() if _route_path(r) == "lora"}
    unknown = sorted(str(c) for c in blocks if c not in by_cell)
    if unknown:
        raise ProxyError(f"{ADAPTERS_KEY} names cell(s) {unknown} that no route serves — "
                         "an adapter nobody selects is a stale or foreign block; refusing")
    non_lora = sorted(str(c) for c in blocks if c in by_cell and c not in lora_cells)
    if non_lora:
        raise ProxyError(
            f"{ADAPTERS_KEY} carries block(s) for non-lora route(s) {non_lora} (paths "
            f"{[_route_path(by_cell[c]) for c in non_lora]}) — only a lora route selects "
            "an adapter; refusing")
    uncovered = sorted(lora_cells - set(blocks))
    if uncovered:
        raise ProxyError(
            f"lora route(s) {uncovered} have no {ADAPTERS_KEY} block — a natively served "
            "adapter without its digests cannot be held to the attestation; refusing")
    return {cid: validated_adapter_block(blocks[cid], cell_id=cid,
                                         intervention_hash=by_cell[cid].intervention_hash)
            for cid in sorted(blocks)}


def verify_adapter_block_on_disk(block: Mapping[str, Any]) -> None:
    """The standup's PRE-LAUNCH binding and the
    certification's: re-hash ``<dir>/adapter_config.json`` and
    ``<dir>/adapter_model.safetensors`` and hold them to the block's
    ``config_sha256`` / ``weights_sha256``; hold the config's ``r``,
    ``petri_zero_adapter``, ``petri_source_kind`` and — for a genuine block —
    ``petri_intervention_hash`` to the block's, so the bytes at ``dir`` are
    bound to the OBJECT the cell serves, not merely to some adapter's digests
    (a block pointing cell A at adapter B's dir with B's digests is
    otherwise internally consistent). A missing file or any disagreement
    is a ``ProxyError`` naming the field — never a tolerance. NOT called by
    :func:`load_routing_table` or the proxy process (shape and consistency
    only there; the proxy stays CPU-light and may run where the bank is not
    mounted)."""
    name = block.get("name") if isinstance(block, Mapping) else None
    if not isinstance(block, Mapping) or set(block) != set(ADAPTER_BLOCK_KEYS):
        raise ProxyError(f"adapter block {name!r} is not a validated block "
                         f"(keys {list(ADAPTER_BLOCK_KEYS)}); validate it first")
    d = Path(block["dir"])
    cfg_path, w_path = d / ADAPTER_CONFIG_FILE, d / ADAPTER_WEIGHTS_FILE
    for p in (cfg_path, w_path):
        if not p.is_file():
            raise ProxyError(f"adapter block {name!r}: {p} is missing — the block names a "
                             "directory that does not hold the exported adapter")
    got_cfg, got_w = _sa.sha256_file(cfg_path), _sa.sha256_file(w_path)
    # equal bytes, or the release's declared rewrite of the same file (a certified scrub-map row);
    # the weights are never rewritten, so they must match exactly
    from sisteer import scrub_map as _sm
    if _sm.check(block["config_sha256"], got_cfg, path=cfg_path) is None:
        raise ProxyError(f"adapter block {name!r}: {cfg_path} hashes to {got_cfg}, the block "
                         f"records config_sha256 {block['config_sha256']} — the adapter on "
                         "disk is not the one the table binds; refusing")
    if got_w != block["weights_sha256"]:
        raise ProxyError(f"adapter block {name!r}: {w_path} hashes to {got_w}, the block "
                         f"records weights_sha256 {block['weights_sha256']} — the adapter "
                         "on disk is not the one the table binds; refusing")
    try:
        cfg = json.loads(cfg_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise ProxyError(f"adapter block {name!r}: {cfg_path} is unreadable ({e})") from e
    if not isinstance(cfg, Mapping) or cfg.get("r") != block["r"]:
        raise ProxyError(f"adapter block {name!r}: {ADAPTER_CONFIG_FILE} r="
                         f"{cfg.get('r') if isinstance(cfg, Mapping) else cfg!r} != the "
                         f"block's r={block['r']}; refusing")
    if bool(cfg.get("petri_zero_adapter") is True) != bool(block["petri_zero_adapter"]):
        raise ProxyError(
            f"adapter block {name!r}: {ADAPTER_CONFIG_FILE} petri_zero_adapter="
            f"{cfg.get('petri_zero_adapter')!r} disagrees with the block's "
            f"{block['petri_zero_adapter']} — the file and the table must agree on "
            "whether this is the no-op control; refusing")
    # The config's OWN provenance is held to the block's too: a
    # block that points cell A at adapter B's directory WITH B's digests
    # re-hashes clean (B's bytes ARE B's digests) and agrees on r and the zero
    # flag, so without this a table edited or regenerated with the wrong dir
    # would serve B's steering under A's name with every check green. The
    # genuine exporter writes petri_intervention_hash = the object's hash into
    # the config; the zero adapter's config carries none (its block carries
    # `none`'s route hash by enumerate_cells' construction), so that field is
    # compared for GENUINE blocks only; petri_source_kind for every block.
    if not block["petri_zero_adapter"] and \
            cfg.get("petri_intervention_hash") != block["petri_intervention_hash"]:
        raise ProxyError(
            f"adapter block {name!r}: {ADAPTER_CONFIG_FILE} petri_intervention_hash="
            f"{cfg.get('petri_intervention_hash')!r} != the block's "
            f"{block['petri_intervention_hash']!r} — the adapter at {d} was exported from "
            "another object than the one this cell serves (a block pointing at another "
            "adapter's directory); refusing")
    if cfg.get("petri_source_kind") != block["petri_source_kind"]:
        raise ProxyError(
            f"adapter block {name!r}: {ADAPTER_CONFIG_FILE} petri_source_kind="
            f"{cfg.get('petri_source_kind')!r} != the block's "
            f"{block['petri_source_kind']!r}; refusing")


def validated_engine_attestation(att: Any) -> Dict[str, Any]:
    """The :data:`ENGINE_ATTESTATION_KEYS` shape a lora-native standup hands
    over (``serving_attestation.engine_identity``): ``kind ==
    "lora_native"``; ``enforce_eager`` the bool False (the mode exists to run
    graphs — an eager attestation is the lens path's, refused here);
    ``cudagraph_mode`` a non-empty string (RECORDED, not asserted — vLLM
    resolves it by platform); ``lora_config`` a mapping (recorded); ``adapters``
    a non-empty ``{name: {config_sha256, weights_sha256, fingerprint_bf16,
    lora_int_id, petri_intervention_hash, petri_zero_adapter}}`` (``residual_slot``
    ``{"armed": bool}`` — the UNARMED schema-1 shape — or ``{"armed": bool,
    "n_workers": int >= 1}``; an ARMED slot MUST carry the count) with 64-hex
    digests, distinct positive int ids, a 16-hex hash or ``None``, and a bool
    zero flag under which the hash MUST be ``None`` (the zero adapter's config
    records no intervention; an adapter exported from an object is not the
    no-op control, whatever the flag says). Exact key sets at both levels: an
    attestation that says more or less than this is a reader drift, refused
    rather than partially trusted."""
    if not isinstance(att, Mapping):
        raise ProxyError(f"engine_attestation must be a mapping, got {type(att).__name__}")
    required = set(ENGINE_ATTESTATION_KEYS)
    if set(att) != required:
        raise ProxyError(
            f"engine_attestation must carry exactly the keys {list(ENGINE_ATTESTATION_KEYS)}: "
            f"missing {sorted(required - set(att))}, unknown {sorted(set(att) - required)}")
    if att["kind"] != ENGINE_KIND_LORA_NATIVE:
        raise ProxyError(f"engine_attestation.kind {att['kind']!r} != "
                         f"{ENGINE_KIND_LORA_NATIVE!r} — the only attested engine this proxy "
                         "binds to; refusing")
    if att["enforce_eager"] is not False:
        raise ProxyError(
            f"engine_attestation.enforce_eager is {att['enforce_eager']!r}, not False — a "
            "lora-native server runs with CUDA graphs on; an eager (or unstated) engine "
            "is not the attested mode; refusing")
    mode = att["cudagraph_mode"]
    if not isinstance(mode, str) or not mode:
        raise ProxyError(f"engine_attestation.cudagraph_mode must be a non-empty string, "
                         f"got {mode!r}")
    if not isinstance(att["lora_config"], Mapping):
        raise ProxyError(f"engine_attestation.lora_config must be a mapping, got "
                         f"{type(att['lora_config']).__name__}")
    rs = att["residual_slot"]
    if not isinstance(rs, Mapping) or set(rs) not in ({"armed"}, {"armed", "n_workers"}) \
            or not isinstance(rs["armed"], bool):
        raise ProxyError(
            f"engine_attestation.residual_slot must be {{'armed': bool}} or {{'armed': bool, 'n_workers': int}}, got {rs!r}")
    if "n_workers" in rs and (isinstance(rs["n_workers"], bool) or not isinstance(rs["n_workers"], int) or rs["n_workers"] < 1):
        raise ProxyError(f"engine_attestation.residual_slot.n_workers must be a positive int, got {rs['n_workers']!r}")
    if rs["armed"] and "n_workers" not in rs:
        # an ARMED residual slot reads failure claims; without the attested worker count a one-entry
        # {"results": [null]} on a multi-worker engine would read as clean — the count-free shape is the
        # UNARMED schema-1 identity's only
        raise ProxyError("engine_attestation.residual_slot is armed but carries no n_workers — an armed "
                         "residual slot must attest its worker count (the schema-2 readiness attestation "
                         "records it); refusing to start")
    adapters = att["adapters"]
    if not isinstance(adapters, Mapping) or not adapters:
        raise ProxyError("engine_attestation.adapters must be a non-empty name -> entry "
                         f"mapping, got {adapters!r}")
    clean: Dict[str, Dict[str, Any]] = {}
    ids: Dict[int, str] = {}
    akeys = set(ENGINE_ADAPTER_KEYS)
    for name, entry in adapters.items():
        if not isinstance(name, str) or not name:
            raise ProxyError(f"engine_attestation.adapters: bad adapter name {name!r}")
        if not isinstance(entry, Mapping) or set(entry) != akeys:
            raise ProxyError(
                f"engine_attestation.adapters[{name!r}] must carry exactly the keys "
                f"{list(ENGINE_ADAPTER_KEYS)}, got {sorted(entry) if isinstance(entry, Mapping) else entry!r}")
        for key in ("config_sha256", "weights_sha256", "fingerprint_bf16"):
            if not _is_hex64(entry[key]):
                raise ProxyError(f"engine_attestation.adapters[{name!r}].{key} must be a "
                                 f"64-hex lowercase sha256, got {entry[key]!r}")
        lid = entry["lora_int_id"]
        if not isinstance(lid, int) or isinstance(lid, bool) or lid <= 0:
            raise ProxyError(f"engine_attestation.adapters[{name!r}].lora_int_id must be a "
                             f"positive int, got {lid!r}")
        if lid in ids:
            raise ProxyError(f"engine_attestation.adapters: lora_int_id {lid} is claimed by "
                             f"both {ids[lid]!r} and {name!r} — one id, one adapter; refusing")
        ids[lid] = name
        ih = entry["petri_intervention_hash"]
        if ih is not None and not _is_hex16(ih):
            raise ProxyError(f"engine_attestation.adapters[{name!r}].petri_intervention_hash "
                             f"must be a 16-hex lowercase hash or None (the zero adapter), got "
                             f"{ih!r}")
        zero = entry["petri_zero_adapter"]
        if not isinstance(zero, bool):
            raise ProxyError(f"engine_attestation.adapters[{name!r}].petri_zero_adapter must "
                             f"be a bool, got {zero!r}")
        if zero and ih is not None:
            raise ProxyError(
                f"engine_attestation.adapters[{name!r}] is flagged petri_zero_adapter=True but "
                f"carries petri_intervention_hash {ih!r} — the zero adapter's config records no "
                "intervention; an adapter exported from an object is not the no-op control; "
                "refusing")
        kind = entry["petri_source_kind"]
        if kind is not None and (not isinstance(kind, str) or not kind):
            raise ProxyError(f"engine_attestation.adapters[{name!r}].petri_source_kind must be a "
                             f"non-empty string or None, got {kind!r}")
        rfp = entry["residual_fingerprint"]
        if rfp is not None and not _is_hex64(rfp):
            raise ProxyError(f"engine_attestation.adapters[{name!r}].residual_fingerprint must be a "
                             f"64-hex lowercase sha256 or None, got {rfp!r}")
        pmode = entry["position_mode"]
        if pmode is not None and (not isinstance(pmode, str) or not pmode):
            raise ProxyError(f"engine_attestation.adapters[{name!r}].position_mode must be a "
                             f"non-empty string or None, got {pmode!r}")
        if (rfp is None) != (pmode is None):
            raise ProxyError(
                f"engine_attestation.adapters[{name!r}]: residual_fingerprint {rfp!r} and "
                f"position_mode {pmode!r} must both be set (a residual entry) or both None")
        if rfp is not None and zero:
            raise ProxyError(
                f"engine_attestation.adapters[{name!r}] is flagged petri_zero_adapter=True but "
                "carries a residual payload — the zero adapter writes nothing; refusing")
        clean[name] = {"config_sha256": str(entry["config_sha256"]),
                       "weights_sha256": str(entry["weights_sha256"]),
                       "fingerprint_bf16": str(entry["fingerprint_bf16"]),
                       "lora_int_id": int(lid),
                       "petri_intervention_hash": (None if ih is None else str(ih)),
                       "petri_zero_adapter": bool(zero),
                       "petri_source_kind": (None if kind is None else str(kind)),
                       "residual_fingerprint": (None if rfp is None else str(rfp)),
                       "position_mode": (None if pmode is None else str(pmode))}
    armed_residual = any(e["residual_fingerprint"] is not None for e in clean.values())
    if bool(rs["armed"]) != armed_residual:
        raise ProxyError(
            f"engine_attestation.residual_slot.armed={rs['armed']!r} but "
            f"{'a' if armed_residual else 'no'} attested adapter carries a residual payload — "
            "the slot is armed iff one does; refusing")
    return {"kind": ENGINE_KIND_LORA_NATIVE, "enforce_eager": False,
            "cudagraph_mode": str(mode), "lora_config": dict(att["lora_config"]),
            # the attested worker count travels with the identity: the proxy binds every
            # failure claim's per-worker result list to it (parse_failure_claim)
            "residual_slot": ({"armed": bool(rs["armed"]), "n_workers": int(rs["n_workers"])}
                              if "n_workers" in rs else {"armed": bool(rs["armed"])}),
            "adapters": clean}


def _validated_lens_served_positions(raw: Any) -> Optional[Dict[str, List[str]]]:
    """:data:`LENS_SERVED_POSITIONS_KEY`'s shape, validated: ``None`` (no
    fact — an older table, a launcher that recorded none) passes through;
    else a mapping of non-empty cell ids to NON-EMPTY lists of
    :data:`POSITION_MODES` members (a list, not a scalar: header-locatedness
    is an OP property and the gate asks whether ANY served mode is
    header-located, so a mixed set is representable and reads as
    header-located). Anything else is a wiring defect (``ProxyError``): the
    gate must decide from a fact it can classify, never from junk that
    happens to read as "not header-located"."""
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ProxyError(f"{LENS_SERVED_POSITIONS_KEY} must be a cell -> [served mode] "
                         f"mapping or None, got {type(raw).__name__}")
    clean: Dict[str, List[str]] = {}
    for cell, modes in raw.items():
        if not isinstance(cell, str) or not cell:
            raise ProxyError(f"{LENS_SERVED_POSITIONS_KEY}: bad cell id {cell!r}")
        if isinstance(modes, str) or not isinstance(modes, (list, tuple)) or not modes:
            raise ProxyError(
                f"{LENS_SERVED_POSITIONS_KEY}[{cell!r}] must be a non-empty list of served "
                f"modes from {POSITION_MODES}, got {modes!r}")
        bad = [m for m in modes if not isinstance(m, str) or m not in POSITION_MODES]
        if bad:
            raise ProxyError(
                f"{LENS_SERVED_POSITIONS_KEY}[{cell!r}] carries mode(s) {bad!r} outside "
                f"{POSITION_MODES} — refusing to start rather than classify a mode the "
                "gate does not know")
        clean[cell] = [str(m) for m in modes]
    return clean


def _validated_served_positions(raw: Any, *, routes: Sequence["RouteEntry"],
                                lens_map: Optional[Mapping[str, Sequence[str]]]) -> Optional[Dict[str, List[str]]]:
    """:data:`SERVED_POSITIONS_KEY`'s shape, validated: ``None`` (an older
    table — the readers fall back) passes through; else a mapping of ROUTED
    cell ids to NON-EMPTY lists of modes from the lens vocabulary
    (:data:`POSITION_MODES`) or the native one
    (``defense_cells.NATIVE_POSITION_MODES``), and — where the lens map names
    the same cell — EQUAL to the lens map's entry (one standup's two records
    agree). Anything else is a wiring defect (``ProxyError``): a reader must
    resolve a served mode from a fact the record binds to a route, never from
    junk that happens to read as a mode."""
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ProxyError(f"{SERVED_POSITIONS_KEY} must be a cell -> [served mode] mapping or None, "
                         f"got {type(raw).__name__}")
    routed = {r.cell_id for r in routes}
    vocabulary = tuple(POSITION_MODES) + tuple(m for m in NATIVE_POSITION_MODES if m not in POSITION_MODES)
    clean: Dict[str, List[str]] = {}
    for cell, modes in raw.items():
        if not isinstance(cell, str) or not cell:
            raise ProxyError(f"{SERVED_POSITIONS_KEY}: bad cell id {cell!r}")
        if cell not in routed:
            raise ProxyError(f"{SERVED_POSITIONS_KEY} names cell {cell!r}, which no route serves")
        if isinstance(modes, str) or not isinstance(modes, (list, tuple)) or not modes:
            raise ProxyError(
                f"{SERVED_POSITIONS_KEY}[{cell!r}] must be a non-empty list of served modes from "
                f"{vocabulary}, got {modes!r}")
        bad = [m for m in modes if not isinstance(m, str) or m not in vocabulary]
        if bad:
            raise ProxyError(
                f"{SERVED_POSITIONS_KEY}[{cell!r}] carries mode(s) {bad!r} outside {vocabulary} — refusing "
                "to start rather than serve a mode no reader can classify")
        clean[cell] = [str(m) for m in modes]
        if lens_map is not None and cell in lens_map and list(lens_map[cell]) != clean[cell]:
            raise ProxyError(
                f"{SERVED_POSITIONS_KEY}[{cell!r}] = {clean[cell]} != {LENS_SERVED_POSITIONS_KEY}[{cell!r}] = "
                f"{list(lens_map[cell])} — one standup's two records disagree; refusing")
    return clean


def route_row(route: "RouteEntry") -> Dict[str, Any]:
    """ONE route's table row. The seven pre-change fields always; the stage-2
    ``petri_position_mode`` (native-psr-serving §1.8) ONLY when set — so a
    table with no restricted-mode residual route is byte-identical to the
    pre-stage-2 exporter's (``tests/fixtures/routing_table_v1_golden.json``
    pins it), and every pre-stage-2 loader (``RouteEntry(**row)`` with the
    seven fields) keeps reading every version-1 table. The loader's default
    reads an absent key as None."""
    row = asdict(route)
    if row.get("petri_position_mode") is None:
        row.pop("petri_position_mode", None)
    return row


def _routes_with_position_modes(routes: Sequence["RouteEntry"],
                                position_modes: Mapping[str, Optional[str]], *,
                                native: bool) -> Tuple["RouteEntry", ...]:
    """Stamp each route named with a NON-None mode (native-psr-serving §1.8:
    the value the exporter wrote into the adapter's ``petri_residual_slot``
    block and the engine attestation carries — the same fact, on the route
    the drivers read). Refuses, before a byte is written: a mode on a
    VERSION-1 table (no ``adapter_blocks`` — only the lora-native standup
    serves a residual slot, and it writes version 2; a v1 table never carries
    the field, so the golden and every old loader stay exact), a mode for a
    cell no route serves, a mode on a non-lora route, and a mode outside
    ``defense_cells.NATIVE_POSITION_MODES`` (the residual slot's vocabulary,
    pinned equal to ``native_residual.POSITION_MODES`` — no torch here)."""
    named = {cid: m for cid, m in position_modes.items() if m is not None}
    if not named:
        return tuple(routes)
    if not native:
        raise ProxyError(
            f"position_modes {sorted(named)} on a version-1 table (no adapter_blocks) — a "
            "native position mode rides a lora-native (version-2) table only")
    by_cell = {r.cell_id: r for r in routes}
    unknown = sorted(set(named) - set(by_cell))
    if unknown:
        raise ProxyError(f"position_modes names cell(s) {unknown} that no route serves")
    for cid, mode in sorted(named.items()):
        if mode not in NATIVE_POSITION_MODES:
            raise ProxyError(
                f"position_modes[{cid!r}] = {mode!r} is not one of {NATIVE_POSITION_MODES}")
        if cm.parse_serving_identity(by_cell[cid].serving_identity).path != "lora":
            raise ProxyError(
                f"position_modes names {cid!r}, whose route is not lora-served "
                f"({by_cell[cid].serving_identity}) — a residual slot rides a lora route")
    return tuple(replace(r, petri_position_mode=named[r.cell_id]) if r.cell_id in named else r
                 for r in routes)


def export_routing_table(cell_set: PetriCellSet, source: CellSource,
                         path: Path, *, base_model: str,
                         serving_mode: str,
                         exact_replay_strategy: Optional[str] = None,
                         base_revision: Optional[str] = None,
                         thinking_header: Optional[Mapping[str, Any]] = None,
                         thinking_header_error: Optional[str] = None,
                         organisms: Sequence[str] = (),
                         adapter_blocks: Optional[Mapping[str, Mapping[str, Any]]] = None,
                         position_modes: Optional[Mapping[str, Optional[str]]] = None,
                         ) -> None:
    """Serialize routes + prefix specs for the on-cluster proxy process
    (which has no object registry — object state lives in the
    vLLM workers; the proxy holds names, hashes, and lengths only).

    ``adapter_blocks`` (optional) is the LORA-NATIVE standup's
    ``{cell_id: adapter block}`` map. ``None``
    writes exactly today's VERSION-1 document — lora routes included, no new
    key anywhere (``tests/fixtures/routing_table_v1_golden.json`` pins the
    bytes). Given, the table is written as VERSION 2 with the validated map
    under :data:`ADAPTERS_KEY`, and a lora route with no block, a block for a
    non-lora cell, or a block for an unrouted cell REFUSES
    (:func:`_validated_adapter_blocks`). The per-route schema is unchanged
    either way.

    ``exact_replay_strategy`` is REQUIRED for ``exact_replay`` (one of
    :data:`~.vllm_server.EXACT_REPLAY_STRATEGIES`) and REFUSED otherwise
    — the mechanism is recorded IN the table, so the proxy can
    verify at startup that an enforceable mechanism is actually in play
    rather than trusting an unattested flag.

    ``base_revision`` (optional) records the served model's pinned
    revision so a proxy started without an explicit agent-loop render
    probe can build one from the SAME pin lazily
    (:meth:`RoutingProxy._agent_loop_probe`); absent, the lazy probe loads
    the unpinned tokenizer — a consistency probe either way, never the
    served ids.

    ``thinking_header`` (optional) is the
    served template's thinking-header fact (``chat_markers.
    thinking_header_facts(tok)``), recorded top-level as
    :data:`THINKING_HEADER_KEY`; a table without it makes every
    ``enable_thinking: true`` on a LENS route unverifiable at the proxy, and
    therefore refused (fail closed). The per-ROUTE schema is unchanged: the
    lens cells' declarations ride the attested ``lens_serve_declarations``
    map, never the table.

    ``thinking_header_error`` (optional) is
    the reason the fact's ON certification is UNAVAILABLE, recorded top-level
    as :data:`THINKING_HEADER_ERROR_KEY` BESIDE the fact (present and null
    beside a complete fact; absent when no fact is recorded) — so a proxy
    built from the table alone (:func:`main`) never reconstructs a partial
    fact without its reason. Both directions of the pairing are enforced
    here as at the constructor (:func:`_validated_thinking_header_error`).

    The per-cell SERVED POSITIONS map (:data:`LENS_SERVED_POSITIONS_KEY`;
    §2.5.4 narrowing) is stamped UNCONDITIONALLY, from ``source``'s resolved
    specs (:func:`lens_served_positions`) — every standup that exports a
    table records it, whether or not it records a thinking-header fact. A
    table without the key (exported before this landing) loads as ``None``
    and the gate keeps refusing every undeclared ON render on a lens route,
    exactly as before (fail closed)."""
    if serving_mode == "exact_replay":
        if exact_replay_strategy not in EXACT_REPLAY_STRATEGIES:
            raise ProxyError(
                f"exact_replay routing table needs an explicit "
                f"exact_replay_strategy from {EXACT_REPLAY_STRATEGIES}, got "
                f"{exact_replay_strategy!r} — the mechanism is a stated, "
                "recorded choice (D0 constraint 3)")
    elif exact_replay_strategy is not None:
        raise ProxyError(
            f"exact_replay_strategy={exact_replay_strategy!r} on a "
            f"{serving_mode!r} table — a replay mechanism on an interleaved "
            "run would mislabel every row's mode")
    routes = routes_from_cells(cell_set)
    if position_modes:
        routes = _routes_with_position_modes(routes, position_modes,
                                             native=adapter_blocks is not None)
    doc = {
        "version": 1,
        "base_model": base_model,
        "serving_mode": serving_mode,
        "exact_replay_strategy": exact_replay_strategy,
        "base_revision": base_revision,
        "routes": [route_row(r) for r in routes],
        "prompted_prefixes": [asdict(s) for s in
                              prompted_prefix_specs(cell_set, source)],
        # the served-positions fact the thinking-ON gate reads header-
        # locatedness from — PRESENT on every table this exporter writes (an
        # empty map is the stated "no lens leg states a mode")
        LENS_SERVED_POSITIONS_KEY: lens_served_positions(cell_set, source),
    }
    # the ENGINE-NEUTRAL map beside it (native-positions-resolve): the lens
    # entries plus every lora route's attested mode; written iff it names a
    # cell, so a table with no attested mode is byte-identical to before
    served = served_positions_stamp(cell_set, source, routes)
    if served:
        doc[SERVED_POSITIONS_KEY] = served
    if thinking_header is not None:
        doc[THINKING_HEADER_KEY] = _validated_thinking_header(thinking_header)
    error = _validated_thinking_header_error(doc.get(THINKING_HEADER_KEY),
                                             thinking_header_error)
    if thinking_header is not None or error is not None:
        doc[THINKING_HEADER_ERROR_KEY] = error
    if organisms:
        # the adapters the standup loaded as vLLM LoRA modules (ORGANISM_SEP
        # overlays); absent on every table that loaded none, byte-for-byte
        orgs = sorted(set(str(o) for o in organisms))
        for o in orgs:
            if not o or ORGANISM_SEP in o:
                raise ProxyError(f"organism name {o!r} is empty or carries {ORGANISM_SEP!r}")
        doc[ORGANISMS_KEY] = orgs
    if adapter_blocks is not None:
        # the LORA-NATIVE table: version 2 + the adapter map,
        # held to the routes before a byte is written
        doc["version"] = 2
        doc[ADAPTERS_KEY] = _validated_adapter_blocks(adapter_blocks, routes)
    Path(path).write_text(json.dumps(doc, indent=2, sort_keys=True))


def _validated_thinking_header(raw: Any) -> Dict[str, Any]:
    """``chat_markers.thinking_header_facts``'s shape, validated: ``switch``
    a bool; ``off_ids`` a non-empty list of non-negative ints; ``on_ids`` a
    non-empty list of such ints when ``switch`` and None otherwise; ``moves``
    ``on_ids != off_ids`` when ``switch`` and None otherwise. ONE more shape
    is admitted (code review X1; ``chat_markers.thinking_header_facts_or_
    error``): ``switch`` True with ``on_ids`` None AND ``moves`` None — the
    template has a switch but its ON header could not be certified, so the
    ON certification is UNAVAILABLE; every declared thinking-ON request on a
    lens route then refuses ``thinking_on_unattested``. Anything else is a
    wiring defect (``ProxyError``) — the proxy decides refusals from this
    fact and must not decide them from junk. The key set is EXACT (code
    review S-P1): a nullable field is PRESENT and null, never absent, so a
    mapping missing ``moves``/``on_ids`` is refused by name rather than read
    as the partial shape."""
    if not isinstance(raw, Mapping):
        raise ProxyError(f"{THINKING_HEADER_KEY} must be a mapping, got "
                         f"{type(raw).__name__}")
    required = {"switch", "moves", "off_ids", "on_ids"}
    if set(raw) != required:
        raise ProxyError(
            f"{THINKING_HEADER_KEY} must carry exactly the keys {sorted(required)} (a "
            "nullable field is present and null, never absent): missing key(s) "
            f"{sorted(required - set(raw))}, unknown key(s) {sorted(set(raw) - required)}")
    switch = raw["switch"]
    if not isinstance(switch, bool):
        raise ProxyError(f"{THINKING_HEADER_KEY}.switch must be a bool, got {switch!r}")

    def _ids(name, value):
        if not isinstance(value, (list, tuple)) or not value or any(
                not isinstance(t, int) or isinstance(t, bool) or t < 0 for t in value):
            raise ProxyError(f"{THINKING_HEADER_KEY}.{name} must be a non-empty list of "
                             f"non-negative ints, got {value!r}")
        return [int(t) for t in value]
    off = _ids("off_ids", raw["off_ids"])
    if switch:
        if raw["on_ids"] is None:
            # the ON certification is UNAVAILABLE: no on_ids means moves is
            # UNKNOWN — never a bool someone guessed
            if raw["moves"] is not None:
                raise ProxyError(f"{THINKING_HEADER_KEY}: on_ids None (ON certification "
                                 f"unavailable) needs moves None, got moves={raw['moves']!r}")
            return {"switch": True, "moves": None, "off_ids": off, "on_ids": None}
        on = _ids("on_ids", raw["on_ids"])
        moves = raw["moves"]
        if not isinstance(moves, bool) or moves != (on != off):
            raise ProxyError(f"{THINKING_HEADER_KEY}.moves must be the bool on_ids != off_ids "
                             f"({on != off}), got {moves!r}")
        return {"switch": True, "moves": moves, "off_ids": off, "on_ids": on}
    if raw["on_ids"] is not None or raw["moves"] is not None:
        raise ProxyError(f"{THINKING_HEADER_KEY}: a template with no switch has on_ids and "
                         f"moves None, got on_ids={raw['on_ids']!r}, "
                         f"moves={raw['moves']!r}")
    return {"switch": False, "moves": None, "off_ids": off, "on_ids": None}


def _validated_thinking_header_error(header: Optional[Mapping[str, Any]],
                                     error: Any) -> Optional[str]:
    """The ``thinking_header_error`` beside a VALIDATED fact, both directions
    of the §2.5.12 pairing enforced (code review S-P1) — the one invariant the
    constructor, :func:`export_routing_table` and :func:`load_routing_table`
    share: a PARTIAL switched fact (``switch`` True, ``on_ids`` None) carries a
    non-empty reason, and a complete fact, or one with no switch to certify,
    carries None. A reason with no fact at all is admitted (a launcher that
    recorded why and nothing else; echoed alone in ``/v1/models``)."""
    if error is not None and (not isinstance(error, str) or not error):
        raise ProxyError(f"{THINKING_HEADER_ERROR_KEY} must be a non-empty string or None")
    if header is None:
        return error
    partial = header["switch"] and header["on_ids"] is None
    if partial and error is None:
        raise ProxyError(
            f"{THINKING_HEADER_KEY} records on_ids None on a switched template (ON "
            f"certification unavailable) with no {THINKING_HEADER_ERROR_KEY} — a partial "
            "fact carries the reason it is partial or is not served; refusing")
    if error is not None and not partial:
        raise ProxyError(
            f"{THINKING_HEADER_ERROR_KEY} given beside a {THINKING_HEADER_KEY} whose ON "
            "certification is not unavailable — a derived fact has no error, and an "
            "error means on_ids None on a switched template; refusing the contradiction")
    return error


def load_routing_table(path: Path) -> Dict[str, Any]:
    """The exported table, validated, as the proxy process consumes it.
    Accepts :data:`ROUTING_TABLE_VERSIONS`: a version-1
    table loads exactly as before with ``adapters: None`` — lora routes
    included, no block required (the lens standups' tables, and
    ``tests/test_petri_proxy.py``'s v1 round trip WITH lora routes, are
    untouched); a version-2 table must carry a valid :data:`ADAPTERS_KEY`
    block for every lora route and none for any other cell
    (:func:`_validated_adapter_blocks` — shape and consistency only; the
    on-disk re-hash is :func:`verify_adapter_block_on_disk`, the standup's and
    the certification's). Anything else refuses by version number, and a
    version-1 document that smuggles an adapters map refuses too."""
    doc = json.loads(Path(path).read_text())
    version = doc.get("version")
    if version not in ROUTING_TABLE_VERSIONS:
        raise ProxyError(f"routing table version {version!r} not in "
                         f"{ROUTING_TABLE_VERSIONS}")
    strategy = doc.get("exact_replay_strategy")
    if doc["serving_mode"] == "exact_replay":
        if strategy not in EXACT_REPLAY_STRATEGIES:
            raise ProxyError(
                f"exact_replay routing table at {path} records no valid "
                f"exact_replay_strategy (got {strategy!r}) — a stale export; "
                "re-run export_routing_table with the stated mechanism "
                "(D0 constraint 3)")
    elif strategy is not None:
        raise ProxyError(
            f"routing table at {path} carries exact_replay_strategy="
            f"{strategy!r} on serving_mode {doc['serving_mode']!r} — "
            "inconsistent export")
    header = doc.get(THINKING_HEADER_KEY)
    header = _validated_thinking_header(header) if header is not None else None
    routes = tuple(RouteEntry(**r) for r in doc["routes"])
    lens_map = _validated_lens_served_positions(doc.get(LENS_SERVED_POSITIONS_KEY))
    if version == 2:
        if ADAPTERS_KEY not in doc:
            raise ProxyError(f"routing table version 2 at {path} carries no {ADAPTERS_KEY} "
                             "map — a lora-native table binds every lora route to its "
                             "adapter block; refusing")
        adapters: Optional[Dict[str, Dict[str, Any]]] = _validated_adapter_blocks(
            doc[ADAPTERS_KEY], routes)
    else:
        if ADAPTERS_KEY in doc:
            raise ProxyError(f"routing table version 1 at {path} carries an {ADAPTERS_KEY} "
                             "map — adapter blocks ride version-2 tables only; refusing "
                             "the contradiction")
        adapters = None
    return {
        "version": int(version),
        "base_model": doc["base_model"],
        "serving_mode": doc["serving_mode"],
        "exact_replay_strategy": strategy,
        "base_revision": doc.get("base_revision"),
        "routes": routes,
        "prompted_prefixes": tuple(PromptedPrefixSpec(**s)
                                   for s in doc["prompted_prefixes"]),
        # the per-lora-route adapter blocks (version 2), None on version 1
        ADAPTERS_KEY: adapters,
        # an OLD table (no fact) reads None: every enable_thinking: true on a
        # lens route is then unverifiable and refused
        THINKING_HEADER_KEY: header,
        # ... and the reason its ON certification is UNAVAILABLE beside it
        # (§2.5.12, S-P1): a partial fact recorded with no reason is a stale
        # export, refused HERE rather than rebuilt reason-less by `main`
        THINKING_HEADER_ERROR_KEY: _validated_thinking_header_error(
            header, doc.get(THINKING_HEADER_ERROR_KEY)),
        ORGANISMS_KEY: tuple(doc.get(ORGANISMS_KEY, ())),
        # the per-cell SERVED POSITIONS map (§2.5.4 narrowing): an OLD table
        # (no key) reads None, and every undeclared enable_thinking: true on
        # a lens route then refuses exactly as it did before the landing
        LENS_SERVED_POSITIONS_KEY: lens_map,
        # the ENGINE-NEUTRAL map (native-positions-resolve): an OLD table (no
        # key) reads None and the readers fall back to the lens map / the
        # routes' own attestation; a present map is held to the routes, the
        # served vocabularies and the lens map before a byte is served
        SERVED_POSITIONS_KEY: _validated_served_positions(
            doc.get(SERVED_POSITIONS_KEY), routes=routes, lens_map=lens_map),
    }



# --- the Petri agent-loop declaration: structure, entry, render probe -----------


def agent_loop_shape(request: Mapping[str, Any]) -> Optional[str]:
    """The request's agent-loop STRUCTURE, or ``None`` when it has none.

    ``AGENT_LOOP_SHAPE_TOOL_CALL``: the final message is an assistant
    turn carrying tool calls (the auditor's pending call — the model's
    turn is open, awaiting a tool result). ``AGENT_LOOP_SHAPE_TOOL_RESULT``:
    the final message is a tool result (awaiting the model's
    continuation). Either way: a multi-turn transcript (more than one
    non-system message) rendered as a GENERATION-PROMPT request (no
    ``continue_final_message``; ``add_generation_prompt`` not false).
    Structure only, never the render — :meth:`RoutingProxy._validate_agent_loop`
    adds the rendered-ids invariant. Both the entry and the verifier read
    this one classifier, so they cannot disagree about what a "turn of
    the agent loop" is; they differ in what else they check."""
    messages = request.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    if request.get("continue_final_message"):
        return None
    if request.get("add_generation_prompt") is False:
        return None
    non_system = [m for m in messages
                  if isinstance(m, Mapping) and m.get("role") != "system"]
    if len(non_system) < 2:
        return None
    last = messages[-1]
    if not isinstance(last, Mapping):
        return None
    if last.get("role") == "tool":
        return AGENT_LOOP_SHAPE_TOOL_RESULT
    if last.get("role") == "assistant":
        calls = last.get("tool_calls")
        if isinstance(calls, list) and calls:
            return AGENT_LOOP_SHAPE_TOOL_CALL
    return None


def declare_agent_loop(request: Mapping[str, Any], *,
                       lens_route: bool = True,
                       probe: Optional[AgentLoopRender] = None
                       ) -> Dict[str, Any]:
    """THE PETRI-FACING ENTRY: attach :data:`AGENT_LOOP_KEY` to a request
    whose message structure is a turn of the agent loop, whose route is
    lens-served, and — when a render ``probe`` is given — whose RENDER
    does not end at the assistant header; leave every other request, and
    any request that already states the key either way, exactly as it
    came.

    ``lens_route`` is the route's fact, not the request's: the audit's
    undefended base (passthrough) and its LoRA cells talk through the
    same proxy and their agent-loop turns serve natively, with no mask to
    relocate — declaring for them would only be refused by the verifier
    (``agent_loop_not_lens``) and break every non-lens leg of an audit.

    ``probe`` supplies the RENDER half, and the proxy always passes it:
    the declaration relocates the boundary to the last header occurrence,
    which is only meaningful when the render does not already END there.
    Llama-3.1 closes a tool turn (``<|eot_id|>``) and re-appends the
    header under ``add_generation_prompt`` — an ordinary tail render, so
    NO declaration (the tail arm serves it, as it always has); gemma-4
    continues the open model turn after tool traffic — a non-tail render,
    so the declaration is attached and the verifier relocates. A render
    with the header nowhere is declared too, so the verifier refuses it
    LOUDLY (400, "occur NOWHERE") instead of the adapter refusing it
    silently downstream. ``probe=None`` is the structural half alone
    (the offline Modal preflight uses it); it is never what the proxy
    runs. So the entry declares exactly where the declaration means
    something.

    Petri's requests enter sisteer here: inspect_ai's openai provider
    (``petri_runner._inspect_driver`` -> ``get_model(...)``) speaks plain
    ``/v1/chat/completions`` and offers no per-request hook, so the proxy
    declares on its behalf — BY DEFAULT (``petri_agent_loop_entry=True``;
    ``--no-petri-agent-loop`` / ``petri_agent_loop_entry=False`` is the
    stated opt-out). This is one of two layers, deliberately: the entry
    asserts intent from STRUCTURE; :meth:`RoutingProxy._validate_agent_loop`
    then verifies that assertion against the route and the RENDER before
    the selector fact is injected, so a declaration the render does not
    support is a 400 here, never a served mask. A proxy that opted out
    forwards agent-loop shapes undeclared, and they refuse at the adapter
    exactly as before (contained)."""
    out = dict(request)
    if AGENT_LOOP_KEY in out or not lens_route:
        return out
    if agent_loop_shape(out) is None:
        return out
    if probe is not None:
        ids = tuple(int(t) for t in probe.render(out.get("messages"),
                                                 out.get("tools")))
        header = tuple(probe.header_ids)
        H = len(header)
        if len(ids) >= H and ids[-H:] == header:
            return out  # a tail render: the tail arm serves it undeclared
    out[AGENT_LOOP_KEY] = True
    return out


class AgentLoopRender(Protocol):
    """What the agent-loop verifier needs from a render probe: the served
    template's GENERATION header ids (``chat_markers.generation_header_ids``
    — the vocabulary the worker registry records and the adapter matches)
    and the request's rendered token ids under ``add_generation_prompt``."""

    header_ids: Tuple[int, ...]

    def render(self, messages: Sequence[Mapping[str, Any]],
               tools: Optional[Sequence[Mapping[str, Any]]]
               ) -> Sequence[int]:
        ...


class TokenizerAgentLoopRender:
    """The served tokenizer as the proxy's :class:`AgentLoopRender`.

    Renders the request's messages + tools through ``apply_chat_template``
    under vLLM's message normalization: OpenAI tool-call ``arguments``
    arrive as JSON STRINGS and vLLM's chat_utils deserializes them before
    the template runs (the pinned gemma-4 template REFUSES string
    arguments outright), so this does the same; ``None`` content becomes
    the empty string (vLLM's parts-parser yields an empty text for it);
    text content parts pass through unchanged (the template's own
    ``is sequence`` branch renders them exactly as vLLM's ``openai``
    content format hands them over); a non-text part refuses — the audit
    is text-only and a render this probe cannot mirror must not be
    guessed. The thinking-off pin travels with the tokenizer
    (``_sisteer_template_kwargs``), the same seam as every other render
    site.

    This render is a CONSISTENCY PROBE for the declaration, not the served
    ids: the worker locates the boundary in the engine's OWN tokens, so a
    normalization drift here can only over-refuse (a visible 400) or
    forward a request the adapter refuses itself (contained) — never
    serve a wrong boundary silently."""

    def __init__(self, tokenizer: Any):
        from sisteer.chat_markers import generation_header_ids
        self._tok = tokenizer
        self.header_ids: Tuple[int, ...] = tuple(
            int(t) for t in generation_header_ids(tokenizer))
        if not self.header_ids:
            raise ProxyError(
                "the served tokenizer derives an EMPTY generation header — "
                "no agent-loop boundary can be located against it; refusing "
                "to build a render probe that would accept every declaration")
        self._turn_end_ids: Any = _UNSET

    @property
    def turn_end_ids(self) -> Optional[Tuple[int, ...]]:
        """The served template's assistant turn-END scaffold
        (``chat_markers.assistant_turn_end_ids`` — terminator + glue; the
        all-turns mask's second marker), derived LAZILY on the first all-turns
        render check so a probe serving no all-turns route never pays the
        multi-turn probe render (and never trips its certification on a
        template the response mode serves fine). ``None`` for a
        successor-bounded template (no terminator), which cannot serve the
        all-turns mask at all."""
        if self._turn_end_ids is _UNSET:
            from sisteer.chat_markers import assistant_turn_end_ids
            scaffold = assistant_turn_end_ids(self._tok)
            self._turn_end_ids = (None if scaffold is None
                                  else tuple(int(t) for t in scaffold))
        return self._turn_end_ids

    def assistant_content_spans(self, messages: Sequence[Mapping[str, Any]]
                                ) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, int], ...]]:
        """The MESSAGE-derived assistant content spans of ``messages``
        (``chat_markers.assistant_content_spans`` over the normalized
        messages, under the tokenizer's own render pin): ``(full render
        WITHOUT a generation prompt, one (lo, hi) per assistant message)``.
        Its ``ChatMarkerError`` propagates — the all-turns check maps it to
        ``all_turns_expected_unavailable``."""
        from sisteer.chat_markers import assistant_content_spans
        clean = self.normalize_messages(list(messages))
        full, spans = assistant_content_spans(self._tok, clean)
        return tuple(int(t) for t in full), tuple((int(lo), int(hi)) for lo, hi in spans)

    @staticmethod
    def normalize_messages(messages: Any) -> List[Dict[str, Any]]:
        if not isinstance(messages, list) or not messages:
            raise ProxyError("agent-loop render needs a non-empty messages "
                             "list")
        clean: List[Dict[str, Any]] = []
        for i, m in enumerate(messages):
            if not isinstance(m, Mapping) or not isinstance(
                    m.get("role"), str):
                raise ProxyError(f"messages[{i}] is not a {{role, ...}} "
                                 f"object: {m!r}")
            out = dict(m)
            content = out.get("content")
            if content is None:
                out["content"] = ""
            elif isinstance(content, list):
                for j, part in enumerate(content):
                    if (not isinstance(part, Mapping)
                            or part.get("type") != "text"
                            or not isinstance(part.get("text"), str)):
                        raise ProxyError(
                            f"messages[{i}].content[{j}] is not a text part "
                            f"({part!r}) — the agent-loop render probe mirrors "
                            "text-only transcripts; a non-text part cannot be "
                            "rendered here and is refused, never guessed")
            elif not isinstance(content, str):
                raise ProxyError(f"messages[{i}].content has unrenderable "
                                 f"type {type(content).__name__}")
            calls = out.get("tool_calls")
            if calls is not None:
                if not isinstance(calls, list):
                    raise ProxyError(f"messages[{i}].tool_calls is not a list")
                fixed = []
                for k, call in enumerate(calls):
                    if not isinstance(call, Mapping):
                        raise ProxyError(
                            f"messages[{i}].tool_calls[{k}] is not an object")
                    call = dict(call)
                    fn = call.get("function")
                    if isinstance(fn, Mapping):
                        fn = dict(fn)
                        args = fn.get("arguments")
                        if isinstance(args, str):
                            try:
                                fn["arguments"] = json.loads(args)
                            except ValueError as e:
                                raise ProxyError(
                                    f"messages[{i}].tool_calls[{k}].function"
                                    ".arguments is not valid JSON "
                                    f"({e}) — vLLM deserializes string "
                                    "arguments before rendering, so this "
                                    "request cannot render there either") \
                                    from e
                        call["function"] = fn
                    fixed.append(call)
                out["tool_calls"] = fixed
            clean.append(out)
        return clean

    def render(self, messages: Sequence[Mapping[str, Any]],
               tools: Optional[Sequence[Mapping[str, Any]]]
               ) -> Tuple[int, ...]:
        clean = self.normalize_messages(list(messages))
        out = self._tok.apply_chat_template(
            clean, tokenize=True, add_generation_prompt=True,
            tools=(list(tools) if tools else None),
            **getattr(self._tok, "_sisteer_template_kwargs", {}))
        if hasattr(out, "keys"):  # transformers>=5 returns a BatchEncoding
            out = out["input_ids"]
        if out and isinstance(out[0], list):
            out = out[0]
        return tuple(int(i) for i in out)


_UNSET = object()


def _rfind_ids(ids: Sequence[int], sub: Sequence[int]) -> int:
    """Index of the LAST occurrence of ``sub`` in ``ids``, or -1 (the
    adapter's ``_rfind_header`` semantics, so the proxy's consistency
    probe and the worker's locator agree on what "occurs" means)."""
    want = tuple(int(t) for t in sub)
    n, m = len(ids), len(want)
    for start in range(n - m, -1, -1):
        if tuple(int(t) for t in ids[start:start + m]) == want:
            return start
    return -1


# --- backend transport -----------------------------------------------------------


class Backend(Protocol):
    """The upstream chat-completions transport (server-mode vLLM in
    production; a stub in tests)."""

    async def chat_completions(self, payload: Mapping[str, Any]
                               ) -> Mapping[str, Any]:
        ...

    async def tokenize(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """vLLM's ``/tokenize`` (server ROOT): the exact prompt length of a
        chat payload under the served template — the served clients use it
        to resolve a FLOOR-ONLY context-length rejection ("at least N input
        tokens") before clamping (served_window). Read-only; no steering."""
        ...

    async def metrics(self) -> str:
        """vLLM's ``GET /metrics`` (server ROOT): the Prometheus text body,
        VERBATIM. The engine guards it (``vllm_auth.SisteerTokenAuthMiddleware``
        guards every path but the liveness trio), so a same-host sampler that
        does not hold the launch token reads it through the proxy's
        :meth:`RoutingProxy.handle_metrics` pass-through, which is how the
        per-launch bearer stays inside the standup process. Read-only;
        no steering, nothing generated or cached."""
        ...

    async def take_request_failure(self, request_id: str
                                   ) -> Optional[Mapping[str, Any]]:
        """Claim the worker-contained per-request steering failure for
        one completed request (``sisteer_take_request_failure`` over the
        server-mode collective_rpc route), or ``None`` when the request
        served cleanly. The proxy calls this for EVERY lens response
        before forwarding it — the loud half of the 2026-08-30
        containment: a request whose response mask could not be placed
        is served unsteered by the worker (so the engine survives) and
        must therefore be REFUSED here, never returned as a normal
        completion. Since native-psr-serving stage 2 the same claim runs
        for every RESTRICTED-MODE native route's response (the native
        worker's ``sisteer_take_request_failure`` answers the same
        contract, plus a ``{"fault": ...}`` payload once its ledger
        latched — :meth:`RoutingProxy._refuse_unlocated_native_response`)."""
        ...


def upstream_refusal(status: int, body: str) -> ProxyRefusal:
    """The refusal for an upstream vLLM non-200, with the upstream's own
    STATUS CLASS preserved for 4xx (5xx and everything else present as
    502, as before).

    WHY the class must survive the hop (context-overflow triage,
    2026-08-27): a vLLM 400 is a DETERMINISTIC client error — the pinned
    0.25.1 answers a transcript that no longer fits ``max_model_len``
    with HTTP 400 "This model's maximum context length is …"
    (vllm/renderers/params.py -> serve/utils/error_response.py maps
    VLLMValidationError to BAD_REQUEST). The old blanket 502 re-presented
    that as a TRANSIENT server error, and the pinned inspect-ai 0.3.251
    retries 5xx forever by default (_util/http.py is_retryable_http_status;
    model/_retry.py "otherwise retry forever") — so one over-length audit
    turn hung its whole eval task group on an unwinnable retry ladder.
    Forwarded as its real 400, the openai client raises BadRequestError,
    inspect does NOT retry it, and petri surfaces it to the auditor in
    the transcript, where the runner's converter classifies it
    (petri_runner.sample_context_overflow) and routes the scenario to
    the failure ledger for the larger-window re-run pass.

    ``code`` stays ``upstream_error`` for every status ON PURPOSE: shaping
    a context-overflow body as OpenAI's ``context_length_exceeded`` would
    make inspect-ai swallow it into a ModelOutput(stop_reason=
    "model_length") — the audit would then CONTINUE on a silently
    truncated exchange and be judged as if nothing happened, which is
    exactly the bias this classification exists to kill."""
    passthrough = status if 400 <= status < 500 else 502
    return ProxyRefusal(
        f"upstream vLLM returned {status}: {body[:2000]}",
        http_status=passthrough, code="upstream_error")


#: Per-request timeout of the proxy -> vLLM hop (``HTTPBackend``), seconds. Timeouts compose
#: as a MIN across hops, so the binding one must be derived from the experiment, not left at a
#: transport default: the 600 s the two hops shipped with 504'd / socket-timed-out every
#: cap-bound draw of the sampled-decode ladder on GLM-4.6V-Flash at the 8192 rung
#: (2026-09-10: ~9-20 tok/s per sequence on the steered lens route at 76 concurrent, so 8192
#: tokens need 400-900 s, and a 16384 rung 800-1,800 s; with the drivers over-subscribing the
#: server by ~25% a request can also wait one full generation in vLLM's queue). Censoring is
#: correlated with the treatment (steering slows decoding), so a short hop biases exactly the
#: leg under measurement. 7,200 s = the controls driver's own aiohttp total
#: (served_capability_run.AIOHTTP_TOTAL_TIMEOUT_S) and ~2x the derived worst case at 16384. A
#: genuinely hung upstream now surfaces in 2 h instead of 10 min; that is the right trade for a
#: measurement. A launcher that names its own value (``mt_serving_up --upstream-timeout-s``)
#: still wins; this is the default it falls back to.
HOP_TIMEOUT_S = 7200.0
#: Client -> proxy hop (``ProxyClientEngine``): strictly LONGER than the upstream hop so the
#: proxy's 504 (a recorded, attributable failure that also aborts the upstream request) always
#: arrives before the client's socket gives up (which would leave vLLM generating for nobody).
CLIENT_TIMEOUT_S = HOP_TIMEOUT_S + 300.0


class HTTPBackend:
    """aiohttp POST to the vLLM server's OpenAI endpoint. The upstream
    URL normalizes to the canonical ``.../v1`` base at construction
    (root or ``/v1`` forms both accepted — one representation inside).

    ``api_key`` is the launch's per-server token
    (``ServerLaunch.api_key``): the upstream now authenticates every
    /v1 request, so the backend attaches the Bearer header. Held on a
    private attribute and surfaced by nothing — no repr, no record, no
    error message carries it."""

    def __init__(self, upstream_base_url: str, *, timeout_s: float = HOP_TIMEOUT_S,
                 api_key: Optional[str] = None):
        self.upstream_base_url = _canonical_v1_base(upstream_base_url)
        self.timeout_s = float(timeout_s)
        self._api_key = api_key or None
        #: the ATTESTED worker count (``engine_attestation.residual_slot.n_workers``), set by the
        #: RoutingProxy that owns this backend: every failure claim's per-worker result list must
        #: carry exactly this many entries — on a four-worker server
        #: ``{"results": [null]}`` used to pass as clean; None = unknown (a lens standup), then only
        #: a non-empty well-formed list is required
        self.expected_workers: Optional[int] = None

    async def chat_completions(self, payload: Mapping[str, Any]
                               ) -> Mapping[str, Any]:
        import aiohttp  # lazy: the HTTP stack is a serving-time dep

        headers = ({"Authorization": f"Bearer {self._api_key}"}
                   if self._api_key else {})
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                    f"{self.upstream_base_url}/chat/completions",
                    json=dict(payload), headers=headers) as resp:
                body = await resp.text()
                if resp.status != 200:
                    raise upstream_refusal(resp.status, body)
                return json.loads(body)

    async def tokenize(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """POST to the upstream's ``/tokenize`` (server ROOT, like the RPC
        route). A non-200 is forwarded with its status class (a 404 from a
        server that lacks the route reaches the client as unavailable)."""
        import aiohttp  # lazy: the HTTP stack is a serving-time dep

        root = self.upstream_base_url[:-len("/v1")]
        headers = ({"Authorization": f"Bearer {self._api_key}"}
                   if self._api_key else {})
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{root}/tokenize", json=dict(payload),
                                    headers=headers) as resp:
                body = await resp.text()
                if resp.status != 200:
                    raise upstream_refusal(resp.status, body)
                return json.loads(body)

    async def metrics(self) -> str:
        """GET the upstream's ``/metrics`` (server ROOT, like ``/tokenize``) with
        the launch bearer; the body text VERBATIM on 200. A non-200 is forwarded
        with its status class (the engine's 401 for a wrong token reaches the
        client as a 401, never as a silently empty scrape)."""
        import aiohttp  # lazy: the HTTP stack is a serving-time dep

        root = self.upstream_base_url[:-len("/v1")]
        headers = ({"Authorization": f"Bearer {self._api_key}"}
                   if self._api_key else {})
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{root}/metrics", headers=headers) as resp:
                body = await resp.text()
                if resp.status != 200:
                    raise upstream_refusal(resp.status, body)
                return body

    async def collective_rpc(self, method: str, args: Sequence[Any]
                             ) -> Mapping[str, Any]:
        """ONE ``collective_rpc`` invocation over the server-mode route (server
        ROOT, the same route and bearer the launcher's attestation RPCs use);
        returns the parsed ``{"results": [per-worker value]}`` body. Used only
        by :meth:`RoutingProxy.handle_lora_routing_rpc`; the lens
        post-check below keeps its own, unchanged, POST."""
        import aiohttp  # lazy: the HTTP stack is a serving-time dep

        root = self.upstream_base_url[:-len("/v1")]
        headers = ({"Authorization": f"Bearer {self._api_key}"}
                   if self._api_key else {})
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{root}/collective_rpc",
                                    json={"method": str(method), "args": list(args)},
                                    headers=headers) as resp:
                body = await resp.text()
                if resp.status != 200:
                    raise upstream_refusal(resp.status, body)
                return json.loads(body) if body else {}

    async def take_request_failure(self, request_id: str
                                   ) -> Optional[Mapping[str, Any]]:
        """``sisteer_take_request_failure`` over the server-mode
        ``/collective_rpc`` route (server ROOT, not ``/v1`` — the same
        route and auth the launcher's install/attestation RPCs use;
        response shape ``{"results": [per-worker value]}``, read the way
        ``vllm_server._post_collective_rpc`` reads it). Any transport or
        HTTP failure REFUSES the response being checked: an unverifiable
        lens response must never be forwarded as a served result."""
        import aiohttp  # lazy: the HTTP stack is a serving-time dep

        root = self.upstream_base_url[:-len("/v1")]
        url = f"{root}/collective_rpc"
        headers = {"Content-Type": "application/json",
                   **({"Authorization": f"Bearer {self._api_key}"}
                      if self._api_key else {})}
        body = {"method": "sisteer_take_request_failure",
                "args": [str(request_id)]}
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=body,
                                        headers=headers) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        raise ProxyRefusal(
                            f"the lens post-check RPC {url} answered HTTP "
                            f"{resp.status} for request {request_id!r} — "
                            "this response's steering cannot be verified, "
                            "so it is refused rather than forwarded: "
                            f"{text[:500]}",
                            http_status=502, code="lens_postcheck_failed")
                    payload = json.loads(text) if text else {}
        except ProxyRefusal:
            raise
        except Exception as e:
            raise ProxyRefusal(
                f"the lens post-check RPC {url} could not be reached for "
                f"request {request_id!r} ({type(e).__name__}: {e}) — this "
                "response's steering cannot be verified, so it is refused "
                "rather than forwarded",
                http_status=502, code="lens_postcheck_failed") from e
        return parse_failure_claim(payload, url=url, request_id=request_id, n_workers=self.expected_workers)


def parse_failure_claim(payload: Any, *, url: str, request_id: str,
                        n_workers: Optional[int] = None) -> Optional[Mapping[str, Any]]:
    """The ONE reading of a ``sisteer_take_request_failure`` collective_rpc reply
    (``{"results": [per-worker value]}``): a clean claim is established ONLY by a
    NON-EMPTY per-worker result list whose every entry is ``None`` (no failure)
    or a Mapping (an entry / a fault payload) — the first non-None entry is the
    claim. An empty body, ``null``, ``{}``, ``{"results": []}``, a non-list, or a
    result of another type is UNVERIFIABLE and refuses (``lens_postcheck_failed``,
    502; the native route re-labels it ``response_boundary_postcheck_failed``):
    a claim that cannot be judged must never read as "no failure" (an earlier
    reading returned None for every one of those shapes). With ``n_workers``
    (the ATTESTED worker count, ``engine_attestation.residual_slot.n_workers``)
    the list must carry EXACTLY that many entries: on a four-worker server
    ``{"results": [null]}`` is one worker's word, not the engine's, and an
    over-long list is not the attested engine's answer either."""
    results = payload.get("results") if isinstance(payload, Mapping) else None
    if not isinstance(results, list) or not results or \
            not all(r is None or isinstance(r, Mapping) for r in results):
        raise ProxyRefusal(
            f"the post-check RPC {url} answered request {request_id!r} without a per-worker "
            f"result list ({str(payload)[:200]!r}) — a claim that cannot be judged must never "
            "read as 'no failure'; this response is refused rather than forwarded",
            http_status=502, code="lens_postcheck_failed")
    if n_workers is not None and len(results) != n_workers:
        raise ProxyRefusal(
            f"the post-check RPC {url} answered request {request_id!r} with {len(results)} per-worker "
            f"result(s) but the attestation names {n_workers} worker(s) — an incomplete claim is not a "
            "clean one; this response is refused rather than forwarded",
            http_status=502, code="lens_postcheck_failed")
    for result in results:
        if result is not None:
            return result
    return None


# --- the routing core -------------------------------------------------------------


class RoutingProxy:
    """The framework-free async routing core (module docstring)."""

    def __init__(
        self,
        routes: Sequence[RouteEntry],
        backend: Backend,
        *,
        base_model: str,
        serving_mode: str,
        prompted_prefixes: Sequence[PromptedPrefixSpec] = (),
        exact_replay_strategy: Optional[str] = None,
        organisms: Sequence[str] = (),
        launch_attestation: Optional[Mapping[str, str]] = None,
        api_token: Optional[str] = None,
        agent_loop_render: Optional[AgentLoopRender] = None,
        petri_agent_loop_entry: bool = True,
        base_revision: Optional[str] = None,
        resumed_prefill_policy_cells: Optional[Mapping[str, str]] = None,
        lens_serve_declarations: Optional[Mapping[str, Mapping[str, Any]]] = None,
        lens_serve_declarations_error: Optional[str] = None,
        thinking_header: Optional[Mapping[str, Any]] = None,
        thinking_header_error: Optional[str] = None,
        lens_salt_policy: str = LENS_SALT_POLICY_ATTESTED,
        object_artifact_sha256_by_cell: Optional[Mapping[str, str]] = None,
        lens_served_positions: Optional[Mapping[str, Sequence[str]]] = None,
        adapter_blocks: Optional[Mapping[str, Mapping[str, Any]]] = None,
        engine_attestation: Optional[Mapping[str, Any]] = None,
        thinking_mode: str = THINKING_MODE_OFF,
    ):
        if serving_mode not in cm.SERVING_MODES:
            raise ProxyError(f"serving_mode {serving_mode!r} not in "
                             f"{cm.SERVING_MODES}")
        if not base_model:
            raise ProxyError("base_model must be a non-empty model id")
        # the launch attestation as handed over (LENS_REDUCTION_MODE_KEY rides
        # in it when the standup attested the served reduction mode)
        self._launch_attestation = dict(launch_attestation or {})
        # exact_replay needs an ENFORCEABLE mechanism, stated and verified
        # at startup: either this proxy serializes, or the server attests
        # VLLM_BATCH_INVARIANT=1 in its launch env — never an unattested
        # flag that quietly mislabels nondeterministic outputs as exact
        # replay.
        if serving_mode == "exact_replay":
            if exact_replay_strategy not in EXACT_REPLAY_STRATEGIES:
                raise ProxyError(
                    f"serving_mode=exact_replay needs an explicit "
                    f"exact_replay_strategy from {EXACT_REPLAY_STRATEGIES}, "
                    f"got {exact_replay_strategy!r} — a stated, recorded "
                    "choice, never a default (D0 constraint 3)")
            if exact_replay_strategy == "serialize":
                self._serialize = True
            else:  # batch_invariant
                att = dict(launch_attestation or {})
                if att.get(BATCH_INVARIANT_ENV) != "1":
                    raise ProxyError(
                        "exact_replay_strategy='batch_invariant' but the "
                        f"launch attestation does not carry "
                        f"{BATCH_INVARIANT_ENV}=1 — refusing to start with "
                        "serialization off and no proof the server-side "
                        "mechanism is active (D0 constraint 3); pass the "
                        "composed ServerLaunch env "
                        "(vllm_server.build_server_launch) as "
                        "launch_attestation")
                self._serialize = False
        else:
            if exact_replay_strategy is not None:
                raise ProxyError(
                    f"exact_replay_strategy={exact_replay_strategy!r} given "
                    f"but serving_mode is {serving_mode!r} — a replay "
                    "mechanism on an interleaved run would mislabel every "
                    "row's mode")
            self._serialize = False
        self.exact_replay_strategy = exact_replay_strategy
        self._routes: Dict[str, RouteEntry] = {}
        for r in routes:
            if r.virtual_model in self._routes:
                raise ProxyError(f"duplicate virtual model "
                                 f"{r.virtual_model!r} in routes")
            ident = cm.parse_serving_identity(r.serving_identity)
            if ident.mode != serving_mode:
                raise ProxyError(
                    f"route {r.virtual_model!r} carries mode {ident.mode!r} "
                    f"but this proxy serves {serving_mode!r} — one audit "
                    "run, one mode (plan §3 H5)")
            self._routes[r.virtual_model] = r
        self._prefixes: Dict[str, PromptedPrefixSpec] = {}
        for spec in prompted_prefixes:
            self._prefixes[spec.virtual_model] = spec
        # every proxy-served prompted vmodel MUST be verifiable, up front
        for r in self._routes.values():
            if (r.si_prefix_hash is not None
                    and cm.parse_serving_identity(
                        r.serving_identity).backend == "proxy_vllm"):
                spec = self._prefixes.get(r.virtual_model)
                if spec is None:
                    raise ProxyError(
                        f"prompted vmodel {r.virtual_model!r} has no "
                        "PromptedPrefixSpec — the proxy cannot verify what "
                        "it cannot measure (H8); refusing to start")
                if spec.sha16 != r.si_prefix_hash:
                    raise ProxyError(
                        f"prompted vmodel {r.virtual_model!r}: prefix spec "
                        f"hash {spec.sha16} != route's {r.si_prefix_hash}")
        #: the per-cell RESUMED-PREFILL POLICY as the workers ATTESTED it
        #: (``lens_worker.resumed_prefill_policy_cells_attested`` over the
        #: launch's steering attestation), handed over by the launcher —
        #: ``None`` when nothing was attested (an old worker, an unreadable
        #: attestation, workers that disagreed), which reads as "nothing is
        #: resumable" and leaves every steered leg on the request-unique salt
        #: it always had. A MALFORMED map is a startup refusal, not a
        #: fall-back: the reader hands over ``None`` or a good map, so junk
        #: here is a wiring defect and must be loud before any traffic.
        if resumed_prefill_policy_cells is None:
            self.resumed_prefill_policy_cells: Optional[Dict[str, str]] = None
        else:
            if not isinstance(resumed_prefill_policy_cells, Mapping):
                raise ProxyError(
                    "resumed_prefill_policy_cells must be a cell -> policy "
                    f"mapping or None, got "
                    f"{type(resumed_prefill_policy_cells).__name__}")
            bad = {str(k): v for k, v in resumed_prefill_policy_cells.items()
                   if v not in RESUMED_PREFILL_POLICIES}
            if bad:
                raise ProxyError(
                    f"resumed_prefill_policy_cells carries values outside "
                    f"{RESUMED_PREFILL_POLICIES}: {sorted(bad.items())} — "
                    "the attested reader hands over a clean map or None; "
                    "refusing to start rather than guess a policy")
            self.resumed_prefill_policy_cells = {
                str(k): str(v) for k, v in resumed_prefill_policy_cells.items()}
        #: the per-cell SERVE DECLARATIONS as the workers ATTESTED them
        #: (``lens_worker.serve_declarations_attested``), handed over by the
        #: launcher: ``None`` when nothing was attested (every thinking-ON
        #: declaration then refuses ``thinking_on_unattested`` — fail closed),
        #: ``{}`` when the standup served no declared cell (a stated fact: a
        #: thinking-ON declaration then refuses
        #: ``thinking_on_cell_not_declared``), else the map. A malformed map
        #: is a startup refusal, like the policy map. The launcher's reason
        #: for attesting nothing rides beside it
        #: (``lens_serve_declarations_error``) for the identity block; a map
        #: AND an error together is a contradiction, refused.
        if lens_serve_declarations is None:
            self.lens_serve_declarations: Optional[Dict[str, Dict[str, Any]]] = None
        else:
            if not isinstance(lens_serve_declarations, Mapping):
                raise ProxyError(
                    f"{SERVE_DECLARATIONS_KEY} must be a cell -> declaration mapping or "
                    f"None, got {type(lens_serve_declarations).__name__}")
            clean: Dict[str, Dict[str, Any]] = {}
            for cell, value in lens_serve_declarations.items():
                if not isinstance(cell, str) or not cell or not isinstance(value, Mapping):
                    raise ProxyError(
                        f"{SERVE_DECLARATIONS_KEY}[{cell!r}] = {value!r} is not a cell -> "
                        "declaration entry")
                try:
                    decl = sd.declarations_of(value)
                except sd.ServeDeclarationError as e:
                    raise ProxyError(f"{SERVE_DECLARATIONS_KEY}[{cell!r}]: {e}") from e
                if not decl or set(decl) != set(value):
                    raise ProxyError(
                        f"{SERVE_DECLARATIONS_KEY}[{cell!r}] = {dict(value)!r} is not exactly "
                        "one closed declaration shape — the attested reader hands over a "
                        "clean map or None; refusing to start rather than guess")
                clean[cell] = decl
            self.lens_serve_declarations = clean
        if lens_serve_declarations_error is not None:
            if not isinstance(lens_serve_declarations_error, str) \
                    or not lens_serve_declarations_error:
                raise ProxyError(f"{SERVE_DECLARATIONS_ERROR_KEY} must be a non-empty "
                                 "string or None")
            if lens_serve_declarations is not None:
                raise ProxyError(
                    f"{SERVE_DECLARATIONS_KEY} given together with "
                    f"{SERVE_DECLARATIONS_ERROR_KEY} — a map that was attested has no "
                    "error, and an error means no map; refusing the contradiction")
        self.lens_serve_declarations_error = lens_serve_declarations_error
        #: the served template's thinking-header fact (routing table
        #: ``thinking_header``; ``chat_markers.thinking_header_facts``) —
        #: validated at construction; ``None`` (an old table) makes every
        #: ``enable_thinking: true`` on a lens route unverifiable and refused
        self._thinking_header = (_validated_thinking_header(thinking_header)
                                 if thinking_header is not None else None)
        #: why the fact's ON certification is UNAVAILABLE (code review X1) —
        #: the launcher's reason, echoed beside the fact; None beside a fully
        #: derived fact. BOTH directions hold (code review S-P1): a partial
        #: fact with no reason is refused, as is a reason beside a fact whose
        #: on_ids ARE present (or that has no switch to certify).
        self.thinking_header_error = _validated_thinking_header_error(
            self._thinking_header, thinking_header_error)
        #: the per-cell SERVED POSITIONS map (routing table
        #: ``lens_served_positions``; §2.5.4 narrowing) the thinking-ON gate
        #: reads a cell's header-locatedness from — validated at construction;
        #: ``None`` (an old table, a launcher that handed none over) keeps
        #: every undeclared ``enable_thinking: true`` on a lens route refusing
        #: as before, and so does a cell the map does not name (fail closed)
        self.lens_served_positions = _validated_lens_served_positions(lens_served_positions)
        # THE THINKING MODE (THINKING_MODES' docstring): a closed, standup-level
        # fact. ``off`` keeps every request path exactly as it was; ``on`` stands
        # on a CERTIFIED header fact and an ATTESTED declarations map — both
        # fail-closed at startup, never per request.
        if thinking_mode not in THINKING_MODES:
            raise ProxyError(
                f"{THINKING_MODE_KEY} {thinking_mode!r} is not one of {THINKING_MODES} — "
                "the render regime is a closed, recorded choice, never a free string")
        self.thinking_mode = thinking_mode
        if thinking_mode == THINKING_MODE_ON:
            if self._thinking_header is None or not self._thinking_header["switch"]:
                raise ProxyError(
                    f"{THINKING_MODE_KEY}='on' needs the served template's thinking-header "
                    f"fact with a thinking switch ({THINKING_HEADER_KEY}), got "
                    f"{self._thinking_header!r} — a template with no switch has no "
                    "thinking-ON render to serve; refusing to start")
            if self._thinking_header["on_ids"] is None:
                raise ProxyError(
                    f"{THINKING_MODE_KEY}='on' but the served template's thinking-ON header "
                    "certification is UNAVAILABLE"
                    + (f" ({self.thinking_header_error})" if self.thinking_header_error else "")
                    + " — a thinking-ON standup cannot stand on an uncertified header; "
                    "refusing to start")
            if self.lens_serve_declarations is None:
                raise ProxyError(
                    f"{THINKING_MODE_KEY}='on' but no attested {SERVE_DECLARATIONS_KEY} map was "
                    "handed over"
                    + (f" ({self.lens_serve_declarations_error})"
                       if self.lens_serve_declarations_error else "")
                    + " — nothing vouches for which cells may serve the declared render; "
                    "refusing to start (fail closed)")
        if lens_salt_policy not in LENS_SALT_POLICIES:
            raise ProxyError(
                f"{LENS_SALT_POLICY_KEY} {lens_salt_policy!r} is not one of "
                f"{LENS_SALT_POLICIES} — the lens cache regime is a closed, recorded "
                "choice, never a free string")
        self.lens_salt_policy = lens_salt_policy
        #: the per-cell fitted-artifact digests the standup was held to
        #: (``{harness id (variant spelling): sha256}``) — a LAUNCH
        #: fact echoed in the identity block, validated nothing against
        if object_artifact_sha256_by_cell is None:
            self.object_artifact_sha256_by_cell: Optional[Dict[str, str]] = None
        else:
            if not isinstance(object_artifact_sha256_by_cell, Mapping) or any(
                    not isinstance(k, str) or not k or not isinstance(v, str) or not v
                    for k, v in object_artifact_sha256_by_cell.items()):
                raise ProxyError(f"{OBJECT_SHA256_BY_CELL_KEY} must be a cell -> sha256 mapping "
                                 f"of non-empty strings, got {object_artifact_sha256_by_cell!r}")
            self.object_artifact_sha256_by_cell = {
                str(k): str(v) for k, v in object_artifact_sha256_by_cell.items()}
        #: the per-lora-route ADAPTER BLOCKS (a version-2 table's
        #: ``adapters`` map) — ``None`` when none
        #: were handed over (every lens standup; a version-1 table), else the
        #: map held to the routes exactly as the exporter and the loader hold
        #: it: every lora route covered, no block for any other cell
        #: (:func:`_validated_adapter_blocks`)
        route_list = list(self._routes.values())
        if adapter_blocks is None:
            self.adapter_blocks: Optional[Dict[str, Dict[str, Any]]] = None
        else:
            self.adapter_blocks = _validated_adapter_blocks(adapter_blocks, route_list)
        #: the ENGINE ATTESTATION a lora-native standup handed over
        #: (:func:`validated_engine_attestation`; ``None`` on every lens
        #: standup). Given, three things are held BEFORE any traffic: (1) no
        #: route carries path ``lens`` (a plugin-free server has no hook — a
        #: lens route would be base served under a ``+lens+`` identity);
        #: (2) every lora route has a block (a natively served adapter with
        #: no digests cannot be held to anything); (3) each block's digests,
        #: zero flag and (genuine) intervention hash equal the attestation's
        #: entry under that name, and the attestation names no adapter the
        #: table does not route — the table, the blocks and the server's
        #: loaded adapters name one set.
        if engine_attestation is None:
            self.engine_attestation: Optional[Dict[str, Any]] = None
        else:
            att = validated_engine_attestation(engine_attestation)
            lens_routes = sorted(r.cell_id for r in route_list if _route_path(r) == "lens")
            if lens_routes:
                raise ProxyError(
                    f"lens route(s) {lens_routes} on a proxy started with a "
                    f"{ENGINE_KIND_LORA_NATIVE} engine attestation — a plugin-free server "
                    "has no lens hook, so such a route would be base served under a "
                    "+lens+ identity; refusing to start (enumerate with lens_controls=False)")
            lora_cells = sorted(r.cell_id for r in route_list if _route_path(r) == "lora")
            if self.adapter_blocks is None:
                raise ProxyError(
                    f"engine attestation given but no {ADAPTERS_KEY} blocks — lora route(s) "
                    f"{lora_cells} cannot be held to the attested adapters; refusing to start")
            for cid in lora_cells:
                block = self.adapter_blocks[cid]   # coverage held by the validator above
                entry = att["adapters"].get(cid)
                if entry is None:
                    raise ProxyError(
                        f"lora route {cid!r} has an adapter block but the engine attestation "
                        f"names no adapter {cid!r} (attested: {sorted(att['adapters'])}) — the "
                        "server did not load what the table serves; refusing to start")
                for key in ("config_sha256", "weights_sha256"):
                    if block[key] != entry[key]:
                        raise ProxyError(
                            f"lora route {cid!r}: adapter block {key} {block[key]} != the "
                            f"engine attestation's {entry[key]} — the adapter the server "
                            "loaded under this name is not the one the table binds; "
                            "refusing to start")
                # WHICH KIND of adapter the server loaded under this name is
                # decided by the ATTESTATION's flag, held to the table's,
                # before any exemption. The table's flag is what
                # validated_adapter_block derived from the cell id, so on
                # its own it only says what the route CLAIMS; a regenerated
                # table that named a genuine adapter `none__lora0` with that
                # adapter's digests copied over used to start here and
                # return genuine steering stamped as the no-op control.
                if entry["petri_zero_adapter"] != block["petri_zero_adapter"]:
                    attested_kind = ("the ZERO adapter" if entry["petri_zero_adapter"]
                                     else "a GENUINE adapter")
                    consequence = (
                        "genuine steering would be returned and stamped as the no-op control"
                        if block["petri_zero_adapter"] else
                        "the base model would be served under a steered identity")
                    raise ProxyError(
                        f"lora route {cid!r}: the table's adapter block flags "
                        f"petri_zero_adapter={block['petri_zero_adapter']} but the engine "
                        f"attestation records {attested_kind} loaded under this name "
                        f"(petri_zero_adapter={entry['petri_zero_adapter']}, "
                        f"petri_intervention_hash={entry['petri_intervention_hash']!r}) — "
                        f"{consequence}; refusing to start")
                # the INTERVENTION the loaded adapter was exported from must be
                # the one the route/block declares (code-review finding,
                # 2026-09-11): the block's hash is held to the route's by
                # validated_adapter_block, so this closes the chain
                # route -> block -> attested adapter_config. Only an adapter
                # BOTH sides flag zero is exempt: its attested hash is None
                # (validated_engine_attestation refuses a zero-flagged entry
                # with a hash) while its block carries `none`'s hash by
                # construction.
                if not block["petri_zero_adapter"]:
                    if entry["petri_intervention_hash"] != block["petri_intervention_hash"]:
                        raise ProxyError(
                            f"lora route {cid!r}: the table binds intervention "
                            f"{block['petri_intervention_hash']!r} but the engine attestation "
                            f"records the adapter loaded under this name was exported from "
                            f"{entry['petri_intervention_hash']!r} — rows would be stamped "
                            "with one intervention's hash while another's weights serve; "
                            "refusing to start")
            unrouted = sorted(set(att["adapters"]) - set(lora_cells))
            if unrouted:
                raise ProxyError(
                    f"engine attestation names adapter(s) {unrouted} that no lora route "
                    "serves — the table and the server must name one adapter set; "
                    "refusing to start")
            self.engine_attestation = att
        #: the all-turns render check's outcomes, counted per label and kept
        #: as a bounded log of refusals (forwarded requests carry the label
        #: on their audit row; a refused request never reaches that row, so
        #: its outcome is recorded HERE and in the structured log)
        self.n_all_turns_check: Dict[str, int] = {label: 0 for label in ALL_TURNS_CHECK_LABELS}
        self.all_turns_refusals: Deque[Dict[str, Any]] = deque(maxlen=REQUEST_LOG_MAX)
        self.base_model = base_model
        orgs = tuple(sorted(set(str(o) for o in organisms)))
        for o in orgs:
            if not o or ORGANISM_SEP in o:
                raise ProxyError(f"organism name {o!r} is empty or carries {ORGANISM_SEP!r}")
        self.organisms: Tuple[str, ...] = orgs
        self.serving_mode = serving_mode
        self._backend = backend
        #: the boundary-refusal control's per-standup nonce (O49): the marker's required VALUE; the
        #: standup reads it off this object, so no client outside the standup can present it
        self.boundary_control_nonce: str = secrets.token_hex(16)
        # the attested worker count binds every failure claim the backend reads (parse_failure_claim);
        # validated_engine_attestation guarantees an ARMED slot carries it, so a claim is never read
        # count-free on an engine that can raise one
        n_workers = ((self.engine_attestation or {}).get("residual_slot") or {}).get("n_workers")
        if n_workers is not None and hasattr(backend, "expected_workers"):
            backend.expected_workers = int(n_workers)
        self._api_token = api_token
        #: the agent-loop verifier's render probe (:data:`AGENT_LOOP_KEY`):
        #: EXPLICIT when the launcher supplied it (manifest-pinned, eager),
        #: else built LAZILY from ``base_model`` (+ ``base_revision``) on
        #: the first agent-loop turn (:meth:`_agent_loop_probe`). A
        #: declaration is never honoured on structure alone: a probe that
        #: cannot be built is a per-request 400.
        self._agent_loop_render = agent_loop_render
        self._agent_loop_render_mode = (
            "explicit" if agent_loop_render is not None else "lazy")
        self._base_revision = base_revision
        #: the Petri-facing entry (:func:`declare_agent_loop`): ON by
        #: default; ``False`` is a stated opt-out (constant docstring).
        self._petri_agent_loop_entry = bool(petri_agent_loop_entry)
        self._lock = asyncio.Lock()
        #: the per-request audit log: one dict per request carrying
        #: the CONTRACT field names — ``virtual_model`` and
        #: ``petri_intervention_hash`` — plus cell/identity context.
        #: BOUNDED in memory (:data:`REQUEST_LOG_MAX`, a deque holding
        #: the most recent entries) so a long audit cannot grow the
        #: proxy without limit; the DURABLE per-request record is the
        #: structured logging stream — every request is also emitted as
        #: a logging record with the same fields, and that stream is
        #: what satisfies the audit requirement over a full run.
        #: ``n_request_log_dropped`` counts entries the in-memory tail
        #: evicted (never silently).
        self.request_log: Deque[Dict[str, str]] = deque(
            maxlen=REQUEST_LOG_MAX)
        self.n_request_log_dropped = 0
        #: the salt-discipline guard's memory (:meth:`_guard_cache_salt`):
        #: every distinct ``cache_salt`` forwarded upstream, mapped to the
        #: :class:`_SaltClaim` of its first use — whether a STEERED
        #: (selector-carrying) request used it, whose cell's namespace it is,
        #: and the regime it was minted under (a repeat is legitimate for
        #: exactly one of them: a cell-stable lens salt on its own cell).
        #: ``_salt_guard_order`` holds the same salts in first-seen order
        #: so the bound (:data:`SALT_GUARD_MAX`) evicts oldest-first;
        #: ``n_salt_guard_evicted`` counts what fell out (never silent).
        self._salt_guard: Dict[str, _SaltClaim] = {}
        self._salt_guard_order: Deque[str] = deque()
        self.n_salt_guard_evicted = 0

    # -- helpers -------------------------------------------------------------

    def virtual_models(self) -> Tuple[str, ...]:
        return tuple(sorted(self._routes)) + self.overlay_virtual_models()

    def overlay_virtual_models(self) -> Tuple[str, ...]:
        """``<vmodel>@<organism>`` for every proxy-served non-``lora`` route x
        declared organism (empty when no organism was loaded)."""
        if not self.organisms:
            return ()
        out = []
        for v, r in sorted(self._routes.items()):
            ident = cm.parse_serving_identity(r.serving_identity)
            if ident.backend != "proxy_vllm" or ident.path == "lora":
                continue
            out.extend(f"{v}{ORGANISM_SEP}{o}" for o in self.organisms)
        return tuple(out)

    def _resolve(self, vmodel: str) -> Tuple[RouteEntry, Optional[str]]:
        """The route a request's ``model`` names, plus the organism when it
        is an overlay ``<vmodel>@<organism>``."""
        route = self._routes.get(vmodel)
        if route is not None:
            return route, None
        if ORGANISM_SEP in vmodel:
            base_v, _, org = vmodel.rpartition(ORGANISM_SEP)
            base_route = self._routes.get(base_v)
            if base_route is not None and org in self.organisms:
                ident = cm.parse_serving_identity(base_route.serving_identity)
                if ident.path == "lora":
                    raise ProxyRefusal(
                        f"{vmodel!r}: a lora-path leg cannot be overlaid with an organism — "
                        "its upstream model is already its own adapter",
                        code="organism_on_lora_leg")
                return base_route, org
        raise ProxyRefusal(
            f"unknown virtual model {vmodel!r} — this proxy serves "
            f"exactly {list(self.virtual_models())}", http_status=404,
            code="unknown_virtual_model")

    @staticmethod
    def _system_text(message: Mapping[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):  # OpenAI content-part lists
            parts = []
            for part in content:
                if not (isinstance(part, Mapping)
                        and part.get("type") == "text"
                        and isinstance(part.get("text"), str)):
                    raise ProxyRefusal(
                        "system message content parts must all be text "
                        "parts for SI-prefix verification (H8)")
                parts.append(part["text"])
            return "".join(parts)
        raise ProxyRefusal("system message content must be a string or a "
                           "list of text parts")

    def _verify_prompted_prefix(self, route: RouteEntry,
                                request: Mapping[str, Any]) -> None:
        """H8: the request's system prefix must hash-match the vmodel's
        expected SI bytes — a missing or wrong prefix can never masquerade
        as a valid prompted baseline. Runs for EVERY route carrying
        ``si_prefix_hash``: prompted baselines AND stacked cells — on a
        stacked cell this is what makes the prompt leg un-droppable (the
        selector injection in ``_rewrite`` makes the object leg so)."""
        spec = self._prefixes[route.virtual_model]
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ProxyRefusal(
                f"prompted vmodel {route.virtual_model!r}: request has no "
                "messages to verify the SI prefix in (H8)",
                code="si_prefix_missing")
        first = messages[0]
        # inspect_ai's openai provider labels the target's system message
        # role "developer" (OpenAI's modern system-role label), NOT
        # "system" — verified live on the gemma4e4b audit (2026-08-30, cell
        # prompted__safety: every request 400'd "got role 'developer'").
        # Both are system-role messages; accept either. The SI-prefix hash
        # check below is unchanged — this only recognizes the role label
        # inspect actually emits, it does not relax what content is
        # verified.
        if (not isinstance(first, Mapping)
                or first.get("role") not in ("system", "developer")):
            raise ProxyRefusal(
                f"prompted vmodel {route.virtual_model!r}: the first "
                "message must be the system-or-developer message carrying "
                "the SI prefix (H8) — got role "
                f"{first.get('role') if isinstance(first, Mapping) else first!r}",
                code="si_prefix_missing")
        data = self._system_text(first).encode("utf-8")
        if len(data) < spec.byte_length:
            raise ProxyRefusal(
                f"prompted vmodel {route.virtual_model!r}: system message "
                f"is {len(data)} bytes, shorter than the expected "
                f"{spec.byte_length}-byte SI prefix (H8)",
                code="si_prefix_missing")
        got = cm.sha256_hex16(data[:spec.byte_length])
        if got != route.si_prefix_hash:
            raise ProxyRefusal(
                f"prompted vmodel {route.virtual_model!r}: system prefix "
                f"hashes to {got}, expected {route.si_prefix_hash} — a "
                "wrong prefix never masquerades as a valid prompted "
                "baseline (H8)", code="si_prefix_mismatch")

    @staticmethod
    def _leading_system_text(message: Mapping[str, Any]) -> str:
        """The LEADING textual content of a system message: string content
        verbatim; for a content-part list, the concatenation of the
        leading consecutive text parts, stopping at the first non-text
        part. A registered SI prefix is a PREFIX — an image or other
        non-text part rides AFTER any leading text, so it can never
        un-say leading bytes that already match (a mixed
        ``[text, image]`` system message must not bypass the check)."""
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if (isinstance(part, Mapping) and part.get("type") == "text"
                        and isinstance(part.get("text"), str)):
                    parts.append(part["text"])
                    continue
                break
            return "".join(parts)
        return ""

    def _refuse_unexpected_prefix(self, request: Mapping[str, Any],
                                  route: RouteEntry) -> None:
        """The reverse masquerade: a NON-prompted, NON-stacked leg
        carrying a registered SI prefix would corrupt that leg's identity
        — refuse. (A stacked route never reaches this: its
        ``si_prefix_hash`` routes it through ``_verify_prompted_prefix``,
        where the prefix is EXPECTED and verified instead.)
        EVERY system message's leading textual bytes are inspected (not
        just an all-text first message): a prefix smuggled behind a
        non-text part or into a later system message refuses too."""
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            return
        for message in messages:
            if not isinstance(message, Mapping) \
                    or message.get("role") != "system":
                continue
            data = self._leading_system_text(message).encode("utf-8")
            for spec in self._prefixes.values():
                if (len(data) >= spec.byte_length
                        and cm.sha256_hex16(data[:spec.byte_length])
                        == spec.sha16):
                    raise ProxyRefusal(
                        f"vmodel {route.virtual_model!r} is not a prompted "
                        "baseline, but the request carries a registered SI "
                        f"prefix (matches {spec.virtual_model!r}) — route "
                        "prompted requests at their own vmodel (H8)",
                        code="si_prefix_unexpected")

    def _validate_response_prefill(self, request: Mapping[str, Any],
                                   route: RouteEntry) -> bool:
        """Verify the ``petri_response_prefill`` declaration against the
        request's ACTUAL shape and the route's serving path, returning
        the verified bit for :meth:`_rewrite` (module constant docstring).

        Both directions refuse, so the seeded-render mask semantics can
        neither be invoked by accident nor skipped by omission:

        - the key set on a NON-lens route, or without a genuinely
          continued final assistant message, is a declaration the serving
          cannot honor — refused, never quietly ignored;
        - a CONTINUED final message on a lens route WITHOUT the key is
          the undeclared seeded render the worker-side mask refuses
          INSIDE the engine, where a refusal is fatal to the whole
          server — refused here instead, per-request, before any engine
          sees it (strictly earlier, nothing widened).
        """
        raw = request.get(RESPONSE_PREFILL_KEY)
        if raw is not None and not isinstance(raw, bool):
            raise ProxyRefusal(
                f"{RESPONSE_PREFILL_KEY} must be a JSON boolean, got "
                f"{raw!r} — the seeded-render declaration is a stated "
                "fact, never a truthy coercion",
                code="response_prefill_invalid")
        declared = bool(raw)
        lens = (cm.parse_serving_identity(route.serving_identity).path
                == "lens")
        continued = bool(request.get("continue_final_message"))
        if not declared:
            if lens and continued:
                raise ProxyRefusal(
                    f"vmodel {route.virtual_model!r} (cell "
                    f"{route.cell_id!r}) is lens-served and the request "
                    "continues its final message (a prefill-seeded "
                    f"render) without declaring {RESPONSE_PREFILL_KEY} — "
                    "the response mask must be re-located for a seeded "
                    "render, and an undeclared one refuses inside the "
                    "engine, fatally; declare it (teacher-forced scoring) "
                    "or drop the seed", code="response_prefill_undeclared")
            return False
        if not lens:
            raise ProxyRefusal(
                f"vmodel {route.virtual_model!r} (cell {route.cell_id!r}) "
                f"is not lens-served, but the request declares "
                f"{RESPONSE_PREFILL_KEY} — there is no position mask on "
                "this route, so the declaration would claim a mask "
                "coverage that does not exist; send the seeded render "
                "without the key", code="response_prefill_not_lens")
        if not continued:
            raise ProxyRefusal(
                f"{RESPONSE_PREFILL_KEY} declared but the request does "
                "not set continue_final_message — the declaration must "
                "match the render (a generation-prompt render has no "
                "seed to mask)", code="response_prefill_mismatch")
        if request.get("add_generation_prompt") is not False:
            raise ProxyRefusal(
                f"{RESPONSE_PREFILL_KEY} declared but add_generation_prompt "
                "is not explicitly false — a continued final message must "
                "state add_generation_prompt: false (the upstream default "
                "is true, and the pinned vLLM's both-set validator only "
                "fires when the key is present)",
                code="response_prefill_mismatch")
        messages = request.get("messages")
        last = messages[-1] if isinstance(messages, list) and messages else None
        if not isinstance(last, Mapping) or last.get("role") != "assistant":
            raise ProxyRefusal(
                f"{RESPONSE_PREFILL_KEY} declared but the final message is "
                f"not an assistant turn (got role="
                f"{last.get('role') if isinstance(last, Mapping) else None!r})"
                " — the seed must be the assistant prefix being continued",
                code="response_prefill_mismatch")
        return True

    def _validate_agent_loop(self, request: Mapping[str, Any],
                             route: RouteEntry) -> bool:
        """Verify the ``petri_agent_loop`` declaration against the
        route, the request's message STRUCTURE and its RENDER, returning
        the verified bit for :meth:`_rewrite` (module constant docstring).

        Only the declared direction is judged here. An UNDECLARED
        agent-loop render is forwarded untouched and refuses at the
        adapter exactly as before (contained per request by the worker's
        failure ledger, surfaced as the lens post-check's 500) — that
        direction stays pinned, so restoring it is never a side effect of
        this gate. Declared, every mismatch is a 400:

        - a non-boolean value, or the key beside ``petri_response_prefill``
          (disjoint shapes: a continued final message vs an open model
          turn under a generation prompt);
        - a non-lens route (no mask to relocate);
        - a structure that is not a turn of the agent loop
          (:func:`agent_loop_shape`): single-turn, a continued final
          message, ``add_generation_prompt: false``, or a final message
          that is neither a tool result nor an assistant tool-call turn;
        - no render probe and the served tokenizer cannot be loaded to
          build one (the declaration cannot be checked; refusing beats
          honouring it on structure alone);
        - a render in which the header ids occur NOWHERE (a foreign
          template — the adapter would refuse too, but here it is a 400
          before any engine work), or one that ENDS at the header (an
          ordinary render: the tail boundary serves it undeclared, so the
          declaration is claiming a relocation that does not exist).
        """
        raw = request.get(AGENT_LOOP_KEY)
        if raw is not None and not isinstance(raw, bool):
            raise ProxyRefusal(
                f"{AGENT_LOOP_KEY} must be a JSON boolean, got {raw!r} — "
                "the agent-loop declaration is a stated fact, never a "
                "truthy coercion", code="agent_loop_invalid")
        if not raw:
            return False
        if request.get(RESPONSE_PREFILL_KEY):
            raise ProxyRefusal(
                f"request declares BOTH {AGENT_LOOP_KEY} and "
                f"{RESPONSE_PREFILL_KEY} — the two describe disjoint shapes "
                "(an open model turn under a generation prompt vs a "
                "continued final message); state exactly one",
                code="agent_loop_mismatch")
        lens = (cm.parse_serving_identity(route.serving_identity).path
                == "lens")
        if not lens:
            raise ProxyRefusal(
                f"vmodel {route.virtual_model!r} (cell {route.cell_id!r}) "
                f"is not lens-served, but the request declares "
                f"{AGENT_LOOP_KEY} — there is no position mask on this "
                "route, so the declaration would claim a mask coverage that "
                "does not exist; send the turn without the key",
                code="agent_loop_not_lens")
        shape = agent_loop_shape(request)
        if shape is None:
            messages = request.get("messages")
            last = (messages[-1] if isinstance(messages, list) and messages
                    else None)
            raise ProxyRefusal(
                f"{AGENT_LOOP_KEY} declared but the request is not a turn of "
                "the agent loop: it needs more than one non-system message, "
                "a generation-prompt render (no continue_final_message, "
                "add_generation_prompt not false) and a final message that "
                "is a tool result or an assistant tool-call turn — got "
                f"{len(messages) if isinstance(messages, list) else 0} "
                f"message(s), continue_final_message="
                f"{request.get('continue_final_message')!r}, "
                f"add_generation_prompt={request.get('add_generation_prompt')!r}, "
                f"final role={last.get('role') if isinstance(last, Mapping) else None!r}",
                code="agent_loop_mismatch")
        probe = self._agent_loop_probe()
        try:
            ids = tuple(int(t) for t in probe.render(
                request.get("messages"), request.get("tools")))
        except ProxyRefusal:
            raise
        except Exception as e:  # a render this probe cannot mirror
            raise ProxyRefusal(
                f"{AGENT_LOOP_KEY} declared but the request does not render "
                f"under the served template: {type(e).__name__}: {e}",
                code="agent_loop_render_failed") from e
        header = tuple(probe.header_ids)
        H = len(header)
        at = _rfind_ids(ids, header)
        if at < 0:
            raise ProxyRefusal(
                f"{AGENT_LOOP_KEY} declared but the assistant header ids "
                f"{list(header)} occur NOWHERE in the rendered prompt "
                f"({len(ids)} tokens) — the render and the served template "
                "disagree, or the transcript has no model turn; there is no "
                "boundary to locate", code="agent_loop_mismatch")
        if len(ids) >= H and ids[-H:] == header:
            raise ProxyRefusal(
                f"{AGENT_LOOP_KEY} declared ({shape}) but the rendered "
                "prompt ENDS at the assistant header — an ordinary render, "
                "which the tail boundary serves undeclared; the declaration "
                "claims a relocation that does not exist. Drop the key",
                code="agent_loop_mismatch")
        return True

    @staticmethod
    def _is_lens(route: RouteEntry) -> bool:
        return cm.parse_serving_identity(route.serving_identity).path == "lens"

    def _attested_declaration(self, route: RouteEntry) -> Optional[Dict[str, Any]]:
        """The attested serve declaration of ``route``'s cell (by
        ``route.cell_id`` — the harness id in its variant spelling; the
        ``none__lens0`` twin under its own id), ``None`` when the map does not
        name it or nothing was attested."""
        if self.lens_serve_declarations is None:
            return None
        return self.lens_serve_declarations.get(route.cell_id)

    def _served_positions(self, route: RouteEntry) -> Optional[List[str]]:
        """The served position modes the routing table STATES for ``route``'s
        cell (by ``route.cell_id``, the harness id in its variant spelling),
        ``None`` when no map was handed over or the map does not name it."""
        if self.lens_served_positions is None:
            return None
        return self.lens_served_positions.get(route.cell_id)

    def _header_located(self, route: RouteEntry) -> bool:
        """Whether ``route``'s cell serves a HEADER-LOCATED mask — ANY stated
        served mode in :data:`HEADER_LOCATED_MODES` (an OP property: one such
        op makes the cell header-located, a mixed set included) — or whether
        that is UNKNOWN (no map, or a cell the map does not name), which reads
        ``True``: the thinking-ON gate then refuses the undeclared render
        exactly as it did before the served-positions fact existed."""
        modes = self._served_positions(route)
        if modes is None:
            return True
        return any(m in HEADER_LOCATED_MODES for m in modes)

    @staticmethod
    def _validate_kwargs_shape(request: Mapping[str, Any],
                               thinking_header: Optional[Mapping[str, Any]]
                               ) -> Optional[bool]:
        """The ``chat_template_kwargs`` SHAPE validation every route and both
        endpoints apply (§2.5.5): an object whose only key is
        ``enable_thinking`` with a JSON boolean value — anything else refuses
        ``thinking_kwargs_invalid`` — and ``enable_thinking: true`` on a
        template with no thinking switch refuses ``thinking_kwargs_inert`` (a
        run recorded ON that renders OFF). Returns the ``enable_thinking``
        value (``None`` = key absent). Absence is MEMBERSHIP, never
        truthiness (code review X2): a present JSON ``null`` is a stated
        value that is not an object, and refuses like any other."""
        if CHAT_TEMPLATE_KWARGS_KEY not in request:
            return None
        raw = request[CHAT_TEMPLATE_KWARGS_KEY]
        if not isinstance(raw, Mapping):
            raise ProxyRefusal(
                f"{CHAT_TEMPLATE_KWARGS_KEY} must be a JSON object, got {raw!r} — "
                "the render regime is a stated fact; refusing a shape the "
                "served template cannot be held to", code=THINKING_KWARGS_INVALID)
        extra = sorted(set(raw) - {ENABLE_THINKING_KEY})
        if extra:
            raise ProxyRefusal(
                f"{CHAT_TEMPLATE_KWARGS_KEY} carries key(s) {extra} — this proxy "
                f"admits exactly one template kwarg, {ENABLE_THINKING_KEY!r}; any "
                "other would change the render in a way no attestation records; "
                "refusing", code=THINKING_KWARGS_INVALID)
        if ENABLE_THINKING_KEY not in raw:
            return None
        enable = raw[ENABLE_THINKING_KEY]
        if not isinstance(enable, bool):
            raise ProxyRefusal(
                f"{CHAT_TEMPLATE_KWARGS_KEY}.{ENABLE_THINKING_KEY} must be a JSON "
                f"boolean, got {enable!r} — the render regime is a declared fact, "
                "never a truthy coercion", code=THINKING_KWARGS_INVALID)
        if enable and thinking_header is not None and thinking_header["switch"] is False:
            raise ProxyRefusal(
                f"{CHAT_TEMPLATE_KWARGS_KEY}.{ENABLE_THINKING_KEY}: true on a served "
                "template with NO thinking switch — the kwarg is inert (the request "
                "would be recorded as thinking-ON and render thinking-OFF); drop it",
                code=THINKING_KWARGS_INERT)
        return enable

    def _validate_chat_template_kwargs(self, request: Mapping[str, Any],
                                       route: RouteEntry) -> Tuple[str, bool]:
        """THE THINKING-ON GATE:
        validate ``chat_template_kwargs`` and the ``petri_thinking_on``
        declaration against each other, the route and the served template's
        thinking-header fact; return ``(render_regime, inject)`` — the regime
        recorded on every forwarded request's audit row, and whether the
        verified declaration travels as the selector's ``thinking_on`` fact
        (only where the template's switch MOVES the header; validated but
        not injected where the mask is identical).

        Validation is UNIFORM over every cell the proxy cannot read (§2.5.4):
        a lens-route request that sends ``enable_thinking: true`` must carry
        the declaration unless the routing table STATES that none of the
        cell's served ops is header-located (:meth:`_header_located` — its
        served modes, ``lens_served_positions``, all outside
        :data:`HEADER_LOCATED_MODES`). The declaration protects a mask that
        is LOCATED by matching the OFF header at the tail, which the ON
        render moves; on an ``all`` / ``last`` cell the mask is every row /
        the last row by construction, no header and no ``thinking_on`` fact
        is read, so an undeclared ON render there can mis-mask nothing and
        is accepted with the regime recorded (narrowed 2026-09-09). Absent
        the fact — no map (an older table), a cell it does not name — the
        undeclared render refuses as before: that is what still closes the
        undeclared-regime hole without the proxy having to know which
        template moves its header. The DECLARED path is unchanged by the
        narrowing: a declaration on an ``all`` / ``last`` cell keeps refusing
        as vacuous below (``thinking_on_unattested`` /
        ``thinking_on_cell_not_declared``), as the registry refuses it
        (``lens_adapters.attach_thinking_header`` rule 3). The matrix, in
        order (§2.5.6):

        - kwargs not an object / a foreign key / a non-bool value (any
          route) -> ``thinking_kwargs_invalid``; ``enable_thinking: true``
          on a switch-less template -> ``thinking_kwargs_inert``;
        - ``petri_thinking_on`` PRESENT and not a JSON boolean (a present
          ``null`` included; presence is membership, never truthiness — code
          review X2) -> ``thinking_on_invalid``;
        - declared beside ``petri_response_prefill`` / ``petri_agent_loop`` on
          a template NOT certified moves-false (``thinking_header.moves`` True,
          or no / partial fact), or declared without ``enable_thinking: true``
          -> ``thinking_on_mismatch``; on a moves-FALSE template that body is
          validated below like any other lens-route ON request (row 8 narrowed
          to the moves-TRUE template, 2026-09-07);
        - the FIELD present on a non-lens route, whatever its value
          (``false`` included) -> ``thinking_on_not_lens``;
        - ``enable_thinking: true`` on a lens route with no declaration ->
          ACCEPTED, nothing injected, when the table states the cell's served
          mask is not header-located (``all`` / ``last``); else
          ``thinking_on_undeclared`` (a header-located cell, or one whose
          served mask this proxy cannot read — fail closed);
        - ... with no attested ``lens_serve_declarations`` (or no
          ``thinking_header`` fact to verify against) -> ``thinking_on_unattested``;
        - ... on a MOVES-TRUE template whose cell's attested declaration lacks
          ``thinking`` (the ``none__lens0`` twin included, by ``route.cell_id``)
          -> ``thinking_on_cell_not_declared``;
        - ``enable_thinking: false`` / absent -> accepted everywhere."""
        lens = self._is_lens(route)
        enable = self._validate_kwargs_shape(request, self._thinking_header)
        if self.thinking_mode == THINKING_MODE_ON:
            # THE THINKING-ON STANDUP (THINKING_MODES): the launch pinned the ON
            # render for every request, so the regime is the standup's, not the
            # request's — an omitted kwarg RENDERS ON and is read as ON; a request
            # flipping it OFF would be served OFF under an ON attestation (and a
            # steered leg's ON header would be nowhere in its render) — refused
            if enable is False:
                raise ProxyRefusal(
                    f"{CHAT_TEMPLATE_KWARGS_KEY}.{ENABLE_THINKING_KEY}: false on a thinking-ON "
                    "standup — the launch pinned the ON render for every request and the "
                    "standup is attested thinking-ON; a request flipping the regime would be "
                    "served OFF under an ON attestation (and a steered leg's ON header would "
                    "be nowhere in its render); drop the key (the default IS thinking-ON) or "
                    "send it to a thinking-OFF standup", code=THINKING_REGIME_MISMATCH)
            enable = True
        regime = (RENDER_REGIME_UNSPECIFIED if enable is None
                  else RENDER_REGIME_THINKING_ON if enable
                  else RENDER_REGIME_THINKING_OFF)
        present = THINKING_ON_KEY in request
        raw_decl = request.get(THINKING_ON_KEY)
        if present and not isinstance(raw_decl, bool):
            raise ProxyRefusal(
                f"{THINKING_ON_KEY} must be a JSON boolean, got {raw_decl!r} — the "
                "thinking-ON declaration is a stated fact, never a truthy coercion",
                code=THINKING_ON_INVALID)
        declared = bool(raw_decl)
        if declared:
            # §2.5.6 row 8, NARROWED to the moves-TRUE template (2026-09-07):
            # where the switch MOVES the header, a seeded render (header at
            # its LAST occurrence) or an agent-loop render (its own locator)
            # under the ON header is a combination nobody derived; where the
            # template is certified NOT to move it (moves False) the
            # declaration injects nothing, so the body is today's OFF shape
            # plus what the ON mode adds outside the header — it validates
            # below like any other lens-route ON request. The exemption needs
            # the POSITIVE certification: no fact, no switch, or a partial
            # fact (moves None) all keep refusing here, as they did.
            moves = (self._thinking_header["moves"]
                     if self._thinking_header is not None else None)
            if ((request.get(RESPONSE_PREFILL_KEY) or request.get(AGENT_LOOP_KEY))
                    and moves is not False):
                raise ProxyRefusal(
                    f"request declares {THINKING_ON_KEY} beside {RESPONSE_PREFILL_KEY} / "
                    f"{AGENT_LOOP_KEY} on a template "
                    + ("whose thinking switch MOVES the assistant header" if moves is True
                       else "NOT certified to keep its assistant header in place under "
                            "the thinking switch")
                    + " — a declared thinking-ON render is an ordinary tail render under "
                    "the ON header; a seeded or agent-loop ON render there is not a shape "
                    "this path serves; state exactly one",
                    code=THINKING_ON_MISMATCH)
            if enable is not True:
                raise ProxyRefusal(
                    f"{THINKING_ON_KEY} declared but {CHAT_TEMPLATE_KWARGS_KEY}."
                    f"{ENABLE_THINKING_KEY} is not true ({enable!r}) — the declaration "
                    "must match the render it declares", code=THINKING_ON_MISMATCH)
            if not lens:
                raise ProxyRefusal(
                    f"vmodel {route.virtual_model!r} (cell {route.cell_id!r}) is not "
                    f"lens-served, but the request declares {THINKING_ON_KEY} — there is "
                    "no position mask on this route whose boundary the declaration could "
                    "relocate; send the thinking-ON render without the key (the render "
                    "regime is still recorded)", code=THINKING_ON_NOT_LENS)
        elif present and not lens:
            # the FIELD is forbidden on a non-lens route whatever its value
            # (§2.5.5): a `false` there is not a harmless no-op but a key
            # this route never admits — refuse rather than strip it silently
            raise ProxyRefusal(
                f"vmodel {route.virtual_model!r} (cell {route.cell_id!r}) is not "
                f"lens-served, but the request carries {THINKING_ON_KEY} ({raw_decl!r}) — "
                "the key is admitted on lens routes only; drop it (the render regime "
                "is still recorded)", code=THINKING_ON_NOT_LENS)
        if not (lens and enable is True):
            return regime, False
        if not declared and self.thinking_mode == THINKING_MODE_ON:
            # a thinking-ON STANDUP: a render regime the fit never saw must be
            # declared on EVERY lens route, whether or not this template's header
            # moves under it and whatever the cell's served positions (the
            # standup-wide rule; the per-request narrowing below is the OFF
            # proxy's) — the drivers of such a standup declare on every lens leg
            raise ProxyRefusal(
                f"vmodel {route.virtual_model!r} (cell {route.cell_id!r}) is lens-served "
                "on a thinking-ON standup and the request does not declare "
                f"{THINKING_ON_KEY}: true — a render regime the fit never saw must be "
                "declared on every lens route, whether or not this template's header "
                f"moves under it; add `{THINKING_ON_KEY}: true`",
                code=THINKING_ON_UNDECLARED)
        if not declared:
            # §2.5.4 NARROWED (2026-09-09): the declaration protects a
            # HEADER-LOCATED mask (the OFF header matched at the tail moves
            # under the ON render — the worker's REASON_NOT_AT_TAIL). Where
            # the routing table STATES that no served op of this cell is
            # header-located (`all` / `last`: every row / the last row by
            # construction, no header and no thinking_on fact read), an
            # undeclared ON render can mis-mask nothing: accepted, nothing
            # injected, the regime recorded from the kwargs like any other
            # forwarded request. UNKNOWN (no map, an unnamed cell) refuses.
            if not self._header_located(route):
                return regime, False
            modes = self._served_positions(route)
            raise ProxyRefusal(
                f"vmodel {route.virtual_model!r} (cell {route.cell_id!r}) is lens-served "
                f"and the request renders thinking-ON ({CHAT_TEMPLATE_KWARGS_KEY}."
                f"{ENABLE_THINKING_KEY}: true) without declaring {THINKING_ON_KEY} — "
                + (f"its served mask is HEADER-LOCATED (served positions {modes}): the "
                   "OFF header the mask is located against moves under the ON render, "
                   "so a render regime the fit never saw must be declared on this route"
                   if modes is not None else
                   f"this proxy holds no {LENS_SERVED_POSITIONS_KEY} fact for the cell ("
                   + ("the routing table records no map — exported before the "
                      "served-positions landing" if self.lens_served_positions is None
                      else "the map does not name it")
                   + "), so whether its served mask is header-located cannot be "
                   "verified; refusing (fail closed)")
                + f"; add `{THINKING_ON_KEY}: true` (fit-support-transfer plan §2.5.4)",
                code=THINKING_ON_UNDECLARED)
        if self.lens_serve_declarations is None:
            raise ProxyRefusal(
                f"{THINKING_ON_KEY} declared on vmodel {route.virtual_model!r} but this "
                f"proxy holds NO attested {SERVE_DECLARATIONS_KEY} map"
                + (f" ({self.lens_serve_declarations_error})"
                   if self.lens_serve_declarations_error else "")
                + " — nothing vouches for which cells may serve the declared render; "
                "refusing (fail closed)", code=THINKING_ON_UNATTESTED)
        if self._thinking_header is None:
            raise ProxyRefusal(
                f"{THINKING_ON_KEY} declared on vmodel {route.virtual_model!r} but this "
                f"proxy's routing table records no {THINKING_HEADER_KEY} fact — whether "
                "the served template's header moves under the declared render cannot "
                "be verified (a table exported before the fit-support-transfer landing); "
                "refusing (fail closed)", code=THINKING_ON_UNATTESTED)
        if self._thinking_header["on_ids"] is None:
            # the template has a switch but its ON header certification is
            # UNAVAILABLE (code review X1): whether the header moves is
            # UNKNOWN, so neither the moves-false acceptance below nor the
            # per-cell check applies — fail closed, naming the reason
            raise ProxyRefusal(
                f"{THINKING_ON_KEY} declared on vmodel {route.virtual_model!r} but the served "
                "template's thinking-ON header certification is UNAVAILABLE"
                + (f" ({self.thinking_header_error})" if self.thinking_header_error else "")
                + " — the standup recorded no on_ids, so whether the switch moves the "
                "header cannot be verified; refusing (fail closed)",
                code=THINKING_ON_UNATTESTED)
        if self._thinking_header["moves"] is not True:
            # the switch does not move the header (gemma-4): the declaration
            # is validated and the regime recorded; the mask is identical, so
            # nothing is injected (§2.5.4)
            return regime, False
        decl = self._attested_declaration(route) or {}
        if (self.thinking_mode == THINKING_MODE_ON
                and sd.OUT_OF_SUPPORT_THINKING not in decl.get(sd.SERVE_OUT_OF_SUPPORT_KEY, [])):
            # a thinking-ON STANDUP declares exactly the cells whose served ops are
            # header-located and its drivers declare on every lens leg, so an
            # undeclared lens cell here is an all-position fit whose mask does not
            # move: validated, recorded, nothing injected (the OFF proxy's per-cell
            # refusal below is for a request declaring what no standup exported)
            return regime, False
        if sd.OUT_OF_SUPPORT_THINKING not in decl.get(sd.SERVE_OUT_OF_SUPPORT_KEY, []):
            raise ProxyRefusal(
                f"{THINKING_ON_KEY} declared on vmodel {route.virtual_model!r} (cell "
                f"{route.cell_id!r}) but the cell's attested serve declaration "
                f"({decl or 'none'}) does not admit the thinking-ON render — on this "
                "template the thinking switch MOVES the assistant header, so only a "
                f"cell declaring `{sd.SERVE_OUT_OF_SUPPORT_KEY}: "
                f"[\"{sd.OUT_OF_SUPPORT_THINKING}\"]` (spelled `<cell>@thinking`) "
                "carries the ON header its boundary is located against; refusing",
                code=THINKING_ON_CELL_NOT_DECLARED)
        return regime, True

    def _all_turns_route(self, route: RouteEntry) -> bool:
        """Whether ``route`` serves the all-turns mask (setting (2)): its
        cell id names the ``@all_turns`` variant — the NAME and the hashed
        spec agree by construction (WS-0's enumeration refuses a variant id
        whose spec does not express it, and a declaration the name does not
        carry), so the suffix is the served identity's own spelling, not an
        inference — OR its attested declaration names
        ``fit_response_serve_all_turns`` (belt and braces: the map covers a
        twin whose donor were an all-turns cell). Either trigger runs the
        check; nothing here decides what is SERVED."""
        try:
            _base, variant = sd.split_variant(route.cell_id)
        except sd.ServeDeclarationError:
            variant = None
        if variant == sd.VARIANT_ALL_TURNS:
            return True
        decl = self._attested_declaration(route) or {}
        return decl.get(sd.SERVE_EXPERIMENT_KEY) == sd.EXPERIMENT_FIT_RESPONSE_SERVE_ALL_TURNS

    def _record_all_turns_check(self, route: RouteEntry, label: str, *,
                                code: Optional[str], detail: str) -> None:
        self.n_all_turns_check[label] = self.n_all_turns_check.get(label, 0) + 1
        if label != ALL_TURNS_CHECK_OK:
            self.all_turns_refusals.append(
                {"virtual_model": route.virtual_model, "cell_id": route.cell_id,
                 LEDGER_KEY_ALL_TURNS_CHECK: label, "code": code, "detail": detail[:2000]})
        log.info("petri-proxy all-turns render check: virtual_model=%s cell=%s "
                 "all_turns_check=%s code=%s", route.virtual_model, route.cell_id,
                 label, code,
                 extra={"virtual_model": route.virtual_model, "cell_id": route.cell_id,
                        LEDGER_KEY_ALL_TURNS_CHECK: label, "code": code})

    def _validate_all_turns_render(self, request: Mapping[str, Any],
                                   route: RouteEntry) -> Optional[str]:
        """THE PER-ITEM RENDER CHECK for setting (2), a fail-closed
        pre-check run iff the route is lens and serves
        ``positions="response_all_turns"``:

        1. ``ids = probe.render(messages, tools)`` — an unrenderable request
           -> 400 ``all_turns_render_failed``;
        2. ``final = span_locator.locate_final_header(ids, header, declared=False)``
           and 3. ``located = span_locator.assistant_turn_spans(ids, header,
           turn_end, final)`` — the WORKER's own arithmetic on the same ids;
           a ``SpanLocatorError`` -> 400 ``all_turns_render_refused``;
        4. ``expected`` — per assistant message ``(lo - H, hi + K)`` from
           ``chat_markers.assistant_content_spans`` (the message-derived
           content span widened to header + content + end scaffold — the
           all-turns mask's own definition) plus ``(final, len(ids))``; a
           ``ChatMarkerError`` (control tokens inside content — the docs-§7
           blind spot, closed here for every request the proxy sees; a shape
           the helper does not describe: tools, a seeded final message) ->
           400 ``all_turns_expected_unavailable``;
        5. equal -> forwarded (label ``ok``); else 400 ``all_turns_span_mismatch``
           naming both span lists.

        Every outcome is counted (``n_all_turns_check``) and logged; a
        refusal is also kept in ``all_turns_refusals``. Returns the label for
        the forwarded request's audit row, ``None`` for a route the check does
        not apply to. The mode's EXPERIMENTAL / UNSUPPORTED status is
        unchanged — this is the experiment's per-item instrument."""
        if not self._is_lens(route) or not self._all_turns_route(route):
            return None
        where = f"vmodel {route.virtual_model!r} (cell {route.cell_id!r}): "

        def _refuse(code, msg):
            self._record_all_turns_check(route, _ALL_TURNS_CODE_LABEL[code],
                                         code=code, detail=msg)
            return ProxyRefusal(where + msg, code=code)

        probe = self._agent_loop_probe()
        content_spans = getattr(probe, "assistant_content_spans", None)
        if not callable(content_spans) or not hasattr(probe, "turn_end_ids"):
            raise _refuse(ALL_TURNS_RENDER_FAILED,
                          "this proxy's render probe cannot derive the served "
                          "template's turn-end scaffold or message-derived spans "
                          "(not a TokenizerAgentLoopRender) — the all-turns render "
                          "check cannot run; refusing rather than forwarding an "
                          "unchecked all-turns request")
        messages = request.get("messages")
        try:
            ids = tuple(int(t) for t in probe.render(messages, request.get("tools")))
        except ProxyRefusal as e:
            raise _refuse(ALL_TURNS_RENDER_FAILED,
                          f"the request does not render under the served template: {e}")
        except Exception as e:  # noqa: BLE001 — a render the probe cannot mirror
            raise _refuse(ALL_TURNS_RENDER_FAILED,
                          "the request does not render under the served template: "
                          f"{type(e).__name__}: {e}")
        header = tuple(probe.header_ids)
        try:
            turn_end = probe.turn_end_ids
        except Exception as e:  # noqa: BLE001 — the template has no certifiable scaffold
            raise _refuse(ALL_TURNS_RENDER_REFUSED,
                          "the served template's turn-end scaffold cannot be derived "
                          f"({type(e).__name__}: {e}); the all-turns mask cannot be "
                          "checked on it")
        if not turn_end:
            raise _refuse(ALL_TURNS_RENDER_REFUSED,
                          "the served template is successor-bounded (no turn "
                          "terminator) — an assistant turn's end cannot be located, "
                          "so the all-turns mask cannot be served or checked on it")
        try:
            final = sl.locate_final_header(ids, header, declared=False)
            located = sl.assistant_turn_spans(ids, header, turn_end, final)
        except sl.SpanLocatorError as e:
            raise _refuse(ALL_TURNS_RENDER_REFUSED,
                          f"the all-turns span locator refuses this render: {e}")
        if request.get("tools") or request.get("continue_final_message") \
                or request.get("add_generation_prompt") is False:
            raise _refuse(ALL_TURNS_EXPECTED_UNAVAILABLE,
                          "the message-derived expectation is defined for a plain "
                          "generation-prompt render of role/content messages only — "
                          "tools, a continued final message or add_generation_prompt: "
                          "false are shapes it does not describe; refusing rather than "
                          "guessing the expected spans")
        try:
            full, content = content_spans(messages)
        except ProxyRefusal as e:
            raise _refuse(ALL_TURNS_EXPECTED_UNAVAILABLE,
                          f"the message-derived spans are unavailable: {e}")
        except Exception as e:  # ChatMarkerError and anything the helper refuses
            raise _refuse(ALL_TURNS_EXPECTED_UNAVAILABLE,
                          "the message-derived spans are unavailable: "
                          f"{type(e).__name__}: {e}")
        if tuple(full) != ids[:len(full)]:
            raise _refuse(ALL_TURNS_SPAN_MISMATCH,
                          "the full-conversation render is not a prefix of the "
                          "generation-prompt render the request will be served under "
                          f"({len(full)} vs {len(ids)} tokens) — the two renders "
                          "disagree; refusing")
        expected = sl.expected_all_turns_spans(
            content, header_len=len(header), turn_end_len=len(turn_end),
            final_start=final, prompt_len=len(ids))
        if tuple(located) != tuple(expected):
            raise _refuse(ALL_TURNS_SPAN_MISMATCH,
                          "the located all-turns spans disagree with the message-derived "
                          f"expectation: located {list(located)} vs expected "
                          f"{list(expected)} — the mask would steer positions the "
                          "messages do not describe; refusing")
        self._record_all_turns_check(route, ALL_TURNS_CHECK_OK, code=None, detail="")
        return ALL_TURNS_CHECK_OK

    def _entry_declare(self, request: Mapping[str, Any],
                       route: RouteEntry) -> Mapping[str, Any]:
        """THE PETRI-FACING ENTRY, applied (:func:`declare_agent_loop`):
        Petri's own client cannot declare, so the proxy declares for
        agent-loop-structured requests on LENS routes whose render does not
        end at the header, at the point its requests enter —
        :meth:`_validate_agent_loop` then verifies that declaration against
        the route and the render exactly as it would a client's. Renders
        only what it must: nothing unless the entry is armed, the route is
        lens-served, the key is absent and the structure matches — so
        single-turn drivers never touch the probe. A turn this proxy cannot
        render (no loadable tokenizer; an unrenderable transcript) is a
        loud 400, never a silent forward."""
        if not self._petri_agent_loop_entry or AGENT_LOOP_KEY in request:
            return request
        lens = (cm.parse_serving_identity(route.serving_identity).path
                == "lens")
        if not lens or agent_loop_shape(request) is None:
            return request
        try:
            return declare_agent_loop(request, lens_route=True,
                                      probe=self._agent_loop_probe())
        except ProxyRefusal:
            raise
        except Exception as e:  # a render the probe cannot mirror
            raise ProxyRefusal(
                f"vmodel {route.virtual_model!r} (cell {route.cell_id!r}): "
                "the Petri-facing entry could not render this agent-loop "
                "turn to decide whether it needs the "
                f"{AGENT_LOOP_KEY} declaration: {type(e).__name__}: {e}",
                code="agent_loop_render_failed") from e

    def _agent_loop_probe(self) -> AgentLoopRender:
        """The agent-loop verifier's render probe. EXPLICIT when the
        launcher supplied one (the production sbatch passes the
        manifest-pinned served tokenizer under the pinned chat template —
        eager, so a misconfiguration fails at startup, never at the
        audit's first tool turn); else built ONCE, lazily, from the
        routing table's ``base_model`` (+ ``base_revision`` when the table
        records it) on the first agent-loop turn — so the many proxies
        that never see such a turn (single-turn jailbreak / BoN /
        capability drivers) never pay a tokenizer load, while a Petri
        proxy started without an explicit probe still serves. A load
        failure is a per-request 400 (``agent_loop_unverifiable``), never
        a silently honoured declaration."""
        if self._agent_loop_render is None:
            pin = (f"{self.base_model}@{self._base_revision}"
                   if self._base_revision else f"{self.base_model} (unpinned)")
            try:
                # local_files_only: a serving proxy never downloads at
                # request time — the served model is in the same cache
                # the engine loaded it from, or the pin is wrong (loud)
                tok = load_served_tokenizer(self.base_model,
                                            revision=self._base_revision,
                                            local_files_only=True)
                self._agent_loop_render = TokenizerAgentLoopRender(tok)
            except Exception as e:
                raise ProxyRefusal(
                    f"{AGENT_LOOP_KEY} declared but this proxy has no render "
                    f"probe and could not load the served tokenizer {pin} "
                    f"to build one ({type(e).__name__}: {e}) — the "
                    "declaration cannot be verified against the rendered "
                    "prompt; refusing rather than honouring it on structure "
                    "alone. Start the proxy with the served tokenizer "
                    "(agent_loop_render / --tokenizer)",
                    code="agent_loop_unverifiable") from e
            log.info("petri-proxy: agent-loop render probe built lazily "
                     "from %s", pin)
        return self._agent_loop_render

    @staticmethod
    def salt_regime_for(*, lens: bool,
                        resumed_prefill_policy: Optional[str] = None,
                        position_mode: Optional[str] = None) -> str:
        """WHICH KV namespace one leg gets (module docstring), from the three
        facts that decide it: whether the leg carries a steering selector,
        — when it does — the cell's ATTESTED resumed-prefill policy, and —
        when it does not — the native residual slot's ATTESTED position mode.

        - not a lens leg at no native position mode, or at ``all`` →
          :data:`SALT_REGIME_CELL_STABLE` (unchanged since the isolation
          landed: unhooked and LoRA legs have no hooks to miss, and an
          all-positions residual write computes every row's edit from that
          row alone, so within-cell prefix reuse is free and cross-cell reuse
          is what the salt prevents). The policy is not consulted.
        - not a lens leg at a RESTRICTED native position mode (``response``
          / ``prompt_last`` / ``prefill_all``) →
          :data:`SALT_REGIME_REQUEST_UNIQUE`: the lens's rule for a
          ``conditional`` / ``never`` mask, transported — KV computed under
          a masked write at generated rows may not be resumed by a later
          request in which those same tokens are an unsteered earlier turn.
        - a lens leg whose cell is attested :data:`RESUMED_PREFILL_ALWAYS`
          → :data:`SALT_REGIME_CELL_STABLE`. Every op of the cell computes
          each row's edit from that row alone at every position in both
          phases, so a prefix cached under the same (cell, intervention)
          carries exactly the edit a cold prefill would have given it, and
          the worker serves the resume as a RESUMED PREFILL.
        - EVERY other lens leg → :data:`SALT_REGIME_REQUEST_UNIQUE`.
          ``conditional`` and ``never`` alike, and every non-policy in
          :data:`POLICY_NON_VALUES`: an unattested proxy shares nothing, a
          cell the map does not name shares nothing. Fail closed."""
        if not lens:
            if position_mode is not None and position_mode != NATIVE_POSITION_MODE_ALL:
                return SALT_REGIME_REQUEST_UNIQUE
            return SALT_REGIME_CELL_STABLE
        if resumed_prefill_policy == RESUMED_PREFILL_ALWAYS:
            return SALT_REGIME_CELL_STABLE
        return SALT_REGIME_REQUEST_UNIQUE

    @staticmethod
    def cache_salt_for(route: RouteEntry, *, lens: bool,
                       resumed_prefill_policy: Optional[str] = None,
                       organism: Optional[str] = None,
                       position_mode: Optional[str] = None) -> str:
        """The proxy-owned KV-cache namespace for one request (module
        docstring), in the regime :meth:`salt_regime_for` decides:

        - CELL-STABLE, unhooked/LoRA/all-positions residual:
          ``petri|<cell>|<intervention hash>`` — within-cell reuse,
          cross-cell isolation.
        - REQUEST-UNIQUE, a restricted-mode native route: the stable
          part plus ``|<mode>`` and a uuid4 — a namespace of one, the mode
          readable off the salt in a log.
        - CELL-STABLE, lens (an attested ``always`` cell): the same plus
          ``|lens``. The marker is not decoration: it keeps the steered
          namespace of a (cell, intervention) UNEQUAL to the unhooked one
          by construction, so the one collision that would silently serve
          steered KV under an unsteered identity cannot be minted at all
          — stronger than catching it in :meth:`_guard_cache_salt` after
          the fact.
        - REQUEST-UNIQUE, lens (everything else): the stable part plus a
          uuid4 — a namespace of one, so the prefill never resumes.

        The intervention hash rides in the stable part so two registries'
        same-named cells never share a namespace either.
        ``resumed_prefill_policy`` defaults to ``None``, which is the
        pre-landing behaviour exactly: every lens leg request-unique."""
        stable = f"petri|{route.cell_id}|{route.intervention_hash}"
        if organism is not None:
            stable = f"{stable}|org={organism}"
        if not lens:
            if RoutingProxy.salt_regime_for(
                    lens=False, position_mode=position_mode) == SALT_REGIME_REQUEST_UNIQUE:
                return f"{stable}|{position_mode}|{uuid.uuid4().hex}"
            return stable
        if RoutingProxy.salt_regime_for(
                lens=True,
                resumed_prefill_policy=resumed_prefill_policy) == SALT_REGIME_CELL_STABLE:
            return f"{stable}|{LENS_STABLE_SALT_MARKER}"
        return f"{stable}|{uuid.uuid4().hex}"

    def resumed_prefill_policy_label(self, route: RouteEntry, *,
                                     lens: bool) -> str:
        """The ``resumed_prefill_policy`` this proxy READ for one leg — the
        attested policy string, or the :data:`POLICY_NON_VALUES` member
        naming why none applied. Recorded verbatim on the request's audit
        row, so a reader can tell an attested ``never`` (the workers said
        so) from ``unattested`` (nobody told this proxy anything) from
        ``absent`` (a map that does not name this cell) — three different
        worlds that all fail closed to the same salt."""
        if not lens:
            return POLICY_NOT_CONSULTED
        if self.resumed_prefill_policy_cells is None:
            return POLICY_UNATTESTED
        return self.resumed_prefill_policy_cells.get(route.cell_id,
                                                     POLICY_CELL_ABSENT)

    def _salt_decision(self, route: RouteEntry, *, lens: bool,
                       organism: Optional[str] = None) -> SaltDecision:
        """ONE request's :class:`SaltDecision` — minted once and carried, so
        the salt that goes upstream, the regime the guard applies and the
        label the audit row records are the same decision (deriving a
        request-unique salt twice would produce two different salts)."""
        label = self.resumed_prefill_policy_label(route, lens=lens)
        policy = label if label in RESUMED_PREFILL_POLICIES else None
        # the NATIVE position mode: read off the attested adapter entry of a
        # lora route (None elsewhere); a restricted mode mints request-unique
        position_mode = None if lens else self.native_position_mode(route)
        if lens and self.lens_salt_policy == LENS_SALT_POLICY_REQUEST_UNIQUE_ALL:
            # the whole regime pinned by declaration: every
            # lens leg request-unique, the attested map unconsulted
            label, policy = POLICY_FORCED_REQUEST_UNIQUE, None
        elif lens and self._attested_declaration(route) is not None:
            # DECLARED CELLS SHARE NOTHING: a cell served
            # out of its fit support is minted request-unique whatever its
            # attested resumed-prefill policy says
            label, policy = POLICY_DECLARED_SHARES_NOTHING, None
        return SaltDecision(
            salt=self.cache_salt_for(route, lens=lens,
                                     resumed_prefill_policy=policy, organism=organism,
                                     position_mode=position_mode),
            regime=self.salt_regime_for(lens=lens,
                                        resumed_prefill_policy=policy,
                                        position_mode=position_mode),
            policy=label, lens=lens, position_mode=position_mode)

    def _guard_cache_salt(self, route: RouteEntry, *, salt: str,
                          lens: bool, regime: str) -> None:
        """The PROXY-LEVEL salt-discipline guard (module docstring): the
        engine-side guard on the same discipline is FATAL — a steered
        request resuming cached rows it may not raises inside EngineCore's
        generate path and takes the whole vLLM server down (job 227309) —
        so the proxy, which mints every salt and knows which routes carry a
        selector, verifies the discipline FIRST and refuses with a
        per-request 409 the engine never sees.

        Since the resumed-prefill landing the discipline is no longer "no
        steered salt is ever reused": a lens leg on a cell attested
        :data:`RESUMED_PREFILL_ALWAYS` rides a CELL-STABLE salt and reusing
        it IS the feature — that request resumes a prefix its own cell and
        intervention wrote, position by position, and the worker serves and
        attests it as a RESUMED PREFILL. So the guard now refuses exactly
        the three reuses that remain wrong, and permits the one that is
        right:

        - a steered request on a REQUEST-UNIQUE salt that was used before
          — impossible by construction (uuid4), so a hit is a construction
          bug, a duplicate-cell routing table or a uuid collision, and it
          is the engine-fatal case;
        - a steered request on ANOTHER cell's namespace — the resumed rows
          carry a different intervention (docs §4: for resumable cells the
          salt namespace is the whole isolation, since the worker accepts
          any resume its policy admits);
        - a steered request on a salt an UNSTEERED request used — it would
          resume rows the hooks never touched, again the engine-fatal case;
        - an UNSTEERED request on a salt a STEERED request used — it would
          resume the steered KV under its own identity, the SILENT
          wrong-serve (no engine guard even fires there: hooks are not
          active on a selector-less request).

        The permitted repeat is narrow on purpose: same salt, both steered,
        both CELL-STABLE, same cell. Everything else is a 409. NOT a
        weakening of isolation on the paths that still hold it: the refusal
        happens STRICTLY BEFORE the engine's own (kept) backstop could, and
        nothing is served in either direction. What DID change is stated in
        docs §4 and repeated here: for an ``always`` cell the worker can no
        longer detect a same-salt resume written by a different cell, so
        this namespace is the whole isolation for it — which is why a
        ``never`` cell is never given a shared namespace at all rather than
        being handed to the worker's fatal refusal.

        Check and record are synchronous (no await), so concurrent handlers
        cannot interleave between them; the salt is recorded before the
        upstream call so a request that dies upstream still holds its claim
        (its blocks may have been committed)."""
        if regime not in SALT_REGIMES:
            raise ProxyError(f"unknown cache-salt regime {regime!r} "
                             f"(expected one of {SALT_REGIMES})")
        claim = self._salt_guard.get(salt)
        where = (f"cell {route.cell_id!r} (vmodel {route.virtual_model!r}): ")
        if lens and claim is not None:
            if not claim.lens:
                raise ProxyRefusal(
                    where + "STEERED request would reuse a cache_salt an "
                    "UNSTEERED request already sent upstream — its prefill "
                    "would resume blocks its hooks never touched, and the "
                    "engine-side guard treats that as fatal (EngineCore "
                    "dies, the server with it); refused at the proxy "
                    "instead, isolation intact and the server still up "
                    "(D2)", http_status=409, code="cache_salt_collision")
            if claim.cell_id != route.cell_id:
                raise ProxyRefusal(
                    where + "STEERED request would reuse the cache_salt of "
                    f"cell {claim.cell_id!r} — resuming another cell's "
                    "blocks serves that cell's intervention under this "
                    "one's identity, and for a resumable cell the salt "
                    "namespace is the whole isolation; refused at the "
                    "proxy (D2)", http_status=409,
                    code="cache_salt_collision")
            if not (regime == SALT_REGIME_CELL_STABLE
                    and claim.regime == SALT_REGIME_CELL_STABLE):
                raise ProxyRefusal(
                    where + "STEERED request would reuse a "
                    f"{SALT_REGIME_REQUEST_UNIQUE} cache_salt already sent "
                    f"upstream (this request {regime}, the claim "
                    f"{claim.regime}) — a request-unique salt is a "
                    "namespace of one by construction, so a repeat is a "
                    "minting bug and the resume it would allow is the "
                    "engine-fatal case; refused at the proxy (D2)",
                    http_status=409, code="cache_salt_collision")
            return                      # the resumable cell's legitimate repeat
        if not lens and claim is not None and claim.lens:
            raise ProxyRefusal(
                where + "UNSTEERED request would reuse a cache_salt a "
                "STEERED request already used — resuming those blocks "
                "would silently inherit the steered KV under this cell's "
                "identity (the wrong-serve prefix-cache isolation exists "
                "to prevent); refused at the proxy (D2)",
                http_status=409, code="cache_salt_collision")
        if not lens and claim is not None and regime == SALT_REGIME_REQUEST_UNIQUE:
            # a RESTRICTED-MODE native route is minted request-unique: a
            # repeat is a minting bug (uuid4 collision, a duplicate-cell table),
            # and the resume it would allow serves KV written under a masked
            # write to rows the mask would not cover — refused, never resumed
            raise ProxyRefusal(
                where + f"a {SALT_REGIME_REQUEST_UNIQUE} cache_salt (restricted-mode "
                "native route) was already sent upstream — a request-unique salt is a "
                "namespace of one by construction, so a repeat is a minting bug and "
                "the resume it would allow is the masked-write wrong-serve; refused at "
                "the proxy (D12)", http_status=409, code="cache_salt_collision")
        if claim is None:
            if len(self._salt_guard_order) >= SALT_GUARD_MAX:
                evicted = self._salt_guard_order.popleft()
                self._salt_guard.pop(evicted, None)
                self.n_salt_guard_evicted += 1
            self._salt_guard_order.append(salt)
            self._salt_guard[salt] = _SaltClaim(
                lens=lens, cell_id=route.cell_id, regime=regime)

    async def _refuse_poisoned_lens_response(
            self, route: RouteEntry, response: Mapping[str, Any]) -> None:
        """Claim this response's worker-contained steering failure, if
        any, and refuse the response when one comes back. The chat
        completion's ``id`` IS the engine request id at the pinned vLLM
        (multi-prompt sub-ids and parallel-sampling child ids are
        derived from it; the worker's claim matching covers both), so a
        response with no id cannot be verified — refused, never
        forwarded on faith."""
        rid = response.get("id")
        if not isinstance(rid, str) or not rid:
            raise ProxyRefusal(
                f"vmodel {route.virtual_model!r} (cell {route.cell_id!r}): "
                "the upstream lens response carries no 'id', so its "
                "per-request steering failure record cannot be claimed — "
                "refusing the response rather than forwarding an "
                "unverifiable completion",
                http_status=502, code="lens_postcheck_failed")
        take = getattr(self._backend, "take_request_failure", None)
        if not callable(take):
            raise ProxyRefusal(
                f"vmodel {route.virtual_model!r} (cell {route.cell_id!r}): "
                "this backend exposes no take_request_failure post-check, "
                "so lens responses cannot be verified against the "
                "worker's per-request failure ledger — refusing to serve "
                "a lens route through it",
                http_status=502, code="lens_postcheck_failed")
        failure = await take(rid)
        if failure is not None:
            raise ProxyRefusal(
                f"request {rid!r} (vmodel {route.virtual_model!r}, cell "
                f"{failure.get('cell_id', route.cell_id)!r}) FAILED its "
                "steering hook and was served unsteered — its completion "
                "is discarded, never returned. Worker diagnosis: "
                f"{failure.get('error_type', 'AdapterError')}: "
                f"{failure.get('error', '<no detail recorded>')}",
                http_status=500, code="lens_request_failed")

    # -- restricted-mode native routes ----

    def native_position_mode(self, route: RouteEntry) -> Optional[str]:
        """The NATIVE position mode the ENGINE ATTESTATION records for a lora
        route's adapter (``all`` / ``response`` / ``prompt_last`` /
        ``prefill_all``), ``None`` for a plain LoRA, for any non-lora route, and
        on a proxy started with no engine attestation (a lens standup; a
        version-2 table admitted without its attestation — production always
        hands one over, ``main`` requires it for a version-2 table)."""
        if self.engine_attestation is None or _route_path(route) != "lora":
            return None
        entry = self.engine_attestation["adapters"].get(route.cell_id)
        return None if entry is None else entry["position_mode"]

    def restricted_position_mode(self, route: RouteEntry) -> Optional[str]:
        """:meth:`native_position_mode` when it is RESTRICTED (not ``None``, not
        ``all``) — the predicate every restricted-route rule keys on; ``None``
        for an all-positions residual route and every LoRA / unhooked route."""
        mode = self.native_position_mode(route)
        return None if mode in (None, NATIVE_POSITION_MODE_ALL) else mode

    def _validate_restricted_route_request(self, request: Mapping[str, Any],
                                           route: RouteEntry, *, position_mode: str) -> None:
        """THE REQUEST ALLOWLIST of a restricted-mode native route: every
        body key must be one of :data:`RESTRICTED_ROUTE_ALLOWED_KEYS` or a
        :data:`RESTRICTED_ROUTE_GOVERNED_KEYS` member whose content the proxy
        governs (``chat_template_kwargs`` carrying ``enable_thinking`` alone);
        ``continue_final_message`` / ``add_generation_prompt`` only in the
        SEEDED shape (final message assistant, ``continue_final_message: true``,
        ``add_generation_prompt: false`` — JSON booleans, never truthy
        coercions). Anything else — ``truncate_prompt_tokens``,
        ``truncation_side``, ``chat_template``, ``prompt_embeds``,
        ``documents``, ``tools``, ``echo``, the proxy's own lens-only
        declaration keys, an ``add_generation_prompt: false`` without an
        assistant seed — refuses :data:`RESTRICTED_ROUTE_OPTION_UNSUPPORTED`
        (400) BEFORE the body is forwarded: the worker locates the response
        boundary from the prompt ids the render produced, and a render-changing
        option can remove the header it locates against. An allowlist, never a
        denylist. All-positions residual routes and LoRA routes never reach
        this method."""
        where = (f"vmodel {route.virtual_model!r} (cell {route.cell_id!r}, native "
                 f"position mode {position_mode!r}): ")
        allowed = (set(RESTRICTED_ROUTE_ALLOWED_KEYS) | set(RESTRICTED_ROUTE_GOVERNED_KEYS)
                   | set(RESTRICTED_ROUTE_CONTROL_KEYS))
        extra = sorted(k for k in request if k not in allowed)
        if extra:
            raise ProxyRefusal(
                where + f"request key(s) {extra} are not on the restricted-route "
                f"allowlist {list(RESTRICTED_ROUTE_ALLOWED_KEYS)} (+ "
                f"{list(RESTRICTED_ROUTE_GOVERNED_KEYS)} as the proxy governs it) — a "
                "position-restricted residual write locates its boundary from the "
                "rendered prompt ids, and an option outside the list may change the "
                "render (truncate_prompt_tokens + truncation_side: right removes the "
                "generation header); refused before forwarding (native-psr-serving D20; "
                "widen the allowlist deliberately, never per request)",
                http_status=400, code=RESTRICTED_ROUTE_OPTION_UNSUPPORTED)
        if CHAT_TEMPLATE_KWARGS_KEY in request:
            kw = request[CHAT_TEMPLATE_KWARGS_KEY]
            inner = (sorted(set(kw) - {ENABLE_THINKING_KEY}) if isinstance(kw, Mapping)
                     else None)
            if inner is None or inner:
                raise ProxyRefusal(
                    where + f"{CHAT_TEMPLATE_KWARGS_KEY} carries "
                    f"{inner if inner is not None else repr(kw)} beyond the one template "
                    f"kwarg this proxy governs ({ENABLE_THINKING_KEY!r}) — a template kwarg "
                    "changes the render the boundary is located in; refused before "
                    "forwarding", http_status=400, code=RESTRICTED_ROUTE_OPTION_UNSUPPORTED)
        if BOUNDARY_REFUSAL_CONTROL_KEY in request:
            # THE BOUNDARY-REFUSAL CONTROL (O49): EXACTLY the one probe body, nothing else — a single
            # fixed user turn under add_generation_prompt: false, the fixed kwargs, no other key. A
            # [user, assistant, user] transcript under the marker would render the EARLIER assistant
            # header, which the last-occurrence locator finds: served steered, no boundary failure, a
            # completion stamped as the cell — the hole this check closes. So the body is compared
            # whole against the constant, and the marker's value must be THIS proxy's nonce.
            messages = request.get("messages")
            body_ok = (set(request) == set(BOUNDARY_REFUSAL_CONTROL_BODY_KEYS)
                       and isinstance(messages, list)
                       and [dict(m) if isinstance(m, Mapping) else m for m in messages]
                       == [dict(m) for m in _sa.BOUNDARY_REFUSAL_CONTROL_MESSAGES]
                       and all(request.get(k) == v and type(request.get(k)) is type(v)
                               for k, v in _sa.BOUNDARY_REFUSAL_CONTROL_KWARGS.items()))
            if not body_ok:
                raise ProxyRefusal(
                    where + f"{BOUNDARY_REFUSAL_CONTROL_KEY} admits EXACTLY the boundary-refusal probe body — "
                    f"messages {list(_sa.BOUNDARY_REFUSAL_CONTROL_MESSAGES)!r}, {_sa.BOUNDARY_REFUSAL_CONTROL_KWARGS!r}, "
                    f"model and the marker, no other key — the one render whose response header the worker "
                    f"provably cannot locate; got keys {sorted(request)} and {len(messages) if isinstance(messages, list) else 0} "
                    "message(s); refused before forwarding", http_status=400, code=BOUNDARY_CONTROL_SHAPE_INVALID)
            token = request.get(BOUNDARY_REFUSAL_CONTROL_KEY)
            if not (isinstance(token, str) and hmac.compare_digest(token, self.boundary_control_nonce)):
                raise ProxyRefusal(
                    where + f"{BOUNDARY_REFUSAL_CONTROL_KEY} does not carry this standup's control nonce — the "
                    "boundary-refusal probe is the standup's own (it reads the nonce off its proxy); refused "
                    "before forwarding", http_status=400, code=BOUNDARY_CONTROL_NONCE_INVALID)
            return
        if any(k in request for k in RESTRICTED_ROUTE_SEEDED_SHAPE_KEYS):
            messages = request.get("messages")
            last = messages[-1] if isinstance(messages, list) and messages else None
            seeded = (request.get("continue_final_message") is True
                      and request.get("add_generation_prompt") is False
                      and isinstance(last, Mapping) and last.get("role") == "assistant")
            if not seeded:
                raise ProxyRefusal(
                    where + f"{list(RESTRICTED_ROUTE_SEEDED_SHAPE_KEYS)} are admitted only in "
                    "the SEEDED shape — final message an assistant turn, "
                    "continue_final_message: true, add_generation_prompt: false (the one "
                    "non-default render whose header the worker's last-occurrence locator "
                    f"can place); got continue_final_message="
                    f"{request.get('continue_final_message')!r}, add_generation_prompt="
                    f"{request.get('add_generation_prompt')!r}, final role="
                    f"{last.get('role') if isinstance(last, Mapping) else None!r}; refused "
                    "before forwarding", http_status=400,
                    code=RESTRICTED_ROUTE_OPTION_UNSUPPORTED)

    async def _refuse_unlocated_native_response(
            self, route: RouteEntry, response: Mapping[str, Any], *,
            position_mode: str) -> str:
        """The lens post-check's twin for a RESTRICTED-MODE native route:
        claim this completion's entry from the worker's
        per-request failure ledger (``sisteer_take_request_failure``, the same
        RPC the lens's flow uses; the chat completion ``id`` IS the engine
        request id) BEFORE it is returned, and refuse the completion on:

        - an ENTRY (``error_type: ResponseBoundaryUnlocated`` — the render
          carried no locatable generation header, the worker served the request
          UNSTEERED and ledgered it) → :data:`RESPONSE_BOUNDARY_UNLOCATED` (500);
        - a FAULT payload (``{"fault": ...}``: the ledger LATCHED at its cap, or
          an armed worker with no bank) → :data:`RESPONSE_BOUNDARY_LEDGER_FAULT`
          (500) — for EVERY request on every restricted-mode route from then on,
          which is the design's fail-closed answer to a ledger nobody claims;
        - no ``id`` / no claim method → :data:`RESPONSE_BOUNDARY_POSTCHECK_FAILED`
          (502): unverifiable, never forwarded on faith.

        Returns the :data:`FAILURE_CLAIM_LABELS` member the audit row records
        (:data:`FAILURE_CLAIM_CLEAN` when nothing was recorded)."""
        where = (f"vmodel {route.virtual_model!r} (cell {route.cell_id!r}, native "
                 f"position mode {position_mode!r}): ")
        rid = response.get("id")
        if not isinstance(rid, str) or not rid:
            raise ProxyRefusal(
                where + "the upstream response carries no 'id', so its per-request "
                "response-boundary failure record cannot be claimed — refusing the "
                "response rather than forwarding an unverifiable completion",
                http_status=502, code=RESPONSE_BOUNDARY_POSTCHECK_FAILED)
        take = getattr(self._backend, "take_request_failure", None)
        if not callable(take):
            raise ProxyRefusal(
                where + "this backend exposes no take_request_failure post-check, so a "
                "position-restricted completion cannot be verified against the worker's "
                "per-request failure ledger — refusing to serve the route through it",
                http_status=502, code=RESPONSE_BOUNDARY_POSTCHECK_FAILED)
        try:
            failure = await take(rid)
        except ProxyRefusal as e:
            # the transport or the reply shape could not establish a claim (parse_failure_claim):
            # unverifiable on THIS route's terms, never forwarded on faith
            raise ProxyRefusal(where + str(e), http_status=502,
                               code=RESPONSE_BOUNDARY_POSTCHECK_FAILED) from e
        if failure is None:
            return FAILURE_CLAIM_CLEAN
        if isinstance(failure, Mapping) and "fault" in failure:
            raise ProxyRefusal(
                where + f"request {rid!r}: the worker's per-request failure ledger reports "
                f"a FAULT ({failure.get('fault')!r}; unclaimed={failure.get('unclaimed')!r}) "
                "— it holds unclaimed response-boundary failures at its cap and answers the "
                "fault for every claim, so no completion on a position-restricted route can "
                "be verified until the standup is replaced; this completion is discarded, "
                "never returned (fact 15)",
                http_status=500, code=RESPONSE_BOUNDARY_LEDGER_FAULT)
        fm = failure if isinstance(failure, Mapping) else {}
        detail = (fm.get("detail", fm.get("error", "<no detail recorded>"))
                  if isinstance(failure, Mapping) else failure)
        raise ProxyRefusal(
            where + f"request {rid!r} FAILED its response-boundary location "
            f"({fm.get('error_type', 'ResponseBoundaryUnlocated')}: "
            f"prompt_len={fm.get('prompt_len')!r}, header_len={fm.get('header_len')!r}) "
            "and was served UNSTEERED by the worker — its completion is discarded, never "
            f"returned stamped as the cell. Worker diagnosis: {detail}",
            http_status=500, code=RESPONSE_BOUNDARY_UNLOCATED,
            # the worker's ACTUAL error_type as a field: the boundary-refusal control requires
            # ResponseBoundaryUnlocated specifically — a RequestFactsMissing / PromptLenDisagreement
            # entry claimed here would otherwise empty the ledger and pass the control for the wrong
            # reason
            extra={"worker_error_type": fm.get("error_type"), "claimed_engine_ids": fm.get("claimed_engine_ids")})

    def _rewrite(self, route: RouteEntry,
                 request: Mapping[str, Any], *,
                 response_prefill: bool = False,
                 agent_loop: bool = False,
                 thinking_on: bool = False,
                 organism: Optional[str] = None
                 ) -> Tuple[Dict[str, Any], SaltDecision]:
        """The upstream body plus the :class:`SaltDecision` it was stamped
        with — returned together because the guard and the audit row need
        the SAME decision the salt came from, and a request-unique salt
        cannot be re-derived (it would be a different uuid)."""
        upstream = dict(request)
        # the declaration keys are proxy vocabulary, never upstream fields
        # (vLLM tolerates unknown keys — extra="allow" — but a stray
        # harness key upstream is drift waiting to be depended on).
        upstream.pop(RESPONSE_PREFILL_KEY, None)
        upstream.pop(AGENT_LOOP_KEY, None)
        upstream.pop(BOUNDARY_REFUSAL_CONTROL_KEY, None)      # the sixth control's marker: proxy vocabulary (O49)
        # the thinking-ON declaration is proxy vocabulary too; its render
        # half (chat_template_kwargs) is vLLM's own key and stays, merged
        # upstream over the launch's --default-chat-template-kwargs pin
        upstream.pop(THINKING_ON_KEY, None)
        ident = cm.parse_serving_identity(route.serving_identity)
        if ident.backend != "proxy_vllm":
            raise ProxyRefusal(
                f"vmodel {route.virtual_model!r} (cell {route.cell_id!r}) "
                f"carries non-proxy serving identity "
                f"{route.serving_identity!r} — the HF shim is the gate's "
                "fp32 reference oracle, never an audit serving path (B1; "
                "the user 2026-08-18): no audit traffic is served there",
                http_status=421, code="non_proxy_backend")
        if ident.path == "passthrough":
            upstream["model"] = self.base_model
        elif ident.path == "lora":
            # vLLM native multi-LoRA selects the adapter by model name;
            # adapters are registered at launch as <cell_id>=<dir>
            # (vllm_server), the zero adapter included (lora_export).
            # (The LoRA NAME is also natively part of every block's
            # cache key at the pin — the salt below is belt-and-braces
            # plus registry-generation separation.)
            upstream["model"] = route.cell_id
        elif ident.path == "lens":
            # a steered route may not be forked inside the engine (E42 / E46)
            # — checked here, the one seam every steered request crosses,
            # before any selector is built
            refuse_steered_forks(request, route)
            upstream["model"] = self.base_model
            selector: Dict[str, Any] = {
                "petri_object": route.cell_id,
                "expect_hash": route.intervention_hash,
            }
            if route.cell_id == NOOP_LENS_CELL_ID:
                selector["scale"] = 0.0
            if response_prefill:
                # the verified declaration travels as a selector FACT the
                # worker threads into HookContext.response_prefill —
                # request-scoped, never registry state. It travels as the
                # INT 1: the pinned frontend types vllm_xargs values
                # str|int|float and pydantic lax-coerces a JSON true to 1
                # anyway (measured — a bool sent here killed the engine on
                # the worker's own strict parse, 2026-09-01), so the proxy
                # sends the transport's one honest spelling and the worker
                # parse accepts exactly {0, 1, false, true}.
                selector["response_prefill"] = 1
            if agent_loop:
                # the second verified declaration, in the same transport
                # spelling for the same reason; the worker threads it
                # into HookContext.agent_loop. Never beside
                # response_prefill (_validate_agent_loop refuses both).
                selector["agent_loop"] = 1
            if thinking_on:
                # the third verified declaration: injected ONLY where the
                # served template's thinking
                # switch moves the header and the cell's attested declaration
                # admits the render (_validate_chat_template_kwargs); the
                # worker threads it into HookContext.thinking_on and locates
                # the boundary against the cell's ON header. Never beside the
                # two non-tail facts (refused there).
                selector["thinking_on"] = 1
            upstream[UPSTREAM_EXTRA_ARGS_KEY] = selector
        else:  # parse_serving_identity refuses first; belt-and-suspenders
            raise ProxyRefusal(f"unroutable serving path {ident.path!r}")
        # PER-CELL KV-CACHE ISOLATION (module docstring): every upstream
        # request carries the proxy-owned salt — the one mechanism that
        # keeps prefix caching from serving one cell's steered KV under
        # another cell's identity. WHICH namespace (shared by the cell, or
        # this request's alone) is decided by the cell's ATTESTED
        # resumed-prefill policy, and only an attested `always` shares.
        if organism is not None:
            # the ORGANISM OVERLAY: the steering leg unchanged, the adapter selected by vLLM's
            # native multi-LoRA model name (module docstring, ORGANISM_SEP); the organism also
            # rides in the salt so an overlaid request never shares KV with the bare leg
            upstream["model"] = organism
        decision = self._salt_decision(route, lens=(ident.path == "lens"), organism=organism)
        upstream["cache_salt"] = decision.salt
        return upstream, decision

    # -- the endpoint ----------------------------------------------------------

    async def handle_chat_completions(
        self, request: Mapping[str, Any], *,
        authorization: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._api_token is not None:
            if authorization != f"Bearer {self._api_token}":
                raise ProxyRefusal("missing or wrong proxy token",
                                   http_status=401, code="bad_token")
        if not isinstance(request, Mapping):
            raise ProxyRefusal("request body must be a JSON object")
        if request.get("stream"):
            raise ProxyRefusal(
                "stream=true is not supported by the petri proxy (v1 serves "
                "non-streaming responses only — a refusal, never a silent "
                "buffer); send stream=false", code="streaming_unsupported")
        smuggled = sorted(set(request) & set(FORBIDDEN_CLIENT_KEYS))
        if smuggled:
            raise ProxyRefusal(
                f"request carries reserved selector key(s) {smuggled} — "
                "selectors are injected by the proxy from the routing "
                "table, never by clients (H3)", code="selector_smuggling")
        vmodel = request.get("model")
        if not isinstance(vmodel, str) or not vmodel:
            raise ProxyRefusal("request has no 'model'")
        route, organism = self._resolve(vmodel)

        # THE RESTRICTED-ROUTE ALLOWLIST, at the earliest point the route
        # is known: on a lora route whose attested adapter writes at a
        # restricted native position mode, a body key outside
        # RESTRICTED_ROUTE_ALLOWED_KEYS refuses here and reaches no later
        # gate and no upstream. All-positions residual routes and LoRA
        # routes: untouched (restricted_position_mode is None).
        restricted_mode = self.restricted_position_mode(route)
        if restricted_mode is not None:
            self._validate_restricted_route_request(request, route,
                                                    position_mode=restricted_mode)
        elif BOUNDARY_REFUSAL_CONTROL_KEY in request:
            raise ProxyRefusal(
                f"vmodel {vmodel!r} (cell {route.cell_id!r}) has no response boundary to fail — "
                f"{BOUNDARY_REFUSAL_CONTROL_KEY} is the restricted-mode routes' control (O49); an "
                "all-positions residual route or a LoRA route never ledgers an unlocated boundary; "
                "refused before forwarding", http_status=400, code=BOUNDARY_CONTROL_NOT_RESTRICTED)

        request = self._entry_declare(request, route)

        if route.si_prefix_hash is not None:
            self._verify_prompted_prefix(route, request)
        else:
            self._refuse_unexpected_prefix(request, route)

        # the thinking-ON gate FIRST (§2.5.6: a declaration beside a non-tail
        # declaration is thinking_on_mismatch), then the two non-tail gates,
        # then the all-turns render check (a pre-check on the route's mask)
        render_regime, thinking_on = self._validate_chat_template_kwargs(request, route)
        agent_loop = self._validate_agent_loop(request, route)
        response_prefill = self._validate_response_prefill(request, route)
        all_turns_check = self._validate_all_turns_render(request, route)
        upstream_payload, salt = self._rewrite(
            route, request, response_prefill=response_prefill,
            agent_loop=agent_loop, thinking_on=thinking_on, organism=organism)
        # the salt discipline, enforced HERE so its violation is a
        # per-request 409 and never the engine-fatal in-worker refusal
        # (module docstring; the engine's own guard stays as backstop).
        self._guard_cache_salt(
            route, salt=salt.salt, lens=salt.lens, regime=salt.regime)

        # the per-request audit entry, BEFORE the call — a request that
        # dies upstream is still on the record. The CONTRACT field names
        # (virtual_model, petri_intervention_hash) appear verbatim in the
        # log entry, in the structured logging record's attributes, and
        # in the response.
        log_entry = {
            "virtual_model": vmodel,
            "petri_intervention_hash": route.intervention_hash,
            "cell_id": route.cell_id,
            "serving_identity": route.serving_identity,
            # WHICH KV namespace this request was minted into, and the
            # resumed-prefill policy the proxy read to decide it — so the
            # audit stream answers "did this request share a prefill, and
            # on whose say-so" without re-deriving anything (the durable
            # audit record is the structured logging stream below, which
            # carries the same two fields).
            LEDGER_KEY_SALT_REGIME: salt.regime,
            LEDGER_KEY_RESUMED_POLICY: salt.policy,
            # the RENDER REGIME (§2.5.5 / §2.5.11): recorded on EVERY route,
            # base included, so the base ON arm and the lens ON arm stay
            # symmetric without the base route needing a declaration
            LEDGER_KEY_RENDER_REGIME: render_regime,
            # the NATIVE position mode the salt was minted under (None off a
            # residual route) and what the per-request failure claim read
            # for this row — filled in after the post-check below; a row whose
            # completion was REFUSED keeps the label naming why (the refused
            # response never reaches the stamp, so the row is its record)
            LEDGER_KEY_POSITION_MODE: salt.position_mode,
            LEDGER_KEY_FAILURE_CLAIM: FAILURE_CLAIM_NOT_APPLICABLE,
        }
        if organism is not None:
            log_entry["organism"] = organism
        if all_turns_check is not None:
            log_entry[LEDGER_KEY_ALL_TURNS_CHECK] = all_turns_check
        if len(self.request_log) == self.request_log.maxlen:
            self.n_request_log_dropped += 1  # the deque evicts the oldest
        self.request_log.append(log_entry)
        log.info("petri-proxy route: virtual_model=%s "
                 "petri_intervention_hash=%s (cell=%s, identity=%s, "
                 "cache_salt_regime=%s, resumed_prefill_policy=%s, "
                 "render_regime=%s)",
                 vmodel, route.intervention_hash, route.cell_id,
                 route.serving_identity, salt.regime, salt.policy,
                 render_regime, extra=dict(log_entry))

        if self._serialize:
            async with self._lock:
                response = await self._backend.chat_completions(
                    upstream_payload)
        else:
            response = await self._backend.chat_completions(upstream_payload)

        # LENS post-check (the loud half of the 2026-08-30 per-request
        # containment): a request whose hook op failed — e.g. a response
        # boundary that could not be located in its render — was served
        # UNSTEERED by the worker so that one bad request could not kill
        # the whole engine. Its completion must therefore be refused
        # HERE, before anything downstream can file it as a served
        # result. Runs for every selector-carrying (lens) request; the
        # unhooked passthrough/prompted legs and plain native-LoRA legs have
        # no hook ops to fail. Since native-psr-serving stage 2 the SAME
        # claim runs for a RESTRICTED-MODE native route:
        # its worker locates the response boundary per request and ledgers
        # an unlocated one exactly as the lens does — the completion is
        # refused here, never returned stamped as the cell; a latched ledger
        # refuses every such completion (fact 15). All-positions residual
        # routes write every row from that row alone and are not checked.
        claim_label = FAILURE_CLAIM_NOT_APPLICABLE
        try:
            if UPSTREAM_EXTRA_ARGS_KEY in upstream_payload:
                await self._refuse_poisoned_lens_response(route, response)
                claim_label = FAILURE_CLAIM_CLEAN
            elif restricted_mode is not None:
                claim_label = await self._refuse_unlocated_native_response(
                    route, response, position_mode=restricted_mode)
        except ProxyRefusal as e:
            log_entry[LEDGER_KEY_FAILURE_CLAIM] = _FAILURE_CLAIM_BY_CODE.get(
                e.code, FAILURE_CLAIM_UNVERIFIABLE)
            log.warning("petri-proxy refused a completion after the per-request "
                        "failure claim: virtual_model=%s cell=%s position_mode=%s "
                        "code=%s claim=%s", vmodel, route.cell_id, salt.position_mode,
                        e.code, log_entry[LEDGER_KEY_FAILURE_CLAIM], extra=dict(log_entry))
            raise
        log_entry[LEDGER_KEY_FAILURE_CLAIM] = claim_label

        out = dict(response)
        # report the VIRTUAL name to the client (the target must never
        # learn the base/adapter routing), plus the bookkeeping stamp.
        # `petri_intervention_hash` is the REQUIRED top-level audit field
        # — the nested `petri` block is context, never the contract.
        out["model"] = vmodel
        out["petri_intervention_hash"] = route.intervention_hash
        out["petri"] = {
            "virtual_model": vmodel,
            **({"organism": organism} if organism is not None else {}),
            "cell_id": route.cell_id,
            "serving_identity": route.serving_identity,
            "intervention_hash": route.intervention_hash,
            # the audit row's two resumed-prefill fields, echoed so an
            # out-of-process observer (the gate) can read the decision this
            # proxy actually made for its request. CONTEXT, not the
            # contract: `proxy_engine.STAMP_FIELDS` names four fields and
            # reads only those, so these are additive to every stamp check.
            LEDGER_KEY_SALT_REGIME: salt.regime,
            LEDGER_KEY_RESUMED_POLICY: salt.policy,
            # the render regime this request was served under (§2.5.11),
            # additive like the two above
            LEDGER_KEY_RENDER_REGIME: render_regime,
            # the native position mode the salt was minted under and the
            # per-request failure claim's reading — a served row's claim is
            # `clean` or `not-applicable`; the refusing labels never reach
            # a stamp
            LEDGER_KEY_POSITION_MODE: salt.position_mode,
            LEDGER_KEY_FAILURE_CLAIM: claim_label,
        }
        if all_turns_check is not None:
            out["petri"][LEDGER_KEY_ALL_TURNS_CHECK] = all_turns_check
        if self.engine_attestation is not None:
            # the ENGINE this row was served by, when a lora-native standup
            # attested one, so every served row is distinguishable — on EVERY
            # route, base included; additive, STAMP_FIELDS unchanged
            out["petri"][ENGINE_KEY] = self.engine_attestation["kind"]
            out["petri"][ENGINE_STATE_KEY] = {
                "enforce_eager": self.engine_attestation["enforce_eager"],
                "cudagraph_mode": self.engine_attestation["cudagraph_mode"]}
        return out

    async def handle_completions(
        self, request: Mapping[str, Any], *,
        authorization: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST ``/v1/completions`` — REFUSED on every route: the proxy serves
        chat completions only (every driver renders through the served chat
        template), and a raw-prompt completion carries no generation header
        the residual slot's locator or the lens's mask could place a boundary
        in. On a RESTRICTED-MODE native route the refusal is the
        restricted-route code (400 — the endpoint is a render-changing
        option); everywhere else :data:`COMPLETIONS_UNSUPPORTED` (404, the
        status class the unmounted route always answered). Same token rule as
        chat. Never returns."""
        if self._api_token is not None:
            if authorization != f"Bearer {self._api_token}":
                raise ProxyRefusal("missing or wrong proxy token",
                                   http_status=401, code="bad_token")
        vmodel = request.get("model") if isinstance(request, Mapping) else None
        route = self._routes.get(vmodel) if isinstance(vmodel, str) else None
        restricted_mode = self.restricted_position_mode(route) if route is not None else None
        if restricted_mode is not None:
            raise ProxyRefusal(
                f"vmodel {vmodel!r} (cell {route.cell_id!r}, native position mode "
                f"{restricted_mode!r}): /v1/completions is refused on a "
                "position-restricted native route — a raw prompt has no generation header "
                "the worker can locate the response boundary against; send a chat "
                "completion", http_status=400, code=RESTRICTED_ROUTE_OPTION_UNSUPPORTED)
        raise ProxyRefusal(
            "/v1/completions is not served by the petri proxy (chat completions only — "
            "every audit render goes through the served chat template); send "
            "/v1/chat/completions", http_status=404, code=COMPLETIONS_UNSUPPORTED)

    async def handle_tokenize(
        self, request: Mapping[str, Any], *,
        authorization: Optional[str] = None,
    ) -> Dict[str, Any]:
        """The read-only ``/tokenize`` pass-through: the served clients
        resolve vLLM's FLOOR-ONLY context-length rejection ("prompt contains
        at least N input tokens") by tokenizing the IDENTICAL messages under
        the same virtual-model resolution, so the once-only clamp works on
        the exact prompt length instead of failing. Same token auth and
        selector-smuggling refusal as chat; the virtual model resolves to the
        same upstream model name (passthrough / lens -> base model, LoRA ->
        adapter name); only the tokenize-relevant keys are forwarded; NO
        steering selector, NO salt (nothing is generated or cached). A backend
        without the route (or an upstream 404) refuses, and the client then
        raises the typed error with reason floor_only_unresolvable rather than
        guess."""
        if self._api_token is not None:
            if authorization != f"Bearer {self._api_token}":
                raise ProxyRefusal("missing or wrong proxy token",
                                   http_status=401, code="bad_token")
        if not isinstance(request, Mapping):
            raise ProxyRefusal("request body must be a JSON object")
        smuggled = sorted(set(request) & set(FORBIDDEN_CLIENT_KEYS))
        if smuggled:
            raise ProxyRefusal(
                f"request carries reserved selector key(s) {smuggled} — "
                "selectors are injected by the proxy from the routing "
                "table, never by clients (H3)", code="selector_smuggling")
        vmodel = request.get("model")
        if not isinstance(vmodel, str) or not vmodel:
            raise ProxyRefusal("request has no 'model'")
        route, organism = self._resolve(vmodel)
        if not isinstance(request.get("messages"), Sequence) or not request.get("messages"):
            raise ProxyRefusal("tokenize needs a non-empty 'messages' list")
        # the kwargs SHAPE validation, no declaration: nothing is
        # generated or cached by a tokenize, so the render regime needs no
        # petri_thinking_on here — but a malformed or inert kwarg must not
        # measure a render the chat endpoint would refuse
        if (self._validate_kwargs_shape(request, self._thinking_header) is False
                and self.thinking_mode == THINKING_MODE_ON):
            # a tokenize measures the render the chat endpoint would serve: on a
            # thinking-ON standup a regime flip refuses there, so it refuses here
            raise ProxyRefusal(
                f"{CHAT_TEMPLATE_KWARGS_KEY}.{ENABLE_THINKING_KEY}: false on a "
                "thinking-ON standup's tokenize — it would measure a render the chat "
                "endpoint refuses; drop the key", code=THINKING_REGIME_MISMATCH)
        ident = cm.parse_serving_identity(route.serving_identity)
        if ident.backend != "proxy_vllm":
            raise ProxyRefusal(
                f"vmodel {vmodel!r} carries non-proxy serving identity "
                f"{route.serving_identity!r}", http_status=421,
                code="non_proxy_backend")
        upstream_model = route.cell_id if ident.path == "lora" else self.base_model
        if organism is not None:
            upstream_model = organism
        payload: Dict[str, Any] = {"model": upstream_model}
        for key in TOKENIZE_FORWARD_KEYS:
            if key in request:
                payload[key] = request[key]
        payload.setdefault("add_generation_prompt", True)
        tokenize = getattr(self._backend, "tokenize", None)
        if tokenize is None:
            raise ProxyRefusal(
                "this proxy's backend exposes no /tokenize route — the exact "
                "prompt length of a floor-only rejection cannot be resolved here",
                http_status=501, code="tokenize_unsupported")
        response = await tokenize(payload)
        out = dict(response)
        out["model"] = vmodel
        return out

    def served_virtual_models(self) -> Tuple[str, ...]:
        """Only the vmodels THIS proxy actually serves (backend
        ``proxy_vllm``). A route with any other identity — impossible via
        enumerate_cells, so foreign/corrupt by construction — stays in
        the internal table so its requests get a precise 421, but is
        never advertised as a served model (a listed-then-refused model
        would poison discovery and the engine health check)."""
        return tuple(sorted(
            v for v, r in self._routes.items()
            if cm.parse_serving_identity(r.serving_identity).backend
            == "proxy_vllm")) + self.overlay_virtual_models()

    async def handle_lora_routing_rpc(
        self, request: Mapping[str, Any], *,
        authorization: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST :data:`LORA_ROUTING_RPC_ROUTE` — the allowlisted pass-through
        to the native worker's LoRA routing recorder. Same token rule as
        chat. Refuses unless this proxy was started with a LORA-NATIVE engine
        attestation (a lens server has no recorder and this must not become a
        generic RPC door), unless ``method`` is one of the three recorder
        methods, and unless ``args`` is a list of strings (the dev route's own
        contract) — each takes exactly ONE (the client-minted epoch
        id). The backend must expose ``collective_rpc``; an
        upstream failure is a 502 naming the method, never a silent ``{}``. Worker REFUSALS (``already_armed`` /
        ``epoch_mismatch`` / ...) ride as ``ok: False`` payloads inside the 200
        body and are forwarded UNCHANGED — the client reader refuses on them
        (transport)."""
        if self._api_token is not None:
            if authorization != f"Bearer {self._api_token}":
                raise ProxyRefusal("missing or wrong proxy token",
                                   http_status=401, code="bad_token")
        if self.engine_attestation is None:
            raise ProxyRefusal(
                "this proxy serves no lora-native engine attestation — the LoRA "
                "routing recorder exists only on a plugin-free lora-native worker; "
                "refusing the RPC pass-through", http_status=404, code="not_lora_native")
        if not isinstance(request, Mapping):
            raise ProxyRefusal("request body must be a JSON object")
        method = request.get("method")
        if method not in LORA_ROUTING_RPC_METHODS:
            raise ProxyRefusal(
                f"method {method!r} is not one of the LoRA routing recorder RPCs "
                f"{list(LORA_ROUTING_RPC_METHODS)} — this route forwards exactly those",
                code="rpc_method_not_allowed")
        args = request.get("args", [])
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise ProxyRefusal(
                f"{method}: args must be a list of strings (the collective_rpc route "
                "passes serialized string args only)", code="rpc_bad_args")
        # arm, observe and disarm each take exactly ONE arg: the client-minted epoch id
        if len(args) != 1 or not args[0]:
            raise ProxyRefusal(
                f"{method}: takes exactly 1 arg (the non-empty epoch id), got {len(args)}",
                code="rpc_bad_args")
        rpc = getattr(self._backend, "collective_rpc", None)
        if not callable(rpc):
            raise ProxyRefusal(
                f"{method}: this backend exposes no collective_rpc, so the routing "
                "recorder cannot be reached through this proxy",
                http_status=502, code="rpc_unavailable")
        try:
            out = await rpc(method, args)
        except ProxyRefusal:
            raise
        except Exception as e:  # noqa: BLE001 — named, never a silent {}
            raise ProxyRefusal(
                f"{method}: the upstream collective_rpc failed "
                f"({type(e).__name__}: {e})", http_status=502, code="rpc_failed") from e
        if not isinstance(out, Mapping):
            raise ProxyRefusal(
                f"{method}: the upstream answered a non-object body "
                f"({type(out).__name__})", http_status=502, code="rpc_failed")
        return dict(out)

    def handle_models(self, *, authorization: Optional[str] = None
                      ) -> Dict[str, Any]:
        """GET /v1/models — authenticated exactly like chat when a token
        is configured (an unauthenticated health endpoint would let a
        wrong key 'pass' the health check and fail only at the first
        generation)."""
        if self._api_token is not None:
            if authorization != f"Bearer {self._api_token}":
                raise ProxyRefusal("missing or wrong proxy token",
                                   http_status=401, code="bad_token")
        return self.models_payload()

    async def handle_metrics(self, *, authorization: Optional[str] = None) -> str:
        """``GET /metrics`` — the read-only pass-through of the ENGINE's Prometheus
        page, body verbatim (F-O, 2026-09-18). WHY it exists: the engine's
        ``/metrics`` sits behind ``vllm_auth.SisteerTokenAuthMiddleware`` like every
        non-liveness path, and the per-launch bearer lives ONLY inside the standup
        process (``ServerLaunch.api_key`` -> this proxy's backend). A same-host
        sampler (the container app's ``vllm_metrics.jsonl`` scrape, the
        ``determinism_check.py`` drained-state read) therefore cannot read the engine
        directly (HTTP 401) and must never be handed the token through a file the
        attempt root syncs to shared storage. It reads THIS route instead, guarded
        exactly like ``/v1/models`` when the proxy has a token and open like the
        proxy's other routes when it has none (``endpoint.json.proxy_token: None``).
        Nothing is generated, cached or steered; a backend without the read (a
        stub, a lens standup whose transport lacks it) refuses by name."""
        if self._api_token is not None:
            if authorization != f"Bearer {self._api_token}":
                raise ProxyRefusal("missing or wrong proxy token",
                                   http_status=401, code="bad_token")
        metrics = getattr(self._backend, "metrics", None)
        if metrics is None:
            raise ProxyRefusal(
                "this proxy's backend exposes no /metrics read — the engine's "
                "Prometheus page cannot be passed through here",
                http_status=501, code="metrics_unsupported")
        body = await metrics()
        if not isinstance(body, str):
            raise ProxyRefusal(
                f"the upstream /metrics read answered {type(body).__name__}, not text",
                http_status=502, code="metrics_failed")
        return body

    def models_payload(self) -> Dict[str, Any]:
        """The /v1/models body: the PROXY-SERVED virtual models plus this
        proxy's identity metadata (so a health check can verify it is
        talking to the petri proxy, not a stale or foreign OpenAI
        endpoint)."""
        return {"object": "list",
                "data": [{"id": v, "object": "model", "owned_by": "petri"}
                         for v in self.served_virtual_models()],
                "petri_proxy": {
                    "version": 1,
                    "serving_mode": self.serving_mode,
                    "exact_replay_strategy": self.exact_replay_strategy,
                    "base_model": self.base_model,
                    # the multi-turn landing's two facts about THIS proxy:
                    # whether it declares agent-loop turns for Petri at
                    # entry, and whether it can verify such declarations
                    # against the render at all
                    "petri_agent_loop_entry": self._petri_agent_loop_entry,
                    "agent_loop_render": {
                        "mode": self._agent_loop_render_mode,
                        "loaded": self._agent_loop_render is not None,
                        "source": (
                            type(self._agent_loop_render).__name__
                            if self._agent_loop_render_mode == "explicit"
                            else f"{self.base_model}@"
                                 f"{self._base_revision or 'unpinned'}"),
                    },
                    # the served lens reduction mode, ONLY when attested at
                    # launch (absent = not attested; never defaulted here)
                    **({LENS_REDUCTION_MODE_KEY:
                        self._launch_attestation[LENS_REDUCTION_MODE_KEY]}
                       if LENS_REDUCTION_MODE_KEY in self._launch_attestation
                       else {}),
                    # the server-default sampling policy the launch was
                    # composed under, ONLY when the standup handed it over
                    # (absent = a pre-key standup; never defaulted here)
                    **({SERVER_DEFAULT_POLICY_KEY:
                        self._launch_attestation[SERVER_DEFAULT_POLICY_KEY]}
                       if SERVER_DEFAULT_POLICY_KEY in self._launch_attestation
                       else {}),
                    # the per-cell RESUMED-PREFILL POLICY this proxy was
                    # handed, ONLY when the workers attested one (absent =
                    # nothing attested, which is exactly "nothing shares a
                    # prefill here"; never defaulted). A zero-traffic read
                    # of the decision every steered request will be minted
                    # under.
                    **({"resumed_prefill_policy_cells":
                        dict(self.resumed_prefill_policy_cells)}
                       if self.resumed_prefill_policy_cells is not None
                       else {}),
                    # the per-cell SERVE DECLARATIONS this proxy validates
                    # thinking-ON declarations against, PRESENT when attested —
                    # an EMPTY map is the stated "no declared cell" — with the
                    # launcher's error beside it (null when the map is present;
                    # the reason when it is not); neither key when no launcher
                    # ever spoke
                    **({SERVE_DECLARATIONS_KEY: {
                            k: dict(v) for k, v in self.lens_serve_declarations.items()},
                        SERVE_DECLARATIONS_ERROR_KEY: None}
                       if self.lens_serve_declarations is not None
                       else ({SERVE_DECLARATIONS_ERROR_KEY: self.lens_serve_declarations_error}
                             if self.lens_serve_declarations_error is not None else {})),
                    # THE THINKING MODE (THINKING_MODES): always stated — a
                    # reader asserting the regime must see "off" or "on", never
                    # infer it from absence — with the declaration key every
                    # lens-route request of an ON proxy must carry
                    THINKING_MODE_KEY: self.thinking_mode,
                    THINKING_DECLARATION_KEY_KEY: THINKING_ON_KEY,
                    # the served template's thinking-header fact the gate
                    # decides from, when the routing table recorded it
                    # ... with the reason its ON certification is UNAVAILABLE
                    # beside it (null beside a fully derived fact; the reason
                    # alone when a launcher recorded no fact at all)
                    **({THINKING_HEADER_KEY: dict(self._thinking_header),
                        THINKING_HEADER_ERROR_KEY: self.thinking_header_error}
                       if self._thinking_header is not None
                       else ({THINKING_HEADER_ERROR_KEY: self.thinking_header_error}
                             if self.thinking_header_error is not None else {})),
                    # the per-cell SERVED POSITIONS the thinking-ON gate reads
                    # header-locatedness from (§2.5.4 narrowing), when the
                    # routing table recorded the map — an EMPTY map is the
                    # stated "no lens leg states a mode"; neither key when the
                    # table predates the landing (every undeclared ON on a
                    # lens route then refuses, as before)
                    **({LENS_SERVED_POSITIONS_KEY: {
                            k: list(v) for k, v in self.lens_served_positions.items()}}
                       if self.lens_served_positions is not None else {}),
                    # the proxy's OWN record of its lens cache regime (§2.4):
                    # it mints every salt, so this is the authority
                    LENS_SALT_POLICY_KEY: self.lens_salt_policy,
                    # the per-cell fitted-artifact digests, a launch fact
                    # echoed when the standup handed it over (§2.6)
                    **({OBJECT_SHA256_BY_CELL_KEY: dict(self.object_artifact_sha256_by_cell)}
                       if self.object_artifact_sha256_by_cell is not None else {}),
                    # the ENGINE a lora-native standup attested: kind, the
                    # eager/graph state, and the per-adapter digests the
                    # blocks were held to — ONLY when attested (absent
                    # otherwise: every lens standup's payload is byte-identical
                    # to before)
                    **({ENGINE_KEY: self.engine_payload()}
                       if self.engine_attestation is not None else {}),
                }}

    def engine_payload(self) -> Dict[str, Any]:
        """The :data:`ENGINE_KEY` block of ``/v1/models``:
        ``{"kind", "enforce_eager", "cudagraph_mode", "residual_slot": {"armed"},
        "adapters": {name: {config_sha256, weights_sha256, fingerprint_bf16,
        lora_int_id, petri_intervention_hash, petri_zero_adapter,
        petri_source_kind, residual_fingerprint, position_mode}}}`` (the
        :data:`ENGINE_ADAPTER_KEYS` entries) from the validated attestation. Refuses when none was attested — the
        caller gates on :attr:`engine_attestation`."""
        att = self.engine_attestation
        if att is None:
            raise ProxyError("engine_payload: this proxy was started with no engine "
                             "attestation")
        return {"kind": att["kind"], "enforce_eager": att["enforce_eager"],
                "cudagraph_mode": att["cudagraph_mode"],
                "residual_slot": dict(att["residual_slot"]),
                "adapters": {name: dict(entry) for name, entry in att["adapters"].items()}}


# --- HTTP surface (aiohttp) -----------------------------------------------------


#: the proxy's read-only pass-through of the engine's Prometheus page (F-O):
#: ``GET`` at the server ROOT, where vLLM serves its own, so one URL shape
#: (``http://host:port/metrics``) reads either the engine (with its bearer) or
#: the proxy (without one, when the standup records ``proxy_token: None``).
METRICS_ROUTE = "/metrics"
#: the Prometheus text exposition content type vLLM's /metrics answers with
METRICS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def make_app(proxy: RoutingProxy):
    """The aiohttp application. Lazy-imported so the routing core stays
    importable in the CPU test venv."""
    from aiohttp import web

    async def chat(request):
        try:
            payload = await request.json()
        except Exception:
            return _error(web, "request body is not valid JSON", 400,
                          "bad_json")
        try:
            out = await proxy.handle_chat_completions(
                payload, authorization=request.headers.get("Authorization"))
        except ProxyRefusal as e:
            return _error(web, str(e), e.http_status, e.code, extra=getattr(e, 'extra', None))
        return web.json_response(out)

    async def models(request):
        try:
            payload = proxy.handle_models(
                authorization=request.headers.get("Authorization"))
        except ProxyRefusal as e:
            return _error(web, str(e), e.http_status, e.code, extra=getattr(e, 'extra', None))
        return web.json_response(payload)

    async def tokenize(request):
        try:
            payload = await request.json()
        except Exception:
            return _error(web, "request body is not valid JSON", 400,
                          "bad_json")
        try:
            out = await proxy.handle_tokenize(
                payload, authorization=request.headers.get("Authorization"))
        except ProxyRefusal as e:
            return _error(web, str(e), e.http_status, e.code, extra=getattr(e, 'extra', None))
        return web.json_response(out)

    async def lora_routing_rpc(request):
        try:
            payload = await request.json()
        except Exception:
            return _error(web, "request body is not valid JSON", 400,
                          "bad_json")
        try:
            out = await proxy.handle_lora_routing_rpc(
                payload, authorization=request.headers.get("Authorization"))
        except ProxyRefusal as e:
            return _error(web, str(e), e.http_status, e.code, extra=getattr(e, 'extra', None))
        return web.json_response(out)

    async def completions(request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        try:
            out = await proxy.handle_completions(
                payload, authorization=request.headers.get("Authorization"))
        except ProxyRefusal as e:
            return _error(web, str(e), e.http_status, e.code, extra=getattr(e, 'extra', None))
        return web.json_response(out)   # unreachable: the handler always refuses

    async def metrics(request):
        try:
            body = await proxy.handle_metrics(
                authorization=request.headers.get("Authorization"))
        except ProxyRefusal as e:
            return _error(web, str(e), e.http_status, e.code, extra=getattr(e, 'extra', None))
        # the Prometheus text exposition content type, the body verbatim
        return web.Response(body=body.encode("utf-8"),
                            headers={"Content-Type": METRICS_CONTENT_TYPE})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    # the raw-prompt endpoint is REFUSED with a JSON error code on every route
    # (a restricted-mode route names its own code)
    app.router.add_post("/v1/completions", completions)
    app.router.add_get("/v1/models", models)
    # the read-only tokenize pass-through (served_window's floor-only
    # resolution); vLLM serves it at the root, OpenAI-style clients under /v1
    app.router.add_post("/v1/tokenize", tokenize)
    app.router.add_post("/tokenize", tokenize)
    # the LoRA routing recorder pass-through: only a lora-native proxy
    # answers it; a lens proxy refuses at the handler
    app.router.add_post(LORA_ROUTING_RPC_ROUTE, lora_routing_rpc)
    # the read-only /metrics pass-through (F-O): the engine's Prometheus page, body
    # verbatim, for same-host samplers that hold no engine bearer (the route the
    # container app's metrics_url_from_endpoint and determinism_check.py read)
    app.router.add_get(METRICS_ROUTE, metrics)
    return app


def _error(web, message: str, status: int, code: str, extra: Optional[Mapping[str, Any]] = None):
    return web.json_response(
        {"error": {"message": message, "type": "invalid_request_error",
                   "code": code, **{k: v for k, v in dict(extra or {}).items()
                                    if k not in ("message", "type", "code")}}},
        status=status)


def load_served_tokenizer(name_or_path: str, *,
                          revision: Optional[str] = None,
                          chat_template: Optional[str] = None,
                          local_files_only: bool = False) -> Any:
    """The served tokenizer for the agent-loop render probe — plain
    ``from_pretrained`` (never trust_remote_code) under the thinking-off
    pin, the same seam as the worker's own preflight tokenizer
    (``lens_worker._sisteer_load_tokenizer``): on a reasoning template the
    probe must render the fit's regime, or its header would disagree with
    the registry's. ``chat_template`` (optional) is the PINNED served
    template text — the file ``build_server_launch`` hashes against the
    manifest — so the probe renders exactly the bytes the server renders
    rather than whatever the hub revision ships."""
    from transformers import AutoTokenizer

    from sisteer.models import pin_reasoning_template_off
    tok = AutoTokenizer.from_pretrained(name_or_path, revision=revision,
                                        local_files_only=local_files_only)
    pin_reasoning_template_off(tok)
    if chat_template is not None:
        tok.chat_template = chat_template
    return tok


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI: serve the routing proxy on a CLUSTER-INTERNAL interface (no
    open port — reached over the SSH tunnel)."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--routing-table", required=True,
                    help="JSON from export_routing_table")
    ap.add_argument("--upstream", required=True,
                    help="the vLLM server base URL, e.g. http://127.0.0.1:8000")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--launch-attestation", default=None,
                    help="JSON file holding the composed ServerLaunch env "
                         "(vllm_server.build_server_launch) — REQUIRED when "
                         "the routing table records exact_replay_strategy="
                         "'batch_invariant': the proxy refuses to run "
                         "unserialized without proof the server-side "
                         "mechanism is active (D0 constraint 3)")
    ap.add_argument("--tokenizer", default=None,
                    help="the SERVED tokenizer (model id or path) — builds "
                         "the agent-loop render probe EAGERLY (a bad pin "
                         "fails at startup); without it the probe loads "
                         "lazily, local files only, from the routing table's "
                         "base_model@base_revision on the first agent-loop "
                         "turn")
    ap.add_argument("--tokenizer-revision", default=None,
                    help="the served tokenizer's revision (the audit pin)")
    ap.add_argument("--chat-template", default=None,
                    help="the PINNED served chat template file (the one "
                         "build_server_launch hashes) — the render probe "
                         "renders exactly the served bytes")
    ap.add_argument("--lens-salt-policy", choices=list(LENS_SALT_POLICIES),
                    default=LENS_SALT_POLICY_ATTESTED,
                    help="the lens cache regime (fit-support-transfer plan §2.4): "
                         "'attested' (default) lets the attested resumed-prefill "
                         "map and the declared-cells rule decide; "
                         "'request_unique_all' mints every lens leg request-unique "
                         "(audit label forced-request-unique). Recorded in "
                         "/v1/models")
    ap.add_argument("--petri-agent-loop", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="the Petri-facing entry: declare petri_agent_loop "
                         "for agent-loop-structured lens requests (inspect_ai's "
                         "client cannot). ON by default; --no-petri-agent-loop "
                         "is a stated opt-out, after which agent-loop renders "
                         "are forwarded undeclared and refuse at the adapter "
                         "(contained). Without --tokenizer the render probe "
                         "loads lazily from the routing table's base_model")
    ap.add_argument("--serving-attestation", default=None,
                    help="the lora-native standup's serving_attestation.json "
                         "(native-multilora-serving plan §1.4) — read through "
                         "serving_attestation.read_serving_attestation and reduced "
                         "to its engine identity; REQUIRED with a version-2 "
                         "(lora-native) routing table, whose adapter blocks are "
                         "then held to the attested adapters; refused with a "
                         "version-1 table (nothing to bind)")
    ap.add_argument("--thinking-mode", choices=list(THINKING_MODES),
                    default=THINKING_MODE_OFF,
                    help="the standup's thinking mode (THINKING_MODES): 'off' (default, "
                         "the per-request regime every proxy to date served) or 'on' — the "
                         "vLLM launch pinned enable_thinking: true and every lens-route "
                         "request must declare petri_thinking_on: true. 'on' needs the "
                         "routing table's certified thinking_header fact and an attested "
                         "lens_serve_declarations map (--lens-serve-declarations)")
    ap.add_argument("--lens-serve-declarations", default=None,
                    help="JSON file holding the workers' attested per-cell serve "
                         "declarations (lens_worker.serve_declarations_attested's "
                         "map) — REQUIRED with --thinking-mode on")
    args = ap.parse_args(argv)

    if args.host in ("0.0.0.0", "::"):
        raise ProxyError("refusing to bind a wildcard interface — the proxy "
                         "is cluster-internal only, reached over the SSH "
                         "tunnel (D0 constraint 5)")
    table = load_routing_table(Path(args.routing_table))
    attestation = None
    if args.launch_attestation:
        attestation = json.loads(Path(args.launch_attestation).read_text())
    engine_attestation = None
    if args.serving_attestation:
        if table[ADAPTERS_KEY] is None:
            raise ProxyError("--serving-attestation given with a version-1 routing table — "
                             "there are no adapter blocks to hold the attested adapters "
                             "to; a lora-native standup exports a version-2 table")
        # the strict reader + its engine-identity reduction; it reads the
        # file, and validated_engine_attestation holds the shape. Imported
        # here so the proxy core stays importable without it
        from sisteer.capeval.serving.serving_attestation import (
            engine_identity, read_serving_attestation)
        engine_attestation = engine_identity(
            read_serving_attestation(Path(args.serving_attestation)))
    elif table[ADAPTERS_KEY] is not None:
        raise ProxyError("a version-2 (lora-native) routing table needs --serving-attestation "
                         "— its adapter blocks are held to the server's attested adapters "
                         "before any traffic; refusing to serve natively unbound")
    render = None
    if args.tokenizer:
        template = (Path(args.chat_template).read_text(encoding="utf-8")
                    if args.chat_template else None)
        render = TokenizerAgentLoopRender(
            load_served_tokenizer(args.tokenizer,
                                  revision=args.tokenizer_revision,
                                  chat_template=template))
    declarations = None
    if args.lens_serve_declarations:
        declarations = json.loads(Path(args.lens_serve_declarations).read_text())
    proxy = RoutingProxy(
        table["routes"], HTTPBackend(args.upstream),
        base_model=table["base_model"], serving_mode=table["serving_mode"],
        prompted_prefixes=table["prompted_prefixes"],
        exact_replay_strategy=table["exact_replay_strategy"],
        launch_attestation=attestation,
        agent_loop_render=render,
        petri_agent_loop_entry=bool(args.petri_agent_loop),
        base_revision=table["base_revision"],
        thinking_header=table.get(THINKING_HEADER_KEY),
        thinking_header_error=table.get(THINKING_HEADER_ERROR_KEY),
        lens_salt_policy=args.lens_salt_policy,
        lens_served_positions=table.get(LENS_SERVED_POSITIONS_KEY),
        adapter_blocks=table[ADAPTERS_KEY],
        engine_attestation=engine_attestation,
        thinking_mode=args.thinking_mode,
        lens_serve_declarations=declarations,
        organisms=table.get(ORGANISMS_KEY, ()))
    from aiohttp import web

    logging.basicConfig(level=logging.INFO)
    web.run_app(make_app(proxy), host=args.host, port=args.port)


# --- the make_engine handle (runtime.py seam) -------------------------------------


#: the bearer a client of a standup's proxy sends when the standup's
#: ``endpoint.json`` names no ``proxy_token`` — ``ProxyClientEngine``'s
#: default, named ONCE here (code-review item 10, 2026-09-11: the gate canary's
#: bootstrap sent no header on a ``None`` token while the certification always
#: sent this default to the same proxy; both now follow
#: :func:`proxy_client_token`, so they agree on whether E's proxy wants a bearer
#: the day it starts requiring one).
PROXY_CLIENT_BEARER_DEFAULT = "petri-proxy"


def proxy_client_token(endpoint: Optional[Mapping[str, Any]]) -> str:
    """THE ONE bearer rule for a client of a standup's proxy: the endpoint's
    ``proxy_token`` when it is a non-empty string, else
    :data:`PROXY_CLIENT_BEARER_DEFAULT`. E's lora-native standup records
    ``proxy_token: None`` (its proxy checks no token), so every client sends
    the default — never nothing."""
    token = (endpoint or {}).get("proxy_token") if isinstance(endpoint, Mapping) else None
    if isinstance(token, str) and token:
        return token
    return PROXY_CLIENT_BEARER_DEFAULT


def proxy_client_headers(endpoint: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """The request headers a client of a standup's proxy sends: JSON plus the
    bearer :func:`proxy_client_token` resolves."""
    return {"Content-Type": "application/json",
            "Authorization": f"Bearer {proxy_client_token(endpoint)}"}


class ProxyClientEngine:
    """The ``runtime.make_engine`` handle for a ``proxy_vllm+…`` engine:
    a synchronous OpenAI-compatible client of the tunnelled proxy
    endpoint. The petri runner hands its ``base_url`` (and the cell's
    virtual model name) to inspect_ai's target role
    (``serving/petri_target.py``); ``chat_completions`` is the direct
    call surface (the gate uses it).

    ``base_url`` is stored in the ONE canonical OpenAI form ending in
    ``/v1`` (legacy root URLs normalize at construction), so it feeds
    ``petri_target.target_client_config`` directly — the engine's own
    calls are ``<base_url>/models`` and ``<base_url>/chat/completions``,
    never a double ``/v1/v1``."""

    def __init__(self, base_url: str, *, identity: str,
                 api_key: str = PROXY_CLIENT_BEARER_DEFAULT, timeout_s: float = CLIENT_TIMEOUT_S,
                 on_request: Optional[Callable[[str, bytes, Mapping[str, str]], None]] = None):
        ident = cm.parse_serving_identity(identity)
        if ident.backend != "proxy_vllm":
            raise ProxyError(f"ProxyClientEngine serves proxy_vllm "
                             f"identities only, got {identity!r}")
        if not base_url:
            raise ProxyError("ProxyClientEngine needs a base_url")
        self.base_url = _canonical_v1_base(base_url)
        self.identity = identity
        self.name = identity  # rows stamp engine = the serving identity
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.served_models: Tuple[str, ...] = ()
        self.proxy_identity: Optional[Dict[str, Any]] = None
        #: the HTTP-boundary hook (tinycap request evidence): called with
        #: ``(url, body_bytes, headers)`` — the EXACT bytes and headers
        #: about to go on the wire — right before every POST; None =
        #: nothing observes the boundary (byte-identical behaviour)
        self.on_request = on_request

    @classmethod
    def from_manifest(cls, manifest: Any) -> "ProxyClientEngine":
        base_url = (getattr(manifest, "proxy_base_url", None)
                    or os.environ.get(PROXY_BASE_URL_ENV))
        if not base_url:
            raise ProxyError(
                f"engine {manifest.engine!r}: no proxy endpoint — set "
                f"manifest.proxy_base_url or ${PROXY_BASE_URL_ENV} to the "
                "tunnelled proxy URL (e.g. http://localhost:8814); the "
                "proxy has no default address on purpose (loud, never "
                "guessed)")
        return cls(base_url, identity=manifest.engine)

    # -- transport (urllib: zero extra deps; injectable for tests) ------------

    def _http_json(self, url: str, payload: Optional[Mapping[str, Any]],
                   headers: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
        data = None
        hdrs = {"Authorization": f"Bearer {self.api_key}"}
        if payload is not None:
            data = json.dumps(dict(payload)).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
        if headers:
            hdrs.update({str(k): str(v) for k, v in headers.items()})
        if data is not None and self.on_request is not None:
            self.on_request(url, data, dict(hdrs))
        req = urllib.request.Request(url, data=data, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:2000]
            raise ProxyError(f"proxy {url} returned {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise ProxyError(
                f"cannot reach the petri proxy at {url}: {e.reason} — is "
                "the proxy job up and the SSH tunnel open (plan §11 launch "
                "steps 2–3)?") from e

    def load(self, *, require_virtual_models: Sequence[str] = ()) -> None:
        """The make_engine load step: verify the tunnelled endpoint IS the
        petri proxy and record the served virtual models — a dead tunnel,
        a stale/foreign OpenAI endpoint (any id outside the ``petri__``
        namespace), or a listing missing a REQUIRED virtual model (the
        pending cell's) all fail HERE, before any audit call. The request
        carries the api key, so a wrong token also fails here when the
        proxy authenticates (it authenticates /v1/models exactly like
        chat)."""
        listing = self._http_json(f"{self.base_url}/models", None)
        data = listing.get("data")
        if not isinstance(data, list) or not data:
            raise ProxyError(
                f"proxy at {self.base_url} lists no served models — the "
                "routing table is empty or the endpoint is not the petri "
                "proxy")
        ids: List[str] = []
        for m in data:
            mid = m.get("id") if isinstance(m, Mapping) else None
            if not isinstance(mid, str) or not mid.startswith(VMODEL_PREFIX):
                raise ProxyError(
                    f"endpoint at {self.base_url} lists {mid!r}, outside "
                    f"the {VMODEL_PREFIX!r} virtual-model namespace — this "
                    "is not the petri proxy (a stale or foreign "
                    "OpenAI-compatible endpoint must fail HERE, before the "
                    "audit starts)")
            owned = m.get("owned_by")
            if owned is not None and owned != "petri":
                raise ProxyError(
                    f"endpoint at {self.base_url} reports {mid!r} owned by "
                    f"{owned!r}, not 'petri' — not the petri proxy")
            ids.append(mid)
        meta = listing.get("petri_proxy")
        if meta is not None and not isinstance(meta, Mapping):
            raise ProxyError(
                f"endpoint at {self.base_url} carries a malformed "
                f"petri_proxy identity block: {meta!r}")
        self.proxy_identity = dict(meta) if isinstance(meta, Mapping) else None
        missing = sorted(set(require_virtual_models) - set(ids))
        if missing:
            raise ProxyError(
                f"proxy at {self.base_url} does not serve required virtual "
                f"model(s) {missing} (served: {sorted(ids)}) — its routing "
                "table predates this cell; refusing before any audit call")
        self.served_models = tuple(sorted(ids))

    def chat_completions(self, payload: Mapping[str, Any], *,
                         headers: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
        """One chat completion; ``headers`` (tinycap's task / sample / request-id
        labels) ride beside the auth header — the usage ledger consumes them."""
        url = f"{self.base_url}/chat/completions"
        if headers is None:
            # the default call stays byte-identical to the pre-hook engine (and to
            # what every existing caller/fake of _http_json expects)
            return self._http_json(url, payload)
        return self._http_json(url, payload, headers=headers)

    def tokenize(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """The proxy's read-only ``/tokenize`` pass-through (the exact prompt
        length of a chat payload); a 404 / connection failure raises
        ``ProxyError`` like every other call."""
        return self._http_json(f"{self.base_url}/tokenize", payload)
