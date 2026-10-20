#!/usr/bin/env python
"""capctrl: stand up the per-cell steered serving launch, then wait.

An IN-TREE adaptation of the shared scale12_serving_up.py (byte-identical
bring-up: registry + routing-table export, lens vLLM launch, RoutingProxy,
atomic endpoint.json, stop-file teardown) with ONE addition for the served
loglikelihood lane: ``--max-logprobs`` (default 64), spliced onto the
composed launch exactly the way --gpu-memory-utilization already is. The
engine default is 20; the MMLU letter scorer reads the four letter
logprobs out of the first generated position's top-K and hard-errors on a
letter outside it, so the launch raises the ceiling rather than the
driver guessing. The shared /vol/repos copy is deliberately untouched —
other lanes launch through it at the engine default.

One process per Slurm task: exports the worker registry + routing table for
THIS condition's cell (plus the 'none' baseline, whose derived none__lens0
twin the steering-bites probe needs), launches the vLLM server with the lens
registry (gpu_memory_utilization 0.9 — the fully-served sweep loads no HF
target), starts the RoutingProxy, writes <out>/endpoint.json ATOMICALLY, and
then blocks until <stop-file> appears (evaluate.py's --serving-stop-cmd
touches it between the jb and benign legs), tearing the server down so the
benign leg gets the card to itself.

The bring-up is petri_families_served_integration.py's, parametrized: config
(model/revision) + cell + defenses-dir + geometry. Nothing here generates a
token; correctness is owned by the landed serving stack (salt-isolation gate,
steering-bites probe, SI-prefix hash) exactly as in the integration driver.

GEOMETRY (the per-eval ruling, 2026-09-03): ``--eval <name>`` sets the
defaults of ``--max-model-len`` / ``--max-num-seqs`` from that eval's entry
in ``sisteer.capeval.serving.geometry`` (the one table the windows live in);
explicit flags still win; with neither, the pre-table 10240 x 64 stands so an
old invocation is unchanged. The resolved geometry and its basis are logged
at startup and written into endpoint.json, and the budget position is
flagged (the standup never refuses — the launcher that chose the numbers
owns that decision, e.g. grokmt_served's --allow-over-budget).

BRING-UP AS A FUNCTION (2026-09-03): ``main`` is the stop-file wrapper around
:func:`bring_up`, which performs the whole bring-up and returns a live
:class:`Standup` (server + proxy + routing table + the launch's host/port/
token). The per-model budget + decode sweep (scripts/served_decode_sweep.py)
brings servers up through the SAME function at explicit geometries, so what it
measures is the production launch, never a look-alike.

THE SERVED REDUCTION MODE (2026-09-04): ``endpoint.json`` records
``lens_reduction_mode`` — ``per_row`` | ``stacked``, the arithmetic the
launched workers ATTEST for grouped lens reductions (every ``scheduler_
geometry.worker_reports`` entry of the steering attestation, read strictly by
:func:`attested_reduction_mode_at` through ``lens_worker.
attested_reduction_mode``) — next to ``lens_reduction_mode_error`` (why it is
None: a missing or unreadable attestation, a worker that attests none, a
disagreement). :func:`bring_up` hands the attested mode to the RoutingProxy,
which echoes it in its ``/v1/models`` identity block; an unattested mode is
passed as nothing, never guessed — a consumer that finds no mode treats the
server as per-row, so a guess here would mislabel a store.

THE FIT-SUPPORT-TRANSFER STANDUP (2026-09-07): ``--cell`` is
REPEATABLE (one value = today's behaviour; the ``none`` base is always in the
enumerated set, so ``--cell safety_refusal__psr_kl_all@thinking`` alone stands
up ``none`` + the declared cell + the derived ``none__lens0`` twin, and an
explicit ``--cell none`` beside a fitted cell is de-duplicated). A cell may
carry a serve VARIANT (``@thinking`` / ``@all_turns`` / ``@all_positions``,
``serving.serve_declaration``): the same artifact under a hashed, attested
declaration. ``--expect-object-sha256`` is repeatable as ``<tt_cell>=<sha>``
(the exact ``--cell`` spelling, variant included; variants of one base share
its sha) with the bare ``<sha>`` form meaning the single fitted cell. The
bring-up derives the turn-end scaffold iff a cell serves
``response_all_turns`` and the thinking-ON header iff a cell declares
``thinking``, reads the workers' attested ``lens_serve_declarations`` beside
the reduction mode and hands it to the proxy, records
``object_artifact_sha256_by_cell`` (the legacy scalar stays the single distinct
digest, null otherwise) and ``lens_salt_policy`` (``--lens-salt-policy``,
default ``attested``) in ``endpoint.json``, and records the served template's
``thinking_header`` fact in the routing table with ``thinking_header_error``
beside it. The table's ``lens_served_positions`` map (stamped by
``export_routing_table`` from the same resolved specs the worker registry binds
its ops to) is handed to the proxy and recorded in ``endpoint.json``: it is what
lets the proxy accept an UNDECLARED thinking-ON render on a lens cell whose
served mask is not header-located (``all`` / ``last``), and refuse as before
everywhere else.

THE SERVING PATH (2026-09-13): ``--serving-path
{lora_native,lens}`` — the ENTRY-POINT default is the NATIVE LoRA engine
(``registry_adapter.DEFAULT_STANDUP_SERVING_PATH``): :func:`bring_up` keeps its
OWN effective spec (the run's model / revision / cells / served template hash,
``--enable-lora`` stripped, the real prompt bytes or the placeholder), passes the
manifest flags through ``native_standup.effective_launch_flags``, partitions the
requested cells over a lens-policy source (an explicit ``--cell`` that is
skip-class — PSR / vector / stacked — REFUSES naming ``--serving-path lens``;
``none`` alone is a zero-adapter-only standup) and hands the shared native core
``native_standup.bring_up_native`` an :class:`EffectiveServing` with EXPLICIT
``tool_calls``, ``--reasoning-parser``, ``--sampling-clamp-override``,
``--text-only-mm``, ``--mamba-backend-contract``, ``--kv-derived-seqs``,
``--enable-log-requests`` forwarded and ``--adapter-bank`` (or the §1.9
resolution: the cluster default, else ``$SISTEER_ADAPTER_BANK_ROOT/<model-key>``).
The returned standup is the core's :class:`native_standup.LoraStandup`, which
``main``'s teardown reads through the same attribute names as :class:`Standup`.
``--serving-path lens`` runs the body below byte for byte (a lens fact the
native engine cannot attest — ``--lens-salt-policy request_unique_all``, the
GLM mrope patch — refuses by name under native).
"""
from __future__ import annotations
import argparse
import asyncio
import dataclasses
import json
import os
import re
import socket
import sys
import threading
import time
from pathlib import Path


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


# The attestation readers and the thinking-header step are the SHARED standup module's
# (sisteer.capeval.serving.standup_thinking): the cluster standup
# scripts/sdfbase/mt_serving_up.py runs the same functions. Kept here by NAME, as thin lazy
# wrappers, so every caller and test of this script reads them off this module exactly as
# before and the script's import-time surface stays free of any sisteer import (its sisteer
# imports have always been lazy — main() puts the tree on sys.path first).


def attested_resumed_prefill_policy_cells_at(path):
    """``standup_thinking.attested_resumed_prefill_policy_cells_at`` — ``(mapping,
    error)``: the per-cell RESUMED-PREFILL POLICY the launched workers attest at
    ``path``, through ``lens_worker.resumed_prefill_policy_cells_attested`` (the
    ONE strict reader); ``(None, "<why>")`` when unreadable or on a tree before
    the landing — recorded, never guessed."""
    from sisteer.capeval.serving import standup_thinking as st
    return st.attested_resumed_prefill_policy_cells_at(path)


def attested_serve_declarations_at(path):
    """``standup_thinking.attested_serve_declarations_at`` — ``(mapping, error)``:
    the per-cell SERVE DECLARATIONS the launched workers attest at ``path``
    (an attested EMPTY map is ``({}, None)``), through
    ``lens_worker.serve_declarations_attested`` (the ONE strict reader);
    ``(None, "<why>")`` when unreadable — fail closed."""
    from sisteer.capeval.serving import standup_thinking as st
    return st.attested_serve_declarations_at(path)


def attested_reduction_mode_at(path):
    """``standup_thinking.attested_reduction_mode_at`` — ``(mode, error)``: the
    REDUCTION MODE the launched workers attest at ``path`` (``per_row`` |
    ``stacked``), through ``lens_worker.attested_reduction_mode``; ``(None,
    "<why>")`` when unreadable — recorded, never guessed."""
    from sisteer.capeval.serving import standup_thinking as st
    return st.attested_reduction_mode_at(path)


#: The pre-table defaults, kept when neither --eval nor the flags are given
#: so an un-parameterised standup behaves exactly as before the per-eval
#: geometry table (the families surfaces' 10240 x 64 — over the certified
#: budget, accepted by this standup's own readiness check at launch).
LEGACY_MAX_MODEL_LEN = 10240
LEGACY_MAX_NUM_SEQS = 64


#: the standup's own note when no model key is given: the per-eval S is then
#: the Llama-certified default's, which undersells every measured model.
NO_MODEL_KEY_NOTE = ("[standup] WARNING: no --model-key — max_num_seqs sized from the "
                     "Llama-certified default budget ({budget:,}); pass --model-key <key> "
                     "({keys}) to size from the model's measured budget "
                     "(ModelBudget.seqs_at, hybrid cap applied)")


def standup_model_key_note(eval_name, model_key):
    """The startup warning ``main`` prints when a table-resolved standup has no
    model key (``None`` when a key was given, or when no eval is named — the
    legacy explicit-flags path sizes nothing from the table)."""
    if model_key is not None or eval_name is None:
        return None
    from sisteer.capeval.serving import geometry as sg
    return NO_MODEL_KEY_NOTE.format(budget=sg.BATCHED_TOKEN_BUDGET, keys=", ".join(sg.MODEL_KEYS))


def resolve_standup_geometry(eval_name, max_model_len, max_num_seqs, *,
                             allow_over_budget=False, model_key=None,
                             requested_max_tokens=None):
    """The launch geometry: explicit flags win; an unset flag comes from the
    ``--eval`` entry of the geometry table; with no eval, the legacy default.

    THE PER-MODEL BUDGET (``--model-key``, 2026-09-04): with a model key an
    unset ``--max-num-seqs`` is sized from THAT model's measured budget at the
    window (``ModelBudget.seqs_at`` — the hybrid per-slot cap applied;
    ``seqs_source="model_budget"``) and the budget guard is held to the model's
    budget, not the Llama one. Without a key the per-eval entry's Llama-sized S
    stands and ``main`` prints :data:`NO_MODEL_KEY_NOTE`. An explicit flag is
    recorded as an override whatever its value (never folded by numeric
    equality): endpoint.json says what the launcher actually passed.

    THE BUDGET GUARD (second adversarial review, 2026-09-03): with ``--eval``
    (table resolution) an explicit flag that breaks the identity or the
    budget REFUSES unless ``--allow-over-budget`` — ``--eval capability
    --max-model-len 16384`` used to be accepted as a silent 851,968-token
    non-identity launch. ONLY the legacy no-``--eval`` path (no table: the
    pre-table 10240 x 64, or explicit flags on their own) accepts an
    over-budget product, and ``describe()`` logs it as LEGACY OVER BUDGET."""
    from sisteer.capeval.serving import geometry as sg
    if eval_name is None:
        return sg.resolve_geometry(
            None, max_model_len=max_model_len, max_num_seqs=max_num_seqs,
            fallback=(LEGACY_MAX_MODEL_LEN, LEGACY_MAX_NUM_SEQS),
            allow_over_budget=True, model_key=model_key)
    # The standup records EXACTLY what it received (review S3): an explicit
    # flag is an override whatever its value — a launcher that resolved the
    # geometry itself passes only the factors whose recorded source is
    # "override" (capctrl_served / grokmt_served _geometry_argv), so the
    # sources here equal the launcher's without inferring them from numbers.
    # ``requested_max_tokens`` (the DRIVER's completion cap, when the launcher knows it):
    # a table window measured under a smaller cap REFUSES (GeometryCapExceedsBasis) —
    # the 4,096-cap multi-turn instrument passes its windows explicitly instead
    return sg.resolve_geometry(
        eval_name, max_model_len=max_model_len, max_num_seqs=max_num_seqs,
        allow_over_budget=bool(allow_over_budget), strict_identity=True,
        model_key=model_key, requested_max_tokens=requested_max_tokens)


@dataclasses.dataclass
class Standup:
    """One LIVE standup — the launched lens vLLM server plus the RoutingProxy in
    front of it — and everything a client needs to drive it and tear it down.
    ``main`` wraps exactly one of these in the stop-file wait; the served
    decode sweep (scripts/served_decode_sweep.py) brings several up in
    sequence through the SAME :func:`bring_up`, so a probe measures the launch
    the production standup performs, not a look-alike."""

    out: Path
    model: str
    revision: str
    cell: str
    harness_cell: str
    geometry: object          # sisteer.capeval.serving.geometry.ResolvedGeometry
    launch: object            # vllm_server.ServerLaunch (host/port/api_key/base_url)
    server: object            # the vllm serve child (subprocess.Popen)
    proxy_loop: object
    proxy_url: str
    table_path: Path
    tokenizer: object
    #: the serving MODALITY pins the launch composed (tinycap PB13 / PI1): a
    #: client records what it was actually served by, never what it meant to
    #: ask for — ``reasoning_parser`` None is the ordinary thinking-OFF standup
    tool_calls: bool = False
    reasoning_parser: object = None
    #: the proxy's Petri-facing agent-loop entry (``--petri-agent-loop``, ON
    #: by default; tinycap U4(b) / PW18): a lens route declares
    #: ``petri_agent_loop`` for agent-loop-structured requests whose render
    #: does not end at the header, so the served region on a tool turn is
    #: last header + tool call + tool response + generated — served identity
    #: for the tinycap driver, recorded so the client never infers it
    petri_agent_loop: bool = True
    #: the preflight-verified sha256 of the fitted cell's artifact this standup
    #: was held to (``--expect-object-sha256``; tinycap E40 H1), or None when
    #: none was given (a base / prompted cell, or a caller outside the tinycap
    #: launcher) — recorded in endpoint.json as ``object_artifact_sha256``.
    #: THE LEGACY SCALAR: the single distinct value of
    #: ``object_sha256_by_cell`` when there is exactly one, else None (the map
    #: is then the only authority)
    object_sha256: object = None
    #: every fitted lens cell's preflight-verified artifact digest, keyed by
    #: the HARNESS id in its variant spelling (``expected_artifact_digests``'s
    #: map; variants of one base share its sha) — endpoint.json's
    #: ``object_artifact_sha256_by_cell``
    object_sha256_by_cell: dict = dataclasses.field(default_factory=dict)
    #: every TrainTest cell this standup was asked for (``--cell``, repeatable,
    #: de-duplicated, ``none`` implicit) and its harness spelling
    cells: tuple = ()
    harness_cells: tuple = ()
    #: the proxy's lens cache regime (``--lens-salt-policy``) —
    #: the proxy's own record, echoed in endpoint.json and /v1/models
    lens_salt_policy: str = "attested"
    #: the served template's thinking-header fact recorded in the routing
    #: table (``chat_markers.thinking_header_facts``; ``on_ids`` / ``moves``
    #: null when the ON certification is UNAVAILABLE — see the next field)
    thinking_header: object = None
    #: why the fact's ON certification is UNAVAILABLE (code review X1): a
    #: template with a thinking switch whose ON generation header cannot be
    #: derived (Nemotron-3's OPEN ``<think>`` prompt) — recorded beside the
    #: fact in endpoint.json and /v1/models, null when the fact fully derived.
    #: The standup serves thinking-OFF regardless; only a cell DECLARING the
    #: thinking render makes the failure fatal (``standup_thinking_header``)
    thinking_header_error: object = None
    #: the per-cell SERVED POSITIONS map the routing table recorded
    #: (``proxy.lens_served_positions``: harness cell id -> [served mode], lens
    #: legs whose resolved spec states a mode) and this
    #: standup handed to the proxy — what decides whether an UNDECLARED
    #: thinking-ON render on a lens cell is accepted (no header-located op) or
    #: refused. Recorded in endpoint.json exactly as passed; null when the
    #: table carries none (every such render then refuses, fail closed)
    lens_served_positions: object = None
    #: the ENGINE-NEUTRAL served-positions map the routing table recorded beside
    #: the lens map (``proxy.SERVED_POSITIONS_KEY``, native-positions-resolve:
    #: the lens legs' stated modes — the same entries, one shape with a native
    #: standup's route attestations), what the served driver reads FIRST.
    #: Recorded in endpoint.json iff the table carries it (a table with no
    #: stated mode has no key, so a pre-landing record is byte-identical)
    served_positions: object = None
    #: 16-hex sha256 over the chat template the SERVER was launched with —
    #: ``chat_template.jinja`` in this out dir, i.e. the tokenizer's template
    #: AFTER ``pin_reasoning_template_off`` (tinycap E45 M5). Recorded in
    #: endpoint.json as ``chat_template_hash`` so a client forced to measure a
    #: render off its OWN separately loaded tokenizer can prove that tokenizer
    #: renders what this server serves — the pin is a no-op on plain templates
    #: (llama, gemma-4) and is NOT on a reasoning template (Qwen3.5, GLM-4.6V),
    #: which is exactly when the two derivations would disagree.
    chat_template_hash: object = None
    #: the proxy's UPSTREAM (proxy -> vLLM) total timeout this standup built
    #: its ``HTTPBackend`` with (``--upstream-timeout-s``), None when none was
    #: given and the proxy's own default stood — recorded in endpoint.json as
    #: ``proxy_upstream_timeout_s`` so a client can hold the served value to
    #: its own request budget (a proxy that gives up on vLLM before the client
    #: does turns every longer draw into a 504 the client re-samples:
    #: differential censoring of the slow, cap-bound draws)
    upstream_timeout_s: object = None
    #: THE TEXT-ONLY MULTIMODAL PIN record:
    #: ``vllm_server.text_only_mm_decision``'s block — whether
    #: ``--limit-mm-per-prompt {"image":0,"video":0}`` is in the launch, composed
    #: by this standup or carried by the manifest, and why not when it is not.
    #: endpoint.json carries ``text_only_mm`` (the bool) + ``text_only_mm_record``;
    #: the launcher refuses ``TEXT_ONLY_MM_UNPINNED`` for the families the
    #: contracts name. Unhashed: a launch flag, the identity geometry stands.
    text_only_mm: object = None
    #: the M-RoPE DISPATCH ATTESTATION asked of every worker at readiness
    #: (``mrope_attest.mrope_attest``) — the block, or None with the
    #: reason beside it. ``bring_up`` REFUSES (the server torn down, endpoint.json
    #: never written) when the block is not ok, so a served ``mrope_attestation``
    #: is always a passing one; the launcher refuses ``MROPE_UNATTESTED`` when it
    #: is absent for a glm4v row. Re-asked at STOP into mrope_attestation.json.
    mrope_attestation: object = None
    mrope_attestation_error: object = None
    #: the NEMOTRON FUSED-KERNEL BACKEND attestation (``mamba_backend.live_attest``;
    #: MANDATORY at every nemotron_h standup): the live per-layer worker report
    #: judged against the reviewed contract — or None with the reason (not a
    #: nemotron_h target / no contract handed in). ``bring_up`` REFUSES a failing
    #: one the same way; the launcher refuses ``NEMO_BACKEND_UNATTESTED``.
    mamba_backend_attestation: object = None
    mamba_backend_attestation_error: object = None
    #: the KV-derived shrink record (``geometry.kv_shrink_record``) when
    #: ``--kv-derived-seqs on`` re-launched at the container's KV-backed S —
    #: ``geometry`` is then the SERVED geometry (``seqs_source="kv_derived"``);
    #: None on a launch that came up at the configured S.
    kv_shrink: object = None
    #: whether the vLLM launch carried ``--enable-log-requests`` (the engine then logs the
    #: RESOLVED sampler of every request; scripts/sdfbase/record_resolved_sampler.py reads it back) —
    #: recorded so a reader tells "no request lines" apart from "the flag was never set"
    enable_log_requests: bool = False
    #: the protocol sampling CLAMP OVERRIDE record the
    #: launch was composed under (``--sampling-clamp-override``;
    #: ``vllm_server.ServerLaunch.sampling_clamp_override`` = ``contracts.
    #: SAMPLING_CLAMP_OVERRIDE``: ``{from: 0.01, to: 0.001, why}``) — the server
    #: runs through ``sisteer.capeval.serving.vllm_entry`` with vLLM's ``_MAX_TEMP``
    #: lifted so BFCL's 0.001 executes literally. None (every other standup):
    #: the key is ABSENT from endpoint.json, so the card identity is byte-identical;
    #: present, the driver hashes it into the served identity.
    sampling_clamp_override: object = None
    #: what holds the override in vLLM's spawned children (``vllm_server.
    #: ServerLaunch.sampling_clamp_override_mechanism``) — recorded beside it, unhashed
    sampling_clamp_override_mechanism: object = None

    @property
    def attestation_path(self) -> Path:
        from sisteer.capeval.serving import vllm_server as vs
        return self.out / vs.STEERING_ATTESTATION_FILE

    @property
    def server_log(self) -> Path:
        return self.out / "vllm_server.log"

    def server_rc(self):
        """The server child's exit code, ``None`` while it is alive."""
        proc = self.server
        if not hasattr(proc, "poll"):
            proc = getattr(proc, "proc", None) or getattr(proc, "process", None)
        return proc.poll() if proc is not None else None

    def endpoint(self, eval_name=None) -> dict:
        mode, mode_error = attested_reduction_mode_at(self.attestation_path)
        rp_cells, rp_error = attested_resumed_prefill_policy_cells_at(
            self.attestation_path)
        decls, decls_error = attested_serve_declarations_at(self.attestation_path)
        out = {"proxy_url": self.proxy_url, "routing_table": str(self.table_path),
                "cell": self.cell, "harness_cell": self.harness_cell,
                # every cell of this standup (repeatable --cell) in
                # both spellings, `none` included
                "cells": list(self.cells), "harness_cells": list(self.harness_cells),
                "model": self.model, "pid": os.getpid(),
                "node": socket.gethostname(),
                # the launched geometry, so a client can read the window it
                # must clamp against from the standup's own record
                "eval": eval_name, "max_model_len": self.geometry.max_model_len,
                "max_num_seqs": self.geometry.max_num_seqs,
                # the shrink record rides INSIDE the geometry block (the driver copies
                # that block into manifest.geometry verbatim; the launcher reads it there)
                "geometry": {**self.geometry.as_dict(),
                             **({"kv_shrink": self.kv_shrink} if self.kv_shrink is not None else {})},
                "kv_shrink": self.kv_shrink,
                # the resolved-sampler readback opt-in this launch carried (default false)
                "enable_log_requests": bool(self.enable_log_requests),
                # the serving MODALITY pins, so a client records what it was
                # actually served by rather than what it meant to ask for
                # (the cell manifest's thinking.reasoning_parser reads this;
                # null is the ordinary thinking-OFF standup)
                "tool_calls": bool(self.tool_calls),
                "reasoning_parser": self.reasoning_parser,
                # the proxy's Petri-facing agent-loop entry (tinycap U4(b)):
                # the driver hashes it into the served identity and refuses
                # BFCL on a lens route without it
                "agent_loop_entry": bool(self.petri_agent_loop),
                # the preflight-verified artifact digest the worker registry's
                # bytes were held to (tinycap E40 H1); null when none was given
                # — THE LEGACY SCALAR: the single distinct digest of the map
                # below, null when the map has zero or two+ distinct values
                "object_artifact_sha256": self.object_sha256,
                # every fitted lens cell's digest, keyed by harness id in its
                # variant spelling — the authority when the
                # standup serves more than one artifact
                "object_artifact_sha256_by_cell": dict(self.object_sha256_by_cell),
                # the proxy's UPSTREAM timeout this standup built its backend
                # with; null = the proxy's own default stood
                "proxy_upstream_timeout_s": self.upstream_timeout_s,
                # the proxy's lens cache regime and the served
                # template's thinking-header fact
                "lens_salt_policy": self.lens_salt_policy,
                "thinking_header": self.thinking_header,
                # ... and why its ON certification is unavailable, when it is
                # (null beside a fully derived fact; code review X1)
                "thinking_header_error": self.thinking_header_error,
                # the per-cell SERVED POSITIONS the proxy was handed — what it
                # accepts an UNDECLARED thinking-ON render against; null when
                # the routing table recorded none
                "lens_served_positions": self.lens_served_positions,
                # ... and the engine-neutral map beside it, iff the table stamped one (native-positions-resolve)
                **({"served_positions": self.served_positions} if self.served_positions is not None else {}),
                # 16-hex sha256 over the chat template this server was
                # LAUNCHED with (tinycap E45 M5): the record a client compares
                # its own loaded tokenizer against before believing a render
                # it measured locally rather than through /v1/tokenize
                "chat_template_hash": self.chat_template_hash,
                # the served lens REDUCTION MODE as the workers ATTEST it
                # (per_row | stacked; None + why when unattested — never a
                # guess), read from the launch's steering attestation. THE
                # MERGE OF THE STACKED LANDING replaced tinycap E25's
                # pass-through of ``self.proxy_identity["lens_reduction_mode"]``
                # with this read: ``bring_up`` now derives the proxy's identity
                # block FROM this same attestation, so the pass-through had
                # become an echo of a value this record already carried, while
                # the attestation is the workers' own statement. Two
                # consequences the tinycap driver's ``_standup_endpoint``
                # encodes: the key is now ALWAYS present, so "no mode" is
                # ``None`` here rather than an absent key (an absent key still
                # means a pre-landing standup, i.e. every store frozen before
                # the landing); and a ``None`` WITH a reason is the standup
                # saying it could not determine what it served — never per_row.
                "lens_reduction_mode": mode,
                "lens_reduction_mode_error": mode_error,
                # the per-cell RESUMED-PREFILL POLICY the workers ATTEST (cell
                # -> never | always | conditional), and the reason when they
                # attest none. The proxy was handed exactly this, so a client
                # can derive the KV-namespace regime any steered request of
                # this standup was minted under — cell-stable (its prefills
                # resume) only for a cell this map calls `always`; null means
                # nothing was attested and nothing shares.
                "resumed_prefill_policy_cells": rp_cells,
                "resumed_prefill_policy_error": rp_error,
                # the TEXT-ONLY MULTIMODAL PIN: True iff the launch
                # carries --limit-mm-per-prompt {"image":0,"video":0} (composed
                # here or by the manifest); the record says which and why not
                "text_only_mm": (self.text_only_mm.get("text_only_mm")
                                 if isinstance(self.text_only_mm, dict) else None),
                "text_only_mm_record": self.text_only_mm,
                # the two ATTESTATIONS asked of the workers at readiness: each
                # None when not asked, with the reason beside it; a present
                # block is a PASSING one (bring_up refuses otherwise)
                "mrope_attestation": self.mrope_attestation,
                "mrope_attestation_error": self.mrope_attestation_error,
                "mamba_backend_attestation": self.mamba_backend_attestation,
                "mamba_backend_attestation_error": self.mamba_backend_attestation_error,
                # the per-cell SERVE DECLARATIONS the workers ATTEST:
                # PRESENT whenever attested — an EMPTY map is the
                # stated "no declared cell" — and ABSENT when the workers
                # attested none, with the reason beside it (null when the map
                # is present). The proxy was handed exactly this map.
                "lens_serve_declarations_error": decls_error}
        if decls is not None:
            out["lens_serve_declarations"] = decls
        if self.sampling_clamp_override is not None:
            # bcb-bfcl-200-widen: PRESENT IFF the launch lifted the clamp
            # (--sampling-clamp-override); the driver copies it into the hashed
            # served block, so a server WITH it never pools with one WITHOUT
            out["sampling_clamp_override"] = dict(self.sampling_clamp_override)
            out["sampling_clamp_override_mechanism"] = self.sampling_clamp_override_mechanism
        return out

    def tear_down(self) -> None:
        from sisteer.capeval.serving import vllm_server as vs
        if self.proxy_loop is not None:
            try:
                self.proxy_loop.call_soon_threadsafe(self.proxy_loop.stop)
            except Exception:
                pass
        try:
            vs._shutdown_child(self.server)
        except Exception:
            pass


_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
#: the standup CLI flag every fitted cell must carry (E45 H1 boundary)
OBJECT_DIGEST_FLAG = "--expect-object-sha256"
#: the three standup flags the launcher composes (the serving modules own the
#: vocabulary; the flag strings are stated here so the launcher and the tests
#: spell them once)
TEXT_ONLY_MM_FLAG = "--text-only-mm"
#: the opt-in, the same spelling as vLLM's own flag and mt_serving_up.py's
ENABLE_LOG_REQUESTS_FLAG = "--enable-log-requests"
TEXT_ONLY_MM_MODES = ("auto", "on", "off")           # == vllm_server.TEXT_ONLY_MM_MODES (pinned by test)
MROPE_ATTEST_FLAG = "--mrope-attest"                  # BooleanOptionalAction: --no-mrope-attest opts out
MAMBA_CONTRACT_FLAG = "--mamba-backend-contract"
#: ``--kv-derived-seqs on`` — when readiness refuses for the ONE reason
#: ``vllm_server.KvCoverageShortfall`` names (every pin held; the KV cache
#: backs fewer than S full-length sequences), re-launch ONCE at
#: S = floor(the container's KV-backed concurrency), same window, and record
#: the shrink (``geometry.kv_shrink`` in endpoint.json). Off (the default):
#: the refusal stands, byte-identical to before. The Modal launcher turns it on
#: for every standup; the per-model budgets were measured on the cluster's H200.
KV_DERIVED_SEQS_FLAG = "--kv-derived-seqs"
#: the FIRST launch's engine log, kept beside the served launch's when the
#: KV-derived shrink re-launched (the shortfall's own evidence)
KV_SHORTFALL_LOG = "vllm_server.kv_shortfall.log"
MAMBA_CONTRACT_DEFAULT = "configs/capeval/mamba_backend_contract.json"   # == mamba_backend.CONTRACT_RELPATH
MROPE_STOP_REPORT_FILENAME = "mrope_attestation.json"


def serves_an_object(cell: str, harness_id: str) -> bool:
    """Whether this standup will export ARTIFACT BYTES for ``cell`` — false
    for the base cell and for a prompted one, whose whole intervention is a
    system turn. The one predicate behind "which cells need a digest"."""
    return not (cell == "none" or harness_id == "none"
                or harness_id.startswith("prompted__"))


def expected_artifact_digests(cell: str, harness_id: str, object_sha256, *,
                              required: bool = True) -> dict:
    """The ``expected_artifact_sha256`` mapping ``export_worker_registry`` is
    handed for this standup's one fitted cell (tinycap E40 H1).

    ``{harness_id: digest}`` for a fitted cell, ``{}`` for a cell that serves
    no object. Every other combination is a REFUSAL:

    * a fitted cell with NO digest (absent, empty, or whitespace) — E45 H1
      boundary. The flag was optional, so a missing value silently produced
      ``{}`` and the export bound nothing: the reviewer reproduced it exactly
      (bound swapped object → refused; the same swap with no expectation →
      the launch composed). Only ``none`` and prompted cells may omit it, and
      the refusal happens here, BEFORE the tokenizer load, the registry export
      and the launch composition, because ``bring_up`` resolves this first;
    * a digest on a cell that serves no object (there are no bytes it could
      bind);
    * a digest that is not a lowercase 64-hex sha256 (it could never match, so
      it is not an expectation).

    ``required=False`` is the ONE narrow exemption, and it is deliberately not
    reachable from the CLI (``main`` never passes it) or from the tinycap
    launcher: a standup that files no measurement — the geometry / decode /
    hook-profiling probes, whose output is timings — has no preflight to bind
    to, and there is nothing an operator could type to disable the rule for a
    run that DOES produce rows. ``bring_up`` says so on stdout when it is used.

    The tinycap launcher already refuses a fitted configuration without a
    preflight-verified digest (``_run_configuration``'s ``NO_OBJECT_DIGEST``);
    this is the same rule at the standup's OWN boundary, which is what a
    hand-run or any other caller crosses."""
    blank = object_sha256 is None or not str(object_sha256).strip()
    if not serves_an_object(cell, harness_id):
        if blank:
            return {}
        raise ValueError(
            f"{OBJECT_DIGEST_FLAG} given for cell {cell!r}, which serves no object — there "
            "are no artifact bytes to hold to that digest")
    if blank:
        if not required:
            return {}
        raise ValueError(
            f"cell {cell!r} (harness {harness_id!r}) serves a FITTED object but no "
            f"{OBJECT_DIGEST_FLAG} was given ({object_sha256!r}) — the worker registry's "
            "bytes would then be bound to nothing and an object replaced on the shared "
            "volume since the preflight would be served silently (tinycap E40 H1). Pass "
            "the preflight-verified artifact_sha256 (the launcher fills it from --pf); "
            "only `none` and prompted cells may omit it")
    digest = str(object_sha256).strip().lower()
    if not _SHA256_HEX.match(digest):
        raise ValueError(
            f"{OBJECT_DIGEST_FLAG} {object_sha256!r} is not a 64-hex sha256 digest")
    return {harness_id: digest}


def _as_cell_list(cell) -> list:
    """``--cell`` as a list: a single TrainTest id or a sequence of them,
    de-duplicated (order kept), the ``none`` base dropped (it is always in the
    enumerated set) unless it is the only cell."""
    cells = [cell] if isinstance(cell, str) else list(cell)
    if not cells or any(not isinstance(c, str) or not c.strip() for c in cells):
        raise ValueError(f"--cell needs one or more non-empty TrainTest cell ids, got {cells!r}")
    seen, out = set(), []
    for c in cells:
        c = c.strip()
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    fitted = [c for c in out if c != "none"]
    return fitted if fitted else ["none"]


def parse_object_digest_args(values, cells: list) -> dict:
    """``--expect-object-sha256`` (repeatable) -> ``{tt_cell: sha}``:
    each value is ``<tt_cell>=<sha>`` where ``<tt_cell>``
    is the exact ``--cell`` spelling (variant included), or the BARE ``<sha>``,
    accepted only when exactly one cell of ``cells`` serves an object (it then
    means that cell). A key naming a cell not in ``cells``, a duplicate key,
    or a value that is not a lowercase 64-hex sha256 refuses."""
    if values is None:
        return {}
    if isinstance(values, str):
        values = [values]
    fitted = [c for c in cells if c != "none" and not c.startswith("prompted__")]
    out: dict = {}
    for raw in values:
        if raw is None or not str(raw).strip():
            continue
        raw = str(raw).strip()
        if "=" in raw:
            key, _, digest = raw.partition("=")
            key, digest = key.strip(), digest.strip().lower()
            if key not in cells:
                raise ValueError(
                    f"{OBJECT_DIGEST_FLAG} {raw!r} names cell {key!r}, which is not one of "
                    f"this standup's --cell values {cells} — the key is the exact --cell "
                    "spelling (variant included)")
        else:
            digest = raw.lower()
            if len(fitted) != 1:
                raise ValueError(
                    f"{OBJECT_DIGEST_FLAG} {raw!r} is a bare digest but this standup has "
                    f"{len(fitted)} fitted cell(s) {fitted} — spell it <tt_cell>=<sha> per "
                    "fitted cell")
            key = fitted[0]
        if key in out:
            raise ValueError(f"{OBJECT_DIGEST_FLAG} names cell {key!r} twice")
        if not _SHA256_HEX.match(digest):
            raise ValueError(
                f"{OBJECT_DIGEST_FLAG} {raw!r}: {digest!r} is not a 64-hex sha256 digest")
        out[key] = digest
    return out


def expected_artifact_digests_for_cells(cells: list, harness_ids: list, digests: dict, *,
                                        required: bool = True) -> dict:
    """The multi-cell :func:`expected_artifact_digests`: ``{harness_id:
    digest}`` over every cell, each resolved by the one-cell rule (a fitted
    cell needs a digest unless ``required=False``; a digest on a cell serving
    no object refuses); a digest keyed to no cell refuses. Variants of one
    base are distinct keys (each variant's harness id), sharing the base's
    sha by the caller's spelling."""
    stray = sorted(set(digests) - set(cells))
    if stray:
        raise ValueError(f"{OBJECT_DIGEST_FLAG} names cell(s) {stray} not in --cell {cells}")
    out: dict = {}
    for cell, hid in zip(cells, harness_ids):
        out.update(expected_artifact_digests(cell, hid, digests.get(cell), required=required))
    return out


def legacy_object_sha256(by_cell: dict):
    """THE LEGACY SCALAR ``object_artifact_sha256``: the single distinct
    digest of ``by_cell`` when there is exactly one (every one-artifact
    standup — experiment 1's included — reads exactly as before), ``None``
    when the map has zero or two or more distinct values."""
    distinct = sorted(set(by_cell.values()))
    return distinct[0] if len(distinct) == 1 else None


def standup_thinking_header(tok, *, declares_thinking: bool):
    """``standup_thinking.standup_thinking_header`` — ``(thinking_header,
    thinking_header_error)`` for the routing table, endpoint.json and the proxy
    (code review X1): HARD (``chat_markers.thinking_header_facts``) for a
    standup with a cell DECLARING the thinking-ON render, NON-FATAL
    (``thinking_header_facts_or_error``: the partial fact + the reason) for
    every other standup, so a template whose ON header cannot be derived
    (the pinned Nemotron-3) still serves thinking-OFF."""
    from sisteer.capeval.serving import standup_thinking as st
    return st.standup_thinking_header(tok, declares_thinking=declares_thinking)


def bring_up(*, repo: Path, config_path: Path, cell, defenses_dir: Path,
             out: Path, geom, gpu_mem_util="0.9", max_logprobs: int = 64,
             ready_timeout_s: int = 1800, install_timeout_s: int = 900,
             petri_agent_loop: bool = True, tool_calls: bool = False,
             reasoning_parser=None,
             manifest="configs/capeval/manifests/petri_twosi.yaml",
             extra_launch_args=(), expect_object_sha256=None,
             require_object_digest: bool = True,
             lens_salt_policy: str = "attested",
             upstream_timeout_s=None,
             text_only_mm: str = "off",
             attest_mrope: bool = False, model_key=None,
             mamba_backend_contract=None, kv_derived_seqs: bool = False,
             # bcb-bfcl-200-widen: the protocol sampling clamp override
             # (--sampling-clamp-override <T>; None = the stock launch, byte for byte)
             sampling_clamp_override=None,
             enable_log_requests: bool = False,
             # the serving path (None -> the entry-point
             # default, registry_adapter.DEFAULT_STANDUP_SERVING_PATH = lora_native; resolved
             # lazily so this module's import-time surface stays free of sisteer imports) and
             # the native engine's adapter bank; probe_max_tokens = the core's positive-control
             # probe cap (native only; the lens body's controls are the launcher's, never this arg)
             serving_path=None, adapter_bank=None, probe_max_tokens: int = 48):
    """The bring-up ``main`` performs, as a function: registry + routing table
    for ``cell`` (plus the ``none`` baseline), the lens vLLM launch at ``geom``
    (a ResolvedGeometry — the caller resolved it, with the budget guard),
    readiness (install + scheduler + KV-coverage attestation, inside
    ``launch_server``), then the RoutingProxy. Returns the live
    :class:`Standup`; any failure after the server spawned shuts it down before
    re-raising, so a caller never inherits a stray engine on the card.
    ``extra_launch_args`` are appended to the composed argv AFTER the
    ``--gpu-memory-utilization`` / ``--max-logprobs`` splice (the sweep passes
    none; kept for a caller that needs another wrapper-owned knob).
    ``upstream_timeout_s`` (``--upstream-timeout-s``) is the proxy's UPSTREAM
    (proxy -> vLLM) total timeout: when given it is handed to
    the ``HTTPBackend`` the RoutingProxy is built over and recorded on the
    Standup (endpoint.json ``proxy_upstream_timeout_s``); when None the
    backend is built exactly as before and the proxy's own default stands
    (every other caller of this function). The tinycap launcher passes
    ``contracts.PROXY_UPSTREAM_TIMEOUT_S`` on every standup so the proxy never
    gives up on a cap-bound draw before the client's own 7,200 s budget does.
    ``reasoning_parser`` (thinking-ON standups only) is handed to
    ``build_server_launch`` unchanged — the composer validates it against the
    family's pin — and recorded on the returned Standup (endpoint.json).
    ``expect_object_sha256`` (tinycap E40 H1) is the preflight-verified sha256
    of the fitted cell's artifact: ``export_worker_registry`` compares it
    against the bytes it copies into the worker registry — the bytes the
    server loads — and refuses an object replaced since the preflight, BEFORE
    any launch is composed. Recorded in endpoint.json. It is REQUIRED for a
    fitted cell (E45 H1 boundary): the digest rule is resolved as the first
    thing this function does after the harness translation, so a fitted cell
    with no expectation refuses before the tokenizer load, the registry export
    and the launch composition — only ``none`` and prompted cells may omit
    it. ``require_object_digest=False`` exempts a standup that files NO
    measurement (the geometry / decode / hook-profiling probes, whose output is
    timings and which have no preflight to bind to). It is a LIBRARY seam only:
    ``main`` never passes it, so nothing an operator can type disables the rule
    for a run that produces rows, and the bring-up says on stdout that the
    standup is unbound whenever it is used.

    ``cell`` may be ONE TrainTest id or a SEQUENCE of them (repeatable
    ``--cell``; ``none`` implicit); ``expect_object_sha256`` one
    bare sha (the single fitted cell) or ``<tt_cell>=<sha>`` items, one per
    fitted cell; ``lens_salt_policy`` the proxy's lens cache regime
    (``attested`` = today's behaviour).

    THE THREE tinycap-full22-modal ADDITIONS, each
    OFF by default for a library caller so every existing standup composes
    byte-for-byte what it did — ``main`` turns them on (its CLI defaults):
    ``text_only_mm`` (``auto`` | ``on`` | ``off``) composes vLLM's
    ``--limit-mm-per-prompt {"image":0,"video":0}`` for a MULTIMODAL target
    (glm4v / qwen3_5: the cluster full22 battery's pin) so a text-only
    workload never builds and profiles the vision encoders — recorded in
    endpoint.json (``text_only_mm``), unhashed; ``attest_mrope`` asks every
    worker ``sisteer_mrope_report`` at readiness and REFUSES (server torn
    down, nothing served) when the reports do not prove the rotary correct
    for ``model_key`` (a non-neox M-RoPE model on the neox-only kernel);
    ``mamba_backend_contract`` (the loaded reviewed contract) makes a
    ``nemotron_h`` target attest its fused Mamba backend — every worker's
    per-layer ``sisteer_mamba_backend_report`` judged against the contract,
    REFUSED on any missing / substituted / unrecognised binding or a missing
    layer (a Nemotron cell never runs on the quiet pure-torch fallback). A
    nemotron_h target WITHOUT a contract serves with
    ``mamba_backend_attestation`` None and the reason recorded; the launcher's
    belt refuses that for a Nemotron row.

    ``serving_path`` (native-serving-default): ``lora_native`` (the default when
    None — ``registry_adapter.DEFAULT_STANDUP_SERVING_PATH``) runs THE CAPCTRL
    FRONT over the shared native core (:func:`_bring_up_native`) and returns a
    ``native_standup.LoraStandup``; ``lens`` runs the body below byte for byte
    and returns a :class:`Standup`. ``adapter_bank`` is the native engine's
    content-addressed bank (None -> the default resolution)."""
    if serving_path is None:
        from sisteer.capeval.serving.native_standup import DEFAULT_STANDUP_SERVING_PATH
        serving_path = DEFAULT_STANDUP_SERVING_PATH
    if serving_path not in ("lora_native", "lens"):
        raise ValueError(f"--serving-path {serving_path!r} is not one of ('lora_native', 'lens')")
    if serving_path == "lora_native":
        return _bring_up_native(
            repo=repo, config_path=config_path, cell=cell, defenses_dir=defenses_dir, out=out, geom=geom,
            gpu_mem_util=gpu_mem_util, max_logprobs=max_logprobs, ready_timeout_s=ready_timeout_s,
            install_timeout_s=install_timeout_s, petri_agent_loop=petri_agent_loop, tool_calls=tool_calls,
            reasoning_parser=reasoning_parser, manifest=manifest, extra_launch_args=extra_launch_args,
            expect_object_sha256=expect_object_sha256, require_object_digest=require_object_digest,
            lens_salt_policy=lens_salt_policy, upstream_timeout_s=upstream_timeout_s,
            text_only_mm=text_only_mm, attest_mrope=attest_mrope, model_key=model_key,
            mamba_backend_contract=mamba_backend_contract, kv_derived_seqs=kv_derived_seqs,
            sampling_clamp_override=sampling_clamp_override, enable_log_requests=enable_log_requests,
            adapter_bank=adapter_bank, probe_max_tokens=probe_max_tokens)
    import yaml
    from transformers import AutoConfig, AutoTokenizer
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.petri import registry_adapter as ra
    from sisteer.capeval.petri.registry_adapter import traintest_cell_source
    from sisteer.capeval.serving import lens_worker as lw
    from sisteer.capeval.serving import mamba_backend as mb
    from sisteer.capeval.serving import mrope_attest as mra
    from sisteer.capeval.serving import vllm_server as vs
    from sisteer.capeval.serving.lens_steer import export_worker_registry
    from sisteer.capeval.serving.proxy import (HTTPBackend, LENS_SALT_POLICIES,
                                               RoutingProxy,
                                               TokenizerAgentLoopRender,
                                               export_routing_table,
                                               load_routing_table, make_app)
    from sisteer.capeval.serving import serve_declaration as sd
    from sisteer.capeval.serving import standup_thinking as st
    from sisteer.chat_markers import assistant_turn_end_ids, generation_header_ids
    from sisteer.models import pin_reasoning_template_off

    repo = Path(repo).resolve()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = repo / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    model, rev = cfg["model"], cfg["model_revision"]
    tt_cells = _as_cell_list(cell)
    cell = tt_cells[-1]  # the primary (today's one-cell semantics for `cell`)
    print(f"[standup] cells={tt_cells} model={model}@{rev[:12]}", flush=True)

    # Harness translation up front: a cell the serving vocabulary cannot
    # express must refuse HERE, before a GPU minute is spent.
    harness_ids = [ra.harness_cell_id(c) for c in tt_cells]
    harness_id = harness_ids[-1]
    cells_list = ["none"]
    for hid in harness_ids:
        if hid != "none" and hid not in cells_list:
            cells_list.append(hid)
    # tinycap E40 H1 / E45 H1 boundary: the digest the registry export is held
    # to, resolved before ANYTHING is read — a bad digest is a refusal, not a
    # no-op, and so is a fitted cell with no digest at all (per fitted cell)
    expected_digests = expected_artifact_digests_for_cells(
        tt_cells, harness_ids, parse_object_digest_args(expect_object_sha256, tt_cells),
        required=bool(require_object_digest))
    object_sha256 = legacy_object_sha256(expected_digests)
    for c, hid in zip(tt_cells, harness_ids):
        if not require_object_digest and serves_an_object(c, hid) \
                and hid not in expected_digests:
            print(f"[standup] UNBOUND: cell {c} serves a fitted object and this standup was "
                  f"asked to skip the {OBJECT_DIGEST_FLAG} binding (a probe standup that "
                  "files no measurement) — the served bytes are held to nothing", flush=True)
    if lens_salt_policy not in LENS_SALT_POLICIES:
        raise ValueError(f"--lens-salt-policy {lens_salt_policy!r} is not one of "
                         f"{LENS_SALT_POLICIES}")

    tok = AutoTokenizer.from_pretrained(model, revision=rev)
    # Reasoning templates (Qwen3.5, GLM-4.6V) render thinking-OFF at every
    # seam: the registry's assistant_header_ids derive from THIS tokenizer
    # below, and unpinned they would be the thinking-ON header — matching
    # neither the fits (GLM H=5 on the closed <think></think>) nor the
    # served render (build_server_launch pins the engine OFF from the
    # same template bytes written here). The install-time header
    # verification would refuse; pinning keeps the two derivations in the
    # same regime. A no-op on plain templates (llama, gemma-4).
    pin_reasoning_template_off(tok)
    tpath = out / "chat_template.jinja"
    tpath.write_bytes(tok.chat_template.encode())
    tsha = cm.sha256_hex16(tok.chat_template.encode())

    mcfg = AutoConfig.from_pretrained(model, revision=rev)
    hidden = getattr(mcfg, "hidden_size", None) or getattr(
        getattr(mcfg, "text_config", None), "hidden_size", None)
    if not hidden:
        raise RuntimeError(f"cannot resolve hidden_size for {model}")

    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo / manifest_path
    spec0 = cm.load_petri_spec(manifest_path)
    spec = dataclasses.replace(
        spec0, target_model=model, target_revision=rev,
        serving_mode="interleaved", cells=cells_list,
        serving_launch_flags=[f for f in spec0.serving_launch_flags
                              if f != "--enable-lora"],
        scenario_bundle_hash="ab" * 8, serving_impl_commit="fedcba9876" * 4,
        chat_template_hash=tsha)
    src = traintest_cell_source(spec, repo_root=repo, config_path=cfg_path,
                                defenses_dir=Path(defenses_dir))
    bf = {k: f"# {k}\n".encode() for k in cm.petri_expected_behavior_keys(spec)}
    # The auditor/judge prompt bytes are harness-hash inputs (H10). Since
    # the Petri lineage merge the twosi manifest pins them by content hash
    # (prompt_provenance override@<hex>, bytes under
    # configs/capeval/petri/prompts/), and enumerate_cells VERIFIES the
    # declared hex against the bytes it is handed, so the placeholder
    # bytes this standup used to pass now refuse at enumerate_cells before
    # a GPU minute is spent (found by the agent-loop live control,
    # 2026-09-02). Load the real override bytes through the runner's own
    # resolver; a package-pinned provenance (bytes inside the petri venv's
    # packages, unresolvable here) keeps the placeholder, as before.
    if spec.prompt_provenance.startswith("override@"):
        from sisteer.capeval.tiers.petri_runner import resolve_prompt_files
        prompt_files = resolve_prompt_files(
            spec, override_root=repo / "configs/capeval/petri/prompts")
    else:
        prompt_files = {"auditor": b"# a\n", "judge": b"# j\n"}
    cells = dc.enumerate_cells(spec, src, behavior_files=bf,
                               prompt_files=prompt_files)
    if cells.absent:
        raise RuntimeError(f"absent cells: {[a.cell_id for a in cells.absent]}")
    # the two OPTIONAL markers: derived iff a lens cell needs them
    # — the turn-end scaffold for a `response_all_turns` cell, the thinking-ON
    # header for a cell declaring the thinking render (on a template whose switch
    # does not move the header — gemma-4 — it equals the first and is recorded
    # so; admitted since the rule 2026-09-10)
    lens_specs = st.lens_resolved_specs(cells, src)
    header_ids = tuple(generation_header_ids(tok))
    turn_end_ids = None
    if any(s.get(sd.POSITIONS_KEY) == "response_all_turns" for s in lens_specs):
        turn_end_ids = assistant_turn_end_ids(tok)
        if not turn_end_ids:
            raise RuntimeError(
                "a cell serves positions='response_all_turns' but the served template is "
                "successor-bounded (no turn terminator) — the all-turns mask cannot be "
                "served on it")
        turn_end_ids = tuple(turn_end_ids)
    declares_thinking = st.declares_thinking(lens_specs)
    on_header_ids = st.thinking_on_header_ids(tok, header_ids, declares_thinking=declares_thinking)
    # the served template's thinking-header fact: HARD for a declaring cell
    # (its ON header must certify), NON-FATAL otherwise — a thinking-OFF
    # standup must not die deriving a header it never uses (code review X1)
    thinking_header, thinking_header_error = standup_thinking_header(
        tok, declares_thinking=declares_thinking)
    if thinking_header_error is not None:
        print("[standup] thinking_header: ON certification UNAVAILABLE "
              f"({thinking_header_error}); serving thinking-OFF; a declared thinking-ON "
              "request on a lens route refuses thinking_on_unattested", flush=True)
    reg_dir = out / "lens_registry"
    export_worker_registry(
        cells, src, reg_dir,
        # the GENERATION-PROMPT header (fit-time width, id-form) — the
        # terminator-anchored located header absorbs gemma-4 prompt-side
        # scaffold and would steer outside the fit support (gemmafix).
        assistant_header_ids=header_ids,
        assistant_turn_end_ids=turn_end_ids,
        assistant_header_ids_thinking_on=on_header_ids,
        zero_op_hidden_size=int(hidden),
        # tinycap E40 H1: the bytes copied into the registry must be the bytes
        # the preflight verified — a swap since then refuses here, before any
        # launch is composed
        expected_artifact_sha256=expected_digests)
    table_path = out / "routing_table.json"
    export_routing_table(cells, src, table_path, base_model=model,
                         serving_mode="interleaved", base_revision=rev,
                         # the reason rides the table BESIDE the fact (S-P1),
                         # so a proxy rebuilt from the table alone (proxy.main)
                         # never holds a partial fact without it
                         thinking_header_error=thinking_header_error,
                         thinking_header=thinking_header)
    print(f"[standup] registry + routing table exported ({len(cells_list)} "
          f"cells, hidden={hidden})", flush=True)

    def compose(max_num_seqs: int):
        # ONE composer for the configured launch and the KV-derived re-launch:
        # build_server_launch pins max_num_batched_tokens = S x L and the fields
        # expected_scheduler_state reads, so a re-launch at another S is REBUILT,
        # never a re-spelled argv over stale fields
        return vs.build_server_launch(
            spec, port=free_port(), lens_registry_dir=reg_dir,
            chat_template_path=tpath, max_model_len=geom.max_model_len,
            max_num_seqs=max_num_seqs, tool_calls=bool(tool_calls),
            reasoning_parser=reasoning_parser,
            # bcb-bfcl-200-widen: None composes today's argv byte for byte
            sampling_clamp_override=sampling_clamp_override)
    launch = compose(geom.max_num_seqs)
    # THE TEXT-ONLY MULTIMODAL PIN: decided from the config the
    # standup already loaded (model_type + vision stack) against the composed
    # argv (a manifest's pin is never duplicated); composed tokens ride with
    # the wrapper-owned extras below and the whole decision is recorded
    family = getattr(mcfg, "model_type", None)
    tom = vs.text_only_mm_decision(text_only_mm, model_id=model, revision=rev,
                                   family=family, cmd=launch.cmd)
    tom_tokens = tuple(tom["tokens"]) if tom.get("source") == "standup" else ()
    print(f"[standup] text_only_mm={tom['text_only_mm']} source={tom.get('source')} "
          f"multimodal={tom['multimodal']} family={family!r}"
          + (f" ({tom['reason']})" if tom.get("reason") else ""), flush=True)
    # The resolved-sampler readback, ported to this standup: OPT-IN,
    # default off — `--enable-log-requests` makes vLLM 0.25.1 log the RESOLVED SamplingParams of every
    # request (one INFO line each; the prompt is a DEBUG record and is NOT logged at INFO), the only
    # in-engine readback of the sampler after the engine merged its defaults; recorded in
    # endpoint.json as `enable_log_requests` so a reader knows whether the log CAN carry the lines.
    # Off leaves the launch command byte-identical (mt_serving_up.launch_cmd_extras' rule)
    trailing = (("--gpu-memory-utilization", str(gpu_mem_util), "--max-logprobs", str(int(max_logprobs)))
                + tuple(str(a) for a in extra_launch_args) + tom_tokens
                + ((ENABLE_LOG_REQUESTS_FLAG,) if enable_log_requests else ()))

    def start(base):
        """``base`` with the trailing argv appended, and the server launched from it — the first attempt and the
        KV-derived re-launch go through the SAME lines."""
        composed = dataclasses.replace(base, cmd=tuple(base.cmd) + trailing)
        return composed, vs.launch_server(composed, log_path=out / "vllm_server.log",
                                          ready_timeout_s=ready_timeout_s, install_timeout_s=install_timeout_s)

    kv_shrink = None
    try:
        launch, server = start(launch)
    except vs.KvCoverageShortfall as short:
        # readiness refused for the ONE reason a KV-derived S may answer —
        # every pin held, the container's KV cache backs fewer than S full-length
        # sequences. Opted in: re-launch ONCE at floor(the reported concurrency),
        # same window; a second shortfall (or any other refusal) raises through.
        # Not opted in: the refusal stands exactly as before.
        if not kv_derived_seqs:
            raise
        geom, kv_shrink = kv_derived_geometry(geom, short, out)
        launch, server = start(compose(geom.max_num_seqs))
    proxy_loop = None
    mrope_att = mrope_err = None
    mamba_att = mamba_err = None
    try:
        # THE ROTARY THE ENGINE ACTUALLY RUNS (mrope_attest): asked
        # of every worker over the same authenticated RPC route as the install
        # + scheduler reports, and REFUSED — server torn down by the handler
        # below, endpoint.json never written — when the reports do not prove a
        # non-neox M-RoPE model (GLM-4V) is off the neox-only kernel
        if attest_mrope:
            mrope_att = mra.mrope_attest(launch, model_key)
            print(f"[standup] mrope attestation: ok={mrope_att['ok']} model_key={model_key!r} "
                  f"reports={json.dumps(mrope_att['reports'], default=str)[:600]}", flush=True)
            if not mrope_att["ok"]:
                raise RuntimeError(
                    "[standup] M-RoPE dispatch attestation FAILED — refusing to serve a rotary "
                    "the workers cannot prove correct:\n  - " + "\n  - ".join(mrope_att["problems"]))
        else:
            mrope_err = "not asked (attest_mrope=False: a library caller; the CLI asks by default)"
        # THE NEMOTRON FUSED-KERNEL BACKEND (mamba_backend): for a
        # nemotron_h target the live per-layer worker report is MANDATORY and
        # judged against the reviewed contract; any deviation refuses the same
        # way. Never the quiet pure-torch fallback.
        if family == mb.NEMOTRON_FAMILY:
            if mamba_backend_contract is None:
                mamba_err = ("nemotron_h target but no backend contract was handed to bring_up "
                             "(mamba_backend_contract=None) — the served Mamba backend is UNATTESTED")
                print(f"[standup] mamba backend: {mamba_err}", flush=True)
            else:
                try:
                    import importlib.metadata as _md
                    installed_vllm = _md.version("vllm")
                except Exception:  # noqa: BLE001 — the contract check then compares nothing
                    installed_vllm = None
                cprobs = mb.contract_problems(mamba_backend_contract,
                                              installed_vllm_version=installed_vllm)
                if cprobs:
                    raise RuntimeError("[standup] the Nemotron backend contract is not usable as a "
                                       "denominator:\n  - " + "\n  - ".join(cprobs))
                expected_layers = mb.expected_n_mamba_layers(mcfg)
                mamba_att = mb.live_attest(launch, mamba_backend_contract,
                                           expected_n_mamba_layers=expected_layers)
                print(f"[standup] mamba backend attestation: ok={mamba_att['ok']} "
                      f"expected_n_mamba_layers={expected_layers} "
                      f"n_reports={len(mamba_att['reports'])} "
                      f"contract={mamba_att['contract_sha256'][:12]}", flush=True)
                if not mamba_att["ok"]:
                    raise RuntimeError(
                        "[standup] Nemotron fused-kernel backend attestation FAILED — a Nemotron cell "
                        "never runs on the pure-torch fallback:\n  - "
                        + "\n  - ".join(mamba_att["problems"]))
        else:
            mamba_err = (f"not a {mb.NEMOTRON_FAMILY} target (model_type={family!r}): "
                         "no Mamba backend to attest")
        table = load_routing_table(table_path)
        # the served REDUCTION MODE the workers attested at readiness (the
        # attestation launch_server wrote next to the log), handed to the
        # proxy so its /v1/models identity block names the arithmetic a
        # client's outputs come from; nothing is handed over when
        # unattested (the block stays silent rather than guessing)
        mode, mode_error = attested_reduction_mode_at(out / vs.STEERING_ATTESTATION_FILE)
        print(f"[standup] lens_reduction_mode={mode!r}"
              + (f" ({mode_error})" if mode_error else ""), flush=True)
        # the per-cell RESUMED-PREFILL POLICY the workers attested, from the
        # SAME attestation: the proxy mints a cell-stable lens salt only for a
        # cell this map calls `always`, so a steered prefill resumes only when
        # the workers said their ops can carry it
        rp_cells, rp_error = attested_resumed_prefill_policy_cells_at(
            out / vs.STEERING_ATTESTATION_FILE)
        print("[standup] resumed_prefill_policy_cells="
              + (f"{len(rp_cells)} cell(s), "
                 f"{sorted(set(rp_cells.values()))}" if rp_cells is not None
                 else f"None ({rp_error})"), flush=True)
        # the per-cell SERVE DECLARATIONS the workers attested, from the SAME
        # attestation: the proxy validates every thinking-ON
        # declaration against this map and mints declared cells
        # request-unique; nothing attested -> None -> every thinking-ON
        # declaration refuses unattested (fail closed)
        decls, decls_error = attested_serve_declarations_at(
            out / vs.STEERING_ATTESTATION_FILE)
        print("[standup] lens_serve_declarations="
              + (f"{decls}" if decls is not None else f"None ({decls_error})"),
              flush=True)
        proxy = RoutingProxy(
            table["routes"],
            HTTPBackend(launch.base_url, api_key=launch.api_key,
                        # the proxy's upstream timeout, ONLY when the caller
                        # named one (the tinycap launcher does); the
                        # proxy's own default stands otherwise
                        **st.backend_timeout_kwargs(upstream_timeout_s)),
            base_model=model, serving_mode="interleaved",
            prompted_prefixes=table["prompted_prefixes"],
            exact_replay_strategy=None,
            launch_attestation=({lw.LENS_REDUCTION_MODE_KEY: mode}
                                if mode is not None else None),
            # nothing attested -> None -> every steered leg keeps the
            # request-unique salt it always had (fail closed)
            resumed_prefill_policy_cells=rp_cells,
            lens_serve_declarations=decls,
            lens_serve_declarations_error=(decls_error if decls is None else None),
            thinking_header=table.get("thinking_header"),
            thinking_header_error=thinking_header_error,
            # the per-cell SERVED POSITIONS the table stamped from the same
            # resolved specs the worker registry binds its ops to: the proxy
            # accepts an UNDECLARED thinking-ON render on a lens cell only
            # where this map states no served op is header-located; None (an
            # older table) keeps every such render refusing (fail closed)
            lens_served_positions=table.get("lens_served_positions"),
            lens_salt_policy=lens_salt_policy,
            # the per-cell digests the registry export was held to,
            # echoed in the proxy's identity block
            object_artifact_sha256_by_cell=dict(expected_digests),
            # the SAME tokenizer the registry's header came from, so the
            # agent-loop verifier's render probe and the worker's
            # recorded header are one vocabulary
            agent_loop_render=TokenizerAgentLoopRender(tok),
            petri_agent_loop_entry=bool(petri_agent_loop))
        proxy_port = free_port()
        proxy_loop = _start_proxy_thread(make_app(proxy), "127.0.0.1",
                                         proxy_port)
    except BaseException:
        if proxy_loop is not None:
            try:
                proxy_loop.call_soon_threadsafe(proxy_loop.stop)
            except Exception:
                pass
        try:
            vs._shutdown_child(server)
        except Exception:
            pass
        raise
    proxy_url = f"http://127.0.0.1:{proxy_port}/v1"
    return Standup(out=out, model=model, revision=rev, cell=cell,
                   harness_cell=harness_id, geometry=geom, launch=launch,
                   server=server, proxy_loop=proxy_loop, proxy_url=proxy_url,
                   table_path=table_path, tokenizer=tok,
                   tool_calls=bool(tool_calls), reasoning_parser=reasoning_parser,
                   petri_agent_loop=bool(petri_agent_loop),
                   object_sha256=object_sha256, chat_template_hash=tsha,
                   object_sha256_by_cell=dict(expected_digests),
                   cells=tuple(tt_cells), harness_cells=tuple(harness_ids),
                   lens_salt_policy=lens_salt_policy,
                   thinking_header_error=thinking_header_error,
                   thinking_header=thinking_header,
                   upstream_timeout_s=(float(upstream_timeout_s)
                                       if upstream_timeout_s is not None else None),
                   lens_served_positions=table.get("lens_served_positions"),
                   served_positions=table.get("served_positions"),
                   text_only_mm=tom,
                   mrope_attestation=mrope_att, mrope_attestation_error=mrope_err,
                   mamba_backend_attestation=mamba_att,
                   mamba_backend_attestation_error=mamba_err,
                   kv_shrink=kv_shrink,
                   # bcb-bfcl-200-widen: the record the LAUNCH was composed
                   # under (None on every stock launch) — endpoint.json carries it iff set
                   sampling_clamp_override=launch.sampling_clamp_override,
                   sampling_clamp_override_mechanism=getattr(launch, "sampling_clamp_override_mechanism", None),
                   enable_log_requests=bool(enable_log_requests))


def _bring_up_native(*, repo: Path, config_path: Path, cell, defenses_dir: Path, out: Path, geom,
                     gpu_mem_util, max_logprobs, ready_timeout_s, install_timeout_s, petri_agent_loop,
                     tool_calls, reasoning_parser, manifest, extra_launch_args, expect_object_sha256,
                     require_object_digest, lens_salt_policy, upstream_timeout_s, text_only_mm,
                     attest_mrope, model_key, mamba_backend_contract, kv_derived_seqs,
                     sampling_clamp_override, enable_log_requests, adapter_bank, probe_max_tokens=48):
    """THE CAPCTRL FRONT over the shared native core. Its OWN effective spec,
    exactly as the lens body derives it —
    the default manifest is a TEMPLATE: the run's model / revision, ``serving_mode``
    interleaved, the requested cells (``none`` always), ``--enable-lora`` stripped,
    placeholder bundle / impl hashes, the SERVED template's hash — then
    ``effective_launch_flags`` (the eager pin stripped and RECORDED), the partition
    over a lens-policy source (an explicit skip-class ``--cell`` REFUSES naming
    ``--serving-path lens``; ``none`` alone -> a zero-adapter-only standup), the
    lora-native source over the served set, and ``bring_up_native`` with EXPLICIT
    ``tool_calls`` and every forwarded option. Digests: ``--expect-object-sha256``
    REQUIRED per fitted cell exactly as today (the core holds the bytes to them;
    ``require_object_digest=False`` — the probe seam — is honoured at the digest
    grammar, but the core still refuses a lora cell it was handed no digest for:
    a native adapter's bytes are never exported unbound)."""
    import yaml
    from transformers import AutoTokenizer
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import registry_adapter as ra
    from sisteer.capeval.petri.registry_adapter import traintest_cell_source
    from sisteer.capeval.serving import native_standup as ns
    from sisteer.capeval.serving.proxy import LENS_SALT_POLICIES
    from sisteer.models import pin_reasoning_template_off

    repo = Path(repo).resolve()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = repo / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    model, rev = cfg["model"], cfg["model_revision"]
    tt_cells = _as_cell_list(cell)
    print(f"[standup] serving_path=lora_native cells={tt_cells} model={model}@{rev[:12]}", flush=True)
    if lens_salt_policy not in LENS_SALT_POLICIES:
        raise ValueError(f"--lens-salt-policy {lens_salt_policy!r} is not one of {LENS_SALT_POLICIES}")
    if lens_salt_policy != "attested":
        raise ns.StandupRefused(
            f"--lens-salt-policy {lens_salt_policy!r} is a LENS cache regime the native engine has no "
            "hook for (a real limitation: no lens salt) — pass --serving-path lens; the default "
            "'attested' is accepted and recorded not_applicable")
    if extra_launch_args:
        raise ns.StandupRefused(f"extra_launch_args {list(extra_launch_args)!r} are a lens-launch seam; the "
                                "native core composes every wrapper-owned flag itself")
    # harness translation + the digest grammar FIRST (E45 H1), before any GPU minute
    harness_ids = [ra.harness_cell_id(c) for c in tt_cells]
    cells_list, tt_list = ["none"], ["none"]
    for c, hid in zip(tt_cells, harness_ids):
        if hid != "none" and hid not in cells_list:
            cells_list.append(hid)
            tt_list.append(c)
    expected_digests = expected_artifact_digests_for_cells(
        tt_cells, harness_ids, parse_object_digest_args(expect_object_sha256, tt_cells),
        required=bool(require_object_digest))
    for c, hid in zip(tt_cells, harness_ids):
        if not require_object_digest and serves_an_object(c, hid) and hid not in expected_digests:
            print(f"[standup] UNBOUND: cell {c} serves a fitted object and this standup was asked to "
                  f"skip the {OBJECT_DIGEST_FLAG} binding — the native core refuses an unbound lora "
                  "cell", flush=True)
    tok = AutoTokenizer.from_pretrained(model, revision=rev)
    pin_reasoning_template_off(tok)
    template_bytes = tok.chat_template.encode()
    tsha = cm.sha256_hex16(template_bytes)
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo / manifest_path
    spec0 = cm.load_petri_spec(manifest_path)
    # the lens body's effective spec (:850-863), unchanged — then the native flag pass
    kept, dropped = ns.effective_launch_flags([f for f in spec0.serving_launch_flags if f != "--enable-lora"])
    spec_all = dataclasses.replace(
        spec0, target_model=model, target_revision=rev,
        serving_mode="interleaved", cells=cells_list,
        serving_launch_flags=kept,
        scenario_bundle_hash="ab" * 8, serving_impl_commit="fedcba9876" * 4,
        chat_template_hash=tsha)
    # THE PARTITION over a LENS-policy source (kind + resolved spec read; nothing exported):
    # every capctrl cell is EXPLICIT, so a skip-class cell refuses naming the escape hatch
    lens_src = traintest_cell_source(spec_all, repo_root=repo, config_path=cfg_path,
                                     defenses_dir=Path(defenses_dir))
    try:
        served_ids, skipped = ns.partition_cells_for_native(cells_list, lens_src)
    except (ra.RegistryAdapterError, ValueError) as e:
        raise ns.StandupRefused(str(e)) from e
    if skipped:
        raise ns.StandupRefused(
            f"--cell {[s.cell_id for s in skipped]} ({[(s.cell_id, s.kind, s.reason) for s in skipped]}) "
            "cannot be served on the native engine (no PEFT form, or a position-masked fit) — "
            "pass --serving-path lens")
    spec = dataclasses.replace(spec_all, cells=served_ids)
    src = traintest_cell_source(spec, repo_root=repo, config_path=cfg_path,
                                defenses_dir=Path(defenses_dir), serving_path_policy="lora_native")
    bf = {k: f"# {k}\n".encode() for k in cm.petri_expected_behavior_keys(spec)}
    if spec.prompt_provenance.startswith("override@"):
        from sisteer.capeval.tiers.petri_runner import resolve_prompt_files
        prompt_files = resolve_prompt_files(spec, override_root=repo / "configs/capeval/petri/prompts")
    else:
        prompt_files = {"auditor": b"# a\n", "judge": b"# j\n"}
    eff = ns.EffectiveServing(
        spec=spec, cell_source=src, lens_source=lens_src, served_cells=tuple(served_ids),
        cells_skipped=tuple(skipped), tokenizer=tok, chat_template_bytes=template_bytes,
        config_used=cfg_path, defenses_dir=Path(defenses_dir).resolve(), model_key=model_key,
        model=model, revision=rev, expected_digests=dict(expected_digests),
        expected_digests_source=ns.DIGESTS_SOURCE_CLI, behavior_files=bf, prompt_files=prompt_files,
        traintest_cells=tuple(tt_list), manifest_path=manifest_path, config_path=cfg_path,
        manifest_flags_dropped=tuple(dropped), front="capctrl_serving_up")
    bank = ns.resolve_adapter_bank(Path(adapter_bank) if adapter_bank else None, model_key=model_key)
    return ns.bring_up_native(
        eff, out=out, eval=getattr(geom, "eval", None), geometry=geom, tool_calls=bool(tool_calls),
        adapter_bank=bank, thinking_on=False, reasoning_parser=reasoning_parser,
        sampling_clamp_override=sampling_clamp_override, text_only_mm=text_only_mm,
        mamba_backend_contract=mamba_backend_contract, kv_derived_seqs=bool(kv_derived_seqs),
        enable_log_requests=bool(enable_log_requests), probe_max_tokens=int(probe_max_tokens),
        gpu_mem_util=gpu_mem_util,
        max_logprobs=max_logprobs, ready_timeout_s=ready_timeout_s, install_timeout_s=install_timeout_s,
        upstream_timeout_s=upstream_timeout_s, petri_agent_loop=bool(petri_agent_loop),
        attest_mrope=bool(attest_mrope), front="capctrl_serving_up",
        # capctrl's trailing argv (the lens body's spelling, minus the text-only tokens the core composes itself)
        launch_extras=(("--gpu-memory-utilization", str(gpu_mem_util), "--max-logprobs", str(int(max_logprobs)))
                       + ((ENABLE_LOG_REQUESTS_FLAG,) if enable_log_requests else ())))


def kv_derived_geometry(geom, short, out: Path):
    """The re-launch geometry after a :class:`vllm_server.KvCoverageShortfall`
    under ``--kv-derived-seqs on`` — ``geom`` (a ``ResolvedGeometry``) re-stated at
    ``S = floor(kv_backed_concurrency)`` (``geometry.kv_shrunk_geometry``) with the
    shrink record (``geometry.kv_shrink_record``: configured S, the container's
    reported concurrency + capacity + basis, the served S, the rule). The first
    launch's engine log is moved aside to :data:`KV_SHORTFALL_LOG` so the served
    launch's log is clean and the shortfall's evidence survives. Refuses (a
    ``VllmServerError``) a shortfall that shrinks to nothing (< 1 sequence) or
    does not shrink at all."""
    from sisteer.capeval.serving import geometry as sg
    from sisteer.capeval.serving import vllm_server as vs
    if not dataclasses.is_dataclass(geom):
        raise TypeError(f"a KV-derived geometry needs a ResolvedGeometry, not {type(geom).__name__}")
    first = out / "vllm_server.log"
    if first.exists():
        first.replace(out / KV_SHORTFALL_LOG)
    try:
        shrink = sg.kv_shrink_record(configured_max_num_seqs=int(geom.max_num_seqs),
                                     max_model_len=int(geom.max_model_len),
                                     kv_backed_concurrency=short.kv_backed_concurrency,
                                     kv_capacity_tokens=short.kv_capacity_tokens, basis=short.basis,
                                     first_launch_log=KV_SHORTFALL_LOG)
        new = sg.kv_shrunk_geometry(geom, shrink)
    except ValueError as e:
        raise vs.VllmServerError(
            f"KV-derived geometry impossible ({e}) — the container's KV cache backs "
            f"{short.kv_backed_concurrency:.2f} full-length sequences at L={geom.max_model_len} "
            f"({short.kv_capacity_tokens:,} tokens; {short.basis}); the original refusal: {short}") from short
    print(f"[geometry] KV-DERIVED SHRINK ({KV_DERIVED_SEQS_FLAG} on): the container's KV cache backs "
          f"{short.kv_backed_concurrency:.2f} full-length sequences at L={geom.max_model_len} "
          f"({short.kv_capacity_tokens:,} tokens; {short.basis}) < the configured S={geom.max_num_seqs} — "
          f"re-launching ONCE at S={new.max_num_seqs} (= floor); the first launch's engine log is "
          f"{KV_SHORTFALL_LOG}", flush=True)
    print(new.describe(), flush=True)
    return new, shrink


class _ScalarOrList(argparse.Action):
    """A REPEATABLE flag whose one-occurrence shape is TODAY'S: a single
    ``--cell X`` parses to the scalar ``"X"`` (every launcher before the
    fit-support-transfer landing reads ``ns.cell`` as a string), while a
    repeated ``--cell A --cell B`` parses to the list ``["A", "B"]``. Both
    downstream readers (``_as_cell_list``, ``parse_object_digest_args``)
    accept either form, so the one-cell CLI is byte-for-byte what it was and
    a multi-cell standup is one argv."""

    def __call__(self, parser, namespace, values, option_string=None):
        current = getattr(namespace, self.dest, None)
        if current is None:
            setattr(namespace, self.dest, values)
        elif isinstance(current, list):
            setattr(namespace, self.dest, current + [values])
        else:
            setattr(namespace, self.dest, [current, values])


def build_parser() -> argparse.ArgumentParser:
    """The standup's CLI (a launcher's argv is parsed by exactly this — the
    end-to-end provenance test feeds capctrl_served / grokmt_served argv
    through it and resolves the geometry the way ``main`` does)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--config", required=True,
                    help="two-SI config path, relative to --repo-dir or absolute")
    ap.add_argument("--cell", required=True, action=_ScalarOrList,
                    help="TrainTest cell id; REPEATABLE (plan §2.6) — every fitted "
                         "cell of one standup, each optionally carrying a serve "
                         "variant (@thinking / @all_turns / @all_positions); `none` "
                         "is always served and may be omitted")
    ap.add_argument("--defenses-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stop-file", required=True)
    ap.add_argument("--eval", default=None,
                    help="the served eval whose geometry entry supplies the "
                         "defaults of --max-model-len / --max-num-seqs "
                         "(sisteer.capeval.serving.geometry, measured per eval: "
                         "petri, multiturn_attacks, capability, families_served, "
                         "families_prompted, bon), as eval[:task] — a sub-entry "
                         "for a standup that serves ONE attack or only short "
                         "arms, e.g. multiturn_attacks:crescendo, capability:short, "
                         "families_served:short, bon:short; explicit flags still "
                         f"win. Absent: the legacy {LEGACY_MAX_MODEL_LEN} x "
                         f"{LEGACY_MAX_NUM_SEQS}")
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="vLLM max_model_len (default: the --eval entry's, "
                         f"else {LEGACY_MAX_MODEL_LEN})")
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="vLLM max_num_seqs (default: the --eval entry's, "
                         f"else {LEGACY_MAX_NUM_SEQS}); the composer pins "
                         "max_num_batched_tokens = max_num_seqs x max_model_len")
    ap.add_argument("--model-key", default=None,
                    help="the served model's canonical key (sisteer.capeval.serving."
                         "geometry.MODEL_KEYS: llama8b, llama3b, gemma4e4b, gemma4e2b, "
                         "qwen35_9b, qwen35_4b, glm46vflash, nemotron3nano4b): an unset "
                         "--max-num-seqs is then sized from THAT model's measured "
                         "batched-token budget at the window (hybrid per-slot cap "
                         "applied) and the budget guard is held to it; absent, the "
                         "--eval entry's Llama-sized S stands (a warning says so)")
    ap.add_argument("--allow-over-budget", action="store_true",
                    help="with --eval, accept an explicit --max-model-len / "
                         "--max-num-seqs override that breaks the identity or "
                         "exceeds the certified 393,216 batched-token budget "
                         "(refused otherwise); the legacy no---eval path is "
                         "accepted as-is and logged LEGACY OVER BUDGET")
    ap.add_argument("--gpu-mem-util", default="0.9")
    ap.add_argument(KV_DERIVED_SEQS_FLAG, choices=("on", "off"), default="off",
                    help="D110: on a readiness refusal that names NOTHING but a KV-coverage "
                         "shortfall (every no-chunking pin held; the container's KV cache backs "
                         "fewer than --max-num-seqs full-length sequences), re-launch ONCE at "
                         "S = floor(the reported KV-backed concurrency) at the same window and "
                         "record the shrink (endpoint.json geometry.kv_shrink). Off: the refusal "
                         "stands. The Modal launcher passes on (its per-model budgets were measured "
                         "on the cluster's H200; a Modal container's KV cache is ~0.5%% smaller)")
    ap.add_argument(ENABLE_LOG_REQUESTS_FLAG, action="store_true", dest="enable_log_requests",
                    help="pass vLLM's --enable-log-requests so the server logs the RESOLVED "
                         "SamplingParams of every request (one INFO line each; the prompt is a "
                         "separate DEBUG record and is NOT logged at INFO) — the only readback of the "
                         "resolved sampler at 0.25.1; scripts/sdfbase/record_resolved_sampler.py reads "
                         "it back; the tinycap launcher passes it on EVERY standup. Default OFF leaves the "
                         "launch command byte-identical. Recorded in endpoint.json as enable_log_requests")
    ap.add_argument("--max-logprobs", type=int, default=64,
                    help="vLLM --max-logprobs (engine default 20): the "
                         "ceiling on per-request top_logprobs — the MMLU "
                         "letter scorer reads top-K logprobs and refuses "
                         "a letter outside it")
    ap.add_argument("--ready-timeout-s", type=int, default=1800)
    ap.add_argument("--upstream-timeout-s", type=float, default=None,
                    help="the proxy's UPSTREAM (proxy -> vLLM) total timeout in "
                         "seconds; unset = the proxy's own default. The tinycap "
                         "launcher passes contracts.PROXY_UPSTREAM_TIMEOUT_S on "
                         "every standup so a cap-bound draw is never 504'd and "
                         "re-sampled below the client's own request budget "
                         "(D73 (d)); recorded in endpoint.json as "
                         "proxy_upstream_timeout_s")
    ap.add_argument("--petri-agent-loop", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="the proxy's Petri-facing entry (declare "
                         "petri_agent_loop for agent-loop turns on lens "
                         "routes) — ON by default, the proxy's own default; "
                         "--no-petri-agent-loop is a stated opt-out. The "
                         "render probe is always attached (this tokenizer)")
    ap.add_argument("--tool-calls", action="store_true",
                    help="AGENT modality: compose --enable-auto-tool-choice + "
                         "the manifest's pinned tool parser (vllm_server "
                         "tool_calls=True). Required for any client that sends "
                         "tools (the agent-loop control); the pinned vLLM "
                         "400s every tools-carrying request otherwise")
    ap.add_argument("--manifest",
                    default="configs/capeval/manifests/petri_twosi.yaml",
                    help="the petri manifest the serving spec derives from "
                         "(model/revision come from --config; the manifest "
                         "supplies the serving pins incl. tool_parser) — pass "
                         "the model family's own manifest for --tool-calls")
    ap.add_argument("--reasoning-parser", default=None,
                    help="THINKING-ON standups only (tinycap plan §1e/P7): "
                         "compose vLLM's --reasoning-parser so the model's "
                         "thought lands in reasoning_content and `content` "
                         "carries the answer alone. Absent (the default) is "
                         "every launch this lane has ever made — the render "
                         "is pinned OFF and there is no reasoning block to "
                         "split. Pass the served family's PINNED name "
                         "(contracts.REASONING_PARSER_BY_FAMILY, e.g. gemma4 "
                         "for gemma-4); build_server_launch refuses any other "
                         "name and refuses a family with no pin, because a "
                         "mis-fitting parser does not fail — it silently "
                         "leaves the scratchpad where every extractor "
                         "(ANSWER:, numeric match, choice, code fence) reads "
                         "it as the answer")
    ap.add_argument(OBJECT_DIGEST_FLAG, default=None, action=_ScalarOrList,
                    help="tinycap E40 H1: the preflight-verified sha256 of the fitted "
                         "--cell's artifact. The worker-registry export compares it "
                         "against the bytes it copies (the bytes the server loads) and "
                         "REFUSES an object replaced since the preflight, before any "
                         "launch is composed; recorded in endpoint.json as "
                         "object_artifact_sha256 (+ object_artifact_sha256_by_cell). "
                         "REQUIRED for a fitted cell (E45 H1 boundary: without it the "
                         "served bytes are bound to nothing); refused on a cell that "
                         "serves no object, where only `none` and prompted cells may omit "
                         "it. REPEATABLE as <tt_cell>=<sha> (the exact --cell spelling, "
                         "variant included); the bare <sha> form means the single fitted "
                         "cell")
    ap.add_argument(TEXT_ONLY_MM_FLAG, choices=list(TEXT_ONLY_MM_MODES), default="auto",
                    help="tinycap-full22-modal plan §4.1 (D7): compose vLLM's --limit-mm-per-prompt "
                         '{"image":0,"video":0} so a MULTIMODAL target (glm4v / qwen3_5 — both '
                         "pinned configs carry a vision stack) never builds and profiles its vision "
                         "encoders for a text-only workload (the cluster's qwen host-RAM OOMs were "
                         "that profiling). auto (default): composed for a multimodal target of a "
                         "family with verified tokens, nothing otherwise (recorded with the reason); "
                         "on: composed for any target; off: never. A manifest already carrying the "
                         "flag is recorded as the source, never duplicated. Recorded in endpoint.json "
                         "as text_only_mm (+ text_only_mm_record); unhashed — a launch flag")
    ap.add_argument(MROPE_ATTEST_FLAG, action=argparse.BooleanOptionalAction, default=True,
                    help="plan §4.2: ask every worker sisteer_mrope_report at readiness and REFUSE to "
                         "serve (endpoint.json never written) when the reports do not prove the rotary "
                         "correct for --model-key (GLM-4.6V-Flash's non-neox M-RoPE must show the "
                         "fused kernel reachable AND the patch applied); re-asked at STOP into "
                         "<out>/mrope_attestation.json (routed_native_calls must have moved). ON by "
                         "default; --no-mrope-attest is a stated opt-out the launcher refuses for glm4v")
    ap.add_argument(MAMBA_CONTRACT_FLAG, default=MAMBA_CONTRACT_DEFAULT,
                    help="plan §5.3: the REVIEWED Nemotron backend contract (repo-relative or absolute) "
                         "a nemotron_h standup judges its live per-layer worker report against — "
                         "mandatory for Nemotron: a missing / substituted / unrecognised kernel "
                         "binding or a missing Mamba layer REFUSES the standup (the user's rule: never "
                         "the quiet pure-torch fallback). Recorded in endpoint.json as "
                         "mamba_backend_attestation and in <out>/mamba_backend_live_report.json. A "
                         "missing file leaves a Nemotron standup UNATTESTED (recorded), which the "
                         "launcher refuses (NEMO_BACKEND_UNATTESTED)")
    ap.add_argument("--lens-salt-policy", choices=["attested", "request_unique_all"],
                    default="attested",
                    help="the proxy's lens cache regime (plan §2.4 S1): 'attested' "
                         "(default, today's behaviour — the attested resumed-prefill map "
                         "and the declared-cells rule decide) or 'request_unique_all' "
                         "(every lens leg request-unique, audit label "
                         "forced-request-unique; the experiment-2 standup). Recorded in "
                         "endpoint.json and /v1/models as lens_salt_policy")
    ap.add_argument("--serving-path", choices=["lora_native", "lens"], default=None,
                    help="native-serving-default (2026-09-13): the ENGINE this standup serves on. Absent = the "
                         "entry-point default, registry_adapter.DEFAULT_STANDUP_SERVING_PATH (lora_native: the "
                         "plugin-free CUDA-graphs-on native multi-LoRA engine behind the shared native core; a "
                         "PSR / vector --cell refuses by name). lens: today's lens vLLM standup, byte for byte")
    ap.add_argument("--adapter-bank", default=None,
                    help="native engine only: the content-addressed adapter bank (default: the cluster's "
                         "$SHARED/exp/multilora_adapters/<model-key>, else $SISTEER_ADAPTER_BANK_ROOT/<model-key> "
                         "— the tinycap launcher exports the latter and passes this flag; recorded in endpoint.json)")
    ap.add_argument("--sampling-clamp-override", type=float, default=None,
                    help="bcb-bfcl-200-widen (D13): lift vLLM 0.25.1's _MAX_TEMP clamp (0.01) to "
                         "this temperature — EXACTLY the pinned protocol value 0.001 "
                         "(contracts.BFCL_PROTOCOL_TEMPERATURE; any other value refuses) — so BFCL's "
                         "reference temperature executes literally. The server then runs through "
                         "`<serving python> -m sisteer.capeval.serving.vllm_entry --sisteer-max-temp "
                         "0.001 serve …` (the same argv tail; the wrapper asserts the wheel's clamp and "
                         "the fork multiprocess method). Passed by the tinycap launcher iff "
                         "--sampling protocol; recorded in endpoint.json as sampling_clamp_override "
                         "{from, to, why} and hashed into the served identity by the driver. Absent "
                         "(the default): the stock launch, byte for byte, and no such key")
    return ap


def resolve_from_args(args):
    """``main``'s geometry resolution from parsed args (shared with the
    provenance test): the standup's resolver with the CLI's eval / dims /
    guard / model key."""
    return resolve_standup_geometry(args.eval, args.max_model_len, args.max_num_seqs,
                                    allow_over_budget=args.allow_over_budget,
                                    model_key=args.model_key)


def main(argv=None):
    args = build_parser().parse_args(argv)

    repo = Path(args.repo_dir).resolve()
    for p in (repo / "src", repo / "scripts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    # E45 H1 boundary: the digest rule is checked at the CLI too, FIRST —
    # before the geometry is resolved, before `out` is created and long before
    # bring_up's tokenizer load. bring_up re-checks it (it is the function's
    # own boundary, for a library caller), so this is a second door on the
    # same rule, not the only one.
    from sisteer.capeval.petri import registry_adapter as ra
    try:
        tt_cells = _as_cell_list(args.cell)
        expected_artifact_digests_for_cells(
            tt_cells,
            [ra.harness_cell_id(c) for c in tt_cells],
            parse_object_digest_args(args.expect_object_sha256, tt_cells))
    except ValueError as e:
        raise SystemExit(f"[standup] refusing {OBJECT_DIGEST_FLAG}: {e}") from None

    from sisteer.capeval.serving.geometry import (GeometryIdentityBroken,
                                                  GeometryOverBudget,
                                                  UnknownModelKey)
    try:
        geom = resolve_from_args(args)
    except (GeometryOverBudget, GeometryIdentityBroken) as e:
        raise SystemExit(f"[standup] refusing the geometry: {e}")
    except UnknownModelKey as e:
        raise SystemExit(f"[standup] refusing --model-key: {e}")
    note = standup_model_key_note(args.eval, args.model_key)
    if note:
        print(note, flush=True)
    print(geom.describe(), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stop_file = Path(args.stop_file)
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = repo / cfg_path

    # the reviewed backend contract, loaded HERE (bring_up judges a
    # nemotron_h target against it; any other target ignores it). A missing
    # file is recorded, never invented — the launcher's belt refuses an
    # unattested Nemotron standup.
    from sisteer.capeval.serving import mamba_backend as mb
    from sisteer.capeval.serving import mrope_attest as mra
    contract_path = Path(args.mamba_backend_contract)
    if not contract_path.is_absolute():
        contract_path = repo / contract_path
    mamba_contract = None
    if contract_path.is_file():
        mamba_contract = mb.load_contract(contract_path)
    else:
        print(f"[standup] no Nemotron backend contract at {contract_path}: a nemotron_h standup "
              "will serve UNATTESTED (recorded; the launcher refuses it)", flush=True)

    from sisteer.capeval.serving import native_standup as ns
    from sisteer.capeval.serving import vllm_server as vs
    serving_path = args.serving_path or ns.DEFAULT_STANDUP_SERVING_PATH
    print(f"[standup] serving path: {serving_path}"
          + ("" if args.serving_path else f" (the entry-point default; --serving-path lens for the lens standup)"),
          flush=True)
    try:
        su = bring_up(repo=repo, config_path=cfg_path, cell=args.cell,
                      defenses_dir=Path(args.defenses_dir), out=out, geom=geom,
                      gpu_mem_util=args.gpu_mem_util, max_logprobs=args.max_logprobs,
                      ready_timeout_s=args.ready_timeout_s, install_timeout_s=900,
                      petri_agent_loop=bool(args.petri_agent_loop),
                      tool_calls=bool(args.tool_calls),
                      reasoning_parser=args.reasoning_parser, manifest=args.manifest,
                      expect_object_sha256=args.expect_object_sha256,
                      lens_salt_policy=args.lens_salt_policy,
                      upstream_timeout_s=args.upstream_timeout_s,
                      text_only_mm=args.text_only_mm,
                      attest_mrope=bool(args.mrope_attest), model_key=args.model_key,
                      mamba_backend_contract=mamba_contract,
                      kv_derived_seqs=(args.kv_derived_seqs == "on"),
                      sampling_clamp_override=args.sampling_clamp_override,
                      enable_log_requests=bool(args.enable_log_requests),
                      serving_path=serving_path, adapter_bank=args.adapter_bank)
    except ns.PositiveControlsFailed as e:
        # native only (the lens body never raises these): the core's rc contract at the CLI
        print(f"[standup] FAILED (exit {ns.EXIT_CONTROLS_FAILED}): {e}", flush=True)
        return ns.EXIT_CONTROLS_FAILED
    except ns.StandupRefused as e:
        print(f"[standup] refused (exit {ns.EXIT_REFUSED}): {e}", flush=True)
        return ns.EXIT_REFUSED
    except vs.VllmServerError as e:
        if serving_path != "lora_native":
            raise
        print(f"[standup] the lora-native launch could not be composed or did not become ready "
              f"(exit {ns.EXIT_REFUSED}): {e}", flush=True)
        return ns.EXIT_REFUSED
    rc = 0
    try:
        # the DURABLE copy of the Nemotron backend attestation (endpoint.json is
        # unlinked at STOP; the launcher persists this file with the logs)
        if su.mamba_backend_attestation is not None:
            (out / mb.LIVE_REPORT_FILENAME).write_text(
                json.dumps(su.mamba_backend_attestation, indent=2, default=str))
        ep = su.endpoint(args.eval)
        tmp = out / "endpoint.json.tmp"
        tmp.write_text(json.dumps(ep, indent=2))
        tmp.rename(out / "endpoint.json")   # atomic: pollers never see a torso
        print(f"[standup] proxy up at {su.proxy_url}; waiting for {stop_file}",
              flush=True)
        while not stop_file.exists():
            rc_srv = su.server_rc()
            if rc_srv is not None:
                print("[standup] FATAL: vLLM server died while serving "
                      f"(rc={rc_srv}) — see vllm_server.log", flush=True)
                rc = 3
                break
            time.sleep(5)
        print("[standup] stop requested; tearing down", flush=True)
    finally:
        # the rotary report asked AGAIN after the run's traffic —
        # for a non-neox M-RoPE model routed_native_calls must have MOVED (the
        # proof the patched route was exercised); recorded, never a refusal
        # (teardown is never refused)
        if su.mrope_attestation is not None:
            try:
                at_stop = mra.mrope_attest(su.launch, args.model_key, timeout_s=60.0)
                stop_rec = mra.stop_report(args.model_key, su.mrope_attestation, at_stop)
                (out / MROPE_STOP_REPORT_FILENAME).write_text(
                    json.dumps(stop_rec, indent=2, default=str))
                print(f"[standup] mrope attestation at STOP: ok={at_stop['ok']} "
                      f"routed_native_calls={stop_rec['routed_native_calls_at_stop']} "
                      f"moved={stop_rec['routed_native_moved']}", flush=True)
            except Exception as e:  # noqa: BLE001 — teardown must never be refused
                print(f"[standup] mrope attestation at STOP unavailable: {type(e).__name__}: {e}",
                      flush=True)
        su.tear_down()
        try:
            (out / "endpoint.json").unlink()
        except FileNotFoundError:
            pass
    print("[standup] down.", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
