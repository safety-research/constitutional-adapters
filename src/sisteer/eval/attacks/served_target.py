"""SERVED target for the attacker-driven (gated) jailbreak battery.

``scripts/jailbreak_run.py generate --target-engine served`` swaps the
in-process HF target (``LLMHFSteered`` over ``load_model_and_tokenizer``) for a
client of the lens-routed :class:`~sisteer.capeval.serving.proxy.RoutingProxy`
— the serving stack the Petri audit, the served BoN lane and the served
capability harness already run on: ONE vLLM server on ONE GPU serves every
defense of the base model as a VIRTUAL MODEL, the proxy injects the per-cell
steering selector and the per-request KV-cache salt, and the worker applies
the recorded response mask. This module presents the ``LLMHFSteered`` chat
surface the attack drivers consume (PyRIT's ``SteeredPromptTarget`` and the
hand-rolled ActorAttack loop call ``defense.chat(messages, max_new_tokens=…,
do_sample=…, temperature=…)`` and read ``defense.n_target_forwards``; the
XSTest pass calls ``defense.generate``), so PAIR / TAP / Crescendo /
ActorAttack run UNCHANGED against a served target.

Why the attack drivers need no declaration: they send ordinary chat
completions that END AT THE GENERATION HEADER (the last message is a user
turn; the server appends the assistant header), which is exactly the boundary
the trunk's response mask locates — no ``petri_response_prefill`` is involved.
Earlier assistant turns of a multi-turn chat sit BEFORE the last header and
are not steered, matching the HF path's ``response_prompt_len`` semantics.

Positive controls are NOT optional: before any attack traffic,
:func:`build_served_target` runs the per-cell salt-isolation assertion over
the FULL route set and, for a steered (lens) cell, the steering-bites probe
against its scale-0 twin through the SAME proxy, and refuses on an inert arm.
The evidence rides ``ServedChatTarget.evidence`` and is written beside
``attacker_engine.json`` by the driver.

Timing is measured HERE, per request (``timing_log``): the wall-clock of each
target call plus the server's token usage, so a served run can report the
target leg's own seconds-per-forward instead of a bundled attacker+scorer+
target number (the decomposition the HF-vs-vLLM conversion decision needs).

THE WINDOW CLAMP (per-eval geometry ruling, 2026-09-03). The served target
runs at the ``multiturn_attacks`` window of
:mod:`sisteer.capeval.serving.geometry` (8704 = the p99 of prompt+completion
over 4,598 real attack calls), so ~1.4% of calls ask for more ``max_tokens``
than fit and vLLM 400s them. ``max_model_len`` (threaded from the launcher's
resolved geometry) arms :mod:`sisteer.capeval.serving.served_window`'s policy
in :meth:`ServedChatTarget._one`: a context-length 400 is retried ONCE with
``max_tokens`` clamped to the room the SERVER-STATED window leaves (the
threaded value is provenance + a cross-check, ``window_mismatch``), and a
prompt that cannot fit raises the TYPED :class:`~sisteer.capeval.serving.
served_window.PromptOverWindow` — never an empty string standing in for a
response. The multi-turn runners catch it per goal and record
``outcome: "error:PromptOverWindow"`` exactly like an attacker-side error,
so the goal is dropped/counted by the analyzers' existing errored-goal rules
and ``--resume`` retries it. Clamped calls record ``max_tokens_clamped_from``
on the timing row; a failed call leaves a timing row tagged
``error: prompt_over_window``; ``n_clamped`` / ``n_prompt_over_window`` /
``window_mismatches`` ride ``served_target.json``, which the driver rewrites
AFTER generation (``manifest_stage: "final"``) so the counters are real.

SERVE VARIANTS AND THEIR ATTESTATION. A
``--defense`` may name a SERVE VARIANT of an object cell — ``<cell>@thinking``
(experiment 1), ``<cell>@all_turns`` / ``<cell>@all_positions`` (experiment
2's settings (2) and (3)) — a hashed cell-level serve declaration the lens
attests per cell in ``/v1/models.petri_proxy.lens_serve_declarations``
(``serve_declaration.py`` owns the vocabulary). :func:`build_served_target`
REQUIRES that attested value to equal the variant's declaration
(``serve_declaration.variant_declarations``) and the BASE cell to be ABSENT
from the map, read from the proxy's identity block — never from the routing
table's informational ``positions`` and never from the cell name. A missing
map, a ``lens_serve_declarations_error``, a missing entry or a different value
is a :class:`ServedTargetError` BEFORE any generation: an undeclared serving
is refused and counted, never run on faith, for all three variants alike.

RENDER REFUSALS ARE TERMINAL; INFRASTRUCTURE ERRORS RETRY.
Setting (2)'s proxy runs a per-item all-turns render check and 400s a
conversation whose message-derived spans and located spans disagree (the
four ``all_turns_*`` codes below). Such a refusal is this experiment's
EXPECTED outcome for that conversation, not a failure to retry:
:meth:`ServedChatTarget._one` raises the typed
:class:`AllTurnsRenderRefused` (``.code`` = the proxy's complete code),
counts it per code (``n_all_turns_refused``), tags a timing row
``error: <code>``, and the PyRIT runner latches it per goal and records
``outcome = "refused:<code>"`` (``jailbreak_artifacts.refused_outcome``) —
NOT ``error:*`` — so ``load_covered_goals`` marks the goal covered, a resume
never re-runs it, and the graders exclude and count it by code. The
context-length clamp check stays FIRST: a context-length 400 is the window's
business, never read as a render refusal. Because ``served_target.json`` is
overwritten by every process, :func:`append_target_attempt` ALSO appends one
``stage="final"`` row per process to ``served_target.attempts.jsonl`` so the
per-attempt counters survive a resume and the analysis sums them.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from ...capeval.serving import proxy as proxy_mod
from ...capeval.serving import proxy_engine as pe
from ...capeval.serving import served_window as sw
from ...capeval.tinycap import contracts as tc
from ...capeval.serving.proxy import FORBIDDEN_CLIENT_KEYS, ProxyClientEngine
from ...capeval.serving.serve_declaration import (ServeDeclarationError,
                                                   split_variant,
                                                   variant_declarations,
                                                   variant_overlay)
from ..jailbreak_artifacts import (REFUSED_OUTCOME_PREFIX, append_jsonl, iter_jsonl_rows,
                                   refused_outcome)
from . import bon_served as bs
from . import target_decode as td
from .. import request_evidence_client as rev  # request-evidence hook (off unless SISTEER_REQUEST_EVIDENCE_DIR)
from .static_common import load_covered_goals

#: The sampler keys a served request may carry under a vendor preset (the
#: full22 SAMPLED re-run) — ONE definition, in bon_served, re-exported here so
#: every served client refuses the same set.
SAMPLER_KEYS = bs.SAMPLER_KEYS

#: The steering-bites probe prompts + width — the same benign set
#: ``scripts/unified_sweep_run._served_probe_fn`` uses (kept in library code
#: here so the attack driver never imports a sibling script).
PROBE_PROMPTS = (
    "List the first five prime numbers.",
    "Explain briefly why the sky is blue.",
    "Write a haiku about a quiet library.",
    "Name three uses for a paperclip.",
    "Summarize the water cycle in two sentences.",
)
PROBE_MAX_TOKENS = 48

#: One sync chat transport: OpenAI chat payload -> chat-completion dict
#: (``ProxyClientEngine.chat_completions``; a stub in tests).
ChatFn = Callable[[Mapping[str, Any]], Mapping[str, Any]]

# --------------------------------------------------------------------------- #
# The proxy's literals, under the names this client has always used.          #
# ``proxy.py`` defines them; this client pinned the VALUES independently, so  #
# tests/test_fst_integration_contract.py asserts equality with proxy's. The   #
# names are now ALIASES of proxy's constants: one definition, the same bytes, #
# the contract test still holds.                                              #
# --------------------------------------------------------------------------- #

#: ``/v1/models.petri_proxy`` + ``endpoint.json``: the attested per-cell
#: declaration map ``{harness_cell_id: {"serve_out_of_support": [...]} |
#: {"serve_experiment": <name>}}`` and the error key beside it in
#: ``endpoint.json`` (why nothing was attested).
ATTEST_SERVE_DECLARATIONS_KEY = proxy_mod.SERVE_DECLARATIONS_KEY
ATTEST_SERVE_DECLARATIONS_ERROR_KEY = proxy_mod.SERVE_DECLARATIONS_ERROR_KEY
#: ``/v1/models.petri_proxy.thinking_header`` — the served template's certified
#: header facts ``{"switch", "moves", "off_ids", "on_ids"}``.
ATTEST_THINKING_HEADER_KEY = proxy_mod.THINKING_HEADER_KEY
#: the proxy's own record of how it mints lens salts and its closed
#: vocabulary — experiment 2's standup runs ``request_unique_all`` so
#: the settings differ in the MASK only.
ATTEST_LENS_SALT_POLICY_KEY = proxy_mod.LENS_SALT_POLICY_KEY
LENS_SALT_POLICY_ATTESTED = proxy_mod.LENS_SALT_POLICY_ATTESTED
LENS_SALT_POLICY_REQUEST_UNIQUE_ALL = proxy_mod.LENS_SALT_POLICY_REQUEST_UNIQUE_ALL
LENS_SALT_POLICIES = tuple(proxy_mod.LENS_SALT_POLICIES)
#: the audit / stamp label ``resumed_prefill_policy`` wears under
#: ``request_unique_all`` and for a declared cell under ``attested``.
SALT_POLICY_LABEL_FORCED_REQUEST_UNIQUE = proxy_mod.POLICY_FORCED_REQUEST_UNIQUE
SALT_POLICY_LABEL_DECLARED_REQUEST_UNIQUE = proxy_mod.POLICY_DECLARED_SHARES_NOTHING
#: ``endpoint.json`` / ``/v1/models.petri_proxy``: ``{harness_id (variant
#: spelling): sha256}`` for every FITTED lens cell;
#: the legacy scalar ``object_artifact_sha256`` stays beside it.
ATTEST_OBJECT_SHA256_BY_CELL_KEY = proxy_mod.OBJECT_SHA256_BY_CELL_KEY
#: the per-request thinking-ON declaration key (experiment 1). Not sent by
#: this client — a multi-turn attack driver never renders thinking-ON —
#: kept for the record.
THINKING_ON_DECLARATION_KEY = proxy_mod.THINKING_ON_KEY
#: the proxy's four per-item all-turns render-check refusal codes (HTTP 400,
#: body ``error.code``): the message-derived and located
#: span lists disagree; the lexical locator refused the render; the
#: message-derived spans could not be computed (``ChatMarkerError`` — a
#: complete fake boundary inside assistant content lands here); the request
#: could not be rendered at all. Each carries its ``all_turns_`` prefix, so
#: the goal outcome is ``refused:<code>`` verbatim.
ALL_TURNS_SPAN_MISMATCH = proxy_mod.ALL_TURNS_SPAN_MISMATCH
ALL_TURNS_RENDER_REFUSED = proxy_mod.ALL_TURNS_RENDER_REFUSED
ALL_TURNS_EXPECTED_UNAVAILABLE = proxy_mod.ALL_TURNS_EXPECTED_UNAVAILABLE
ALL_TURNS_RENDER_FAILED = proxy_mod.ALL_TURNS_RENDER_FAILED
ALL_TURNS_REFUSAL_CODES = tuple(proxy_mod.ALL_TURNS_REFUSAL_CODES)
#: the four exact terminal outcomes those codes produce (pinned by test — no
#: double prefix)
ALL_TURNS_REFUSED_OUTCOMES = tuple(refused_outcome(c) for c in ALL_TURNS_REFUSAL_CODES)

#: the append-only per-process record beside ``served_target.json``
TARGET_ATTEMPTS_FILE = "served_target.attempts.jsonl"

# --------------------------------------------------------------------------- #
# THE SERVING ENGINE ON EVERY PERSISTED SERVED ROW (native-serving-default plan  #
# §1.3, decided). ``serving_engine`` in {"lens", "lora_native"} is ADDITIVE and  #
# UNHASHED — the row-facing engine field beside the served identity, derived    #
# from the proxy's ``/v1/models`` ``petri_proxy`` identity block the target      #
# already captures (``engine.kind == "lora_native"`` -> native, else lens),     #
# NEVER from the environment. Passthrough legs (``none``, prompted) carry the  #
# same ``proxy_vllm+passthrough+…`` identity on both engines, so this field is #
# what keeps a lens-era row and a native-era row apart. A row WITHOUT          #
# it predates this landing and was lens BY CONSTRUCTION (the                    #
# ``REDUCTION_MODE_LEGACY`` pattern): every reader here treats absence as lens. #
# The HF path stamps nothing (``serving_engine_of`` -> None): byte-identical.   #
# The vocabulary is ``sisteer.capeval.tinycap.contracts``'s (torch-free), by     #
# reference — one spelling for the capability stores, Petri, BoN and here.      #
# --------------------------------------------------------------------------- #
SERVING_ENGINE_FIELD = tc.SERVED_ENGINE_FIELD
SERVING_ENGINE_LENS = tc.SERVING_ENGINE_LENS
SERVING_ENGINE_LORA_NATIVE = tc.SERVING_ENGINE_LORA_NATIVE
SERVING_ENGINES = tuple(tc.SERVING_ENGINES)
#: the manifest beside a served attack store (written by the driver)
TARGET_MANIFEST_FILE = "served_target.json"
#: the ``target_engine`` :func:`write_target_manifest` records for every served store — the value the
#: static engine-tag guard maps a SILENT row's manifest onto ``bon_served.SERVED_ENGINE``; one spelling
#: for the writer and both readers
TARGET_ENGINE_SERVED = "served"


def _check_serving_engine_value(value: Any, *, what: str) -> str:
    if value not in SERVING_ENGINES:
        raise ServedTargetError(
            f"{what}: serving_engine must be one of {list(SERVING_ENGINES)}, got {value!r}")
    return str(value)


def serving_engine_from_identity(identity: Any) -> str:
    """The engine a proxy's ``/v1/models`` ``petri_proxy`` identity block names:
    :data:`SERVING_ENGINE_LORA_NATIVE` when ``engine.kind`` is the native
    engine's (``contracts.endpoint_is_native`` — the same predicate the
    capability stores read), else :data:`SERVING_ENGINE_LENS` (a lens proxy
    publishes no ``engine`` block; a missing / None identity is a lens proxy
    predating the block). A TORN identity — a native ``engine`` block beside
    a lens ``lens_reduction_mode`` attestation, or two disagreeing spellings —
    is a :class:`ServedTargetError` (``proxy_engine.served_engine_problem``),
    never read around."""
    try:
        native = tc.endpoint_is_native(identity)
    except ValueError as e:   # engine vs serving_engine spellings disagree
        raise ServedTargetError(f"served target: torn proxy identity: {e}") from e
    if not native:
        return SERVING_ENGINE_LENS
    problem = pe.served_engine_problem(identity, expect=pe.ENGINE_EXPECT_LORA_NATIVE)
    if problem is not None:
        raise ServedTargetError(
            f"served target: the proxy identity names the native engine but is not a "
            f"native standup's: {problem}")
    return SERVING_ENGINE_LORA_NATIVE


def serving_engine_of(defense: Any) -> Optional[str]:
    """The engine ``defense`` serves on, for the row writers: its
    ``serving_engine`` attribute (:class:`ServedChatTarget`,
    ``bon_served.ServedDefense`` / ``ServedStaticDefense``), or ``None`` for
    anything without one — the in-process HF target, which stamps nothing
    (its rows stay byte-identical). A defense that STATES an engine outside
    the vocabulary is refused rather than stamped."""
    value = getattr(defense, "serving_engine", None)
    if value is None:
        return None
    return _check_serving_engine_value(value, what=f"{type(defense).__name__}.serving_engine")


def _engine_or_lens(value: Any) -> Any:
    """A stored ``serving_engine`` read: absent (None) = lens by era."""
    return SERVING_ENGINE_LENS if value is None else value


def _manifest_serving_engine(out_dir: str | os.PathLike, *,
                             manifest_name: str = TARGET_MANIFEST_FILE) -> Optional[str]:
    """The engine the store's run manifest NAMES, or ``None`` when the manifest is absent or carries
    no recognised ``serving_engine`` — never a guess. Reads through :func:`read_target_manifest`,
    the ONE run-manifest lookup every store guard shares, so an unreadable or non-object manifest is
    the same :class:`ServedTargetError` here as in the serving-engine compare and the static
    engine-tag guard; the three cannot disagree about which manifest a store has or what it says.
    Used to describe rows that state no engine of their own."""
    recorded = read_target_manifest(out_dir, manifest_name=manifest_name)
    if recorded is None:
        return None
    value = recorded.get(SERVING_ENGINE_FIELD)
    return str(value) if isinstance(value, str) and value in SERVING_ENGINES else None


def read_target_manifest(out_dir: str | os.PathLike, *,
                        manifest_name: str = TARGET_MANIFEST_FILE) -> Optional[Mapping[str, Any]]:
    """THE ONE run-manifest lookup every store guard shares: ``out_dir/<manifest_name>``
    (``served_target.json``, :func:`write_target_manifest`) parsed as an object, ``None`` when the
    file is absent. An unreadable or non-object manifest is a :class:`ServedTargetError` — its
    engine cannot be established, and a guard that read around it would admit exactly the
    relabelling it exists to refuse. The serving-engine compare
    (:func:`assert_manifest_serving_engine_compatible`) and the static engine-tag guard
    (``static_common._assert_static_engine_consistent``, which describes a SILENT row by this
    manifest) both read through here, so the two cannot disagree about which manifest a store has."""
    manifest = Path(out_dir) / manifest_name
    if not manifest.is_file():
        return None
    try:
        recorded = json.loads(manifest.read_text())
    except ValueError as e:
        raise ServedTargetError(
            f"{manifest}: unreadable run manifest ({e}); refusing to append to a store whose "
            "serving engine cannot be established") from e
    if not isinstance(recorded, Mapping):
        raise ServedTargetError(
            f"{manifest}: the run manifest is not an object ({type(recorded).__name__}); "
            "refusing to append to a store whose serving engine cannot be established")
    return recorded


def assert_store_serving_engine_compatible(out_dir: str | os.PathLike, serving_engine: Optional[str], *,
                                           responses_name: str = "responses.jsonl",
                                           manifest_name: str = TARGET_MANIFEST_FILE) -> None:
    """THE PRE-WRITE ENGINE COMPARE every served row writer runs BEFORE its
    first append and before any resume filter rewrites a file:
    refuse — with nothing written, both ``responses.jsonl`` and
    ``progress.jsonl`` byte-identical — to put rows of ``serving_engine`` into
    a store another engine drew.

    ``serving_engine`` None = the HF path: nothing is checked (it stamps
    nothing, and its stores are guarded by the ``engine`` tag alone, as
    before). Otherwise every parseable row of ``out_dir/<responses_name>``
    (a torn tail line is skipped, exactly as resume does) and, when present,
    the run manifest ``out_dir/<manifest_name>`` must name this engine — an
    ABSENT ``serving_engine`` on a row or in the manifest is LENS by era (a
    lens run onto such an in-flight store is the ordinary case and passes;
    a native attempt onto it refuses). The manifest is compared
    whenever it exists, rows or no rows: it is the owning run's own
    declaration of the store's engine, and an in-flight lens shard has
    written it before its first reply. Every row is read, not only the first:
    a well-formed store is single-engine, so the first row decides, and a
    torn / mixed store is caught rather than admitted on its first line."""
    if serving_engine is None:
        return
    if serving_engine not in SERVING_ENGINES:
        raise ValueError(f"assert_store_serving_engine_compatible: serving_engine must be one of "
                         f"{list(SERVING_ENGINES)}, got {serving_engine!r}")
    out_dir = Path(out_dir)
    responses = out_dir / responses_name
    # A SILENT ROW IS NOT EVIDENCE OF LENS WHEN THE MANIFEST SPEAKS.
    # The era inference — no `serving_engine` key => lens — is only sound when nothing better
    # exists. A store whose rows carry NO engine key at all (its producing run stamped none:
    # `engine_tag` None on that call path, so neither `engine` nor `serving_engine` was written)
    # but whose run MANIFEST names an engine is described by that manifest: it is the owning run's
    # own declaration, written at the write boundary by the run that drew the rows. Reading such a
    # store as lens turned every NATIVE static store into a lens store on its second touch and
    # refused the re-run with a message that misdescribed the store's own provenance (68 of 69
    # driver failures on one job, the passthrough `none` cells included). So: a row that STATES an
    # engine is compared literally, as before; a SILENT row is compared against the manifest's
    # engine when there is one, and only falls back to the era inference when there is not.
    manifest_engine = _manifest_serving_engine(out_dir, manifest_name=manifest_name)
    silent_default = manifest_engine if manifest_engine is not None else SERVING_ENGINE_LENS
    for rec in iter_jsonl_rows(responses):
        if not isinstance(rec, Mapping):
            continue
        stated = rec.get(SERVING_ENGINE_FIELD)
        got = silent_default if stated is None else stated
        if got != serving_engine:
            raise ServedTargetError(
                f"{responses} holds a row drawn on the {got!r} serving engine (goal "
                f"{rec.get('goal_id')!r}"
                + ("; the row records no serving_engine and the run manifest names "
                   f"{manifest_engine!r}" if SERVING_ENGINE_FIELD not in rec and manifest_engine is not None
                   else "; the row records no serving_engine and no run manifest names an engine, so it "
                        "predates the native engine and was lens by construction"
                   if SERVING_ENGINE_FIELD not in rec else "")
                + f") while this run serves on {serving_engine!r} — refusing to append: a store "
                "stays single-engine (a lens row and a native row are two measurements, and the "
                "passthrough identity cannot tell them apart); nothing was written. Run this cell "
                "into a fresh dir")
    assert_manifest_serving_engine_compatible(out_dir, serving_engine, manifest_name=manifest_name)


def assert_manifest_serving_engine_compatible(out_dir: str | os.PathLike, serving_engine: Optional[str], *,
                                              manifest_name: str = TARGET_MANIFEST_FILE) -> None:
    """The MANIFEST half of :func:`assert_store_serving_engine_compatible`, on
    its own so the manifest WRITER (:func:`write_target_manifest`) runs it at
    the write boundary: an existing ``out_dir/<manifest_name>`` must name
    ``serving_engine`` — ABSENT = lens by era; an unreadable or non-object
    manifest refuses (its engine cannot be established, and a manifest that
    would be rewritten over it is exactly the relabelling this guards).
    ``serving_engine`` None compares nothing (the HF path and a stand-in
    target that stamps no engine). Nothing is written."""
    if serving_engine is None:
        return
    if serving_engine not in SERVING_ENGINES:
        raise ValueError(f"assert_manifest_serving_engine_compatible: serving_engine must be one of "
                         f"{list(SERVING_ENGINES)}, got {serving_engine!r}")
    manifest = Path(out_dir) / manifest_name
    # ONE lookup, shared with the static engine-tag guard (read_target_manifest): same file, same parse
    recorded = read_target_manifest(out_dir, manifest_name=manifest_name)
    if recorded is not None:
        got = _engine_or_lens(recorded.get(SERVING_ENGINE_FIELD))
        if got != serving_engine:
            raise ServedTargetError(
                f"{manifest} records the store's serving engine as {got!r}"
                + (" (no serving_engine recorded: a manifest written before the native engine, lens by "
                   "construction)" if SERVING_ENGINE_FIELD not in recorded else "")
                + f" while this run serves on {serving_engine!r} — refusing to append rows of another "
                "engine to this store; nothing was written. Run this cell into a fresh dir")


#: the sampling keys a sampled call may carry beside ``temperature`` (vLLM's OpenAI
#: frontend accepts them as extra body fields; the proxy forwards unknown keys as-is).
DECODE_EXTRA_KEYS = ("top_p", "top_k", "repetition_penalty")


#: the ``top_k`` values that DISABLE top-k at the engine, both accepted by :func:`normalize_decode` and sent as given.
#: EVIDENCE FROM THE ENGINE'S OWN CODE, read 2026-09-16 from the installed wheels on the cluster's shared storage at
#: BOTH served versions -- vllm 0.25.1+cu129 (envs/vllm-attacker) and 0.26.0+cu129 (envs/twosi-vllm; the standups'
#: pin, configs/capeval/vllm-requirements.txt) -- identical at each site:
#:   * ``vllm/entrypoints/openai/chat_completion/protocol.py`` (0.25.1 L250/L607/L675; 0.26.0 L250/L633/L672):
#:     ``top_k: int | None = None``; ``if (top_k := self.top_k) is None: top_k = default_sampling_params.get("top_k", 0)``;
#:     ``SamplingParams(..., top_k=top_k)`` -- a PROVIDED 0 or -1 passes through unchanged (only None takes the default).
#:   * ``vllm/sampling_params.py`` ``_verify_args`` (0.25.1 L560-564; 0.26.0 L562-566): ``# quietly accept -1 as
#:     disabled, but prefer 0`` / ``if self.top_k < -1: raise ValueError("top_k must be 0 (disable), or at least 1")``;
#:     the only ``self.top_k = 0`` assignment (L493 / L495) is the greedy reset at ``temperature < _SAMPLING_EPS`` -- -1
#:     is never rewritten and reaches the batch as -1.
#:   * ``vllm/v1/worker/gpu_input_batch.py`` (0.25.1 L393-398; 0.26.0 L395-400): ``top_k = sampling_params.top_k;
#:     if 0 < top_k < self.vocab_size: self.top_k_reqs.add(req_id) else: top_k = self.vocab_size`` -- 0 and -1 (and any
#:     k >= vocab_size) take the SAME branch: mapped to vocab_size, not in top_k_reqs.
#:   * ``vllm/v1/sample/ops/topk_topp_sampler.py`` ``apply_top_k_top_p`` (L417-426 / L418-427): ``no_top_k_mask = k ==
#:     logits.shape[1]`` -> those rows get no top-k mask. So 0 and -1 are ONE sampler in the engine we serve.
#: History: 0 is the repo's canonical protocol constant (``tinycap.contracts.PROTOCOL_TOP_K``); this validator accepted -1
#: alone until 2026-09-16 while refusing 0 -- the two validators were INVERTED (a peer ran both through this function: the
#: canonical constant was the value the driver rejected). Reconciled on the evidence above; this is the validator the
#: driver enforces, and contracts' comment names it. tests/test_multiturn_vendor_default.py pins the agreement.
TOP_K_DISABLE_VALUES = (0, -1)


def normalize_decode(decode) -> Optional[dict]:
    """Validate a sampled-target decode block: a mapping over DECODE_EXTRA_KEYS (None
    values dropped), ``top_p`` in (0, 1], ``top_k`` a positive int or a disable value
    (:data:`TOP_K_DISABLE_VALUES`: 0 = the wheel's disable value / ``contracts.PROTOCOL_TOP_K``,
    -1 = its tolerated legacy alias; both sent as given), ``repetition_penalty`` > 0. THIS is the
    validator the driver enforces on the served target's extras (``--target-top-k``, a pin's
    extras); ``contracts.PROTOCOL_TOP_K`` is the canonical protocol constant and agrees with it
    since 2026-09-16 (before that this function refused 0). Returns None for an empty / None
    block. Refuses an unknown key loudly -- a typo here would silently run the server default."""
    if decode is None:
        return None
    if not isinstance(decode, Mapping):
        raise ServedTargetError(f"decode block must be a mapping, got {type(decode).__name__}")
    unknown = sorted(set(decode) - set(DECODE_EXTRA_KEYS))
    if unknown:
        raise ServedTargetError(f"unknown decode key(s) {unknown}; known: {list(DECODE_EXTRA_KEYS)}")
    out: dict = {}
    if decode.get("top_p") is not None:
        tp = float(decode["top_p"])
        if not (0.0 < tp <= 1.0):
            raise ServedTargetError(f"top_p must be in (0, 1], got {tp}")
        out["top_p"] = tp
    if decode.get("top_k") is not None:
        tk = int(decode["top_k"])
        if tk < -1:
            raise ServedTargetError(f"top_k must be a positive int, or a disable value in {TOP_K_DISABLE_VALUES} "
                                    f"(0 = the wheel's disable value, contracts.PROTOCOL_TOP_K; -1 = its legacy alias), got {tk}")
        out["top_k"] = tk
    if decode.get("repetition_penalty") is not None:
        rp = float(decode["repetition_penalty"])
        if rp <= 0.0:
            raise ServedTargetError(f"repetition_penalty must be > 0, got {rp}")
        out["repetition_penalty"] = rp
    return out or None


class ServedTargetError(RuntimeError):
    """A served-target wiring refusal — loud, never a silent fall-back to
    the in-process HF target (a run asked to serve must serve or refuse)."""


class AllTurnsRenderRefused(ServedTargetError):
    """The proxy REFUSED a request on its per-item all-turns render check:
    ``code`` is one of :data:`ALL_TURNS_REFUSAL_CODES`, complete
    and ``all_turns_``-prefixed. TERMINAL for the goal — the expected outcome
    of experiment 2 for a conversation the all-turns mask cannot be located
    on, never an infrastructure error to retry: the PyRIT runner latches it
    and records ``outcome = refused_outcome(code)`` (``refused:<code>``), which
    ``static_common.load_covered_goals`` treats as covered."""

    def __init__(self, message: str, *, code: str,
                 virtual_model: Optional[str] = None) -> None:
        if code not in ALL_TURNS_REFUSAL_CODES:
            raise ValueError(f"AllTurnsRenderRefused: {code!r} is not one of "
                             f"{ALL_TURNS_REFUSAL_CODES}")
        super().__init__(message)
        self.code = code
        self.virtual_model = virtual_model
        self.http_status = 400

    @property
    def outcome(self) -> str:
        """The exact ``progress.jsonl`` outcome string for this refusal."""
        return refused_outcome(self.code)


#: ``ProxyClientEngine._http_json``'s wire form: ``"proxy <url> returned <status>:
#: <json error body>"`` — the status and the proxy's JSON body in the text.
_RE_PROXY_STATUS_BODY = re.compile(r"returned (\d{3}): (.*)\Z", re.DOTALL)
_RE_CODE_FALLBACK = re.compile(r'["\']code["\']\s*:\s*["\'](all_turns_[a-z_]+)["\']')


def all_turns_refusal_code(exc: BaseException) -> Optional[str]:
    """The all-turns refusal code ``exc`` carries, or None. Recognised by the
    proxy's ``error.code`` — never by prose — in either form it reaches a
    client: an in-process refusal object with ``.code`` (+ ``.http_status``),
    or the wire form :class:`ProxyClientEngine` raises (``ProxyError("proxy
    <url> returned 400: {"error": {..., "code": ...}}")``). Anything else — a
    context-length 400 (the clamp's business, which the caller checks FIRST),
    a 409 salt refusal, a 5xx, a transport error — is None and propagates as
    the infrastructure error it is."""
    code = getattr(exc, "code", None)
    status = getattr(exc, "http_status", None)
    if isinstance(code, str):
        if code in ALL_TURNS_REFUSAL_CODES and status in (None, 400):
            return code
        return None
    m = _RE_PROXY_STATUS_BODY.search(str(exc))
    if m is None or m.group(1) != "400":
        return None
    body = m.group(2).strip()
    found: Optional[str] = None
    try:
        doc = json.loads(body)
        err = doc.get("error") if isinstance(doc, Mapping) else None
        found = err.get("code") if isinstance(err, Mapping) else None
    except ValueError:
        m2 = _RE_CODE_FALLBACK.search(body)
        found = m2.group(1) if m2 else None
    return found if found in ALL_TURNS_REFUSAL_CODES else None


class ServedChatTarget:
    """A per-cell, proxy-served defense with the ``LLMHFSteered`` surface the
    gated attacks drive. ``chat_fn`` is the proxy transport; ``virtual_model``
    selects the defense (the proxy owns routing, selector and cache_salt —
    this class never sends a reserved key)."""

    #: THE ENGINE TAG ``static_common.run_static`` keys on — ``bon_served.SERVED_ENGINE``, the
    #: same class attribute :class:`bon_served.ServedStaticDefense` declares. ``run_static``
    #: reads it OFF THE DEFENSE (``getattr(defense, "engine_tag", None)``) and its presence is
    #: what (a) stamps ``engine`` + ``serving_engine`` + the geometry stamp on every row,
    #: (b) arms both resume engine guards, and (c) switches a failed chunk onto the LOUD served
    #: error path (goals left uncovered, never finalized as ``error:<type>``). The static /
    #: sampled dispatch (jailbreak_run's replayable branch) hands ``run_static`` THIS class, not
    #: the ServedStaticDefense adapter, and until 2026-09-17 it carried no tag: every served
    #: sampled-static store came out with neither key — HF-shaped rows a resume guard, a reader
    #: and a figure could not tell from an HF store — while the multi-turn writers, keyed on
    #: ``serving_engine`` alone (``serving_engine_of``), stamped. Declared on the class so no
    #: call site has to remember it. A store written that way still resumes: its run manifest
    #: (served_target.json: target_engine + serving_engine) describes the silent rows to the
    #: engine-tag guard (static_common._assert_static_engine_consistent). Pinned by
    #: tests/test_sampled_static.py and tests/test_attack_writers_serving_engine.py.
    engine_tag = bs.SERVED_ENGINE

    def __init__(
        self,
        chat_fn: ChatFn,
        *,
        virtual_model: str,
        cell_id: str,
        system: Optional[str] = None,
        timing_log: Optional[Path] = None,
        route_info: Optional[Mapping[str, Any]] = None,
        max_model_len: Optional[int] = None,
        tokenize_fn: Optional[Callable[[Mapping[str, Any]], Mapping[str, Any]]] = None,
        decode: Optional[Mapping[str, Any]] = None,
        decode_note: Optional[str] = None,
        serving_engine: Optional[str] = None,
    ) -> None:
        if isinstance(chat_fn, (str, bytes)) or not callable(chat_fn):
            raise ServedTargetError(
                "ServedChatTarget needs a chat callable (a ProxyClientEngine's "
                "chat_completions), never a bare base_url — routing, the "
                "per-cell cache_salt and SI-prefix verification are the "
                "proxy's to own")
        if not virtual_model:
            raise ServedTargetError("ServedChatTarget needs a virtual_model")
        self._chat = chat_fn
        self.virtual_model = virtual_model
        self.cell_id = cell_id
        #: request-evidence: the ``{request_id, completion_id}`` binding of the LAST response
        #: this target returned (None with the hook off) — the multi-turn writers read it beside
        #: ``last_decode`` when they persist a reply; ``last_request_id`` / ``last_completion_id`` are
        #: the same two values as the attributes the drivers read (``getattr(defense, "last_<k>")``)
        self.last_request_binding: Optional[Mapping[str, Any]] = None
        self.last_request_id: Optional[str] = None
        self.last_completion_id: Optional[str] = None
        #: request-evidence: the item identity of the NEXT request, set by the multi-turn
        #: drivers through :meth:`set_request_context` right before each forward;
        #: consumed by the send (one context, one request)
        self._request_context: Optional[rev.Label] = None
        #: THE SERVING ENGINE this target's rows are stamped with (module
        #: constants above): read ONCE here, from the proxy's /v1/models
        #: identity block ``build_served_target`` put in ``route_info
        #: ["proxy_identity"]`` (``engine.kind == "lora_native"`` -> native,
        #: else lens — a lens proxy publishes no engine block), or stated
        #: outright by the caller (a scripted target in a test; validated).
        #: Never derived from the environment.
        if serving_engine is not None:
            self.serving_engine = _check_serving_engine_value(serving_engine, what="ServedChatTarget")
        else:
            self.serving_engine = serving_engine_from_identity(
                (route_info or {}).get("proxy_identity") if isinstance(route_info, Mapping) else None)
        #: the attack drivers print ``defense.mode``; the cell id is the
        #: served identity (there is no HF mode vocabulary here).
        self.mode = cell_id
        self.system = system
        #: an UNSEEDED vendor-preset draw is refused by _payload unless the caller declares it (jailbreak_run --multiturn-target-seed
        #: none: no per-request seed on the multi-turn target); never on by default
        self.allow_unseeded_decode = False
        self.timing_log = Path(timing_log) if timing_log else None
        self.route_info = dict(route_info or {})
        self.evidence: dict = {}
        #: JB7: the authoritative query metric, at THIS boundary.
        self.n_target_forwards = 0
        self.n_timing_rows = 0
        #: the serving window the launcher resolved (geometry.py) — arms the
        #: once-only max_tokens clamp (served_window); None still clamps
        #: against the window the server states in its rejection.
        if max_model_len is not None and (
                isinstance(max_model_len, bool) or int(max_model_len) < 1):
            raise ServedTargetError(
                f"max_model_len must be a positive int, got {max_model_len!r}")
        self.max_model_len = int(max_model_len) if max_model_len is not None else None
        #: clamp bookkeeping (served_target.json): calls served with a
        #: clamped max_tokens, calls that raised PromptOverWindow, and the
        #: distinct (threaded, server) window disagreements seen.
        self.n_clamped = 0
        self.n_prompt_over_window = 0
        self.n_window_mismatch = 0
        self.window_mismatches: list = []
        #: the proxy's read-only /tokenize (ProxyClientEngine.tokenize): how a
        #: FLOOR-ONLY rejection ("at least N input tokens") gets its exact
        #: prompt length before the clamp; None -> such a rejection is the
        #: typed error with reason floor_only_unresolvable (never a guess).
        self._tokenize = tokenize_fn if callable(tokenize_fn) else None
        self.n_tokenize_resolved = 0
        self.n_floor_only_unresolvable = 0
        #: THE TARGET'S DECODE MODE is the ``temperature`` each ``chat`` call carries
        #: (``target_decode``): None = SAMPLE with no sampling key on the request, so the
        #: server's generation config governs (the campaign default 2026-09-11 -> 2026-09-16;
        #: ``attacks.target_decode: generation_config`` since); 0.0 = greedy; t > 0 = an explicit
        #: temperature -- the vendor preset (the campaign default since 2026-09-16) arrives as
        #: ``decode=`` with its own temperature, see ``_payload``. ``last_decode["temperature_source"]``
        #: records which.
        #: THE SAMPLED-TARGET DECODE BLOCK (the GLM errored-goal re-run, 2026-09-10; the
        #: BoN lesson: a record that stores no temperature is later misread as greedy).
        #: ``decode`` = the extra sampling parameters a SAMPLING call sends beside
        #: ``temperature`` -- top_p / top_k / repetition_penalty (None = omit the key, so the
        #: server's own default governs). It applies ONLY when the caller asks to sample
        #: (do_sample=True); a greedy call (temperature 0.0) never carries them, so the
        #: default greedy path's request bytes are unchanged. ``last_decode`` is the decode
        #: block of the most recent request AS SENT, recorded on every timing row and, by the
        #: runners, on every response row.
        self.decode = normalize_decode(decode)
        self.last_decode: Optional[dict] = None
        #: an operator NOTE that rides every decode block AS SENT (the 4,096-cap multi-turn
        #: re-run, 2026-09-11): where the sampling keys a row does NOT carry come from -- e.g.
        #: "the model ships no generation_config.json; vLLM's defaults govern top_p/top_k" --
        #: so a null top_p on a sampled row is never read as "unknown". Absent (None) = no key.
        self.decode_note = str(decode_note) if decode_note else None
        #: setting (2)'s per-item render check: how many calls the
        #: proxy refused, per ``all_turns_*`` code — the per-PROCESS counter
        #: (``served_target.json`` / ``served_target.attempts.jsonl``); the
        #: per-GOAL code is latched in ``progress.jsonl`` by the runner.
        self.n_all_turns_refused: Dict[str, int] = {}
        #: requests issued under a sampler preset (``decode``), for the
        #: served_target.json ``n_sampled_requests`` field (0 on a greedy run)
        self.n_sampled_requests = 0
        #: the counters above are bumped under this lock: the escalation
        #: driver fans ``generate_detailed`` out over a thread pool, and an
        #: unguarded ``+=`` from many threads under-counts the authoritative
        #: query metric (n_target_forwards, JB7).
        self._lock = threading.Lock()

    @property
    def n_all_turns_refused_total(self) -> int:
        return sum(self.n_all_turns_refused.values())

    def _raise_if_all_turns_refused(self, exc: BaseException, *, t0: float,
                                    n_messages: int, max_new_tokens: int) -> None:
        """Convert a proxy all-turns render refusal into the TYPED
        :class:`AllTurnsRenderRefused` (counted per code, a timing row tagged
        ``error: <code>``); any other exception returns and the caller
        re-raises it unchanged. Called only after the context-length clamp
        check found nothing to clamp (that check stays first)."""
        code = all_turns_refusal_code(exc)
        if code is None:
            return
        self.n_all_turns_refused[code] = self.n_all_turns_refused.get(code, 0) + 1
        if self.timing_log is not None:
            append_jsonl(self.timing_log, {
                "t_start": t0,
                "latency_s": round(time.time() - t0, 4),
                "virtual_model": self.virtual_model,
                "n_messages": int(n_messages),
                "max_tokens": int(max_new_tokens),
                "finish_reason": None,
                "error": code,
                "completion_tokens": 0,
                "intervention_hash": None,
            })
            self.n_timing_rows += 1
        raise AllTurnsRenderRefused(
            f"{code}: the proxy refused this conversation on its per-item "
            f"all-turns render check for vmodel {self.virtual_model!r} (TERMINAL "
            f"for the goal: outcome {refused_outcome(code)}): {str(exc)[:600]}",
            code=code, virtual_model=self.virtual_model) from exc

    def _plan(self, exc: BaseException, req: Mapping[str, Any],
              min_completion: Optional[int] = None) -> Optional[sw.ClampPlan]:
        """The clamp plan for a failed request (None when the failure is not a
        context-length 400). A floor-only rejection is resolved through
        /tokenize on the IDENTICAL messages; without a transport, or on any
        tokenize failure, it becomes the typed floor_only_unresolvable.
        ``min_completion`` = the requested cap makes ANY overflow a FAIL (the
        strict multi-turn policy: room < the cap is never a clamped retry)."""
        kw = {} if min_completion is None else {"min_completion": int(min_completion)}
        plan = sw.plan_for_exception(exc, requested_max_tokens=req["max_tokens"],
                                     window=self.max_model_len, **kw)
        if plan is None or not plan.needs_exact_prompt_tokens:
            return plan
        if self._tokenize is None:
            with self._lock:
                self.n_floor_only_unresolvable += 1
            return plan.unresolvable("no /tokenize transport on this target")
        try:
            count = sw.exact_prompt_tokens(self._tokenize(sw.tokenize_payload(req)))
        except Exception as exc2:  # noqa: BLE001 — 404 / connection / shape: never guess
            with self._lock:
                self.n_floor_only_unresolvable += 1
            return plan.unresolvable(f"/tokenize failed: {type(exc2).__name__}: "
                                     f"{str(exc2)[:200]}")
        with self._lock:
            self.n_tokenize_resolved += 1
        return plan.with_exact_prompt_tokens(count)

    def _note_mismatch(self, plan: sw.ClampPlan) -> None:
        """Log LOUDLY (once per distinct pair) and record a disagreement
        between the threaded window and the server's stated one."""
        mm = plan.mismatch_record()
        if mm is None:
            return
        with self._lock:
            self.n_window_mismatch += 1
            fresh = mm not in self.window_mismatches
            if fresh:
                self.window_mismatches.append(mm)
        if fresh:
            print(f"[served-target] WINDOW MISMATCH for {self.virtual_model}: the "
                  f"launcher threaded max_model_len={mm['threaded']} but the server "
                  f"states {mm['server']} — the server's value governs the clamp; "
                  "fix the launcher's geometry provenance", flush=True)

    # -- request-evidence: the multi-turn drivers' seam ---------------------- #
    def set_request_context(self, *, kind: str, goal_id: Any = None, turn: Any = None, attack: Optional[str] = None,
                            trial: Any = None, family: Optional[str] = None, sample_idx: Any = None, **_more: Any) -> None:
        """The caller's seam (``pyrit_adapters.SteeredPromptTarget(request_context_fn=)`` and
        ActorAttack's ``chat_fn``): the item identity of the request that FOLLOWS — ``kind="multi_turn"``
        with ``attack`` / ``goal_id`` / ``turn`` (the seed's turn index), or ``kind="static"`` with
        ``family`` / ``goal_id`` / ``sample_idx``. The label (the store's canonical task through
        ``request_evidence_client.store_task_name``, ``<goal_id>/<idx>``) is set here and consumed by the
        next send; with the hook OFF it is a no-op (nothing is labelled, nothing recorded)."""
        if not rev.enabled():
            self._request_context = None
            return
        if kind == rev.LEG_MULTI_TURN:
            task, idx = rev.store_task_name(rev.LEG_MULTI_TURN, attack), turn
        elif kind == rev.LEG_STATIC:
            task, idx = rev.store_task_name(rev.LEG_STATIC, family), sample_idx
        else:
            raise ServedTargetError(f"set_request_context: unknown kind {kind!r} (multi_turn | static)")
        if goal_id is None or idx is None:
            raise ServedTargetError(f"set_request_context({kind}): needs goal_id and the turn / sample index, got {goal_id!r} / {idx!r}")
        # the ONE label constructor validates the index on its ORIGINAL value (`int(True)` is 1)
        self._request_context = rev.make_label(task, str(goal_id), idx)

    def _take_request_context(self) -> Optional[rev.Label]:
        lab, self._request_context = self._request_context, None
        return lab

    # -- request construction ---------------------------------------------- #
    def _payload(self, messages: Sequence[Mapping[str, Any]], *,
                 max_new_tokens: int, do_sample: bool, temperature: Optional[float],
                 sampling_seed: Optional[int],
                 decode: Optional[Mapping[str, Any]] = None) -> dict:
        """The chat-completion body. ``temperature`` ``None`` with ``do_sample``
        = THE SERVER'S GENERATION CONFIG GOVERNS (main's 4,096-cap multi-turn
        instrument, ``target_decode.GENERATION_CONFIG``): no sampling key
        travels. ``decode`` (a vendor sampler preset — the full22 SAMPLED
        re-run's static families and its over-refusal controls' clients) puts
        EXACTLY the preset's keys on the body (SAMPLER_KEYS only; its
        temperature replaces the positional one) and needs ``do_sample=True``
        plus a per-item ``sampling_seed``; a key outside the sampler set, an
        unseeded sampled draw, or a preset on a greedy call is refused, and
        the target's own ``self.decode`` extras stand aside for it (one decode
        authority per call). Without either the body is byte-identical to
        before."""
        msgs = [dict(m) for m in messages]
        if self.system:
            # prompted / stacked cells: the SI is the AUTHORITATIVE system
            # message (the proxy hash-verifies it, H8); attack-supplied system
            # content is folded into user text exactly as the HF path does.
            from ..jailbreak import install_authoritative_system

            msgs = install_authoritative_system(msgs, self.system)
        req: dict = {
            "model": self.virtual_model,
            "messages": msgs,
            "max_tokens": int(max_new_tokens),
        }
        if do_sample and temperature is None:
            # THE SERVER'S GENERATION CONFIG GOVERNS (target_decode.GENERATION_CONFIG: the
            # campaign default 2026-09-11 -> 2026-09-16, `attacks.target_decode: generation_config`
            # since; the vendor preset -- the default since 2026-09-16 -- arrives as `decode=`
            # below): the request carries NO sampling key, so vLLM fills temperature / top_p /
            # top_k / min_p / repetition_penalty from the served model's shipped
            # generation_config.json or the standup's --override-generation-config (Qwen3.5).
            # Nothing about the model's decode is written here, so the driver can never contradict
            # what the server was pinned to. presence_penalty is NOT among the keys a server default
            # can carry, so on this path Qwen3.5 decodes at presence_penalty 0.0 (the vendor row's 1.5
            # reaches the engine only through `decode=`).
            pass
        else:
            # greedy (0.0) unless the attack asks to sample at an EXPLICIT temperature (the
            # PyRIT target passes do_sample=target_decode.do_sample_for(t)).
            req["temperature"] = float(temperature) if do_sample else 0.0
        if sampling_seed is not None:
            req["seed"] = int(sampling_seed)
        if decode is not None:
            problem = bs.sampler_preset_problem(decode)
            if problem is not None:
                raise ServedTargetError(f"served target decode preset: {problem}")
            if not do_sample:
                raise ServedTargetError(
                    "a decode preset needs do_sample=True — a preset on a "
                    "greedy call would silently be ignored by the server's "
                    "temperature-0 path; refusing")
            if sampling_seed is None and not getattr(self, "allow_unseeded_decode", False):
                raise ServedTargetError(
                    "a decode preset needs a per-item sampling_seed — an "
                    "unseeded sampled draw cannot be reproduced; refusing (an unseeded "
                    "draw must be DECLARED: allow_unseeded_decode, set by jailbreak_run "
                    "--multiturn-target-seed none)")
            req.update(bs.sampler_preset(decode))
        elif do_sample and temperature is not None and self.decode:
            # the extras ride an EXPLICIT temperature only: under the server's generation
            # config (temperature None) no sampling key may travel, or the request would
            # contradict the very config it defers to
            for k in DECODE_EXTRA_KEYS:
                v = self.decode.get(k)
                if v is not None:
                    req[k] = v
        # the decode block AS SENT: every key that shapes the server's sampling for this
        # call, whether it travelled in the body or is the server's default (absent key)
        self.last_decode = {"do_sample": bool(do_sample), "temperature": req.get("temperature"),
                            # None + server_generation_config = the server's generation config
                            # governed every sampling key (never "unknown"); a vendor preset
                            # is an explicit temperature, named as such
                            "temperature_source": ("vendor_preset" if decode is not None
                                                   else td.temperature_source(temperature, do_sample)),
                            "seed": req.get("seed"),
                            **{k: req.get(k) for k in DECODE_EXTRA_KEYS}}
        if decode is not None:
            self.last_decode["preset"] = dict(bs.sampler_preset(decode))
        if self.decode_note:
            self.last_decode["note"] = self.decode_note
        bad = sorted(set(req) & set(FORBIDDEN_CLIENT_KEYS))
        if bad:  # defensive: never smuggle a proxy-owned key
            raise ServedTargetError(
                f"served target would send reserved proxy key(s) {bad}")
        return req

    def _one(self, messages, *, max_new_tokens, do_sample, temperature: Optional[float],
             sampling_seed, decode=None, strict_overflow=False) -> str:
        return self._one_detailed(messages, max_new_tokens=max_new_tokens,
                                  do_sample=do_sample, temperature=temperature,
                                  sampling_seed=sampling_seed, decode=decode,
                                  strict_overflow=strict_overflow)["text"]

    def _one_detailed(self, messages, *, max_new_tokens, do_sample, temperature,
                      sampling_seed, decode=None, strict_overflow=False) -> dict:
        """ONE request -> the detailed record (:func:`bon_served.turn_record`
        shape: text, finish_reason, usage, max_tokens actually sent, seed,
        clamp fields when the clamp fired). :meth:`_one` is its ``["text"]``;
        the timing log rows are unchanged.

        ``strict_overflow`` (the multi-turn policy): a context-length 400 is a
        TYPED PromptOverWindow, never a clamped retry — the plan is built with
        ``min_completion = the requested cap`` so ``room < cap`` FAILS. The
        once-only clamp stays available to the single-shot :meth:`generate`
        path only, where it is a deliberately supported recovery."""
        req = self._payload(messages, max_new_tokens=max_new_tokens,
                            do_sample=do_sample, temperature=temperature,
                            sampling_seed=sampling_seed, decode=decode)
        t0 = time.time()
        # request-evidence: the send runs under the caller's label — a `labelled(...)` context
        # (generate_detailed(labels=…)) or the pending set_request_context (the multi-turn drivers); with the
        # hook ON an unlabelled task request is refused here by name; off = a no-op
        pending = self._take_request_context() if rev.current_label() is None else None
        with rev.labelled_or_none(pending):
            rev.require_label(f"served target {self.virtual_model!r}")
            return self._one_send(req, messages, t0=t0, max_new_tokens=max_new_tokens, sampling_seed=sampling_seed,
                                  decode=decode, strict_overflow=strict_overflow)

    def _one_send(self, req: dict, messages, *, t0: float, max_new_tokens: int, sampling_seed, decode, strict_overflow: bool) -> dict:
        """The HTTP attempt(s) of ONE detailed request and its record — the body of the pre-hook
        :meth:`_one_detailed` from its first ``self._chat`` on, unchanged except the binding stamp at the end."""
        clamped_from: Optional[int] = None
        mismatch: Optional[dict] = None
        failed: Optional[sw.ClampPlan] = None
        prompt_source: Optional[str] = None
        try:
            resp = self._chat(req)
        except Exception as exc:  # noqa: BLE001 — a context-length 400 (clamp / the strict multi-turn overflow) or an all-turns refusal
            plan = self._plan(exc, req, min_completion=(int(max_new_tokens) if strict_overflow else None))
            if plan is None:
                # not the window's business: an all-turns render refusal is
                # the typed TERMINAL error; anything else propagates unchanged
                self._raise_if_all_turns_refused(
                    exc, t0=t0, n_messages=len(messages), max_new_tokens=max_new_tokens)
                raise
            self._note_mismatch(plan)
            mismatch = plan.mismatch_record()
            prompt_source = plan.prompt_tokens_source
            if plan.action == sw.RETRY:
                # ONE clamped retry (to the SERVER's window); a second
                # rejection is terminal.
                clamped_from = int(req["max_tokens"])
                req["max_tokens"] = int(plan.max_tokens)
                try:
                    resp = self._chat(req)
                except Exception as exc2:  # noqa: BLE001
                    plan2 = self._plan(exc2, req)
                    if plan2 is None:
                        self._raise_if_all_turns_refused(
                            exc2, t0=t0, n_messages=len(messages),
                            max_new_tokens=max_new_tokens)
                        raise
                    failed = plan2.failed(
                        f"rejected again after the single clamped retry "
                        f"({clamped_from} -> {req['max_tokens']}): {plan2.reason}")
            else:
                failed = plan
        dt = time.time() - t0
        if failed is not None:
            # a TYPED error, never a response: the prompt does not fit. The
            # timing log keeps a diagnostic row tagged error=prompt_over_window
            # (so the per-eval basis can count these); the runner records the
            # GOAL as errored.
            with self._lock:
                self.n_prompt_over_window += 1
            err = sw.PromptOverWindow.from_plan(
                failed, requested_max_tokens=int(max_new_tokens))
            if self.timing_log is not None:
                with self._lock:
                    append_jsonl(self.timing_log, {
                        "t_start": t0,
                        "latency_s": round(dt, 4),
                        "virtual_model": self.virtual_model,
                        "n_messages": len(messages),
                        "max_tokens": int(max_new_tokens),
                        "finish_reason": None,
                        **err.record(),
                        "intervention_hash": None,
                    })
                    self.n_timing_rows += 1
            raise err
        try:
            choice = resp["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ServedTargetError(
                f"served target returned no choices/content for vmodel "
                f"{self.virtual_model!r}: {str(resp)[:400]}") from e
        if text is None:
            text = ""
        usage = resp.get("usage") or {}
        finish_reason = choice.get("finish_reason") if isinstance(choice, Mapping) else None
        # request-evidence: the attempt this response binds to ({} with the hook off)
        binding = rev.row_fields(resp)
        with self._lock:
            self.last_request_binding = binding or None
            self.last_request_id = binding.get(rev.ROW_REQUEST_ID_FIELD) if binding else None
            self.last_completion_id = binding.get(rev.ROW_COMPLETION_ID_FIELD) if binding else None
            if clamped_from is not None:
                self.n_clamped += 1
            if decode is not None:
                self.n_sampled_requests += 1
            if self.timing_log is not None:
                row = {
                    "t_start": t0,
                    "latency_s": round(dt, 4),
                    "virtual_model": self.virtual_model,
                    "n_messages": len(messages),
                    "max_tokens": int(req["max_tokens"]),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "finish_reason": finish_reason,
                    "intervention_hash": resp.get("petri_intervention_hash"),
                    "decode": dict(self.last_decode or {}),
                }
                if clamped_from is not None:
                    row["max_tokens_clamped_from"] = clamped_from
                    row["prompt_tokens_source"] = prompt_source
                if mismatch is not None:
                    row["window_mismatch"] = mismatch
                append_jsonl(self.timing_log, row)
                self.n_timing_rows += 1
        turn = bs.ServedTurn(text=text, finish_reason=finish_reason,
                             prompt_tokens=usage.get("prompt_tokens"),
                             completion_tokens=usage.get("completion_tokens"),
                             max_tokens=int(req["max_tokens"]),
                             max_tokens_clamped_from=clamped_from,
                             window_mismatch=mismatch,
                             prompt_tokens_source=prompt_source)
        record = bs.turn_record(turn, sampling_seed=sampling_seed)
        record.update(binding)   # request-evidence: {request_id, completion_id} ONLY when recorded
        return record

    # -- the LLMHFSteered surface ------------------------------------------- #
    def chat(self, messages: list[dict], *, max_new_tokens: int = 512,
             do_sample: bool = False, temperature: Optional[float] = 1.0,
             sampling_seed: Optional[int] = None,
             decode: Optional[Mapping[str, Any]] = None) -> str:
        """Multi-turn generation (the Crescendo / ActorAttack / PAIR path).
        Counts ONE target forward (JB7 — backtracks and retries call this
        again and are counted). ``decode`` (a vendor sampler preset, SAMPLER_KEYS
        only — the sdfbase-llama8b wave's multi-turn target replies under the
        preset, 2026-09-10) rides the body exactly as :meth:`generate_detailed`
        threads it: it needs ``do_sample=True`` and a per-call ``sampling_seed``
        (the adapter derives one per goal and turn), and the request then carries
        EXACTLY the preset's keys + the seed; absent, the body is byte-identical
        to before (the greedy reply every earlier wave drew)."""
        with self._lock:
            self.n_target_forwards += 1
        # STRICT: a reply that would have to be shortened to fit is not a reply — the goal
        # records error:PromptOverWindow and is re-run at a wider window
        return self._one(messages, max_new_tokens=max_new_tokens,
                         do_sample=do_sample, temperature=temperature,
                         sampling_seed=sampling_seed, decode=decode, strict_overflow=True)

    def generate(self, goals: list[str], *, max_new_tokens: int = 512,
                 do_sample: bool = False, temperature: Optional[float] = 1.0,
                 sampling_seed: Optional[int] = None) -> list[str]:
        """Single-shot generation for raw goal strings (the XSTest pass).
        Counts ``len(goals)`` forwards. Sequential — the gated battery's
        single-shot use is the small benign pass; concurrent fan-out lives in
        :mod:`bon_served`."""
        with self._lock:
            self.n_target_forwards += len(goals)
        return [self._one([{"role": "user", "content": g}],
                          max_new_tokens=max_new_tokens, do_sample=do_sample,
                          temperature=temperature, sampling_seed=sampling_seed)
                for g in goals]

    def generate_detailed(self, prompts: Sequence[str], *, max_new_tokens: int = 512,
                          decode: Optional[Mapping[str, Any]] = None,
                          seeds: Optional[Sequence[int]] = None,
                          labels: Optional[Sequence[Tuple[str, str, int]]] = None) -> list[dict]:
        """The DETAILED single-shot surface ``run_static`` drives under a
        sampler preset (and the escalation driver per item): one user-turn
        request per prompt, returning :func:`bon_served.turn_record` dicts in
        prompt order — ``text``, ``finish_reason`` (what the at-cap ladder keys
        on), ``completion_tokens`` / ``prompt_tokens``, the ``max_tokens``
        actually sent, ``sampling_seed``, and the clamp fields only when the
        once-only window clamp fired. ``decode`` (SAMPLER_KEYS only) makes the
        draws sampled (``do_sample=True``, the preset's temperature) and then
        REQUIRES ``seeds``, one per prompt; without it the draws are greedy
        exactly like :meth:`generate` (temperature 0.0; ``seeds`` optional).
        Counts ``len(prompts)`` forwards. Sequential like :meth:`generate` —
        the wave fills the server width with many one-request drivers; the
        concurrent twin is :meth:`bon_served.ServedStaticDefense.
        generate_detailed`. Safe to call from a thread pool (the counters are
        locked)."""
        prompts = list(prompts)
        if decode is not None:
            problem = bs.sampler_preset_problem(decode)
            if problem is not None:
                raise ServedTargetError(f"served target decode preset: {problem}")
            if seeds is None:
                raise ServedTargetError(
                    "generate_detailed(decode=...) needs one sampling seed per "
                    "prompt — a sampled draw without a recorded seed cannot be "
                    "reproduced; refusing")
        seed_list: list = ([None] * len(prompts) if seeds is None
                           else [int(s) for s in seeds])
        if len(seed_list) != len(prompts):
            raise ServedTargetError(
                f"generate_detailed: {len(seed_list)} seeds for {len(prompts)} "
                "prompts — one seed per prompt, aligned")
        # request-evidence: ``labels`` = one ``(task, item_id, sample_idx)`` per prompt, the
        # X-Tinycap labels each request carries (the caller that knows the goal ids passes them);
        # None = unlabelled (refused by _one_detailed only when the hook is ON)
        label_list: list = [None] * len(prompts) if labels is None else [tuple(x) for x in labels]
        if len(label_list) != len(prompts):
            raise ServedTargetError(
                f"generate_detailed: {len(label_list)} labels for {len(prompts)} prompts — one label per prompt, aligned")
        with self._lock:
            self.n_target_forwards += len(prompts)
        sampled = decode is not None
        temperature = float(decode["temperature"]) if sampled else 0.0
        out: list[dict] = []
        for p, seed, lab in zip(prompts, seed_list, label_list):
            with rev.labelled_or_none(lab):
                out.append(self._one_detailed([{"role": "user", "content": p}],
                                              max_new_tokens=max_new_tokens,
                                              do_sample=sampled, temperature=temperature,
                                              sampling_seed=seed, decode=decode))
        return out


# --------------------------------------------------------------------------- #
# Construction from a proxy endpoint + routing table                          #
# --------------------------------------------------------------------------- #


def probe_fn(chat_fn: ChatFn, vmodel: str, twin_vmodel: str) -> bs.SteeringProbe:
    """A :data:`bon_served.SteeringProbe` over the SAME serving path the run
    uses: greedy requests to the cell's vmodel and to its scale-0 twin, one
    row per probe prompt, texts byte-encoded as the id lists
    ``fidelity._check_steering_bites`` compares."""
    def _greedy(model_name: str, prompt: str) -> list[int]:
        resp = chat_fn({"model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0, "max_tokens": PROBE_MAX_TOKENS,
                        "seed": 0})
        return list((resp["choices"][0]["message"]["content"] or "")
                    .encode("utf-8"))

    def probe(_arm: str):
        return ([_greedy(vmodel, p) for p in PROBE_PROMPTS],
                [_greedy(twin_vmodel, p) for p in PROBE_PROMPTS])

    return probe


def harness_cell_id(tt_cell_id: str) -> str:
    """TrainTest cell id -> the routing table's (petri harness) cell id."""
    from ...capeval.petri.registry_adapter import harness_cell_id as _h

    return _h(tt_cell_id)


def route_for(tt_cell_id: str, routes: Sequence[proxy_mod.RouteEntry]
              ) -> proxy_mod.RouteEntry:
    hid = harness_cell_id(tt_cell_id)
    by_cell = {r.cell_id: r for r in routes}
    route = by_cell.get(hid)
    if route is None:
        raise ServedTargetError(
            f"cell {tt_cell_id!r} (harness {hid!r}) has no route in the "
            f"routing table ({len(routes)} routes: "
            f"{sorted(by_cell)}) — the serving launch did not export it; "
            "re-export with this cell or run --target-engine hf")
    return route


def served_evidence(chat_fn: ChatFn, route: proxy_mod.RouteEntry,
                    routes: Sequence[proxy_mod.RouteEntry]) -> dict:
    """The two positive controls, through the live proxy: (1) per-cell
    KV-cache salt isolation over the FULL route set (raises on violation);
    (2) for a STEERED route — a lens hook or a native LoRA adapter
    (``bon_served.is_steered``) — steering-bites vs its path-matched scale-0
    twin (``proxy_engine.scale0_twin``: ``<cell>__lens0`` / ``none__lens0``
    for a lens route, ``<cell>__lora0`` / ``none__lora0`` for a lora route;
    raises :class:`~sisteer.capeval.serving.fidelity.FidelityError` on an
    inert arm). Returns the evidence dict the driver persists."""
    bs.assert_cell_salt_isolation(routes)
    salt = bs.cell_salt_isolation_report(routes)
    ev: dict = {
        "cell_id": route.cell_id, "virtual_model": route.virtual_model,
        "serving_identity": route.serving_identity,
        "intervention_hash": route.intervention_hash,
        "salt_isolation": {"ok": True, "n_routes": len(routes),
                           "this_cell": salt["per_cell"].get(route.virtual_model)},
    }
    if not bs.is_steered(route):
        ev["steering_bites"] = {
            "applicable": False,
            "reason": f"route {route.cell_id!r} carries no steering intervention "
                      "(serving path is neither lens nor lora)"}
        return ev
    twin = pe.scale0_twin(route, routes)
    if twin is None:
        raise ServedTargetError(
            f"steered cell {route.cell_id!r} has no scale-0 twin route in the "
            f"routing table (neither its own {pe.scale0_twin_suffix(route)} twin nor "
            f"none{pe.scale0_twin_suffix(route)}) — a steered serve without a bites "
            "probe is refused, never run on faith")
    arm = route.cell_id.split("__")[-1]
    t0 = time.time()
    diag = bs.assert_steering_bites(
        (arm,), probe_fn(chat_fn, route.virtual_model, twin.virtual_model))
    d = diag["arms"].get(arm, {})
    ev["steering_bites"] = {
        "applicable": True, "verified": True, "twin": twin.cell_id,
        "twin_vmodel": twin.virtual_model, "arm": arm,
        "n_probes": d.get("n"), "n_differing": d.get("n_differing"),
        "first_divergences": d.get("first_divergences"),
        "probe_wall_s": round(time.time() - t0, 2),
    }
    return ev


def attested_declaration_for(proxy_identity: Any, *, harness_id: str,
                             base_harness_id: str, variant: Optional[str]
                             ) -> Optional[dict]:
    """The ATTESTED serve declaration this run may rely on for ``harness_id``,
    read from the proxy's ``/v1/models.petri_proxy`` block — never from the
    routing table's informational ``positions`` and never from the cell name.

    A VARIANT cell (``variant`` not None) requires
    ``proxy_identity[lens_serve_declarations][harness_id]`` to equal the
    variant's declaration (``serve_declaration.variant_declarations``) and the
    BASE cell to be ABSENT from the map. A missing map, a
    ``lens_serve_declarations_error``, a missing entry, or a different value is
    a :class:`ServedTargetError` naming both values — BEFORE any generation:
    setting (2) with no attested ``fit_response_serve_all_turns`` is refused
    exactly as thinking-ON is (revision 1). Sibling entries of the
    other closed shape (the ``none__lens0`` twin, experiment-1 cells) are
    tolerated: only this cell's entry is validated.

    A NON-variant cell needs no attestation; but if the map exists and names
    this cell, the standup serves it under a declaration this run did not ask
    for (a mislabel in the other direction) — refused as well. Returns the
    attested value (a fresh dict) for a variant, None otherwise."""
    meta = proxy_identity if isinstance(proxy_identity, Mapping) else {}
    decl_map = meta.get(ATTEST_SERVE_DECLARATIONS_KEY)
    decl_err = meta.get(ATTEST_SERVE_DECLARATIONS_ERROR_KEY)
    if variant is None:
        if isinstance(decl_map, Mapping) and harness_id in decl_map:
            raise ServedTargetError(
                f"cell {harness_id!r} is DECLARED out of support on this standup "
                f"({ATTEST_SERVE_DECLARATIONS_KEY}[{harness_id!r}] = "
                f"{decl_map[harness_id]!r}) but this run asked for the in-support "
                "cell — a declared cell must be addressed by its variant spelling; "
                "refusing rather than mislabel the arm")
        return None
    expected = variant_declarations(variant)
    what = (f"variant cell {harness_id!r} (serve variant {variant!r}) requires the "
            f"attested declaration {expected!r} under "
            f"{ATTEST_SERVE_DECLARATIONS_KEY}[{harness_id!r}]")
    if decl_map is None:
        raise ServedTargetError(
            f"{what}, but the proxy attests NO {ATTEST_SERVE_DECLARATIONS_KEY} map"
            + (f" ({ATTEST_SERVE_DECLARATIONS_ERROR_KEY}: {decl_err!r})" if decl_err else "")
            + f" (petri_proxy keys: {sorted(meta)}) — an undeclared serving is "
              "refused before any generation, exactly as thinking-ON is (plan §2.3, §3.1)")
    if decl_err:
        raise ServedTargetError(
            f"{what}, but the standup reports {ATTEST_SERVE_DECLARATIONS_ERROR_KEY} = "
            f"{decl_err!r} — the map cannot be trusted; refusing before any generation")
    if not isinstance(decl_map, Mapping):
        raise ServedTargetError(
            f"{what}, but {ATTEST_SERVE_DECLARATIONS_KEY} is not a per-cell map: "
            f"{decl_map!r}")
    if harness_id not in decl_map:
        raise ServedTargetError(
            f"{what}, but the attested map has no entry for it (attested cells: "
            f"{sorted(decl_map)}) — the standup did not serve this variant; "
            "refusing before any generation")
    got = decl_map[harness_id]
    if not isinstance(got, Mapping) or dict(got) != expected:
        raise ServedTargetError(
            f"{what}, but the proxy attests {got!r} for it — the served declaration "
            "is not the one this run asked for; refusing before any generation")
    if base_harness_id in decl_map:
        raise ServedTargetError(
            f"{what}: satisfied, but the BASE cell {base_harness_id!r} is ALSO in the "
            f"attested map ({decl_map[base_harness_id]!r}) — the in-support base "
            "must be absent (plan §2.3); the standup mislabels its cells; refusing")
    return dict(got)


def served_positions_of(route: Any, variant: Optional[str]) -> Optional[str]:
    """The served position mode for the record: the routing table's
    informational ``routes[*].positions`` when the export carries it (an
    optional additive field), else the variant's overlay
    ``positions`` for a position variant, else None. INFORMATIONAL — never an
    input to the attestation decision."""
    pos = getattr(route, "positions", None)
    if pos is not None:
        return str(pos)
    if variant is not None:
        return variant_overlay(variant).get("positions")
    return None


def build_served_target(*, proxy_url: str, routing_table: str | os.PathLike,
                        cell_id: str, timing_log: Optional[Path] = None,
                        controls: bool = True,
                        max_model_len: Optional[int] = None,
                        decode: Optional[Mapping[str, Any]] = None,
                        timeout_s: Optional[float] = None,
                        decode_note: Optional[str] = None) -> ServedChatTarget:
    """The served twin of ``build_defense_cell``: resolve the cell's route,
    connect a :class:`ProxyClientEngine` (``load`` verifies the endpoint IS
    the petri proxy and serves this vmodel), resolve the SI text for a
    prompted cell, run the positive controls (``controls=True``), and return
    the target with its evidence attached. ``max_model_len`` is the serving
    window the launcher resolved (geometry.py), threaded into the target's
    max_tokens clamp (module docstring) and recorded in ``route_info``.

    A serve VARIANT (``<cell>@thinking`` / ``@all_turns`` / ``@all_positions``)
    resolves through ``harness_cell_id`` to its own vmodel and REQUIRES the
    attested declaration (:func:`attested_declaration_for`) — read after
    ``client.load`` from ``/v1/models.petri_proxy`` — BEFORE the controls and
    before any generation. ``route_info`` records ``serve_variant``,
    ``positions`` (informational) and ``serve_declarations`` (the attested
    value).
    ``timeout_s`` is the CLIENT's per-request timeout (``ProxyClientEngine``,
    default 600 s): a 4,096-token reply behind a full scheduler queue can
    outlive 600 s (four tasks on one compute node died on it, 2026-09-10), so a
    long-reply run raises it DELIBERATELY and the value is recorded
    (``client_timeout_s``); None keeps the engine's default. ``decode_note``
    rides every decode block."""

    try:
        base_id, variant = split_variant(cell_id)
    except ServeDeclarationError as e:
        raise ServedTargetError(f"--defense {cell_id!r}: {e}") from e
    table = proxy_mod.load_routing_table(Path(routing_table))
    routes = list(table["routes"])
    route = route_for(cell_id, routes)
    if timeout_s is not None and float(timeout_s) <= 0:
        raise ServedTargetError(f"timeout_s must be > 0, got {timeout_s!r}")
    client = ProxyClientEngine(proxy_url, identity=route.serving_identity,
                               **({"timeout_s": float(timeout_s)} if timeout_s is not None else {}))
    client.load(require_virtual_models=[route.virtual_model])
    # request-evidence: off = the client is returned untouched; on = every chat_completions
    # through it is a recorded, labelled attempt (the positive controls below are `probe/bites`)
    client = rev.maybe_attach_proxy_client(client, unlabelled_kind=rev.PROBE_KIND_BITES)
    base_hid = "none" if base_id == "none" else harness_cell_id(base_id)
    declaration = attested_declaration_for(
        client.proxy_identity, harness_id=route.cell_id,
        base_harness_id=base_hid, variant=variant)
    meta = client.proxy_identity if isinstance(client.proxy_identity, Mapping) else {}
    system = None
    if route.si_prefix_hash is not None:
        from ..defense_cell import si_of
        from ..jailbreak import resolve_instruction

        si_id = si_of(cell_id)
        if si_id is None:
            raise ServedTargetError(
                f"route {route.cell_id!r} expects an SI prefix but cell "
                f"{cell_id!r} names no SI — cannot build its system text")
        system = resolve_instruction(si_id).text
    info = {"proxy_url": client.base_url, "routing_table": str(routing_table),
            "tt_cell_id": cell_id, "harness_cell_id": route.cell_id,
            "virtual_model": route.virtual_model,
            "serving_identity": route.serving_identity,
            "intervention_hash": route.intervention_hash,
            "base_model": table.get("base_model"),
            "served_models": list(client.served_models),
            "proxy_identity": client.proxy_identity,
            "prompted": system is not None,
            "max_model_len": (int(max_model_len) if max_model_len is not None
                              else None),
            # the serve variant and what the proxy ATTESTS for it; positions
            # is informational (the table's field, else the variant's)
            "serve_variant": variant,
            "base_cell_id": base_id,
            "positions": served_positions_of(route, variant),
            "serve_declarations": declaration,
            # the proxy's own records this run's identity binds: None when
            # the proxy predates them
            "lens_salt_policy": meta.get(ATTEST_LENS_SALT_POLICY_KEY),
            "object_artifact_sha256_by_cell": meta.get(ATTEST_OBJECT_SHA256_BY_CELL_KEY),
            # the client's per-request timeout actually in force (engine default when
            # the caller passed none) -- a record, so a timed-out leg can be read back
            "client_timeout_s": float(getattr(client, "timeout_s", 600.0))}
    target = ServedChatTarget(client.chat_completions,
                              virtual_model=route.virtual_model,
                              cell_id=cell_id, system=system,
                              timing_log=timing_log, route_info=info,
                              max_model_len=max_model_len,
                              # the proxy's read-only /tokenize (floor-only
                              # resolution); a client without it leaves a
                              # floor-only rejection floor_only_unresolvable
                              tokenize_fn=getattr(client, "tokenize", None),
                              decode=decode, decode_note=decode_note)
    if controls:
        target.evidence = served_evidence(client.chat_completions, route,
                                          routes)
    return target


#: the multi-turn overflow policy every wave since 2026-09-10 runs under (recorded in served_target.json)
MULTITURN_OVERFLOW_POLICY = "strict"

#: the keys that IDENTIFY a target decode (the sampler that drew a store's replies): the sampler
#: values, the seed policy and the trial — never the provenance strings (source, source_kind)
TARGET_DECODE_IDENTITY_KEYS = ("do_sample", *bs.SAMPLER_KEYS, "cap", "arm", "seed_base", "trial",
                               "seed_rule", "multiturn_target_decode", "multiturn_target_seed")


def _decode_identity(block) -> Optional[dict]:
    if block is None:
        return None
    return {k: block[k] for k in TARGET_DECODE_IDENTITY_KEYS if k in block}


def _rows_record_a_sampler(responses: Path) -> bool:
    """True when any parseable row of a store's ``responses.jsonl`` records a sampler — a ``decode``
    block or a ``sampling_seed`` (a SAMPLED store's rows; a greedy static store's rows carry neither).
    Read only by the static leg's historical exception in :func:`assert_target_decode_compatible`; a
    torn tail line is skipped exactly as resume does."""
    responses = Path(responses)
    if not responses.is_file():
        return False
    return any(isinstance(rec, dict) and ("decode" in rec or "sampling_seed" in rec)
               for rec in iter_jsonl_rows(responses))


def assert_target_decode_compatible(out_dir: Path, decode_block, *,
                                    legacy_leg_without_sidecar: bool = False,
                                    serving_engine: Optional[str] = None) -> None:
    """Refuse to resume a served store under a different TARGET sampler.
    ``served_target.json`` records the decode block the store's replies were drawn under; a store
    with covered goals (``progress.jsonl`` non-empty) and a recorded block that differs from the
    requested one — or a LEGACY store (a manifest with no block recorded) when a preset is
    requested — is refused BEFORE any manifest is (re)written, so coverage is never reused under
    a relabelled sampler. An empty store accepts any sampler; a legacy store accepts the greedy
    target it was drawn with.

    A store with covered goals and NO ``served_target.json`` at all is CORRUPTION, not
    legacy: every served run has written the manifest before its first generation
    (``write_target_manifest(stage="started")``), so an absent sidecar means a partial copy or a
    lost file — the sampler, seed policy and trial of the covered goals cannot be established, and
    the store is refused before a fresh manifest could stamp it as the current identity.

    "Covered" is the adapters' RESUME SKIP SET (P2-c): ``static_common.
    load_covered_goals`` — the goals whose latest terminal record is a success; a goal whose latest
    record is a typed ``error:*`` (``error:PromptOverWindow``: the multi-turn over-window goals, a
    typed error by decision and exactly the rows a re-run must retry) is NOT covered, the same way
    ``multiturn`` / ``pyrit_adapters`` / ``gcg`` / ``run_static`` read it. A store holding only such
    rows covers nothing, so an absent sidecar on it is no refusal: nothing drawn under an unknown
    sampler would be reused.

    ``legacy_leg_without_sidecar`` (P2-b) is the STATIC leg's one historical
    exception, stated by its caller (jailbreak_run's replayable branch) and by nobody else: the
    static leg's served runs wrote NO ``served_target.json`` before the sampled re-run of 2026-09-10,
    so a covered static store with no sidecar whose rows record no sampler is a legacy GREEDY store,
    and a GREEDY run (``decode_block is None``) may resume it — stamped afterwards as exactly that (a
    manifest without a decode block). It is never a licence for anything else: a sampled request over
    such a store refuses (a historical greedy store is never relabelled, and a sampled static store
    has carried the sidecar since its first generation, so one without it is a partial copy), and a
    greedy request over a sidecar-less store whose ROWS record a sampler refuses (a sampled store that
    lost its sidecar). The multi-turn tiers, which have written the manifest all along, keep the
    default rule.

    ``serving_engine`` (None = not compared, the pre-landing call) adds THE ENGINE COMPARE
    to the same pre-write gate, under the same covered-goals scope:
    the manifest's recorded ``serving_engine`` must equal this run's — ABSENT = lens by era (a
    manifest written before the native engine; a lens run resumes it, a native run refuses it),
    PRESENT and different = refusal — and the static leg's sidecar-less legacy store is lens by
    era as well (it predates 2026-09-10, let alone the native engine), so a native run refuses it
    even greedy. Both refusals land BEFORE the driver's first manifest write, so a cross-engine
    attempt leaves the store byte-identical; the row writers run their own compare
    (:func:`assert_store_serving_engine_compatible`) before their first append."""
    out_dir = Path(out_dir)
    if serving_engine is not None:
        _check_serving_engine_value(serving_engine, what="assert_target_decode_compatible")
    progress = out_dir / "progress.jsonl"
    covered = load_covered_goals(progress)
    if not covered:
        return
    manifest = out_dir / "served_target.json"
    want = _decode_identity(decode_block)
    if not manifest.is_file():
        if legacy_leg_without_sidecar and want is None:
            if not _rows_record_a_sampler(out_dir / "responses.jsonl"):
                if serving_engine is not None and serving_engine != SERVING_ENGINE_LENS:
                    raise ServedTargetError(
                        f"{out_dir}: progress.jsonl holds {len(covered)} covered goal(s) with no {manifest.name} — "
                        "the static leg's legacy greedy store, drawn before 2026-09-10 on the LENS serving path "
                        f"by construction — and this run serves on {serving_engine!r}; refusing to resume it or "
                        "to stamp a native manifest over it. Run this cell into a fresh dir")
                return   # the static leg's pre-2026-09-10 greedy store, resumed greedy: what it is
            raise ServedTargetError(
                f"{out_dir}: progress.jsonl holds {len(covered)} covered goal(s) whose responses.jsonl rows record a "
                f"sampler (a decode block / sampling_seed), but {manifest.name} is ABSENT — a SAMPLED store that lost "
                "its sidecar, not the static leg's legacy greedy store; refusing to resume it greedy or to stamp a "
                "greedy manifest over it. Restore the sidecar or regenerate into a fresh store")
        if legacy_leg_without_sidecar:
            raise ServedTargetError(
                f"{out_dir}: progress.jsonl holds {len(covered)} covered goal(s) but {manifest.name} is ABSENT and this "
                f"run asks for the target decode {want} — a static store drawn before 2026-09-10 was GREEDY and is never "
                "relabelled as a sampled one, and a sampled static store has carried the sidecar since its first "
                "generation, so one without it is a partial copy; refusing to resume it or to write a fresh manifest "
                "over it. Resume it greedy (no sampled_decode), restore the sidecar, or regenerate into a fresh store")
        raise ServedTargetError(
            f"{out_dir}: progress.jsonl holds {len(covered)} covered goal(s) (successful terminal records; typed "
            f"error:* goals are uncovered and would be retried) but {manifest.name} is ABSENT — every served "
            "run writes the manifest before its first generation, so this store is a partial copy or has "
            "lost its sidecar; the sampler / seed policy / trial of its covered goals cannot be "
            "established. Refusing to resume it or to write a fresh manifest over it; restore the "
            "sidecar or regenerate into a fresh store")
    recorded = json.loads(manifest.read_text())
    if serving_engine is not None:
        have_engine = _engine_or_lens(recorded.get(SERVING_ENGINE_FIELD))
        if have_engine != serving_engine:
            raise ServedTargetError(
                f"{manifest}: the store's {len(covered)} covered goal(s) were drawn on the {have_engine!r} "
                "serving engine"
                + (" (no serving_engine recorded — a manifest written before the native engine, lens by "
                   "construction)" if SERVING_ENGINE_FIELD not in recorded else "")
                + f" and this run serves on {serving_engine!r} — refusing to resume or relabel it (a store "
                "stays single-engine: a lens row and a native row are two measurements); use a fresh store")
    have = _decode_identity(recorded.get("decode"))
    if recorded.get("decode") is None:
        if want is not None:
            raise ServedTargetError(
                f"{manifest}: a LEGACY store (its target decode was never recorded; drawn greedy before "
                f"2026-09-10) holds covered goals, and this run asks for the target decode {want} — refusing "
                "to relabel it; regenerate into a fresh store or run --multiturn-target-decode greedy")
        return
    if have != want:
        raise ServedTargetError(
            f"{manifest}: the store's covered goals were drawn under a different target decode "
            f"{have} than this run's {want} — refusing to resume or relabel; use a fresh store")
def _window_clamp_payload(target: ServedChatTarget) -> dict:
    return {"max_model_len": target.max_model_len,
            "min_clamped_completion": sw.MIN_CLAMPED_COMPLETION,
            # chat() (multi-turn) never clamps: an overflow is a typed error;
            # generate() (single-shot) keeps the once-only clamp
            "multiturn_overflow_policy": MULTITURN_OVERFLOW_POLICY,
            "n_clamped": target.n_clamped,
            "n_prompt_over_window": target.n_prompt_over_window,
            "n_window_mismatch": target.n_window_mismatch,
            "window_mismatches": list(target.window_mismatches),
            "n_tokenize_resolved": target.n_tokenize_resolved,
            "n_floor_only_unresolvable": target.n_floor_only_unresolvable,
            "n_target_forwards": target.n_target_forwards}


def _all_turns_check_payload(target: ServedChatTarget) -> dict:
    """``all_turns_check``: the per-PROCESS refusal counters of setting (2)'s
    render check — ``n_refused_total`` and ``by_code`` over the
    four codes. The per-GOAL code is in ``progress.jsonl``."""
    by_code = dict(target.n_all_turns_refused)
    return {"n_refused_total": sum(by_code.values()),
            "by_code": {c: int(by_code.get(c, 0)) for c in ALL_TURNS_REFUSAL_CODES},
            "codes": list(ALL_TURNS_REFUSAL_CODES),
            "outcome_prefix": REFUSED_OUTCOME_PREFIX}


def write_target_manifest(out_dir: Path, target: ServedChatTarget, *,
                          stage: str = "started",
                          decode: Optional[Mapping[str, Any]] = None,
                          extra: Optional[Mapping[str, Any]] = None) -> Path:
    """``served_target.json`` beside ``attacker_engine.json``: the served
    identity (vmodel, intervention hash, proxy) + the positive-control
    evidence, so every transcript dir is self-identifying about WHAT served
    it and that the steering was live. Written twice by the driver: once
    BEFORE generation as the ``started`` marker (evidence is already known),
    and again in a ``finally`` AFTER generation (``stage="final"``) so the
    window-clamp counters are the real ones — a pre-generation write alone
    reports zeros forever (adversarial review, 2026-09-03).

    ``decode`` (the sampled re-run's row-level decode block: do_sample, the
    preset keys, cap, arm, seed_base, trial) adds a ``decode`` block plus
    ``n_sampled_requests`` (requests the target issued under a preset) — so a
    sampled store's manifest says HOW it was decoded; absent, the manifest is
    byte-identical to before. ``extra`` (the sampled ladder's rung drivers):
    additional top-level facts merged LAST (e.g. the client hop timeout the
    hop_timeout_guard reads back); absent, nothing is added.

    THE ENGINE BOUNDARY: this is the ONE writer of the store's manifest, so
    the engine compare is enforced HERE, before the ``started`` or the
    ``final`` payload is even built — an existing ``served_target.json`` that
    records another engine (ABSENT = lens by era, the rule every reader uses)
    refuses with the store byte-identical, whatever the caller checked first.
    Before this, the driver's pre-write check ran WITHOUT the engine
    (``assert_target_decode_compatible`` was called with no ``serving_engine``)
    and this writer stamped the new manifest unconditionally, so a native
    target pointed at a lens store rewrote the lens manifest — lens responses
    under a native manifest — and the CLI's ``finally`` wrote it again after
    the row writer had refused; an ordinary lens resume then refused its own
    store. A target with no ``serving_engine`` (a duck-typed stand-in) compares
    nothing, exactly as it stamps nothing."""
    out_dir = Path(out_dir)
    assert_manifest_serving_engine_compatible(out_dir, getattr(target, "serving_engine", None))
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"target_engine": TARGET_ENGINE_SERVED, **target.route_info,
               "manifest_stage": stage,
               # the engine this target's rows are stamped with: additive, unhashed;
               # the pre-write compare reads it back (absent = a pre-native manifest, lens by era)
               **({SERVING_ENGINE_FIELD: getattr(target, "serving_engine", None)}
                  if getattr(target, "serving_engine", None) is not None else {}),
               "evidence": target.evidence,
               "window_clamp": _window_clamp_payload(target),
               "all_turns_check": _all_turns_check_payload(target),
               "timing_log": (str(target.timing_log) if target.timing_log
                              else None),
               # the sampled-target decode block this target was built with (None = the
               # default greedy path, no extras) and the last request's block as sent
               "target_decode": (dict(getattr(target, "decode", None) or {}) or None),
               "target_decode_note": getattr(target, "decode_note", None),
               "last_request_decode": (dict(getattr(target, "last_decode", None) or {}) or None),
               "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if decode is not None:
        payload["decode"] = dict(decode)
        payload["n_sampled_requests"] = int(getattr(target, "n_sampled_requests", 0))
    if extra:
        payload.update(dict(extra))
    p = out_dir / "served_target.json"
    p.write_text(json.dumps(payload, indent=2, default=str))
    return p


def append_target_attempt(out_dir: Path, target: ServedChatTarget, *,
                          stage: str = "final") -> Path:
    """One APPEND-ONLY row per process in ``served_target.attempts.jsonl``
    beside ``served_target.json``: the manifest is overwritten by
    every process, so a resumed shard's clamp and render-refusal counters
    would otherwise be lost; the analysis SUMS the ``stage == "final"`` rows.
    Identity fields ride along so a row is self-describing about the vmodel
    that produced its counts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = target.route_info
    row = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "stage": stage, "pid": os.getpid(),
           "tt_cell_id": info.get("tt_cell_id", target.cell_id),
           "harness_cell_id": info.get("harness_cell_id"),
           "virtual_model": target.virtual_model,
           "intervention_hash": info.get("intervention_hash"),
           "serve_variant": info.get("serve_variant"),
           "n_target_forwards": target.n_target_forwards,
           "n_timing_rows": target.n_timing_rows,
           "window_clamp": _window_clamp_payload(target),
           "all_turns_check": _all_turns_check_payload(target)}
    p = out_dir / TARGET_ATTEMPTS_FILE
    append_jsonl(p, row)
    return p
