"""Served-route generation engine for the EvalPlus runner: HumanEval+/MBPP+
generation through the RoutingProxy's (or the usage ledger's)
``/v1/chat/completions``, on the same proxy contract every other served
client obeys.

Why a second engine next to :class:`~sisteer.capeval.serving.proxy.ProxyClientEngine`:
that class is the petri runner's SYNC handle (``load`` / ``chat_completions``
/ ``tokenize``, one call at a time) and has no ``generate``; the EvalPlus
runner needs the ``Engine.generate(prompts, arm, gen_cfg)`` seam
(``serving/__init__.py``) fanned out at the served concurrency. This module
provides it — taking CHAT ITEMS (user text + optional assistant prefill)
instead of rendered strings, because the proxy templates SERVER-side and a
client-rendered string would bypass the prompted-prefix hash check and the
lens response-mask declaration alike.

What every request carries (E1/E3/E4/E6 — binding):

* the SI as the ``system`` message on a prompted cell (the proxy verifies
  the registered prefix hash; it never injects) — nothing for a fitted cell
  or the ``none`` route;
* ``chat_template_kwargs: {"enable_thinking": <bool>}`` per configuration
  (``contracts.chat_template_kwargs``);
* greedy (``temperature 0``), ``max_tokens`` from the tier1 gen config,
  ``stop`` from its ``until``; a ``seed`` set client-side per
  (item, sample_idx) — ``stable_seed`` — never ``n=`` (E4);
* ``stream: false`` explicitly (the proxy 400s a truthy stream);
* NONE of :data:`~sisteer.capeval.serving.proxy.FORBIDDEN_CLIENT_KEYS`;
* for the stock EvalPlus instrument (assistant prefill = response prefix +
  an open ```` ```python ```` fence) the request is a CONTINUED final
  message (``continue_final_message: true``, ``add_generation_prompt:
  false``) and, on a LENS route only, DECLARES it with
  :data:`~sisteer.capeval.serving.proxy.RESPONSE_PREFILL_KEY` so the
  response-fit mask covers header + prefill + every generated position
  (E6: a steered number that did not go through the declared path is an
  unsteered number). The proxy refuses the declaration on a non-lens route
  and refuses an undeclared continued message on a lens route
  (``proxy._validate_response_prefill``), so ``is_lens_route`` decides it
  from the route's serving identity — the same source ``_rewrite`` keys on;
* the two usage-ledger headers ``X-Tinycap-Task`` / ``X-Tinycap-Sample``
  (P27) — consumed by the ledger, harmless on the bare proxy.

What every 200 must PROVE before it is persisted: the proxy stamps every
chat response with ``model`` (the virtual name), ``petri_intervention_hash``
and a ``petri`` block carrying ``cell_id`` / ``serving_identity``
(``RoutingProxy.handle_chat``, the audit fields).
:func:`assert_response_stamps` compares all four against the ROUTE this
engine was built for — plain string equality, because the proxy stamps
``route.serving_identity`` as the raw ``<backend>+<path>+<mode>`` string,
never its parsed form — and a missing or mismatched stamp is the TYPED
:class:`StampMismatch`, filed per item with the values seen vs expected
and never written to the sidecar. Why: a client routing table one
generation behind the live proxy (same virtual name, an older fitted
object or another serving path) would otherwise persist and grade an
UNSTEERED completion under the client-side route, and ``/v1/models``
listing the name cannot tell generations apart.

What every 200 RECORDS: when the standup composed a reasoning parser
(thinking-ON: ``--reasoning-parser gemma4``, recorded in the standup's
``endpoint.json``) vLLM SPLITS the assistant message into
``reasoning_content`` (the thought) and ``content`` (the answer).
:func:`parse_chat_response` reads BOTH channels from every response —
``text`` is ``content`` (``""`` when null), ``reasoning`` the thought
(None when absent) — and the runner persists both in the sidecar row.
Scoring is on ``content`` only. A response whose ``content`` is
empty/None while ``reasoning`` is non-empty is a FLAGGED ROW
(``GenResult.content_empty_with_reasoning``): the item genuinely produced
no answer within ``max_tokens`` and that IS the measurement, so it scores
as-is (an empty solution fails) and is counted per fan-out
(``FanoutStats.n_content_empty_with_reasoning``). Only a response carrying
NEITHER channel under a composed parser (``reasoning_parser`` set on the
engine) is refused — the typed :class:`EmptyResponse`, filed per item and
never a sidecar row — because a client cannot tell "the model said
nothing" from "the parser lost the text", and scoring a maybe-lost answer
as wrong is the silent failure this lane exists to refuse. Without a
composed parser an empty ``content`` is an ordinary (wrong) answer.

Discipline (the served driver's ``_fanout``): bounded concurrency, NO
client-side retry that could re-roll a refusal or double-count a
completion, failures COLLECTED per item (:class:`ItemError`) and reported
together; the runner's generations sidecar makes a re-invoke fill only
the gaps. The ONE sanctioned retry is the served clients' once-only
``max_tokens`` clamp (:mod:`~sisteer.capeval.serving.served_window`):
a 400 that is a context-length rejection — recognised ONLY by that module's
precise parser, never any other 400 — is re-planned against the
server-stated window; a floor-only rejection ("at least N input tokens")
has its exact prompt length resolved through the proxy's read-only
``/v1/tokenize`` on the IDENTICAL render (the prefill render controls —
``continue_final_message``, ``add_generation_prompt: false``, the assistant
seed, ``chat_template_kwargs`` — via ``prefill_logprob.build_tokenize_request``;
the open variant's generation-prompt render plus its ``chat_template_kwargs``);
when at least :data:`~sisteer.capeval.serving.served_window.MIN_CLAMPED_COMPLETION`
tokens remain the body is resent ONCE with only ``max_tokens`` changed
(the rejected attempt produced no completion, so nothing is counted
twice); otherwise, or on a second rejection, the item is the TYPED
:class:`~sisteer.capeval.serving.served_window.PromptOverWindow` — never a
generation, never a sidecar row. Without this a deterministic 400 was a
permanent gap: resume resent the identical body, got the identical 400,
and the complete-item-set requirement blocked the whole task forever.

Transport is injectable (``transport(url, headers, body) -> (status,
text)``, async) so every test runs offline; the default is one aiohttp
session per fan-out with the served driver's long total timeout.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import (Any, Awaitable, Callable, Dict, List, Mapping, NamedTuple,
                    Optional, Sequence, Tuple)

from sisteer.capeval import manifest as cm
from sisteer.capeval.serving import GenOut
from sisteer.capeval.serving import prefill_logprob as pl
from sisteer.capeval.serving import served_window as sw
from sisteer.capeval.serving.proxy import (ENGINE_KEY, ENGINE_KIND_LORA_NATIVE,
                                           FORBIDDEN_CLIENT_KEYS,
                                           LENS_REDUCTION_MODE_KEY,
                                           LORA_ZERO_CELL_SUFFIX,
                                           RESPONSE_PREFILL_KEY,
                                           STEERED_ROUTE_PATHS, RouteEntry,
                                           _canonical_v1_base)
from sisteer.capeval.serving.served_window import PromptOverWindow
from sisteer.capeval.tinycap import contracts as C

__all__ = [
    "ProxyEngineError", "StampMismatch", "PromptOverWindow", "EmptyResponse", "ItemError",
    "ChatItem", "GenResult", "FanoutStats", "Transport",
    "ProxyGenerateEngine", "stable_seed", "is_lens_route", "is_steered_route",
    "STEERED_ROUTE_PATHS", "SCALE0_LORA_TWIN_SUFFIX", "scale0_twin_suffix",
    "routes_by_cell", "scale0_twin", "bounded_gather", "ledger_headers",
    "ENGINE_EXPECTATIONS", "ENGINE_EXPECT_LENS_STACKED", "ENGINE_EXPECT_LENS_PER_ROW",
    "ENGINE_EXPECT_LORA_NATIVE", "served_engine_problem",
    "assert_client_body", "parse_chat_response",
    "CHANNEL_OK", "CHANNEL_FLAGGED", "channel_verdict",
    "TOKEN_ID_PREFIX", "first_position_logprobs_by_id",
    "STAMP_FIELDS", "ROUTE_STAMP_ATTRS", "expected_stamps", "response_stamps",
    "assert_response_stamps",
    "TOKENIZE_STAMP_FIELDS", "assert_tokenize_stamp",
    "tokenize_body_for",
    "ERR_TRANSPORT", "ERR_HTTP", "ERR_MALFORMED", "ERR_STAMP_MISMATCH",
    "ERR_PROMPT_OVER_WINDOW", "ERR_EMPTY_RESPONSE",
    "GEN_CFG_KEYS", "AIOHTTP_TOTAL_TIMEOUT_S",
]


class ProxyEngineError(RuntimeError):
    """A request the proxy contract forbids, a malformed response, or a
    fan-out with failures — loud, never a silently partial generation."""


class StampMismatch(ProxyEngineError):
    """A 200 whose attestation stamps do not name the route this engine was
    built for: the response is refused — never a :class:`GenResult`, never a
    sidecar row. Carries ``expected`` (from the route), ``seen`` (from the
    response; ``None`` where absent) and the ``missing`` / ``mismatched``
    stamp names, keyed by :data:`STAMP_FIELDS`."""

    def __init__(self, item_id: str, *, expected: Mapping[str, Any],
                 seen: Mapping[str, Any], missing: Sequence[str],
                 mismatched: Sequence[str]) -> None:
        self.item_id = item_id
        self.expected = dict(expected)
        self.seen = dict(seen)
        self.missing = list(missing)
        self.mismatched = list(mismatched)
        parts = []
        if missing:
            parts.append(f"missing {list(missing)}")
        if mismatched:
            parts.append("mismatched " + ", ".join(
                f"{k}: seen {self.seen[k]!r} expected {self.expected[k]!r}" for k in mismatched))
        super().__init__(
            f"{item_id}: response stamps do not attest the client route "
            f"({'; '.join(parts)}) — the proxy served something other than the route "
            "this engine was built for (stale routing table?); refusing the completion")

    def record(self) -> Dict[str, Any]:
        """The typed error's facts for the per-item error record."""
        return {"error": ERR_STAMP_MISMATCH, "expected": dict(self.expected),
                "seen": dict(self.seen), "missing": list(self.missing),
                "mismatched": list(self.mismatched)}


class EmptyResponse(ProxyEngineError):
    """A 200 served under a COMPOSED reasoning parser whose message carries
    NEITHER channel — ``content`` empty/None AND no ``reasoning_content``
    (module docstring): refused per item, never a :class:`GenResult`,
    never a sidecar row. Carries the parser name and the response facts
    that let a reader tell "nothing generated" from "text lost"
    (``finish_reason``, ``completion_tokens``, whether ``content`` was
    null or the empty string)."""

    def __init__(self, item_id: str, *, reasoning_parser: str,
                 finish_reason: Optional[str], completion_tokens: Optional[int],
                 content_null: bool) -> None:
        self.item_id = item_id
        self.reasoning_parser = reasoning_parser
        self.finish_reason = finish_reason
        self.completion_tokens = completion_tokens
        self.content_null = content_null
        super().__init__(
            f"{item_id}: response carries neither content nor reasoning_content under the "
            f"composed reasoning parser {reasoning_parser!r} (content "
            f"{'null' if content_null else 'empty'}, finish_reason={finish_reason!r}, "
            f"completion_tokens={completion_tokens!r}) — cannot tell an empty answer from "
            "text the parser lost; refusing to score it")

    def record(self) -> Dict[str, Any]:
        """The typed error's facts for the per-item error record."""
        return {"error": ERR_EMPTY_RESPONSE, "reasoning_parser": self.reasoning_parser,
                "finish_reason": self.finish_reason,
                "completion_tokens": self.completion_tokens,
                "content_null": self.content_null}


#: ``ItemError.kind`` values — the per-item failure classes the fan-out files.
ERR_TRANSPORT = "transport_error"          # the POST itself raised (connection, timeout)
ERR_HTTP = "http_error"                    # a non-200 that is NOT a context-length 400
ERR_MALFORMED = "malformed_response"       # a 200 whose body is not one chat completion
ERR_STAMP_MISMATCH = "stamp_mismatch"      # StampMismatch
ERR_PROMPT_OVER_WINDOW = sw.PROMPT_OVER_WINDOW   # PromptOverWindow
ERR_EMPTY_RESPONSE = "empty_response"      # EmptyResponse: neither channel under a composed parser


class ItemError(NamedTuple):
    """One failed item of a fan-out. Indexes ``[0]`` / ``[1]`` as the
    ``(item_id, message)`` pair the runner always consumed; ``kind`` is one
    of the ``ERR_*`` tags and ``record`` the typed error's facts
    (``StampMismatch.record()`` / ``PromptOverWindow.record()``), None for
    the untyped kinds."""

    item_id: str
    message: str
    kind: str
    record: Optional[Dict[str, Any]] = None
    #: the sample index of the failed (item, sample) — 0 for a single-sample
    #: fan-out; with k samples per item (U8 / PW-sampling) the pair
    #: ``(item_id, sample_idx)`` names the generation that reruns on resume
    sample_idx: int = 0


def generation_key(item_id: str, sample_idx: int = 0) -> str:
    """The key ONE generation is filed under in a fan-out's results and the
    runner's sidecar — ``<item_id>/<sample_idx>``, exactly the ledger's
    ``X-Tinycap-Sample`` value (``contracts.sample_header_value``), so the
    sidecar row, the ledger row and the request headers name one draw the
    same way."""
    return C.sample_header_value(item_id, sample_idx)


#: aiohttp total timeout — the served driver's constant (a queued request
#: at high server load can sit far longer than its own decode). ONE number
#: with the ledger's, the Inspect provider's and the lens proxy's upstream
#: timeout (``contracts.PROXY_UPSTREAM_TIMEOUT_S``).
AIOHTTP_TOTAL_TIMEOUT_S = C.PROXY_UPSTREAM_TIMEOUT_S

#: The generation-config keys this engine understands
#: (``evalplus_runner.DEFAULT_GEN_CFG``'s keys +
#: ``batch_size``, which the HF path adds and the async fan-out ignores).
#: Anything else is refused: an unrecognised sampling knob silently ignored
#: would be a metric change nobody declared.
GEN_CFG_KEYS: Tuple[str, ...] = ("do_sample", "max_new_tokens", "until", "batch_size")

#: async ``(url, headers, json_body) -> (http_status, response_text)``
Transport = Callable[[str, Mapping[str, str], Mapping[str, Any]],
                     Awaitable[Tuple[int, str]]]


@dataclass(frozen=True)
class ChatItem:
    """One generation request's content: RAW user text plus an optional
    assistant prefill (the continued final message); the proxy renders the
    chat template server-side."""

    item_id: str
    user: str
    assistant_prefill: Optional[str] = None
    sample_idx: int = 0


@dataclass(frozen=True)
class GenResult:
    """One served generation with the facts the runner records."""

    item_id: str
    sample_idx: int
    text: str
    finish_reason: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    #: the reasoning parser's channel under thinking-ON (``reasoning_content``
    #: at vLLM 0.25.1; ``reasoning`` on newer servers) — None when absent.
    #: Recorded next to ``text`` (= ``content``) in the sidecar row; never
    #: scored (module docstring).
    reasoning: Optional[str]
    seed: int
    latency_s: float
    #: the ``petri_intervention_hash`` the response was stamped with —
    #: VERIFIED equal to the route's (:func:`assert_response_stamps`) before
    #: this object exists; recorded so the sidecar row attests it too.
    intervention_hash: Optional[str] = None
    #: set iff this generation came from the once-only clamped retry: the
    #: ``max_tokens`` the FIRST attempt asked for (the served value is the
    #: clamp) and where the prompt length came from (``error_message`` /
    #: ``tokenize_endpoint``).
    max_tokens_clamped_from: Optional[int] = None
    prompt_tokens_source: Optional[str] = None
    #: True iff ``reasoning`` is non-empty and ``text`` is empty — the
    #: reasoning parser was composed and the item produced NO answer within
    #: ``max_tokens`` (module docstring). The row still scores as-is.
    content_empty_with_reasoning: bool = False


@dataclass
class FanoutStats:
    """The counters of ONE ``agenerate_items`` call (reset per call;
    ``engine.stats`` after it): HTTP attempts including clamped retries,
    non-2xx responses, and the typed outcomes. ``window_mismatches`` lists
    the distinct ``{"threaded", "server"}`` disagreements seen."""

    n_attempts: int = 0
    n_rejected: int = 0
    n_clamped: int = 0
    n_prompt_over_window: int = 0
    n_tokenize_resolved: int = 0
    n_floor_only_unresolvable: int = 0
    n_stamp_mismatch: int = 0
    #: flagged rows: reasoning present, content empty (a result, counted)
    n_content_empty_with_reasoning: int = 0
    #: typed refusals: neither channel under a composed parser (an error)
    n_empty_response: int = 0
    window_mismatches: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"n_attempts": self.n_attempts, "n_rejected": self.n_rejected,
                "n_clamped": self.n_clamped,
                "n_prompt_over_window": self.n_prompt_over_window,
                "n_tokenize_resolved": self.n_tokenize_resolved,
                "n_floor_only_unresolvable": self.n_floor_only_unresolvable,
                "n_stamp_mismatch": self.n_stamp_mismatch,
                "n_content_empty_with_reasoning": self.n_content_empty_with_reasoning,
                "n_empty_response": self.n_empty_response,
                "window_mismatches": [dict(m) for m in self.window_mismatches]}


def stable_seed(*fields: str) -> int:
    """A deterministic per-(task, item, sample) seed in ``[0, 2**31 - 2]``:
    blake2b over the NUL-joined fields (E4: seeds are set client-side per
    (item, sample) and recorded; reproducible in distribution only)."""
    h = hashlib.blake2b("\0".join(str(f) for f in fields).encode("utf-8"),
                        digest_size=8).digest()
    return int.from_bytes(h, "big") % (2 ** 31 - 1)


def is_lens_route(route: RouteEntry) -> bool:
    """True iff the route carries a steering selector — the leg whose
    continued final message MUST be declared (read from the serving
    identity, the same source the proxy's ``_rewrite`` keys on). LENS-ONLY
    by design: the declaration keys it gates are lens vocabulary; a native
    LoRA route sends none. Gate a positive CONTROL on
    :func:`is_steered_route` instead."""
    return cm.parse_serving_identity(route.serving_identity).path == "lens"


#: ``STEERED_ROUTE_PATHS`` — the serving paths that STEER the model (lens or
#: lora) — is re-exported from the proxy
#: vocabulary above, the one literal every bites gate reads.
#: a LORA route's scale-0 twin suffix (the zero adapter under ``<cell>__lora0``,
#: the table's derived control being ``none__lora0``) — the lens path's is
#: ``contracts.SCALE0_TWIN_SUFFIX`` (``__lens0``); one literal, re-exported from
#: the proxy vocabulary (``proxy.LORA_ZERO_CELL_SUFFIX``) so the twin the
#: drivers resolve and the block the table validates spell the same control
SCALE0_LORA_TWIN_SUFFIX = LORA_ZERO_CELL_SUFFIX


def is_steered_route(route: RouteEntry) -> bool:
    """True iff the route serves a STEERING INTERVENTION — its path is in
    :data:`STEERED_ROUTE_PATHS` (lens or lora). The predicate every driver
    gates its steering-bites proof on: a natively served LoRA cell steers the
    model exactly as a lens cell does, so it needs the same inert-serve proof
    against its path-matched twin (:func:`scale0_twin`)."""
    return cm.parse_serving_identity(route.serving_identity).path in STEERED_ROUTE_PATHS


def scale0_twin_suffix(route: RouteEntry) -> Optional[str]:
    """The scale-0 twin suffix for the route's path: ``__lens0`` for a lens
    route, ``__lora0`` for a lora route, ``None`` for an unsteered route (no
    twin applies)."""
    path = cm.parse_serving_identity(route.serving_identity).path
    if path == "lens":
        return C.SCALE0_TWIN_SUFFIX
    if path == "lora":
        return SCALE0_LORA_TWIN_SUFFIX
    return None


#: What a served client may REQUIRE of the proxy it is about to drive
#: (the ``--require-engine`` flag of the wave
#: drivers, default ``lens:stacked`` so every existing invocation is
#: byte-identical): the stacked lens, the per-row lens, or the native
#: multi-LoRA engine. ONE helper (:func:`served_engine_problem`) replaces the
#: four ``ident.get("lens_reduction_mode") == "stacked"`` compares, so no
#: driver can read a native proxy as "not stacked" and stop there, or read a
#: lens proxy as native.
ENGINE_EXPECT_LENS_STACKED = "lens:stacked"
ENGINE_EXPECT_LENS_PER_ROW = "lens:per_row"
ENGINE_EXPECT_LORA_NATIVE = ENGINE_KIND_LORA_NATIVE
ENGINE_EXPECTATIONS: Tuple[str, ...] = (ENGINE_EXPECT_LENS_STACKED, ENGINE_EXPECT_LENS_PER_ROW,
                                        ENGINE_EXPECT_LORA_NATIVE)


def served_engine_problem(ident: Mapping[str, Any], *, expect: str) -> Optional[str]:
    """Why the proxy whose ``/v1/models`` ``petri_proxy`` identity block is
    ``ident`` is NOT the engine ``expect`` names — one of
    :data:`ENGINE_EXPECTATIONS` — or ``None`` when it is. Read off the two
    facts the proxy echoes ONLY when its launcher attested them (nothing is
    defaulted here, exactly as the proxy defaults nothing):

    - ``lens:<mode>``: :data:`proxy.LENS_REDUCTION_MODE_KEY` must be PRESENT
      and equal ``<mode>`` (the four compares this replaces were this exact
      equality — an absent key never reads as per_row here), AND no
      :data:`proxy.ENGINE_KEY` block may name the native engine (a native
      proxy carries no lens hook, so it cannot be the lens under any mode);
    - ``lora_native``: :data:`proxy.ENGINE_KEY` must be a mapping whose
      ``kind`` is :data:`proxy.ENGINE_KIND_LORA_NATIVE`, AND
      :data:`proxy.LENS_REDUCTION_MODE_KEY` must be ABSENT (a native proxy
      attesting a lens arithmetic is a torn identity, refused rather than
      read around).

    An ``expect`` outside the vocabulary is a ``ValueError`` (a driver flag
    typo must not read as "no requirement"); a non-mapping ``ident`` is a
    problem (the client's ``load`` never captured an identity block)."""
    if expect not in ENGINE_EXPECTATIONS:
        raise ValueError(f"served_engine_problem: expect must be one of {list(ENGINE_EXPECTATIONS)}, "
                         f"got {expect!r}")
    if not isinstance(ident, Mapping):
        return (f"the proxy exposes no petri_proxy identity block ({type(ident).__name__}) — "
                f"the served engine cannot be verified against {expect!r}")
    engine = ident.get(ENGINE_KEY)
    kind = engine.get("kind") if isinstance(engine, Mapping) else None
    mode_present = LENS_REDUCTION_MODE_KEY in ident
    mode = ident.get(LENS_REDUCTION_MODE_KEY)
    if expect == ENGINE_EXPECT_LORA_NATIVE:
        if engine is None:
            return (f"the proxy attests no {ENGINE_KEY!r} block — not a native multi-LoRA standup "
                    f"(its identity names {sorted(ident)})")
        if kind != ENGINE_KIND_LORA_NATIVE:
            return (f"the proxy's {ENGINE_KEY}.kind is {kind!r}, not {ENGINE_KIND_LORA_NATIVE!r}")
        if mode_present:
            return (f"the proxy attests {ENGINE_KEY}.kind={ENGINE_KIND_LORA_NATIVE!r} AND "
                    f"{LENS_REDUCTION_MODE_KEY}={mode!r} — a native standup has no lens hook to "
                    "attest a reduction mode for; a torn identity, refusing")
        return None
    want_mode = expect.split(":", 1)[1]
    if kind == ENGINE_KIND_LORA_NATIVE:
        return (f"the proxy is the native multi-LoRA engine ({ENGINE_KEY}.kind="
                f"{ENGINE_KIND_LORA_NATIVE!r}), not the lens under {want_mode!r}")
    if mode != want_mode:
        return (f"the proxy attests {LENS_REDUCTION_MODE_KEY}={mode!r}, not {want_mode!r}"
                + ("" if mode_present else " (the key is absent: nothing was attested)"))
    return None


def routes_by_cell(routes: Sequence[RouteEntry]) -> Dict[str, RouteEntry]:
    """``{cell_id: entry}`` over a routing table's ``routes``."""
    return {r.cell_id: r for r in routes}


def scale0_twin(route: RouteEntry, routes: Sequence[RouteEntry]) -> Optional[RouteEntry]:
    """The scale-0 twin a steered route's positive control compares against,
    PATH-MATCHED: for a LENS route the route's own ``<cell_id>__lens0`` entry,
    else the table's ``none__lens0`` (``contracts.SCALE0_TWIN_SUFFIX`` /
    ``ROUTE_NONE_CELL``) — exactly as before; for a LORA route its own
    ``<cell_id>__lora0``, else the table's ``none__lora0`` (the zero
    adapter, :data:`SCALE0_LORA_TWIN_SUFFIX`).
    A lens route never resolves to a lora twin or vice versa: the twin must
    exercise the SAME serving machinery at zero, or the comparison proves
    nothing about that machinery. ``None`` for an unsteered route (no twin
    applies) and when the table exports neither spelling — the caller refuses
    in its own words (a served number without its positive control is never
    run on faith)."""
    suffix = scale0_twin_suffix(route)
    if suffix is None:
        return None
    by_cell = routes_by_cell(routes)
    return (by_cell.get(f"{route.cell_id}{suffix}")
            or by_cell.get(f"{C.ROUTE_NONE_CELL}{suffix}"))


#: the type of :func:`bounded_gather`'s ``on_done`` callback: ``(key, outcome)`` where
#: ``outcome`` is the thunk's result, or the exception it raised
OnDone = Callable[[Any, Any], None]


async def bounded_gather(work: Sequence[Tuple[Any, Callable[[], Awaitable[Any]]]], *,
                         concurrency: int,
                         on_done: Optional[OnDone] = None) -> Tuple[Dict[Any, Any], List[Tuple[Any, str]]]:
    """The served clients' ONE fan-out discipline: ``work`` is ``[(key,
    thunk)]``, the thunks run under a semaphore of ``concurrency``, and every
    failure is COLLECTED as ``(key, "ExcType: message")`` — never dropped,
    never retried client-side (a proxy refusal must surface, not be
    re-rolled). Returns ``({key: result}, errors)``.

    ``on_done`` (ADDITIVE; default None = the fan-out exactly as it was): a
    SYNCHRONOUS callback ``on_done(key, outcome)`` invoked once per thunk IN
    COMPLETION ORDER — the moment each thunk finishes, after its result (or
    its collected error) is recorded and before the semaphore slot is
    released — so a caller can publish per-result facts (the sampled
    controls' durable ``controls_events.jsonl`` stream the ladder tailer
    reads) WITHOUT
    waiting for the whole fan-out; the returned maps are the same with and
    without it. ``outcome`` is the thunk's return value, or the exception it
    raised (the failure is ALSO collected in ``errors``, as before). The
    callback runs on the event-loop thread and must be quick (an fsynced
    append, a queue put). An exception it raises is NOT a request failure: it
    is never collected, it cancels the remaining thunks and propagates out
    of ``bounded_gather`` — a durable publication that cannot be written
    must stop the run rather than lose an item silently."""
    sem = asyncio.Semaphore(max(1, int(concurrency)))
    results: Dict[Any, Any] = {}
    errors: List[Tuple[Any, str]] = []

    async def one(key, fn):
        async with sem:
            try:
                outcome = results[key] = await fn()
            except Exception as e:  # noqa: BLE001 — collected, reported loudly
                errors.append((key, f"{type(e).__name__}: {e}"))
                outcome = e
            if on_done is not None:
                on_done(key, outcome)

    if on_done is None:
        await asyncio.gather(*(one(k, fn) for k, fn in work))
        return results, errors
    tasks = [asyncio.ensure_future(one(k, fn)) for k, fn in work]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            t.cancel()
        raise
    return results, errors


def ledger_headers(task: str, item_id: str, sample_idx: int = 0) -> Dict[str, str]:
    """The two ``X-Tinycap-*`` headers (P27) a served request carries so the
    usage ledger attributes its row: ``contracts.HEADER_TASK`` = the row task,
    ``contracts.HEADER_SAMPLE`` = ``sample_header_value(item_id, sample_idx)``."""
    return {C.HEADER_TASK: str(task), C.HEADER_SAMPLE: C.sample_header_value(item_id, sample_idx)}


def assert_client_body(body: Mapping[str, Any], *, lens: bool) -> None:
    """The client-side statement of the proxy contract (E3/E4/E6; the
    fit-support-transfer contract §2.5.4–§2.5.6, §2.5.12). Raises
    :class:`ProxyEngineError` on: a forbidden key, a truthy ``stream``, an
    ``n``, an empty message list, a continued final message that is not an
    assistant turn or that leaves ``add_generation_prompt`` unset/true, a
    declaration on a non-lens route, or a continued message on a lens
    route WITHOUT the declaration; and, for the thinking-ON declaration
    ``petri_thinking_on`` (``contracts.OOS_THINKING_DECLARATION_KEY``): present
    with any value but the JSON boolean ``true`` (``false`` / ``null`` are
    "not a declaration" / ``thinking_on_invalid``), present on a non-lens
    route (``thinking_on_not_lens``), present without
    ``chat_template_kwargs.enable_thinking: true`` (``thinking_on_mismatch``),
    or ABSENT on a lens route that sends ``enable_thinking: true``
    (``thinking_on_undeclared`` — required on EVERY such request, regardless
    of support). Pure; the proxy re-checks server-side."""
    smuggled = sorted(set(body) & set(FORBIDDEN_CLIENT_KEYS))
    if smuggled:
        raise ProxyEngineError(
            f"request carries proxy-owned key(s) {smuggled} — "
            f"FORBIDDEN_CLIENT_KEYS are never sent (E3)")
    if body.get("stream"):
        raise ProxyEngineError("stream=true is refused by the proxy; the engine sends stream=false")
    if "n" in body:
        raise ProxyEngineError("n= is never sent on a served route (E4: independent requests)")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ProxyEngineError("request has no messages")
    _assert_thinking_on_declaration(body, lens=lens)
    continued = bool(body.get("continue_final_message"))
    declared = body.get(RESPONSE_PREFILL_KEY)
    if declared is not None and not isinstance(declared, bool):
        raise ProxyEngineError(f"{RESPONSE_PREFILL_KEY} must be a bool, got {declared!r}")
    if continued:
        if messages[-1].get("role") != "assistant":
            raise ProxyEngineError(
                "continue_final_message set but the final message is not an assistant turn")
        if body.get("add_generation_prompt") is not False:
            raise ProxyEngineError(
                "a continued final message must state add_generation_prompt: false")
        if lens and declared is not True:
            raise ProxyEngineError(
                f"lens route + continued final message without {RESPONSE_PREFILL_KEY}: "
                "the served number would be an undeclared (refused) seeded render (E6)")
    elif declared:
        raise ProxyEngineError(
            f"{RESPONSE_PREFILL_KEY} declared on a request that does not continue "
            "its final message")
    if declared and not lens:
        raise ProxyEngineError(
            f"{RESPONSE_PREFILL_KEY} declared on a non-lens route — there is no "
            "position mask there and the proxy refuses the claim")


def _assert_thinking_on_declaration(body: Mapping[str, Any], *, lens: bool) -> None:
    """The ``petri_thinking_on`` half of :func:`assert_client_body` (contract
    §2.5.4 / §2.5.5 / §2.5.6 / §2.5.12): the key rides a body iff the body
    sends ``chat_template_kwargs.enable_thinking: true`` on a LENS route, and
    then only as the JSON boolean ``true`` — ``contracts.thinking_on_declaration``
    is the ONE predicate every builder applies; this is its mirror at the
    client's door, so a builder that forgets the key (or a caller that adds it
    where it does not belong) is refused before the wire."""
    key = C.OOS_THINKING_DECLARATION_KEY
    kwargs = body.get(C.CHAT_TEMPLATE_KWARGS_KEY)
    thinking_on = isinstance(kwargs, Mapping) and kwargs.get(C.ENABLE_THINKING_KEY) is True
    if key in body:
        value = body[key]
        if not (isinstance(value, bool) and value is True):
            raise ProxyEngineError(
                f"{key} = {value!r} ({type(value).__name__}): the declaration is the JSON boolean "
                "true or absent — never false (not a declaration: thinking_on_undeclared), never "
                "null or another type (thinking_on_invalid) (contract §2.5.1, §2.5.12)")
        if not lens:
            raise ProxyEngineError(
                f"{key} declared on a non-lens route — the base route accepts enable_thinking "
                "undeclared and refuses the declaration (400 thinking_on_not_lens, contract §2.5.5)")
        if not thinking_on:
            raise ProxyEngineError(
                f"{key} declared without {C.CHAT_TEMPLATE_KWARGS_KEY}.{C.ENABLE_THINKING_KEY}: true "
                "(400 thinking_on_mismatch, contract §2.5.6)")
    elif lens and thinking_on:
        raise ProxyEngineError(
            f"a lens-route request sends {C.ENABLE_THINKING_KEY}: true without {key}: true — "
            "required on EVERY such request, regardless of support (400 thinking_on_undeclared, "
            "contract §2.5.4); every builder applies contracts.thinking_on_declaration")


def parse_chat_response(resp: Mapping[str, Any]) -> Dict[str, Any]:
    """The facts of one non-streaming chat completion: ``text`` (content,
    ``""`` when null), ``finish_reason``, ``prompt_tokens`` /
    ``completion_tokens`` (from ``usage``; None when absent), the
    ``reasoning`` channel (``reasoning_content``, or ``reasoning`` on newer
    servers; None when absent — BOTH channels are always read, module
    docstring) and ``content_empty_with_reasoning`` (reasoning non-empty,
    content empty/None). A response without exactly one choice, or whose
    choice has no ``message``, is malformed (refused). Pure: the
    neither-channel refusal needs the engine's parser knob and lives in
    the fan-out."""
    choices = resp.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ProxyEngineError(
            f"expected exactly one choice, got {len(choices) if isinstance(choices, list) else choices!r}")
    choice = choices[0]
    msg = choice.get("message") if isinstance(choice, Mapping) else None
    if not isinstance(msg, Mapping):
        raise ProxyEngineError("choice carries no message object")
    content = msg.get("content")
    if content is not None and not isinstance(content, str):
        raise ProxyEngineError(f"message.content is not a string: {type(content).__name__}")
    reasoning = msg.get("reasoning_content")
    if reasoning is None:
        reasoning = msg.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, str):
        raise ProxyEngineError("reasoning channel is not a string")
    usage = resp.get("usage") if isinstance(resp.get("usage"), Mapping) else {}

    def _int(k):
        v = usage.get(k)
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    return {"text": content or "", "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": _int("prompt_tokens"),
            "completion_tokens": _int("completion_tokens"),
            "reasoning": reasoning,
            "content_empty_with_reasoning": (
                channel_verdict(content, reasoning) == CHANNEL_FLAGGED)}


#: :func:`channel_verdict` outcomes for one message's two channels; the third
#: is :data:`ERR_EMPTY_RESPONSE` (refused).
CHANNEL_OK = "ok"
CHANNEL_FLAGGED = "content_empty_with_reasoning"


def channel_verdict(content: Optional[str], reasoning: Optional[str], *,
                    has_tool_call: bool = False,
                    reasoning_parser: Optional[str] = None) -> str:
    """The ONE reading of a message's ``content`` / ``reasoning`` channels
    (module docstring), shared by this engine and the Inspect provider +
    adapter: :data:`ERR_EMPTY_RESPONSE` when a parser is composed and the
    message carries neither content nor reasoning nor a tool call
    (refused — an empty answer cannot be told from text the parser lost);
    :data:`CHANNEL_FLAGGED` when reasoning is present and content empty/None
    (a row, flagged, scored as-is); :data:`CHANNEL_OK` otherwise. ``content``
    None and ``""`` are the same case; an empty ``reasoning`` is no
    reasoning. Pure."""
    if reasoning_parser and not content and not reasoning and not has_tool_call:
        return ERR_EMPTY_RESPONSE
    if reasoning and not content:
        return CHANNEL_FLAGGED
    return CHANNEL_OK


#: The ``token`` form a ``return_tokens_as_token_ids`` completion carries.
TOKEN_ID_PREFIX = "token_id:"


def first_position_logprobs_by_id(resp: Mapping[str, Any], *, what: str) -> Dict[int, float]:
    """``{token_id: logprob}`` at the FIRST generated position of a
    ``logprobs`` + ``top_logprobs`` + ``return_tokens_as_token_ids`` chat
    completion: the top-K entries plus the sampled token (first occurrence
    of an id wins). Refuses (:class:`ProxyEngineError`) a response without
    logprobs at that position, a token not in ``token_id:<int>`` form (the
    server did not honor ``return_tokens_as_token_ids`` — letters are never
    matched by surface string) and a non-finite / non-numeric logprob. The
    shared half of the served letter reader (the driver's MMLU letter
    scorer); the per-letter selection is its own."""
    try:
        choice = resp["choices"][0]
        content = (choice.get("logprobs") or {}).get("content") or []
        first = content[0]
    except (KeyError, IndexError, TypeError):
        raise ProxyEngineError(f"{what}: response carries no logprobs at the first generated "
                               "position — the server did not honor logprobs/top_logprobs")
    entries = list(first.get("top_logprobs") or [])
    entries.append({"token": first.get("token"), "logprob": first.get("logprob")})
    by_id: Dict[int, float] = {}
    for e in entries:
        token = e.get("token")
        if not isinstance(token, str) or not token.startswith(TOKEN_ID_PREFIX):
            raise ProxyEngineError(f"{what}: logprob token {token!r} is not token_id:<int> — "
                                   "return_tokens_as_token_ids was not honored; refusing to "
                                   "match letters by surface string")
        lp = e.get("logprob")
        if not isinstance(lp, (int, float)) or isinstance(lp, bool) or not math.isfinite(lp):
            raise ProxyEngineError(f"{what}: non-finite/non-numeric logprob {lp!r}")
        by_id.setdefault(int(token[len(TOKEN_ID_PREFIX):]), float(lp))
    return by_id


#: The four attestation stamps every proxy chat response carries and this
#: engine verifies: the top-level ``model`` and ``petri_intervention_hash``
#: (the audit fields) plus the ``petri`` block's ``cell_id`` and
#: ``serving_identity``. Dotted names are the nested block's members.
STAMP_FIELDS: Tuple[str, ...] = ("model", "petri_intervention_hash",
                                 "petri.cell_id", "petri.serving_identity")
#: The routing-table attribute each stamp must equal, in :data:`STAMP_FIELDS`
#: order (a ``RouteEntry`` carries exactly these four).
ROUTE_STAMP_ATTRS: Tuple[str, ...] = ("virtual_model", "intervention_hash",
                                      "cell_id", "serving_identity")


def expected_stamps(route: RouteEntry) -> Dict[str, Any]:
    """What the proxy stamps a response served on ``route`` with — the raw
    strings of the routing-table entry (``serving_identity`` is stamped as
    the ``<backend>+<path>+<mode>`` string, not its parsed form)."""
    return {f: getattr(route, a) for f, a in zip(STAMP_FIELDS, ROUTE_STAMP_ATTRS)}


def response_stamps(resp: Mapping[str, Any]) -> Dict[str, Any]:
    """The :data:`STAMP_FIELDS` as they appear on ``resp`` — ``None`` for
    an absent field (a missing ``petri`` block yields None for both of its
    members; a non-object ``petri`` counts as missing)."""
    petri = resp.get("petri")
    petri = petri if isinstance(petri, Mapping) else {}
    return {"model": resp.get("model"),
            "petri_intervention_hash": resp.get("petri_intervention_hash"),
            "petri.cell_id": petri.get("cell_id"),
            "petri.serving_identity": petri.get("serving_identity")}


def assert_response_stamps(resp: Mapping[str, Any], route: RouteEntry, *,
                           item_id: str = "") -> Dict[str, Any]:
    """Verify a 200's attestation against ``route`` (module docstring):
    every stamp in :data:`STAMP_FIELDS` must be PRESENT and ``==`` the
    route's value — string equality, like with like, since the proxy stamps
    the routing-table strings verbatim. Returns the stamps seen; raises
    :class:`StampMismatch` (naming the missing / mismatched fields with the
    values seen vs expected) otherwise. Pure."""
    return _check_stamps(expected_stamps(route), response_stamps(resp), STAMP_FIELDS, item_id)


def _check_stamps(expected: Dict[str, Any], seen: Dict[str, Any], fields: Sequence[str],
                  item_id: str) -> Dict[str, Any]:
    """The shared verdict of both stamp checks: every field PRESENT and
    ``==`` its expectation, else :class:`StampMismatch` naming the missing /
    mismatched ones; returns ``seen``."""
    missing = [k for k in fields if seen[k] is None]
    mismatched = [k for k in fields if seen[k] is not None and seen[k] != expected[k]]
    if missing or mismatched:
        raise StampMismatch(item_id, expected=expected, seen=seen,
                            missing=missing, mismatched=mismatched)
    return seen


#: What ``RoutingProxy.handle_tokenize`` stamps a ``/v1/tokenize`` reply
#: with: ``model`` (the VIRTUAL name it resolved the request under) and
#: nothing else — a tokenize generates nothing, so there is no
#: ``petri_intervention_hash`` and no ``petri`` block on it (proxy.py:
#: ``out["model"] = vmodel``). Verified against the proxy source, not assumed.
TOKENIZE_STAMP_FIELDS: Tuple[str, ...] = ("model",)


def assert_tokenize_stamp(resp: Mapping[str, Any], route: RouteEntry, *,
                          item_id: str = "") -> Dict[str, Any]:
    """Verify a ``/v1/tokenize`` 200 against ``route`` (E10 N2): the reply's
    ``model`` must be PRESENT and ``==`` the route's virtual model — the
    one stamp the proxy puts on a tokenize reply (:data:`TOKENIZE_STAMP_FIELDS`).
    A prefill scorer locates its scored span on THIS render, so a count that
    came back under another virtual model (a stale table, a second live
    standup on the same loopback host) would place the span on the wrong
    prompt with every number looking fine. Same discipline as
    :func:`assert_response_stamps`: string equality, like with like; raises
    :class:`StampMismatch` naming the missing / mismatched field. Pure."""
    return _check_stamps({"model": route.virtual_model},
                         {"model": resp.get("model") if isinstance(resp, Mapping) else None},
                         TOKENIZE_STAMP_FIELDS, item_id)


def tokenize_body_for(body: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``/v1/tokenize`` body that counts EXACTLY the render of chat
    ``body`` (a floor-only rejection is resolved on the identical render,
    never a generic one). A continued final message (the prefilled
    instrument) reuses Lane A's
    :func:`~sisteer.capeval.serving.prefill_logprob.build_tokenize_request`
    — ``continue_final_message: true``, ``add_generation_prompt: false``,
    the assistant seed message and the same ``chat_template_kwargs``; the
    open variant is the served clients' generation-prompt payload
    (:func:`~sisteer.capeval.serving.served_window.tokenize_payload`) plus
    the same ``chat_template_kwargs`` (a thinking-mode kwarg changes the
    render, so it must ride the count too). Both carry the VIRTUAL model —
    the proxy's ``handle_tokenize`` resolves it exactly as chat does and
    forwards every one of these keys (``TOKENIZE_FORWARD_KEYS``)."""
    kwargs = body.get(C.CHAT_TEMPLATE_KWARGS_KEY)
    if not isinstance(kwargs, Mapping) or C.ENABLE_THINKING_KEY not in kwargs:
        raise ProxyEngineError(
            f"chat body carries no {C.CHAT_TEMPLATE_KWARGS_KEY}.{C.ENABLE_THINKING_KEY} — "
            "the tokenize render could not match the chat render")
    if body.get("continue_final_message"):
        return pl.build_tokenize_request(
            virtual_model=body["model"], messages=body["messages"],
            enable_thinking=bool(kwargs[C.ENABLE_THINKING_KEY]))
    out = sw.tokenize_payload(body)
    out[C.CHAT_TEMPLATE_KWARGS_KEY] = dict(kwargs)
    return out


class ProxyGenerateEngine:
    """``Engine.generate`` over the served route for chat items.

    ``base_url`` is the ledger's or the RoutingProxy's ``/v1`` base (P36:
    tinycap traffic goes through the ledger when one is given). ``route``
    is the cell's routing-table entry — its ``virtual_model`` is READ, never
    typed (P20). ``system`` is the SI text for a prompted cell, None
    otherwise. ``thinking`` is ``on``/``off`` (P33: the caller states it).
    ``task`` is the row task name the ledger files requests under.
    ``max_model_len`` is the launcher's resolved serving window — PROVENANCE
    and a cross-check for the once-only clamp (the server-stated window in
    a rejection governs; a disagreement is recorded in ``stats``), None
    when the caller has none. ``reasoning_parser`` is the parser the
    standup COMPOSED (``endpoint.json`` ``reasoning_parser``, read by the
    driver — never guessed from the arm), None when none was: it arms the
    neither-channel refusal (module docstring) and is recorded in
    ``gen_meta``; the flagged-row bookkeeping needs no knob. ``sampling``
    (U8 / PW-sampling) is the :class:`contracts.SamplingRow` every request is
    drawn under — the model's frozen card row under ``--sampling card``,
    :data:`contracts.GREEDY_SAMPLING` when None (0.0, no nucleus: what every
    pre-U8 body sent); its three numbers go onto every body
    (:meth:`build_request`) and k samples of one item are k
    :class:`ChatItem`s with distinct ``sample_idx`` (distinct seeds:
    :func:`stable_seed`), never an ``n``."""

    def __init__(self, base_url: str, *, route: RouteEntry, system: Optional[str],
                 thinking: str, concurrency: int, task: str,
                 api_key: Optional[str] = None,
                 timeout_s: float = AIOHTTP_TOTAL_TIMEOUT_S,
                 transport: Optional[Transport] = None,
                 max_model_len: Optional[int] = None,
                 reasoning_parser: Optional[str] = None,
                 sampling: Optional[C.SamplingRow] = None):
        self.base_url = _canonical_v1_base(base_url)
        if sampling is not None and not isinstance(sampling, C.SamplingRow):
            raise ProxyEngineError(
                f"sampling must be a contracts.SamplingRow or None, got {type(sampling).__name__}")
        self.sampling = sampling if sampling is not None else C.GREEDY_SAMPLING  # the registry's greedy policy row
        self.route = route
        self.system = system or None
        C.enable_thinking(thinking)  # refuses anything but on/off
        self.thinking = thinking
        self.concurrency = max(1, int(concurrency))
        self.task = task
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.declare = is_lens_route(route)
        self._transport = transport
        self.max_model_len = int(max_model_len) if max_model_len is not None else None
        if reasoning_parser is not None and (
                not isinstance(reasoning_parser, str) or not reasoning_parser.strip()):
            raise ProxyEngineError(
                f"reasoning_parser={reasoning_parser!r} must be a non-empty parser name or None")
        self.reasoning_parser = reasoning_parser
        #: the counters of the most recent ``agenerate_items`` call
        self.stats = FanoutStats()
        self.name = f"proxy_generate:{route.serving_identity}"

    # -- request construction (pure) ---------------------------------------- #

    def seed_for(self, item: ChatItem) -> int:
        return stable_seed(self.task, item.item_id, str(item.sample_idx))

    def build_request(self, item: ChatItem, *, max_tokens: int,
                      stop: Sequence[str] = ()) -> Dict[str, Any]:
        """One chat body for ``item`` (module docstring). Asserted against
        the client contract before it is returned."""
        messages: List[Dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": item.user})
        body: Dict[str, Any] = {
            "model": self.route.virtual_model,
            "messages": messages,
            # the engine's SamplingRow (U8): temperature always; top_p / top_k
            # only when the row sets them (None = no key, vLLM's defaults)
            "temperature": float(self.sampling.temperature),
            "max_tokens": int(max_tokens),
            "seed": self.seed_for(item),
            "stream": False,
            # the arm's chat_template_kwargs and (contract §2.5.4) the
            # thinking-ON declaration on EVERY lens-route request that sends
            # enable_thinking: true — {} on the base route and under OFF
            **C.render_fragment(lens=self.declare, arm=self.thinking),
        }
        if self.sampling.top_p is not None:
            body["top_p"] = float(self.sampling.top_p)
        if self.sampling.top_k is not None:
            body["top_k"] = int(self.sampling.top_k)
        if stop:
            body["stop"] = list(stop)
        if item.assistant_prefill is not None:
            if not item.assistant_prefill:
                raise ProxyEngineError(
                    f"{item.item_id}: an empty assistant prefill continues nothing; "
                    "use assistant_prefill=None for the open (no-prefill) variant")
            messages.append({"role": "assistant", "content": item.assistant_prefill})
            body["continue_final_message"] = True
            body["add_generation_prompt"] = False
            if self.declare:
                body[RESPONSE_PREFILL_KEY] = True
        assert_client_body(body, lens=self.declare)
        return body

    def headers(self, item: ChatItem) -> Dict[str, str]:
        h = {"Content-Type": "application/json",
             **ledger_headers(self.task, item.item_id, item.sample_idx)}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    @staticmethod
    def resolve_gen_cfg(gen_cfg: Mapping[str, Any]) -> Tuple[int, Tuple[str, ...]]:
        """``(max_tokens, stop)`` from a tier1-shaped gen config; refuses
        ``do_sample: true`` and unknown keys. The gen config is the HF path's
        shape and carries the CAP and the stop strings only; how the served
        engine DRAWS is its :class:`contracts.SamplingRow` (``sampling=``),
        never a knob smuggled through this mapping."""
        unknown = sorted(set(gen_cfg) - set(GEN_CFG_KEYS))
        if unknown:
            raise ProxyEngineError(
                f"gen_cfg carries keys this engine does not implement: {unknown} "
                f"(known: {list(GEN_CFG_KEYS)}) — refusing to ignore a sampling knob")
        if gen_cfg.get("do_sample", False):
            raise ProxyEngineError(
                "do_sample=true: the gen config is the HF path's shape and carries no draw — the "
                "served engine samples by its SamplingRow (sampling=), which is greedy by default")
        if "max_new_tokens" not in gen_cfg:
            raise ProxyEngineError("gen_cfg lacks max_new_tokens")
        stop = tuple(gen_cfg.get("until") or ())
        if any(not isinstance(s, str) or not s for s in stop):
            raise ProxyEngineError(f"until must be non-empty strings, got {stop!r}")
        return int(gen_cfg["max_new_tokens"]), stop

    # -- transport ------------------------------------------------------------ #

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def tokenize_url(self) -> str:
        """The proxy's read-only ``/v1/tokenize`` (the ledger passes it
        through un-ledgered) — used ONLY to resolve a floor-only rejection."""
        return f"{self.base_url}/tokenize"

    async def _default_transport_session(self):
        import aiohttp  # heavy-lane import; tests inject a transport

        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        return aiohttp.ClientSession(timeout=timeout)

    # -- the once-only clamp --------------------------------------------------------------- #

    async def _clamp_plan(self, status: int, text: str, body: Mapping[str, Any],
                          headers: Mapping[str, str], transport: Transport
                          ) -> Optional[sw.ClampPlan]:
        """``None`` unless ``(status, text)`` is a context-length 400 by
        :func:`served_window.parse_context_overflow`'s precise rule (the
        explicit status is passed, so no other 400 — a contract refusal, a
        salt collision — can read as an overflow). A floor-only rejection
        is resolved on the IDENTICAL render through ``/v1/tokenize``; a
        tokenize failure of any kind makes it the typed
        ``floor_only_unresolvable`` — never a guess from the floor."""
        overflow = sw.parse_context_overflow(text, http_status=status)
        if overflow is None:
            return None
        plan = sw.clamp_plan(overflow, requested_max_tokens=int(body["max_tokens"]),
                             window=self.max_model_len)
        if not plan.needs_exact_prompt_tokens:
            return plan
        try:
            t_status, t_text = await transport(self.tokenize_url, headers, tokenize_body_for(body))
            if t_status != 200:
                raise ProxyEngineError(f"/v1/tokenize returned {t_status}: {t_text[:200]}")
            count = sw.exact_prompt_tokens(json.loads(t_text))
        except Exception as e:  # noqa: BLE001 — 404 / 501 / connection / shape: never guess
            self.stats.n_floor_only_unresolvable += 1
            return plan.unresolvable(f"/v1/tokenize failed: {type(e).__name__}: {str(e)[:200]}")
        self.stats.n_tokenize_resolved += 1
        return plan.with_exact_prompt_tokens(count)

    def _note_mismatch(self, plan: sw.ClampPlan) -> None:
        mm = plan.mismatch_record()
        if mm is None or mm in self.stats.window_mismatches:
            return
        self.stats.window_mismatches.append(mm)
        print(f"[proxy-engine] WINDOW MISMATCH for {self.route.virtual_model}: the caller "
              f"threaded max_model_len={mm['threaded']} but the server states {mm['server']} "
              "— the server's value governs the clamp; fix the launcher's geometry provenance",
              flush=True)

    async def agenerate_items(self, items: Sequence[ChatItem], gen_cfg: Mapping[str, Any]
                              ) -> Tuple[Dict[str, GenResult], List[ItemError]]:
        """Fan ``items`` out under the concurrency semaphore. Returns
        ``(results_by_generation_key, errors)`` — results keyed by
        :func:`generation_key` (``<item_id>/<sample_idx>``: k samples of one
        item are k entries, U8 / PW-sampling); errors are :class:`ItemError`
        records (``(item_id, message)`` by index), collected, never
        re-rolled; the caller decides (the runner refuses to write
        samples.jsonl on any error and resumes from its sidecar). The
        counters of this call land in ``self.stats``. Per item, in order:
        the request; on a context-length 400 the once-only clamp (module
        docstring); on a 200 the attestation stamps are verified against
        the route BEFORE the choice is read; both channels are read; a
        neither-channel response under a composed parser is the typed
        :class:`EmptyResponse`; only then a :class:`GenResult` (flagged
        when reasoning is present and content empty)."""
        max_tokens, stop = self.resolve_gen_cfg(gen_cfg)
        keys = [generation_key(i.item_id, i.sample_idx) for i in items]
        if len(keys) != len(set(keys)):
            raise ProxyEngineError("duplicate (item_id, sample_idx) in one fan-out")
        sem = asyncio.Semaphore(self.concurrency)
        results: Dict[str, GenResult] = {}
        errors: List[ItemError] = []
        stats = self.stats = FanoutStats()
        session = None
        transport = self._transport
        if transport is None:
            session = await self._default_transport_session()

            async def transport(url, headers, body):  # type: ignore[misc]
                async with session.post(url, headers=dict(headers),
                                        data=json.dumps(body).encode("utf-8")) as r:
                    return r.status, await r.text()

        async def post(headers, body) -> Tuple[int, str]:
            stats.n_attempts += 1
            status, text = await transport(self.chat_url, headers, body)
            if status != 200:
                stats.n_rejected += 1
            return status, text

        async def one(item: ChatItem):
            def _err(message: str, kind: str, record: Optional[Dict[str, Any]] = None) -> ItemError:
                # the failed GENERATION, named by (item, sample)
                return ItemError(item_id=item.item_id, message=message, kind=kind, record=record,
                                 sample_idx=item.sample_idx)

            body = self.build_request(item, max_tokens=max_tokens, stop=stop)
            headers = self.headers(item)
            t0 = time.time()
            clamped_from: Optional[int] = None
            prompt_source: Optional[str] = None
            async with sem:
                try:
                    status, text = await post(headers, body)
                    if status != 200:
                        plan = await self._clamp_plan(status, text, body, headers, transport)
                        if plan is None:
                            errors.append(_err(f"proxy returned {status}: {text[:400]}",
                                                    ERR_HTTP))
                            return
                        self._note_mismatch(plan)
                        failed: Optional[sw.ClampPlan] = None
                        if plan.action == sw.RETRY:
                            # ONE clamped retry: the same body, only max_tokens
                            # changed (same seed, same render, same declaration).
                            # The rejected attempt produced no completion, so
                            # this cannot double-count.
                            retry = dict(body)
                            retry["max_tokens"] = int(plan.max_tokens)
                            status, text = await post(headers, retry)
                            if status != 200:
                                plan2 = await self._clamp_plan(status, text, retry, headers,
                                                               transport)
                                if plan2 is None:
                                    errors.append(_err(f"proxy returned {status} on the clamped retry "
                                        f"(max_tokens {body['max_tokens']} -> "
                                        f"{retry['max_tokens']}): {text[:400]}", ERR_HTTP))
                                    return
                                failed = plan2.failed(
                                    f"rejected again after the single clamped retry "
                                    f"({body['max_tokens']} -> {retry['max_tokens']}): "
                                    f"{plan2.reason}")
                            else:
                                clamped_from = int(body["max_tokens"])
                                prompt_source = plan.prompt_tokens_source
                                body = retry
                        else:
                            failed = plan
                        if failed is not None:
                            err = PromptOverWindow.from_plan(
                                failed, requested_max_tokens=int(max_tokens))
                            stats.n_prompt_over_window += 1
                            errors.append(_err(f"{type(err).__name__}: {err}",
                                                    ERR_PROMPT_OVER_WINDOW, err.record()))
                            return
                except Exception as e:  # noqa: BLE001 — collected, reported loudly
                    errors.append(_err(f"{type(e).__name__}: {e}",
                                            ERR_TRANSPORT))
                    return
            try:
                resp = json.loads(text)
                if not isinstance(resp, Mapping):
                    raise ProxyEngineError(f"response is not a JSON object: {type(resp).__name__}")
            except (ValueError, ProxyEngineError) as e:
                errors.append(_err(f"malformed response: {e}", ERR_MALFORMED))
                return
            # the stamps must attest THIS route before anything in the body
            # is read as a completion.
            try:
                stamps = assert_response_stamps(resp, self.route, item_id=item.item_id)
            except StampMismatch as e:
                stats.n_stamp_mismatch += 1
                errors.append(_err(f"{type(e).__name__}: {e}",
                                        ERR_STAMP_MISMATCH, e.record()))
                return
            try:
                parsed = parse_chat_response(resp)
            except ProxyEngineError as e:
                errors.append(_err(f"malformed response: {e}", ERR_MALFORMED))
                return
            if channel_verdict(parsed["text"], parsed["reasoning"],
                               reasoning_parser=self.reasoning_parser) == ERR_EMPTY_RESPONSE:
                # NEITHER channel under a composed parser: refused, never a row
                # (module docstring). Without a parser an empty content is an
                # ordinary empty (wrong) answer and becomes a row like any other.
                msg = resp["choices"][0]["message"]
                err = EmptyResponse(item.item_id, reasoning_parser=self.reasoning_parser,
                                    finish_reason=parsed["finish_reason"],
                                    completion_tokens=parsed["completion_tokens"],
                                    content_null=msg.get("content") is None)
                stats.n_empty_response += 1
                errors.append(_err(f"{type(err).__name__}: {err}",
                                        ERR_EMPTY_RESPONSE, err.record()))
                return
            if parsed["content_empty_with_reasoning"]:
                stats.n_content_empty_with_reasoning += 1
            if clamped_from is not None:
                stats.n_clamped += 1
            results[generation_key(item.item_id, item.sample_idx)] = GenResult(
                item_id=item.item_id, sample_idx=item.sample_idx, seed=body["seed"],
                latency_s=time.time() - t0,
                intervention_hash=stamps["petri_intervention_hash"],
                max_tokens_clamped_from=clamped_from, prompt_tokens_source=prompt_source,
                **parsed)

        try:
            await asyncio.gather(*(one(i) for i in items))
        finally:
            if session is not None:
                await session.close()
        return results, errors

    def generate_items(self, items: Sequence[ChatItem], gen_cfg: Mapping[str, Any]
                       ) -> Tuple[Dict[str, GenResult], List[ItemError]]:
        return asyncio.run(self.agenerate_items(items, gen_cfg))

    # -- the Engine seam ------------------------------------------------------- #

    def generate(self, prompts: Sequence[ChatItem], arm: Any, gen_cfg: Mapping[str, Any]
                 ) -> List[GenOut]:
        """``Engine.generate`` (serving/__init__.py) for CHAT items. ``arm``
        must be None: the steer lives on the served route (the object the
        proxy's routing table binds), never in an in-process handle — an
        arm here would claim a second intervention this engine cannot
        apply (raise loudly for any arm).
        All-or-nothing: any failed item raises with the collected errors."""
        if arm is not None:
            raise ProxyEngineError(
                "ProxyGenerateEngine.generate takes arm=None — the intervention is the "
                f"served route {self.route.cell_id!r}; an in-process arm cannot be applied here")
        items = list(prompts)
        for p in items:
            if not isinstance(p, ChatItem):
                raise ProxyEngineError(
                    f"ProxyGenerateEngine.generate takes ChatItem objects (the proxy renders "
                    f"the template server-side), got {type(p).__name__}")
        results, errors = self.generate_items(items, gen_cfg)
        if errors:
            raise ProxyEngineError(
                f"{len(errors)} of {len(items)} served generations FAILED "
                f"(first: {errors[:3]}) — no partial output is returned")
        return [GenOut(text=results[generation_key(i.item_id, i.sample_idx)].text,
                       gen_token_logprobs=[],
                       ended_with_eos=(results[generation_key(i.item_id, i.sample_idx)].finish_reason
                                       == "stop"))
                for i in items]
