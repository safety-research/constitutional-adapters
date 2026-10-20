"""Usage ledger — a loopback proxy in FRONT of the RoutingProxy that measures
EVERY attempted ``/v1/chat/completions`` call, served or rejected.

WHY. The served geometry of a task class is sized to the p99 of prompt +
completion over the calls a run actually makes (``geometry.sizing_problem``).
vLLM's ``usage`` counters cover only the calls it SERVED: a request whose
prompt + ``max_tokens`` exceeds ``max_model_len`` is 400-rejected with no
usage at all, and those are exactly the calls a window must be sized for.
A basis built from ``usage`` alone excludes the over-window tail by
construction. So every tinycap client — our aiohttp posters, the
``tinycap`` Inspect provider, the EvalPlus engine — points at THIS proxy
instead of the RoutingProxy, and for every chat body it

1. builds the ``/v1/tokenize`` body from the EXACT outbound body (``model``
   + the RoutingProxy's :data:`TOKENIZE_FORWARD_KEYS` — ``messages``,
   ``tools``, ``chat_template_kwargs``, ``continue_final_message``,
   ``add_generation_prompt`` (defaulting as vLLM does: true unless
   ``continue_final_message``), ``add_special_tokens``) and POSTs it to the
   RoutingProxy BEFORE forwarding, recording ``prompt_tokens_pre``; a
   tokenize failure BLOCKS the forward (an unmeasured call is not sent) and
   is itself a ledger row. The RoutingProxy's tokenize applies the chat
   endpoint's ``chat_template_kwargs`` render gate, so a body the chat
   endpoint would refuse by name (``thinking_regime_mismatch`` on a
   thinking-ON standup, ``thinking_kwargs_invalid`` / ``_inert``) is refused
   THERE with the same typed 400 — that refusal is the served answer to the
   client's request, not a ledger fault, and is RELAYED as such
   (:data:`RELAYED_TOKENIZE_REFUSAL_CODES`: same status and body, plus the
   ledger's ``via: ledger_tokenize`` marker; re-labelling it a 502 would
   hide the engine's typed answer from every thinking-ON GSM8K probe). The
   forward is still blocked and the row is the blocked row; every OTHER
   tokenize failure (transport, a 5xx, an untyped 4xx, a malformed body) is
   the ledger's own ``502 ledger_tokenize_failed``;
2. forwards the ORIGINAL request bytes untouched — the two
   :data:`TINYCAP_HEADERS` (``X-Tinycap-Task``, ``X-Tinycap-Sample``) are
   consumed here and never forwarded; ``Authorization`` and the rest travel;
   when a client sends no ``X-Tinycap-Sample``, the row's ``sample`` falls back
   to its :data:`HEADER_SESSION_ID` (``X-Session-ID`` — the per-session id
   vLLM-router-style clients tag every request of one session with), which is
   read but NOT consumed: it travels upstream with the rest;
3. appends ONE flat JSON row (:data:`LEDGER_ROW_FIELDS`, fsync per row) to
   ``usage_ledger.jsonl`` regardless of outcome — ``rejected`` is true on any
   non-2xx (or a transport failure), ``usage_*`` / ``finish_reason`` are null
   then, ``tokenize_mismatch`` flags a served call whose
   ``usage.prompt_tokens`` disagrees with the pre-tokenized count, and
   ``error_class`` is the ledger's NORMALIZED class of the outcome
   (:func:`error_class_of`: ``none`` / ``context_length`` — the
   pinned vLLM forms via ``served_window.parse_context_overflow`` — /
   ``bad_request_other`` / ``auth`` / ``route_refused`` / ``rate_limited`` /
   ``server_error`` / ``transport`` / ``ledger_blocked``), so the geometry
   measurer never has to infer one from a raw 400.

Because every BFCL turn is its own HTTP request, dynamic agent turns are
tokenized just before submission for free. ``GET /v1/models`` and the
clients' own ``/v1/tokenize`` calls pass through un-ledgered (they generate
nothing). The ledger REFUSES TO START unless the RoutingProxy's
``/v1/tokenize`` answers (:meth:`UsageLedger.probe`): a ledger that cannot
measure must not stand in front of anything.

Deployment: one process per container, in the CLIENT venv, on loopback
(``python -m sisteer.capeval.serving.usage_ledger --upstream <proxy /v1>
--port N --ledger-path <cell out_root>/usage_ledger.jsonl``); the launcher
passes its URL as the driver's ``--ledger-url`` (P36). aiohttp is imported
lazily so the row/body helpers stay importable in the CPU test venv; the
transport is injectable so the ordering / byte-identity contract is tested
without sockets.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from sisteer.capeval.serving.proxy import (
    THINKING_KWARGS_INERT,
    THINKING_KWARGS_INVALID,
    THINKING_REGIME_MISMATCH,
    TOKENIZE_FORWARD_KEYS,
    _canonical_v1_base,
)
from sisteer.capeval.serving.served_window import (
    exact_prompt_tokens,
    parse_context_overflow,
)
from sisteer.capeval.tinycap.completion import atomic_write_json
from sisteer.capeval.tinycap.contracts import (
    CHAT_TEMPLATE_KWARGS_KEY,
    ENABLE_THINKING_KEY,
    HEADER_REQUEST_ID,
    HEADER_SAMPLE,
    HEADER_TASK,
    LEDGER_CONSUMED_HEADERS,
    LEDGER_ERROR_CLASS_AUTH,
    LEDGER_ERROR_CLASS_BAD_REQUEST_OTHER,
    LEDGER_ERROR_CLASS_CONTEXT_LENGTH,
    LEDGER_ERROR_CLASS_LEDGER_BLOCKED,
    LEDGER_ERROR_CLASS_NONE,
    LEDGER_ERROR_CLASS_RATE_LIMITED,
    LEDGER_ERROR_CLASS_ROUTE_REFUSED,
    LEDGER_ERROR_CLASS_SERVER_ERROR,
    PROXY_UPSTREAM_TIMEOUT_S,
    LEDGER_ERROR_CLASS_TRANSPORT,
    LEDGER_FILENAME,
    LEDGER_ROW_FIELDS,
    wire_body_sha256,
    LEDGER_ATTEMPT_ID_FIELD,
    check_attempt_id_arg,
)

log = logging.getLogger("tinycap.usage_ledger")

__all__ = [
    "CHAT_PATH", "TOKENIZE_PATH", "MODELS_PATH", "LEDGER_FILENAME",
    "ERROR_CODE_TOKENIZE_FAILED", "ERROR_CODE_TRANSPORT", "ERROR_CODE_BAD_JSON",
    "RELAYED_TOKENIZE_REFUSAL_CODES", "RELAY_VIA_KEY", "RELAY_VIA_LEDGER_TOKENIZE",
    "relayed_tokenize_refusal",
    "HOP_BY_HOP_HEADERS", "HEADER_SESSION_ID", "PROBE_MESSAGES", "DEFAULT_TIMEOUT_S",
    "LedgerError", "LedgerStartupRefused",
    "tokenize_body_for", "body_facts", "ledger_row", "forward_headers", "error_class_of",
    "LedgerWriter", "Transport", "UsageLedger", "aiohttp_transport",
    "rejected_rows_for_task",
    "make_app", "build_parser", "main",
]

CHAT_PATH = "/v1/chat/completions"
TOKENIZE_PATH = "/v1/tokenize"
MODELS_PATH = "/v1/models"

#: ``error_code`` of a row whose forward was BLOCKED because the pre-tokenize
#: failed (no chat request was sent; ``status`` is null).
ERROR_CODE_TOKENIZE_FAILED = "ledger_tokenize_failed"
#: ``error_code`` of a row whose forward died in transport (connection
#: refused / reset / timeout — no HTTP status came back).
ERROR_CODE_TRANSPORT = "ledger_transport_error"
#: ``error_code`` of a row whose body was not a JSON object (nothing to
#: tokenize, nothing forwarded).
ERROR_CODE_BAD_JSON = "ledger_bad_json"

#: The typed 400 codes the RoutingProxy's ``/v1/tokenize`` RENDER GATE emits
#: (``RoutingProxy.handle_tokenize``: ``_validate_kwargs_shape`` ->
#: ``thinking_kwargs_invalid`` / ``thinking_kwargs_inert``, then the thinking-ON
#: regime check -> ``thinking_regime_mismatch``) — each is the SAME refusal, same
#: code, the chat endpoint gives that body. When the pre-tokenize answers one of
#: these, the ledger RELAYS it to the client as the engine's answer (same status,
#: same body, :data:`RELAY_VIA_KEY` added inside ``error``) instead of
#: re-labelling it :data:`ERROR_CODE_TOKENIZE_FAILED`; the forward stays blocked and
#: the row stays the blocked row. Exactly these three — a chat-only declaration code
#: (``thinking_on_*``) never comes off a tokenize, an untyped 400 says nothing about
#: the render, and anything else is a ledger-side failure the client must see as one.
RELAYED_TOKENIZE_REFUSAL_CODES: Tuple[str, ...] = (
    THINKING_KWARGS_INVALID, THINKING_KWARGS_INERT, THINKING_REGIME_MISMATCH)
#: the marker the ledger adds INSIDE the relayed ``error`` object: where the client's
#: refusal was answered (the pre-tokenize, so nothing was forwarded), a fact beside the
#: engine's own ``code`` / ``type`` / ``message``, which travel untouched
RELAY_VIA_KEY = "via"
RELAY_VIA_LEDGER_TOKENIZE = "ledger_tokenize"

#: Never forwarded: the connection-scoped headers (RFC 7230 §6.1) plus the
#: ones the forwarding client recomputes for the new body.
HOP_BY_HOP_HEADERS = frozenset(h.lower() for h in (
    "host", "content-length", "transfer-encoding", "connection", "keep-alive",
    "proxy-connection", "te", "trailer", "upgrade", "accept-encoding"))

#: Everything :func:`forward_headers` strips: the tinycap headers (consumed
#: here) plus the hop-by-hop set.
_DROP_HEADERS = frozenset(h.lower() for h in LEDGER_CONSUMED_HEADERS) | HOP_BY_HOP_HEADERS

#: The per-session header vLLM-router-style clients send with every request
#: of one session. The ledger's ``sample`` column FALLS BACK to it when
#: ``X-Tinycap-Sample`` is absent (the tinycap headers keep precedence); it is
#: never consumed — it is not in :data:`_DROP_HEADERS`, so it is forwarded
#: upstream like any other header.
HEADER_SESSION_ID = "X-Session-ID"

#: The startup probe's tokenize body (a synthetic one-line user turn).
PROBE_MESSAGES = [{"role": "user", "content": "ledger probe"}]

#: The upstream total timeout — the served driver's constant (a queued request
#: at high server load can sit far longer than its own decode), ONE number with
#: the Inspect provider's and the lens proxy's upstream timeout
#: (``contracts.PROXY_UPSTREAM_TIMEOUT_S``).
DEFAULT_TIMEOUT_S = PROXY_UPSTREAM_TIMEOUT_S

#: ``(status, body_bytes, content_type)`` of an upstream call.
Response = Tuple[int, bytes, str]
#: ``await transport(method, path, body_bytes | None, headers) -> Response``.
Transport = Callable[[str, str, Optional[bytes], Mapping[str, str]], Awaitable[Response]]


class LedgerError(RuntimeError):
    """A ledger wiring / body error (never a silent skip)."""


class LedgerStartupRefused(LedgerError):
    """The RoutingProxy's ``/v1/tokenize`` did not answer the startup probe:
    the ledger cannot measure, so it refuses to stand up."""


# --------------------------------------------------------------------------- #
# pure helpers (tested without sockets)                                        #
# --------------------------------------------------------------------------- #


def tokenize_body_for(chat_body: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``/v1/tokenize`` body of an outbound chat body: ``model`` plus
    every :data:`TOKENIZE_FORWARD_KEYS` key PRESENT in the chat body, with
    ``add_generation_prompt`` defaulting exactly as vLLM renders the chat
    request — true unless ``continue_final_message`` is truthy. Nothing
    else travels (sampling knobs, ``max_tokens``, the proxy declarations):
    the render is a function of these keys alone, so the count is the
    count of the request that follows."""
    if not isinstance(chat_body, Mapping):
        raise LedgerError("chat body must be a JSON object")
    model = chat_body.get("model")
    if not isinstance(model, str) or not model:
        raise LedgerError("chat body has no 'model' — nothing to tokenize under")
    out: Dict[str, Any] = {"model": model}
    for key in TOKENIZE_FORWARD_KEYS:
        if key in chat_body:
            out[key] = chat_body[key]
    if "add_generation_prompt" not in out:
        out["add_generation_prompt"] = not bool(chat_body.get("continue_final_message", False))
    return out


def body_facts(chat_body: Mapping[str, Any]) -> Dict[str, Any]:
    """The ledger columns read from the body (P27): ``vmodel``, ``thinking``
    (``chat_template_kwargs.enable_thinking`` as a bool, or null when the
    body carries no switch), ``max_tokens_requested`` (``max_tokens``, else
    ``max_completion_tokens``, else null), ``has_tools`` / ``n_tools``."""
    ctk = chat_body.get(CHAT_TEMPLATE_KWARGS_KEY)
    thinking: Optional[bool] = None
    if isinstance(ctk, Mapping) and ENABLE_THINKING_KEY in ctk:
        thinking = bool(ctk[ENABLE_THINKING_KEY])
    max_tokens = chat_body.get("max_tokens")
    if max_tokens is None:
        max_tokens = chat_body.get("max_completion_tokens")
    if max_tokens is not None:
        try:
            max_tokens = int(max_tokens)
        except (TypeError, ValueError):
            max_tokens = None
    tools = chat_body.get("tools")
    n_tools = len(tools) if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes)) else 0
    return {
        "vmodel": chat_body.get("model") if isinstance(chat_body.get("model"), str) else None,
        "thinking": thinking,
        "max_tokens_requested": max_tokens,
        "has_tools": n_tools > 0,
        "n_tools": n_tools,
    }


#: The ledger's OPTIONAL appended ``completion_id`` column — the engine's EXTERNAL
#: completion ``id`` of a served 2xx row (the ``Received request <id>-<8 hex>`` line of
#: the engine log is that id plus one suffix, ``serving_attestation.is_engine_id_of``),
#: null on a rejected / blocked row — present on EVERY row iff the ledger runs under
#: ``--record-completion-id`` (``UsageLedger(record_completion_id=True)``).
#: :data:`LEDGER_ROW_FIELDS` is unchanged; a ledger without the flag writes the old bytes.
LEDGER_COMPLETION_ID_FIELD = "completion_id"


def ledger_row(**fields: Any) -> Dict[str, Any]:
    """One flat row in :data:`LEDGER_ROW_FIELDS` order; refuses a missing or
    an unknown column (the schema is the contract, never a superset). The
    optional columns are :data:`LEDGER_ATTEMPT_ID_FIELD` (a ledger run under
    ``--attempt-id``) and :data:`LEDGER_COMPLETION_ID_FIELD` (``--record-completion-id``),
    appended after the schema in that order when present; a row without them
    is byte-identical to the pre-attempt form."""
    allowed = (*LEDGER_ROW_FIELDS, LEDGER_ATTEMPT_ID_FIELD, LEDGER_COMPLETION_ID_FIELD)
    missing = [k for k in LEDGER_ROW_FIELDS if k not in fields]
    unknown = sorted(set(fields) - set(allowed))
    if missing or unknown:
        raise LedgerError(f"ledger row: missing {missing}, unknown {unknown}")
    return {k: fields[k] for k in allowed if k in fields}


def error_class_of(status: Optional[int], body: Any) -> str:
    """The ``error_class`` column (``contracts.LEDGER_ERROR_CLASSES``) of an
    upstream reply, derived HERE from its status + body (E12 item 5) so no
    reader ever infers a class from the raw status: ``none`` for a 2xx;
    for a 400, ``context_length`` iff the body is one of the pinned vLLM
    context-length forms (``served_window.parse_context_overflow`` — the
    only recogniser), else ``bad_request_other`` (a tool schema, an
    unsupported field, a malformed prompt — a 400 that says nothing about
    the window); ``auth`` 401/403; ``route_refused`` 404/409/421 (the
    RoutingProxy refusing the request as addressed); ``rate_limited`` 429;
    ``server_error`` 5xx; any other 4xx ``bad_request_other``; no status
    (the forward died) ``transport``. The status decides FIRST: a 5xx whose
    body happens to quote the context text is a server error."""
    if status is None:
        return LEDGER_ERROR_CLASS_TRANSPORT
    status = int(status)
    if 200 <= status < 300:
        return LEDGER_ERROR_CLASS_NONE
    if status == 400:
        text = (body.decode("utf-8", "replace") if isinstance(body, (bytes, bytearray))
                else ("" if body is None else str(body)))
        if parse_context_overflow(text, http_status=400) is not None:
            return LEDGER_ERROR_CLASS_CONTEXT_LENGTH
        return LEDGER_ERROR_CLASS_BAD_REQUEST_OTHER
    if status in (401, 403):
        return LEDGER_ERROR_CLASS_AUTH
    if status in (404, 409, 421):
        return LEDGER_ERROR_CLASS_ROUTE_REFUSED
    if status == 429:
        return LEDGER_ERROR_CLASS_RATE_LIMITED
    if status >= 500:
        return LEDGER_ERROR_CLASS_SERVER_ERROR
    return LEDGER_ERROR_CLASS_BAD_REQUEST_OTHER


def forward_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    """The headers that travel to the RoutingProxy: everything except the
    :data:`TINYCAP_HEADERS` (consumed here) and the hop-by-hop set."""
    return {k: v for k, v in headers.items() if k.lower() not in _DROP_HEADERS}


def _header(headers: Mapping[str, str], name: str) -> Optional[str]:
    lname = name.lower()
    for k, v in headers.items():
        if k.lower() == lname:
            return v
    return None


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def _error_body(message: str, code: str) -> bytes:
    return json.dumps({"error": {"message": message, "type": "ledger_error",
                                 "code": code}}).encode("utf-8")


def _json_or_none(raw: bytes) -> Optional[Any]:
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def relayed_tokenize_refusal(status: Optional[int], raw: Any) -> Optional[bytes]:
    """The body to RELAY when the RoutingProxy's ``/v1/tokenize`` answered the
    outbound body with a typed render refusal (a 400 whose ``error.code`` is
    one of :data:`RELAYED_TOKENIZE_REFUSAL_CODES`): the proxy's error object
    verbatim — ``message`` / ``type`` / ``code`` and any ``extra`` facts — plus
    ``via: ledger_tokenize``. ``None`` for anything else (not a 400, an
    untyped or foreign code, a non-JSON or non-object body, no ``error``
    object): those take the ledger's own 502. Reads the WIRE SHAPE only
    (``proxy._error``), never the prose."""
    if status is None or int(status) != 400 or not isinstance(raw, (bytes, bytearray)):
        return None
    payload = _json_or_none(bytes(raw))
    if not isinstance(payload, Mapping):
        return None
    err = payload.get("error")
    if not isinstance(err, Mapping):
        return None
    code = err.get("code")
    if not isinstance(code, str) or code not in RELAYED_TOKENIZE_REFUSAL_CODES:
        return None
    out = dict(payload)
    out["error"] = {**dict(err), RELAY_VIA_KEY: RELAY_VIA_LEDGER_TOKENIZE}
    return json.dumps(out).encode("utf-8")


# --------------------------------------------------------------------------- #
# the writer                                                                   #
# --------------------------------------------------------------------------- #


def read_rows(path: Path) -> List[Dict[str, Any]]:
    """Every parsed row of a JSONL file this module's shape (the usage ledger,
    the driver's request-evidence file): blank lines and a torn trailing line
    (an interrupted write) are skipped, as every reader of these files does;
    a non-object line is skipped too. The file must exist."""
    out: List[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def rejected_rows_for_task(path: Path, task: str) -> int:
    """The number of ledger rows naming ``task`` with ``rejected`` true — the
    ledger's own reading of how many of the task's attempted calls the
    upstream refused (the Inspect tier files it as the summary's
    ``n_http_rejected_retried``, the analysis re-reads the file the same way).
    Every row is on disk before its response is returned (``handle_chat``
    writes + fsyncs first), so at task end the count is complete
    (:func:`read_rows` skips a torn trailing line)."""
    return sum(1 for row in read_rows(path) if row.get("task") == task and row.get("rejected") is True)


class LedgerWriter:
    """Append-only JSONL, one ``write`` + ``flush`` + ``fsync`` per row: a
    container that dies mid-run leaves every row it ledgered on disk."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self.n_rows = 0
        self.n_rejected = 0

    def write(self, row: Mapping[str, Any]) -> None:
        row = ledger_row(**row)
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self.n_rows += 1
        if row["rejected"]:
            self.n_rejected += 1

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# the ledger                                                                   #
# --------------------------------------------------------------------------- #


class UsageLedger:
    """The measuring proxy core: tokenize-before-forward on every chat body,
    byte-identical forwarding, one row per attempt. ``transport`` is the
    upstream HTTP call (:func:`aiohttp_transport` in production; a fake in
    tests). ``probe_bearer`` is the ``Authorization`` bearer the STARTUP
    probe sends (the RoutingProxy authenticates ``/v1/models`` and
    ``/v1/tokenize`` when it carries a token; clients send their own)."""

    def __init__(self, upstream_base_url: str, ledger_path: Path, *,
                 transport: Optional[Transport] = None,
                 probe_bearer: Optional[str] = None,
                 timeout_s: float = DEFAULT_TIMEOUT_S,
                 attempt_id: Optional[str] = None,
                 record_completion_id: bool = False):
        self.upstream_base_url = _canonical_v1_base(upstream_base_url)
        #: the OPTIONAL appended completion_id column (LEDGER_COMPLETION_ID_FIELD): off by default, so
        #: a ledger without the flag writes byte-identical rows
        self.record_completion_id = bool(record_completion_id)
        # the OPTIONAL appended column: set only when the ledger runs under --attempt-id; the row schema
        # (LEDGER_ROW_FIELDS) is deliberately unchanged, so a ledger without it writes the same bytes
        self.attempt_id = check_attempt_id_arg(attempt_id) if attempt_id is not None else None
        self.writer = LedgerWriter(Path(ledger_path))
        self._transport: Transport = transport or aiohttp_transport(self.upstream_base_url,
                                                                    timeout_s=timeout_s)
        self._probe_bearer = probe_bearer
        self.n_forwarded = 0
        self.n_blocked = 0
        self.n_mismatch = 0

    async def aclose(self) -> None:
        """Release the transport's connections (:func:`aiohttp_transport`
        keeps ONE session open for keep-alive); a no-op for a transport
        without ``aclose`` (the tests' fakes)."""
        aclose = getattr(self._transport, "aclose", None)
        if aclose is not None:
            await aclose()

    # -- startup -------------------------------------------------------------

    async def probe(self) -> Dict[str, Any]:
        """Refuse to start unless the RoutingProxy answers ``/v1/models`` AND
        tokenizes a synthetic body under the first served virtual model."""
        headers = {"Content-Type": "application/json"}
        if self._probe_bearer:
            headers["Authorization"] = f"Bearer {self._probe_bearer}"
        try:
            status, body, _ = await self._transport("GET", MODELS_PATH, None, headers)
        except Exception as e:
            raise LedgerStartupRefused(
                f"the RoutingProxy at {self.upstream_base_url} did not answer "
                f"{MODELS_PATH}: {type(e).__name__}: {e}") from e
        payload = _json_or_none(body)
        if status != 200 or not isinstance(payload, Mapping):
            raise LedgerStartupRefused(
                f"{MODELS_PATH} on {self.upstream_base_url} returned {status}: {body[:300]!r}")
        models = [m.get("id") for m in payload.get("data", []) if isinstance(m, Mapping)]
        models = [m for m in models if isinstance(m, str) and m]
        if not models:
            raise LedgerStartupRefused(
                f"{MODELS_PATH} lists no served virtual model — nothing to tokenize under")
        tok_body = {"model": models[0], "messages": PROBE_MESSAGES,
                    "add_generation_prompt": True}
        try:
            status, body, _ = await self._transport(
                "POST", TOKENIZE_PATH, json.dumps(tok_body).encode("utf-8"), headers)
        except Exception as e:
            raise LedgerStartupRefused(
                f"{TOKENIZE_PATH} on {self.upstream_base_url} did not answer: "
                f"{type(e).__name__}: {e}") from e
        count = _count_of(_json_or_none(body))
        if status != 200 or count is None:
            raise LedgerStartupRefused(
                f"{TOKENIZE_PATH} on {self.upstream_base_url} returned {status} / no integer "
                f"'count': {body[:300]!r} — the ledger cannot measure, so it will not stand "
                "in front of the proxy")
        return {"upstream": self.upstream_base_url, "virtual_models": models,
                "probe_prompt_tokens": count}

    # -- the measured endpoint -------------------------------------------------

    async def handle_chat(self, raw_body: bytes, headers: Mapping[str, str]) -> Response:
        """Tokenize the exact outbound body, forward the ORIGINAL bytes, ledger
        one row whatever happens; return the upstream response verbatim (or
        the ledger's own 502 when the forward was blocked / died)."""
        t0 = time.monotonic()
        task = _header(headers, HEADER_TASK)
        sample = _header(headers, HEADER_SAMPLE)
        if sample is None:
            # additive: a client that sets no X-Tinycap-Sample but tags every
            # request of a session with its session id; the tinycap header
            # wins whenever it is present
            sample = _header(headers, HEADER_SESSION_ID)
        fwd = forward_headers(headers)
        fwd.setdefault("Content-Type", "application/json")
        # every column null, then the non-null defaults of a row that never
        # reached the server (the schema is the contract; ledger_row re-checks).
        # The per-attempt identity goes on EVERY row — the client's
        # X-Tinycap-Request-Id verbatim (null when it sent none) and the sha256
        # of the EXACT bytes received (= forwarded), blocked and bad-JSON rows
        # included, so the driver's request-evidence record of the same attempt
        # (the same id, the same bytes) reconciles by identity
        base = dict.fromkeys(LEDGER_ROW_FIELDS)
        if self.attempt_id is not None:
            base[LEDGER_ATTEMPT_ID_FIELD] = self.attempt_id
        if self.record_completion_id:
            base[LEDGER_COMPLETION_ID_FIELD] = None   # filled from a served 2xx body below; null otherwise
        base.update(ts=_now_iso(), task=task, sample=sample, has_tools=False, n_tools=0,
                    rejected=True, request_id=_header(headers, HEADER_REQUEST_ID),
                    body_sha256=wire_body_sha256(raw_body))

        body = _json_or_none(raw_body)
        if not isinstance(body, Mapping):
            base.update(error_code=ERROR_CODE_BAD_JSON, error_class=LEDGER_ERROR_CLASS_LEDGER_BLOCKED,
                        latency_s=_lat(t0))
            self.writer.write(base)
            self.n_blocked += 1
            return 400, _error_body("request body is not a JSON object", ERROR_CODE_BAD_JSON), \
                "application/json"
        base.update(body_facts(body))

        # 1. tokenize the EXACT outbound body BEFORE forwarding
        relay: Optional[bytes] = None
        try:
            tok_body = tokenize_body_for(body)
            t_status, t_raw, _ = await self._transport(
                "POST", TOKENIZE_PATH, json.dumps(tok_body).encode("utf-8"), fwd)
            count = _count_of(_json_or_none(t_raw)) if t_status == 200 else None
            detail = "" if count is not None else f" ({t_status}: {t_raw[:300]!r})"
            if count is None:
                # the proxy's render gate refused THIS body by name on the
                # tokenize — the same typed 400 its chat endpoint would give; relayed
                # below as the engine's answer (RELAYED_TOKENIZE_REFUSAL_CODES)
                relay = relayed_tokenize_refusal(t_status, t_raw)
        except Exception as e:  # LedgerError (no model) or transport
            count, detail = None, f" ({type(e).__name__}: {e})"
        if count is None:
            # the ledger's accounting is one thing whatever the reply below: a blocked
            # row (status null, prompt_tokens_pre null, ERROR_CODE_TOKENIZE_FAILED,
            # ledger_blocked) — nothing was measured and nothing was forwarded
            base.update(error_code=ERROR_CODE_TOKENIZE_FAILED,
                        error_class=LEDGER_ERROR_CLASS_LEDGER_BLOCKED, latency_s=_lat(t0))
            self.writer.write(base)
            self.n_blocked += 1
            if relay is not None:
                log.warning("ledger: tokenize REFUSED the render by name, forward BLOCKED, "
                            "refusal relayed task=%s sample=%s%s", task, sample, detail)
                return 400, relay, "application/json"
            log.warning("ledger: tokenize failed, forward BLOCKED task=%s sample=%s%s",
                        task, sample, detail)
            return 502, _error_body(
                f"usage ledger: /v1/tokenize of the outbound body failed{detail}; an "
                "unmeasured call is not forwarded", ERROR_CODE_TOKENIZE_FAILED), \
                "application/json"
        base["prompt_tokens_pre"] = count

        # 2. forward the ORIGINAL bytes, untouched
        try:
            status, resp_raw, ctype = await self._transport("POST", CHAT_PATH, raw_body, fwd)
        except Exception as e:
            base.update(error_code=ERROR_CODE_TRANSPORT, error_class=LEDGER_ERROR_CLASS_TRANSPORT,
                        latency_s=_lat(t0))
            self.writer.write(base)
            self.n_blocked += 1
            log.warning("ledger: forward died task=%s sample=%s: %s: %s", task, sample,
                        type(e).__name__, e)
            return 502, _error_body(
                f"usage ledger: forwarding to {self.upstream_base_url}{CHAT_PATH} failed: "
                f"{type(e).__name__}: {e}", ERROR_CODE_TRANSPORT), "application/json"
        self.n_forwarded += 1

        # 3. the row
        latency = _lat(t0)
        rejected = not (200 <= int(status) < 300)
        resp = _json_or_none(resp_raw)
        usage = resp.get("usage") if isinstance(resp, Mapping) else None
        u_prompt = _int_or_none(usage.get("prompt_tokens")) if isinstance(usage, Mapping) else None
        u_comp = _int_or_none(usage.get("completion_tokens")) if isinstance(usage, Mapping) else None
        finish = None
        if isinstance(resp, Mapping) and isinstance(resp.get("choices"), list) and resp["choices"]:
            c0 = resp["choices"][0]
            if isinstance(c0, Mapping):
                finish = c0.get("finish_reason")
        error_code = None
        if rejected:
            err = resp.get("error") if isinstance(resp, Mapping) else None
            code = err.get("code") if isinstance(err, Mapping) else None
            error_code = str(code) if code is not None else f"http_{status}"
        mismatch: Optional[bool] = None
        if not rejected and u_prompt is not None:
            mismatch = (u_prompt != count)
            if mismatch:
                self.n_mismatch += 1
        if self.record_completion_id and not rejected and isinstance(resp, Mapping):
            cid = resp.get("id")
            base[LEDGER_COMPLETION_ID_FIELD] = cid if isinstance(cid, str) and cid else None
        base.update(status=int(status), error_code=error_code, rejected=rejected,
                    # E12 item 5: the class is derived here, from status + body
                    error_class=error_class_of(int(status), resp_raw),
                    usage_prompt_tokens=u_prompt if not rejected else None,
                    usage_completion_tokens=u_comp if not rejected else None,
                    finish_reason=finish if not rejected else None,
                    latency_s=latency, tokenize_mismatch=mismatch)
        self.writer.write(base)
        return int(status), resp_raw, ctype or "application/json"

    # -- un-ledgered pass-throughs ---------------------------------------------

    async def passthrough(self, method: str, path: str, raw_body: Optional[bytes],
                          headers: Mapping[str, str]) -> Response:
        """``GET /v1/models`` and the clients' own ``/v1/tokenize``: forwarded
        with the tinycap headers stripped, never ledgered (nothing generates)."""
        fwd = forward_headers(headers)
        if raw_body is not None:
            fwd.setdefault("Content-Type", "application/json")
        try:
            return await self._transport(method, path, raw_body, fwd)
        except Exception as e:
            return 502, _error_body(
                f"usage ledger: forwarding {method} {path} failed: {type(e).__name__}: {e}",
                ERROR_CODE_TRANSPORT), "application/json"

    def stats(self) -> Dict[str, Any]:
        return {"ledger_path": str(self.writer.path), "n_rows": self.writer.n_rows,
                "n_rejected": self.writer.n_rejected, "n_forwarded": self.n_forwarded,
                "n_blocked": self.n_blocked, "n_tokenize_mismatch": self.n_mismatch}


def _lat(t0: float) -> float:
    return round(time.monotonic() - t0, 4)


def _int_or_none(v: Any) -> Optional[int]:
    if isinstance(v, bool) or v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _count_of(payload: Any) -> Optional[int]:
    """``served_window.exact_prompt_tokens``'s rule (an integer, non-bool,
    non-negative ``count``), read as ``None`` instead of a raise: the ledger
    records a bad tokenize reply as a blocked row, never as an exception."""
    try:
        return exact_prompt_tokens(payload)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# transport + HTTP surface (aiohttp, lazy)                                     #
# --------------------------------------------------------------------------- #


def aiohttp_transport(upstream_base_url: str, *, timeout_s: float = DEFAULT_TIMEOUT_S
                      ) -> Transport:
    """The production transport: ONE aiohttp session, opened lazily on the
    running loop and kept for keep-alive to the loopback proxy (a session
    per call opened and tore down a connection twice per ledgered request —
    tokenize + forward — for every request of a run), against
    ``<upstream root>/<path>`` (``upstream_base_url`` is the ``/v1`` base;
    every path here already carries its ``/v1`` prefix). ``call.aclose()``
    closes the session (:meth:`UsageLedger.aclose`, at shutdown)."""
    root = _canonical_v1_base(upstream_base_url)[:-len("/v1")]
    state: Dict[str, Any] = {"session": None}

    async def session():
        import aiohttp  # lazy: serving-time dep
        s = state["session"]
        if s is None or s.closed:
            s = state["session"] = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout_s))
        return s

    async def call(method: str, path: str, body: Optional[bytes],
                   headers: Mapping[str, str]) -> Response:
        async with (await session()).request(method, f"{root}{path}", data=body,
                                             headers=dict(headers)) as resp:
            raw = await resp.read()
            return resp.status, raw, resp.headers.get("Content-Type", "application/json")

    async def aclose() -> None:
        s = state["session"]
        if s is not None and not s.closed:
            await s.close()

    call.aclose = aclose  # type: ignore[attr-defined]
    return call


def make_app(ledger: UsageLedger):
    """The aiohttp application: the measured chat endpoint plus the
    un-ledgered pass-throughs."""
    from aiohttp import web

    def _reply(r: Response):
        status, raw, ctype = r
        ctype = (ctype or "application/json").split(";")[0].strip()
        return web.Response(status=status, body=raw, content_type=ctype)

    async def chat(request):
        raw = await request.read()
        return _reply(await ledger.handle_chat(raw, dict(request.headers)))

    async def models(request):
        return _reply(await ledger.passthrough("GET", MODELS_PATH, None, dict(request.headers)))

    async def tokenize(request):
        raw = await request.read()
        return _reply(await ledger.passthrough("POST", TOKENIZE_PATH, raw, dict(request.headers)))

    async def stats(request):
        return web.json_response(ledger.stats())

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_post(CHAT_PATH, chat)
    app.router.add_get(MODELS_PATH, models)
    app.router.add_post(TOKENIZE_PATH, tokenize)
    app.router.add_post("/tokenize", tokenize)
    app.router.add_get("/ledger/stats", stats)
    return app


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--upstream", required=True,
                    help="the RoutingProxy base URL (its /v1), e.g. http://127.0.0.1:PORT/v1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1",
                    help="loopback only — the ledger is container-internal")
    ap.add_argument("--ledger-path", required=True,
                    help=f"the append-only {LEDGER_FILENAME} (the cell's out_root)")
    ap.add_argument("--endpoint-json", default=None,
                    help="written ATOMICALLY once the probe passed and the site is up: "
                         "{ledger_url, upstream, ledger_path, pid, probe} — a launcher polls it")
    ap.add_argument("--probe-bearer-env", default=None,
                    help="env var holding the bearer the startup probe sends (only when the "
                         "RoutingProxy authenticates; clients send their own Authorization)")
    ap.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--attempt-id", type=check_attempt_id_arg, default=None,
                    help="OPTIONAL attempt id (a<k>) stamped on every row as the appended "
                         f"'{LEDGER_ATTEMPT_ID_FIELD}' column — the attempt-scoped producer contract; "
                         "absent = rows byte-identical to before (the schema is unchanged)")
    ap.add_argument("--record-completion-id", action="store_true",
                    help="OPTIONAL (qwen-seedrep-native-battery): append the served response's completion "
                         f"'id' as the '{LEDGER_COMPLETION_ID_FIELD}' column on every row (null when rejected / "
                         "blocked) so a ledger attempt binds to the engine log by identity; absent = the old bytes")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.host in ("0.0.0.0", "::"):
        raise SystemExit("refusing to bind a wildcard interface — the ledger is loopback-only")
    logging.basicConfig(level=logging.INFO)
    bearer = os.environ.get(args.probe_bearer_env) if args.probe_bearer_env else None
    ledger = UsageLedger(args.upstream, Path(args.ledger_path), probe_bearer=bearer,
                         timeout_s=args.timeout_s, attempt_id=args.attempt_id,
                         record_completion_id=bool(args.record_completion_id))
    from aiohttp import web

    async def serve():
        probe = await ledger.probe()          # refuses to start otherwise
        runner = web.AppRunner(make_app(ledger))
        await runner.setup()
        await web.TCPSite(runner, args.host, args.port).start()
        url = f"http://{args.host}:{args.port}/v1"
        print(f"[ledger] up at {url} -> {ledger.upstream_base_url}; rows -> "
              f"{ledger.writer.path}; probe {probe}", flush=True)
        if args.endpoint_json:
            # atomic (tmp + fsync + replace): a poller never sees a torso
            atomic_write_json(Path(args.endpoint_json),
                              {"ledger_url": url, "upstream": ledger.upstream_base_url,
                               "ledger_path": str(ledger.writer.path),
                               "pid": os.getpid(), "probe": probe})
        try:
            while True:
                await asyncio.sleep(3600)
        finally:
            await runner.cleanup()
            await ledger.aclose()
            ledger.writer.close()

    try:
        asyncio.run(serve())
    except LedgerStartupRefused as e:
        print(f"[ledger] REFUSED to start: {e}", flush=True)
        return 2
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
