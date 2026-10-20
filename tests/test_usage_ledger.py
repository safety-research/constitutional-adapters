"""The usage ledger (``sisteer.capeval.serving.usage_ledger``): the loopback
proxy in FRONT of the RoutingProxy that pre-tokenizes every outbound chat
body and ledgers one flat row per attempted call.

Covered, without sockets (an injected fake transport records the upstream
call order and bodies):
- tokenize-BEFORE-forward ordering; the tokenize body is built from the
  exact outbound body (TOKENIZE_FORWARD_KEYS + model, add_generation_prompt
  defaulting as vLLM does);
- byte-identical forwarding of the ORIGINAL request bytes; the two
  X-Tinycap-* headers are consumed, Authorization travels;
- a 400 upstream still yields a row: rejected=true, prompt_tokens_pre set,
  usage null, error_code from the upstream body;
- a tokenize failure BLOCKS the forward (no chat call) and is itself a row;
- a transport failure is a row (rejected, ledger_transport_error);
- tokenize_mismatch on a served call whose usage.prompt_tokens disagrees;
- every row is exactly LEDGER_ROW_FIELDS, appended + fsync'd, re-readable;
- the startup probe refuses when /v1/models or /v1/tokenize do not answer;
and one in-process aiohttp end-to-end (TestServer) run through make_app with
a fake upstream app. Everything synthetic; no real session data.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from sisteer.capeval.serving import usage_ledger as ul
from sisteer.capeval.serving.proxy import TOKENIZE_FORWARD_KEYS
from sisteer.capeval.tinycap import contracts as C


# --------------------------------------------------------------------------- #
# a scripted fake transport                                                    #
# --------------------------------------------------------------------------- #


class FakeTransport:
    """Records every upstream call; answers from per-path scripts."""

    def __init__(self, *, tokenize_count=17, chat=None, tokenize_status=200,
                 models=None, fail_chat: Exception | None = None):
        self.calls: list[tuple[str, str, bytes | None, dict]] = []
        self.tokenize_count = tokenize_count
        self.tokenize_status = tokenize_status
        self.chat = chat or (200, {"id": "x", "model": "petri__base",
                                   "choices": [{"index": 0, "finish_reason": "stop",
                                                "message": {"role": "assistant", "content": "ok"}}],
                                   "usage": {"prompt_tokens": 17, "completion_tokens": 3,
                                             "total_tokens": 20}})
        self.models = models if models is not None else {
            "object": "list", "data": [{"id": "petri__base", "object": "model"}]}
        self.fail_chat = fail_chat

    async def __call__(self, method, path, body, headers):
        self.calls.append((method, path, body, dict(headers)))
        if path == ul.MODELS_PATH:
            return 200, json.dumps(self.models).encode(), "application/json"
        if path == ul.TOKENIZE_PATH:
            if self.tokenize_status != 200:
                return self.tokenize_status, b'{"error": {"message": "nope", "code": "boom"}}', \
                    "application/json"
            return 200, json.dumps({"count": self.tokenize_count, "max_model_len": 16384,
                                    "tokens": list(range(self.tokenize_count))}).encode(), \
                "application/json"
        if path == ul.CHAT_PATH:
            if self.fail_chat is not None:
                raise self.fail_chat
            status, payload = self.chat
            return status, json.dumps(payload).encode(), "application/json"
        raise AssertionError(f"unexpected upstream path {path}")


def _ledger(tmp_path, transport, **kw) -> ul.UsageLedger:
    return ul.UsageLedger("http://127.0.0.1:1/v1", tmp_path / C.LEDGER_FILENAME,
                          transport=transport, **kw)


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


BODY = {"model": "petri__base",
        "messages": [{"role": "system", "content": "be brief"},
                     {"role": "user", "content": "2+2?"},
                     {"role": "assistant", "content": "4"}],
        "continue_final_message": True, "add_generation_prompt": False,
        "petri_response_prefill": True, "max_tokens": 1, "temperature": 0.0, "seed": 0,
        "prompt_logprobs": 1, "chat_template_kwargs": {"enable_thinking": False}}
HEADERS = {"Authorization": "Bearer tok", "Content-Type": "application/json",
           "X-Tinycap-Task": "inspect/math", "X-Tinycap-Sample": "12/0",
           "Host": "127.0.0.1:9", "Content-Length": "999"}


# --------------------------------------------------------------------------- #
# pure helpers                                                                 #
# --------------------------------------------------------------------------- #


def test_tokenize_body_is_built_from_the_exact_outbound_body():
    tb = ul.tokenize_body_for(BODY)
    assert tb["model"] == "petri__base"
    assert tb["messages"] == BODY["messages"]
    assert tb["continue_final_message"] is True and tb["add_generation_prompt"] is False
    assert tb["chat_template_kwargs"] == {"enable_thinking": False}
    assert set(tb) <= {"model", *TOKENIZE_FORWARD_KEYS}
    for absent in ("max_tokens", "temperature", "seed", "prompt_logprobs",
                   "petri_response_prefill"):
        assert absent not in tb
    # vLLM's default: add_generation_prompt true unless continue_final_message
    assert ul.tokenize_body_for({"model": "m", "messages": [{"role": "user", "content": "x"}]})[
        "add_generation_prompt"] is True
    assert ul.tokenize_body_for({"model": "m", "messages": [], "continue_final_message": True})[
        "add_generation_prompt"] is False
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    assert ul.tokenize_body_for({"model": "m", "messages": [], "tools": tools})["tools"] == tools
    with pytest.raises(ul.LedgerError):
        ul.tokenize_body_for({"messages": []})
    with pytest.raises(ul.LedgerError):
        ul.tokenize_body_for(["not", "a", "mapping"])


def test_body_facts_read_the_columns_from_the_body():
    f = ul.body_facts(BODY)
    assert f == {"vmodel": "petri__base", "thinking": False, "max_tokens_requested": 1,
                 "has_tools": False, "n_tools": 0}
    f = ul.body_facts({"model": "m", "messages": [], "max_completion_tokens": 512,
                       "tools": [{"a": 1}, {"b": 2}],
                       "chat_template_kwargs": {"enable_thinking": True}})
    assert (f["thinking"], f["max_tokens_requested"], f["has_tools"], f["n_tools"]) == (
        True, 512, True, 2)
    f = ul.body_facts({"model": "m", "messages": []})
    assert f["thinking"] is None and f["max_tokens_requested"] is None and f["n_tools"] == 0


def test_ledger_row_is_exactly_the_contract_schema():
    base = {k: None for k in C.LEDGER_ROW_FIELDS}
    row = ul.ledger_row(**base)
    assert tuple(row) == C.LEDGER_ROW_FIELDS
    with pytest.raises(ul.LedgerError, match="missing"):
        ul.ledger_row(**{k: None for k in C.LEDGER_ROW_FIELDS[:-1]})
    with pytest.raises(ul.LedgerError, match="unknown"):
        ul.ledger_row(**base, extra=1)


def test_forward_headers_consume_tinycap_and_hop_by_hop():
    fwd = ul.forward_headers(HEADERS)
    assert fwd == {"Authorization": "Bearer tok", "Content-Type": "application/json"}
    assert ul.forward_headers({"x-tinycap-task": "t", "ACCEPT": "*/*"}) == {"ACCEPT": "*/*"}


# --------------------------------------------------------------------------- #
# the measured endpoint (fake transport)                                       #
# --------------------------------------------------------------------------- #


def test_tokenize_before_forward_and_byte_identical_forwarding(tmp_path):
    ft = FakeTransport(tokenize_count=17)
    led = _ledger(tmp_path, ft)
    # odd whitespace + key order: the forwarded bytes must be THESE bytes
    raw = json.dumps(BODY, indent=3, sort_keys=True).encode("utf-8") + b"\n"
    status, resp, ctype = asyncio.run(led.handle_chat(raw, HEADERS))
    assert status == 200 and json.loads(resp)["choices"][0]["finish_reason"] == "stop"
    assert [c[1] for c in ft.calls] == [ul.TOKENIZE_PATH, ul.CHAT_PATH]  # order
    tok_call, chat_call = ft.calls
    assert json.loads(tok_call[2]) == ul.tokenize_body_for(BODY)
    assert chat_call[2] == raw                                   # byte-identical
    for call in (tok_call, chat_call):
        hdrs = {k.lower(): v for k, v in call[3].items()}
        assert hdrs["authorization"] == "Bearer tok"
        assert not any(k.startswith("x-tinycap-") for k in hdrs)
        assert "host" not in hdrs and "content-length" not in hdrs
    rows = _rows(led.writer.path)
    assert len(rows) == 1
    r = rows[0]
    assert tuple(r) == C.LEDGER_ROW_FIELDS
    assert (r["task"], r["sample"], r["vmodel"], r["thinking"]) == (
        "inspect/math", "12/0", "petri__base", False)
    assert r["prompt_tokens_pre"] == 17 and r["max_tokens_requested"] == 1
    assert r["status"] == 200 and r["rejected"] is False and r["error_code"] is None
    assert (r["usage_prompt_tokens"], r["usage_completion_tokens"]) == (17, 3)
    assert r["finish_reason"] == "stop" and r["tokenize_mismatch"] is False
    assert isinstance(r["latency_s"], float) and r["ts"].endswith("+00:00")
    assert led.stats()["n_forwarded"] == 1 and led.stats()["n_rejected"] == 0


def test_a_400_upstream_still_produces_a_rejected_row_with_the_pre_count(tmp_path):
    err = (400, {"error": {"message": "maximum context length is 16384 tokens ...",
                           "type": "invalid_request_error", "code": "upstream_error"}})
    ft = FakeTransport(tokenize_count=15400, chat=err)
    led = _ledger(tmp_path, ft)
    body = dict(BODY, max_tokens=2048)
    status, resp, _ = asyncio.run(led.handle_chat(json.dumps(body).encode(), HEADERS))
    assert status == 400 and json.loads(resp)["error"]["code"] == "upstream_error"  # passthrough
    assert [c[1] for c in ft.calls] == [ul.TOKENIZE_PATH, ul.CHAT_PATH]
    r = _rows(led.writer.path)[0]
    assert r["rejected"] is True and r["status"] == 400
    assert r["prompt_tokens_pre"] == 15400 and r["max_tokens_requested"] == 2048
    assert r["error_code"] == "upstream_error"
    assert r["error_class"] == C.LEDGER_ERROR_CLASS_CONTEXT_LENGTH == "context_length"
    assert r["usage_prompt_tokens"] is None and r["usage_completion_tokens"] is None
    assert r["finish_reason"] is None and r["tokenize_mismatch"] is None
    assert led.writer.n_rejected == 1


CONTEXT_400 = {"error": {"message": "This model's maximum context length is 16384 tokens. However, "
                                    "you requested 18000 tokens (15952 in the messages, 2048 in the "
                                    "completion). Please reduce the length of the messages or completion.",
                         "type": "BadRequestError", "code": 400}}
FLOOR_400 = {"error": {"message": "The decoder prompt (length 17000) is longer than the maximum model "
                                  "length of 16384. Make sure that max_model_len is no smaller than the "
                                  "number of text tokens.", "type": "BadRequestError", "code": 400}}
SCHEMA_400 = {"error": {"message": "1 validation error for ChatCompletionRequest\ntools.0.function."
                                   "parameters\n  Input should be a valid dictionary", "type": "BadRequestError",
                        "code": 400}}
FIELD_400 = {"error": {"message": "unsupported field: prompt_logprobs is not allowed with tools",
                       "type": "BadRequestError", "code": 400}}


def test_error_class_is_derived_in_the_ledger_from_status_and_body():
    """E12 item 5: the ledger, not a reader, classifies every non-2xx — a 400
    is ``context_length`` ONLY when its body is one of the pinned vLLM
    context-length forms (``served_window.parse_context_overflow``); a
    tool-schema, unsupported-field or malformed-prompt 400 is
    ``bad_request_other``; auth / route / rate-limit / server classes by
    status; the status decides first (a 5xx quoting the context text is a
    server error); no status = transport."""
    cls = C
    assert ul.error_class_of(400, json.dumps(CONTEXT_400).encode()) == cls.LEDGER_ERROR_CLASS_CONTEXT_LENGTH
    assert ul.error_class_of(400, json.dumps(FLOOR_400).encode()) == cls.LEDGER_ERROR_CLASS_CONTEXT_LENGTH
    assert ul.error_class_of(400, json.dumps(CONTEXT_400)) == cls.LEDGER_ERROR_CLASS_CONTEXT_LENGTH   # str body too
    for body in (SCHEMA_400, FIELD_400, {"error": {"message": "malformed prompt: role 'tool' without a call"}},
                 {"error": {"message": "BadRequestError", "code": "BadRequestError"}}):
        assert ul.error_class_of(400, json.dumps(body).encode()) == cls.LEDGER_ERROR_CLASS_BAD_REQUEST_OTHER, body
    assert ul.error_class_of(400, b"not json at all") == cls.LEDGER_ERROR_CLASS_BAD_REQUEST_OTHER
    assert ul.error_class_of(400, None) == cls.LEDGER_ERROR_CLASS_BAD_REQUEST_OTHER
    assert ul.error_class_of(401, b'{"error": {"code": "bad_token"}}') == cls.LEDGER_ERROR_CLASS_AUTH
    assert ul.error_class_of(403, b"") == cls.LEDGER_ERROR_CLASS_AUTH
    for status in (404, 409, 421):                                     # the RoutingProxy's refusals
        assert ul.error_class_of(status, b"") == cls.LEDGER_ERROR_CLASS_ROUTE_REFUSED, status
    assert ul.error_class_of(429, b"") == cls.LEDGER_ERROR_CLASS_RATE_LIMITED
    for status in (500, 502, 503):
        assert ul.error_class_of(status, b"") == cls.LEDGER_ERROR_CLASS_SERVER_ERROR, status
    assert ul.error_class_of(502, json.dumps(CONTEXT_400).encode()) == cls.LEDGER_ERROR_CLASS_SERVER_ERROR
    assert ul.error_class_of(418, b"") == cls.LEDGER_ERROR_CLASS_BAD_REQUEST_OTHER   # any other 4xx: never context-length
    assert ul.error_class_of(200, b"") == ul.error_class_of(201, b"") == cls.LEDGER_ERROR_CLASS_NONE
    assert ul.error_class_of(None, b"") == cls.LEDGER_ERROR_CLASS_TRANSPORT
    assert set(C.LEDGER_ERROR_CLASSES) == {"none", "context_length", "bad_request_other", "auth",
                                           "route_refused", "rate_limited", "server_error", "transport",
                                           "ledger_blocked"}
    assert len(C.LEDGER_ERROR_CLASSES) == len(set(C.LEDGER_ERROR_CLASSES))
    # appended, additive: error_class (E12 item 5), then the per-attempt identity;
    # a row from before either lacks the later keys
    assert C.LEDGER_ROW_FIELDS[-3:] == ("error_class", "request_id", "body_sha256")


def test_every_row_carries_the_attempts_request_id_and_the_digest_of_the_bytes_received(tmp_path):
    """The ledger row of EVERY attempt — served, rejected upstream, blocked at
    tokenize, bad JSON — carries `request_id` (the client's X-Tinycap-Request-Id
    verbatim; null when it sent none) and `body_sha256` = sha256 of the exact
    bytes received, which are the bytes forwarded; the request-id header is
    consumed here and never travels upstream, like the task / sample labels."""
    import hashlib
    rid = "7f" * 16
    hdr = {**HEADERS, C.HEADER_REQUEST_ID: rid}
    raw = json.dumps(BODY).encode()
    # served
    t = FakeTransport(); led = _ledger(tmp_path, t)
    status, _, _ = asyncio.run(led.handle_chat(raw, hdr))
    [row] = _rows(tmp_path / C.LEDGER_FILENAME)
    assert status == 200 and row["request_id"] == rid and row["body_sha256"] == hashlib.sha256(raw).hexdigest()
    assert row["body_sha256"] == C.wire_body_sha256(raw) and tuple(row) == C.LEDGER_ROW_FIELDS
    fwd = [h for m, p, b, h in t.calls if p == ul.CHAT_PATH]
    assert fwd and all(C.HEADER_REQUEST_ID.lower() not in {k.lower() for k in h} for h in fwd)
    assert [b for m, p, b, h in t.calls if p == ul.CHAT_PATH] == [raw]                 # the same bytes, forwarded
    # rejected upstream: identity still on the row
    led2 = _ledger(tmp_path / "r", FakeTransport(chat=(400, {"error": {"code": "boom", "message": "x"}})))
    asyncio.run(led2.handle_chat(raw, hdr))
    [row2] = _rows(tmp_path / "r" / C.LEDGER_FILENAME)
    assert row2["rejected"] is True and row2["request_id"] == rid and row2["body_sha256"] == C.wire_body_sha256(raw)
    # blocked at tokenize, and a body that is not JSON: still identified
    led3 = _ledger(tmp_path / "b", FakeTransport(tokenize_status=500))
    asyncio.run(led3.handle_chat(raw, hdr))
    [row3] = _rows(tmp_path / "b" / C.LEDGER_FILENAME)
    assert row3["error_code"] == ul.ERROR_CODE_TOKENIZE_FAILED and row3["request_id"] == rid
    assert row3["body_sha256"] == C.wire_body_sha256(raw)
    led4 = _ledger(tmp_path / "j", FakeTransport())
    asyncio.run(led4.handle_chat(b"[1, 2", hdr))
    [row4] = _rows(tmp_path / "j" / C.LEDGER_FILENAME)
    assert row4["error_code"] == ul.ERROR_CODE_BAD_JSON and row4["body_sha256"] == C.wire_body_sha256(b"[1, 2")
    # a client that sent no id: null, never invented
    led5 = _ledger(tmp_path / "n", FakeTransport())
    asyncio.run(led5.handle_chat(raw, HEADERS))
    [row5] = _rows(tmp_path / "n" / C.LEDGER_FILENAME)
    assert row5["request_id"] is None and row5["body_sha256"] == C.wire_body_sha256(raw)
    # a labelled probe rows like any attempt, its label verbatim
    led6 = _ledger(tmp_path / "p", FakeTransport())
    asyncio.run(led6.handle_chat(raw, {**hdr, "X-Tinycap-Task": C.probe_task_label(C.PROBE_KIND_RENDER),
                                       "X-Tinycap-Sample": "petri__base/1"}))
    [row6] = _rows(tmp_path / "p" / C.LEDGER_FILENAME)
    assert (row6["task"], row6["sample"]) == ("probe/render", "petri__base/1") and C.is_probe_task(row6["task"])


@pytest.mark.parametrize("case", ["served", "context_length", "schema_400", "field_400", "auth_401",
                                  "route_404", "rate_429", "server_502", "transport",
                                  "tokenize_failed", "bad_json"])
def test_every_row_carries_the_ledgers_error_class(tmp_path, case):
    """The column on the WIRE: every outcome of ``handle_chat`` writes its
    class, and the row stays exactly ``LEDGER_ROW_FIELDS``."""
    body = json.dumps(BODY).encode()
    chat = None
    kw: dict = {}
    if case == "served":
        want = "none"
    elif case == "context_length":
        chat, want = (400, CONTEXT_400), "context_length"
    elif case == "schema_400":
        chat, want = (400, SCHEMA_400), "bad_request_other"
    elif case == "field_400":
        chat, want = (400, FIELD_400), "bad_request_other"
    elif case == "auth_401":
        chat, want = (401, {"error": {"message": "bad token", "code": "bad_token"}}), "auth"
    elif case == "route_404":
        chat, want = (404, {"error": {"message": "unknown virtual model", "code": "unknown_model"}}), "route_refused"
    elif case == "rate_429":
        chat, want = (429, {"error": {"message": "slow down"}}), "rate_limited"
    elif case == "server_502":
        chat, want = (502, {"error": {"message": "lens postcheck failed", "code": "lens_postcheck_failed"}}), "server_error"
    elif case == "transport":
        kw["fail_chat"], want = ConnectionResetError("reset"), "transport"
    elif case == "tokenize_failed":
        kw["tokenize_status"], want = 500, "ledger_blocked"
    else:
        body, want = b"[1, 2", "ledger_blocked"
    led = _ledger(tmp_path, FakeTransport(chat=chat, **kw))
    asyncio.run(led.handle_chat(body, HEADERS))
    [r] = _rows(led.writer.path)
    assert tuple(r) == C.LEDGER_ROW_FIELDS
    assert r["error_class"] == want, (case, r)
    assert r["rejected"] is (want != "none")
    assert (r["error_class"] == "context_length") == (r["status"] == 400 and "context" in json.dumps(chat or {}))


def test_tokenize_failure_blocks_the_forward_and_is_a_row(tmp_path):
    ft = FakeTransport(tokenize_status=500)
    led = _ledger(tmp_path, ft)
    status, resp, _ = asyncio.run(led.handle_chat(json.dumps(BODY).encode(), HEADERS))
    assert status == 502 and json.loads(resp)["error"]["code"] == ul.ERROR_CODE_TOKENIZE_FAILED
    assert [c[1] for c in ft.calls] == [ul.TOKENIZE_PATH]          # NO chat call
    r = _rows(led.writer.path)[0]
    assert r["rejected"] is True and r["status"] is None
    assert r["prompt_tokens_pre"] is None and r["error_code"] == ul.ERROR_CODE_TOKENIZE_FAILED
    assert r["task"] == "inspect/math" and r["vmodel"] == "petri__base"
    assert led.stats()["n_blocked"] == 1 and led.stats()["n_forwarded"] == 0
    # a body without 'model' cannot be tokenized either: blocked, ledgered
    status, _, _ = asyncio.run(led.handle_chat(b'{"messages": []}', {}))
    assert status == 502
    assert _rows(led.writer.path)[1]["error_code"] == ul.ERROR_CODE_TOKENIZE_FAILED
    # and a non-object body is its own row
    status, _, _ = asyncio.run(led.handle_chat(b"[1, 2", {}))
    assert status == 400
    assert _rows(led.writer.path)[2]["error_code"] == ul.ERROR_CODE_BAD_JSON
    assert len(ft.calls) == 1                                       # nothing else went upstream


def test_transport_failure_on_forward_is_a_rejected_row(tmp_path):
    ft = FakeTransport(fail_chat=ConnectionRefusedError("down"))
    led = _ledger(tmp_path, ft)
    status, resp, _ = asyncio.run(led.handle_chat(json.dumps(BODY).encode(), HEADERS))
    assert status == 502 and json.loads(resp)["error"]["code"] == ul.ERROR_CODE_TRANSPORT
    r = _rows(led.writer.path)[0]
    assert r["rejected"] is True and r["status"] is None and r["prompt_tokens_pre"] == 17
    assert r["error_code"] == ul.ERROR_CODE_TRANSPORT


def test_tokenize_mismatch_flags_a_served_disagreement(tmp_path):
    ft = FakeTransport(tokenize_count=16)          # usage says 17
    led = _ledger(tmp_path, ft)
    asyncio.run(led.handle_chat(json.dumps(BODY).encode(), HEADERS))
    r = _rows(led.writer.path)[0]
    assert r["prompt_tokens_pre"] == 16 and r["usage_prompt_tokens"] == 17
    assert r["tokenize_mismatch"] is True and led.stats()["n_tokenize_mismatch"] == 1


def test_rows_append_across_calls_and_the_file_is_reread_whole(tmp_path):
    ft = FakeTransport()
    led = _ledger(tmp_path, ft)
    for i in range(3):
        hdr = dict(HEADERS, **{"X-Tinycap-Sample": f"{i}/0"})
        asyncio.run(led.handle_chat(json.dumps(BODY).encode(), hdr))
    rows = _rows(led.writer.path)
    assert [r["sample"] for r in rows] == ["0/0", "1/0", "2/0"]
    assert led.writer.n_rows == 3
    # a second writer APPENDS (resume never truncates)
    led2 = _ledger(tmp_path, FakeTransport())
    asyncio.run(led2.handle_chat(json.dumps(BODY).encode(), HEADERS))
    assert len(_rows(led.writer.path)) == 4


def test_headers_are_optional_and_thinking_is_read_from_the_body(tmp_path):
    ft = FakeTransport()
    led = _ledger(tmp_path, ft)
    body = dict(BODY, chat_template_kwargs={"enable_thinking": True},
                tools=[{"type": "function", "function": {"name": "f"}}])
    asyncio.run(led.handle_chat(json.dumps(body).encode(), {"Content-Type": "application/json"}))
    r = _rows(led.writer.path)[0]
    assert r["task"] is None and r["sample"] is None
    assert r["thinking"] is True and r["has_tools"] is True and r["n_tools"] == 1


# --------------------------------------------------------------------------- #
# startup probe                                                                #
# --------------------------------------------------------------------------- #


def test_probe_passes_when_models_and_tokenize_answer(tmp_path):
    ft = FakeTransport(tokenize_count=4)
    led = _ledger(tmp_path, ft, probe_bearer="secret")
    out = asyncio.run(led.probe())
    assert out["virtual_models"] == ["petri__base"] and out["probe_prompt_tokens"] == 4
    assert [c[1] for c in ft.calls] == [ul.MODELS_PATH, ul.TOKENIZE_PATH]
    assert json.loads(ft.calls[1][2])["model"] == "petri__base"
    assert ft.calls[0][3]["Authorization"] == "Bearer secret"


def test_probe_refuses_without_tokenize_or_models(tmp_path):
    led = _ledger(tmp_path, FakeTransport(tokenize_status=501))
    with pytest.raises(ul.LedgerStartupRefused, match="tokenize"):
        asyncio.run(led.probe())
    led = _ledger(tmp_path, FakeTransport(models={"object": "list", "data": []}))
    with pytest.raises(ul.LedgerStartupRefused, match="no served virtual model"):
        asyncio.run(led.probe())

    async def dead(method, path, body, headers):
        raise ConnectionRefusedError("nobody home")
    led = _ledger(tmp_path, dead)
    with pytest.raises(ul.LedgerStartupRefused, match="did not answer"):
        asyncio.run(led.probe())


def test_passthrough_strips_tinycap_headers_and_is_not_ledgered(tmp_path):
    ft = FakeTransport()
    led = _ledger(tmp_path, ft)
    status, raw, _ = asyncio.run(led.passthrough("GET", ul.MODELS_PATH, None, HEADERS))
    assert status == 200 and json.loads(raw)["data"][0]["id"] == "petri__base"
    status, raw, _ = asyncio.run(led.passthrough(
        "POST", ul.TOKENIZE_PATH, json.dumps({"model": "petri__base", "messages": []}).encode(),
        HEADERS))
    assert status == 200 and json.loads(raw)["count"] == 17
    assert not any(k.lower().startswith("x-tinycap") for c in ft.calls for k in c[3])
    assert not led.writer.path.exists() or _rows(led.writer.path) == []


def test_cli_parser_and_wildcard_refusal(tmp_path):
    args = ul.build_parser().parse_args(
        ["--upstream", "http://127.0.0.1:5/v1", "--port", "7", "--ledger-path", "x.jsonl"])
    assert args.host == "127.0.0.1" and args.port == 7 and args.endpoint_json is None
    with pytest.raises(SystemExit):
        ul.main(["--upstream", "http://127.0.0.1:5/v1", "--port", "7",
                 "--ledger-path", str(tmp_path / "x.jsonl"), "--host", "0.0.0.0"])


# --------------------------------------------------------------------------- #
# in-process aiohttp end to end                                                #
# --------------------------------------------------------------------------- #


def test_aiohttp_end_to_end_through_make_app(tmp_path):
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    seen: dict = {}

    async def up_chat(request):
        seen["raw"] = await request.read()
        seen["headers"] = dict(request.headers)
        return web.json_response({"choices": [{"index": 0, "finish_reason": "length",
                                               "message": {"role": "assistant", "content": "x"}}],
                                  "usage": {"prompt_tokens": 9, "completion_tokens": 1}})

    async def up_tokenize(request):
        body = await request.json()
        seen.setdefault("tokenize_bodies", []).append(body)
        return web.json_response({"count": 9, "max_model_len": 16384, "tokens": list(range(9))})

    async def up_models(request):
        return web.json_response({"object": "list", "data": [{"id": "petri__base"}]})

    async def run():
        upstream = web.Application()
        upstream.router.add_post("/v1/chat/completions", up_chat)
        upstream.router.add_post("/v1/tokenize", up_tokenize)
        upstream.router.add_get("/v1/models", up_models)
        async with TestServer(upstream) as up:
            led = ul.UsageLedger(f"http://127.0.0.1:{up.port}/v1", tmp_path / "led.jsonl")
            await led.probe()
            async with TestClient(TestServer(ul.make_app(led))) as client:
                raw = json.dumps(BODY, indent=2).encode()
                # the fake-transport fixture's bogus Host / Content-Length would be
                # sent verbatim by a real client (and the server would wait for
                # 999 bytes); a real client computes its own
                wire_headers = {k: v for k, v in HEADERS.items()
                                if k.lower() not in ("host", "content-length")}
                resp = await client.post("/v1/chat/completions", data=raw, headers=wire_headers)
                assert resp.status == 200
                assert (await resp.json())["choices"][0]["finish_reason"] == "length"
                m = await client.get("/v1/models", headers={"X-Tinycap-Task": "t"})
                assert (await m.json())["data"][0]["id"] == "petri__base"
                s = await client.get("/ledger/stats")
                stats = await s.json()
            await led.aclose()     # the production transport holds one keep-alive session
            return stats

    stats = asyncio.run(run())
    assert seen["raw"] == json.dumps(BODY, indent=2).encode()
    hdrs = {k.lower(): v for k, v in seen["headers"].items()}
    assert hdrs["authorization"] == "Bearer tok" and "x-tinycap-task" not in hdrs
    assert seen["tokenize_bodies"][-1] == ul.tokenize_body_for(BODY)
    rows = _rows(tmp_path / "led.jsonl")
    assert len(rows) == 1 and rows[0]["prompt_tokens_pre"] == 9 and rows[0]["rejected"] is False
    assert rows[0]["finish_reason"] == "length" and rows[0]["tokenize_mismatch"] is False
    assert stats["n_rows"] == 1 and stats["n_forwarded"] == 1


# --------------------------------------------------------------------------- #
# per-session attribution fallback: X-Session-ID                                #
# --------------------------------------------------------------------------- #


def test_sample_falls_back_to_x_session_id_when_tinycap_sample_absent(tmp_path):
    """A client that sets no X-Tinycap-Sample but tags every request of a session
    with ``X-Session-ID``: the row's ``sample`` takes it, and the header is NOT
    consumed — it travels upstream."""
    ft = FakeTransport()
    led = _ledger(tmp_path, ft)
    headers = {"Content-Type": "application/json", "Authorization": "Bearer petri",
               C.HEADER_TASK: "agentic/demo", "X-Session-ID": "task-a__Ab3dEfG__agent"}
    asyncio.run(led.handle_chat(json.dumps(BODY).encode(), headers))
    r = _rows(led.writer.path)[0]
    assert r["task"] == "agentic/demo" and r["sample"] == "task-a__Ab3dEfG__agent"
    # forwarded on BOTH upstream legs (tokenize + chat), unlike the two tinycap headers
    for _, path, _, fwd in ft.calls:
        assert fwd.get("X-Session-ID") == "task-a__Ab3dEfG__agent", path
        assert C.HEADER_TASK not in fwd and C.HEADER_SAMPLE not in fwd
    # a sub-session id lands verbatim, whatever the header's casing
    sub = {k: v for k, v in headers.items() if k != "X-Session-ID"}
    sub["x-session-id"] = "task-a__Ab3dEfG__agent-sub-1"      # any casing
    asyncio.run(led.handle_chat(json.dumps(BODY).encode(), sub))
    assert _rows(led.writer.path)[1]["sample"] == "task-a__Ab3dEfG__agent-sub-1"
    assert ul.HEADER_SESSION_ID == "X-Session-ID" and "HEADER_SESSION_ID" in ul.__all__


def test_x_tinycap_sample_keeps_precedence_over_x_session_id(tmp_path):
    ft = FakeTransport()
    led = _ledger(tmp_path, ft)
    headers = {"Content-Type": "application/json", C.HEADER_SAMPLE: "HumanEval/3/0",
               "X-Session-ID": "regex-log__Ab3dEfG__agent"}
    asyncio.run(led.handle_chat(json.dumps(BODY).encode(), headers))
    r = _rows(led.writer.path)[0]
    assert r["sample"] == "HumanEval/3/0"
    # and with neither header the column stays null, exactly as before
    asyncio.run(led.handle_chat(json.dumps(BODY).encode(), {"Content-Type": "application/json"}))
    assert _rows(led.writer.path)[1]["sample"] is None
    # forward_headers is unchanged: X-Session-ID is not in the consumed set
    assert ul.forward_headers({"X-Session-ID": "s", C.HEADER_SAMPLE: "x"}) == {"X-Session-ID": "s"}


# ---------------------------------------------------------------------------- #
# the OPTIONAL attempt_id column (the attempt-scoped producer contract):       #
# stamped only under --attempt-id; the row schema LEDGER_ROW_FIELDS is         #
# unchanged, so a ledger without the flag writes the same bytes it always did  #
# ---------------------------------------------------------------------------- #


def test_ledger_row_admits_only_the_optional_attempt_column_appended_last():
    base = {k: None for k in C.LEDGER_ROW_FIELDS}
    assert tuple(ul.ledger_row(**base)) == C.LEDGER_ROW_FIELDS            # absent: the schema, byte for byte
    stamped = ul.ledger_row(**base, **{C.LEDGER_ATTEMPT_ID_FIELD: "a2"})
    assert tuple(stamped) == (*C.LEDGER_ROW_FIELDS, C.LEDGER_ATTEMPT_ID_FIELD)
    assert stamped[C.LEDGER_ATTEMPT_ID_FIELD] == "a2"
    assert C.LEDGER_ATTEMPT_ID_FIELD not in C.LEDGER_ROW_FIELDS            # the schema itself did not move
    with pytest.raises(ul.LedgerError, match="unknown"):                  # still exactly ONE optional column
        ul.ledger_row(**base, **{C.LEDGER_ATTEMPT_ID_FIELD: "a2"}, extra=1)


def test_a_ledger_under_attempt_id_stamps_every_row_and_one_without_writes_the_old_bytes(tmp_path):
    raw = json.dumps(BODY).encode()
    led = ul.UsageLedger("http://127.0.0.1:1/v1", tmp_path / "a.jsonl", transport=FakeTransport(),
                         attempt_id="a3")
    asyncio.run(led.handle_chat(raw, HEADERS))
    asyncio.run(led.handle_chat(raw, HEADERS))
    rows = _rows(led.writer.path)
    assert len(rows) == 2 and all(r[C.LEDGER_ATTEMPT_ID_FIELD] == "a3" for r in rows)
    assert all(tuple(r) == (*C.LEDGER_ROW_FIELDS, C.LEDGER_ATTEMPT_ID_FIELD) for r in rows)
    # a REJECTED row (a tokenize failure blocks the forward) is stamped too: attribution never
    # depends on the outcome, or the rows a reader most needs to attribute would be the unattributed ones
    led_rej = ul.UsageLedger("http://127.0.0.1:1/v1", tmp_path / "r.jsonl",
                             transport=FakeTransport(tokenize_status=500), attempt_id="a3")
    asyncio.run(led_rej.handle_chat(raw, HEADERS))
    (rr,) = _rows(led_rej.writer.path)
    assert rr["rejected"] is True and rr[C.LEDGER_ATTEMPT_ID_FIELD] == "a3"
    # without the flag: exactly the contract schema and no column at all
    plain = ul.UsageLedger("http://127.0.0.1:1/v1", tmp_path / "p.jsonl", transport=FakeTransport())
    asyncio.run(plain.handle_chat(raw, HEADERS))
    (pr,) = _rows(plain.writer.path)
    assert tuple(pr) == C.LEDGER_ROW_FIELDS and C.LEDGER_ATTEMPT_ID_FIELD not in pr


def test_attempt_id_is_validated_at_the_constructor_and_at_the_cli(tmp_path):
    with pytest.raises(C.AttemptScopeError):
        ul.UsageLedger("http://127.0.0.1:1/v1", tmp_path / "v.jsonl", transport=FakeTransport(),
                       attempt_id="1")
    common = ["--upstream", "http://127.0.0.1:5/v1", "--port", "7", "--ledger-path", "x.jsonl"]
    assert ul.build_parser().parse_args(common).attempt_id is None
    assert ul.build_parser().parse_args([*common, "--attempt-id", "a1"]).attempt_id == "a1"
    with pytest.raises(SystemExit):                                       # argparse turns the ValueError into a usage error
        ul.build_parser().parse_args([*common, "--attempt-id", "attempt-1"])


def test_read_rows_skips_blank_torn_and_non_object_lines_and_requires_the_file(tmp_path):
    """The simplify pass lifted `rejected_rows_for_task`'s JSONL reading into
    `read_rows` (which `inspect_served._jsonl_records` now calls too). The
    reading is the pre-change one, line for line: a blank line, a line that is
    not a JSON object (an array, a string) and a TORN line (an interrupted
    write — trailing or not) are skipped; an empty file is no rows; a MISSING
    file is refused with the OS error, exactly as `read_text` did (the callers
    decide — `_jsonl_records` answers `[]` for an absent file,
    `rejected_rows_for_task` lets the refusal through, and the runner names
    it); `rejected` counts on the JSON boolean `true` alone."""
    p = tmp_path / C.LEDGER_FILENAME
    task = C.GSM8K_STANDARD_TASK
    rows = [{"task": task, "rejected": True, "request_id": "a"},
            {"task": task, "rejected": False, "request_id": "b"},
            {"task": C.inspect_task("math"), "rejected": True, "request_id": "c"}]
    p.write_text(json.dumps(rows[0]) + "\n\n" + json.dumps(rows[1]) + "\n[1, 2]\n\"a string\"\n"
                 + json.dumps(rows[2]) + '\n{"task": "' + task + '", "rejec', encoding="utf-8")
    assert ul.read_rows(p) == rows                                   # blank, array, string, torn tail: skipped
    assert ul.read_rows(str(p)) == rows
    assert ul.rejected_rows_for_task(p, task) == 1
    assert ul.rejected_rows_for_task(p, C.inspect_task("math")) == 1 and ul.rejected_rows_for_task(p, "nope") == 0
    # a torn line in the MIDDLE is skipped the same way (every reader of these files does)
    p.write_text(json.dumps(rows[0]) + "\n{torn\n" + json.dumps(rows[1]) + "\n", encoding="utf-8")
    assert ul.read_rows(p) == rows[:2] and ul.rejected_rows_for_task(p, task) == 1
    # an empty file is no rows; a missing file is refused, not read as empty
    p.write_text("", encoding="utf-8")
    assert ul.read_rows(p) == [] and ul.rejected_rows_for_task(p, task) == 0
    with pytest.raises(FileNotFoundError):
        ul.read_rows(tmp_path / "missing.jsonl")
    with pytest.raises(FileNotFoundError):
        ul.rejected_rows_for_task(tmp_path / "missing.jsonl", task)
    # `rejected` is the JSON boolean true — a truthy string, 1 or null never counts
    p.write_text("\n".join(json.dumps({"task": task, "rejected": v}) for v in (True, "true", 1, None)) + "\n",
                 encoding="utf-8")
    assert len(ul.read_rows(p)) == 4 and ul.rejected_rows_for_task(p, task) == 1
    # the writer's rows re-read whole through the same reader
    w = ul.LedgerWriter(tmp_path / "w.jsonl")
    base = dict.fromkeys(ul.LEDGER_ROW_FIELDS)
    base.update(task=task, rejected=True)
    w.write(base)
    w.write({**base, "rejected": False})
    assert [r["rejected"] for r in ul.read_rows(tmp_path / "w.jsonl")] == [True, False]
    assert ul.rejected_rows_for_task(tmp_path / "w.jsonl", task) == 1
