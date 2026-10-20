"""The usage ledger RELAYS the RoutingProxy's typed render refusal off its
pre-tokenize instead of re-labelling it ``502 ledger_tokenize_failed``.

THE CHAIN. Every served client talks to the usage ledger
(``sisteer.capeval.serving.usage_ledger``, a loopback proxy in FRONT of the
RoutingProxy); the ledger tokenizes the EXACT outbound body through the
RoutingProxy's ``/v1/tokenize`` before forwarding ("an unmeasured call is not
forwarded"). ``RoutingProxy.handle_tokenize`` applies the chat endpoint's
``chat_template_kwargs`` render gate: on a thinking-ON standup
``enable_thinking: false`` refuses ``400 thinking_regime_mismatch`` — the SAME
typed refusal the chat endpoint would give the same body. At c65b47ba4 the
ledger folded that into ``502 ledger_tokenize_failed`` (its "tokenize broke"
code), so the client could not see the engine's typed answer: every GSM8K
``none`` child died ``THINKING_UNSERVABLE`` quoting the 502.

THE FIX. A typed render refusal from the tokenize (one of
:data:`usage_ledger.RELAYED_TOKENIZE_REFUSAL_CODES`, exactly the codes the
RoutingProxy's tokenize render gate emits) is relayed: same status, same
body, plus the ledger's own ``via: ledger_tokenize`` marker inside the
``error`` object. Nothing is forwarded to chat (the ledger's invariant holds)
and the ledger row is the blocked row it always was (accounting untouched).
Every OTHER tokenize failure — transport, a 5xx, an untyped 4xx, a malformed
body — stays ``502 ledger_tokenize_failed``.

Three layers, all synthetic (placeholder vmodels, invented header ids, the
proxy tests' canonical cell set; no real recorded data):

1. the ledger core over an injected fake transport (no sockets);
2. the FULL REAL CHAIN over loopback aiohttp: a real ``RoutingProxy`` in
   ``thinking_mode="on"`` in front of a stub engine, the real ``UsageLedger``
   app in front of the proxy, an HTTP client in front of the ledger — the
   stub engine's chat hit count is asserted 0 on the relayed refusal;
3. the driver's REAL predicate-(A) probe (``served_capability_run._render_probe``
   through ``ProxyClientEngine``) against that chain — it pins what the GSM8K
   ``none`` child now sees: the typed 400, not the 502. NOTE (the residual,
   named in the test): ``_render_probe`` runs the OFF arm first and treats ANY
   refusal as fatal, and ``contracts.render_certified`` requires the OFF probe's
   ``render_regime`` to read ``thinking_off`` — so on a thinking-ON standup the
   probe still exits ``THINKING_UNSERVABLE``, now quoting the engine's own
   refusal by code. Making predicate (A) accept the typed OFF-arm refusal on an
   ON-pinned standup is a contract change (``contracts.render_certified``) and is
   NOT made here.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sisteer.capeval.serving import proxy as px  # noqa: E402
from sisteer.capeval.serving import usage_ledger as ul  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402

from test_petri_proxy import StubBackend, _TailRender, _chat  # noqa: E402
from test_thinking_on_proxy import MOVES, _cells  # noqa: E402

VMODEL = "petri__base"
OFF_BODY = {"model": VMODEL, "messages": [{"role": "user", "content": "2+2?"}],
            "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False}}
ON_BODY = {"model": VMODEL, "messages": [{"role": "user", "content": "2+2?"}], "max_tokens": 8}
HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer tok",
           "X-Tinycap-Task": "probe", "X-Tinycap-Sample": "render"}


def _typed_refusal_bytes(code: str, message: str = "refused by name") -> bytes:
    """The RoutingProxy's wire shape for a ProxyRefusal (``proxy._error``)."""
    return json.dumps({"error": {"message": message, "type": "invalid_request_error",
                                 "code": code}}).encode("utf-8")


class RegimeGateTransport:
    """A fake RoutingProxy transport: ``/v1/tokenize`` answers the typed
    ``thinking_regime_mismatch`` 400 for an ``enable_thinking: false`` body
    (mirroring ``RoutingProxy.handle_tokenize`` on a thinking-ON standup) and a
    count otherwise; ``/v1/chat/completions`` counts its hits."""

    def __init__(self, *, tokenize_override=None, tokenize_raises: Exception | None = None):
        self.calls: list[tuple[str, str, bytes | None, dict]] = []
        self.chat_hits = 0
        self.tokenize_override = tokenize_override      # (status, bytes) for every tokenize
        self.tokenize_raises = tokenize_raises

    async def __call__(self, method, path, body, headers):
        self.calls.append((method, path, body, dict(headers)))
        if path == ul.MODELS_PATH:
            return 200, json.dumps({"object": "list", "data": [{"id": VMODEL}]}).encode(), "application/json"
        if path == ul.TOKENIZE_PATH:
            if self.tokenize_raises is not None:
                raise self.tokenize_raises
            if self.tokenize_override is not None:
                status, raw = self.tokenize_override
                return status, raw, "application/json"
            req = json.loads(body)
            if (req.get(C.CHAT_TEMPLATE_KWARGS_KEY) or {}).get(C.ENABLE_THINKING_KEY) is False:
                return 400, _typed_refusal_bytes(
                    px.THINKING_REGIME_MISMATCH,
                    "chat_template_kwargs.enable_thinking: false on a thinking-ON standup's tokenize"
                    " — it would measure a render the chat endpoint refuses; drop the key"), "application/json"
            return 200, json.dumps({"count": 5, "max_model_len": 4096, "tokens": [1, 2, 3, 4, 5]}).encode(), \
                "application/json"
        if path == ul.CHAT_PATH:
            self.chat_hits += 1
            return 200, json.dumps({"id": "x", "model": VMODEL,
                                    "choices": [{"index": 0, "finish_reason": "stop",
                                                 "message": {"role": "assistant", "content": "4"}}],
                                    "usage": {"prompt_tokens": 5, "completion_tokens": 1,
                                              "total_tokens": 6}}).encode(), "application/json"
        raise AssertionError(f"unexpected upstream path {path}")


def _ledger(tmp_path, transport) -> ul.UsageLedger:
    return ul.UsageLedger("http://127.0.0.1:1/v1", tmp_path / C.LEDGER_FILENAME, transport=transport)


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _post(led, body):
    return asyncio.run(led.handle_chat(json.dumps(body).encode(), HEADERS))


# --------------------------------------------------------------------------- #
# 1. the ledger core over a fake transport                                      #
# --------------------------------------------------------------------------- #


def test_the_relay_class_is_exactly_the_tokenize_render_gates_codes():
    # the RoutingProxy's tokenize render gate (handle_tokenize: _validate_kwargs_shape + the thinking-ON
    # regime check) emits exactly these three; every one is a THINKING_REFUSAL_CODE
    assert set(ul.RELAYED_TOKENIZE_REFUSAL_CODES) == {px.THINKING_KWARGS_INVALID, px.THINKING_KWARGS_INERT,
                                                      px.THINKING_REGIME_MISMATCH}
    assert set(ul.RELAYED_TOKENIZE_REFUSAL_CODES) <= set(px.THINKING_REFUSAL_CODES)
    assert ul.RELAY_VIA_KEY == "via" and ul.RELAY_VIA_LEDGER_TOKENIZE == "ledger_tokenize"


def test_typed_regime_refusal_off_the_tokenize_is_relayed_not_relabelled_502(tmp_path):
    # known-bad at c65b47ba4: status 502, code ledger_tokenize_failed, the engine's code buried in the prose
    tr = RegimeGateTransport()
    led = _ledger(tmp_path, tr)
    status, raw, ctype = _post(led, OFF_BODY)
    err = json.loads(raw)["error"]
    assert status == 400, (status, raw)
    assert err["code"] == px.THINKING_REGIME_MISMATCH, err
    assert err["type"] == "invalid_request_error"
    assert err[ul.RELAY_VIA_KEY] == ul.RELAY_VIA_LEDGER_TOKENIZE
    assert "thinking-ON standup" in err["message"]
    assert ctype == "application/json"
    # nothing reached chat: the call is still unmeasured-and-not-forwarded
    assert tr.chat_hits == 0
    assert [p for _, p, _, _ in tr.calls] == [ul.TOKENIZE_PATH]
    # the tokenized body is the EXACT outbound render (chat_template_kwargs travel — never dropped)
    tok_body = json.loads(tr.calls[0][2])
    assert tok_body[C.CHAT_TEMPLATE_KWARGS_KEY] == {C.ENABLE_THINKING_KEY: False}
    assert tok_body == ul.tokenize_body_for(OFF_BODY)
    # the ledger's accounting is the blocked row it always was
    rows = _rows(led.writer.path)
    assert len(rows) == 1
    r = rows[0]
    assert r["rejected"] is True and r["status"] is None and r["prompt_tokens_pre"] is None
    assert r["error_code"] == ul.ERROR_CODE_TOKENIZE_FAILED
    assert r["error_class"] == C.LEDGER_ERROR_CLASS_LEDGER_BLOCKED
    assert r["thinking"] is False and r["task"] == "probe" and r["sample"] == "render"
    assert led.stats()["n_blocked"] == 1 and led.stats()["n_forwarded"] == 0


@pytest.mark.parametrize("code", [px.THINKING_KWARGS_INVALID, px.THINKING_KWARGS_INERT])
def test_the_other_render_gate_codes_relay_the_same_way(tmp_path, code):
    tr = RegimeGateTransport(tokenize_override=(400, _typed_refusal_bytes(code)))
    led = _ledger(tmp_path, tr)
    status, raw, _ = _post(led, ON_BODY)
    err = json.loads(raw)["error"]
    assert status == 400 and err["code"] == code and err[ul.RELAY_VIA_KEY] == ul.RELAY_VIA_LEDGER_TOKENIZE
    assert tr.chat_hits == 0
    assert _rows(led.writer.path)[0]["error_code"] == ul.ERROR_CODE_TOKENIZE_FAILED


@pytest.mark.parametrize("case", ["tokenize_500", "tokenize_400_untyped", "tokenize_400_other_thinking_code",
                                  "tokenize_400_not_json", "tokenize_200_no_count", "transport_failure"])
def test_every_other_tokenize_failure_stays_502_ledger_tokenize_failed(tmp_path, case):
    if case == "tokenize_500":
        tr = RegimeGateTransport(tokenize_override=(500, b'{"error": {"message": "engine down", "code": "boom"}}'))
    elif case == "tokenize_400_untyped":
        tr = RegimeGateTransport(tokenize_override=(400, b'{"error": {"message": "nope", "code": "boom"}}'))
    elif case == "tokenize_400_other_thinking_code":
        # a chat-only declaration code the tokenize gate never emits: NOT relayed (the class is exact)
        tr = RegimeGateTransport(tokenize_override=(400, _typed_refusal_bytes(px.THINKING_ON_UNDECLARED)))
    elif case == "tokenize_400_not_json":
        tr = RegimeGateTransport(tokenize_override=(400, b"<html>bad gateway</html>"))
    elif case == "tokenize_200_no_count":
        tr = RegimeGateTransport(tokenize_override=(200, b'{"tokens": [1, 2]}'))
    else:
        tr = RegimeGateTransport(tokenize_raises=ConnectionRefusedError("refused"))
    led = _ledger(tmp_path, tr)
    status, raw, _ = _post(led, OFF_BODY)
    err = json.loads(raw)["error"]
    assert status == 502, (case, status, raw)
    assert err["code"] == ul.ERROR_CODE_TOKENIZE_FAILED and err["type"] == "ledger_error"
    assert ul.RELAY_VIA_KEY not in err
    assert tr.chat_hits == 0
    r = _rows(led.writer.path)[0]
    assert r["error_code"] == ul.ERROR_CODE_TOKENIZE_FAILED and r["status"] is None


def test_a_normal_body_is_tokenized_ledgered_and_forwarded_as_before(tmp_path):
    tr = RegimeGateTransport()
    led = _ledger(tmp_path, tr)
    status, raw, _ = _post(led, ON_BODY)
    assert status == 200 and json.loads(raw)["choices"][0]["message"]["content"] == "4"
    assert [p for _, p, _, _ in tr.calls] == [ul.TOKENIZE_PATH, ul.CHAT_PATH]
    assert tr.chat_hits == 1 and tr.calls[1][2] == json.dumps(ON_BODY).encode()
    r = _rows(led.writer.path)[0]
    assert r["rejected"] is False and r["prompt_tokens_pre"] == 5 and r["status"] == 200
    assert r["usage_prompt_tokens"] == 5 and r["tokenize_mismatch"] is False and r["error_code"] is None
    assert led.stats() == {**led.stats(), "n_forwarded": 1, "n_blocked": 0}


def test_relayed_tokenize_refusal_helper_reads_the_wire_shape_only():
    good = _typed_refusal_bytes(px.THINKING_REGIME_MISMATCH, "m")
    out = ul.relayed_tokenize_refusal(400, good)
    assert out is not None
    assert json.loads(out)["error"] == {"message": "m", "type": "invalid_request_error",
                                        "code": px.THINKING_REGIME_MISMATCH,
                                        ul.RELAY_VIA_KEY: ul.RELAY_VIA_LEDGER_TOKENIZE}
    # the proxy's `extra` facts on a refusal ride along untouched
    extra = json.dumps({"error": {"message": "m", "type": "invalid_request_error",
                                  "code": px.THINKING_REGIME_MISMATCH, "worker_error_type": "x"}}).encode()
    assert json.loads(ul.relayed_tokenize_refusal(400, extra))["error"]["worker_error_type"] == "x"
    # not a 400 / not typed / not a JSON object / no error object -> None (the 502 path)
    assert ul.relayed_tokenize_refusal(502, good) is None
    assert ul.relayed_tokenize_refusal(400, _typed_refusal_bytes("boom")) is None
    assert ul.relayed_tokenize_refusal(400, b"not json") is None
    assert ul.relayed_tokenize_refusal(400, b'{"error": "string"}') is None
    assert ul.relayed_tokenize_refusal(400, b'{"error": {"code": 7}}') is None
    assert ul.relayed_tokenize_refusal(None, good) is None


# --------------------------------------------------------------------------- #
# 2. the full real chain over loopback aiohttp                                  #
# --------------------------------------------------------------------------- #


class TokBackend(StubBackend):
    """The stub engine with a ``/tokenize``: counts renders; chat hits are
    ``len(self.payloads)`` minus the tokenize count (the stub appends chat
    payloads; tokenize payloads land in ``tokenized``)."""

    def __init__(self):
        super().__init__()
        self.tokenized: list[dict] = []

    async def tokenize(self, payload):
        self.tokenized.append(json.loads(json.dumps(payload)))
        return {"count": 3, "tokens": [1, 2, 3]}


def _on_proxy(tmp_path):
    spec, source, cells = _cells(tmp_path)
    backend = TokBackend()
    proxy = px.RoutingProxy(px.routes_from_cells(cells), backend, base_model=spec.target_model,
                            serving_mode="interleaved", prompted_prefixes=px.prompted_prefix_specs(cells, source),
                            agent_loop_render=_TailRender(), thinking_mode="on",
                            thinking_header=MOVES, lens_serve_declarations={})
    base = next(r for r in proxy._routes.values() if r.cell_id == C.ROUTE_NONE_CELL)
    return proxy, backend, base


@contextlib.contextmanager
def _serve_chain(tmp_path):
    """The real RoutingProxy app and the real ledger app on loopback ports,
    driven by a background event loop so a SYNC client (urllib in
    ProxyClientEngine) can call them from the test thread. Yields
    ``(ledger_url, proxy_url, backend, base_route, ledger)``."""
    from aiohttp import web

    proxy, backend, base = _on_proxy(tmp_path)
    ready = threading.Event()
    stop = threading.Event()
    out: dict = {}

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def main():
            prunner = web.AppRunner(px.make_app(proxy))
            await prunner.setup()
            psite = web.TCPSite(prunner, "127.0.0.1", 0)
            await psite.start()
            pport = prunner.addresses[0][1]
            proxy_url = f"http://127.0.0.1:{pport}/v1"
            led = ul.UsageLedger(proxy_url, tmp_path / C.LEDGER_FILENAME)
            out["probe"] = await led.probe()
            lrunner = web.AppRunner(ul.make_app(led))
            await lrunner.setup()
            lsite = web.TCPSite(lrunner, "127.0.0.1", 0)
            await lsite.start()
            lport = lrunner.addresses[0][1]
            out.update(ledger_url=f"http://127.0.0.1:{lport}/v1", proxy_url=proxy_url, ledger=led)
            ready.set()
            try:
                while not stop.is_set():
                    await asyncio.sleep(0.02)
            finally:
                await lrunner.cleanup()
                await led.aclose()
                led.writer.close()
                await prunner.cleanup()

        try:
            loop.run_until_complete(main())
        except Exception as e:  # pragma: no cover — surfaced to the test thread
            out["error"] = e
            ready.set()
        finally:
            loop.close()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    assert ready.wait(20), "the chain did not come up"
    if "error" in out:
        raise out["error"]
    try:
        yield out["ledger_url"], out["proxy_url"], backend, base, out["ledger"]
    finally:
        stop.set()
        t.join(20)


def _http(url: str, body: dict | None, headers: dict | None = None):
    """A raw sync POST/GET returning ``(status, json)`` — never raises on 4xx/5xx."""
    import urllib.error
    import urllib.request

    data = json.dumps(body).encode() if body is not None else None
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def test_full_chain_relays_the_proxys_typed_refusal_and_forwards_nothing(tmp_path):
    pytest.importorskip("aiohttp")
    with _serve_chain(tmp_path) as (ledger_url, proxy_url, backend, base, led):
        vm = base.virtual_model
        # the RoutingProxy itself (behind the ledger) refuses the OFF render on tokenize AND on chat, same code
        s_tok, j_tok = _http(f"{proxy_url}/tokenize", {**OFF_BODY, "model": vm})
        s_chat, j_chat = _http(f"{proxy_url}/chat/completions", {**OFF_BODY, "model": vm})
        assert (s_tok, j_tok["error"]["code"]) == (400, px.THINKING_REGIME_MISMATCH)
        assert (s_chat, j_chat["error"]["code"]) == (400, px.THINKING_REGIME_MISMATCH)
        assert ul.RELAY_VIA_KEY not in j_tok["error"] and ul.RELAY_VIA_KEY not in j_chat["error"]
        chat_hits_before = len(backend.payloads)
        n_tok_before = len(backend.tokenized)
        # through the LEDGER: the engine's typed 400 with the via marker — never a 502
        status, j = _http(f"{ledger_url}/chat/completions", {**OFF_BODY, "model": vm},
                          {"X-Tinycap-Task": "probe", "X-Tinycap-Sample": "render"})
        assert status == 400, (status, j)
        assert j["error"]["code"] == px.THINKING_REGIME_MISMATCH
        assert j["error"]["type"] == "invalid_request_error"
        assert j["error"][ul.RELAY_VIA_KEY] == ul.RELAY_VIA_LEDGER_TOKENIZE
        assert j["error"]["message"] == j_tok["error"]["message"]      # the proxy's own words, relayed
        # the stub engine saw NO chat call and NO tokenize render (the proxy's gate refused before its backend)
        assert len(backend.payloads) == chat_hits_before
        assert len(backend.tokenized) == n_tok_before
        # a normal (ON-regime) body: tokenized through the proxy, forwarded, served
        status, j = _http(f"{ledger_url}/chat/completions", _chat(vm),
                          {"X-Tinycap-Task": "probe", "X-Tinycap-Sample": "render"})
        assert status == 200 and j["choices"][0]["message"]["content"] == "stub reply"
        assert len(backend.payloads) == chat_hits_before + 1
        assert len(backend.tokenized) == n_tok_before + 1
        rows = _rows(led.writer.path)
    assert [r["error_code"] for r in rows] == [ul.ERROR_CODE_TOKENIZE_FAILED, None]
    assert rows[0]["status"] is None and rows[0]["thinking"] is False
    assert rows[1]["status"] == 200 and rows[1]["prompt_tokens_pre"] == 3


# --------------------------------------------------------------------------- #
# 3. the driver's REAL predicate-(A) probe against the chain                    #
# --------------------------------------------------------------------------- #


def test_predicate_a_render_probe_now_sees_the_engines_typed_refusal_not_a_502(tmp_path):
    """The GSM8K ``none`` child's exact code path (``_render_probe`` through
    ``ProxyClientEngine.chat_completions`` at the ledger URL, the OFF arm
    first). The defect was the 502 in this SystemExit; now the SystemExit
    quotes the RoutingProxy's typed 400. The probe still exits (see the
    module docstring: predicate (A)'s reader has no ON-pinned-standup path
    — a contract change, not this fix's)."""
    pytest.importorskip("aiohttp")
    import served_capability_run as srun

    with _serve_chain(tmp_path) as (ledger_url, _proxy_url, backend, base, _led):
        client = px.ProxyClientEngine(ledger_url, identity=base.serving_identity)
        client.load(require_virtual_models=[base.virtual_model])
        chat_hits_before = len(backend.payloads)
        with pytest.raises(SystemExit) as ei:
            srun._render_probe(client.chat_completions, base.virtual_model, lens=False, header_located=False,
                               system=None, sampling_row=None, family=None)
        msg = str(ei.value)
        assert len(backend.payloads) == chat_hits_before     # nothing forwarded to the engine
    assert msg.startswith(f"[served-cap] {C.REC_THINKING_UNSERVABLE}: predicate (A)'s 'off' render probe"), msg
    assert "returned 400" in msg and px.THINKING_REGIME_MISMATCH in msg, msg
    assert f'"{ul.RELAY_VIA_KEY}": "{ul.RELAY_VIA_LEDGER_TOKENIZE}"' in msg, msg
    # the old face is gone: the status as the producer phrases it (the client's "returned <status>", the typed
    # refusal's "(<status> <code>"), never a bare "502" — the message embeds the chain's loopback URLs, whose
    # ephemeral ports can themselves contain "502" (gates drew :38502 and :65028 on green runs)
    assert "returned 502" not in msg and f"(502 {px.THINKING_REGIME_MISMATCH}" not in msg, msg
    assert ul.ERROR_CODE_TOKENIZE_FAILED not in msg, msg
