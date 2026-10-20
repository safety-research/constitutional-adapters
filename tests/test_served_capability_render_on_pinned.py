"""Predicate (A) on a standup ATTESTED thinking-ON.

THE DEFECT (behind F-AC). The GSM8K standups are pinned thinking-ON at the routing proxy: the proxy refuses an OFF
render BY DESIGN — on chat (``RoutingProxy._validate_chat_template_kwargs``, typed ``400 thinking_regime_mismatch``)
and on tokenize (``handle_tokenize``; the usage ledger relays it as the same 400 plus ``via: ledger_tokenize``,
fb2). ``served_capability_run._render_probe`` ran the OFF arm FIRST and raised ``SystemExit THINKING_UNSERVABLE`` on
ANY ``ProxyError``, and ``contracts.render_certified`` required ``probe_regimes[off] == thinking_off`` with the OFF
header ids measured live — so predicate (A) was structurally unpassable on an ON-pinned standup, and
``measure_thinking_header_by_arm`` slid onto the local tokenizer under a broad ``except Exception`` without saying so.

THE RULING. (1) The acceptance needs the BOUND standup record attesting
``thinking_mode: on`` (``endpoint.json``, the key ``seedrep_served.readiness_problems`` holds the row to) AND the exact
typed ``400 thinking_regime_mismatch`` to the deliberate OFF probe; a 502, an untyped 400, another code, a foreign
``via`` all still exit ``THINKING_UNSERVABLE``. (2) The live ON header, the ON regime stamp and the think shape stay
MANDATORY, measured through the proxy. (3) The OFF header ids are a LOCAL REFERENCE with PER-ARM provenance — the
records never label the mix wholly ``served_tokenize``. (4) The template comparison and the object / lens guards
are untouched.

Every fixture is synthetic (placeholder vmodels, the proxy tests' canonical cell set, invented header ids off the
family fact; no recorded data). The chain is REAL: a real ``RoutingProxy`` (``thinking_mode`` on or off) in front of
a stub engine, the real ``UsageLedger`` in front of the proxy, the driver's real ``ProxyClientEngine`` at the ledger
URL, the driver's REAL ``_render_probe`` / ``measure_thinking_header_by_arm`` / ``_render_certification`` and the
real ``contracts.render_certified``.
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

import served_capability_run as srun  # noqa: E402
from test_petri_proxy import StubBackend, _TailRender  # noqa: E402
from test_thinking_on_proxy import MOVES, _cells  # noqa: E402

FAMILY = "qwen3_5"
OFF_IDS = list(C.FAMILY_RENDER_FACTS[FAMILY].off_ids)
ON_IDS = list(C.FAMILY_RENDER_FACTS[FAMILY].on_ids)
PREFIX = [1, 2, 3]                     # the rendered user turn, both arms (synthetic)
THINK_SHAPE_REPLY = "17 + 26 = 43\n</think>\n43"     # the family's implicit-open shape: closed, no stray open
ON = C.THINKING_ON
OFF = C.THINKING_OFF


class RenderBackend(StubBackend):
    """The stub engine with a ``/tokenize`` that RENDERS: the user turn, plus the family's per-arm assistant header
    when ``add_generation_prompt`` is set (ON ids under ``enable_thinking`` true / absent, OFF ids under false).
    Its chat reply carries the family's think shape so the ON draw certifies; chat hits are ``len(self.payloads)``."""

    def __init__(self):
        super().__init__()
        self.tokenized: list[dict] = []

    async def tokenize(self, payload):
        self.tokenized.append(json.loads(json.dumps(payload)))
        ids = list(PREFIX)
        if payload.get("add_generation_prompt"):
            enable = (payload.get(C.CHAT_TEMPLATE_KWARGS_KEY) or {}).get(C.ENABLE_THINKING_KEY)
            ids += OFF_IDS if enable is False else ON_IDS
        return {"count": len(ids), "tokens": ids}

    async def chat_completions(self, payload):
        resp = await super().chat_completions(payload)
        resp["choices"][0]["message"]["content"] = THINK_SHAPE_REPLY
        return resp


class LocalTok:
    """A LOCAL tokenizer rendering the same two headers (the reference the OFF arm falls back to under the rule)."""

    chat_template = "{{ synthetic template }}"

    def apply_chat_template(self, messages, *, add_generation_prompt=False, tokenize=True, enable_thinking=False, **kw):
        ids = list(PREFIX)
        if add_generation_prompt:
            ids += ON_IDS if enable_thinking else OFF_IDS
        return ids


def _proxy(tmp_path, *, thinking_mode: str):
    spec, source, cells = _cells(tmp_path)
    backend = RenderBackend()
    proxy = px.RoutingProxy(px.routes_from_cells(cells), backend, base_model=spec.target_model,
                            serving_mode="interleaved", prompted_prefixes=px.prompted_prefix_specs(cells, source),
                            agent_loop_render=_TailRender(), thinking_mode=thinking_mode,
                            thinking_header=MOVES, lens_serve_declarations={})
    base = next(r for r in proxy._routes.values() if r.cell_id == C.ROUTE_NONE_CELL)
    return proxy, backend, base


@contextlib.contextmanager
def _serve_chain(tmp_path, *, thinking_mode: str = "on"):
    """The real RoutingProxy app and the real ledger app on loopback ports, driven by a background event loop so the
    driver's SYNC client can call them. Yields ``(ledger_url, backend, base_route)``."""
    from aiohttp import web

    proxy, backend, base = _proxy(tmp_path, thinking_mode=thinking_mode)
    ready, stop, out = threading.Event(), threading.Event(), {}

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def main():
            prunner = web.AppRunner(px.make_app(proxy))
            await prunner.setup()
            psite = web.TCPSite(prunner, "127.0.0.1", 0)
            await psite.start()
            proxy_url = f"http://127.0.0.1:{prunner.addresses[0][1]}/v1"
            led = ul.UsageLedger(proxy_url, tmp_path / C.LEDGER_FILENAME)
            await led.probe()
            lrunner = web.AppRunner(ul.make_app(led))
            await lrunner.setup()
            lsite = web.TCPSite(lrunner, "127.0.0.1", 0)
            await lsite.start()
            out["ledger_url"] = f"http://127.0.0.1:{lrunner.addresses[0][1]}/v1"
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
        yield out["ledger_url"], backend, base
    finally:
        stop.set()
        t.join(20)


def _client(ledger_url, base):
    client = px.ProxyClientEngine(ledger_url, identity=base.serving_identity)
    client.load(require_virtual_models=[base.virtual_model])
    return client


def _probe(client, base, **kw):
    return srun._render_probe(client.chat_completions, base.virtual_model, lens=False, header_located=False,
                              system=None, sampling_row=None, family=FAMILY, **kw)


def _header(client, base, **kw):
    return srun.measure_thinking_header_by_arm(
        {C.GSM8K_STANDARD_TASK: ON}, lens=False, tok=LocalTok(), tokenize=client.tokenize,
        vmodel=base.virtual_model, served_template_hash="ab" * 8, allow_out_of_support=True,
        measure_both_arms=True, header_located=False, **kw)


# --------------------------------------------------------------------------- #
# the constants the rule is keyed on                                            #
# --------------------------------------------------------------------------- #


def test_the_rule_is_keyed_on_the_attested_key_and_the_exact_typed_refusal():
    # the key the driver reads is the standup record's own (mt_serving_up.thinking_record writes it; seedrep_served
    # .readiness_problems holds the row to it) — the proxy's constant, never a literal of this lane's
    assert C.STANDUP_THINKING_MODE_KEY == px.THINKING_MODE_KEY == "thinking_mode"
    assert C.OFF_ARM_REFUSAL_CODE == px.THINKING_REGIME_MISMATCH and C.OFF_ARM_REFUSAL_STATUS == 400
    # the ledger's relay marker (fb2), pinned equal without importing the ledger into contracts
    assert C.OFF_ARM_REFUSAL_VIA_KEY == ul.RELAY_VIA_KEY and C.OFF_ARM_REFUSAL_VIA_LEDGER == ul.RELAY_VIA_LEDGER_TOKENIZE
    assert C.OFF_ARM_REFUSAL_CODE in ul.RELAYED_TOKENIZE_REFUSAL_CODES
    assert C.PROBE_REGIME_REFUSED_REGIME_MISMATCH == "refused:thinking_regime_mismatch"
    assert C.PROBE_REGIME_REFUSED_REGIME_MISMATCH not in C.RENDER_REGIME_LITERALS      # never a served regime
    assert srun.HEADER_SOURCE_SERVED == C.RENDER_SOURCE_SERVED_TOKENIZE == "served_tokenize"
    assert srun.HEADER_SOURCE_LOCAL == C.RENDER_SOURCE_LOCAL_TOKENIZER == "local_tokenizer"
    assert srun.HEADER_SOURCE_SERVED_ON_LOCAL_REFERENCE_OFF != srun.HEADER_SOURCE_SERVED


def test_off_arm_refusal_evidence_reads_the_wire_shape_and_admits_exactly_one_refusal():
    err = {"message": "m", "type": "invalid_request_error", "code": px.THINKING_REGIME_MISMATCH}
    direct = C.off_arm_refusal_evidence(400, err)
    assert direct == {"status": 400, "code": px.THINKING_REGIME_MISMATCH, "via": None, "message": "m", "error": err}
    relayed = C.off_arm_refusal_evidence(400, {**err, "via": "ledger_tokenize"})
    assert relayed["via"] == "ledger_tokenize" and relayed["error"]["via"] == "ledger_tokenize"
    # anything else is NOT the evidence: a 502, an untyped 400, another typed code, a foreign via, no error object
    assert C.off_arm_refusal_evidence(502, {**err, "code": ul.ERROR_CODE_TOKENIZE_FAILED}) is None
    assert C.off_arm_refusal_evidence(400, {"message": "nope", "code": "boom"}) is None
    assert C.off_arm_refusal_evidence(400, {**err, "code": px.THINKING_KWARGS_INERT}) is None
    assert C.off_arm_refusal_evidence(400, {**err, "via": "somewhere_else"}) is None
    assert C.off_arm_refusal_evidence(400, "not an object") is None
    assert C.off_arm_refusal_evidence(True, err) is None and C.off_arm_refusal_evidence(None, err) is None
    # the acceptance: the typed refusal AND an attested 'on'; absent / off / anything else refuses BY NAME
    assert C.off_arm_refusal_accepted(direct, standup_thinking_mode="on") == (True, None)
    ok, why = C.off_arm_refusal_accepted(direct, standup_thinking_mode=None)
    assert not ok and "carries no thinking_mode" in why
    ok, why = C.off_arm_refusal_accepted(direct, standup_thinking_mode="off")
    assert not ok and "attests thinking_mode 'off', not 'on'" in why and "must be MEASURED" in why
    ok, why = C.off_arm_refusal_accepted(None, standup_thinking_mode="on")
    assert not ok and "not the typed 400 thinking_regime_mismatch" in why
    ok, _ = C.off_arm_refusal_accepted({"status": 502, "error": err}, standup_thinking_mode="on")
    assert not ok


# --------------------------------------------------------------------------- #
# (a) known-bad first: the real chain, the real probe                           #
# --------------------------------------------------------------------------- #


def test_known_bad_the_probe_with_no_attested_mode_still_exits_with_fb2s_face_naming_the_absent_key(tmp_path):
    """The known-bad face, pinned: with NO attested ``thinking_mode`` the OFF probe on the ON-pinned chain exits
    ``THINKING_UNSERVABLE`` quoting the typed 400 + ``via`` — and now also names the absent key (the OFF arm must be
    measured on a standup not attested ON)."""
    pytest.importorskip("aiohttp")
    with _serve_chain(tmp_path) as (ledger_url, backend, base):
        client = _client(ledger_url, base)
        hits = len(backend.payloads)
        with pytest.raises(SystemExit) as ei:
            _probe(client, base)                                   # standup_thinking_mode absent
        assert len(backend.payloads) == hits                       # nothing forwarded
    msg = str(ei.value)
    assert msg.startswith(f"[served-cap] {C.REC_THINKING_UNSERVABLE}: predicate (A)'s 'off' render probe"), msg
    assert "returned 400" in msg and px.THINKING_REGIME_MISMATCH in msg and '"via": "ledger_tokenize"' in msg, msg
    assert "carries no thinking_mode" in msg, msg
    # the status as the producer phrases it (the typed refusal's "(400 <code>, via <via>)" and the client's
    # "returned <status>"), never a bare "502": the message embeds the probe's proxy URL, whose ephemeral port can
    # itself contain "502" (e.g. :50296) on a green run
    assert f"(400 {px.THINKING_REGIME_MISMATCH}, via {ul.RELAY_VIA_LEDGER_TOKENIZE})" in msg, msg
    assert "returned 502" not in msg and f"(502 {px.THINKING_REGIME_MISMATCH}" not in msg, msg


def test_on_an_attested_on_standup_the_typed_off_refusal_is_the_off_arm_evidence_and_the_on_arm_is_measured(tmp_path):
    """Known-bad at 4ba4fd2e6 (``_render_probe`` took no ``standup_thinking_mode``; ``render_certified`` required the
    OFF regime stamp). Fixed: the OFF probe's typed refusal is recorded (``probe_regimes[off]``, the body with
    ``via``), the ON arm is drawn through the proxy (one stub chat hit, none for OFF), the OFF header is the local
    REFERENCE with per-arm provenance, the ON header live, and the certification PASSES recording every acceptance."""
    pytest.importorskip("aiohttp")
    with _serve_chain(tmp_path) as (ledger_url, backend, base):
        client = _client(ledger_url, base)
        hits, toks = len(backend.payloads), len(backend.tokenized)
        probe = _probe(client, base, standup_thinking_mode="on")
        # the OFF arm: refused by design, recorded — no chat reached the engine for it
        assert probe["regimes"][OFF] == C.PROBE_REGIME_REFUSED_REGIME_MISMATCH
        refusal = probe["off_arm_refusal"]
        assert refusal["status"] == 400 and refusal["code"] == px.THINKING_REGIME_MISMATCH
        assert refusal["via"] == ul.RELAY_VIA_LEDGER_TOKENIZE and "thinking-ON standup" in refusal["message"]
        assert probe["responses"][OFF]["refusal"] == refusal and probe["responses"][OFF]["content"] == ""
        assert probe["responses"][OFF]["render_regime"] == C.PROBE_REGIME_REFUSED_REGIME_MISMATCH
        # the ON arm: MEASURED through the proxy — exactly one chat hit, the proxy's own stamp, the think shape
        assert len(backend.payloads) == hits + 1
        on_body = backend.payloads[-1]
        assert on_body[C.CHAT_TEMPLATE_KWARGS_KEY] == {C.ENABLE_THINKING_KEY: True}
        assert probe["regimes"][ON] == C.RENDER_REGIME_THINKING_ON
        assert probe["responses"][ON]["content"] == THINK_SHAPE_REPLY
        assert [a["class"] for a in probe["on_attempts"]] == [srun.PROBE_CLASS_THINK_SHAPE]
        # the header: ON live through /v1/tokenize (two renders), OFF refused there too -> the local REFERENCE
        block, err = _header(client, base, standup_thinking_mode="on")
        assert err is None, err
        assert block["header_ids_by_arm"] == {OFF: OFF_IDS, ON: ON_IDS}
        assert block["source"] == srun.HEADER_SOURCE_SERVED_ON_LOCAL_REFERENCE_OFF     # never wholly served
        assert block[srun.HEADER_KEY_SOURCE_BY_ARM] == {OFF: srun.HEADER_SOURCE_LOCAL, ON: srun.HEADER_SOURCE_SERVED}
        assert block[srun.HEADER_KEY_OFF_HEADER_SOURCE] == srun.HEADER_SOURCE_LOCAL == "local_tokenizer"
        assert block[srun.HEADER_KEY_OFF_HEADER_ROLE] == C.OFF_HEADER_ROLE_LOCAL_REFERENCE
        assert "refused the 'off' arm's render by design (400 thinking_regime_mismatch)" in block["fallback_reason"]
        assert block[srun.HEADER_KEY_OFF_HEADER_FALLBACK_REASON] == block["fallback_reason"]
        assert block[srun.HEADER_KEY_OFF_ARM_REFUSAL]["code"] == px.THINKING_REGIME_MISMATCH
        assert block[srun.HEADER_KEY_OFF_ARM_REFUSAL]["via"] is None       # the tokenize refusal passes the ledger through
        assert block[srun.HEADER_KEY_OFF_ARM_ACCEPTED_UNDER] == C.OFF_ARM_ACCEPTED_UNDER_ON_PINNED
        assert block["standup_thinking_mode"] == "on" and block["header_moves"] is True and block["verified"] is True
        assert block["local_chat_template_hash"] is not None                # the reference's template is recorded
        assert block["template_comparison"]["required"] is False            # a non-lens route: guard (4) untouched
        # the header-probe renders that reached the ENGINE (the one-turn probe conversation): the ON arm's two
        # (add_generation_prompt True / False) and none for OFF — the proxy's gate refused it before its backend
        header_renders = [t for t in backend.tokenized[toks:] if t["messages"] == list(srun.HEADER_PROBE_MESSAGES)]
        assert sorted(bool(t["add_generation_prompt"]) for t in header_renders) == [False, True]
        assert all(t[C.CHAT_TEMPLATE_KWARGS_KEY] == {C.ENABLE_THINKING_KEY: True} for t in header_renders)
        assert srun._header_measured_live(block) is True
        # the certification: PASS, every acceptance recorded, the mix labelled per arm
        record, error = srun._render_certification(
            probe, family=FAMILY, header_ids_by_arm=block["header_ids_by_arm"],
            measured_live=(block["source"] == srun.HEADER_SOURCE_SERVED),
            header_source_by_arm=block[srun.HEADER_KEY_SOURCE_BY_ARM], standup_thinking_mode="on")
    assert error is None, error
    assert record["render_certified"] is True and record[C.OOS_EVIDENCE_HEADER_CERTIFIED] is True
    assert record["render_regime_certified"] is True and record["think_shape_certified"] is True
    assert record["certification_problems"] == [] and record[srun.CERTIFYING_DRAW_CLASS_FIELD] == srun.PROBE_CLASS_THINK_SHAPE
    assert record["render_source"] == C.RENDER_SOURCE_SERVED_ON_LOCAL_REFERENCE_OFF
    assert record["render_source_by_arm"] == {OFF: "local_tokenizer", ON: "served_tokenize"}
    assert record["off_arm_evidence"] == C.OFF_ARM_EVIDENCE_REFUSED == "refused:thinking_regime_mismatch"
    assert record["off_arm_refusal"]["code"] == px.THINKING_REGIME_MISMATCH and record["off_arm_refusal"]["via"] == "ledger_tokenize"
    assert record["off_header_source"] == "local_tokenizer" and record["off_header_role"] == "local_reference"
    assert record["off_arm_accepted_under"] == C.OFF_ARM_ACCEPTED_UNDER_ON_PINNED
    assert record["standup_thinking_mode"] == "on"
    assert record["measured_header_ids_by_arm"] == {OFF: OFF_IDS, ON: ON_IDS}


# --------------------------------------------------------------------------- #
# (b) the same chain, the record attesting OFF or nothing: refused by name      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode, clause", [("off", "attests thinking_mode 'off', not 'on'"),
                                          (None, "carries no thinking_mode")])
def test_a_standup_not_attested_on_still_needs_the_measured_off_arm(tmp_path, mode, clause):
    pytest.importorskip("aiohttp")
    with _serve_chain(tmp_path) as (ledger_url, backend, base):
        client = _client(ledger_url, base)
        hits = len(backend.payloads)
        with pytest.raises(SystemExit) as ei:
            _probe(client, base, standup_thinking_mode=mode)
        assert len(backend.payloads) == hits
        msg = str(ei.value)
        assert msg.startswith(f"[served-cap] {C.REC_THINKING_UNSERVABLE}: predicate (A)'s 'off' render probe")
        assert "as a thinking-ON standup's OFF render (400 thinking_regime_mismatch, via ledger_tokenize)" in msg
        assert clause in msg and "must be MEASURED" in msg and "never worked around" in msg
        # the header measurement refuses the same way — no local tokenizer is substituted
        block, err = _header(client, base, standup_thinking_mode=mode)
        assert isinstance(err, srun.HeaderMeasurementError) and clause in str(err)
        assert "not silently substituted" in str(err) and block["error"] == str(err)
        assert OFF not in block["header_ids_by_arm"] and block[srun.HEADER_KEY_OFF_ARM_ACCEPTED_UNDER] is None
        assert block["verified"] is False and srun._header_measured_live(block) is False


# --------------------------------------------------------------------------- #
# (c) a proxy WITHOUT the ON pin serves the OFF arm: measured as before         #
# --------------------------------------------------------------------------- #


def test_a_proxy_that_serves_the_off_render_is_measured_as_before_with_no_change_to_the_evidence_keys(tmp_path):
    pytest.importorskip("aiohttp")
    with _serve_chain(tmp_path, thinking_mode="off") as (ledger_url, backend, base):
        client = _client(ledger_url, base)
        hits = len(backend.payloads)
        probe = _probe(client, base, standup_thinking_mode="off")
        assert len(backend.payloads) == hits + 2                                   # both arms drawn
        assert probe["regimes"] == {OFF: C.RENDER_REGIME_THINKING_OFF, ON: C.RENDER_REGIME_THINKING_ON}
        assert probe["off_arm_refusal"] is None and "refusal" not in probe["responses"][OFF]
        block, err = _header(client, base, standup_thinking_mode="off")
        assert err is None and block["source"] == srun.HEADER_SOURCE_SERVED and block["fallback_reason"] is None
        assert block["header_ids_by_arm"] == {OFF: OFF_IDS, ON: ON_IDS}
        assert block[srun.HEADER_KEY_SOURCE_BY_ARM] == {OFF: srun.HEADER_SOURCE_SERVED, ON: srun.HEADER_SOURCE_SERVED}
        assert block[srun.HEADER_KEY_OFF_HEADER_SOURCE] == srun.HEADER_SOURCE_SERVED
        assert block[srun.HEADER_KEY_OFF_HEADER_ROLE] == C.OFF_HEADER_ROLE_SERVED
        assert block[srun.HEADER_KEY_OFF_ARM_REFUSAL] is None and block[srun.HEADER_KEY_OFF_ARM_ACCEPTED_UNDER] is None
        assert block["local_chat_template_hash"] is None                            # nothing local was read
        record, error = srun._render_certification(
            probe, family=FAMILY, header_ids_by_arm=block["header_ids_by_arm"],
            measured_live=(block["source"] == srun.HEADER_SOURCE_SERVED),
            header_source_by_arm=block[srun.HEADER_KEY_SOURCE_BY_ARM], standup_thinking_mode="off")
    assert error is None and record["render_certified"] is True
    assert record["render_source"] == "served_tokenize"                             # the original label, unchanged
    assert record["render_source_by_arm"] == {OFF: "served_tokenize", ON: "served_tokenize"}
    assert record["off_arm_evidence"] == C.OFF_ARM_EVIDENCE_MEASURED and record["off_arm_refusal"] is None
    assert record["off_header_source"] == "served_tokenize" and record["off_header_role"] == C.OFF_HEADER_ROLE_SERVED
    assert record["off_arm_accepted_under"] is None
    # every earlier key of the record is still there with its meaning (the validators' RENDER_CERTIFIED_FIELDS)
    for key in ("family", "render_source", C.OOS_EVIDENCE_HEADER_CERTIFIED, "render_regime_certified", "think_shape_certified",
                "render_certified", "expected_header_ids_by_arm", "measured_header_ids_by_arm", "split_mode",
                "evidence_status", "certification_problems"):
        assert key in record


# --------------------------------------------------------------------------- #
# (d) any OTHER ProxyError on the OFF arm still exits, mode 'on' or not         #
# --------------------------------------------------------------------------- #


def _proxy_error(status: int, body: object) -> px.ProxyError:
    raw = body if isinstance(body, str) else json.dumps(body)
    return px.ProxyError(f"proxy http://127.0.0.1:1/v1/chat/completions returned {status}: {raw}")


@pytest.mark.parametrize("case, exc", [
    ("502 ledger_tokenize_failed (the W19 face)", _proxy_error(502, {"error": {"message": "usage ledger: /v1/tokenize failed",
                                                                                "type": "ledger_error",
                                                                                "code": ul.ERROR_CODE_TOKENIZE_FAILED}})),
    ("untyped 400", _proxy_error(400, {"error": {"message": "nope", "type": "invalid_request_error", "code": "boom"}})),
    ("another typed code", _proxy_error(400, {"error": {"message": "inert", "type": "invalid_request_error",
                                                         "code": px.THINKING_KWARGS_INERT}})),
    ("the right code with a foreign via", _proxy_error(400, {"error": {"message": "m", "type": "invalid_request_error",
                                                                        "code": px.THINKING_REGIME_MISMATCH,
                                                                        "via": "somewhere_else"}})),
    ("the right code on the wrong status", _proxy_error(403, {"error": {"message": "m", "type": "invalid_request_error",
                                                                         "code": px.THINKING_REGIME_MISMATCH}})),
    ("a non-JSON body", _proxy_error(400, "<html>bad gateway</html>")),
    ("a transport failure", px.ProxyError("cannot reach the petri proxy at http://127.0.0.1:1/v1/chat/completions: refused")),
])
def test_any_other_proxy_error_on_the_off_arm_still_exits_thinking_unservable(case, exc):
    def chat(body):
        raise exc

    with pytest.raises(SystemExit) as ei:
        srun._render_probe(chat, "petri__base", lens=False, header_located=False, system=None, sampling_row=None,
                           family=FAMILY, standup_thinking_mode="on")
    msg = str(ei.value)
    assert msg.startswith(f"[served-cap] {C.REC_THINKING_UNSERVABLE}: predicate (A)'s 'off' render probe"), (case, msg)
    assert "REFUSED by the proxy (" in msg and "never worked around" in msg, (case, msg)
    assert "is the OFF-arm evidence" not in msg, (case, msg)


def test_the_typed_refusal_on_the_on_arm_is_never_off_arm_evidence():
    """The rule is about the deliberate OFF probe: the same code answering the ON arm is an ordinary refusal."""
    calls = []

    def chat(body):
        calls.append(body)
        if body[C.CHAT_TEMPLATE_KWARGS_KEY][C.ENABLE_THINKING_KEY] is False:
            raise _proxy_error(400, {"error": {"message": "m", "type": "invalid_request_error",
                                               "code": px.THINKING_REGIME_MISMATCH, "via": "ledger_tokenize"}})
        raise _proxy_error(400, {"error": {"message": "m", "type": "invalid_request_error",
                                           "code": px.THINKING_REGIME_MISMATCH}})

    with pytest.raises(SystemExit) as ei:
        srun._render_probe(chat, "petri__base", lens=False, header_located=False, system=None, sampling_row=None,
                           family=FAMILY, standup_thinking_mode="on")
    assert "predicate (A)'s 'on' render probe" in str(ei.value) and len(calls) == 2


# --------------------------------------------------------------------------- #
# (e) the broad silent fallback is gone                                         #
# --------------------------------------------------------------------------- #


def test_a_live_tokenize_failure_outside_the_rule_is_an_error_not_a_silent_local_measurement():
    by_task = {C.GSM8K_STANDARD_TASK: ON}

    def dead(body):
        raise px.ProxyError("cannot reach the petri proxy at http://127.0.0.1:1/v1/tokenize: connection refused")

    block, err = srun.measure_thinking_header_by_arm(by_task, lens=False, tok=LocalTok(), tokenize=dead, vmodel="petri__base",
                                                     measure_both_arms=True, standup_thinking_mode="on")
    assert isinstance(err, srun.HeaderMeasurementError)
    assert "never silently replaced by a local tokenizer" in str(err) and "connection refused" in str(err)
    assert block["header_ids_by_arm"] == {} and block["source"] == srun.HEADER_SOURCE_SERVED and block["error"] == str(err)

    # the ON arm NEVER falls back, whatever the record attests: a typed refusal on the ON arm is an error
    def on_refused(body):
        if (body.get(C.CHAT_TEMPLATE_KWARGS_KEY) or {}).get(C.ENABLE_THINKING_KEY) is False:
            return {"count": 10, "tokens": PREFIX + (OFF_IDS if body.get("add_generation_prompt") else [])}
        raise _proxy_error(400, {"error": {"message": "m", "type": "invalid_request_error", "code": px.THINKING_REGIME_MISMATCH}})

    block, err = srun.measure_thinking_header_by_arm(by_task, lens=False, tok=LocalTok(), tokenize=on_refused,
                                                     vmodel="petri__base", measure_both_arms=True, standup_thinking_mode="on")
    assert isinstance(err, srun.HeaderMeasurementError) and "under the 'on' arm" in str(err)
    assert block["header_ids_by_arm"] == {OFF: OFF_IDS} and block[srun.HEADER_KEY_SOURCE_BY_ARM] == {OFF: srun.HEADER_SOURCE_SERVED, ON: None}

    # the OFF arm under the rule but NO local tokenizer to read the reference off: an error, never an absent arm
    def off_refused(body):
        if (body.get(C.CHAT_TEMPLATE_KWARGS_KEY) or {}).get(C.ENABLE_THINKING_KEY) is False:
            raise _proxy_error(400, {"error": {"message": "m", "type": "invalid_request_error", "code": px.THINKING_REGIME_MISMATCH}})
        return {"count": 8, "tokens": PREFIX + (ON_IDS if body.get("add_generation_prompt") else [])}

    block, err = srun.measure_thinking_header_by_arm(by_task, lens=False, tok=None, tokenize=off_refused, vmodel="petri__base",
                                                     measure_both_arms=True, standup_thinking_mode="on")
    assert isinstance(err, srun.HeaderMeasurementError) and "no local tokenizer" in str(err) and "['off']" in str(err)
    # ... and WITH one, the rule admits the reference (the same shape the chain test pins)
    block, err = srun.measure_thinking_header_by_arm(by_task, lens=False, tok=LocalTok(), tokenize=off_refused, vmodel="petri__base",
                                                     measure_both_arms=True, standup_thinking_mode="on")
    assert err is None and block["source"] == srun.HEADER_SOURCE_SERVED_ON_LOCAL_REFERENCE_OFF
    assert block["header_ids_by_arm"] == {OFF: OFF_IDS, ON: ON_IDS}


def test_the_lens_template_comparison_guard_is_intact_on_the_off_reference():
    """On a LENS route a local OFF reference still needs the local-vs-served template comparison to RUN and
    AGREE — the rule admits the reference's provenance, not its template."""
    from sisteer.models import chat_template_hash

    def off_refused(body):
        if (body.get(C.CHAT_TEMPLATE_KWARGS_KEY) or {}).get(C.ENABLE_THINKING_KEY) is False:
            raise _proxy_error(400, {"error": {"message": "m", "type": "invalid_request_error", "code": px.THINKING_REGIME_MISMATCH}})
        return {"count": 8, "tokens": PREFIX + (ON_IDS if body.get("add_generation_prompt") else [])}

    tok = LocalTok()
    same = chat_template_hash(tok)
    block, err = srun.measure_thinking_header_by_arm({C.GSM8K_STANDARD_TASK: ON}, lens=True, tok=tok, tokenize=off_refused,
                                                     vmodel="petri__x", served_template_hash=same, measure_both_arms=True,
                                                     allow_out_of_support=True, header_located=False, standup_thinking_mode="on")
    assert err is None and block["template_comparison"] == {"required": True, "ran": True, "equal": True, "unavailable": [], "reason": None}
    block, err = srun.measure_thinking_header_by_arm({C.GSM8K_STANDARD_TASK: ON}, lens=True, tok=tok, tokenize=off_refused,
                                                     vmodel="petri__x", served_template_hash="ff" * 8, measure_both_arms=True,
                                                     allow_out_of_support=True, header_located=False, standup_thinking_mode="on")
    assert isinstance(err, srun.HeaderMeasurementError) and "renders measured are not the renders served" in str(err)
    block, err = srun.measure_thinking_header_by_arm({C.GSM8K_STANDARD_TASK: ON}, lens=True, tok=tok, tokenize=off_refused,
                                                     vmodel="petri__x", served_template_hash=None, measure_both_arms=True,
                                                     allow_out_of_support=True, header_located=False, standup_thinking_mode="on")
    assert isinstance(err, srun.HeaderMeasurementError) and "CANNOT BE CHECKED" in str(err)


# --------------------------------------------------------------------------- #
# the certification's own rules (pure)                                          #
# --------------------------------------------------------------------------- #

REFUSAL = {"status": 400, "code": px.THINKING_REGIME_MISMATCH, "via": "ledger_tokenize", "message": "m",
           "error": {"message": "m", "type": "invalid_request_error", "code": px.THINKING_REGIME_MISMATCH, "via": "ledger_tokenize"}}
ON_COMPLETION = {"content": THINK_SHAPE_REPLY, "reasoning": None, "finish_reason": "stop"}
IDS = {OFF: OFF_IDS, ON: ON_IDS}
MIXED = {OFF: "local_tokenizer", ON: "served_tokenize"}
REFUSED_REGIMES = {OFF: C.PROBE_REGIME_REFUSED_REGIME_MISMATCH, ON: C.RENDER_REGIME_THINKING_ON}


def _problems(**kw) -> list[str]:
    with pytest.raises(C.RenderCertificationError) as ei:
        C.render_certified(family=FAMILY, header_ids_by_arm=IDS, on_completion=ON_COMPLETION, **kw)
    ev = ei.value.evidence
    assert ev["render_certified"] is False and ev[C.OOS_EVIDENCE_HEADER_CERTIFIED] is False
    assert ev["off_arm_accepted_under"] is None
    return ev["certification_problems"]


def test_render_certified_accepts_the_mix_only_with_all_three_facts_together():
    rec = C.render_certified(family=FAMILY, header_ids_by_arm=IDS, measured_live=False, probe_regimes=REFUSED_REGIMES,
                             on_completion=ON_COMPLETION, header_source_by_arm=MIXED, standup_thinking_mode="on",
                             off_arm_refusal=REFUSAL)
    assert rec["render_certified"] is True and rec["render_source"] == C.RENDER_SOURCE_SERVED_ON_LOCAL_REFERENCE_OFF
    assert rec["off_header_role"] == "local_reference" and rec["off_arm_refusal"]["via"] == "ledger_tokenize"
    # the record attests OFF (or nothing): the refused regime and the local OFF header BOTH refuse by name
    for mode in ("off", None):
        problems = _problems(measured_live=False, probe_regimes=REFUSED_REGIMES, header_source_by_arm=MIXED,
                             standup_thinking_mode=mode, off_arm_refusal=REFUSAL)
        assert any("not admissible as OFF-arm evidence" in p for p in problems), problems
        assert any("came from a LOCAL tokenizer" in p for p in problems), problems
        assert any("not measured through the live /v1/tokenize" in p for p in problems), problems
    # the regime says refused but no refusal body travelled with it: the label alone is not evidence
    problems = _problems(measured_live=False, probe_regimes=REFUSED_REGIMES, header_source_by_arm=MIXED,
                         standup_thinking_mode="on", off_arm_refusal=None)
    assert any("not the typed 400" in p for p in problems), problems
    # a body that is not the typed refusal (a 502) does not admit it either
    problems = _problems(measured_live=False, probe_regimes=REFUSED_REGIMES, header_source_by_arm=MIXED,
                         standup_thinking_mode="on", off_arm_refusal={**REFUSAL, "status": 502})
    assert any("not the typed 400" in p for p in problems), problems
    # a local OFF header on an attested-ON standup whose OFF probe was NOT refused (it was served thinking_off):
    # the OFF header must then be measured live — the reference is admitted by the refusal, never by the mode alone
    problems = _problems(measured_live=False, probe_regimes={OFF: C.RENDER_REGIME_THINKING_OFF, ON: C.RENDER_REGIME_THINKING_ON},
                         header_source_by_arm=MIXED, standup_thinking_mode="on", off_arm_refusal=None)
    assert any("came from a LOCAL tokenizer" in p and "not the typed 400" in p for p in problems), problems
    # a refusal body recorded while the regime reads thinking_off: the two must agree
    problems = _problems(measured_live=True, probe_regimes={OFF: C.RENDER_REGIME_THINKING_OFF, ON: C.RENDER_REGIME_THINKING_ON},
                         header_source_by_arm={OFF: "served_tokenize", ON: "served_tokenize"}, standup_thinking_mode="on",
                         off_arm_refusal=REFUSAL)
    assert any("the two must agree" in p for p in problems), problems
    # the ON arm local is never admitted, mode 'on' or not
    problems = _problems(measured_live=False, probe_regimes=REFUSED_REGIMES,
                         header_source_by_arm={OFF: "local_tokenizer", ON: "local_tokenizer"}, standup_thinking_mode="on",
                         off_arm_refusal=REFUSAL)
    assert any("the ON header ids were not measured through the live /v1/tokenize" in p for p in problems), problems
    # the ON think shape and regime stay MANDATORY under the rule
    with pytest.raises(C.RenderCertificationError) as ei:
        C.render_certified(family=FAMILY, header_ids_by_arm=IDS, measured_live=False, probe_regimes=REFUSED_REGIMES,
                           on_completion={"content": "43", "reasoning": None, "finish_reason": "stop"},
                           header_source_by_arm=MIXED, standup_thinking_mode="on", off_arm_refusal=REFUSAL)
    assert any("think shape" in p for p in ei.value.evidence["certification_problems"])
    with pytest.raises(C.RenderCertificationError) as ei:
        C.render_certified(family=FAMILY, header_ids_by_arm=IDS, measured_live=False,
                           probe_regimes={OFF: C.PROBE_REGIME_REFUSED_REGIME_MISMATCH, ON: C.RENDER_REGIME_THINKING_OFF},
                           on_completion=ON_COMPLETION, header_source_by_arm=MIXED, standup_thinking_mode="on",
                           off_arm_refusal=REFUSAL)
    assert any("render_regime stamps" in p for p in ei.value.evidence["certification_problems"])


def test_render_certified_legacy_callers_are_byte_identical_in_meaning():
    """No per-arm map, no mode, no refusal: the legacy reading — `measured_live` is the whole measurement's."""
    served = {OFF: C.RENDER_REGIME_THINKING_OFF, ON: C.RENDER_REGIME_THINKING_ON}
    rec = C.render_certified(family=FAMILY, header_ids_by_arm=IDS, measured_live=True, probe_regimes=served,
                             on_completion=ON_COMPLETION)
    assert rec["render_certified"] is True and rec["render_source"] == "served_tokenize"
    assert rec["render_source_by_arm"] == {OFF: "served_tokenize", ON: "served_tokenize"}
    assert rec["off_arm_evidence"] == "measured" and rec["off_header_role"] == "served_measurement"
    assert rec["standup_thinking_mode"] is None and rec["off_arm_refusal"] is None and rec["off_arm_accepted_under"] is None
    with pytest.raises(C.RenderCertificationError) as ei:
        C.render_certified(family=FAMILY, header_ids_by_arm=IDS, measured_live=False, probe_regimes=served,
                           on_completion=ON_COMPLETION)
    ev = ei.value.evidence
    assert ev["render_source"] == "local_tokenizer" and ev["render_source_by_arm"] == {OFF: "local_tokenizer", ON: "local_tokenizer"}
    assert any("a local-tokenizer fallback certifies nothing" in p for p in ev["certification_problems"])


def test_header_measured_live_reads_the_blocks_own_labels():
    assert srun._header_measured_live({"source": srun.HEADER_SOURCE_SERVED}) is True
    assert srun._header_measured_live({"source": srun.HEADER_SOURCE_LOCAL}) is False
    mixed = {"source": srun.HEADER_SOURCE_SERVED_ON_LOCAL_REFERENCE_OFF}
    assert srun._header_measured_live(mixed) is False                       # the mix alone is not live …
    assert srun._header_measured_live({**mixed, srun.HEADER_KEY_OFF_ARM_ACCEPTED_UNDER: C.OFF_ARM_ACCEPTED_UNDER_ON_PINNED}) is True
