"""The CACHE POLICY of native routes: the attested ``position_mode`` drives the
KV-cache salt regime — ``all``
(and a plain LoRA) stays CELL-STABLE (today's), every RESTRICTED mode
(``response`` / ``prompt_last`` / ``prefill_all``) is minted REQUEST-UNIQUE (the
lens's ``conditional`` / ``never`` rule transported); the stamp triple carries the
regime and the row carries the mode; the salt-discipline guard refuses a reuse
of a request-unique native salt; ``expected_stamp_regimes(lens=False,
position_mode=...)`` derives the same regime (the contract's rule is the proxy's
— that file is not on this tree yet; the test xfails with the exact patch until
it lands). CPU-only, a stub backend, no server."""

from __future__ import annotations

import inspect
import uuid

import pytest

from sisteer.capeval.serving import native_residual as nr
from sisteer.capeval.serving import proxy as px
from sisteer.capeval.tinycap import contracts as C

from test_petri_proxy import StubBackend, _ask, _chat  # noqa: E402
from test_restricted_route_requests import ALL_CELL, LORA_CELL, RESP_CELL, V, _proxy, _routes  # noqa: E402

RESTRICTED = tuple(m for m in nr.POSITION_MODES if m != "all")


def test_salt_regime_for_is_driven_by_the_native_position_mode():
    for mode in RESTRICTED:
        assert px.RoutingProxy.salt_regime_for(lens=False, position_mode=mode) == px.SALT_REGIME_REQUEST_UNIQUE
    assert px.RoutingProxy.salt_regime_for(lens=False, position_mode="all") == px.SALT_REGIME_CELL_STABLE
    assert px.RoutingProxy.salt_regime_for(lens=False, position_mode=None) == px.SALT_REGIME_CELL_STABLE
    assert px.RoutingProxy.salt_regime_for(lens=False) == px.SALT_REGIME_CELL_STABLE     # today's call, unchanged
    # the lens branch does not read the mode
    assert px.RoutingProxy.salt_regime_for(lens=True, position_mode="all") == px.SALT_REGIME_REQUEST_UNIQUE
    assert px.RoutingProxy.salt_regime_for(
        lens=True, resumed_prefill_policy=px.RESUMED_PREFILL_ALWAYS, position_mode="response") == px.SALT_REGIME_CELL_STABLE


def test_cache_salt_for_mints_request_unique_salts_on_a_restricted_mode_only():
    r = next(x for x in _routes() if x.cell_id == RESP_CELL)
    stable = px.RoutingProxy.cache_salt_for(r, lens=False)
    assert stable == f"petri|{RESP_CELL}|{r.intervention_hash}"
    assert px.RoutingProxy.cache_salt_for(r, lens=False, position_mode="all") == stable
    for mode in RESTRICTED:
        a = px.RoutingProxy.cache_salt_for(r, lens=False, position_mode=mode)
        b = px.RoutingProxy.cache_salt_for(r, lens=False, position_mode=mode)
        assert a != b and a.startswith(f"{stable}|{mode}|") and b.startswith(f"{stable}|{mode}|")
        assert len(a.rsplit("|", 1)[1]) == 32                      # a uuid4 hex: a namespace of one
    # the organism overlay rides in the stable part as before
    assert px.RoutingProxy.cache_salt_for(r, lens=False, organism="org1", position_mode="response").startswith(
        f"{stable}|org=org1|response|")


def test_the_proxy_stamps_the_regime_and_the_mode_per_route():
    proxy, backend = _proxy()
    # the RESTRICTED route: two identical requests, two salts, request-unique, mode on the row + stamp
    outs = [_ask(proxy, _chat(V[RESP_CELL], "same")) for _ in range(2)]
    salts = [p["cache_salt"] for p in backend.payloads[-2:]]
    assert salts[0] != salts[1] and all(f"|{RESP_CELL}|" in s and "|response|" in s for s in salts)
    for out in outs:
        assert out["petri"][px.LEDGER_KEY_SALT_REGIME] == px.SALT_REGIME_REQUEST_UNIQUE
        assert out["petri"][px.LEDGER_KEY_RESUMED_POLICY] == px.POLICY_NOT_CONSULTED   # no lens policy read
        assert out["petri"][px.LEDGER_KEY_POSITION_MODE] == "response"
    rows = list(proxy.request_log)[-2:]
    assert all(r[px.LEDGER_KEY_SALT_REGIME] == px.SALT_REGIME_REQUEST_UNIQUE for r in rows)
    assert all(r[px.LEDGER_KEY_POSITION_MODE] == "response" for r in rows)
    # the ALL-positions route: one salt, cell-stable — today's LoRA behaviour
    outs = [_ask(proxy, _chat(V[ALL_CELL], "same")) for _ in range(2)]
    salts = [p["cache_salt"] for p in backend.payloads[-2:]]
    assert salts[0] == salts[1] == f"petri|{ALL_CELL}|{'cc' * 8}"
    for out in outs:
        assert out["petri"][px.LEDGER_KEY_SALT_REGIME] == px.SALT_REGIME_CELL_STABLE
        assert out["petri"][px.LEDGER_KEY_POSITION_MODE] == "all"
    # the LoRA route and the base route: cell-stable, no mode
    for vm, cell in ((V[LORA_CELL], LORA_CELL), (V["none"], "none")):
        out = _ask(proxy, _chat(vm))
        assert out["petri"][px.LEDGER_KEY_SALT_REGIME] == px.SALT_REGIME_CELL_STABLE
        assert out["petri"][px.LEDGER_KEY_POSITION_MODE] is None
    # every restricted mode is request-unique through the proxy
    for mode in RESTRICTED:
        p, b = _proxy(resp_mode=mode)
        _ask(p, _chat(V[RESP_CELL])); _ask(p, _chat(V[RESP_CELL]))
        assert b.payloads[0]["cache_salt"] != b.payloads[1]["cache_salt"]
        assert p.request_log[-1][px.LEDGER_KEY_SALT_REGIME] == px.SALT_REGIME_REQUEST_UNIQUE


def test_the_salt_decision_carries_the_mode():
    proxy, _ = _proxy()
    by_v = {r.virtual_model: r for r in _routes()}
    d = proxy._salt_decision(by_v[V[RESP_CELL]], lens=False)
    assert isinstance(d, px.SaltDecision) and d.position_mode == "response"
    assert d.regime == px.SALT_REGIME_REQUEST_UNIQUE and d.policy == px.POLICY_NOT_CONSULTED and d.lens is False
    d = proxy._salt_decision(by_v[V[ALL_CELL]], lens=False)
    assert d.position_mode == "all" and d.regime == px.SALT_REGIME_CELL_STABLE
    assert proxy._salt_decision(by_v[V[LORA_CELL]], lens=False).position_mode is None


def test_the_guard_refuses_a_reuse_of_a_request_unique_native_salt(monkeypatch):
    """A request-unique salt is a namespace of one by construction; a repeat (a
    uuid collision, here forced) is a minting bug and the resume it would allow
    is the masked-write wrong-serve — 409 at the proxy, never the engine."""
    proxy, backend = _proxy()
    fixed = uuid.UUID(int=7)
    monkeypatch.setattr(px.uuid, "uuid4", lambda: fixed)
    _ask(proxy, _chat(V[RESP_CELL]))
    with pytest.raises(px.ProxyRefusal) as ei:
        _ask(proxy, _chat(V[RESP_CELL]))
    assert ei.value.http_status == 409 and ei.value.code == "cache_salt_collision"
    assert "request-unique" in str(ei.value) and "D12" in str(ei.value)
    assert len(backend.payloads) == 1                        # the repeat never went upstream
    # the cell-stable routes' legitimate repeats are untouched by the new rule
    _ask(proxy, _chat(V[ALL_CELL])); _ask(proxy, _chat(V[ALL_CELL]))
    _ask(proxy, _chat(V[LORA_CELL])); _ask(proxy, _chat(V[LORA_CELL]))
    assert len(backend.payloads) == 5


def test_expected_stamp_regimes_gains_the_native_position_mode():
    """The contract's derivation (``contracts.expected_stamp_regimes``) must read
    the mode through the proxy's rule: a non-lens leg at a restricted mode is
    ``request-unique`` / ``not-consulted``; at ``all`` or with no mode, today's
    ``cell-stable``. ``tinycap/contracts.py`` is lane R's file — until its
    ``position_mode`` keyword lands this xfails, naming the patch."""
    sig = inspect.signature(C.expected_stamp_regimes)
    if "position_mode" not in sig.parameters:
        pytest.xfail(
            "contracts.expected_stamp_regimes has no position_mode keyword yet (lane R's file). Patch: "
            "add `position_mode: Optional[str] = None` to the signature and, in the `if not lens:` branch, "
            "return StampRegimes(RoutingProxy.salt_regime_for(lens=False, position_mode=position_mode), "
            "POLICY_NOT_CONSULTED, render) instead of the literal SALT_REGIME_CELL_STABLE")
    for mode in RESTRICTED:
        got = C.expected_stamp_regimes(lens=False, arm=C.THINKING_OFF, policy=C.POLICY_NOT_CONSULTED,
                                       position_mode=mode)
        assert tuple(got)[:2] == (px.SALT_REGIME_REQUEST_UNIQUE, px.POLICY_NOT_CONSULTED), mode
    for mode in ("all", None):
        got = C.expected_stamp_regimes(lens=False, arm=C.THINKING_OFF, policy=C.POLICY_NOT_CONSULTED,
                                       position_mode=mode)
        assert tuple(got)[:2] == (px.SALT_REGIME_CELL_STABLE, px.POLICY_NOT_CONSULTED), mode
    # the call every existing consumer makes is unchanged
    assert tuple(C.expected_stamp_regimes(lens=False, arm=C.THINKING_OFF, policy=C.POLICY_NOT_CONSULTED))[:2] == (
        px.SALT_REGIME_CELL_STABLE, px.POLICY_NOT_CONSULTED)


def test_a_bare_backend_stub_still_sees_every_salt_regime_field():
    """The stamp triple's field names are the proxy's own literals — pinned so a
    reader of the audit stream (the gate) keys on the same spelling."""
    assert px.LEDGER_KEY_SALT_REGIME == "cache_salt_regime"
    assert px.LEDGER_KEY_RESUMED_POLICY == "resumed_prefill_policy"
    assert px.LEDGER_KEY_POSITION_MODE == "position_mode"
    assert px.LEDGER_KEY_FAILURE_CLAIM == "request_failure_claim"
    proxy, _ = _proxy(StubBackend())
    out = _ask(proxy, _chat(V[RESP_CELL]))
    assert {px.LEDGER_KEY_SALT_REGIME, px.LEDGER_KEY_RESUMED_POLICY, px.LEDGER_KEY_RENDER_REGIME,
            px.LEDGER_KEY_POSITION_MODE, px.LEDGER_KEY_FAILURE_CLAIM} <= set(out["petri"])
