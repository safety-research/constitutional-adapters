"""The routing proxy's read-only ``/tokenize`` pass-through: the served
clients resolve vLLM's FLOOR-ONLY context-length rejection ("prompt contains
at least N input tokens") by tokenizing the IDENTICAL messages through the
proxy — same token auth, same virtual-model resolution as chat, NO steering
selector and NO cache salt (nothing is generated or cached), only the
render-shaping keys forwarded, and a backend without the route refuses so
the client files the typed floor_only_unresolvable error instead of
guessing. CPU-only, stub backend (the proxy tests' own scaffolding)."""

from __future__ import annotations

import asyncio
import json

import pytest

from sisteer.capeval import manifest as cm
from sisteer.capeval.serving import proxy as px

from test_petri_proxy import StubBackend, _proxy


class TokenizeBackend(StubBackend):
    """A stub upstream with vLLM's /tokenize: records payloads, answers with a
    count derived from the messages."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.tokenize_payloads = []

    async def tokenize(self, payload):
        self.tokenize_payloads.append(json.loads(json.dumps(payload)))
        n = sum(len(m.get("content") or "") for m in payload["messages"])
        return {"count": n, "max_model_len": 16384, "tokens": list(range(n))}


def _routes_by_path(proxy):
    out = {}
    for r in proxy._routes.values():
        path = cm.parse_serving_identity(r.serving_identity).path
        out.setdefault(path, r)
    return out


def test_tokenize_resolves_the_virtual_model_like_chat_without_steering(tmp_path):
    backend = TokenizeBackend()
    proxy, _, _ = _proxy(tmp_path, backend=backend)
    routes = _routes_by_path(proxy)
    msgs = [{"role": "user", "content": "hello there"}]
    for path, expected_model in (("passthrough", proxy.base_model),
                                 ("lens", proxy.base_model)):
        r = routes[path]
        out = asyncio.run(proxy.handle_tokenize(
            {"model": r.virtual_model, "messages": msgs, "max_tokens": 32,
             "temperature": 0.0, "seed": 1}))
        sent = backend.tokenize_payloads[-1]
        assert sent["model"] == expected_model, path
        assert sent["messages"] == msgs and sent["add_generation_prompt"] is True
        # render-shaping keys only: no sampling knobs, no selector, no salt
        assert set(sent) <= {"model", *px.TOKENIZE_FORWARD_KEYS}
        assert px.UPSTREAM_EXTRA_ARGS_KEY not in sent and "cache_salt" not in sent
        assert "max_tokens" not in sent and "seed" not in sent
        # the client sees its virtual model, and vLLM's count
        assert out["model"] == r.virtual_model and out["count"] == len("hello there")
    if "lora" in routes:
        r = routes["lora"]
        asyncio.run(proxy.handle_tokenize({"model": r.virtual_model, "messages": msgs}))
        assert backend.tokenize_payloads[-1]["model"] == r.cell_id   # the adapter name
    # explicit render knobs pass through
    r = routes["passthrough"]
    asyncio.run(proxy.handle_tokenize({"model": r.virtual_model, "messages": msgs,
                                       "add_generation_prompt": False,
                                       "continue_final_message": True}))
    sent = backend.tokenize_payloads[-1]
    assert sent["add_generation_prompt"] is False and sent["continue_final_message"] is True
    # nothing was generated: chat_completions never ran
    assert backend.payloads == []


def test_tokenize_refusals(tmp_path):
    backend = TokenizeBackend()
    proxy, _, _ = _proxy(tmp_path, backend=backend, api_token="sekrit")
    r = _routes_by_path(proxy)["passthrough"]
    msgs = [{"role": "user", "content": "x"}]
    with pytest.raises(px.ProxyRefusal) as ei:
        asyncio.run(proxy.handle_tokenize({"model": r.virtual_model, "messages": msgs}))
    assert ei.value.http_status == 401
    ok = asyncio.run(proxy.handle_tokenize({"model": r.virtual_model, "messages": msgs},
                                           authorization="Bearer sekrit"))
    assert ok["count"] == 1
    with pytest.raises(px.ProxyRefusal) as ei:
        asyncio.run(proxy.handle_tokenize({"model": "petri__nope", "messages": msgs},
                                          authorization="Bearer sekrit"))
    assert ei.value.http_status == 404
    with pytest.raises(px.ProxyRefusal, match="reserved selector"):
        asyncio.run(proxy.handle_tokenize(
            {"model": r.virtual_model, "messages": msgs, "cache_salt": "x"},
            authorization="Bearer sekrit"))
    with pytest.raises(px.ProxyRefusal, match="messages"):
        asyncio.run(proxy.handle_tokenize({"model": r.virtual_model},
                                          authorization="Bearer sekrit"))
    # a backend without the route: 501, so the client never guesses
    proxy2, _, _ = _proxy(tmp_path, backend=StubBackend())
    r2 = _routes_by_path(proxy2)["passthrough"]
    with pytest.raises(px.ProxyRefusal) as ei:
        asyncio.run(proxy2.handle_tokenize({"model": r2.virtual_model, "messages": msgs}))
    assert ei.value.http_status == 501 and ei.value.code == "tokenize_unsupported"


def test_http_surface_exposes_tokenize_routes(tmp_path):
    pytest.importorskip("aiohttp")
    proxy, _, _ = _proxy(tmp_path, backend=TokenizeBackend())
    app = px.make_app(proxy)
    paths = {(getattr(res, "canonical", None) or str(res)) for res in app.router.resources()}
    assert "/v1/tokenize" in paths and "/tokenize" in paths
    assert "/v1/chat/completions" in paths


def test_client_engine_tokenize_posts_to_the_proxy(monkeypatch):
    client = px.ProxyClientEngine("http://127.0.0.1:9", identity="proxy_vllm+lens+interleaved")
    seen = {}

    def fake_http(url, payload):
        seen["url"] = url; seen["payload"] = payload
        return {"count": 3}
    monkeypatch.setattr(client, "_http_json", fake_http)
    assert client.tokenize({"model": "petri__x", "messages": []}) == {"count": 3}
    assert seen["url"] == "http://127.0.0.1:9/v1/tokenize"


def test_http_backend_tokenize_targets_the_server_root(monkeypatch):
    """vLLM serves /tokenize at the server ROOT (like the RPC route), not
    under /v1: the backend derives the root from its canonical /v1 base."""
    b = px.HTTPBackend("http://127.0.0.1:8000/v1", api_key="k")
    assert b.upstream_base_url.endswith("/v1")
    src = open(px.__file__).read()
    assert 'f"{root}/tokenize"' in src and 'root = self.upstream_base_url[:-len("/v1")]' in src
