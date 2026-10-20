"""The all-routes bearer-auth middleware
(sisteer/capeval/serving/vllm_auth.py) — CPU-only, pure ASGI.

The threat it closes: the pinned vLLM 0.25.1 guards only /v1|/v2|
/inference with --api-key, so the VLLM_SERVER_DEV_MODE routes — above
all POST /collective_rpc, arbitrary worker RPC by method name — are
open to every same-host tenant for the server's lifetime. This class
rides vLLM's supported --middleware hook and closes exactly that gap
with the same per-launch token.
"""

import asyncio
import secrets

import pytest

from sisteer.capeval.serving import vllm_auth


async def _inner_app(scope, receive, send):
    await send({"type": "inner.ok", "path": scope.get("path")})


def _call(mw, scope):
    sent = []

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        sent.append(message)

    asyncio.run(mw(scope, receive, send))
    return sent


def _scope(path, *, token=None, method="POST", type_="http"):
    headers = []
    if token is not None:
        headers.append((b"authorization", b"Bearer " + token.encode()))
    return {"type": type_, "method": method, "path": path,
            "headers": headers}


@pytest.fixture()
def mw(monkeypatch):
    token = secrets.token_urlsafe(16)
    monkeypatch.setenv(vllm_auth.TOKEN_ENV, token)
    return vllm_auth.SisteerTokenAuthMiddleware(_inner_app), token


def test_refuses_to_start_without_a_token(monkeypatch):
    monkeypatch.delenv(vllm_auth.TOKEN_ENV, raising=False)
    with pytest.raises(RuntimeError, match="open to every same-host"):
        vllm_auth.SisteerTokenAuthMiddleware(_inner_app)
    monkeypatch.setenv(vllm_auth.TOKEN_ENV, "")
    with pytest.raises(RuntimeError):
        vllm_auth.SisteerTokenAuthMiddleware(_inner_app)


def test_collective_rpc_requires_the_exact_token(mw):
    middleware, token = mw
    # no header -> 401, never the inner app
    sent = _call(middleware, _scope("/collective_rpc"))
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 401
    # wrong token -> 401
    sent = _call(middleware, _scope("/collective_rpc", token="wrong"))
    assert sent[0]["status"] == 401
    # wrong scheme -> 401
    sent = _call(middleware, {
        "type": "http", "method": "POST", "path": "/collective_rpc",
        "headers": [(b"authorization", b"Basic " + token.encode())]})
    assert sent[0]["status"] == 401
    # the launch's token -> through to the app
    sent = _call(middleware, _scope("/collective_rpc", token=token))
    assert sent == [{"type": "inner.ok", "path": "/collective_rpc"}]


def test_every_other_dev_and_api_path_is_guarded_too(mw):
    middleware, token = mw
    for path in ("/v1/models", "/v1/chat/completions", "/sleep",
                 "/server_info", "/reset_prefix_cache", "/anything"):
        assert _call(middleware, _scope(path))[0]["status"] == 401
        assert _call(middleware,
                     _scope(path, token=token))[0]["type"] == "inner.ok"


def test_liveness_paths_and_options_preflight_stay_open(mw):
    middleware, _token = mw
    for path in vllm_auth.UNGUARDED_PATHS:
        sent = _call(middleware, _scope(path, method="GET"))
        assert sent == [{"type": "inner.ok", "path": path}]
    sent = _call(middleware, _scope("/collective_rpc", method="OPTIONS"))
    assert sent[0]["type"] == "inner.ok"  # CORS preflight, no secrets


def test_root_path_prefix_is_stripped_like_the_stock_middleware(mw):
    middleware, token = mw
    scope = _scope("/prefix/health", method="GET")
    scope["root_path"] = "/prefix"
    assert _call(middleware, scope)[0]["type"] == "inner.ok"
    scope = _scope("/prefix/collective_rpc")
    scope["root_path"] = "/prefix"
    assert _call(middleware, scope)[0]["status"] == 401


def test_websocket_scopes_are_closed_not_crashed(mw):
    middleware, _token = mw
    sent = _call(middleware, {"type": "websocket", "path": "/v1/ws",
                              "headers": []})
    assert sent == [{"type": "websocket.close", "code": 1008}]


def test_lifespan_scopes_pass_through(mw):
    middleware, _token = mw
    sent = _call(middleware, {"type": "lifespan", "path": "", "headers": []})
    assert sent[0]["type"] == "inner.ok"
