"""The RoutingProxy's two facts.

1. THE RESPONSE ``id`` PASSES THROUGH UNCHANGED. The lora-native selection
   control keys the worker's routing observations by the engine request id,
   which for a single-prompt chat completion IS the response body's ``id``
   (vLLM 0.25.1 ``entrypoints/openai/chat_completion/serving.py:297``). The
   proxy copies the upstream body (``out = dict(response)``) and sets only
   ``model``, ``petri_intervention_hash`` and ``petri`` — verified by reading
   ``handle_chat_completions`` and pinned here on a lora route, a passthrough
   route and through the aiohttp app.
2. THE ALLOWLISTED RPC PASS-THROUGH (``LORA_ROUTING_RPC_ROUTE``): the gate's
   lora-native canary attaches to a running standup with no launch token, so
   it reaches the three recorder RPCs through the proxy — exactly those
   methods, string args only (exactly one, the client-minted epoch, for each), the
   proxy's token rule, a 404 on a proxy without a lora-native engine
   attestation (a lens proxy: no recorder exists and this is no generic RPC
   door), a 502 on a backend without ``collective_rpc`` or one that fails,
   and the body forwarded as the backend answered it.

The proxy is built by ``tests/test_petri_proxy.py``'s helpers (a synthetic
cell source with one native-LoRA route); no server."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_petri_proxy import StubBackend, _ask, _chat, _proxy  # noqa: E402

from sisteer.capeval.serving import proxy as px  # noqa: E402
from sisteer.capeval.serving import vllm_server as vs  # noqa: E402

LORA_VMODEL = "petri__si_safety__lora_r1"


def _resp(rid, model="m"):
    return {"id": rid, "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "token_ids": [1, 2],
                         "message": {"role": "assistant", "content": "x"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}


def test_the_upstream_response_id_reaches_the_client_unchanged(tmp_path):
    backend = StubBackend(responses=[_resp("chatcmpl-aaaa"), _resp("chatcmpl-bbbb")])
    proxy, _, _ = _proxy(tmp_path, backend=backend)
    out = _ask(proxy, _chat(LORA_VMODEL))
    assert out["id"] == "chatcmpl-aaaa"                          # the engine request id, verbatim
    assert out["model"] == LORA_VMODEL and out["petri"]["serving_identity"].startswith("proxy_vllm+lora")
    assert out["choices"][0]["token_ids"] == [1, 2]
    out = _ask(proxy, _chat("petri__base"))
    assert out["id"] == "chatcmpl-bbbb"
    # the proxy adds exactly its three keys and rewrites nothing else
    added = set(out) - set(_resp("x"))
    assert added == {"petri", "petri_intervention_hash"}


def test_the_id_passes_through_the_aiohttp_app_too(tmp_path):
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestClient, TestServer
    backend = StubBackend(responses=[_resp("chatcmpl-app")])
    proxy, _, _ = _proxy(tmp_path, backend=backend)

    async def go():
        async with TestClient(TestServer(px.make_app(proxy))) as client:
            r = await client.post("/v1/chat/completions", json=_chat(LORA_VMODEL))
            assert r.status == 200
            body = await r.json()
            assert body["id"] == "chatcmpl-app"
            # the RPC route is mounted; a proxy WITHOUT a lora-native attestation refuses 404
            r = await client.post(px.LORA_ROUTING_RPC_ROUTE,
                                  json={"method": vs.LORA_ROUTING_ARM_RPC_METHOD, "args": ["e" * 32]})
            assert r.status == 404
            err = await r.json()
            assert err["error"]["code"] == "not_lora_native"
    asyncio.run(go())
    del aiohttp


class _RpcBackend(StubBackend):
    def __init__(self, answers=None, fail=None):
        super().__init__()
        self.rpc_calls = []
        self.answers = answers or {}
        self.fail = fail

    async def collective_rpc(self, method, args):
        self.rpc_calls.append((method, list(args)))
        if self.fail is not None:
            raise self.fail
        return self.answers.get(method, {"results": [{"ok": True, "method": method}]})


def _native_proxy(tmp_path, backend):
    proxy, _, _ = _proxy(tmp_path, backend=backend)
    # the constructor's engine-attestation validation is the standup's business
    # (test_lora_serving_up); the handler keys on the attribute's presence
    proxy.engine_attestation = {"kind": "lora_native", "enforce_eager": False,
                                "cudagraph_mode": "FULL_AND_PIECEWISE", "lora_config": {},
                                "adapters": {}}
    return proxy


def _rpc(proxy, body, authorization=None):
    return asyncio.run(proxy.handle_lora_routing_rpc(body, authorization=authorization))


def test_the_pass_through_forwards_exactly_the_three_methods_with_string_args(tmp_path):
    backend = _RpcBackend(answers={vs.LORA_ROUTING_OBSERVE_RPC_METHOD: {"results": [{"armed": True,
                                                                                    "observations": {}}]}})
    proxy = _native_proxy(tmp_path, backend)
    ep = vs.lora_routing_epoch_args("0f" * 16)
    out = _rpc(proxy, {"method": vs.LORA_ROUTING_ARM_RPC_METHOD, "args": ep})
    assert out == {"results": [{"ok": True, "method": vs.LORA_ROUTING_ARM_RPC_METHOD}]}
    out = _rpc(proxy, {"method": vs.LORA_ROUTING_OBSERVE_RPC_METHOD, "args": ep})
    assert out == {"results": [{"armed": True, "observations": {}}]}
    out = _rpc(proxy, {"method": vs.LORA_ROUTING_DISARM_RPC_METHOD, "args": ep})
    assert out["results"][0]["method"] == vs.LORA_ROUTING_DISARM_RPC_METHOD
    assert backend.rpc_calls == [(vs.LORA_ROUTING_ARM_RPC_METHOD, ep),
                                 (vs.LORA_ROUTING_OBSERVE_RPC_METHOD, ep),
                                 (vs.LORA_ROUTING_DISARM_RPC_METHOD, ep)]
    # a worker REFUSAL rides as an ok: False payload in the 200 body, forwarded UNCHANGED;
    # the client's reader is what refuses (transport)
    refusal = {"results": [{"ok": False, "refusal": "already_armed", "detail": "armed by epoch x",
                            "live_epoch": "x", "runner_kind": "v2", "runner_class": "c"}]}
    proxy2 = _native_proxy(tmp_path, _RpcBackend(answers={vs.LORA_ROUTING_ARM_RPC_METHOD: refusal}))
    out = _rpc(proxy2, {"method": vs.LORA_ROUTING_ARM_RPC_METHOD, "args": ep})
    assert out == refusal
    with pytest.raises(vs.VllmServerError, match="refused — already_armed"):
        vs.lora_routing_rpc_result(out, what="w")
    # the launcher's reader reads a forwarded SUCCESS body (ok: True)
    ok_body = {"results": [{"ok": True, "armed": False, "disarmed": True, "epoch": "0f" * 16}]}
    proxy3 = _native_proxy(tmp_path, _RpcBackend(answers={vs.LORA_ROUTING_DISARM_RPC_METHOD: ok_body}))
    out = _rpc(proxy3, {"method": vs.LORA_ROUTING_DISARM_RPC_METHOD, "args": ep})
    assert vs.lora_routing_rpc_result(out, what="w")["n_workers"] == 1


def test_the_pass_through_refuses_other_methods_bad_args_and_non_native_proxies(tmp_path):
    backend = _RpcBackend()
    proxy = _native_proxy(tmp_path, backend)
    with pytest.raises(px.ProxyRefusal, match="not one of the LoRA routing recorder RPCs") as ei:
        _rpc(proxy, {"method": "sisteer_install_steering", "args": []})
    assert ei.value.code == "rpc_method_not_allowed" and ei.value.http_status == 400
    with pytest.raises(px.ProxyRefusal, match="not one of the LoRA routing recorder RPCs"):
        _rpc(proxy, {"method": "sisteer_scheduler_config"})
    with pytest.raises(px.ProxyRefusal, match="not one of the LoRA routing recorder RPCs"):
        _rpc(proxy, {"args": []})
    with pytest.raises(px.ProxyRefusal, match="args must be a list of strings") as ei:
        _rpc(proxy, {"method": vs.LORA_ROUTING_OBSERVE_RPC_METHOD, "args": [["a"]]})
    assert ei.value.code == "rpc_bad_args"
    # exactly ONE non-empty arg (the client-minted epoch) for all three
    with pytest.raises(px.ProxyRefusal, match="takes exactly 1 arg"):
        _rpc(proxy, {"method": vs.LORA_ROUTING_ARM_RPC_METHOD, "args": []})
    with pytest.raises(px.ProxyRefusal, match="takes exactly 1 arg"):
        _rpc(proxy, {"method": vs.LORA_ROUTING_ARM_RPC_METHOD, "args": [""]})
    with pytest.raises(px.ProxyRefusal, match="takes exactly 1 arg"):
        _rpc(proxy, {"method": vs.LORA_ROUTING_OBSERVE_RPC_METHOD, "args": ["a", "b"]})
    with pytest.raises(px.ProxyRefusal, match="takes exactly 1 arg"):
        _rpc(proxy, {"method": vs.LORA_ROUTING_DISARM_RPC_METHOD})
    with pytest.raises(px.ProxyRefusal, match="request body must be a JSON object"):
        _rpc(proxy, ["nope"])
    assert backend.rpc_calls == []                                    # nothing reached the upstream
    # a proxy without a lora-native engine attestation (a lens proxy): 404
    lens, _, _ = _proxy(tmp_path, backend=_RpcBackend())
    with pytest.raises(px.ProxyRefusal, match="no lora-native engine attestation") as ei:
        _rpc(lens, {"method": vs.LORA_ROUTING_ARM_RPC_METHOD, "args": ["e" * 32]})
    assert ei.value.http_status == 404 and ei.value.code == "not_lora_native"


def test_the_pass_through_honours_the_token_rule_and_names_upstream_failures(tmp_path):
    proxy, _, _ = _proxy(tmp_path, backend=_RpcBackend(), api_token="sekrit")
    proxy.engine_attestation = {"kind": "lora_native"}
    with pytest.raises(px.ProxyRefusal, match="missing or wrong proxy token") as ei:
        _rpc(proxy, {"method": vs.LORA_ROUTING_ARM_RPC_METHOD, "args": ["e" * 32]})
    assert ei.value.http_status == 401
    _rpc(proxy, {"method": vs.LORA_ROUTING_ARM_RPC_METHOD, "args": ["e" * 32]}, authorization="Bearer sekrit")
    # a backend without collective_rpc: 502 naming it
    bare = _native_proxy(tmp_path, StubBackend())
    with pytest.raises(px.ProxyRefusal, match="exposes no collective_rpc") as ei:
        _rpc(bare, {"method": vs.LORA_ROUTING_ARM_RPC_METHOD, "args": ["e" * 32]})
    assert ei.value.http_status == 502 and ei.value.code == "rpc_unavailable"
    # an upstream failure: 502 naming the method and the error, never a silent {}
    failing = _native_proxy(tmp_path, _RpcBackend(fail=RuntimeError("connection reset")))
    with pytest.raises(px.ProxyRefusal, match="sisteer_arm_lora_routing_recorder: the upstream collective_rpc failed \\(RuntimeError: connection reset\\)") as ei:
        _rpc(failing, {"method": vs.LORA_ROUTING_ARM_RPC_METHOD, "args": ["e" * 32]})
    assert ei.value.http_status == 502 and ei.value.code == "rpc_failed"
    # an upstream refusal keeps its own status
    refusing = _native_proxy(tmp_path, _RpcBackend(fail=px.upstream_refusal(401, "nope")))
    with pytest.raises(px.ProxyRefusal) as ei:
        _rpc(refusing, {"method": vs.LORA_ROUTING_ARM_RPC_METHOD, "args": ["e" * 32]})
    assert ei.value.http_status in (401, 502)
    # a non-object upstream body
    odd = _native_proxy(tmp_path, _RpcBackend(answers={vs.LORA_ROUTING_ARM_RPC_METHOD: ["x"]}))
    with pytest.raises(px.ProxyRefusal, match="non-object body"):
        _rpc(odd, {"method": vs.LORA_ROUTING_ARM_RPC_METHOD, "args": ["e" * 32]})


def test_http_backend_collective_rpc_posts_to_the_server_root_with_the_bearer():
    """``HTTPBackend.collective_rpc`` builds the same POST the launcher's
    ``_post_collective_rpc`` does (server ROOT, ``{"method", "args"}``, the
    upstream bearer) — pinned against a fake aiohttp session."""
    import types
    captured = {}

    class _Resp:
        status = 200

        async def text(self):
            return json.dumps({"results": [{"armed": True}]})

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Session:
        def __init__(self, timeout=None):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def post(self, url, json=None, headers=None):
            captured.update(url=url, json=json, headers=headers)
            return _Resp()

    fake_aiohttp = types.SimpleNamespace(ClientSession=_Session,
                                         ClientTimeout=lambda total: ("timeout", total))
    sys.modules["aiohttp"], real = fake_aiohttp, sys.modules.get("aiohttp")
    try:
        be = px.HTTPBackend("http://127.0.0.1:9/v1", api_key="k", timeout_s=7.0)
        out = asyncio.run(be.collective_rpc(vs.LORA_ROUTING_ARM_RPC_METHOD, ["e" * 32]))
    finally:
        if real is not None:
            sys.modules["aiohttp"] = real
        else:
            sys.modules.pop("aiohttp", None)
    assert out == {"results": [{"armed": True}]}
    assert captured["url"] == "http://127.0.0.1:9/collective_rpc"
    assert captured["json"] == {"method": vs.LORA_ROUTING_ARM_RPC_METHOD, "args": ["e" * 32]}
    assert captured["headers"] == {"Authorization": "Bearer k"}
    assert captured["timeout"] == ("timeout", 7.0)
