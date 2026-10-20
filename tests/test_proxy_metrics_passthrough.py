"""F-O (qwen-seedrep-native-battery, launch report 06:44-06:48Z): the metrics scrape reads the
engine's Prometheus page THROUGH THE ROUTING PROXY, never the guarded engine.

The defect: ten proving smokes all passed readiness and then ended ``METRICS_UNSCRAPABLE
rc=1`` -- the container app's ``metrics_url_from_endpoint`` targeted the ENGINE's ``vllm_port`` and
``_http_get_text`` was a bare ``urlopen``; the engine guards ``/metrics`` (every non-liveness path)
with ``vllm_auth.SisteerTokenAuthMiddleware`` and its per-launch bearer lives ONLY inside the standup
process, so the scrape was an HTTP 401 by construction. The in-container sampler and
``determinism_check.py``'s drained-state read did the same bare GET.

The fix, pinned here: the proxy gains a read-only ``GET /metrics`` pass-through that forwards to the
engine with the launch's bearer and answers the body verbatim (guarded like ``/v1/models`` when the
proxy has a token, open like the proxy's other routes when ``endpoint.json`` names none); the app's
``metrics_url_from_endpoint`` targets ``proxy_port``; the sampler reads through ``_http_get_text``;
and the engine token never reaches the attempt root (which ``TreeSyncer`` mirrors to shared storage).

CPU-only: a raw ``http.server`` stands in for the guarded engine (the same bearer rule the real
middleware enforces -- proven against the REAL middleware below), the REAL proxy app serves the route
on a loopback port, and the REAL app module does the scrape.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import secrets
import sys
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sisteer.capeval.serving import proxy as px  # noqa: E402
from sisteer.capeval.serving import vllm_auth  # noqa: E402
from sisteer.capeval.serving.native_standup import _start_proxy_thread, free_port  # noqa: E402

from test_petri_proxy import StubBackend, _proxy  # noqa: E402
from test_seedrep_served_app import srv  # noqa: E402,F401  (the app module, modal stubbed)

pytest.importorskip("aiohttp")

APP = REPO / "scripts" / "modal" / "seedrep_served.py"
STANDUP = REPO / "src" / "sisteer" / "capeval" / "serving" / "native_standup.py"
PROXY = REPO / "src" / "sisteer" / "capeval" / "serving" / "proxy.py"
DETERMINISM = REPO / "scripts" / "sdfbase" / "seedrep" / "determinism_check.py"

#: a vLLM-shaped /metrics body (the families the sampler and the check read, with labels, plus comments)
ENGINE_BODY = (
    "# HELP vllm:num_requests_running Number of requests currently running on GPU.\n"
    "# TYPE vllm:num_requests_running gauge\n"
    'vllm:num_requests_running{engine="0",model_name="m"} 2.0\n'
    'vllm:num_requests_waiting{engine="0",model_name="m"} 5.0\n'
    'vllm:generation_tokens_total{engine="0",model_name="m"} 1234.0\n'
    'vllm:prompt_tokens_total{engine="0",model_name="m"} 777.0\n'
    'vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25\n'
)


# --------------------------------------------------------------------------- the stub GUARDED engine

class _GuardedEngine:
    """A raw HTTP server on loopback that enforces the SAME rule as ``SisteerTokenAuthMiddleware``:
    every path but the liveness trio needs ``Authorization: Bearer <token>`` (401 otherwise, the
    middleware's exact body); ``/metrics`` answers ENGINE_BODY with the Prometheus content type."""

    def __init__(self, token: str):
        self.token = token
        self.hits: list = []
        engine = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a, **k):  # quiet
                pass

            def do_GET(self):
                auth = self.headers.get("Authorization", "")
                engine.hits.append((self.path, auth))
                if self.path in vllm_auth.UNGUARDED_PATHS:
                    body = b"{}"
                    self.send_response(200)
                elif auth != f"Bearer {engine.token}":
                    body = b'{"error": "Unauthorized"}'
                    self.send_response(401)
                elif self.path == "/metrics":
                    body = ENGINE_BODY.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                else:
                    body = b"not found"
                    self.send_response(404)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.port = free_port()
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def engine():
    e = _GuardedEngine(secrets.token_urlsafe(16))
    try:
        yield e
    finally:
        e.close()


def _serve_proxy(tmp_path, engine, *, api_token=None):
    """The REAL proxy over the REAL HTTPBackend, holding the engine's bearer in-process, served on a loopback port."""
    backend = px.HTTPBackend(engine.base_url, api_key=engine.token, timeout_s=10.0)
    proxy, _, _ = _proxy(tmp_path, backend=backend, api_token=api_token)
    port = free_port()
    _start_proxy_thread(px.make_app(proxy), "127.0.0.1", port)
    return proxy, port


def _endpoint(engine, proxy_port, *, proxy_token=None):
    """The slice of ``endpoint.json`` the scrape reads, exactly as ``LoraStandup.endpoint()`` records it."""
    return {"proxy_url": f"http://127.0.0.1:{proxy_port}/v1", "proxy_port": proxy_port, "vllm_port": engine.port,
            "proxy_token": proxy_token}


# --------------------------------------------------------------------------- the premise: the REAL middleware guards /metrics

def test_the_real_engine_middleware_guards_metrics(monkeypatch):
    """The engine's ``/metrics`` is NOT in the liveness trio: a bare GET is a 401 under the real middleware,
    the bearer passes. This is the fact the proving smokes tripped on."""
    token = secrets.token_urlsafe(16)
    monkeypatch.setenv(vllm_auth.TOKEN_ENV, token)

    async def inner(scope, receive, send):
        await send({"type": "inner.ok"})
    mw = vllm_auth.SisteerTokenAuthMiddleware(inner)

    def call(headers):
        sent = []

        async def receive():
            return {"type": "http.request"}

        async def send(m):
            sent.append(m)
        asyncio.run(mw({"type": "http", "method": "GET", "path": "/metrics", "headers": headers}, receive, send))
        return sent
    assert "/metrics" not in vllm_auth.UNGUARDED_PATHS
    assert call([])[0]["status"] == 401
    assert call([(b"authorization", b"Bearer " + token.encode())])[0]["type"] == "inner.ok"


# --------------------------------------------------------------------------- (1) the failing shape reproduces; the pass-through fixes it

def test_w15_shape_bare_get_on_the_engine_port_is_401(engine, srv):
    """The exact failing read -- the app's fetch on the ENGINE's /metrics -- is an HTTP 401 (the stub enforces the
    real middleware's rule), which ``verify_metrics_scrape`` names as METRICS_UNSCRAPABLE's problem."""
    with pytest.raises(urllib.error.HTTPError) as ei:
        srv._http_get_text(f"{engine.base_url}/metrics")
    assert ei.value.code == 401
    _, problems = srv.verify_metrics_scrape(f"{engine.base_url}/metrics", srv._http_get_text)
    assert problems and "HTTPError" in problems[0] and "401" in problems[0]


def test_app_scrape_via_metrics_url_from_endpoint_reads_the_engine_body_through_the_proxy(tmp_path, engine, srv):
    """The fix end to end: ``metrics_url_from_endpoint`` targets the PROXY's port; the proxy forwards to the engine
    with the launch bearer (held in-process, never given to the app); the app's scrape gets the vLLM-shaped body
    VERBATIM and ``verify_metrics_scrape`` / ``parse_vllm_metrics`` accept it."""
    proxy, proxy_port = _serve_proxy(tmp_path, engine)
    endpoint = _endpoint(engine, proxy_port)
    url = srv.metrics_url_from_endpoint(endpoint)
    assert url == f"http://127.0.0.1:{proxy_port}/metrics" and str(engine.port) not in url
    headers = srv.metrics_headers_from_endpoint(endpoint)
    assert headers == {}                                        # proxy_token: None -> the proxy checks no token, the app sends none
    body = srv._http_get_text(url, headers=headers)
    assert body == ENGINE_BODY                                  # verbatim
    row, problems = srv.verify_metrics_scrape(url, lambda u: srv._http_get_text(u, headers=headers))
    assert problems == []
    assert row == srv.parse_vllm_metrics(ENGINE_BODY) == {
        "vllm:num_requests_running": 2.0, "vllm:num_requests_waiting": 5.0, "vllm:generation_tokens_total": 1234.0,
        "vllm:prompt_tokens_total": 777.0, "vllm:kv_cache_usage_perc": 0.25}
    # the proxy authenticated to the engine with the launch bearer; the app never saw it
    assert engine.hits[-1] == ("/metrics", f"Bearer {engine.token}")
    assert engine.token not in json.dumps(endpoint) and engine.token not in url
    # the content type is Prometheus text exposition
    import urllib.request
    with urllib.request.urlopen(url, timeout=5) as r:
        assert r.headers.get("Content-Type") == px.METRICS_CONTENT_TYPE


def test_proxy_metrics_is_guarded_like_models_when_the_proxy_has_a_token(tmp_path, engine, srv):
    """A token-bearing proxy guards ``/metrics`` exactly like ``/v1/models``: no bearer -> 401 ``bad_token``; the
    endpoint's ``proxy_token`` (what ``metrics_headers_from_endpoint`` sends) -> the body."""
    proxy, proxy_port = _serve_proxy(tmp_path, engine, api_token="ptok")
    endpoint = _endpoint(engine, proxy_port, proxy_token="ptok")
    url = srv.metrics_url_from_endpoint(endpoint)
    with pytest.raises(urllib.error.HTTPError) as ei:
        srv._http_get_text(url)
    assert ei.value.code == 401 and json.loads(ei.value.read())["error"]["code"] == "bad_token"
    with pytest.raises(urllib.error.HTTPError) as ei:
        srv._http_get_text(url, headers={"Authorization": "Bearer wrong"})
    assert ei.value.code == 401
    assert srv._http_get_text(url, headers=srv.metrics_headers_from_endpoint(endpoint)) == ENGINE_BODY


def test_proxy_forwards_the_engines_status_class_and_refuses_without_the_read(tmp_path):
    """A wrong ENGINE bearer inside the proxy surfaces as the engine's 401 (never an empty scrape); a backend without
    the read refuses 501 ``metrics_unsupported`` (a stub, a transport that lacks it)."""
    e = _GuardedEngine(secrets.token_urlsafe(16))
    try:
        backend = px.HTTPBackend(e.base_url, api_key="not-the-engine-token", timeout_s=10.0)
        proxy, _, _ = _proxy(tmp_path, backend=backend)
        with pytest.raises(px.ProxyRefusal) as ei:
            asyncio.run(proxy.handle_metrics())
        assert ei.value.http_status == 401
    finally:
        e.close()
    proxy, _, _ = _proxy(tmp_path, backend=StubBackend())
    with pytest.raises(px.ProxyRefusal) as ei:
        asyncio.run(proxy.handle_metrics())
    assert ei.value.http_status == 501 and ei.value.code == "metrics_unsupported"


def test_http_surface_exposes_the_metrics_route(tmp_path):
    proxy, _, _ = _proxy(tmp_path, backend=StubBackend())
    app = px.make_app(proxy)
    paths = {(getattr(res, "canonical", None) or str(res)) for res in app.router.resources()}
    assert px.METRICS_ROUTE == "/metrics" and "/metrics" in paths
    src = PROXY.read_text()
    assert 'app.router.add_get(METRICS_ROUTE, metrics)' in src
    assert 'f"{root}/metrics"' in src                          # the backend reads the engine's ROOT /metrics with the bearer


# --------------------------------------------------------------------------- (2) the determinism check's drained-state read

def test_determinism_check_reads_the_drained_state_through_the_proxy(tmp_path, engine):
    """``engine_state`` did the same bare GET on ``--metrics-url``: against the engine it REFUSES (401);
    against the proxy's pass-through (what ``determinism_check_argv`` now passes) it reads the gauges. A token-guarded
    proxy gets the same ``$VLLM_API_KEY`` bearer the replay sends."""
    import audit_common as ac
    import determinism_check as dc
    assert dc.METRICS_FN is None
    with pytest.raises(ac.AuditRefusal, match="401"):
        dc.engine_state(f"{engine.base_url}/metrics")
    _, proxy_port = _serve_proxy(tmp_path, engine)
    state = dc.engine_state(f"http://127.0.0.1:{proxy_port}/metrics")
    assert state == {"num_requests_running": 2.0, "num_requests_waiting": 5.0, "generation_tokens_total": 1234.0}
    _, guarded_port = _serve_proxy(tmp_path, engine, api_token="ptok")
    with pytest.raises(ac.AuditRefusal, match="401"):
        dc.engine_state(f"http://127.0.0.1:{guarded_port}/metrics")
    assert dc.engine_state(f"http://127.0.0.1:{guarded_port}/metrics", "ptok")["num_requests_running"] == 2.0
    # the argv the app composes hands the check the SAME url the app scrapes
    src = APP.read_text()
    assert '"--metrics-url", metrics_url' in src and "metrics_url=metrics_url" in src


# --------------------------------------------------------------------------- (3) the sampler reads through _http_get_text (AST pin)

def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"no function {name}")


def _called_names(node: ast.AST) -> set:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            out.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None))
    return out


def test_sampler_reads_through_http_get_text_not_a_bare_urlopen(srv):
    tree = ast.parse(APP.read_text())
    sampler = _function(tree, "_start_samplers")
    called = _called_names(sampler)
    assert "urlopen" not in called and "Request" not in called
    assert "_http_get_text" in called and "parse_vllm_metrics" in called
    # the only urlopen in the app is inside _http_get_text
    owners = {f.name for f in ast.walk(tree) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
              and "urlopen" in _called_names(f)}
    assert owners == {"_http_get_text"}, owners
    # and the live scrape hands the sampler the verified url + headers
    src = APP.read_text()
    assert '_start_samplers(serve_dir, root, stop, {"metrics_url": metrics_url, "metrics_headers": metrics_headers})' in src
    assert 'verify_metrics_scrape(metrics_url, lambda u: _http_get_text(u, headers=metrics_headers))' in src


# --------------------------------------------------------------------------- (4) the engine token never reaches the attempt root

TOKEN_NAMES = {"api_key", "_api_key", "VLLM_API_KEY", "TOKEN_ENV", "API_KEY_ENV", "PROXY_BEARER_ENV", "auth_headers", "metrics_headers"}
WRITERS = {"write_json", "dump", "dumps", "write_text", "write_bytes", "open", "writelines"}


def _writes_touching_tokens(path: Path) -> list:
    tree = ast.parse(path.read_text())
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name not in WRITERS:
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Name) and sub.id in TOKEN_NAMES) or (isinstance(sub, ast.Attribute) and sub.attr in TOKEN_NAMES):
                hits.append((path.name, node.lineno, name, ast.unparse(sub)))
    return hits


def test_no_writer_in_the_standup_app_or_proxy_takes_the_token(srv):
    """Every JSON/file writer call in the standup (which writes endpoint.json), the container app (whose attempt root
    ``TreeSyncer`` mirrors to the shared volume) and the proxy is free of the token names: the bearer stays
    in-process, the way ``endpoint.json`` ``proxy_token: None`` promises."""
    hits = _writes_touching_tokens(STANDUP) + _writes_touching_tokens(APP) + _writes_touching_tokens(PROXY) + _writes_touching_tokens(DETERMINISM)
    assert hits == [], hits
    # endpoint.json records no proxy token, and the app's metrics evidence carries url / first_scrape / problems only
    assert '"proxy_token": None,' in STANDUP.read_text()
    src = APP.read_text()
    assert 'evidence["metrics"] = {"url": metrics_url, "first_scrape": first_scrape, "problems": m_problems}' in src
    # the app names the ENGINE's token nowhere: it cannot leak what it never holds
    assert "VLLM_API_KEY" not in src.replace("$VLLM_API_KEY", "")   # the one mention is the determinism docstring's env name


def test_metrics_url_contract(srv):
    """The endpoint contract: ``proxy_port`` first, the port of ``proxy_url`` when the int is absent, a refusal when
    neither names one -- and the ENGINE's ``vllm_port`` is never the target."""
    assert srv.metrics_url_from_endpoint({"proxy_port": 9100, "vllm_port": 8100}) == "http://127.0.0.1:9100/metrics"
    assert srv.metrics_url_from_endpoint({"proxy_url": "http://127.0.0.1:9101/v1", "vllm_port": 8100}) == "http://127.0.0.1:9101/metrics"
    assert srv.metrics_url_from_endpoint({"proxy_port": True, "proxy_url": "http://127.0.0.1:9102/v1"}) == "http://127.0.0.1:9102/metrics"
    for bad in ({"vllm_port": 8100}, {"proxy_port": 0, "proxy_url": "nope"}, {}):
        with pytest.raises(ValueError, match="proxy_port"):
            srv.metrics_url_from_endpoint(bad)
    assert srv.metrics_headers_from_endpoint({"proxy_token": None}) == {}
    assert srv.metrics_headers_from_endpoint({"proxy_token": ""}) == {}
    assert srv.metrics_headers_from_endpoint({"proxy_token": "t"}) == {"Authorization": "Bearer t"}
