"""Bearer-token auth for EVERY route of the audit's vLLM server.

WHY THIS EXISTS. The pinned vLLM 0.25.1's own ``--api-key`` /
``VLLM_API_KEY`` middleware guards only paths under ``GUARDED_PREFIX =
("/v1", "/v2", "/inference")`` (read out of
``vllm/entrypoints/serve/utils/server_utils.py`` at the tag) — the dev
routes that ``VLLM_SERVER_DEV_MODE`` mounts, above all
``POST /collective_rpc`` (arbitrary worker RPC by method name), are NOT
covered by it. A loopback bind is not tenant isolation on a shared
Slurm node: any same-host process of any user can open a loopback
connection, so an unauthenticated RPC route would hand every tenant
worker-level control of the audit server for its whole lifetime.

The route cannot be closed after the one-shot steering install: the
pinned server exposes no unmount/disable toggle for a mounted FastAPI
router, and the launcher sits outside the process. So authentication is
MANDATORY, and this class is how it reaches the unguarded paths: passed
through vLLM's supported ``--middleware`` hook (the pinned
``api_server.py`` imports the dotted class and ``app.add_middleware``\\ s
it), it enforces the SAME per-launch random token the stock middleware
checks — read once, at server startup, from the server process's own
``VLLM_API_KEY`` (which ``launch_server`` injects at spawn; the token
never appears on argv, where ``/proc/<pid>/cmdline`` is world-readable
on a shared node, and never in any recorded mapping).

Deliberately dependency-free (stdlib only): it imports inside the
serving venv's API server process, where sisteer arrives via PYTHONPATH
and torch may be busy elsewhere.
"""

from __future__ import annotations

import hashlib
import os
import secrets

#: where the per-launch token lives in the SERVER process env — the same
#: variable the stock vLLM middleware reads, so one token guards both
#: the stock-guarded /v1* paths and (via this class) everything else.
TOKEN_ENV = "VLLM_API_KEY"

#: liveness endpoints left unauthenticated (no state, no control):
#: matches the stock middleware's intent for /health-style probes.
UNGUARDED_PATHS = ("/health", "/ping", "/version")


class SisteerTokenAuthMiddleware:
    """Pure-ASGI bearer auth over every path except the liveness trio.

    Constructed by the pinned server's ``--middleware`` hook with the
    app alone, so configuration comes from the process env: a missing
    or empty token is a REFUSAL at startup — a lens server must never
    come up with its RPC surface open."""

    def __init__(self, app):
        self.app = app
        token = os.environ.get(TOKEN_ENV, "")
        if not token:
            raise RuntimeError(
                f"SisteerTokenAuthMiddleware: {TOKEN_ENV} is empty in the "
                "server environment — refusing to start an audit server "
                "whose dev RPC surface (/collective_rpc) would be open to "
                "every same-host tenant; launch through "
                "vllm_server.launch_server, which injects the per-launch "
                "token")
        self._token_hash = hashlib.sha256(token.encode("utf-8")).digest()

    def _authorized(self, scope) -> bool:
        value = ""
        for name, raw in scope.get("headers") or ():
            if name.decode("latin1").lower() == "authorization":
                value = raw.decode("latin1")
                break
        scheme, _, param = value.partition(" ")
        if scheme.lower() != "bearer":
            return False
        given = hashlib.sha256(param.encode("utf-8")).digest()
        return secrets.compare_digest(given, self._token_hash)

    async def __call__(self, scope, receive, send):
        if (scope["type"] not in ("http", "websocket")
                or scope.get("method") == "OPTIONS"):
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        root = scope.get("root_path", "")
        if root and path.startswith(root):
            path = path[len(root):]
        if path in UNGUARDED_PATHS:
            return await self.app(scope, receive, send)
        if self._authorized(scope):
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            return await send({"type": "websocket.close", "code": 1008})
        body = b'{"error": "Unauthorized"}'
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length",
                                 str(len(body)).encode("ascii"))]})
        await send({"type": "http.response.body", "body": body})
