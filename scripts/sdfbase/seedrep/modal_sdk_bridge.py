#!/usr/bin/env python
"""modal_sdk_bridge.py — the rung consumer's Modal SDK helper (fx1).

The rung container runs ``rung_redraw.py consume`` under the DRIVER venv (``uv sync --frozen --extra cu128``), which carries no
``modal`` distribution, with ``PYTHONPATH=<tree>/src:<tree>/scripts`` -- so a bare ``import modal`` there answers with the repo's
``scripts/modal/`` NAMESPACE package (no ``Volume``, no ``Dict``) and the consumer died ``AttributeError: module 'modal' has no
attribute 'Volume'`` (11 seed-1 rungs, 2026-09-21). The container's ONE interpreter that carries the SDK is the served app's own
(``sys.executable`` of ``run_rung``), which the app names to the consumer child in ``$SEEDREP_MODAL_SDK_PYTHON``. The consumer spawns
THIS file under that interpreter in ISOLATED mode (``python -I <this file>``: no PYTHONPATH, no user site, no script directory on
``sys.path``), so the repo's ``scripts/`` can never shadow anything here.

fx1b: inside a Modal container the SDK is NOT in the interpreter's site-packages -- modal 1.5.4 mounts it at ``/pkg``
and ``/pkg/sitecustomize.py`` puts ``/__modal/deps`` on ``sys.path`` -- and ``-I`` sees neither (no PYTHONPATH, no site customisation).
So the app, which HAS them on its own ``sys.path``, hands the helper the SDK IMPORT ROOTS in ``$SEEDREP_MODAL_SDK_PATH`` (os.pathsep-joined:
the parent of ``modal``'s package directory first, then its other non-repo ``sys.path`` entries; ``seedrep_served.sdk_import_roots``), and
:func:`import_sdk` PREPENDS them, in order, before its ``import modal``. The import is checked (:func:`checked_sdk`): a namespace package or a
module without ``Dict`` / ``Volume`` is refused BY NAME, naming the roots tried -- in-band, as the ``ok: false`` reply to ``hello``, so the
consumer's :class:`rung_redraw.SdkBridge` raises that message at its handshake and nothing is drawn.

Stdlib + ``modal`` ONLY: nothing of the repo is imported here (the driver venv's packages are not in this interpreter, and this file
must not need them). The consumer end is ``rung_redraw.SdkBridge`` / ``rung_redraw.BridgeKV``.

Protocol -- newline-delimited JSON, one request per line on stdin, one reply per line on stdout, in order::

    {"id": <any>, "op": "hello"}                                       -> {"modal": <version>, "file": <modal.__file__>, "python": <sys.executable>, "roots": [...]}
    {"id", "op": "dict.put",      "name", "key", "value", "skip_if_exists": bool}  -> bool   (modal.Dict.put's "added": False when held)
    {"id", "op": "dict.get",      "name", "key", ["default"]}          -> the value, or the default (None when none is given)
    {"id", "op": "dict.pop",      "name", "key", ["default"]}          -> the value; absent -> the default when one is given, else a KeyError reply
    {"id", "op": "dict.contains", "name", "key"}                       -> bool
    {"id", "op": "dict.items",    "name"}                              -> [[key, value], ...]
    {"id", "op": "volume.commit" | "volume.reload", "name"}            -> null

A reply is ``{"id": <the request's id, null when it had none>, "ok": true, "result": ...}`` or
``{"id": ..., "ok": false, "type": <exception class name>, "error": <message>}``; a line that is not a JSON object is answered with an
``ok: false`` reply (``id: null``) and serving continues. Dicts are ``modal.Dict.from_name(name, create_if_missing=True)``, Volumes
``modal.Volume.from_name(name)``, each opened once per name. The SDK is imported on the FIRST request, so an interpreter / root set
without it answers ``hello`` with an ``ok: false`` reply naming the roots tried (and every later request the same way); EOF on stdin ends
the process (exit 0).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, TextIO

#: == seedrep_served.RUNG_SDK_PATH_ENV: the os.pathsep-joined import roots the app computed from ITS interpreter (fx1b)
SDK_PATH_ENV = "SEEDREP_MODAL_SDK_PATH"


def sdk_roots_from_env(env: Optional[Dict[str, str]] = None) -> List[str]:
    e = os.environ if env is None else env
    return [r for r in str(e.get(SDK_PATH_ENV, "") or "").split(os.pathsep) if r]


def checked_sdk(modal: Any, roots: Sequence[str] = ()) -> Any:
    """``modal`` if it is the SDK (``Dict`` and ``Volume`` present); a namespace package (the repo's ``scripts/modal`` shadowing the
    name under an interpreter with no ``modal`` distribution) or a module without them is refused by name, naming the roots tried."""
    if hasattr(modal, "Dict") and hasattr(modal, "Volume"):
        return modal
    where = getattr(modal, "__file__", None) or ", ".join(str(p) for p in (getattr(modal, "__path__", None) or [])) or "?"
    raise RuntimeError(f"`import modal` under {sys.executable} resolved to {where}, which is not the Modal SDK (a namespace package or a module "
                       f"without Dict / Volume; this interpreter carries no `modal` distribution on its own path); SDK import roots tried "
                       f"({SDK_PATH_ENV}): {list(roots)}")


def import_sdk(roots: Optional[Sequence[str]] = None) -> Any:
    """Prepend the SDK import roots (``roots``, default ``$SEEDREP_MODAL_SDK_PATH``) to ``sys.path`` IN ORDER, then ``import modal`` and check
    it. Isolated mode keeps the repo's ``scripts/`` off the path; the roots put the container's ``/pkg`` + ``/__modal/deps`` (or a venv's
    site-packages) on it. An import that fails is refused by name with the roots tried."""
    roots = list(sdk_roots_from_env() if roots is None else roots)
    for r in reversed(roots):
        if r and r not in sys.path:
            sys.path.insert(0, r)
    try:
        import modal  # never the repo's scripts/modal (-I); the roots above decide what answers
    except ImportError as e:
        raise RuntimeError(f"`import modal` failed under {sys.executable}: {type(e).__name__}: {e}; SDK import roots tried ({SDK_PATH_ENV}): {roots} "
                           f"-- the served app computes them from its own interpreter (seedrep_served.sdk_import_roots) and the consumer child "
                           f"must carry them") from e
    return checked_sdk(modal, roots)


class Server:
    """The SDK side: one handle per Dict / Volume name, the seven verbs. The SDK is resolved on the first request (``sdk`` given = a test
    seam; else :func:`import_sdk` over ``roots``), so a missing SDK is an in-band refusal, never a dead pipe."""

    def __init__(self, sdk: Any = None, *, roots: Optional[Sequence[str]] = None) -> None:
        self._sdk = sdk
        self._roots = list(sdk_roots_from_env() if roots is None else roots)
        self._dicts: Dict[str, Any] = {}
        self._vols: Dict[str, Any] = {}

    @property
    def sdk(self) -> Any:
        if self._sdk is None:
            self._sdk = import_sdk(self._roots)
        return self._sdk

    def _dict(self, name: str) -> Any:
        d = self._dicts.get(name)
        if d is None:
            d = self._dicts[name] = self.sdk.Dict.from_name(name, create_if_missing=True)
        return d

    def _vol(self, name: str) -> Any:
        v = self._vols.get(name)
        if v is None:
            v = self._vols[name] = self.sdk.Volume.from_name(name)
        return v

    def handle(self, req: Dict[str, Any]) -> Any:
        op = req.get("op")
        if op == "hello":
            sdk = self.sdk
            return {"modal": getattr(sdk, "__version__", None), "file": getattr(sdk, "__file__", None), "python": sys.executable, "roots": list(self._roots)}
        if op == "dict.put":
            return bool(self._dict(req["name"]).put(req["key"], req.get("value"), skip_if_exists=bool(req.get("skip_if_exists", False))))
        if op == "dict.get":
            return self._dict(req["name"]).get(req["key"], req.get("default"))
        if op == "dict.pop":
            d = self._dict(req["name"])
            if "default" not in req:
                return d.pop(req["key"])                        # absent -> KeyError -> an ok:false reply of type KeyError
            try:
                return d.pop(req["key"])
            except KeyError:
                return req["default"]
        if op == "dict.contains":
            return bool(self._dict(req["name"]).contains(req["key"]))
        if op == "dict.items":
            return [[k, v] for k, v in self._dict(req["name"]).items()]
        if op == "volume.commit":
            self._vol(req["name"]).commit()
            return None
        if op == "volume.reload":
            self._vol(req["name"]).reload()
            return None
        raise ValueError(f"unknown op {op!r}")


def serve(inp: TextIO, out: TextIO, *, sdk: Any = None, roots: Optional[Sequence[str]] = None) -> int:
    """Answer every request line on ``inp`` with one reply line on ``out`` until EOF. ``sdk`` is a test seam (else the checked SDK over the
    roots, imported on the first request)."""
    srv = Server(sdk, roots=roots)
    for line in inp:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise ValueError("the request line is not a JSON object")
            rid = req.get("id")
            rep: Dict[str, Any] = {"id": rid, "ok": True, "result": srv.handle(req)}
        except json.JSONDecodeError as e:
            rep = {"id": rid, "ok": False, "type": type(e).__name__, "error": f"not a JSON request line: {e}"}
        except Exception as e:  # noqa: BLE001 -- every failure is answered by type and message, never a dead pipe
            rep = {"id": rid, "ok": False, "type": type(e).__name__, "error": str(e)}
        out.write(json.dumps(rep, default=str) + "\n")
        out.flush()
    return 0


def main() -> int:
    return serve(sys.stdin, sys.stdout)


if __name__ == "__main__":
    sys.exit(main())
