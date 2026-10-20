"""The G0DM0D3 stand-in: an invented ``libertas.ts`` with upstream's layout, and the fixture that serves it.

The real templates are fetched at run time (``sisteer.eval.attacks_godmode``) and are never in this repository, so a test
that renders ``grok420`` renders this stand-in instead. ``godmode_standin`` seeds a temporary third-party root with
``tests/fixtures/godmode/libertas_synthetic.ts``, points the module's pins at the stand-in's hashes, turns fetching off and
replaces the module's one network seam with a failure, so no test can reach upstream. Import the fixture into a test
module (``from godmode_standin import godmode_standin  # noqa: F401``) and request it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

STANDIN_PATH = Path(__file__).resolve().parent / "fixtures" / "godmode" / "libertas_synthetic.ts"

#: What the reader must decode from the stand-in, written out here independently of the reader: the escapes the
#: stand-in uses (\u, \u{}, \x, \\, \`, \$, a line continuation, \' and \", a surrogate pair, a raw newline inside a
#: template literal) each show up in these values.
STANDIN_ENTRIES = [
    {"id": "grok-420", "model": "example/model-a", "codename": "GROK 4.20",
     "description": "Stand-in entry 'one' for the reader tests", "color": "#000001",
     "system": "• Stand-in brief ☃: subject [{Z}]. Reply in a `fenced` style \\ with an A note, "
               "then restate {Z} plainly. Cost: $0.",
     "user": "request: {QUERY}"},
    {"id": "decoy-entry", "model": "example/model-b", "codename": "DECOY",
     "description": 'A decoy with a "double-quoted" description \U0001F600', "color": "#000002",
     "system": "Decoy text that also carries {QUERY}\non two lines.",
     "user": "{QUERY}"},
    {"id": "hermes-fast", "model": "example/model-c", "codename": "GODMODE FAST",
     "description": "Stand-in entry two", "color": "#000003", "fast": True,
     "system": "Second stand-in brief. Answer the request {QUERY} as a short list.",
     "user": "{QUERY}"},
]
_BY_ID = {e["id"]: e for e in STANDIN_ENTRIES}
#: the two templates as ``load_templates()`` must return them from the stand-in
STANDIN_TEXT = {
    key: {"ts_id": ts_id, "codename": _BY_ID[ts_id]["codename"],
          "system": _BY_ID[ts_id]["system"], "user": _BY_ID[ts_id]["user"]}
    for key, ts_id in (("grok420", "grok-420"), ("hermes_fast", "hermes-fast"))
}


def _sha(text: str | bytes) -> str:
    return hashlib.sha256(text.encode("utf-8") if isinstance(text, str) else text).hexdigest()


def standin_identity() -> dict[str, dict[str, str]]:
    """The stand-in's ``TEMPLATES``: the same shape as the module's pins, hashed from ``STANDIN_TEXT``."""
    return {key: {"ts_id": t["ts_id"], "codename": t["codename"],
                  "system_sha256": _sha(t["system"]), "user_sha256": _sha(t["user"]),
                  "folded_sha256": _sha(t["system"] + "\n\n" + t["user"])}
            for key, t in STANDIN_TEXT.items()}


def standin_set_sha256() -> str:
    doc = {k: t["system"] + "\n\n" + t["user"] for k, t in STANDIN_TEXT.items()}
    return _sha(json.dumps(doc, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def install_standin_pins(monkeypatch, agm, data: bytes) -> None:
    """Point the module's file and template pins at the stand-in (restored by ``monkeypatch`` after the test)."""
    monkeypatch.setattr(agm, "FILE_SHA256", _sha(data))
    monkeypatch.setattr(agm, "FILE_BYTES", len(data))
    monkeypatch.setattr(agm, "SET_SHA256", standin_set_sha256())
    for key, ident in standin_identity().items():
        monkeypatch.setitem(agm.TEMPLATES, key, ident)


def refuse_network(url, *a, **k):
    raise AssertionError(f"a test reached for the network ({url}); render the G0DM0D3 stand-in instead")


@pytest.fixture
def godmode_standin(tmp_path, monkeypatch):
    """A third-party root holding the stand-in at the pinned relative path, the pins pointed at it, fetching off."""
    from sisteer.eval import attacks_godmode as agm

    data = STANDIN_PATH.read_bytes()
    root = tmp_path / "third_party"
    path = root.joinpath(*agm.CACHE_RELPATH.parts)
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    monkeypatch.setenv(agm.THIRD_PARTY_DIR_ENV, str(root))
    monkeypatch.setenv(agm.OFFLINE_ENV, "1")
    install_standin_pins(monkeypatch, agm, data)
    monkeypatch.setattr(agm, "_open_url", refuse_network)
    return SimpleNamespace(root=root, path=path, data=data, text=STANDIN_TEXT)
