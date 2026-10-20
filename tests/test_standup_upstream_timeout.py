"""``capctrl_serving_up.py --upstream-timeout-s`` — the lens proxy's
UPSTREAM (proxy -> vLLM) timeout, threaded by the REAL ``bring_up`` into the
``HTTPBackend`` the RoutingProxy is built over and recorded in endpoint.json,
so the tinycap launcher can hold the proxy to the driver's 7,200 s request
budget (``contracts.PROXY_UPSTREAM_TIMEOUT_S``). Without the flag the backend
is built exactly as before and the proxy's own 600 s default stands (every
other caller of ``bring_up``); ``proxy.py`` is not changed.

Executes the standup's real bring-up on the pinned Nemotron config through
``test_serve_declarations_attestation``'s harness (only the GPU launch, the
upstream client and the aiohttp thread are replaced) — skips when the pinned
tokenizer is not in the local HF cache, never fetches.

main's version of this file (bcb-bfcl-200 §S5, 2026-09-10) pinned the same
contract by reading ``bring_up``'s source, having no stubbed harness; its
parser and ``Standup``-record seams are kept below, its source regex is not
(the real bring-up above holds the same seam).
"""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_serve_declarations_attestation import (
    NEMOTRON,
    _att,
    _cached_tok,
    _load,
    _RecordingBackend,
    _stubbed_bring_up,
)

from sisteer.capeval.tinycap import contracts as C


@pytest.fixture
def capctrl():
    return _load("_upstream_timeout_capctrl_serving_up", "scripts/capctrl_serving_up.py")


def test_bring_up_threads_the_upstream_timeout_into_the_backend_and_records_it(capctrl, tmp_path, monkeypatch):
    _cached_tok(*NEMOTRON)
    _RecordingBackend.instances.clear()
    st, proxy = _stubbed_bring_up(capctrl, tmp_path / "given", monkeypatch, attest=_att({}, {}),
                                  upstream_timeout_s=C.PROXY_UPSTREAM_TIMEOUT_S)
    backend = _RecordingBackend.instances[-1]
    assert proxy._backend is backend                       # the backend the proxy actually forwards through
    assert backend.timeout_s == 7200.0 == C.PROXY_UPSTREAM_TIMEOUT_S
    assert st.upstream_timeout_s == 7200.0
    assert st.endpoint()[C.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S] == 7200.0


def test_without_the_flag_the_backend_is_built_as_before_and_the_record_says_null(capctrl, tmp_path, monkeypatch):
    _cached_tok(*NEMOTRON)
    _RecordingBackend.instances.clear()
    st, proxy = _stubbed_bring_up(capctrl, tmp_path / "left", monkeypatch, attest=_att({}, {}))
    backend = _RecordingBackend.instances[-1]
    assert proxy._backend is backend and backend.timeout_s is None      # no timeout_s kwarg: proxy.py's default
    assert st.upstream_timeout_s is None
    assert st.endpoint()[C.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S] is None
    assert inspect.signature(capctrl.bring_up).parameters["upstream_timeout_s"].default is None


def test_main_threads_the_parsed_flag_into_bring_up(capctrl):
    """``main`` hands the parsed flag to ``bring_up`` (the parser -> float read-back
    is exercised by the launcher's argv test); a bare argv leaves it None."""
    assert "upstream_timeout_s=args.upstream_timeout_s" in inspect.getsource(capctrl.main)
    base = ["--repo-dir", "r", "--config", "c", "--cell", "none", "--defenses-dir", "d",
            "--out", "o", "--stop-file", "s"]
    assert capctrl.build_parser().parse_args(base).upstream_timeout_s is None
    assert capctrl.build_parser().parse_args(
        base + [C.FLAG_UPSTREAM_TIMEOUT_S, f"{C.PROXY_UPSTREAM_TIMEOUT_S:g}"]).upstream_timeout_s == 7200.0
    assert C.FLAG_UPSTREAM_TIMEOUT_S == "--upstream-timeout-s"


# The Standup record's own seam (main's version of this file, bcb-bfcl-200 §S5):
# the value or null, as a dataclass default a pre-knob Standup still constructs with.
def _standup(capctrl, **over):
    """A Standup record with every required field filled by a placeholder — the
    endpoint() writer is what is under test."""
    base = dict(out=Path("/tmp/x"), model="synthetic-org/synthetic-model", revision="0" * 40,
                cell="none", harness_cell="none",
                geometry=SimpleNamespace(max_model_len=16384, max_num_seqs=36, as_dict=lambda: {}),
                launch=SimpleNamespace(base_url="http://127.0.0.1:1/v1", api_key=None, port=1, host="127.0.0.1"),
                server=None, proxy_loop=None, proxy_url="http://127.0.0.1:2/v1",
                table_path=Path("/tmp/x/routing_table.json"), tokenizer=None)
    base.update(over)
    return capctrl.Standup(**base)


def test_the_standup_record_carries_the_value_or_null(capctrl):
    given = _standup(capctrl, upstream_timeout_s=7200.0)
    left = _standup(capctrl)
    assert given.upstream_timeout_s == 7200.0 and left.upstream_timeout_s is None
    assert given.endpoint()[C.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S] == 7200.0
    assert left.endpoint()[C.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S] is None
    assert C.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S == "proxy_upstream_timeout_s"
    # the field is a dataclass default, so a Standup built before the knob still constructs
    assert capctrl.Standup.__dataclass_fields__["upstream_timeout_s"].default is None
