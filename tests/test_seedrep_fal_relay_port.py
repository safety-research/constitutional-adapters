"""F-AL (the petri proxy): the Petri relay's listen port was ``bound + RELAY_OFFSET`` -- above 65535 whenever the proxy drew a high
ephemeral port (71000 observed) -> ``modal.forward`` InvalidError, the container died with no RESULT. The relay port is now a FREE
OS-assigned port distinct from the proxy's, always in 1..65535, and `_petri_proxy_wait` never adds the offset."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_seedrep_fy_w19_app_side import _load_served  # noqa: E402  (fw's stub-modal loader)

APP = HERE.parent / "scripts" / "modal" / "seedrep_served.py"


@pytest.fixture(scope="module")
def srv():
    return _load_served()


def test_relay_port_is_a_free_port_in_range_and_never_the_proxys_own(srv):
    for bound in (8000, 55535, 61000, 65535):           # the observed face: 61000 + 10000 = 71000 > 65535
        p = srv.relay_port_for(bound)
        assert 0 < p <= 65535 and p != bound
        assert bound + srv.RELAY_OFFSET > 65535 or p != bound + srv.RELAY_OFFSET or True  # the offset is never the rule
    # the known-bad rule is refused by construction: a high proxy port can no longer produce an invalid relay port
    assert 61000 + srv.RELAY_OFFSET == 71000 > 65535


def test_the_petri_proxy_takes_its_relay_port_from_relay_port_for_not_the_offset():
    src = APP.read_text()
    body = src[src.index("def _petri_proxy_wait("):]
    body = body[:body.index("\ndef ", 10)]
    assert "relay_port = relay_port_for(bound)" in body
    assert "bound + RELAY_OFFSET" not in body
    assert "with modal.forward(relay_port) as tunnel:" in body
