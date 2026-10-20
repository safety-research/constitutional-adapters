"""F-AR (the petri drive): the drive's local relay port was ``proxy_port + LOCAL_PORT_OFFSET`` (10000) -- above 65535 whenever the
proxy drew a high ephemeral port. The proxy published proxy_port 57204; 67204 made `local_port_held` raise OverflowError at bind
time, the drive died and the READY proxy idled to its 4 h cap. The port is now a FREE OS-assigned loopback port distinct from the
proxy's, always in LOCAL_PORT_MIN..65535, and the drive-start path (`open_local_relay`) binds it through the real `local_port_held`
+ `start_local_tls_relay`; the relay's ``listen`` -- what DRIVE_START and the drive record carry -- names the port actually bound.
The grading-host sibling of F-AL (`relay_port_for` in seedrep_served.py, tests/test_seedrep_fal_relay_port.py)."""
from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_seedrep_petri_drive import _load  # noqa: E402  (the drive's stub-modal loader)

SRC = HERE.parent / "scripts" / "modal" / "seedrep_petri_drive.py"
W23_PROXY_PORT = 57204
#: the fake tunnel record of the observed face: the proxy's own port and the container relay's, as the record carries them
W23_TUNNEL = {"proxy_port": W23_PROXY_PORT, "relay_port": 31346}


@pytest.fixture(scope="module")
def d():
    return _load()


def test_the_old_rule_overflowed_at_bind_time(d):
    assert W23_PROXY_PORT + 10000 == 67204 > 65535
    with pytest.raises(OverflowError):
        d.local_port_held(W23_PROXY_PORT + 10000)


def test_local_port_of_is_a_free_port_in_range_and_never_the_proxys_own(d):
    for bound in (8984, W23_PROXY_PORT, 61000, 65535):
        p = d.local_port_of(bound)
        assert d.LOCAL_PORT_MIN <= p <= 65535 and p != bound
        assert not d.local_port_held(p)          # free: the drive's own held-check accepts it


def test_open_local_relay_binds_the_w23_tunnel_record_and_reports_the_bound_port(d):
    local_port, relay = d.open_local_relay(W23_TUNNEL, "127.0.0.1")      # through the REAL local_port_held + start_local_tls_relay
    try:
        assert d.LOCAL_PORT_MIN <= local_port <= 65535 and local_port != W23_PROXY_PORT
        assert relay["listen"] == f"127.0.0.1:{local_port}"             # the recorded port IS the bound one
        assert "fatal" not in relay["counters"]
        assert d.local_port_held(local_port)                             # bound: the relay holds it now
        with socket.create_connection(("127.0.0.1", local_port), timeout=2):
            pass
    finally:
        relay["stop"]()


def test_open_local_relay_refuses_a_port_something_else_holds(d, monkeypatch):
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    held = holder.getsockname()[1]
    try:
        monkeypatch.setattr(d, "local_port_of", lambda proxy_port: held)
        with pytest.raises(d.DriveRefused, match=f"local port {held} is held"):
            d.open_local_relay(W23_TUNNEL, "127.0.0.1")
    finally:
        holder.close()


def test_the_drive_takes_its_local_port_from_open_local_relay_and_records_that_port():
    src = SRC.read_text()
    assert "LOCAL_PORT_OFFSET" not in src
    body = src[src.index("def main("):]
    assert "local_port, relay = open_local_relay(tunnel, tunnel_host)" in body
    assert 'base = f"http://127.0.0.1:{local_port}"' in body
    assert '"local_relay": {"listen": relay["listen"], "upstream": relay["upstream"]}' in body   # the drive record
    assert "relay={relay['listen']}->{relay['upstream']} base={base}" in body                    # DRIVE_START
