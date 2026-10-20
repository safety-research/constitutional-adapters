"""The Petri proxy PORT TABLE (petri_proxy.sbatch) and its mirror in the grading host's driver
(petri_drive_full22.sh) — one table, parsed from both scripts and held equal (extends
tests/test_shortconst_view_mechanism's single-entry pin to the whole table).

* every (TAG -> port) entry the proxy names, the driver names with the same value, and vice versa.
* ports are distinct, in the 89xx block, and the +100 (overflow re-run) / +200 (second pass) offsets of every base port
  collide with no base port.
* the seed-replicate views (`qwen35_4b_s1|s2|s3`) sit at 8978 / 8979 / 8980 — 8991 / 8992 / 8993 were NOT free
  (8991 gemma4e4b_halfdose, 8993 qwen35_4b_halfdose, local 18992 already in use), and 8990 was kept clear because
  18990 was the auditor chain's abliterated-Qwen tunnel port in the driver -- a leg that has since been removed, so
  the driver exports no such endpoint any more; the table's 8990 gap simply stays.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
F22 = REPO / "scripts" / "sdfbase" / "full22"
PROXY = (F22 / "petri_proxy.sbatch").read_text()
DRIVER = (F22 / "petri_drive_full22.sh").read_text()

SEEDREP_PORTS = {"qwen35_4b_s1": 8978, "qwen35_4b_s2": 8979, "qwen35_4b_s3": 8980}


def _table(text: str, var: str) -> dict:
    return {tag: int(port) for tag, port in re.findall(rf"([a-z0-9_]+)\) {var}=(\d+);;", text)}


def test_the_two_port_tables_are_one_table():
    proxy, driver = _table(PROXY, "PROXY_PORT"), _table(DRIVER, "PPORT")
    assert proxy and driver
    assert proxy == driver, {"proxy_only": sorted(set(proxy) - set(driver)), "driver_only": sorted(set(driver) - set(proxy)),
                             "differ": {t: (proxy[t], driver.get(t)) for t in proxy if driver.get(t) != proxy[t]}}


def test_ports_are_distinct_in_block_and_offset_safe():
    proxy = _table(PROXY, "PROXY_PORT")
    ports = list(proxy.values())
    assert len(set(ports)) == len(ports), "two tags share a port"
    assert all(8900 <= p < 9000 for p in ports), ports
    base = set(ports)
    assert not (base & {p + 100 for p in ports}), "an overflow re-run port (+100) collides with a base port"
    assert not (base & {p + 200 for p in ports}), "a second-pass port (+200) collides with a base port"
    # the re-run / second-pass offsets ride the same arithmetic on both sides
    assert 'PROXY_PORT=$((PROXY_PORT + 100))' in PROXY and 'PROXY_PORT=$((PROXY_PORT + 200))' in PROXY
    assert 'PPORT=$((PPORT + 100))' in DRIVER and 'PPORT=$((PPORT + 200))' in DRIVER


def test_the_seed_replicate_views_have_their_ports_and_the_taken_values_stayed_taken():
    proxy = _table(PROXY, "PROXY_PORT")
    for tag, port in SEEDREP_PORTS.items():
        assert proxy[tag] == port, (tag, proxy.get(tag))
    # the pre-existing assignments the plan's suggested values would have collided with are untouched
    assert proxy["gemma4e4b_halfdose"] == 8991 and proxy["qwen35_4b_halfdose"] == 8993
    assert 8992 not in proxy.values() and 8990 not in proxy.values()
    # the abliterated-Qwen auditor leg is gone, so the driver EXPORTS no
    # PETRI_QWEN_AUDITOR_BASE_URL and defaults no 18990 tunnel for it -- only its removal note remains
    assert re.search(r"^export PETRI_QWEN_AUDITOR_BASE_URL", DRIVER, re.M) is None
    assert re.search(r"PETRI_QWEN_AUDITOR_BASE_URL:-http://", DRIVER) is None
    assert "abliterated-Qwen third leg was removed from the chain" in DRIVER
    # the driver derives the proxy tunnel's local port as base + 10000, the tunnel's other end; no
    # base port maps onto another base port's tunnel
    assert "LOCALPORT=$((PPORT + 10000))" in DRIVER
    assert not ({p + 10000 for p in proxy.values()} & set(proxy.values()))


def test_the_seed_replicate_tags_are_what_the_roster_grammar_and_the_driver_derive():
    # TAG = KEY + "_" + VIEW for a non-main view (the driver's rule), so `qwen35_4b` + `s1` -> `qwen35_4b_s1`
    assert 'TAG="$KEY$( [ "$VIEW" = main ] || echo "_$VIEW" )"' in DRIVER
    for s in ("s1", "s2", "s3"):
        assert f"qwen35_4b_{s}" in _table(PROXY, "PROXY_PORT")
