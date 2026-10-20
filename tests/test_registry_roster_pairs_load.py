"""Every committed roster/rulings pair loads strictly.

The release rewrites the free text of the committed transcripts: withheld ruling provenance becomes a fixed sentinel, and
reasons and notes are paraphrased. The loader refuses a ruling with a missing or empty required field and a post-cutoff
ruling without a valid provenance block, so loading every pair with ``strict_counts=True`` proves the rewritten transcripts
still satisfy it.
"""
from __future__ import annotations

import glob
import os
import re

import pytest

from sisteer.registry.adapters import roster as RO

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(os.path.dirname(HERE), "configs", "registry")


def _pairs():
    out = []
    for path in glob.glob(os.path.join(REGISTRY, "roster_r2_v*.yaml")):
        v = re.fullmatch(r"roster_r2_v(\d+)\.yaml", os.path.basename(path)).group(1)
        rulings = os.path.join(REGISTRY, f"rulings_r2_v{v}.jsonl")
        if os.path.exists(rulings):
            out.append((int(v), path, rulings))
    return sorted(out)


PAIRS = _pairs()


def test_every_roster_has_its_rulings():
    rosters = glob.glob(os.path.join(REGISTRY, "roster_r2_v*.yaml"))
    assert len(PAIRS) == len(rosters) >= 11


@pytest.mark.parametrize("version,roster,rulings", PAIRS, ids=[f"v{v}" for v, _, _ in PAIRS])
def test_pair_loads_strictly(version, roster, rulings):
    ld = RO.load(roster, rulings, strict_counts=True)
    assert ld.entries and ld.rulings
    for r in ld.rulings:
        for k in ("reason", "ruled_by", "source_ref"):
            assert isinstance(r[k], str) and r[k].strip(), (r["ruling_id"], k)
        prov = (r.get("extra") or {}).get("provenance")
        if prov is not None:
            assert prov["kind"] in RO.RULING_PROVENANCE_KINDS
            if prov["kind"] == "relayed":
                assert prov["relayed_by"] and prov["message_utc"]
            if prov["kind"] == "peer_encoded":
                assert prov["encoded_by"] and re.fullmatch(r"[0-9a-f]{7,40}", str(prov["commit"]))


def test_rulings_contents_stable_across_versions():
    """Every later roster version carries at least its predecessor's rulings."""
    seen = {}
    for v, roster, rulings in PAIRS:
        ld = RO.load(roster, rulings, strict_counts=True)
        seen[v] = {r["ruling_id"] for r in ld.rulings}
    versions = sorted(seen)
    for a, b in zip(versions, versions[1:]):
        assert seen[a] <= seen[b], (a, b)
