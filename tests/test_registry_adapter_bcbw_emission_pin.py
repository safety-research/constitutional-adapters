"""The four 200-item bcbw readers (`bigcodebench_allmodels`, `bfcl_allmodels`, `bigcodebench_seedrep`, `bfcl_seedrep`) emit BYTE FOR BYTE what they emitted
at the last deliberate change (2026-10-03: the anchor currency on every contrast, commit 7 of the adapter re-anchoring plan; before that 2026-09-24, when the
full-suite readers were added): every record of every type over the synthetic mirrors of `_bcbw_allmodels_fixtures` (the canary
row, the roster set, the launch + carried table, a second family) and `_bcbw_seedrep_fixtures` (the paired tag's own `none`, the launch + carried table)
hashes to the value pinned here, which was computed on the base tree 429ed730c (the rosters branch) and equal on this one -- the shared reading's new
item-count and item-set knobs (`TaskSpec.n_items` / `item_set_id` / `item_set_label` / `item_set_pin_file` / `launch_set_eval`) sit at their 200-item
defaults on these specs and change nothing they file. The pin is over the records with their paths normalised (the fixture root and the test's temp
directory) and WITHOUT `record_id` (`identity.adapter_record_id` hashes the payload, so it moves with the temp path; the payload is what is compared),
plus the `source_status` extras and the `source_state` rows. A legitimate emission change to these readers moves the hash AND their EMISSION_VERSION
together: re-pin here in the same commit, with the reason in the reread ledger (`tests/test_registry_adapter_reread.py`). A move of the
synthetic CONTEXT alone (the envelope's producer block, which the pin also hashes) re-pins here with the reason beside the pins and bumps
nothing: the live harvester must not re-read a source because a test constant changed."""
from __future__ import annotations

import hashlib
import json
import os

import pytest

import _bcbw_allmodels_fixtures as Y
import _bcbw_seedrep_fixtures as X
from sisteer.registry.adapters import bfcl_allmodels as BFA
from sisteer.registry.adapters import bfcl_seedrep as BF
from sisteer.registry.adapters import bigcodebench_allmodels as BCA
from sisteer.registry.adapters import bigcodebench_seedrep as BC
from sisteer.registry.adapters import lineage_common as LC

#: n_records / n_states beside each pin.
#: Re-pinned with the site rename, with NO EMISSION_VERSION bump and no ledger line: the readers derive a record's substrate and one
#: field-confidence phrase from the site's labels (cluster_a / cluster_b / grading_host now), and the synthetic producer's substrate moved
#: with them. Every reader built on lineage_common moved the same way, and the registry's fold maps a record written under the site's older
#: names through its alias table, so the records the live registry holds keep their identities and a re-read would change none of them.
PINNED = {
    "bigcodebench_allmodels": ("f814a9e6c522167b17d0d4953afffbf13fc3836d3e9b5b08138c81f41375c4f6", 2471, 6),
    "bfcl_allmodels": ("c90cd541b2bd10d97e27e1e1f665259f133682c52f0b731d5c751c02a54d4dd3", 2059, 4),
    "bigcodebench_seedrep": ("645a39787311a72a4696cfb37b42cb154fe3ce84fe4c1aa2ed2d3c2b53f71400", 2471, 4),
    "bfcl_seedrep": ("bd0ec51ab00e3f5e4eeed9e600fe0abe91783b50e979fe73fdbea98bfc796be1", 2471, 4),
}
READERS = {"bigcodebench_allmodels": (BCA.BigcodebenchAllmodelsAdapter, "bigcodebench", "allmodels"), "bfcl_allmodels": (BFA.BfclAllmodelsAdapter, "bfcl", "allmodels"),
           "bigcodebench_seedrep": (BC.BigcodebenchSeedrepAdapter, "bigcodebench", "seedrep"), "bfcl_seedrep": (BF.BfclSeedrepAdapter, "bfcl", "seedrep")}


@pytest.fixture(scope="module")
def seeds():
    return LC.Seeds.locate()


def emission(name: str, tmp_path, seeds):
    """(canonical sha256, n_records, n_states) of one reader's emission over its fixture; paths normalised, record ids dropped."""
    cls, eval_id, fixt = READERS[name]
    d = tmp_path / name
    d.mkdir()
    if fixt == "allmodels":
        fx = Y.mirror(d, seeds, eval_id=eval_id, root=str(d / "mirror"), llama_none=True, roster_set=True, launch_layout="carried")
        ad = Y.adapter(cls, fx, seeds)
    else:
        fx = X.mirror(d, seeds, eval_id=eval_id, root=str(d / "mirror"), paired_none=True, launch_layout="carried")
        ad = X.adapter(cls, fx, seeds)
    inputs = ad.discover()
    groups = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), []).append(i)
    conn = X.primed_conn(ad)          # the steady state after the fold (emission 3 / 2: the anchor currency decides every contrast)
    recs = []
    for g in sorted(groups):
        for batch in ad.read(X.ctx(conn), g, groups[g]):
            recs.extend({k: v for k, v in r.items() if k != "record_id"} for r in batch.records)
        recs.append({"status_extra": ad.status_extra(g)})
    states = [s.__dict__ for s in ad.source_states(X.ctx(), inputs)]
    txt = json.dumps({"records": recs, "states": states}, sort_keys=True, separators=(",", ":"), default=str)
    txt = txt.replace(fx["root"], "<ROOT>").replace(str(d), "<TMP>").replace(str(tmp_path), "<TMP>")
    return hashlib.sha256(txt.encode()).hexdigest(), len(recs), len(states)


@pytest.mark.parametrize("name", sorted(PINNED))
def test_the_200_item_readers_emit_byte_for_byte_what_they_emitted_before_the_full_readers(name, tmp_path, seeds):
    got = emission(name, tmp_path, seeds)
    assert got == PINNED[name], (name, got, "the emission moved: a deliberate change re-pins here with an EMISSION_VERSION bump and a ledger line; an accidental one is a regression")
