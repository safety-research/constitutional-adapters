"""The target-decode PIN's resolver, exercised against the REAL corpus -- a standing count check.

Two peer findings on real data (2026-09-16), both invisible to unit tests because the stores they failed on are
real data whose shape nobody had asserted: (1) the pin's first resolver chain was INERT on all 5,117 shard stores
under the ``transcripts_*`` roots -- each rung true only of data shaped after it; (2) its successor was CREDULOUS:
it established the four canonical Qwen 4,096 anchors' wire (temperature 1.0, nothing else) and would have re-sent
those bytes over a standup whose fill of the omitted keys has since moved (e631261d: Qwen3.5's
--override-generation-config) -- the request reproduced, the engine values not. A table invites a reader to
eyeball it; these COUNT assertions fail the build when the chain goes inert, or credulous, again. Runs only where
the corpus is (the grading host; ``$SISTEER_TRANSCRIPTS_ROOT`` re-points it) and SKIPS BY NAME elsewhere -- CI's Modal
suite holds no stores, and a skip there is stated, not silent. ~40 s over the corpus.

The honest after-table (2026-09-16, 5,117 shard stores): ESTABLISHED but FILL UNRECORDED 733 (Qwen 616 -- every
protected-anchor shard, `explicit` 1.0 -- plus GLM 117: their omitted server-defaultable keys are pinned by no
shipped file, so a pin refuses rather than re-send the bytes); AMBIGUOUS 1,691 (gemma 702, nemotron 496, llama 208,
GLM 285: the values are the shipped file's, which generation_config, an explicit temperature with the file's
extras and the vendor preset all reproduce -- no value read separates them; the DEFAULT reproduces them); vendor
666 (llama's lane); record-less 2,027 (pre-2026-09-10 512-cap stores whose manifests carry no decode at all; a
replay states ``--target-greedy``). PIN-ABLE: 0. No existing store records the standup's fill of the keys it
omitted, and every one omitted at least ``min_p`` / ``repetition_penalty``; the pin becomes usable once
``served_target.json`` records that fill.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))

import anchor_decode_sweep as ads  # noqa: E402

ROOT = ads.DEFAULT_ROOT
pytestmark = pytest.mark.skipif(
    not ROOT.is_dir() or not any(p.name.startswith("transcripts_") for p in ROOT.iterdir()),
    reason=f"the transcripts_* corpus is not on this machine ({ROOT}); set SISTEER_TRANSCRIPTS_ROOT to run the "
           "standing pin-resolver corpus check")


@pytest.fixture(scope="module")
def rows():
    return ads.sweep(ROOT)


def test_the_corpus_is_the_corpus(rows):
    assert len(rows) >= 5000, len(rows)          # 5,117 on 2026-09-16; stores only accrue


def test_every_protected_qwen_anchor_shard_is_established_and_refuses_a_pin_for_the_unrecorded_fill_only(rows):
    """The first chain refused these 112 shards outright (inert); the second applied to them (credulous). The truth:
    every shard ESTABLISHES `explicit` temperature 1.0 from its values (no refusal of any kind), and every shard
    OMITTED top_p / top_k / min_p / repetition_penalty, which Qwen ships no file to pin -- so none is pin-able, for
    that one reason, and the per-store view agrees across all shards."""
    prot = [r for r in rows if r["store_root"] in ads.PROTECTED_QWEN_ANCHOR_ROOTS]
    assert len(prot) >= 48, len(prot)             # 112 shards over the four roots on 2026-09-16
    refused = [r for r in prot if r["outcome"].startswith("REFUSED")]
    assert refused == [], [(r["store_root"], r["view"], r["attack"], r["cell"], r["shard"], r["outcome"]) for r in refused][:5]
    assert collections.Counter(ads.classify(r) for r in prot) == {"established_fill_unrecorded": len(prot)}
    for r in prot:
        rec = r["record"]
        assert rec["mode"] == "explicit" and rec["temperature"] == 1.0 and rec["extras"] is None, (r["store_root"], r["shard"], rec["wire"])
        assert rec["fill_reproducible"] is False and rec["omitted_unpinned"] == ["top_p", "top_k", "min_p", "repetition_penalty"]
        assert rec["shipped_pin_state"] == "ships_none"
    st = ads.store_table(prot)
    assert st and all(len(v["outcomes"]) == 1 and not v["pinable"] for v in st.values()), st


def test_no_existing_store_is_pin_able_because_no_store_records_its_server_fill(rows):
    """The honest count today. A pin-able shard appearing here means either a store now records its fill (the
    durable fix landed: update this test to assert the new truth) or the resolver started trusting an unrecorded
    fill again (the credulous defect): find out which before touching the assertion."""
    pinable = [r for r in rows if r["pinable"]]
    assert pinable == [], collections.Counter((r["store_root"], r["outcome"]) for r in pinable).most_common(5)
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        by[r["family"]][ads.classify(r)] += 1
    assert by["qwen35_4b"]["established_fill_unrecorded"] > 0
    # gemma / llama / nemotron anchors decoded at their shipped file are AMBIGUOUS by construction, never pin-able:
    # a resolver that starts naming a mode for them has reintroduced the inference
    for fam in ("gemma4e4b", "llama8b", "nemotron3nano4b"):
        assert by[fam]["pinable"] == 0 and by[fam]["established_fill_unrecorded"] == 0, (fam, dict(by[fam]))
        assert by[fam]["ambiguous"] > 0 or by[fam]["vendor"] > 0, (fam, dict(by[fam]))


def test_no_shard_refuses_for_a_reason_other_than_a_recordless_or_inconsistent_store(rows):
    """Refusals at the shard level are the two fail-closed reasons only: no decode record anywhere (pre-2026-09-10
    stores), or records that disagree. Any other refusal text is a resolver defect, not a data property. (The fill
    refusal is a PIN-level refusal on an established store and is classified, not counted here.)"""
    other = collections.Counter(r["outcome"] for r in rows
                                if r["outcome"].startswith("REFUSED") and not any(k in r["outcome"] for k in ads.REFUSED_KNOWN))
    assert not other, other
