"""frozen_demo — the WORKED EXAMPLE for the generic frozen-corpus (ingested-kind)
adapter, wired end to end so the pattern is PROVEN rather than asserted.

It binds a small, committed, SYNTHETIC and BENIGN corpus
(``corpora/frozen_demo.jsonl`` — invented placeholder behaviours, safe to
commit; a REAL ingested corpus of harmful adversarial prompts would live in a
private 0700 dir, never in the tree) into a first-class held-out family via
``frozen_corpus.make_frozen_corpus_family``. It demonstrates the whole ingested-kind
path: content-hash-verified load (``PINNED_SHA``), the standard replay contract
(``run_frozen_demo`` -> ``static_common.run_static``), the leakage gate and
manifest (exercised in ``tests/test_heldout_conformance.py``).

To ingest a REAL artifact instead: run ``frozen_corpus.ingest_artifact`` on the
pinned artifact with a :class:`frozen_corpus.MappingSpec` and the training
fingerprint, then register it with ``make_frozen_corpus_family`` exactly as here.
"""

from __future__ import annotations

from pathlib import Path

from .frozen_corpus import make_frozen_corpus_family

_CORPUS = Path(__file__).resolve().parent / "corpora" / "frozen_demo.jsonl"

# The pinned content hash of the committed demo corpus — load verifies against
# it, so an accidental edit to the fixture fails loudly rather than silently
# replaying a different corpus.
PINNED_SHA = "4db5730fc942a5261045dd3447f3a934debd3793213a6c25c88cdcacfcbf31b9"

SPEC = make_frozen_corpus_family(
    "frozen_demo", _CORPUS, spec=None, expected_sha=PINNED_SHA,
    provenance={"method": "worked_example_synthetic",
                "artifact": "frozen_demo.jsonl"},
)

frozen_demo_variants = SPEC.variants_fn
run_frozen_demo = SPEC.run_fn
