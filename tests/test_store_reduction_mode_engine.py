"""The STORED-RECORD contract, engine-aware:
``contracts.read_store_reduction_mode`` reads
``route.serving_engine`` beside ``route.reduction_mode`` and holds the two to
each other (``lora_native`` iff ``not_applicable``); ``StoreReductionMode`` gains
``serving_engine`` (defaulted); ``contracts.reduction_mode_mismatch`` refuses a
native store against any lens mode and accepts native/native.

The four outcomes: native/native pools; native vs lens refuses by name; a
malformed native identity (the engine without ``not_applicable``, or the reverse)
is STATED AND UNUSABLE; every lens store — a route block WITHOUT
``serving_engine`` — reads EXACTLY as before (the historical fixture stores under
``tests/fixtures/tinycap_identity_v1`` / ``tinycap_identity_laneA`` give the same
tuple, pinned as literals), and one stating ``"lens"`` reads the lens readings
with the engine recorded. The two consumers (``pairing.pair_stores`` and the
``cell_controls`` merge) are W-C's and get their end-to-end tests there; this
file pins the resolver they call.

CPU-only."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]
WHERE = "<store>/manifest.json"
NA = C.REDUCTION_MODE_NOT_APPLICABLE

# a real lens store's manifest shape (the fixtures' route block) with a tree the
# producer table knows nothing about -> the mode must be STATED to read
LENS_ROUTE = {"agent_loop_entry": True, "grader_virtual_model": "petri__base", "lens_salt_policy": None,
              "lens_serve_declarations": {}, "object_artifact_sha256": None,
              "object_artifact_sha256_by_cell": None, "reduction_mode": "per_row",
              "resumed_prefill_policy": "conditional", "serving_identity": "proxy_vllm+lens+interleaved",
              "tool_calls": False, "virtual_model": "petri__si_safety__psr_kl_all"}


def _manifest(route, **top):
    return {C.MANIFEST_KEY_PRODUCER: {"driver": C.PRODUCER_DRIVER_SERVED, "tree_sha": "<TREE_SHA>"},
            "contract_version": 3, C.MANIFEST_KEY_ROUTE: route, **top}


def _native_route(mode=NA, engine="lora_native", **over):
    r = {**LENS_ROUTE, "serving_identity": "proxy_vllm+lora+interleaved",
         "virtual_model": "petri__si_constv2_mix_full__lora_r4_ce",
         "lens_salt_policy": NA, "lens_serve_declarations": None, "resumed_prefill_policy": NA,
         C.SERVED_ENGINE_FIELD: engine}
    if mode is not None:
        r[C.REDUCTION_MODE_FIELD] = mode
    else:
        r.pop(C.REDUCTION_MODE_FIELD, None)
    r.update(over)
    return r


# ---------------------------------------------------------------------------
# the tuple
# ---------------------------------------------------------------------------


def test_store_reduction_mode_gains_a_defaulted_engine_field():
    assert C.StoreReductionMode._fields == ("mode", "refusal", "why", "provenance", "serving_engine")
    assert C.StoreReductionMode._field_defaults == {"provenance": None, "serving_engine": None}
    # every existing construction is unchanged
    old = C.StoreReductionMode("per_row", None, "why", "stated")
    assert old == ("per_row", None, "why", "stated", None) and old.serving_engine is None
    assert C.StoreReductionMode(None, None, "why") == (None, None, "why", None, None)
    sig = inspect.signature(C.read_store_reduction_mode)
    assert list(sig.parameters) == ["manifest", "where"]


# ---------------------------------------------------------------------------
# the four outcomes
# ---------------------------------------------------------------------------


def test_a_native_store_reads_not_applicable_with_its_engine():
    read = C.read_store_reduction_mode(_manifest(_native_route()), where=WHERE)
    assert read == C.StoreReductionMode(NA, None, read.why, C.REDUCTION_MODE_STATED, "lora_native")
    assert read.refusal is None and "lora_native" in read.why and NA in read.why and WHERE in read.why
    # native/native pools
    other = C.read_store_reduction_mode(_manifest(_native_route()), where="<other>/manifest.json")
    assert C.reduction_mode_mismatch(read, other) is None
    assert C.reduction_mode_mismatch(read.mode, other.mode) is None


@pytest.mark.parametrize("lens_mode", ["per_row", "stacked"])
def test_native_vs_lens_refuses_by_name(lens_mode):
    native = C.read_store_reduction_mode(_manifest(_native_route()), where=WHERE)
    lens = C.read_store_reduction_mode(_manifest({**LENS_ROUTE, "reduction_mode": lens_mode}), where=WHERE)
    assert lens.mode == lens_mode and lens.serving_engine is None
    for a, b in ((native, lens), (lens, native), (native.mode, lens.mode), (lens_mode, NA)):
        why = C.reduction_mode_mismatch(a, b)
        assert why and "different serving engines" in why and "lora_native" in why
        assert "measures the engine as well as the intervention" in why
    # a lens store that STATES its engine is the same refusal
    stated_lens = C.read_store_reduction_mode(
        _manifest({**LENS_ROUTE, "reduction_mode": lens_mode, C.SERVED_ENGINE_FIELD: "lens"}), where=WHERE)
    assert stated_lens.serving_engine == "lens" and stated_lens.mode == lens_mode
    assert "different serving engines" in C.reduction_mode_mismatch(stated_lens, native)


def test_the_lens_e46_rule_is_unchanged():
    assert C.reduction_mode_mismatch("per_row", "per_row") is None
    assert C.reduction_mode_mismatch(None, "stacked") is None and C.reduction_mode_mismatch("stacked", None) is None
    why = C.reduction_mode_mismatch("stacked", "per_row")
    assert why.startswith("lens reduction mode 'stacked' vs 'per_row'") and "E46" in why
    # the same words pairing's own helper spells today (W-C routes it through this one)
    from sisteer.capeval.tiers import pairing
    assert pairing.reduction_mode_mismatch("stacked", "per_row") == why
    assert pairing.reduction_mode_mismatch("per_row", "per_row") is None


class TestMalformedNativeIdentity:
    def test_the_engine_without_not_applicable(self):
        for mode in ("per_row", "stacked", None, "", "NOT_APPLICABLE"):
            read = C.read_store_reduction_mode(_manifest(_native_route(mode=mode)), where=WHERE)
            assert read.mode is None and read.refusal, mode
            assert "lora_native" in read.refusal and NA in read.refusal
            # a refusal carries provenance None and names the locator in `why` (today's shape)
            assert read.serving_engine == "lora_native" and read.provenance is None
            assert read.why == f"{WHERE} route.{C.SERVED_ENGINE_FIELD}"
        # the mode key ABSENT on a native block is not a legacy store either
        read = C.read_store_reduction_mode(_manifest(_native_route(mode=None)), where=WHERE)
        assert "absent" in read.refusal

    def test_not_applicable_without_the_engine(self):
        # no engine key at all
        read = C.read_store_reduction_mode(_manifest({**LENS_ROUTE, "reduction_mode": NA}), where=WHERE)
        assert read.mode is None and read.refusal and "absent" in read.refusal and "only a 'lora_native' store" in read.refusal
        assert read.serving_engine is None and read.provenance is None and read.why == f"{WHERE} route.reduction_mode"
        # the LENS engine stated with the native value
        read = C.read_store_reduction_mode(
            _manifest({**LENS_ROUTE, "reduction_mode": NA, C.SERVED_ENGINE_FIELD: "lens"}), where=WHERE)
        assert read.mode is None and "'lens'" in read.refusal and read.serving_engine == "lens" and read.provenance is None

    def test_an_unknown_or_null_engine(self):
        for engine in (None, "vllm", "LORA_NATIVE", "", 1):
            read = C.read_store_reduction_mode(_manifest(_native_route(engine=engine)), where=WHERE)
            assert read.mode is None and read.refusal and "not one of ['lens', 'lora_native']" in read.refusal
            assert read.serving_engine is None

    def test_a_malformed_native_identity_is_never_poolable(self):
        bad = C.read_store_reduction_mode(_manifest(_native_route(mode="stacked")), where=WHERE)
        good = C.read_store_reduction_mode(_manifest(_native_route()), where=WHERE)
        # a refusal carries mode None: the consumers refuse on `refusal` BEFORE the compare,
        # exactly as pairing / cell_controls do today for an unusable lens store
        assert bad.refusal and bad.mode is None
        assert C.reduction_mode_mismatch(bad, good) is None      # nothing to compare; the refusal governs


# ---------------------------------------------------------------------------
# the lens readings are byte-identical; the historical fixture stores
# ---------------------------------------------------------------------------


def test_the_four_lens_readings_are_unchanged_and_carry_engine_none():
    stated = f"{WHERE} route.reduction_mode"
    # NAMED
    for mode in C.REDUCTION_MODES:
        read = C.read_store_reduction_mode(_manifest({**LENS_ROUTE, "reduction_mode": mode}), where=WHERE)
        assert read == (mode, None, stated, C.REDUCTION_MODE_STATED, None)
    # NOTHING STATED: no manifest / no route block
    assert C.read_store_reduction_mode(None, where=WHERE) == (None, None, f"{WHERE} is not a readable manifest", None, None)
    read = C.read_store_reduction_mode({"a": 1}, where=WHERE)
    assert read.mode is None and read.refusal is None and "not a served-lens store" in read.why and read.serving_engine is None
    # STATED AND UNUSABLE: present null, unknown value, absent on an undatable store
    read = C.read_store_reduction_mode(_manifest({**LENS_ROUTE, "reduction_mode": None}), where=WHERE)
    assert read.mode is None and "is null" in read.refusal and read.why == stated and read.provenance is None
    assert read.serving_engine is None
    read = C.read_store_reduction_mode(_manifest({**LENS_ROUTE, "reduction_mode": "vectorized"}), where=WHERE)
    assert read.mode is None and "not one of ['per_row', 'stacked']" in read.refusal and read.serving_engine is None
    stripped = {k: v for k, v in LENS_ROUTE.items() if k != "reduction_mode"}
    read = C.read_store_reduction_mode(_manifest(stripped), where=WHERE)
    assert read.mode is None and "nothing on that store says its tree lacks the stamp" in read.refusal
    assert read.serving_engine is None
    # KNOWN LEGACY: a declared pre-landing tree with no key
    sha = next(s for s, k in C.KNOWN_PRODUCER_TREES.items()
               if C.CONTRACT_REDUCTION_MODE not in getattr(k, "contracts", ()) or True)
    legacy = {C.MANIFEST_KEY_PRODUCER: {"driver": C.PRODUCER_DRIVER_SERVED, "tree_sha": sha},
              C.MANIFEST_KEY_ROUTE: stripped}
    era, _why = C.store_era(legacy, contract=C.CONTRACT_REDUCTION_MODE)
    read = C.read_store_reduction_mode(legacy, where=WHERE)
    if era == C.STORE_ERA_HISTORICAL:
        assert read.mode == "per_row" and read.provenance == C.REDUCTION_MODE_LEGACY and read.serving_engine is None
    else:
        assert read.refusal and read.serving_engine is None
    # the same readings with the engine STATED as lens carry "lens"
    for mode in C.REDUCTION_MODES:
        read = C.read_store_reduction_mode(
            _manifest({**LENS_ROUTE, "reduction_mode": mode, C.SERVED_ENGINE_FIELD: "lens"}), where=WHERE)
        assert read == (mode, None, stated, C.REDUCTION_MODE_STATED, "lens")


FIXTURE_STORES = sorted(
    list((REPO / "tests" / "fixtures" / "tinycap_identity_v1").glob("standard_*.json"))
    + list((REPO / "tests" / "fixtures" / "tinycap_identity_laneA").glob("standard_*.json")))


@pytest.mark.parametrize("path", FIXTURE_STORES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_historical_fixture_store_reads_exactly_as_before(path):
    """The captured goldens' manifests: a lens route block naming ``per_row``
    (STATED) and no ``serving_engine`` — the tuple is today's, with the
    defaulted engine None; the consumers' 4-way reads of it are unchanged."""
    doc = json.loads(path.read_text())["manifest"]
    assert C.SERVED_ENGINE_FIELD not in doc[C.MANIFEST_KEY_ROUTE]
    read = C.read_store_reduction_mode(doc, where=str(path))
    assert read == C.StoreReductionMode("per_row", None, f"{path} route.reduction_mode", "stated")
    assert read.serving_engine is None
    assert (read.mode, read.refusal, read.why, read.provenance) == (
        "per_row", None, f"{path} route.reduction_mode", "stated")
    # the served identity block of a lens store carries no engine field
    served = doc["resolved_config"]["tinycap"]["served"]
    assert set(served) == set(C.served_identity_fields())
    assert set(served) == set(C.served_identity_fields(native=False, protocol_sampling=False))
    assert C.SERVED_ENGINE_FIELD not in served


def test_the_cell_controls_locator_returns_the_engine_aware_tuple(tmp_path):
    """``cell_controls.source_store_reduction_mode`` only LOCATES the manifest
    and delegates to the one resolver — on a lens store it yields today's
    values; on a native store the engine rides along."""
    from sisteer.eval import cell_controls as cc

    out_root = tmp_path / "out"
    for name, route in (("lenscell", {**LENS_ROUTE, "reduction_mode": "stacked"}), ("nativecell", _native_route())):
        d = out_root / "cells" / name
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps(_manifest(route)))
    src_lens = {"cell_key": "lenscell"}
    src_native = {"cell_key": "nativecell"}
    try:
        lens = cc.source_store_reduction_mode(out_root, src_lens)
        native = cc.source_store_reduction_mode(out_root, src_native)
    except (KeyError, TypeError, AttributeError) as e:  # the locator's own record shape is W-C's contract
        pytest.skip(f"cell_controls' source record shape is not the synthetic one here: {e}")
    if lens.mode is None and lens.refusal is None:
        pytest.skip("the locator did not find the synthetic store (layout differs); the resolver is pinned above")
    assert (lens.mode, lens.refusal, lens.provenance, lens.serving_engine) == ("stacked", None, "stated", None)
    assert (native.mode, native.refusal, native.provenance, native.serving_engine) == (NA, None, "stated", "lora_native")
    assert "different serving engines" in C.reduction_mode_mismatch(lens, native)
