"""The STORED-RECORD contract at its two CONSUMERS, and the byte-identical read
of the four real fixture stores.

``contracts.read_store_reduction_mode`` is the one resolver (W-A's, pinned by
``tests/test_store_reduction_mode_engine.py``); this file exercises the two seams
that READ it and the historical stores they must keep reading unchanged:

- the four real fixture stores
  (``tests/fixtures/tinycap_identity_v1/`` and ``tinycap_identity_laneA/``) read
  through the UPDATED ``route_class_of`` / ``steered_path_of`` /
  ``arm_contract_problem`` / ``read_store_reduction_mode`` / ``pairing`` yield
  byte-identical manifests (sha256 of the canonical JSON, pinned as literals
  computed from the unchanged fixture files) and the SAME verdicts as before the
  change — only ``route_class`` reads ``"steered"`` where it read ``"lens"``, the
  accepted value move in new analysis outputs;
- ``pairing.compare_sides`` over two synthetic stores: native/native pools,
  native-vs-lens refuses BY NAME (different serving engines), a malformed native
  identity is stated-and-unusable, and lens/lens is byte-identical to before;
- the ``cell_controls`` merge: a native store's payload is stamped
  ``not_applicable`` plus its engine, a native payload never pools with a lens
  one, and every lens store's payload is byte-identical (the absent-is-per_row
  spelling is untouched).

CPU-only; every synthetic store is written here, and no fixture file is rewritten."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import tinycap_math500_thinking as M  # noqa: E402

from sisteer.capeval.tiers import pairing as P  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer.eval import cell_controls as cc  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
NATIVE, LENS = C.SERVING_ENGINE_LORA_NATIVE, C.SERVING_ENGINE_LENS
NA = C.REDUCTION_MODE_NOT_APPLICABLE

#: The four historical fixture stores and, per store, the sha256 of the canonical
#: JSON of its MANIFEST — the bytes every reader below must leave untouched. The
#: literals were computed from these very files (unchanged since origin/main), so
#: they are main's values: a reader that rewrote, normalised or re-keyed anything
#: it read would move them.
#: Re-pinned 2026-10-14: the decode recorder (59b7b368a) added one request-sampler line to each fixture
#: manifest -- the observed decode the recorder files -- without moving these pins, which are the canonical-JSON sha256 of the fixtures as
#: committed (`doc["manifest"]`); the readers below still leave every byte untouched.
#: Re-pinned again when the tiny loglikelihood battery was removed: the fixtures were re-captured (their frameworks
#: and versions blocks lost the removed harness's keys, and the client-lock sha moved), so their canonical bytes did.
FIXTURE_MANIFEST_SHA256 = {
    "tinycap_identity_v1/standard_off_gsm8k.json":
        "17085beeb540b0edf80201c6965f6c02d57139f84dbccfc8f31cb8c372d84220",
    "tinycap_identity_v1/standard_on_gemma_gsm8k.json":
        "409a0651f833b5de44c6b2073577bcd49ea78c94b6f7f2e86da84150b0a90cdb",
    "tinycap_identity_laneA/standard_off_gsm8k.json":
        "17085beeb540b0edf80201c6965f6c02d57139f84dbccfc8f31cb8c372d84220",
    "tinycap_identity_laneA/standard_on_gemma_gsm8k.json":
        "409a0651f833b5de44c6b2073577bcd49ea78c94b6f7f2e86da84150b0a90cdb",
}
#: What ``arm_contract_problem`` says about each fixture store — its geometry block
#: records an OVERRIDE window (these are identity fixtures, not geometry ones), so the
#: window rule refuses first, exactly as it did before the §1.8 branch was added. The
#: point of the pin is that the wording and the ORDER of the historical refusal are
#: unchanged.
FIXTURE_ARM_PROBLEM = ("geometry.window_source 'override' != 'table' — the served window was "
                       "not the table entry's")


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha(obj) -> str:
    return hashlib.sha256(_canon(obj).encode("utf-8")).hexdigest()


@pytest.mark.parametrize("rel", sorted(FIXTURE_MANIFEST_SHA256))
def test_a_real_lens_fixture_store_reads_byte_identically_through_every_updated_reader(rel):
    """A read of an existing lens-path store through the updated
    readers yields byte-identical rows."""
    path = FIXTURES / rel
    doc = json.loads(path.read_text())
    manifest = doc["manifest"]
    before = _sha(manifest)
    assert before == FIXTURE_MANIFEST_SHA256[rel]

    cell = manifest["harness_cell_id"]
    identity = manifest[C.MANIFEST_KEY_ROUTE]["serving_identity"]
    # the route class: "steered" where it read "lens" (§8.3 — a derived analysis value,
    # recorded in no eval store), and the PATH says which engine served it
    assert M.route_class_of(cell) == M.ROUTE_STEERED
    assert M.steered_path_of(identity) == "lens"
    # the arm contract: the same verdict, in the same words, as before the change
    assert M.arm_contract_problem(manifest, row={"cell": cell}) == FIXTURE_ARM_PROBLEM
    # the stored-record contract: a lens store, read exactly as it always was — the
    # engine is None (no key absence is ever read as native)
    reading = C.read_store_reduction_mode(manifest, where=str(path))
    assert (reading.mode, reading.refusal, reading.provenance, reading.serving_engine) == (
        "per_row", None, C.REDUCTION_MODE_STATED, None)
    # ... and through the PAIRING seam's own locator, over a store laid out on disk
    assert _sha(manifest) == before      # nothing any reader did moved a byte
    assert _sha(doc) == _sha(json.loads(path.read_text()))


def _store(root: Path, name: str, *, engine: str | None, mode: str | None,
           acc=(1.0, 0.0, 1.0, 1.0)) -> Path:
    """A capeval store on disk: ``cells/<key>/{manifest.json,rows.jsonl}``, whose route
    block states the engine and mode a producer of that era would have written."""
    cell_dir = root / name / "cells" / "cellkey"
    cell_dir.mkdir(parents=True)
    route: dict = {"virtual_model": "petri__none", "serving_identity":
                   ("proxy_vllm+lora+interleaved" if engine == NATIVE else "proxy_vllm+lens+interleaved"),
                   "agent_loop_entry": True, "tool_calls": False}
    if mode is not None:
        route[C.REDUCTION_MODE_FIELD] = mode
    if engine is not None:
        route[C.SERVED_ENGINE_FIELD] = engine
    (cell_dir / "manifest.json").write_text(json.dumps({C.MANIFEST_KEY_ROUTE: route}, indent=1))
    rows = [{"task": "gsm8k", "metric": "acc", "item_id": f"i{i}", "value": v,
             "engine": "vllm"} for i, v in enumerate(acc)]
    (cell_dir / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return cell_dir / "rows.jsonl"


def _side(label: str, rows: Path) -> P.Side:
    return P.Side(label=label, rows_path=rows, task="gsm8k", engine="vllm", metric="acc")


def test_pairing_pools_native_with_native_and_refuses_a_cross_engine_pair(tmp_path):
    native_a = _store(tmp_path, "native_a", engine=NATIVE, mode=NA)
    native_b = _store(tmp_path, "native_b", engine=NATIVE, mode=NA, acc=(1.0, 1.0, 0.0, 1.0))
    lens = _store(tmp_path, "lens", engine=None, mode="per_row")
    stacked = _store(tmp_path, "stacked", engine=None, mode="stacked")

    rep = P.compare_sides(_side("A", native_a), _side("B", native_b))
    assert rep["sides"]["a"][C.REDUCTION_MODE_FIELD] == NA
    # the engine each side states rides the report (additive; absent on a lens store)
    assert rep["sides"]["a"]["serving_engine"] == NATIVE
    assert rep["sides"]["b"]["serving_engine"] == NATIVE

    for other, label in ((lens, "per_row"), (stacked, "stacked")):
        with pytest.raises(P.PairingError) as ei:
            P.compare_sides(_side("A", native_a), _side("B", other))
        assert "different serving engines" in str(ei.value)
        # and in the other order
        with pytest.raises(P.PairingError, match="different serving engines"):
            P.compare_sides(_side("A", other), _side("B", native_a))

    # lens vs lens is untouched: the E46 wording, and a pooled pair carries NO engine key
    rep = P.compare_sides(_side("A", lens), _side("B", _store(tmp_path, "lens2", engine=None,
                                                              mode="per_row")))
    assert "serving_engine" not in rep["sides"]["a"]
    with pytest.raises(P.PairingError, match="stacked store and a per_row store"):
        P.compare_sides(_side("A", lens), _side("B", stacked))


@pytest.mark.parametrize("engine,mode", [(NATIVE, "per_row"), (None, NA), (LENS, NA)])
def test_pairing_refuses_a_malformed_native_identity_as_stated_and_unusable(tmp_path, engine, mode):
    bad = _store(tmp_path, "bad", engine=engine, mode=mode)
    ok = _store(tmp_path, "ok", engine=NATIVE, mode=NA)
    with pytest.raises(P.PairingError) as ei:
        P.compare_sides(_side("A", bad), _side("B", ok))
    assert "not_applicable" in str(ei.value) or "lora_native" in str(ei.value)


def test_the_reduction_mode_mismatch_verdicts_are_the_one_engine_aware_rule():
    """``pairing`` and ``cell_controls`` both delegate the verdict to contracts, so no
    seam can re-derive a different answer about one pair of stores again."""
    assert P.reduction_mode_mismatch(NA, NA) is None
    assert P.reduction_mode_mismatch("per_row", "per_row") is None
    for a, b in ((NA, "per_row"), ("per_row", NA), (NA, "stacked")):
        why = P.reduction_mode_mismatch(a, b)
        assert why == C.reduction_mode_mismatch(a, b) and "different serving engines" in why
    # the payload-level seam (cell_controls) inherits it, absent still reading as per_row
    assert cc.reduction_mode_mismatch({C.REDUCTION_MODE_FIELD: NA},
                                      {C.REDUCTION_MODE_FIELD: NA}) is None
    assert cc.reduction_mode_mismatch({}, {}) is None
    why = cc.reduction_mode_mismatch({C.REDUCTION_MODE_FIELD: NA}, {})
    assert why and "different serving engines" in why
    why = cc.reduction_mode_mismatch({C.REDUCTION_MODE_FIELD: "stacked"}, {})
    assert why and "stacked" in why and "different serving engines" not in why


def _merge_source(cell_key="cellkey"):
    return {"cell_key": cell_key, "tree_sha": "ab" * 20, "driver": "served_capability_run.py"}


def _payload(**over):
    payload = {"acc": 0.5, "n": 4, "task": "gsm8k"}
    payload.update(over)
    return payload


def test_the_cell_controls_merge_stamps_a_native_stores_payload_with_mode_and_engine(tmp_path):
    """The WRITE half (E49) under the engine: a native store's payload carries
    ``not_applicable`` AND ``serving_engine``; a lens store's payload is byte-identical
    (per_row stays spelled by absence, so every existing summary re-merges unchanged)."""
    native_root = tmp_path / "native"
    _store(native_root.parent, "native", engine=NATIVE, mode=NA)
    lens_root = tmp_path / "lens"
    _store(lens_root.parent, "lens", engine=None, mode="per_row")
    stacked_root = tmp_path / "stacked"
    _store(stacked_root.parent, "stacked", engine=None, mode="stacked")

    entry = C.evalplus_task("humaneval", open_variant=False)
    for root, want in ((native_root, {C.REDUCTION_MODE_FIELD: NA, C.SERVED_ENGINE_FIELD: NATIVE}),
                       (lens_root, {}),
                       (stacked_root, {C.REDUCTION_MODE_FIELD: "stacked"})):
        payloads = {entry: _payload()}
        out = cc._stamp_reduction_modes(payloads, {entry: _merge_source()}, root,
                                       root / "summary.json")
        assert out[entry] == {**_payload(), **want}, root.name

    # the READ half: a payload that contradicts its store refuses by name
    payloads = {entry: _payload(**{C.REDUCTION_MODE_FIELD: "per_row"})}
    with pytest.raises(ValueError) as ei:
        cc._validate_reduction_mode_stamps(payloads, {entry: _merge_source()}, native_root,
                                           native_root / "summary.json")
    assert "different serving engines" in str(ei.value)
    # ... and a native payload sourced from a LENS store refuses too
    payloads = {entry: _payload(**{C.REDUCTION_MODE_FIELD: NA})}
    with pytest.raises(ValueError, match="different serving engines"):
        cc._validate_reduction_mode_stamps(payloads, {entry: _merge_source()}, lens_root,
                                           lens_root / "summary.json")


def test_the_cell_controls_locator_reads_the_engine_aware_tuple(tmp_path):
    """``source_store_reduction_mode`` returns the resolver's own five-field reading —
    read BY NAME everywhere in this module, because it grew ``serving_engine``."""
    root = tmp_path / "native"
    _store(root.parent, "native", engine=NATIVE, mode=NA)
    reading = cc.source_store_reduction_mode(root, _merge_source())
    assert isinstance(reading, C.StoreReductionMode)
    assert (reading.mode, reading.refusal, reading.serving_engine) == (NA, None, NATIVE)
    # an absent store makes no claim at all (reported, never raised)
    gone = cc.source_store_reduction_mode(root, _merge_source(cell_key="absent"))
    assert gone.mode is None and gone.refusal is None and gone.serving_engine is None
