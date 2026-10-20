"""Fit-time provenance for the sdf-base lineages the served capability
driver refused (full22 controls smoke, 2026-09-07): DOCUMENT fits and
DERIVED objects.

Synthetic throughout (invented model ids, placeholder revisions), but the
record SHAPES are the real ones: the training pipeline's document-schema
``train_budget_meta.json`` (``training_budget_kind: document_tokens``,
``chat_template_bypassed: true``, ``completion_tokens: null``) and the
derived builders' ``derived_meta.json`` (top-level model identity + ≥ 2
``parent_*`` blocks carrying the parents' fit-time facts). The one real
constant used is sha256("")[:16] = e3b0c44298fc1c14 — what
``template_hash_of`` records for a tokenizer that ships NO chat template
(gemma-4-E4B base); a mathematical constant, not recorded data.

What is pinned:

* a document fit resolves with status ``document_fit`` — whether its
  tokenizer shipped no template (the empty hash) or its instruct sibling's
  (the vacuous-pass case) — with the RECORDED hash kept on the record,
  never erased or replaced;
* a legacy SI-pair fit stays ``verified`` (regression);
* a record that contradicts itself (document kind + bypassed=False, or
  bypassed=True + non-document kind) refuses;
* the served==trained guard admits a document-fit row with differing hashes,
  keeps refusing the SAME hashes under ``verified``, and refuses a
  document-fit stamp on a baseline — so the two row kinds stay
  distinguishable by ``provenance_status`` in everything downstream reads;
* a derived object resolves from its parents' AGREEING template hash
  (source ``derived_meta:<kind>``), inherits the document-fit status from
  document parents, and refuses on disagreeing parents, a self-contradicting
  top level, a broken sha binding, or a second identity sidecar;
* the driver's evidence.json spells the fit side out beside the served side
  (``fit_provenance``: transfer + whether the template check ran).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sisteer.capeval.engine_identity import EngineIdentityError
from sisteer.eval import sweep, train_budget
from sisteer.eval.defense_cell import sha256_file
from sisteer.eval.sweep import DOCUMENT_TOKENS_KIND

#: the derived-object verdict lives on the sdf-base round-2 lineage; on a tree
#: without it the reader REFUSES derived objects (pinned below), and the
#: positive derived tests cannot run
HAS_DERIVED_VERDICT = hasattr(train_budget, "derived_object_problems")
needs_derived_verdict = pytest.mark.skipif(
    not HAS_DERIVED_VERDICT,
    reason="train_budget.derived_object_problems absent on this tree")


def test_the_derived_verdict_is_on_this_tree():
    """Since 2026-09-10 (bcb-bfcl-200: the lora_subtraction port) main
    carries ``train_budget.derived_object_problems``, so the derived tests
    below RUN here — this pin keeps the skip marker from ever hiding them
    again (a skipped positive test reads as green)."""
    assert HAS_DERIVED_VERDICT

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))

EMPTY_HASH = hashlib.sha256(b"").hexdigest()[:16]   # a template-free tokenizer
SHIPPED_HASH = "1111222233334444"                    # a tokenizer that ships one
SERVED_HASH = "aaaabbbbccccdddd"                     # the instruct sibling's
BASE_MODEL, BASE_REV = "synthetic-org/tiny-base", "1" * 40
IT_MODEL, IT_REV = "synthetic-org/tiny-it", "2" * 40
CELL = "constv2-mix-full__psr_ce_all"


def _blob():
    return {"kind": "psr", "layers": [0], "positions": "all",
            "response_header_len": None,
            "directions": {"0": torch.zeros(4)}, "gate_w": {"0": torch.zeros(4)},
            "gate_b": {"0": torch.zeros(1)}}


def _doc_sidecar(obj: Path, **over) -> dict:
    """The document-schema budget sidecar, key set of the real staged copies."""
    rec = {"cell_id": CELL, "arm": "psr_ce_all", "si_id": "constv2-mix-full",
           "training_budget_kind": DOCUMENT_TOKENS_KIND, "data_mode": "documents",
           "chat_template_bypassed": True, "completion_tokens": None,
           "sequence_max_len": 8192, "sample_manifest_sha": "ab" * 32,
           "model_id": BASE_MODEL, "model_revision": BASE_REV,
           "template_hash": EMPTY_HASH, "positions": "all",
           "positions_at_fit": "all", "response_header_len": None,
           "object_sha256": sha256_file(obj)}
    rec.update(over)
    return rec


def _stage_doc(d: Path, **over) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / "object.pt"
    torch.save(_blob(), p)
    (d / "train_budget_meta.json").write_text(json.dumps(_doc_sidecar(p, **over)))
    return p


def _stage_pair_fit(d: Path) -> Path:
    """A legacy SI-pair fit (no training_budget_kind): chat-rendered."""
    d.mkdir(parents=True, exist_ok=True)
    p = d / "object.pt"
    torch.save(_blob(), p)
    (d / "train_budget_meta.json").write_text(json.dumps(
        {"cell_id": "safety_refusal__psr_kl_all", "completion_tokens": 512,
         "model_id": IT_MODEL, "model_revision": IT_REV,
         "template_hash": SERVED_HASH, "positions": "response",
         "positions_at_fit": "response", "response_header_len": 3,
         "object_sha256": sha256_file(p)}))
    return p


def _stage_derived(d: Path, *, kind="lora_subtraction",
                   pos_hash=EMPTY_HASH, neg_hash=EMPTY_HASH,
                   top_hash=..., parent_kind=DOCUMENT_TOKENS_KIND,
                   top_model=BASE_MODEL, break_sha=False,
                   also_budget_sidecar=False) -> Path:
    """A staged DERIVED dir shaped like the real staging layout: object.pt
    whose embedded provenance names the same parents the sidecar does."""
    d.mkdir(parents=True, exist_ok=True)
    p = d / "object.pt"

    def parent(cell, obj_sha, th):
        b = {"cell_id": cell, "model_key": "tiny_base", "object_sha256": obj_sha,
             "budget_meta_sha256": "bb" * 32, "model_id": BASE_MODEL,
             "model_revision": BASE_REV, "positions": "all",
             "response_header_len": None, "target_layers": [0],
             "template_hash": th}
        if parent_kind is not None:
            b["training_budget_kind"] = parent_kind
        return b
    prov = {"kind": kind,
            "parent_pos": parent(f"constv2-mix-full__{kind}", "aa" * 32, pos_hash),
            "parent_neg": parent(f"constv2ctl2-full__{kind}", "cc" * 32, neg_hash)}
    blob = {"kind": kind, "layers": [0], "positions": "all",
            "response_header_len": None, "provenance": prov,
            "A_pos": {"0": torch.zeros(2, 1)}, "B_pos": {"0": torch.zeros(3, 1)},
            "A_neg": {"0": torch.zeros(2, 1)}, "B_neg": {"0": torch.zeros(3, 1)}}
    torch.save(blob, p)
    meta = {**prov, "derived_id": d.name, "model_key": "tiny_base",
            "model_id": top_model, "model_revision": BASE_REV, "positions": "all",
            "response_header_len": None,
            "object_sha256": ("00" * 32) if break_sha else sha256_file(p)}
    if top_hash is not ...:
        meta["template_hash"] = top_hash
    (d / "derived_meta.json").write_text(json.dumps(meta))
    (d / "staged_from.json").write_text(json.dumps({"source": "synthetic"}))
    if also_budget_sidecar:
        (d / "train_budget_meta.json").write_text(json.dumps(
            {"cell_id": d.name, "completion_tokens": 512,
             "object_sha256": sha256_file(p)}))
    return p


# --------------------------------------------------------------------------- #
# document fits                                                                #
# --------------------------------------------------------------------------- #


def test_the_empty_template_hash_constant():
    # what template_hash_of records for a tokenizer with NO chat template —
    # the value the full22 gemma base cells carry; a measurement, not a gap
    assert EMPTY_HASH == "e3b0c44298fc1c14"


def test_document_fit_with_a_template_free_tokenizer_is_document_fit(tmp_path):
    op = _stage_doc(tmp_path / CELL)
    prov = sweep.resolve_training_mirror(CELL, op)
    assert prov.status == sweep.DOCUMENT_FIT_STATUS
    assert prov.source == "train_budget_meta"
    assert prov.model_id == BASE_MODEL and prov.revision == BASE_REV
    # the RECORDED hash stays on the record — never erased, never replaced
    assert prov.trained_template_hash == EMPTY_HASH
    assert prov.training_budget_kind == DOCUMENT_TOKENS_KIND
    assert prov.chat_template_bypassed is True
    assert prov.positions_at_fit == "all"


def test_document_fit_with_a_shipped_template_is_also_document_fit(tmp_path):
    # the vacuous-pass case: the base tokenizer ships its instruct sibling's
    # template, so the recorded hash EQUALS the served one — that equality
    # is a packaging fact, not a training fact, and the row must not be read
    # as a verified template match
    op = _stage_doc(tmp_path / CELL, template_hash=SERVED_HASH)
    prov = sweep.resolve_training_mirror(CELL, op)
    assert prov.status == sweep.DOCUMENT_FIT_STATUS
    assert prov.trained_template_hash == SERVED_HASH


def test_legacy_pair_fit_stays_verified(tmp_path):
    op = _stage_pair_fit(tmp_path / "safety_refusal__psr_kl_all")
    prov = sweep.resolve_training_mirror("safety_refusal__psr_kl_all", op)
    assert prov.status == "verified"
    assert prov.trained_template_hash == SERVED_HASH
    assert prov.training_budget_kind is None
    assert prov.chat_template_bypassed is None
    # the kind was ESTABLISHED, from the SI-pair schema's positive marker
    assert prov.fit_kind == sweep.FIT_KIND_SI_PAIR
    assert "completion_tokens=512" in prov.fit_kind_evidence


def test_three_outcomes_a_record_with_no_marker_refuses(tmp_path):
    # neither the document kind nor the SI-pair cap: UNKNOWN, refused — never
    # read as a chat fit (which would run the template check on a guess) nor
    # as a document fit (which would waive it on one)
    d = tmp_path / "torn"; d.mkdir()
    p = d / "object.pt"; torch.save(_blob(), p)
    rec = {"cell_id": CELL, "model_id": IT_MODEL, "model_revision": IT_REV,
           "template_hash": SERVED_HASH, "positions": "response",
           "positions_at_fit": "response", "completion_tokens": None,
           "object_sha256": sha256_file(p)}
    (d / "train_budget_meta.json").write_text(json.dumps(rec))
    with pytest.raises(sweep.UnrecordedMirrorError, match="establishes NO fit kind"):
        sweep.resolve_training_mirror(CELL, p)
    # the key ABSENT entirely: the same refusal
    rec.pop("completion_tokens"); (d / "train_budget_meta.json").write_text(json.dumps(rec))
    with pytest.raises(sweep.UnrecordedMirrorError, match="establishes NO fit kind"):
        sweep.resolve_training_mirror(CELL, p)
    # an UNREADABLE kind value: refused as unknown, not treated as either
    rec["training_budget_kind"] = "documents_v9"; rec["completion_tokens"] = None
    (d / "train_budget_meta.json").write_text(json.dumps(rec))
    with pytest.raises(sweep.UnrecordedMirrorError, match="unknown training_budget_kind"):
        sweep.resolve_training_mirror(CELL, p)
    # a non-positive cap is not a marker either
    rec.pop("training_budget_kind"); rec["completion_tokens"] = 0
    (d / "train_budget_meta.json").write_text(json.dumps(rec))
    with pytest.raises(sweep.UnrecordedMirrorError, match="establishes NO fit kind"):
        sweep.resolve_training_mirror(CELL, p)


def _stage_fit_provenance(d: Path, writer: str, bind_sha: bool = False, **over) -> Path:
    """The HPO-promotion / refit lineages' record (native spelling).
    ``bind_sha`` records object_sha256 (the refit lineage does; tune_arm's
    promotion record does not)."""
    d.mkdir(parents=True, exist_ok=True)
    p = d / "object.pt"; torch.save(_blob(), p)
    rec = {"resolved_model_id": IT_MODEL, "resolved_revision": IT_REV,
           "chat_template_hash": SERVED_HASH, "writer": writer,
           "positions_at_fit": "response"}
    if bind_sha:
        rec["object_sha256"] = sha256_file(p)
    rec.update(over)
    (d / "fit_provenance.json").write_text(json.dumps(rec))
    return p


def _mutate_object(p: Path) -> None:
    """Replace the object bytes, leaving the sidecar in place — the
    interrupted-replacement / inconsistent-staged-copy state."""
    torch.save({**_blob(), "directions": {"0": torch.ones(4)}}, p)
    assert json.loads((p.parent / ("train_budget_meta.json"
                                   if (p.parent / "train_budget_meta.json").is_file()
                                   else "fit_provenance.json")).read_text())


def test_p1a_a_stale_object_sidecar_pair_is_refused_never_downgraded(tmp_path):
    # Reproduction: an old object beside a new document-fit sidecar (wrong
    # sidecar sha) used to be ACCEPTED as document_fit. Now: refused,
    # naming both shas, for the document waiver AND for an SI-pair record
    # that carries a sha.
    op = _stage_doc(tmp_path / CELL)
    _mutate_object(op)
    with pytest.raises(sweep.UnrecordedMirrorError, match="stale object/sidecar pair"):
        sweep.resolve_training_mirror(CELL, op)
    op = _stage_pair_fit(tmp_path / "pair")
    _mutate_object(op)
    with pytest.raises(sweep.UnrecordedMirrorError, match="stale object/sidecar pair"):
        sweep.resolve_training_mirror("safety_refusal__psr_kl_all", op)
    op = _stage_fit_provenance(tmp_path / "native", "candsi_refit_fits", bind_sha=True)
    _mutate_object(op)
    with pytest.raises(sweep.UnrecordedMirrorError, match="stale object/sidecar pair"):
        sweep.resolve_training_mirror("x", op)


def test_p1a_the_document_waiver_requires_the_sidecar_bound_to_the_bytes(tmp_path):
    # a document record with NO object_sha256 cannot grant the waiver: the
    # waiver skips the template check, so it may rest only on a sidecar
    # proven to describe the served object — refused, never a weaker status
    op = _stage_doc(tmp_path / CELL, object_sha256=None)
    with pytest.raises(sweep.UnrecordedMirrorError, match="not BOUND to these bytes"):
        sweep.resolve_training_mirror(CELL, op)
    # a native document record without a sha: the same refusal
    op = _stage_fit_provenance(tmp_path / "docnative", "x",
                               training_budget_kind=DOCUMENT_TOKENS_KIND,
                               completion_tokens=None)
    with pytest.raises(sweep.UnrecordedMirrorError, match="not BOUND to these bytes"):
        sweep.resolve_training_mirror("y", op)
    # a bound document record resolves; the binding is recorded
    op = _stage_doc(tmp_path / "bound")
    prov = sweep.resolve_training_mirror(CELL, op)
    assert prov.status == sweep.DOCUMENT_FIT_STATUS
    # an SI-pair record without a sha is NOT waived anything: the compared
    # path runs, so the legacy (sha-less) sidecars stay resolvable
    op = _stage_pair_fit(tmp_path / "legacy")
    meta = op.parent / "train_budget_meta.json"
    rec = json.loads(meta.read_text()); rec.pop("object_sha256"); meta.write_text(json.dumps(rec))
    assert sweep.resolve_training_mirror("safety_refusal__psr_kl_all", op).status == "verified"


def test_contradictory_markers_are_an_unknown_not_a_vote(tmp_path):
    # document kind beside a positive SI-pair cap: neither wins — refused
    op = _stage_doc(tmp_path / CELL, completion_tokens=512)
    with pytest.raises(sweep.UnrecordedMirrorError, match="contradictory markers"):
        sweep.resolve_training_mirror(CELL, op)
    assert sweep.fit_kind_of(DOCUMENT_TOKENS_KIND, 512)[0] is None


def test_fit_provenance_records_establish_their_kind_by_producer(tmp_path):
    # the native spelling carries no budget schema: a writer in the allowlist
    # is the positive producer marker of an SI-pair fit ...
    op = _stage_fit_provenance(tmp_path / "hpo", "tune_arm.py")
    prov = sweep.resolve_training_mirror("safety_refusal__psr_kl_all", op)
    assert prov.status == "verified" and prov.fit_kind == sweep.FIT_KIND_SI_PAIR
    assert "writer='tune_arm.py'" in prov.fit_kind_evidence
    # ... an unlisted writer establishes nothing and refuses (extend the
    # allowlist deliberately; never infer from the schema)
    op = _stage_fit_provenance(tmp_path / "other", "some_new_fitter")
    with pytest.raises(sweep.UnrecordedMirrorError, match="SI_PAIR_FIT_WRITERS"):
        sweep.resolve_training_mirror("x", op)
    # ... and a native record that DOES carry the budget schema is read by it
    # (bound to its bytes, as the waiver requires)
    op = _stage_fit_provenance(tmp_path / "docnative", "some_new_fitter", bind_sha=True,
                               training_budget_kind=DOCUMENT_TOKENS_KIND,
                               completion_tokens=None)
    prov = sweep.resolve_training_mirror("y", op)
    assert prov.status == sweep.DOCUMENT_FIT_STATUS
    assert prov.fit_kind == sweep.FIT_KIND_DOCUMENT


def test_partial_record_stays_partial(tmp_path):
    # the object's own save sidecar: no template hash, so no comparison for
    # a kind to license or waive — partial, as before, not refused as unknown
    d = tmp_path / "antidote"; d.mkdir()
    p = d / "object.safetensors"; p.write_bytes(b"obj")
    (d / "object.json").write_text(json.dumps({"model_name": IT_MODEL}))
    prov = sweep.resolve_training_mirror("antidote", p)
    assert prov.status == "partial"
    assert prov.fit_kind is None and prov.trained_template_hash is None


def test_self_contradicting_records_refuse(tmp_path):
    op = _stage_doc(tmp_path / "a", chat_template_bypassed=False)
    with pytest.raises(sweep.UnrecordedMirrorError, match="contradicts itself"):
        sweep.resolve_training_mirror(CELL, op)
    op = _stage_doc(tmp_path / "b", training_budget_kind=None,
                    completion_tokens=512)
    with pytest.raises(sweep.UnrecordedMirrorError, match="contradicts itself"):
        sweep.resolve_training_mirror(CELL, op)


def test_guard_admits_document_fit_and_keeps_refusing_verified_mismatch():
    # the SAME hashes: admitted under document_fit (inapplicable, recorded),
    # refused under verified (a real served != trained mismatch)
    doc = sweep.engine_stamp(
        target_engine="vllm", served_template_hash=SERVED_HASH,
        trained_template_hash=EMPTY_HASH, object_sha="ab" * 8,
        provenance_status=sweep.DOCUMENT_FIT_STATUS,
        provenance_source="train_budget_meta")
    sweep.assert_served_matches_trained(doc, what="document-fit row")
    verified = sweep.engine_stamp(
        target_engine="vllm", served_template_hash=SERVED_HASH,
        trained_template_hash=EMPTY_HASH, object_sha="ab" * 8,
        provenance_status="verified", provenance_source="train_budget_meta")
    with pytest.raises(EngineIdentityError, match="served==trained template"):
        sweep.assert_served_matches_trained(verified, what="verified row")
    # distinguishable by what every downstream consumer reads
    assert doc["provenance_status"] != verified["provenance_status"]
    assert doc["trained_template_hash"] == EMPTY_HASH  # recorded, not erased
    # a document-fit stamp on a baseline (no object) is malformed
    bad = sweep.engine_stamp(
        target_engine="vllm", served_template_hash=SERVED_HASH,
        provenance_status=sweep.DOCUMENT_FIT_STATUS)
    with pytest.raises(EngineIdentityError, match="malformed"):
        sweep.assert_served_matches_trained(bad, what="baseline")


def test_engine_stamp_accepts_document_fit_and_still_rejects_unknown():
    st = sweep.engine_stamp(target_engine="hf",
                            provenance_status=sweep.DOCUMENT_FIT_STATUS)
    assert st["provenance_status"] == "document_fit"
    with pytest.raises(ValueError, match="provenance_status"):
        sweep.engine_stamp(target_engine="hf", provenance_status="trusted")


# --------------------------------------------------------------------------- #
# derived objects                                                              #
# --------------------------------------------------------------------------- #


@needs_derived_verdict
def test_derived_resolves_from_the_parents_agreeing_hash(tmp_path):
    cid = "constv2-mix-full__lora_r4_ce_minus_ctl"
    op = _stage_derived(tmp_path / cid)
    prov = sweep.resolve_training_mirror(cid, op)
    assert prov.source == "derived_meta:lora_subtraction"
    assert prov.trained_template_hash == EMPTY_HASH
    assert prov.model_id == BASE_MODEL and prov.revision == BASE_REV
    assert prov.positions_at_fit == "all"
    # document parents -> a document fit (the derivation of two template-free
    # fits has no training template either)
    assert prov.status == sweep.DOCUMENT_FIT_STATUS
    assert prov.training_budget_kind == DOCUMENT_TOKENS_KIND
    assert prov.object_sha == sha256_file(op)[:16]


@needs_derived_verdict
def test_derived_top_level_hash_is_checked_against_the_parents(tmp_path):
    cid = "x__minus_ctl"
    op = _stage_derived(tmp_path / cid, pos_hash=SHIPPED_HASH,
                        neg_hash=SHIPPED_HASH, top_hash=SHIPPED_HASH)
    prov = sweep.resolve_training_mirror(cid, op)
    assert prov.trained_template_hash == SHIPPED_HASH
    op = _stage_derived(tmp_path / "y", pos_hash=SHIPPED_HASH,
                        neg_hash=SHIPPED_HASH, top_hash=EMPTY_HASH)
    with pytest.raises(sweep.UnrecordedMirrorError, match="disagrees with itself"):
        sweep.resolve_training_mirror("y", op)


@needs_derived_verdict
def test_derived_parents_without_a_kind_refuse(tmp_path):
    # parent admission requires training_budget_kind; a derived record whose
    # parents carry none establishes no kind and refuses (never "verified")
    op = _stage_derived(tmp_path / "prekind", parent_kind=None)
    with pytest.raises(sweep.UnrecordedMirrorError, match="establishes NO fit kind"):
        sweep.resolve_training_mirror("prekind", op)


@needs_derived_verdict
def test_derived_refusals(tmp_path):
    with pytest.raises(sweep.UnrecordedMirrorError, match="disagree on template_hash"):
        sweep.resolve_training_mirror("a", _stage_derived(
            tmp_path / "a", pos_hash=EMPTY_HASH, neg_hash=SHIPPED_HASH))
    with pytest.raises(sweep.UnrecordedMirrorError, match="object_sha256 mismatch"):
        sweep.resolve_training_mirror("b", _stage_derived(tmp_path / "b",
                                                          break_sha=True))
    with pytest.raises(sweep.UnrecordedMirrorError, match="exactly one identity"):
        sweep.resolve_training_mirror("c", _stage_derived(
            tmp_path / "c", also_budget_sidecar=True))
    with pytest.raises(sweep.UnrecordedMirrorError, match="model_id"):
        sweep.resolve_training_mirror("d", _stage_derived(
            tmp_path / "d", top_model=IT_MODEL))


def test_read_derived_provenance_is_none_without_the_sidecar(tmp_path):
    op = _stage_doc(tmp_path / CELL)
    assert sweep.read_derived_provenance(op.parent, op, cell_id=CELL) is None


@needs_derived_verdict
def test_derived_sidecar_is_bound_to_the_resolved_cell_id(tmp_path):
    """A coherent derived artifact resolved under ANOTHER cell id — or one
    whose sidecar names no derived_id at all — used to relay its parents'
    facts as a document fit. The mirror threads the authoritative cell_id
    into the derived verdict, which requires an exact match and a recorded
    id."""
    cid = "constv2-mix-full__lora_r4_ce_minus_ctl"
    op = _stage_derived(tmp_path / cid)
    assert sweep.resolve_training_mirror(cid, op).status == sweep.DOCUMENT_FIT_STATUS
    with pytest.raises(sweep.UnrecordedMirrorError, match="wrong-identity"):
        sweep.resolve_training_mirror("definitely-the-wrong-cell", op)
    with pytest.raises(sweep.UnrecordedMirrorError, match="wrong-identity"):
        sweep.read_derived_provenance(op.parent, op, cell_id="constv2ctl2-full__lora_r4_ce_minus_ctl")
    meta_path = op.parent / "derived_meta.json"
    meta = json.loads(meta_path.read_text()); meta.pop("derived_id")
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(sweep.UnrecordedMirrorError, match="records no derived_id"):
        sweep.resolve_training_mirror(cid, op)


@pytest.mark.skipif(HAS_DERIVED_VERDICT,
                    reason="this tree carries the derived verdict")
def test_a_tree_without_the_derived_verdict_refuses_derived_objects(tmp_path):
    # never a partial read: without the verdict the sidecar's parent
    # identities cannot be checked, so the object is refused, loudly
    op = _stage_derived(tmp_path / "d")
    with pytest.raises(sweep.UnrecordedMirrorError, match="no derived-object verdict"):
        sweep.resolve_training_mirror("d", op)


# --------------------------------------------------------------------------- #
# the driver's evidence spells the fit side out                                #
# --------------------------------------------------------------------------- #

from test_served_capability import BASE_MODEL as SERVED_BASE  # noqa: E402
from test_served_capability import REV as SERVED_REV  # noqa: E402
from test_served_capability import TEMPLATE_HASH as SERVED_TEMPLATE  # noqa: E402
from test_served_capability import world  # noqa: E402,F401  (fixture)
import served_capability_run as srun  # noqa: E402


def _evidence(out_root: Path) -> dict:
    cells = sorted((out_root / "cells").iterdir())
    assert len(cells) == 1
    return json.loads((cells[0] / "evidence.json").read_text())


def _manifest(out_root: Path) -> dict:
    cells = sorted((out_root / "cells").iterdir())
    assert len(cells) == 1
    return json.loads((cells[0] / "manifest.json").read_text())


def _assert_manifest_mirrors_evidence(out_root: Path, expected: str) -> None:
    """The class the merge writer reads (the manifest — stated top-level on
    every store, hashed into resolved_config when not compared/baseline)
    equals the human-readable mirror in evidence.json, on every store the
    driver writes; and the store is dated (producer + contract_version) so a
    later reader never infers from absence."""
    from sisteer.capeval.tinycap import contracts as tc
    m, ev = _manifest(out_root), _evidence(out_root)
    key = sweep.FIT_PROVENANCE_TEMPLATE_CHECK_KEY
    assert m[key] == ev["fit_provenance"]["template_check"] == expected
    assert m["fit_provenance"]["template_check"] == expected
    assert m["fit_provenance"] == ev["fit_provenance"]
    assert (key in m["resolved_config"]) == (expected not in sweep.TEMPLATE_CHECK_UNHASHED)
    if key in m["resolved_config"]:
        assert m["resolved_config"][key] == expected
    assert sweep.template_check_of_manifest(m) == expected
    # the immutable witness is on the manifest and the refreshable block agrees
    # with it field by field; the consumption-boundary validator is silent
    w = m[sweep.MANIFEST_KEY_FIT_WITNESS]
    assert set(w) == set(sweep.FIT_WITNESS_FIELDS)        # the manifest writer sorts keys
    assert {f: m["fit_provenance"][f] for f in sweep.FIT_WITNESS_FIELDS} == w
    assert sweep.store_provenance_problems(m) == []
    assert m[tc.MANIFEST_KEY_CONTRACT_VERSION] == tc.CONTRACT_VERSION
    assert m[tc.MANIFEST_KEY_PRODUCER]["driver"] == tc.PRODUCER_DRIVER_SERVED
    assert len(m[tc.MANIFEST_KEY_PRODUCER]["tree_sha"]) == 40


def test_evidence_records_a_document_fit_transfer(world, monkeypatch):
    # a base-model document fit served on its instruct sibling under a
    # template the fit never saw: the row runs, and says exactly that
    monkeypatch.setattr(srun, "_mirror_provenance", lambda cell, **kw: SimpleNamespace(
        model_id=BASE_MODEL, revision=BASE_REV,
        trained_template_hash=EMPTY_HASH, object_sha="ab" * 8,
        status=sweep.DOCUMENT_FIT_STATUS, source="train_budget_meta",
        positions_at_fit="all", training_budget_kind=DOCUMENT_TOKENS_KIND,
        chat_template_bypassed=True))
    srun.main(world.argv)
    ev = _evidence(world.out_root)
    assert ev["engine_stamp"]["provenance_status"] == "document_fit"
    assert ev["engine_stamp"]["trained_template_hash"] == EMPTY_HASH
    assert ev["engine_stamp"]["served_template_hash"] == SERVED_TEMPLATE
    fp = ev["fit_provenance"]
    assert fp["model_id"] == BASE_MODEL and fp["revision"] == BASE_REV
    assert fp["served_model_id"] == SERVED_BASE
    assert fp["served_revision"] == SERVED_REV
    assert fp["transfer"] is True
    assert fp["template_check"] == "inapplicable"
    assert fp["training_budget_kind"] == DOCUMENT_TOKENS_KIND
    assert fp["chat_template_bypassed"] is True
    _assert_manifest_mirrors_evidence(world.out_root, "inapplicable")


def test_evidence_records_a_verified_same_model_fit(world):
    # the fixture's default provenance: fit and served are the same model
    # and template — compared, not a transfer
    srun.main(world.argv)
    fp = _evidence(world.out_root)["fit_provenance"]
    assert fp["status"] == "verified"
    assert fp["transfer"] is False
    assert fp["template_check"] == "compared"
    assert fp["template_hash"] == fp["served_template_hash"] == SERVED_TEMPLATE
    _assert_manifest_mirrors_evidence(world.out_root, "compared")


# --------------------------------------------------------------------------- #
# the equivalence boundary: the merge and the plan (P1-1)                        #
# --------------------------------------------------------------------------- #


def _merge_store(root: Path, key: str, *, cls: str, payload: dict, task: str = "inspect/math") -> Path:
    """A minimal tinycap-shaped store the merge reads: a dated manifest that
    STATES a template-check class (hashed when not compared/baseline), its
    witness + block, a provenance block, and a done task summary."""
    tc = pytest.importorskip("sisteer.capeval.tinycap.contracts")
    from sisteer.capeval import store as cs
    d = root / "cells" / key
    (d / tc.SUMMARY_DIR).mkdir(parents=True, exist_ok=True)
    status = {"compared": "verified", "inapplicable": sweep.DOCUMENT_FIT_STATUS}[cls]
    resolved = {"driver": "served_capability_run_v2", "engine": "vllm", "cell_id": "synthetic"}
    if cls not in sweep.TEMPLATE_CHECK_UNHASHED:
        resolved[sweep.FIT_PROVENANCE_TEMPLATE_CHECK_KEY] = cls
    witness = {**{f: None for f in sweep.FIT_WITNESS_FIELDS}, "model_id": BASE_MODEL,
               "revision": BASE_REV, "template_hash": EMPTY_HASH if cls == "inapplicable" else SERVED_HASH,
               "status": status, "source": "train_budget_meta", "object_sha": "ab" * 8}
    m = {"cell_id": "synthetic", "harness_config_hash": cs.harness_config_hash(resolved),
         "resolved_config": resolved, tc.MANIFEST_KEY_PROVENANCE: {"cell_id": "synthetic"},
         tc.MANIFEST_KEY_PRODUCER: {"tree_sha": "ab" * 20, "driver": tc.PRODUCER_DRIVER_SERVED},
         tc.MANIFEST_KEY_CONTRACT_VERSION: tc.CONTRACT_VERSION,
         tc.MANIFEST_KEY_GEOMETRY: {tc.GEOMETRY_PASS_FIELD: 1},
         sweep.FIT_PROVENANCE_TEMPLATE_CHECK_KEY: cls,
         sweep.MANIFEST_KEY_FIT_WITNESS: witness,
         sweep.MANIFEST_KEY_FIT_PROVENANCE: {**witness, "template_check": cls,
                                             "template_check_reason": sweep.TEMPLATE_CHECK_REASON[cls]}}
    (d / "manifest.json").write_text(json.dumps(m))
    (d / tc.SUMMARY_DIR / tc.marker_filename(task)).write_text(json.dumps(payload))
    return d


def test_p1_1_the_merge_source_carries_the_class_and_refuses_an_inconsistent_store(tmp_path):
    tc = pytest.importorskip("sisteer.capeval.tinycap.contracts")
    from sisteer.capeval.tinycap import completion
    payload = {"acc": 0.4, "n": 100}
    d = _merge_store(tmp_path, "k1__controls__t0__a0__aaaaaaaa__bbbbbbbb", cls="compared", payload=payload)
    src = completion.merge_source(d, json.loads((d / "manifest.json").read_text()), legacy_pass=True)
    assert src[tc.MERGE_SOURCE_TEMPLATE_CHECK_FIELD] == "compared"
    d2 = _merge_store(tmp_path, "k2__controls__t0__a0__aaaaaaaa__cccccccc", cls="inapplicable", payload=payload)
    assert completion.merge_source(d2, json.loads((d2 / "manifest.json").read_text()),
                                   legacy_pass=True)[tc.MERGE_SOURCE_TEMPLATE_CHECK_FIELD] == "inapplicable"
    # a current store whose class is MISSING is not a merge candidate (the
    # reader refused it and the merge used to accept it)
    m = json.loads((d / "manifest.json").read_text()); m.pop(sweep.FIT_PROVENANCE_TEMPLATE_CHECK_KEY)
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(completion.CompletionError, match="not a merge candidate"):
        completion.merge_source(d, m, legacy_pass=True)


def test_p1_1_cross_class_stores_are_never_equal_witnesses(tmp_path):
    # Reproduction: a compared store and an inapplicable store, same cell and
    # task, identical metrics, an UNRECORDED standing entry — the plan made
    # the inapplicable one adopt with the compared one as its equal witness.
    # Now the plan pairs nothing across classes; two stores of ONE class
    # still plan as before.
    tc = pytest.importorskip("sisteer.capeval.tinycap.contracts")
    from sisteer.capeval.tinycap import completion
    task = "inspect/math"
    payload = {"acc": 0.4, "n": 100}
    a = _merge_store(tmp_path, "a__controls__t0__a0__aaaaaaaa__11111111", cls="compared", payload=payload)
    b = _merge_store(tmp_path, "b__controls__t0__a0__aaaaaaaa__22222222", cls="inapplicable", payload=payload)
    c = _merge_store(tmp_path, "c__controls__t0__a0__aaaaaaaa__33333333", cls="compared", payload=payload)
    summary_dir = tmp_path / "synthetic"; summary_dir.mkdir()
    (summary_dir / tc.CONTROLS_SUMMARY_FILENAME).write_text(json.dumps(
        {"controls": {task: payload}, "provenance": {"cell_id": "synthetic"}}))   # unrecorded standing
    def emitted(*dirs):
        # the producing tree's emitted payload (contracts.MERGE_PAYLOAD_FIELDS)
        return [(d, {"cell_dir": str(d), "payloads": {task: dict(payload)}}) for d in dirs]
    plan = completion.equal_witness_plan(emitted(a, b))
    # contested, EXPLICITLY, on both candidates: an empty plan would not
    # prohibit adoption by order of arrival (fourth review)
    contested = {a.name: "compared", b.name: "inapplicable"}
    assert plan[str(a)] == {"witnesses": {}, "yield": [], "contested": {task: contested}}
    assert plan[str(b)] == {"witnesses": {}, "yield": [], "contested": {task: contested}}
    # the writer half refuses the whole store write unless the winner is named
    for d in (a, b):
        with pytest.raises(completion.CompletionError, match="CONTESTED"):
            completion.merge_emitted_payload(d, emitted(d)[0][1], plan=plan[str(d)])
    with pytest.raises(completion.CompletionError, match="none of the contesting stores"):
        completion.merge_emitted_payload(a, emitted(a)[0][1], plan=plan[str(a)],
                                         replace_source={task: "someone_else"})
    # same class: adoption planned, the smallest cell_key adopts
    plan = completion.equal_witness_plan(emitted(a, c))
    assert plan[str(a)]["witnesses"][task][0]["cell_key"] == c.name
    assert plan[str(a)]["witnesses"][task][0][tc.MERGE_SOURCE_TEMPLATE_CHECK_FIELD] == "compared"
    assert plan[str(c)]["yield"] == [task]
    # and the writer refuses a cross-class record even if handed one
    from sisteer.eval.cell_controls import _resolve_merge_sources
    src_a = completion.merge_source(a, json.loads((a / "manifest.json").read_text()), legacy_pass=True)
    src_b = completion.merge_source(b, json.loads((b / "manifest.json").read_text()), legacy_pass=True)
    with pytest.raises(ValueError, match="never equal witnesses"):
        _resolve_merge_sources({task: payload}, {}, {task: payload}, {task: src_a},
                               replace_source={}, path="synthetic",
                               equal_witnesses={task: [{**src_b}]})


def _legacy_manifest(m: dict) -> dict:
    """Strip a fresh store's manifest down to the shape the c91a0568 legacy
    XSTest/OR-Bench path wrote: no stated class, no producer, no
    contract_version, no fit_provenance block."""
    from sisteer.capeval.tinycap import contracts as tc
    for k in (KEY, tc.MANIFEST_KEY_PRODUCER, tc.MANIFEST_KEY_CONTRACT_VERSION,
              sweep.MANIFEST_KEY_FIT_PROVENANCE):
        m.pop(k, None)
    m["resolved_config"].pop(KEY, None)
    return m


def test_p1b_resume_never_upgrades_a_stores_evidence(world, monkeypatch):
    # Reproduction: a store generated under partial provenance by the legacy
    # driver (no producer guard), resumed under verified provenance: same
    # identity, zero new generations, evidence rewritten to compared. Now:
    # the resume REFUSES (the recorded manifest states no class), the old
    # evidence is untouched, and a fresh out root gets a NEW store.
    srun.main(world.argv)
    cells = sorted((world.out_root / "cells").iterdir()); assert len(cells) == 1
    cell_dir = cells[0]
    mp, ep = cell_dir / "manifest.json", cell_dir / "evidence.json"
    legacy = _legacy_manifest(json.loads(mp.read_text()))
    mp.write_text(json.dumps(legacy))
    ev_before = ep.read_bytes()
    world.posted.clear()
    # a store holding generations whose class cannot be read is refused by the
    # canonical validator first (P2); an undatable one by the era; either way
    # the evidence is written once and nothing is generated
    with pytest.raises(SystemExit, match="does not read consistently|refusing to resume onto it"):
        srun.main(world.argv)
    assert ep.read_bytes() == ev_before            # evidence written once
    assert json.loads(mp.read_text()) == legacy    # manifest not amended
    assert not world.posted                        # nothing generated
    # the same run into a fresh out root is a NEW store, stated and dated
    out2 = world.tmp_path / "fresh"
    srun.main([a if a != str(world.out_root) else str(out2) for a in world.argv])
    _assert_manifest_mirrors_evidence(out2, "compared")


def test_p1b_a_provenance_change_is_not_a_resume(world, monkeypatch):
    # a store that STATES compared, resumed under document-fit provenance:
    # the class differs (and is hashed), so it is a different store key —
    # and were the keys ever to coincide, the stated-class check refuses
    srun.main(world.argv)
    m1 = _manifest(world.out_root)
    monkeypatch.setattr(srun, "_mirror_provenance", lambda cell, **kw: SimpleNamespace(
        model_id=BASE_MODEL, revision=BASE_REV,
        trained_template_hash=EMPTY_HASH, object_sha="ab" * 8,
        status=sweep.DOCUMENT_FIT_STATUS, source="train_budget_meta",
        positions_at_fit="all", training_budget_kind=DOCUMENT_TOKENS_KIND,
        chat_template_bypassed=True))
    srun.main(world.argv)
    cells = sorted((world.out_root / "cells").iterdir())
    assert len(cells) == 2                                  # two stores, not one
    m2 = [json.loads((c / "manifest.json").read_text()) for c in cells
          if json.loads((c / "manifest.json").read_text())[KEY] == "inapplicable"][0]
    assert m1["harness_config_hash"] != m2["harness_config_hash"]
    # the stated-class guard itself, on a same-key manifest
    store = SimpleNamespace(manifest_path=cells[0] / "manifest.json", cell_dir=cells[0],
                            _content_artifacts=lambda: ["rows.jsonl"])   # a store that holds work
    recorded = json.loads(store.manifest_path.read_text())
    with pytest.raises(SystemExit, match="a provenance change is not a resume"):
        srun._refuse_resume_across_provenance(
            store, {**recorded, KEY: "inapplicable" if recorded[KEY] == "compared" else "compared"})


def test_p1_2_an_altered_sidecar_never_replaces_the_recorded_fit_witness(world, monkeypatch):
    # Reproduction: same class, same key, but the sidecar was restaged with
    # another trained template hash and model revision — the resume kept the
    # store and REPLACED the recorded fit facts. Now: refused before anything
    # is touched; manifest and evidence bytes unchanged.
    srun.main(world.argv)
    cells = sorted((world.out_root / "cells").iterdir()); assert len(cells) == 1
    mp, ep = cells[0] / "manifest.json", cells[0] / "evidence.json"
    m_before, e_before = mp.read_bytes(), ep.read_bytes()
    monkeypatch.setattr(srun, "_mirror_provenance", lambda cell, **kw: SimpleNamespace(
        model_id=SERVED_BASE, revision="9" * 40,               # restaged revision
        trained_template_hash="ff" * 8,                         # restaged hash
        object_sha="ab" * 8, status="verified", source="train_budget_meta",
        positions_at_fit="response"))
    world.posted.clear()
    with pytest.raises(SystemExit, match="recorded fit witness differs") as ei:
        srun.main(world.argv)
    assert "revision" in str(ei.value) and "template_hash" in str(ei.value)
    assert mp.read_bytes() == m_before and ep.read_bytes() == e_before
    assert not world.posted


def test_p1_2_a_refused_resume_leaves_the_existing_manifest_intact(world, monkeypatch):
    # the destructive path: the first write used to replace the manifest
    # BEFORE the served==trained check, with a mapping lacking fit_provenance.
    # Now a resumed store is re-written only after its evidence step passes,
    # so a run the identity check refuses leaves the store exactly as it was.
    srun.main(world.argv)
    cells = sorted((world.out_root / "cells").iterdir()); assert len(cells) == 1
    mp = cells[0] / "manifest.json"
    m_before = mp.read_bytes()
    assert json.loads(m_before)[sweep.MANIFEST_KEY_FIT_PROVENANCE]["template_check"] == "compared"
    monkeypatch.setattr(srun, "_tok_evidence", lambda mid, rev: {
        "served_template_hash": "ee" * 8,                       # a template the fit never saw
        "serve_header_ids": [1, 2, 3], "serve_header_len": 3})
    with pytest.raises(EngineIdentityError, match="served==trained template"):
        srun.main(world.argv)
    assert mp.read_bytes() == m_before
    assert sweep.store_provenance_problems(json.loads(mp.read_text())) == []


def test_p2_a_torn_recorded_manifest_is_never_resumed_nor_repaired_in_place(world, monkeypatch):
    # the recorded manifest states "compared" but its hashed copy says
    # "inapplicable" — store_provenance_problems rejects it as torn, yet
    # direct resume compared the one top-level field, accepted the store,
    # made zero requests and REWROTE the manifest with the contradiction
    # erased. Now the canonical validator runs on the recorded manifest first.
    srun.main(world.argv)
    cells = sorted((world.out_root / "cells").iterdir()); assert len(cells) == 1
    mp = cells[0] / "manifest.json"
    m = json.loads(mp.read_text())
    m["resolved_config"][KEY] = "inapplicable"           # the hashed copy contradicts the stated class
    mp.write_text(json.dumps(m)); torn = mp.read_bytes()
    assert sweep.store_provenance_problems(m)
    world.posted.clear()
    with pytest.raises(SystemExit, match="does not read consistently"):
        srun.main(world.argv)
    assert mp.read_bytes() == torn and not world.posted   # untouched, nothing generated
    # an INCOMPLETE witness (a field missing, not None) is a difference too
    m = json.loads(torn); m["resolved_config"].pop(KEY)
    del m[sweep.MANIFEST_KEY_FIT_WITNESS]["revision"]
    mp.write_text(json.dumps(m)); incomplete = mp.read_bytes()
    with pytest.raises(SystemExit, match="does not read consistently"):
        srun.main(world.argv)
    assert mp.read_bytes() == incomplete
    cur = sweep.fit_witness_of(SimpleNamespace(model_id="m", revision=None, trained_template_hash="h",
                                               status="verified", source="s", positions_at_fit="all"))
    assert "revision" in sweep.fit_witness_diff({k: v for k, v in cur.items() if k != "revision"}, cur)
    assert sweep.fit_witness_diff(dict(cur), cur) == []


def test_p2_an_empty_pre_evidence_store_is_recovered_not_validated_for_generations(world):
    # a run that died before its evidence step leaves a manifest with the
    # class and the witness but no fit_provenance block and no artifacts;
    # recovering it is a different operation from validating a store that
    # holds generations — it needs a consistent class and a complete witness,
    # not the block the run never wrote
    srun.main(world.argv)
    cells = sorted((world.out_root / "cells").iterdir()); assert len(cells) == 1
    d = cells[0]
    for p in list(d.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            p.unlink()
    m = json.loads((d / "manifest.json").read_text()); m.pop(sweep.MANIFEST_KEY_FIT_PROVENANCE)
    (d / "manifest.json").write_text(json.dumps(m))
    assert sweep.store_provenance_problems(m)               # not valid as a store of generations
    (world.out_root / "safety_refusal__psr_kl_all" / "controls_summary.json").unlink(missing_ok=True)
    srun.main(world.argv)                                   # recovered
    assert sorted((world.out_root / "cells").iterdir()) == [d]
    _assert_manifest_mirrors_evidence(world.out_root, "compared")
    # but an empty store whose class does not read is refused
    m = json.loads((d / "manifest.json").read_text())
    for p in list(d.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            p.unlink()
    m.pop(sweep.MANIFEST_KEY_FIT_PROVENANCE); m["resolved_config"][KEY] = "impossible"
    (d / "manifest.json").write_text(json.dumps(m))
    (world.out_root / "safety_refusal__psr_kl_all" / "controls_summary.json").unlink(missing_ok=True)
    with pytest.raises(SystemExit, match="class does not read"):
        srun.main(world.argv)


def test_p2_a_linked_spool_layout_is_populated_not_empty_and_never_recovered(world, monkeypatch):
    # fifth review: a partial restore — manifest.json plus `private/` as a
    # DIRECTORY SYMLINK to retained generation spools — read as EMPTY to the
    # rglob/is_file emptiness check and as full to the generation reader, so
    # it entered empty-store recovery: zero requests, cached generations kept,
    # manifest rewritten, the validator then accepting it. Now the linked
    # directory is content (full validation), and a recorded fit_provenance
    # block is validated in the recovery branch rather than ignored.
    import os
    srun.main(world.argv)
    cells = sorted((world.out_root / "cells").iterdir()); assert len(cells) == 1
    d = cells[0]
    assert (d / "private").is_dir(), "the driver's spools live under private/"
    retained = world.tmp_path / "retained_private"
    (d / "private").rename(retained)
    os.symlink(retained, d / "private", target_is_directory=True)
    spool_bytes = {p: p.read_bytes() for p in retained.rglob("*") if p.is_file()}
    mp = d / "manifest.json"
    # (A) the block missing: the store is populated (the link is content), so it
    # is validated as a store of generations and refused — never recovered
    m = json.loads(mp.read_text()); m.pop(sweep.MANIFEST_KEY_FIT_PROVENANCE)
    mp.write_text(json.dumps(m)); before = mp.read_bytes()
    world.posted.clear()
    with pytest.raises(SystemExit, match="holds generations but its provenance does not read consistently"):
        srun.main(world.argv)
    assert mp.read_bytes() == before and not world.posted
    assert {p: p.read_bytes() for p in retained.rglob("*") if p.is_file()} == spool_bytes
    # (B) a block present but contradicting the immutable witness: refused too
    m = json.loads(before)
    m[sweep.MANIFEST_KEY_FIT_PROVENANCE] = {**json.loads(mp.read_text()).get(sweep.MANIFEST_KEY_FIT_PROVENANCE, {}),
                                            **m[sweep.MANIFEST_KEY_FIT_WITNESS], "template_check": "compared",
                                            "template_hash": "ff" * 8}
    mp.write_text(json.dumps(m)); before = mp.read_bytes()
    with pytest.raises(SystemExit, match="does not read consistently"):
        srun.main(world.argv)
    assert mp.read_bytes() == before and not world.posted
    # (C) and a manifest-only store that CARRIES a block is not a pre-evidence
    # store either: the block is validated, not ignored
    (d / "private").unlink()
    for p in list(d.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            p.unlink()
    with pytest.raises(SystemExit, match="does not read consistently"):
        srun.main(world.argv)
    assert mp.read_bytes() == before and not world.posted


def test_a_legitimate_resume_of_a_stated_store_proceeds(world):
    srun.main(world.argv)
    n_posted = len(world.posted)
    srun.main(world.argv)          # same class, same store: resume, no refusal
    cells = sorted((world.out_root / "cells").iterdir()); assert len(cells) == 1
    assert len(world.posted) >= n_posted


def test_document_fit_and_chat_fit_stores_of_one_cell_cannot_pool(world, monkeypatch):
    # the class is INSIDE the hashed resolved_config, so the two stores land
    # under different cell keys (chash) — different witnesses at merge
    from sisteer.capeval import store as cs
    srun.main(world.argv)
    compared = _manifest(world.out_root)
    monkeypatch.setattr(srun, "_mirror_provenance", lambda cell, **kw: SimpleNamespace(
        model_id=BASE_MODEL, revision=BASE_REV,
        trained_template_hash=EMPTY_HASH, object_sha="ab" * 8,
        status=sweep.DOCUMENT_FIT_STATUS, source="train_budget_meta",
        positions_at_fit="all", training_budget_kind=DOCUMENT_TOKENS_KIND,
        chat_template_bypassed=True))
    out2 = world.tmp_path / "out2"
    argv = [a if a != str(world.out_root) else str(out2) for a in world.argv]
    srun.main(argv)
    doc = _manifest(out2)
    assert compared["harness_config_hash"] != doc["harness_config_hash"]
    assert (cs.harness_config_hash(compared["resolved_config"])
            != cs.harness_config_hash(doc["resolved_config"]))


# --------------------------------------------------------------------------- #
# the store boundary: a store written BEFORE the field existed                 #
# --------------------------------------------------------------------------- #


KEY = sweep.FIT_PROVENANCE_TEMPLATE_CHECK_KEY


def _manifest_with(resolved: dict, **top) -> dict:
    return {"cell_id": CELL, "resolved_config": resolved, **top}


def test_a_stated_class_is_read_and_checked_against_its_hashed_copy():
    # stated top-level, hashed when not compared/baseline: consistent -> class
    assert sweep.template_check_of_manifest(_manifest_with(
        {"driver": "served_capability_run_v2"}, **{KEY: "compared"})) == "compared"
    assert sweep.template_check_of_manifest(_manifest_with(
        {"driver": "served_capability_run_v2", KEY: "unverified"},
        **{KEY: "unverified"})) == "unverified"
    # a hashed copy that disagrees, a hashable class with no hashed copy, a
    # hashed copy with no stated class, an unknown class: all refuse
    with pytest.raises(ValueError, match="torn or inconsistent"):
        sweep.template_check_of_manifest(_manifest_with(
            {KEY: "impossible"}, **{KEY: "compared"}))
    with pytest.raises(ValueError, match="must be hashed"):
        sweep.template_check_of_manifest(_manifest_with({}, **{KEY: "inapplicable"}))
    with pytest.raises(ValueError, match="states no top-level class"):
        sweep.template_check_of_manifest(_manifest_with({KEY: "inapplicable"}))
    with pytest.raises(ValueError, match="not a known class"):
        sweep.template_check_of_manifest(_manifest_with({}, **{KEY: "probably_fine"}))


def test_an_absent_class_is_dated_by_the_one_era_mechanism_never_by_absence():
    """A missing marker is not positive evidence of an older producer. An
    absent class is dated by tinycap.contracts.store_era for the registered
    template-check contract — the SAME mechanism every other contract uses —
    and only a POSITIVE statement of age (a stamped contract_version below
    the contract's, or a declared producer tree that does not contain it)
    yields the stated-absence bucket. Everything else refuses:
    current-and-absent (torn), an unknown producer, a manifest that says
    nothing, an empty or missing manifest."""
    from sisteer.capeval.tinycap import contracts as tc
    assert sweep.CONTRACT_TEMPLATE_CHECK in tc.CONTRACTS
    spec = tc.CONTRACTS[sweep.CONTRACT_TEMPLATE_CHECK]
    assert spec.stamped_from == tc.CONTRACT_VERSION == 3
    wave_tree = next(iter(tc.KNOWN_PRODUCER_TREES))
    assert sweep.CONTRACT_TEMPLATE_CHECK in tc.KNOWN_PRODUCER_TREES[wave_tree].does_not_contain
    # historical by its own statement (stamped below the contract), or by its
    # declared producer tree: a stated absence, never "compared"
    for m in (_manifest_with({}, contract_version=2),
              _manifest_with({}, producer={"tree_sha": wave_tree,
                                           "driver": tc.PRODUCER_DRIVER_SERVED})):
        got = sweep.template_check_of_manifest(m)
        assert got == sweep.TEMPLATE_CHECK_UNRECORDED
        assert got not in sweep.TEMPLATE_CHECK_CLASSES
    # current-and-absent: torn
    with pytest.raises(ValueError, match="torn or inconsistent"):
        sweep.template_check_of_manifest(_manifest_with({}, contract_version=3))
    # unknown: an undeclared producer, a record that says nothing, an empty
    # manifest, no manifest — all REFUSE (never "unrecorded")
    for m in (_manifest_with({}, producer={"tree_sha": "ab" * 20,
                                           "driver": tc.PRODUCER_DRIVER_SERVED}),
              _manifest_with({"driver": "served_capability_run_v2"}),
              {}):
        with pytest.raises(ValueError, match="cannot be dated"):
            sweep.template_check_of_manifest(m)
    with pytest.raises(ValueError, match="no readable manifest"):
        sweep.template_check_of_manifest(None)


def test_resume_refuses_a_store_not_written_under_the_contract(world):
    # a store that even STATES a class but dates itself before the contract
    # (contract_version 2) is not resumed: age is decided by the era
    # mechanism, and a pre-contract store's evidence is never re-stated
    from sisteer.capeval.tinycap import contracts as tc
    srun.main(world.argv)
    cells = sorted((world.out_root / "cells").iterdir()); assert len(cells) == 1
    mp = cells[0] / "manifest.json"
    m = json.loads(mp.read_text()); m[tc.MANIFEST_KEY_CONTRACT_VERSION] = 2
    mp.write_text(json.dumps(m))
    ev_before = (cells[0] / "evidence.json").read_bytes()
    with pytest.raises(SystemExit, match="not written under the template-check contract"):
        srun.main(world.argv)
    assert (cells[0] / "evidence.json").read_bytes() == ev_before


def test_the_non_compared_classes_are_hashed_and_compared_is_not():
    assert sweep.TEMPLATE_CHECK_UNHASHED == {"compared", "baseline"}
    for c in sweep.TEMPLATE_CHECK_CLASSES:
        if c not in sweep.TEMPLATE_CHECK_UNHASHED:
            assert c in ("inapplicable", "impossible", "unverified")


def test_the_class_table_is_total_over_the_status_vocabulary():
    for status in sweep.PROVENANCE_STATUSES:
        assert sweep.template_check_class(status) in sweep.TEMPLATE_CHECK_CLASSES
    with pytest.raises(ValueError, match="no template-check class"):
        sweep.template_check_class("trusted")
    assert sweep.template_check_class(None) == sweep.TEMPLATE_CHECK_BASELINE


def test_a_verified_mismatch_still_refuses_before_generation(world, monkeypatch):
    # the guard is unchanged for chat-rendered fits: a verified record whose
    # template differs from the served one refuses, and nothing is posted
    monkeypatch.setattr(srun, "_mirror_provenance", lambda cell, **kw: SimpleNamespace(
        model_id=SERVED_BASE, revision=SERVED_REV,
        trained_template_hash=EMPTY_HASH, object_sha="ab" * 8,
        status="verified", source="train_budget_meta",
        positions_at_fit="response"))
    with pytest.raises(EngineIdentityError, match="served==trained template"):
        srun.main(world.argv)
    assert not world.posted
    # ... and nothing was PROBED either: the served==trained assertion now runs before the
    # first steered request, so a wrongly-provenanced object is refused before it is served
    # once on the way to the refusal
    assert world.chat.calls == []


def test_a_verified_match_still_probes_after_the_assertion(world):
    # the reorder moved the assertion, not the probe: a matching record still runs the
    # steering-bites probe (calls on both legs) and the run proceeds
    srun.main(world.argv)
    assert world.chat.calls and world.posted
