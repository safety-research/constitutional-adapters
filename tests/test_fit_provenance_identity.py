"""The THIRD identity sidecar: ``fit_provenance.json``, the HPO-promotion
lineage's fit record (2026-09-06). CPU-only, synthetic records throughout.

Two fitting lineages publish the SAME fit-time facts under different
spellings. The training lane writes ``train_budget_meta.json`` (the gemma
scale-12 si12 copies); the HPO-promotion lineage writes
``fit_provenance.json`` (the qwen scale-12 si12 copies — tune_arm.py trials
promoted into the bank). Nothing upstream carries the other's file, so a
re-stage cannot fix a consumer that demands one spelling; the consumer must
accept both. The ONE normaliser both are read through is
``sweep.read_stage3_fit_provenance`` — the stage-3 mirror gate's positive
record gate, which already accepted either spelling and never translates one
into the other on disk.

THE LOAD-BEARING TEST HERE is an equivalence that is ASSERTED rather than
assumed: a gemma-shaped budget sidecar and a qwen-shaped fit record
describing the SAME fit must normalise to identical fit-time facts —
resolved model id, revision, chat-template hash, fit leg, status —
differing only in the ``source`` label. If they did not, the two targets'
si12 cells would not be comparable under one gate. The synthetic records
here carry the REAL key names of both lineages (read off the staged
copies) with invented values.

Also pinned: the dispatch (``object_identity_kind`` grows a third value; any
two sidecars together are ambiguous), the fit-record verdict's refusals
(missing/null required keys, a companion ``provenance.json`` naming another
cell, a blob whose fit leg disagrees with the record's, a second identity
sidecar), the promotion lineage's PLAIN-TEXT ``COMPLETE`` note being accepted
where the training pipeline's JSON manifest is verified, and every consumer
entry point routing the new kind.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from sisteer.eval import object_identity as oi
from sisteer.eval import train_budget as tb
from sisteer.eval.sweep import (STAGE3_FIT_REQUIRED_KEYS,
                                TRAIN_BUDGET_REQUIRED_KEYS,
                                read_stage3_fit_provenance)

CELL = "safety_refusal__psr_kl_all"
#: the ONE fit, described twice below. Synthetic values, real key names.
FIT = {"model": "synthetic-org/synthetic-4B", "revision": "0" * 40,
       "template_hash": "1111222233334444", "positions": "response"}


def _blob(positions="response", header=7):
    return {"kind": "psr", "layers": [0], "positions": positions,
            "response_header_len": header if positions == "response" else None,
            "directions": {"0": torch.zeros(4)}, "gate_w": {"0": torch.zeros(4)},
            "gate_b": {"0": torch.zeros(1)}}


def _fit_provenance_record(**over) -> dict:
    """The HPO-promotion lineage's record — the key set of the real staged
    copies (a ``note`` and an ``evidence`` block ride along; neither is
    read by any gate)."""
    rec = {"resolved_model_id": FIT["model"], "resolved_revision": FIT["revision"],
           "chat_template_hash": FIT["template_hash"], "writer": "tune_arm.py",
           "positions_at_fit": FIT["positions"],
           "evidence": {"resolved_model_id": "FIT-AUTHENTIC (synthetic)"},
           "note": "synthetic promotion record"}
    rec.update(over)
    return rec


def _budget_sidecar(**over) -> dict:
    """The training lane's sidecar — the key set of the real staged copies
    (legacy SI-pair schema: completion_tokens, no training_budget_kind)."""
    rec = {"cell_id": CELL, "arm": "psr_kl_all", "si_id": "safety_refusal",
           "completion_tokens": 512, "model_id": FIT["model"],
           "model_revision": FIT["revision"], "template_hash": FIT["template_hash"],
           "positions": FIT["positions"], "positions_at_fit": FIT["positions"],
           "response_header_len": 3, "max_len": 8192}
    rec.update(over)
    return rec


def _stage_fit_provenance(d: Path, *, record=None, provenance=..., complete=...,
                          blob=None) -> Path:
    """A promotion-lineage staged dir: object.pt + fit_provenance.json (+ the
    companion provenance.json, + a plain-text COMPLETE note by default)."""
    d.mkdir(parents=True, exist_ok=True)
    p = d / "object.pt"
    torch.save(blob if blob is not None else _blob(), p)
    (d / "fit_provenance.json").write_text(json.dumps(
        record if record is not None else _fit_provenance_record()))
    if provenance is ...:
        provenance = {"si": "safety_refusal", "arm": "psr_kl_all",
                      "positions": "response", "response_header_len": 7,
                      "promoted": "2026-01-01"}
    if provenance is not None:
        (d / "provenance.json").write_text(json.dumps(provenance))
    if complete is ...:
        complete = "promoted from out/hpo/synthetic/r1_t00 on 2026-01-01\n"
    if complete is not None:
        (d / "COMPLETE").write_text(complete)
    (d / "staged_from.json").write_text(json.dumps({"source": "synthetic"}))
    return p


def _stage_budget(d: Path, **over) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / "object.pt"
    torch.save(_blob(), p)
    from sisteer.eval.defense_cell import sha256_file
    (d / "train_budget_meta.json").write_text(json.dumps(
        _budget_sidecar(object_sha256=sha256_file(p), **over)))
    return p


# ---------------------------------------------------------------------------
# THE equivalence: two spellings, one normalised record
# ---------------------------------------------------------------------------


class TestTwoSpellingsOneRecord:
    NORMALISED = ("resolved_model_id", "resolved_revision", "chat_template_hash",
                  "positions_at_fit", "status",
                  # the fit-kind facts the document-fit status is assigned from
                  # (None on both SI-pair spellings; sweep.DOCUMENT_FIT_STATUS)
                  "training_budget_kind", "chat_template_bypassed",
                  # the fit kind each spelling POSITIVELY establishes (equal:
                  # si_pair on both), from a spelling-specific marker
                  "fit_kind")
    #: the marker that established the kind differs BY SPELLING (the budget
    #: schema's completion cap vs the native record's producer), so it is in
    #: the key set but not in the value equality
    SPELLING_SPECIFIC = ("fit_kind_evidence",
                         # the budget sidecar records object_sha256 and is bound to
                         # the bytes; the promotion record records none (the PIN
                         # binds it) — True vs None, by construction
                         "object_sha_bound",
                         # WHAT bound it: "record" for the sidecar's own sha,
                         # "expected_object_sha256" when a caller's pin binds a null-sha record, None otherwise —
                         # spelling-specific for the same reason as object_sha_bound
                         "object_sha_bound_by")

    def test_same_fit_normalises_identically_under_both_spellings(self, tmp_path):
        a = tmp_path / "gemma_shaped"
        b = tmp_path / "qwen_shaped"
        _stage_budget(a)
        _stage_fit_provenance(b)
        na = read_stage3_fit_provenance(a, object_path=a / "object.pt")
        nb = read_stage3_fit_provenance(b, object_path=b / "object.pt")
        assert na is not None and nb is not None
        # the same key set out of both, and the same VALUES on every fact
        assert set(na) == set(nb) == set(self.NORMALISED) | {"source"} | set(self.SPELLING_SPECIFIC)
        assert {k: na[k] for k in self.NORMALISED} == {k: nb[k] for k in self.NORMALISED}
        assert na["status"] == nb["status"] == "verified"
        assert na["fit_kind"] == nb["fit_kind"] == "si_pair"
        assert "completion_tokens" in na["fit_kind_evidence"]
        assert "tune_arm.py" in nb["fit_kind_evidence"]
        # the ONLY difference is which schema was read — and it is recorded
        assert na["source"] == "train_budget_meta"
        assert nb["source"] == "fit_provenance:tune_arm.py"
        # and both name the fit the synthetic records describe
        assert na["resolved_model_id"] == FIT["model"]
        assert na["chat_template_hash"] == FIT["template_hash"]
        assert na["positions_at_fit"] == FIT["positions"]

    def test_the_required_key_sets_are_the_same_three_facts_plus_writer(self):
        """The two schemas' REQUIRED keys are the same facts under two
        spellings; the native one additionally names its writer."""
        native = set(STAGE3_FIT_REQUIRED_KEYS)
        budget = set(TRAIN_BUDGET_REQUIRED_KEYS)
        assert budget == {"model_id", "model_revision", "template_hash"}
        assert native == {"resolved_model_id", "resolved_revision",
                          "chat_template_hash", "writer"}

    def test_a_record_that_disagrees_does_not_normalise_equal(self, tmp_path):
        """Non-vacuity: the equality above is a property of the records, not
        of the reader — two records describing DIFFERENT fits normalise
        differently on exactly the differing fact."""
        a = tmp_path / "a"
        b = tmp_path / "b"
        _stage_budget(a)
        _stage_fit_provenance(b, record=_fit_provenance_record(
            chat_template_hash="ffffeeeeddddcccc"))
        na = read_stage3_fit_provenance(a)
        nb = read_stage3_fit_provenance(b)
        assert na["chat_template_hash"] != nb["chat_template_hash"]
        assert {k: na[k] for k in self.NORMALISED if k != "chat_template_hash"} == \
               {k: nb[k] for k in self.NORMALISED if k != "chat_template_hash"}


# ---------------------------------------------------------------------------
# the dispatch: a third identity, any two together is ambiguous
# ---------------------------------------------------------------------------


class TestIdentityKind:
    def test_three_sidecars_three_kinds(self, tmp_path):
        p = _stage_fit_provenance(tmp_path / "fp")
        assert tb.object_identity_kind(p) == "fit_provenance"
        assert tb.IDENTITY_SIDECARS == ("train_budget_meta.json", "derived_meta.json",
                                        "fit_provenance.json")
        p2 = _stage_budget(tmp_path / "budget")
        assert tb.object_identity_kind(p2) == "staged"
        bare = tmp_path / "bare"
        bare.mkdir()
        torch.save(_blob(), bare / "object.pt")
        assert tb.object_identity_kind(bare / "object.pt") is None

    @pytest.mark.parametrize("extra", ["train_budget_meta.json", "derived_meta.json"])
    def test_a_second_sidecar_beside_the_fit_record_is_ambiguous(self, tmp_path, extra):
        p = _stage_fit_provenance(tmp_path / "fp")
        (p.parent / extra).write_text("{}")
        assert tb.object_identity_kind(p) == "ambiguous"
        with pytest.raises(SystemExit, match="more than one identity sidecar"):
            tb.assert_object_identity(p, CELL, consumer="t")
        assert any("exactly one identity" in s for s in
                   tb.fit_provenance_problems(p, CELL))

    def test_all_three_at_once_is_ambiguous_too(self, tmp_path):
        p = _stage_fit_provenance(tmp_path / "fp")
        (p.parent / "train_budget_meta.json").write_text("{}")
        (p.parent / "derived_meta.json").write_text("{}")
        assert tb.object_identity_kind(p) == "ambiguous"


# ---------------------------------------------------------------------------
# the fit-record verdict
# ---------------------------------------------------------------------------


class TestFitProvenanceVerdict:
    def test_a_complete_record_passes_and_dispatches(self, tmp_path):
        p = _stage_fit_provenance(tmp_path / "fp")
        assert tb.fit_provenance_problems(p, CELL) == []
        assert tb.assert_object_identity(p, CELL, consumer="t") == "fit_provenance"
        assert oi.validate_artifact_identity(p, CELL, consumer="t") == "fit_provenance"
        # with no cell named, the companion provenance.json's si__arm is the identity
        assert oi.recorded_identity(p) == CELL
        assert oi.validate_artifact_identity(p, None, consumer="t") == "fit_provenance"

    def test_legacy_alias_of_the_cell_is_accepted_in_the_companion(self, tmp_path):
        """The staged scale-12 lora cell's provenance names lora_r4; the
        legacy spelling lora_r1_all of the SAME cell is the one accepted
        mismatch, exactly as for the budget sidecar."""
        p = _stage_fit_provenance(tmp_path / "fp", provenance={
            "si": "safety_refusal", "arm": "lora_r1_all"})
        assert tb.fit_provenance_problems(p, "safety_refusal__lora_r4") == []
        with pytest.raises(SystemExit, match="names cell"):
            tb.assert_object_identity(p, "si_constitution__lora_r4", consumer="t")

    @pytest.mark.parametrize("missing", list(STAGE3_FIT_REQUIRED_KEYS))
    def test_a_missing_or_null_required_key_refuses(self, tmp_path, missing):
        for variant in ("absent", "null", "empty"):
            d = tmp_path / f"{missing}_{variant}"
            rec = _fit_provenance_record()
            if variant == "absent":
                rec.pop(missing)
            else:
                rec[missing] = None if variant == "null" else ""
            p = _stage_fit_provenance(d, record=rec)
            probs = tb.fit_provenance_problems(p, CELL)
            assert any(missing in s and "lacks" in s for s in probs), (variant, probs)
            # the shared reader agrees: no verified native record
            rec_n = read_stage3_fit_provenance(d)
            assert rec_n is None or not rec_n["source"].startswith("fit_provenance")

    def test_a_companion_naming_another_cell_refuses(self, tmp_path):
        p = _stage_fit_provenance(tmp_path / "fp", provenance={
            "si": "si_constitution", "arm": "psr_kl_all"})
        probs = tb.fit_provenance_problems(p, CELL)
        assert any("names cell 'si_constitution__psr_kl_all'" in s for s in probs)

    def test_no_companion_is_fine_and_the_dir_name_is_the_identity(self, tmp_path):
        p = _stage_fit_provenance(tmp_path / CELL, provenance=None)
        assert tb.fit_provenance_problems(p, CELL) == []
        assert oi.recorded_identity(p) == CELL   # the bank layout's dir name

    def test_blob_fit_leg_must_agree_with_the_records(self, tmp_path):
        """The lens serves the BLOB's recorded leg; a record claiming another
        describes a different fit."""
        p = _stage_fit_provenance(tmp_path / "fp", blob=_blob(positions="all"))
        probs = tb.fit_provenance_problems(p, CELL)
        assert any("positions='all'" in s and "positions_at_fit='response'" in s
                   for s in probs), probs
        # a record that does not state the leg makes no claim to disagree with
        rec = _fit_provenance_record()
        rec.pop("positions_at_fit")
        p2 = _stage_fit_provenance(tmp_path / "fp2", record=rec, blob=_blob(positions="all"))
        assert tb.fit_provenance_problems(p2, CELL) == []

    def test_plain_text_complete_is_accepted_json_complete_is_verified(self, tmp_path):
        from sisteer.eval.defense_cell import sha256_file

        # the promotion lineage's note: no verifiable claim, accepted
        p = _stage_fit_provenance(tmp_path / "note",
                                  complete="promoted from somewhere on a date\n")
        assert tb.fit_provenance_problems(p, CELL) == []
        # a JSON manifest is held to the training pipeline's coherence rules
        d = tmp_path / "manifest"
        p = _stage_fit_provenance(d, complete=None)
        good = {"cell_id": CELL, "files": {"object.pt": sha256_file(p)}}
        (d / "COMPLETE").write_text(json.dumps(good))
        assert tb.fit_provenance_problems(p, CELL) == []
        (d / "COMPLETE").write_text(json.dumps({**good, "files": {"object.pt": "00" * 32}}))
        assert any("does not match its published sha256" in s
                   for s in tb.fit_provenance_problems(p, CELL))
        (d / "COMPLETE").write_text(json.dumps({**good, "cell_id": "other__cell"}))
        assert any("belongs to a different cell" in s
                   for s in tb.fit_provenance_problems(p, CELL))
        # and the budget-sidecar path still REFUSES a plain-text marker (unchanged)
        pb = _stage_budget(tmp_path / "budget")
        (pb.parent / "COMPLETE").write_text("a note, not a manifest\n")
        assert any("does not parse" in s for s in tb.staged_cell_problems(pb, CELL))

    def test_an_absent_artifact_refuses_first(self, tmp_path):
        d = tmp_path / "gone"
        d.mkdir()
        (d / "fit_provenance.json").write_text(json.dumps(_fit_provenance_record()))
        assert tb.fit_provenance_problems(d / "object.pt", CELL) == [
            f"no staged artifact at {d / 'object.pt'}"]


# ---------------------------------------------------------------------------
# a consumer with a STRICTER bar refuses with the real reason
# ---------------------------------------------------------------------------


def test_probe_score_refuses_a_fit_record_cell_naming_its_own_publication_bar(tmp_path):
    """probe_score's stamp requires a JSON COMPLETE manifest tied to a fit
    fingerprint — a stricter bar than the serving stack's identity gate. The
    promotion lineage writes a one-line note, so the scorer cannot stamp such
    a cell; it must say THAT, not fall through to 'carries no COMPLETE
    marker' (which reads as an unpublished object). The identity gate itself
    admits the cell — the two verdicts differ on purpose."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "sdfbase"))
    import probe_score as ps

    p = _stage_fit_provenance(tmp_path / CELL)
    # the shared identity gate admits it (no parents root needed: not derived)
    assert ps._validate_object_identity(p, CELL) == "fit_provenance"
    with pytest.raises(SystemExit, match="promotion note") as err:
        ps.object_identity_stamp(p, CELL, "fit_provenance")
    assert "servable" in str(err.value) and "carries no COMPLETE" not in str(err.value)


# ---------------------------------------------------------------------------
# the explicit registry accepts the third spelling
# ---------------------------------------------------------------------------


def test_explicit_registry_accepts_a_fit_record_cell_and_refuses_ambiguity(tmp_path):
    from sisteer.eval.defense_cell import registry_from_explicit_cells

    bank = tmp_path / "view"
    p = _stage_fit_provenance(bank / CELL)
    cfg = {"explicit_cells": ["none", CELL], "defenses_dir": str(bank)}
    cells = registry_from_explicit_cells(cfg)
    assert [c.id for c in cells] == ["none", CELL]
    assert cells[1].class_ == "A" and Path(cells[1].object_path) == p
    (p.parent / "train_budget_meta.json").write_text("{}")
    with pytest.raises(ValueError, match="more than one identity sidecar"):
        registry_from_explicit_cells(cfg)
