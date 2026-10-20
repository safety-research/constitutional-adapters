"""fit_sidecars: synthetic sidecar bundles → `object` payloads that validate AND satisfy the fit_target_kind
paired-nullness CHECK once ingested (`test_reseed_fields`); refusals name the keys seen.

Fixtures are invented (placeholder shas, TESTHOST, no real transcript); a tiny `object.pt` is built with torch
when it is importable, else the blob tests are skipped.
"""
from __future__ import annotations

import json
import os

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry import validate as VAL
from sisteer.registry.adapters import fit_sidecars as FS

from registry_fit_fixtures import (CORPUS_SHA, NOW, PARENT_A, PARENT_B, POOL_CONTENT_SHA, POOL_SHA, REV, T, _blob, _sha, _write,  # noqa: E402
                                   corpus_tbm, seeds, si_tbm)


# --------------------------------------------------------------------------------------------
# ingest helper: the payload must pass the DDL CHECK, not only the validator
# --------------------------------------------------------------------------------------------


def _ingest_objects(tmp_path, payloads, extra_dims=()):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    dims = synth.seed_dimensions(with_fitted_object=False)
    dims += [synth.rec("object", synth.object_lora(sha=PARENT_A), T(20)), synth.rec("object", synth.object_lora(sha=PARENT_B, paper_label="control_lora", role="control"), T(21))]
    dims += list(extra_dims)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), dims)
    # the PRODUCER's statement of every object the bank payload states no status for: stamped before the bank's record, so the
    # bank's omission folds onto the producer's row and keeps its status
    producer = [synth.rec("object", dict(p, status="approved"), T(50 + i)) for i, p in enumerate(payloads) if "status" not in p]
    if producer:
        R.write_records(os.path.join(root, "records", "producer", "000001-object.jsonl"), producer)
    recs = [synth.rec("object", p, T(100 + i)) for i, p in enumerate(payloads)]
    R.write_records(os.path.join(root, "records", "objects", "000001-object.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    return root, rep


def _row(root, object_id):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        conn.row_factory = None
        cur = conn.execute("SELECT * FROM object WHERE object_id = ?", (object_id,))
        cols = [c[0] for c in cur.description]
        r = cur.fetchone()
        return dict(zip(cols, r)) if r else None
    finally:
        conn.close()


# --------------------------------------------------------------------------------------------
# the six shapes
# --------------------------------------------------------------------------------------------


def test_corpus_fit_payload_records_every_field_with_its_source(tmp_path):
    d = str(tmp_path / "bank" / "base__synth-corpus-full__lora_r4_ce")
    blob = _blob(d)
    sha = _sha(blob)
    _write(d, "train_budget_meta.json", corpus_tbm(sha))
    _write(d, "COMPLETE", {"fit_fingerprint": "0" * 64, "object_sha256": sha, "published_at": "2026-09-04T02:48:47+00:00", "files": {}})
    _write(d, "superseded_by.json", {"fit_fingerprint": "9" * 64, "superseded_by": {"expected_refit_fingerprint": "0" * 64}, "snapshot": "/vol/out/prior"})
    _write(d, "staged_from.json", {"source": "twosi-fits:out/synth/cells_all/synth-corpus-full__lora_r4_ce", "object_sha256": sha})
    b = FS.read_bundle(d, read_blob=False, bank="full22_objects", campaign_hint="sdfbase_full22")
    p = FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False)
    assert p["object_id"] == sha and p["artifact_sha256"] == sha
    assert p["object_kind"] == "lora" and p["fit_target_kind"] == "corpus_document_fit" and p["derivation"] == "none"
    assert p["fit_corpus_id"] == "synth-corpus-full" and p["fit_corpus_sha256"] == CORPUS_SHA and p["fit_pool_sha256"] == POOL_SHA
    assert p["fit_layers_json"] == [3, 7, 11, 15] and p["fit_layer_count"] == 4 and p["fit_layer_index_convention"] == "zero_based_decoder_block"
    assert p["fit_rank"] == 4 and p["fit_positions"] == "all" and p["fit_objective"] == "ce"
    assert p["fit_lr"] == 1.26e-3 and p["fit_lr_schedule"] == "warmup_cosine" and p["fit_effective_batch"] == 8
    assert p["fit_epochs_cap"] == 16 and p["fit_epochs_seen"] == 5 and p["fit_best_epoch"] == 4 and p["fit_patience"] == 1
    assert p["fit_early_stopped"] == 1 and p["fit_stopped_on"] == "patience" and p["fit_steps"] == 10845
    assert p["fit_best_heldout_loss"] == 2.04 and p["fit_final_heldout_loss"] == 2.10
    assert p["fit_model_id"] == synth.FIT_MODEL_ID and p["fit_thinking_render"] == "not_applicable"
    assert p["paper_label"] == "constitution_lora" and p["object_role"] == "constitution" and p["campaign"] == "sdfbase_full22"
    assert p["campaign_cell_id_legacy"] == "base__synth-corpus-full__lora_r4_ce"
    # this object is the expected refit of the superseded snapshot: NOT superseded -- and the bank's "approved" default is no statement
    #: the payload omits `status` (the producer's folded status stands) and parks the evidence under extra
    assert "status" not in p and "status" not in p["field_confidence_json"] and p["extra"]["supersedes_fit_fingerprint"] == "9" * 64
    assert p["extra"]["status_unstated"].startswith("the bank states no status for this object (D721)") and "expected refit" in p["extra"]["status_unstated"]
    fc = p["field_confidence_json"]
    assert fc["fit_layers_json"] == {"confidence": "recorded", "source": "train_budget_meta.json:target_layers"}
    assert fc["fit_corpus_sha256"]["source"] == "train_budget_meta.json:sample_manifest_sha"
    assert fc["fit_seed"]["confidence"] == "unverified" and "seed" in fc["fit_seed"]["reason"] and p["fit_seed_confidence"] == "unverified"
    assert fc["fit_chat_template_sha256"]["confidence"] == "unverified" and p["extra"]["fit_chat_template_sha16"] == "a4aee8afcf2e0711"
    assert fc["fit_rank"]["confidence"] == "derived"     # blob not read: the rank comes from the arm spelling
    # every hyperparameter verbatim
    hp = p["fit_hyperparameters_json"]
    assert hp["lr"] == 1.26e-3 and hp["warmup_frac"] == 0.15 and hp["accum_steps"] == 4 and hp["early_stop"]["patience"] == 1
    assert hp["continuation"]["new_cap"] == 16
    assert [l["uri"] for l in p["artifact_locations_json"]][:2] == [blob, "twosi-fits:out/synth/cells_all/synth-corpus-full__lora_r4_ce"]
    # validates and passes the paired-nullness CHECK
    v = VAL.validate_record(synth.rec("object", p, NOW), "<memory>")
    assert isinstance(v, VAL.Validated), v
    root, rep = _ingest_objects(tmp_path, [p])
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    row = _row(root, sha)
    assert row["fit_target_kind"] == "corpus_document_fit" and json.loads(row["fit_layers_json"]) == [3, 7, 11, 15]


def test_si_distillation_payload_and_check(tmp_path):
    d = str(tmp_path / "mirror" / "synthetic_si__psr_kl_all")
    sha = _sha(_blob(d, b"psr bytes"))
    _write(d, "train_budget_meta.json", si_tbm(sha))
    _write(d, "object.pt.provenance.json", {"kind": "psr", "layers": list(range(8))})
    _write(d, "metrics.jsonl", '{"step": 0}\n')
    b = FS.read_bundle(d, read_blob=False, bank="fits_mirror", campaign_hint="two_si")
    p = FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False)
    assert p["object_kind"] == "relu_steer" and p["fit_target_kind"] == "system_instruction_distillation"
    assert p["fit_teacher_si_id"] == "synthetic_si" and p["fit_teacher_si_text_sha256"] == synth.SI_SHA
    assert p["fit_prompt_pool_id"] == "synthmodel/pairs_synthetic_si" and p["fit_prompt_pool_sha256"] == POOL_CONTENT_SHA
    assert p["fit_objective"] == "kl" and p["fit_positions"] == "response_incl_assistant_header" and p["fit_response_header_len"] == 3
    assert p["fit_capture_site"] == "resid_post" and "fit_corpus_id" not in p and p["fit_stopped_on"] == "epoch_cap"
    assert p["fit_heldout_curve_uri"].endswith("metrics.jsonl") and p["paper_label"] == "safety_relu" and p["object_role"] == "safety"
    # the two-SI Class A record carries no lr: unverified, naming the keys looked for (a search link may fill it later)
    assert "fit_lr" not in p and "train_budget_meta.json:lr" in p["field_confidence_json"]["fit_lr"]["reason"]
    assert p["fit_epochs_cap"] == 10 and p["field_confidence_json"]["fit_epochs_cap"]["confidence"] == "derived"
    assert isinstance(VAL.validate_record(synth.rec("object", p, NOW), "<memory>"), VAL.Validated)
    root, rep = _ingest_objects(tmp_path, [p])
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    assert _row(root, sha)["fit_teacher_si_id"] == "synthetic_si"


def test_derived_subtraction_from_derived_meta(tmp_path):
    d = str(tmp_path / "bank" / "derived__synth-corpus-full__lora_r4_ce_minus_ctl")
    sha = _sha(_blob(d, b"sub"))
    _write(d, "derived_meta.json", {"kind": "lora_subtraction", "derived_id": "synth-corpus-full__lora_r4_ce_minus_ctl", "derivation": "delta(pos) - delta(neg)",
                                    "model_id": "synthetic/base-4b", "model_revision": REV, "positions": "all", "layers": [3, 7], "rank": 4,
                                    "parent_pos": {"cell_id": "a", "object_sha256": PARENT_A, "fit_fingerprint": "0" * 64, "data_scale": {"official_tokens": 42}},
                                    "parent_neg": {"cell_id": "b", "object_sha256": PARENT_B},
                                    "scale_matched": True, "built_at": "2026-09-04T18:09:45+00:00", "builder": "scripts/sdfbase/lora_subtract.py", "object_sha256": sha})
    b = FS.read_bundle(d, read_blob=False, bank="full22_objects", campaign_hint="sdfbase_full22")
    p = FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False)
    assert p["object_kind"] == "lora_subtraction" and p["fit_target_kind"] == "derived" and p["derivation"] == "lora_subtraction"
    assert p["parent_object_ids_json"] == [PARENT_A, PARENT_B] and p["fit_positions"] == "inherited_from_parents"
    assert p["fit_objective"] == "not_applicable" and p["fit_layer_selection_rule"] == "inherited" and p["fit_rank"] == 4
    assert p["fit_lr_selected_by"] == f"inherited_from:{PARENT_A}" and p["fit_finished_utc"] == "2026-09-04T18:09:45.000Z"
    assert p["paper_label"] == "difference_lora" and p["object_role"] == "difference" and p["fit_writer"] == "scripts/sdfbase/lora_subtract.py"
    assert "fit_corpus_id" not in p and "fit_teacher_si_id" not in p
    assert isinstance(VAL.validate_record(synth.rec("object", p, NOW), "<memory>"), VAL.Validated)
    root, rep = _ingest_objects(tmp_path, [p])
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals


def test_subtraction_anchor_alone_reads_as_the_subtraction_and_reaches_its_parents_for_the_batch(tmp_path):
    """`scale_lora_dose.py --anchor` re-serves a subtraction's own bytes under the dose-1 staged id with the builder's record plus a
    `dose_scaled_from` stamp (anchor true, scale 1.0, parent sha == own sha) and an `object.pt.provenance.json` with dose_scale 1.0. That
    is not a dose copy: the object is the subtraction, its parents the two fits; the batch is the parents' when their fit records
    (supplied by the objects_bank adapter as `parents_in_bank`) agree, and NULL with the reason when they cannot be reached."""
    d = str(tmp_path / "bank" / "d1__synth-corpus-full__lora_r4_ce_minus_ctl")
    sha = _sha(_blob(d, b"sub"))
    meta = {"kind": "lora_subtraction", "derived_id": "synth-corpus-full__lora_r4_ce_minus_ctl", "derivation": "delta(parent_pos) - delta(parent_neg)",
            "model_id": "synthetic/base-4b", "model_revision": REV, "positions": "all", "layers": [3, 7], "rank": 4,
            "parent_pos": {"cell_id": "a", "object_sha256": PARENT_A, "fit_fingerprint": "0" * 64, "data_scale": {"official_tokens": 42}},
            "parent_neg": {"cell_id": "b", "object_sha256": PARENT_B},
            "scale_matched": True, "built_at": "2026-09-04T18:09:45+00:00", "builder": "scripts/sdfbase/lora_subtract.py", "object_sha256": sha,
            "dose_scaled_from": {"anchor": True, "scale": 1.0, "parent_object_sha256": sha, "parent_staged_id": "derived__synth-corpus-full__lora_r4_ce_minus_ctl",
                                 "parent_identity": "derived", "producer": "scripts/sdfbase/scale_lora_dose.py", "producer_commit": "5" * 40,
                                 "written_at": "2026-09-15T16:18:05Z", "scaling_method": "NONE: byte-identical copy", "tensors_scaled": [],
                                 "norms_by_tensor_and_layer": {"B_pos": {"3": {"before": 1.0, "after": 1.0}}}, "verification": {"status": "PASS", "anchor": True}}}
    _write(d, "derived_meta.json", meta)
    _write(d, "object.pt.provenance.json", {"kind": "lora_subtraction", "layers": [3, 7], "dose_scale": 1.0, "parent_object_sha256": sha})
    b = FS.read_bundle(d, read_blob=False, bank="full22_objects", campaign_hint="sdfbase_full22")
    p = FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False)                       # a direct read: no bank index
    assert p["object_kind"] == "lora_subtraction" and p["derivation"] == "lora_subtraction" and p["fit_target_kind"] == "derived"
    assert p["parent_object_ids_json"] == [PARENT_A, PARENT_B] and "base_scale" not in p and "fit_effective_batch" not in p
    assert p["derivation_params_json"]["dose_scaled_from"]["anchor"] is True and "norms_by_tensor_and_layer" not in p["derivation_params_json"]["dose_scaled_from"]
    assert p["fit_lr_selected_by"] == f"inherited_from:{PARENT_A}" and p["fit_writer"] == "scripts/sdfbase/lora_subtract.py" and p["fit_finished_utc"] == "2026-09-04T18:09:45.000Z"
    assert p["paper_label"] == "difference_lora" and p["object_role"] == "difference"
    why = p["extra"]["fit_effective_batch_unresolved"]
    assert f"parent_pos ({PARENT_A[:12]}…), parent_neg ({PARENT_B[:12]}…) are not staged in a bank on this machine" in why
    assert p["field_confidence_json"]["fit_effective_batch"]["confidence"] == "unverified"
    assert isinstance(VAL.validate_record(synth.rec("object", p, NOW), "<memory>"), VAL.Validated)
    root, rep = _ingest_objects(tmp_path, [p])
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    row = _row(root, sha)
    assert row["derivation"] == "lora_subtraction" and json.loads(row["parent_object_ids_json"]) == [PARENT_A, PARENT_B] and row["fit_effective_batch"] is None
    # with the bank's fit records: the parents' batch when they agree
    bank = {PARENT_A: {"dir": "/synthetic/bank/a", "train_budget_meta": corpus_tbm(PARENT_A)}, PARENT_B: {"dir": "/synthetic/bank/b", "train_budget_meta": corpus_tbm(PARENT_B)}}
    p2 = FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False, parents_in_bank=bank)
    assert p2["fit_effective_batch"] == 8 and p2["field_confidence_json"]["fit_effective_batch"]["confidence"] == "derived"
    assert p2["extra"]["parent_fit_effective_batch"]["parent_neg"]["source"] == "/synthetic/bank/b/train_budget_meta.json:effective_batch"
    assert "fit_effective_batch_unresolved" not in p2["extra"]
    # every other field reads the same with or without the index
    assert {k: v for k, v in p.items() if k not in ("extra", "field_confidence_json", "fit_effective_batch")} == \
        {k: v for k, v in p2.items() if k not in ("extra", "field_confidence_json", "fit_effective_batch")}


def _paired_write_bundle(tmp_path):
    d = str(tmp_path / "bank" / "derived__synthetic_si__psr_paired_write")
    sha = _sha(_blob(d, b"pw"))
    _write(d, "derived_meta.json", {"kind": "psr_paired_write", "derived_id": "synthetic_si__psr_paired_write", "model_id": "synthetic/base-4b",
                                    "model_revision": REV, "positions": "all", "layers": [3, 7],
                                    "parent_pos": {"cell_id": "a", "object_sha256": PARENT_A}, "parent_neg": {"cell_id": "b", "object_sha256": PARENT_B},
                                    "built_at": "2026-09-16T00:00:00+00:00", "builder": "scripts/sdfbase/paired_write.py", "object_sha256": sha})
    return FS.read_bundle(d, read_blob=False, bank="full22_objects", campaign_hint="sdfbase_full22")


def _paired_write_seeds(derivation="composition"):
    s = seeds()
    s.object_kind_by_spelling["psr_paired_write"] = {"object_kind": "composed", "derivation": derivation, "fit_rank": None, "paper_label": None,
                                                     "fit_objective": None, "source": "decisions.md D37 (synthetic copy)"}
    s.paper_label_by_spelling["synthetic_si__psr_paired_write"] = {"paper_label": "other", "object_role": "safety"}
    return s


def test_paired_write_composition_comes_from_the_seeded_object_kinds_override(tmp_path):
    """derived_meta.json kind=psr_paired_write is a paired-write COMPOSITION of two ReLU-steer parents; the vocabulary spells no
    such derived kind, so legacy_spellings.yaml object_kinds rules it composed / composition. The derived branch used to refuse BEFORE
    reading that override — its own message told the operator to add one — so the seeded ruling never took effect."""
    b = _paired_write_bundle(tmp_path)
    with pytest.raises(FS.SidecarRefusal, match="object_kinds override"):           # no override: the refusal still names the remedy
        FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False)
    p = FS.build_object_payload(b, _paired_write_seeds(), NOW, blob_reads_enabled=False)
    assert p["object_kind"] == "composed" and p["fit_target_kind"] == "derived" and p["derivation"] == "composition"
    assert p["parent_object_ids_json"] == [PARENT_A, PARENT_B] and p["fit_positions"] == "inherited_from_parents"
    assert p["fit_layer_selection_rule"] == "inherited" and p["fit_objective"] == "not_applicable"
    fc = p["field_confidence_json"]
    assert fc["object_kind"]["confidence"] == "derived" and "legacy_spellings.yaml object_kinds[psr_paired_write]" in fc["object_kind"]["source"]
    assert "object_kinds[psr_paired_write].derivation" in fc["derivation"]["source"]
    assert isinstance(VAL.validate_record(synth.rec("object", p, NOW), "<memory>"), VAL.Validated)
    root, rep = _ingest_objects(tmp_path, [p])
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    assert _row(root, p["object_id"])["object_kind"] == "composed"


def test_override_naming_a_derivation_outside_the_vocabulary_refuses(tmp_path):
    with pytest.raises(FS.SidecarRefusal, match="names no derivation in the vocabulary.*not one of"):
        FS.build_object_payload(_paired_write_bundle(tmp_path), _paired_write_seeds(derivation="banana"), NOW, blob_reads_enabled=False)


def test_dose_copy_is_derived_with_the_parent_and_the_scale(tmp_path):
    d = str(tmp_path / "bank" / "d0p5__synth-corpus-full__lora_r4_ce")
    sha = _sha(_blob(d, b"half"))
    tbm = corpus_tbm(sha)
    tbm["dose_scaled_from"] = {"scale": 0.5, "parent_object_sha256": PARENT_A, "producer": "scripts/sdfbase/scale_lora_dose.py", "producer_commit": "5" * 40,
                               "written_at": "2026-09-15T16:17:45Z", "parent_dir": "/home/synthetic/bank/base__x", "scaling_method": "B x scale", "tensors_scaled": ["B"],
                               "note": "a post-hoc dose-scaled copy of the parent object"}
    _write(d, "train_budget_meta.json", tbm)
    _write(d, "object.pt.provenance.json", {"kind": "lora_all", "layers": [3, 7, 11, 15], "dose_scale": 0.5, "parent_object_sha256": PARENT_A})
    b = FS.read_bundle(d, read_blob=False, bank="full22_objects", campaign_hint="sdfbase_full22")
    p = FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False)
    assert p["fit_target_kind"] == "derived" and p["derivation"] == "dose_scale" and p["base_scale"] == 0.5
    # the derivation rule read the dose stamp itself (the corpus train_budget_meta.json is the PARENT's fit record)
    ftk = p["field_confidence_json"]["fit_target_kind"]
    assert ftk["confidence"] == "recorded" and "dose_scale" in ftk["source"] and "dose_scaled_from" in ftk["source"]
    assert p["parent_object_ids_json"] == [PARENT_A] and p["derivation_params_json"]["scale"] == 0.5
    assert "fit_corpus_id" not in p and "fit_lr" not in p and p["fit_seed_confidence"] == "not_applicable"
    assert p["fit_code_sha"] == "5" * 40 and p["fit_writer"] == "scripts/sdfbase/scale_lora_dose.py" and p["fit_substrate"] == "grading_host"
    assert p["fit_finished_utc"] == "2026-09-15T16:17:45.000Z" and p["object_role"] == "constitution"
    # the legacy reading (the default, relu_paired's): the batch is neither stated nor given a confidence entry -- "nothing was fit"
    assert "fit_effective_batch" not in p and "fit_effective_batch" not in p["field_confidence_json"] and "dose_parent_fit_effective_batch" not in p["extra"]
    assert isinstance(VAL.validate_record(synth.rec("object", p, NOW), "<memory>"), VAL.Validated)
    root, rep = _ingest_objects(tmp_path, [p])
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals


def _dose_copy_bundle(tmp_path, name="d0p5__synth-corpus-full__lora_r4_ce", parent=PARENT_A, tbm_over=None):
    """A two-SI dose copy as `scale_lora_dose.py` leaves it: the PARENT's `train_budget_meta.json` (effective_batch 8 unless overridden)
    plus the `dose_scaled_from` stamp, and the provenance naming the parent."""
    d = str(tmp_path / "bank" / name)
    sha = _sha(_blob(d, b"half " + name.encode()))
    tbm = corpus_tbm(sha, **(tbm_over or {}))
    tbm["dose_scaled_from"] = {"scale": 0.5, "parent_object_sha256": parent, "producer": "scripts/sdfbase/scale_lora_dose.py", "producer_commit": "5" * 40,
                               "written_at": "2026-09-15T16:17:45Z", "parent_dir": "/home/synthetic/bank/base__x", "scaling_method": "B x scale",
                               "tensors_scaled": ["B"], "note": "a post-hoc dose-scaled copy of the parent object"}
    _write(d, "train_budget_meta.json", tbm)
    _write(d, "object.pt.provenance.json", {"kind": "lora_all", "layers": [3, 7, 11, 15], "dose_scale": 0.5, "parent_object_sha256": parent})
    return FS.read_bundle(d, read_blob=False, bank="full22_objects", campaign_hint="sdfbase_full22"), sha


#: the derived entry's source, word for word (the registry owner's single-source decision): the sidecar hook's dose
#: path writes the same words, and `reconcile fit-effective-batch` reads the entry by its confidence, never by this text or its writer
DERIVED_SOURCE = (f"the parent's effective batch ({PARENT_A[:12]}…), read from train_budget_meta.json:effective_batch (the PARENT's fit record copied "
                  "beside the copy); a dose-scaled copy was not fit, so its effective batch is its PARENT's (the user 2026-10-13)")


def test_dose_copy_states_the_parents_effective_batch_as_derived_from_the_copied_record(tmp_path):
    """With `dose_batch_from_parent` -- the objects bank's switch -- a dose copy's `fit_effective_batch` is its PARENT's effective batch,
    marked `derived` with the parent and the rule, read from ONE source: the parent's fit record copied beside the copy (the copy's own
    `train_budget_meta.json`, the record the block is read from, so the value is the block's own). The bank index of the parents' own
    records changes nothing: the adapter reads the copy's own files alone; agreement with the parent's registry row is reconcile's
    read-only cross-check. The row folds with the value (the derived kind's paired-nullness rule leaves the column free), beside the
    block it agrees with by construction."""
    b, sha = _dose_copy_bundle(tmp_path)
    p = FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False, dose_batch_from_parent=True)
    assert p["fit_effective_batch"] == 8 and p["fit_hyperparameters_json"]["effective_batch"] == 8
    fc = p["field_confidence_json"]["fit_effective_batch"]
    assert fc == {"confidence": "derived", "source": DERIVED_SOURCE}
    ev = p["extra"]["dose_parent_fit_effective_batch"]
    assert ev == {"parent_object_sha256": PARENT_A, "unresolved": None,
                  "copied_record": {"effective_batch": 8, "source": "train_budget_meta.json:effective_batch (the PARENT's fit record copied beside the copy)"}}
    # the top-level reason is STATED as null on a resolved emission (the fold merges extra by key: an earlier unresolved emission's reason
    # would otherwise stand beside the derived integer for good -- review round sol-2)
    assert "fit_effective_batch_unresolved" in p["extra"] and p["extra"]["fit_effective_batch_unresolved"] is None
    # ONE source: a parent record in the bank index -- agreeing, disagreeing or batchless -- is not read for a dose copy
    for bank_tbm in (corpus_tbm(PARENT_A), corpus_tbm(PARENT_A, effective_batch=64), {k: v for k, v in corpus_tbm(PARENT_A).items() if k != "effective_batch"}):
        p_bank = FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False, dose_batch_from_parent=True,
                                         parents_in_bank={PARENT_A: {"dir": "/synthetic/bank/a", "train_budget_meta": bank_tbm}})
        assert p_bank["fit_effective_batch"] == 8 and p_bank["field_confidence_json"]["fit_effective_batch"] == fc and p_bank["extra"] == p["extra"]
    # every other field reads the same as the legacy reading
    legacy = FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False)
    assert {k: v for k, v in legacy.items() if k not in ("extra", "field_confidence_json")} == \
        {k: v for k, v in p.items() if k not in ("extra", "field_confidence_json", "fit_effective_batch")}
    assert isinstance(VAL.validate_record(synth.rec("object", p, NOW), "<memory>"), VAL.Validated)
    root, rep = _ingest_objects(tmp_path, [p])
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    row = _row(root, sha)
    assert row["fit_effective_batch"] == 8 and row["derivation"] == "dose_scale" and json.loads(row["fit_hyperparameters_json"])["effective_batch"] == 8
    assert json.loads(row["field_confidence_json"])["fit_effective_batch"] == {"confidence": "derived", "source": DERIVED_SOURCE}


def test_the_derived_entrys_source_text_is_exactly_the_ruled_wording():
    """The source text is a contract: the sidecar hook's dose path (another branch) writes the same words for the copies it files, and
    the pass recognises the statement by the entry's confidence -- so the adapter's wording is pinned word for word, the parent named by
    the first twelve hex digits of its sha and an ellipsis."""
    assert FS.dose_batch_derived_source(PARENT_A) == DERIVED_SOURCE
    assert FS.dose_batch_derived_source(PARENT_B) == ("the parent's effective batch (" + PARENT_B[:12] + "…), read from train_budget_meta.json:effective_batch "
                                                      "(the PARENT's fit record copied beside the copy); a dose-scaled copy was not fit, so its effective "
                                                      "batch is its PARENT's (the user 2026-10-13)")
    p: dict = {}
    extra: dict = {}
    conf = FS._Conf()
    FS._dose_batch_from_parent(p, conf, extra, {"parent_object_sha256": PARENT_B}, corpus_tbm(PARENT_B, effective_batch=64))
    assert p == {"fit_effective_batch": 64} and conf.entries["fit_effective_batch"] == {"confidence": "derived", "source": FS.dose_batch_derived_source(PARENT_B)}


def test_dose_copy_batch_is_null_by_name_when_the_copied_record_names_no_integer_or_the_copy_carries_none(tmp_path):
    """The ruling's other half. A record PRESENT is a reading -- its value, or its lack of one -- never skipped: a copied record that
    names no `effective_batch` (the key absent or null), or one that is not an integer in its own type (an integral 64.0, a numeric "8", a
    boolean: nothing coerced, the value and its type named), leaves the column NULL by name -- stated as an explicit null in the payload,
    the reason in `extra.fit_effective_batch_unresolved` and an `unverified` entry -- and so does a copy carrying no `train_budget_meta.json`
    at all (a dose copy of a derived object's bytes: `derived_meta.json` beside it). Never a guess, never a value from another directory."""
    # (1) the copied record is present and names no effective_batch: the key null, then the key gone
    for name, over in (("dn0p5__synth-corpus-full__lora_r4_ce", {"effective_batch": None}), ("dg0p5__synth-corpus-full__lora_r4_ce", "drop")):
        b, _ = _dose_copy_bundle(tmp_path, name=name, tbm_over=None if over == "drop" else over)
        if over == "drop":
            tbm = FS.read_json(os.path.join(b.dir, "train_budget_meta.json"))
            del tbm["effective_batch"]
            _write(b.dir, "train_budget_meta.json", tbm)
            b = FS.read_bundle(b.dir, read_blob=False, bank="full22_objects", campaign_hint="sdfbase_full22")
        p = FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False, dose_batch_from_parent=True)
        assert "fit_effective_batch" in p and p["fit_effective_batch"] is None
        why = p["extra"]["fit_effective_batch_unresolved"]
        assert why == ("a dose-scaled copy was not fit, so its effective batch is its PARENT's (the user 2026-10-13), and the parent's fit record copied beside "
                       f"the copy (train_budget_meta.json, parent {PARENT_A[:12]}…) names no effective_batch; a record present and naming none is read as "
                       "naming none; left NULL")
        assert p["field_confidence_json"]["fit_effective_batch"] == {"confidence": "unverified", "source": why, "reason": why}
        assert p["extra"]["dose_parent_fit_effective_batch"] == {"parent_object_sha256": PARENT_A, "unresolved": why,
                                                                "copied_record": {"effective_batch": None, "source": "train_budget_meta.json:effective_batch (the PARENT's fit record copied beside the copy)"}}
    # (2) a reading that is not an integer in its own type: NULL, the value and its type named, nothing coerced (strict int, not bool)
    for odd, name in ((64.0, "d0p25__synth-corpus-full__lora_r4_ce"), ("8", "d2__synth-corpus-full__lora_r4_ce"), (True, "d4__synth-corpus-full__lora_r4_ce")):
        b_odd, _ = _dose_copy_bundle(tmp_path, name=name, tbm_over={"effective_batch": odd})
        p_odd = FS.build_object_payload(b_odd, seeds(), NOW, blob_reads_enabled=False, dose_batch_from_parent=True)
        assert "fit_effective_batch" in p_odd and p_odd["fit_effective_batch"] is None
        why = p_odd["extra"]["fit_effective_batch_unresolved"]
        assert why == ("a dose-scaled copy was not fit, so its effective batch is its PARENT's (the user 2026-10-13), and the parent's fit record copied beside "
                       f"the copy (train_budget_meta.json, parent {PARENT_A[:12]}…) names effective_batch {odd!r} ({type(odd).__name__}), not an integer; "
                       "nothing is coerced; left NULL")
        assert p_odd["field_confidence_json"]["fit_effective_batch"] == {"confidence": "unverified", "source": why, "reason": why}
        assert p_odd["extra"]["dose_parent_fit_effective_batch"]["copied_record"]["effective_batch"] == odd
    assert not FS._integral(True) and not FS._integral(64.0) and not FS._integral("8") and FS._integral(8) and FS._integral(0)
    # (3) the copy carries no train_budget_meta.json (a derived parent's dose child: derived_meta.json beside it) -- the rule alone, on the
    # builder's inputs; a parent record in the bank index would not be read either
    for bank in (None, {PARENT_A: {"dir": "/synthetic/bank/a", "train_budget_meta": corpus_tbm(PARENT_A)}}):
        p_none: dict = {}
        extra_none: dict = {}
        conf_none = FS._Conf()
        FS._dose_batch_from_parent(p_none, conf_none, extra_none, {"parent_object_sha256": PARENT_A}, None)
        assert p_none == {"fit_effective_batch": None} and conf_none.entries["fit_effective_batch"]["confidence"] == "unverified"
        why = extra_none["fit_effective_batch_unresolved"]
        assert why == ("a dose-scaled copy was not fit, so its effective batch is its PARENT's (the user 2026-10-13), and the copy carries no train_budget_meta.json "
                       f"(the parent {PARENT_A[:12]}… is a derived object with no fit record of its own to be copied beside the copy); left NULL")
        assert extra_none["dose_parent_fit_effective_batch"] == {"parent_object_sha256": PARENT_A, "copied_record": None, "unresolved": why}
    # the NULL-by-name row still validates and folds: the column NULL, the reason on the row
    assert isinstance(VAL.validate_record(synth.rec("object", p_odd, NOW), "<memory>"), VAL.Validated)
    root, rep = _ingest_objects(tmp_path, [p_odd])
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    row = _row(root, p_odd["object_id"])
    assert row["fit_effective_batch"] is None and "not an integer" in json.loads(row["extra_json"])["fit_effective_batch_unresolved"]
    assert json.loads(row["field_confidence_json"])["fit_effective_batch"]["confidence"] == "unverified"


def test_a_dose_copy_re_emitted_unresolved_nulls_its_derived_batch_and_reconcile_writes_no_correction(tmp_path):
    """The sequence the fold must survive. A dose copy is emitted RESOLVED (the copied record 8: `derived` 8) and folded; then its copied
    record is re-stamped with a value the adapter refuses (effective_batch 64.0, a float -- the block carries it verbatim). Had the
    re-emission OMITTED the column, the fold would have kept the derived 8 beside a block saying 64.0 while the `unverified` entry replaced
    the `derived` one -- a disagreement `reconcile fit-effective-batch` would select and refuse as unclassified, standing forever. Stated
    as null, the fold writes NULL, the pass selects nothing and lists the row as left NULL by name, `--apply` writes no correction, and the
    verify is clean (a NULL copy has no derived value for the parent cross-check either)."""
    from sisteer.registry import reconcile as RC
    b, sha = _dose_copy_bundle(tmp_path, name="seq0p5__synth-corpus-full__lora_r4_ce")
    first = FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False, dose_batch_from_parent=True)
    assert first["fit_effective_batch"] == 8 and first["field_confidence_json"]["fit_effective_batch"]["confidence"] == "derived"
    root, rep = _ingest_objects(tmp_path, [first])
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    assert _row(root, sha)["fit_effective_batch"] == 8
    # the copied record moves to a value the adapter refuses
    tbm = FS.read_json(os.path.join(b.dir, "train_budget_meta.json"))
    tbm["effective_batch"] = 64.0
    _write(b.dir, "train_budget_meta.json", tbm)
    b2 = FS.read_bundle(b.dir, read_blob=False, bank="full22_objects", campaign_hint="sdfbase_full22")
    second = FS.build_object_payload(b2, seeds(), NOW, blob_reads_enabled=False, dose_batch_from_parent=True)
    assert "fit_effective_batch" in second and second["fit_effective_batch"] is None and second["fit_hyperparameters_json"]["effective_batch"] == 64.0
    assert second["field_confidence_json"]["fit_effective_batch"]["confidence"] == "unverified" and "64.0 (float)" in second["extra"]["fit_effective_batch_unresolved"]
    assert isinstance(VAL.validate_record(synth.rec("object", second, NOW), "<memory>"), VAL.Validated)
    R.write_records(os.path.join(root, "records", "objects", "000002-object.jsonl"), [synth.rec("object", second, T(200))])
    rep2 = H.ingest_once(root, workers=0, min_free=0)
    assert rep2.refused == 0 and rep2.blocked == 0, rep2.refusals
    row = _row(root, sha)
    assert row["fit_effective_batch"] is None and json.loads(row["fit_hyperparameters_json"])["effective_batch"] == 64.0
    conf = json.loads(row["field_confidence_json"])["fit_effective_batch"]
    assert conf["confidence"] == "unverified" and "64.0 (float)" in conf["reason"]
    assert json.loads(row["extra_json"])["dose_parent_fit_effective_batch"]["unresolved"] == conf["reason"]
    # the pass: nothing selected (a NULL is no disagreement), the row listed as left NULL by name, nothing written on --apply, nothing to verify
    dry = RC.reconcile_fit_effective_batch(root)
    assert dry.rows == [] and [(g.fit_writer, g.recorded, g.stated, g.rows) for g in dry.ambiguous] == [("scripts/sdfbase/scale_lora_dose.py", 64.0, "unverified", 1)]
    assert dry.ambiguous[0].members == [(sha, conf["reason"])] and dry.parent_checks == []
    app = RC.reconcile_fit_effective_batch(root, apply=True)
    assert app.rows == [] and app.written == [] and app.held == []
    assert not os.path.isdir(os.path.join(root, "records", "registry_owner"))
    ver = RC.verify_fit_effective_batch(root)
    assert ver.rows == [] and ver.listed == [] and ver.parent_checks == [] and ver.statement_checks == []     # NULL beside unverified: the adapter's statement to the letter


def test_a_dose_copy_repaired_after_an_unresolved_emission_folds_with_a_null_reason_never_left_null_beside_the_integer(tmp_path):
    """Review round sol-2 (2026-10-13): the mirror of the sequence above. A dose copy is emitted UNRESOLVED (its copied record names
    effective_batch 64.0: the column NULL by name, `extra.fit_effective_batch_unresolved` the reason) and folded; then the copied record is
    repaired to the integer 8 and the copy re-emitted. The successful path set the column and the derived entry but did not restate the
    top-level reason, and the fold merges `extra` by key, so the folded row said "left NULL" beside a valid derived 8 for good -- contradictory
    provenance after a normal repair, while `dose_parent_fit_effective_batch.unresolved` (rewritten whole) said null. The resolved emission
    states the key as null: the fold writes null over the reason, the derived integer stands beside a null reason in both places, the pass
    reads the copy consistent (its statement check included), lists nothing and verifies clean, and a rebuild agrees byte for byte."""
    from sisteer.registry import reconcile as RC
    b, sha = _dose_copy_bundle(tmp_path, name="rep0p5__synth-corpus-full__lora_r4_ce", tbm_over={"effective_batch": 64.0})
    first = FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False, dose_batch_from_parent=True)
    assert first["fit_effective_batch"] is None and first["field_confidence_json"]["fit_effective_batch"]["confidence"] == "unverified"
    why = first["extra"]["fit_effective_batch_unresolved"]
    assert "64.0 (float)" in why and first["extra"]["dose_parent_fit_effective_batch"]["unresolved"] == why
    root, rep = _ingest_objects(tmp_path, [first])
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    row = _row(root, sha)
    assert row["fit_effective_batch"] is None and json.loads(row["extra_json"])["fit_effective_batch_unresolved"] == why
    # the copied record is repaired to the integer the trainer records; the re-emission resolves
    tbm = FS.read_json(os.path.join(b.dir, "train_budget_meta.json"))
    tbm["effective_batch"] = 8
    _write(b.dir, "train_budget_meta.json", tbm)
    b2 = FS.read_bundle(b.dir, read_blob=False, bank="full22_objects", campaign_hint="sdfbase_full22")
    second = FS.build_object_payload(b2, seeds(), NOW, blob_reads_enabled=False, dose_batch_from_parent=True)
    assert second["fit_effective_batch"] == 8 and second["fit_hyperparameters_json"]["effective_batch"] == 8
    assert second["field_confidence_json"]["fit_effective_batch"] == {"confidence": "derived", "source": DERIVED_SOURCE}
    assert "fit_effective_batch_unresolved" in second["extra"] and second["extra"]["fit_effective_batch_unresolved"] is None      # stated, as null
    assert second["extra"]["dose_parent_fit_effective_batch"]["unresolved"] is None
    assert isinstance(VAL.validate_record(synth.rec("object", second, NOW), "<memory>"), VAL.Validated)
    R.write_records(os.path.join(root, "records", "objects", "000002-object.jsonl"), [synth.rec("object", second, T(200))])
    rep2 = H.ingest_once(root, workers=0, min_free=0)
    assert rep2.refused == 0 and rep2.blocked == 0, rep2.refusals
    row = _row(root, sha)
    extra = json.loads(row["extra_json"])
    assert row["fit_effective_batch"] == 8 and json.loads(row["fit_hyperparameters_json"])["effective_batch"] == 8
    assert json.loads(row["field_confidence_json"])["fit_effective_batch"] == {"confidence": "derived", "source": DERIVED_SOURCE}
    # the folded reason is null in BOTH places: the key stands (the fold merges by key) and says nothing is unresolved
    assert "fit_effective_batch_unresolved" in extra and extra["fit_effective_batch_unresolved"] is None
    assert extra["dose_parent_fit_effective_batch"] == {"parent_object_sha256": PARENT_A, "unresolved": None,
                                                        "copied_record": {"effective_batch": 8, "source": "train_budget_meta.json:effective_batch (the PARENT's fit record copied beside the copy)"}}
    # the pass: in neither class, no ambiguous row, the statement check consistent (derived 8 = the block's own 8), nothing listed, verify clean
    dry = RC.reconcile_fit_effective_batch(root)
    assert dry.rows == [] and dry.ambiguous == [] and dry.nothing_to_compare == 0
    assert [(c.object_id, c.status, c.value, c.recorded) for c in dry.statement_checks] == [(sha, RC.BATCH_STATEMENT_CONSISTENT, 8, 8)]
    assert dry.statement_inconsistencies == [] and [c.object_id for c in dry.parent_checks] == [sha] and dry.parent_disagreements == []
    ver = RC.verify_fit_effective_batch(root)
    assert ver.listed == [] and RC.format_batch_report(ver).endswith("inconsistent 0 row(s); the class is consistent")
    # a rebuild from the log folds the same row (the null reason included)
    live = DB.connect(DB.db_path(root), readonly=True)
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, out, workers=0)
    rebuilt = DB.connect(out, readonly=True)
    diff = {k: v for k, v in H.verify_equal(live, rebuilt).items() if v}
    assert not diff, diff
    live.close()
    rebuilt.close()


#: a LoRA SUBTRACTION, the parent of a dose child that carries no fit record: a derived object, no row of the fixtures' registry
PARENT_SUB = "3" * 64


def _subtraction_dose_child_bundle(tmp_path, name="d0p5__synth-corpus-full__lora_r4_ce_minus_ctl", parent=PARENT_SUB):
    """A dose child of a LoRA subtraction as `scale_lora_dose.py` leaves it: the parent's `derived_meta.json` (`lora_subtract.py`'s record,
    its `object_sha256` the copy's own) plus a non-anchor `dose_scaled_from` stamp naming the parent, the provenance naming it too -- and
    NO `train_budget_meta.json`: the parent is a derived object with no fit record of its own to be copied beside the copy."""
    d = str(tmp_path / "bank" / name)
    sha = _sha(_blob(d, b"half " + name.encode()))
    meta = {"kind": "lora_subtraction", "derived_id": "synth-corpus-full__lora_r4_ce_minus_ctl", "derivation": "delta(parent_pos) - delta(parent_neg)",
            "model_id": "synthetic/base-4b", "model_revision": REV, "positions": "all", "layers": [3, 7], "rank": 4,
            "parent_pos": {"cell_id": "a", "object_sha256": PARENT_A, "fit_fingerprint": "0" * 64, "data_scale": {"official_tokens": 42}},
            "parent_neg": {"cell_id": "b", "object_sha256": PARENT_B},
            "scale_matched": True, "built_at": "2026-09-04T18:09:45+00:00", "builder": "scripts/sdfbase/lora_subtract.py", "object_sha256": sha,
            "dose_scaled_from": {"scale": 0.5, "parent_object_sha256": parent, "parent_staged_id": "derived__synth-corpus-full__lora_r4_ce_minus_ctl",
                                 "parent_identity": "derived", "producer": "scripts/sdfbase/scale_lora_dose.py", "producer_commit": "5" * 40,
                                 "written_at": "2026-09-15T16:18:05Z", "scaling_method": "B x scale", "tensors_scaled": ["B_pos", "B_neg"]}}
    _write(d, "derived_meta.json", meta)
    _write(d, "object.pt.provenance.json", {"kind": "lora_subtraction", "layers": [3, 7], "dose_scale": 0.5, "parent_object_sha256": parent})
    return FS.read_bundle(d, read_blob=False, bank="full22_objects", campaign_hint="sdfbase_full22"), sha


def test_a_dose_copy_left_null_by_name_with_nothing_recorded_folds_and_reconcile_lists_it_by_name(tmp_path, capsys):
    """Review round astra (2026-10-13), through the real adapter and harvester. Three copies the adapter leaves NULL by name whose folded
    block records NO effective_batch: the copied record with the key ABSENT (the block without the key), with the key NULL (the block
    carries the null), and a subtraction's dose child with no `train_budget_meta.json` at all (no block). Each folds with the explicit
    NULL, the `unverified` entry and the reason -- and `reconcile fit-effective-batch` selected its ambiguous class by the recorded value
    alone, so all three vanished from the dry run's text and JSON (zero ambiguous, zero nothing to compare) while standing unresolved in
    the table. They are listed by object id with the adapter's own reason under the by-name statement, the recorded value `-`; nothing is
    written for them on `--apply`; the verify is clean and still names them; none is read against a parent (no derived value)."""
    from sisteer.registry import cli
    from sisteer.registry import reconcile as RC
    absent, _ = _dose_copy_bundle(tmp_path, name="da0p5__synth-corpus-full__lora_r4_ce")
    tbm = FS.read_json(os.path.join(absent.dir, "train_budget_meta.json"))
    del tbm["effective_batch"]
    _write(absent.dir, "train_budget_meta.json", tbm)
    absent = FS.read_bundle(absent.dir, read_blob=False, bank="full22_objects", campaign_hint="sdfbase_full22")
    null, _ = _dose_copy_bundle(tmp_path, name="dn0p5__synth-corpus-full__lora_r4_ce", parent=PARENT_B, tbm_over={"effective_batch": None})
    child, _ = _subtraction_dose_child_bundle(tmp_path)
    payloads = [FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False, dose_batch_from_parent=True) for b in (absent, null, child)]
    p_absent, p_null, p_child = payloads
    for p in payloads:
        assert "fit_effective_batch" in p and p["fit_effective_batch"] is None and p["field_confidence_json"]["fit_effective_batch"]["confidence"] == "unverified"
        assert p["derivation"] == "dose_scale" and p["extra"]["fit_effective_batch_unresolved"] == p["field_confidence_json"]["fit_effective_batch"]["reason"]
        assert isinstance(VAL.validate_record(synth.rec("object", p, NOW), "<memory>"), VAL.Validated)
    assert "effective_batch" not in p_absent["fit_hyperparameters_json"] and p_null["fit_hyperparameters_json"]["effective_batch"] is None
    assert "fit_hyperparameters_json" not in p_child and p_child["parent_object_ids_json"] == [PARENT_SUB] and p_child["fit_writer"] == "scripts/sdfbase/lora_subtract.py"
    why = {p["object_id"]: p["extra"]["fit_effective_batch_unresolved"] for p in payloads}
    assert "names no effective_batch" in why[p_absent["object_id"]] and f"parent {PARENT_B[:12]}…) names no effective_batch" in why[p_null["object_id"]]
    assert f"carries no train_budget_meta.json (the parent {PARENT_SUB[:12]}…" in why[p_child["object_id"]]
    root, rep = _ingest_objects(tmp_path, payloads)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    for p in payloads:
        row = _row(root, p["object_id"])
        assert row["fit_effective_batch"] is None and json.loads(row["field_confidence_json"])["fit_effective_batch"]["reason"] == why[p["object_id"]]
    assert "effective_batch" not in json.loads(_row(root, p_absent["object_id"])["fit_hyperparameters_json"])
    assert json.loads(_row(root, p_null["object_id"])["fit_hyperparameters_json"])["effective_batch"] is None and _row(root, p_child["object_id"])["fit_hyperparameters_json"] is None
    # the pass: nothing selected, nothing to compare, no parent check -- and every one of the three listed by name with its reason
    dry = RC.reconcile_fit_effective_batch(root)
    assert dry.rows == [] and dry.parent_checks == [] and dry.nothing_to_compare == 0
    listed = {oid: w for g in dry.ambiguous for oid, w in g.members}
    assert listed == why
    assert all(g.recorded is None and g.stated == "unverified" and g.statement == RC.BATCH_AMBIGUOUS_BY_NAME for g in dry.ambiguous)
    assert sorted((g.fit_writer, g.rows) for g in dry.ambiguous) == [("scripts/sdfbase/lora_subtract.py", 1), ("scripts/sdfbase/scale_lora_dose.py", 2)]
    text = RC.format_batch_report(dry)
    assert "listed by object id, not corrected): 3 row(s)" in text
    assert f"  {'scripts/sdfbase/scale_lora_dose.py':<40} {'-':<4} {synth.FIT_MODEL_ID:<20} {2:>6}  {RC.BATCH_AMBIGUOUS_BY_NAME}" in text
    for oid, w in why.items():
        assert f"\n    {oid}  {w}" in text
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--dry-run", "--json"]) == 0
    j = json.loads(capsys.readouterr().out)
    assert j["ambiguous_rows"] == 3 and {o["object_id"]: o["unresolved"] for a in j["ambiguous"] for o in a["objects"]} == why
    assert all(a["recorded"] is None for a in j["ambiguous"])
    app = RC.reconcile_fit_effective_batch(root, apply=True)
    assert app.rows == [] and app.written == [] and app.held == [] and not os.path.isdir(os.path.join(root, "records", "registry_owner"))
    ver = RC.verify_fit_effective_batch(root)
    assert ver.rows == [] and ver.listed == [] and ver.parent_checks == [] and {oid for g in ver.ambiguous for oid, _w in g.members} == set(why)


def test_contrastive_direction_from_class_b_meta(tmp_path):
    d = str(tmp_path / "wave" / "classB" / "objects_b" / "antidote")
    _blob(d, b"antidote")
    _write(d, "meta.json", {"arm": "antidote", "cell_id": "antidote", "class": "B", "fit": "si_agnostic_content_fit_once", "layer": 15, "si_id": None,
                            "content_meta": {"harmful_sha256": "a" * 64, "benign_sha256": "b" * 64, "n_harmful": 2000, "n_benign": 14000, "seed": 0}})
    _write(d, "object.json", {"method": "antidote", "model_name": "synthetic/instruct-4b", "layers": [15], "metadata": {"positions": "last"}})
    b = FS.read_bundle(d, read_blob=False, bank="stage_objects", campaign_hint="two_si")
    p = FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False)
    assert p["object_kind"] == "antidote" and p["fit_target_kind"] == "contrastive_direction" and p["fit_objective"] == "antidote"
    assert p["fit_contrast_sets_json"]["positive"]["sha256"] == "a" * 64 and p["fit_contrast_sets_json"]["negative"]["n"] == 14000
    assert p["fit_positions"] == "last_token_of_appended_assistant_header" and p["fit_layers_json"] == [15] and p["fit_layer_selection_rule"] == "published_set"
    assert p["object_role"] == "baseline_published" and p["fit_model_id"] == synth.MODEL_ID
    assert isinstance(VAL.validate_record(synth.rec("object", p, NOW), "<memory>"), VAL.Validated)
    root, rep = _ingest_objects(tmp_path, [p])
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals


def test_promotion_trio_is_an_si_distillation_with_the_trial_hyperparameters(tmp_path):
    d = str(tmp_path / "bank" / "si12__synthetic_si__psr_kl_all")
    _blob(d, b"promoted")
    _write(d, "fit_provenance.json", {"resolved_model_id": "synthetic/instruct-4b", "resolved_revision": REV, "chat_template_hash": "a4aee8afcf2e0711",
                                      "writer": "tune_arm.py", "positions_at_fit": "response"})
    _write(d, "provenance.json", {"source": "out/hpo/synth/psr_kl_all_response/r1_t02", "arm": "psr_kl_all", "si": "synthetic_si", "model": "synthetic/instruct-4b",
                                  "model_revision": REV, "lr": 2.19e-3, "clip": 0.19, "effective_batch": 64, "epochs": 10, "positions": "response",
                                  "response_header_len": 7, "fit_fingerprint": "7" * 64, "wandb_run": "abcd1234", "promoted": "2026-08-29"})
    _write(d, "summary.json", {"trial": {"index": 2, "lr": 2.19e-3, "warmup_frac": 0.15, "max_grad_norm": 0.19}, "round": 1, "status": "ok", "arm": "psr_kl_all",
                               "model": "synthetic/instruct-4b", "si": "synthetic_si", "position": "response", "epochs": 10, "teacher_forward": "precomputed_topk",
                               "best_heldout_loss": 0.015, "best_epoch": 10, "heldout_curve": [0.03, 0.02, 0.015], "stopped_early": False, "wall_s": 12876.2,
                               "wandb_run": "abcd1234", "fit_fingerprint": "7" * 64, "writer": "tune_arm.py", "template_hash": "a4aee8afcf2e0711"})
    _write(d, "object.pt.provenance.json", {"kind": "psr", "layers": list(range(4))})
    _write(d, "COMPLETE", "promoted from out/hpo/synth/psr_kl_all_response/r1_t02 on 2026-08-29")
    b = FS.read_bundle(d, read_blob=False, bank="full22_objects", campaign_hint="sdfbase_full22")
    p = FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False)
    assert p["fit_target_kind"] == "system_instruction_distillation" and p["fit_objective"] == "kl"
    assert p["fit_prompt_pool_id"] == "unverified" and p["field_confidence_json"]["fit_prompt_pool_id"]["confidence"] == "unverified"
    assert p["fit_lr"] == 2.19e-3 and p["fit_lr_selected_by"] == "search" and p["fit_wandb_run_id"] == "abcd1234" and p["fit_walltime_s"] == 12876.2
    assert p["fit_epochs_cap"] == 10 and p["fit_stopped_on"] == "epoch_cap" and p["fit_response_header_len"] == 7
    assert "promotion record read as the fit record" in p["field_confidence_json"]["fit_target_kind"]["source"]
    assert p["fit_finished_utc"] == "2026-08-29T00:00:00.000Z" and p["field_confidence_json"]["fit_finished_utc"]["confidence"] == "derived"
    assert isinstance(VAL.validate_record(synth.rec("object", p, NOW), "<memory>"), VAL.Validated)
    root, rep = _ingest_objects(tmp_path, [p])
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals


# --------------------------------------------------------------------------------------------
# The reseed campaign
# --------------------------------------------------------------------------------------------


def test_reseed_fields(tmp_path):
    d = str(tmp_path / "bank" / "base64s2__synth-corpus-full__lora_r4_ce")
    sha = _sha(_blob(d, b"reseed"))
    tbm = corpus_tbm(sha, seed=2, effective_batch=64, accum_steps=32, batch_size=2)
    tbm["fit_record"]["realized_eff_batch"] = [64, 64, 64]
    _write(d, "train_budget_meta.json", tbm)
    b = FS.read_bundle(d, read_blob=False, bank="full22_objects", campaign_hint="sdfbase_full22")
    p = FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False)
    assert p["campaign"] == "reseed"
    assert p["fit_seed"] == 2 and p["fit_seed_confidence"] == "recorded" and p["field_confidence_json"]["fit_seed"]["source"] == "train_budget_meta.json:seed"
    assert p["fit_replicate_label"] == "s2" and p["extra"]["split_salt"] == "sdfbase-heldout-v1-s2"
    assert "fit_split_seed" not in p and "salt" in p["field_confidence_json"]["fit_split_seed"]["reason"]
    assert p["fit_effective_batch"] == 64
    assert p["fit_realized_batch_json"] == {"batch_size": 2, "accum_steps": 32, "resolved_batch_size": 2, "realized_eff_batch": [64, 64, 64]}
    assert isinstance(VAL.validate_record(synth.rec("object", p, NOW), "<memory>"), VAL.Validated)


# --------------------------------------------------------------------------------------------
# refusals (never a default)
# --------------------------------------------------------------------------------------------


def test_unclassifiable_sidecar_refuses_listing_keys(tmp_path):
    d = str(tmp_path / "bank" / "candsi__Z9__lora_r4")
    sha = _sha(_blob(d, b"cand"))
    _write(d, "train_budget_meta.json", si_tbm(sha, si_id="Z9", arm="lora_r4", cell_id="Z9__lora_r4"))
    b = FS.read_bundle(d, read_blob=False, bank="full22_objects")
    with pytest.raises(FS.SidecarRefusal) as ei:
        FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False)
    assert "fit_target_kind undecidable" in str(ei.value) and "Z9" in str(ei.value) and "fit_target_overrides" in str(ei.value)


def _candidate_seeds(kind: str = "system_instruction_distillation"):
    s = seeds()
    s.fit_target_overrides = {"Z9": {"fit_target_kind": kind, "object_role": "safety", "citation": "synthetic ruling"}}
    s.si_sha_by_id["Z9"] = "9" * 64
    return s


def test_override_from_seeds_classifies_the_candidate_si(tmp_path):
    """The ruling decides when the sidecar alone cannot: a candidate-instruction fit whose record names no teacher forward."""
    d = str(tmp_path / "bank" / "candsi__Z9__lora_r4")
    sha = _sha(_blob(d, b"cand"))
    _write(d, "train_budget_meta.json", si_tbm(sha, si_id="Z9", arm="lora_r4", cell_id="Z9__lora_r4", teacher_forward=None))
    b = FS.read_bundle(d, read_blob=False, bank="full22_objects")
    p = FS.build_object_payload(b, _candidate_seeds(), NOW, blob_reads_enabled=False)
    assert p["fit_target_kind"] == "system_instruction_distillation" and p["fit_teacher_si_id"] == "Z9" and p["object_role"] == "safety"
    assert p["field_confidence_json"]["fit_target_kind"]["confidence"] == "derived" and "synthetic ruling" in p["field_confidence_json"]["fit_target_kind"]["source"]


def test_seeded_teacher_classifies_from_the_sidecar_and_an_agreeing_ruling_is_cited(tmp_path):
    """With an `si_text` row for Z9 and a teacher forward in the record, the sidecar's own bytes decide (`recorded`) —
    the legacy roster in derive.py never knew Z9 — and a ruling that agrees rides along in the source, never as the confidence."""
    d = str(tmp_path / "bank" / "candsi__Z9__lora_r4")
    sha = _sha(_blob(d, b"cand"))
    _write(d, "train_budget_meta.json", si_tbm(sha, si_id="Z9", arm="lora_r4", cell_id="Z9__lora_r4"))
    b = FS.read_bundle(d, read_blob=False, bank="full22_objects")
    p = FS.build_object_payload(b, _candidate_seeds(), NOW, blob_reads_enabled=False)
    ftk = p["field_confidence_json"]["fit_target_kind"]
    assert p["fit_target_kind"] == "system_instruction_distillation" and p["fit_teacher_si_id"] == "Z9" and p["object_role"] == "safety"
    assert ftk["confidence"] == "recorded" and "si_text row" in ftk["source"] and "agrees: synthetic ruling" in ftk["source"]
    # the same record with a ruling that names only the role classifies identically: the seeds, not the override, carry the teacher id
    s = _candidate_seeds()
    del s.fit_target_overrides["Z9"]["fit_target_kind"]
    p2 = FS.build_object_payload(FS.read_bundle(d, read_blob=False, bank="full22_objects"), s, NOW, blob_reads_enabled=False)
    ftk2 = p2["field_confidence_json"]["fit_target_kind"]
    assert p2["fit_target_kind"] == "system_instruction_distillation" and ftk2["confidence"] == "recorded" and "agrees" not in ftk2["source"]
    assert p2["object_role"] == "safety"


def test_ruling_that_disagrees_with_the_sidecar_refuses(tmp_path):
    """Two authorities in conflict — the record reads as a distillation, the ruling says corpus fit — is a refusal naming both."""
    d = str(tmp_path / "bank" / "candsi__Z9__lora_r4")
    sha = _sha(_blob(d, b"cand"))
    _write(d, "train_budget_meta.json", si_tbm(sha, si_id="Z9", arm="lora_r4", cell_id="Z9__lora_r4"))
    b = FS.read_bundle(d, read_blob=False, bank="full22_objects")
    with pytest.raises(FS.SidecarRefusal) as ei:
        FS.build_object_payload(b, _candidate_seeds("corpus_document_fit"), NOW, blob_reads_enabled=False)
    msg = str(ei.value)
    assert "fit_target_kind conflict" in msg and "system_instruction_distillation" in msg and "corpus_document_fit" in msg and "synthetic ruling" in msg


def test_unknown_teacher_text_refuses(tmp_path):
    d = str(tmp_path / "bank" / "synthetic_si__psr_kl_all")
    sha = _sha(_blob(d, b"x"))
    _write(d, "train_budget_meta.json", si_tbm(sha))
    s = seeds()
    s.si_sha_by_id = {}
    b = FS.read_bundle(d, read_blob=False, bank="fits_mirror")
    with pytest.raises(FS.SidecarRefusal) as ei:
        FS.build_object_payload(b, s, NOW, blob_reads_enabled=False)
    # with no si_text row the teacher cannot be classified at all: the refusal names the instruction
    assert "synthetic_si" in str(ei.value) and "fit_target_kind undecidable" in str(ei.value)


def test_sha_mismatch_refuses_naming_both(tmp_path):
    d = str(tmp_path / "bank" / "base__synth-corpus-full__lora_r4_ce")
    _blob(d, b"actual bytes")
    _write(d, "train_budget_meta.json", corpus_tbm("0" * 64))
    b = FS.read_bundle(d, read_blob=False, bank="full22_objects")
    with pytest.raises(FS.SidecarRefusal) as ei:
        FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False)
    assert "000000000000" in str(ei.value) and "which copy did you hash" in str(ei.value)


def test_synthetic_control_refuses(tmp_path):
    """A synthetic control direction is refused as before -- and, as an OUT-OF-SCOPE object (a SidecarRefusal subclass, so this
    `raises` still holds) citing the scope table's entry, so the objects bank can park it by name instead of refusing its group every read."""
    d = str(tmp_path / "wave" / "_controls" / "synthetic_si__diffmean_random")
    sha = _sha(_blob(d, b"rand"))
    _write(d, "train_budget_meta.json", {"arm": "diffmean_random", "cell_id": "synthetic_si__diffmean_random", "layer": 15, "model_id": "synthetic/instruct-4b",
                                         "si_id": "synthetic_si", "positions": "all", "synthetic_control": True, "control_note": "NOT A FITTED OBJECT", "object_sha256": sha})
    b = FS.read_bundle(d, read_blob=False, bank="stage_objects")
    with pytest.raises(FS.SidecarRefusal) as ei:
        FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False)
    assert "synthetic_control" in str(ei.value) and "D37" in str(ei.value)
    assert isinstance(ei.value, FS.OutOfScopeObject) and ei.value.kind_legacy == "diffmean_random"
    assert "diffmean_random" in str(ei.value.entry["legacy"]) and ei.value.entry["status"] == "withdrawn" and ei.value.entry["object_role"] == "other"
    # a `rand_*` spelling matches the table's PREFIX; with NO table in the seeds (a caller-built Seeds) the plain refusal stands, as before
    d2 = str(tmp_path / "wave" / "_controls" / "synthetic_si__rand_recipeB_s31")
    sha2 = _sha(_blob(d2, b"rand2"))
    _write(d2, "train_budget_meta.json", {"arm": "rand_recipeB_s31", "cell_id": "synthetic_si__rand_recipeB_s31", "layer": 15, "model_id": "synthetic/instruct-4b",
                                          "si_id": "synthetic_si", "positions": "all", "synthetic_control": True, "control_note": "NOT FITTED", "object_sha256": sha2})
    b2 = FS.read_bundle(d2, read_blob=False, bank="stage_objects")
    with pytest.raises(FS.OutOfScopeObject) as ei2:
        FS.build_object_payload(b2, seeds(), NOW, blob_reads_enabled=False)
    assert ei2.value.kind_legacy == "rand_recipeB_s31" and "rand_" in str(ei2.value.entry["legacy"])
    bare = seeds()
    bare.out_of_scope_by_spelling, bare.out_of_scope_prefixes = {}, ()
    with pytest.raises(FS.SidecarRefusal) as ei3:
        FS.build_object_payload(b, bare, NOW, blob_reads_enabled=False)
    assert not isinstance(ei3.value, FS.OutOfScopeObject) and "synthetic_control" in str(ei3.value)


def test_out_of_scope_kinds_raise_out_of_scope(tmp_path):
    """A bundle whose kind the table rules out of scope -- the blob's own `kind` (lofit, alphakl via its `alphakl_soft` directory,
    token_appended) or an `object.json` method (bidpo) -- raises OutOfScopeObject naming the entry; an UNKNOWN kind still refuses plainly."""
    cases = {"lofit": ("synthetic_si__lofit", {"object.pt.provenance.json": {"kind": "lofit", "layers": [15]}}, "lofit"),
             # the declared `alphakl` parks through the table's own alias of the soft-AlphaKL row (never through the directory's name)
             "alphakl": ("synthetic_si__alphakl_soft", {"object.pt.provenance.json": {"kind": "alphakl", "layers": [15]}}, "alphakl"),
             "token_appended": ("synthetic_si__token_faithful", {"object.pt.provenance.json": {"kind": "token_appended", "layers": [15]}}, "token_appended"),
             "bidpo": ("synthetic_si__bidpo", {"object.json": {"method": "bidpo", "model_name": "synthetic/instruct-4b"}, "meta.json": {"layers": [15]}}, "bidpo")}
    for label, (name, sidecars, expect_spelling) in cases.items():
        d = str(tmp_path / "wave" / "objects_a" / name)
        _blob(d, label.encode())
        for fn, obj in sidecars.items():
            _write(d, fn, obj)
        b = FS.read_bundle(d, read_blob=False, bank="stage_objects")
        with pytest.raises(FS.OutOfScopeObject) as ei:
            FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False)
        assert ei.value.kind_legacy == expect_spelling, (label, ei.value.kind_legacy)
        assert expect_spelling in str(ei.value.entry["legacy"]) and "D37" in ei.value.why and ei.value.where == d, label
    unknown = str(tmp_path / "wave" / "objects_a" / "synthetic_si__frobnicate")
    _blob(unknown, b"f")
    _write(unknown, "object.pt.provenance.json", {"kind": "frobnicate", "layers": [15]})
    b = FS.read_bundle(unknown, read_blob=False, bank="stage_objects")
    with pytest.raises(FS.SidecarRefusal) as ei:
        FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False)
    assert not isinstance(ei.value, FS.OutOfScopeObject) and "frobnicate" in str(ei.value)


def test_a_ruled_out_name_beside_an_unruled_declared_kind_refuses_plainly(tmp_path):
    """A bundle that DECLARES its kind -- the blob's or provenance's `kind`, object.json's `method` -- is parked only when the
    scope table spells THAT kind (its exact spelling or an alias the table maps). A ruled-out directory / arm_legacy /
    cell_id_legacy spelling beside an unruled declared kind is a conflict (a producer rewrote `<si>__lofit` with new contents) and refuses plainly, naming
    both, so the bank never accepts an empty revision for contents nobody ruled on and the status never attributes them to the old ruling."""
    s = seeds()
    # 1. provenance declares a kind nobody ruled; the directory, arm_legacy and cell_id_legacy all spell lofit -> a plain refusal naming the conflict
    d = str(tmp_path / "bank" / "objects_a" / "synthetic_si__lofit")
    sha = _sha(_blob(d, b"a new method's weights"))
    _write(d, "object.pt.provenance.json", {"kind": "new_unruled_method", "layers": [15], "object_sha256": sha})
    _write(d, "train_budget_meta.json", si_tbm(sha, arm="lofit", cell_id="synthetic_si__lofit"))
    b = FS.read_bundle(d, read_blob=False, bank="stage_objects")
    with pytest.raises(FS.SidecarRefusal) as ei:
        FS.build_object_payload(b, s, NOW, blob_reads_enabled=False)
    assert not isinstance(ei.value, FS.OutOfScopeObject) and ei.value.where == d
    msg = ei.value.why
    assert "object.pt.provenance.json:kind" in msg and "'new_unruled_method'" in msg and "maps to no object_kind" in msg
    assert "NOT parked" in msg and "spells 'lofit'" in msg and "not proof" in msg and "D37" in msg
    # 2. the blob's own `kind` the same way (the shape `read_bundle` fills when the blob is read): the name spells lofit, the blob does not
    b_blob = FS.read_bundle(d, read_blob=False, bank="stage_objects")
    b_blob.sidecars.pop("object.pt.provenance.json")
    b_blob.blob_meta = {"kind": "new_unruled_method", "layers": [15]}
    with pytest.raises(FS.SidecarRefusal) as ei_b:
        FS.build_object_payload(b_blob, s, NOW, blob_reads_enabled=True)
    assert not isinstance(ei_b.value, FS.OutOfScopeObject) and "blob:kind" in ei_b.value.why and "NOT parked" in ei_b.value.why
    # 3. object.json's `method` (the bidpo fits' declaration): dir + arm_legacy spell bidpo, the method is unruled -> a plain refusal naming it
    d2 = str(tmp_path / "bank" / "objects_a" / "synthetic_si__bidpo")
    sha2 = _sha(_blob(d2, b"another method"))
    _write(d2, "object.json", {"method": "new_unruled_method", "model_name": "synthetic/instruct-4b"})
    _write(d2, "meta.json", {"layers": [15]})
    _write(d2, "train_budget_meta.json", si_tbm(sha2, arm="bidpo", cell_id="synthetic_si__bidpo"))
    b2 = FS.read_bundle(d2, read_blob=False, bank="stage_objects")
    with pytest.raises(FS.SidecarRefusal) as ei2:
        FS.build_object_payload(b2, s, NOW, blob_reads_enabled=False)
    assert not isinstance(ei2.value, FS.OutOfScopeObject)
    assert "object.json:method='new_unruled_method'" in ei2.value.why and "spells 'bidpo'" in ei2.value.why and "NOT parked" in ei2.value.why
    # 4. an unruled declared kind under an UNRULED name is the plain refusal it always was (no conflict to name)
    d3 = str(tmp_path / "bank" / "objects_a" / "synthetic_si__mystery")
    sha3 = _sha(_blob(d3, b"m"))
    _write(d3, "object.pt.provenance.json", {"kind": "new_unruled_method", "layers": [15]})
    _write(d3, "train_budget_meta.json", si_tbm(sha3, arm="mystery", cell_id="synthetic_si__mystery"))
    with pytest.raises(FS.SidecarRefusal) as ei3:
        FS.build_object_payload(FS.read_bundle(d3, read_blob=False, bank="stage_objects"), s, NOW, blob_reads_enabled=False)
    assert not isinstance(ei3.value, FS.OutOfScopeObject) and "NOT parked" not in ei3.value.why and "'new_unruled_method'" in ei3.value.why
    # 5. the declaration decides the other way too: a RULED declared kind under a name nobody ruled parks, naming the declared spelling
    d4 = str(tmp_path / "bank" / "objects_a" / "synthetic_si__renamed_fit")
    sha4 = _sha(_blob(d4, b"lofit under another name"))
    _write(d4, "object.pt.provenance.json", {"kind": "lofit", "layers": [15]})
    _write(d4, "train_budget_meta.json", si_tbm(sha4, arm="renamed_fit", cell_id="synthetic_si__renamed_fit"))
    with pytest.raises(FS.OutOfScopeObject) as ei4:
        FS.build_object_payload(FS.read_bundle(d4, read_blob=False, bank="stage_objects"), s, NOW, blob_reads_enabled=False)
    assert ei4.value.kind_legacy == "lofit" and "lofit" in str(ei4.value.entry["legacy"])
    # 6. with NO declaration at all (the live lofit bundles when blob reads are off: object.pt + train_budget_meta.json only) the name is
    #    the only evidence and still parks -- the rule narrows what a DECLARED kind may be parked as, not the name-only path
    d5 = str(tmp_path / "bank" / "objects_c" / "synthetic_si__lofit")
    sha5 = _sha(_blob(d5, b"lofit weights, blob unread"))
    _write(d5, "train_budget_meta.json", si_tbm(sha5, arm="lofit", cell_id="synthetic_si__lofit"))
    with pytest.raises(FS.OutOfScopeObject) as ei5:
        FS.build_object_payload(FS.read_bundle(d5, read_blob=False, bank="stage_objects"), s, NOW, blob_reads_enabled=False)
    assert ei5.value.kind_legacy == "lofit" and "object_kind undecidable" in ei5.value.why
    # 7. the alias is the TABLE's, not the reader's: with a table that spells only `alphakl_soft`, a declared `alphakl` under the
    #    `alphakl_soft` directory is the conflict refusal, not a park through the name
    narrow = seeds()
    narrow.out_of_scope_by_spelling = {k: v for k, v in narrow.out_of_scope_by_spelling.items() if k != "alphakl"}
    d6 = str(tmp_path / "bank" / "objects_a" / "synthetic_si__alphakl_soft")
    _blob(d6, b"soft alphakl")
    _write(d6, "object.pt.provenance.json", {"kind": "alphakl", "layers": [15]})
    b6 = FS.read_bundle(d6, read_blob=False, bank="stage_objects")
    with pytest.raises(FS.SidecarRefusal) as ei6:
        FS.build_object_payload(b6, narrow, NOW, blob_reads_enabled=False)
    assert not isinstance(ei6.value, FS.OutOfScopeObject) and "spells 'alphakl_soft'" in ei6.value.why
    with pytest.raises(FS.OutOfScopeObject) as ei7:                      # and with the alias in the table, parked as the declared spelling
        FS.build_object_payload(b6, s, NOW, blob_reads_enabled=False)
    assert ei7.value.kind_legacy == "alphakl" and ei7.value.entry is not None and "alphakl_soft" in str(ei7.value.entry["legacy"])


def test_an_unruled_synthetic_control_refuses_plainly_instead_of_inheriting_the_gaussian_ruling(tmp_path):
    """`synthetic_control: true` parks only when a spelling of the control's
    OWN -- its `arm` / `cell_id`, its directory, a kind it declares -- is in the scope table. The lookup used to end on an unconditional
    `diffmean_random`, so a control under any novel spelling parked AS the Gaussian control (kind_legacy the novel spelling, the ruling the
    Gaussian row's) and the bank omitted it under a ruling that never named it. Now it refuses plainly, naming every spelling it tried;
    the genuine controls park as before, as their own spelling."""
    s = seeds()
    # 1. a novel control: arm / cell_id / directory all unruled -> a plain refusal naming them, never OutOfScopeObject
    d = str(tmp_path / "bank" / "_controls" / "synthetic_si__other_method")
    sha = _sha(_blob(d, b"some other control direction"))
    _write(d, "train_budget_meta.json", {"arm": "other_method", "cell_id": "synthetic_si__other_method", "layer": 15, "model_id": "synthetic/instruct-4b",
                                         "si_id": "synthetic_si", "positions": "all", "synthetic_control": True, "control_note": "NOT A FITTED OBJECT", "object_sha256": sha})
    with pytest.raises(FS.SidecarRefusal) as ei:
        FS.build_object_payload(FS.read_bundle(d, read_blob=False, bank="stage_objects"), s, NOW, blob_reads_enabled=False)
    assert not isinstance(ei.value, FS.OutOfScopeObject) and ei.value.where == d
    msg = ei.value.why
    assert "synthetic_control" in msg and "NOT parked" in msg and "D37" in msg
    assert "train_budget_meta.json:arm='other_method'" in msg and "cell_id='synthetic_si__other_method'" in msg and "directory='synthetic_si__other_method'" in msg
    assert "out of scope per legacy_spellings.yaml" not in msg                      # no ruling cited: none names this control
    # 2. the same control with a declared NOVEL kind (provenance, then the blob's own): still refused -- a declared kind is one more
    #    spelling to match, and this one is unruled too
    _write(d, "object.pt.provenance.json", {"kind": "other_method_vector", "layers": [15], "object_sha256": sha})
    with pytest.raises(FS.SidecarRefusal) as ei2:
        FS.build_object_payload(FS.read_bundle(d, read_blob=False, bank="stage_objects"), s, NOW, blob_reads_enabled=False)
    assert not isinstance(ei2.value, FS.OutOfScopeObject) and "object.pt.provenance.json:kind='other_method_vector'" in ei2.value.why
    b_blob = FS.read_bundle(d, read_blob=False, bank="stage_objects")
    b_blob.sidecars.pop("object.pt.provenance.json")
    b_blob.blob_meta = {"kind": "other_method_vector", "layers": [15]}
    with pytest.raises(FS.SidecarRefusal) as ei3:
        FS.build_object_payload(b_blob, s, NOW, blob_reads_enabled=True)
    assert not isinstance(ei3.value, FS.OutOfScopeObject) and "blob:kind='other_method_vector'" in ei3.value.why and "NOT parked" in ei3.value.why
    # 3. a declared RULED kind under a novel control spelling parks as the declared spelling (the bundle's own statement of its kind)
    _write(d, "object.pt.provenance.json", {"kind": "diffmean_random", "layers": [15], "object_sha256": sha})
    with pytest.raises(FS.OutOfScopeObject) as ei4:
        FS.build_object_payload(FS.read_bundle(d, read_blob=False, bank="stage_objects"), s, NOW, blob_reads_enabled=False)
    assert ei4.value.kind_legacy == "diffmean_random" and "diffmean_random" in str(ei4.value.entry["legacy"]) and "D37" in ei4.value.why
    # 4. the genuine controls park as before, as their own spelling: `diffmean_random` by arm, a `rand_*` arm through the table's prefix,
    #    and a control whose sidecar names neither arm nor cell_id through its directory's `<si>__<kind>` tail
    for name, over, expect in (("synthetic_si__diffmean_random", {"arm": "diffmean_random", "cell_id": "synthetic_si__diffmean_random"}, "diffmean_random"),
                               ("synthetic_si__rand_recipeC_s7", {"arm": "rand_recipeC_s7", "cell_id": "synthetic_si__rand_recipeC_s7"}, "rand_recipeC_s7"),
                               ("synthetic_si__rand_recipeD_s9", {}, "rand_recipeD_s9")):
        dd = str(tmp_path / "bank" / "_controls" / name)
        ssha = _sha(_blob(dd, name.encode()))
        _write(dd, "train_budget_meta.json", dict({"layer": 15, "model_id": "synthetic/instruct-4b", "si_id": "synthetic_si", "positions": "all", "synthetic_control": True,
                                                   "control_note": "NOT FITTED", "object_sha256": ssha}, **over))
        with pytest.raises(FS.OutOfScopeObject) as eip:
            FS.build_object_payload(FS.read_bundle(dd, read_blob=False, bank="stage_objects"), s, NOW, blob_reads_enabled=False)
        assert eip.value.kind_legacy == expect and "rand_" in str(eip.value.entry["legacy"]) and "D37" in eip.value.why, name
    # 5. with no table in the seeds the plain refusal stands, with nothing to cite and no conflict to name
    bare = seeds()
    bare.out_of_scope_by_spelling, bare.out_of_scope_prefixes = {}, ()
    with pytest.raises(FS.SidecarRefusal) as ei5:
        FS.build_object_payload(FS.read_bundle(d, read_blob=False, bank="stage_objects"), bare, NOW, blob_reads_enabled=False)
    assert not isinstance(ei5.value, FS.OutOfScopeObject) and "synthetic_control" in ei5.value.why and "NOT parked" not in ei5.value.why


def test_unknown_arm_refuses(tmp_path):
    d = str(tmp_path / "wave" / "objects_a" / "synthetic_si__mystery")
    sha = _sha(_blob(d, b"m"))
    _write(d, "train_budget_meta.json", si_tbm(sha, arm="mystery", cell_id="synthetic_si__mystery"))
    b = FS.read_bundle(d, read_blob=False, bank="stage_objects")
    with pytest.raises(FS.SidecarRefusal) as ei:
        FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False)
    assert "object_kind undecidable" in str(ei.value) and "mystery" in str(ei.value)


def test_missing_layers_refuses_for_a_fitted_object(tmp_path):
    d = str(tmp_path / "bank" / "synthetic_si__psr_kl_all")
    sha = _sha(_blob(d, b"nolayers"))
    tbm = si_tbm(sha)
    del tbm["target_layers"]
    _write(d, "train_budget_meta.json", tbm)
    b = FS.read_bundle(d, read_blob=False, bank="fits_mirror")
    with pytest.raises(FS.SidecarRefusal) as ei:
        FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False)
    assert "exact fit layers" in str(ei.value) and "blob_read_disabled" in str(ei.value)


def test_blob_not_local_fills_from_sidecars_and_names_the_reason(tmp_path):
    d = str(tmp_path / "bank" / "synthetic_si__psr_kl_all")
    os.makedirs(d)
    _write(d, "train_budget_meta.json", si_tbm("3" * 64))
    b = FS.read_bundle(d, read_blob=True, bank="fits_mirror")
    p = FS.build_object_payload(b, seeds(), NOW)
    assert p["artifact_sha256"] == "3" * 64 and "blob_not_local" in p["field_confidence_json"]["artifact_sha256"]["source"]
    assert p["field_confidence_json"]["artifact_bytes"] == {"confidence": "unverified", "source": "blob_not_local", "reason": "blob_not_local"}
    assert p["extra"]["blob_read"] == "blob_not_local"
    assert p["artifact_locations_json"][0]["note"].startswith("sidecars only")


# --------------------------------------------------------------------------------------------
# the blob (torch) — rank / layers / positions read from the weights file itself
# --------------------------------------------------------------------------------------------


def test_blob_metadata_read_with_torch(tmp_path):
    torch = pytest.importorskip("torch")
    d = str(tmp_path / "bank" / "base__synth-corpus-full__lora_r4_ce")
    os.makedirs(d)
    obj = {"kind": "lora_all", "layers": [3, 7], "rank": 4, "positions": "all", "response_header_len": None,
           "A": {3: torch.zeros(8, 4), 7: torch.zeros(8, 4)}, "B": {3: torch.zeros(6, 4), 7: torch.zeros(6, 4)}}
    torch.save(obj, os.path.join(d, "object.pt"))
    sha = _sha(os.path.join(d, "object.pt"))
    tbm = corpus_tbm(sha, target_layers=[3, 7], arm="lora_rX_ce", cell_id="synth-corpus-full__lora_rX_ce")   # the arm no longer spells the rank
    _write(d, "train_budget_meta.json", tbm)
    b = FS.read_bundle(d, read_blob=True, bank="full22_objects", campaign_hint="sdfbase_full22")
    assert b.blob_meta and b.blob_meta["kind"] == "lora_all" and b.blob_meta["_tensors"]["A"]["shapes"] == [[8, 4]]
    p = FS.build_object_payload(b, seeds(), NOW)
    assert p["object_kind"] == "lora" and p["field_confidence_json"]["object_kind"]["source"] == "blob:kind=lora_all"
    assert p["fit_rank"] == 4 and p["field_confidence_json"]["fit_rank"]["source"] == "blob:rank"
    assert p["extra"]["blob_keys"] == ["kind", "layers", "positions", "rank", "response_header_len"]
    assert p["artifact_bytes"] == os.path.getsize(os.path.join(d, "object.pt"))


def test_blob_rank_disagreeing_with_the_alias_is_kept_in_extra(tmp_path):
    torch = pytest.importorskip("torch")
    d = str(tmp_path / "mirror" / "synthetic_si__lora_r1_all")
    os.makedirs(d)
    torch.save({"kind": "lora_all", "layers": [1, 2], "rank": 4, "positions": "response", "response_header_len": 3,
                "A": {1: torch.zeros(4, 4)}, "B": {1: torch.zeros(4, 4)}}, os.path.join(d, "object.pt"))
    sha = _sha(os.path.join(d, "object.pt"))
    _write(d, "train_budget_meta.json", si_tbm(sha, arm="lora_r1_all", cell_id="synthetic_si__lora_r1_all", lora_rank=4, target_layers=[1, 2], site="o_proj_out"))
    b = FS.read_bundle(d, read_blob=True, bank="fits_mirror", campaign_hint="two_si")
    p = FS.build_object_payload(b, seeds(), NOW)
    assert p["fit_rank"] == 4 and p["extra"]["fit_rank_arm_spelling_disagrees"] == {"arm": "lora_r1_all", "recorded": 4}
    assert p["object_role"] == "safety" and p["paper_label"] == "other"


# --------------------------------------------------------------------------------------------
# seeds loader over the on-disk YAML shape (synthetic copy)
# --------------------------------------------------------------------------------------------


def test_load_seeds_reads_the_legacy_yaml_shape(tmp_path):
    pytest.importorskip("yaml")
    sd = str(tmp_path / "seeds")
    os.makedirs(sd)
    _write(sd, "legacy_spellings.yaml", "\n".join([
        "object_kinds:",
        '  - {legacy: "psr | psr_kl_all", object_kind: relu_steer, source: "synthetic"}',
        '  - {legacy: "lora_r1_all", object_kind: lora, fit_rank: 4, source: "alias"}',
        '  - {legacy: "vector_all | vector", object_kind: "vector_all | vector", source: "s"}',
        '  - {legacy: "bidpo", object_kind: "OUT OF SCOPE (x)", source: "s"}',
        "paper_labels:",
        '  - {legacy: "const_lora | synth-corpus-full__lora_r4_ce", paper_label: constitution_lora, object_role: constitution, fit_corpus_id: synth-corpus-full, source: "s"}',
        "campaigns:",
        '  - {legacy: "full22 | sdfbase_full22", campaign: sdfbase_full22, source: "s"}',
        "fit_target_overrides:",
        '  - {legacy: "Z9", fit_target_kind: system_instruction_distillation, citation: "synthetic"}',
    ]))
    _write(sd, "si_texts.yaml", "si_texts:\n  - {si_id: synthetic_si, si_text_sha256: %s}\n" % synth.SI_SHA)
    _write(sd, "models.yaml", "models:\n  - {model_id: %s, hf_repo: synthetic/base-4b, hf_revision: %s}\n" % (synth.FIT_MODEL_ID, "b" * 40))
    s = FS.load_seeds(sd)
    assert s.object_kind_for("psr_kl_all")["object_kind"] == "relu_steer" and s.object_kind_for("vector")["object_kind"] == "vector"
    assert s.object_kind_for("lora_r1_all")["fit_rank"] == 4 and s.object_kind_for("bidpo") is None
    assert s.paper_label_for("synth-corpus-full__lora_r4_ce")["object_role"] == "constitution" and s.corpus_ids == {"synth-corpus-full": "synth-corpus-full"}
    assert s.campaign_by_spelling["full22"] == "sdfbase_full22" and s.fit_target_overrides["Z9"]["fit_target_kind"] == "system_instruction_distillation"
    assert s.si_sha_by_id == {"synthetic_si": synth.SI_SHA} and s.model_id_by_hf_repo == {"synthetic/base-4b": synth.FIT_MODEL_ID}


# --------------------------------------------------------------------------------------------
# the paper label is keyed on the campaign STEM, never on the fit-variant prefix (the arm -> `other` gap of parity FINAL-1..3)
# --------------------------------------------------------------------------------------------


def _staged_corpus_fit(tmp_path, staged_id: str, cell_id: str, **tbm_over):
    """A staged full22 object directory `<staged_id>/` whose train_budget_meta.json records `cell_id` (the stem, as the trainer writes it)."""
    d = str(tmp_path / "bank" / staged_id)
    sha = _sha(_blob(d, staged_id.encode()))
    _write(d, "train_budget_meta.json", corpus_tbm(sha, cell_id=cell_id, **tbm_over))
    return FS.read_bundle(d, read_blob=False, bank="full22_objects", campaign_hint="sdfbase_full22")


def test_paper_label_is_spelled_from_the_stem_whatever_the_fit_variant_prefix(tmp_path):
    """The staged ids of one campaign cell differ only in the fit-variant prefix (`base__`, `itdoc__`, `itdoc64__`, `base64s<s>__`,
    `halfdoseit__`, `d0p5_eb64s1__`): every one is the SAME paper arm. The label comes from the seeded `paper_labels` entry of the stem; the
    prefix and the stem are parked on the object; the seeded entry's own spelling need not be prefixed."""
    for prefix in ("base", "itdoc", "itdoc64", "base64s2", "halfdoseit", "d0p5_eb64s1", "derived_itdoc64"):
        b = _staged_corpus_fit(tmp_path, f"{prefix}__synth-corpus-full__lora_r4_ce", "synth-corpus-full__lora_r4_ce")
        p = FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False)
        assert p["paper_label"] == "constitution_lora" and p["object_role"] == "constitution", prefix
        assert p["campaign_cell_id_legacy"] == f"{prefix}__synth-corpus-full__lora_r4_ce"
        assert p["extra"]["fit_variant_prefix_legacy"] == prefix and p["extra"]["campaign_stem_legacy"] == "synth-corpus-full__lora_r4_ce"
        assert "paper_labels[synth-corpus-full__lora_r4_ce]" in p["field_confidence_json"]["paper_label"]["source"]
    # the trainer sometimes records the STAGED spelling as the cell id too: the stem is still read
    b = _staged_corpus_fit(tmp_path, "itdoc64__synth-corpus-full__lora_r4_ce", "itdoc64__synth-corpus-full__lora_r4_ce")
    p = FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False)
    assert p["paper_label"] == "constitution_lora" and p["extra"]["fit_variant_prefix_legacy"] == "itdoc64"
    assert "campaign_stem_legacy" in p["extra"] and p["extra"]["campaign_stem_legacy"] == "synth-corpus-full__lora_r4_ce"
    # a bare stem parks no prefix (and no stem: it IS the campaign cell id)
    b = _staged_corpus_fit(tmp_path, "synth-corpus-full__lora_r4_ce", "synth-corpus-full__lora_r4_ce")
    p = FS.build_object_payload(b, seeds(), NOW, blob_reads_enabled=False)
    assert p["paper_label"] == "constitution_lora" and "fit_variant_prefix_legacy" not in p["extra"] and "campaign_stem_legacy" not in p["extra"]


def test_paper_arm_table_spells_a_real_stem_when_the_seeds_do_not(tmp_path):
    """The harvester's objects adapter read NO seeds for two days (its registered factory passed none), so every fitted object was `other`.
    The paper's own arm table (declared_common.ARMS) is the second spelling: a real campaign stem resolves even against EMPTY label
    seeds — and a stem the paper spells for ANOTHER object kind is parked, never borrowed."""
    s = seeds()
    s.paper_label_by_spelling = {}
    for prefix, stem, kind, label, role in (("base", "constv2ctl2-full__lora_r4_ce", "lora", "control_lora", "control"),
                                            ("itdoc64", "constv2-mix-full__lora_r4_ce", "lora", "constitution_lora", "constitution"),
                                            ("d0p25", "constv2ctl2-full__lora_r4_ce", "lora", "control_lora", "control")):
        b = _staged_corpus_fit(tmp_path, f"{prefix}__{stem}", stem, si_id=stem.split("__")[0])
        p = FS.build_object_payload(b, s, NOW, blob_reads_enabled=False)
        assert (p["object_kind"], p["paper_label"], p["object_role"]) == (kind, label, role), (prefix, stem)
        assert p["extra"]["fit_variant_prefix_legacy"] == prefix and p["extra"]["campaign_stem_legacy"] == stem
        assert p["field_confidence_json"]["paper_label"]["source"].startswith(f"declared_common.ARMS[{stem}]")
    # the direct helper: a stem the paper spells for another kind is parked, and the caller keeps `other`
    extra = {}
    got, src, got_stem, got_prefix = FS.paper_label_entry_for("lora", ("base__constv2-mix-full__psr_ce_all",), s, extra)
    assert got is None and got_stem == "constv2-mix-full__psr_ce_all" and got_prefix == "base"
    assert extra["paper_label_kind_disagrees"]["paper_object_kind"] == "relu_steer" and extra["paper_label_kind_disagrees"]["paper_label_of_stem"] == "constitution_relu"
    # a stem neither table spells stays `other` (the 10M-token constitution corpus is no paper arm), the role still comes from the
    # corpus, and the confidence names BOTH tables and the stem
    b = _staged_corpus_fit(tmp_path, "base__constv2-mix-10m__lora_r4_ce", "constv2-mix-10m__lora_r4_ce", si_id="constv2-mix-10m")
    p = FS.build_object_payload(b, s, NOW, blob_reads_enabled=False)
    assert p["paper_label"] == "other" and p["object_role"] == "constitution"
    reason = p["field_confidence_json"]["paper_label"]["source"]
    assert "declared_common.ARMS" in reason and "paper_labels" in reason and "'constv2-mix-10m__lora_r4_ce'" in reason
    assert p["extra"]["fit_variant_prefix_legacy"] == "base" and p["extra"]["campaign_stem_legacy"] == "constv2-mix-10m__lora_r4_ce"
    # a synthetic corpus no table knows and no role rule covers is REFUSED by name, never defaulted (unchanged behaviour)
    with pytest.raises(FS.SidecarRefusal) as ei:
        FS.build_object_payload(_staged_corpus_fit(tmp_path, "base__synth-corpus-full__lora_r4_ce", "synth-corpus-full__lora_r4_ce"), s, NOW,
                                blob_reads_enabled=False)
    assert "object_role undecidable" in str(ei.value) and "declared_common.ARMS" in str(ei.value)


def test_fit_target_override_outranks_the_stem_rule(tmp_path):
    """A candidate-instruction distillation (`candsi__A10__lora_r4`) is ruled by fit_target_overrides: its stem `A10__lora_r4` is no
    paper arm and the override's `other` stands even when a seeded label would match the tail."""
    s = seeds()
    s.si_sha_by_id["A10"] = "a" * 64
    s.si_kind_by_id["A10"] = "safety_instruction"
    s.fit_target_overrides["A10"] = {"fit_target_kind": "system_instruction_distillation", "fit_teacher_si_id": "A10", "object_role": "safety",
                                     "paper_label": "other", "citation": "synthetic D37"}
    s.paper_label_by_spelling["A10__lora_r4"] = {"paper_label": "safety_lora", "object_role": "safety"}
    d = str(tmp_path / "bank" / "candsi__A10__lora_r4")
    sha = _sha(_blob(d, b"candsi"))
    _write(d, "train_budget_meta.json", si_tbm(sha, arm="lora_r4", cell_id="candsi__A10__lora_r4", si_id="A10", lora_rank=4))
    b = FS.read_bundle(d, read_blob=False, bank="full22_objects", campaign_hint="sdfbase_full22")
    p = FS.build_object_payload(b, s, NOW, blob_reads_enabled=False)
    assert p["paper_label"] == "other" and p["object_role"] == "safety" and p["fit_target_kind"] == "system_instruction_distillation"
    assert p["extra"]["fit_variant_prefix_legacy"] == "candsi" and p["extra"]["campaign_stem_legacy"] == "A10__lora_r4"


def test_scope_verdict_classifies_a_bundle_the_way_the_builder_would_without_building_it(tmp_path):
    """Round 2 of the review of registry-hardening-2b (2026-10-07): `scope_verdict` runs the two steps of `build_object_payload` that can
    rule a bundle out of scope -- the synthetic-control check, then the kind -- and returns their outcome: in scope with the kind, parked
    with the ruling the builder raises (same directory, spelling and table row), or undecided with the plain refusal the builder raises
    (same text). The objects bank runs it on every twin before it parks a group."""
    s = seeds()
    # in scope: a fitted relu object (its spelling starts with psr)
    d = str(tmp_path / "bank" / "objects_a" / "synthetic_si__psr_kl_all")
    sha = _sha(_blob(d, b"relu weights"))
    _write(d, "train_budget_meta.json", si_tbm(sha))
    v = FS.scope_verdict(FS.read_bundle(d, read_blob=False, bank="stage_objects"), s)
    assert (v.state, v.object_kind, v.ruling, v.refusal) == (FS.SCOPE_IN_SCOPE, "relu_steer", None, None)
    assert FS.build_object_payload(FS.read_bundle(d, read_blob=False, bank="stage_objects"), s, NOW, blob_reads_enabled=False)["object_kind"] == "relu_steer"
    # parked: provenance declares lofit -- the verdict carries the ruling the builder raises
    d2 = str(tmp_path / "bank" / "objects_a" / "synthetic_si__lofit")
    sha2 = _sha(_blob(d2, b"lofit weights"))
    _write(d2, "object.pt.provenance.json", {"kind": "lofit", "layers": [15], "object_sha256": sha2})
    v2 = FS.scope_verdict(FS.read_bundle(d2, read_blob=False, bank="stage_objects"), s)
    assert v2.state == FS.SCOPE_PARKED and isinstance(v2.ruling, FS.OutOfScopeObject) and (v2.object_kind, v2.refusal) == (None, None)
    with pytest.raises(FS.OutOfScopeObject) as ei2:
        FS.build_object_payload(FS.read_bundle(d2, read_blob=False, bank="stage_objects"), s, NOW, blob_reads_enabled=False)
    assert (v2.ruling.where, v2.ruling.kind_legacy, v2.ruling.entry, v2.ruling.why) == (d2, "lofit", ei2.value.entry, ei2.value.why)
    # parked: a synthetic control under a spelling of its own the table names (the first of the builder's two steps)
    d3 = str(tmp_path / "bank" / "_controls" / "synthetic_si__rand_recipeB_s31")
    sha3 = _sha(_blob(d3, b"gaussian direction"))
    _write(d3, "train_budget_meta.json", {"arm": "rand_recipeB_s31", "cell_id": "synthetic_si__rand_recipeB_s31", "layer": 15, "model_id": "synthetic/instruct-4b",
                                          "si_id": "synthetic_si", "positions": "all", "synthetic_control": True, "control_note": "NOT FITTED", "object_sha256": sha3})
    v3 = FS.scope_verdict(FS.read_bundle(d3, read_blob=False, bank="stage_objects"), s)
    assert v3.state == FS.SCOPE_PARKED and v3.ruling.kind_legacy == "rand_recipeB_s31" and "synthetic_control=true" in v3.ruling.why
    # undecided: a declared kind nobody ruled under a ruled-out name -- the plain refusal, the builder's own text
    d4 = str(tmp_path / "bank" / "objects_a" / "synthetic_si__lofit_rewritten")
    sha4 = _sha(_blob(d4, b"a new method's weights"))
    _write(d4, "object.pt.provenance.json", {"kind": "new_unruled_method", "layers": [15], "object_sha256": sha4})
    _write(d4, "train_budget_meta.json", si_tbm(sha4, arm="lofit", cell_id="synthetic_si__lofit"))
    v4 = FS.scope_verdict(FS.read_bundle(d4, read_blob=False, bank="stage_objects"), s)
    assert v4.state == FS.SCOPE_UNDECIDED and isinstance(v4.refusal, FS.SidecarRefusal) and not isinstance(v4.refusal, FS.OutOfScopeObject)
    with pytest.raises(FS.SidecarRefusal) as ei4:
        FS.build_object_payload(FS.read_bundle(d4, read_blob=False, bank="stage_objects"), s, NOW, blob_reads_enabled=False)
    assert v4.refusal.why == ei4.value.why and "NOT parked" in v4.refusal.why and (v4.ruling, v4.object_kind) == (None, None)
