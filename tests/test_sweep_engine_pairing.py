"""The unified sweep's matched-pairing contract (engine + chat template +
serving path), the per-object training-mirror resolution, eager baseline-replica
enumeration, and the conditions table.

The load-bearing invariant: two rows are contrasted only if produced on the same
serving engine, the same chat template (each object served on the template it
was TRAINED under), and the same serving path. Every row stamps all three (plus
the object sha + trained template as provenance); the aggregator refuses any
comparison differing on any axis. These tests pin the refusal on each axis, the
unrecorded-mirror refusal + override, and the baseline-product replication.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sisteer.capeval.engine_identity import EngineIdentityError
from sisteer.eval import sweep

# a canonical template hash pair (NousResearch clean vs meta-llama injecting)
NR = "b48c47f644389271"
ML = "e10ca381b1ccc5cf"
#: the stage-3 wave's real pinned identity (verified across all 17 Class A/C
#: sidecars on disk, 2026-08-24: a set-of-one on all three fields)
NR_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
NR_REV = "d10aef7999a2b5ba950ab3974312feeedbfe0b77"
#: machine-local tree holding the ONLY copy of the stage-3 wave objects; the
#: real-artifact tests below run against it in place (read-only) and SKIP on
#: machines that don't hold it.
WAVE = Path.home() / "si-steering-wave-objects"
#: each real-artifact test skips unless the content it reads is there, not merely the tree (the repository's own pull loop
#: creates the tree, and an empty one must skip, never fail)
_WAVE_A_C_METAS = sorted(WAVE.glob("*/objects_*/*/train_budget_meta.json"))
_WAVE_ANTIDOTE = WAVE / "classB" / "objects_b" / "antidote" / "object.safetensors"
_WAVE_CAA = WAVE / "classB" / "objects_b" / "caa_static_r" / "object.pt"


def _stamp(target="hf", attacker="none", path="hf", served=NR, trained=None,
           sha=None):
    return sweep.engine_stamp(
        target_engine=target, attacker_engine=attacker, serving_path=path,
        served_template_hash=served, trained_template_hash=trained,
        object_sha=sha)


# -- stamp validation -------------------------------------------------------- #

def test_engine_stamp_rejects_unknown_engines_and_paths():
    with pytest.raises(ValueError):
        sweep.engine_stamp(target_engine="tensorrt")
    with pytest.raises(ValueError):
        sweep.engine_stamp(target_engine="hf", attacker_engine="sglang")
    with pytest.raises(ValueError):
        sweep.engine_stamp(target_engine="hf", serving_path="triton")


def test_endpoint_and_object_sha_are_not_pairing_components():
    a = _stamp(attacker="vllm", sha="aaaa"); a["attacker_endpoint"] = "http://x:8000"
    b = _stamp(attacker="vllm", sha="bbbb"); b["attacker_endpoint"] = "http://y:9999"
    # different endpoints + object shas, same engine/template/path -> pairable
    sweep.assert_pairable(a, b, what="same triple, diff endpoint/sha")


# -- the three pairing axes -------------------------------------------------- #

def test_same_triple_pairs():
    m = sweep.assert_pairable(_stamp(), _stamp(), what="pair vs baseline")
    assert m is True


def test_cross_target_engine_refuses():
    with pytest.raises(EngineIdentityError) as ei:
        sweep.assert_pairable(_stamp(target="hf"), _stamp(target="vllm"),
                              what="bon: psr vs none")
    assert "target_engine" in ei.value.differing


def test_cross_attacker_engine_refuses():
    with pytest.raises(EngineIdentityError) as ei:
        sweep.assert_pairable(_stamp(attacker="vllm"), _stamp(attacker="hf"),
                              what="pair: psr vs none")
    assert "attacker_engine" in ei.value.differing


def test_cross_template_refuses():
    # a meta-llama-trained object vs a NousResearch baseline replica -> refuse
    with pytest.raises(EngineIdentityError) as ei:
        sweep.assert_pairable(_stamp(served=ML), _stamp(served=NR),
                              what="bon: ml-object vs nr-baseline")
    assert ei.value.axis.startswith("chat template")


def test_cross_serving_path_refuses():
    with pytest.raises(EngineIdentityError) as ei:
        sweep.assert_pairable(_stamp(path="hf"), _stamp(path="lens"),
                              what="bon: hf vs lens")
    assert ei.value.axis.startswith("serving path")


def test_missing_template_refuses():
    # a row with no served template cannot be paired
    bare = sweep.engine_stamp(target_engine="hf", attacker_engine="none")
    with pytest.raises(EngineIdentityError):
        sweep.assert_pairable(_stamp(), bare, what="x")


def test_served_matches_trained_invariant():
    ok = _stamp(served=NR, trained=NR)      # compiled: served == trained
    sweep.assert_served_matches_trained(ok, what="psr row")
    bad = _stamp(served=NR, trained=ML)     # contaminated: served != trained
    with pytest.raises(EngineIdentityError):
        sweep.assert_served_matches_trained(bad, what="psr row")
    baseline = _stamp(served=NR, trained=None)  # baseline replica: exempt
    sweep.assert_served_matches_trained(baseline, what="baseline")


# -- per-object training-mirror resolution ----------------------------------- #

def _obj_with_fit(tmp_path, name, *, model, rev, template, writer="tune_arm.py",
                  bytes_=b"stage3-object"):
    """A native fit record. ``writer`` defaults to the repo's REAL producer
    of this spelling (tune_arm.FIT_PROVENANCE_WRITER): the record carries no
    budget schema, so its writer is the positive marker of an SI-pair fit
    (sweep.SI_PAIR_FIT_WRITERS) — an invented writer establishes no fit kind
    and resolve_training_mirror refuses it (three outcomes, never two)."""
    import json
    d = tmp_path / name; d.mkdir()
    (d / "object.pt").write_bytes(bytes_)
    (d / "fit_provenance.json").write_text(json.dumps({
        "resolved_model_id": model, "resolved_revision": rev,
        "chat_template_hash": template, "writer": writer}))
    return d / "object.pt"


def test_resolve_mirror_accepts_stage3_fit_provenance(tmp_path):
    # POSITIVE allowlist: a fit-time record written by the tokenizer-loading code
    op = _obj_with_fit(tmp_path, "safety_refusal__psr_kl_all",
                       model="NousResearch/Meta-Llama-3.1-8B-Instruct",
                       rev="d10aef79", template=NR)
    prov = sweep.resolve_training_mirror("safety_refusal__psr_kl_all", op)
    assert prov.model_id.startswith("NousResearch")
    assert prov.trained_template_hash == NR
    assert prov.source.startswith("fit_provenance")
    assert prov.status == "verified"


def test_resolve_mirror_incomplete_train_budget_meta_refuses(tmp_path):
    # train_budget_meta.json is now an ACCEPTED SPELLING of the fit-time record
    # (see test_resolve_mirror_accepts_train_budget_meta), but an incomplete
    # one — a bare recorded model NAME with no model_id/model_revision/
    # template_hash — is still a negative inference and still refuses.
    d = tmp_path / "safety_refusal__psr_kl_all"; d.mkdir()
    (d / "object.pt").write_bytes(b"obj")
    (d / "train_budget_meta.json").write_text(json.dumps(
        {"arm": "psr_kl_all", "model": "NousResearch/Meta-Llama-3.1-8B-Instruct"}))
    with pytest.raises(sweep.UnrecordedMirrorError, match="fit-time"):
        sweep.resolve_training_mirror("safety_refusal__psr_kl_all", d / "object.pt")


def test_resolve_mirror_no_provenance_refuses(tmp_path):
    d = tmp_path / "c"; d.mkdir(); (d / "object.pt").write_bytes(b"obj")
    with pytest.raises(sweep.UnrecordedMirrorError):
        sweep.resolve_training_mirror("c", d / "object.pt")


def test_resolve_mirror_override_nousresearch(tmp_path):
    # audited transition escape hatch: model + fit-time evidence (not a sidecar)
    d = tmp_path / "c"; d.mkdir(); (d / "object.pt").write_bytes(b"stage3-object")
    prov = sweep.resolve_training_mirror(
        "c", d / "object.pt",
        override="NousResearch/Meta-Llama-3.1-8B-Instruct@d10aef79|"
                 "cluster B fit sbatch config names NousResearch")
    assert prov.model_id.startswith("NousResearch")
    assert prov.trained_template_hash == NR
    assert prov.source == "override"
    # the audited hatch is PROVENANCE-UNVERIFIED, and says so — every row
    # stamped from it carries the distinction (test_partial_and_override_...)
    assert prov.status == "override"


def test_resolve_mirror_override_needs_evidence(tmp_path):
    d = tmp_path / "c"; d.mkdir(); (d / "object.pt").write_bytes(b"x")
    with pytest.raises(ValueError, match="evidence"):
        sweep.resolve_training_mirror("c", d / "object.pt",
                                      override="NousResearch/Meta-Llama-3.1-8B-Instruct")


def test_resolve_mirror_override_rejects_backfill_evidence(tmp_path):
    d = tmp_path / "c"; d.mkdir(); (d / "object.pt").write_bytes(b"x")
    with pytest.raises(ValueError, match="backfill"):
        sweep.resolve_training_mirror(
            "c", d / "object.pt",
            override="NousResearch/Meta-Llama-3.1-8B-Instruct@d10aef79|"
                     "backfilled sidecar says NousResearch")


# -- the training lane's sidecar spelling (train_budget_meta.json) ----------- #
# Two lanes wrote the SAME fit-time facts under two schemas: the native
# fit_provenance.json and the training lane's train_budget_meta.json. The
# reader accepts EITHER spelling, normalises internally (no derived second
# file on disk), and records WHICH schema it read — while refusing exactly as
# strictly as before: no sidecar, any missing/null required field, and a
# template-hash mismatch all still refuse.


def _train_budget_meta(model=NR_MODEL, rev=NR_REV, template=NR,
                       positions="all", **extra):
    """A training-lane sidecar shaped like the REAL ones on disk (the stage-3
    wave's train_two_si_objects._save_object emission)."""
    rec = {"arm": "vector", "cell_id": "safety_refusal__vector",
           "si_id": "safety_refusal", "max_len": 2048,
           "completion_tokens": 512, "batch_size": 4, "effective_batch": 4,
           "resolved_batch_size": 4, "site": "resid_post",
           "positions": positions, "positions_at_fit": positions,
           "parallel": {"arm_world_size": 1, "job_gpus": 4, "mode": "none"},
           "model_id": model, "model_revision": rev, "template_hash": template}
    rec.update(extra)
    return rec


def _obj_with_train_budget_meta(tmp_path, name="safety_refusal__vector",
                                bytes_=b"stage3-object-tbm", **meta_kw):
    d = tmp_path / name; d.mkdir()
    (d / "object.pt").write_bytes(bytes_)
    (d / "train_budget_meta.json").write_text(
        json.dumps(_train_budget_meta(**meta_kw)))
    return d / "object.pt"


def test_resolve_mirror_accepts_train_budget_meta(tmp_path):
    # the training lane's spelling of a COMPLETE fit-time record is accepted,
    # normalised, and the source records which schema was read
    op = _obj_with_train_budget_meta(tmp_path)
    prov = sweep.resolve_training_mirror("safety_refusal__vector", op)
    assert prov.model_id == NR_MODEL
    assert prov.revision == NR_REV
    assert prov.trained_template_hash == NR
    assert prov.source == "train_budget_meta"   # which schema it read
    assert prov.status == "verified"
    assert prov.positions_at_fit == "all"       # the recorded fit phase rides along


def test_resolve_mirror_train_budget_meta_class_c_positions(tmp_path):
    op = _obj_with_train_budget_meta(tmp_path, positions="response")
    prov = sweep.resolve_training_mirror("safety_refusal__vector", op)
    assert prov.positions_at_fit == "response"


@pytest.mark.parametrize("missing", sweep.TRAIN_BUDGET_REQUIRED_KEYS)
def test_resolve_mirror_train_budget_meta_missing_field_refuses(tmp_path, missing):
    # each individually-missing required field refuses (default-deny)
    d = tmp_path / "c"; d.mkdir()
    (d / "object.pt").write_bytes(b"obj")
    rec = _train_budget_meta()
    del rec[missing]
    (d / "train_budget_meta.json").write_text(json.dumps(rec))
    with pytest.raises(sweep.UnrecordedMirrorError, match="fit-time"):
        sweep.resolve_training_mirror("c", d / "object.pt")


@pytest.mark.parametrize("nulled", sweep.TRAIN_BUDGET_REQUIRED_KEYS)
def test_resolve_mirror_train_budget_meta_null_field_refuses(tmp_path, nulled):
    # a null value (e.g. the writer's stated-absence null model_revision for a
    # local-path load) is as unverifiable as a missing key: refuse, exactly as
    # strictly as the native spelling does
    d = tmp_path / "c"; d.mkdir()
    (d / "object.pt").write_bytes(b"obj")
    rec = _train_budget_meta()
    rec[nulled] = None
    (d / "train_budget_meta.json").write_text(json.dumps(rec))
    with pytest.raises(sweep.UnrecordedMirrorError):
        sweep.resolve_training_mirror("c", d / "object.pt")


@pytest.mark.parametrize("missing", sweep.STAGE3_FIT_REQUIRED_KEYS)
def test_resolve_mirror_native_missing_field_refuses(tmp_path, missing):
    # the native spelling refuses each individually-missing field too
    d = tmp_path / "c"; d.mkdir()
    (d / "object.pt").write_bytes(b"obj")
    rec = {"resolved_model_id": NR_MODEL, "resolved_revision": NR_REV,
           "chat_template_hash": NR, "writer": "stage3_fit"}
    del rec[missing]
    (d / "fit_provenance.json").write_text(json.dumps(rec))
    with pytest.raises(sweep.UnrecordedMirrorError):
        sweep.resolve_training_mirror("c", d / "object.pt")


def test_template_hash_mismatch_refuses_naming_both_hashes(tmp_path):
    # THE correctness check the whole mechanism exists for: an object whose
    # RECORDED fit-time hash (the real b48c47f644389271) does not match the
    # template the sweep actually serves refuses, and the refusal names BOTH
    # hashes — proving the comparison is live, not vacuous.
    from sisteer.capeval.engine_identity import EngineIdentityError
    op = _obj_with_train_budget_meta(tmp_path, template=NR)
    prov = sweep.resolve_training_mirror("safety_refusal__vector", op)
    assert prov.trained_template_hash == NR
    wrong_served = "0123456789abcdef"          # deliberately wrong
    st = sweep.engine_stamp(
        target_engine="hf", served_template_hash=wrong_served,
        trained_template_hash=prov.trained_template_hash,
        object_sha=prov.object_sha, provenance_status=prov.status,
        provenance_source=prov.source)
    with pytest.raises(EngineIdentityError) as ei:
        sweep.assert_served_matches_trained(st, what="mismatch test")
    assert NR in str(ei.value)
    assert wrong_served in str(ei.value)


def test_full_record_wins_over_object_json(tmp_path):
    # precedence: a COMPLETE fit-time record is never degraded to partial by
    # the object's own save sidecar sitting beside it
    op = _obj_with_train_budget_meta(tmp_path)
    op.with_suffix(".json").write_text(json.dumps({"model_name": NR_MODEL}))
    prov = sweep.resolve_training_mirror("safety_refusal__vector", op)
    assert prov.status == "verified"
    assert prov.source == "train_budget_meta"


# -- partial provenance: the object's own save sidecar (model name, no hash) - #


def _obj_with_save_sidecar(tmp_path, name="antidote", model_name=NR_MODEL,
                           bytes_=b"class-b-object"):
    d = tmp_path / name; d.mkdir()
    (d / "object.safetensors").write_bytes(bytes_)
    # the SteeringVector save sidecar: path.with_suffix('.json'), stamped from
    # the LIVE model's _name_or_path at fit time — a real fit artifact that
    # names the MODEL but records no revision and no template hash
    (d / "object.json").write_text(json.dumps(
        {"method": "antidote", "model_name": model_name, "layers": [15]}))
    return d / "object.safetensors"


def test_resolve_mirror_object_json_is_partial(tmp_path):
    # a real fit-time artifact naming the model -> accepted as PARTIAL: model
    # identity is real, the template hash is a recorded ABSENCE (never
    # inferred from the model name), and the status says so
    op = _obj_with_save_sidecar(tmp_path)
    prov = sweep.resolve_training_mirror("antidote", op)
    assert prov.model_id == NR_MODEL
    assert prov.status == "partial"
    assert prov.source == "object_json"
    assert prov.revision is None
    assert prov.trained_template_hash is None   # not recorded — NOT looked up
    assert prov.positions_at_fit is None


def test_object_json_empty_model_name_refuses(tmp_path):
    # the two-SI vector containers save model_name="" — an empty name is no
    # identity, so the partial path must NOT fire on it
    d = tmp_path / "c"; d.mkdir()
    (d / "object.safetensors").write_bytes(b"obj")
    (d / "object.json").write_text(json.dumps({"model_name": ""}))
    with pytest.raises(sweep.UnrecordedMirrorError):
        sweep.resolve_training_mirror("c", d / "object.safetensors")


def test_partial_and_override_rows_carry_their_status_and_hashless_refuses():
    # three states, not two. A compiled row with NO trained hash passes the
    # served==trained check ONLY when its stamp says why (partial/override);
    # an unexplained hashless compiled row refuses — otherwise it would
    # silently skip the check, indistinguishable from a baseline.
    from sisteer.capeval.engine_identity import EngineIdentityError
    partial = sweep.engine_stamp(target_engine="hf", served_template_hash=NR,
                                 trained_template_hash=None, object_sha="abcd",
                                 provenance_status="partial",
                                 provenance_source="object_json")
    sweep.assert_served_matches_trained(partial, what="partial row")
    override = sweep.engine_stamp(target_engine="hf", served_template_hash=NR,
                                  trained_template_hash=None, object_sha="abcd",
                                  provenance_status="override",
                                  provenance_source="override")
    sweep.assert_served_matches_trained(override, what="override row")
    unexplained = sweep.engine_stamp(target_engine="hf", served_template_hash=NR,
                                     trained_template_hash=None,
                                     object_sha="abcd")
    with pytest.raises(EngineIdentityError, match="provenance"):
        sweep.assert_served_matches_trained(unexplained, what="unexplained row")
    # a baseline (no object sha) stays exempt
    baseline = sweep.engine_stamp(target_engine="hf", served_template_hash=NR)
    sweep.assert_served_matches_trained(baseline, what="baseline row")


def test_engine_stamp_rejects_unknown_provenance_status():
    with pytest.raises(ValueError, match="provenance_status"):
        sweep.engine_stamp(target_engine="hf", provenance_status="trusted")


def test_content_only_meta_refuses(tmp_path):
    # the caa_static_r shape: a meta.json with content provenance (prompt-list
    # shas, counts, seed) but NO model identity anywhere -> refuse; inclusion
    # takes the explicit per-object override (stamped provenance-unverified),
    # never an inference from its siblings
    d = tmp_path / "caa_static_r"; d.mkdir()
    (d / "object.pt").write_bytes(b"caa-object")
    (d / "meta.json").write_text(json.dumps(
        {"arm": "caa_static_r", "cell_id": "caa_static_r", "class": "B",
         "content_meta": {"n_benign": 14000, "n_harmful": 2000, "seed": 0},
         "layer": 15, "n_params": 40960, "si_id": None}))
    with pytest.raises(sweep.UnrecordedMirrorError):
        sweep.resolve_training_mirror("caa_static_r", d / "object.pt")
    prov = sweep.resolve_training_mirror(
        "caa_static_r", d / "object.pt",
        override=f"{NR_MODEL}@{NR_REV}|fit_class_b.sbatch invocation config "
                 "names the model (fit log records no identity)")
    assert prov.status == "override"


# -- the REAL stage-3 wave objects, read in place (machine-local; skip absent) #


@pytest.mark.skipif(not _WAVE_A_C_METAS,
                    reason="machine-local stage-3 wave objects (Class A/C train_budget_meta.json) absent")
def test_real_wave_class_a_c_objects_accepted_end_to_end():
    """All 17 real Class A/C objects resolve through the training-lane schema
    for the RIGHT reason: one identity (NousResearch @ d10aef79, template
    b48c47f644389271) across all 17, the served==trained check RUNS (verified
    status, real hash) and passes against the template the sweep serves, and
    the recorded fit phase matches the objects_a/objects_c split."""
    metas = _WAVE_A_C_METAS
    assert len(metas) == 17
    seen = set()
    for m in metas:
        d = m.parent
        obj = next(f for f in sorted(d.iterdir())
                   if f.name.startswith("object.")
                   and not f.name.endswith(".json"))
        prov = sweep.resolve_training_mirror(d.name, obj)
        assert prov.status == "verified"
        assert prov.source == "train_budget_meta"
        assert prov.model_id == NR_MODEL
        assert prov.revision == NR_REV
        assert prov.trained_template_hash == NR
        expected_phase = ("response" if d.parent.name == "objects_c" else "all")
        assert prov.positions_at_fit == expected_phase
        st = sweep.engine_stamp(
            target_engine="hf", served_template_hash=NR,
            trained_template_hash=prov.trained_template_hash,
            object_sha=prov.object_sha, provenance_status=prov.status,
            provenance_source=prov.source)
        sweep.assert_served_matches_trained(st, what=d.name)
        seen.add(prov.object_sha)
    # six cell_ids exist in BOTH an all-position and a response-position fit,
    # so the sha (never the cell id) is the provenance key: all 17 distinct
    assert len(seen) == 17


@pytest.mark.skipif(not _WAVE_ANTIDOTE.is_file(),
                    reason="machine-local stage-3 wave object (classB antidote) absent")
def test_real_wave_antidote_is_partial():
    # antidote's own save sidecar (object.json) names the LIVE model -> partial
    op = _WAVE_ANTIDOTE
    prov = sweep.resolve_training_mirror("antidote", op)
    assert prov.status == "partial"
    assert prov.source == "object_json"
    assert prov.model_id == NR_MODEL
    assert prov.revision is None
    assert prov.trained_template_hash is None


@pytest.mark.skipif(not _WAVE_CAA.is_file(),
                    reason="machine-local stage-3 wave object (classB caa_static_r) absent")
def test_real_wave_caa_static_r_refuses_without_override():
    # caa_static_r records NO model identity in any artifact (meta.json is
    # content-only, the fit log names no model, object.pt is a bare tensor
    # blob) -> refuses by default; only the explicit per-object override
    # (stamped provenance-unverified) can include it
    op = _WAVE_CAA
    with pytest.raises(sweep.UnrecordedMirrorError):
        sweep.resolve_training_mirror("caa_static_r", op)


# -- stage-2 obsolescence: HARD refusal --------------------------------------- #

def test_stage2_refused_by_sha_denylist():
    with pytest.raises(sweep.Stage2ObsoleteError, match="STAGE-2"):
        sweep.assert_not_stage2("950886beabcdef01", cell_id="psr_kl_all")
    # a full [:16] cells-dir sha (a TrainTest stage-2 cell)
    with pytest.raises(sweep.Stage2ObsoleteError):
        sweep.assert_not_stage2("9f789d68670967db",
                                cell_id="safety_refusal__psr_kl_all")
    # the antidote (Class B) stage-2 cell
    with pytest.raises(sweep.Stage2ObsoleteError):
        sweep.assert_not_stage2("690cd67a603ac9ae", cell_id="antidote")


def test_stage2_denylist_covers_both_populations():
    # both pre-stage-3 populations are pinned: the jailbreak lane's psr pair
    # AND all 25 TrainTest two-SI cells.
    pfx = sweep.STAGE2_OBSOLETE_SHA_PREFIXES
    assert "950886be" in pfx and "48efb10c" in pfx          # jailbreak psr pair
    assert "9f789d68670967db" in pfx and "690cd67a603ac9ae" in pfx  # TrainTest cells
    assert len(pfx) >= 27  # 2 psr + 25 cells


def test_stage3_nousresearch_sha_not_denylisted():
    # a non-denylisted sha passes assert_not_stage2 (denylist-only now)
    sweep.assert_not_stage2("1234123412341234", cell_id="stage3-object")


# -- eager baseline replicas ------------------------------------------------- #

def test_single_population_degenerates_to_one_replica():
    reps = sweep.required_baseline_replicas([(NR, "hf")], ["none", "prompted"])
    assert reps == {"none": [(NR, "hf")], "prompted": [(NR, "hf")]}


def test_two_populations_replicate_over_product():
    reps = sweep.required_baseline_replicas([(NR, "hf"), (ML, "hf")], ["none"])
    assert reps == {"none": [(NR, "hf"), (ML, "hf")]}  # sorted: b48.. < e10..


# -- conditions table -------------------------------------------------------- #

def _write(tmp_path, body):
    p = tmp_path / "conditions.tsv"; p.write_text(body); return p


def test_load_conditions_full_row(tmp_path):
    p = _write(tmp_path,
        "c\tsafety_refusal__psr_kl_all\t0.25\tall\t*\t*\t*\thf\tvllm\thf\t\n")
    c = sweep.load_conditions(p)[0]
    assert c.cell == "safety_refusal__psr_kl_all" and c.coeff == "0.25"
    assert c.serving_path == "hf" and c.mirror_override == ""


def test_load_conditions_override_column(tmp_path):
    p = _write(tmp_path,
        "c\tsafety_refusal__psr_kl_all\t0.25\tall\t-\t-\tmmlu\thf\tnone\thf\t"
        "meta-llama/Llama-3.1-8B-Instruct@0e9e39f2|forensic\n")
    c = sweep.load_conditions(p)[0]
    assert c.mirror_override.startswith("meta-llama/") and "forensic" in c.mirror_override


def test_duplicate_label_refuses(tmp_path):
    p = _write(tmp_path,
        "d\tnone\t\tall\t*\t-\t*\thf\tnone\thf\t\n"
        "d\tprompted__safety_refusal\t\tall\t*\t-\t*\thf\tnone\thf\t\n")
    with pytest.raises(ValueError, match="duplicate"):
        sweep.load_conditions(p)


def test_adaptive_without_attacker_refuses(tmp_path):
    p = _write(tmp_path, "bad\tnone\t\tall\t-\tpair\t-\thf\tnone\thf\t\n")
    with pytest.raises(ValueError, match="need the shared attacker pool"):
        sweep.load_conditions(p)


# -- the aggregate comparison over replicas ---------------------------------- #

def _row(leg, member, cond, metric, *, target="hf", attacker="none",
         served=NR, path="hf", family=None):
    return sweep.MetricRow(
        leg=leg, member=member, condition=cond, metric=metric,
        stamp=_stamp(target=target, attacker=attacker, served=served, path=path),
        baseline_family=family)


def _baseline_of(leg, condition):
    if condition.startswith(("none", "prompted")):
        return None
    return "prompted" if leg == "benign" else "none"


def test_comparison_matches_replica_and_emits_delta():
    rows = [
        # baseline replicas on NR/hf (family set)
        _row("static", "bon", "none@NR", 0.60, family="none"),
        _row("adaptive", "pair", "none@NR", 0.55, attacker="vllm", family="none"),
        _row("benign", "mmlu", "prompted@NR", 0.89, family="prompted"),
        # object rows on NR/hf
        _row("static", "bon", "s_kl", 0.20),
        _row("adaptive", "pair", "s_kl", 0.30, attacker="vllm"),
        _row("benign", "mmlu", "s_kl", 0.91),
    ]
    table = sweep.build_comparison_table(rows, baseline_of=_baseline_of)
    got = {(c.leg, c.member): (round(c.delta, 2), c.baseline, c.chat_template)
           for c in table}
    assert got == {
        ("static", "bon"): (-0.40, "none", NR),
        ("adaptive", "pair"): (-0.25, "none", NR),
        ("benign", "mmlu"): (0.02, "prompted", NR),
    }


def test_comparison_refuses_when_replica_template_missing():
    # object is meta-llama-trained (served ML) but only a NousResearch baseline
    # replica exists -> loud refusal (missing replica), not a silent drop.
    rows = [
        _row("static", "bon", "none@NR", 0.60, served=NR, family="none"),
        _row("static", "bon", "s_ml", 0.20, served=ML),
    ]
    with pytest.raises(ValueError, match="baseline replica on template"):
        sweep.build_comparison_table(rows, baseline_of=_baseline_of)


def test_comparison_refuses_cross_attacker_engine():
    rows = [
        _row("adaptive", "pair", "none@NR", 0.55, attacker="hf", family="none"),
        _row("adaptive", "pair", "s_kl", 0.30, attacker="vllm"),
    ]
    with pytest.raises(EngineIdentityError):
        sweep.build_comparison_table(rows, baseline_of=_baseline_of)


def test_array_decode(tmp_path):
    p = _write(tmp_path,
        "a\tnone\t\tall\t*\t-\t*\thf\tnone\thf\t\n"
        "b\tprompted__safety_refusal\t\tall\t*\t-\t*\thf\tnone\thf\t\n")
    conds = sweep.load_conditions(p)
    assert sweep.condition_for_array_index(conds, 0).label == "a"
    with pytest.raises(IndexError):
        sweep.condition_for_array_index(conds, 2)


def test_cross_decode_mechanism_refuses():
    """The two incompatible decode-only implementations (engine_phase_gate vs
    position_mask) must never be compared — the same silent confound as
    cross-engine/template/path, refused the same way and named as its own axis."""
    a = sweep.engine_stamp(target_engine="hf", served_template_hash=NR,
                           decode_mechanism="engine_phase_gate")
    b = sweep.engine_stamp(target_engine="hf", served_template_hash=NR,
                           decode_mechanism="position_mask")
    with pytest.raises(EngineIdentityError) as ei:
        sweep.assert_pairable(a, b, what="decode-only: engine_gate vs mask")
    assert ei.value.axis.startswith("decode mechanism")


def test_all_position_rows_pair_despite_absent_decode_mechanism():
    """All-position rows carry no decode_mechanism; the 'all_position' sentinel
    keeps the empty case from refusing (backward compatibility)."""
    assert sweep.assert_pairable(_stamp(), _stamp(), what="all-position") is True


def test_engine_stamp_rejects_unknown_decode_mechanism():
    with pytest.raises(ValueError, match="decode_mechanism"):
        sweep.engine_stamp(target_engine="hf", decode_mechanism="bogus")
