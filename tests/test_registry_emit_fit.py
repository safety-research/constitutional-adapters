"""Native emission from the fit sidecar writers: ``object_from_sidecar`` reads a
``train_budget_meta.json`` (the two-SI / HPO trainers' sidecar) or ``fit_meta.json`` +
``fit_provenance.json`` (the Class B alphasteer sidecars) beside the artifact and yields an ``object`` +
``artifact`` record that validates; the writer states kind / label / role / target kind; a sidecar whose
recorded ``object_sha256`` disagrees with the artifact bytes is refused; the sidecar bytes never change;
the fit writers' own hook functions are wired and inert when emission is off.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from sisteer.registry import emit, hooks
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry import validate as VAL

REPO = Path(__file__).resolve().parents[1]


def _tbm_dir(tmp_path, si="si_constitution", **over):
    d = tmp_path / "cells_all" / f"{si}__lora_r1_all"
    d.mkdir(parents=True)
    (d / "object.pt").write_bytes(b"synthetic lora object bytes " * 64)
    sha = hashlib.sha256((d / "object.pt").read_bytes()).hexdigest()
    meta = {"arm": "lora_r1_all", "si_id": si, "positions_at_fit": "all", "response_header_len": 5, "batch_size": 8,
            "resolved_batch_size": 8, "max_len": 1024, "epochs": 3, "seed": 0, "lr": 1e-3, "lr_schedule": "cosine", "warmup_steps": 10,
            "max_grad_norm": 1.0, "length_sorted": True, "teacher_prefix_cache": True, "trainer": "train_two_si_objects.py",
            "completion_tokens": 256, "object_sha256": sha, "model_id": "synthetic/base-4b", "model_revision": "b" * 40,
            "template_hash": "b48c47f644389271", "site": "resid_post", "target_layers": [4, 5, 6], "cell_id": f"{si}__lora_r1_all"}
    meta.update(over)
    (d / "train_budget_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    (d / "COMPLETE").write_text(json.dumps({"cell_id": meta["cell_id"], "object_sha256": sha}))
    return d, sha


def test_object_from_train_budget_sidecar_validates(tmp_path):
    d, sha = _tbm_dir(tmp_path)
    obj, art = hooks.object_from_sidecar(str(d), campaign="two_si", object_kind="lora", paper_label="constitution_lora", object_role="constitution",
                                         fit_target_kind="system_instruction_distillation", fit_objective="kl", fit_thinking_render="off",
                                         fit_teacher_si_id="si_constitution", producer=synth.producer())
    assert obj["object_id"] == sha == obj["artifact_sha256"] and obj["artifact_bytes"] == 28 * 64
    assert obj["fit_positions"] == "all" and obj["fit_seed"] == 0 and obj["fit_seed_confidence"] == "recorded"
    assert obj["fit_layers_json"] == [4, 5, 6] and obj["fit_layer_count"] == 3 and obj["fit_capture_site"] == "resid_post"
    assert obj["fit_lr"] == 1e-3 and obj["fit_epochs_cap"] == 3 and obj["fit_effective_batch"] == 8 and obj["fit_completion_tokens"] == 256
    assert obj["fit_teacher_si_id"] == "si_constitution" and obj["fit_model_id"] == "synthetic/base-4b" and obj["fit_writer"] == "train_two_si_objects.py"
    assert obj["object_sidecars_json"] == ["COMPLETE", "train_budget_meta.json"] and "field_confidence_json" not in obj
    assert art["kind"] == "object_weights" and art["artifact_id"] == sha
    # the artifact record validates; the OBJECT record does not: a two-SI distillation row needs its teacher text sha and prompt
    # pool identity (vocab.FIT_TARGET_RULES["system_instruction_distillation"], the DB's pairing CHECK), which no tbm sidecar carries —
    # since 2026-10-02 the validator mirrors the CHECK and names the column, instead of letting the record die at the grading host
    rec = R.make_record("artifact", art, synth.producer(), "00000000-1111-2222-3333-444444444444", synth.t(0))
    assert isinstance(VAL.validate_record(rec, "<mem>"), VAL.Validated)
    rec = R.make_record("object", obj, synth.producer(), "00000000-1111-2222-3333-444444444444", synth.t(0))
    m = VAL.validate_record(rec, "<mem>")
    assert isinstance(m, VAL.Malformed) and m.field == "fit_teacher_si_text_sha256" and "object_fit_target_kind_pairing" in m.reason
    # an absent positions / seed / thinking render is unverified WITH a reason, never defaulted
    d2, _ = _tbm_dir(tmp_path / "b", si="si_irrelevant")
    meta = json.loads((d2 / "train_budget_meta.json").read_text())
    meta.pop("positions_at_fit"), meta.pop("seed")
    (d2 / "train_budget_meta.json").write_text(json.dumps(meta))
    obj2, _ = hooks.object_from_sidecar(str(d2), campaign="two_si", object_kind="lora", paper_label="control_lora", object_role="irrelevant",
                                        fit_target_kind="system_instruction_distillation", fit_objective="kl")
    assert obj2["fit_positions"] == "unverified" and obj2["fit_seed_confidence"] == "unverified" and obj2["fit_thinking_render"] == "unverified"
    assert set(obj2["field_confidence_json"]) == {"fit_positions", "fit_seed_confidence", "fit_thinking_render"}
    rec = R.make_record("object", obj2, synth.producer(), "00000000-1111-2222-3333-444444444444", synth.t(0))
    m2 = VAL.validate_record(rec, "<mem>")
    assert isinstance(m2, VAL.Malformed) and m2.field == "fit_teacher_si_text_sha256"
    # the corpus-document shape the live fits use (training_budget_kind document_tokens) validates end to end
    d3, _ = _tbm_dir(tmp_path / "c", si="docs_pool", training_budget_kind="document_tokens")
    obj3, art3 = hooks.object_from_sidecar(str(d3), campaign="two_si", object_kind="relu_steer", paper_label="constitution_relu", object_role="constitution",
                                           fit_target_kind="corpus_document_fit", fit_objective="ce", fit_thinking_render="off")
    assert obj3["fit_corpus_id"] == "docs_pool" and obj3["fit_corpus_sha256"] == "unverified"
    for rt, payload in (("object", obj3), ("artifact", art3)):
        v = VAL.validate_record(R.make_record(rt, payload, synth.producer(), "00000000-1111-2222-3333-444444444444", synth.t(0)), "<mem>")
        assert isinstance(v, VAL.Validated), v


def test_object_from_alphasteer_sidecars(tmp_path):
    d = tmp_path / "alphasteer"
    d.mkdir()
    (d / "object.pt").write_bytes(b"alphasteer bytes " * 10)
    sha = hashlib.sha256((d / "object.pt").read_bytes()).hexdigest()
    (d / "fit_meta.json").write_text(json.dumps({"layers": [10, 11, 12, 13], "object_sha256": sha, "cell_id": "alphasteer"}))
    (d / "fit_provenance.json").write_text(json.dumps({"resolved_model_id": "synthetic/instruct-4b", "resolved_revision": "c" * 40,
                                                       "chat_template_hash": "e10ca381b1ccc5cf", "writer": "fit_class_b.py", "positions_at_fit": "response"}))
    (d / "meta.json").write_text(json.dumps({"cell_id": "alphasteer", "class": "B", "sidecars": ["fit_meta.json", "fit_provenance.json"]}))
    obj, art = hooks.object_from_sidecar(str(d), campaign="class_b", object_kind="alphasteer", paper_label=hooks.paper_label_for("alphasteer", "safety"),
                                         object_role="safety", fit_target_kind="contrastive_direction", fit_objective="alphasteer", fit_thinking_render="off")
    assert obj["paper_label"] == "alphasteer_as_fit" and obj["fit_positions"] == "response_incl_assistant_header"
    assert obj["fit_layers_json"] == [10, 11, 12, 13] and obj["fit_model_id"] == "synthetic/instruct-4b" and obj["fit_writer"] == "fit_class_b.py"
    assert obj["fit_seed_confidence"] == "unverified"
    # a contrastive-direction row needs its contrast sets (vocab.FIT_TARGET_RULES; the DB's pairing CHECK): these sidecars carry none,
    # so the record is refused by name — the writer states them (fit_contrast_sets_json=…, as the refusal-direction fitter does)
    rec = R.make_record("object", obj, synth.producer(), "00000000-1111-2222-3333-444444444444", synth.t(0))
    m = VAL.validate_record(rec, "<mem>")
    assert isinstance(m, VAL.Malformed) and m.field == "fit_contrast_sets_json" and "object_fit_target_kind_pairing" in m.reason
    obj_ok, _ = hooks.object_from_sidecar(str(d), campaign="class_b", object_kind="alphasteer", paper_label=hooks.paper_label_for("alphasteer", "safety"),
                                          object_role="safety", fit_target_kind="contrastive_direction", fit_objective="alphasteer", fit_thinking_render="off",
                                          fit_contrast_sets_json={"harmful": "sha256:" + "d" * 64, "harmless": "sha256:" + "e" * 64})
    rec = R.make_record("object", obj_ok, synth.producer(), "00000000-1111-2222-3333-444444444444", synth.t(0))
    assert isinstance(VAL.validate_record(rec, "<mem>"), VAL.Validated)


def test_sidecar_sha_disagreement_is_refused_and_no_artifact_is_refused(tmp_path):
    d, sha = _tbm_dir(tmp_path)
    (d / "object.pt").write_bytes(b"other bytes")
    with pytest.raises(ValueError, match="refusing to describe"):
        hooks.object_from_sidecar(str(d), campaign="c", object_kind="lora", paper_label="other", object_role="control",
                                  fit_target_kind="system_instruction_distillation", fit_objective="kl")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        hooks.object_from_sidecar(str(empty), campaign="c", object_kind="lora", paper_label="other", object_role="control",
                                  fit_target_kind="system_instruction_distillation", fit_objective="kl")


def test_emit_object_from_sidecar_writes_two_records_and_leaves_the_sidecar_bytes(tmp_path, monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    d, sha = _tbm_dir(tmp_path, si="docs_pool", training_budget_kind="document_tokens")     # the corpus-document shape the live fits are
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()}
    spool = tmp_path / "spool"
    paths = hooks.emit_object_from_sidecar(str(d), campaign="two_si", object_kind="lora", paper_label="constitution_lora", object_role="constitution",
                                           fit_target_kind="corpus_document_fit", fit_objective="ce", fit_thinking_render="off",
                                           script="tests.synthetic", spool=str(spool), registry_emit=True)
    assert len(paths) == 2 and all(os.path.dirname(p).endswith(f"object__{sha[:16]}") for p in paths)
    # (the writer-side pairing refusal — hooks.refuse_unpairable_object — is scoped to derived records tonight; a two-SI distillation
    #  payload from a tbm sidecar is still written by the hook and refused by name at ingest by the validator, see the two tests above)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()}
    assert before == after
    types_ = [rec["record_type"] for p in paths for _, rec, _ in R.read_record_file(p)]
    assert types_ == ["object", "artifact"]
    # off: nothing
    monkeypatch.setenv(emit.DISABLED_ENV, "1")
    assert hooks.emit_object_from_sidecar(str(d), campaign="two_si", object_kind="lora", paper_label="constitution_lora", object_role="constitution",
                                          fit_target_kind="corpus_document_fit", fit_objective="ce", script="t", spool=str(tmp_path / "off")) == []
    assert not (tmp_path / "off").exists()


def test_writer_tables_never_guess():
    assert hooks.object_kind_for_legacy_kind("lora_r1_all") == "lora" and hooks.object_kind_for_legacy_kind("psr") == "relu_steer"
    assert hooks.object_kind_for_legacy_kind("some_new_arm") is None
    assert hooks.object_role_for_si("safety_refusal") == "safety" and hooks.object_role_for_si("unknown_si") is None
    assert hooks.paper_label_for("lora", "constitution") == "constitution_lora" and hooks.paper_label_for("relu_steer", "irrelevant") == "control_relu"
    assert hooks.paper_label_for("vector", "constitution") == "other" and hooks.paper_label_for("static_r_arditi", "safety") == "arditi"
    from sisteer.registry import vocab as V
    for kind in hooks.LEGACY_KIND_TO_OBJECT_KIND.values():
        assert kind in V.OBJECT_KINDS
    for role in hooks.SI_TO_OBJECT_ROLE.values():
        assert role in V.OBJECT_ROLES
    for kind in ("lora", "relu_steer", "vector", "static_r_arditi", "alphasteer"):
        for role in ("constitution", "control", "irrelevant", "safety", "difference"):
            assert hooks.paper_label_for(kind, role) in V.VOCABULARIES["PAPER_LABELS"]


def test_fit_writers_carry_the_hook_and_it_is_inert_off(tmp_path, monkeypatch):
    """The three writers call the hook AFTER their own writes; with emission off the hook writes nothing."""
    t2 = (REPO / "scripts" / "train_two_si_objects.py").read_text()
    assert "_registry_publish(final_dir, si_id=si_id, arm=arm)" in t2 and t2.index("COMPLETE_MARKER_NAME).write_text(") < t2.index("_registry_publish(final_dir")
    tl = (REPO / "scripts" / "tune_lora_all.py").read_text()
    assert "_registry_object(out, args.si)" in tl and tl.index('"train_budget_meta.json").write_text') < tl.index("_registry_object(out, args.si)")
    fb = (REPO / "scripts" / "fit_class_b.py").read_text()
    assert "_registry_object(cfg, arm, path.parent)" in fb and fb.index("_meta_path(cfg, arm).write_text") < fb.index("_registry_object(cfg, arm, path.parent)")
    # the hook body, exercised through the module's own function with emission off: no spool created
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    monkeypatch.setenv(emit.DISABLED_ENV, "1")
    monkeypatch.setenv(emit.SPOOL_ENV, str(tmp_path / "spool"))
    monkeypatch.setenv("REGISTRY_CAMPAIGN", "two_si")        # the campaign is the job env's statement (the emitter refuses without one when on)
    d, _ = _tbm_dir(tmp_path, si="docs_pool", training_budget_kind="document_tokens")   # the corpus-document fit the live cells are
    import importlib.util
    spec = importlib.util.spec_from_file_location("t2si_hook_probe", REPO / "scripts" / "train_two_si_objects.py")
    # importing the whole trainer pulls torch; read the hook's source and exec it in a tiny namespace instead
    src = t2[t2.index("def _registry_publish"):t2.index("\n\n\n", t2.index("def _registry_publish"))]
    ns = {"Path": Path, "os": os, "sys": sys}
    exec(src, ns)
    ns["_registry_publish"](d, si_id="si_constitution", arm="lora_r1_all")
    assert not (tmp_path / "spool").exists()
    monkeypatch.delenv(emit.DISABLED_ENV)
    ns["_registry_publish"](d, si_id="si_constitution", arm="lora_r1_all")
    recs = [rec for p in R.iter_record_files(str(tmp_path / "spool")) for _, rec, _ in R.read_record_file(p)]
    assert [r["record_type"] for r in recs] == ["object", "artifact"]
    o = recs[0]["payload"]
    assert o["object_role"] in ("constitution", "control", "irrelevant", "safety") and o["fit_target_kind"] == "corpus_document_fit"
    assert o["fit_objective"] == "ce" and o["campaign"] == "two_si" and recs[0]["producer"]["script"] == "scripts/train_two_si_objects.py publish_cell"


# ---- fit-time provenance is the FIT's, on every emission ----------------------------------------------------

def test_fit_provenance_sidecar_then_caller_then_context_and_the_dry_run_writes_nothing(tmp_path, monkeypatch):
    """The hook derived fit_code_sha / fit_job_id / fit_node_hostname from the EMITTING context — right at the first emission, wrong on
    every replay. Precedence now: the cell's fit_job_provenance.json sidecar (with its sources recorded on the row) > an explicit
    fit_provenance (a replay restating the prior record) > the context. dry_run returns every record the hook would write and touches
    neither the spool nor the switch."""
    from sisteer.registry import emit
    d, sha = _tbm_dir(tmp_path, si="docs_pool", training_budget_kind="document_tokens")
    monkeypatch.setenv("SLURM_JOB_ID", "999001")
    monkeypatch.setattr(emit, "producer_block", lambda script, code_sha=None, session=None: {"script": script, "code_sha": "c" * 40, "host": "emitter-host", "substrate": "slurm", "session": None})
    kw = dict(campaign="paper_frame", object_kind="relu_steer", paper_label="constitution_relu", object_role="constitution",
              fit_target_kind="corpus_document_fit", fit_objective="ce", fit_thinking_render="off")
    # context (a first emission: the fit job is the emitter)
    obj, art = hooks.object_from_sidecar(str(d), producer=emit.producer_block("w"), **kw)
    assert (obj["fit_code_sha"], obj["fit_job_id"], obj["fit_node_hostname"]) == ("c" * 40, "999001", "emitter-host") and "fit_provenance" not in obj["extra"]
    # the caller's statement (a replay restating the prior record): wins over the context; a None unstates
    prior = {"fit_code_sha": "a" * 40, "fit_job_id": "821364", "fit_node_hostname": "gpu-23", "fit_substrate": "slurm", "fit_producer_session": None,
             "artifact_created_utc": "2026-10-01T19:32:29.000Z", "_source": "/spool/two_si/x#object"}
    obj2, art2 = hooks.object_from_sidecar(str(d), producer=emit.producer_block("w"), fit_provenance=prior, **kw)
    assert (obj2["fit_code_sha"], obj2["fit_job_id"], obj2["fit_node_hostname"]) == ("a" * 40, "821364", "gpu-23")
    assert "fit_producer_session" not in obj2 and art2["created_utc"] == "2026-10-01T19:32:29.000Z"
    assert obj2["extra"]["fit_provenance"] == {"restated_from": "/spool/two_si/x#object"}
    # the sidecar (an authoritative statement from the fit's own records) wins over both, and its sources ride the row
    (d / hooks.FIT_PROVENANCE_SIDECAR).write_text(json.dumps({
        "fit_code_sha": "b" * 40, "fit_job_id": "821352", "fit_node_hostname": "gpu-29",
        "sources": {"result": str(d / "RESULT"), "attempt_record": "attempts.jsonl#control_relu@cluster_a@2"},
        "note": "fit provenance corrected from RESULT + attempt record; previous values were a re-emission's context",
        "written_by": "tests", "written_utc": "2026-10-02T06:00:00Z"}))
    obj3, _ = hooks.object_from_sidecar(str(d), producer=emit.producer_block("w"), fit_provenance=prior, **kw)
    assert (obj3["fit_code_sha"], obj3["fit_job_id"], obj3["fit_node_hostname"]) == ("b" * 40, "821352", "gpu-29")
    assert obj3["extra"]["fit_provenance"]["sources"]["attempt_record"].endswith("control_relu@cluster_a@2") and "note" in obj3["extra"]["fit_provenance"]
    # a malformed sidecar is an error, never ignored
    (d / hooks.FIT_PROVENANCE_SIDECAR).write_text(json.dumps({"hello": 1}))
    with pytest.raises(ValueError, match="not a fit-provenance sidecar"):
        hooks.object_from_sidecar(str(d), producer=emit.producer_block("w"), **kw)
    (d / hooks.FIT_PROVENANCE_SIDECAR).unlink()
    # dry run: the records the hook WOULD write, with emission OFF and no spool — nothing written
    monkeypatch.setenv(emit.DISABLED_ENV, "1")
    monkeypatch.delenv(emit.SPOOL_ENV, raising=False)
    recs = hooks.emit_object_from_sidecar(str(d), script="w", dry_run=True, fit_provenance=prior, **kw)
    assert [r for r, _ in recs] == ["object", "artifact"] and recs[0][1]["fit_job_id"] == "821364" and recs[1][1]["created_utc"] == "2026-10-01T19:32:29.000Z"
    assert not list(tmp_path.glob("**/*.jsonl")) and not (d / "registry_emission.json").exists()
    assert hooks.emit_object_from_sidecar(str(d), script="w", **kw) == []            # the real call with emission off: inert
