"""scripts/sdfbase/registry_emission.py — the ONE owner of the object writers' registry-emission evidence: the
evidence file + failure marker, the spool scan, emit-with-evidence, verify-or-replay; and the pins that every
producer in the area goes through it (the trainer's self-contained ``_registry_publish`` keeps an inline writer
whose literals equal the module's constants).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts" / "sdfbase", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import registry_emission as re_mod  # noqa: E402

SHA = "a" * 64


def _dir(tmp_path: Path, sha: str = SHA) -> Path:
    d = tmp_path / "obj"
    d.mkdir()
    (d / "object.pt").write_bytes(b"synthetic object bytes" * 8)
    (d / "COMPLETE").write_text(json.dumps({"object_sha256": sha, "object_file": "object.pt"}))
    (d / "derived_meta.json").write_text(json.dumps({"object_sha256": sha}))
    return d


def test_constants_and_marker_reader(tmp_path):
    assert re_mod.EMISSION_NAME == "registry_emission.json" and re_mod.EMIT_FAILED_NAME == "REGISTRY_EMIT_FAILED"
    assert re_mod.EMIT_FAILED_RC == 3 and re_mod.STATUSES == ("emitted", "not_emitted", "failed")
    d = _dir(tmp_path)
    assert re_mod.marker_object_id(d) == SHA and re_mod.json_field(d / "derived_meta.json", "object_sha256") == SHA
    assert re_mod.marker_object_id(tmp_path) is None and re_mod.read_evidence(d) is None
    assert re_mod.utc_now().endswith("Z")


def test_write_read_verify_and_the_marker(tmp_path):
    d = _dir(tmp_path)
    ev = re_mod.write_evidence(d, "emitted", object_id=SHA, writer="w", records=["/spool/x.jsonl"], campaign="two_si")
    assert re_mod.read_evidence(d) == ev and re_mod.evidence_verifies(ev, SHA)
    assert not re_mod.evidence_verifies(ev, "b" * 64) and not re_mod.evidence_verifies(ev, None) and not re_mod.evidence_verifies(None, SHA)
    assert not (d / re_mod.EMIT_FAILED_NAME).exists()
    bad = re_mod.write_evidence(d, "failed", object_id=SHA, writer="w", error="RuntimeError: boom")
    assert json.loads((d / re_mod.EMIT_FAILED_NAME).read_text()) == bad and not re_mod.evidence_verifies(bad, SHA)
    off = re_mod.write_evidence(d, "not_emitted", object_id=SHA, writer="w", reason="off")
    assert not (d / re_mod.EMIT_FAILED_NAME).exists() and not re_mod.evidence_verifies(off, SHA)
    with pytest.raises(ValueError, match="not one of"):
        re_mod.write_evidence(d, "maybe", object_id=SHA, writer="w")


def test_emit_with_evidence_maps_every_hook_outcome(tmp_path, capsys):
    d = _dir(tmp_path)
    ev = re_mod.emit_with_evidence(d, object_id=SHA, writer="w", campaign="two_si", emit=lambda: ["/s/1.jsonl", "/s/2.jsonl"])
    assert ev["status"] == "emitted" and ev["records"] == ["/s/1.jsonl", "/s/2.jsonl"] and ev["campaign"] == "two_si"
    ev = re_mod.emit_with_evidence(d, object_id=SHA, writer="w", campaign="two_si", emit=lambda: [])
    assert ev["status"] == "not_emitted" and "off on this host" in ev["reason"]
    ev = re_mod.emit_with_evidence(d, object_id=SHA, writer="w", campaign="two_si", emit=lambda: None, what="thing")
    assert ev["status"] == "failed" and "inside its guard" in ev["error"] and (d / re_mod.EMIT_FAILED_NAME).is_file()
    assert "thing not recorded" in capsys.readouterr().err

    def boom():
        raise RuntimeError("synthetic spool failure")

    ev = re_mod.emit_with_evidence(d, object_id=SHA, writer="scripts/x.py", campaign="two_si", emit=boom)
    assert ev["status"] == "failed" and ev["error"] == "RuntimeError: synthetic spool failure" and ev["writer"] == "scripts/x.py"
    err = capsys.readouterr().err
    assert "scripts/x.py" in err and "REGISTRY_EMIT_FAILED written" in err and "replays the record without rebuilding" in err

    def no_pkg():
        raise ImportError("no sisteer.registry here")

    ev = re_mod.emit_with_evidence(d, object_id=SHA, writer="w", campaign="two_si", emit=no_pkg)
    assert ev["status"] == "not_emitted" and "registry package unavailable" in ev["reason"]
    assert not (d / re_mod.EMIT_FAILED_NAME).exists()


def test_ensure_emitted_verifies_scans_the_spool_or_replays_exactly_once(tmp_path, monkeypatch, capsys):
    from sisteer.registry import emit

    d = _dir(tmp_path)
    calls = []
    emit_fn = lambda: calls.append(1) or ["/s/1.jsonl"]  # noqa: E731
    # nothing yet: replay once, then verified
    ev = re_mod.ensure_emitted(d, object_id=SHA, writer="w", campaign="two_si", emit=emit_fn)
    assert ev["status"] == "emitted" and calls == [1] and "re-emitting the object record" in capsys.readouterr().out
    assert re_mod.ensure_emitted(d, object_id=SHA, writer="w", campaign="two_si", emit=emit_fn) == ev and calls == [1]
    # stale evidence (another object id) / failed / not_emitted are never terminal
    for status, more in (("failed", {"error": "x"}), ("not_emitted", {"reason": "off"})):
        re_mod.write_evidence(d, status, object_id=SHA, writer="w", **more)
        re_mod.ensure_emitted(d, object_id=SHA, writer="w", campaign="two_si", emit=emit_fn)
    assert calls == [1, 1, 1]
    re_mod.write_evidence(d, "emitted", object_id="b" * 64, writer="w", records=["/s/old.jsonl"])
    re_mod.ensure_emitted(d, object_id=SHA, writer="w", campaign="two_si", emit=emit_fn)
    assert calls == [1, 1, 1, 1] and re_mod.read_evidence(d)["object_id"] == SHA
    # a record of the id already in the spool counts: evidence written, nothing re-emitted
    (d / re_mod.EMISSION_NAME).unlink()
    spool = tmp_path / "spool"
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    from sisteer.registry import synth
    w = emit.RecordWriter(str(spool), "two_si", f"object__{SHA[:16]}", emit.producer_block("synthetic writer"))
    # presence is ADMISSION-AWARE: a bare stub the registry would refuse is `refused`, not a record it holds
    w.emit_object({"object_id": SHA, "object_kind": "lora", "campaign": "two_si"})
    stub = re_mod.spool_records_for(SHA, "two_si")
    assert stub["object"] == [] and len(stub["refused"]) == 1 and re_mod.reconcile_spool(SHA, "two_si")["state"] == "absent"
    # an admissible pair counts
    w.emit_object(dict(synth.object_lora(sha=SHA), campaign="two_si"))
    w.emit_artifact({"artifact_id": SHA, "kind": "object_weights", "uri_primary": "/x/object.pt", "extra": {"object_id": SHA}})
    recs = re_mod.spool_object_records(SHA, "two_si")
    assert len(recs) == 1 and re_mod.spool_object_records("c" * 64, "two_si") == [] and re_mod.spool_object_records("", "two_si") == []
    both = re_mod.spool_records_for(SHA, "two_si")
    assert both["object"] == recs and len(both["artifact"]) == 1 and both["artifact"][0].endswith("-artifact.jsonl") and len(both["refused"]) == 1
    assert re_mod.reconcile_spool(SHA, "two_si")["state"] == "complete" and re_mod.reconcile_spool("c" * 64, "two_si")["state"] == "absent"
    assert re_mod.reconcile_spool(None, "two_si") == {"state": "absent", "present": {"object": [], "artifact": []},
                                                      "missing": ["object", "artifact"], "records": [], "refused": []}
    ev = re_mod.ensure_emitted(d, object_id=SHA, writer="w", campaign="two_si", emit=emit_fn)
    assert calls == [1, 1, 1, 1] and ev["verified_from"] == "spool" and ev["records"] == recs + both["artifact"]
    assert "already in the spool" in capsys.readouterr().out
    # the spool is read on EVERY non-verifying status, `failed` included: nothing re-emitted
    re_mod.write_evidence(d, "failed", object_id=SHA, writer="w", error="x")
    ev = re_mod.ensure_emitted(d, object_id=SHA, writer="w", campaign="two_si", emit=emit_fn)
    assert calls == [1, 1, 1, 1] and ev["status"] == "emitted" and ev["verified_from"] == "spool"
    assert not (d / re_mod.EMIT_FAILED_NAME).exists()


def _real_object_dir(tmp_path: Path) -> tuple[Path, str]:
    """A synthetic object the REAL hook can describe: the artefact, its train_budget_meta.json (sha recorded) and COMPLETE."""
    import hashlib
    # the corpus-document shape every live fit is (training_budget_kind document_tokens): the record the hook builds from it is one
    # the registry ADMITS (a two-SI distillation shape from a tbm sidecar is not — see tests/test_registry_emit_fit.py), and since
    # 2026-10-02 spool presence counts admissible records only
    d = tmp_path / "cells_all" / "docs_pool__lora_r1_all"
    d.mkdir(parents=True)
    (d / "object.pt").write_bytes(b"synthetic lora object bytes " * 64)
    sha = hashlib.sha256((d / "object.pt").read_bytes()).hexdigest()
    (d / "train_budget_meta.json").write_text(json.dumps({
        "arm": "lora_r1_all", "si_id": "docs_pool", "training_budget_kind": "document_tokens", "positions_at_fit": "all", "seed": 0,
        "lr": 1e-3, "epochs": 3, "max_grad_norm": 1.0, "object_sha256": sha, "model_id": "synthetic/base-4b", "model_revision": "b" * 40,
        "template_hash": "b48c47f644389271", "target_layers": [4, 5, 6], "cell_id": "docs_pool__lora_r1_all"}))
    (d / "COMPLETE").write_text(json.dumps({"cell_id": "docs_pool__lora_r1_all", "object_sha256": sha}))
    return d, sha


HOOK_KW = dict(object_kind="lora", paper_label="constitution_lora", object_role="constitution",
               fit_target_kind="corpus_document_fit", fit_objective="ce", script="scripts/synthetic_writer.py")


def _spool_records(spool: Path) -> list:
    from sisteer.registry import records as R
    return [rec for f in R.iter_record_files(str(spool)) for _, rec, _ in R.read_record_file(f)]


def _artifact_write_fails(monkeypatch):
    """The realistic partial failure: the spool fills BETWEEN the hook's two writes."""
    from sisteer.registry import emit

    def boom(self, payload):
        raise OSError(28, "No space left on device (synthetic)")

    monkeypatch.setattr(emit.RecordWriter, "emit_artifact", boom)


def test_a_failure_between_the_two_writes_replays_only_the_missing_record(tmp_path, monkeypatch, capsys):
    """The REAL hook writes ``object`` then ``artifact``; when the second write fails its guard returns None and the
    evidence says ``failed`` although the object row is in the spool. The resume must reconcile BOTH records and
    replay ONLY the missing one — byte-for-byte what the hook would have written — never a second object row; a
    writer with no per-component replay fails loudly instead of duplicating."""
    from sisteer.registry import emit, hooks

    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    spool = tmp_path / "spool"
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))
    d, sha = _real_object_dir(tmp_path)
    full_emit = lambda: hooks.emit_object_from_sidecar(str(d), campaign="two_si", **HOOK_KW)  # noqa: E731
    per_component = lambda c: re_mod.component_emitter(c)(str(d), campaign="two_si", **HOOK_KW)  # noqa: E731
    # 1. plant the partial row: the artifact write fails inside the hook -> failed evidence, marker, object row present
    _artifact_write_fails(monkeypatch)
    ev = re_mod.emit_with_evidence(d, object_id=sha, writer="w", campaign="two_si", emit=full_emit)
    assert ev["status"] == "failed" and "inside its guard" in ev["error"] and (d / re_mod.EMIT_FAILED_NAME).is_file()
    assert [r["record_type"] for r in _spool_records(spool)] == ["object"]
    rec = re_mod.reconcile_spool(sha, "two_si")
    assert rec["state"] == "partial" and rec["missing"] == ["artifact"] and len(rec["present"]["object"]) == 1
    monkeypatch.undo()                                                    # the spool is writable again
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))
    capsys.readouterr()
    # 2. a writer WITHOUT a per-component replay: refused by name, the spool untouched (no duplicate object row)
    ev = re_mod.ensure_emitted(d, object_id=sha, writer="w", campaign="two_si", emit=full_emit)
    assert ev["status"] == "failed" and "PartialEmission" in ev["error"] and "no artifact record" in ev["error"]
    assert "would duplicate the object row" in ev["error"] and (d / re_mod.EMIT_FAILED_NAME).is_file()
    assert [r["record_type"] for r in _spool_records(spool)] == ["object"]
    # 3. the recovery: ONLY the artifact record is replayed; the object row is not duplicated; evidence says so
    ev = re_mod.ensure_emitted(d, object_id=sha, writer="w", campaign="two_si", emit=full_emit, emit_component=per_component)
    out = capsys.readouterr().out
    assert "replaying ONLY the artifact record" in out and "no row is duplicated" in out
    assert ev["status"] == "emitted" and ev["replayed"] == ["artifact"] and ev["verified_from"] == "spool+replay"
    assert not (d / re_mod.EMIT_FAILED_NAME).exists()
    recs = _spool_records(spool)
    assert [r["record_type"] for r in recs] == ["object", "artifact"] and len(ev["records"]) == 2
    assert ev["records"][0] == rec["present"]["object"][0] and ev["records"][1].endswith("-artifact.jsonl")
    assert recs[1]["payload"]["artifact_id"] == sha and recs[1]["payload"]["extra"]["object_id"] == sha
    # the replayed artifact record IS the hook's: a full emission of the same object elsewhere writes the same payload
    ref_spool = tmp_path / "ref_spool"
    monkeypatch.setenv(emit.SPOOL_ENV, str(ref_spool))
    assert len(full_emit()) == 2
    ref = [r for r in _spool_records(ref_spool) if r["record_type"] == "artifact"][0]["payload"]
    got = dict(recs[1]["payload"])
    for k in ("created_utc",):
        ref.pop(k, None); got.pop(k, None)
    assert got == ref
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))
    # verified now: a second ensure emits nothing, and `failed` evidence over a COMPLETE spool is healed without emitting
    assert re_mod.ensure_emitted(d, object_id=sha, writer="w", campaign="two_si", emit=full_emit, emit_component=per_component) == ev
    re_mod.write_evidence(d, "failed", object_id=sha, writer="w", error="stale failure")
    ev2 = re_mod.ensure_emitted(d, object_id=sha, writer="w", campaign="two_si", emit=full_emit, emit_component=per_component)
    assert ev2["status"] == "emitted" and ev2["verified_from"] == "spool" and len(_spool_records(spool)) == 2
    # 4. the other direction — the object record lost, the artifact present: only the object is replayed
    Path(rec["present"]["object"][0]).unlink()
    assert re_mod.reconcile_spool(sha, "two_si")["missing"] == ["object"]
    (d / re_mod.EMISSION_NAME).unlink()
    ev3 = re_mod.ensure_emitted(d, object_id=sha, writer="w", campaign="two_si", emit=full_emit, emit_component=per_component)
    assert ev3["status"] == "emitted" and ev3["replayed"] == ["object"]
    recs = _spool_records(spool)
    assert sorted(r["record_type"] for r in recs) == ["artifact", "object"]
    obj = [r for r in recs if r["record_type"] == "object"][0]["payload"]
    ref_obj = [r for r in _spool_records(ref_spool) if r["record_type"] == "object"][0]["payload"]
    assert obj["object_id"] == sha == ref_obj["object_id"] and obj["object_kind"] == ref_obj["object_kind"] == "lora"
    assert obj["paper_label"] == ref_obj["paper_label"] and obj["fit_target_kind"] == ref_obj["fit_target_kind"]
    # the component emitter honours the hook's own switch and refuses an unknown component
    monkeypatch.setenv(emit.DISABLED_ENV, "1")
    assert per_component("artifact") == []
    with pytest.raises(ValueError, match="not one of"):
        re_mod.component_emitter("weights")


def test_every_producer_in_the_area_goes_through_the_one_module():
    """The fitter, the two difference builders and the trainer's resume helpers import registry_emission; the
    trainer's self-contained _registry_publish keeps an inline writer whose literals equal the module's."""
    scripts = REPO / "scripts"
    for rel in ("arditi/fit_refusal_direction.py", "sdfbase/lora_subtract.py", "sdfbase/psr_residual_distill.py",
                "train_two_si_objects.py"):
        src = (scripts / rel).read_text()
        assert 'registry_emission.py"' in src or '"registry_emission"' in src, rel
        assert "emit_with_evidence(" in src or "_emission_mod()" in src, rel
        assert "def _spool_object_records" not in src or "spool_object_records(object_id, campaign)" in src, rel
    frd = (scripts / "arditi" / "fit_refusal_direction.py").read_text()
    assert "def _write_emission_evidence" not in frd and "def _spool_object_records" not in frd
    assert ".ensure_emitted(" in frd and ".emit_with_evidence(" in frd
    # every ensure path offers the per-component replay (the hook call takes an emitter)
    for rel in ("arditi/fit_refusal_direction.py", "sdfbase/lora_subtract.py", "sdfbase/psr_residual_distill.py"):
        src = (scripts / rel).read_text()
        assert "emit_component=lambda c:" in src and ".component_emitter(c)" in src and ", emitter=None" in src, rel
        assert "fn = emitter if emitter is not None else _rh.emit_object_from_sidecar" in src, rel
    t2_src = (scripts / "train_two_si_objects.py").read_text()
    assert "hook = emitter if emitter is not None else _rh.emit_object_from_sidecar" in t2_src
    assert "_emission_mod()" in t2_src and "mod.reconcile_spool(object_id, campaign)" in t2_src
    assert "emitter=mod.component_emitter(missing)" in t2_src
    ls = (scripts / "sdfbase" / "lora_subtract.py").read_text()
    assert ".emit_with_evidence(" in ls and ".ensure_emitted(" in ls and "class RegistryEmissionFailed" in ls
    prd = (scripts / "sdfbase" / "psr_residual_distill.py").read_text()
    assert ".emit_with_evidence(" in prd and ".ensure_emitted(" in prd and "def ensure_registry_emission_derived" in prd
    t2 = (scripts / "train_two_si_objects.py").read_text()
    body = t2[t2.index("def _registry_publish"):t2.index("\n\n\n", t2.index("def _registry_publish"))]
    assert f'"{re_mod.EMISSION_NAME}"' in body and f'"{re_mod.EMIT_FAILED_NAME}"' in body
    # self-contained: the body never imports or loads the module (its docstring may NAME the evidence file)
    assert "import registry_emission" not in body and "_emission_mod(" not in body and "registry_emission.py" not in body
    assert "_emission_mod().spool_object_records(" in t2 and "_emission_mod().read_evidence(" in t2
    frd_consts = {ln.split("=")[0].strip(): ln.split("=")[1].split("#")[0].strip().strip('"')
                  for ln in frd.splitlines() if ln.startswith(("EMISSION_NAME =", "EMIT_FAILED_NAME =", "EMIT_FAILED_RC ="))}
    assert frd_consts == {"EMISSION_NAME": re_mod.EMISSION_NAME, "EMIT_FAILED_NAME": re_mod.EMIT_FAILED_NAME,
                          "EMIT_FAILED_RC": str(re_mod.EMIT_FAILED_RC)}
    stage = (REPO / "reproduce" / "04_fit_objects.sh").read_text()
    assert stage.count("published_unfolded ") >= 4          # fit cells, refusal direction, difference_lora, difference_relu


# ---- the campaign is the writer's statement, never a module default (stage-E ruling 2026-10-02) ----------------------

def _parent_with_evidence(tmp_path: Path, name: str, campaign: str | None) -> str:
    """A parent cell dir with (or without) emission evidence naming a campaign; returns its object path."""
    d = tmp_path / name
    d.mkdir()
    (d / "object.pt").write_bytes(b"p" * 16)
    if campaign is not None:
        re_mod.write_evidence(d, "emitted", object_id="b" * 64, writer="w", records=["/s/1.jsonl"], campaign=campaign)
    return str(d / "object.pt")


def test_campaign_from_env_has_no_module_default(monkeypatch):
    monkeypatch.delenv(re_mod.CAMPAIGN_ENV, raising=False)
    assert not hasattr(re_mod, "DEFAULT_CAMPAIGN")
    with pytest.raises(re_mod.CampaignUnset, match="REGISTRY_CAMPAIGN is unset"):
        re_mod.campaign_from_env()
    assert re_mod.campaign_from_env(default="explicit_default") == "explicit_default"     # a caller's EXPLICIT default is honoured
    monkeypatch.setenv(re_mod.CAMPAIGN_ENV, "stated")
    assert re_mod.campaign_from_env() == "stated" and re_mod.campaign_from_env(default="x") == "stated"
    monkeypatch.setenv(re_mod.CAMPAIGN_ENV, "")
    with pytest.raises(re_mod.CampaignUnset):
        re_mod.campaign_from_env()


def test_campaign_for_derived_is_the_parents_campaign_and_never_a_default(tmp_path, monkeypatch):
    """A derived object carries its PARENTS' recorded campaign (both must agree); an explicit REGISTRY_CAMPAIGN only when
    it agrees (or when no parent carries evidence); disagreement of any kind refuses by name — the object table is keyed by
    object_id alone, so a wrong-campaign record would UPSERT a row out of the paper frame."""
    monkeypatch.delenv(re_mod.CAMPAIGN_ENV, raising=False)
    pos = _parent_with_evidence(tmp_path, "pos", "paper_frame")
    neg = _parent_with_evidence(tmp_path, "neg", "paper_frame")
    meta = {"parent_pos": {"object_path": pos}, "parent_neg": {"object_path": neg}}
    assert re_mod.parents_recorded_campaigns(meta) == {"parent_pos": "paper_frame", "parent_neg": "paper_frame"}
    assert re_mod.campaign_for_derived(meta) == "paper_frame"
    monkeypatch.setenv(re_mod.CAMPAIGN_ENV, "paper_frame")
    assert re_mod.campaign_for_derived(meta) == "paper_frame"                     # explicit and agreeing: fine
    monkeypatch.setenv(re_mod.CAMPAIGN_ENV, "two_si")
    with pytest.raises(re_mod.CampaignUnset, match="parents' recorded campaign is 'paper_frame'"):
        re_mod.campaign_for_derived(meta)                                          # explicit but disagreeing: refused by name
    monkeypatch.delenv(re_mod.CAMPAIGN_ENV, raising=False)
    # parents that disagree with each other refuse, env or no env
    neg2 = _parent_with_evidence(tmp_path, "neg2", "other_frame")
    with pytest.raises(re_mod.CampaignUnset, match="disagree"):
        re_mod.campaign_for_derived({"parent_pos": {"object_path": pos}, "parent_neg": {"object_path": neg2}})
    # no parent evidence at all: the explicit env is the statement; nothing -> refused (never `two_si`)
    bare = {"parent_pos": {"object_path": _parent_with_evidence(tmp_path, "p0", None)},
            "parent_neg": {"object_path": _parent_with_evidence(tmp_path, "n0", None)}}
    with pytest.raises(re_mod.CampaignUnset, match="no parent carries emission evidence"):
        re_mod.campaign_for_derived(bare)
    monkeypatch.setenv(re_mod.CAMPAIGN_ENV, "stated")
    assert re_mod.campaign_for_derived(bare) == "stated"
    # a sidecar without parent paths (a pre-2026-10-02 LoRA meta) behaves as "no evidence"
    assert re_mod.campaign_for_derived({"parent_pos": {"cell_id": "x"}, "parent_neg": {"cell_id": "y"}}) == "stated"
    assert re_mod.campaign_for_derived({}) == "stated"


def test_failed_is_the_public_spelling_of_the_pre_hook_refusal(tmp_path, capsys):
    d = _dir(tmp_path)
    ev = re_mod.failed(d, SHA, "scripts/x.py", "derived object", "CampaignUnset: synthetic")
    assert ev["status"] == "failed" and "CampaignUnset: synthetic" in ev["error"] and (d / re_mod.EMIT_FAILED_NAME).is_file()
    assert "derived object not recorded" in capsys.readouterr().err


# ---- terminal evidence is campaign- and contract-aware; spool presence is admission-aware ----------------

def test_evidence_is_terminal_only_for_the_current_contract_and_the_stated_campaign(tmp_path):
    d = _dir(tmp_path)
    ev = re_mod.write_evidence(d, "emitted", object_id=SHA, writer="w", records=["/s/1.jsonl"], campaign="paper_frame")
    assert ev["contract"] == re_mod.EVIDENCE_CONTRACT == "2"
    assert re_mod.evidence_verifies(ev, SHA) and re_mod.evidence_verifies(ev, SHA, "paper_frame")
    assert not re_mod.evidence_verifies(ev, SHA, "two_si")                        # another campaign stated: not terminal
    assert "this run states 'two_si'" in re_mod.evidence_mismatch(ev, SHA, "two_si")
    old = {"object_id": SHA, "status": "emitted", "campaign": "paper_frame", "records": ["/s/1.jsonl"]}   # pre-2026-10-02 evidence
    assert not re_mod.evidence_verifies(old, SHA) and not re_mod.evidence_verifies(old, SHA, "paper_frame")
    assert "emission contract 1" in re_mod.evidence_mismatch(old, SHA, "paper_frame")
    assert not re_mod.evidence_verifies(dict(ev, status="failed"), SHA) and not re_mod.evidence_verifies(ev, "b" * 64)
    assert re_mod.evidence_mismatch(None, SHA) == "no emission evidence"


def _plant_records(spool: Path, campaign: str, obj_dir: Path, sha: str, *, fit_objective: str, only: str | None = None) -> list:
    """The REAL hook's records for a derived object, written straight into the spool the way the pre-fix emitter did (the
    RecordWriter validates nothing): `fit_objective="ce"` plants the formerly emitted inadmissible shape; `only` writes one component."""
    from sisteer.registry import emit, hooks
    payload, artifact = hooks.object_from_sidecar(str(obj_dir), campaign=campaign, object_kind="composed", paper_label="difference_relu",
                                                  object_role="difference", fit_target_kind="derived", fit_objective=fit_objective,
                                                  derivation="composition", fit_model_id="synthmodel_4b_base",
                                                  parent_object_ids_json=["1" * 64, "2" * 64])
    w = emit.RecordWriter(str(spool), campaign, f"object__{sha[:16]}", emit.producer_block("tests.synthetic"))
    out = []
    if only in (None, "object"):
        out.append(w.emit_object(payload))
    if only in (None, "artifact"):
        out.append(w.emit_artifact(artifact))
    return [p for p in out if p]


def test_spool_presence_is_admission_aware_and_the_replay_restates_only_the_refused_record(tmp_path, monkeypatch, capsys):
    from sisteer.registry import emit
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    spool = tmp_path / "spool"
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))
    d = tmp_path / "v"; d.mkdir()
    (d / "object.pt").write_bytes(b"V bytes" * 32)
    import hashlib
    sha = hashlib.sha256((d / "object.pt").read_bytes()).hexdigest()
    (d / "derived_meta.json").write_text(json.dumps({"object_sha256": sha, "derivation": "u_pos - u_neg", "scale_matched": True,
                                                       "builder": "b", "built_at": "t", "parent_pos": {"object_sha256": "1" * 64},
                                                       "parent_neg": {"object_sha256": "2" * 64}}))
    planted = _plant_records(spool, "two_si", d, sha, fit_objective="ce")
    assert len(planted) == 2
    found = re_mod.spool_records_for(sha, "two_si")
    assert found["object"] == [] and len(found["artifact"]) == 1                # the ce object row is NOT a record the registry holds
    assert len(found["refused"]) == 1 and found["refused"][0][1] == "fit_objective" and "object_fit_target_kind_pairing" in found["refused"][0][2]
    rec = re_mod.reconcile_spool(sha, "two_si")
    assert rec["state"] == "partial" and rec["missing"] == ["object"] and len(rec["refused"]) == 1
    # the kept-object replay with old-contract evidence: the object record is re-stated through the per-component emitter,
    # the artifact is not re-written, the whole hook never runs
    (d / re_mod.EMISSION_NAME).write_text(json.dumps({"object_id": sha, "status": "emitted", "campaign": "two_si", "records": planted}))
    calls = []
    def whole(): calls.append("whole"); return ["x"]
    def component(c): calls.append(c); return _plant_records(spool, "two_si", d, sha, fit_objective="not_applicable", only=c)
    ev = re_mod.ensure_emitted(d, object_id=sha, writer="w", campaign="two_si", emit=whole, emit_component=component)
    assert calls == ["object"] and ev["status"] == "emitted" and ev["replayed"] == ["object"] and ev["contract"] == "2"
    assert "REFUSED at ingest" in capsys.readouterr().out
    after = re_mod.spool_records_for(sha, "two_si")
    assert len(after["object"]) == 1 and len(after["artifact"]) == 1 and len(after["refused"]) == 1   # the ce record stays on disk, still refused; a valid one now present
    # terminal now: the same call is a no-op
    assert re_mod.ensure_emitted(d, object_id=sha, writer="w", campaign="two_si", emit=whole, emit_component=component) == ev and calls == ["object"]
    # the same object under a NEWLY stated campaign: nothing in that campaign's spool dir -> the whole hook re-emits once
    ev2 = re_mod.ensure_emitted(d, object_id=sha, writer="w", campaign="paper_frame", emit=whole, emit_component=component)
    assert calls == ["object", "whole"] and ev2["status"] == "emitted" and ev2["campaign"] == "paper_frame"


def test_fit_provenance_from_prior_record_reads_the_builders_statement(tmp_path, monkeypatch):
    from sisteer.registry import emit
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    spool = tmp_path / "spool"
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))
    assert re_mod.fit_provenance_from_prior_record(SHA, "two_si") is None                 # nothing spooled yet
    w = emit.RecordWriter(str(spool), "two_si", f"object__{SHA[:16]}", emit.producer_block("synthetic writer"))
    w.emit_object({"object_id": SHA, "object_kind": "lora", "campaign": "two_si", "fit_code_sha": "a" * 40, "fit_job_id": "821364",
                   "fit_node_hostname": "gpu-23", "fit_substrate": "slurm"})
    w.emit_artifact({"artifact_id": SHA, "kind": "object_weights", "uri_primary": "/x/object.pt", "created_utc": "2026-10-01T19:32:29.000Z"})
    w.emit_object({"object_id": "e" * 64, "object_kind": "lora", "campaign": "two_si", "fit_job_id": "999"})   # an epoch row: not the object's
    prov = re_mod.fit_provenance_from_prior_record(SHA, "two_si")
    assert prov["fit_code_sha"] == "a" * 40 and prov["fit_job_id"] == "821364" and prov["fit_node_hostname"] == "gpu-23"
    assert prov["fit_substrate"] == "slurm" and prov["fit_producer_session"] is None and prov["artifact_created_utc"] == "2026-10-01T19:32:29.000Z"
    assert prov["_source"].endswith("-object.jsonl#object")
    assert re_mod.fit_provenance_from_prior_record(SHA, "paper_frame") is None and re_mod.fit_provenance_from_prior_record(None, "two_si") is None
