"""The Qwen seed-replicate battery: the wave's objects filed ONCE through ``hooks.emit_object_from_sidecar``.

A synthetic bank (invented bytes, the lane's REAL id grammar and sidecar shapes, the registry's real ``legacy_spellings.yaml``
for what the writer states): a ReLU-steer fit, a control LoRA fit, a derived LoRA difference and an Arditi operating point.
Asserted on the emitted ``object`` / ``artifact`` records: the object id IS the pinned sha, kind / label / role / target kind /
objective / seed / positions / campaign per object. Known-bad: a pin that disagrees with the bytes, a sidecar seed that
disagrees with the id, an unknown arm spelling, a derived id without ``derived_meta.json`` — each REFUSED, nothing emitted.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts" / "sdfbase" / "seedrep",):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import emit_objects as EO  # noqa: E402
from sisteer.registry import emit  # noqa: E402
from sisteer.registry import records as R  # noqa: E402
from sisteer.registry import validate as VAL  # noqa: E402
from sisteer.registry import synth

SEEDS_DIR = REPO / "configs" / "registry"
POLICY = REPO / "configs" / "registry" / "decode_policy.json"
BASE_REV = "1001bb4d826a52d1f399e183466143f4da7b741b"


def _blob(d: Path, seed: bytes) -> str:
    d.mkdir(parents=True, exist_ok=True)
    data = hashlib.sha256(seed).digest() * 40
    (d / "object.pt").write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _fitted(bank: Path, staged_id: str, s: int, corpus: str, arm: str, *, seed_in_sidecar=None) -> str:
    d = bank / staged_id
    sha = _blob(d, staged_id.encode())
    (d / "train_budget_meta.json").write_text(json.dumps({
        "arm": arm, "cell_id": f"{corpus}__{arm}", "seed": s if seed_in_sidecar is None else seed_in_sidecar, "object_sha256": sha,
        "data_mode": "documents", "training_budget_kind": "document_tokens", "model_id": "Qwen/Qwen3.5-4B-Base", "model_revision": BASE_REV,
        "chat_template_bypassed": True, "positions_at_fit": "all", "target_layers": [3, 7, 11, 15], "lr": 0.0045, "epochs": 16, "batch_size": 2,
        "resolved_batch_size": 2, "effective_batch": 64, "trainer": "sdfbase.fit", "template_hash": "a4aee8afcf2e0711", "site": "resid_post",
        "fit_record": {"fit_fingerprint": "f" * 64, "epochs_run": 16}}))
    (d / "COMPLETE").write_text(json.dumps({"object_sha256": sha, "cell_id": f"{corpus}__{arm}"}))
    (d / "staged_from.json").write_text(json.dumps({"staged_id": staged_id, "seed": s, "object_sha256": sha}))
    return sha


def _derived(bank: Path, staged_id: str, parents) -> str:
    d = bank / staged_id
    sha = _blob(d, staged_id.encode())
    (d / "derived_meta.json").write_text(json.dumps({
        "kind": "lora_subtraction", "derived_id": "constv2-mix-full__lora_r4_ce_minus_ctl", "derivation": "delta(parent_pos) - delta(parent_neg), structural",
        "model_key": "qwen35_4b_base_eb64_s1", "model_id": "Qwen/Qwen3.5-4B-Base", "model_revision": BASE_REV, "positions": "all",
        "parent_pos": {"cell_id": "constv2-mix-full__lora_r4_ce", "object_sha256": parents[0]},
        "parent_neg": {"cell_id": "constv2ctl2-full__lora_r4_ce", "object_sha256": parents[1]}, "object_sha256": sha}))
    (d / "staged_from.json").write_text(json.dumps({"staged_id": staged_id, "object_sha256": sha}))
    return sha


def _arditi(root: Path) -> tuple:
    d = root / "ladder" / "cells" / EO.ARDITI_STAGED_ID
    sha = _blob(d, b"arditi")
    (d / "fit_provenance.json").write_text(json.dumps({"arditi_allpos": {
        "c": 0.3, "kind": "vector_all", "positions": "all", "phase": "both", "fit_positions": "all", "fit_layer": 13, "served_layer": 12,
        "stored_layer": 12,                                        # the rung producer records both: a vector_all blob is served AS STORED
        "direction_sha256": "9" * 64, "producer": "scripts/arditi/make_arditi_allpos_objects.py", "producer_commit": "e" * 40,
        "cross_model_caveat": "served in the paper's own configuration"}}))
    (d / "staged_from.json").write_text(json.dumps({"staged_id": EO.ARDITI_STAGED_ID, "object_sha256": sha}))
    pins = {"qwen35_4b": {EO.ARDITI_STAGED_ID: {"object_sha256": sha, "kind": "vector_all", "c": 0.3, "dose_unit": "raw_coefficient", "served_layer": 12,
                                                "fit_layer": 13, "direction_sha256": "9" * 64, "alpha_equiv": 0.1054, "positions": "all", "phase": "both",
                                                "fit_positions": "all", "view": "ar_c0p3"}}}
    p = root / "arditi_pins.json"
    p.write_text(json.dumps(pins))
    return d, p, sha


def _bank(tmp_path: Path):
    bank = tmp_path / "bank" / "cells"
    shas = {
        "base64s1__constv2-mix-full__psr_ce_all": _fitted(bank, "base64s1__constv2-mix-full__psr_ce_all", 1, "constv2-mix-full", "psr_ce_all"),
        "base64s2__constv2ctl2-full__lora_r4_ce": _fitted(bank, "base64s2__constv2ctl2-full__lora_r4_ce", 2, "constv2ctl2-full", "lora_r4_ce"),
    }
    shas["derived_base64s3__constv2-mix-full__lora_r4_ce_minus_ctl"] = _derived(bank, "derived_base64s3__constv2-mix-full__lora_r4_ce_minus_ctl",
                                                                              ("1" * 64, "2" * 64))
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps({"qwen35_4b": {k: {"object_sha256": v, "source": "synthetic"} for k, v in shas.items()},
                                "gemma4e4b": {"base__x": {"object_sha256": "0" * 64}}}))
    return bank, pins, shas


def _records(spool: Path):
    out = []
    for p in R.iter_record_files(str(spool)):
        for _i, rec, sha in R.read_record_file(p):
            assert isinstance(VAL.validate_record(rec, p, sha), VAL.Validated), rec
            out.append(rec)
    return out


@pytest.fixture(autouse=True)
def _emission_on(monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")


def test_lane_objects_and_arditi_are_filed_with_pinned_ids_and_declared_facts(tmp_path):
    bank, pins, shas = _bank(tmp_path)
    ar_dir, ar_pins, ar_sha = _arditi(tmp_path / "arditi")
    spool = tmp_path / "spool"
    out = EO.emit_all(bank=bank, pins_path=pins, arditi_dir=ar_dir, arditi_pins_path=ar_pins, campaign="reseed", arditi_campaign="arditi_strength",
                      seeds_dir=SEEDS_DIR, policy_path=POLICY, spool=str(spool), dry_run=False, code_sha=synth.code_sha("c"))
    assert [e["staged_id"] for e in out] == sorted(shas) + [EO.ARDITI_STAGED_ID] and all(len(e["paths"]) == 2 for e in out)
    recs = _records(spool)
    objs = {r["payload"]["object_id"]: r["payload"] for r in recs if r["record_type"] == "object"}
    arts = {r["payload"]["artifact_id"]: r["payload"] for r in recs if r["record_type"] == "artifact"}
    assert set(objs) == set(shas.values()) | {ar_sha} and set(arts) == set(objs)      # object_id IS the pinned sha, one artifact each
    psr = objs[shas["base64s1__constv2-mix-full__psr_ce_all"]]
    assert (psr["object_kind"], psr["paper_label"], psr["object_role"], psr["fit_target_kind"], psr["fit_objective"]) == \
        ("relu_steer", "constitution_relu", "constitution", "corpus_document_fit", "ce")
    assert psr["fit_seed"] == 1 and psr["fit_seed_confidence"] == "recorded" and psr["fit_positions"] == "all" and psr["campaign"] == "reseed"
    assert psr["fit_corpus_id"] == "constv2-mix-full" and psr["fit_model_id"] == "qwen35_4b_base" and psr["fit_thinking_render"] == "not_applicable"
    assert psr["fit_layers_json"] == [3, 7, 11, 15] and psr["status"] == "approved" and psr["derivation"] == "none"
    assert psr["extra"]["fit_replicate_label"] == "s1" and psr["extra"]["training_seed"] == 1 and psr["extra"]["pin_sha256"] == psr["object_id"]
    assert psr["extra"]["code_sha_at_emission"] == synth.code_sha("c") and psr["campaign_cell_id_legacy"] == "constv2-mix-full__psr_ce_all"
    lora = objs[shas["base64s2__constv2ctl2-full__lora_r4_ce"]]
    assert (lora["object_kind"], lora["paper_label"], lora["object_role"], lora["fit_objective"], lora["fit_seed"], lora["fit_corpus_id"]) == \
        ("lora", "control_lora", "control", "ce", 2, "constv2ctl2-full")
    diff = objs[shas["derived_base64s3__constv2-mix-full__lora_r4_ce_minus_ctl"]]
    assert (diff["object_kind"], diff["paper_label"], diff["object_role"], diff["fit_target_kind"], diff["fit_objective"], diff["derivation"]) == \
        ("lora_subtraction", "difference_lora", "difference", "derived", "not_applicable", "lora_subtraction")
    assert diff["fit_positions"] == "inherited_from_parents" and diff["fit_seed_confidence"] == "not_applicable" and "fit_seed" not in diff
    assert diff["extra"]["parent_object_sha256s"] == ["1" * 64, "2" * 64] and diff["extra"]["fit_replicate_label"] == "s3"
    # the derived record carries its parents in the COLUMN the pairing rule requires (FIT_TARGET_RULES["derived"]: parent_object_ids_json
    # NOT NULL) and the derivation record — and passes the writer-side validator the registry's DB CHECK mirrors
    assert diff["parent_object_ids_json"] == ["1" * 64, "2" * 64] and diff["derivation_params_json"]["parents"]["parent_pos"]["object_sha256"] == "1" * 64
    assert VAL.fit_target_pairing_problem(diff) is None and all(VAL.fit_target_pairing_problem(o) is None for o in objs.values())
    ar = objs[ar_sha]
    assert (ar["object_kind"], ar["paper_label"], ar["object_role"], ar["fit_target_kind"], ar["fit_objective"], ar["campaign"]) == \
        ("vector_all", "arditi", "baseline_published", "contrastive_direction", "arditi_search", "arditi_strength")
    assert ar["extra"]["dose"] == {"c": 0.3, "unit": "raw_coefficient"} and ar["fit_model_id"] == "qwen35_4b_it"
    assert ar["fit_positions"] == "unverified" and ar["field_confidence_json"]["fit_positions"]["confidence"] == "unverified"   # honest: the sidecar nests it
    # the contrastive columns the pairing rule requires — the SERVED block from the pins (served_layer 12, zero_based_decoder_block: the
    # arditi_allpos adapter's statement; fit_layer 13 is the fit-time residual index, another convention, riding extra — demo-import, Q14's
    # corollary); the contrast sets stated honestly (the staged direction records no capture-set ids or shas); and the record passes the rule
    assert ar["fit_layers_json"] == [12] and ar["fit_layer_index_convention"] == "zero_based_decoder_block" and ar["extra"]["fit_layer"] == 13
    assert ar["fit_contrast_sets_json"]["confidence"] == "unverified" and "no capture-set ids" in ar["fit_contrast_sets_json"]["reason"]
    assert VAL.fit_target_pairing_problem(ar) is None
    for a in arts.values():
        assert a["kind"] == "object_weights" and a["bytes"] == 32 * 40
    # every object lands in its own record directory under the spool, under its campaign
    dirs = {p.parent.name for e in out for p in map(Path, e["paths"])}
    assert dirs == {f"object__{sha[:16]}" for sha in list(shas.values()) + [ar_sha]}
    assert {p.parent.parent.name for e in out for p in map(Path, e["paths"])} == {"reseed", "arditi_strength"}


def test_known_bad_objects_are_refused_and_nothing_is_emitted(tmp_path):
    bank, pins, shas = _bank(tmp_path)
    ar_dir, ar_pins, ar_sha = _arditi(tmp_path / "arditi")
    common = dict(arditi_dir=ar_dir, arditi_pins_path=ar_pins, campaign="reseed", arditi_campaign="arditi_strength", seeds_dir=SEEDS_DIR,
                  policy_path=POLICY, dry_run=False)
    # (a) a pin that disagrees with the bytes
    bad_pins = tmp_path / "bad_pins.json"
    d = json.loads(pins.read_text())
    d["qwen35_4b"]["base64s1__constv2-mix-full__psr_ce_all"]["object_sha256"] = "0" * 64
    bad_pins.write_text(json.dumps(d))
    with pytest.raises(EO.ObjectsRefused, match="the pin says"):
        EO.emit_all(bank=bank, pins_path=bad_pins, spool=str(tmp_path / "s1"), **common)
    assert not (tmp_path / "s1").exists()
    # (b) a sidecar seed that disagrees with the id's training seed
    _fitted(bank, "base64s3__constv2-mix-full__lora_r4_ce", 3, "constv2-mix-full", "lora_r4_ce", seed_in_sidecar=1)
    d = json.loads(pins.read_text())
    d["qwen35_4b"]["base64s3__constv2-mix-full__lora_r4_ce"] = {"object_sha256": hashlib.sha256(hashlib.sha256(b"base64s3__constv2-mix-full__lora_r4_ce").digest() * 40).hexdigest()}
    p2 = tmp_path / "pins2.json"
    p2.write_text(json.dumps(d))
    with pytest.raises(EO.ObjectsRefused, match="sidecar seed 1 disagrees"):
        EO.emit_all(bank=bank, pins_path=p2, spool=str(tmp_path / "s2"), **common)
    assert not (tmp_path / "s2").exists()
    # (c) an unknown arm spelling — no legacy_spellings entry
    sha_x = _fitted(bank, "base64s1__constv2-mix-full__mystery_arm", 1, "constv2-mix-full", "mystery_arm")
    p3 = tmp_path / "pins3.json"
    p3.write_text(json.dumps({"qwen35_4b": {"base64s1__constv2-mix-full__mystery_arm": {"object_sha256": sha_x}}}))
    with pytest.raises(EO.ObjectsRefused, match="no legacy_spellings.yaml object_kinds entry"):
        EO.emit_all(bank=bank, pins_path=p3, spool=str(tmp_path / "s3"), **common)
    # (d) a derived id whose directory carries no derived_meta.json
    sha_d = _fitted(bank, "derived_base64s1__constv2-mix-full__lora_r4_ce_minus_ctl", 1, "constv2-mix-full", "lora_r4_ce_minus_ctl")
    p4 = tmp_path / "pins4.json"
    p4.write_text(json.dumps({"qwen35_4b": {"derived_base64s1__constv2-mix-full__lora_r4_ce_minus_ctl": {"object_sha256": sha_d}}}))
    with pytest.raises(EO.ObjectsRefused, match="derived=True but derived_meta.json present=False"):
        EO.emit_all(bank=bank, pins_path=p4, spool=str(tmp_path / "s4"), **common)
    # (e) the Arditi pin of another kind / a sidecar that records a different object sha (the hook's own refusal, surfaced)
    d = json.loads(ar_pins.read_text())
    d["qwen35_4b"][EO.ARDITI_STAGED_ID]["kind"] = "vector"
    p5 = tmp_path / "ar5.json"
    p5.write_text(json.dumps(d))
    with pytest.raises(EO.ObjectsRefused, match="not 'vector_all'"):
        EO.emit_all(bank=bank, pins_path=pins, spool=str(tmp_path / "s5"), **{**common, "arditi_pins_path": p5, "only": [EO.ARDITI_STAGED_ID]})
    # (f) an EDITED pin: the right bytes, served_layer 13 — the pin's sha protects the bytes, not its layer metadata, so the served block is
    # read from the object's own provenance and the pin must agree (the registry owner's review of the hook diff)
    d = json.loads(ar_pins.read_text())
    d["qwen35_4b"][EO.ARDITI_STAGED_ID]["served_layer"] = 13
    p7 = tmp_path / "ar7.json"
    p7.write_text(json.dumps(d))
    with pytest.raises(EO.ObjectsRefused, match="the arditi pin's served_layer 13 disagrees with fit_provenance.json:arditi_allpos.served_layer 12"):
        EO.emit_all(bank=bank, pins_path=pins, spool=str(tmp_path / "s7"), **{**common, "arditi_pins_path": p7, "only": [EO.ARDITI_STAGED_ID]})
    # (g) a provenance whose stored_layer is not its served_layer, or that records no stored_layer, describes no one layer: refused by name
    prov_path = ar_dir / "fit_provenance.json"
    prov = json.loads(prov_path.read_text())
    good = json.dumps(prov)
    prov["arditi_allpos"]["stored_layer"] = 11
    prov_path.write_text(json.dumps(prov))
    with pytest.raises(EO.ObjectsRefused, match="served_layer 12 disagrees with .stored_layer 11"):
        EO.emit_all(bank=bank, pins_path=pins, spool=str(tmp_path / "s8"), **{**common, "only": [EO.ARDITI_STAGED_ID]})
    del prov["arditi_allpos"]["stored_layer"]
    prov_path.write_text(json.dumps(prov))
    with pytest.raises(EO.ObjectsRefused, match="fit_provenance.json:arditi_allpos.stored_layer is not an integer"):
        EO.emit_all(bank=bank, pins_path=pins, spool=str(tmp_path / "s9"), **{**common, "only": [EO.ARDITI_STAGED_ID]})
    prov_path.write_text(good)
    assert not any((tmp_path / f"s{i}").exists() for i in range(7, 10))
    tbm = bank / "base64s2__constv2ctl2-full__lora_r4_ce" / "train_budget_meta.json"
    t = json.loads(tbm.read_text())
    t["object_sha256"] = "f" * 64
    tbm.write_text(json.dumps(t))
    with pytest.raises(EO.ObjectsRefused, match="wrote nothing"):
        EO.emit_all(bank=bank, pins_path=pins, spool=str(tmp_path / "s6"), **{**common, "only": ["base64s2__constv2ctl2-full__lora_r4_ce"]})
    assert not any((tmp_path / f"s{i}").exists() for i in range(3, 7))


def test_dry_run_describes_without_writing_and_cli_round_trips(tmp_path, capsys):
    bank, pins, shas = _bank(tmp_path)
    ar_dir, ar_pins, ar_sha = _arditi(tmp_path / "arditi")
    rc = EO.main(["--bank", str(bank), "--pins", str(pins), "--arditi-dir", str(ar_dir), "--arditi-pins", str(ar_pins), "--seeds-dir", str(SEEDS_DIR),
                  "--policy", str(POLICY), "--spool", str(tmp_path / "spool"), "--dry-run"])
    assert rc == 0 and not (tmp_path / "spool").exists()
    lines = [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]
    assert len(lines) == 4 and all(l["paths"] == [] for l in lines)
    assert lines[0]["would_emit"]["object_id"] == shas["base64s1__constv2-mix-full__psr_ce_all"] and lines[0]["would_emit"]["fit_seed"] == 1
    assert lines[-1]["would_emit"]["object_kind"] == "vector_all" and lines[-1]["would_emit"]["campaign"] == "arditi_strength"
    derived = next(l for l in lines if l["staged_id"].startswith("derived_"))
    assert derived["would_emit"]["parent_object_ids_json"] == ["1" * 64, "2" * 64] and derived["would_emit"]["derivation"] == "lora_subtraction"
    rc = EO.main(["--bank", str(bank), "--pins", str(pins), "--no-arditi", "--seeds-dir", str(SEEDS_DIR), "--policy", str(POLICY),
                  "--spool", str(tmp_path / "spool"), "--only", "base64s1__constv2-mix-full__psr_ce_all"])
    assert rc == 0
    recs = _records(tmp_path / "spool")
    assert [r["record_type"] for r in recs] == ["object", "artifact"] and recs[0]["payload"]["object_id"] == shas["base64s1__constv2-mix-full__psr_ce_all"]
    rc = EO.main(["--bank", str(bank), "--pins", str(pins), "--no-arditi", "--seeds-dir", str(SEEDS_DIR), "--policy", str(POLICY),
                  "--spool", str(tmp_path / "spool"), "--status", "nonsense"])
    assert rc == 2 and "OBJECT_STATUSES" in capsys.readouterr().err


def test_a_derived_cell_without_both_parents_is_refused_before_any_record(tmp_path):
    """The fixture's parents are what the registry requires; a derived sidecar naming only one parent is refused by
    the emitter itself, by name, with nothing spooled — never a record the registry would refuse unread."""
    bank, pins, shas = _bank(tmp_path)
    d = bank / "derived_base64s3__constv2-mix-full__lora_r4_ce_minus_ctl"
    dm = json.loads((d / "derived_meta.json").read_text()); dm["parent_neg"] = {"cell_id": "constv2ctl2-full__lora_r4_ce"}
    (d / "derived_meta.json").write_text(json.dumps(dm))
    with pytest.raises(EO.ObjectsRefused, match="names no complete parents"):
        EO.emit_all(bank=bank, pins_path=pins, arditi_dir=None, arditi_pins_path=None, campaign="reseed", arditi_campaign="arditi_strength",
                    seeds_dir=SEEDS_DIR, policy_path=POLICY, spool=str(tmp_path / "spool"), dry_run=False, code_sha="c" * 40)
    assert not (tmp_path / "spool").exists()
