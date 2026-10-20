"""The manual LoRA architecture pick is recorded in TWO append-only rulings: phase A at the pick --
`registry ruling add --action operating_point --scope-kind model --provenance-kind user_direct` naming the rank / module set /
per-expert rank / search / winning trial (model-scoped: matches no roster row) -- and phase B after the three LoRA objects publish
(object-scoped, binding phase A). Every LoRA `object` record stamps `fit_lr_selected_by = manual`,
`extra.architecture_selected_by = manual` and `extra.architecture_selection_ruling_id` = the phase-A id at publish time. Driven through
the REAL `registry ruling add` and the REAL object hook on a synthetic registry; both rulings fold with their provenance; no roster row's
selection changes (the pick never touches the paper's `v_paper_current`)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import emit, hooks, synth
from sisteer.registry import harvest as H
from sisteer.registry import records as R

MODEL = synth.MODEL_ID
FIT_MODEL = synth.FIT_MODEL_ID
SEARCH_ID = "hpo-native:fedcba9876543210"
PHASE_A = f"r_synth-operating_point-{MODEL}-lora-architecture-selection"
PHASE_B = f"r_synth-operating_point-{MODEL}-lora-architecture-objects"
PICK = {"lora_rank": 16, "lora_modules": ["k_proj", "o_proj", "q_proj", "v_proj"], "per_expert_rank": {"shared_expert": 16, "routed_expert": 2},
        "fit_search_id": SEARCH_ID, "trial_id": "r1_t07", "phase": "architecture_selection"}


def _registry(tmp_path: Path) -> str:
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    # a judged run of the paper's synthetic object, so v_paper_current has a selection to hold still
    rid, recs = synth.run_records("paper_k", 100, object_key="lora@abcd", object_id=synth.OBJECT_SHA)
    R.write_records(os.path.join(root, "records", "c", "000001-run.jsonl"), recs)
    R.write_records(os.path.join(root, "records", "dims", "000002-search.jsonl"), [synth.rec("fit_search", {
        "search_id": SEARCH_ID, "campaign": synth.CAMPAIGN, "target_object_class_json": {"model_id": FIT_MODEL, "object_kind": "lora"},
        "method": "sobol", "n_trials": 16, "n_trials_completed": 16, "search_space_json": {"lr": {"kind": "log_uniform", "low": 1e-5, "high": 1e-3},
                                                                                             "lora_rank": {"kind": "choice", "values": [4, 16, 64]}},
        "objective": "best_heldout_loss", "objective_direction": "minimize", "substrate": "cluster_a", "code_sha": synth.code_sha("a"), "started_utc": synth.t(0)}, synth.t(0))])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    return root


def _selection_state(conn) -> str:
    rows = conn.execute("SELECT run_id, in_current_paper, selection_state, decisive_ruling_id FROM v_paper_current ORDER BY run_id").fetchall()
    return hashlib.sha256(json.dumps([list(r) for r in rows]).encode()).hexdigest()


def _lora_object(tmp_path: Path, name: str, role: str, ruling_id: str) -> str:
    d = tmp_path / "objects" / name
    d.mkdir(parents=True)
    (d / "object.pt").write_bytes(name.encode() * 128)
    (d / "train_budget_meta.json").write_text(json.dumps({"arm": "lora_r4_ce", "si_id": name.split("__")[0], "positions_at_fit": "all", "seed": 1,
                                                          "model_id": FIT_MODEL, "target_layers": [3, 7, 11], "training_budget_kind": "document_tokens",
                                                          "sample_manifest_sha": synth.SHA256_B, "lr": 1e-4, "trainer": "train_two_si_objects.py"}))
    (d / hooks.FIT_SELECTION_SIDECAR).write_text(json.dumps({"fit_search_id": SEARCH_ID, "fit_lr_selected_by": "manual", "lora_rank": PICK["lora_rank"],
                                                             "lora_modules": PICK["lora_modules"], "per_expert_rank": PICK["per_expert_rank"],
                                                             "architecture_selection_ruling_id": ruling_id, "init_scheme": "random_lecun_1e-3"}))
    kind = "lora_subtraction" if role == "difference" else "lora"
    if role == "difference":
        (d / "derived_meta.json").write_text(json.dumps({"parent_object_ids": ["p" * 64, "q" * 64], "params": {"basis": "concat"}}))
    out = hooks.emit_object_from_sidecar(str(d), campaign=synth.CAMPAIGN, object_kind=kind, paper_label=hooks.paper_label_for(kind, role), object_role=role,
                                         fit_target_kind="derived" if role == "difference" else "corpus_document_fit",
                                         fit_objective="not_applicable" if role == "difference" else "ce", fit_thinking_render="off", fit_model_id=FIT_MODEL,
                                         derivation="lora_subtraction" if role == "difference" else "none", script="tests.pick")
    assert out and len(out) == 2
    return hashlib.sha256((d / "object.pt").read_bytes()).hexdigest()


def test_pick_then_fits_then_bind_records_both_rulings_and_moves_no_paper_selection(tmp_path, monkeypatch, capsys):
    root = _registry(tmp_path)
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.setenv(emit.SPOOL_ENV, os.path.join(root, "records", "fits"))
    monkeypatch.setenv(emit.SUBSTRATE_ENV, "cluster_a")
    with DB.reading(DB.db_path(root)) as conn:
        before = _selection_state(conn)
        assert conn.execute("SELECT COUNT(*) FROM v_paper_current").fetchone()[0] >= 1
    # -- phase A: at the pick, immediately (model-scoped; the fits may now start)
    rc = cli.main(["--root", root, "ruling", "add", "--round", "r_synth", "--action", "operating_point", "--scope-kind", "model",
                   "--scope-json", json.dumps({"match": {"model_ids": [MODEL]}}), "--target-json", json.dumps(PICK),
                   "--reason", f"LoRA architecture for {MODEL} picked by the user from the published rank x module-set table: rank 16, set qkvo, per-expert rank 2",
                   "--source-ref", f"synthetic://hpo-table#{SEARCH_ID}", "--ruled-by", "user", "--ruled-utc", "2026-09-29T12:00:00.000Z",
                   "--provenance-kind", "user_direct", "--id", PHASE_A])
    assert rc == 0
    path_a = capsys.readouterr().out.strip()
    assert path_a.startswith(os.path.join(root, "rulings", "added")) and PHASE_A in path_a
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    # -- the fits (fixture objects): every LoRA object stamps the phase-A id at publish time
    ids = [_lora_object(tmp_path, "constv2ctl2-full__lora_r4_ce", "control", PHASE_A),
           _lora_object(tmp_path, "constv2-mix-full__lora_r4_ce", "constitution", PHASE_A),
           _lora_object(tmp_path, "constv2-mix-full__lora_r4_ce_minus_ctl", "difference", PHASE_A)]
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    # -- phase B: after the three artifacts are published (object-scoped, binding phase A); idempotent
    argv_b = ["--root", root, "ruling", "add", "--round", "r_synth", "--action", "operating_point", "--scope-kind", "object",
              "--scope-json", json.dumps({"match": {"model_ids": [MODEL], "object_ids": ids}}),
              "--target-json", json.dumps({"binds": PHASE_A, "phase": "object_binding"}),
              "--reason", "binds the user's architecture selection to the published LoRA objects", "--source-ref", PHASE_A, "--ruled-by", "user",
              "--ruled-utc", "2026-09-29T13:00:00.000Z", "--provenance-kind", "user_direct", "--id", PHASE_B]
    assert cli.main(argv_b) == 0
    capsys.readouterr()
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    with DB.reading(DB.db_path(root)) as conn:
        rulings = {r["ruling_id"]: dict(r) for r in conn.execute("SELECT * FROM ruling WHERE ruling_id IN (?, ?)", (PHASE_A, PHASE_B)).fetchall()}
        assert set(rulings) == {PHASE_A, PHASE_B}
        a, b = rulings[PHASE_A], rulings[PHASE_B]
        assert a["action"] == "operating_point" and a["scope_kind"] == "model" and json.loads(a["scope_json"])["match"]["model_ids"] == [MODEL]
        assert json.loads(a["target_json"]) == PICK and a["ruled_by"] == "user" and a["ruled_utc"].startswith("2026-09-29T12:00:00")
        assert json.loads(a["extra_json"])["provenance"]["kind"] == "user_direct"
        assert b["scope_kind"] == "object" and json.loads(b["scope_json"])["match"]["object_ids"] == ids
        assert json.loads(b["target_json"]) == {"binds": PHASE_A, "phase": "object_binding"} and b["source_ref"] == PHASE_A
        assert json.loads(b["extra_json"])["provenance"]["kind"] == "user_direct"
        # the object rows: manual selection, the architecture in the real columns, the phase-A id stamped at publish time
        for oid in ids:
            o = conn.execute("SELECT fit_lr_selected_by, fit_rank, fit_alpha, fit_target_modules_json, fit_search_id, extra_json, paper_label FROM object WHERE object_id = ?", (oid,)).fetchone()
            assert o[0] == "manual" and o[1] == 16 and o[2] == 16.0 and json.loads(o[3]) == PICK["lora_modules"] and o[4] == SEARCH_ID, tuple(o)
            ex = json.loads(o[5])
            assert ex["architecture_selection_ruling_id"] == PHASE_A and ex["architecture_selected_by"] == "manual" and ex["per_expert_rank"] == PICK["per_expert_rank"]
        assert {conn.execute("SELECT paper_label FROM object WHERE object_id = ?", (oid,)).fetchone()[0] for oid in ids} == {"control_lora", "constitution_lora", "difference_lora"}
        # no roster row's selection changed: the paper's frame is byte-identical before and after both rulings
        assert _selection_state(conn) == before
        assert conn.execute("SELECT COUNT(*) FROM v_paper_current WHERE decisive_ruling_id IN (?, ?)", (PHASE_A, PHASE_B)).fetchone()[0] == 0
    # idempotent: adding phase B again writes another record file naming the same ruling id and the fold keeps ONE row
    assert cli.main(argv_b) == 0
    capsys.readouterr()
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM ruling WHERE ruling_id = ?", (PHASE_B,)).fetchone()[0] == 1


def test_phase_a_without_provenance_or_with_a_bad_action_is_refused(tmp_path, capsys):
    root = _registry(tmp_path)
    rc = cli.main(["--root", root, "ruling", "add", "--round", "r_synth", "--action", "pick", "--scope-kind", "model",
                   "--scope-json", json.dumps({"match": {"model_ids": [MODEL]}}), "--reason", "x", "--source-ref", "y", "--id", "bad"])
    assert rc == 1 and "ruling refused" in capsys.readouterr().err
    assert not [f for f in os.listdir(os.path.join(root, "rulings", "added")) if "bad" in f] if os.path.isdir(os.path.join(root, "rulings", "added")) else True
