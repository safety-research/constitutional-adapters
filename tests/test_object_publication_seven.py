"""The SEVEN objects the pipeline publishes reach the registry natively: the four corpus fits through the PRODUCTION
`train_two_si_objects.publish_cell` → `_registry_publish` transition (the selection
serialised into the TEMP cell dir by `write_fit_selection` BEFORE publication, hashed into the COMPLETE manifest, read from the
PUBLISHED dir), the two differences and the refusal direction through the hook the derived-object builders call
(`hooks.emit_object_from_sidecar` with `derivation` / role `baseline_published`) -- including the OVERWRITE case (a torn final dir is
removed) and the RESUME case (a `validated_complete` cell is skipped and its record not re-emitted); every record folds with the reducer
(adapters off) and validates; the rows carry the selection fields in their REAL columns, the architecture in `extra`, the labels
`difference_lora` / `difference_relu` / `arditi_allpos`, and the manifest hash of the selection sidecar."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from sisteer.registry import db as DB
from sisteer.registry import emit, hooks, synth
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import validate as VAL

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
import train_two_si_objects as T  # noqa: E402  (the production trainer module; torch is imported lazily by its fit paths)

SEARCH_ID = "hpo-native:0123456789abcdef"
RULING_ID = "r_synth-operating_point-synthmodel-lora-architecture-selection"
FITS = (("constv2-mix-full", "lora_r4_ce", "constitution", "lora", "constitution_lora"),
        ("constv2ctl2-full", "lora_r4_ce", "control", "lora", "control_lora"),
        ("constv2-mix-full", "psr_ce_all", "constitution", "relu_steer", "constitution_relu"),
        ("constv2ctl2-full", "psr_ce_all", "control", "relu_steer", "control_relu"))


def _registry(tmp_path: Path) -> str:
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    R.write_records(os.path.join(root, "records", "dims", "000002-search.jsonl"), [synth.rec("fit_search", {
        "search_id": SEARCH_ID, "campaign": synth.CAMPAIGN, "target_object_class_json": {"model_id": synth.FIT_MODEL_ID, "object_kind": "lora"},
        "method": "sobol", "n_trials": 16, "n_trials_completed": 16, "search_space_json": {"lr": {"kind": "log_uniform", "low": 1e-5, "high": 1e-3}},
        "objective": "best_heldout_loss", "objective_direction": "minimize", "substrate": "cluster_a", "code_sha": "a" * 40, "started_utc": synth.t(0)}, synth.t(0))])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    return root


def _selection(arm: str, si: str) -> dict:
    if arm == "lora_r4_ce":
        return {"fit_search_id": SEARCH_ID, "fit_lr_selected_by": "manual", "lora_rank": 16, "lora_modules": ["v_proj", "q_proj", "k_proj", "o_proj"],
                "per_expert_rank": {"shared_expert": 16, "routed_expert": 2}, "architecture_selection_ruling_id": RULING_ID,
                "init_scheme": "random_lecun_1e-3", "lora_set_id": "qkvo", "trainable_params": 3440640}
    return {"fit_search_id": SEARCH_ID, "fit_lr_selected_by": "search", "init_scheme": "random_lecun_1e-3"}


def _temp_cell(save_dir: Path, si: str, arm: str, seed_bytes: bytes) -> Path:
    """What the trainer's fit leaves in its TEMP cell dir before publication (T._cell_tmp_dir shape): the artifact, the budget
    sidecar, the metrics mirror and -- through the PRODUCTION writer -- the selection sidecar."""
    cell_id = f"{si}__{arm}"
    tmp = save_dir / f"{cell_id}.tmp-{os.getpid()}"
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    # the production selection writer holds the sidecar against the SAVED object, so the synthetic artefact embodies the cell's
    # selection (a readable torch blob with the architecture keys; the seed bytes keep every cell's sha distinct), no weights
    import torch
    sel_ = _selection(arm, si)
    blob = ({"kind": "lora_all", "layers": [3, 7, 11], "rank": sel_["lora_rank"], "lora_modules": list(sel_["lora_modules"]),
             "lora_set_id": sel_["lora_set_id"], "per_expert_rank": dict(sel_["per_expert_rank"]), "init_scheme": sel_["init_scheme"]}
            if arm == "lora_r4_ce" else {"kind": "psr", "layers": [3, 7, 11], "train_direction": True, "init_scheme": sel_["init_scheme"]})
    torch.save({**blob, "cell_id": cell_id, "seed_tag": seed_bytes.hex()}, tmp / "object.pt")
    sha = hashlib.sha256((tmp / "object.pt").read_bytes()).hexdigest()
    meta = {"arm": arm, "si_id": si, "positions_at_fit": "all", "response_header_len": 5, "batch_size": 2, "resolved_batch_size": 64, "max_len": 8192,
            "epochs": 3, "seed": 1, "lr": 1e-4, "lr_schedule": "cosine", "warmup_steps": 10, "max_grad_norm": 0.37, "length_sorted": True,
            "teacher_prefix_cache": False, "trainer": "train_two_si_objects.py", "completion_tokens": 0, "object_sha256": sha,
            "model_id": synth.FIT_MODEL_ID, "model_revision": "b" * 40, "template_hash": "b48c47f644389271", "site": "resid_post",
            "target_layers": [3, 7, 11], "cell_id": cell_id, "training_budget_kind": "document_tokens", "sample_manifest_sha": synth.SHA256_B,
            "fit_record": {"floor_ratio_max": None}}
    (tmp / "train_budget_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    (tmp / "metrics.jsonl").write_text(json.dumps({"step": 1, "train/loss": 2.0}) + "\n")
    cfg = {"train_kwargs": {arm: {"fit_selection": _selection(arm, si)}}}
    p = T.write_fit_selection(tmp, cfg, si, arm)
    assert p is not None and p.name == T.FIT_SELECTION_NAME and p.parent == tmp
    return tmp


def _publish(save_dir: Path, si: str, arm: str, tmp: Path) -> Path:
    return T.publish_cell(save_dir, f"{si}__{arm}", tmp, si_id=si, arm=arm, positions="all", audit={"metrics_rows": 1}, fingerprint="f" * 64)


def _records(spool: str):
    out = []
    for p in R.iter_record_files(spool):
        for _i, rec, _sha in R.read_record_file(p, compute_sha=False):
            out.append(rec)
    return out


def test_seven_objects_publish_natively_with_overwrite_and_resume(tmp_path, monkeypatch):
    root = _registry(tmp_path)
    spool = os.path.join(root, "records", "fits")
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.setenv(emit.SPOOL_ENV, spool)
    monkeypatch.setenv(emit.SUBSTRATE_ENV, "cluster_a")
    monkeypatch.setenv("REGISTRY_CAMPAIGN", synth.CAMPAIGN)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    save_dir = tmp_path / "bank" / "cells_all"
    save_dir.mkdir(parents=True)
    # --- the four fits through the PRODUCTION publish_cell (one of them over a TORN final dir: the overwrite case) -----------------
    torn = save_dir / "constv2-mix-full__lora_r4_ce"
    torn.mkdir()
    (torn / "object.pt").write_bytes(b"torn earlier publish")
    shas = {}
    for k, (si, arm, _role, _kind, _label) in enumerate(FITS):
        tmp = _temp_cell(save_dir, si, arm, bytes([k + 1]))
        final = _publish(save_dir, si, arm, tmp)
        assert final == save_dir / f"{si}__{arm}" and not tmp.exists()
        marker = json.loads((final / T.COMPLETE_MARKER_NAME).read_text())
        assert T.FIT_SELECTION_NAME in marker["files"]                                         # publication evidence: the selection is hashed into COMPLETE
        assert marker["files"][T.FIT_SELECTION_NAME] == hashlib.sha256((final / T.FIT_SELECTION_NAME).read_bytes()).hexdigest()
        shas[(si, arm)] = marker["object_sha256"]
    assert not (torn / "object.pt").read_bytes().startswith(b"torn")                              # the torn dir was superseded, not merged
    # --- RESUME: a validated_complete cell is skipped -- the trainer never re-publishes it, so no second record --------------------
    n_before = len([r for r in _records(spool) if r["record_type"] == "object"])
    assert n_before == 4
    resumed = T.validated_complete(save_dir, "constv2ctl2-full__psr_ce_all", "f" * 64)
    assert resumed == save_dir / "constv2ctl2-full__psr_ce_all" / "object.pt"
    assert T.validated_complete(save_dir, "constv2ctl2-full__psr_ce_all", "0" * 64) is None        # another fit identity: not this cell's
    assert len([r for r in _records(spool) if r["record_type"] == "object"]) == n_before
    # --- the two DIFFERENCES and the REFUSAL DIRECTION through the hook the builders call ---------------------------------------------
    derived_root = tmp_path / "bank" / "derived"
    parents_lora = [shas[("constv2-mix-full", "lora_r4_ce")], shas[("constv2ctl2-full", "lora_r4_ce")]]
    parents_relu = [shas[("constv2-mix-full", "psr_ce_all")], shas[("constv2ctl2-full", "psr_ce_all")]]
    for name, kind, deriv, parents, params in (("constv2-mix-full__lora_r4_ce_minus_ctl", "lora_subtraction", "lora_subtraction", parents_lora, {"rank_out": 32, "basis": "concat"}),
                                                ("constv2-mix-full__psr_ce_all_minus_ctl", "composed", "composition", parents_relu, {"write": "paired", "scale": 1.0})):
        d = derived_root / name
        d.mkdir(parents=True)
        (d / "object.pt").write_bytes(name.encode() * 64)
        (d / "derived_meta.json").write_text(json.dumps({"parent_object_ids": parents, "params": params, "derivation": deriv}))
        (d / T.FIT_SELECTION_NAME).write_text(json.dumps({"fit_search_id": SEARCH_ID, "fit_lr_selected_by": "inherited_from:" + parents[0],
                                                          "architecture_selection_ruling_id": RULING_ID if kind == "lora_subtraction" else None,
                                                          **({"lora_rank": 32, "lora_modules": ["k_proj", "o_proj", "q_proj", "v_proj"]} if kind == "lora_subtraction" else {})}))
        out = hooks.emit_object_from_sidecar(str(d), campaign=synth.CAMPAIGN, object_kind=kind, paper_label=hooks.paper_label_for(kind, "difference"),
                                             object_role="difference", fit_target_kind="derived", fit_objective="not_applicable", fit_thinking_render="off",
                                             fit_model_id=synth.FIT_MODEL_ID, derivation=deriv, script=f"scripts/sdfbase/{'lora_subtract' if kind == 'lora_subtraction' else 'psr_residual_distill'}.py",
                                             extra={"per_expert_rank": {"shared_expert": 32, "routed_expert": 4}} if kind == "lora_subtraction" else None)
        assert out and len(out) == 2
    ar = derived_root / "ar_c0p3__arditi_allpos"
    ar.mkdir(parents=True)
    (ar / "object.pt").write_bytes(b"refusal direction bytes" * 64)
    (ar / "train_budget_meta.json").write_text(json.dumps({"positions_at_fit": "all", "seed": 1, "model_id": synth.MODEL_ID, "site": "resid_post",
                                                            "target_layers": [3, 7, 11, 15], "trainer": "fit_refusal_direction.py"}))
    out = hooks.emit_object_from_sidecar(str(ar), campaign=synth.CAMPAIGN, object_kind="vector_all", paper_label=hooks.paper_label_for("vector_all", "baseline_published"),
                                         object_role="baseline_published", fit_target_kind="contrastive_direction", fit_objective="diffmean", fit_thinking_render="off",
                                         fit_model_id=synth.MODEL_ID, script="scripts/arditi/fit_refusal_direction.py",
                                         fit_contrast_sets_json={"harmful": "sha256:" + "d" * 64, "harmless": "sha256:" + "e" * 64},
                                         extra={"coefficient": 0.3, "staged_id": "ar_c0p3__arditi_allpos"})
    assert out and len(out) == 2
    # --- fold (the reducer; no adapter ever runs here) and validate all seven ------------------------------------------------------
    recs = _records(spool)
    objs = [r for r in recs if r["record_type"] == "object"]
    assert len(objs) == 7 and all(isinstance(VAL.validate_record(r, "<mem>"), VAL.Validated) for r in recs), [str(VAL.validate_record(r, "<mem>")) for r in recs]
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        rows = {r["object_id"]: dict(r) for r in conn.execute("SELECT * FROM object WHERE campaign = ? AND object_id NOT IN ('none', ?)",
                                                              (synth.CAMPAIGN, synth.OBJECT_SHA)).fetchall()}      # the seeds' own two objects aside
        assert len(rows) == 7
        by_label = {r["paper_label"]: r for r in rows.values()}
        assert set(by_label) == {"constitution_lora", "control_lora", "constitution_relu", "control_relu", "difference_lora", "difference_relu", "arditi_allpos"}
        cl = by_label["constitution_lora"]
        assert cl["object_kind"] == "lora" and cl["object_role"] == "constitution" and cl["fit_target_kind"] == "corpus_document_fit" and cl["fit_objective"] == "ce"
        assert cl["fit_corpus_id"] == "constv2-mix-full" and cl["fit_corpus_sha256"] == synth.SHA256_B and cl["fit_model_id"] == synth.FIT_MODEL_ID
        assert cl["fit_search_id"] == SEARCH_ID and cl["fit_lr_selected_by"] == "manual" and cl["fit_rank"] == 16 and cl["fit_alpha"] == 16.0
        assert json.loads(cl["fit_target_modules_json"]) == ["k_proj", "o_proj", "q_proj", "v_proj"] and json.loads(cl["fit_layers_json"]) == [3, 7, 11]
        ex = json.loads(cl["extra_json"])
        assert ex["per_expert_rank"] == {"shared_expert": 16, "routed_expert": 2} and ex["architecture_selection_ruling_id"] == RULING_ID
        assert ex["architecture_selected_by"] == "manual" and ex["init_scheme"] == "random_lecun_1e-3" and ex["lora_set_id"] == "qkvo"
        assert cl["object_id"] == cl["artifact_sha256"] == shas[("constv2-mix-full", "lora_r4_ce")]
        cr = by_label["control_relu"]
        assert cr["object_kind"] == "relu_steer" and cr["object_role"] == "control" and cr["fit_lr_selected_by"] == "search" and cr["fit_rank"] is None
        assert cr["fit_search_id"] == SEARCH_ID and cr["fit_corpus_id"] == "constv2ctl2-full"
        dl = by_label["difference_lora"]
        assert dl["object_kind"] == "lora_subtraction" and dl["derivation"] == "lora_subtraction" and dl["fit_target_kind"] == "derived"
        assert json.loads(dl["parent_object_ids_json"]) == parents_lora and dl["fit_positions"] == "inherited_from_parents" and dl["fit_rank"] == 32
        assert dl["fit_lr_selected_by"] == "inherited_from:" + parents_lora[0]
        dr = by_label["difference_relu"]
        assert dr["object_kind"] == "composed" and dr["derivation"] == "composition" and json.loads(dr["parent_object_ids_json"]) == parents_relu
        ad = by_label["arditi_allpos"]
        assert ad["object_kind"] == "vector_all" and ad["object_role"] == "baseline_published" and ad["fit_target_kind"] == "contrastive_direction"
        assert ad["fit_objective"] == "diffmean" and json.loads(ad["fit_contrast_sets_json"])["harmful"].startswith("sha256:") and ad["fit_model_id"] == synth.MODEL_ID
        assert json.loads(ad["extra_json"])["coefficient"] == 0.3
        # every natively emitted object: the producer's script, no adapter revision anywhere, the fit substrate declared
        for r in rows.values():
            assert r["fit_substrate"] == "cluster_a" and r["status"] == "reference"
        assert conn.execute("SELECT COUNT(*) FROM artifact WHERE kind = 'object_weights'").fetchone()[0] == 7
        assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    finally:
        conn.close()
    scripts = {r["producer"]["script"] for r in objs}
    assert "scripts/train_two_si_objects.py publish_cell" in scripts and "scripts/arditi/fit_refusal_direction.py" in scripts


def test_hook_maps_admit_the_pipelines_arms_and_pools():
    assert hooks.object_kind_for_legacy_kind("lora_r4_ce") == "lora" and hooks.object_kind_for_legacy_kind("psr_ce_all") == "relu_steer"
    assert hooks.object_kind_for_legacy_kind("lora_r4_ce_minus_ctl") == "lora_subtraction" and hooks.object_kind_for_legacy_kind("psr_ce_all_minus_ctl") == "composed"
    assert hooks.object_role_for_corpus_or_si("constv2-mix-full") == "constitution" and hooks.object_role_for_corpus_or_si("constv2ctl2-full") == "control"
    assert hooks.object_role_for_corpus_or_si("si_constitution") == "constitution" and hooks.object_role_for_corpus_or_si("some-other-pool") is None
    assert hooks.paper_label_for("lora_subtraction", "difference") == "difference_lora" and hooks.paper_label_for("composed", "difference") == "difference_relu"
    assert hooks.paper_label_for("vector_all", "baseline_published") == "arditi_allpos" and hooks.paper_label_for("vector_all", "control") == "other"
    assert hooks.FIT_SELECTION_SIDECAR in hooks.SIDECAR_NAMES and hooks.FIT_SELECTION_SIDECAR == T.FIT_SELECTION_NAME
