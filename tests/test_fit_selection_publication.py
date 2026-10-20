"""The fit SELECTION travels temp dir -> published dir -> COMPLETE manifest -> registry kwargs.

Drives the REAL ``publish_cell -> _registry_publish`` transition of scripts/train_two_si_objects.py from a
fixture TEMP cell dir (a hand-built general-shape ``lora_all`` blob per D contract §2, ``train_budget_meta.json``,
``metrics.jsonl``, and the ``fit_selection.json`` the driver serialises from the generated config's
``train_kwargs.<arm>.fit_selection`` block) with ``hooks.emit_object_from_sidecar`` monkeypatched to capture its
kwargs. Pins: the manifest lists ``fit_selection.json`` with its sha256; ``validated_complete`` accepts the
publication and refuses a tampered, deleted or unmanifested selection; a torn final dir is overwritten; a
pre-selection config publishes byte-identical cells (no file, no manifest entry); the captured kwargs carry the
selection columns + the architecture read off the BLOB (``fit_rank``, ``fit_alpha`` = rank, sorted
``fit_target_modules_json``, ``fit_layers_json``, ``extra.per_expert_rank`` / ``lora_set_id`` /
``trainable_params`` / ``init_scheme`` / ``architecture_selection_ruling_id`` / ``architecture_selected_by``);
facts the hook's signature does not accept ride ``extra``; the published sidecar is what C's hook reads.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import train_two_si_objects as t2  # noqa: E402

SI = "constv2-mix-full"
ARM = "lora_r4_ce"
CELL = f"{SI}__{ARM}"
FP = "f" * 64
SEL = {"fit_search_id": "hpo-native:0123456789abcdef", "fit_lr_selected_by": "manual",
       "lora_rank": 4, "lora_modules": ["up_proj", "gate_proj", "down_proj"],
       "per_expert_rank": {"shared_expert": 4, "routed_expert": 1},
       "architecture_selection_ruling_id": "r2-operating_point-testmodel-lora-architecture-selection",
       "init_scheme": "random_lecun_1e-3"}
CFG = {"model": "fake/model", "model_revision": "0" * 40,
       "train_kwargs": {"default": {"positions": "all"},
                        ARM: {"lr": 1e-3, "lora_rank": 4, "fit_selection": SEL}}}
CFG_NO_SEL = {"model": "fake/model", "train_kwargs": {"default": {}, ARM: {"lr": 1e-3, "lora_rank": 4}}}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _general_lora_blob(seed: int = 0) -> dict:
    """A general-shape lora_all blob (D contract §2): the mlp set on two layers, rank 4, MoE per-expert map."""
    g = torch.Generator().manual_seed(seed)
    layers, mods = [3, 7], ["down_proj", "gate_proj", "up_proj"]
    dims = {"gate_proj": (16, 24), "up_proj": (16, 24), "down_proj": (24, 16)}
    A = {f"{l}.{m}": torch.randn(dims[m][0], 4, generator=g) for l in layers for m in mods}
    B = {f"{l}.{m}": torch.zeros(dims[m][1], 4) for l in layers for m in mods}
    return {"kind": "lora_all", "layers": layers, "rank": 4, "lora_modules": mods, "lora_set_id": "mlp",
            "per_expert_rank": {"shared_expert": 4, "routed_expert": 1}, "init_scheme": "random_lecun_1e-3",
            "targets": {m: {"site": f"mlp.{m}", "layers": layers, "rank": 4, "kind": "linear"} for m in mods},
            "A": A, "B": B, "positions": "all", "response_header_len": None}


def _legacy_lora_blob(seed: int = 1) -> dict:
    g = torch.Generator().manual_seed(seed)
    layers = [0, 1]
    return {"kind": "lora_all", "layers": layers, "rank": 4,
            "A": {str(l): torch.randn(16, 4, generator=g) for l in layers},
            "B": {str(l): torch.randn(12, 4, generator=g) for l in layers},
            "positions": "all", "response_header_len": None}


def _psr_blob() -> dict:
    layers = [0, 1]
    return {"kind": "psr", "layers": layers, "train_direction": True, "positions": "all",
            "directions": {str(l): torch.randn(16) for l in layers},
            "gate_w": {str(l): torch.randn(16) for l in layers},
            "gate_b": {str(l): torch.ones(1) for l in layers}}


def _temp_cell(save_dir: Path, *, blob: dict, cfg: dict = CFG, si: str = SI, arm: str = ARM) -> Path:
    tmp = t2._cell_tmp_dir(save_dir, f"{si}__{arm}")
    torch.save(blob, tmp / "object.pt")
    (tmp / "train_budget_meta.json").write_text(json.dumps({
        "cell_id": f"{si}__{arm}", "si_id": si, "arm": arm, "model_id": "fake/model", "model_revision": "0" * 40,
        "training_budget_kind": "document_tokens", "positions_at_fit": "all", "positions": "all",
        "target_layers": [int(l) for l in blob["layers"]], "object_sha256": _sha(tmp / "object.pt"),
        "seed": 0, "lr": 1e-3, "epochs": 4, "template_hash": "e3b0c44298fc1c14", "trainer": "sdfbase.fit"}))
    (tmp / "metrics.jsonl").write_text(json.dumps({"step": 0, f"{arm}/loss": 1.0}) + "\n")
    t2.write_fit_selection(tmp, cfg, si, arm)
    return tmp


def _publish(save_dir: Path, tmp: Path, *, si: str = SI, arm: str = ARM, fp: str = FP) -> Path:
    return t2.publish_cell(save_dir, f"{si}__{arm}", tmp, si_id=si, arm=arm, positions="all",
                           audit={"metrics_rows": 1, "object_file": "object.pt", "monitors_on": False},
                           fingerprint=fp)


@pytest.fixture(autouse=True)
def _stated_campaign(monkeypatch):
    """Since 2026-10-02 the fit emitter files under the campaign the job env STATES (REGISTRY_CAMPAIGN) and refuses without one
    when emission is on — never the old `two_si` module default. These tests state it explicitly (the literal the file pins)."""
    monkeypatch.setenv("REGISTRY_CAMPAIGN", "two_si")


@pytest.fixture
def capture(monkeypatch):
    from sisteer.registry import hooks

    calls: list = []

    def fake(obj_dir, **kw):
        calls.append((str(obj_dir), kw))
        return ["synthetic-record"]

    monkeypatch.setattr(hooks, "emit_object_from_sidecar", fake)
    return calls


# ----------------------------------------------------------------------------- the sidecar writer


def test_write_fit_selection_stamps_identity_and_refuses_bad_blocks(tmp_path):
    torch.save(_general_lora_blob(), tmp_path / "object.pt")        # the artefact the selection is held against
    p = t2.write_fit_selection(tmp_path, CFG, SI, ARM)
    assert p == tmp_path / t2.FIT_SELECTION_NAME
    rec = json.loads(p.read_text())
    assert {k: rec[k] for k in SEL} == SEL
    assert rec["arm"] == ARM and rec["si_id"] == SI
    assert rec["config_sha"] == t2._cfg_dict_sha(CFG)
    # a pre-selection config writes nothing
    other = tmp_path / "nosel"
    other.mkdir()
    assert t2.write_fit_selection(other, CFG_NO_SEL, SI, ARM) is None
    assert not (other / t2.FIT_SELECTION_NAME).exists()
    # a block that is not a mapping, or that carries a driver stamp, refuses
    bad = {"train_kwargs": {ARM: {"fit_selection": "search-7"}}}
    with pytest.raises(t2.CellAuditError, match="must be a mapping"):
        t2.write_fit_selection(other, bad, SI, ARM)
    clash = {"train_kwargs": {ARM: {"fit_selection": {**SEL, "arm": "psr_ce_all"}}}}
    with pytest.raises(t2.CellAuditError, match="identity stamps"):
        t2.write_fit_selection(other, clash, SI, ARM)


def test_fit_selection_is_never_a_trainer_kwarg():
    """The per-arm block key is driver-consumed on both paths: the lockstep
    driver-keys allow-list names it, and the solo path pops it before the
    protected set (a typo'd config key must still fail loudly, so the pop
    is by the exact name)."""
    assert "fit_selection" in t2._LOCKSTEP_DRIVER_KEYS
    assert t2.lockstep_dropped_keys({"train_kwargs": {"lora_r1_all": {"fit_selection": SEL}}}, "lora_r1_all") == []
    # the solo path's ONE construction of the trainer-facing kwargs drops it by name (stamp-fix-2: arm_trainer_kwargs)
    assert "fit_selection" in t2.DRIVER_ONLY_ARM_KEYS
    kw = t2.arm_trainer_kwargs({"lora_r1_all": {"fit_selection": SEL, "lr": 1e-3, "lora_rank": 4, "floor_ratio_max": None,
                                                  "ce_floor_ratio_max": 1.1, "gradient_checkpointing": True}}, "lora_r1_all")
    assert kw == {"lr": 1e-3, "lora_rank": 4}
    src = (REPO / "scripts" / "train_two_si_objects.py").read_text()
    assert "arm_kw = arm_trainer_kwargs(per_arm_tk, arm)" in src


# ----------------------------------------------------------------------------- publish-time parity


def test_the_selection_is_held_against_the_saved_object_and_a_disagreement_refuses_by_name(tmp_path):
    """stamp-fix-2: the sidecar may publish only an architecture the saved object embodies. The legacy o_proj / zero
    blob under a selection claiming the MLP set, a per-expert map and the LeCun init refuses naming every disagreeing key; the general
    blob that embodies the selection passes; no object to read against is a refusal, never a pass."""
    torch.save(_legacy_lora_blob(), tmp_path / "object.pt")
    with pytest.raises(t2.CellAuditError) as ei:
        t2.write_fit_selection(tmp_path, CFG, SI, ARM)
    msg = str(ei.value)
    assert "does not match the saved object object.pt" in msg and "lora_modules: selection ['down_proj', 'gate_proj', 'up_proj'] vs artefact ['o_proj']" in msg
    assert "init_scheme: selection 'random_lecun_1e-3' vs artefact 'zero'" in msg, msg
    assert "per_expert_rank: selection {'shared_expert': 4, 'routed_expert': 1} vs artefact None" in msg, msg
    assert not (tmp_path / t2.FIT_SELECTION_NAME).exists()
    ok = tmp_path / "ok"; ok.mkdir(); torch.save(_general_lora_blob(), ok / "object.pt")
    assert t2.write_fit_selection(ok, CFG, SI, ARM) == ok / t2.FIT_SELECTION_NAME
    none = tmp_path / "none"; none.mkdir()
    with pytest.raises(t2.CellAuditError, match="no saved object is at"):
        t2.write_fit_selection(none, CFG, SI, ARM)


def test_a_psr_blob_without_init_scheme_is_zero_init_and_refuses_a_lecun_selection(tmp_path):
    """The published control_relu case: PSRObject.to_blob records init_scheme ONLY when it is not zero, so a blob without the key WAS
    built with the zero default — a selection claiming random_lecun_1e-3 over it refuses by name; the same blob with the key passes."""
    psr_cfg = {"model": "fake/model", "model_revision": "0" * 40,
               "train_kwargs": {"default": {}, "psr_ce_all": {"lr": 1e-3, "init_scheme": "random_lecun_1e-3",
                                                             "fit_selection": {"fit_search_id": "hpo-native:0123456789abcdef", "fit_lr_selected_by": "search",
                                                                               "init_scheme": "random_lecun_1e-3"}}}}
    torch.save(_psr_blob(), tmp_path / "object.pt")
    with pytest.raises(t2.CellAuditError, match=r"init_scheme: selection 'random_lecun_1e-3' vs artefact 'zero'"):
        t2.write_fit_selection(tmp_path, psr_cfg, SI, "psr_ce_all")
    ok = tmp_path / "ok"; ok.mkdir(); torch.save({**_psr_blob(), "init_scheme": "random_lecun_1e-3"}, ok / "object.pt")
    assert t2.write_fit_selection(ok, psr_cfg, SI, "psr_ce_all") == ok / t2.FIT_SELECTION_NAME
    assert t2.artefact_architecture(_psr_blob()) == {"init_scheme": "zero"}
    assert t2.artefact_architecture(_legacy_lora_blob()) == {"init_scheme": "zero", "lora_rank": 4, "lora_modules": ["o_proj"], "per_expert_rank": None}


# ----------------------------------------------------------------------------- publication + manifest


def test_publish_manifests_and_hashes_the_selection(tmp_path, capture):
    tmp = _temp_cell(tmp_path, blob=_general_lora_blob())
    sel_sha = _sha(tmp / t2.FIT_SELECTION_NAME)
    final = _publish(tmp_path, tmp)
    assert final == tmp_path / CELL and not tmp.exists()
    assert (final / t2.FIT_SELECTION_NAME).is_file()
    marker = json.loads((final / "COMPLETE").read_text())
    assert marker["files"][t2.FIT_SELECTION_NAME] == sel_sha == _sha(final / t2.FIT_SELECTION_NAME)
    assert set(marker["files"]) == {"object.pt", "train_budget_meta.json", "metrics.jsonl", t2.FIT_SELECTION_NAME}
    assert t2.validated_complete(tmp_path, CELL, FP) == final / "object.pt"
    # resume skip is identity-bound as before
    assert t2.validated_complete(tmp_path, CELL, "e" * 64) is None
    assert len(capture) == 1


def test_tampered_deleted_or_unmanifested_selection_refuses_on_resume(tmp_path, capture):
    final = _publish(tmp_path, _temp_cell(tmp_path, blob=_general_lora_blob()))
    sel = final / t2.FIT_SELECTION_NAME
    original = sel.read_bytes()
    # tampered: the sha in the manifest no longer matches
    rec = json.loads(original)
    rec["lora_rank"] = 64
    sel.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    assert t2.validated_complete(tmp_path, CELL, FP) is None
    sel.write_bytes(original)
    assert t2.validated_complete(tmp_path, CELL, FP) is not None
    # deleted: a manifested file is missing
    sel.unlink()
    assert t2.validated_complete(tmp_path, CELL, FP) is None
    # unmanifested: a selection dropped into a dir whose COMPLETE predates it is not evidence
    legacy = _publish(tmp_path, _temp_cell(tmp_path, blob=_general_lora_blob(), cfg=CFG_NO_SEL))
    assert t2.FIT_SELECTION_NAME not in json.loads((legacy / "COMPLETE").read_text())["files"]
    assert t2.validated_complete(tmp_path, CELL, FP) is not None
    (legacy / t2.FIT_SELECTION_NAME).write_text(json.dumps(SEL))
    assert t2.validated_complete(tmp_path, CELL, FP) is None


def test_torn_final_dir_is_overwritten_by_the_publication(tmp_path, capture):
    torn = tmp_path / CELL
    torn.mkdir()
    (torn / "object.pt").write_bytes(b"stale bytes")
    (torn / t2.FIT_SELECTION_NAME).write_text(json.dumps({**SEL, "lora_rank": 99, "arm": ARM, "si_id": SI, "config_sha": "x"}))
    (torn / "stray.txt").write_text("left over")
    assert t2.validated_complete(tmp_path, CELL, FP) is None            # no COMPLETE: a torn earlier publish
    final = _publish(tmp_path, _temp_cell(tmp_path, blob=_general_lora_blob()))
    assert final == torn
    assert not (final / "stray.txt").exists()
    assert json.loads((final / t2.FIT_SELECTION_NAME).read_text())["lora_rank"] == 4
    assert t2.validated_complete(tmp_path, CELL, FP) is not None


def test_pre_selection_config_publishes_byte_identical_cells(tmp_path, capture):
    final = _publish(tmp_path, _temp_cell(tmp_path, blob=_legacy_lora_blob(), cfg=CFG_NO_SEL))
    assert not (final / t2.FIT_SELECTION_NAME).exists()
    marker = json.loads((final / "COMPLETE").read_text())
    assert set(marker["files"]) == {"object.pt", "train_budget_meta.json", "metrics.jsonl"}
    assert t2.validated_complete(tmp_path, CELL, FP) is not None
    # ...and the hook is still called with the architecture read off the legacy blob
    _, kw = capture[-1]
    assert kw["fit_rank"] == 4 and kw["fit_target_modules_json"] == ["o_proj"]
    assert kw["extra"]["lora_set_id"] == "wo" and "per_expert_rank" not in kw["extra"]
    assert "fit_search_id" not in kw and "fit_selection_sha256" not in kw["extra"]


# ----------------------------------------------------------------------------- the registry kwargs


def test_registry_publish_passes_selection_and_blob_architecture(tmp_path, capture):
    blob = _general_lora_blob()
    final = _publish(tmp_path, _temp_cell(tmp_path, blob=blob))
    assert len(capture) == 1
    obj_dir, kw = capture[0]
    assert obj_dir == str(final)
    assert kw["object_kind"] == "lora" and kw["object_role"] == "constitution"
    assert kw["paper_label"] == "constitution_lora"
    assert kw["fit_target_kind"] == "corpus_document_fit" and kw["fit_objective"] == "ce"
    assert kw["fit_corpus_id"] == SI and kw["fit_teacher_si_id"] is None
    assert kw["fit_thinking_render"] == "off" and kw["status"] == "reference"
    assert kw["script"] == "scripts/train_two_si_objects.py publish_cell"
    # the selection columns
    assert kw["fit_search_id"] == SEL["fit_search_id"]
    assert kw["fit_lr_selected_by"] == "manual"
    # the architecture, read off the BLOB (sorted modules; alpha = rank, scaling 1)
    assert kw["fit_rank"] == 4 and kw["fit_alpha"] == 4.0
    assert kw["fit_target_modules_json"] == ["down_proj", "gate_proj", "up_proj"]
    assert kw["fit_layers_json"] == [3, 7]
    extra = kw["extra"]
    assert extra["per_expert_rank"] == {"shared_expert": 4, "routed_expert": 1}
    assert extra["lora_set_id"] == "mlp"
    assert extra["trainable_params"] == sum(t.numel() for d in (blob["A"], blob["B"]) for t in d.values())
    assert extra["init_scheme"] == "random_lecun_1e-3"
    assert extra["architecture_selection_ruling_id"] == SEL["architecture_selection_ruling_id"]
    assert extra["architecture_selected_by"] == "manual"
    assert extra["fit_selection_sha256"] == _sha(final / t2.FIT_SELECTION_NAME)
    # the published sidecar is exactly what C's hook reads authoritatively
    from sisteer.registry import hooks

    payload, _art = hooks.object_from_sidecar(
        str(final), campaign="two_si", object_kind="lora", paper_label="constitution_lora",
        object_role="constitution", fit_target_kind="corpus_document_fit", fit_objective="ce")
    assert payload["fit_search_id"] == SEL["fit_search_id"] and payload["fit_rank"] == 4
    assert payload["fit_target_modules_json"] == ["down_proj", "gate_proj", "up_proj"]
    assert payload["extra"]["per_expert_rank"] == SEL["per_expert_rank"]
    assert payload["extra"]["architecture_selection_ruling_id"] == SEL["architecture_selection_ruling_id"]
    assert t2.FIT_SELECTION_NAME in payload["object_sidecars_json"]


def test_registry_publish_control_relu_cell(tmp_path, capture):
    si, arm = "constv2ctl2-full", "psr_ce_all"
    cfg = {"model": "fake/model", "train_kwargs": {"default": {}, arm: {"lr": 5e-3, "fit_selection": {
        "fit_search_id": "hpo-native:fedcba9876543210", "fit_lr_selected_by": "search"}}}}
    blob = _psr_blob()
    _publish(tmp_path, _temp_cell(tmp_path, blob=blob, cfg=cfg, si=si, arm=arm), si=si, arm=arm)
    _, kw = capture[-1]
    assert kw["object_kind"] == "relu_steer" and kw["object_role"] == "control"
    assert kw["paper_label"] == "control_relu" and kw["fit_corpus_id"] == si
    assert kw["fit_search_id"] == "hpo-native:fedcba9876543210" and kw["fit_lr_selected_by"] == "search"
    assert "fit_rank" not in kw and "fit_target_modules_json" not in kw
    assert kw["fit_layers_json"] == [0, 1]
    assert kw["extra"]["trainable_params"] == sum(
        t.numel() for k in ("directions", "gate_w", "gate_b") for t in blob[k].values())
    assert kw["extra"]["init_scheme"] == "zero"
    assert "architecture_selected_by" not in kw["extra"] and "lora_set_id" not in kw["extra"]


def test_undeclared_arm_or_pool_records_nothing(tmp_path, capture):
    # the blob embodies CFG's selection (the production writer holds the two together); the subject here is the undeclared pool
    _publish(tmp_path, _temp_cell(tmp_path, blob=_general_lora_blob(), si="some-other-pool"), si="some-other-pool")
    assert capture == []


def test_unaccepted_facts_ride_extra_on_a_narrow_hook(tmp_path, monkeypatch, capture):
    """An installed hook whose ``object_from_sidecar`` does not accept the selection columns still
    gets a valid call: the facts it cannot take ride ``extra`` (a blind kwarg would raise inside the
    guarded hook and lose the record)."""
    from sisteer.registry import hooks

    def narrow(obj_dir, *, campaign, object_kind, paper_label, object_role, fit_target_kind, fit_objective,
               fit_thinking_render="unverified", fit_model_id=None, status="reference", derivation="none",
               fit_teacher_si_id=None, fit_corpus_id=None, producer=None, extra=None, script=None):
        return {}, {}

    monkeypatch.setattr(hooks, "object_from_sidecar", narrow)
    _publish(tmp_path, _temp_cell(tmp_path, blob=_general_lora_blob()))
    _, kw = capture[-1]
    for col in ("fit_search_id", "fit_lr_selected_by", "fit_rank", "fit_alpha", "fit_target_modules_json", "fit_layers_json"):
        assert col not in kw and kw["extra"][col] is not None, col
    assert kw["extra"]["fit_rank"] == 4 and kw["extra"]["fit_target_modules_json"] == ["down_proj", "gate_proj", "up_proj"]
    assert kw["extra"]["lora_set_id"] == "mlp"


def test_registry_publish_is_self_contained_source():
    """tests/test_registry_emit_fit.py execs this function's SOURCE alone in a namespace of
    {Path, os, sys}: every import sits inside the body, no module-level helper is referenced,
    and the slice (to the next triple blank line) ends with the function."""
    src = (REPO / "scripts" / "train_two_si_objects.py").read_text()
    body = src[src.index("def _registry_publish"):src.index("\n\n\n", src.index("def _registry_publish"))]
    assert body.rstrip().endswith("flush=True)") and "def validated_complete" not in body
    ns = {"Path": Path, "os": __import__("os"), "sys": sys}
    exec(compile(body, "t2si_probe", "exec"), ns)       # noqa: S102 — the coverage test's own probe
    assert callable(ns["_registry_publish"])
    for helper in ("_fit_selection_of", "_object_architecture", "_registry_model_id", "_hook_kwargs",
                   "FIT_SELECTION_NAME", "_sha256_file(", "object_artifact_name"):
        assert helper not in body, helper


def test_hook_failure_never_costs_the_publication_but_is_a_visible_error(tmp_path, monkeypatch, capsys):
    """The publication stands (COMPLETE, validated) and the failed emission
    is RECORDED — registry_emission.json {status: failed}, the bare REGISTRY_EMIT_FAILED marker — and RAISED
    after the marker (RegistryEmissionError -> the arm is recorded failed, the job exits non-zero), never
    swallowed into a print."""
    from sisteer.registry import hooks

    def boom(obj_dir, **kw):
        raise RuntimeError("synthetic hook failure")

    monkeypatch.setattr(hooks, "emit_object_from_sidecar", boom)
    final = t2.cell_dir_for(tmp_path, CELL)
    with pytest.raises(t2.RegistryEmissionError, match="synthetic hook failure.*replays the record without retraining"):
        _publish(tmp_path, _temp_cell(tmp_path, blob=_general_lora_blob()))
    assert (final / "COMPLETE").is_file()
    assert t2.validated_complete(tmp_path, CELL, FP) is not None
    assert "published object not recorded" in capsys.readouterr().err
    marker = json.loads((final / "COMPLETE").read_text())
    ev = t2.registry_emission_evidence(final)
    assert ev["status"] == "failed" and ev["object_id"] == marker["object_sha256"]
    assert "RuntimeError: synthetic hook failure" in ev["error"]
    failed = json.loads((final / t2.REGISTRY_EMIT_FAILED_NAME).read_text())
    assert failed == ev
    # the resume path with the hook repaired: re-emits ONCE, clears the marker, never re-fits (nothing here can train)
    calls = []

    def fixed(obj_dir, **kw):
        calls.append(str(obj_dir))
        return ["synthetic-record"]

    monkeypatch.setattr(hooks, "emit_object_from_sidecar", fixed)
    ev2 = t2.ensure_registry_emission(final, si_id=SI, arm=ARM)
    assert calls == [str(final)] and ev2["status"] == "emitted" and ev2["records"] == ["synthetic-record"]
    assert not (final / t2.REGISTRY_EMIT_FAILED_NAME).exists()
    assert t2.ensure_registry_emission(final, si_id=SI, arm=ARM) == ev2 and calls == [str(final)]


def test_publish_writes_emission_evidence_keyed_by_the_object_id(tmp_path, capture):
    final = _publish(tmp_path, _temp_cell(tmp_path, blob=_general_lora_blob()))
    marker = json.loads((final / "COMPLETE").read_text())
    ev = t2.registry_emission_evidence(final)
    assert ev["status"] == "emitted" and ev["object_id"] == marker["object_sha256"] == _sha(final / "object.pt")
    assert ev["records"] == ["synthetic-record"] and ev["campaign"] == "two_si"
    assert ev["writer"] == "scripts/train_two_si_objects.py publish_cell" and ev["utc"].endswith("Z")
    assert not (final / t2.REGISTRY_EMIT_FAILED_NAME).exists()
    assert (final / t2.REGISTRY_EMISSION_NAME).is_file()
    # the evidence is not publication evidence: the manifest never lists it, validated_complete never asks for it
    assert t2.REGISTRY_EMISSION_NAME not in marker["files"] and t2.validated_complete(tmp_path, CELL, FP) is not None


def test_resume_replays_a_missing_or_stale_emission_exactly_once_and_never_retrains(tmp_path, capture):
    """COMPLETE written, then death before the record (or a pre-evidence publication): the validated-COMPLETE
    resume re-emits exactly once, keyed by the marker's object id; a second resume verifies and emits nothing."""
    final = _publish(tmp_path, _temp_cell(tmp_path, blob=_general_lora_blob()))
    assert len(capture) == 1
    (final / t2.REGISTRY_EMISSION_NAME).unlink()                     # the gap the resume closes
    ev = t2.ensure_registry_emission(final, si_id=SI, arm=ARM)
    assert len(capture) == 2 and ev["status"] == "emitted" and capture[-1][0] == str(final)
    assert t2.ensure_registry_emission(final, si_id=SI, arm=ARM) == ev and len(capture) == 2
    # evidence naming ANOTHER object id (a superseded publication's) is stale: re-emitted once more
    ev["object_id"] = "0" * 64
    (final / t2.REGISTRY_EMISSION_NAME).write_text(json.dumps(ev))
    ev3 = t2.ensure_registry_emission(final, si_id=SI, arm=ARM)
    assert len(capture) == 3 and ev3["object_id"] == json.loads((final / "COMPLETE").read_text())["object_sha256"]
    # the trainer's skip path goes through the same verifier: the skip prints the emission status
    assert "ensure_registry_emission(p.parent, si_id=si_id, arm=arm)" in Path(t2.__file__).read_text()
    assert "ensure_registry_emission(p.parent, si_id=args.si, arm=arm)" in Path(t2.__file__).read_text()


def test_emission_off_by_configuration_is_stated_not_failed_and_reemits_once_enabled(tmp_path, monkeypatch, capture):
    """With emission disabled the REAL hook writes nothing: the evidence says not_emitted with the reason, no
    marker, no raise (the operator's call — 04_fit_objects.sh refuses it by name); once emission is on, the
    resume re-emits because not_emitted is never terminal evidence."""
    from sisteer.registry import emit, hooks

    monkeypatch.undo()                                    # the real hook, emission off
    monkeypatch.setenv(emit.DISABLED_ENV, "1")
    monkeypatch.setenv("REGISTRY_CAMPAIGN", "two_si")       # undo() dropped the module fixture's statement; the re-emit below needs one
    final = _publish(tmp_path, _temp_cell(tmp_path, blob=_general_lora_blob()))
    ev = t2.registry_emission_evidence(final)
    assert ev["status"] == "not_emitted" and "off on this host" in ev["reason"]
    assert not (final / t2.REGISTRY_EMIT_FAILED_NAME).exists() and t2.validated_complete(tmp_path, CELL, FP) is not None
    monkeypatch.delenv(emit.DISABLED_ENV)
    calls = []
    monkeypatch.setattr(hooks, "emit_object_from_sidecar", lambda obj_dir, **kw: calls.append(obj_dir) or ["rec"])
    ev2 = t2.ensure_registry_emission(final, si_id=SI, arm=ARM)
    assert calls == [str(final)] and ev2["status"] == "emitted"


def test_an_undeclared_arm_or_pool_states_not_emitted(tmp_path, capture):
    final = _publish(tmp_path, _temp_cell(tmp_path, blob=_general_lora_blob(), si="unknown-pool", arm=ARM),
                     si="unknown-pool")
    assert capture == []
    ev = t2.registry_emission_evidence(final)
    assert ev["status"] == "not_emitted" and "not declared" in ev["reason"] and "unknown-pool" in ev["reason"]


def test_a_spool_record_of_the_object_is_emission_evidence_without_a_reemit(tmp_path, monkeypatch, capture):
    """A publication that predates the evidence file but whose object record already sits in the producer spool
    (the directory the hook writes) is recognised from the spool: evidence written, nothing re-emitted."""
    from sisteer.registry import emit

    final = _publish(tmp_path, _temp_cell(tmp_path, blob=_general_lora_blob()))
    (final / t2.REGISTRY_EMISSION_NAME).unlink()
    sha = json.loads((final / "COMPLETE").read_text())["object_sha256"]
    spool = tmp_path / "spool"
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.setenv("REGISTRY_CAMPAIGN", "two_si")
    from sisteer.registry import hooks
    # the records as the hook writes them (ADMISSIBLE — presence is admission-aware since 2026-10-02: a bare stub would be `refused`)
    payload, artifact = hooks.object_from_sidecar(str(final), campaign="two_si", object_kind="lora", paper_label="constitution_lora",
                                                  object_role="constitution", fit_target_kind="corpus_document_fit", fit_objective="ce",
                                                  fit_thinking_render="off")
    w = emit.RecordWriter(str(spool), "two_si", f"object__{sha[:16]}", emit.producer_block("synthetic writer"))
    w.emit_object(payload)
    art = w.emit_artifact(artifact)
    recs = t2._spool_object_records(sha, "two_si")
    assert len(recs) == 1 and recs[0].startswith(str(spool))
    assert t2._spool_object_records("0" * 64, "two_si") == []
    ev = t2.ensure_registry_emission(final, si_id=SI, arm=ARM)
    assert len(capture) == 1                                        # nothing re-emitted
    assert ev["status"] == "emitted" and ev["verified_from"] == "spool" and ev["records"] == recs + [art]
    assert t2.registry_emission_evidence(final) == ev
    # `failed` evidence over a complete spool is healed the same way: read, never re-emitted
    t2._emission_mod().write_evidence(final, "failed", object_id=sha, writer="w", error="stale")
    assert t2.ensure_registry_emission(final, si_id=SI, arm=ARM)["verified_from"] == "spool" and len(capture) == 1
    assert not (final / t2.REGISTRY_EMIT_FAILED_NAME).exists()


def test_a_failure_after_the_object_record_replays_only_the_artifact_record(tmp_path, monkeypatch, capsys):
    """Publish through the REAL hook with the spool failing between
    its two writes (object landed, artifact did not) — the publication stands, the emission is `failed` and raised;
    the resume reconciles the spool and replays ONLY the artifact record: exactly one object row, one artifact row,
    evidence `replayed: [artifact]`; a second resume verifies and emits nothing."""
    from sisteer.registry import emit, hooks
    from sisteer.registry import records as R

    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    spool = tmp_path / "spool"
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))
    monkeypatch.setenv("REGISTRY_CAMPAIGN", "two_si")

    def artifact_write_fails(self, payload):
        raise OSError(28, "No space left on device (synthetic)")

    def spool_types():
        return [rec["record_type"] for f in R.iter_record_files(str(spool)) for _, rec, _ in R.read_record_file(f)]

    orig_emit_artifact = emit.RecordWriter.emit_artifact
    monkeypatch.setattr(emit.RecordWriter, "emit_artifact", artifact_write_fails)
    final = t2.cell_dir_for(tmp_path, CELL)
    with pytest.raises(t2.RegistryEmissionError, match="inside its guard"):
        _publish(tmp_path, _temp_cell(tmp_path, blob=_general_lora_blob()))
    sha = json.loads((final / "COMPLETE").read_text())["object_sha256"]
    assert t2.validated_complete(tmp_path, CELL, FP) is not None and spool_types() == ["object"]
    assert t2.registry_emission_evidence(final)["status"] == "failed" and (final / t2.REGISTRY_EMIT_FAILED_NAME).is_file()
    # the spool is writable again: the resume replays the artifact record ONLY
    monkeypatch.setattr(emit.RecordWriter, "emit_artifact", orig_emit_artifact)
    capsys.readouterr()
    ev = t2.ensure_registry_emission(final, si_id=SI, arm=ARM)
    out = capsys.readouterr().out
    assert "replaying ONLY the artifact record" in out and "no row is duplicated" in out
    assert ev["status"] == "emitted" and ev["replayed"] == ["artifact"] and ev["verified_from"] == "spool+replay"
    assert spool_types() == ["object", "artifact"] and len(ev["records"]) == 2
    assert not (final / t2.REGISTRY_EMIT_FAILED_NAME).exists()
    recs = [rec for f in R.iter_record_files(str(spool)) for _, rec, _ in R.read_record_file(f)]
    assert recs[0]["payload"]["object_id"] == sha and recs[1]["payload"]["artifact_id"] == sha
    assert recs[1]["producer"]["script"] == "scripts/train_two_si_objects.py publish_cell"     # the hook's own producer block
    assert recs[1]["payload"]["uri_primary"] == str(final / "object.pt") and recs[1]["payload"]["extra"]["object_id"] == sha
    # verified: nothing more is written on the next resume
    assert t2.ensure_registry_emission(final, si_id=SI, arm=ARM) == ev and spool_types() == ["object", "artifact"]
    # the whole-hook replay is still what an EMPTY spool gets (nothing for the id in the spool)
    calls = []
    monkeypatch.setattr(hooks, "emit_object_from_sidecar", lambda obj_dir, **kw: calls.append(obj_dir) or ["rec"])
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv(emit.SPOOL_ENV, str(other / "spool"))
    (final / t2.REGISTRY_EMISSION_NAME).unlink()
    assert t2.ensure_registry_emission(final, si_id=SI, arm=ARM)["status"] == "emitted" and calls == [str(final)]


def test_emission_evidence_literals_match_the_module_constants():
    """_registry_publish is self-contained (it cannot name the module constants), so its literals are pinned to them."""
    src = Path(t2.__file__).read_text()
    body = src[src.index("def _registry_publish"):src.index("\n\n\n", src.index("def _registry_publish"))]
    assert f'"{t2.REGISTRY_EMISSION_NAME}"' in body and f'"{t2.REGISTRY_EMIT_FAILED_NAME}"' in body
    assert t2.REGISTRY_EMISSION_NAME == "registry_emission.json" and t2.REGISTRY_EMIT_FAILED_NAME == "REGISTRY_EMIT_FAILED"
    # the inline evidence stamps the emission contract the module defines (older-contract evidence is not terminal)
    assert f'"contract": "{t2._emission_mod().EVIDENCE_CONTRACT}"' in body and t2._emission_mod().EVIDENCE_CONTRACT == "2"
    # and 04_fit_objects.sh reads the same two names
    stage = (REPO / "reproduce" / "04_fit_objects.sh").read_text()
    assert stage.count("REGISTRY_EMIT_FAILED") >= 2 and stage.count("registry_emission.json") >= 2
    assert "published_unfolded" in stage and "waiting for the fold" in stage and "replays the record without retraining" in stage


def test_non_torch_artifact_emits_the_legacy_call(tmp_path, capture):
    """A safetensors / opaque artifact states no architecture: the call carries no rank, no
    modules, no extra beyond the selection's own facts — the legacy payload plus the selection."""
    tmp = t2._cell_tmp_dir(tmp_path, CELL)
    (tmp / "object.pt").write_bytes(b"synthetic opaque object bytes " * 8)
    (tmp / "train_budget_meta.json").write_text(json.dumps({
        "cell_id": CELL, "si_id": SI, "arm": ARM, "model_id": "fake/model", "training_budget_kind": "document_tokens",
        "positions_at_fit": "all", "object_sha256": _sha(tmp / "object.pt"), "target_layers": [3, 7]}))
    (tmp / "metrics.jsonl").write_text("")
    # the PRODUCTION writer refuses to publish an architecture claim over an artefact it cannot read — by name
    with pytest.raises(t2.CellAuditError, match="cannot be read as a torch artefact"):
        t2.write_fit_selection(tmp, CFG, SI, ARM)
    # the HOOK's handling of an opaque artefact beside a selection sidecar is this test's subject: the sidecar is laid down directly
    (tmp / t2.FIT_SELECTION_NAME).write_text(json.dumps({**SEL, "arm": ARM, "si_id": SI, "config_sha": t2._cfg_dict_sha(CFG),
                                                         "fit_width": 1}, indent=1, sort_keys=True) + "\n")
    _publish(tmp_path, tmp)
    _, kw = capture[-1]
    assert kw["fit_layers_json"] == [3, 7] and kw["fit_rank"] == 4        # the selection's lora_rank, the sidecar's layers
    assert kw["fit_target_modules_json"] == SEL["lora_modules"]            # the selection's list, as stated (the hook sorts)
    assert "lora_set_id" not in kw["extra"] and "trainable_params" not in kw["extra"]
    assert kw["extra"]["per_expert_rank"] == SEL["per_expert_rank"] and kw["extra"]["init_scheme"] == SEL["init_scheme"]


def test_publish_refuses_to_file_without_a_stated_campaign_when_emission_is_on(tmp_path, monkeypatch, capsys):
    """The campaign is the writer's statement: with emission ON and REGISTRY_CAMPAIGN unset, the publication stands and the
    emission is RECORDED failed (evidence + marker naming REGISTRY_CAMPAIGN), raised as RegistryEmissionError — never a record
    under a default campaign (the old `two_si` default filed this model's fit objects outside the paper frame; a default-campaign
    record UPSERTs an adapter-filed object's campaign). With emission OFF the campaign is not needed: `not_emitted`, as before."""
    from sisteer.registry import emit, hooks

    calls = []
    real_hook = hooks.emit_object_from_sidecar
    monkeypatch.setattr(hooks, "emit_object_from_sidecar", lambda obj_dir, **kw: calls.append(kw) or ["synthetic-record"])
    monkeypatch.delenv("REGISTRY_CAMPAIGN", raising=False)
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)           # the suite's off switch lifted …
    monkeypatch.setenv(emit.SPOOL_ENV, str(tmp_path / "spool"))    # … a fixture spool (never the host's) …
    monkeypatch.setenv(hooks.EMIT_ENV, "1")                        # … and emission forced on: the campaign is required
    final = t2.cell_dir_for(tmp_path, CELL)
    with pytest.raises(t2.RegistryEmissionError, match="REGISTRY_CAMPAIGN is unset"):
        _publish(tmp_path, _temp_cell(tmp_path, blob=_general_lora_blob()))
    assert calls == [] and (final / "COMPLETE").is_file()
    ev = t2.registry_emission_evidence(final)
    assert ev["status"] == "failed" and "REGISTRY_CAMPAIGN is unset" in ev["error"] and (final / t2.REGISTRY_EMIT_FAILED_NAME).is_file()
    assert "published object not recorded" in capsys.readouterr().err
    # the replay path (ensure_registry_emission) refuses the same way, loudly
    with pytest.raises(t2.RegistryEmissionError, match="REGISTRY_CAMPAIGN is unset"):
        t2.ensure_registry_emission(final, si_id=SI, arm=ARM)
    assert calls == []
    # stated: the same publication files under the statement, nothing retrained
    monkeypatch.setenv("REGISTRY_CAMPAIGN", "paper_frame")
    ev2 = t2.ensure_registry_emission(final, si_id=SI, arm=ARM)
    assert ev2["status"] == "emitted" and ev2["campaign"] == "paper_frame" and len(calls) == 1 and calls[0]["campaign"] == "paper_frame"
    # emission OFF (the REAL hook, which honours the switch): no campaign needed, stated not_emitted (the pre-existing contract)
    monkeypatch.setattr(hooks, "emit_object_from_sidecar", real_hook)
    monkeypatch.delenv("REGISTRY_CAMPAIGN", raising=False)
    monkeypatch.delenv(hooks.EMIT_ENV, raising=False)
    monkeypatch.setenv(emit.DISABLED_ENV, "1")
    final2 = _publish(tmp_path / "two", _temp_cell(tmp_path / "two", blob=_general_lora_blob()))
    assert t2.registry_emission_evidence(final2)["status"] == "not_emitted" and len(calls) == 1
    assert not (tmp_path / "spool").exists()


def test_kept_cell_replay_reemits_under_the_newly_stated_campaign_and_is_then_terminal(tmp_path, monkeypatch, capsys):
    """The parents' case: a cell published and emitted under the emitters' old `two_si` default; the job now
    states sdfbase_full22. ensure_registry_emission used to return the old evidence as terminal. Now the evidence is terminal
    only for the stated campaign under the current contract: the replay re-emits the complete record set under sdfbase_full22
    (the fit is not re-run), the evidence follows, and a second replay is a no-op. Emission off: no campaign needed, no raise."""
    from sisteer.registry import emit, hooks, records as R

    monkeypatch.undo()                                    # the REAL hook
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    spool = tmp_path / "spool"
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))
    monkeypatch.setenv("REGISTRY_CAMPAIGN", "two_si")
    final = _publish(tmp_path, _temp_cell(tmp_path, blob=_general_lora_blob()))
    sha = json.loads((final / "COMPLETE").read_text())["object_sha256"]
    ev = t2.registry_emission_evidence(final)
    assert ev["status"] == "emitted" and ev["campaign"] == "two_si" and ev["contract"] == "2"
    assert (spool / "two_si" / f"object__{sha[:16]}").is_dir() and not (spool / "sdfbase_full22").exists()
    # the job now states the paper frame: the kept-cell replay re-emits under it
    monkeypatch.setenv("REGISTRY_CAMPAIGN", "sdfbase_full22")
    capsys.readouterr()
    ev2 = t2.ensure_registry_emission(final, si_id=SI, arm=ARM)
    out = capsys.readouterr().out
    assert ev2["status"] == "emitted" and ev2["campaign"] == "sdfbase_full22" and "this run states 'sdfbase_full22'" in out
    new_dir = spool / "sdfbase_full22" / f"object__{sha[:16]}"
    recs = [rec for f in R.iter_record_files(str(new_dir)) for _, rec, _ in R.read_record_file(f)]
    assert [r["record_type"] for r in recs] == ["object", "artifact"] and recs[0]["payload"]["campaign"] == "sdfbase_full22"
    assert recs[0]["payload"]["object_id"] == sha and recs[0]["payload"]["fit_target_kind"] == "corpus_document_fit"
    assert t2.validated_complete(tmp_path, CELL, FP) is not None                   # nothing retrained
    # terminal under the stated campaign: a second replay is a no-op
    assert t2.ensure_registry_emission(final, si_id=SI, arm=ARM) == ev2 and len(list(R.iter_record_files(str(new_dir)))) == 2
    # emission OFF and no campaign stated: the replay neither raises nor re-emits (the evidence verifies by id + contract alone)
    monkeypatch.delenv("REGISTRY_CAMPAIGN", raising=False)
    monkeypatch.setenv(emit.DISABLED_ENV, "1")
    assert t2.ensure_registry_emission(final, si_id=SI, arm=ARM) == ev2


def test_a_cross_campaign_replay_restates_the_fits_provenance_never_the_replaying_jobs(tmp_path, monkeypatch, capsys):
    """A kept-cell replay from another job (another checkout, SLURM_JOB_ID, host) used to restate THAT job's
    context as the fit's. Now the replay restates the fit's provenance from the object's prior record; a fit_job_provenance.json sidecar
    (the fit's own RESULT + attempt record) wins over both, and the row carries the correction's sources."""
    from sisteer.registry import emit, hooks, records as R

    monkeypatch.undo()                                    # the REAL hook
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    spool = tmp_path / "spool"
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))
    monkeypatch.setenv("REGISTRY_CAMPAIGN", "two_si")
    monkeypatch.setenv("SLURM_JOB_ID", "821364")
    monkeypatch.setenv("SISTEER_REGISTRY_CLUSTER_LABEL", "cluster_a")      # a cluster substrate needs its site label on every record
    monkeypatch.setattr(emit, "producer_block", lambda script, code_sha=None, session=None: {"script": script, "code_sha": "a" * 40, "host": "gpu-23", "substrate": "cluster_a", "session": None})
    final = _publish(tmp_path, _temp_cell(tmp_path, blob=_general_lora_blob()))
    sha = json.loads((final / "COMPLETE").read_text())["object_sha256"]
    first = [rec for f in R.iter_record_files(str(spool / "two_si" / f"object__{sha[:16]}")) for _, rec, _ in R.read_record_file(f)]
    o1 = next(r["payload"] for r in first if r["record_type"] == "object" and r["payload"]["object_id"] == sha)
    a1 = next(r["payload"] for r in first if r["record_type"] == "artifact" and r["payload"]["artifact_id"] == sha)
    assert (o1["fit_code_sha"], o1["fit_job_id"], o1["fit_node_hostname"]) == ("a" * 40, "821364", "gpu-23")
    # the replay from ANOTHER job's context, under the newly stated campaign
    monkeypatch.setenv("REGISTRY_CAMPAIGN", "sdfbase_full22")
    monkeypatch.setenv("SLURM_JOB_ID", "828839")
    monkeypatch.setattr(emit, "producer_block", lambda script, code_sha=None, session=None: {"script": script, "code_sha": "d" * 40, "host": "gpu-15", "substrate": "cluster_a", "session": None})
    capsys.readouterr()
    ev = t2.ensure_registry_emission(final, si_id=SI, arm=ARM)
    assert ev["status"] == "emitted" and ev["campaign"] == "sdfbase_full22" and "fit provenance restated from its prior record" in capsys.readouterr().out
    second = [rec for f in R.iter_record_files(str(spool / "sdfbase_full22" / f"object__{sha[:16]}")) for _, rec, _ in R.read_record_file(f)]
    o2 = next(r["payload"] for r in second if r["record_type"] == "object" and r["payload"]["object_id"] == sha)
    a2 = next(r["payload"] for r in second if r["record_type"] == "artifact" and r["payload"]["artifact_id"] == sha)
    assert (o2["fit_code_sha"], o2["fit_job_id"], o2["fit_node_hostname"], o2["fit_substrate"]) == ("a" * 40, "821364", "gpu-23", "cluster_a")   # the fit's, not the replay's
    assert a2["created_utc"] == a1["created_utc"] and o2["extra"]["fit_provenance"]["restated_from"].endswith("-object.jsonl#object")
    assert next(r for r in second if r["record_type"] == "object")["producer"]["code_sha"] == "d" * 40          # the envelope still names the emitter
    # the sidecar (the fit's own RESULT + attempt record) wins over the prior record and the row explains itself
    (final / hooks.FIT_PROVENANCE_SIDECAR).write_text(json.dumps({
        "fit_code_sha": "b" * 40, "fit_job_id": "821352", "fit_node_hostname": "gpu-29", "fit_substrate": "cluster_a",
        "artifact_created_utc": a1["created_utc"],
        "sources": {"result": str(final / "RESULT"), "attempt_record": "attempts.jsonl#control_relu@cluster_a@2"},
        "note": "fit provenance corrected from RESULT + attempt record; previous values were the D758 re-emission's context",
        "written_by": "tests", "written_utc": "2026-10-02T06:00:00Z"}))
    import shutil
    shutil.rmtree(spool / "sdfbase_full22")                # nothing for it in that campaign's spool: the whole hook re-emits (no spool-complete shortcut)
    (final / t2.REGISTRY_EMISSION_NAME).unlink()
    t2.ensure_registry_emission(final, si_id=SI, arm=ARM)
    third = [rec for f in R.iter_record_files(str(spool / "sdfbase_full22" / f"object__{sha[:16]}")) for _, rec, _ in R.read_record_file(f)]
    o3 = [r["payload"] for r in third if r["record_type"] == "object" and r["payload"]["object_id"] == sha][-1]
    assert (o3["fit_code_sha"], o3["fit_job_id"], o3["fit_node_hostname"]) == ("b" * 40, "821352", "gpu-29")
    assert o3["extra"]["fit_provenance"]["sources"]["attempt_record"].endswith("@2") and "corrected from RESULT" in o3["extra"]["fit_provenance"]["note"]
    # dry run through the trainer's own call site: the exact records, nothing written, evidence untouched
    before = (final / t2.REGISTRY_EMISSION_NAME).read_text()
    recs = t2._registry_publish(final, si_id=SI, arm=ARM, dry_run=True)
    assert [r for r, _ in recs][:2] == ["object", "artifact"] and recs[0][1]["fit_job_id"] == "821352"
    assert (final / t2.REGISTRY_EMISSION_NAME).read_text() == before and len(third) == len([rec for f in R.iter_record_files(str(spool / "sdfbase_full22" / f"object__{sha[:16]}")) for _, rec, _ in R.read_record_file(f)])
