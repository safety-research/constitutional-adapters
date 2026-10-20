"""Shared artifact-identity validation for staged cell dirs (fix item 4).

ONE consumer-level verdict — ``train_budget.staged_cell_problems`` /
``assert_staged_cell`` — runs BEFORE any model load at every consumer:
the explicit-cell registry (``defense_cell.registry_from_explicit_cells``),
``cell_controls_run`` generate, ``scripts/sdfbase/base_mmlu.py`` and
``scripts/sdfbase/probe_score.py``. It checks: sidecar cell_id matches the
requested cell (declared legacy alias accepted), object sha256 matches the
sidecar, budget kind valid (``document_budget_problems`` for
document_tokens, with the per-consumer sample-manifest strictness knob),
and a PRESENT COMPLETE manifest is coherent. A planted wrong-cell object or
sidecar must refuse at EVERY consumer.

CPU-only, synthetic bytes only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "scripts" / "sdfbase"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import base_mmlu  # noqa: E402
import cell_controls_run  # noqa: E402
import probe_score as ps  # noqa: E402
from sisteer.eval.defense_cell import (  # noqa: E402
    registry_from_explicit_cells,
    sha256_file,
)
from sisteer.eval.train_budget import (  # noqa: E402
    assert_staged_cell,
    staged_cell_problems,
)
from sisteer.methods import object_artifact_name  # noqa: E402

CELL = "constv2-mix-1k__lora_r4_ce"


def _doc_meta(cell_id: str, obj: Path, over: dict | None = None) -> dict:
    meta = {"cell_id": cell_id, "object_sha256": sha256_file(obj),
            "training_budget_kind": "document_tokens",
            "sequence_max_len": 256, "completion_tokens": None,
            "sample_manifest_sha": "ab" * 32, "max_len": 256}
    meta.update(over or {})
    return meta


def _stage(root: Path, cell_id: str, *, kind: str = "doc", sidecar=True,
           meta_over: dict | None = None, complete: bool | dict = False,
           dirname: str | None = None) -> Path:
    """A synthetic staged cell dir; returns the object path."""
    d = Path(root) / (dirname or cell_id)
    d.mkdir(parents=True, exist_ok=True)
    obj = d / object_artifact_name(cell_id.split("__", 1)[-1]
                                   if "__" in cell_id else cell_id)
    obj.write_text(f"synthetic object bytes for {cell_id}")
    if sidecar:
        if kind == "doc":
            meta = _doc_meta(cell_id, obj, meta_over)
        else:
            meta = {"cell_id": cell_id, "completion_tokens": 512,
                    "object_sha256": sha256_file(obj), "max_len": 256}
            meta.update(meta_over or {})
        (d / "train_budget_meta.json").write_text(json.dumps(meta))
    if complete:
        marker = {"cell_id": cell_id, "object_file": obj.name,
                  "object_sha256": sha256_file(obj),
                  "files": {obj.name: sha256_file(obj)},
                  "fit_fingerprint": "f" * 64}
        if isinstance(complete, dict):
            marker.update(complete)
        (d / "COMPLETE").write_text(json.dumps(marker))
    return obj


# --------------------------------------------------------------------------- #
# the shared verdict                                                            #
# --------------------------------------------------------------------------- #


def test_valid_cells_pass_both_kinds(tmp_path):
    obj_doc = _stage(tmp_path / "a", CELL, kind="doc", complete=True)
    assert staged_cell_problems(obj_doc, CELL) == []
    obj_leg = _stage(tmp_path / "b", "safety_refusal__vector", kind="legacy")
    assert staged_cell_problems(obj_leg, "safety_refusal__vector") == []


def test_wrong_cell_sidecar_refused(tmp_path):
    obj = _stage(tmp_path, CELL,
                 meta_over={"cell_id": "constv2-mix-1m__lora_r4_ce"})
    probs = staged_cell_problems(obj, CELL)
    assert any("wrong-cell" in p or "records cell_id" in p for p in probs)


def test_object_sha_mismatch_refused(tmp_path):
    obj = _stage(tmp_path, CELL)
    obj.write_text("DIFFERENT bytes than the sidecar hashed")
    probs = staged_cell_problems(obj, CELL)
    assert any("object_sha256 mismatch" in p for p in probs)


def test_budget_kind_and_document_schema_route_through_shared_verdict(
        tmp_path):
    # unknown kind refuses
    obj = _stage(tmp_path / "k", CELL,
                 meta_over={"training_budget_kind": "martian_tokens"})
    assert any("unknown training_budget_kind" in p
               for p in staged_cell_problems(obj, CELL))
    # document schema violation (a numeric completion cap) refuses
    obj2 = _stage(tmp_path / "c", CELL, meta_over={"completion_tokens": 512})
    assert any("completion cap" in p or "completion_tokens" in p
               for p in staged_cell_problems(obj2, CELL))


def test_sample_manifest_strictness_knob(tmp_path):
    obj = _stage(tmp_path, CELL, meta_over={"sample_manifest_sha": None})
    # resume-grade: a stated absence passes
    assert staged_cell_problems(obj, CELL) == []
    # serving-grade: it refuses
    assert any("sample_manifest_sha" in p for p in staged_cell_problems(
        obj, CELL, require_sample_manifest_sha=True))


def test_complete_manifest_coherence(tmp_path):
    # wrong-cell marker
    obj = _stage(tmp_path / "w", CELL,
                 complete={"cell_id": "constv2-adv-1k__lora_r4_ce"})
    assert any("COMPLETE marker names" in p
               for p in staged_cell_problems(obj, CELL))
    # manifested file drifted after publication
    obj2 = _stage(tmp_path / "d", CELL, complete=True)
    marker_path = obj2.parent / "COMPLETE"
    marker = json.loads(marker_path.read_text())
    marker["files"][obj2.name] = "0" * 64
    marker_path.write_text(json.dumps(marker))
    assert any("COMPLETE-manifested" in p
               for p in staged_cell_problems(obj2, CELL))
    # absent marker is NOT a problem (staged handovers ship object+sidecar)
    obj3 = _stage(tmp_path / "n", CELL, complete=False)
    assert staged_cell_problems(obj3, CELL) == []


def test_class_b_sidecar_exemption_and_legacy_alias(tmp_path):
    # Class B: no sidecar is fine ONLY with require_sidecar=False
    obj = _stage(tmp_path / "b", "antidote", sidecar=False)
    assert staged_cell_problems(obj, "antidote",
                                require_sidecar=False) == []
    assert any("sidecar" in p for p in staged_cell_problems(obj, "antidote"))
    # legacy alias: a lora_r4 cell whose sidecar records the declared
    # lora_r1_all spelling is the SAME cell (methods.LEGACY_ARM_ALIASES)
    cid = "si_constitution__lora_r4"
    obj2 = _stage(tmp_path / "l", cid, kind="legacy",
                  meta_over={"cell_id": "si_constitution__lora_r1_all"})
    assert staged_cell_problems(obj2, cid) == []


def test_assert_staged_cell_raises_systemexit(tmp_path):
    obj = _stage(tmp_path, CELL,
                 meta_over={"cell_id": "constv2-mix-1m__lora_r4_ce"})
    with pytest.raises(SystemExit, match="identity validation"):
        assert_staged_cell(obj, CELL, consumer="test")


def test_assert_object_identity_dispatches_on_the_sidecar(tmp_path):
    """The ONE consumer entry point (round 2): a budget sidecar routes to
    the staged gate, a derived sidecar to the derived-object verdict, both
    at once is refused rather than guessed, and a bare dir goes down the
    staged path whose own 'sidecar missing' verdict names the fix. (The
    derived branch's positive cases ride tests/test_lora_subtract.py, where
    a real derived artifact is built.)"""
    from sisteer.eval.train_budget import (assert_object_identity,
                                           object_identity_kind)

    obj = _stage(tmp_path / "s", CELL, complete=True)
    assert object_identity_kind(obj) == "staged"
    assert assert_object_identity(obj, CELL, consumer="t") == "staged"
    (obj.parent / "derived_meta.json").write_text("{}")
    assert object_identity_kind(obj) == "ambiguous"
    # the wording widened 2026-09-06 when fit_provenance.json became the
    # third identity sidecar ("both" -> "more than one"); the refusal is
    # the same, and any two of the three are equally ambiguous
    with pytest.raises(SystemExit, match="more than one identity sidecar"):
        assert_object_identity(obj, CELL, consumer="t")
    (obj.parent / "derived_meta.json").unlink()
    (obj.parent / "fit_provenance.json").write_text("{}")
    assert object_identity_kind(obj) == "ambiguous"
    with pytest.raises(SystemExit, match="more than one identity sidecar"):
        assert_object_identity(obj, CELL, consumer="t")
    (obj.parent / "fit_provenance.json").unlink()
    bare = tmp_path / "bare" / "object.pt"
    bare.parent.mkdir(parents=True)
    bare.write_text("scratch")
    assert object_identity_kind(bare) is None
    with pytest.raises(SystemExit, match="sidecar"):
        assert_object_identity(bare, CELL, consumer="t")


# --------------------------------------------------------------------------- #
# consumer seams: a planted wrong-cell artifact refuses at each                 #
# --------------------------------------------------------------------------- #

_WRONG = {"cell_id": "constv2-mix-1m__lora_r4_ce"}


def _plant(root: Path, cell_id: str, defect: str) -> None:
    if defect == "wrong-cell-sidecar":
        _stage(root, cell_id, meta_over=_WRONG)
    else:  # wrong-object bytes under a valid-looking sidecar
        obj = _stage(root, cell_id)
        obj.write_text("swapped object bytes")


@pytest.mark.parametrize("defect", ["wrong-cell-sidecar", "swapped-object"])
def test_registry_from_explicit_cells_refuses(tmp_path, defect,
                                              monkeypatch):
    import sisteer.data.instructions as instructions_mod

    si = "constv2-mix-1k"
    inst = instructions_mod.Instruction(
        id=si, category="corpus_cell", text="corpus cell", anti_text=None,
        checker="external")
    monkeypatch.setitem(instructions_mod.BY_ID, si, inst)
    ddir = tmp_path / "cells"
    _plant(ddir, CELL, defect)
    cfg = {"defenses_dir": str(ddir), "explicit_cells": [CELL]}
    with pytest.raises(ValueError, match="validation failed"):
        registry_from_explicit_cells(cfg, repo_root=tmp_path)


@pytest.mark.parametrize("defect", ["wrong-cell-sidecar", "swapped-object"])
def test_cell_controls_gate_refuses(tmp_path, defect):
    ddir = tmp_path / "cells"
    _plant(ddir, CELL, defect)
    obj = ddir / CELL / object_artifact_name("lora_r4_ce")
    cell = SimpleNamespace(id=CELL, class_="A", object_path=obj)
    with pytest.raises(SystemExit, match="cell-controls"):
        cell_controls_run._check_staged_cell(cell)


def test_cell_controls_gate_class_semantics(tmp_path):
    # baseline / not-yet-staged: skip (the loader's own error later)
    cell_controls_run._check_staged_cell(
        SimpleNamespace(id="none", class_="baseline", object_path=None))
    cell_controls_run._check_staged_cell(SimpleNamespace(
        id=CELL, class_="A", object_path=tmp_path / "missing" / "object.pt"))
    # Class B without a sidecar passes; serving strictness holds for A:
    # a doc cell with a stated-absence sample sha refuses HERE (never
    # serve judged controls from unverified sample provenance)
    obj_b = _stage(tmp_path / "b", "antidote", sidecar=False)
    cell_controls_run._check_staged_cell(
        SimpleNamespace(id="antidote", class_="B", object_path=obj_b))
    obj_a = _stage(tmp_path / "a", CELL,
                   meta_over={"sample_manifest_sha": None})
    with pytest.raises(SystemExit, match="sample_manifest_sha"):
        cell_controls_run._check_staged_cell(
            SimpleNamespace(id=CELL, class_="A", object_path=obj_a))


@pytest.mark.parametrize("defect", ["wrong-cell-sidecar", "swapped-object"])
def test_base_mmlu_refuses_before_load(tmp_path, defect):
    bank = tmp_path / "bank"
    _plant(bank / "cells_all", CELL, defect)
    with pytest.raises(SystemExit, match="base-mmlu"):
        base_mmlu.resolve_object_path(bank, CELL)


def test_base_mmlu_accepts_a_valid_cell(tmp_path):
    bank = tmp_path / "bank"
    obj = _stage(bank / "cells_all", CELL, complete=True)
    assert base_mmlu.resolve_object_path(bank, CELL) == obj


@pytest.mark.parametrize("defect", ["wrong-cell-sidecar", "swapped-object"])
def test_probe_score_refuses_before_load(tmp_path, defect):
    bank = tmp_path / "bank"
    _plant(bank, CELL, defect)
    obj = bank / CELL / object_artifact_name("lora_r4_ce")
    with pytest.raises(SystemExit, match="probe-score"):
        ps._validate_object_identity(obj, CELL)


def test_probe_score_self_consistency_without_cell_flag(tmp_path):
    # sidecar present: validated against its own recorded identity even
    # with no --cell (a swapped object still refuses)...
    obj = _stage(tmp_path / "s", CELL)
    ps._validate_object_identity(obj, None)  # valid: passes
    obj.write_text("swapped bytes")
    with pytest.raises(SystemExit, match="object_sha256"):
        ps._validate_object_identity(obj, None)
    # ...while a bare ad-hoc artifact (no sidecar) passes untouched
    bare = tmp_path / "adhoc" / "object.pt"
    bare.parent.mkdir(parents=True)
    bare.write_text("scratch object")
    ps._validate_object_identity(bare, None)


def test_cell_controls_gate_runs_before_the_model_loads():
    """Source-order pin: cmd_generate invokes the identity gate before
    load_model_and_tokenizer — a wrong-cell artifact must fail in
    milliseconds, never after a GPU allocation."""
    src = (REPO / "scripts" / "cell_controls_run.py").read_text()
    gen = src.split("def cmd_generate", 1)[1]
    assert gen.index("_check_staged_cell(") < gen.index(
        "load_model_and_tokenizer(")
