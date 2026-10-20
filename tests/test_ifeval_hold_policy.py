"""ifeval-canonical-battery (P1): the frozen matrix's HOLD / DEFERRED policy is enforced at the planner and
at the container, not only in the dispatcher's queue.

``plan_configurations`` never read the matrix: explicitly selecting ``constv2-mix-full__lora_r4_ce`` on
``gemma4e4b-sdfbase-it`` or ``qwen35_4b-sdfbase-it`` — the two HELD rows (P41, a hard exclusion until the refit lands) —
planned and spawned through the direct launcher's ``--cells … --pass1 --pf`` path. Now ``contracts.ifeval_hold_problem``
reads the same row statuses the dispatcher's ``select_rows`` reads (``status`` / ``held_reason`` / ``deferred.record``)
from the frozen matrix: a held or deferred row refuses by name at plan time, and ``_run_configuration`` repeats the check
against the STAGED tree's matrix before any directory is made or standup started (REC ``IFEVAL_ROW_HELD``, a refusal the
dispatcher never retries). A cell the matrix has no row for (the shakedown's probe cell) is admitted.
"""
from __future__ import annotations

import json
import shutil

import pytest

from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import ifeval_contracts as IC

import test_tinycap_launcher as TL
from test_tinycap_launcher_port import GEMMA, L, QWEN, REPO, _plan  # noqa: F401

HELD_CELL = "constv2-mix-full__lora_r4_ce"
QWEN_IT = C.IFEVAL_SLUGS[("qwen35_4b", "it")]
MATRIX = json.loads((REPO / IC.IFEVAL_MATRIX_REL).read_text(encoding="utf-8"))


def test_the_contracts_read_the_dispatchers_statuses_off_the_frozen_matrix():
    assert C.IFEVAL_MATRIX_REL == IC.IFEVAL_MATRIX_REL == "configs/sdfbase/ifeval/matrix.json"
    assert C.load_ifeval_matrix(REPO)["rows"] == MATRIX["rows"]
    held = [(r["slug"], r["cell"]) for r in MATRIX["rows"] if r["status"] == C.IFEVAL_STATUS_HELD]
    assert held == [(GEMMA, HELD_CELL), (QWEN_IT, HELD_CELL)]                       # the two held rows, verbatim
    for slug, cell in held:
        status, reason = C.ifeval_matrix_row_status(MATRIX, slug=slug, cell=cell)
        assert status == C.IFEVAL_STATUS_HELD and "P41" in reason
        assert "HELD" in C.ifeval_hold_problem(MATRIX, slug=slug, cell=cell) and "P41" in C.ifeval_hold_problem(MATRIX, slug=slug, cell=cell)
    status, reason = C.ifeval_matrix_row_status(MATRIX, slug=GEMMA, cell="alphasteer")
    assert status == C.IFEVAL_STATUS_DEFERRED and "deferred/alphasteer__gemma4e4b.json" in reason
    assert "DEFERRED" in C.ifeval_hold_problem(MATRIX, slug=GEMMA, cell="alphasteer")
    assert C.ifeval_matrix_row_status(MATRIX, slug=QWEN, cell="none") == (C.IFEVAL_STATUS_LAUNCH, None)
    assert C.ifeval_hold_problem(MATRIX, slug=QWEN, cell="none") is None
    assert C.ifeval_matrix_row_status(MATRIX, slug=QWEN, cell="constv2-mix-full__psr_ce_all") == (None, None)   # excluded: no row
    assert C.ifeval_hold_problem(MATRIX, slug=QWEN, cell="constv2-mix-full__psr_ce_all") is None
    assert "not one of" in C.ifeval_hold_problem({"rows": [{"slug": QWEN, "cell": "none", "status": "paused"}]}, slug=QWEN, cell="none")
    with pytest.raises(ValueError, match="keyed once per slug"):
        C.ifeval_matrix_row_status({"rows": [{"slug": QWEN, "cell": "none", "status": "launch"}] * 2}, slug=QWEN, cell="none")
    with pytest.raises(C.IfevalMatrixUnreadable, match="cannot be read"):
        C.load_ifeval_matrix("/nowhere")
    assert C.REC_IFEVAL_ROW_HELD in C.REFUSAL_REC_NAMES and C.is_refusal_status(C.REC_IFEVAL_ROW_HELD)
    # the planner and the container read EVERY registered document, merged — the frozen matrix's 40 rows + the
    # seed sibling's 30; a (slug, cell) keyed twice refuses; a missing registered document refuses by its name
    merged = C.load_ifeval_matrices(REPO)
    assert C.IFEVAL_MATRIX_DOCUMENTS == (C.IFEVAL_MATRIX_REL, "configs/sdfbase/ifeval/matrix_qwen_seedrep.json")
    assert merged["documents"] == list(C.IFEVAL_MATRIX_DOCUMENTS) and len(merged["rows"]) == 40 + 30
    assert C.ifeval_matrix_row_status(merged, slug=QWEN_IT, cell=HELD_CELL)[0] == C.IFEVAL_STATUS_HELD
    assert C.ifeval_matrix_row_status(merged, slug="qwen35_4b-sdfbase-s1", cell="arditi_allpos") == (C.IFEVAL_STATUS_LAUNCH, None)
    assert C.ifeval_hold_problem(merged, slug="qwen35_4b-sdfbase-s2p", cell="none") is None       # no own none: no row, admitted
    with pytest.raises(C.IfevalMatrixUnreadable, match="matrix_qwen_seedrep.json.missing cannot be read"):
        C.load_ifeval_matrices(REPO, documents=(C.IFEVAL_MATRIX_REL, "configs/sdfbase/ifeval/matrix_qwen_seedrep.json.missing"))
    with pytest.raises(ValueError, match="both carry a row"):          # the same rows under a second spelling: keyed twice
        C.load_ifeval_matrices(REPO, documents=(C.IFEVAL_MATRIX_REL, C.IFEVAL_MATRIX_QWEN_SEEDREP_REL,
                                                "configs/sdfbase/ifeval/../ifeval/matrix_qwen_seedrep.json"))


def test_the_planner_refuses_both_held_rows_and_a_deferred_row_and_stamps_the_launch_rows(L):
    # the reproduction: the two HELD constitution rows, selected explicitly — refused BY NAME at plan time
    for slug in (GEMMA, QWEN_IT):
        with pytest.raises(ValueError, match="HELD") as e:
            _plan(L, slug, cell=HELD_CELL)
        assert "P41" in str(e.value) and slug in str(e.value) and HELD_CELL in str(e.value)
    # a deferred design slot through the same door
    with pytest.raises(ValueError, match="DEFERRED"):
        _plan(L, GEMMA, cell="alphasteer")
    # the direct launcher path `plan_for` (what `main --cells … --pass1 --pf` calls) refuses identically
    with pytest.raises(ValueError, match="HELD"):
        L.plan_for(cells=[HELD_CELL], modes=[C.THINKING_OFF], controls=[C.CONTROL_IFEVAL], model=GEMMA,
                   sampling_mode=C.SAMPLING_VENDOR_PRESET, samples=C.IFEVAL_SAMPLES_PER_ITEM,
                   reduction_mode=C.REDUCTION_MODE_STACKED, pass_no=1)
    # a launch row plans and carries the matrix's verdict; a cell the matrix has no row for is admitted with None
    (cfg,) = _plan(L, QWEN, cell="none")
    assert cfg["matrix_status"] == C.IFEVAL_STATUS_LAUNCH
    (probe,) = _plan(L, QWEN, cell="constv2-mix-full__psr_ce_all")
    assert probe["matrix_status"] is None
    # the same cell on the it view of a model whose it-view row is a LAUNCH row plans (the hold is (slug, cell)-scoped)
    launch_lora = next(r for r in MATRIX["rows"] if r["status"] == "launch" and r["fitted"] and r["slug"] == QWEN)
    (ok,) = _plan(L, QWEN, cell=launch_lora["cell"])
    assert ok["matrix_status"] == C.IFEVAL_STATUS_LAUNCH
    # an injected matrix decides: the same held row under a matrix that launches it plans (the tests' bypass)
    relaxed = {"rows": [{**r, "status": "launch", "held_reason": None} if r["status"] == "held" else r for r in MATRIX["rows"]]}
    (bypassed,) = _plan(L, GEMMA, cell=HELD_CELL, ifeval_matrix=relaxed)
    assert bypassed["matrix_status"] == C.IFEVAL_STATUS_LAUNCH
    # a non-IFEval configuration never reads the matrix
    L.plan_configurations(cells=["none"], modes=[C.THINKING_OFF], controls=["gsm8k"], model=QWEN, ifeval_matrix={"rows": []})


def test_the_container_refuses_a_held_row_against_the_staged_trees_matrix_before_any_directory_or_standup(L, tmp_path, monkeypatch):
    # a hand-built configuration for the held row (the planner bypassed through an injected relaxed matrix)
    relaxed = {"rows": [{**r, "status": "launch", "held_reason": None} if r["status"] == "held" else r for r in MATRIX["rows"]]}
    out = tmp_path / "out"
    (cfg,) = _plan(L, GEMMA, cell=HELD_CELL, ifeval_matrix=relaxed, out_dir=str(out))
    repos = tmp_path / "repos"
    tree = TL._tree(repos, "f6cf462")
    dst = tree / IC.IFEVAL_MATRIX_REL
    dst.parent.mkdir(parents=True, exist_ok=True)
    for rel in C.IFEVAL_MATRIX_DOCUMENTS:                            # the STAGED tree carries EVERY registered document
        shutil.copy(REPO / rel, tree / rel)
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    monkeypatch.setenv(C.LENS_STACKED_REDUCTIONS_ENV, C.REDUCTION_MODE_STACKED)
    monkeypatch.setattr(L, "_standup_cmd", lambda *a, **k: (_ for _ in ()).throw(AssertionError("standup composed for a HELD row")))
    monkeypatch.setattr(L, "served_geometry", lambda *a, **k: (_ for _ in ()).throw(AssertionError("geometry resolved for a HELD row")))
    rec = L._run_configuration(cfg, "f6cf462", controls=[C.CONTROL_IFEVAL], samples=2, reduction_mode=C.REDUCTION_MODE_STACKED)
    assert rec["status"] == C.REC_IFEVAL_ROW_HELD and C.is_refusal_status(rec["status"]), rec
    assert "HELD" in rec["error"] and "P41" in rec["error"] and rec["matrix_status"] is None
    assert not out.exists(), "the refusal comes before any directory is made"
    # the staged tree WITHOUT the matrix refuses too (never admitted around a missing document)
    dst.unlink()
    rec2 = L._run_configuration(cfg, "f6cf462", controls=[C.CONTROL_IFEVAL], samples=2, reduction_mode=C.REDUCTION_MODE_STACKED)
    assert rec2["status"] == C.REC_IFEVAL_ROW_HELD and "cannot be read" in rec2["error"] and not out.exists()
    # ... and a tree carrying the frozen matrix but NOT the registered seed sibling refuses the same way (a document
    # nobody registers is a hold nobody enforces — and a registered one that is missing is never admitted around)
    shutil.copy(REPO / IC.IFEVAL_MATRIX_REL, dst)
    (tree / C.IFEVAL_MATRIX_QWEN_SEEDREP_REL).unlink()
    rec3 = L._run_configuration(cfg, "f6cf462", controls=[C.CONTROL_IFEVAL], samples=2, reduction_mode=C.REDUCTION_MODE_STACKED)
    assert rec3["status"] == C.REC_IFEVAL_ROW_HELD and "cannot be read" in rec3["error"] and "matrix_qwen_seedrep.json" in rec3["error"]
    # source order: the container's hold door precedes the ladder admission, the out_dir mkdir and the standup
    src = TL.LAUNCHER.read_text(encoding="utf-8")
    i_run = src.index("def _run_configuration(")
    i_hold = src.index("problem = C.ifeval_hold_problem(matrix_doc, slug=row.slug, cell=cfg[\"cell\"])", i_run)
    assert i_hold < src.index("out_dir = Path(cfg[\"out_dir\"])", i_run) < src.index("standup_argv = _standup_cmd(", i_run)
