"""The two-SI serving surface honours each artifact's FIT leg (audit
2026-08-27).

The hazard this pins against lives one seam down and is itself pinned by
tests/test_class_c_arms.py::test_default_build_silently_unmasks_a_response_fit_artifact:
``_set_response_positions``'s explicit "all" ACTIVELY resets a response-fit
object (the fit x serve 2x2's deliberate seam), so any default-"all" caller
silently unmasks a decode-only artifact — and the failure is invisible,
because an all-position object serving under a decode-only label produces
perfectly well-formed output. ``build_defense_cell`` WAS such a caller: the
two-SI attack path (two_si_run.cmd_generate) served every Class C fit
(psr_kl_all / psr_mse_all / lora_r1_all) all-position while the fit driver's
provenance comment promised a fit==serve assertion that was never
implemented (train_two_si_objects.py stamps ``positions_at_fit`` but nothing
read it back).

This file pins the fix, at the cell-keyed surface:

- ``positions="fit"`` (the new default) reads the artifact's persisted
  ``positions_at_fit`` sidecar record and serves at THAT leg — a
  response-fit object keeps its decode-only mask (active exactly at
  idx >= prompt_len - H, so the forward producing the FIRST generated token
  is steered), an all-fit object serves all-position, baselines are
  untouched;
- the fit==serve guard: an EXPLICIT leg that mismatches the record RAISES
  (never warns) in both directions;
- a missing sidecar RAISES for a Class A cell (the record is part of the
  publish contract) and resolves to "all" for Class B (fit_class_b publishes
  no sidecar by design; every Class B fit is all-position closed-form).

CPU-only: the tiny_llama fixture; no network, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from sisteer.eval.defense_cell import DefenseCell
from sisteer.eval.jailbreak import (TRAIN_BUDGET_META, build_defense_cell,
                                    cell_positions_at_fit)
from sisteer.extraction.diffmean import DiffMeanObject
from sisteer.methods import object_artifact_name

SI_ID = "safety_refusal"
H = 2  # synthetic assistant-header allowance (test_class_c_arms convention)


def _with_header(monkeypatch, h: int = H):
    """The serve seam derives H from the tokenizer; the tiny fixture has no
    chat template, so patch the derivation (the test_positions_steering
    convention — never mutate the shared fixture's tokenizer)."""
    monkeypatch.setattr(
        "sisteer.eval.jailbreak.response_header_len", lambda tok: h)


def _published_cell(model, tmp_path, *, at_fit, sidecar=True, class_="A",
                    arm="diffmean"):
    """A DefenseCell over a published diffmean artifact, the way the fit
    driver lays cells out: ``<store>/<cell_id>/<artifact>`` with the
    ``train_budget_meta.json`` provenance sidecar beside it. diffmean is a
    real Class C member whose torch blob persists AND restores the fit leg
    (the same object the hazard test uses), so the cell exercises the
    production load path end to end."""
    si = SI_ID if class_ == "A" else None
    cell_id = f"{si}__{arm}" if si else arm
    cell_dir = tmp_path / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    obj = DiffMeanObject(model, {1: torch.randn(model.config.hidden_size)})
    if at_fit == "response":
        obj.set_fit_positions("response", H)
    path = cell_dir / object_artifact_name(arm)
    obj.save(path)
    if sidecar:
        (cell_dir / TRAIN_BUDGET_META).write_text(json.dumps(
            {"cell_id": cell_id, "arm": arm, "positions_at_fit": at_fit}))
    return DefenseCell(id=cell_id, class_=class_, arm=arm, si_id=si,
                       object_path=path)


# --------------------------------------------------------------------------- #
# serve auto-matches fit (the "fit" default)                                    #
# --------------------------------------------------------------------------- #


def test_default_serve_honours_a_response_fit_artifact(
        tiny_llama, tmp_path, monkeypatch):
    """The exact contamination path, closed: a response-fit non-token object
    served through the two-SI surface with NO positions argument keeps its
    decode-only mask instead of being silently reset to all-positions."""
    _with_header(monkeypatch)
    model, tok = tiny_llama
    cell = _published_cell(model, tmp_path, at_fit="response")

    assert cell_positions_at_fit(cell) == "response"
    d = build_defense_cell(model, tok, cell)  # DEFAULT build — the old trap
    assert d._obj.fit_positions == "response", (
        "the default two-SI build unmasked a response-fit object — the "
        "fit==serve auto-match regressed back to the pre-audit 'all' default")
    assert d._obj.fit_response_header_len == H

    # The served object's live mask is the decode-only leg: active exactly
    # at global idx >= prompt_len - H — the trailing assistant-header
    # positions AND every generated position, so the forward producing the
    # FIRST generated token is steered (for a refusal defense, the decisive
    # one). prompt_len on the handle is already the boundary (the serve path
    # sets it to response_prompt_len = prompt width - H).
    prompt_width, seq_len = 10, 16
    boundary = prompt_width - H
    with d._obj.apply(model, scale=1.0) as handle:
        handle.reset(prompt_len=boundary)
        mask = handle._position_mask(1, seq_len, torch.device("cpu"))
        idx = torch.arange(seq_len)
        expect = (idx >= boundary).float()[None, :]
        assert torch.equal(mask, expect), (
            f"decode-only serve mask wrong: got {mask.tolist()}, want "
            f"active exactly at idx >= {boundary} (= prompt_len - H)")


def test_default_serve_keeps_an_all_fit_artifact_all_position(
        tiny_llama, tmp_path, monkeypatch):
    """The other leg of the auto-match: an all-token fit (token_faithful's
    leg — here the same blob kind recorded positions_at_fit='all') serves
    all-position, unchanged."""
    _with_header(monkeypatch)
    model, tok = tiny_llama
    cell = _published_cell(model, tmp_path, at_fit="all")

    assert cell_positions_at_fit(cell) == "all"
    d = build_defense_cell(model, tok, cell)
    assert d._obj.fit_positions == "all"
    assert d._obj.fit_response_header_len is None


def test_baseline_cells_take_the_fit_default_as_all(tiny_llama):
    """Baselines have no artifact and no fit leg: the 'fit' default resolves
    to 'all' and builds exactly what the old default built."""
    model, tok = tiny_llama
    cell = DefenseCell(id="none", class_="baseline", arm=None, si_id=None)
    d = build_defense_cell(model, tok, cell)
    assert d.mode == "none" and d.obj is None
    # an explicit non-"all" still hits build_defense's loud refusal
    with pytest.raises(ValueError, match="no intervention to mask"):
        build_defense_cell(model, tok, cell, positions="response")


# --------------------------------------------------------------------------- #
# the fit==serve guard                                                          #
# --------------------------------------------------------------------------- #


def test_guard_raises_on_explicit_all_over_a_response_fit(
        tiny_llama, tmp_path, monkeypatch):
    """The mismatch the audit found, now refused: serving a response-fit
    object all-position through the cell surface RAISES instead of silently
    unmasking (what the pre-guard default did)."""
    _with_header(monkeypatch)
    model, tok = tiny_llama
    cell = _published_cell(model, tmp_path, at_fit="response")
    with pytest.raises(ValueError, match="positions_at_fit 'response'"):
        build_defense_cell(model, tok, cell, positions="all")


def test_guard_raises_on_explicit_response_over_an_all_fit(
        tiny_llama, tmp_path, monkeypatch):
    """Both directions: masking an all-token fit at serve is the OTHER
    cross-phase cell, equally a mislabel under this surface."""
    _with_header(monkeypatch)
    model, tok = tiny_llama
    cell = _published_cell(model, tmp_path, at_fit="all")
    with pytest.raises(ValueError, match="positions_at_fit 'all'"):
        build_defense_cell(model, tok, cell, positions="response")


def test_explicit_leg_matching_the_record_serves(
        tiny_llama, tmp_path, monkeypatch):
    """An explicit leg that MATCHES the record is not a mismatch — the guard
    asserts, it does not forbid stating the leg."""
    _with_header(monkeypatch)
    model, tok = tiny_llama
    cell = _published_cell(model, tmp_path, at_fit="response")
    d = build_defense_cell(model, tok, cell, positions="response")
    assert d._obj.fit_positions == "response"
    assert d._obj.fit_response_header_len == H


# --------------------------------------------------------------------------- #
# the provenance record itself                                                  #
# --------------------------------------------------------------------------- #


def test_missing_sidecar_raises_for_class_a(tiny_llama, tmp_path, monkeypatch):
    """A Class A cell without its train_budget_meta.json is an incomplete
    publish — the leg is never guessed."""
    _with_header(monkeypatch)
    model, tok = tiny_llama
    cell = _published_cell(model, tmp_path, at_fit="response", sidecar=False)
    with pytest.raises(ValueError, match=TRAIN_BUDGET_META):
        build_defense_cell(model, tok, cell)


def test_missing_sidecar_resolves_all_for_class_b(
        tiny_llama, tmp_path, monkeypatch):
    """fit_class_b publishes no sidecar by design; every Class B fit is an
    all-position closed form, so the missing record resolves to 'all' and
    the cell serves."""
    _with_header(monkeypatch)
    model, tok = tiny_llama
    cell = _published_cell(model, tmp_path, at_fit="all", sidecar=False,
                           class_="B")
    assert cell_positions_at_fit(cell) == "all"
    d = build_defense_cell(model, tok, cell)
    assert d._obj.fit_positions == "all"


def test_unreadable_or_alien_sidecar_refuses(tiny_llama, tmp_path,
                                             monkeypatch):
    """The record is authoritative: a torn json or an unknown leg value
    refuses rather than serving around it."""
    _with_header(monkeypatch)
    model, tok = tiny_llama
    cell = _published_cell(model, tmp_path, at_fit="response")
    sidecar = cell.object_path.parent / TRAIN_BUDGET_META

    sidecar.write_text("{not json")
    with pytest.raises(ValueError, match="unreadable"):
        cell_positions_at_fit(cell)

    sidecar.write_text(json.dumps({"positions_at_fit": "generated"}))
    with pytest.raises(ValueError, match="'generated'"):
        cell_positions_at_fit(cell)

    sidecar.write_text(json.dumps({"cell_id": cell.id}))  # key absent
    with pytest.raises(ValueError, match="None"):
        cell_positions_at_fit(cell)
