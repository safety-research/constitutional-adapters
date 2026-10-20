"""SteeringVector.load tolerates the .tmp-rename sidecar misplacement.

train_two_si_objects._save_object saves via a temp path then renames the DATA:
``obj.save(object.safetensors.tmp); tmp.replace(object.safetensors)``. Because
``SteeringVector.save`` writes the sidecar at ``save_path.with_suffix('.json')``
and ``with_suffix`` strips ``.tmp`` (not ``.safetensors``), the sidecar lands at
``object.safetensors.json`` and the rename never moves it — so the canonical
``object.json`` is absent. The loader must find the misplaced sidecar (the same
historical misplacement the stage-3 provenance reader already tolerates), or a
train-lane vector/bidpo artifact crashes at hydration with FileNotFoundError:
object.json (the stage-3 wave smoke's exact failure, 2026-08-24).
"""

from __future__ import annotations

import json

import pytest
import torch

from sisteer.vectors import SteeringVector


def _sv(d=8):
    return SteeringVector(vectors={1: torch.randn(d)}, method="distill_soft",
                          model_name="NousResearch/Meta-Llama-3.1-8B-Instruct",
                          instruction_id="safety_refusal",
                          metadata={"note": "roundtrip"})


def test_load_finds_canonical_sidecar(tmp_path):
    # a DIRECT save (fit_class_b path) -> object.json, loads as before
    sv = _sv()
    path = tmp_path / "object.safetensors"
    sv.save(path)
    assert (tmp_path / "object.json").exists()
    got = SteeringVector.load(path)
    assert got.method == sv.method and got.model_name == sv.model_name
    assert set(got.vectors) == {1}


def test_load_tolerates_tmp_rename_misplacement(tmp_path):
    # reproduce the train-lane save EXACTLY: save to <name>.tmp, rename DATA
    sv = _sv()
    final = tmp_path / "object.safetensors"
    tmp = final.with_suffix(final.suffix + ".tmp")   # object.safetensors.tmp
    sv.save(tmp)
    tmp.replace(final)                               # data moves; sidecar does not
    # the misplacement is real: canonical name absent, misplaced name present
    assert not (tmp_path / "object.json").exists()
    assert (tmp_path / "object.safetensors.json").exists()
    # load must still succeed, reading the misplaced sidecar
    got = SteeringVector.load(final)
    assert got.method == sv.method
    assert got.model_name == "NousResearch/Meta-Llama-3.1-8B-Instruct"
    assert got.instruction_id == "safety_refusal"
    assert set(got.vectors) == {1}


def test_load_prefers_canonical_when_both_present(tmp_path):
    # if both names exist, the canonical <stem>.json wins (a direct re-save
    # beside a stale misplaced sidecar must not be shadowed by the stale one)
    sv = _sv()
    final = tmp_path / "object.safetensors"
    sv.save(final)                                    # writes object.json
    # plant a DIFFERENT (stale) misplaced sidecar
    (tmp_path / "object.safetensors.json").write_text(json.dumps(
        {"method": "STALE", "model_name": "x", "instruction_id": "y",
         "layers": [1], "norms": {}, "metadata": {}}))
    got = SteeringVector.load(final)
    assert got.method == "distill_soft"               # canonical, not STALE


def test_load_raises_when_no_sidecar_at_all(tmp_path):
    sv = _sv()
    final = tmp_path / "object.safetensors"
    sv.save(final)
    (tmp_path / "object.json").unlink()               # remove the only sidecar
    with pytest.raises(FileNotFoundError):
        SteeringVector.load(final)
