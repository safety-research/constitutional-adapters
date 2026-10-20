"""lora_r1_all cell registration + rank-general lora_all artifacts (2026-08-28).

Two defects pinned:

1. ``lora_r1_all`` has been a registered arm in ``methods.ARMS`` since the
   all-layer wave, but was never listed in ``defense_cell.CLASS_A_ARMS``
   (or ``configs/two_si.yaml class_a_arms``), so ``<si>__lora_r1_all`` was
   not a cell id: the scale-12 campaign's staged objects (fitted with
   ``arm=lora_r1_all``, target_layers 0..31) died at ``cell_controls_run
   --cell`` with a KeyError. NOT fixable by aliasing onto ``lora_r1`` —
   that is the SINGLE-layer rank-1 arm, a genuinely different method
   (train_budget's ``_ARM_TRAINER_SOURCES`` lists them separately).

2. The repo's ``MultiLayerLoRAObject`` was rank-1-only (A/B as [d]
   vectors), while the fit lane lands rank-k artifacts under the
   same "lora_all" kind (A/B as [d, k] matrices + ``rank``/``layers``
   metadata; the staged scale-12 objects are rank 4). A rank-4 blob
   loaded SILENTLY and then crashed with a broadcast error inside the
   o_proj hook at the first forward. Now: both shapes load and steer,
   and any self-inconsistent artifact refuses loudly at load.

CPU; the model tests use the conftest tiny_llama.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from sisteer.eval.defense_cell import CLASS_A_ARMS, by_id, registry_from_cfg
from sisteer.extraction.multilayer import MultiLayerLoRAObject
from sisteer.methods import ARMS, load_object, object_artifact_name

REPO = Path(__file__).resolve().parents[1]

#: The scale-12 campaign's six cells: three SIs x two arms. The LoRA arm
#: was registered as lora_r1_all and RENAMED to lora_r4 the same day
#: (2026-08-28, user instruction: the staged objects are rank 4, "all" is
#: implicit) — the canonical cell ids carry lora_r4; the staged
#: `<si>__lora_r1_all/` artifact dirs resolve through the legacy alias
#: (methods.LEGACY_ARM_ALIASES; pinned by tests/test_wave_vocabulary.py).
CAMPAIGN_SIS = ("safety_refusal", "si_constitution", "si_irrelevant")
CAMPAIGN_ARMS = ("psr_kl_all", "lora_r4")


def _cfg(tmp_path):
    cfg = yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())
    cfg["defenses_dir"] = str(tmp_path)  # absolute -> wins the resolution
    return cfg


def test_campaign_cell_ids_resolve_from_the_shipped_config(tmp_path):
    """All six cell ids the scale-12 campaign actually submits must
    resolve from configs/two_si.yaml, with the object path keyed on the
    cell id and the arm's registered artifact name."""
    cells = by_id(registry_from_cfg(_cfg(tmp_path), repo_root=REPO))
    for si in CAMPAIGN_SIS:
        for arm in CAMPAIGN_ARMS:
            cid = f"{si}__{arm}"
            assert cid in cells, (
                f"{cid} is not a registered cell id — the scale-12 "
                "campaign fails at --cell with a KeyError")
            cell = cells[cid]
            assert cell.arm == arm and cell.si_id == si
            assert cell.object_path == (
                tmp_path / cid / object_artifact_name(arm))
            assert cell.object_path.name == "object.pt"


def test_lora_r4_is_a_distinct_arm_not_an_alias_of_lora_r1(tmp_path):
    """lora_r1 (single-layer rank-1) and lora_r4 (all-layer, ex
    lora_r1_all) are both cells, with distinct ids and distinct artifact
    paths — aliasing one onto the other would run an all-layer adapter
    under the single-layer cell's identity with plausible-looking
    numbers. (lora_r1_all -> lora_r4 IS a recorded alias — one method,
    two spellings; lora_r1 is a different method entirely.)"""
    assert "lora_r1" in ARMS and "lora_r1_all" in ARMS and "lora_r4" in ARMS
    cells = by_id(registry_from_cfg(_cfg(tmp_path), repo_root=REPO))
    a = cells["safety_refusal__lora_r1"]
    b = cells["safety_refusal__lora_r4"]
    assert a.arm != b.arm and a.object_path != b.object_path
    # appended after every pre-existing arm (the psr_xfer_* transferred-PSR
    # arms were appended after it at the 2026-09-02 merge): every earlier
    # arm's arm-major array index unchanged
    assert CLASS_A_ARMS[-5:] == ("lora_r4",
                                 "psr_xfer_from_e2b", "psr_xfer_from_e2b_rw1",
                                 "psr_xfer_from_e4b", "psr_xfer_from_e4b_rw1")


# ---------------------------------------------------------------------------
# rank-general lora_all artifacts
# ---------------------------------------------------------------------------

D = 16  # tiny_llama hidden size (square o_proj)


def _rank4_blob(layers=(0, 1), rank=4, seed=3):
    g = torch.Generator().manual_seed(seed)
    return {
        "kind": "lora_all",
        "A": {str(l): torch.randn(D, rank, generator=g) for l in layers},
        "B": {str(l): torch.randn(D, rank, generator=g) * 0.5
              for l in layers},
        "rank": rank,
        "layers": list(layers),
        # the staged scale-12 objects are RESPONSE fits with H=4 baked in
        "positions": "response",
        "response_header_len": 4,
    }


def test_rank4_traintest_blob_loads_and_steers(tiny_llama, tmp_path):
    """The fit lane's schema (A/B [d, k], rank/layers metadata, response
    fit leg) loads, restores the fit mask, and actually steers — with the
    persisted response boundary: only the last H positions of a prefill
    move, earlier positions stay bit-identical."""
    model, tok = tiny_llama
    path = tmp_path / "object.pt"
    torch.save(_rank4_blob(), path)
    obj = load_object(path, model, tok)
    assert isinstance(obj, MultiLayerLoRAObject)
    assert obj.layers == [0, 1]
    assert obj.A[0].shape == (D, 4) and obj.B[0].shape == (D, 4)
    assert obj.fit_positions == "response"
    assert obj.fit_response_header_len == 4

    text = "The quick brown fox jumps over the lazy dog"
    ids = tok(text, return_tensors="pt")["input_ids"]
    assert ids.shape[1] > 4
    attn = torch.ones_like(ids)
    with torch.no_grad():
        base = model(input_ids=ids, attention_mask=attn).logits
        with obj.apply(model, scale=1.0):
            steered = model(input_ids=ids, attention_mask=attn).logits
    H = 4
    assert torch.equal(steered[:, :-H], base[:, :-H]), (
        "positions before the persisted response boundary moved")
    assert not torch.equal(steered[:, -H:], base[:, -H:]), (
        "a rank-4 artifact produced no effect — the adapter did not load")


def test_legacy_rank1_blob_roundtrips_bit_identically(tiny_llama, tmp_path):
    """The historical 1-D artifact form keeps its exact arithmetic: a
    save/load round trip reproduces the original object's steered forward
    bit-for-bit."""
    model, tok = tiny_llama
    g = torch.Generator().manual_seed(5)
    orig = MultiLayerLoRAObject(model, layers=[0, 1])
    for l in orig.layers:
        orig.B[l].data = torch.randn(D, generator=g)
    path = tmp_path / "object.pt"
    orig.save(path)
    loaded = load_object(path, model, tok)
    assert isinstance(loaded, MultiLayerLoRAObject)
    assert loaded.A[0].dim() == 1

    ids = tok("Salt and flour and water", return_tensors="pt")["input_ids"]
    attn = torch.ones_like(ids)
    with torch.no_grad():
        with orig.apply(model, scale=1.0):
            a = model(input_ids=ids, attention_mask=attn).logits
        with loaded.apply(model, scale=1.0):
            b = model(input_ids=ids, attention_mask=attn).logits
    assert torch.equal(a, b)


@pytest.mark.parametrize("mutate, match", [
    (lambda b: b["B"].__setitem__("0", torch.randn(D, 2)), "rank"),
    (lambda b: b.__setitem__("rank", 8), "internally inconsistent"),
    (lambda b: b.__setitem__("layers", [0]), "layers"),
    (lambda b: b["B"].__setitem__("0", torch.randn(D)), "pair"),
])
def test_self_inconsistent_artifacts_refuse_loudly(tiny_llama, tmp_path,
                                                   mutate, match):
    """A/B rank disagreement, a rank key contradicting the tensors, layer
    metadata not covering the tensors, and a mixed 1-D/2-D pair all
    refuse AT LOAD with the defect named — never a silent load that
    crashes with a broadcast error inside the o_proj hook later."""
    model, tok = tiny_llama
    blob = _rank4_blob()
    mutate(blob)
    path = tmp_path / "object.pt"
    torch.save(blob, path)
    with pytest.raises(ValueError, match=match):
        load_object(path, model, tok)
