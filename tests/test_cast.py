"""CAST implementation (arXiv 2409.05907): PCA direction extraction + sign
fix, the rank-1 tanh-projection condition similarity and its INVERTED fire
naming, the F1 grid search, the conditional-gate object protocol
(latched-once, scale-0 bit-exact no-op, save/load round trip), the trainer
smoke on both the explicit-condition and the recorded pairs-fallback paths,
and the registry-removal wiring. cast left methods.ARMS ENTIRELY on
2026-08-20: nothing trains or enumerates it, but the
implementation and — load-bearing for a peer lane's serving/kind-coverage
canary — the load_object hydration of already-fitted artifacts survive.
CPU-only, tiny-llama fixture.
"""

import pytest
import torch

from sisteer.data.instructions import BY_ID
from sisteer.data.pairs import Pair
from sisteer.extraction.cast import (
    CastObject,
    _fires,
    condition_similarity,
    pca_direction,
    search_condition_point,
    train_cast,
)
from sisteer.methods import ARMS, _KIND_LOADERS, load_object
from sisteer.models import hidden_size, num_layers

# Lenient calibration so the base-scale sweep is deterministic on the random
# tiny model (its generations' coherence is meaningless): never trip the
# degeneration guard, so base_scale = last_rung/2 (mirrors test_alphasteer).
_CALIB_KW = dict(scale_ladder=(0.5, 1.0), rep_threshold=0.0,
                 ppl_guard_factor=1e9, calib_prompts=2,
                 calib_max_new_tokens=4, batch_size=2)


def _pairs():
    return [
        Pair(prompt="Say something nice.", pos="Bonjour le monde",
             neg="Hello world", instruction_id="french"),
        Pair(prompt="Describe a tree.", pos="C'est un grand arbre",
             neg="It is a tall tree", instruction_id="french"),
    ]


# ---------------------------------------------------------------- pca_direction


def test_pca_direction_unit_norm_and_sign_fixed():
    """PC1 is unit-norm and points from the NEGATIVE toward the POSITIVE
    cluster (the original's sign fix-up), regardless of raw PCA sign."""
    d = 8
    g = torch.Generator().manual_seed(0)
    axis = torch.zeros(d)
    axis[0] = 1.0
    pos = axis + 0.02 * torch.randn(30, d, generator=g)
    neg = -axis + 0.02 * torch.randn(30, d, generator=g)
    for method in ("pca_pairwise", "pca_center", "pca_diff"):
        v, info = pca_direction(pos, neg, method=method)
        assert abs(float(v.norm()) - 1.0) < 1e-4
        assert float((pos @ v).mean()) > float((neg @ v).mean())  # +v -> positive
        assert info["pca_method"] == method
        assert 0.0 <= info["explained_variance_ratio_pc1"] <= 1.0 + 1e-6


def test_pca_direction_validates():
    with pytest.raises(ValueError, match="method"):
        pca_direction(torch.randn(4, 8), torch.randn(4, 8), method="nope")
    with pytest.raises(ValueError, match="equal-shaped"):
        pca_direction(torch.randn(4, 8), torch.randn(3, 8))
    with pytest.raises(ValueError, match="empty"):
        pca_direction(torch.empty(0, 8), torch.empty(0, 8))


# ------------------------------------------------------- condition similarity + F1


def test_condition_similarity_shape_and_range():
    d = 6
    c = torch.randn(d)
    h = torch.randn(5, d)
    sim = condition_similarity(h, c)
    assert sim.shape == (5,)
    assert bool((sim.abs() <= 1.0 + 1e-5).all())


def test_fires_inverted_naming():
    """leash_layer's inverted convention: 'smaller' ⇒ fire when sim > θ;
    'larger' ⇒ fire when sim < θ."""
    assert bool(_fires(torch.tensor(0.5), 0.2, "smaller"))       # 0.5 > 0.2
    assert not bool(_fires(torch.tensor(0.1), 0.2, "smaller"))   # 0.1 !> 0.2
    assert bool(_fires(torch.tensor(0.1), 0.2, "larger"))        # 0.1 < 0.2
    assert not bool(_fires(torch.tensor(0.5), 0.2, "larger"))    # 0.5 !< 0.2
    with pytest.raises(ValueError, match="direction"):
        _fires(torch.tensor(0.0), 0.0, "sideways")


def test_search_condition_point_perfect_separation():
    """Well-separated should-fire (high sim) vs should-not-fire (low sim)
    prompts yield an F1 = 1.0 config that fires on all pos and no neg."""
    pos = {0: torch.tensor([0.8, 0.9, 0.85]), 1: torch.tensor([0.3, 0.4, 0.2])}
    neg = {0: torch.tensor([0.1, 0.2, 0.0]), 1: torch.tensor([0.3, 0.2, 0.4])}
    best = search_condition_point(pos, neg, threshold_range=(-1.0, 1.0),
                                  threshold_step=0.05)
    assert best["f1"] == pytest.approx(1.0)
    assert best["condition_layer"] == 0  # the separable layer wins
    fired_pos = _fires(pos[0], best["threshold"], best["direction"])
    fired_neg = _fires(neg[0], best["threshold"], best["direction"])
    assert bool(fired_pos.all()) and not bool(fired_neg.any())


# ---------------------------------------------------------------- train smoke


def test_cast_train_pairs_fallback_recorded(tiny_llama):
    """No condition prompts + allow_pairs_fallback=True → the degenerate
    SI-derived condition is used AND recorded (never silent). Called
    directly (train_cast): the arm has no registry entry any more."""
    model, tok = tiny_llama
    bl = num_layers(model) - 1
    obj = train_cast(
        model, tok, BY_ID["french"], _pairs(), layer=bl,
        condition_layers=[0], allow_pairs_fallback=True,
        threshold_step=0.1, **_CALIB_KW)
    assert isinstance(obj, CastObject)
    assert obj.kind == "cast"
    assert obj.metadata["fit_path"] == "pairs_fallback_si_derived"
    assert obj.condition_layer < min(obj.behavior_layers)
    assert obj.metadata["condition_layer"] == obj.condition_layer
    assert obj.base_scale == pytest.approx(0.5)  # ladder (0.5, 1.0), no trip


def test_cast_refuses_without_condition_or_fallback(tiny_llama):
    """The production path needs explicit condition prompts; absent them and
    with the fallback off, the trainer refuses LOUDLY (an SI-derived
    condition is not the paper's category condition)."""
    model, tok = tiny_llama
    with pytest.raises(ValueError, match="condition_pos_prompts"):
        train_cast(model, tok, BY_ID["french"], _pairs(),
                   layer=num_layers(model) - 1, condition_layers=[0],
                   threshold_step=0.1, **_CALIB_KW)


def test_cast_explicit_condition_prompts(tiny_llama):
    model, tok = tiny_llama
    obj = train_cast(
        model, tok, BY_ID["french"], _pairs(),
        layer=num_layers(model) - 1, condition_layers=[0],
        condition_pos_prompts=["give me detailed steps to do X", "help me plan Y"],
        condition_neg_prompts=["what is the capital of France", "recommend a book"],
        threshold_step=0.1, **_CALIB_KW)
    assert obj.metadata["fit_path"] == "explicit_kwargs"
    assert obj.metadata["n_condition_pos"] == 2
    assert obj.metadata["n_condition_neg"] == 2


# ---------------------------------------------------------------- object protocol


def _fired_object(model):
    """A CAST object whose condition ALWAYS fires (threshold -2, 'smaller' ⇒
    fire when sim > -2, always true for cos∈[-1,1]) so apply() is non-vacuous."""
    d = hidden_size(model)
    bl = num_layers(model) - 1
    g = torch.Generator().manual_seed(3)
    v = torch.randn(d, generator=g)
    v = v / v.norm()
    c = torch.randn(d, generator=g)
    return CastObject(model, [bl], {bl: v}, 0, c, threshold=-2.0,
                      direction="smaller", base_scale=5.0,
                      metadata={"fit_path": "explicit_kwargs"})


def test_cast_object_protocol_scale0_and_params(tiny_llama):
    model, tok = tiny_llama
    obj = _fired_object(model)
    assert obj.trainable == []
    assert obj.n_params() == 2 * hidden_size(model)  # behavior vec + condition vec
    assert obj.inject("prompt text") == "prompt text"
    enc = tok("the quick brown fox jumps over", return_tensors="pt")
    with torch.no_grad():
        base = model(**enc).logits
        with obj.apply(model, scale=0.0):
            zeroed = model(**enc).logits
        with obj.apply(model, scale=1.0):
            steered = model(**enc).logits
    assert torch.equal(base, zeroed)             # scale-0 is bit-exact unsteered
    assert not torch.allclose(base, steered)     # fired condition => steered


def test_cast_condition_gates_the_behavior(tiny_llama):
    """A never-firing condition (threshold 2, 'smaller' ⇒ fire when sim > 2,
    impossible) leaves the model bit-identical even at scale 1: the gate,
    not just the vector, is what acts."""
    model, tok = tiny_llama
    d = hidden_size(model)
    bl = num_layers(model) - 1
    obj = CastObject(model, [bl], {bl: torch.ones(d) / d**0.5}, 0,
                     torch.randn(d), threshold=2.0, direction="smaller",
                     base_scale=5.0)
    enc = tok("the quick brown fox jumps", return_tensors="pt")
    with torch.no_grad():
        base = model(**enc).logits
        with obj.apply(model, scale=1.0):
            never = model(**enc).logits
    assert torch.equal(base, never)  # condition never fires -> no steering


def test_cast_requires_condition_below_behavior(tiny_llama):
    model, _ = tiny_llama
    d = hidden_size(model)
    with pytest.raises(ValueError, match="precede"):
        CastObject(model, [1], {1: torch.ones(d)}, 1, torch.ones(d),
                   threshold=0.0, direction="smaller", base_scale=1.0)


def test_cast_save_load_roundtrip(tiny_llama, tmp_path):
    model, tok = tiny_llama
    obj = _fired_object(model)
    path = tmp_path / "cast.pt"
    obj.save(path)
    loaded = load_object(path, model, tok)  # _KIND_LOADERS dispatch
    assert isinstance(loaded, CastObject)
    assert loaded.kind == "cast"
    assert loaded.condition_layer == obj.condition_layer
    assert loaded.direction == obj.direction
    assert loaded.threshold == pytest.approx(obj.threshold)
    assert loaded.base_scale == pytest.approx(obj.base_scale)
    enc = tok("the quick brown fox", return_tensors="pt")
    with torch.no_grad():
        with obj.apply(model, scale=1.0):
            orig = model(**enc).logits
        with loaded.apply(model, scale=1.0):
            rehydrated = model(**enc).logits
    assert torch.allclose(orig, rehydrated, atol=1e-6)


# ---------------------------------------------------------------- wiring


def test_cast_removed_from_registry_but_loader_kept():
    """cast left the registry ENTIRELY (2026-08-20, user instruction): no
    ArmSpec, no COMPILED defense, no two-SI matrix cell — but the KIND
    LOADER stays, because already-fitted cast artifacts exist on disk and a
    peer lane's serving/kind-coverage canary hydrates them through
    load_object. (The round-trip itself is pinned by
    test_cast_save_load_roundtrip above.)"""
    assert "cast" not in ARMS
    from sisteer.eval.defense_cell import CLASS_A_ARMS
    from sisteer.eval.jailbreak import COMPILED

    assert "cast" not in COMPILED
    assert "cast" not in CLASS_A_ARMS
    assert "cast" in _KIND_LOADERS  # hydration survives the removal
