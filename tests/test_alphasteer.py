"""AlphaSteer arm (arXiv 2506.07022): null-space projector and ridge
regression on synthetic data, the registry train smoke (with the lenient
calibration kwargs the random tiny model needs), the object protocol
(trainable / n_params / scale-0 no-op / save + load_blob round trip), and
the arm wiring on this branch (appended at index 18, full core suite set).

Ported from the lofit-arm branch's test_methods_s2.py hunks for the two-SI
bring-in; the object-protocol test is new (the acceptance contract
every brought-in arm must satisfy). CPU-only, tiny-llama fixture.
"""

import pytest
import torch

from sisteer.data.instructions import BY_ID
from sisteer.data.pairs import Pair
from sisteer.extraction.alphasteer import (
    AlphaSteerObject,
    null_space_projector,
    ridge_delta,
    train_alphasteer,
)
from sisteer.methods import ARMS, load_object
from sisteer.models import hidden_size

# Lenient calibration so the base-scale sweep is deterministic on the random
# tiny model (its generations' coherence is meaningless): never trip the
# degeneration guard, so base_scale = last_rung/2.
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


def _base_logits(model, enc):
    with torch.no_grad():
        return model(**enc).logits


# ---------------------------------------------------------------- closed form


def test_alphasteer_nullspace_zeroes_planted_benign():
    """P projects onto the LEAST-benign-variance subspace, so benign
    activations (living in the top-variance span) are nulled; P is a rank-k
    idempotent projector."""
    d, rank = 12, 3
    g = torch.Generator().manual_seed(5)
    basis = torch.eye(d)
    coeffs = torch.randn(40, rank, generator=g)
    benign = coeffs @ basis[:rank]        # benign spans only the first `rank` dims
    P, info = null_space_projector(benign, ratio=0.5)
    k = info["nullspace_rank"]
    assert k == int(0.5 * d)
    assert (benign @ P).abs().max() < 1e-4          # benign nulled
    assert torch.allclose(P @ P, P, atol=1e-4)      # idempotent
    assert abs(P.diagonal().sum() - k) < 1e-3       # rank-k projector
    assert info["benign_energy_kept"] < 1e-6
    # a direction outside the benign span survives the projection
    outside = basis[rank + 1]
    assert (outside @ P).norm() > 0.1


def test_alphasteer_nullspace_validates_ratio():
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="nullspace_ratio"):
            null_space_projector(torch.randn(8, 4), ratio=bad)


def test_alphasteer_ridge_maps_projected_to_refusal():
    """AlphaSteer maps every (projected) malicious activation onto the SAME
    refusal direction; the ridge fit therefore achieves low residual when the
    malicious activations CLUSTER (as real ones do — they share structure),
    which is the regime the method relies on. With the full projector
    (identity) and clustered inputs the fit sends them all to R."""
    d = 8
    g = torch.Generator().manual_seed(7)
    mu = torch.randn(d, generator=g)
    H = mu.unsqueeze(0).expand(64, d) + 0.02 * torch.randn(64, d, generator=g)
    refusal = torch.randn(d, generator=g)
    P = torch.eye(d)
    tilde_delta, info = ridge_delta(H, P, refusal, lambda_reg=0.01)
    assert tilde_delta.shape == (d, d)
    assert info["regression_residual"] < 0.2       # clustered X @ Δ̃ ≈ R
    assert 0.0 <= info["malicious_energy_kept"] <= 1.0 + 1e-6


# ---------------------------------------------------------------- train smoke


def test_alphasteer_train_smoke_and_prefill_only(tiny_llama):
    model, tok = tiny_llama
    obj = train_alphasteer(model, tok, BY_ID["french"], _pairs(), layer=1,
                           nullspace_ratio=0.5, epochs=1, max_len=64,
                           **_CALIB_KW)
    assert isinstance(obj, AlphaSteerObject)
    assert obj.kind == "alphasteer"
    assert "nullspace_rank" in obj.metadata
    assert "regression_residual" in obj.metadata
    # decode-step forward (seq_len == 1) is a no-op: prefill-only edit
    one = tok("hello", return_tensors="pt")
    one = {k: v[:, :1] for k, v in one.items()}
    base = _base_logits(model, one)
    with torch.no_grad(), obj.apply(model, scale=1.0):
        decode = model(**one).logits
    assert torch.equal(base, decode), "alphasteer must not steer decode steps"


def test_alphasteer_registry_arm_trains(tiny_llama):
    """The registered arm trains through the uniform ArmSpec.train signature
    (its dedicated smoke — alphasteer sits outside the generic
    test_methods_s2 smoke because its closed-form fit needs the lenient
    calibration kwargs on the random tiny model)."""
    model, tok = tiny_llama
    obj = ARMS["alphasteer"].train(
        model, tok, BY_ID["french"], _pairs(), layer=1,
        epochs=1, max_len=64, **_CALIB_KW)
    assert isinstance(obj, AlphaSteerObject)
    assert obj.base_scale == pytest.approx(0.5)  # ladder (0.5, 1.0), no trip
    enc = tok(obj.inject("the quick brown fox"), return_tensors="pt")
    with torch.no_grad(), obj.apply(model, scale=0.5):
        model(**enc)


# ---------------------------------------------------------------- object protocol


def _nonzero_object(model):
    d = hidden_size(model)
    g = torch.Generator().manual_seed(17)
    return AlphaSteerObject(
        model, layer=1, M=0.1 * torch.randn(d, d, generator=g),
        base_scale=2.0, refusal=0.1 * torch.randn(d, generator=g),
        metadata={"nullspace_rank": 8})


def test_alphasteer_object_protocol_scale0_and_params(tiny_llama):
    model, tok = tiny_llama
    d = hidden_size(model)
    obj = _nonzero_object(model)
    assert obj.trainable == []          # closed-form: nothing gradient-trains
    assert obj.n_params() == d * d
    assert obj.inject("prompt text") == "prompt text"
    enc = tok("the quick brown fox jumps", return_tensors="pt")
    base = _base_logits(model, enc)
    with torch.no_grad():
        with obj.apply(model, scale=0.0):
            zeroed = model(**enc).logits
        with obj.apply(model, scale=1.0):
            steered = model(**enc).logits
    assert torch.equal(base, zeroed)    # scale-0 is bit-exact unsteered
    assert not torch.allclose(base, steered)  # ... and not vacuously so


def test_alphasteer_save_load_blob_roundtrip(tiny_llama, tmp_path):
    model, tok = tiny_llama
    obj = _nonzero_object(model)
    path = tmp_path / "alphasteer.pt"
    obj.save(path)
    loaded = load_object(path, model, tok)   # _KIND_LOADERS dispatch
    assert isinstance(loaded, AlphaSteerObject)
    assert loaded.kind == "alphasteer"
    assert loaded.layer == obj.layer
    assert loaded.base_scale == pytest.approx(obj.base_scale)
    assert torch.equal(loaded.M.cpu(), obj.M.cpu())
    assert torch.equal(loaded.refusal.cpu(), obj.refusal.cpu())
    assert loaded.metadata["nullspace_rank"] == 8
    enc = tok("the quick brown fox", return_tensors="pt")
    with torch.no_grad():
        with obj.apply(model, scale=1.0):
            orig = model(**enc).logits
        with loaded.apply(model, scale=1.0):
            rehydrated = model(**enc).logits
    assert torch.allclose(orig, rehydrated, atol=1e-6)


# ---------------------------------------------------------------- wiring


def test_alphasteer_arm_wiring():
    """Appended between antidote and lofit (the append order — index
    19 since the 2026-08-28 lora_r4 alias joined at index 4 next to its
    lora_r1_all sibling; see test_methods_s2's order test), with the
    FULL core suite set: the lofit-arm branch registered it
    IFEval-only, but here it must run the same suites as the comparable
    arms (psr_kl_all's set)."""
    assert list(ARMS)[19] == "alphasteer"
    assert ARMS["alphasteer"].suites == frozenset({"ifeval"})
    assert ARMS["alphasteer"].suites == ARMS["psr_kl_all"].suites
def test_alphasteer_nullspace_robust_to_degenerate_gram():
    """Regression for jobs 141778_{11,23,26} (alphasteer FAILED with
    torch.linalg.svd 'too many repeated singular values'): the benign Gram is
    LOW-RANK (n_prompts ≪ d), a maximally-degenerate spectrum. eigh must build
    the projector WITHOUT raising and still null benign; the eigh code path is
    pinned by the gram_*_eigenvalue diagnostic key."""
    d = 128
    g = torch.Generator().manual_seed(9)
    # 200 rows drawn from only 4 fixed directions ⇒ rank 4, so the Gram has
    # 124 repeated zero eigenvalues + repeated positive ones — the exact
    # degenerate spectrum that broke svd at d=4096.
    dirs = torch.eye(d)[:4]
    idx = torch.randint(0, 4, (200,), generator=g)
    benign = dirs[idx] * torch.randn(200, 1, generator=g)
    P, info = null_space_projector(benign, ratio=0.5)   # must not raise
    assert "gram_bottom_eigenvalue" in info             # eigh path (not svd)
    assert (benign @ P).abs().max() < 1e-4              # benign still nulled
    assert torch.allclose(P @ P, P, atol=1e-4)         # idempotent
    assert abs(P.diagonal().sum() - info["nullspace_rank"]) < 1e-2


