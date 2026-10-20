"""CPU tests for scripts/tune_lora_all.py — the lora_r1_all hyperparameter
sweep driver: the hand-rolled base-2 Sobol generator (no scipy in the env),
the log-box mapping, the joint quadratic fit, the two-KL-ratio probe
(residual KL(prompted||steered) and displacement KL(steered||unsteered),
both over the no-op floor), and the pre-clip gradient-norm tap."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "tune_lora_all",
    Path(__file__).resolve().parent.parent / "scripts" / "tune_lora_all.py")
tla = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tla)


# ---------------------------------------------------------------- Sobol --

def test_sobol_known_prefix():
    pts = tla.sobol_points(2, d=3)
    assert pts[0] == (0.0, 0.0, 0.0)
    assert pts[1] == (0.5, 0.5, 0.5)
    assert pts[2] == (0.75, 0.25, 0.25)
    assert pts[3] == (0.25, 0.75, 0.75)


def test_sobol_16_stratified_per_dim_and_pairwise():
    """The property that makes 2^m Sobol a sweep grid: at 16 points every
    dimension hits each of 16 equal bins exactly once, and every PAIR of
    dimensions covers the 4x4 grid exactly once per cell. Wrong direction
    numbers break this immediately."""
    pts = tla.sobol_points(4, d=3)
    assert len(pts) == 16
    for dim in range(3):
        bins = sorted(int(p[dim] * 16) for p in pts)
        assert bins == list(range(16)), f"dim {dim} not stratified: {bins}"
    for a in range(3):
        for b in range(a + 1, 3):
            cells = sorted((int(p[a] * 4), int(p[b] * 4)) for p in pts)
            assert cells == sorted(
                (i, j) for i in range(4) for j in range(4)
            ), f"dims ({a},{b}) not 4x4 balanced"


def test_map_log_endpoints_and_midpoint():
    assert tla.map_log(0.0, 1e-4, 1e-2) == pytest.approx(1e-4)
    assert tla.map_log(1.0, 1e-4, 1e-2) == pytest.approx(1e-2)
    assert tla.map_log(0.5, 1e-4, 1e-2) == pytest.approx(1e-3)


# ------------------------------------------------------ quadratic fit --

def test_fit_quadratic_recovers_planted_minimum():
    """Plant a positive-definite quadratic in log-space over the real sweep
    box, evaluate it at the 16 Sobol trial points, and require the fitted
    argmin to land on the planted one."""
    box = {"lr": (3e-5, 3e-3), "warmup_frac": (0.01, 0.3),
           "max_grad_norm": (0.1, 10.0)}
    keys = ("lr", "warmup_frac", "max_grad_norm")
    target = {"lr": 3e-4, "warmup_frac": 0.15, "max_grad_norm": 1.0}
    tz = [math.log(target[k]) for k in keys]
    H = torch.tensor([[2.0, 0.3, 0.0], [0.3, 1.0, 0.1], [0.0, 0.1, 0.5]],
                     dtype=torch.float64)
    rows = []
    for u in tla.sobol_points(4, d=3):
        vals = {k: tla.map_log(u[i], *box[k]) for i, k in enumerate(keys)}
        z = torch.tensor([math.log(vals[k]) for k in keys],
                         dtype=torch.float64) - torch.tensor(tz)
        logy = -3.0 + 0.5 * float(z @ H @ z)
        rows.append({**vals, "best_heldout_loss": math.exp(logy)})
    fit = tla.fit_quadratic(rows, keys)
    assert fit["positive_definite"]
    for k in keys:
        assert fit["argmin"][k] == pytest.approx(target[k], rel=1e-3)


# ------------------------------------------------------- KL ratio probe --

def _mini_examples(tok):
    from sisteer.data.instructions import BY_ID
    from sisteer.data.pairs import Pair, pairs_to_distill_examples

    pairs = [
        Pair(prompt="Say something nice.", pos="Bonjour le monde",
             neg="Hello world", instruction_id="french"),
        Pair(prompt="Describe a tree.", pos="C'est un grand arbre",
             neg="It is a tall tree", instruction_id="french"),
        Pair(prompt="Name a color.", pos="Le bleu", neg="Blue",
             instruction_id="french"),
        Pair(prompt="Count to three.", pos="Un deux trois",
             neg="One two three", instruction_id="french"),
    ]
    return pairs_to_distill_examples(tok, pairs, BY_ID["french"])


def test_klratio_probe_zero_displacement_at_init(tiny_llama):
    """A fresh MultiLayerLoRAObject is B-zero-init, so the steered forward
    is bit-identical to the unsteered one: displacement KL must be ~0 and
    the residual ratio ~1 (the object has neither helped nor moved)."""
    from sisteer.extraction.multilayer import MultiLayerLoRAObject

    model, tok = tiny_llama
    exs = _mini_examples(tok)
    probe = tla.KLRatioProbe(tok, exs, max_len=64, batch_size=2, n=4)
    base = probe.measure(model)
    assert base["probe/noop_floor"] > 0
    out = probe.measure(model, MultiLayerLoRAObject(model))
    assert abs(out["probe/displacement_kl"]) < 1e-8
    assert out["probe/displacement_ratio"] == pytest.approx(0.0, abs=1e-6)
    assert out["probe/residual_ratio"] == pytest.approx(1.0, rel=1e-5)
    # The denominator is computed once and pinned.
    assert out["probe/noop_floor"] == base["probe/noop_floor"]


def test_klratio_probe_moves_when_steered(tiny_llama):
    """Perturbing one layer's B must register as displacement > 0."""
    from sisteer.extraction.multilayer import MultiLayerLoRAObject

    model, tok = tiny_llama
    exs = _mini_examples(tok)
    probe = tla.KLRatioProbe(tok, exs, max_len=64, batch_size=2, n=4)
    probe.measure(model)
    obj = MultiLayerLoRAObject(model)
    with torch.no_grad():
        obj.B[obj.layers[0]].data += 5.0
    out = probe.measure(model, obj)
    assert out["probe/displacement_kl"] > 1e-4
    assert out["probe/displacement_ratio"] > 0


# ------------------------------------------------------- grad-norm tap --

def test_gradnormtap_preclip_vs_postclip():
    a = torch.nn.Parameter(torch.ones(4))
    b = torch.nn.Parameter(torch.ones(3))
    tap = tla.GradNormTap([a, b])
    loss = (3.0 * a).sum() + (4.0 * b).sum()
    loss.backward()
    expected = math.sqrt(4 * 9.0 + 3 * 16.0)
    assert tap.flush() == pytest.approx(expected, rel=1e-6)
    # Clip AFTER the hooks fired: .grad shrinks, the tap already recorded raw.
    torch.nn.utils.clip_grad_norm_([a, b], 1.0)
    post = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in (a, b)))
    assert post == pytest.approx(1.0, rel=1e-5)
    tap.remove()
    # After remove(), a fresh backward leaves the tap empty.
    (a.sum() + b.sum()).backward()
    assert math.isnan(tap.flush())
