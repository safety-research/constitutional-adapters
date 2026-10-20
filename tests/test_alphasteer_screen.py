"""The F.9-AUROC screen (sisteer.extraction.alphasteer_screen).

The load-bearing tests are the ones a plausible implementation fails:
``test_screen_scores_the_deployed_row_vector_operation`` uses an ASYMMETRIC
``M = u rᵀ`` (a symmetric fixture passes either orientation and would have
let the transpose through); the split tests plant the SAME prompt at two
indices in two source slices and check it lands on ONE side while every row
and its multiplicity survives; ``test_eigh_is_computed_once_per_layer_for_all_p``
counts eigh calls. Synthetic tensors throughout; the one model-backed test
runs the whole screen end to end on the tiny-llama fixture through the CLI's
``run_screen``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from sisteer.extraction import alphasteer_screen as scr
from sisteer.extraction.alphasteer import (
    null_space_projector,
    refusal_direction,
    ridge_delta,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _planted(d=12, n_b=200, n_h=60, seed=0):
    """Benign living in a low-variance-free span; harmful shifted along a
    direction OUTSIDE the benign span — separable by construction."""
    g = torch.Generator().manual_seed(seed)
    span = torch.eye(d)[:4]
    benign = torch.randn(n_b, 4, generator=g) @ span
    outside = torch.eye(d)[7]
    harmful = (torch.randn(n_h, 4, generator=g) @ span
               + 3.0 * outside.unsqueeze(0)
               + 0.05 * torch.randn(n_h, d, generator=g))
    return benign, harmful


def _fit_M(benign, harmful, p=0.5, lam=10.0):
    r = refusal_direction(harmful, benign)
    P, _ = null_space_projector(benign, p)
    tilde, _ = ridge_delta(harmful, P, r, lam, solver="psd_pinv")
    return P @ tilde, r


# ---------------------------------------------------------------- the deployed operation


def test_screen_scores_the_deployed_row_vector_operation(monkeypatch):
    """Asymmetric M = u r^T with u != r: ||H @ M|| and ||(M @ H^T)^T|| DIFFER,
    and the screen returns the first — through deployed_score."""
    d, n = 10, 40
    g = torch.Generator().manual_seed(1)
    u = torch.zeros(d); u[2] = 1.0
    r = torch.zeros(d); r[7] = 3.0
    M = torch.outer(u, r)                       # rank-1, deliberately asymmetric
    assert not torch.allclose(M, M.t())
    H = torch.randn(n, d, generator=g)
    deployed = torch.linalg.vector_norm(H @ M, dim=-1)
    transpose = torch.linalg.vector_norm((M @ H.t()).t(), dim=-1)
    assert not torch.allclose(deployed, transpose), "fixture must separate the orientations"
    assert torch.allclose(scr.deployed_score(H, M), deployed)
    assert torch.allclose(deployed, (H @ u).abs() * r.norm())   # |h.u| ||r||
    # and the screen loop's ranking is produced by deployed_score, not a local expression
    seen = []
    real = scr.deployed_score

    def spy(Hh, Mm):
        seen.append((Hh.shape, Mm.shape))
        return real(Hh, Mm)

    monkeypatch.setattr(scr, "deployed_score", spy)
    benign, harmful = _planted()
    rows = scr.screen_layers_and_p(
        benign_fit={0: benign[:150]}, harmful_fit={0: harmful[:40]},
        benign_held={0: benign[150:]}, harmful_held={0: harmful[40:]},
        layers=[0], n_blocks=4, p_grid=(0.5,), lambda_reg=10.0)
    assert seen and all(s[1] == (12, 12) for s in seen)
    assert rows[0]["auroc"] > 0.9


def test_llama_recovery_uses_the_same_scoring_helper(monkeypatch):
    """The released-matrix reading (plan Phase D) and the screen call ONE
    helper; a spy sees both."""
    calls = []
    real = scr.deployed_score
    monkeypatch.setattr(scr, "deployed_score",
                        lambda H, M: (calls.append("call"), real(H, M))[1])
    benign, harmful = _planted()
    scr.screen_layers_and_p(
        benign_fit={0: benign[:150]}, harmful_fit={0: harmful[:40]},
        benign_held={0: benign[150:]}, harmful_held={0: harmful[40:]},
        layers=[0], n_blocks=4, p_grid=(0.5,), lambda_reg=10.0)
    n_screen = len(calls)
    assert n_screen >= 1
    M, _ = _fit_M(benign[:150], harmful[:40])
    # layer 1 is the synthetic paper layer: layer 0 is never eligible under
    # the shared predicate, on this path as on the screen's
    reading = scr.released_matrix_reading(
        benign_held={0: benign[150:], 1: benign[150:]},
        harmful_held={0: harmful[40:], 1: harmful[40:]},
        released={1: M, 0: torch.zeros_like(M)}, paper_layers=[1],
        benign_fit={1: benign[:150]}, paper_ratios={1: 0.5},
        recovery_threshold=1)
    assert len(calls) > n_screen
    assert reading["score"]["expression"] == scr.SCORE_EXPRESSION
    assert reading["released_nonzero_layers"] == [1]
    assert reading["per_layer"]["0"]["auroc"] == 0.5          # zero slice: forced 0.5
    assert reading["per_layer"]["0"]["leakage"] is None       # no ratio: never imputed
    assert reading["per_layer"]["1"]["eligible"] and reading["passed"]
    assert reading["recovered"] == [1] and reading["n_recovered"] == 1


def test_released_reading_eligibility_is_auroc_only_and_leakage_is_diagnostic():
    """Eligibility on the released-matrix path (the Llama gate): a paper
    layer whose reconstructed projector keeps FAR more benign energy than
    the retired 1e-3 bound is eligible when its AUROC passes; the leakage
    is still computed and reported beside it; the rule string says so; and
    the reading uses the screen's predicate (no leakage parameter
    anywhere)."""
    import inspect

    # a FULL-RANK benign Gram (n=400 > d=12, anisotropic noise): every kept
    # null-space direction carries real benign variance, so the in-sample
    # leakage is honestly ~1e-1 — the regime the rule describes, not the
    # rank-deficient one (the `_planted` benign spans 4 dims and would give
    # ~0 by construction, exactly the artefact the retired bound measured)
    g = torch.Generator().manual_seed(1)
    d = 12
    span = torch.eye(d)[:4]
    scale = torch.full((d,), 0.3)
    scale[7] = 0.1                                   # the harmful direction: least benign variance
    benign = torch.randn(400, 4, generator=g) @ span + torch.randn(400, d, generator=g) * scale
    harmful = (torch.randn(60, 4, generator=g) @ span + torch.randn(60, d, generator=g) * scale
               + 3.0 * torch.eye(d)[7])
    M, _ = _fit_M(benign[:300], harmful[:40])
    reading = scr.released_matrix_reading(
        benign_held={0: benign[300:], 1: benign[300:]},
        harmful_held={0: harmful[40:], 1: harmful[40:]},
        released={0: M, 1: M}, paper_layers=[1], benign_fit={0: benign[:300], 1: benign[:300]},
        paper_ratios={0: 0.5, 1: 0.5}, recovery_threshold=1)
    row = reading["per_layer"]["1"]
    assert row["auroc"] >= 0.95
    assert row["leakage"] is not None and row["leakage"] > 10 * scr.LEAKAGE_REPORT_REFERENCE
    assert row["eligible"] and row["reason"] is None
    assert reading["passed"] and reading["recovered"] == [1]
    # the same matrix at layer 0 scores the same AUROC and is still never
    # eligible — the shared predicate's layer-0 clause, on this path too
    zero = reading["per_layer"]["0"]
    assert zero["auroc"] == row["auroc"] and not zero["eligible"] and "layer 0" in zero["reason"]
    assert reading["auroc_threshold"] == scr.DEFAULT_AUROC_THRESHOLD
    assert reading["eligibility"] == scr.ELIGIBILITY_RULE
    for text in (reading["rule"], scr.ELIGIBILITY_RULE):
        assert "AUROC ONLY" in text and "diagnostic" in text and "D38" in text
        assert "AND leakage" not in text and "leakage <=" not in text
    params = [p for p in inspect.signature(scr.released_matrix_reading).parameters
              if p != "benign_fit"]
    assert not any("leak" in p for p in params)
    # and the same predicate rejects on AUROC alone, whatever the leakage
    weak = scr.released_matrix_reading(
        benign_held={1: benign[300:]}, harmful_held={1: benign[300:360]},   # inseparable
        released={1: M}, paper_layers=[1], benign_fit={1: benign[:300]},
        paper_ratios={1: 0.5}, recovery_threshold=1)
    assert not weak["per_layer"]["1"]["eligible"] and "auroc" in weak["per_layer"]["1"]["reason"]


def test_rank1_identity_matches_the_matmul_score():
    benign, harmful = _planted(d=16, n_b=300, n_h=80)
    M, r = _fit_M(benign, harmful, p=0.5)
    u = (M @ r) / float(r @ r)                  # M = u r^T  =>  M r = u ||r||^2
    H = torch.cat([benign[:20], harmful[:20]])
    s = scr.deployed_score(H.double(), M.double())
    s1 = scr.rank1_score(H.double(), u.double(), r.double())
    assert scr.rank1_agreement(s, s1, H, M) <= 1e-6
    # the helper's own screen-side factorisation u = P A+ X^T 1 gives the same
    assert float((M - torch.outer(u, r)).norm() / M.norm()) < 1e-5


def _near_null(d=256, n_b=600, n_h=200, seed=0, common=1000.0, harm_shift=0.05,
               var=0.5, noise=0.02, k=24):
    """bf16-quantised activations dominated by ONE shared component (a
    template token every prompt ends on) with small variation and a small
    harmful shift: the null-space projector nulls nearly everything, so the
    deployed score is a CANCELLATION of terms of size ~‖h‖·‖M‖. The
    well-separated ``_planted`` fixture cannot exercise this."""
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(d, generator=g)
    c = common * c / c.norm()
    span = torch.linalg.qr(torch.randn(d, k, generator=g))[0]
    benign = c + var * torch.randn(n_b, k, generator=g) @ span.t()
    outside = torch.randn(d, generator=g)
    outside -= span @ (span.t() @ outside)
    outside /= outside.norm()
    harmful = (c + var * torch.randn(n_h, k, generator=g) @ span.t()
               + harm_shift * outside + noise * torch.randn(n_h, d, generator=g))
    q = lambda x: x.to(torch.bfloat16).float()        # noqa: E731 — bf16 capture
    return q(benign), q(harmful)


def test_rank1_guard_survives_near_null_cancellation(monkeypatch):
    """On nearly nulled activations the fitted M is rank-1 to fp32
    rounding (~4e-7 relative), yet the OLD statistic — fp32 matmul score vs
    float64 rank-1 score, relative to the MAX SCORE — read that rounding as
    disagreement at the 1e-5 tolerance and aborted the whole screen;
    evaluating both paths in float64 but still dividing by the max score is
    not enough. Relative to the roundoff scale max‖h‖·‖M‖ the same fit sits
    decades under the tolerance and the screen runs. A real disagreement at
    that scale still refuses.

    What is asserted about the two OLD forms is their SEPARATION from the new
    statistic, not an absolute crossing of the tolerance: their size on this
    synthetic fixture is fp32/f64 matmul ROUNDOFF and moves with the BLAS
    kernel and thread count. Measured 2026-09-11 (torch 2.11 CPU, MKL):
    The grading host x86_64 at the fixture's depth, 4 threads / 1 thread: old 2.22e-5 /
    2.68e-5, f64-over-max-score 1.31e-5 / 1.98e-5, new 2.9e-9 / 5.9e-9; over
    common in {500..8000} x {1,4} threads: old 1.6e-5..1.5e-4,
    f64-over-max-score 8.1e-7..6.9e-5 (UNDER 1e-5 in 3 of 10 cells), new
    2.2e-10..8.8e-9. The Modal CPU-suite container read one of the two old
    forms at 9.25e-6. So the absolute checks `old > tol` / `f64 > tol` were
    a platform-marginal reproduction of the historical misfire (the real
    screen's was not marginal), while the separation held at >= 3.6 decades
    in every cell and the new statistic sat >= 3 decades under the tolerance
    in every cell. The floors below (2 decades each) keep >= 1.5 decades of
    headroom to the worst measured cell; DEFAULT_RANK1_TOL is untouched."""
    benign, harmful = _near_null()
    Hb, Hm, Hb_h, Hm_h = benign[:500], harmful[:160], benign[500:], harmful[160:]
    p, lam = 0.6, 10.0
    r = refusal_direction(Hm, Hb)
    P, _ = null_space_projector(Hb, p)
    tilde, _ = ridge_delta(Hm, P, r, lam, solver="psd_pinv")
    M = P @ tilde
    u = (M @ r) / float(r @ r)
    assert float((M - torch.outer(u, r)).norm() / M.norm()) < 1e-6   # rank-1 to fp32 rounding
    H = torch.cat([Hm_h, Hb_h])
    s32 = scr.deployed_score(H, M)
    s64 = scr.deployed_score(H.double(), M.double())
    s1 = scr.rank1_score(H.double(), u.double(), r.double())
    old = float((s32.double() - s1).abs().max() / s32.abs().max())
    f64_max_score = float((s64 - s1).abs().max() / s64.abs().max())
    new = scr.rank1_agreement(s64, s1, H, M)
    assert new < 1e-2 * scr.DEFAULT_RANK1_TOL, new              # the guard: decades under the tolerance
    assert old > 1e2 * new, (old, new)                           # the misfire's statistic is decades noisier ...
    assert f64_max_score > 1e2 * new, (f64_max_score, new)      # ... and dtype alone does not close the gap
    rows = scr.screen_layers_and_p(
        benign_fit={3: Hb}, harmful_fit={3: Hm}, benign_held={3: Hb_h},
        harmful_held={3: Hm_h}, layers=[3], n_blocks=10, p_grid=(p,), lambda_reg=lam)
    assert rows[0]["rank1_agreement_max_rel"] < 1e-2 * scr.DEFAULT_RANK1_TOL
    assert rows[0]["matrix_rank1_residual"] < 1e-6
    # a genuine disagreement at the roundoff scale (10x the tolerance) refuses
    real = scr.rank1_score
    monkeypatch.setattr(scr, "rank1_score", lambda H_, u_, r_: real(H_, u_, r_)
                        + 10 * scr.DEFAULT_RANK1_TOL
                        * torch.linalg.vector_norm(H_, dim=-1).max() * u_.norm() * r_.norm())
    with pytest.raises(AssertionError, match="not rank-1"):
        scr.screen_layers_and_p(
            benign_fit={3: Hb}, harmful_fit={3: Hm}, benign_held={3: Hb_h},
            harmful_held={3: Hm_h}, layers=[3], n_blocks=10, p_grid=(p,), lambda_reg=lam)


def test_auroc_is_1_on_known_separable_synthetic():
    assert scr.auroc(torch.tensor([5.0, 6.0, 7.0]), torch.tensor([1.0, 2.0])) == 1.0
    assert scr.auroc(torch.tensor([1.0, 2.0]), torch.tensor([5.0, 6.0])) == 0.0


def test_auroc_is_half_on_known_inseparable_synthetic():
    g = torch.Generator().manual_seed(3)
    a = torch.randn(4000, generator=g)
    b = torch.randn(4000, generator=g)
    assert abs(scr.auroc(a, b) - 0.5) < 0.02
    # all-tied scores are EXACTLY 0.5 (ties count half)
    assert scr.auroc(torch.ones(5), torch.ones(7)) == 0.5
    # partial ties: hand-computed Mann-Whitney U
    pos = torch.tensor([1.0, 2.0, 2.0])
    neg = torch.tensor([2.0, 0.0])
    # pairs: (1>0)=1,(1 vs 2)=0,(2 vs 2)=.5,(2>0)=1,(2 vs 2)=.5,(2>0)=1 -> 4/6
    assert scr.auroc(pos, neg) == pytest.approx(4.0 / 6.0)
    with pytest.raises(ValueError):
        scr.auroc(torch.tensor([]), neg)


# ---------------------------------------------------------------- the sharing


def test_eigh_is_computed_once_per_layer_for_all_p(monkeypatch):
    """1 (Gram) + len(p_grid) (one psd_pinv per ridge) eigh calls per layer,
    not 2 * len(p_grid)."""
    calls = []
    real = torch.linalg.eigh
    monkeypatch.setattr(torch.linalg, "eigh",
                        lambda A, *a, **k: (calls.append(A.shape), real(A, *a, **k))[1])
    benign, harmful = _planted()
    p_grid = (0.3, 0.4, 0.5, 0.6)
    rows = scr.screen_layers_and_p(
        benign_fit={0: benign[:150], 1: benign[:150] * 2},
        harmful_fit={0: harmful[:40], 1: harmful[:40] * 2},
        benign_held={0: benign[150:], 1: benign[150:] * 2},
        harmful_held={0: harmful[40:], 1: harmful[40:] * 2},
        layers=[0, 1], n_blocks=4, p_grid=p_grid, lambda_reg=10.0)
    assert len(calls) == 2 * (1 + len(p_grid))
    assert len(rows) == 2 * len(p_grid)
    # resume: existing rows are kept and only the missing cells are solved
    calls.clear()
    partial = [r for r in rows if not (r["layer"] == 1 and r["p"] == 0.6)]
    again = scr.screen_layers_and_p(
        benign_fit={0: benign[:150], 1: benign[:150] * 2},
        harmful_fit={0: harmful[:40], 1: harmful[:40] * 2},
        benign_held={0: benign[150:], 1: benign[150:] * 2},
        harmful_held={0: harmful[40:], 1: harmful[40:] * 2},
        layers=[0, 1], n_blocks=4, p_grid=p_grid, lambda_reg=10.0,
        existing_rows=partial)
    assert len(again) == len(rows) and len(calls) == 2      # one Gram eigh + one ridge


def test_p_prefix_identity():
    """P(0.3) sliced from the shared eigh ordering equals
    null_space_projector(0.3) BIT FOR BIT — by construction since S2 (both
    call the one projector_from_prefix); kept as the regression net."""
    benign, _ = _planted(d=20, n_b=64)
    A = benign.float()
    gram = A.t() @ A
    evals, evecs = scr._eigh_with_jitter(gram)
    for p in (0.3, 0.4, 0.5, 0.6):
        P_prefix, k = scr.projector_from_prefix(evecs, p)
        P_ref, info = null_space_projector(benign, p)
        assert torch.equal(P_prefix, P_ref), f"p={p}"
        assert k == info["nullspace_rank"]
    with pytest.raises(ValueError, match="nullspace_ratio"):
        scr.projector_from_prefix(evecs, 1.0)


# ---------------------------------------------------------------- band / selection


def test_band_excludes_out_of_band_layers():
    """15–65 % band arithmetic per block count, the five targets."""
    expect = {42: (7, 27), 32: (5, 20), 40: (6, 26)}
    for n_blocks, (lo, hi) in expect.items():
        got = scr.band_layers(n_blocks)
        assert got == list(range(lo, hi + 1)), (n_blocks, got[0], got[-1])
        assert not scr.in_depth_band(lo - 1, n_blocks)
        assert not scr.in_depth_band(hi + 1, n_blocks)
    assert not scr.in_depth_band(0, 42)


def _rows(spec: dict, n_blocks: int, p_grid=(0.3, 0.4, 0.5, 0.6)) -> list[dict]:
    """spec: layer -> {p: (auroc, leakage)} (missing p -> (0.5, 1e-4))."""
    rows = []
    for l in range(n_blocks):
        for p in p_grid:
            a, leak = spec.get(l, {}).get(p, (0.5, 1e-4))
            rows.append({"layer": l, "p": p, "auroc": a, "leakage": leak,
                         "depth": l / n_blocks,
                         "in_band": scr.in_depth_band(l, n_blocks)})
    return rows


def test_layer_zero_never_eligible():
    n = 10
    spec = {l: {0.5: (1.0, 1e-5)} for l in range(n)}
    sel = scr.select_layers(_rows(spec, n), n_blocks=n, depth_band=(0.0, 1.0))
    pl0 = sel["per_layer"][0]
    assert pl0["layer"] == 0 and pl0["auroc"] == 1.0 and not pl0["eligible"]
    assert "layer 0" in pl0["reason"]
    assert 0 not in sel["chosen"]["layers"] and 0 not in sel["band_layers"]
    assert sel["chosen"]["layers"] == list(range(1, n))


def test_leaky_layer_with_passing_auroc_is_eligible_and_leakage_is_reported():
    """Eligibility is AUROC ONLY. A layer whose projector keeps 50x the
    retired 1e-3 bound is ELIGIBLE when its AUROC passes and its leakage is
    still carried per layer as a diagnostic; a clean projector that fails
    AUROC is ineligible for AUROC alone; no key of the selection reads as a
    leakage gate, and neither predicate takes a leakage argument."""
    import inspect

    n = 20
    spec = {l: {0.5: (0.99, 1e-4)} for l in range(3, 13)}
    spec[5] = {0.5: (0.999, 5e-3)}                  # best AUROC, 50x the retired bound
    spec[6] = {0.5: (0.90, 1e-6)}                   # cleanest projector, fails AUROC
    sel = scr.select_layers(_rows(spec, n), n_blocks=n)
    by = {pl["layer"]: pl for pl in sel["per_layer"]}
    assert by[5]["eligible"] and by[5]["reason"] is None
    assert by[5]["leakage"] == 5e-3                 # reported, not gated
    assert 5 in sel["chosen"]["layers"]
    assert not by[6]["eligible"] and "auroc" in by[6]["reason"]
    assert "leakage" not in by[6]["reason"] and 6 not in sel["chosen"]["layers"]
    assert "leakage_ok" not in by[5] and "eligible_counts_auroc_only" not in sel
    assert sel["eligible_counts"]["0.95"] == sum(
        1 for l in range(3, 13) if l != 6 and scr.in_depth_band(l, n))
    assert sel["eligibility"] == scr.ELIGIBILITY_RULE
    for fn in (scr.layer_eligible, scr.select_layers, scr.rule_block):
        assert "leak" not in " ".join(inspect.signature(fn).parameters), fn.__name__
    assert scr.layer_eligible(layer=5, auroc=0.999, auroc_threshold=0.95) == []
    assert scr.layer_eligible(layer=0, auroc=1.0, auroc_threshold=0.95) == ["layer 0 is never eligible"]
    assert "leakage_max" not in scr.rule_block() and scr.rule_block()["eligibility"] == scr.ELIGIBILITY_RULE
    assert "AUROC ONLY" in scr.ELIGIBILITY_RULE and "diagnostic" in scr.ELIGIBILITY_RULE
    assert "AND leakage" not in scr.ELIGIBILITY_RULE and not hasattr(scr, "DEFAULT_LEAKAGE_MAX")


def test_argmax_ties_break_to_smaller_p():
    n = 20
    spec = {6: {0.3: (0.97, 1e-4), 0.4: (0.98, 1e-4), 0.5: (0.98, 1e-4), 0.6: (0.98, 1e-4)},
            7: {0.3: (0.97, 1e-4), 0.4: (0.96, 1e-4), 0.5: (0.97, 1e-4), 0.6: (0.90, 1e-4)},
            8: {0.3: (0.97, 1e-4)}}
    sel = scr.select_layers(_rows(spec, n), n_blocks=n)
    by = {pl["layer"]: pl for pl in sel["per_layer"]}
    assert by[6]["p_star"] == 0.4 and by[7]["p_star"] == 0.3
    assert sel["chosen"]["nullspace_ratios"] == {"6": 0.4, "7": 0.3, "8": 0.3}
    # determinism: shuffled row order gives the same selection
    import random
    rows = _rows(spec, n)
    random.Random(0).shuffle(rows)
    assert scr.select_layers(rows, n_blocks=n) == sel
    with pytest.raises(ValueError, match="grid"):
        scr.select_layers(rows + [{"layer": 6, "p": 0.7, "auroc": 1.0, "leakage": 0.0}],
                          n_blocks=n)


def test_eligible_counts_recorded_at_three_thresholds():
    n = 40
    spec = {l: {0.5: (a, 1e-4)} for l, a in
            zip(range(6, 27), [0.91, 0.96, 0.99, 0.985, 0.93, 0.951, 0.97] * 3)}
    sel = scr.select_layers(_rows(spec, n), n_blocks=n)
    c = sel["eligible_counts"]
    assert set(c) == {"0.90", "0.95", "0.98"} and c["0.90"] >= c["0.95"] >= c["0.98"]
    assert c["0.98"] == 6 and c["0.95"] == 15 and c["0.90"] == 21


def test_a5_fallback_takes_top6_and_flags():
    n = 32                                          # band 5..20
    spec = {l: {0.5: (0.92 + 0.001 * l, 1e-4)} for l in range(5, 21)}
    spec[9] = {0.5: (0.99, 1e-4)}                   # eligible
    spec[12] = {0.5: (0.995, 5e-3)}                 # high leakage: eligible all the same
    sel = scr.select_layers(_rows(spec, n), n_blocks=n)
    fb = sel["chosen"]["fallback_a5"]
    assert fb and fb["taken"] and "FLAG" in fb
    assert fb["n_eligible_at_threshold"] == 2
    assert sel["chosen"]["n_layers"] == 6 and 12 in sel["chosen"]["layers"]
    assert 9 in sel["chosen"]["layers"]
    assert sel["chosen"]["layers"] == sorted(sel["chosen"]["layers"])
    # top-6 by AUROC among band layers: 12, 9, then 20,19,18,17 — AUROC only
    assert sel["chosen"]["layers"] == [9, 12, 17, 18, 19, 20]
    assert "leakage" not in fb["rule"] or "AUROC only" in fb["rule"]


def test_a5_stop_raises_below_three_at_0p90():
    n = 32
    spec = {l: {0.5: (0.80, 1e-4)} for l in range(5, 21)}
    spec[9] = {0.5: (0.99, 1e-4)}
    spec[10] = {0.5: (0.93, 1e-4)}
    with pytest.raises(scr.A5StopError, match="only 2 reach 0.9"):
        scr.select_layers(_rows(spec, n), n_blocks=n)
    # a high-leakage layer with passing AUROC DOES count (AUROC only) —
    # three now reach 0.90, so the STOP is not raised; the fallback is taken
    spec[11] = {0.5: (0.99, 5e-3)}
    sel = scr.select_layers(_rows(spec, n), n_blocks=n)
    assert sel["chosen"]["fallback_a5"]["taken"] and 11 in sel["chosen"]["layers"]


# ---------------------------------------------------------------- the split


def _two_source(prompts, boundary):
    return [{"name": "src_a", "start": 0, "end": boundary},
            {"name": "src_b", "start": boundary, "end": len(prompts)}]


def test_normalise_prompt_is_the_shared_identity():
    assert scr.normalise_prompt("  Héllo, WORLD!!  ﬁne\tthing ") == "héllo world fine thing"
    assert scr.normalise_prompt("How do I do X?") == scr.normalise_prompt("how do i do x")
    assert scr.normalise_prompt("a-b") == "a b"


def test_split_groups_duplicate_prompts_onto_one_side():
    """The SAME prompt at two indices in two different source slices (with
    case/punctuation variation) lands on ONE side, both occurrences."""
    prompts = [f"prompt number {i}" for i in range(40)]
    prompts[3] = "Explain how to pick a lock."
    prompts[27] = "explain how to pick a lock"
    prompts[31] = "EXPLAIN, how to pick a lock!!"
    split = scr.split_side_content_grouped(
        prompts, side="harmful", slices=_two_source(prompts, 20), heldout_frac=0.3, seed=0)
    dup = {3, 27, 31}
    held = set(split.heldout_idx)
    assert dup <= held or not (dup & held), "duplicate group was split across sides"
    assert split.n_groups == 38 and split.n_groups_spanning_sources == 1
    # try several seeds so the group lands on BOTH sides at least once each,
    # proving the invariant is grouping and not luck
    sides_seen = set()
    for seed in range(12):
        s = scr.split_side_content_grouped(
            prompts, side="harmful", slices=_two_source(prompts, 20),
            heldout_frac=0.3, seed=seed)
        h = set(s.heldout_idx)
        assert dup <= h or not (dup & h), seed
        sides_seen.add("held" if dup <= h else "fit")
    assert sides_seen == {"held", "fit"}


def test_split_is_content_disjoint():
    prompts = [f"row {i % 13}" for i in range(60)]          # heavy duplication
    split = scr.split_side_content_grouped(prompts, side="harmful", heldout_frac=0.1)
    fit = {scr.normalise_prompt(prompts[i]) for i in split.fit_idx}
    held = {scr.normalise_prompt(prompts[i]) for i in split.heldout_idx}
    assert not (fit & held) and split.content_disjoint_verified
    assert split.index_disjoint_verified
    # the index check alone would have accepted a leaking split: show one
    naive_fit, naive_held = list(range(54)), list(range(54, 60))
    assert not (set(naive_fit) & set(naive_held))                       # passes
    assert {scr.normalise_prompt(prompts[i]) for i in naive_fit} & \
           {scr.normalise_prompt(prompts[i]) for i in naive_held}       # yet leaks


def test_split_preserves_every_row_and_multiplicity():
    prompts = [f"p{i % 7}" for i in range(50)] + ["unique one", "unique two"]
    split = scr.split_side_content_grouped(prompts, side="benign", heldout_frac=0.2)
    assert sorted(split.fit_idx + split.heldout_idx) == list(range(len(prompts)))
    from collections import Counter
    assert (Counter(prompts[i] for i in split.fit_idx)
            + Counter(prompts[i] for i in split.heldout_idx)) == Counter(prompts)
    assert split.n_rows == len(prompts)


def test_split_records_realised_per_source_counts():
    """Exact 10 % is unreachable with indivisible groups; the realised
    per-source counts are recorded and nothing hardcodes the nominal."""
    prompts = ["dup"] * 8 + [f"a{i}" for i in range(22)] + [f"b{i}" for i in range(37)]
    slices = _two_source(prompts, 30)
    split = scr.split_side_content_grouped(prompts, side="harmful", slices=slices,
                                           heldout_frac=0.10, seed=0)
    rb = split.realised_by_source
    assert set(rb) == {"src_a", "src_b"}
    assert rb["src_a"]["rows"] == 30 and rb["src_b"]["rows"] == 37
    assert rb["src_a"]["target_heldout"] == 3 and rb["src_b"]["target_heldout"] == 4
    for name in rb:
        assert rb[name]["heldout"] >= rb[name]["target_heldout"]
        assert rb[name]["fit"] + rb[name]["heldout"] == rb[name]["rows"]
    assert len(split.heldout_idx) == rb["src_a"]["heldout"] + rb["src_b"]["heldout"]
    other = scr.split_side_content_grouped(prompts, side="benign", heldout_frac=0.10)
    payload = scr.split_v1_payload(other, split)
    assert payload["n_heldout_harmful"] == len(split.heldout_idx)
    assert payload["n_fit_harmful"] == len(split.fit_idx)
    assert payload["content_disjoint_verified"] is True
    assert payload["convention"] == scr.SPLIT_CONVENTION
    assert "realised_by_source" in scr.split_summary(payload)
    assert "sides" not in scr.split_summary(payload)


def test_split_is_deterministic():
    prompts = [f"prompt {i}" for i in range(80)]
    a = scr.split_side_content_grouped(prompts, side="benign", heldout_frac=0.1, seed=0)
    b = scr.split_side_content_grouped(prompts, side="benign", heldout_frac=0.1, seed=0)
    assert a.heldout_idx == b.heldout_idx and a.sha256 == b.sha256
    # input order independence: the held-out IDENTITIES are the same set
    import random
    shuffled = list(prompts)
    random.Random(1).shuffle(shuffled)
    c = scr.split_side_content_grouped(shuffled, side="benign", heldout_frac=0.1, seed=0)
    assert set(c.heldout_identities) == set(a.heldout_identities)
    # the seed genuinely reseeds
    d = scr.split_side_content_grouped(prompts, side="benign", heldout_frac=0.1, seed=1)
    assert set(d.heldout_identities) != set(a.heldout_identities)
    with pytest.raises(ValueError, match="heldout_frac"):
        scr.split_side_content_grouped(prompts, side="benign", heldout_frac=1.0)
    with pytest.raises(ValueError, match="slices"):
        scr.split_side_content_grouped(prompts, side="benign",
                                       slices=[{"name": "x", "start": 0, "end": 10}])


# ---------------------------------------------------------------- the checkpoint


def _capture_binding(side="benign", **over):
    b = dict(model="TEST/tiny", model_revision="abc", dtype="float32",
             attn_implementation="sdpa", device="cpu", site="pre", side=side,
             list_sha256=("b" if side == "benign" else "h") * 64, n_rows=20,
             batch_size=8, code_sha="c" * 40)
    b.update(over)
    return scr.capture_binding(**b)


def _binding(**over):
    """A screen_binding over two synthetic capture bindings. An ``over`` key
    that is a CAPTURE key (dtype, batch_size, model_revision, ...) lands in
    BOTH sides' capture bindings; ``code_sha`` lands in both and at the top."""
    cap_keys = set(scr.CAPTURE_BINDING_KEYS) - {"side"}
    cap_over = {k: v for k, v in over.items() if k in cap_keys}
    top = {k: v for k, v in over.items() if k not in cap_keys or k == "code_sha"}
    b = dict(capture={s: _capture_binding(s, **cap_over) for s in ("benign", "harmful")},
             n_blocks=4, hidden_dim=12, split_sha256="s" * 64,
             layers=[0, 1, 2, 3], p_grid=(0.3, 0.5), lambda_reg=10.0,
             depth_band=(0.15, 0.65), code_sha="c" * 40)
    b.update(top)
    return scr.screen_binding(**b)


def test_row_checkpoint_recovers_a_torn_tail_and_refuses_interior_corruption(tmp_path):
    """A write interrupted mid-append must not block every later
    resume. The complete records before an incomplete FINAL record are
    recovered and the tail truncated (its cell is recomputed); an
    unparseable INTERIOR record is corruption and refuses; a complete but
    unterminated final record counts as torn (the next append would glue
    onto it)."""
    path = tmp_path / "screen_rows.jsonl"
    rows = [{"layer": l, "p": 0.3, "auroc": 0.9} for l in range(3)]
    ck = scr.RowCheckpoint(path, _binding(), resume=False)
    for r in rows:
        ck.append(r)
    torn = json.dumps({"layer": 3, "p": 0.3, "auroc": 0.95})[:-9]      # cut mid-record
    with path.open("a") as f:
        f.write(torn)
    again = scr.RowCheckpoint(path, _binding(), resume=True)
    assert again.rows == rows and again.truncated_torn_tail
    assert not path.read_bytes().endswith(torn.encode())               # tail gone
    again.append({"layer": 3, "p": 0.3, "auroc": 0.95})                # recomputed cell
    lines = path.read_text().splitlines()
    assert len(lines) == 5 and all(json.loads(l) for l in lines)       # header + 4 clean
    assert scr.RowCheckpoint(path, _binding(), resume=True).rows == rows + [
        {"layer": 3, "p": 0.3, "auroc": 0.95}]
    # complete JSON but no trailing newline: torn, not a record
    with path.open("a") as f:
        f.write(json.dumps({"layer": 4, "p": 0.3, "auroc": 0.1}))
    assert len(scr.RowCheckpoint(path, _binding(), resume=True).rows) == 4
    assert path.read_bytes().endswith(b"\n")
    # an INTERIOR record that does not parse is corruption: refuse
    lines = path.read_text().splitlines()
    lines[2] = lines[2][:-4]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="interior"):
        scr.RowCheckpoint(path, _binding(), resume=True)
    # a checkpoint with no binding header (pre-binding format) refuses
    path.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(scr.CheckpointMismatch, match="header"):
        scr.RowCheckpoint(path, _binding(), resume=True)
    # an empty or absent file is a fresh start, header written
    path.write_text("")
    assert scr.RowCheckpoint(path, _binding(), resume=True).rows == []
    assert scr.RowCheckpoint.HEADER_KEY in json.loads(path.read_text().splitlines()[0])
    # resume=False discards whatever was there
    ck = scr.RowCheckpoint(path, _binding(), resume=False)
    ck.append(rows[0])
    assert len(path.read_text().splitlines()) == 2


def test_row_checkpoint_refuses_another_binding(tmp_path):
    path = tmp_path / "screen_rows.jsonl"
    ck = scr.RowCheckpoint(path, _binding(), resume=False)
    ck.append({"layer": 0, "p": 0.3, "auroc": 0.9})
    assert scr.RowCheckpoint(path, _binding(), resume=True).rows          # same inputs: fine
    for change, key in (({"split_sha256": "t" * 64}, "split_sha256"),
                        ({"model_revision": "def"}, "model_revision"),
                        ({"lambda_reg": 100.0}, "lambda_reg"),
                        ({"site": "post"}, "site"),
                        ({"code_sha": "d" * 40}, "code_sha"),
                        ({"layers": [1, 2]}, "layers"),
                        # the rows inherit the captures' precision,
                        # kernel, device and width — named to the leaf
                        ({"dtype": "bfloat16"}, r"capture\.benign\.dtype"),
                        ({"attn_implementation": "eager"}, r"capture\.harmful\.attn_implementation"),
                        ({"device": "cuda"}, "device"),
                        ({"batch_size": 32}, "batch_size")):
        with pytest.raises(scr.CheckpointMismatch, match=key):
            scr.RowCheckpoint(path, _binding(**change), resume=True)
    # the digest is a function of the canonical binding only
    assert scr.binding_sha256(_binding()) == scr.binding_sha256(
        {**_binding(), "p_grid": [0.3, 0.5]})
    assert scr.binding_sha256(_binding()) != scr.binding_sha256(_binding(lambda_reg=100.0))
    with pytest.raises(ValueError, match="sha256"):
        _capture_binding(list_sha256=None)


# ---------------------------------------------------------------- rho + artifact


def test_rho_by_layer_quantiles_and_dtype():
    g = torch.Generator().manual_seed(5)
    H = torch.randn(50, 16, generator=g)
    M = 0.01 * torch.randn(16, 16, generator=g)
    rho = scr.rho_by_layer({3: H}, {3: M}, scale=2.0, model_dtype=torch.bfloat16)
    r = rho[3]
    assert set(r["rho_fp32"]) == {"p05", "p50", "p95"}
    assert r["rho_fp32"]["p05"] <= r["rho_fp32"]["p50"] <= r["rho_fp32"]["p95"]
    want = 2.0 * torch.linalg.vector_norm(H @ M, dim=-1) / torch.linalg.vector_norm(H, dim=-1)
    assert r["rho_fp32"]["p50"] == pytest.approx(
        float(torch.quantile(want.double(), 0.5)), rel=1e-4)
    assert r["model_dtype"] == "bfloat16" and r["n_rows"] == 50
    assert abs(r["rho_model_dtype"]["p50"] - r["rho_fp32"]["p50"]) / r["rho_fp32"]["p50"] < 0.05
    # linear in scale
    rho1 = scr.rho_by_layer({3: H}, {3: M}, scale=1.0)[3]
    assert r["rho_fp32"]["p50"] == pytest.approx(2 * rho1["rho_fp32"]["p50"], rel=1e-6)


def test_screen_json_schema_round_trip(tmp_path):
    benign, harmful = _planted(d=12)
    rows = scr.screen_layers_and_p(
        benign_fit={l: benign[:150] for l in range(4)},
        harmful_fit={l: harmful[:40] for l in range(4)},
        benign_held={l: benign[150:] for l in range(4)},
        harmful_held={l: harmful[40:] for l in range(4)},
        layers=list(range(4)), n_blocks=4, p_grid=(0.3, 0.5), lambda_reg=10.0)
    sel = scr.select_layers(rows, n_blocks=4, depth_band=(0.15, 0.65),
                            auroc_threshold=0.5, p_grid=(0.3, 0.5), min_eligible=1)
    b_split = scr.split_side_content_grouped([f"b{i}" for i in range(20)], side="benign")
    h_split = scr.split_side_content_grouped([f"h{i}" for i in range(20)], side="harmful")
    payload = scr.build_screen_payload(
        model_key="tiny", model="TEST/tiny", model_revision="abc", n_blocks=4,
        hidden_dim=12, site="pre",
        rule=scr.rule_block(p_grid=(0.3, 0.5), auroc_threshold=0.5),
        split=scr.split_summary(scr.split_v1_payload(b_split, h_split)),
        lists={"benign_n": 20, "harmful_n": 20}, rows=rows, selection=sel,
        capture={"ids_match_serving_render": True}, acts_dir=str(tmp_path / "acts"),
        binding=_binding())
    scr.check_screen_payload(payload)
    for key in scr.SCREEN_REQUIRED_KEYS:
        assert key in payload
    # the rule block says in words that eligibility is AUROC only
    assert payload["rule"]["eligibility"] == scr.ELIGIBILITY_RULE
    assert "leakage_max" not in payload["rule"] and "diagnostic" in payload["rule"]["leakage"]
    assert "eligible_counts_auroc_only" not in payload
    # a screen must say what it was computed under: the binding is required
    # at build time and at read time (the fit refuses one without it)
    assert "inputs_binding" in scr.SCREEN_REQUIRED_KEYS
    assert payload["inputs_binding"] == _binding()
    assert payload["inputs_sha256"] == scr.binding_sha256(_binding())
    with pytest.raises(ValueError, match="screen_binding"):
        scr.build_screen_payload(
            model_key="tiny", model="TEST/tiny", model_revision="abc", n_blocks=4,
            hidden_dim=12, site="pre", rule=payload["rule"], split=payload["split"],
            lists=payload["lists"], rows=rows, selection=sel, capture=payload["capture"],
            acts_dir=None, binding={"capture_batch_size": 8})
    with pytest.raises(ValueError, match="inputs_binding"):
        scr.check_screen_payload({k: v for k, v in payload.items() if k != "inputs_binding"})
    assert payload["score"]["expression"] == scr.SCORE_EXPRESSION
    assert payload["score"]["rank1_agreement_max_rel"] <= scr.DEFAULT_RANK1_TOL
    out = tmp_path / "screen_f9.json"
    final = scr.write_fingerprinted_json(out, payload, code_sha="deadbeef")
    assert (tmp_path / "screen_f9.json.COMPLETE").read_text().strip() == final["sha256"]
    back = scr.read_fingerprinted_json(out)
    assert back["chosen"] == sel["chosen"] and back["code_sha"] == "deadbeef"
    assert back["curve"][0]["auroc"] == rows[0]["auroc"]
    # tampering refuses
    edited = json.loads(out.read_text())
    edited["chosen"]["layers"] = [1, 2, 3]
    out.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="sha256"):
        scr.read_fingerprinted_json(out)
    (tmp_path / "screen_f9.json.COMPLETE").unlink()
    with pytest.raises(FileNotFoundError, match="COMPLETE"):
        scr.read_fingerprinted_json(out)
    # schema checks: layer 0 / off-grid ratio / out-of-band refuse
    bad = dict(payload); bad["chosen"] = {"layers": [0], "nullspace_ratios": {"0": 0.3}}
    with pytest.raises(ValueError, match="layer 0"):
        scr.check_screen_payload(bad)
    bad["chosen"] = {"layers": [2], "nullspace_ratios": {"2": 0.7}}
    with pytest.raises(ValueError, match="not in"):
        scr.check_screen_payload(bad)


# ---------------------------------------------------------------- end to end on the fixture


def test_run_screen_end_to_end_on_tiny_model(tiny_llama, tmp_path, monkeypatch):
    """The CLI's run_screen on the tiny fixture: split, capture with the ids
    assertion, screen every block, select (permissive thresholds — the
    random model separates nothing), write the fingerprinted artifact and
    resume idempotently."""
    import alphasteer_screen as cli

    model, tok = tiny_llama
    benign = [f"Write a friendly note about topic number {i}." for i in range(30)]
    harmful = [f"Describe a harmful plan number {i} in detail." for i in range(12)]
    bp, hp = tmp_path / "benign.json", tmp_path / "harmful.json"
    bp.write_text(json.dumps(benign)); hp.write_text(json.dumps(harmful))
    out_dir = tmp_path / "screen"
    cfg = {"model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
           "model_revision": None, "dtype": "float32", "layer": 1,
           "class_b": {"benign_prompts_path": str(bp), "harmful_prompts_path": str(hp),
                       "min_prompts": 1, "batch_size": 8,
                       "alphasteer": {"layer_rule": "f9_auroc",
                                      "screen_path": str(out_dir / "screen_f9.json"),
                                      "auroc_threshold": 0.51,
                                      "heldout_frac": 0.2, "capture_batch_size": 8,
                                      "verify_sample": 8, "depth_band": [0.4, 0.6]}}}
    import fit_class_b as fcb

    rule = fcb.alphasteer_layer_rule(cfg)
    # the random tiny model may not clear even 0.51 on 1-2 held-out rows;
    # make selection permissive for the e2e mechanics
    monkeypatch.setattr(scr, "A5_MIN_ELIGIBLE", 1)
    real_select = scr.select_layers
    monkeypatch.setattr(scr, "select_layers", lambda rows, **kw: real_select(
        rows, **{**kw, "min_eligible": 1, "stop_auroc": 0.0}))
    rc = cli.run_screen(cfg, rule, model_key="tiny", out=out_dir / "screen_f9.json",
                        acts_dir=out_dir / "acts", split_out=out_dir / "split_v1.json",
                        device="cpu", model_and_tok=(model, tok))
    assert rc == 0
    payload = scr.read_fingerprinted_json(out_dir / "screen_f9.json")
    scr.check_screen_payload(payload)
    assert payload["n_blocks"] == 2 and payload["hidden_dim"] == 16
    assert len(payload["curve"]) == 2 * 4                 # every block x every p
    assert payload["capture"]["benign"]["ids_match_serving_render"] is True
    assert payload["capture"]["benign"]["n_verified"] == 8
    assert payload["split"]["n_heldout_benign"] + payload["split"]["n_fit_benign"] == 30
    assert payload["chosen"]["layers"] == [1]             # 2 blocks: only layer 1 is in band
    assert (out_dir / "acts" / "benign" / "L000.pt").is_file()
    assert (out_dir / "acts" / "benign" / "capture_done.json").is_file()
    assert (out_dir / "figs" / "auroc_vs_depth.png").is_file()
    split = scr.read_fingerprinted_json(out_dir / "split_v1.json")
    assert split["content_disjoint_verified"] is True
    lines = (out_dir / "screen_rows.jsonl").read_text().splitlines()
    assert len(lines) == 1 + 8                            # binding header + every cell
    header = json.loads(lines[0])
    assert header[scr.RowCheckpoint.HEADER_KEY] == payload["inputs_binding"]
    assert header["binding_sha256"] == payload["inputs_sha256"]
    assert payload["inputs_binding"]["split_sha256"] == split["split_sha256"]
    assert payload["inputs_binding"]["code_sha"] == payload["code_sha"]
    sent = payload["capture"]["benign"]
    assert sent["binding"]["batch_size"] == 8 and sent["binding"]["side"] == "benign"
    assert sent["binding"]["list_sha256"] == payload["lists"]["benign_sha256"]
    # the rows inherit BOTH capture bindings, precision and kernel included
    assert payload["inputs_binding"]["capture"]["benign"] == sent["binding"]
    assert payload["inputs_binding"]["capture"]["harmful"] == payload["capture"]["harmful"]["binding"]
    assert sent["binding"]["dtype"] == "float32" and sent["binding"]["device"] == "cpu"
    assert sent["binding"]["attn_implementation"] == model.config._attn_implementation
    assert set(payload["inputs_binding"]) == set(scr.SCREEN_BINDING_KEYS)
    # idempotent: a second call under the SAME inputs finds the complete
    # artifact and does nothing
    forwards = []
    h = model.register_forward_pre_hook(lambda *_: forwards.append(1))
    try:
        assert cli.run_screen(cfg, rule, model_key="tiny", out=out_dir / "screen_f9.json",
                              acts_dir=out_dir / "acts", split_out=out_dir / "split_v1.json",
                              device="cpu", model_and_tok=(model, tok), resume=True) == 0
        # ... but under OTHER inputs a complete artifact is not "nothing to
        # do": it REFUSES, naming the input, with no forward and the artifact
        # untouched (the sbatch runs this step before every fit)
        import copy
        model16 = copy.deepcopy(model).to(torch.bfloat16)      # the hook is copied with it
        for kw, key in (({"batch_size": 4}, "batch_size"),
                        ({"model_and_tok": (model16, tok)}, "dtype")):
            with pytest.raises(scr.CheckpointMismatch, match=key):
                cli.run_screen(cfg, rule, model_key="tiny", out=out_dir / "screen_f9.json",
                               acts_dir=out_dir / "acts", split_out=out_dir / "split_v1.json",
                               device="cpu", resume=True,
                               **{"model_and_tok": (model, tok), **kw})
    finally:
        h.remove()
    assert forwards == []
    assert scr.read_fingerprinted_json(out_dir / "screen_f9.json")["sha256"] == payload["sha256"]


def _interrupted_screen(cli, fcb, model, tok, cfg, out_dir, monkeypatch, *, after=3):
    """run_screen on the tiny fixture, killed after ``after`` cells are
    checkpointed (the real append runs, then the kill) — a preemption."""
    real = scr.RowCheckpoint.append
    n = {"appends": 0}

    def dying_append(self, row):
        real(self, row)
        n["appends"] += 1
        if n["appends"] >= after:
            raise RuntimeError("simulated preemption")

    monkeypatch.setattr(scr.RowCheckpoint, "append", dying_append)
    rule = fcb.alphasteer_layer_rule(cfg)
    with pytest.raises(RuntimeError, match="preemption"):
        cli.run_screen(cfg, rule, model_key="tiny", out=out_dir / "screen_f9.json",
                       acts_dir=out_dir / "acts", split_out=out_dir / "split_v1.json",
                       device="cpu", model_and_tok=(model, tok))
    monkeypatch.setattr(scr.RowCheckpoint, "append", real)
    return n["appends"]


def test_resume_refuses_cells_solved_under_another_split(tiny_llama, tmp_path, monkeypatch):
    """A screen interrupted after three cells, then resumed with
    a different split seed, must NOT publish those three cells under the new
    split — the checkpoint header binds them to the old split sha and the
    resume REFUSES (no artifact is written). The capture, whose inputs did
    not change, is reused. The same seed resumes normally, solving only the
    missing cells — including the one behind a torn final record."""
    import alphasteer_screen as cli
    import fit_class_b as fcb

    model, tok = tiny_llama
    benign = [f"Write a friendly note about topic number {i}." for i in range(30)]
    harmful = [f"Describe a harmful plan number {i} in detail." for i in range(12)]
    bp, hp = tmp_path / "benign.json", tmp_path / "harmful.json"
    bp.write_text(json.dumps(benign)); hp.write_text(json.dumps(harmful))
    out_dir = tmp_path / "screen"
    cfg = {"model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
           "model_revision": None, "dtype": "float32", "layer": 1,
           "class_b": {"benign_prompts_path": str(bp), "harmful_prompts_path": str(hp),
                       "min_prompts": 1, "batch_size": 8,
                       "alphasteer": {"layer_rule": "f9_auroc",
                                      "screen_path": str(out_dir / "screen_f9.json"),
                                      "auroc_threshold": 0.51,
                                      "heldout_frac": 0.2, "capture_batch_size": 8,
                                      "verify_sample": 8, "depth_band": [0.4, 0.6],
                                      "split_seed": 0}}}
    monkeypatch.setattr(scr, "A5_MIN_ELIGIBLE", 1)
    real_select = scr.select_layers
    monkeypatch.setattr(scr, "select_layers", lambda rows, **kw: real_select(
        rows, **{**kw, "min_eligible": 1, "stop_auroc": 0.0}))

    assert _interrupted_screen(cli, fcb, model, tok, cfg, out_dir, monkeypatch) == 3
    rows_path = out_dir / "screen_rows.jsonl"
    assert len(rows_path.read_text().splitlines()) == 1 + 3
    assert not (out_dir / "screen_f9.json").exists()
    old_split = scr.read_fingerprinted_json(out_dir / "split_v1.json")["split_sha256"]

    # the split seed changes -> resume refuses, names the split, writes nothing
    cfg_new = json.loads(json.dumps(cfg))
    cfg_new["class_b"]["alphasteer"]["split_seed"] = 7
    rule_new = fcb.alphasteer_layer_rule(cfg_new)
    forwards = []
    h = model.register_forward_pre_hook(lambda *_: forwards.append(1))
    try:
        with pytest.raises(scr.CheckpointMismatch, match="split_sha256"):
            cli.run_screen(cfg_new, rule_new, model_key="tiny", out=out_dir / "screen_f9.json",
                           acts_dir=out_dir / "acts", split_out=out_dir / "split_v1.json",
                           device="cpu", model_and_tok=(model, tok), resume=True)
    finally:
        h.remove()
    assert forwards == [], "the capture's inputs did not change: reused, no forward"
    assert not (out_dir / "screen_f9.json").exists()
    assert len(rows_path.read_text().splitlines()) == 1 + 3          # untouched
    new_split = scr.read_fingerprinted_json(out_dir / "split_v1.json")["split_sha256"]
    assert new_split != old_split

    # a foreign capture binding refuses too (model revision moved)
    header = json.loads(rows_path.read_text().splitlines()[0])[scr.RowCheckpoint.HEADER_KEY]
    with pytest.raises(scr.CheckpointMismatch, match="model_revision"):
        scr.capture_or_load_side(
            model, tok, benign, side="benign", acts_dir=out_dir / "acts",
            binding=scr.capture_binding(
                **{**scr.model_load_identity(model, name=cfg["model"], cfg_revision=None),
                   "model_revision": "some-other-commit"},
                site="pre", side="benign",
                list_sha256=header["capture"]["benign"]["list_sha256"], n_rows=30,
                batch_size=8, code_sha=scr.repo_code_sha()))

    # the SAME seed resumes: a torn final record is truncated and recomputed,
    # only the 5 missing cells are solved, the artifact carries all 8
    with rows_path.open("a") as f:
        f.write(json.dumps({"layer": 0, "p": 0.6, "auroc": 0.5})[:-6])
    appends = []
    real_append = scr.RowCheckpoint.append
    monkeypatch.setattr(scr.RowCheckpoint, "append",
                        lambda self, row: (appends.append(row), real_append(self, row))[1])
    rule = fcb.alphasteer_layer_rule(cfg)
    assert cli.run_screen(cfg, rule, model_key="tiny", out=out_dir / "screen_f9.json",
                          acts_dir=out_dir / "acts", split_out=out_dir / "split_v1.json",
                          device="cpu", model_and_tok=(model, tok), resume=True) == 0
    assert len(appends) == 5 and (0, 0.6) in {(r["layer"], r["p"]) for r in appends}
    payload = scr.read_fingerprinted_json(out_dir / "screen_f9.json")
    assert len(payload["curve"]) == 8
    assert payload["inputs_binding"]["split_sha256"] == old_split == payload["split"]["split_sha256"]
    lines = rows_path.read_text().splitlines()
    assert len(lines) == 1 + 8 and all(json.loads(l) for l in lines)


def _tiny_screen_cfg(tmp_path, out_dir):
    benign = [f"Write a friendly note about topic number {i}." for i in range(30)]
    harmful = [f"Describe a harmful plan number {i} in detail." for i in range(12)]
    bp, hp = tmp_path / "benign.json", tmp_path / "harmful.json"
    bp.write_text(json.dumps(benign)); hp.write_text(json.dumps(harmful))
    cfg = {"model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
           "model_revision": None, "dtype": "float32", "layer": 1,
           "class_b": {"benign_prompts_path": str(bp), "harmful_prompts_path": str(hp),
                       "min_prompts": 1, "batch_size": 8,
                       "alphasteer": {"layer_rule": "f9_auroc",
                                      "screen_path": str(out_dir / "screen_f9.json"),
                                      "auroc_threshold": 0.51,
                                      "heldout_frac": 0.2, "capture_batch_size": 8,
                                      "verify_sample": 8, "depth_band": [0.4, 0.6]}}}
    return cfg, benign


def test_resume_refuses_reusing_a_capture_at_another_precision(tiny_llama, tmp_path, monkeypatch):
    """Benign capture completes in fp32, the run dies before the
    harmful capture, the operator changes the model dtype and resumes. The
    benign sentinel's binding names the RESOLVED capture dtype, so the resume
    REFUSES (naming dtype) before any forward — never a screen whose two
    sides were captured at different precisions under one fingerprint. The
    premise is measured on this fixture, not assumed: fp32 and bf16 captures
    of the same prompts are NOT equal (3.3e-3 relative
    L2 at d=16). The attention kernel binds the same way; a resume at the
    SAME precision reuses benign, captures harmful and completes."""
    import copy

    import alphasteer_screen as cli
    import fit_class_b as fcb

    model, tok = tiny_llama
    out_dir = tmp_path / "screen"
    cfg, benign = _tiny_screen_cfg(tmp_path, out_dir)
    monkeypatch.setattr(scr, "A5_MIN_ELIGIBLE", 1)
    real_select = scr.select_layers
    monkeypatch.setattr(scr, "select_layers", lambda rows, **kw: real_select(
        rows, **{**kw, "min_eligible": 1, "stop_auroc": 0.0}))
    rule = fcb.alphasteer_layer_rule(cfg)
    run = dict(model_key="tiny", out=out_dir / "screen_f9.json", acts_dir=out_dir / "acts",
               split_out=out_dir / "split_v1.json", device="cpu")

    # the premise: precision is an input of the bytes on this very fixture
    model16 = copy.deepcopy(model).to(torch.bfloat16)
    texts = [scr.build_chat_prompt(tok, p, system=None) for p in benign[:8]]
    a32, _ = scr.capture_all_layers(model, tok, texts, site="pre", batch_size=8,
                                    layers=[1], verify_sample=1)
    a16, _ = scr.capture_all_layers(model16, tok, texts, site="pre", batch_size=8,
                                    layers=[1], verify_sample=1)
    rel = float((a32[1] - a16[1]).norm() / a32[1].norm())
    assert not torch.equal(a32[1], a16[1]) and rel > 1e-4, rel
    ident32 = scr.model_load_identity(model, name=cfg["model"], cfg_revision=None)
    ident16 = scr.model_load_identity(model16, name=cfg["model"], cfg_revision=None)
    assert ident32["dtype"] == "float32" and ident16["dtype"] == "bfloat16"
    assert {k: v for k, v in ident32.items() if k != "dtype"} == {
        k: v for k, v in ident16.items() if k != "dtype"}      # ONLY the dtype moved

    # benign completes; the run dies before the harmful capture
    real_capture = scr.capture_all_layers
    calls = []

    def dying(m, t, texts_, **kw):
        calls.append(len(texts_))
        if len(calls) == 2:
            raise RuntimeError("simulated preemption")
        return real_capture(m, t, texts_, **kw)

    monkeypatch.setattr(scr, "capture_all_layers", dying)
    with pytest.raises(RuntimeError, match="preemption"):
        cli.run_screen(cfg, rule, model_and_tok=(model, tok), **run)
    monkeypatch.setattr(scr, "capture_all_layers", real_capture)
    assert calls == [30, 12]
    sent = json.loads((out_dir / "acts" / "benign" / "capture_done.json").read_text())
    assert sent["binding"]["dtype"] == "float32"
    assert not (out_dir / "acts" / "harmful").exists()

    # resume at bf16: refused, naming dtype, before any forward; harmful is
    # never captured and no artifact appears
    forwards = []
    h = model16.register_forward_pre_hook(lambda *_: forwards.append(1))
    try:
        with pytest.raises(scr.CheckpointMismatch, match=r"dtype: checkpoint='float32' requested='bfloat16'"):
            cli.run_screen(cfg, rule, model_and_tok=(model16, tok), resume=True, **run)
    finally:
        h.remove()
    assert forwards == []
    assert not (out_dir / "acts" / "harmful").exists()
    assert not (out_dir / "screen_f9.json").exists()
    # the attention kernel binds the same way (the seam, directly)
    with pytest.raises(scr.CheckpointMismatch, match="attn_implementation"):
        scr.capture_or_load_side(model, tok, benign, side="benign", acts_dir=out_dir / "acts",
                                 binding={**sent["binding"], "attn_implementation": "eager"})

    # the same precision resumes: benign reused, harmful captured, screen complete
    monkeypatch.setattr(scr, "capture_all_layers",
                        lambda m, t, texts_, **kw: (calls.append(len(texts_)),
                                                    real_capture(m, t, texts_, **kw))[1])
    assert cli.run_screen(cfg, rule, model_and_tok=(model, tok), resume=True, **run) == 0
    assert calls == [30, 12, 12]                          # only harmful was (re)captured
    payload = scr.read_fingerprinted_json(out_dir / "screen_f9.json")
    assert payload["inputs_binding"]["capture"]["benign"] == sent["binding"]
    assert payload["inputs_binding"]["capture"]["harmful"]["dtype"] == "float32"


def test_leading_token_policy_cannot_reach_the_capture_seam(tiny_llama):
    """The reason CAPTURE_INPUTS EXEMPTS ``leading_special_tokens`` (main's
    loader parameter) instead of binding it, proved rather than
    asserted: the policy installs the ``<bos> $A`` post-processor, which a
    tokenizer applies only under ``add_special_tokens=True``; the capture seam
    tokenizes chat-rendered texts with ``add_special_tokens=False``
    (``CAPTURE_ADD_SPECIAL_TOKENS``). On a tokenizer stripped to a bare
    post-processor the policy is LIVE — the ``add_special_tokens=True`` ids
    gain the BOS — while the seam's ids are byte-identical before and after.
    Binding it would have refused every existing capture sentinel over an input
    the captured bytes cannot depend on."""
    import copy

    from sisteer.extraction.leading_tokens import (apply_leading_special_tokens,
                                                   template_free_prefix)

    _model, tok0 = tiny_llama
    tok = copy.deepcopy(tok0)                       # never mutate the shared fixture
    tok.backend_tokenizer.post_processor = None     # a bare tokenizer: prepends nothing
    assert template_free_prefix(tok) == []
    texts = [scr.build_chat_prompt(tok, p, system=None) for p in ("hello there", "a second prompt")]
    seam_before = [tok(x, add_special_tokens=scr.CAPTURE_ADD_SPECIAL_TOKENS)["input_ids"] for x in texts]
    plain_before = [tok(x, add_special_tokens=True)["input_ids"] for x in texts]
    apply_leading_special_tokens(tok, "bos")
    assert template_free_prefix(tok) == [int(tok.bos_token_id)]          # the policy is live ...
    plain_after = [tok(x, add_special_tokens=True)["input_ids"] for x in texts]
    assert plain_after != plain_before                                     # ... and moves the True path
    assert all(ids[0] == int(tok.bos_token_id) for ids in plain_after)
    seam_after = [tok(x, add_special_tokens=scr.CAPTURE_ADD_SPECIAL_TOKENS)["input_ids"] for x in texts]
    assert seam_after == seam_before                                       # the seam's ids do not move
    assert isinstance(scr.CAPTURE_INPUTS["leading_special_tokens"], str)  # exempted, with its reason


def test_capture_binding_names_every_model_load_and_capture_input(tiny_llama, monkeypatch):
    """The rule behind the resume checks, made checkable: a checkpoint's
    identity must name every input that can change its bytes.
    The set of inputs is computed from the LIVE code — the declared
    parameters of load_model_and_tokenizer, the keywords load_class_b_model
    actually passes through its **kwargs (recorded by stubbing the loader),
    and the parameters of capture_all_layers and _collect_last_acts — and
    must equal the keys of CAPTURE_INPUTS exactly. Every input maps to
    binding keys that exist; every binding key is reached by some input (or
    is one of the two no parameter carries: side, code_sha); dtype, the
    attention kernel and the device are BOUND, not exempt; capture_binding
    takes every key keyword-only with no default; the identity is read off
    the LOADED model. A new loader keyword or capture parameter fails here
    until it is bound or exempted with its reason written down."""
    import inspect

    import fit_class_b as fcb
    from sisteer import models
    from sisteer.extraction import alphasteer as als

    model, tok = tiny_llama
    loader = inspect.signature(models.load_model_and_tokenizer).parameters
    assert any(p.kind is p.VAR_KEYWORD for p in loader.values())    # the **kwargs passthrough
    declared = {n for n, p in loader.items() if p.kind is not p.VAR_KEYWORD}
    passed = {}
    monkeypatch.setattr(models, "load_model_and_tokenizer",
                        lambda name, **kw: passed.update(kw) or (model, tok))
    cfg = {"model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
           "model_revision": None, "dtype": "float32"}
    assert fcb.load_class_b_model(cfg) == (model, tok)
    assert passed, "load_class_b_model passed nothing through the loader's **kwargs?"
    capture_params = (set(inspect.signature(scr.capture_all_layers).parameters)
                      | set(inspect.signature(als._collect_last_acts).parameters))
    consumed = declared | set(passed) | capture_params
    assert consumed == set(scr.CAPTURE_INPUTS), (
        "inputs the code consumes that CAPTURE_INPUTS does not place: "
        f"{sorted(consumed - set(scr.CAPTURE_INPUTS))}; placed but no longer "
        f"consumed: {sorted(set(scr.CAPTURE_INPUTS) - consumed)}")
    bound = set()
    for param, where in scr.CAPTURE_INPUTS.items():
        if isinstance(where, str):
            assert len(where) > 30, f"{param}: an exemption needs its reason"
            continue
        assert where and set(where) <= set(scr.CAPTURE_BINDING_KEYS), (param, where)
        bound |= set(where)
    assert bound | set(scr.CAPTURE_BINDING_KEYS_WITHOUT_A_PARAMETER) == set(scr.CAPTURE_BINDING_KEYS)
    assert not bound & set(scr.CAPTURE_BINDING_KEYS_WITHOUT_A_PARAMETER)
    for must in ("dtype", "attn_implementation", "device", "batch_size", "revision", "name"):
        assert isinstance(scr.CAPTURE_INPUTS[must], tuple), must
    # every binding key is a keyword-only argument with no default — except the
    # seam-fixed tokenization constant, which is held by EXECUTING the seam:
    # the value capture_all_layers actually passes to _collect_last_acts is
    # the value the binding records
    sig = inspect.signature(scr.capture_binding).parameters
    assert set(sig) == set(scr.CAPTURE_BINDING_KEYS) - {"add_special_tokens"}
    assert all(p.kind is p.KEYWORD_ONLY and p.default is p.empty for p in sig.values())
    seam_kw = {}
    monkeypatch.setattr(scr, "_collect_last_acts",
                        lambda m, t, texts, layers, site, bs, **kw: seam_kw.update(kw)
                        or {l: torch.zeros(len(texts), 4) for l in layers})
    scr.capture_all_layers(model, tok, ["x"], site="pre", batch_size=1, layers=[0])
    assert seam_kw["add_special_tokens"] is scr.CAPTURE_ADD_SPECIAL_TOKENS is False
    assert _capture_binding()["add_special_tokens"] is scr.CAPTURE_ADD_SPECIAL_TOKENS
    with pytest.raises(TypeError):                                # dtype/attn/device left out
        scr.capture_binding(model="m", model_revision=None, site="pre", side="benign",
                            list_sha256="b" * 64, n_rows=3, batch_size=8, code_sha="c" * 40)
    # the identity is RESOLVED off the loaded model, and is exactly the
    # model-load half of the binding
    ident = scr.model_load_identity(model, name=cfg["model"], cfg_revision=None)
    assert ident == {"model": cfg["model"], "model_revision": model.config._commit_hash,
                     "dtype": "float32", "attn_implementation": model.config._attn_implementation,
                     "device": "cpu"}
    assert set(ident) | {"site", "side", "list_sha256", "n_rows", "batch_size",
                         "add_special_tokens", "code_sha"} == set(scr.CAPTURE_BINDING_KEYS)
    assert scr.model_load_identity(model, name="x", cfg_revision="pin")["model_revision"] \
        == model.config._commit_hash                              # the resolved commit wins over the pin
    # the screen binding: exactly its keys, both captures required and whole
    sb = _binding()
    assert set(sb) == set(scr.SCREEN_BINDING_KEYS)
    assert all(set(sb["capture"][s]) == set(scr.CAPTURE_BINDING_KEYS) for s in ("benign", "harmful"))
    with pytest.raises(ValueError, match="capture="):
        _binding(capture={"benign": _capture_binding()})
    with pytest.raises(ValueError, match="capture="):
        _binding(capture={"benign": scr.binding_sha256(_capture_binding()),
                          "harmful": scr.binding_sha256(_capture_binding("harmful"))})


def test_auroc_figure_marks_chosen_layers_at_their_depth_under_band_mode(tmp_path, monkeypatch):
    """Under --layers band the rows do not start at layer 0, so
    max(layer)+1 is not n_blocks; the chosen-layer vlines sit at each row's
    recorded depth."""
    import alphasteer_screen as cli
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_blocks = 10
    rows = [{"layer": l, "p": p, "auroc": 0.9, "leakage": 1e-4, "depth": l / n_blocks,
             "in_band": True} for l in (2, 3, 4, 5) for p in (0.3, 0.5)]
    selection = {"chosen": {"layers": [3, 5]}}
    xs = []
    real = plt.Axes.axvline
    monkeypatch.setattr(plt.Axes, "axvline",
                        lambda self, x, *a, **k: (xs.append(x), real(self, x, *a, **k))[1])
    png = cli._auroc_figure(rows, selection, model_key="t",
                            rule={"depth_band": [0.15, 0.65], "auroc_threshold": 0.95},
                            out_png=tmp_path / "figs" / "auroc_vs_depth.png")
    assert png is not None and png.is_file()
    assert xs == [0.3, 0.5]                   # 3/10, 5/10 — not 3/6, 5/6


def test_capture_progress_logging_is_logging_only(tiny_llama, monkeypatch, tmp_path):
    """The per-N-batches progress line (job 470950 sat silent for 21 minutes
    between "split written" and a cancel, so a healthy slow capture and a hung
    one read the same): capture_all_layers -> _collect_last_acts emits one
    line every alphasteer.CAPTURE_LOG_EVERY_BATCHES batches with rows done,
    batches, elapsed and rate; capture_or_load_side prefixes the side and its
    final line carries the rate. And it is LOGGING ONLY: the matrices captured
    with and without log_fn are bit-identical, and log_fn is an exempted
    CAPTURE_INPUTS parameter with its reason written down."""
    import re

    from sisteer.extraction import alphasteer as als
    from sisteer.models import build_chat_prompt

    model, tok = tiny_llama
    prompts = [f"prompt number {i}" for i in range(7)]
    texts = [build_chat_prompt(tok, p, system=None) for p in prompts]
    monkeypatch.setattr(als, "CAPTURE_LOG_EVERY_BATCHES", 2)
    lines: list[str] = []
    got, _ = scr.capture_all_layers(model, tok, texts, site="pre", batch_size=1,
                                    layers=[0, 1], log_fn=lines.append)
    want, _ = scr.capture_all_layers(model, tok, texts, site="pre", batch_size=1, layers=[0, 1])
    assert set(got) == {0, 1} and all(torch.equal(got[l], want[l]) for l in got)
    # batches 2, 4, 6 of 7 (never the final batch -- the caller's completion line covers it)
    assert [l.split(" rows")[0] for l in lines] == ["capture 2/7", "capture 4/7", "capture 6/7"]
    assert all(re.search(r"^capture \d/7 rows \(\d batches\) in \d+s -- [\d.]+ rows/s$", l)
               for l in lines), lines
    assert isinstance(scr.CAPTURE_INPUTS["log_fn"], str) and "logging only" in scr.CAPTURE_INPUTS["log_fn"]
    # through the ONE capture seam: side-prefixed progress, a start line, a rate on the end line
    log: list[str] = []
    acts, sent = scr.capture_or_load_side(
        model, tok, prompts, side="benign", acts_dir=tmp_path / "acts",
        binding=_capture_binding(batch_size=1, n_rows=7), log_fn=log.append)
    assert log[0].startswith("benign: capturing 7 rows x every block at site=pre width 1 -> ")
    assert [l.split(" rows")[0] for l in log[1:-1]] == [
        "benign: capture 2/7", "benign: capture 4/7", "benign: capture 6/7"]
    assert re.search(r"^benign: captured 7 rows x \d+ layers at site=pre in \d+s "
                     r"\([\d.]+ rows/s\) -> ", log[-1]), log[-1]
    assert all(torch.equal(acts[l], want[l]) for l in (0, 1))
    assert sent["n_rows"] == 7 and (tmp_path / "acts" / "benign" / scr.CAPTURE_SENTINEL).is_file()
