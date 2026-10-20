"""The AlphaSteer -> KL-constrained steering ladder (extraction/alphakl.py).

CPU-only, tiny-llama fixture + synthetic tensors. Covers the spec's
acceptance tests that need no GPU campaign:
- T1 metric nesting: fit_stage3(s == 1) is fit_stage2 BITWISE (same code path);
- T2 projector limit: eigenvalue-thresholded Gram + beta -> inf recovers the
  hard null space (cos > 0.99);
- stage-1 parity with the d x d alphasteer arm: the paper's
  "learnable matrix" factors as Delta = u r', so the rank-1 probe and the
  implied gate of M = P Delta~ must agree;
- Fisher weights: the JVP joint tangent vs finite differences (exact), and
  the MC-VJP estimator vs the JVP on a single-position mask, where the two
  estimators provably coincide (score of a softmax = u_y - E_p[u], so
  E[(score)^2] = Var_p[u]);
- calibrate_beta: budget met, boundary tight, unmeetable budget fails loud;
- the stage-2 derived beta (2026-08-24): beta=None derives against eps the
  way stage 3 does (the manual-beta raise had turned a crash workaround
  into the shipped step size, and the arm diverged under it), a pinned
  beta reproduces the old path BITWISE, the canary path re-bisects at the
  deployed scale, the unmeetable budget still refuses, and the bisection's
  monotone-where-matched / never-over-budget properties are pinned;
- the benign-KL operating-point gate (2026-08-22 wave fix): a canary-
  calibrated scale whose MEASURED benign KL busts the budget is capped down
  the ladder, and a calibration where no scale fits REFUSES (the canary's
  distinct-3/ppl guard is blind to this arm's low-perplexity mode collapse,
  so it shipped base_scale=8 at 0.719 nats/position of benign drift);
- predicted vs measured benign KL: at a small uniform write the second-order
  prediction (1/2) lam^2 s_t must match the exact teacher-forced KL — the
  machinery behind acceptance test T3 (the campaign-scale T3/T4/T5 runs are
  not here);
- deployment mechanics: scale-0 bit-identity, ReLU clamp semantics, decode-
  position edits (the deliberate difference from the prefill-only alphasteer
  arm), save/load through methods.load_object;
- the registry trainer end to end for all three stages, and this lane's
  registry wiring: indices 20/21/22, the full core suite set (matching
  psr_kl_all), the fisher batch_size default, and the
  probe-quality metadata gate — plus the config / S2_ARMS parity the other
  appended arms keep.

PORT NOTE: ported additively from a development branch at
429015e; the stage2-evals merge later brought that branch's own config
(stage2_llama8b.yaml arms[17:20] since the 2026-08-19 registry-order
reorder — were [19:22]) / stage2_common.sh S2_ARMS wiring into this tree,
so its config-parity test (test_alphakl_rungs_in_the_config_and_s2_arms) is
kept below alongside this lane's test_alphakl_arm_wiring.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import yaml

from sisteer.data.pairs import Pair
from sisteer.extraction.alphakl import (
    BENIGN_KL_BUDGET_FACTOR,
    DEFAULT_CAPTURE_BUDGET_BYTES,
    DEFAULT_EPS,
    DEFAULT_SCALE_LADDER,
    MIN_CAPTURE_POSITIONS_PER_DIM,
    AlphaKLObject,
    calibrate_beta,
    cap_scale_to_benign_budget,
    fisher_weights_jvp,
    fisher_weights_mc,
    fit_stage1,
    fit_stage2,
    fit_stage3,
    default_max_capture_positions,
    harvest_positions,
    measured_benign_kl,
    null_basis,
    predicted_benign_kl,
    ridge,
    rollout_ids,
    train_alphakl,
)
from sisteer.extraction.alphasteer import (
    AlphaSteerObject,
    null_space_projector,
    ridge_delta,
)
from sisteer.methods import ARMS, load_object
from sisteer.models import hidden_size

REPO = Path(__file__).resolve().parents[1]


class _Instruction:
    id = "french"
    text = "Always answer in French."


def _pairs(n=12):
    prompts = [
        "Say something nice.", "Describe a tree.", "Name a color.",
        "What is water?", "Greet a friend.", "Describe the moon.",
        "Name an animal.", "What is bread?", "Describe rain.",
        "Name a season.", "What is a river?", "Describe a mountain.",
    ]
    return [Pair(prompt=p, pos="Bonjour", neg="Hello",
                 instruction_id="french") for p in prompts[:n]]


def _synth(d=12, n_mal=20, n_ben=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    scales = torch.logspace(0.5, -2.5, d)          # anisotropic benign spectrum
    Hb = torch.randn(n_ben, d, generator=g) * scales
    Hm = torch.randn(n_mal, d, generator=g) + 2.0 * torch.eye(d)[0]
    return Hm, Hb


def _cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


# ---------------------------------------------------------- fits (no model) ----


def test_t1_stage3_unit_weights_is_stage2_bitwise():
    """Spec T1: fit_stage3(s == 1) == fit_stage2, bitwise (same code path)."""
    Hm, Hb = _synth()
    w2 = fit_stage2(Hm, Hb, beta=3.7, alpha=1e-2)
    w3 = fit_stage3(Hm, Hb, torch.ones(Hb.shape[0]), beta=3.7, alpha=1e-2)
    assert torch.equal(w2, w3)
    # and with the LEACE-style centering flag on, still the same path
    w2c = fit_stage2(Hm, Hb, beta=3.7, alpha=1e-2, center_gram=True)
    w3c = fit_stage3(Hm, Hb, torch.ones(Hb.shape[0]), beta=3.7, alpha=1e-2,
                     center_gram=True)
    assert torch.equal(w2c, w3c)


def test_t2_thresholded_gram_at_beta_inf_recovers_stage1():
    """Spec T2: eigenvalue-threshold M_b at tau, beta = 1e8 -> the stage-2
    solution collapses onto the hard null-space fit (cos > 0.99)."""
    Hm, Hb = _synth()
    tau, alpha = 1e-3, 1e-2
    gram = Hb.t() @ Hb
    evals, evecs = torch.linalg.eigh(gram)
    keep = evals > tau * evals[-1]                 # the NON-null directions
    Mb_thr = (evecs[:, keep] * evals[keep]) @ evecs[:, keep].t()
    w_limit = ridge(Hm, Mb_thr, beta=1e8, alpha=alpha)
    w1 = fit_stage1(Hm, Hb, alpha=alpha, tau=tau)
    assert _cos(w_limit, w1) > 0.99


def test_stage1_rank1_matches_dxd_alphasteer_factorization():
    """The d x d arm's target matrix R stacks copies of r, so
    M = P pinv(X'X + lam P)(X'R) factors as (P u) r' exactly — the implied
    probe M r / ||r||^2 must be parallel to fit_stage1's w (same rank rule,
    alpha <-> lambda_reg), and the deployed edits proportional."""
    Hm, Hb = _synth(d=16, n_mal=24, n_ben=48)
    g = torch.Generator().manual_seed(3)
    r = torch.randn(16, generator=g)
    ratio, lam_reg = 0.5, 10.0
    P, _ = null_space_projector(Hb, ratio)
    M, _ = ridge_delta(Hm, P, r, lam_reg)
    M = P @ M
    w_implied = (M @ r) / float(r.norm() ** 2)
    w1 = fit_stage1(Hm, Hb, alpha=lam_reg, nullspace_ratio=ratio)
    assert _cos(w_implied, w1) > 0.999
    # deployed edit parity: h @ M == (w1 . h) * r up to the ||r|| scale
    h = torch.randn(5, 16, generator=g)
    edit_dxd = h @ M                               # [5, 16], rows ~ r
    edit_r1 = torch.outer(h @ w1, r)
    scale = float((edit_dxd * edit_r1).sum() / (edit_r1 * edit_r1).sum())
    assert torch.allclose(edit_dxd, scale * edit_r1, atol=1e-4)


def test_null_basis_rules_and_validation():
    Hm, Hb = _synth()
    Q_tau, evals = null_basis(Hb, tau=1e-3)
    Q_ratio, _ = null_basis(Hb, nullspace_ratio=0.5)
    assert Q_ratio.shape[1] == 6                   # ratio * d
    assert int((evals <= 1e-3 * evals[-1]).sum()) == Q_tau.shape[1]
    with pytest.raises(ValueError):
        null_basis(Hb)                             # neither rule
    with pytest.raises(ValueError):
        null_basis(Hb, tau=1e-3, nullspace_ratio=0.5)  # both rules


def test_center_gram_at_unit_weight_equals_uncentered():
    """The centered covariance + explicit mean-shift term at weight 1.0 is
    the uncentered Gram (cross terms vanish at the weighted mean) — the
    flag only bites through mean_shift_weight."""
    from sisteer.extraction.alphakl import benign_penalty
    _, Hb = _synth()
    s = torch.rand(Hb.shape[0]) + 0.5
    a = benign_penalty(Hb, s, center_gram=False)
    b = benign_penalty(Hb, s, center_gram=True, mean_shift_weight=1.0)
    assert torch.allclose(a, b, atol=1e-4)
    c = benign_penalty(Hb, s, center_gram=True, mean_shift_weight=0.0)
    assert not torch.allclose(a, c, atol=1e-4)


# ------------------------------------------------------------ Fisher weights ----


def _enc(tok, texts):
    prev = tok.padding_side
    tok.padding_side = "right"
    try:
        return tok(texts, return_tensors="pt", padding=True)
    finally:
        tok.padding_side = prev


def _unit(d, seed):
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(d, generator=g)
    return r / r.norm()


def test_fisher_jvp_matches_finite_differences(tiny_llama):
    """s_t = Var_{y~p_t}[u_t], u = J . (joint write): the JVP tangent must
    match a central finite difference of the same hooked forward — also
    exercised with TWO steered layers (the joint multi-layer tangent)."""
    model, tok = tiny_llama
    d = hidden_size(model)
    enc = _enc(tok, ["the quick brown fox", "hello world of tensors"])
    ids, mask = enc["input_ids"], enc["attention_mask"]
    writes = {0: _unit(d, 1), 1: _unit(d, 2)}   # both blocks: joint tangent
    s = fisher_weights_jvp(model, writes, ids, mask)

    from sisteer.extraction.alphakl import _write_hooks
    steer = mask.float()

    def logits_at(eps):
        handles = _write_hooks(model, writes, torch.tensor(eps), steer)
        try:
            with torch.no_grad():
                return model(input_ids=ids, attention_mask=mask,
                             use_cache=False).logits
        finally:
            for h in handles:
                h.remove()

    e = 1e-3
    u = ((logits_at(e) - logits_at(-e)) / (2 * e)).float()
    p = logits_at(0.0).float().softmax(-1)
    m1 = (p * u).sum(-1)
    m2 = (p * u * u).sum(-1)
    s_fd = ((m2 - m1 * m1) * steer)
    assert torch.allclose(s, s_fd, rtol=1e-2, atol=1e-6)
    assert (s >= -1e-6).all()                      # a variance
    assert (s[mask == 0] == 0).all()               # zero off the mask


def test_fisher_mc_matches_jvp_on_a_single_position_mask(tiny_llama):
    """On a mask selecting ONE position, the MC score estimator and the JVP
    variance provably coincide (softmax score = u_y - E_p[u], and no
    downstream positions are in the loss): mean over many samples must land
    within MC noise of the exact value."""
    model, tok = tiny_llama
    d = hidden_size(model)
    enc = _enc(tok, ["the quick brown fox jumps over"])
    ids, mask = enc["input_ids"], enc["attention_mask"]
    steer = torch.zeros_like(mask)
    steer[0, 2] = 1                                # one steered position
    writes = {1: _unit(d, 5)}
    s_jvp = fisher_weights_jvp(model, writes, ids, mask, steer_mask=steer)
    s_mc = fisher_weights_mc(model, writes, ids, mask, steer_mask=steer,
                             mc_samples=512, seed=11)
    exact, est = float(s_jvp[0, 2]), float(s_mc[0, 2])
    assert exact > 0
    assert abs(est - exact) / exact < 0.35         # K=512 MC noise band
    assert (s_mc[steer == 0] == 0).all()


def test_predicted_matches_measured_kl_for_a_small_uniform_write(tiny_llama):
    """The T3 machinery: for a UNIFORM write (gate == 1, i.e. the joint unit
    write scaled by lam) the second-order prediction (1/2) lam^2 s_t must
    match the exact teacher-forced per-position KL(p || p_delta) to O(lam^3).
    Routed through the real deployment object with w = 0 gate ... impossible
    (gate 0 = no edit), so the write is applied by the same hook helper the
    Fisher pass uses."""
    model, tok = tiny_llama
    d = hidden_size(model)
    enc = _enc(tok, ["the quick brown fox", "a small red house"])
    ids, mask = enc["input_ids"], enc["attention_mask"]
    writes = {1: _unit(d, 7)}
    lam = 1e-2
    s = fisher_weights_jvp(model, writes, ids, mask)

    from sisteer.extraction.alphakl import _write_hooks
    with torch.no_grad():
        base = model(input_ids=ids, attention_mask=mask,
                     use_cache=False).logits
        handles = _write_hooks(model, writes, torch.tensor(lam), mask.float())
        try:
            steered = model(input_ids=ids, attention_mask=mask,
                            use_cache=False).logits
        finally:
            for h in handles:
                h.remove()
    logp = base.float().log_softmax(-1)
    logq = steered.float().log_softmax(-1)
    kl = (logp.exp() * (logp - logq)).sum(-1)[mask.bool()]
    pred = (0.5 * lam * lam * s)[mask.bool()]
    ratio = float(pred.sum() / kl.sum())
    assert 0.9 < ratio < 1.1


def test_calibrate_beta_meets_budget_tightly_and_fails_loud():
    Hm, Hb = _synth(d=10, n_mal=16, n_ben=60)
    s = torch.rand(60) + 0.1
    idx = torch.arange(60) // 3                    # 3 "positions" per example
    lam = 1.0

    def fit(b):
        return {0: fit_stage3(Hm, Hb, s, beta=b, alpha=1e-3)}

    def kl(ws):
        return predicted_benign_kl(ws, lam, {0: Hb}, s, idx)["mean"]

    eps = 1e-4
    beta, ws = calibrate_beta(fit, kl, eps)
    got = kl(ws)
    assert got <= eps                              # certified under budget
    assert kl(fit(beta / 100.0)) > eps             # and not trivially small
    with pytest.raises(RuntimeError, match="unmeetable"):
        calibrate_beta(fit, kl, 0.0)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="the device seam needs a real second device")
def test_gates_align_cuda_probe_to_cpu_harvest():
    """Regression (canary 147484, all 3 rungs): post-fit diagnostics call
    _gates with the deployed object's CUDA ws against CPU harvest
    activations — must align to the acts' device, not crash."""
    from sisteer.extraction.alphakl import _gates
    acts = {1: torch.randn(6, 8)}
    ws = {1: torch.randn(8).cuda()}
    g = _gates(ws, acts, clamp=False)
    assert g.device.type == "cpu" and g.shape == (6,)


def test_predicted_kl_clamp_uses_the_deployed_gate():
    Hm, Hb = _synth()
    w = {0: torch.randn(12, generator=torch.Generator().manual_seed(2))}
    s = torch.ones(40)
    idx = torch.arange(40) // 4
    raw = predicted_benign_kl(w, 1.0, {0: Hb}, s, idx, clamp=False)["mean"]
    clamped = predicted_benign_kl(w, 1.0, {0: Hb}, s, idx, clamp=True)["mean"]
    assert clamped < raw                           # negative gates dropped


# ------------------------------------------------------- deployment object ----


def test_alphakl_scale_zero_is_bit_identical(tiny_llama):
    model, tok = tiny_llama
    d = hidden_size(model)
    enc = _enc(tok, ["the quick brown fox jumps"])
    with torch.no_grad():
        base = model(**enc).logits
    obj = AlphaKLObject(model, [1], {1: torch.randn(d)}, {1: _unit(d, 1)},
                        base_scale=2.0)
    with obj.apply(model, scale=0.0), torch.no_grad():
        steered = model(**enc).logits
    assert torch.equal(base, steered)


def test_alphakl_edits_decode_positions_unlike_the_dxd_arm(tiny_llama):
    """The ladder's stage-3 benign certificate lives on generation
    positions, so the edit MUST apply at decode steps (seq_len == 1) — the
    deliberate deployment difference from the prefill-only alphasteer arm."""
    model, tok = tiny_llama
    d = hidden_size(model)
    ids = tok("word", return_tensors="pt")["input_ids"][:, :1]  # seq_len 1
    with torch.no_grad():
        base = model(input_ids=ids).logits
    w, r = torch.full((d,), 0.3), _unit(d, 4)
    obj = AlphaKLObject(model, [1], {1: w}, {1: r}, base_scale=1.0,
                        clamp_gate=False)
    with obj.apply(model), torch.no_grad():
        ours = model(input_ids=ids).logits
    assert not torch.equal(base, ours)
    dxd = AlphaSteerObject(model, layer=1, M=torch.outer(w, r),
                           base_scale=1.0)
    with dxd.apply(model), torch.no_grad():
        theirs = model(input_ids=ids).logits
    assert torch.equal(base, theirs)               # prefill-only: untouched


def test_clamp_gate_blocks_anti_instruction_steering(tiny_llama):
    """clamp_gate=True: positions whose projection is negative receive NO
    edit (the spec's fix for the papers' unbounded linear gate, which gives
    adversarial prompts anti-SI steering)."""
    model, tok = tiny_llama
    d = hidden_size(model)
    enc = _enc(tok, ["the quick brown fox jumps over the lazy dog"])
    from sisteer.extraction.capture import capture_layer_outputs
    with capture_layer_outputs(model, [1]) as cap, torch.no_grad():
        base = model(**enc).logits
    h = cap[1][0].float()                          # [T, d]
    # exact all-negative gates: solve h @ w = -1 (underdetermined, T < d)
    w = torch.linalg.pinv(h) @ (-torch.ones(h.shape[0]))
    gates = h @ w
    assert (gates < 0).all()
    r = _unit(d, 9)
    clamped = AlphaKLObject(model, [1], {1: w}, {1: r}, clamp_gate=True)
    with clamped.apply(model), torch.no_grad():
        out_c = model(**enc).logits
    assert torch.allclose(base, out_c)             # all gates ReLU'd to 0
    raw = AlphaKLObject(model, [1], {1: w}, {1: r}, clamp_gate=False)
    with raw.apply(model), torch.no_grad():
        out_r = model(**enc).logits
    assert not torch.allclose(base, out_r)         # anti-SI edit fires


def test_save_load_roundtrip_through_load_object(tiny_llama, tmp_path):
    model, tok = tiny_llama
    d = hidden_size(model)
    obj = AlphaKLObject(model, [0, 1],
                        {0: torch.randn(d), 1: torch.randn(d)},
                        {0: _unit(d, 1), 1: _unit(d, 2)},
                        base_scale=1.5, clamp_gate=False, stage=3,
                        metadata={"beta": 12.5})
    path = tmp_path / "object.pt"
    obj.save(path)
    back = load_object(path, model, tok)
    assert isinstance(back, AlphaKLObject)
    assert back.stage == 3 and back.clamp_gate is False
    assert back.base_scale == 1.5 and back.metadata["beta"] == 12.5
    enc = _enc(tok, ["the quick brown fox"])
    with obj.apply(model), torch.no_grad():
        a = model(**enc).logits
    with back.apply(model), torch.no_grad():
        b = model(**enc).logits
    assert torch.allclose(a, b)


# ---------------------------------------------------------- harvest helpers ----


def test_harvest_positions_shapes_and_alignment(tiny_llama):
    model, tok = tiny_llama
    d = hidden_size(model)
    rows = [torch.tensor(tok(t)["input_ids"]) for t in
            ["one two three", "a much longer sentence about nothing", "hi"]]
    acts, s, idx = harvest_positions(
        model, rows, [1], pad_id=tok.pad_token_id,
        writes={1: _unit(d, 3)}, batch_size=2)
    total = sum(len(r) for r in rows)
    assert acts[1].shape == (total, d)
    assert s.shape == (total,) and idx.shape == (total,)
    assert idx.tolist() == sum(([i] * len(r) for i, r in enumerate(rows)), [])
    acts2, s2, _ = harvest_positions(model, rows, [1],
                                     pad_id=tok.pad_token_id, batch_size=2)
    assert s2 is None and torch.allclose(acts[1], acts2[1])


def test_rollout_ids_prompt_prefix_and_pad_stripping(tiny_llama):
    model, tok = tiny_llama
    rendered = ["count to three", "name a color please now"]
    rows = rollout_ids(model, tok, rendered, max_new_tokens=4, batch_size=2)
    assert len(rows) == 2
    for text, row in zip(rendered, rows):
        prompt = torch.tensor(tok(text)["input_ids"])
        assert torch.equal(row[: len(prompt)], prompt)   # exact-id replay
        assert len(row) <= len(prompt) + 4


# ----------------------------------------------------------- trainer + arms ----


def _train(model, tok, stage, **kw):
    kw.setdefault("layer", 1)
    kw.setdefault("batch_size", 4)
    kw.setdefault("rollout_tokens", 6)
    kw.setdefault("calib_frac", 0.25)
    return train_alphakl(model, tok, _Instruction(), _pairs(), stage=stage,
                         **kw)


def test_trainer_stage1_both_cutoff_rules(tiny_llama):
    model, tok = tiny_llama
    obj = _train(model, tok, 1, lam=0.1)
    assert obj.stage == 1 and obj.metadata["tau"] == 1e-5
    assert 0.0 <= obj.metadata["probe_auc"] <= 1.0
    assert obj.metadata["kl_measured_prompted"] >= 0.0
    obj2 = _train(model, tok, 1, lam=0.1, nullspace_ratio=0.5)
    assert obj2.metadata["nullspace_ratio"] == 0.5
    assert obj2.metadata["tau"] is None


def _last_token_split_acts(model, tok, layer=1, batch_size=4):
    """Replicate the trainer's exact fit-split rendering + last-token
    capture (12 pairs, calib_frac 0.25 -> n_calib 3), for the bit-identity
    pins below: same renderer, same capture call, same batch size."""
    from sisteer.extraction.capture import collect_pooled_activations
    from sisteer.models import build_chat_prompt

    prompts = [p.prompt for p in _pairs()][:-3]
    benign = [build_chat_prompt(tok, p, system=_Instruction().text)
              for p in prompts]
    malicious = [build_chat_prompt(tok, p, system=None) for p in prompts]
    b = collect_pooled_activations(model, tok, benign, [layer],
                                   position="last", batch_size=batch_size)
    m = collect_pooled_activations(model, tok, malicious, [layer],
                                   position="last", batch_size=batch_size)
    return m.acts[layer], b.acts[layer]


def test_trainer_stage2_manual_beta_pins_bit_identically(tiny_llama):
    """An explicit beta still pins the step size — and the pinned fit is
    BIT-identical to the pre-derivation manual path (the same fit_stage2
    call on the same last-token splits), so a recorded run reproduces."""
    model, tok = tiny_llama
    obj = _train(model, tok, 2, lam=0.1, beta=1.0)
    assert obj.stage == 2 and obj.metadata["beta"] == 1.0
    assert obj.metadata["beta_derived_from_eps"] is False
    # no certificate was issued on the pinned path: nothing pretends one was
    assert obj.metadata["eps"] is None
    assert "kl_predicted_prompted" not in obj.metadata
    assert obj.metadata["fisher_mode"] is None
    assert obj.metadata["position_subsample"] is None
    Hm, Hb = _last_token_split_acts(model, tok)
    direct = fit_stage2(Hm, Hb, beta=1.0, alpha=1e-2)     # DEFAULT_ALPHA
    assert torch.equal(obj.ws[1], direct)


def test_trainer_stage1_ws_bit_identical_to_direct_solve(tiny_llama):
    """Stage 1 is untouched by the stage-2 beta derivation — pinned
    bitwise against fit_stage1 on the same recollected splits."""
    from sisteer.extraction.alphakl import DEFAULT_ALPHA, DEFAULT_TAU

    model, tok = tiny_llama
    obj = _train(model, tok, 1, lam=0.1)
    Hm, Hb = _last_token_split_acts(model, tok)
    direct = fit_stage1(Hm, Hb, alpha=DEFAULT_ALPHA, tau=DEFAULT_TAU)
    assert torch.equal(obj.ws[1], direct)


def test_trainer_stage2_derives_beta_when_unpinned(tiny_llama):
    """beta=None at stage 2 DERIVES beta against the eps budget instead of
    raising (2026-08-24 — the raise is what turned a crash workaround into
    the shipped hyperparameter). At lam=1.0 / eps=1e-2 the budget BINDS on
    this fixture (predicted KL at BETA_LO is ~3.5e-2), so the bisection
    must land strictly above the floor and certify under eps."""
    from sisteer.extraction.alphakl import BETA_LO

    model, tok = tiny_llama
    obj = _train(model, tok, 2, lam=1.0, eps=1e-2)
    md = obj.metadata
    assert obj.stage == 2
    assert md["beta_derived_from_eps"] is True
    assert md["beta"] > BETA_LO                    # the budget actually bound
    assert md["kl_predicted_prompted"] <= 1e-2     # certified at lam
    assert md["eps"] == 1e-2
    assert md["fisher_mode"] == "jvp"              # the certificate's pricing
    assert md["rollout_tokens"] == 6
    assert obj.base_scale == 1.0                   # lam given -> verbatim
    # the certificate basis is the CALIB harvest only — the fit stays
    # last-token, so there is no fit leg to misread as a stage-3 harvest
    sub = md["position_subsample"]
    assert "fit" not in sub and sub["calib"]["capped"] is False
    assert sub["calib"]["seed"] == 1               # the trainer's seed + 1
    # a NON-binding budget returns the bisection floor, exactly like the
    # stage-3 runs on record (the config comment's beta=1e-4 note)
    obj2 = _train(model, tok, 2, lam=0.1, eps=1e-2)
    assert obj2.metadata["beta"] == pytest.approx(BETA_LO, rel=1e-6)
    assert obj2.metadata["kl_predicted_prompted"] <= 1e-2


def test_trainer_stage2_unmeetable_budget_fails_loud(tiny_llama):
    """The loud failure survives the derivation: when even BETA_HI cannot
    meet the budget (eps=0 with the raw linear gate — strictly positive
    predicted KL everywhere), the fit refuses rather than shipping the
    least-bad step size."""
    model, tok = tiny_llama
    with pytest.raises(RuntimeError, match="unmeetable"):
        _train(model, tok, 2, lam=0.1, eps=0.0, clamp_gate=False)


def test_trainer_stage2_derived_rebisects_at_the_calibrated_scale(
        tiny_llama, monkeypatch):
    """On the canary path (lam None) a derived stage-2 beta is re-bisected
    at the calibrated operating scale, exactly like stage 3's fixed-point
    pass — so the eps certificate covers the scale that actually deploys."""
    import sisteer.extraction.alphakl as ak

    model, tok = tiny_llama
    monkeypatch.setattr(
        ak, "calibrate_base_scale",
        lambda model_, tok_, prompts, gen, **kw: {
            "base_scale": 0.5, "warnings": []})
    real = ak.calibrate_beta
    calls: list[float] = []

    def spy(fit_fn, kl_fn, eps, **kw):
        out = real(fit_fn, kl_fn, eps, **kw)
        calls.append(out[0])
        return out

    monkeypatch.setattr(ak, "calibrate_beta", spy)
    obj = _train(model, tok, 2, eps=1e-2, benign_kl_budget=float("inf"))
    assert len(calls) == 2                         # initial + re-bisect
    assert obj.metadata["beta"] == calls[-1]
    assert obj.metadata["beta_derived_from_eps"] is True
    assert obj.base_scale == 0.5
    # certified AT the deployed scale by the re-bisect
    assert obj.metadata["kl_predicted_prompted"] <= 1e-2


def test_stage2_bisection_monotone_where_matched_and_safe_where_not():
    """The monotonicity evidence behind giving stage 2 the bisection
    (measured 2026-08-24, and the theory it matches): with the eval form
    drawn from the SAME distribution as the fit penalty, beta -> predicted
    KL is nonincreasing across the whole [BETA_LO, BETA_HI] bracket
    (d KL/d beta = -2 w' M S^-1 M w <= 0 when the forms coincide) — and
    when the forms are adversarially mismatched the curve CAN bump, but
    calibrate_beta still cannot return an over-budget iterate: it verifies
    every candidate and raises when even BETA_HI fails."""
    from sisteer.extraction.alphakl import BETA_HI, BETA_LO, benign_penalty

    grid = torch.logspace(math.log10(BETA_LO), math.log10(BETA_HI), 41)
    for seed in range(4):
        g = torch.Generator().manual_seed(seed)
        d, n_fit, n_eval = 32, 64, 400
        scales = torch.logspace(0.5, -2.5, d)
        Hb_fit = torch.randn(n_fit, d, generator=g) * scales
        Hm = torch.randn(48, d, generator=g) + 2.0 * torch.eye(d)[0]
        Hb_eval = torch.randn(n_eval, d, generator=g) * scales   # matched
        s = torch.rand(n_eval, generator=g) + 0.05
        idx = torch.arange(n_eval) // 8
        Mb = benign_penalty(Hb_fit, torch.ones(n_fit))
        curve = torch.tensor([
            predicted_benign_kl(
                {0: ridge(Hm, Mb, float(b), 1e-2)}, 1.0,
                {0: Hb_eval}, s, idx, clamp=True)["mean"]
            for b in grid])
        steps = curve[1:] - curve[:-1]
        assert (steps <= 1e-6 * curve.max()).all(), (
            f"seed {seed}: matched-form curve must be nonincreasing")
        assert curve[-1] < 1e-3 * curve[0]         # and it really decays
    # the safety property on a deliberately BUMPY curve: the certificate
    # holds at whatever the bisection returns
    bump = lambda b: (1.0 / b) * (1.0 + 100.0 * math.exp(  # noqa: E731
        -((math.log10(b) - 3.0) ** 2)))
    beta, ws = calibrate_beta(
        lambda b: {0: torch.tensor([b])},
        lambda ws_: bump(float(ws_[0])), eps=1e-3)
    assert bump(float(ws[0])) <= 1e-3


def test_trainer_stage3_derives_beta_from_eps_and_verifies(tiny_llama):
    model, tok = tiny_llama
    obj = _train(model, tok, 3, lam=0.1, eps=1e-3)
    md = obj.metadata
    assert md["beta_derived_from_eps"] is True and md["beta"] > 0
    assert md["kl_predicted_prompted"] <= 1e-3     # certified at lam
    assert md["kl_measured_prompted"] >= 0.0
    assert md["kl_measured_over_predicted"] > 0.0
    assert md["fisher_mode"] == "jvp"
    assert obj.base_scale == 0.1                   # lam given -> verbatim


def test_trainer_stage3_mc_mode_and_manual_beta(tiny_llama):
    model, tok = tiny_llama
    obj = _train(model, tok, 3, lam=0.1, beta=5.0, fisher_mode="mc_vjp",
                 mc_samples=2)
    assert obj.metadata["beta"] == 5.0
    assert obj.metadata["beta_derived_from_eps"] is False
    assert obj.metadata["fisher_mode"] == "mc_vjp"


def test_trainer_unprompted_leakage_probe_and_penalty(tiny_llama):
    model, tok = tiny_llama
    docs = ["Plain document text with no template.",
            "Another raw continuation-style document."]
    obj = _train(model, tok, 3, lam=0.1, eps=1e-3, unprompted_texts=docs,
                 include_unprompted_in_penalty=True)
    assert obj.metadata["kl_measured_unprompted"] >= 0.0
    assert obj.metadata["unprompted_in_penalty"] is True


def test_trainer_validation_errors(tiny_llama):
    model, tok = tiny_llama
    with pytest.raises(ValueError, match="stage"):
        _train(model, tok, 4, lam=0.1)
    with pytest.raises(ValueError, match="calibration split"):
        train_alphakl(model, tok, _Instruction(), _pairs(2), stage=1,
                      layer=1, lam=0.1)


def test_registry_arm_pins_its_stage(tiny_llama):
    """The arm name pins the rung: a config train_kwargs `stage` cannot make
    alphakl_null silently run a different fit."""
    model, tok = tiny_llama
    obj = ARMS["alphakl_null"].train(
        model, tok, _Instruction(), _pairs(), layer=1, stage=3, lam=0.1,
        batch_size=4, rollout_tokens=6, epochs=6, max_len=768)
    assert obj.stage == 1


def test_registry_factories_and_suites():
    """Registered on this lane's FULL core suite set (matching psr_kl_all
    and the antidote/alphasteer/lofit bring-in) — the source branch had the
    rungs IFEval-only, but here every arm on the comparison axis must run
    the same suites."""
    for name in ("alphakl_null", "alphakl_soft", "alphakl_fisher"):
        assert name in ARMS
        assert ARMS[name].suites == frozenset({"ifeval"})
        assert ARMS[name].suites == ARMS["psr_kl_all"].suites


def test_alphakl_arm_wiring():
    """Appended after lofit (the port), so every existing
    positional decode keeps its meaning (the append rule), in ladder
    order: null, soft, fisher. The 2026-08-20 diffmean floor was later
    appended after them, then the stage2-evals did/adasteer and cheap-estimator arms
    at the tail, so the alphakl rungs are not the registry tail.
    (cast / circuit_breaker briefly sat between; both left the registry
    entirely 2026-08-20. Indices 21-23 since the 2026-08-28 lora_r4 alias
    joined at index 4 next to its lora_r1_all sibling — the one audited
    non-append insertion, see test_methods_s2's order test.) The derived
    lora_r4_ce_minus_ctl arm was appended at the tail 2026-09-10 (bcb-bfcl-200)."""
    order = list(ARMS)
    assert order[21:24] == ["alphakl_null", "alphakl_soft", "alphakl_fisher"]
    assert order[24] == "diffmean"
    # positional pin, no longer a TAIL pin: the sdf-base-steering CE arms
    # (psr_ce_all/lora_r4_ce, 2026-09-01) are appended AFTER these — the
    # append-only rule at work — so the cheap-estimator arms are not the registry
    # tail anymore.
    assert order[27:32] == ["did", "adasteer", "caa", "actadd",
                            "actadd_front"]


def test_fisher_arm_defaults_batch_size_to_two(monkeypatch):
    """Encoded as a default rather than discovered on a GPU: the
    stage-3 trainer forces math-backend SDPA inside its AD passes (flash
    has no forward-mode AD), so attention memory is O(T^2) and the bf16
    logits + tangent are each [B, T, V] — the source branch's batch_size=8
    default OOMs at the 5,908-token constitution SI used here. The fisher
    registry arm therefore defaults batch_size to 2; an explicit
    train_kwargs batch_size still overrides, and stages 1-2 keep the
    trainer's own default."""
    import sisteer.methods as methods

    seen: dict = {}

    def fake(model, tok, instruction, pairs, **kw):
        seen.clear()
        seen.update(kw)
        return "trained"

    monkeypatch.setattr(methods, "train_alphakl", fake)
    assert ARMS["alphakl_fisher"].train(
        None, None, _Instruction(), [], layer=1) == "trained"
    assert seen["batch_size"] == 2 and seen["stage"] == 3
    ARMS["alphakl_fisher"].train(None, None, _Instruction(), [], layer=1,
                                 batch_size=8)
    assert seen["batch_size"] == 8                 # explicit wins
    ARMS["alphakl_null"].train(None, None, _Instruction(), [], layer=1)
    assert "batch_size" not in seen and seen["stage"] == 1
    ARMS["alphakl_soft"].train(None, None, _Instruction(), [], layer=1,
                               beta=1.0)
    assert "batch_size" not in seen and seen["stage"] == 2


def test_metadata_surfaces_probe_quality_gate(tiny_llama):
    """Interpretation gate: with a ~5,900-token SI, r (diff-in-means at
    the LAST prompt token) can encode length/position features instead of
    SI content — that failure shows up as a chance-level probe, NOT a
    crash, so the object metadata must carry probe_auc and the held-out
    gate means for every rung, to be read before any cell is interpreted."""
    model, tok = tiny_llama
    obj = _train(model, tok, 1, lam=0.1)
    md = obj.metadata
    for key in ("probe_auc", "gate_benign_mean", "gate_malicious_mean"):
        assert key in md, f"metadata must surface {key}"
        assert md[key] == md[key]                  # not NaN
    assert 0.0 <= md["probe_auc"] <= 1.0


# ------------------------------------------- capture-position cap (RAM) ----
# The stage-3 harvest at the realised pool (~23k pairs x multi-k-token
# rollouts) materialises hundreds of GB of fp32 activations without a cap.
# The cap subsamples POSITIONS uniformly ACROSS rows (never a row/pair
# cap), deterministically under the harvest seed, with a hard rank floor
# of MIN_CAPTURE_POSITIONS_PER_DIM x d.


def _cap_rows(tok, n=10):
    texts = [f"row {i}: " + " ".join(["token"] * (4 + (i * 3) % 7))
             for i in range(n)]
    return [torch.tensor(tok(t)["input_ids"]) for t in texts]


def test_harvest_cap_honoured_and_is_the_uniform_position_rule(tiny_llama):
    model, tok = tiny_llama
    d = hidden_size(model)
    rows = _cap_rows(tok, 10)
    total = sum(len(r) for r in rows)
    cap = total // 2
    full_a, full_s, full_idx = harvest_positions(
        model, rows, [1], pad_id=tok.pad_token_id, writes={1: _unit(d, 3)},
        batch_size=3, seed=11)
    acts, s, idx = harvest_positions(
        model, rows, [1], pad_id=tok.pad_token_id, writes={1: _unit(d, 3)},
        batch_size=3, seed=11, max_positions=cap)
    assert acts[1].shape == (cap, d)
    assert s.shape == (cap,) and idx.shape == (cap,)
    # The selection is EXACTLY the documented rule — a uniform sample over
    # the joint position axis (randperm over T_total at the harvest seed) —
    # so the capped harvest equals the full harvest at those positions,
    # across batch boundaries.
    g = torch.Generator().manual_seed(11)
    mask = torch.zeros(total, dtype=torch.bool)
    mask[torch.randperm(total, generator=g)[:cap]] = True
    assert torch.equal(idx, full_idx[mask])
    assert torch.allclose(acts[1], full_a[1][mask])
    assert torch.allclose(s, full_s[mask])
    # ...and it is spread ACROSS rows, not concentrated in a prefix of
    # rows the way a pair cap would be (deterministic under seed=11).
    assert idx.unique().numel() >= 8
    assert int(idx.max()) == len(rows) - 1


def test_harvest_cap_deterministic_and_content_independent(tiny_llama):
    model, tok = tiny_llama
    rows = _cap_rows(tok, 8)
    total = sum(len(r) for r in rows)
    cap = max(1, total // 3)
    kw = dict(pad_id=tok.pad_token_id, batch_size=3, max_positions=cap)
    a1, _, i1 = harvest_positions(model, rows, [1], seed=5, **kw)
    a2, _, i2 = harvest_positions(model, rows, [1], seed=5, **kw)
    assert torch.equal(i1, i2) and torch.allclose(a1[1], a2[1])
    # a different seed moves the sample (different positions -> different
    # activation rows)
    a3, _, _ = harvest_positions(model, rows, [1], seed=6, **kw)
    assert a3[1].shape == a1[1].shape and not torch.allclose(a1[1], a3[1])
    # the RULE depends only on (row lengths, seed, cap), never content —
    # what keeps the sampling identical across the three SIs' harvests
    alt = [torch.roll(r, 1) for r in rows]
    _, _, i4 = harvest_positions(model, alt, [1], seed=5, **kw)
    assert torch.equal(i1, i4)


def test_harvest_uncapped_unchanged_when_knob_absent_or_loose(tiny_llama):
    model, tok = tiny_llama
    rows = _cap_rows(tok, 5)
    total = sum(len(r) for r in rows)
    kw = dict(pad_id=tok.pad_token_id, batch_size=2)
    base, _, bidx = harvest_positions(model, rows, [1], **kw)
    assert bidx.numel() == total          # every real position, as before
    for cap in (None, total, total + 999):
        a, _, i = harvest_positions(model, rows, [1], max_positions=cap, **kw)
        assert torch.equal(i, bidx) and torch.allclose(a[1], base[1])
    with pytest.raises(ValueError, match="max_positions"):
        harvest_positions(model, rows, [1], max_positions=0, **kw)


def test_auto_cap_is_a_memory_budget_at_d_and_layer_count():
    # 40 GiB at d=4096 fp32 (16 KiB / position / layer): the numbers the
    # module comment states, and both are far above the 8 x d rank floor
    assert default_max_capture_positions(4096, 1) == 2_621_440
    assert default_max_capture_positions(4096, 10) == 262_144
    for n_layers in (1, 10):
        assert (default_max_capture_positions(4096, n_layers)
                >= MIN_CAPTURE_POSITIONS_PER_DIM * 4096)
    # halving the budget halves the cap
    assert default_max_capture_positions(
        4096, 1, budget_bytes=20 * 2**30) == 1_310_720


def test_trainer_stage3_cap_below_rank_floor_raises(tiny_llama):
    model, tok = tiny_llama
    floor = MIN_CAPTURE_POSITIONS_PER_DIM * hidden_size(model)
    with pytest.raises(ValueError, match="rank floor"):
        _train(model, tok, 3, lam=0.1, beta=5.0,
               max_capture_positions=floor - 1)
    # ...and through the registry arm (the train_kwargs.alphakl_fisher
    # path), so the knob cannot be silently dropped nor the floor bypassed
    with pytest.raises(ValueError, match="rank floor"):
        ARMS["alphakl_fisher"].train(
            model, tok, _Instruction(), _pairs(), layer=1,
            max_capture_positions=floor - 1)
    with pytest.raises(ValueError, match="auto"):
        _train(model, tok, 3, lam=0.1, beta=5.0,
               max_capture_positions="always")


def test_trainer_stage3_records_position_subsample(tiny_llama):
    model, tok = tiny_llama
    d = hidden_size(model)
    floor = MIN_CAPTURE_POSITIONS_PER_DIM * d
    obj = _train(model, tok, 3, lam=0.1, beta=5.0)
    sub = obj.metadata["position_subsample"]
    # default = "auto": the cap is the byte budget resolved at THIS model's
    # d and captured-layer count, and the basis is recorded
    assert sub["cap_source"] == "auto"
    assert sub["max_capture_positions"] == default_max_capture_positions(d, 1)
    assert sub["capture_budget_bytes"] == DEFAULT_CAPTURE_BUDGET_BYTES
    fit = sub["fit"]
    assert fit["capped"] is False and fit["seed"] == 0
    assert fit["kept_positions"] == fit["total_positions"]
    assert fit["pairs_contributing"] == fit["pairs_total"]
    assert sub["calib"]["capped"] is False
    total = fit["total_positions"]
    assert total > floor            # the fixture can exercise a legal cap
    cap = max(floor, total - 7)
    obj2 = _train(model, tok, 3, lam=0.1, beta=5.0, seed=3,
                  max_capture_positions=cap)
    sub2 = obj2.metadata["position_subsample"]
    assert sub2["max_capture_positions"] == cap
    assert sub2["cap_source"] == "configured"
    assert sub2["capture_budget_bytes"] is None
    fit2 = sub2["fit"]
    assert fit2["capped"] is True and fit2["seed"] == 3
    assert fit2["kept_positions"] == cap
    # greedy rollouts are seed-independent, so the harvest total matches
    assert fit2["total_positions"] == fit["total_positions"]
    # positions still drawn across (essentially) all pairs
    assert fit2["pairs_total"] == fit["pairs_total"]
    assert fit2["pairs_contributing"] >= fit2["pairs_total"] - 1
    # null = explicitly uncapped, recorded as such
    obj3 = _train(model, tok, 3, lam=0.1, beta=5.0,
                  max_capture_positions=None)
    sub3 = obj3.metadata["position_subsample"]
    assert sub3["max_capture_positions"] is None
    assert sub3["cap_source"] == "uncapped"
    assert sub3["fit"]["capped"] is False
    # stages 1/2 harvest nothing — no subsample block to misread
    assert _train(model, tok, 1, lam=0.1).metadata[
        "position_subsample"] is None


def test_alphakl_rungs_in_the_config_and_s2_arms():
    """The ladder rungs follow the appended arms: carried by the config list +
    S2_ARMS (config-parity for --all-cells / analysis and the S2_ARMS==arms
    symmetry); the rungs moved from [19:22] to [17:20] at the 2026-08-19
    registry-order reorder (did/adasteer to the tail)."""
    cfg = yaml.safe_load((REPO / "configs" / "stage2_llama8b.yaml").read_text())
    assert cfg["arms"][17:20] == ["alphakl_null", "alphakl_soft",
                                  "alphakl_fisher"]
    tk = cfg["train_kwargs"]
    assert tk["alphakl_soft"]["beta"] == 1.0       # stage 2: manual beta
    assert tk["alphakl_fisher"]["eps"] == 0.01     # stage 3: budget, not beta
    assert "beta" not in tk["alphakl_fisher"]
    common = (REPO / "scripts" / "sbatch" / "stage2_common.sh").read_text()
    # in S2_ARMS, contiguous (did/adasteer follow since the 2026-08-19
    # registry-order reorder, so no closing paren after fisher)
    assert "alphakl_null alphakl_soft alphakl_fisher" in common


# ------------------------------------- benign-KL operating-point gate ----
# The 2026-08-22 wave: the canary calibration shipped alphakl_soft at
# base_scale=8 (safety_refusal) / 4 (si_constitution) while the measured
# benign KL at those operating points was 0.719 / 0.0627 nats/position —
# 72x / 6x the ladder's eps=0.01 benign budget, on the SI-PRESENT rollouts
# where this arm's edit is contractually supposed to VANISH. Both objects
# steered the student AWAY from the SI-prompted teacher (floor_ratio 2.63 /
# 3.14 vs a no-op's 1.0). The canary guard couldn't fire: this arm
# over-steers into fluent low-perplexity MODE COLLAPSE (gen_ppl fell 10.52
# -> 5.26 -> 4.31 -> 3.72 up the ladder while distinct3 held >= 0.55 until
# scale 32), and a one-sided `gen_ppl > 3x base` guard cannot see falling
# perplexity. The gate makes the already-measured KL binding: cap the scale
# down the ladder, refuse when nothing fits.


def test_cap_scale_walks_down_to_the_largest_fitting_rung():
    calls: list[float] = []

    def measure(s: float) -> float:
        calls.append(s)
        return 0.01 * s * s                    # the second-order lam^2 law

    final, trail, capped = cap_scale_to_benign_budget(
        measure, base_scale=8.0, scale_ladder=DEFAULT_SCALE_LADDER,
        budget=0.02, arm="alphakl2")
    assert final == 1.0 and capped is True
    # walked exactly the canary point then the smaller rungs, in order,
    # stopping at the first fit — never a rung above the canary's choice
    assert calls == [8.0, 4.0, 2.0, 1.0]
    assert [t["scale"] for t in trail] == [8.0, 4.0, 2.0, 1.0]
    assert trail[-1]["kl_measured_prompted"] == pytest.approx(0.01)
    # a canary choice already under budget ships untouched, one measurement
    final2, trail2, capped2 = cap_scale_to_benign_budget(
        measure, base_scale=0.5, scale_ladder=DEFAULT_SCALE_LADDER,
        budget=0.02, arm="alphakl2")
    assert final2 == 0.5 and capped2 is False and len(trail2) == 1


def test_cap_scale_refuses_when_no_scale_fits():
    """A calibration where EVERY scale drifts past the benign budget must
    REFUSE — mirroring calibrate_base_scale's 'NO coherent scale' — never
    ship the least-bad large scale."""
    with pytest.raises(RuntimeError, match="benign KL budget"):
        cap_scale_to_benign_budget(
            lambda s: 0.719, base_scale=16.0,
            scale_ladder=DEFAULT_SCALE_LADDER, budget=0.02, arm="alphakl1")


def test_trainer_caps_calibrated_scale_on_measured_benign_kl(
        tiny_llama, monkeypatch):
    """The wave, reproduced through the trainer with its real numbers: the
    canary picks 8, the measured benign KL there is 0.719 nats/position
    (safety_refusal alphakl_soft), and KL falls as lam^2 — the gate must
    ship the largest rung within the default 2 x eps budget (scale 1 at
    ~0.0112), not the canary's midpoint."""
    import sisteer.extraction.alphakl as ak
    model, tok = tiny_llama

    monkeypatch.setattr(
        ak, "calibrate_base_scale",
        lambda model_, tok_, prompts, gen, **kw: {
            "base_scale": 8.0, "scale_bounds": [0.0, 16.0], "warnings": []})

    def fake_measured(model_, obj, id_rows, *, pad_id, scale=1.0,
                      batch_size=8):
        mag = obj.base_scale * scale
        return {"mean": 0.719 * (mag / 8.0) ** 2, "per_example": []}

    monkeypatch.setattr(ak, "measured_benign_kl", fake_measured)
    obj = _train(model, tok, 1)                # no lam -> calibrated path
    assert obj.base_scale == 1.0               # 0.719/64 = 0.0112 <= 0.02
    gate = obj.metadata["benign_kl_gate"]
    assert gate["capped"] is True
    assert gate["canary_scale"] == 8.0
    assert gate["budget"] == pytest.approx(BENIGN_KL_BUDGET_FACTOR
                                           * DEFAULT_EPS)
    assert gate["budget_source"] == "default_2x_eps"
    assert [t["scale"] for t in gate["trail"]] == [8.0, 4.0, 2.0, 1.0]
    assert gate["trail"][0]["kl_measured_prompted"] == pytest.approx(0.719)
    assert obj.metadata["kl_measured_prompted"] == pytest.approx(0.719 / 64)
    assert obj.metadata["kl_lam_operating"] == 1.0
    assert any("benign-KL gate CAPPED" in w for w in obj.metadata["warnings"])


def test_trainer_refuses_when_every_scale_drifts(tiny_llama, monkeypatch):
    """A calibration that only finds degenerate operating points must
    REFUSE, not ship a large scale: flat 0.719 nats/position at every rung
    (the wave's safety_refusal drift, scale-independent) exceeds the budget
    everywhere, so the trainer raises instead of returning an object."""
    import sisteer.extraction.alphakl as ak
    model, tok = tiny_llama
    monkeypatch.setattr(
        ak, "calibrate_base_scale",
        lambda model_, tok_, prompts, gen, **kw: {
            "base_scale": 16.0, "warnings": []})
    monkeypatch.setattr(
        ak, "measured_benign_kl",
        lambda model_, obj, id_rows, *, pad_id, scale=1.0, batch_size=8: {
            "mean": 0.719, "per_example": []})
    with pytest.raises(RuntimeError, match="benign KL budget"):
        _train(model, tok, 1)


def test_trainer_lam_given_bypasses_the_gate(tiny_llama):
    """The spec pins a caller-supplied lam VERBATIM (the certificate is
    issued AT lam) — even an impossible budget must not gate it."""
    model, tok = tiny_llama
    obj = _train(model, tok, 1, lam=0.1, benign_kl_budget=0.0)
    assert obj.base_scale == 0.1
    assert obj.metadata["benign_kl_gate"] is None


def test_trainer_gate_end_to_end_records_the_trail(tiny_llama):
    """No monkeypatching: the real canary (guard neutered so the tiny
    fixture's short ladder stays coherent) hands its choice to the real
    gate, which measures real KL and records the audit trail."""
    model, tok = tiny_llama
    obj = _train(model, tok, 1, scale_ladder=(0.25, 0.5),
                 rep_threshold=0.0, ppl_guard_factor=1e9,
                 calib_max_new_tokens=4,
                 benign_kl_budget=float("inf"))
    gate = obj.metadata["benign_kl_gate"]
    assert gate is not None and gate["capped"] is False
    assert gate["budget_source"] == "configured"
    assert obj.base_scale == 0.25              # hi=0.5 all-coherent -> hi/2
    assert gate["trail"][0]["scale"] == 0.25
    assert gate["trail"][0]["kl_measured_prompted"] >= 0.0
    assert obj.metadata["kl_lam_operating"] == 0.25


def test_trainer_gate_refuses_end_to_end_at_zero_budget(tiny_llama):
    """No monkeypatching: an impossible budget refuses through the real
    path (clamp_gate=False so the raw linear gate guarantees a strictly
    positive measured KL at every rung)."""
    model, tok = tiny_llama
    with pytest.raises(RuntimeError, match="benign KL budget"):
        _train(model, tok, 1, scale_ladder=(0.25, 0.5),
               rep_threshold=0.0, ppl_guard_factor=1e9,
               calib_max_new_tokens=4, clamp_gate=False,
               benign_kl_budget=0.0)
