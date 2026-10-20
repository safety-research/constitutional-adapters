"""The coherent virtual teacher, pinned on a toy model whose recurrence is
computable by hand (the risk: teacher infidelity that trains anyway):

- reference recurrence: EVERY captured layer output under the paired
  write equals ``z_{l+1} = f_l(z_l + V_l(z_l))`` with both gates read on
  the SHARED pre-write state — planted B and C with distinct,
  simultaneously-active gates at every layer, nonlinear blocks between
  hook points;
- order-swap sensitivity: the two SEQUENTIAL (two-hook) orders differ from
  the shared-state trajectory and from each other on this fixture, and the
  shared-state one is the declared recurrence — so an implementation that
  silently became sequential would fail here;
- the planted identities: R(B ⊕ C, B) equals the +C trajectory exactly
  (to fp tolerance) at every layer; R(A, ∅) equals +A; R(A, A) equals the
  bare model BITWISE (the write cancels exactly) and must be reported as
  "teacher ≈ bare", nothing else;
- the arithmetic residual T_arith is NOT the coherent trajectory here
  (the diagnostic reads nonzero), which is why it is a labeled number and
  never a gate.
"""

from __future__ import annotations

import pytest
import torch

from residual_fixtures import (ApplyPerturbed, capture_trajectory, planted_psr,
                               reference_recurrence, toy_cap_batch, toy_ids,
                               toy_model, zero_psr)
from sisteer.extraction.paired_write import (APPLICATION_ORDER,
                                             PSRPairedWriteObject,
                                             SharedStateWriteObject,
                                             sequential_paired_apply)
from sisteer.extraction.residual import (arithmetic_residual_inconsistency,
                                         hook_contract_problems,
                                         teacher_identity_problems)

LAYERS = [0, 1, 2]


@pytest.fixture(scope="module")
def world():
    model = toy_model(n_layers=3, seed=3)
    B = planted_psr(model, LAYERS, seed=11, open_frac=0.6, write_scale=0.6)
    C = planted_psr(model, LAYERS, seed=23, open_frac=0.5, write_scale=0.6)
    ids = toy_ids(seed=5, batch=3, seq=8)
    return model, B, C, ids


def _gate_open_frac(obj, model, ids):
    """Per-layer fraction of positions with the gate open, on the BARE
    trajectory (documentation that the planted gates are live)."""
    traj = capture_trajectory(model, ids, None, LAYERS)
    out = {}
    with torch.no_grad():
        for l in LAYERS:
            pre = traj[l].float() @ obj.gate_w[l] + obj.gate_b[l]
            out[l] = float((pre > 0).float().mean())
    return out


def test_planted_gates_are_distinct_and_simultaneously_active(world):
    model, B, C, ids = world
    fb, fc = _gate_open_frac(B, model, ids), _gate_open_frac(C, model, ids)
    for l in LAYERS:
        assert 0.2 < fb[l] < 0.95, (l, fb)
        assert 0.2 < fc[l] < 0.95, (l, fc)
        assert not torch.allclose(B.gate_w[l], C.gate_w[l])
        assert not torch.allclose(B.directions[l], C.directions[l])


def test_reference_recurrence_matches_every_captured_layer(world):
    model, B, C, ids = world
    V = PSRPairedWriteObject(B, C)
    got = capture_trajectory(model, ids, V.apply(model, scale=1.0), LAYERS)
    ref = reference_recurrence(model, ids, [(1.0, B), (-1.0, C)])
    bare = capture_trajectory(model, ids, None, LAYERS)
    for l in LAYERS:
        assert torch.allclose(got[l], ref[l], atol=1e-5, rtol=0), l
        # and the teacher really moves the trajectory at every layer
        assert (got[l] - bare[l]).abs().max() > 1e-3, l
    # at scale 2 the recurrence with a doubled write
    got2 = capture_trajectory(model, ids, V.apply(model, scale=2.0), LAYERS)
    ref2 = reference_recurrence(model, ids, [(1.0, B), (-1.0, C)], scale=2.0)
    for l in LAYERS:
        assert torch.allclose(got2[l], ref2[l], atol=1e-5, rtol=0), l


def test_sequential_orders_differ_and_shared_state_is_the_declared_one(world):
    model, B, C, ids = world
    V = PSRPairedWriteObject(B, C)
    assert V.application_order == APPLICATION_ORDER == "shared_pre_write_state"
    shared = capture_trajectory(model, ids, V.apply(model), LAYERS)
    seq_pn = capture_trajectory(
        model, ids, sequential_paired_apply(B, C, model, order="pos_then_neg"),
        LAYERS)
    seq_np = capture_trajectory(
        model, ids, sequential_paired_apply(B, C, model, order="neg_then_pos"),
        LAYERS)
    ref = reference_recurrence(model, ids, [(1.0, B), (-1.0, C)])
    # The two-hook compositions are DIFFERENT functions on this fixture
    # (the second gate reads a post-write state) ...
    d_pn = max(float((seq_pn[l] - shared[l]).abs().max()) for l in LAYERS)
    d_np = max(float((seq_np[l] - shared[l]).abs().max()) for l in LAYERS)
    d_xx = max(float((seq_pn[l] - seq_np[l]).abs().max()) for l in LAYERS)
    assert d_pn > 1e-4 and d_np > 1e-4 and d_xx > 1e-4, (d_pn, d_np, d_xx)
    # ... and only the shared-state one is the declared recurrence.
    for l in LAYERS:
        assert torch.allclose(shared[l], ref[l], atol=1e-5, rtol=0)
    assert not all(torch.allclose(seq_pn[l], ref[l], atol=1e-5, rtol=0)
                   for l in LAYERS)


def test_R_BplusC_B_equals_plus_C_trajectory(world):
    """The load-bearing planted identity: the coherent teacher of parents
    (B ⊕ C) and B is EXACTLY the +C trajectory at every layer."""
    model, B, C, ids = world
    composite = SharedStateWriteObject([(1.0, B), (1.0, C), (-1.0, B)])
    got = capture_trajectory(model, ids, composite.apply(model), LAYERS)
    want = capture_trajectory(model, ids, C.apply(model), LAYERS)
    bare = capture_trajectory(model, ids, None, LAYERS)
    for l in LAYERS:
        assert torch.allclose(got[l], want[l], atol=1e-5, rtol=0), l
        assert (want[l] - bare[l]).abs().max() > 1e-3, l  # C is live
    # and the identity check helper agrees, with a report per layer
    cb = toy_cap_batch(ids, [2, 0, 3], ["mix", "ctl", "mix"])
    problems, report = teacher_identity_problems(
        model, [cb], composite=composite, reference=C, layers=LAYERS)
    assert problems == []
    assert all(report[l]["pass"] for l in LAYERS)
    # a WRONG composite (B ⊕ C minus nothing) fails the same check loudly
    wrong = SharedStateWriteObject([(1.0, B), (1.0, C)])
    problems, _ = teacher_identity_problems(
        model, [cb], composite=wrong, reference=C, layers=LAYERS)
    assert problems and "discrepancy" in problems[0]


def test_endpoint_R_A_empty_equals_A(world):
    model, B, _C, ids = world
    V = PSRPairedWriteObject(B, zero_psr(model, LAYERS))
    got = capture_trajectory(model, ids, V.apply(model), LAYERS)
    want = capture_trajectory(model, ids, B.apply(model), LAYERS)
    for l in LAYERS:
        assert torch.allclose(got[l], want[l], atol=1e-6, rtol=0), l


def test_endpoint_R_A_A_is_bare_bitwise(world):
    model, B, _C, ids = world
    V = PSRPairedWriteObject(B, B)
    got = capture_trajectory(model, ids, V.apply(model), LAYERS)
    bare = capture_trajectory(model, ids, None, LAYERS)
    for l in LAYERS:
        assert torch.equal(got[l], bare[l]), l


def test_scale_zero_is_bitwise_noop_and_scale_is_linear_at_the_hook(world):
    model, B, C, ids = world
    V = PSRPairedWriteObject(B, C)
    bare = capture_trajectory(model, ids, None, LAYERS)
    z0 = capture_trajectory(model, ids, V.apply(model, scale=0.0), LAYERS)
    for l in LAYERS:
        assert torch.equal(z0[l], bare[l])
    # layer 0's write reads the bare layer-0 state, so its delta is exactly
    # linear in scale (downstream layers compose nonlinearly).
    d1 = capture_trajectory(model, ids, V.apply(model, scale=1.0), LAYERS)[0] - bare[0]
    d2 = capture_trajectory(model, ids, V.apply(model, scale=2.0), LAYERS)[0] - bare[0]
    assert d1.abs().max() > 0
    assert torch.allclose(d2, 2 * d1, atol=1e-6)


def test_arithmetic_residual_is_not_the_coherent_trajectory(world):
    """T_arith (three separate forwards) differs from h^V on this fixture
    — the labeled diagnostic reads nonzero, which is exactly why it is
    never a gate."""
    model, B, C, ids = world
    V = PSRPairedWriteObject(B, C)
    cb = toy_cap_batch(ids, [2, 0, 3], ["mix", "ctl", "mix"])
    diag = arithmetic_residual_inconsistency(model, [cb], pos=B, neg=C,
                                             paired=V, layers=LAYERS)
    # layer 0: identical (no upstream write to disagree about) ...
    assert diag[0]["rms_inconsistency"] < 1e-5
    # ... deeper layers: the separately-evolved trajectories disagree.
    assert diag[2]["rms_inconsistency"] > 1e-4
    assert diag[2]["ratio"] is not None


def test_identity_and_arithmetic_reductions_stream_per_batch_exactly(world):
    """``teacher_identity_problems`` and ``arithmetic_residual_inconsistency``
    reduce ONE batch at a time (never holding every context's full [N, d]
    table). Across a batch boundary — two batches of different lengths —
    their numbers must equal the one-table formula bit for bit."""
    model, B, C, ids = world
    composite = SharedStateWriteObject([(1.0, B), (1.0, C), (-1.0, B)])
    V = PSRPairedWriteObject(B, C)
    cbs = [toy_cap_batch(ids, [2, 0, 3], ["mix", "ctl", "mix"]),
           toy_cap_batch(toy_ids(seed=9, batch=2, seq=6), [1, 0],
                         ["ctl", "mix"])]

    def table(ctx_factory):
        # every position of every batch under the context, concatenated
        # into one [N, d] table per layer (the toy rows are all valid)
        per = [capture_trajectory(model, cb.batch.input_ids, ctx_factory(),
                                  LAYERS) for cb in cbs]
        return {l: torch.cat([h[l].float().reshape(-1, h[l].shape[-1])
                              for h in per]) for l in LAYERS}

    def rms(rows):
        return rows.pow(2).sum(-1).mean().sqrt().item()

    def pre_write_table(obj):
        # the pre-write z the object's OWN forward hands its hook, every
        # position of every batch, one [N, d] table per layer
        from sisteer.extraction.psr import _capture_hidden
        per = []
        for cb in cbs:
            with torch.no_grad(), _capture_hidden(model, LAYERS) as cap, \
                    obj.apply(model):
                model(input_ids=cb.batch.input_ids,
                      attention_mask=cb.batch.attention_mask)
                per.append({l: cap[l].detach().clone() for l in LAYERS})
        return {l: torch.cat([z[l].reshape(-1, z[l].shape[-1]) for z in per])
                for l in LAYERS}

    def write(obj, l, z):
        z3 = z.unsqueeze(0)  # [1, N, d] — the hook's [B, S, d] shape
        if hasattr(obj, "paired_write"):
            return obj.paired_write(l, z3)[0]
        return (torch.relu(z3.float() @ obj.gate_w[l] + obj.gate_b[l])
                .unsqueeze(-1) * obj.directions[l])[0]

    h_c = table(lambda: composite.apply(model))
    h_r = table(lambda: C.apply(model))
    h_0 = table(lambda: None)
    z_c = pre_write_table(composite)
    _problems, report = teacher_identity_problems(
        model, cbs, composite=composite, reference=C, layers=LAYERS)
    for l in LAYERS:
        # the gate: the WRITE-level identity on the composite's own z ...
        u_c, u_r = write(composite, l, z_c[l]), write(C, l, z_c[l])
        assert report[l]["rms_write_discrepancy"] == rms(u_c - u_r)
        assert report[l]["rms_reference_write"] == rms(u_r)
        assert report[l]["rms_cancellation_scale"] == max(
            rms(u_r), max(rms(c * write(o, l, z_c[l])) for c, o in composite.terms))
        # ... and the trajectory numbers ride as diagnostics
        assert report[l]["trajectory_rms_divergence"] == rms(h_c[l] - h_r[l])
        assert report[l]["trajectory_rms_reference_write"] == rms(h_r[l] - h_0[l])
        assert report[l]["stream_rms_per_elem"] == pytest.approx(
            z_c[l].float().pow(2).mean().sqrt().item())
    h_v, h_p, h_n = (table(lambda: V.apply(model)), table(lambda: B.apply(model)),
                     table(lambda: C.apply(model)))
    diag = arithmetic_residual_inconsistency(model, cbs, pos=B, neg=C,
                                             paired=V, layers=LAYERS)
    for l in LAYERS:
        t_arith = h_0[l] + (h_p[l] - h_0[l]) - (h_n[l] - h_0[l])
        assert diag[l]["rms_inconsistency"] == rms(h_v[l] - t_arith)
        assert diag[l]["rms_teacher_deviation"] == rms(h_v[l] - h_0[l])


# --------------------------------------------------------------------------- #
# the identity is a WRITE identity, checked at fp32 precision; the            #
# bf16 trajectory divergence is the model's chaos floor, never the gate       #
# --------------------------------------------------------------------------- #

from residual_fixtures import chaos_toy_model, one_ulp_bump_hook  # noqa: E402
from sisteer.extraction.residual import (IDENTITY_ATOL, IDENTITY_RTOL,  # noqa: E402
                                         stream_perturbation_floor)

CHAOS_LAYERS = list(range(6))


def _old_trajectory_check(model, ids, composite, reference, layers,
                          rtol: float = 0.02, extra_hook=None):
    """The check as it shipped earlier (trajectory RMS divergence vs
    0.02 · trajectory write RMS), reproduced here so the test can show
    what it measured. ``extra_hook`` perturbs the COMPOSITE forward only."""
    from sisteer.models import get_decoder_layers
    dec = get_decoder_layers(model)
    handle = dec[layers[0]].register_forward_hook(extra_hook) if extra_hook else None
    try:
        h_c = capture_trajectory(model, ids, composite.apply(model), layers)
    finally:
        if handle:
            handle.remove()
    h_r = capture_trajectory(model, ids, reference.apply(model), layers)
    h_0 = capture_trajectory(model, ids, None, layers)

    def rms(x):
        return x.float().pow(2).sum(-1).mean().sqrt().item()
    out = {}
    for l in layers:
        div, write = rms(h_c[l] - h_r[l]), rms(h_r[l] - h_0[l])
        out[l] = {"divergence": div, "write": write,
                  "pass": div <= max(1e-6, rtol * write)}
    return out


@pytest.fixture(scope="module")
def chaos_world():
    model = chaos_toy_model(n_layers=len(CHAOS_LAYERS), seed=3)
    # the planted gates read a |h|~500 stream; writes ~2.5/elem — the
    # write-to-ulp ratio the real gemma layers with substantial writes have
    B = planted_psr(model, CHAOS_LAYERS, seed=11, open_frac=0.6,
                    write_scale=25.0)
    C = planted_psr(model, CHAOS_LAYERS, seed=23, open_frac=0.5,
                    write_scale=25.0)
    ids = toy_ids(seed=5, batch=4, seq=16)
    return model, B, C, ids


def test_identity_holds_to_fp32_precision_and_order_free(world):
    """Regression: R(B ⊕ C, B) reproduces the +C WRITE at fp32 precision
    (the only residue is fp32 cancellation of the cancelled terms), in
    either summation order; the exact-cancellation order ((B − B) + C)
    reproduces it BIT FOR BIT; the empty parent adds exactly nothing."""
    model, B, C, ids = world
    cb = toy_cap_batch(ids, [2, 0, 3], ["mix", "ctl", "mix"])
    for order in ([(1.0, B), (1.0, C), (-1.0, B)],
                  [(1.0, C), (1.0, B), (-1.0, B)],
                  [(1.0, B), (-1.0, B), (1.0, C)]):
        composite = SharedStateWriteObject(order)
        problems, report = teacher_identity_problems(
            model, [cb], composite=composite, reference=C, layers=LAYERS)
        assert problems == [], order
        for l in LAYERS:
            row = report[l]
            assert row["pass"]
            assert row["tolerance"] == max(IDENTITY_ATOL,
                                           IDENTITY_RTOL * row["rms_cancellation_scale"])
            # fp32 cancellation: ~2^-24 of the cancelled term, never more
            assert row["rms_write_discrepancy"] <= 1e-6 * row["rms_cancellation_scale"]
            assert row["rms_reference_write"] > 1e-3          # C is live
            assert row["write_identity_pass"]
            # the hook contract on an fp32 stream is BITWISE: the runtime
            # performs the identical fp32 add the check re-derives
            for side in ("hook_contract_composite", "hook_contract_reference"):
                hc = row[side]
                assert hc["pass"] and hc["n_violations"] == 0
                assert hc["max_abs_err"] == 0.0 and hc["max_err_half_ulps"] == 0.0
                assert hc["n_elements"] > 0 and hc["write_active_frac"] > 0
                assert hc["stream_dtype"] == "float32"
        if order[1][0] == -1.0:  # (B − B) + C: the cancellation is exact
            assert all(report[l]["rms_write_discrepancy"] == 0.0 for l in LAYERS)
    # ∅ contributes exactly nothing to the write
    composite = SharedStateWriteObject([(1.0, C), (1.0, zero_psr(model, LAYERS))])
    _p, report = teacher_identity_problems(
        model, [cb], composite=composite, reference=C, layers=LAYERS)
    assert all(report[l]["rms_write_discrepancy"] == 0.0 for l in LAYERS)


def test_bf16_stream_trajectory_check_fails_where_write_check_holds(chaos_world):
    """The defect, reproduced on a bf16 toy with a |h|~500 stream and
    chaotic blocks. A ONE-ulp perturbation of ONE stream element — the
    smallest seed a rounding difference can leave, and the one the real
    models were found to amplify (gemma-4-E4B: 5e-8 → ~2–3 per position by
    layer 14; R² 0.97/0.99 against exactly this perturbation) — makes the
    old trajectory-level check FAIL at the deep layers (the bf16 divergence
    dwarfs 0.02 · write), while BOTH halves of the current check on the
    same seeded composite forward PASS: the write algebra is exact on
    whatever z the hook is handed, and the hook contract compares each
    forward with ITSELF (h against z + cast(u(z)) on the same forward), so
    chaos between two forwards cannot touch it. The seed sits on the tensor
    ENTERING the first hooked layer (a forward pre-hook inside the
    composite's apply) — a stand-in for the model's own rounding, which the
    object's declared hook cannot be blamed for; the same seed on the
    layer's OUTPUT inside apply() is a foreign edit of the stream and is
    caught (next test). The trajectory divergence still rides in the
    report, and the perturbation floor matches it."""
    model, B, C, ids = chaos_world
    assert next(model.parameters()).dtype == torch.bfloat16
    composite = SharedStateWriteObject([(1.0, B), (1.0, C), (-1.0, B)])
    bump = one_ulp_bump_hook(row=0, position=3, element=0)
    # the stream really is at real magnitude, with real-magnitude writes
    z0 = capture_trajectory(model, ids, None, CHAOS_LAYERS)[0].float()
    assert 200 < z0.pow(2).mean().sqrt() < 2000
    old = _old_trajectory_check(model, ids, composite, C, CHAOS_LAYERS,
                                extra_hook=bump)
    deep = CHAOS_LAYERS[-1]
    # the seed is one element of one position; the network amplifies it
    # across every position and dimension (≥ 10× by the last layer here,
    # ~×10^7 over 14 layers on the real gemma) ...
    assert old[deep]["divergence"] > 10 * old[CHAOS_LAYERS[0]]["divergence"], old
    # ... to well past the old write-scaled tolerance
    assert not old[deep]["pass"], old
    assert old[deep]["divergence"] > 0.02 * old[deep]["write"], old

    def bump_input(mod, args):
        # the same one-ulp seed, on the tensor ENTERING the first hooked layer
        return (bump(mod, args, args[0]),) + tuple(args[1:])

    class SeededForward:
        """The composite, with the one-ulp seed in ITS forward's input."""
        terms = composite.terms
        layers = composite.layers

        @staticmethod
        def paired_write(layer, hidden):
            return composite.paired_write(layer, hidden)

        @staticmethod
        def apply(m, scale: float = 1.0):
            from contextlib import contextmanager
            from sisteer.models import get_decoder_layers

            @contextmanager
            def _cm():
                h = get_decoder_layers(m)[CHAOS_LAYERS[0]] \
                    .register_forward_pre_hook(bump_input)
                try:
                    with composite.apply(m, scale):
                        yield None
                finally:
                    h.remove()
            return _cm()

    cb = toy_cap_batch(ids, [2, 0, 3, 1], ["mix", "ctl", "mix", "ctl"])
    problems, report = teacher_identity_problems(
        model, [cb], composite=SeededForward(), reference=C, layers=CHAOS_LAYERS)
    assert problems == [], problems
    for l in CHAOS_LAYERS:
        row = report[l]
        assert row["pass"] and row["write_identity_pass"]
        assert row["rms_write_discrepancy"] <= 1e-6 * row["rms_cancellation_scale"]
        for side in ("hook_contract_composite", "hook_contract_reference"):
            hc = row[side]
            assert hc["pass"] and hc["n_violations"] == 0
            assert hc["stream_dtype"] == "bfloat16"
            # the bf16 add's rounding and NOTHING more: never past half an
            # ulp of the stream (ties land exactly on it)
            assert hc["max_err_half_ulps"] <= 1.0
            assert hc["write_active_frac"] > 0.2      # the check has teeth
    # the diagnostics carry what the old check gated on ...
    assert report[deep]["trajectory_rms_divergence"] > \
        0.02 * report[deep]["trajectory_rms_reference_write"]
    # ... and the perturbation floor is that divergence (same seed, same
    # forwards up to the object): the instrument measures the model's chaos
    pert = stream_perturbation_floor(model, [cb], layers=CHAOS_LAYERS,
                                     ctx_factory=lambda: C.apply(model))
    assert pert[CHAOS_LAYERS[0]] > 0
    assert pert[deep] > 0.02 * report[deep]["trajectory_rms_reference_write"]
    # an UNSEEDED composite on this stream: exact write identity, and a
    # trajectory divergence far below the seeded one (only fp32
    # cancellation could seed it here, and on 4.6k elements it rarely does)
    seeded_deep = report[deep]["trajectory_rms_divergence"]
    problems, report = teacher_identity_problems(
        model, [cb], composite=composite, reference=C, layers=CHAOS_LAYERS)
    assert problems == []
    assert report[deep]["trajectory_rms_divergence"] < 0.1 * seeded_deep


def test_a_foreign_one_ulp_edit_inside_apply_is_caught(chaos_world):
    """The bump-inside-apply wrapper of the previous test as it was FIRST
    written — the one-ulp seed on the layer's OUTPUT, inside apply(): a
    runtime that edits the stream beyond its declared write by the smallest
    amount there is. It PASSED the write-level gate (the algebra cannot see
    apply()); the hook contract flags exactly layer 0, exactly one element,
    past half an ulp — and the gate refuses. The write algebra still passes
    at every layer, the reference is clean."""
    model, B, C, ids = chaos_world
    composite = SharedStateWriteObject([(1.0, B), (1.0, C), (-1.0, B)])
    cb = toy_cap_batch(ids, [2, 0, 3, 1], ["mix", "ctl", "mix", "ctl"])
    problems, report = teacher_identity_problems(
        model, [cb], composite=ApplyPerturbed(composite, "bump"), reference=C,
        layers=CHAOS_LAYERS)
    assert len(problems) == 1, problems
    assert problems[0].startswith("layer 0: composite HOOK CONTRACT violated")
    assert all(report[l]["write_identity_pass"] for l in CHAOS_LAYERS)
    hc = report[CHAOS_LAYERS[0]]["hook_contract_composite"]
    assert not hc["pass"] and hc["n_violations"] == 1
    assert 1.0 < hc["max_err_half_ulps"] <= 4.0        # one ulp of the stream
    assert not report[CHAOS_LAYERS[0]]["pass"]
    assert all(report[l]["hook_contract_composite"]["pass"]
               for l in CHAOS_LAYERS[1:])
    assert all(report[l]["hook_contract_reference"]["pass"] for l in CHAOS_LAYERS)
    # on the bf16 stream the coarser defects are off by ~the write itself,
    # i.e. hundreds of elements per layer
    for how in ("noop", "double", "scale", "wrong_layer"):
        problems, report = teacher_identity_problems(
            model, [cb], composite=ApplyPerturbed(composite, how), reference=C,
            layers=CHAOS_LAYERS)
        assert len(problems) == len(CHAOS_LAYERS), (how, problems)
        assert all("composite HOOK CONTRACT" in p for p in problems)
        assert all(report[l]["hook_contract_composite"]["n_violations"] > 50
                   for l in CHAOS_LAYERS), how


@pytest.mark.parametrize("how", list(ApplyPerturbed.HOWS))
def test_runtime_apply_defects_fail_the_gate_though_paired_write_is_right(world, how):
    """A composite whose paired_write() is exactly right but whose runtime
    apply() is not — no hooks, a doubled write, a wrong scale, hooks on the
    wrong layer, a one-ulp foreign edit — PASSED the write-level check,
    because that check re-derives both writes off the forward with the same
    helper. Each must now FAIL: the write-identity half still passes
    (nothing is wrong with the algebra), the hook-contract half refuses,
    per layer, with the reference clean."""
    model, B, C, ids = world
    cb = toy_cap_batch(ids, [2, 0, 3], ["mix", "ctl", "mix"])
    composite = SharedStateWriteObject([(1.0, B), (1.0, C), (-1.0, B)])
    problems, report = teacher_identity_problems(
        model, [cb], composite=ApplyPerturbed(composite, how), reference=C,
        layers=LAYERS)
    assert problems, how
    assert all("composite HOOK CONTRACT" in p for p in problems), problems
    flagged = {l for l in LAYERS if not report[l]["hook_contract_composite"]["pass"]}
    assert flagged == ({LAYERS[0]} if how == "bump" else set(LAYERS)), (how, flagged)
    for l in LAYERS:
        assert report[l]["write_identity_pass"]
        assert report[l]["pass"] == (l not in flagged)
        assert report[l]["hook_contract_reference"]["pass"]
        assert report[l]["hook_contract_composite"]["tolerance"].startswith(
            "per element: half an ulp of the float32 stream")
    if how != "bump":
        # a missing / doubled / mis-scaled / misplaced write is off by ~the
        # write itself: many orders of magnitude past half an fp32 ulp
        assert min(report[l]["hook_contract_composite"]["max_err_half_ulps"]
                   for l in LAYERS) > 1e6


def test_a_defective_reference_runtime_fails_its_own_contract(world):
    """The reference object gets the same treatment: its hook contract is
    checked on its own forward, so a reference whose apply() doubles its
    write is refused even though the composite is perfect — and the
    standalone :func:`hook_contract_problems` sees the same defect."""
    model, B, C, ids = world
    cb = toy_cap_batch(ids, [2, 0, 3], ["mix", "ctl", "mix"])
    composite = SharedStateWriteObject([(1.0, B), (1.0, C), (-1.0, B)])
    bad_ref = ApplyPerturbed(C, "double")
    problems, report = teacher_identity_problems(
        model, [cb], composite=composite, reference=bad_ref, layers=LAYERS)
    assert len(problems) == len(LAYERS)
    assert all("reference HOOK CONTRACT" in p for p in problems)
    for l in LAYERS:
        assert report[l]["write_identity_pass"]
        assert report[l]["hook_contract_composite"]["pass"]
        assert not report[l]["hook_contract_reference"]["pass"]
        assert not report[l]["pass"]
    p2, r2 = hook_contract_problems(model, [cb], obj=bad_ref, layers=LAYERS,
                                    label="C")
    assert len(p2) == len(LAYERS) and all("C HOOK CONTRACT" in p for p in p2)
    p3, r3 = hook_contract_problems(model, [cb], obj=C, layers=LAYERS)
    assert p3 == [] and all(r3[l]["max_abs_err"] == 0.0 for l in LAYERS)


def test_wrong_composite_still_fails_on_the_bf16_stream(chaos_world):
    """Loudness is kept: a composite missing its cancelling term (B ⊕ C,
    nothing subtracted) fails the write-level check by ORDERS of magnitude
    on the bf16 chaos toy — and by the same margin on the fp32 toy."""
    model, B, C, ids = chaos_world
    cb = toy_cap_batch(ids, [2, 0, 3, 1], ["mix", "ctl", "mix", "ctl"])
    wrong = SharedStateWriteObject([(1.0, B), (1.0, C)])
    problems, report = teacher_identity_problems(
        model, [cb], composite=wrong, reference=C, layers=CHAOS_LAYERS)
    assert len(problems) == len(CHAOS_LAYERS)
    assert all("WRITE discrepancy" in p for p in problems)
    for l in CHAOS_LAYERS:
        assert report[l]["rms_write_discrepancy"] > 1e3 * report[l]["tolerance"]
    # a mis-SIGNED cancellation (B ⊕ C ⊕ B) fails just as loudly
    signed = SharedStateWriteObject([(1.0, B), (1.0, C), (1.0, B)])
    problems, _r = teacher_identity_problems(
        model, [cb], composite=signed, reference=C, layers=CHAOS_LAYERS)
    assert len(problems) == len(CHAOS_LAYERS)


def test_shared_state_refusals(world):
    model, B, C, _ids = world
    other = toy_model(n_layers=3, seed=99)
    B_other = planted_psr(other, LAYERS, seed=11)
    with pytest.raises(RuntimeError, match="SAME model"):
        SharedStateWriteObject([(1.0, B), (-1.0, B_other)])
    B2 = planted_psr(model, [0, 1], seed=11)
    with pytest.raises(ValueError, match="layer set"):
        PSRPairedWriteObject(B, B2)
    masked = planted_psr(model, LAYERS, seed=1)
    masked.set_fit_positions("response", 0)
    with pytest.raises(ValueError, match="fit_positions"):
        PSRPairedWriteObject(masked, C)
    # the hook contract is only defined for a plain all-position object:
    # a masked PSR is refused up front (TypeError), never mis-measured
    cb = toy_cap_batch(_ids, [2, 0, 3], ["mix", "ctl", "mix"])
    with pytest.raises(TypeError, match="position-masked"):
        teacher_identity_problems(model, [cb], composite=SharedStateWriteObject([(1.0, B)]),
                                  reference=masked, layers=LAYERS)
    with pytest.raises(TypeError, match="position-masked"):
        hook_contract_problems(model, [cb], obj=masked, layers=LAYERS)
    with pytest.raises(ValueError, match="order must be"):
        with sequential_paired_apply(B, C, model, order="sideways"):
            pass


# --------------------------------------------------------------------------- #
# Hook contract: the per-layer stats stay on the device (the earlier form     #
# cost four host/device syncs per layer per forward)                          #
# --------------------------------------------------------------------------- #

from contextlib import contextmanager  # noqa: E402

_TRANSFER_NAMES = ("item", "tolist", "numpy", "cpu", "__float__", "__int__",
                   "__bool__")


@contextmanager
def _count_host_transfers():
    """Count every python-level host transfer off a tensor (item / tolist /
    numpy / cpu / float() / int() / bool()) while the block runs."""
    counts = {"n": 0, "by": {}}
    saved = {n: (n in torch.Tensor.__dict__, getattr(torch.Tensor, n))
             for n in _TRANSFER_NAMES}

    def make(name, orig):
        def wrapper(self, *a, **k):
            counts["n"] += 1
            counts["by"][name] = counts["by"].get(name, 0) + 1
            return orig(self, *a, **k)
        return wrapper

    for n, (_own, orig) in saved.items():
        setattr(torch.Tensor, n, make(n, orig))
    try:
        yield counts
    finally:
        for n, (own, orig) in saved.items():
            if own:
                setattr(torch.Tensor, n, orig)
            else:
                delattr(torch.Tensor, n)


def _reference_stats(z, h, u, valid) -> dict:
    """The 1ff64eb hook_contract_stats, verbatim (python scalars inline)."""
    from sisteer.extraction.residual import _half_ulp

    dtype = z.dtype
    zv, hv, uv = z[valid], h[valid], u[valid]
    expected = zv.float() + uv.to(dtype).float()
    err = (hv.float() - expected).abs()
    tol = _half_ulp(expected, dtype)
    n = int(err.numel())
    return {
        "max_abs_err": float(err.max()) if n else 0.0,
        "max_err_half_ulps": float((err / tol).max()) if n else 0.0,
        "n_violations": int((err > tol).sum()),
        "n_elements": n,
        "n_positions": int(zv.shape[0]),
        "n_write_active_positions": int((uv != 0).any(-1).sum()),
        "stream_dtype": str(dtype).replace("torch.", ""),
    }


_NUMERIC = ("max_abs_err", "max_err_half_ulps", "n_violations", "n_elements",
            "n_positions", "n_write_active_positions", "stream_dtype")


@pytest.mark.parametrize("how", [None, "double", "bump"])
def test_hook_contract_numbers_unchanged_with_device_side_aggregation(world, how):
    """The deferred-transfer stats equal the inline-scalar formula field
    for field — on a clean composite (zero violations) and on runtime
    defects (a doubled write: every element; a one-ulp bump at layer 0:
    one element) — per forward and merged across two batches."""
    from sisteer.extraction.residual import (_checked_capture,
                                             _finish_contract, _fp32_write,
                                             _merge_contract,
                                             _pre_and_post_capture,
                                             hook_contract_stats)

    model, B, C, ids = world
    composite = SharedStateWriteObject([(1.0, B), (1.0, C), (-1.0, B)])
    obj = composite if how is None else ApplyPerturbed(composite, how)
    cbs = [toy_cap_batch(ids, [2, 0, 3], ["mix", "ctl", "mix"]),
           toy_cap_batch(toy_ids(seed=9, batch=2, seq=6), [1, 4], ["ctl", "mix"])]
    ref_acc: dict = {l: None for l in LAYERS}
    new_acc: dict = {l: None for l in LAYERS}
    for cb in cbs:
        valid = cb.batch.attention_mask.bool()
        # the OLD numbers, off a separate (deterministic, CPU) forward
        zh = _pre_and_post_capture(model, cb, obj.apply(model), LAYERS)
        _post, stats = _checked_capture(model, cb, obj, LAYERS)
        for l in LAYERS:
            z, h = zh[l]
            with torch.no_grad():
                ref = _reference_stats(z, h, _fp32_write(obj, l, z), valid)
            st = stats[l]
            # raw stats are device tensors, not python scalars
            for k in ("max_abs_err", "max_err_half_ulps", "n_violations",
                      "n_write_active_positions"):
                assert isinstance(st[k], torch.Tensor) and st[k].dim() == 0, k
            fin = _finish_contract(st)
            assert {k: fin[k] for k in _NUMERIC} == ref, (how, l)
            # hook_contract_stats itself agrees on the same z / h
            fin2 = _finish_contract(hook_contract_stats(z, h, _fp32_write(obj, l, z), valid))
            assert {k: fin2[k] for k in _NUMERIC} == ref
            ref_acc[l] = ref if ref_acc[l] is None else {
                **ref, "max_abs_err": max(ref_acc[l]["max_abs_err"], ref["max_abs_err"]),
                "max_err_half_ulps": max(ref_acc[l]["max_err_half_ulps"],
                                         ref["max_err_half_ulps"]),
                "n_violations": ref_acc[l]["n_violations"] + ref["n_violations"],
                "n_elements": ref_acc[l]["n_elements"] + ref["n_elements"],
                "n_positions": ref_acc[l]["n_positions"] + ref["n_positions"],
                "n_write_active_positions": ref_acc[l]["n_write_active_positions"]
                + ref["n_write_active_positions"]}
            new_acc[l] = _merge_contract(new_acc[l], st)
            assert isinstance(new_acc[l]["n_violations"], torch.Tensor)  # still on device
    for l in LAYERS:
        fin = _finish_contract(new_acc[l])
        assert {k: fin[k] for k in _NUMERIC} == ref_acc[l], (how, l)
        assert fin["pass"] == (ref_acc[l]["n_violations"] == 0)
        assert fin["write_active_frac"] == pytest.approx(
            ref_acc[l]["n_write_active_positions"] / ref_acc[l]["n_positions"])
    expect_bad = set() if how is None else ({LAYERS[0]} if how == "bump" else set(LAYERS))
    assert {l for l in LAYERS if not _finish_contract(new_acc[l])["pass"]} == expect_bad


def test_hook_contract_check_transfers_once_after_the_forward(world):
    """No host transfer happens inside the hooks: a checked capture makes
    ZERO python-level transfers during the forward, the trainer's
    per-forward check makes exactly ONE (the stacked violation counts), and
    the multi-batch report transfers only when it finishes. The 1ff64eb
    hook made four per layer per forward (12 here, before the verdict)."""
    from sisteer.extraction.residual import (_checked_capture,
                                             checked_teacher_capture,
                                             hook_contract_problems)

    model, B, C, ids = world
    composite = SharedStateWriteObject([(1.0, B), (1.0, C), (-1.0, B)])
    cb = toy_cap_batch(ids, [2, 0, 3], ["mix", "ctl", "mix"])
    with _count_host_transfers() as c:
        _post, stats = _checked_capture(model, cb, composite, LAYERS)
    assert c["n"] == 0, c["by"]
    assert set(stats) == set(LAYERS)
    with _count_host_transfers() as c:
        post = checked_teacher_capture(model, cb, composite, LAYERS)
    assert c["n"] == 1 and c["by"] == {"tolist": 1}, c["by"]
    assert set(post) == set(LAYERS)
    with _count_host_transfers() as c:
        problems, report = hook_contract_problems(model, [cb, cb], obj=composite,
                                                  layers=LAYERS)
    assert problems == [] and all(report[l]["pass"] for l in LAYERS)
    # the finish: four scalars per layer, once — independent of batch count
    assert c["n"] == 4 * len(LAYERS), c["by"]
    # a defective runtime still refuses through the same one transfer
    with _count_host_transfers() as c, pytest.raises(Exception, match="HOOK CONTRACT"):
        checked_teacher_capture(model, cb, ApplyPerturbed(composite, "double"),
                                LAYERS)
    assert c["by"].get("tolist") == 1
