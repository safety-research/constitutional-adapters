"""fitmon — the shared production/tuning monitoring library.

CPU-only: a stub language model with analytically known logit
distributions (T4's requirement — the KL ratios are checked against
closed-form categorical KLs, not against another torch implementation of
the same ops... the EXPECTED values are computed from the distributions
directly), plus the tiny tokenizer for encoding. No store data.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sisteer.extraction.distill import DistillExample
from sisteer.extraction.fitmon import (
    CellMetricsWriter,
    FitMonitor,
    GradNormTap,
    Heartbeat,
    KLRatioProbe,
    ProbeCadence,
    TrainingDiverged,
    TrialDiverged,
    assert_finite,
    length_sorted,
    norm_groups,
    postclip_grad_norm,
)

# --------------------------------------------------------------------------- #
# The analytic stub: a "model" that emits one fixed categorical distribution   #
# per role (teacher / unsteered student / steered student), constant across    #
# positions, so every per-token KL is a closed-form categorical KL and the     #
# probe's ratios have exact expected values.                                   #
# --------------------------------------------------------------------------- #

V = 4  # stub vocabulary for the emitted logits (independent of the tokenizer)

# Three distinct distributions with hand-computable KLs.
P_TEACHER = torch.tensor([0.7, 0.1, 0.1, 0.1])
P_UNSTEERED = torch.tensor([0.25, 0.25, 0.25, 0.25])
P_STEERED = torch.tensor([0.4, 0.3, 0.2, 0.1])


def _kl(p: torch.Tensor, q: torch.Tensor) -> float:
    """Closed-form categorical KL(p || q)."""
    return float((p * (p / q).log()).sum())


class _StubLM(nn.Module):
    """Returns role-keyed constant logits. Teacher batches are recognised
    by sequence width (the SI-prompted rendering is strictly longer);
    'steered' is set by the stub object's apply()."""

    def __init__(self):
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1))  # device probe only
        self.steered = False
        self.teacher_width: int | None = None

    def forward(self, input_ids=None, attention_mask=None, **kw):
        b, s = input_ids.shape
        if self.teacher_width is not None and s >= self.teacher_width:
            dist = P_TEACHER
        elif self.steered:
            dist = P_STEERED
        else:
            dist = P_UNSTEERED
        logits = dist.log().expand(b, s, V).clone()
        return SimpleNamespace(logits=logits)


class _StubObj:
    """Minimal harness object: identity inject, apply() flips the stub
    model into its 'steered' distribution."""

    trainable: list = []

    def inject(self, rendered_prompt: str) -> str:
        return rendered_prompt

    @contextmanager
    def apply(self, model, scale: float = 1.0):
        model.steered = True
        try:
            yield None
        finally:
            model.steered = False


def _examples(n: int = 4) -> list[DistillExample]:
    # prompt_with_si strictly longer than every prompt_without, so the stub
    # can tell teacher batches from student batches by width.
    si = ("You are a careful assistant. Always follow the standing "
          "instruction in every reply, without exception. ")
    out = []
    for i in range(n):
        q = f"Question number {i}: what colour is the sky today?"
        out.append(DistillExample(
            prompt_with_si=si + q, prompt_without=q,
            completion=f"The sky is blue, case {i}.", instruction_idx=0))
    return out


def _probe(tok, n=4, batch_size=2, examples=None):
    probe = KLRatioProbe(tok, examples or _examples(), max_len=128,
                         batch_size=batch_size, n=n)
    return probe


def _stub_pair(tok, **kw):
    probe = _probe(tok, **kw)
    model = _StubLM()
    model.teacher_width = probe.batches[0][0].input_ids.shape[1]
    # sanity: teacher rendering must be strictly wider than the student's
    assert model.teacher_width > probe.batches[0][1].input_ids.shape[1]
    return probe, model


# --------------------------------------------------------------------------- #
# T4: KL-ratio math on synthetic logits with analytically known ratios         #
# --------------------------------------------------------------------------- #


def test_kl_ratios_match_closed_form(tiny_llama):
    _model, tok = tiny_llama
    probe, stub = _stub_pair(tok)
    out = probe.measure(stub, _StubObj())

    denom = _kl(P_TEACHER, P_UNSTEERED)
    residual = _kl(P_TEACHER, P_STEERED)
    displacement = _kl(P_STEERED, P_UNSTEERED)

    assert out["probe/noop_floor"] == pytest.approx(denom, rel=1e-5)
    assert out["probe/residual_kl"] == pytest.approx(residual, rel=1e-5)
    assert out["probe/displacement_kl"] == pytest.approx(displacement,
                                                         rel=1e-5)
    assert out["probe/residual_ratio"] == pytest.approx(residual / denom,
                                                        rel=1e-5)
    assert out["probe/displacement_ratio"] == pytest.approx(
        displacement / denom, rel=1e-5)


def test_denominator_computed_once_and_cached(tiny_llama):
    """The no-op floor is measured on the FIRST call and never recomputed:
    mutating the unsteered distribution afterwards moves displacement (it
    is recomputed per call) but cannot move the cached floor."""
    global P_UNSTEERED
    _model, tok = tiny_llama
    probe, stub = _stub_pair(tok)
    first = probe.measure(stub, _StubObj())
    assert probe.denom == pytest.approx(_kl(P_TEACHER, P_UNSTEERED), rel=1e-5)

    original = P_UNSTEERED
    try:
        P_UNSTEERED = torch.tensor([0.1, 0.2, 0.3, 0.4])
        second = probe.measure(stub, _StubObj())
        # cached: the floor did not move...
        assert second["probe/noop_floor"] == first["probe/noop_floor"]
        # ...while the per-call displacement leg sees the mutated world
        assert second["probe/displacement_kl"] == pytest.approx(
            _kl(P_STEERED, P_UNSTEERED), rel=1e-5)
        assert (second["probe/displacement_kl"]
                != pytest.approx(first["probe/displacement_kl"]))
    finally:
        P_UNSTEERED = original


def test_measure_without_object_measures_floor_only(tiny_llama):
    _model, tok = tiny_llama
    probe, stub = _stub_pair(tok)
    base = probe.measure(stub)  # denominator only
    assert set(base) == {"probe/noop_floor"}
    assert probe.denom is not None


def test_probe_subsets_nest(tiny_llama):
    """sha-rank selection nests: the n=2 probe's examples are a subset of
    the n=4 probe's over the same pool — smaller probes stay comparable
    to larger ones."""
    _model, tok = tiny_llama
    pool = _examples(6)
    small = _probe(tok, n=2, examples=pool)
    large = _probe(tok, n=4, examples=pool)
    assert set(small.selected_sha8) < set(large.selected_sha8)
    assert set(small.selected_idx) < set(large.selected_idx)
    # and never a pool prefix by construction of the convention
    assert small.n == 2 and large.n == 4


def test_probe_inject_reaches_the_student_encoding(tiny_llama):
    """The soft-token arms' seam: inject= must change the STUDENT ids (the
    sentinel has to be in the ids for their apply() to act) and leave the
    teacher side un-injected."""
    _model, tok = tiny_llama
    marker = tok.unk_token
    plain = _probe(tok)
    injected = KLRatioProbe(tok, _examples(), max_len=128, batch_size=2,
                            n=4, inject=lambda p: marker + p)
    t_plain, s_plain = plain.batches[0]
    t_inj, s_inj = injected.batches[0]
    assert torch.equal(t_plain.input_ids, t_inj.input_ids)  # teacher untouched
    assert s_inj.input_ids.shape[1] > s_plain.input_ids.shape[1]
    assert (s_inj.input_ids == tok.unk_token_id).any()


# --------------------------------------------------------------------------- #
# GradNormTap: pre-clip capture (tensor backward hooks)                        #
# --------------------------------------------------------------------------- #


def test_gradnormtap_sees_preclip_gradients():
    p = nn.Parameter(torch.tensor([3.0, 4.0]))
    q = nn.Parameter(torch.tensor([12.0]))
    tap = GradNormTap([p, q])
    loss = (p * torch.tensor([1.0, 1.0])).sum() * 1.0 + q.sum() * 2.0
    # grads: p -> [1, 1], q -> [2]; global norm = sqrt(1+1+4) = sqrt(6)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p, q], 1e-6)  # clip to ~zero
    pre = tap.flush()
    post = float(postclip_grad_norm([p, q]))
    assert pre == pytest.approx(math.sqrt(6.0), rel=1e-6)
    assert post < 1e-5 < pre  # pre-clip != post-clip when clipping bites
    # flushed: a second flush without a backward reports NaN (float shape)
    assert math.isnan(tap.flush())
    assert tap.flush_tensor() is None  # and None on the tensor path
    tap.remove()


def test_gradnormtap_hooks_do_not_change_grads():
    p1 = nn.Parameter(torch.tensor([1.0, 2.0]))
    p2 = nn.Parameter(torch.tensor([1.0, 2.0]))
    tap = GradNormTap([p2])
    (p1 ** 2).sum().backward()
    (p2 ** 2).sum().backward()
    assert torch.equal(p1.grad, p2.grad)
    tap.remove()


# --------------------------------------------------------------------------- #
# Probe cadence: dense early, then sparse; log flush cadence                   #
# --------------------------------------------------------------------------- #


def test_cadence_dense_then_sparse():
    c = ProbeCadence(probe_every=250, dense_every=25, dense_until=250)
    assert c.due(0) and c.due(25) and c.due(250)
    assert not c.due(26) and not c.due(275)
    assert c.due(500)
    assert not c.due(255)  # past dense_until, off the sparse grid


def test_cadence_from_train_kwargs_and_validation():
    c = ProbeCadence.from_train_kwargs(
        {"probe_every": 10, "dense_every": 2, "dense_until": 6,
         "log_every": 5})
    assert (c.probe_every, c.dense_every, c.dense_until, c.log_every) == \
        (10, 2, 6, 5)
    assert c.log_due(5) and c.log_due(10) and not c.log_due(7)
    assert c.log_due(4)  # probe-due steps always flush
    with pytest.raises(ValueError):
        ProbeCadence(probe_every=0)
    # defaults survive an empty train_kwargs
    d = ProbeCadence.from_train_kwargs({})
    assert (d.probe_every, d.dense_every, d.dense_until) == (250, 25, 250)


# --------------------------------------------------------------------------- #
# Non-finite abort (the TrialDiverged pattern, library home)                   #
# --------------------------------------------------------------------------- #


def test_assert_finite_names_the_offender():
    assert_finite({"a": 1.0, "b": 2})  # fine
    with pytest.raises(TrainingDiverged, match="train/loss"):
        assert_finite({"train/loss": float("nan")}, step=7, context="lora")
    with pytest.raises(TrainingDiverged, match="grad_norm"):
        assert_finite({"train/grad_norm": float("inf")})
    # the harness alias is the same class
    assert TrialDiverged is TrainingDiverged


# --------------------------------------------------------------------------- #
# Norm groups: per-arm reporters, sync-free (0-dim tensors)                    #
# --------------------------------------------------------------------------- #


def test_norm_groups_multilayer_lora(tiny_llama):
    from sisteer.extraction.multilayer import MultiLayerLoRAObject

    model, _tok = tiny_llama
    obj = MultiLayerLoRAObject(model, layers=[0, 1])
    obj.B[1].data.fill_(2.0)
    out = norm_groups(obj)
    for v in out.values():  # sync-free contract: tensors, not floats
        assert isinstance(v, torch.Tensor) and v.dim() == 0
    d = model.config.hidden_size
    assert float(out["norms/B_01"]) == pytest.approx(2.0 * math.sqrt(d))
    assert float(out["norms/B_00"]) == 0.0
    assert float(out["norms/B_total"]) == pytest.approx(
        float(out["norms/B_01"]))
    assert float(out["norms/A_total"]) > 0.0
    assert "norms/B_max" in out


def test_norm_groups_psr_gates(tiny_llama):
    from sisteer.extraction.psr import PSRObject

    model, _tok = tiny_llama
    obj = PSRObject(model, [0, 1], train_direction=True)
    out = norm_groups(obj)
    # zero-init: gates alive (b=1, w=0), directions zero
    assert float(out["norms/gate_param_dead_frac"]) == 0.0
    assert float(out["norms/gate_b_min"]) == pytest.approx(1.0)
    assert float(out["norms/direction_total"]) == 0.0
    for l in (0, 1):
        assert f"norms/gate_w_{l:02d}" in out
        assert f"norms/gate_b_{l:02d}" in out
        assert f"norms/direction_{l:02d}" in out
    # a provably parameter-shut gate (b <= 0, ||w|| == 0) is counted dead
    obj.gate_b[0].data.fill_(-1.0)
    assert float(norm_groups(obj)["norms/gate_param_dead_frac"]) == 0.5


def test_norm_groups_sentinel_and_vector(tiny_llama):
    from sisteer.extraction.objects import SoftTokenObject

    model, tok = tiny_llama
    st = SoftTokenObject(model, tok, sentinel=(tok.unk_token,
                                               tok.unk_token_id))
    out = norm_groups(st)
    assert float(out["norms/sentinel_embedding"]) == pytest.approx(
        float(st.param.detach().norm()))
    # single-trainable fallback: the vector norm
    vec = SimpleNamespace(trainable=[nn.Parameter(torch.tensor([3.0, 4.0]))])
    assert float(norm_groups(vec)["norms/vec_norm"]) == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# FitMonitor: buffered flush (sync discipline) + divergence abort + cadence    #
# --------------------------------------------------------------------------- #


def _monitor(tok, emit, cadence=None, **kw):
    probe, stub = _stub_pair(tok)
    return FitMonitor("stub_arm", stub, probe,
                      cadence or ProbeCadence(probe_every=100, dense_every=2,
                                              dense_until=4, log_every=3),
                      emit, **kw), stub


def test_monitor_buffers_and_flushes_at_log_cadence(tiny_llama):
    _model, tok = tiny_llama
    emitted = []
    mon, _stub = _monitor(tok, lambda rec, step: emitted.append((step, rec)))
    obj = _StubObj()
    p = nn.Parameter(torch.tensor([1.0]))
    obj.trainable = [p]
    mon.attach(obj)
    for s in range(1, 4):  # steps 1, 2, 3 — flush lands at 2 (probe) and 3
        (p * 2.0).sum().backward()
        mon.on_step(s, obj, 0.5 / s)
    steps = [s for s, _ in emitted]
    assert steps == sorted(steps)
    by_step = {s: r for s, r in emitted}
    assert set(by_step) == {1, 2, 3}
    # per-step scalars are floats after the flush, and the buffer syncs at
    # cadence: step 1's record was HELD until step 2's flush
    assert isinstance(by_step[1]["train/loss"], float)
    assert by_step[1]["train/grad_norm"] == pytest.approx(2.0)
    # probe keys ride only cadence-due steps (dense_every=2)
    assert "probe/residual_ratio" in by_step[2]
    assert "probe/heldout_loss" in by_step[2]
    assert "probe/residual_ratio" not in by_step[1]
    mon.close()


def test_monitor_nonfinite_metric_raises_at_flush(tiny_llama):
    _model, tok = tiny_llama
    mon, _stub = _monitor(tok, lambda rec, step: None)
    obj = _StubObj()
    with pytest.raises(TrainingDiverged, match="train/loss"):
        for s in range(1, 10):
            mon.on_step(s, obj, float("nan") if s == 2 else 0.1)
    # detection latency: bounded by the flush cadence (log_every=3), and
    # the record itself was emitted before the raise (emit-then-raise)


def test_monitor_final_emits_dual_kl(tiny_llama):
    _model, tok = tiny_llama
    emitted = []
    mon, _stub = _monitor(tok, lambda rec, step: emitted.append(rec))
    fin = mon.final(_StubObj())
    denom = _kl(P_TEACHER, P_UNSTEERED)
    assert fin["final/residual_ratio"] == pytest.approx(
        _kl(P_TEACHER, P_STEERED) / denom, rel=1e-5)
    assert fin["final/displacement_ratio"] == pytest.approx(
        _kl(P_STEERED, P_UNSTEERED) / denom, rel=1e-5)
    assert any("final/residual_ratio" in r for r in emitted)


def test_monitor_on_probe_is_cadence_gated(tiny_llama):
    _model, tok = tiny_llama
    emitted = []
    mon, _stub = _monitor(tok, lambda rec, step: emitted.append((step, rec)))
    obj = _StubObj()
    mon.on_probe(1, obj)   # not due (dense_every=2)
    assert emitted == []
    mon.on_probe(2, obj)   # due
    assert emitted and "probe/displacement_ratio" in emitted[0][1]


# --------------------------------------------------------------------------- #
# CellMetricsWriter: numeric-only, parseable, append-flushed                   #
# --------------------------------------------------------------------------- #


def test_cell_metrics_writer_numeric_only_and_parseable(tmp_path):
    import json

    w = CellMetricsWriter(tmp_path / "cell" / "metrics.jsonl")
    w.write({"train/loss": 0.5, "note": "attack text", "flag": True}, step=3)
    w.write({"not_numeric": "x"}, step=4)  # nothing numeric -> no line
    w.close()
    lines = [json.loads(l) for l in
             (tmp_path / "cell" / "metrics.jsonl").read_text().splitlines()]
    assert lines == [{"step": 3, "train/loss": 0.5}]  # text/bool never leaks


# --------------------------------------------------------------------------- #
# Heartbeat + length bucketing (production ports)                              #
# --------------------------------------------------------------------------- #


def test_heartbeat_phases_and_steps(capsys):
    hb = Heartbeat(total_steps=10, interval=3600.0)
    hb.phase("encoding examples")
    hb._line()  # pre-training line names the phase
    hb.step(3, 0.25, 1e-4)
    hb._line()
    hb.close()
    out = capsys.readouterr().out
    assert "encoding examples" in out
    assert "step 3/10" in out and "loss=0.2500" in out


def test_length_sorted_orders_by_rendered_length(tiny_llama):
    _model, tok = tiny_llama
    exs = [DistillExample("si " + "q" * n, "word " * n,
                          "answer " * n, 0) for n in (9, 1, 5)]
    ordered = length_sorted(tok, exs)
    lens = [len(tok(e.prompt_without)["input_ids"]) for e in ordered]
    assert lens == sorted(lens)
    assert {e.completion for e in ordered} == {e.completion for e in exs}
