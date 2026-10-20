"""train_loreft probe-drift guards (2026-08-24 post-mortem — reft.py's
module docstring): the safety_refusal wave shipped a 5.2x-floor loreft
object (probe KL 0.4202 vs no-op floor 0.0803) while its OWN training CE
improved monotonically for all 15,498 steps (per-epoch means 0.2177 →
0.1859) — a gradual dissociation between the CE-on-sampled-tokens
objective and the distributional soft-KL-to-teacher metric the driver's
floor guard rules on, which no amount of watching the training loss can
see. The guards, mirroring lofit.py's: linear LR decay + grad clipping
(the reference trainer's HF defaults, dropped by this port), affine
weight decay pinned to 0.0 (torch's 0.01 default pushes W away from the
W=R no-op point every step), and the per-epoch held-out probe stop with
best-epoch restore.

CPU-only, tiny-llama fixture. Synthetic pairs only (privacy rule: no real
session/pool data in the repo).
"""

import copy

import pytest
import torch

from sisteer.data.instructions import BY_ID
from sisteer.data.pairs import Pair, pairs_to_distill_examples
from sisteer.extraction.lofit import heldout_probe
from sisteer.extraction.reft import LoReFTObject, train_loreft


INSTR = BY_ID["french"]


_TEXTS = [
    ("Say something nice.", "Bonjour le monde"),
    ("Describe a tree.", "C'est un grand arbre"),
    ("Name a color.", "Le ciel est bleu"),
    ("Greet a friend.", "Salut mon ami"),
    ("Describe rain.", "Il pleut doucement"),
    ("Name an animal.", "Un chat noir"),
]


def _many_pairs(n: int, tag: str = "p") -> list[Pair]:
    return [
        Pair(prompt=f"[{tag}{i}] {_TEXTS[i % len(_TEXTS)][0]}",
             pos=_TEXTS[i % len(_TEXTS)][1],
             neg="placeholder", instruction_id="french")
        for i in range(n)
    ]


def _sharpened(model):
    """A peaked-teacher copy of the random fixture: on the raw fixture the
    teacher's distribution is near-uniform, so the soft-KL probe stays
    immaterial however degenerate the object gets (KL between two
    near-uniform distributions is ~0 even at training CE ~10.3 — verified
    on the lofit twin of this test). Real models are confident; the
    sharpening is what makes the fixture's guard metric mean something."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.lm_head.weight.mul_(12.0)
    return m


def test_loreft_probe_drift_reproduced_and_arrested(tiny_llama):
    """The wave failure's signature, reproduced synthetically: under the
    wave's configuration (fixed LR, no clip, no stop) the shipped object's
    soft-KL-to-teacher probe drifts up epoch over epoch and lands
    materially past the 3x floor guard — the safety_refusal 5.2x, in
    miniature. The held-out stop, ALONE (decay/clip still off, identical
    trajectory), arrests it: the shipped params are the best-probe
    epoch's, and re-probing the restored object reproduces the recorded
    best point."""
    model, tok = tiny_llama
    m = _sharpened(model)
    train_exs = pairs_to_distill_examples(tok, _many_pairs(6), INSTR)
    probe_exs = pairs_to_distill_examples(tok, _many_pairs(4, "h"), INSTR)
    kw = dict(lr=0.01, epochs=6, batch_size=2, max_len=128,
              lr_decay=False,        # the wave configuration (pre-fix)
              max_grad_norm=None)    # ditto

    # -- unguarded (the wave's path): the drift ships
    obj_a = LoReFTObject(m, [1], rank=2)
    train_loreft(m, tok, obj_a, list(train_exs), heldout=[], **kw)
    act_a, floor = heldout_probe(m, tok, obj_a, probe_exs, max_len=128)
    assert act_a > 3.0 * max(floor, 1e-2), (
        f"expected the unguarded run to trip the floor guard, got "
        f"{act_a:.4f} vs floor {floor:.4f}")

    # -- guarded: same trajectory, per-epoch probe + best-epoch restore
    diag: dict = {}
    obj_b = LoReFTObject(m, [1], rank=2)
    train_loreft(m, tok, obj_b, list(train_exs), heldout=probe_exs,
                 diagnostics=diag, **kw)
    assert diag["enabled"] is True and diag["n_heldout"] == len(probe_exs)
    # curve[0] is the pre-training baseline (LoReFT inits as EXACT identity,
    # so it sits at the floor); the stopper ruled on the rest
    assert diag["heldout_curve"][0] == pytest.approx(floor, rel=1e-3)
    assert diag["stopped_epoch"] is not None  # the drift DID stop it
    assert diag["epochs_run"] < kw["epochs"]
    best = min(diag["heldout_curve"][1:])
    assert diag["best_heldout_probe"] == pytest.approx(best)
    act_b, _ = heldout_probe(m, tok, obj_b, probe_exs, max_len=128)
    # the shipped params ARE the best epoch's (same probe, same rows)
    assert act_b == pytest.approx(best, rel=1e-4)
    # and the arrest: strictly better than the unguarded final object
    assert act_b < 0.75 * act_a


def test_loreft_early_stop_restores_best_epoch_scripted(tiny_llama,
                                                        monkeypatch):
    """The stop wiring, pinned deterministically with a scripted probe:
    patience-2 stopping fires at the right epoch, the pre-training
    baseline (which LoReFT's identity init makes the GLOBAL minimum —
    curve[0] == floor) never becomes the restore target, and the restored
    params are exactly the ones recorded at the best-probe call."""
    import sisteer.extraction.reft as reft_mod

    model, tok = tiny_llama
    exs = pairs_to_distill_examples(tok, _many_pairs(4), INSTR)
    obj = LoReFTObject(model, [1], rank=2)
    script = iter([0.05, 0.4, 0.1, 0.7, 0.8, 0.9, 1.0])
    seen: list[list[torch.Tensor]] = []

    def fake_probe(model_, tok_, obj_, heldout_, *, max_len, batch_size=4):
        seen.append([p.detach().clone() for p in obj_.trainable])
        return next(script), 0.05

    monkeypatch.setattr(reft_mod, "heldout_probe", fake_probe)
    diag: dict = {}
    train_loreft(model, tok, obj, exs, epochs=6, batch_size=2, max_len=128,
                 heldout=exs[:1], diagnostics=diag)
    # calls: -1 (baseline 0.05, curve only), e1 0.4 (best), e2 0.1 (best),
    # e3 0.7 (1 stale), e4 0.8 (2 stale -> stop, patience 2)
    assert diag["stopped_epoch"] == 4
    assert diag["best_epoch"] == 2
    assert diag["epochs_run"] == 4
    assert diag["heldout_curve"] == [0.05, 0.4, 0.1, 0.7, 0.8]
    assert len(seen) == 5
    for p, snap in zip(obj.trainable, seen[2]):   # best epoch's params
        assert torch.equal(p.detach(), snap)
    assert any(not torch.equal(p.detach(), snap)  # ...not the last epoch's
               for p, snap in zip(obj.trainable, seen[-1]))


def test_loreft_optimizer_weight_decay_zero_everywhere(tiny_llama,
                                                       monkeypatch):
    """BOTH AdamW groups train at weight_decay 0.0. The rotation group
    always did (decoupled decay collapses the householder original — the
    long-standing comment); the affine group used to ride torch's 0.01
    default, which (a) deviates from the reference trainer's HF default of
    0.0 and (b) pushes W toward 0 when the method's no-op point is W=R —
    an every-step anti-identity drift of exactly the kind the wave's floor
    guard caught."""
    model, tok = tiny_llama
    exs = pairs_to_distill_examples(tok, _many_pairs(2), INSTR)
    created = []
    real_adamw = torch.optim.AdamW

    class RecordingAdamW(real_adamw):
        def __init__(self, params, **kw):
            super().__init__(params, **kw)
            created.append(self)

    monkeypatch.setattr(torch.optim, "AdamW", RecordingAdamW)
    obj = LoReFTObject(model, [1], rank=2)
    train_loreft(model, tok, obj, exs, epochs=1, batch_size=2, max_len=128)
    assert len(created) == 1
    groups = created[0].param_groups
    assert len(groups) == 2
    assert all(g["weight_decay"] == 0.0 for g in groups)


def test_loreft_lr_decay_and_clip_wiring(tiny_llama, monkeypatch):
    """Grad clipping engages at the HF-default 1.0 (and can be disabled);
    the linear decay schedule starts at full LR (a one-step run is
    bit-identical with decay on or off) and decays thereafter (a
    multi-step run differs)."""
    model, tok = tiny_llama
    exs = pairs_to_distill_examples(tok, _many_pairs(2), INSTR)
    calls: list[float] = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def rec_clip(params, max_norm, **kw):
        calls.append(float(max_norm))
        return real_clip(params, max_norm, **kw)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", rec_clip)
    kw = dict(batch_size=2, max_len=128)

    obj = LoReFTObject(model, [1], rank=2)
    train_loreft(model, tok, obj, exs, epochs=1, **kw)
    assert calls == [1.0]  # 1 batch x 1 epoch, at the HF default
    calls.clear()
    obj = LoReFTObject(model, [1], rank=2)
    train_loreft(model, tok, obj, exs, epochs=1, max_grad_norm=None, **kw)
    assert calls == []

    def _train(epochs: int, lr_decay: bool) -> list[torch.Tensor]:
        o = LoReFTObject(model, [1], rank=2)
        train_loreft(model, tok, o, exs, epochs=epochs, lr_decay=lr_decay,
                     **kw)
        return [p.detach().clone() for p in o.trainable]

    # decay starts at factor 1.0: a single-step run is bit-identical
    on, off = _train(1, True), _train(1, False)
    assert all(torch.equal(a, b) for a, b in zip(on, off))
    # ...and bites from step 2 on
    on2, off2 = _train(2, True), _train(2, False)
    assert any(not torch.equal(a, b) for a, b in zip(on2, off2))


def test_loreft_default_tiny_pool_keeps_fixed_budget(tiny_llama):
    """Backward compatibility: on a pool too small to carve a probe slice
    (every existing fixture), the defaults run the historical fixed epoch
    budget — no carve, no probe forwards, no stop — and the orthonormal-R
    training invariant still holds under the new decay/clip defaults."""
    model, tok = tiny_llama
    exs = pairs_to_distill_examples(tok, _many_pairs(2), INSTR)
    obj = LoReFTObject(model, [1], rank=2)
    before = [p.detach().clone() for p in obj.trainable]
    logged: list[dict] = []
    diag: dict = {}
    history = train_loreft(model, tok, obj, exs, epochs=2, batch_size=2,
                           max_len=128, diagnostics=diag,
                           log_fn=lambda d, s: logged.append(d))
    assert len(history) == 2  # 1 batch x 2 epochs: the FULL budget ran
    assert diag["enabled"] is False and diag["n_heldout"] == 0
    assert diag["heldout_curve"] == [] and diag["stopped_epoch"] is None
    assert diag["epochs_run"] == 2
    assert not any("loreft/heldout_probe" in d for d in logged)
    moved = any(not torch.allclose(b, p.detach())
                for b, p in zip(before, obj.trainable))
    assert moved, "no LoReFT parameter update"
    R = obj.R(1).detach()
    assert torch.allclose(R @ R.T, torch.eye(2), atol=1e-4)
