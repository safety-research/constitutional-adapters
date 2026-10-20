"""LoFiT arm (arXiv 2406.01563): head-selection ranking, planted-head
recovery, the pre-o_proj application site, bias-phase training, the artifact
round trip with geometry guard, and the arm wiring on this branch (appended
at index 19, full core suite set).

Ported from the lofit-arm branch's test_methods_s2.py hunks for the two-SI
bring-in. CPU-only, tiny-llama fixture (2 layers x 4 heads x
head_dim 4).

Phase-v divergence guards (2026-08-24 post-mortem — lofit.py's module
docstring): the si_constitution wave shipped a 58.4x-floor object while
phase-A select_loss sat flat and healthy for all 30,990 of its steps —
phase-v bias tuning fell off an Adam cliff five steps into epoch 2 (bias_ce
0.24 → 12.1 in two steps) and never recovered, at fixed lr 1e-2, unclipped,
with no held-out stop. The tests below reproduce that signature (training
metric healthy, assembled object degenerate) on the tiny fixture and pin
the three guards: linear LR decay + grad clipping (the HF-Trainer defaults
the reference runs trained under), and the per-epoch held-out
soft-KL-to-teacher probe with best-epoch restore.
"""

import pytest
import torch

from sisteer.data.instructions import BY_ID
from sisteer.data.pairs import Pair, pairs_to_distill_examples
from sisteer.extraction.lofit import (
    HELDOUT_MIN,
    INIT_STD,
    LoFiTObject,
    carve_heldout,
    head_geometry,
    heldout_probe,
    select_heads,
    train_lofit,
)
from sisteer.methods import ARMS, load_object


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


# ---------------------------------------------------------------- selection


def test_lofit_select_heads_ranking_and_bounds():
    """select_heads mirrors their argsort-descending flat-index convention
    (lofit_trainer.py L304-312) and fails loudly on a bad k or shape."""
    mags = torch.tensor([[0.1, 3.0, 0.2, 0.0],
                         [2.0, 0.05, 1.0, 0.3]])
    assert select_heads(mags, 3) == [(0, 1), (1, 0), (1, 2)]
    assert select_heads(mags, 8)[-1] == (0, 3)  # the exact-zero head ranks last
    with pytest.raises(ValueError, match="k must be"):
        select_heads(mags, 0)
    with pytest.raises(ValueError, match="k must be"):
        select_heads(mags, 9)
    with pytest.raises(ValueError, match="n_layers"):
        select_heads(mags.flatten(), 1)


def test_lofit_selection_recovers_planted_important_head(tiny_llama):
    """Synthetic planted importance: zero every o_proj column EXCEPT the
    slice of head (1, 2), so only that head's scaler can move the logits
    (a dead head's CE gradient is exactly zero — its slice multiplies zeroed
    o_proj columns — and the L1 keeps it near its 1e-3 init). Phase A must
    rank the live head first, across ALL layers (no layer pin)."""
    import copy

    from sisteer.models import get_decoder_layers

    model, tok = tiny_llama
    m = copy.deepcopy(model)
    n_layers_, n_heads, hd = head_geometry(m)
    live_l, live_h = 1, 2
    with torch.no_grad():
        for l in range(n_layers_):
            W = get_decoder_layers(m)[l].self_attn.o_proj.weight
            keep = (W[:, live_h * hd: (live_h + 1) * hd].clone()
                    if l == live_l else None)
            W.zero_()
            if keep is not None:
                W[:, live_h * hd: (live_h + 1) * hd] = keep
    obj = ARMS["lofit"].train(m, tok, BY_ID["french"], _pairs(), layer=1,
                              n_bias_heads=1, epochs=8, batch_size=2,
                              max_len=64)
    assert obj.heads == [(live_l, live_h)]
    mags = obj.scaler_magnitudes
    assert mags.shape == (n_layers_, n_heads)
    flat_max = int(mags.flatten().argmax())
    assert (flat_max // n_heads, flat_max % n_heads) == (live_l, live_h)


# ---------------------------------------------------------------- application


def test_lofit_bias_applies_pre_oproj_at_selected_slices_only(tiny_llama):
    """The application site, verified at the o_proj INPUT (the method's
    identity: v_h is added to the head's attention output BEFORE o_proj,
    modeling_llama.py L366-369): the captured input delta equals
    scale * base_scale * v_h exactly in the selected head's slice at EVERY
    position (their forward bakes it in everywhere, BOS included), is zero
    elsewhere, and untouched layers see a bitwise-identical input."""
    from sisteer.models import get_decoder_layers

    model, tok = tiny_llama
    _, _, hd = head_geometry(model)
    obj = LoFiTObject(model, heads=[(1, 2)])
    obj.v.data = torch.randn(1, hd, generator=torch.Generator().manual_seed(9))
    enc = tok("the quick brown fox", return_tensors="pt")
    layers = get_decoder_layers(model)
    captured = {}

    def cap(name, layer):
        def hook(_m, inputs, _out):
            captured[(name, layer)] = inputs[0].detach().clone()

        return hook

    handles = [layers[l].self_attn.o_proj.register_forward_hook(cap("base", l))
               for l in (0, 1)]
    with torch.no_grad():
        model(**enc)
    for h in handles:
        h.remove()
    handles = [layers[l].self_attn.o_proj.register_forward_hook(cap("steer", l))
               for l in (0, 1)]
    with torch.no_grad(), obj.apply(model, scale=2.0):
        model(**enc)
    for h in handles:
        h.remove()

    # layer 0 carries no selected head: its o_proj input is untouched
    assert torch.equal(captured[("steer", 0)], captured[("base", 0)])
    delta = (captured[("steer", 1)] - captured[("base", 1)])[0]  # [S, d_in]
    sl = slice(2 * hd, 3 * hd)
    expected = (2.0 * obj.v.detach()[0]).to(delta.dtype)
    for pos in range(delta.shape[0]):  # every position, BOS included
        assert torch.allclose(delta[pos, sl], expected, atol=1e-6)
    off = torch.ones(delta.shape[-1], dtype=torch.bool)
    off[sl] = False
    assert float(delta[:, off].abs().max()) == 0.0


# ---------------------------------------------------------------- training


def test_lofit_bias_training_moves_outputs(tiny_llama):
    """Phase v actually trains: the biases move far off their N(0, 1e-3)
    init and the applied object changes the logits at scale 1 (the generic
    scale-0 identity holds via test_methods_s2's _new_objects_nonzero)."""
    model, tok = tiny_llama
    obj = ARMS["lofit"].train(model, tok, BY_ID["french"], _pairs(), layer=1,
                              n_bias_heads=2, epochs=2, batch_size=2,
                              max_len=64)
    # 2 AdamW steps at lr 1e-2 move each coord ~2e-2 >> the 1e-3 init scale
    assert float(obj.v.detach().norm()) > 5 * INIT_STD * obj.v.numel() ** 0.5
    assert obj.metadata["final_bias_ce"] is not None
    enc = tok("the quick brown fox", return_tensors="pt")
    base = _base_logits(model, enc)
    with torch.no_grad(), obj.apply(model, scale=1.0):
        steered = model(**enc).logits
    assert not torch.allclose(base, steered)


def test_lofit_train_smoke_metadata_and_roundtrip(tiny_llama, tmp_path):
    """End-to-end registry train: the artifact records the head map (the
    'where does SI-following live' result), the fixed base_scale=1.0, the
    ignored layer pin, and both phases' param counts; the blob round-trips
    through load_object with the magnitude matrix intact."""
    model, tok = tiny_llama
    n_layers_, n_heads, hd = head_geometry(model)
    obj = ARMS["lofit"].train(model, tok, BY_ID["french"], _pairs(), layer=1,
                              n_bias_heads=3, epochs=1, batch_size=2,
                              max_len=64)
    assert len(obj.heads) == 3 and len(set(obj.heads)) == 3
    assert obj.base_scale == 1.0
    md = obj.metadata
    assert md["base_scale"] == 1.0
    assert md["pinned_layer_ignored"] == 1  # the deliberate deviation
    assert md["selected_heads"] == [[l, h] for l, h in obj.heads]
    # ranking is descending by scaler magnitude
    sel = md["selected_magnitudes"]
    assert sel == sorted(sel, reverse=True)
    assert md["n_params_bias"] == 3 * hd == obj.n_params()
    assert md["n_params_selection_phase"] == n_layers_ * n_heads * hd
    assert md["total_heads"] == n_layers_ * n_heads
    assert obj.scaler_magnitudes.shape == (n_layers_, n_heads)

    path = tmp_path / "lofit.pt"
    obj.save(path)
    loaded = load_object(path, model, tok)
    assert loaded.kind == "lofit"
    assert loaded.heads == obj.heads
    assert loaded.base_scale == 1.0
    assert torch.equal(loaded.v.detach().cpu(), obj.v.detach().float().cpu())
    assert torch.equal(loaded.scaler_magnitudes, obj.scaler_magnitudes)
    assert loaded.metadata["selected_heads"] == md["selected_heads"]

    # geometry guard: an artifact from a different architecture refuses
    blob = torch.load(path, map_location="cpu", weights_only=True)
    blob["n_heads"] = 99
    with pytest.raises(ValueError, match="geometry"):
        LoFiTObject.load_blob(blob, model, tok)


def test_lofit_object_validates_heads_and_empty_is_noop(tiny_llama):
    model, tok = tiny_llama
    with pytest.raises(ValueError, match="out of range"):
        LoFiTObject(model, heads=[(0, 99)])
    with pytest.raises(ValueError, match="duplicate"):
        LoFiTObject(model, heads=[(0, 1), (0, 1)])
    # the registry build factory's fresh object (no heads) is a no-op at
    # any scale — nothing to apply yet
    empty = LoFiTObject(model, heads=[])
    enc = tok("the quick brown fox", return_tensors="pt")
    base = _base_logits(model, enc)
    with torch.no_grad(), empty.apply(model, scale=4.0):
        out = model(**enc).logits
    assert torch.equal(base, out)


# ------------------------------------------- phase-v divergence guards


def _many_pairs(n: int, tag: str = "p") -> list[Pair]:
    texts = [
        ("Say something nice.", "Bonjour le monde"),
        ("Describe a tree.", "C'est un grand arbre"),
        ("Name a color.", "Le ciel est bleu"),
        ("Greet a friend.", "Salut mon ami"),
        ("Describe rain.", "Il pleut doucement"),
        ("Name an animal.", "Un chat noir"),
        ("Describe bread.", "Le pain est frais"),
        ("Say goodbye.", "Au revoir et merci"),
    ]
    return [
        Pair(prompt=f"[{tag}{i}] {texts[i % len(texts)][0]}",
             pos=texts[i % len(texts)][1],
             neg="placeholder", instruction_id="french")
        for i in range(n)
    ]


def test_lofit_carve_heldout_deterministic_and_tiny_pool_skips():
    """The self-carve is deterministic, disjoint, capped — and a pool too
    small for a meaningful probe (< HELDOUT_MIN carved rows) is returned
    UNTOUCHED (same list object, no RNG drawn): tiny fixtures keep the
    historical fixed-budget path bit-for-bit."""
    examples = list(range(200))  # carve is type-agnostic
    train, held = carve_heldout(examples, 0.05, seed=3)
    train2, held2 = carve_heldout(examples, 0.05, seed=3)
    assert (train, held) == (train2, held2)  # deterministic
    assert len(held) == 10 and len(train) == 190
    assert set(held).isdisjoint(train)
    assert sorted(train + held) == examples
    # cap: a huge pool still carves at most HELDOUT_MAX
    _, held_big = carve_heldout(list(range(10_000)), 0.5, seed=0)
    assert len(held_big) == 16
    # tiny pool: skipped, input list returned as-is
    tiny = list(range(40))  # int(0.05 * 40) = 2 < HELDOUT_MIN
    assert 2 < HELDOUT_MIN
    train_t, held_t = carve_heldout(tiny, 0.05, seed=3)
    assert held_t == [] and train_t is tiny
    with pytest.raises(ValueError, match="heldout_frac"):
        carve_heldout(examples, 1.0, seed=0)


def test_lofit_phase_v_divergence_reproduced_and_arrested(tiny_llama):
    """The wave failure's signature, reproduced on a synthetic fixture:
    with the wave's configuration (fixed LR, no clip, no stop) a hot
    phase-v LR ships an object whose soft-KL-to-teacher probe worsens
    epoch over epoch and lands materially past the 3x floor guard, while
    phase-A select_loss stays healthy — the si_constitution dissociation
    (select_ce flat ~0.35 for 30,990 steps, shipped probe 58.4x floor).
    Then the held-out stop, ALONE (decay/clip still off, same trajectory),
    arrests it: the shipped biases are the best-probe epoch's, strictly
    better than the unguarded final ones, and re-probing the restored
    object reproduces the recorded best point.

    The lm_head sharpening makes the random fixture's teacher PEAKED, like
    a real model's: on the raw random fixture teacher and destroyed student
    are both near-uniform, so the KL probe stays immaterial no matter how
    degenerate the biases get (verified: CE 10.3 with probe 0.006) — the
    guard's signal needs a confident teacher, which is also why the wave's
    verdicts vary with each SI's teacher/floor."""
    import copy

    model, tok = tiny_llama
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.lm_head.weight.mul_(12.0)  # peaked teacher (see docstring)
    instr = BY_ID["french"]
    train_exs = pairs_to_distill_examples(tok, _many_pairs(6), instr)
    probe_exs = pairs_to_distill_examples(tok, _many_pairs(4, tag="h"), instr)
    kw = dict(layer=1, n_bias_heads=2, epochs=5, batch_size=2, max_len=128,
              lr_bias=0.1,           # hot: the divergence, compressed
              lr_decay=False,        # the wave configuration (pre-fix)
              max_grad_norm=None)    # ditto

    # -- unguarded (the wave's path): degenerate object, healthy select loss
    logged: list[dict] = []
    obj_a = train_lofit(m, tok, list(train_exs), heldout=[],
                        log_fn=lambda d, s: logged.append(d), **kw)
    act_a, floor = heldout_probe(m, tok, obj_a, probe_exs, max_len=128)
    assert act_a > 3.0 * max(floor, 1e-2), (
        f"expected the hot-LR phase v to trip the floor guard, got "
        f"{act_a:.4f} vs floor {floor:.4f}")
    # ...while the phase-A training metric stays FLAT AND HEALTHY — the
    # dissociation that made the wave failure invisible until the guard
    # (si_constitution: select_loss 0.49 start, 0.49 end, object 58.4x).
    sel = [d["lofit/select_loss"] for d in logged if "lofit/select_loss" in d]
    assert sel[-1] <= sel[0] * 1.2
    assert obj_a.metadata["early_stop"]["enabled"] is False

    # -- guarded: same trajectory, per-epoch probe + best-epoch restore
    diag: dict = {}
    obj_b = train_lofit(m, tok, list(train_exs), heldout=probe_exs,
                        diagnostics=diag, **kw)
    es = obj_b.metadata["early_stop"]
    assert es["enabled"] is True and es["n_heldout"] == len(probe_exs)
    # curve[0] is the pre-training baseline; the stopper ruled on the rest
    assert len(es["heldout_curve"]) >= 2
    assert es["stopped_epoch"] is not None  # the worsening curve DID stop it
    best = min(es["heldout_curve"][1:])
    assert es["best_heldout_probe"] == pytest.approx(best)
    act_b, _ = heldout_probe(m, tok, obj_b, probe_exs, max_len=128)
    # the shipped biases ARE the best epoch's (same probe, same rows)
    assert act_b == pytest.approx(best, rel=1e-4)
    # and the arrest: strictly better than the unguarded final object
    assert act_b < 0.75 * act_a


def test_lofit_early_stop_restores_best_epoch_scripted(tiny_llama,
                                                       monkeypatch):
    """The stop wiring, pinned deterministically with a scripted probe:
    patience-2 stopping fires at the right epoch, the pre-training baseline
    never becomes the restore target, and the restored biases are exactly
    the ones recorded at the best-probe call."""
    import sisteer.extraction.lofit as lofit_mod

    model, tok = tiny_llama
    instr = BY_ID["french"]
    exs = pairs_to_distill_examples(tok, _many_pairs(4), instr)
    script = iter([0.05, 0.4, 0.1, 0.7, 0.8, 0.9, 1.0])
    seen: list[torch.Tensor] = []

    def fake_probe(model_, tok_, obj_, heldout_, *, max_len, batch_size=4):
        seen.append(obj_.v.detach().clone())
        return next(script), 0.05

    monkeypatch.setattr(lofit_mod, "heldout_probe", fake_probe)
    obj = train_lofit(model, tok, exs, layer=1, n_bias_heads=1, epochs=6,
                      batch_size=2, max_len=128, heldout=exs[:1])
    es = obj.metadata["early_stop"]
    # calls: -1 (baseline 0.05, curve only), e1 0.4 (best), e2 0.1 (best),
    # e3 0.7 (1 stale), e4 0.8 (2 stale -> stop). The 0.05 baseline being
    # the global minimum must NOT make the init the restore target.
    assert es["stopped_epoch"] == 4
    assert es["best_epoch"] == 2
    assert es["bias_epochs_run"] == 4
    assert es["heldout_curve"] == [0.05, 0.4, 0.1, 0.7, 0.8]
    assert len(seen) == 5
    assert torch.equal(obj.v.detach(), seen[2])       # best epoch's params
    assert not torch.equal(obj.v.detach(), seen[-1])  # not the last epoch's


def test_lofit_lr_decay_and_clip_wiring(tiny_llama, monkeypatch):
    """Grad clipping engages at the HF-default 1.0 in BOTH phases (and can
    be disabled); the linear decay schedule starts at full LR (a one-step
    run is bit-identical with decay on or off) and decays thereafter (a
    multi-step run differs)."""
    model, tok = tiny_llama
    instr = BY_ID["french"]
    exs = pairs_to_distill_examples(tok, _many_pairs(2), instr)
    calls: list[float] = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def rec_clip(params, max_norm, **kw):
        calls.append(float(max_norm))
        return real_clip(params, max_norm, **kw)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", rec_clip)
    kw = dict(layer=1, n_bias_heads=1, batch_size=2, max_len=128)
    train_lofit(model, tok, exs, epochs=1, **kw)
    # 1 batch x 1 epoch x 2 phases = 2 clip calls at the HF default
    assert calls == [1.0, 1.0]
    calls.clear()
    train_lofit(model, tok, exs, epochs=1, max_grad_norm=None, **kw)
    assert calls == []

    # decay starts at factor 1.0: a single-step run is bit-identical
    on = train_lofit(model, tok, exs, epochs=1, lr_decay=True, **kw)
    off = train_lofit(model, tok, exs, epochs=1, lr_decay=False, **kw)
    assert torch.equal(on.v.detach(), off.v.detach())
    # ...and bites from step 2 on
    on2 = train_lofit(model, tok, exs, epochs=2, lr_decay=True, **kw)
    off2 = train_lofit(model, tok, exs, epochs=2, lr_decay=False, **kw)
    assert not torch.equal(on2.v.detach(), off2.v.detach())


def test_lofit_default_tiny_pool_keeps_fixed_budget(tiny_llama):
    """Backward compatibility: on a pool too small to carve a probe slice
    (every existing fixture), the defaults run the historical fixed epoch
    budget — no carve, no probe forwards, no stop — and the metadata says
    so instead of silently claiming a guard that never armed."""
    model, tok = tiny_llama
    logged: list[dict] = []
    obj = ARMS["lofit"].train(
        model, tok, BY_ID["french"], _pairs(), layer=1, n_bias_heads=1,
        epochs=2, batch_size=2, max_len=64,
        log_fn=lambda d, s: logged.append(d))
    es = obj.metadata["early_stop"]
    assert es["enabled"] is False
    assert es["n_heldout"] == 0
    assert es["heldout_curve"] == [] and es["stopped_epoch"] is None
    assert es["bias_epochs_run"] == 2
    assert not any("lofit/heldout_probe" in d for d in logged)


# ---------------------------------------------------------------- wiring


def test_lofit_arm_wiring():
    """Appended last of the trio (antidote, alphasteer, lofit — no
    longer registry-last since the alphakl rungs were appended after it;
    index 20 since the 2026-08-28 lora_r4 alias joined at index 4 next to
    its lora_r1_all sibling, see test_methods_s2's order test), with this
    repository's FULL core suite set: it was registered IFEval-only when
    ported, but here it must run the same suites as the comparable arms
    (psr_kl_all's set)."""
    assert list(ARMS)[20] == "lofit"
    assert ARMS["lofit"].suites == frozenset({"ifeval"})
    assert ARMS["lofit"].suites == ARMS["psr_kl_all"].suites
