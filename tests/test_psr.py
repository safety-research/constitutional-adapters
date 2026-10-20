"""PSR init + dead-gate guard regressions.

The all-layer KL PSR (psr_kl_all / psr_kl_all_hard) once initialized its
directions from the diffmean vectors — ~32 simultaneous full-strength
steering vectors (per-layer norms up to ~17) under the wide-open init gate
(w=0, b=1, gate == 1). A fit's step-0 KL was 9.28 against a measured 0.148
no-op floor, and the FIRST AdamW step (lr 5e-3) drove w.h + b below zero at
every token of the deep layers: ReLU dead, zero gradient to the gate AND
that layer's direction, forever (the KL path then had no anti-dead-gate
regularizer; the mse path's 0.1 * gate_reg is why psr_mse_all survived the
same init). The saved artifacts showed a contiguous deep block frozen at
gate_b=0.924214 — AdamW's exact impulse response to one nonzero gradient
followed by zero gradients — on BOTH SIs (18/32 layers, 13..30, on
safety_refusal; 11/32, 19, 20, 22..30, on si_constitution), with final
losses 4.82/5.72, ~30x WORSE than the no-op floor.

These tests pin (a) the fix, widened 2026-08-20 to remove a confound: the
two protective treatments used to be split across the objectives (zero-init
only on the KL all-layer arms, the gate regularizer only on the mse path),
so an MSE-vs-KL difference conflated the objective with its protection. Now
EVERY all-layer variant (psr_mse_all included) zero-inits — step 0 an exact
no-op — while the single-layer arms of BOTH objectives keep the informed
diffmean init, and the anti-dead-gate regularizer applies on BOTH paths at
the same 0.1 default — and (b) the guards that make a silent
repeat impossible: the dead-gate detector (check_psr_gates, wired into
train_psr and the lockstep save path) and the startup step-0-vs-no-op-floor
warning (distill.startup_loss_check, wired into train_object_kl and
train_lockstep) — and (c) the guard's usability contract, added after a
port of it hit two avoidable failures: up-front pairs-path
validation (shared storage, before any model work), keyword-only
obj/examples, the n_probe floor (a 4-example probe once yielded a monotone
false result that vanished at n=32), and the forward-free frozen-gate_b
fingerprint — including the two thresholds (FROZEN_LAYER_FLOOR,
FROZEN_DEPART_FLOOR) that keep it from false-positiving on small or
short-trained fits, where 2/2 layers coinciding 5e-5 from the 1.0 init is
expected, not pathological.

CPU-only: the tiny_llama fixture and synthetic French pairs (never real
store data).
"""

from __future__ import annotations

import warnings

import pytest
import torch

from sisteer.data.instructions import BY_ID
from sisteer.data.pairs import Pair, pairs_to_distill_examples
from sisteer.extraction.fitmon import (
    FitMonitor,
    KLRatioProbe,
    ProbeCadence,
    TrainingDiverged,
)
from sisteer.extraction.lockstep import build_lockstep_object
from sisteer.extraction.objects import train_object_kl
from sisteer.extraction.psr import (
    FROZEN_DEPART_FLOOR,
    FROZEN_LAYER_FLOOR,
    GUARD_PROBE_FLOOR,
    DeadGateError,
    ProbePairsPathError,
    PSRObject,
    check_psr_gates,
    psr_frozen_gate_b,
    psr_gate_liveness,
    train_psr,
    validate_probe_pairs_path,
)
from sisteer.methods import ARMS
from sisteer.models import hidden_size, num_layers

INSTRUCTION = BY_ID["french"]


def _pairs(n: int = GUARD_PROBE_FLOOR):
    """n synthetic French pairs (invented text only, never real store
    data). Defaults to GUARD_PROBE_FLOOR so the direct check_psr_gates
    calls below exercise the guard's REAL default floor path rather than
    a lowered one."""
    base = [
        ("Say something nice.", "Bonjour le monde", "Hello world"),
        ("Describe a tree.", "C'est un grand arbre", "It is a tall tree"),
        ("Greet the reader.", "Salut a tous", "Hi everyone"),
        ("Name a color.", "Le bleu est joli", "Blue is nice"),
    ]
    data = [(f"{base[i % 4][0]} (variant {i})",
             f"{base[i % 4][1]}, numero {i}",
             f"{base[i % 4][2]}, number {i}") for i in range(n)]
    return [Pair(prompt=p, pos=pos, neg=neg, instruction_id="french")
            for p, pos, neg in data]


def _examples(tok):
    return pairs_to_distill_examples(tok, _pairs(), INSTRUCTION)


@pytest.fixture(scope="module")
def deep_llama(tiny_llama):
    """A REAL-DEPTH (32-layer), tiny-width random llama sharing the tiny
    fixture's tokenizer. The frozen-fingerprint thresholds are calibrated
    on 32-layer artifacts, and FROZEN_LAYER_FLOOR is deliberately
    unreachable on the 2-layer fixture — exercising both sides of the
    detector needs the depth the real objects have."""
    from transformers import LlamaConfig, LlamaForCausalLM

    _, tok = tiny_llama
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=16,
                      intermediate_size=32, num_hidden_layers=32,
                      num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=512)
    model = LlamaForCausalLM(cfg).eval()
    return model, tok


# epochs=0 runs the full construction (and the embedded guards) without a
# single optimizer step — the exact state "at construction".
CONSTRUCT_ONLY = dict(epochs=0, batch_size=2, max_len=96, seed=0)


def _all_layer_dead_gate_obj(model):
    """A PSR whose every gate is closed at every token: w=0, b=-1 gives
    ReLU(0.h - 1) == 0 unconditionally."""
    obj = PSRObject(model, list(range(num_layers(model))),
                    train_direction=True)
    for l in obj.layers:
        obj.gate_b[l].data.fill_(-1.0)
    return obj


# --------------------------------------------------------------------------- #
# The fix: all-layer KL variants zero-init (exact no-op at step 0)             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("arm", ["psr_kl_all", "psr_kl_all_hard",
                                 "psr_mse_all"])
def test_all_layer_psr_is_exact_noop_at_construction(tiny_llama, arm):
    """The registry path, BOTH objectives (the 2026-08-20 confound fix put
    psr_mse_all on the same zero-init as the KL all-layer arms): directions
    are EXACTLY zero at construction, so the as-initialized object is
    bitwise the unsteered model."""
    model, tok = tiny_llama
    obj = ARMS[arm].train(model, tok, INSTRUCTION, _pairs(), layer=1,
                          **CONSTRUCT_ONLY)
    assert obj.layers == list(range(num_layers(model)))
    for l in obj.layers:
        assert torch.equal(obj.directions[l].detach(),
                           torch.zeros(hidden_size(model))), (
            f"{arm}: layer {l} direction not zero at construction")
    enc = tok(["Say something nice."], return_tensors="pt")
    with torch.no_grad():
        plain = model(**enc).logits
        with obj.apply(model, scale=1.0):
            steered = model(**enc).logits
    assert torch.equal(plain, steered), (
        f"{arm}: as-constructed object is not an exact no-op")


def test_lockstep_builder_zero_inits_psr_kl_all(tiny_llama):
    """The lockstep construction site mirrors the registry path — zero
    directions for psr_kl_all (the bitwise solo==lockstep contract needs
    both sites to agree)."""
    model, tok = tiny_llama
    obj = build_lockstep_object("psr_kl_all", model, tok, INSTRUCTION,
                                _pairs(), layer=1)
    for l in obj.layers:
        assert torch.equal(obj.directions[l].detach(),
                           torch.zeros(hidden_size(model)))


# --------------------------------------------------------------------------- #
# The fix does not over-reach: informed inits stay                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("arm", ["psr_kl", "psr_kl_hard", "psr"])
def test_single_layer_arms_keep_diffmean_init(tiny_llama, arm):
    """The SINGLE-layer arms of both objectives keep the diffmean init (one
    vector is the standard informed start; for the paper-faithful `psr` the
    init is the point): directions are NONZERO at construction. The
    all-layer arms — psr_mse_all included since the 2026-08-20 confound
    fix — zero-init instead (the test above)."""
    model, tok = tiny_llama
    obj = ARMS[arm].train(model, tok, INSTRUCTION, _pairs(), layer=1,
                          **CONSTRUCT_ONLY)
    for l in obj.layers:
        assert float(obj.directions[l].detach().norm()) > 0, (
            f"{arm}: layer {l} lost its diffmean init")


def test_lockstep_builder_keeps_diffmean_init_for_psr_kl(tiny_llama):
    model, tok = tiny_llama
    obj = build_lockstep_object("psr_kl", model, tok, INSTRUCTION,
                                _pairs(), layer=1)
    assert obj.layers == [1]
    assert float(obj.directions[1].detach().norm()) > 0


# --------------------------------------------------------------------------- #
# The regularizer applies on BOTH objectives (the 2026-08-20 confound fix)     #
# --------------------------------------------------------------------------- #

_LOSS_LOG_KEY = {"mse": "psr/mse_loss", "kl": "object_psr/kl_loss",
                 "kl_hard": "object_psr/hard_loss"}
_REG_LOG_KEY = {"mse": "psr/gate_reg", "kl": "object_psr/gate_reg",
                "kl_hard": "object_psr/gate_reg"}


@pytest.mark.parametrize("loss", ["mse", "kl", "kl_hard"])
@pytest.mark.parametrize("all_layers", [False, True])
def test_gate_regularizer_present_in_the_loss(tiny_llama, loss, all_layers):
    """Objective x layer-axis sweep: with a partially-open gate (row gate
    mass < 1, so the anti-dead-gate term is strictly positive) the
    first-step loss with reg_coeff=0.1 exceeds the reg_coeff=0.0 run's by
    exactly 0.1 * the logged regularizer value — the term is IN the loss
    on every path, mse and kl alike (before the 2026-08-20 confound fix
    the KL paths carried nothing, and the psr_kl_all fits gate-died)."""
    model, tok = tiny_llama
    layers = list(range(num_layers(model))) if all_layers else [1]

    def first_record(reg_coeff):
        obj = PSRObject(model, layers, train_direction=True)
        for l in obj.layers:
            # Partially open: g = 1e-3 > 0 at every token (alive), but the
            # row's total gate mass stays far below 1 -> the term is live.
            obj.gate_b[l].data.fill_(1e-3)
        recs: list[dict] = []

        def grab(rec, step):
            if _LOSS_LOG_KEY[loss] in rec:
                recs.append(rec)

        try:
            train_psr(model, tok, obj, _examples(tok)[:4], loss=loss,
                      epochs=1, batch_size=2, max_len=96,
                      reg_coeff=reg_coeff, log_fn=grab)
        except DeadGateError:
            pass  # near-closed gates may end the fit dead; the arithmetic
            # under test is the FIRST step, already captured by the log.
        return recs[0]

    with_reg = first_record(0.1)
    without = first_record(0.0)
    assert _REG_LOG_KEY[loss] in with_reg, (
        f"{loss}: no regularizer value logged — the term never entered "
        "the training step")
    reg_val = with_reg[_REG_LOG_KEY[loss]]
    assert reg_val > 0.0
    assert (with_reg[_LOSS_LOG_KEY[loss]] - without[_LOSS_LOG_KEY[loss]]
            == pytest.approx(0.1 * reg_val, abs=1e-6)), (
        f"{loss}: first-step loss does not carry 0.1 * gate_reg")


def test_regularizer_keeps_gates_alive_under_closure_pressure(tiny_llama):
    """The regularizer's job, isolated: a synthetic descent that PAYS for
    gate mass (mean gate value as the loss — closure pressure, the
    psr_kl_all failure mode in miniature) closes the gate at EVERY
    position without the regularizer, while the identical pressure WITH
    the 0.1 term settles with the gate still open (the term's gradient
    overpowers the per-position pressure once a row's mass drops below
    1). Keeps gates alive — not merely reports them dead."""
    model, tok = tiny_llama
    for prm in model.parameters():
        prm.requires_grad_(False)
    enc = tok("one two three four five six seven eight nine ten eleven "
              "twelve thirteen fourteen fifteen sixteen",
              return_tensors="pt")
    S = enc["input_ids"].shape[1]
    # The pressure's per-position gradient is 1/S; the regularizer's is
    # the 0.1 coefficient once row mass < 1 — S must be large enough that
    # the regularizer wins, else the test probes nothing.
    assert S >= 12

    def open_fraction(reg_coeff: float) -> float:
        obj = PSRObject(model, [1], train_direction=True)  # d stays 0
        opt = torch.optim.SGD([obj.gate_w[1], obj.gate_b[1]], lr=0.05)
        for _ in range(300):
            with obj.apply(model, scale=1.0):
                model(**enc)
                pressure = obj._last_gate[1].mean()
                reg = obj.gate_reg_term(enc["attention_mask"])
            loss = pressure + reg_coeff * reg
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad(), obj.apply(model, scale=1.0):
            model(**enc)
            g = obj._last_gate[1]
        return float((g > 0).float().mean())

    assert open_fraction(0.0) == 0.0, (
        "closure pressure alone should kill every gate position — the "
        "synthetic case is not adversarial enough to test the rescue")
    assert open_fraction(0.1) > 0.0, (
        "the 0.1 regularizer failed to keep any gate position alive "
        "under the same closure pressure")


def test_kl_reg_coeff_default_mirrors_train_psr_everywhere():
    """train_psr's reg_coeff default (0.1, the mse path's paper value) is
    what the KL path now inherits — and default_lockstep_params must
    mirror it for the lockstep-trained PSR KL arms, or the solo-vs-
    lockstep bitwise contract breaks. Non-gate arms carry no term."""
    import inspect

    from sisteer.extraction.lockstep import default_lockstep_params

    default = inspect.signature(train_psr).parameters["reg_coeff"].default
    assert default == 0.1
    for arm in ("psr_kl", "psr_kl_all"):
        assert default_lockstep_params(arm).reg_coeff == default, (
            f"{arm}: lockstep reg_coeff drifted from train_psr's default")
    for arm in ("vector", "vector_all", "lora_r1", "lora_r1_all",
                "token_bos", "token_faithful"):
        assert default_lockstep_params(arm).reg_coeff == 0.0


def test_reg_coeff_refused_for_an_object_without_gates(tiny_llama):
    """reg_coeff > 0 on an object with no gate_reg_term is a config error
    and must refuse loudly (never a silent no-term fit)."""
    from sisteer.extraction.objects import RankOneLoRAObject

    model, tok = tiny_llama
    obj = RankOneLoRAObject(model, layer=1)
    with pytest.raises(ValueError, match="gate_reg_term"):
        train_object_kl(model, tok, obj, _examples(tok)[:2], epochs=1,
                        batch_size=2, max_len=96, reg_coeff=0.1)


def test_gate_reg_term_requires_the_live_stash(tiny_llama):
    """gate_reg_term outside apply() (stash cleared) refuses — the term
    must read the gates the forward actually used, never a recompute."""
    model, tok = tiny_llama
    obj = PSRObject(model, [1])
    enc = tok("alpha beta gamma", return_tensors="pt")
    with pytest.raises(RuntimeError, match="empty gate stash"):
        obj.gate_reg_term(enc["attention_mask"])


# --------------------------------------------------------------------------- #
# The dead-gate guard                                                          #
# --------------------------------------------------------------------------- #


def test_dead_gate_guard_fires_on_a_dead_gate_object(tiny_llama):
    model, tok = tiny_llama
    obj = _all_layer_dead_gate_obj(model)
    with pytest.raises(DeadGateError, match="DEAD"):
        check_psr_gates(model, tok, obj=obj, examples=_examples(tok),
                        max_len=96)


def test_dead_gate_guard_passes_a_healthy_object(tiny_llama):
    """Default init (w=0, b=1) keeps every gate open at every token —
    liveness is exactly 1.0 per layer and nothing raises or warns."""
    model, tok = tiny_llama
    obj = PSRObject(model, list(range(num_layers(model))),
                    train_direction=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        liveness = check_psr_gates(model, tok, obj=obj,
                                   examples=_examples(tok), max_len=96)
    assert set(liveness) == set(obj.layers)
    assert all(v == 1.0 for v in liveness.values())


def test_dead_gate_guard_threshold_and_warning(tiny_llama):
    """One dead layer of two is over the default 25% bar (raises); under a
    loosened bar it still WARNS — a dead layer never passes silently."""
    model, tok = tiny_llama
    obj = PSRObject(model, list(range(num_layers(model))),
                    train_direction=True)
    obj.gate_b[0].data.fill_(-1.0)   # layer 0 dead, layer 1 alive
    exs = _examples(tok)
    liveness = psr_gate_liveness(model, tok, obj=obj, examples=exs,
                                 max_len=96)
    assert liveness[0] == 0.0 and liveness[1] == 1.0
    with pytest.raises(DeadGateError):
        check_psr_gates(model, tok, obj=obj, examples=exs, max_len=96)
    with pytest.warns(RuntimeWarning, match="dead"):
        check_psr_gates(model, tok, obj=obj, examples=exs, max_len=96,
                        max_dead_frac=0.6)


@pytest.mark.parametrize("loss", ["kl", "kl_hard", "mse"])
def test_train_psr_refuses_to_return_a_dead_gate_object(tiny_llama, loss):
    """The guard is wired into BOTH train_psr branches: a dead-gate object
    never comes back as 'trained' from any PSR loss."""
    model, tok = tiny_llama
    obj = _all_layer_dead_gate_obj(model)
    with pytest.raises(DeadGateError):
        train_psr(model, tok, obj, _examples(tok), loss=loss,
                  epochs=0, batch_size=2, max_len=96)


# --------------------------------------------------------------------------- #
# Guard usability: fail fast, fail instructively (a peer's port hit both)      #
# --------------------------------------------------------------------------- #


def test_missing_pairs_path_fails_before_any_model_work(tmp_path):
    """A nonexistent/node-invisible pairs path must fail BEFORE any model
    work — model/tok/obj are None here, proving nothing model-shaped is
    touched — and the error must explain the shared-storage requirement
    (cluster /home is node-local), not just say ENOENT."""
    missing = tmp_path / "pairs" / "nope.jsonl"
    with pytest.raises(ProbePairsPathError) as ei:
        check_psr_gates(None, None, obj=None, examples=str(missing))
    msg = str(ei.value)
    assert str(missing) in msg
    assert "SHARED storage" in msg
    assert "$SHARED/..." in msg
    assert "node-local" in msg
    # The standalone validator (the thing to call at the top of a job
    # script, before load_model) raises the same explanation.
    with pytest.raises(ProbePairsPathError, match="shared storage"):
        validate_probe_pairs_path(missing)


def test_empty_pairs_file_is_refused(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    with pytest.raises(ProbePairsPathError, match="empty"):
        validate_probe_pairs_path(p)


def test_guard_redirects_a_valid_pairs_file_to_the_loading_recipe(tmp_path):
    """Handing the guard the pairs FILE instead of the rendered examples
    used to die deep in encoding (AttributeError on a string, after the
    model load); now it names the two-line fix up front."""
    p = tmp_path / "pairs.jsonl"
    p.write_text('{"prompt": "hi", "pos": "salut", "neg": "hello"}\n')
    with pytest.raises(TypeError, match="pairs_to_distill_examples"):
        check_psr_gates(None, None, obj=None, examples=str(p))


def test_guard_refuses_empty_examples_early():
    with pytest.raises(ValueError, match="empty"):
        check_psr_gates(None, None, obj=None, examples=[])


def test_guard_refuses_below_probe_floor_without_touching_the_model():
    """The save/no-save verdict REFUSES (raises, never warns) below
    GUARD_PROBE_FLOOR examples: a 4-example probe once produced a clean
    monotone result, corroborated across two independently-trained
    objects, that vanished and partially reversed at n=32. model/tok/obj
    are None: the refusal precedes any model work."""
    thin = [object()] * 4   # 4 stand-ins; the floor check reads only len()
    with pytest.raises(ValueError, match=r"at least 64") as ei:
        check_psr_gates(None, None, obj=None, examples=thin)
    assert "GUARD_PROBE_FLOOR" in str(ei.value)
    assert "4 probe example(s)" in str(ei.value)


def test_liveness_refuses_to_silently_shrink_the_probe(tiny_llama):
    """n_probe over the available examples raises, naming both numbers —
    silent shrinkage (the min(4, len) misread) is exactly how a peer ended
    up measuring on n=4 without realising. Explicit thinness is allowed:
    the caller writes the smaller n_probe at the call site."""
    model, tok = tiny_llama
    obj = PSRObject(model, list(range(num_layers(model))),
                    train_direction=True)
    exs = _examples(tok)[:8]
    with pytest.raises(ValueError, match=r"n_probe=64 but only 8"):
        psr_gate_liveness(model, tok, obj=obj, examples=exs, max_len=96)
    liveness = psr_gate_liveness(model, tok, obj=obj, examples=exs,
                                 max_len=96, n_probe=8)
    assert all(v == 1.0 for v in liveness.values())


def test_guard_end_to_end_from_a_validated_pairs_file(tiny_llama, tmp_path):
    """The full compute-node recipe on the CPU fixture: write the pairs
    jsonl, validate the path (as a job script would, before load_model),
    load + render, run the guard — a healthy object passes with every
    gate open."""
    import json
    from dataclasses import asdict

    model, tok = tiny_llama
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(json.dumps(asdict(p)) for p in _pairs()))

    checked = validate_probe_pairs_path(path)   # before any model work
    rows = [json.loads(line)
            for line in checked.read_text().splitlines() if line.strip()]
    pairs = [Pair(**r) for r in rows]
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)
    obj = PSRObject(model, list(range(num_layers(model))),
                    train_direction=True)
    liveness = check_psr_gates(model, tok, obj=obj, examples=examples,
                               max_len=96)
    assert set(liveness) == set(obj.layers)
    assert all(v == 1.0 for v in liveness.values())


@pytest.mark.parametrize("n_frozen", [18, 11])
def test_frozen_gate_b_fingerprint_catches_open_but_frozen_gates(
        deep_llama, n_frozen):
    """Many layers sharing one DEPARTED gate_b to 6 decimals is AdamW's
    frozen signature — both REAL signatures (0.924214, i.e. 0.0758 from
    the 1.0 init, across 18/32 layers on safety_refusal and 11/32 on
    si_constitution) must fire, even though these gates are OPEN at every
    probe token (w=0, b=0.924214 -> g > 0), i.e. exactly where liveness
    alone would pass. Parameter-space, forward-free, immune to thin
    probes — and the error names the thresholds it applied."""
    model, tok = deep_llama
    obj = PSRObject(model, list(range(num_layers(model))),
                    train_direction=True)
    for l in range(13, 13 + n_frozen):   # a contiguous deep block, as real
        obj.gate_b[l].data.fill_(0.924214)
    with pytest.raises(DeadGateError,
                       match="frozen-parameter fingerprint") as ei:
        check_psr_gates(model, tok, obj=obj, examples=_examples(tok),
                        max_len=96)
    msg = str(ei.value)
    assert f"{n_frozen}/32" in msg
    assert f"FROZEN_LAYER_FLOOR={FROZEN_LAYER_FLOOR}" in msg
    assert f"FROZEN_DEPART_FLOOR={FROZEN_DEPART_FLOOR}" in msg


def test_frozen_fingerprint_ignores_barely_trained_gates_small_model(
        tiny_llama):
    """The false positive that motivated the thresholds, exactly as the
    smoke fits produced it: a 2-layer object whose gates sit at 0.99995 —
    5e-5 from the 1.0 init, gates that have barely moved after a couple
    of tiny optimizer steps, expected rather than pathological — and
    2/2 coinciding layers is unremarkable on a 2-layer object. The guard
    must pass CLEAN: no raise AND no warning."""
    model, tok = tiny_llama
    obj = PSRObject(model, list(range(num_layers(model))),
                    train_direction=True)
    for l in obj.layers:
        obj.gate_b[l].data.fill_(0.99995)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        liveness = check_psr_gates(model, tok, obj=obj,
                                   examples=_examples(tok), max_len=96)
    assert all(v == 1.0 for v in liveness.values())


def test_frozen_fingerprint_ignores_near_init_gates_at_full_depth(
        deep_llama):
    """Depth does not rescue a near-init coincidence: 32/32 layers sharing
    0.99995 clears the layer floor and the fraction bar, but the value
    sits 5e-5 from init — barely-trained, not departed-then-frozen — so
    the departure floor keeps the guard silent."""
    model, tok = deep_llama
    obj = PSRObject(model, list(range(num_layers(model))),
                    train_direction=True)
    for l in obj.layers:
        obj.gate_b[l].data.fill_(0.99995)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        check_psr_gates(model, tok, obj=obj, examples=_examples(tok),
                        max_len=96)


def test_frozen_fingerprint_passes_a_healthy_trained_object(deep_llama):
    """Distinct departed gate_b values across every layer — what
    independent per-layer training actually produces (the psr_mse_all
    artifacts: 0 frozen gates, 32/32 distinct) — form no group at all:
    no raise, no warning."""
    model, tok = deep_llama
    obj = PSRObject(model, list(range(num_layers(model))),
                    train_direction=True)
    for i, l in enumerate(obj.layers):
        obj.gate_b[l].data.fill_(0.3 + 0.017 * i)   # distinct, departed
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        check_psr_gates(model, tok, obj=obj, examples=_examples(tok),
                        max_len=96)


def test_frozen_fingerprint_layer_floor_blocks_a_tiny_departed_group(
        tiny_llama):
    """Isolate the ABSOLUTE floor from the departure band: 2/2 layers
    sharing a clearly-departed value (0.5) is 100% of a 2-layer object —
    over max_dead_frac — but 2 < FROZEN_LAYER_FLOOR coinciding layers is
    not evidence of the freeze, so the guard WARNS instead of raising
    (never silent: the shared value IS genuinely departed)."""
    model, tok = tiny_llama
    obj = PSRObject(model, list(range(num_layers(model))),
                    train_direction=True)
    for l in obj.layers:
        obj.gate_b[l].data.fill_(0.5)
    with pytest.warns(RuntimeWarning, match="identical non-init gate_b"):
        check_psr_gates(model, tok, obj=obj, examples=_examples(tok),
                        max_len=96)


def test_frozen_gate_b_groups_exclude_init_band_and_singletons(tiny_llama):
    """The fingerprint reads parameters only (no forward), excludes the
    whole NEAR-init band |b - 1.0| < FROZEN_DEPART_FLOOR — layers exactly
    at the b=1.0 init (an epochs=0 construction is not a freeze) AND
    layers that merely haven't moved much yet (a smoke fit's shared
    0.99995, 5e-5 from init, once tripped an exact-init-only exclusion)
    — and ignores a value held by a single layer."""
    model, _ = tiny_llama
    obj = PSRObject(model, list(range(8)))
    for l in (2, 3, 4):
        obj.gate_b[l].data.fill_(0.924214)
    obj.gate_b[5].data.fill_(0.5)   # singleton: not a group
    for l in (6, 7):
        obj.gate_b[l].data.fill_(0.99995)   # shared but near-init: excluded
    assert psr_frozen_gate_b(obj) == {0.924214: [2, 3, 4]}


# --------------------------------------------------------------------------- #
# The startup sanity check (step-0 loss vs the no-op floor)                    #
# --------------------------------------------------------------------------- #


def test_startup_check_warns_when_step0_is_far_above_the_noop_floor(
        tiny_llama):
    """Reproduce the bug's signature synthetically: full-strength random
    directions at every layer under the wide-open init gate put step-0 KL
    far above the no-op floor — train_object_kl must say so loudly."""
    model, tok = tiny_llama
    obj = PSRObject(model, list(range(num_layers(model))),
                    train_direction=True)
    g = torch.Generator().manual_seed(0)
    d = hidden_size(model)
    obj.set_directions(
        {l: 30.0 * torch.randn(d, generator=g) for l in obj.layers})
    with pytest.warns(RuntimeWarning, match="no-op floor"):
        train_object_kl(model, tok, obj, _examples(tok),
                        epochs=0, batch_size=2, max_len=96)


def test_startup_check_is_silent_at_the_noop_floor_and_records_both(
        tiny_llama):
    """A zero-init (exact no-op) object starts exactly AT the floor: no
    warning — and both numbers are RECORDED at step 0 either way."""
    model, tok = tiny_llama
    obj = PSRObject(model, list(range(num_layers(model))),
                    train_direction=True)   # directions stay zero
    logged: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        train_object_kl(model, tok, obj, _examples(tok),
                        epochs=0, batch_size=2, max_len=96,
                        log_fn=lambda d, step: logged.update(d))
    assert logged["object_psr/step0_loss"] == logged["object_psr/noop_floor"]


def test_startup_check_criterion_on_the_measured_bug_numbers():
    """The criterion itself, pinned on the real fit's measured numbers: KL
    9.28 against the 0.148 no-op floor (63x) must warn; starting at the
    floor, or modestly above it (an informed init), must not."""
    from sisteer.extraction.distill import startup_loss_check

    with pytest.warns(RuntimeWarning, match="no-op floor"):
        startup_loss_check(9.28, 0.148, tag="psr_kl_all")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        startup_loss_check(0.148, 0.148, tag="zero_init")   # at the floor
        startup_loss_check(0.30, 0.148, tag="informed")     # ~2x: fine



def test_check_psr_gates_defined_exactly_once():
    """Shadow guard: the hardened dead-gate guard must have EXACTLY ONE
    definition. A duplicate -- the kind an auto-merge that re-copies a block
    can introduce -- would silently REBIND the name to whichever ran last,
    reverting the probe floor and the frozen-parameter fingerprint while
    every import still succeeds. psr.py raises a loud ImportError on this at
    module load (a name collision, not a silent shadow); this mirrors it in
    CI. If the import itself started raising, that IS the guard firing --
    also the intended, louder failure."""
    import inspect
    from sisteer.extraction import psr as psr_mod
    assert inspect.getsource(psr_mod).count("def check_psr_gates") == 1


# --------------------------------------------------------------------------- #
# The 23k-pair wave fixes (2026-08-22): the reg's straight-through leak and    #
# the mse loop's LR warmup/decay + clipping (psr.py module docstring). At      #
# 5,165 steps/epoch the first AdamW step blew the loss up 60-85x, gates then   #
# died as a RATCHET (26/32, 17/32, 27/32 layers on psr_mse_all; the KL arms    #
# plateaued 34-38x over the no-op floor with ~20 layers dead) because a        #
# closed ReLU silences BOTH gradient paths — the objective's and the           #
# regularizer's own.                                                           #
# --------------------------------------------------------------------------- #


def _forward_reg(model, tok, obj, text="one two three four five six seven"):
    """One forward under apply(), returning (reg_term, attention_mask)."""
    enc = tok(text, return_tensors="pt")
    with obj.apply(model, scale=1.0):
        model(**enc)
        reg = obj.gate_reg_term(enc["attention_mask"])
    return reg, enc


def test_reg_leak_never_changes_the_term_value(tiny_llama):
    """The straight-through leak is VALUE-inert everywhere: a fully dead
    layer still pays exactly 1, and on a mixed alive/dying object the
    term with pre_acts is bit-identical to the post-ReLU-only term — the
    leak enters as leak * (neg - neg.detach()), exactly zero in the
    forward. Losses, logged metrics and the untouched exact-arithmetic
    tests (test_methods_s2) therefore see no change; only the backward
    differs, and only where a row's gate mass sits below 1."""
    from sisteer.extraction.psr import gate_regularizer

    model, tok = tiny_llama
    for prm in model.parameters():
        prm.requires_grad_(False)
    obj = PSRObject(model, [0, 1], train_direction=True)
    obj.gate_b[0].data.fill_(1e-3)   # barely alive: mass < 1, term live
    obj.gate_b[1].data.fill_(-2.0)   # fully dead: pays exactly 1
    reg, enc = _forward_reg(model, tok, obj)
    with obj.apply(model, scale=1.0):
        model(**enc)
        valid = enc["attention_mask"].bool()
        gates = {l: obj._last_gate[l] for l in (0, 1)}
        pre = {l: obj._last_pre[l] for l in (0, 1)}
        with_leak = gate_regularizer(gates, valid, pre_acts=pre)
        without = gate_regularizer(gates, valid)
    assert float(with_leak) == float(without)   # bit-identical, not approx
    dead_only = gate_regularizer({1: gates[1]}, valid,
                                 pre_acts={1: pre[1]})
    assert float(dead_only) == 1.0   # the dead layer's price is unchanged


def test_reg_leak_gives_a_dead_gate_a_reopening_gradient(tiny_llama):
    """The wave's ratchet, at the gradient level: through the post-ReLU
    gates alone (the pre-fix assembly) a fully closed gate's regularizer
    gradient is EXACTLY zero — the term reports the death it can no
    longer do anything about — while the shared assembly (gate_reg_term,
    what every trainer calls) now hands the same gate a reopening
    gradient (descent raises b, hence w.h + b)."""
    from sisteer.extraction.psr import gate_regularizer

    model, tok = tiny_llama
    for prm in model.parameters():
        prm.requires_grad_(False)

    # Pre-fix assembly: gradient identically zero at a closed gate.
    obj = PSRObject(model, [1], train_direction=True)
    obj.gate_b[1].data.fill_(-2.0)
    enc = tok("one two three four five six seven", return_tensors="pt")
    with obj.apply(model, scale=1.0):
        model(**enc)
        old = gate_regularizer({1: obj._last_gate[1]},
                               enc["attention_mask"].bool())
    old.backward()
    assert float(obj.gate_b[1].grad.abs()) == 0.0
    assert float(obj.gate_w[1].grad.abs().sum()) == 0.0

    # The shared assembly: same object state, nonzero reopening gradient.
    obj2 = PSRObject(model, [1], train_direction=True)
    obj2.gate_b[1].data.fill_(-2.0)
    reg, _ = _forward_reg(model, tok, obj2)
    reg.backward()
    assert float(obj2.gate_b[1].grad) < 0.0   # descent raises b
    assert float(obj2.gate_w[1].grad.abs().sum()) > 0.0


def test_reg_leak_reopens_a_dead_gate_under_adamw(tiny_llama):
    """End of the ratchet: a fully dead gate driven ONLY by the
    regularizer (a dead layer's objective gradient is exactly zero, so
    this IS the dead layer's whole training signal) walks back to life
    under AdamW at the production lr — the leak is the sole gradient, so
    AdamW's sign-normalized steps march b upward at ~lr per step no
    matter how small the leak slope. Pre-fix this loop was a fixed
    point: zero gradient, zero movement, dead forever."""
    model, tok = tiny_llama
    for prm in model.parameters():
        prm.requires_grad_(False)
    obj = PSRObject(model, [1], train_direction=True)
    obj.gate_b[1].data.fill_(-1.0)
    enc = tok("one two three four five six seven", return_tensors="pt")
    opt = torch.optim.AdamW([obj.gate_w[1], obj.gate_b[1]], lr=5e-3)
    reopened_at = None
    for i in range(400):   # b must climb ~1.0 at ~lr/step, plus w's help
        with obj.apply(model, scale=1.0):
            model(**enc)
            loss = 0.1 * obj.gate_reg_term(enc["attention_mask"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad(), obj.apply(model, scale=1.0):
            model(**enc)
            if float((obj._last_gate[1] > 0).float().sum()) > 0:
                reopened_at = i
                break
    assert reopened_at is not None, (
        "the regularizer's leak never reopened a fully dead gate — the "
        "23k-pair wave's permanent-death ratchet is back")


def test_scaled_mse_fit_dies_without_the_fix_and_survives_with_it(
        tiny_llama, monkeypatch):
    """The wave failure end-to-end, scaled to the fixture: the SAME
    zero-init all-layer mse fit, at a step size that oversizes the
    fixture's geometry the way lr 5e-3 oversized Llama-3.1-8B's (one
    AdamW step swings w.h by ~lr * ||h||_1, far past the b=1 margin),
    trained long enough for the ratchet to accumulate.

    Under the PRE-FIX dynamics — leak pinned to 0, constant lr, no
    clipping: exactly the wave's trainer — every gate dies at every
    probe token and train_psr refuses the object (DeadGateError), which
    is verbatim what psr_mse_all did on all three SIs (26/32, 17/32,
    27/32 dead layers). Under the shipped defaults (leak + warmup +
    cosine decay + clipping) the identical fit converges with every
    gate alive."""
    import sisteer.extraction.psr as psr_mod

    model, tok = tiny_llama
    exs = _examples(tok)[:32]
    hostile = dict(loss="mse", epochs=6, batch_size=2, max_len=96,
                   lr=0.3, seed=1)

    # A: the wave's dynamics (pre-fix), reproduced by opting every
    # protection out — the regularizer itself stays ON at its 0.1
    # default, as it was in the wave, and still cannot save the fit.
    monkeypatch.setattr(psr_mod, "GATE_REG_LEAK", 0.0)
    obj_old = PSRObject(model, list(range(num_layers(model))),
                        train_direction=True)
    with pytest.raises(DeadGateError):
        train_psr(model, tok, obj_old, exs, lr_schedule="constant",
                  warmup_steps=0, max_grad_norm=float("inf"), **hostile)

    # B: the shipped defaults on the identical fit.
    monkeypatch.setattr(psr_mod, "GATE_REG_LEAK", 0.01)
    obj_new = PSRObject(model, list(range(num_layers(model))),
                        train_direction=True)
    hist = train_psr(model, tok, obj_new, exs, **hostile)
    liveness = psr_gate_liveness(model, tok, obj=obj_new, examples=exs,
                                 max_len=96, n_probe=len(exs))
    assert all(v > 0.5 for v in liveness.values()), (
        f"fixed fit still lost its gates: {liveness}")
    # ... and the loss CONVERGED instead of diverging (the wave's fits
    # ended 10-38x over the no-op floor; this one ends near zero).
    assert hist[-1] < 0.05, f"fixed fit did not converge: {hist[-3:]}"


def test_mse_lr_schedule_warms_up_and_decays(tiny_llama):
    """The default mse schedule: lr ramps linearly from lr/warmup to the
    peak, then cosine-decays toward 0 — the step-1 AdamW impulse (85x
    loss blow-up in the wave) cannot happen at full size, and the end of
    training anneals instead of plateauing at the fixed-lr noise floor.
    lr_schedule='constant' reproduces the historical loop exactly."""
    model, tok = tiny_llama
    exs = _examples(tok)[:8]

    def lrs(**kw):
        seen = []

        def grab(rec, step):
            if "psr/lr" in rec:
                seen.append(rec["psr/lr"])

        obj = PSRObject(model, list(range(num_layers(model))),
                        train_direction=True)
        train_psr(model, tok, obj, exs, loss="mse", epochs=2, batch_size=2,
                  max_len=96, lr=1e-3, log_fn=grab, **kw)
        return seen

    warm = lrs()   # 8 steps total, auto warmup = 1
    assert len(warm) == 8
    assert warm[0] == pytest.approx(1e-3)          # warmup completes at peak
    assert max(warm) == pytest.approx(1e-3)
    assert warm[-1] < 0.2e-3                       # cosine decayed
    assert all(a >= b for a, b in zip(warm, warm[1:]))  # monotone after peak

    const = lrs(lr_schedule="constant")
    assert const == [1e-3] * 8                     # the historical loop

    explicit = lrs(warmup_steps=4)
    assert explicit[:4] == pytest.approx(
        [1e-3 * i / 4 for i in range(1, 5)])       # linear ramp to peak


def test_schedule_args_refused_on_the_kl_path(tiny_llama):
    """The KL arms' optimizer loop lives in train_object_kl (a different
    lane); train_psr must refuse schedule/clipping arguments there rather
    than silently dropping them (the authoritative-sources rule: no
    silent degradation)."""
    model, tok = tiny_llama
    obj = PSRObject(model, [1], train_direction=True)
    with pytest.raises(ValueError, match="train_object_kl"):
        train_psr(model, tok, obj, _examples(tok)[:2], loss="kl",
                  epochs=0, batch_size=2, max_len=96,
                  lr_schedule="warmup_cosine")
    with pytest.raises(ValueError, match="train_object_kl"):
        train_psr(model, tok, obj, _examples(tok)[:2], loss="kl_hard",
                  epochs=0, batch_size=2, max_len=96, max_grad_norm=1.0)


# --------------------------------------------------------------------------- #
# The probe seam (review finding F7): psr_mse_all had NO mid-fit monitoring    #
# — train_psr(..., probe_cb=None, probe_every=None) is the trainer-level       #
# contract (probe_cb(step, obj), obj = the LIVE PSRObject) the production      #
# driver and the tuning adapter wire against, on BOTH loss paths.              #
# --------------------------------------------------------------------------- #


def test_psr_probe_cb_and_probe_every_come_together(tiny_llama):
    """The bidpo convention, exactly: a callback with no cadence (or a
    cadence with no callback) is refused rather than left silently inert —
    on the mse path AND before the KL delegation (shared validation)."""
    model, tok = tiny_llama
    obj = PSRObject(model, [1], train_direction=True)
    exs = _examples(tok)[:2]
    with pytest.raises(ValueError, match="probe"):
        train_psr(model, tok, obj, exs, loss="mse", epochs=0, batch_size=2,
                  max_len=96, probe_cb=lambda s, o: None)
    with pytest.raises(ValueError, match="probe"):
        train_psr(model, tok, obj, exs, loss="mse", epochs=0, batch_size=2,
                  max_len=96, probe_every=5)
    with pytest.raises(ValueError, match="probe_every"):
        train_psr(model, tok, obj, exs, loss="mse", epochs=0, batch_size=2,
                  max_len=96, probe_cb=lambda s, o: None, probe_every=0)
    with pytest.raises(ValueError, match="probe"):
        train_psr(model, tok, obj, exs, loss="kl", epochs=0, batch_size=2,
                  max_len=96, probe_cb=lambda s, o: None)


def test_probe_cb_fires_at_cadence_with_the_live_object(tiny_llama):
    """8 optimizer steps at probe_every=3 fire at steps 3 and 6 (1-based,
    right after opt.step()), each handed the LIVE PSRObject being fit —
    not a clone (the monitor reads directions/gate_w/gate_b off it and
    apply()s it for the dual-KL probe)."""
    model, tok = tiny_llama
    obj = PSRObject(model, list(range(num_layers(model))),
                    train_direction=True)
    seen = []
    train_psr(model, tok, obj, _examples(tok)[:8], loss="mse", epochs=2,
              batch_size=2, max_len=96,
              probe_cb=lambda step, o: seen.append((step, o)),
              probe_every=3)
    assert [s for s, _ in seen] == [3, 6]
    assert all(o is obj for _, o in seen), (
        "probe_cb must receive the LIVE object being fit, never a copy")


def test_probe_cb_fires_on_the_kl_path(tiny_llama):
    """Both PSR losses carry the seam: the KL path (whose optimizer loop
    lives in train_object_kl) invokes probe_cb(step, obj) at the same
    1-based post-step cadence, with the same live object — and the
    startup record (step 0) is never probed. log_fn still receives its
    own records when both are set (the seam rides the per-step log event
    without eating it)."""
    model, tok = tiny_llama
    obj = PSRObject(model, [1], train_direction=True)
    seen, logged = [], []
    train_psr(model, tok, obj, _examples(tok)[:8], loss="kl", epochs=2,
              batch_size=2, max_len=96,
              log_fn=lambda rec, step: logged.append((step, rec)),
              probe_cb=lambda step, o: seen.append((step, o)),
              probe_every=3)
    assert [s for s, _ in seen] == [3, 6]
    assert all(o is obj for _, o in seen)
    # The startup record went to log_fn at step 0 and was NOT probed.
    assert any(s == 0 and "object_psr/step0_loss" in r for s, r in logged)
    assert [s for s, _ in logged if s >= 1] == list(range(1, 9))


def test_fitmonitor_mid_fit_metrics_on_psr_mse(tiny_llama):
    """The F7 deliverable, end to end through the registry arm the finding
    named (psr_mse_all -> methods._make_train_psr -> train_psr): a fit
    wired to a real FitMonitor emits, MID-FIT (strictly before the final
    probe), the pre-clip grad norm, the object/gate norm groups, TRUE
    gate activity, and BOTH KL ratios — none of which the dead-gate check
    or the floor-ratio guard could see until the fit had already
    finished."""
    model, tok = tiny_llama
    pairs = _pairs(8)
    exs = pairs_to_distill_examples(tok, pairs, INSTRUCTION)
    probe = KLRatioProbe(tok, exs, max_len=96, batch_size=2, n=4)
    emitted: list[tuple[int, dict]] = []
    # ratio_guard=False: this test verifies mid-fit METRIC EMISSION, not
    # divergence handling. The RANDOM tiny model's residual_ratio sits ~5.5
    # (> the 1.5 hard-stop) by construction, so the default-on
    # RatioDivergenceGuard would correctly kill the fit before the final
    # probe. The guard's own behaviour is pinned in tests/test_ratio_guard.py.
    mon = FitMonitor(
        "psr_mse_all", model, probe,
        ProbeCadence(probe_every=2, dense_every=2, dense_until=8,
                     log_every=2),
        lambda rec, step: emitted.append((step, dict(rec))),
        gate_batch=probe.batches[0][1], ratio_guard=False)
    logged: list[tuple[int, dict]] = []
    obj = ARMS["psr_mse_all"].train(
        model, tok, INSTRUCTION, pairs, layer=1, epochs=2, batch_size=2,
        max_len=96, seed=0,
        log_fn=lambda rec, step: logged.append((step, dict(rec))),
        probe_cb=lambda step, o: mon.on_probe(step, o), probe_every=2)
    total_steps = 8  # 8 examples / batch_size 2 x 2 epochs
    assert max(s for s, _ in logged) == total_steps

    # Mid-fit = strictly before the fit's last step (and therefore before
    # any final probe). The monitor probed at steps 2/4/6/8.
    mid = {s: r for s, r in emitted if s < total_steps}
    assert set(mid) == {2, 4, 6}
    for key in ("probe/residual_ratio", "probe/displacement_ratio",
                "probe/gate_dead_frac", "probe/gate_active_frac",
                "norms/direction_total", "norms/gate_b_min",
                "norms/gate_param_dead_frac"):
        assert all(key in r for r in mid.values()), (
            f"{key} missing from a mid-fit probe record")
    # Gates start wide open — TRUE activity (a forward, not parameters)
    # must see them live mid-fit.
    assert mid[2]["probe/gate_active_frac"] > 0.0
    # The pre-clip grad norm rides the trainer's native per-step log.
    grad_steps = [s for s, r in logged
                  if "psr/grad_norm" in r and r["psr/grad_norm"] > 0.0]
    assert any(s < total_steps for s in grad_steps), (
        "no mid-fit pre-clip grad norm logged")
    # The final probe still works on the returned object, AFTER the
    # mid-fit records above.
    final = mon.final(obj)
    assert "final/residual_ratio" in final
    assert "final/displacement_ratio" in final


def test_non_finite_mid_fit_probe_aborts_without_an_object(tiny_llama):
    """A diverged fit dies at probe cadence, not after the full fit: a NaN
    parameter at the first mid-fit probe makes the REAL monitor's flush
    raise fitmon.TrainingDiverged, which must propagate out of train_psr
    un-swallowed — the registry trainer never returns the object, the
    post-fit guards never bless it, and no later probe fires."""
    model, tok = tiny_llama
    pairs = _pairs(8)
    exs = pairs_to_distill_examples(tok, pairs, INSTRUCTION)
    probe = KLRatioProbe(tok, exs, max_len=96, batch_size=2, n=4)
    mon = FitMonitor(
        "psr_mse_all", model, probe,
        ProbeCadence(probe_every=2, dense_every=2, dense_until=8,
                     log_every=2),
        lambda rec, step: None, gate_batch=probe.batches[0][1])
    probed: list[int] = []
    logged: list[int] = []

    def poison_then_probe(step, o):
        probed.append(step)
        # Simulate divergence: one NaN gate parameter. Norm groups and the
        # dual-KL probe both go non-finite; the monitor's flush aborts.
        o.gate_b[o.layers[0]].data.fill_(float("nan"))
        mon.on_probe(step, o)

    with pytest.raises(TrainingDiverged):
        ARMS["psr_mse_all"].train(
            model, tok, INSTRUCTION, pairs, layer=1, epochs=2, batch_size=2,
            max_len=96, seed=0,
            log_fn=lambda rec, step: logged.append(step),
            probe_cb=poison_then_probe, probe_every=2)
    assert probed == [2], "the fit must abort at the first diverged probe"
    assert max(logged) == 2, "training continued past the diverged probe"


@pytest.mark.parametrize("loss", ["mse", "kl"])
def test_probe_seam_is_purely_observational_bitwise(tiny_llama, loss):
    """The probe_cb=None default path is byte-identical to a monitored
    fit: same seed, one fit with the seam idle and one probed by a real
    FitMonitor (dual-KL forwards, gate-activity forward, norm reads),
    bitwise-equal trained parameters and identical loss history — so
    attaching the monitor cannot perturb a trajectory, and the default
    path carries no new work at all."""
    model, tok = tiny_llama
    exs = _examples(tok)[:8]

    def fit(probe_cb=None, probe_every=None):
        obj = PSRObject(model, list(range(num_layers(model))),
                        train_direction=True)
        history = train_psr(model, tok, obj, exs, loss=loss, epochs=2,
                            batch_size=2, max_len=96, seed=0,
                            probe_cb=probe_cb, probe_every=probe_every)
        return obj, history

    plain_obj, plain_hist = fit()
    probe = KLRatioProbe(tok, exs, max_len=96, batch_size=2, n=4)
    # ratio_guard=False: the comparison needs BOTH fits to run to
    # completion — the RANDOM tiny model's residual_ratio breaches the 1.5
    # hard-stop by construction, and the guard raising is a deliberate
    # intervention, not an observation perturbing the trajectory (its
    # behaviour is pinned in tests/test_ratio_guard.py).
    mon = FitMonitor(
        "psr", model, probe,
        ProbeCadence(probe_every=2, dense_every=2, dense_until=8,
                     log_every=2),
        None, gate_batch=probe.batches[0][1], ratio_guard=False)
    mon_obj, mon_hist = fit(probe_cb=lambda s, o: mon.on_probe(s, o),
                            probe_every=2)
    assert plain_hist == mon_hist
    for l in plain_obj.layers:
        assert torch.equal(plain_obj.directions[l].detach(),
                           mon_obj.directions[l].detach())
        assert torch.equal(plain_obj.gate_w[l].detach(),
                           mon_obj.gate_w[l].detach())
        assert torch.equal(plain_obj.gate_b[l].detach(),
                           mon_obj.gate_b[l].detach())
