"""Stage-2 method arms: new intervention objects, registry, data registry.

CPU-only, tiny-llama fixture. Covers the workstream's required checks:
scale-0 == unsteered logits for every new object; the ReFT-r1 unit-norm
invariant after real optimizer steps; LoReFT prefill/decode position
gating; Steering-Tokens T=10 / loss x T^2 wiring; all-layer scale-0
identity; plus registry completeness, save/load round trips, and the
compiled-SI / behavior data registry.
"""

import pytest
import torch

from sisteer.data.instructions import (
    BEHAVIORS,
    BY_ID,
    REGISTRY,
    joined_behavior_instruction,
)
from sisteer.data.pairs import Pair, pairs_to_distill_examples
from sisteer.extraction.distill import DistillExample
from sisteer.extraction.multilayer import (
    MultiLayerLoRAObject,
    MultiLayerVectorObject,
)
from sisteer.extraction.objects import SoftTokenObject, train_object_kl
from sisteer.extraction.psr import DeadGateError, PSRObject, train_psr
from sisteer.extraction.reft import (
    LoReFTObject,
    ReFTr1Object,
    train_loreft,
    train_reft_r1,
)
from sisteer.extraction.tokens import (
    FAITHFUL_LOSS_SCALE,
    FAITHFUL_TEMPERATURE,
    AppendedSoftTokenObject,
)
from sisteer.extraction.adasteer import (
    AdaSteerObject,
    affine_pos,
    build_directions,
    fit_logistic_1d,
    train_adasteer,
)
from sisteer.extraction.alphasteer import (
    AlphaSteerObject,
    null_space_projector,
    ridge_delta,
    train_alphasteer,
)
from sisteer.extraction.did import did_combine, train_did
from sisteer.extraction.lofit import (
    INIT_STD,
    LoFiTObject,
    head_geometry,
    select_heads,
)
from sisteer.methods import ARMS, VectorObject, arms_for_suite, load_object
from sisteer.models import hidden_size
from sisteer.vectors import SteeringVector

# The 12 plan arms + the 3 hard-LL/MSE-all PSR arms + the 2
# divergence-ablation arms + the 3 brought-in defense arms of the two-SI
# set (antidote / alphasteer / lofit) + the 3 probe-ladder rungs
# (alphakl_null / alphakl_soft / alphakl_fisher) + the 2026-08-20
# diffmean floor + the 2026-08-21 bidpo_lora cell (bidpo's DPO
# objective on lora_r1's exact substrate) + the 2026-08-21 caa_static_r
# Class B floor (an earlier development branch's static_r fit,
# deployment-matched to alphasteer), tagged S_TWO_SI only (see
# _TWO_SI_ONLY_ARMS below) + the 2 stage-2 IFEval-only arms
# (did / adasteer) + the 2026-08-28 lora_r4 alias (the lora
# arm's post-re-tune name — rank-4 winner, 'all' implicit; same
# builder/trainer as lora_r1_all, registered NEXT TO its sibling rather
# than appended) = 29; the stage-2 cheap-estimator arms ride separately in
# CHEAP_ESTIMATOR_ARMS below. cast and circuit_breaker (the two-SI
# net-new arms, formerly positions 23-24) left the registry ENTIRELY on
# 2026-08-20: no ArmSpec, nothing trains or enumerates them — while
# _KIND_LOADERS keeps hydrating their fitted artifacts (pinned in
# test_cast.py / test_circuit_breaker.py).
ALL_ARMS = {
    "vector", "vector_all", "lora_r1", "lora_r1_all", "lora_r4", "token_bos",
    "token_faithful", "psr", "psr_kl", "psr_kl_all", "psr_mse_all",
    "psr_kl_hard", "psr_kl_all_hard", "loreft", "reft_r1",
    "bidpo", "vector_rkl", "vector_js",
    "antidote", "alphasteer", "lofit",
    "alphakl_null", "alphakl_soft", "alphakl_fisher",
    "diffmean", "bidpo_lora", "caa_static_r",
    "did", "adasteer",
    # lora_r4 = the canonical spelling of lora_r1_all (2026-08-28 rename;
    # same class/trainer/kind — methods.LEGACY_ARM_ALIASES), appended LAST
    # in the registry. It shares lora_r1_all's smoke path.
    "lora_r4",
    # psr_xfer_* (2026-08-28, landed at the 2026-09-02 petri-lineage merge):
    # CROSS-MODEL TRANSFERRED PSR units, appended LAST — registration and
    # dispatch pinned in tests/test_psr_xfer_registration.py.
    "psr_xfer_from_e2b", "psr_xfer_from_e2b_rw1",
    "psr_xfer_from_e4b", "psr_xfer_from_e4b_rw1",
    # The sdf-base-steering document-CE arms (2026-09-01, appended LAST):
    # psr_kl_all's / lora_r4's exact objects trained by hard-target LM CE
    # on document pools. S_TWO_SI only (see _TWO_SI_ONLY_ARMS below);
    # dedicated test homes tests/test_ce_arms.py + tests/test_ce_trainer.py
    # (their trainers need doc-shaped pairs, an explicit lr and a mandatory
    # held-out pool the generic 2-pair smoke does not carry).
    "psr_ce_all", "lora_r4_ce",
    # The sdf-base-steering DERIVED arms (2026-09-06, appended LAST): the
    # corpus fit MINUS its register-matched control fit, registered so the
    # full22 wave can SERVE them. Never trained here — build AND train
    # refuse (methods._make_build_derived / _make_train_derived), like the
    # transferred arms below; test home tests/test_derived_arms_serving.py.
    "psr_ce_all_minus_ctl_distilled", "lora_r4_ce_minus_ctl",
    # The UNDISTILLED paired write (relu-differenced-native-qwen, 2026-09-17,
    # appended LAST): PSRPairedWriteObject, kind psr_paired_write — the same
    # two parents composed on the shared pre-write state, never trained here.
    "psr_ce_all_minus_ctl",
}
# The transferred-PSR arms cannot train in-repo BY DESIGN — their trainer
# refuses (methods._make_train_transferred; the bytes come from the
# psr-transfer lane) — so they are serving/loading-only and sit outside
# the generic train smoke.
_TRANSFER_ONLY_ARMS = {"psr_xfer_from_e2b", "psr_xfer_from_e2b_rw1",
                       "psr_xfer_from_e4b", "psr_xfer_from_e4b_rw1"}
# The DERIVED arms likewise: composed from two fitted parents by the
# sdfbase builders, so BOTH their build and their train refuse — outside
# the generic train smoke for a stronger reason than the transferred ones
# (there is not even an untrained object to construct).
_DERIVED_ONLY_ARMS = {"psr_ce_all_minus_ctl_distilled",
                      "lora_r4_ce_minus_ctl",
                      "psr_ce_all_minus_ctl"}
# The net-new two-SI arms carry the S_TWO_SI suite ONLY (never a _CORE
# suite): their cells derive from configs/two_si.yaml's class_a_arms (or
# class_b_arms for caa_static_r), not from arms_for_suite, so keeping them
# out of the ifeval suite protects every stage-2 positional array decode. Each has a dedicated
# test home: tests/test_diffmean_arm.py pins diffmean's r against the
# alphakl computation it mirrors; tests/test_bidpo_lora.py pins
# bidpo_lora's substrate to lora_r1's and its objective to bidpo's;
# tests/test_caa_static_r.py pins caa_static_r's fit to the vendored peer
# functions and its deployment to alphasteer's.
_TWO_SI_ONLY_ARMS = {"diffmean", "bidpo_lora", "caa_static_r",
                     # sdf-base-steering CE arms (2026-09-01): cells come
                     # from the sdfbase configs' si_ids x explicit arms,
                     # never arms_for_suite; test homes test_ce_arms.py /
                     # test_ce_trainer.py.
                     "psr_ce_all", "lora_r4_ce",
                     # sdf-base-steering DERIVED arms (2026-09-06): named
                     # only by an explicit_cells config's cell ids, never
                     # enumerated by a suite (and never trained at all).
                     *_DERIVED_ONLY_ARMS}
# The did/adasteer arms are IFEval-suite ONLY: they join
# arms_for_suite("ifeval") and no other suite.
_IFEVAL_ONLY_ARMS = {"did", "adasteer"}
# Arms whose registry trainer runs end-to-end on the tiny 2-pair fixture in
# the generic smoke below. alphasteer is covered by its OWN train smoke
# (tests/test_alphasteer.py) with the lenient calibration kwargs its
# closed-form fit needs on the random tiny model, and the three alphakl
# rungs likewise run end-to-end in tests/test_alphakl.py (their trainer
# refuses < 4 pairs and needs the rollout/beta kwargs the generic smoke
# does not carry); did/adasteer need HarmBench jailbreak prompts the generic
# smoke does not carry, so they run in their dedicated tests below with
# synthetic inputs; everything else — antidote included, via the smoke's
# per-arm calibration kwargs — runs here.
SMOKE_ARMS = ALL_ARMS - {"alphasteer",
                         "alphakl_null", "alphakl_soft", "alphakl_fisher"} \
             - _TWO_SI_ONLY_ARMS - _IFEVAL_ONLY_ARMS - _TRANSFER_ONLY_ARMS \
             - _DERIVED_ONLY_ARMS

# stage-2 cheap-estimator arms: ifeval-only (composition membership
# deliberately not granted).
CHEAP_ESTIMATOR_ARMS = {"caa", "actadd", "actadd_front"}


def _sentinel_for(tok):
    return (tok.unk_token, tok.unk_token_id)


def _pairs():
    return [
        Pair(prompt="Say something nice.", pos="Bonjour le monde",
             neg="Hello world", instruction_id="french"),
        Pair(prompt="Describe a tree.", pos="C'est un grand arbre",
             neg="It is a tall tree", instruction_id="french"),
    ]


def _examples():
    return [
        DistillExample(
            prompt_with_si=f"Always answer in French.\n\n{p}\n",
            prompt_without=f"{p}\n",
            completion=c,
        )
        for p, c in [("Say something nice.", "Bonjour le monde"),
                     ("Describe a tree.", "C'est un grand arbre")]
    ]


def _base_logits(model, enc):
    with torch.no_grad():
        return model(**enc).logits


# -- scale-0 identity for every new object --------------------------------------


def _new_objects_nonzero(model, tok):
    """Every new stage-2 object, with parameters perturbed away from the
    identity so the scale-0 check is not vacuous."""
    d = hidden_size(model)
    g = torch.Generator().manual_seed(7)

    v_all = MultiLayerVectorObject(model)
    for l in v_all.layers:
        v_all.params[l].data = 0.1 * torch.randn(d, generator=g)

    l_all = MultiLayerLoRAObject(model)
    for l in l_all.layers:
        l_all.B[l].data = 0.1 * torch.randn(d, generator=g)

    psr = PSRObject(model, [1])
    psr.set_directions({1: torch.randn(d, generator=g)})
    psr.gate_w[1].data = 0.1 * torch.randn(d, generator=g)

    loreft = LoReFTObject(model, [1], rank=2)
    with torch.no_grad():
        loreft.learned[1].bias.add_(1.0)  # break the W=R,b=0 identity init

    reft = ReFTr1Object(model, layer=1)
    reft.w.data = 0.1 * torch.randn(d, generator=g)
    reft.max_act = 3.25

    alpha = AlphaSteerObject(
        model, layer=1, M=0.1 * torch.randn(d, d, generator=g),
        base_scale=2.0, refusal=0.1 * torch.randn(d, generator=g))

    ada_params = {
        k: 0.1 * torch.randn(d, generator=g)
        for k in ("v_RD", "v_HD", "mu_a_rd", "mu_b_rd", "mu_a_hd", "mu_b_hd")
    }
    ada_params["rlaw"] = {"w": 0.3, "b": 0.1}
    ada_params["hlaw"] = {"w": -0.2, "b": 0.05}
    ada = AdaSteerObject(model, layer=1, params=ada_params, base_scale=2.0)

    lofit = LoFiTObject(model, heads=[(0, 1), (1, 2)])
    lofit.v.data = 0.5 * torch.randn(2, lofit.head_dim, generator=g)

    return {"vector_all": v_all, "lora_all": l_all, "psr": psr,
            "loreft": loreft, "reft_r1": reft,
            "alphasteer": alpha, "adasteer": ada, "lofit": lofit}


def test_scale_zero_is_unsteered_for_every_new_object(tiny_llama):
    model, tok = tiny_llama
    enc = tok("the quick brown fox jumps", return_tensors="pt")
    base = _base_logits(model, enc)
    for name, obj in _new_objects_nonzero(model, tok).items():
        with torch.no_grad():
            with obj.apply(model, scale=0.0):
                zeroed = model(**enc).logits
            with obj.apply(model, scale=1.0):
                steered = model(**enc).logits
        assert torch.equal(base, zeroed), f"{name}: scale-0 is not a no-op"
        assert not torch.allclose(base, steered), (
            f"{name}: scale-1 changed nothing — the scale-0 check is vacuous"
        )


def test_scale_zero_is_unsteered_for_token_kinds(tiny_llama):
    """With the sentinel PRESENT in the prompt, scale 0 passes its
    native embedding through — bitwise identical to running the injected
    ids with no hook — instead of zeroing it."""
    model, tok = tiny_llama
    for cls in (SoftTokenObject, AppendedSoftTokenObject):
        obj = cls(model, tok, sentinel=_sentinel_for(tok))
        obj.param.data = torch.randn_like(obj.param.data)
        enc = tok(obj.inject("the quick brown fox"), return_tensors="pt")
        base = _base_logits(model, enc)  # sentinel ids, hook inactive
        with torch.no_grad():
            with obj.apply(model, scale=0.0):
                zeroed = model(**enc).logits
            with obj.apply(model, scale=1.0):
                steered = model(**enc).logits
        assert torch.equal(base, zeroed), f"{cls.__name__}: scale-0 not a no-op"
        assert not torch.allclose(base, steered), (
            f"{cls.__name__}: scale-1 changed nothing — the check is vacuous"
        )


def test_skip_bos_masks_true_bos_at_batch_gt1_left_padding(tiny_llama):
    """Root fix: under generate.py's forced LEFT padding, the true
    BOS sits at pad_len per row, not at global index 0. The per-row
    attention-mask BOS masking must leave each row's REAL first token
    unsteered while later real positions move."""
    from sisteer.hooks import steering
    from sisteer.models import get_decoder_layers

    model, tok = tiny_llama
    d = hidden_size(model)
    torch.manual_seed(3)
    vec = torch.randn(d) * 3
    prompts = ["alpha beta gamma delta epsilon zeta", "one two"]
    prev = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = tok(prompts, return_tensors="pt", padding=True)
    finally:
        tok.padding_side = prev
    bos_pos = enc["attention_mask"].argmax(dim=1)
    assert int(bos_pos[1]) > 0, "second row must actually be left-padded"

    captured = {}

    def capture(name):
        def hook(_m, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            captured[name] = hidden.detach().clone()
            return None

        return hook

    next_block = get_decoder_layers(model)[1]  # steer layer 0, observe at 1
    h = next_block.register_forward_pre_hook(capture("base"), with_kwargs=True)
    with torch.no_grad():
        model(**enc)
    h.remove()
    h = next_block.register_forward_pre_hook(capture("steered"), with_kwargs=True)
    with torch.no_grad(), steering(model, {0: vec}, scale=1.0, skip_bos=True):
        model(**enc)
    h.remove()

    base, steered = captured["base"], captured["steered"]
    seq_len = base.shape[1]
    for row in range(2):
        p = int(bos_pos[row])
        assert torch.allclose(base[row, p], steered[row, p]), (
            f"row {row}: true BOS at {p} was steered"
        )
        for pos in range(p + 1, seq_len):
            assert not torch.allclose(base[row, pos], steered[row, pos]), (
                f"row {row}: real position {pos} not steered"
            )


def test_skip_bos_holds_across_sequential_forwards(tiny_llama):
    """A handle held open across several full forwards (one obj.apply
    around a question loop / several training batches) must mask the BOS
    in EVERY forward, not just the first — the position counter resets on
    each multi-token forward."""
    from sisteer.hooks import steering
    from sisteer.models import get_decoder_layers

    model, tok = tiny_llama
    d = hidden_size(model)
    torch.manual_seed(4)
    vec = torch.randn(d) * 3
    enc = tok("alpha beta gamma", return_tensors="pt")

    captured = {}

    def capture(name):
        def hook(_m, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            captured[name] = hidden.detach().clone()
            return None

        return hook

    next_block = get_decoder_layers(model)[1]
    h = next_block.register_forward_pre_hook(capture("base"), with_kwargs=True)
    with torch.no_grad():
        model(**enc)
    h.remove()

    with torch.no_grad(), steering(model, {0: vec}, scale=1.0, skip_bos=True):
        model(**enc)  # first forward
        h = next_block.register_forward_pre_hook(capture("second"),
                                                 with_kwargs=True)
        model(**enc)  # SECOND forward under the same handle
        h.remove()
    assert torch.allclose(captured["base"][0, 0], captured["second"][0, 0]), (
        "second forward steered the BOS — the counter did not reset"
    )
    assert not torch.allclose(captured["base"][0, 1:], captured["second"][0, 1:])


def test_all_layer_objects_start_at_identity(tiny_llama):
    """Untrained all-layer objects are exactly the unsteered model (vector
    zeros; LoRA B zero-init), even at large scale."""
    model, tok = tiny_llama
    enc = tok("the quick brown fox", return_tensors="pt")
    base = _base_logits(model, enc)
    for obj in (MultiLayerVectorObject(model), MultiLayerLoRAObject(model)):
        with torch.no_grad(), obj.apply(model, scale=4.0):
            out = model(**enc).logits
        assert torch.equal(base, out)


# -- ReFT-r1 ---------------------------------------------------------------------


def test_reft_r1_axbench_init_and_unit_norm_after_steps(tiny_llama):
    model, tok = tiny_llama
    d = hidden_size(model)
    obj = ReFTr1Object(model, layer=1)
    # Their init: the 0.01 fill — deliberately NOT unit norm and NOT a
    # no-op at scale 1 (the module docstring's documented deviation from
    # the harness zero-init convention).
    assert torch.equal(obj.w.detach(), torch.full((d,), 0.01))
    history = train_reft_r1(
        model, tok, obj, _examples(), epochs=2, batch_size=1, max_len=64,
    )
    assert len(history) >= 3  # >= 3 real AdamW steps taken
    assert abs(float(obj.w.detach().norm()) - 1.0) < 1e-5  # on the sphere
    # the detection pass ran and produced the persisted calibration
    assert obj.max_act is not None and obj.max_act >= 0.0


def test_reft_r1_uniform_edit_with_topk_mean_magnitude(tiny_llama):
    """The LsReFT forward: EVERY intervened (non-BOS) position moves by
    the SAME delta m*w, where m is the mean of the k largest ReLU(h.w)
    over the sequence — not a per-position edit at the top-k positions
    (that was the pre-rewrite, non-AxBench behavior)."""
    import torch.nn.functional as F

    from sisteer.extraction.capture import capture_layer_outputs

    model, tok = tiny_llama
    d = hidden_size(model)
    obj = ReFTr1Object(model, layer=1, topk=2)
    enc = tok("the quick brown fox jumps over the lazy dog",
              return_tensors="pt")
    with capture_layer_outputs(model, [1]) as cap:
        with torch.no_grad():
            model(**enc)
        base_h = cap[1].clone()
    # aim w at a real hidden state so some latents are guaranteed active
    w = base_h[0, 2].float()
    obj.w.data = w / w.norm()
    with obj.latent_context(model, attn_mask=enc["attention_mask"]):
        with capture_layer_outputs(model, [1]) as cap:
            with torch.no_grad():
                model(**enc)
            steered_h = cap[1].clone()

    z = F.relu(base_h[0].float() @ obj.w.detach())  # [S]
    z[0] = 0.0  # BOS is not an intervened position
    m = z.topk(2).values.mean()
    assert float(m) > 0, "test premise: some latent must be active"
    expected = m * obj.w.detach()
    delta = (steered_h - base_h)[0]
    S = delta.shape[0]
    assert torch.allclose(delta[0], torch.zeros(d), atol=1e-6)  # BOS untouched
    for pos in range(1, S):
        assert torch.allclose(delta[pos], expected, atol=1e-4), (
            f"position {pos}: edit is not the uniform m*w"
        )


def test_reft_r1_l1_targets_only_non_topk_mass(tiny_llama):
    """The L1 term penalizes the latents the top-k DROPPED: zero when all
    activation mass sits inside the top-k, positive when mass falls
    outside it (the pre-rewrite code penalized the survivors —
    sign-flipped sparsity pressure)."""
    from sisteer.extraction.capture import capture_layer_outputs

    model, tok = tiny_llama
    enc = tok("the quick brown fox jumps over the lazy dog",
              return_tensors="pt")
    with capture_layer_outputs(model, [1]) as cap:
        with torch.no_grad():
            model(**enc)
        base_h = cap[1].clone()
    # >= 2 active latents by construction (asserted below)
    w = (base_h[0, 1] + base_h[0, 2]).float()
    w = w / w.norm()
    z_pre = torch.relu(base_h[0].float() @ w)
    z_pre[0] = 0.0  # BOS excluded from the intervened positions
    assert int((z_pre > 0).sum()) >= 2, "test premise: need >= 2 active latents"

    def nontopk_mass(topk: int) -> float:
        obj = ReFTr1Object(model, layer=1, topk=topk)
        obj.w.data = w.clone()
        with obj.latent_context(model, attn_mask=enc["attention_mask"]):
            with torch.no_grad():
                model(**enc)
            nontopk = obj._last_nontopk.clone()
            valid = obj._last_valid.clone()
        nv = valid.to(nontopk.dtype)
        return float((nontopk * nv).sum() / nv.sum().clamp_min(1.0))

    S = enc["input_ids"].shape[1]
    assert nontopk_mass(topk=S) == 0.0  # everything inside the top-k
    assert nontopk_mass(topk=1) > 0.0  # active mass outside the top-k


def test_reft_r1_max_act_persists_through_save_load(tiny_llama, tmp_path):
    """AxBench's factor*max_act calibration must survive the artifact
    round trip — eval never retrains (the driver loads saved objects)."""
    model, tok = tiny_llama
    obj = ReFTr1Object(model, layer=1)
    obj.w.data = 0.1 * torch.randn(hidden_size(model),
                                   generator=torch.Generator().manual_seed(11))
    obj.max_act = 4.75
    path = tmp_path / "reft.pt"
    obj.save(path)
    loaded = load_object(path, model, tok)
    assert loaded.kind == "reft_r1"
    assert loaded.max_act == pytest.approx(4.75)
    # an artifact saved before calibration loads with max_act absent
    obj.max_act = None
    obj.save(path)
    assert load_object(path, model, tok).max_act is None


# -- LoReFT ------------------------------------------------------------------------


def test_loreft_prefill_edits_and_decode_passes_through(tiny_llama):
    model, tok = tiny_llama
    obj = LoReFTObject(model, [1], rank=2)
    with torch.no_grad():
        obj.learned[1].bias.add_(1.0)

    multi = tok("the quick brown fox jumps", return_tensors="pt")
    base_multi = _base_logits(model, multi)
    with torch.no_grad(), obj.apply(model, scale=1.0):
        steered_multi = model(**multi).logits
    assert not torch.allclose(base_multi, steered_multi)  # prefill edited

    single = {k: v[:, :1] for k, v in multi.items()}  # seq_len == 1 forward
    base_single = _base_logits(model, single)
    with torch.no_grad(), obj.apply(model, scale=1.0):
        steered_single = model(**single).logits
    assert torch.equal(base_single, steered_single)  # decode step untouched


def test_loreft_prompt_lens_gate_edits_prompt_positions_only(tiny_llama):
    from sisteer.extraction.capture import capture_layer_outputs

    model, tok = tiny_llama
    obj = LoReFTObject(model, [1], rank=2)
    with torch.no_grad():
        obj.learned[1].bias.add_(1.0)
    enc = tok("the quick brown fox jumps", return_tensors="pt")

    with capture_layer_outputs(model, [1]) as cap:
        with torch.no_grad():
            model(**enc)
        base_h = cap[1].clone()
    try:
        obj.set_prompt_lens(torch.tensor([2]))
        # capture registered INSIDE apply -> sees the post-edit hidden
        with obj.apply(model, scale=1.0):
            with capture_layer_outputs(model, [1]) as cap:
                with torch.no_grad():
                    model(**enc)
                steered_h = cap[1].clone()
    finally:
        obj.set_prompt_lens(None)
    assert not torch.allclose(steered_h[0, :2], base_h[0, :2])
    assert torch.equal(steered_h[0, 2:], base_h[0, 2:])


def test_loreft_orthonormal_R_survives_training(tiny_llama):
    model, tok = tiny_llama
    obj = LoReFTObject(model, [1], rank=2)
    before = [p.detach().clone() for p in obj.trainable]
    history = train_loreft(
        model, tok, obj, _examples(), epochs=2, batch_size=1, max_len=64,
    )
    assert len(history) >= 3
    moved = any(
        not torch.allclose(b, p.detach())
        for b, p in zip(before, obj.trainable)
    )
    assert moved, "no LoReFT parameter update"
    R = obj.R(1).detach()
    assert torch.allclose(R @ R.T, torch.eye(2), atol=1e-4)


# -- Steering Tokens (Radevski-faithful) ------------------------------------------


def test_appended_token_sentinel_position_and_noop_without_it(tiny_llama):
    model, tok = tiny_llama
    obj = AppendedSoftTokenObject(model, tok, sentinel=_sentinel_for(tok))
    rendered = "Say something nice.\n"
    assert obj.inject(rendered) == rendered + obj.sentinel_text  # appended
    bos_obj = SoftTokenObject(model, tok, sentinel=_sentinel_for(tok))
    assert bos_obj.inject(rendered).startswith(obj.sentinel_text)  # contrast

    # On a prompt WITHOUT the sentinel the hook is a no-op at any scale.
    enc = tok("the quick brown fox", return_tensors="pt")
    base = _base_logits(model, enc)
    with torch.no_grad(), obj.apply(model, scale=3.0):
        out = model(**enc).logits
    assert torch.equal(base, out)


def test_token_faithful_T10_and_loss_x_T2_wiring(tiny_llama):
    model, tok = tiny_llama
    sent = _sentinel_for(tok)
    exs = _examples()
    kw = dict(epochs=1, batch_size=2, max_len=64)

    def first_loss(temperature, loss_scale):
        obj = AppendedSoftTokenObject(model, tok, sentinel=sent)
        hist = train_object_kl(model, tok, obj, exs,
                               temperature=temperature,
                               loss_scale=loss_scale, **kw)
        return hist[0]

    assert FAITHFUL_TEMPERATURE == 10.0
    assert FAITHFUL_LOSS_SCALE == FAITHFUL_TEMPERATURE**2 == 100.0
    scaled = first_loss(10.0, 100.0)
    unscaled = first_loss(10.0, 1.0)
    t1 = first_loss(1.0, 1.0)
    # identical init + identical batch => the first-step loss (and hence
    # the gradient, by linearity) is exactly x100
    assert scaled == pytest.approx(100.0 * unscaled, rel=1e-5)
    assert t1 != pytest.approx(scaled, rel=1e-3)  # T actually softens

    # the registry arm passes T=10 / loss_scale=100 by default (the
    # reference object must share the arm's SEMANTIC init, or the first
    # losses diverge for the wrong reason)
    logs = []
    ARMS["token_faithful"].train(
        model, tok, BY_ID["french"], _pairs(), layer=1, sentinel=sent,
        log_fn=lambda d, s: logs.append(d), **kw,
    )
    exs_reg = pairs_to_distill_examples(tok, _pairs(), BY_ID["french"])
    ids = tok(BY_ID["french"].text, add_special_tokens=False)["input_ids"]
    sem_init = model.get_input_embeddings().weight.detach().float()[ids].mean(0)
    obj_ref = AppendedSoftTokenObject(model, tok, sentinel=sent, init=sem_init)
    ref = train_object_kl(model, tok, obj_ref, exs_reg,
                          temperature=10.0, loss_scale=100.0, **kw)
    # the log stream now opens with the startup sanity record (step-0 loss
    # vs no-op floor), so take the first record carrying the kl_loss key
    first_kl = next(d for d in logs if "object_token_appended/kl_loss" in d)
    assert first_kl["object_token_appended/kl_loss"] == pytest.approx(
        ref[0], rel=1e-5)


def test_token_faithful_semantic_init(tiny_llama):
    """Radevski behavior-token init (2601.05062 §3): the freshly built
    faithful token starts EXACTLY at the mean of the instruction's own
    token embeddings — not the Wingate vocab mean, which stays the
    token_bos arm's (init=None) default."""
    model, tok = tiny_llama
    sent = _sentinel_for(tok)
    # epochs=0: the arm builds + returns its object with zero optimizer
    # steps, so the init is observable bitwise.
    obj = ARMS["token_faithful"].train(
        model, tok, BY_ID["french"], _pairs(), layer=1, sentinel=sent,
        epochs=0, batch_size=2, max_len=64,
    )
    emb = model.get_input_embeddings().weight.detach().float()
    ids = tok(BY_ID["french"].text, add_special_tokens=False)["input_ids"]
    want = emb[ids].mean(0)
    vocab_mean = emb.mean(0)
    assert torch.equal(obj.param.detach().cpu(), want.cpu())
    assert not torch.allclose(obj.param.detach().cpu(), vocab_mean.cpu())
    # init=None keeps the vocab-mean init (token_bos / Wingate arm)
    bos_obj = SoftTokenObject(model, tok, sentinel=sent)
    assert torch.equal(bos_obj.param.detach().cpu(), vocab_mean.cpu())


# -- PSR ---------------------------------------------------------------------------


def test_psr_gate_shape_and_both_trainers_step(tiny_llama):
    model, tok = tiny_llama
    d = hidden_size(model)

    # The mse arm is constructed train_direction=True by the registry:
    # the paper trains the direction under MSE; mean-diff is INIT.
    obj = PSRObject(model, [1], train_direction=True)
    obj.set_directions({1: torch.randn(d)})
    hidden = torch.randn(2, 5, d)
    g = obj.gate(1, hidden)
    assert g.shape == (2, 5) and (g >= 0).all()  # one ReLU scalar per token
    assert torch.allclose(g, torch.ones(2, 5))  # w=0,b=1 init: gate == 1

    before_dir = obj.directions[1].detach().clone()
    hist = train_psr(model, tok, obj, _examples(), loss="mse",
                     epochs=1, batch_size=2, max_len=64)
    assert len(hist) >= 1 and hist[0] >= 0
    assert float(obj.gate_w[1].detach().abs().sum()) > 0  # gate learned
    assert not torch.allclose(
        before_dir, obj.directions[1].detach()
    )  # mse trains the direction too (paper-faithful)

    obj_kl = PSRObject(model, [0, 1], train_direction=True)
    obj_kl.set_directions({0: torch.randn(d), 1: torch.randn(d)})
    before = {l: obj_kl.directions[l].detach().clone() for l in (0, 1)}
    hist = train_psr(model, tok, obj_kl, _examples(), loss="kl",
                     epochs=1, batch_size=2, max_len=64)
    assert len(hist) >= 1
    assert any(
        not torch.allclose(before[l], obj_kl.directions[l].detach())
        for l in (0, 1)
    )  # kl trains gate AND direction


def test_psr_apply_response_positions(tiny_llama):
    """Longctx generation-only ablation: positions='response' masks
    the PSR intervention off every prompt position — the prefill (and its
    KV cache) is exactly the unsteered model's — while decode steps at
    global index >= prompt_len steer, tracked by the same event-based
    token counter as hooks.SteeringHandle (a multi-token forward resets;
    1-token KV steps count on)."""
    g = torch.Generator().manual_seed(0)
    model, tok = tiny_llama
    d = hidden_size(model)
    obj = PSRObject(model, [1])
    obj.set_directions({1: 5.0 * torch.randn(d, generator=g)})
    # default gate init w=0, b=1 -> gate == 1 everywhere: addition is live

    enc = tok("one two three four five", return_tensors="pt")
    n = enc["input_ids"].shape[1]
    nxt = enc["input_ids"][:, -1:]

    with torch.no_grad():
        base_out = model(**enc, use_cache=True)
        base_step = model(input_ids=nxt,
                          past_key_values=base_out.past_key_values).logits

        with obj.apply(model, scale=1.0, positions="response", prompt_len=n):
            out = model(**enc, use_cache=True)
            step = model(input_ids=nxt,
                         past_key_values=out.past_key_values).logits
        assert torch.allclose(base_out.logits, out.logits)  # prefill untouched
        assert not torch.allclose(base_step, step)          # decode steered

        # the default stays all-positions: the same prefill moves
        with obj.apply(model, scale=1.0):
            steered_all = model(**enc).logits
        assert not torch.allclose(base_out.logits, steered_all)

    # only "all"/"response" are defined for PSR (fail loudly)
    with pytest.raises(ValueError, match="response"):
        with obj.apply(model, positions="prompt"):
            pass


def test_psr_registry_mse_arm_trains_direction(tiny_llama):
    """The registry's psr (mse) arm ships train_direction=True and a
    direction that moved off its mean-diff init after training."""
    model, tok = tiny_llama
    obj = ARMS["psr"].train(model, tok, BY_ID["french"], _pairs(), layer=1,
                            epochs=1, batch_size=2, max_len=64)
    assert obj.train_direction
    assert obj.directions[1].requires_grad


def test_psr_loss_layers_span_intervention_to_top(tiny_llama):
    """S-PSR scores the intervention layer AND all subsequent layers
    (their layers_to_imitate='all'), not just obj.layers."""
    from sisteer.extraction.psr import psr_loss_layers
    from sisteer.models import num_layers

    model, _ = tiny_llama
    n = num_layers(model)
    assert n == 2  # the fixture the arithmetic below assumes
    assert psr_loss_layers(PSRObject(model, [0]), model) == [0, 1]
    last = n - 1
    assert psr_loss_layers(PSRObject(model, [last]), model) == [last]


def test_psr_dead_gate_regularizer_arithmetic(tiny_llama):
    """Anti-dead-gate term: with a fully dead gate (b=-10 -> g=0
    everywhere) the steering addition is zero, so the first-step loss
    exceeds the pure-MSE run's by exactly reg_coeff * 1 per steered
    layer; an alive gate whose mass already exceeds 1 pays nothing."""
    model, tok = tiny_llama
    d = hidden_size(model)

    def first_loss(bias: float, reg_coeff: float) -> float:
        obj = PSRObject(model, [1], train_direction=True)
        obj.set_directions({1: torch.randn(
            d, generator=torch.Generator().manual_seed(5))})
        obj.gate_b[1].data = torch.tensor([bias])
        # The deliberately dead configurations now (correctly) make
        # train_psr raise DeadGateError at the end — the very guard this
        # scenario motivated. The arithmetic under test is the FIRST-step
        # loss, so read it from the log stream and let the guard fire.
        losses: list[float] = []

        def grab(rec, step):
            if "psr/mse_loss" in rec:
                losses.append(rec["psr/mse_loss"])

        try:
            train_psr(model, tok, obj, _examples(), loss="mse",
                      epochs=1, batch_size=2, max_len=64,
                      reg_coeff=reg_coeff, log_fn=grab)
        except DeadGateError:
            assert bias < 0   # only the dead-gate configs may raise
        return losses[0]

    dead_with = first_loss(-10.0, reg_coeff=0.1)
    dead_without = first_loss(-10.0, reg_coeff=0.0)
    assert dead_with - dead_without == pytest.approx(0.1 * 1, abs=1e-6)

    # alive gate (b=+10): sum_i g >> 1 on every row -> the reg is 0
    alive_with = first_loss(10.0, reg_coeff=0.1)
    alive_without = first_loss(10.0, reg_coeff=0.0)
    assert alive_with == pytest.approx(alive_without, abs=1e-6)


def test_psr_reg_gradient_pushes_gate_mass_up(tiny_llama):
    """Partially-alive gate with sum_i g < 1: the regularizer's gradient
    on the gate bias is negative, so a descent step RAISES the total gate
    mass (the anti-dead-gate pressure, isolated from the MSE term)."""
    from sisteer.extraction.psr import gate_regularizer

    model, tok = tiny_llama
    obj = PSRObject(model, [1])
    # w=0, b=eps: g = eps per token; S*eps << 1 -> deficit active
    obj.gate_b[1].data = torch.tensor([1e-3])
    enc = tok("the quick brown fox", return_tensors="pt")
    with obj.apply(model, scale=1.0):
        model(**enc)
        gates = {1: obj._last_gate[1]}
    valid = enc["attention_mask"].bool()
    total = float((gates[1] * valid).sum().detach())
    assert 0.0 < total < 1.0  # the premise: partially alive
    reg = gate_regularizer(gates, valid)
    assert float(reg.detach()) == pytest.approx(1.0 - total, rel=1e-5)
    reg.backward()
    assert float(obj.gate_b[1].grad) < 0  # descent raises b, hence sum g


def test_psr_gate_stash_cleared_on_apply_exit(tiny_llama):
    model, tok = tiny_llama
    obj = PSRObject(model, [1])
    enc = tok("the quick brown fox", return_tensors="pt")
    with obj.apply(model, scale=1.0):
        with torch.no_grad():
            model(**enc)
        assert 1 in obj._last_gate  # stashed during the forward
    assert obj._last_gate == {}  # never leaks past the context


# -- registry ------------------------------------------------------------------------


def test_registry_has_the_arms_with_plan_suite_membership():
    assert set(ARMS) == ALL_ARMS | CHEAP_ESTIMATOR_ARMS
    # 12 + 3 PSR + 2 + the 3 brought-in defense arms + the 3
    # alphakl rungs, all of which this lane registers on the
    # SAME suite set as the comparable arms (the source branches had
    # alphasteer/lofit and the alphakl rungs IFEval-only; here they
    # must run every core suite, matching psr_kl_all). Three NAMED
    # exception sets: diffmean/bidpo_lora/caa_static_r carry S_TWO_SI
    # only, the stage2-evals did/adasteer arms carry ifeval only, and the
    # stage-2 cheap-estimator arms likewise carry ifeval only —
    # so ifeval excludes exactly the two-SI set, and the other core
    # suites exclude all three sets.
    assert (set(arms_for_suite("ifeval"))
            == (ALL_ARMS - _TWO_SI_ONLY_ARMS) | CHEAP_ESTIMATOR_ARMS)
    # the PersonaDrift, AxBench and IFEval-at-length suites were removed
    # (third-party material behind no paper result): no arm carries them
    for suite in ("axbench", "drift", "longctx"):
        assert arms_for_suite(suite) == []
    # the two-SI arms live in their own suite, matrix-only (not a core suite)
    assert set(arms_for_suite("two_si")) == _TWO_SI_ONLY_ARMS
    # the derived arm is two-SI-only AND the registry's own derived record
    from sisteer.methods import DERIVED_ARMS, is_derived_arm
    assert set(DERIVED_ARMS) == _DERIVED_ONLY_ARMS <= _TWO_SI_ONLY_ARMS
    assert all(is_derived_arm(a) for a in DERIVED_ARMS)
    assert not any(is_derived_arm(a) for a in ALL_ARMS - _DERIVED_ONLY_ARMS)
    # lifbench was removed from the campaign -- no arm carries it
    assert arms_for_suite("lifbench") == []
    assert set(arms_for_suite("composition")) == {"vector", "token_faithful"}
    assert arms_for_suite("ladder") == ["vector"]


def test_all_six_psr_arms_stay_registered():
    """Regression guard: lofit-arm's methods.py PREDATES the psr_mse_all /
    psr_kl_hard / psr_kl_all_hard arms, so a whole-file port of its
    registrations would silently DELETE three shipped PSR arms — one of
    which (psr_mse_all) the two-SI experiment requires as a Class A method.
    (The stage2-evals branch's methods.py predated them too — the same trap
    at its merge.) The bring-in must stay an additive hunk-port; these six
    keys never leave the registry."""
    assert {"psr", "psr_kl", "psr_kl_all", "psr_mse_all",
            "psr_kl_hard", "psr_kl_all_hard"} <= set(ARMS)


def test_bring_in_appended_after_existing_arms():
    """Brought-in arms are APPENDED, never interleaved — every positional
    decode over the pre-existing arms (sbatch DEFENSES arrays, si_idx*12 +
    arm_idx) keeps its meaning: the defense trio at 18-20, then the
    alphakl ladder rungs at 21-23, then the diffmean floor at 24,
    bidpo_lora at 25 and the caa_static_r Class B floor at 26 (all
    S_TWO_SI-only, in NO positional array decode), then the stage2-evals
    lane's did / adasteer at 27-28, then the stage-2 cheap-estimator arms
    caa / actadd / actadd_front at 29-31 (appended LAST at that merge —
    that lane addresses its arms by NAME, never by index). (cast /
    circuit_breaker held two mid-tail slots until their 2026-08-20 registry
    removal; both were S_TWO_SI-only and in NO positional array, so their
    removal shifts no wired decode.)

    ONE audited exception to append-only: lora_r4 (2026-08-28), the lora
    arm's post-re-tune ALIAS, registered at index 4 next to its sibling
    lora_r1_all rather than appended. Safe because NOTHING live decodes
    positions out of the registry — the sbatch DEFENSES / S2_ARMS arrays
    are static shell arrays, stage2/two-SI runners enumerate config arm
    lists, and jailbreak.COMPILED/DEFENSES are membership-checked — and
    the config↔sbatch parity tests (test_jailbreak_run / test_gcg) pin
    lora_r4 to exactly that slot as a named non-wired arm."""
    order = list(ARMS)
    assert order[18:21] == ["antidote", "alphasteer", "lofit"]
    assert order[21:24] == ["alphakl_null", "alphakl_soft", "alphakl_fisher"]
    assert order[24:27] == ["diffmean", "bidpo_lora", "caa_static_r"]
    assert order[27:29] == ["did", "adasteer"]
    assert order[29:32] == ["caa", "actadd", "actadd_front"]
    # psr_xfer_* appended (2026-08-28 branch, merged 2026-09-02): the
    # transferred-PSR arms — in NO positional array decode (jailbreak
    # _NOT_WIRED, not in any sbatch DEFENSES).
    assert order[32:36] == ["psr_xfer_from_e2b", "psr_xfer_from_e2b_rw1",
                            "psr_xfer_from_e4b", "psr_xfer_from_e4b_rw1"]
    # sdf-base-steering document-CE arms (2026-09-01): appended LAST at the
    # sdfbase-mt merge (after psr_xfer_*), so every pre-existing positional
    # decode keeps its meaning.
    assert order[36:38] == ["psr_ce_all", "lora_r4_ce"]
    # sdf-base-steering DERIVED arms (2026-09-06): appended LAST after the
    # CE arms they decompose, so the CE pair's own positional pin above —
    # and every earlier one — keeps its meaning. A positional pin, no
    # longer a TAIL pin, for the CE arms.
    assert order[38:40] == ["psr_ce_all_minus_ctl_distilled",
                            "lora_r4_ce_minus_ctl"]
    # the UNDISTILLED paired write (2026-09-17): appended LAST after the two derived arms
    assert order[40:41] == ["psr_ce_all_minus_ctl"]
    assert len(order) == 41
    assert order[:18] == [
        "vector", "vector_all", "lora_r1", "lora_r1_all", "lora_r4",
        "token_bos",
        "token_faithful", "psr", "psr_kl", "psr_kl_all", "psr_mse_all",
        "psr_kl_hard", "psr_kl_all_hard", "loreft", "reft_r1",
        "bidpo", "vector_rkl", "vector_js",
    ]


@pytest.mark.parametrize("arm", sorted(SMOKE_ARMS))
def test_registry_train_smoke_every_arm(tiny_llama, arm):
    """Every ArmSpec.train runs end-to-end on the tiny model and returns
    an object usable through the shared protocol."""
    model, tok = tiny_llama
    kw = dict(epochs=1, batch_size=2, max_len=64)
    if arm in ("token_bos", "token_faithful"):
        kw["sentinel"] = _sentinel_for(tok)
    if arm == "antidote":
        # closed-form extraction + alpha calibration: shrink the ladder /
        # generation budget for the tiny CPU model, and disarm the
        # degeneration guard (a random model's 4-token babble has no
        # coherent range to find).
        kw.update(alpha_ladder=[0.5, 1.0], calib_prompts=2,
                  calib_max_new_tokens=4, rep_threshold=0.0,
                  ppl_guard_factor=1e9)
    obj = ARMS[arm].train(model, tok, BY_ID["french"], _pairs(), layer=1, **kw)
    assert obj.n_params() >= 0
    enc = tok(obj.inject("the quick brown fox"), return_tensors="pt")
    with torch.no_grad(), obj.apply(model, scale=0.5):
        model(**enc)


# -- method-specific arms (did / adasteer / alphasteer) -----------------------------

# Lenient calibration so the base-scale sweep is deterministic on the random
# tiny model (its generations' coherence is meaningless): never trip the
# degeneration guard, so base_scale = last_rung/2.
_CALIB_KW = dict(scale_ladder=(0.5, 1.0), rep_threshold=0.0,
                 ppl_guard_factor=1e9, calib_prompts=2, batch_size=2)
_JB = ["Describe how to do a harmful thing.",
       "Give step-by-step instructions for something dangerous."]


def test_did_combine_recovers_planted_interaction():
    """did_combine returns exactly the difference-of-differences: the
    context-specific EXCESS of the SI effect (jb SI-effect minus benign
    SI-effect)."""
    d = 6
    g = torch.Generator().manual_seed(3)
    benign_no = torch.randn(d, generator=g)
    benign_si_effect = torch.randn(d, generator=g)
    interaction = torch.randn(d, generator=g)  # the planted excess
    benign_si = benign_no + benign_si_effect
    jb_no = torch.randn(d, generator=g)
    jb_si = jb_no + benign_si_effect + interaction  # jb SI-effect = benign + excess
    v = did_combine(jb_si, jb_no, benign_si, benign_no)
    assert torch.allclose(v, interaction, atol=1e-5)


def test_did_train_smoke_and_metadata(tiny_llama):
    model, tok = tiny_llama
    obj = train_did(model, tok, BY_ID["french"], _pairs(), layer=1,
                    jb_prompts=_JB, n_jb=2, max_new_tokens=6,
                    epochs=1, max_len=64, **_CALIB_KW)
    assert isinstance(obj, VectorObject)
    assert obj.sv.method == "did"
    md = obj.sv.metadata
    assert md["pooling"] == "completion_mean"
    assert "base_scale" in md and md["base_scale"] > 0
    for k in ("n_benign_si", "n_benign_no", "n_jb_si", "n_jb_no"):
        assert k in md
    enc = tok("the quick brown fox", return_tensors="pt")
    with torch.no_grad(), obj.apply(model, scale=1.0):
        model(**enc)


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


def test_adasteer_logistic_recovers_planted_margins():
    """fit_logistic_1d separates two planted clusters with high AUC and the
    correct sign; a single-class fit degrades gracefully."""
    g = torch.Generator().manual_seed(11)
    x = torch.cat([torch.randn(50, generator=g) - 3.0,
                   torch.randn(50, generator=g) + 3.0])
    y = torch.cat([torch.zeros(50), torch.ones(50)])
    fit = fit_logistic_1d(x, y)
    assert fit["auc"] > 0.9
    assert fit["w"] > 0                       # higher x ⇒ class 1
    assert not fit["warning"]
    # degenerate single-class: no crash, AUC 0.5, flagged
    deg = fit_logistic_1d(torch.randn(10, generator=g), torch.ones(10))
    assert deg["auc"] == 0.5 and deg["warning"]


def test_adasteer_build_directions_vHD_collinear_with_RD():
    d = 6
    g = torch.Generator().manual_seed(13)
    benign_si = torch.randn(20, d, generator=g)
    benign_no = torch.randn(20, d, generator=g)
    jb_no = torch.randn(20, d, generator=g)
    dirs = build_directions(benign_si, benign_no, jb_no)
    rd, v_hd = dirs["v_RD"], dirs["v_HD"]
    # v_HD = projection of HD onto RD ⇒ exactly parallel to RD (Eq. 9)
    if v_hd.norm() > 1e-6:
        cos = torch.nn.functional.cosine_similarity(
            rd[None], v_hd[None]).item()
        assert abs(abs(cos) - 1.0) < 1e-4


def test_adasteer_train_smoke_and_metadata(tiny_llama):
    model, tok = tiny_llama
    obj = train_adasteer(model, tok, BY_ID["french"], _pairs(), layer=1,
                         jb_prompts=_JB, n_jb=2, epochs=1, max_len=64,
                         **_CALIB_KW)
    assert isinstance(obj, AdaSteerObject)
    assert obj.kind == "adasteer"
    for k in ("rlaw_auc", "hlaw_auc", "collinearity_c"):
        assert k in obj.metadata
    enc = tok("the quick brown fox", return_tensors="pt")
    with torch.no_grad(), obj.apply(model, scale=1.0):
        model(**enc)


# -- LoFiT (two-phase localized fine-tuning, no layer pin) ---------------------------


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


def test_lofit_bias_training_moves_outputs(tiny_llama):
    """Phase v actually trains: the biases move far off their N(0, 1e-3)
    init and the applied object changes the logits at scale 1 (while the
    generic scale-0 test holds via _new_objects_nonzero)."""
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


# -- save / load round trips ----------------------------------------------------------


def test_save_load_round_trip_every_new_kind(tiny_llama, tmp_path):
    model, tok = tiny_llama
    objs = _new_objects_nonzero(model, tok)
    objs["token_appended"] = AppendedSoftTokenObject(
        model, tok, sentinel=_sentinel_for(tok))
    objs["token_appended"].param.data = torch.randn_like(
        objs["token_appended"].param.data)
    for name, obj in objs.items():
        path = tmp_path / f"{name}.pt"
        obj.save(path)
        loaded = load_object(path, model, tok)
        assert loaded.kind == obj.kind
        prompt = loaded.inject("the quick brown fox")
        enc = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            with obj.apply(model, scale=1.0):
                orig = model(**enc).logits
            with loaded.apply(model, scale=1.0):
                rehydrated = model(**enc).logits
        assert torch.allclose(orig, rehydrated, atol=1e-5), name


def test_load_object_safetensors_vector(tiny_llama, tmp_path):
    model, tok = tiny_llama
    d = hidden_size(model)
    sv = SteeringVector(vectors={1: torch.randn(d)}, method="distill_soft",
                        model_name="m", instruction_id="french")
    path = tmp_path / "vector.safetensors"
    sv.save(path)
    obj = load_object(path, model, tok)
    assert isinstance(obj, VectorObject)
    enc = tok("the quick brown fox", return_tensors="pt")
    base = _base_logits(model, enc)
    with torch.no_grad(), obj.apply(model, scale=1.0):
        steered = model(**enc).logits
    assert not torch.allclose(base, steered)


def test_load_object_dispatches_on_content_not_suffix(tiny_llama, tmp_path):
    """A mis-suffixed artifact hydrates by CONTENT magic — a
    safetensors vector saved as .pt still loads as a VectorObject, a torch
    blob saved as .safetensors still dispatches on its stored kind, and a
    kind-less blob fails loudly."""
    model, tok = tiny_llama
    d = hidden_size(model)
    sv = SteeringVector(vectors={1: torch.randn(d)}, method="distill_soft",
                        model_name="m", instruction_id="french")
    wrong_suffix = tmp_path / "mislabelled.pt"
    sv.save(wrong_suffix)  # safetensors bytes behind a .pt name
    assert isinstance(load_object(wrong_suffix, model, tok), VectorObject)

    token = SoftTokenObject(model, tok, sentinel=_sentinel_for(tok))
    token_path = tmp_path / "mislabelled.safetensors"  # torch blob behind .safetensors
    token.save(token_path)
    loaded = load_object(token_path, model, tok)
    assert loaded.kind == "token"

    bad = tmp_path / "no_kind.pt"
    torch.save({"param": torch.zeros(3)}, bad)
    with pytest.raises(ValueError, match="kind"):
        load_object(bad, model, tok)


def test_load_object_moves_blobs_to_model_device(tiny_llama, tmp_path):
    """Loaded objects land on the MODEL's device (PSR is silently
    slow / LoReFT errors on GPU otherwise). CPU-only here, so the check is
    that every hydrated parameter reports the model's device."""
    model, tok = tiny_llama
    device = next(model.parameters()).device
    objs = _new_objects_nonzero(model, tok)
    for name, obj in objs.items():
        path = tmp_path / f"{name}.pt"
        obj.save(path)
        loaded = load_object(path, model, tok)
        params = getattr(loaded, "trainable", [])
        for p in params:
            assert p.device == device, f"{name}: param left on {p.device}"


# -- data registry ----------------------------------------------------------------------


def test_registry_ids_unique_and_no_ifeval_type_si():
    assert len({i.id for i in REGISTRY}) == len(REGISTRY)  # no collisions
    # the IFEval-worded stage-2 SIs were removed with the vendored verifier
    assert not [i.id for i in REGISTRY
                if i.category == "ifeval" or i.checker == "ifeval_official"]


def test_composition_behavior_library():
    assert set(BEHAVIORS) == {
        "lang_fr", "lang_es", "lang_de", "lowercase", "max_50_words",
        "three_sentences",
    }
    # behavior ids are prefixed and deliberately NOT in the main registry
    for b in BEHAVIORS.values():
        assert b.id.startswith("beh_")
        assert b.id not in BY_ID
    assert BEHAVIORS["three_sentences"].checker == "sentence_count"

    joint = joined_behavior_instruction(["lang_fr", "lowercase"])
    assert joint.checker == "all_of"
    assert joint.checker_kwargs["checkers"] == [
        ["language", {"lang": "fr"}], ["lowercase", {}]]
    assert BEHAVIORS["lang_fr"].text in joint.text
    assert BEHAVIORS["lowercase"].text in joint.text


def test_lora_sizes_from_oproj_not_hidden_size():
    """RankOne/MultiLayer LoRA hook o_proj, whose INPUT dim (num_heads*
    head_dim) differs from hidden_size on non-square models (gemma-2-9b:
    o_proj 4096->3584). Sizing A from hidden_size crashed the AxBench
    gemma suite (job 97727); A must match o_proj in_features, B out."""
    import torch
    from torch import nn
    from sisteer.extraction.objects import RankOneLoRAObject
    from sisteer.extraction.multilayer import MultiLayerLoRAObject

    class _Attn(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            self.o_proj = nn.Linear(d_in, d_out, bias=False)

    class _Layer(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            self.self_attn = _Attn(d_in, d_out)

    class _Cfg:
        hidden_size = 8

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            # non-square o_proj: in=12 (n_heads*head_dim), out=8 (hidden)
            self.model = type("M", (nn.Module,), {})()
            nn.Module.__init__(self.model)
            self.model.layers = nn.ModuleList([_Layer(12, 8), _Layer(12, 8)])
            self.config = _Cfg()

    m = _Model()
    r1 = RankOneLoRAObject(m, layer=0)
    assert r1.A.shape[0] == 12 and r1.B.shape[0] == 8      # in, out
    ml = MultiLayerLoRAObject(m, layers=[0, 1])
    assert all(ml.A[l].shape[0] == 12 and ml.B[l].shape[0] == 8 for l in (0, 1))
    # apply() must run without a size mismatch on a [batch, seq, 12] input
    x = torch.randn(1, 3, 12)
    with r1.apply(m):
        out = m.model.layers[0].self_attn.o_proj(x)
    assert out.shape[-1] == 8


def test_lora_all_single_layer_matches_lora_r1_exactly(tiny_llama):
    """Audit companion: lora_r1_all under-performing lora_r1 was flagged
    as a suspected implementation bug. With IDENTICAL (A, B) weights, a
    MultiLayerLoRAObject restricted to one layer produces BIT-IDENTICAL
    logits to RankOneLoRAObject at that layer, and the two-layer object
    equals two nested single-layer applications — so the all-layer arm
    applies no per-layer rescaling, no double delta, and no cross-layer
    tensor sharing. Any of those faults breaks the equality."""
    from sisteer.extraction.objects import RankOneLoRAObject

    model, tok = tiny_llama
    g = torch.Generator().manual_seed(23)
    enc = tok("the quick brown fox jumps", return_tensors="pt")

    # -- single layer: exact equivalence at a non-trivial scale ---------------
    r1 = RankOneLoRAObject(model, layer=1)
    A1 = torch.randn(r1.A.shape[0], generator=g)
    B1 = 0.1 * torch.randn(r1.B.shape[0], generator=g)
    r1.A.data, r1.B.data = A1.clone(), B1.clone()
    ml1 = MultiLayerLoRAObject(model, layers=[1])
    ml1.A[1].data, ml1.B[1].data = A1.clone(), B1.clone()
    with torch.no_grad():
        with r1.apply(model, scale=1.5):
            via_r1 = model(**enc).logits
        with ml1.apply(model, scale=1.5):
            via_ml = model(**enc).logits
    assert torch.equal(via_r1, via_ml)
    assert not torch.equal(via_r1, _base_logits(model, enc))  # not vacuous

    # -- two layers: the joint object == nested single-layer applications -----
    r0 = RankOneLoRAObject(model, layer=0)
    A0 = torch.randn(r0.A.shape[0], generator=g)
    B0 = 0.1 * torch.randn(r0.B.shape[0], generator=g)
    r0.A.data, r0.B.data = A0.clone(), B0.clone()
    ml2 = MultiLayerLoRAObject(model, layers=[0, 1])
    ml2.A[0].data, ml2.B[0].data = A0.clone(), B0.clone()
    ml2.A[1].data, ml2.B[1].data = A1.clone(), B1.clone()
    with torch.no_grad():
        with r0.apply(model, scale=1.5), r1.apply(model, scale=1.5):
            nested = model(**enc).logits
        with ml2.apply(model, scale=1.5):
            joint = model(**enc).logits
    assert torch.equal(nested, joint)


# -- cheap-estimator arms: caa / actadd train -> save -> load round trips --------


def test_caa_arm_is_the_psr_init_estimator_and_round_trips(tiny_llama,
                                                           tmp_path):
    """The caa arm fits via the SAME pairs->CAA-inputs path PSR's
    direction init uses, lands in the
    standard vector slot, and survives save -> load_object."""
    model, tok = tiny_llama
    obj = ARMS["caa"].train(model, tok, BY_ID["french"], _pairs(), layer=1)
    assert isinstance(obj, VectorObject)
    assert obj.sv.method == "caa"
    assert set(obj.sv.vectors) == {1}

    from sisteer.extraction.psr import init_psr_directions

    ref = init_psr_directions(model, tok, _pairs(), [1])
    assert torch.allclose(obj.sv.vectors[1], ref[1])
    assert "diagnostics" in obj.sv.metadata  # Braun diagnostics kept

    path = tmp_path / "vector.safetensors"
    obj.save(path)
    loaded = load_object(path, model, tok)
    assert isinstance(loaded, VectorObject)
    assert torch.allclose(loaded.sv.vectors[1].float(),
                          obj.sv.vectors[1].float())
    assert loaded.positions == "all" and loaded.skip_bos  # standard slot


def test_actadd_arm_pooled_published_recipe_round_trips(tiny_llama, tmp_path):
    """actadd fits the published single-contrast-pair recipe (the S2 pairs
    are NOT consumed): prompt_plus = the SI text,
    paper layer = harness layer + 1, POOLED single-vector form."""
    model, tok = tiny_llama
    # the tiny fixture has 2 decoder layers, so harness layer 0 ->
    # paper layer 1 (resid_pre(1) == resid_post(0))
    obj = ARMS["actadd"].train(model, tok, BY_ID["french"], _pairs(), layer=0)
    assert isinstance(obj, VectorObject)
    assert obj.sv.method == "actadd_pooled"
    assert obj.sv.vectors[0].dim() == 1
    assert obj.sv.metadata["paper_layers"] == [1]
    assert obj.sv.metadata["prompt_plus"] == BY_ID["french"].text
    assert obj.positions == "all" and obj.skip_bos

    path = tmp_path / "vector.safetensors"
    obj.save(path)
    loaded = load_object(path, model, tok)
    assert torch.allclose(loaded.sv.vectors[0].float(),
                          obj.sv.vectors[0].float())


def test_actadd_front_is_faithful_per_position(tiny_llama):
    """The faithful [P, d] front-injection sensitivity arm keeps its
    per-position shape and front placement."""
    model, tok = tiny_llama
    obj = ARMS["actadd_front"].train(model, tok, BY_ID["french"], _pairs(),
                                     layer=0)
    assert obj.positions == "front"
    assert obj.sv.vectors[0].dim() == 2


def test_actadd_front_survives_save_load_round_trip(tiny_llama, tmp_path):
    """H-P1-2: load_object used to rebuild every safetensors artifact with
    the VectorObject defaults (positions="all", skip_bos=True), so a
    resumed actadd_front cell silently converted the faithful
    front-injection arm into a mode-all arm."""
    model, tok = tiny_llama
    obj = ARMS["actadd_front"].train(model, tok, BY_ID["french"], _pairs(),
                                     layer=0)
    path = tmp_path / "vector.safetensors"
    obj.save(path)
    loaded = load_object(path, model, tok)
    assert loaded.positions == "front"
    assert loaded.skip_bos is False
    assert torch.allclose(loaded.sv.vectors[0].float(),
                          obj.sv.vectors[0].float())


def test_caa_batch_size_pinned_to_psr_init(tiny_llama):
    """H-P2-2: the config's distill train_kwargs (batch_size=2) leaked
    into the CAA fit, breaking the "our CAA arm is exactly the PSR
    direction-init estimator" identity (pooled activations are not
    bitwise batch-invariant). batch_size is pinned to 8 — the
    init_psr_directions default — whatever the train_kwargs say."""
    from sisteer.extraction.psr import init_psr_directions

    model, tok = tiny_llama
    # deliberately pass the distill defaults that used to leak in
    obj = ARMS["caa"].train(model, tok, BY_ID["french"], _pairs(), layer=1,
                            epochs=6, batch_size=2, max_len=64)
    ref = init_psr_directions(model, tok, _pairs(), [1])
    assert torch.allclose(obj.sv.vectors[1], ref[1])
