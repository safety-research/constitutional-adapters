"""``PSRPairedWriteObject`` (kind ``psr_paired_write``) on the CPU tiny
model: loader round trip through ``methods.load_object``, parents bit-
exact, ``trainable == []``, ONE combined hook per layer whose two gates
read the IDENTICAL pre-write state, scale linearity at the hooked layer
and the bitwise scale-0 no-op, compose/load refusals (wrong parent kind,
layer sets, fit legs, a foreign application order), and the paired gate
liveness telemetry.
"""

from __future__ import annotations

import types

import pytest
import torch

from residual_fixtures import random_psr
from sisteer.extraction.distill import DistillExample
from sisteer.extraction.paired_write import (APPLICATION_ORDER,
                                             PSRPairedWriteObject,
                                             paired_gate_liveness)
from sisteer.extraction.psr import PSRObject
from sisteer.models import get_decoder_layers


def _psr(model, seed: int, b: float = 0.5) -> PSRObject:
    # an all-layer random gate: unnormalised 0.05 write, 0.3-scale gate
    return random_psr(model, seed, write=0.05, normalize=False,
                      gate_scale=0.3, b=b)


@pytest.fixture(scope="module")
def world(tiny_llama, tmp_path_factory):
    from sisteer.methods import load_object

    model, tok = tiny_llama
    root = tmp_path_factory.mktemp("pairedwrite")
    mix, ctl = _psr(model, 11), _psr(model, 22)
    mix.save(root / "mix.pt")
    ctl.save(root / "ctl.pt")
    blob = PSRPairedWriteObject.compose_blob(
        torch.load(root / "mix.pt", map_location="cpu", weights_only=True),
        torch.load(root / "ctl.pt", map_location="cpu", weights_only=True),
        provenance={"parent_pos": {"cell_id": "constv2-mix-100k__psr_ce_all"},
                    "parent_neg": {"cell_id": "constv2ctl-100k__psr_ce_all"}})
    torch.save(blob, root / "v.pt")
    V = load_object(root / "v.pt", model=model, tok=tok)
    return types.SimpleNamespace(model=model, tok=tok, mix=mix, ctl=ctl, V=V,
                                 root=root, blob=blob)


def _layer_out(model, tok, layer, ctx, capture_before: bool = False):
    """Layer ``layer``'s output under ctx; ``capture_before`` registers
    the capture BEFORE the object's hooks so it sees the PRE-write state."""
    from contextlib import nullcontext

    captured = {}

    def cap(_m, _i, out):
        captured["x"] = (out[0] if isinstance(out, tuple) else out
                         ).detach().float().clone()

    enc = tok("a paired write reads one shared state", return_tensors="pt")
    mod = get_decoder_layers(model)[layer]
    with torch.no_grad():
        if capture_before:
            h = mod.register_forward_hook(cap)
            with (ctx if ctx is not None else nullcontext()):
                model(**enc)
            h.remove()
        else:
            with (ctx if ctx is not None else nullcontext()):
                h = mod.register_forward_hook(cap)
                try:
                    model(**enc)
                finally:
                    h.remove()
    return captured["x"]


def test_round_trip_kind_and_shape(world):
    V = world.V
    assert isinstance(V, PSRPairedWriteObject)
    assert V.kind == "psr_paired_write"
    assert V.layers == world.mix.layers
    assert V.trainable == [] and V.n_params() == 0
    assert V.application_order == APPLICATION_ORDER
    assert world.blob["application_order"] == APPLICATION_ORDER
    assert V.provenance["parent_pos"]["cell_id"] == "constv2-mix-100k__psr_ce_all"
    assert V.inject("prompt") == "prompt"


def test_parents_ride_bit_exact(world):
    for l in world.mix.layers:
        for name in ("directions", "gate_w", "gate_b"):
            assert torch.equal(getattr(world.V.pos, name)[l],
                               getattr(world.mix, name)[l])
            assert torch.equal(getattr(world.V.neg, name)[l],
                               getattr(world.ctl, name)[l])


def test_one_combined_hook_per_layer_and_identical_gate_inputs(world):
    m, tok, V = world.model, world.tok, world.V
    layers = get_decoder_layers(m)
    n_before = [len(layers[l]._forward_hooks) for l in V.layers]
    with V.apply(m):
        for l, n0 in zip(V.layers, n_before):
            assert len(layers[l]._forward_hooks) == n0 + 1  # ONE hook
        enc = tok("a paired write reads one shared state", return_tensors="pt")
        with torch.no_grad():
            m(**enc)
        # both gates' pre-activations were evaluated on the SAME z: the
        # pre-write layer-0 output, captured by a hook registered before
        pre_pos, pre_neg = V._last_pre[0][0], V._last_pre[1][0]
    z = _layer_out(m, tok, 0, V.apply(m), capture_before=True)
    assert torch.allclose(pre_pos, world.mix.gate_pre(0, z), atol=1e-5)
    assert torch.allclose(pre_neg, world.ctl.gate_pre(0, z), atol=1e-5)
    assert not V._last_pre  # stash cleared at exit
    for l, n0 in zip(V.layers, n_before):
        assert len(layers[l]._forward_hooks) == n0  # hooks removed


def test_hook_algebra_is_pos_minus_neg_on_the_shared_state(world):
    m, tok = world.model, world.tok
    base = _layer_out(m, tok, 0, None)
    d_mix = _layer_out(m, tok, 0, world.mix.apply(m)) - base
    d_ctl = _layer_out(m, tok, 0, world.ctl.apply(m)) - base
    d_v = _layer_out(m, tok, 0, world.V.apply(m)) - base
    # At layer 0 every gate reads the bare layer-0 state, so the paired
    # write equals the difference of the parents' single writes exactly.
    assert torch.allclose(d_v, d_mix - d_ctl, atol=1e-5)
    assert d_v.abs().max() > 0


def test_scale_linear_at_hook_and_scale_zero_bitwise(world):
    m, tok = world.model, world.tok
    base = _layer_out(m, tok, 0, None)
    d1 = _layer_out(m, tok, 0, world.V.apply(m, scale=1.0)) - base
    d2 = _layer_out(m, tok, 0, world.V.apply(m, scale=2.0)) - base
    assert torch.allclose(d2, 2 * d1, atol=1e-5)
    z0 = _layer_out(m, tok, 0, world.V.apply(m, scale=0.0))
    assert torch.equal(z0, base)


def test_save_reload_is_bit_exact(world, tmp_path):
    from sisteer.methods import load_object

    world.V.save(tmp_path / "again.pt")
    back = load_object(tmp_path / "again.pt", model=world.model, tok=world.tok)
    for l in world.V.layers:
        assert torch.equal(back.pos.directions[l], world.mix.directions[l])
        assert torch.equal(back.neg.gate_w[l], world.ctl.gate_w[l])
    assert back.provenance == world.V.provenance


def test_compose_and_load_refusals(world):
    mix_blob = torch.load(world.root / "mix.pt", map_location="cpu",
                          weights_only=True)
    ctl_blob = torch.load(world.root / "ctl.pt", map_location="cpu",
                          weights_only=True)
    with pytest.raises(ValueError, match="kind"):
        PSRPairedWriteObject.compose_blob({**mix_blob, "kind": "lora_all"},
                                          ctl_blob)
    fewer = dict(ctl_blob)
    keep = sorted(ctl_blob["directions"])[:1]
    for tag in ("directions", "gate_w", "gate_b"):
        fewer[tag] = {k: ctl_blob[tag][k] for k in keep}
    fewer["layers"] = [int(k) for k in keep]
    with pytest.raises(ValueError, match="layer sets differ"):
        PSRPairedWriteObject.compose_blob(mix_blob, fewer)
    with pytest.raises(ValueError, match="positions"):
        PSRPairedWriteObject.compose_blob(
            mix_blob, {**ctl_blob, "positions": "response",
                       "response_header_len": 3})
    bad_order = dict(world.blob)
    bad_order["application_order"] = "pos_then_neg"
    with pytest.raises(ValueError, match="application_order"):
        PSRPairedWriteObject.load_blob(bad_order, world.model, world.tok)


def test_paired_gate_liveness_reports_each_term(world):
    exs = [DistillExample(prompt_with_si="", prompt_without="",
                          completion=f"liveness probe text number {i}")
           for i in range(3)]
    live = paired_gate_liveness(world.model, world.tok, obj=world.V,
                                examples=exs, max_len=64)
    assert set(live["terms"]) == {0, 1}
    for i in (0, 1):
        for l in world.V.layers:
            assert 0.0 <= live["terms"][i][l] <= 1.0
    for l in world.V.layers:
        assert live["any"][l] >= max(live["terms"][0][l], live["terms"][1][l])
    assert live["n_valid_positions"] > 0


# -- the hook contract on the PERSISTED-AND-RELOADED V ----------------------


def _cap_batches(tok):
    from sisteer.extraction.residual import CaptureDoc, make_cap_batches

    docs = [CaptureDoc(doc_id="m0", source="mix", family="advice",
                       prompt="Sys one\n\n",
                       text="a paired write reads one shared state"),
            CaptureDoc(doc_id="m1", source="mix", family="stories", prompt="",
                       text="a longer story body with more tokens in it than "
                            "the first one has"),
            CaptureDoc(doc_id="c0", source="ctl", family="stories", prompt="",
                       text="control text of some length here"),
            CaptureDoc(doc_id="c1", source="ctl", family="advice",
                       prompt="Ctl sys\n\n", text="short one")]
    return make_cap_batches(tok, docs, batch_size=2, max_len=48, device="cpu",
                            label="t")


def test_reloaded_V_satisfies_the_hook_contract_on_every_layer(world):
    """The hook contract is exercised on the PERSISTED-AND-RELOADED
    ``psr_paired_write`` object (compose_blob → torch.save →
    methods.load_object — the derived app's path), not only on an
    in-process composite. On the fp32 tiny model the contract is BITWISE
    (max |err| == 0) at every layer, the write is live on a real fraction
    of positions, and as the REFERENCE of the identity check against the
    in-process +mix − ctl combination the two writes agree bit for bit
    (same terms, same order) while both contracts hold."""
    from sisteer.extraction.paired_write import SharedStateWriteObject
    from sisteer.extraction.residual import (hook_contract_problems,
                                             teacher_identity_problems)

    batches = _cap_batches(world.tok)
    layers = list(world.V.layers)
    problems, rep = hook_contract_problems(world.model, batches, obj=world.V,
                                           layers=layers, label="V")
    assert problems == []
    assert set(rep) == set(layers)
    for l in layers:
        assert rep[l]["pass"] and rep[l]["n_violations"] == 0
        assert rep[l]["max_abs_err"] == 0.0 and rep[l]["max_err_half_ulps"] == 0.0
        assert rep[l]["n_elements"] > 0 and rep[l]["stream_dtype"] == "float32"
        assert rep[l]["tolerance"].startswith("per element: half an ulp")
    assert max(rep[l]["write_active_frac"] for l in layers) > 0.1
    composite = SharedStateWriteObject([(1.0, world.mix), (-1.0, world.ctl)])
    problems, rep = teacher_identity_problems(
        world.model, batches, composite=composite, reference=world.V,
        layers=layers)
    assert problems == []
    for l in layers:
        assert rep[l]["rms_write_discrepancy"] == 0.0
        assert rep[l]["hook_contract_reference"]["pass"]
        assert rep[l]["hook_contract_reference"]["max_abs_err"] == 0.0
        assert rep[l]["hook_contract_composite"]["pass"]


@pytest.mark.parametrize("how", ["noop", "double", "scale", "wrong_layer", "bump"])
def test_reloaded_V_with_a_defective_runtime_fails_the_contract(world, how):
    """The same reloaded V wrapped so that ``paired_write`` stays exactly
    right while ``apply()`` no-ops / doubles / mis-scales / mis-places its
    write, or edits one element by one ulp: every variant fails the hook
    contract — the standalone check and the identity check's reference
    contract alike — while the write-identity half keeps passing (it never
    saw apply()); the unwrapped V next to it passes."""
    from residual_fixtures import ApplyPerturbed
    from sisteer.extraction.paired_write import SharedStateWriteObject
    from sisteer.extraction.residual import (hook_contract_problems,
                                             teacher_identity_problems)

    batches = _cap_batches(world.tok)
    layers = list(world.V.layers)
    bad = ApplyPerturbed(world.V, how)
    problems, rep = hook_contract_problems(world.model, batches, obj=bad,
                                           layers=layers, label="V")
    assert problems and all("V HOOK CONTRACT" in p for p in problems), how
    flagged = {l for l in layers if not rep[l]["pass"]}
    assert flagged == ({layers[0]} if how == "bump" else set(layers)), (how, flagged)
    composite = SharedStateWriteObject([(1.0, world.mix), (-1.0, world.ctl)])
    p2, r2 = teacher_identity_problems(world.model, batches,
                                       composite=composite, reference=bad,
                                       layers=layers)
    assert p2 and all("reference HOOK CONTRACT" in p for p in p2)
    assert all(r2[l]["write_identity_pass"] for l in layers)
    assert all(r2[l]["rms_write_discrepancy"] == 0.0 for l in layers)
    assert {l for l in layers if not r2[l]["pass"]} == flagged
