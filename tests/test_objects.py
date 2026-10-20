"""Object-axis ablation: soft-token and rank-1-LoRA interventions."""

import torch

from sisteer.extraction.distill import DistillExample
from sisteer.extraction.objects import (
    RankOneLoRAObject,
    SoftTokenObject,
    train_object_kl,
)


def _sentinel_for(tok):
    # tiny-llama has no reserved tokens; unk is a single-token stand-in
    return (tok.unk_token, tok.unk_token_id)


def test_soft_token_hook_replaces_only_sentinel(tiny_llama):
    model, tok = tiny_llama
    obj = SoftTokenObject(model, tok, sentinel=_sentinel_for(tok))
    ids = tok("hello world", return_tensors="pt")["input_ids"]
    ids_with = torch.cat(
        [ids[:, :1], torch.tensor([[obj.sentinel_id]]), ids[:, 1:]], dim=1
    )
    emb = model.get_input_embeddings()
    with obj.apply(model, scale=1.0):
        out = emb(ids_with)
    assert torch.allclose(out[0, 1], obj.param.to(out.dtype))
    assert torch.allclose(out[0, 0], emb.weight[ids_with[0, 0]])
    # without the context the sentinel keeps its ordinary embedding
    out_plain = emb(ids_with)
    assert not torch.allclose(out_plain[0, 1], obj.param.to(out_plain.dtype))


def test_soft_token_scale_zero_passes_sentinel_through(tiny_llama):
    """Scale 0 == unsteered: the sentinel keeps its NATIVE
    embedding rather than being zeroed, matching every residual arm's
    bitwise identity at scale 0."""
    model, tok = tiny_llama
    obj = SoftTokenObject(model, tok, sentinel=_sentinel_for(tok))
    ids = torch.tensor([[obj.sentinel_id]])
    emb = model.get_input_embeddings()
    native = emb(ids).detach().clone()
    with obj.apply(model, scale=0.0):
        out = emb(ids)
    assert torch.equal(out, native)
    # and scale 1 does replace it (the scale-0 check is not vacuous)
    with obj.apply(model, scale=1.0):
        steered = emb(ids)
    assert torch.allclose(steered[0, 0], obj.param.to(steered.dtype))


def test_lora_zero_init_is_identity_and_scale_zero_noop(tiny_llama):
    model, tok = tiny_llama
    obj = RankOneLoRAObject(model, layer=1)
    enc = tok("the quick brown fox", return_tensors="pt")
    with torch.no_grad():
        base = model(**enc).logits
        with obj.apply(model, scale=4.0):
            zeroed = model(**enc).logits  # B == 0 -> identical
        obj.B.data = torch.randn_like(obj.B.data)
        with obj.apply(model, scale=1.0):
            steered = model(**enc).logits
        with obj.apply(model, scale=0.0):
            noop = model(**enc).logits
    assert torch.allclose(base, zeroed)
    assert not torch.allclose(base, steered)
    assert torch.allclose(base, noop)


def _examples(tok, obj_kind):
    exs = []
    for prompt in ["Say something nice.", "Describe a tree."]:
        exs.append(
            DistillExample(
                prompt_with_si=f"Always answer in French.\n\n{prompt}\n",
                prompt_without=f"{prompt}\n",
                completion="Bonjour le monde",
            )
        )
    return exs


def test_train_object_kl_updates_both_objects(tiny_llama):
    model, tok = tiny_llama
    for make in (
        lambda: SoftTokenObject(model, tok, sentinel=_sentinel_for(tok)),
        lambda: RankOneLoRAObject(model, layer=1),
    ):
        obj = make()
        before = [p.detach().clone() for p in obj.trainable]
        history = train_object_kl(
            model, tok, obj, _examples(tok, obj.kind),
            epochs=1, batch_size=2, max_len=64,
        )
        assert len(history) >= 1 and history[0] >= 0
        moved = any(
            not torch.allclose(b, p.detach().cpu().to(b.dtype), atol=0)
            for b, p in zip(before, [q.cpu() for q in obj.trainable])
        )
        assert moved, f"{obj.kind}: no parameter update"
