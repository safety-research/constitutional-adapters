"""Composition operators + the learned <and> trainer (extraction/compose.py).

CPU-only, tiny-llama fixture. Covers: additive-sum compose scale-0
identity and sum-of-hooks equality (order-invariant, one scored
ordering); the [E_x, e_b1, e_<and>, e_b2] token layout and its zero-shot
k=3 extension; hook inertness without an injected block and native-
embedding passthrough at scale 0; <and> training reducing loss with the LLM
and behavior tokens frozen (bitwise); the orthogonality regularizer; and
the exact compose_fn(objs, order) contract eval/composition.py calls.
"""

import pytest
import torch
from torch import nn

from sisteer.data.pairs import Pair
from sisteer.extraction.compose import (
    ComposedTokenObject,
    compose_additive,
    compose_tokens,
    orthogonality_penalty,
    token_compose_fn,
    train_and_token,
)
from sisteer.extraction.tokens import AppendedSoftTokenObject
from sisteer.methods import VectorObject, load_object
from sisteer.models import hidden_size
from sisteer.vectors import SteeringVector

FR_LOWER = "bonjour mes amis, je suis très heureux de vous voir aujourd'hui."


def _sentinel_for(tok):
    return (tok.unk_token, tok.unk_token_id)


def _vector_obj(model, layer, seed, instruction_id="beh", **kw):
    d = hidden_size(model)
    g = torch.Generator().manual_seed(seed)
    sv = SteeringVector(
        vectors={layer: 0.05 * torch.randn(d, generator=g)},
        method="distill_soft", model_name="tiny",
        instruction_id=instruction_id,
    )
    return VectorObject(sv, **kw)


def _token_obj(model, tok, seed):
    obj = AppendedSoftTokenObject(model, tok, sentinel=_sentinel_for(tok))
    g = torch.Generator().manual_seed(seed)
    obj.param.data = 0.05 * torch.randn(obj.param.shape, generator=g)
    return obj


def _and_param(model, seed):
    g = torch.Generator().manual_seed(seed)
    return nn.Parameter(0.05 * torch.randn(hidden_size(model), generator=g))


def _logits(model, enc):
    with torch.no_grad():
        return model(**enc).logits


# -- mechanism 1: residual-vector additive sum -----------------------------------


class TestAdditive:
    def test_scale_zero_is_unsteered(self, tiny_llama):
        model, tok = tiny_llama
        enc = tok("the quick brown fox jumps", return_tensors="pt")
        base = _logits(model, enc)
        composed = compose_additive(
            [_vector_obj(model, 1, 0, "a"), _vector_obj(model, 0, 1, "b")],
            (0, 1),
        )
        with torch.no_grad():
            with composed.apply(model, scale=0.0):
                zeroed = model(**enc).logits
            with composed.apply(model, scale=1.0):
                steered = model(**enc).logits
        assert torch.equal(base, zeroed)
        assert not torch.allclose(base, steered), (
            "scale-1 changed nothing — the scale-0 check is vacuous"
        )

    def test_sum_equals_stacked_hooks(self, tiny_llama):
        """Composed logits == logits under the individual hooks applied
        together — for same-layer (vectors sum) and cross-layer parts."""
        model, tok = tiny_llama
        enc = tok("steering vectors add", return_tensors="pt")
        o1 = _vector_obj(model, 1, 2, "a")
        o2 = _vector_obj(model, 1, 3, "b")  # same layer as o1
        o3 = _vector_obj(model, 0, 4, "c")  # different layer
        composed = compose_additive([o1, o2, o3])
        assert composed.sv.method == "composed_additive"
        assert composed.sv.metadata["components"] == ["a", "b", "c"]
        with torch.no_grad():
            with composed.apply(model, scale=1.0):
                got = model(**enc).logits
            with o1.apply(model, scale=1.0), o2.apply(model, scale=1.0), \
                    o3.apply(model, scale=1.0):
                want = model(**enc).logits
        assert torch.allclose(got, want, atol=1e-4, rtol=1e-4)

    def test_order_invariant(self, tiny_llama):
        model, tok = tiny_llama
        assert compose_additive.order_invariant is True
        enc = tok("order should not matter", return_tensors="pt")
        o1, o2 = _vector_obj(model, 1, 5, "a"), _vector_obj(model, 0, 6, "b")
        with torch.no_grad():
            with compose_additive([o1, o2], (0, 1)).apply(model):
                ab = model(**enc).logits
            with compose_additive([o2, o1], (1, 0)).apply(model):
                ba = model(**enc).logits
        assert torch.allclose(ab, ba)

    def test_rejects_norm_relative_mismatch_and_singletons(self, tiny_llama):
        model, _ = tiny_llama
        o1 = _vector_obj(model, 1, 7, "a")
        with pytest.raises(ValueError, match="norm_relative"):
            compose_additive([o1, _vector_obj(model, 0, 8, "b",
                                              norm_relative=True)])
        with pytest.raises(ValueError, match="apply settings"):
            compose_additive([o1, _vector_obj(model, 0, 8, "b",
                                              skip_bos=False)])
        with pytest.raises(ValueError, match=">= 2"):
            compose_additive([o1])
        with pytest.raises(TypeError, match="VectorObject"):
            compose_additive([o1, object()])


# -- mechanism 2: token_faithful <and> composition --------------------------------


class TestComposedToken:
    def test_layout_pair_and_zero_shot_triple(self, tiny_llama):
        """[E_x, e_b1, e_<and>, e_b2] for k=2; the same <and> between each
        adjacent pair for the zero-shot k=3 layout."""
        model, tok = tiny_llama
        b1, b2, b3 = (_token_obj(model, tok, s) for s in (10, 11, 12))
        and_param = _and_param(model, 13)
        cases = [
            ([b1, b2], [b1.param, and_param, b2.param]),
            ([b1, b2, b3],
             [b1.param, and_param, b2.param, and_param, b3.param]),
        ]
        emb = model.get_input_embeddings()
        for objs, want in cases:
            composed = compose_tokens(objs, and_param)
            L = len(want)
            rendered = "The sky is blue"
            injected = composed.inject(rendered)
            assert injected == rendered + tok.unk_token * L
            enc = tok(injected, return_tensors="pt")
            assert (enc["input_ids"][0, -L:].tolist()
                    == [tok.unk_token_id] * L)
            captured = {}
            with composed.apply(model, scale=1.0):
                h = emb.register_forward_hook(
                    lambda _m, _i, out: captured.__setitem__(
                        "e", out.detach().clone())
                )
                try:
                    with torch.no_grad():
                        model(**enc)
                finally:
                    h.remove()
            got = captured["e"][0, -L:]
            for pos, (row, param) in enumerate(zip(got, want)):
                assert torch.allclose(
                    row, param.detach().to(row.dtype)
                ), f"slot {pos} of k={len(objs)} has the wrong embedding"

    def test_scale_zero_and_inert_without_block(self, tiny_llama):
        model, tok = tiny_llama
        b1, b2 = _token_obj(model, tok, 20), _token_obj(model, tok, 21)
        composed = compose_tokens([b1, b2], _and_param(model, 22))

        # no injected block in the prompt -> the hook is bitwise inert
        enc = tok("no sentinels here at all", return_tensors="pt")
        base = _logits(model, enc)
        with torch.no_grad(), composed.apply(model, scale=1.0):
            inert = model(**enc).logits
        assert torch.equal(base, inert)

        # injected prompt at scale 0: the hook passes the block's NATIVE
        # embeddings through unchanged (scale-0 == unsteered) —
        # bitwise identical to running the injected ids with no hook at all.
        enc_inj = tok(composed.inject("The sky is blue"), return_tensors="pt")
        base_inj = _logits(model, enc_inj)
        with torch.no_grad(), composed.apply(model, scale=0.0):
            zero_logits = model(**enc_inj).logits
        assert torch.equal(zero_logits, base_inj)
        with torch.no_grad(), composed.apply(model, scale=1.0):
            one_logits = model(**enc_inj).logits
        assert not torch.allclose(zero_logits, one_logits), (
            "scale-1 changed nothing — the scale-0 check is vacuous"
        )

    def test_order_changes_logits_and_compose_fn_flag(self, tiny_llama):
        model, tok = tiny_llama
        b1, b2 = _token_obj(model, tok, 30), _token_obj(model, tok, 31)
        fn = token_compose_fn(_and_param(model, 32))
        assert fn.order_invariant is False
        c12, c21 = fn([b1, b2], (0, 1)), fn([b2, b1], (1, 0))
        assert isinstance(c12, ComposedTokenObject)
        injected = c12.inject("The sky is blue")
        assert c21.inject("The sky is blue") == injected  # same block text
        enc = tok(injected, return_tensors="pt")
        with torch.no_grad():
            with c12.apply(model):
                l12 = model(**enc).logits
            with c21.apply(model):
                l21 = model(**enc).logits
        assert not torch.allclose(l12, l21)

    def test_rejects_bad_inputs(self, tiny_llama):
        model, tok = tiny_llama
        b1 = _token_obj(model, tok, 33)
        and_param = _and_param(model, 34)
        with pytest.raises(ValueError, match=">= 2"):
            compose_tokens([b1], and_param)
        with pytest.raises(TypeError, match="soft-token"):
            compose_tokens([b1, object()], and_param)

    def test_save_load_blob_round_trip(self, tiny_llama, tmp_path):
        model, tok = tiny_llama
        b1, b2 = _token_obj(model, tok, 35), _token_obj(model, tok, 36)
        composed = compose_tokens([b1, b2], _and_param(model, 37))
        path = tmp_path / "composed.pt"
        composed.save(path)
        blob = torch.load(path, weights_only=True)
        assert blob["kind"] == "token_composed"
        again = ComposedTokenObject.load_blob(blob, model, tok)
        assert again.sentinel_ids == composed.sentinel_ids
        assert again.sentinel_texts == composed.sentinel_texts
        for a, b in zip(again.embeds, composed.embeds):
            assert torch.equal(a.detach(), b.detach().float().cpu())
        # and the generic loader fails LOUDLY on it, never mis-hydrates
        with pytest.raises(ValueError, match="token_composed"):
            load_object(path, model, tok)


# -- the orthogonality regularizer -------------------------------------------------


class TestOrthogonality:
    def test_penalty_values(self):
        v = torch.zeros(8)
        v[0] = 2.0
        w = torch.zeros(8)
        w[1] = 3.0
        assert float(orthogonality_penalty(v, [v.clone()])) == pytest.approx(1.0)
        assert float(orthogonality_penalty(v, [-v])) == pytest.approx(1.0)
        assert float(orthogonality_penalty(v, [w])) == pytest.approx(0.0)
        assert float(orthogonality_penalty(v, [v.clone(), w])) == pytest.approx(1.0)

    def test_gradient_step_reduces_alignment(self):
        g = torch.Generator().manual_seed(0)
        b = torch.randn(16, generator=g)
        perp = torch.randn(16, generator=g)
        perp = perp - (perp @ b) / (b @ b) * b
        p = nn.Parameter((b + 0.3 * perp).clone())
        before = float(orthogonality_penalty(p, [b]).detach())
        orthogonality_penalty(p, [b]).backward()
        assert p.grad is not None and float(p.grad.norm()) > 0
        with torch.no_grad():
            p -= p.grad
        assert float(orthogonality_penalty(p, [b]).detach()) < before


# -- the <and> trainer --------------------------------------------------------------


def _teacher_pairs():
    return [
        Pair(prompt="Say something nice.", pos="bonjour le monde", neg="",
             instruction_id="j"),
        Pair(prompt="Describe a tree.", pos="c'est un grand arbre", neg="",
             instruction_id="j"),
    ]


class TestTrainAndToken:
    def test_training_reduces_loss_and_freezes_everything_else(
        self, tiny_llama
    ):
        model, tok = tiny_llama
        behavior_tokens = {
            "lang_fr": _token_obj(model, tok, 40),
            "lowercase": _token_obj(model, tok, 41),
            "max_50_words": _token_obj(model, tok, 42),
        }
        before = {
            n: o.param.detach().clone() for n, o in behavior_tokens.items()
        }
        w_before = next(model.parameters()).detach().clone()
        train_pairs = [("lang_fr", "lowercase"), ("lang_fr", "max_50_words")]
        logged = []
        and_param = train_and_token(
            model, tok, behavior_tokens, train_pairs,
            pairs_by_target={t: _teacher_pairs() for t in train_pairs},
            ortho_lambda=0.5, lr=2e-2, epochs=8, batch_size=4,
            log_fn=lambda d, step: logged.append(d),
        )
        assert isinstance(and_param, nn.Parameter)
        assert and_param.shape == (hidden_size(model),)
        assert logged and all("and_token/kl" in d and "and_token/ortho" in d
                              for d in logged)
        last_epoch = max(d["epoch"] for d in logged)
        first = [d["and_token/loss"] for d in logged if d["epoch"] == 0]
        last = [d["and_token/loss"] for d in logged
                if d["epoch"] == last_epoch]
        assert sum(last) / len(last) < sum(first) / len(first), (
            "<and> training did not reduce the loss"
        )
        # frozen per-behavior tokens: bitwise identical before/after
        for n, o in behavior_tokens.items():
            assert torch.equal(o.param.detach(), before[n]), (
                f"behavior token {n} moved during <and> training"
            )
        # frozen LLM
        assert torch.equal(next(model.parameters()).detach(), w_before)
        # the <and> actually moved off its pinned zero init (Radevski
        # Table 4: init is load-bearing for unseen-composition accuracy)
        assert not torch.allclose(and_param.detach().cpu(),
                                  torch.zeros(hidden_size(model)))

    def test_and_param_zero_init_bitwise_at_epochs0(self, tiny_llama):
        """The paper pins e_<and> to ZERO init (2601.05062 Table 4). With
        zero optimizer steps the returned parameter is bitwise zeros —
        not the repo's Wingate mean-embedding init."""
        model, tok = tiny_llama
        behavior_tokens = {
            "lang_fr": _token_obj(model, tok, 46),
            "lowercase": _token_obj(model, tok, 47),
        }
        train_pairs = [("lang_fr", "lowercase")]
        and_param = train_and_token(
            model, tok, behavior_tokens, train_pairs,
            pairs_by_target={t: _teacher_pairs() for t in train_pairs},
            epochs=0,
        )
        assert torch.equal(and_param.detach(),
                           torch.zeros(hidden_size(model)))

    def test_orthogonality_scope_excludes_unseen_behaviors(self, tiny_llama):
        """L_orth ranges over B_seen only (paper §3): a behavior token
        that appears in NO training pair provably cannot affect the
        trained <and> — two identical-seed runs differing only in the
        held-out behavior's embedding are bitwise-equal."""
        model, tok = tiny_llama

        def run(heldout_seed: int):
            behavior_tokens = {
                "lang_fr": _token_obj(model, tok, 70),
                "lowercase": _token_obj(model, tok, 71),
                # in the token map, in no train pair -> outside B_seen
                "max_50_words": _token_obj(model, tok, heldout_seed),
            }
            train_pairs = [("lang_fr", "lowercase")]
            return train_and_token(
                model, tok, behavior_tokens, train_pairs,
                pairs_by_target={t: _teacher_pairs() for t in train_pairs},
                ortho_lambda=0.5, lr=2e-2, epochs=2, batch_size=4, seed=0,
            ), behavior_tokens["max_50_words"].param.detach().clone()

        a, held_a = run(72)
        b, held_b = run(73)
        assert not torch.equal(held_a, held_b)  # the held-out DID differ
        assert torch.equal(a.detach(), b.detach())

    def test_rejects_non_pair_targets_and_missing_data(self, tiny_llama):
        model, tok = tiny_llama
        behavior_tokens = {
            "lang_fr": _token_obj(model, tok, 43),
            "lowercase": _token_obj(model, tok, 44),
            "max_50_words": _token_obj(model, tok, 45),
        }
        with pytest.raises(ValueError, match="2-behavior"):
            train_and_token(
                model, tok, behavior_tokens,
                [("lang_fr", "lowercase", "max_50_words")],
                pairs_by_target={},
            )
        with pytest.raises(ValueError, match="no behavior token"):
            train_and_token(
                model, tok, behavior_tokens, [("lang_fr", "lang_es")],
                pairs_by_target={},
            )
        with pytest.raises(ValueError, match="base_prompts or pairs_by_target"):
            train_and_token(model, tok, behavior_tokens,
                            [("lang_fr", "lowercase")])


# -- the compose_fn contract eval/composition.py calls ------------------------------


class TestEvalContract:
    def test_compose_fn_signature_matches_eval(self, tiny_llama, tmp_path):
        """eval_composition calls compose_fn(objs_in_order, order): the
        token operator gets k! orderings (k=2 and zero-shot k=3), the
        additive operator is scored once as order_invariant."""
        from sisteer.eval.composition import eval_composition

        model, tok = tiny_llama
        targets = {
            "seen": [("lang_fr", "lowercase")],
            "triples": [("lang_fr", "lowercase", "max_50_words")],
        }

        token_objs = {
            "lang_fr": _token_obj(model, tok, 50),
            "lowercase": _token_obj(model, tok, 51),
            "max_50_words": _token_obj(model, tok, 52),
        }
        inner = token_compose_fn(_and_param(model, 53))
        built = []

        def fn(objs, order):
            obj = inner(objs, order)
            built.append(obj)
            return obj

        fn.order_invariant = inner.order_invariant

        records = eval_composition(
            None, tok,
            targets=targets,
            prompts=["Say hi"],
            out_dir=tmp_path / "tok",
            per_behavior_objs=token_objs,
            compose_fn=fn,
            arms=("composed",),
            generate_fn=lambda rendered, obj: [FR_LOWER] * len(rendered),
        )
        by_k = {r["k"]: r for r in records}
        assert by_k[2]["n_orderings_scored"] == 2
        assert by_k[3]["n_orderings_scored"] == 6
        assert all(r["order_invariant"] is False for r in records)
        assert all(isinstance(b, ComposedTokenObject) for b in built)
        # pair layout has 3 slots, zero-shot triple has 5
        assert sorted({len(b.sentinel_ids) for b in built}) == [3, 5]

        vec_objs = {
            "lang_fr": _vector_obj(model, 1, 60, "lang_fr"),
            "lowercase": _vector_obj(model, 1, 61, "lowercase"),
            "max_50_words": _vector_obj(model, 0, 62, "max_50_words"),
        }
        records2 = eval_composition(
            None, tok,
            targets=targets,
            prompts=["Say hi"],
            out_dir=tmp_path / "vec",
            per_behavior_objs=vec_objs,
            compose_fn=compose_additive,
            arms=("composed",),
            generate_fn=lambda rendered, obj: [FR_LOWER] * len(rendered),
        )
        assert [r["n_orderings_scored"] for r in records2] == [1, 1]
        assert all(r["order_invariant"] is True for r in records2)
