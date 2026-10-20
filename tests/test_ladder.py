"""On-policyness ladder: rung-2 MC-KL data and the rung-3 GKD trainer.

CPU-only. Generation is stubbed (monkeypatched generate_batch in the
ladder module) so the tests are deterministic and assert the trainer's
CONTRACT: rung 2 builds N x prompts sampled-teacher examples; rung 3
regenerates student rollouts every epoch, under the current vector.
"""

import torch

from sisteer.data.instructions import BY_ID
from sisteer.data.pairs import pairs_to_distill_examples
from sisteer.extraction import ladder

PROMPTS = ["Say something nice.", "Describe a tree."]


def test_sample_teacher_pairs_builds_n_times_prompts(tiny_llama, monkeypatch):
    model, tok = tiny_llama
    calls = []

    def fake_generate(model_, tok_, rendered, **kw):
        calls.append(kw)
        return [f"réponse {i}" for i in range(len(rendered))]

    monkeypatch.setattr(ladder, "generate_batch", fake_generate)
    pairs = ladder.sample_teacher_pairs(
        model, tok, BY_ID["french"], PROMPTS, n_samples=3, temperature=1.0,
    )
    assert len(pairs) == 3 * len(PROMPTS)  # N x prompts examples
    assert all(p.instruction_id == "french" for p in pairs)
    assert all(p.pos and p.neg == "" for p in pairs)
    # one sampled generation pass per MC sample, at the MC-KL settings
    assert len(calls) == 3
    assert all(c["do_sample"] is True and c["temperature"] == 1.0
               for c in calls)


def test_rung2_pairs_feed_the_existing_soft_distill_path(tiny_llama,
                                                         monkeypatch):
    model, tok = tiny_llama
    monkeypatch.setattr(
        ladder, "generate_batch",
        lambda m, t, rendered, **kw: ["Bonjour le monde"] * len(rendered),
    )
    pairs = ladder.sample_teacher_pairs(
        model, tok, BY_ID["french"], PROMPTS, n_samples=2,
    )
    exs = pairs_to_distill_examples(tok, pairs, BY_ID["french"])
    assert len(exs) == len(pairs)  # rung 2 trains on the positives only
    assert all(ex.completion == "Bonjour le monde" for ex in exs)
    assert all(BY_ID["french"].text in ex.prompt_with_si for ex in exs)
    assert all(BY_ID["french"].text not in ex.prompt_without for ex in exs)


def test_gkd_regenerates_rollouts_each_epoch_under_current_vector(
        tiny_llama, monkeypatch):
    model, tok = tiny_llama
    calls = []

    def fake_generate(model_, tok_, rendered, **kw):
        # snapshot the vector VALUES at call time (detach() aliases the
        # live parameter, so a late read would see the trained values)
        calls.append({
            "kw": kw,
            "sv_snapshot": {k: v.clone() for k, v in kw["sv"].items()},
        })
        return ["Bonjour le monde"] * len(rendered)

    monkeypatch.setattr(ladder, "generate_batch", fake_generate)
    sv = ladder.train_gkd(
        model, tok, BY_ID["french"], PROMPTS, layer=1,
        epochs=2, batch_size=2, max_len=64, lr=1e-2,
    )

    # fresh rollouts EVERY epoch — the rung-3 property
    assert len(calls) == 2
    for c in calls:
        assert c["kw"]["do_sample"] is True
        assert c["kw"]["temperature"] == 1.0
        assert c["kw"]["scale"] == 1.0
        # the rollout runs under the SAME policy as train/eval —
        # skip_bos must be passed through (defaulting False silently made
        # the student roll out under a BOS-steered policy)
        assert c["kw"]["skip_bos"] is True
        assert set(c["sv_snapshot"]) == {1}  # student steered at the layer
    # epoch 1 rolls out under the zero-init vector; epoch 2 under the
    # UPDATED vector (same weights, new state distribution)
    assert float(calls[0]["sv_snapshot"][1].abs().sum()) == 0.0
    assert float(calls[1]["sv_snapshot"][1].abs().sum()) > 0.0

    assert sv.method == "gkd"
    assert sv.metadata["rung"] == 3
    assert sv.layers == [1]
    assert float(sv.vectors[1].abs().sum()) > 0.0  # training moved the vector


def test_rung2_resamples_per_epoch_and_budget_matches_rung3(tiny_llama,
                                                            monkeypatch):
    """Rungs 2 and 3 must differ ONLY in trajectory source. Under
    one config (same epochs, n_samples, batch_size) rung 2 draws a FRESH
    teacher sample set each epoch (not one set reused), and both rungs see
    identical trajectory counts and optimizer-step counts — recorded in
    metadata and logged as budget/*."""
    model, tok = tiny_llama
    calls = {"teacher": 0, "student": 0}

    def fake_generate(model_, tok_, rendered, **kw):
        if "sv" in kw:
            calls["student"] += 1  # gkd rollout (steered student)
        else:
            calls["teacher"] += 1  # mc_kl teacher sampling (SI in context)
        return [f"bonjour le monde {i}" for i in range(len(rendered))]

    monkeypatch.setattr(ladder, "generate_batch", fake_generate)
    common = dict(epochs=2, n_samples=2, batch_size=2, max_len=64, lr=1e-2)
    logged2, logged3 = [], []
    sv2 = ladder.train_mc_kl(
        model, tok, BY_ID["french"], PROMPTS, layer=1, temperature=1.0,
        log_fn=lambda d, step: logged2.append(d), **common,
    )
    sv3 = ladder.train_gkd(
        model, tok, BY_ID["french"], PROMPTS, layer=1,
        log_fn=lambda d, step: logged3.append(d), **common,
    )

    # rung 2 REGENERATES per epoch: one sampled pass per (epoch, sample) —
    # a set sampled once and reused would show n_samples calls, not
    # epochs * n_samples
    assert calls["teacher"] == 2 * 2
    assert calls["student"] == 2 * 2

    m2, m3 = sv2.metadata, sv3.metadata
    expected_traj = 2 * 2 * len(PROMPTS)  # epochs * n_samples * prompts
    assert m2["n_trajectories_seen"] == m3["n_trajectories_seen"] == expected_traj
    assert m2["n_optimizer_steps"] == m3["n_optimizer_steps"] > 0
    assert sv2.method == "mc_kl" and m2["rung"] == 2
    assert sv3.method == "gkd" and m3["rung"] == 3

    # the budget match is verifiable in W&B: budget/* logged per rung
    b2 = next(d for d in logged2 if "budget/mc_kl/trajectories_seen" in d)
    b3 = next(d for d in logged3 if "budget/gkd/trajectories_seen" in d)
    assert (b2["budget/mc_kl/trajectories_seen"]
            == b3["budget/gkd/trajectories_seen"] == expected_traj)
    assert (b2["budget/mc_kl/optimizer_steps"]
            == b3["budget/gkd/optimizer_steps"]
            == m2["n_optimizer_steps"])


def test_mc_kl_draws_fresh_samples_each_epoch(tiny_llama, monkeypatch):
    """The per-epoch sets are genuinely fresh draws (distinct seeds), not
    one cached set: the trainer must request generation with a different
    seed context each epoch, visible as distinct completion batches."""
    model, tok = tiny_llama
    seen_batches = []

    def fake_generate(model_, tok_, rendered, **kw):
        # label each generation pass uniquely so reuse would be visible
        out = [f"réponse {len(seen_batches)}-{i}" for i in range(len(rendered))]
        seen_batches.append(out)
        return out

    monkeypatch.setattr(ladder, "generate_batch", fake_generate)
    ladder.train_mc_kl(
        model, tok, BY_ID["french"], PROMPTS, layer=1,
        epochs=3, n_samples=1, batch_size=2, max_len=64, lr=1e-2,
    )
    assert len(seen_batches) == 3  # one fresh draw per epoch
    assert len({tuple(b) for b in seen_batches}) == 3  # all distinct
