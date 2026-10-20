"""train_object_ce — the solo document-CE trainer (sdf-base-steering §5.1).

CPU-only, tiny model, synthetic doc-shaped examples. Pins the four
reviewed contract items (token-share accumulation vs one concatenated
batch; micro-averaged held-out CE + the once-pre-epoch-1 bare-base
denominator; forward+loss+backward inside apply() under gradient
checkpointing; checkpoints at completed optimizer-group boundaries with
bitwise resume) plus the trainer's refusals (over-length rows, NaN,
gate_reg on gateless objects, positions != all) and the empty-prompt/BOS
edge under BOTH tokenizer shapes (BOS-adding and BOS-less).
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F

from sisteer.data.pairs import Pair, docs_to_ce_examples
from sisteer.extraction.distill import (DistillExample, EarlyStopSpec,
                                        _encode_aligned,
                                        _gather_completion_logits)
from sisteer.extraction.fitmon import TrainingDiverged
from sisteer.extraction.multilayer import MultiLayerLoRAObject
from sisteer.extraction.objects import train_object_ce
from sisteer.extraction.psr import PSRObject
from sisteer.extraction.traindata import collate

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _docs(n=4):
    """Doc-shaped pairs: a loss-masked prefix in the prompt slot (empty for
    the story-like rows), trained text in pos — deliberately UNEQUAL
    lengths so token-share vs flat weighting differ."""
    rows = [
        ("Sys scaffold\n\n", "A short doc."),
        ("", "A much longer story body that runs on for quite a few more "
             "tokens than the short one does, on purpose."),
        ("Another sys\n\n", "Medium length advice body follows here."),
        ("", "Tiny."),
        ("Third sys\n\n", "One more medium advice body for good measure."),
        ("", "Final story text, middling length overall."),
    ]
    return [Pair(prompt=p, pos=pos, neg="", instruction_id="constv2-mix-1k")
            for p, pos in rows[:n]]


def _examples(tok, n=4):
    return docs_to_ce_examples(tok, _docs(n))


def _fresh(kind: str, model):
    """A fresh CE-arm object, deterministic construction."""
    if kind == "psr":
        return PSRObject(model, [0, 1], train_direction=True)
    return MultiLayerLoRAObject(model, rank=4)


def _manual_micro_avg_ce(model, tok, examples, obj=None, max_len=256):
    """INDEPENDENT reference: sum(token NLL)/sum(target tokens) over the
    pool, one example at a time (so batching/bucketing cannot leak in),
    completion tokenized without specials + ONE EOS appended — a trainer
    that double-appends EOS, drops it, or batch-mean-averages diverges."""
    tot_nll, tot_tok = 0.0, 0
    for ex in examples:
        p_ids = tok(ex.prompt_without, add_special_tokens=True)["input_ids"]
        c_ids = tok(ex.completion, add_special_tokens=False)["input_ids"]
        c_ids = c_ids + [tok.eos_token_id]
        ids = torch.tensor([p_ids + c_ids])
        with torch.no_grad():
            ctx = obj.apply(model, scale=1.0) if obj is not None \
                else nullcontext()
            with ctx:
                logits = model(input_ids=ids,
                               attention_mask=torch.ones_like(ids)).logits
        # targets: positions max(len(p),1) .. end — position 0 is never a
        # target (the labels[:, 1:] shift), the BOS-less empty-prompt edge.
        start = max(len(p_ids), 1)
        lp = torch.log_softmax(logits[0, :-1].float(), -1)
        for t in range(start, ids.shape[1]):
            tot_nll += -float(lp[t - 1, ids[0, t]])
            tot_tok += 1
    return tot_nll / tot_tok


# -- contract (i): token-share accumulation == one concatenated batch ---------


@pytest.mark.parametrize("kind", ["psr", "lora"])
def test_accumulation_equals_one_concatenated_batch(tiny_llama, kind):
    """Ragged, unequal-length micro-batches under token-share weighting
    train EXACTLY like one concatenated effective batch — for both CE-arm
    object classes. batch_size=1 + accum=N vs batch_size=N + accum=1 over
    one epoch = one optimizer step over the same rows either way."""
    model, tok = tiny_llama
    exs = _examples(tok, 4)
    kw = dict(lr=1e-3, epochs=1, max_len=256, seed=0,
              lr_schedule="constant", max_grad_norm=math.inf)

    def run(bs, accum):
        obj = _fresh(kind, model)
        train_object_ce(model, tok, obj, exs, batch_size=bs,
                        accum_steps=accum,
                        reg_coeff=(0.1 if kind == "psr" else 0.0), **kw)
        return [p.detach().clone() for p in obj.trainable]

    concat = run(4, 1)
    accum = run(1, 4)
    assert len(concat) == len(accum) > 0
    moved = 0.0
    for a, b in zip(concat, accum):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5), (
            f"{kind}: accumulated != concatenated (max abs diff "
            f"{float((a - b).abs().max()):.3e})")
        moved += float(a.abs().sum())
    assert moved > 0.0  # something actually trained


def test_flat_one_over_accum_weighting_is_wrong(tiny_llama):
    """The sensitivity half of the invariance pin: on UNEQUAL-length
    micro-batches, gradients under the token-share rule match one
    concatenated batch while a flat 1/accum weighting does NOT — so an
    implementation that regressed to flat weighting fails the test above
    for a reason this test makes visible."""
    model, tok = tiny_llama
    exs = _examples(tok, 2)  # one short, one long — very unequal
    obj = _fresh("lora", model)
    enc = [_encode_aligned(tok, ex, 256)[1] for ex in exs]
    n_tok = []
    batches = []
    for e in enc:
        b = collate([e], tok.pad_token_id)
        batches.append(b)
        n_tok.append(int((b.labels[:, 1:] != -100).sum()))
    assert n_tok[0] != n_tok[1], "fixture must be unequal-length"

    def grads(weights):
        for p in obj.trainable:
            p.grad = None
        for b, w in zip(batches, weights):
            with obj.apply(model, scale=1.0):
                logits = model(input_ids=b.input_ids,
                               attention_mask=b.attention_mask).logits
                labels = b.labels[:, 1:]
                ce = F.cross_entropy(
                    _gather_completion_logits(logits, b).float(),
                    labels[labels != -100])
                (w * ce).backward()
        return [p.grad.detach().clone() for p in obj.trainable
                if p.grad is not None]

    total = sum(n_tok)
    g_share = grads([n / total for n in n_tok])
    g_flat = grads([0.5, 0.5])
    # concatenated reference
    for p in obj.trainable:
        p.grad = None
    big = collate(enc, tok.pad_token_id)
    with obj.apply(model, scale=1.0):
        logits = model(input_ids=big.input_ids,
                       attention_mask=big.attention_mask).logits
        labels = big.labels[:, 1:]
        F.cross_entropy(_gather_completion_logits(logits, big).float(),
                        labels[labels != -100]).backward()
    g_concat = [p.grad.detach().clone() for p in obj.trainable
                if p.grad is not None]
    share_ok = all(torch.allclose(a, b, atol=1e-6, rtol=1e-5)
                   for a, b in zip(g_share, g_concat))
    flat_ok = all(torch.allclose(a, b, atol=1e-6, rtol=1e-5)
                  for a, b in zip(g_flat, g_concat))
    assert share_ok, "token-share gradients must equal the concatenated batch"
    assert not flat_ok, ("flat 1/accum gradients must DIFFER on ragged "
                         "token counts — if this passes, the fixture lost "
                         "its length asymmetry")


def test_epoch_boundary_flush_and_realized_eff_batch(tiny_llama):
    """Accumulation groups never span epochs: 3 batches at accum_steps=2
    give 2 optimizer steps per epoch (the ragged final group is its own
    step), and the realized effective batch says what actually ran."""
    model, tok = tiny_llama
    exs = _examples(tok, 3)
    obj = _fresh("lora", model)
    diag: dict = {}
    hist = train_object_ce(model, tok, obj, exs, lr=1e-3, epochs=2,
                           batch_size=1, accum_steps=2, max_len=256,
                           diagnostics=diag)
    assert diag["steps_per_epoch"] == 2
    assert diag["optimizer_steps_run"] == 4
    assert len(hist) == 4
    assert sorted(diag["realized_eff_batch"]) == [1, 2]  # 2-row + ragged 1


# -- contract (ii): held-out micro-averaging + early stopping ----------------


def test_heldout_ce_is_micro_averaged_and_eos_counted_once(tiny_llama):
    """heldout_noop_ce equals the independent sum(NLL)/sum(tokens)
    reference on a crafted unequal-length pool — and DIFFERS from the
    mean-of-batch-means on the same pool (batch means up-weight short
    docs). The reference appends exactly ONE EOS per completion, so a
    double/missing EOS also fails here."""
    model, tok = tiny_llama
    fit = _examples(tok, 2)
    held = _examples(tok, 6)[2:]  # 4 rows, unequal lengths
    obj = _fresh("lora", model)
    diag: dict = {}
    train_object_ce(model, tok, obj, fit, lr=1e-4, epochs=1, batch_size=2,
                    max_len=256, heldout=held, diagnostics=diag)
    manual = _manual_micro_avg_ce(model, tok, held, obj=None)
    assert diag["heldout_noop_ce"] == pytest.approx(manual, rel=1e-5)
    # mean of batch means (batch_size=2 buckets, the trainer's own
    # encoding) is a DIFFERENT number on unequal-length rows
    enc = sorted((_encode_aligned(tok, ex, 256)[1] for ex in held),
                 key=lambda e: len(e.ids))
    batch_means = []
    for i in range(0, len(enc), 2):
        b = collate(enc[i:i + 2], tok.pad_token_id)
        with torch.no_grad():
            logits = model(input_ids=b.input_ids,
                           attention_mask=b.attention_mask).logits
        labels = b.labels[:, 1:]
        batch_means.append(float(F.cross_entropy(
            _gather_completion_logits(logits, b).float(),
            labels[labels != -100])))
    mobm = sum(batch_means) / len(batch_means)
    assert abs(mobm - manual) > 1e-6, (
        "fixture failed to separate micro-average from mean-of-batch-means")


def test_early_stop_triggers_and_best_epoch_restores(tiny_llama):
    """A destructive lr makes held-out CE worsen after epoch 1; the stopper
    stops on patience and the returned object's held-out CE equals the
    CURVE MINIMUM (best-epoch restore), not the last epoch's."""
    model, tok = tiny_llama
    exs = _examples(tok, 4)
    held = _examples(tok, 6)[4:]
    obj = _fresh("lora", model)
    diag: dict = {}
    train_object_ce(model, tok, obj, exs, lr=5.0, epochs=8, batch_size=2,
                    max_len=256, heldout=held,
                    early_stop=EarlyStopSpec(patience=2, min_epochs=1),
                    diagnostics=diag)
    curve = diag["heldout_curve"]
    per_epoch = curve[1:]  # curve[0] = pre-training baseline point
    assert diag["stopped_early"] is True
    assert diag["epochs_run"] < 8
    assert diag["best_epoch"] == per_epoch.index(min(per_epoch)) + 1
    assert diag["best_heldout_ce"] == pytest.approx(min(per_epoch))
    restored = _manual_micro_avg_ce(model, tok, held, obj=obj)
    assert restored == pytest.approx(min(per_epoch), rel=1e-5)


def test_early_stop_without_heldout_refused(tiny_llama):
    model, tok = tiny_llama
    with pytest.raises(ValueError, match="early_stop without heldout"):
        train_object_ce(model, tok, _fresh("lora", model),
                        _examples(tok, 2), lr=1e-3, epochs=1, batch_size=2,
                        max_len=256, early_stop=EarlyStopSpec())


# -- loss identity, startup floor, schedule math ------------------------------


def test_step_loss_is_exactly_completion_ce_and_zero_init_floor(tiny_llama):
    """The logged step-0 startup pair: a zero-init object's as-initialized
    CE equals the no-op floor EXACTLY (batch-0, same batch) — the loss
    reads exactly the completion labels, nothing else. And with accum=1 the
    first optimizer step's logged loss equals a manual CE over the first
    bucket's completion positions."""
    model, tok = tiny_llama
    exs = _examples(tok, 2)
    obj = _fresh("lora", model)  # B zero-init => exact no-op at step 0
    logs: list = []

    def log_fn(rec, step=None, **_kw):
        logs.append((step, dict(rec)))

    train_object_ce(model, tok, obj, exs, lr=1e-4, epochs=1, batch_size=2,
                    max_len=256, log_fn=log_fn, lr_schedule="constant")
    step0 = next(r for s, r in logs if s == 0 and
                 "object_ce/step0_loss" in r)
    assert step0["object_ce/step0_loss"] == pytest.approx(
        step0["object_ce/noop_floor"])
    # manual CE over the single (sorted) batch == the first step's loss
    enc = sorted((_encode_aligned(tok, ex, 256)[1] for ex in exs),
                 key=lambda e: len(e.ids))
    b = collate(enc, tok.pad_token_id)
    with torch.no_grad():
        logits = model(input_ids=b.input_ids,
                       attention_mask=b.attention_mask).logits
    labels = b.labels[:, 1:]
    manual = float(F.cross_entropy(
        _gather_completion_logits(logits, b).float(),
        labels[labels != -100]))
    # the fit mutated obj AFTER computing step 1's loss on the pre-step
    # weights, but obj was an exact no-op at that moment => manual on the
    # UNSTEERED model equals the logged first step loss.
    first = next(r for s, r in logs if s == 1)
    assert first["object_ce/ce_loss"] == pytest.approx(manual, rel=1e-6)


def test_warmup_cosine_matches_driver_step_arithmetic(tiny_llama):
    """The lr trace follows warmup/cosine over total = epochs x
    ceil(ceil(n/bs)/accum) optimizer steps with warmup_steps resolved by
    the DRIVER's own policy (lockstep.resolve_warmup_steps) — the exact
    arithmetic train_two_si_objects uses to resolve warmup_frac, so
    trainer and driver can never disagree on the schedule position."""
    from sisteer.extraction.lockstep import resolve_warmup_steps

    model, tok = tiny_llama
    exs = _examples(tok, 3)
    epochs, bs, accum, lr = 2, 1, 2, 1e-3
    total = epochs * math.ceil(math.ceil(len(exs) / bs) / accum)  # 4
    wu = resolve_warmup_steps(0.25, total)  # 1
    logs: list = []
    train_object_ce(model, tok, _fresh("lora", model), exs, lr=lr,
                    epochs=epochs, batch_size=bs, accum_steps=accum,
                    max_len=256, lr_schedule="warmup_cosine",
                    warmup_steps=wu,
                    log_fn=lambda r, step=None, **_k: logs.append((step, r)))
    lrs = [r["object_ce/lr"] for s, r in logs if "object_ce/lr" in r]

    def expect(i):
        if i < wu:
            return lr * (i + 1) / wu
        frac = (i - wu) / (total - wu)
        return lr * 0.5 * (1.0 + math.cos(math.pi * frac))

    assert len(lrs) == total
    for i, got in enumerate(lrs):
        assert got == pytest.approx(expect(i), rel=1e-9), f"step {i}"


# -- refusals ------------------------------------------------------------------


def test_overlength_row_refused_never_silently_truncated(tiny_llama):
    """A row whose prompt+completion+EOS exceeds max_len REFUSES loudly —
    _encode_aligned alone would silently truncate it (comp[:avail]),
    quietly shrinking the cell's realized budget."""
    model, tok = tiny_llama
    long_doc = Pair(prompt="Sys\n\n", pos="word " * 300, neg="",
                    instruction_id="constv2-mix-1k")
    exs = docs_to_ce_examples(tok, [long_doc])
    # sanity: _encode_aligned itself would NOT raise (silent truncation)
    _encode_aligned(tok, exs[0], 64)
    with pytest.raises(ValueError, match="max_len"):
        train_object_ce(model, tok, _fresh("lora", model), exs, lr=1e-3,
                        epochs=1, batch_size=1, max_len=64)


def test_empty_completion_refused(tiny_llama):
    model, tok = tiny_llama
    exs = docs_to_ce_examples(tok, [Pair(prompt="Sys\n\n", pos="", neg="",
                                         instruction_id="constv2-mix-1k")])
    with pytest.raises(ValueError, match="EMPTY completion"):
        train_object_ce(model, tok, _fresh("lora", model), exs, lr=1e-3,
                        epochs=1, batch_size=1, max_len=64)


def test_gate_reg_requires_gate_bearing_object(tiny_llama):
    model, tok = tiny_llama
    with pytest.raises(ValueError, match="gate_reg_term"):
        train_object_ce(model, tok, _fresh("lora", model),
                        _examples(tok, 2), lr=1e-3, epochs=1, batch_size=2,
                        max_len=256, reg_coeff=0.1)


def test_positions_response_refused(tiny_llama):
    model, tok = tiny_llama
    with pytest.raises(ValueError, match="all-position"):
        train_object_ce(model, tok, _fresh("psr", model),
                        _examples(tok, 2), lr=1e-3, epochs=1, batch_size=2,
                        max_len=256, positions="response",
                        response_header_len=4)


def test_nan_circuit_breaker(tiny_llama):
    """A non-finite training loss raises TrainingDiverged — never train
    through NaN. Forced via an object whose steering hook injects a
    param-dependent NaN."""
    model, tok = tiny_llama

    class NaNObject:
        kind = "nan_stub"

        def __init__(self, model):
            self.p = torch.nn.Parameter(torch.ones(1))
            self._layer = model.model.layers[0]

        trainable = property(lambda self: [self.p])

        def inject(self, s):
            return s

        def apply(self, model, scale=1.0):
            from contextlib import contextmanager

            @contextmanager
            def _ctx():
                def hook(_m, _i, out):
                    h = out[0] if isinstance(out, tuple) else out
                    h = h + self.p * float("nan")
                    return (h,) + out[1:] if isinstance(out, tuple) else h
                hnd = self._layer.register_forward_hook(hook)
                try:
                    yield
                finally:
                    hnd.remove()
            return _ctx()

    with pytest.raises(TrainingDiverged):
        train_object_ce(model, tok, NaNObject(model), _examples(tok, 2),
                        lr=1e-3, epochs=1, batch_size=2, max_len=256)


# -- contract (iii): backward inside apply() under gradient checkpointing -----


@pytest.mark.parametrize("kind", ["psr", "lora"])
def test_backward_inside_apply_under_gradient_checkpointing(tiny_llama, kind):
    """Non-reentrant gradient checkpointing RECOMPUTES the forward during
    backward; the hooks must be live at that moment or torch refuses
    (CheckpointError: different tensor set) / silently diverges. The fit
    must complete AND move parameters. (Full grad parity vs the
    uncheckpointed path is pinned in tests/test_ckpt_grad_parity.py.)"""
    import torch as _t

    from sisteer.models import load_model_and_tokenizer

    model, tok = load_model_and_tokenizer(TINY, device="cpu",
                                          dtype=_t.float32)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    obj = _fresh(kind, model)
    hist = train_object_ce(model, tok, obj, _examples(tok, 3), lr=1e-3,
                           epochs=1, batch_size=2, max_len=256,
                           reg_coeff=(0.1 if kind == "psr" else 0.0))
    assert all(math.isfinite(h) for h in hist)
    assert sum(float(p.abs().sum()) for p in obj.trainable) > 0.0


# -- contract (iv): checkpoint save/resume --------------------------------------


def test_checkpoints_only_at_completed_group_boundaries(tiny_llama, tmp_path,
                                                        monkeypatch):
    """Every checkpoint's batches_done is a completed accumulation-group
    boundary — never mid-group (3 batches at accum 2: boundaries are 2 and
    3; a mid-group write would show 1)."""
    import sisteer.extraction.objects as objects_mod

    seen: list[int] = []
    real = objects_mod.save_checkpoint

    def spy(d, state):
        seen.append(int(state["batches_done"]))
        return real(d, state)

    monkeypatch.setattr(objects_mod, "save_checkpoint", spy)
    model, tok = tiny_llama
    train_object_ce(model, tok, _fresh("lora", model), _examples(tok, 3),
                    lr=1e-3, epochs=2, batch_size=1, accum_steps=2,
                    max_len=256, checkpoint_every=1,
                    checkpoint_dir=tmp_path / "ck", checkpoint_key="fp-test")
    assert seen, "checkpoint_every=1 must write checkpoints"
    assert set(seen) <= {2, 3}, (
        f"checkpoint at a non-boundary micro-batch position: {seen}")


def test_checkpoint_resume_is_bitwise(tiny_llama, tmp_path):
    """Kill the fit mid-way (an exception from log_fn), resume from the
    checkpoint with an identical call: the final trainables are BITWISE
    the uninterrupted run's (CPU fp32), stopper state and held-out curve
    included."""
    model, tok = tiny_llama
    exs = _examples(tok, 4)
    held = _examples(tok, 6)[4:]
    kw = dict(lr=1e-3, epochs=3, batch_size=1, accum_steps=2, max_len=256,
              heldout=held, early_stop=EarlyStopSpec(patience=3),
              checkpoint_every=1)

    def run(ckdir, interrupt_at=None):
        obj = _fresh("lora", model)
        diag: dict = {}
        calls = {"n": 0}

        def log_fn(rec, step=None, **_k):
            if (interrupt_at is not None and step == interrupt_at
                    and any(k.endswith("/ce_loss") for k in rec)):
                calls["n"] += 1
                raise RuntimeError("simulated kill")

        try:
            train_object_ce(model, tok, obj, exs, diagnostics=diag,
                            log_fn=log_fn, checkpoint_dir=ckdir,
                            checkpoint_key="fp-parity", **kw)
        except RuntimeError:
            assert interrupt_at is not None
            return obj, diag, True
        return obj, diag, False

    ref, ref_diag, _ = run(tmp_path / "full")
    obj1, _d, killed = run(tmp_path / "resume", interrupt_at=3)
    assert killed
    obj2, diag2, _ = run(tmp_path / "resume")  # resumes from the checkpoint
    for a, b in zip(ref.trainable, obj2.trainable):
        assert torch.equal(a, b), "resume is not bitwise"
    assert diag2["heldout_curve"] == ref_diag["heldout_curve"]
    assert diag2["best_epoch"] == ref_diag["best_epoch"]
    assert diag2["heldout_noop_ce"] == ref_diag["heldout_noop_ce"]


# -- empty-prompt / BOS edges, both tokenizer shapes ---------------------------


def test_empty_prompt_bos_adding_and_bosless_shapes(tiny_llama):
    """gemma-shape ('' -> [BOS], prompt_len 1) and qwen-shape ('' -> [],
    prompt_len 0) both train and evaluate correctly: the trainer's held-out
    CE equals the independent per-shape reference (position 0 is never a
    target under either shape — the labels[:, 1:] shift)."""
    from transformers import AutoTokenizer

    model, tok_bos = tiny_llama
    tok_nobos = AutoTokenizer.from_pretrained(TINY)
    tok_nobos.add_bos_token = False
    assert tok_bos("", add_special_tokens=True)["input_ids"] != []
    assert tok_nobos("", add_special_tokens=True)["input_ids"] == []

    docs = [Pair(prompt="", pos="Story body with no prefix at all.",
                 neg="", instruction_id="constv2-mix-1k"),
            Pair(prompt="", pos="Second bare story body.", neg="",
                 instruction_id="constv2-mix-1k")]
    for tok in (tok_bos, tok_nobos):
        exs = docs_to_ce_examples(tok, docs)
        diag: dict = {}
        train_object_ce(model, tok, _fresh("lora", model), exs, lr=1e-4,
                        epochs=1, batch_size=2, max_len=256, heldout=exs,
                        diagnostics=diag)
        manual = _manual_micro_avg_ce(model, tok, exs, obj=None)
        assert diag["heldout_noop_ce"] == pytest.approx(manual, rel=1e-5), (
            f"shape mismatch for add_bos_token="
            f"{getattr(tok, 'add_bos_token', None)}")


def test_templated_tokenizer_examples_stay_verbatim(tiny_llama):
    """A tokenizer that SHIPS a chat template changes nothing: the doc
    builder is verbatim by construction (no build_chat_prompt call), so
    the encoded prompt ids are the raw prompt's ids, template or not."""
    from transformers import AutoTokenizer

    _model, _ = tiny_llama
    tok = AutoTokenizer.from_pretrained(TINY)
    tok.chat_template = ("{% for m in messages %}<|{{ m['role'] }}|>"
                         "{{ m['content'] }}{% endfor %}")
    docs = _docs(2)
    exs = docs_to_ce_examples(tok, docs)
    for p, ex in zip(docs, exs):
        assert ex.prompt_with_si == ex.prompt_without == p.prompt
        assert ex.completion == p.pos
    t_enc, s_enc = _encode_aligned(tok, exs[0], 256)
    raw = tok(docs[0].prompt, add_special_tokens=True)["input_ids"]
    assert s_enc.ids[:s_enc.prompt_len] == raw
    assert t_enc.ids == s_enc.ids  # one leg, identical strings
