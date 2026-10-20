"""Teacher-side prefix KV caching.

Every teacher prompt of a given SI is one shared rendered prefix (chat
header + system=SI) plus a per-pair suffix; extraction/prefix_cache.py runs
the prefix ONCE and replays it as past_key_values with positions offset by
P. The failure class these tests exist for: an off-by-P between the suffix
forward's positions and the completion gather returns WELL-FORMED logits
over the WRONG positions — the KL still computes, the floors still look
plausible, nothing raises. So the parity gate here has four mandatory
parts (step 2, the KL-arm half):

1. parity on GATHERED completion-position logits (raw-logit parity is
   blind to a misaligned gather — both sides would gather identically
   wrong);
2. an INDEPENDENT assertion that the gathered index set equals the
   uncached path's, computed from the encoding's own prompt/completion
   arithmetic, never from the gather code under test;
3. MUTATION tests: an injected off-by-one on the cache_position offset
   (and on the gather offset) must turn the parity test RED — a parity
   test that cannot fail on the bug it exists to catch is worse than none;
4. the no-op floor probe (arm_probe_losses) equal with and without the
   cache — SUPERSEDED by the stronger review-driven design (2026-08-24):
   every decision-bearing pass (the floor probes the floor_ratio_max bar
   rules on, held-out epoch selection) is AUTHORITATIVE AND UNCACHED, flag
   or no flag, so equality is exact by construction. The cache-call-count
   tests below go red if a future change routes any of those passes
   through the cache.

Step 1 (this file's first half) covers the capture/diffmean half — no
completion gather exists on that path (extract_instruction_diff pools at
the last prompt token), so its parity is on pooled activations and the
fitted vector.

CPU-only: the conftest tiny_llama, synthetic prompts — never real store
data.
"""

from __future__ import annotations

import pytest
import torch

from sisteer.data.instructions import Instruction
from sisteer.extraction.capture import collect_pooled_activations
from sisteer.extraction.diffmean import extract_instruction_diff, train_diffmean
from sisteer.extraction.prefix_cache import (
    TeacherPrefixCache,
    common_prefix_ids,
    gather_completion_logits_suffix,
)
from sisteer.extraction.traindata import EncodedSeq, collate
from sisteer.models import build_chat_prompt

# A shared system prefix long enough that P dominates every rendered prompt
# (the production shape: si_constitution renders ~5.9k prefix tokens ahead
# of a ~dozens-token attack suffix).
LONG_SI = ("You must always answer in careful, numbered steps and cite a "
           "source for every claim you make. " * 8).strip()

PROMPTS = [
    "Say something nice.",
    "Describe a tall tree in one sentence.",
    "Greet the reader warmly and briefly.",
    "Name a color and why you like it, in a short reply.",
    "Count to three.",
]


def _instruction(text=LONG_SI) -> Instruction:
    return Instruction(id="test_prefix_si", category="test", text=text,
                       anti_text=None, checker="contains")


def _batch(tok, rows: list[tuple[list[int], int]]):
    """A right-padded Batch from (ids, prompt_len) rows."""
    return collate([EncodedSeq(ids=ids, prompt_len=pl) for ids, pl in rows],
                   tok.pad_token_id)


# ---------------------------------------------------------------------------
# machinery: common-prefix derivation and its caps
# ---------------------------------------------------------------------------


def test_common_prefix_ids_caps_at_min_prompt_len_minus_one(tiny_llama):
    """Identical prompts must NOT cache the whole prompt: every row keeps at
    least one prompt token in the suffix, so the logit predicting the FIRST
    completion token (position prompt_len - 1) is computed by the suffix
    forward."""
    _model, tok = tiny_llama
    prompt = [5, 6, 7, 8]
    b = _batch(tok, [(prompt + [9, 10], 4), (prompt + [11], 4)])
    ids = common_prefix_ids([b])
    assert ids is not None and ids.tolist() == [5, 6, 7]  # 4 - 1


def test_common_prefix_ids_stops_at_first_divergence(tiny_llama):
    _model, tok = tiny_llama
    b = _batch(tok, [([5, 6, 7, 8, 9], 4), ([5, 6, 30, 8, 9], 4)])
    ids = common_prefix_ids([b])
    assert ids is not None and ids.tolist() == [5, 6]


def test_common_prefix_ids_none_when_nothing_shared(tiny_llama):
    _model, tok = tiny_llama
    b = _batch(tok, [([5, 6, 7], 2), ([9, 6, 7], 2)])
    assert common_prefix_ids([b]) is None
    assert common_prefix_ids([]) is None


def test_common_prefix_spans_all_batches(tiny_llama):
    """The group-level prefix is the min across BATCHES too — a later batch
    diverging earlier must shorten it (one cache serves every batch)."""
    _model, tok = tiny_llama
    b1 = _batch(tok, [([5, 6, 7, 8, 9], 4), ([5, 6, 7, 8, 10], 4)])
    b2 = _batch(tok, [([5, 6, 20, 8, 9], 4)])
    ids = common_prefix_ids([b1, b2])
    assert ids is not None and ids.tolist() == [5, 6]


def test_match_len_rederives_per_batch_and_crops(tiny_llama):
    """A batch sharing only part of the stored prefix (held-out rows, a
    probe batch) gets exactly the part it shares — never the full stored
    length on trust."""
    model, tok = tiny_llama
    cache = TeacherPrefixCache.build(model, [5, 6, 7, 8])
    full = _batch(tok, [([5, 6, 7, 8, 9, 10], 5)])
    partial = _batch(tok, [([5, 6, 99, 8, 9, 10], 5)])
    foreign = _batch(tok, [([50, 60, 70, 80, 90], 4)])
    assert cache.match_len(full) == 4
    assert cache.match_len(partial) == 2
    assert cache.match_len(foreign) == 0


def test_gather_suffix_refuses_completion_inside_prefix(tiny_llama):
    """The off-by-P guard: a completion token at or before the prefix
    boundary has no computed logit — refuse loudly, never return
    well-formed logits over the wrong positions."""
    _model, tok = tiny_llama
    b = _batch(tok, [([5, 6, 7, 8, 9], 2)])  # completion starts at index 2
    fake = torch.zeros((1, 2, 11))  # suffix logits for p=3 (positions 3..4)
    with pytest.raises(ValueError, match="off-by-P"):
        gather_completion_logits_suffix(fake, b, p=3)


# ---------------------------------------------------------------------------
# step 1 — the capture/diffmean half
# ---------------------------------------------------------------------------


def _rendered(tok, system):
    return [build_chat_prompt(tok, p, system=system) for p in PROMPTS]


def test_collect_pooled_last_prefix_cache_parity(tiny_llama):
    """position='last' (the extract_instruction_diff pool): cached and
    uncached sweeps pool the SAME positions to fp tolerance, across a
    batch boundary (batch_size=2 over 5 texts)."""
    model, tok = tiny_llama
    texts = _rendered(tok, LONG_SI)
    plain = collect_pooled_activations(model, tok, texts, [0, 1],
                                       position="last", batch_size=2)
    cached = collect_pooled_activations(model, tok, texts, [0, 1],
                                        position="last", batch_size=2,
                                        prefix_cache=True)
    for l in (0, 1):
        assert plain.acts[l].shape == cached.acts[l].shape
        torch.testing.assert_close(cached.acts[l], plain.acts[l],
                                   atol=1e-5, rtol=1e-4)


def test_collect_pooled_completion_specs_prefix_cache_parity(tiny_llama):
    """completion_mean / completion_last (the CAA pools): parity with
    per-row prompt_lens — the pooled support starts at each row's own
    prompt boundary, all inside the suffix."""
    model, tok = tiny_llama
    prompts = _rendered(tok, LONG_SI)
    comps = [" One.", " Two two.", " Three three three.", " Four.", " Five."]
    texts = [p + c for p, c in zip(prompts, comps)]
    plens = [len(tok(p, add_special_tokens=True)["input_ids"])
             for p in prompts]
    for spec in ("completion_mean", "completion_last"):
        plain = collect_pooled_activations(model, tok, texts, [1],
                                           position=spec, prompt_lens=plens,
                                           batch_size=2)
        cached = collect_pooled_activations(model, tok, texts, [1],
                                            position=spec, prompt_lens=plens,
                                            batch_size=2, prefix_cache=True)
        torch.testing.assert_close(cached.acts[1], plain.acts[1],
                                   atol=1e-5, rtol=1e-4)


def test_collect_pooled_refuses_mean_with_prefix_cache(tiny_llama):
    """'mean' pools every real position, prefix included — a suffix-only
    capture would silently pool a truncated support, so it must refuse."""
    model, tok = tiny_llama
    with pytest.raises(ValueError, match="mean"):
        collect_pooled_activations(model, tok, _rendered(tok, LONG_SI), [1],
                                   position="mean", prefix_cache=True)


def test_collect_pooled_no_shared_prefix_falls_back(tiny_llama):
    """Texts sharing nothing still work under the flag (the exact uncached
    forward — 'nothing to cache' is an event, not an error)."""
    model, tok = tiny_llama
    texts = ["alpha beta", "zebra yak", "quartz opal"]
    plain = collect_pooled_activations(model, tok, texts, [1],
                                       position="last", batch_size=2)
    cached = collect_pooled_activations(model, tok, texts, [1],
                                        position="last", batch_size=2,
                                        prefix_cache=True)
    torch.testing.assert_close(cached.acts[1], plain.acts[1],
                               atol=1e-5, rtol=1e-4)


def test_extract_instruction_diff_prefix_cache_parity(tiny_llama):
    """The step-1 headline: the Stolfo instruction-diff vector (and its
    projection targets) computed with the cache matches the plain path to
    fp tolerance on BOTH sweeps (with-SI: long shared prefix; without-SI:
    bare chat header)."""
    model, tok = tiny_llama
    with_si = _rendered(tok, LONG_SI)
    without = _rendered(tok, None)
    plain = extract_instruction_diff(model, tok, with_si, without,
                                     layers=[1], batch_size=2)
    cached = extract_instruction_diff(model, tok, with_si, without,
                                      layers=[1], batch_size=2,
                                      prefix_cache=True)
    torch.testing.assert_close(cached.vectors[1], plain.vectors[1],
                               atol=1e-5, rtol=1e-4)
    t_plain = plain.metadata["projection_targets"]["1"]
    t_cached = cached.metadata["projection_targets"]["1"]
    assert abs(t_plain - t_cached) < 1e-4


# ---------------------------------------------------------------------------
# step 2 — the KL-arm half: the four-part parity gate
#
# The tolerances here are CPU float32 (the tiny fixture): a correct cached
# forward lands ~1e-7 from the plain one, an injected off-by-one lands
# ~1e-4+, so PARITY_ATOL=1e-5 separates them by an order of magnitude on
# each side. (The spec's ~1e-3 figure is for bf16 on the real model, where
# baseline fp noise is ~1e-3; the SAME gate at that tolerance would still
# catch the mutation there, whose divergence grows with real trained
# position sensitivity.)
# ---------------------------------------------------------------------------

PARITY_ATOL = 1e-5
KL_MAX_LEN = 512  # holds the LONG_SI teacher prompts + full completions

COMPLETIONS = [" Oui, bien sur.", " C'est un arbre.", " Salut!",
               " Le bleu.", " Un, deux, trois.", " Bonjour a tous.",
               " Merci beaucoup.", " D'accord, tres bien."]
MORE_PROMPTS = PROMPTS + [
    "Thank the reader politely.",
    "Agree with the previous statement.",
    "Say goodbye in one short sentence.",
]


def _kl_examples(tok, n=8):
    from sisteer.extraction.distill import DistillExample

    return [DistillExample(
        prompt_with_si=build_chat_prompt(tok, p, system=LONG_SI),
        prompt_without=build_chat_prompt(tok, p, system=None),
        completion=c)
        for p, c in zip(MORE_PROMPTS[:n], COMPLETIONS[:n])]


def _teacher_batches(tok, exs, batch_size=2):
    from sisteer.extraction.distill import _encode_aligned

    encoded = [_encode_aligned(tok, ex, KL_MAX_LEN) for ex in exs]
    return [collate([t for t, _ in encoded[i:i + batch_size]],
                    tok.pad_token_id)
            for i in range(0, len(encoded), batch_size)]


def _gathered_parity_max_diff(model, tpc, t_batches) -> float:
    """THE parity check (requirement 1): max abs difference between the
    prefix-cached and plain paths' GATHERED completion-position logits —
    gathered, never raw, because a misaligned gather applied to BOTH sides
    would cancel in a raw-logit comparison and hide exactly the bug this
    gate exists for. The mutation tests reuse this same check, so 'the
    parity test goes red' is literal."""
    from sisteer.extraction.distill import _gather_completion_logits

    worst = 0.0
    for tb in t_batches:
        with torch.no_grad():
            plain = _gather_completion_logits(
                model(input_ids=tb.input_ids,
                      attention_mask=tb.attention_mask).logits, tb)
        cached = tpc.completion_logits(model, tb)
        assert cached.shape == plain.shape
        worst = max(worst, float((cached - plain).abs().max()))
    return worst


def test_kl_parity_gathered_completion_logits(tiny_llama):
    """Requirement 1: cached and plain teacher legs agree on the gathered
    completion-position logits, across batches, with the cache genuinely
    engaged (the prefix must cover the whole rendered SI, not just a few
    header tokens)."""
    from sisteer.extraction.prefix_cache import prefix_cache_for_batches

    model, tok = tiny_llama
    t_batches = _teacher_batches(tok, _kl_examples(tok))
    tpc = prefix_cache_for_batches(model, t_batches)
    assert tpc is not None
    # The shared SI must dominate the prefix: LONG_SI alone is > 100 tokens.
    assert len(tpc) > 100, f"prefix only {len(tpc)} tokens — cache not engaged"
    assert _gathered_parity_max_diff(model, tpc, t_batches) < PARITY_ATOL


def test_kl_parity_gather_index_set_independent(tiny_llama):
    """Requirement 2: the gathered index SET (and order) equals the uncached
    path's — computed INDEPENDENTLY from the encoding's own prompt/length
    arithmetic (EncodedSeq: completion tokens occupy [prompt_len, n), so
    gathered logit positions are [prompt_len-1, n-1) per row, row-major),
    never from the gather code under test. Checked by fingerprinting: fake
    'logits' encode (row, position), so the gather's output IS its index
    list."""
    from sisteer.extraction.distill import (_encode_aligned,
                                            _gather_completion_logits)
    from sisteer.extraction.prefix_cache import (
        common_prefix_ids, gather_completion_logits_suffix)

    model, tok = tiny_llama
    exs = _kl_examples(tok, n=4)
    encoded = [_encode_aligned(tok, ex, KL_MAX_LEN)[0] for ex in exs]
    tb = collate(encoded, tok.pad_token_id)
    p = int(common_prefix_ids([tb]).numel())
    assert p > 100  # the cache is engaged, not vacuously small

    # Independent expectation, straight from the encoding arithmetic.
    expected = [(i, pos)
                for i, seq in enumerate(encoded)
                for pos in range(seq.prompt_len - 1, len(seq.ids) - 1)]

    b, s = tb.input_ids.shape
    fake = torch.zeros(b, s, 2)
    fake[..., 0] = torch.arange(b)[:, None].float()  # row fingerprint
    fake[..., 1] = torch.arange(s)[None, :].float()  # position fingerprint
    plain = _gather_completion_logits(fake, tb)
    cached = gather_completion_logits_suffix(fake[:, p:], tb, p)
    got_plain = [(int(r), int(q)) for r, q in plain.tolist()]
    got_cached = [(int(r), int(q)) for r, q in cached.tolist()]
    assert got_plain == expected      # the uncached path == the arithmetic
    assert got_cached == expected     # the cached path == the same set+order


def test_kl_parity_mutation_cache_offset_goes_red(tiny_llama, monkeypatch):
    """Requirement 3 (the named one): an injected off-by-one on the
    cache_position offset must turn the parity test RED. The failure mode is
    SILENT — same shapes, well-formed logits, wrong values (RoPE shifted by
    one) — which is exactly why a parity test that cannot fail on it would
    be worse than none. Both directions."""
    import sisteer.extraction.prefix_cache as pc

    model, tok = tiny_llama
    t_batches = _teacher_batches(tok, _kl_examples(tok))
    tpc = pc.prefix_cache_for_batches(model, t_batches)
    # green before the mutation, on the same check
    assert _gathered_parity_max_diff(model, tpc, t_batches) < PARITY_ATOL
    for delta in (-1, +1):
        monkeypatch.setattr(pc, "_cache_offset", lambda p, d=delta: p + d)
        diff = _gathered_parity_max_diff(model, tpc, t_batches)
        assert diff > PARITY_ATOL, (
            f"cache_position off-by-{delta:+d} was NOT caught: max gathered "
            f"diff {diff:.2e} <= {PARITY_ATOL:.0e} — the parity gate cannot "
            "detect the bug it exists for")
        monkeypatch.undo()


def test_kl_parity_mutation_gather_offset_goes_red(tiny_llama, monkeypatch):
    """Requirement 3, second seam: an off-by-one on the GATHER offset (the
    labels-side re-basing) must also turn the parity test red — here the
    misalignment surfaces as a loud failure (the off-by-P guard or a mask
    width mismatch) or a value divergence; either way the gate goes red,
    never silently green."""
    import sisteer.extraction.prefix_cache as pc

    model, tok = tiny_llama
    t_batches = _teacher_batches(tok, _kl_examples(tok))
    tpc = pc.prefix_cache_for_batches(model, t_batches)
    assert _gathered_parity_max_diff(model, tpc, t_batches) < PARITY_ATOL
    for delta in (-1, +1):
        monkeypatch.setattr(pc, "_gather_offset", lambda p, d=delta: p + d)
        try:
            diff = _gathered_parity_max_diff(model, tpc, t_batches)
        except (ValueError, RuntimeError, IndexError):
            pass  # red by raising — acceptable; silent green is the bug
        else:
            assert diff > PARITY_ATOL, (
                f"gather off-by-{delta:+d} was NOT caught (max diff "
                f"{diff:.2e})")
        monkeypatch.undo()


# ---------------------------------------------------------------------------
# decision-bearing passes are AUTHORITATIVE AND UNCACHED (review finding
# 2026-08-24): the floor probes rule the hard floor_ratio_max bar (a
# rejection writes NO artifact, so a cache-flipped verdict is
# unrecoverable), and the held-out pass selects the restored epoch. These
# tests go RED if a future change routes any decision pass through the
# cache.
# ---------------------------------------------------------------------------


def _count_cache_forwards(monkeypatch):
    """Spy on the ONE model-call seam every cached forward goes through
    (TeacherPrefixCache.forward_ids). Any decision pass routed through the
    cache inflates the count — the regression these tests exist for."""
    import sisteer.extraction.prefix_cache as pc

    calls = []
    real = pc.TeacherPrefixCache.forward_ids

    def spy(self, model, input_ids, attention_mask, p):
        calls.append(int(input_ids.shape[0]))
        return real(self, model, input_ids, attention_mask, p)

    monkeypatch.setattr(pc.TeacherPrefixCache, "forward_ids", spy)
    return calls


def test_arm_probe_losses_never_touches_the_cache(tiny_llama, monkeypatch):
    """The solo floor guard's probe is authoritative by construction: it
    takes no cache knob, and it must never call into the cache machinery —
    its numbers are what floor_ratio_check rules on."""
    import inspect

    import sisteer.extraction.prefix_cache as pc
    from sisteer.data.pairs import Pair
    from sisteer.extraction.lockstep import (arm_probe_losses,
                                             build_lockstep_object)

    assert "teacher_prefix_cache" not in inspect.signature(
        arm_probe_losses).parameters

    def boom(*a, **k):
        raise AssertionError(
            "arm_probe_losses reached the teacher prefix cache — the "
            "decision-bearing probe must stay uncached")

    monkeypatch.setattr(pc.TeacherPrefixCache, "build", boom)
    monkeypatch.setattr(pc.TeacherPrefixCache, "forward_ids", boom)
    model, tok = tiny_llama
    instruction = _instruction()
    pairs = [Pair(prompt=p, pos="oui", neg="non", instruction_id=instruction.id)
             for p in MORE_PROMPTS]
    obj = build_lockstep_object("vector", model, tok, instruction, pairs,
                                layer=1)
    est = arm_probe_losses(model, tok, obj, _kl_examples(tok, n=4),
                           max_len=KL_MAX_LEN, batch_size=2, n_probe=None)
    assert est.n_batches == 2  # it really ran


def test_lockstep_cache_serves_training_forwards_only(tiny_llama,
                                                      monkeypatch):
    """train_lockstep under the flag: the cache serves EXACTLY the
    gradient-training shared forwards — epochs x n_batches, not one more.
    The startup floor, every held-out pass, and the final probe run
    uncached (they decide the floor_ratio verdict and the restored epoch),
    so their numbers are BITWISE identical with the flag on or off — also
    asserted, on the floor and step-0 numbers that do not depend on the
    (fp-divergent) trained weights."""
    from sisteer.data.pairs import Pair, pairs_to_distill_examples
    from sisteer.extraction.distill import EarlyStopSpec
    from sisteer.extraction.lockstep import (LockstepArm,
                                             build_lockstep_object,
                                             default_lockstep_params,
                                             train_lockstep)

    model, tok = tiny_llama
    instruction = _instruction()
    pairs = [Pair(prompt=p, pos=c.strip(), neg="no",
                  instruction_id=instruction.id)
             for p, c in zip(MORE_PROMPTS[:4], COMPLETIONS)]
    examples = pairs_to_distill_examples(tok, pairs, instruction)
    heldout = pairs_to_distill_examples(
        tok, [Pair(prompt=MORE_PROMPTS[4], pos="oui", neg="non",
                   instruction_id=instruction.id)], instruction)
    epochs, batch_size = 2, 2
    n_batches = -(-len(examples) // batch_size)

    def run(flag, calls=None):
        group = [LockstepArm(
            "vector_all",
            build_lockstep_object("vector_all", model, tok, instruction,
                                  pairs, layer=1),
            default_lockstep_params("vector_all"))]
        diag: dict = {}
        train_lockstep(model, tok, group, examples, epochs=epochs,
                       batch_size=batch_size, max_len=KL_MAX_LEN, seed=0,
                       heldout=heldout,
                       # patience > epochs: no early stop, so the expected
                       # training-forward count is exact.
                       early_stop=EarlyStopSpec(patience=99),
                       diagnostics=diag, teacher_prefix_cache=flag)
        return diag["vector_all"]

    calls = _count_cache_forwards(monkeypatch)
    d_cached = run(True)
    assert len(calls) == epochs * n_batches, (
        f"cache served {len(calls)} forwards, expected exactly "
        f"{epochs * n_batches} training forwards — a decision pass "
        "(probe/held-out) is reading cached kernels")
    monkeypatch.undo()
    d_plain = run(False)
    for key in ("noop_floor", "noop_floor_sd", "step0_loss"):
        assert d_cached[key] == d_plain[key], (
            f"{key} differs with the flag ({d_cached[key]!r} vs "
            f"{d_plain[key]!r}) — the decision path is not cache-independent")


def test_steered_lm_decision_probes_uncached(tiny_llama, monkeypatch):
    """train_steered_lm under the flag: cache-served forwards == training
    steps exactly; the probe (floor_ratio_check's numbers) and every
    held-out pass (epoch selection) stay uncached."""
    from torch import nn

    from sisteer.extraction.distill import EarlyStopSpec, train_steered_lm
    from sisteer.models import hidden_size

    model, tok = tiny_llama
    exs = _kl_examples(tok, n=4)
    heldout = _kl_examples(tok, n=6)[4:]
    epochs, batch_size = 2, 2
    n_batches = -(-len(exs) // batch_size)
    vec = nn.Parameter(torch.zeros(hidden_size(model)))
    calls = _count_cache_forwards(monkeypatch)
    train_steered_lm(
        model, tok, exs, 1, vector_provider=lambda idxs: vec,
        trainable_params=[vec], epochs=epochs, batch_size=batch_size,
        max_len=KL_MAX_LEN, seed=0, heldout_examples=heldout,
        early_stop=EarlyStopSpec(patience=99), floor_ratio_max=1e9,
        teacher_prefix_cache=True)
    assert len(calls) == epochs * n_batches, (
        f"cache served {len(calls)} forwards, expected exactly "
        f"{epochs * n_batches} training forwards")


def test_train_object_kl_probe_uncached(tiny_llama, monkeypatch):
    """train_object_kl under the flag: cache-served forwards == training
    steps exactly; the startup probe (the same-batch floor convention)
    stays uncached."""
    from sisteer.data.pairs import Pair
    from sisteer.extraction.lockstep import build_lockstep_object
    from sisteer.extraction.objects import train_object_kl

    model, tok = tiny_llama
    instruction = _instruction()
    pairs = [Pair(prompt=p, pos=c.strip(), neg="no",
                  instruction_id=instruction.id)
             for p, c in zip(MORE_PROMPTS[:4], COMPLETIONS)]
    obj = build_lockstep_object("lora_r1", model, tok, instruction, pairs,
                                layer=1)
    exs = _kl_examples(tok, n=4)
    epochs, batch_size = 2, 2
    n_batches = -(-len(exs) // batch_size)
    calls = _count_cache_forwards(monkeypatch)
    train_object_kl(model, tok, obj, exs, epochs=epochs,
                    batch_size=batch_size, max_len=KL_MAX_LEN, seed=0,
                    teacher_prefix_cache=True)
    assert len(calls) == epochs * n_batches


def test_batch_cache_materialises_copy_on_this_transformers(tiny_llama):
    """Pins the DOCUMENTED memory model: the expand() views handed to
    DynamicCache are materialised into a contiguous batch-sized COPY by
    transformers 5.14.x at construction — so per-batch prefix memory is
    ~B x P, NOT P, and the MEMORY notes in prefix_cache.py /
    configs/two_si.yaml say so. If this test goes red, transformers
    changed the semantics under us: re-verify and update those notes
    (in whichever direction) before trusting them."""
    from sisteer.extraction.prefix_cache import _kv_pairs

    model, tok = tiny_llama
    cache = TeacherPrefixCache.build(model, [5, 6, 7, 8])
    dc = cache._batch_cache(3, 4)
    src_k, src_v = cache._kv[0]
    in_k, in_v = _kv_pairs(dc)[0]
    assert in_k.shape[0] == 3 and in_v.shape[0] == 3
    # The copy: different storage, contiguous, B x the source bytes.
    assert in_k.data_ptr() != src_k.data_ptr()
    assert in_k.is_contiguous()
    # The source stays the single pristine [1, ...] build — never widened,
    # never mutated (what keeps it valid across batches).
    assert src_k.shape[0] == 1 and src_v.shape[0] == 1


def test_lockstep_bitwise_contract_holds_under_prefix_cache(tiny_llama):
    """The load-bearing lockstep invariant survives the cache: solo (registry
    trainer, flag on) == lockstep (flag on), BITWISE — both sides derive the
    identical prefix from the identical fit rows and run identical cached
    kernels on identical batches. The flag must never be split across the
    two sides of a bitwise comparison (cached-vs-uncached is fp-tolerance
    equivalence, not bitwise — the row-sharding class)."""
    from sisteer.data.pairs import Pair, pairs_to_distill_examples
    from sisteer.extraction.lockstep import (LockstepArm,
                                             build_lockstep_object,
                                             default_lockstep_params,
                                             train_lockstep)
    from sisteer.methods import ARMS

    model, tok = tiny_llama
    instruction = _instruction()
    pairs = [Pair(prompt=p, pos=c.strip(), neg="no",
                  instruction_id=instruction.id)
             for p, c in zip(MORE_PROMPTS[:4], COMPLETIONS)]
    layer = 1
    kw = dict(epochs=2, batch_size=2, max_len=KL_MAX_LEN, seed=0)

    solo = ARMS["lora_r1"].train(model, tok, instruction, pairs, layer=layer,
                                 teacher_prefix_cache=True, **kw)

    examples = pairs_to_distill_examples(tok, pairs, instruction)
    group = [
        LockstepArm("lora_r1",
                    build_lockstep_object("lora_r1", model, tok, instruction,
                                          pairs, layer=layer),
                    default_lockstep_params("lora_r1")),
        LockstepArm("vector_all",
                    build_lockstep_object("vector_all", model, tok,
                                          instruction, pairs, layer=layer),
                    default_lockstep_params("vector_all")),
    ]
    train_lockstep(model, tok, group, examples, teacher_prefix_cache=True,
                   **kw)
    for i, (a, b) in enumerate(zip(solo.trainable, group[0].obj.trainable)):
        assert torch.equal(a.detach(), b.detach()), (
            f"lora_r1 param {i} not bitwise identical under the cache "
            f"(max abs diff {float((a - b).abs().max())})")


def test_train_lockstep_cached_vs_uncached_histories_close(tiny_llama):
    """End to end: a lockstep group trained with and without the cache walks
    fp-tolerance-close loss trajectories (drift compounds over steps, so the
    bound is looser than single-forward parity — this is the documented
    'semantically exact, not bitwise' contract, the same class as row
    sharding)."""
    from sisteer.data.pairs import Pair, pairs_to_distill_examples
    from sisteer.extraction.lockstep import (LockstepArm,
                                             build_lockstep_object,
                                             default_lockstep_params,
                                             train_lockstep)

    model, tok = tiny_llama
    instruction = _instruction()
    pairs = [Pair(prompt=p, pos=c.strip(), neg="no",
                  instruction_id=instruction.id)
             for p, c in zip(MORE_PROMPTS[:4], COMPLETIONS)]
    examples = pairs_to_distill_examples(tok, pairs, instruction)
    kw = dict(epochs=2, batch_size=2, max_len=KL_MAX_LEN, seed=0)

    def run(flag):
        group = [LockstepArm("vector_all",
                             build_lockstep_object("vector_all", model, tok,
                                                   instruction, pairs,
                                                   layer=1),
                             default_lockstep_params("vector_all"))]
        hist = train_lockstep(model, tok, group, examples,
                              teacher_prefix_cache=flag, **kw)
        return hist["vector_all"]

    plain, cached = run(False), run(True)
    assert len(plain) == len(cached)
    for a, b in zip(plain, cached):
        assert abs(a - b) <= 1e-4 + 1e-3 * abs(a)


def test_config_flag_documented_and_default_true():
    """The two-SI config carries the flag, ON — the wave's default — with
    library defaults OFF (the driver is the one place the config value
    fans out from)."""
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load(
        (Path(__file__).parent.parent / "configs" / "two_si.yaml")
        .read_text())
    assert cfg["teacher_prefix_cache"] is True


def test_train_diffmean_prefix_cache_parity_and_provenance(tiny_llama):
    """The full registry arm: cached and uncached fits agree to fp
    tolerance, and the artifact records which path fitted it (the escape
    hatch must be auditable after the fact)."""
    from sisteer.data.pairs import Pair

    model, tok = tiny_llama
    instruction = _instruction()
    pairs = [Pair(prompt=p, pos="oui", neg="no", instruction_id="test_prefix_si")
             for p in PROMPTS]
    plain = train_diffmean(model, tok, instruction, pairs, layer=1,
                           teacher_prefix_cache=False)
    cached = train_diffmean(model, tok, instruction, pairs, layer=1,
                            teacher_prefix_cache=True)
    torch.testing.assert_close(cached.vectors[1], plain.vectors[1],
                               atol=1e-5, rtol=1e-4)
    assert cached.metadata["teacher_prefix_cache"] is True
    assert plain.metadata["teacher_prefix_cache"] is False
