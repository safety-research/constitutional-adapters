"""Per-SI max_len + per-(SI, arm) batch-size resolution and the CPU-only
measurement harness (efficiency optimisations 1, 2, 5).

The load-bearing test is the COMPARABILITY invariant of opt. 1: per-SI
``max_len`` is a memory knob, so it must NOT change the completion-token budget
entering the loss. As long as each SI's ``max_len`` holds prompt + full
completion, the supervised completion tokens are IDENTICAL across SIs — which
is the invariant that protects the two-SI comparison. This is asserted
directly against ``_encode_aligned``'s own truncation.

CPU-only: a hermetic whitespace stub tokenizer for the invariant/wiring tests
(no cache, no network), the tiny_llama fixture for the object-param harness,
and the pinned Llama-3.1 tokenizer (skips when absent) for the real numbers.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pathlib

import pytest

from sisteer.extraction.distill import DistillExample, _encode_aligned
from sisteer.extraction.traindata import collate
from sisteer.eval.train_budget import (
    DEFAULT_BATCH_SIZE,
    FISHER_MAX_BATCH,
    CellBudget,
    attention_cost_class,
    derive_max_len,
    format_budget_table,
    measure_si_budget,
    resolve_batch_size,
    resolve_max_len,
)

PINNED_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PINNED_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
TWO_SI_CONFIG = pathlib.Path(__file__).resolve().parents[1] / "configs/two_si.yaml"


class WordTokenizer:
    """Whitespace stub exposing exactly the surface the budget code uses."""

    eos_token_id = 1
    pad_token_id = 0
    bos_token_id = 2

    def __call__(self, text, add_special_tokens=True):
        ids = [3 + (i % 1000) for i, _ in enumerate(text.split())]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return {"input_ids": ids}


def _example(si_words: int, attack_words: int, completion_words: int):
    si = " ".join(["si"] * si_words)
    attack = " ".join(["attack"] * attack_words)
    completion = " ".join(["c"] * completion_words)
    return DistillExample(prompt_with_si=f"{si} {attack}",
                          prompt_without=attack, completion=completion)


# -- opt. 1: per-SI max_len derivation + resolution ---------------------------


def test_derive_max_len_arithmetic_and_pow2():
    # 124 SI + 1024 attack + 512 comp + 1 EOS = 1661 -> next pow2 2048
    assert derive_max_len(124, 512, attack_allowance=1024) == 2048
    # 5908 + 1024 + 512 + 1 = 7445 -> 8192
    assert derive_max_len(5908, 512, attack_allowance=1024) == 8192
    # raw (no rounding) returns the exact minimum
    assert derive_max_len(124, 512, attack_allowance=1024,
                          round_up_pow2=False) == 1661


def test_resolve_max_len_prefers_per_si_then_falls_back():
    cfg = {"max_len": 8192, "max_len_by_si": {"safety_refusal": 2048}}
    assert resolve_max_len(cfg, "safety_refusal") == 2048       # per-SI
    assert resolve_max_len(cfg, "si_constitution") == 8192      # scalar fallback
    # absent per-SI map -> scalar (behaviour identical to one shared budget)
    assert resolve_max_len({"max_len": 4096}, "anything") == 4096
    with pytest.raises(KeyError):
        resolve_max_len({}, "safety_refusal")


def test_completion_supervision_identical_across_per_si_max_len():
    """THE comparability invariant (opt. 1): per-SI max_len must not
    change the completion tokens entering the loss. A short-SI arm at a small
    max_len and a long-SI arm at a large max_len supervise the SAME number of
    completion tokens, because neither truncates the completion.

    Counted directly from _encode_aligned + collate's label mask — the exact
    tensor the loss gathers over."""
    tok = WordTokenizer()
    completion_words = 30
    safety_ex = _example(si_words=10, attack_words=20,
                         completion_words=completion_words)
    constitution_ex = _example(si_words=400, attack_words=20,
                               completion_words=completion_words)

    safety_max_len = resolve_max_len(
        {"max_len_by_si": {"safety_refusal": 64}}, "safety_refusal")
    constitution_max_len = resolve_max_len(
        {"max_len_by_si": {"si_constitution": 512}}, "si_constitution")
    assert safety_max_len != constitution_max_len  # different memory budgets

    def supervised_tokens(ex, max_len):
        _t, student = _encode_aligned(tok, ex, max_len)
        batch = collate([student], tok.pad_token_id)
        # labels != -100 marks the supervised completion positions
        return int((batch.labels != -100).sum())

    n_safety = supervised_tokens(safety_ex, safety_max_len)
    n_constitution = supervised_tokens(constitution_ex, constitution_max_len)

    # full completion (30 words) + EOS = 31 tokens, uncut on BOTH sides
    assert n_safety == n_constitution == completion_words + 1


def test_truncation_would_break_the_invariant_is_caught():
    """A max_len too small to hold the full completion DOES change supervision
    volume — the exact thing per-SI sizing must avoid. Shown so the invariant
    test above is not vacuous."""
    tok = WordTokenizer()
    ex = _example(si_words=10, attack_words=20, completion_words=30)
    _t, student_full = _encode_aligned(tok, ex, 64)
    _t, student_cut = _encode_aligned(tok, ex, 40)  # truncates the completion
    full = int((collate([student_full], tok.pad_token_id).labels != -100).sum())
    cut = int((collate([student_cut], tok.pad_token_id).labels != -100).sum())
    assert cut < full


# -- opt. 2: per-(SI, arm) batch size -----------------------------------------


def test_batch_size_resolution_precedence():
    cfg = {
        "train_kwargs": {"default": {"batch_size": 4}},
        "batch_size_by_si": {"si_constitution": 2},
        "batch_size_by_arm": {"lora_r1": 16},
    }
    # per-arm override wins over per-SI and default
    assert resolve_batch_size(cfg, "si_constitution", "lora_r1") == 16
    # per-SI wins over the global default
    assert resolve_batch_size(cfg, "si_constitution", "psr_kl_all") == 2
    # falls back to train_kwargs.default for an SI with no per-SI entry
    assert resolve_batch_size(cfg, "safety_refusal", "psr_kl_all") == 4
    # bare config -> the module default
    assert resolve_batch_size({}, "safety_refusal", "vector_all") == DEFAULT_BATCH_SIZE


def test_fisher_rung_is_hard_capped_regardless_of_config():
    """alphakl_fisher's O(T^2) attention caps its batch at 2 even when the
    config asks for more — a memory fact, not a preference."""
    cfg = {"batch_size_by_arm": {"alphakl_fisher": 32},
           "batch_size_by_si": {"si_constitution": 16}}
    assert resolve_batch_size(cfg, "si_constitution", "alphakl_fisher") == FISHER_MAX_BATCH
    # and the non-fisher arm at the same SI is NOT capped
    assert resolve_batch_size(cfg, "si_constitution", "vector_all") == 16


def test_attention_cost_class():
    assert attention_cost_class("alphakl_fisher") == "O(T^2)"
    assert attention_cost_class("vector_all") == "O(T)"
    assert attention_cost_class("lora_r1") == "O(T)"


# -- opt. 5: the measurement harness ------------------------------------------


def test_measure_si_budget_shapes_token_counts_and_gpu_marks():
    tok = WordTokenizer()
    examples = [_example(si_words=100, attack_words=10, completion_words=5),
                _example(si_words=100, attack_words=30, completion_words=5)]
    cfg = {"completion_tokens": 512, "max_len_by_si": {"si_constitution": 4096},
           "train_kwargs": {"default": {"epochs": 3}}}
    rows = measure_si_budget(
        cfg, tok, examples, si_id="si_constitution",
        arms=["vector_all", "alphakl_fisher"],
        candidate_batch_sizes=[1, 4],
        compatible=lambda a: a in {"vector_all"})
    # one row per (arm, candidate batch size)
    assert len(rows) == 4
    by = {(r.arm, r.batch_size): r for r in rows}
    v = by[("vector_all", 4)]
    assert v.max_len == 4096
    assert v.completion_tokens == 512
    assert v.teacher_prompt_tokens == 1 + 100 + 30  # BOS + si + longest attack
    assert v.attention_cost == "O(T)"
    assert v.teacher_forward == "shared"
    # fisher is O(T^2) and solo, and its work proxy scales with T^2
    f1 = by[("alphakl_fisher", 1)]
    assert f1.attention_cost == "O(T^2)"
    assert f1.teacher_forward == "solo"
    assert f1.per_step_work_proxy == 1 * f1.total_seq_tokens ** 2
    # GPU-only fields are explicitly pending, never fabricated
    assert v.peak_memory_gb == "NEEDS_GPU"
    assert v.tokens_per_s == "NEEDS_GPU"
    assert v.est_seconds == "NEEDS_GPU"


def test_measure_si_budget_time_estimate_with_measured_tokens_per_s():
    tok = WordTokenizer()
    examples = [_example(si_words=10, attack_words=5, completion_words=5)]
    cfg = {"completion_tokens": 8, "max_len": 128,
           "train_kwargs": {"default": {"epochs": 2}}}
    rows = measure_si_budget(cfg, tok, examples, si_id="safety_refusal",
                             arms=["vector_all"], tokens_per_s=1000.0)
    assert rows[0].est_seconds != "NEEDS_GPU"
    assert float(rows[0].est_seconds) > 0


def test_measure_si_budget_counts_params_and_inject_on_real_objects(tiny_llama):
    """With the loaded model, the harness counts trainable params and reflects
    a soft-token arm's +1 sentinel in the student prompt length."""
    from sisteer.extraction.lockstep import build_lockstep_object, lockstep_compatible
    from sisteer.data.instructions import BY_ID
    from sisteer.data.pairs import Pair, pairs_to_distill_examples

    model, tok = tiny_llama
    instruction = BY_ID["french"]
    pairs = [Pair(prompt="Greet.", pos="Bonjour", neg="Hi", instruction_id="french")]
    examples = pairs_to_distill_examples(tok, pairs, instruction)

    def provider(arm):
        kw = {"sentinel": (tok.unk_token, tok.unk_token_id)} if arm == "token_bos" else {}
        return build_lockstep_object(arm, model, tok, instruction, pairs,
                                     layer=1, **kw)

    rows = measure_si_budget(
        {"completion_tokens": 16, "max_len": 128}, tok, examples,
        si_id="safety_refusal", arms=["lora_r1", "token_bos"],
        object_provider=provider, compatible=lockstep_compatible)
    by = {r.arm: r for r in rows}
    assert by["lora_r1"].n_trainable_params and by["lora_r1"].n_trainable_params > 0
    # the soft token adds exactly one sentinel token to the student prompt
    assert by["token_bos"].student_prompt_tokens == by["lora_r1"].student_prompt_tokens + 1


def test_format_budget_table_is_a_small_table_marking_gpu_fields():
    tok = WordTokenizer()
    examples = [_example(si_words=5, attack_words=5, completion_words=5)]
    rows = measure_si_budget({"completion_tokens": 8, "max_len": 64}, tok,
                             examples, si_id="safety_refusal", arms=["vector_all"])
    table = format_budget_table(rows)
    assert "arm" in table and "max_len" in table
    assert "NEEDS_GPU" in table
    assert "vector_all" in table


# -- the config wires the additive keys (pinned tokenizer where present) ------


def _two_si_cfg() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(TWO_SI_CONFIG.read_text(encoding="utf-8"))


def test_config_declares_per_si_max_len_falling_back_compatibly():
    cfg = _two_si_cfg()
    assert cfg["max_len_by_si"]["si_constitution"] == cfg["max_len"] == 8192
    # the safety arm is sized far below the shared budget (the opt. 1 win)
    assert cfg["max_len_by_si"]["safety_refusal"] < cfg["max_len"]
    # completion cap stays a single shared value (the invariant that must NOT
    # become per-SI)
    assert "completion_tokens" in cfg and "completion_tokens_by_si" not in cfg
    # batch widths are MEASURED, never invented (capeval-controls §5.3):
    # by_si stays EMPTY (by_arm wins under resolve_batch_size precedence, so
    # the min-over-SIs projection is the only safe per-arm value) and by_arm
    # equals EXACTLY that projection from the committed measured table.
    assert cfg["batch_size_by_si"] == {}
    import json

    measured = json.loads(
        (TWO_SI_CONFIG.parent / "two_si_budget_measured.json").read_text())
    safe = measured["safe_batch_size"]
    # the projection spans the MEASURED SIs (the two stage-2 SIs); the
    # stage-3 si_irrelevant is unmeasured and shares si_constitution's 8192
    # max_len bucket, so the min over the measured pair stays the committed
    # conservative value (re-measured before the full matrix, per the spec)
    measured_arms = set.intersection(*(set(v) for v in safe.values()))
    projection = {arm: min(safe[si][arm] for si in safe)
                  for arm in cfg["batch_size_by_arm"] if arm in measured_arms}
    assert {a: v for a, v in cfg["batch_size_by_arm"].items()
            if a in measured_arms} == projection
    # vector, bidpo_lora and token_faithful (2026-08-21) POSTDATE the 156964
    # measurement run, so their entries are documented PROJECTIONS, not
    # measurements (re-measured with the other unmeasured cells before the
    # full matrix):
    # - vector, by domination: pinned to vector_all's projected min — the
    #   same lockstep member with the same teacher+student forwards and
    #   strictly MORE trainable state (~n_layers x the vectors and hooks) —
    #   so a re-measured vector_all row moves this bound with it. The entry
    #   exists so the lockstep group's min-width is not dragged to the
    #   module default by an unmeasured member.
    # - bidpo_lora at the conservative module default: a direction="both"
    #   DPO step holds four differentiable graphs before its backward, so
    #   lora_r1's measured width (one KL graph per step) does not transfer —
    #   2 is the width the same objective runs at on the vector substrate
    #   (bidpo, via this same default).
    # - token_faithful, by domination: pinned to token_bos's projected min —
    #   the same memory shape (one trainable embedding vector carried by one
    #   injected sentinel token) through the same train_object_kl loop; its
    #   T=10 / loss_scale=100 are scalar multipliers that allocate nothing.
    #   The entry exists for the same lockstep min-width reason as vector's.
    # - lora_r1_all at 16 (2026-08-25, H200) is a DIFFERENT category from
    #   the three above and is marked as such: those are projections by
    #   DOMINATION, a principled bound from a measured sibling. This one is
    #   ASSERTED from a partial measurement -- the LoRA lane measured bs 16
    #   at 110.4 GiB against the 0.85 guard's 120.2, but only on
    #   safety_refusal at max_len 2048, never at the 8192 the other two SIs
    #   run. It is the one width in the config with no measured value at
    #   the sequence length it will be used at, so if an 8192 cell OOMs this
    #   is the first row to suspect.
    # - diffmean (2026-08-21, accel-fit): MEASURED, but in its own
    #   capture-width run, not the 156964 table this projection is built
    #   from — so it sits outside measured_arms while still being a
    #   considered width, 16, not the module-default fallback (the pin
    #   test_accel_config_pins guards against losing).
    # - lora_r4 (2026-08-28): the canonical spelling of lora_r1_all
    #   (LEGACY_ARM_ALIASES) — same trainer, same substrate, so it inherits
    #   the sibling's asserted width verbatim; the alias sits in the same
    #   asserted-not-measured category and shares its caveat.
    from sisteer.eval.train_budget import DEFAULT_BATCH_SIZE
    assert set(cfg["batch_size_by_arm"]) - measured_arms == {
        "vector", "bidpo_lora", "token_faithful", "lora_r1_all", "lora_r4",
        "diffmean"}
    assert (cfg["batch_size_by_arm"]["lora_r1_all"] == 16
            == cfg["batch_size_by_arm"]["lora_r4"])
    assert cfg["batch_size_by_arm"]["diffmean"] == 16
    assert (cfg["batch_size_by_arm"]["vector"]
            == projection["vector_all"]
            == cfg["batch_size_by_arm"]["vector_all"])
    assert cfg["batch_size_by_arm"]["bidpo_lora"] == DEFAULT_BATCH_SIZE
    assert (cfg["batch_size_by_arm"]["token_faithful"]
            == projection["token_bos"]
            == cfg["batch_size_by_arm"]["token_bos"])
    # the fisher entry is documentary — the code cap clamps regardless
    assert cfg["batch_size_by_arm"]["alphakl_fisher"] <= FISHER_MAX_BATCH
    # every per-SI max_len still resolves via the module resolver
    for si in cfg["si_ids"]:
        assert resolve_max_len(cfg, si) == cfg["max_len_by_si"][si]
