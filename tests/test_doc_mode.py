"""Document-mode data path (sdf-base-steering §4): the Pair-shaped doc
pool rows through _load_pairs, the verbatim docs_to_ce_examples builder
(template-shipping tokenizers included), the doc-mode token-sidecar branch
(prewarm => 100% cache hits at encode), the heldout_pool/holdout_frac
guard through the driver, and the train_kwargs_by_si merge precedence
(default < by_si < train_kwargs.<arm>).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_two_si_objects as t2

from sisteer.data.instructions import BY_ID
from sisteer.data.pairs import (Pair, docs_to_ce_examples,
                                pairs_to_distill_examples, save_pairs)
from sisteer.extraction import tokcache
from sisteer.extraction.distill import _encode_aligned

SI = "constv2-mix-1k"
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"

#: The pool-row shape: Pair-shaped, extra keys deliberately present.
DOC_ROWS = [
    {"prompt": "System scaffold text\n\n", "pos": "Advice body one.",
     "neg": "", "instruction_id": SI, "doc_id": "adv-p00-t02-s00",
     "family": "advice", "official_tokens": 2588, "truncated": False},
    {"prompt": "", "pos": "Story body with no masked prefix.", "neg": "",
     "instruction_id": SI, "doc_id": "sto-c00-u00-k00", "family": "stories",
     "official_tokens": 812, "truncated": True},
]


def _write_pool(tmp_path, rows=None):
    pool = tmp_path / f"pairs_{SI}.jsonl"
    pool.write_text("".join(json.dumps(r) + "\n" for r in (rows or DOC_ROWS)))
    return pool


# -- pool loading + the verbatim builder ---------------------------------------


def test_load_pairs_consumes_doc_rows_extra_keys_ignored(tmp_path):
    """The doc pools are DELIBERATELY Pair-shaped so _load_pairs consumes
    them unchanged: extra keys (doc_id/family/official_tokens/truncated)
    are ignored, the Pair fields land verbatim."""
    pool = _write_pool(tmp_path)
    pairs = t2._load_pairs(str(pool))
    assert len(pairs) == 2
    for row, p in zip(DOC_ROWS, pairs):
        assert p.prompt == row["prompt"]
        assert p.pos == row["pos"]
        assert p.neg == ""
        assert p.instruction_id == SI


def test_docs_to_ce_examples_is_verbatim_and_the_pairs_path_is_not():
    """prompt_with_si == prompt_without == p.prompt, completion == p.pos —
    byte-for-byte, even when the tokenizer SHIPS a chat template — while
    pairs_to_distill_examples on the same tokenizer chat-renders (the
    exact behavior the doc path must bypass)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TINY)
    tok.chat_template = ("{% for m in messages %}<|{{ m['role'] }}|>"
                         "{{ m['content'] }}{% endfor %}")
    pairs = [Pair(prompt=r["prompt"], pos=r["pos"], neg="",
                  instruction_id=SI) for r in DOC_ROWS]
    exs = docs_to_ce_examples(tok, pairs)
    for p, ex in zip(pairs, exs):
        assert ex.prompt_with_si == p.prompt
        assert ex.prompt_without == p.prompt
        assert ex.completion == p.pos
        assert ex.instruction_idx == 0
    chat = pairs_to_distill_examples(tok, pairs, BY_ID[SI])
    assert chat[0].prompt_without != pairs[0].prompt  # the bypassed render


def test_corpus_cell_pseudo_instructions_registered():
    """All ten data-cell ids resolve (cell naming through the SI slot;
    constv2-mix-{10m,full} are the round-2 extension, constv2ctl2-full the
    r2 register-matched control); their text is inert and never rendered
    by the doc builder."""
    from sisteer.eval.jailbreak import resolve_instruction

    ids = ["constv2-mix-1k", "constv2-mix-10k", "constv2-mix-100k",
           "constv2-mix-1m", "constv2-mix-10m", "constv2-mix-full",
           "constv2-adv-1k", "constv2-adv-100k",
           "constv2ctl-100k", "constv2ctl2-full"]
    for cid in ids:
        instr = resolve_instruction(cid)
        assert instr.category == "corpus_cell"
        assert instr.checker == "external"  # inert — scoring it fails loudly
        assert instr.anti_text is None


# -- doc-mode token sidecar ------------------------------------------------------


def test_doc_mode_sidecar_prewarms_every_encode_key(tmp_path):
    """build_token_sidecar under data_mode: documents prewarms exactly the
    (prompt, True) + (completion, False) keys the fit encodes — warming
    from the sidecar then encoding every example produces 100% cache hits
    (zero misses). The pairs-mode builder would have warmed CHAT-RENDERED
    strings instead, missing on every doc-mode encode."""
    from sisteer.models import load_tokenizer

    tok = load_tokenizer(TINY)
    pool = _write_pool(tmp_path)
    cfg = {"model": TINY, "data_mode": "documents"}
    sidecar = t2.build_token_sidecar(cfg, SI, str(pool),
                                     out=str(pool) + ".tokcache.npz",
                                     tok=tok)
    cache = tokcache.warm_from_sidecar(sidecar, tok, expect_model_id=TINY,
                                       expect_pool_path=str(pool))
    prev = tokcache.active()
    tokcache.set_active(cache)
    try:
        exs = docs_to_ce_examples(tok, t2._load_pairs(str(pool)))
        before = cache.misses
        for ex in exs:
            _encode_aligned(tok, ex, 256)
        assert cache.misses == before, (
            "doc-mode encode MISSED the prewarmed sidecar — the sidecar "
            "was built over different strings (chat-rendered?)")
        assert cache.hits > 0
    finally:
        tokcache.set_active(prev)


# -- heldout_pool guard through the driver ---------------------------------------


def test_heldout_pool_with_holdout_frac_refused_via_driver(tiny_llama,
                                                           tmp_path):
    model, tok = tiny_llama
    hp = tmp_path / "heldout.jsonl"
    save_pairs([Pair(prompt="", pos="Held doc.", neg="",
                     instruction_id=SI)], hp)
    cfg = {
        "layer": 1, "max_len": 256, "data_mode": "documents",
        "model": TINY,
        "train_kwargs": {
            "default": {"epochs": 1, "holdout_frac": 0.1,
                        "heldout_pool": str(hp)},
            "lora_r4_ce": {"lr": 1e-3, "floor_ratio_max": None},
        },
    }
    pairs = [Pair(prompt="", pos="Doc.", neg="", instruction_id=SI)]
    with pytest.raises(SystemExit, match="mutually exclusive"):
        t2.train_si_objects(model, tok, SI, pairs, ["lora_r4_ce"],
                            cfg=cfg, save_dir=None)


def test_heldout_pool_requires_documents_mode():
    cfg = {"data_mode": "pairs",
           "train_kwargs": {"default": {"holdout_frac": 0.0,
                                        "heldout_pool": "/x/pool.jsonl"}}}
    with pytest.raises(SystemExit, match="data_mode"):
        t2.refuse_ce_misconfiguration(cfg, [])


# -- train_kwargs_by_si merge ------------------------------------------------------


def test_by_si_merges_over_default_and_only_whitelisted_keys():
    cfg = {"train_kwargs": {"default": {"epochs": 2, "seed": 0,
                                        "early_stop_patience": 5}},
           "train_kwargs_by_si": {SI: {"epochs": 60,
                                       "early_stop_patience": 6,
                                       "accum_steps": 4}}}
    tk = t2._train_kwargs_for_si(cfg, SI)
    assert tk["epochs"] == 60
    assert tk["early_stop_patience"] == 6
    assert tk["accum_steps"] == 4
    assert tk["seed"] == 0  # untouched default
    # another SI: pure defaults, byte-identical
    assert t2._train_kwargs_for_si(cfg, "constv2-mix-10k") == {
        "epochs": 2, "seed": 0, "early_stop_patience": 5}
    # unknown per-cell key: refused, never silently dropped
    bad = {"train_kwargs_by_si": {SI: {"lr": 1e-3}}}
    with pytest.raises(SystemExit, match="per-cell caps"):
        t2._train_kwargs_for_si(bad, SI)


def test_by_si_refused_for_lockstep_arms():
    cfg = {"train_kwargs_by_si": {SI: {"epochs": 60}}}
    with pytest.raises(SystemExit, match="LOCKSTEP"):
        t2.refuse_by_si_for_lockstep_arms(cfg, SI, ["psr_kl_all"])
    # solo/CE arms: fine
    t2.refuse_by_si_for_lockstep_arms(cfg, SI, ["psr_ce_all", "psr_mse_all"])


def test_doc_mode_fit_never_chat_renders_anywhere(tiny_llama, tmp_path,
                                                  monkeypatch):
    """The blanket pin for §4's two driver seams (the length-bucket sorter
    and the probe/held-out example builders both used to call
    pairs_to_distill_examples, which chat-renders): poison the pairs-mode
    builder and run a full doc-mode CE fit with length bucketing AND
    monitors ON — it completes only if NO driver seam chat-renders."""
    import sisteer.data.pairs as pairs_mod

    def poisoned(*_a, **_k):
        raise AssertionError(
            "pairs_to_distill_examples called during a doc-mode fit — a "
            "driver seam is chat-rendering document rows")

    monkeypatch.setattr(pairs_mod, "pairs_to_distill_examples", poisoned)
    model, tok = tiny_llama
    hp = tmp_path / "heldout.jsonl"
    save_pairs([Pair(prompt="", pos="Held doc body one.", neg="",
                     instruction_id=SI),
                Pair(prompt="Held sys\n\n", pos="Held doc body two.",
                     neg="", instruction_id=SI)], hp)
    docs = [Pair(prompt="Sys\n\n", pos="Doc body one for the fit.", neg="",
                 instruction_id=SI),
            Pair(prompt="", pos="Doc body two, a little longer than one.",
                 neg="", instruction_id=SI)]
    cfg = {
        "layer": 1, "max_len": 256, "data_mode": "documents",
        "model": TINY,
        "batch_size_by_arm": {"lora_r4_ce": 2},
        "train_kwargs": {
            "default": {"epochs": 1, "seed": 0, "holdout_frac": 0.0,
                        "heldout_pool": str(hp), "probe_n": 2,
                        "length_bucket": True, "monitors": True},
            "lora_r4_ce": {"lr": 1e-3, "floor_ratio_max": None},
        },
    }
    out = t2.train_si_objects(model, tok, SI, docs, ["lora_r4_ce"],
                              cfg=cfg, save_dir=None)
    assert out.ok, {a: (r.error_type, r.error)
                    for a, r in out.failed.items()}


def test_precedence_default_by_si_arm_reaches_the_trainer(
        tiny_llama, tmp_path, monkeypatch):
    """The full precedence chain observed at the trainer: by_si epochs
    override the default, and train_kwargs.<arm> overrides by_si; by_si
    accum_steps reach the CE trainer too."""
    import sisteer.methods as methods_mod

    seen: dict = {}
    real = methods_mod.train_object_ce

    def spy(model_, tok_, obj_, exs_, **kw):
        seen.update({k: kw.get(k) for k in ("epochs", "accum_steps")})
        return real(model_, tok_, obj_, exs_, **kw)

    monkeypatch.setattr(methods_mod, "train_object_ce", spy)
    model, tok = tiny_llama
    hp = tmp_path / "heldout.jsonl"
    save_pairs([Pair(prompt="", pos="Held doc body.", neg="",
                     instruction_id=SI)], hp)
    docs = [Pair(prompt="", pos="Doc body one.", neg="",
                 instruction_id=SI),
            Pair(prompt="Sys\n\n", pos="Doc body two.", neg="",
                 instruction_id=SI)]

    def cfg(arm_epochs=None):
        arm = {"lr": 1e-3, "floor_ratio_max": None}
        if arm_epochs is not None:
            arm["epochs"] = arm_epochs
        return {
            "layer": 1, "max_len": 256, "data_mode": "documents",
            "model": TINY,
            "train_kwargs": {
                "default": {"epochs": 1, "seed": 0, "holdout_frac": 0.0,
                            "heldout_pool": str(hp), "monitors": False},
                "lora_r4_ce": arm,
            },
            "train_kwargs_by_si": {SI: {"epochs": 3, "accum_steps": 2}},
        }

    out = t2.train_si_objects(model, tok, SI, docs, ["lora_r4_ce"],
                              cfg=cfg(), save_dir=None)
    assert out.ok, {a: (r.error_type, r.error) for a, r in
                    out.failed.items()}
    assert seen["epochs"] == 3       # by_si over default
    assert seen["accum_steps"] == 2  # by_si accum threads to the trainer
    out = t2.train_si_objects(model, tok, SI, docs, ["lora_r4_ce"],
                              cfg=cfg(arm_epochs=4), save_dir=None)
    assert out.ok
    assert seen["epochs"] == 4       # train_kwargs.<arm> over by_si
