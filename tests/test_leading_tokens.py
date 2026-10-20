"""Leading special tokens of a template-free encode: the policy
declared per config is applied at tokenizer load and asserted on the encoded
training examples — fake-tokenizer semantics here, plus the REAL pinned
tokenizers when the local HF cache has them (skipped otherwise)."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from sisteer.extraction.leading_tokens import (  # noqa: E402
    LEADING_KEY, POLICIES, apply_leading_special_tokens,
    check_leading_special_tokens, template_free_prefix)

GEMMA_BASE = ("google/gemma-4-E4B", "411aa17b749aa952df1359d2dcea73917a544d9a")
GEMMA_IT = ("google/gemma-4-E4B-it", "ee0ef6023621cff504d758262d4e04895a5af4a2")
QWEN_BASE = ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b")
QWEN_IT = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")


@dataclass
class Seq:
    ids: list
    prompt_len: int


class _FakeTok:
    """A tokenizer whose template-free encode prepends ``prefix``."""

    name_or_path = "fake"

    def __init__(self, prefix, bos_token_id=2, eos_token_id=1):
        self.prefix = list(prefix)
        self.bos_token_id = bos_token_id
        self.bos_token = "<bos>" if bos_token_id is not None else None
        self.eos_token_id = eos_token_id

    def __call__(self, text, add_special_tokens=True):
        ids = [100 + (ord(c) % 50) for c in text]
        return {"input_ids": (self.prefix if add_special_tokens else []) + ids}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_policy_absent_is_a_noop_and_observes_only():
    tok = _FakeTok(prefix=[])
    assert apply_leading_special_tokens(tok, None) is tok
    assert apply_leading_special_tokens(tok, "") is tok
    assert not hasattr(tok, "_sisteer_leading_special_tokens")
    out = check_leading_special_tokens(tok, [Seq([7, 8], 0)], None)
    assert "observed only" in out["line"] and out["n_checked"] == 1


def test_unknown_policy_refuses():
    with pytest.raises(SystemExit, match="unknown leading_special_tokens"):
        apply_leading_special_tokens(_FakeTok([]), "eos")
    with pytest.raises(SystemExit, match="unknown leading_special_tokens"):
        check_leading_special_tokens(_FakeTok([]), [Seq([1], 0)], "eos")
    assert POLICIES == ("bos", "none") and LEADING_KEY == "leading_special_tokens"


def test_bos_policy_noop_when_tokenizer_already_leads_with_bos():
    tok = _FakeTok(prefix=[2])
    assert apply_leading_special_tokens(tok, "bos") is tok
    assert template_free_prefix(tok) == [2]
    assert tok._sisteer_leading_special_tokens == "bos"


def test_bos_policy_refuses_without_a_bos_token_or_with_a_foreign_prefix():
    with pytest.raises(SystemExit, match="has no bos token"):
        apply_leading_special_tokens(_FakeTok([], bos_token_id=None), "bos")
    with pytest.raises(SystemExit, match="neither bare nor BOS-only"):
        apply_leading_special_tokens(_FakeTok([9, 2]), "bos")
    # a bare SLOW tokenizer (no backend to install a post-processor on)
    with pytest.raises(SystemExit, match="not a fast tokenizer"):
        apply_leading_special_tokens(_FakeTok([]), "bos")


def test_none_policy_accepts_bare_and_refuses_a_prefix():
    tok = _FakeTok(prefix=[], bos_token_id=None)
    assert apply_leading_special_tokens(tok, "none") is tok
    with pytest.raises(SystemExit, match="prepends \\[2\\]"):
        apply_leading_special_tokens(_FakeTok([2]), "none")


def test_check_bos_needs_exactly_one_leading_masked_bos():
    tok = _FakeTok(prefix=[2])
    good = [Seq([2, 5, 6, 1], 1), Seq([2, 7, 1], 3)]
    out = check_leading_special_tokens(tok, good, "bos", label="cell: ")
    assert out["first_ids"] == [2, 5, 6, 1] and out["n_checked"] == 2
    assert out["line"].startswith("[leading-tokens] cell: leading_special_tokens=bos")
    for bad, why in ((Seq([5, 6, 1], 0), "missing"),
                     (Seq([2, 2, 5], 1), "double"),
                     (Seq([2, 5, 2, 1], 1), "bos inside completion"),
                     (Seq([2, 5, 1], 0), "bos not in the prompt span")):
        with pytest.raises(SystemExit, match="REFUSING"):
            check_leading_special_tokens(tok, good + [bad], "bos"), why


def test_check_none_refuses_a_leading_bos_or_a_prefixing_tokenizer():
    tok = _FakeTok(prefix=[], bos_token_id=None)
    out = check_leading_special_tokens(tok, [Seq([5, 6, 1], 0)], "none")
    assert out["template_free_prefix"] == [] and out["bos_token_id"] is None
    tok2 = _FakeTok(prefix=[], bos_token_id=2)
    check_leading_special_tokens(tok2, [Seq([5, 6, 1], 0)], "none")
    with pytest.raises(SystemExit, match="REFUSING"):
        check_leading_special_tokens(tok2, [Seq([2, 5, 1], 1)], "none")
    with pytest.raises(SystemExit, match="REFUSING"):
        check_leading_special_tokens(_FakeTok(prefix=[2]), [Seq([5], 0)], "none")


def test_check_refuses_an_empty_example_set():
    with pytest.raises(SystemExit, match="no examples"):
        check_leading_special_tokens(_FakeTok([2]), [], "bos")


# --------------------------------------------------------------------------
# The real pinned tokenizers (local HF cache only; skipped when absent)
# --------------------------------------------------------------------------

def _real_tok(model, rev, policy=None):
    from sisteer.models import load_tokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(model, revision=rev, local_files_only=True)
    except Exception as e:  # noqa: BLE001 — not cached here
        pytest.skip(f"pinned {model} @ {rev[:8]} tokenizer not in the local "
                    f"HF cache ({type(e).__name__})")
    return load_tokenizer(model, revision=rev, leading_special_tokens=policy)


def _pp(tok) -> dict:
    return json.loads(tok.backend_tokenizer.to_str())["post_processor"]


def _doc_seq(tok, prompt: str, completion: str):
    from sisteer.data.pairs import Pair, docs_to_ce_examples
    from sisteer.extraction.distill import _encode_aligned

    pair = Pair(prompt=prompt, pos=completion, neg="", instruction_id="x")
    ex = docs_to_ce_examples(tok, [pair])[0]
    return _encode_aligned(tok, ex, 8192)[1]


def test_real_gemma_it_adds_no_bos_template_free_and_the_bos_policy_fixes_it():
    """The finding behind the policy: gemma-4-E4B-it's post-processor is
    ``$A`` (its chat template supplies <bos> as text), the base's is
    ``<bos> $A``. Under leading_special_tokens=bos the -it tokenizer gets
    the base's post-processor byte for byte, and a doc-CE example then
    leads with exactly one masked BOS — as the base lane's examples do."""
    raw = _real_tok(*GEMMA_IT)
    assert template_free_prefix(raw) == []
    assert raw.bos_token_id == 2
    story_raw = _doc_seq(raw, "", "The terminal in Hollow.")
    assert story_raw.ids[0] != 2 and story_raw.prompt_len == 0  # the defect

    base = _real_tok(*GEMMA_BASE)
    assert template_free_prefix(base) == [2]
    fixed = _real_tok(*GEMMA_IT, policy="bos")
    assert template_free_prefix(fixed) == [2]
    assert _pp(fixed) == _pp(base)  # the installed post-processor IS the base's
    assert fixed._sisteer_leading_special_tokens == "bos"
    for prompt in ("", "Internal HR operations workspace\n\n"):
        s_fixed = _doc_seq(fixed, prompt, "The terminal in Hollow.")
        s_base = _doc_seq(base, prompt, "The terminal in Hollow.")
        assert s_fixed.ids[0] == 2 and s_fixed.ids.count(2) == 1
        assert s_fixed.prompt_len >= 1  # BOS sits in the loss-masked prompt span
        assert s_fixed.prompt_len == s_base.prompt_len
        check_leading_special_tokens(fixed, [s_fixed], "bos")
    # the base tokenizer under the same policy is a no-op
    base_pol = _real_tok(*GEMMA_BASE, policy="bos")
    assert _pp(base_pol) == _pp(base)
    # and the policy the base lane did NOT declare leaves -it untouched
    assert _pp(_real_tok(*GEMMA_IT)) == _pp(raw)


def test_real_qwen_has_no_bos_on_either_checkpoint_so_none_is_the_convention():
    base = _real_tok(*QWEN_BASE)
    it = _real_tok(*QWEN_IT, policy="none")
    for tok in (base, it):
        assert tok.bos_token_id is None
        assert template_free_prefix(tok) == []
    s = _doc_seq(it, "", "The terminal in Hollow.")
    check_leading_special_tokens(it, [s], "none")
    assert s.ids == _doc_seq(base, "", "The terminal in Hollow.").ids[:-1] + [it.eos_token_id]
    with pytest.raises(SystemExit, match="has no bos token"):
        _real_tok(*QWEN_IT, policy="bos")
    with pytest.raises(SystemExit, match="prepends \\[2\\]"):
        _real_tok(*GEMMA_BASE, policy="none")


# --------------------------------------------------------------------------
# The tuning path: tune_arm honours the same policy
# --------------------------------------------------------------------------

def _load_tune_arm():
    import tune_arm  # scripts/ is on sys.path (module top); shared with test_tune_arm

    return tune_arm


def test_tune_arm_setup_threads_the_policy_to_both_tokenizer_loads():
    """memprobe and HPO trials run through tune_arm._setup, whose two
    tokenizer constructions (model-bearing and tokenizer-only) must carry
    the config's policy — the driver's ONE spelling — or a trial ranks
    hyperparameters under a token geometry the fit never trains on, and a
    policy-built sidecar refuses to warm."""
    src = (REPO / "scripts/tune_arm.py").read_text()
    setup_src = src.split("def _setup(", 1)[1].split("\ndef ", 1)[0]
    assert "leading = driver._leading_policy(cfg)" in setup_src
    assert setup_src.count("leading_special_tokens=leading") == 2
    loads = [ln for ln in setup_src.splitlines()
             if "load_model_and_tokenizer(" in ln or "load_tokenizer(" in ln]
    assert len(loads) == 2  # exactly the two constructions, both threaded
    driver_src = (REPO / "scripts/train_two_si_objects.py").read_text()
    assert "def _leading_policy(cfg: dict)" in driver_src


def _ce_setup(tune_arm, tok, cfg_extra, tmp_path, pos_docs):
    """A TuneSetup for CESoloAdapter.prepare_data: doc pairs, a shared
    held-out pool file, a stub driver whose _load_pairs returns it."""
    import types

    from sisteer.data.pairs import Pair

    held_path = tmp_path / "pairs_heldout.jsonl"
    held_pairs = [Pair(prompt="", pos="Held-out story text.", neg="",
                       instruction_id="x")]
    held_path.write_text(json.dumps({"prompt": "", "pos": "Held-out story "
                                     "text.", "neg": "", "instruction_id": "x"})
                         + "\n")
    fit_pairs = [Pair(prompt=p, pos=d, neg="", instruction_id="x")
                 for p, d in pos_docs]
    cfg = {"model": "TESTMODEL", "data_mode": "documents",
           "train_kwargs": {"default": {"holdout_frac": 0.0,
                                        "heldout_pool": str(held_path)},
                            "psr_ce_all": {}},
           **cfg_extra}
    driver = types.SimpleNamespace(_load_pairs=lambda p: held_pairs)
    setup = tune_arm.TuneSetup(
        cfg=cfg, driver=driver, model=None, tok=tok, instruction=None,
        si="constv2-mix-full", fit_pairs=fit_pairs, held_pairs=held_pairs,
        held_ex=[], max_len=512, positions="all", response_header_len=None,
        layer=None)
    return setup, fit_pairs


DOCS = [("", "The terminal in Hollow Wend."),
        ("Internal HR operations workspace\n\n", "Advice body text.")]


def test_ce_solo_adapter_asserts_the_policy_on_its_encoded_examples(tmp_path,
                                                                     capsys):
    """CESoloAdapter runs its own optimizer path, so it asserts the policy
    itself on the encoded fit + held-out examples before any step: a
    tokenizer that already leads with BOS passes under "bos"; the SAME
    tokenizer refuses under "none"; a tokenizer the policy was NOT applied
    to (bare, no BOS — the tuner-path hole) refuses under "bos"; an
    undeclared policy observes and passes."""
    tune_arm = _load_tune_arm()
    ad = tune_arm.CESoloAdapter("psr_ce_all")
    bos_tok, bare_tok = _FakeTok(prefix=[2]), _FakeTok(prefix=[])

    setup, pairs = _ce_setup(tune_arm, bos_tok, {LEADING_KEY: "bos"},
                             tmp_path, DOCS)
    assert ad.prepare_data(setup, pairs, length_sort=False) == 2
    out = capsys.readouterr().out
    assert "[leading-tokens] psr_ce_all: leading_special_tokens=bos" in out
    assert "3 example(s) checked" in out  # 2 fit + 1 held-out
    # the probe-mem path (take_longest, no held-out) is covered too
    assert ad.prepare_data(setup, pairs, take_longest=1) == 1
    assert "1 example(s) checked" in capsys.readouterr().out

    setup, pairs = _ce_setup(tune_arm, bos_tok, {LEADING_KEY: "none"},
                             tmp_path, DOCS)
    with pytest.raises(SystemExit, match="REFUSING"):
        ad.prepare_data(setup, pairs, length_sort=False)

    setup, pairs = _ce_setup(tune_arm, bare_tok, {LEADING_KEY: "bos"},
                             tmp_path, DOCS)
    with pytest.raises(SystemExit, match="REFUSING"):
        ad.prepare_data(setup, pairs, length_sort=False)

    setup, pairs = _ce_setup(tune_arm, bare_tok, {}, tmp_path, DOCS)
    assert ad.prepare_data(setup, pairs, length_sort=False) == 2
    assert "policy observed only" in capsys.readouterr().out
