"""Toy tokenizer.json fixtures with hand-computed counts + min-selection +
staged-hash verification."""

from __future__ import annotations

import hashlib
import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from corpus_generator.tokens import (
    StagingError,
    TokenCounter,
    official_tokens,
    render_advice_text,
    render_story_text,
)

VOCAB = {"hello": 0, "world": 1, "again": 2, "[UNK]": 3}


def _word_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return tok


def _char_vocab():
    # dedupe first: duplicate letters would collide ids with [UNK]
    chars = {c: i for i, c in enumerate(dict.fromkeys("helo wrdagain"))}
    chars["[UNK]"] = len(chars)
    return chars


def _char_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.WordLevel(_char_vocab(), unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Split("", "removed")
    return tok


def _stage(dest, toks: dict[str, Tokenizer]) -> None:
    manifest = {"tokenizers": {}}
    for name, tok in toks.items():
        d = dest / name
        d.mkdir(parents=True)
        tok.save(str(d / "tokenizer.json"))
        sha = hashlib.sha256((d / "tokenizer.json").read_bytes()).hexdigest()
        manifest["tokenizers"][name] = {
            "source_repo": f"toy/{name}", "snapshot": "abc",
            "files": {"tokenizer.json": sha},
        }
    (dest / "MANIFEST.json").write_text(json.dumps(manifest))


def test_hand_computed_counts_and_min_selection(tmp_path):
    _stage(tmp_path, {"words": _word_tokenizer(), "chars": _char_tokenizer()})
    tc = TokenCounter.load(tmp_path)
    text = "hello world hello"
    counts = tc.counts(text)
    assert counts["words"] == 3  # hand-computed: 3 whitespace tokens
    assert counts["chars"] == 17  # every char incl. spaces
    # official = min across per-tokenizer corpus totals
    assert official_tokens([{"token_counts": counts}]) == 3
    assert official_tokens([{"token_counts": counts}, {"token_counts": counts}]) == 6
    assert official_tokens([]) == 0


def test_add_special_tokens_false(tmp_path):
    # WordLevel adds no specials anyway; assert the count is exactly the words
    _stage(tmp_path, {"words": _word_tokenizer()})
    tc = TokenCounter.load(tmp_path)
    assert tc.counts("hello world again")["words"] == 3


def test_staged_hash_mismatch_refuses(tmp_path):
    _stage(tmp_path, {"words": _word_tokenizer()})
    p = tmp_path / "words" / "tokenizer.json"
    p.write_text(p.read_text().replace("hello", "hellO"))
    with pytest.raises(StagingError, match="sha256 mismatch"):
        TokenCounter.load(tmp_path)


def test_missing_manifest_refuses(tmp_path):
    with pytest.raises(StagingError, match="stage-tokenizers"):
        TokenCounter.load(tmp_path)


def test_renderers():
    adv = {"system": "S", "user": "U", "assistant": "A"}
    assert render_advice_text(adv) == "S\n\nU\n\nA"
    assert render_advice_text({"system": None, "user": "U", "assistant": "A"}) == "U\n\nA"
    assert render_story_text({"text": "T"}) == "T"


def test_real_staged_tokenizers_if_present():
    from corpus_generator.templates import repo_root

    staged = repo_root() / "data" / "tokenizers"
    if not (staged / "MANIFEST.json").exists():
        pytest.skip("real tokenizers not staged")
    tc = TokenCounter.load(staged)
    assert set(tc.tokenizers) == {"llama3", "gemma", "qwen"}
    counts = tc.counts("The quick brown fox jumps over the lazy dog.")
    assert all(5 <= v <= 20 for v in counts.values())
