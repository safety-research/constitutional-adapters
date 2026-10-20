"""The served capability driver's tokenizer probe measures the render that
is SERVED (scripts/served_capability_run.py ``_load_served_tok``).

Offline, synthetic. The fake below is Qwen3.5-shaped: a ChatML template
that honours ``enable_thinking``. Thinking ON (the template's default) makes
the generation prompt ``<|im_start|>assistant\\n<think>\\n`` while an
assistant turn in the history renders the EMPTY think block
``<think>\\n\\n</think>\\n\\n{content}``; thinking OFF makes the generation
prompt render that same empty block. Whitespace runs are single tokens, so
``\\n`` and ``\\n\\n`` are different ids — which is exactly the measured
shape of the full22 controls-smoke refusal on the pinned Qwen3.5-4B
tokenizer (2026-09-07): ``boundary retokenization at token 10 ([198] vs
[271, 248069, 271])``, id 198 = ``\\n``, 271 = ``\\n\\n``, 248069 =
``</think>``.

The server pins ``enable_thinking=false`` (vllm_server.
THINKING_OFF_TEMPLATE_KWARGS), and so does every HF-side render seam through
``models.pin_reasoning_template_off`` — the fit that derived
``response_header_len`` included. A probe that loads a BARE tokenizer
measures a render nobody serves, and its refusal says nothing about the
served boundary. These tests pin (1) that the bare probe reproduces the
wave's refusal on this template shape, (2) that the driver's loader is the
shared construction (the pin applies, the header is the served H=7, and the
render kwargs are recorded in the evidence), and (3) that a template without
the switch is left alone (no inert kwargs recorded).
"""

from __future__ import annotations

import importlib
import re
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import served_capability_run as srun  # noqa: E402

import sisteer.models as sm  # noqa: E402
from sisteer.chat_markers import ChatMarkerError, generation_header_ids  # noqa: E402

_CONTROL = ["<|im_start|>", "<|im_end|>"]
_THINK = ["<think>", "</think>"]  # added, NOT special (as on Qwen3.5)

FAKE_MODEL = "synthetic-org/fake-qwen3.5"
FAKE_REV = "0" * 40


class FakeQwenTok:
    """Minimal Qwen3.5-shaped tokenizer (see module docstring). ``switch``
    controls whether the template carries ``enable_thinking`` at all — a
    plain (gemma-like) template is modelled by ``switch=False``, where the
    generation prompt is the bare header and no think block exists."""

    bos_token = None
    bos_token_id = None
    pad_token = None
    eos_token = "<|im_end|>"

    def __init__(self, switch: bool = True):
        self._switch = switch
        self.chat_template = ("fake-qwen3.5 {% if enable_thinking %}..{% endif %}"
                              if switch else "fake-plain-chatml template")
        self.name_or_path = FAKE_MODEL
        self._tok2id: dict = {}
        self._id2tok: dict = {}
        self.added_tokens_decoder = {}
        self.all_special_ids: list = []
        for t in _CONTROL + _THINK:
            tid = self._intern(t)
            self.added_tokens_decoder[tid] = types.SimpleNamespace(
                special=(t in _CONTROL))
            if t in _CONTROL:
                self.all_special_ids.append(tid)
        self.eos_token_id = self._tok2id["<|im_end|>"]
        self._marker_re = re.compile(
            "|".join(re.escape(t) for t in _CONTROL + _THINK))

    def _intern(self, piece: str) -> int:
        if piece not in self._tok2id:
            tid = len(self._tok2id)
            self._tok2id[piece] = tid
            self._id2tok[tid] = piece
        return self._tok2id[piece]

    def _encode(self, text: str):
        ids, pos = [], 0
        for m in self._marker_re.finditer(text):
            for piece in re.findall(r"\S+|\s+", text[pos:m.start()]):
                ids.append(self._intern(piece))
            ids.append(self._intern(m.group()))
            pos = m.end()
        for piece in re.findall(r"\S+|\s+", text[pos:]):
            ids.append(self._intern(piece))
        return ids

    def decode(self, ids, **kw):
        return "".join(self._id2tok[i] for i in ids)

    def __call__(self, text, add_special_tokens=False, **kw):
        return {"input_ids": self._encode(text)}

    def _render(self, messages, add_generation_prompt, enable_thinking):
        empty_block = "<think>\n\n</think>\n\n" if self._switch else ""
        s = ""
        for m in messages:
            if m["role"] == "assistant":
                s += ("<|im_start|>assistant\n" + empty_block
                      + m["content"] + "<|im_end|>\n")
            else:
                s += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            s += "<|im_start|>assistant\n"
            if self._switch:
                s += "<think>\n" if enable_thinking else empty_block
        return s

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False,
                            enable_thinking=True, **kw):
        s = self._render(list(messages), add_generation_prompt, enable_thinking)
        return self._encode(s) if tokenize else s


def _fake_auto_tokenizer(monkeypatch, tok):
    """Hand the fake to BOTH constructions: ``sisteer.models.load_tokenizer``
    (module-level ``AutoTokenizer`` import in sisteer.models) and the driver's
    own ``_load_served_tok`` (``from transformers import AutoTokenizer`` at call
    time, then ``pin_reasoning_template_off``)."""
    calls = []

    def from_pretrained(name, revision=None, **kw):
        calls.append((name, revision))
        return tok

    fake = types.SimpleNamespace(from_pretrained=from_pretrained)
    monkeypatch.setattr(sm, "AutoTokenizer", fake)
    # resolved at TEST time, not at this module's import: the driver binds
    # `from transformers import AutoTokenizer` when it is called, against
    # whatever sys.modules holds then
    monkeypatch.setattr(importlib.import_module("transformers"), "AutoTokenizer", fake)
    # the driver loads ONCE per (model, revision) (lru_cache); a fake from an
    # earlier test must not be served from that cache
    srun._load_served_tok.cache_clear()
    return calls


def test_bare_probe_reproduces_the_wave_refusal():
    # the failure as recorded in the full22 smoke: the add_generation_prompt
    # render (thinking ON: `<think>\n`) is not a prefix of an assistant turn
    # (`<think>\n\n</think>\n\n...`); the divergence is `\n` vs `\n\n`
    # chat_markers now diagnoses the CAUSE rather than the symptom: on a
    # template that branches on a thinking switch whose regime is not pinned,
    # the generation prompt stops inside the header (`<think>\n`, 5 tokens)
    # where a completed turn renders the empty block (7 tokens), so the
    # refusal names the two renders and asks for the pin
    tok = FakeQwenTok()
    with pytest.raises(ChatMarkerError, match="render regime is NOT pinned") as ei:
        generation_header_ids(tok)
    msg = str(ei.value)
    assert "stops INSIDE the assistant header" in msg
    assert repr("<|im_start|>assistant\n<think>\n") in msg
    assert repr("<|im_start|>assistant\n<think>\n\n</think>\n\n") in msg
    assert "(5 tokens)" in msg and "(7 tokens)" in msg
    assert "pin_reasoning_template_off" in msg
    # the underlying divergence is still the recorded one: `\n` vs `\n\n` at token 10
    pre = tok.apply_chat_template([{"role": "user", "content": "Hi"}],
                                  tokenize=True, add_generation_prompt=True)
    full = tok.apply_chat_template([{"role": "user", "content": "Hi"},
                                    {"role": "assistant", "content": "x"}],
                                   tokenize=True)
    d = next(i for i, (a, b) in enumerate(zip(pre, full)) if a != b)
    assert tok.decode([pre[d]]) == "\n"
    assert tok.decode([full[d]]) == "\n\n"
    assert d == 10  # the same offset the real tokenizer reported


def test_driver_loader_is_the_shared_pinned_construction(monkeypatch):
    tok = FakeQwenTok()
    calls = _fake_auto_tokenizer(monkeypatch, tok)
    got = srun._load_served_tok(FAKE_MODEL, FAKE_REV)
    assert got is tok
    assert calls == [(FAKE_MODEL, FAKE_REV)]
    # the served-render pin (the ported loader pins the regime and nothing
    # else — a pad default is the fit loader's business, not the probe's)
    assert got._sisteer_template_kwargs == {"enable_thinking": False}
    # and on the pinned render the SAME probe derives the served header —
    # the empty think block is the assistant header (Qwen3.5: H=7)
    ev = srun._tok_evidence(FAKE_MODEL, FAKE_REV)
    assert ev["serve_header_len"] == 7
    assert tok.decode(ev["serve_header_ids"]) == (
        "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert ev["template_kwargs"] == {"enable_thinking": False}
    assert ev["served_template_hash"] == sm.chat_template_hash(tok)


def test_plain_template_gets_no_pin_and_records_no_kwargs(monkeypatch):
    tok = FakeQwenTok(switch=False)
    _fake_auto_tokenizer(monkeypatch, tok)
    ev = srun._tok_evidence(FAKE_MODEL, FAKE_REV)
    assert not hasattr(tok, "_sisteer_template_kwargs")
    assert ev["template_kwargs"] == {}
    assert tok.decode(ev["serve_header_ids"]) == "<|im_start|>assistant\n"
