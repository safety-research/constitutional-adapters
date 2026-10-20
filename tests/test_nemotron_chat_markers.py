"""Nemotron-3-Nano-4B (NemotronH): ChatML chat markers, the reserved-special
sentinel, the eos trap, and the KV-cache pin.

The template is ChatML with an EMPTY think block under the thinking-off pin:
``<|im_start|>assistant\\n<think></think>{content}<|im_end|>\\n`` — a NORMAL
terminator-anchored template (closing scaffold ``<|im_end|>\\n``, terminator
id 11), NOT the GLM successor-bounded shape. ``<think>``/``</think>`` (ids
12/13) are added tokens flagged special=False, so they sit inside the
located header without tripping the control-token guard. Two facts specific
to this template are pinned as well:

* it renders an EMPTY system turn (``<|im_start|>system\\n<|im_end|>\\n``)
  when no system message is passed — constant scaffold in every "no system
  prompt" render (not the meta-llama date-block injection: no content, but
  every prompt carries it);
* the eos trap: generation_config.json says ``eos_token_id: 2`` (``</s>``)
  while every turn ends with ``<|im_end|>`` (11). The tokenizer's eos IS
  11, so tokenizer-keyed paths were always right; ``models.
  pin_generation_stop_ids`` adds 11 to the generation config at load.

And the architecture trap the same load path closes: transformers 5.14.1's
DynamicCache cannot allocate NemotronH's "mlp" layer type, so cached
forwards raise KeyError('mlp') — ``models.pin_kv_cache_off_if_unallocatable``
turns use_cache off. Fakes pin the mechanics; the real pinned tokenizer /
config files (cache-gated, same offline pattern as test_glm_chat_markers)
pin the measured ids.
"""

import re
import types

import pytest

import sisteer.chat_markers as cm
import sisteer.models as sm
from sisteer.chat_markers import ChatMarkerError

NEMO_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
NEMO_REV = "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"

# Measured ground truth on the pinned tokenizer (2026-09-02). The
# generation-prompt header under the thinking-off pin is
#   <|im_start|> assist ant \n <think> </think>
# (no trailing newline — the template's thinking-off branch ends at
# </think>); the LOCATED header additionally carries the newline that
# follows the previous turn's <|im_end|>.
NEMO_GEN_HEADER_IDS = (10, 1503, 19464, 1010, 12, 13)
NEMO_LOCATED_HEADER_IDS = (1010,) + NEMO_GEN_HEADER_IDS
NEMO_TERMINATOR = 11          # <|im_end|>
NEMO_CLOSING = (11, 1010)     # <|im_end|> \n
NEMO_SENTINEL = ("<SPECIAL_999>", 999)
NEMO_GENERATION_CONFIG_EOS = 2   # </s> — the trap
NEMO_ATTENTION_LAYERS = [12, 17, 24, 32]


# ---------------------------------------------------------------------------
# fake ChatML-with-empty-think tokenizer (offline, deterministic)
# ---------------------------------------------------------------------------

_CONTROL = ["<|im_start|>", "<|im_end|>", "<SPECIAL_999>"]
_THINK = ["<think>", "</think>"]  # added but NOT special (as on Nemotron)


class FakeNemoTok:
    """Minimal Nemotron-shaped tokenizer: ChatML turns, an empty system turn
    when none is given, assistant turns opened by
    ``<|im_start|>assistant\\n<think></think>`` and closed by
    ``<|im_end|>\\n``; no BOS anywhere."""

    bos_token = None
    bos_token_id = None
    chat_template = "fake-nemotron-template enable_thinking"

    def __init__(self):
        self._tok2id: dict = {}
        self._id2tok: dict = {}
        self.added_tokens_decoder = {}
        self.all_special_ids: list = []
        for t in _CONTROL + _THINK:
            tid = self._intern(t)
            self.added_tokens_decoder[tid] = types.SimpleNamespace(
                special=(t in _CONTROL))
        self.eos_token_id = self._tok2id["<|im_end|>"]
        self.eos_token = "<|im_end|>"
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

    def _render(self, messages, add_generation_prompt):
        s = ""
        if not messages or messages[0]["role"] != "system":
            s += "<|im_start|>system\n<|im_end|>\n"
        for m in messages:
            if m["role"] == "assistant":
                s += ("<|im_start|>assistant\n<think></think>"
                      + m["content"] + "<|im_end|>\n")
            else:
                s += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            s += "<|im_start|>assistant\n<think></think>"
        return s

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, **kw):
        s = self._render(list(messages), add_generation_prompt)
        return self._encode(s) if tokenize else s


# ---------------------------------------------------------------------------
# terminator-anchored derivation on the fake (the NORMAL path, not GLM's)
# ---------------------------------------------------------------------------


def test_fake_nemo_markers_are_terminator_anchored():
    tok = FakeNemoTok()
    m = cm.derive_assistant_markers(tok)
    assert m.terminator == tok._tok2id["<|im_end|>"]
    assert m.closing == (tok._tok2id["<|im_end|>"], tok._tok2id["\n"])
    assert m.glue == 0
    # anchored AFTER the previous turn's terminator: the newline that
    # follows <|im_end|> rides in front of the appended header
    assert tok.decode(list(m.header)) == \
        "\n<|im_start|>assistant\n<think></think>"


def test_fake_nemo_generation_header_is_the_located_tail():
    tok = FakeNemoTok()
    m = cm.derive_assistant_markers(tok)
    gen = cm.generation_header_ids(tok, m)
    assert gen == m.header[1:]
    assert tok.decode(list(gen)) == "<|im_start|>assistant\n<think></think>"
    # fit-time width (hooks.response_header_len) == the id derivation
    from sisteer.hooks import response_header_len
    assert response_header_len(tok) == len(gen)


def test_fake_nemo_think_tokens_are_header_not_content():
    """<think>/</think> are added tokens with special=False: NOT control
    ids, so a content span may not be bounded by them — they belong to the
    located header, and the probe must not refuse because of them."""
    tok = FakeNemoTok()
    ctrl = cm.control_token_ids(tok)
    assert tok._tok2id["<think>"] not in ctrl
    assert tok._tok2id["</think>"] not in ctrl
    assert tok._tok2id["<|im_start|>"] in ctrl
    m = cm.derive_assistant_markers(tok)
    assert tok._tok2id["</think>"] == m.header[-1]


def test_fake_nemo_spans_multi_turn():
    tok = FakeNemoTok()
    msgs = [{"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "Four."},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "content": "Six."}]
    full, spans = cm.assistant_content_spans(tok, msgs)
    assert [tok.decode(full[lo:hi]) for lo, hi in spans] == ["Four.", "Six."]
    ctrl = cm.control_token_ids(tok)
    for lo, hi in spans:
        assert full[hi] == tok._tok2id["<|im_end|>"]  # closes at the terminator
    assert not [p for lo, hi in spans for p in range(lo, hi)
                if full[p] in ctrl]


def test_fake_nemo_empty_system_turn_is_constant_scaffold():
    tok = FakeNemoTok()
    ids = cm.render_ids(tok, [{"role": "user", "content": "Hi"}],
                        add_generation_prompt=True)
    assert tok.decode(ids).startswith("<|im_start|>system\n<|im_end|>\n")


# ---------------------------------------------------------------------------
# the two load-path pins, on fakes
# ---------------------------------------------------------------------------


def _fake_model(eos, layer_types=None, text_config=False):
    cfg = types.SimpleNamespace(use_cache=True)
    if layer_types is not None:
        if text_config:
            cfg.text_config = types.SimpleNamespace(layer_types=layer_types,
                                                    use_cache=True)
        else:
            cfg.layer_types = layer_types
    return types.SimpleNamespace(
        config=cfg,
        generation_config=types.SimpleNamespace(eos_token_id=eos,
                                                use_cache=True))


def test_pin_generation_stop_ids_adds_the_template_terminator():
    tok = FakeNemoTok()
    model = _fake_model(eos=2)
    with pytest.warns(RuntimeWarning, match="did not include"):
        added = sm.pin_generation_stop_ids(model, tok)
    assert added == [tok.eos_token_id]
    assert model.generation_config.eos_token_id == [2, tok.eos_token_id]
    # idempotent, and nothing removed
    assert sm.pin_generation_stop_ids(model, tok) == []
    assert model.generation_config.eos_token_id == [2, tok.eos_token_id]


def test_pin_generation_stop_ids_is_inert_when_already_listed():
    tok = FakeNemoTok()
    for have in (tok.eos_token_id, [7, tok.eos_token_id]):
        model = _fake_model(eos=have)
        assert sm.pin_generation_stop_ids(model, tok) == []
        assert model.generation_config.eos_token_id == have  # untouched, same form


def test_pin_generation_stop_ids_without_template_uses_tokenizer_eos():
    tok = FakeNemoTok()
    tok.chat_template = None
    model = _fake_model(eos=None)
    with pytest.warns(RuntimeWarning, match="did not include"):
        assert sm.pin_generation_stop_ids(model, tok) == [tok.eos_token_id]
    assert model.generation_config.eos_token_id == [tok.eos_token_id]


def test_kv_cache_pin_fires_only_on_unmappable_layer_types():
    from transformers.cache_utils import DYNAMIC_LAYER_TYPE_MAPPING

    # The pin is about the cache MAPPING, not about 'mlp' itself: 5.14.1
    # could not map Nemotron's 'mlp' (the trap), 5.15.0 registers it. Both
    # worlds are asserted; an invented type is unmappable under every version.
    nemo_types = ["linear_attention", "mlp", "full_attention", "mlp"]
    nemo = _fake_model(eos=2, layer_types=list(nemo_types))
    if "mlp" in DYNAMIC_LAYER_TYPE_MAPPING:
        assert sm.pin_kv_cache_off_if_unallocatable(nemo) == []
        assert nemo.config.use_cache is True
    else:
        with pytest.warns(RuntimeWarning, match="cannot allocate"):
            assert sm.pin_kv_cache_off_if_unallocatable(nemo) == ["mlp"]
        assert nemo.config.use_cache is False
    bad = _fake_model(eos=2, layer_types=["linear_attention", "no_such_block",
                                          "full_attention", "no_such_block"])
    with pytest.warns(RuntimeWarning, match="cannot allocate"):
        assert sm.pin_kv_cache_off_if_unallocatable(bad) == ["no_such_block"]
    assert bad.config.use_cache is False
    assert bad.generation_config.use_cache is False

    ok = _fake_model(eos=2, layer_types=["linear_attention", "full_attention",
                                         "sliding_attention"])
    assert sm.pin_kv_cache_off_if_unallocatable(ok) == []
    assert ok.config.use_cache is True and ok.generation_config.use_cache is True

    nested = _fake_model(eos=2, layer_types=["no_such_block"], text_config=True)
    with pytest.warns(RuntimeWarning):
        assert sm.pin_kv_cache_off_if_unallocatable(nested) == ["no_such_block"]
    assert nested.config.use_cache is False
    assert nested.config.text_config.use_cache is False

    plain = _fake_model(eos=2)  # no layer_types at all (Llama-style)
    assert sm.pin_kv_cache_off_if_unallocatable(plain) == []


# ---------------------------------------------------------------------------
# the real pinned tokenizer / config files (offline-gated)
# ---------------------------------------------------------------------------


def _pinned_nemo_tok():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            NEMO_MODEL, revision=NEMO_REV, local_files_only=True)
    except Exception as e:  # cache miss, offline resolution failure, ...
        pytest.skip(f"pinned {NEMO_MODEL} tokenizer @ {NEMO_REV[:8]} not in "
                    f"the local HF cache ({type(e).__name__})")
    # models.load_tokenizer's thinking-off pin, replicated so this test
    # needs no network: the template consults enable_thinking
    assert "enable_thinking" in (tok.chat_template or "")
    tok._sisteer_template_kwargs = {"enable_thinking": False}
    return tok


def _pinned_nemo_file(loader, what):
    try:
        return loader(NEMO_MODEL, revision=NEMO_REV, local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"pinned {NEMO_MODEL} {what} @ {NEMO_REV[:8]} not in "
                    f"the local HF cache ({type(e).__name__})")


def test_real_nemo_markers_and_header():
    tok = _pinned_nemo_tok()
    m = cm.derive_assistant_markers(tok)
    assert m.terminator == NEMO_TERMINATOR == tok.eos_token_id
    assert m.closing == NEMO_CLOSING
    assert m.glue == 0
    assert m.header == NEMO_LOCATED_HEADER_IDS
    assert cm.generation_header_ids(tok, m) == NEMO_GEN_HEADER_IDS
    assert tok.decode(list(NEMO_GEN_HEADER_IDS)) == \
        "<|im_start|>assistant\n<think></think>"
    # fit-time width (hooks.response_header_len) and the id derivation
    # must agree — the invariant hf_engine asserts at arm time
    from sisteer.hooks import response_header_len
    assert response_header_len(tok) == len(NEMO_GEN_HEADER_IDS) == 6


def test_real_nemo_control_ids():
    tok = _pinned_nemo_tok()
    ctrl = cm.control_token_ids(tok)
    assert 10 in ctrl and 11 in ctrl          # <|im_start|>, <|im_end|>
    assert 12 not in ctrl and 13 not in ctrl  # <think>, </think>: special=False
    assert 999 in ctrl                        # the sentinel is a control token


def test_real_nemo_spans_multi_turn():
    tok = _pinned_nemo_tok()
    msgs = [{"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "Four."},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "content": "Six."}]
    full, spans = cm.assistant_content_spans(tok, msgs)
    assert [tok.decode(full[lo:hi]) for lo, hi in spans] == ["Four.", "Six."]
    for lo, hi in spans:
        assert full[hi] == NEMO_TERMINATOR
    ctrl = cm.control_token_ids(tok)
    assert not [p for lo, hi in spans for p in range(lo, hi)
                if full[p] in ctrl]


def test_real_nemo_no_bos_and_empty_system_scaffold():
    tok = _pinned_nemo_tok()
    assert tok("", add_special_tokens=True)["input_ids"] == []
    ids = cm.render_ids(tok, [{"role": "user", "content": "Hi"}],
                        add_generation_prompt=True)
    assert tok.bos_token_id not in ids
    assert tok.decode(ids) == ("<|im_start|>system\n<|im_end|>\n"
                               "<|im_start|>user\nHi<|im_end|>\n"
                               "<|im_start|>assistant\n<think></think>")


def test_real_nemo_sentinel():
    from sisteer.extraction.objects import _SENTINEL_PREFS, find_sentinel

    tok = _pinned_nemo_tok()
    assert find_sentinel(tok) == NEMO_SENTINEL
    # the "appending changes nothing for the models above" claim: no
    # earlier preference encodes atomically on this tokenizer
    for cand in _SENTINEL_PREFS[:_SENTINEL_PREFS.index("<SPECIAL_999>")]:
        assert len(tok(cand, add_special_tokens=False)["input_ids"]) > 1, cand


def test_real_nemo_eos_trap_is_closed_by_the_pin():
    from transformers import GenerationConfig

    tok = _pinned_nemo_tok()
    gc = _pinned_nemo_file(GenerationConfig.from_pretrained,
                           "generation_config.json")
    assert gc.eos_token_id == NEMO_GENERATION_CONFIG_EOS  # </s>, never emitted
    assert cm.derive_assistant_markers(tok).terminator == NEMO_TERMINATOR
    model = types.SimpleNamespace(generation_config=gc)
    with pytest.warns(RuntimeWarning, match="did not include"):
        assert sm.pin_generation_stop_ids(model, tok) == [NEMO_TERMINATOR]
    assert set(gc.eos_token_id) == {NEMO_GENERATION_CONFIG_EOS,
                                    NEMO_TERMINATOR}


def test_real_nemo_config_layer_types_and_cache_pin():
    from transformers import AutoConfig

    cfg = _pinned_nemo_file(AutoConfig.from_pretrained, "config.json")
    assert cfg.model_type == "nemotron_h"
    assert cfg.num_hidden_layers == 42 and cfg.hidden_size == 3136
    lt = list(cfg.layer_types)
    assert len(lt) == 42
    assert [i for i, t in enumerate(lt) if t == "full_attention"] == \
        NEMO_ATTENTION_LAYERS
    assert lt.count("mlp") == 17 and lt.count("linear_attention") == 21
    # the depth-fraction layer (round(0.469 x 42) = 20) is an MLP block
    assert lt[20] == "mlp"
    from transformers.cache_utils import DYNAMIC_LAYER_TYPE_MAPPING
    model = types.SimpleNamespace(
        config=cfg, generation_config=types.SimpleNamespace(use_cache=True))
    if "mlp" in DYNAMIC_LAYER_TYPE_MAPPING:       # transformers >= 5.15
        assert sm.kv_cache_unmappable_layer_types(cfg) == []
        assert sm.pin_kv_cache_off_if_unallocatable(model) == []
        assert cfg.use_cache is True
    else:                                          # 5.14.1: the trap
        assert sm.kv_cache_unmappable_layer_types(cfg) == ["mlp"]
        with pytest.warns(RuntimeWarning, match="cannot allocate"):
            assert sm.pin_kv_cache_off_if_unallocatable(model) == ["mlp"]
        assert cfg.use_cache is False
