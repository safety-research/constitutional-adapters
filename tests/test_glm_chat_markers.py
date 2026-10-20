"""GLM-4.6V-Flash: successor-bounded chat markers + the VLM loader seam.

GLM-4.x chat templates close NOTHING: an assistant turn renders as
``<|assistant|>\\n<think></think>\\n{content}`` (thinking pinned off) and
ends bare — the model's EOS set carries the NEXT turn's role marker
(``<|user|>``/``<|observation|>``), so mid-conversation the successor
turn's opening control token is the only bound on assistant content.
``chat_markers`` used to refuse these templates outright ("no control
token closes the assistant turn"); the successor-bounded extension
accepts them ONLY after a second probe proves that appending a follow-up
user turn (a) leaves every token through the content end untouched and
(b) puts a control token immediately at the content end. Everything
weaker keeps the hard refusal — these tests pin both directions.

Also pinned here: the loader seam (``models.auto_model_class``) — GLM-4.6V
(model_type ``glm4v``) has NO causal-LM auto-mapping in transformers 5.14
and loads only under ``AutoModelForImageTextToText``; and the
``response_header_len`` marker fallback for a hypothetical flag-ignoring
GLM mirror (``<|assistant|>`` joined the marker list).

Real-tokenizer tests use the same offline gating as
test_capeval_mask_mirrors: pinned revision from the local cache only,
loud skip naming what is missing otherwise.
"""

import re
import types

import pytest

import sisteer.chat_markers as cm
import sisteer.models as sm
from sisteer.chat_markers import ChatMarkerError

GLM_MODEL = "zai-org/GLM-4.6V-Flash"
GLM_REV = "411bb4d77144a3f03accbf4b780f5acb8b7cde4e"

# measured ground truth on the pinned tokenizer (2026-09-02): the
# generation-prompt header under the thinking-off pin is
# <|assistant|> \n <think> </think> \n
GLM_HEADER_IDS = (151337, 198, 151350, 151351, 198)
GLM_SENTINEL = ("[sMASK]", 151332)


# ---------------------------------------------------------------------------
# fake GLM-shaped tokenizer (offline, deterministic)
# ---------------------------------------------------------------------------

_ROLE_SPECIALS = ["[gMASK]", "<sop>", "<|system|>", "<|user|>",
                  "<|assistant|>", "<|observation|>"]
_THINK_TOKENS = ["<think>", "</think>"]  # added but NOT special (as on GLM)


class FakeGlmTok:
    """Minimal GLM-shaped tokenizer: ``[gMASK]<sop>`` preamble, bare role
    markers, assistant turns opened by ``<|assistant|>\\n<think></think>\\n``
    and closed by NOTHING. Knobs turn it into each refusal case."""

    bos_token = None
    chat_template = "fake-glm-template"

    def __init__(self, role_special=True, glitch_on_extend=False,
                 text_tail="", flag_ignoring=False):
        self._role_special = role_special
        self._glitch_on_extend = glitch_on_extend
        self._text_tail = text_tail
        self._flag_ignoring = flag_ignoring
        self._tok2id: dict = {}
        self._id2tok: dict = {}
        self.added_tokens_decoder = {}
        self.all_special_ids: list = []
        for t in _ROLE_SPECIALS + _THINK_TOKENS:
            tid = self._intern(t)
            self.added_tokens_decoder[tid] = types.SimpleNamespace(
                special=(role_special and t in _ROLE_SPECIALS))
        self._marker_re = re.compile(
            "|".join(re.escape(t) for t in _ROLE_SPECIALS + _THINK_TOKENS))

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

    def decode(self, ids):
        return "".join(self._id2tok[i] for i in ids)

    def __call__(self, text, add_special_tokens=False, **kw):
        return {"input_ids": self._encode(text)}

    def _render(self, messages, add_generation_prompt):
        s = "[gMASK]<sop>"
        if self._glitch_on_extend and len(messages) >= 3:
            # simulate boundary retokenization when the conversation grows
            s += "GLITCH"
        for m in messages:
            if m["role"] == "assistant":
                s += "<|assistant|>\n<think></think>\n" + m["content"]
            else:
                s += f"<|{m['role']}|>\n" + m["content"]
        if messages and messages[-1]["role"] == "assistant":
            s += self._text_tail
        if add_generation_prompt or self._flag_ignoring:
            if messages and messages[-1]["role"] != "assistant":
                s += "<|assistant|>\n<think></think>\n"
        return s

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, **kw):
        s = self._render(list(messages), add_generation_prompt)
        return self._encode(s) if tokenize else s


# ---------------------------------------------------------------------------
# successor-bounded derivation on the fake
# ---------------------------------------------------------------------------


def test_fake_glm_markers_are_successor_bounded():
    tok = FakeGlmTok()
    m = cm.derive_assistant_markers(tok)
    assert m.terminator is None
    assert m.closing == ()
    assert m.glue == 0
    assert tok.decode(list(m.header)) == "<|assistant|>\n<think></think>\n"
    # the header anchors INCLUSIVELY at the role marker (a control token)
    assert m.header[0] == tok._tok2id["<|assistant|>"]


def test_fake_glm_generation_header_matches_located():
    tok = FakeGlmTok()
    m = cm.derive_assistant_markers(tok)
    assert cm.generation_header_ids(tok, m) == m.header


def test_fake_glm_spans_multi_turn():
    tok = FakeGlmTok()
    msgs = [{"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "Four."},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "content": "Six."}]
    full, spans = cm.assistant_content_spans(tok, msgs)
    assert [tok.decode(full[lo:hi]) for lo, hi in spans] == ["Four.", "Six."]
    ctrl = cm.control_token_ids(tok)
    # the mid-conversation span is bounded by the successor's control token
    lo, hi = spans[0]
    assert full[hi] in ctrl
    # the final span runs to the end of the render — nothing follows
    assert spans[1][1] == len(full)
    assert not [p for lo, hi in spans for p in range(lo, hi)
                if full[p] in ctrl]


def test_successor_turn_without_control_token_refuses():
    """Role markers that are NOT control tokens leave no verifiable bound
    on the content — the extension must keep the hard refusal."""
    tok = FakeGlmTok(role_special=False)
    with pytest.raises(ChatMarkerError, match="refusing to guess where"):
        cm.derive_assistant_markers(tok)


def test_successor_probe_retokenization_refuses():
    """If appending the follow-up turn disturbs earlier tokens, the
    successor probe proves nothing — refuse."""
    tok = FakeGlmTok(glitch_on_extend=True)
    with pytest.raises(ChatMarkerError, match="successor turn"):
        cm.derive_assistant_markers(tok)


def test_noncontrol_closing_scaffold_still_refuses():
    """A template that appends plain TEXT after assistant content keeps
    the original refusal — successor-bounding only covers closing == ()."""
    tok = FakeGlmTok(text_tail=" END")
    with pytest.raises(ChatMarkerError,
                       match="no control token closes the assistant turn"):
        cm.derive_assistant_markers(tok)


def test_flag_ignoring_glm_mirror_marker_fallback():
    """response_header_len's marker fallback: a flag-ignoring GLM-style
    mirror (header appended unconditionally, True/False renders identical)
    must derive H from the <|assistant|> marker, not collapse to 0."""
    from sisteer.hooks import response_header_len
    tok = FakeGlmTok(flag_ignoring=True)
    # header = <|assistant|> \n <think> </think> \n  = 5 fake tokens
    assert response_header_len(tok) == 5


def test_assistant_markers_accepts_positional_int_terminator():
    """The Optional[terminator] change must not disturb existing positional
    construction (test_capeval_mask_mirrors builds AssistantMarkers((),
    eot, (eot,)) directly)."""
    m = cm.AssistantMarkers((), 7, (7,))
    assert m.terminator == 7 and m.glue == 0


# ---------------------------------------------------------------------------
# the loader seam
# ---------------------------------------------------------------------------


def _patch_config(monkeypatch, model_type):
    monkeypatch.setattr(
        sm.AutoConfig, "from_pretrained",
        staticmethod(lambda *a, **k: types.SimpleNamespace(
            model_type=model_type)))


def test_auto_model_class_causal_lm(monkeypatch):
    _patch_config(monkeypatch, "llama")
    assert sm.auto_model_class("x").__name__ == "AutoModelForCausalLM"


def test_auto_model_class_vlm_fallback(monkeypatch):
    """glm4v maps ONLY under AutoModelForImageTextToText in transformers
    5.14 — the seam this function exists for."""
    _patch_config(monkeypatch, "glm4v")
    assert sm.auto_model_class("x").__name__ == "AutoModelForImageTextToText"


def test_auto_model_class_unmapped_refuses(monkeypatch):
    _patch_config(monkeypatch, "not-a-model-type")
    with pytest.raises(ValueError, match="refusing to guess a loader"):
        sm.auto_model_class("x")


# ---------------------------------------------------------------------------
# the real pinned tokenizer (offline-gated, same pattern as the mirrors)
# ---------------------------------------------------------------------------


def _pinned_glm_tok():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            GLM_MODEL, revision=GLM_REV, local_files_only=True)
    except Exception as e:  # cache miss, offline resolution failure, ...
        pytest.skip(f"pinned {GLM_MODEL} tokenizer @ {GLM_REV[:8]} not in "
                    f"the local HF cache ({type(e).__name__})")
    # models.load_tokenizer's thinking-off pin, replicated so this test
    # needs no network: the GLM template consults enable_thinking
    assert "enable_thinking" in (tok.chat_template or "")
    tok._sisteer_template_kwargs = {"enable_thinking": False}
    return tok


def test_real_glm_markers_and_header():
    tok = _pinned_glm_tok()
    m = cm.derive_assistant_markers(tok)
    assert m.terminator is None and m.closing == ()
    assert m.header == GLM_HEADER_IDS
    assert cm.generation_header_ids(tok, m) == GLM_HEADER_IDS
    # fit-time width (hooks.response_header_len) and the id derivation
    # must agree — the invariant hf_engine asserts at arm time
    from sisteer.hooks import response_header_len
    assert response_header_len(tok) == len(GLM_HEADER_IDS)


def test_real_glm_spans_multi_turn():
    tok = _pinned_glm_tok()
    msgs = [{"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "Four."},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "content": "Six."}]
    full, spans = cm.assistant_content_spans(tok, msgs)
    assert [tok.decode(full[lo:hi]) for lo, hi in spans] == ["Four.", "Six."]
    ctrl = cm.control_token_ids(tok)
    assert not [p for lo, hi in spans for p in range(lo, hi)
                if full[p] in ctrl]


def test_real_glm_sentinel():
    from sisteer.extraction.objects import find_sentinel
    assert find_sentinel(_pinned_glm_tok()) == GLM_SENTINEL
