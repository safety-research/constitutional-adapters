"""build_chat_prompt must fold the system text into the user turn for
templates that reject the system role (Gemma, older Mistral)."""

from sisteer.models import build_chat_prompt


class GemmaLikeTok:
    chat_template = "stub"

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        if any(m["role"] == "system" for m in messages):
            raise ValueError("System role not supported")
        joined = "".join(f"<{m['role']}>{m['content']}</>" for m in messages)
        return joined + "<assistant>"


def test_system_folded_into_user_on_template_error():
    tok = GemmaLikeTok()
    out = build_chat_prompt(tok, "What is 2+2?", system="Always answer in French.")
    assert "Always answer in French." in out
    assert "What is 2+2?" in out
    assert "<user>" in out and "<assistant>" in out


def test_no_system_passthrough():
    tok = GemmaLikeTok()
    out = build_chat_prompt(tok, "Hello")
    assert out == "<user>Hello</><assistant>"


class ThinkingTok:
    """Qwen3.5-like template: emits a <think> block unless rendered with
    enable_thinking=False."""

    chat_template = "... enable_thinking is defined and enable_thinking is false ..."

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kwargs):
        tail = "<assistant>"
        if kwargs.get("enable_thinking", True):
            tail += "<think>"
        return "".join(
            f"<{m['role']}>{m['content']}</>" for m in messages) + tail


def test_template_kwargs_pin_thinking_off():
    tok = ThinkingTok()
    # load_model_and_tokenizer sets this when the template mentions
    # enable_thinking; simulate that here.
    tok._sisteer_template_kwargs = {"enable_thinking": False}
    out = build_chat_prompt(tok, "Hello", system="Answer in French.")
    assert "<think>" not in out
    assert out.endswith("<assistant>")


def test_hidden_size_text_config_fallback():
    from types import SimpleNamespace

    from sisteer.models import hidden_size

    wrapped = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=5376))
    )
    assert hidden_size(wrapped) == 5376


def test_revision_forwarded_to_from_pretrained(monkeypatch):
    """model_revision must reach both the tokenizer and model loads —
    shared caches can have refs/main pointing at an undownloaded
    snapshot, so offline runs need to pin one that exists."""
    import sisteer.models as M

    seen = {}

    class FakeTok:
        pad_token = "<pad>"
        chat_template = None

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def parameters(self):
            return iter([])

    def fake_tok(name, revision=None):
        seen["tok"] = revision
        return FakeTok()

    def fake_model(name, torch_dtype=None, **kw):
        seen["model"] = kw.get("revision")
        return FakeModel()

    monkeypatch.setattr(M.AutoTokenizer, "from_pretrained", fake_tok)
    monkeypatch.setattr(M.AutoModelForCausalLM, "from_pretrained", fake_model)
    M.load_model_and_tokenizer("stub", device="cpu", revision="abc123")
    assert seen == {"tok": "abc123", "model": "abc123"}


def test_decoder_layers_multimodal_wrapper_path():
    import torch.nn as nn

    from sisteer.models import get_decoder_layers

    class TextStack(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Identity() for _ in range(3)])

    class Wrapper(nn.Module):
        """gemma-4 shape: model.language_model.layers."""

        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = TextStack()

    assert len(get_decoder_layers(Wrapper())) == 3


def test_pin_reasoning_template_off_sets_and_reports():
    """The shared pin seam (Qwen3.5 onboarding, 2026-08-30): every render
    site keys on the SAME signal the fit side keyed on, so a reasoning
    template can never render thinking-ON in one place and OFF in
    another — which would MOVE the response-mask boundary (H=7 counts
    the empty think block), not merely add text."""
    from sisteer.models import pin_reasoning_template_off

    tok = ThinkingTok()
    assert pin_reasoning_template_off(tok) is True
    assert tok._sisteer_template_kwargs == {"enable_thinking": False}

    class PlainTok:
        chat_template = "{% for m in messages %}...{% endfor %}"

    plain = PlainTok()
    assert pin_reasoning_template_off(plain) is False
    assert not hasattr(plain, "_sisteer_template_kwargs")


def test_openai_shim_render_honors_thinking_pin():
    """The fp32 reference oracle renders the FIT's regime: _render must
    pass the pinned template kwargs through, or an exact-equality canary
    compares a thinking-ON reference against a thinking-OFF serve —
    different prompts, a guaranteed spurious divergence."""
    from types import SimpleNamespace

    from sisteer.capeval.serving.openai_shim import OpenAIShimEngine

    tok = ThinkingTok()
    tok._sisteer_template_kwargs = {"enable_thinking": False}
    shim = OpenAIShimEngine.__new__(OpenAIShimEngine)  # no engine bring-up
    shim._engine = SimpleNamespace(tokenizer=tok)      # the property's slot
    out = shim._render([{"role": "user", "content": "hi"}])
    assert "<think>" not in out and out.endswith("<assistant>")

    # unpinned tokenizer (llama/gemma): behavior unchanged
    tok2 = ThinkingTok()                           # no pin attribute
    shim._engine = SimpleNamespace(tokenizer=tok2)
    assert shim._render([{"role": "user", "content": "hi"}]).endswith(
        "<assistant><think>")
