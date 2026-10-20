"""The vLLM onboarding gate serves the SAME render the served wave feeds.

``onboard_model_smoke.check_vllm`` is the "does the pinned stack serve this
model" probe. It must hand the engine the harness's own render as token
ids (apply_chat_template_folding under load_tokenizer's thinking-off
marker + encode_kwargs — the served wave's exact path), never vLLM's own
chat templating, and the completion it certifies must carry no think block
under the pin and must stop on eos. Measured motivation (GLM-4.6V-Flash on
vllm 0.25.1, cluster job 345699, 2026-09-02): the unpinned ``LLM.chat``
render returned a 16-token ``<think>`` fragment, which the old check
accepted as a served completion.

Since the paper-companion port (FD 19) the gate runs on the NATIVE serving
path: ``compose_vllm_gate_launch`` composes the lora-native server launch
(tests/test_onboard_smoke_native_launch.py pins its argv / env), the ONE
launcher brings it up, and the render goes over ``/v1/completions`` as a
token-id ``prompt``. No vLLM here: the composer, the launcher, the
shutdown and the HTTP round-trip are stubbed, and the stub records the
request body the gate sends; the tokenizer is a stub whose template
appends the thinking-off suffix + empty prefill when the pin is present.
"""

import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import onboard_model_smoke as oms  # noqa: E402
from sisteer.capeval.serving import vllm_server as vs  # noqa: E402

SPEC = oms.MODEL_SPECS["glm46vflash"]
HEADER_OFF = "<|assistant|>\n<think></think>\n"
HEADER_ON = "<|assistant|>\n"


class _StubTok:
    """A reasoning-template stub: renders every message as
    ``<|role|>\\ncontent``, appends ``/nothink`` + the empty-prefill header
    under enable_thinking=False and the thinking-ON header otherwise;
    encodes by whitespace word count so ids are deterministic."""
    bos_token = None
    chat_template = "{% if enable_thinking %}...{% endif %}"

    def __init__(self, pinned: bool):
        if pinned:
            self._sisteer_template_kwargs = {"enable_thinking": False}

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        body = "".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)
        if kw.get("enable_thinking") is False:
            return body + "/nothink" + HEADER_OFF
        return body + HEADER_ON

    def __call__(self, text, **kw):
        self.last_encode_kwargs = kw
        return {"input_ids": [len(w) for w in text.split()]}


class _Launch:
    """The composed launch's surface the gate reads: argv, the loopback
    URL and the per-launch auth header."""
    cmd = ("vllm", "serve", SPEC.model, "--worker-extension-cls",
           vs.NATIVE_WORKER_EXTENSION_CLS)
    base_url = "http://127.0.0.1:8899"
    auth_headers = {"Authorization": "Bearer test-token"}


class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._b


def _install_fake_stack(monkeypatch, *, text, finish_reason="stop", n_tokens=3,
                        seen):
    """A stand-in serving stack: `vllm` importable, the composer / launcher /
    shutdown stubbed, and the HTTP round-trip recording the request body the
    gate POSTs to /v1/completions."""
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))

    def compose(spec, **kw):
        seen["compose"] = kw
        return _Launch()

    monkeypatch.setattr(oms, "compose_vllm_gate_launch", compose)
    monkeypatch.setattr(oms, "_text_only_mm_flags", lambda spec: [])
    monkeypatch.setattr(vs, "launch_server",
                        lambda launch, *, log_path, **kw: seen.setdefault("launched", launch))
    monkeypatch.setattr(vs, "_shutdown_child",
                        lambda proc, *a, **k: seen.setdefault("shutdown", proc))

    import urllib.request

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["headers"] = {k.lower(): v for k, v in req.header_items()}
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _Resp({"choices": [{"text": text, "finish_reason": finish_reason}],
                      "usage": {"completion_tokens": n_tokens}})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_gate_prompt_ids_are_the_harness_render_under_the_pin():
    tok = _StubTok(pinned=True)
    ids = oms.vllm_gate_prompt_ids(tok)
    rendered = tok.apply_chat_template(list(oms.VLLM_GATE_MESSAGES),
                                       enable_thinking=False)
    assert rendered.endswith("/nothink" + HEADER_OFF)
    assert ids == [len(w) for w in rendered.split()]


def test_gate_prompt_ids_plain_template_gets_no_pin():
    tok = _StubTok(pinned=False)
    rendered = tok.apply_chat_template(list(oms.VLLM_GATE_MESSAGES))
    assert rendered.endswith(HEADER_ON) and "/nothink" not in rendered
    assert oms.vllm_gate_prompt_ids(tok) == [len(w) for w in rendered.split()]


def test_check_vllm_serves_token_ids_and_stops(monkeypatch):
    seen = {}
    tok = _StubTok(pinned=True)
    monkeypatch.setattr(oms, "_tok", lambda spec: tok)
    _install_fake_stack(monkeypatch, text="Blue.", seen=seen)
    detail = oms.check_vllm(SPEC)
    # the harness render rides as token ids over /v1/completions (never a chat body),
    # greedy at the gate's cap, against THIS launch's token
    assert seen["url"] == "http://127.0.0.1:8899/v1/completions"
    assert seen["body"] == {"model": SPEC.model, "prompt": oms.vllm_gate_prompt_ids(tok),
                            "max_tokens": oms.VLLM_GATE_MAX_TOKENS, "temperature": 0}
    assert seen["headers"]["authorization"] == "Bearer test-token"
    assert "messages" not in seen["body"]
    # the composed launch is what was brought up and torn down; the composer got the tokenizer
    assert isinstance(seen["launched"], _Launch) and "shutdown" in seen
    assert seen["compose"]["tok"] is tok and seen["compose"]["bank"].name.startswith("onboard-vllm-gate-")
    assert "thinking pinned off" in detail and "stopped on eos" in detail
    assert "NATIVE lora path" in detail and vs.NATIVE_WORKER_EXTENSION_CLS in detail


def test_check_vllm_refuses_a_think_block(monkeypatch):
    seen = {}
    monkeypatch.setattr(oms, "_tok", lambda spec: _StubTok(pinned=True))
    _install_fake_stack(monkeypatch, text="<think>reasoning", finish_reason="length",
                        n_tokens=16, seen=seen)
    with pytest.raises(AssertionError, match="think block"):
        oms.check_vllm(SPEC)
    assert "shutdown" in seen                      # the child is torn down on the refusal path too


def test_check_vllm_refuses_a_capped_completion(monkeypatch):
    seen = {}
    monkeypatch.setattr(oms, "_tok", lambda spec: _StubTok(pinned=False))
    _install_fake_stack(monkeypatch, text="Blue " * 20, finish_reason="length",
                        n_tokens=256, seen=seen)
    with pytest.raises(AssertionError, match="did not stop on eos"):
        oms.check_vllm(SPEC)


def test_check_vllm_refuses_an_empty_completion(monkeypatch):
    seen = {}
    monkeypatch.setattr(oms, "_tok", lambda spec: _StubTok(pinned=False))
    _install_fake_stack(monkeypatch, text="   ", seen=seen)
    with pytest.raises(AssertionError, match="empty completion"):
        oms.check_vllm(SPEC)


def test_check_vllm_skips_without_vllm(monkeypatch):
    monkeypatch.setattr(oms, "_tok", lambda spec: _StubTok(pinned=False))
    monkeypatch.setitem(sys.modules, "vllm", None)   # import -> ImportError
    assert oms.check_vllm(SPEC)[0] == "SKIP"
    assert oms._vllm_available() is False
