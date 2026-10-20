"""``scripts/onboard_model_smoke.py``'s vLLM gate composes on the NATIVE
serving path —
CPU-only, no vllm, no network: a tiny llama saved as a local snapshot is the
target, ``compose_vllm_gate_launch`` is the pure composer.

Pinned: the composed argv names ``vllm_server.NATIVE_WORKER_EXTENSION_CLS``
(= ``native_worker.WORKER_EXTENSION_CLS``) in the ONE extension slot, the env
carries ``VLLM_PLUGINS == ""`` and ``VLLM_LENS_DISABLE == "1"`` and no lens
registry, the banked zero adapter rides ``--lora-modules none__lora0=...``
with ``--enable-lora`` and the o_proj class, the pinned chat template is the
tokenizer's and hashes into the spec, ONE sequence at the gate window, and
nothing on the path — argv, env, the smoke's source — names the lens worker
module; a template-less checkpoint refuses (the gate cannot render a chat
prompt for it); ``check_vllm`` SKIPs where vllm is absent ONLY on the
implicit / additive path — the explicit ``--mode vllm`` tier (stage 02's
serving-venv GPU job) FAILS without vllm and ``main`` exits 1, so a SKIP can
never be recorded as the DONE.smoke.vllm marker.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch

from sisteer.capeval.serving import lora_export as lx
from sisteer.capeval.serving import native_worker as nw
from sisteer.capeval.serving import vllm_server as vs

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import onboard_model_smoke as oms  # noqa: E402
from derive_model_config import ModelSpec  # noqa: E402

REV = "0123456789abcdef0123456789abcdef01234567"
TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>"
            "{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}")


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast

    cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=97,
                      max_position_embeddings=64, tie_word_embeddings=False)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    d = tmp_path_factory.mktemp("tiny-llama-snapshot")
    model.save_pretrained(d)
    return d


class _Tok:
    """A stand-in tokenizer: only the chat template is read by the composer."""

    def __init__(self, template):
        self.chat_template = template
        self._sisteer_template_kwargs = None


def _spec(model: Path) -> ModelSpec:
    return ModelSpec(slug="tiny_llama", model=str(model), revision=None,
                     n_layers=2, hidden_size=32)


def test_compose_vllm_gate_launch_is_the_native_lora_launch(tiny_base, tmp_path):
    bank = tmp_path / "bank"
    launch = oms.compose_vllm_gate_launch(
        _spec(tiny_base), bank=bank, port=8865, tok=_Tok(TEMPLATE),
        target_revision=REV, launch_flags=["--limit-mm-per-prompt", '{"image":0,"video":0}'])
    cmd = list(launch.cmd)
    assert cmd[:3] == ["vllm", "serve", str(tiny_base)]
    # the NATIVE extension in the one slot, never the lens worker; the auth middleware beside it
    assert cmd.count("--worker-extension-cls") == 1
    assert cmd[cmd.index("--worker-extension-cls") + 1] == vs.NATIVE_WORKER_EXTENSION_CLS \
        == nw.WORKER_EXTENSION_CLS
    assert cmd[cmd.index("--middleware") + 1] == vs.AUTH_MIDDLEWARE_CLS
    joined = " ".join(cmd) + " " + json.dumps(dict(launch.env))
    assert "lens_worker" not in joined and "vllm_lens" not in joined.lower().replace("vllm_lens_disable", "")
    # the two plugin guards, dev mode, the sisteer tree — and no registry
    assert launch.env["VLLM_PLUGINS"] == "" and launch.env[vs.LENS_DISABLE_ENV] == "1"
    assert launch.env[vs.SERVER_DEV_MODE_ENV] == "1"
    assert (Path(launch.env["PYTHONPATH"]) / "sisteer" / "__init__.py").is_file()
    assert vs.LENS_REGISTRY_DIR_ENV not in launch.env and launch.lens_registry_dir is None
    assert launch.lora_native is True and "--no-enforce-eager" in cmd and "--enforce-eager" not in cmd
    # the banked zero adapter is the one registered module, o_proj class, rank 1
    assert launch.lora_modules == (lx.ZERO_ADAPTER_NAME,)
    zdir = Path(launch.lora_adapter_dirs[lx.ZERO_ADAPTER_NAME])
    assert zdir.is_dir() and zdir.parent == bank.resolve()
    lx.verify_zero_adapter(zdir)
    assert launch.lora_target_modules == ("o_proj",)
    assert cmd[cmd.index("--lora-modules") + 1] == f"{lx.ZERO_ADAPTER_NAME}={zdir}"
    assert "--enable-lora" in cmd and launch.max_loras == 1 and launch.max_lora_rank >= 1
    # ONE sequence at the gate window; the no-chunking identity
    assert (launch.max_num_seqs, launch.max_model_len) == (1, oms.VLLM_GATE_MAX_MODEL_LEN)
    assert launch.max_num_batched_tokens == oms.VLLM_GATE_MAX_MODEL_LEN
    # the pinned template IS the tokenizer's, written under the bank and hashed into the spec
    tpath = bank / oms.VLLM_GATE_ADAPTER_BANK / "chat_template.jinja"
    assert tpath.read_text() == TEMPLATE
    assert cmd[cmd.index("--chat-template") + 1] == str(tpath)
    # the manifest flag rode through (the text-only multimodal pin)
    assert "--limit-mm-per-prompt" in cmd
    # a dense llama: prefix caching resolves ON; the api token is minted and never on argv
    assert launch.prefix_caching is True and launch.api_key and launch.api_key not in cmd
    assert launch.revision == REV


def test_gate_refuses_a_template_less_checkpoint_and_the_smoke_skips_without_vllm(tiny_base, tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="no chat template"):
        oms.compose_vllm_gate_launch(_spec(tiny_base), bank=tmp_path / "b", port=8866,
                                     tok=_Tok(None), target_revision=REV)
    # check_vllm: SKIP when vllm is not importable (the grading host venv), never a lens import
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    got = oms.check_vllm(_spec(tiny_base))
    assert got[0] == "SKIP" and "vllm not installed" in got[1]


def test_explicit_vllm_mode_fails_without_vllm_and_the_additive_flag_skips(tiny_base, tmp_path, monkeypatch, capsys):
    """``--mode vllm`` is the EXPLICIT tier (stage 02 runs it in the serving
    venv); an absent vllm there is a FAIL and
    ``main`` exits 1 — the orchestration's DONE marker hangs off that exit.
    The additive ``--vllm`` flag (``required=False``) keeps the SKIP, and a
    compose refusal (a template-less checkpoint) is not swallowed under the
    explicit tier either."""
    spec = _spec(tiny_base)
    monkeypatch.setitem(sys.modules, "vllm", None)          # import -> ImportError; find_spec is not consulted
    assert oms._vllm_available() is False
    # the additive path: SKIP, as before
    assert oms.check_vllm(spec)[0] == "SKIP"
    # the explicit tier: a refusal naming the serving venv, never a SKIP
    with pytest.raises(RuntimeError, match="SERVING venv"):
        oms.check_vllm(spec, required=True)
    # main --mode vllm: the one gate row FAILs and the exit status is 1 (before: SKIP row, exit 0, DONE marker written)
    cfg = tmp_path / "tiny_fits.yaml"
    cfg.write_text("model: tiny\n")
    monkeypatch.setattr(oms, "_resolve_spec_from_config", lambda p: spec)
    rc = oms.main(["--model-config", str(cfg), "--mode", "vllm"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "[FAIL] vLLM load + system-role chat" in out and "SERVING venv" in out and "[SKIP]" not in out
    assert "1 failed" in out
    # a compose-time refusal (a template-less checkpoint) is re-raised under the explicit tier and stays a SKIP on the
    # additive one — the prompt render and the multimodal pin are stubbed so the call reaches the composer
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setattr(oms, "_tok", lambda s: _Tok(None))
    monkeypatch.setattr(oms, "vllm_gate_prompt_ids", lambda tok: [1, 2, 3])
    monkeypatch.setattr(oms, "_text_only_mm_flags", lambda s: [])
    with pytest.raises(RuntimeError, match="no chat template"):
        oms.check_vllm(spec, required=True)
    got = oms.check_vllm(spec)
    assert got[0] == "SKIP" and "no chat template" in got[1]


def test_smoke_source_never_names_the_lens_worker():
    src = (REPO / "scripts/onboard_model_smoke.py").read_text()
    assert "lens_worker" not in src
    assert "from vllm import LLM" not in src
    assert "compose_vllm_gate_launch" in src and "lora_native=True" in src
    assert "export_or_reuse_zero" in src and "launch_server" in src
