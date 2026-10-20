"""GLM-4.6V-Flash in the vLLM SERVING lane: model-family registration.

What serving a dense text-only VLM through the lens needs, and what this
branch already had: the shared seams (successor-bounded chat markers,
the VLM auto-class loader, the M-RoPE (3, T) row reduction, the hybrid
metadata groups) are pinned in their own test modules; THIS module pins
the per-family registration on top of them —

- the LoRA export derives GLM's decoder layout (``model.language_model.
  layers``) from the REAL pinned config, weight-free, never from a
  hardcoded llama string;
- the launch composer treats GLM as a dense full-attention target
  (prefix caching ON, no hybrid refusal), pins the thinking-off render
  from the real template, composes the ``glm45`` tool parser ONLY under
  the agent modality and refuses a parser that does not belong to the
  family (the job-232411 class, caught at compose time);
- the HF engine's response-mask arm assertion needs no per-family
  change: the armed width (H=5, ``<|assistant|> \\n <think> </think>
  \\n`` under the thinking-off pin) equals the generation-header ids.

Real-config/tokenizer tests use the same offline gating as
test_glm_chat_markers: pinned revision from the local HF cache only, a
loud skip naming what is missing otherwise. No weights, no GPU.
"""

from __future__ import annotations

import pytest
import torch

from sisteer.capeval.serving import lora_export as lx
from lora_snapshot_fixtures import snapshot_like

GLM_MODEL = "zai-org/GLM-4.6V-Flash"
GLM_REV = "411bb4d77144a3f03accbf4b780f5acb8b7cde4e"
#: the HF decoder layout of Glm4vForConditionalGeneration (one of
#: models._LAYER_PATHS): the text stack sits under ``language_model``.
GLM_DECODER_PREFIX = "model.language_model.layers"
#: measured on the pinned tokenizer (test_glm_chat_markers): the
#: generation-prompt header under the thinking-off pin.
GLM_HEADER_IDS = (151337, 198, 151350, 151351, 198)


def _glm_config():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoConfig.from_pretrained(
            GLM_MODEL, revision=GLM_REV, local_files_only=True)
    except Exception as e:  # noqa: BLE001 — any cache miss skips loudly
        pytest.skip(f"pinned {GLM_MODEL} config @ {GLM_REV[:8]} not in "
                    f"the local HF cache ({type(e).__name__}); this test "
                    "needs the real config, not a stand-in")


def _glm_tok():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            GLM_MODEL, revision=GLM_REV, local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"pinned {GLM_MODEL} tokenizer @ {GLM_REV[:8]} not in "
                    f"the local HF cache ({type(e).__name__})")
    from sisteer.models import pin_reasoning_template_off
    assert pin_reasoning_template_off(tok) is True  # template has the switch
    return tok


# --------------------------------------------------------------------------
# LoRA export: the decoder layout comes off the real config
# --------------------------------------------------------------------------


def test_glm_config_is_dense_full_attention_under_text_config():
    """The facts every registration below rests on, read off the pinned
    config: 40 layers x hidden 4096 under text_config; NO layer_types
    (every layer is full attention — not a mamba/GDN hybrid); M-RoPE
    (mrope_section present, so vLLM hands the decoder (3, T) positions,
    which the worker's row reduction already handles for text-only)."""
    cfg = _glm_config()
    assert cfg.model_type == "glm4v"
    txt = cfg.text_config
    assert (txt.num_hidden_layers, txt.hidden_size) == (40, 4096)
    assert (txt.num_attention_heads, txt.num_key_value_heads) == (32, 2)
    assert getattr(txt, "layer_types", None) in (None, [], ())
    assert txt.rope_parameters["mrope_section"] == [8, 12, 12]


def test_glm_decoder_prefix_resolves_from_the_pinned_config():
    """resolve_decoder_prefix builds the weight-free skeleton from the
    cached config and reads the layout off the real module tree."""
    _glm_config()
    assert lx.resolve_decoder_prefix(GLM_MODEL, GLM_REV) == GLM_DECODER_PREFIX


def test_glm_rank1_export_derives_the_wrapped_layout(tmp_path):
    """No stated prefix: the export derives GLM's layout from base_model +
    base_revision and spells the HF path PEFT would — the one the pinned
    vLLM's glm4_1v hf_to_vllm_mapper resolves onto
    ``language_model.model.layers.N.self_attn.o_proj``."""
    import io
    import json

    from safetensors.torch import load_file

    _glm_config()
    g = torch.Generator().manual_seed(11)
    blob = {"kind": "lora", "A": torch.randn(4096, generator=g),
            "B": torch.randn(4096, generator=g), "layer": 15,
            "positions": "all"}
    buf = io.BytesIO()
    torch.save(blob, buf)
    # the names are spelled as the checkpoint spells them; GLM's index
    # (704 keys, 40 o_proj) spells the decoder exactly as the skeleton does
    # (model.language_model.layers), so the exact-match path keeps the names.
    # This cache holds the config only: a snapshot shaped like it stands in.
    adapter = lx.export_genuine_adapter(
        {"kind": "lora", "layer": 15}, buf.getvalue(), tmp_path / "glm",
        base_model=GLM_MODEL, base_revision=GLM_REV,
        snapshot_dir=snapshot_like(GLM_MODEL, tmp_path / "snapshot", revision=GLM_REV))
    tensors = load_file(str(adapter / lx.ADAPTER_WEIGHTS))
    prefix = f"base_model.model.{GLM_DECODER_PREFIX}.15.self_attn.o_proj"
    assert sorted(tensors) == [f"{prefix}.lora_A.weight",
                               f"{prefix}.lora_B.weight"]
    assert tuple(tensors[f"{prefix}.lora_A.weight"].shape) == (1, 4096)
    assert tuple(tensors[f"{prefix}.lora_B.weight"].shape) == (4096, 1)
    cfg = json.loads((adapter / lx.ADAPTER_CONFIG).read_text())
    assert cfg["petri_decoder_prefix"] == GLM_DECODER_PREFIX
    assert cfg["base_model_name_or_path"] == GLM_MODEL


# --------------------------------------------------------------------------
# launch composition: dense target, thinking-off render, glm45 under the
# agent modality only, reasoning parser wrapper-owned
# --------------------------------------------------------------------------

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]
#: a synthetic template carrying the reasoning switch (like GLM's real
#: one), for the hermetic composer tests below.
THINK_TEMPLATE_BYTES = (b"{# synthetic glm-shaped template #}"
                        b"{% if enable_thinking %}<think>{% endif %}\n")


def _spec_for(target: str, revision: str, *, template_bytes: bytes,
              tool_parser: str = "glm45",
              flags=("--enforce-eager",)):
    """The audit manifest re-targeted, exactly the way the served-wave
    driver does it (dataclasses.replace over load_petri_spec): only the
    target pin, the template hash, the parser and the flags change."""
    import dataclasses

    from sisteer.capeval import manifest as cm

    spec0 = cm.load_petri_spec(
        REPO / "configs/capeval/manifests/petri_twosi.yaml")
    return dataclasses.replace(
        spec0, target_model=target, target_revision=revision,
        serving_mode="interleaved", cells=["none", "prompted__safety"],
        serving_launch_flags=list(flags),
        scenario_bundle_hash="ab" * 8, serving_impl_commit="fedcba9876" * 4,
        chat_template_hash=cm.sha256_hex16(template_bytes),
        tool_parser=tool_parser)


def _write(tmp_path, name: str, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_model_family_resolution_is_architecture_first_then_name():
    """model_family reads the cached config's model_type (glm4v here when
    the pinned snapshot is cached); for an id whose snapshot is NOT
    cached it falls to the narrow name markers (CPU-test path), and an
    unrelated id is None — unknown, never a wrong family."""
    from sisteer.capeval.serving import vllm_server as vs

    assert vs.model_family(GLM_MODEL, GLM_REV) == "glm4v"
    assert vs.model_family(GLM_MODEL, "0" * 40) == "glm4v"   # name path
    assert vs.model_family("meta-llama/Llama-3.1-8B-Instruct",
                           "0" * 40) == "llama"
    assert vs.model_family("google/gemma-4-E4B-it", "0" * 40) == "gemma4"
    assert vs.model_family("TESTORG/mystery-7b", "0" * 40) is None
    # GLM is dense: no hybrid refusal, by config (when cached) and by name
    assert vs.hybrid_recurrent_model(GLM_MODEL, GLM_REV) is False
    assert vs.hybrid_recurrent_model(GLM_MODEL, "0" * 40) is False


def test_glm_generation_launch_from_the_real_template(tmp_path):
    """The REAL pinned template carries enable_thinking, so the composer
    pins the render OFF (the fits' H=5 regime); GLM is dense, so prefix
    caching composes ON; generation serving composes no tool parser and
    never a reasoning parser."""
    from sisteer.capeval.serving import vllm_server as vs

    tok = _glm_tok()
    template = tok.chat_template.encode("utf-8")
    assert b"enable_thinking" in template
    tpl = _write(tmp_path, "glm_chat_template.jinja", template)
    launch = vs.build_server_launch(
        _spec_for(GLM_MODEL, GLM_REV, template_bytes=template),
        port=8801, chat_template_path=tpl)
    cmd = list(launch.cmd)
    assert cmd[:3] == ["vllm", "serve", GLM_MODEL]
    assert cmd[cmd.index("--revision") + 1] == GLM_REV
    assert (cmd[cmd.index("--default-chat-template-kwargs") + 1]
            == vs.THINKING_OFF_TEMPLATE_KWARGS)
    assert (launch.default_chat_template_kwargs
            == vs.THINKING_OFF_TEMPLATE_KWARGS)
    assert "--enable-prefix-caching" in cmd
    assert "--no-enable-prefix-caching" not in cmd
    assert launch.prefix_caching is True
    assert vs.expected_scheduler_state(launch)["enable_prefix_caching"] is True
    assert "--no-enable-auto-tool-choice" in cmd
    assert "--tool-call-parser" not in cmd
    assert "--reasoning-parser" not in cmd
    assert "--enforce-eager" in cmd and vs.SYNC_SCHEDULING_FLAG in cmd
    assert launch.model == GLM_MODEL
    # the no-chunking identity holds at the lane default geometry
    assert launch.max_num_batched_tokens == launch.max_num_seqs * 16384


def test_glm_agent_launch_composes_glm45_and_refuses_a_foreign_parser(
        tmp_path):
    """Only the agent modality composes a parser, and for GLM it must be
    glm45 (TOOL_PARSER_BY_FAMILY): llama3_json on a GLM target refuses at
    compose time naming the pin — the job-232411 class, caught before any
    generation spend. Symmetrically, glm45 on a llama target refuses;
    an unknown family composes the manifest's parser unchecked."""
    from sisteer.capeval.serving import vllm_server as vs

    tpl = _write(tmp_path, "t.jinja", THINK_TEMPLATE_BYTES)
    agent = vs.build_server_launch(
        _spec_for(GLM_MODEL, "0" * 40, template_bytes=THINK_TEMPLATE_BYTES,
                  tool_parser="glm45"),
        port=8801, chat_template_path=tpl, tool_calls=True)
    cmd = list(agent.cmd)
    assert "--enable-auto-tool-choice" in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == "glm45"
    assert "--reasoning-parser" not in cmd
    assert vs.TOOL_PARSER_BY_FAMILY["glm4v"] == "glm45"

    with pytest.raises(vs.VllmServerError, match="glm45") as err:
        vs.build_server_launch(
            _spec_for(GLM_MODEL, "0" * 40,
                      template_bytes=THINK_TEMPLATE_BYTES,
                      tool_parser="llama3_json"),
            port=8801, chat_template_path=tpl, tool_calls=True)
    assert "232411" in str(err.value) and "glm4v" in str(err.value)
    # the generation modality never composes (and so never checks) it
    gen = vs.build_server_launch(
        _spec_for(GLM_MODEL, "0" * 40, template_bytes=THINK_TEMPLATE_BYTES,
                  tool_parser="llama3_json"),
        port=8801, chat_template_path=tpl)
    assert "--tool-call-parser" not in gen.cmd
    # symmetric: the llama family keeps its own pin
    with pytest.raises(vs.VllmServerError, match="llama3_json"):
        vs.build_server_launch(
            _spec_for("meta-llama/Llama-3.1-8B-Instruct", "0" * 40,
                      template_bytes=THINK_TEMPLATE_BYTES,
                      tool_parser="glm45"),
            port=8801, chat_template_path=tpl, tool_calls=True)
    # unknown family: unverifiable here, composed as pinned (H11 checks it)
    other = vs.build_server_launch(
        _spec_for("TESTORG/mystery-7b", "0" * 40,
                  template_bytes=THINK_TEMPLATE_BYTES, tool_parser="hermes"),
        port=8801, chat_template_path=tpl, tool_calls=True)
    assert list(other.cmd)[list(other.cmd).index("--tool-call-parser") + 1] \
        == "hermes"


def test_reasoning_parser_surface_is_wrapper_owned(tmp_path):
    """Thinking is decided ONCE, on the render (pinned OFF); a manifest
    cannot add a reasoning parser in any accepted spelling."""
    from sisteer.capeval.serving import vllm_server as vs

    tpl = _write(tmp_path, "t.jinja", THINK_TEMPLATE_BYTES)
    for flags in (["--enforce-eager", "--reasoning-parser", "glm45"],
                  ["--reasoning-parser=glm45"],
                  ["--reasoning_parser", "glm45"],
                  ["--reasoning-parser-plugin", "/tmp/x.py"]):
        with pytest.raises(vs.VllmServerError, match="wrapper-owned"):
            vs.build_server_launch(
                _spec_for(GLM_MODEL, "0" * 40,
                          template_bytes=THINK_TEMPLATE_BYTES, flags=flags),
                port=8801, chat_template_path=tpl)
    assert {"--reasoning-parser", "--reasoning-parser-plugin"} \
        <= vs._WRAPPER_OWNED_OPTIONS


# --------------------------------------------------------------------------
# HF engine: the response-mask arm assertion holds on GLM with no
# per-family change (H=5 both ways)
# --------------------------------------------------------------------------


class _MaskableObject:
    """The PositionMaskedFit seam an HF-side psr/lora object exposes."""

    def __init__(self, recorded=None):
        self.fit_response_header_len = recorded
        self.calls = []

    def set_fit_positions(self, positions, header_len):
        self.calls.append((positions, header_len))
        self.fit_response_header_len = header_len


def test_hf_engine_response_mask_arms_h5_on_the_real_glm_tokenizer():
    """hooks.response_header_len (the COUNT the fit-time mask used) and
    chat_markers.generation_header_ids (the ID form the serve boundary
    uses) agree at H=5 on GLM under the thinking-off pin, so the engine
    arms the mask through the real seam; an object fit under a different
    template's header (a Qwen H=7 record) refuses like-with-like."""
    from sisteer.capeval.serving.hf_engine import EngineError, HFEngine
    from sisteer.chat_markers import generation_header_ids
    from sisteer.hooks import response_header_len

    tok = _glm_tok()
    assert tuple(generation_header_ids(tok)) == GLM_HEADER_IDS
    assert response_header_len(tok) == len(GLM_HEADER_IDS) == 5
    eng = HFEngine(GLM_MODEL, revision=GLM_REV, preloaded=(object(), tok))
    obj = _MaskableObject()
    assert eng._arm_response_mask("si_safety__psr_kl_all", obj) == 5
    assert obj.calls == [("response", 5)]
    with pytest.raises(EngineError, match="different chat template"):
        eng._arm_response_mask("si_safety__psr_kl_all",
                               _MaskableObject(recorded=7))


def test_mmlu_parity_locates_the_glm_header_behind_the_answer_prefill():
    """The mmlu reference path locates the header per request (exact id
    subsequence, tiers/parity._rfind_subseq) because "Answer:" follows
    it; on GLM the located header is the 5-id generation header, so the
    effective boundary is header + prefill with no per-family logic."""
    from sisteer.capeval.tiers.parity import _rfind_subseq
    from sisteer.chat_markers import generation_header_ids
    from sisteer.models import apply_chat_template_folding

    tok = _glm_tok()
    header = list(generation_header_ids(tok))
    rendered = apply_chat_template_folding(
        tok, [{"role": "user", "content": "Which letter?\nA. x\nB. y"}],
        add_generation_prompt=True)
    ids = tok(rendered + "Answer:", add_special_tokens=False)["input_ids"]
    start = _rfind_subseq(ids, header)
    assert start >= 0
    assert ids[start:start + len(header)] == header
    assert len(ids) - start > len(header)       # the prefill sits after it


# --------------------------------------------------------------------------
# capctrl (served-MMLU) registration: the CPU preflight's per-model checks,
# mirrored here on the real pinned tokenizer, plus text pins on the two
# scripts (the Modal app is not importable in the CPU venv)
# --------------------------------------------------------------------------


def _synthetic_mmlu_row():
    return {"item_id": "synthetic-0",
            "question": "Which letter comes first in the alphabet?",
            "choices": ["b", "a", "d", "c"]}


def test_capctrl_preflight_seeded_render_parity_needs_the_thinking_pin():
    """What capctrl_served.preflight's mmlu_render_parity predicts for GLM,
    run here on the pinned tokenizer: the server-side seeded render
    (continue_final_message) equals the HF instrument's
    build_chat_prompt + "Answer:" ONLY under the thinking-off kwargs the
    server composes — the raw render lacks '/nothink' on the user turn —
    the 5-id header is located at the same boundary by the served mask's
    rfind and parity's, the seed tokens follow it, and the answer letters
    are single tokens."""
    import sys

    from sisteer.capeval.serving import lens_adapters as la
    from sisteer.capeval.tiers import parity
    from sisteer.chat_markers import generation_header_ids
    from sisteer.eval.capability_ext import _letter_token_ids, _letters
    from sisteer.generate import encode_kwargs

    sys.path.insert(0, str(REPO / "scripts"))
    import served_capability_run as srun

    tok = _glm_tok()
    render_kw = dict(tok._sisteer_template_kwargs)
    assert render_kw == {"enable_thinking": False}
    row = _synthetic_mmlu_row()
    want = parity._mmlu_prompt(tok, row, system=None)
    msgs = [{"role": "user", "content": srun._mmlu_user_text(row)},
            {"role": "assistant", "content": srun.MMLU_ANSWER_SEED}]
    got = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False,
        continue_final_message=True, **render_kw)
    assert got == want
    raw = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False,
        continue_final_message=True)
    assert raw != want and "/nothink" in want and "/nothink" not in raw
    header = [int(t) for t in generation_header_ids(tok)]
    assert header == list(GLM_HEADER_IDS)
    ids = list(tok(want, **encode_kwargs(tok, want))["input_ids"])
    start = la._rfind_header(ids, header)
    assert start >= 0 and start == parity._rfind_subseq(ids, header)
    assert ids[start:start + len(header)] == header
    assert ids[start + len(header):]                 # the "Answer:" seed
    letters = _letter_token_ids(tok, _letters(4))
    assert len(letters) == 4 and len(set(letters)) == 4


def test_capctrl_scripts_register_glm_under_the_pinned_regime():
    """The Modal app is not importable here (no `modal` in the CPU venv),
    so its registration is pinned by text: the MODELS entry carries the
    pinned revision and the derived config path, the preflight renders
    through the thinking-off pin and threads the kwargs into the seeded
    render, and the standup pins the tokenizer it derives the registry's
    assistant_header_ids from."""
    app = (REPO / "scripts/modal/capctrl_served.py").read_text()
    assert '"glm46vflash": dict(hf="zai-org/GLM-4.6V-Flash"' in app
    assert 'rev="411bb4d77144a3f03accbf4b780f5acb8b7cde4e"' in app
    assert 'cfg="configs/two_si_glm46vflash.yaml"' in app
    assert "pin_reasoning_template_off(tok)" in app
    assert "continue_final_message=True, **render_kw)" in app
    assert 'for model in ("llama8b", "gemma4e4b")' not in app  # MODELS-driven
    standup = (REPO / "scripts/capctrl_serving_up.py").read_text()
    assert "pin_reasoning_template_off(tok)" in standup
    # the derived config the app points at exists and pins the same model
    import yaml
    cfg = yaml.safe_load(
        (REPO / "configs/two_si_glm46vflash.yaml").read_text())
    assert cfg["model"] == GLM_MODEL and cfg["model_revision"] == GLM_REV
    # the served-wave vocabulary (tests/test_wave_vocabulary.WAVE_CONFIGS)
    # is NOT this file's home: the generator emits the FIT arm set, so the
    # lora_r4 wave cells cannot resolve from it — the same shape as the
    # generated two_si_gemma4e4b.yaml, whose served twin is the hand-staged
    # _reduced config. A GLM served-wave config follows its fits.
    assert "lora_r4" not in cfg["class_a_arms"]
