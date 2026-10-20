"""The serving library's per-family tables for the routed-experts member of the Qwen3.5 family — the RAW HF ``model_type``
``qwen3_5_moe`` (architecture vocabulary: `Qwen3_5MoeForConditionalGeneration` checkpoints read it off their config).

`vllm_server.model_family` returns the HF config's `model_type`, and the serving path reads it RAW off the config
(`capctrl_serving_up.py` / `native_standup.py`) — no alias — so the MoE member needs its OWN rows beside ``qwen3_5``'s:
  * `TEXT_ONLY_MM_TOKENS_BY_FAMILY["qwen3_5_moe"]`: the family's checkpoints are multimodal and serve under the same two
    tokens; without the row a MoE standup that carries no manifest composes nothing under `--text-only-mm auto` and the
    launcher's full22 gate refuses `REC_TEXT_ONLY_MM_UNPINNED`;
  * `TOOL_PARSER_BY_FAMILY["qwen3_5_moe"]`: the family's `qwen3_xml` parser. A ROW is the family's parser, never a model's
    certificate: a model key's BFCL leg still needs its own live-PASS record, and no checkpoint of this family is a paper
    model, so no such record is on this tree.
No model is loaded: a synthetic id's multimodal reading comes from the name marker; the architecture readings come from
transformers' tiny Qwen3_5Moe config saved under a temp dir (the fused-3-D-experts + shared-expert + Gated-DeltaNet shape),
skipped when transformers is absent. None of the four paper models routes experts; the rows stay as an option."""
from __future__ import annotations

import pytest

from sisteer.capeval.serving import standup_manifest as sm
from sisteer.capeval.serving import vllm_server as vs
from sisteer.capeval.tinycap import contracts as C

#: transformers' tiny Qwen3_5Moe TEXT config (tests/test_lora_modules.py's fixture shape)
MOE_TEXT = dict(hidden_size=32, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                intermediate_size=64, moe_intermediate_size=16, shared_expert_intermediate_size=16, num_experts=4,
                num_experts_per_tok=2, vocab_size=128, full_attention_interval=4, linear_num_key_heads=2,
                linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4)
#: a synthetic served id whose NAME carries the family marker (the CPU-test fallback of multimodal_target / model_family)
SYNTHETIC_MOE_ID = "TESTORG/qwen3.5-tiny-moe"


def test_the_text_only_table_carries_the_moes_raw_model_type_and_auto_composes_the_pin():
    assert vs.TEXT_ONLY_MM_TOKENS_BY_FAMILY["qwen3_5_moe"] == vs.TEXT_ONLY_MM_FLAG_TOKENS == vs.TEXT_ONLY_MM_TOKENS_BY_FAMILY["qwen3_5"]
    assert set(vs.TEXT_ONLY_MM_TOKENS_BY_FAMILY) == {"qwen3_5", "qwen3_5_moe"}
    rec = vs.text_only_mm_decision("auto", model_id=SYNTHETIC_MOE_ID, revision=None, family="qwen3_5_moe", cmd=())
    assert rec["multimodal"] is True and rec["family"] == "qwen3_5_moe"
    assert rec["text_only_mm"] is True and rec["source"] == "standup" and rec["tokens"] == list(vs.TEXT_ONLY_MM_FLAG_TOKENS) and rec["reason"] is None
    # a manifest that already carries the pin stays the recorded source (never duplicated)
    man = vs.text_only_mm_decision("auto", model_id=SYNTHETIC_MOE_ID, revision=None, family="qwen3_5_moe",
                                   cmd=("--limit-mm-per-prompt", '{"image":0,"video":0}'))
    assert man["source"] == "manifest" and man["text_only_mm"] is True
    # an UNCERTIFIED multimodal variant still composes nothing under auto and says so by name — unchanged behaviour
    other = vs.text_only_mm_decision("auto", model_id=SYNTHETIC_MOE_ID, revision=None, family="qwen3_5_mamba", cmd=())
    assert other["text_only_mm"] is False and other["tokens"] is None and "no verified text-only tokens" in str(other["reason"])
    # the dense family reads byte-identically to before
    dense = vs.text_only_mm_decision("auto", model_id="TESTORG/qwen3.5-tiny", revision=None, family="qwen3_5", cmd=())
    assert dense["text_only_mm"] is True and dense["tokens"] == rec["tokens"]


def test_the_tool_parser_table_has_the_family_row_and_a_row_is_never_a_models_certificate():
    assert vs.TOOL_PARSER_BY_FAMILY["qwen3_5_moe"] == "qwen3_xml" == vs.TOOL_PARSER_BY_FAMILY["qwen3_5"]
    assert set(C.REASONING_PARSER_BY_FAMILY) <= set(vs.TOOL_PARSER_BY_FAMILY)
    # the live gate is a separate fact from the table: a family with a parser row and no live-PASS record of its own
    # is still refused a bfcl leg when it is a probe family, and gated by its manifest alone otherwise — the row above
    # admits nothing on its own. No MoE checkpoint is a paper model, so this tree carries no such record.
    if "qwen3_5_moe" in C.TOOL_PARSER_PROBE_FAMILIES:
        assert C.tool_parser_live_gate_problem("qwen3_5_moe") is not None
    else:
        assert C.tool_parser_live_gate_problem("qwen3_5_moe") is None and "qwen3_5_moe" not in C.TOOL_PARSER_LIVE_PASS


def test_a_tiny_qwen3_5_moe_config_reads_as_the_moe_family_multimodal_hybrid_and_the_pin_composes(tmp_path):
    transformers = pytest.importorskip("transformers")
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
    cfg = Qwen3_5MoeConfig(text_config=dict(MOE_TEXT))
    assert cfg.model_type == "qwen3_5_moe"
    d = tmp_path / "tiny-moe"
    cfg.save_pretrained(d)
    # the serving-side readings, as capctrl_serving_up makes them off a LOCAL config: the RAW model_type reaches the table,
    # the vision stack makes it multimodal, the linear_attention blocks make it a hybrid, and the pin composes under auto
    assert vs.model_family(str(d), None) == "qwen3_5_moe"
    assert vs.multimodal_target(str(d), None) is True
    assert vs.hybrid_recurrent_model(str(d), None) is True
    rec = vs.text_only_mm_decision("auto", model_id=str(d), revision=None, family=vs.model_family(str(d), None), cmd=())
    assert rec["text_only_mm"] is True and rec["source"] == "standup"
    # the same config sizes the hybrid's per-sequence recurrent state for the serving probe's KV admission (no constant)
    assert sm.gated_deltanet_state_per_seq_bytes(transformers.AutoConfig.from_pretrained(str(d), local_files_only=True)) == 2_688
