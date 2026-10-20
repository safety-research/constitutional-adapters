"""standup_manifest.gated_deltanet_state_per_seq_bytes — the per-SEQUENCE recurrent state of a Gated-DeltaNet hybrid, DERIVED from
the checkpoint's HF text config (never a typed per-model constant; the restored MoE serving path's probe prices it through this):
``linear blocks × (SSM v_heads·k_dim·v_dim + conv (kernel−1)·(2·k_heads·k_dim + v_heads·v_dim)) × dtype_bytes``; the block count from
``layer_types`` when stated, else ``n − n // full_attention_interval``; a dense config (no GDN field) is 0; a partial GDN block
refuses by name. Synthetic configs only."""
from __future__ import annotations

import pytest

from sisteer.capeval.serving import standup_manifest as sm

GDN = {"linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 8, "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4}
PER_BLOCK = 4 * 8 * 16 + (4 - 1) * (2 * 2 * 8 + 4 * 16)      # SSM 512 + conv 3 x 96 = 800 elements per linear block


def test_the_state_is_derived_from_the_configs_layer_types_and_dtype():
    cfg = {**GDN, "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"], "num_hidden_layers": 4}
    assert sm.gated_deltanet_state_per_seq_bytes(cfg) == 3 * PER_BLOCK * 2
    assert sm.gated_deltanet_state_per_seq_bytes(cfg, dtype_bytes=4) == 3 * PER_BLOCK * 4       # --mamba-ssm-cache-dtype float32 widens it
    # a multimodal top-level config: the text_config is read
    assert sm.gated_deltanet_state_per_seq_bytes({"text_config": cfg, "architectures": ["X"]}) == 3 * PER_BLOCK * 2


def test_without_layer_types_the_block_count_is_n_minus_the_full_attention_blocks():
    cfg = {**GDN, "num_hidden_layers": 8, "full_attention_interval": 4}
    assert sm.gated_deltanet_state_per_seq_bytes(cfg) == (8 - 8 // 4) * PER_BLOCK * 2


def test_a_dense_config_prices_zero_and_a_partial_gdn_block_refuses_by_name():
    assert sm.gated_deltanet_state_per_seq_bytes({"num_hidden_layers": 32, "hidden_size": 2560, "intermediate_size": 9216}) == 0
    partial = {k: v for k, v in GDN.items() if k != "linear_conv_kernel_dim"}
    with pytest.raises(sm.ManifestError, match="linear_conv_kernel_dim"):
        sm.gated_deltanet_state_per_seq_bytes({**partial, "layer_types": ["linear_attention"]})
    with pytest.raises(sm.ManifestError, match="positive int"):
        sm.gated_deltanet_state_per_seq_bytes({**GDN, "linear_conv_kernel_dim": 0, "layer_types": ["linear_attention"]})
