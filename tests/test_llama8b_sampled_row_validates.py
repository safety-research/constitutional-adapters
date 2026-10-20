"""The llama8b sampled configs' `readouts.gsm8k` is the roster's OMISSION (`omitted: true, reason: ...` — the
sdfbase-llama8b configs drop GSM8K: inspect/gsm8k_paired200 is the paper's one GSM8K instrument). At de78ab46 the
served driver's readouts check refused those keys and every controls-ladder leg died in minutes; on the merged tree both drivers'
validators pass the readouts sub-block through unread. Pinned here: the six BASE sampled configs (the ones the controls ladder
runs) validate through cell_controls.controls_gen_config AND served_capability_run._gen_cfg, the omission form intact; the
drivers' default injector renders the same omission (one contract, capability_omissions.gsm8k_omission — the overlay the drivers apply after the default-decode resolution); the static rung configs
(_cap8192 / _cap16384) are the STATIC ladder's and are refused by the controls reader BY DESIGN (never fed to it)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from sisteer.eval import cell_controls as cc  # noqa: E402
from sisteer.eval import sampled_decode as sdd  # noqa: E402

TAGS = ("llama8b", "llama8b_it", "llama8b_fullconst")


def _srun():
    import served_capability_run as srun
    return srun


@pytest.mark.parametrize("tag", TAGS)
@pytest.mark.parametrize("trial", (0, 1))
def test_the_llama8b_base_sampled_config_validates_through_both_drivers(tag, trial):
    cfg = yaml.safe_load((REPO / f"configs/sdfbase_full22_{tag}_sampled_t{trial}.yaml").read_text())
    sd = cfg["sampled_decode"]
    assert sd["readouts"]["gsm8k"] == {"omitted": True, "reason": sd["readouts"]["gsm8k"]["reason"]}
    assert "owned by the evaluation lane" in sd["readouts"]["gsm8k"]["reason"] and set(sd["readouts"]) == {"gsm8k", "mmlu"}
    g1 = cc.controls_gen_config(cfg)              # the controls driver's validator (the allowed-set check lives here)
    g2 = _srun()._gen_cfg(cfg)                    # the served driver's config-time reader
    assert int(g1["max_new_tokens"]) == cc.controls_cap(cfg) == 512
    assert g2["max_new_tokens"] == g1["max_new_tokens"] and g2.get("do_sample") is True
    assert (g1["temperature"], g1["top_p"]) == (0.6, 0.9) and "top_k" not in g1
    # the drivers' default injector (the vendor-preset default) renders the SAME omission for this model
    presets = sdd.load_decode_presets()
    # the shared builder (under the sampled configs' VENDOR_CONTRACT) renders a gsm8k block; the drivers apply the lane overlay
    # capability_omissions right after the default-decode resolution, and THAT rendering equals the generator's omission record
    from sisteer.eval import capability_omissions as co
    assert co.readouts_with_omissions(sdd._readouts_for(presets, "llama8b"), "llama8b")["gsm8k"] == sd["readouts"]["gsm8k"]


@pytest.mark.parametrize("tag", TAGS)
def test_the_static_rung_configs_are_not_the_controls_readers_business(tag):
    for esc in ("cap8192", "cap16384"):
        cfg = yaml.safe_load((REPO / f"configs/sdfbase_full22_{tag}_sampled_t0_{esc}.yaml").read_text())
        assert cfg["sampled_decode"]["escalated_from"]["kind"] == "static"
        with pytest.raises(cc.SampledDecodeError, match="controls ladder"):
            cc.controls_gen_config(cfg)
