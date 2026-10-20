"""The sdfbase-llama8b lane's own registration contract:
the two Llama-3.1-8B fit keys (`llama8b_base`, `llama8b_it_doc`), their
per-KEY effective batch 64, the five-cell scope with its caps, the header,
the measured `bos` leading-token policy on both, the derived-builder
admissions, and the generator reproducing the committed bytes — while the
six frozen configs' bytes do not move.

CPU-only; no model, no network. The shared registration pins (FILES,
KEY_OF, IT_BASE_OF, the scoped-key set, EXPECTED_LEADING, the verbatim
fold reuse) live in tests/test_sdfbase_configs.py; this file carries what
is specific to the lane.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for sub in ("scripts", "scripts/sdfbase"):
    if str(REPO / sub) not in sys.path:
        sys.path.insert(0, str(REPO / sub))

from test_sdfbase_configs import (  # noqa: E402
    FROZEN_CONFIG_SHA16, msc, _load_module)

KEYS = ("llama8b_base", "llama8b_it_doc")
FILES = {k: f"sdfbase_{k}_fits.yaml" for k in KEYS}
PINS = {
    "llama8b_base": ("NousResearch/Meta-Llama-3.1-8B",
                     "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"),
    "llama8b_it_doc": ("NousResearch/Meta-Llama-3.1-8B-Instruct",
                       "d10aef7999a2b5ba950ab3974312feeedbfe0b77"),
}
FROZEN_KEYS = ("gemma4e4b_base", "qwen35_4b_base", "glm46vflash_doc",
               "gemma4e4b_it_doc", "qwen35_4b_it_doc",
               "nemotron3nano4b_doc_full", "glm46vflash_doc_full",
               "glm46vflash_doc_full_r2t12")
EXPECTED_SI_IDS = ["constv2-mix-1k", "constv2-mix-1m", "constv2-mix-10m",
                   "constv2-mix-full", "constv2ctl2-full"]
EXPECTED_CAPS = {
    "constv2-mix-1k": {"epochs": 200, "early_stop_patience": 20},
    "constv2-mix-1m": {"epochs": 24, "early_stop_patience": 3},
    "constv2-mix-10m": {"epochs": 8, "early_stop_patience": 2},
    "constv2-mix-full": {"epochs": 16, "early_stop_patience": 2},
    "constv2ctl2-full": {"epochs": 16, "early_stop_patience": 2},
}
#: The three HPO seams _arm_block fills from the folds store's "hpo"
#: section (sentinel until the winner fold; float(winner) after).
HPO_FIELDS = ("lr", "warmup_frac", "max_grad_norm")


@pytest.fixture(scope="module")
def built():
    return msc.build_all()


def _cfg(built, key):
    return yaml.safe_load(built[FILES[key]])


# --------------------------------------------------------------------------
# effective batch 64, per key
# --------------------------------------------------------------------------

def test_effective_batch_per_key():
    assert msc.EFFECTIVE_BATCH == 8                       # the constant is untouched
    assert msc.LLAMA8B_EFFECTIVE_BATCH == 64
    for key in KEYS:
        assert msc.MODELS[key]["effective_batch"] == 64
        assert msc.effective_batch_for(key) == 64
    for key in FROZEN_KEYS:
        assert "effective_batch" not in msc.MODELS[key]   # nothing declared
        assert msc.effective_batch_for(key) == 8          # -> the constant
    # the accumulation arithmetic: ceil(64 / bs) on the lane, ceil(8 / bs) off it
    assert msc._accum_for(1, 64) == 64
    assert msc._accum_for(2, 64) == 32
    assert msc._accum_for(4, 64) == 16
    assert msc._accum_for(3, 64) == 22                    # ceil, never floor
    assert msc._accum_for(1) == 8 and msc._accum_for(2) == 4
    assert msc._accum_for(4) == 2 and msc._accum_for(8) == 1
    # the sentinel NAMES the effective batch it stands in for
    assert msc.ACCUM_SENTINEL_FMT == "PENDING_MEMPROBE_EFF{eff}_ACCUM"
    assert msc.ACCUM_SENTINEL == "PENDING_MEMPROBE_EFF8_ACCUM"
    assert msc._accum_for(None) == msc.ACCUM_SENTINEL     # frozen spelling
    assert msc._accum_for(None, 64) == "PENDING_MEMPROBE_EFF64_ACCUM"
    with pytest.raises(SystemExit, match="< 1"):
        msc._accum_for(0, 64)


def test_effective_batch_for_refuses_a_bad_declaration(monkeypatch):
    monkeypatch.setitem(msc.MODELS, "_tmp", {"effective_batch": "sixty-four"})
    with pytest.raises(SystemExit, match="not an int"):
        msc.effective_batch_for("_tmp")
    monkeypatch.setitem(msc.MODELS, "_tmp", {"effective_batch": 0})
    with pytest.raises(SystemExit, match="< 1"):
        msc.effective_batch_for("_tmp")


def _is_positive_finite_number(v) -> bool:
    """A real, positive, finite int/float — never a bool (True == 1.0
    would otherwise pass an equality against a folded value) and never a
    str (the sentinel)."""
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v) and v > 0)


def test_committed_llama_configs_carry_the_folded_eff64_widths(built):
    """The post-fold invariant: on both arms of both keys
    the committed width and its accumulation are real ints whose product
    is the key's effective batch 64, the width is the folds store's, and
    no MEMPROBE / PENDING_MEMPROBE_EFF64_ACCUM placeholder is left in
    either file.

    The HPO seams are STATE-AWARE (the same shape one level up: a
    pin on the pre-search sentinel would make the winner fold refuse at
    the fold launcher's own test gate mid-run). Per key, the folds
    store's "hpo" section decides which state the committed config must
    be in, and the config must be ENTIRELY in it:
      * no store entry -> lr / warmup_frac / max_grad_norm are the
        sentinel on BOTH arms (pre-search)
      * a store entry  -> all three are positive finite numbers (never a
        bool, never a str) EQUAL to the store's value for that key x arm,
        on BOTH arms (post-search: folded and regenerated)
    Anything mixed fails: a number beside a sentinel, one arm folded and
    the other not, a store entry the config does not carry, a config
    number the store does not back. And in EITHER state the it_doc key's
    three fields equal the base key's on both arms — the search runs once,
    on the base lane, and its winners are reused verbatim."""
    folds = msc.load_folds()
    hpo_by_key = {}
    for key in KEYS:
        cfg, text = _cfg(built, key), built[FILES[key]]
        store = folds["hpo"].get(key, {})
        for arm in msc.CE_ARMS:
            bs = cfg["batch_size_by_arm"][arm]
            accum = cfg["train_kwargs"][arm]["accum_steps"]
            assert type(bs) is int and type(accum) is int, (key, arm)
            assert bs >= 1 and accum >= 1
            assert bs * accum == 64 == msc.effective_batch_for(key)
            assert accum == msc._accum_for(bs, 64)          # ceil(64 / bs)
            assert bs == folds["memprobe"][key][arm]        # the store's width
            got = {k: cfg["train_kwargs"][arm][k] for k in HPO_FIELDS}
            if not store:                                   # pre-search
                assert got == {k: msc.HPO_SENTINEL for k in HPO_FIELDS}, \
                    (key, arm, got)
            else:                                           # folded
                assert arm in store, (key, arm, sorted(store))
                for k in HPO_FIELDS:
                    assert _is_positive_finite_number(got[k]), \
                        (key, arm, k, got[k])
                    assert _is_positive_finite_number(store[arm][k]), \
                        (key, arm, k, store[arm][k])
                    assert got[k] == store[arm][k], \
                        (key, arm, k, got[k], store[arm][k])
            hpo_by_key.setdefault(key, {})[arm] = \
                tuple(got[k] for k in HPO_FIELDS)
        assert "MEMPROBE" not in text
        assert "PENDING_MEMPROBE_EFF64_ACCUM" not in text
        assert "EFF8" not in text
        assert "#   accum_steps = ceil(64 / bs) — effective batch 64." in text
    # the reuse licence: it_doc carries its base key's values, both
    # arms, in either state — one key folded without the other is mixed
    assert hpo_by_key["llama8b_it_doc"] == hpo_by_key["llama8b_base"], \
        hpo_by_key


def test_folded_render_states_the_keys_own_effective_batch(built):
    fold = {k: {"psr_ce_all": 1, "lora_r4_ce": 2} for k in KEYS}
    for key in KEYS:
        cfg = msc.build_config(key, memprobe=fold)
        assert cfg["train_kwargs"]["psr_ce_all"]["accum_steps"] == 64
        assert cfg["train_kwargs"]["lora_r4_ce"]["accum_steps"] == 32
        text = msc.render_config(key, cfg)
        assert "#   accum_steps = ceil(64 / bs) — effective batch 64." in text
        assert "effective batch 8" not in text
    # the frozen folded configs render the constant, byte-identically
    for name in ("sdfbase_gemma4e4b_base_fits.yaml",
                 "sdfbase_qwen35_4b_base_fits.yaml"):
        assert "#   accum_steps = ceil(8 / bs) — effective batch 8." in built[name]


# --------------------------------------------------------------------------
# scope, caps, header, leading tokens, pins
# --------------------------------------------------------------------------

def test_scope_caps_and_pins(built):
    assert msc.LLAMA8B_SI_IDS == EXPECTED_SI_IDS
    assert msc.LLAMA8B_CAPS == EXPECTED_CAPS
    for key in KEYS:
        spec, cfg = msc.MODELS[key], _cfg(built, key)
        assert (spec["model"], spec["revision"]) == PINS[key]
        assert (cfg["model"], cfg["model_revision"]) == PINS[key]
        assert cfg["si_ids"] == EXPECTED_SI_IDS
        assert cfg["train_kwargs_by_si"] == EXPECTED_CAPS
        assert spec["si_ids"] is msc.LLAMA8B_SI_IDS
        assert spec["caps"] is msc.LLAMA8B_CAPS
        # the three working cells sit at exactly the shared CAPS rows; the
        # shared table itself is untouched (the frozen mix-full row is 4 / 1)
        for si in ("constv2-mix-1k", "constv2-mix-1m", "constv2-mix-10m"):
            assert msc.CAPS[si] == EXPECTED_CAPS[si]
        assert msc.CAPS["constv2-mix-full"] == {"epochs": 4,
                                                "early_stop_patience": 1}
        assert cfg["layer"] == 15 == round(0.469 * 32)
        assert spec["gradient_checkpointing"] is False
        assert "pool_key" not in spec                    # own pool dir per key
        assert msc.pool_key_for(key) == key
        assert cfg["train_kwargs"]["default"]["heldout_pool"] == \
            f"/vol/pool/{key}/pairs_constv2-heldout.jsonl"
        assert cfg["training_budget_kind"] == "document_tokens"
        assert cfg["sequence_max_len"] == 8192 and cfg["max_len"] == 8192
        assert cfg["data_mode"] == "documents"
        assert cfg["teacher_prefix_cache"] is False
        assert "gradient_checkpointing" not in cfg["train_kwargs"]["psr_ce_all"]


def test_leading_special_tokens_is_bos_on_both_keys(built):
    for key in KEYS:
        assert msc.MODELS[key]["leading_special_tokens"] == "bos"
        assert msc._leading_policy(key) == "bos"
        assert _cfg(built, key)[msc.LEADING_TOKEN_KEY] == "bos"
    # the frozen base configs declare nothing (their bytes are pinned)
    for name in ("sdfbase_gemma4e4b_base_fits.yaml",
                 "sdfbase_qwen35_4b_base_fits.yaml"):
        assert msc.LEADING_TOKEN_KEY not in yaml.safe_load(built[name])


def test_headers_say_what_each_key_is(built):
    base, it = msc.MODELS["llama8b_base"], msc.MODELS["llama8b_it_doc"]
    assert "chat_model" not in base and it["chat_model"] is True
    assert base["header_kind"].startswith("BASE model, trained at FULL scale directly")
    assert "no frozen 1k-1m parent" in base["header_kind"]
    assert "working cells" in base["header_kind"]
    assert it["header_kind"].startswith("INSTRUCT model, trained DIRECTLY")
    assert "llama8b_base" in it["header_kind"]
    for key in KEYS:
        head = "\n".join(built[FILES[key]].splitlines()[:40])
        assert f"({msc.MODELS[key]['header_kind']} — no chat template is" in head
        assert "CHAT model; no base variant" not in head
        assert "frozen" not in head.lower().replace("no frozen 1k-1m parent", "")
    assert "BASE model — no chat template is" not in built[FILES["llama8b_base"]]


# --------------------------------------------------------------------------
# the derived builders admit both keys
# --------------------------------------------------------------------------

def test_derived_model_keys_admit_both_llama_keys():
    prd = _load_module("_lane_psr_residual_distill",
                       REPO / "scripts/sdfbase/psr_residual_distill.py")
    ls = _load_module("_lane_lora_subtract",
                      REPO / "scripts/sdfbase/lora_subtract.py")
    for key in KEYS:
        assert key in prd.MODEL_KEYS
        assert key in ls.MODEL_KEYS
    # the existing admissions are untouched
    assert prd.MODEL_KEYS[:3] == ("gemma4e4b_base", "qwen35_4b_base",
                                  "nemotron3nano4b_doc")
    assert ls.MODEL_KEYS[:2] == ("gemma4e4b_base", "qwen35_4b_base")
    # ...and so is the LoRA subtraction's DEFAULT sweep: the Llama keys
    # are supported, never swept unless named
    assert ls.DEFAULT_MODEL_KEYS == ("gemma4e4b_base", "qwen35_4b_base")
    assert not (set(KEYS) & set(ls.DEFAULT_MODEL_KEYS))


def test_derive_model_config_specs_and_slugs():
    import derive_model_config as dmc

    for key in KEYS:
        spec = dmc.MODEL_SPECS[key]
        assert (spec.model, spec.revision) == PINS[key]
        assert spec.n_layers == 32 and spec.hidden_size == 4096
        assert key in dmc.SDFBASE_SLUGS
    # appended as an adjacent, ordered pair — not reordered; later lanes
    # (the `_eb64` siblings 2026-09-11, the qwen-base seed replicates
    # 2026-09-15) append AFTER them, never ahead of the pair
    i = dmc.SDFBASE_SLUGS.index(KEYS[0])
    assert dmc.SDFBASE_SLUGS[i:i + 2] == KEYS
    assert i > max(dmc.SDFBASE_SLUGS.index(k) for k in FROZEN_KEYS)
    assert dmc.SDFBASE_SLUGS[i + 2:i + 4] == ("gemma4e4b_it_doc_eb64",
                                              "qwen35_4b_it_doc_eb64")


# --------------------------------------------------------------------------
# bytes: the two new files reproduce; the six frozen files do not move
# --------------------------------------------------------------------------

def test_build_all_reproduces_the_committed_llama_bytes(built):
    for key in KEYS:
        assert built[FILES[key]] == (REPO / "configs" / FILES[key]).read_text()


def test_the_six_frozen_configs_did_not_move(built):
    from test_sdfbase_configs import _fingerprint_holds
    for name, sha16 in FROZEN_CONFIG_SHA16.items():
        # the recorded fingerprint holds for the generator's bytes and the committed bytes alike: exactly, or through the
        # rewrite configs/SCRUB_MAP.tsv certifies
        assert built[name].encode() == (REPO / "configs" / name).read_bytes(), name
        assert _fingerprint_holds(name, (REPO / "configs" / name).read_bytes()), name
    assert len(FROZEN_CONFIG_SHA16) == 5   # the Nemotron ladder lane (_doc) was retired 2026-10-08
    assert not any(k in FROZEN_CONFIG_SHA16 for k in FILES.values())
