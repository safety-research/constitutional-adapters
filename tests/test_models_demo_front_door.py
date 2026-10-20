"""The worked example's DEMO key, qwen35_4b_demo (the paper's Qwen3.5-4B checkpoints under a key of their own), on the
architecture-generic seams — CPU-only, a tiny in-code instance, no weights, no network.

- a tiny ``Qwen3_5ForCausalLM`` (the hybrid: linear attention + full attention): only the full-attention blocks carry an o_proj;
- its ModelSpec pins the paper's base checkpoint sha and ``layer == round(0.469 n)``; it is the LAST SDFBASE_SLUG;
- its GENERATED config is the paper's qwen35_4b_base lane under the demo key, with the plan-§3.7 ``fit_selection`` block per arm
  (the paper's defaults), while every earlier config carries none; the folds API accepts the selection keys, the init-scheme
  override stamps both arms and refuses a byte-pinned lane, a partial winner refuses, a per-expert map on a dense lane refuses;
- the generator's local module / init / per-expert-rank vocabulary equals ``sisteer.extraction.lora_modules``'s;
- ``configs/sdfbase_full22_ports.yaml`` equals the proxy's case table for every tag the proxy names and adds the demo key's tags
  (its one wave + the two dose views) on free ports;
- ``geometry``: the sweep's pinned tables are untouched; the onboarded SERVED key lives in the ONBOARDED tables and
  ``rows_for`` / ``budget_for`` / ``model_key_for`` serve both kinds — the demo key is a HYBRID (the Qwen3.5 gated-deltanet stack),
  has no built-in budget row and sizes from the Llama-certified default with the warning; a provisional onboarded row, when
  one exists, sizes from itself and warns.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import sys
from pathlib import Path

import pytest
import yaml

from sisteer.capeval.serving import geometry as sg
from sisteer.extraction import lora_modules as lm
from sisteer.models import decoder_layers_path, hidden_size, oproj_layers

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from derive_model_config import MODEL_SPECS, SDFBASE_SLUGS  # noqa: E402

DEMO_KEY = "qwen35_4b_demo_base"
DEMO_SHA = "1001bb4d826a52d1f399e183466143f4da7b741b"
SERVED_KEY, SERVED_HF = "qwen35_4b_demo", "Qwen/Qwen3.5-4B"
PAPER_KEY = "qwen35_4b"                      # the paper's key: the same two checkpoints


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def msc():
    return _load_module("_qm_make_sdfbase_configs",
                        REPO / "scripts/sdfbase/make_sdfbase_configs.py")


@pytest.fixture(scope="module")
def tiny_qwen35():
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5 import Qwen3_5Config

    text_cls = type(Qwen3_5Config().get_text_config())
    cfg = text_cls(hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                   num_key_value_heads=2, vocab_size=128, max_position_embeddings=512,
                   layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"])
    m = AutoModelForCausalLM.from_config(cfg)
    m.eval()
    return m


# --------------------------------------------------------------------------- #
# the architecture                                                             #
# --------------------------------------------------------------------------- #


def test_tiny_qwen35_only_the_full_attention_blocks_carry_an_oproj(tiny_qwen35):
    m = tiny_qwen35
    assert decoder_layers_path(m) == "model.layers"
    assert oproj_layers(m) == [3]                       # the hybrid: the o_proj set is the full-attention blocks (models.yaml layer_sets)
    assert hidden_size(m) == 32


# --------------------------------------------------------------------------- #
# the spec and the slugs                                                       #
# --------------------------------------------------------------------------- #


def test_spec_pins_the_cache_resolved_sha_and_the_depth_fraction_layer():
    g = MODEL_SPECS[DEMO_KEY]
    assert (g.model, g.revision, g.n_layers, g.hidden_size) == ("Qwen/Qwen3.5-4B-Base", DEMO_SHA, 32, 2560)
    base = MODEL_SPECS["qwen35_4b_base"]
    assert (base.model, base.revision, base.n_layers, base.hidden_size) == (g.model, g.revision, g.n_layers, g.hidden_size)   # the paper's pin, under the demo key
    assert g.layer == 15 == round(0.469 * 32)
    assert re.fullmatch(r"[0-9a-f]{40}", g.revision)
    # appended at the END: the slug order is pinned by the llama lane test
    assert SDFBASE_SLUGS[-1] == DEMO_KEY
    assert len(set(SDFBASE_SLUGS)) == len(SDFBASE_SLUGS)


# --------------------------------------------------------------------------- #
# the generated config: fit_selection                                          #
# --------------------------------------------------------------------------- #


def test_generated_config_carries_fit_selection_and_the_earlier_ones_do_not(msc):
    built = msc.build_all()
    key = DEMO_KEY
    name = msc.config_filename(key)
    assert (REPO / "configs" / name).read_text() == built[name]
    cfg = yaml.safe_load(built[name])
    paper = yaml.safe_load(built[msc.config_filename("qwen35_4b_base")])
    assert cfg["model"] == msc.MODELS[key]["model"] == paper["model"]
    assert cfg["model_revision"] == msc.MODELS[key]["revision"] == paper["model_revision"]
    # the paper's lane verbatim: the same scope, caps, layer and checkpointing; only the pool dir, the sentinels (its folds are
    # pending) and the fit_selection block differ
    assert cfg["si_ids"] == paper["si_ids"] == msc.SI_IDS
    assert cfg["train_kwargs_by_si"] == paper["train_kwargs_by_si"]
    assert (cfg["layer"], cfg["max_len"]) == (paper["layer"], paper["max_len"]) == (15, 8192)
    assert cfg["train_kwargs"]["default"]["heldout_pool"] == f"/vol/pool/{key}/pairs_constv2-heldout.jsonl"
    folds = msc.load_folds()
    folded = (folds.get("memprobe") or {}).get(key) or {}
    hpo_rows = (folds.get("hpo") or {}).get(key) or {}
    for arm in ("psr_ce_all", "lora_r4_ce"):
        tk = cfg["train_kwargs"][arm]
        assert tk["gradient_checkpointing"] is True
        # the memprobe FOLD (02 --step memprobe) replaces the sentinel with ceil(64 / measured micro-batch) — the demo fits at
        # EFFECTIVE BATCH 64, the committed objects' batch (the seedrep eb-64 recipe's field, never the campaign's 8; the test below);
        # an unfolded lane keeps the sentinel, which spells the lane's own batch (EFF64)
        if arm in folded:
            assert tk["accum_steps"] == -(-64 // int(folded[arm]))
            assert cfg["batch_size_by_arm"][arm] == int(folded[arm])
        else:
            assert tk["accum_steps"] == "PENDING_MEMPROBE_EFF64_ACCUM"
        # the HPO FOLD (03's closed search -> make_sdfbase_configs --fold-hpo) replaces the
        # lr / warm-up / clip sentinels with the recorded winner and carries its provenance;
        # an unfolded arm keeps the sentinel (a configured-not-run lane's search has not closed)
        row = hpo_rows.get(arm) or {}
        if "lr" in row:   # a winner row; a stamp-only row (init_scheme alone) leaves the arm unfolded
            assert tk["lr"] == row["lr"] and tk["warmup_frac"] == row["warmup_frac"] \
                and tk["max_grad_norm"] == row["max_grad_norm"]
            assert row["fit_lr_selected_by"] in msc.FIT_LR_SELECTED_BY and row["fit_search_id"].startswith("hpo-native:")
            guard = row.get("selection_guard") or {}
            assert guard.get("selected_trial") and guard.get("artefact_sha256") and guard.get("fit_fingerprint"), \
                "a folded row names the trial it came from and the artefact that selected it"
            if row["fit_lr_selected_by"] == "manual":
                # a hand-picked operating point: the same guard shape plus its own provenance — basis user_pick, the ruling the row
                # was recorded under, the closed search it came from, the menu identity the authorization read
                assert guard["basis"] == "user_pick" and guard["ruling_id"] == row["architecture_selection_ruling_id"] \
                    and guard["search_id"] == row["fit_search_id"] and guard["search_closed"] is True and guard.get("menu_sha256"), guard
                assert arm == "lora_r4_ce" and str(row["architecture_selection_ruling_id"]).startswith("r2-operating_point-")
            else:
                assert guard.get("search_closed_utc") and guard.get("selected_from"), "a searched row carries the closer's record"
        else:
            assert tk["lr"] == msc.HPO_SENTINEL
    # a lane that declares "fit_width" renders it top-level and header-named; the demo lane, like the paper's, fits on one
    # GPU and declares none (absent = 1)
    assert "fit_width" not in cfg and msc.fit_width_for(key) is None
    assert "# fit_width" not in built[name].split("\nmodel:", 1)[0]
    psr = cfg["train_kwargs"]["psr_ce_all"]["fit_selection"]
    psr_row = hpo_rows.get("psr_ce_all")
    if psr_row is not None and "lr" in psr_row:
        assert psr == {"fit_search_id": psr_row["fit_search_id"],
                       "fit_lr_selected_by": psr_row["fit_lr_selected_by"],
                       "init_scheme": psr_row.get("init_scheme", "zero")}
    else:
        assert psr == {"fit_search_id": msc.HPO_SENTINEL,
                       "fit_lr_selected_by": "search", "init_scheme": "zero"}
    lora = cfg["train_kwargs"]["lora_r4_ce"]["fit_selection"]
    lora_row = hpo_rows.get("lora_r4_ce") or {}
    # FOLD-OR-DEFAULT: a row carrying a hand-picked architecture renders ITS rank / modules / init / search id /
    # ruling; a row without one renders the paper's LITERAL default — rank 4 on o_proj, no per-expert map, the sentinel search id,
    # no ruling — so the attention-out default stays guarded for every lane that has not picked (the demo lane today)
    exp_modules = sorted(lora_row.get("lora_modules", ["o_proj"]))
    exp_rank = lora_row.get("lora_rank", 4)
    # `per_expert_rank: null` in a row means DERIVED by FD 6's rule from the lane's declared active experts — a
    # sparse-MoE lane's MLP set renders the explicit map the construction builds, with the rule and the divisor beside it; a dense
    # lane / attention-only set stays null with no rule (the demo lane: dense)
    exp_per = lora_row.get("per_expert_rank") if lora_row.get("per_expert_rank") is not None else msc.per_expert_rank_for(key, exp_rank, exp_modules)
    assert exp_per is None
    expected = {"fit_search_id": lora_row.get("fit_search_id", msc.HPO_SENTINEL), "lora_rank": exp_rank,
                "lora_modules": exp_modules, "per_expert_rank": None,
                "fit_lr_selected_by": lora_row.get("fit_lr_selected_by", "search"),
                "architecture_selection_ruling_id": lora_row.get("architecture_selection_ruling_id"),
                "init_scheme": lora_row.get("init_scheme", "zero")}
    assert lora == expected, (lora, expected)
    if "fit_search_id" not in lora_row:
        assert lora["lora_rank"] == 4 and lora["lora_modules"] == ["o_proj"] and lora["per_expert_rank"] is None \
            and lora["fit_search_id"] == msc.HPO_SENTINEL and lora["architecture_selection_ruling_id"] is None and "per_expert_rank_rule" not in lora
    assert list(lora) == ["fit_search_id", "lora_rank", "lora_modules",
                          "per_expert_rank", "fit_lr_selected_by",
                          "architecture_selection_ruling_id", "init_scheme"]
    tk_lora = cfg["train_kwargs"]["lora_r4_ce"]
    assert tk_lora["lora_rank"] == lora_row.get("lora_rank", 4)
    # stamp-fix-2: what the TRAINER receives — the fitter pops fit_selection, so the architecture keys ride
    # the arm block top-level iff the fold row states them, equal to the sidecar; a lane whose row says nothing keeps the trainer's
    # (paper) defaults and no such key
    for k in ("lora_modules", "per_expert_rank", "init_scheme"):
        if k in lora_row:
            assert tk_lora[k] == lora[k], k
        else:
            assert k not in tk_lora, k
    psr_tk = cfg["train_kwargs"]["psr_ce_all"]
    if "init_scheme" in (hpo_rows.get("psr_ce_all") or {}):
        assert psr_tk["init_scheme"] == hpo_rows["psr_ce_all"]["init_scheme"] == psr_tk["fit_selection"]["init_scheme"]
    else:
        assert "init_scheme" not in psr_tk
    assert "# fit_selection" in built[name].split("\nmodel:", 1)[0]
    # a dense lane declares no per-expert divisor
    assert "moe_active_experts" not in msc.MODELS[DEMO_KEY]
    # the recipe the demo fits under (the test below): batch 64, seed 3 (the committed objects' seed) — declared through the seedrep fields, rendered
    # into the config
    assert cfg["train_kwargs"]["default"]["seed"] == 3 and "seed" not in paper["train_kwargs"]["default"]
    assert "EFF8" not in built[name] and "effective batch 8" not in built[name].split("\nmodel:", 1)[0]
    # ABSENT on every other lane (their bytes are the fit identity of finished cells)
    assert msc.PIPELINE_TEST_KEYS == (DEMO_KEY,)
    for key in msc.MODELS:
        if key in msc.PIPELINE_TEST_KEYS:
            continue
        cfg = yaml.safe_load(built[msc.config_filename(key)])
        for arm in ("psr_ce_all", "lora_r4_ce"):
            assert "fit_selection" not in cfg["train_kwargs"][arm], key
        assert "fit_selection" not in built[msc.config_filename(key)]
        assert not msc.emits_fit_selection(key, None)


def test_demo_lane_fits_at_effective_batch_64_under_seed_3_through_the_seedrep_recipe_fields(msc):
    """The demo lane's own fits use effective batch 64: the demo's fits match the paper's COMMITTED
    Qwen3.5-4B objects in batch and seed — EFFECTIVE BATCH 64 under SEED 3 (the one-seed objects/DEMO.json, 2026-10-12: every fitted
    object is training seed 3, the fit-your-own default) — declared through the two MODELS fields the seedrep
    `_eb64_s<s>` keys already use (`effective_batch: SEEDREP_EFFECTIVE_BATCH`, `seed: 3`; no new constant, no fit_width). The campaign
    constant (8) and the paper lane are untouched; the rendered config says both (the EFF64 accum sentinel, `train_kwargs.default.seed:
    3`, the header's seed line), the fold arithmetic is ceil(64 / bs), and the trainer's sentinel rule (tune_arm config_effective_batch:
    an unfolded config's EFF<n>) reads 64 from it."""
    key = DEMO_KEY
    spec = msc.MODELS[key]
    assert spec["effective_batch"] == msc.SEEDREP_EFFECTIVE_BATCH == 64 and spec["seed"] == 3 and "fit_width" not in spec
    assert msc.effective_batch_for(key) == 64 == msc.SEEDREP_EFFECTIVE_BATCH and msc.seed_for(key) == 3 and msc.fit_width_for(key) is None
    assert msc.EFFECTIVE_BATCH == 8 and msc.ACCUM_SENTINEL == "PENDING_MEMPROBE_EFF8_ACCUM"          # the campaign constant: untouched
    assert msc.effective_batch_for("qwen35_4b_base") == 8 and msc.seed_for("qwen35_4b_base") is None   # the paper lane: untouched
    for bs in (1, 2, 4):
        assert msc._accum_for(bs, msc.effective_batch_for(key)) == -(-64 // bs)
    assert msc._accum_for(None, msc.effective_batch_for(key)) == "PENDING_MEMPROBE_EFF64_ACCUM"
    built = msc.build_all()
    name = msc.config_filename(key)
    text = built[name]
    assert (REPO / "configs" / name).read_text() == text                      # the committed file is the rendering
    cfg = yaml.safe_load(text)
    assert cfg["train_kwargs"]["default"]["seed"] == 3
    folded = (msc.load_folds().get("memprobe") or {}).get(key) or {}
    for arm in ("psr_ce_all", "lora_r4_ce"):
        tk = cfg["train_kwargs"][arm]
        assert "seed" not in tk                                               # the default block only (the trainer's precedence)
        if arm in folded:
            bs = int(folded[arm])
            assert cfg["batch_size_by_arm"][arm] == bs and tk["accum_steps"] == -(-64 // bs) and bs * tk["accum_steps"] >= 64
        else:
            assert cfg["batch_size_by_arm"][arm] == "MEMPROBE"
            assert re.fullmatch(r"PENDING_MEMPROBE_EFF(\d+)_ACCUM", tk["accum_steps"]).group(1) == "64"
    head = text.split("\nmodel:", 1)[0]
    assert "# seed: 3 (train_kwargs.default.seed)" in head
    assert "effective batch 8" not in head and "EFF8" not in text
    if folded:
        assert "#   accum_steps = ceil(64 / bs) — effective batch 64." in head
    # the same recipe as the paper lane's seed-3 replicate (model, revision, layer, batch, seed — the replicate the committed objects came
    # from; the tag follows the declared seed through seedrep_key, never a separate spelling), under the demo's own pool dir and the
    # paper lane's cell scope — the replicate's scope is its own
    s3 = yaml.safe_load(built[msc.config_filename(msc.seedrep_key(3))])
    assert (cfg["model"], cfg["model_revision"], cfg["layer"]) == (s3["model"], s3["model_revision"], s3["layer"])
    assert cfg["train_kwargs"]["default"]["seed"] == s3["train_kwargs"]["default"]["seed"] == 3
    assert msc.effective_batch_for(msc.seedrep_key(3)) == msc.effective_batch_for(key) == 64
    assert cfg["train_kwargs"]["default"]["heldout_pool"] == f"/vol/pool/{key}/pairs_constv2-heldout.jsonl" != s3["train_kwargs"]["default"]["heldout_pool"]
    assert cfg["si_ids"] == msc.SI_IDS != s3["si_ids"]


def test_generator_vocabulary_equals_lora_modules(msc, monkeypatch):
    assert msc.INIT_SCHEMES == lm.INIT_SCHEMES
    assert set(msc.LORA_MODULE_NAMES) == set(lm.KNOWN_MODULES)
    assert set(msc.LORA_MLP_MODULES) == set(lm.MLP_MODULES)
    assert msc.PAPER_LORA_RANK == lm.PAPER_LORA_RANK
    assert msc.PAPER_LORA_MODULES == list(lm.MODULE_SETS["wo"])
    # the per-expert rank rule (FD 6: r_e = max(1, r // active_experts)) is lora_modules' — derived for a lane that declares its
    # active experts and an MLP set, never for a dense lane or an attention-only set. No lane in the table is sparse today, so the
    # MoE case is a SYNTHETIC lane: the demo's row plus a divisor
    monkeypatch.setitem(msc.MODELS, "synthetic_moe_lane", {**msc.MODELS[DEMO_KEY], "moe_active_experts": 8})
    for rank in (4, 16, 64):
        assert msc.per_expert_rank_for("synthetic_moe_lane", rank, ["down_proj", "gate_proj", "up_proj"]) == \
            lm.per_expert_rank_map(rank, 8)
        assert msc.per_expert_rank_for("synthetic_moe_lane", rank, ["o_proj"]) is None
        assert msc.per_expert_rank_for(DEMO_KEY, rank, ["down_proj"]) is None
    assert msc.per_expert_rank_for("synthetic_moe_lane", 4, ["down_proj"]) == {"shared_expert": 4, "routed_expert": 1}
    assert msc.per_expert_rank_for("synthetic_moe_lane", 64, ["down_proj"]) == {"shared_expert": 64, "routed_expert": 8}


def test_fold_row_with_selection_keys_renders_the_pick(msc):
    win = {DEMO_KEY: {"lora_r4_ce": {
        "lr": 1e-3, "warmup_frac": 0.15, "max_grad_norm": 0.5,
        "fit_search_id": "hpo-native:abcdef0123456789", "lora_rank": 16,
        "lora_modules": ["up_proj", "gate_proj", "down_proj", "o_proj", "q_proj", "k_proj", "v_proj"],
        "init_scheme": "random_lecun_1e-3", "fit_lr_selected_by": "manual",
        "architecture_selection_ruling_id": "ruling:0123456789abcdef"},
        "psr_ce_all": {"init_scheme": "random_lecun_1e-3",
                       "fit_search_id": "hpo-native:fedcba9876543210"}}}
    cfg = yaml.safe_load(msc.build_all(hpo=win, folds_path=None)[msc.config_filename(DEMO_KEY)])
    lora = cfg["train_kwargs"]["lora_r4_ce"]
    assert lora["lora_rank"] == 16 and lora["lr"] == pytest.approx(1e-3)
    assert lora["fit_selection"] == {
        "fit_search_id": "hpo-native:abcdef0123456789", "lora_rank": 16,
        "lora_modules": ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"],
        "per_expert_rank": None,
        "fit_lr_selected_by": "manual",
        "architecture_selection_ruling_id": "ruling:0123456789abcdef",
        "init_scheme": "random_lecun_1e-3"}
    # stamp-fix-3: a dense lane carries no per-expert map — a row stating null stays null, and a row stating a map on a dense lane
    # refuses at generation (FD 6's rule derives the map from the lane's declared active experts, which a dense lane has none of)
    assert lora.get("per_expert_rank") is None and "per_expert_rank_rule" not in lora["fit_selection"]   # a row that states no map emits none
    win_null = {DEMO_KEY: {"lora_r4_ce": {**win[DEMO_KEY]["lora_r4_ce"], "per_expert_rank": None}}}
    cfg_null = yaml.safe_load(msc.build_all(hpo=win_null, folds_path=None)[msc.config_filename(DEMO_KEY)])
    assert cfg_null["train_kwargs"]["lora_r4_ce"]["fit_selection"]["per_expert_rank"] is None
    with pytest.raises(SystemExit, match="FD 6's rule"):
        msc.build_all(hpo={DEMO_KEY: {"lora_r4_ce": {**win[DEMO_KEY]["lora_r4_ce"], "per_expert_rank": {"shared_expert": 16, "routed_expert": 2}}}}, folds_path=None)
    psr = cfg["train_kwargs"]["psr_ce_all"]
    # a selection-only row keeps the optimizer sentinels (the search has not closed)
    assert psr["lr"] == msc.HPO_SENTINEL
    assert psr["fit_selection"] == {"fit_search_id": "hpo-native:fedcba9876543210",
                                    "fit_lr_selected_by": "search",
                                    "init_scheme": "random_lecun_1e-3"}
    # refusals: a partial winner, an unknown key, a bad scheme / module / rank
    with pytest.raises(SystemExit, match="missing"):
        msc.build_all(hpo={DEMO_KEY: {"psr_ce_all": {"lr": 1e-3}}}, folds_path=None)
    with pytest.raises(SystemExit, match="unknown key"):
        msc.build_all(hpo={DEMO_KEY: {"psr_ce_all": {"typo": 1}}}, folds_path=None)
    with pytest.raises(SystemExit, match="init_scheme"):
        msc.build_all(hpo={DEMO_KEY: {"psr_ce_all": {"init_scheme": "xavier"}}}, folds_path=None)
    with pytest.raises(SystemExit, match="unknown lora module"):
        msc.build_all(hpo={DEMO_KEY: {"lora_r4_ce": {"lora_modules": ["experts"]}}}, folds_path=None)
    with pytest.raises(SystemExit, match="lora_rank"):
        msc.build_all(hpo={DEMO_KEY: {"lora_r4_ce": {"lora_rank": 0}}}, folds_path=None)
    # a selection key on a byte-pinned lane would emit the block there: the generator does it
    # (the byte pins catch it), so the OVERRIDE refuses such a lane up front
    cfg2 = yaml.safe_load(msc.build_all(
        hpo={"gemma4e4b_base": {"lora_r4_ce": {"init_scheme": "zero"}}},
        folds_path=None)["sdfbase_gemma4e4b_base_fits.yaml"])
    assert "fit_selection" in cfg2["train_kwargs"]["lora_r4_ce"]


def test_init_scheme_override_stamps_both_arms_and_refuses_pinned_lanes(msc, tmp_path):
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / msc.FOLDS_FILENAME
    # a recorded winner survives the stamp (merged, never replaced)
    msc.update_folds("hpo", {DEMO_KEY: {"lora_r4_ce": {
        "lr": 2e-3, "warmup_frac": 0.15, "max_grad_norm": 0.3}}}, path=path, repo=tmp_path)
    folds = msc.stamp_init_scheme("random_lecun_1e-3", [DEMO_KEY], path=path, repo=tmp_path)
    assert folds["hpo"][DEMO_KEY]["lora_r4_ce"] == {
        "lr": 2e-3, "warmup_frac": 0.15, "max_grad_norm": 0.3, "init_scheme": "random_lecun_1e-3"}
    assert folds["hpo"][DEMO_KEY]["psr_ce_all"] == {"init_scheme": "random_lecun_1e-3"}
    built = msc.build_all(folds_path=path)
    cfg = yaml.safe_load(built[msc.config_filename(DEMO_KEY)])
    for arm in ("psr_ce_all", "lora_r4_ce"):
        assert cfg["train_kwargs"][arm]["fit_selection"]["init_scheme"] == "random_lecun_1e-3"
    assert "random_lecun_1e-3/random_lecun_1e-3" in built[msc.config_filename(DEMO_KEY)]
    assert cfg["train_kwargs"]["lora_r4_ce"]["lr"] == pytest.approx(2e-3)
    # the earlier lanes are untouched by the stamp
    for name, text in msc.build_all(folds_path=path).items():
        if name != msc.config_filename(DEMO_KEY):
            assert text == built[name] and "fit_selection" not in text
    with pytest.raises(SystemExit, match="byte-pinned"):
        msc.stamp_init_scheme("random_lecun_1e-3", ["gemma4e4b_base"], path=path, repo=tmp_path)
    with pytest.raises(SystemExit, match="not one of"):
        msc.stamp_init_scheme("xavier", [DEMO_KEY], path=path, repo=tmp_path)
    with pytest.raises(SystemExit, match="unknown model key"):
        msc.stamp_init_scheme("zero", ["nope"], path=path, repo=tmp_path)
    # the CLI needs the lanes named
    with pytest.raises(SystemExit):
        msc.main(["--init-scheme-override", "random_lecun_1e-3", "--check"])
    with pytest.raises(SystemExit):
        msc.main(["--init-scheme-override", "random_lecun_1e-3"])


def test_models_table_cross_check(msc, tmp_path):
    notes = msc.check_models_table(REPO / "reproduce/models.yaml")
    assert any(n.startswith(f"{DEMO_KEY}: models.yaml row '{SERVED_KEY}' agrees") for n in notes)
    assert any(n.startswith(f"qwen35_4b_base: models.yaml row '{PAPER_KEY}' agrees") for n in notes)
    # matched by fit.checkpoint.model_id == the lane key, never by hf_repo: the it_doc lanes
    # (same repo as a served checkpoint) are NOT front-door fit models
    assert any(n.startswith("qwen35_4b_it_doc: no models.yaml row") for n in notes)
    assert msc.check_models_table(tmp_path / "absent.yaml") == [
        f"models table {tmp_path / 'absent.yaml'} absent — pins not cross-checked"]
    assert msc.check_models_table(None) == []
    bad = tmp_path / "models.yaml"
    bad.write_text(yaml.safe_dump({"schema": 1, "models": {SERVED_KEY: {"fit": {"checkpoint": {
        "hf_repo": "Qwen/Qwen3.5-4B-Base", "revision": "e2ee613a3820d4482636d51b6d2eec45eb6d025b",
        "model_id": DEMO_KEY}}}}}))
    with pytest.raises(SystemExit, match="one pin, one table"):
        msc.check_models_table(bad)


def test_models_table_pins_agree_with_the_spec():
    doc = yaml.safe_load((REPO / "reproduce/models.yaml").read_text())
    g = doc["models"][SERVED_KEY]
    q = doc["models"][PAPER_KEY]
    assert g["fit"]["checkpoint"] == {"hf_repo": "Qwen/Qwen3.5-4B-Base", "revision": DEMO_SHA, "model_id": DEMO_KEY}
    assert g["serve"]["checkpoint"]["hf_repo"] == sg.ONBOARDED_MODEL_HF_IDS[SERVED_KEY] == SERVED_HF
    # the paper's two checkpoints under the demo's own ids; the architecture, thinking, layer sets and LoRA architecture copied
    for a, b in ((g["fit"]["checkpoint"], q["fit"]["checkpoint"]), (g["serve"]["checkpoint"], q["serve"]["checkpoint"])):
        assert (a["hf_repo"], a["revision"]) == (b["hf_repo"], b["revision"]) and a["model_id"] != b["model_id"]
    assert g["serve"]["chat_template_hash"] == q["serve"]["chat_template_hash"]
    for k in ("architecture", "thinking", "layer_sets", "lora_architecture", "relu_steer", "attacker", "transfer_kind"):
        assert g[k] == q[k], k
    assert g["architecture"]["attention_layers"] == list(range(3, 32, 4)) and g["architecture"]["n_layers"] == 32
    assert "moe" not in g["architecture"] and g["lora_architecture"]["per_expert_rank"] is None
    assert g["view_legacy_prefix"] == SERVED_KEY and g["display"] == "Qwen3.5-4B (worked example)"
    # the SINGLE-PHASE recipe: one wave carrying every arm beside its own unsteered legs + the two dose views
    assert list(g["views"]) == ["battery", "d0p75", "d0p75_relu"] and "u0" not in g["views"]
    assert g["views"]["battery"]["cells"][:3] == ["none", "prompted__safety_refusal", "prompted__si_constitution"]
    assert set(g["views"]["battery"]["cells"][3:]) == set(g["objects"]) - {"arm_lora", "arm_relu", "corpus_constitution", "corpus_control"}
    assert g["run_in_this_work"] is False


# --------------------------------------------------------------------------- #
# the Petri proxy port table                                                   #
# --------------------------------------------------------------------------- #


def test_ports_yaml_equals_the_proxy_case_table_and_adds_the_test_model():
    doc = yaml.safe_load((REPO / "configs/sdfbase_full22_ports.yaml").read_text())
    assert doc["schema"] == 1
    assert doc["offsets"] == {"overflow_rerun": 100, "second_pass": 200, "local_tunnel": 10000}
    ports = doc["ports"]
    proxy = (REPO / "scripts/sdfbase/full22/petri_proxy.sbatch").read_text()
    case = {tag: int(port) for tag, port in re.findall(r"([a-z0-9_]+)\) PROXY_PORT=(\d+);;", proxy)}
    assert case and all(ports[tag] == port for tag, port in case.items()), \
        {t: (case[t], ports.get(t)) for t in case if ports.get(t) != case[t]}
    new = {t: p for t, p in ports.items() if t not in case}
    # the new rows: the test model's `main` tag + one per PHASE VIEW the models table names — read from the table, never a
    # hand-listed suffix tuple (every table view needs its port, and a view only one model's table names is that model's alone)
    doc_m = yaml.safe_load((REPO / "reproduce/models.yaml").read_text())["models"]
    assert set(new) == {f"{SERVED_KEY}{v}" for v in ("", *(f"_{x}" for x in doc_m[SERVED_KEY]["views"]))}, sorted(new)
    assert ports[SERVED_KEY] == 8966 and f"{SERVED_KEY}_battery" in ports and f"{SERVED_KEY}_u0" not in ports and f"{SERVED_KEY}_s2r" not in ports
    vals = list(ports.values())
    assert len(set(vals)) == len(vals)
    assert all(8900 <= p < 9000 for p in vals)
    base = set(vals)
    assert not (base & {p + 100 for p in vals}) and not (base & {p + 200 for p in vals})
    assert not (base & {p + 10000 for p in vals})
    assert 8990 not in base and 8992 not in base
    # the phase views are the models table's
    for view in doc_m[SERVED_KEY]["views"]:
        assert f"{SERVED_KEY}_{view}" in ports, view


# --------------------------------------------------------------------------- #
# geometry                                                                     #
# --------------------------------------------------------------------------- #


def test_geometry_onboarded_rows_leave_the_pinned_tables_alone(caplog, monkeypatch):
    """MODEL_KEYS / BUDGET_BY_MODEL / HYBRID_MODEL_KEYS / MODEL_HF_IDS stay
    EXACTLY the sweep's eight (the tinycap contracts and the served-decode
    sweep pin them); an onboarded model lives in the ONBOARDED tables and
    rows_for / budget_for / model_key_for serve both."""
    assert sg.MODEL_KEYS == ("llama8b", "llama3b", "gemma4e4b", "gemma4e2b",
                             "qwen35_9b", "qwen35_4b", "glm46vflash", "nemotron3nano4b")
    assert tuple(sg.BUDGET_BY_MODEL) == sg.MODEL_KEYS and set(sg.MODEL_HF_IDS) == set(sg.MODEL_KEYS)
    assert sg.HYBRID_MODEL_KEYS == ("qwen35_9b", "qwen35_4b", "nemotron3nano4b")
    assert sg.ONBOARDED_MODEL_KEYS == (SERVED_KEY,)
    assert sg.KNOWN_MODEL_KEYS == sg.MODEL_KEYS + sg.ONBOARDED_MODEL_KEYS
    assert sg.ONBOARDED_HYBRID_MODEL_KEYS == (SERVED_KEY,)      # the Qwen3.5 gated-deltanet stack: the sweep's qwen35_4b is hybrid too
    assert sg.ONBOARDED_MODEL_HF_IDS == {SERVED_KEY: SERVED_HF}
    assert not (set(sg.ONBOARDED_MODEL_KEYS) & set(sg.MODEL_KEYS))
    # no onboarded model carries a built-in budget row today: the tuple is empty and so is the table built from it
    assert sg.ONBOARDED_BUDGETS == () and sg.ONBOARDED_BUDGET_BY_MODEL == {}
    # the lookup: an onboarded key from the onboarded rows (hybrid, no row), a paper key from the sweep's
    g = sg.rows_for(SERVED_KEY)
    assert (g.onboarded, g.hybrid, g.hf_id, g.budget) == (True, True, SERVED_HF, None)
    assert not g.provisional
    ll = sg.rows_for("llama8b")
    assert ll.onboarded is False and ll.budget is sg.BUDGET_BY_MODEL["llama8b"] and not ll.hybrid
    assert sg.rows_for(PAPER_KEY).hybrid is True and sg.is_hybrid(SERVED_KEY)
    with pytest.raises(sg.UnknownModelKey):
        sg.rows_for(f"{SERVED_KEY}_it")                                           # a suffix spelling of the served id is not a key
    # budget_for routes through it: the default WITH the warning
    with caplog.at_level(logging.WARNING, logger="sisteer.capeval.serving.geometry"):
        rg = sg.budget_for(SERVED_KEY)
    assert rg.source == sg.DEFAULT_BUDGET_SOURCE and rg.batched_token_budget == 393_216 and not rg.per_model
    assert rg.entry is sg._LLAMA8B_CERTIFIED
    assert any(SERVED_KEY in r.getMessage() and "NO measured" in r.getMessage() for r in caplog.records)
    assert sg.identity_seqs(49152, model_key=SERVED_KEY) == 393_216 // 49152 == 8      # under HYBRID_SEQS_CAP
    with pytest.raises(sg.UnknownModelKey):
        sg.budget_for("qwen3.5-4b-demo")
    # a PROVISIONAL onboarded row, when one exists, sizes from itself and warns (the table is read at call time: the documented seam)
    prov = sg.ModelBudget(
        model_key=SERVED_KEY, batched_token_budget=24 * 16384, certified=False, source=sg.PROVISIONAL_BUDGET_SOURCE,
        basis=sg.BudgetBasis(method="CONSERVATIVE CEILING, not a measurement (synthetic)", max_model_len=16384, max_num_seqs=24,
                             gpu_memory_utilization=0.9, kv_tokens_reported=None, probe_job_id=None, measured_on=sg.UNMEASURED,
                             source="synthetic", first_refused=None, note="NOT observed to fail; a test row"))
    monkeypatch.setitem(sg.ONBOARDED_BUDGET_BY_MODEL, SERVED_KEY, prov)
    q = sg.rows_for(SERVED_KEY)
    assert q.provisional and q.budget is prov and q.onboarded and q.hybrid
    assert prov.seqs_at(16384) == 24 and prov.seqs_at(49152) == 8 and prov.window_ceiling(49152) is None
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="sisteer.capeval.serving.geometry"):
        rq = sg.budget_for(SERVED_KEY)
    assert rq.provisional and rq.per_model and rq.batched_token_budget == 393_216 and rq.entry is prov
    assert any("PROVISIONAL" in r.getMessage() and SERVED_KEY in r.getMessage() for r in caplog.records)
    monkeypatch.delitem(sg.ONBOARDED_BUDGET_BY_MODEL, SERVED_KEY)
    assert sg.rows_for(SERVED_KEY).budget is None
    # the HF-id lookup serves both maps in KNOWN_MODEL_KEYS order: the demo SHARES the sweep's Qwen3.5-4B checkpoint, so its HF id
    # resolves to the PAPER's key (the sweep's rows come first); the sweep's eight resolve exactly as before
    assert sg.model_key_for(SERVED_HF) == PAPER_KEY
    assert sg.model_key_for("/staging/models/Qwen--Qwen3.5-4B") == PAPER_KEY
    assert sg.model_key_for("NousResearch/Meta-Llama-3.1-8B-Instruct") == "llama8b"
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="sisteer.capeval.serving.geometry"):
        assert sg.model_key_for("org/unknown-model") is None
    assert any("matches no canonical model key" in r.getMessage() for r in caplog.records)
    # a ModelBudget accepts an onboarded key and refuses a stray one
    sg.ModelBudget(model_key=SERVED_KEY, batched_token_budget=16384 * 4, source=sg.PROVISIONAL_BUDGET_SOURCE,
                   basis=sg.BudgetBasis("m", 16384, 4, 0.9, None, None, sg.UNMEASURED, "s"))
    with pytest.raises(ValueError, match="unknown model key"):
        sg.ModelBudget(model_key=f"{SERVED_KEY}_it", batched_token_budget=16384 * 4,
                       basis=sg.BudgetBasis("m", 16384, 4, 0.9, None, None, "d", "s"))
    # describe_budgets is the sweep's table, untouched (eight rows; an onboarded model is NOT a table row); the ceilings section
    # below the table lists the measured per-window ceilings — the shipped nemotron row, nothing for the onboarded model
    md = sg.describe_budgets()
    rows = [l for l in md.splitlines() if l.startswith("| ") and not l.startswith("| model_key")]
    assert len(rows) == 8 and not any(l.startswith(f"| {SERVED_KEY}") for l in rows)
    assert "nemotron3nano4b @ 4608: S 167" in md and f"{SERVED_KEY} @" not in md
    for name in ("ONBOARDED_MODEL_KEYS", "KNOWN_MODEL_KEYS", "ONBOARDED_BUDGETS", "rows_for", "ModelRows"):
        assert name in sg.__all__
    src = (REPO / "src/sisteer/capeval/serving/geometry.py").read_text()
    assert not re.search(r"/home/|/workspace", src[src.index("ONBOARDED_BUDGETS: Tuple"):])


def test_models_py_has_one_auto_model_class():
    src = (REPO / "src/sisteer/models.py").read_text()
    assert src.count("def auto_model_class(") == 1
    assert json.loads(json.dumps({"ok": True}))  # the module imports (top of file)
