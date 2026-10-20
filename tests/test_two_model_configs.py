"""The derived per-model configs.

No network: the per-SI teacher-prompt token counts are FIXTURES (provenance
below), and every assert runs against the committed YAML plus the pure
derivation function, so drift in the base config, the derivation, or the
committed files fails here rather than on a pod.
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from derive_model_config import (ATTACK_ALLOWANCE, BATCH_PLACEHOLDER,
                                 CLASS_A_ARMS, CLASS_C_ARMS, COMPLETION_CAP,
                                 EOS, MODEL_SPECS, TRAIN_FAMILIES,
                                 derive_config, load_base_config,
                                 translate_hpo_results)
from sisteer.eval.train_budget import derive_max_len, resolve_batch_size

# Measured 2026-08-25 on the LIVE tokenizers at the pinned revisions
# (google/gemma-4-E4B-it @ ee0ef602..., Qwen/Qwen3.8-27B @ 1d4bf0f2...) via
# `scripts/derive_model_config.py --print-counts`: each SI rendered as a
# teacher chat prompt (build_chat_prompt(tok, "", system=<SI text>)),
# counted with add_special_tokens=True — the base config's documented
# convention. NEVER the Llama counts (104/5888/6396).
# glm46vflash measured 2026-09-02 the same way (zai-org/GLM-4.6V-Flash @
# 411bb4d7..., under load_tokenizer's enable_thinking=False pin — the
# template appends /nothink to the user turn and renders the
# <|assistant|>\n<think></think>\n header; bos is None and "" encodes to
# [], so add_special_tokens=True adds nothing).
FIXTURE_COUNTS = {
    "gemma4e4b": {
        "safety_refusal": 101,
        "si_constitution": 6041,
        "si_irrelevant": 6574,
    },
    "qwen38_27b": {
        "safety_refusal": 105,
        "si_constitution": 5975,
        "si_irrelevant": 6695,
    },
    "glm46vflash": {
        "safety_refusal": 100,
        "si_constitution": 5883,
        "si_irrelevant": 6419,
    },
    # Measured 2026-09-02 on the live nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
    # tokenizer @ dfaf35de... (same --print-counts convention; ChatML
    # template, empty system turn rendered when none is given, no BOS).
    "nemotron3nano4b": {
        "safety_refusal": 104,
        "si_constitution": 5968,
        "si_irrelevant": 6665,
    },
}

# The pins the sweep runs at (verified against the HF API 2026-08-25;
# glm46vflash and Nemotron 2026-09-02). layer = round(0.469 x n_layers):
# 20 / 30 / 19 / 20.
EXPECTED_PINS = {
    "gemma4e4b": ("google/gemma-4-E4B-it",
                  "ee0ef6023621cff504d758262d4e04895a5af4a2", 20),
    "qwen38_27b": ("Qwen/Qwen3.8-27B",
                   "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", 30),
    "glm46vflash": ("zai-org/GLM-4.6V-Flash",
                    "411bb4d77144a3f03accbf4b780f5acb8b7cde4e", 19),
    "nemotron3nano4b": ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
                        "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f", 20),
}

# This module pins the twosi-model-sweeps INSTRUCT campaign's derived
# two_si_<slug>.yaml configs. The sdf-base-steering base-model entries in
# MODEL_SPECS exist for onboarding-pin resolution only and have their own
# generator + pins (tests/test_sdfbase_configs.py) — scope them out here.
MODEL_SPECS = {slug: spec for slug, spec in MODEL_SPECS.items()
               if slug in FIXTURE_COUNTS}

# Measured/HPO values folded into the COMMITTED per-model configs AFTER
# derivation — applied to the files, not the generator (their provenance
# lives in the yaml comments), so the regeneration pin overlays exactly
# these seams and pins their values here. Anything else diverging from a
# clean regeneration still fails.
#
# E4B HPO winners: all 4 fitted argmins (2026-08-26, f451e7d;
# e4b_hpo_winners_v4.json — psr_kl r1+r2 cluster, psr_mse r1+r3 / lora r2 /
# tf r3 Modal).
E4B_FOLDED_WINNERS = {
    "psr_kl_all": {"lr": 0.0010641062986669792,
                   "lr_schedule": "warmup_cosine", "warmup_frac": 0.05,
                   "max_grad_norm": 0.18028742734440237},
    "psr_mse_all": {"lr": 0.0014216398292720336,
                    "lr_schedule": "warmup_cosine", "warmup_frac": 0.05,
                    "max_grad_norm": 0.6712460967470124},
    "lora_r1_all": {"lr": 0.0025720858234871626,
                    "lr_schedule": "warmup_cosine", "warmup_frac": 0.15,
                    "max_grad_norm": 0.06459902506503866},
    "token_faithful": {"lr": 0.020848410154465532,
                       "lr_schedule": "warmup_cosine", "warmup_frac": 0.15,
                       "max_grad_norm": 0.28820578791616136},
}
# Qwen Phase-D memprobe widths (2026-08-26, with the lora_r1_all
# hybrid-attention capability gate): lora_r1_all bs1 is a measured 87%-VRAM
# guard exception; bidpo is deferred and still a placeholder.
QWEN_MEASURED_BS = {"psr_kl_all": 2, "psr_mse_all": 2, "lora_r1_all": 1,
                    "token_faithful": 2, "bidpo": BATCH_PLACEHOLDER}
FOLDED = {
    "gemma4e4b": {"train_kwargs": E4B_FOLDED_WINNERS},
    "qwen38_27b": {"batch_size_by_arm": QWEN_MEASURED_BS},
    # glm46vflash: registered 2026-09-02, nothing measured yet — the
    # committed config IS the bare derivation (all-placeholder widths,
    # default + bidpo train_kwargs only).
    "glm46vflash": {},
    # Nemotron (2026-09-02): nothing folded yet — no memprobe, no HPO. The
    # committed file IS the bare derivation: every arm width a placeholder,
    # per-arm train_kwargs empty but for bidpo's semantics block.
    "nemotron3nano4b": {},
}


@pytest.fixture(scope="module")
def base():
    return load_base_config()


def _committed(slug: str) -> dict:
    path = MODEL_SPECS[slug].config_path
    assert path.exists(), f"{path} missing — run scripts/derive_model_config.py"
    with open(path) as fh:
        return yaml.safe_load(fh)


@pytest.mark.parametrize("slug", sorted(MODEL_SPECS))
def test_committed_config_equals_regeneration(base, slug):
    """The strongest pin: the committed file IS derive_config(base, spec,
    fixture counts) — any hand edit, stale regeneration, or silent base
    drift fails loudly — MODULO the two folded seams pinned in FOLDED
    above (per-arm HPO winner blocks; measured batch widths), which are
    asserted to their exact folded values before being overlaid, so a
    third kind of drift cannot hide in the overlay."""
    cfg = derive_config(base, MODEL_SPECS[slug], FIXTURE_COUNTS[slug])
    committed = _committed(slug)
    folds = FOLDED[slug]
    for arm, block in (folds.get("train_kwargs") or {}).items():
        assert committed["train_kwargs"].get(arm) == block, (slug, arm)
        cfg["train_kwargs"][arm] = dict(block)
    if "batch_size_by_arm" in folds:
        assert committed["batch_size_by_arm"] == folds["batch_size_by_arm"]
        cfg["batch_size_by_arm"] = dict(folds["batch_size_by_arm"])
    assert committed == cfg


@pytest.mark.parametrize("slug", sorted(MODEL_SPECS))
def test_model_revision_pinned(slug):
    cfg = _committed(slug)
    model, revision, _layer = EXPECTED_PINS[slug]
    assert cfg["model"] == model
    assert cfg["model_revision"] == revision
    assert cfg["dtype"] == "bfloat16"
    assert cfg["attn_implementation"] == "sdpa"


@pytest.mark.parametrize("slug", sorted(MODEL_SPECS))
def test_layer_is_depth_fraction(slug):
    """round(0.469 x n_layers): 20 on the 42-layer E4B, 30 on the 64-layer
    Qwen, 19 on the 40-layer GLM-4.6V-Flash (bidpo only; all-layer arms
    ignore it)."""
    assert _committed(slug)["layer"] == EXPECTED_PINS[slug][2]


@pytest.mark.parametrize("slug", sorted(MODEL_SPECS))
def test_max_len_by_si_matches_derive_max_len_on_fixture_counts(slug):
    cfg = _committed(slug)
    want = {
        si: derive_max_len(n, COMPLETION_CAP,
                           attack_allowance=ATTACK_ALLOWANCE, eos=EOS)
        for si, n in FIXTURE_COUNTS[slug].items()
    }
    assert cfg["max_len_by_si"] == want
    assert cfg["max_len"] == max(want.values())
    # The one bucket the Llama numbers could never predict: Qwen's
    # si_irrelevant renders to 6,695 tokens -> 8,232 required -> 16384.
    # If this moves, someone changed the SI text, the allowance, or the
    # tokenizer — all of which must re-derive, never inherit.
    if slug == "qwen38_27b":
        assert cfg["max_len_by_si"]["si_irrelevant"] == 16384
        assert cfg["max_len"] == 16384
    # Nemotron's ByteLevel-BPE renders si_constitution to 5,968 tokens ->
    # 7,505 required -> 8192 (the bucket edge: 687 tokens of slack) and
    # si_irrelevant to 6,665 -> 8,202 -> 16384.
    if slug == "nemotron3nano4b":
        assert cfg["max_len_by_si"] == {"safety_refusal": 2048,
                                        "si_constitution": 8192,
                                        "si_irrelevant": 16384}
        assert cfg["max_len"] == 16384


@pytest.mark.parametrize("slug", sorted(MODEL_SPECS))
def test_class_lists_exact_and_ordered(slug):
    cfg = _committed(slug)
    assert cfg["class_a_arms"] == ["psr_kl_all", "psr_mse_all",
                                   "lora_r1_all", "token_faithful", "bidpo"]
    assert cfg["class_c_arms"] == ["psr_kl_all", "psr_mse_all",
                                   "lora_r1_all", "bidpo"]
    # class_c ⊆ class_a with class_a's order preserved (arm-major
    # enumeration depends on it).
    a = cfg["class_a_arms"]
    positions = [a.index(arm) for arm in cfg["class_c_arms"]]
    assert positions == sorted(positions)
    assert set(cfg["class_c_arms"]) <= set(a)


@pytest.mark.parametrize("slug", sorted(MODEL_SPECS))
def test_batch_widths_measured_or_fail_loudly(slug):
    """Every arm's width is either the MEASURED value folded into the
    committed file (QWEN_MEASURED_BS — the Phase-D memprobe landed for
    Qwen 2026-08-26) or the placeholder string, on which
    resolve_batch_size RAISES (int() on the placeholder) instead of
    silently training at the module default. Gemma's Phase-D memprobe has
    not run, so its table is all placeholders — as is GLM-4.6V-Flash's
    (registered 2026-09-02, memprobe pending)."""
    cfg = _committed(slug)
    assert cfg["batch_size_by_si"] == {}
    assert set(cfg["batch_size_by_arm"]) == set(CLASS_A_ARMS)
    expected = (FOLDED[slug].get("batch_size_by_arm")
                or {arm: BATCH_PLACEHOLDER for arm in CLASS_A_ARMS})
    assert cfg["batch_size_by_arm"] == expected
    for si in cfg["si_ids"]:
        for arm in cfg["class_a_arms"]:
            if expected[arm] == BATCH_PLACEHOLDER:
                with pytest.raises(ValueError, match="UNMEASURED"):
                    resolve_batch_size(cfg, si, arm)
            else:
                assert resolve_batch_size(cfg, si, arm) == expected[arm]


@pytest.mark.parametrize("slug", sorted(MODEL_SPECS))
def test_train_kwargs_default_inherited_and_arm_blocks_empty(base, slug):
    """default verbatim from the base; per-arm blocks empty until HPO
    winners land — EXCEPT bidpo's floor_ratio_max 3.0, which is objective
    semantics (DPO preference margin, not KL distillation), not tuning.
    E4B's winners LANDED 2026-08-26 (f451e7d), so gemma additionally
    carries exactly the four folded blocks pinned in E4B_FOLDED_WINNERS;
    Qwen's HPO has not landed and keeps only default + bidpo, and so does
    GLM-4.6V-Flash (no HPO yet)."""
    cfg = _committed(slug)
    assert cfg["train_kwargs"]["default"] == base["train_kwargs"]["default"]
    assert cfg["train_kwargs"]["bidpo"] == {"floor_ratio_max": 3.0}
    folded = FOLDED[slug].get("train_kwargs") or {}
    assert set(cfg["train_kwargs"]) == {"default", "bidpo", *folded}


@pytest.mark.parametrize("slug", sorted(MODEL_SPECS))
def test_pool_block(base, slug):
    """train_families = the five headroom families + the harmful corpus
    (spec Fixed 11) — NOT the base's twelve; per-model store_dir and
    manifest; everything else in the pool block inherited unchanged."""
    cfg = _committed(slug)
    pool = cfg["pool"]
    assert pool["train_families"] == TRAIN_FAMILIES == [
        "bon", "pasttense", "codeattack", "renellm", "grok420",
        "wildjailbreak_harmful"]
    assert pool["store_dir"] == \
        f"$STATE/twosi-model-sweeps/pool/{slug}"
    assert cfg["pool_manifest"] == f"out/two_si/pool_manifest_{slug}.json"
    # all-vLLM pool generation, BOTH halves declared: pool.engine selects for
    # generate-pos/benign, pos_generation_engine cross-checks it, and
    # neg_generation_engine puts the none/prompted side on vLLM (two_si_run's
    # adapter defaults it to hf when unset) — no HF-vs-vLLM engine confound.
    assert pool["engine"] == "vllm"
    assert pool["pos_generation_engine"] == "vllm"
    assert pool["neg_generation_engine"] == "vllm"
    # goal-level holdout, threshold, and the untouched remainder mirror the
    # base exactly.
    assert pool["family_disjointness"] is False
    assert pool["threshold"] == 0.5
    assert cfg["prereg"]["threshold_grid"][1] == 0.5
    assert pool["static_families"] == base["pool"]["static_families"]
    assert pool["eval_families"] == base["pool"]["eval_families"]
    # anti_neg / bidpo_benign are DELIBERATELY not inherited any more. The
    # base was turned off on 2026-08-26 because dataset generation no longer
    # produces an anti-neg or reversed-benign side; twosweep's fit configs
    # still fit `bidpo` (and `bidpo_lora` in the HPO set) and keep them ON.
    # Both sides are self-consistent, so this pair is checked below on its
    # own terms instead of against the base.
    assert cfg["pool"]["anti_neg"]["enabled"] is True, (
        "a fit config that trains a bidpo arm needs its anti-neg input; "
        "the manifest refuses a bidpo with no harmful preferences")
    assert cfg["pool"]["bidpo_benign"]["enabled"] is True
    for key in ("hard_floor_pairs", "fit_floor", "benign_sources",
                "category_holdout"):
        if key == "category_holdout":
            assert cfg["category_holdout"] == base["category_holdout"]
        else:
            assert pool.get(key) == base["pool"].get(key)
    # structural sanity the builder relies on: train ⊆ static.
    assert set(pool["train_families"]) <= set(pool["static_families"])


@pytest.mark.parametrize("slug", sorted(MODEL_SPECS))
def test_train_families_within_registered_vocabulary(slug):
    """Every train family resolves to registered attack machinery: the
    JB_ATTACKS registry keys plus the wildjailbreak_harmful corpus family
    (which is a dataset split, not an attack transform)."""
    from sisteer.eval.attacks import JB_ATTACKS

    vocab = set(JB_ATTACKS) | {"wildjailbreak_harmful"}
    assert set(_committed(slug)["pool"]["train_families"]) <= vocab


@pytest.mark.parametrize("slug", sorted(MODEL_SPECS))
def test_teacher_prefix_cache_off_pending_hybrid_support(slug):
    """Teacher prefix caching is off on purpose: TeacherPrefixCache crashes
    on both models' hybrid cache layouts (measured 2026-08-25 — see
    tests/test_model_onboarding.py), so the derived configs run the exact
    uncached path until prefix_cache.py supports those layouts."""
    assert _committed(slug)["teacher_prefix_cache"] is False


@pytest.mark.parametrize("slug", sorted(MODEL_SPECS))
def test_inherited_shared_knobs(base, slug):
    cfg = _committed(slug)
    for key in ("si_ids", "completion_tokens", "controls", "wandb_fits",
                "wandb_project", "wandb_entity", "grader", "prereg",
                "datasets"):
        assert cfg[key] == base[key], key
    assert cfg["completion_tokens"] == 512


# ---- --hpo-results translation ------------------------------------------------


WINNERS = {
    "psr_kl_all": {"lr": 3e-4, "warmup_frac": 0.05, "max_grad_norm": 1.0},
    "bidpo": {"lr": 1e-5, "warmup_frac": 0.15, "max_grad_norm": 0.5,
              "accum_steps": 16},
}


def test_hpo_results_translation(base):
    cfg = derive_config(base, MODEL_SPECS["gemma4e4b"],
                        FIXTURE_COUNTS["gemma4e4b"], hpo_results=WINNERS)
    tk = cfg["train_kwargs"]
    assert tk["psr_kl_all"] == {"lr": 3e-4, "lr_schedule": "warmup_cosine",
                                "warmup_frac": 0.05, "max_grad_norm": 1.0}
    # bidpo: winner keys merged OVER the semantics block — accum_steps
    # carried, floor_ratio_max 3.0 kept.
    assert tk["bidpo"] == {"floor_ratio_max": 3.0, "lr": 1e-5,
                           "lr_schedule": "warmup_cosine",
                           "warmup_frac": 0.15, "max_grad_norm": 0.5,
                           "accum_steps": 16}
    # untouched arms stay empty; default inherited.
    assert set(tk) == {"default", "bidpo", "psr_kl_all"}
    assert tk["default"] == base["train_kwargs"]["default"]


def test_hpo_translation_refusals():
    with pytest.raises(SystemExit, match="position-split"):
        translate_hpo_results({"psr_kl_all:response": {
            "lr": 1e-4, "warmup_frac": 0.05, "max_grad_norm": 1.0}},
            CLASS_A_ARMS)
    with pytest.raises(SystemExit, match="unknown arm"):
        translate_hpo_results({"cast": {
            "lr": 1e-4, "warmup_frac": 0.05, "max_grad_norm": 1.0}},
            CLASS_A_ARMS)
    with pytest.raises(SystemExit, match="missing"):
        translate_hpo_results({"psr_kl_all": {"lr": 1e-4}}, CLASS_A_ARMS)
    with pytest.raises(SystemExit, match="accum_steps"):
        translate_hpo_results({"bidpo": {
            "lr": 1e-5, "warmup_frac": 0.05, "max_grad_norm": 1.0}},
            CLASS_A_ARMS)
