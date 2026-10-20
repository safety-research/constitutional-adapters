#!/usr/bin/env python
"""Derive the per-model two-SI sweep configs from configs/two_si.yaml.

One deterministic function (`derive_config`) turns the committed base config
plus a model spec plus that model's MEASURED per-SI teacher-prompt token
counts into a complete, self-contained per-model config
(configs/two_si_gemma4e4b.yaml / configs/two_si_qwen38_27b.yaml). The CLI
wraps it with a live-tokenizer measurement (tokenizer + config downloads
only — never weights); tests call the pure function on fixture counts, so
the committed files are pinned byte-for-byte with no network.

What is derived vs inherited:

- model / model_revision pinned; dtype bfloat16; attn_implementation sdpa.
- layer = round(0.469 * n_layers) -> 20 (E4B), 30 (Qwen). bidpo only; the
  all-layer arms ignore it and token_faithful is layer-agnostic. 0.469 is
  the Llama-3.1 anchor ratio (layer 15 of 32) carried across depths.
- max_len_by_si RE-DERIVED per tokenizer via train_budget.derive_max_len
  (completion 512, attack_allowance 1024, eos 1) on each SI's teacher
  prompt re-measured through build_chat_prompt — NEVER the Llama counts
  (5872/5888/6380/6396). Scalar max_len = max over SIs.
- class_a_arms / class_c_arms: exactly the sweep's five/four, in the
  load-bearing arm-major order.
- batch_size_by_arm: NO measured widths exist for these models yet, so
  every arm carries a fail-loud placeholder string — resolve_batch_size
  int()s the value, so any attempt to run a wave before Phase D's memprobe
  table lands raises instead of silently training at the module default.
- train_kwargs.default inherited verbatim; per-arm blocks start EMPTY
  (the base's per-arm blocks are Llama-tuned HPO outcomes and do not
  transfer) except bidpo's floor_ratio_max: 3.0, which is objective
  semantics (a DPO preference margin is not a KL-to-teacher distillation;
  the 2026-08-24 1.0 ruling scoped itself to distillation objectives),
  not a tuned number. --hpo-results merges winners in.
- teacher_prefix_cache: FALSE —
  forced by evidence measured on 2026-08-25 (CPU, tiny real-class
  instances, transformers 5.14.1): qwen3_5's hybrid cache mixes
  LinearAttentionLayer entries with no .keys/.values, so
  TeacherPrefixCache.build crashes at ANY prefix length; gemma4 crashes
  whenever the prefix exceeds sliding_window=512 (35 of E4B's 42 layers
  are sliding_attention), i.e. for si_constitution/si_irrelevant. Both
  failures are loud (build-time AttributeError/RuntimeError), so `true`
  would abort every fit. Flip back per model only after
  extraction/prefix_cache.py learns those cache layouts AND
  onboard_model_smoke.py's prefix-cache equivalence gate passes on the
  pod.
- pool: store_dir -> the per-model mode-0700 store
  (~/.local/state/twosi-model-sweeps/pool/<slug>); train_families -> the
  five headroom families + wildjailbreak_harmful (the base's twelve
  are NOT inherited); pool_manifest -> a counts-only per-model
  manifest. static_families /
  eval_families / goal-level holdout / floors inherited unchanged.
- Everything else (si_ids, completion_tokens, controls, datasets, prereg,
  wandb keys, ...) inherited verbatim from the base.

The committed Llama two_si_budget_measured.json is invalid for these
models BY CONSTRUCTION — measurement_inputs_sha hashes the model pin
(train_budget.py) — so consumption gates refuse it; Phase D re-measures.

--hpo-results translates a winners JSON (arm-keyed:
``{"<arm>": {"lr": ..., "warmup_frac": ..., "max_grad_norm": ...,
"accum_steps": ...}}``) into train_kwargs.<arm> blocks of
``{lr, lr_schedule: warmup_cosine, warmup_frac, max_grad_norm}`` plus
``accum_steps`` for bidpo. NOTE the schema has no per-POSITION slot: HPO
is per (model, arm, position) but train_kwargs is per arm, so a
position-split winner (a key like "psr_kl_all:response") is REFUSED here
— that gap is recorded, not papered over.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO / "configs" / "two_si.yaml"

# The Llama-3.1 anchor: the campaign's single-layer arms sit at layer 15 of
# 32 (= 0.469 of the depth); the same fractional depth is carried to each
# new model and rounded.
LAYER_FRAC = 0.469

# Sweep arm sets (spec Fixed 3/5; order is LOAD-BEARING for arm-major
# enumeration — keep it).
CLASS_A_ARMS = ["psr_kl_all", "psr_mse_all", "lora_r1_all",
                "token_faithful", "bidpo"]
CLASS_C_ARMS = ["psr_kl_all", "psr_mse_all", "lora_r1_all", "bidpo"]

# Spec Fixed 11: the five headroom families + the undefended harmful
# corpus. Deliberately NOT the base config's twelve.
TRAIN_FAMILIES = ["bon", "pasttense", "codeattack", "renellm", "grok420",
                  "wildjailbreak_harmful"]

# The fail-loud batch placeholder: resolve_batch_size does
# int(by_arm[arm]), so this string raises ValueError (naming itself) the
# moment any wave tries to resolve a width before Phase D's measured table
# replaces it.
BATCH_PLACEHOLDER = "UNMEASURED_RUN_PHASE_D_MEMPROBE_FIRST"

# derive_max_len inputs (the base config's documented arithmetic, applied
# per tokenizer): shared completion cap, attack-prompt allowance, EOS.
COMPLETION_CAP = 512
ATTACK_ALLOWANCE = 1024
EOS = 1

STORE_ROOT = "$STATE/twosi-model-sweeps/pool"


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    model: str
    revision: str
    n_layers: int
    hidden_size: int

    @property
    def config_path(self) -> Path:
        return REPO / "configs" / f"two_si_{self.slug}.yaml"

    @property
    def layer(self) -> int:
        return round(LAYER_FRAC * self.n_layers)


# Pins verified against the HF API on 2026-08-25 (both ungated).
MODEL_SPECS: dict[str, ModelSpec] = {
    "gemma4e4b": ModelSpec(
        slug="gemma4e4b",
        model="google/gemma-4-E4B-it",
        revision="ee0ef6023621cff504d758262d4e04895a5af4a2",
        n_layers=42,
        hidden_size=2560,
    ),
    "qwen38_27b": ModelSpec(
        slug="qwen38_27b",
        model="Qwen/Qwen3.8-27B",
        revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        n_layers=64,
        hidden_size=5120,
    ),
    # sdf-base-steering BASE models: registered here so
    # scripts/onboard_model_smoke.py resolves them (--model-config on the
    # sdfbase configs matches these pins). Their committed configs are the
    # GENERATED configs/sdfbase_<slug>_fits.yaml
    # (scripts/sdfbase/make_sdfbase_configs.py) — NOT two_si_<slug>.yaml,
    # so this script's own two_si derivation must never be pointed at
    # these slugs. layer = round(0.469 x n_layers): 20 / 15.
    "gemma4e4b_base": ModelSpec(
        slug="gemma4e4b_base",
        model="google/gemma-4-E4B",
        revision="411aa17b749aa952df1359d2dcea73917a544d9a",
        n_layers=42,
        hidden_size=2560,
    ),
    "qwen35_4b_base": ModelSpec(
        slug="qwen35_4b_base",
        model="Qwen/Qwen3.5-4B-Base",
        revision="1001bb4d826a52d1f399e183466143f4da7b741b",
        n_layers=32,
        hidden_size=2560,
    ),
    # GLM-4.6V-Flash (zai-org; MIT, ungated; pin verified against the HF
    # API 2026-09-02). A dense VLM — 10.3B total, ~9B text decoder + 0.9B
    # ViT — used TEXT-ONLY here. text_config: 40 layers, hidden 4096, GQA
    # 32/2, full attention everywhere (a standard DynamicCache — none of
    # the hybrid/sliding layouts the teacher_prefix_cache deviation below
    # was measured on; the derivation still emits false uniformly, and
    # flipping it is gated on the onboarding equivalence check on GPU).
    # model_type glm4v has NO causal-LM auto-mapping in transformers
    # 5.14.1 — models.auto_model_class loads it as
    # Glm4vForConditionalGeneration; the decoder resolves through
    # _LAYER_PATHS' model.language_model.layers. Its chat template closes
    # assistant turns with no scaffold (chat_markers' successor-bounded
    # shape), carries enable_thinking (load_tokenizer pins it off ->
    # header <|assistant|>\n<think></think>\n, H=5), and accepts the
    # system role natively. layer = round(0.469 x 40) = 19.
    "glm46vflash": ModelSpec(
        slug="glm46vflash",
        model="zai-org/GLM-4.6V-Flash",
        revision="411bb4d77144a3f03accbf4b780f5acb8b7cde4e",
        n_layers=40,
        hidden_size=4096,
    ),
    # The SAME checkpoint's sdf-base-steering (doc-CE) entry. Named `_doc`,
    # NOT `_base`: no base (pretrained-only) variant of GLM-4.6V exists, so
    # the psr_ce_all / lora_r4_ce document fits target the chat model
    # itself (the flat-document rendering is unchanged — data_mode:
    # documents never applies a chat template). Committed config =
    # configs/sdfbase_glm46vflash_doc_fits.yaml (make_sdfbase_configs.py);
    # this script refuses the slug like the other SDFBASE_SLUGS. Same
    # pin as glm46vflash above — onboard_model_smoke's --model-config
    # resolution disambiguates the two by which committed config it was
    # handed.
    "glm46vflash_doc": ModelSpec(
        slug="glm46vflash_doc",
        model="zai-org/GLM-4.6V-Flash",
        revision="411bb4d77144a3f03accbf4b780f5acb8b7cde4e",
        n_layers=40,
        hidden_size=4096,
    ),
    # NVIDIA Nemotron-3-Nano-4B (2026-09-02; pin verified against the HF
    # API; NVIDIA Open Model License, NOT gated; "-BF16" is the repo id —
    # there is no un-suffixed 4B repo). NemotronH dense HYBRID: 42 blocks
    # = 21 Mamba2 + 17 MLP + 4 attention (indices 12, 17, 24, 32),
    # hidden 3136, ChatML template, <|im_end|> (id 11) terminator.
    # layer = round(0.469 x 42) = 20 is an MLP block. That is FINE for
    # this sweep's single-layer arm: bidpo is a block-OUTPUT steering
    # vector (extraction/bidpo.py trains one d_model vector through
    # hooks.steering; methods.ARMS builds it as a VectorObject), and every
    # NemotronH block — Mamba2, MLP, attention — carries that hook, so 20
    # stands. What CANNOT sit at 20 is a single-layer o_proj arm (lora_r1 /
    # bidpo_lora = RankOneLoRAObject): models.attention_oproj refuses it
    # loudly, naming the four attention blocks. Neither is in CLASS_A/C
    # here; tests/test_two_si_layer_targets.py pins that any two_si config
    # advertising one targets an attention layer of ITS model.
    "nemotron3nano4b": ModelSpec(
        slug="nemotron3nano4b",
        model="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        revision="dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
        n_layers=42,
        hidden_size=3136,
    ),
    # sdf-base-steering INSTRUCT-DIRECT lanes (2026-09-03): the two
    # instruct checkpoints the transfer battery serves, trained on the
    # corpus DIRECTLY (doc-CE, no chat template applied in training). Same
    # pins as the transfer configs' model_revision — gemma4e4b_it_doc pins
    # the SAME checkpoint as the two_si slug gemma4e4b above, so
    # onboard_model_smoke's --model-config resolution settles the two by
    # the committed config file, exactly as for glm46vflash(_doc). Qwen
    # 3.5-4B (instruct) has no two_si spec here; architecture facts = the
    # base sibling's (32 layers, hidden 2560; 42 / 2560 for gemma).
    # Configs are GENERATED (make_sdfbase_configs.py), never derived here.
    "gemma4e4b_it_doc": ModelSpec(
        slug="gemma4e4b_it_doc",
        model="google/gemma-4-E4B-it",
        revision="ee0ef6023621cff504d758262d4e04895a5af4a2",
        n_layers=42,
        hidden_size=2560,
    ),
    "qwen35_4b_it_doc": ModelSpec(
        slug="qwen35_4b_it_doc",
        model="Qwen/Qwen3.5-4B",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        n_layers=32,
        hidden_size=2560,
    ),
    # sdf-base-steering FULL-SCALE lanes of the two chat checkpoints
    # (2026-09-06, run on Slurm): the SAME pins as the two_si nemotron3nano4b
    # spec and glm46vflash_doc above — sibling configs under `_full` keys so
    # the `_doc` configs' bytes stayed untouched (the Nemotron `_doc` data-
    # scaling ladder itself was retired 2026-10-08; the `_full` lane is the
    # paper's). Configs are GENERATED (make_sdfbase_configs.py).
    "nemotron3nano4b_doc_full": ModelSpec(
        slug="nemotron3nano4b_doc_full",
        model="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        revision="dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
        n_layers=42,
        hidden_size=3136,
    ),
    "glm46vflash_doc_full": ModelSpec(
        slug="glm46vflash_doc_full",
        model="zai-org/GLM-4.6V-Flash",
        revision="411bb4d77144a3f03accbf4b780f5acb8b7cde4e",
        n_layers=40,
        hidden_size=4096,
    ),
    # the `_full` lane's gated-arm refit config (2026-09-07; same checkpoint)
    "glm46vflash_doc_full_r2t12": ModelSpec(
        slug="glm46vflash_doc_full_r2t12",
        model="zai-org/GLM-4.6V-Flash",
        revision="411bb4d77144a3f03accbf4b780f5acb8b7cde4e",
        n_layers=40,
        hidden_size=4096,
    ),
    # sdf-base-steering Llama-3.1-8B lane (2026-09-09): the ungated
    # NousResearch mirrors, base AND instruct, both dense LlamaForCausalLM
    # (32 layers, hidden 4096, vocab 128,256; pins verified against the HF
    # API 2026-09-09). layer = round(0.469 x 32) = 15. The instruct pin is
    # the campaign's canonical `llama8b` serving checkpoint (the two_si
    # llama configs pin the same model + revision, so --model-config
    # resolution settles the two by the committed config file name, exactly
    # as for glm46vflash(_doc)). Configs are GENERATED
    # (make_sdfbase_configs.py), never derived here.
    "llama8b_base": ModelSpec(
        slug="llama8b_base",
        model="NousResearch/Meta-Llama-3.1-8B",
        revision="1f47e50cdbe801ad8a5174156ec3a0655108fb9f",
        n_layers=32,
        hidden_size=4096,
    ),
    "llama8b_it_doc": ModelSpec(
        slug="llama8b_it_doc",
        model="NousResearch/Meta-Llama-3.1-8B-Instruct",
        revision="d10aef7999a2b5ba950ab3974312feeedbfe0b77",
        n_layers=32,
        hidden_size=4096,
    ),
    # sdf-base-steering instruct-direct BATCH-64 lanes (2026-09-11): the
    # SAME pins as gemma4e4b_it_doc / qwen35_4b_it_doc above, as SIBLING
    # keys `_eb64` so the frozen `_it_doc` configs' bytes (their four
    # published cells' fit identity) never move. Configs are GENERATED
    # (make_sdfbase_configs.py), never derived here.
    "gemma4e4b_it_doc_eb64": ModelSpec(
        slug="gemma4e4b_it_doc_eb64",
        model="google/gemma-4-E4B-it",
        revision="ee0ef6023621cff504d758262d4e04895a5af4a2",
        n_layers=42,
        hidden_size=2560,
    ),
    "qwen35_4b_it_doc_eb64": ModelSpec(
        slug="qwen35_4b_it_doc_eb64",
        model="Qwen/Qwen3.5-4B",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        n_layers=32,
        hidden_size=2560,
    ),
    # sdf-base-steering qwen-BASE batch-64 SEED REPLICATES (2026-09-15): the
    # SAME pin as qwen35_4b_base above, as SIBLING keys `_eb64_s<seed>` so
    # the frozen `_base` config's bytes (its four published batch-8 seed-0
    # cells' fit identity) never move. Configs are GENERATED
    # (make_sdfbase_configs.py), never derived here.
    "qwen35_4b_base_eb64_s1": ModelSpec(
        slug="qwen35_4b_base_eb64_s1",
        model="Qwen/Qwen3.5-4B-Base",
        revision="1001bb4d826a52d1f399e183466143f4da7b741b",
        n_layers=32,
        hidden_size=2560,
    ),
    "qwen35_4b_base_eb64_s2": ModelSpec(
        slug="qwen35_4b_base_eb64_s2",
        model="Qwen/Qwen3.5-4B-Base",
        revision="1001bb4d826a52d1f399e183466143f4da7b741b",
        n_layers=32,
        hidden_size=2560,
    ),
    "qwen35_4b_base_eb64_s3": ModelSpec(
        slug="qwen35_4b_base_eb64_s3",
        model="Qwen/Qwen3.5-4B-Base",
        revision="1001bb4d826a52d1f399e183466143f4da7b741b",
        n_layers=32,
        hidden_size=2560,
    ),
    # The WORKED EXAMPLE's fit config (2026-10-08): the paper's Qwen3.5-4B base
    # checkpoint under the demo key `qwen35_4b_demo_base` (reproduce/models.yaml
    # `qwen35_4b_demo`, fit.checkpoint.model_id) — the SAME pin as qwen35_4b_base,
    # its own config (configs/sdfbase_qwen35_4b_demo_base_fits.yaml) and pool dir
    # (/vol/pool/qwen35_4b_demo_base/), so a reader's fresh fits never land beside
    # the paper's committed objects. Architecture facts = the base config's (32
    # layers, hidden 2560). Config GENERATED; configured, not run.
    "qwen35_4b_demo_base": ModelSpec(
        slug="qwen35_4b_demo_base",
        model="Qwen/Qwen3.5-4B-Base",
        revision="1001bb4d826a52d1f399e183466143f4da7b741b",
        n_layers=32,
        hidden_size=2560,
    ),
}

#: MODEL_SPECS entries that belong to the sdf-base-steering campaign: they
#: exist for onboard_model_smoke.py's pin resolution, and this script's
#: two_si derivation REFUSES them (their configs come from
#: scripts/sdfbase/make_sdfbase_configs.py).
SDFBASE_SLUGS = ("gemma4e4b_base", "qwen35_4b_base", "glm46vflash_doc",
                 "gemma4e4b_it_doc",
                 "qwen35_4b_it_doc", "nemotron3nano4b_doc_full",
                 "glm46vflash_doc_full", "glm46vflash_doc_full_r2t12",
                 "llama8b_base", "llama8b_it_doc",
                 "gemma4e4b_it_doc_eb64", "qwen35_4b_it_doc_eb64",
                 # the qwen-base batch-64 seed replicates (2026-09-15)
                 "qwen35_4b_base_eb64_s1", "qwen35_4b_base_eb64_s2",
                 "qwen35_4b_base_eb64_s3",
                 # the worked example's demo lane (2026-10-08) — APPENDED (the
                 # order is pinned by tests/test_sdfbase_llama8b_lane.py)
                 "qwen35_4b_demo_base")


def load_base_config(path: Path = BASE_CONFIG) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def si_texts() -> dict[str, str]:
    """The three SI texts, keyed by si_id (the base config's si_ids)."""
    from sisteer.data.instructions import SAFETY_SI
    from sisteer.data.safety_texts import CONSTITUTION_PROMPT, IRRELEVANT_PROMPT

    return {
        "safety_refusal": SAFETY_SI.text,
        "si_constitution": CONSTITUTION_PROMPT,
        "si_irrelevant": IRRELEVANT_PROMPT,
    }


def measure_si_prompt_tokens(tok) -> dict[str, int]:
    """Each SI rendered as a TEACHER chat prompt (system = the SI text,
    empty user turn) through build_chat_prompt, counted with
    add_special_tokens=True — the exact convention the base config's
    max_len arithmetic and tests/test_constitution_si.py use."""
    from sisteer.models import build_chat_prompt

    return {
        si: len(tok(build_chat_prompt(tok, "", system=text),
                    add_special_tokens=True)["input_ids"])
        for si, text in si_texts().items()
    }


def translate_hpo_results(winners: dict, arms: list[str]) -> dict[str, dict]:
    """HPO winners -> train_kwargs.<arm> blocks.

    Input: ``{arm: {lr, warmup_frac, max_grad_norm[, accum_steps]}}``.
    Output blocks carry exactly {lr, lr_schedule: warmup_cosine,
    warmup_frac, max_grad_norm} (+ accum_steps for bidpo — spec Fixed 4's
    effective-batch-via-accumulation lives there). Unknown arms and
    position-split keys refuse loudly.
    """
    out: dict[str, dict] = {}
    for arm, win in winners.items():
        if ":" in arm or "@" in arm:
            raise SystemExit(
                f"[derive-model-config] HPO winner key {arm!r} is "
                "position-split, but train_kwargs has no per-position slot "
                "— the schema gap is recorded in this script's docstring. "
                "Resolve it (one config per position leg, or a schema "
                "extension) before translating this winner.")
        if arm not in arms:
            raise SystemExit(
                f"[derive-model-config] HPO winner for unknown arm {arm!r}; "
                f"this sweep's arms are {arms}")
        missing = [k for k in ("lr", "warmup_frac", "max_grad_norm")
                   if k not in win]
        if missing:
            raise SystemExit(
                f"[derive-model-config] HPO winner for {arm!r} is missing "
                f"{missing} (got {sorted(win)})")
        block = {
            "lr": float(win["lr"]),
            "lr_schedule": "warmup_cosine",
            "warmup_frac": float(win["warmup_frac"]),
            "max_grad_norm": float(win["max_grad_norm"]),
        }
        if arm == "bidpo" and "accum_steps" not in win:
            raise SystemExit(
                "[derive-model-config] bidpo's HPO winner must carry "
                "accum_steps (spec Fixed 4: effective batch >= 128 via "
                "gradient accumulation)")
        if "accum_steps" in win:
            block["accum_steps"] = int(win["accum_steps"])
        out[arm] = block
    return out


def derive_config(base: dict, spec: ModelSpec, si_counts: dict[str, int],
                  hpo_results: dict | None = None) -> dict:
    """The pure derivation: base config + model spec + measured per-SI
    teacher-prompt token counts -> the per-model config dict. Deterministic
    — same inputs, same output — so tests regenerate and compare."""
    from sisteer.eval.train_budget import derive_max_len

    cfg = dict(base)  # top-level copy; nested blocks replaced explicitly

    missing = [si for si in base["si_ids"] if si not in si_counts]
    if missing:
        raise SystemExit(
            f"[derive-model-config] no measured token count for {missing}; "
            f"got counts for {sorted(si_counts)}")

    cfg["model"] = spec.model
    cfg["model_revision"] = spec.revision
    cfg["dtype"] = "bfloat16"
    cfg["attn_implementation"] = "sdpa"
    cfg["layer"] = spec.layer

    max_len_by_si = {
        si: derive_max_len(si_counts[si], COMPLETION_CAP,
                           attack_allowance=ATTACK_ALLOWANCE, eos=EOS)
        for si in base["si_ids"]
    }
    cfg["max_len_by_si"] = max_len_by_si
    cfg["max_len"] = max(max_len_by_si.values())

    cfg["class_a_arms"] = list(CLASS_A_ARMS)
    cfg["class_c_arms"] = list(CLASS_C_ARMS)

    cfg["batch_size_by_si"] = {}
    cfg["batch_size_by_arm"] = {arm: BATCH_PLACEHOLDER
                                for arm in CLASS_A_ARMS}

    train_kwargs: dict = {"default": dict(base["train_kwargs"]["default"])}
    # Objective semantics, not tuning: DPO preference margins keep the
    # historical 3.0 degeneracy backstop (the 1.0 ruling scoped itself to
    # distillation objectives — see the base config's train_kwargs notes).
    train_kwargs["bidpo"] = {"floor_ratio_max": 3.0}
    if hpo_results:
        for arm, block in translate_hpo_results(
                hpo_results, CLASS_A_ARMS).items():
            merged = dict(train_kwargs.get(arm, {}))
            merged.update(block)
            train_kwargs[arm] = merged
    cfg["train_kwargs"] = train_kwargs

    # Deviation from `true`, forced by measured evidence — see the
    # module docstring. Loud either way: `true` crashes at
    # TeacherPrefixCache.build on both models' hybrid cache layouts.
    cfg["teacher_prefix_cache"] = False

    pool = dict(base["pool"])
    # engine: vllm — pool generation is all-vLLM (this supersedes the earlier
    # HF-only rule, once the HF/vLLM equivalence failure was resolved by
    # standardising on vLLM). build_two_si_pool defaults pos/benign to HF when
    # pool.engine is absent, so this must be set explicitly or those legs
    # generate on the wrong engine silently; HF stays the recorded per-model
    # contingency if a model fails the pilot vLLM gate.
    pool["engine"] = "vllm"
    # BOTH generation halves declared vllm EXPLICITLY, not merely inherited:
    # pos_generation_engine cross-checks generate-pos's selected engine
    # (_resolve_engine refuses a contradiction), and neg_generation_engine is
    # what two_si_run's ed01306 adapter reads to put the none/prompted side on
    # vLLM (it defaults to hf when unset). Pinning them here keeps the sweep
    # configs all-vLLM even if the base flips its defaults — an HF none-side
    # against a vLLM pos-side is the +25.8pp engine confound this closed.
    pool["pos_generation_engine"] = "vllm"
    pool["neg_generation_engine"] = "vllm"
    # anti-neg + reversed-benign PINNED ON, not inherited — the same reasoning
    # as the engine keys above, and pinned here for the same reason: the base
    # flipped these OFF on 2026-08-26 for DATASET GENERATION, which produces
    # no anti-neg side. These are FIT configs and they train a `bidpo` arm
    # (plus `bidpo_lora` in the HPO set), and the manifest refuses a bidpo
    # with no harmful preferences — so inheriting a base that has since turned
    # generation-oriented would silently break every fit derived here. The
    # split is deliberate: the base stays off and the fit configs keep their
    # own on.
    pool["anti_neg"] = {**base["pool"]["anti_neg"], "enabled": True}
    pool["bidpo_benign"] = {**base["pool"]["bidpo_benign"], "enabled": True}
    pool["store_dir"] = f"{STORE_ROOT}/{spec.slug}"
    pool["train_families"] = list(TRAIN_FAMILIES)
    cfg["pool"] = pool

    # Per-model counts-only manifest — two models building on one checkout
    # must not overwrite each other's manifests.
    cfg["pool_manifest"] = f"out/two_si/pool_manifest_{spec.slug}.json"

    return cfg


def render_config(cfg: dict, spec: ModelSpec, si_counts: dict[str, int],
                  counts_provenance: str) -> str:
    """The committed YAML text: a deterministic provenance header + the
    derived mapping. No timestamps — same inputs, same bytes."""
    lines = [
        f"# two_si_{spec.slug}.yaml — DERIVED, do not hand-edit.",
        "# Regenerate: uv run --no-sync python scripts/derive_model_config.py"
        f" --models {spec.slug}",
        "# (--hpo-results <winners.json> to fold HPO winners into"
        " train_kwargs.<arm>.)",
        "#",
        f"# model:    {spec.model} @ {spec.revision}",
        f"# layer:    round({LAYER_FRAC} x {spec.n_layers}) = {spec.layer}"
        " (bidpo only; all-layer arms ignore it)",
        "#",
        "# max_len_by_si derived per SI from THIS tokenizer"
        " (train_budget.derive_max_len:",
        f"#   rendered teacher prompt + {ATTACK_ALLOWANCE} attack allowance"
        f" + {COMPLETION_CAP} completion + {EOS} EOS, next pow2):",
    ]
    from sisteer.eval.train_budget import derive_max_len
    for si, n in si_counts.items():
        need = n + ATTACK_ALLOWANCE + COMPLETION_CAP + EOS
        got = derive_max_len(n, COMPLETION_CAP,
                             attack_allowance=ATTACK_ALLOWANCE, eos=EOS)
        lines.append(f"#   {si}: {n} rendered -> {need} required -> {got}")
    lines += [
        f"# counts measured: {counts_provenance}",
        "#",
        "# batch_size_by_arm: PLACEHOLDERS on purpose — no width for this",
        "# model is measured yet, and resolve_batch_size int()s the value,",
        "# so a wave submitted before Phase D's memprobe table lands FAILS",
        "# LOUDLY instead of silently training at the module default. The",
        "# committed Llama two_si_budget_measured.json is invalid here by",
        "# construction (measurement_inputs_sha hashes the model pin), so",
        "# the consumption gates refuse it too.",
        "#",
        "# teacher_prefix_cache: false — DEVIATION from plan §WS3.2's true,",
        "# forced by evidence (2026-08-25, CPU, tiny real-class instances,",
        "# transformers 5.14.1): TeacherPrefixCache crashes on qwen3_5's",
        "# LinearAttentionLayer cache entries (any prefix) and on gemma4's",
        "# DynamicSlidingWindowLayer once the prefix exceeds",
        "# sliding_window=512 (35/42 E4B layers slide; the long SIs render",
        "# ~6k tokens). Flip per model only after prefix_cache.py supports",
        "# the layout AND onboard_model_smoke.py's equivalence gate passes",
        "# on the pod. Tests: tests/test_model_onboarding.py pins the",
        "# never-silently-wrong contract.",
        "",
    ]
    body = yaml.safe_dump(cfg, sort_keys=False, width=79,
                          default_flow_style=False, allow_unicode=True)
    return "\n".join(lines) + body


def _live_counts(spec: ModelSpec) -> dict[str, int]:
    from sisteer.models import load_tokenizer

    tok = load_tokenizer(spec.model, revision=spec.revision)
    return measure_si_prompt_tokens(tok)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", choices=sorted(MODEL_SPECS),
                    default=[s for s in sorted(MODEL_SPECS)
                             if s not in SDFBASE_SLUGS],
                    help="which per-model configs to (re)write (the "
                         "sdfbase base-model slugs refuse: their configs "
                         "are generated by make_sdfbase_configs.py)")
    ap.add_argument("--counts", type=Path, default=None,
                    help="offline override: JSON {slug: {si_id: tokens}} "
                         "instead of measuring the live tokenizer")
    ap.add_argument("--hpo-results", type=Path, default=None,
                    help="JSON {arm: {lr, warmup_frac, max_grad_norm"
                         "[, accum_steps]}} of HPO winners to fold into "
                         "train_kwargs.<arm> (single --models required)")
    ap.add_argument("--print-counts", action="store_true",
                    help="measure and print the per-SI token counts as "
                         "JSON (fixture provenance), write nothing")
    args = ap.parse_args(argv)

    if args.hpo_results and len(args.models) != 1:
        ap.error("--hpo-results applies to exactly one --models slug")

    base = load_base_config()
    counts_by_slug: dict[str, dict[str, int]] = {}
    if args.counts:
        counts_by_slug = json.loads(args.counts.read_text())
        provenance = f"supplied via --counts {args.counts}"
    else:
        provenance = ("live tokenizer at the pinned revision "
                      "(scripts/derive_model_config.py)")

    hpo = (json.loads(args.hpo_results.read_text())
           if args.hpo_results else None)

    out = {}
    for slug in args.models:
        if slug in SDFBASE_SLUGS:
            raise SystemExit(
                f"[derive-model-config] {slug} is an sdf-base-steering "
                "model: its config is configs/sdfbase_"
                f"{slug}_fits.yaml, generated by scripts/sdfbase/"
                "make_sdfbase_configs.py — this script must not write a "
                f"two_si_{slug}.yaml for it")
        spec = MODEL_SPECS[slug]
        counts = counts_by_slug.get(slug) or _live_counts(spec)
        if args.print_counts:
            out[slug] = counts
            continue
        cfg = derive_config(base, spec, counts, hpo_results=hpo)
        text = render_config(cfg, spec, counts, provenance)
        spec.config_path.write_text(text)
        print(f"[derive-model-config] wrote {spec.config_path} "
              f"(layer {spec.layer}, max_len {cfg['max_len']}, "
              f"max_len_by_si {cfg['max_len_by_si']})")
    if args.print_counts:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
