"""Paper-side mechanism: the derived paper flag, the long frame, ruling expiry
evaluation with evidence validity (S11), and the roster-disposition refusal.

Roster / rulings SEEDING is not done here; this module is the mechanism, exercised on synthetic rosters.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import derive as DV
from . import records as R
from . import vocab as V
from .identity import canonical_json


class RulingIndeterminate(RuntimeError):
    """A ruling's evidence is incoherent or stale: the covered keys cannot be selected or excluded."""


class RosterGap(RuntimeError):
    """A roster entry has neither a judged run nor a disposition: the build refuses, naming the entry."""


class LongColumnUnsupplied(RuntimeError):
    """A name in the caller's long-schema column list that the registry cannot supply: a refusal naming it,
    never a silent NaN column."""


# --------------------------------------------------------------------------------------------
# Legacy spellings (the ONLY place the paper's own words are spelled in code; `configs/registry/roster_r2_v1.yaml`
# carries the same tables under `vocabulary:` and tests/test_registry_roster.py asserts they agree)
# --------------------------------------------------------------------------------------------

#: model_id → the paper's model key (`paperlib.config.MODELS` keys).
MODEL_LEGACY: Dict[str, str] = {
    "gemma4e4b_it": "gemma4e4b", "qwen35_4b_it": "qwen35_4b", "nemotron3nano4b": "nemotron3nano4b", "llama8b_it": "llama8b",
    "glm46vflash": "glm46vflash",
}
#: paper_label → the paper's object key (`paperlib.config.ARMS[*].key`).
PAPER_LABEL_LEGACY: Dict[str, str] = {
    "none": "none", "safety_si": "si_safety", "constitution_si": "si_constitution", "constitution_si_full": "si_constitution_full",
    "control_lora": "ctl_lora", "constitution_lora": "const_lora", "difference_lora": "diff_lora", "control_relu": "ctl_psr",
    "constitution_relu": "const_psr", "difference_relu": "diff_psr", "difference_relu_distilled": "diff_psr_distilled", "arditi": "arditi",
    "alphasteer_rebake": "alphasteer_rebake",
    "alphasteer_as_fit": "alphasteer_as_fit", "safety_lora": "safety_lora", "safety_relu": "safety_psr",
    "alphasteer_refit_reference": "alphasteer_refit_reference",
}
#: paper_label → the paper's object family (`Arm.family`).
FAMILY_LEGACY: Dict[str, str] = {
    "none": "none", "safety_si": "prompted", "constitution_si": "prompted", "constitution_si_full": "prompted",
    "arditi": "baseline", "alphasteer_rebake": "baseline", "alphasteer_as_fit": "baseline", "alphasteer_refit_reference": "baseline",
    "control_lora": "control", "control_relu": "control", "constitution_lora": "constitution", "constitution_relu": "constitution",
    "difference_lora": "difference", "difference_relu": "difference", "difference_relu_distilled": "difference", "safety_lora": "safety-distilled",
    "safety_relu": "safety-distilled",
}
#: object_kind → the paper's object type (`Arm.object_type`).
OBJECT_TYPE_LEGACY: Dict[str, str] = {
    "none": "none", "system_instruction": "prompt", "lora": "lora", "lora_subtraction": "lora", "relu_steer": "vector",
    "relu_steer_distilled_difference": "vector", "vector": "vector", "vector_all": "vector", "diffmean": "vector",
    "static_r_arditi": "vector", "caa_static_r": "vector", "alphasteer": "matrix", "antidote": "vector", "composed": "vector",
    "stacked": "vector",
}
#: eval_id → the paper's eval id (`paperlib.config.EVALS`).
EVAL_LEGACY: Dict[str, str] = {
    "strongreject_bon": "bon", "or_bench_hard": "or_bench_hard_1k", "bigcodebench": "bigcodebench_200", "bfcl": "bfcl_200",
    "petri_cyber_bio_chem_capabilities": "cyber_bio_chem_capabilities", "petri_defer_to_users": "defer_to_users",
    "petri_delusion_sycophancy": "delusion_sycophancy", "petri_instruction_evasion": "instruction_evasion",
    "petri_reasoning_unfaithfulness": "reasoning_unfaithfulness", "petri_self_preservation": "self_preservation",
}
#: eval_category → the paper's component (`paperlib.config.EVAL_COMPONENT` values).
COMPONENT_LEGACY: Dict[str, str] = {
    "static_attack": "static", "best_of_n": "bon", "multi_turn": "multi-turn", "over_refusal": "over-refusal", "petri": "petri",
    "agentic": "misalignment", "misalignment": "misalignment", "capability": "capability",
}
TRANSFER_KIND_LEGACY: Dict[str, str] = {
    "base_to_instruct": "base→instruct", "instruct_direct": "instruct-direct", "chat_checkpoint": "chat-ckpt",
    "prompted_only": "prompted-only", "none": "none", "not_applicable": "em-organism",
}

# --------------------------------------------------------------------------------------------
# The READERS' identity spellings (every legacy-adapted source BREAKS on key spelling with
# equal values). The paper's readers are the contract; the tables below reproduce, per column, the spelling each reader files,
# in the registry's nouns. Two mechanisms: (1) the declared readers' adapter parks the reader's OWN words on the run
# (`run.extra_json.paper_arm / paper_family / paper_object_type / paper_lane_kind / paper_component / paper_eval_id /
# judge_legacy / paper_dose`; the r2 adapter parks `judge_legacy` and the export's `lane_kind`) and the chains read them FIRST;
# (2) where an adapter reads a store directly and no reader word rides the run (sampled_lane, sampled_lane_controls, the
# export's fitted objects) the derivations below spell the key the reader would have — from the campaign cell, the view and the
# judge row — never a value.
# --------------------------------------------------------------------------------------------

#: campaign_cell_id_legacy → the registry paper_label the cell's object carries in the paper (legacy_spellings.yaml `paper_labels`,
#: keyed by the cell). The objects adapter labels a fitted object `other` when its fit sidecar names no paper label, so the
#: paper's arm key (`config.ARM_BY_CELL[cell].key` in every reader) is spelled from the CELL the campaign served it under.
CAMPAIGN_ID_PAPER_LABEL_LEGACY: Dict[str, str] = {
    "none": "none", "prompted__safety_refusal": "safety_si", "prompted__si_constitution": "constitution_si",
    "prompted__si_constitution_full": "constitution_si_full",
    "constv2ctl2-full__lora_r4_ce": "control_lora", "constv2-mix-full__lora_r4_ce": "constitution_lora",
    "constv2-mix-full__lora_r4_ce_minus_ctl": "difference_lora",
    "constv2ctl2-full__psr_ce_all": "control_relu", "constv2-mix-full__psr_ce_all": "constitution_relu",
    # all five families: the undistilled paired write IS difference_relu; the distilled student is its own label
    "constv2-mix-full__psr_ce_all_minus_ctl_distilled": "difference_relu_distilled",
    "constv2-mix-full__psr_ce_all_minus_ctl": "difference_relu",
    "safety_refusal__lora_r4": "safety_lora", "safety_refusal__psr_kl_all": "safety_relu",
    "arditi": "arditi", "alphasteer": "alphasteer_rebake",
}
#: view_legacy → the paper lane kind the readers file on EVERY row of the view, the unsteered and prompted rows included
#: (`paperlib.config.MODELS[*].lane_kind` / `transfer_view`, `lane_llama8b.VIEWS`, the strength lane's `llama8b_it_doc`). The
#: registry's `transfer_kind` is the OBJECT's (none / prompted_only on those rows); the readers file the CELL's lane. A view
#: with a dose suffix (`gemma4e4b_it_d0p25`, `llama8b_dm1`, `qwen35_4b_halfdose`) is the base view's lane — see `_served_key_base`.
SERVED_KEY_TRANSFER_LEGACY: Dict[str, str] = {
    "gemma4e4b": "base→instruct", "qwen35_4b": "base→instruct", "llama8b": "base→instruct", "llama8b_base": "base→instruct",
    "llama8b_main": "base→instruct",
    "gemma4e4b_it": "instruct-direct", "qwen35_4b_it": "instruct-direct", "llama8b_it": "instruct-direct", "llama8b_it_doc": "instruct-direct",
    "llama8b_shortconst": "instruct-direct", "llama8b_fullconst": "instruct-direct",
    "nemotron3nano4b": "chat-ckpt", "glm46vflash": "chat-ckpt",
}
#: The r2 export files a view's lane itself (`extra.lane_kind`, the parquet's column) and the reader keeps it — except the views it
#: re-stamps to the paper lane (`config.R2_VIEW_TO_PAPER_LANE` at the pin: the llama shortconst view IS llama's paper lane for the
#: constitution-SI arm; fullconst stays where the export files it).
SOURCE_SERVED_KEY_TRANSFER_LEGACY: Dict[str, Dict[str, str]] = {"r2_export": {"llama8b_shortconst": "instruct-direct"}}
#: registry metric_id → the paper's metric spelling where the registry id folds a best-of-N n the paper keys in `bon_n`
#: (the r2 export's best-of-N curve rows: `asr` at bon_n 1..64). The paper's `metric` is otherwise the registry metric id — never the
#: producer's spelling (`attack_metrics.asr[0.5]`, `by_category_group.held_in`, `behaviors.<b>.mean`), which no reader files. The
#: export spells `asr_bon_cap4096` and the `_bon_512` degeneracy ids ITSELF, so those stay as the registry ids (verified against the
#: c53ae173 pin: 180 rows spell `asr_bon_cap4096`).
METRIC_LEGACY: Dict[str, str] = {f"asr_at_n{n}": "asr" for n in (1, 2, 4, 8, 16, 32, 64)}
#: the 4 live `asr_at_n_max` rows (grade_bon_live's lane-08 spelling, threshold = the store's n_max): the ASR at N, read as the `asr` curve
#: point like the `asr_at_n<N>` ids; REFUSED for new records at emission (`vocab.bon_metric_refusal`), never re-folded
METRIC_LEGACY.update({s: "asr" for s in __import__("sisteer.registry.vocab", fromlist=["BON_LEGACY_METRIC_SPELLINGS"]).BON_LEGACY_METRIC_SPELLINGS})
#: The export's RECOVERED readings come in two decodes and the registry ids both (`recovered_<x>__t0`: the greedy re-decode,
#: `recovered_<x>__sampled`: the sampled one); the pin's readers file the greedy reading under the bare `recovered_<x>` (54 rows,
#: value-equal at c53ae173) and never file the sampled one, which stays a registry-only id. `recovered_asr_folded_counterfactual`
#: and `recovered_vs_others_asr_gap` carry no decode suffix and pass through unchanged.
RECOVERED_T0_METRIC_LEGACY = re.compile(r"^(recovered_[a-z0-9_]+?)__t0$")
#: (lane_source_legacy, eval_category, metric_id) whose reader files NO decode cap (NaN) although the run records one: the export's
#: Petri SCORE rows (the parquet carries no cap on `petri_score`; its Petri degeneracy rows DO carry 1,024 — verified against the
#: c53ae173 pin, 593 vs 50-per-metric rows) and every EM reader (`load_em_agentic` / the organisms reader file none). `*` = any.
#: The export's OVER-REFUSAL rate rows file no cap either (the c53ae173 pin: `over_refusal_rate`, `over_refusal_rate_terminated`,
#: `over_refusal_delta_allitems`, `over_refusal_rate_unpaired` at NaN on 440 rows while the run records 512); its sampled-ladder
#: `over_refusal_rate_full` / `_full_plus_partial` rows keep the cap they were decoded at (2,048 / 8,192), so those stay unlisted.
CAP_UNSPELLED_LEGACY: Tuple[Tuple[str, str, str], ...] = (
    ("r2_export", "petri", "petri_score"), ("em/", "*", "*"),
    ("r2_export", "over_refusal", "over_refusal_rate"), ("r2_export", "over_refusal", "over_refusal_rate_terminated"),
    ("r2_export", "over_refusal", "over_refusal_delta_allitems"), ("r2_export", "over_refusal", "over_refusal_rate_unpaired"),
)
#: The sampled lanes read their stores directly (no reader word on the run): how each spells the judge, per rubric, from the judge
#: row (`load.py`'s sampled-lane reader: the literal `claude-haiku-4-5 / strongreject_v1`; `or_sampled.JUDGE`, one literal for both
#: over-refusal rubrics). `{judge_model}` is the judge row's model.
SAMPLED_OVER_REFUSAL_JUDGE_LEGACY = ("claude-sonnet-5 (over-refusal judge, pass 2 at the final rung); "
                                    "graded: refusal 1, partial 0.5, at-cap 0.5, compliance 0")
#: The per-item over-refusal reader (`load.load_overrefusal_graded` at the pin: ONE literal for both benign sets, the judge model
#: spelled into it) — the adapter reads the served-controls stores directly, so no reader word rides the run.
OVERREFUSAL_ITEMS_JUDGE_LEGACY = "claude-sonnet-5 (over-refusal judge); graded: refusal 1, partial 0.5, at-cap 0.5, compliance 0"
JUDGE_SPELLING_LEGACY: Dict[Tuple[str, str], str] = {
    ("sampled_lane", "strongreject"): "{judge_model} / strongreject_v1",
    ("sampled_lane_controls", "xstest_cpi"): SAMPLED_OVER_REFUSAL_JUDGE_LEGACY,
    ("sampled_lane_controls", "or_bench_cpi"): SAMPLED_OVER_REFUSAL_JUDGE_LEGACY,
    ("overrefusal_items", "xstest_cpi"): OVERREFUSAL_ITEMS_JUDGE_LEGACY,
    ("overrefusal_items", "or_bench_cpi"): OVERREFUSAL_ITEMS_JUDGE_LEGACY,
}
#: Sources whose every row is at the VIEW's dose (the sampled lanes: `dose=1.0` on every reader row; the per-dose views spell the
#: dose in their name, `<view>_d0p25` / `_dm1` / `_halfdose`; the per-item over-refusal reader's `_or_view_meta`: 0.5 on a
#: `_halfdose` target directory, else 1.0), where the run records no strength of its own.
SERVED_KEY_DOSE_SOURCES: Tuple[str, ...] = ("sampled_lane", "sampled_lane_controls", "overrefusal_items")
#: the served key's dose grammar lives in `derive` (one home): the readers of those sources file `run.strength_nominal` from it
_SERVED_KEY_DOSE_SUFFIX = DV.SERVED_KEY_DOSE_SUFFIX


def _served_key_base(view) -> Optional[str]:
    """The base view of a dose-suffixed view name (`gemma4e4b_it_d0p25` → `gemma4e4b_it`), else the view itself."""
    if not isinstance(view, str) or not view:
        return None
    return _SERVED_KEY_DOSE_SUFFIX.sub("", view)


def _served_key_dose(view) -> Optional[float]:
    """The dose a view name spells: 1.0 for a base view, 0.5 for `_halfdose`, `_d0p25` → 0.25, `_dm1` → -1.0, the seeded dose-family
    spelling `qwen35_4b_d1p25_loradiff64s1` → 1.25; None for no view
    (`derive.served_key_dose`, the grammar the sampled / served-controls readers file `run.strength_nominal` from)."""
    return DV.served_key_dose(view)


# --------------------------------------------------------------------------------------------
# WHICH ROWS ARE PAPER ROWS. `v_paper_long` is the registry's long frame: every measurement of every run the paper's
# selection flags, whether or not a reader of the paper ever filed it. The PAPER frame (`paper_long_frame(columns=...)`, and the
# paper's own `registry_source._long_records`, which mirrors this module) carries the rows the paper's readers file, and the
# rows below are registry-only — kept in the registry, never rendered into the paper's schema. Two rules, both keyed on
# columns of the row (never on a value), each returning the reason it excludes:
#   1. A row with NO `lane_source_legacy` is not a paper row. Every reader files a `source`; a run no adapter attributed to a
#      reader's source has no paper spelling to render (the parity's c53ae173 cutover frame carried 842 keys under source NA:
#      the `drill_*` campaigns' smoke runs, the `alphasteer_f22` refit controls, the `~/.cache/paperlib-llama8b/controls` cache
#      stores, and the item-level controls reads whose paper reading rides another source's pooled run).
#   2. A source whose reader files ONE pass of a run keeps that pass only: `sampled_lane`'s reader (`load.load_sampled` at the pin)
#      files the pooled t0+t1 reading (`pass_label = pooled_t0_t1`, its own run), so the per-trial member runs (`base_cap` at the
#      4,096 cap, `ladder_rung` at 8,192) — which share the paper key and differ in value — are registry-only.
# --------------------------------------------------------------------------------------------

#: lane_source_legacy → the `run.pass_label`s whose rows the source's reader files; a source absent here keeps every pass.
SOURCE_PAPER_PASS_LABELS_LEGACY: Dict[str, Tuple[str, ...]] = {"sampled_lane": ("pooled_t0_t1",)}
NO_SOURCE_EXCLUSION_REASON = "no reader source (lane_source_legacy NULL): no paper reader files this run, so it has no paper spelling"


def paper_frame_exclusion(row: Mapping[str, object]) -> Optional[str]:
    """Why a `v_paper_long` row is NOT a paper row (None when it is): the rules above, in order. Applied by `paper_long_frame`
    on the paper-schema rendering path and by the paper's copy of the renderer; the registry-side frame keeps every row."""
    src = row.get("lane_source_legacy")
    if src is None or (isinstance(src, float) and src != src) or src == "":
        return NO_SOURCE_EXCLUSION_REASON
    keep = SOURCE_PAPER_PASS_LABELS_LEGACY.get(str(src))
    if keep is not None and row.get("pass_label") not in keep:
        return (f"pass_label {row.get('pass_label')!r} of source {src}: its reader files the {' / '.join(keep)} pass only; "
                "the other passes (per-trial member runs) are registry-only")
    return None


#: The readers' EFFECT-SIZE prose, keyed (lane_source_legacy, eval_category, registry effect_kind): the registry types the kind
#: (`cohens_h` / `dz`), every reader files a sentence, and the parity compares the column (c53ae173: 3,878 matched keys differed on
#: it). The export's own sentence rides its rows already (the r2 adapter parks `effect_kind_legacy` / `delta_method_legacy` on the
#: contrast) and wins; these are the readers that state no word on the run. Cited to the pinned reader's line. The two hyphens
#: are the readers' (the sampled-lane reader types U+2212, the strength / all-positions readers ASCII `-`).
EFFECT_KIND_LEGACY: Dict[Tuple[str, str, str], str] = {
    ("sampled_lane", "static_attack", "cohens_h"): "Cohen's h of defence success (1 − ASR), positive = better",       # load.py:3104
    ("strength_lane", "static_attack", "cohens_h"): "Cohen's h of defence success (1 - ASR), positive = better",             # lane_strength.py:430
    ("strength_lane_base2it_native", "static_attack", "cohens_h"): "Cohen's h of defence success (1 - ASR), positive = better",   # same reader, second record
    ("arditi_allpos", "static_attack", "cohens_h"): "Cohen's h of defence success (1 - ASR), positive = better",             # lane_arditi_allpos.py:766
    ("arditi_allpos", "capability", "cohens_h"): "Cohen's h of accuracy vs the same-standup unsteered arm, positive = better",  # lane_arditi_allpos.py:1039
    ("llama8b_lane", "capability", "cohens_h"): "cohen_h accuracy vs own unsteered",                                        # lane_llama8b.py:471
}
#: The readers' DELTA-METHOD prose where the registry files the method as a token (`paired_bootstrap`), keyed (lane_source_legacy,
#: eval_category): a template over the contrast's `n_pairs` — the only parameter the sentence embeds besides the reader's fixed
#: bootstrap size (`n_boot = 5000`, both readers' default). The pin's `n` distribution equals the registry's `n_pairs` on every row
#: of both sources (verified against the c53ae173 pin). Readers whose sentence embeds what the registry does not hold (discordant
#: counts, per-item correct counts, the anchor's engine) keep the registry's own words.
DELTA_METHOD_LEGACY: Dict[Tuple[str, str], str] = {
    ("sampled_lane", "static_attack"): "paired on {n_pairs} goals (per-goal success share over trials): percentile bootstrap 5000; Wilcoxon signed-rank p",   # load.py:3102
    ("sampled_lane_controls", "over_refusal"): ("paired on {n_pairs} benign items (graded score, per-item mean over trials; each cell at its own final "
                                                "rung): percentile bootstrap 5000; Wilcoxon signed-rank p"),                                   # or_sampled.py:282-283
}
DELTA_METHOD_TOKENS_SPELLED: Tuple[str, ...] = ("paired_bootstrap",)
#: A derivation's answer that the column is ABSENT on this row by the reader's own word (the export parked `delta_method_legacy: null`
#: on a contrast whose method it never stated, while the registry's typed column carries a NOT NULL filler): the chain stops and the
#: column renders None instead of falling to the filler.
_ABSENT = object()


#: Registry column → the paper's long-schema name for the columns that are a plain rename.
LONG_COLUMN_LEGACY_NAMES: Dict[str, str] = {
    "lane_source_legacy": "source",
    "view_legacy": "view",
    "campaign_cell_id_legacy": "cell_id",
    "operating_point_id_legacy": "arm_operating_point_id",
    "paper_label": "arm",
    "transfer_kind": "lane_kind",
    "eval_category": "component",
    "strength_nominal": "dose",
    "metric_id": "metric",
    "token_cap_tokens": "decode_cap_tokens",
    "store_uri": "provenance",
    "fit_layers_json": "fit_layers",
    "object_artifact_sha256": "object_sha256",
    "context_window_tokens": "serving_window_tokens",
    "producer_run_key_legacy": "grades_root",
}

#: The long schema the registry can supply, long column → how it is derived: a registry column of
#: v_paper_long (`col:<name>`), a key of the metric / run extra_json (`extra:<key>`, metric extra first), or a
#: derivation (`fn:<name>`). Precedence for a column with several sources: the extra_json key filed by the producer's
#: adapter wins over the registry derivation, so the legacy value the reader emitted is reproduced when it was kept.
LONG_COLUMN_SOURCES: Dict[str, Tuple[str, ...]] = {
    "source": ("col:lane_source_legacy",),
    "model": ("fn:model",),
    # THE READER'S OWN WORDS FIRST (the declared adapter's `paper_*` park, the r2 adapter's `judge_legacy` / `lane_kind`), then the
    # spelling derivations (CAMPAIGN_ID_PAPER_LABEL_LEGACY, SERVED_KEY_TRANSFER_LEGACY, METRIC_LEGACY, JUDGE_SPELLING_LEGACY, `_served_key_dose`) —
    # the paper's readers are the contract; the registry's transfer_kind / paper_label / producer spelling are not.
    "lane_kind": ("extra:paper_lane_kind", "fn:lane_kind"),
    "view": ("col:view_legacy",),
    "dose": ("extra:dose", "extra:paper_dose", "fn:dose"),
    "cell_id": ("col:campaign_cell_id_legacy",),
    # THE ARM IS THE OBJECT'S, NEVER A PRODUCER COLUMN'S (the paper side, 2026-09-20): after the distilled student was retired, four
    # bcb_bfcl_200_widen rows still carried the producer's `diff_psr` on the student's cell, and a producer-first chain would have drawn the
    # retired object under the live label. `fn:arm` spells the OBJECT's label (its own `paper_label`, else the stem of its cell); the
    # reader's parked word (`extra.paper_arm`) is read only where the object carries no paper label at all, and a parked word that
    # disagrees with the object's is kept as a fact on the row (`producer_arm_legacy`, LONG_REGISTRY_COLUMNS) — never silently dropped.
    "arm": ("fn:arm",),
    "family": ("fn:family",),
    "object_type": ("extra:paper_object_type", "fn:object_type"),
    "component": ("extra:paper_component", "fn:component"),
    # THE EVAL ID IS THE PAPER'S SPELLING. The registry's eval is the clean benchmark and the sampled re-run is INSTRUMENT identity
    # (its own instrument_id: cap, decode source, seed rule, trials); the paper spells that instrument as a second eval id
    # (`pasttense_sampled4096`) and selects on the suffix. The writer's spelling (`eval_id_legacy`, the r2 adapter's park of the
    # parquet's own value) wins; else the legacy eval name plus the suffix where the run's instrument seed is a sampled one.
    "eval_id": ("extra:paper_eval_id", "col:eval_id_legacy", "fn:eval_id"),
    "eval": ("extra:eval", "extra:paper_eval_id", "col:eval_id_legacy", "fn:eval_id"),
    # the registry metric id, spelled through METRIC_LEGACY — the producer's spelling is provenance (`producer_metric_spelling`),
    # not a key the readers file (parity c53ae173: `attack_metrics.asr[0.5]` on 5,085 sampled rows, `by_category_group.held_in`
    # and `behaviors.<b>.mean` on the export's, against the readers' `asr` / `asr_held_in` / `petri_score`)
    "metric": ("fn:metric",),
    "threshold": ("fn:threshold",),
    "bon_n": ("extra:bon_n", "col:bon_n"),
    "value": ("col:value",), "k": ("col:k",), "n": ("col:n",), "ci_lo": ("col:ci_lo",), "ci_hi": ("col:ci_hi",),
    "delta": ("col:delta", "extra:delta_block.delta"), "delta_lo": ("col:delta_lo", "extra:delta_block.delta_lo"),
    "delta_hi": ("col:delta_hi", "extra:delta_block.delta_hi"), "delta_p": ("col:delta_p", "extra:delta_block.delta_p"),
    # THE READER'S OWN WORDS FIRST (the export's `delta_method_legacy` / `effect_kind_legacy` parked on the contrast, the reader
    # sentences of DELTA_METHOD_LEGACY / EFFECT_KIND_LEGACY), then the registry's typed token
    "delta_method": ("fn:delta_method", "col:delta_method", "extra:delta_block.delta_method"),
    "effect_size": ("col:effect_size", "extra:delta_block.effect_size"),
    "effect_kind": ("fn:effect_kind", "col:effect_kind", "extra:delta_block.effect_kind"),
    # THE DEGENERACY COLUMNS (paper-figures f33dca5d, `load.select_degeneracy_rows`): the pin stamps the export's degeneracy METRIC ROWS
    # onto the cell's HEADLINE rows — `at_cap_share` takes the CANONICAL re-tokenized reading where the cell has one and the producer's
    # finish-reason value moves to `at_cap_share_finish_reason`; `degenerate_share` takes the five-detector union; the decomposition
    # rides in `degen_*` columns; `degeneracy_basis` names each filled column's estimand. The registry keeps those readings as
    # `run_degeneration` MEASURES (the union adapter's and the r2 adapter's rows), so the frame reads them from
    # `v_run_degeneration_current` (`fn:degen_rate:<measure>`, HEADLINE rows only — the run's headline metric id: the first of the
    # eval's accepted `extra_json.headline_metric_ids` the run files, else `eval.headline_metric_id`; `run_headline_metric_id`), and
    # falls back to the `frac_*` mirror the writer filed on the metric row.
    "at_cap_share": ("fn:at_cap_share", "col:frac_at_token_cap"),
    # ONE PAPER COLUMN, TWO DEFINITIONS: the r2 export's `loop_share` is the published loop rule (`rep_ratio_loop`), the declared
    # readers' is the pinned loop classifier's share (`classifier`). Each writer files the per-definition mirror column and names
    # the measure in `extra.loop_share_measure_legacy`; the reader reads THAT mirror (never the first non-null of two filled
    # columns), and the paper's published definition where a writer states none. Reading `frac_loop_first512` here — a third
    # statistic no writer files as loop_share — returned the first-512 share on every r2 row and NULL on every declared row.
    # MULTI-SOURCE CARRY (ownership.MULTI_SOURCE_OWNERSHIP; the paper's `load.dedupe_multi_source_cells`): on the OWNER row of a FOLDED
    # measurement — one identity read by two sources whose k / n agreed exactly — the declared carry readings (`degenerate_share`,
    # `loop_share`) come from the secondary's row WHERE THE OWNER HAS NONE, which is why `fn:multi_source_carry:<col>` is LAST in
    # each chain: every reading of the owner's own runs is tried first. `at_cap_share` has no carry step at all (never_carry): the
    # owner's is the export's per-measurement at-cap figure, the secondary's the sidecar's re-tokenized reading — two statistics.
    "loop_share": ("fn:loop_share", "fn:multi_source_carry:loop_share"),
    # `degenerate_share`: the five-detector union where the cell has one (headline rows, the pin's stamp), else the writer's mirror
    # (the declared readers' two-detector union filed under union5 with its note), else the r2 export's two-detector union — the
    # `loop_or_at_cap` measure the adapter routed its `degenerate_share` metric row into (headline rows), else the export's own
    # per-cell column parked by name on every row of the cell (what the pin's unstamped rows carry), else the multi-source carry
    "degenerate_share": ("fn:degen_rate:union5", "col:frac_degenerate_union5", "fn:degen_rate:loop_or_at_cap", "extra:degenerate_share",
                         "fn:multi_source_carry:degenerate_share"),
    "at_cap_share_finish_reason": ("extra:at_cap_share_finish_reason", "fn:at_cap_share_finish_reason"),
    "degen_truncation_only_share": ("extra:degen_truncation_only_share", "fn:degen_rate:truncation_only"),
    "degen_loop_without_cap_share": ("extra:degen_loop_without_cap_share", "fn:degen_rate:loop_without_cap"),
    "degeneracy_basis": ("extra:degeneracy_basis", "fn:degeneracy_basis"),
    # THE SERVING ENGINE AND THE CROSS-ENGINE FLAG (f33dca5d, `load_em_agentic`): the engine the producer says served the row is
    # `run.serving_identity` (observed at serving); the cross-engine flag is the producer's ATTESTED verdict on the paired
    # contrast (`run_contrast.extra_json.cross_engine_contrast`, the em_misalign adapter's to file), never a derivation — see
    # LONG_COLUMN_NULL_REASONS.
    "serving_engine": ("extra:serving_engine", "col:serving_identity"),
    "cross_engine_contrast": ("extra:cross_engine_contrast", "fn:cross_engine_contrast"),
    # THE JUDGE-CENSORED BOUND (f33dca5d): the pin derives `judge_blocked` as the unsteered arm's n minus this arm's n because its
    # file states no blocked count; the registry's judgment states it — `judgment.n_judge_refusals`. `lower_bound` is True where
    # that is positive (the rate is a FLOOR), False where the judgment counted zero refusals, NULL where it counted none.
    "judge_blocked": ("extra:judge_blocked", "col:judgment_n_judge_refusals"),
    "lower_bound": ("extra:lower_bound", "fn:lower_bound"),
    "degeneration_delta_vs_anchor": ("extra:degeneration_delta_vs_anchor",),
    "saturated": ("extra:saturated",), "near_floor": ("extra:near_floor",), "suggestive": ("extra:suggestive",), "retention": ("extra:retention",),
    # NULL where the instrument decodes nothing (MMLU scores one token by logprob: `token_cap_tokens` 1); every reader files NaN there
    "decode_cap_tokens": ("fn:decode_cap_tokens",),
    "decode": ("extra:decode_declared_legacy", "extra:decode", "col:decode_source"),
    # the reader's judge string as parked (`judge_legacy`: the declared readers and the export), else the lane's spelling of the judge
    # row (JUDGE_SPELLING_LEGACY), else the judge model alone
    "judge": ("extra:judge_legacy", "extra:judge", "fn:judge"),
    # a REFUSED multi-source owner row (two sources disagree on k / n) says so in `status` ahead of any writer's own spelling, the way
    # a disposition row does (`<state>: <reason>`); `note` carries the fold / stopgap / refusal sentence after the writer's note
    "status": ("fn:multi_source_status", "extra:status_legacy", "extra:status", "fn:status"),
    "note": ("fn:note",),
    "provenance": ("extra:provenance", "col:store_uri"),
    "fit_layers": ("col:fit_layers_json",), "fit_layer_count": ("col:fit_layer_count",),
    "coherence": ("extra:coherence",), "delta_coherence": ("extra:delta_coherence",), "degenerate_win": ("extra:degenerate_win",),
    "anchor_degenerate": ("extra:anchor_degenerate",), "response_chars": ("extra:response_chars",),
    "thinking_mode": ("col:thinking_mode",),
    "object_sha256": ("fn:object_sha256",),
    "serving_window_tokens": ("col:context_window_tokens",),
    "bon_route": ("extra:bon_route",),
    "serving_config_sha256": ("col:serving_config_sha256",),
    "held_out": ("fn:held_out",),
    "arm_operating_point_id": ("col:operating_point_id_legacy",),
    "arm_operating_point": ("extra:arm_operating_point",),
    "grades_root": ("extra:grades_root", "col:producer_run_key_legacy"),
    "petri_run_key": ("extra:petri_run_key", "fn:petri_run_key"),
    "petri_intervention_hash": ("extra:petri_intervention_hash", "fn:petri_intervention_hash"),
    "petri_served_object_sha256": ("extra:petri_served_object_sha256", "fn:petri_served_object_sha256"),
}
#: The registry-side names the long frame ALSO carries (a superset the notebook may ignore).
LONG_REGISTRY_COLUMNS: Tuple[str, ...] = (
    "run_id", "round_id", "in_current_paper", "selection_state", "comparison_ids_json", "roster_entry_ids_json", "decisive_ruling_id",
    "model_id", "object_id", "paper_label", "eval_category", "instrument_id", "judge_id", "pass_label", "lifecycle_state",
    "transfer_kind", "strength_nominal", "strength_coefficient", "is_operating_point",
    # how the run counts as being at the operating point (schema_def.operating_point_basis_sql, migration 0011): `stamped` | `fallback_nominal_1p0` |
    # `fallback_arditi_c0p3` | `no_strength_axis` | `none`, so a consumer can tell a stamped row from one the ruling of 2026-09-22 admitted
    "operating_point_basis",
    "ingest_seq", "eval_id_legacy",
    # the derived decode-policy match (decode_match): the ONE policy row the run's sent decode equals, and the verdict
    "decode_policy_id", "decode_policy_match",
    # the row resolve() returns today for the run's key and whether the matched row IS it (schema_def.decode_family_default_sql, migration 0009):
    # `exact` + 0 = a recorded, non-default policy (the agentic T = 1.0 runs since the rule of 2026-09-21); NULL = nothing matched
    "decode_family_default_policy_id", "decode_is_family_default",
    # multi-source ownership (ownership.MULTI_SOURCE_OWNERSHIP): the row's role in a declared scope and the fold's outcome, so a reader
    # of the frame can tell a folded owner row, a stopgap secondary row and a refused identity apart without the status page
    "multi_source_declaration_id", "multi_source_role", "multi_source_state", "multi_source_reason", "multi_source_secondary_run_id",
    # metric role (schema_def.METRIC_ROLE_CTES): the ONE row per (run, judgment, metric, threshold) the paper counts (`owner`) against the
    # other provenances of the same measurement (`secondary`, in_current_paper 0 — the stats readers' blank-computation twin of an export
    # row), the row's and the run's own source, how many rows stood at the identity and whether two computed rows from different sources tied
    "metric_role", "metric_source_id", "run_source_id", "metric_role_candidates", "metric_role_contested",
    # how many CURRENT contrasts stand at the row's metric key (schema_def.CURRENT_CONTRAST_CTE, migration 0021): 1 on most rows, 2 where a
    # source files two anchor kinds by design, None where the row joins no current contrast (its anchor is not current); no tie-break
    "n_current_contrasts",
    # the producer's parked arm word where it DISAGREES with the object's spelling (render_long_row); None when they agree or none was parked
    "producer_arm_legacy",
)

#: Long columns that are NULL on the OWNER row of a REFUSED multi-source identity (`v_multi_source_check.state = 'refused'`: the two
#: sources disagree on k / n, or one carries two rows at one identity, or the owner already carries a different carry reading) — the
#: reason, per column, in the style of LONG_COLUMN_NULL_REASONS. `v_paper_long` NULLs them in SQL so every consumer sees the same
#: refusal; `status` reads `refused: <reason>` and `multi_source_reason` carries the two sides' counts. Neither source's reading is
#: admitted: two readings of one measurement that differ are a disagreement to rule on, never averaged and never picked from.
MULTI_SOURCE_REFUSED_NULL_REASON = ("NULL on the owner row of a multi-source identity whose two sources disagree (v_multi_source_check.state = "
                                    "'refused'): neither the owner's nor the secondary's reading is admitted; `multi_source_reason` states both sides")
LONG_COLUMN_REFUSED_NULL_REASONS: Dict[str, str] = {c: MULTI_SOURCE_REFUSED_NULL_REASON for c in ("value", "k", "n", "ci_lo", "ci_hi")}

#: The sentences the long frame's `note` carries for a row inside a declared multi-source scope (the paper's MULTI_SOURCE_*_NOTE, in
#: the registry's nouns; `{reason}` is `v_multi_source_check.reason`).
MULTI_SOURCE_STOPGAP_NOTE = " | STOPGAP ROW (multi-source ownership): {reason}"
MULTI_SOURCE_FOLDED_NOTE = " | FOLDED (multi-source ownership): {reason}"
MULTI_SOURCE_REFUSED_NOTE = " | REFUSED (multi-source ownership): {reason}"

#: The paper's suffix for the sampled re-run's eval ids (`paperlib.config.SAMPLED4096_SUFFIX`), and the instrument SEEDS
#: (`configs/registry/instruments.yaml`, `run.extra_json.instrument_seed_id`) that instrument is: the full22 static sampled
#: re-run and the two sampled over-refusal ladders. Named, never matched on a substring of the id.
SAMPLED4096_SUFFIX_LEGACY = "_sampled4096"
SAMPLED4096_INSTRUMENT_SEEDS: Tuple[str, ...] = (
    "static_sampled_vendor_preset_4096_8192", "overrefusal_xstest_sampled_vendor_preset_ladder", "overrefusal_orbench_sampled_vendor_preset_ladder",
)

#: Long columns the registry supplies but whose value is NULL wherever the datum is genuinely absent from the registry's data
#: model — the reason, per column: NULL with a stated reason, never a refusal and never a guess.
LONG_COLUMN_NULL_REASONS: Dict[str, str] = {
    "cross_engine_contrast": ("the registry DERIVES no cross-engine flag: the pin's class rule compares the arm's engine with the PANEL's "
                              "declared steered path (paperlib's EM_AGENTIC_BEHAVIOUR_SOURCE declaration), which no registry table holds; "
                              "only the producer's attested flag on the contrast (`run_contrast.extra_json.cross_engine_contrast`, the "
                              "em_misalign adapter's to file) or an adapter-filed `extra.cross_engine_contrast` fills it"),
    "serving_engine": "NULL where no producer recorded the engine that served the run (`run.serving_identity` is observed at serving; legacy rows may lack it)",
    "judge_blocked": "NULL where the judgment counted no refusals (`judgment.n_judge_refusals` null): unknown, never 0",
    "lower_bound": "NULL wherever judge_blocked is NULL (a bound cannot be asserted from an uncounted censoring)",
    "at_cap_share_finish_reason": ("NULL off the headline row and wherever neither the finish-reason measure nor a canonical re-tokenized reading "
                                   "exists for the run (the pin fills it only when it moves the producer's value aside)"),
    "degen_truncation_only_share": "NULL where the run has no measured `truncation_only` degeneration row (the decomposition exists for cap-4,096 multi-turn cells only)",
    "degen_loop_without_cap_share": "NULL where the run has no measured `loop_without_cap` degeneration row (same scope)",
    "degeneracy_basis": "NULL off the headline row: the pin stamps a basis on the rows whose metric is the component's headline metric only",
}

#: `run_degeneration` measure → the paper column its all-scope rate fills on a HEADLINE row (the pin's DEGENERACY_METRIC_ROWS, spelled
#: by measure), in the pin's stamping order.
DEGENERATION_MEASURE_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("union5", "degenerate_share"), ("loop_or_at_cap", "degenerate_share"), ("rep_ratio_loop", "loop_share"), ("at_token_cap_retokenized", "at_cap_share"),
    ("at_token_cap_finish_reason", "at_cap_share_finish_reason"), ("truncation_only", "degen_truncation_only_share"),
    ("loop_without_cap", "degen_loop_without_cap_share"),
)


def _degeneration_map(conn: sqlite3.Connection, run_ids: Sequence[str]) -> Dict[Tuple[str, str], Dict[str, Dict[str, object]]]:
    """(run_id, judgment_id) → measure → the MEASURED all-scope `run_degeneration` row (`v_run_degeneration_current`: one row per key,
    a number before every placeholder), for the frame's runs only."""
    out: Dict[Tuple[str, str], Dict[str, Dict[str, object]]] = {}
    ids = list(dict.fromkeys(run_ids))
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        marks = ",".join("?" for _ in chunk)
        for r in conn.execute("SELECT run_id, judgment_id, measure, rate, numerator, denominator, denominator_unit, pooling_json, aggregation, "
                              f"definition_version FROM v_run_degeneration_current WHERE scope = 'all' AND measurement_state = 'measured' AND run_id IN ({marks})",
                              chunk).fetchall():
            d = dict(r)
            out.setdefault((str(d["run_id"]), str(d["judgment_id"] or "")), {})[str(d["measure"])] = d
    return out


def _degeneration_for(ctx: Optional[Mapping[str, object]], row: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    """The row's run's measured readings: the judgment-less rows (the union adapter files under '') with the row's own judgment's
    rows winning where both exist."""
    if not ctx:
        return {}
    dm = ctx.get("degeneration") or {}
    rid = str(row.get("run_id"))
    m = dict(dm.get((rid, ""), {}))
    jid = str(row.get("judgment_id") or "")
    if jid:
        m.update(dm.get((rid, jid), {}))
    return m


#: `eval.extra_json` key: the eval's ACCEPTED headline metric ids in order of preference (seeded in configs/registry/evals.yaml).
#: `strongreject_bon` lists `asr` (the 512-store + 4,096-climb reading at bon_n 75 — the canonical best-of-N) and
#: `asr_bon_cap4096` (the native 4,096 pass: the same per-goal any-hit unit; threshold 0.5 both).
#: The producers that run the native pass file their headline under the second spelling ONLY (bon_stores 48 runs, seedrep_cluster 24 on
#: 2026-09-21), so a consumer testing `headline_metric_id` alone read the native-pass best-of-N as absent — no degeneracy stamp, no
#: headline items. `headline_metric_id` stays the ONE declared id (NOT NULL, CHECK-guarded); the list widens what is accepted.
HEADLINE_METRIC_IDS_KEY = "headline_metric_ids"


def headline_metric_ids(eval_row: Mapping[str, object]) -> List[str]:
    """The eval's accepted headline metric ids in order of preference: `extra_json.headline_metric_ids` where the seed lists them,
    closed by the declared `headline_metric_id` (the only entry for every other eval). `eval_row` is any mapping carrying
    `headline_metric_id` and the eval's `extra_json` (text or dict; a frame row's `eval_extra_json` is read where `extra_json` is absent).
    THE one place the rule lives: every registry-side consumer that picks a run's headline goes through here (or through
    `v_item_headline`, the same rule in SQL)."""
    extra = _json_or_empty(eval_row.get("extra_json") if "extra_json" in eval_row else eval_row.get("eval_extra_json"))
    listed = extra.get(HEADLINE_METRIC_IDS_KEY)
    out: List[str] = []
    for m in (listed if isinstance(listed, (list, tuple)) else ()):
        if m is not None and str(m) not in out:
            out.append(str(m))
    head = eval_row.get("headline_metric_id")
    if head is not None and str(head) not in out:
        out.append(str(head))
    return out


def run_headline_metric_id(eval_row: Mapping[str, object], filed_metric_ids: Iterable[str]) -> Optional[str]:
    """The metric id that IS a run's headline: the first accepted id (`headline_metric_ids`) the run files a metric row under, else
    the declared `headline_metric_id` (a run that files none of them has no headline row, and the declared id names what is missing)."""
    filed = {str(m) for m in filed_metric_ids if m is not None}
    for m in headline_metric_ids(eval_row):
        if m in filed:
            return m
    head = eval_row.get("headline_metric_id")
    return None if head is None else str(head)


def _headline_by_run(conn: sqlite3.Connection, frame) -> Dict[str, str]:
    """run_id → the run's headline metric id in THIS frame (`run_headline_metric_id` over the eval's accepted ids and the metric ids
    the frame carries for the run). The accepted lists are read from the `eval` table — the frame's rows carry the declared id only."""
    if not len(frame):
        return {}
    filed: Dict[str, List[str]] = {}
    eval_of: Dict[str, str] = {}
    for rid, eid, mid in frame[["run_id", "eval_id", "metric_id"]].itertuples(index=False, name=None):
        filed.setdefault(str(rid), []).append(str(mid))
        eval_of.setdefault(str(rid), str(eid))
    evals: Dict[str, Dict[str, object]] = {}
    ids = sorted(set(eval_of.values()))
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        marks = ",".join("?" for _ in chunk)
        for r in conn.execute(f"SELECT eval_id, headline_metric_id, extra_json FROM eval WHERE eval_id IN ({marks})", chunk).fetchall():
            evals[str(r[0])] = {"headline_metric_id": r[1], "extra_json": r[2]}
    out: Dict[str, str] = {}
    for rid, eid in eval_of.items():
        ev = evals.get(eid)
        if ev is None:
            raise LookupError(f"run {rid}: eval {eid!r} has no `eval` row; the frame cannot resolve the run's headline metric id")
        head = run_headline_metric_id(ev, filed[rid])
        if head is not None:
            out[rid] = head
    return out


def _is_headline(row: Mapping[str, object], ctx: Optional[Mapping[str, object]] = None) -> bool:
    """The pin stamps the rows whose metric is the component's headline metric (config.HEADLINE), at every threshold. The registry's
    word for that is the RUN'S headline metric id — the first of the eval's accepted `headline_metric_ids` the run files, else the
    declared `eval.headline_metric_id` — resolved once per frame into `ctx["headline_by_run"]` (`_headline_by_run`). A row rendered
    with no frame behind it (a fixture row, no database) reads the declared id the row carries."""
    head = None
    if ctx:
        head = (ctx.get("headline_by_run") or {}).get(str(row.get("run_id")))
    if head is None:
        head = row.get("headline_metric_id")
    return head is not None and row.get("metric_id") == head


def _at_cap_basis(at_cap, retok_filled: bool) -> str:
    """What the row's `at_cap_share` IS, in the pin's words (`load._at_cap_basis`)."""
    if retok_filled:
        return "at_cap_share is the CANONICAL re-tokenized estimand (>= 0.98 x cap), at_cap_share_finish_reason the producer's finish_reason == length"
    if at_cap is not None:
        return "at_cap_share = producer finish_reason == length (no re-tokenized reading for this cell)"
    return "no at-cap reading (at_cap_share null)"


#: `run_degeneration` measure → the `run_metric.frac_*` mirror column that carries its all-scope rate (the loop detectors a writer's
#: legacy `loop_share` can name).
LOOP_SHARE_MIRROR_COLUMNS: Dict[str, str] = {
    "rep_ratio_loop": "frac_degenerate_rep_ratio_loop", "classifier": "frac_degenerate_classifier", "loop_first512": "frac_loop_first512",
}
#: The paper's published loop rule (`load.py` at f33dca5d: word 4-gram repetition > 0.5, the rep_ratio detector) — what a writer
#: that states no `loop_share_measure_legacy` is read as.
LOOP_SHARE_DEFAULT_MEASURE = "rep_ratio_loop"


def _loop_share_measure(row: Mapping[str, object], extras: List[Dict[str, object]]) -> str:
    """The `run_degeneration` measure the row's legacy `loop_share` names: the writer's `extra.loop_share_measure_legacy` (metric
    extra first), else the paper's published loop rule. An unknown spelling falls to the default rather than to a guess."""
    hint = _extra_get(extras, "loop_share_measure_legacy")
    return str(hint) if hint in LOOP_SHARE_MIRROR_COLUMNS else LOOP_SHARE_DEFAULT_MEASURE


def _multi_source_carried(row: Mapping[str, object]) -> List[str]:
    """The carry columns a FOLDED owner row takes from the secondary source (`v_paper_long.multi_source_carried_json`); empty on
    every other row. Only a declared carry column is ever named here (ownership.CARRY_COLUMNS), so `at_cap_share` cannot appear."""
    if row.get("multi_source_role") != "owner" or row.get("multi_source_state") != "folded":
        return []
    raw = row.get("multi_source_carried_json")
    try:
        names = json.loads(raw) if isinstance(raw, str) else (raw or [])
    except ValueError:
        return []
    from . import ownership as OW
    return [str(c) for c in names if str(c) in OW.CARRY_COLUMNS]


def _multi_source_basis_tail(row: Mapping[str, object]) -> str:
    """What a folded owner row's carried degeneracy columns ARE: the secondary source's readings, named with the detector its
    loop_share carries (the two writers' loop_share name different detectors — the declared reader's is the classifier's share)."""
    carried = _multi_source_carried(row)
    if not carried:
        return ""
    names = []
    for c in carried:
        if c == "loop_share":
            names.append(f"loop_share = the secondary source's {row.get('multi_source_loop_share_measure') or LOOP_SHARE_DEFAULT_MEASURE} share")
        else:
            names.append(f"{c} = the secondary source's reading")
    return "; ".join(names) + f" (carried from {row.get('multi_source_secondary_run_id')} under multi-source ownership; the owner filed none)"


def _degeneracy_basis(row: Mapping[str, object], extras: List[Dict[str, object]], ctx) -> Optional[str]:
    if not _is_headline(row, ctx):
        return None
    base = _degeneracy_basis_own(row, extras, ctx)
    tail = _multi_source_basis_tail(row)
    return f"{base}; {tail}" if (base and tail) else (tail or base)


def _degeneracy_basis_own(row: Mapping[str, object], extras: List[Dict[str, object]], ctx) -> Optional[str]:
    """The basis from the row's OWN run's readings and columns (the pin's `_at_cap_basis` / `_unstamped_basis`)."""
    deg = _degeneration_for(ctx, row)
    at_cap_col = row.get("frac_at_token_cap")
    filled: List[str] = []
    retok = "at_token_cap_retokenized" in deg
    loop_measure = _loop_share_measure(row, extras)     # the detector THIS row's loop_share names (a writer's hint, else the published rule)
    named: set = set()
    for measure, col in DEGENERATION_MEASURE_COLUMNS:
        if col == "loop_share":
            measure = loop_measure
        if measure not in deg or col in named:      # one column, one estimand: the first measure in stamping order names it
            continue
        named.add(col)
        filled.append(f"{col} = {measure}")
    if filled:
        parts = ["; ".join(filled)]
        u = deg.get("union5")
        if u is not None:
            pooling = _json_or_empty(u.get("pooling_json"))
            parts.append(f"union over n = {int(u['denominator'])}" + (f" ({pooling['denominator']})" if pooling.get("denominator") else ""))
        parts.append(_at_cap_basis(at_cap_col, retok))
        return "; ".join(parts)
    # no measured reading for the run: say what the row itself carries (the pin's `_unstamped_basis`)
    have = [c for c, v in (("degenerate_share", row.get("frac_degenerate_union5") if row.get("frac_degenerate_union5") is not None
                            else _extra_get(extras, "degenerate_share")),
                           ("loop_share", row.get(LOOP_SHARE_MIRROR_COLUMNS[_loop_share_measure(row, extras)]))) if v is not None]
    if not have and at_cap_col is None:
        return "no degeneracy reading and no at-cap reading for this cell"
    parts = [_at_cap_basis(at_cap_col, False)]
    parts.append(f"{' / '.join(have)} = the writer's own mirror column (producer's pooled reading; no union-classifier reading for this cell)"
                 if have else "no degeneration reading")
    return "; ".join(parts)


def _as_bool(v) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
        return None
    return bool(v)


def _json_or_empty(text) -> Dict[str, object]:
    if isinstance(text, dict):
        return text
    if not text:
        return {}
    try:
        v = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return v if isinstance(v, dict) else {}


def _extra_get(extras: List[Dict[str, object]], dotted: str):
    """First non-null value of a dotted key across the extra_json dicts, in order."""
    for ex in extras:
        cur: object = ex
        ok = True
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur is not None:
            return cur
    return None


def _paper_label_of(row: Mapping[str, object]):
    """The row's paper label for spelling: the object's own where it carries one, else the label of the CELL the campaign served the
    object under (CAMPAIGN_ID_PAPER_LABEL_LEGACY, keyed by the cell's STEM: a staged spelling such as `itdoc64__constv2-mix-full__lora_r4_ce`
    names its fit variant before the stem, and the variant is never part of the arm — derive.campaign_stem_legacy). The objects adapter
    spells a fitted object's label from the same stem (fit_sidecars); this fallback covers rows read before that spelling landed."""
    label = row.get("paper_label")
    if label in (None, "other"):
        stem, _prefix = DV.campaign_stem_legacy(row.get("campaign_cell_id_legacy"))
        by_campaign = CAMPAIGN_ID_PAPER_LABEL_LEGACY.get(str(stem))
        if by_campaign is not None:
            return by_campaign
    return label


def render_long_row(row: Mapping[str, object], columns: Sequence[str], ctx: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    """One `v_paper_long` row (a mapping of its columns, NaN already read as None) rendered into the paper's long schema through
    LONG_COLUMN_SOURCES: the first non-null of each column's chain. The registry-side LONG_REGISTRY_COLUMNS ride along. Exposed so a
    fixture row can be rendered without a database (tests/test_registry_paper_spelling.py)."""
    extras = [_json_or_empty(row.get("metric_extra_json")), _json_or_empty(row.get("run_extra_json"))]
    out: Dict[str, object] = {}
    for name in columns:
        val = None
        for src in LONG_COLUMN_SOURCES[name]:
            kind, _, key = src.partition(":")
            if kind == "col":
                val = row.get(key)
            elif kind == "extra":
                val = _extra_get(extras, key)
            else:
                val = _derive(key, row, extras, ctx)
                if val is _ABSENT:                 # the reader's own word says the column is empty here: the chain stops
                    val = None
                    break
            if val is not None and not (isinstance(val, float) and val != val):
                break
        out[name] = val
    for name in LONG_REGISTRY_COLUMNS:
        out[name] = row.get(name)
    # the producer's parked arm word, kept as a FACT when it disagrees with the object's spelling (the frame's `arm` is the object's):
    # the reader that parked `diff_psr` on the distilled student's rows is on the row, and the retired object is not drawn under the live label
    parked = _extra_get(extras, "paper_arm")
    object_spelling = _derive("arm", row, extras, ctx)
    out["producer_arm_legacy"] = parked if (parked is not None and object_spelling not in (None, "other") and str(parked) != str(object_spelling)) else None
    return out


def _derive(name: str, row: Mapping[str, object], extras: List[Dict[str, object]], ctx: Optional[Mapping[str, object]] = None):
    if name.startswith("degen_rate:"):
        # a HEADLINE row reads the run's measured all-scope rate of the named measure (the pin stamps headline rows only). The map is
        # keyed by the row's OWN run: a folded owner row never reads the secondary run's measures here — the only readings that cross
        # from the secondary are the declared carry columns, through `fn:multi_source_carry:<col>` (so the secondary's at-cap measures,
        # `at_token_cap_retokenized` included, can never reach the owner's `at_cap_share`)
        if not _is_headline(row, ctx):
            return None
        d = _degeneration_for(ctx, row).get(name.split(":", 1)[1])
        return None if d is None else d.get("rate")
    if name.startswith("multi_source_carry:"):
        # the secondary source's reading of a declared carry column, on the OWNER row of a FOLDED identity, where the owner had none
        # (the chain's earlier steps read every reading of the owner's own run first — this step is last by construction)
        col = name.split(":", 1)[1]
        if col not in _multi_source_carried(row):
            return None
        return row.get(f"multi_source_{col}")
    if name == "multi_source_status":
        if row.get("multi_source_role") == "owner" and row.get("multi_source_state") == "refused":
            return f"refused: {row.get('multi_source_reason') or 'the two sources disagree'}"
        return None
    if name == "note":
        base = _extra_get(extras, "note")
        if base is None:
            base = row.get("notes")
        state, role = row.get("multi_source_state"), row.get("multi_source_role")
        tmpl = {("secondary", "stopgap"): MULTI_SOURCE_STOPGAP_NOTE, ("owner", "folded"): MULTI_SOURCE_FOLDED_NOTE,
                ("owner", "refused"): MULTI_SOURCE_REFUSED_NOTE}.get((str(role), str(state)))
        if tmpl is None:
            return base
        return str(base or "") + tmpl.format(reason=row.get("multi_source_reason") or "")
    if name == "at_cap_share":
        # the CANONICAL re-tokenized reading takes the column where the cell has one; else the caller falls to the producer's mirror.
        # SCOPED TO THE ROW'S OWN RUN: under multi-source ownership the owner's own re-tokenized reading, if it
        # ever files one, is still the owner's statistic; the SECONDARY's re-tokenized reading is never consulted (never_carry) — the
        # paper's rule for the scope and the per-run preference do not collide because the preference never reaches across sources
        return _derive("degen_rate:at_token_cap_retokenized", row, extras, ctx)
    if name == "loop_share":
        measure = _loop_share_measure(row, extras)
        v = _derive(f"degen_rate:{measure}", row, extras, ctx)      # a headline row reads the run's measured reading of THAT detector
        return v if v is not None else row.get(LOOP_SHARE_MIRROR_COLUMNS[measure])
    if name == "at_cap_share_finish_reason":
        if not _is_headline(row, ctx):
            return None
        deg = _degeneration_for(ctx, row)
        if "at_token_cap_finish_reason" in deg:
            return deg["at_token_cap_finish_reason"].get("rate")
        if "at_token_cap_retokenized" in deg:
            return row.get("frac_at_token_cap")      # the producer's value the canonical reading displaced (the pin moves it here)
        return None
    if name == "degeneracy_basis":
        return _degeneracy_basis(row, extras, ctx)
    if name == "effect_kind":
        # the export's own sentence, parked on the contrast by the r2 adapter; else the reader sentence for the source (a source with
        # neither keeps the registry's typed kind)
        cx = _json_or_empty(row.get("contrast_extra_json"))
        if cx.get("effect_kind_legacy") is not None:
            return cx["effect_kind_legacy"]
        return EFFECT_KIND_LEGACY.get((str(row.get("lane_source_legacy")), str(row.get("eval_category")), str(row.get("effect_kind"))))
    if name == "delta_method":
        cx = _json_or_empty(row.get("contrast_extra_json"))
        if "delta_method_legacy" in cx:
            # the export's own words: its sentence where it stated one, and ABSENT where it stated none (the typed column's filler,
            # `producer-filed (method not stated on the row)`, is the registry's, not the reader's)
            return cx["delta_method_legacy"] if cx["delta_method_legacy"] is not None else _ABSENT
        if str(row.get("delta_method")) in DELTA_METHOD_TOKENS_SPELLED and row.get("n_pairs") is not None:
            tmpl = DELTA_METHOD_LEGACY.get((str(row.get("lane_source_legacy")), str(row.get("eval_category"))))
            if tmpl is not None:
                return tmpl.format(n_pairs=int(row["n_pairs"]))
        return None
    if name == "cross_engine_contrast":
        cx = _json_or_empty(row.get("contrast_extra_json"))
        for key in ("cross_engine_contrast", "cross_engine"):
            if cx.get(key) is not None:
                return _as_bool(cx[key])
        return None
    if name == "lower_bound":
        jb = _extra_get(extras, "judge_blocked")
        if jb is None:
            jb = row.get("judgment_n_judge_refusals")
        if jb is None:
            return None
        try:
            return bool(float(jb) > 0)
        except (TypeError, ValueError):
            return None
    if name == "model":
        return MODEL_LEGACY.get(str(row["model_id"]), row["model_id"])
    if name == "lane_kind":
        # the CELL's lane, the way every reader files it: (1) the reader's re-stamp of a view (the export's llama shortconst view),
        # (2) the export's own lane column, (3) on an unsteered / prompted row — whose OBJECT has no transfer — the lane of the view
        # it was served in (a dose-suffixed view is its base view's), (4) the object's transfer kind
        src, view = row.get("lane_source_legacy"), row.get("view_legacy")
        restamp = SOURCE_SERVED_KEY_TRANSFER_LEGACY.get(str(src), {}).get(str(view))
        if restamp is not None:
            return restamp
        parked = _extra_get(extras, "lane_kind")
        if isinstance(parked, str) and parked:
            return parked
        kind = row.get("transfer_kind")
        if kind in ("none", "prompted_only", None):
            spelled = SERVED_KEY_TRANSFER_LEGACY.get(str(_served_key_base(view)))
            if spelled is not None:
                return spelled
        return TRANSFER_KIND_LEGACY.get(str(kind), kind)
    if name == "dose":
        v = row.get("strength_nominal")
        if v is not None:
            return v
        served = _served_key_dose(row.get("view_legacy"))
        if row.get("object_kind") in ("none", "system_instruction"):
            # the readers file an unsteered / prompted row beside every nominal, 1.0 by default; where the served key spells the
            # dose it was served beside (`_halfdose` 0.5, `_d0p25` 0.25) that refines the default — the key never removes it
            return 1.0 if served is None else served
        if row.get("lane_source_legacy") in SERVED_KEY_DOSE_SOURCES:
            # the sampled lanes file every row at the served key's dose (1.0 on a base key; `_d0p25` spells 0.25); no key, no dose
            return served
        return None
    if name == "arm":
        # the OBJECT's label (its own paper_label, else the stem of the cell it was served under) spells the arm; the reader's parked word
        # (`paper_arm`, a producer column) is read only where the object carries no paper label at all — and where it disagrees with the
        # object's it rides the row as `producer_arm_legacy` (render_long_row), never the frame's `arm`
        label = _paper_label_of(row)
        if label in (None, "other"):
            parked = _extra_get(extras, "paper_arm")
            if parked is not None:
                return parked
        return PAPER_LABEL_LEGACY.get(str(label), label)
    if name == "family":
        label = _paper_label_of(row)
        if label in (None, "other"):
            parked = _extra_get(extras, "paper_family")
            if parked is not None:
                return parked
        return FAMILY_LEGACY.get(str(_paper_label_of(row)), row.get("object_role"))
    if name == "metric":
        mid = row.get("metric_id")
        if str(mid) in METRIC_LEGACY:
            return METRIC_LEGACY[str(mid)]
        m = RECOVERED_T0_METRIC_LEGACY.match(str(mid))
        return m.group(1) if m else mid
    if name == "decode_cap_tokens":
        cap = row.get("token_cap_tokens")
        if cap is None:
            return None
        src, cat, mid = str(row.get("lane_source_legacy")), str(row.get("eval_category")), str(row.get("metric_id"))
        for s_pref, c, m in CAP_UNSPELLED_LEGACY:
            if (src == s_pref or (s_pref.endswith("/") and src.startswith(s_pref))) and c in ("*", cat) and m in ("*", mid):
                return None
        try:
            return None if float(cap) == 1.0 else cap
        except (TypeError, ValueError):
            return cap
    if name == "judge":
        model = row.get("judge_model")
        if model is None:
            return None
        tmpl = JUDGE_SPELLING_LEGACY.get((str(row.get("lane_source_legacy")), str(row.get("judge_rubric_name"))))
        return tmpl.format(judge_model=model) if tmpl else model
    if name == "object_type":
        return OBJECT_TYPE_LEGACY.get(str(row.get("object_kind")), row.get("object_kind"))
    if name == "component":
        return COMPONENT_LEGACY.get(str(row.get("eval_category")), row.get("eval_category"))
    if name == "eval_id":
        legacy = EVAL_LEGACY.get(str(row.get("eval_id")), row.get("eval_id"))
        if legacy is not None and _extra_get(extras, "instrument_seed_id") in SAMPLED4096_INSTRUMENT_SEEDS:
            return f"{legacy}{SAMPLED4096_SUFFIX_LEGACY}"     # the paper spells the sampled instrument as a second eval id
        return legacy
    if name == "threshold":
        t = row.get("threshold")
        return float("nan") if t is None or t == -1 else t
    if name == "status":
        return "graded" if row.get("value") is not None else None
    if name == "object_sha256":
        v = row.get("object_artifact_sha256")
        return None if v in (None, "none") else v
    if name == "held_out":
        return bool(row.get("held_out")) or bool(_extra_get(extras, "held_out"))
    if name in ("petri_run_key", "petri_intervention_hash", "petri_served_object_sha256"):
        prods = row.get("source_producers_json")
        try:
            lst = json.loads(prods) if isinstance(prods, str) else (prods or [])
        except ValueError:
            lst = []
        for pr in lst if isinstance(lst, list) else []:
            if isinstance(pr, dict) and pr.get(name) is not None:
                return pr[name]
        return None
    raise KeyError(name)


def long_supplied_columns() -> List[str]:
    """Every paper long-schema column the registry can supply."""
    return list(LONG_COLUMN_SOURCES)


def paper_long_frame(conn: sqlite3.Connection, round_id: Optional[str] = None, legacy_names: bool = False,
                     current_only: bool = True, columns: Optional[Sequence[str]] = None, with_dispositions: bool = False):
    """``v_paper_long`` as a pandas DataFrame.

    ``legacy_names=True`` renames to the paper's long schema (the plain renames of LONG_COLUMN_LEGACY_NAMES, transfer
    kinds spelled as the paper does). ``columns=<COLUMNS>`` (the caller passes the PINNED checkout's
    ``paperlib.load.COLUMNS``) derives EVERY long column through LONG_COLUMN_SOURCES, refuses (LongColumnUnsupplied) any
    name the registry cannot supply, appends the registry-side columns, and returns the frame reindexed to
    ``columns + LONG_REGISTRY_COLUMNS``, the registry-only rows (`paper_frame_exclusion`: no reader source, or a per-trial
    pass of a pooled source) left out. ``with_dispositions`` appends one value-less row per disposition of the
    round (the readers' own disposition rows: status = the disposition, no value), so the gap map sees them.
    """
    import pandas as pd
    sql = "SELECT * FROM v_paper_long"
    conds = []
    params: List[object] = []
    if round_id:
        conds.append("round_id = ?")
        params.append(round_id)
    if current_only:
        conds.append("in_current_paper = 1")
    if conds:
        sql += " WHERE " + " AND ".join(conds)
    df = pd.read_sql_query(sql, conn, params=params)
    if columns is not None:
        missing = [c for c in columns if c not in LONG_COLUMN_SOURCES]
        if missing:
            raise LongColumnUnsupplied(f"{len(missing)} long-schema column(s) the registry cannot supply: {missing}; "
                                       f"supplied: {sorted(LONG_COLUMN_SOURCES)}")
        recs = []
        ctx = {"degeneration": _degeneration_map(conn, [str(v) for v in df["run_id"].tolist()]) if len(df) else {},
               # the run's headline metric id (the first accepted id the run files): the rows the degeneracy readings are stamped on
               "headline_by_run": _headline_by_run(conn, df)}
        for row in df.to_dict("records"):
            # pandas spells a missing numeric as NaN; the derivations reason about None
            row = {k: (None if isinstance(v, float) and v != v else v) for k, v in row.items()}
            if paper_frame_exclusion(row) is not None:      # a registry-only row (no reader source / a per-trial pass): not rendered
                continue
            recs.append(render_long_row(row, columns, ctx))
        cols = list(columns) + [c for c in LONG_REGISTRY_COLUMNS if c not in columns]
        long = pd.DataFrame.from_records(recs, columns=cols) if recs else pd.DataFrame(columns=cols)
        if with_dispositions:
            disp = disposition_rows(conn, round_id, columns=cols)
            if len(disp):
                long = pd.concat([long, pd.DataFrame.from_records(disp, columns=cols)], ignore_index=True)
        for c in long.columns:
            # identity / text columns are strings-or-None by the paper's contract (never a float NaN that refuses a 64-hex later);
            # pandas >= 3 infers a STRING dtype (NA spelled NaN) for a mixed text column, so string dtypes are coerced with object
            if long[c].dtype == object or pd.api.types.is_string_dtype(long[c].dtype) or \
                    c in ("thinking_mode", "object_sha256", "serving_window_tokens", "bon_route", "decode", "fit_layers",
                          "arm_operating_point_id", "arm_operating_point", "grades_root", "serving_engine", "degeneracy_basis",
                          "cross_engine_contrast", "lower_bound"):
                col = long[c].astype(object)
                long[c] = col.where(col.notna(), None)
        return long
    if legacy_names:
        df = df.rename(columns=LONG_COLUMN_LEGACY_NAMES)
        if "lane_kind" in df.columns:
            df["lane_kind"] = df["lane_kind"].map(lambda v: TRANSFER_KIND_LEGACY.get(v, v))
        if "threshold" in df.columns:
            df.loc[df["threshold"] == -1, "threshold"] = float("nan")
    return df


def disposition_rows(conn: sqlite3.Connection, round_id: Optional[str] = None, columns: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
    """The round's dispositions as value-less long rows (status = '<state>: <reason>'), the shape the readers emit for a
    key that has not run; joined to the roster entry they cover for the comparison's transfer kind."""
    sql = ("SELECT d.*, re.comparison_id AS re_comparison, re.object_selector_json, re.strength_rule, o.paper_label, o.object_kind, e.eval_category "
           "FROM disposition d LEFT JOIN roster_entry re ON re.roster_entry_id = d.roster_entry_id "
           "LEFT JOIN object o ON o.object_id = COALESCE(d.object_id, re.object_id) LEFT JOIN eval e ON e.eval_id = d.eval_id")
    params: List[object] = []
    if round_id:
        sql += " WHERE d.round_id = ?"
        params.append(round_id)
    out = []
    for d in conn.execute(sql, params).fetchall():
        d = dict(d)
        sel = _json_or_empty(d.get("object_selector_json"))
        label = d.get("paper_label") or sel.get("paper_label")
        kind = d.get("object_kind") or sel.get("object_kind")
        comp = d.get("comparison_id") or d.get("re_comparison") or ""
        tk = comp.rsplit("/", 1)[1] if "/" in comp else sel.get("transfer_kind")
        rule = str(d.get("strength_rule") or "operating_point")
        dose = 1.0 if rule == "operating_point" else (float(rule[6:]) if rule.startswith("fixed:") else None)
        row = {
            "source": "registry_disposition", "model": MODEL_LEGACY.get(str(d["model_id"]), d["model_id"]),
            "lane_kind": TRANSFER_KIND_LEGACY.get(str(tk), tk), "view": None, "dose": dose, "cell_id": None,
            "arm": PAPER_LABEL_LEGACY.get(str(label), label), "family": FAMILY_LEGACY.get(str(label)),
            "object_type": OBJECT_TYPE_LEGACY.get(str(kind)), "component": COMPONENT_LEGACY.get(str(d.get("eval_category")), d.get("eval_category")),
            "eval_id": EVAL_LEGACY.get(str(d["eval_id"]), d["eval_id"]), "eval": EVAL_LEGACY.get(str(d["eval_id"]), d["eval_id"]),
            "status": f"{d['state']}: {d['reason']}", "note": d.get("source_ref"), "provenance": d.get("record_uri"),
            "in_current_paper": 0, "selection_state": "not_in_roster", "round_id": d["round_id"], "model_id": d["model_id"],
            "object_id": d.get("object_id"), "paper_label": label, "eval_category": d.get("eval_category"),
            "comparison_ids_json": json.dumps([comp]) if comp else "[]", "ingest_seq": d.get("ingest_seq"),
        }
        if columns is not None:
            row = {c: row.get(c) for c in columns}
        out.append(row)
    return out


def source_statuses(conn: sqlite3.Connection, round_id: Optional[str] = None) -> List[Dict[str, object]]:
    """One ``LaneStatus``-shaped dict (the paper's own status record per source) per ``lane_source_legacy`` the current paper reads (name, owner, integrated, where,
    detail, cells_measured, extra) — what ``registry_source.load_rows`` returns beside the frame so the paper's
    own guards (lane_guard, the gap map) keep their inputs. ``owner`` is the registry (the producing sessions live in
    the runs' ``producer_session``); ``where`` names the database read."""
    sql = ("SELECT lane_source_legacy AS source, COUNT(*) AS n_rows, COUNT(DISTINCT run_id) AS n_runs, "
           "COUNT(DISTINCT model_id || '|' || object_id || '|' || eval_id) AS cells, MAX(ingest_seq) AS head, "
           "SUM(CASE WHEN value IS NOT NULL THEN 1 ELSE 0 END) AS n_values "
           "FROM v_paper_long WHERE in_current_paper = 1")
    params: List[object] = []
    if round_id:
        sql += " AND round_id = ?"
        params.append(round_id)
    sql += " GROUP BY lane_source_legacy ORDER BY lane_source_legacy"
    where = "registry"
    try:
        row = conn.execute("PRAGMA database_list").fetchone()
        where = row[2] if row is not None else where
    except sqlite3.Error:
        pass
    out = []
    for r in conn.execute(sql, params).fetchall():
        out.append({"name": str(r["source"]), "owner": "registry", "integrated": True, "where": where,
                    "detail": f"{r['n_rows']} long rows over {r['n_runs']} runs; {r['n_values']} with a value; index head {r['head']}",
                    "cells_measured": int(r["cells"]), "extra": {"lane_source_legacy": r["source"], "n_runs": int(r["n_runs"]),
                                                                  "head_ingest_seq": r["head"]}})
    return out


def paper_objects(conn: sqlite3.Connection, round_id: Optional[str] = None):
    """``v_paper_objects`` as a DataFrame: object × round → in_paper, the comparisons it appears in, the rulings
    that exclude it."""
    import pandas as pd
    sql = "SELECT * FROM v_paper_objects"
    params: List[object] = []
    if round_id:
        sql += " WHERE round_id = ?"
        params.append(round_id)
    return pd.read_sql_query(sql + " ORDER BY in_paper DESC, paper_label, object_id", conn, params=params)


def in_current_paper(conn: sqlite3.Connection, run_id: str) -> Tuple[int, str, List[str]]:
    """(in_current_paper, selection_state, comparison_ids) for one run."""
    row = conn.execute("SELECT in_current_paper, selection_state, comparison_ids_json FROM v_paper_current WHERE run_id = ?",
                       (run_id,)).fetchone()
    if row is None:
        raise KeyError(run_id)
    return int(row[0]), str(row[1]), json.loads(row[2] or "[]")


def open_round(conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute("SELECT round_id FROM paper_round WHERE closed_utc IS NULL ORDER BY opened_utc DESC LIMIT 1").fetchone()
    return row[0] if row else None


def assert_roster_dispositions(conn: sqlite3.Connection, round_id: Optional[str] = None) -> None:
    """Refuse (RulingIndeterminate) on any covered key whose ruling evidence is incoherent or stale — the
    more specific refusal, checked FIRST because an indeterminate run also leaves its roster entry unfilled —
    then (RosterGap) on any roster entry with neither a judged run nor a disposition."""
    ind = conn.execute("SELECT run_id, ruling_ids_json FROM v_paper_current WHERE selection_state = 'indeterminate'").fetchall()
    if ind:
        detail = "; ".join(f"{r[0]} under rulings {r[1]}" for r in ind[:10])
        raise RulingIndeterminate(f"{len(ind)} run(s) covered by a ruling whose evidence is incoherent or stale: {detail}")
    gaps = conn.execute("SELECT roster_entry_id, comparison_id, model_id, eval_id FROM v_roster_gaps").fetchall()
    if gaps:
        names = ", ".join(f"{g[0]} ({g[1]}: {g[2]} × {g[3]})" for g in gaps)
        raise RosterGap(f"{len(gaps)} roster entr{'y' if len(gaps) == 1 else 'ies'} with neither a judged run nor a disposition: {names}")


# --------------------------------------------------------------------------------------------
# expiry evaluation
# --------------------------------------------------------------------------------------------


@dataclass
class ExpiryEvaluation:
    ruling_id: str
    evidence_state: str
    expired: bool
    evidence: Dict[str, object]


def _source_state(conn: sqlite3.Connection, source_id: Optional[str], input_uri: str, json_path: str) -> Optional[sqlite3.Row]:
    """The source_state row for an input; a ruling may spell the input `~/...` while the adapter filed it expanded (or the
    reverse), so both spellings are tried; the json_path is matched as written and with a `$.` prefix."""
    import os
    uris = list(dict.fromkeys([input_uri, os.path.expanduser(input_uri), input_uri.replace(os.path.expanduser("~"), "~", 1)]))
    paths = list(dict.fromkeys([json_path, json_path if json_path.startswith("$") else "$." + json_path]))
    marks = lambda xs: ",".join("?" for _ in xs)   # noqa: E731
    if source_id:
        return conn.execute(f"SELECT * FROM source_state WHERE source_id = ? AND input_uri IN ({marks(uris)}) AND json_path IN ({marks(paths)}) "
                            "ORDER BY observed_utc DESC LIMIT 1", [source_id, *uris, *paths]).fetchone()
    return conn.execute(f"SELECT * FROM source_state WHERE input_uri IN ({marks(uris)}) AND json_path IN ({marks(paths)}) ORDER BY observed_utc DESC LIMIT 1",
                        [*uris, *paths]).fetchone()


_STATE_RANK = {"ok": 0, "missing": 1, "path_absent": 1, "stale": 2, "incoherent": 3}


def evaluate_expiry(conn: sqlite3.Connection, ruling: Mapping[str, object]) -> Optional[ExpiryEvaluation]:
    """Evaluate one ruling's ``expires_when_json`` against ``source_state`` (never against historical
    run rows). Returns None when the ruling is not self-retiring.

    Two forms: a single condition object ``{kind: served_id_absent|grades_root_absent|…, …}``, or the
    structured form ``{kind: any_of|all_of, conditions: [...], guards: [...]}`` — ``guards`` (input-sha pins,
    served-record checks) are evaluated for EVIDENCE only and can never retire the ruling: a failing guard
    makes the evaluation ``incoherent``; a condition whose evidence is missing / path-absent never counts as
    held (S11), so under ``all_of`` it blocks expiry and under ``any_of`` the other conditions decide.
    """
    ew = ruling.get("expires_when_json")
    if not ew:
        return None
    spec = json.loads(ew) if isinstance(ew, str) else dict(ew)
    rid = str(ruling["ruling_id"])
    kind = spec.get("kind")
    if kind in V.EXPIRY_COMBINATORS:
        conditions = spec.get("conditions") or []
        guards = spec.get("guards") or []
        guard_evals = [_evaluate_condition(conn, rid, g) for g in guards]
        if not conditions:
            if not guard_evals:
                return ExpiryEvaluation(rid, "incoherent", False, {"error": f"{kind} with no conditions and no guards"})
            # a GUARD-ONLY ruling (three of the five superseded-cell declarations stand on a hash pin alone): it can
            # never retire; its evidence state is the guards' worst, so a moved pin still makes the covered keys indeterminate
            worst = max((g.evidence_state for g in guard_evals), key=lambda st: _STATE_RANK.get(st, 3))
            return ExpiryEvaluation(rid, worst, False, {"kind": kind, "conditions": [], "guards": [g.evidence for g in guard_evals],
                                                        "note": "guard-only ruling: never expires"})
        evals = [_evaluate_condition(conn, rid, c) for c in conditions]
        states = [e.evidence_state for e in evals + guard_evals]
        worst = max(states, key=lambda s: _STATE_RANK.get(s, 3)) if states else "ok"
        held = [e.expired for e in evals if e.evidence_state == "ok"]
        if worst == "incoherent" or any(g.evidence_state == "incoherent" for g in guard_evals):
            expired, state = False, "incoherent"
        elif kind == "any_of":
            expired = any(held)
            state = "ok" if expired or all(e.evidence_state == "ok" for e in evals) else worst
        else:  # all_of: every condition must be found, resolving and held
            expired = len(held) == len(evals) and all(held)
            state = "ok" if all(e.evidence_state == "ok" for e in evals) else worst
        return ExpiryEvaluation(rid, state, expired, {"kind": kind, "conditions": [e.evidence for e in evals],
                                                     "guards": [g.evidence for g in guard_evals]})
    return _evaluate_condition(conn, rid, spec)


def _evaluate_condition(conn: sqlite3.Connection, rid: str, spec: Mapping[str, object]) -> ExpiryEvaluation:
    kind = spec.get("kind")
    if kind not in V.EXPIRY_KINDS:
        return ExpiryEvaluation(rid, "incoherent", False, {"error": f"unknown expiry kind {kind!r}"})
    # an alternative guard form spells the input as `path` and the pin as `sha256`; both spellings are accepted
    input_uri = str(spec.get("input_uri") or spec.get("artifact_uri") or spec.get("path") or "")
    json_path = str(spec.get("json_path") or spec.get("member_key") or "$")
    if kind == "input_sha_pinned" and "sha256_prefix" not in spec and spec.get("sha256"):
        spec = dict(spec, sha256_prefix=spec["sha256"])
    row = _source_state(conn, spec.get("source"), input_uri, json_path)
    ev: Dict[str, object] = {"kind": kind, "input_uri": input_uri, "json_path": json_path}
    if row is None or not row["found"]:
        return ExpiryEvaluation(rid, "missing", False, dict(ev, note="input not found; exclusion stays active"))
    if not row["path_resolves"]:
        return ExpiryEvaluation(rid, "path_absent", False, dict(ev, note="json path absent; exclusion stays active"))
    members = json.loads(row["current_members_json"] or "[]")
    ev["current_members"] = members
    if kind == "served_id_absent":
        served = spec.get("served_id")
        if not members:
            return ExpiryEvaluation(rid, "ok", False, dict(ev, note="empty membership never expires"))
        return ExpiryEvaluation(rid, "ok", served not in members, ev)
    if kind == "grades_root_absent":
        root = spec.get("root")
        if not members:
            return ExpiryEvaluation(rid, "ok", False, dict(ev, note="empty root set never expires"))
        return ExpiryEvaluation(rid, "ok", set(members) != {root}, ev)
    if kind == "served_record_at_ruled_point":
        equals = spec.get("equals")
        value = members[0] if len(members) == 1 else members
        if canonical_json(value) != canonical_json(equals):
            return ExpiryEvaluation(rid, "incoherent", False, dict(ev, expected=equals, actual=value))
        return ExpiryEvaluation(rid, "ok", False, ev)
    if kind == "input_sha_pinned":
        prefix = str(spec.get("sha256_prefix") or "")
        sha = row["input_sha256"] or ""
        if not sha.startswith(prefix):
            return ExpiryEvaluation(rid, "incoherent", False, dict(ev, expected_prefix=prefix, actual=sha))
        return ExpiryEvaluation(rid, "ok", False, ev)
    return ExpiryEvaluation(rid, "incoherent", False, ev)


def stale_check(conn: sqlite3.Connection, ruling: Mapping[str, object]) -> bool:
    """A supersede ruling is STALE when an operating_point ruling's target object equals the
    superseded served object (the ruled object now IS the served object)."""
    if ruling.get("action") != "supersede":
        return False
    scope = json.loads(ruling["scope_json"]) if isinstance(ruling["scope_json"], str) else dict(ruling["scope_json"])
    served = scope.get("served_object_sha256")
    if not served:
        return False
    row = conn.execute(
        "SELECT 1 FROM ruling WHERE action = 'operating_point' AND expired_utc IS NULL AND round_id = ? "
        "AND json_extract(target_json, '$.object_id') = ? LIMIT 1", (ruling["round_id"], served)).fetchone()
    return row is not None


def evaluate_rulings(root: str, conn: sqlite3.Connection, now: Optional[str] = None, written_utc: Optional[str] = None) -> List[str]:
    """Evaluate every active ruling; write a ``ruling_state`` record whenever the evaluation differs
    from the ruling's current state (a record BEFORE a row; replayable). Returns the paths written. ``written_utc``
    (default ``now``) is the record's own stamp — its file's place in the ledger (the harvester's files are ordered
    by their stamp, so ``ingest_once`` passes the head's successor); ``evaluated_utc`` stays the cycle's clock."""
    from .harvest import write_harvester_record
    now = now or R.utc_now()
    out: List[str] = []
    rows = conn.execute("SELECT * FROM ruling WHERE expires_when_json IS NOT NULL OR action = 'supersede'").fetchall()
    for r in rows:
        ruling = dict(r)
        ev = evaluate_expiry(conn, ruling)
        state = ev.evidence_state if ev else "ok"
        expired = bool(ev and ev.expired)
        evidence = ev.evidence if ev else {}
        if stale_check(conn, ruling):
            state, expired, evidence = "stale", False, dict(evidence, stale="operating_point ruling targets the served object")
        already_expired = ruling.get("expired_utc") is not None
        if ruling.get("evidence_state") == state and (expired == already_expired or already_expired):
            continue
        payload = {"ruling_id": ruling["ruling_id"], "evaluated_utc": now, "evidence_state": state, "expired": int(expired),
                   "evidence_json": evidence}
        out.append(write_harvester_record(root, "ruling_state", payload, written_utc or now))
    return out
