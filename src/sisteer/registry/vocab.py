"""Controlled vocabularies of the results registry — THE single source.

Every ``CHECK (col IN (...))`` in ``schema.sql`` is rendered from a tuple in this module, the
record validator checks payload fields against the same tuples, and ``docs/registry/vocabulary.md``
(the sidecar glossary) is generated from :data:`EVIDENCE`. ``tests/test_registry_schema.py::
test_checks_equal_vocab`` parses the DDL text independently and asserts set equality with the tuple
of the same name, so a value can never be added in one place only.

STDLIB ONLY: ``identity.py`` and ``emit.py`` (compute-node side, imported under ``python3 -S``)
import this module.

Vocabulary rule: one specific noun per concept — model, checkpoint, object,
eval, item_set, instrument, run, judgment, judge, paper_round, substrate. Campaign jargon appears
only inside ``_legacy`` columns and the glossary's legacy-mapping table.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Dict, Mapping, Optional, Tuple, List

# --------------------------------------------------------------------------------------------
# Booleans (INTEGER 0/1 everywhere)
# --------------------------------------------------------------------------------------------
BOOLEAN: Tuple[int, ...] = (0, 1)

# --------------------------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------------------------
LIFECYCLE_STATES = (
    "queued", "running", "generated", "judging", "judged", "ingested",
    "failed", "cancelled", "superseded", "withdrawn",
)
#: Lifecycle states that carry their own ``*_utc`` column on ``run``; the trigger
#: ``trg_run_state_utc`` requires every column up to the reached state to be non-null.
LIFECYCLE_UTC_COLUMN: Mapping[str, str] = {
    "queued": "queued_utc",
    "running": "started_utc",
    "generated": "generated_utc",
    "judged": "judged_utc",
    "ingested": "ingested_utc",
}
#: The forward order of the reached states (a state implies every earlier one was reached).
LIFECYCLE_PROGRESSION = ("queued", "running", "generated", "judged", "ingested")
#: Terminal states that stamp ``finished_utc``.
LIFECYCLE_TERMINAL_FINISHED = ("failed", "cancelled")
#: The states outside the progression (``judging`` is a transient beside ``generated``): terminal (``failed``, ``cancelled``) and
#: lateral (``superseded``, ``withdrawn``). A ``run_state`` to one of them carries no rank in the fold's monotonic lifecycle AND
#: takes ``tie_rank`` 1 in the fold's order key ``(written_utc, tie_rank, record_sha256)`` (``reducer.ORDER_SQL``,
#: ``validate.Validated.order_key``): at one ``written_utc`` it folds AFTER a progression-state event.
LIFECYCLE_LATERAL = tuple(s for s in LIFECYCLE_STATES if s not in LIFECYCLE_PROGRESSION and s != "judging")
assert LIFECYCLE_LATERAL == ("failed", "cancelled", "superseded", "withdrawn"), LIFECYCLE_LATERAL
#: The ``run.*_utc`` columns a row in each state must carry — ONE source for the DDL trigger ``trg_run_state_utc``
#: and for the harvester's prerequisite check (a ``run_state`` whose earlier evidence has not arrived is
#: dependency-blocked on it, never refused).
LIFECYCLE_REQUIRED_UTC: Mapping[str, Tuple[str, ...]] = {
    **{state: tuple(LIFECYCLE_UTC_COLUMN[s] for s in LIFECYCLE_PROGRESSION[: i + 1]) for i, state in enumerate(LIFECYCLE_PROGRESSION)},
    "judging": ("queued_utc", "started_utc", "generated_utc"),
    "failed": ("queued_utc", "finished_utc"),
    "cancelled": ("queued_utc", "finished_utc"),
    "superseded": ("queued_utc",),
    "withdrawn": ("queued_utc",),
}
assert set(LIFECYCLE_REQUIRED_UTC) == set(LIFECYCLE_STATES)

EVIDENCE_KINDS = ("submit_wrapper", "job_log", "completion_artifact", "scheduler_state", "ruling", "harvester")
WALLTIME_SOURCES = ("sacct", "modal_call", "store_span")
TRANSFER_KINDS = ("base_to_instruct", "instruct_direct", "chat_checkpoint", "prompted_only", "none", "not_applicable")
STRENGTH_NOMINAL_SOURCES = ("filed", "derived")
#: The UNIT of `run.strength_nominal`: dose ladders of different object families scale different quantities, so a dose
#: alone never identifies a point; `nominal_only` is a campaign nominal (1.0 = the shipped object) with no unit declared by the
#: record; `unverified` (the column default) where the record declares none — including unsteered and prompted rows, which
#: carry no strength at all.
STRENGTH_UNITS = ("lora_write_basis_multiplier", "psr_direction_norm_multiplier", "alpha", "lambda", "raw_coefficient", "nominal_only",
                  "unverified")
INSTRUMENT_MATCH = ("match", "mismatch", "unverified")
#: DERIVED per run from `v_run_decode_policy_match` (decode_match.stamp_run_rows, end of every ingest cycle and of a rebuild;
#: never a hand edit): `exact` = exactly one decode_policy row whose scope admits the run and whose every pinned parameter the
#: run sent at the same value (and nothing else); `ambiguous` = several; `no_match` = parameters recorded, no row matches;
#: `unrecorded` = the run recorded no sent parameter. `no_policy` is the producer-written placeholder before derivation.
DECODE_POLICY_MATCH = ("exact", "ambiguous", "no_match", "unrecorded", "no_policy")
MAX_NUM_SEQS_SOURCES = ("attested", "runner_concurrency")
PREFIX_CACHE_POLICIES = ("on", "off", "skip_reading")
#: `slurm` (migration 0018): the GENERIC value for a Slurm cluster that is neither of the two named ones; a
#: row carrying it MUST name the site in `extra_json.cluster_label` (the emitter refuses `slurm` without one, by name), so a
#: third cluster's rows stay distinguishable; cluster_a / cluster_b are never re-labelled and an UNKNOWN substrate refuses
#: by name -- never aliased to `grading_host` (see emit.RecordWriter).
SUBSTRATES = ("cluster_a", "cluster_b", "modal", "grading_host", "runpod", "slurm")
#: ``fit_search.substrate`` ONLY: a sweep directory that carries no substrate record at all (no PODS.md beside the
#: root, no Modal profiler trace beside a trial, no Modal manifest) files ``unverified`` with the missing evidence named in
#: ``field_confidence_json`` instead of being refused. ``run.substrate`` and ``object.fit_substrate`` stay on SUBSTRATES.
FIT_SEARCH_SUBSTRATES = SUBSTRATES + ("unverified",)
CONFIDENCE = ("recorded", "derived", "assumed_default", "inferred", "unverified")
SERVING_PATHS = ("lora_native", "lens", "passthrough", "hf", "unverified")
DECODE_SOURCES = (
    "generation_config", "vendor_recommendation", "model_card", "vllm_default", "greedy",
    "grandfathered_T1", "runner_forced_T1", "legacy_contracts_row", "unverified",
)
#: How a sampled decode's ``seed`` is derived: a label names the CLOSURE the seed is a function of. Strictness rule
#: (the seed-replicate battery): a seed that closes over more than the item -- a turn, a sample -- takes the label naming
#: that closure (``per_goal_turn``, ``per_scenario_turn``, ``per_response``), never ``per_item`` plus a free-text
#: ``seed_rule_detail``.
DECODE_SEED_RULES = ("per_item", "per_draw", "per_goal_turn", "per_scenario_turn", "per_response", "fixed", "none", "unverified")
THINKING_MODES = ("on", "off", "not_available")
APPLIED_POSITIONS_MODES = (
    "all", "response", "response_all_turns", "prompt_last", "prefill_all",
    "prefill_last_broadcast", "last", "non_template", "none", "unverified",
)
MULTI_TURN_SCOPES = (
    "every_turn_prefill_and_decode", "final_header_and_current_decode_only", "every_assistant_span",
    "prefill_only_each_turn", "single_turn", "not_applicable", "unverified",
)

# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------
MODEL_FAMILIES = ("gemma4", "qwen35", "qwen36", "qwen38", "nemotron3", "llama31", "llama32", "glm46v")
#: `organism` (2026-09-17): an EM MODEL ORGANISM -- a LoRA fine-tune of a served instruct checkpoint on a Betley et al. emergent-misalignment
#: dataset, served on its parent as a vLLM LoRA module; `extra_json.parent_model_id` names the parent (the schema has no parent column). The
#: organism is the MODEL and em8q the eval (organisms are checkpoints, not model + object).
CHECKPOINT_KINDS = ("base", "instruct", "chat", "organism")
THINKING_CHANNELS = ("available", "not_available", "unverified")
MODEL_PAPER_STATUSES = ("in_paper", "descoped", "reference")


def model_spelling_owner(claimants: Mapping[str, Optional[str]]) -> Optional[str]:
    """Which of several models claiming ONE spelling (an HF repo, a producer spelling, a legacy key) owns it:
    ``{model_id: paper_status}`` -> the model whose status comes first in :data:`MODEL_PAPER_STATUSES` (in_paper, then
    descoped, then reference), or ``None`` when two models of that first rank claim it -- a clash, dropped, never
    guessed. The paper's checkpoint owns its names: a ``reference`` row that re-pins a paper checkpoint under its own
    key (the worked example ``qwen35_4b_demo``) never captures the HF repo spelling the
    paper's row and the readers resolve. One claimant owns the spelling outright, whatever its status."""
    if not claimants:
        return None
    if len(claimants) == 1:
        return next(iter(claimants))
    rank = {s: i for i, s in enumerate(MODEL_PAPER_STATUSES)}
    best = min(rank.get(str(s), len(rank)) for s in claimants.values())
    owners = [mid for mid, s in claimants.items() if rank.get(str(s), len(rank)) == best]
    return owners[0] if len(owners) == 1 else None

# --------------------------------------------------------------------------------------------
# Objects and fits
# --------------------------------------------------------------------------------------------
OBJECT_KINDS = (
    "none", "system_instruction", "lora", "lora_subtraction", "relu_steer",
    "relu_steer_distilled_difference", "vector", "vector_all", "diffmean", "static_r_arditi",
    "caa_static_r", "alphasteer", "antidote", "composed", "stacked",
)
PAPER_LABELS = (
    "none", "safety_si", "constitution_si", "constitution_si_full", "control_lora", "constitution_lora",
    "difference_lora", "control_relu", "constitution_relu", "difference_relu", "difference_relu_distilled", "arditi", "arditi_allpos",
    "alphasteer_rebake", "alphasteer_as_fit", "alphasteer_refit_reference", "safety_lora", "safety_relu",
    "other",
)
OBJECT_ROLES = ("constitution", "control", "difference", "safety", "irrelevant", "baseline_none", "baseline_published", "prompt")
FIT_TARGET_KINDS = (
    "system_instruction_distillation", "corpus_document_fit", "contrastive_direction", "derived",
    "declared_prompt", "none",
)
FIT_OBJECTIVES = ("ce", "kl", "diffmean", "arditi_search", "alphasteer", "antidote", "not_applicable")
FIT_THINKING_RENDERS = ("on", "off", "not_applicable", "unverified")
FIT_POSITIONS = (
    "all", "response_incl_assistant_header", "last_token_of_appended_assistant_header",
    "searched_post_instruction_span", "completion_mean", "inherited_from_parents", "not_applicable",
    "unverified",
)
FIT_CAPTURE_SITES = ("resid_pre", "resid_post")
LAYER_INDEX_CONVENTIONS = ("zero_based_decoder_block", "one_based", "hf_module_path")
LAYER_SELECTION_RULES = ("all", "listed_in_config", "separability_search", "published_set", "inherited")
FIT_STOPPED_ON = ("patience", "epoch_cap", "unverified")
FIT_SEED_CONFIDENCE = ("recorded", "assumed_default_0", "not_applicable", "unverified")
FIT_LR_SELECTED_BY_FIXED = ("search", "default", "manual", "unverified")  # + 'inherited_from:<object_id>'
DERIVATIONS = ("none", "lora_subtraction", "residual_distillation", "dose_scale", "composition", "stack")
OBJECT_STATUSES = ("approved", "reference", "superseded", "withdrawn", "pending_fit")
#: The named role of an epoch-checkpoint child inside its parent's lineage (the first-class column `object.named_role` since the
#: record-refusal-fact migration): `epoch_1` / `epoch_2` the per-epoch checkpoints, `intermediate` a checkpoint the fit kept between the
#: named ones, `final` the fit's final weights. Producers wrote it under `extra_json.named_role` before the column existed; the fold lifts
#: that key when the record states no column (``reducer``). `extra_json.named_roles` (a list) stays in extra. `object_role`
#: (constitution / control / difference ...) is a separate, existing column.
OBJECT_NAMED_ROLES = ("epoch_1", "epoch_2", "intermediate", "final")
SI_KINDS = ("safety_instruction", "constitution_short", "constitution_full", "irrelevant_control", "reference_other")
SEARCH_METHODS = ("sobol", "grid", "random", "bayesian", "manual", "reconnaissance")
OBJECTIVE_DIRECTIONS = ("minimize", "maximize")
TRIAL_STATUSES = ("complete", "failed", "pruned")

#: Column groups of the ``fit_target_kind`` paired-nullness CHECK.
FIT_GROUP_T = ("fit_teacher_si_id", "fit_teacher_si_text_sha256", "fit_prompt_pool_id", "fit_prompt_pool_sha256")
FIT_GROUP_C = ("fit_corpus_id", "fit_corpus_sha256", "fit_corpus_n_documents", "fit_corpus_n_tokens", "fit_corpus_description")
FIT_GROUP_X = ("fit_contrast_sets_json",)
FIT_GROUP_P = ("parent_object_ids_json", "derivation_params_json")
FIT_GROUP_S = ("si_text_sha256", "si_token_count")

#: Per kind: required NON-NULL columns, required NULL columns, admitted ``fit_objective`` values,
#: admitted ``derivation`` values, and extra column equalities. Rendered into the DDL CHECK by
#: ``schema_def.render_fit_target_check`` and consumed by the validator; the test
#: ``test_fit_target_kind_forbidden_extra_field`` walks every required-NULL cell.
FIT_TARGET_RULES: Dict[str, Dict[str, tuple]] = {
    "system_instruction_distillation": {
        "nonnull": FIT_GROUP_T + ("fit_model_id", "fit_layers_json"),
        "null": FIT_GROUP_C + FIT_GROUP_X + FIT_GROUP_P + FIT_GROUP_S,
        "fit_objective": ("kl", "ce"),
        "derivation": ("none",),
        "equals": (),
    },
    "corpus_document_fit": {
        "nonnull": ("fit_corpus_id", "fit_corpus_sha256", "fit_model_id", "fit_layers_json"),
        "null": FIT_GROUP_T + FIT_GROUP_X + FIT_GROUP_P + FIT_GROUP_S,
        "fit_objective": ("ce",),
        "derivation": ("none",),
        "equals": (),
    },
    "contrastive_direction": {
        "nonnull": FIT_GROUP_X + ("fit_model_id", "fit_layers_json"),
        "null": FIT_GROUP_T + FIT_GROUP_C + FIT_GROUP_P + FIT_GROUP_S,
        "fit_objective": ("diffmean", "arditi_search", "alphasteer", "antidote"),
        "derivation": ("none",),
        "equals": (),
    },
    "derived": {
        "nonnull": ("parent_object_ids_json",),
        "null": FIT_GROUP_T + FIT_GROUP_C + FIT_GROUP_X + FIT_GROUP_S,
        "fit_objective": ("not_applicable",),
        "derivation": ("lora_subtraction", "residual_distillation", "dose_scale", "composition", "stack"),
        "equals": (),
    },
    "declared_prompt": {
        "nonnull": ("si_text_sha256",),
        "null": FIT_GROUP_T + FIT_GROUP_C + FIT_GROUP_X + FIT_GROUP_P,
        "fit_objective": ("not_applicable",),
        "derivation": ("none",),
        "equals": (("object_kind", "system_instruction"),),
    },
    "none": {
        "nonnull": (),
        "null": FIT_GROUP_T + FIT_GROUP_C + FIT_GROUP_X + FIT_GROUP_P + FIT_GROUP_S,
        "fit_objective": ("not_applicable",),
        "derivation": ("none",),
        "equals": (("object_kind", "none"),),
    },
}

# --------------------------------------------------------------------------------------------
# Evals, item sets, instruments, judges
# --------------------------------------------------------------------------------------------
EVAL_CATEGORIES = (
    "static_attack", "multi_turn", "best_of_n", "over_refusal", "capability", "petri", "agentic", "misalignment",
    "em_organism", "omit_params",
)
#: Categories whose runs are multi-turn conversations (the ``multi_turn_steering_scope`` rule).
MULTI_TURN_EVAL_CATEGORIES = ("multi_turn", "petri", "agentic")
METRIC_DIRECTIONS = ("higher_is_better", "lower_is_better")
EVAL_PAPER_STATUSES = ("in_paper", "appendix", "dropped")
HARNESSES = (
    "bespoke_sisteer", "inspect_ai", "inspect_evals", "evalplus", "bfcl_eval", "pyrit", "actorattack", "petri",
    "strongreject", "agentic_misalignment", "em_organism", "lm_eval",
)
COT_PROMPTING = ("none", "explicit_instruction", "few_shot_cot")
SCORING_RULES = (
    "exact_match", "numeric_match", "sympy_equiv", "letter_logprob_prefill", "judge_threshold",
    "judge_graded_weights", "pass_at_1_sandbox", "ast_match", "rate_over_scenarios", "ifeval_checker", "judge_score_mean",
)
ANSWER_EXTRACTIONS = ("regex", "last_number", "boxed", "judge", "none", "unverified")
#: `ladder_final` = the metric is read at the final level of the token-cap ladder (spelled this way because the shorter
#: name is a banned vocabulary token).
READ_AT = ("ladder_final", "fixed", "unverified")
SUBSAMPLE_RULE_FIXED = ("full", "fixed_ids_file")  # + 'seeded_sample:<seed>' | 'paired_with:<item_set_id>'
JUDGE_ROLES = ("judge", "auditor")
JUDGE_BACKENDS = ("batch", "live", "batch_or_live", "unverified")
JUDGMENT_STATUSES = ("in_progress", "complete", "partial", "superseded", "abandoned")
API_POOLS = ("lowprio", "highprio", "batch")
SAMPLING_MODES = ("card", "thinking_card", "general", "battery", "greedy", "vendor_preset")

# --------------------------------------------------------------------------------------------
# Metrics, contrasts, degeneration
# --------------------------------------------------------------------------------------------
COMPUTED_BY = ("producer", "registry")
#: `solo_stand_in` (migration 0027): the shared-standup comparison sources' shared-vs-solo contrast -- the SOLO run of another source
#: on the same instrument stands in as the anchor (`_bcbw_common.solo_anchor_plan`); until 0027 it shared `borrowed` with the tier-1c
#: re-anchoring on another source's roster-matched `none` (anchor_current.KIND_BORROWED), so `v_paper_long`'s one tie-break could not tell
#: the two apart. `borrowed` keeps the re-anchoring and r2_export's cross-spelling anchors.
ANCHOR_KINDS = ("same_standup_none", "sibling_none", "borrowed", "full_dose", "solo_stand_in")
EFFECT_KINDS = ("cohens_h", "dz")
DEGENERATION_MEASURES = (
    "classifier", "contentless", "rep_ratio_loop", "zlib_collapse", "union5", "at_token_cap",
    "at_token_cap_retokenized", "at_token_cap_finish_reason",     # The two at-cap readings, measured separately
    "loop_first512", "defended_degenerate", "tail_loop", "line_loop",
    # The union classifier's DECOMPOSITION of the cap-4,096 multi-turn readings (the producer's `degeneracy_union_mt_cap4096/1`
    # v2 file, `turn_level.truncation_only_share` / `turn_level.loop_without_cap_share`; the paper's `degen_truncation_only_share`
    # / `degen_loop_without_cap_share` since paper-figures f33dca5d): at the canonical cap with NO other detector firing —
    # truncation that is not collapse — and flagged by a loop detector and NOT at cap — collapse that is not truncation. One
    # pooled share cannot tell a loop from a truncation (three regimes measured at 4,096), so the parts are measures of their own.
    "truncation_only", "loop_without_cap",
    # The r2 readers' TWO-detector union — loop OR at-cap (the export's `degenerate_share` column and metric row: "share of responses
    # that looped OR ran to the cap (union)") — a different statistic from the five-detector `union5`, named so the two are never
    # averaged under one column (spec: frac degenerate per definition, as separate named columns).
    "loop_or_at_cap",
)
DEGENERATION_SCOPES = ("all", "defended_only", "first_512_tokens", "terminated_only")
DENOMINATOR_UNITS = ("items", "draws", "replies", "turns", "transcripts", "scenarios")
AGGREGATIONS = ("single_detector", "union", "max_single_detector", "floor")
#: A degeneration row says whether it was MEASURED; consumers must branch on this, never on a NULL rate.
MEASUREMENT_STATES = ("measured", "not_measured", "not_applicable", "refused")
#: Migration 0021: what `v_paper_current.coverage_state` says about a run's headline measurement -- a consumer that counts
#: coverage branches on it, never on the absence of a metric row alone (a run whose prompts cannot fit its windows measured nothing
#: and is not missing a measurement).
COVERAGE_STATES = ("measured", "undefined_by_geometry", "absent")
#: The `run_metric.frac_*` mirror columns of `run_degeneration`'s `all`-scope rates (the two at-cap readings are
#: separate measures; the mirror keeps the earlier `frac_at_token_cap` spelling).
DEGENERATION_FRAC_COLUMNS = (
    "frac_at_token_cap", "frac_degenerate_classifier", "frac_degenerate_contentless", "frac_degenerate_rep_ratio_loop",
    "frac_degenerate_zlib_collapse", "frac_degenerate_union5", "frac_loop_first512", "frac_defended_degenerate",
)

# --------------------------------------------------------------------------------------------
# Metric ids (`run_metric` / `run_contrast` / `eval.headline_metric_id`) — a BAN, not an allow-list
# --------------------------------------------------------------------------------------------
#: `metric_id` is free text scoped by the run's eval (spec "run_metric": asr | over_refusal_graded | acc | petri_score |
#: rate | …; the eval is the qualifier, so `acc` on a gsm8k run is a complete name), so the naming rule is a banned set,
#: not a vocabulary: (1) a bare generic noun names no statistic; (2) a DEGENERATION statistic belongs in
#: `run_degeneration`, keyed by measure × scope with `measurement_state`, and reaches `run_metric` only through the
#: `frac_*` mirror columns — never as a metric row under an unqualified spelling. A producer's own
#: spelling that differs from the id rides `producer_metric_spelling`. Rendered as CHECK constraints by schema_def
#: (`*_metric_id_not_generic`) and refused by name in validate.py before any insert.
GENERIC_METRIC_IDS = (
    "score", "value", "metric", "result", "accuracy", "mean", "count", "total", "number", "frac", "fraction", "share",
    "pct", "percent", "stat", "statistic", "measure", "outcome",
)
#: Unqualified degeneration spellings: the phenomenon with no detector, scope or reading named.
UNQUALIFIED_DEGENERATION_METRIC_IDS = (
    "degenerate", "degeneration", "degeneracy", "degen", "loop", "looping", "loops", "at_cap", "capped", "truncated",
    "truncation", "collapse", "frac_degenerate", "degenerate_rate", "degenerate_frac", "degenerate_fraction", "loop_rate",
    "frac_loop", "loop_frac", "at_cap_rate", "frac_at_cap", "at_cap_frac", "at_cap_fraction",
)


#: The r2 export's own spellings of the degeneration statistics it files as metric rows per cell (`at_cap_share` x1066,
#: `loop_share` x752, `degenerate_share` x650 on the live parquet, plus the union instrument's rows). Each names a
#: `DEGENERATION_MEASURES` member (`adapters.r2_export.DEGENERATION_SPELLINGS`) and the adapter routes it into `run_degeneration`;
#: as a `run_metric.metric_id` it is the statistic filed at the wrong door and is refused. Qualified statistics no measure names
#: (`residual_at_cap_share_bon_4096`, `at_cap_share_all`, `goal_*_share`, …) are NOT here and stay metric rows.
LEGACY_DEGENERATION_METRIC_SPELLINGS = (
    "at_cap_share", "loop_share", "degenerate_share", "degenerate_share_union5", "at_cap_share_retokenized", "at_cap_share_finish_reason",
)


#: The best-of-N spelling the producers may NOT file on a NEW record -- the N belongs in the id (`asr_at_n<N>`) and in
#: `extra.bon_n`; the 4 live `asr_at_n_max` rows stay as legacy (their `producer_metric_spelling` the provenance;
#: `paper.METRIC_LEGACY` maps the spelling), because the rule is EMIT-side only (:func:`bon_metric_refusal`), never a validator
#: rule: a validator rule would alter folded rows at every rebuild of the log.
BON_LEGACY_METRIC_SPELLINGS: Tuple[str, ...] = ("asr_at_n_max",)
#: the live best-of-N conventions: `asr_bon_cap4096` at the export's three judge cutoffs; `bon_nauc` at 0.5 or the AUC's
#: no-threshold sentinel -1.0 (152 live rows at thresholds other than 0.5 on 2026-10-06, every one admitted by this table)
BON_METRIC_THRESHOLDS: Dict[str, Tuple[float, ...]] = {"asr_bon_cap4096": (0.25, 0.5, 0.75), "bon_nauc": (0.5, -1.0)}
_ASR_AT_N = __import__("re").compile(r"^asr_at_n(\d+)$")


def bon_metric_refusal(row: Mapping[str, object]) -> Optional[str]:
    """The emit-time best-of-N guard: the reason a `run_metric` row is REFUSED at emission, or None when it passes. Pure; called
    from `emit.RecordWriter.emit_metrics` and `hooks.JudgingScope.metrics` only -- never from `validate_record` (whose rules fold the
    record log at every rebuild; the 156 live rows keep folding exactly as they do). Rules, scoped to the live conventions:
    `asr_bon_cap4096` carries a threshold in {0.25, 0.5, 0.75}; `bon_nauc` one in {0.5, -1.0}; `asr_at_n<N>` carries `extra.bon_n`
    equal to N; the spelling `asr_at_n_max` is refused by name (write `asr_at_n<N>` with `extra.bon_n`, keeping the producer's own text
    in `producer_metric_spelling`). Every other metric id passes untouched."""
    mid = str(row.get("metric_id") or "")
    if mid in BON_LEGACY_METRIC_SPELLINGS:
        return (f"metric_id {mid!r} is a legacy best-of-N spelling refused for new records (D934 / D870): the N belongs in the id -- write "
                f"`asr_at_n<N>` with extra.bon_n = N, and keep the producer's own text in producer_metric_spelling")
    if mid in BON_METRIC_THRESHOLDS:
        allowed = BON_METRIC_THRESHOLDS[mid]
        try:
            thr = float(row["threshold"])      # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError):
            return f"metric_id {mid!r} needs a threshold in {{{', '.join(str(t) for t in allowed)}}} (D934): none readable on the row"
        if not any(abs(thr - a) < 1e-9 for a in allowed):
            return f"metric_id {mid!r} at threshold {thr!r} is not one of the live conventions {{{', '.join(str(t) for t in allowed)}}} (D934)"
        return None
    m = _ASR_AT_N.match(mid)
    if m:
        extra = row.get("extra")
        bon_n = extra.get("bon_n") if isinstance(extra, Mapping) else None
        try:
            ok = bon_n is not None and int(str(bon_n)) == int(m.group(1)) and str(bon_n).strip().lstrip("+").isdigit()
        except (TypeError, ValueError):
            ok = False
        if not ok:
            return f"metric_id {mid!r} must carry extra.bon_n == {m.group(1)} (D934 / D870): the row carries {bon_n!r}"
    return None


def counted_metric_refusal(row: Mapping[str, object]) -> Optional[str]:
    """The emit-time denominator guard: the reason a COUNTED `run_metric` row -- one that carries a denominator ``n`` -- is REFUSED at
    emission when that denominator is zero, or None when the row passes. No producer files an ``n = 0`` metric row: a statistic over nothing
    is a non-measurement, not a measurement (a judgment that completed with no verdict files no metric for it; the row would read as a
    value over an empty set). A row with no denominator (``n`` absent or null -- not every metric has one, the schema's own why-null) is not
    counted and passes; a zero NUMERATOR over a non-zero denominator is a measurement and passes. Pure; called from
    `emit.RecordWriter.emit_metrics` (and so from `hooks.JudgingScope.metrics`), never from `validate_record`: a validator rule would alter
    folded rows at every rebuild, and the live rows that carry ``n = 0`` keep folding as they do.

    A denominator the guard carries is judged by its NUMERIC VALUE, whatever its type (``0``, ``0.0``, ``'0'``, ``Decimal(0)`` are all
    ``n = 0``), with two refusals of their own: a BOOLEAN ``n`` (``True`` or ``False``) is refused by name -- a boolean is never a
    denominator, and the validator folds a boolean into its integer for every INTEGER column, so ``n = False`` would otherwise reach the
    store as ``n = 0`` through a guard that could not read it (the first guard did ``int('False')``, caught the ValueError and SKIPPED
    the row); and an ``n`` with no numeric reading (``'abc'``, a list, NaN) is refused by name as not a readable count, never skipped on
    the promise that the validator's column rule will name it (a skip is the one shape this guard must never take: the rule is
    enforced here or nowhere)."""
    if "n" not in row or row.get("n") is None:
        return None
    n = row.get("n")
    where = f"metric_id {str(row.get('metric_id') or '')!r} carries n = {n!r} (k = {row.get('k')!r}, value = {row.get('value')!r})"
    if isinstance(n, bool):
        return (f"{where}: a boolean is never a denominator (the validator would fold it into the integer {int(n)}, so this guard refuses "
                "it by name) -- file the count of items the statistic ranges over, or omit n for a metric that has no denominator")
    try:
        count = Decimal(str(n).strip())
    except (InvalidOperation, ValueError, TypeError):
        count = None
    if count is None or not count.is_finite():
        return (f"{where}: not a readable count -- a denominator is a number (file the count of items the statistic ranges over, or omit n "
                "for a metric that has no denominator); a value this guard cannot read is refused, never skipped")
    if count == 0:
        return (f"metric_id {str(row.get('metric_id') or '')!r} carries n = 0 (given as {n!r}; k = {row.get('k')!r}, value = "
                f"{row.get('value')!r}): a counted metric over an empty denominator is a non-measurement, not a measurement -- file no "
                "metric row for a judgment that measured nothing (a row that legitimately has no denominator omits n)")
    return None


def banned_metric_ids() -> Tuple[str, ...]:
    """Every metric id the guard refuses, in one deterministic order: the bare generic nouns, the unqualified degeneration
    spellings, every `DEGENERATION_MEASURES` name as a bare id and as `frac_<m>` / `<m>_rate` / `<m>_frac` / `<m>_share` /
    `degen_<m>_share`, the r2 export's legacy spellings, and every `run_metric.frac_*` mirror column name (a metric row spelled
    like the mirror column is the statistic filed at the wrong door)."""
    out: List[str] = list(GENERIC_METRIC_IDS) + list(UNQUALIFIED_DEGENERATION_METRIC_IDS)
    for m in DEGENERATION_MEASURES:
        out += [m, f"frac_{m}", f"{m}_rate", f"{m}_frac", f"{m}_share", f"degen_{m}_share"]
    out += list(LEGACY_DEGENERATION_METRIC_SPELLINGS)
    out += list(DEGENERATION_FRAC_COLUMNS)
    seen: Dict[str, None] = {}
    for v in out:
        seen.setdefault(v, None)
    return tuple(seen)

# --------------------------------------------------------------------------------------------
# Artifacts, rounds, rulings, dispositions
# --------------------------------------------------------------------------------------------
ARTIFACT_KINDS = (
    "store_root", "generations", "judge_items", "rows", "object_weights", "report", "record", "snapshot",
    "log", "completion", "figure", "page",
)
#: `analysis_input.input_kind`: what an analysis read -- an ingested run (its transcripts, its captures) or an
#: object (the intervention it studied). The row names the input twice: `input_id` (part of the key) and the one FK column of its kind.
ANALYSIS_INPUT_KINDS = ("run", "object")
#: `analysis_artifact.role`: what a registered file is TO the analysis -- its report (the numbers), a file of a published page
#: directory (the page and the figures it serves), or a figure filed on its own.
ANALYSIS_ARTIFACT_ROLES = ("report", "page", "figure")
RULING_ACTIONS = ("include", "exclude", "supersede", "coexist", "operating_point", "descope", "drop_eval", "reference", "hold_out")
SCOPE_KINDS = ("run", "object", "model", "eval", "key", "comparison")
EVIDENCE_STATES = ("ok", "missing", "path_absent", "incoherent", "stale", "not_evaluated")
EXPIRY_KINDS = ("served_id_absent", "grades_root_absent", "served_record_at_ruled_point", "input_sha_pinned")
#: Combinators of the structured `expires_when_json` form: {kind: any_of|all_of, conditions: [...], guards: [...]};
#: guards are checked (evidence only) and can never retire a ruling.
EXPIRY_COMBINATORS = ("any_of", "all_of")
RULED_UTC_PRECISIONS = ("datetime", "date")
PARITY_LEVELS = ("L0", "L1", "L1b", "L1c", "L2")
PARITY_RESULTS = ("pass", "fail", "refused")
DISPOSITIONS = ("pending", "held", "not_served", "not_applicable", "declined")
#: `v_run_current.selection_state` (the frame's selection; not a table CHECK). `member_pass` (migrations 0021 and 0022): a complete,
#: roster-matched run filed under a MEMBER pass label of its source (:data:`MEMBER_PASS_LABELS_BY_SOURCE`) whose own point -- (model,
#: object, eval, `view_legacy`) -- carries a POOLED reading of that source (:data:`POOLED_PASS_LABELS`, in any state): a current
#: measurement (`is_current` 1) that is not a paper row (`in_current_paper` 0); the paper reads the source's pooled pass instead.
#: `unpooled_member` (migration 0022): such a run whose point carries NO pooled reading of its source -- a current measurement whose
#: point's pooled reading the source has not filed; not a paper row (`in_current_paper` 0, as the paper side's pinned renderer
#: contract reads it); the measurable gap the adapter branch's pooled-emission repair closes (adapter plan 10).
#: `source_local_anchor` (migration 0027): an object-`none` run of a source in :data:`SOURCE_LOCAL_ANCHOR_SOURCES` -- its OWN
#: standup's unsteered reading, not the paper's -- that no ruling, no later run and no producer retirement retired: a current
#: measurement (`is_current` 1, an eligible anchor for its own source's contrasts, `anchor_current` e1) that is not a paper row
#: (`in_current_paper` 0); the campaign's `none` stays the paper's one unsteered reading. A decisive include still selects it.
SELECTION_STATES = ("selected", "excluded", "superseded", "withdrawn", "indeterminate", "not_in_roster", "member_pass", "unpooled_member",
                    "source_local_anchor")
#: Migration 0027: the sources whose object-`none` run is their OWN standup's unsteered
#: reading -- the leg every paired contrast of the source anchors on -- and never the paper's unsteered reading. Such a run reads
#: `selection_state` `source_local_anchor` (`in_current_paper` 0, `is_current` 1), rendered into both CASEs of `v_run_current` by
#: `schema_def.source_local_anchor_sql`, so the SQL and this list cannot drift. Exact spellings, or a GLOB pattern (`*`) for a family an
#: adapter mints from its directory names. Keyed on `run.lane_source_legacy`; every other source an adapter declares is classified in
#: :data:`CAMPAIGN_NONE_SOURCES` with its reason, and tests/test_registry_source_local_anchor.py refuses a source in neither list. The
#: members, each with what makes its none source-local:
#:   alphasteer_rebake, arditi_strength -- the two retired re-bakes (excluded wholesale by ruling; the decisive branch wins first);
#:     their nones are the re-bake's own standup legs, 5 and 1 per key beside the campaign's;
#:   arditi_allpos -- the all-positions re-fit; its none is its own standup's (the paper's multi-turn none is r2_export's);
#:   arditi_vendor_mt -- the vendor-decode multi-turn re-run; its none is its own standup's (the roster's multi-turn source is r2_export);
#:   strength_mtdiff_s1 -- the differenced multi-turn battery; its none is the battery's own s1 standup leg;
#:   strength_lane and strength_lane_* -- the dose ladders (`strength_ladder.source_for_dir` mints `strength_lane_<dir>`: base2it_native,
#:     base2it_native_mmlu, lora_base2it_native_gemma_hidose, any new ladder directory); each ladder's none is its own standup's 0-dose leg;
#:   em/agentic/qwen35_4b_allpos -- the EM all-positions panel's own none;
#:   em/agentic/*_bl, em/agentic/*_bl__*-superseded, em/agentic/qwen35_4b_bl_vendor_leak -- the EM published-baselines panels (every
#:     model), their `__<tag>-superseded` record copies and the Qwen vendor-leak baseline: the retired-baselines panel's second unsteered
#:     reading (the Qwen ones are excluded by ruling; the decisive branch wins anyway).
SOURCE_LOCAL_ANCHOR_SOURCES: Tuple[str, ...] = (
    "alphasteer_rebake", "arditi_strength", "arditi_allpos", "arditi_vendor_mt", "strength_mtdiff_s1",
    "strength_lane", "strength_lane_*",
    "em/agentic/qwen35_4b_allpos", "em/agentic/*_bl", "em/agentic/*_bl__*-superseded", "em/agentic/qwen35_4b_bl_vendor_leak",
)
#: The complement over the sources the adapters declare: a source whose object-`none` run IS the paper's unsteered reading (roster-matched
#: or admitted by ruling), is governed by rulings today, or whose adapter files no object-none run at all -- none of these moves under
#: the frame rule (one line each; the registry owner moves an entry to :data:`SOURCE_LOCAL_ANCHOR_SOURCES` by ruling, never by default):
#:   r2_export -- the campaign's export: its none is the paper's unsteered reading;
#:   sampled_lane -- the sampled-decode campaign: its pooled none is the paper's reading (the members read member_pass);
#:   sampled_lane_controls, ctl_neg1 -- the control-negative battery over the sampled tree and its cellstore: the battery's own readings,
#:     replicates under the later-run key, governed by rulings, no paper figure's unsteered baseline;
#:   bfcl_full, bigcodebench_full, mmlu_full -- the canonical capability rows: their none is the paper's reading (roster-matched, ruled in);
#:   bfcl_full_shared, bfcl_full_shared_l2, bfcl_full_shared_l3, bigcodebench_full_shared, bigcodebench_full_shared_l2,
#:     bigcodebench_full_shared_l3 -- the shared-standup comparison sources: every row is out of the frame by ruling;
#:   bfcl_allmodels, bigcodebench_allmodels, bfcl_seedrep, bigcodebench_seedrep, gsm8k_seedrep, ifeval_seedrep, math500_seedrep,
#:     seedrep_battery, seedrep_cluster, seedrep_petri, orxs_seedrep -- the seed-replicate and all-models batteries: their nones are the
#:     replicate panels' own paper readings, roster-matched or ruled by name;
#:   bcb_bfcl_200, bcb_bfcl_200_widen, evalcheck/gsm8k_paired200, evalcheck/math500 -- the 200-item spot checks and eval checks: ruled
#:     (superseded) by name, the full readers replaced them;
#:   gsm8k_paired_allmodels, misalign_paired_allmodels -- the paired all-models batteries (no object-none run filed live);
#:   llama8b_lane -- owns the Llama models' own-wave readings: its none IS the paper's Llama baseline;
#:   relu_paired, relu_native -- the ReLU paired battery (pooled passes; members read member_pass) and native-engine readings: the paper's
#:     ReLU-engine baselines;
#:   multiturn_grades -- the registry re-grade of the multi-turn stores: the paper's multi-turn baseline (the default owner);
#:   overrefusal_items, static_padding -- item-level readers of the campaign's rows: their nones are the campaign's own runs re-read;
#:   bon_stores, bon_native4096:llama8b_relu_paired -- best-of-N stores and native readings onto the campaign's and the ReLU battery's runs;
#:   petri_stats, degeneracy_union, legacy_glm -- metric readers onto other sources' runs (no none run of their own) and the legacy GLM rows;
#:   em/agentic/gemma4e4b, em/agentic/qwen35_4b, em/agentic/nemotron3nano4b, em/agentic/glm46vflash, em/agentic/llama8b_it,
#:     em/agentic/llama8b_main, em/agentic/qwen35_4b_it, em/agentic/gemma4e4b_it, em/agentic/qwen35_4b_vendor_leak,
#:     em/agentic/qwen35_4b_it_vendor_leak -- the EM agentic panels' main readings: the paper's EM baselines (included by ruling; a
#:     second ruling excludes two);
#:   em/em_organisms, em/organisms/qwen35_4b_org, em/organisms/qwen35_4b_org_vendor -- the EM organism panels (a ruling excludes the Qwen
#:     organisms);
#:   misalign_seedrep, misalign_16k, misalign_ladder -- the misalignment seed-replicate, 16k and ladder panels: those panels' own paper
#:     readings (included by ruling).
CAMPAIGN_NONE_SOURCES: Tuple[str, ...] = (
    "r2_export", "sampled_lane", "sampled_lane_controls", "ctl_neg1",
    "bfcl_full", "bigcodebench_full", "mmlu_full",
    "bfcl_full_shared", "bfcl_full_shared_l2", "bfcl_full_shared_l3", "bigcodebench_full_shared", "bigcodebench_full_shared_l2", "bigcodebench_full_shared_l3",
    "bfcl_allmodels", "bigcodebench_allmodels", "bfcl_seedrep", "bigcodebench_seedrep", "gsm8k_seedrep", "ifeval_seedrep", "math500_seedrep",
    "seedrep_battery", "seedrep_cluster", "seedrep_petri", "orxs_seedrep",
    "bcb_bfcl_200", "bcb_bfcl_200_widen", "evalcheck/gsm8k_paired200", "evalcheck/math500",
    "gsm8k_paired_allmodels", "misalign_paired_allmodels",
    "llama8b_lane", "relu_paired", "relu_native", "multiturn_grades", "overrefusal_items", "static_padding",
    "bon_stores", "bon_native4096:llama8b_relu_paired", "petri_stats", "degeneracy_union", "legacy_glm",
    "em/agentic/gemma4e4b", "em/agentic/qwen35_4b", "em/agentic/nemotron3nano4b", "em/agentic/glm46vflash", "em/agentic/llama8b_it",
    "em/agentic/llama8b_main", "em/agentic/qwen35_4b_it", "em/agentic/gemma4e4b_it", "em/agentic/qwen35_4b_vendor_leak", "em/agentic/qwen35_4b_it_vendor_leak",
    "em/em_organisms", "em/organisms/qwen35_4b_org", "em/organisms/qwen35_4b_org_vendor",
    "misalign_seedrep", "misalign_16k", "misalign_ladder",
)
assert not set(SOURCE_LOCAL_ANCHOR_SOURCES) & set(CAMPAIGN_NONE_SOURCES)
#: Every `run.pass_label` the producers file (live census 2026-10-02; the column itself is free text, so this is a vocabulary of the
#: frame, not a CHECK): `base_cap` and `ladder_rung` the per-trial measurements of a token-cap ladder at its base cap and at a higher
#: cap, `unrequested_rung` a measurement at a cap the ladder did not ask for, `pooled_t0_t1` / `pooled_final_rung` a reading pooled
#: over a source's trials (at the paper cap / at each pair's final cap), `single_pass` a one-pass source's label.
PASS_LABELS = ("base_cap", "ladder_rung", "pooled_t0_t1", "pooled_final_rung", "single_pass", "unrequested_rung")
#: The POOLED pass labels (migration 0022): a reading pooled over a source's trials, at the paper cap (`pooled_t0_t1`) or at each
#: pair's final cap (`pooled_final_rung`). A member pass (:data:`MEMBER_PASS_LABELS_BY_SOURCE`) is a member (`member_pass`) only where
#: ITS source filed a run under one of these labels at the member's own POINT -- (model, object, eval, `view_legacy`), the unit the
#: sources pool by -- in any lifecycle or selection state: a pooled run that a later pooled run superseded, or that a ruling excluded,
#: or that was withdrawn, still marks its members, because the members are never the paper reading once the source filed a pooled one.
#: Rendered into `v_run_current` by `schema_def.POOLED_GROUP_CTE` (one pass over `run`), so the SQL and this tuple cannot drift.
POOLED_PASS_LABELS = ("pooled_t0_t1", "pooled_final_rung")
assert set(POOLED_PASS_LABELS) <= set(PASS_LABELS) and len(set(POOLED_PASS_LABELS)) == len(POOLED_PASS_LABELS)
#: Migration 0021: a source whose PAPER reading is a pooled pass files its per-trial / per-cap measurements under these MEMBER
#: pass labels. Such a run is a current measurement (`v_run_current.is_current` is unaffected; the later-run supersession never
#: retires it) but NOT a paper row: `selection_state` reads `member_pass` and `in_current_paper` 0, so the paper counts the
#: pooled reading once and never its members beside it. Since migration 0022 the label alone does not make a member: the run
#: reads `member_pass` only where a pooled run of the SAME source (:data:`POOLED_PASS_LABELS`) exists at its own point -- (model,
#: object, eval, `view_legacy`) -- and `unpooled_member` where its point carries no pooled reading of its source; neither is a paper
#: row (`in_current_paper` 0: the paper's reading of a listed source's point is its pooled run, the pinned renderer contract), and an
#: include ruling still selects either. Keyed on
#: `run.lane_source_legacy`; the selection SQL is rendered from this table (`schema_def.member_pass_sql`), so the SQL and the table
#: cannot drift, and the rule lives in `v_run_current` alone: the registry's renderer (`paper.paper_long_frame`) honors
#: `in_current_paper`, while `paper.paper_frame_exclusion`'s per-source keep-list (`sampled_lane`: `pooled_t0_t1` alone) stays as the
#: paper side's pinned contract, subsumed by this rule for the registry's own rendering (the paper's pinned copy reads `v_paper_long`
#: with no `in_current_paper` filter and is the paper side's to cut over). A source absent here keeps every pass in the paper
#: (`ctl_neg1`, `seedrep_battery`, `seedrep_cluster` file `base_cap` / `ladder_rung` with no pooled pass: those ARE their readings,
#: replicates coexisting under the run identity key). The pooled run's `extra.member_run_ids` is the authoritative membership relation; this
#: table and the pooled-point test are the rendered rule that mirrors it today, and a data-driven rule (a run named by a current pooled
#: run's `member_run_ids` is a member) is the candidate for a later migration.
MEMBER_PASS_LABELS_BY_SOURCE: Dict[str, Tuple[str, ...]] = {
    "sampled_lane": ("base_cap", "ladder_rung"),
    "relu_paired": ("base_cap", "ladder_rung", "unrequested_rung"),
}
assert all(set(ls) <= set(PASS_LABELS) and len(set(ls)) == len(ls) for ls in MEMBER_PASS_LABELS_BY_SOURCE.values())
assert not any((set(POOLED_PASS_LABELS) | {"single_pass"}) & set(ls) for ls in MEMBER_PASS_LABELS_BY_SOURCE.values())
#: Migration 0026: two complete runs of ONE source under a POOLED pass label at one run identity key whose
#: `view_legacy` DIFFER are readings of two POINTS (the tags pool separately: the pin tag's and a dose tag's `none` at one model, eval and
#: instrument), never predecessor and successor -- the later-run supersession of `v_run_current` skips such a pair, so a different-tag
#: pooled pair coexists by identity (not by the equal timestamps it rests on today), a later SAME-tag pooled run still retires its
#: predecessor, and every other source keeps the later-run rule. `view_legacy` does NOT enter the key itself: measured read-only, that
#: would release ~1,600 earlier runs across 30+ source pairs, the cross-source twins the rule dedupes today included. Keyed on
#: `run.lane_source_legacy` -> the pooled pass labels the exception covers; rendered into the `later` CTE by
#: `schema_def.coexist_across_views_sql`, so the rule and the SQL cannot drift. Scope: every object of the pass (the pin's steered pooled
#: rows of two transfer standups share a key exactly as the nones do); it may be narrowed to `object_id = 'none'`.
COEXIST_ACROSS_VIEWS_BY_SOURCE: Dict[str, Tuple[str, ...]] = {
    "sampled_lane": ("pooled_t0_t1",),
}
assert all(set(ls) <= set(POOLED_PASS_LABELS) and len(set(ls)) == len(ls) for ls in COEXIST_ACROSS_VIEWS_BY_SOURCE.values())

# --------------------------------------------------------------------------------------------
# Ledgers, sources, mirrors
# --------------------------------------------------------------------------------------------
DELIVERY_STATUSES = ("ok", "refused", "duplicate", "blocked", "partial", "conflict")
#: The two sides of a file's delivery audit a record is counted on (`ordered_fact_carrier.delivery`; the ninth closing pass of the
#: analysis filing, 2026-10-10): `accepted` is the file's `n_ok` or `n_duplicate` -- which of the two is the fold's arrival order and is
#: not recorded -- and `refused` its `n_refused`.
DELIVERY_SIDES = ("accepted", "refused")
WITHDRAWAL_REASONS = ("removed_measurement", "removed_items", "input_vanished")
MIRROR_DESTINATIONS = ("cluster_a", "cluster_b", "modal")
#: The kinds of fact the harvester OBSERVES about a record at the fold without refusing it (`fold_observation`; registry decisions on the
#: serving-jobs restatement, 2026-10-04): a detector over the fold's own output, never a fold rule. `serving_jobs_strict_subset`: a
#: `run` / `run_patch` record whose `extra.serving_jobs` (a list, compared as a set) names a STRICT subset of the set the run's folded
#: `extra_json.serving_jobs` already held -- the fold replaced the key as it always does (replace-at-key is universal), and the dropped
#: members are recorded so the producer whose cumulative set is the only belt (job-start registration on a same-identity rerun) learns it slipped.
FOLD_OBSERVATION_KINDS = ("serving_jobs_strict_subset",)
RNG_SCOPES = ("none", "group", "reader_traversal")

#: Every record type the contract admits (``docs/registry/records.md``): the spec's 13, the
#: dimension/paper types §13 C1 added, ``run_patch`` (S7), the item-scale ``item_flags`` (F10), and
#: the harvester-originated types (F3) that make every harvester fact a log record first.
RECORD_TYPES = (
    # producer-side facts
    "run", "run_patch", "run_state", "run_metric", "run_contrast", "item_result", "item_flags",
    "run_degeneration", "judgment", "object", "artifact", "model", "eval", "item_set", "instrument", "judge",
    "si_text", "fit_search", "fit_trial", "paper_round", "roster_entry", "ruling", "decode_policy",
    "sampling_mode_policy", "computation", "disposition", "snapshot", "parity_gate",
    # a producer retracts item_result rows it filed under the wrong run (migration 0020): one record, a list of exact item keys
    "item_withdrawal",
    # harvester-originated facts (records BEFORE rows, so a rebuild replays them)
    "run_ingest", "ruling_state", "source_watermark", "source_state", "source_revision", "observation",
    "withdrawal", "mirror_ack", "file_arrival",
    # the adapter runner's status of one read of one source, per paper source string (item-scale, revision-tagged)
    "source_status",
    # the harvester's durable fact of ONE refusal judgment: written into the log as a record and folded into the ledger-scale table
    # `record_refusal_fact` by its own apply path (`INSERT OR IGNORE`, no `row_history`), so a rebuild reproduces every refusal ever made
    # under every judging stamp. It is NOT in the two ledgers' record_type CHECKs (``LEDGERED_RECORD_TYPES``): a kind that never reaches
    # `blocked_records` or `row_history` must not rebuild them (the 0020 rebuild of `row_history` wrote a 25 GB WAL)
    "record_refusal",
    # an ANALYSIS filed over ingested runs and objects: a prefill-only or offline computation with results but no
    # run-shaped home (no eval, no generation). Three ledger-fact kinds, one row of one table each (`analysis`, `analysis_input`,
    # `analysis_artifact`), folded by the same apply path as `record_refusal` (INSERT OR IGNORE, no `row_history`): the row IS the fact.
    # Written by `registry analysis file --spec` under records/analyses/<day>/, never by the harvester; the FKs of the two child tables
    # (the analysis, a run, an object, an artifact) are dependency-checked by the harvester before the apply
    "analysis", "analysis_input", "analysis_artifact",
)

#: The three record kinds of a filed analysis, in the order a filing writes them after its artifact records: the parent row,
#: then its inputs, then its artifact links -- so a filing's own files fold parent-first by stamp and by path alike.
ANALYSIS_RECORD_TYPES = ("analysis", "analysis_input", "analysis_artifact")

#: Record kinds that fold into a ledger-scale table by a dedicated apply path and never enter `blocked_records` or `row_history`
#: the two ledgers' ``record_type`` CHECKs render from :data:`LEDGERED_RECORD_TYPES` -- :data:`RECORD_TYPES` without these --
#: so adding one here leaves both CHECK lists byte-identical and the ledgers unrebuilt (``tests/test_registry_refusal_facts.py`` pins it).
#: A kind here can never WAIT in `blocked_records` either (its CHECK is the ledgered list), so the harvester refuses such a record by
#: name when a parent it names is absent (``harvest.Ingestor.apply_validated``); `registry analysis file` read-checks every input
#: before it writes, so the log carries no such record unless hand-written.
UNLEDGERED_FACT_TYPES = ("record_refusal",) + ANALYSIS_RECORD_TYPES
LEDGERED_RECORD_TYPES = tuple(t for t in RECORD_TYPES if t not in UNLEDGERED_FACT_TYPES)
assert len(LEDGERED_RECORD_TYPES) == len(RECORD_TYPES) - len(UNLEDGERED_FACT_TYPES)

#: Item-scale tables: whole-row fold, two stamp columns, NO row_history, shipped as Parquet only.
ITEM_SCALE_TABLES = ("item_result", "item_flags", "run_degeneration", "source_status")
ITEM_SCALE_RECORD_TYPES = ("item_result", "item_flags", "run_degeneration", "source_status")

# --------------------------------------------------------------------------------------------
# The registry of vocabularies by name (for the DDL renderer, the validator, the glossary)
# --------------------------------------------------------------------------------------------
VOCABULARIES: Dict[str, Tuple] = {
    "BOOLEAN": BOOLEAN,
    "LIFECYCLE_STATES": LIFECYCLE_STATES,
    "EVIDENCE_KINDS": EVIDENCE_KINDS,
    "WALLTIME_SOURCES": WALLTIME_SOURCES,
    "TRANSFER_KINDS": TRANSFER_KINDS,
    "STRENGTH_NOMINAL_SOURCES": STRENGTH_NOMINAL_SOURCES,
    "STRENGTH_UNITS": STRENGTH_UNITS,
    "INSTRUMENT_MATCH": INSTRUMENT_MATCH,
    "DECODE_POLICY_MATCH": DECODE_POLICY_MATCH,
    "MAX_NUM_SEQS_SOURCES": MAX_NUM_SEQS_SOURCES,
    "PREFIX_CACHE_POLICIES": PREFIX_CACHE_POLICIES,
    "SUBSTRATES": SUBSTRATES,
    "FIT_SEARCH_SUBSTRATES": FIT_SEARCH_SUBSTRATES,
    "CONFIDENCE": CONFIDENCE,
    "SERVING_PATHS": SERVING_PATHS,
    "DECODE_SOURCES": DECODE_SOURCES,
    "DECODE_SEED_RULES": DECODE_SEED_RULES,
    "THINKING_MODES": THINKING_MODES,
    "APPLIED_POSITIONS_MODES": APPLIED_POSITIONS_MODES,
    "MULTI_TURN_SCOPES": MULTI_TURN_SCOPES,
    "MODEL_FAMILIES": MODEL_FAMILIES,
    "CHECKPOINT_KINDS": CHECKPOINT_KINDS,
    "THINKING_CHANNELS": THINKING_CHANNELS,
    "MODEL_PAPER_STATUSES": MODEL_PAPER_STATUSES,
    "OBJECT_KINDS": OBJECT_KINDS,
    "PAPER_LABELS": PAPER_LABELS,
    "OBJECT_ROLES": OBJECT_ROLES,
    "FIT_TARGET_KINDS": FIT_TARGET_KINDS,
    "FIT_OBJECTIVES": FIT_OBJECTIVES,
    "FIT_THINKING_RENDERS": FIT_THINKING_RENDERS,
    "FIT_POSITIONS": FIT_POSITIONS,
    "FIT_CAPTURE_SITES": FIT_CAPTURE_SITES,
    "LAYER_INDEX_CONVENTIONS": LAYER_INDEX_CONVENTIONS,
    "LAYER_SELECTION_RULES": LAYER_SELECTION_RULES,
    "FIT_STOPPED_ON": FIT_STOPPED_ON,
    "FIT_SEED_CONFIDENCE": FIT_SEED_CONFIDENCE,
    "DERIVATIONS": DERIVATIONS,
    "OBJECT_STATUSES": OBJECT_STATUSES,
    "OBJECT_NAMED_ROLES": OBJECT_NAMED_ROLES,
    "SI_KINDS": SI_KINDS,
    "SEARCH_METHODS": SEARCH_METHODS,
    "OBJECTIVE_DIRECTIONS": OBJECTIVE_DIRECTIONS,
    "TRIAL_STATUSES": TRIAL_STATUSES,
    "EVAL_CATEGORIES": EVAL_CATEGORIES,
    "METRIC_DIRECTIONS": METRIC_DIRECTIONS,
    "EVAL_PAPER_STATUSES": EVAL_PAPER_STATUSES,
    "HARNESSES": HARNESSES,
    "COT_PROMPTING": COT_PROMPTING,
    "SCORING_RULES": SCORING_RULES,
    "ANSWER_EXTRACTIONS": ANSWER_EXTRACTIONS,
    "READ_AT": READ_AT,
    "JUDGE_ROLES": JUDGE_ROLES,
    "JUDGE_BACKENDS": JUDGE_BACKENDS,
    "JUDGMENT_STATUSES": JUDGMENT_STATUSES,
    "API_POOLS": API_POOLS,
    "SAMPLING_MODES": SAMPLING_MODES,
    "COMPUTED_BY": COMPUTED_BY,
    "ANCHOR_KINDS": ANCHOR_KINDS,
    "EFFECT_KINDS": EFFECT_KINDS,
    "DEGENERATION_MEASURES": DEGENERATION_MEASURES,
    "DEGENERATION_SCOPES": DEGENERATION_SCOPES,
    "DENOMINATOR_UNITS": DENOMINATOR_UNITS,
    "AGGREGATIONS": AGGREGATIONS,
    "MEASUREMENT_STATES": MEASUREMENT_STATES,
    "COVERAGE_STATES": COVERAGE_STATES,
    "ARTIFACT_KINDS": ARTIFACT_KINDS,
    "ANALYSIS_INPUT_KINDS": ANALYSIS_INPUT_KINDS,
    "ANALYSIS_ARTIFACT_ROLES": ANALYSIS_ARTIFACT_ROLES,
    "RULING_ACTIONS": RULING_ACTIONS,
    "SCOPE_KINDS": SCOPE_KINDS,
    "EVIDENCE_STATES": EVIDENCE_STATES,
    "DISPOSITIONS": DISPOSITIONS,
    "RULED_UTC_PRECISIONS": RULED_UTC_PRECISIONS,
    "PARITY_LEVELS": PARITY_LEVELS,
    "PARITY_RESULTS": PARITY_RESULTS,
    "DELIVERY_STATUSES": DELIVERY_STATUSES,
    "DELIVERY_SIDES": DELIVERY_SIDES,
    "WITHDRAWAL_REASONS": WITHDRAWAL_REASONS,
    "MIRROR_DESTINATIONS": MIRROR_DESTINATIONS,
    "FOLD_OBSERVATION_KINDS": FOLD_OBSERVATION_KINDS,
    "RECORD_TYPES": RECORD_TYPES,
    "LEDGERED_RECORD_TYPES": LEDGERED_RECORD_TYPES,
    "ITEM_SCALE_TABLES": ITEM_SCALE_TABLES,
}

# --------------------------------------------------------------------------------------------
# Glossary evidence: every controlled value with its producer-side evidence (the glossary
# is generated from this). Keep entries short; cite the producer symbol.
# --------------------------------------------------------------------------------------------
EVIDENCE: Dict[str, Dict[object, str]] = {
    "MODEL_FAMILIES": {
        'gemma4': 'google/gemma-4-* checkpoints.',
        'qwen35': 'Qwen/Qwen3.5-* checkpoints.',
        'qwen36': 'the multi-turn attacker / Petri auditor-of-last-resort (huihui-ai/Huihui-Qwen3.6-27B-abliterated).',
        'qwen38': 'the HPO model (Qwen/Qwen3.8-27B).',
        'nemotron3': 'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16.',
        'llama31': 'Llama-3.1 checkpoints (the NousResearch mirrors).',
        'llama32': 'unsloth/Llama-3.2-3B-Instruct (an earlier fit model).',
        'glm46v': 'zai-org/GLM-4.6V-Flash (descoped).',
    },
    "CHECKPOINT_KINDS": {
        'base': 'a pretrained checkpoint with no chat post-training (the fit checkpoints of the base-to-instruct transfer; never served).',
        'instruct': 'an instruction-tuned checkpoint (the served checkpoints of the four families: the -it / -Instruct repos, and Qwen3.5\'s post-trained checkpoint whose repo name carries no suffix).',
        'chat': 'a chat checkpoint with no base / instruct pair (nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16; zai-org/GLM-4.6V-Flash, descoped).',
        'organism': 'an emergent-misalignment model organism: a LoRA fine-tune of a served instruct checkpoint on a Betley et al. emergent-misalignment dataset (`scripts/sbatch/em_organism_train.sbatch`), served on its parent as a vLLM LoRA module (`misalign_proxy.sbatch` organisms mode) with the intervention objects overlaid; `extra_json.parent_model_id` names the parent (the schema has no parent column). The organism is the model and `em8q` the eval (organisms are checkpoints, not model + object); `hf_revision` / the adapter sha are `unverified` until the producer sends them.',
    },
    "DECODE_POLICY_MATCH": {
        'exact': 'exactly one decode_policy row admits the run (model, eval or eval category, thinking mode) and every parameter it pins (temperature, top_p, top_k, min_p, repetition_penalty, presence_penalty, seed_rule) the run sent at the same value -- or lists under params_not_sent_json and the run did not send -- and the run sent nothing the row leaves null -- or, when several rows in force match and exactly one is eval-scoped, that row over the category-scoped ones (the tie-break of the stamp pass; `extra_json.decode_policy_tiebreak` names the losers); `run.decode_policy_id` names it. Derived by decode_match.stamp_run_rows from v_run_decode_policy_match at the end of every ingest cycle and of a rebuild. Whether the matched row is the FAMILY DEFAULT (the row resolve() returns today for the run\'s (model, eval-or-category, thinking) key) is a separate fact the paper views render beside it -- `decode_family_default_policy_id` / `decode_is_family_default` in v_paper_current and v_paper_long: `exact` with `decode_is_family_default` 0 reads \'matched a recorded, non-default policy\' (every agentic run at the runner\'s T = 1.0, the vendor preset being every family\'s default), never a mismatch; the match never reads the default flag.',
        'ambiguous': 'several rows match exactly, the effective_from_utc / superseded_by_policy_id windows do not single one out at generated_utc, and neither does the tie-break of the stamp pass (one eval-scoped row over category-scoped rows): a true duplicate policy; the candidates are listed in v_run_decode_policy_match.candidates_json; decode_policy_id stays null.',
        'no_match': 'the run recorded sent parameters (decode_sent_json / decode_*_sent) and no row matches them exactly; decode_policy_id is null.',
        'unrecorded': 'the run recorded no sent parameter at all (empty decode_sent_json, null decode_*_sent): nothing to verify against.',
        'no_policy': 'the producer-written placeholder every run record carries before the derived pass has stamped the row.',
    },
    "THINKING_CHANNELS": {
        'available': 'the chat template carries a thinking switch (ModelRow thinking_modes has `on`).',
        'not_available': 'no switch (the Llama models; base checkpoints).',
        'unverified': 'no ModelRow / manifest read establishes whether the checkpoint has a switch (reference models only); reason in `field_confidence_json`.',
    },
    "SI_KINDS": {
        'safety_instruction': 'instruction A1 `safety_refusal`.',
        'constitution_short': '`si_constitution` (the compressed excerpt).',
        'constitution_full': '`si_constitution_full` (the vendored constitution.md).',
        'irrelevant_control': '`si_irrelevant` (the metre history).',
        'reference_other': 'a text registered in `sisteer.data.instructions` that no in-scope object distils and no prompted object declares (si_caa); kept so a prefix hash met in a manifest can be NAMED.',
    },
    "SCORING_RULES": {
        'ifeval_checker': "IFEval's rule checker (prompt_level_strict headline; loose / instruction-level alongside).",
        'judge_score_mean': 'Petri: mean of the 1-10 judge score over (behaviour, scenario) transcripts.',
    },
    "ANSWER_EXTRACTIONS": {
        'unverified': 'a superseded instrument whose extraction rule was never read; reason in `field_confidence_json`.',
    },
    "READ_AT": {
        'ladder_final': 'the metric is read at the final level of the token-cap ladder.',
        'fixed': 'one cap, no ladder.',
        'unverified': 'a superseded instrument whose read point was never recorded; reason in `field_confidence_json`.',
    },
    "DECODE_SEED_RULES": {
        'per_item': 'one seed per item (sha256 of seed_base|trial|readout|item); a seed that closes over more than the item (a turn, a sample) takes the label naming that closure, never `per_item` plus free text.',
        'per_draw': 'one seed per best-of-N draw (`bon_seed`).',
        'per_goal_turn': 'one seed per (seed_base, goal, turn), seed_base the literal training seed s: the multi-turn target\'s per-turn rule under the seed-replicate battery (crescendo / pair / actorattack; the attackers keep their own rows).',
        'per_scenario_turn': 'one seed per (seed_base, behaviour, scenario_id, turn), seed_base the literal training seed s: the Petri target turns under the seed-replicate battery (the target `GenerateConfig` carries the vendor row + `seed` per scenario-turn; the auditor and the judge are read-outs on their own rows).',
        'per_response': 'one seed per (seed_base, family / behaviour, item, sample), seed_base the literal training seed s: the misalignment runner\'s per-response rule under the seed-replicate battery (`--sampler vendor` + a per-response seed recorded in `meta.json`).',
        'fixed': 'one fixed seed for the run.',
        'none': 'no seed sent.',
        'unverified': 'a superseded instrument whose seeding was never read; reason in `field_confidence_json`.',
    },
    "JUDGE_BACKENDS": {
        'batch': 'Anthropic Batch API (`scripts/judge_batch.py`).',
        'live': 'the live client (`grading_live.py`, `judging.ClaudeJudge`).',
        'batch_or_live': 'one judge identity driven through both the Batch API and the live client (StrongREJECT v1 on Haiku 4.5); the per-judgment pool is `judgment.api_pool`.',
        'unverified': 'the backend is not recorded.',
    },
    "LIFECYCLE_STATES": {
        "queued": "submit wrapper wrote the planned `run` + `run_state queued` (`sbatch_registered.sh`, Modal launcher).",
        "running": "job start: `run_state running` from the job log / `stage2_common.sh`, with node and job id.",
        "generated": "store writer wrote the completion artifact (`CellStore` results, `bon_served_manifest.json`).",
        "judging": "a judge registered an in-progress `judgment` (the claim; `registry pending-judgment --claim`).",
        "judged": "a `judgment` completed and the judging driver wrote `run_state judged`.",
        "ingested": "harvester `run_ingest` record: completion requirements held (complete judgment with coverage, artifact present).",
        "failed": "scheduler state FAILED/TIMEOUT or a job-log failure line, stored as evidence beside the state.",
        "cancelled": "scheduler state CANCELLED (a STOP sentinel or scancel) with a partial store.",
        "superseded": "a `supersede` ruling covers the run; `superseded_by_run_id` names the successor.",
        "withdrawn": "a source revision no longer carries the run (`withdrawal` record) or a withdrawal ruling.",
    },
    "OBJECT_KINDS": {
        "none": "unsteered execution; `object_id = 'none'`.",
        "system_instruction": "prompted object; `si:<sha256 of the instruction text>`.",
        "lora": "LoRA adapter (`fit_class_b.py`, `train_budget.py`); rank in `fit_rank`.",
        "lora_subtraction": "difference adapter A − B (`derived_meta.json`, derivation `lora_subtraction`).",
        "relu_steer": "ReLU-gated steering object (legacy spelling `psr`, never used in new text).",
        "relu_steer_distilled_difference": "ReLU-steer object distilled from a difference (`residual_distillation`).",
        "vector": "additive residual vector at listed layers.",
        "vector_all": "additive residual vector at every layer.",
        "diffmean": "difference-of-means direction (`extraction/*`, capture at the appended assistant header).",
        "static_r_arditi": "Arditi refusal direction: post-instruction span × layer search.",
        "caa_static_r": "CAA direction served as a static residual.",
        "alphasteer": "AlphaSteer projection object (`fit_meta.json`, `alphasteer_fit_meta/1`).",
        "antidote": "Antidote last-position direction.",
        "composed": "composition of parents (`derivation = composition`).",
        "stacked": "stack of parents (`derivation = stack`).",
    },
    "PAPER_LABELS": {
        "none": "unsteered baseline.",
        "safety_si": "prompted safety system instruction.",
        "constitution_si": "prompted short constitution.",
        "constitution_si_full": "prompted full constitution.",
        "control_lora": "control-corpus LoRA (`constv2ctl2-full`).",
        "constitution_lora": "constitution-corpus LoRA (`constv2-mix-full`).",
        "difference_lora": "constitution − control LoRA.",
        "control_relu": "control-corpus ReLU-steer object.",
        "constitution_relu": "constitution-corpus ReLU-steer object.",
        "difference_relu": "constitution − control ReLU-steer object: the undistilled paired write u_pos − u_neg (object_kind `composed`).",
        "difference_relu_distilled": "the residual-distilled student of the constitution − control ReLU-steer difference (object_kind "
                                     "`relu_steer_distilled_difference`); registry-visible, outside the paper's method names (the paper's "
                                     "difference_relu entry reads blank on a family until a paired write exists for it).",
        "arditi": "Arditi refusal direction.",
        "arditi_allpos": "Arditi refusal direction added at every position of prefill and decode at raw coefficient c (the paper's configuration); a different intervention from `arditi`, whose alpha-dosed ablation is prefill-only.",
        "alphasteer_rebake": "AlphaSteer re-baked on our checkpoint.",
        "alphasteer_as_fit": "AlphaSteer as fit (superseded ladder).",
        "alphasteer_refit_reference": "the authors' llama refusal vector re-projected onto our checkpoint (a separate object).",
        "safety_lora": "safety-instruction distilled LoRA.",
        "safety_relu": "safety-instruction distilled ReLU-steer object.",
        "other": "any object outside the paper's method names.",
    },
    "FIT_POSITIONS": {
        "all": "every token (`train_budget_meta.json:positions = all`).",
        "response_incl_assistant_header": "response tokens including the assistant header (`response` + `response_header_len`).",
        "last_token_of_appended_assistant_header": "diff-mean / CAA / AlphaSteer capture: the generation header is included.",
        "searched_post_instruction_span": "Arditi: every post-instruction position × layer searched; selected offset and token text recorded.",
        "completion_mean": "mean over completion tokens.",
        "inherited_from_parents": "derived objects; read through `v_object_provenance`'s parent join.",
        "not_applicable": "`none` and system-instruction objects.",
        "unverified": "no sidecar or blob key names the positions; reason in `field_confidence_json`.",
    },
    "APPLIED_POSITIONS_MODES": {
        "all": "every token of every prefill and decode; every natively served LoRA (`native_residual.POSITION_MODES`).",
        "response": "the final assistant header plus that turn's decode; earlier assistant turns re-encoded unsteered.",
        "response_all_turns": "lens only (`lens_adapters._SERVED_POSITION_MODES`).",
        "prompt_last": "prefill only, last prompt token.",
        "prefill_all": "every prompt token at prefill, nothing at decode.",
        "prefill_last_broadcast": "AlphaSteer: last-prompt-token vector added to every prompt position (`prompt_hold` variant in `extra_json`).",
        "last": "antidote.",
        "non_template": "every token of every prefill and decode EXCEPT the chat-template scaffold (`<|im_start|>` / the role name / its "
                        "newline, `<|im_end|>` / its newline, the empty think header): the position class the routing analysis found "
                        "rerouting concentrated on; the token rule is the serving config's (`native_residual.POSITION_MODES`), applied to "
                        "the realized ids at decode as at prefill.",
        "none": "passthrough / unsteered.",
        "unverified": "the serving record names no mode; reason in `field_confidence_json`.",
    },
    "MULTI_TURN_SCOPES": {
        "every_turn_prefill_and_decode": "modes `all` / `non_template` on a multi-turn eval.",
        "final_header_and_current_decode_only": "mode `response` on a multi-turn eval.",
        "every_assistant_span": "mode `response_all_turns` (lens).",
        "prefill_only_each_turn": "modes `prompt_last` / `prefill_all` / `prefill_last_broadcast`.",
        "single_turn": "single-turn eval.",
        "not_applicable": "`none` object; a system-instruction object on a multi-turn eval (a prompted object steers no positions, "
                          "whatever mode the row carries).",
        "unverified": "a legacy store that records neither the eval's turn structure nor the steering mode (adapters "
                      "file the row with a reason in `field_confidence_json` instead of refusing it).",
    },
    "SERVING_PATHS": {
        "lora_native": "vLLM multi-LoRA engine (`serving/native_residual.py`, launch attestation).",
        "lens": "lens hook serving (`serving/lens_adapters.py`).",
        "passthrough": "no intervention on the served path.",
        "hf": "transformers forward hooks.",
        "unverified": "no attestation names the path.",
    },
    "DECODE_SOURCES": {
        "generation_config": "the checkpoint's shipped `generation_config.json` at the pinned revision (the default source).",
        "vendor_recommendation": "vendor documentation numbers (the Qwen instruct row as the vLLM server default).",
        "model_card": "numbers stated in the HF model card (Qwen thinking-card rows).",
        "vllm_default": "nothing sent; the engine's defaults.",
        "greedy": "temperature 0.",
        "grandfathered_T1": "best-of-N's T=1, kept for every store drawn before the per-model `vendor_preset` rows became the default for new draws; readable by explicit reference.",
        "runner_forced_T1": "agentic-misalignment runner forces T=1.",
        "legacy_contracts_row": "`contracts.SAMPLING_BY_MODEL_KEY` row (0.7/0.95/20) matching no vendor row.",
        "unverified": "no request record names the source.",
    },
    "STRENGTH_UNITS": {
        "lora_write_basis_multiplier": "LoRA dose ladders: the multiplier on the shipped object's write basis (battery_curves.json `dose_unit`, `dose_semantics`).",
        "psr_direction_norm_multiplier": "ReLU-steer (PSR) dose ladders: the multiplier on the direction norm (gauge freedom keeps the gate byte-identical).",
        "alpha": "Arditi refusal-direction ablation: the record's alpha (battery_curves.json under arditi_strength).",
        "lambda": "AlphaSteer: the record's lambda (the ladder record beside each object; alphasteer_operating_points.json).",
        "raw_coefficient": "the all-positions Arditi object: the raw coefficient c on the difference-of-means vector added at every position (battery_curves.parameters.dose_unit; the `arditi_allpos` ladder). Never an alpha: alpha_equiv is a conversion for talking to the prefill-only ladder, not this object's dose.",
        "nominal_only": "a campaign nominal dose (1.0 = the shipped object as served) whose record declares no unit.",
        "unverified": "the record declares no unit (the column default); also the value carried by unsteered and prompted rows, which have no strength.",
    },
    "TRANSFER_KINDS": {
        "base_to_instruct": "object fit on the base checkpoint, served on the instruct one.",
        "instruct_direct": "fit and served on the same instruct checkpoint.",
        "chat_checkpoint": "a chat checkpoint (one model, e.g. nemotron).",
        "prompted_only": "a system-instruction object; nothing was fit.",
        "none": "unsteered execution.",
        "not_applicable": "emergent-misalignment organisms are checkpoints, not model + object pairs.",
    },
    "OBJECT_ROLES": {
        "constitution": "the constitution corpus / instruction.",
        "control": "the control corpus / instruction.",
        "difference": "constitution minus control.",
        "safety": "the safety instruction.",
        "irrelevant": "the irrelevant control instruction.",
        "baseline_none": "the unsteered baseline.",
        "baseline_published": "a published vector (AlphaSteer / Arditi authors' releases) re-projected onto our checkpoint.",
        "prompt": "a declared prompt object.",
    },
    "OBJECT_NAMED_ROLES": {
        "epoch_1": "the fit's first-epoch checkpoint (the pipeline's `epoch_checkpoints.epochs.epoch_01` child).",
        "epoch_2": "the fit's second-epoch checkpoint.",
        "intermediate": "a checkpoint the fit kept between the named epochs (`extra_json.named_roles = []`).",
        "final": "the fit's final weights, the parent of the epoch children.",
    },
    "SCOPE_KINDS": {
        "run": "one run_id.", "object": "one object_id.", "model": "one model_id.", "eval": "one eval_id.",
        "key": "a partial key (model, object, eval, instrument, campaign, operating point, cap).",
        "comparison": "every run a roster entry of that comparison_id matches (figure-composition rulings).",
    },
    "SUBSTRATES": {
        "cluster_a": "a path under cluster A's shared root (the site config) or `SLURM_CLUSTER_NAME`.",
        "cluster_b": "a path under cluster B's shared root (the site config).",
        "modal": "`twosi-fits:` volume or `/vol/` path.",
        "grading_host": "a local path on the registry host; `substrate_source = derived`. Also the registry's spelling of an unknown substrate on `run`: the column admits no `unverified` value, so a run whose substrate no evidence decides carries `substrate = grading_host`, `substrate_source = unverified` and the reason in `field_confidence_json`.",
        "runpod": "standalone pod (`launch.sh` manifest).",
        "slurm": "a Slurm cluster that is neither `cluster_a` nor `cluster_b`: the producer declares it "
                 "(`SISTEER_REGISTRY_SUBSTRATE=slurm` / `--substrate slurm`) and MUST name the site in `extra_json.cluster_label` "
                 "(`SISTEER_REGISTRY_CLUSTER_LABEL`); the emitter refuses `slurm` without a label and an unknown substrate by name, "
                 "never aliasing either to `grading_host`.",
    },
    "CONFIDENCE": {
        "recorded": "the producer wrote the value. For `substrate`: only a producer field that literally names it (a drive log's `cluster=`, a store's `substrate` field, a manifest's `served_cluster`).",
        "derived": "computed by a documented rule from recorded values. For `substrate`: a routing-table / serving-dir / store path on a shared root, a job-id or cluster-tag prefix -- a prefix is a derivation, never a recorded substrate.",
        "assumed_default": "a documented default filled a value the producer did not write. For `substrate`: what a reader knows about its source as a whole (its submit scripts, its manifests checked once), with the basis in `field_confidence_json`; never filed as `recorded`.",
        "inferred": "a seed value chosen by analogy or read off prose rather than a record (`confidence: inferred` in `configs/registry/*.yaml`, with its `inferred_reason`).",
        "unverified": "unknown; a reason is required in `field_confidence_json`.",
    },
    "EVAL_CATEGORIES": {
        "static_attack": "single-turn jailbreak prompts (pasttense, codeattack, renellm, grok420).",
        "multi_turn": "crescendo, pair, actorattack.",
        "best_of_n": "strongreject_bon.",
        "over_refusal": "xstest, or_bench_hard.",
        "capability": "mmlu, gsm8k, math500, ifeval, bigcodebench, bfcl.",
        "petri": "petri_pooled, petri_<behaviour>.",
        "agentic": "agentic_leaking, agentic_murder.",
        "misalignment": "the paper build's pooled component spelling covering agentic AND em_organism (paperlib `EVAL_COMPONENT`); a legacy label, never seeded on an eval row.",
        "em_organism": "em_<panel> (the emergent-misalignment organisms' own figure panel; never in the agentic category mean).",
        "omit_params": "a route, not a benchmark: the decode_policy row an omit-params request resolves through (the composed vLLM server default); referenced by eval-level rows via `same_values_as`, never carried by an eval.",
    },
    "DEGENERATION_MEASURES": {
        "classifier": "`eval/degeneracy.py` tail/line loop classifier.",
        "contentless": "< 10 alphanumerics after unescape.",
        "rep_ratio_loop": "4-gram repetition ratio > 0.5.",
        "zlib_collapse": "compressed/utf8 bytes < 0.15.",
        "union5": "OR of the five detectors.",
        "at_token_cap": "`finish_reason == length` (the original single column; the two estimands below supersede it).",
        "at_token_cap_retokenized": "the canonical at-cap estimand: the reply re-tokenized with the target's tokenizer (add_special_tokens=False) is >= 0.98 x the token cap (`degeneracy_union` at_cap_retokenized).",
        "at_token_cap_finish_reason": "the producer's declaration: the engine reported `finish_reason == length` / stop reason max_tokens (the stop reason, not the canonical re-tokenized reading); filed BESIDE the canonical estimand, never as it.",
        "loop_first512": "loop within the first 512 tokens (paper-side config only).",
        "defended_degenerate": "degenerate among defended replies (paper-side config only).",
        "tail_loop": "tail-window loop (`TAIL_WINDOW 2000`).",
        "line_loop": "repeated-line loop.",
    },
    "MEASUREMENT_STATES": {
        "measured": "numerator, denominator, denominator_unit and rate are present.",
        "not_measured": "the producer computed no degeneration statistic for this (run, measure, scope); numerator/denominator/rate are NULL. A consumer MUST branch on this state: a NULL rate compared to a threshold once rendered every canonical multi-turn run as 'checked'.",
        "not_applicable": "the measure does not apply to this eval (e.g. at_token_cap on a rule-scored capability item set).",
        "refused": "the producer's inputs disagreed or were unreadable; the row records the refusal, never a number.",
    },
    "COVERAGE_STATES": {
        "measured": "a headline metric row of the run is current in the frame: its metric id is one the eval accepts as its headline "
                    "(`headline_metric_id` or `extra_json.headline_metric_ids`), its judgment is live (not superseded or abandoned) "
                    "and its observation is current. Derived in `v_paper_current`.",
        "undefined_by_geometry": "the run carries `extra_json.nofit_certificate` with kind `nofit_by_design` and `n_items_generated = 0`: "
                                 "no prompt of its windows left room for the minimum completion, so the measurement is undefined rather than "
                                 "missing; `coverage_detail_json` carries the windows, the prompt-token range, the minimum completion, the "
                                 "slots proven not to fit and the slot count. Read before `measured`.",
        "absent": "neither of the above: no current headline row and no certificate -- a gap, or a run that has not finished.",
    },
    "ANALYSIS_INPUT_KINDS": {
        'run': 'an ingested run the analysis read (its transcripts or its captures): `analysis_input.run_id` names it.',
        'object': 'an object the analysis studied (the intervention): `analysis_input.object_id` names it.',
    },
    "ANALYSIS_ARTIFACT_ROLES": {
        'report': 'the analysis\'s report file (`analysis.report_uri`), registered as an artifact by `registry analysis file`.',
        'page': 'a file of a published page directory the filing registered (the page, its figures, its numbers), `extra_json.page_url` naming the page.',
        'figure': 'a figure filed on its own, outside a page directory.',
    },
    "RULING_ACTIONS": {
        "include": "names a run for the paper explicitly.",
        "exclude": "keeps a run out of the paper.",
        "supersede": "a later measurement replaces an earlier one (`target_json.superseded_by`).",
        "coexist": "a same-instrument re-measurement coexists with a `pass_label`.",
        "operating_point": "binds the operating point to an object hash.",
        "descope": "a model leaves the paper.",
        "drop_eval": "an eval leaves the paper.",
        "reference": "kept beside the canonical reading, not drawn.",
        "hold_out": "held out of the figure by ruling (a figure-composition ruling, often scoped to a comparison).",
    },
    "EVIDENCE_STATES": {
        "ok": "evidence found, path resolved, membership readable.",
        "missing": "the named input is absent; the exclusion stays active.",
        "path_absent": "the JSON path does not resolve; the exclusion stays active.",
        "incoherent": "the served record disagrees with the ruled value — covered keys are `indeterminate`; the build refuses.",
        "stale": "the ruled object now equals the declared served object — covered keys are `indeterminate`.",
        "not_evaluated": "no evaluation has run yet.",
    },
    "DISPOSITIONS": {
        "pending": "the source has not run yet.",
        "held": "held back by a ruling or a gate.",
        "not_served": "the model or object was never served for this eval.",
        "not_applicable": "the comparison does not apply to this key.",
        "declined": "the operator declined the measurement (`config.DECLINED`).",
    },
}
#: The SUBSTRATES evidence plus the one row `fit_search.substrate` alone admits.
EVIDENCE["FOLD_OBSERVATION_KINDS"] = {
    'serving_jobs_strict_subset': 'a `run` or `run_patch` record whose `extra.serving_jobs` (a list, compared as a set of its members) names a '
                                  'STRICT subset of the set the run\'s folded `extra_json.serving_jobs` held before it (`observations.derive`, '
                                  'applied by `reducer.Reducer.apply_run_scale` after every refold of a run key; the migration that created the '
                                  'table derived the rows of the runs already folded). The fold replaced the key as it always does; the row records '
                                  'the folded set, the restated set and the dropped members. A detector, never a refusal: `registry observations '
                                  'summary | list` reads it. Equal sets in any order, duplicates, a superset, a first statement and a value that is '
                                  'not a list are not observed.',
}
EVIDENCE["DELIVERY_SIDES"] = {
    'accepted': 'the arrival of a record of an ordered ledger-fact table (`ordered_fact_carrier`) that its file\'s delivery audit counts under '
                '`n_ok` or `n_duplicate` (`harvest.Ingestor.apply_validated` returned `ok` or `duplicate`); which of the two is the fold\'s '
                'arrival order and is not recorded. A record judged again after its fold moves every file that carried it to the other side '
                '(`harvest.Ingestor._move_record_delivery`).',
    'refused': 'the arrival counted under the file\'s `n_refused` (`apply_validated` returned `refused`: a missing parent, a filing not kept, a '
               'conflicting re-filing, a displaced filing); moved to `accepted` when the record is accepted on retry or judged again.',
}
EVIDENCE["FIT_SEARCH_SUBSTRATES"] = dict(
    EVIDENCE["SUBSTRATES"],
    unverified="no PODS.md beside the sweep root, no Modal profiler trace beside a trial and no Modal manifest name the substrate; "
               "the missing evidence is named in `field_confidence_json`.",
)

#: Legacy term → registry noun (the glossary's mapping table; the ONLY place legacy words are
#: spelled outside ``_legacy`` columns).
LEGACY_MAPPING: Tuple[Tuple[str, str, str], ...] = (
    ("arm", "object (`object_id`, `paper_label`, `object_role`)", "`campaign_cell_id_legacy`, `producer_object_key_legacy`"),
    ("cell", "run (one execution) or object, by context", "`campaign_cell_id_legacy`"),
    ("lane", "source of a paper row / transfer setting", "`lane_source_legacy`, `transfer_kind`"),
    ("view", "model_id + transfer_kind", "`view_legacy`"),
    ("leg", "run", "`producer_run_key_legacy`"),
    ("wave", "campaign", "`campaign`"),
    ("grader", "judge / judgment", "`judge_id`, `judgment_id`"),
    ("psr", "relu_steer", "`producer_spellings_json`, `campaign_cell_id_legacy`"),
    ("rung (dose sense)", "strength_nominal", "`operating_point_id_legacy`"),
    ("rung (cap sense)", "token_cap_rung_index", "—"),
    ("dose", "strength_nominal / strength_coefficient", "—"),
    ("pass (bare)", "pass_label", "—"),
    ("cluster (bare)", "substrate", "—"),
    ("r2 (bare)", "paper_round", "—"),
    ("component", "eval_category", "—"),
    ("family (of models)", "model.family", "—"),
    ("target", "model_id", "—"),
)


def check_value(vocab_name: str, value) -> bool:
    """True when ``value`` is admitted by the named vocabulary."""
    return value in VOCABULARIES[vocab_name]
