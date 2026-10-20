-- 0012_instrument_padding_identity.sql
-- Prompt PADDING enters the INSTRUMENT identity (2026-09-22). The padded static-attack stores on qwen35_4b (PG-19 filler of
-- 16,384 or 65,536 tokens placed BEFORE the attack prompt; pasttense / codeattack / renellm / grok420; unsteered and two prompted
-- conditions; cluster B jobs 127302-05) were generated at a 98,304-token window with a 16,384 completion cap in ONE pass, where the
-- unpadded static rows were generated at a 4,864 window with a 4,096 cap and a batch width of 64. A bare `run.padding_tokens` column
-- defaulting to 0 would let a consumer put both on one axis as if the pad were the only difference -- the confound flagged
-- when the padded run was designed (there is no pad-0 store BY DECISION: the unpadded stores already existed and are not a matched
-- zero point). So the pad is an INSTRUMENT: two nullable columns on `instrument` -- `padding_tokens` (the level; NULL = unpadded;
-- 0 refused) and `padding_identity_sha256` (sha256 over the canonical JSON of the filler contract {filler_sha256, mode, placement,
-- separator, tolerance}; `identity.padding_identity_sha256`) -- set together or not at all (`CONSTRAINT instrument_padding_pairing`),
-- the full block riding `extra_json.static_padding`. Both join `identity.INSTRUMENT_ID_FIELDS`, and both are OMITTED from the
-- canonical JSON while NULL (`identity.INSTRUMENT_ID_OPTIONAL_FIELDS`), so every instrument minted before this file keeps its id
-- byte for byte: measured read-only against the live database before landing (1,261 instrument rows, every id reproduced by the new
-- `identity.instrument_id`; tests/test_registry_instrument_padding_identity.py repeats the check on every checkout that has the file).
-- `v_paper_long` exposes `padding_tokens` / `padding_identity_sha256` beside `token_cap_tokens` and `context_window_tokens`, and
-- `v_run_wide` as `instrument_padding_tokens` / `instrument_padding_identity_sha256`; `v_paper_current` joins no instrument column and
-- is unchanged. The reader is `adapters/static_padding.py` (source `static_padding`).
-- Table rebuild: `instrument` (1,261 rows on the live database; a dimension table, well under a second), the two new columns NULL on
-- every copied row; the views that name `instrument` survive the rebuild under the runner's legacy ALTER semantics and the two whose
-- text changed are recreated below. Rendered by `registry migrate --plan --sql` against a schema-version-11 database. Every writer
-- refuses a database behind version 12: `registry migrate` in the read window right after landing.
-- drift: table instrument lacks column(s) padding_tokens, padding_identity_sha256
-- drift: table instrument: CONSTRAINT instrument_padding_pairing is absent
-- drift: v_run_wide differs from the code's SQL
-- drift: v_paper_long differs from the code's SQL

-- rebuild instrument: SQLite cannot alter a constraint in place (migrations/README.md)
-- column padding_tokens is new in the code: nullable: NULL
-- column padding_identity_sha256 is new in the code: nullable: NULL
CREATE TABLE instrument__new (
  instrument_id TEXT NOT NULL,
  eval_id TEXT NOT NULL REFERENCES eval(eval_id),
  item_set_id TEXT NOT NULL REFERENCES item_set(item_set_id),
  harness TEXT NOT NULL CHECK (harness IN ('bespoke_sisteer', 'inspect_ai', 'inspect_evals', 'evalplus', 'bfcl_eval', 'pyrit', 'actorattack', 'petri', 'strongreject', 'agentic_misalignment', 'em_organism', 'lm_eval')),
  harness_version TEXT NOT NULL,
  harness_config_hash TEXT NOT NULL,
  n_shot INTEGER,  -- why-null: not a shot task
  prompt_template_sha256 TEXT NOT NULL CHECK (prompt_template_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]' OR prompt_template_sha256 = 'unverified'),
  cot_prompting TEXT CHECK (cot_prompting IN ('none', 'explicit_instruction', 'few_shot_cot')),  -- why-null: not applicable to attack and judge-graded evals
  scoring_rule TEXT NOT NULL CHECK (scoring_rule IN ('exact_match', 'numeric_match', 'sympy_equiv', 'letter_logprob_prefill', 'judge_threshold', 'judge_graded_weights', 'pass_at_1_sandbox', 'ast_match', 'rate_over_scenarios', 'ifeval_checker', 'judge_score_mean')),
  answer_extraction TEXT NOT NULL CHECK (answer_extraction IN ('regex', 'last_number', 'boxed', 'judge', 'none', 'unverified')),
  token_cap_tokens INTEGER,  -- why-null: unverified for a superseded instrument whose cap was never read (reason in field_confidence_json)
  token_cap_ladder_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(token_cap_ladder_json)),
  ladder_promotion_rule TEXT,  -- why-null: a fixed cap has no ladder
  read_at TEXT NOT NULL CHECK (read_at IN ('ladder_final', 'fixed', 'unverified')),
  reply_cap_tokens INTEGER,  -- why-null: only multi-turn instruments cap replies separately
  context_window_tokens INTEGER,  -- why-null: not declared for every instrument
  padding_tokens INTEGER,  -- why-null: unpadded: the prompt carries no filler (NULL is the unpadded instrument; 0 is refused by CHECK instrument_padding_pairing)
  padding_identity_sha256 TEXT CHECK (padding_identity_sha256 IS NULL OR padding_identity_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'),  -- why-null: unpadded: no filler contract to hash (sha256 over the canonical JSON of {filler_sha256, mode, placement, separator, tolerance}, identity.padding_identity_sha256; the block itself rides extra_json.static_padding)
  decode_source TEXT NOT NULL CHECK (decode_source IN ('generation_config', 'vendor_recommendation', 'model_card', 'vllm_default', 'greedy', 'grandfathered_T1', 'runner_forced_T1', 'legacy_contracts_row', 'unverified')),
  decode_temperature REAL,  -- why-null: not sent (engine default) — resolved value in the run's decode_resolved_json
  decode_top_p REAL,  -- why-null: not sent (engine default)
  decode_top_k INTEGER,  -- why-null: not sent (engine default)
  decode_min_p REAL,  -- why-null: not sent (engine default)
  decode_repetition_penalty REAL,  -- why-null: not sent (engine default)
  decode_presence_penalty REAL,  -- why-null: not sent (engine default)
  decode_seed_rule TEXT NOT NULL CHECK (decode_seed_rule IN ('per_item', 'per_draw', 'per_goal_turn', 'per_scenario_turn', 'per_response', 'fixed', 'none', 'unverified')),
  thinking_mode TEXT NOT NULL CHECK (thinking_mode IN ('on', 'off', 'not_available')),
  n_trials INTEGER,  -- why-null: unverified for a superseded instrument whose trial count was never read (reason in field_confidence_json)
  bon_n INTEGER,  -- why-null: only best-of-n instruments
  max_turns INTEGER,  -- why-null: only multi-turn instruments
  serving_path TEXT NOT NULL CHECK (serving_path IN ('lora_native', 'lens', 'passthrough', 'hf', 'unverified')),
  applied_positions_mode TEXT NOT NULL CHECK (applied_positions_mode IN ('all', 'response', 'response_all_turns', 'prompt_last', 'prefill_all', 'prefill_last_broadcast', 'last', 'none', 'unverified')),
  attacker_model_id TEXT REFERENCES model(model_id),  -- why-null: only attacker-driven evals
  attacker_decode_json TEXT CHECK (attacker_decode_json IS NULL OR json_valid(attacker_decode_json)),  -- why-null: only attacker-driven evals
  judge_id TEXT REFERENCES judge(judge_id),  -- why-null: rule-scored evals (exact match, sandbox) have no judge
  prefill_scheme TEXT,  -- why-null: only prefill-scored instruments
  field_confidence_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(field_confidence_json)),
  notes TEXT,  -- why-null: free text
  field_versions_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(field_versions_json)),  -- {column: [written_utc, record_sha256] of the record that last set it} — the FULL order key (S2).
  extra_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(extra_json)),  -- Unknown record fields, never dropped (Fixed decision 13).
  record_uri TEXT NOT NULL,  -- The record file (logical path) that produced the current row.
  ingest_seq INTEGER NOT NULL,  -- The producing FILE's rank in the ledger's arrival order (record_ingest: arrival_utc, then first_written_utc, sha, uri); derived, never allocated — every row above a head a reader has seen is new to it.
  PRIMARY KEY (instrument_id),
  CONSTRAINT instrument_padding_pairing CHECK ((padding_tokens IS NULL) = (padding_identity_sha256 IS NULL) AND (padding_tokens IS NULL OR padding_tokens > 0))
);
INSERT INTO instrument__new (instrument_id, eval_id, item_set_id, harness, harness_version, harness_config_hash, n_shot, prompt_template_sha256, cot_prompting, scoring_rule, answer_extraction, token_cap_tokens, token_cap_ladder_json, ladder_promotion_rule, read_at, reply_cap_tokens, context_window_tokens, decode_source, decode_temperature, decode_top_p, decode_top_k, decode_min_p, decode_repetition_penalty, decode_presence_penalty, decode_seed_rule, thinking_mode, n_trials, bon_n, max_turns, serving_path, applied_positions_mode, attacker_model_id, attacker_decode_json, judge_id, prefill_scheme, field_confidence_json, notes, field_versions_json, extra_json, record_uri, ingest_seq) SELECT instrument_id, eval_id, item_set_id, harness, harness_version, harness_config_hash, n_shot, prompt_template_sha256, cot_prompting, scoring_rule, answer_extraction, token_cap_tokens, token_cap_ladder_json, ladder_promotion_rule, read_at, reply_cap_tokens, context_window_tokens, decode_source, decode_temperature, decode_top_p, decode_top_k, decode_min_p, decode_repetition_penalty, decode_presence_penalty, decode_seed_rule, thinking_mode, n_trials, bon_n, max_turns, serving_path, applied_positions_mode, attacker_model_id, attacker_decode_json, judge_id, prefill_scheme, field_confidence_json, notes, field_versions_json, extra_json, record_uri, ingest_seq FROM instrument ORDER BY instrument_id;
DROP TABLE instrument;
ALTER TABLE instrument__new RENAME TO instrument;
ANALYZE instrument;

DROP VIEW IF EXISTS v_paper_long;
CREATE VIEW v_paper_long AS
WITH withdrawn_metric AS MATERIALIZED (
    -- a metric row whose observation key is no longer current under any revision (one pass; the per-row scan does not scale)
    SELECT json_extract(observation_key, '$.run_id') AS run_id, json_extract(observation_key, '$.judgment_id') AS judgment_id,
           json_extract(observation_key, '$.metric_id') AS metric_id, json_extract(observation_key, '$.threshold') AS threshold,
           json_extract(observation_key, '$.computation_id') AS computation_id
    FROM observation WHERE table_name = 'run_metric' GROUP BY observation_key HAVING MAX(current) = 0
),
degen_rate AS MATERIALIZED (
    -- the MEASURED rates of the current degeneration rows of every run at the frac_* mirrors own (measure, scope), one row per
    -- (run_id, judgment_id); the paper objects fill a NULL mirror from here (migration 0010: a reading filed by another reader counts).
    -- No apostrophe and no bare table name in a comment here: the reference scan strips quoted literals and reads identifiers
    SELECT run_id, judgment_id,
           MAX(CASE WHEN measure = 'at_token_cap' AND scope = 'all' THEN rate END) AS at_token_cap,
           MAX(CASE WHEN measure = 'at_token_cap_finish_reason' AND scope = 'all' THEN rate END) AS at_token_cap_finish_reason,
           MAX(CASE WHEN measure = 'classifier' AND scope = 'all' THEN rate END) AS classifier,
           MAX(CASE WHEN measure = 'contentless' AND scope = 'all' THEN rate END) AS contentless,
           MAX(CASE WHEN measure = 'rep_ratio_loop' AND scope = 'all' THEN rate END) AS rep_ratio_loop,
           MAX(CASE WHEN measure = 'zlib_collapse' AND scope = 'all' THEN rate END) AS zlib_collapse,
           MAX(CASE WHEN measure = 'union5' AND scope = 'all' THEN rate END) AS union5,
           MAX(CASE WHEN measure = 'loop_or_at_cap' AND scope = 'all' THEN rate END) AS loop_or_at_cap,
           MAX(CASE WHEN measure = 'loop_first512' AND scope = 'first_512_tokens' THEN rate END) AS loop_first512,
           MAX(CASE WHEN measure = 'defended_degenerate' AND scope = 'defended_only' THEN rate END) AS defended_degenerate
    FROM v_run_degeneration_current WHERE measurement_state = 'measured' GROUP BY run_id, judgment_id
),
metric_source AS MATERIALIZED (
    -- the source whose head revision carries each run_metric key (NULL for a natively emitted row); two sources carrying one key
    -- resolve by source rank (priority, then source id)
    SELECT run_id, judgment_id, metric_id, threshold, computation_id, source_id, priority FROM (
        SELECT CAST(json_extract(ob.observation_key, '$.run_id') AS TEXT) AS run_id,
               CAST(json_extract(ob.observation_key, '$.judgment_id') AS TEXT) AS judgment_id,
               CAST(json_extract(ob.observation_key, '$.metric_id') AS TEXT) AS metric_id,
               CAST(json_extract(ob.observation_key, '$.threshold') AS REAL) AS threshold,
               CAST(json_extract(ob.observation_key, '$.computation_id') AS TEXT) AS computation_id, ob.source_id, COALESCE(sr.priority, 0) AS priority,
               ROW_NUMBER() OVER (PARTITION BY ob.observation_key ORDER BY COALESCE(sr.priority, 0) DESC, ob.source_id) AS rk
        FROM observation ob LEFT JOIN source_revision sr ON sr.revision_id = ob.revision_id
        WHERE ob.table_name = 'run_metric' AND ob.current = 1
    ) WHERE rk = 1
),
run_source AS MATERIALIZED (
    -- the source that OWNS the run: the top-ranked source among those whose head revision carries the run key (NULL for a run no
    -- adapter carries)
    SELECT run_id, source_id FROM (
        SELECT CAST(json_extract(ob.observation_key, '$.run_id') AS TEXT) AS run_id, ob.source_id,
               ROW_NUMBER() OVER (PARTITION BY ob.observation_key ORDER BY COALESCE(sr.priority, 0) DESC, ob.source_id) AS rk
        FROM observation ob LEFT JOIN source_revision sr ON sr.revision_id = ob.revision_id
        WHERE ob.table_name = 'run' AND ob.current = 1
    ) WHERE rk = 1
),
metric_role AS MATERIALIZED (
    -- ONE owner row per (run, judgment, metric, threshold): the row of the source that owns the run first, then a computed row (a non-blank
    -- computation_id) before a producer-filed blank one (the twin the stats readers file beside an export row), then the higher declared priority,
    -- the lexically smaller source id, the smaller computation_id — a total order, so every other row at the identity is secondary
    SELECT rm.run_id, rm.judgment_id, rm.metric_id, rm.threshold, rm.computation_id, ms.source_id AS metric_source_id, rs.source_id AS run_source_id,
           ROW_NUMBER() OVER (PARTITION BY rm.run_id, rm.judgment_id, rm.metric_id, rm.threshold
                              ORDER BY CASE WHEN ms.source_id IS rs.source_id THEN 0 ELSE 1 END, CASE WHEN rm.computation_id = '' THEN 1 ELSE 0 END,
                                       COALESCE(ms.priority, 0) DESC, COALESCE(ms.source_id, ''), rm.computation_id) AS rk,
           COUNT(*) OVER (PARTITION BY rm.run_id, rm.judgment_id, rm.metric_id, rm.threshold) AS n_candidates
    FROM run_metric rm
    LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
    LEFT JOIN metric_source ms ON ms.run_id = rm.run_id AND ms.judgment_id = rm.judgment_id AND ms.metric_id = rm.metric_id
         AND ms.threshold = rm.threshold AND ms.computation_id = rm.computation_id
    LEFT JOIN run_source rs ON rs.run_id = rm.run_id
    WHERE wm.run_id IS NULL
),
metric_role_contest AS MATERIALIZED (
    -- the tie the rank resolves by priority: two or more computed rows (non-blank computation_id) from different sources at one identity
    SELECT run_id, judgment_id, metric_id, threshold,
           COUNT(DISTINCT CASE WHEN computation_id <> '' THEN COALESCE(metric_source_id, '') END) AS n_computed_sources
    FROM metric_role GROUP BY run_id, judgment_id, metric_id, threshold
)
SELECT pc.round_id,
       -- the flag of the run on its OWNER metric row only: a secondary row at the same identity is evidence, never a second paper row
       CASE WHEN mr.rk = 1 THEN pc.in_current_paper ELSE 0 END AS in_current_paper,
       pc.selection_state, pc.held_out,
       -- per MEASUREMENT: the comparisons / roster entries the judge of this row satisfies, among the entries the run
       -- still carries (a comparison-scoped exclude removed the others from the run in v_paper_current, never the row from the paper)
       (SELECT json_group_array(DISTINCT vm.comparison_id) FROM v_roster_match vm JOIN roster_entry re ON re.roster_entry_id = vm.roster_entry_id
         WHERE vm.run_id = r.run_id
           AND EXISTS (SELECT 1 FROM json_each(pc.roster_entry_ids_json) kept WHERE kept.value = vm.roster_entry_id)
           AND CASE
                          WHEN re.judge_id IS NOT NULL THEN jm.judge_id = re.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                          WHEN re.judge_selector_json IS NOT NULL THEN j.judge_id IS NOT NULL AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                               AND (json_extract(re.judge_selector_json, '$.judge_model') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.judge_model') = j.judge_model)
                               AND (json_extract(re.judge_selector_json, '$.rubric_name') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.rubric_name') = j.rubric_name)
                          ELSE (rm.judgment_id = '' AND i.judge_id IS NULL)
                               OR (jm.judge_id IS NOT NULL AND jm.judge_id IS i.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0))
                        END) AS comparison_ids_json,
       (SELECT json_group_array(DISTINCT vm.roster_entry_id) FROM v_roster_match vm JOIN roster_entry re ON re.roster_entry_id = vm.roster_entry_id
         WHERE vm.run_id = r.run_id
           AND EXISTS (SELECT 1 FROM json_each(pc.roster_entry_ids_json) kept WHERE kept.value = vm.roster_entry_id)
           AND CASE
                          WHEN re.judge_id IS NOT NULL THEN jm.judge_id = re.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                          WHEN re.judge_selector_json IS NOT NULL THEN j.judge_id IS NOT NULL AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                               AND (json_extract(re.judge_selector_json, '$.judge_model') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.judge_model') = j.judge_model)
                               AND (json_extract(re.judge_selector_json, '$.rubric_name') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.rubric_name') = j.rubric_name)
                          ELSE (rm.judgment_id = '' AND i.judge_id IS NULL)
                               OR (jm.judge_id IS NOT NULL AND jm.judge_id IS i.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0))
                        END) AS roster_entry_ids_json,
       pc.decisive_ruling_id, pc.decisive_action,
       r.run_id, r.campaign, r.lane_source_legacy, r.view_legacy, r.campaign_cell_id_legacy, r.operating_point_id_legacy,
       r.producer_run_key_legacy, r.producer_object_key_legacy,
       r.model_id, m.family AS model_family, r.transfer_kind, r.strength_nominal, r.strength_coefficient, r.is_operating_point,
       -- how the run counts as being at the operating point (migration 0011; operating_point_basis_sql)
       CASE WHEN r.is_operating_point = 1 THEN 'stamped'
            WHEN r.is_operating_point IS NULL AND o.object_kind IN ('lora', 'lora_subtraction', 'composed', 'relu_steer', 'relu_steer_distilled_difference', 'alphasteer', 'static_r_arditi', 'vector', 'vector_all', 'diffmean', 'antidote')
                 AND ABS(r.strength_nominal - 1.0) < 1e-9 THEN 'fallback_nominal_1p0'
            WHEN r.is_operating_point IS NULL AND o.object_kind = 'vector_all'
                 AND ABS(r.strength_coefficient - 0.3) < 1e-9 THEN 'fallback_arditi_c0p3'
            WHEN o.object_kind IN ('none', 'system_instruction') THEN 'no_strength_axis'
            ELSE 'none' END AS operating_point_basis,
       r.object_id, o.paper_label, o.object_role, o.object_kind, o.fit_model_id, o.fit_layers_json, o.fit_layer_count, o.fit_replicate_label,
       o.status AS object_status, o.artifact_sha256 AS object_artifact_sha256,
       r.eval_id, e.eval_category, e.headline_metric_id, r.instrument_id, r.item_set_id, r.harness, r.n_shot, r.cot_prompting, r.thinking_mode,
       i.bon_n, i.n_trials, i.max_turns,
       r.token_cap_tokens, r.token_cap_rung_index, r.context_window_tokens,
       -- the prompt padding of the instrument beside the cap and the window (migration 0012): NULL is unpadded; a padded run is
       -- another instrument (its cap and window differ too), so a consumer grouping by instrument never averages a padded row into an unpadded one
       i.padding_tokens, i.padding_identity_sha256,
       r.serving_path, r.serving_config_sha256,
       r.serving_identity, r.serving_engine_version,
       r.decode_source, r.decode_sent_json, r.decode_policy_id, r.decode_policy_match,
       -- the decode_policy row resolve() returns TODAY for the key of the run, and whether the matched row IS it (migration 0009: the
       -- vendor preset is the default of every family): `exact` beside decode_is_family_default 0 reads "matched
       -- a recorded, NON-default policy" (the agentic runs at the T = 1.0 of the runner), never a mismatch; NULL when nothing matched
       COALESCE((SELECT p.policy_id FROM decode_policy p WHERE p.model_id = r.model_id AND p.thinking_mode = r.thinking_mode AND p.eval_id = r.eval_id AND p.is_default = 1 AND p.superseded_by_policy_id IS NULL), (SELECT p.policy_id FROM decode_policy p JOIN eval fe ON fe.eval_id = r.eval_id WHERE p.model_id = r.model_id AND p.thinking_mode = r.thinking_mode AND p.eval_id IS NULL AND p.eval_category = fe.eval_category AND p.is_default = 1 AND p.superseded_by_policy_id IS NULL)) AS decode_family_default_policy_id,
       CASE WHEN r.decode_policy_id IS NULL THEN NULL WHEN r.decode_policy_id = COALESCE((SELECT p.policy_id FROM decode_policy p WHERE p.model_id = r.model_id AND p.thinking_mode = r.thinking_mode AND p.eval_id = r.eval_id AND p.is_default = 1 AND p.superseded_by_policy_id IS NULL), (SELECT p.policy_id FROM decode_policy p JOIN eval fe ON fe.eval_id = r.eval_id WHERE p.model_id = r.model_id AND p.thinking_mode = r.thinking_mode AND p.eval_id IS NULL AND p.eval_category = fe.eval_category AND p.is_default = 1 AND p.superseded_by_policy_id IS NULL)) THEN 1 ELSE 0 END AS decode_is_family_default,
       jm.judge_id, j.judge_model, j.rubric_name AS judge_rubric_name, r.judge_id AS run_judge_id,
       jm.n_judge_refusals AS judgment_n_judge_refusals,
       r.pass_label, r.lifecycle_state, CASE WHEN r.lifecycle_state IN ('judged', 'ingested') THEN 1 ELSE 0 END AS is_complete, r.store_uri, r.intervention_hash, r.notes, r.extra_json AS run_extra_json,
       COALESCE(json_extract(rm.extra_json, '$.eval_id_legacy'), json_extract(r.extra_json, '$.eval_id_legacy')) AS eval_id_legacy,
       rm.judgment_id, rm.metric_id, rm.producer_metric_spelling, rm.threshold, rm.computation_id,
       CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE rm.value END AS value, CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE rm.k END AS k, CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE rm.n END AS n, CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE rm.ci_lo END AS ci_lo, CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE rm.ci_hi END AS ci_hi,
       rm.ci_method,
       -- the frac_* mirror the writer filed, else the MEASURED degeneration rate of the run at the measure and scope of that mirror
       -- (migration 0010: the exact-judgment row first, then the judgment-less rows); a mirrored value still wins where a writer filed one
       COALESCE(rm.frac_at_token_cap, dj.at_token_cap, dn.at_token_cap, dj.at_token_cap_finish_reason, dn.at_token_cap_finish_reason) AS frac_at_token_cap, COALESCE(rm.frac_degenerate_classifier, dj.classifier, dn.classifier) AS frac_degenerate_classifier, COALESCE(rm.frac_degenerate_contentless, dj.contentless, dn.contentless) AS frac_degenerate_contentless, COALESCE(rm.frac_degenerate_rep_ratio_loop, dj.rep_ratio_loop, dn.rep_ratio_loop) AS frac_degenerate_rep_ratio_loop, COALESCE(rm.frac_degenerate_zlib_collapse, dj.zlib_collapse, dn.zlib_collapse) AS frac_degenerate_zlib_collapse, COALESCE(rm.frac_degenerate_union5, dj.union5, dn.union5) AS frac_degenerate_union5, COALESCE(rm.frac_loop_first512, dj.loop_first512, dn.loop_first512) AS frac_loop_first512, COALESCE(rm.frac_defended_degenerate, dj.defended_degenerate, dn.defended_degenerate) AS frac_defended_degenerate,
       rm.n_abstentions, rm.n_error_outcomes, rm.n_incomplete, rm.source_uri,
       rm.source_producers_json, rm.extra_json AS metric_extra_json,
       rc.anchor_run_id, rc.anchor_kind, rc.delta, rc.delta_lo, rc.delta_hi, rc.delta_p, rc.delta_method, rc.effect_size,
       rc.effect_kind, rc.n_pairs, rc.extra_json AS contrast_extra_json,
       -- the token cap of the anchor run beside the cap of the run itself, and whether the two agree (migration 0006): a contrast
       -- differences the values of two runs, and a consumer reading `delta` here never joined both caps, so a pair whose caps
       -- differ was invisible (202 of 8,640 live contrasts paired different caps, 2026-09-21, every one an escalation-ladder pair,
       -- most often a cap-8192 run against a cap-4096 or cap-2048 anchor). 1 when the caps are equal, 0 when they differ, NULL
       -- when the row carries no anchor or either cap is NULL
       ra.token_cap_tokens AS anchor_token_cap_tokens,
       CASE WHEN r.token_cap_tokens IS NULL OR ra.token_cap_tokens IS NULL THEN NULL
            WHEN r.token_cap_tokens = ra.token_cap_tokens THEN 1 ELSE 0 END AS cap_matched,
       -- multi-source ownership (ownership.MULTI_SOURCE_OWNERSHIP): the role of the row inside a declared scope and the outcome of the
       -- fold; NULL everywhere outside a declared scope. An owner row reads the state of the matched identity (folded | refused; NULL
       -- when the secondary carries no row at that identity); a secondary row that reaches here is a stopgap (folded and refused ones
       -- are dropped below)
       msr.declaration_id AS multi_source_declaration_id, msr.role AS multi_source_role,
       CASE WHEN msr.role = 'owner' THEN msc.state WHEN msr.role = 'secondary' THEN 'stopgap' END AS multi_source_state,
       msc.reason AS multi_source_reason,
       CASE WHEN msr.role = 'owner' THEN msc.secondary_run_id END AS multi_source_secondary_run_id,
       CASE WHEN msr.role = 'owner' THEN msc.carried_json END AS multi_source_carried_json,
       CASE WHEN msr.role = 'owner' THEN msc.degenerate_share_secondary END AS multi_source_degenerate_share,
       CASE WHEN msr.role = 'owner' THEN msc.loop_share_secondary END AS multi_source_loop_share,
       CASE WHEN msr.role = 'owner' THEN msc.loop_share_measure_secondary END AS multi_source_loop_share_measure,
       -- metric role (METRIC_ROLE_CTES): the ONE row per (run, judgment, metric, threshold) the paper counts (owner) against the other
       -- provenances of the same measurement kept as evidence (secondary, in_current_paper 0); the source of the row and the source that owns the run
       -- (NULL = natively emitted / no adapter carries the run); how many rows stood at the identity; whether two or more of them were
       -- computed rows (non-blank computation_id) from different sources (the tie resolved by declared priority, then source id)
       CASE WHEN mr.rk = 1 THEN 'owner' ELSE 'secondary' END AS metric_role,
       mr.metric_source_id, mr.run_source_id,
       mr.n_candidates AS metric_role_candidates,
       CASE WHEN mc.n_computed_sources > 1 THEN 1 ELSE 0 END AS metric_role_contested,
       MAX(r.ingest_seq, rm.ingest_seq, COALESCE(rc.ingest_seq, 0), COALESCE(msc.ingest_seq, 0)) AS ingest_seq
FROM v_paper_current pc
JOIN run r ON r.run_id = pc.run_id
JOIN model m ON m.model_id = r.model_id
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN instrument i ON i.instrument_id = r.instrument_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN judgment jm ON jm.judgment_id = rm.judgment_id
LEFT JOIN judge j ON j.judge_id = jm.judge_id
LEFT JOIN run_contrast rc ON rc.run_id = rm.run_id AND rc.judgment_id = rm.judgment_id AND rc.metric_id = rm.metric_id
     AND rc.threshold = rm.threshold AND rc.computation_id = rm.computation_id
LEFT JOIN run ra ON ra.run_id = rc.anchor_run_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
LEFT JOIN v_multi_source_rows msr ON msr.run_id = rm.run_id AND msr.judgment_id = rm.judgment_id AND msr.metric_id = rm.metric_id
     AND msr.threshold = rm.threshold AND msr.computation_id = rm.computation_id
LEFT JOIN v_multi_source_check msc ON msc.declaration_id = msr.declaration_id AND msc.pairing_key = msr.pairing_key AND msc.eval_id = msr.eval_id
     AND msc.metric_id = msr.metric_id AND msc.threshold = msr.threshold
JOIN metric_role mr ON mr.run_id = rm.run_id AND mr.judgment_id = rm.judgment_id AND mr.metric_id = rm.metric_id
     AND mr.threshold = rm.threshold AND mr.computation_id = rm.computation_id
JOIN metric_role_contest mc ON mc.run_id = rm.run_id AND mc.judgment_id = rm.judgment_id AND mc.metric_id = rm.metric_id AND mc.threshold = rm.threshold
WHERE wm.run_id IS NULL
  -- a secondary row whose identity the owner also carries leaves the frame: folded (the owner row stands) or refused (neither is admitted)
  AND (msr.role IS NULL OR msr.role <> 'secondary' OR msc.state IS NULL OR msc.state = 'stopgap')
  AND (NOT EXISTS (SELECT 1 FROM v_roster_match vm WHERE vm.run_id = r.run_id)
       OR EXISTS (SELECT 1 FROM v_roster_match vm JOIN roster_entry re ON re.roster_entry_id = vm.roster_entry_id
                  WHERE vm.run_id = r.run_id AND CASE
                          WHEN re.judge_id IS NOT NULL THEN jm.judge_id = re.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                          WHEN re.judge_selector_json IS NOT NULL THEN j.judge_id IS NOT NULL AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                               AND (json_extract(re.judge_selector_json, '$.judge_model') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.judge_model') = j.judge_model)
                               AND (json_extract(re.judge_selector_json, '$.rubric_name') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.rubric_name') = j.rubric_name)
                          ELSE (rm.judgment_id = '' AND i.judge_id IS NULL)
                               OR (jm.judge_id IS NOT NULL AND jm.judge_id IS i.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0))
                        END));
DROP VIEW IF EXISTS v_run_wide;
CREATE VIEW v_run_wide AS
SELECT r.*,
       -- 1 once the run is judged or ingested (a complete run reads `ingested` at rest, `judged` before the harvester stamps it)
       CASE WHEN r.lifecycle_state IN ('judged', 'ingested') THEN 1 ELSE 0 END AS is_complete,
       m.family, m.checkpoint_kind, m.hf_repo, m.hf_revision, m.paper_status AS model_paper_status,
       o.object_kind, o.paper_label, o.object_role, o.fit_target_kind, o.fit_model_id, o.fit_positions,
       o.fit_layers_json, o.fit_layer_count, o.fit_rank, o.fit_seed, o.fit_seed_confidence, o.fit_replicate_label,
       o.status AS object_status, o.artifact_sha256 AS object_artifact_sha256,
       fm.checkpoint_kind AS fit_model_checkpoint_kind,
       e.eval_category, e.headline_metric_id, e.metric_direction, e.paper_status AS eval_paper_status,
       s.dataset_version, s.n_items AS item_set_n_items, s.subsample_rule,
       i.harness AS instrument_harness, i.harness_version AS instrument_harness_version,
       i.n_shot AS instrument_n_shot, i.cot_prompting AS instrument_cot_prompting,
       i.scoring_rule AS instrument_scoring_rule, i.token_cap_tokens AS instrument_token_cap_tokens,
       i.padding_tokens AS instrument_padding_tokens, i.padding_identity_sha256 AS instrument_padding_identity_sha256,
       i.decode_source AS instrument_decode_source, i.decode_temperature AS instrument_decode_temperature,
       i.decode_top_p AS instrument_decode_top_p, i.decode_top_k AS instrument_decode_top_k,
       i.thinking_mode AS instrument_thinking_mode, i.serving_path AS instrument_serving_path,
       i.applied_positions_mode AS instrument_applied_positions_mode, i.judge_id AS instrument_judge_id,
       j.judge_model, j.rubric_name, j.rubric_version, j.threshold_rule
FROM run r
JOIN model m ON m.model_id = r.model_id
JOIN object o ON o.object_id = r.object_id
LEFT JOIN model fm ON fm.model_id = o.fit_model_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN item_set s ON s.item_set_id = r.item_set_id
JOIN instrument i ON i.instrument_id = r.instrument_id
LEFT JOIN judge j ON j.judge_id = r.judge_id;
