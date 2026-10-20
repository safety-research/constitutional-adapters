-- 0019_paper_long_wilson_interval.sql
-- `v_paper_long` derives the 95 percent Wilson score interval where a metric row of a yes-or-no proportion carries k and n, no filed
-- interval, and a value that IS k / n (a ruling of 2026-10-01; the metric roster and the value guard after the two reviews
-- of the change). Before 0019 the frame returned `run_metric.ci_lo` / `ci_hi` / `ci_method` as filed, so a reading whose writer filed k and
-- n without an interval reached every consumer with NULL bounds and each consumer that wanted a bar computed its own (the notebook, from
-- value and n at a z of 1.959964). Since 0019 the frame fills them in SQL (schema_def.WILSON_CI_LO_SQL / WILSON_CI_HI_SQL /
-- WILSON_CI_METHOD_SQL): a filed bound is returned as filed, byte for byte, under its filed `ci_method`; a row of a metric on the roster
-- schema_def.WILSON_PROPORTION_METRIC_IDS with k and n > 0, no filed bound and a value equal to k / n (to 1e-9, WILSON_VALUE_TOLERANCE)
-- reads the Wilson bounds -- paperlib.stats.wilson / adapters.declared_common.wilson term for term, p = k / n as a REAL, d = 1 + z^2 / n,
-- center (p + z^2 / 2n) / d, half-width z * sqrt(p (1 - p) / n + z^2 / 4n^2) / d, at the same z = 1.959963984540054 -- under `ci_method`
-- wilson_95 (a method text a writer filed WITHOUT bounds gives way to it there: the label describes the bounds in the row); a metric off
-- the roster, k alone, n alone, n = 0 (the Python returns nan there), a NULL value, or a value that is NOT k / n read the filed columns,
-- NULL included. The roster is the estimand check: the Wilson interval is the interval of a Bernoulli proportion, and a run_metric row's
-- k and n do not always describe one. The registry carries no unit or estimand per metric (metric_id is a bare string), so the frame
-- carries one -- the metric ids whose writers file k as a count of items with a yes-or-no outcome and n as the items judged, each read in
-- its writer: `asr`, `asr_held_in`, `asr_held_out`, the `asr_at_n*` ladder (n1 .. n64, n75), `asr_bon_cap4096` / `asr_bon_cap8192`, `acc`,
-- `xst_refused` / `orb_refused`, the BoN share rows `at_cap_share_bon_512`, `coherent_at_cap_share_bon_512`, `loop_share_bon_512`,
-- `secondary_flag_only_share_bon_512`, `residual_at_cap_share_bon_4096`. Off it, by design: `over_refusal_graded` and
-- `over_refusal_label_mean` score each item 0, 0.5 or 1 and file the score SUM as k beside the mean, so a value that equals k / n exactly
-- (a whole score sum: 561 ones + 28 halves + 728 zeros = 575 of 1,317; or 250 halves, value 0.5 at k 125, a row with no item variation for
-- which a Bernoulli interval would read about [0.44, 0.56]) says nothing about the per-item variance the interval is about -- their
-- interval is adapters.declared_common.label_mean_interval, from the {0, 0.5, 1} census the frame does not hold; `misaligned_rate` files
-- value = 100 k / n (the percent scale); `recovered_vs_others_asr_gap` files a difference beside the k and n of one of its components.
-- The value guard stays beside the roster as the scale check: a listed metric filed as a percent, or a value beside the k and n of some
-- other quantity, is not k / n and derives nothing either. Every row that derives nothing keeps its NULL and its filed method text, what
-- it read before 0019. The multi-source refusal still wraps the bounds: a refused owner row reads NULL whether its interval was filed or
-- would be derived.
-- Live census, read-only, 2026-10-01 (31,553 run_metric rows): 6,670 rows carry k and n > 0 with no interval; 6,611 of them are roster
-- rows with a value equal to k / n (every one exactly, bit for bit) and derive (asr 4,393; asr_held_in / asr_held_out 515 each;
-- xst_refused 133; the asr_at_n* ladder 101 each and asr_at_n75 1; orb_refused 100; the BoN share rows 48-50 each; acc 4;
-- asr_bon_cap4096 1), and the 59 that do not are every over_refusal_graded row (44, 19 of them with value = k / n), every
-- recovered_vs_others_asr_gap row (9, under the method text `none`) and every misaligned_rate row (6, the k = 0 row included); the 16,810
-- rows with a filed bound read identically under the old and the new expressions, and no row that derives nothing has its method text
-- moved. On the frame (the old object as installed and the new one as a temporary object over one database): 30,166 rows and 29,672
-- distinct keys either way, the same key set; 6,387 rows gain an interval (1,340 of them in the current paper), every one with
-- value = k / n under wilson_95; the frame rows with k, n > 0 and no interval under the new object are exactly the 59 off-roster rows
-- (8 in the current paper, all over_refusal_graded); the 16,454 frame rows with a filed interval read identically (the multiset of
-- (key, ci_lo, ci_hi, ci_method) over them is the same under both objects, and the 6,387 groups new to it are the gained rows); no value,
-- k or n moved and no ci_method moved outside the gained rows; on every derived row the SQL bound equals the Python bound exactly (max abs
-- difference 0.0). `sqrt` is the SQLite math function (SQLITE_ENABLE_MATH_FUNCTIONS, 3.35+; 3.45.1 on the grading host and the cluster gate
-- venvs): a build without it cannot compile the object. View-only: no table changes, no rows copied, the column set of v_paper_long
-- unchanged. Rendered by `registry migrate --plan --sql` against a schema-version-18 database.
-- drift: v_paper_long differs from the code's SQL

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
),
withdrawn_contrast AS MATERIALIZED (
    -- a contrast whose observation key is no longer current under any revision: the anchor no head revision pairs the run with (0017)
    SELECT CAST(json_extract(observation_key, '$.run_id') AS TEXT) AS run_id, CAST(json_extract(observation_key, '$.anchor_run_id') AS TEXT) AS anchor_run_id,
           CAST(json_extract(observation_key, '$.judgment_id') AS TEXT) AS judgment_id, CAST(json_extract(observation_key, '$.metric_id') AS TEXT) AS metric_id,
           CAST(json_extract(observation_key, '$.threshold') AS REAL) AS threshold, CAST(json_extract(observation_key, '$.computation_id') AS TEXT) AS computation_id
    FROM observation WHERE table_name = 'run_contrast' GROUP BY observation_key HAVING MAX(current) = 0
),
current_contrast AS MATERIALIZED (
    -- the contrasts the frame joins: every run_contrast row whose observation key is still current under some revision (or was never
    -- observed); the row of an anchor only a superseded revision paired the run with is left out, so a metric row joins each anchor once (0017)
    SELECT rc.* FROM run_contrast rc
    LEFT JOIN withdrawn_contrast wc ON wc.run_id = rc.run_id AND wc.anchor_run_id = rc.anchor_run_id AND wc.judgment_id = rc.judgment_id
         AND wc.metric_id = rc.metric_id AND wc.threshold = rc.threshold AND wc.computation_id = rc.computation_id
    WHERE wc.run_id IS NULL
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
       CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE rm.value END AS value, CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE rm.k END AS k, CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE rm.n END AS n,
       -- the interval the writer filed, else the 95 percent Wilson score interval of k over n derived here (migration 0019;
       -- WILSON_CI_LO_SQL / WILSON_CI_HI_SQL / WILSON_CI_METHOD_SQL): a filed bound is returned as filed, byte for byte, under its filed
       -- method; a row of a metric on the yes-or-no roster (WILSON_PROPORTION_METRIC_IDS: k counts items with a yes-or-no outcome, so k
       -- over n is a Bernoulli proportion and the Wilson interval is its interval) with k, n above zero, no filed bound and a value that
       -- IS k over n (the scale check: a percent-scale value or a difference beside some other count is not) reads the derived bounds
       -- under the method wilson_95; a metric off the roster (a graded mean whose k is a sum of 0, 0.5 and 1 scores, whatever its value),
       -- k alone, n alone, n = 0 or a value that is not k over n read the filed columns, NULL included
       -- (no apostrophe in this comment: the reference scan strips quoted literals)
       CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE CASE WHEN rm.ci_lo IS NOT NULL THEN rm.ci_lo WHEN rm.ci_lo IS NULL AND rm.ci_hi IS NULL AND rm.metric_id IN ('asr', 'asr_held_in', 'asr_held_out', 'asr_at_n1', 'asr_at_n2', 'asr_at_n4', 'asr_at_n8', 'asr_at_n16', 'asr_at_n32', 'asr_at_n64', 'asr_at_n75', 'asr_bon_cap4096', 'asr_bon_cap8192', 'acc', 'xst_refused', 'orb_refused', 'at_cap_share_bon_512', 'coherent_at_cap_share_bon_512', 'loop_share_bon_512', 'secondary_flag_only_share_bon_512', 'residual_at_cap_share_bon_4096') AND rm.k IS NOT NULL AND rm.n > 0 AND abs(rm.value - rm.k*1.0/rm.n) <= 1e-9 THEN ((rm.k*1.0/rm.n) + 1.959963984540054*1.959963984540054/(2.0*rm.n))/(1.0 + 1.959963984540054*1.959963984540054/rm.n) - 1.959963984540054*sqrt((rm.k*1.0/rm.n)*(1.0 - rm.k*1.0/rm.n)/rm.n + 1.959963984540054*1.959963984540054/(4.0*rm.n*rm.n))/(1.0 + 1.959963984540054*1.959963984540054/rm.n) ELSE NULL END END AS ci_lo,
       CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE CASE WHEN rm.ci_hi IS NOT NULL THEN rm.ci_hi WHEN rm.ci_lo IS NULL AND rm.ci_hi IS NULL AND rm.metric_id IN ('asr', 'asr_held_in', 'asr_held_out', 'asr_at_n1', 'asr_at_n2', 'asr_at_n4', 'asr_at_n8', 'asr_at_n16', 'asr_at_n32', 'asr_at_n64', 'asr_at_n75', 'asr_bon_cap4096', 'asr_bon_cap8192', 'acc', 'xst_refused', 'orb_refused', 'at_cap_share_bon_512', 'coherent_at_cap_share_bon_512', 'loop_share_bon_512', 'secondary_flag_only_share_bon_512', 'residual_at_cap_share_bon_4096') AND rm.k IS NOT NULL AND rm.n > 0 AND abs(rm.value - rm.k*1.0/rm.n) <= 1e-9 THEN ((rm.k*1.0/rm.n) + 1.959963984540054*1.959963984540054/(2.0*rm.n))/(1.0 + 1.959963984540054*1.959963984540054/rm.n) + 1.959963984540054*sqrt((rm.k*1.0/rm.n)*(1.0 - rm.k*1.0/rm.n)/rm.n + 1.959963984540054*1.959963984540054/(4.0*rm.n*rm.n))/(1.0 + 1.959963984540054*1.959963984540054/rm.n) ELSE NULL END END AS ci_hi,
       CASE WHEN rm.ci_lo IS NULL AND rm.ci_hi IS NULL AND rm.metric_id IN ('asr', 'asr_held_in', 'asr_held_out', 'asr_at_n1', 'asr_at_n2', 'asr_at_n4', 'asr_at_n8', 'asr_at_n16', 'asr_at_n32', 'asr_at_n64', 'asr_at_n75', 'asr_bon_cap4096', 'asr_bon_cap8192', 'acc', 'xst_refused', 'orb_refused', 'at_cap_share_bon_512', 'coherent_at_cap_share_bon_512', 'loop_share_bon_512', 'secondary_flag_only_share_bon_512', 'residual_at_cap_share_bon_4096') AND rm.k IS NOT NULL AND rm.n > 0 AND abs(rm.value - rm.k*1.0/rm.n) <= 1e-9 THEN 'wilson_95' ELSE rm.ci_method END AS ci_method,
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
-- the CURRENT contrasts of the metric row (current_contrast, migration 0017): one row per anchor some head revision still pairs the run
-- with; a contrast only a superseded revision emitted is left out, so a re-anchored run no longer appears once per anchor
LEFT JOIN current_contrast rc ON rc.run_id = rm.run_id AND rc.judgment_id = rm.judgment_id AND rc.metric_id = rm.metric_id
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
