-- 0026_twin_fold_view_legacy_value_with_draws.sql
-- The `r2_export` <-> `sampled_lane` twins fold onto the pooled reading, and two pooled runs of one source at one key whose tags differ
-- coexist (a ruling of 2026-10-03: fold the duplicate (identity,
-- metric, threshold) pairs, keep every uniquely filed measurement). Numbered 0026 in this tree -- max landed + 1 (0025
-- `pending_excludes_superseded` is the newest on main), re-checked at the merge as the migrations README says; renumbered before its gate if
-- another branch's 0026 lands first (no Python post-step, so nothing is keyed by the slug). THREE views recreated, nothing else: no table is
-- rebuilt, no column is added, no row is copied.
--   * `v_multi_source_rows` projects the declaration's `check_rule` and the pooled draws `json_extract(rm.extra_json, '$.n_draws')` beside
--     `value`, `k`, `n`; the two twin declarations (`ownership.TWIN_OWNERSHIP`: the three models' `none` and Qwen's ReLU constitution steer
--     on `instruct_direct`, owner `sampled_lane`, secondary `r2_export`, paired by `view_legacy`, checked by `value_with_draws`) add their
--     scopes to the UNION.
--   * `v_multi_source_check` aggregates both and branches the state and the reason on the check rule: under `value_with_draws` the readings
--     agree when the values agree to 1e-9, the secondary's `n` equals the owner's `n_draws` and the secondary carries `k` (a pooled owner
--     files `k` NULL and `n` as goals); an owner row without draws never folds. `check_rule` and `n_draws_owner` are projected. Every
--     `counts` declaration reads exactly as before.
--   * `v_run_current`'s `later` CTE skips a pair of `sampled_lane` runs under `pooled_t0_t1` whose `view_legacy` differ
--     (`vocab.COEXIST_ACROSS_VIEWS_BY_SOURCE`, rendered by `schema_def.coexist_across_views_sql`): two tags' pooled readings at one key are
--     two points, never predecessor and successor; a later same-tag pooled run still supersedes; every other source keeps the later-run rule.
-- `v_paper_long` and `v_paper_current` read the check view and are unchanged in text, so they are not recreated: at each duplicate identity
-- the pooled row is the one paper row (`multi_source_state folded` on the export's), the export's unique identities stand (`stopgap`).
-- Rendered by `registry migrate --plan --sql` against a schema-version-25 database (0001..0025 applied to an empty database by the runner):
-- the planner reports three view recreates and nothing else, and tests/test_registry_twin_fold.py pins the plan -- any `rebuild` line is
-- a STOP (the 0020 rebuild of row_history wrote a 25 GB WAL). Expected live effect (the acceptance check): the set
-- of (run identity, metric, threshold) with a paper value unchanged; each duplicate identity goes from 2 paper rows to 1 (16 export runs
-- x 2 thresholds = 32 rows); zero identities with both sources' rows in the paper.
-- drift: v_run_current differs from the code's SQL
-- drift: v_multi_source_rows differs from the code's SQL
-- drift: v_multi_source_check differs from the code's SQL

DROP VIEW IF EXISTS v_multi_source_check;
CREATE VIEW v_multi_source_check AS
WITH scoped AS MATERIALIZED (SELECT * FROM v_multi_source_rows),
owner AS MATERIALIZED (
    SELECT declaration_id, pairing_key, eval_id, metric_id, threshold, COUNT(*) AS n_rows,
           MIN(owner_source) AS owner_source, MIN(secondary_source) AS secondary_source, MIN(model_id) AS model_id, MIN(paper_label) AS paper_label,
           MIN(eval_category) AS eval_category, MIN(transfer_kind) AS transfer_kind, MIN(producer_run_key_legacy) AS producer_run_key_legacy,
           MIN(run_id) AS run_id, MIN(judgment_id) AS judgment_id, MIN(computation_id) AS computation_id,
           MIN(check_rule) AS check_rule, MIN(n_draws) AS n_draws,
           MIN(value) AS value, MIN(k) AS k, MIN(n) AS n, MIN(degenerate_share) AS degenerate_share, MIN(loop_share) AS loop_share,
           MIN(loop_share_measure) AS loop_share_measure, MIN(at_cap_share) AS at_cap_share, MAX(ingest_seq) AS ingest_seq
    FROM scoped WHERE role = 'owner' GROUP BY declaration_id, pairing_key, eval_id, metric_id, threshold
),
secondary AS MATERIALIZED (
    SELECT declaration_id, pairing_key, eval_id, metric_id, threshold, COUNT(*) AS n_rows,
           MIN(owner_source) AS owner_source, MIN(secondary_source) AS secondary_source, MIN(model_id) AS model_id, MIN(paper_label) AS paper_label,
           MIN(eval_category) AS eval_category, MIN(transfer_kind) AS transfer_kind, MIN(producer_run_key_legacy) AS producer_run_key_legacy,
           MIN(run_id) AS run_id, MIN(judgment_id) AS judgment_id, MIN(computation_id) AS computation_id,
           MIN(check_rule) AS check_rule, MIN(n_draws) AS n_draws,
           MIN(value) AS value, MIN(k) AS k, MIN(n) AS n, MIN(degenerate_share) AS degenerate_share, MIN(loop_share) AS loop_share,
           MIN(loop_share_measure) AS loop_share_measure, MIN(at_cap_share) AS at_cap_share, MAX(ingest_seq) AS ingest_seq
    FROM scoped WHERE role = 'secondary' GROUP BY declaration_id, pairing_key, eval_id, metric_id, threshold
)
SELECT s.declaration_id, s.owner_source, s.secondary_source, s.model_id, s.paper_label, s.eval_category, s.transfer_kind,
       s.pairing_key, s.eval_id, s.metric_id, s.threshold,
       s.check_rule, ow.n_draws AS n_draws_owner,
       CASE WHEN ow.run_id IS NULL THEN 'stopgap'
     WHEN ow.n_rows > 1 OR s.n_rows > 1 THEN 'refused'
     WHEN s.check_rule = 'counts' AND (NOT (ow.k IS s.k) OR NOT (ow.n IS s.n)) THEN 'refused'
     WHEN s.check_rule = 'value_with_draws' AND (ow.value IS NULL OR s.value IS NULL OR ABS(ow.value - s.value) > 1e-9 OR ow.n_draws IS NULL OR NOT (s.n IS ow.n_draws) OR s.k IS NULL) THEN 'refused'
     WHEN (ow.degenerate_share IS NOT NULL AND s.degenerate_share IS NOT NULL AND ABS(ow.degenerate_share - s.degenerate_share) > 1e-9) THEN 'refused'
     WHEN (ow.loop_share IS NOT NULL AND s.loop_share IS NOT NULL AND ABS(ow.loop_share - s.loop_share) > 1e-9) THEN 'refused'
     ELSE 'folded' END AS state,
       CASE WHEN ow.run_id IS NULL THEN 'the owner (' || s.owner_source || ') carries no row for this measurement yet; the secondary (' || s.secondary_source || ') row stands as a stopgap until it does, and is then dropped once k and n agree exactly'
     WHEN ow.n_rows > 1 THEN 'the owner (' || s.owner_source || ') carries ' || ow.n_rows || ' rows at one identity; one owner row per measurement is the precondition of the fold, so neither value is admitted'
     WHEN s.n_rows > 1 THEN 'the secondary (' || s.secondary_source || ') carries ' || s.n_rows || ' rows at one identity; one row per measurement is the precondition of the fold, so neither value is admitted'
     WHEN s.check_rule = 'counts' AND (NOT (ow.k IS s.k) OR NOT (ow.n IS s.n)) THEN 'the two sources disagree on k / n (owner k=' || COALESCE(CAST(ow.k AS TEXT), 'null') || ' n=' || COALESCE(CAST(ow.n AS TEXT), 'null') || '; secondary k=' || COALESCE(CAST(s.k AS TEXT), 'null') || ' n=' || COALESCE(CAST(s.n AS TEXT), 'null') || '); two readings of one measurement that differ are a disagreement to rule on, never averaged and never picked from, so neither value is admitted'
     WHEN s.check_rule = 'value_with_draws' AND (ow.value IS NULL OR s.value IS NULL OR ABS(ow.value - s.value) > 1e-9 OR ow.n_draws IS NULL OR NOT (s.n IS ow.n_draws) OR s.k IS NULL) THEN 'the two readings disagree under value_with_draws (owner value=' || COALESCE(CAST(ow.value AS TEXT), 'null') || ' n_draws=' || COALESCE(CAST(ow.n_draws AS TEXT), 'null') || '; secondary value=' || COALESCE(CAST(s.value AS TEXT), 'null') || ' k=' || COALESCE(CAST(s.k AS TEXT), 'null') || ' n=' || COALESCE(CAST(s.n AS TEXT), 'null') || '): the values must agree to 1e-9, the secondary''s n must equal the owner''s pooled draws and the secondary must carry a count; two readings of one measurement that differ are a disagreement to rule on, never averaged and never picked from, so neither value is admitted'
     WHEN (ow.degenerate_share IS NOT NULL AND s.degenerate_share IS NOT NULL AND ABS(ow.degenerate_share - s.degenerate_share) > 1e-9) THEN 'the owner (' || s.owner_source || ') already carries degenerate_share=' || COALESCE(CAST(ow.degenerate_share AS TEXT), 'null') || ' and the secondary (' || s.secondary_source || ') reads ' || COALESCE(CAST(s.degenerate_share AS TEXT), 'null') || '; a disagreement is never overwritten and neither value is admitted'
     WHEN (ow.loop_share IS NOT NULL AND s.loop_share IS NOT NULL AND ABS(ow.loop_share - s.loop_share) > 1e-9) THEN 'the owner (' || s.owner_source || ') already carries loop_share=' || COALESCE(CAST(ow.loop_share AS TEXT), 'null') || ' and the secondary (' || s.secondary_source || ') reads ' || COALESCE(CAST(s.loop_share AS TEXT), 'null') || '; a disagreement is never overwritten and neither value is admitted'
     ELSE CASE WHEN s.check_rule = 'value_with_draws'
               THEN 'the value agrees to 1e-9 and the secondary''s n equals the owner''s pooled draws (value=' || COALESCE(CAST(ow.value AS TEXT), 'null') || ', n_draws=' || COALESCE(CAST(ow.n_draws AS TEXT), 'null') || '); the owner (' || s.owner_source || ') row stands and the secondary (' || s.secondary_source || ') row is dropped; carried onto the owner where it had none: ' || COALESCE(NULLIF(REPLACE(TRIM(CASE WHEN (ow.degenerate_share IS NULL AND s.degenerate_share IS NOT NULL) THEN '"degenerate_share",' ELSE '' END || CASE WHEN (ow.loop_share IS NULL AND s.loop_share IS NOT NULL) THEN '"loop_share",' ELSE '' END, ','), '"', ''), ''), 'nothing')
               ELSE 'k and n agree exactly (k=' || COALESCE(CAST(ow.k AS TEXT), 'null') || ', n=' || COALESCE(CAST(ow.n AS TEXT), 'null') || '); the owner (' || s.owner_source || ') row stands and the secondary (' || s.secondary_source || ') row is dropped; carried onto the owner where it had none: ' || COALESCE(NULLIF(REPLACE(TRIM(CASE WHEN (ow.degenerate_share IS NULL AND s.degenerate_share IS NOT NULL) THEN '"degenerate_share",' ELSE '' END || CASE WHEN (ow.loop_share IS NULL AND s.loop_share IS NOT NULL) THEN '"loop_share",' ELSE '' END, ','), '"', ''), ''), 'nothing') END END AS reason,
       ow.run_id AS owner_run_id, ow.judgment_id AS owner_judgment_id, ow.computation_id AS owner_computation_id, COALESCE(ow.n_rows, 0) AS owner_rows,
       ow.producer_run_key_legacy AS owner_producer_run_key_legacy,
       s.run_id AS secondary_run_id, s.judgment_id AS secondary_judgment_id, s.computation_id AS secondary_computation_id, s.n_rows AS secondary_rows,
       s.producer_run_key_legacy AS secondary_producer_run_key_legacy,
       ow.value AS value_owner, s.value AS value_secondary,
       ow.k AS k_owner, ow.n AS n_owner, s.k AS k_secondary, s.n AS n_secondary,
       ow.degenerate_share AS degenerate_share_owner, s.degenerate_share AS degenerate_share_secondary,
       ow.loop_share AS loop_share_owner, s.loop_share AS loop_share_secondary,
       ow.loop_share_measure AS loop_share_measure_owner, s.loop_share_measure AS loop_share_measure_secondary,
       ow.at_cap_share AS at_cap_share_owner, s.at_cap_share AS at_cap_share_secondary,
       CASE WHEN CASE WHEN ow.run_id IS NULL THEN 'stopgap'
     WHEN ow.n_rows > 1 OR s.n_rows > 1 THEN 'refused'
     WHEN s.check_rule = 'counts' AND (NOT (ow.k IS s.k) OR NOT (ow.n IS s.n)) THEN 'refused'
     WHEN s.check_rule = 'value_with_draws' AND (ow.value IS NULL OR s.value IS NULL OR ABS(ow.value - s.value) > 1e-9 OR ow.n_draws IS NULL OR NOT (s.n IS ow.n_draws) OR s.k IS NULL) THEN 'refused'
     WHEN (ow.degenerate_share IS NOT NULL AND s.degenerate_share IS NOT NULL AND ABS(ow.degenerate_share - s.degenerate_share) > 1e-9) THEN 'refused'
     WHEN (ow.loop_share IS NOT NULL AND s.loop_share IS NOT NULL AND ABS(ow.loop_share - s.loop_share) > 1e-9) THEN 'refused'
     ELSE 'folded' END = 'folded' THEN '[' || TRIM(CASE WHEN (ow.degenerate_share IS NULL AND s.degenerate_share IS NOT NULL) THEN '"degenerate_share",' ELSE '' END || CASE WHEN (ow.loop_share IS NULL AND s.loop_share IS NOT NULL) THEN '"loop_share",' ELSE '' END, ',') || ']' ELSE '[]' END AS carried_json,
       MAX(COALESCE(ow.ingest_seq, 0), s.ingest_seq) AS ingest_seq
FROM secondary s
LEFT JOIN owner ow ON ow.declaration_id = s.declaration_id AND ow.pairing_key = s.pairing_key AND ow.eval_id = s.eval_id
     AND ow.metric_id = s.metric_id AND ow.threshold = s.threshold;
DROP VIEW IF EXISTS v_multi_source_rows;
CREATE VIEW v_multi_source_rows AS
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
)
SELECT 'qwen35_4b_it/arditi_allpos/multi_turn' AS declaration_id, 'r2_export' AS owner_source, 'arditi_allpos' AS secondary_source,
       CASE WHEN r.lane_source_legacy = 'r2_export' THEN 'owner' ELSE 'secondary' END AS role,
       '' AS pairing_key,
       r.lane_source_legacy, r.run_id, r.model_id, o.paper_label, e.eval_category, e.headline_metric_id, r.eval_id,
       r.transfer_kind, r.producer_run_key_legacy, r.view_legacy, r.operating_point_id_legacy, r.strength_nominal,
       rm.judgment_id, rm.metric_id, rm.threshold, rm.computation_id, rm.value, rm.k, rm.n,
       'counts' AS check_rule,
       json_extract(rm.extra_json, '$.n_draws') AS n_draws,
       COALESCE(rm.frac_at_token_cap, dj.at_token_cap, dn.at_token_cap, dj.at_token_cap_finish_reason, dn.at_token_cap_finish_reason) AS at_cap_share,
       COALESCE(rm.frac_degenerate_union5, dj.union5, dn.union5, dj.loop_or_at_cap, dn.loop_or_at_cap, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share,
       CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END AS loop_share_measure,
       CASE CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END WHEN 'classifier' THEN COALESCE(rm.frac_degenerate_classifier, dj.classifier, dn.classifier) WHEN 'loop_first512' THEN COALESCE(rm.frac_loop_first512, dj.loop_first512, dn.loop_first512) ELSE COALESCE(rm.frac_degenerate_rep_ratio_loop, dj.rep_ratio_loop, dn.rep_ratio_loop) END AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN judgment jm ON jm.judgment_id = rm.judgment_id
LEFT JOIN instrument i ON i.instrument_id = r.instrument_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a non-current judgment is in no scope: ownership counts current rows only
  -- a row of a judgment that a later judgment displaced or a claim released is not current (migration 0021; NON_CURRENT_JUDGMENT_STATUSES):
  -- the new judgment stands in its place; a row with no judgment, or of an in-progress, partial or complete one, is unaffected
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  -- a partial judgment beside a complete one of the SAME run_id and judge_id is not current: the complete reading is the paper row
  -- (the displacing complete judgment itself identity-current under the run: a stale complete displaces nothing)
  AND (COALESCE(jm.status, '') <> 'partial' OR NOT EXISTS (SELECT 1 FROM judgment jcp WHERE jcp.run_id = jm.run_id AND jcp.judge_id = jm.judge_id AND jcp.status = 'complete' AND ((jcp.claimed_instrument_id IS NULL OR jcp.claimed_instrument_id IS r.instrument_id) AND (jcp.claimed_required_judge_id IS NULL OR jcp.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))))
  -- a judgment whose identity bound at claim differs from the current run instrument_id or required judge_id is not current
  -- (no apostrophe in this comment: the reference scan strips quoted literals);
  -- a NULL binding is current, so every legacy row and every additional reading by another judge_id stand
  AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
  AND r.model_id IN ('qwen35_4b_it') AND o.paper_label = 'arditi_allpos' AND e.eval_category = 'multi_turn'
    AND r.transfer_kind = 'instruct_direct'
    AND r.lane_source_legacy IN ('r2_export', 'arditi_allpos')
    AND (r.operating_point_id_legacy = 'ar_c0p3__arditi_allpos' OR (r.operating_point_id_legacy IS NULL AND r.strength_nominal = 1.0))
UNION ALL
SELECT 'llama8b_it/constitution_lora/petri/base_to_instruct' AS declaration_id, 'llama8b_lane' AS owner_source, 'r2_export' AS secondary_source,
       CASE WHEN r.lane_source_legacy = 'llama8b_lane' THEN 'owner' ELSE 'secondary' END AS role,
       r.producer_run_key_legacy AS pairing_key,
       r.lane_source_legacy, r.run_id, r.model_id, o.paper_label, e.eval_category, e.headline_metric_id, r.eval_id,
       r.transfer_kind, r.producer_run_key_legacy, r.view_legacy, r.operating_point_id_legacy, r.strength_nominal,
       rm.judgment_id, rm.metric_id, rm.threshold, rm.computation_id, rm.value, rm.k, rm.n,
       'counts' AS check_rule,
       json_extract(rm.extra_json, '$.n_draws') AS n_draws,
       COALESCE(rm.frac_at_token_cap, dj.at_token_cap, dn.at_token_cap, dj.at_token_cap_finish_reason, dn.at_token_cap_finish_reason) AS at_cap_share,
       COALESCE(rm.frac_degenerate_union5, dj.union5, dn.union5, dj.loop_or_at_cap, dn.loop_or_at_cap, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share,
       CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END AS loop_share_measure,
       CASE CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END WHEN 'classifier' THEN COALESCE(rm.frac_degenerate_classifier, dj.classifier, dn.classifier) WHEN 'loop_first512' THEN COALESCE(rm.frac_loop_first512, dj.loop_first512, dn.loop_first512) ELSE COALESCE(rm.frac_degenerate_rep_ratio_loop, dj.rep_ratio_loop, dn.rep_ratio_loop) END AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN judgment jm ON jm.judgment_id = rm.judgment_id
LEFT JOIN instrument i ON i.instrument_id = r.instrument_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a non-current judgment is in no scope: ownership counts current rows only
  -- a row of a judgment that a later judgment displaced or a claim released is not current (migration 0021; NON_CURRENT_JUDGMENT_STATUSES):
  -- the new judgment stands in its place; a row with no judgment, or of an in-progress, partial or complete one, is unaffected
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  -- a partial judgment beside a complete one of the SAME run_id and judge_id is not current: the complete reading is the paper row
  -- (the displacing complete judgment itself identity-current under the run: a stale complete displaces nothing)
  AND (COALESCE(jm.status, '') <> 'partial' OR NOT EXISTS (SELECT 1 FROM judgment jcp WHERE jcp.run_id = jm.run_id AND jcp.judge_id = jm.judge_id AND jcp.status = 'complete' AND ((jcp.claimed_instrument_id IS NULL OR jcp.claimed_instrument_id IS r.instrument_id) AND (jcp.claimed_required_judge_id IS NULL OR jcp.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))))
  -- a judgment whose identity bound at claim differs from the current run instrument_id or required judge_id is not current
  -- (no apostrophe in this comment: the reference scan strips quoted literals);
  -- a NULL binding is current, so every legacy row and every additional reading by another judge_id stand
  AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
  AND r.model_id IN ('llama8b_it') AND o.paper_label = 'constitution_lora' AND e.eval_category = 'petri'
    AND r.eval_id NOT IN ('petri_pooled')
    AND r.transfer_kind = 'base_to_instruct'
    AND r.lane_source_legacy IN ('llama8b_lane', 'r2_export')
    AND (r.operating_point_id_legacy = 'base__constv2-mix-full__lora_r4_ce' OR (r.operating_point_id_legacy IS NULL AND r.strength_nominal = 1.0))
UNION ALL
SELECT 'llama8b_it/constitution_lora/petri/instruct_direct' AS declaration_id, 'llama8b_lane' AS owner_source, 'r2_export' AS secondary_source,
       CASE WHEN r.lane_source_legacy = 'llama8b_lane' THEN 'owner' ELSE 'secondary' END AS role,
       r.producer_run_key_legacy AS pairing_key,
       r.lane_source_legacy, r.run_id, r.model_id, o.paper_label, e.eval_category, e.headline_metric_id, r.eval_id,
       r.transfer_kind, r.producer_run_key_legacy, r.view_legacy, r.operating_point_id_legacy, r.strength_nominal,
       rm.judgment_id, rm.metric_id, rm.threshold, rm.computation_id, rm.value, rm.k, rm.n,
       'counts' AS check_rule,
       json_extract(rm.extra_json, '$.n_draws') AS n_draws,
       COALESCE(rm.frac_at_token_cap, dj.at_token_cap, dn.at_token_cap, dj.at_token_cap_finish_reason, dn.at_token_cap_finish_reason) AS at_cap_share,
       COALESCE(rm.frac_degenerate_union5, dj.union5, dn.union5, dj.loop_or_at_cap, dn.loop_or_at_cap, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share,
       CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END AS loop_share_measure,
       CASE CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END WHEN 'classifier' THEN COALESCE(rm.frac_degenerate_classifier, dj.classifier, dn.classifier) WHEN 'loop_first512' THEN COALESCE(rm.frac_loop_first512, dj.loop_first512, dn.loop_first512) ELSE COALESCE(rm.frac_degenerate_rep_ratio_loop, dj.rep_ratio_loop, dn.rep_ratio_loop) END AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN judgment jm ON jm.judgment_id = rm.judgment_id
LEFT JOIN instrument i ON i.instrument_id = r.instrument_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a non-current judgment is in no scope: ownership counts current rows only
  -- a row of a judgment that a later judgment displaced or a claim released is not current (migration 0021; NON_CURRENT_JUDGMENT_STATUSES):
  -- the new judgment stands in its place; a row with no judgment, or of an in-progress, partial or complete one, is unaffected
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  -- a partial judgment beside a complete one of the SAME run_id and judge_id is not current: the complete reading is the paper row
  -- (the displacing complete judgment itself identity-current under the run: a stale complete displaces nothing)
  AND (COALESCE(jm.status, '') <> 'partial' OR NOT EXISTS (SELECT 1 FROM judgment jcp WHERE jcp.run_id = jm.run_id AND jcp.judge_id = jm.judge_id AND jcp.status = 'complete' AND ((jcp.claimed_instrument_id IS NULL OR jcp.claimed_instrument_id IS r.instrument_id) AND (jcp.claimed_required_judge_id IS NULL OR jcp.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))))
  -- a judgment whose identity bound at claim differs from the current run instrument_id or required judge_id is not current
  -- (no apostrophe in this comment: the reference scan strips quoted literals);
  -- a NULL binding is current, so every legacy row and every additional reading by another judge_id stand
  AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
  AND r.model_id IN ('llama8b_it') AND o.paper_label = 'constitution_lora' AND e.eval_category = 'petri'
    AND r.eval_id NOT IN ('petri_pooled')
    AND r.transfer_kind = 'instruct_direct'
    AND r.lane_source_legacy IN ('llama8b_lane', 'r2_export')
    AND (r.operating_point_id_legacy = 'itdoc__constv2-mix-full__lora_r4_ce' OR (r.operating_point_id_legacy IS NULL AND r.strength_nominal = 1.0))
UNION ALL
SELECT 'llama8b_it/control_lora/petri/base_to_instruct' AS declaration_id, 'llama8b_lane' AS owner_source, 'r2_export' AS secondary_source,
       CASE WHEN r.lane_source_legacy = 'llama8b_lane' THEN 'owner' ELSE 'secondary' END AS role,
       r.producer_run_key_legacy AS pairing_key,
       r.lane_source_legacy, r.run_id, r.model_id, o.paper_label, e.eval_category, e.headline_metric_id, r.eval_id,
       r.transfer_kind, r.producer_run_key_legacy, r.view_legacy, r.operating_point_id_legacy, r.strength_nominal,
       rm.judgment_id, rm.metric_id, rm.threshold, rm.computation_id, rm.value, rm.k, rm.n,
       'counts' AS check_rule,
       json_extract(rm.extra_json, '$.n_draws') AS n_draws,
       COALESCE(rm.frac_at_token_cap, dj.at_token_cap, dn.at_token_cap, dj.at_token_cap_finish_reason, dn.at_token_cap_finish_reason) AS at_cap_share,
       COALESCE(rm.frac_degenerate_union5, dj.union5, dn.union5, dj.loop_or_at_cap, dn.loop_or_at_cap, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share,
       CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END AS loop_share_measure,
       CASE CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END WHEN 'classifier' THEN COALESCE(rm.frac_degenerate_classifier, dj.classifier, dn.classifier) WHEN 'loop_first512' THEN COALESCE(rm.frac_loop_first512, dj.loop_first512, dn.loop_first512) ELSE COALESCE(rm.frac_degenerate_rep_ratio_loop, dj.rep_ratio_loop, dn.rep_ratio_loop) END AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN judgment jm ON jm.judgment_id = rm.judgment_id
LEFT JOIN instrument i ON i.instrument_id = r.instrument_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a non-current judgment is in no scope: ownership counts current rows only
  -- a row of a judgment that a later judgment displaced or a claim released is not current (migration 0021; NON_CURRENT_JUDGMENT_STATUSES):
  -- the new judgment stands in its place; a row with no judgment, or of an in-progress, partial or complete one, is unaffected
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  -- a partial judgment beside a complete one of the SAME run_id and judge_id is not current: the complete reading is the paper row
  -- (the displacing complete judgment itself identity-current under the run: a stale complete displaces nothing)
  AND (COALESCE(jm.status, '') <> 'partial' OR NOT EXISTS (SELECT 1 FROM judgment jcp WHERE jcp.run_id = jm.run_id AND jcp.judge_id = jm.judge_id AND jcp.status = 'complete' AND ((jcp.claimed_instrument_id IS NULL OR jcp.claimed_instrument_id IS r.instrument_id) AND (jcp.claimed_required_judge_id IS NULL OR jcp.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))))
  -- a judgment whose identity bound at claim differs from the current run instrument_id or required judge_id is not current
  -- (no apostrophe in this comment: the reference scan strips quoted literals);
  -- a NULL binding is current, so every legacy row and every additional reading by another judge_id stand
  AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
  AND r.model_id IN ('llama8b_it') AND o.paper_label = 'control_lora' AND e.eval_category = 'petri'
    AND r.eval_id NOT IN ('petri_pooled')
    AND r.transfer_kind = 'base_to_instruct'
    AND r.lane_source_legacy IN ('llama8b_lane', 'r2_export')
    AND (r.operating_point_id_legacy = 'base__constv2-mix-full__lora_r4_ce' OR (r.operating_point_id_legacy IS NULL AND r.strength_nominal = 1.0))
UNION ALL
SELECT 'llama8b_it/control_lora/petri/instruct_direct' AS declaration_id, 'llama8b_lane' AS owner_source, 'r2_export' AS secondary_source,
       CASE WHEN r.lane_source_legacy = 'llama8b_lane' THEN 'owner' ELSE 'secondary' END AS role,
       r.producer_run_key_legacy AS pairing_key,
       r.lane_source_legacy, r.run_id, r.model_id, o.paper_label, e.eval_category, e.headline_metric_id, r.eval_id,
       r.transfer_kind, r.producer_run_key_legacy, r.view_legacy, r.operating_point_id_legacy, r.strength_nominal,
       rm.judgment_id, rm.metric_id, rm.threshold, rm.computation_id, rm.value, rm.k, rm.n,
       'counts' AS check_rule,
       json_extract(rm.extra_json, '$.n_draws') AS n_draws,
       COALESCE(rm.frac_at_token_cap, dj.at_token_cap, dn.at_token_cap, dj.at_token_cap_finish_reason, dn.at_token_cap_finish_reason) AS at_cap_share,
       COALESCE(rm.frac_degenerate_union5, dj.union5, dn.union5, dj.loop_or_at_cap, dn.loop_or_at_cap, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share,
       CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END AS loop_share_measure,
       CASE CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END WHEN 'classifier' THEN COALESCE(rm.frac_degenerate_classifier, dj.classifier, dn.classifier) WHEN 'loop_first512' THEN COALESCE(rm.frac_loop_first512, dj.loop_first512, dn.loop_first512) ELSE COALESCE(rm.frac_degenerate_rep_ratio_loop, dj.rep_ratio_loop, dn.rep_ratio_loop) END AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN judgment jm ON jm.judgment_id = rm.judgment_id
LEFT JOIN instrument i ON i.instrument_id = r.instrument_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a non-current judgment is in no scope: ownership counts current rows only
  -- a row of a judgment that a later judgment displaced or a claim released is not current (migration 0021; NON_CURRENT_JUDGMENT_STATUSES):
  -- the new judgment stands in its place; a row with no judgment, or of an in-progress, partial or complete one, is unaffected
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  -- a partial judgment beside a complete one of the SAME run_id and judge_id is not current: the complete reading is the paper row
  -- (the displacing complete judgment itself identity-current under the run: a stale complete displaces nothing)
  AND (COALESCE(jm.status, '') <> 'partial' OR NOT EXISTS (SELECT 1 FROM judgment jcp WHERE jcp.run_id = jm.run_id AND jcp.judge_id = jm.judge_id AND jcp.status = 'complete' AND ((jcp.claimed_instrument_id IS NULL OR jcp.claimed_instrument_id IS r.instrument_id) AND (jcp.claimed_required_judge_id IS NULL OR jcp.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))))
  -- a judgment whose identity bound at claim differs from the current run instrument_id or required judge_id is not current
  -- (no apostrophe in this comment: the reference scan strips quoted literals);
  -- a NULL binding is current, so every legacy row and every additional reading by another judge_id stand
  AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
  AND r.model_id IN ('llama8b_it') AND o.paper_label = 'control_lora' AND e.eval_category = 'petri'
    AND r.eval_id NOT IN ('petri_pooled')
    AND r.transfer_kind = 'instruct_direct'
    AND r.lane_source_legacy IN ('llama8b_lane', 'r2_export')
    AND (r.operating_point_id_legacy = 'itdoc__constv2-mix-full__lora_r4_ce' OR (r.operating_point_id_legacy IS NULL AND r.strength_nominal = 1.0))
UNION ALL
SELECT 'llama8b_it/difference_lora/petri/base_to_instruct' AS declaration_id, 'llama8b_lane' AS owner_source, 'r2_export' AS secondary_source,
       CASE WHEN r.lane_source_legacy = 'llama8b_lane' THEN 'owner' ELSE 'secondary' END AS role,
       r.producer_run_key_legacy AS pairing_key,
       r.lane_source_legacy, r.run_id, r.model_id, o.paper_label, e.eval_category, e.headline_metric_id, r.eval_id,
       r.transfer_kind, r.producer_run_key_legacy, r.view_legacy, r.operating_point_id_legacy, r.strength_nominal,
       rm.judgment_id, rm.metric_id, rm.threshold, rm.computation_id, rm.value, rm.k, rm.n,
       'counts' AS check_rule,
       json_extract(rm.extra_json, '$.n_draws') AS n_draws,
       COALESCE(rm.frac_at_token_cap, dj.at_token_cap, dn.at_token_cap, dj.at_token_cap_finish_reason, dn.at_token_cap_finish_reason) AS at_cap_share,
       COALESCE(rm.frac_degenerate_union5, dj.union5, dn.union5, dj.loop_or_at_cap, dn.loop_or_at_cap, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share,
       CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END AS loop_share_measure,
       CASE CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END WHEN 'classifier' THEN COALESCE(rm.frac_degenerate_classifier, dj.classifier, dn.classifier) WHEN 'loop_first512' THEN COALESCE(rm.frac_loop_first512, dj.loop_first512, dn.loop_first512) ELSE COALESCE(rm.frac_degenerate_rep_ratio_loop, dj.rep_ratio_loop, dn.rep_ratio_loop) END AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN judgment jm ON jm.judgment_id = rm.judgment_id
LEFT JOIN instrument i ON i.instrument_id = r.instrument_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a non-current judgment is in no scope: ownership counts current rows only
  -- a row of a judgment that a later judgment displaced or a claim released is not current (migration 0021; NON_CURRENT_JUDGMENT_STATUSES):
  -- the new judgment stands in its place; a row with no judgment, or of an in-progress, partial or complete one, is unaffected
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  -- a partial judgment beside a complete one of the SAME run_id and judge_id is not current: the complete reading is the paper row
  -- (the displacing complete judgment itself identity-current under the run: a stale complete displaces nothing)
  AND (COALESCE(jm.status, '') <> 'partial' OR NOT EXISTS (SELECT 1 FROM judgment jcp WHERE jcp.run_id = jm.run_id AND jcp.judge_id = jm.judge_id AND jcp.status = 'complete' AND ((jcp.claimed_instrument_id IS NULL OR jcp.claimed_instrument_id IS r.instrument_id) AND (jcp.claimed_required_judge_id IS NULL OR jcp.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))))
  -- a judgment whose identity bound at claim differs from the current run instrument_id or required judge_id is not current
  -- (no apostrophe in this comment: the reference scan strips quoted literals);
  -- a NULL binding is current, so every legacy row and every additional reading by another judge_id stand
  AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
  AND r.model_id IN ('llama8b_it') AND o.paper_label = 'difference_lora' AND e.eval_category = 'petri'
    AND r.eval_id NOT IN ('petri_pooled')
    AND r.transfer_kind = 'base_to_instruct'
    AND r.lane_source_legacy IN ('llama8b_lane', 'r2_export')
    AND (r.operating_point_id_legacy = 'base__constv2-mix-full__lora_r4_ce' OR (r.operating_point_id_legacy IS NULL AND r.strength_nominal = 1.0))
UNION ALL
SELECT 'llama8b_it/difference_lora/petri/instruct_direct' AS declaration_id, 'llama8b_lane' AS owner_source, 'r2_export' AS secondary_source,
       CASE WHEN r.lane_source_legacy = 'llama8b_lane' THEN 'owner' ELSE 'secondary' END AS role,
       r.producer_run_key_legacy AS pairing_key,
       r.lane_source_legacy, r.run_id, r.model_id, o.paper_label, e.eval_category, e.headline_metric_id, r.eval_id,
       r.transfer_kind, r.producer_run_key_legacy, r.view_legacy, r.operating_point_id_legacy, r.strength_nominal,
       rm.judgment_id, rm.metric_id, rm.threshold, rm.computation_id, rm.value, rm.k, rm.n,
       'counts' AS check_rule,
       json_extract(rm.extra_json, '$.n_draws') AS n_draws,
       COALESCE(rm.frac_at_token_cap, dj.at_token_cap, dn.at_token_cap, dj.at_token_cap_finish_reason, dn.at_token_cap_finish_reason) AS at_cap_share,
       COALESCE(rm.frac_degenerate_union5, dj.union5, dn.union5, dj.loop_or_at_cap, dn.loop_or_at_cap, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share,
       CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END AS loop_share_measure,
       CASE CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END WHEN 'classifier' THEN COALESCE(rm.frac_degenerate_classifier, dj.classifier, dn.classifier) WHEN 'loop_first512' THEN COALESCE(rm.frac_loop_first512, dj.loop_first512, dn.loop_first512) ELSE COALESCE(rm.frac_degenerate_rep_ratio_loop, dj.rep_ratio_loop, dn.rep_ratio_loop) END AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN judgment jm ON jm.judgment_id = rm.judgment_id
LEFT JOIN instrument i ON i.instrument_id = r.instrument_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a non-current judgment is in no scope: ownership counts current rows only
  -- a row of a judgment that a later judgment displaced or a claim released is not current (migration 0021; NON_CURRENT_JUDGMENT_STATUSES):
  -- the new judgment stands in its place; a row with no judgment, or of an in-progress, partial or complete one, is unaffected
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  -- a partial judgment beside a complete one of the SAME run_id and judge_id is not current: the complete reading is the paper row
  -- (the displacing complete judgment itself identity-current under the run: a stale complete displaces nothing)
  AND (COALESCE(jm.status, '') <> 'partial' OR NOT EXISTS (SELECT 1 FROM judgment jcp WHERE jcp.run_id = jm.run_id AND jcp.judge_id = jm.judge_id AND jcp.status = 'complete' AND ((jcp.claimed_instrument_id IS NULL OR jcp.claimed_instrument_id IS r.instrument_id) AND (jcp.claimed_required_judge_id IS NULL OR jcp.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))))
  -- a judgment whose identity bound at claim differs from the current run instrument_id or required judge_id is not current
  -- (no apostrophe in this comment: the reference scan strips quoted literals);
  -- a NULL binding is current, so every legacy row and every additional reading by another judge_id stand
  AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
  AND r.model_id IN ('llama8b_it') AND o.paper_label = 'difference_lora' AND e.eval_category = 'petri'
    AND r.eval_id NOT IN ('petri_pooled')
    AND r.transfer_kind = 'instruct_direct'
    AND r.lane_source_legacy IN ('llama8b_lane', 'r2_export')
    AND (r.operating_point_id_legacy = 'itdoc__constv2-mix-full__lora_r4_ce' OR (r.operating_point_id_legacy IS NULL AND r.strength_nominal = 1.0))
UNION ALL
SELECT 'sampled_static_8192/none/static_attack/none' AS declaration_id, 'sampled_lane' AS owner_source, 'r2_export' AS secondary_source,
       CASE WHEN r.lane_source_legacy = 'sampled_lane' THEN 'owner' ELSE 'secondary' END AS role,
       (COALESCE(r.view_legacy, '') || '|' || r.instrument_id) AS pairing_key,
       r.lane_source_legacy, r.run_id, r.model_id, o.paper_label, e.eval_category, e.headline_metric_id, r.eval_id,
       r.transfer_kind, r.producer_run_key_legacy, r.view_legacy, r.operating_point_id_legacy, r.strength_nominal,
       rm.judgment_id, rm.metric_id, rm.threshold, rm.computation_id, rm.value, rm.k, rm.n,
       'value_with_draws' AS check_rule,
       json_extract(rm.extra_json, '$.n_draws') AS n_draws,
       COALESCE(rm.frac_at_token_cap, dj.at_token_cap, dn.at_token_cap, dj.at_token_cap_finish_reason, dn.at_token_cap_finish_reason) AS at_cap_share,
       COALESCE(rm.frac_degenerate_union5, dj.union5, dn.union5, dj.loop_or_at_cap, dn.loop_or_at_cap, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share,
       CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END AS loop_share_measure,
       CASE CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END WHEN 'classifier' THEN COALESCE(rm.frac_degenerate_classifier, dj.classifier, dn.classifier) WHEN 'loop_first512' THEN COALESCE(rm.frac_loop_first512, dj.loop_first512, dn.loop_first512) ELSE COALESCE(rm.frac_degenerate_rep_ratio_loop, dj.rep_ratio_loop, dn.rep_ratio_loop) END AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN judgment jm ON jm.judgment_id = rm.judgment_id
LEFT JOIN instrument i ON i.instrument_id = r.instrument_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a non-current judgment is in no scope: ownership counts current rows only
  -- a row of a judgment that a later judgment displaced or a claim released is not current (migration 0021; NON_CURRENT_JUDGMENT_STATUSES):
  -- the new judgment stands in its place; a row with no judgment, or of an in-progress, partial or complete one, is unaffected
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  -- a partial judgment beside a complete one of the SAME run_id and judge_id is not current: the complete reading is the paper row
  -- (the displacing complete judgment itself identity-current under the run: a stale complete displaces nothing)
  AND (COALESCE(jm.status, '') <> 'partial' OR NOT EXISTS (SELECT 1 FROM judgment jcp WHERE jcp.run_id = jm.run_id AND jcp.judge_id = jm.judge_id AND jcp.status = 'complete' AND ((jcp.claimed_instrument_id IS NULL OR jcp.claimed_instrument_id IS r.instrument_id) AND (jcp.claimed_required_judge_id IS NULL OR jcp.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))))
  -- a judgment whose identity bound at claim differs from the current run instrument_id or required judge_id is not current
  -- (no apostrophe in this comment: the reference scan strips quoted literals);
  -- a NULL binding is current, so every legacy row and every additional reading by another judge_id stand
  AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
  AND r.model_id IN ('gemma4e4b_it', 'qwen35_4b_it', 'llama8b_it') AND o.paper_label = 'none' AND e.eval_category = 'static_attack'
    AND r.transfer_kind = 'none'
    AND r.lane_source_legacy IN ('sampled_lane', 'r2_export')
    AND (r.operating_point_id_legacy IS NULL AND r.strength_nominal IS NULL)
    AND r.token_cap_tokens = 8192
    AND (r.lane_source_legacy = 'r2_export' OR r.pass_label IN ('pooled_t0_t1'))
UNION ALL
SELECT 'sampled_static_8192/constitution_relu/static_attack/instruct_direct' AS declaration_id, 'sampled_lane' AS owner_source, 'r2_export' AS secondary_source,
       CASE WHEN r.lane_source_legacy = 'sampled_lane' THEN 'owner' ELSE 'secondary' END AS role,
       (COALESCE(r.view_legacy, '') || '|' || r.instrument_id) AS pairing_key,
       r.lane_source_legacy, r.run_id, r.model_id, o.paper_label, e.eval_category, e.headline_metric_id, r.eval_id,
       r.transfer_kind, r.producer_run_key_legacy, r.view_legacy, r.operating_point_id_legacy, r.strength_nominal,
       rm.judgment_id, rm.metric_id, rm.threshold, rm.computation_id, rm.value, rm.k, rm.n,
       'value_with_draws' AS check_rule,
       json_extract(rm.extra_json, '$.n_draws') AS n_draws,
       COALESCE(rm.frac_at_token_cap, dj.at_token_cap, dn.at_token_cap, dj.at_token_cap_finish_reason, dn.at_token_cap_finish_reason) AS at_cap_share,
       COALESCE(rm.frac_degenerate_union5, dj.union5, dn.union5, dj.loop_or_at_cap, dn.loop_or_at_cap, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share,
       CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END AS loop_share_measure,
       CASE CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END WHEN 'classifier' THEN COALESCE(rm.frac_degenerate_classifier, dj.classifier, dn.classifier) WHEN 'loop_first512' THEN COALESCE(rm.frac_loop_first512, dj.loop_first512, dn.loop_first512) ELSE COALESCE(rm.frac_degenerate_rep_ratio_loop, dj.rep_ratio_loop, dn.rep_ratio_loop) END AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN judgment jm ON jm.judgment_id = rm.judgment_id
LEFT JOIN instrument i ON i.instrument_id = r.instrument_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a non-current judgment is in no scope: ownership counts current rows only
  -- a row of a judgment that a later judgment displaced or a claim released is not current (migration 0021; NON_CURRENT_JUDGMENT_STATUSES):
  -- the new judgment stands in its place; a row with no judgment, or of an in-progress, partial or complete one, is unaffected
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  -- a partial judgment beside a complete one of the SAME run_id and judge_id is not current: the complete reading is the paper row
  -- (the displacing complete judgment itself identity-current under the run: a stale complete displaces nothing)
  AND (COALESCE(jm.status, '') <> 'partial' OR NOT EXISTS (SELECT 1 FROM judgment jcp WHERE jcp.run_id = jm.run_id AND jcp.judge_id = jm.judge_id AND jcp.status = 'complete' AND ((jcp.claimed_instrument_id IS NULL OR jcp.claimed_instrument_id IS r.instrument_id) AND (jcp.claimed_required_judge_id IS NULL OR jcp.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))))
  -- a judgment whose identity bound at claim differs from the current run instrument_id or required judge_id is not current
  -- (no apostrophe in this comment: the reference scan strips quoted literals);
  -- a NULL binding is current, so every legacy row and every additional reading by another judge_id stand
  AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
  AND r.model_id IN ('qwen35_4b_it') AND o.paper_label = 'constitution_relu' AND e.eval_category = 'static_attack'
    AND r.transfer_kind = 'instruct_direct'
    AND r.lane_source_legacy IN ('sampled_lane', 'r2_export')
    AND (r.operating_point_id_legacy IS NULL AND ABS(r.strength_nominal - 1.0) <= 1e-9)
    AND r.token_cap_tokens = 8192
    AND (r.lane_source_legacy = 'r2_export' OR r.pass_label IN ('pooled_t0_t1'));
DROP VIEW IF EXISTS v_run_current;
CREATE VIEW v_run_current AS
WITH open_round AS (
    SELECT round_id FROM paper_round WHERE closed_utc IS NULL ORDER BY opened_utc DESC LIMIT 1
),
matched_raw AS MATERIALIZED (
    SELECT run_id, roster_entry_id, comparison_id FROM v_roster_match
),
active AS (
    SELECT ru.* FROM ruling ru JOIN open_round orr ON orr.round_id = ru.round_id WHERE ru.expired_utc IS NULL
),
match_terms AS MATERIALIZED (
    -- the match lists flattened once: (ruling, field, term); the flag then joins runs to terms instead of parsing JSON per (run, ruling)
    SELECT ru.ruling_id, k.key AS field, v.value AS term
    FROM active ru, json_each(ru.scope_json, '$.match') k, json_each(k.value) v
    WHERE json_type(ru.scope_json, '$.match') = 'object' AND k.type = 'array' AND k.key <> 'instrument_alternatives'
),
match_fields AS MATERIALIZED (
    SELECT ruling_id, field FROM match_terms GROUP BY ruling_id, field
),
match_shape AS MATERIALIZED (
    SELECT ru.ruling_id, (SELECT COUNT(*) FROM json_each(ru.scope_json, '$.match')) AS n_keys,
           json_type(ru.scope_json, '$.match.instrument_alternatives') AS has_alternatives
    FROM active ru WHERE json_type(ru.scope_json, '$.match') = 'object'
),
covered AS MATERIALIZED (
    -- the match form: every list the ruling carries must contain the value of the run (an unknown field never matches; a ruling with
    -- no list covers nothing); the alternatives are OR-ed
    SELECT r.run_id, ru.ruling_id, ru.action, ru.scope_kind, ru.evidence_state, ru.ruled_utc
    FROM run r
    JOIN object o ON o.object_id = r.object_id
    JOIN eval e ON e.eval_id = r.eval_id
    JOIN match_shape ms ON ms.n_keys > 0
    JOIN active ru ON ru.ruling_id = ms.ruling_id
    WHERE (SELECT COUNT(*) FROM match_fields f WHERE f.ruling_id = ru.ruling_id)
        = (SELECT COUNT(*) FROM match_fields f WHERE f.ruling_id = ru.ruling_id
             AND EXISTS (SELECT 1 FROM match_terms t WHERE t.ruling_id = f.ruling_id AND t.field = f.field
                         AND CASE f.field
                               WHEN 'run_ids' THEN t.term = r.run_id
                               WHEN 'model_ids' THEN t.term = r.model_id
                               WHEN 'object_ids' THEN t.term = r.object_id
                               WHEN 'object_id_prefixes' THEN r.object_id GLOB t.term || '*'
                               WHEN 'object_kinds' THEN t.term = o.object_kind
                               WHEN 'paper_labels' THEN t.term = o.paper_label
                               WHEN 'object_roles' THEN t.term = o.object_role
                               WHEN 'eval_ids' THEN t.term = r.eval_id
                               WHEN 'eval_categories' THEN t.term = e.eval_category
                               WHEN 'transfer_kinds' THEN t.term = r.transfer_kind
                               WHEN 'strength_nominals' THEN t.term = r.strength_nominal
                               WHEN 'operating_point_ids_legacy' THEN t.term = r.operating_point_id_legacy
                               WHEN 'lane_sources_legacy' THEN t.term = r.lane_source_legacy
                               WHEN 'views_legacy' THEN t.term = r.view_legacy
                               WHEN 'campaigns' THEN t.term = r.campaign
                               WHEN 'pass_labels' THEN t.term = r.pass_label
                               WHEN 'instrument_ids' THEN t.term = r.instrument_id
                               WHEN 'token_caps' THEN t.term = r.token_cap_tokens
                               WHEN 'decode_sources' THEN t.term = r.decode_source
                               WHEN 'thinking_modes' THEN t.term = r.thinking_mode
                               WHEN 'harnesses' THEN t.term = r.harness
                               WHEN 'item_set_ids' THEN t.term = r.item_set_id
                               WHEN 'n_shots' THEN t.term = r.n_shot
                               WHEN 'judge_ids' THEN t.term = r.judge_id
                               WHEN 'comparison_ids' THEN EXISTS (SELECT 1 FROM matched_raw m WHERE m.run_id = r.run_id AND m.comparison_id GLOB t.term)
                               ELSE 0
                             END))
      AND (ms.has_alternatives IS NULL OR EXISTS (SELECT 1 FROM json_each(ru.scope_json, '$.match.instrument_alternatives') alt WHERE (json_type(alt.value, '$.token_caps') IS NULL OR EXISTS (SELECT 1 FROM json_each(alt.value, '$.token_caps') w WHERE w.value = r.token_cap_tokens)) AND (json_type(alt.value, '$.decode_sources') IS NULL OR EXISTS (SELECT 1 FROM json_each(alt.value, '$.decode_sources') w WHERE w.value = r.decode_source)) AND (json_type(alt.value, '$.thinking_modes') IS NULL OR EXISTS (SELECT 1 FROM json_each(alt.value, '$.thinking_modes') w WHERE w.value = r.thinking_mode)) AND (json_type(alt.value, '$.harnesses') IS NULL OR EXISTS (SELECT 1 FROM json_each(alt.value, '$.harnesses') w WHERE w.value = r.harness)) AND (json_type(alt.value, '$.item_set_ids') IS NULL OR EXISTS (SELECT 1 FROM json_each(alt.value, '$.item_set_ids') w WHERE w.value = r.item_set_id)) AND (json_type(alt.value, '$.n_shots') IS NULL OR EXISTS (SELECT 1 FROM json_each(alt.value, '$.n_shots') w WHERE w.value = r.n_shot)) AND (json_type(alt.value, '$.scoring_rules') IS NULL OR EXISTS (SELECT 1 FROM json_each(alt.value, '$.scoring_rules') w WHERE w.value = r.scoring_rule)) AND (json_type(alt.value, '$.serving_paths') IS NULL OR EXISTS (SELECT 1 FROM json_each(alt.value, '$.serving_paths') w WHERE w.value = r.serving_path))))
    UNION ALL
    -- rulings written before the match form (no `match` key at all): the key of the scope_kind itself
    SELECT r.run_id, ru.ruling_id, ru.action, ru.scope_kind, ru.evidence_state, ru.ruled_utc
    FROM run r
    JOIN object o ON o.object_id = r.object_id
    JOIN active ru ON json_type(ru.scope_json, '$.match') IS NULL
    WHERE (ru.scope_kind = 'run' AND json_extract(ru.scope_json, '$.run_id') = r.run_id)
       OR (ru.scope_kind = 'object'
           AND (json_extract(ru.scope_json, '$.object_id') IS NOT NULL OR json_extract(ru.scope_json, '$.object_kind') IS NOT NULL
                OR json_extract(ru.scope_json, '$.paper_label') IS NOT NULL)
           AND (json_extract(ru.scope_json, '$.object_id') IS NULL OR json_extract(ru.scope_json, '$.object_id') = r.object_id)
           AND (json_extract(ru.scope_json, '$.object_kind') IS NULL OR json_extract(ru.scope_json, '$.object_kind') = o.object_kind)
           AND (json_extract(ru.scope_json, '$.paper_label') IS NULL OR json_extract(ru.scope_json, '$.paper_label') = o.paper_label)
           AND (json_extract(ru.scope_json, '$.model_id') IS NULL OR json_extract(ru.scope_json, '$.model_id') = r.model_id))
       OR (ru.scope_kind = 'model' AND json_extract(ru.scope_json, '$.model_id') = r.model_id)
       OR (ru.scope_kind = 'eval' AND json_extract(ru.scope_json, '$.eval_id') = r.eval_id)
       OR (ru.scope_kind = 'comparison' AND EXISTS (SELECT 1 FROM matched_raw m WHERE m.run_id = r.run_id
               AND (m.comparison_id = json_extract(ru.scope_json, '$.comparison_id')
                    OR EXISTS (SELECT 1 FROM json_each(ru.scope_json, '$.comparison_ids') v WHERE m.comparison_id GLOB v.value))))
       OR (ru.scope_kind = 'key' AND (json_extract(ru.scope_json, '$.model_id') IS NOT NULL OR json_extract(ru.scope_json, '$.object_id') IS NOT NULL OR json_extract(ru.scope_json, '$.eval_id') IS NOT NULL OR json_extract(ru.scope_json, '$.instrument_id') IS NOT NULL OR json_extract(ru.scope_json, '$.campaign') IS NOT NULL OR json_extract(ru.scope_json, '$.operating_point_id_legacy') IS NOT NULL OR json_extract(ru.scope_json, '$.token_cap_tokens') IS NOT NULL) AND (json_extract(ru.scope_json, '$.model_id') IS NULL OR json_extract(ru.scope_json, '$.model_id') = r.model_id) AND (json_extract(ru.scope_json, '$.object_id') IS NULL OR json_extract(ru.scope_json, '$.object_id') = r.object_id) AND (json_extract(ru.scope_json, '$.eval_id') IS NULL OR json_extract(ru.scope_json, '$.eval_id') = r.eval_id) AND (json_extract(ru.scope_json, '$.instrument_id') IS NULL OR json_extract(ru.scope_json, '$.instrument_id') = r.instrument_id) AND (json_extract(ru.scope_json, '$.campaign') IS NULL OR json_extract(ru.scope_json, '$.campaign') = r.campaign) AND (json_extract(ru.scope_json, '$.operating_point_id_legacy') IS NULL OR json_extract(ru.scope_json, '$.operating_point_id_legacy') = r.operating_point_id_legacy) AND (json_extract(ru.scope_json, '$.token_cap_tokens') IS NULL OR json_extract(ru.scope_json, '$.token_cap_tokens') = r.token_cap_tokens))
),
matched AS MATERIALIZED (
    -- a comparison-scoped EXCLUDE removes the comparison from the run (the figure no longer reads it there), never the run from the paper
    SELECT m.run_id, m.roster_entry_id, m.comparison_id FROM matched_raw m
    WHERE NOT EXISTS (
        SELECT 1 FROM covered c JOIN active ru ON ru.ruling_id = c.ruling_id
        WHERE c.run_id = m.run_id AND ru.action = 'exclude' AND ru.scope_kind = 'comparison'
          AND (m.comparison_id = json_extract(ru.scope_json, '$.comparison_id')
               OR EXISTS (SELECT 1 FROM json_each(ru.scope_json, '$.match.comparison_ids') v WHERE m.comparison_id GLOB v.value)
               OR (json_type(ru.scope_json, '$.match') IS NULL
                   AND EXISTS (SELECT 1 FROM json_each(ru.scope_json, '$.comparison_ids') v WHERE m.comparison_id GLOB v.value))))
),
decisive AS MATERIALIZED (
    -- the LATEST ruled include / exclude / supersede covering the run decides its membership; a comparison-scoped ruling never does
    SELECT run_id, ruling_id, action,
           ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY ruled_utc DESC, CASE action WHEN 'include' THEN 1 ELSE 0 END, ruling_id) AS rn
    FROM covered WHERE action IN ('include', 'exclude', 'supersede') AND scope_kind <> 'comparison'
),
decision AS MATERIALIZED (
    SELECT run_id, ruling_id, action FROM decisive WHERE rn = 1
),
later AS MATERIALIZED (
    -- a later COMPLETE run at the SAME key supersedes the run: the key is the model_id, object_id, eval_id, instrument_id and judge_id
    -- columns, the two strength columns, operating_point_id_legacy, and token_cap_tokens, trial_index and pass_label -- two token
    -- caps, two trials or two passes of one object are distinct runs, never successors (per-cap registration files a run per token cap
    -- under one seeded static instrument_id, and the smaller cap must not read superseded when the larger one completes). Every
    -- nullable term compares with IS, so NULL pairs with NULL (pass_label is NULL on most rows; trial_index is NOT NULL with default 0)
    -- and a supersession that held before the key grew still holds. The ONE pairwise exception (Q11, migration 0026; rendered from
    -- vocab.COEXIST_ACROSS_VIEWS_BY_SOURCE): two pooled runs of a listed source at one key whose view_legacy differ are the readings of two
    -- POINTS (the tags pool separately), never predecessor and successor -- the pair is skipped; a later same-tag pooled run still supersedes
    SELECT r.run_id
    FROM run r JOIN run r2 ON r2.model_id = r.model_id AND r2.object_id = r.object_id AND r2.eval_id = r.eval_id
        AND r2.instrument_id = r.instrument_id AND r2.judge_id IS r.judge_id AND r2.run_id <> r.run_id
        AND r2.strength_nominal IS r.strength_nominal AND r2.strength_coefficient IS r.strength_coefficient
        AND r2.operating_point_id_legacy IS r.operating_point_id_legacy
        AND r2.token_cap_tokens IS r.token_cap_tokens AND r2.trial_index IS r.trial_index AND r2.pass_label IS r.pass_label
        AND NOT ((r.lane_source_legacy = 'sampled_lane' AND r2.lane_source_legacy = 'sampled_lane' AND r.pass_label IN ('pooled_t0_t1') AND r2.view_legacy IS NOT r.view_legacy))
        AND r2.lifecycle_state IN ('judged', 'ingested')
        AND COALESCE(r2.generated_utc, r2.queued_utc) > COALESCE(r.generated_utc, r.queued_utc)
    WHERE NOT EXISTS (SELECT 1 FROM covered c WHERE c.run_id = r2.run_id AND c.action = 'coexist')
      AND NOT EXISTS (SELECT 1 FROM covered c WHERE c.run_id = r.run_id AND c.action = 'coexist')
),
withdrawn AS MATERIALIZED (
    -- a run every observation of which is no longer current (one pass over the observations; the per-run scan took 145 s live)
    SELECT r.run_id FROM run r WHERE r.lifecycle_state = 'withdrawn'
    UNION
    SELECT json_extract(ob.observation_key, '$.run_id') AS run_id FROM observation ob WHERE ob.table_name = 'run'
    GROUP BY json_extract(ob.observation_key, '$.run_id') HAVING MAX(ob.current) = 0
),
per_run_cover AS MATERIALIZED (
    SELECT run_id, json_group_array(DISTINCT ruling_id) AS ruling_ids_json,
           MAX(CASE WHEN evidence_state IN ('incoherent', 'stale') THEN 1 ELSE 0 END) AS indeterminate,
           MAX(CASE WHEN action = 'hold_out' THEN 1 ELSE 0 END) AS held_out,
           MIN(CASE WHEN action = 'hold_out' THEN ruling_id END) AS held_out_ruling_id
    FROM covered GROUP BY run_id
),
per_run_match AS MATERIALIZED (
    SELECT run_id, json_group_array(DISTINCT comparison_id) AS comparison_ids_json, json_group_array(DISTINCT roster_entry_id) AS roster_entry_ids_json
    FROM matched GROUP BY run_id
),
later_runs AS MATERIALIZED (SELECT DISTINCT run_id FROM later),
pooled_group AS MATERIALIZED (
    -- the points (source, model_id, object_id, eval_id, view_legacy) at which the source filed a pooled reading, in any state
    -- (migration 0022): a member pass is a member (member_pass) at one of these and an unpooled_member elsewhere; neither is a paper row
    SELECT DISTINCT lane_source_legacy, model_id, object_id, eval_id, view_legacy FROM run WHERE pass_label IN ('pooled_t0_t1', 'pooled_final_rung')
)
SELECT run_id,
       -- is_current (migration 0021): 1 when nothing below retires the run -- a selection_state of superseded / withdrawn / excluded, a
       -- hold_out mark, a lifecycle of failed / cancelled / superseded / withdrawn, a superseded_by_run_id, or a decisive
       -- exclude / supersede ruling on a run whose selection_state is not_in_roster (the ruling retires the run whatever its roster
       -- membership; selection_state keeps its roster-first order for the paper and still reads not_in_roster); the paper reads the
       -- rows of a current run only where the frame says so (hold_out keeps a row in the paper and still reads is_current 0;
       -- a member pass of a pooled reading, selection_state member_pass, is current and not a paper row, and so is a member
       -- pass whose own point carries no pooled reading of its source, selection_state unpooled_member)
       CASE WHEN selection_state IN ('superseded', 'withdrawn', 'excluded') OR held_out = 1
                 OR lifecycle_state IN ('failed', 'cancelled', 'superseded', 'withdrawn') OR superseded_by_run_id IS NOT NULL
                 OR (selection_state = 'not_in_roster' AND decisive_action IN ('exclude', 'supersede')) THEN 0 ELSE 1 END AS is_current,
       -- the FIRST failing rule, in the order above, with the ruling id where a ruling decided it; NULL on a current run
       CASE WHEN selection_state = 'superseded'
                 THEN 'selection_state superseded' || CASE WHEN decisive_action = 'supersede' THEN ' (ruling ' || decisive_ruling_id || ')'
                                                           WHEN lifecycle_state = 'superseded' OR superseded_by_run_id IS NOT NULL THEN ' (lifecycle or superseded_by_run_id)'
                                                           ELSE ' (a later run at the same key)' END
            WHEN selection_state = 'withdrawn' THEN 'selection_state withdrawn'
            WHEN selection_state = 'excluded' THEN 'selection_state excluded (ruling ' || decisive_ruling_id || ')'
            WHEN held_out = 1 THEN 'held_out (ruling ' || held_out_ruling_id || ')'
            WHEN lifecycle_state IN ('failed', 'cancelled', 'superseded', 'withdrawn') THEN 'lifecycle_state ' || lifecycle_state
            WHEN superseded_by_run_id IS NOT NULL THEN 'superseded_by_run_id ' || superseded_by_run_id
            WHEN selection_state = 'not_in_roster' AND decisive_action IN ('exclude', 'supersede')
                 THEN 'decisive ruling ' || decisive_action || ' (ruling ' || decisive_ruling_id || ')'
            END AS why,
       decisive_ruling_id, decisive_action, selection_state, in_current_paper, held_out, round_id,
       comparison_ids_json, roster_entry_ids_json, ruling_ids_json
FROM (
    SELECT r.run_id, r.lifecycle_state, r.superseded_by_run_id,
           d.ruling_id AS decisive_ruling_id,
           d.action AS decisive_action,
           COALESCE(pc.held_out, 0) AS held_out,
           pc.held_out_ruling_id,
           CASE
             WHEN COALESCE(pc.indeterminate, 0) = 1 THEN 'indeterminate'
             WHEN w.run_id IS NOT NULL THEN 'withdrawn'
             WHEN d.action = 'include' THEN 'selected'
             WHEN pm.run_id IS NULL THEN 'not_in_roster'
             WHEN r.lifecycle_state = 'superseded' OR r.superseded_by_run_id IS NOT NULL
                  OR d.action = 'supersede' OR l.run_id IS NOT NULL THEN 'superseded'
             WHEN d.action = 'exclude' THEN 'excluded'
             WHEN (((r.lane_source_legacy = 'relu_paired' AND r.pass_label IN ('base_cap', 'ladder_rung', 'unrequested_rung')) OR (r.lane_source_legacy = 'sampled_lane' AND r.pass_label IN ('base_cap', 'ladder_rung'))) AND EXISTS (SELECT 1 FROM pooled_group pg WHERE pg.lane_source_legacy = r.lane_source_legacy AND pg.model_id = r.model_id AND pg.object_id = r.object_id AND pg.eval_id = r.eval_id AND pg.view_legacy IS r.view_legacy)) THEN 'member_pass'
             WHEN ((r.lane_source_legacy = 'relu_paired' AND r.pass_label IN ('base_cap', 'ladder_rung', 'unrequested_rung')) OR (r.lane_source_legacy = 'sampled_lane' AND r.pass_label IN ('base_cap', 'ladder_rung'))) THEN 'unpooled_member'
             WHEN r.lifecycle_state NOT IN ('judged', 'ingested') THEN 'not_in_roster'
             ELSE 'selected'
           END AS selection_state,
           CASE
             WHEN COALESCE(pc.indeterminate, 0) = 1 THEN 0
             WHEN w.run_id IS NOT NULL THEN 0
             WHEN r.lifecycle_state NOT IN ('judged', 'ingested') THEN 0
             WHEN d.action = 'include' THEN 1
             WHEN ((r.lane_source_legacy = 'relu_paired' AND r.pass_label IN ('base_cap', 'ladder_rung', 'unrequested_rung')) OR (r.lane_source_legacy = 'sampled_lane' AND r.pass_label IN ('base_cap', 'ladder_rung'))) THEN 0
             WHEN pm.run_id IS NULL THEN 0
             WHEN r.lifecycle_state = 'superseded' OR r.superseded_by_run_id IS NOT NULL THEN 0
             WHEN d.action IN ('exclude', 'supersede') THEN 0
             WHEN l.run_id IS NOT NULL THEN 0
             ELSE 1
           END AS in_current_paper,
           (SELECT round_id FROM open_round) AS round_id,
           COALESCE(pm.comparison_ids_json, '[]') AS comparison_ids_json,
           COALESCE(pm.roster_entry_ids_json, '[]') AS roster_entry_ids_json,
           COALESCE(pc.ruling_ids_json, '[]') AS ruling_ids_json
    FROM run r
    LEFT JOIN per_run_cover pc ON pc.run_id = r.run_id
    LEFT JOIN per_run_match pm ON pm.run_id = r.run_id
    LEFT JOIN decision d ON d.run_id = r.run_id
    LEFT JOIN withdrawn w ON w.run_id = r.run_id
    LEFT JOIN later_runs l ON l.run_id = r.run_id
);
