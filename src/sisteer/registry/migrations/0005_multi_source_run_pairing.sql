-- 0005_multi_source_run_pairing.sql
-- The multi-source ownership views matched an owner row to a secondary row on (eval, metric, threshold) inside a scope of
-- model x object label x eval category at the served operating point, and counted EVERY metric row of a source at an identity.
-- Two facts of the Llama-3.1-8B Petri measurements broke that contract (verified read-only on the live registry 2026-09-20/21):
-- each fitted paper label (constitution_lora, control_lora, difference_lora) carries TWO objects, a fit on the base checkpoint
-- served on the instruct one (transfer_kind base_to_instruct) and a fit on the instruct checkpoint (instruct_direct), seven Petri
-- evals each in both sources (the model's own reader llama8b_lane and r2_export), so a scope ignoring the transfer kind saw two
-- runs of one source at one identity; and the petri_stats reader files a producer-filed blank-computation twin of every export
-- row onto the export's run (the row v_paper_long's metric role ranks second since 0004), so the export's side counted two rows
-- per identity. Every measurement of such a declaration would have read `refused` and emptied the paper's measurements. The scope now
-- names the transfer kind (a declaration is per transfer kind), a strength rule (`served_operating_point`: the declared id, or no
-- id at nominal 1.0; `unfitted`: no id and no nominal) and any eval ids excluded by name (`run.eval_id NOT IN (...)`); the fold
-- identity gains a `pairing_key` rendered per declaration (`producer_run_key`: run.producer_run_key_legacy, which the two Llama
-- readers spell alike, so a row folds only onto the row read from the same store; `identity_only`: '' for the Qwen pair, whose two
-- readers spell the key differently and whose scope holds one run per eval per side); and BOTH sides are their source's own
-- computation rows only (run_metric.computation_id <> '' — the export carries no blank-computation key anywhere, 8,231 computed;
-- the readers' rows all name one). Six declarations land with it, `llama8b_it/<label>/petri/<transfer_kind>`: the model's own
-- reader (llama8b_lane) OWNS and the export is the secondary — paperlib's rule since 2026-09-14 (load_r2 drops the export's Llama
-- Petri rows at read), restated on 2026-09-21 with the finding behind it: the export's n for these
-- measurements counts scenarios PLANNED (build_export.py has no transfer-stats join for the Llama settings and takes petri_agg's
-- n_items) where the reader counts scenarios SCORED (the Petri judge refused one finished transcript); the export's own pooled n of
-- 59 for the same object proves the planned count wrong (10 x 6 is not 59). `petri_pooled` is out of the six scopes: the reader's
-- pooled runs are not_in_roster (their instruments carry thinking_mode unverified and miss the roster's thinking_mode = off
-- selector), so an owner row there would sit outside the paper frame while the fold removed the export's row and the pooled cell
-- would empty; outside the scope it stays as today (the export's row selected, the reader's not in the roster, no double) until
-- the roster admits the reader's pooled instrument. On the live registry: 36 identities (three labels x two transfer kinds x six
-- Petri sub-evals), 35 fold with equal values and n, the reader's row standing as the owner, and ONE is refused per identity
-- (control_lora base_to_instruct x petri_cyber_bio_chem_capabilities, value 7.333 on both sides, n 9 in the reader against 10 in
-- the export) and stays refused rather than swallowed by a tolerance. The Qwen declaration renders to the same fold as before
-- (15 identities, all stopgap on the live registry). v_multi_source_rows gains pairing_key, transfer_kind,
-- producer_run_key_legacy, view_legacy; v_multi_source_check gains transfer_kind, pairing_key, owner_producer_run_key_legacy,
-- secondary_producer_run_key_legacy and folds per pairing key; v_paper_long joins the check on the pairing key too. Views only;
-- no table changes.
-- Rendered by `registry migrate --plan --sql` against a schema-version-4 database; only these three objects differ.
-- drift: v_multi_source_rows differs from the code's SQL
-- drift: v_multi_source_check differs from the code's SQL
-- drift: v_paper_long differs from the code's SQL

DROP VIEW IF EXISTS v_multi_source_check;
CREATE VIEW v_multi_source_check AS
WITH scoped AS MATERIALIZED (SELECT * FROM v_multi_source_rows),
owner AS MATERIALIZED (
    SELECT declaration_id, pairing_key, eval_id, metric_id, threshold, COUNT(*) AS n_rows,
           MIN(owner_source) AS owner_source, MIN(secondary_source) AS secondary_source, MIN(model_id) AS model_id, MIN(paper_label) AS paper_label,
           MIN(eval_category) AS eval_category, MIN(transfer_kind) AS transfer_kind, MIN(producer_run_key_legacy) AS producer_run_key_legacy,
           MIN(run_id) AS run_id, MIN(judgment_id) AS judgment_id, MIN(computation_id) AS computation_id,
           MIN(value) AS value, MIN(k) AS k, MIN(n) AS n, MIN(degenerate_share) AS degenerate_share, MIN(loop_share) AS loop_share,
           MIN(loop_share_measure) AS loop_share_measure, MIN(at_cap_share) AS at_cap_share, MAX(ingest_seq) AS ingest_seq
    FROM scoped WHERE role = 'owner' GROUP BY declaration_id, pairing_key, eval_id, metric_id, threshold
),
secondary AS MATERIALIZED (
    SELECT declaration_id, pairing_key, eval_id, metric_id, threshold, COUNT(*) AS n_rows,
           MIN(owner_source) AS owner_source, MIN(secondary_source) AS secondary_source, MIN(model_id) AS model_id, MIN(paper_label) AS paper_label,
           MIN(eval_category) AS eval_category, MIN(transfer_kind) AS transfer_kind, MIN(producer_run_key_legacy) AS producer_run_key_legacy,
           MIN(run_id) AS run_id, MIN(judgment_id) AS judgment_id, MIN(computation_id) AS computation_id,
           MIN(value) AS value, MIN(k) AS k, MIN(n) AS n, MIN(degenerate_share) AS degenerate_share, MIN(loop_share) AS loop_share,
           MIN(loop_share_measure) AS loop_share_measure, MIN(at_cap_share) AS at_cap_share, MAX(ingest_seq) AS ingest_seq
    FROM scoped WHERE role = 'secondary' GROUP BY declaration_id, pairing_key, eval_id, metric_id, threshold
)
SELECT s.declaration_id, s.owner_source, s.secondary_source, s.model_id, s.paper_label, s.eval_category, s.transfer_kind,
       s.pairing_key, s.eval_id, s.metric_id, s.threshold,
       CASE WHEN ow.run_id IS NULL THEN 'stopgap'
     WHEN ow.n_rows > 1 OR s.n_rows > 1 THEN 'refused'
     WHEN (NOT (ow.k IS s.k) OR NOT (ow.n IS s.n)) THEN 'refused'
     WHEN (ow.degenerate_share IS NOT NULL AND s.degenerate_share IS NOT NULL AND ABS(ow.degenerate_share - s.degenerate_share) > 1e-9) THEN 'refused'
     WHEN (ow.loop_share IS NOT NULL AND s.loop_share IS NOT NULL AND ABS(ow.loop_share - s.loop_share) > 1e-9) THEN 'refused'
     ELSE 'folded' END AS state,
       CASE WHEN ow.run_id IS NULL THEN 'the owner (' || s.owner_source || ') carries no row for this measurement yet; the secondary (' || s.secondary_source || ') row stands as a stopgap until it does, and is then dropped once k and n agree exactly'
     WHEN ow.n_rows > 1 THEN 'the owner (' || s.owner_source || ') carries ' || ow.n_rows || ' rows at one identity; one owner row per measurement is the precondition of the fold, so neither value is admitted'
     WHEN s.n_rows > 1 THEN 'the secondary (' || s.secondary_source || ') carries ' || s.n_rows || ' rows at one identity; one row per measurement is the precondition of the fold, so neither value is admitted'
     WHEN (NOT (ow.k IS s.k) OR NOT (ow.n IS s.n)) THEN 'the two sources disagree on k / n (owner k=' || COALESCE(CAST(ow.k AS TEXT), 'null') || ' n=' || COALESCE(CAST(ow.n AS TEXT), 'null') || '; secondary k=' || COALESCE(CAST(s.k AS TEXT), 'null') || ' n=' || COALESCE(CAST(s.n AS TEXT), 'null') || '); two readings of one measurement that differ are a disagreement to rule on, never averaged and never picked from, so neither value is admitted'
     WHEN (ow.degenerate_share IS NOT NULL AND s.degenerate_share IS NOT NULL AND ABS(ow.degenerate_share - s.degenerate_share) > 1e-9) THEN 'the owner (' || s.owner_source || ') already carries degenerate_share=' || COALESCE(CAST(ow.degenerate_share AS TEXT), 'null') || ' and the secondary (' || s.secondary_source || ') reads ' || COALESCE(CAST(s.degenerate_share AS TEXT), 'null') || '; a disagreement is never overwritten and neither value is admitted'
     WHEN (ow.loop_share IS NOT NULL AND s.loop_share IS NOT NULL AND ABS(ow.loop_share - s.loop_share) > 1e-9) THEN 'the owner (' || s.owner_source || ') already carries loop_share=' || COALESCE(CAST(ow.loop_share AS TEXT), 'null') || ' and the secondary (' || s.secondary_source || ') reads ' || COALESCE(CAST(s.loop_share AS TEXT), 'null') || '; a disagreement is never overwritten and neither value is admitted'
     ELSE 'k and n agree exactly (k=' || COALESCE(CAST(ow.k AS TEXT), 'null') || ', n=' || COALESCE(CAST(ow.n AS TEXT), 'null') || '); the owner (' || s.owner_source || ') row stands and the secondary (' || s.secondary_source || ') row is dropped; carried onto the owner where it had none: ' || COALESCE(NULLIF(REPLACE(TRIM(CASE WHEN (ow.degenerate_share IS NULL AND s.degenerate_share IS NOT NULL) THEN '"degenerate_share",' ELSE '' END || CASE WHEN (ow.loop_share IS NULL AND s.loop_share IS NOT NULL) THEN '"loop_share",' ELSE '' END, ','), '"', ''), ''), 'nothing') END AS reason,
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
     WHEN (NOT (ow.k IS s.k) OR NOT (ow.n IS s.n)) THEN 'refused'
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
)
SELECT 'qwen35_4b_it/arditi_allpos/multi_turn' AS declaration_id, 'r2_export' AS owner_source, 'arditi_allpos' AS secondary_source,
       CASE WHEN r.lane_source_legacy = 'r2_export' THEN 'owner' ELSE 'secondary' END AS role,
       '' AS pairing_key,
       r.lane_source_legacy, r.run_id, r.model_id, o.paper_label, e.eval_category, e.headline_metric_id, r.eval_id,
       r.transfer_kind, r.producer_run_key_legacy, r.view_legacy, r.operating_point_id_legacy, r.strength_nominal,
       rm.judgment_id, rm.metric_id, rm.threshold, rm.computation_id, rm.value, rm.k, rm.n,
       rm.frac_at_token_cap AS at_cap_share,
       COALESCE(rm.frac_degenerate_union5, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share,
       CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END AS loop_share_measure,
       CASE CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END WHEN 'classifier' THEN rm.frac_degenerate_classifier WHEN 'loop_first512' THEN rm.frac_loop_first512 ELSE rm.frac_degenerate_rep_ratio_loop END AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
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
       rm.frac_at_token_cap AS at_cap_share,
       COALESCE(rm.frac_degenerate_union5, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share,
       CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END AS loop_share_measure,
       CASE CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END WHEN 'classifier' THEN rm.frac_degenerate_classifier WHEN 'loop_first512' THEN rm.frac_loop_first512 ELSE rm.frac_degenerate_rep_ratio_loop END AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
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
       rm.frac_at_token_cap AS at_cap_share,
       COALESCE(rm.frac_degenerate_union5, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share,
       CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END AS loop_share_measure,
       CASE CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END WHEN 'classifier' THEN rm.frac_degenerate_classifier WHEN 'loop_first512' THEN rm.frac_loop_first512 ELSE rm.frac_degenerate_rep_ratio_loop END AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
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
       rm.frac_at_token_cap AS at_cap_share,
       COALESCE(rm.frac_degenerate_union5, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share,
       CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END AS loop_share_measure,
       CASE CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END WHEN 'classifier' THEN rm.frac_degenerate_classifier WHEN 'loop_first512' THEN rm.frac_loop_first512 ELSE rm.frac_degenerate_rep_ratio_loop END AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
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
       rm.frac_at_token_cap AS at_cap_share,
       COALESCE(rm.frac_degenerate_union5, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share,
       CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END AS loop_share_measure,
       CASE CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END WHEN 'classifier' THEN rm.frac_degenerate_classifier WHEN 'loop_first512' THEN rm.frac_loop_first512 ELSE rm.frac_degenerate_rep_ratio_loop END AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
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
       rm.frac_at_token_cap AS at_cap_share,
       COALESCE(rm.frac_degenerate_union5, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share,
       CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END AS loop_share_measure,
       CASE CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END WHEN 'classifier' THEN rm.frac_degenerate_classifier WHEN 'loop_first512' THEN rm.frac_loop_first512 ELSE rm.frac_degenerate_rep_ratio_loop END AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
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
       rm.frac_at_token_cap AS at_cap_share,
       COALESCE(rm.frac_degenerate_union5, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share,
       CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END AS loop_share_measure,
       CASE CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END WHEN 'classifier' THEN rm.frac_degenerate_classifier WHEN 'loop_first512' THEN rm.frac_loop_first512 ELSE rm.frac_degenerate_rep_ratio_loop END AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  AND r.model_id IN ('llama8b_it') AND o.paper_label = 'difference_lora' AND e.eval_category = 'petri'
    AND r.eval_id NOT IN ('petri_pooled')
    AND r.transfer_kind = 'instruct_direct'
    AND r.lane_source_legacy IN ('llama8b_lane', 'r2_export')
    AND (r.operating_point_id_legacy = 'itdoc__constv2-mix-full__lora_r4_ce' OR (r.operating_point_id_legacy IS NULL AND r.strength_nominal = 1.0));
DROP VIEW IF EXISTS v_paper_long;
CREATE VIEW v_paper_long AS
WITH withdrawn_metric AS MATERIALIZED (
    -- a metric row whose observation key is no longer current under any revision (one pass; the per-row scan does not scale)
    SELECT json_extract(observation_key, '$.run_id') AS run_id, json_extract(observation_key, '$.judgment_id') AS judgment_id,
           json_extract(observation_key, '$.metric_id') AS metric_id, json_extract(observation_key, '$.threshold') AS threshold,
           json_extract(observation_key, '$.computation_id') AS computation_id
    FROM observation WHERE table_name = 'run_metric' GROUP BY observation_key HAVING MAX(current) = 0
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
                          WHEN re.judge_id IS NOT NULL THEN jm.judge_id = re.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                          WHEN re.judge_selector_json IS NOT NULL THEN j.judge_id IS NOT NULL AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                               AND (json_extract(re.judge_selector_json, '$.judge_model') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.judge_model') = j.judge_model)
                               AND (json_extract(re.judge_selector_json, '$.rubric_name') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.rubric_name') = j.rubric_name)
                          ELSE (rm.judgment_id = '' AND i.judge_id IS NULL)
                               OR (jm.judge_id IS NOT NULL AND jm.judge_id IS i.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0))
                        END) AS comparison_ids_json,
       (SELECT json_group_array(DISTINCT vm.roster_entry_id) FROM v_roster_match vm JOIN roster_entry re ON re.roster_entry_id = vm.roster_entry_id
         WHERE vm.run_id = r.run_id
           AND EXISTS (SELECT 1 FROM json_each(pc.roster_entry_ids_json) kept WHERE kept.value = vm.roster_entry_id)
           AND CASE
                          WHEN re.judge_id IS NOT NULL THEN jm.judge_id = re.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                          WHEN re.judge_selector_json IS NOT NULL THEN j.judge_id IS NOT NULL AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                               AND (json_extract(re.judge_selector_json, '$.judge_model') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.judge_model') = j.judge_model)
                               AND (json_extract(re.judge_selector_json, '$.rubric_name') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.rubric_name') = j.rubric_name)
                          ELSE (rm.judgment_id = '' AND i.judge_id IS NULL)
                               OR (jm.judge_id IS NOT NULL AND jm.judge_id IS i.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0))
                        END) AS roster_entry_ids_json,
       pc.decisive_ruling_id, pc.decisive_action,
       r.run_id, r.campaign, r.lane_source_legacy, r.view_legacy, r.campaign_cell_id_legacy, r.operating_point_id_legacy,
       r.producer_run_key_legacy, r.producer_object_key_legacy,
       r.model_id, m.family AS model_family, r.transfer_kind, r.strength_nominal, r.strength_coefficient, r.is_operating_point,
       r.object_id, o.paper_label, o.object_role, o.object_kind, o.fit_model_id, o.fit_layers_json, o.fit_layer_count, o.fit_replicate_label,
       o.status AS object_status, o.artifact_sha256 AS object_artifact_sha256,
       r.eval_id, e.eval_category, e.headline_metric_id, r.instrument_id, r.item_set_id, r.harness, r.n_shot, r.cot_prompting, r.thinking_mode,
       i.bon_n, i.n_trials, i.max_turns,
       r.token_cap_tokens, r.token_cap_rung_index, r.context_window_tokens, r.serving_path, r.serving_config_sha256,
       r.serving_identity, r.serving_engine_version,
       r.decode_source, r.decode_sent_json, r.decode_policy_id, r.decode_policy_match, jm.judge_id, j.judge_model, j.rubric_name AS judge_rubric_name, r.judge_id AS run_judge_id,
       jm.n_judge_refusals AS judgment_n_judge_refusals,
       r.pass_label, r.lifecycle_state, r.store_uri, r.intervention_hash, r.notes, r.extra_json AS run_extra_json,
       COALESCE(json_extract(rm.extra_json, '$.eval_id_legacy'), json_extract(r.extra_json, '$.eval_id_legacy')) AS eval_id_legacy,
       rm.judgment_id, rm.metric_id, rm.producer_metric_spelling, rm.threshold, rm.computation_id,
       CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE rm.value END AS value, CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE rm.k END AS k, CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE rm.n END AS n, CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE rm.ci_lo END AS ci_lo, CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE rm.ci_hi END AS ci_hi,
       rm.ci_method, rm.frac_at_token_cap, rm.frac_degenerate_classifier, rm.frac_degenerate_contentless,
       rm.frac_degenerate_rep_ratio_loop, rm.frac_degenerate_zlib_collapse, rm.frac_degenerate_union5, rm.frac_loop_first512,
       rm.frac_defended_degenerate, rm.n_abstentions, rm.n_error_outcomes, rm.n_incomplete, rm.source_uri,
       rm.source_producers_json, rm.extra_json AS metric_extra_json,
       rc.anchor_run_id, rc.anchor_kind, rc.delta, rc.delta_lo, rc.delta_hi, rc.delta_p, rc.delta_method, rc.effect_size,
       rc.effect_kind, rc.n_pairs, rc.extra_json AS contrast_extra_json,
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
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
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
                          WHEN re.judge_id IS NOT NULL THEN jm.judge_id = re.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                          WHEN re.judge_selector_json IS NOT NULL THEN j.judge_id IS NOT NULL AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                               AND (json_extract(re.judge_selector_json, '$.judge_model') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.judge_model') = j.judge_model)
                               AND (json_extract(re.judge_selector_json, '$.rubric_name') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.rubric_name') = j.rubric_name)
                          ELSE (rm.judgment_id = '' AND i.judge_id IS NULL)
                               OR (jm.judge_id IS NOT NULL AND jm.judge_id IS i.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0))
                        END));

