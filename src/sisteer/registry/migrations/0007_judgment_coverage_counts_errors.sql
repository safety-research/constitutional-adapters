-- 0007_judgment_coverage_counts_errors.sql
-- Coverage counts the items whose judging ERRORED. The one coverage predicate (schema_def.judgment_covers_sql, now over
-- schema_def.judgment_covered_items_sql) read `n_items_judged + n_judge_refusals`; a judge that covered every planned item but
-- recorded one error left its run short by that item, so the run stayed `judged` forever (or_bench_hard: 1,314 judged + 4 declined
-- + 1 error = 1,319 planned; xstest: 249 judged + 1 error = 250; 38 live runs of two readers on 2026-09-21). An error is an outcome
-- for a planned item with no verdict -- never a verdict and never a gap a re-grade would fill -- so it now counts: `judgment.n_errors`
-- when the judging driver counted it, else the largest `n_error_outcomes` the judgment's own run_metric rows carry (one judgment's
-- metric rows describe the same items). `n_incomplete` never counts: an incomplete item has no outcome at all. No table changes, no
-- row moves, no existing column changes meaning; the three objects that read the predicate are recreated -- v_pending_judgment
-- (`items_missing` counts the same way; `existing_judgments_json` also carries `n_errors`), v_roster_match, v_paper_long -- so the
-- run the grading queue stops offering is the run the harvester ingests and the paper build counts, never one and not the other.
-- The harvester's `completion_requirements_hold` reads the same fragment and needs no DDL of its own.
-- Rendered by `registry migrate --plan --sql` against a schema-version-6 database; only these three objects differ.
-- drift: v_pending_judgment differs from the code's SQL
-- drift: v_roster_match differs from the code's SQL
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
       r.object_id, o.paper_label, o.object_role, o.object_kind, o.fit_model_id, o.fit_layers_json, o.fit_layer_count, o.fit_replicate_label,
       o.status AS object_status, o.artifact_sha256 AS object_artifact_sha256,
       r.eval_id, e.eval_category, e.headline_metric_id, r.instrument_id, r.item_set_id, r.harness, r.n_shot, r.cot_prompting, r.thinking_mode,
       i.bon_n, i.n_trials, i.max_turns,
       r.token_cap_tokens, r.token_cap_rung_index, r.context_window_tokens, r.serving_path, r.serving_config_sha256,
       r.serving_identity, r.serving_engine_version,
       r.decode_source, r.decode_sent_json, r.decode_policy_id, r.decode_policy_match, jm.judge_id, j.judge_model, j.rubric_name AS judge_rubric_name, r.judge_id AS run_judge_id,
       jm.n_judge_refusals AS judgment_n_judge_refusals,
       r.pass_label, r.lifecycle_state, CASE WHEN r.lifecycle_state IN ('judged', 'ingested') THEN 1 ELSE 0 END AS is_complete, r.store_uri, r.intervention_hash, r.notes, r.extra_json AS run_extra_json,
       COALESCE(json_extract(rm.extra_json, '$.eval_id_legacy'), json_extract(r.extra_json, '$.eval_id_legacy')) AS eval_id_legacy,
       rm.judgment_id, rm.metric_id, rm.producer_metric_spelling, rm.threshold, rm.computation_id,
       CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE rm.value END AS value, CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE rm.k END AS k, CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE rm.n END AS n, CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE rm.ci_lo END AS ci_lo, CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE rm.ci_hi END AS ci_hi,
       rm.ci_method, rm.frac_at_token_cap, rm.frac_degenerate_classifier, rm.frac_degenerate_contentless,
       rm.frac_degenerate_rep_ratio_loop, rm.frac_degenerate_zlib_collapse, rm.frac_degenerate_union5, rm.frac_loop_first512,
       rm.frac_defended_degenerate, rm.n_abstentions, rm.n_error_outcomes, rm.n_incomplete, rm.source_uri,
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
DROP VIEW IF EXISTS v_pending_judgment;
CREATE VIEW v_pending_judgment AS
WITH cov AS (
    SELECT r.run_id, COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)) AS required_coverage FROM run r
)
SELECT r.run_id, r.producer_run_key_legacy, r.producer_object_key_legacy, r.store_uri, r.n_items_generated, r.campaign,
       r.model_id, r.object_id, r.eval_id, e.eval_category, r.instrument_id, i.harness, i.decode_source,
       r.lifecycle_state, r.judge_id AS run_judge_id, jd.judge_id AS required_judge_id,
       CASE WHEN i.judge_id = jd.judge_id THEN 1 ELSE 0 END AS is_instrument_judge,
       cov.required_coverage,
       (SELECT json_group_array(json_object('judgment_id', j2.judgment_id, 'judge_id', j2.judge_id, 'status', j2.status,
                                            'n_items_judged', j2.n_items_judged, 'n_judge_refusals', j2.n_judge_refusals,
                                            'n_errors', j2.n_errors, 'claimed_by_session', j2.claimed_by_session))
          FROM judgment j2 WHERE j2.run_id = r.run_id) AS existing_judgments_json,
       cl.judgment_id AS claim_judgment_id, cl.claimed_by_session AS claim_session, cl.claimed_utc AS claim_utc,
       CAST((julianday('now') - julianday(cl.claimed_utc)) * 86400 AS INTEGER) AS claim_age_s,
       (SELECT MAX(COALESCE(cov.required_coverage, 0) - (COALESCE(j4.n_items_judged, 0) + COALESCE(j4.n_judge_refusals, 0) + COALESCE(j4.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = j4.run_id AND rm_err.judgment_id = j4.judgment_id), 0)), 0)
          FROM judgment j4 WHERE j4.run_id = r.run_id AND j4.judge_id = jd.judge_id AND j4.status IN ('partial', 'complete')) AS items_missing,
       (SELECT COUNT(*) FROM judgment j5 WHERE j5.run_id = r.run_id AND j5.judge_id = jd.judge_id AND j5.status = 'abandoned') AS n_abandoned,
       (SELECT MAX(j5.claimed_utc) FROM judgment j5 WHERE j5.run_id = r.run_id AND j5.judge_id = jd.judge_id
          AND j5.status = 'abandoned') AS last_abandoned_utc,
       (SELECT COUNT(*) FROM judgment j6 WHERE j6.run_id = r.run_id AND j6.judge_id <> jd.judge_id
          AND j6.status = 'complete') AS n_complete_other_judges,
       CASE WHEN cl.judgment_id IS NOT NULL THEN 'claimed'
            WHEN EXISTS (SELECT 1 FROM judgment j7 WHERE j7.run_id = r.run_id AND j7.judge_id = jd.judge_id
                           AND j7.status = 'partial') THEN 'partial'
            WHEN EXISTS (SELECT 1 FROM judgment j7 WHERE j7.run_id = r.run_id AND j7.judge_id = jd.judge_id
                           AND j7.status = 'complete') THEN 'coverage_short'
            WHEN EXISTS (SELECT 1 FROM judgment j7 WHERE j7.run_id = r.run_id AND j7.judge_id = jd.judge_id
                           AND j7.status = 'abandoned') THEN 'abandoned'
            WHEN EXISTS (SELECT 1 FROM judgment j7 WHERE j7.run_id = r.run_id AND j7.judge_id = jd.judge_id
                           AND j7.status = 'superseded') THEN 'superseded'
            WHEN EXISTS (SELECT 1 FROM judgment j8 WHERE j8.run_id = r.run_id AND j8.judge_id <> jd.judge_id)
                 THEN 'awaiting_second_judge'
            ELSE 'no_judgment' END AS pending_reason
FROM run r
JOIN cov ON cov.run_id = r.run_id
JOIN instrument i ON i.instrument_id = r.instrument_id
JOIN eval e ON e.eval_id = r.eval_id
-- a judge pairs with a run of an instrument that names a judge, or with a run it has filed a judgment for
JOIN judge jd ON (i.judge_id IS NOT NULL
                  OR EXISTS (SELECT 1 FROM judgment jj WHERE jj.run_id = r.run_id AND jj.judge_id = jd.judge_id))
LEFT JOIN judgment cl ON cl.judgment_id = (
        SELECT j3.judgment_id FROM judgment j3 WHERE j3.run_id = r.run_id AND j3.judge_id = jd.judge_id
          AND j3.status = 'in_progress' ORDER BY j3.claimed_utc DESC LIMIT 1)
WHERE r.lifecycle_state IN ('generated', 'judging', 'judged', 'ingested')
  AND NOT EXISTS (
        SELECT 1 FROM judgment jc
        WHERE jc.run_id = r.run_id AND jc.judge_id = jd.judge_id AND jc.status = 'complete' AND COALESCE(jc.n_items_judged, 0) + COALESCE(jc.n_judge_refusals, 0) + COALESCE(jc.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jc.run_id AND rm_err.judgment_id = jc.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0));
DROP VIEW IF EXISTS v_roster_match;
CREATE VIEW v_roster_match AS
SELECT r.run_id, re.roster_entry_id, re.comparison_id
    FROM run r
    JOIN object o ON o.object_id = r.object_id
    JOIN instrument i ON i.instrument_id = r.instrument_id
    -- the roster_entry index (round_id, model_id, eval_id) carries this join; a CTE over the table would not be indexed
    JOIN roster_entry re ON re.round_id = (SELECT round_id FROM paper_round WHERE closed_utc IS NULL ORDER BY opened_utc DESC LIMIT 1)
      AND re.model_id = r.model_id AND re.eval_id = r.eval_id
      AND (re.object_id = r.object_id
           OR (re.object_id IS NULL
               AND (json_extract(re.object_selector_json, '$.paper_label') IS NULL OR json_extract(re.object_selector_json, '$.paper_label') = o.paper_label)
               AND (json_extract(re.object_selector_json, '$.object_kind') IS NULL OR json_extract(re.object_selector_json, '$.object_kind') = o.object_kind)
               AND (json_extract(re.object_selector_json, '$.object_role') IS NULL OR json_extract(re.object_selector_json, '$.object_role') = o.object_role)
               AND (json_extract(re.object_selector_json, '$.fit_model_id') IS NULL OR json_extract(re.object_selector_json, '$.fit_model_id') = o.fit_model_id)
               AND (json_extract(re.object_selector_json, '$.transfer_kind') IS NULL OR json_extract(re.object_selector_json, '$.transfer_kind') = r.transfer_kind)
               AND (json_type(re.object_selector_json, '$.transfer_kinds') IS NULL OR EXISTS (SELECT 1 FROM json_each(re.object_selector_json, '$.transfer_kinds') sv WHERE sv.value = r.transfer_kind))
               AND (json_extract(re.object_selector_json, '$.object_id_prefix') IS NULL
                    OR r.object_id GLOB json_extract(re.object_selector_json, '$.object_id_prefix') || '*')))
      AND (re.instrument_id = r.instrument_id
           OR (re.instrument_id IS NULL
               AND (json_extract(re.instrument_selector_json, '$.harness') IS NULL OR json_extract(re.instrument_selector_json, '$.harness') = i.harness)
               AND (json_extract(re.instrument_selector_json, '$.item_set_id') IS NULL OR json_extract(re.instrument_selector_json, '$.item_set_id') = i.item_set_id)
               AND (json_extract(re.instrument_selector_json, '$.thinking_mode') IS NULL OR json_extract(re.instrument_selector_json, '$.thinking_mode') = i.thinking_mode)
               AND (json_extract(re.instrument_selector_json, '$.serving_path') IS NULL OR json_extract(re.instrument_selector_json, '$.serving_path') = i.serving_path)
               AND (json_extract(re.instrument_selector_json, '$.decode_source') IS NULL OR json_extract(re.instrument_selector_json, '$.decode_source') = i.decode_source)
               AND (json_extract(re.instrument_selector_json, '$.scoring_rule') IS NULL OR json_extract(re.instrument_selector_json, '$.scoring_rule') = i.scoring_rule)
               AND (json_extract(re.instrument_selector_json, '$.n_shot') IS NULL OR json_extract(re.instrument_selector_json, '$.n_shot') = i.n_shot)
               AND (json_type(re.instrument_selector_json, '$.token_cap_tokens') IS NULL OR EXISTS (SELECT 1 FROM json_each(re.instrument_selector_json, '$.token_cap_tokens') sv WHERE sv.value = i.token_cap_tokens))))
      -- the entry is satisfied only by a COMPLETE judgment of its judge covering the items of the run; an entry that
      -- names no judge reads the judge of the instrument (a rule-scored instrument: a generation-time metric row)
      AND ((re.judge_id IS NOT NULL
            AND EXISTS (SELECT 1 FROM judgment jx WHERE jx.run_id = r.run_id AND jx.judge_id = re.judge_id AND jx.status = 'complete' AND COALESCE(jx.n_items_judged, 0) + COALESCE(jx.n_judge_refusals, 0) + COALESCE(jx.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jx.run_id AND rm_err.judgment_id = jx.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)))
           OR (re.judge_selector_json IS NOT NULL
               AND EXISTS (SELECT 1 FROM judgment jx JOIN judge jj ON jj.judge_id = jx.judge_id WHERE jx.run_id = r.run_id
                           AND jx.status = 'complete' AND COALESCE(jx.n_items_judged, 0) + COALESCE(jx.n_judge_refusals, 0) + COALESCE(jx.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jx.run_id AND rm_err.judgment_id = jx.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                           AND (json_extract(re.judge_selector_json, '$.judge_model') IS NULL
                                OR json_extract(re.judge_selector_json, '$.judge_model') = jj.judge_model)
                           AND (json_extract(re.judge_selector_json, '$.rubric_name') IS NULL
                                OR json_extract(re.judge_selector_json, '$.rubric_name') = jj.rubric_name)))
           OR (re.judge_id IS NULL AND re.judge_selector_json IS NULL
               AND ((i.judge_id IS NULL AND EXISTS (SELECT 1 FROM run_metric rm0 WHERE rm0.run_id = r.run_id AND rm0.judgment_id = ''))
                    OR (i.judge_id IS NOT NULL
                        AND EXISTS (SELECT 1 FROM judgment jx WHERE jx.run_id = r.run_id AND jx.judge_id = i.judge_id AND jx.status = 'complete' AND COALESCE(jx.n_items_judged, 0) + COALESCE(jx.n_judge_refusals, 0) + COALESCE(jx.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jx.run_id AND rm_err.judgment_id = jx.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0))))))
      AND ((re.strength_rule = 'operating_point' AND (r.is_operating_point = 1 OR o.object_kind IN ('none', 'system_instruction')))
           OR (re.strength_rule GLOB 'fixed:*' AND r.strength_nominal = CAST(substr(re.strength_rule, 7) AS REAL))
           OR (re.strength_rule GLOB 'ladder:*' AND r.strength_nominal IS NOT NULL));
