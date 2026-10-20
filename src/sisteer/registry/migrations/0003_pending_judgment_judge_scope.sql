-- 0003_pending_judgment_judge_scope.sql
-- v_pending_judgment paired every eligible run with EVERY judge row (`JOIN judge jd` with no condition; the requested
-- judge was meant to be the WHERE filter). For an instrument that names a judge that is the design: a second judge is
-- eligible before it has filed anything. For an instrument WITHOUT a judge — rule-scored (exact match, sandbox,
-- letter-logprob prefill: every capability eval), whose headline metric is filed at generation and which the harvester
-- now ingests from that metric (harvest.completion_requirements_hold) — it listed each run once per judge in the table
-- (12 relu_paired MMLU runs x 15 judges = 180 rows awaiting judges that will never read them, 2026-09-20). The join is
-- now scoped by the instrument: a judge pairs with a run whose instrument names a judge, or with a run it has itself
-- filed a judgment for (a claim, a second-judge read); a rule-scored run with no judgment is pending for nobody.
-- Rendered by `registry migrate --plan --sql` against a schema-version-2 database; only this object differs.
-- drift: v_pending_judgment differs from the code's SQL

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
                                            'claimed_by_session', j2.claimed_by_session))
          FROM judgment j2 WHERE j2.run_id = r.run_id) AS existing_judgments_json,
       cl.judgment_id AS claim_judgment_id, cl.claimed_by_session AS claim_session, cl.claimed_utc AS claim_utc,
       CAST((julianday('now') - julianday(cl.claimed_utc)) * 86400 AS INTEGER) AS claim_age_s,
       (SELECT MAX(COALESCE(cov.required_coverage, 0) - COALESCE(j4.n_items_judged, 0) - COALESCE(j4.n_judge_refusals, 0), 0)
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
        WHERE jc.run_id = r.run_id AND jc.judge_id = jd.judge_id AND jc.status = 'complete' AND COALESCE(jc.n_items_judged, 0) + COALESCE(jc.n_judge_refusals, 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0));

