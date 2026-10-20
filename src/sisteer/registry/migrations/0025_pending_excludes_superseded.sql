-- 0025_pending_excludes_superseded.sql
-- A superseded run is never pending; its successor is (a registry ruling of 2026-10-04, after it was found, while
-- applying the identity-current sibling rule on its branch, that an identity correction -- instrument X to X2, the same judge -- landing
-- on a SUPERSEDED run (`run.superseded_by_run_id` set) whose complete judgment had been claimed under X made `v_pending_judgment` list
-- the retired run itself as `identity_stale`: a retired run claimable again beside its successor). Numbered 0025 in this tree -- max landed
-- + 1, re-checked at the merge as the migrations README says; renumbered before its gate if another branch's 0025 lands first (no Python
-- post-step, so nothing is keyed by the slug `pending_excludes_superseded`). ONE object recreated: `v_pending_judgment` gains the conjunct
-- `AND NOT (r.superseded_by_run_id IS NOT NULL)` (`schema_def.run_superseded_sql`, the one predicate `pending.not_pending_why` reads too,
-- so a claim of a superseded run refuses `ClaimNotPending` naming the successor: `superseded by run <id>; admits no judging`). Every other
-- row of the selection is identical; no table is rebuilt, no column is added, no row is copied. Rendered by `registry migrate --plan --sql`
-- against a schema-version-24 database (0001..0024 applied to an empty database by the runner): the planner reports one view recreate and
-- nothing else, and tests/test_registry_pending_claim_recheck.py pins the plan -- any `rebuild` line is a STOP (the 0020 rebuild of
-- row_history wrote a 25 GB WAL). Live, read-only, 2026-10-04: the exclusion moves no row that is not a superseded run's.
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
                                            'n_errors', j2.n_errors, 'claimed_by_session', j2.claimed_by_session))
          FROM judgment j2 WHERE j2.run_id = r.run_id) AS existing_judgments_json,
       cl.judgment_id AS claim_judgment_id, cl.claimed_by_session AS claim_session, cl.claimed_utc AS claim_utc,
       CAST((julianday('now') - julianday(cl.claimed_utc)) * 86400 AS INTEGER) AS claim_age_s,
       (SELECT MAX(COALESCE(cov.required_coverage, 0) - (COALESCE(j4.n_items_judged, 0) + COALESCE(j4.n_judge_refusals, 0) + COALESCE(j4.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = j4.run_id AND rm_err.judgment_id = j4.judgment_id), 0)), 0)
          FROM judgment j4 WHERE j4.run_id = r.run_id AND j4.judge_id = jd.judge_id AND j4.status IN ('partial', 'complete')
          AND ((j4.claimed_instrument_id IS NULL OR j4.claimed_instrument_id IS r.instrument_id) AND (j4.claimed_required_judge_id IS NULL OR j4.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))) AS items_missing,
       (SELECT COUNT(*) FROM judgment j5 WHERE j5.run_id = r.run_id AND j5.judge_id = jd.judge_id AND j5.status = 'abandoned') AS n_abandoned,
       (SELECT MAX(j5.claimed_utc) FROM judgment j5 WHERE j5.run_id = r.run_id AND j5.judge_id = jd.judge_id
          AND j5.status = 'abandoned') AS last_abandoned_utc,
       (SELECT COUNT(*) FROM judgment j6 WHERE j6.run_id = r.run_id AND j6.judge_id <> jd.judge_id
          AND j6.status = 'complete') AS n_complete_other_judges,
       CASE WHEN cl.judgment_id IS NOT NULL THEN 'claimed'
            WHEN EXISTS (SELECT 1 FROM judgment j7 WHERE j7.run_id = r.run_id AND j7.judge_id = jd.judge_id
                           AND j7.status = 'partial' AND ((j7.claimed_instrument_id IS NULL OR j7.claimed_instrument_id IS r.instrument_id) AND (j7.claimed_required_judge_id IS NULL OR j7.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))) THEN 'partial'
            WHEN EXISTS (SELECT 1 FROM judgment j7 WHERE j7.run_id = r.run_id AND j7.judge_id = jd.judge_id
                           AND j7.status = 'complete' AND ((j7.claimed_instrument_id IS NULL OR j7.claimed_instrument_id IS r.instrument_id) AND (j7.claimed_required_judge_id IS NULL OR j7.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))) THEN 'coverage_short'
            -- the only judgments by J (a claim, a partial or a complete reading) were bound at claim to a since-corrected run identity:
            -- a fresh grade is wanted, and a stale-bound claim holds nothing (pending.claim closes it and binds a new one)
            WHEN EXISTS (SELECT 1 FROM judgment j7 WHERE j7.run_id = r.run_id AND j7.judge_id = jd.judge_id
                           AND j7.status IN ('in_progress', 'partial', 'complete')
                           AND NOT ((j7.claimed_instrument_id IS NULL OR j7.claimed_instrument_id IS r.instrument_id) AND (j7.claimed_required_judge_id IS NULL OR j7.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))) THEN 'identity_stale'
            WHEN EXISTS (SELECT 1 FROM judgment j7 WHERE j7.run_id = r.run_id AND j7.judge_id = jd.judge_id
                           AND j7.status = 'superseded') THEN 'superseded'
            WHEN EXISTS (SELECT 1 FROM judgment j7 WHERE j7.run_id = r.run_id AND j7.judge_id = jd.judge_id
                           AND j7.status = 'abandoned') THEN 'abandoned'
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
-- the active claim is an identity-current in-progress row: a claim bound to a since-corrected identity is not held
LEFT JOIN judgment cl ON cl.judgment_id = (
        SELECT j3.judgment_id FROM judgment j3 WHERE j3.run_id = r.run_id AND j3.judge_id = jd.judge_id
          AND j3.status = 'in_progress' AND ((j3.claimed_instrument_id IS NULL OR j3.claimed_instrument_id IS r.instrument_id) AND (j3.claimed_required_judge_id IS NULL OR j3.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
        ORDER BY j3.claimed_utc DESC LIMIT 1)
WHERE r.lifecycle_state IN ('generated', 'judging', 'judged', 'ingested')
  -- a superseded run is never pending; its successor is (the registry decision of 2026-10-04, migration 0025): an identity correction
  -- landing on a retired run whose complete judgment was claimed under the old identity must not list it as identity_stale beside
  -- its successor (no apostrophe and no bare table name in this comment: the reference scan strips quoted literals and reads identifiers)
  AND NOT (r.superseded_by_run_id IS NOT NULL)
  -- no identity-current judgment by J covers it: a covering judgment bound at claim to a since-corrected identity covers nothing
  AND NOT EXISTS (
        SELECT 1 FROM judgment jc
        WHERE jc.run_id = r.run_id AND jc.judge_id = jd.judge_id AND jc.status = 'complete' AND COALESCE(jc.n_items_judged, 0) + COALESCE(jc.n_judge_refusals, 0) + COALESCE(jc.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jc.run_id AND rm_err.judgment_id = jc.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
          AND ((jc.claimed_instrument_id IS NULL OR jc.claimed_instrument_id IS r.instrument_id) AND (jc.claimed_required_judge_id IS NULL OR jc.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id))));
