-- 0008_item_headline_accepted_metric_ids.sql
-- `v_item_headline` accepts every headline spelling the eval's seed lists. `eval.headline_metric_id` names ONE metric id (NOT NULL,
-- CHECK-guarded) and the old join read `e.headline_metric_id = ir.metric_id`; strongreject_bon's is `asr` (the 512-store + 4,096-climb
-- reading at bon_n 75 -- the canonical best-of-N since 2026-09-11), but the producers that run the NATIVE 4,096 pass file the same
-- per-goal any-hit estimand as `asr_bon_cap4096` (bon_stores 48 runs, seedrep_cluster 24 runs on 2026-09-21; a 2026-09-19
-- ruling makes it one unit, threshold 0.5 both), so a consumer testing the declared id alone read the native-pass best-of-N as absent.
-- The seed now lists the ACCEPTED ids in order (`eval.extra_json.headline_metric_ids`, configs/registry/evals.yaml) and this object
-- resolves a run's headline as the most preferred listed id the run files current item results under, else the declared id
-- (paper.run_headline_metric_id is the same rule in Python; the preference test is row-scoped so a consumer's `run_id = ?` still
-- seeks one run). No table changes, no row moves, no existing column changes meaning; the one object that read the equality is
-- recreated. Rendered by `registry migrate --plan --sql` against a schema-version-7 database; only this object differs.
-- drift: v_item_headline differs from the code's SQL

DROP VIEW IF EXISTS v_item_headline;
CREATE VIEW v_item_headline AS
WITH accepted AS (
    -- the ACCEPTED headline metric ids of the eval in order of preference: extra_json.headline_metric_ids where the seed lists them
    -- (strongreject_bon: asr, then asr_bon_cap4096 -- the native 4,096 pass, the same per-goal any-hit unit), closed by the declared
    -- headline_metric_id (the only entry for every other eval); one row per (eval, metric id) at its best preference
    SELECT eval_id, metric_id, MIN(preference) AS preference FROM (
        SELECT e.eval_id, j.value AS metric_id, j.key AS preference
        FROM eval e, json_each(COALESCE(json_extract(e.extra_json, '$.headline_metric_ids'), '[]')) j
        UNION ALL
        SELECT e.eval_id, e.headline_metric_id, COALESCE(json_array_length(json_extract(e.extra_json, '$.headline_metric_ids')), 0)
        FROM eval e
    ) GROUP BY eval_id, metric_id
)
-- ROW-SCOPED on purpose: a per-run choice computed in its own CTE is not scoped by a consumer filter run_id = ? and probes every run in
-- the registry (10,007 runs, 8 M item rows on 2026-09-21: reading one run took minutes); the NOT EXISTS below costs nothing for the
-- most preferred id (no more-preferred candidate) and one indexed probe per row for a lower-preferred one
SELECT ir.run_id, ir.item_id, ir.trial_index, ir.sample_index, ir.turn_index, ir.judgment_id,
       ir.metric_id, ir.value AS score, ir.label, ir.api_pool, ir.created_utc, ir.judged_utc,
       f.at_token_cap, f.finish_reason, f.response_tokens, f.response_chars,
       f.degenerate_classifier, f.degenerate_contentless, f.degenerate_rep_ratio_loop, f.degenerate_zlib_collapse,
       f.degenerate_union5, f.loop_first512, f.rep_ratio, f.zlib_ratio, f.alnum_count, f.tail_loop, f.line_loop,
       f.decode_seed, f.generation_artifact_id, f.generation_offset, f.rationale_artifact_id, f.rationale_offset,
       ir.source_revision_id, ir.ingest_seq
FROM v_item_result_current ir
JOIN run r ON r.run_id = ir.run_id
JOIN accepted a ON a.eval_id = r.eval_id AND a.metric_id = ir.metric_id
LEFT JOIN v_item_flags_current f ON f.run_id = ir.run_id AND f.item_id = ir.item_id AND f.trial_index = ir.trial_index
     AND f.sample_index = ir.sample_index AND f.turn_index = ir.turn_index AND f.judgment_id = ir.judgment_id
WHERE NOT EXISTS (
    -- the headline of a run is the MOST preferred accepted id it files item results under (paper.run_headline_metric_id in SQL): a row
    -- under a less preferred id stands only where no more preferred one has current item results in the run
    SELECT 1 FROM accepted a2
    WHERE a2.eval_id = r.eval_id AND a2.preference < a.preference
      AND EXISTS (SELECT 1 FROM v_item_result_current ir2 WHERE ir2.run_id = r.run_id AND ir2.metric_id = a2.metric_id));

