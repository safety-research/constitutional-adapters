-- 0006_completion_flag_contrast_cap.sql
-- Two additive column families on the consumer-facing objects; no table changes, no row moves, no existing column changes meaning.
-- (1) `is_complete` beside `lifecycle_state` in v_run_wide, v_producer_key, v_paper_current and v_paper_long: 1 when
-- lifecycle_state IN ('judged', 'ingested'), else 0 (schema_def.is_complete_sql / COMPLETE_LIFECYCLE_STATES). The lifecycle is
-- queued -> running -> generated -> (judging) -> judged -> ingested; `ingested` FOLLOWS `judged` and means the run is complete and
-- stamped (the harvester stamps every complete run `ingested`, S6), so at rest a complete row reads `ingested` and `judged` is the
-- transient state before the stamp. Two sessions searched `lifecycle_state = 'judged'` for finished runs on 2026-09-21 and found
-- nothing. On the live registry at this migration: 3,893 ingested, 4,946 judged (complete under the flag, the stamp still owed),
-- 581 superseded, 197 queued, 16 running, 5 generated.
-- (2) `anchor_token_cap_tokens` and `cap_matched` on v_paper_long beside the run_contrast columns: the anchor run's token cap
-- (`LEFT JOIN run ra ON ra.run_id = rc.anchor_run_id`) and 1 when it equals the run's own, 0 when it differs, NULL when the row
-- carries no anchor or either cap is NULL (run.token_cap_tokens is NOT NULL, so on the frame NULL means no anchor). A consumer
-- reading `delta` off the frame never joined both runs' caps, so a pair whose caps differ was invisible: the paired dose ladder
-- differenced cap-512 steered runs against cap-2048 anchors on 3 of its 5 steps (it files no run_contrast row yet), and a
-- registry audit found 202 of 8,640 filed contrasts pairing different caps, every one an escalation-ladder pair, most often a
-- cap-8192 run against a cap-4096 or cap-2048 anchor (2026-09-21). run_contrast itself is unchanged.
-- Rendered by `registry migrate --plan --sql` against a schema-version-5 database; only these four objects differ.
-- drift: v_run_wide differs from the code's SQL
-- drift: v_producer_key differs from the code's SQL
-- drift: v_paper_current differs from the code's SQL
-- drift: v_paper_long differs from the code's SQL

DROP VIEW IF EXISTS v_paper_current;
CREATE VIEW v_paper_current AS
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
    SELECT r.run_id
    FROM run r JOIN run r2 ON r2.model_id = r.model_id AND r2.object_id = r.object_id AND r2.eval_id = r.eval_id
        AND r2.instrument_id = r.instrument_id AND r2.judge_id IS r.judge_id AND r2.run_id <> r.run_id
        AND r2.strength_nominal IS r.strength_nominal AND r2.strength_coefficient IS r.strength_coefficient
        AND r2.operating_point_id_legacy IS r.operating_point_id_legacy
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
           MAX(CASE WHEN action = 'hold_out' THEN 1 ELSE 0 END) AS held_out
    FROM covered GROUP BY run_id
),
per_run_match AS MATERIALIZED (
    SELECT run_id, json_group_array(DISTINCT comparison_id) AS comparison_ids_json, json_group_array(DISTINCT roster_entry_id) AS roster_entry_ids_json
    FROM matched GROUP BY run_id
),
later_runs AS MATERIALIZED (SELECT DISTINCT run_id FROM later)
SELECT r.run_id, r.campaign, r.model_id, r.object_id, r.eval_id, r.instrument_id, r.judge_id, r.lifecycle_state,
       -- 1 once the run is judged or ingested (a complete run reads `ingested` at rest, `judged` before the harvester stamps it)
       CASE WHEN r.lifecycle_state IN ('judged', 'ingested') THEN 1 ELSE 0 END AS is_complete,
       r.strength_nominal, r.is_operating_point, r.ingest_seq,
       (SELECT round_id FROM open_round) AS round_id,
       COALESCE(pm.comparison_ids_json, '[]') AS comparison_ids_json,
       COALESCE(pm.roster_entry_ids_json, '[]') AS roster_entry_ids_json,
       COALESCE(pc.ruling_ids_json, '[]') AS ruling_ids_json,
       d.ruling_id AS decisive_ruling_id,
       d.action AS decisive_action,
       COALESCE(pc.held_out, 0) AS held_out,
       CASE
         WHEN COALESCE(pc.indeterminate, 0) = 1 THEN 'indeterminate'
         WHEN w.run_id IS NOT NULL THEN 'withdrawn'
         WHEN d.action = 'include' THEN 'selected'
         WHEN pm.run_id IS NULL THEN 'not_in_roster'
         WHEN r.lifecycle_state = 'superseded' OR r.superseded_by_run_id IS NOT NULL
              OR d.action = 'supersede' OR l.run_id IS NOT NULL THEN 'superseded'
         WHEN d.action = 'exclude' THEN 'excluded'
         WHEN r.lifecycle_state NOT IN ('judged', 'ingested') THEN 'not_in_roster'
         ELSE 'selected'
       END AS selection_state,
       CASE
         WHEN COALESCE(pc.indeterminate, 0) = 1 THEN 0
         WHEN w.run_id IS NOT NULL THEN 0
         WHEN r.lifecycle_state NOT IN ('judged', 'ingested') THEN 0
         WHEN d.action = 'include' THEN 1
         WHEN pm.run_id IS NULL THEN 0
         WHEN r.lifecycle_state = 'superseded' OR r.superseded_by_run_id IS NOT NULL THEN 0
         WHEN d.action IN ('exclude', 'supersede') THEN 0
         WHEN l.run_id IS NOT NULL THEN 0
         ELSE 1
       END AS in_current_paper
FROM run r
LEFT JOIN per_run_cover pc ON pc.run_id = r.run_id
LEFT JOIN per_run_match pm ON pm.run_id = r.run_id
LEFT JOIN decision d ON d.run_id = r.run_id
LEFT JOIN withdrawn w ON w.run_id = r.run_id
LEFT JOIN later_runs l ON l.run_id = r.run_id;
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
                          WHEN re.judge_id IS NOT NULL THEN jm.judge_id = re.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                          WHEN re.judge_selector_json IS NOT NULL THEN j.judge_id IS NOT NULL AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                               AND (json_extract(re.judge_selector_json, '$.judge_model') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.judge_model') = j.judge_model)
                               AND (json_extract(re.judge_selector_json, '$.rubric_name') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.rubric_name') = j.rubric_name)
                          ELSE (rm.judgment_id = '' AND i.judge_id IS NULL)
                               OR (jm.judge_id IS NOT NULL AND jm.judge_id IS i.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0))
                        END));
DROP VIEW IF EXISTS v_producer_key;
CREATE VIEW v_producer_key AS
SELECT producer_run_key_legacy, producer_object_key_legacy, run_id, campaign, eval_id, instrument_id, lifecycle_state,
       CASE WHEN lifecycle_state IN ('judged', 'ingested') THEN 1 ELSE 0 END AS is_complete
FROM run;
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

