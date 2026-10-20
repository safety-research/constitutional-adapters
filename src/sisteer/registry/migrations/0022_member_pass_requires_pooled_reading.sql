-- 0022_member_pass_requires_pooled_reading.sql
-- A registry ruling of 2026-10-02: a member pass is a member only where its source filed a pooled reading at its own point;
-- elsewhere it is an `unpooled_member`, not a paper row either -- one change of ONE `v_*` object (`v_run_current`) and nothing else: no
-- table moves, no row is copied, the object's columns are unchanged. Rendered by `registry migrate --plan --sql` against a
-- schema-version-21 database (0001..0021 applied to an empty database by `registry init`); the planner reports one recreate and no
-- rebuild.
-- The defect (live, schema 21, read-only, 2026-10-02 21:10Z): migration 0021 rendered `member_pass` from
-- vocab.MEMBER_PASS_LABELS_BY_SOURCE alone (`sampled_lane`: `base_cap`, `ladder_rung`; `relu_paired`: `base_cap`, `ladder_rung`,
-- `unrequested_rung`), so every roster-matched run of a listed (source, pass) read member_pass whether or not its source had filed the
-- pooled reading the paper reads for its POINT -- and the sources pool by point, (model, object, eval, view_legacy), not by (model,
-- object, eval). 406 complete `sampled_lane` runs read member_pass at a point with no pooled run of their source: 266 in a (model,
-- object, eval) carrying no pooled run at all (264 such groups; mostly llama8b_it and nemotron3nano4b on the four static attack evals,
-- with both the 4,096 and the 8,192 caps present, plus a few gemma4e4b_it / qwen35_4b_it groups) and 140 beside a pooled run of ANOTHER
-- point of the same (model, object, eval). The first draft of this file read those runs selected / in_current_paper 1, which the paper
-- side's pinned renderer contract contradicts (paper.paper_frame_exclusion and the paper's pinned load_rows keep `sampled_lane`'s
-- `pooled_t0_t1` pass alone: the paper's reading of a sampled point is its pooled run), so the frame would have claimed paper
-- rows the rendered paper never carries; the owner fixed the design below instead (review of 358af8e6d).
-- The rule: `member_pass` requires a pooled reading of the SAME source at the run's own point. The rendered predicate
-- (schema_def.member_pass_sql, once, in SELECTION_STATE_SQL) is the per-(source, pass) table half (schema_def.member_label_sql, 0021's
-- text) AND EXISTS (SELECT 1 FROM pooled_group pg WHERE pg.lane_source_legacy = r.lane_source_legacy AND pg.model_id = r.model_id AND
-- pg.object_id = r.object_id AND pg.eval_id = r.eval_id AND pg.view_legacy IS r.view_legacy), where `pooled_group` is a new MATERIALIZED
-- CTE closing schema_def.selection_ctes_sql (schema_def.POOLED_GROUP_CTE): SELECT DISTINCT lane_source_legacy, model_id, object_id,
-- eval_id, view_legacy FROM run WHERE pass_label IN ('pooled_t0_t1', 'pooled_final_rung') -- the labels of vocab.POOLED_PASS_LABELS,
-- rendered so the SQL and the tuple cannot drift; view_legacy pairs with IS, so a pooled run filing no point string marks the unlabeled
-- members and no other. ANY lifecycle or selection state of the pooled run marks: one a later pooled run superseded, one a ruling
-- excluded or one withdrawn still marks its members, because the members are never the paper reading once the source filed a pooled one.
-- A labeled run whose point carries no pooled reading of its source reads the NEW selection_state `unpooled_member` (vocab.SELECTION_STATES;
-- the next branch, the table half alone): a current measurement whose point's pooled reading the source has not filed; not a paper row;
-- the measurable gap the adapter branch's pooled-emission repair closes (adapter plan 10). Both member states sit where 0021's member
-- branch sat -- after the superseded and excluded branches, before the incomplete-lifecycle branch; an include ruling still selects --
-- and IN_CURRENT_PAPER_SQL reads the table half alone right after the include branch (0021's text: both states are 0), so the flag agrees
-- with the pinned renderer contract on every labeled run. is_current and why are untouched (neither state is non-current); an
-- incomplete labeled run reads not_in_roster by the lifecycle rule as before.
-- Live, read-only, 2026-10-02, 0021's body against this one on the same 12,401 runs: exactly 406 runs move, every one member_pass ->
-- unpooled_member (`sampled_lane`; `base_cap` at 4,096: 132 trial 0 + 132 trial 1; `ladder_rung` at 8,192: 71 + 71; llama8b_it 206,
-- nemotron3nano4b 140, qwen35_4b_it 37, gemma4e4b_it 23; by (pass, model): `base_cap` llama8b_it 120 / nemotron3nano4b 104 /
-- qwen35_4b_it 24 / gemma4e4b_it 16, `ladder_rung` 86 / 36 / 13 / 7; 266 of them in a (model, object, eval) with no pooled run, 140
-- beside another point's; all ingested); in_current_paper, is_current, why and the decisive ruling move on NO run. In the paper 3,444
-- and selected 3,563 before and after; member_pass 940 -> 534 (`sampled_lane` 406: `ladder_rung` 222, `base_cap` 184; `relu_paired`
-- 128: `base_cap` 87, `ladder_rung` 41); unpooled_member 0 -> 406 (`base_cap` 264, `ladder_rung` 142); is_current = 1 6,326 both. The
-- 528 pooled runs stand at 498 points (446 four-column groups; none files a NULL view_legacy, nor does any labeled run); of the 2,465
-- labeled runs 1,302 sit at a pooled point, 478 (all complete) in a pooled four-column group at an unpooled point, 685 in a group with
-- no pooled run. `relu_paired`'s 128 members all sit at pooled points (0 unpooled, 0 moved); the 82 include-ruled labeled runs, the
-- pooled runs and every unlabeled run read as before. The fold reads in 62.2 s against 53.6 s for 0021's body on the live registry (the
-- CTE is one pass over `run`; the probe runs once per candidate row), within the 1.2x budget.
-- The body below is the render from its drift line; tests/test_registry_migrations.py::test_0022_... asserts the committed body IS
-- that render against the version-21 shape, that the plan is one recreate, and that the object and the two that read it are
-- selectable afterwards; the behavior is tests/test_registry_views_0022.py's.
-- drift: v_run_current differs from the code's SQL

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
    -- and a supersession that held before the key grew still holds
    SELECT r.run_id
    FROM run r JOIN run r2 ON r2.model_id = r.model_id AND r2.object_id = r.object_id AND r2.eval_id = r.eval_id
        AND r2.instrument_id = r.instrument_id AND r2.judge_id IS r.judge_id AND r2.run_id <> r.run_id
        AND r2.strength_nominal IS r.strength_nominal AND r2.strength_coefficient IS r.strength_coefficient
        AND r2.operating_point_id_legacy IS r.operating_point_id_legacy
        AND r2.token_cap_tokens IS r.token_cap_tokens AND r2.trial_index IS r.trial_index AND r2.pass_label IS r.pass_label
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
