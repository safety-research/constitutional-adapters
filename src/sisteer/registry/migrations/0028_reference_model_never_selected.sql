-- 0028_reference_model_never_selected.sql
-- A run of the front-door WORKED EXAMPLE's key is never selected into the paper and never a paper row, whatever the roster or the rulings
-- say (the requirement that qwen35_4b_demo never be a result, 2026-10-08; a closing review's finding on dc54efe25, checked
-- against the live registry read-only; a revision of 2026-10-08: the predicate is the worked example's KEY SET, not
-- `paper_status reference`). Numbered 0028 in this tree -- max landed + 1 (0027 `decisive_first_source_local_anchor_solo_stand_in` is the
-- newest on main), re-checked at the merge as the migrations README says; renumbered before its gate if another branch's 0028 lands first
-- (no Python post-step, so nothing is keyed by the slug; the slug keeps the name the branch's tests were written against). ONE object
-- recreated, nothing else: no table is rebuilt, no column is added, no row is copied, no data is rewritten.
--
-- The finding: both selection CASEs tested the decisive `include` ruling BEFORE the missing-roster branch, so a completed run of a model no
-- roster names was still SELECTED, in_current_paper 1, when a model-unrestricted include covered it -- and the live registry carries one
-- (`r2v4-include-full-suite-bigcodebench-bfcl-appendix-own-instruments`, key-scoped on campaign / eval / item set / source, no model
-- term). A completed worked-example run matching it would have been a paper row.
-- The rule (schema_def.WORKED_EXAMPLE_MODEL_IDS / WORKED_EXAMPLE_MODEL_SQL, `r.model_id IN ('qwen35_4b_demo', 'qwen35_4b_demo_base')`):
--   * selection_state: a new branch right after `withdrawn` and BEFORE `include`: a worked-example run reads `not_in_roster` -- the label a
--     roster-less run reads already (no roster entry of any version names the key) -- unless a DECISIVE exclude / supersede ruling retired
--     it, which keeps reading `excluded` / `superseded` by name as for any run (retirement is the stronger fact).
--   * in_current_paper: a worked-example run reads 0 before the include branch, unconditionally.
-- Why a key set and not `paper_status`: `reference` means "not one of the paper's models", never "never a result"
-- -- the live `model` table carries 13 such rows, qwen35_35b_a3b, gemma4_31b and gemma4e2b_it among them -- and a reference run IS in
-- the frame when a roster entry or a ruling puts it there: the 35B's 26 full-suite BigCodeBench / BFCL runs (13 + 13, campaign
-- sdfbase_full22, the full item sets, sources bigcodebench_full / bfcl_full) read `selected` / in_current_paper 1 by that deliberate,
-- unexpired appendix include and KEEP doing so; a frame change for them is a ruling, never this object's. The first choice, a
-- `worked_example` value of paper_status on the two demo rows, would be a new value of the CHECK on `model` -- a table-rebuild migration
-- (migrations/README.md "Adding a vocabulary value"), which the landing window refuses without WINDOW_ALLOW_REBUILD -- so the allowed
-- fallback is taken: the key set is carried here instead, and a new worked-example key is added to the constant (and to
-- configs/registry/models.yaml) together.
-- Rows that change label: NONE live. Plain-table census of the live registry at version 27 (2026-10-08, read-only; the object itself was
-- not folded by this branch): 0 runs carry model_id qwen35_4b_demo or qwen35_4b_demo_base (the worked example runs --dry-run only; it has
-- no decode rows, and the live `model` table has no row of either key yet -- configs/registry/models.yaml's two rows load with this
-- branch); the reference models' 619 judged / ingested runs (qwen35_35b_a3b 484, the live-only drill model 135) keep their reading, the
-- 26 appendix runs included, and 0 roster entries name a reference model. is_current moves on no run: `selected` and `not_in_roster` are
-- both current states (schema_def.NON_CURRENT_SELECTION_STATES).
-- Test: tests/test_registry_views_0028.py (the file is the render against the v27 shape; the rule on a synthetic corpus: a completed
-- worked-example run under a model-unrestricted include reads not_in_roster / 0 while the same ruling selects the paper model's run AND a
-- reference model's run; a decisive exclude still retires it); tests/test_registry_seeds_demo_model.py pins the CASE order and the key
-- set against the seeds. The older files' tests take the code's `v_run_current` text back to 0027's
-- (tests/test_registry_migrations.py::_pre_0028_run_current).
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
       -- hold_out mark, a lifecycle of failed / cancelled / superseded / withdrawn, or a superseded_by_run_id. A decisive exclude /
       -- supersede ruling retires the run whatever its roster membership: since migration 0027 the ruling's branch precedes
       -- the roster branch in selection_state, so such a run reads excluded / superseded by name and the 0021 belt for the
       -- (not_in_roster, decisive_action) pair is no longer needed; the paper reads the rows of a current run only where the frame says
       -- so (hold_out keeps a row in the paper and still reads is_current 0; a member pass of a pooled reading, selection_state
       -- member_pass, is current and not a paper row, and so is a member pass whose own point carries no pooled reading of its
       -- source, selection_state unpooled_member, and so is a source-local none, selection_state source_local_anchor;
       -- since migration 0028 a run of a worked-example key -- WORKED_EXAMPLE_MODEL_IDS, the front door's qwen35_4b_demo -- reads
       -- not_in_roster before any include ruling can select it and is never a paper row, unless a decisive exclude / supersede retired
       -- it by name; a run of a `reference` checkpoint is in the frame by roster or ruling exactly as before)
       CASE WHEN selection_state IN ('superseded', 'withdrawn', 'excluded') OR held_out = 1
                 OR lifecycle_state IN ('failed', 'cancelled', 'superseded', 'withdrawn') OR superseded_by_run_id IS NOT NULL THEN 0 ELSE 1 END AS is_current,
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
             WHEN r.model_id IN ('qwen35_4b_demo', 'qwen35_4b_demo_base') AND COALESCE(d.action, '') NOT IN ('exclude', 'supersede') THEN 'not_in_roster'
             WHEN d.action = 'include' THEN 'selected'
             WHEN d.action = 'supersede' THEN 'superseded'
             WHEN d.action = 'exclude' THEN 'excluded'
             WHEN (r.object_id = 'none' AND (r.lane_source_legacy IN ('alphasteer_rebake', 'arditi_strength', 'arditi_allpos', 'arditi_vendor_mt', 'strength_mtdiff_s1', 'strength_lane', 'em/agentic/qwen35_4b_allpos', 'em/agentic/qwen35_4b_bl_vendor_leak') OR r.lane_source_legacy GLOB 'strength_lane_*' OR r.lane_source_legacy GLOB 'em/agentic/*_bl' OR r.lane_source_legacy GLOB 'em/agentic/*_bl__*-superseded') AND l.run_id IS NULL AND r.lifecycle_state <> 'superseded' AND r.superseded_by_run_id IS NULL) THEN 'source_local_anchor'
             WHEN pm.run_id IS NULL THEN 'not_in_roster'
             WHEN r.lifecycle_state = 'superseded' OR r.superseded_by_run_id IS NOT NULL OR l.run_id IS NOT NULL THEN 'superseded'
             WHEN (((r.lane_source_legacy = 'relu_paired' AND r.pass_label IN ('base_cap', 'ladder_rung', 'unrequested_rung')) OR (r.lane_source_legacy = 'sampled_lane' AND r.pass_label IN ('base_cap', 'ladder_rung'))) AND EXISTS (SELECT 1 FROM pooled_group pg WHERE pg.lane_source_legacy = r.lane_source_legacy AND pg.model_id = r.model_id AND pg.object_id = r.object_id AND pg.eval_id = r.eval_id AND pg.view_legacy IS r.view_legacy)) THEN 'member_pass'
             WHEN ((r.lane_source_legacy = 'relu_paired' AND r.pass_label IN ('base_cap', 'ladder_rung', 'unrequested_rung')) OR (r.lane_source_legacy = 'sampled_lane' AND r.pass_label IN ('base_cap', 'ladder_rung'))) THEN 'unpooled_member'
             WHEN r.lifecycle_state NOT IN ('judged', 'ingested') THEN 'not_in_roster'
             ELSE 'selected'
           END AS selection_state,
           CASE
             WHEN COALESCE(pc.indeterminate, 0) = 1 THEN 0
             WHEN w.run_id IS NOT NULL THEN 0
             WHEN r.model_id IN ('qwen35_4b_demo', 'qwen35_4b_demo_base') THEN 0
             WHEN r.lifecycle_state NOT IN ('judged', 'ingested') THEN 0
             WHEN d.action = 'include' THEN 1
             WHEN (r.object_id = 'none' AND (r.lane_source_legacy IN ('alphasteer_rebake', 'arditi_strength', 'arditi_allpos', 'arditi_vendor_mt', 'strength_mtdiff_s1', 'strength_lane', 'em/agentic/qwen35_4b_allpos', 'em/agentic/qwen35_4b_bl_vendor_leak') OR r.lane_source_legacy GLOB 'strength_lane_*' OR r.lane_source_legacy GLOB 'em/agentic/*_bl' OR r.lane_source_legacy GLOB 'em/agentic/*_bl__*-superseded') AND l.run_id IS NULL AND r.lifecycle_state <> 'superseded' AND r.superseded_by_run_id IS NULL) THEN 0
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
