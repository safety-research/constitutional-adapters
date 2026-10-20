-- 0013_roster_match_padded_instrument_by_name.sql
-- A roster SELECTOR never matches a PADDED instrument; a padded instrument is reached only by NAME.
-- Migration 0012 made prompt padding INSTRUMENT identity (`instrument.padding_tokens` + `padding_identity_sha256`; NULL = unpadded) and
-- roster v6 named the eight padded static instruments of qwen35_4b_it by `instrument_id` under `figP_padding/instruct_direct`, so the 24
-- padded runs (source `static_padding`: pasttense / codeattack / renellm / grok420 x {none, safety_si, constitution_si} x pads 16,384 /
-- 65,536) are `selected` through that comparison. But `v_roster_match`'s instrument SELECTOR branch knew nothing about padding: a padded
-- instrument satisfies every selector field of the eval's ordinary static entries (harness, thinking_mode, decode_source, scoring_rule,
-- a cap list that carries 16,384), so on the live frame at 10:14Z after window 8 every padded run ALSO matched fig1_headline /
-- fig2_jailbreak / fig5_frontier / figA_attacks_absolute / figQ_frontier / figQ_jailbreak / figS_degeneracy by selector (12 comparisons
-- per padded run, 288 (run, entry) pairs where 24 belong) -- the pooling the pad-as-instrument decision exists to prevent (the padded rows
-- differ in cap 16,384 and window 98,304 as well as the pad; the unpadded static rows are cap 4,096 / window 4,864).
-- Since this file the selector branch (`re.instrument_id IS NULL AND ...`) requires `i.padding_tokens IS NULL`: an unpadded instrument
-- matches selector entries exactly as before, a padded instrument matches only an entry that NAMES it (`re.instrument_id =
-- r.instrument_id`, unchanged). `v_paper_current`, `v_paper_long` and `v_roster_gaps` read `v_roster_match` by name and re-implement no
-- selector, so they follow without a text change. The roster's own selector grammar (`adapters/roster._instrument_selector`) is untouched.
-- View-only: no table changes, no rows copied. Exactly the one object below differs. Rendered by `registry migrate --plan --sql` against
-- a schema-version-12 database. Every writer refuses a database behind version 13: `registry migrate` in the read window right after landing.
-- Measured against the live database read-only before landing (the new v_roster_match / v_paper_current as TEMP copies over the live
-- tables, one read transaction): the numbers are in docs/registry/operations.md under "A roster selector never matches a padded instrument".
-- drift: v_roster_match differs from the code's SQL

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
           -- a selector never matches a PADDED instrument (0013): padding is instrument identity (0012) that no selector
           -- field carries, so a padded instrument satisfies the fields of the unpadded entries and would pool with them; it is
           -- reached only by name, through the branch above (no apostrophe in this comment: the reference scan strips quoted literals)
           OR (re.instrument_id IS NULL
               AND i.padding_tokens IS NULL
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
      -- operating_point: a filed stamp of 1, else the fallback decided 2026-09-22 (operating_point_basis_sql: a fallback kind at nominal
      -- 1.0, the all-positions vector at c = 0.3, an unsteered or prompted kind); a filed stamp of 0 is never at the point
      AND ((re.strength_rule = 'operating_point' AND ((CASE WHEN r.is_operating_point = 1 THEN 'stamped'
            WHEN r.is_operating_point IS NULL AND o.object_kind IN ('lora', 'lora_subtraction', 'composed', 'relu_steer', 'relu_steer_distilled_difference', 'alphasteer', 'static_r_arditi', 'vector', 'vector_all', 'diffmean', 'antidote')
                 AND ABS(r.strength_nominal - 1.0) < 1e-9 THEN 'fallback_nominal_1p0'
            WHEN r.is_operating_point IS NULL AND o.object_kind = 'vector_all'
                 AND ABS(r.strength_coefficient - 0.3) < 1e-9 THEN 'fallback_arditi_c0p3'
            WHEN o.object_kind IN ('none', 'system_instruction') THEN 'no_strength_axis'
            ELSE 'none' END) <> 'none'))
           OR (re.strength_rule GLOB 'fixed:*' AND r.strength_nominal = CAST(substr(re.strength_rule, 7) AS REAL))
           OR (re.strength_rule GLOB 'ladder:*' AND r.strength_nominal IS NOT NULL));
