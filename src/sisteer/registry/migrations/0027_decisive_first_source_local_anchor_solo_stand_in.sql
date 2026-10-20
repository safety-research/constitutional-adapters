-- 0027_decisive_first_source_local_anchor_solo_stand_in.sql
-- Three registry follow-ups; numbered 0027 in this tree -- max landed + 1 (0026
-- `twin_fold_view_legacy_value_with_draws` is the newest on main), re-checked at the merge as the migrations README says; renumbered
-- before its gate if another branch's 0027 lands first (no Python post-step, so nothing is keyed by the slug). Lands only inside a
-- landing window (scripts/registry/landing/window.sh).
--
-- (1) The DECISIVE supersede / exclude ruling precedes the roster branch in `v_run_current`'s selection_state CASE.
--     Rows that change label: every run a decisive key-, run-, object-, model- or eval-scoped ruling excluded or superseded and that NO
--     roster entry of the open round matches -- until this file they read `not_in_roster` beside `decisive_action` exclude / supersede;
--     from this file they read `excluded` / `superseded`, the spelling the same ruling already gave a roster-matched run. Live census
--     (one bounded fold of v_run_current, 2026-10-06 03:4xZ): 2,089 rows move -- 674 to `excluded`, 1,415 to `superseded` (r2_export
--     657 + 73, multiturn_grades 160 + 12, the source-less rows 140 + 86, evalcheck/gsm8k_paired200 139 + 2, alphasteer_rebake 33 +
--     137, sampled_lane 60 + 76, seedrep_battery 37 + 71, relu_paired 9 + 65, overrefusal_items 64 + 4, the shared-standup
--     comparison sources 50 excluded, ...). is_current UNCHANGED: every one of the 2,089 already read is_current 0 through the 0021
--     belt `(selection_state = 'not_in_roster' AND decisive_action IN ('exclude', 'supersede'))`, which this file retires from the
--     is_current CASE and from `why` (both spellings were non-current; the belt's `why` text `decisive ruling <action> (ruling <id>)`
--     becomes `selection_state excluded (ruling <id>)` / `selection_state superseded (ruling <id>)`, the texts a roster-matched run
--     already carried; `adapters.anchor_current._classify_why` keeps reading the old spelling in recorded decision logs). The producer's
--     own retirement (lifecycle superseded, superseded_by_run_id) and the later-run rule keep their roster-first place: this file moves the
--     RULING alone, and a roster-less run a later run superseded reads not_in_roster and STAYS current, exactly as before this file.
--     An INCOMPLETE or failed run a decisive ruling covers (never roster-matched: the match needs a covering judgment) reads the
--     ruling's label too, its `why` the ruling's instead of `lifecycle_state <state>` (is_current 0 either way; the census above counts
--     these rows). in_current_paper is unchanged (its CASE already read the decisive ruling before the roster test). Test:
--     tests/test_registry_views_0027.py; the later-run pins in tests/test_registry_views_0021.py read the new spelling.
-- (2) The `source_local_anchor` frame rule. An object-`none` run whose `lane_source_legacy` is in
--     vocab.SOURCE_LOCAL_ANCHOR_SOURCES (alphasteer_rebake, arditi_strength, arditi_allpos, arditi_vendor_mt, strength_mtdiff_s1, the
--     strength ladders `strength_lane` / `strength_lane_*`, em/agentic/qwen35_4b_allpos, the EM published-baselines panels em/agentic/*_bl
--     with their `__<tag>-superseded` copies and em/agentic/qwen35_4b_bl_vendor_leak) -- its own standup's unsteered reading, never the
--     paper's -- reads selection_state `source_local_anchor` (schema_def.source_local_anchor_sql, rendered into both CASEs): a branch
--     AFTER the decisive ruling branches (an include / exclude / supersede still wins), guarded by the later-run rule and the producer's
--     retirement (`l.run_id IS NULL AND lifecycle_state <> 'superseded' AND superseded_by_run_id IS NULL`, so a none a later run or its
--     producer retired keeps reading superseded -- alphasteer_rebake's 226 later-superseded nones are NOT resurrected) and BEFORE the
--     roster branch, so it relabels only a none that would otherwise read `selected` or `not_in_roster`. Rows that change label (one
--     bounded fold of v_run_current, 2026-10-06 03:5xZ): ~545 `selected` rows of the eight listed sources
--     (alphasteer_rebake 70, arditi_strength 117, arditi_allpos 25, strength_lane 150, strength_lane_base2it_native 90,
--     strength_lane_base2it_native_mmlu 6, strength_lane_lora_base2it_native_gemma_hidose 10, plus the three added groups' selected
--     nones) read `source_local_anchor` and leave the paper flag (in_current_paper 1 -> 0: the paper's unsteered reading is the
--     campaign's), and ~52 `not_in_roster` rows relabel (in_current_paper 0 unchanged). is_current UNCHANGED on every run: the state is
--     not a non-current one (schema_def.NON_CURRENT_SELECTION_STATES), so a relabelled `selected` / `not_in_roster` none keeps is_current 1
--     and stays an eligible anchor for its source's contrasts (adapters.anchor_current e1), and the guard keeps every retired none at 0.
--     No campaign source's row changes (vocab.CAMPAIGN_NONE_SOURCES names them with a reason; the 340 live none runs with a NULL source
--     are on neither list). A borrowed-reference re-anchoring (anchor_current tier 1c) that picked one of these nones as the paper's
--     roster-matched baseline no longer can (in_current_paper 0), and re-picks the campaign's none at the next pass, as intended.
--     Test: tests/test_registry_source_local_anchor.py.
-- (3) The shared-standup comparison sources' shared-vs-solo stand-in contrast gets its own anchor_kind `solo_stand_in`
--     (vocab.ANCHOR_KINDS; `_bcbw_common.ANCHOR_KIND_SOLO_STAND_IN`, `anchor_kind_for`): until this file the stand-in and the anchor
--     currency's tier-1c re-anchoring on another source's roster-matched `none` both filed `borrowed`, and `v_paper_long`'s one documented
--     tie-break (anchor_kind) could not tell a shared key's two contrasts apart. `run_contrast.anchor_kind` carries a CHECK over the
--     kinds (the live DDL: CHECK (anchor_kind IN ('same_standup_none', 'sibling_none', 'borrowed', 'full_dose')); a value a fold writes
--     outside it fails the whole revision), and SQLite cannot alter a CHECK in place, so this file REBUILDS run_contrast by the documented
--     dance (migrations/README.md, 0002 the worked example): CREATE run_contrast__new from the code's DDL, INSERT ... SELECT every column
--     in primary-key order, DROP, RENAME, the six sentinel-column FK triggers recreated, ANALYZE. Measured live (read-only, 2026-10-06):
--     11,869 rows (same_standup_none 8,814, sibling_none 1,587, full_dose 933, borrowed 535) -- a trivial copy, seconds. No row's value
--     changes in the migration: the 535 live `borrowed` rows keep their kind until the six shared adapters re-emit under their bumped
--     EMISSION_VERSION (bfcl_full_shared / bigcodebench_full_shared 3 -> 4, their _l2 / _l3 readers 2 -> 3) in the window's adapters step,
--     when the stand-in rows re-file as `solo_stand_in` and the re-anchored steering contrasts keep `borrowed`. WINDOW NOTE: window.sh
--     exits 15 on any `rebuild table` plan line unless WINDOW_ALLOW_REBUILD=1 names a reviewed rebuild -- the operator of the landing
--     window sets it for this file (the plan line reads `rebuild table run_contrast: 11869 rows to copy`). No selection label moves here.
-- Rendered by `registry migrate --plan --sql` against a schema-version-26 database (0001..0026 applied to an empty database by the
-- runner); tests/test_registry_views_0027.py pins the plan.
-- drift: table run_contrast: column anchor_kind CHECK (anchor_kind IN (...)) lacks 'solo_stand_in'
-- drift: v_run_current differs from the code's SQL

-- rebuild run_contrast: SQLite cannot alter a constraint in place (migrations/README.md)
CREATE TABLE run_contrast__new (
  run_id TEXT NOT NULL REFERENCES run(run_id),
  anchor_run_id TEXT NOT NULL REFERENCES run(run_id),
  judgment_id TEXT NOT NULL DEFAULT '',
  metric_id TEXT NOT NULL,
  threshold REAL NOT NULL DEFAULT -1,
  computation_id TEXT NOT NULL DEFAULT '',
  anchor_judgment_id TEXT NOT NULL DEFAULT '',
  anchor_kind TEXT NOT NULL CHECK (anchor_kind IN ('same_standup_none', 'sibling_none', 'borrowed', 'full_dose', 'solo_stand_in')),
  delta REAL,  -- why-null: undefined when n_pairs is 0
  delta_lo REAL,  -- why-null: no interval computed
  delta_hi REAL,  -- why-null: no interval computed
  delta_p REAL,  -- why-null: no p-value computed
  delta_method TEXT NOT NULL,
  effect_size REAL,  -- why-null: no effect size computed
  effect_kind TEXT CHECK (effect_kind IN ('cohens_h', 'dz')),  -- why-null: null when effect_size is null
  n_pairs INTEGER NOT NULL,
  computed_by TEXT NOT NULL CHECK (computed_by IN ('producer', 'registry')),
  computed_utc TEXT CHECK (computed_utc IS NULL OR computed_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),  -- why-null: unknown for legacy rows
  field_versions_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(field_versions_json)),  -- {column: [written_utc, record_sha256] of the record that last set it} — the FULL order key (S2).
  extra_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(extra_json)),  -- Unknown record fields, never dropped (Fixed decision 13).
  record_uri TEXT NOT NULL,  -- The record file (logical path) that produced the current row.
  ingest_seq INTEGER NOT NULL,  -- The producing FILE's rank in the ledger's arrival order (record_ingest: arrival_utc, then first_written_utc, sha, uri); derived, never allocated — every row above a head a reader has seen is new to it.
  PRIMARY KEY (run_id, anchor_run_id, judgment_id, metric_id, threshold, computation_id),
  CONSTRAINT run_contrast_metric_id_not_generic CHECK (metric_id NOT IN ('score', 'value', 'metric', 'result', 'accuracy', 'mean', 'count', 'total', 'number', 'frac', 'fraction', 'share', 'pct', 'percent', 'stat', 'statistic', 'measure', 'outcome', 'degenerate', 'degeneration', 'degeneracy', 'degen', 'loop', 'looping', 'loops', 'at_cap', 'capped', 'truncated', 'truncation', 'collapse', 'frac_degenerate', 'degenerate_rate', 'degenerate_frac', 'degenerate_fraction', 'loop_rate', 'frac_loop', 'loop_frac', 'at_cap_rate', 'frac_at_cap', 'at_cap_frac', 'at_cap_fraction', 'classifier', 'frac_classifier', 'classifier_rate', 'classifier_frac', 'classifier_share', 'degen_classifier_share', 'contentless', 'frac_contentless', 'contentless_rate', 'contentless_frac', 'contentless_share', 'degen_contentless_share', 'rep_ratio_loop', 'frac_rep_ratio_loop', 'rep_ratio_loop_rate', 'rep_ratio_loop_frac', 'rep_ratio_loop_share', 'degen_rep_ratio_loop_share', 'zlib_collapse', 'frac_zlib_collapse', 'zlib_collapse_rate', 'zlib_collapse_frac', 'zlib_collapse_share', 'degen_zlib_collapse_share', 'union5', 'frac_union5', 'union5_rate', 'union5_frac', 'union5_share', 'degen_union5_share', 'at_token_cap', 'frac_at_token_cap', 'at_token_cap_rate', 'at_token_cap_frac', 'at_token_cap_share', 'degen_at_token_cap_share', 'at_token_cap_retokenized', 'frac_at_token_cap_retokenized', 'at_token_cap_retokenized_rate', 'at_token_cap_retokenized_frac', 'at_token_cap_retokenized_share', 'degen_at_token_cap_retokenized_share', 'at_token_cap_finish_reason', 'frac_at_token_cap_finish_reason', 'at_token_cap_finish_reason_rate', 'at_token_cap_finish_reason_frac', 'at_token_cap_finish_reason_share', 'degen_at_token_cap_finish_reason_share', 'loop_first512', 'frac_loop_first512', 'loop_first512_rate', 'loop_first512_frac', 'loop_first512_share', 'degen_loop_first512_share', 'defended_degenerate', 'frac_defended_degenerate', 'defended_degenerate_rate', 'defended_degenerate_frac', 'defended_degenerate_share', 'degen_defended_degenerate_share', 'tail_loop', 'frac_tail_loop', 'tail_loop_rate', 'tail_loop_frac', 'tail_loop_share', 'degen_tail_loop_share', 'line_loop', 'frac_line_loop', 'line_loop_rate', 'line_loop_frac', 'line_loop_share', 'degen_line_loop_share', 'truncation_only', 'frac_truncation_only', 'truncation_only_rate', 'truncation_only_frac', 'truncation_only_share', 'degen_truncation_only_share', 'loop_without_cap', 'frac_loop_without_cap', 'loop_without_cap_rate', 'loop_without_cap_frac', 'loop_without_cap_share', 'degen_loop_without_cap_share', 'loop_or_at_cap', 'frac_loop_or_at_cap', 'loop_or_at_cap_rate', 'loop_or_at_cap_frac', 'loop_or_at_cap_share', 'degen_loop_or_at_cap_share', 'at_cap_share', 'loop_share', 'degenerate_share', 'degenerate_share_union5', 'at_cap_share_retokenized', 'at_cap_share_finish_reason', 'frac_degenerate_classifier', 'frac_degenerate_contentless', 'frac_degenerate_rep_ratio_loop', 'frac_degenerate_zlib_collapse', 'frac_degenerate_union5'))
);
INSERT INTO run_contrast__new (run_id, anchor_run_id, judgment_id, metric_id, threshold, computation_id, anchor_judgment_id, anchor_kind, delta, delta_lo, delta_hi, delta_p, delta_method, effect_size, effect_kind, n_pairs, computed_by, computed_utc, field_versions_json, extra_json, record_uri, ingest_seq) SELECT run_id, anchor_run_id, judgment_id, metric_id, threshold, computation_id, anchor_judgment_id, anchor_kind, delta, delta_lo, delta_hi, delta_p, delta_method, effect_size, effect_kind, n_pairs, computed_by, computed_utc, field_versions_json, extra_json, record_uri, ingest_seq FROM run_contrast ORDER BY run_id, anchor_run_id, judgment_id, metric_id, threshold, computation_id;
DROP TABLE run_contrast;
ALTER TABLE run_contrast__new RENAME TO run_contrast;
CREATE TRIGGER trg_run_contrast_judgment_id_fk_ins BEFORE INSERT ON run_contrast
WHEN NEW.judgment_id <> '' AND NOT EXISTS (SELECT 1 FROM judgment WHERE judgment_id = NEW.judgment_id)
BEGIN
  SELECT RAISE(ABORT, 'run_contrast.judgment_id names no judgment.judgment_id (sentinel-column FK)');
END;
CREATE TRIGGER trg_run_contrast_judgment_id_fk_upd BEFORE UPDATE OF judgment_id ON run_contrast
WHEN NEW.judgment_id <> '' AND NOT EXISTS (SELECT 1 FROM judgment WHERE judgment_id = NEW.judgment_id)
BEGIN
  SELECT RAISE(ABORT, 'run_contrast.judgment_id names no judgment.judgment_id (sentinel-column FK)');
END;
CREATE TRIGGER trg_run_contrast_computation_id_fk_ins BEFORE INSERT ON run_contrast
WHEN NEW.computation_id <> '' AND NOT EXISTS (SELECT 1 FROM computation WHERE computation_id = NEW.computation_id)
BEGIN
  SELECT RAISE(ABORT, 'run_contrast.computation_id names no computation.computation_id (sentinel-column FK)');
END;
CREATE TRIGGER trg_run_contrast_computation_id_fk_upd BEFORE UPDATE OF computation_id ON run_contrast
WHEN NEW.computation_id <> '' AND NOT EXISTS (SELECT 1 FROM computation WHERE computation_id = NEW.computation_id)
BEGIN
  SELECT RAISE(ABORT, 'run_contrast.computation_id names no computation.computation_id (sentinel-column FK)');
END;
CREATE TRIGGER trg_run_contrast_anchor_judgment_id_fk_ins BEFORE INSERT ON run_contrast
WHEN NEW.anchor_judgment_id <> '' AND NOT EXISTS (SELECT 1 FROM judgment WHERE judgment_id = NEW.anchor_judgment_id)
BEGIN
  SELECT RAISE(ABORT, 'run_contrast.anchor_judgment_id names no judgment.judgment_id (sentinel-column FK)');
END;
CREATE TRIGGER trg_run_contrast_anchor_judgment_id_fk_upd BEFORE UPDATE OF anchor_judgment_id ON run_contrast
WHEN NEW.anchor_judgment_id <> '' AND NOT EXISTS (SELECT 1 FROM judgment WHERE judgment_id = NEW.anchor_judgment_id)
BEGIN
  SELECT RAISE(ABORT, 'run_contrast.anchor_judgment_id names no judgment.judgment_id (sentinel-column FK)');
END;
ANALYZE run_contrast;

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
       -- source, selection_state unpooled_member, and so is a source-local none, selection_state source_local_anchor)
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
