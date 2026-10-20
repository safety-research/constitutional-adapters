-- 0016_roster_version_scope.sql
-- The roster VERSION scopes the roster match (roster v9, 2026-09-24). A roster entry's identity is a uuid5 over
-- its selector fields, so a version that re-shapes an entry's selector mints a NEW roster_entry_id while the earlier id stays in the table
-- (roster_entry has no version and no withdrawal: 84 ids left behind by v2, 372 by v4, 124 by v5, 580 in all, 90 of them
-- still matching a run on 2026-09-24). Until this file that was harmless: the left-behind ids were phantom gaps, or matched the same runs their
-- replacements did. Roster v9 re-shapes 1,332 entries so that every single-fit comparison reads a fitted object's BASE-checkpoint fit served
-- on the instruct checkpoint (`transfer_kind base_to_instruct` + `fit_model_id <model>_base`) instead of the IT-model fit -- the rule
-- to use the base-to-instruct fit wherever one exists, as PER-COMPARISON roster entries -- and the 1,332 v8 ids it leaves behind would
-- keep matching the IT-model fits under the very comparisons v9 re-points, so v9 would ADD the base fit beside the IT fit instead of
-- replacing it. Since this file:
-- (a) `paper_round.roster_version` and `roster_entry.roster_version` (both nullable) carry the roster version whose `roster load` last wrote
--     the row (adapters/roster.py stamps both from the yaml's meta.roster_version; a load re-writes every identity-stable entry, so every
--     entry the current version expands carries the current version, and an entry it no longer expands keeps its older stamp);
-- (b) `v_roster_match` admits an entry of the open round only when the round's paper_round row carries no version (a round loaded before this
--     file: every entry matches, as before) or the entry's stamp equals it; `v_roster_gaps` lists gaps among those entries alone. v_paper_current,
--     v_paper_long and v_paper_objects read v_roster_match by name and follow without a text change.
-- Rebuilds paper_round (1 row) and roster_entry (7,643 rows on the live database) to add the column (SQLite cannot add a column ahead of the
-- run-scale tail, whose last column must be ingest_seq); the two rebuilt tables are run-scale and copy in under a second. The FIRST load after
-- this file (roster v9) stamps the round; between the migration and that load every entry keeps matching exactly as under 0015. Rendered by
-- `registry migrate --plan --sql` against a schema-version-15 database; measured read-only against the live database before landing (the
-- numbers are in docs/registry/operations.md under "The roster version scopes the roster match").

-- rebuild paper_round: SQLite cannot alter a constraint in place (migrations/README.md)
-- column roster_version is new in the code: nullable: NULL
CREATE TABLE paper_round__new (
  round_id TEXT NOT NULL,
  opened_utc TEXT NOT NULL CHECK (opened_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),
  closed_utc TEXT CHECK (closed_utc IS NULL OR closed_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),  -- why-null: the open round has not closed
  description TEXT NOT NULL,
  paper_build_commit TEXT CHECK (paper_build_commit IS NULL OR paper_build_commit GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]' OR paper_build_commit = 'unverified'),  -- why-null: the paper-figures sha roster v1 was transcribed from is pinned after transcription
  roster_version TEXT,  -- why-null: a round whose roster loaded before migration 0016 carries none; every roster_entry of the round then matches, whatever its own stamp
  field_versions_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(field_versions_json)),  -- {column: [written_utc, record_sha256] of the record that last set it} — the FULL order key (S2).
  extra_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(extra_json)),  -- Unknown record fields, never dropped (Fixed decision 13).
  record_uri TEXT NOT NULL,  -- The record file (logical path) that produced the current row.
  ingest_seq INTEGER NOT NULL,  -- The producing FILE's rank in the ledger's arrival order (record_ingest: arrival_utc, then first_written_utc, sha, uri); derived, never allocated — every row above a head a reader has seen is new to it.
  PRIMARY KEY (round_id)
);
INSERT INTO paper_round__new (round_id, opened_utc, closed_utc, description, paper_build_commit, field_versions_json, extra_json, record_uri, ingest_seq) SELECT round_id, opened_utc, closed_utc, description, paper_build_commit, field_versions_json, extra_json, record_uri, ingest_seq FROM paper_round ORDER BY round_id;
DROP TABLE paper_round;
ALTER TABLE paper_round__new RENAME TO paper_round;
ANALYZE paper_round;

-- rebuild roster_entry: SQLite cannot alter a constraint in place (migrations/README.md)
-- column roster_version is new in the code: nullable: NULL
CREATE TABLE roster_entry__new (
  roster_entry_id TEXT NOT NULL,
  round_id TEXT NOT NULL REFERENCES paper_round(round_id),
  comparison_id TEXT NOT NULL,
  model_id TEXT NOT NULL REFERENCES model(model_id),
  object_id TEXT REFERENCES object(object_id),  -- why-null: the object may be chosen by rule (object_selector_json; CHECK one of the two)
  object_selector_json TEXT CHECK (object_selector_json IS NULL OR json_valid(object_selector_json)),  -- why-null: null when object_id names the object (CHECK one of the two)
  eval_id TEXT NOT NULL REFERENCES eval(eval_id),
  instrument_id TEXT REFERENCES instrument(instrument_id),  -- why-null: the roster may select an instrument FAMILY (instrument_selector_json; C2; CHECK one of the two)
  instrument_selector_json TEXT CHECK (instrument_selector_json IS NULL OR json_valid(instrument_selector_json)),  -- why-null: null when instrument_id names the instrument (CHECK one of the two)
  judge_id TEXT REFERENCES judge(judge_id),  -- why-null: rule-scored comparisons have no judge; or the judge is chosen by rule (judge_selector_json)
  judge_selector_json TEXT CHECK (judge_selector_json IS NULL OR json_valid(judge_selector_json)),  -- why-null: null when judge_id names the judge or the comparison is rule-scored (C2)
  computation_id TEXT REFERENCES computation(computation_id),  -- why-null: raw producer values name no computation
  strength_rule TEXT NOT NULL CHECK (strength_rule = 'operating_point' OR strength_rule GLOB 'fixed:*' OR strength_rule GLOB 'ladder:*'),  -- `operating_point` | `fixed:<x>` | `ladder:<record_id>` (every strength level the record carries).
  source_citation TEXT NOT NULL,
  inferred INTEGER NOT NULL DEFAULT 0 CHECK (inferred IN (0, 1)),
  drawn INTEGER NOT NULL DEFAULT 1 CHECK (drawn IN (0, 1)),  -- 0 = required by the gap map but not drawn by the figure at the transcribed sha.
  roster_version TEXT,  -- why-null: an entry written by a roster load before migration 0016 carries none (it matches while its round's paper_round row carries none)
  field_versions_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(field_versions_json)),  -- {column: [written_utc, record_sha256] of the record that last set it} — the FULL order key (S2).
  extra_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(extra_json)),  -- Unknown record fields, never dropped (Fixed decision 13).
  record_uri TEXT NOT NULL,  -- The record file (logical path) that produced the current row.
  ingest_seq INTEGER NOT NULL,  -- The producing FILE's rank in the ledger's arrival order (record_ingest: arrival_utc, then first_written_utc, sha, uri); derived, never allocated — every row above a head a reader has seen is new to it.
  PRIMARY KEY (roster_entry_id),
  CONSTRAINT roster_entry_object_xor_selector CHECK ((object_id IS NULL) <> (object_selector_json IS NULL)),
  CONSTRAINT roster_entry_instrument_xor_selector CHECK ((instrument_id IS NULL) <> (instrument_selector_json IS NULL)),
  CONSTRAINT roster_entry_judge_not_both CHECK (NOT (judge_id IS NOT NULL AND judge_selector_json IS NOT NULL))
);
INSERT INTO roster_entry__new (roster_entry_id, round_id, comparison_id, model_id, object_id, object_selector_json, eval_id, instrument_id, instrument_selector_json, judge_id, judge_selector_json, computation_id, strength_rule, source_citation, inferred, drawn, field_versions_json, extra_json, record_uri, ingest_seq) SELECT roster_entry_id, round_id, comparison_id, model_id, object_id, object_selector_json, eval_id, instrument_id, instrument_selector_json, judge_id, judge_selector_json, computation_id, strength_rule, source_citation, inferred, drawn, field_versions_json, extra_json, record_uri, ingest_seq FROM roster_entry ORDER BY roster_entry_id;
DROP TABLE roster_entry;
ALTER TABLE roster_entry__new RENAME TO roster_entry;
CREATE INDEX idx_roster_entry_key ON roster_entry (round_id, model_id, eval_id);
ANALYZE roster_entry;

DROP VIEW IF EXISTS v_roster_gaps;
CREATE VIEW v_roster_gaps AS
WITH open_round AS (
    SELECT round_id FROM paper_round WHERE closed_utc IS NULL ORDER BY opened_utc DESC LIMIT 1
)
SELECT re.roster_entry_id, re.round_id, re.comparison_id, re.model_id, re.object_id, re.object_selector_json, re.eval_id,
       re.instrument_id, re.instrument_selector_json, re.judge_id, re.strength_rule, re.source_citation, re.inferred, re.drawn,
       (SELECT d.state FROM disposition d WHERE d.roster_entry_id = re.roster_entry_id ORDER BY d.ruled_utc DESC LIMIT 1) AS disposition_state,
       (SELECT d.reason FROM disposition d WHERE d.roster_entry_id = re.roster_entry_id ORDER BY d.ruled_utc DESC LIMIT 1) AS disposition_reason
FROM roster_entry re JOIN open_round orr ON orr.round_id = re.round_id
-- the roster version (migration 0016): only the entries of the version the paper_round row of the round carries are gaps; an entry a later
-- version re-shaped (its older id left behind) is neither matched nor listed here (no apostrophe here: the reference scan)
JOIN paper_round pr ON pr.round_id = re.round_id AND (pr.roster_version IS NULL OR re.roster_version = pr.roster_version)
WHERE NOT EXISTS (
        SELECT 1 FROM v_roster_match m JOIN v_paper_current pc ON pc.run_id = m.run_id
        WHERE m.roster_entry_id = re.roster_entry_id AND pc.in_current_paper = 1)
  AND NOT EXISTS (SELECT 1 FROM disposition d WHERE d.roster_entry_id = re.roster_entry_id);
DROP VIEW IF EXISTS v_roster_match;
CREATE VIEW v_roster_match AS
SELECT r.run_id, re.roster_entry_id, re.comparison_id
    FROM run r
    JOIN object o ON o.object_id = r.object_id
    JOIN instrument i ON i.instrument_id = r.instrument_id
    -- the roster_entry index (round_id, model_id, eval_id) carries this join; a CTE over the table would not be indexed
    JOIN roster_entry re ON re.round_id = (SELECT round_id FROM paper_round WHERE closed_utc IS NULL ORDER BY opened_utc DESC LIMIT 1)
      AND re.model_id = r.model_id AND re.eval_id = r.eval_id
      -- the roster version (migration 0016): an entry matches only under the version the paper_round row of its round carries; a load
      -- re-stamps every entry it expands, so an entry a later version re-shaped keeps its older stamp and matches nothing (before 0016
      -- such an entry kept matching beside its replacement, a second selector for one key); a round loaded before
      -- 0016 carries no version and admits every entry as before (no apostrophe in this comment: the reference scan strips quoted literals)
      AND ((SELECT roster_version FROM paper_round pr WHERE pr.round_id = re.round_id) IS NULL
           OR re.roster_version = (SELECT roster_version FROM paper_round pr WHERE pr.round_id = re.round_id))
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
