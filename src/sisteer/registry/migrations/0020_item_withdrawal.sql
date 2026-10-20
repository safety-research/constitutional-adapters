-- 0020_item_withdrawal.sql
-- A producer retracts item_result rows it filed under the wrong run (a registry ruling of 2026-10-01), and `n_items_planned`
-- joins records.RUN_PATCHABLE_FIELDS (no schema change for that half). Motivating case: a per-behavior Petri run carrying three
-- generation-time item_result rows of one item that belong to a derived control run of the same campaign (the producer's bug, not
-- the registry's); the producer files ONE `item_withdrawal` record {run_id, items: [exact item keys], reason, evidence_uri?,
-- belongs_to_run_id?} and the rows leave every current-item reading while the raw table and the record log keep them.
-- What this file changes:
-- (a) `item_result` gains two DERIVED nullable columns, `withdrawn_by_record_sha256` (the withdrawing record's sha) and
--     `withdrawn_reason`, declared AFTER the item-scale tail on purpose: SQLite's ALTER TABLE ADD COLUMN appends a column after the
--     last existing one (before the table constraints), so only a column the code declares last lets the live table (12.2 M rows)
--     be migrated by two O(1) ALTERs -- no row copied, no scan: the sha column carries no CHECK because ADD COLUMN with a CHECK makes
--     SQLite run pragma_quick_check over every existing row (3.37+), and the value is the harvester's copy of a record sha the
--     validator already checked. The planner learned this shape (`migrations.addable_columns`: verified by executing the ALTERs in a
--     scratch database and comparing the normalized text, never inferred), so `registry migrate --plan` says "add column(s)" here
--     and "rebuild" for anything else. The views that define a CURRENT item row -- `v_item_result_current` (both its outer predicate
--     and the provenance anti-join) and `v_item_result_provenances` (hence `v_item_conflict`; `v_item_headline` and
--     `v_judge_side_by_side` read the current view by name) -- exclude a marked row exactly as they exclude a superseded revision's.
-- (b) a new run-scale table `item_withdrawal`, one row per retracted key (the observation pattern: the record's `items` list folds
--     into one row per key, the latest record naming a key stands by order key); `belongs_to_run_id` is a name, not an FK.
--     `v_paper_inputs_head` reads its MAX(ingest_seq) as a member in its own right (the marks do not move the item rows' ranks) and,
--     through it by primary key, the rank of the native item_result row each key names: a row re-filed above its withdrawal clears the
--     marks and is current again at a rank only that row carries (CROSS JOIN fixes the loop order; found in review, 2026-10-01).
-- (c) the two ledgers whose record_type CHECK lists every record kind, `blocked_records` and `row_history` (whose table_name CHECK
--     lists every run-scale table too), are REBUILT -- SQLite cannot alter a CHECK in place, and every new record kind costs this
--     copy once. blocked_records is 875 rows live; row_history is 4,165,237 rows of JSON (about 2 KB a row sampled read-only), the
--     migration's ONE real cost -- minutes with the write lock held, measured on a copy before the window (docs/registry/operations.md
--     under "A producer retracts item rows filed under the wrong run").
-- The fold (reducer._apply_item_withdrawal / remark_item_withdrawals): the natively emitted row of a key (source_revision_id = '';
-- an adapter revision's rows are the revision's to carry) takes the mark iff its order key (written_utc, record_sha256) is BELOW
-- the withdrawing record's; a row re-filed with a greater order key stands as a new measurement and reads NULL again, and the
-- harvester re-derives a run's marks after every item flush, so the marks are a pure function of the records (rebuild equality
-- holds whatever the delivery order). A withdrawal whose run or whose named row is not in the table yet is dependency-blocked
-- (`missing_parent_table` run, or item_result with the row's canonical key), never refused for it. Rendered by
-- `registry migrate --plan --sql` against a schema-version-19 database (0001..0019 applied to an empty database).
-- drift: table item_result lacks column(s) withdrawn_by_record_sha256, withdrawn_reason
-- drift: table item_result: CREATE text differs — have «CREATE TABLE item_result(run_id TEXT NOT NULL,item_id TEXT NOT NULL,metric_id TEXT NOT NULL,trial_index INTEGER NOT NULL DEFAULT 0,sample_index INTEGER NOT NUL…» want «CREATE TABLE item_result(run_id TEXT NOT NULL,item_id TEXT NOT NULL,metric_id TEXT NOT NULL,trial_index INTEGER NOT NULL DEFAULT 0,sample_index INTEGER NOT NUL…»
-- drift: table item_withdrawal is absent
-- drift: table blocked_records: column record_type CHECK (record_type IN (...)) lacks 'item_withdrawal'
-- drift: table row_history: column table_name CHECK (table_name IN (...)) lacks 'item_withdrawal'
-- drift: table row_history: column record_type CHECK (record_type IN (...)) lacks 'item_withdrawal'
-- drift: v_item_result_current differs from the code's SQL
-- drift: v_item_result_provenances differs from the code's SQL
-- drift: v_paper_inputs_head differs from the code's SQL

-- create item_withdrawal
CREATE TABLE item_withdrawal (
  run_id TEXT NOT NULL REFERENCES run(run_id),  -- The run the retracted rows were filed under (FK; the record is dependency-blocked until it exists).
  item_id TEXT NOT NULL,
  metric_id TEXT NOT NULL,
  trial_index INTEGER NOT NULL DEFAULT 0,
  sample_index INTEGER NOT NULL DEFAULT 0,
  turn_index INTEGER NOT NULL DEFAULT -1,  -- -1 for single-turn items (the item_result sentinel).
  judgment_id TEXT NOT NULL DEFAULT '',  -- '' for generation-time metrics (the item_result sentinel; no FK: the key names a row, not a judgment).
  reason TEXT NOT NULL,  -- Free text: why the rows are retracted (required; not a vocabulary).
  evidence_uri TEXT,  -- why-null: optional: a file or page that shows the misfiling
  belongs_to_run_id TEXT,  -- why-null: optional: the run the rows should have been filed under; a name, not an FK (that run may never reach the registry)
  field_versions_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(field_versions_json)),  -- {column: [written_utc, record_sha256] of the record that last set it} — the FULL order key (S2).
  extra_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(extra_json)),  -- Unknown record fields, never dropped (Fixed decision 13).
  record_uri TEXT NOT NULL,  -- The record file (logical path) that produced the current row.
  ingest_seq INTEGER NOT NULL,  -- The producing FILE's rank in the ledger's arrival order (record_ingest: arrival_utc, then first_written_utc, sha, uri); derived, never allocated — every row above a head a reader has seen is new to it.
  PRIMARY KEY (run_id, item_id, metric_id, trial_index, sample_index, turn_index, judgment_id)
);

-- rebuild blocked_records: SQLite cannot alter a constraint in place (migrations/README.md)
CREATE TABLE blocked_records__new (
  record_sha256 TEXT NOT NULL CHECK (record_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'),
  record_uri TEXT NOT NULL,
  record_type TEXT NOT NULL CHECK (record_type IN ('run', 'run_patch', 'run_state', 'run_metric', 'run_contrast', 'item_result', 'item_flags', 'run_degeneration', 'judgment', 'object', 'artifact', 'model', 'eval', 'item_set', 'instrument', 'judge', 'si_text', 'fit_search', 'fit_trial', 'paper_round', 'roster_entry', 'ruling', 'decode_policy', 'sampling_mode_policy', 'computation', 'disposition', 'snapshot', 'parity_gate', 'item_withdrawal', 'run_ingest', 'ruling_state', 'source_watermark', 'source_state', 'source_revision', 'observation', 'withdrawal', 'mirror_ack', 'file_arrival', 'source_status')),
  missing_parent_table TEXT NOT NULL,
  missing_parent_id TEXT NOT NULL,
  first_seen_utc TEXT NOT NULL CHECK (first_seen_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),
  last_checked_utc TEXT NOT NULL CHECK (last_checked_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),
  attempts INTEGER NOT NULL DEFAULT 1,
  stale INTEGER NOT NULL DEFAULT 0 CHECK (stale IN (0, 1)),
  extra_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(extra_json)),
  PRIMARY KEY (record_sha256)
);
INSERT INTO blocked_records__new (record_sha256, record_uri, record_type, missing_parent_table, missing_parent_id, first_seen_utc, last_checked_utc, attempts, stale, extra_json) SELECT record_sha256, record_uri, record_type, missing_parent_table, missing_parent_id, first_seen_utc, last_checked_utc, attempts, stale, extra_json FROM blocked_records ORDER BY record_sha256;
DROP TABLE blocked_records;
ALTER TABLE blocked_records__new RENAME TO blocked_records;
ANALYZE blocked_records;

-- rebuild row_history: SQLite cannot alter a constraint in place (migrations/README.md)
CREATE TABLE row_history__new (
  table_name TEXT NOT NULL CHECK (table_name IN ('paper_round', 'model', 'si_text', 'fit_search', 'fit_trial', 'eval', 'item_set', 'judge', 'instrument', 'decode_policy', 'sampling_mode_policy', 'computation', 'ruling', 'ruling_state', 'object', 'run', 'run_state_event', 'judgment', 'run_metric', 'run_contrast', 'artifact', 'item_withdrawal', 'roster_entry', 'disposition', 'snapshot', 'parity_gate', 'source_revision', 'observation', 'withdrawal', 'source_watermark', 'source_state', 'mirror_ack', 'file_arrival')),
  natural_key_json TEXT NOT NULL CHECK (json_valid(natural_key_json)),
  written_utc TEXT NOT NULL CHECK (written_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),
  record_sha256 TEXT NOT NULL CHECK (record_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'),
  record_type TEXT NOT NULL CHECK (record_type IN ('run', 'run_patch', 'run_state', 'run_metric', 'run_contrast', 'item_result', 'item_flags', 'run_degeneration', 'judgment', 'object', 'artifact', 'model', 'eval', 'item_set', 'instrument', 'judge', 'si_text', 'fit_search', 'fit_trial', 'paper_round', 'roster_entry', 'ruling', 'decode_policy', 'sampling_mode_policy', 'computation', 'disposition', 'snapshot', 'parity_gate', 'item_withdrawal', 'run_ingest', 'ruling_state', 'source_watermark', 'source_state', 'source_revision', 'observation', 'withdrawal', 'mirror_ack', 'file_arrival', 'source_status')),
  record_uri TEXT NOT NULL,
  payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),  -- The effect this event has on the key's row.
  folded_state_json TEXT NOT NULL CHECK (json_valid(folded_state_json)),  -- The row after folding every event up to and including this one.
  PRIMARY KEY (table_name, natural_key_json, written_utc, record_sha256)
);
INSERT INTO row_history__new (table_name, natural_key_json, written_utc, record_sha256, record_type, record_uri, payload_json, folded_state_json) SELECT table_name, natural_key_json, written_utc, record_sha256, record_type, record_uri, payload_json, folded_state_json FROM row_history ORDER BY table_name, natural_key_json, written_utc, record_sha256;
DROP TABLE row_history;
ALTER TABLE row_history__new RENAME TO row_history;
ANALYZE row_history;

-- add column(s) to item_result: ALTER TABLE ADD COLUMN, no row copied (the code declares them after every live column, so SQLite's appended definition equals the code's CREATE text; migrations/README.md)
-- column withdrawn_by_record_sha256: derived, why-null: set when a producer's item_withdrawal record retracted this natively emitted row: the sha of that record; NULL for a row no withdrawal names, and again for a row re-filed with an order key above the withdrawal's
ALTER TABLE item_result ADD COLUMN withdrawn_by_record_sha256 TEXT;
-- column withdrawn_reason: derived, why-null: the item_withdrawal record's free-text reason, beside its sha; NULL whenever withdrawn_by_record_sha256 is
ALTER TABLE item_result ADD COLUMN withdrawn_reason TEXT;

DROP VIEW IF EXISTS v_item_result_current;
CREATE VIEW v_item_result_current AS
SELECT ir.* FROM item_result ir LEFT JOIN source_revision sr ON sr.revision_id = ir.source_revision_id
WHERE (ir.source_revision_id = '' OR EXISTS (SELECT 1 FROM source_revision sr_v WHERE sr_v.revision_id = ir.source_revision_id AND sr_v.accepted = 1 AND sr_v.superseded_by_revision_id IS NULL)) AND ir.withdrawn_by_record_sha256 IS NULL
  AND NOT EXISTS (SELECT 1 FROM item_result o LEFT JOIN source_revision so ON so.revision_id = o.source_revision_id
                  WHERE o.run_id = ir.run_id AND o.item_id = ir.item_id AND o.metric_id = ir.metric_id AND o.trial_index = ir.trial_index AND o.sample_index = ir.sample_index AND o.turn_index = ir.turn_index AND o.judgment_id = ir.judgment_id AND o.source_revision_id <> ir.source_revision_id
                    AND (o.source_revision_id = '' OR EXISTS (SELECT 1 FROM source_revision sr_v WHERE sr_v.revision_id = o.source_revision_id AND sr_v.accepted = 1 AND sr_v.superseded_by_revision_id IS NULL)) AND o.withdrawn_by_record_sha256 IS NULL AND (CASE WHEN o.source_revision_id = '' THEN 0 ELSE 1 END, -COALESCE(so.priority, 0), COALESCE(so.source_id, ''), o.source_revision_id) < (CASE WHEN ir.source_revision_id = '' THEN 0 ELSE 1 END, -COALESCE(sr.priority, 0), COALESCE(sr.source_id, ''), ir.source_revision_id));
DROP VIEW IF EXISTS v_item_result_provenances;
CREATE VIEW v_item_result_provenances AS
SELECT ir.*, sr.source_id, COALESCE(sr.priority, 0) AS priority
FROM item_result ir LEFT JOIN source_revision sr ON sr.revision_id = ir.source_revision_id
WHERE (ir.source_revision_id = '' OR EXISTS (SELECT 1 FROM source_revision sr_v WHERE sr_v.revision_id = ir.source_revision_id AND sr_v.accepted = 1 AND sr_v.superseded_by_revision_id IS NULL)) AND ir.withdrawn_by_record_sha256 IS NULL;
DROP VIEW IF EXISTS v_paper_inputs_head;
CREATE VIEW v_paper_inputs_head AS
SELECT MAX(seq) AS ingest_seq FROM (
    -- the base tables an in-paper row of the paper frame draws its ingest_seq from, read whole (migration 0015): a superset of the
    -- frame member it replaces -- the head moves at least whenever that did, and also when a run outside the frame lands -- never a
    -- subset. A run enters the frame only once judged or ingested (the selection CASE), so the run member reads those states alone;
    -- a metric row of any run can reach an in-paper row through the multi-source fold, so the metric and contrast members read every row
    SELECT MAX(ingest_seq) AS seq FROM run WHERE lifecycle_state IN ('judged', 'ingested')
    UNION ALL SELECT MAX(ingest_seq) FROM run_metric
    UNION ALL SELECT MAX(ingest_seq) FROM run_contrast
    UNION ALL SELECT MAX(ingest_seq) FROM roster_entry
    UNION ALL SELECT MAX(ingest_seq) FROM ruling
    UNION ALL SELECT MAX(ingest_seq) FROM ruling_state
    UNION ALL SELECT MAX(ingest_seq) FROM disposition
    UNION ALL SELECT MAX(ingest_seq) FROM source_revision WHERE accepted = 1
    UNION ALL SELECT MAX(ingest_seq) FROM withdrawal
    -- a producer's item withdrawals (migration 0020): a small run-scale table whose rank moves when one folds -- the marks it
    -- leaves on item_result rows do not move those rows' ingest_seq, so the table itself is the member
    UNION ALL SELECT MAX(ingest_seq) FROM item_withdrawal
    -- and the native item_result row each item_withdrawal key names, sought by primary key from the withdrawal table (CROSS JOIN fixes
    -- that loop order: without statistics the planner drove a plain join from item_result through idx_item_result_revision on
    -- source_revision_id = '', a walk of every native row). A row RE-FILED above its withdrawal clears the marks and is current again
    -- at a rank no other member reads; this member carries it (a row re-filed below the withdrawal, marked as it stays, moves the head
    -- too: a rebuild that reads an unchanged frame, the superset price)
    UNION ALL SELECT MAX(ir.ingest_seq) FROM item_withdrawal iw CROSS JOIN item_result ir
        ON ir.run_id = iw.run_id AND ir.item_id = iw.item_id AND ir.metric_id = iw.metric_id AND ir.trial_index = iw.trial_index
        AND ir.sample_index = iw.sample_index AND ir.turn_index = iw.turn_index AND ir.judgment_id = iw.judgment_id
        AND ir.source_revision_id = ''
    -- one seek through the partial index on ingest_seq over the withdrawn rows, named so the plan holds whatever the statistics say
    UNION ALL SELECT MAX(ingest_seq) FROM item_result INDEXED BY idx_item_result_withdrawn_seq WHERE withdrawn_by_revision_id IS NOT NULL
);
