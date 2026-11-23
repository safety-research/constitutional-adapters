-- 0033_prune_superseded_item_rows.sql
-- Superseded revisions' item rows are DELETED by the harvester's reclaim from now on (a registry decision of 2026-10-19, reversing
-- the earlier rule that a superseded revision's rows are never deleted): a re-read, an EMISSION_VERSION bump or a forced re-read used to
-- leave a full copy of every item row behind in item_result, item_flags and run_degeneration, read by no v_* object (32.7M such rows
-- on the live database at version 32). The rule is prune-rule/3 (src/sisteer/registry/prune.py): a carrier -- one record file's rows of one
-- accepted, superseded revision in one item table -- is deleted when it is not staged by a held bundle, its log file is present with the
-- bytes the ledger ingested, and the current code re-parses it to exactly the rows present plus those already deleted; every deletion is
-- counted in item_reclaim_ledger, and a deleted carrier is re-folded from the record log before anything is written under its revision
-- again, so live ingest, `registry rebuild` and the replica reach the same state. Pending, held and refused revisions' rows, source_status
-- and observation are never touched.
-- This file only ARMS the rule; it reads and writes no item row. The rows are deleted afterwards, online, by reclaim at the end of each
-- harvester cycle in bounded batches, under the reclaim mode the operator sets (state/reclaim_mode.json: hold | cohort | normal, written
-- by `registry prune-superseded mode`); `registry rebuild --verify` equality holds once that deletion completes. Four statements:
--   1. CREATE TABLE item_reclaim_ledger (schema_def's DDL verbatim; WITHOUT ROWID; empty);
--   2. UPDATE source_revision SET reclaimed = 0 -- every accepted, superseded revision is due for the new reclaim (the flag meant "marked"
--      until now, and nothing marks any more); about 7,000 rows of a small table;
--   3. v_paper_inputs_head recreated WITHOUT its item member (MAX(ingest_seq) over item_result's withdrawn_by_revision_id marks, which nothing
--      writes since this file): the head's value is unchanged, and the schema version's move rebuilds the paper exactly once;
--   4. DROP INDEX idx_item_result_withdrawn_seq, the partial index only that member read. idx_item_result_withdrawn stays (reclaim's pass (i)
--      still clears a re-activated head's leftover marks through it), and so does the withdrawn_by_revision_id column (a payload column every
--      item_result record carries; an inert mark from here on, excluded from rebuild equality).
-- No table is rebuilt, altered or recreated (the landing window runs WITHOUT WINDOW_ALLOW_REBUILD; a rebuild line is a STOP) and there is
-- no Python post-step. Expected migration-ledger delta: exactly one schema_migration row (33, prune_superseded_item_rows). Rendered by
-- `registry migrate --plan --sql` against a schema-version-32 database (0001..0032 applied to an empty database by the runner), with
-- statement 2 added by hand; tests/test_registry_migrations.py::test_0033_on_a_v32_database pins the effects, and the fold's tests are
-- tests/test_registry_prune_fold.py.
-- drift: table item_reclaim_ledger is absent
-- drift: v_paper_inputs_head differs from the code's SQL
-- drift: index idx_item_result_withdrawn_seq is not declared by the code

-- create item_reclaim_ledger
CREATE TABLE item_reclaim_ledger (
  table_name TEXT NOT NULL,  -- The item table the rows were deleted from: item_result, item_flags or run_degeneration (prune.PRUNE_TABLES).
  revision_id TEXT NOT NULL,  -- The accepted, superseded revision they were deleted from (no FK: a ledger row names, never guards).
  record_file_id INTEGER NOT NULL,  -- The carrier: the record file the rows were folded from (record_ingest.record_file_id).
  file_uri TEXT NOT NULL,  -- The carrier's logical uri, the file rehydration re-parses.
  expected_rows INTEGER NOT NULL CHECK (expected_rows >= 0),  -- The rows the current code built from the file when deletion began (its re-parse).
  expected_digest_fold TEXT NOT NULL CHECK (expected_digest_fold GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'),  -- The additive digest of those rows over every column but ingest_seq and withdrawn_by_revision_id (prune.FOLD_EXCLUDE).
  rows_deleted INTEGER NOT NULL CHECK (rows_deleted >= 0),  -- The rows deleted from the carrier so far (running total).
  deleted_digest_fold TEXT NOT NULL CHECK (deleted_digest_fold GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'),  -- The additive digest of the deleted rows, the columns of expected_digest_fold.
  deleted_digest_bind TEXT NOT NULL CHECK (deleted_digest_bind GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'),  -- The additive digest of the deleted rows over every column but ingest_seq (an archive group's digest_bind).
  parsed_with TEXT NOT NULL,  -- The stamp of the code that admitted the deletion (harvest.validator_stamp).
  PRIMARY KEY (table_name, revision_id, record_file_id)
) WITHOUT ROWID;

-- every accepted, superseded revision is due for the new reclaim (no other revision carries the flag)
UPDATE source_revision SET reclaimed = 0 WHERE reclaimed <> 0;

-- the partial index only the dropped v_paper_inputs_head member read
DROP INDEX IF EXISTS idx_item_result_withdrawn_seq;

-- the paper-input head without its item member
DROP VIEW IF EXISTS v_paper_inputs_head;
CREATE VIEW v_paper_inputs_head AS
SELECT MAX(seq) AS ingest_seq FROM (
    -- the base tables an in-paper row of the paper frame draws its ingest_seq from, read whole (migration 0015): a superset of the
    -- frame member it replaces -- the head moves at least whenever that did, and also when a run outside the frame lands -- never a
    -- subset. The run member reads every row (migration 0021): a run that retires leaves the two states the frame admits, and the
    -- frame reads that retirement on its own rows and on every contrast anchored on it, so the two-state member of
    -- migration 0015 missed exactly the move it had to announce; a metric row of any run can reach an in-paper row through the
    -- multi-source fold, so the metric and contrast members read every row too
    SELECT MAX(ingest_seq) AS seq FROM run
    UNION ALL SELECT MAX(ingest_seq) FROM run_metric
    UNION ALL SELECT MAX(ingest_seq) FROM run_contrast
    -- a judgment re-filed superseded or abandoned (migration 0021) takes its metric and item rows out of the frame while moving
    -- no metric or item rank; the fold re-stamps the run row beside the judgment row (a side effect, not a rule), so the judgment table
    -- is a member in its own right and the head does not lean on that side effect
    UNION ALL SELECT MAX(ingest_seq) FROM judgment
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
);
