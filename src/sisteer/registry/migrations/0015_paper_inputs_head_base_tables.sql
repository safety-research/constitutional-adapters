-- 0015_paper_inputs_head_base_tables.sql
-- `v_paper_inputs_head` -- the paper rebuild trigger and the cache key of every paper consumer (`_write_paper_head` at the end of every
-- harvester cycle, `registry paper-head`, the R2 export, the parity gate, the status page) -- is MAX(ingest_seq) over the rows the paper
-- reads. Its first member was `SELECT MAX(ingest_seq) FROM v_paper_long WHERE in_current_paper = 1`, which evaluates the whole paper frame
-- (roster match, rulings, supersession, withdrawals, the metric-role ranking) to produce one number: 452 s of a 2,011 s cycle on a cold
-- 25 GB snapshot (a profile of the fold), every cycle, outside any transaction and so invisible to
-- `state/metrics.jsonl`; 205 s read-only on the live database on 2026-09-24. Its last member, `MAX(ingest_seq) FROM item_result WHERE
-- withdrawn_by_revision_id IS NOT NULL`, walked the partial index `idx_item_result_withdrawn` and probed every withdrawn row (509,125 live)
-- for its ingest_seq: 115 s cold. Since this file the head is a lookup, in two different ways:
-- (b) a SUPERSET for the frame member. No cheap exact form exists -- whether a row is in the paper IS the frame -- so the member is
--     replaced by MAX(ingest_seq) over the base tables an in-paper v_paper_long row draws its ingest_seq from. That column is
--     `MAX(r.ingest_seq, rm.ingest_seq, COALESCE(rc.ingest_seq, 0), COALESCE(msc.ingest_seq, 0))` with r = run, rm = run_metric,
--     rc = run_contrast and msc = v_multi_source_check, whose ingest_seq is itself a MAX over run_metric.ingest_seq (the owner and the
--     secondary rows of a declared scope, of ANY run in it). `run` is read at the two lifecycle states the selection CASE admits into the
--     paper (`in_current_paper` is 0 for every other state before any ruling is consulted); `run_metric` and `run_contrast` are read whole
--     (a metric row of a run outside the frame can reach an in-paper row through the multi-source fold). Every value the old member could
--     take is therefore <= the new one at the same instant: the head moves at least whenever it did -- no consumer is ever left on a
--     stale cache -- and ALSO when a run outside the frame is judged or ingested or files a metric or contrast row: a rebuild that reads an
--     unchanged frame, the price of the superset, never a missed one. The membership rule holds exactly as before: accepted source revisions and
--     withdrawals stay members in their own right, so a withdrawal moves the head whether or not the withdrawn observation was in the paper.
-- (a) EXACT for the item member, through the new partial index `idx_item_result_withdrawn_seq ON item_result (ingest_seq) WHERE
--     withdrawn_by_revision_id IS NOT NULL`: MAX(ingest_seq) is one seek to its last entry. The member NAMES the index (INDEXED BY): the
--     live `sqlite_stat1` puts `idx_item_result_withdrawn` at 480 rows (an ANALYZE the withdrawn set has since outgrown a thousandfold),
--     and with those statistics the planner keeps walking the old index and probing every row; the name holds the plan whatever the
--     statistics say (a missing index is a loud error, and `db.schema_drift` refuses a database without it before that).
--     `idx_item_result_withdrawn` itself is untouched: the reclaim scans (`Ingestor.reclaim`) read it as a covering index.
-- The other seven members (roster_entry, ruling, ruling_state, disposition, source_revision WHERE accepted = 1, withdrawal) are unchanged.
-- Measured read-only on the live database (version 14; one read transaction, so every number is from the same snapshot; 2026-09-24): the
-- object as it stood 205.1 s -> the new body with the nine base-table members alone 0.2 s (its item member still probing without the
-- index, after the old object had warmed those rows: 20.6 s); both read 201280 on that snapshot -- the frame's newest row was in the paper,
-- so old = new there; the frame member alone 60.8 s warm. On the profile's cold snapshot the two replaced members cost 452 s + 115 s.
-- Cost of applying: one CREATE INDEX over `item_result` (12.2 M rows read once to collect the 509k withdrawn entries; minutes, under the
-- harvester lock as every migration is), then DROP / CREATE of the one v_* object. No table rebuild, no rows copied. Rendered by
-- `registry migrate --plan --sql` against a schema-version-14 database. Every writer refuses a database behind version 15:
-- `registry migrate` in the read window right after landing.
-- Tests: `tests/test_registry_migrations.py::test_0015_paper_inputs_head_reads_base_tables_on_a_v14_database` (the file on a v14 database:
-- exactly these two objects differ before and nothing after; the item member's plan is the new index and the plan touches no paper object;
-- the file carries schema_def's text verbatim); `tests/test_registry_paper_views.py::test_paper_inputs_head_is_a_superset_of_its_pre_0015_text`
-- (new >= old at every step of a synthetic frame -- in-paper rows, a ruling, an accepted revision and a withdrawal, a judged run outside the
-- frame -- equal until the last step and greater there, where the old text stands still).
-- drift: v_paper_inputs_head differs from the code's SQL
-- drift: index idx_item_result_withdrawn_seq is absent

DROP INDEX IF EXISTS idx_item_result_withdrawn_seq;
CREATE INDEX idx_item_result_withdrawn_seq ON item_result (ingest_seq) WHERE withdrawn_by_revision_id IS NOT NULL;
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
    -- one seek through the partial index on ingest_seq over the withdrawn rows, named so the plan holds whatever the statistics say
    UNION ALL SELECT MAX(ingest_seq) FROM item_result INDEXED BY idx_item_result_withdrawn_seq WHERE withdrawn_by_revision_id IS NOT NULL
);
