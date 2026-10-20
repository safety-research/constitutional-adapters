-- 0029_analysis_filing.sql
-- Analyses filed over ingested runs and objects (a registry decision of 2026-10-08). Five tables are
-- CREATED and nothing is rebuilt, altered or recreated: `analysis` (one row per filed analysis: name, code sha, cluster job ids, substrate,
-- report uri + sha, filed_utc / filed_by, provenance under extra_json), `analysis_input` (its runs and objects, each named in the key and in
-- the FK column of its kind), `analysis_artifact` (its links to the `artifact` rows the filing registered), `ordered_fact_record` (the
-- harvester's index of every record of the three the fold has applied, by key: the carriers it judges again when a key's kept record moves)
-- and `ordered_fact_carrier` (every arrival of such a record with the side of its file's delivery audit it is counted on, so a record judged
-- again moves every file that carried it).
-- The first three are LEDGER-scale facts written by `registry analysis file --spec` under records/analyses/<day>/ and folded by the
-- ledger-fact apply path (reducer.apply_ledger_fact -- the 0023 `record_refusal` precedent: no row_history, no refold; the row kept for a key
-- is the record with the smallest (written_utc, record_sha256), the log's order, carried in the three derived columns record_written_utc /
-- record_sha256 / record_uri of each, so live ingest and a rebuild keep the same row whatever the arrival order, and a later record with
-- another payload is refused by name); their kinds are in vocab.UNLEDGERED_FACT_TYPES, so the two ledgers whose record_type CHECK lists every
-- record kind (`blocked_records`, `row_history`) are NOT touched: their CHECK lists render from vocab.LEDGERED_RECORD_TYPES and are
-- byte-identical to the ones 0020 wrote. Rendered by `registry migrate --plan --sql` against a schema-version-28 database (0001..0028
-- applied to an empty database by the runner; 0028 `reference_model_never_selected`, the vocabulary branch's one recreate of v_run_current,
-- landed first, so this file took the next number at the merge, as the migrations README says); the planner reports five creates (with
-- the analysis tables' four indexes) and NO rebuild -- tests/test_registry_analysis_filing.py pins the plan and the body, and any rebuild line is a STOP
-- for the landing window (scripts/registry/landing/window.sh exits 15 on one). No Python post-step: the tables start empty; the first filing
-- (three MoE-routing analyses) happens in the window after this file lands.
-- The fifth closing pass (2026-10-09) added `filing_sha256` to the two child tables -- the record sha of the `analysis` header that
-- wrote the row, so an input or a link folds only while its filing is the kept one or a duplicate of it (records.LEDGER_FILING_CHILDREN)
-- -- and this file was re-rendered and re-frozen: the tables are its own CREATEs, so still NO rebuild.
-- The ninth closing pass (2026-10-10) added `ordered_fact_carrier` -- the delivery audit's per-record detail: one row per arrival of a
-- record of the three tables, with the side (`accepted` / `refused`) of the carrying file's `record_delivery` counts it stands on -- so a
-- filing displaced or judged again moves EVERY file that carried it, not the index's one canonical carrier; re-rendered and re-frozen
-- again: its own CREATE, still NO rebuild.
-- drift: table analysis is absent
-- drift: table analysis_input is absent
-- drift: table analysis_artifact is absent
-- drift: table ordered_fact_record is absent
-- drift: table ordered_fact_carrier is absent
-- drift: index idx_analysis_artifact_artifact is absent
-- drift: index idx_analysis_filed is absent
-- drift: index idx_analysis_input_object is absent
-- drift: index idx_analysis_input_run is absent

-- create analysis
CREATE TABLE analysis (
  analysis_id TEXT NOT NULL,  -- uuid5 over the analysis's identity: canonical JSON of {name, code_sha, inputs} (sisteer.registry.analysis).
  name TEXT NOT NULL,  -- The analysis's name, dotted by study (`moe_routing.capture`); free text, unique only with the code sha and inputs.
  code_sha TEXT NOT NULL CHECK (code_sha GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]' OR code_sha = 'unverified'),  -- The commit the analysis ran from (forty hex), or 'unverified'.
  code_ref TEXT,  -- why-null: a code sha alone names the code
  cluster_job_ids_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(cluster_job_ids_json)),  -- The scheduler job ids that produced it (strings), in the filer's order; [] for a local computation.
  substrate TEXT NOT NULL CHECK (substrate IN ('cluster_a', 'cluster_b', 'modal', 'grading_host', 'runpod', 'slurm')),  -- Where it ran.
  report_uri TEXT,  -- why-null: an analysis whose outputs are pages alone files no report
  report_sha256 TEXT CHECK (report_sha256 IS NULL OR report_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'),  -- why-null: paired with report_uri (the table CHECK): null exactly when no report is filed
  filed_utc TEXT NOT NULL CHECK (filed_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),  -- When the filing was written (the filer's clock).
  filed_by TEXT NOT NULL,  -- Whose work the analysis is (by default); the relaying session is `extra_json.provenance.relayed_by`.
  notes TEXT,  -- why-null: free text
  extra_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(extra_json)),  -- `provenance` {kind: user_direct|relayed, relayed_by, message_utc}, the spec's own `extra`, and any unknown record field.
  record_written_utc TEXT NOT NULL CHECK (record_written_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),  -- The written_utc of the filing whose header this row is: with record_sha256 the row's ordering key (the smallest among the id's filings is kept).
  record_sha256 TEXT NOT NULL CHECK (record_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'),  -- The record sha of the filing kept (see record_written_utc).
  record_uri TEXT NOT NULL,  -- The logical uri (`<file>#<line>`) of the filing kept, so a displaced filing can be refused by name.
  PRIMARY KEY (analysis_id),
  CONSTRAINT analysis_report_pairing CHECK ((report_uri IS NULL) = (report_sha256 IS NULL))
);
CREATE INDEX idx_analysis_filed ON analysis (filed_utc);

-- create analysis_input
CREATE TABLE analysis_input (
  analysis_id TEXT NOT NULL REFERENCES analysis(analysis_id),
  filing_sha256 TEXT NOT NULL CHECK (filing_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'),  -- The record sha of the `analysis` header (the filing) that wrote this row; the row folds only while that filing is the kept one or a duplicate of it.
  input_kind TEXT NOT NULL CHECK (input_kind IN ('run', 'object')),
  input_id TEXT NOT NULL,  -- The run_id or object_id, as the kind says.
  run_id TEXT REFERENCES run(run_id),  -- why-null: an object input names no run
  object_id TEXT REFERENCES object(object_id),  -- why-null: a run input names no object
  extra_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(extra_json)),  -- What the analysis read of the input (`role`: the steering condition, the transcript set), as the filer wrote it.
  record_written_utc TEXT NOT NULL CHECK (record_written_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),  -- The written_utc of the record this row is: with record_sha256 the row's ordering key (the smallest among the key's records is kept).
  record_sha256 TEXT NOT NULL CHECK (record_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'),  -- The record sha of the record kept (see record_written_utc).
  record_uri TEXT NOT NULL,  -- The logical uri (`<file>#<line>`) of the record kept, so a displaced record can be refused by name.
  PRIMARY KEY (analysis_id, input_kind, input_id),
  CONSTRAINT analysis_input_kind_pairing CHECK ((input_kind = 'run' AND run_id = input_id AND object_id IS NULL) OR (input_kind = 'object' AND object_id = input_id AND run_id IS NULL))
);
CREATE INDEX idx_analysis_input_run ON analysis_input (run_id) WHERE run_id IS NOT NULL;
CREATE INDEX idx_analysis_input_object ON analysis_input (object_id) WHERE object_id IS NOT NULL;

-- create analysis_artifact
CREATE TABLE analysis_artifact (
  analysis_id TEXT NOT NULL REFERENCES analysis(analysis_id),
  filing_sha256 TEXT NOT NULL CHECK (filing_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'),  -- The record sha of the `analysis` header (the filing) that wrote this link; the link folds only while that filing is the kept one or a duplicate of it.
  artifact_id TEXT NOT NULL REFERENCES artifact(artifact_id),
  role TEXT NOT NULL CHECK (role IN ('report', 'page', 'figure')),
  extra_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(extra_json)),  -- `page_url` / `page_dir` for a file of a published page directory; `declared_sha256` for a remote report.
  record_written_utc TEXT NOT NULL CHECK (record_written_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),  -- The written_utc of the record this row is: with record_sha256 the row's ordering key (the smallest among the key's records is kept).
  record_sha256 TEXT NOT NULL CHECK (record_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'),  -- The record sha of the record kept (see record_written_utc).
  record_uri TEXT NOT NULL,  -- The logical uri (`<file>#<line>`) of the record kept, so a displaced record can be refused by name.
  PRIMARY KEY (analysis_id, artifact_id)
);
CREATE INDEX idx_analysis_artifact_artifact ON analysis_artifact (artifact_id);

-- create ordered_fact_record
CREATE TABLE ordered_fact_record (
  table_name TEXT NOT NULL,  -- The ordered table (a key of records.LEDGER_ORDERED_TABLES); no CHECK list, so a table ordered later adds rows and the shape stands.
  natural_key_json TEXT NOT NULL CHECK (json_valid(natural_key_json)),  -- Canonical JSON of the record's natural-key columns, as `row_history` keys a run-scale row.
  record_sha256 TEXT NOT NULL CHECK (record_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'),  -- The record (records.record_sha).
  record_type TEXT NOT NULL,  -- The record's kind (the refusal row of a record judged again names it).
  record_uri TEXT NOT NULL,  -- The record's logical uri (`<file>#<line>`); the smallest when the same record arrives under two names.
  written_utc TEXT NOT NULL CHECK (written_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),  -- The record's written_utc: with record_sha256 its place in the key's order.
  PRIMARY KEY (table_name, natural_key_json, record_sha256)
);

-- create ordered_fact_carrier
CREATE TABLE ordered_fact_carrier (
  table_name TEXT NOT NULL,  -- The ordered table (a key of records.LEDGER_ORDERED_TABLES); no CHECK list, as `ordered_fact_record`.
  record_sha256 TEXT NOT NULL CHECK (record_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'),  -- The record (records.record_sha).
  record_uri TEXT NOT NULL,  -- The logical uri (`<file>#<line>`) this arrival of the record came under; the file before `#` is the delivery row's.
  delivery TEXT NOT NULL CHECK (delivery IN ('accepted', 'refused')),  -- The side of the file's delivery audit the record is counted on: `accepted` (`n_ok` or `n_duplicate`) or `refused` (`n_refused`); moved with the file's counts when the record is judged again.
  PRIMARY KEY (table_name, record_sha256, record_uri)
);
