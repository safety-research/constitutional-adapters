-- 0024_fold_observation.sql
-- The fold-observation change's one migration (2026-10-04). Numbered 0024 in this tree
-- -- max landed + 1, re-checked at the merge as the migrations README says; the runner requires contiguous files, so a later number cannot
-- exist beside an unlanded one -- and renumbered before its gate if another branch's 0024 lands first (the Python post-step is keyed by
-- the slug `fold_observation`, so a renumbering never detaches it). One CREATE TABLE with its one index; no table is rebuilt, no column is
-- added, no v_* object is recreated, no row is copied. Rendered by `registry migrate --plan --sql` against a schema-version-23 database
-- (0001..0023 applied to an empty database by the runner): the planner reports one create and nothing else, and `tests/test_registry_fold_observations.py`
-- pins the plan -- any `rebuild` line is a STOP (the 0020 rebuild of row_history wrote a 25 GB WAL).
-- `fold_observation` (ledger-scale; pk (observation_kind, run_id, record_sha256)): the harvester's OBSERVATIONS about a record at the fold,
-- none of them a refusal -- a detector over the fold's own output, never a fold rule; the record folds exactly as it always did
-- (replace-at-key is the universal rule for `extra`, ruled). The one kind, `serving_jobs_strict_subset` (vocab.FOLD_OBSERVATION_KINDS): a
-- `run` or `run_patch` record whose `extra.serving_jobs` (a list, compared as a set) names a STRICT subset of the set the run's folded
-- `extra_json.serving_jobs` already held -- a job-start registration restating the key on a same-identity rerun replaces the set, and the
-- producer whose cumulative set is the only belt learns it slipped here (`registry observations summary | list`). The row records the
-- folded set, the restated set, the dropped members, the restating record and the record whose statement the folded set came from.
-- Derived, not recorded: `reducer.Reducer.apply_run_scale` derives a run key's rows from its ordered events and the fold's snapshots after
-- every refold of the key (`run` always refolds its whole history) and replaces them whole, so a late-arriving earlier-stamped record
-- re-derives them as it re-derives `row_history.folded_state_json`; a rebuild carries the same rows with no column excluded
-- (`harvest.verify_equal`). The cycle that folded the restating record is not a column: it is the arrival of the record's file
-- (`record_ingest.arrival_utc`), joined at read time. No record kind is added, so the two ledgers' `record_type` CHECK lists
-- (`blocked_records`, `row_history`) are byte-identical to 0023's. The rows of the runs already folded are derived ONCE by this file's
-- Python post-step (migrations.POST_APPLY["fold_observation"] = migrations.derive_fold_observations -> observations.backfill: every live
-- run whose `extra_json` carries the key -- exactly the keys with an event stating it, the fold never removes an `extra` key of a run --
-- read back from row_history, folded and observed through the same code the live path runs; idempotent; counted in the line `registry
-- migrate` prints). Live, read-only, 2026-10-04: 162 of 12,699 run rows carry `extra_json.serving_jobs`, every value a list.
-- drift: table fold_observation is absent
-- drift: index idx_fold_observation_written is absent

-- create fold_observation
CREATE TABLE fold_observation (
  observation_kind TEXT NOT NULL CHECK (observation_kind IN ('serving_jobs_strict_subset')),  -- What was observed (vocab.FOLD_OBSERVATION_KINDS).
  run_id TEXT NOT NULL REFERENCES run(run_id),  -- The run whose folded state the record restated.
  record_sha256 TEXT NOT NULL CHECK (record_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'),  -- The RESTATING record (records.record_sha): the `run` or `run_patch` event that carried the smaller set.
  record_type TEXT NOT NULL,  -- The restating record's kind: `run` (a re-registration) or `run_patch`.
  record_uri TEXT NOT NULL,  -- The record file (logical path) that carried the restating record -- the event's carrier in `row_history`, following the canonical carrier when the same record arrives under two names.
  written_utc TEXT NOT NULL CHECK (written_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),  -- The restating record's written_utc: its position in the fold's order (never the cycle that folded it).
  folded_record_sha256 TEXT NOT NULL CHECK (folded_record_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'),  -- The record whose statement the folded set came from: the latest earlier event of the key that stated `extra.serving_jobs`.
  folded_json TEXT NOT NULL CHECK (json_valid(folded_json)),  -- The set the run's extra_json held before the restatement, as a sorted JSON array of its members.
  restated_json TEXT NOT NULL CHECK (json_valid(restated_json)),  -- The set the record stated, as a sorted JSON array of its distinct members (order and duplicates in the record are not facts).
  dropped_json TEXT NOT NULL CHECK (json_valid(dropped_json)),  -- The members of the folded set the restatement left out, sorted.
  n_dropped INTEGER NOT NULL CHECK (n_dropped >= 0),  -- len(dropped_json): the rank for `registry observations summary`.
  PRIMARY KEY (observation_kind, run_id, record_sha256)
);
CREATE INDEX idx_fold_observation_written ON fold_observation (written_utc);
