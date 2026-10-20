-- 0030_fold_order_barrier.sql
-- The deployment barrier of the fold-order version (registry-fold-order-version, 2026-10-09), in two layers that one file lands together:
-- the VERSION every pre-guard writer refuses by its own check, and the FENCE the database enforces on every write of a `run` row.
--
-- Why. The reducer's same-stamp fold order is versioned (reducer.FOLD_ORDER_VERSION = "fold_order/2: written_utc, tie_rank,
-- record_sha256") and stamped in the database as the schema_migration row version = 0, written by the first writer after the code lands
-- and checked by every writer OF THIS CODE before it folds (db.require_fold_order). But a writer from BEFORE the stamp never reads row 0:
-- a checkout with the pre-tie-break order and the same migration files -- the harvester's base until 2026-10-06 22:26 PDT, a judgment
-- daemon on an older worktree until 2026-10-08, the writers of the incident -- passes its own version-and-shape check against a stamped
-- database at 29, refolds a same-stamp pair back to `judged`, and leaves row 0 untouched, so the residue grows while the new code reads
-- agreement (the review's round-2 HIGH). And one shipped writer checks NOTHING: `registry replica refold-apply --db <file> --keys <keys>`
-- of commit 5ea3fa411 opens the file it is given with a bare db.connect, not open_or_create, and rewrites the named run rows under its
-- own order with no look at the version, the shape or the stamp (the review's round-3 MEDIUM) -- a version row fences a writer only if
-- that writer reads it.
--
-- Layer 1, the version. The schema_migration row the runner inserts moves the database from 29 to 30, and at 30 every pre-guard writer
-- that opens through its own open_or_create -- at 29, 28 or 27 -- refuses by name ("is at schema version 30, code expects 27") and writes
-- nothing. Pinned against the exported code of commit 880be4e1c (code version 27, the old order, no fold-order guard) in
-- tests/test_registry_fold_order_barrier.py, with the hole shown at that writer's own version as the control.
--
-- Layer 2, the fence: the two triggers below, `trg_run_fold_order_fence_ins` / `trg_run_fold_order_fence_upd` (schema_def.
-- fold_order_fence_triggers; the render against the 0001..0029 shape, which is this file's body). Every connection db.connect opens
-- registers the SQL function registry_fold_order() (db.FOLD_ORDER_FUNCTION) returning the code's order; the triggers refuse a write of a
-- `run` row when that order IS NOT the one row 0 records, and a connection that never registered the function cannot write a `run` row at
-- all -- SQLite resolves the function as it compiles the INSERT or UPDATE, so the write fails `no such function: registry_fold_order`
-- before a row moves, in the old process, with no code of ours running there. A database that records NO order (row 0 absent) is not
-- fenced: db.require_fold_order stamps it on the first writer, and the trigger compares only against a recorded order, so the bootstrap
-- is the same on both sides. Only `run` is fenced: reducer.tie_rank tells run_state events apart and nothing else, so `run` is the one
-- table whose rows the two orders fold differently. Pinned against the exported code of commit 5ea3fa411 (`python -m
-- sisteer.registry.replica refold-apply` as a subprocess against a stamped database at 30 with an adversarial same-stamp pair: refused
-- at the row, rows byte-identical; the same command against a copy with the fence dropped rewrites the pair back to `judged`, the hole).
--
-- What the file does to the shape: two triggers created, nothing else -- no table rebuilt, no column added, no row copied, no data
-- rewritten, no v_* object touched. A database at 30 without them reads as shape drift (`trigger trg_run_fold_order_fence_ins is
-- absent`) and every writer refuses it; a rebuild of `run` recreates them with the table's other triggers (schema_def.triggers_on).
--
-- Apply it in the landing window right after this branch lands: `registry migrate` with WINDOW_EXPECT_MIGRATION=0030_fold_order_barrier.
-- The plan reads `pending: 0030_fold_order_barrier`, `shape differs from the code's schema:` with the two `trigger ... is absent` lines,
-- `recreate trigger(s): trg_run_fold_order_fence_ins, trg_run_fold_order_fence_upd`, no rebuild, no create, no add (the window script
-- says `rebuilds: none`), and the verify stage reads version 30 with the shape matching. The bootstrap stamp (row 0) is NOT this file's:
-- `registry migrate` is a schema writer, not a fold writer, and stamps nothing; the first writer tick after the window -- the harvester's
-- open_or_create -- stamps fold_order/2 on the same database, so from then on the ledger carries both the barrier row (30) and the
-- fold-order row (0). docs/registry/records.md, "Identity and order": the run-scale order has a version.
-- drift: trigger trg_run_fold_order_fence_ins is absent
-- drift: trigger trg_run_fold_order_fence_upd is absent

DROP TRIGGER IF EXISTS trg_run_fold_order_fence_ins;
CREATE TRIGGER trg_run_fold_order_fence_ins BEFORE INSERT ON run
WHEN EXISTS (SELECT 1 FROM schema_migration WHERE version = 0 AND description IS NOT registry_fold_order())
BEGIN
  SELECT RAISE(ABORT, 'run: fold-order fence -- this connection folds under a same-stamp order the database does not record (registry_fold_order() against schema_migration row 0); nothing written: fast-forward the checkout or rebuild the database (docs/registry/records.md, Identity and order: the run-scale order has a version)');
END;
DROP TRIGGER IF EXISTS trg_run_fold_order_fence_upd;
CREATE TRIGGER trg_run_fold_order_fence_upd BEFORE UPDATE ON run
WHEN EXISTS (SELECT 1 FROM schema_migration WHERE version = 0 AND description IS NOT registry_fold_order())
BEGIN
  SELECT RAISE(ABORT, 'run: fold-order fence -- this connection folds under a same-stamp order the database does not record (registry_fold_order() against schema_migration row 0); nothing written: fast-forward the checkout or rebuild the database (docs/registry/records.md, Identity and order: the run-scale order has a version)');
END;
