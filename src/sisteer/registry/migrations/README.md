# Schema migrations

The registry's SQLite schema is declared in `schema_def.py` and rendered to `schema.sql`, both in
`src/sisteer/registry/`; the files here, `NNNN_<slug>.sql`, bring a database to that shape. Every database is
their product: `registry init` applies all of them in order, never `schema.sql`, so a later table rebuild on a
fresh database copies zero rows. A reader of the paper never runs one; a live registry applies pending ones inside a landing window
(`registry migrate` under `scripts/registry/landing/window.sh`, see Landing below).

A landed file is frozen: `tests/test_registry_schema.py` pins its sha256 and requires the migrations, applied to
an empty database, to reproduce `schema.sql`.

There is one exception, made once. Before the repository was published, `0001`–`0031` were edited in place:
the site labels in their CHECK lists became the neutral `cluster_a`, `cluster_b` and `grading_host`, and comments
lost references to internal process. Every changed file was re-pinned in the same change. This is safe for a
database built from either spelling. The runner records only `(version, applied_utc, slug)` per file, never its
bytes. The drift check compares normalised DDL, and normalisation strips comments, so a comment edit is never drift.
`0032_site_label_rename` rebuilds the five tables that carry a site label and maps any legacy spelling through the
site's alias table, so every database converges on the code's shape at version 32.

## Applying

The runner wraps each file in one transaction with `PRAGMA foreign_keys=OFF` and `PRAGMA legacy_alter_table=ON`,
records its `schema_migration` row, runs `PRAGMA foreign_key_check` over the tables the file recreated and their
referrers, and rolls back on a violation. A file carries no transaction or `PRAGMA` statements; a Python
post-step for a data lift is registered in `migrations.POST_APPLY` under the file's slug and runs in the same
transaction.

A file never spells a value it renames. Before the first pending file the runner fills `temp.site_alias(old, new)`
from the site's alias table (`site_aliases.toml` in the registry root, or the file `registry migrate --site-aliases`
names; empty by default), and a rebuild maps a column through it with
`COALESCE((SELECT new FROM temp.site_alias WHERE old = t.c), t.c)`. A check registered in `migrations.PRE_CHECK`
under the file's slug runs first and refuses by name what the file could not map: for `0032`, a stored label value
that is neither a label nor aliased, with its table, column and row count. A mirror's natural key is its destination, so
`0032` also re-keys `mirror_ack`'s `row_history` through the same table: each record stays as written, and one mirror's
events under either spelling are one history.

A database below version 31 that still holds a site's own spelling cannot reach `0032` here: the earlier files rebuild
tables with CHECK lists that admit only the labels and copy each value as stored. The runner refuses such a database by
name before applying anything, naming the file, the column, the value and its rows. Bring it to version 31 with the code
that wrote it, whose migrations admit its spellings, then migrate it here with the site's alias table.

## Writing one

1. Change `schema_def.py` (and `vocab.py`, beside it, for a CHECK vocabulary); a nullable column needs a
   `why_null` reason.
2. `registry docs schema` re-renders `schema.sql` and `docs/registry/schema.md`.
3. `registry migrate --plan` against a database at the previous shape names what
   it lacks; `registry migrate --plan --sql` renders the body.
4. Save it as the next `NNNN_<slug>.sql` and review it: a NOT NULL column without a DEFAULT needs a value in the
   rebuild's `INSERT`; undeclared columns are not copied, undeclared tables never dropped.
5. Pin its sha256 in `FROZEN_MIGRATIONS` (`tests/test_registry_schema.py`), run
   `PYTHONPATH=src pytest tests/test_registry_schema.py` from the repository root, then `registry migrate`
   applies and verifies it.

| change | body | example |
|---|---|---|
| a column after every live column of its table | `ALTER TABLE … ADD COLUMN` (impossible for a NOT NULL column without a DEFAULT; a CHECK on it scans every row) | `0020` |
| any other column change, or a CHECK vocabulary value (SQLite cannot alter a CHECK in place) | a rebuild (`CREATE TABLE <t>__new`, `INSERT … SELECT` the common columns, `DROP`, `RENAME`, indexes, triggers); rewrite rows with a removed value first | `0002`, `0016` |
| a record kind | a kind in `vocab.LEDGERED_RECORD_TYPES` is in the `record_type` CHECK of `blocked_records` and `row_history`, so adding one rebuilds both ledgers; a fact that never enters them (`record_refusal`) goes in `vocab.UNLEDGERED_FACT_TYPES` and rebuilds nothing | `0020`, `0023` |
| a `v_*` object or trigger alone | `DROP … IF EXISTS`, then the code's `CREATE` | `0003`, `0019` |
| a deployment barrier for a rule the rows carry (the fold order): a version that writers from before the rule refuse by their own version check, plus a FENCE the database enforces on writers that check nothing | the runner's `schema_migration` row (the version), and triggers that compare a SQL function every `db.connect` registers (`registry_fold_order()`) with the stamp in row 0 -- a connection without the function cannot write the fenced table at all | `0030` (the fold-order version: two triggers on `run`) |

A rebuild copies the whole table: dimension tables in under a second, an item-scale table (`item_result`,
`observation`, `run_metric`) or `row_history` (millions of JSON rows) in gigabytes of WAL.

## Landing

The number is "max landed + 1" at the merge: when two branches carry the same next number, the second to land
renumbers before its gate (the landing test battery): file name, frozen sha, tests naming the file, body
re-rendered against the then-live shape. `POST_APPLY` keys by slug, so renumbering keeps the post-step.

The plan is read before `registry migrate` is run. The maintainers' `scripts/registry/landing/window.sh`
refuses to apply a plan that rebuilds a table unless `WINDOW_ALLOW_REBUILD=1` names a reviewed rebuild. It pins
the pending list with `WINDOW_EXPECT_MIGRATION=<NNNN_slug>` and waits for an approval
bound to the displayed plan by its token (`WINDOW_PLAN_APPROVAL=<file>`, the plan's sha256).

A rebuild must fold the record log exactly as the live database did, so a validator rule landing with or before a
migration must accept every record already in the log or list its hits in `validate.GRANDFATHERED_RECORD_SHAS`;
`scripts/registry/audit_new_validator_rules.py --root <root>` checks the rules it enumerates against that list;
any other rule needs its own whole-log audit.
