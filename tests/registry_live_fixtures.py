"""Shared read-only access to the LIVE registry for the ``live_db`` test modules.

Two of the modules that open ``~/registry/registry.sqlite`` scan ``v_paper_long`` — the frame object whose full
scan is the landing chain's one expensive read (452 s on a pristine copy; two scans ≈ 15 min of the 10.5-min-median
live-DB step, Q4.4). The session fixture ``live_paper_long`` scans it ONCE per pytest session and serves both:

* ``test_registry_paper_spelling`` renders the rows whose ``value`` is not NULL (filtered there, in Python);
* ``test_registry_paper_views_contrast_current`` derives its (metric key, anchor) pairs from the same rows and holds them
  against the withdrawn contrast keys read in the SAME snapshot.

One read transaction: the deployed object's CREATE text, the database's migration version, every row of the deployed
object and the withdrawn ``run_contrast`` observation keys are read inside one ``BEGIN … COMMIT`` on one read-only
connection (``db.reading``), so a commit that lands while a test consumes the snapshot (a harvest, a withdrawal) cannot
make the rows and the keys disagree — no consumer opens a second connection.

The scan is of the DEPLOYED object. Its normalized text is compared with the code's through the pair ``db.schema_drift``
uses (``db.ddl_master`` against ``db.code_ddl_master``): EQUAL → the one scan is a scan of the code's SQL as well.
DIFFERENT → allowed only while a migration is PENDING (the code's newest migration file is newer than the database's
version: the code moved, the live database has not taken it yet) — the contrast module then runs the candidate SQL's
invariants on its SYNTHETIC corpus and holds the live rows to the deployed object's invariants, selected by the
DEPLOYED VERSION, never by a substring of the candidate's text — and otherwise SCHEMA DRIFT, refused loudly
(:class:`PaperLongDrift`) BEFORE the scan: a deployed text that is neither the code's nor awaiting a declared migration
is a broken database, not a migration in flight. The fixture never runs the candidate SQL against the live database (a
second ~450 s scan) and never asserts equality (which would block every declared migration until the window applies it).

Registered ONCE, as a plugin (``pytest_plugins = ["registry_live_fixtures"]`` in ``tests/conftest.py``): a fixture
imported into two test modules is registered twice (one FixtureDef per module) and a session scope alone would scan
twice. ``REGISTRY_LIVE_DB`` overrides the database path (the one-scan rehearsal in ``tests/test_registry_landing_chain.py``
points it at a synthetic database).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import FrozenSet, Tuple

import pytest

LIVE_DB = os.path.expanduser(os.environ.get("REGISTRY_LIVE_DB", "~/registry/registry.sqlite"))
OBJECT = "v_paper_long"
# every run_contrast observation key no revision still holds current: the anchors the frame must not join — the contrast
# module's INDEPENDENT oracle for schema_def's WITHDRAWN_CONTRAST_CTE, kept as its own text on purpose
WITHDRAWN_CONTRAST_KEYS_SQL = ("SELECT observation_key FROM observation WHERE table_name = 'run_contrast' "
                               "GROUP BY observation_key HAVING MAX(current) = 0")


class PaperLongDrift(RuntimeError):
    """The deployed ``v_paper_long`` is neither the code's text nor awaiting a pending migration."""


def anchor_key(d: dict) -> tuple:
    """(run, judgment, metric, threshold, computation, anchor) — the key of ``run_contrast``, as the contrast module spells a pair."""
    return (str(d["run_id"]), str(d["judgment_id"]), str(d["metric_id"]), float(d["threshold"]), str(d["computation_id"]),
            str(d["anchor_run_id"]))


def withdrawn_anchor_keys(conn) -> FrozenSet[tuple]:
    return frozenset(anchor_key(json.loads(k)) for (k,) in conn.execute(WITHDRAWN_CONTRAST_KEYS_SQL))


@dataclass(frozen=True)
class LivePaperLong:
    """One snapshot of the deployed ``v_paper_long``: its rows, the texts and version that say what was scanned, the withdrawn keys."""
    db_path: str
    columns: Tuple[str, ...]
    rows: Tuple[dict, ...]                    # every row of the DEPLOYED object, column -> value, values as SQLite returned them
    deployed_sql: str                         # db.normalize_ddl(the deployed CREATE VIEW text), via db.ddl_master
    candidate_sql: str                        # the code's text, normalized the same way (db.code_ddl_master)
    migration_version: int                    # the database's version (migrations.db_version) in the snapshot
    code_version: int                         # the code's newest migration file (migrations.code_version)
    withdrawn_anchor_keys: FrozenSet[tuple]   # withdrawn_anchor_keys(conn) in the snapshot
    scan_s: float
    scanned_utc: str

    @property
    def candidate_is_deployed(self) -> bool:
        """True when the code's SQL and the deployed object have one normal form: the scan covers both."""
        return self.deployed_sql == self.candidate_sql

    @property
    def migration_pending(self) -> bool:
        """True when the code carries a migration the database has not applied yet."""
        return self.code_version > self.migration_version

    def status_line(self) -> str:
        which = ("the code's SQL IS the deployed text: this scan covers both" if self.candidate_is_deployed
                 else f"the code's SQL DIFFERS from the deployed text and migration {self.code_version} is pending: this scan is the "
                      "deployed object's; the candidate's invariants run on the synthetic corpus")
        return (f"live {OBJECT}: {len(self.rows)} rows in {self.scan_s:.1f} s at {self.scanned_utc} "
                f"(migration {self.migration_version}); {which}")


def load_live_paper_long(db_path: str = LIVE_DB) -> LivePaperLong:
    """The snapshot (module docstring). LookupError when ``db_path`` has no ``v_paper_long``; PaperLongDrift on drift, before any scan."""
    from sisteer.registry import db as DB           # imported here, not at module load: conftest registers this module at every
    from sisteer.registry import migrations as M    # session start and keeps its zero-import posture
    candidate = DB.code_ddl_master()[("view", OBJECT)]
    code_version = M.code_version()
    with DB.reading(db_path) as conn:
        conn.execute("BEGIN")   # one read transaction: the texts, the version, the rows and the keys are one snapshot
        try:
            deployed = DB.ddl_master(conn).get(("view", OBJECT))
            if deployed is None:
                raise LookupError(f"{db_path} has no {OBJECT}")
            version = M.db_version(conn)
            if deployed != candidate and code_version <= version:
                raise PaperLongDrift(f"SCHEMA DRIFT: the deployed {OBJECT} in {db_path} differs from the code's SQL and no migration is "
                                     f"pending (database at migration {version}, code at {code_version}) — not a migration in flight; "
                                     "run `registry status` (db.schema_drift) and repair the database before landing")
            t0 = time.time()
            cur = conn.execute(f"SELECT * FROM {OBJECT}")
            columns = tuple(str(d[0]) for d in cur.description)
            rows = tuple(dict(zip(columns, r)) for r in cur)
            scan_s = time.time() - t0
            withdrawn = withdrawn_anchor_keys(conn)
        finally:
            conn.execute("COMMIT")
    return LivePaperLong(db_path=db_path, columns=columns, rows=rows, deployed_sql=deployed, candidate_sql=candidate,
                         migration_version=version, code_version=code_version, withdrawn_anchor_keys=withdrawn,
                         scan_s=round(scan_s, 2), scanned_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


@pytest.fixture(scope="session")
def live_paper_long() -> LivePaperLong:
    """The one shared snapshot of the live ``v_paper_long`` (module docstring); skipped where the live registry is absent."""
    if not os.path.exists(LIVE_DB):
        pytest.skip("live registry is not on this machine")
    out = load_live_paper_long(LIVE_DB)
    print("\n" + out.status_line())
    return out
