"""The `r2_export` <-> `sampled_lane` TWINS against the LIVE registry (a `live_db` module: read-only on
`~/registry/registry.sqlite`, skipped where it is absent; the landing chain's local step runs it, the cluster gate never sees it).

Section 5.1 MEASURED the twin set: `r2_export` runs at the 8,192 static instrument that twin a `sampled_lane` pooled run (`pass_label
pooled_t0_t1`) of the same `(model, object, eval, instrument)` and the same tag (`view_legacy`), both `in_current_paper`. Per export run the
export files `asr@{0.25, 0.5, 0.75}` with `k` and `n` = the pooled draws, plus degeneration shares; the pooled run files `asr@{0.5, 0.75}` only
with `k` NULL (the pooled `k_effective` rides `extra`) and `n` the goals; so `asr@0.5` and `asr@0.75` are DUPLICATE identities (value equal on
the same-tag pair, the pooled `extra.n_draws` = the export's `n`) and `asr@0.25` / the degeneration shares are UNIQUE to the export. Those facts
hold before and after the twin fold and are asserted here against the live rows.

The FOLD itself (section 5.2: `Ownership.pairing_rule view_legacy`, `check_rule value_with_draws`, `v_multi_source_rows` projecting `n_draws`,
the duplicate identities 2 -> 1 paper row as `folded`, the unique ones standing as `stopgap`, zero identities with both sources' rows in the paper)
is plan commit 7c's MIGRATION, 0026 `twin_fold_view_legacy_value_with_draws` on this tree (numbered max landed + 1 at the merge). Until the
deployed `v_multi_source_rows` carries `n_draws` (the window applies the file), the post-fold assertions SKIP by name against the live database --
``awaits migration twin_fold_view_legacy_value_with_draws`` -- rather than fail; the pre-fold facts never skip. The SAME post-fold assertions
(:func:`assert_post_fold`) RUN against a synthetic registry at the new shape in tests/test_registry_twin_fold.py, so the fold is exercised on
every tree and the live run is the deployment check alone.

Bounded against the live database: every assertion here reads v_paper_long in ONE fold. :func:`assert_post_fold` used to run a grouped
fold and then three ``... FROM v_paper_long WHERE run_id = ?`` reads PER PAIR -- each a full fold, since v_paper_long does not push the run_id
predicate down -- so 16 pairs cost ~49 folds (the follow-ups-2 landing's live step ran 4 h on it). It now folds v_paper_long once into a TEMP
table keyed by run_id (the paper's two sources' rows plus every row of the pairs' runs) and reads the per-pair rows from that; the bounded
rule ("at most one fold per query") holds per TEST. A TEMP table lives in the connection's own temp schema, which a ``mode=ro`` connection may
write (verified on a throwaway file, 2026-10-05). The pre-fold fixture's one read of v_run_current is a CTE two joins share, declared
MATERIALIZED so it is computed once by contract and not by the planner's default.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Dict, List, Tuple

import pytest

pytestmark = pytest.mark.live_db

# the live database (the landing chain's marker scan reads this spelling; `REGISTRY_LIVE_DB` overrides it, as in registry_live_fixtures)
LIVE_DB = os.environ.get("REGISTRY_LIVE_DB") or os.path.expanduser("~/registry/registry.sqlite")

TWIN_MIGRATION = "twin_fold_view_legacy_value_with_draws"      # plan commit 7c (section 5.2 + 10.6): migration 0026 on this tree
POOLED_PASS = "pooled_t0_t1"
DUPLICATE_THRESHOLDS = (0.5, 0.75)
EXPORT_ONLY_THRESHOLD = 0.25


def _conn() -> sqlite3.Connection:
    if not os.path.exists(LIVE_DB):
        pytest.skip("live registry is not on this machine")
    conn = sqlite3.connect(f"file:{LIVE_DB}?mode=ro", uri=True, timeout=120)
    conn.row_factory = sqlite3.Row
    return conn


def _fold_landed(conn: sqlite3.Connection) -> Tuple[bool, str]:
    """Whether the twin fold is deployed: the code's ownership declares the pairing rule AND the deployed view projects the pooled draws."""
    from sisteer.registry import ownership as OWN
    cols = [r[1] for r in conn.execute("PRAGMA table_info(v_multi_source_rows)")]
    code_ok = "view_legacy" in getattr(OWN, "PAIRING_RULES", ()) and hasattr(OWN.Ownership, "check_rule")
    db_ok = "n_draws" in cols
    why = f"code pairing rules {tuple(getattr(OWN, 'PAIRING_RULES', ()))}, check_rule {'declared' if hasattr(OWN.Ownership, 'check_rule') else 'absent'}; " \
          f"deployed v_multi_source_rows {'projects' if db_ok else 'lacks'} n_draws"
    return code_ok and db_ok, why


@pytest.fixture(scope="module")
def twins() -> Dict[str, object]:
    """The live twin set, read in ONE read transaction: every in-paper `r2_export` run at a static instrument whose `(model, object, eval, instrument,
    view_legacy)` a `sampled_lane` pooled run of the paper shares, with both sides' `asr` metric rows; bounded (the paper's current rows only)."""
    conn = _conn()
    conn.execute("BEGIN")
    try:
        rows = conn.execute("""
            WITH cur AS MATERIALIZED (SELECT run_id, in_current_paper FROM v_run_current WHERE in_current_paper = 1),
                 ex AS (SELECT r.run_id, r.model_id, r.object_id, r.eval_id, r.instrument_id, r.view_legacy, r.token_cap_tokens
                        FROM run r JOIN cur ON cur.run_id = r.run_id WHERE r.lane_source_legacy = 'r2_export' AND r.pass_label IS NOT 'pooled_t0_t1'),
                 po AS (SELECT r.run_id, r.model_id, r.object_id, r.eval_id, r.instrument_id, r.view_legacy
                        FROM run r JOIN cur ON cur.run_id = r.run_id WHERE r.lane_source_legacy = 'sampled_lane' AND r.pass_label = ?)
            SELECT ex.run_id AS export_run, po.run_id AS pooled_run, ex.model_id, ex.object_id, ex.eval_id, ex.instrument_id, ex.view_legacy, ex.token_cap_tokens
            FROM ex JOIN po ON po.model_id = ex.model_id AND po.object_id = ex.object_id AND po.eval_id = ex.eval_id AND po.instrument_id = ex.instrument_id
                            AND po.view_legacy = ex.view_legacy
            ORDER BY ex.model_id, ex.eval_id, ex.object_id, ex.view_legacy""", (POOLED_PASS,)).fetchall()
        pairs = [dict(r) for r in rows]
        run_ids = sorted({p["export_run"] for p in pairs} | {p["pooled_run"] for p in pairs})
        metrics: Dict[str, List[dict]] = {}
        if run_ids:
            q = ",".join("?" * len(run_ids))
            for m in conn.execute(f"SELECT run_id, metric_id, threshold, computation_id, value, k, n, extra_json FROM run_metric WHERE run_id IN ({q}) AND metric_id = 'asr'", run_ids):
                metrics.setdefault(m["run_id"], []).append(dict(m))
        landed, why = _fold_landed(conn)
        version = conn.execute("SELECT MAX(version) FROM schema_migration").fetchone()[0]
    finally:
        conn.execute("COMMIT")
        conn.close()
    return {"pairs": pairs, "metrics": metrics, "fold_landed": landed, "fold_why": why, "db_version": version}


def _by_threshold(rows: List[dict]) -> Dict[float, dict]:
    out: Dict[float, dict] = {}
    for r in rows:
        out.setdefault(float(r["threshold"]), r)      # the first row per threshold (the export files one per identity)
    return out


def test_the_live_twin_set_is_the_measured_one_each_export_run_twins_exactly_one_pooled_run_of_its_own_tag(twins):
    pairs = twins["pairs"]
    assert pairs, f"no r2_export run twins a sampled_lane pooled run in the paper today (database at migration {twins['db_version']})"
    per_export: Dict[str, List[str]] = {}
    for p in pairs:
        per_export.setdefault(p["export_run"], []).append(p["pooled_run"])
    assert all(len(v) == 1 for v in per_export.values()), {k: v for k, v in per_export.items() if len(v) != 1}
    assert all(p["token_cap_tokens"] in (8192, None) for p in pairs), sorted({p["token_cap_tokens"] for p in pairs})
    assert {p["object_id"] == "none" for p in pairs} <= {True, False} and len(per_export) >= 12, f"the section-5 census: 12 none twins + 4 steered; found {len(per_export)}"


def test_the_duplicate_identities_agree_on_value_and_draws_and_the_pooled_side_files_no_count(twins):
    """Section 5.1 / 5.2: at asr@0.5 and asr@0.75 the two readings carry one value (to 1e-9), the pooled row's `extra.n_draws` equals the export's
    `n`, the pooled `k` is NULL (the fold's `value_with_draws` predicate, not the exact-count one); asr@0.25 is the export's alone."""
    pairs, metrics = twins["pairs"], twins["metrics"]
    checked = 0
    for p in pairs:
        ex, po = _by_threshold(metrics.get(p["export_run"], [])), _by_threshold(metrics.get(p["pooled_run"], []))
        for thr in DUPLICATE_THRESHOLDS:
            if thr not in ex or thr not in po:
                continue
            e, q = ex[thr], po[thr]
            assert e["value"] is not None and q["value"] is not None and abs(float(e["value"]) - float(q["value"])) <= 1e-9, (p, thr, e["value"], q["value"])
            n_draws = json.loads(q["extra_json"] or "{}").get("n_draws")
            assert q["k"] is None and e["k"] is not None and n_draws == e["n"], (p, thr, q["k"], e["k"], n_draws, e["n"])
            checked += 1
        assert EXPORT_ONLY_THRESHOLD not in po, f"the pooled reader files no asr@{EXPORT_ONLY_THRESHOLD}: {p}"
    assert checked > 0, "no duplicate identity carried both readings"


TWINS_TEMP_TABLE = "twins_paper_long"      # the ONE fold of v_paper_long the post-fold assertions read


def fold_paper_long_once(conn: sqlite3.Connection, run_ids: List[str]) -> str:
    """Fold v_paper_long ONCE into a TEMP table and return its name: the paper's `r2_export` / `sampled_lane` rows (what the
    both-sources check groups) plus EVERY row of ``run_ids`` whatever its `in_current_paper` (what the per-pair checks count: the export's
    asr@0.5 / asr@0.75 rows that must have LEFT the frame are counted over all of the run's rows, not the paper's), with the columns the
    assertions read, indexed by (run_id, metric_id). The temp schema is the connection's own, so a ``mode=ro`` connection may create it;
    a table left by an earlier call on the same connection is dropped first."""
    assert run_ids, "no run to fold the frame for"
    q = ",".join("?" * len(run_ids))
    conn.execute(f"DROP TABLE IF EXISTS temp.{TWINS_TEMP_TABLE}")
    conn.execute(f"""CREATE TEMP TABLE {TWINS_TEMP_TABLE} AS
                     SELECT run_id, model_id, object_id, eval_id, instrument_id, view_legacy, lane_source_legacy, metric_id, threshold, value,
                            in_current_paper, multi_source_state, multi_source_secondary_run_id, multi_source_reason
                     FROM v_paper_long
                     WHERE (in_current_paper = 1 AND lane_source_legacy IN ('r2_export', 'sampled_lane')) OR run_id IN ({q})""", list(run_ids))
    conn.execute(f"CREATE INDEX temp.{TWINS_TEMP_TABLE}_run ON {TWINS_TEMP_TABLE} (run_id, metric_id)")
    return TWINS_TEMP_TABLE


def assert_post_fold(conn: sqlite3.Connection, pairs: List[dict]) -> None:
    """Section 5.2 (4) / 5.3 over a connection at the new shape: zero identities with both sources' rows `in_current_paper`; for every twin pair
    the export's asr@0.5 / asr@0.75 rows have LEFT the frame and the pooled run's two carry `multi_source_state folded` naming the export run as
    the secondary (the owner row is the one paper row); the export's asr@0.25 stands `in_current_paper` as a `stopgap`. A same-tag pair whose
    readings disagree reads `refused` on the pooled side -- a disagreement to rule on, reported by name, never silently folded.
    ONE fold of v_paper_long (:func:`fold_paper_long_once`): every read below goes to the temp table, none back to v_paper_long."""
    assert pairs, "no twin pair to check"
    run_ids = sorted({p["export_run"] for p in pairs} | {p["pooled_run"] for p in pairs})
    t = fold_paper_long_once(conn, run_ids)
    try:
        # `n` is a column of v_paper_long (the metric's count), so the aggregate is repeated in HAVING rather than aliased to it: the first
        # draft's `HAVING n > 1` read every row's count (the fixture run of this function found it)
        both = conn.execute(f"""SELECT model_id, object_id, eval_id, instrument_id, view_legacy, metric_id, threshold, COUNT(DISTINCT lane_source_legacy) AS n_sources
                                FROM {t} WHERE in_current_paper = 1 AND value IS NOT NULL AND lane_source_legacy IN ('r2_export', 'sampled_lane')
                                GROUP BY 1,2,3,4,5,6,7 HAVING COUNT(DISTINCT lane_source_legacy) > 1""").fetchall()
        assert both == [], [tuple(r) for r in both[:5]]
        for p in pairs:
            gone = conn.execute(f"SELECT COUNT(*) FROM {t} WHERE run_id = ? AND metric_id = 'asr' AND threshold IN (0.5, 0.75)", (p["export_run"],)).fetchone()[0]
            folded = [tuple(r) for r in conn.execute(f"""SELECT threshold, multi_source_state, multi_source_secondary_run_id, multi_source_reason FROM {t}
                                                          WHERE run_id = ? AND metric_id = 'asr' AND threshold IN (0.5, 0.75) ORDER BY threshold""", (p["pooled_run"],))]
            assert gone == 0 and len(folded) == 2, (p, gone, folded)
            assert all(r[1] == "folded" and r[2] == p["export_run"] for r in folded), (p, [(r[0], r[1], (r[3] or "")[:160]) for r in folded])
            stop = conn.execute(f"SELECT multi_source_state, in_current_paper FROM {t} WHERE run_id = ? AND metric_id = 'asr' AND threshold = 0.25",
                                (p["export_run"],)).fetchone()
            assert stop is not None and tuple(stop) == ("stopgap", 1), (p, stop)
    finally:
        conn.execute(f"DROP TABLE IF EXISTS temp.{t}")


def test_after_the_twin_fold_each_duplicate_identity_has_one_paper_row_and_the_unique_export_identities_stand(twins):
    """Section 5.2 (4) / 5.3 against the LIVE database (:func:`assert_post_fold`). Skips by name until the fold's migration (plan commit 7c,
    0026) is deployed; the same assertions run on a fixture registry in tests/test_registry_twin_fold.py."""
    if not twins["fold_landed"]:
        pytest.skip(f"awaits migration {TWIN_MIGRATION}: {twins['fold_why']}")
    conn = _conn()
    try:
        assert_post_fold(conn, twins["pairs"])
    finally:
        conn.close()
