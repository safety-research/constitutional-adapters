"""The paper frame derives the 95 percent Wilson score interval where a metric row of a yes-or-no proportion carries k and n, no filed
interval, and a value that IS k / n (migration 0019; `schema_def.WILSON_PROPORTION_METRIC_IDS`,
`WILSON_CI_LO_SQL` / `WILSON_CI_HI_SQL` / `WILSON_CI_METHOD_SQL`).

Before 0019 `v_paper_long` returned `run_metric.ci_lo` / `ci_hi` / `ci_method` as filed, so a reading whose writer filed k and n
without an interval reached every consumer with NULL bounds (6,670 live rows, 2026-10-01) and each consumer computed its own bar.
The rule, pinned here:

* the Wilson interval is the interval of a Bernoulli proportion, so the frame derives it only for a metric on the roster
  `WILSON_PROPORTION_METRIC_IDS` -- the ids whose writers file k as a count of items with a yes-or-no outcome and n as the items judged
  (`asr` and its ladder, `acc`, the refused rates, the BoN share rows), each read in its writer before it was listed. A metric off the
  roster derives nothing whatever its numbers: `over_refusal_graded` scores each item 0, 0.5 or 1 and files the score SUM as k, so a
  value that equals k / n exactly (a whole score sum: 561 ones + 28 halves + 728 zeros = 575 of 1,317; or 250 halves, value 0.5 at
  k 125) is a graded mean whose per-item variance a Bernoulli interval does not describe (the second review of the change; 19 live rows,
  5 of them in the current paper); its interval is adapters.declared_common.label_mean_interval, from the {0, 0.5, 1} census the frame
  does not hold;
* on the roster, a row with k and n > 0, no filed bound and a value equal to k / n (to 1e-9, `WILSON_VALUE_TOLERANCE`: the SCALE check)
  reads the Wilson bounds -- paperlib.stats.wilson / adapters.declared_common.wilson term for term at z = 1.959963984540054, so the SQL
  bound equals the Python bound to the last bit -- under `ci_method` wilson_95; k = 0 and k = n included (the sqrt term is z^2 / 4n^2
  there, in both);
* a value that is NOT k / n derives nothing (the first review of the change found live rows whose k and n describe another quantity: a
  percent-scale rate, value = 100 k / n; a difference filed beside one component's counts) -- the row keeps its NULL bounds and its filed
  method text, what it read before 0019, and a consumer that wants a bar there still computes its own on the value's scale;
* a filed interval is returned byte-identical under its filed `ci_method`, and a filed interval WITHOUT a method keeps its NULL method;
  a method text filed WITHOUT bounds on a row that derives (`none filed (rate over the judged pairs)`, 12 live rows) gives way to
  wilson_95, because the label describes the bounds in the row;
* k without n, n without k, n = 0 (the Python returns (nan, nan)) and a NULL value read the filed columns, NULL included;
* the multi-source refusal wraps the derived bounds as it wraps the filed ones (tests/test_registry_multi_source.py pins the refused
  owner row at k 4 / n 259 with no filed interval reading NULL bounds);
* `sqrt` is SQLite's math function (SQLITE_ENABLE_MATH_FUNCTIONS): the first math function in the schema, so the requirement is
  named by a test rather than discovered by a reader whose build lacks it.
"""
from __future__ import annotations

import os
import re
import sqlite3

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from sisteer.registry import vocab as V
from sisteer.registry.adapters.declared_common import OVER_REFUSAL_LABEL_MEAN_CI_METHOD, label_mean_interval, wilson

from test_registry_migrations import _insert_row, _valid_row   # the migration suite's own row builder + inserter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CI = ("ci_lo", "ci_hi", "ci_method")
#: the roster as it stands: a change here is a change to what the frame derives for, so it is a deliberate edit of both places
ROSTER = (
    "asr", "asr_held_in", "asr_held_out",
    "asr_at_n1", "asr_at_n2", "asr_at_n4", "asr_at_n8", "asr_at_n16", "asr_at_n32", "asr_at_n64", "asr_at_n75",
    "asr_bon_cap4096", "asr_bon_cap8192",
    "acc",
    "xst_refused", "orb_refused",
    "at_cap_share_bon_512", "coherent_at_cap_share_bon_512", "loop_share_bon_512", "secondary_flag_only_share_bon_512",
    "residual_at_cap_share_bon_4096",
)
#: metrics whose k and n are NOT a yes-or-no count, off the roster by design
OFF_ROSTER = ("over_refusal_graded", "over_refusal_label_mean", "over_refusal_graded_noncollapsed", "misaligned_rate",
              "recovered_vs_others_asr_gap", "petri_score", "mean_best_score")


def _corpus(tmp_path):
    """Two judged synthetic runs of 4 items: each files ONE `asr` row at threshold 0.5 with k = 1, n = 4, value 0.25 and no interval."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=4)
    H.ingest_once(root, workers=0, min_free=0)
    return root, ids


def _long(conn, run_id):
    """The frame's metric rows of a run, keyed by (metric_id, threshold): a probe on a roster name takes its own threshold."""
    return {(r["metric_id"], r["threshold"]): dict(r) for r in conn.execute(
        "SELECT metric_id, threshold, value, k, n, ci_lo, ci_hi, ci_method, metric_role FROM v_paper_long WHERE run_id = ?", (run_id,))}


def test_a_row_with_k_and_n_and_no_filed_interval_reads_the_wilson_bounds_under_wilson_95(tmp_path):
    root, ids = _corpus(tmp_path)
    conn = DB.connect(DB.db_path(root), readonly=True)
    lo, hi = wilson(1, 4)
    for rid in ids["run_ids"]:
        rows = _long(conn, rid)
        asr = rows[("asr", 0.5)]
        assert (asr["k"], asr["n"], asr["value"]) == (1, 4, 0.25)                      # the filed numbers are untouched; value = k / n
        assert asr["ci_lo"] == pytest.approx(lo, abs=1e-12) and asr["ci_hi"] == pytest.approx(hi, abs=1e-12)
        assert asr["ci_method"] == "wilson_95"
        # the table row itself is as the writer filed it: the derivation is the frame's, never written back
        assert tuple(conn.execute("SELECT ci_lo, ci_hi, ci_method FROM run_metric WHERE run_id = ? AND metric_id = 'asr'", (rid,)).fetchone()) == (None, None, None)
    conn.close()


def _probe_rows(conn, rid, jid, probes):
    """Extra metric rows on a run (direct inserts, as the migration suite seeds rows): ``probes`` maps a label to the row's metric_id,
    threshold and columns; returns the frame's reading of each label. Probes on one roster name take distinct thresholds (the key)."""
    for label, fields in probes.items():
        f = dict(fields)
        metric_id, threshold = f.pop("metric_id"), f.pop("threshold")
        _insert_row(conn, "run_metric", _valid_row("run_metric", run_id=rid, judgment_id=jid, metric_id=metric_id, threshold=threshold,
                                                   computed_by="producer", record_uri="synthetic://probe", ingest_seq=1, **f))
    conn.commit()
    rows = _long(conn, rid)
    out = {}
    for label, fields in probes.items():
        key = (fields["metric_id"], fields["threshold"])
        assert key in rows and rows[key]["metric_role"] == "owner", label
        out[label] = rows[key]
    return out


def _asr(threshold, **fields):
    return dict(metric_id="asr", threshold=threshold, **fields)


def test_filed_intervals_and_the_null_cases_on_the_frame(tmp_path):
    """On a roster name: a filed interval with a method, a filed interval without one, k = 0, k = n, k without n, n without k, n = 0,
    a NULL value."""
    root, ids = _corpus(tmp_path)
    rid, jid = ids["run_ids"][0], ids["judgment_ids"][0]
    conn = DB.connect(DB.db_path(root))
    conn.execute("PRAGMA foreign_keys=OFF")
    rows = _probe_rows(conn, rid, jid, {
        "filed": _asr(0.60, value=0.5, k=2, n=4, ci_lo=0.5, ci_hi=0.6, ci_method="bootstrap"),
        "filed_no_method": _asr(0.61, value=0.5, k=2, n=4, ci_lo=0.1, ci_hi=0.9, ci_method=None),
        "k_zero": _asr(0.62, value=0.0, k=0, n=4),
        "k_equals_n": _asr(0.63, value=1.0, k=4, n=4),
        "k_without_n": _asr(0.64, value=0.5, k=3, n=None),
        "n_without_k": _asr(0.65, value=0.5, k=None, n=4),
        "n_zero": _asr(0.66, value=0.5, k=3, n=0),
        "no_value": _asr(0.67, value=None, k=2, n=4),
    })
    # a filed interval: byte-identical, its method kept
    assert tuple(rows["filed"][c] for c in CI) == (0.5, 0.6, "bootstrap")
    # a filed interval without a method keeps its NULL method: wilson_95 is written only where the bounds are DERIVED
    assert tuple(rows["filed_no_method"][c] for c in CI) == (0.1, 0.9, None)
    # k = 0 and k = n derive (the sqrt term is z^2 / 4n^2 there, as in the Python)
    for label, k in (("k_zero", 0), ("k_equals_n", 4)):
        lo, hi = wilson(k, 4)
        r = rows[label]
        assert r["ci_lo"] == pytest.approx(lo, abs=1e-12) and r["ci_hi"] == pytest.approx(hi, abs=1e-12) and r["ci_method"] == "wilson_95"
    assert rows["k_zero"]["ci_lo"] == 0.0 and rows["k_equals_n"]["ci_hi"] == pytest.approx(1.0, abs=1e-12)
    # no denominator, no numerator, a zero denominator, or no value: nothing is derived and nothing is labeled
    for label in ("k_without_n", "n_without_k", "n_zero", "no_value"):
        assert tuple(rows[label][c] for c in CI) == (None, None, None), label
    conn.close()


def test_a_metric_off_the_roster_derives_nothing_whatever_its_value_on_the_frame(tmp_path):
    """The second review's rows, in synthetic form. `over_refusal_graded` scores each item 0, 0.5 or 1 and files the score SUM as k, so a
    whole score sum makes value = k / n exactly -- 561 ones + 28 halves + 728 zeros = 575 of 1,317 (a live row in the current paper), or
    250 halves, value 0.5 at k 125 (a live row with no item variation at all, for which the Bernoulli interval would read about
    [0.44, 0.56] while the label-mean interval of the same census is a point) -- and numeric equality says nothing about the estimand.
    The frame reads NULL on both, on the label mean whose effective count is whole, and on a name it has never seen; the SAME numbers
    under a roster name derive, because there k counts items with a yes-or-no outcome."""
    root, ids = _corpus(tmp_path)
    rid, jid = ids["run_ids"][0], ids["judgment_ids"][0]
    conn = DB.connect(DB.db_path(root))
    conn.execute("PRAGMA foreign_keys=OFF")
    rows = _probe_rows(conn, rid, jid, {
        "graded_whole_sum": dict(metric_id="over_refusal_graded", threshold=-1.0, value=575 / 1317, k=575, n=1317),
        "graded_all_halves": dict(metric_id="over_refusal_graded", threshold=0.5, value=0.5, k=125, n=250),
        "label_mean_whole": dict(metric_id="over_refusal_label_mean", threshold=-1.0, value=0.7, k=14, n=20),
        "unlisted_name": dict(metric_id="asr_probe_unlisted", threshold=0.5, value=0.25, k=1, n=4),
        "roster_same_numbers_a": dict(metric_id="xst_refused", threshold=0.5, value=575 / 1317, k=575, n=1317),
        "roster_same_numbers_b": _asr(0.60, value=0.5, k=125, n=250),
    })
    for label in ("graded_whole_sum", "graded_all_halves", "label_mean_whole", "unlisted_name"):
        assert tuple(rows[label][c] for c in CI) == (None, None, None), label
        assert rows[label]["value"] == pytest.approx(rows[label]["k"] / rows[label]["n"])   # value = k / n, and still nothing: the roster decides
    lo, hi = wilson(575, 1317)
    assert tuple(rows["roster_same_numbers_a"][c] for c in CI) == (pytest.approx(lo, abs=1e-12), pytest.approx(hi, abs=1e-12), "wilson_95")
    lo, hi = wilson(125, 250)
    assert tuple(rows["roster_same_numbers_b"][c] for c in CI) == (pytest.approx(lo, abs=1e-12), pytest.approx(hi, abs=1e-12), "wilson_95")
    # the numbers the review gives: the Bernoulli bound the graded row would have read, and the interval the census actually defines
    assert (lo, hi) == pytest.approx((0.4385, 0.5615), abs=5e-4)
    assert label_mean_interval(full=0, partial=250, n=250) == (0.5, 0.5)
    conn.close()


def test_a_value_that_is_not_k_over_n_derives_nothing_on_the_frame(tmp_path):
    """The first review's rows, in synthetic form: a percent-scale rate (value = 100 k / n) and a difference filed beside one component's
    k and n under the method text `none` -- off the roster besides -- and the SAME percent scale and a half-count value on a roster name,
    where the scale check alone refuses: each keeps NULL bounds and its filed method, what it read before 0019, so a consumer's own
    scaled fallback (paperlib.figs._point_intervals at value_scale 100) is not bypassed by a bound in the wrong units. Beside them, a
    proportion whose writer filed a method text without bounds derives, and the text gives way to wilson_95: the label describes the
    bounds in the row."""
    root, ids = _corpus(tmp_path)
    rid, jid = ids["run_ids"][0], ids["judgment_ids"][0]
    conn = DB.connect(DB.db_path(root))
    conn.execute("PRAGMA foreign_keys=OFF")
    rows = _probe_rows(conn, rid, jid, {
        "misaligned_rate": dict(metric_id="misaligned_rate", threshold=0.5, value=100.0 * 232 / 510, k=232, n=510),     # the percent scale (em_misalign's spelling)
        "gap": dict(metric_id="recovered_vs_others_asr_gap", threshold=0.5, value=-0.321011673151751, k=1, n=2, ci_method="none"),   # a difference beside one component
        "asr_percent_scale": _asr(0.60, value=100.0 * 232 / 510, k=232, n=510),      # a roster name filed as a percent: the scale check refuses
        "asr_half_count": _asr(0.61, value=12.5 / 250, k=12, n=250),                 # half a count off k / n on a roster name
        "orb_refused": dict(metric_id="orb_refused", threshold=0.5, value=0.5, k=2, n=4, ci_method="none filed (rate over the judged pairs)"),
    })
    for label in ("misaligned_rate", "asr_percent_scale", "asr_half_count"):
        assert tuple(rows[label][c] for c in CI) == (None, None, None), label
    assert tuple(rows["gap"][c] for c in CI) == (None, None, "none")
    assert rows["misaligned_rate"]["value"] == pytest.approx(45.490196078431374)                # the value itself is untouched
    lo, hi = wilson(2, 4)
    assert tuple(rows["orb_refused"][c] for c in CI) == (pytest.approx(lo, abs=1e-12), pytest.approx(hi, abs=1e-12), "wilson_95")
    conn.close()


_GRID = [(0, 1), (1, 1), (0, 10), (5, 10), (10, 10), (27, 259), (258, 259), (4, 259), (1, 100000), (99999, 100000)]


def _expression_rows(rows):
    """The three expressions evaluated over a table aliased `rm` (the alias the frame uses) with the given
    (metric_id, value, k, n, ci_lo, ci_hi, ci_method)."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE rm (metric_id TEXT, value REAL, k INTEGER, n INTEGER, ci_lo REAL, ci_hi REAL, ci_method TEXT)")
    conn.executemany("INSERT INTO rm VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    out = conn.execute(f"SELECT k, n, {S.WILSON_CI_LO_SQL}, {S.WILSON_CI_HI_SQL}, {S.WILSON_CI_METHOD_SQL} FROM rm").fetchall()
    conn.close()
    return out


def test_the_sql_equals_the_python_wilson_on_a_grid_including_the_live_check_rows():
    """Term for term the same formula at the same z: the bounds agree to 1e-12 across the grid (they agree exactly, bit for bit, on
    the grading host and on every live row that derives -- 6,611 of them, 2026-10-01); the two rows the ruling names read the values it gives."""
    got = _expression_rows([("asr", k / n, k, n, None, None, None) for k, n in _GRID])
    for (k, n, lo, hi, method) in got:
        plo, phi = wilson(k, n)
        assert lo == pytest.approx(plo, abs=1e-12) and hi == pytest.approx(phi, abs=1e-12), (k, n, lo, hi, plo, phi)
        assert method == "wilson_95"
    by_kn = {(k, n): (lo, hi) for k, n, lo, hi, _ in got}
    assert by_kn[(258, 259)] == pytest.approx((0.9784576202034251, 0.9993181125637599), abs=1e-12)   # the crescendo none run, asr @ 0.5
    assert by_kn[(27, 259)] == pytest.approx((0.0726384481874397, 0.14742371050021455), abs=1e-12)    # the pair constitution run, asr @ 0.5
    assert by_kn[(258, 259)] == pytest.approx((0.978, 0.999), abs=5e-4) and by_kn[(27, 259)] == pytest.approx((0.073, 0.147), abs=5e-4)
    # every roster name derives the same bound from the same numbers: the roster selects, it does not weight
    for m in ROSTER:
        (_, _, lo, hi, method), = _expression_rows([(m, 27 / 259, 27, 259, None, None, None)])
        assert (lo, hi, method) == (pytest.approx(0.0726384481874397, abs=1e-12), pytest.approx(0.14742371050021455, abs=1e-12), "wilson_95"), m


def test_the_expressions_on_the_null_filed_and_degenerate_rows():
    got = _expression_rows([
        ("asr", 0.3, None, 10, None, None, None),            # n without k
        ("asr", 0.3, 3, None, None, None, None),             # k without n
        ("asr", 0.3, 3, 0, None, None, None),                # n = 0: the Python returns (nan, nan); the frame reads NULL
        ("asr", None, 3, 10, None, None, None),              # no value: an interval about nothing is not derived
        ("asr", 0.5, 4, 8, 0.1, 0.9, None),                  # a filed interval without a method: kept, method stays NULL
        ("asr", 0.5, 4, 8, 0.5, 0.6, "bootstrap"),           # a filed interval with a method: kept
        ("asr", 0.5, 4, 8, 0.1, None, None),                 # one filed bound: both sides as filed, nothing derived on the empty side
        ("asr", None, None, None, None, None, "none filed"),  # a spelling that says no interval, and no k / n: the text is kept, nothing derived
        (None, 0.5, 4, 8, None, None, None),                 # no metric id at all: off the roster
    ])
    for i in (0, 1, 2, 3, 8):
        assert got[i][2:] == (None, None, None), got[i]
    assert got[4][2:] == (0.1, 0.9, None) and got[5][2:] == (0.5, 0.6, "bootstrap") and got[6][2:] == (0.1, None, None)
    assert got[7][2:] == (None, None, "none filed")
    # k > n is a writer error no CHECK refuses today: SQLite's sqrt of a negative is NULL, so the bounds read NULL rather than raising
    bad = _expression_rows([("asr", 9 / 8, 9, 8, None, None, None)])
    assert bad[0][2] is None and bad[0][3] is None


def test_the_expressions_derive_only_on_the_roster_and_only_where_the_value_is_k_over_n():
    """The two guards, row by row. The roster decides the estimand: the graded rows of the second review (a whole score sum, 250
    halves) read NULL at value = k / n exactly, and so does the percent-scale metric at k = 0 (value 0 is both 0 percent and 0 / 490; the
    first version of the guard, equality alone, derived there). The value decides the scale on a roster name: a percent-scale value, a
    half-count value and a value past the tolerance read NULL; a value computed by another route to the same proportion, within the
    tolerance, derives; k = 0 with value 0 derives on a roster name; a method text without bounds gives way to wilson_95 where the row
    derives, and a row that derives nothing keeps its text."""
    tol = float(S.WILSON_VALUE_TOLERANCE)
    assert tol == 1e-9 and S.WILSON_VALUE_TOLERANCE in S.WILSON_DERIVE_GUARD and S.WILSON_DERIVE_GUARD in S.WILSON_DERIVES
    got = _expression_rows([
        ("over_refusal_graded", 575 / 1317, 575, 1317, None, None, None),       # 0: the whole score sum, value = k / n: off the roster
        ("over_refusal_graded", 0.5, 125, 250, None, None, None),               # 1: 250 halves: off the roster
        ("over_refusal_label_mean", 0.7, 14, 20, None, None, None),             # 2: the label mean at a whole effective count: off the roster
        ("misaligned_rate", 0.0, 0, 490, None, None, None),                     # 3: k = 0, value 0 on the percent-scale metric: off the roster
        ("misaligned_rate", 100.0 * 232 / 510, 232, 510, None, None, None),     # 4: the percent scale
        ("recovered_vs_others_asr_gap", -0.321011673151751, 1, 2, None, None, "none"),   # 5: a difference beside one component's k and n
        ("asr", 100.0 * 232 / 510, 232, 510, None, None, None),                 # 6: a roster name filed as a percent: the scale check
        ("asr", 11.5 / 250, 12, 250, None, None, None),                         # 7: half a count below k / n
        ("asr", 12.5 / 250, 12, 250, None, None, None),                         # 8: half a count above
        ("asr", 0.5 + 2e-9, 5, 10, None, None, None),                           # 9: past the tolerance
        ("asr", 0.5 - 2e-9, 5, 10, None, None, None),                           # 10
        ("orb_refused", 0.5, 2, 4, None, None, "none filed (rate over the judged pairs)"),   # 11: a proportion whose writer filed a spelling and no bounds
        ("asr", 0.0, 0, 490, None, None, None),                                 # 12: k = 0, value 0, on a roster name: derives
        ("asr", sum([0.1] * 5), 5, 10, None, None, None),                       # 13: 0.1 summed five times: another route to 0.5, within the tolerance
    ])
    for i in (0, 1, 2, 3, 4, 6, 7, 8, 9, 10):
        assert got[i][2:] == (None, None, None), (i, got[i])
    assert got[5][2:] == (None, None, "none")
    lo, hi = wilson(2, 4)
    assert got[11][2:] == (pytest.approx(lo, abs=1e-12), pytest.approx(hi, abs=1e-12), "wilson_95")
    lo0, hi0 = wilson(0, 490)
    assert got[12][2:] == (0.0, pytest.approx(hi0, abs=1e-12), "wilson_95") and lo0 == 0.0
    lo5, hi5 = wilson(5, 10)
    assert got[13][2:] == (pytest.approx(lo5, abs=1e-12), pytest.approx(hi5, abs=1e-12), "wilson_95")


def test_the_roster_is_pinned_and_names_only_yes_or_no_proportions():
    """The roster is a deliberate list: each name was read in its writer (k a count of items with a yes-or-no outcome, n the items
    judged) before it was listed, and a change here changes what the frame derives for, so it is pinned twice. The graded over-refusal
    metrics, the percent-scale rate, the difference and the non-count metrics are off it; the roster rides into all three expressions,
    so the migration carries it; and the frame never writes the label mean's own method spelling, which is a different interval."""
    assert S.WILSON_PROPORTION_METRIC_IDS == ROSTER and len(set(ROSTER)) == len(ROSTER) == 21
    assert not set(ROSTER) & set(OFF_ROSTER)
    assert not set(ROSTER) & set(V.banned_metric_ids())
    assert S.WILSON_METRIC_GUARD == "rm.metric_id IN (" + ", ".join(f"'{m}'" for m in ROSTER) + ")"
    assert S.WILSON_DERIVE_GUARD.startswith(S.WILSON_METRIC_GUARD + " AND rm.k IS NOT NULL AND rm.n > 0 AND abs(rm.value - rm.k*1.0/rm.n) <= ")
    for expr in (S.WILSON_CI_LO_SQL, S.WILSON_CI_HI_SQL, S.WILSON_CI_METHOD_SQL):
        assert expr.count(S.WILSON_METRIC_GUARD) == 1
    assert S.VIEW_SQL["v_paper_long"].count(S.WILSON_METRIC_GUARD) == 3
    assert OVER_REFUSAL_LABEL_MEAN_CI_METHOD == "normal_95_exact_item_variance" and "normal_95" not in S.VIEW_SQL["v_paper_long"]


def test_z_is_the_pinned_z95_of_paperlib_and_the_adapters():
    # the z the paper's figure library pinned (paperlib/stats.py Z95; the file left the tree 2026-10-08 with the earlier figure
    # build) is held here as the literal, and the adapters' default must spell the same number
    assert S.WILSON_Z95 == repr(wilson.__defaults__[0]) == "1.959963984540054"
    # every z in a bound is the pin: z^2 twice in the center (numerator, d), z and z^2 twice in the half-width (z *, inside the sqrt, d)
    assert S.WILSON_CI_LO_SQL.count(S.WILSON_Z95) == 9 and S.WILSON_CI_HI_SQL.count(S.WILSON_Z95) == 9
    assert not re.search(r"1\.9[0-9]*", S.WILSON_CI_LO_SQL.replace(S.WILSON_Z95, ""))                       # and no other z spelling
    assert "centre" not in S.WILSON_CI_LO_SQL and "1.959964" not in S.VIEW_SQL["v_paper_long"]


def test_the_frame_wraps_the_derived_bounds_in_the_multi_source_refusal_and_keeps_the_column_order():
    sql = S.VIEW_SQL["v_paper_long"]
    refused = "CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE "
    assert f"{refused}{S.WILSON_CI_LO_SQL} END AS ci_lo" in sql and f"{refused}{S.WILSON_CI_HI_SQL} END AS ci_hi" in sql
    assert f"{S.WILSON_CI_METHOD_SQL} AS ci_method" in sql
    assert "rm.ci_lo END AS ci_lo" not in sql and "rm.ci_hi END AS ci_hi" not in sql and "\n       rm.ci_method," not in sql
    conn = sqlite3.connect(":memory:")
    conn.executescript(S.render_schema_sql())
    cols = [c[1] for c in conn.execute("PRAGMA table_info(v_paper_long)")]
    i = cols.index("value")
    assert cols[i:i + 6] == ["value", "k", "n", "ci_lo", "ci_hi", "ci_method"]
    assert conn.execute("SELECT COUNT(*) FROM v_paper_long").fetchone()[0] == 0        # the object compiles on this build
    conn.close()


def test_sqlite_provides_sqrt_the_first_math_function_the_schema_uses():
    """SQLITE_ENABLE_MATH_FUNCTIONS (SQLite 3.35+): the frame's object cannot compile without it (`no such function: sqrt`), and
    neither can `registry migrate` apply 0019. The grading host and the cluster gate venvs run Ubuntu's 3.45.1 with the option; a reader
    on another build learns it here, by name."""
    conn = sqlite3.connect(":memory:")
    assert conn.execute("SELECT sqrt(4.0)").fetchone()[0] == 2.0, f"SQLite {sqlite3.sqlite_version} lacks sqrt(): built without SQLITE_ENABLE_MATH_FUNCTIONS"
    assert conn.execute("SELECT sqrt(-1.0)").fetchone()[0] is None
    assert tuple(int(x) for x in sqlite3.sqlite_version.split(".")[:2]) >= (3, 35)
    conn.close()
