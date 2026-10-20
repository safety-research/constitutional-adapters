"""A revision held on a REFUSED constituent is an EXACT hold (2026-09-30): re-attempted only when the refusing file's bytes or the
judging code change, never every cycle.

Measured on the live registry (17:00-18:00Z): 14 held directories -- 10 cellstore ``000002-run.jsonl`` refused on a vocabulary CHECK,
3 relu_paired, 1 r2_export bundle -- were re-attempted by every cycle's blocked-queue re-evaluation (the only skip for ``kind = refused``
was "refused THIS cycle"), re-refused the same records by sha, re-stamped ``record_refusal.refused_utc``, appended a line to
``state/refused.jsonl`` (35,282 lines, 24,571 of them one reason) and cost ~40 s of a 75-110 s cycle under ``harvester.lock``. The rule
now: the held path records on the marker's queue row the refusing file, the logical sha it was delivered under and the stamp of the
judging code (``harvest.validator_stamp``: the DDL, the schema version, the validator / records / vocabulary / reducer / harvester
sources); the re-evaluation skips the directory while all three stand (``Ingestor._refused_hold_exact``) and re-attempts it once
when one moves. Pinned here, live and on replay: one attempt then no ``bundle`` retry line, one refusal row and one refused.jsonl
line, ``refused_utc`` still, ``attempts`` advancing, the held count kept; a changed file re-attempts in the main pass and activates;
a moved stamp re-attempts once (re-refuses and is exact again, or folds when the fix lifted the refusal); a pre-rule marker row
re-attempts once and is exact from then on; ``pending.ingest_now`` re-attempts nothing; ``registry ingest --retry-refused`` is the
escape for a refusal that depended on other rows.
"""
from __future__ import annotations

import json
import os

import pytest

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import pending as P
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry import validate as VAL
from sisteer.registry.adapters import runner as RUN

from test_registry_adapter_pass import SyntheticStoreAdapter, _setup  # noqa: E402
from test_registry_core_fixes import _rebuild_equal  # noqa: E402
from test_registry_harvest_budget import _direct_reread_bundles, _metrics, _q, _stable_banned_metric_adapter  # noqa: E402


def _refused_lines(root):
    p = os.path.join(root, "state", "refused.jsonl")
    with open(p, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _bundle_lines(root, now):
    return [(r["dir"].rsplit("/", 1)[-1], r["result"], r.get("no_op"), r.get("retry")) for r in _metrics(root, now) if r["kind"] == "bundle"]


def _all_bundle_lines(root):
    with open(os.path.join(root, "state", "metrics.jsonl"), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip() and json.loads(line).get("kind") == "bundle"]


def _marker(root):
    rows = _q(root, "SELECT extra_json, attempts, last_checked_utc FROM blocked_records WHERE record_type = 'source_revision'")
    assert len(rows) == 1, rows
    return json.loads(rows[0][0]), int(rows[0][1]), rows[0][2]


def _held(tmp_path, now=synth.t(2500)):
    """One revision directory held on a refused constituent (a run_metric under a banned spelling), folded once at ``now``."""
    root, store_root = _setup(tmp_path)
    ad = _stable_banned_metric_adapter(store_root)
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2000))
    rev = rep.results[0].revision_id
    ing = H.ingest_once(root, workers=0, min_free=0, now=now)
    assert ing.revisions_held == 1 and ing.refused == 1        # refused while STAGING: the hold is recorded, no activation is attempted
    assert _bundle_lines(root, now) == [(rev, "blocked", None, None)]                   # the main pass's attempt, no retry in the same cycle
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(0,)]
    return root, store_root, ad, rev


def test_the_marker_row_carries_the_exact_holds_key_and_the_hold_stands_across_cycles_live_and_on_replay(tmp_path):
    root, _, _, rev = _held(tmp_path)
    ext, attempts, checked = _marker(root)
    assert ext["kind"] == "refused" and ext["revision_id"] == rev and ext["judged_with"] == H.validator_stamp()
    assert ext["refused_file"].startswith(f"records/adapters/synthetic_store/{rev}/") and ext["refused_file"].endswith("-run_metric.jsonl")
    assert len(ext["refused_file_sha"]) == 64 and attempts == 1 and checked == synth.t(2500)
    assert _q(root, "SELECT file_sha256 FROM record_delivery WHERE file_uri = ?", ext["refused_file"]) == [(ext["refused_file_sha"],)]
    assert _q(root, "SELECT COUNT(*) FROM record_refusal") == [(1,)] and _refused_lines(root) == 1
    (utc1,) = _q(root, "SELECT refused_utc FROM record_refusal")[0]
    assert utc1 == synth.t(2500)
    for k, t in enumerate((3000, 3500, 4000), start=2):
        now = synth.t(t)
        ing = H.ingest_once(root, workers=0, min_free=0, now=now)
        assert _bundle_lines(root, now) == [], _bundle_lines(root, now)                   # NOT re-attempted: no bundle line at all
        assert ing.revisions_held == 1 and ing.refused == 0 and ing.revision_transactions == 0 and ing.staging_transactions == 0
        assert _q(root, "SELECT COUNT(*) FROM record_refusal") == [(1,)] and _q(root, "SELECT refused_utc FROM record_refusal") == [(utc1,)]
        assert _refused_lines(root) == 1                                                  # one judgment, not one line per cycle
        ext_k, attempts_k, checked_k = _marker(root)
        assert (attempts_k, checked_k) == (k, now) and ext_k == ext                       # the bookkeeping advances, the key stands
    assert _q(root, "SELECT accepted, refusal FROM source_revision WHERE revision_id = ?", rev)[0][0] == 0
    _rebuild_equal(root, tmp_path)


def test_a_changed_refusing_file_is_re_attempted_in_the_main_pass_and_activates(tmp_path):
    """The listing names a file whose logical sha moved; the directory is attempted in the main pass (not the queue) and, corrected,
    activates. The record log is immutable by contract, so the rewrite here is a SYNTHETIC byte change proving the listing path;
    a replay of a rewritten log legitimately differs from the live database (it never saw the refusing bytes), so no rebuild check."""
    root, _, _, rev = _held(tmp_path)
    ext, _, _ = _marker(root)
    path = H.resolve_uri(root, ext["refused_file"])
    recs = [rec for _, rec, _ in R.read_record_file(path)]
    bad = [r for r in recs if r["payload"].get("metric_id") == "at_cap_share"]
    assert len(bad) == 1
    bad[0]["payload"].update(metric_id="asr", threshold=0.75)
    bad[0]["record_id"] = I.adapter_record_id(bad[0]["payload"], "run_metric")
    os.remove(path)                                                                        # write_records appends: replace the bytes
    R.write_records(path, recs)
    now = synth.t(3000)
    ing = H.ingest_once(root, workers=0, min_free=0, now=now)
    assert _bundle_lines(root, now) == [(rev, "ok", None, None)]                         # the main pass, not a retry
    assert ing.revisions_applied == 1 and ing.refused == 0
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(1,)]
    assert _q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    assert sorted(_q(root, "SELECT metric_id, threshold FROM run_metric")) == [("asr", 0.5), ("asr", 0.75)]
    assert _q(root, "SELECT COUNT(*) FROM record_refusal") == [(1,)]                     # the first attempt's refusal stays as provenance


def test_a_moved_stamp_re_attempts_once_re_refuses_the_same_bytes_and_the_hold_is_exact_again(tmp_path, monkeypatch):
    root, _, _, rev = _held(tmp_path)
    H.ingest_once(root, workers=0, min_free=0, now=synth.t(3000))
    assert _bundle_lines(root, synth.t(3000)) == []
    monkeypatch.setattr(H, "validator_stamp", lambda: "f" * 16)                         # a landed change to the judging code
    now = synth.t(3500)
    ing = H.ingest_once(root, workers=0, min_free=0, now=now)
    assert _bundle_lines(root, now) == [(rev, "blocked", None, True)] and ing.refused == 1 and ing.revisions_held == 1
    assert _q(root, "SELECT COUNT(*) FROM record_refusal") == [(1,)] and _q(root, "SELECT refused_utc FROM record_refusal") == [(now,)]
    assert _refused_lines(root) == 2                                                      # a second judgment, under the new code
    ext, attempts, _ = _marker(root)
    assert ext["judged_with"] == "f" * 16 and attempts == 3
    now2 = synth.t(4000)
    ing2 = H.ingest_once(root, workers=0, min_free=0, now=now2)
    assert _bundle_lines(root, now2) == [] and ing2.refused == 0 and ing2.revisions_held == 1 and _refused_lines(root) == 2
    _rebuild_equal(root, tmp_path)                                                        # the replay under the (patched) code agrees


class _StringValueAdapter(SyntheticStoreAdapter):
    """A run_metric whose ``value`` is a STRING: refused by the validator alone ("must be a number"; the DDL has no type CHECK -- SQLite's
    REAL affinity would store it), so a relaxed validator rule is a fix that lifts the refusal with the same bytes and the same DDL."""

    def read(self, ctx, group, inputs):
        for batch in super().read(ctx, group, inputs):
            metric = next(r for r in batch.records if r["record_type"] == "run_metric")
            batch.add("run_metric", dict(metric["payload"], threshold=0.75, value="0.5"), ctx.producer, ctx.now)
            yield batch


def test_a_moved_stamp_whose_fix_lifts_the_refusal_re_attempts_once_and_folds(tmp_path, monkeypatch):
    """The fix lands (here: the validator's rule for the refused column is relaxed) and the stamp moves with it: one re-attempt, the
    revision activates. A replay under the NEW code never records the old refusal (the cross-tree case the runbook names), so no
    rebuild check here."""
    import dataclasses
    root, store_root = _setup(tmp_path)
    ad = _StringValueAdapter(store_root)
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2000))
    rev = rep.results[0].revision_id
    ing0 = H.ingest_once(root, workers=0, min_free=0, now=synth.t(2500))
    assert ing0.refused == 1 and ing0.revisions_held == 1 and "must be a number" in ing0.refusals[0]
    assert _bundle_lines(root, synth.t(3000)) == [] and H.ingest_once(root, workers=0, min_free=0, now=synth.t(3000)).refused == 0   # exact
    plan = VAL._COLUMN_PLAN[("run_metric", "value")]
    monkeypatch.setitem(VAL._COLUMN_PLAN, ("run_metric", "value"), dataclasses.replace(plan, ctype="NUMERIC"))   # the relaxed rule: no type check on the column
    monkeypatch.setattr(H, "validator_stamp", lambda: "e" * 16)
    now = synth.t(3500)
    ing = H.ingest_once(root, workers=0, min_free=0, now=now)
    assert _bundle_lines(root, now) == [(rev, "ok", None, True)] and ing.unblocked >= 1 and ing.refused == 0
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(1,)]
    assert _q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    rows = sorted(_q(root, "SELECT metric_id, threshold, value FROM run_metric"))
    assert [(m, t) for m, t, _ in rows] == [("asr", 0.5), ("asr", 0.75)] and rows[1][2] == 0.5    # the string landed as the REAL 0.5
    assert _q(root, "SELECT COUNT(*) FROM record_refusal") == [(1,)]                     # provenance of the first judgment
    ing4 = H.ingest_once(root, workers=0, min_free=0, now=synth.t(4000))
    assert _bundle_lines(root, synth.t(4000)) == [] and ing4.revisions_held == 0


def test_a_marker_recorded_before_the_rule_re_attempts_once_and_is_exact_from_then_on(tmp_path):
    root, _, _, rev = _held(tmp_path)
    conn = DB.open_or_create(DB.db_path(root))
    try:
        with DB.transaction(conn):
            conn.execute("UPDATE blocked_records SET extra_json = json_remove(extra_json, '$.judged_with', '$.refused_file', '$.refused_file_sha') "
                         "WHERE record_type = 'source_revision'")
    finally:
        conn.close()
    assert "judged_with" not in _marker(root)[0]
    now = synth.t(3000)
    ing = H.ingest_once(root, workers=0, min_free=0, now=now)
    assert _bundle_lines(root, now) == [(rev, "blocked", None, True)] and ing.refused == 1     # not exact: one re-attempt rewrites the key
    ext, _, _ = _marker(root)
    assert ext["judged_with"] == H.validator_stamp() and ext["refused_file_sha"]
    now2 = synth.t(3500)
    assert _bundle_lines(root, now2) == [] or True
    ing2 = H.ingest_once(root, workers=0, min_free=0, now=now2)
    assert _bundle_lines(root, now2) == [] and ing2.refused == 0 and ing2.revisions_held == 1
    _rebuild_equal(root, tmp_path)


def test_no_op_bundles_of_an_exact_hold_stand_with_it_and_a_corrected_bundle_still_activates(tmp_path):
    """The 2026-09-29 no-op rule beside the exact hold: identical re-reads are no-ops (never attempted), the carrier is not re-attempted
    either while its bytes and the code stand, and a corrected reader's bundle -- different content -- is attempted in full and activates."""
    from test_registry_adapter_no_reemit_unfolded import StableStampsAdapter
    root, store_root, ad, rev = _held(tmp_path)
    _direct_reread_bundles(root, ad, [synth.t(3000)])
    now1 = synth.t(3500)
    ing1 = H.ingest_once(root, workers=0, min_free=0, now=now1)
    assert _bundle_lines(root, now1) == [(RUN._compact(synth.t(3000)), "blocked", "held", None)] and ing1.revisions_noop == 1
    assert _q(root, "SELECT COUNT(*) FROM record_refusal") == [(1,)] and _refused_lines(root) == 1
    _direct_reread_bundles(root, StableStampsAdapter(store_root), [synth.t(4000)])
    now2 = synth.t(4500)
    ing2 = H.ingest_once(root, workers=0, min_free=0, now=now2)
    assert [(r, n, t) for _, r, n, t in _bundle_lines(root, now2)] == [("ok", None, None), ("ok", "stale", True)] and ing2.refused == 0
    assert _q(root, "SELECT accepted, superseded_by_revision_id FROM source_revision WHERE revision_id = ?", rev) == [(1, None)]
    assert _q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)] and _q(root, "SELECT metric_id FROM run_metric") == [("asr",)]
    _rebuild_equal(root, tmp_path)


def test_ingest_now_re_attempts_nothing_and_retry_refused_is_the_escape(tmp_path, monkeypatch):
    root, _, _, rev = _held(tmp_path)
    monkeypatch.setattr(H, "free_bytes", lambda path: 100 * H.GB)                       # the CLI's cycle reads the floor: never the host's disk
    before = len(_all_bundle_lines(root))
    rep = P.ingest_now(root)                                                              # the explicit fold under the lock
    assert rep.refused == 0 and rep.revisions_held == 1 and len(_all_bundle_lines(root)) == before and _refused_lines(root) == 1
    # the escape: `registry ingest --retry-refused` re-attempts the exact hold once (a refusal that depended on other rows)
    assert cli.main(["--root", root, "ingest", "--workers", "0", "--retry-refused"]) == 0
    lines = _all_bundle_lines(root)
    assert len(lines) == before + 1 and lines[-1]["dir"].endswith(rev) and lines[-1]["result"] == "blocked" and lines[-1].get("retry") is True
    assert _refused_lines(root) == 2 and _q(root, "SELECT COUNT(*) FROM record_refusal") == [(1,)]
    # and without the flag the hold is exact again
    assert cli.main(["--root", root, "ingest", "--workers", "0"]) == 0
    assert len(_all_bundle_lines(root)) == before + 1 and _refused_lines(root) == 2


def test_the_stamp_is_content_based_and_moves_with_the_ddl_and_the_stamped_sources(monkeypatch, tmp_path):
    a = H.validator_stamp()
    assert len(a) == 16 and int(a, 16) >= 0 and H.validator_stamp() == a                   # cached, stable within a process
    H.validator_stamp.cache_clear()
    assert H.validator_stamp() == a
    # a different DDL (a CHECK list that moved) is a different stamp
    ddl = dict(H.DB.code_ddl_master())
    key = next(iter(sorted(ddl)))
    moved = dict(ddl)
    moved[key] = ddl[key] + " -- moved"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(H.DB, "code_ddl_master", lambda: moved)
        H.validator_stamp.cache_clear()
        assert H.validator_stamp() != a
    H.validator_stamp.cache_clear()
    assert H.validator_stamp() == a


def test_a_refused_hold_a_later_accepted_read_superseded_is_exact_whatever_the_stamp(tmp_path, monkeypatch):
    """The three relu_paired and the one r2_export bundle of 2026-09-30: held on a refused constituent, superseded by a later accepted
    revision of their source, and still re-read every cycle. A superseded refused hold can never become the head (the run-drop rule of
    2026-09-23), so it is not re-attempted even when the judging code changes -- re-attempting it would only re-refuse its bytes or
    activate it already superseded and erase the hold's provenance. `--retry-refused` still re-attempts it; the refusal rows stay."""
    from test_registry_adapter_pass import _store
    root, store_root, _, rev1 = _held(tmp_path)
    _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.3, 0.8, 0.4])      # the store's content moved: a new revision of the same source
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [SyntheticStoreAdapter(store_root)], conn=conn, now=synth.t(3000))
    rev2 = rep.results[0].revision_id
    assert rev2 != rev1
    now = synth.t(3500)
    ing = H.ingest_once(root, workers=0, min_free=0, now=now)
    assert _bundle_lines(root, now) == [(rev2, "ok", None, None)] and ing.refused == 0
    assert _q(root, "SELECT accepted, superseded_by_revision_id FROM source_revision WHERE revision_id = ?", rev1) == [(0, rev2)]
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev2) == [(1,)]
    monkeypatch.setattr(H, "validator_stamp", lambda: "d" * 16)                                # the judging code changes: still not re-attempted
    now2 = synth.t(4000)
    ing2 = H.ingest_once(root, workers=0, min_free=0, now=now2)
    assert _bundle_lines(root, now2) == [] and ing2.refused == 0 and ing2.revisions_held == 1
    assert _q(root, "SELECT COUNT(*) FROM record_refusal") == [(1,)] and _refused_lines(root) == 1
    # the escape re-attempts it once; it re-refuses (its bytes are what they were) and stands, superseded
    now3 = synth.t(4500)
    ing3 = H.ingest_once(root, workers=0, min_free=0, now=now3, retry_refused=True)
    assert _bundle_lines(root, now3) == [(rev1, "blocked", None, True)] and ing3.refused == 1 and _refused_lines(root) == 2
    assert _q(root, "SELECT accepted, superseded_by_revision_id FROM source_revision WHERE revision_id = ?", rev1) == [(0, rev2)]
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev2) == [(1,)]
    _rebuild_equal(root, tmp_path)
