"""The measurement scripts on synthetic fixtures -- stdlib scripts loaded from
their path, every one with ``--json``:

* ``fold_latency.py`` -- S7's four named cases on a fresh schema database with ledger rows written the way the harvester writes them:
  a revision delivered ``ok`` and accepted completes at its ``ingested_utc``; a held revision (marker ``blocked``, placeholder
  ``accepted = 0``) is censored; a held-THEN-activated revision's latency runs to the activation's ``ok`` delivery, not the hold -- by
  the REAL path (the directory's marker row stays ``blocked``, the correcting ``_activations/<rev>/<stamp>/`` bundle's own marker row is
  the ``ok``, a later receipt pulled the bundle: ONE sample across the two receipts, never a censored one plus a completed one); a
  successfully folded revision later superseded by another successful one -- BOTH count, each at its own ``ingested_utc``, and the
  supersession / reclaim changes neither; the successor re-activated from committed bytes for a NEW pin is the new pin's sample, clocked
  to the bundle, the directory's earlier ``ok`` completing nothing of it; a pin is BOUND to the markers its own operations listed (the
  directory for ``pulled``, the named bundle for ``bundle``, the bundles stamped between the operation's attestation and its receipt for
  ``reactivated``) and an earlier generation's ``ok`` delivered between the new pin's change and its receipt completes nothing of the new
  pin, whose own bundle's delivery does -- even one the harvester folded before the receipt was written; an entry whose successful shard's
  listings stand beside a sibling shard's refusal keeps the listings as samples judged as usual and records the refusal as its own row
  (``refused_shards``), the all-shards-refused entry still censoring ``refused_operation``. Plus: an undelivered revision, a pin with
  no PUBLISHED record and a delivery after the observation's end are censored by name; the clock is ``change_utc``, never
  ``published_utc``; the class is the publish mode; the database is opened read-only.
* ``changed_set.py`` -- the tick's op record (the reasons by kind) and the window's receipt, joined per operation.
* ``benchmarks.py`` -- the levers from a synthetic chain log, shard summaries, the pull stage's attestation records (plans and receipts,
  one entry per attestation, time-filtered) and an R14 record that is baseline evidence only; targets applied; ``--write`` merges.
* ``figures.py`` -- every figure renders from the fixtures (PNG magic, non-trivial size), and a figure whose input is absent is skipped
  by name (needs matplotlib; skipped when the venv has none).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import subprocess
import sys

import pytest

from sisteer.registry import db as DB

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEASURE = os.path.join(REPO, "scripts", "registry", "measure")


def _load(name):
    spec = importlib.util.spec_from_file_location(f"measure_{name}_slurm_only", os.path.join(MEASURE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# --------------------------------------------------------------------------------------------
# fold latency (S7)
# --------------------------------------------------------------------------------------------

T0 = "2026-10-10T10:00:00.000Z"          # the observation's start
ADAPTER = "alpha"
REV_OK, REV_HELD, REV_LATE, REV_SUP1, REV_SUP2, REV_UNDELIVERED = ("a" * 16, "b" * 16, "c" * 16, "d" * 16, "e" * 16, "f" * 16)


#: a bundle directory's name is ``_compact(now)`` of the pass that wrote it (runner.py), 17 digits: REV_LATE's corrected re-read bundle was
#: written on the cluster at 10:33 (pulled by tick-1b at 10:35, delivered 10:40); REV_SUP2's re-activation bundle by tick-5's operation at
#: 11:44 (attested 11:43, receipt 11:45; delivered 12:30)
STAMPS = {REV_LATE: "20261010103300000", REV_SUP2: "20261010114400000"}


def _marker_uri(rev: str, bundle: bool = False, stamp: str = None) -> str:
    return f"records/adapters/{ADAPTER}/" + (f"_activations/{rev}/{stamp or STAMPS[rev]}/" if bundle else f"{rev}/") + "000003-source_revision.jsonl"


def _deliver(conn, uri: str, status: str, ingested: str) -> None:
    """A delivery row as ``harvest._deliver`` writes it (an upsert by file_uri: the latest status wins)."""
    conn.execute("INSERT INTO record_delivery (file_uri, file_sha256, arrived_utc, ingested_utc, status, record_type_counts_json, n_records) "
                 "VALUES (?,?,?,?,?,?,?) ON CONFLICT(file_uri) DO UPDATE SET status = excluded.status, ingested_utc = excluded.ingested_utc",
                 (uri, "0" * 64, ingested, ingested, status, "{}", 1))


def _revision(conn, rev: str, accepted: int, gen: str, observed: str, refusal=None, superseded_by=None) -> None:
    extra = json.dumps({"read_copy": {"components": {"comp": {"generation": gen, "route": "generation"}}}})
    conn.execute("INSERT INTO source_revision (revision_id, source_id, adapter, observed_utc, inputs_json, accepted, refusal, superseded_by_revision_id, "
                 "record_uri, ingest_seq, extra_json) VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(revision_id) DO UPDATE SET accepted = excluded.accepted, "
                 "refusal = excluded.refusal, superseded_by_revision_id = excluded.superseded_by_revision_id",
                 (rev, "src", ADAPTER, observed, "{}", accepted, refusal, superseded_by, _marker_uri(rev), 1, extra))


def _receipt(op_id: str, utc: str, adapters: dict, attestation_utc: str = None) -> dict:
    """A receipt as ``reads_protocol.receipt_from_plan`` writes it; ``attestation`` is the pull stage's ``attestation_summary`` (its ``utc``
    the shard's admission check before the read -- the operation's earliest stamp), None when the fixture's case needs none."""
    att = {"present": True, "window": None, "this_window": True, "ok": True, "utc": attestation_utc, "host": "TESTHOST", "refused": [], "generations": {}}
    return {"schema": "reads_receipt/1", "op": {"op_id": op_id, "owner": "tick"}, "utc": utc, "window": None, "tree_sha": "t" * 40,
            "destination": "cluster_a", "jobs": {}, "attestation": (att if attestation_utc else None), "barrier": None, "adapters": adapters,
            "n_read_whole": sum(1 for a in adapters.values() if a.get("read_whole")), "n_refused": sum(1 for a in adapters.values() if a.get("refused"))}


def _entry(pins: dict, published: dict, pulled, reactivated=(), refused=None, shards_ok=None) -> dict:
    """A receipt entry as ``receipt_from_plan`` writes it: two shards; a refusal beside listings is a mixed-success entry (one shard ok, the
    other refused -- the pull withholds only the refusing shard's publications), a refusal with no listing is the adapter refused whole."""
    return {"route": "cluster", "pins": pins, "published": published, "module_blobs": {}, "shards_total": 2,
            "shards_ok": (shards_ok if shards_ok is not None else (2 if not refused else 1)),
            "revisions_pulled": [f"{ADAPTER}/{r}" for r in pulled], "reactivated": list(reactivated), "refused": refused, "read_whole": not refused}


@pytest.fixture
def world(tmp_path):
    return _latency_world(tmp_path)


def _latency_world(tmp_path):
    """A fresh schema database with the ledger rows of the module doc's cases, the receipts that pulled them, and the paths."""
    root = tmp_path / "reg"
    root.mkdir()
    conn = DB.open_fresh(str(root / "registry.sqlite"))
    pub_push = {"published_utc": "2026-10-10T10:05:00.000Z", "change_utc": "2026-10-10T10:00:00.000Z", "mode": "push", "producer": "grading_host:x"}
    pub_carry = {"published_utc": "2026-10-10T11:05:00.000Z", "change_utc": "2026-10-10T11:00:00.000Z", "mode": "carry", "producer": None}
    pub_carry2 = {"published_utc": "2026-10-10T11:42:00.000Z", "change_utc": "2026-10-10T11:40:00.000Z", "mode": "carry", "producer": None}
    # (1) delivered ok and accepted -> completed at its ingested_utc (12 min after the change)
    _revision(conn, REV_OK, 1, "11111111", "2026-10-10T10:08:00.000Z")
    _deliver(conn, _marker_uri(REV_OK), "ok", "2026-10-10T10:12:00.000Z")
    # (2) held: marker blocked, placeholder accepted = 0 -> censored
    _revision(conn, REV_HELD, 0, "11111111", "2026-10-10T10:08:00.000Z", refusal="activation pending: run r1 (missing)")
    _deliver(conn, _marker_uri(REV_HELD), "blocked", "2026-10-10T10:12:00.000Z")
    # (3) held THEN activated -- the REAL path (harvest._apply_revision / _activate_revision): the DIRECTORY's marker row was delivered
    #     blocked at 10:12 and STAYS blocked; the corrected re-read arrived as an _activations/<rev>/1/ bundle (pulled by the next tick),
    #     whose OWN marker row is delivered ok at 10:40, and that activation sets accepted = 1 on the shared row -> 40 min, not 12
    _revision(conn, REV_LATE, 0, "11111111", "2026-10-10T10:08:00.000Z", refusal="activation pending: run r2 (missing)")
    _deliver(conn, _marker_uri(REV_LATE), "blocked", "2026-10-10T10:12:00.000Z")
    _deliver(conn, _marker_uri(REV_LATE, bundle=True), "ok", "2026-10-10T10:40:00.000Z")
    _revision(conn, REV_LATE, 1, "11111111", "2026-10-10T10:08:00.000Z")
    # (4) folded (11:12) then superseded by a later successful revision (11:30); reclaim marked the first -> BOTH completed
    _revision(conn, REV_SUP1, 1, "22222222", "2026-10-10T11:08:00.000Z", superseded_by=REV_SUP2)
    conn.execute("UPDATE source_revision SET reclaimed = 1 WHERE revision_id = ?", (REV_SUP1,))
    _deliver(conn, _marker_uri(REV_SUP1), "ok", "2026-10-10T11:12:00.000Z")
    _revision(conn, REV_SUP2, 1, "22222222", "2026-10-10T11:25:00.000Z")
    _deliver(conn, _marker_uri(REV_SUP2), "ok", "2026-10-10T11:30:00.000Z")
    # (5) the successor RE-ACTIVATED from committed bytes for a NEW pin of the component (33333333, change 11:40): the shard refused to
    #     re-read a known revision, the grading host wrote the _activations/<rev>/<stamp 11:44>/ bundle, delivered ok at 12:30 -> the new pin's
    #     sample, 50 min; the directory's 11:30 ok is not a marker the re-activating operation listed and completes nothing of it (nor would
    #     it after the new clock -- the pin is bound to its own operation's bundle); the row's read_copy still names the original
    #     read's generation (22222222, by construction) -> marker_generation beside the pin, a pin_mismatch note, no censor
    _deliver(conn, _marker_uri(REV_SUP2, bundle=True), "ok", "2026-10-10T12:30:00.000Z")
    conn.close()
    state = tmp_path / "state"
    rd = state / "reads" / "receipts"
    rd.mkdir(parents=True)
    r1 = _receipt("tick-1", "2026-10-10T10:10:00.000Z", {ADAPTER: _entry({"comp": "11111111"}, {"comp": pub_push}, [REV_OK, REV_HELD, REV_LATE, REV_UNDELIVERED])})
    # the next tick pulled the corrected re-read's bundle of REV_LATE: the same pin, listed a second time, as a bundle
    r1b = _receipt("tick-1b", "2026-10-10T10:35:00.000Z", {ADAPTER: _entry({"comp": "11111111"}, {"comp": pub_push}, [f"_activations/{REV_LATE}/{STAMPS[REV_LATE]}"])})
    r2 = _receipt("tick-2", "2026-10-10T11:10:00.000Z", {ADAPTER: _entry({"comp": "22222222"}, {"comp": pub_carry}, [REV_SUP1])})
    r3 = _receipt("tick-3", "2026-10-10T11:28:00.000Z", {ADAPTER: _entry({"comp": "22222222"}, {"comp": pub_carry}, [REV_SUP2])})
    r5 = _receipt("tick-5", "2026-10-10T11:45:00.000Z", {ADAPTER: _entry({"comp": "33333333"}, {"comp": pub_carry2}, [], reactivated=[REV_SUP2])},
                  attestation_utc="2026-10-10T11:43:00.000Z")
    # an adapter refused WHOLE (both shards refused, nothing listed): nothing counts, the pin is on the record; and a pin with no PUBLISHED
    # record: no clock
    r4 = _receipt("tick-4", "2026-10-10T12:00:00.000Z", {"beta": _entry({"other": "33333333"}, {"other": pub_push}, [], refused=["pull: 2 shard(s) refused"], shards_ok=0),
                                                          "gamma": _entry({"nopub": "44444444"}, {"nopub": None}, [REV_OK])})
    for r in (r1, r1b, r2, r3, r5, r4):
        (rd / f"{r['op']['op_id']}.json").write_text(json.dumps(r))
    (rd / "foreign.json").write_text(json.dumps({"schema": "other/1"}))
    (rd / "torn.json").write_text("{not json")
    return {"root": str(root), "state": str(state), "db": str(root / "registry.sqlite")}


def test_fold_latency_completes_at_the_ok_delivery_censors_the_held_and_the_undelivered_and_names_the_rest(world):
    FL = _load("fold_latency")
    since, until = FL.parse_utc(T0), FL.parse_utc("2026-10-11T10:00:00.000Z")
    t = FL.measure(world["root"], world["state"], since, until, tick_min=10.0)
    assert t["receipts"] == 6 and t["receipts_unreadable"] == 2 and t["db_error"] is None
    by = {(s["op_id"], s["revision"], s.get("component"), s.get("how")): s for s in t["samples"]}
    ok = by[("tick-1", REV_OK, "comp", "pulled")]
    # (1) completed at the ok delivery's ingested_utc: 10:12 - change 10:00 = 12 min; the class is the publish mode; the clock is change_utc
    assert ok["outcome"] == "completed" and ok["latency_s"] == 720.0 and ok["class"] == "push" and ok["change_utc"] == "2026-10-10T10:00:00.000Z"
    assert ok["ingested_utc"] == "2026-10-10T10:12:00.000Z" and ok["accepted"] == 1 and ok["status"] == "ok" and ok["marker_uri"] == _marker_uri(REV_OK)
    assert ok["op_ids"] == ["tick-1"] and ok["ways"] == ["pulled"] and ok["n_markers"] == 1 and ok["marker_generation"] == "11111111" and "pin_mismatch" not in ok
    # (2) held: the marker delivered blocked, the placeholder accepted = 0 -> censored, by name, resting on the directory's row
    held = by[("tick-1", REV_HELD, "comp", "pulled")]
    assert held["outcome"] == "censored" and held["censor"] == "marker_blocked" and held["latency_s"] is None and held["accepted"] == 0
    assert held["status"] == "blocked" and held["marker_uri"] == _marker_uri(REV_HELD)
    # (3) held then activated by the REAL path: the directory's marker row STILL reads blocked in the ledger, the bundle's own row is the
    #     ok (distinct URIs), and the revision is listed by two receipts (tick-1 the directory, tick-1b the bundle) -> ONE sample, completed
    #     at the activation's ok delivery (10:40), never a censored sample for the directory plus a completed one for the bundle
    assert _marker_uri(REV_LATE) != _marker_uri(REV_LATE, bundle=True)
    with sqlite3.connect("file:" + world["db"] + "?mode=ro", uri=True) as ro:
        assert ro.execute("SELECT status FROM record_delivery WHERE file_uri = ?", (_marker_uri(REV_LATE),)).fetchone()[0] == "blocked"
        assert ro.execute("SELECT status FROM record_delivery WHERE file_uri = ?", (_marker_uri(REV_LATE, bundle=True),)).fetchone()[0] == "ok"
    lates = [s for s in t["samples"] if s["revision"] == REV_LATE]
    assert len(lates) == 1
    late = by[("tick-1", REV_LATE, "comp", "pulled")]
    assert late is lates[0] and late["outcome"] == "completed" and late["latency_s"] == 2400.0 and late["ingested_utc"] == "2026-10-10T10:40:00.000Z"
    assert late["marker_uri"] == _marker_uri(REV_LATE, bundle=True) and late["status"] == "ok" and late["accepted"] == 1 and late["n_markers"] == 2
    assert late["op_ids"] == ["tick-1", "tick-1b"] and late["ways"] == ["bundle", "pulled"] and late["receipt_utc"] == "2026-10-10T10:10:00.000Z"
    # a revision the receipt listed and the harvester never folded: undelivered
    und = by[("tick-1", REV_UNDELIVERED, "comp", "pulled")]
    assert und["outcome"] == "censored" and und["censor"] == "undelivered" and und["status"] is None and und["n_markers"] == 0
    # an adapter refused whole: censored refused_operation with the pin on the record; a pin without a PUBLISHED record: no clock
    refs = [s for s in t["samples"] if s["op_id"] == "tick-4" and s["adapter"] == "beta"]
    assert len(refs) == 1
    ref = refs[0]
    assert ref["outcome"] == "censored" and ref["censor"] == "refused_operation" and ref["component"] == "other" and ref["generation"] == "33333333"
    assert ref["shard_refusal"] == ["pull: 2 shard(s) refused"] and ref["shards"] == "0/2" and ref["revision"] is None and ok["shard_refusal"] is None
    nop = next(s for s in t["samples"] if s["op_id"] == "tick-4" and s["adapter"] == "gamma")
    assert nop["outcome"] == "censored" and nop["censor"] == "no_published_record"
    s = t["summary"]
    # nine receipt listings (tick-1 four, tick-1b one, tick-2 / tick-3 / tick-5 one each, gamma's one; a refused entry lists nothing)
    # reconcile to eight revision samples + the refused operation's one
    assert t["receipt_listings"] == 9 and s["N"] == len(t["samples"]) == 9
    assert s["censored_by_reason"] == {"marker_blocked": 1, "undelivered": 1, "refused_operation": 1, "no_published_record": 1}
    assert s["completed"] == 5 and s["tick_min"] == 10.0 and s["overall"]["p50_bound_min"] == 20.0 and s["overall"]["p90_bound_min"] == 30.0


def test_fold_latency_a_held_revision_corrected_through_a_bundle_is_one_sample_completed_at_the_bundle_delivery(tmp_path):
    """A held revision corrected and accepted through a later `_activations/<revision>/<stamp>/` bundle. The
    harvester delivers `ok` to the BUNDLE's marker row while the directory's row stays `blocked` (an upsert by file_uri, one row per
    marker file), and the bundle is listed by a later receipt. One revision, one component pin: ONE sample, completed at the bundle's
    delivery -- never a censored `marker_blocked` sample for the first receipt plus a completed one for the second (two samples for one
    pin, an activated revision reported as still held). Without the correction the same revision is one censored sample."""
    FL = _load("fold_latency")
    root = tmp_path / "reg"
    root.mkdir()
    conn = DB.open_fresh(str(root / "registry.sqlite"))
    pub = {"published_utc": "2026-10-10T10:05:00.000Z", "change_utc": "2026-10-10T10:00:00.000Z", "mode": "push", "producer": "grading_host:x"}
    _revision(conn, REV_LATE, 0, "11111111", "2026-10-10T10:08:00.000Z", refusal="activation pending: run r2 (missing)")
    _deliver(conn, _marker_uri(REV_LATE), "blocked", "2026-10-10T10:12:00.000Z")
    conn.close()
    rd = tmp_path / "state" / "reads" / "receipts"
    rd.mkdir(parents=True)
    (rd / "tick-1.json").write_text(json.dumps(_receipt("tick-1", "2026-10-10T10:10:00.000Z", {ADAPTER: _entry({"comp": "11111111"}, {"comp": pub}, [REV_LATE])})))
    (rd / "tick-2.json").write_text(json.dumps(_receipt("tick-2", "2026-10-10T10:35:00.000Z", {ADAPTER: _entry({"comp": "11111111"}, {"comp": pub}, [f"_activations/{REV_LATE}/{STAMPS[REV_LATE]}"])})))
    since, until = FL.parse_utc(T0), FL.parse_utc("2026-10-11T10:00:00.000Z")
    # held, the correction not yet folded: one censored sample across both receipts (the bundle listed, no delivery of it yet)
    t = FL.measure(str(root), str(tmp_path / "state"), since, until, tick_min=10.0)
    assert t["receipt_listings"] == 2 and len(t["samples"]) == 1
    held = t["samples"][0]
    assert held["outcome"] == "censored" and held["censor"] == "marker_blocked" and held["op_ids"] == ["tick-1", "tick-2"] and held["ways"] == ["bundle", "pulled"]
    assert t["summary"]["completed"] == 0 and t["summary"]["censored_by_reason"] == {"marker_blocked": 1}
    # the correction folds: the bundle's marker row ok at 10:40, accepted = 1 on the shared row, the directory's row untouched
    conn = sqlite3.connect(str(root / "registry.sqlite"))
    _deliver(conn, _marker_uri(REV_LATE, bundle=True), "ok", "2026-10-10T10:40:00.000Z")
    _revision(conn, REV_LATE, 1, "11111111", "2026-10-10T10:08:00.000Z")
    conn.commit()
    assert conn.execute("SELECT status FROM record_delivery WHERE file_uri = ?", (_marker_uri(REV_LATE),)).fetchone()[0] == "blocked"
    conn.close()
    t = FL.measure(str(root), str(tmp_path / "state"), since, until, tick_min=10.0)
    assert len(t["samples"]) == 1, [(s["op_id"], s["outcome"], s["censor"]) for s in t["samples"]]
    one = t["samples"][0]
    assert one["outcome"] == "completed" and one["latency_s"] == 2400.0 and one["ingested_utc"] == "2026-10-10T10:40:00.000Z"
    assert one["marker_uri"] == _marker_uri(REV_LATE, bundle=True) and one["n_markers"] == 2 and one["op_ids"] == ["tick-1", "tick-2"]
    assert t["summary"] == FL.summarize(t["samples"], 10.0) and t["summary"]["completed"] == 1 and t["summary"]["censored"] == 0
    assert t["summary"]["overall"]["p50_min"] == 40.0 and t["summary"]["by_class"]["push"]["n"] == 1
    # the same bundle is an earlier operation's fold of the revision, not a later pin's: a pin changed at 10:50 whose operation (attested
    # 10:53, receipt 10:55) re-activated the revision is bound to bundles stamped in [10:53, 10:55] -- the 10:33 bundle is outside it, the
    # directory is not a re-activation's marker at all -> no bound delivery, undelivered (two markers refused), never the 10:40 fold
    late_pub = dict(pub, change_utc="2026-10-10T10:50:00.000Z", published_utc="2026-10-10T10:52:00.000Z")
    (rd / "tick-3.json").write_text(json.dumps(_receipt("tick-3", "2026-10-10T10:55:00.000Z", {ADAPTER: _entry({"comp": "99999999"}, {"comp": late_pub}, [], reactivated=[REV_LATE])},
                                                        attestation_utc="2026-10-10T10:53:00.000Z")))
    t = FL.measure(str(root), str(tmp_path / "state"), since, until, tick_min=10.0)
    by = {s["generation"]: s for s in t["samples"]}
    assert by["11111111"]["outcome"] == "completed" and by["99999999"]["outcome"] == "censored" and by["99999999"]["censor"] == "undelivered"
    assert by["99999999"]["n_markers"] == 0 and by["99999999"]["n_markers_unbound"] == 2 and by["99999999"]["how"] == "reactivated"
    assert by["99999999"]["bindings"] == [{"op_id": "tick-3", "way": "reactivated", "listed": None, "marker_prefix": f"records/adapters/{ADAPTER}/_activations/{REV_LATE}/",
                                           "stamp_floor": "2026-10-10T10:53:00.000Z", "floor_source": "attestation", "stamp_ceiling": "2026-10-10T10:55:00.000Z"}]
    # the completed pin's bindings are the directory (tick-1) and the named bundle (tick-2): exact paths, no stamp interval
    assert [(b["op_id"], b["way"], b["listed"]) for b in by["11111111"]["bindings"]] == [("tick-1", "pulled", f"{ADAPTER}/{REV_LATE}"), ("tick-2", "bundle", f"{ADAPTER}/_activations/{REV_LATE}/{STAMPS[REV_LATE]}")]


def test_fold_latency_counts_a_folded_then_superseded_revision_and_its_successor_each_at_its_own_delivery(world):
    """Rev 3.1's rule: a revision that folded and was superseded twenty minutes later still COUNTS at its own ingested_utc; the superseding
    revision counts at its own; `superseded_by_revision_id` and `reclaimed` are never consulted. The successor re-activated from committed
    bytes for a NEW pin (`how: reactivated`) is that pin's sample, clocked to the bundle's ok delivery; the directory's earlier ok is before
    the new clock and completes nothing of it; the marker's read_copy (the original read's generation) stands beside the pin as a note."""
    FL = _load("fold_latency")
    since, until = FL.parse_utc(T0), FL.parse_utc("2026-10-11T10:00:00.000Z")
    t = FL.measure(world["root"], world["state"], since, until, tick_min=10.0)
    by = {(s["op_id"], s["revision"], s.get("how")): s for s in t["samples"]}
    first = by[("tick-2", REV_SUP1, "pulled")]
    second = by[("tick-3", REV_SUP2, "pulled")]
    react = by[("tick-5", REV_SUP2, "reactivated")]
    assert first["outcome"] == "completed" and first["latency_s"] == 720.0 and first["class"] == "carry"      # 11:12 - 11:00
    assert second["outcome"] == "completed" and second["latency_s"] == 1800.0 and second["marker_uri"] == _marker_uri(REV_SUP2)      # 11:30 - 11:00
    assert second["generation"] == "22222222" and second["n_markers"] == 1 and second["n_markers_unbound"] == 1 and "pin_mismatch" not in second
    assert react["outcome"] == "completed" and react["latency_s"] == 3000.0 and react["ingested_utc"] == "2026-10-10T12:30:00.000Z"      # 12:30 - 11:40
    assert react["marker_uri"] == _marker_uri(REV_SUP2, bundle=True) and react["generation"] == "33333333" and react["change_utc"] == "2026-10-10T11:40:00.000Z"
    assert react["marker_generation"] == "22222222" and react["pin_mismatch"] == "receipt pin 33333333 != marker generation 22222222"
    assert react["n_markers"] == 1 and react["n_markers_unbound"] == 1 and react["bindings"][0]["floor_source"] == "attestation"
    assert react["bindings"][0]["stamp_floor"] == "2026-10-10T11:43:00.000Z" and react["bindings"][0]["stamp_ceiling"] == "2026-10-10T11:45:00.000Z"
    # two pins of one revision are two samples; one pin of one revision is never two
    assert sorted(s["generation"] for s in t["samples"] if s["revision"] == REV_SUP2) == ["22222222", "33333333"]
    # per class: push (12, 40 min) and carry (12, 30, 50 min); the carry class FAILS S7 at N = 10 (p90 bound 30) and is reported, never excluded
    s = t["summary"]
    assert s["by_class"]["push"]["n"] == 2 and s["by_class"]["push"]["p50_min"] == 26.0 and s["by_class"]["push"]["s7"] == "fail"
    assert s["by_class"]["carry"]["n"] == 3 and s["by_class"]["carry"]["p90_min"] == 46.0 and s["by_class"]["carry"]["s7"] == "fail"
    assert s["by_component"]["comp"]["n"] == 5 and s["overall"]["n"] == 5 and s["overall"]["p50_min"] == 30.0
    # a generous N passes: p50 30 <= 60 + 10, p90 46 <= 130
    assert FL.summarize(t["samples"], 60.0)["overall"]["s7"] == "pass"


def test_fold_latency_a_pin_completes_only_on_a_marker_its_own_operation_listed_never_an_earlier_generation_ok(tmp_path):
    """Generation B changes while revision R from generation A is still folding. A's directory marker is
    delivered ``ok`` AFTER B's change clock but BEFORE B's receipt; B's operation re-activated R from committed bytes. The judge used to take
    A's ``ok`` for B (it followed B's clock) and report B complete before the operation that read B -- understating B's latency when B's
    bundle folds later. Now B is bound to the bundles ITS operation stamped (between its attestation and its receipt): A's ``ok`` is
    refused, B is ``undelivered`` until its own bundle delivers, then completed at that delivery, clocked from B's change. Three more
    faces: (C) an EARLIER operation's bundle delivered after a later pin's change clock is refused by the stamp floor, not merely by the
    clock; (B's own bundle) a bundle the harvester folded BEFORE the receipt was written still completes -- the receipt's ``utc`` is no
    floor on ``ingested_utc`` (the receipt follows its own operation's merge and re-activations); and A's sample never changes, because the
    later bundles are not markers A's operation listed."""
    FL = _load("fold_latency")
    R = "9" * 16
    root = tmp_path / "reg"
    root.mkdir()
    conn = DB.open_fresh(str(root / "registry.sqlite"))
    _revision(conn, R, 1, "22222222", "2026-10-10T11:25:00.000Z")
    _deliver(conn, _marker_uri(R), "ok", "2026-10-10T11:42:00.000Z")          # A's fold of R: after B's change (11:40), before B's receipt (11:45)
    conn.close()
    rd = tmp_path / "state" / "reads" / "receipts"
    rd.mkdir(parents=True)
    pub_a = {"published_utc": "2026-10-10T11:05:00.000Z", "change_utc": "2026-10-10T11:00:00.000Z", "mode": "push", "producer": "grading_host:x"}
    pub_b = {"published_utc": "2026-10-10T11:41:00.000Z", "change_utc": "2026-10-10T11:40:00.000Z", "mode": "push", "producer": "grading_host:x"}
    pub_c = {"published_utc": "2026-10-10T12:41:00.000Z", "change_utc": "2026-10-10T12:40:00.000Z", "mode": "push", "producer": "grading_host:x"}
    # A: tick-3 (attested 11:26, receipt 11:28) pulled R's directory. B: tick-5 (attested 11:43, receipt 11:45) re-activated R. C: tick-7
    # (attested 12:43, receipt 12:45) re-activated R again for a third generation
    (rd / "tick-3.json").write_text(json.dumps(_receipt("tick-3", "2026-10-10T11:28:00.000Z", {ADAPTER: _entry({"comp": "22222222"}, {"comp": pub_a}, [R])}, attestation_utc="2026-10-10T11:26:00.000Z")))
    (rd / "tick-5.json").write_text(json.dumps(_receipt("tick-5", "2026-10-10T11:45:00.000Z", {ADAPTER: _entry({"comp": "33333333"}, {"comp": pub_b}, [], reactivated=[R])}, attestation_utc="2026-10-10T11:43:00.000Z")))
    (rd / "tick-7.json").write_text(json.dumps(_receipt("tick-7", "2026-10-10T12:45:00.000Z", {ADAPTER: _entry({"comp": "44444444"}, {"comp": pub_c}, [], reactivated=[R])}, attestation_utc="2026-10-10T12:43:00.000Z")))
    since, until = FL.parse_utc(T0), FL.parse_utc("2026-10-11T10:00:00.000Z")
    stamp_b, stamp_c = "20261010114400000", "20261010124400000"          # B's bundle written 11:44, C's 12:44 (each inside its operation)

    def judged():
        t = FL.measure(str(root), str(tmp_path / "state"), since, until, tick_min=10.0)
        assert t["receipt_listings"] == 3 and len(t["samples"]) == 3
        return t, {s["generation"]: s for s in t["samples"]}

    # (1) the trigger: only A's directory ok exists. A completes at 11:42 (42 min). B does NOT: the 11:42 ok is after B's clock but it is
    #     the directory's, which B's operation never listed -> undelivered, the marker counted as refused; C likewise
    t, by = judged()
    a, b, c = by["22222222"], by["33333333"], by["44444444"]
    assert a["outcome"] == "completed" and a["latency_s"] == 2520.0 and a["marker_uri"] == _marker_uri(R) and a["n_markers"] == 1 and a["n_markers_unbound"] == 0
    assert b["outcome"] == "censored" and b["censor"] == "undelivered" and b["latency_s"] is None and b["n_markers"] == 0 and b["n_markers_unbound"] == 1
    assert b["bindings"] == [{"op_id": "tick-5", "way": "reactivated", "listed": None, "marker_prefix": f"records/adapters/{ADAPTER}/_activations/{R}/",
                              "stamp_floor": "2026-10-10T11:43:00.000Z", "floor_source": "attestation", "stamp_ceiling": "2026-10-10T11:45:00.000Z"}]
    assert c["outcome"] == "censored" and c["censor"] == "undelivered"
    assert t["summary"]["completed"] == 1 and t["summary"]["censored_by_reason"] == {"undelivered": 2}
    # (2) the mirror, and face C: B's bundle (stamped 11:44) folds LATE, at 12:50 -- after C's change (12:40) and C's receipt (12:45). B
    #     completes at that delivery, 70 min from B's change, on its own bundle. C does not: the delivery is after C's clock, but the bundle
    #     is stamped before C's operation attested (11:44 < 12:43) -> an earlier operation's, refused by the stamp floor
    conn = sqlite3.connect(str(root / "registry.sqlite"))
    _deliver(conn, _marker_uri(R, bundle=True, stamp=stamp_b), "ok", "2026-10-10T12:50:00.000Z")
    conn.commit()
    conn.close()
    t, by = judged()
    a, b, c = by["22222222"], by["33333333"], by["44444444"]
    assert b["outcome"] == "completed" and b["latency_s"] == 4200.0 and b["ingested_utc"] == "2026-10-10T12:50:00.000Z"
    assert b["marker_uri"] == _marker_uri(R, bundle=True, stamp=stamp_b) and b["status"] == "ok" and b["n_markers"] == 1 and b["n_markers_unbound"] == 1
    assert c["outcome"] == "censored" and c["censor"] == "undelivered" and c["n_markers"] == 0 and c["n_markers_unbound"] == 2
    assert a["outcome"] == "completed" and a["latency_s"] == 2520.0 and a["marker_uri"] == _marker_uri(R) and a["n_markers_unbound"] == 1
    # (3) C's own bundle (stamped 12:44) is folded by the harvester at 12:44:30 -- BEFORE C's receipt was written (12:45): the receipt follows
    #     its operation's merge and re-activations, so this is the ordinary case, not an anomaly. C completes at 12:44:30, 4.5 min
    conn = sqlite3.connect(str(root / "registry.sqlite"))
    _deliver(conn, _marker_uri(R, bundle=True, stamp=stamp_c), "ok", "2026-10-10T12:44:30.000Z")
    conn.commit()
    conn.close()
    t, by = judged()
    a, b, c = by["22222222"], by["33333333"], by["44444444"]
    assert c["outcome"] == "completed" and c["latency_s"] == 270.0 and c["marker_uri"] == _marker_uri(R, bundle=True, stamp=stamp_c)
    assert b["outcome"] == "completed" and b["latency_s"] == 4200.0 and b["marker_uri"] == _marker_uri(R, bundle=True, stamp=stamp_b)   # not C's 12:44 bundle: outside B's interval
    assert a["outcome"] == "completed" and a["latency_s"] == 2520.0
    assert t["summary"]["completed"] == 3 and t["summary"]["censored"] == 0 and t["summary"]["by_component"]["comp"]["n"] == 3
    assert sorted(s["n_markers_unbound"] for s in t["samples"]) == [2, 2, 2]
    # a bundle whose directory name is not a stamp is no operation's re-activation: never bound to a reactivated listing
    assert FL.bundle_stamp(f"records/adapters/{ADAPTER}/_activations/{R}/1/000003-source_revision.jsonl") is None
    assert FL.bundle_stamp(_marker_uri(R, bundle=True, stamp=stamp_b)) == FL.parse_utc("2026-10-10T11:44:00.000Z")
    assert FL.bundle_stamp(_marker_uri(R)) is None


def test_fold_latency_judges_admitted_listings_beside_a_sibling_shard_refusal_and_records_the_refusal_as_its_own_row(tmp_path):
    """A two-shard adapter with one shard publishing a revision and the other refusing. `pull_window`
    withholds only the refusing shard's publications (and may replay a trusted refused-known revision), and `receipt_from_plan` records
    the admitted listings and the adapter-level refusal in ONE entry. The judge used to `continue` at the refusal, dropping every listing
    and substituting a refused_operation row per pin -- a marker this very operation produced and the harvester folded ok was absent from
    the completed samples, so S7 could pass by omitting its slow successful folds. Now the listings are judged as usual (noting the
    refusal) and the refusal is its own row (`refused_shards`); an entry refused whole still censors `refused_operation` alone."""
    FL = _load("fold_latency")
    X, Y = "7" * 16, "8" * 16
    root = tmp_path / "reg"
    root.mkdir()
    conn = DB.open_fresh(str(root / "registry.sqlite"))
    pub = {"published_utc": "2026-10-10T10:05:00.000Z", "change_utc": "2026-10-10T10:00:00.000Z", "mode": "push", "producer": "grading_host:x"}
    pub2 = {"published_utc": "2026-10-10T11:41:00.000Z", "change_utc": "2026-10-10T11:40:00.000Z", "mode": "push", "producer": "grading_host:x"}
    _revision(conn, X, 1, "11111111", "2026-10-10T10:08:00.000Z")
    _deliver(conn, _marker_uri(X), "ok", "2026-10-10T10:12:00.000Z")                     # the successful shard's fast fold, 12 min
    _revision(conn, Y, 1, "11111111", "2026-10-10T10:08:00.000Z")
    _deliver(conn, _marker_uri(Y), "ok", "2026-10-10T10:30:00.000Z")                     # ... and its SLOW one, 30 min: the sample that went missing
    stamp = "20261010114400000"
    _deliver(conn, _marker_uri(X, bundle=True, stamp=stamp), "ok", "2026-10-10T11:50:00.000Z")   # a trusted refused-known re-activation, 10 min
    conn.close()
    rd = tmp_path / "state" / "reads" / "receipts"
    rd.mkdir(parents=True)
    why = ["pull: shard 1 of 2 refused: attestation_refused (store listing differs from the manifest)"]
    # tick-1: shard 0 ok and pulled X and Y, shard 1 refused (shards_ok 1/2) -- a mixed-success entry
    (rd / "tick-1.json").write_text(json.dumps(_receipt("tick-1", "2026-10-10T10:10:00.000Z", {ADAPTER: _entry({"comp": "11111111"}, {"comp": pub}, [X, Y], refused=why)})))
    # tick-2: both shards refused, nothing listed -- the adapter refused whole
    (rd / "tick-2.json").write_text(json.dumps(_receipt("tick-2", "2026-10-10T10:20:00.000Z", {ADAPTER: _entry({"comp": "11111111"}, {"comp": pub}, [], refused=["pull: 2 shard(s) refused"], shards_ok=0)})))
    # tick-3: a new pin; shard 1 refused again while the trusted refused-known id X was re-activated from committed bytes (attested 11:43)
    (rd / "tick-3.json").write_text(json.dumps(_receipt("tick-3", "2026-10-10T11:45:00.000Z", {ADAPTER: _entry({"comp": "22222222"}, {"comp": pub2}, [], reactivated=[X], refused=why)},
                                                        attestation_utc="2026-10-10T11:43:00.000Z")))
    since, until = FL.parse_utc(T0), FL.parse_utc("2026-10-11T10:00:00.000Z")
    t = FL.measure(str(root), str(tmp_path / "state"), since, until, tick_min=10.0)
    assert t["receipts"] == 3 and t["receipt_listings"] == 3
    by_op = {}
    for smp in t["samples"]:
        by_op.setdefault(smp["op_id"], []).append(smp)
    # tick-1: X and Y judged as usual (completed at their own deliveries, each noting the sibling refusal), plus ONE refused_shards row for
    # the pin -- never a refused_operation row in place of the two
    order = lambda smp: (smp.get("revision") is not None, str(smp.get("revision")))     # the refusal row (no revision) first, then by revision  # noqa: E731
    one = sorted(by_op["tick-1"], key=order)
    assert [(smp["revision"], smp["outcome"], smp["censor"], smp["latency_s"]) for smp in one] == [
        (None, "censored", "refused_shards", None), (X, "completed", None, 720.0), (Y, "completed", None, 1800.0)]
    assert all(smp["shard_refusal"] == why for smp in one) and one[0]["shards"] == "1/2" and one[0]["component"] == "comp" and one[0]["generation"] == "11111111"
    assert one[1]["marker_uri"] == _marker_uri(X) and one[2]["marker_uri"] == _marker_uri(Y) and one[1]["bindings"][0]["way"] == "pulled"
    # tick-2: refused whole -> refused_operation alone
    assert [(smp["revision"], smp["censor"], smp["shards"]) for smp in by_op["tick-2"]] == [(None, "refused_operation", "0/2")]
    # tick-3: the re-activation beside the refusal completes on its own bundle (10 min from the new pin's change), plus its refused_shards row
    three = sorted(by_op["tick-3"], key=order)
    assert [(smp["revision"], smp["outcome"], smp["censor"], smp["latency_s"], smp["generation"]) for smp in three] == [
        (None, "censored", "refused_shards", None, "22222222"), (X, "completed", None, 600.0, "22222222")]
    assert three[1]["marker_uri"] == _marker_uri(X, bundle=True, stamp=stamp) and three[1]["how"] == "reactivated" and three[1]["shard_refusal"] == why
    s = t["summary"]
    assert s["N"] == 6 and s["completed"] == 3 and s["censored"] == 3
    assert s["censored_by_reason"] == {"refused_shards": 2, "refused_operation": 1}
    # the slow successful fold is IN the percentiles: 10, 12, 30 min -> p50 12, p90 26.4; without it (the old judge) nothing of tick-1 counted
    assert s["overall"]["n"] == 3 and s["overall"]["p50_min"] == 12.0 and s["overall"]["p90_min"] == 26.4 and s["by_class"]["push"]["n"] == 3
    # the JSON lines carry the same rows
    out = tmp_path / "latency.jsonl"
    r = subprocess.run([sys.executable, os.path.join(MEASURE, "fold_latency.py"), "--since", T0, "--root", str(root), "--state", str(tmp_path / "state"), "--out", str(out), "--json"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert sorted(str(x.get("censor")) for x in rows) == ["None", "None", "None", "refused_operation", "refused_shards", "refused_shards"]


def test_fold_latency_censors_a_delivery_after_the_observation_end_and_a_component_with_no_receipt(world, tmp_path):
    FL = _load("fold_latency")
    since, until = FL.parse_utc(T0), FL.parse_utc("2026-10-10T11:20:00.000Z")       # the end falls between REV_SUP1 (11:12) and REV_SUP2 (11:30)
    auth = tmp_path / "store_authority.json"
    auth.write_text(json.dumps({"schema": "store_authority_state/1", "checked_utc": "2026-10-10T11:19:00.000Z",
                                "components": {"comp": {"current": "22222222", "change_utc": "2026-10-10T11:00:00.000Z", "publish_mode": "carry"},
                                               "never_read": {"current": "99999999", "change_utc": "2026-10-10T10:30:00.000Z", "published_utc": "2026-10-10T10:31:00.000Z", "publish_mode": "push"},
                                               "future": {"current": "88888888", "change_utc": "2026-10-10T12:00:00.000Z", "publish_mode": "push"},
                                               "unpublished": {"current": None}}}))
    t = FL.measure(world["root"], world["state"], since, until, tick_min=10.0, authority=str(auth))
    # receipts after the end (tick-3 at 11:28, tick-5 at 11:45, tick-4 at 12:00) are outside the observation
    assert t["receipts"] == 3
    by = {(s["op_id"], s["revision"]): s for s in t["samples"] if s.get("revision")}
    assert by[("tick-2", REV_SUP1)]["outcome"] == "completed"
    # the held-then-activated revision's activation (10:40) is inside. Shrink the end to 10:30: the receipt that listed the correcting bundle
    # (tick-1b, 10:35) is outside the span, so the pin is bound to the directory alone and the end sees a HELD revision -- marker_blocked,
    # never a verdict from a listing or a delivery after the end
    t2 = FL.measure(world["root"], world["state"], since, FL.parse_utc("2026-10-10T10:30:00.000Z"), tick_min=10.0)
    late = next(s for s in t2["samples"] if s["revision"] == REV_LATE)
    assert t2["receipts"] == 1 and late["outcome"] == "censored" and late["censor"] == "marker_blocked" and late["n_markers"] == 1 and late["n_markers_unbound"] == 1
    # a BOUND marker delivered after the end is after_end: tick-3 (11:28) pulled REV_SUP2's directory, delivered ok at 11:30; end at 11:29
    t3 = FL.measure(world["root"], world["state"], since, FL.parse_utc("2026-10-10T11:29:00.000Z"), tick_min=10.0)
    sup2 = next(s for s in t3["samples"] if s["revision"] == REV_SUP2)
    assert sup2["op_id"] == "tick-3" and sup2["outcome"] == "censored" and sup2["censor"] == "after_end" and sup2["ingested_utc"] == "2026-10-10T11:30:00.000Z"
    # authority: `comp` is pinned by a receipt (no sample); `never_read` has no receipt -> no_receipt; `future` changed after the end -> nothing;
    # `unpublished` has no current -> nothing
    nr = [s for s in t["samples"] if s.get("censor") == "no_receipt"]
    assert [s["component"] for s in nr] == ["never_read"] and nr[0]["generation"] == "99999999" and nr[0]["class"] == "push"
    assert t["summary"]["censored_by_reason"]["no_receipt"] == 1


def test_fold_latency_cli_json_out_and_the_database_opened_read_only(world, tmp_path):
    out = tmp_path / "latency.jsonl"
    r = subprocess.run([sys.executable, os.path.join(MEASURE, "fold_latency.py"), "--since", T0, "--hours", "24", "--root", world["root"],
                        "--state", world["state"], "--tick-min", "10", "--out", str(out), "--json"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["summary"]["completed"] == 5 and d["summary"]["N"] == len(d["samples"]) and d["db"] == world["db"]
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(lines) == d["summary"]["N"] and all(l["since"] == d["since"] for l in lines)
    r = subprocess.run([sys.executable, os.path.join(MEASURE, "fold_latency.py"), "--since", T0, "--root", world["root"], "--state", world["state"]],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "5 completed" in r.stdout and "| carry |" in r.stdout and "| push |" in r.stdout and "S7 fail" in r.stdout
    # read-only: a database that does not exist is a named error, every revision sample censored database_unreadable, not a created file
    FL = _load("fold_latency")
    missing = str(tmp_path / "none" / "registry.sqlite")
    t = FL.measure(world["root"], world["state"], FL.parse_utc(T0), FL.parse_utc("2026-10-11T10:00:00.000Z"), 10.0, db_path=missing)
    assert t["db_error"] and not os.path.exists(missing)
    # every revision sample the ledger would have judged is database_unreadable; the one with no clock (no PUBLISHED record) stays so
    assert {s["censor"] for s in t["samples"] if s.get("revision")} == {"database_unreadable", "no_published_record"}
    assert t["summary"]["completed"] == 0 and t["summary"]["censored_by_reason"]["database_unreadable"] == 7
    # the bounded range: a prefix range on the primary key, never a LIKE
    assert FL._prefix_range("records/adapters/a/b/") == ("records/adapters/a/b/", "records/adapters/a/b/\U0010ffff")
    assert "LIKE" not in open(os.path.join(MEASURE, "fold_latency.py")).read().split("def marker_deliveries")[1].split("def revision_row")[0]
    assert "row_history" not in open(os.path.join(MEASURE, "fold_latency.py")).read().replace("never ``row_history``", "").replace("never `row_history`", "")


# --------------------------------------------------------------------------------------------
# the changed set per operation
# --------------------------------------------------------------------------------------------


def _ops_world(tmp_path):
    state = tmp_path / "state"
    ops = state / "reads" / "ops"
    rec = state / "reads" / "receipts"
    ops.mkdir(parents=True)
    rec.mkdir(parents=True)
    tick = {"schema": "reads_op/1", "op_id": "tick-1", "owner": "tick", "started_utc": "2026-10-10T10:00:00.000Z", "ended_utc": "2026-10-10T10:06:00.000Z",
            "state": "receipted", "changed": {"alpha": ["pin moved: comp 11111111 -> 22222222"], "beta": ["never read whole (no receipt)", "module changed: src/x.py a -> b"]},
            "skipped": {"gamma": "declared cluster: refused (no pin)"}, "unchanged": ["delta"],
            "history": [{"state": "reserved", "utc": "2026-10-10T10:00:00.000Z"}, {"state": "submitted", "utc": "2026-10-10T10:01:00.000Z"},
                        {"state": "receipted", "utc": "2026-10-10T10:06:00.000Z"}]}
    quiet = {"schema": "reads_op/1", "op_id": "tick-2", "owner": "tick", "started_utc": "2026-10-10T10:10:00.000Z", "ended_utc": "2026-10-10T10:10:05.000Z",
             "state": "unchanged", "changed": {}, "skipped": {}, "unchanged": ["alpha", "beta", "delta"], "history": [{"state": "unchanged", "utc": "2026-10-10T10:10:05.000Z"}]}
    window = {"schema": "reads_op/1", "op_id": "W1", "owner": "window", "started_utc": "2026-10-10T11:00:00.000Z", "ended_utc": "2026-10-10T11:08:00.000Z",
              "state": "receipted", "changed": {}, "skipped": {}, "history": [{"state": "reserved", "utc": "2026-10-10T11:00:00.000Z"}]}
    for op in (tick, quiet, window):
        (ops / f"{op['op_id']}.json").write_text(json.dumps(op))
    (ops / "tick-1.lock").write_text("")
    pub = {"published_utc": "2026-10-10T09:50:00.000Z", "change_utc": "2026-10-10T09:45:00.000Z", "mode": "push", "producer": "grading_host:x"}
    (rec / "tick-1.json").write_text(json.dumps(_receipt("tick-1", "2026-10-10T10:06:00.000Z", {
        "alpha": _entry({"comp": "22222222"}, {"comp": pub}, ["r1", "r2"]), "beta": _entry({"other": "33333333"}, {"other": pub}, [], refused=["pull: refused"])})))
    (rec / "W1.json").write_text(json.dumps(_receipt("W1", "2026-10-10T11:08:00.000Z", {"alpha": _entry({"comp": "22222222"}, {"comp": pub}, ["r3"])})))
    root = tmp_path / "reg"
    man = root / "state" / "store_manifests" / "comp"
    man.mkdir(parents=True)
    (man / "MANIFEST-22222222.json").write_text(json.dumps({"schema": "store_manifest/1", "slug": "comp", "content_sha8": "22222222", "bytes": 1500, "n_files": 3}))
    return str(state), str(root)


def test_changed_set_joins_op_records_and_receipts_with_the_reasons_by_kind(tmp_path):
    CS = _load("changed_set")
    state, root = _ops_world(tmp_path)
    t = CS.collect(state, root)
    rows = {r["op_id"]: r for r in t["rows"]}
    assert [r["op_id"] for r in t["rows"]] == ["tick-1", "tick-2", "W1"]
    t1 = rows["tick-1"]
    assert t1["changed_set"] == ["alpha", "beta"] and t1["changed_set_n"] == 2 and t1["reason_kinds"] == {"pin_moved": 1, "never_read": 1, "module_changed": 1}
    assert t1["skipped_n"] == 1 and t1["unchanged_n"] == 1 and t1["has_receipt"] and t1["read_whole_n"] == 1 and t1["refused_n"] == 1 and t1["revisions_pulled"] == 2
    assert t1["change_utc"] == "2026-10-10T09:45:00.000Z" and t1["input_bytes"] is None and t1["pins_without_manifest"] == ["other/33333333"]
    assert [h["state"] for h in t1["history"]] == ["reserved", "submitted", "receipted"]
    assert rows["tick-2"]["changed_set_n"] == 0 and rows["tick-2"]["has_receipt"] is False and rows["tick-2"]["read_whole_n"] is None
    w = rows["W1"]
    # a window's record carries no changed map: the set is what its receipt read; the pin's bytes resolve from the grading host manifest
    assert w["changed_set"] == ["alpha"] and w["reason_kinds"] == {} and w["input_bytes"] == 1500 and w["input_files"] == 3 and w["revisions_pulled"] == 1
    s = t["summary"]
    assert s["operations"] == 3 and s["by_owner"] == {"tick": 2, "window": 1} and s["reason_kinds"] == {"pin_moved": 1, "never_read": 1, "module_changed": 1}
    assert s["changed_set_n"]["zero"] == 1 and s["changed_set_n"]["max"] == 2 and s["with_receipt"] == 2 and s["with_refusal"] == 1
    assert tuple(s["most_changed"][0]) == ("alpha", 2) and tuple(s["most_changed"][1]) == ("beta", 1)
    assert CS.collect(state, root, owner="window")["summary"]["operations"] == 1 and CS.collect(state, root, since="2026-10-10T10:05:00Z")["summary"]["operations"] == 2
    r = subprocess.run([sys.executable, os.path.join(MEASURE, "changed_set.py"), "--state", state, "--root", root, "--json"], capture_output=True, text=True)
    assert r.returncode == 0 and json.loads(r.stdout)["summary"]["operations"] == 3
    r = subprocess.run([sys.executable, os.path.join(MEASURE, "changed_set.py"), "--state", state, "--root", root], capture_output=True, text=True)
    assert "| tick-1 | tick |" in r.stdout and "never_read 1" in r.stdout and "| W1 | window |" in r.stdout
    # an absent namespace is an empty table, not an error
    assert CS.collect(str(tmp_path / "nowhere"))["summary"]["operations"] == 0


# --------------------------------------------------------------------------------------------
# the benchmark recorder
# --------------------------------------------------------------------------------------------

BENCH_LOG = """=== ff chain start 2026-10-10T10:00:00Z ADAPTERS=alpha beta
-- reads reservation 10:00:00 (op W1, holder pid 7)
writers quiet 10:00:20
base at 1111111 10:00:21
-- install units 10:00:22
-- migrate (apply) 10:00:23
-- dimensions load SKIPPED (x) 10:00:24
-- reader-state export 10:00:25
-- tree export 10:00:40
-- cluster reads plan 10:00:45
-- cluster reads stage 10:00:50
-- cluster reads submit 10:00:55
-- cluster reads revalidate 10:02:25
-- cluster reads pull 10:02:30
-- cluster reads receipt 10:02:40
--- ingest 10:02:45
 "files": 12,
 "records": 340,
--- ingest end 10:04:15
timers restored (trap; the set above) 10:04:20
=== ff chain end 2026-10-10T10:04:20Z
=== ff chain start 2026-10-10T12:00:00Z ADAPTERS=
writers quiet 12:00:10
base at 2222222 12:00:11
--- ingest 12:00:12
--- ingest end 12:01:12
timers NOT restored: x (trap) 12:01:15
=== ff chain ABORTED (restore) 2026-10-10T12:01:15Z
"""


def _attestation(utc, job_id, elapsed_s, gens, window=None):
    """An attestation record as ``cluster_reads.attestation_summary`` puts it on the plan's pull stage and the receipt: per generation the
    listing's wall (``listing_s``), the generation's whole wall (``elapsed_s`` = listing + seal + sample) and its entries."""
    return {"present": True, "window": window, "this_window": window is not None, "ok": True, "utc": utc, "host": "gpu-3", "job_id": job_id,
            "elapsed_s": elapsed_s, "refused": [],
            "generations": {slug: {"content_sha8": "11111111", "ok": True, "refusal": None, "reason": None, "content_mode": "sample",
                                   "listing_entries": entries, "listing_s": listing_s, "elapsed_s": gen_s} for slug, (entries, listing_s, gen_s) in gens.items()}}


def _bench_world(tmp_path):
    log = tmp_path / "chain.log"
    log.write_text(BENCH_LOG)
    root = tmp_path / "reg"
    sums = root / "state" / "cluster_reads" / "W1.summaries"
    for adapter, timings in (("alpha", [(7.3, 29.5, 3.0), (8.0, 95.0, 2.5)]), ("beta", [(1.0, 10.0, 1.0)])):
        d = sums / adapter
        d.mkdir(parents=True)
        for i, (pre, pas, post) in enumerate(timings):
            (d / f"{i}-of-{len(timings)}.json").write_text(json.dumps({"schema": "shard_summary/1", "ok": True, "wall_s": pre + pas + post,
                                                                      "timings_s": {"admission_pre": pre, "pass": pas, "admission_post": post}}))
        (d / "9-of-9.json").write_text(json.dumps({"schema": "shard_summary/1", "ok": False, "wall_s": 0.1, "timings_s": {"admission_pre": 0.1}, "refusal": "x"}))
    # the attestation evidence the pull stage writes (C5 / C7, cluster_reads.attestation_summary's shape): every operation's plan under
    # state/cluster_reads/<op>.json carries stages.pull.attestation -- a window's (stamped 10:01) and a tick's (tick-9, 13:00) -- and
    # the window's receipt carries the same record (one entry, not two); an older operation's (09-29) lies before a --since 2026-10-10 span
    w1_att = _attestation("2026-10-10T10:01:00.000Z", "7", 41.0, {"comp": (86978, 23.5, 38.0), "other": (900, 9.0, 12.0)}, window="W1")
    (root / "state" / "cluster_reads" / "W1.json").write_text(json.dumps({"window": "W1", "stages": {"pull": {"utc": "2026-10-10T10:02:30.000Z", "attestation": w1_att}}}))
    (root / "state" / "cluster_reads" / "tick-9.json").write_text(json.dumps({"stages": {"pull": {
        "utc": "2026-10-10T13:00:40.000Z", "attestation": _attestation("2026-10-10T13:00:00.000Z", "8", 70.0, {"comp": (86978, 66.0, 69.0)})}}}))
    (root / "state" / "cluster_reads" / "20260929T120000Z.json").write_text(json.dumps({"stages": {"pull": {
        "utc": "2026-09-29T12:05:00.000Z", "attestation": _attestation("2026-09-29T12:01:00.000Z", "5", 33.0, {"comp": (80000, 30.0, 31.0)})}}}))
    # a plan from before the attestation existed (the pre-cutover shape: no attestation in its pull stage) contributes nothing
    (root / "state" / "cluster_reads" / "20260928T120000Z.json").write_text(json.dumps({"stages": {"pull": {"utc": "2026-09-28T12:05:00.000Z", "pulled": 3}}}))
    rcpt = tmp_path / "state" / "reads" / "receipts"
    rcpt.mkdir(parents=True, exist_ok=True)          # the figures test builds this world beside the latency world on one tmp_path
    (rcpt / "W1.json").write_text(json.dumps({"schema": "reads_receipt/1", "op": {"op_id": "W1", "owner": "window"}, "utc": "2026-10-10T10:02:40.000Z",
                                              "attestation": w1_att, "adapters": {}}))
    spec = tmp_path / "spec"
    (spec / "rehearsals" / "R14").mkdir(parents=True)
    (spec / "rehearsals" / "R14" / "r14-result.json").write_text(json.dumps({"schema": "r14_drill/1", "utc": "u", "job_id": "1", "listings": [
        {"path": "/g/a", "entries": 86978, "listing_s": 22.76, "dir_mode_check_s": 8.553}, {"path": "/g/b", "entries": 97951, "listing_s": 52.7, "dir_mode_check_s": 16.2}]}))
    return str(log), str(root), str(spec)


def test_benchmarks_measures_every_lever_from_evidence_names_sources_and_merges_on_write(tmp_path):
    B = _load("benchmarks")
    log, root, spec = _bench_world(tmp_path)
    rec = B.record(spec, root, str(tmp_path / "state"), None, None, os.path.join(spec, "rehearsals", "R14", "r14-result.json"), logs=[log])
    lv = rec["levers"]
    assert rec["windows_in_span"] == 2 and rec["completed_in_span"] == 1
    assert lv["reader_state_export"]["measured_s"]["max"] == 15.0 and lv["reader_state_export"]["verdict"] == "within"
    assert lv["array_span"]["measured_s"]["max"] == 90.0 and lv["array_span"]["verdict"] == "within"
    # the ok shards' pass per adapter: alpha max 95 (over 90), beta 10; the refused shard counted, never timed as a read
    assert lv["shard_read"]["measured_s"]["per_adapter_max"] == {"alpha": 95.0, "beta": 10.0} and lv["shard_read"]["verdict"] == "over"
    assert lv["admission"]["measured_s"]["per_adapter_max"] == {"alpha": 10.5, "beta": 2.0} and lv["admission"]["verdict"] == "reported"
    # the attestation listing: measured from the pull stage's attestation records -- a plan and its receipt
    # are ONE entry, tick-9's plan another, the 09-29 operation's a third (no --since); the pre-attestation plan contributes nothing -- per
    # generation listing_s 23.5 / 9.0, 66.0, 30.0 -> max 66.0 over the 60-s rule; R14's 68.9 is the LABELLED BASELINE, never a sample
    al = lv["attestation_listing"]
    assert [e["source"] for e in al["detail"]["attestations"]] == ["plan:20260929T120000Z", "plan:W1", "plan:tick-9"]
    assert al["measured_s"]["n"] == 4 and al["measured_s"]["max"] == 66.0 and al["measured_s"]["min"] == 9.0 and al["verdict"] == "over"
    assert al["detail"]["scanned"]["plans"] == 4 and al["detail"]["scanned"]["receipts"] == 1 and al["detail"]["excluded_by_span"] == 0
    assert al["detail"]["per_generation_s"]["n"] == 4 and al["detail"]["per_generation_s"]["max"] == 69.0
    base = al["detail"]["baseline_r14"]
    # the fixture's R14 record has TWO generations (22.76 + 8.553 = 31.313 s, 52.7 + 16.2 = 68.9 s)
    assert base["label"].startswith("baseline") and base["measured_s"]["n"] == 2 and base["measured_s"]["max"] == 68.9 and base["verdict"] == "over"
    assert [l["total_s"] for l in base["listings"]] == [31.313, 68.9] and "baseline" in al["source"] and "68.9" not in json.dumps(al["measured_s"])
    # the whole attestation's wall (every pinned generation, parallel threads) is reported beside it
    assert lv["attestation"]["measured_s"]["n"] == 3 and lv["attestation"]["measured_s"]["max"] == 70.0 and lv["attestation"]["verdict"] == "reported"
    # the documented post-cutover read: --since <cutover> excludes the older operation; --until bounds it; the verdict follows the span's
    # OWN attestations (W1 alone: max 23.5 -> within) and is never forced `over` by the pre-cutover baseline
    r_since = B.record(spec, root, str(tmp_path / "state"), "2026-10-10", None, os.path.join(spec, "rehearsals", "R14", "r14-result.json"), logs=[log])
    a2 = r_since["levers"]["attestation_listing"]
    assert [e["source"] for e in a2["detail"]["attestations"]] == ["plan:W1", "plan:tick-9"] and a2["measured_s"]["n"] == 3 and a2["detail"]["excluded_by_span"] == 1
    r_w1 = B.record(spec, root, str(tmp_path / "state"), "2026-10-10", "2026-10-10T12:00:00Z", os.path.join(spec, "rehearsals", "R14", "r14-result.json"), logs=[log])
    a3 = r_w1["levers"]["attestation_listing"]
    assert [e["source"] for e in a3["detail"]["attestations"]] == ["plan:W1"] and a3["measured_s"] == {"n": 2, "median": 16.2, "mean": 16.2, "max": 23.5, "min": 9.0}
    assert a3["verdict"] == "within" and a3["detail"]["baseline_r14"]["verdict"] == "over"
    assert r_w1["levers"]["attestation"]["measured_s"]["n"] == 1 and r_w1["levers"]["attestation"]["measured_s"]["max"] == 41.0
    # no attestation yet (the first post-cutover run before any tick) with R14 on disk: unmeasured, the baseline shown beside, never `over`
    r_none = B.record(spec, str(tmp_path / "noroot"), str(tmp_path / "nostate"), "2026-10-10", None, os.path.join(spec, "rehearsals", "R14", "r14-result.json"), logs=[log])
    a4 = r_none["levers"]["attestation_listing"]
    assert a4["verdict"] == "unmeasured" and a4["measured_s"]["n"] == 0 and a4["detail"]["baseline_r14"]["measured_s"]["max"] == 68.9
    assert r_none["levers"]["attestation"]["verdict"] == "unmeasured"
    assert lv["fold"]["measured_s"]["n"] == 2 and lv["fold"]["measured_s"]["max"] == 90.0 and lv["drain"]["measured_s"]["median"] == 15.0 and lv["drain"]["verdict"] == "within"
    # S1 needs all three: the one window's 4.3 min keeps the mean (<= 5) and the max (<= 12) but not the median (<= 4) -> over
    assert lv["window_total"]["measured_min"] == {"n": 1, "median_min": 4.3, "mean_min": 4.3, "max_min": 4.3} and lv["window_total"]["verdict"] == "over"
    assert lv["window_total"]["end_definitions"] == ["rev3"]
    assert all("source" in v and v["source"] for v in lv.values())
    # no evidence -> unmeasured, never a number (a state root with no receipts: the recorder reads the receipts' attestation records now)
    empty = B.record(spec, str(tmp_path / "noroot"), str(tmp_path / "nostate"), None, None, str(tmp_path / "no-r14.json"), logs=[str(tmp_path / "no.log")])
    assert all(v["verdict"] == "unmeasured" for v in empty["levers"].values())
    # --write merges: a later record that measures nothing keeps the earlier levers (recorded under the same method)
    path = os.path.join(spec, "measure", "benchmarks.json")
    B.merge_write(path, rec)
    merged = B.merge_write(path, empty)
    assert merged["levers"]["shard_read"]["verdict"] == "over" and len(merged["history"]) == 2 and json.load(open(path))["levers"]["drain"]["verdict"] == "within"
    assert all(v["method"] == B.RECORD_METHOD for v in merged["levers"].values())
    # a lever the FIRST record computed by the rule since corrected (the attestation lever forced `over` by R14, no method stamp) is
    # replaced by this record's unmeasured-with-baseline lever, never kept; a same-method lever is kept as before
    legacy = os.path.join(spec, "measure", "legacy.json")
    with open(legacy, "w") as fh:
        json.dump({"levers": {"attestation_listing": {"verdict": "over", "measured_s": {"n": 3, "max": 68.9}, "source": "R14 drill record ..."},
                              "shard_read": {"verdict": "over", "method": B.RECORD_METHOD, "measured_s": {"overall": {"n": 1}}}}, "history": []}, fh)
    relegacy = B.merge_write(legacy, r_none)
    assert relegacy["levers"]["attestation_listing"]["verdict"] == "unmeasured" and relegacy["levers"]["attestation_listing"]["detail"]["baseline_r14"]["measured_s"]["max"] == 68.9
    assert relegacy["levers"]["shard_read"]["verdict"] == "over"
    r = subprocess.run([sys.executable, os.path.join(MEASURE, "benchmarks.py"), "--spec", spec, "--root", root, "--logs", log, "--json"], capture_output=True, text=True)
    assert r.returncode == 0 and json.loads(r.stdout)["levers"]["array_span"]["verdict"] == "within"
    r = subprocess.run([sys.executable, os.path.join(MEASURE, "benchmarks.py"), "--spec", spec, "--root", root, "--logs", log], capture_output=True, text=True)
    assert "| shard_read |" in r.stdout and "| over |" in r.stdout


# --------------------------------------------------------------------------------------------
# the figures
# --------------------------------------------------------------------------------------------


def test_figures_render_every_figure_from_the_fixtures_and_skip_an_absent_input_by_name(tmp_path, world):
    pytest.importorskip("matplotlib")
    W = _load("windows")
    FL = _load("fold_latency")
    F = _load("figures")
    log, root, _spec = _bench_world(tmp_path)
    # windows JSON as windows.py prints it (the same shape the CLI emits)
    windows = W.read_logs([log])
    after = {"summary": W.summarize(windows), "windows": W.window_rows(windows), "stages": [w["stages"] for w in windows]}
    before = json.loads(json.dumps(after))
    for w in before["windows"]:
        w["end_definition"] = "pre-rev3"
    (tmp_path / "after.json").write_text(json.dumps(after, default=str))
    (tmp_path / "before.json").write_text(json.dumps(before, default=str))
    lat = FL.measure(world["root"], world["state"], FL.parse_utc(T0), FL.parse_utc("2026-10-11T10:00:00.000Z"), 10.0)
    (tmp_path / "latency.json").write_text(json.dumps(lat, default=str))
    state, _ = _ops_world(tmp_path / "ops")
    out, copy = str(tmp_path / "figs"), str(tmp_path / "copy")
    res = F.render_all(out, copy, str(tmp_path / "before.json"), str(tmp_path / "after.json"), str(tmp_path / "latency.json"), None, state, root)
    names = sorted(os.path.basename(p) for p in res["written"])
    assert names == ["admission_vs_read_per_shard.png", "changed_set_vs_duration.png", "fold_latency_ecdf.png", "tick_timeline.png",
                     "window_durations_before_after.png", "window_stage_breakdown.png"] and res["skipped"] == {}
    for p in res["written"]:
        with open(p, "rb") as fh:
            head = fh.read(8)
        assert head == b"\x89PNG\r\n\x1a\n" and os.path.getsize(p) > 5000, p
        assert os.path.isfile(os.path.join(copy, os.path.basename(p)))
    # every input absent: every figure skipped by name, nothing written, no error
    res2 = F.render_all(str(tmp_path / "figs2"), None, None, None, None, None, str(tmp_path / "nostate"), str(tmp_path / "noroot"))
    assert res2["written"] == [] and sorted(res2["skipped"]) == names
    r = subprocess.run([sys.executable, os.path.join(MEASURE, "figures.py"), "--out", str(tmp_path / "figs3"), "--copy", "", "--windows-after", str(tmp_path / "after.json"),
                        "--state", str(tmp_path / "nostate"), "--root", str(tmp_path / "noroot"), "--json"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert len(d["written"]) == 3 and "fold_latency_ecdf.png" in d["skipped"] and os.path.isfile(os.path.join(str(tmp_path / "figs3"), "figures.json"))
    # the palette's slot rule: the stage classes are folded to the eight slots, never cycled
    assert len(F.STAGE_ORDER) == len(set(F.STAGE_SLOT.values())) == 8 and set(F.STAGE_FOLD.values()) <= set(F.STAGE_ORDER)
