"""em_misalign parks a row whose fitted object the registry does not carry.

A panel row naming a fitted object (`_object_id`, a 64-hex sha) that the connected registry's `object` table does not carry -- and that
no record of this batch mints -- would fold as a `run` with a missing parent: the harvester HOLDS it as a blocked record, and with it the
whole revision (`activation pending: object <sha> (<uri>#<n>)`; em_misalign's 11 held revisions on the superseded panel
em/agentic/llama8b_bl__r0p1-superseded, 2026-10-06). The adapter now checks the object table through the pass's connection before
emitting such a row and PARKS it by name in its per-row refusal ledger (`status_extra["row_refusals"]`), so the panel's other rows publish;
the harvester's hold semantics are untouched (a row that reaches the fold without its parent is still held). Without a connection, or on a
connection without an object table (an in-memory frame), there is nothing to check against and the row emits as before.
"""
from __future__ import annotations

import sqlite3

import pytest

import _anchor_frame_fixtures as AF  # noqa: E402
import registry_declared_fixtures as F  # noqa: E402
from test_registry_em_catchup import ALLPOS_CELL, ALLPOS_STEM, NOW, PRODUCER, _adapter, _allpos_world  # noqa: E402

from sisteer.registry.adapters import em_misalign as EM
from sisteer.registry.adapters.base import AdapterContext


def _ctx(ad, group, objects=None, object_table=True):
    """The pass's context: the frame primed from the adapter's own emissions (the anchor currency every em read consults), with an `object`
    table carrying ``objects`` beside it -- the table the adapter probes; ``object_table`` False leaves the frame without one (nothing to check
    against); ``objects`` None with the table present means an EMPTY registry (no fitted object registered)."""
    ctx = AF.primed_ctx(ad, now=NOW, producer=PRODUCER, groups={group: []})
    if object_table:
        ctx.conn.execute("CREATE TABLE IF NOT EXISTS object (object_id TEXT PRIMARY KEY)")
        ctx.conn.executemany("INSERT OR IGNORE INTO object (object_id) VALUES (?)", [(o,) for o in (objects or ())])
    return ctx


def _read(ad, group, ctx):
    recs = []
    for batch in ad.read(ctx, group, []):
        recs.extend(batch.records)
    out = {}
    for r in recs:
        out.setdefault(r["record_type"], []).append(r["payload"])
    return out, ad.last_report, ad.status_extra(group)


def _registry_conn(*object_ids):
    """A bare connection in the shape the adapter probes (the pure-rule tests): an `object` table carrying ``object_ids``."""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE object (object_id TEXT PRIMARY KEY)")
    con.executemany("INSERT INTO object (object_id) VALUES (?)", [(o,) for o in object_ids])
    return con


def test_a_row_whose_object_the_registry_does_not_carry_is_parked_by_name_and_the_panels_other_rows_publish(tmp_path):
    root, grade = _allpos_world(tmp_path)
    ad = _adapter(root, grade)
    # the object is registered: every row emits (the all-positions arm and the none, two behaviours each)
    out, report, extra = _read(ad, ALLPOS_STEM, _ctx(ad, ALLPOS_STEM, [F.SHA_A]))
    objects = sorted({r["object_id"] for r in out["run"]})
    assert objects == sorted({"none", F.SHA_A}) and len(out["run"]) == 4 and not [r for r in report.refusals if "parked" in r]
    assert extra["row_refusals"] == []
    # the object is NOT registered (another adapter owns the object row and has not emitted it): the two arm rows park, the two none rows publish
    ad2 = _adapter(root, grade)
    out2, report2, extra2 = _read(ad2, ALLPOS_STEM, _ctx(ad2, ALLPOS_STEM, []))
    assert {r["object_id"] for r in out2["run"]} == {"none"} and len(out2["run"]) == 2
    parked = [r for r in report2.refusals if "parked" in r]
    assert len(parked) == 2 and all(ALLPOS_CELL in r and F.SHA_A[:16] in r and "no object row" in r and "held at ingest" in r for r in parked), parked
    assert extra2["row_refusals"] == list(report2.refusals) and all("parked" in r for r in extra2["row_refusals"])
    # no run_contrast names the parked rows, and nothing of theirs reached the batch
    assert not [c for c in out2.get("run_contrast", []) if c["run_id"] not in {r["run_id"] for r in out2["run"]}]
    assert all(r["object_id"] != F.SHA_A for r in out2.get("run_state", []) if "object_id" in r)


def test_without_an_object_table_or_a_connection_the_row_emits_as_before(tmp_path):
    root, grade = _allpos_world(tmp_path)
    # the primed frame the other em tests read with, no object table: nothing to check against
    ad = _adapter(root, grade)
    out, report, _ = _read(ad, ALLPOS_STEM, _ctx(ad, ALLPOS_STEM, object_table=False))
    assert len(out["run"]) == 4 and not [r for r in report.refusals if "parked" in r]
    # (a context with no connection at all is the pure rule's case below: None, nothing to check against)


def test_the_rule_is_pure_and_the_emission_version_moved():
    row_fitted = {"_object_id": F.SHA_A, "model": "qwen35_4b", "cell_id": ALLPOS_CELL, "eval_id": "agentic_leaking"}
    row_none = {"_object_id": None, "model": "qwen35_4b", "cell_id": "none", "eval_id": "agentic_leaking"}
    con = _registry_conn()
    ctx = AdapterContext(root="", now=NOW, producer=dict(PRODUCER), conn=con)
    from sisteer.registry.adapters.base import RecordBatch
    batch = RecordBatch("em_misalign:test", "test")
    assert EM.EmMisalignAdapter.unregistered_object_why(ctx, row_fitted, batch) is not None
    assert EM.EmMisalignAdapter.unregistered_object_why(ctx, row_none, batch) is None
    batch.add("object", {"object_id": F.SHA_A}, dict(PRODUCER), NOW)                    # minted in this batch: registered enough
    assert EM.EmMisalignAdapter.unregistered_object_why(ctx, row_fitted, batch) is None
    assert EM.EmMisalignAdapter.unregistered_object_why(AdapterContext(root="", now=NOW, producer=dict(PRODUCER), conn=None), row_fitted, RecordBatch("x", "x")) is None
    assert EM.EmMisalignAdapter.EMISSION_VERSION.startswith("12+emitter")


@pytest.mark.parametrize("sha", ["not-a-sha", "", None, 12])
def test_a_row_with_no_well_formed_object_sha_is_not_the_rules_business(sha):
    ctx = AdapterContext(root="", now=NOW, producer=dict(PRODUCER), conn=_registry_conn())
    from sisteer.registry.adapters.base import RecordBatch
    assert EM.EmMisalignAdapter.unregistered_object_why(ctx, {"_object_id": sha}, RecordBatch("x", "x")) is None
