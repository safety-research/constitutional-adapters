"""The status page's "Pending source revisions" section: every `source_revision` row an adapter pass wrote
`accepted = 0` that no later revision superseded, grouped by adapter and by the HEAD of its refusal (the text up to the first `: `),
with the count and the latest observation -- so a held read (an activation pending on an unregistered parent, a cell no standup record
names) is visible on the page instead of in a journal (371 objects_bank + 155 regrade_twin + 11 em_misalign + 4 sampled_static
of 690 live rows on 2026-10-06). One query in `status.QUERIES`, rendered with the template's table macro.
"""
from __future__ import annotations

import os

from sisteer.registry import db as DB
from sisteer.registry import status

from test_registry_status_ops import _render, _setup  # noqa: E402

T0 = "2026-10-06T01:11:00.000Z"
T1 = "2026-10-06T03:04:00.000Z"
T2 = "2026-10-06T03:40:00.000Z"


def _revision(conn, rid, source, adapter, utc, accepted, refusal=None, superseded_by=None):
    conn.execute("INSERT INTO source_revision (revision_id, source_id, adapter, observed_utc, inputs_json, accepted, refusal, superseded_by_revision_id, "
                 "priority, reclaimed, field_versions_json, extra_json, record_uri, ingest_seq) VALUES (?,?,?,?,?,?,?,?,0,0,'{}','{}',?,?)",
                 (rid, source, adapter, utc, "{}", accepted, refusal, superseded_by, f"records/adapters/{adapter}/{rid}.jsonl", 1))


def test_pending_revisions_group_held_reads_by_adapter_and_reason_head(tmp_path):
    root = _setup(tmp_path)
    conn = DB.connect(DB.db_path(root))
    conn.execute("PRAGMA foreign_keys=OFF")
    held = "activation pending: object 0f654e88aa00bb11 (records/adapters/em_misalign/000005-run.jsonl#9)"
    _revision(conn, "r" * 16, "em/agentic/llama8b_bl__r0p1-superseded", "em_misalign", T0, 0, held)
    _revision(conn, "s" * 16, "em/agentic/llama8b_bl", "em_misalign", T1, 0, held.replace("#9", "#12"))
    _revision(conn, "t" * 16, "sampled_lane", "sampled_static", T2, 0, "synthmodel/constitution_lora@deadbeef: no standup record on this machine names this cell's served object sha256")
    _revision(conn, "u" * 16, "sampled_lane", "sampled_static", T0, 1)                                   # accepted: not pending
    _revision(conn, "v" * 16, "r2_export", "r2_export", T0, 0, "torn store: the parquet is shorter than its sidecar", superseded_by="w" * 16)   # superseded: gone
    _revision(conn, "w" * 16, "r2_export", "r2_export", T1, 1)
    _revision(conn, "x" * 16, "objects_bank", "objects_bank", T1, 0, "CHECK constraint failed: paper_label IN (...)")        # no `: ` head -> the whole text
    conn.commit()
    conn.close()
    ctx = status.status_json(root)
    rows = {(r["adapter"], r["reason"]): r for r in ctx["pending_revisions"]}
    assert set(rows) == {("em_misalign", "activation pending"), ("sampled_static", "synthmodel/constitution_lora@deadbeef"),
                         ("objects_bank", "CHECK constraint failed")}, rows
    assert rows[("em_misalign", "activation pending")]["n_revisions"] == 2 and rows[("em_misalign", "activation pending")]["last_observed"] == T1
    assert rows[("sampled_static", "synthmodel/constitution_lora@deadbeef")]["n_revisions"] == 1
    assert rows[("objects_bank", "CHECK constraint failed")]["n_revisions"] == 1
    # the newest refusal text rides along in full, so the page shows the exact hold without a journal read
    assert rows[("em_misalign", "activation pending")]["example"].endswith("#12)")
    html = _render(root, tmp_path)
    assert "Pending source revisions" in html and "activation pending" in html and "em_misalign" in html and "objects_bank" in html
    assert "no standup record on this machine" in html
    # the section is one of the page's SQL sections (cached like the rest) and its state is tracked
    assert "pending_revisions" in status.QUERIES and ctx["section_state"]["pending_revisions"]["state"] == status.STATE_LIVE


def test_pending_revisions_is_empty_on_a_root_with_no_held_read(tmp_path):
    root = _setup(tmp_path)
    ctx = status.status_json(root)
    assert ctx["pending_revisions"] == []
    html = _render(root, tmp_path)
    assert "Pending source revisions" in html
    assert os.path.isfile(DB.db_path(root))
