"""Status page completions: the "objects in / out of the current round" section renders
``v_paper_objects`` (shipped in the schema) and says "roster not loaded" on a database without it; the operational
sections render from the state files (footprint, mirror lag, paper rebuild RESULT, timer health, live
leases, disk headroom); the snapshot push outcome shows per destination; the not-measured degeneration count
per eval category is a tile; every section heading is present and the page stays free of the banned words.
"""
from __future__ import annotations

import json
import os
import re

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import mirrors as M
from sisteer.registry import snapshot as SN
from sisteer.registry import status
from sisteer.registry import synth
from sisteer.registry import reads_protocol as RP
from front_door_testlib import MATCHERS, SCAN, token_hits

#: the process tier's case-insensitive rows of the publication scan's token table (the session manager's name and the private-path
#: shapes), matched as the scan matches them; its case-sensitive rows are capitalised names, one of which is also a method's
#: display name on this page
SESSION_TOKENS = SCAN.Matcher([r for r in MATCHERS["rows"] if r.tier == "process" and not r.cs])


def _setup(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=2, n_items=5)
    H.ingest_once(root, workers=0, min_free=0)
    return root


def _render(root, tmp_path, name="index.html"):
    out = str(tmp_path / name)
    status.render(root, out_path=out, register=False)
    return open(out, encoding="utf-8").read()


def test_paper_objects_section_renders_the_shipped_rows_and_says_roster_not_loaded_without_them(tmp_path):
    """The schema ships ``v_paper_objects``, so a fresh root renders the section's rows (every object,
    ``in_paper`` 0 until a roster is loaded); a database predating that migration (``v_paper_objects`` dropped
    here) says "roster not loaded" and names it — never an empty table that looks complete."""
    root = _setup(tmp_path)
    ctx = status.status_json(root)
    assert ctx["paper_objects_available"] is True and len(ctx["paper_objects"]) >= 1
    assert all(r["in_paper"] == 0 for r in ctx["paper_objects"])
    html = _render(root, tmp_path)
    assert "Objects in / out of the current round" in html and "roster not loaded" not in html
    assert "excluding_ruling_ids_json" in html
    conn = DB.connect(DB.db_path(root))
    conn.execute("DROP VIEW v_paper_objects")
    conn.commit()
    conn.close()
    ctx2 = status.status_json(root)
    assert ctx2["paper_objects_available"] is False and ctx2["paper_objects"] == []
    html2 = _render(root, tmp_path, "b.html")
    assert "roster not loaded" in html2 and "v_paper_objects" in html2


def test_operational_sections_render_from_state_files(tmp_path):
    root = _setup(tmp_path)
    state = os.path.join(root, "state")
    info = SN.build(root, DB.db_path(root), check_free_space=False)
    with SN.footprint_watcher(root, interval_s=0.05) as w:
        pass
    SN.write_footprint(root, w, info, DB.db_path(root))
    M.write_state(root, [{"destination": "cluster_a", "kind": "ssh", "manifest": "CHECKSUMS-000000000042.sha256", "lag_files": 3,
                          "unsynced_local_files": 1, "switched": False, "error": "ssh: connect timed out", "utc": synth.t(1)}])
    with open(os.path.join(state, "paper_rebuild.RESULT"), "w") as fh:
        fh.write("REFUSED 77 2026-01-01T00:00:00Z roster entries with neither a judged run nor a disposition: 2\n")
    with open(os.path.join(state, "paper_failed_head"), "w") as fh:
        fh.write("77 refused\n")
    os.makedirs(os.path.join(state, "last_run"))
    with open(os.path.join(state, "last_run", "ingest.json"), "w") as fh:
        json.dump({"command": "ingest", "utc": synth.t(2), "rc": 0, "seconds": 0.5}, fh)
    # a live lease from a remote reader
    ld = os.path.join(SN.snapshots_dir(root), SN.LEASES_DIR, info.snapshot_id)
    os.makedirs(ld)
    with open(os.path.join(ld, "fc-abc123.deadbeef"), "w") as fh:
        fh.write(json.dumps({"touched_utc": synth.t(3)}))
    ctx = status.status_json(root)
    assert ctx["footprint"]["snapshot_id"] == info.snapshot_id and ctx["footprint"]["peak_bytes"] > 0
    assert ctx["mirror_lag"][0]["lag_files"] == 3 and ctx["mirror_lag"][0]["error"] == "ssh: connect timed out"
    assert ctx["paper_rebuild"]["result"].startswith("REFUSED 77") and ctx["paper_rebuild"]["failed_head"] == "77 refused"
    assert ctx["timers"][0]["command"] == "ingest" and ctx["timers"][0]["rc"] == 0
    assert ctx["leases"][0]["call_id"] == "fc-abc123" and ctx["leases"][0]["snapshot_id"] == info.snapshot_id
    assert ctx["snapshots_resident"][0]["snapshot_id"] == info.snapshot_id and ctx["snapshots_resident"][0]["leases"] == 1
    assert ctx["snapshot_current"] == info.snapshot_id
    assert ctx["disk"]["floor_gb"] == 5.0 and ctx["disk"]["budget_gb"] == 10 and "headroom_gb" in ctx["disk"]
    html = _render(root, tmp_path)
    for heading in ("Snapshot freshness per destination", "Live snapshot leases", "Mirror lag", "Disk headroom", "Paper rebuild",
                    "Timer health", "Degeneration not measured", "Dispositions", "Objects in / out of the current round"):
        assert heading in html, heading
    assert "ssh: connect timed out" in html and "REFUSED 77" in html and "fc-abc123" in html and "CHECKSUMS-000000000042" in html
    assert "degeneration pairs not measured" in html


def test_snapshot_push_outcome_renders_per_destination(tmp_path):
    root = _setup(tmp_path)
    info = SN.build(root, DB.db_path(root), check_free_space=False)
    pushed = SN.push(root, info.snapshot_id, [{"name": "cluster_a", "kind": "local", "path": str(tmp_path / "b")},
                                               {"name": "legacy", "kind": "modal", "volume": "v", "path": "/registry/snapshot",
                                                "python": "/no/python", "app": "/no/app"}])      # the removed kind (C10): an error row by name
    H.write_harvester_record(root, "snapshot", SN.snapshot_record_payload(info, pushed))
    H.ingest_once(root, workers=0, min_free=0)
    ctx = status.status_json(root)
    rows = {r["destination"]: r for r in ctx["snapshot_pushes"]}
    assert rows["cluster_a"]["ok"] is True and rows["legacy"]["ok"] is False and "unknown snapshot destination kind" in rows["legacy"]["error"]
    assert ctx["snapshots"][0]["scope"] == "full"
    html = _render(root, tmp_path)
    assert "unknown snapshot destination kind" in html


def test_page_stays_free_of_banned_words_with_every_section_populated(tmp_path, monkeypatch):
    monkeypatch.setenv(RP.STATE_ENV, str(tmp_path / "landing-state"))      # the page reads the landing state: an empty per-test one, never the host's
    root = _setup(tmp_path)
    html = _render(root, tmp_path)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\b\w*_legacy\b", " ", text)
    for word in ("arm", "leg", "lane", "cell", "wave", "grader"):
        assert not re.search(r"(?<![A-Za-z0-9_])" + word + r"(?![A-Za-z0-9_])", text, re.IGNORECASE), word
    for jargon in ("card", "board", "column", "nudge"):   # no session-manager vocabulary on the page
        assert not re.search(r"(?<![A-Za-z0-9_-])" + jargon + r"(?![A-Za-z0-9_-])", text, re.IGNORECASE), jargon
    assert not token_hits(text, SESSION_TOKENS), token_hits(text, SESSION_TOKENS)   # nor the session manager's own name
