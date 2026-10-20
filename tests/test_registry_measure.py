"""The measurement scripts under ``scripts/registry/measure/`` (registry-slurm-fast-windows; the homework's Q1.2 / Q1.4 and the S4 /
S5 instruments), each on a small fixture: the stage-stamp parser (two windows, one aborted, a midnight rollover, the dims key after
the time), the journal grouping (an ok pass, a ``LockHeld`` exit, a timeout, a running pass; the indent-1 summary reassembled), the
render-series analysis (consecutive totals within budget, the stamp cadence, section freshness) and the sar parser (ISO header, the
window filter, the median). Standard-library scripts loaded from their path; every one has ``--json``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEASURE = os.path.join(REPO, "scripts", "registry", "measure")


def _load(name):
    spec = importlib.util.spec_from_file_location(f"measure_{name}", os.path.join(MEASURE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


CHAIN_LOG = """=== ff chain start 2026-09-24T23:50:00Z ADAPTERS=alpha beta
timers active at stop: registry-harvest.timer
writers quiet 23:50:15
base at 1111111 23:50:16
-- migrate (apply) 23:50:17
-- dimensions load 23:52:00 (seeds=abc schema=17 db=5)
-- adapter alpha 23:58:00
 "records_written": 0
{"ingest_ok": 0, "refused": 0, "blocked": 0}
-- adapter beta --full 00:02:00
--- ingest 00:05:00
=== ff chain end 2026-09-25T00:10:00Z
timers restored (trap; the set active at stop) 00:10:00
=== ff chain start 2026-09-25T01:00:00Z ADAPTERS=gamma
timers active at stop: none
writers quiet 01:00:15
base at 2222222 01:00:16
-- install units 01:00:17
-- migrate (apply) 01:00:20
-- dimensions load SKIPPED (seeds + schema unchanged since the last load: seeds=abc) 01:00:21
-- adapter gamma 01:00:22
=== ff chain start 2026-09-25T02:00:00Z ADAPTERS=delta
writers quiet 02:00:15
base at 3333333 02:00:16
-- migrate (apply) 02:00:17
-- dimensions load SKIPPED (x) 02:00:18
-- adapter delta 02:00:19
--- ingest 02:03:19
=== ff chain end 2026-09-25T02:05:19Z
"""


def test_windows_parser_stages_totals_rollover_and_abort(tmp_path):
    W = _load("windows")
    log = tmp_path / "chain.log"
    log.write_text(CHAIN_LOG)
    windows = W.read_logs([str(log)])
    assert [w["start"] for w in windows] == ["2026-09-24T23:50:00Z", "2026-09-25T01:00:00Z", "2026-09-25T02:00:00Z"]
    w0, w1, w2 = windows
    assert w0["completed"] and w0["total_s"] == 1200 and w0["base"] == "1111111" and w0["adapters"] == "alpha beta" and w0["quiet_wait_s"] == 15
    stages = {s["label"]: (s["class"], s["seconds"]) for s in w0["stages"]}
    # the stamp is printed BEFORE its stage: wall = next stamp − own; the last stage ends at the end marker; midnight rolls the date
    assert stages["migrate (apply)"] == ("migrate", 103) and stages["dimensions load"] == ("dims", 360)
    assert stages["adapter alpha"] == ("adapter", 240) and stages["adapter beta --full"] == ("adapter", 180) and stages["ingest"] == ("ingest", 300)
    assert stages["base at 1111111"] == ("base", 1)
    assert not w1["completed"] and w1["end"] is None and w1["total_s"] == 22          # aborted: no end marker before the next start
    assert [s["class"] for s in w1["stages"]] == ["base", "install", "migrate", "dims_skip", "adapter"]
    assert w1["stages"][-1]["seconds"] is None
    # C6: these historical windows wrote the end marker BEFORE the timer restore -- marked `pre-rev3`, no abort cause recorded
    assert all(w["end_definition"] == "pre-rev3" and w["abort_cause"] is None for w in windows)
    s = W.summarize(windows)
    assert s["n_started"] == 3 and s["n_completed"] == 2 and s["n_aborted"] == 1
    assert s["median_min"] == 12.7 and s["max_min"] == 20.0 and s["min_min"] == 5.3 and s["sum_min"] == 25.3
    assert s["by_class"]["adapter"]["n"] == 3 and s["by_class"]["adapter"]["sum_min"] == 10.0 and s["by_class"]["quiet"]["n"] == 3
    rows = W.window_rows(windows)
    assert rows[0]["longest"] == "dimensions load 6.0" and rows[0]["second"] == "ingest 5.0" and rows[0]["total_min"] == 20.0
    # the compat copy of the same window is counted once
    log2 = tmp_path / "compat.log"
    log2.write_text(CHAIN_LOG.split("=== ff chain start 2026-09-25T01:00:00Z")[0])
    assert len(W.read_logs([str(log), str(log2)])) == 3
    r = subprocess.run([sys.executable, os.path.join(MEASURE, "windows.py"), str(log), "--since", "2026-09-25", "--json"], capture_output=True, text=True)
    assert r.returncode == 0
    out = json.loads(r.stdout)
    assert out["summary"]["n_started"] == 2 and out["summary"]["n_completed"] == 1
    r = subprocess.run([sys.executable, os.path.join(MEASURE, "windows.py"), str(log)], capture_output=True, text=True)
    assert "Completed windows n=2 of 3 started (1 aborted): median 12.7 min" in r.stdout and "(ABORTED: no end marker)" in r.stdout


CLUSTER_LOG = """=== ff chain start 2026-09-29T20:00:00Z ADAPTERS=heavy_one alpha
-- store mirror 20:00:00
writers quiet 20:05:00
base at 4444444 20:05:01
-- install units 20:05:02
-- migrate (apply) 20:05:03
-- dimensions load SKIPPED (x) 20:05:04
-- reader-state export 20:05:05
-- tree export 20:06:05
-- cluster reads plan 20:06:35
-- cluster reads stage 20:07:05
-- cluster reads submit 20:07:15
-- cluster reads revalidate 20:17:15
-- cluster reads pull 20:17:45
-- adapter heavy_one [fallback grading host] 20:18:00
-- adapter alpha 20:18:30
--- ingest 20:19:00
=== ff chain end 2026-09-29T20:24:00Z
=== ff chain start 2026-09-29T21:00:00Z ADAPTERS=alpha beta
writers quiet 21:00:10
base at 5555555 21:00:11
-- install units 21:00:12
-- migrate (apply) 21:00:13
-- dimensions load SKIPPED (x) 21:00:14
-- cluster reads SKIPPED (no named adapter has a cluster policy) 21:00:15
-- adapter alpha 21:00:15
-- adapter beta 21:01:15
--- ingest 21:02:15
=== ff chain end 2026-09-29T21:05:15Z
"""


def test_windows_parser_classes_the_cluster_stages_the_store_mirror_and_the_skipped_cluster_block(tmp_path):
    """The cluster stages (reader-state export, tree export, every `cluster reads <stage>`) are one class, the store
    mirror (before the timers stop) its own, and a window that names no adapter with a cluster policy carries ONE `cluster reads SKIPPED`
    stamp classed `cluster_skip` with (here) a zero wall -- so the by-class table separates a cluster window from a reader-only one."""
    W = _load("windows")
    log = tmp_path / "chain.log"
    log.write_text(CLUSTER_LOG)
    w0, w1 = W.read_logs([str(log)])
    c0 = {s["label"]: (s["class"], s["seconds"]) for s in w0["stages"]}
    assert c0["store mirror"] == ("store_mirror", 301) and c0["reader-state export"] == ("cluster", 60) and c0["tree export"] == ("cluster", 30)
    assert c0["cluster reads submit"] == ("cluster", 600) and c0["cluster reads pull"] == ("cluster", 15)
    assert c0["adapter heavy_one [fallback grading host]"] == ("adapter", 30) and c0["adapter alpha"] == ("adapter", 30)
    assert [s["class"] for s in w1["stages"]] == ["base", "install", "migrate", "dims_skip", "cluster_skip", "adapter", "adapter", "ingest"]
    skip = next(s for s in w1["stages"] if s["class"] == "cluster_skip")
    assert skip["label"] == "cluster reads SKIPPED (no named adapter has a cluster policy)" and skip["seconds"] == 0
    by = W.summarize([w0, w1])["by_class"]
    assert by["cluster"]["n"] == 7 and by["cluster"]["sum_min"] == 12.9 and by["store_mirror"]["n"] == 1 and by["cluster_skip"]["n"] == 1
    assert W.stage_class("cluster reads SKIPPED (no named adapter has a cluster policy)") == "cluster_skip" and W.stage_class("cluster reads plan") == "cluster"


REV3_LOG = """=== ff chain start 2026-10-05T10:00:00Z ADAPTERS=heavy_one alpha
maintenance flag written: /x/state/maintenance.json 10:00:00
timers active at stop: registry-harvest.timer
retired readers stopped and confirmed (terminal, no main PID, cgroup empty): registry-adapters.service registry-modal-pull.service 10:00:03
writers quiet 10:00:10
base at 6666666 10:00:11
-- install units 10:00:12
-- migrate (apply) 10:00:20
-- dimensions load SKIPPED (x) 10:00:21
-- reader-state export 10:00:22
-- tree export 10:01:22
-- cluster reads plan 10:01:52
-- cluster reads stage 10:02:00
-- cluster reads submit 10:02:10
-- cluster reads revalidate 10:05:10
-- cluster reads pull 10:05:40
--- ingest 10:06:00
--- ingest end 10:07:30
maintenance flag removed 10:07:30
restore set (active at stop ∪ newly installed − registry-snapshot.timer registry-adapters.timer registry-modal-pull.timer): registry-harvest.timer
timers restored (trap; the set above) 10:07:35
=== ff chain end 2026-10-05T10:07:35Z
=== ff chain start 2026-10-05T11:00:00Z ADAPTERS=heavy_one alpha
timers active at stop: registry-harvest.timer
writers quiet 11:00:10
base at 7777777 11:00:11
-- install units 11:00:12
-- migrate (apply) 11:00:13
-- dimensions load SKIPPED (x) 11:00:14
-- cluster reads SKIPPED (no named adapter has a cluster policy) 11:00:15
-- adapter heavy_one REFUSED 11:00:15 (no cluster policy named for it this window (CLUSTER_ADAPTERS); the grading host reads no store)
-- adapter alpha REFUSED 11:00:15 (declared cluster: refused in the configuration -- its pinned paper worktree is absent (D13))
--- ingest 11:00:16
--- ingest end 11:01:16
refused: heavy_one alpha; abort
window ended early (rc=15) 11:01:16
timers restored (trap; the set above) 11:01:20
=== ff chain ABORTED (refused: heavy_one alpha) 2026-10-05T11:01:20Z
=== ff chain start 2026-10-05T12:00:00Z ADAPTERS=
timers active at stop: registry-status.timer
writers quiet 12:00:10
base at 8888888 12:00:11
-- install units 12:00:12
-- migrate (apply) 12:00:13
-- dimensions load SKIPPED (x) 12:00:14
--- ingest 12:00:15
--- ingest end 12:01:15
timers NOT restored: not active after the start: registry-status.timer (trap) 12:01:20
=== ff chain ABORTED (restore) 2026-10-05T12:01:20Z
"""


def test_windows_parser_reads_the_rev3_end_marker_the_aborted_marker_the_ingest_end_stage_and_the_refused_adapter_class(tmp_path):
    """The end marker is written by the trap AFTER the verified timer restore, so the measured
    window includes it and a window whose end follows a `timers restored` line is `rev3`; `=== ff chain ABORTED (<cause>) <ISO>` closes a
    window as NOT completed with the cause on the record (a refused adapter, a failed restore); `--- ingest end` is its own stage class
    (its wall = the trap's restore); `-- adapter A REFUSED HH:MM:SS (reason)` is classed `adapter_refused` and the reason inside the
    parenthesis never shifts the stamp's clock. The pre-rev3 fixture above stays readable as it was."""
    W = _load("windows")
    log = tmp_path / "chain.log"
    log.write_text(REV3_LOG)
    w0, w1, w2 = W.read_logs([str(log)])
    assert w0["completed"] and w0["end_definition"] == "rev3" and w0["abort_cause"] is None and w0["total_s"] == 455
    c0 = {s["label"]: (s["class"], s["seconds"]) for s in w0["stages"]}
    assert c0["ingest"] == ("ingest", 90) and c0["ingest end"] == ("ingest_end", 5) and c0["cluster reads pull"] == ("cluster", 20)
    assert not w1["completed"] and w1["end"] is None and w1["end_definition"] == "rev3"
    assert w1["abort_cause"] == "refused: heavy_one alpha" and w1["aborted_at"] == "2026-10-05T11:01:20Z" and w1["total_s"] == 80
    c1 = {s["label"]: (s["class"], s["seconds"]) for s in w1["stages"]}
    assert c1["adapter heavy_one REFUSED"] == ("adapter_refused", 0) and c1["adapter alpha REFUSED"] == ("adapter_refused", 1)
    assert c1["ingest end"] == ("ingest_end", 4)            # the last stage ends at the ABORTED marker
    assert W.stage_class("adapter heavy_one REFUSED") == "adapter_refused" and W.stage_class("adapter heavy_one --full") == "adapter"
    assert W.stage_class("ingest end") == "ingest_end" and W.stage_class("ingest") == "ingest"
    assert W.stage_class("status heavy refresh started once (cache initializer)") == "status_heavy"      # the trap's stamp, inside a rev3 window
    assert not w2["completed"] and w2["abort_cause"] == "restore" and w2["end_definition"] == "rev3"
    s = W.summarize([w0, w1, w2])
    assert s["n_started"] == 3 and s["n_completed"] == 1 and s["n_aborted"] == 2 and s["by_class"]["adapter_refused"]["n"] == 2
    assert s["by_class"]["ingest_end"]["n"] == 3
    rows = W.window_rows([w0, w1, w2])
    assert [r["end_definition"] for r in rows] == ["rev3", "rev3", "rev3"] and rows[1]["abort_cause"] == "refused: heavy_one alpha"
    r = subprocess.run([sys.executable, os.path.join(MEASURE, "windows.py"), str(log)], capture_output=True, text=True)
    assert r.returncode == 0 and "(ABORTED: refused: heavy_one alpha)" in r.stdout and "(ABORTED: restore)" in r.stdout, r.stdout
    assert "| rev3 |" in r.stdout and "Completed windows n=1 of 3 started (2 aborted)" in r.stdout


C12_LOG = """=== ff chain start 2026-10-10T10:00:00Z ADAPTERS=alpha beta
-- reads reservation 10:00:00 (op W1, holder pid 7)
writers quiet 10:00:20
base at 1111111 10:00:21
-- install units 10:00:22
-- migrate (apply) 10:00:23
-- dimensions load SKIPPED (x) 10:00:24
-- reader-state export 10:00:25
-- tree export 10:00:40
-- cluster reads plan 10:00:45
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
=== ff chain start 2026-10-10T11:00:00Z ADAPTERS=
-- reads reservation 11:00:00 (op W2, holder pid 8)
writers quiet 11:00:10
base at 2222222 11:00:11
--- ingest 11:00:12
--- ingest end 11:01:12
timers NOT restored: x (trap) 11:01:15
=== ff chain ABORTED (restore) 2026-10-10T11:01:15Z
=== ff chain start 2026-10-10T12:00:00Z ADAPTERS=gamma
-- reads reservation 12:00:00 (op W3, holder pid 9)
writers quiet 12:00:30
base at 3333333 12:00:31
--- ingest 12:00:32
--- ingest end 12:03:32
timers restored (trap; the set above) 12:03:35
=== ff chain end 2026-10-10T12:03:35Z
=== ff chain start 2026-10-10T13:00:00Z ADAPTERS=
writers quiet 13:00:05
base at 4444444 13:00:06
--- ingest 13:00:07
=== ff chain end 2026-10-10T13:02:07Z
"""


def _c12_receipt(op_id, utc, adapters):
    return {"schema": "reads_receipt/1", "op": {"op_id": op_id, "owner": "window"}, "utc": utc, "window": op_id, "adapters": adapters}


#: the live read before the tenth completion: one completed window, then an ABORTED one and nothing after it
PARTIAL_LOG = """=== ff chain start 2026-10-10T10:00:00Z ADAPTERS=alpha
-- reads reservation 10:00:00 (op P1, holder pid 7)
writers quiet 10:00:20
base at 1111111 10:00:21
--- ingest 10:00:22
--- ingest end 10:03:22
timers restored (trap; the set above) 10:03:25
=== ff chain end 2026-10-10T10:03:25Z
=== ff chain start 2026-10-10T11:00:00Z ADAPTERS=alpha
-- reads reservation 11:00:00 (op P2, holder pid 8)
writers quiet 11:00:10
base at 2222222 11:00:11
-- adapter alpha REFUSED 11:00:30 (pull: 1 shard(s) refused)
timers restored (trap; the set above) 11:00:40
=== ff chain ABORTED (refused: alpha) 2026-10-10T11:00:45Z
"""


def test_windows_first_ten_rule_until_bound_and_the_changed_set_columns_from_the_receipts(tmp_path):
    """`--first N` keeps the first N COMPLETED windows of the span -- an ABORTED window inside it is listed with its cause and never fills
    a slot; `--until` bounds the span; each window's op id is the reservation stamp's, and its read receipt is joined by it (adapters read
    whole / refused, revisions pulled, re-activations, the newest change_utc, the pins' input bytes from the grading host's manifests --
    None when a pin has no manifest here); `records_folded` / `files_folded` come from the harvester's summary lines inside the ingest
    stage; `drain_s` is the measured drain; the sub-populations' medians stand beside the summary."""
    W = _load("windows")
    log = tmp_path / "chain.log"
    log.write_text(C12_LOG)
    rd = tmp_path / "reads" / "receipts"
    rd.mkdir(parents=True)
    pub = {"published_utc": "2026-10-10T09:50:00.000Z", "change_utc": "2026-10-10T09:45:00.000Z", "mode": "push", "producer": "grading_host:x"}
    pub2 = {"published_utc": "2026-10-10T09:58:00.000Z", "change_utc": "2026-10-10T09:55:00.000Z", "mode": "carry", "producer": None}
    (rd / "W1.json").write_text(json.dumps(_c12_receipt("W1", "2026-10-10T10:02:40.000Z", {
        "alpha": {"pins": {"comp": "11111111"}, "published": {"comp": pub}, "shards_total": 2, "shards_ok": 2, "revisions_pulled": ["alpha/r1", "alpha/r2"],
                  "reactivated": ["r0"], "refused": None, "read_whole": True},
        "beta": {"pins": {"other": "22222222"}, "published": {"other": pub2}, "shards_total": 2, "shards_ok": 1, "revisions_pulled": [],
                 "reactivated": [], "refused": ["pull: 1 shard(s) refused"], "read_whole": False}})))
    (rd / "W3.json").write_text(json.dumps(_c12_receipt("W3", "2026-10-10T12:03:00.000Z", {
        "gamma": {"pins": {"comp": "11111111"}, "published": {"comp": pub}, "shards_total": 1, "shards_ok": 1, "revisions_pulled": ["gamma/r9"],
                  "reactivated": [], "refused": None, "read_whole": True}})))
    (rd / "foreign.json").write_text(json.dumps({"schema": "x/1", "op": {"op_id": "W2"}}))
    root = tmp_path / "reg"
    man = root / "state" / "store_manifests" / "comp"
    man.mkdir(parents=True)
    (man / "MANIFEST-11111111.json").write_text(json.dumps({"schema": "store_manifest/1", "slug": "comp", "content_sha8": "11111111", "bytes": 4096, "n_files": 7}))
    windows = W.read_logs([str(log)])
    assert [w["op_id"] for w in windows] == ["W1", "W2", "W3", None]
    W.join_receipts(windows, W.read_receipts(str(rd)), str(root))
    w1, w2, w3, w4 = windows
    # the changed-set columns: the set named on the start line; the receipt's reading of it; the pins' bytes (beta's pin has no manifest here -> None)
    assert w1["changed_set"] == ["alpha", "beta"] and w1["changed_set_n"] == 2 and w1["cluster_read"] and w1["receipt"]
    assert w1["read_whole_n"] == 1 and w1["refused_n"] == 1 and w1["revisions_pulled"] == 2 and w1["reactivated"] == 1
    assert w1["change_utc"] == "2026-10-10T09:55:00.000Z" and w1["published_utc"] == "2026-10-10T09:58:00.000Z"
    assert w1["input_bytes"] is None and w1["pins_without_manifest"] == ["other/22222222"] and w1["pins"] == {"comp": "11111111", "other": "22222222"}
    assert w1["records_folded"] == 340 and w1["files_folded"] == 12 and w1["drain_s"] == 20 and w1["quiet_wait_s"] == 20 and w1["end_definition"] == "rev3"
    assert w2["receipt"] is False and w2["read_whole_n"] is None and not w2["completed"] and w2["abort_cause"] == "restore" and w2["records_folded"] is None
    assert w3["receipt"] and w3["input_bytes"] == 4096 and w3["input_files"] == 7 and w3["revisions_pulled"] == 1 and w3["changed_set_n"] == 1 and not w3["cluster_read"]
    assert w4["op_id"] is None and w4["receipt"] is False and w4["changed_set_n"] == 0 and w4["end_definition"] == "pre-rev3"
    # the first-N rule: the span's first 2 completed windows (W1, W3); the aborted W2 sits between them and is listed, never counted
    sel, aborted = W.select_span(windows, "2026-10-10", None, 2)
    assert [w["op_id"] for w in sel] == ["W1", "W3"] and [w["op_id"] for w in aborted] == ["W2"]
    sel1, aborted1 = W.select_span(windows, "2026-10-10", None, 1)
    assert [w["op_id"] for w in sel1] == ["W1"] and aborted1 == []          # W2 starts after the first completed window: outside the ONE
    assert [w["start"] for w in W.select_span(windows, None, "2026-10-10T12:00:00Z")[0]] == ["2026-10-10T10:00:00Z", "2026-10-10T11:00:00Z"]
    # the live read before the tenth completion: with FEWER than N completed windows every abort of the span
    # is listed, one after the latest completion included (the cutoff used to be the latest completion's start even then, so a later abort
    # vanished from aborted_in_span until another window completed)
    sel10, aborted10 = W.select_span(windows, "2026-10-10", None, 10)
    assert [w["op_id"] for w in sel10] == ["W1", "W3", None] and [w["op_id"] for w in aborted10] == ["W2"]
    plog = tmp_path / "partial.log"
    plog.write_text(PARTIAL_LOG)
    partial = W.read_logs([str(plog)])
    assert [w["completed"] for w in partial] == [True, False] and partial[1]["abort_cause"] == "refused: alpha"
    sel_p, aborted_p = W.select_span(partial, "2026-10-10", None, 10)
    assert [w["op_id"] for w in sel_p] == ["P1"] and [(w["op_id"], w["abort_cause"]) for w in aborted_p] == [("P2", "refused: alpha")]
    # once the Nth completion EXISTS its start is the cutoff: the same abort, after the one completion, is outside the one
    assert W.select_span(partial, "2026-10-10", None, 1)[1] == []
    # --until bounds the aborts as it bounds the span
    assert W.select_span(partial, "2026-10-10", "2026-10-10T11:00:00Z", 10) == ([partial[0]], [])
    r = subprocess.run([sys.executable, os.path.join(MEASURE, "windows.py"), str(plog), "--since", "2026-10-10", "--first", "10", "--receipts", str(rd),
                        "--root", str(root), "--json"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    pj = json.loads(r.stdout)
    assert pj["summary"]["n_completed"] == 1 and pj["summary"]["aborted_in_span"] == [{"start": "2026-10-10T11:00:00Z", "abort_cause": "refused: alpha", "adapters": "alpha"}]
    s = W.summarize(sel)
    assert s["n_completed"] == 2 and s["median_min"] == 4.0 and s["end_definitions"] == ["rev3"] and s["n_with_receipt"] == 2 and s["n_without_receipt"] == 0   # (260 + 215) / 2 s
    sp = s["subpopulations"]
    assert sp["with_cluster_read"]["n"] == 1 and sp["with_cluster_read"]["median_min"] == 4.3 and sp["without_cluster_read"]["median_min"] == 3.6
    assert sp["changed_set_1_2"]["n"] == 2 and sp["changed_set_0"]["n"] == 0 and sp["receipt_with_refusal"]["n"] == 1 and sp["receipt_all_read_whole"]["n"] == 1
    rows = W.window_rows(sel)
    assert rows[0]["op_id"] == "W1" and rows[0]["revisions_pulled"] == 2 and rows[0]["input_bytes"] is None and rows[1]["input_bytes"] == 4096 and rows[0]["drain_s"] == 20
    # a window whose reservation stamp has no receipt is counted as such
    assert W.summarize(W.read_logs([str(log)]))["n_without_receipt"] == 3
    # the CLI: --first 2 --json names the aborted window; --until excludes; the text table carries the new columns
    r = subprocess.run([sys.executable, os.path.join(MEASURE, "windows.py"), str(log), "--since", "2026-10-10", "--first", "2", "--receipts", str(rd),
                        "--root", str(root), "--json"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["summary"]["first"] == 2 and out["summary"]["n_completed"] == 2 and [w["op_id"] for w in out["windows"]] == ["W1", "W3"]
    assert out["summary"]["aborted_in_span"] == [{"start": "2026-10-10T11:00:00Z", "abort_cause": "restore", "adapters": ""}]
    r = subprocess.run([sys.executable, os.path.join(MEASURE, "windows.py"), str(log), "--until", "2026-10-10T11:00:00Z", "--receipts", str(rd), "--root", str(root)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "Completed windows n=1 of 1 started" in r.stdout and "| 2 | 1/1 | 2 | — | 20 |" in r.stdout, r.stdout
    assert "sub-populations (completed): with_cluster_read n=1" in r.stdout
    r = subprocess.run([sys.executable, os.path.join(MEASURE, "windows.py"), str(log), "--first", "2", "--receipts", str(rd), "--root", str(root)], capture_output=True, text=True)
    assert "ABORTED in the span (not counted): 2026-10-10T11:00:00Z (restore)" in r.stdout and "first 2 completed windows of the span" in r.stdout
    assert "| 1 | 1/0 | 1 | 0 | 30 |" in r.stdout          # W3: changed 1, whole/refused 1/0, 1 revision, 4096 bytes -> 0 MB, drain 30 s
    # an absent receipts directory joins nothing and is not an error
    assert W.read_receipts(str(tmp_path / "nowhere")) == {} and W.manifest_bytes(str(root), "comp", "deadbeef") == (None, None)


def _journal(unit, inv, t0, lines, end=None):
    """Synthetic journal lines for one pass: systemd's Starting line, the program's lines, the terminal systemd line."""
    us = lambda s: str(int((t0 + s) * 1e6))  # noqa: E731
    out = [{"__REALTIME_TIMESTAMP": us(0), "SYSLOG_IDENTIFIER": "systemd", "_COMM": "systemd", "USER_UNIT": f"{unit}.service", "USER_INVOCATION_ID": inv,
            "MESSAGE": f"Starting {unit}.service - x...", "JOB_TYPE": "start"}]
    for i, msg in enumerate(lines):
        out.append({"__REALTIME_TIMESTAMP": us(1 + i), "SYSLOG_IDENTIFIER": unit, "_COMM": "python", "_SYSTEMD_INVOCATION_ID": inv, "MESSAGE": msg})
    if end is not None:
        kind, secs = end
        if kind == "ok":
            out.append({"__REALTIME_TIMESTAMP": us(secs), "SYSLOG_IDENTIFIER": "systemd", "_COMM": "systemd", "USER_UNIT": f"{unit}.service",
                        "USER_INVOCATION_ID": inv, "MESSAGE": f"Finished {unit}.service - x.", "JOB_RESULT": "done"})
        else:
            out.append({"__REALTIME_TIMESTAMP": us(secs - 0.001), "SYSLOG_IDENTIFIER": "systemd", "_COMM": "systemd", "USER_UNIT": f"{unit}.service",
                        "USER_INVOCATION_ID": inv, "MESSAGE": f"{unit}.service: Main process exited, code=exited, status=1/FAILURE",
                        "EXIT_CODE": "exited", "EXIT_STATUS": "1"})
            out.append({"__REALTIME_TIMESTAMP": us(secs), "SYSLOG_IDENTIFIER": "systemd", "_COMM": "systemd", "USER_UNIT": f"{unit}.service",
                        "USER_INVOCATION_ID": inv, "MESSAGE": f"{unit}.service: Failed with result '{kind}'.", "UNIT_RESULT": kind})
    return out


def test_timers_groups_passes_by_invocation_and_reads_outcomes_lockheld_and_summaries(tmp_path):
    T = _load("timers")
    t0 = 1_790_600_000.0
    lines = []
    lines += _journal("registry-harvest", "a1", t0, ["{", ' "files": 3,', ' "ok": 3,', ' "lock_hold_s": 12.5', "}"], ("ok", 150))
    lines += _journal("registry-harvest", "a2", t0 + 200, ["Traceback", "sisteer.registry.db.LockHeld: lock held: /x/harvester.lock"], ("exit-code", 30))
    lines += _journal("registry-harvest", "a3", t0 + 400, ["{", ' "files": 0', "}"], ("timeout", 2700))
    lines += _journal("registry-harvest", "a4", t0 + 3200, ["refused: x"])   # still running: no terminal line
    lines += _journal("registry-adapters", "b1", t0, ["{", ' "adapters": 49,', ' "records_written": 2', "}", '{"ingest_ok": 2, "refused": 0, "blocked": 0}'], ("ok", 500))
    lines += _journal("registry-adapters", "b2", t0 + 600, ["{", ' "adapters": 49,', ' "records_written": 0', "}", '{"ingest_ok": 0, "refused": 16, "blocked": 0}'], ("ok", 480))
    export = tmp_path / "journal.jsonl"
    export.write_text("".join(json.dumps(l) + "\n" for l in lines))
    grouped = T.passes_from(lines)
    h = grouped["registry-harvest"]
    assert [p["outcome"] for p in h] == ["ok", "exit-code", "timeout", "running"]
    assert [p["duration_s"] for p in h] == [150.0, 30.0, 2700.0, None]
    assert [p["lock_held"] for p in h] == [False, True, False, False] and h[1]["exit_status"] == "exited/1"
    assert h[0]["summaries"] == [{"files": 3, "ok": 3, "lock_hold_s": 12.5}]
    a = grouped["registry-adapters"]
    assert a[0]["ingest_ok"] == {"ingest_ok": 2, "refused": 0, "blocked": 0} and a[0]["summaries"][0]["records_written"] == 2
    th = T.unit_table(h, "registry-harvest")
    assert th["passes"] == 4 and th["ok"] == 1 and th["lock_held"] == 1 and th["timeouts"] == 1 and th["outcomes"]["running"] == 1
    assert th["duration_s"]["median"] == 150.0 and th["duration_s"]["max"] == 2700.0 and th["start_to_start_s"]["median"] == 200.0
    assert th["latest_summary"] == {"files": 0}
    ta = T.unit_table(a, "registry-adapters")
    assert ta["records_written_gt0"] == 1 and ta["ingest_ok_passes"] == 2 and ta["latest_summary"] == {"adapters": 49, "records_written": 0}
    r = subprocess.run([sys.executable, os.path.join(MEASURE, "timers.py"), "--from", str(export), "--json"], capture_output=True, text=True)
    assert r.returncode == 0
    out = json.loads(r.stdout)
    assert set(out["units"]) == {"registry-harvest", "registry-adapters"} and out["units"]["registry-harvest"]["lock_held"] == 1
    r = subprocess.run([sys.executable, os.path.join(MEASURE, "timers.py"), "--from", str(export)], capture_output=True, text=True)
    assert "| registry-harvest | 4 | 1 | 1 | 1 | 1 | 1 |" in r.stdout


def test_timers_reads_unit_result_when_the_failure_line_carries_no_quotes():
    """`Failed with result …` without a quoted result (or a locale that drops the quotes): systemd's UNIT_RESULT field decides;
    the message's quoted result is the fallback, "failed" the last resort (the `or` binds tighter than the conditional)."""
    T = _load("timers")
    t0 = 1_790_600_000.0
    us = lambda s: str(int((t0 + s) * 1e6))  # noqa: E731
    systemd = {"SYSLOG_IDENTIFIER": "systemd", "_COMM": "systemd", "USER_UNIT": "registry-status.service"}
    lines = _journal("registry-status", "c1", t0, ["rendering"])
    lines.append(dict(systemd, __REALTIME_TIMESTAMP=us(30), USER_INVOCATION_ID="c1", MESSAGE="registry-status.service: Failed with result timeout.",
                      UNIT_RESULT="timeout"))
    lines += _journal("registry-status", "c2", t0 + 100, ["rendering"])
    lines.append(dict(systemd, __REALTIME_TIMESTAMP=us(130), USER_INVOCATION_ID="c2", MESSAGE="registry-status.service: Failed with result 'exit-code'."))
    lines += _journal("registry-status", "c3", t0 + 200, ["rendering"])
    lines.append(dict(systemd, __REALTIME_TIMESTAMP=us(230), USER_INVOCATION_ID="c3", MESSAGE="registry-status.service: Failed with result unknown."))
    passes = T.passes_from(lines)["registry-status"]
    assert [p["outcome"] for p in passes] == ["timeout", "exit-code", "failed"] and [p["duration_s"] for p in passes] == [30.0, 30.0, 30.0]


def test_status_footer_series_consecutive_budget_cadence_and_freshness(tmp_path):
    S = _load("status_footer")
    now = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)

    def line(minutes_ago, total, cached=False):
        built = now.timestamp() - minutes_ago * 60
        b = datetime.fromtimestamp(built, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        old = datetime.fromtimestamp(built - 3600, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        return {"built_utc": b, "total_ms": total, "heavy_set": ["item_conflicts"] if cached else [],
                "sections": {"in_flight": {"ms": 1, "as_of": b, "from_cache": False, "state": "live"},
                             "item_conflicts": {"ms": 2, "as_of": old if cached else b, "from_cache": cached, "state": "cached" if cached else "live"},
                             "refused": {"ms": 0, "as_of": None, "from_cache": True, "state": "no_cache"}}}
    lines = [line(25 * 60, 50000), line(70, 130000), line(60, 100000), line(50, 90000), line(40, 80000), line(30, 70000), line(20, 60000, cached=True),
             line(5, 55000, cached=True)]
    rep = S.analyze(lines, hours=24, budget_ms=120000, cadence_s=600, now=now)
    assert rep["renders_in_window"] == 7 and rep["path_lines"] == 8            # the 25-h-old line is outside the window
    assert rep["totals_ms"]["n_over_budget"] == 1 and rep["totals_ms"]["max"] == 130000
    assert rep["consecutive_within_budget"] == {"longest": 6, "current": 6, "at_least_5": True}
    assert rep["cadence"]["n_gaps"] == 6 and rep["cadence"]["max_gap_s"] == 900 and rep["cadence"]["n_gaps_over"] == 1 and rep["cadence"]["every_gap_within"] is False
    assert rep["cadence"]["age_of_last_s"] == 300
    assert rep["latest"]["from_cache"] == ["item_conflicts", "refused"] and rep["latest"]["no_cache"] == ["refused"] and rep["latest"]["heavy_set"] == ["item_conflicts"]
    assert rep["freshness"]["item_conflicts"]["as_of_age_s"] == 3600 and rep["freshness"]["in_flight"]["as_of_age_s"] == 0
    assert rep["latest"]["max_as_of_age_s"] == 3600
    path = tmp_path / "status_render.jsonl"
    path.write_text("".join(json.dumps(l) + "\n" for l in lines))
    r = subprocess.run([sys.executable, os.path.join(MEASURE, "status_footer.py"), "--path", str(path), "--hours", "100000", "--json"], capture_output=True, text=True)
    assert r.returncode == 0 and json.loads(r.stdout)["renders_in_window"] == 8
    r = subprocess.run([sys.executable, os.path.join(MEASURE, "status_footer.py"), "--path", str(path), "--hours", "100000"], capture_output=True, text=True)
    assert "consecutive within budget: longest 6" in r.stdout


SAR_TEXT = """Linux 6.17.0-1022-gcp (TESTHOST) 	2026-09-28 	_x86_64_	(8 CPU)

00:00:00      runq-sz  plist-sz   ldavg-1   ldavg-5  ldavg-15   blocked
00:10:00            1       973      3.73      3.88      3.77         2
00:20:00            0       954      6.28      5.35      4.52         4
00:30:00            0       935      3.70      4.72      4.70         2
12:40:00            4      1094      7.90      9.24      9.76         2
12:50:00            7      1019      7.60      6.92      8.19         2
Average:            3       994      5.78      5.83      5.79         3
"""


def test_loadavg_parses_sar_iso_output_filters_the_window_and_reports_the_median(tmp_path):
    L = _load("loadavg")
    samples = L.parse_sar(SAR_TEXT)
    assert len(samples) == 5 and samples[0][0] == datetime(2026, 9, 28, 0, 10) and samples[0][1] == 3.73 and samples[-1][4] == 7
    rep = L.analyze(samples, hours=24, now=datetime(2026, 9, 28, 13, 0), threshold=4.0)
    assert rep["samples"] == 5 and rep["ldavg_1"]["median"] == 6.28 and rep["ldavg_1"]["max"] == 7.9 and rep["ldavg_1"]["share_over_threshold"] == 0.6
    assert rep["interval_s"] == 600
    rep2 = L.analyze(samples, hours=1, now=datetime(2026, 9, 28, 13, 0))
    assert rep2["samples"] == 2 and rep2["ldavg_1"]["median"] == 7.75
    # the mm/dd/yy header (sar without S_TIME_FORMAT=ISO) parses too
    assert L.parse_sar(SAR_TEXT.replace("2026-09-28", "09/28/26"))[0][0] == datetime(2026, 9, 28, 0, 10)
    p = tmp_path / "sar.txt"
    p.write_text(SAR_TEXT)
    r = subprocess.run([sys.executable, os.path.join(MEASURE, "loadavg.py"), "--from-text", str(p), "--now", "2026-09-28T13:00:00", "--json"], capture_output=True, text=True)
    assert r.returncode == 0 and json.loads(r.stdout)["ldavg_1"]["median"] == 6.28
    r = subprocess.run([sys.executable, os.path.join(MEASURE, "loadavg.py"), "--from-text", str(p), "--now", "2026-09-28T13:00:00"], capture_output=True, text=True)
    assert "median 6.28" in r.stdout


def test_windows_receipt_groups_all_whole_by_every_adapter_whole_and_counts_standing_refusals():
    """`receipt_all_read_whole` grouped a receipt by `refused_n == 0`, so a window whose
    receipt left an adapter standing unread (or read it in part without a refusal) counted as all-whole. The group now requires every adapter
    in the receipt read whole; a receipt with no refusal and not every adapter whole is `receipt_partial`; one whose only non-whole results
    are STANDING refusals (re-confirmed, recorded, never an abort) is `receipt_standing_only`; with a refusal, `receipt_with_refusal`.
    The four partition the windows with a receipt. `receipt_columns` carries `standing_refusals_n` (the contract's one shape: the entry's
    `standing_refusals` list, state `standing_refusal`; a string under `refused_standing` is an id left standing unread, not a refusal, and a
    dict there is not a standing refusal either) and `not_whole_n`."""
    W = _load("windows")

    def standing(group, refusal):
        return {"group": group, "state": "standing_refusal", "refusal": refusal, "prior_refusal": {"window": "W29", "utc": "2026-10-06T12:00:00.000Z", "digest": "d" * 16}}

    whole = {"pins": {}, "published": {}, "shards_total": 2, "shards_ok": 2, "revisions_pulled": ["a/r1"], "reactivated": [], "refused": None, "read_whole": True}
    unread = dict(whole, refused_standing=["revS"], read_whole=False)                                   # left standing unread
    partial = dict(whole, shards_ok=1, read_whole=False)
    refused = dict(whole, shards_ok=1, refused=["pull: 1 shard(s) refused"], read_whole=False)
    # the cluster read writes a standing-only adapter `read_whole: true`, `refused` None (so the tick leaves it alone): the LIST decides
    standing_a = dict(whole, standing_refusals=[standing("g1", "sha mismatch")], read_whole=True)
    standing_b = dict(whole, standing_refusals=[standing("g2", "no fit model"), {"group": "g3", "state": "accepted"}], read_whole=True)
    standing_c = dict(whole, standing_refusals=[standing("g4", "unchanged")], read_whole=True, refused=[])
    assert W.standing_refusals_n(dict(whole, refused_standing=[{"group": "g9", "state": "standing_refusal"}])) == 0     # not the H2 key
    cols = {name: W.receipt_columns(_c12_receipt(name, "2026-10-10T10:02:40.000Z", adapters))
            for name, adapters in {"all_whole": {"a": whole, "b": whole}, "unread": {"a": whole, "b": unread}, "partial": {"a": whole, "b": partial},
                                   "refusal": {"a": whole, "b": refused}, "standing": {"a": whole, "b": standing_a, "c": standing_b, "d": standing_c},
                                   "standing_and_partial": {"a": standing_a, "b": partial}}.items()}
    assert (cols["all_whole"]["read_whole_n"], cols["all_whole"]["refused_n"], cols["all_whole"]["standing_refusals_n"], cols["all_whole"]["not_whole_n"]) == (2, 0, 0, 0)
    assert (cols["unread"]["read_whole_n"], cols["unread"]["refused_n"], cols["unread"]["standing_refusals_n"], cols["unread"]["not_whole_n"]) == (1, 0, 0, 1)
    assert (cols["partial"]["read_whole_n"], cols["partial"]["refused_n"], cols["partial"]["not_whole_n"]) == (1, 0, 1)
    assert (cols["refusal"]["read_whole_n"], cols["refusal"]["refused_n"], cols["refusal"]["not_whole_n"]) == (1, 1, 0)
    assert (cols["standing"]["read_whole_n"], cols["standing"]["refused_n"], cols["standing"]["standing_refusals_n"], cols["standing"]["not_whole_n"]) == (4, 0, 3, 0)
    assert cols["standing"]["whole_or_standing_n"] == 4 and cols["all_whole"]["whole_or_standing_n"] == 2 and cols["unread"]["whole_or_standing_n"] == 1
    assert (cols["standing_and_partial"]["standing_refusals_n"], cols["standing_and_partial"]["not_whole_n"], cols["standing_and_partial"]["read_whole_n"]) == (1, 1, 1)
    assert W.receipt_columns(None)["standing_refusals_n"] is None and W.receipt_columns(None)["not_whole_n"] is None and W.receipt_columns(None)["whole_or_standing_n"] is None
    # a standing-only receipt under the OLDER shape (read_whole false beside the list) groups the same way: the list decides
    old_shape = W.receipt_columns(_c12_receipt("old", "2026-10-10T10:02:40.000Z", {"a": whole, "b": dict(standing_a, read_whole=False)}))
    assert (old_shape["read_whole_n"], old_shape["standing_refusals_n"], old_shape["whole_or_standing_n"], old_shape["not_whole_n"]) == (1, 1, 2, 0)
    windows = [dict(c, completed=True, total_s=60.0 * (i + 1), cluster_read=True, changed_set_n=1) for i, c in enumerate(cols.values())]
    sp = W.subpopulations(windows)
    assert (sp["receipt_all_read_whole"]["n"], sp["receipt_with_refusal"]["n"], sp["receipt_standing_only"]["n"], sp["receipt_partial"]["n"]) == (1, 1, 1, 3)
    assert sp["receipt_all_read_whole"]["median_min"] == 1.0 and sp["receipt_standing_only"]["median_min"] == 5.0
    assert sum(sp[k]["n"] for k in ("receipt_all_read_whole", "receipt_with_refusal", "receipt_standing_only", "receipt_partial")) == len(windows)
    rows = W.window_rows([dict(w, start="2026-10-10T10:00:00Z", base=None, adapters="a b", stages=[], end_definition="rev3") for w in windows])
    assert rows[4]["standing_refusals_n"] == 3 and rows[4]["receipt_adapters_n"] == 4 and rows[4]["whole_or_standing_n"] == 4 and rows[1]["not_whole_n"] == 1
    # the partition under the old shape too: standing-only, never all-whole
    sp_old = W.subpopulations([dict(old_shape, completed=True, total_s=60.0, cluster_read=True, changed_set_n=1)])
    assert (sp_old["receipt_all_read_whole"]["n"], sp_old["receipt_standing_only"]["n"], sp_old["receipt_partial"]["n"]) == (0, 1, 0)


def test_windows_an_adapter_left_standing_unread_beside_a_standing_refusal_is_partial_never_standing_only(tmp_path):
    """An adapter's receipt entry can carry BOTH a re-confirmed standing refusal (`standing_refusals`) and a revision the pull left
    standing UNREAD (`refused_standing`). The list alone used to count the adapter whole-or-standing, so the window read
    `not_whole_n = 0` and grouped `receipt_standing_only`: an incomplete read presented as a standing-only outcome in the acceptance
    summary. An adapter is whole-or-standing only with nothing left unread and every shard ok (the status page's precedence); the
    standing list is the ONLY non-whole condition of a standing-only receipt. A receipt that claims `read_whole` beside unread ids reads
    the same, a shard missing beside a standing refusal the same; a standing-only receipt with nothing left unread groups as before,
    under both receipt shapes; the receipt column of the listing names the unread count."""
    W = _load("windows")

    def standing(group, refusal):
        return {"group": group, "state": "standing_refusal", "refusal": refusal, "prior_refusal": {"window": "W29", "utc": "2026-10-06T12:00:00.000Z", "digest": "d" * 16}}

    whole = {"pins": {}, "published": {}, "shards_total": 2, "shards_ok": 2, "revisions_pulled": ["a/r1"], "reactivated": [], "refused": None, "read_whole": True}
    standing_only = dict(whole, standing_refusals=[standing("g1", "sha mismatch")], read_whole=True)
    standing_and_unread = dict(whole, standing_refusals=[standing("g1", "sha mismatch")], refused_standing=["revision-unread"], read_whole=False)
    claims_whole = dict(standing_and_unread, read_whole=True)                       # a receipt claiming whole beside the unread ids: still not whole
    standing_and_shard_missing = dict(whole, standing_refusals=[standing("g1", "sha mismatch")], shards_ok=1, read_whole=False)
    assert W.standing_unread_n(standing_and_unread) == 1 and W.standing_unread_n(standing_only) == 0 and W.standing_unread_n(whole) == 0
    assert W.shards_missing(standing_and_shard_missing) and not W.shards_missing(whole) and not W.shards_missing(dict(whole, shards_total=0, shards_ok=0))
    utc = "2026-10-10T10:02:40.000Z"
    cols = {name: W.receipt_columns(_c12_receipt(name, utc, adapters))
            for name, adapters in {"standing_only": {"a": whole, "b": standing_only}, "standing_and_unread": {"a": whole, "b": standing_and_unread},
                                   "claims_whole": {"a": whole, "b": claims_whole}, "standing_and_shard_missing": {"a": whole, "b": standing_and_shard_missing},
                                   "old_shape_standing_only": {"a": whole, "b": dict(standing_only, read_whole=False)}}.items()}

    def counts(c):
        return (c["receipt_adapters_n"], c["read_whole_n"], c["refused_n"], c["standing_refusals_n"], c["standing_unread_n"], c["whole_or_standing_n"], c["not_whole_n"])
    assert counts(cols["standing_and_unread"]) == (2, 1, 0, 1, 1, 1, 1)          # the finding's receipt: not_whole_n 1, never 0
    assert counts(cols["claims_whole"]) == (2, 2, 0, 1, 1, 1, 1)
    assert counts(cols["standing_and_shard_missing"]) == (2, 1, 0, 1, 0, 1, 1)
    assert counts(cols["standing_only"]) == (2, 2, 0, 1, 0, 2, 0)                   # retained: nothing left unread, standing-only
    assert counts(cols["old_shape_standing_only"]) == (2, 1, 0, 1, 0, 2, 0)
    assert W.receipt_columns(None)["standing_unread_n"] is None
    windows = [dict(c, completed=True, total_s=60.0 * (i + 1), cluster_read=True, changed_set_n=1) for i, c in enumerate(cols.values())]
    sp = W.subpopulations(windows)
    assert (sp["receipt_all_read_whole"]["n"], sp["receipt_with_refusal"]["n"], sp["receipt_standing_only"]["n"], sp["receipt_partial"]["n"]) == (0, 0, 2, 3)
    assert sum(sp[k]["n"] for k in ("receipt_all_read_whole", "receipt_with_refusal", "receipt_standing_only", "receipt_partial")) == len(windows)
    one = W.subpopulations([dict(cols["standing_and_unread"], completed=True, total_s=60.0, cluster_read=True, changed_set_n=1)])
    assert (one["receipt_partial"]["n"], one["receipt_standing_only"]["n"], one["receipt_all_read_whole"]["n"]) == (1, 0, 0)
    rows = W.window_rows([dict(w, start="2026-10-10T10:00:00Z", base=None, adapters="a b", stages=[], end_definition="rev3") for w in windows])
    assert rows[1]["standing_unread_n"] == 1 and rows[1]["not_whole_n"] == 1 and rows[0]["standing_unread_n"] == 0
    # the listing's receipt column says so: whole/refused, the standing count, the unread count
    log = tmp_path / "chain.log"
    log.write_text(PARTIAL_LOG, encoding="utf-8")
    rd = tmp_path / "receipts"
    rd.mkdir()
    (rd / "P1.json").write_text(json.dumps(_c12_receipt("P1", "2026-10-10T10:02:40.000Z", {"a": whole, "b": standing_and_unread})), encoding="utf-8")
    r = subprocess.run([sys.executable, os.path.join(MEASURE, "windows.py"), str(log), "--receipts", str(rd), "--root", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0 and "| 1/0 +1 standing +1 unread |" in r.stdout, r.stdout + r.stderr
    r = subprocess.run([sys.executable, os.path.join(MEASURE, "windows.py"), str(log), "--receipts", str(rd), "--root", str(tmp_path), "--json"], capture_output=True, text=True)
    out = json.loads(r.stdout)
    assert out["windows"][0]["standing_unread_n"] == 1 and out["windows"][0]["not_whole_n"] == 1 and out["windows"][0]["whole_or_standing_n"] == 1
