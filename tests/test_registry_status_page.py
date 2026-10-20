"""Status page: renders from the database; the page is an artifact row FIRST; banned-word
free; carries the build stamp and the disk-guard banner; every section query is named and timed.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import status
from sisteer.registry import synth
from sisteer.registry import reads_protocol as RP


def _setup(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=2, n_items=5)
    H.ingest_once(root, workers=0, min_free=0)
    return root


def test_renders_and_registers_artifact_first(tmp_path):
    root = _setup(tmp_path)
    out = str(tmp_path / "figs" / "index.html")
    path, timings = status.render(root, out_path=out)
    assert path == out and os.path.exists(out)
    html = open(out, encoding="utf-8").read()
    assert "Results registry" in html and "Built 20" in html
    assert set(timings) == set(status.QUERIES)
    # the artifact record was written to the spool BEFORE the page; ingesting it yields the row with the page's sha
    import hashlib
    sha = hashlib.sha256(html.encode("utf-8")).hexdigest()
    pub = os.path.join(root, "records", "publish")
    assert any(f.startswith("artifact-") for _, _, fs in os.walk(pub) for f in fs)
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT kind, uri_primary FROM artifact WHERE artifact_id = ?", (sha,)).fetchone()
    assert row is not None and row[0] == "page" and row[1] == os.path.abspath(out)
    conn.close()


def test_sections_present_and_banned_words_absent(tmp_path, monkeypatch):
    """The page renders every section and carries no banned word. The cluster-reads section reads the landing state directory
    (``$REGISTRY_LANDING_STATE``; on the host that runs the landing chain its default holds live receipts naming store paths), so the
    test renders from an EMPTY per-test state directory whatever the environment set -- the assertion is over this page alone."""
    monkeypatch.setenv(RP.STATE_ENV, str(tmp_path / "landing-state"))
    root = _setup(tmp_path)
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)
    html = open(out, encoding="utf-8").read()
    for heading in ("Runs in flight", "Roster gaps", "Refused records", "Blocked records", "Recent ingests", "Superseded readings",
                    "Snapshot freshness", "Mirror acknowledgements", "Adapter pass state", "Decode-policy mismatches",
                    "Instrument mismatches", "Analyses (filed over ingested runs and objects)", "Row counts", "Query timings", "DEPRECATED"):
        assert heading in html, heading
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\b\w*_legacy\b", " ", text)
    for word in ("arm", "leg", "lane", "cell", "wave", "grader"):
        assert not re.search(r"(?<![A-Za-z0-9_])" + word + r"(?![A-Za-z0-9_])", text, re.IGNORECASE), word


def test_paused_banner_and_json(tmp_path, monkeypatch):
    root = _setup(tmp_path)
    monkeypatch.setattr(H, "free_bytes", lambda path: 1024 ** 3)      # a controlled reading: 1 GB free, whatever the host has
    H.ingest_once(root, workers=0, min_free=2 * 1024 ** 3)             # trips the guard
    ctx = status.status_json(root)
    assert ctx["paused"] is not None and ctx["table_counts"]["run"] == 2
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)
    assert "Ingest paused" in open(out, encoding="utf-8").read()


def test_blocked_and_refused_sections_list_rows(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    from sisteer.registry import records as R
    bad = synth.rec("run_state", {"run_id": "orphan", "state": "running", "event_utc": synth.t(0), "evidence_kind": "job_log"}, synth.t(0))
    p = synth.run_payload("k")
    p["serving_path"] = "vllm"
    R.write_records(os.path.join(root, "records", "x", "000001-x.jsonl"), [bad, synth.rec("run", p, synth.t(1))])
    H.ingest_once(root, workers=0, min_free=0)
    ctx = status.status_json(root)
    assert len(ctx["blocked"]) == 1 and ctx["blocked"][0]["missing_parent_table"] == "run"
    assert len(ctx["refused"]) == 1 and ctx["refused"][0]["field"] == "serving_path"
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)
    html = open(out, encoding="utf-8").read()
    assert "orphan" in html and "serving_path" in html


def test_stamp_advances_between_renders(tmp_path):
    root = _setup(tmp_path)
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)
    a = re.search(r"Built (\S+)", open(out).read()).group(1)
    import time
    time.sleep(0.01)
    status.render(root, out_path=out, register=False)
    b = re.search(r"Built (\S+)", open(out).read()).group(1)
    assert b >= a


def test_ingest_html_does_not_re_render_on_its_own_page_artifact(tmp_path, monkeypatch, capsys):
    """``registry ingest --html`` re-renders when a cycle changed something — but the render itself registers the page
    as an ``artifact`` record, which the NEXT cycle ingests. Counting that as a change re-rendered, re-registered, and
    looped: the live install wrote one page artifact every 20 s (2026-09-17). A cycle whose only applied record is the
    previous render's own page is ``self_change_only`` and renders nothing; a real record still re-renders."""
    from sisteer.registry import cli
    from sisteer.registry import records as R
    root = _setup(tmp_path)
    out = str(tmp_path / "figs" / "index.html")
    monkeypatch.setenv("REGISTRY_ROOT", root)
    monkeypatch.setenv("REGISTRY_STATUS_HTML", out)

    def page_records():
        pub = os.path.join(root, "records", "publish")
        return sorted(f for _, _, fs in os.walk(pub) for f in fs if f.startswith("artifact-"))

    status.render(root, out_path=out)                       # the timer's render: one page artifact in the spool
    assert len(page_records()) == 1
    rep = H.ingest_once(root, workers=0, min_free=0)         # cycle 1 folds it: a change, but a self-inflicted one
    # the page's record plus the arrival fact the fold wrote for the page's file: both bookkeeping, no render
    assert rep.ok == 2 and rep.status_page_records == 1 and rep.arrival_records == 1 and rep.changed and rep.self_change_only
    status.render(root, out_path=out)                       # a second page → the CLI's cycle must NOT render a third
    assert cli.main(["ingest", "--html", "--workers", "0"]) == 0
    printed = capsys.readouterr().out
    assert json.loads(printed[: printed.index("}") + 1])["self_change_only"] is True
    assert len(page_records()) == 2 and out not in printed.splitlines()[-1]
    assert cli.main(["ingest", "--html", "--workers", "0"]) == 0          # nothing new: no change, no render
    assert len(page_records()) == 2
    # a real record arriving is a substantive change: the page renders and registers itself once more
    R.write_records(os.path.join(root, "records", "late", "000001-late.jsonl"),
                    [synth.rec("si_text", {"si_id": "late", "si_text_sha256": "e" * 64, "si_kind": "irrelevant_control",
                                           "source_uri": "synthetic://x"}, synth.t(7000))])
    assert cli.main(["ingest", "--html", "--workers", "0"]) == 0
    printed = capsys.readouterr().out
    assert printed.strip().splitlines()[-1] == out and len(page_records()) == 3
    rep = H.ingest_once(root, workers=0, min_free=0)         # and the cycle after folds that page: self-only again
    assert rep.status_page_records == 1 and rep.self_change_only


# --------------------------------------------------------------------------------------------
# Rendered within budget, content kept (registry-slurm-fast-windows Phase 1, S4): the per-section budget, the last-good cache,
# the dynamic cached-with-refresh set, the render history, `status --refresh-heavy`
# --------------------------------------------------------------------------------------------


def test_in_budget_render_writes_every_section_cache_and_one_history_line(tmp_path):
    root = _setup(tmp_path)
    out = str(tmp_path / "index.html")
    _, timings = status.render(root, out_path=out, register=False)
    assert set(timings) == set(status.QUERIES)
    assert sorted(f[:-5] for f in os.listdir(status.sections_dir(root))) == sorted(status.QUERIES)
    hist = status.read_render_history(root)
    assert len(hist) == 1 and set(hist[0]["sections"]) == set(status.QUERIES)
    assert hist[0]["total_ms"] > 0 and not any(s["from_cache"] for s in hist[0]["sections"].values()) and hist[0]["heavy_set"] == []
    assert status.read_heavy_state(root)["members"] == []
    html = open(out, encoding="utf-8").read()
    assert "total_ms" in html and "from_cache" in html and "Last good rows" not in html


def test_over_budget_section_renders_cached_rows_with_the_mark_and_joins_the_set(tmp_path):
    root = _setup(tmp_path)
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)                      # caches every section
    cached_as_of = status.read_section_cache(root, "refused")["as_of"]
    status.render(root, out_path=out, register=False, budget_overrides={"refused": 0})
    html = open(out, encoding="utf-8").read()
    assert f"Last good rows as of <strong>{cached_as_of}</strong>; skipped this render: over its 0 s budget" in html
    assert status.read_heavy_state(root)["members"] == ["refused"]
    hist = status.read_render_history(root)
    assert hist[-1]["sections"]["refused"] == {"ms": hist[-1]["sections"]["refused"]["ms"], "as_of": cached_as_of, "from_cache": True, "state": "cached"}
    assert hist[-1]["heavy_set"] == ["refused"]
    # the other sections ran live: their as_of is this render's, not the cache's
    assert not hist[-1]["sections"]["blocked"]["from_cache"] and hist[-1]["sections"]["blocked"]["as_of"] > cached_as_of
    # a later in-budget render clears the member (no fixed list; the set follows the measurements)
    status.render(root, out_path=out, register=False)
    assert status.read_heavy_state(root)["members"] == []
    assert "Last good rows" not in open(out, encoding="utf-8").read()


def test_section_with_no_cache_says_so_never_an_empty_table_that_looks_complete(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    from sisteer.registry import records as R
    bad = synth.rec("run_state", {"run_id": "orphan", "state": "running", "event_utc": synth.t(0), "evidence_kind": "job_log"}, synth.t(0))
    R.write_records(os.path.join(root, "records", "x", "000001-x.jsonl"), [bad])
    H.ingest_once(root, workers=0, min_free=0)
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False, budget_overrides={"blocked": 0})       # a blocked row exists, never computed
    html = open(out, encoding="utf-8").read()
    assert "Not computed this render and no cached rows yet" in html and "orphan" not in html
    ctx = json.loads(open(os.path.join(root, "state", "status_render.jsonl")).read().splitlines()[-1])
    assert ctx["sections"]["blocked"] == {"ms": ctx["sections"]["blocked"]["ms"], "as_of": None, "from_cache": True, "state": "no_cache"}
    assert status.read_heavy_state(root)["members"] == ["blocked"]


def test_table_counts_exact_for_cheap_tables_cached_with_stamp_for_the_rest(tmp_path, monkeypatch):
    root = _setup(tmp_path)
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)
    cache = status.read_section_cache(root, "table_counts")
    assert cache["counts"]["run"] == 2 and cache["exact"] is True and set(cache["counts_as_of"]) == set(cache["counts"])
    with DB.reading(DB.db_path(root)) as conn:
        ctx, _ = status.gather(conn, root, budget_overrides={"table_counts": 0})
    assert ctx["table_counts"]["run"] == 2 and ctx["table_counts_as_of"]["run"] == cache["counts_as_of"]["run"]
    assert len(ctx["table_counts_cached"]) == len(ctx["table_counts"]) and ctx["section_state"]["table_counts"]["state"] == "cached"
    assert status.read_heavy_state(root)["members"] == ["table_counts"]
    status.render(root, out_path=out, register=False, budget_overrides={"table_counts": 0})
    html = open(out, encoding="utf-8").read()
    assert f"<td>run</td><td>2</td><td>{cache['counts_as_of']['run']}</td>" in html
    # the section's own mark names how many tables came from the cache
    assert "skipped this render for" in html and "tables: over its 0 s budget" in html
    # cheapest-by-last-count first: the counts run in the cache's ascending order (the uncached tables last), so a heavy
    # table never starves a cheap one of the section budget
    seen = []
    monkeypatch.setattr(status, "_count_within", lambda conn, table, deadline: seen.append(table) or 0)
    with DB.reading(DB.db_path(root)) as conn:
        status._table_counts_within(conn, cache, time.monotonic() + 60, time.monotonic() + 60)
    cached_seen = [t for t in seen if t in cache["counts"]]
    assert cached_seen == seen[:len(cached_seen)] and [cache["counts"][t] for t in cached_seen] == sorted(cache["counts"].values())


def test_render_budget_exhausted_serves_from_cache_without_joining_the_set(tmp_path):
    root = _setup(tmp_path)
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)
    with open(os.path.join(root, "config.toml"), "w") as fh:
        fh.write("[status]\nrender_budget_s = 0\n")
    with DB.reading(DB.db_path(root)) as conn:
        ctx, _ = status.gather(conn, root)
    states = {n: s["state"] for n, s in ctx["section_state"].items()}
    assert set(states.values()) == {"cached"} and all(s["from_cache"] for s in ctx["section_state"].values())
    assert all("render budget was exhausted" in s["reason"] for s in ctx["section_state"].values())
    assert status.read_heavy_state(root)["members"] == [], "not reached is not heavy"
    assert ctx["table_counts"]["run"] == 2                      # the cached counts, not an empty table


def test_render_history_is_bounded_by_an_atomic_tail_rewrite(tmp_path, monkeypatch):
    """Append-only, bounded to the last RENDER_HISTORY_MAX lines: the file is rewritten to that tail once it holds more than twice
    as many (so the rewrite runs once per bound, not per render), atomically (tmp + rename)."""
    root = _setup(tmp_path)
    monkeypatch.setattr(status, "RENDER_HISTORY_MAX", 4)
    for i in range(8):
        n = status.append_render_history(root, {"i": i})
    assert n == 8 and [l["i"] for l in status.read_render_history(root)] == list(range(8))
    n = status.append_render_history(root, {"i": 8})                       # the 9th line: 9 > 2 × 4 → the last 4 kept
    assert n == 4 and [l["i"] for l in status.read_render_history(root)] == [5, 6, 7, 8]
    status.append_render_history(root, {"i": 9})
    assert [l["i"] for l in status.read_render_history(root)] == [5, 6, 7, 8, 9]
    assert not [f for f in os.listdir(os.path.join(root, "state")) if f.startswith("status_render.jsonl.")], "no tmp left behind"
    # a render appends exactly one line
    before = len(status.read_render_history(root))
    status.render(root, out_path=str(tmp_path / "i.html"), register=False)
    assert len(status.read_render_history(root)) == before + 1


def test_refresh_heavy_initializes_every_cache_then_refreshes_members_gated_on_the_ledger_head(tmp_path, monkeypatch, capsys):
    from sisteer.registry import cli
    from sisteer.registry import records as R
    root = _setup(tmp_path)
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert not os.path.isdir(status.sections_dir(root))
    # first run: uninitialized → every section computed exactly and cached; the set derived from the measured times (all cheap here)
    assert cli.main(["status", "--refresh-heavy"]) == 0
    r1 = json.loads(capsys.readouterr().out)
    assert r1["skipped"] is False and r1["initialized_now"] is True and set(r1["results"]) == set(status.QUERIES)
    assert sorted(f[:-5] for f in os.listdir(status.sections_dir(root))) == sorted(status.QUERIES)
    st = status.read_heavy_state(root)
    assert st["initialized"] is True and st["head"] == r1["head"] and st["members"] == [] and st["refreshed_utc"]
    assert status.read_section_cache(root, "table_counts")["source"] == "heavy" and status.read_section_cache(root, "table_counts")["counts"]["run"] == 2
    # the head has not moved: skipped, the state untouched
    assert cli.main(["status", "--refresh-heavy"]) == 0
    r2 = json.loads(capsys.readouterr().out)
    assert r2["skipped"] is True and "ledger head unchanged" in r2["reason"] and status.read_heavy_state(root)["refreshed_utc"] == st["refreshed_utc"]
    # a member joins (a render found it over budget); the head moves (a record folds): the refresh recomputes exactly that member
    status.render(root, out_path=str(tmp_path / "i.html"), register=False, budget_overrides={"item_conflicts": 0})
    assert status.read_heavy_state(root)["members"] == ["item_conflicts"]
    R.write_records(os.path.join(root, "records", "late", "000001-late.jsonl"),
                    [synth.rec("si_text", {"si_id": "late", "si_text_sha256": "e" * 64, "si_kind": "irrelevant_control",
                                           "source_uri": "synthetic://x"}, synth.t(7000))])
    H.ingest_once(root, workers=0, min_free=0)
    before = status.read_section_cache(root, "refused")["as_of"]
    assert cli.main(["status", "--refresh-heavy"]) == 0
    r3 = json.loads(capsys.readouterr().out)
    assert r3["skipped"] is False and list(r3["results"]) == ["item_conflicts"] and r3["head"] > r1["head"]
    assert r3["cleared"] == ["item_conflicts"] and r3["members"] == [], "a member that came in under the section budget is cleared"
    assert status.read_section_cache(root, "item_conflicts")["source"] == "heavy"
    assert status.read_section_cache(root, "refused")["as_of"] == before, "only the members are recomputed"
    # --force refreshes with the head unchanged
    assert cli.main(["status", "--refresh-heavy", "--force"]) == 0
    assert json.loads(capsys.readouterr().out)["skipped"] is False


def test_refresh_heavy_keeps_the_previous_cache_when_a_member_exceeds_the_heavy_budget(tmp_path):
    root = _setup(tmp_path)
    status.render(root, out_path=str(tmp_path / "i.html"), register=False)
    status.update_heavy_state(root, add=["item_conflicts"], initialized=True)
    before = status.read_section_cache(root, "item_conflicts")
    with open(os.path.join(root, "config.toml"), "w") as fh:
        fh.write("[status]\nheavy_budget_s = 0\n")
    out = status.refresh_heavy(root, force=True)
    assert out["results"]["item_conflicts"]["ok"] is False and "interrupted" in out["results"]["item_conflicts"]["error"]
    assert out["members"] == ["item_conflicts"] and status.read_section_cache(root, "item_conflicts") == before


def test_gather_budgets_are_config_toml_s(tmp_path):
    """The render's budgets come from `[status]` in config.toml (the defaults and a partial block: test_registry_ops_config.py)."""
    root = _setup(tmp_path)
    with open(os.path.join(root, "config.toml"), "w") as fh:
        fh.write("[status]\nsection_budget_s = 5\nrender_budget_s = 30\n")
    with DB.reading(DB.db_path(root)) as conn:
        ctx, _ = status.gather(conn, root)
    assert ctx["budgets"] == {"section_budget_s": 5.0, "render_budget_s": 30.0, "heavy_budget_s": 900.0}


def test_count_within_is_interrupted_at_the_production_handler_interval():
    """An unrestricted `SELECT COUNT(*)` compiles to SQLite's single `Count` opcode (~9 VM steps in all), so the progress handler
    (every PROGRESS_HANDLER_STEPS steps) never ran and an expired deadline was ignored: 100,000 rows came back under a deadline
    already past. The count is a row loop the handler paces (`COUNT(1)`: Rewind / AggStep / Next), and a deadline already past
    is refused before the statement runs — at the PRODUCTION interval, on a table large enough that the deadline expires
    mid-count."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row                                             # the registry's connections read rows as mappings
    conn.execute("CREATE TABLE big (x INTEGER)")
    conn.execute("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 1000000) INSERT INTO big SELECT x FROM c")
    assert status.PROGRESS_HANDLER_STEPS == 10_000, "the production interval, not a test knob"
    opcodes = [r[1] for r in conn.execute("EXPLAIN SELECT COUNT(1) AS n FROM big")]
    assert "Count" not in opcodes and "Next" in opcodes, opcodes          # a row loop, not the Count opcode: the handler runs
    assert status._count_within(conn, "big", time.monotonic() + 60) == 1_000_000
    with pytest.raises(status.SectionOverBudget):                          # already expired: refused before the statement runs
        status._count_within(conn, "big", time.monotonic() - 1)
    with pytest.raises(status.SectionOverBudget):                          # expires mid-count (~40 ms of counting): the handler interrupts
        status._count_within(conn, "big", time.monotonic() + 0.005)
    assert status._count_within(conn, "big", time.monotonic() + 60) == 1_000_000   # the connection is whole after the interrupt
    assert status._count_within(conn, "no_such_table", time.monotonic() + 60) is None


def test_a_failed_query_is_never_the_last_good_rows_and_the_previous_cache_serves_later(tmp_path, monkeypatch):
    """A SQLite error (not the budget) shows its error row in place, marked `error`; the cache keeps the previous good rows and the
    set does not move (no timing was measured); the heavy refresh refuses it the same way; a later render that skips the query
    serves the previous good rows."""
    root = _setup(tmp_path)
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)
    good = status.read_section_cache(root, "refused")
    good_sql = status.QUERIES["refused"]
    monkeypatch.setitem(status.QUERIES, "refused", "SELECT * FROM no_such_table_for_the_test")
    status.render(root, out_path=out, register=False)
    assert status.read_section_cache(root, "refused") == good, "an error row never becomes the last good rows"
    st = status.read_render_history(root)[-1]["sections"]["refused"]
    assert st["state"] == "error" and st["from_cache"] is False and st["as_of"] is None
    html = open(out, encoding="utf-8").read()
    assert "Query failed this render" in html and "no_such_table_for_the_test" in html and "Last good rows" not in html
    assert status.read_heavy_state(root)["members"] == [], "a failed query is no timing measurement"
    rep = status.refresh_heavy(root, force=True)
    assert rep["results"]["refused"]["ok"] is False and "no_such_table_for_the_test" in rep["results"]["refused"]["error"]
    assert status.read_section_cache(root, "refused") == good and rep["members"] == []
    assert status.read_section_cache(root, "blocked")["source"] == "heavy", "the other sections were refreshed"
    monkeypatch.setitem(status.QUERIES, "refused", good_sql)
    status.render(root, out_path=out, register=False, budget_overrides={"refused": 0})
    assert f"Last good rows as of <strong>{good['as_of']}</strong>" in open(out, encoding="utf-8").read()


def test_pending_count_shows_the_query_error_not_a_healthy_zero(tmp_path, monkeypatch):
    root = _setup(tmp_path)
    with DB.reading(DB.db_path(root)) as conn:
        assert status.gather(conn, root, persist=False)[0]["pending_count"] == 0
    monkeypatch.setitem(status.QUERIES, "pending", "SELECT COUNT(*) AS n FROM no_such_table_for_the_test")
    with DB.reading(DB.db_path(root)) as conn:
        ctx, _ = status.gather(conn, root, persist=False)
    assert "no_such_table_for_the_test" in str(ctx["pending_count"]) and ctx["pending_count"] != 0
    assert "no_such_table_for_the_test" in status.render_html(ctx)


def test_status_json_is_a_read_that_writes_no_cache_and_moves_no_set(tmp_path, monkeypatch, capsys):
    """`registry status` / `--json` serve the same sections within the same budgets but persist nothing: no section cache, no
    heavy state, no history line — only the render (`--html`) writes them."""
    from sisteer.registry import cli
    root = _setup(tmp_path)
    ctx = status.status_json(root)
    assert ctx["head_seq"] is not None and set(ctx["section_state"]) == set(status.QUERIES) and ctx["table_counts"]["run"] == 2
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["status", "--json"]) == 0 and json.loads(capsys.readouterr().out)["head_seq"] == ctx["head_seq"]
    assert cli.main(["status"]) == 0 and "head ingest_seq" in capsys.readouterr().out
    for p in (status.sections_dir(root), status.heavy_state_path(root), status.render_history_path(root)):
        assert not os.path.exists(p), p
    # a read that finds a section over budget serves it from the cache (none here) but does not make it a member
    with DB.reading(DB.db_path(root)) as conn:
        ctx, _ = status.gather(conn, root, budget_overrides={"refused": 0}, persist=False)
    assert ctx["section_state"]["refused"]["state"] == "no_cache" and not os.path.exists(status.heavy_state_path(root))
    assert status.read_heavy_state(root)["members"] == []
    # the render persists: the cache appears and the member joins
    status.render(root, out_path=str(tmp_path / "i.html"), register=False, budget_overrides={"refused": 0})
    assert os.path.isdir(status.sections_dir(root)) and status.read_heavy_state(root)["members"] == ["refused"]


def test_paper_objects_absent_is_cached_the_same_by_the_render_and_the_heavy_refresh(tmp_path, monkeypatch):
    """While `v_paper_objects` is absent both paths cache `rows: []` for the section, so a later cache-served
    render shows the same empty result the live render did — one rule, not two."""
    root = _setup(tmp_path)
    real = status.sql_name_exists
    monkeypatch.setattr(status, "sql_name_exists", lambda conn, name: False if name == status.PAPER_OBJECTS_NAME else real(conn, name))
    status.render(root, out_path=str(tmp_path / "i.html"), register=False)
    by_render = status.read_section_cache(root, "paper_objects")
    assert by_render["rows"] == [] and by_render["exact"] is True and by_render["source"] == "render"
    status.refresh_heavy(root, force=True)
    by_heavy = status.read_section_cache(root, "paper_objects")
    assert by_heavy["rows"] == [] and by_heavy["exact"] is True and by_heavy["source"] == "heavy"
    assert {k: v for k, v in by_render.items() if k not in ("as_of", "elapsed_ms", "source")} == \
           {k: v for k, v in by_heavy.items() if k not in ("as_of", "elapsed_ms", "source")}


def test_refresh_heavy_retries_a_member_whose_refresh_failed_while_the_ledger_head_stands(tmp_path):
    """A member that times out keeps its old cache, the recorded head does not advance and the member is pending, so the next
    6-hour tick recomputes it although the ledger head has not moved. Before: the current head was recorded regardless and the
    next tick skipped (head unchanged + a cache exists), so a slow count or conflict scan stayed stale through a quiet ledger."""
    from sisteer.registry import records as R
    root = _setup(tmp_path)
    status.render(root, out_path=str(tmp_path / "i.html"), register=False)
    first = status.refresh_heavy(root)
    h0 = first["head"]
    assert first["failed"] == [] and status.read_heavy_state(root)["head"] == h0
    status.update_heavy_state(root, add=["item_conflicts"])
    stale = status.read_section_cache(root, "item_conflicts")
    # phase 1 — the head moves, the member's refresh times out: the head recorded stays h0, the member is pending
    R.write_records(os.path.join(root, "records", "late", "000001-late.jsonl"),
                    [synth.rec("si_text", {"si_id": "late", "si_text_sha256": "e" * 64, "si_kind": "irrelevant_control",
                                           "source_uri": "synthetic://x"}, synth.t(7000))])
    H.ingest_once(root, workers=0, min_free=0)
    with open(os.path.join(root, "config.toml"), "w") as fh:
        fh.write("[status]\nheavy_budget_s = 0\n")
    failed = status.refresh_heavy(root)
    h1 = failed["head"]
    assert h1 > h0 and failed["skipped"] is False and failed["failed"] == ["item_conflicts"] and failed["head_recorded"] == h0
    st = status.read_heavy_state(root)
    assert st["head"] == h0 and st["pending_members"] == ["item_conflicts"] and status.read_section_cache(root, "item_conflicts") == stale
    # phase 2 — the ledger is quiet (head still h1) and the budget is back: the tick is NOT skipped, the member is recomputed
    os.remove(os.path.join(root, "config.toml"))
    retried = status.refresh_heavy(root)
    assert retried["skipped"] is False and list(retried["results"]) == ["item_conflicts"] and retried["failed"] == []
    assert retried["head_recorded"] == h1 and status.read_section_cache(root, "item_conflicts")["source"] == "heavy"
    assert status.read_heavy_state(root)["pending_members"] == []
    # phase 3 — a failure with the head ALREADY recorded at h1 (forced, budget 0): pending, and the next quiet tick recomputes
    # exactly that member through the unchanged-head gate; the tick after it is skipped
    status.update_heavy_state(root, add=["item_conflicts"])
    with open(os.path.join(root, "config.toml"), "w") as fh:
        fh.write("[status]\nheavy_budget_s = 0\n")
    assert status.refresh_heavy(root, force=True)["failed"] == ["item_conflicts"]
    os.remove(os.path.join(root, "config.toml"))
    again = status.refresh_heavy(root)
    assert again["skipped"] is False and list(again["results"]) == ["item_conflicts"] and again["failed"] == []
    skipped = status.refresh_heavy(root)
    assert skipped["skipped"] is True and "every member current" in skipped["reason"]


def test_cluster_reads_section_reads_the_latest_window_plan_and_names_routes_and_refusals(tmp_path):
    """The "Cluster reads" section (registry-slurm-fast-windows Phase 3b) comes from state/cluster_reads/<window>.json -- no database, no
    schema change: routes, shards, reasons, the recorded refusals per adapter (C6: an adapter routed `grading_host` is NOT read; the row says why);
    an empty state says so. A pre-C6 plan file spelled the key `fallback`; the section reads it the same way."""
    import json
    from sisteer.registry import cluster_reads as CR
    root = _setup(tmp_path)
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)
    html = open(out, encoding="utf-8").read()
    assert "Cluster reads" in html and "no cluster-read window recorded yet" in html
    plan = {"schema": CR.PLAN_SCHEMA, "window": "20260928T230000Z", "built_utc": "2026-09-28T23:00:00.000Z", "tree_sha": "a" * 40,
            "adapters": {"relu_paired": {"route": "cluster", "shards": 8, "reason": None, "pins": {"relu_paired_run_state": "aabbccdd"}},
                         "em_misalign": {"route": "grading_host", "shards": 6, "reason": "admission refused component(s): misalign_grades: content_changed (1 entry differs)", "pins": {}}},
            "refusals": {"em_misalign": ["plan: admission refused component(s): misalign_grades: content_changed (1 entry differs)"]},
            "stages": {"submit": {"utc": "2026-09-28T23:05:00.000Z", "job_id": "777001", "tasks": 8}}, "tasks": []}
    os.makedirs(CR.plan_state_dir(root), exist_ok=True)
    with open(CR.plan_path(root, plan["window"]), "w") as fh:
        json.dump(plan, fh)
    status.render(root, out_path=out, register=False)
    html = open(out, encoding="utf-8").read()
    assert "20260928T230000Z" in html and "777001" in html and "relu_paired" in html and "cluster" in html
    assert "misalign_grades: content_changed" in html and "relu_paired_run_state@aabbccdd" in html
    sec = status.gather_operational(root)["cluster_reads"]
    assert sec["present"] and [r["route"] for r in sec["rows"]] == ["grading_host", "cluster"] and sec["job_id"] == "777001"
    assert sec["rows"][0]["refusals"].startswith("plan: admission refused component(s)") and sec["rows"][1]["refusals"] == ""
    # a pre-C6 plan file (the key spelled `fallback`) renders the same row
    old = dict(plan, window="20260928T240000Z", refusals=None)
    old.pop("refusals")
    old["fallback"] = plan["refusals"]
    with open(CR.plan_path(root, old["window"]), "w") as fh:
        json.dump(old, fh)
    sec2 = status.gather_operational(root)["cluster_reads"]
    assert sec2["window"] == old["window"] and sec2["rows"][0]["refusals"] == sec["rows"][0]["refusals"]


# --------------------------------------------------------------------------------------------
# "Cluster reads, store authority and the replica": state files only, no database
# --------------------------------------------------------------------------------------------


def _slurm_only_fixture(root, reads_root, monkeypatch, stale_pid=2 ** 22 - 7):
    """The state files the section reads, in the shapes their writers spell: `stores check-authority`'s store_authority.json (one
    component ok with an overdue carry, one dangling without its record, one unsealed), `stores mirror`'s store_mirror.json (an extra
    component the check never read; a red rsync outcome on the unsealed one), live_db_slurm.sh's replica_check.json (second-resolution
    stamp, UNEQUAL), window.sh's maintenance flag (second-resolution stamp), a config.toml carrying the removed table (the header built
    at run time so this module passes the no-Modal scan it is part of), a STALE reservation (a pid no process has) and two receipts
    (the newer one refusing relu_paired) plus a torn receipt file under $REGISTRY_LANDING_STATE/reads/."""
    import socket
    from sisteer.registry import config as CFG
    from sisteer.registry import reads_protocol as RP
    from sisteer.registry import records as R
    from sisteer.registry import stores as ST
    monkeypatch.setenv(RP.STATE_ENV, reads_root)
    state = os.path.join(root, "state")
    sha40 = "c" * 40
    auth = {"schema": ST.AUTHORITY_STATE_SCHEMA, "checked_utc": "2026-10-05T07:00:00.000Z", "destination": "cluster_a", "stores_dir": "/x/stores",
            "config": "c.yaml", "ok": False, "n_components": 3, "n_problems": 2, "problems": ["bfcl_full: dangling, no_published_record", "misalign_grades: unsealed"],
            "counts": {"ok": 1, "dangling": 1, "unsealed": 1},
            "components": {
                "relu_paired_sidecars": {"authority": "cluster_a", "producer": "grading_host:ReLU", "current": "aabbccdd", "real": "/x/aabbccdd", "exists": True, "verified": True,
                                         "published_record": True, "sealed": True, "change_utc": "2026-10-04T10:00:00.000Z", "published_utc": "2026-10-04T10:05:00.000Z",
                                         "publish_mode": "push", "grading_host_mirror": "/h/specs/relu", "grading_host_mirror_present": True, "transitional_carry": True,
                                         "carry_deadline": "2026-10-01", "carry_overdue": True, "problems": [], "notes": ["carry_overdue"], "state": "ok"},
                "bfcl_full": {"authority": "cluster_a", "producer": "frozen", "current": "644c6365", "real": None, "exists": False, "verified": False, "published_record": False,
                              "sealed": False, "change_utc": None, "published_utc": None, "publish_mode": None, "grading_host_mirror": "/h/bfcl", "grading_host_mirror_present": True,
                              "transitional_carry": False, "carry_deadline": None, "carry_overdue": False, "problems": ["dangling", "no_published_record"], "notes": [],
                              "state": "dangling"},
                "misalign_grades": {"authority": "grading_host_copy", "producer": None, "current": "85febd26", "real": "/x/85", "exists": True, "verified": True, "published_record": True,
                                    "sealed": False, "change_utc": "2026-09-29T00:00:00.000Z", "published_utc": "2026-09-29T00:01:00.000Z", "publish_mode": "mirror",
                                    "grading_host_mirror": "/h/grade", "grading_host_mirror_present": None, "transitional_carry": False, "carry_deadline": None, "carry_overdue": False,
                                    "problems": ["unsealed"], "notes": [], "state": "unsealed"}}}
    R.write_json_atomic(os.path.join(state, ST.AUTHORITY_STATE_FILE), auth, indent=1)
    mirror = {"schema": ST.STATE_SCHEMA, "updated_utc": "2026-10-05T06:54:02.026Z", "destination": "cluster_a",
              "components": {"misalign_grades": {"authority": "grading_host_copy", "generation": {"content_sha8": "85febd26", "state": "copying"},
                                                 "last_run": {"outcome": "rsync_failed", "utc": "2026-10-05T06:53:00.000Z"}},
                             "sampled": {"authority": "grading_host_copy", "generation": {"content_sha8": "0badf00d", "state": "published"},
                                         "last_run": {"outcome": "unchanged", "utc": "2026-10-05T06:53:10.000Z"}},
                             "relu_paired_sidecars": {"authority": "cluster_a", "producer": "grading_host:ReLU", "generation": {"content_sha8": "aabbccdd", "state": "verified"},
                                                      "last_run": {"outcome": "unpushed_changes", "utc": "2026-10-05T06:53:20.000Z"}}}}
    R.write_json_atomic(ST.state_path(root), mirror, indent=1)
    rep = {"schema": "replica_check_state/2", "headline": "unequal", "utc": "2026-10-05T05:00:00Z", "equal": False, "ok": False,
           "problems": ["replica check UNEQUAL: projection differs: {\"lifecycle_state\": 4}", "the job's own verdict is 'FAIL'"],
           "branch": "registry-slurm-only", "tip": sha40, "job": "900001", "job_state": "COMPLETED", "submission": "t", "out_remote": "/o", "replica_tree": None,
           "dispatcher": {"source": "tip", "sha256": None, "path": None},
           "head": {"n_files": 477086, "taken_utc": "2026-10-04T23:33:47.000Z", "allowlist_sha12": "abc", "projection_sha12": "cebbc428aaaa", "head_sha12": "h"},
           "replica": {"projection_sha12": "deadbeefcafe", "counts": {"rows": 12708, "in_current_paper": 3444, "selected": 3563}}, "projection_version": "frame_projection/1",
           "generation": "g2", "current_generation": "g1", "parent_generation": "g1", "schema_version": 24,
           "pytest": {"rc": 0, "counts": {"tests": 9, "skipped": 0}, "expected_skipped": 0, "status": "FAIL"}, "reason": "projection differs"}
    R.write_json_atomic(os.path.join(state, status.REPLICA_CHECK_FILE), rep, indent=1)
    # window.sh's shape: a second-resolution stamp, the pid, the adapters. The stamp is FIVE MINUTES BEFORE NOW, never a literal: the page
    # judges staleness against the real clock (status.MAINTENANCE_STALE_S, 3 h), so a literal stamp made this fixture a clock bomb -- it read
    # amber only within three hours of its authoring and red for ever after (the 2026-10-05 10:00Z battery)
    since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 5 * 60))
    with open(H.maintenance_path(root), "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"since_utc": since, "reason": "landing window (window.sh pid 4242)", "pid": 4242, "adapters": "relu_paired em_misalign"}) + "\n")
    with open(CFG.config_path(root), "w", encoding="utf-8") as fh:
        fh.write("[status]\nsection_budget_s = 7\n[" + CFG.REMOVED_TABLE + "]\nenabled = true\nvolume = \"twosi-fits\"\n")
    rd = RP.reads_dir()
    os.makedirs(os.path.join(rd, RP.RECEIPTS_SUBDIR), exist_ok=True)
    # the holder identity is (owner, op id, host, pid, the pid's /proc start time): a live holder carries its REAL start time (a recorded
    # start that differs from the process's reads as a reused pid, by design -- I3), a dead pid any value
    R.write_json_atomic(RP.reservation_path(rd), {"schema": RP.RESERVATION_SCHEMA, "owner": "tick", "op_id": "tick-20261005T0650Z", "host": socket.gethostname(),
                                                  "pid": stale_pid, "pid_start": RP.process_start(stale_pid) or "1", "started_utc": "2026-10-05T06:50:00.000Z",
                                                  "tree_sha": sha40, "job": None}, indent=1)

    def receipt(op, utc, refused):
        adapters = {"relu_paired": {"route": "cluster", "pins": {"relu_paired_run_state": "11112222", "relu_paired_sidecars": "aabbccdd"},
                                    "published": {"relu_paired_run_state": {"published_utc": "2026-10-04T10:05:00.000Z", "change_utc": "2026-10-04T10:00:00.000Z", "mode": "push",
                                                                            "producer": "grading_host:ReLU"}, "relu_paired_sidecars": None},
                                    "module_blobs": {}, "shards_total": 8, "shards_ok": 7 if refused else 8, "revisions_pulled": ["relu_paired/aaaa"], "reactivated": [],
                                    "refused": (["shard 3 of 8 missing"] if refused else None)},
                    "em_misalign": {"route": "cluster", "pins": {"misalign_grades": "85febd26"}, "published": {}, "module_blobs": {}, "shards_total": 6, "shards_ok": 6,
                                    "revisions_pulled": [], "reactivated": [], "refused": None},
                    "declared_only": {"route": "refused", "pins": {}, "published": {}, "module_blobs": {}, "shards_total": 0, "shards_ok": 0, "revisions_pulled": [],
                                      "reactivated": [], "refused": ["declared cluster: refused"]}}
        for a in adapters:
            adapters[a]["read_whole"] = RP.read_whole(adapters[a])
        RP.write_receipt(rd, {"schema": RP.RECEIPT_SCHEMA, "op": {"op_id": op, "owner": "tick"}, "utc": utc, "window": op, "tree_sha": sha40, "destination": "cluster_a",
                              "jobs": {}, "attestation": None, "barrier": None, "adapters": adapters,
                              "n_read_whole": sum(1 for v in adapters.values() if v["read_whole"]), "n_refused": sum(1 for v in adapters.values() if v["refused"])})
    receipt("tick-20261005T0600Z", "2026-10-05T06:10:00.000Z", False)
    receipt("tick-20261005T0630Z", "2026-10-05T06:40:00.000Z", True)
    with open(os.path.join(rd, RP.RECEIPTS_SUBDIR, "torn.json"), "w", encoding="utf-8") as fh:
        fh.write("{not json")
    return rd


def test_slurm_only_section_names_a_check_accepted_under_the_declared_c1_allowance(tmp_path, monkeypatch):
    """A replica_check.json whose check was UNEQUAL on the certified C1 same-stamp class alone and accepted under the operator's
    declaration (accepted_as c1_class, ok) is green, says `equal modulo the C1 same-stamp class (N movers, declared)` with the
    declaration, and shows no STOP note; the section carries c1_class / accepted_as / n_movers / declaration for the page."""
    from sisteer.registry import reads_protocol as RP
    from sisteer.registry import records as R
    monkeypatch.setenv(RP.STATE_ENV, str(tmp_path / "landing"))
    root = _setup(tmp_path)
    rep = {"schema": "replica_check_state/2", "headline": "pass", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 60)), "equal": False, "ok": True,
           "problems": [], "branch": "registry-slurm-only", "tip": "c" * 40, "job": "923300", "job_state": "COMPLETED", "submission": "t", "out_remote": "/o",
           "replica_tree": None, "dispatcher": {"source": "tip", "sha256": None, "path": None},
           "head": {"n_files": 492409, "taken_utc": "2026-10-06T21:20:57.000Z", "allowlist_sha12": "abc", "projection_sha12": "aaaaaaaaaaaa", "head_sha12": "h"},
           "replica": {"projection_sha12": "bbbbbbbbbbbb", "counts": {"rows": 12863, "in_current_paper": 2948, "selected": 3064}}, "projection_version": "frame_projection/1",
           "generation": "g3", "current_generation": "g3", "parent_generation": "g2", "schema_version": 24,
           "pytest": {"rc": 0, "counts": {"tests": 9, "skipped": 0}, "expected_skipped": 0, "status": "PASS"}, "reason": None,
           "c1_class": True, "accepted_as": "c1_class", "n_movers": 106, "columns_differing": {"lifecycle_state": 106, "is_complete": 106},
           "declaration": "refold-compare job 922769, the refold27 record", "job_check_accepted_as": "c1_class"}
    R.write_json_atomic(os.path.join(root, "state", status.REPLICA_CHECK_FILE), rep, indent=1)
    so = status.status_json(root)["slurm_only"]
    rp = so["replica"]
    assert rp["present"] and rp["ok"] and rp["equal"] is False and rp["level"] == "green" and rp["accepted_as"] == "c1_class" and rp["c1_class"] is True
    assert rp["n_movers"] == 106 and rp["declaration"] == "refold-compare job 922769, the refold27 record" and so["n_problems"] == 0
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)
    html = open(out, encoding="utf-8").read()
    assert "equal modulo the C1 same-stamp class (106 movers, declared)" in html and "Accepted under the declared cutover allowance (D136): refold-compare job 922769" in html
    replica_para = html.split("Cluster reads, store authority and the replica", 1)[1]
    assert "An unequal check is a STOP (D12)" not in html and "<strong>UNEQUAL" not in replica_para
    # a refused UNEQUAL check (no acceptance) still reads UNEQUAL with the STOP note, as before
    rep.update(ok=False, headline="unequal", accepted_as=None, problems=["replica check UNEQUAL: projection differs"])
    R.write_json_atomic(os.path.join(root, "state", status.REPLICA_CHECK_FILE), rep, indent=1)
    status.render(root, out_path=out, register=False)
    html = open(out, encoding="utf-8").read()
    assert "<strong>UNEQUAL" in html and "An unequal check is a STOP (D12)" in html and "equal modulo the C1" not in html


def test_slurm_only_section_says_so_on_a_root_with_no_state_files(tmp_path, monkeypatch):
    """A fresh root (no check-authority, no landing, no reads namespace, no flag): every part of the section says it has nothing, the
    level is green with zero problems, no banner fires, and the page renders. The reads namespace is $REGISTRY_LANDING_STATE/reads/,
    OUTSIDE the registry root (the landing state), so the test points it at an empty directory."""
    from sisteer.registry import reads_protocol as RP
    monkeypatch.setenv(RP.STATE_ENV, str(tmp_path / "landing"))
    root = _setup(tmp_path)
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)
    html = open(out, encoding="utf-8").read()
    so = status.status_json(root)["slurm_only"]
    assert so["level"] == "green" and so["problems"] == [] and so["n_problems"] == 0
    assert so["authority"] == {**so["authority"], "present": False, "rows": [], "error": None, "mirror_present": False}
    assert so["replica"] == {"present": False, "error": None, "level": "green"}
    assert so["reservation"] == {"held": False, "reads_dir": RP.reads_dir(), "level": "green"}
    assert so["receipts"]["present"] is False and so["receipts"]["n_receipts"] == 0 and so["receipts"]["rows"] == [] and so["receipts"]["adapters"] == []
    assert so["maintenance"] == {"set": False, "level": "green"} and so["config"]["removed_table"] is False and so["config"]["present"] is False
    assert so["reads_dir"] == RP.reads_dir() and so["elapsed_ms"] >= 0
    for needle in ("Cluster reads, store authority and the replica", "check-authority has not run on this root yet", "no Slurm live-DB step recorded",
                   "none held", "cluster-side problems"):
        assert needle in html, needle
    for absent in ("Maintenance flag set", "Configuration refused", "Configuration unreadable", "State file problem"):
        assert absent not in html, absent


def test_slurm_only_section_renders_authority_replica_reservation_receipts_and_flags_from_state_files(tmp_path, monkeypatch):
    """Every part populated from the files its writer spells: the authority rows joined with the mirror outcome (a dangling `current`
    and an unsealed one red; an overdue carry amber; a component the check never read carried from the mirror alone), the UNEQUAL
    replica check (second-resolution stamp parsed), the stale reservation, the receipts (the newer one refusing relu_paired; the
    per-adapter last WHOLE read is the older receipt's; a torn receipt counted), the maintenance flag (window.sh's second-resolution
    stamp: a real age, not stale), the removed configuration table (the page still renders, on the budget DEFAULTS, since the loader
    refuses the file) -- each red fact named in `problems`, in the banners, in the rows and in the text command; no database query."""
    from sisteer.registry import cli
    root = _setup(tmp_path)
    rd = _slurm_only_fixture(root, str(tmp_path / "landing"), monkeypatch)
    out = str(tmp_path / "index.html")
    with DB.reading(DB.db_path(root)) as conn:
        ctx, _ = status.gather(conn, root, persist=False)
    assert ctx["budgets"] == {"section_budget_s": 20.0, "render_budget_s": 120.0, "heavy_budget_s": 900.0}, "the refused file leaves the defaults"
    so = ctx["slurm_only"]
    assert so["level"] == "red" and so["reads_dir"] == rd
    # the authority rows: the check's three components plus the mirror-only one, joined
    au = so["authority"]
    assert au["present"] and au["ok"] is False and au["n_problems"] == 2 and au["n_components"] == 4 and au["checked_utc"] == "2026-10-05T07:00:00.000Z"
    rows = {r["slug"]: r for r in au["rows"]}
    assert rows["bfcl_full"]["level"] == "red" and rows["bfcl_full"]["problems"] == "dangling, no_published_record" and rows["bfcl_full"]["exists"] is False
    assert rows["misalign_grades"]["level"] == "red" and rows["misalign_grades"]["sealed"] is False and rows["misalign_grades"]["mirror_outcome"] == "rsync_failed"
    assert rows["relu_paired_sidecars"]["level"] == "amber" and rows["relu_paired_sidecars"]["carry_overdue"] is True and rows["relu_paired_sidecars"]["problems"] == ""
    assert rows["relu_paired_sidecars"]["mirror_outcome"] == "unpushed_changes" and rows["relu_paired_sidecars"]["change_utc"] == "2026-10-04T10:00:00.000Z"
    assert rows["sampled"]["checked"] is False and rows["sampled"]["state"] == "unchecked" and rows["sampled"]["level"] == "green" and rows["sampled"]["mirror_outcome"] == "unchanged"
    # the replica check
    rp = so["replica"]
    assert rp["present"] and rp["equal"] is False and rp["ok"] is False and rp["headline"] == "unequal" and rp["level"] == "red"
    assert rp["head_n_files"] == 477086 and rp["head_sha12"] == "cebbc428aaaa" and rp["replica_sha12"] == "deadbeefcafe" and rp["projection_version"] == "frame_projection/1"
    assert rp["generation"] == "g2" and rp["current_generation"] == "g1" and rp["schema_version"] == 24 and rp["age_s"] is not None and rp["age_s"] > 0
    # the stale reservation
    rs = so["reservation"]
    assert rs["held"] and rs["state"] == "stale" and rs["alive"] is False and rs["owner"] == "tick" and rs["op_id"] == "tick-20261005T0650Z" and rs["level"] == "amber"
    # the receipts: newest first; the last WHOLE read of relu_paired is the older receipt's; the torn file counted; declared_only never read whole
    rc = so["receipts"]
    assert rc["n_receipts"] == 2 and rc["n_unreadable"] == 1 and [r["op_id"] for r in rc["rows"]] == ["tick-20261005T0630Z", "tick-20261005T0600Z"]
    assert rc["rows"][0]["level"] == "red" and rc["rows"][0]["n_refused"] == 2 and rc["rows"][0]["refused"].startswith("declared_only: declared cluster: refused; relu_paired: shard 3 of 8 missing")
    # the older receipt read both cluster adapters whole, but it too carries the declared-refused adapter: red, like the window that
    # recorded it (a declared `cluster: refused` adapter ends its window ABORTED -- the standing refusal is never quiet)
    assert rc["rows"][1]["level"] == "red" and rc["rows"][1]["n_read_whole"] == 2 and rc["rows"][1]["n_refused"] == 1 and rc["rows"][1]["refused"] == "declared_only: declared cluster: refused"
    ad = {r["adapter"]: r for r in rc["adapters"]}
    assert ad["relu_paired"]["level"] == "red" and ad["relu_paired"]["op_id"] == "tick-20261005T0600Z" and ad["relu_paired"]["newest_refused"] == "shard 3 of 8 missing"
    assert ad["relu_paired"]["pins"] == "relu_paired_run_state@11112222, relu_paired_sidecars@aabbccdd" and ad["relu_paired"]["change_utc"] == "2026-10-04T10:00:00.000Z"
    assert ad["em_misalign"]["level"] == "green" and ad["em_misalign"]["op_id"] == "tick-20261005T0630Z" and ad["em_misalign"]["shards"] == "6/6"
    assert ad["declared_only"]["level"] == "red" and ad["declared_only"]["utc"] is None
    # the flag (a real age from the second-resolution stamp) and the configuration
    mt = so["maintenance"]
    with open(H.maintenance_path(root), encoding="utf-8") as fh:
        since = json.load(fh)["since_utc"]                      # the fixture's stamp, five minutes before its now
    assert mt["set"] and mt["age_s"] is not None and 0 <= mt["age_s"] < status.MAINTENANCE_STALE_S and mt["stale"] is False and mt["level"] == "amber"
    assert mt["adapters"] == "relu_paired em_misalign" and mt["since_utc"] == since
    cf = so["config"]
    assert cf["removed_table"] is True and cf["table_keys"] == ["enabled", "volume"] and cf["level"] == "red" and cf["error"] is None
    problems = "\n".join(so["problems"])
    for needle in ("store authority: 2 component(s) with a problem", "bfcl_full: dangling, no_published_record", "replica check UNEQUAL", "reservation held by tick tick-20261005T0650Z is stale",
                   "latest read receipt tick-20261005T0630Z refused 2 adapter(s)", "1 unreadable receipt file(s)", f"maintenance flag set since {since}",
                   "still carries the removed `modal` table (enabled, volume)"):
        assert needle in problems, needle
    assert "STALE" not in problems and so["n_problems"] == 7
    # the page
    status.render(root, out_path=out, register=False)
    html = open(out, encoding="utf-8").read()
    for needle in ("Maintenance flag set", "relu_paired em_misalign", "Configuration refused", "enabled, volume", "convert_config.py", "UNEQUAL", "D12", "cebbc428aaaa", "deadbeefcafe",
                   "bfcl_full", "dangling, no_published_record", "carry_overdue", "rsync_failed", "unpushed_changes", "holder <strong>stale</strong>", "tick-20261005T0630Z",
                   "shard 3 of 8 missing", "1 receipt file(s) unreadable", "relu_paired_run_state@11112222", "6/6", 'class="red"', 'class="amber"'):
        assert needle in html, needle
    assert "age unknown" not in html and "STALE</strong>" not in html
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\b\w*_legacy\b", " ", text)
    for word in ("arm", "leg", "lane", "cell", "wave", "grader"):
        assert not re.search(r"(?<![A-Za-z0-9_])" + word + r"(?![A-Za-z0-9_])", text, re.IGNORECASE), word
    # the history line carries the section's cost beside total_ms (S9: within budget WITH the section)
    hist = status.read_render_history(root)[-1]
    assert hist["slurm_only_ms"] == pytest.approx(ctx["slurm_only"]["elapsed_ms"], abs=50) or hist["slurm_only_ms"] >= 0
    assert hist["operational_ms"] >= hist["slurm_only_ms"] >= 0 and hist["slurm_only_level"] == "red" and hist["slurm_only_problems"] == 7
    assert hist["total_ms"] >= hist["gather_ms"] >= hist["operational_ms"]
    # the text command prints the same headlines
    monkeypatch.setenv("REGISTRY_ROOT", root)
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli.main(["status"]) == 0
    printed = buf.getvalue()
    for needle in ("cluster side: red (7 problem(s))", "store authority: checked 2026-10-05T07:00:00.000Z, 4 component(s), 2 with a problem",
                   "replica check: UNEQUAL REFUSED (unequal) at 2026-10-05T05:00:00Z, generation g2", "reads reservation: tick tick-20261005T0650Z", "(stale)",
                   "read receipts: 2, latest tick-20261005T0630Z", "PROBLEM: config.toml still carries the removed"):
        assert needle in printed, needle


def test_slurm_only_section_shows_a_torn_state_file_as_an_error_and_a_stale_flag_red_and_a_live_holder_green(tmp_path, monkeypatch):
    """A torn store_authority.json, a replica_check.json that is not an object, an unreadable reservation and a foreign-schema authority
    file are each that part's `error` (red), never a page that fails to render; the rows still come from the mirror state. A flag older
    than MAINTENANCE_STALE_S is red and the banner says a window left it behind; a flag whose stamp does not parse has an unknown age
    and stays amber. A reservation whose pid is THIS process is live (green); one on another host is `unknown`."""
    import socket
    from sisteer.registry import reads_protocol as RP
    from sisteer.registry import records as R
    from sisteer.registry import stores as ST
    root = _setup(tmp_path)
    rd = _slurm_only_fixture(root, str(tmp_path / "landing"), monkeypatch, stale_pid=os.getpid())
    out = str(tmp_path / "index.html")
    so = status.status_json(root)["slurm_only"]
    assert so["reservation"]["state"] == "live" and so["reservation"]["alive"] is True and so["reservation"]["level"] == "green"
    assert not any("reservation" in p for p in so["problems"])
    # another host: unknown, amber, never replaced from here
    res = RP.read_reservation(rd)
    R.write_json_atomic(RP.reservation_path(rd), dict(res, host=socket.gethostname() + "-elsewhere"), indent=1)
    so = status.status_json(root)["slurm_only"]
    assert so["reservation"]["state"] == "unknown" and so["reservation"]["alive"] is None and so["reservation"]["level"] == "amber"
    # torn files
    state = os.path.join(root, "state")
    with open(os.path.join(state, ST.AUTHORITY_STATE_FILE), "w", encoding="utf-8") as fh:
        fh.write("{torn")
    with open(os.path.join(state, status.REPLICA_CHECK_FILE), "w", encoding="utf-8") as fh:
        fh.write("[]")
    with open(RP.reservation_path(rd), "w", encoding="utf-8") as fh:
        fh.write("not json")
    # a flag older than the bound: a window left it behind -- stamped FOUR HOURS before now, explicitly past status.MAINTENANCE_STALE_S (3 h)
    old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 4 * 3600))
    with open(H.maintenance_path(root), "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"since_utc": old, "reason": "landing window (window.sh pid 1)", "pid": 1}) + "\n")
    status.render(root, out_path=out, register=False)
    html = open(out, encoding="utf-8").read()
    so = status.status_json(root)["slurm_only"]
    assert so["authority"]["present"] is False and "unreadable or not JSON" in so["authority"]["error"] and so["authority"]["level"] == "red"
    assert len(so["authority"]["rows"]) == 3 and all(r["checked"] is False for r in so["authority"]["rows"]), "the mirror's components still render"
    assert so["replica"]["present"] is False and "not a JSON object" in so["replica"]["error"] and so["replica"]["level"] == "red"
    assert so["reservation"]["held"] and so["reservation"]["state"] == "unreadable" and so["reservation"]["level"] == "red"
    assert so["maintenance"]["stale"] is True and so["maintenance"]["level"] == "red"
    assert any(p.startswith("authority: store_authority.json is unreadable") for p in so["problems"])
    assert any(p.startswith("replica: replica_check.json is not a JSON object") for p in so["problems"])
    assert any("STALE: a window left it behind" in p for p in so["problems"])
    for needle in ("State file problem", "store_authority.json is unreadable or not JSON", "replica_check.json is not a JSON object",
                   "reservation.json is unreadable or not JSON", "STALE</strong>", "a window left it behind"):
        assert needle in html, needle
    # a foreign schema in the authority file is named, not read
    R.write_json_atomic(os.path.join(state, ST.AUTHORITY_STATE_FILE), {"schema": "something_else/9", "components": {"x": {}}}, indent=1)
    so = status.status_json(root)["slurm_only"]
    assert so["authority"]["present"] is False and "carries schema 'something_else/9'" in so["authority"]["error"]
    # a flag whose stamp does not parse: age unknown, amber, never called stale
    with open(H.maintenance_path(root), "w", encoding="utf-8") as fh:
        fh.write('{"since_utc": "yesterday", "reason": "x"}\n')
    so = status.status_json(root)["slurm_only"]
    assert so["maintenance"]["age_s"] is None and so["maintenance"]["stale"] is None and so["maintenance"]["level"] == "amber"
    status.render(root, out_path=out, register=False)
    assert "age unknown: the stamp does not parse" in open(out, encoding="utf-8").read()
    # a config.toml that does not parse is named too (the loader refuses it as well); the page still renders on the defaults
    from sisteer.registry import config as CFG
    with open(CFG.config_path(root), "w", encoding="utf-8") as fh:
        fh.write("[status\nnot toml\n")
    status.render(root, out_path=out, register=False)
    so = status.status_json(root)["slurm_only"]
    assert so["config"]["removed_table"] is False and so["config"]["error"] and so["config"]["level"] == "red"
    assert "Configuration unreadable" in open(out, encoding="utf-8").read()


def test_slurm_only_section_reads_no_database_and_costs_milliseconds(tmp_path, monkeypatch):
    """The section is state files only (no DB query, no schema change): it renders the same with the database
    connection closed under it, and its gather on the populated fixture takes milliseconds -- the within-budget claim (S9) rests on the
    database sections, not on this one."""
    from sisteer.registry import reads_protocol as RP
    root = _setup(tmp_path)
    _slurm_only_fixture(root, str(tmp_path / "landing"), monkeypatch)
    before = status.slurm_only_section(root)
    os.replace(DB.db_path(root), DB.db_path(root) + ".away")        # no database at all
    try:
        after = status.slurm_only_section(root)
    finally:
        os.replace(DB.db_path(root) + ".away", DB.db_path(root))
    strip = lambda d: json.loads(json.dumps({k: v for k, v in d.items() if k != "elapsed_ms"}, default=str))   # noqa: E731
    a, b = strip(before), strip(after)
    for part in ("authority", "replica", "reservation", "receipts", "maintenance", "config"):
        for k in ("age_s", "checked_age_s", "change_age_s"):
            a[part].pop(k, None) if isinstance(a[part], dict) else None
            b[part].pop(k, None) if isinstance(b[part], dict) else None
    assert a["problems"] == b["problems"] and a["level"] == b["level"] and a["authority"]["rows"][0]["slug"] == b["authority"]["rows"][0]["slug"]
    assert before["elapsed_ms"] < 2000, before["elapsed_ms"]
    assert RP.reads_dir() == before["reads_dir"]


def test_authority_rows_read_the_check_and_verify_authority_stamps_by_the_command_that_wrote_them(tmp_path, monkeypatch):
    """`state/store_mirror.json`'s `last_run.outcome` is stamped by `stores mirror` / `push` (a mirror outcome), by
    `stores check` as `check:<state>` and by the one-time `stores verify-authority` as `verify-authority:<state>`, and the first cut read
    every stamp through OUTCOME_LEVELS -- so a correctly verified store (`verify-authority:verified`) and a matching in-place check
    (`check:match`) rendered RED, with a `store mirror:` headline, during the very cutover they certify. Every stamp the writers spell,
    rendered through the section: the successes green (no headline, a green row on the page), the failures red by name in the row AND
    in the headline with their stamp, a run's in-flight stamps amber, a stamp nobody named red; then the mirror file with only successes
    in it is a green section with zero problems."""
    from sisteer.registry import reads_protocol as RP
    from sisteer.registry import records as R
    from sisteer.registry import stores as ST
    monkeypatch.setenv(RP.STATE_ENV, str(tmp_path / "landing"))
    root = _setup(tmp_path)
    stamps = {"orig_match": "check:match", "orig_mismatch": "check:mismatch", "orig_error": "check:error",
              "moved_ok": "verify-authority:verified", "moved_na": "verify-authority:not_applicable", "moved_unequal": "verify-authority:unequal",
              "moved_gone": "verify-authority:no_current", "moved_bad": "verify-authority:verify_failed", "moved_raced": "verify-authority:source_changed",
              "moved_error": "verify-authority:error", "mid_inline": "verify_inline", "mid_job": "verify_submitted",
              "plain_ok": "unchanged", "plain_bad": "rsync_failed", "plain_unnamed": "an_outcome_nobody_named"}
    expected = {"orig_match": "green", "orig_mismatch": "red", "orig_error": "red", "moved_ok": "green", "moved_na": "green", "moved_unequal": "red",
                "moved_gone": "red", "moved_bad": "red", "moved_raced": "red", "moved_error": "red", "mid_inline": "amber", "mid_job": "amber",
                "plain_ok": "green", "plain_bad": "red", "plain_unnamed": "red"}

    def mirror_doc(slugs):
        return {"schema": ST.STATE_SCHEMA, "updated_utc": "2026-10-05T08:00:00.000Z", "destination": "cluster_a",
                "components": {s: {"authority": "cluster_a_origin" if s.startswith("orig") else "cluster_a", "generation": {"content_sha8": "aabbccdd", "state": "published"},
                                   "last_run": {"outcome": stamps[s], "utc": "2026-10-05T07:59:00.000Z"}} for s in slugs}}

    R.write_json_atomic(ST.state_path(root), mirror_doc(list(stamps)), indent=1)
    # check-authority has read the verified move and found nothing wrong with it: its row must be green end to end
    auth = {"schema": ST.AUTHORITY_STATE_SCHEMA, "checked_utc": "2026-10-05T08:00:30.000Z", "destination": "cluster_a", "stores_dir": "/x/stores", "config": "c.yaml",
            "ok": True, "n_components": 1, "n_problems": 0, "problems": [], "counts": {"ok": 1},
            "components": {"moved_ok": {"authority": "cluster_a", "producer": "grading_host:ReLU", "current": "aabbccdd", "real": "/x/aabbccdd", "exists": True, "verified": True,
                                        "published_record": True, "sealed": True, "change_utc": "2026-10-04T10:00:00.000Z", "published_utc": "2026-10-04T10:05:00.000Z",
                                        "publish_mode": "move", "grading_host_mirror": "/h/relu", "grading_host_mirror_present": True, "transitional_carry": False, "carry_deadline": None,
                                        "carry_overdue": False, "problems": [], "notes": [], "state": "ok"}}}
    R.write_json_atomic(os.path.join(root, "state", ST.AUTHORITY_STATE_FILE), auth, indent=1)
    so = status.slurm_only_section(root)
    rows = {r["slug"]: r for r in so["authority"]["rows"]}
    assert {s: rows[s]["mirror_level"] for s in stamps} == expected
    assert {s: rows[s]["level"] for s in stamps} == expected, "no authority problem anywhere: the row's level is the stamp's"
    assert rows["moved_ok"]["checked"] is True and rows["moved_ok"]["state"] == "ok" and rows["moved_ok"]["mirror_outcome"] == "verify-authority:verified"
    assert so["authority"]["level"] == "red" and so["level"] == "red"
    headline = next(p for p in so["problems"] if p.startswith("store mirror: "))
    red_in_slug_order = [s for s in sorted(stamps) if expected[s] == "red"]
    assert headline == ("store mirror: 9 component(s) with a red last outcome (" + ", ".join(f"{s}: {stamps[s]}" for s in red_in_slug_order[:6]) + "; …)"), headline
    for s in ("moved_ok", "moved_na", "orig_match", "mid_inline", "mid_job", "plain_ok"):
        assert s not in headline, s
    assert so["n_problems"] == 1, so["problems"]
    # the page: the successes in green rows, the failures in red ones, the in-flight stamp amber
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)
    html = open(out, encoding="utf-8").read()

    def row_class(slug):
        m = re.search(r'<tr class="(\w+)">(?:(?!</tr>).)*?<td>' + re.escape(slug) + r'</td>', html, re.S)
        assert m, slug
        return m.group(1)

    assert row_class("moved_ok") == "green" and row_class("orig_match") == "green" and row_class("moved_na") == "green"
    assert row_class("orig_mismatch") == "red" and row_class("moved_unequal") == "red" and row_class("plain_unnamed") == "red" and row_class("mid_job") == "amber"
    for needle in ("check:match", "verify-authority:verified", "check:mismatch", "verify-authority:unequal", "whichever command stamped it"):
        assert needle in html, needle
    # only successes left (the cutover's end state: every store moved and verified, every in-place original matching): green, no headline
    R.write_json_atomic(ST.state_path(root), mirror_doc(["orig_match", "moved_ok", "moved_na", "plain_ok"]), indent=1)
    so = status.slurm_only_section(root)
    assert so["authority"]["level"] == "green" and so["level"] == "green" and so["problems"] == [] and so["n_problems"] == 0
    assert all(r["level"] == "green" and r["mirror_level"] == "green" for r in so["authority"]["rows"]) and len(so["authority"]["rows"]) == 4


def test_receipts_level_is_the_newest_receipt_and_each_adapters_newest_result_while_older_rows_keep_their_color(tmp_path, monkeypatch):
    """A receipt that refused an adapter stayed among the newest ten for nine more ticks after the next tick had read every
    adapter whole, and the part's level took the worst of all ten rows -- a red tile with no headline, since only the newest receipt
    earns one. The level now rests on the newest receipt, each adapter's newest result and the unreadable files: after the recovering
    tick the part is green with the older row still red (history); a tick that refuses an adapter is red with the latest-receipt
    headline; an adapter refused in an older receipt that the latest receipt did not read at all is still red -- that refusal is its
    newest result -- and its own headline names it with that receipt; an adapter never read whole is amber with a headline. Every red
    or amber level has at least one headline from this part."""
    from sisteer.registry import reads_protocol as RP
    monkeypatch.setenv(RP.STATE_ENV, str(tmp_path / "landing"))
    root = _setup(tmp_path)
    rd = RP.reads_dir()
    os.makedirs(os.path.join(rd, RP.RECEIPTS_SUBDIR), exist_ok=True)

    def entry(total, ok, refused=None):
        e = {"route": "cluster", "pins": {"c": "aabbccdd"}, "published": {}, "module_blobs": {}, "shards_total": total, "shards_ok": ok,
             "revisions_pulled": [], "reactivated": [], "refused": refused}
        e["read_whole"] = RP.read_whole(e)
        return e

    def receipt(op, utc, adapters):
        RP.write_receipt(rd, {"schema": RP.RECEIPT_SCHEMA, "op": {"op_id": op, "owner": "tick"}, "utc": utc, "window": op, "tree_sha": "c" * 40, "destination": "cluster_a",
                              "jobs": {}, "attestation": None, "barrier": None, "adapters": adapters,
                              "n_read_whole": sum(1 for v in adapters.values() if v["read_whole"]), "n_refused": sum(1 for v in adapters.values() if v["refused"])})

    def part():
        so = status.slurm_only_section(root)
        rc = so["receipts"]
        if rc["level"] != "green":
            assert so["problems"], rc["level"]              # a red or amber level always has a headline explaining it
        return so, rc, so["problems"]

    # tick 1 refuses A; tick 2 reads every adapter whole: the part recovers (green), tick 1's row stays red as history
    receipt("tick-1", "2026-10-05T06:10:00.000Z", {"A": entry(8, 7, ["shard 3 of 8 missing"]), "B": entry(6, 6), "C": entry(2, 2)})
    so, rc, problems = part()
    assert rc["level"] == "red" and [r["level"] for r in rc["rows"]] == ["red"] and problems == ["latest read receipt tick-1 refused 1 adapter(s)"]
    receipt("tick-2", "2026-10-05T06:40:00.000Z", {"A": entry(8, 8), "B": entry(6, 6), "C": entry(2, 2)})
    so, rc, problems = part()
    assert [r["op_id"] for r in rc["rows"]] == ["tick-2", "tick-1"] and [r["level"] for r in rc["rows"]] == ["green", "red"]
    assert {r["adapter"]: r["level"] for r in rc["adapters"]} == {"A": "green", "B": "green", "C": "green"}
    assert rc["level"] == "green" and problems == [] and so["level"] == "green" and so["n_problems"] == 0
    # tick 3 refuses A again and reads D in part (3 of 4 shards, no refusal): red with the latest-receipt headline; D amber with its own
    receipt("tick-3", "2026-10-05T07:10:00.000Z", {"A": entry(8, 7, ["shard 5 of 8 missing"]), "B": entry(6, 6), "D": entry(4, 3)})
    so, rc, problems = part()
    assert rc["level"] == "red" and rc["rows"][0]["level"] == "red"
    assert {r["adapter"]: r["level"] for r in rc["adapters"]} == {"A": "red", "B": "green", "C": "green", "D": "amber"}
    assert problems == ["latest read receipt tick-3 refused 1 adapter(s)", "1 adapter(s) never read whole (D)"]
    # tick 4 reads only B: the latest receipt is green, but A's newest result is still tick-3's refusal -- red, named with that receipt
    receipt("tick-4", "2026-10-05T07:40:00.000Z", {"B": entry(6, 6)})
    so, rc, problems = part()
    assert rc["rows"][0]["op_id"] == "tick-4" and [r["level"] for r in rc["rows"]] == ["green", "red", "green", "red"]
    ad = {r["adapter"]: r for r in rc["adapters"]}
    assert ad["A"]["level"] == "red" and ad["A"]["newest_op_id"] == "tick-3" and ad["A"]["op_id"] == "tick-2", "the last whole read is tick-2's, the newest result tick-3's refusal"
    assert ad["A"]["newest_refused"] == "shard 5 of 8 missing" and ad["D"]["level"] == "amber" and ad["D"]["op_id"] is None
    assert rc["level"] == "red" and so["level"] == "red"
    assert problems == ["1 adapter(s) whose newest read refused them, in a receipt before the latest (A (tick-3))", "1 adapter(s) never read whole (D)"]
    # the page carries the history row red and the explanation
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)
    html = open(out, encoding="utf-8").read()
    for needle in ("A (tick-3)", "never read whole (D)", "keeps that color while it is listed", "tick-1", "shard 3 of 8 missing"):
        assert needle in html, needle
    # tick 5 reads everything whole: green, nothing to say; then one unreadable receipt file is a current problem with its headline
    receipt("tick-5", "2026-10-05T08:10:00.000Z", {"A": entry(8, 8), "B": entry(6, 6), "C": entry(2, 2), "D": entry(4, 4)})
    so, rc, problems = part()
    assert rc["level"] == "green" and problems == [] and [r["level"] for r in rc["rows"]] == ["green", "green", "red", "green", "red"]
    with open(os.path.join(rd, RP.RECEIPTS_SUBDIR, "torn.json"), "w", encoding="utf-8") as fh:
        fh.write("{not json")
    so, rc, problems = part()
    assert rc["level"] == "red" and rc["n_unreadable"] == 1 and problems == [f"1 unreadable receipt file(s) under {rd}"]


def test_receipts_newest_result_is_never_green_when_it_did_not_read_whole_and_a_standing_refusal_is_its_own_non_red_state(tmp_path, monkeypatch):
    """An adapter whose NEWEST receipt left refused-known ids standing unread -- or read it in
    part -- colored GREEN whenever an older receipt had read it whole; the older read is the baseline, not the newest result, so it is amber
    with a headline naming the state. A refusal the cluster read re-confirmed byte for byte against the prior recorded refusal is
    recorded as its own `standing` state -- shown with the refusal and the recorded refusal it stands on, never red, never an abort -- read
    from the ONE shape the contract names (the entry's `standing_refusals` list, state `standing_refusal`, `prior_refusal` with its window and
    time); `refused_standing` keeps its one meaning (standing-UNREAD ids) and is never read as a standing refusal; an older receipt without
    the list renders as before."""
    from sisteer.registry import reads_protocol as RP
    monkeypatch.setenv(RP.STATE_ENV, str(tmp_path / "landing"))
    root = _setup(tmp_path)
    rd = RP.reads_dir()
    os.makedirs(os.path.join(rd, RP.RECEIPTS_SUBDIR), exist_ok=True)

    def entry(total, ok, refused=None, **more):
        e = {"route": "cluster", "pins": {"c": "aabbccdd"}, "published": {}, "module_blobs": {}, "shards_total": total, "shards_ok": ok,
             "revisions_pulled": [], "reactivated": [], "refused": refused}
        e.update(more)
        # the cluster-read side's `read_whole`: main's rule (every shard ok, no refusal, no id left standing unread) -- an adapter whose only
        # refusals STAND is therefore written whole, with `refused` None, so the tick leaves it alone; the page decides by the list
        e["read_whole"] = RP.read_whole(e)
        if more.get("standing_refusals") and not e["refused"]:
            assert e["read_whole"] is True, "the contract: a standing-only adapter is read_whole true (refused None / [] / absent)"
        return e

    def receipt(op, utc, adapters):
        RP.write_receipt(rd, {"schema": RP.RECEIPT_SCHEMA, "op": {"op_id": op, "owner": "tick"}, "utc": utc, "window": op, "tree_sha": "c" * 40, "destination": "cluster_a",
                              "jobs": {}, "attestation": None, "barrier": None, "adapters": adapters,
                              "n_read_whole": sum(1 for v in adapters.values() if v["read_whole"]), "n_refused": sum(1 for v in adapters.values() if v["refused"])})

    def part():
        so = status.slurm_only_section(root)
        rc = so["receipts"]
        if rc["level"] != "green":
            assert so["problems"], rc["level"]
        return so, rc, so["problems"], {r["adapter"]: r for r in rc["adapters"]}

    # the newest-state decision: a non-empty `standing_refusals` list outranks `read_whole: true` (the cluster read writes both for a
    # standing-only adapter); a NEW refusal beside it outranks both, red; `refused` None / [] / absent are all "no refusal"
    sr = [{"group": "g0", "state": status.STANDING_REFUSAL, "refusal": "r", "prior_refusal": {"window": "W29", "utc": "2026-10-06T12:00:00.000Z", "digest": "d" * 16,
                                                                                                 "attempt_revision_id": "a" * 16, "code_sha": "c" * 40}}]
    for no_refusal in (None, []):
        e = dict(entry(8, 8, refused=no_refusal, standing_refusals=sr), read_whole=True)
        assert status.newest_entry_state(e, True) == (status.STANDING_REFUSAL, "standing")
    e = {k: v for k, v in entry(8, 8, standing_refusals=sr).items() if k != "refused"}
    assert status.newest_entry_state(e, True) == (status.STANDING_REFUSAL, "standing")
    assert status.newest_entry_state(dict(entry(8, 7, refused=["shard 5 of 8 missing"], standing_refusals=sr), read_whole=False), False) == ("refused", "red")
    assert status.newest_entry_state(entry(8, 8), True) == ("whole", "green")

    # tick 1 reads A, B and C whole: green, as before
    receipt("tick-1", "2026-10-07T06:10:00.000Z", {"A": entry(8, 8), "B": entry(6, 6), "C": entry(2, 2)})
    so, rc, problems, ad = part()
    assert rc["level"] == "green" and problems == [] and {a: r["newest_state"] for a, r in ad.items()} == {"A": "whole", "B": "whole", "C": "whole"}
    # tick 2: A's pull leaves an id standing UNREAD, B is read in part (3 of 6 shards, no refusal): both AMBER although tick 1
    # read them whole -- the last whole read is still tick 1's (the baseline column), the newest result is not whole
    receipt("tick-2", "2026-10-07T06:40:00.000Z", {"A": entry(8, 8, refused_standing=["revS"]), "B": entry(6, 3), "C": entry(2, 2)})
    so, rc, problems, ad = part()
    assert ad["A"]["level"] == "amber" and ad["A"]["newest_state"] == "standing_unread" and ad["A"]["standing_unread"] == "revS"
    assert ad["A"]["op_id"] == "tick-1" and ad["A"]["newest_op_id"] == "tick-2", "the last whole read is the baseline; the newest result is tick-2's"
    assert ad["B"]["level"] == "amber" and ad["B"]["newest_state"] == "partial" and ad["C"]["level"] == "green"
    assert rc["level"] == "amber" and rc["rows"][0]["level"] == "green" and rc["rows"][0]["n_standing_refusals"] == 0
    assert problems == ["2 adapter(s) whose newest read did not read them whole, their last whole read older (A (standing_unread, tick-2), B (partial, tick-2))"]

    def standing(group, refusal, window="W29", utc="2026-10-06T12:00:00.000Z"):
        return {"group": group, "state": status.STANDING_REFUSAL, "refusal": refusal, "prior_refusal": {"window": window, "utc": utc, "digest": "d" * 16}}

    # tick 3: STANDING refusals in the contract's one shape -- A one group, B two groups, C one beside an item of another state --
    # and D read whole: the receipt and the adapters are `standing`, not red; each shows its refusal and the recorded refusal it stands on;
    # the headline says recorded, not new
    receipt("tick-3", "2026-10-07T07:10:00.000Z", {
        "A": entry(8, 8, standing_refusals=[standing("g1", "object_sha256 records sha aaaa… but the bytes hash to bbbb…")]),
        "B": entry(6, 6, standing_refusals=[standing("g2", "no sidecar names the fit model"), standing("g3", "no sidecar names the fit model", "W30", "2026-10-07T01:00:00.000Z")]),
        "C": entry(2, 2, standing_refusals=[standing("g4", "the reader's refusal text, unchanged"), {"group": "g5", "state": "accepted", "refusal": None}]),
        "D": entry(4, 4)})
    so, rc, problems, ad = part()
    assert {a: r["level"] for a, r in ad.items()} == {"A": "standing", "B": "standing", "C": "standing", "D": "green"}
    assert {a: r["newest_state"] for a, r in ad.items()} == {"A": "standing_refusal", "B": "standing_refusal", "C": "standing_refusal", "D": "whole"}
    assert ad["A"]["standing_on"] == "g1: object_sha256 records sha aaaa… but the bytes hash to bbbb… (stands on W29 at 2026-10-06T12:00:00.000Z)"
    assert ad["B"]["standing_on"] == ("g2: no sidecar names the fit model (stands on W29 at 2026-10-06T12:00:00.000Z); "
                                      "g3: no sidecar names the fit model (stands on W30 at 2026-10-07T01:00:00.000Z)")
    assert ad["C"]["standing_on"] == "g4: the reader's refusal text, unchanged (stands on W29 at 2026-10-06T12:00:00.000Z)" and ad["A"]["newest_refused"] == ""
    assert ad["A"]["standing_unread"] == "" and status.standing_refusals(entry(2, 2, standing_refusals=[{"group": "g5", "state": "accepted"}])) == []
    row = rc["rows"][0]
    assert row["level"] == "standing" and row["n_refused"] == 0 and row["n_standing_refusals"] == 3 and "g1: object_sha256" in row["standing_refusals"]
    assert row["n_read_whole"] == 4                                   # the cluster read counted the standing adapters whole; the list decided the row
    assert ad["A"]["op_id"] == "tick-3" and ad["A"]["shards"] == "8/8"  # and the tick's baseline (the last whole read) is this receipt, as intended
    assert "(stands on W29 at 2026-10-06T12:00:00.000Z)" in row["standing_refusals"]
    assert rc["level"] == "standing" and so["level"] == "standing" and len(problems) == 1
    assert problems[0].startswith("3 adapter(s) with a standing refusal -- re-confirmed byte for byte") and "recorded, not new, not an abort" in problems[0]
    assert "A (g1: object_sha256 records sha aaaa" in problems[0]
    assert status.LEVEL_RANK["standing"] < status.LEVEL_RANK["amber"] < status.LEVEL_RANK["red"]
    # the string `refused_standing` is NOT the H2 state: a dict there is not a standing refusal (and the entry is still not whole)
    assert status.standing_refusals(entry(2, 2, refused_standing=[{"group": "g9", "state": status.STANDING_REFUSAL}])) == []
    # a NEW refusal beside a standing one is still red, with the latest-receipt headline first
    receipt("tick-4", "2026-10-07T07:40:00.000Z", {"A": entry(8, 7, ["shard 5 of 8 missing"]), "C": entry(2, 2, standing_refusals=[standing("g4", "unchanged")])})
    so, rc, problems, ad = part()
    assert ad["A"]["level"] == "red" and ad["C"]["level"] == "standing" and rc["rows"][0]["level"] == "red" and rc["level"] == "red"
    # B's newest result is still tick-3's standing refusal (tick-4 did not read B), so two adapters stand: B (tick-3) and C (tick-4)
    assert problems[0] == "latest read receipt tick-4 refused 1 adapter(s)" and any(p.startswith("2 adapter(s) with a standing refusal") for p in problems)
    assert ad["B"]["level"] == "standing" and ad["B"]["newest_op_id"] == "tick-3" and ad["D"]["level"] == "green"
    # the page: the standing rows carry their own class and color, the prior refusal is on the page, the history rows keep their colors
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)
    html = open(out, encoding="utf-8").read()
    for needle in ('class="standing"', "level-standing", "the reader's refusal text, unchanged", "no sidecar names the fit model", "standing_on", "newest_state",
                   "stands on W29 at 2026-10-06T12:00:00.000Z", "recorded, not new, not an abort", "never red", "standing_refusals"):
        assert needle in html, needle


def test_parked_by_name_section_lists_each_parked_bundle_with_its_ruling(tmp_path):
    """A group an adapter read and PARKED (an accepted read with no record; `source_status.extra_json.parked`, ok = 0) is listed
    on the page one line per bundle with the matched spelling, the ruled status / role, the reason and the ruling -- off the per-group status
    rows, so an adapter's parked groups never fold into one line."""
    from sisteer.registry import records as R
    root = _setup(tmp_path)
    parked = [{"dir": "/home/synthetic/stage_objects/_controls/synthetic_si__rand_recipeB_s31", "kind_legacy": "rand_recipeB_s31", "legacy": "diffmean_random | rand_",
               "status_ruled": "withdrawn", "object_role_ruled": "other", "reason": "synthetic: a Gaussian direction control of the fitting pipeline",
               "ruling": "results-registry D37 (2026-09-16): legacy_spellings.yaml out_of_scope_object_kinds", "ruling_source": "decisions.md D37"},
              {"dir": "/home/synthetic/stage_objects/synthetic_si/objects_a/synthetic_si__lofit", "kind_legacy": "lofit", "legacy": "lofit",
               "status_ruled": "withdrawn", "object_role_ruled": "other", "reason": "synthetic: a learned-feature intervention kind",
               "ruling": "results-registry D37 (2026-09-16): legacy_spellings.yaml out_of_scope_object_kinds", "ruling_source": "decisions.md D37"}]
    recs = []
    for i, ent in enumerate(parked):
        payload = {"lane_source_legacy": "objects_bank", "adapter": "objects_bank", "source_id": f"objects_bank:{i:064x}", "owner": "registry",
                   "where_json": [ent["dir"] + "/object.pt"], "ok": 0, "detail": f"parked by name: {ent['dir']} ({ent['ruling']})", "cells_measured": 0,
                   "observed_utc": synth.t(900 + i), "source_revision_id": "", "extra": {"refuse": [f"parked by name: {ent['dir']}"], "parked": [ent]}}
        recs.append(synth.rec("source_status", payload, synth.t(900 + i)))
    # an ok row with no parked entry beside them: never listed
    recs.append(synth.rec("source_status", {"lane_source_legacy": "objects_bank", "adapter": "objects_bank", "source_id": "objects_bank:" + "f" * 64, "owner": "registry",
                                            "where_json": ["/home/synthetic/stage_objects/x/object.pt"], "ok": 1, "detail": "read 1 input", "cells_measured": 0,
                                            "observed_utc": synth.t(950), "source_revision_id": "", "extra": {}}, synth.t(950)))
    R.write_records(os.path.join(root, "records", "status", "000001-source_status.jsonl"), recs)
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        ctx, _timings = status.gather(conn, root)
    finally:
        conn.close()
    rows = ctx["parked_sources"]
    assert [(r["kind_legacy"], r["status_ruled"], r["object_role_ruled"]) for r in rows] == [("lofit", "withdrawn", "other"), ("rand_recipeB_s31", "withdrawn", "other")]
    assert all(r["adapter"] == "objects_bank" and r["ruling"].startswith("results-registry D37") and r["reason"].startswith("synthetic:") for r in rows)
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)
    html = open(out, encoding="utf-8").read()
    for needle in ("Parked by name", "rand_recipeB_s31", "synthetic_si__lofit", "out_of_scope_object_kinds", "nothing is silent and nothing is refused"):
        assert needle in html, needle
    assert html.count("results-registry D37 (2026-09-16)") >= 2
