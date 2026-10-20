"""Status page: ``registry status --html`` renders ``templates/status.html.j2``
from the database into the figure mirror's ``results-registry/index.html`` (served under
``$SISTEER_FIGS_URL_BASE`` when the figure mirror has a public base URL, :data:`PAGE_URL`) AFTER registering
the page as an ``artifact`` (kind=page, sha of the rendered bytes) — "published implies recorded" holds
for the page itself. Every SQL section is ONE query named in :data:`QUERIES` so the write-performance
report can time them; the operational sections (snapshot freshness per destination, mirror lag, disk
headroom, the paper rebuild's RESULT, timer health, live leases, the footprint) read the small state files
the timers write under ``<root>/state/``.

Glanceable and neutral: a row of counts up top, then one table per section, no colour beyond the two
banners (ingest paused; the r2-compat deprecation notice). No campaign jargon — the vocabulary test scans
the template. The "objects in / out of the current round" section renders from ``v_paper_objects`` when
that view exists and otherwise says so — never an empty table that looks complete.

Rendered within budget, content kept. Every section query runs under a
SQLite progress handler: past ``[status] section_budget_s`` (default 20 s) it is interrupted and the section renders its
LAST GOOD rows from ``<root>/state/status_sections/<name>.json`` with a visible "as of <stamp>; skipped this render" mark
(a section with no cache says so, never an empty table). Every section that exceeds its budget joins the DYNAMIC
cached-with-refresh set in ``state/status_heavy.json`` — no fixed list; ``registry status --refresh-heavy`` (the 6-hourly
``registry-status-heavy.timer``, skipped while the ledger head stands still) recomputes each member exactly under
``heavy_budget_s`` into the same cache files and clears members that came in under the section budget, and a later
in-budget render clears a member too. ``render_budget_s`` (120 s) bounds the whole render: the sections not in the set
run first, the members get what remains, and a section the budget cannot reach is served from cache and marked. The
row counts are exact for the cheap tables and cached for the heavy ones (per table, the cheapest-by-last-count first).
The footer keeps the "Query timings (ms)" table with ``total_ms`` and a per-row ``as_of`` / ``from_cache``, and every
render APPENDS one line to ``state/status_render.jsonl`` (bounded to the last :data:`RENDER_HISTORY_MAX` renders by an
atomic tail rewrite) — the acceptance instrument ``scripts/registry/measure/status_footer.py`` reads as a series.

"Cluster reads, store authority and the replica" renders from STATE FILES ONLY — no
database query, no schema change (:func:`slurm_only_section`): ``state/store_authority.json`` (the last `stores check-authority`
reading of every component's `current`: authority, producer, the sha8, VERIFIED / SEALED / PUBLISHED beside it, ``published_utc`` /
``change_utc``, the transitional carry and its deadline) joined with ``state/store_mirror.json`` (the grading host mirror's last outcome per
component and its level); ``state/replica_check.json`` (the last Slurm live-DB step: equal or not, the head's file count, both
projection sha12s, the projection version, the generation and schema version); the reads reservation holder and the latest read
receipts under ``$REGISTRY_LANDING_STATE/reads/`` (``reads_protocol``; the per-adapter last whole read, the publication clock's inputs); the
maintenance flag (``state/maintenance.json``); and a RED row while ``config.toml`` still carries the removed ``modal`` table (the
loader refuses it; the page itself keeps rendering on the budget defaults). A dangling, unverified or unsealed `current`, an unequal
replica check, a refused receipt, a stale flag or the removed table is red; a torn state file is shown as an error, never a page
that fails to render. The render's history line carries ``operational_ms`` and ``slurm_only_ms`` beside ``total_ms``, so the
within-budget claim is checkable with the new section in place.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import time
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from . import config as CFG
from . import db as DB
from . import decode_precedence as DP
from . import harvest as H
from . import records as R
from . import site_aliases as SA
from .analysis import PAGE_URLS_SQL as _AN_PAGE_URLS_SQL

DEFAULT_OUT = os.path.join(os.path.expanduser("~"), "figs", "results-registry", "index.html")
#: The URL the page is served at: ``$SISTEER_FIGS_URL_BASE`` (the public base URL of the figure mirror) + ``results-registry/``;
#: None while the variable is unset, and the page is then known by its path alone.
PAGE_URL = (os.environ["SISTEER_FIGS_URL_BASE"].rstrip("/") + "/results-registry/") if os.environ.get("SISTEER_FIGS_URL_BASE") else None
#: The SQL name (a v_* object) the "objects in / out of the current round" section renders from.
PAPER_OBJECTS_NAME = "v_paper_objects"
DISK_FLOOR_GB = H.MIN_FREE_BYTES / 1024 ** 3

#: Every this many SQLite VM steps the budget handler reads the clock. The handler runs only while a statement LOOPS over
#: rows: an unrestricted ``SELECT COUNT(*)`` compiles to SQLite's single ``Count`` opcode (~9 steps in all, the handler never
#: called, an expired deadline ignored), so the row counts are ``COUNT(1)`` — Rewind / AggStep / Next per row — which it paces.
PROGRESS_HANDLER_STEPS = 10_000
#: ``<root>/state/status_sections/<name>.json`` — the last good rows of every section, written on each in-budget render and
#: by ``--refresh-heavy``; ``<root>/state/status_heavy.json`` — the cached-with-refresh set and the last heavy refresh;
#: ``<root>/state/status_render.jsonl`` — one line per render (the render-budget record).
SECTIONS_DIR = "status_sections"
HEAVY_STATE = "status_heavy.json"
RENDER_HISTORY = "status_render.jsonl"
#: ≈ 30 days at one render per 10 min; the file is rewritten to this tail once it holds twice as many lines.
RENDER_HISTORY_MAX = 4320
#: the ``[status]`` budget keys; their defaults are :data:`sisteer.registry.config.DEFAULTS`'s
BUDGET_KEYS = ("section_budget_s", "render_budget_s", "heavy_budget_s")
#: ``section_state[name]["state"]``: the query ran within budget; rendered from the cache (over budget or not reached);
#: no cache to render from; the query FAILED (a SQLite error, not the budget — its error row is shown in place, never cached).
STATE_LIVE, STATE_CACHED, STATE_NO_CACHE, STATE_ERROR = "live", "cached", "no_cache", "error"
#: "Cluster reads, store authority and the replica": how many read receipts the section lists (newest first; every receipt
#: still feeds the per-adapter last-whole-read rows); ``state/replica_check.json``'s schema prefix (live_db_slurm.sh writes
#: ``replica_check_state/2``); a maintenance flag older than this is one a window left behind (its trap removes it; the harvester
#: exits at the door while it exists, so its row goes LATE too) and is shown red
RECEIPTS_SHOWN = 10
REPLICA_CHECK_FILE = "replica_check.json"
REPLICA_CHECK_SCHEMA_PREFIX = "replica_check_state/"
MAINTENANCE_STALE_S = 3 * 3600
#: The levels a row or part can take, worst last. ``standing`` is a refusal the cluster read RE-CONFIRMED byte for byte against the
#: group's previous recorded refusal under the same inputs and tree: recorded and shown with the prior refusal it stands on,
#: never an abort and never red -- red is for something NEW that did not read.
LEVEL_RANK = {"green": 0, "standing": 1, "amber": 2, "red": 3}
#: The contract with the cluster-read side, which writes the receipt (ONE exact shape): each adapter entry carries a list
#: ``standing_refusals`` of ``{"group", "state": "standing_refusal", "refusal", "prior_refusal": {"window", "utc", "digest", ...}}`` (extra
#: keys inside ``prior_refusal`` -- ``attempt_revision_id``, ``code_sha`` -- are ignored here); a per-group state string in a groups /
#: results map is the same word. An adapter whose ONLY refusals are standing is written ``read_whole: true`` with ``refused`` None / [] /
#: absent (main's shape for "no refusal"), deliberately, so the tick does not re-read it every cycle; the page therefore decides the
#: newest state by the list, and a non-empty list outranks ``whole`` (a NEW refusal beside it still outranks both, red). The string
#: ``refused_standing`` keeps its ONE existing meaning -- the revision ids a pull left standing UNREAD, string items -- and is
#: never read as a standing refusal.
STANDING_REFUSAL = "standing_refusal"
STANDING_REFUSALS_KEY = "standing_refusals"

QUERIES: Dict[str, Optional[str]] = {
    "in_flight": "SELECT * FROM v_in_flight ORDER BY age_s DESC LIMIT 200",
    "roster_gaps": "SELECT * FROM v_roster_gaps ORDER BY comparison_id, model_id, eval_id",
    # one measurement read by two sources (ownership.MULTI_SOURCE_OWNERSHIP) — every matched identity with its state; a
    # REFUSED identity (the two sources disagree on k / n) is listed first so it is never quiet
    "multi_source": "SELECT declaration_id, state, model_id, paper_label, eval_category, transfer_kind, pairing_key, eval_id, metric_id, threshold, "
                    "value_owner, value_secondary, k_owner, n_owner, k_secondary, n_secondary, degenerate_share_secondary, loop_share_secondary, "
                    "at_cap_share_owner, at_cap_share_secondary, carried_json, owner_rows, secondary_rows, owner_run_id, secondary_run_id, reason "
                    "FROM v_multi_source_check "
                    "ORDER BY CASE state WHEN 'refused' THEN 0 WHEN 'stopgap' THEN 1 ELSE 2 END, declaration_id, pairing_key, eval_id, metric_id, threshold LIMIT 500",
    "refused": "SELECT record_uri, record_type, field, reason, refused_utc, judged_with FROM record_refusal ORDER BY refused_utc DESC LIMIT 200",
    # the durable refusal FACTS (one per refused record x judging stamp, records in the log) beside the current judgment per record;
    # `still_refused` = this fact IS the record's current judgment. The page shows the facts of the last 3 harvest cadences (gather filters
    # by cycle_utc); the whole ledger is `registry refusals list|explain|summary`
    "refusal_facts": "SELECT f.record_uri, f.record_type, f.field, f.reason, f.judged_with, f.refused_utc, f.cycle_utc, "
                     "CASE WHEN r.record_sha256 IS NOT NULL AND COALESCE(r.judged_with, 'unverified') = f.judged_with THEN 1 ELSE 0 END AS still_refused, "
                     "json_extract(f.extra_json, '$.backfilled') AS backfilled "
                     "FROM record_refusal_fact f LEFT JOIN record_refusal r ON r.record_sha256 = f.record_sha256 "
                     "ORDER BY f.cycle_utc DESC, f.refused_utc DESC, f.record_sha256 LIMIT 200",
    "recent_ingests": "SELECT file_uri, status, n_records, n_ok, n_refused, n_blocked, ingested_utc FROM record_delivery "
                      "ORDER BY ingested_utc DESC LIMIT 50",
    "superseded": "SELECT * FROM v_superseded ORDER BY kind, id LIMIT 500",
    "snapshots": "SELECT snapshot_id, head_ingest_seq, built_utc, pushed_to_json, extra_json FROM snapshot ORDER BY built_utc DESC LIMIT 5",
    "unverified_by_campaign": "SELECT campaign, SUM(n_unverified_fields) AS n_unverified, COUNT(*) AS n_objects "
                              "FROM v_object_provenance GROUP BY campaign ORDER BY campaign",
    # the derived decode-policy match (decode_match.stamp_run_rows from v_run_decode_policy_match): the counts read the STAMPED
    # column, so a `no_policy` count > 0 says the derived pass has not run over those rows; the list reads the derivation itself
    "decode_policy_match_counts": "SELECT decode_policy_match AS match, COUNT(*) AS n_runs, COUNT(DISTINCT decode_policy_id) AS n_policies "
                                  "FROM run GROUP BY decode_policy_match ORDER BY decode_policy_match",
    "decode_policy_unmatched": "SELECT v.match, r.run_id, r.campaign, r.model_id, r.eval_id, r.thinking_mode, r.decode_seed_rule, "
                               "r.decode_sent_json, v.candidates_json, v.in_scope_json FROM v_run_decode_policy_match v "
                               "JOIN run r ON r.run_id = v.run_id WHERE v.match IN ('ambiguous', 'no_match') "
                               "ORDER BY v.match, r.campaign, r.model_id, r.eval_id, r.run_id LIMIT 200",
    "instrument_mismatch": "SELECT run_id, model_id, eval_id, differing_fields FROM v_instrument_mismatch ORDER BY run_id LIMIT 200",
    # the best-of-N decode precedence guard (ruling r2v4-reference-bon-climb-decode-precedence-vendor-preset-over-t1): every steered
    # key where a selected T = 1 climb row sits beside a vendor-preset row; `both_selected` is the violation, `vendor_not_in_roster` pending
    "bon_decode_precedence": DP.PRECEDENCE_SQL,
    "blocked": "SELECT record_uri, record_type, missing_parent_table, missing_parent_id, first_seen_utc, attempts, stale "
               "FROM blocked_records ORDER BY first_seen_utc LIMIT 200",
    "mirror_acks": "SELECT destination, acked_through_seq, manifest_name, file_count, bytes, acked_utc, last_error FROM mirror_ack "
                   "ORDER BY destination",
    # the held reads measured on 2026-10-06 (objects_bank 371, regrade_twin 155, em_misalign 11, sampled_static 4 of 690 pending rows):
    # every source_revision an adapter pass wrote accepted = 0 that no later revision superseded, grouped by adapter and by the HEAD of its
    # refusal (the text up to the first `: `), with the count, the sources, the latest observation and the newest refusal in full
    "pending_revisions": "WITH pending AS (SELECT adapter, source_id, observed_utc, refusal, "
                         "CASE WHEN instr(COALESCE(refusal, ''), ': ') > 0 THEN substr(refusal, 1, instr(refusal, ': ') - 1) "
                         "ELSE COALESCE(refusal, '(no refusal text)') END AS reason "
                         "FROM source_revision WHERE accepted = 0 AND superseded_by_revision_id IS NULL) "
                         "SELECT p.adapter, p.reason, COUNT(*) AS n_revisions, COUNT(DISTINCT p.source_id) AS n_sources, MAX(p.observed_utc) AS last_observed, "
                         "(SELECT q.refusal FROM pending q WHERE q.adapter = p.adapter AND q.reason = p.reason ORDER BY q.observed_utc DESC LIMIT 1) AS example "
                         "FROM pending p GROUP BY p.adapter, p.reason ORDER BY n_revisions DESC, p.adapter, p.reason",
    # the groups an adapter read and PARKED by name -- a bundle of a kind ruled out of scope, read whole, accepted with no record,
    # the ruling on its status row (`extra_json.parked`, `ok = 0`). Per group, off the per-group status rows (the per-source-string
    # aggregate folds an adapter's groups into one row); one line per parked bundle
    "parked_sources": "SELECT s.adapter, s.source_id, s.lane_source_legacy, s.observed_utc, json_extract(p.value, '$.dir') AS dir, "
                      "json_extract(p.value, '$.kind_legacy') AS kind_legacy, json_extract(p.value, '$.status_ruled') AS status_ruled, "
                      "json_extract(p.value, '$.object_role_ruled') AS object_role_ruled, json_extract(p.value, '$.reason') AS reason, "
                      "json_extract(p.value, '$.ruling') AS ruling "
                      "FROM v_source_status_current s, json_each(json_extract(s.extra_json, '$.parked')) p "
                      "WHERE s.ok = 0 AND json_type(s.extra_json, '$.parked') = 'array' ORDER BY s.adapter, kind_legacy, dir LIMIT 500",
    "adapter_pass": "SELECT adapter, source_id, COUNT(*) AS n_inputs, MAX(observed_utc) AS last_observed, "
                    "SUM(records_written) AS records_written, MAX(last_error) AS last_error FROM source_watermark "
                    "GROUP BY adapter, source_id ORDER BY adapter, source_id",
    "degeneration_not_measured": "SELECT e.eval_category, d.measure, COUNT(*) AS n_run_measure_pairs FROM v_run_degeneration_current d "
                                 "JOIN run r ON r.run_id = d.run_id JOIN eval e ON e.eval_id = r.eval_id "
                                 "WHERE d.measurement_state = 'not_measured' GROUP BY e.eval_category, d.measure ORDER BY e.eval_category, d.measure",
    "dispositions": "SELECT d.round_id, d.comparison_id, d.model_id, d.object_id, d.eval_id, d.state, d.reason, d.ruled_utc FROM disposition d "
                    "ORDER BY d.ruled_utc DESC LIMIT 200",
    "item_conflicts": "SELECT run_id, item_id, metric_id, judgment_id, n_provenances, n_distinct_values, value_min, value_max, revision_ids_json "
                      "FROM v_item_conflict ORDER BY run_id, item_id, metric_id LIMIT 200",
    "paper_objects": "SELECT object_id, paper_label, object_kind, in_paper, n_runs, n_runs_in_paper, comparison_ids_json, excluding_ruling_ids_json, "
                     "run_excluding_ruling_ids_json FROM v_paper_objects ORDER BY in_paper DESC, paper_label, object_id LIMIT 500",
    "parity_gates": "SELECT gate_id, paper_sha, level, result, registry_ingest_seq, run_utc, report_uri FROM parity_gate ORDER BY run_utc DESC LIMIT 20",
    # the analyses filed over ingested runs and objects (`registry analysis file`; ledger facts under records/analyses/<day>/), newest
    # first, with their input and artifact counts, the report and the published pages' urls (from the artifact links' extra_json.page_url
    # and the page_url of every `same_bytes` entry, analysis.PAGE_URLS_SQL: a page whose files are all another page's bytes is named by
    # the entries alone -- the closing pass, 2026-10-09)
    "analyses": "SELECT a.analysis_id, a.name, a.code_sha, a.substrate, a.filed_utc, a.filed_by, "
                "COALESCE(json_extract(a.extra_json, '$.provenance.kind'), 'in_filed_by_prose') AS provenance, "
                "json_extract(a.extra_json, '$.provenance.relayed_by') AS relayed_by, "
                "(SELECT COUNT(*) FROM analysis_input i WHERE i.analysis_id = a.analysis_id AND i.input_kind = 'run') AS n_runs, "
                "(SELECT COUNT(*) FROM analysis_input i WHERE i.analysis_id = a.analysis_id AND i.input_kind = 'object') AS n_objects, "
                "(SELECT COUNT(*) FROM analysis_artifact x WHERE x.analysis_id = a.analysis_id) AS n_artifacts, a.report_uri, "
                f"({_AN_PAGE_URLS_SQL}) AS page_urls "
                "FROM analysis a ORDER BY a.filed_utc DESC, a.name LIMIT 200",
    # every ruling of the open round with where it came from — `relayed` (the words reached the registry second-hand and the
    # reading has not been confirmed) is told apart from `user_direct`; the seed's earlier rulings say `in_ruled_by_prose`
    "rulings": "SELECT ru.ruling_id, ru.action, ru.scope_kind, "
               "COALESCE(json_extract(ru.extra_json, '$.provenance.kind'), 'in_ruled_by_prose') AS provenance, "
               "json_extract(ru.extra_json, '$.provenance.relayed_by') AS relayed_by, "
               "json_extract(ru.extra_json, '$.provenance.message_utc') AS message_utc, "
               "COALESCE(json_extract(ru.extra_json, '$.provenance.confirmed_by_user'), '') AS confirmed_by_user, "
               "ru.ruled_utc, ru.ruled_by, ru.evidence_state, ru.expired_utc, "
               "(SELECT COUNT(*) FROM v_paper_current pc WHERE pc.decisive_ruling_id = ru.ruling_id) AS n_runs_decided "
               "FROM ruling ru WHERE ru.round_id = (SELECT round_id FROM paper_round WHERE closed_utc IS NULL ORDER BY opened_utc DESC LIMIT 1) "
               "ORDER BY CASE WHEN json_extract(ru.extra_json, '$.provenance.kind') = 'relayed' THEN 0 ELSE 1 END, ru.ruled_utc DESC, ru.ruling_id LIMIT 300",
    "table_counts": None,  # computed
    "head": "SELECT ingest_seq FROM v_paper_inputs_head",
    "index_head": "SELECT COALESCE(MAX(ingest_seq), 0) AS ingest_seq FROM record_ingest",
    "pending": "SELECT COUNT(*) FROM v_pending_judgment WHERE is_instrument_judge = 1",
    "open_round": "SELECT round_id, opened_utc FROM paper_round WHERE closed_utc IS NULL ORDER BY opened_utc DESC LIMIT 1",
}


class SectionOverBudget(Exception):
    """A section query was interrupted at its deadline (the progress handler returned non-zero)."""


@contextlib.contextmanager
def _under_deadline(conn: sqlite3.Connection, deadline: float, what: str) -> Iterator[None]:
    """Run the statements inside under ``deadline`` (``time.monotonic()``): an already-expired deadline raises before anything
    runs; past it, SQLite's progress handler (every :data:`PROGRESS_HANDLER_STEPS` VM steps) interrupts the statement and the
    ``interrupted`` error becomes :class:`SectionOverBudget`. Any other SQLite error passes through to the caller."""
    if time.monotonic() >= deadline:
        raise SectionOverBudget(what)
    conn.set_progress_handler(lambda: 1 if time.monotonic() >= deadline else 0, PROGRESS_HANDLER_STEPS)
    try:
        yield
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise SectionOverBudget(what) from exc
        raise
    finally:
        conn.set_progress_handler(None, 0)


def _rows_within(conn: sqlite3.Connection, sql: str, deadline: float) -> List[Dict[str, object]]:
    """The rows of ``sql`` under ``deadline``; :class:`SectionOverBudget` on the interrupt, the error row (``[{"error": …}]``,
    the page's in-band convention) for any other SQLite error."""
    try:
        with _under_deadline(conn, deadline, sql):
            return [dict(r) for r in conn.execute(sql).fetchall()]
    except sqlite3.Error as exc:
        return [{"error": str(exc)}]


def _count_within(conn: sqlite3.Connection, table: str, deadline: float) -> Optional[int]:
    """The row count of one table under ``deadline`` — ``COUNT(1)``, a row loop the progress handler paces (``COUNT(*)`` is one
    opcode it never reaches); None when the table cannot be counted, :class:`SectionOverBudget` on the interrupt."""
    rows = _rows_within(conn, f"SELECT COUNT(1) AS n FROM {table}", deadline)
    return None if "error" in rows[0] else int(rows[0]["n"])


def _section_rows(conn: sqlite3.Connection, name: str, deadline: float, has_paper_objects: bool) -> Tuple[List[Dict[str, object]], Optional[str]]:
    """The rows of a row section under ``deadline``: ``(rows, None)``, or ``(the error row, its message)`` when the query failed —
    the page shows the error row in place (its convention) and it is never a section's last good rows. ``paper_objects`` is
    ``[]`` while its relation is absent. :class:`SectionOverBudget` at the deadline."""
    if name == "paper_objects" and not has_paper_objects:
        return [], None
    rows = _rows_within(conn, QUERIES[name], deadline)   # type: ignore[arg-type]
    if rows and "error" in rows[0]:
        return rows, str(rows[0]["error"])
    return rows, None


def _first_row(ctx: Mapping[str, object], name: str) -> Optional[Dict[str, object]]:
    """The first row of a one-row section; None when it is empty or the error row."""
    rows = ctx.get(name) or []
    return rows[0] if rows and "error" not in rows[0] else None  # type: ignore[index, return-value]


# --------------------------------------------------------------------------------------------
# the section cache, the cached-with-refresh set, the render history (state files under <root>/state/)
# --------------------------------------------------------------------------------------------


def status_config(root: str) -> Dict[str, float]:
    """The ``[status]`` budgets (seconds): config.toml's values over :data:`sisteer.registry.config.DEFAULTS` (``config.load``
    fills an absent file or key); a file the render cannot read leaves every default, so the page still renders."""
    out = {k: float(CFG.DEFAULTS["status"][k]) for k in BUDGET_KEYS}
    try:
        st = CFG.load(root).get("status") or {}
    except (OSError, ValueError):   # unreadable, or malformed TOML (TOMLDecodeError is a ValueError)
        return out
    out.update({k: float(st[k]) for k in BUDGET_KEYS if st.get(k) is not None})
    return out


def sections_dir(root: str) -> str:
    return os.path.join(root, "state", SECTIONS_DIR)


def section_cache_path(root: str, name: str) -> str:
    return os.path.join(sections_dir(root), f"{name}.json")


def heavy_state_path(root: str) -> str:
    return os.path.join(root, "state", HEAVY_STATE)


def render_history_path(root: str) -> str:
    return os.path.join(root, "state", RENDER_HISTORY)


def _write_atomic(path: str, text: str) -> None:
    """Write ``text`` to ``path`` through a per-process temp name + ``os.replace``: a reader sees the old file or the new one."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _write_json_atomic(path: str, payload: Mapping[str, object]) -> None:
    _write_atomic(path, json.dumps(payload, default=str))


def read_section_cache(root: str, name: str) -> Optional[Dict[str, object]]:
    """The last good rows of a section: ``{"name", "as_of", "rows" | "counts" (+ "counts_as_of"), "elapsed_ms", "source"}``."""
    return _read_json(section_cache_path(root, name))


def write_section_cache(root: str, name: str, payload: Mapping[str, object]) -> None:
    _write_json_atomic(section_cache_path(root, name), dict(payload, name=name))


def read_heavy_state(root: str) -> Dict[str, object]:
    """``{"members": [...], "head": <ingest_seq at the last refresh>, "refreshed_utc", "last_refresh": {name: {...}}, "initialized"}``."""
    st = _read_json(heavy_state_path(root)) or {}
    st.setdefault("members", [])
    st.setdefault("last_refresh", {})
    return st


def update_heavy_state(root: str, add: Sequence[str] = (), remove: Sequence[str] = (), **fields: object) -> Dict[str, object]:
    """Read-modify-write of ``status_heavy.json`` in one short step: the render adds the sections that ran over budget and
    removes the ones that came back in budget; the heavy refresh removes the members it cleared and stamps the head. Both
    writers touch the file only through this, so a member added mid-refresh survives the refresh's final write."""
    st = read_heavy_state(root)
    members = set(st.get("members") or [])
    members |= set(add)
    members -= set(remove)
    st["members"] = sorted(members)
    st.update(fields)
    _write_json_atomic(heavy_state_path(root), st)
    return st


def append_render_history(root: str, line: Mapping[str, object], max_lines: Optional[int] = None) -> int:
    """Append one render to ``status_render.jsonl``; once the file holds more than twice ``max_lines`` lines it is rewritten
    to its last ``max_lines`` (an atomic tail rewrite). Returns the line count after the write."""
    max_lines = RENDER_HISTORY_MAX if max_lines is None else int(max_lines)
    path = render_history_path(root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, default=str, separators=(",", ":")) + "\n")
    with open(path, "rb") as fh:
        n = sum(1 for _ in fh)
    if n > 2 * max_lines:
        with open(path, "r", encoding="utf-8") as fh:
            tail = fh.readlines()[-max_lines:]
        _write_atomic(path, "".join(tail))
        n = len(tail)
    return n


def read_render_history(root: str) -> List[Dict[str, object]]:
    path = render_history_path(root)
    if not os.path.exists(path):
        return []
    out: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except ValueError:
                continue
    return out


def _read_json(path: str) -> Optional[Dict[str, object]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _read_text(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return None


def _age_s(utc: Optional[str], now_s: float) -> Optional[int]:
    if not utc:
        return None
    try:
        return int(now_s - R.utc_to_epoch_ms(str(utc)) / 1000.0)
    except (ValueError, TypeError):
        return None


#: The harvester row turns LATE (red) when the last completed cycle is older than this many harvest cadences ([harvest] cadence_s).
HARVEST_LATE_CADENCES = 3


def harvest_cycle_section(root: Optional[str], now_s: Optional[float] = None) -> Dict[str, object]:
    """The status page's harvester row: the age of the last COMPLETED harvester cycle (``state/harvest_cycle.json``, written at the end
    of every cycle that ran to completion -- a cycle its unit killed writes nothing), its state -- ``ok``, ``late`` (older than
    :data:`HARVEST_LATE_CADENCES` x ``[harvest] cadence_s``, the page's red), ``unknown`` (no completed cycle recorded yet) -- and the
    cycle's own facts (wall time, whether its budget stopped it, what it left). ``state/last_run/ingest.json`` beside it stamps every
    exit of the command, a schema refusal included; this row is the fold's own completion."""
    now_s = time.time() if now_s is None else now_s
    cadence = float(CFG.DEFAULTS["harvest"]["cadence_s"])
    if root:
        try:
            cadence = float(CFG.load(root)["harvest"].get("cadence_s") or cadence)
        except Exception:  # noqa: BLE001 -- a torn config renders the default cadence, never a page that fails
            pass
    late_after = HARVEST_LATE_CADENCES * cadence
    cycle = (H.read_cycle_state(root) or {}) if root else {}
    last = cycle.get("last_completed_cycle_utc")
    age = _age_s(str(last) if last else None, now_s)
    state = "unknown" if age is None else ("late" if age > late_after else "ok")
    return {"last_completed_cycle_utc": last, "age_s": age, "cadence_s": int(cadence), "late_after_s": int(late_after), "state": state,
            "late": state == "late", "cycle_utc": cycle.get("cycle_utc"), "elapsed_s": cycle.get("elapsed_s"), "stopped": cycle.get("stopped"),
            "budget_s": cycle.get("budget_s"), "files": cycle.get("files"), "revisions_applied": cycle.get("revisions_applied"),
            # what the last completed cycle refused and how many refusal facts it wrote (its START + END passes)
            "refused": cycle.get("refused"), "refusal_facts": cycle.get("refusal_facts"),
            "revisions_held": cycle.get("revisions_held"), "revisions_held_skipped": cycle.get("revisions_held_skipped"),
            "revisions_deferred_budget": cycle.get("revisions_deferred_budget"),
            "plain_files_deferred_budget": cycle.get("plain_files_deferred_budget")}


def wal_size_bytes(root: Optional[str]) -> int:
    """The size of ``registry.sqlite-wal`` under ``root`` in bytes, 0 when absent (checkpoint starvation -- 13.7 GB of WAL on 2026-10-05,
    every read and write paying the WAL-index cost -- is visible on the page, beside the last completed harvester cycle, the one event
    that checkpoints it; the landing window checkpoints after its drain too)."""
    if not root:
        return 0
    try:
        return int(os.path.getsize(DB.db_path(root) + "-wal"))
    except OSError:
        return 0


def sql_name_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'view' AND name = ?", (name,)).fetchone() is not None


def _read_state_doc(path: str) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    """``(document, None)`` for a readable JSON object at ``path``; ``(None, None)`` when there is no file; ``(None, <why>)`` when the file
    is there but unreadable or not a JSON object -- a torn state file is SHOWN, never read as absent."""
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        return None, f"{os.path.basename(path)} is unreadable or not JSON ({type(exc).__name__})"
    if not isinstance(doc, dict):
        return None, f"{os.path.basename(path)} is not a JSON object"
    return doc, None


def _age_s_any(utc: Optional[str], now_s: float) -> Optional[int]:
    """:func:`_age_s` for the registry's millisecond stamps, AND the second-resolution ``YYYY-MM-DDTHH:MM:SSZ`` the shell writers spell
    (``window.sh``'s maintenance flag, ``live_db_slurm.sh``'s replica verdict: ``date -u +%FT%TZ``); None for anything else."""
    age = _age_s(utc, now_s)
    if age is not None or not utc:
        return age
    try:
        from datetime import datetime, timezone
        dt = datetime.strptime(str(utc).strip(), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return int(now_s - dt.timestamp())
    except ValueError:
        return None


def _worst(*levels: Optional[str]) -> str:
    return max((lv for lv in levels if lv in LEVEL_RANK), key=LEVEL_RANK.__getitem__, default="green")


def _authority_section(root: str, now_s: float) -> Dict[str, object]:
    """Per component: the last `stores check-authority` reading of its `current` (``state/store_authority.json``) joined with the last
    `stores` command's outcome for the component as the mirror state records it (``state/store_mirror.json`` ``last_run.outcome``: a
    `stores mirror` / `push` outcome, `stores check` as ``check:<state>``, the one-time `stores verify-authority` as
    ``verify-authority:<state>``; the row keeps the ``mirror_*`` names because the FILE is the mirror state). Rows exist for every
    component either file names; a component the check never read carries the mirror's facts alone (``checked`` false). Level per
    row: red on any authority problem (dangling, unverified, unsealed, no published record, no current, no grading host mirror) or a red last
    outcome; amber on an overdue carry, an amber outcome or a run's in-flight stamp; green otherwise. The outcome's level is
    :func:`stores.last_run_level`'s, by the command that stamped it -- a matching ``check:match`` and a verified move
    ``verify-authority:verified`` are green; the first cut read every stamp as a mirror outcome and called both red."""
    from . import stores as ST
    # the state files may carry keys written under the site's older names: the root's alias table maps them. A refused alias file is
    # named in this section; a refused config.toml is the header's to name, and the rows still render from the state files either way
    alias_err = None
    try:
        ST.install_site_aliases(root)
    except SA.SiteAliasError as exc:
        alias_err = f"{SA.ALIASES_NAME} is refused, so state written under the site's older names is not read: {exc}"
    except (ValueError, OSError):
        pass
    auth, auth_err = _read_state_doc(os.path.join(root, "state", ST.AUTHORITY_STATE_FILE))
    if auth is not None and auth.get("schema") != ST.AUTHORITY_STATE_SCHEMA:
        auth, auth_err = None, f"{ST.AUTHORITY_STATE_FILE} carries schema {auth.get('schema')!r}, not {ST.AUTHORITY_STATE_SCHEMA}"
    mirror, mirror_err = _read_state_doc(ST.state_path(root))
    mirror_comps: Dict[str, Dict[str, object]] = dict((mirror or {}).get("components") or {})   # type: ignore[arg-type]
    auth_comps: Dict[str, Dict[str, object]] = dict((auth or {}).get("components") or {})         # type: ignore[arg-type]
    rows: List[Dict[str, object]] = []
    for slug in sorted(set(auth_comps) | set(mirror_comps)):
        a = auth_comps.get(slug) or {}
        m = mirror_comps.get(slug) or {}
        last_run = m.get("last_run") or {}
        gen = m.get("generation") or {}
        outcome = str(last_run.get("outcome") or "") or None
        mirror_level = ST.last_run_level(outcome)
        problems = [str(x) for x in (a.get("problems") or [])]
        notes = [str(x) for x in (a.get("notes") or [])]
        checked = bool(a)
        not_applicable = a.get("state") == "not_applicable"
        if not_applicable:
            level = mirror_level or "green"
        else:
            level = _worst("red" if problems else None, "amber" if a.get("carry_overdue") else None, mirror_level)
        rows.append({"slug": slug, "authority": ST.canonical_authority(a.get("authority") or m.get("authority")), "producer": a.get("producer") or m.get("producer"),
                     "checked": checked, "current": a.get("current"), "exists": a.get("exists"), "verified": a.get("verified"),
                     "sealed": a.get("sealed"), "published_record": a.get("published_record"), "published_utc": a.get("published_utc"),
                     "change_utc": a.get("change_utc"), "change_age_s": _age_s_any(a.get("change_utc"), now_s), "publish_mode": a.get("publish_mode"),
                     "grading_host_mirror_present": SA.get_compat(a, "grading_host_mirror_present"),
                     "transitional_carry": bool(a.get("transitional_carry") or m.get("transitional_carry")),
                     "carry_deadline": a.get("carry_deadline") or m.get("carry_deadline"), "carry_overdue": bool(a.get("carry_overdue")),
                     "problems": ", ".join(problems), "notes": "; ".join(notes), "state": a.get("state") if checked else "unchecked",
                     "mirror_outcome": outcome, "mirror_utc": last_run.get("utc"), "mirror_level": mirror_level,
                     "generation_state": gen.get("state"), "generation_sha8": gen.get("content_sha8"), "published": m.get("published"),
                     "level": level})
    problems_all = [str(x) for x in (auth or {}).get("problems") or []]
    return {"present": auth is not None, "checked_utc": (auth or {}).get("checked_utc"), "checked_age_s": _age_s_any((auth or {}).get("checked_utc"), now_s),
            "ok": (auth or {}).get("ok"), "destination": (auth or {}).get("destination") or (mirror or {}).get("destination"),
            "n_components": len(rows), "n_problems": len(problems_all), "problems": problems_all, "counts": (auth or {}).get("counts") or {},
            "mirror_present": mirror is not None, "mirror_updated_utc": (mirror or {}).get("updated_utc"),
            "error": "; ".join(e for e in (auth_err, mirror_err, alias_err) if e) or None, "rows": rows,
            "level": _worst("red" if (auth_err or mirror_err or alias_err) else None, *(str(r["level"]) for r in rows))}


def _replica_section(root: str, now_s: float) -> Dict[str, object]:
    """The last Slurm live-DB step's verdict (``state/replica_check.json``, written by live_db_slurm.sh): equal or not, ok or the problems
    by name, the head's file count, both projection sha12s, the projection version, the generation and schema version. Red unless ok."""
    doc, err = _read_state_doc(os.path.join(root, "state", REPLICA_CHECK_FILE))
    if doc is not None and not str(doc.get("schema") or "").startswith(REPLICA_CHECK_SCHEMA_PREFIX):
        doc, err = None, f"{REPLICA_CHECK_FILE} carries schema {doc.get('schema')!r}, not {REPLICA_CHECK_SCHEMA_PREFIX}*"
    if doc is None:
        return {"present": False, "error": err, "level": "red" if err else "green"}
    head = doc.get("head") or {}
    rep = doc.get("replica") or {}
    ok = bool(doc.get("ok"))
    return {"present": True, "error": None, "utc": doc.get("utc"), "age_s": _age_s_any(doc.get("utc"), now_s), "equal": bool(doc.get("equal")), "ok": ok,
            "headline": doc.get("headline"), "problems": [str(p) for p in (doc.get("problems") or [])], "branch": doc.get("branch"),
            "tip": doc.get("tip"), "job": doc.get("job"), "job_state": doc.get("job_state"), "head_n_files": head.get("n_files"),
            "head_taken_utc": head.get("taken_utc"), "head_sha12": head.get("projection_sha12"), "replica_sha12": rep.get("projection_sha12"),
            "replica_counts": rep.get("counts"), "projection_version": doc.get("projection_version"), "generation": doc.get("generation"),
            "current_generation": doc.get("current_generation"), "parent_generation": doc.get("parent_generation"),
            "schema_version": doc.get("schema_version"), "pytest": doc.get("pytest") or {}, "reason": doc.get("reason"),
            # the cutover allowance: the rule that passed the check and what it covered
            "c1_class": bool(doc.get("c1_class")), "accepted_as": doc.get("accepted_as"), "n_movers": doc.get("n_movers"), "declaration": doc.get("declaration"),
            "level": "green" if ok else "red"}


def _reservation_section(rd: str, now_s: float) -> Dict[str, object]:
    """Who holds the cluster-read namespace (``$STATE/reads/reservation.json``): owner, operation, host, pid, since; ``live`` when the
    holder process exists on this host (``reads_protocol.holder_alive``), ``stale`` when it is gone (the next operation replaces it),
    ``unknown`` on another host. A live holder is the normal state of a tick or window in flight (green); stale is amber; a torn file red."""
    from . import reads_protocol as RP
    path = RP.reservation_path(rd)
    if not os.path.exists(path):
        return {"held": False, "reads_dir": rd, "level": "green"}
    res = RP.read_reservation(rd)
    if res is None:
        return {"held": True, "reads_dir": rd, "error": "reservation.json is unreadable or not JSON", "state": "unreadable", "level": "red"}
    alive = RP.holder_alive(res)
    state = "live" if alive else ("stale" if alive is False else "unknown")
    return {"held": True, "reads_dir": rd, "owner": res.get("owner"), "op_id": res.get("op_id"), "host": res.get("host"), "pid": res.get("pid"),
            "started_utc": res.get("started_utc"), "age_s": _age_s_any(res.get("started_utc"), now_s), "tree_sha": res.get("tree_sha"),
            "job": res.get("job"), "alive": alive, "state": state, "error": None,
            "level": "green" if alive else "amber"}


def standing_refusals(entry: Mapping[str, object]) -> List[Dict[str, object]]:
    """The receipt entry's STANDING refusals: each a group whose refusal the cluster read re-confirmed byte for byte against the group's
    previous recorded refusal under the same inputs and tree -- recorded, never an abort. Read from the ONE shape the contract names:
    the entry's ``standing_refusals`` list, each item ``{"group", "state": "standing_refusal", "refusal", "prior_refusal": {"window",
    "utc", "digest"}}`` (an item whose ``state`` is another word is not one; an older receipt carries no list and yields none). Each comes
    back as ``{"group", "refusal", "prior"}``, ``prior`` = ``"<window> at <utc>"`` of the recorded refusal it stands on."""
    out: List[Dict[str, object]] = []
    for it in entry.get(STANDING_REFUSALS_KEY) or []:
        if not isinstance(it, Mapping) or str(it.get("state")) != STANDING_REFUSAL:
            continue
        prior = it.get("prior_refusal") if isinstance(it.get("prior_refusal"), Mapping) else {}
        out.append({"group": it.get("group"), "refusal": str(it.get("refusal") or ""),
                    "prior": " at ".join(str(prior.get(k)) for k in ("window", "utc") if prior.get(k))})
    return out


def standing_unread_ids(entry: Mapping[str, object]) -> List[str]:
    """The refused-known ids a pull left standing UNREAD (``refused_standing`` string items -- that string's one meaning) and the
    retirements it WITHHELD (``withheld``: the shard's claim was not the live head): the group was neither read nor committed, so the
    entry is not whole -- its adapter's newest result is "not read whole", whatever an older receipt read."""
    return [str(x) for x in (entry.get("refused_standing") or [])] + [str(x) for x in (entry.get("withheld") or [])]


def newest_entry_state(entry: Mapping[str, object], read_whole: bool) -> Tuple[str, str]:
    """``(state, level)`` of an adapter's NEWEST receipt entry -- what the adapter row's color rests on, in this order: ``refused`` (red:
    something NEW did not read; outranks everything); ``partial`` (amber: a shard missing or not ok, no refusal recorded); ``standing_unread``
    (amber: ids left standing unread -- this used to read GREEN when an older receipt had read the adapter whole);
    :data:`STANDING_REFUSAL` (standing: a non-empty ``standing_refusals`` list -- REGARDLESS of ``read_whole``, which the cluster-read side
    writes true for an adapter whose only refusals stand, so the tick leaves it alone; standing outranks whole); ``whole`` (green);
    ``not_attempted`` (routed off the cluster with no refusal: the level falls back to whether the adapter was ever read whole)."""
    total = int(entry.get("shards_total") or 0)
    ok = int(entry.get("shards_ok") or 0)
    if entry.get("refused"):
        return "refused", "red"
    if total > 0 and ok < total:
        return "partial", "amber"
    if standing_unread_ids(entry):
        return "standing_unread", "amber"
    if standing_refusals(entry):
        return STANDING_REFUSAL, "standing"
    if read_whole:
        return "whole", "green"
    return "not_attempted", ""


def _standing_line(x: Mapping[str, object], width: int = 120) -> str:
    """One standing refusal as the page shows it: the group, the refusal text, and the recorded refusal it stands on."""
    text = f"{x['group']}: {str(x['refusal'])[:width]}" if x.get("refusal") else str(x["group"])
    return f"{text} (stands on {x['prior']})" if x.get("prior") else text


def _standing_text(entries: Mapping[str, Mapping[str, object]]) -> str:
    parts = []
    for a, e in sorted(entries.items()):
        st = standing_refusals(e)
        if st:
            parts.append(f"{a}: " + "; ".join(_standing_line(x) for x in st))
    return " | ".join(parts)


def _receipts_section(rd: str, now_s: float) -> Dict[str, object]:
    """The latest read receipts (``$STATE/reads/receipts/``; the newest :data:`RECEIPTS_SHOWN` listed) and, from EVERY receipt, the
    per-adapter LAST WHOLE READ (every shard ok, no refusal -- ``reads_protocol.read_whole``; what the tick's changed set reads)
    with its pins and the newest ``published_utc`` / ``change_utc`` among them, plus each adapter's NEWEST result
    (:func:`newest_entry_state`). A receipt row is red when it refused an adapter or stopped at a pull barrier, ``standing`` when its only
    non-whole results are re-confirmed refusals (recorded, with the prior refusal each stands on, never an abort), and KEEPS that
    color while it is listed (history); the section's ``level`` is its CURRENT health -- the newest receipt, every adapter's newest result
    (red: its newest receipt refused it; amber: never read whole, or its newest read did not read it whole -- shards missing, ids left
    standing unread -- whatever an older whole receipt says; standing: a re-confirmed refusal) and the receipt
    files that would not read -- never the worst of the ten listed rows, which held the tile red for nine more ticks after a recovery,
    with no headline to explain it. An adapter refused in an older receipt that no later receipt read at all is still red:
    that refusal IS its newest result."""
    from . import reads_protocol as RP
    d = os.path.join(rd, RP.RECEIPTS_SUBDIR)
    n_files = sum(1 for fn in os.listdir(d) if fn.endswith(".json")) if os.path.isdir(d) else 0
    recs = RP.receipts(rd)
    rows: List[Dict[str, object]] = []
    for rec in recs[:RECEIPTS_SHOWN]:
        op = rec.get("op") or {}
        adapters = rec.get("adapters") or {}
        refused = {a: e.get("refused") for a, e in adapters.items() if e.get("refused")}
        n_standing = sum(1 for e in adapters.values() if standing_refusals(e))
        rows.append({"op_id": op.get("op_id"), "owner": op.get("owner"), "utc": rec.get("utc"), "age_s": _age_s_any(rec.get("utc"), now_s),
                     "window": rec.get("window"), "tree_sha8": str(rec.get("tree_sha") or "")[:8] or None, "n_adapters": len(adapters),
                     "n_read_whole": rec.get("n_read_whole"), "n_refused": rec.get("n_refused"), "n_standing_refusals": n_standing,
                     "refused": "; ".join(f"{a}: {str((why or ['refused'])[0])[:160]}" for a, why in sorted(refused.items())),
                     "standing_refusals": _standing_text(adapters),
                     "barrier": rec.get("barrier"), "level": "red" if refused or rec.get("barrier") else ("standing" if n_standing else "green")})
    last_whole: Dict[str, Dict[str, object]] = {}
    newest: Dict[str, Dict[str, object]] = {}
    for rec in recs:                       # newest first: the first entry per adapter is its newest receipt
        for a, e in (rec.get("adapters") or {}).items():
            if a not in newest:
                newest[a] = {"op_id": (rec.get("op") or {}).get("op_id"), "utc": rec.get("utc"), "refused": e.get("refused"), "entry": e}
            if a not in last_whole and RP.read_whole(e):
                pub = {k: v for k, v in (e.get("published") or {}).items() if isinstance(v, dict)}
                last_whole[a] = {"adapter": a, "op_id": (rec.get("op") or {}).get("op_id"), "utc": rec.get("utc"), "age_s": _age_s_any(rec.get("utc"), now_s),
                                 "pins": ", ".join(f"{s}@{g}" for s, g in sorted((e.get("pins") or {}).items())),
                                 "published_utc": max((str(v.get("published_utc")) for v in pub.values() if v.get("published_utc")), default=None),
                                 "change_utc": max((str(v.get("change_utc")) for v in pub.values() if v.get("change_utc")), default=None),
                                 "shards": f"{e.get('shards_ok')}/{e.get('shards_total')}", "revisions_pulled": len(e.get("revisions_pulled") or [])}
    adapter_rows: List[Dict[str, object]] = []
    for a in sorted(set(newest) | set(last_whole)):
        lw = last_whole.get(a)
        nw = newest[a]
        e = nw["entry"]
        row = dict(lw) if lw else {"adapter": a, "op_id": None, "utc": None, "age_s": None, "pins": "", "published_utc": None, "change_utc": None,
                                   "shards": None, "revisions_pulled": None}
        state, level = newest_entry_state(e, RP.read_whole(e))
        standing = standing_refusals(e)
        row["newest_refused"] = ("; ".join(str(x)[:160] for x in nw["refused"]) if nw.get("refused") else "")
        row["newest_op_id"] = nw["op_id"]
        row["newest_state"] = state
        row["standing_on"] = "; ".join(_standing_line(x, 160) for x in standing)
        row["standing_unread"] = ", ".join(standing_unread_ids(e))
        row["level"] = level or ("amber" if lw is None else "green")
        if level == "green" and lw is None:
            row["level"] = "amber"          # never read whole (cannot happen with a whole newest entry; kept for the invariant)
        adapter_rows.append(row)
    newest_row_level = str(rows[0]["level"]) if rows else None       # the older rows keep their color but do not set the level
    return {"present": bool(recs), "n_receipts": len(recs), "n_unreadable": max(0, n_files - len(recs)), "rows": rows, "adapters": adapter_rows,
            "level": _worst("red" if n_files > len(recs) else None, newest_row_level, *(str(r["level"]) for r in adapter_rows))}


def _maintenance_section(root: str, now_s: float) -> Dict[str, object]:
    """``state/maintenance.json`` (a landing window's drain flag): set or not, since when, why, by which window; red once older than
    :data:`MAINTENANCE_STALE_S` -- a window's trap removes it, so an old flag is a window that died before its trap ran, and the
    harvester exits at the door while it exists."""
    flag = H.maintenance(root)
    if flag is None:
        return {"set": False, "level": "green"}
    age = _age_s_any(str(flag.get("since_utc")) if flag.get("since_utc") else None, now_s)
    stale: Optional[bool] = None if age is None else age > MAINTENANCE_STALE_S     # an unparseable stamp: the age is unknown, not stale
    return {"set": True, "since_utc": flag.get("since_utc"), "age_s": age, "reason": flag.get("reason"), "window": flag.get("window") or flag.get("window_id"),
            "adapters": flag.get("adapters"), "pid": flag.get("pid"), "stale": stale, "stale_after_s": MAINTENANCE_STALE_S,
            "level": "red" if stale else "amber"}


def _config_section(root: str) -> Dict[str, object]:
    """Does ``config.toml`` still carry the removed table (``config.REMOVED_TABLE``)? Read with ``tomllib`` directly -- never through
    ``config.load``, which refuses such a file -- so the page can name the refusal every registry command would make. A file that does
    not parse is named too (the loader refuses it as well)."""
    import tomllib
    path = CFG.config_path(root)
    if not os.path.exists(path):
        return {"present": False, "path": path, "removed_table": False, "table_keys": [], "error": None, "level": "green"}
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, ValueError) as exc:   # TOMLDecodeError is a ValueError
        return {"present": True, "path": path, "removed_table": False, "table_keys": [], "error": f"{type(exc).__name__}: {exc}", "level": "red"}
    table = data.get(CFG.REMOVED_TABLE)
    has = CFG.REMOVED_TABLE in data
    keys = sorted(str(k) for k in table) if isinstance(table, Mapping) else []
    return {"present": True, "path": path, "removed_table": has, "table": CFG.REMOVED_TABLE, "table_keys": keys, "error": None,
            "converter": CFG.CONVERTER, "level": "red" if has else "green"}


def slurm_only_section(root: str, now_s: Optional[float] = None, state_dir: Optional[str] = None) -> Dict[str, object]:
    """The "Cluster reads, store authority and the replica" section from STATE FILES ONLY --
    no database: the store authority rows (:func:`_authority_section`), the replica check (:func:`_replica_section`), the reads
    reservation holder and the read receipts (:func:`_reservation_section`, :func:`_receipts_section`; ``state_dir`` = the landing state
    directory, default ``$REGISTRY_LANDING_STATE``), the maintenance flag and the configuration's removed table. Every part is read
    under its own guard: a torn or foreign file is that part's ``error``, never a page that fails to render. ``problems`` lists the red
    and amber headlines for the tiles and the text command; ``level`` is the worst of the parts; ``elapsed_ms`` is this gather's cost."""
    from . import reads_protocol as RP
    t0 = time.perf_counter()
    now_s = time.time() if now_s is None else now_s
    rd = RP.reads_dir(state_dir)
    out: Dict[str, object] = {}
    for name, fn in (("authority", lambda: _authority_section(root, now_s)), ("replica", lambda: _replica_section(root, now_s)),
                     ("reservation", lambda: _reservation_section(rd, now_s)), ("receipts", lambda: _receipts_section(rd, now_s)),
                     ("maintenance", lambda: _maintenance_section(root, now_s)), ("config", lambda: _config_section(root))):
        try:
            out[name] = fn()
        except Exception as exc:  # noqa: BLE001 -- a part that cannot be read is shown as its error, never a page that fails to render
            out[name] = {"present": False, "error": f"{type(exc).__name__}: {exc}", "level": "red"}
    a, rp, rs, rc, mt, cf = (out[k] for k in ("authority", "replica", "reservation", "receipts", "maintenance", "config"))   # type: ignore[misc]
    problems: List[str] = []
    for name, part in out.items():
        if part.get("error"):   # type: ignore[union-attr]
            problems.append(f"{name}: {part['error']}")   # type: ignore[index]
    if a.get("present") and a.get("n_problems"):   # type: ignore[union-attr]
        problems.append(f"store authority: {a['n_problems']} component(s) with a problem ({'; '.join(a['problems'][:6])}{'; …' if len(a['problems']) > 6 else ''})")   # type: ignore[index]
    # the last `stores` run's outcome, named with the stamp (a mirror outcome, `check:<state>` or `verify-authority:<state>`), for the
    # rows the authority check did not already fault
    red_mirror = [f"{r['slug']}: {r.get('mirror_outcome')}" for r in a.get("rows") or [] if r.get("mirror_level") == "red" and not r.get("problems")]   # type: ignore[union-attr]
    if red_mirror:
        problems.append(f"store mirror: {len(red_mirror)} component(s) with a red last outcome ({', '.join(red_mirror[:6])}{'; …' if len(red_mirror) > 6 else ''})")
    if rp.get("present") and not rp.get("ok"):   # type: ignore[union-attr]
        problems.append(f"replica check {'UNEQUAL' if not rp.get('equal') else 'not ok'} ({rp.get('headline')}; {rp.get('utc')})")   # type: ignore[union-attr]
    if rs.get("held") and rs.get("state") != "live":   # type: ignore[union-attr]
        problems.append(f"reads reservation held by {rs.get('owner')} {rs.get('op_id')} is {rs.get('state')}")   # type: ignore[union-attr]
    # the receipts' CURRENT problems, one headline per red or amber fact the level rests on (the level is the newest receipt's, every
    # adapter's newest result and the unreadable files -- an older red row is history and gets no headline)
    newest_rc = rc["rows"][0] if rc.get("rows") else {}   # type: ignore[index, union-attr]
    if newest_rc.get("level") == "red":
        if newest_rc.get("n_refused"):
            problems.append(f"latest read receipt {newest_rc.get('op_id')} refused {newest_rc.get('n_refused')} adapter(s)")
        else:
            problems.append(f"latest read receipt {newest_rc.get('op_id')} stopped at a pull barrier ({str(newest_rc.get('barrier'))[:120]})")
    carried = [f"{r['adapter']} ({r.get('newest_op_id')})" for r in rc.get("adapters") or []   # type: ignore[union-attr]
               if r.get("level") == "red" and r.get("newest_op_id") != newest_rc.get("op_id")]
    if carried:
        problems.append(f"{len(carried)} adapter(s) whose newest read refused them, in a receipt before the latest ({', '.join(carried[:6])}{'; …' if len(carried) > 6 else ''})")
    never_whole = [str(r["adapter"]) for r in rc.get("adapters") or [] if r.get("level") == "amber" and r.get("op_id") is None]   # type: ignore[union-attr]
    if never_whole:
        problems.append(f"{len(never_whole)} adapter(s) never read whole ({', '.join(never_whole[:6])}{'; …' if len(never_whole) > 6 else ''})")
    # an adapter whose NEWEST read did not read it whole (ids left standing unread, a shard missing) is amber even when an older receipt
    # read it whole -- that older read is the baseline, not the newest result
    not_whole_now = [f"{r['adapter']} ({r.get('newest_state')}, {r.get('newest_op_id')})" for r in rc.get("adapters") or []   # type: ignore[union-attr]
                     if r.get("level") == "amber" and r.get("op_id") is not None]
    if not_whole_now:
        problems.append(f"{len(not_whole_now)} adapter(s) whose newest read did not read them whole, their last whole read older "
                        f"({', '.join(not_whole_now[:6])}{'; …' if len(not_whole_now) > 6 else ''})")
    # a standing refusal is recorded and shown, never an abort and never red -- the headline names it as standing, not as new
    standing = [f"{r['adapter']} ({r.get('standing_on')})" if r.get("standing_on") else str(r["adapter"]) for r in rc.get("adapters") or []   # type: ignore[union-attr]
                if r.get("level") == "standing"]
    if standing:
        problems.append(f"{len(standing)} adapter(s) with a standing refusal -- re-confirmed byte for byte against the prior refusal under the "
                        f"same inputs and tree; recorded, not new, not an abort ({'; '.join(standing[:6])}{'; …' if len(standing) > 6 else ''})")
    if rc.get("n_unreadable"):   # type: ignore[union-attr]
        problems.append(f"{rc['n_unreadable']} unreadable receipt file(s) under {rd}")   # type: ignore[index]
    if mt.get("set"):   # type: ignore[union-attr]
        problems.append(f"maintenance flag set since {mt.get('since_utc')} ({mt.get('reason')}){' -- STALE: a window left it behind' if mt.get('stale') else ''}")   # type: ignore[union-attr]
    if cf.get("removed_table"):   # type: ignore[union-attr]
        problems.append(f"config.toml still carries the removed `{cf.get('table')}` table ({', '.join(cf.get("table_keys") or []) or 'no keys'}): every registry command refuses it")   # type: ignore[union-attr]
    out["problems"] = problems
    out["n_problems"] = len(problems)
    out["level"] = _worst(*(str(part.get("level")) for part in (a, rp, rs, rc, mt, cf)))   # type: ignore[union-attr]
    out["reads_dir"] = rd
    out["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return out


def gather_operational(root: str, now_s: Optional[float] = None) -> Dict[str, object]:
    """The state-file sections (no database): snapshots resident + live leases, mirror state and lag,
    footprint, the paper rebuild RESULT, timer health, disk headroom."""
    from . import mirrors as M
    from . import snapshot as SN
    t_op0 = time.perf_counter()
    now_s = time.time() if now_s is None else now_s
    ctx: Dict[str, object] = {}
    present = os.path.isdir(root)
    resident = []
    for sid in (SN.snapshot_ids(root) if present else []):
        m = SN.manifest_of(root, sid) or {}
        resident.append({"snapshot_id": sid, "scope": m.get("scope", "full"), "ingest_seq": m.get("ingest_seq"),
                         "built_utc": m.get("built_utc"), "age_s": _age_s(m.get("built_utc"), now_s), "files": len(m.get("files", {})),
                         "leases": len(SN.leases_of(root, sid))})
    ctx["snapshots_resident"] = resident
    ctx["snapshot_current"] = SN.pointer(root) if present else None
    leases = []
    ld = os.path.join(SN.snapshots_dir(root), SN.LEASES_DIR) if present else ""
    if ld and os.path.isdir(ld):
        for sid in sorted(os.listdir(ld)):
            for tok in SN.leases_of(root, sid):
                info = _read_json(os.path.join(ld, sid, tok)) or {}
                leases.append({"snapshot_id": sid, "reader": tok, "call_id": tok.split(".", 1)[0], "touched_utc": info.get("touched_utc"),
                               "heartbeat_age_s": _age_s(info.get("touched_utc"), now_s)})
    ctx["leases"] = leases
    ctx["footprint"] = SN.read_footprint(root) if present else None
    ctx["mirror_state"] = M.read_state(root) if present else None
    lag_rows = []
    ms = ctx["mirror_state"] or {}
    for d in (ms.get("destinations") or []):
        lag_rows.append({"destination": d.get("destination"), "kind": d.get("kind"), "checked_utc": d.get("utc") or ms.get("utc"),
                         "manifest": d.get("manifest") or d.get("latest_manifest"),
                         "lag_files": d.get("lag_files", d.get("missing")), "unsynced_local_files": d.get("unsynced_local_files"),
                         "switched": d.get("switched"), "error": d.get("error")})
    ctx["mirror_lag"] = lag_rows
    ctx["latest_manifest"] = M.latest_manifest(root) if present else None
    state = os.path.join(root, "state")
    ctx["paper_rebuild"] = {"result": _read_text(os.path.join(state, "paper_rebuild.RESULT")),
                            "last_head": _read_text(os.path.join(state, "paper_last_head")),
                            "failed_head": _read_text(os.path.join(state, "paper_failed_head")),
                            "head_file": _read_text(os.path.join(state, "paper_head"))}
    timers = []
    lr = os.path.join(state, "last_run")
    if os.path.isdir(lr):
        for fn in sorted(os.listdir(lr)):
            if not fn.endswith(".json"):
                continue
            info = _read_json(os.path.join(lr, fn)) or {}
            timers.append({"command": info.get("command", fn[:-5]), "last_run_utc": info.get("utc"), "age_s": _age_s(info.get("utc"), now_s),
                           "rc": info.get("rc"), "seconds": info.get("seconds")})
    ctx["timers"] = timers
    ctx["harvest_cycle"] = harvest_cycle_section(root, now_s) if present else harvest_cycle_section(None, now_s)
    ctx["wal_bytes"] = wal_size_bytes(root) if present else 0
    ctx["wal_gb"] = round(ctx["wal_bytes"] / 1e9, 2)
    # the cluster reads of the latest window: routes, refusals, stale mirrors -- from the plan file
    # state/cluster_reads/<window>.json the window script writes; no database, no schema change
    from . import cluster_reads as CR
    try:
        ctx["cluster_reads"] = CR.status_section(root)          # the empty section when the root or its plan directory is absent
    except Exception as exc:  # noqa: BLE001 -- a torn plan file is shown, never a page that fails to render
        ctx["cluster_reads"] = dict(CR.EMPTY_SECTION, error=f"{type(exc).__name__}: {exc}")
    # store authority, the replica check, the reads reservation and receipts, the maintenance flag, the removed configuration
    # table -- state files only, every part under its own guard
    ctx["slurm_only"] = slurm_only_section(root, now_s)
    free = H.free_bytes(root if present else ".")
    db_bytes = DB.file_size(DB.db_path(root)) if os.path.exists(DB.db_path(root)) else 0
    ctx["disk"] = {"free_gb": round(free / 1024 ** 3, 2), "floor_gb": round(DISK_FLOOR_GB, 1), "db_gb": round(db_bytes / 1024 ** 3, 3),
                   "snapshot_need_gb": round((0.4 * db_bytes + 2 * 1024 ** 3) / 1024 ** 3, 2),
                   "headroom_gb": round((free - H.MIN_FREE_BYTES) / 1024 ** 3, 2), "budget_gb": 10}
    ctx["operational_ms"] = round((time.perf_counter() - t_op0) * 1000, 2)
    return ctx


def _source_coverage(root: str) -> Dict[str, object]:
    """The declared-source coverage for the page: `sources.live_coverage` over the configured paper checkout."""
    from . import sources as SRC
    return SRC.live_coverage(root)


def _multi_source_declarations() -> List[Dict[str, object]]:
    """The multi-source ownership declarations in force (ownership.MULTI_SOURCE_OWNERSHIP), one row each, for the page: the scope,
    the owner and the secondary, what is matched, checked, carried and never carried — so an empty check table reads as "no
    identity carried by the secondary yet", never as "no rule"."""
    from . import ownership as OW
    return [{"declaration_id": d.declaration_id, "models": ", ".join(d.model_ids), "object_label": d.paper_label, "eval_category": d.eval_category,
             "transfer_kind": d.transfer_kind, "excluded_eval_ids": ", ".join(d.exclude_eval_ids), "owner": d.owner_source, "secondary": d.secondary_source,
             "strength_rule": d.strength_rule, "served_operating_point_id": d.served_operating_point_id, "pairing_rule": d.pairing_rule,
             "match": ", ".join(d.match), "check": ", ".join(d.check), "carry": ", ".join(d.carry), "never_carry": ", ".join(d.never_carry),
             "since": d.since} for d in OW.declarations()]


def source_coverage_section(root: str) -> Dict[str, object]:
    """`_source_coverage` with its failure made VISIBLE: a section that says "not computed: <error>" rather than an empty
    table that looks like zero gaps."""
    try:
        return dict(_source_coverage(root), error=None)
    except Exception as exc:  # noqa: BLE001 -- the page must render; the error is the content of the section
        return {"error": f"{type(exc).__name__}: {exc}", "without_adapter": [], "claimed_without_priority": [], "n_declared": None,
                "n_covered": None, "placeholders": [], "inputs": [], "errors": [], "paper_checkout": None}


def _cached_count(cache: Mapping[str, object], table: str) -> Tuple[Optional[int], Optional[str]]:
    """One table's ``(count, as_of)`` from a ``table_counts`` cache payload; ``(None, None)`` when the table is not in it."""
    v = (cache.get("counts") or {}).get(table)   # type: ignore[union-attr]
    if not isinstance(v, int):
        return None, None
    as_of = (cache.get("counts_as_of") or {}).get(table) or cache.get("as_of")   # type: ignore[union-attr]
    return v, (str(as_of) if as_of else None)


def _table_counts_within(conn: sqlite3.Connection, cache: Optional[Mapping[str, object]], section_deadline: float,
                         render_deadline: float) -> Tuple[Dict[str, Optional[int]], Dict[str, Optional[str]], List[str], bool]:
    """The row count per table, cheapest-by-last-count first, each under the section's deadline (never past the render's); once
    a count is interrupted the remaining tables come from ``cache``. Returns (counts, as_of per table — None for a live count,
    the cached stamp otherwise, the cached tables, over_budget): ``over_budget`` is True when the SECTION deadline was the
    binding one (the render deadline cutting it short is "not reached", not "heavy")."""
    cache = dict(cache or {})
    cached_counts = cache.get("counts") or {}

    def key(t: str):
        v = cached_counts.get(t)   # type: ignore[union-attr]
        return (0, int(v)) if isinstance(v, int) else (1, 0)
    counts: Dict[str, Optional[int]] = {}
    as_of: Dict[str, Optional[str]] = {}
    cached_tables: List[str] = []
    deadline = min(section_deadline, render_deadline)
    exhausted = False
    for t in sorted(R.ALL_TABLES, key=key):
        if not exhausted:
            try:
                counts[t] = _count_within(conn, t, deadline)
                as_of[t] = None
                continue
            except SectionOverBudget:
                exhausted = True
        counts[t], as_of[t] = _cached_count(cache, t)
        cached_tables.append(t)
    over_budget = exhausted and section_deadline <= render_deadline
    return ({t: counts[t] for t in R.ALL_TABLES}, {t: as_of[t] for t in R.ALL_TABLES}, cached_tables, over_budget)


def gather(conn: sqlite3.Connection, root: str, budget_overrides: Optional[Mapping[str, float]] = None,
           persist: bool = True) -> Tuple[Dict[str, object], Dict[str, float]]:
    """Run every section query within budget; returns (context, per-query seconds).

    Order: the sections NOT in the cached-with-refresh set first, each under ``section_budget_s``; then the members with what
    remains of ``render_budget_s``. A section interrupted by its own budget renders from its cache, is marked, and joins the
    set; a member that comes back in budget leaves it; a section the render budget cannot reach renders from its cache and is
    marked "not reached" without joining. Every in-budget result refreshes that section's cache file. ``budget_overrides``
    (tests, operators) set one section's budget in seconds; the budgets themselves are config.toml's ``[status]``. ``persist``
    is the render's: it writes the section caches and moves the set; ``status_json`` (a read) passes False and writes nothing."""
    cfg = status_config(root)
    overrides = {k: float(v) for k, v in (budget_overrides or {}).items()}
    section_budget = cfg["section_budget_s"]
    render_budget = cfg["render_budget_s"]
    heavy = read_heavy_state(root)
    members = set(heavy.get("members") or [])
    ctx: Dict[str, object] = {}
    timings: Dict[str, float] = {}
    section_state: Dict[str, Dict[str, object]] = {}
    joined: List[str] = []
    cleared: List[str] = []
    t_gather0 = time.perf_counter()
    render_deadline = time.monotonic() + render_budget
    has_paper_objects = sql_name_exists(conn, PAPER_OBJECTS_NAME)
    order = [n for n in QUERIES if n not in members] + [n for n in QUERIES if n in members]
    for name in order:
        sql = QUERIES[name]
        t0 = time.perf_counter()
        budget = overrides.get(name, section_budget)
        remaining = render_deadline - time.monotonic()
        # the last good rows are read only where they can be served: a section the render budget cannot reach, the table
        # counts (their cache orders the tables cheapest-first and backs the ones not reached), a query interrupted below
        cache = read_section_cache(root, name) if (remaining <= 0 or sql is None) else None
        st: Dict[str, object] = {"budget_s": budget, "from_cache": False, "as_of": None, "state": STATE_LIVE, "reason": None,
                                 "in_heavy_set": name in members}
        if remaining <= 0:
            reason = f"skipped this render: the {render_budget:g} s render budget was exhausted before this section"
            _serve_from_cache(ctx, name, sql, cache, st, reason)
        elif sql is None:
            section_deadline = time.monotonic() + budget
            counts, as_of, cached_tables, over = _table_counts_within(conn, cache, section_deadline, render_deadline)
            ctx["table_counts"] = counts
            ctx["table_counts_as_of"] = as_of
            ctx["table_counts_cached"] = cached_tables
            now = R.utc_now()
            merged_as_of = {t: (as_of[t] if as_of[t] else now) for t in counts if counts[t] is not None}
            if cached_tables:
                st.update(from_cache=True, state=STATE_CACHED if any(counts[t] is not None for t in cached_tables) else STATE_NO_CACHE,
                          as_of=min((v for t, v in merged_as_of.items() if t in cached_tables), default=None),
                          reason=(f"skipped this render for {len(cached_tables)} of {len(counts)} tables: over its {budget:g} s budget" if over
                                  else f"skipped this render for {len(cached_tables)} of {len(counts)} tables: the {render_budget:g} s render budget ran out"))
                if over:
                    joined.append(name)
            else:
                st["as_of"] = now
                if name in members:
                    cleared.append(name)
            if persist and any(counts[t] is not None for t in counts):
                write_section_cache(root, name, {"as_of": now, "counts": {t: v for t, v in counts.items() if v is not None},
                                                 "counts_as_of": merged_as_of, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
                                                 "source": "render", "exact": not cached_tables})
        else:
            deadline = time.monotonic() + min(budget, remaining)
            try:
                rows, error = _section_rows(conn, name, deadline, has_paper_objects)
            except SectionOverBudget:
                own = budget <= remaining   # the section's own budget was the binding deadline, not the render's
                reason = (f"skipped this render: over its {budget:g} s budget" if own
                          else f"skipped this render: the {render_budget:g} s render budget ran out during this section")
                _serve_from_cache(ctx, name, sql, read_section_cache(root, name), st, reason)
                if own:
                    joined.append(name)
            else:
                ctx[name] = rows
                if error is not None:
                    # the error row is the section's content this render (the page's convention) and never its last good rows:
                    # the cache keeps the previous result, and the set is untouched — a failed query is no timing measurement
                    st.update(state=STATE_ERROR, reason=f"query failed: {error}")
                else:
                    now = R.utc_now()
                    st["as_of"] = now
                    if persist:
                        write_section_cache(root, name, {"as_of": now, "rows": rows, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
                                                         "source": "render", "exact": True})
                    if name in members:
                        cleared.append(name)
        timings[name] = time.perf_counter() - t0
        st["ms"] = round(timings[name] * 1000, 2)
        section_state[name] = st
    if persist and (joined or cleared):
        heavy = update_heavy_state(root, add=joined, remove=cleared, last_render_utc=R.utc_now())
        members = set(heavy.get("members") or [])
    ctx["section_state"] = section_state
    ctx["sections_from_cache"] = [n for n in QUERIES if section_state[n]["from_cache"]]
    ctx["sections_no_cache"] = [n for n in QUERIES if section_state[n]["state"] == STATE_NO_CACHE]
    ctx["heavy_set"] = sorted(members)
    ctx["heavy_state"] = {"refreshed_utc": heavy.get("refreshed_utc"), "head": heavy.get("head"), "initialized": bool(heavy.get("initialized"))}
    ctx["budgets"] = dict(cfg)
    ctx["paper_objects_available"] = has_paper_objects
    ctx["paper_objects_name"] = PAPER_OBJECTS_NAME
    head, index_head = (_first_row(ctx, n) for n in ("head", "index_head"))
    ctx["head_seq"] = head.get("ingest_seq") if head else None
    ctx["index_head_seq"] = index_head.get("ingest_seq") if index_head else None
    # the pending tile shows the count — or the query's error text, never a healthy-looking 0 for a failed query
    ctx["pending_count"] = next(iter(ctx["pending"][0].values())) if ctx["pending"] else 0   # type: ignore[index, union-attr]
    ctx["open_round"] = _first_row(ctx, "open_round")
    ctx["decode_unmatched_total"] = sum(int(r.get("n_runs") or 0) for r in ctx["decode_policy_match_counts"]
                                        if "error" not in r and r.get("match") in ("ambiguous", "no_match"))
    ctx["not_measured_total"] = sum(int(r.get("n_run_measure_pairs") or 0) for r in ctx["degeneration_not_measured"] if "error" not in r)
    # best-of-N decode precedence (decode_precedence): the violations are the fact; the pending keys ride the section beside them
    dp_rows = [r for r in ctx["bon_decode_precedence"] if "error" not in r]
    ctx["bon_decode_precedence_violations"] = [r for r in dp_rows if r.get("state") == DP.VIOLATION_STATE]
    ctx["bon_decode_precedence_pending"] = [r for r in dp_rows if r.get("state") == DP.PENDING_STATE]
    ctx["bon_decode_precedence_ruling_id"] = DP.RULING_ID
    # multi-source ownership: the count per state is a tile; a refused identity is the one that must never be quiet
    ms_rows = [r for r in ctx["multi_source"] if "error" not in r]
    ctx["multi_source_counts"] = {st: sum(1 for r in ms_rows if r.get("state") == st) for st in ("refused", "stopgap", "folded")}
    ctx["multi_source_refused"] = [r for r in ms_rows if r.get("state") == "refused"]
    ctx["multi_source_declarations"] = _multi_source_declarations()
    ctx["rulings_relayed_unconfirmed"] = [r for r in ctx["rulings"] if "error" not in r and r.get("provenance") == "relayed"
                                          and str(r.get("confirmed_by_user")) in ("0", "", "False", "false")]
    pushes: List[Dict[str, object]] = []
    for row in ctx["snapshots"][:1]:
        if "error" in row:
            break
        try:
            for p in json.loads(row.get("pushed_to_json") or "[]"):
                pushes.append({"snapshot_id": row["snapshot_id"], "destination": p.get("destination"), "pushed_utc": p.get("pushed_utc"),
                               "ok": p.get("ok"), "reclaimed": len(p.get("reclaimed") or []), "leases_blocking": len(p.get("leases_blocking") or []),
                               "error": p.get("error")})
        except (ValueError, TypeError):
            pass
    ctx["snapshot_pushes"] = pushes
    for row in ctx["snapshots"]:
        try:
            row["scope"] = json.loads(row.get("extra_json") or "{}").get("scope", "full")
        except (ValueError, TypeError, AttributeError):
            row["scope"] = None
    ctx.update(gather_operational(root))
    # the refusal facts derived within the last 3 harvest cadences (the harvester row's own window), from the section's rows
    now_s = time.time()
    window_s = int(ctx["harvest_cycle"]["late_after_s"])
    ctx["refusal_facts_window_s"] = window_s
    ctx["refusal_facts_recent"] = [r for r in ctx["refusal_facts"] if "error" not in r
                                   and (lambda a: a is not None and a <= window_s)(_age_s(r.get("cycle_utc"), now_s))]
    ctx["source_coverage"] = source_coverage_section(root)
    ctx["paused"] = H.paused(root)
    ctx["free_gb"] = ctx["disk"]["free_gb"]
    ctx["built_utc"] = R.utc_now()
    ctx["root"] = root
    ctx["schema_version"] = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migration").fetchone()[0]
    # the writers' guard, computed from the database itself: a version behind the code, or a shape that differs from the
    # code's DDL at the same version (db.schema_drift). Both refuse every write; the page must say so (2026-09-19)
    from .migrations import code_version
    ctx["schema_version_code"] = code_version()
    try:
        ctx["schema_drift"] = DB.schema_drift(conn)
        # the fold order the database RECORDS against the code's (db.require_fold_order refuses every writer on a difference): shown as a
        # drift line, so the banner says so; a reader, this page displays the stamp and never checks or writes it (no stamp: nothing shown)
        from .reducer import FOLD_ORDER_VERSION
        recorded = DB.fold_order_version(conn)
        if recorded is not None and recorded != FOLD_ORDER_VERSION:
            ctx["schema_drift"] = list(ctx["schema_drift"]) + [f"fold order: the database records {recorded!r}, the code folds under "
                                                               f"{FOLD_ORDER_VERSION!r} (every writer refuses; db.FoldOrderDrift)"]
    except sqlite3.Error as exc:
        ctx["schema_drift"] = [f"not computed: {exc}"]
    ctx["deprecation_notice"] = ("The r2-compat parquet export is kept for one paper round and is DEPRECATED "
                                 "(Fixed decision 21); read the registry.")
    ctx["timings_ms"] = {k: round(v * 1000, 2) for k, v in timings.items()}
    ctx["gather_ms"] = round((time.perf_counter() - t_gather0) * 1000, 2)
    ctx["total_ms"] = ctx["gather_ms"]   # render() adds the template render + write once they are known
    return ctx, timings


def _serve_from_cache(ctx: Dict[str, object], name: str, sql: Optional[str], cache: Optional[Mapping[str, object]],
                      st: Dict[str, object], reason: str) -> None:
    """Fill ``ctx[name]`` from the section's cache (or an empty result when there is none) and mark the state."""
    cache = dict(cache or {})
    st["from_cache"] = True
    st["reason"] = reason
    if sql is None:
        cached = {t: _cached_count(cache, t) for t in R.ALL_TABLES}
        ctx["table_counts"] = {t: count for t, (count, _) in cached.items()}
        ctx["table_counts_as_of"] = {t: as_of for t, (_, as_of) in cached.items()}
        ctx["table_counts_cached"] = list(R.ALL_TABLES)
        has = bool(cache.get("counts"))
    else:
        rows = cache.get("rows")
        ctx[name] = list(rows) if isinstance(rows, list) else []
        has = isinstance(rows, list)
    st["as_of"] = cache.get("as_of") if has else None
    st["state"] = STATE_CACHED if has else STATE_NO_CACHE


def render_html(ctx: Dict[str, object]) -> str:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "templates")),
                      autoescape=select_autoescape(["html"]))
    return env.get_template("status.html.j2").render(**ctx)


def default_out_path(root: str) -> str:
    """``[status] page_dir`` from config.toml when present, else the figure mirror's ``results-registry/index.html``."""
    try:
        from . import config as CFG
        return os.path.join(str(CFG.load(root)["status"]["page_dir"]), "index.html")
    except Exception:
        return DEFAULT_OUT


def render(root: str, conn: Optional[sqlite3.Connection] = None, out_path: Optional[str] = None,
           register: bool = True, budget_overrides: Optional[Mapping[str, float]] = None) -> Tuple[str, Dict[str, float]]:
    """Render the page within budget, register it as an artifact (record first), write it, and append the render's line to
    ``state/status_render.jsonl``. Returns (path, timings)."""
    t_render0 = time.perf_counter()
    out_path = out_path or os.environ.get("REGISTRY_STATUS_HTML") or default_out_path(root)
    if conn is None:
        with DB.reading(DB.db_path(root)) as own:
            ctx, timings = gather(own, root, budget_overrides=budget_overrides)
    else:
        ctx, timings = gather(conn, root, budget_overrides=budget_overrides)
    ctx["total_ms"] = round((time.perf_counter() - t_render0) * 1000, 2)   # the footer's total; the history line's adds the template render + write
    html = render_html(ctx)
    data = html.encode("utf-8")
    sha = hashlib.sha256(data).hexdigest()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if register:
        H.write_harvester_record(root, "artifact", {
            "artifact_id": sha, "kind": "page", "uri_primary": os.path.abspath(out_path),
            "locations_json": [{"uri": os.path.abspath(out_path), "verified_utc": ctx["built_utc"], "bytes": len(data)}],
            "bytes": len(data), "created_utc": ctx["built_utc"], "verified_utc": ctx["built_utc"],
            "extra": {"page": "status", "built_utc": ctx["built_utc"]}}, str(ctx["built_utc"]), subdir="publish")
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, out_path)
    total_ms = round((time.perf_counter() - t_render0) * 1000, 2)
    try:
        append_render_history(root, render_history_line(ctx, total_ms))
    except OSError:
        pass   # the page is written; a history line the disk refused is not the render's failure
    return out_path, timings


def render_history_line(ctx: Mapping[str, object], total_ms: float) -> Dict[str, object]:
    """One ``status_render.jsonl`` line: the build stamp, the totals, per section ms / as_of / from_cache / state, the set."""
    sections = {name: {"ms": st.get("ms"), "as_of": st.get("as_of"), "from_cache": bool(st.get("from_cache")), "state": st.get("state")}
                for name, st in (ctx.get("section_state") or {}).items()}
    slurm_only = ctx.get("slurm_only") or {}
    return {"built_utc": ctx.get("built_utc"), "total_ms": total_ms, "gather_ms": ctx.get("gather_ms"), "sections": sections,
            "heavy_set": list(ctx.get("heavy_set") or []), "budgets": dict(ctx.get("budgets") or {}),
            "index_head_seq": ctx.get("index_head_seq"),
            # the state-file sections' cost beside the database's, so the page's within-budget claim is checkable with the new section
            "operational_ms": ctx.get("operational_ms"), "slurm_only_ms": slurm_only.get("elapsed_ms"),   # type: ignore[union-attr]
            "slurm_only_level": slurm_only.get("level"), "slurm_only_problems": len(slurm_only.get("problems") or [])}   # type: ignore[union-attr]


# --------------------------------------------------------------------------------------------
# registry status --refresh-heavy: the cached-with-refresh set recomputed exactly (the 6-hourly unit; the window's cache initializer)
# --------------------------------------------------------------------------------------------


def ledger_head(conn: sqlite3.Connection) -> int:
    """``MAX(ingest_seq)`` of ``record_ingest`` — the ledger head the heavy refresh is gated on."""
    return int(conn.execute(QUERIES["index_head"]).fetchone()[0] or 0)


def _compute_exact(conn: sqlite3.Connection, name: str, deadline: float) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    """One section computed exactly under ``deadline``: ``(cache payload, None)``, or ``(None, the error)`` when its query failed —
    an error is never a cache payload. :class:`SectionOverBudget` at the deadline."""
    if QUERIES[name] is None:
        counts: Dict[str, int] = {}
        for t in R.ALL_TABLES:
            v = _count_within(conn, t, deadline)
            if v is not None:
                counts[t] = v
        now = R.utc_now()
        return {"as_of": now, "counts": counts, "counts_as_of": {t: now for t in counts}, "exact": True}, None
    rows, error = _section_rows(conn, name, deadline, sql_name_exists(conn, PAPER_OBJECTS_NAME))
    if error is not None:
        return None, f"query failed: {error}"
    return {"as_of": R.utc_now(), "rows": rows, "exact": True}, None


def refresh_heavy(root: str, conn: Optional[sqlite3.Connection] = None, force: bool = False) -> Dict[str, object]:
    """Recompute every member of the cached-with-refresh set exactly (each under ``heavy_budget_s``) into the section cache
    files; a member that came in under ``section_budget_s`` is cleared from the set. Event-gated: with the set initialized, a
    refresh whose ledger head equals the last recorded one recomputes only the members still due — the ones whose last refresh
    did not succeed (``pending_members``) or that have no cache yet — and is skipped when none is (``force`` recomputes every
    member). The recorded head advances only when EVERY targeted section succeeded: a member that timed out or whose query
    failed keeps its old cache and is due again at the next tick whatever the ledger does (before this, the head was recorded
    regardless and a slow member stayed stale through a quiet ledger).

    Initialization (the first run after install, ``initialized`` unset): EVERY section is computed exactly and cached, and
    the set is derived from the measured times — the sections over ``section_budget_s`` — so the first timed render already
    finds cached rows and no fixed list decides what is heavy."""
    if conn is None:
        with DB.reading(DB.db_path(root)) as own:
            return refresh_heavy(root, own, force=force)
    cfg = status_config(root)
    heavy_budget = cfg["heavy_budget_s"]
    section_budget = cfg["section_budget_s"]
    st = read_heavy_state(root)
    members = sorted(set(st.get("members") or []))
    head = ledger_head(conn)
    initialized = bool(st.get("initialized"))
    pending = [n for n in (st.get("pending_members") or []) if n in members]   # members whose last refresh did not succeed
    if not initialized:
        targets = list(QUERIES)
    elif force or st.get("head") != head:
        targets = members
    else:
        # the head stands, so every member's cache is current except a pending one (its last refresh failed) or one with no cache
        targets = [n for n in members if n in pending or read_section_cache(root, n) is None]
        if not targets:
            return {"skipped": True, "reason": "ledger head unchanged since the last refresh and every member current", "head": head,
                    "members": members, "pending_members": pending, "refreshed_utc": st.get("refreshed_utc")}
    started = R.utc_now()
    results: Dict[str, Dict[str, object]] = {}
    cleared: List[str] = []
    joined: List[str] = []
    for name in targets:
        t0 = time.perf_counter()
        try:
            payload, error = _compute_exact(conn, name, time.monotonic() + heavy_budget)
            interrupted = False
        except SectionOverBudget:
            payload, error, interrupted = None, f"interrupted at the {heavy_budget:g} s heavy budget; the previous cache stands", True
        elapsed = time.perf_counter() - t0
        # membership follows the measured time only: an interrupt at the heavy budget or a result over the section budget keeps
        # (or adds) the member, a result under it clears; a failed query is no measurement and leaves the set as it was
        still_heavy = interrupted or (payload is not None and elapsed > section_budget)
        rec: Dict[str, object] = {"elapsed_ms": round(elapsed * 1000, 2), "ok": payload is not None,
                                  "as_of": payload["as_of"] if payload is not None else None,
                                  "cleared": payload is not None and elapsed <= section_budget and name in members}
        if payload is not None:
            payload.update(elapsed_ms=rec["elapsed_ms"], source="heavy")
            write_section_cache(root, name, payload)
        else:
            rec["error"] = error
        if still_heavy and name not in members:
            joined.append(name)
        if rec["cleared"]:
            cleared.append(name)
        results[name] = rec
    last = dict(st.get("last_refresh") or {})
    last.update(results)
    failed = [n for n, rec in results.items() if not rec["ok"]]
    # the recorded head advances only when every targeted section succeeded; the failures are pending for the next tick, which
    # the unchanged-head gate lets through for them
    fields: Dict[str, object] = dict(refreshed_utc=R.utc_now(), started_utc=started, last_refresh=last, initialized=True, pending_members=failed)
    if not failed:
        fields["head"] = head
    new_state = update_heavy_state(root, add=joined, remove=cleared, **fields)
    return {"skipped": False, "initialized_now": not initialized, "head": head, "head_recorded": new_state.get("head"),
            "members": new_state["members"], "cleared": cleared, "joined": joined, "failed": failed, "results": results,
            "refreshed_utc": new_state["refreshed_utc"]}


def status_json(root: str, conn: Optional[sqlite3.Connection] = None) -> Dict[str, object]:
    """The page's context as JSON — a READ: the same sections within the same budgets, served from the caches where the render
    would be, but no cache file written and the set not moved (only the render persists)."""
    if conn is None:
        with DB.reading(DB.db_path(root)) as own:
            return gather(own, root, persist=False)[0]
    return gather(conn, root, persist=False)[0]
