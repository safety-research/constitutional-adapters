"""The harvester — the ONLY writer to ``registry.sqlite``.

Layout under the registry root (``~/registry`` on the grading host)::

    registry.sqlite                       the rebuildable index (WAL)
    records/<campaign>/<run_dir>/<seq>-<type>.jsonl[.gz]   the merged, mirrored record log (system of record)
    records/harvester/<utc-day>/…         harvester-originated facts (run_ingest, ruling_state, …) — records BEFORE rows
    records/adapters/<adapter>/<rev>/…    adapter-produced records (deterministic ids)
    records/dimensions/<rev>/…            one seed revision's dimension records (``registry dimensions load``): applied whole, ONE transaction
    records/claims/<run_dir>/…            grading claims (judgment records)
    rulings/*.jsonl                       rulings (also part of the log)
    inbox/<host>/<campaign>/<run_dir>/…   what the clusters push over the write-only key; merged into records/
    state/                                paused.json, maintenance.json (a landing window's drain flag; C6), paper_head, refused.jsonl, metrics.jsonl
    manifests/, snapshots/, CURRENT       mirrors.py / snapshot.py
    harvester.lock                        the flock every writer takes

``ingest_under_lock(root, pull=True)``: the cluster pull backstops run FIRST and OUTSIDE the write lock (they write
``inbox/<host>/`` only: rsync's dot-named temp files and atomic renames, which the inbox merge is blind to until the
rename; 2026-09-30 -- an rsync of up to 600 s per cluster used to sit inside the lock), then the lock is taken and
``ingest_once(root)`` runs: free-space guard → inbox merge → parse new/changed files in
worker processes → every dimension revision directory (``records/dimensions/<rev>/``) applied whole in ONE
transaction FIRST (a reader sees the old dimension set or the new one, never a mix; the seeds are every other
table's FK roots) → apply the plain files through the reducer in one transaction per ≤ 50 files, records in
``(written_utc, record_sha256)`` order; every FILE takes its ledger rank ``ingest_seq`` in ARRIVAL order
(``record_ingest.arrival_utc`` = the cycle's clock at the file's first fold, strictly after the
head's, so a new file is always head + 1 and nothing already ranked moves) → dependency-blocked queue
re-evaluation → harvester-originated facts (``run_ingest`` only when completion requirements hold;
one ``file_arrival`` fact per file folded, so the order is in the log) written as records and ingested →
the cycle's one dense pass (a no-op unless a file sorted below the head) → gzip-after-ingest → status page.

``rebuild(from_dir, into)`` replays the same fold over the whole log, ordering files by the arrival the
``file_arrival`` facts recorded (``first_written_utc`` for a file with none); ``verify`` diffs every table
except the delivery audit.
"""
from __future__ import annotations

import collections
import fnmatch
import functools
import gzip
import hashlib
import itertools
import json
import operator
import os
import re
import shutil
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Collection, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from . import config as CFG
from . import db as DB
from . import decode_match as DM
from . import records as R
from . import reducer as RED
from . import schema_def as S
from . import site_aliases as SA
from . import validate as VAL
from . import vocab as V
from .identity import canonical_json
from .reducer import ITEM_PLANS, LifecycleEvidenceMissing, Reducer, VerifyMismatch, effects_of, fold_item_events, item_row_prefix, matches_key, remark_item_withdrawals

GB = 1024 ** 3
MIN_FREE_BYTES = 5 * GB
FILES_PER_TRANSACTION = 50
ITEM_ROWS_PER_TRANSACTION = 10_000      # a transaction is capped by files AND by rows (F11): ≈ 1.3 s of b-tree writes at the rate
                                        # measured on a loaded grading host, so every window — a revision's staging windows included —
                                        # stays well inside the 5 s bound on the longest transaction
PARSE_WINDOW_PER_WORKER = 4             # parsed files a worker pool may run ahead of the writer (memory O(window), F1)
FILE_STAT_NAME = "file_stat.json"       # state/ sidecar: (size, mtime_ns, inode) -> logical sha per on-disk file (F12)
BLOCKED_STALE_S = 7 * 86400
LARGE_FILE_BYTES = 64 * 1024 ** 2         # a plain record file above this is parsed in-process in bounded windows, never materialised
RECLAIM_BATCH_ROWS = 10_000               # item rows one bookkeeping transaction after an activation touches
#: An adapter revision directory — ``records/adapters/<adapter>/<rev>/`` — or a re-activation bundle of a committed
#: revision — ``records/adapters/<adapter>/_activations/<rev>/<stamp>/``. Everything under one is applied by
#: :meth:`Ingestor.apply_revision`: streamed in bounded windows, constituents first and the ``source_revision`` marker
#: LAST, in ONE transaction that rolls back whole on any blocked or refused constituent.
REVISION_DIR_RE = re.compile(r"^records/adapters/[^/]+/(?:_activations/([0-9a-f]{16})/\d+|([0-9a-f]{16}))/")


#: The record types a bundle's FACT files carry -- folded whatever the no-op decision, on the head path and, at the carrier's
#: activation, on the held path -- and the ones set apart from the measurement comparison (2026-09-29, ``Ingestor._same_measurements``):
#: the marker and the status rows are the read's facts; the artifact files register the revision's inputs (fixed by its id, so never a
#: measurement difference; a re-emission of committed bytes omits them by design) and carry the input's ``verified_utc``, which a
#: re-read of a touched input advances -- a fact too, never a duplicate.
FACT_RECORD_TYPES: Tuple[str, ...] = ("source_status", "artifact")
COMPARED_APART: Tuple[str, ...] = ("source_revision", "source_status", "artifact")


def revision_of_uri(file_uri: str) -> Optional[str]:
    m = REVISION_DIR_RE.match(file_uri)
    return (m.group(1) or m.group(2)) if m else None


def bundle_of_uri(file_uri: str) -> str:
    """Files that must land in ONE transaction share a bundle: a revision directory; every other file is its own."""
    m = REVISION_DIR_RE.match(file_uri)
    return m.group(0) if m else file_uri


#: A dimension seed revision directory — ``records/dimensions/<rev>/`` (``adapters.dimensions.write_seed_records``: the nine
#: ``<seq>-<table>.jsonl`` files of one seed revision, published by one directory rename). Everything under one that a cycle
#: finds new or changed is applied by :meth:`Ingestor.apply_dimension_bundle` in ONE transaction, before the cycle's plain
#: files: the seeds are the FK roots of every other table, and a reader of the paper objects must see the previous dimension
#: set or the new one, never a mix (the nine files folded as plain files in ≤ 50-file chunks, interleaved by stamp with every
#: other pending file of the cycle — hundreds on the live database, minutes per chunk — so a reader between two commits saw
#: revision-2 ``model`` / ``eval`` / ``item_set`` rows beside revision-1 ``instrument`` rows; 2026-09-22).
DIMENSION_DIR_RE = re.compile(r"^records/dimensions/([0-9a-f]{16})/")

#: Test hook: called with the directory path INSIDE the dimension bundle's transaction, after its records are folded and
#: before the commit (``tests/test_registry_dimensions_atomic.py`` pauses there and reads from a second connection). None in
#: production.
DIMENSION_APPLY_HOOK = None


def dimension_revision_of_uri(file_uri: str) -> Optional[str]:
    m = DIMENSION_DIR_RE.match(file_uri)
    return m.group(1) if m else None
GZIP_MIN_AGE_S = 600
HARVESTER_PRODUCER_SCRIPT = "sisteer.registry.harvest"

#: Every table carrying an ingest_seq column (remapped by the dense pass when a file sorted below the head).
SEQ_TABLES: Tuple[str, ...] = tuple(t.name for t in S.TABLES if "ingest_seq" in t.column_names and t.name != "record_ingest")
#: The seq-bearing tables with a ``(source_revision_id, record_file_id)`` index: the dense pass remaps their rows through it
#: (``Ranker._remap_pairs``) rather than scanning them -- the item tables, millions of rows with no index on ``ingest_seq``.
REVISION_INDEXED_TABLES: Tuple[str, ...] = tuple(
    t.name for t in S.TABLES if t.name in SEQ_TABLES and any(tuple(cols) == ("source_revision_id", "record_file_id") for _, cols, _ in t.indexes))
#: Above this many ``(revision, file)`` pairs the dense pass scans the item tables instead (a remap of thousands of files at
#: once -- a replay's closing pass -- is cheaper as one scan than as millions of index probes). Both paths are exact.
SEQ_REMAP_PAIRS_MAX = 4_000_000
#: The indexed remap of the item tables; tests flip it off to compare the two paths byte for byte.
INDEXED_SEQ_REMAP = True
#: Where the harvester writes its own facts (``write_harvester_record``'s default ``subdir``); a file under it is stamped
#: at fold time by the harvester itself, so its stamp IS its arrival and no ``file_arrival`` fact is emitted for it.
HARVESTER_SUBDIR = "harvester"
HARVESTER_URI_PREFIX = f"records/{HARVESTER_SUBDIR}/"
#: The one harvester file a cycle writes for every file that arrived in it: ``records/harvester/<day>/file_arrival-<sha12>.jsonl``.
ARRIVAL_FILE_PREFIX = "file_arrival-"
#: The harvester file a cycle's refusal-fact pass writes: ``records/harvester/<day>/record_refusal-<sha12>.jsonl``, one per pass
#: that found a refusal judgment with no fact yet (the START pass, before anything can replace a `record_refusal` row; the END pass, after
#: the cycle's own refusals).
REFUSAL_FACT_FILE_PREFIX = "record_refusal-"


def is_harvester_file(file_uri: str) -> bool:
    """A harvester-originated record file (``records/harvester/…``): ordered by its own stamp on every path."""
    return file_uri.startswith(HARVESTER_URI_PREFIX)


def is_refusal_fact_file(file_uri: str) -> bool:
    """One of the harvester's own refusal-fact files (:data:`REFUSAL_FACT_FILE_PREFIX` under ``records/harvester/``)."""
    return is_harvester_file(file_uri) and os.path.basename(file_uri).startswith(REFUSAL_FACT_FILE_PREFIX)
#: Per parent table, every column a dependency check may look a parent up by (the declared foreign keys' target columns,
#: the effect types' parents, and the `judged` evidence's judgment-by-run lookup) — what ``_held_still_blocked`` consults.
_PARENT_COLUMNS: Dict[str, Set[str]] = {}
for _t in S.TABLES:
    for _col, _ptable, _pcol in S.foreign_keys(_t.name):
        _PARENT_COLUMNS.setdefault(_ptable, set()).add(_pcol)
for _spec in R.EFFECTS.values():
    for _ptable, _pcol, _key in _spec.parents:
        _PARENT_COLUMNS.setdefault(_ptable, set()).add(_pcol)
_PARENT_COLUMNS.setdefault("judgment", set()).add("run_id")
del _t, _col, _ptable, _pcol, _spec, _key


# --------------------------------------------------------------------------------------------
# the run-set drop guard: a revision that drops runs the accepted head carries must say so
# --------------------------------------------------------------------------------------------

#: The marker's ``extra_json`` key under which a reader -- or the operator, through ``registry revision accept`` -- DECLARES the
#: run ids a revision drops on purpose: ``{"run_ids": [<run id or fnmatch glob>, ...] | "*", "reason": <why>, "declared_by": [...]}``.
#: The runner writes it from the adapter's ``runs_retired`` declaration (``adapters.base.runs_retired_of``), for the run ids that
#: left with an input the adapter declared retired (``inputs_retired``), and for a group whose every input left discovery
#: (``input_vanished``: ``"*"``); ``registry revision accept`` writes it on a bundle marker with the operator's reason.
DECLARED_RUN_DROPS_KEY = "declared_run_drops"
#: ``blocked_records.extra_json.kind`` of a marker held for an undeclared run drop (the other kinds: ``blocked``, ``refused``).
RUN_SET_DROP_KIND = "run_drop"
#: The label the hold carries inside ``source_revision.refusal`` (``activation pending: source_revision <head> (run_set_drop_undeclared: ...)``).
RUN_SET_DROP_KEY = "run_set_drop_undeclared"
#: How many dropped run ids the hold names (the counts are exact; the names are a sample).
RUN_SET_DROP_NAMED = 8
#: RULE VERSIONING. The guard judges a marker against the database's STATE at fold
#: time, so a replay reproduces the live fold only if the rule's version is part of what the replay knows: a revision the live fold
#: ACCEPTED before the rule existed is held by a replay that applies today's rule to it, and the head it leaves in place makes every later
#: revision of the source compare against a stale parent (the R3 bootstrap, 2026-10-05: 45 revisions accepted live between 09-18 and
#: 09-23 held, 6 more chained behind them; 0 the other way). The rule therefore applies only to a marker OBSERVED at or after the stamp
#: the LIVE HARVESTER began applying it, so a replay applies the rule in force when each revision was accepted.
#:
#: Two stamps, and the second is the one that counts. The commit landed on main at 2026-09-23T18:30:43Z (f4ad71fd7, committer date
#: 11:30:43 -0700; :data:`RUN_SET_DROP_UNDECLARED_LANDED`), but the live harvester runs the base checkout's code
#: (``registry-harvest.service``: ``PYTHONPATH=<checkout>/src``), whose ``main`` first contained the commit at the fast-forward to
#: 6c07514e4 at 2026-09-23 14:16:01 -0700 = 2026-09-23T21:16:01Z -- the activation. The difference is
#: one revision: relu_paired:s2 d366f7644c815fcf, observed 20:22:17Z, an undeclared 88 -> 86 shrink against its parent 02a6d9f47d740223,
#: ACCEPTED live after the commit and before the activation (the next read of the source, 986f126caad01b46 at 21:58:10Z, folded under
#: the rule and passed against it; the first live hold is 35cf85d9c435bc5d at 23:21:54Z). Gated at the commit stamp a replay holds
#: d366f764 and moves the source's head off it; gated at the activation it reproduces the live chain (every stamp in
#: (20:22:17.033Z, 21:58:10.449Z] does; verified read-only 2026-10-05 with scripts/registry/run_set_drop_versioning_check.py: 0 holds
#: of the versioned replay, 5 live holds all observed after the stamp).
#:
#: The live database is unchanged by construction: every marker observed before the stamp was folded (accepted or held) before this gate
#: existed and is never re-judged, and every live hold under the rule was observed after it. A future fold-rule change that judges a
#: revision against the state at fold time needs the same: its live activation stamp, or an acceptance record in the log.
RUN_SET_DROP_UNDECLARED_LANDED = "2026-09-23T18:30:43.000Z"
RUN_SET_DROP_UNDECLARED_SINCE = "2026-09-23T21:16:01.000Z"
#: The one-line rule the hold carries, so a consumer reading the marker knows what fired without opening this module.
RUN_SET_DROP_RULE = ("a revision that no longer names a run id the source's accepted head carries, and does not declare the drop "
                     "(extra.declared_run_drops: the id, a glob over it, or '*', with a reason), is published but NOT accepted: the head stays "
                     "current and the marker sits at accepted = 0 with the reason until a bundle carrying the declaration arrives -- the "
                     "adapter's runs_retired re-read, or `registry revision accept`. The rule is a NET SHRINK: it fires only when the revision "
                     "names FEWER run ids than the head (runs_now < runs_previous); a re-keying that replaces N ids with N others, a superset, "
                     "or the same ids re-emitted is unaffected (D562)")


# --------------------------------------------------------------------------------------------
# the retirement head gate (registry-hardening 3a): a retirement folds only against the head it was computed from
# --------------------------------------------------------------------------------------------

#: The marker's ``extra_json`` key under which every RETIREMENT marker (an EMPTY revision, ``inputs_json = {}``; ``runner._retirement_records``)
#: records the premise it was computed from: ``{"outgoing_revision_id": <the source's accepted head when the retirement was built, or null>,
#: "inputs": [<every file input it retires, canonical, sorted>]}``. A shard writes it from the reader-state sidecar it read, the grading host path
#: and the pull's re-activation by id from the live database; :meth:`Ingestor._retirement_head_moved` compares it with the database at
#: fold time.
RETIREMENT_EXPECTS_KEY = "retirement_expects"
#: ``blocked_records.extra_json.kind`` of a retirement marker held because its premise is not the live state (the other kinds: ``blocked``,
#: ``refused``, :data:`RUN_SET_DROP_KIND`).
RETIREMENT_HEAD_KIND = "retirement_head"
#: The label the hold carries inside ``source_revision.refusal`` (``activation pending: source_revision <live head> (retirement_head_moved: ...)``).
RETIREMENT_HEAD_KEY = "retirement_head_moved"
#: The one-line rule the hold carries, so a consumer reading the marker knows what fired without opening this module.
RETIREMENT_HEAD_RULE = ("a retirement (an empty revision) names the accepted head it was computed against and every input it retires "
                        "(extra.retirement_expects); it folds only when that head is still the source's accepted head and the live inventory "
                        "(the source's non-retired watermarks and the head's inputs) holds no input it does not name. Otherwise the marker is "
                        "published but NOT accepted: the head stays current, nothing is withdrawn or retired, and the marker sits at accepted = 0 "
                        "with the reason until a retirement computed against today's head arrives -- the next window's shard refuses the same "
                        "empty id as known with a claim against the live head and the pull re-activates it as a bundle. A marker without the "
                        "key (published before the rule) is judged as the fold of its day judged it")


def retirement_head_moved(conn: sqlite3.Connection, rev: str, source_id: str, expects: object) -> Optional[Tuple[Optional[str], str]]:
    """The retirement head gate (registry-hardening 3a) as a pure function of the database and a marker's
    :data:`RETIREMENT_EXPECTS_KEY` value: ``(live head, detail)`` when the retirement ``rev`` of ``source_id`` must NOT fold -- its
    ``outgoing_revision_id`` is not the source's accepted head today, or the live inventory holds a file input the retirement does not
    name -- else None. ``expects`` is the marker's value as parsed (None when the marker carries none: a retirement published before the
    rule, judged as the fold of its day judged it -- the rule's own versioning, no stamp needed: only a producer of this tree writes the
    key). A present but malformed value holds by name: holding is the safe side of a premise the registry cannot read.

    WHY (the review's scenario). A shard exports reader state with head H1 and publishes a NEW empty retirement R after the group's staged
    inputs vanished, declared retired. Before R is pulled and folded, an EARLIER window's delayed revision H2 of the same source -- with an
    input the shard never saw -- becomes the accepted head. R's withdrawals name H1's keys and its watermark inventory is H1's; folded, R
    would become the head and silently withdraw H2's measurements and retire H2's added input, which nobody declared retired (the pull
    validates the attestation and the module, not the live head, and the run-drop guard accepts R's wildcard declaration). The
    refused-known RETIREMENT already carries the shard's claim and is withheld at the pull when the head moved (round 1); a NEW
    retirement had no such gate, and a pull-time check alone would not do -- another writer can advance the head between the pull and
    the fold -- so the comparison is made HERE, inside the activation transaction, against the state the fold would act on. The live
    inventory is what ``runner._retirement_inventory`` reads: the source's watermarks not marked retired (``last_error`` not a
    ``runner.VANISHED_PREFIX`` one) and the accepted head's ``inputs_json`` keys, the synthetic anchor-decision input
    (``registry://anchor-decision/<source_id>``) set aside -- it is no file and the retirement names it nowhere."""
    from .adapters.anchor_current import DECISION_URI_PREFIX     # at call time: the adapters package imports this module
    from .adapters.runner import VANISHED_PREFIX
    head = conn.execute("SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC, revision_id DESC LIMIT 1",
                        (source_id,)).fetchone()
    head_rev = str(head[0]) if head is not None else None
    if head_rev == rev:
        return None
    if expects is None:
        return None
    if not isinstance(expects, Mapping) or "outgoing_revision_id" not in expects or not isinstance(expects.get("inputs"), list) \
            or not all(isinstance(u, str) for u in expects["inputs"]):
        return head_rev, (f"{RETIREMENT_HEAD_KEY}: the retirement {rev} of {source_id} carries a malformed {RETIREMENT_EXPECTS_KEY} ({expects!r}); "
                          "the premise it was computed from cannot be read, so the head stays current")
    expected = expects.get("outgoing_revision_id")
    expected = str(expected) if expected is not None else None
    if expected != head_rev:
        return head_rev, (f"{RETIREMENT_HEAD_KEY}: the retirement {rev} of {source_id} was computed against the accepted head "
                          f"{expected or '(none)'}, and the live accepted head is {head_rev or '(none)'}; its withdrawals and its watermark "
                          "inventory are the other head's, so the head stays current and nothing is withdrawn or retired -- the next window's "
                          "shard judges the group against today's head")
    live: Set[str] = {str(r[0]) for r in conn.execute(
        "SELECT input_uri FROM source_watermark WHERE source_id = ? AND (last_error IS NULL OR substr(last_error, 1, ?) <> ?)",
        (source_id, len(VANISHED_PREFIX), VANISHED_PREFIX)).fetchall()}
    if head_rev is not None:
        row = conn.execute("SELECT inputs_json FROM source_revision WHERE revision_id = ?", (head_rev,)).fetchone()
        try:
            inputs = json.loads(row[0]) if row is not None and row[0] else {}
        except ValueError:
            inputs = {}
        if isinstance(inputs, Mapping):
            live |= {str(u) for u in inputs}
    named = set(expects["inputs"])
    undeclared = sorted(u for u in live if not u.startswith(DECISION_URI_PREFIX) and u not in named)
    if undeclared:
        shown = undeclared[:RUN_SET_DROP_NAMED]
        return head_rev, (f"{RETIREMENT_HEAD_KEY}: the live inventory of {source_id} holds {len(undeclared)} input(s) the retirement {rev} does not "
                          f"name ({len(shown)} of {len(undeclared)} named): " + ", ".join(shown)
                          + (f" (+{len(undeclared) - len(shown)} more)" if len(undeclared) > len(shown) else "")
                          + "; retiring the group would leave them standing on a declaration that never covered them, so the head stays current "
                          "-- the next window's shard judges the group against today's head")
    return None


#: The keys of a held retirement's IDENTITY as the reads protocol records and compares it (``cluster_reads.holds_seen_of`` -> the plan's
#: ``holds_seen`` -> the receipt; ``reads_protocol.held_retirements_unread``): the revision, the held carrier (the queue row's ``record_uri``)
#: and the head the hold was computed against (``blocked_records.missing_parent_id``, the premise the gate compared). The head is part of it
#: (registry-hardening 3a, closing pass): once the compared head is superseded, :meth:`Ingestor._held_still_blocked` re-attempts the same
#: carrier, the gate holds it against the NEW head, and :meth:`Ingestor._block` rewrites the SAME queue row's ``missing_parent_id`` in
#: place (the row is keyed by the record's sha; the carrier's uri never changes) -- so a hold named by revision and carrier alone read as
#: already seen by a receipt that had named it against the OLD head, and no later tick retried the retirement while the new head's
#: measurements stayed current after its input vanished.
RETIREMENT_HOLD_IDENTITY: Tuple[str, ...] = ("revision_id", "record_uri", "blocking_head")


def retirement_hold_identity(hold: Mapping[str, object]) -> Dict[str, str]:
    """The identity of a hold (:data:`RETIREMENT_HOLD_IDENTITY`) from a row of :func:`retirements_held_for_moved_head`, a plan's
    ``holds_seen`` entry or a receipt's -- every key a string, a missing one empty. A receipt entry written before the head joined the
    identity names no head and so matches no hold the database lists (``missing_parent_id`` is NOT NULL: the head, or ``(none)``): the
    conservative side, one more read of the adapter."""
    return {k: str(hold.get(k) or "") for k in RETIREMENT_HOLD_IDENTITY}


def retirements_held_for_moved_head(conn: sqlite3.Connection, adapter: str) -> List[Dict[str, str]]:
    """The retirement markers of ``adapter`` the harvester HOLDS under :data:`RETIREMENT_HEAD_KIND` -- ``[{revision_id, source_id,
    first_seen_utc, record_uri, blocking_head}]``, oldest hold first -- read off the queue (the marker's ``blocked_records`` row: the hold's
    record; ``record_uri`` the carrier that was held and ``blocking_head`` its ``missing_parent_id``, the accepted head the retirement's
    premise was compared against when the hold was recorded or last re-attempted -- the hold's identity with the revision,
    :data:`RETIREMENT_HOLD_IDENTITY`), joined with the revision's ``source_revision`` row for its source and adapter ONLY -- never filtered
    by that row's ``accepted``. Runs against the live database and against a reader-state sidecar alike (``reader_state.export`` ships
    both tables; ``cluster_reads.holds_seen_of`` reads the sidecar). The row is ``accepted = 0`` when the hold is the retirement's first attempt (a NEW
    retirement, its own directory held) and ``accepted = 1`` -- a superseded historical revision -- when the retirement was accepted once,
    its source came back, and what is held is a later bundle re-activating it under its known id: :meth:`Ingestor._placeholder_revision`
    leaves an accepted row accepted, so the row says nothing about the hold and only the queue row does (registry-hardening 3a, review
    round 3: filtered by ``accepted = 0``, such a hold was invisible here, the window's receipt recorded a successful re-activation and
    read whole, and no later tick asked for a fresh claim while the vanished group's measurements stayed current). The one revision never
    listed is the source's accepted head today: the fold accepted it, and :func:`retirement_head_moved` itself returns None for the head.
    What the reads protocol asks before it decides whether an adapter is read again (``reads_protocol.held_retirements_unread``): such a
    hold is new information the cluster has not seen -- the group's watermarks stand live, and the next shard that reads the adapter judges
    the group against today's head and refuses the same empty id as known with a claim the pull can re-activate -- so the adapter is not
    parked under a complete-looking receipt while the hold stands."""
    rows = conn.execute(
        "SELECT json_extract(b.extra_json, '$.revision_id'), s.source_id, b.first_seen_utc, b.record_uri, b.missing_parent_id FROM blocked_records b "
        "JOIN source_revision s ON s.revision_id = json_extract(b.extra_json, '$.revision_id') "
        "WHERE b.record_type = 'source_revision' AND json_extract(b.extra_json, '$.kind') = ? AND json_extract(b.extra_json, '$.no_op') IS NULL "
        "AND s.adapter = ? AND s.revision_id <> COALESCE((SELECT h.revision_id FROM source_revision h WHERE h.source_id = s.source_id "
        "AND h.accepted = 1 ORDER BY h.observed_utc DESC, h.revision_id DESC LIMIT 1), '') "
        "ORDER BY b.first_seen_utc, b.record_uri", (RETIREMENT_HEAD_KIND, adapter)).fetchall()
    return [{"revision_id": str(r[0]), "source_id": str(r[1]), "first_seen_utc": str(r[2]), "record_uri": str(r[3]), "blocking_head": str(r[4])}
            for r in rows]


def run_set_drop_rule_applies(observed_utc: str) -> bool:
    """Whether the run-set drop guard judges a marker observed at ``observed_utc`` (the revision's ``observed_utc``; the marker record's
    ``written_utc`` stands in when a caller has only that): at or after :data:`RUN_SET_DROP_UNDECLARED_SINCE`, the stamp the live
    harvester began applying the rule. A stamp the registry cannot parse counts as AFTER the gate (the rule applies: holding is the safe
    side of an unreadable stamp, and the validator refuses such a marker before the fold reaches here)."""
    try:
        return R.utc_to_epoch_ms(observed_utc) >= R.utc_to_epoch_ms(RUN_SET_DROP_UNDECLARED_SINCE)
    except (ValueError, TypeError):
        return True


def run_ids_of_observation_keys(keys: Iterable[object]) -> Set[str]:
    """The run ids a set of run-scale observation keys speaks of: every key (canonical JSON, or a dict) carrying a ``run_id`` --
    a ``run`` key, a ``run_metric`` / ``judgment`` / ``run_state_event`` key alike -- so a run leaves the set only when the
    revision names it under no table."""
    out: Set[str] = set()
    for k in keys:
        d = k
        if isinstance(k, str):
            try:
                d = json.loads(k)
            except ValueError:
                continue
        if isinstance(d, Mapping) and d.get("run_id") not in (None, ""):
            out.add(str(d["run_id"]))
    return out


def declared_run_drops_of(extra: Optional[Mapping[str, object]]) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    """The declaration a marker's parked extras carry under :data:`DECLARED_RUN_DROPS_KEY`, checked: ``(declaration, None)`` when
    well formed, ``(None, None)`` when absent, ``(None, why)`` when present but malformed -- it then declares nothing and the hold
    says so. Well formed: an object whose ``run_ids`` is ``"*"`` or a non-empty list of non-empty strings (run ids or ``fnmatch``
    globs over them) and whose ``reason`` is a non-empty string."""
    raw = (extra or {}).get(DECLARED_RUN_DROPS_KEY)
    if raw is None:
        return None, None
    if not isinstance(raw, Mapping):
        return None, f"{DECLARED_RUN_DROPS_KEY} must be an object, not {type(raw).__name__}"
    ids = raw.get("run_ids")
    if ids != "*" and not (isinstance(ids, list) and ids and all(isinstance(x, str) and x for x in ids)):
        return None, f"{DECLARED_RUN_DROPS_KEY}.run_ids must be '*' or a non-empty list of run ids or globs"
    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return None, f"{DECLARED_RUN_DROPS_KEY}.reason must be a non-empty string"
    return dict(raw), None


def run_set_drop_comparison(prev_ids: Set[str], new_ids: Set[str], declaration: Optional[Mapping[str, object]]) -> Dict[str, object]:
    """A revision's run set against the accepted head's: ``runs_previous``, ``runs_now``, ``runs_dropped`` (head run ids the
    revision no longer names), ``runs_added``, ``runs_dropped_declared`` (dropped ids the declaration covers -- an exact id, an
    ``fnmatch`` glob, or ``"*"``), ``runs_dropped_undeclared`` with a sorted sample of them (``runs_dropped_undeclared_sample``, up to
    :data:`RUN_SET_DROP_NAMED`), and ``tripped`` = a NET shrink: at least one undeclared drop AND fewer run ids than the head
    (``runs_now < runs_previous``). A re-keying that replaces N ids with N others does not shrink and does not
    trip (the framework's standing contract: a store replaced under the same group withdraws the old key); the 2026-09-23 s2 / s3
    reads that replaced seven head run ids with three new ones DO -- the source names fewer runs than it did."""
    dropped = sorted(prev_ids - new_ids)
    star = False
    patterns: Tuple[str, ...] = ()
    if declaration is not None:
        ids = declaration.get("run_ids")
        star = ids == "*"
        patterns = tuple(str(x) for x in ids) if isinstance(ids, list) else ()
    declared = [d for d in dropped if star or any(d == pat or fnmatch.fnmatchcase(d, pat) for pat in patterns)]
    covered = set(declared)
    undeclared = [d for d in dropped if d not in covered]
    return {"runs_previous": len(prev_ids), "runs_now": len(new_ids), "runs_dropped": len(dropped), "runs_added": len(new_ids - prev_ids),
            "runs_dropped_declared": len(declared), "runs_dropped_undeclared": len(undeclared),
            "runs_dropped_undeclared_sample": undeclared[:RUN_SET_DROP_NAMED], "tripped": bool(undeclared) and len(new_ids) < len(prev_ids)}


# --------------------------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------------------------


def records_dir(root: str) -> str:
    return os.path.join(root, "records")


def inbox_dir(root: str) -> str:
    return os.path.join(root, "inbox")


def state_dir(root: str) -> str:
    p = os.path.join(root, "state")
    os.makedirs(p, exist_ok=True)
    return p


def rulings_dir(root: str) -> str:
    return os.path.join(root, "rulings")


def ensure_layout(root: str) -> None:
    for d in ("records", "inbox", "state", "rulings", "manifests", "snapshots"):
        os.makedirs(os.path.join(root, d), exist_ok=True)


def load_config(root: str) -> Dict[str, object]:
    """``config.toml`` (:mod:`sisteer.registry.config`) derives ``mirrors`` / ``pulls`` /
    ``snapshot_destinations``; an older ``config.json`` beside it overlays those keys verbatim.

    Fail-closed: ``config.load`` refuses a TOML that still carries the removed ``modal``
    table, and the merged document is checked by :func:`sisteer.registry.config.check_destinations` whenever a ``config.json``
    overlaid it — a ``kind: modal`` entry in any destination list raises :class:`sisteer.registry.config.ConfigError` by name
    (the entry is taken verbatim from that file, so it is the one place such a destination can still be spelled)."""
    out: Dict[str, object] = {}
    from . import config as CFG
    if os.path.exists(CFG.config_path(root)):
        out.update(CFG.as_json_config(CFG.load(root)))
    p = os.path.join(root, "config.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as fh:
            out.update(json.load(fh))
        CFG.check_destinations(out, p)
    return out


def logical_uri(root: str, path: str) -> str:
    """Record URIs are root-relative logical paths (no .gz), so a mirror rebuild reproduces them."""
    rel = os.path.relpath(path, root)
    return R.logical_uri(rel)


def resolve_uri(root: str, uri: str) -> Optional[str]:
    """The on-disk path of a logical URI (plain or .gz), else None."""
    p = os.path.join(root, uri)
    if os.path.exists(p):
        return p
    if os.path.exists(p + ".gz"):
        return p + ".gz"
    return None


# --------------------------------------------------------------------------------------------
# free-space guard
# --------------------------------------------------------------------------------------------


def free_bytes(path: str) -> int:
    """The registry's one free-space reading (``db.free_bytes``: clamped to the filesystem's total)."""
    return DB.free_bytes(path)


def free_space_guard(root: str, min_free: int = MIN_FREE_BYTES, now: Optional[str] = None) -> Tuple[bool, int]:
    """Below ``min_free`` free bytes: write ``state/paused.json`` and return (False, free). Above:
    clear the flag and return (True, free). Records stay in the spool either way."""
    free = free_bytes(root if os.path.exists(root) else os.path.dirname(root) or ".")
    flag = os.path.join(state_dir(root), "paused.json")
    if free < min_free:
        with open(flag, "w", encoding="utf-8") as fh:
            json.dump({"paused_utc": now or R.utc_now(), "free_bytes": free, "min_free_bytes": min_free,
                       "reason": "free space below the 5 GB floor; ingest paused, records stay in the spool"}, fh)
        return False, free
    if os.path.exists(flag):
        os.remove(flag)
    return True, free


def paused(root: str) -> Optional[Dict[str, object]]:
    flag = os.path.join(root, "state", "paused.json")
    if os.path.exists(flag):
        with open(flag, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


# --------------------------------------------------------------------------------------------
# the maintenance flag (the honest drain: a cycle that finds it neither pulls nor folds)
# --------------------------------------------------------------------------------------------

#: ``state/maintenance.json`` -- a landing window writes it BEFORE it stops the timers and removes it from its EXIT trap; ``registry ingest
#: --honor-maintenance`` (the harvest unit's mode) exits 0 at once while it exists, BEFORE the cluster pull and before the write lock, so a
#: cycle the timer fires during the drain neither pulls nor folds. Distinct from ``paused.json`` (the disk-pressure flag, never reused):
#: ``pending.claim`` / ``release`` / ``ingest_now`` do not read it (a claim folds its own file under the lock whatever the window does).
MAINTENANCE_FILE = "maintenance.json"


def maintenance_path(root: str) -> str:
    return os.path.join(root, "state", MAINTENANCE_FILE)


def maintenance(root: str) -> Optional[Dict[str, object]]:
    """The maintenance flag's content when it is set (``{since_utc, reason, ...}``), else None. An unreadable or non-JSON flag is still a
    flag: ``{"reason": "unreadable", ...}`` -- the drain must hold whatever the file says."""
    flag = maintenance_path(root)
    if not os.path.exists(flag):
        return None
    try:
        with open(flag, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {"reason": "malformed maintenance flag", "content": data}
    except (OSError, ValueError) as exc:
        return {"reason": f"unreadable maintenance flag ({type(exc).__name__})"}


def write_maintenance(root: str, reason: str, now: Optional[str] = None, **extra: object) -> str:
    """Set the flag (atomically; a unique temp name and a rename) and return its path."""
    path = maintenance_path(root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    R.write_json_atomic(path, {"since_utc": now or R.utc_now(), "reason": str(reason), "pid": os.getpid(), **extra}, indent=1, sort_keys=True)
    return path


def clear_maintenance(root: str) -> bool:
    """Remove the flag; returns whether one was there."""
    try:
        os.remove(maintenance_path(root))
        return True
    except FileNotFoundError:
        return False


# --------------------------------------------------------------------------------------------
# the judging code's stamp (the exact refused hold, 2026-09-30)
# --------------------------------------------------------------------------------------------


#: The modules a refusal can come from, stamped by their SOURCE BYTES: validate (the field and vocabulary checks, the banned metric
#: ids), records (the envelope), vocab, reducer (the fold's verify rules) and this module (the staging and dependency checks that
#: raise a refused hold). Extended, never shortened: a module left out is a fix the stamp does not see.
_STAMPED_MODULES = (VAL, R, V, RED)


@functools.lru_cache(maxsize=1)
def validator_stamp() -> str:
    """The judging code's stamp, 16 hex: sha256 over the code's DDL (``db.code_ddl_master``: every table, CHECK, index, trigger and
    v_* object the fold writes against -- a vocabulary list is a CHECK, so a vocabulary change moves it), ``records.SCHEMA_VERSION``,
    and the source bytes of :data:`_STAMPED_MODULES` plus this module. Content-based, never a path and never a version a person
    bumps (the EMISSION_VERSION lesson: a hand-bumped version is forgotten exactly when a fix lifts a refusal). A revision held on a
    REFUSED constituent records the stamp that judged it (``blocked_records.extra_json.judged_with``, :meth:`Ingestor.apply_revision`'s
    held path) and is re-attempted when the stamp differs -- once, after which the new judgment stands
    (:meth:`Ingestor._refused_hold_exact`). An install without ``.py`` sources stamps the DDL and the schema version alone: coarser (a
    validator-only fix is then seen only when its DDL or vocabulary moves), never wrong (nothing is skipped that the same bytes under
    the same DDL did not already judge). Cached per process; the timer's cycle is a fresh process."""
    h = hashlib.sha256()
    for (kind, name), ddl in sorted(DB.code_ddl_master().items()):
        h.update(f"{kind} {name}\n{ddl}\n".encode("utf-8"))
    h.update(f"schema_version={R.SCHEMA_VERSION}\n".encode("utf-8"))
    for mod in _STAMPED_MODULES + (sys.modules[__name__],):
        path = getattr(mod, "__file__", None)
        if path and path.endswith(".py") and os.path.isfile(path):
            with open(path, "rb") as fh:
                h.update(fh.read())
        h.update(b"\0")
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------------------------
# inbox merge
# --------------------------------------------------------------------------------------------


#: an adapter revision directory's name under ``inbox/<host>/adapters/<a>/`` (``records/adapters/<a>/<rev16>``; the same shape
#: ``reader_state.REV16_RE`` reads) and the marker file every complete one holds
_REV16_RE = re.compile(r"^[0-9a-f]{16}$")
REVISION_MARKER_TOKEN = "source_revision"


def _revision_dir(hroot: str, dirpath: str) -> bool:
    """Is ``dirpath`` a cluster-read revision directory of the inbox: ``adapters/<a>/<rev16>``, ``adapters/<a>/_activations/<rev16>/<stamp>``,
    or a shard's refusal record ``adapters/<a>/_refused/<rev16>`` (pulled like a revision, its one marker-shaped file the record)?"""
    parts = os.path.relpath(dirpath, hroot).split(os.sep)
    return (len(parts) == 3 and parts[0] == "adapters" and bool(_REV16_RE.match(parts[2]))) or \
           (len(parts) == 5 and parts[0] == "adapters" and parts[2] == "_activations" and bool(_REV16_RE.match(parts[3]))) or \
           (len(parts) == 4 and parts[0] == "adapters" and parts[2] == "_refused" and bool(_REV16_RE.match(parts[3])))


def _incomplete_revision_dir(hroot: str, dirpath: str, filenames: Sequence[str]) -> bool:
    """A cluster-read revision directory in the inbox (:func:`_revision_dir`) WITHOUT its ``*source_revision*`` marker: a partial
    delivery (the pull's pass 1 landed its batches and its pass 2 never ran). Such a directory is never merged --
    ``records/adapters/<a>/<rev16>`` is what ``reader_state.published_dirs_of`` counts as a revision the grading host holds, marker or not,
    so a marker-less one merged there would make every later cluster read refuse the id as known while nothing could ever activate it.
    It stays in the inbox for the pull of a window whose summaries list the id to complete (pass 1 ``--ignore-existing`` fills it,
    pass 2 brings the marker)."""
    return _revision_dir(hroot, dirpath) and not any(REVISION_MARKER_TOKEN in fn for fn in filenames)


def scan_inbox(root: str) -> List[Tuple[str, str]]:
    """Move every ``inbox/<host>/<campaign>/<run_dir>/<file>`` into ``records/<campaign>/<run_dir>/``;
    ``inbox/<host>/_<anything>/`` (``_probe``) stays where the cluster put it, and so does a cluster-read revision directory
    (``adapters/<a>/<rev16>``, an ``_activations`` bundle) that does not hold its ``*source_revision*`` marker -- a partial
    delivery, reported ``(directory, "incomplete")`` and left for a later pull to complete (:func:`_incomplete_revision_dir`).
    THE LOGICAL FILENAME IS ONE RECORD: ``<name>.jsonl`` and its ``<name>.jsonl.gz`` twin are the same logical URI
    (:func:`records.logical_uri`; :func:`gzip_after_ingest` turns the first into the second), so a name is OCCUPIED when EITHER form stands
    in ``records/``, and an incoming file is compared by its LOGICAL sha (the decompressed bytes) with whichever form is there: the same ->
    a duplicate, dropped; other bytes -> a producer bug, kept as ``<incoming name>.conflict-<sha8>`` and reported -- never installed as a
    second active form beside the first (before this rule a plain file arriving after the harvester had compressed the accepted record was
    installed beside the twin, and the two forms of one URI carried different bytes). Returns [(destination path, status)]."""
    out: List[Tuple[str, str]] = []
    ib = inbox_dir(root)
    if not os.path.isdir(ib):
        return out
    for host in sorted(os.listdir(ib)):
        hroot = os.path.join(ib, host)
        if not os.path.isdir(hroot):
            continue
        for dirpath, dirnames, filenames in os.walk(hroot):
            if dirpath == hroot:
                # an underscore-prefixed campaign dir (`_probe/`, the push transport's proof of route; the pull's `_failed/`, `_stale/`,
                # `_partial/`) is operator space: left in the inbox for a person to read, never merged — its file is not a record and
                # would sit in the log refused forever (the live install's two probes did, 2026-09-17). A dot-prefixed one is a pull's
                # attempt-local receive directory (`.recv-<window>-…`, cluster_reads.pull_window): in flight, never merged
                dirnames[:] = [d for d in dirnames if not (d.startswith("_") or d.startswith("."))]
            dirnames.sort()
            if filenames and _incomplete_revision_dir(hroot, dirpath, filenames):
                out.append((dirpath, "incomplete"))
                continue
            for fn in sorted(filenames):
                if fn.startswith(".") or not (fn.endswith(".jsonl") or fn.endswith(".jsonl.gz")):
                    continue
                src = os.path.join(dirpath, fn)
                rel = os.path.relpath(dirpath, hroot)
                dest_dir = os.path.join(records_dir(root), rel)
                os.makedirs(dest_dir, exist_ok=True)
                dest = os.path.join(dest_dir, fn)
                # the logical filename: `<name>.jsonl` and `<name>.jsonl.gz` are one record; occupied when EITHER form stands in records/
                logical = dest[:-3] if dest.endswith(".gz") else dest
                occupied = [p for p in (logical, logical + ".gz") if os.path.exists(p)]
                if occupied:
                    src_sha = R.file_logical_sha(src)
                    if any(R.file_logical_sha(p) == src_sha for p in occupied):
                        os.remove(src)
                        out.append((occupied[0], "duplicate"))
                        continue
                    conflict = dest + f".conflict-{src_sha[:8]}"
                    os.replace(src, conflict)
                    out.append((conflict, "conflict"))
                    continue
                os.replace(src, dest)
                out.append((dest, "merged"))
            if filenames and _revision_dir(hroot, dirpath):
                # a merged revision directory leaves the inbox whole: an emptied one left behind would read as an inbox directory to
                # every later pull (`unlisted`) for nothing
                try:
                    os.rmdir(dirpath)
                except OSError:
                    pass
    return out


# --------------------------------------------------------------------------------------------
# parsing (worker processes)
# --------------------------------------------------------------------------------------------


@dataclass
class ParsedRecord:
    """A run-scale record, or any malformed one: the validator's verdict travels to the writer, the raw record does not
    (except when a revision is being staged, which persists the few run-scale constituents as records)."""
    line: int
    sha: str
    result: object              # Validated | Malformed
    raw: Optional[Dict[str, object]] = None


@dataclass
class ParsedItem:
    """An item-scale record reduced IN THE WORKER to what the writer needs (F3): the row tuple minus the file's
    rank (appended once the file is ranked), the parent rows its dependency check names, the run ids it indexes."""
    line: int
    sha: str
    written_utc: str
    table: str
    row: Tuple
    parents: Tuple[Tuple[str, str, str], ...]   # (parent_table, parent_column, parent_id)
    run_ids: Tuple[str, ...]
    source_revision_id: str


@dataclass
class ParsedFile:
    path: str
    uri: str
    file_sha: str
    records: List[ParsedRecord] = field(default_factory=list)
    items: List[ParsedItem] = field(default_factory=list)
    error: Optional[str] = None
    producer_script: Optional[str] = None
    producer_session: Optional[str] = None
    schema_version: Optional[int] = None
    stat: Optional[Tuple[int, int, int]] = None   # (size, mtime_ns, inode) taken BEFORE the file was read (F12 sidecar)
    retry: bool = False                           # the bytes changed while being read: deferred to the next cycle
    part_index: int = 0                           # a file written across transactions is a sequence of parts
    last: bool = True                             # the final part carries the whole file's record count and verified sha
    whole_records: Optional[int] = None           # the WHOLE file's record count (known on the last part) and first key
    whole_first_key: Optional[Tuple[str, str]] = None
    skipped: int = 0                              # records a selective pass (stream_parts want=) counted but did not build
    resumed_from: int = 0                         # records an earlier run already committed (record_cursor); this parse skipped them

    @property
    def n_records(self) -> int:
        return len(self.records) + len(self.items) + self.skipped

    @property
    def total_records(self) -> int:
        return self.whole_records if self.whole_records is not None else self.n_records + self.resumed_from

    @property
    def is_last_part(self) -> bool:
        return self.last

    @property
    def is_single(self) -> bool:
        """The whole file in one part — never a resumed parse, whose earlier windows another run committed."""
        return self.part_index == 0 and self.last and not self.resumed_from

    @property
    def first_key(self) -> Optional[Tuple[str, str]]:
        """THE definition of a file's position in the log: the smallest ``(written_utc, record_sha256)`` among
        its well-formed records (None when it has none). Its first element is the ledger's ``first_written_utc``."""
        if self.whole_first_key is not None:
            return self.whole_first_key
        keys = [(pr.result.written_utc, pr.sha) for pr in self.records if isinstance(pr.result, VAL.Validated)]
        keys += [(pi.written_utc, pi.sha) for pi in self.items]
        return min(keys) if keys else None

    def parts(self, max_rows: int) -> List["ParsedFile"]:
        """Split a file larger than ``max_rows`` into parts of at most ``max_rows`` records that share its ledger
        identity — run-scale records AND item rows alike."""
        if self.error or self.retry or self.n_records <= max_rows or max_rows <= 0:
            return [self]
        # PHYSICAL record order: every window is a contiguous prefix of the file, run-scale and item records
        # interleaved as written, so the cursor's "records done" count and a resumed parse's skip agree
        seq: List[Tuple[str, object]] = sorted([("r", pr) for pr in self.records] + [("i", pi) for pi in self.items],
                                               key=lambda t: t[1].line)
        windows = [seq[i: i + max_rows] for i in range(0, len(seq), max_rows)]
        out = []
        for i, w in enumerate(windows):
            out.append(ParsedFile(self.path, self.uri, self.file_sha, records=[x for k, x in w if k == "r"], items=[x for k, x in w if k == "i"],
                                  producer_script=self.producer_script, producer_session=self.producer_session,
                                  schema_version=self.schema_version, stat=self.stat, part_index=i, last=(i == len(windows) - 1),
                                  whole_records=self.total_records, whole_first_key=self.first_key, resumed_from=self.resumed_from))
        return out


def _run_ids_of(payload: Mapping[str, object]) -> Tuple[str, ...]:
    return tuple(str(payload[c]) for c in ("run_id", "anchor_run_id") if payload.get(c))


def _stat3(path: str) -> Tuple[int, int, int]:
    st = os.stat(path)
    return (st.st_size, st.st_mtime_ns, st.st_ino)


def parse_file(args: Tuple) -> ParsedFile:
    """Parse + validate one record file (runs in a worker process). ``args = (root, path[, discovery sha])``. The
    logical sha recorded for the file is computed from the BYTES THIS PARSE READ (streamed beside the JSON), with
    the file's stat taken before and after: a stat that moved means the file changed under the reader — the file
    is deferred (``retry``) and nothing about it is cached; a discovery sha that differs from the parsed one is
    simply stale (an append between discovery and parse) and the parsed one wins. An optional fourth
    element skips the first ``skip`` records — the ones an interrupted run already committed."""
    root, path = args[0], args[1]
    skip = int(args[3]) if len(args) > 3 and args[3] else 0
    uri = logical_uri(root, path)
    try:
        before = _stat3(path)
    except OSError as exc:
        return ParsedFile(path, uri, "", error=f"unreadable: {exc}")
    pf = ParsedFile(path, uri, "", stat=before, resumed_from=skip)
    hasher = hashlib.sha256()
    try:
        n = 0
        for line, rec, rsha in R.read_record_file(path, hasher=hasher):
            n += 1
            if n <= skip:
                continue
            _classify(pf, uri, line, rec, rsha)
    except R.RecordLineError as exc:
        pf.error = f"{uri}#{exc.line}: {exc.detail}"     # the logical uri: the same reason on every replaying machine
    except (ValueError, EOFError, gzip.BadGzipFile) as exc:
        pf.error = str(exc)
    except OSError as exc:
        return ParsedFile(path, uri, "", error=f"unreadable: {exc}")
    pf.file_sha = hasher.hexdigest()
    try:
        after = _stat3(path)
    except OSError:
        after = None
    if after != before:
        pf.retry = True
    return pf


def _classify(pf: ParsedFile, uri: str, line: int, rec: Dict[str, object], rsha: str, want: str = "all",
              keep_raw: bool = False) -> None:
    """Validate one record and file it on ``pf`` as a run-scale ParsedRecord or a worker-built ParsedItem. With
    ``want`` = ``run_scale`` / ``items`` the other kind is only COUNTED (``pf.skipped``), so a revision can be read
    once for its few run-scale constituents and once for its item rows without holding either. ``keep_raw``
    keeps the record itself on a run-scale ParsedRecord (revision staging persists it)."""
    is_item = rec.get("record_type") in R.ITEM_SCALE_RECORD_TYPES
    if (want == "run_scale" and is_item) or (want == "items" and not is_item):
        pf.skipped += 1
        return
    if rsha is None:
        rsha = R.record_sha(rec)      # a selective pass hashes only the records it keeps (the skipped kind costs a json.loads)
    where = f"{uri}#{line}"
    res = VAL.validate_record(rec, where, rsha)
    if pf.producer_script is None and isinstance(rec.get("producer"), dict):
        pf.producer_script = rec["producer"].get("script")
        pf.producer_session = rec["producer"].get("session")
        pf.schema_version = rec.get("schema_version")
    if isinstance(res, VAL.Validated) and is_item and uri.startswith("records/adapters/"):
        # under an adapter's directory the bundle's revision is authoritative — an item row must carry exactly the
        # directory's revision tag; a missing / native / foreign tag would target another revision's (visible) key
        want_rev = revision_of_uri(uri)
        tag = str(res.payload.get("source_revision_id") or "")
        if want_rev is None or tag != want_rev:
            res = VAL.Malformed("source_revision_id",
                                (f"item row under records/adapters/ carries source_revision_id {tag!r}; the directory's revision is {want_rev!r} "
                                 "(natively tagged rows belong outside records/adapters/)"), where, res.record_type, rsha)
    if isinstance(res, VAL.Validated):
        res.uri = where
        if res.record_type in R.ITEM_SCALE_RECORD_TYPES:
            table = R.RECORD_TYPE_TABLE[res.record_type]
            pf.items.append(ParsedItem(line, rsha, res.written_utc, table, item_row_prefix(ITEM_PLANS[table], res, R.record_file_id_for_uri(uri)),
                                       tuple(VAL.parent_references(res)), _run_ids_of(res.payload),
                                       str(res.payload.get("source_revision_id") or "")))
            return
    pf.records.append(ParsedRecord(line, rsha, res, rec if keep_raw else None))


def stream_parts(root: str, path: str, sha: Optional[str], max_rows: int, want: str = "all", skip: int = 0,
                 keep_raw: bool = False) -> Iterator[ParsedFile]:
    """Parse one file IN BOUNDED WINDOWS: yields ParsedFile parts of at most ``max_rows`` records without ever
    holding the file; every part carries the file's stat-before-read, the discovery sha (or a pre-pass sha) as its
    provisional identity, and ONLY the last part is marked ``last`` — with the verified sha of the bytes parsed, the
    whole record count and, when the bytes changed under the reader, ``retry``. ``skip`` resumes after the
    records an interrupted run committed (they are hashed, never rebuilt). A file the parser cannot read yields
    one part with ``error``."""
    uri = logical_uri(root, path)
    try:
        before = _stat3(path)
        provisional = sha or R.file_logical_sha(path)
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        yield ParsedFile(path, uri, "", error=f"unreadable: {exc}")
        return
    hasher = hashlib.sha256()
    held: Optional[ParsedFile] = None
    cur = ParsedFile(path, uri, provisional, stat=before, last=False, resumed_from=skip)
    index = 0
    total = skip
    n = 0
    producer: Tuple = (None, None, None)
    error: Optional[str] = None
    try:
        for line, rec, rsha in R.read_record_file(path, hasher=hasher, compute_sha=(want == "all")):
            n += 1
            if n <= skip:
                continue
            _classify(cur, uri, line, rec, rsha, want, keep_raw)
            if producer[0] is None and cur.producer_script is not None:
                producer = (cur.producer_script, cur.producer_session, cur.schema_version)
            if len(cur.records) + len(cur.items) >= max_rows:
                if held is not None:
                    yield held
                total += cur.n_records
                held = cur
                index += 1
                cur = ParsedFile(path, uri, provisional, stat=before, part_index=index, last=False, resumed_from=skip,
                                 producer_script=producer[0], producer_session=producer[1], schema_version=producer[2])
    except R.RecordLineError as exc:
        error = f"{uri}#{exc.line}: {exc.detail}"          # the logical uri: the same reason on every replaying machine
    except (ValueError, EOFError, gzip.BadGzipFile) as exc:
        error = str(exc)
    except OSError as exc:
        error = f"unreadable: {exc}"
    total += cur.n_records
    final = cur if (cur.records or cur.items or held is None) else held
    if final is held:
        final.skipped += cur.skipped
    if final is held:
        pass
    elif held is not None:
        yield held
    final.last = True
    final.whole_records = total
    final.file_sha = hasher.hexdigest()
    final.error = error
    if final.producer_script is None:
        final.producer_script, final.producer_session, final.schema_version = producer
    try:
        after = _stat3(path)
    except OSError:
        after = None
    if after != before:
        final.retry = True
    yield final


def _nice_worker() -> None:
    try:
        os.nice(10)
    except OSError:
        pass


def _parse_worker_start(table: Mapping[str, str], source: Optional[str]) -> None:
    """A parsing worker's start: a lower priority, then the site-alias table the parent folds under. A worker that starts from a
    fresh interpreter (the spawn and forkserver start methods; forkserver is Linux's default from Python 3.14 on) imports
    ``site_aliases`` with the empty table, and it validates every record it parses, so without this a legacy spelling would be
    refused in a worker and accepted in-process."""
    _nice_worker()
    if table:
        SA.install_table(table, source or "<parent>")
    else:
        SA.reset()


def parse_pool(workers: int) -> ProcessPoolExecutor:
    """The process pool records are parsed in: each worker carries the table the parent's fold installed from its root
    (:func:`_parse_worker_start`), whatever the start method, so a file parses to the same result in a worker and in-process."""
    table, _, source = SA.active()
    return ProcessPoolExecutor(max_workers=workers, initializer=_parse_worker_start, initargs=(table, source))


def iter_parsed(root: str, work: Sequence[Tuple[str, Optional[str]]], workers: int = 3,
                max_rows: Optional[int] = None, skips: Optional[Mapping[str, int]] = None) -> Iterator[ParsedFile]:
    """Parse ``[(path, sha or None)]`` IN ORDER, yielding each file as it completes; a worker pool runs at most
    ``PARSE_WINDOW_PER_WORKER × workers`` files ahead of the consumer, so the parent holds O(window) parsed files,
    never the corpus (F1). With ``max_rows`` every file arrives as parts of at most that many records: a file above
    ``LARGE_FILE_BYTES`` is streamed in-process in bounded windows, a smaller one is parsed by a worker and
    split. ``skips`` = {path: records already committed} resumes a file after its cursor."""
    skips = skips or {}

    def emit(pf: ParsedFile) -> Iterator[ParsedFile]:
        return iter(pf.parts(max_rows)) if max_rows else iter([pf])

    def is_large(path: str) -> bool:
        try:
            return max_rows is not None and os.path.getsize(path) > LARGE_FILE_BYTES
        except OSError:
            return False

    def streamed(path: str, sha: Optional[str]) -> Iterator[ParsedFile]:
        if skips.get(path):
            return stream_parts(root, path, sha, max_rows or 0, skip=int(skips[path]))
        return stream_parts(root, path, sha, max_rows or 0)

    small = [(root, path, sha, skips.get(path, 0)) for path, sha in work if not is_large(path)]
    plan: List[Tuple[str, object]] = [("large", w) if is_large(w[0]) else ("small", None) for w in work]
    if workers <= 0 or len(small) < 4:
        small_it = iter(small)
        for kind, w in plan:
            if kind == "large":
                yield from streamed(w[0], w[1])
            else:
                yield from emit(parse_file(next(small_it)))
        return
    window = max(1, workers * PARSE_WINDOW_PER_WORKER)
    with parse_pool(workers) as ex:
        queue: "collections.deque" = collections.deque()
        it = iter(small)
        for a in itertools.islice(it, window):
            queue.append(ex.submit(parse_file, a))
        for kind, w in plan:
            if kind == "large":
                yield from streamed(w[0], w[1])
                continue
            fut = queue.popleft()
            nxt = next(it, None)
            if nxt is not None:
                queue.append(ex.submit(parse_file, nxt))
            yield from emit(fut.result())


def parse_files(root: str, paths: Sequence[str], workers: int = 3) -> List[ParsedFile]:
    """Every file parsed (the list-returning form; :func:`iter_parsed` is the streaming one the cycle uses)."""
    return list(iter_parsed(root, [(p, None) for p in paths], workers))


def chunk_parsed(parsed: Iterable[ParsedFile], max_files: int = FILES_PER_TRANSACTION,
                 max_rows: int = ITEM_ROWS_PER_TRANSACTION) -> Iterator[List[ParsedFile]]:
    """Group parsed files (or parts) into transactions of at most ``max_files`` files AND ``max_rows`` records (F11).
    A materialised file larger than the row cap is split into parts sharing its ledger row; revision
    directories never come through here (``Ingestor.apply_revision`` applies them whole)."""
    chunk: List[ParsedFile] = []
    rows = 0
    for pf in parsed:
        for part in (pf.parts(max_rows) if (pf.part_index == 0 and pf.last) else [pf]):   # an unsplit file splits here
            if chunk and (len(chunk) + 1 > max_files or rows + part.n_records > max_rows):
                yield chunk
                chunk, rows = [], 0
            chunk.append(part)
            rows += part.n_records
    if chunk:
        yield chunk


def split_work(root: str, work: Sequence[Tuple[str, Optional[str]]]
               ) -> Tuple[List[Tuple[str, Optional[str]]], List[str], List[Tuple[str, List[Tuple[str, Optional[str]]]]]]:
    """Route a cycle's files three ways: plain files (parsed in the streamed pool, ≤ 50 per transaction), the revision /
    activation DIRECTORIES they belong to (each applied whole by apply_revision, from its complete listing on disk), and the
    dimension seed revision directories (``records/dimensions/<rev>/``; each returned with ITS new or changed files, applied in
    one transaction by apply_dimension_bundle, first in the cycle). The dimension bundles keep the order ``work`` had."""
    plain: List[Tuple[str, Optional[str]]] = []
    dirs: List[str] = []
    seen: Set[str] = set()
    dims: Dict[str, List[Tuple[str, Optional[str]]]] = {}
    for path, sha in work:
        uri = logical_uri(root, path)
        if dimension_revision_of_uri(uri):
            dims.setdefault(os.path.join(root, DIMENSION_DIR_RE.match(uri).group(0).rstrip("/")), []).append((path, sha))
            continue
        b = bundle_of_uri(uri) if revision_of_uri(uri) else None
        if b is None:
            plain.append((path, sha))
        elif b not in seen:
            seen.add(b)
            dirs.append(os.path.join(root, b.rstrip("/")))
    return plain, dirs, list(dims.items())


def partition_harvester_files(root: str, work: Sequence[Tuple[str, Optional[str]]]
                              ) -> Tuple[List[Tuple[str, Optional[str]]], List[Tuple[str, Optional[str]]]]:
    """A replay's plain files split into ``(producer files, the harvester's own facts)`` -- the files under ``records/harvester/``
    (:func:`is_harvester_file`: ``run_ingest``, ``ruling_state``, ``file_arrival``, ``record_refusal``, the status page's artifact,
    ``mirror_ack``, ``snapshot`` ...), each list in the order given. A replay folds the facts AFTER its revision phase and that phase's
    blocked-queue pass, the place a live cycle writes and folds them (``ingest_once``: revisions, ``reevaluate_blocked``, then the derived
    facts, then ``reevaluate_blocked`` again); folded before the revisions they describe, a ``run_ingest`` blocked on its not-yet-folded
    run and was retried over whatever carriers had landed by then (the R3 bootstrap refused 128 of them where live refused
    none). Ranks are untouched by the order (a file's rank is a pure function of the ledger's keys)."""
    producer: List[Tuple[str, Optional[str]]] = []
    facts: List[Tuple[str, Optional[str]]] = []
    for path, sha in work:
        (facts if is_harvester_file(logical_uri(root, path)) else producer).append((path, sha))
    return producer, facts


ANALYSES_URI_PREFIX = "records/analyses/"


def is_analysis_file(file_uri: str) -> bool:
    """A record file of an analysis filing (``records/analyses/<day>/…``, `registry analysis file`): folded in the cycle's LATE phase."""
    return file_uri.startswith(ANALYSES_URI_PREFIX)


def partition_analysis_files(root: str, work: Sequence[Tuple[str, Optional[str]]]
                             ) -> Tuple[List[Tuple[str, Optional[str]]], List[Tuple[str, Optional[str]]]]:
    """The plain files split into ``(the rest, the analysis filings)`` -- the files under ``records/analyses/`` -- each list in the order
    given. Both paths fold the filings AFTER the revision phase and its blocked-queue pass (``ingest_once``, ``rebuild``): an
    `analysis_input` whose run rides an adapter revision directory would otherwise meet an absent run in the plain phase and take the
    ledger-fact refusal path, so a rebuild lacked a link the live database held and carried a refusal it never made. Ranks are
    untouched by the order (a file's rank is a pure function of the ledger's keys)."""
    rest: List[Tuple[str, Optional[str]]] = []
    late: List[Tuple[str, Optional[str]]] = []
    for path, sha in work:
        (late if is_analysis_file(logical_uri(root, path)) else rest).append((path, sha))
    return rest, late


#: The reason a ledger-fact record is refused for a parent the registry does not hold -- parseable (:func:`dependency_refusal_parent`), so the
#: cycle's retry pass (``Ingestor.retry_dependency_refusals``) can look the parent up again every cycle and accept the record once it exists.
_DEPENDENCY_REFUSAL_RE = re.compile(r"^dependency: (\w+) (\S+) is not in this registry")


def dependency_refusal_reason(parent_table: str, parent_id: str, record_type: str) -> str:
    return (f"dependency: {parent_table} {parent_id} is not in this registry -- a {record_type} record is a ledger fact and cannot wait in the "
            f"blocked queue; re-selected every cycle and accepted once the {parent_table} folds (the refusal fact stands), or correct the id")


def dependency_refusal_parent(reason: Optional[str]) -> Optional[Tuple[str, str]]:
    """``(parent_table, parent_id)`` of a dependency refusal's reason (:func:`dependency_refusal_reason`), else None."""
    m = _DEPENDENCY_REFUSAL_RE.match(reason or "")
    return (m.group(1), m.group(2)) if m else None


#: The pseudo-parent of a dependency refusal on a FILING (records.LEDGER_FILING_CHILDREN; the fifth closing pass, 2026-10-09): a child whose
#: `analysis` header the record index does not hold yet is refused `dependency: analysis_filing <sha> is not in this registry ...`, and the
#: retry pass looks the sha up in ``records.LEDGER_ORDERED_RECORD_INDEX`` (indexed = folded, whatever its outcome) instead of in a table,
#: then applies the child again -- accepted when the header is the kept filing or a duplicate of it, refused by name when it stands refused.
FILING_PARENT = "analysis_filing"


def filing_dependency_refusal_reason(filing_sha: str, parent_table: str, parent_id: str, record_type: str) -> str:
    """The reason of a child record whose filing has not folded: parseable by :func:`dependency_refusal_parent` as (``FILING_PARENT``, sha)."""
    return (f"dependency: {FILING_PARENT} {filing_sha} is not in this registry -- the {record_type} names the filing (the `{parent_table}` header "
            f"record) {filing_sha[:12]} of {parent_table} {parent_id}, which has not folded; a ledger fact cannot wait in the blocked queue, so it "
            f"is re-selected every cycle and judged against that filing once it folds (the refusal fact stands), or correct the binding")


def order_for_ingest(root: str, work: Iterable[Tuple[str, Optional[str]]]) -> List[Tuple[str, Optional[str]]]:
    """A cycle's files in the ledger's own order — (first record's written_utc as peeked, logical sha, uri), the
    ranker's key with the peek standing in for the exact first key — so ranks are assigned in order and a
    transaction rarely has to re-rank (F4). Files of one revision directory are kept adjacent (bundle key after the
    stamp) so the chunker can hold them in one transaction; among files sharing a stamp that may cost a re-rank."""
    def key(w):
        uri = logical_uri(root, w[0])
        return (R.peek_written_utc(w[0]), bundle_of_uri(uri) if revision_of_uri(uri) else "", w[1] or "", uri)
    return sorted(work, key=key)


def source_of_revision_dir(root: str, rev_dir: str, conn: Optional[sqlite3.Connection] = None) -> str:
    """The ``source_id`` a revision directory or activation bundle belongs to: the database's row for its revision when there is
    one, else the ``source_revision`` record its marker file carries (a few KB; the marker is the only file read); the revision
    id itself when neither answers (a torn directory), so the grouping below still has a key."""
    uri = logical_uri(root, rev_dir).rstrip("/") + "/"
    rev = revision_of_uri(uri) or uri
    if conn is not None:
        try:
            row = conn.execute("SELECT source_id FROM source_revision WHERE revision_id = ?", (rev,)).fetchone()
        except sqlite3.Error:
            row = None
        if row is not None and row[0]:
            return str(row[0])
    try:
        for fn in sorted(os.listdir(rev_dir)):
            if "source_revision" not in fn or not (fn.endswith(".jsonl") or fn.endswith(".jsonl.gz")):
                continue
            try:
                for _, rec, _ in R.read_record_file(os.path.join(rev_dir, fn), compute_sha=False):
                    if rec.get("record_type") == "source_revision" and (rec.get("payload") or {}).get("source_id"):
                        return str(rec["payload"]["source_id"])
            except (ValueError, OSError, EOFError, gzip.BadGzipFile):
                continue
    except OSError:
        pass
    return rev


def group_revision_dirs(root: str, rev_dirs: Sequence[str], conn: Optional[sqlite3.Connection] = None) -> List[str]:
    """A cycle's revision directories and activation bundles SOURCE-MAJOR: every directory of one source adjacent, in the order
    the cycle first met each source, and in the LEDGER's order within the source (a stable sort: the source's own directories and
    bundles keep their relative order -- a revision directory before the bundles that re-activate it, the bundles by stamp).
    Processing order never moves a rank (every file a cycle folds arrives at the cycle's clock), but it decides which
    bundle of a held revision is the real attempt and which fold as no-ops (:meth:`Ingestor._noop_held_bundle`, 2026-09-29): with a
    source's bundles folded right behind its first, a cycle makes one real attempt per held revision, and a wall-clock budget
    cutting the cycle short costs at most one extra attempt per unfinished source per cycle -- 99 real attempts and ~1,550 no-ops
    for the 2026-09-29 backlog instead of ~1,650 attempts. Grouping by SOURCE and never by revision keeps what the fold depends on
    in ledger order: whether a bundle finds its revision the head (the head no-op) is decided by the same source's earlier
    revisions, which a per-revision grouping would have moved behind it (an A -> B -> A source's re-activation of A folding before
    B's directory, a no-op live and a re-emission on replay, or the reverse). The replay (:func:`rebuild`) groups the same way."""
    first: Dict[str, int] = {}
    keyed: List[Tuple[int, int, str]] = []
    for i, d in enumerate(rev_dirs):
        sid = source_of_revision_dir(root, d, conn)
        first.setdefault(sid, i)
        keyed.append((first[sid], i, d))
    return [d for _, _, d in sorted(keyed)]


def revision_dir_files(root: str, rev_dir: str, only: Optional[Collection[str]] = None,
                       unlisted: Optional[List[str]] = None) -> List[Tuple[str, Optional[str]]]:
    """Every record file of a revision directory, in name order (the marker sorts last only by our processing rule). ``only``
    (a replay bounded by an allowlist of logical uris) keeps exactly the files the list names: a file on
    disk the list does not name is NOT applied -- it arrived after the head the replay reproduces -- and its logical uri is
    appended to ``unlisted`` when a list is given, so the caller can log it. Needs ``root`` to spell the uris; a caller with no
    root (the prefetcher's name-only scan) passes none and gets the whole listing."""
    out = []
    for fn in sorted(os.listdir(rev_dir)):
        if fn.endswith(".jsonl") or fn.endswith(".jsonl.gz"):
            path = os.path.join(rev_dir, fn)
            if only is not None and root:
                uri = logical_uri(root, path)
                if uri not in only:
                    if unlisted is not None:
                        unlisted.append(uri)
                    continue
            out.append((path, None))
    return out


# --------------------------------------------------------------------------------------------
# ranking (ingest_seq derived, never allocated)
# --------------------------------------------------------------------------------------------


#: The ledger's order, as SQL: what ``ingest_seq`` is the dense rank of (ONE spelling for the head lookup and the dense pass).
LEDGER_ORDER_SQL = "arrival_utc, first_written_utc, file_sha256, file_uri"
LedgerKey = Tuple[str, str, str, str]


class Ranker:
    """Maintains ``record_ingest.ingest_seq`` as the dense rank of record FILES in ARRIVAL order — the key
    ``(arrival_utc, first_written_utc, file_sha256, file_uri)``. Files are appended PROVISIONALLY; when a
    file sorts before an already-ranked one (``dirty``) :meth:`finalize` performs ONE dense re-rank at the end of the
    transaction (a window function + a temp old→new map applied to every table carrying ``ingest_seq``).

    **Arrival.** ``arrival_utc`` is set ONCE, when the file enters the ledger, by :meth:`arrival_for` — the one
    function every path uses:

    * a **harvester-originated file** (``records/harvester/…``) takes its own ``first_written_utc``: the harvester
      stamped it at fold time (``ingest_once`` stamps the cycle's facts at :meth:`next_arrival`, strictly after the
      head), so its stamp is its arrival on the live path and on a replay alike, and no fact is written for it (a fact
      about the file carrying the facts would need a fact of its own, forever).
    * on the **live path** (``arrivals is None``) every other file takes ``max(cycle clock, head arrival + 1 ms)`` —
      strictly after the head, so a new file is ALWAYS head + 1, nothing already ranked moves, and :meth:`finalize`
      has no work. Every pulled file used to be a late arrival under the producer-stamp order (the harvester's own
      facts stamped the head at "now" each cycle; a file pulled from a cluster carried yesterday's stamp), and each
      cost one dense pass over the 35 seq-bearing tables — 307 s for 22,760 ledger rows. The cycle then
      writes a ``file_arrival`` fact for the file (``derive_file_arrival_records``), so the order is in the log.
    * on a **replay** (``arrivals`` = the facts read out of the log, ``read_arrival_facts``) a file takes the arrival
      the live registry recorded, and a file with no fact — never folded live, or folded at the very millisecond
      of its stamp — takes its ``first_written_utc``: the same key the live path had, and for a log with no facts at
      all the earlier order ``(first_written_utc, sha, uri)`` exactly.

    A rebuild is therefore the ONLY way to (re)create an index from a log with the same ranks: a live fold of an
    empty database stamps the clock and writes NEW facts (the replay keeps the EARLIEST fact per file). The
    arrival is monotone by construction: a producer clock ahead of the grading host's, carried into a ``run_ingest``
    stamp, drags the head forward, and every arrival after it is ``head + 1 ms`` until the clock passes.

    ``defer=True`` (a REPLAY, and the live cycle too): the per-transaction re-rank is skipped and every
    rank stays provisional until :meth:`finalize` is called with ``final=True`` once, at the end. A dense rank is a
    pure function of the ledger's keys, so the ranks written are the same either way; only WHEN they are made dense
    changes. A replay applies plain files before revision directories (a revision's dependencies must be present
    when it activates), so a revision-directory file may sort before files applied earlier — and re-ranking per
    staging window rewrote ``ingest_seq`` across every seq-bearing table once per transaction: O(transactions ×
    rows), 237 GB written for a 500 MB database on the 2026-09-17 live rebuild."""

    def __init__(self, conn: sqlite3.Connection, defer: bool = False, arrivals: Optional[Mapping[str, str]] = None):
        self.conn = conn
        self.defer = defer
        self.arrivals = arrivals          # a replay's recorded arrivals by file_uri; None = the live path (the clock)
        self.deferred_dirty = False       # defer: an inversion was seen in some transaction; the final re-rank is owed
        self.reload()

    def reload(self) -> None:
        """Re-read the ledger's head (call at the start of every transaction that assigns ranks)."""
        self.dirty = False
        self.reranks = 0
        row = self.conn.execute("SELECT ingest_seq FROM record_ingest ORDER BY ingest_seq DESC LIMIT 1").fetchone()
        self.max_seq: int = int(row[0]) if row else 0
        row2 = self.conn.execute(f"SELECT {LEDGER_ORDER_SQL} FROM record_ingest ORDER BY arrival_utc DESC, first_written_utc DESC, "
                                 "file_sha256 DESC, file_uri DESC LIMIT 1").fetchone()
        self.max_key: Optional[LedgerKey] = (row2[0], row2[1], row2[2], row2[3]) if row2 else None

    def next_arrival(self, now: str) -> str:
        """The stamp a file (or a harvester fact) entering the ledger NOW takes: the clock, or one millisecond after
        the head's arrival when the clock has not passed it — strictly after every ranked file."""
        if self.max_key is None or self.max_key[0] < now:
            return now
        return R.epoch_ms_to_utc(R.utc_to_epoch_ms(self.max_key[0]) + 1)

    def arrival_for(self, file_uri: str, first_written_utc: str, now: str) -> str:
        """THE arrival of a file entering the ledger (see the class docstring) — one function, every caller."""
        if is_harvester_file(file_uri):
            return first_written_utc
        if self.arrivals is not None:
            return self.arrivals.get(file_uri, first_written_utc)
        return self.next_arrival(now)

    def seq_of(self, file_uri: str) -> Optional[int]:
        """The ledger rank of a file (None when the file is not in the ledger) — the ONE ingest_seq lookup."""
        row = self.conn.execute("SELECT ingest_seq FROM record_ingest WHERE file_uri = ?", (file_uri,)).fetchone()
        return int(row[0]) if row else None

    def uri_of(self, record_file_id: int) -> Optional[str]:
        row = self.conn.execute("SELECT file_uri FROM record_ingest WHERE record_file_id = ?", (record_file_id,)).fetchone()
        return str(row[0]) if row else None

    def assign(self, file_uri: str, file_sha: str, first_written_utc: str, n_records: int, now: Optional[str] = None) -> int:
        """Rank one FILE (idempotent per uri; a changed sha re-keys the row's tie-break, never its arrival). ``now`` is
        the cycle's clock, the floor of a live arrival."""
        rfid = R.record_file_id_for_uri(file_uri)
        row = self.conn.execute("SELECT ingest_seq, file_sha256, first_written_utc FROM record_ingest WHERE file_uri = ?", (file_uri,)).fetchone()
        if row is not None:
            if row[1] == file_sha and row[2] <= first_written_utc:
                self.conn.execute("UPDATE record_ingest SET n_records = ? WHERE file_uri = ?", (n_records, file_uri))
                return int(row[0])
            # the arrival stays: the file's place among files of OTHER arrivals is unchanged; only its tie-break moved
            self.conn.execute("UPDATE record_ingest SET file_sha256 = ?, first_written_utc = ?, n_records = ? WHERE file_uri = ?",
                              (file_sha, min(first_written_utc, row[2]), n_records, file_uri))
            self.dirty = self.deferred_dirty = True
            self.reranks += 1
            return int(row[0])
        arrival = self.arrival_for(file_uri, first_written_utc, now or R.utc_now())
        key: LedgerKey = (arrival, first_written_utc, file_sha, file_uri)
        if self.max_key is not None and key < self.max_key:
            self.dirty = self.deferred_dirty = True
            self.reranks += 1
        else:
            self.max_key = key
        seq = self.max_seq + 1
        self.max_seq = seq
        self.conn.execute("INSERT INTO record_ingest(record_file_id, file_uri, file_sha256, first_written_utc, arrival_utc, n_records, ingest_seq) "
                          "VALUES (?,?,?,?,?,?,?)", (rfid, file_uri, file_sha, first_written_utc, arrival, n_records, seq))
        return seq

    def finalize(self, final: bool = False) -> int:
        """Dense re-rank of the ledger and every seq-bearing table; returns the rows remapped in the ledger. In
        ``defer`` mode nothing happens until the one call with ``final=True`` (inside a transaction), which re-ranks
        iff any transaction of the replay saw an inversion."""
        if self.defer and not final:
            return 0
        if not (self.dirty or (final and self.deferred_dirty)):
            return 0
        self.deferred_dirty = False
        c = self.conn
        c.execute("DROP TABLE IF EXISTS temp.seqmap")
        c.execute("CREATE TEMP TABLE seqmap AS SELECT ingest_seq AS old, "
                  f"ROW_NUMBER() OVER (ORDER BY {LEDGER_ORDER_SQL}) AS new FROM record_ingest")
        c.execute("DELETE FROM temp.seqmap WHERE old = new")
        c.execute("CREATE INDEX temp.idx_seqmap_old ON seqmap(old)")
        n = int(c.execute("SELECT COUNT(*) FROM temp.seqmap").fetchone()[0])
        if n:
            # the files whose rank moves, read BEFORE the ledger is rewritten: their rows are what the item tables remap
            moved = [int(r[0]) for r in c.execute("SELECT record_file_id FROM record_ingest WHERE ingest_seq IN (SELECT old FROM temp.seqmap)")]
            c.execute("UPDATE record_ingest SET ingest_seq = -(SELECT new FROM temp.seqmap WHERE seqmap.old = record_ingest.ingest_seq) "
                      "WHERE ingest_seq IN (SELECT old FROM temp.seqmap)")
            c.execute("UPDATE record_ingest SET ingest_seq = -ingest_seq WHERE ingest_seq < 0")
            indexed = self._remap_pairs(moved) if INDEXED_SEQ_REMAP else False
            for t in SEQ_TABLES:
                if indexed and t in REVISION_INDEXED_TABLES:
                    # the moved files' rows through the (source_revision_id, record_file_id) index -- never a scan of the item tables
                    c.execute(f"UPDATE {t} SET ingest_seq = (SELECT new FROM temp.seqmap WHERE seqmap.old = {t}.ingest_seq) "
                              f"WHERE (source_revision_id, record_file_id) IN (SELECT rev, rfid FROM temp.seqpairs) "
                              f"AND ingest_seq IN (SELECT old FROM temp.seqmap)")
                else:
                    c.execute(f"UPDATE {t} SET ingest_seq = (SELECT new FROM temp.seqmap WHERE seqmap.old = {t}.ingest_seq) "
                              f"WHERE ingest_seq IN (SELECT old FROM temp.seqmap)")
            c.execute("DROP TABLE IF EXISTS temp.seqpairs")
        c.execute("DROP TABLE temp.seqmap")
        reranks = self.reranks
        self.reload()
        self.reranks = reranks
        return n

    def _remap_pairs(self, moved: Sequence[int]) -> bool:
        """Materialize ``temp.seqpairs`` -- every ``(source_revision_id, record_file_id)`` an item row of a moved file can carry --
        so the dense pass rewrites the item tables through their ``(source_revision_id, record_file_id)`` index instead of a
        scan (2026-09-24: the pass read every row of ``item_result`` / ``item_flags`` -- 11.5 M and 6.4 M rows, no index on
        ``ingest_seq`` -- to remap the rows of 4 to 438 files, 729-1,029 s per pass and 13-20k s a day on the live database).
        Exact, by two facts the schema states: every row's ``ingest_seq`` IS its producing file's ledger rank (each writer sets
        both from the same file, the dense pass rewrites both), so the rows whose rank moved are the moved files' rows; and every
        ``source_revision_id`` is ``''`` or a ``source_revision`` row (the FK trigger; a pending revision's placeholder row exists
        before its rows are staged), so ``(every revision ∪ '') × (moved files)`` names every such row. The ``ingest_seq IN
        (old ranks)`` guard stays on the statement, so a row outside the mapping is never rewritten. Returns False -- the scan
        instead, equally exact -- when the pair set would exceed :data:`SEQ_REMAP_PAIRS_MAX`."""
        c = self.conn
        n_revs = int(c.execute("SELECT COUNT(*) FROM source_revision").fetchone()[0]) + 1
        if not moved or n_revs * len(moved) > SEQ_REMAP_PAIRS_MAX:
            return False
        c.execute("DROP TABLE IF EXISTS temp.seqfiles")
        c.execute("CREATE TEMP TABLE seqfiles(rfid INTEGER PRIMARY KEY)")
        c.executemany("INSERT INTO temp.seqfiles(rfid) VALUES (?)", [(r,) for r in moved])
        c.execute("DROP TABLE IF EXISTS temp.seqpairs")
        c.execute("CREATE TEMP TABLE seqpairs AS SELECT sr.rev AS rev, f.rfid AS rfid FROM "
                  "(SELECT revision_id AS rev FROM source_revision UNION ALL SELECT '') sr, temp.seqfiles f")
        c.execute("DROP TABLE temp.seqfiles")
        return True


# --------------------------------------------------------------------------------------------
# applying parsed files
# --------------------------------------------------------------------------------------------


@dataclass
class IngestReport:
    files: int = 0
    records: int = 0
    ok: int = 0
    refused: int = 0
    blocked: int = 0
    unblocked: int = 0
    duplicates: int = 0
    conflicts: int = 0
    reranks: int = 0
    rerank_rows: int = 0            # ledger rows the cycle's one dense pass remapped
    transactions: int = 0
    longest_transaction_s: float = 0.0
    write_s: float = 0.0
    parse_s: float = 0.0
    harvester_records: int = 0        # run_ingest / ruling_state facts the cycle wrote (never the arrival facts below)
    lifecycle_regressions_held: int = 0   # run_state events the fold held: each would have moved a run's lifecycle backwards (reducer.lifecycle_regresses)
    arrival_facts: int = 0            # file_arrival facts the cycle wrote: one per file that arrived in it
    refusal_facts: int = 0            # record_refusal facts the cycle wrote: the START pass (judgments with no fact yet, the previous
                                      # judgment of a row about to be re-judged, the backfill of rows refused before the stamp) + the END pass
                                      # (this cycle's own refusals and re-judgments); each folded in the cycle that wrote it
    refusal_fact_records: int = 0     # record_refusal records APPLIED this cycle (the facts above, plus a pending fact file folded at step 0):
                                      # bookkeeping the fold caused, counted apart from `ok` like the arrival records
    arrival_records: int = 0          # file_arrival records applied this cycle (bookkeeping the fold CAUSED, like the page below)
    analysis_records: int = 0         # analysis / analysis_input / analysis_artifact records applied this cycle: the ledger-fact rows
                                      # of a filed analysis, counted apart from `ok` as the refusal facts are
    dependency_refusals_retried: int = 0   # standing dependency refusals of ledger-fact records re-selected this cycle (their parent looked up again)
    dependency_refusals_resolved: int = 0  # of those, the records accepted because the parent now exists (the refusal row withdrawn; its fact stands)
    refilings_rejudged: int = 0            # records of an ordered ledger-fact key judged again this cycle because the key's kept record moved
                                           # (Ingestor._rejudge_refilings): a new conflict among them counts under `refused` as well
    filing_children_rejudged: int = 0      # child records of a filing (analysis_input / analysis_artifact) derived again this cycle because the
                                           # kept filing of their analysis moved (Ingestor._rejudge_filing_children): a withdrawn child counts under `refused` too
    deliveries_rejudged: int = 0           # record_delivery rows moved this cycle because a record of the file was judged again after its fold
                                           # (Ingestor._move_record_delivery / _move_delivery): a displaced filing, a re-judged re-filing or child,
                                           # a dependency refusal accepted on retry -- one move per record whose verdict changed sides, in EVERY
                                           # file that carried the record (the carrier audit, records.LEDGER_ORDERED_CARRIERS)
    gzipped: int = 0
    deferred: int = 0                 # files that changed while being read; retried next cycle
    files_replaced_refused: int = 0   # physical files at an INGESTED uri with other bytes (a plain twin after the .gz, an edit after the fold): refused by name, the ledger row stands
    blocked_transactions: int = 0     # transactions the blocked-queue re-evaluation used
    revision_transactions: int = 0    # activation attempts of revision directories, committed or rolled back
    staging_transactions: int = 0     # bounded transactions that staged a revision's rows inactive before activation
    reclaim_transactions: int = 0     # bounded bookkeeping transactions after activations (withdrawn marks of a re-activated head)
    dimension_transactions: int = 0   # dimension revision directories applied whole, one transaction each (records/dimensions/<rev>/)
    dimension_write_s: float = 0.0    # seconds under the write lock those transactions took (the load's lock hold time)
    revisions_held: int = 0           # revision directories left pending (a constituent blocked / refused; retried whole)
    revisions_held_run_drop: int = 0  # of revisions_held, the markers held for dropping run ids of the accepted head undeclared
    revisions_held_retirement_head: int = 0   # of revisions_held, the retirement markers held because the head they were computed against is not the live one (registry-hardening 3a, round 2)
    revisions_held_skipped: int = 0   # of revisions_held, the held directories NOT re-attempted this cycle because nothing about them moved since the
                                      # last attempt (the attempt identity memo): no staging, no rollback
    revisions_applied: int = 0        # revision directories / activation bundles of the cycle's NEW work apply_revision ran over (each
                                      # committed on its own: bounded staging windows, one activation, or the held path's one transaction)
    revisions_noop: int = 0           # of the bundles applied, those folded as NO-OPS: a later bundle of a revision this cycle held (delivered
                                      # blocked under the same hold, nothing staged), or a bundle of the accepted head (marker + status only)
    revisions_deferred_budget: int = 0    # of the cycle's new revision directories / bundles, those NOT taken because the wall-clock budget was spent
    plain_files_deferred_budget: int = 0  # of the cycle's new plain files, those not taken for the same reason
    budget_s: Optional[float] = None  # the cycle's wall-clock budget (``ingest --budget-s``), None when unbounded
    stopped: Optional[str] = None     # why the cycle stopped taking new work early: "budget"; None when it took everything
    elapsed_s: float = 0.0            # wall time of the whole cycle under the lock (inbox merge included; the cluster pull runs before the lock and is not counted)
    completed_utc: Optional[str] = None   # the clock at the cycle's completion: what state/harvest_cycle.json records as last_completed_cycle_utc
    namespaces: List[str] = field(default_factory=list)   # log namespaces a rebuild replayed
    paused: bool = False
    unlisted_ignored: int = 0         # files on disk inside a replayed revision directory that the replay's allowlist does not name: ignored, logged
    unlisted_sample: List[str] = field(default_factory=list)   # the first few of them, by logical uri
    free_bytes: int = 0
    changed: bool = False
    status_page_records: int = 0      # the status page's own artifact(kind=page) registrations applied this cycle
    self_change_only: bool = False    # changed ONLY by those and by arrival facts: `ingest --html` must not re-render on its own last page (it would loop)
    refusals: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


class _Existence:
    """Per-transaction cache of parent existence for the item-row fast path."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.cache: Dict[Tuple[str, str, str], bool] = {}

    def exists(self, table: str, col: str, val) -> bool:
        k = (table, col, str(val))
        if k not in self.cache:
            self.cache[k] = self.conn.execute(f"SELECT 1 FROM {table} WHERE {col} = ? LIMIT 1", (val,)).fetchone() is not None
        return self.cache[k]

    def note(self, table: str, col: str, val) -> None:
        self.cache[(table, col, str(val))] = True

    def accepted_revision(self, revision_id: str) -> bool:
        k = ("source_revision/accepted", "revision_id", revision_id)
        if k not in self.cache:
            self.cache[k] = self.conn.execute("SELECT 1 FROM source_revision WHERE revision_id = ? AND accepted = 1 LIMIT 1",
                                              (revision_id,)).fetchone() is not None
        return self.cache[k]

    def note_accepted(self, revision_id: str) -> None:
        self.cache[("source_revision/accepted", "revision_id", revision_id)] = True

    _ITEM_ROW_SQL = ("SELECT 1 FROM item_result WHERE " + " AND ".join(f"{c} = ?" for c in R.NATURAL_KEYS["item_withdrawal"])
                     + " AND source_revision_id = '' LIMIT 1")

    def item_row_exists(self, key: Mapping[str, object]) -> bool:
        """The natively emitted item_result row an ``item_withdrawal`` key names: one probe of the table's key."""
        k = ("item_result/native", "key", canonical_json(dict(key)))
        if k not in self.cache:
            vals = tuple(key[c] for c in R.NATURAL_KEYS["item_withdrawal"])
            self.cache[k] = self.conn.execute(self._ITEM_ROW_SQL, vals).fetchone() is not None
        return self.cache[k]


class _Prefetcher:
    """A replay's item files parsed AHEAD of the revision they belong to, in the worker pool the plain phase used and the
    revision phase left idle (F16). Files are submitted in the order the replay will consume them, at most ``window`` in
    flight (memory O(window), F1); :meth:`take` hands the consumer its file's whole parse — the same ``parse_file`` the
    plain phase runs, so the items are the same worker-built rows — or None when the file was never submitted (a name
    that did not look like an item file, a file above LARGE_FILE_BYTES), and the consumer streams it in-process as
    before. A file the consumer skips (already staged) is discarded from the queue as it is passed. Only an
    optimisation: which files are prefetched changes nothing about what is staged."""

    def __init__(self, root: str, executor, paths: Sequence[str], window: int):
        self.root = root
        self.ex = executor
        self.pending: List[str] = list(paths)
        self.queue: "collections.deque" = collections.deque()      # (path, future), in consumption order
        self.window = max(1, window)
        self._fill()

    def _fill(self) -> None:
        while self.pending and len(self.queue) < self.window:
            path = self.pending.pop(0)
            self.queue.append((path, self.ex.submit(parse_file, (self.root, path, None, 0))))

    def take(self, path: str) -> Optional[ParsedFile]:
        if path not in self.pending and not any(p == path for p, _ in self.queue):
            return None
        while True:
            if not self.queue:
                self._fill()
                if not self.queue:
                    return None
            p, fut = self.queue.popleft()
            if p == path:
                res = fut.result()
                self._fill()
                return res
            fut.cancel()          # a file the consumer passed over (already staged): its parse is dropped, never awaited
            self._fill()

    @staticmethod
    def item_files(rev_dirs: Sequence[str], root: str = "", only: Optional[Collection[str]] = None) -> List[str]:
        """The files of the revision directories whose NAME says item-scale, in consumption order, below the streaming
        size — the ones pass B would stream (a file named otherwise but carrying items is simply streamed). ``root`` + ``only``:
        a replay's allowlist, so nothing unlisted is parsed ahead either."""
        out: List[str] = []
        for d in rev_dirs:
            for path, _ in revision_dir_files(root, d, only=only):
                m = R.RECORD_FILE_RE.match(os.path.basename(path))
                if m is None or m.group("type") not in R.ITEM_SCALE_RECORD_TYPES:
                    continue
                try:
                    if os.path.getsize(path) > LARGE_FILE_BYTES:
                        continue
                except OSError:
                    continue
                out.append(path)
        return out


def parent_absent(conn: sqlite3.Connection, ptable: str, pid: str) -> bool:
    """No row of ``ptable`` carries ``pid`` under any column a dependency check may look it up by; a revision counts as present only
    when accepted. Unknown (pseudo) parent tables are never declared absent. The blocked queue re-attempts a record only once this reads
    False for its missing parent (:meth:`Ingestor.reevaluate_blocked`); the claim probe asks the same question."""
    if ptable == "source_revision":
        return conn.execute("SELECT 1 FROM source_revision WHERE revision_id = ? AND accepted = 1 LIMIT 1", (pid,)).fetchone() is None
    cols = _PARENT_COLUMNS.get(ptable)
    if not cols or ptable not in S.TABLE_BY_NAME:
        return False
    return all(conn.execute(f"SELECT 1 FROM {ptable} WHERE {col} = ? LIMIT 1", (pid,)).fetchone() is None for col in sorted(cols))


#: A held revision whose attempt identity stands is re-attempted at most once per this many seconds
#: (``[harvest] held_sweep_s``; the window's uncapped cycles and `registry ingest --retry-held` re-attempt every held revision regardless)
HELD_SWEEP_S = 3600.0


#: the keys of the marker's queue-row record that enter the attempt identity: the hold's kind and reason, the refused kind's exact-hold key
#: (refusing file, its sha, the judging stamp) -- a row stripped of them (a hold recorded before a rule) re-attempts once, as the rules say
HOLD_IDENTITY_EXTRA_KEYS: Tuple[str, ...] = ("kind", "blocked_on", "revision_id", "refused_file", "refused_file_sha", "judged_with", "rule")


def hold_identity(root: str, rev_dir: str, marker_sha: Optional[str], kind: Optional[str], parent_table: Optional[str], parent_id: Optional[str],
                  parent_present: Optional[bool] = None, marker_extra: Optional[Mapping[str, object]] = None) -> str:
    """The attempt identity of a held revision directory: sha256 over the directory's sorted listing (root-relative path, size, mtime
    of every regular file), the marker's sha, the hold's kind and parent, the hold's own record on the marker's queue row
    (:data:`HOLD_IDENTITY_EXTRA_KEYS`), the judging code's stamp and -- for a blocked kind -- whether the parent is present. A changed or
    added file, a landed fix, an arrived parent, a different marker or a changed hold record moves it; a re-emission of the same bytes under
    fresh stamps does too (the mtimes), which is the directory being re-published and so a real attempt."""
    h = hashlib.sha256()
    try:
        for dirpath, dirnames, filenames in os.walk(rev_dir):
            dirnames.sort()
            for fn in sorted(filenames):
                p = os.path.join(dirpath, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                h.update(f"{os.path.relpath(p, rev_dir)}\t{st.st_size}\t{st.st_mtime_ns}\n".encode("utf-8"))
    except OSError:
        pass
    h.update(f"marker={marker_sha or ''}\nkind={kind or ''}\nparent={parent_table or ''}:{parent_id or ''}\nparent_present={parent_present}\n"
             f"judged_with={validator_stamp()}\n".encode("utf-8"))
    rec = {k: marker_extra.get(k) for k in HOLD_IDENTITY_EXTRA_KEYS if marker_extra is not None and k in marker_extra}
    h.update(("record=" + canonical_json(rec) + "\n").encode("utf-8"))
    return h.hexdigest()


def refused_hold_exact(conn: sqlite3.Connection, ext: Mapping[str, object], listed_uris: Optional[Set[str]] = None,
                       retry_refused: bool = False, root: Optional[str] = None) -> bool:
    """Whether a ``kind = refused`` marker row records an EXACT hold (2026-09-30): the refusing file's bytes are the ones the refusal
    was recorded on -- its logical sha in ``record_delivery`` equals the row's ``refused_file_sha`` and the file has not changed (the
    cycle's listing did not name it, ``listed_uris``; or, asked outside a cycle with ``root``, its bytes on disk still hash to that sha)
    -- and the judging code is the one that refused it (``judged_with`` == :func:`validator_stamp`). Such a hold is not re-attempted:
    the same bytes under the same code refuse again, and the refusal on record (``record_refusal``, ``state/refused.jsonl``) is the
    judgment. A marker without the key (a hold recorded before this rule), a changed file, a moved stamp (a landed fix, or a changed
    CHECK) or ``retry_refused`` (``registry ingest --retry-refused``: for a refusal that depended on OTHER rows rather than on the
    record's bytes or the code) re-attempts -- once, after which the attempt records the key afresh and the new judgment stands. A
    refused hold whose source a later accepted read has SUPERSEDED is exact whatever the stamp (it can never become the head). Pure in
    bytes, code and the tables, so a replay under the same code makes the same decision; a verify across two code trees differs on the
    held markers' ``extra_json`` (as it did on ``blocked_on``). Module-level since 2026-10-04: the claim probe
    (``pending.pending_identity_corrections``) asks whether the next cycle re-attempts a hold with exactly this rule."""
    if retry_refused:
        return False
    rev = str(ext.get("revision_id") or "")
    if rev and conn.execute("SELECT 1 FROM source_revision WHERE revision_id = ? AND accepted = 0 AND superseded_by_revision_id IS NOT NULL",
                            (rev,)).fetchone() is not None:
        # a refused hold a LATER accepted read of the same source has superseded can never become the head (the run-drop rule of
        # 2026-09-23): re-attempting it would only re-refuse its bytes, or activate it already superseded and erase the hold's
        # provenance -- the refusal and the queue row are the record of what the reader did. Exact whatever the stamp; only
        # `--retry-refused` re-attempts it (three relu_paired and one r2_export bundle re-read every live cycle this way)
        return True
    if ext.get("judged_with") != validator_stamp():
        return False
    furi, fsha = ext.get("refused_file"), ext.get("refused_file_sha")
    if not furi or not fsha:
        return False
    if listed_uris is not None:
        if furi in listed_uris:
            return False
    elif root is not None:
        path = resolve_uri(root, str(furi))
        if path is None:
            return False
        try:
            if R.file_logical_sha(path) != str(fsha):
                return False                            # the bytes changed: the next cycle lists the file and re-attempts the hold
        except (OSError, EOFError, gzip.BadGzipFile):
            return False
    row = conn.execute("SELECT file_sha256 FROM record_delivery WHERE file_uri = ?", (str(furi),)).fetchone()
    return row is not None and str(row[0]) == str(fsha)


def apply_order(v: VAL.Validated) -> Tuple[int, Tuple[str, int, str]]:
    """The order a transaction's run-scale records are APPLIED in: the fold's own order -- ``Validated.order_key`` =
    ``reducer.order_key``, ``(written_utc, tie_rank, record_sha256)`` -- with a revision's ``source_revision`` marker ahead of the records
    that wait on it, so they land in the marker's transaction. The fold itself is order-independent (every apply refolds the key's whole
    history), so this decides only what each history append costs and which transaction a marker lands in; it is spelled THROUGH
    ``order_key`` so that no sort in this module reads as a second spelling of the rule (two apply sorts spelled ``(written_utc, sha)``
    outlived the tie-break for five days, harmless to the fold and the only places a reader could mistake for it)."""
    return (0 if v.record_type == "source_revision" else 1, v.order_key)


class Ingestor:
    def __init__(self, conn: sqlite3.Connection, root: str, report: Optional[IngestReport] = None, live: bool = True,
                 defer_ranks: bool = False, arrivals: Optional[Mapping[str, str]] = None):
        self.conn = conn
        self.root = root
        SA.install(root)            # the root's site aliases: a legacy spelling of a site label folds as its label
        self.report = report or IngestReport()
        # a replay ranks once, at the end (Ranker.finalize(final=True)) and orders files by the arrivals the log recorded
        self.ranker = Ranker(conn, defer=defer_ranks, arrivals=arrivals)
        self.reducer = Reducer(conn, self.ranker.seq_of)
        self.live = live
        self._blocked: Optional[Set[str]] = None        # blocked shas at transaction start; None = not inside a chunk (F5)
        self._indexed: Set[Tuple[str, str]] = set()     # record_run_index rows written in this transaction (F5)
        self.learned: Dict[str, Dict[str, object]] = {}  # path -> stat sidecar entry of every file parsed this run (F12)
        self._partial: Dict[str, Tuple[Dict[str, int], Dict[str, int], int, int]] = {}   # per-file counts across parts
        self._activating: Optional[str] = None           # the revision whose bundle this transaction applies (its constituents need no marker yet)
        self._staged_cache: Optional[Tuple[str, List[Tuple[VAL.Validated, str]]]] = None   # one attempt's validated staged constituents (F16)
        self.prefetch: Optional[_Prefetcher] = None       # a replay's item files parsed ahead in the worker pool (F16)
        self.file_shas: Dict[str, str] = {}               # path -> logical sha the cycle's listing computed (a no-op bundle's files are not re-hashed)
        self.listed_uris: Set[str] = set()                # the logical uris the cycle's listing found new or changed: a refused hold on one of them is not exact
        self._replaced_refused: Set[str] = set()          # uris refused this pass as other bytes at an ingested uri: their later parts are skipped
        self._dimension_bundle = False                    # inside apply_dimension_bundle: a --force re-publication APPENDS to the seed files by design (exempt)
        self.retry_refused: bool = False                  # `registry ingest --retry-refused`: re-attempt every refused hold this cycle, exact or not
        self.retry_held: bool = False                     # `registry ingest --retry-held` / the window's uncapped `dimensions load`: re-attempt every held revision this cycle
        self.held_sweep_s: float = HELD_SWEEP_S           # a held revision whose attempt identity stands is re-attempted at most once per this interval ([harvest] held_sweep_s)
        self._held_attempts_cache: Optional[Dict[str, Dict[str, str]]] = None   # state/held_attempts.json, read once per cycle
        self._noop_kind: Optional[str] = None            # set by _apply_revision when a bundle folded as a no-op; read by apply_revision's metric
        self.allowlist: Optional[Set[str]] = None         # a replay bounded to these logical uris: every directory listing of
                                                          # the revision path is filtered through it; None = every file on disk, the live path's rule

    def _rev_files(self, rev_dir: str) -> List[Tuple[str, Optional[str]]]:
        """The record files of a revision directory this fold may apply: the directory's listing, filtered through the allowlist when one
        is set (a file the list does not name is counted and sampled on the report, never applied)."""
        if self.allowlist is None:
            return revision_dir_files(self.root, rev_dir)
        skipped: List[str] = []
        files = revision_dir_files(self.root, rev_dir, only=self.allowlist, unlisted=skipped)
        if skipped:
            self.report.unlisted_ignored += len(skipped)
            room = 20 - len(self.report.unlisted_sample)
            if room > 0:
                self.report.unlisted_sample.extend(skipped[:room])
            self._metric({"kind": "unlisted_ignored", "dir": logical_uri(self.root, rev_dir).rstrip("/"), "n": len(skipped), "sample": skipped[:5]})
        return files

    # -- helpers ----------------------------------------------------------------------------
    def _refused(self, sha: str) -> bool:
        return self.conn.execute("SELECT 1 FROM record_refusal WHERE record_sha256 = ?", (sha,)).fetchone() is not None

    def _ingested_sha(self, uri: str) -> Optional[str]:
        """The logical sha the ledger ranked ``uri`` under (``record_ingest``), or None for a uri never ranked."""
        row = self.conn.execute("SELECT file_sha256 FROM record_ingest WHERE file_uri = ?", (uri,)).fetchone()
        return str(row[0]) if row else None

    def _refuse_replaced(self, pf: ParsedFile, ingested_sha: str, now: str) -> None:
        """An ingested record file is immutable: ``pf`` is a physical file at a logical uri the
        ledger has RANKED (``record_ingest``) whose bytes are not the ones ranked -- a plain ``.jsonl`` arriving beside the harvester's
        ``.jsonl.gz`` with other content, a file edited or appended after its fold -- and is refused BY NAME: one ``record_refusal`` row
        keyed by the arriving file's logical sha (``record_type`` and ``field`` NULL: a file-level judgment), its reason naming the uri
        and both shas, a ``record_refusal_fact`` derived from it like every judgment, the cycle's report and sidecar line. The
        ledger row, the delivery audit and every folded row stand; nothing on disk is moved or removed (the file stays where it arrived,
        beside the ingested one). Before this the file folded as a CHANGED file -- the same ledger row with its sha replaced, its
        records re-folded, 0 refusals: a silent replacement of an ingested record (the L2c-replay probe, 2026-10-06). The same bytes at
        the uri again are a no-op the listing never raises; a file never ranked (wholly refused, unreadable, empty) is not ingested and
        may still change before its first fold. The listing (:func:`new_or_changed_files`) does not raise a replacement this code has
        already refused, so the judgment is made once per (bytes, uri, judging stamp). The ONE changed-file case kept: the files of a
        dimension seed revision directory, which ``dimensions load --force`` re-publishes by APPENDING (tests/test_registry_dimensions_atomic.py)
        and :meth:`apply_dimension_bundle` re-applies whole in one transaction -- never checked here."""
        reason = (f"replaced: {pf.uri} was ingested under sha {ingested_sha} and the file now at that uri hashes to {pf.file_sha}; an "
                  f"ingested record file is immutable (D968) -- the ledger row stands and these bytes are refused by name; publish a "
                  f"correction as a new record file")
        self._refuse(pf.file_sha, pf.uri, None, None, reason, now)
        self._replaced_refused.add(pf.uri)
        self.report.files_replaced_refused += 1
        self.conn.execute("DELETE FROM record_cursor WHERE file_uri = ?", (pf.uri,))        # a cursor for other bytes at the uri is moot
        if pf.stat is not None:
            self.learned[pf.path] = {"size": pf.stat[0], "mtime_ns": pf.stat[1], "ino": pf.stat[2], "sha": pf.file_sha}

    def _rejudge_refilings(self, v: VAL.Validated, now: str) -> None:
        """The kept record of ``v``'s natural key MOVED (an earlier-sorting record arrived late, or an identical one under an earlier key):
        judge EVERY other record of the key the fold has applied (``ordered_fact_record``, records.LEDGER_ORDERED_RECORD_INDEX: the
        duplicates, the displaced, the conflicting) again, against the record the row now carries -- each re-read from its (immutable) file,
        never from the database, and applied: a `duplicate` outcome (identical substance) withdraws its standing conflict refusal, if any (the
        refusal fact in the log stands as history); a `conflict` refuses it by name when no refusal stands (a record accepted as a duplicate of
        the displaced one -- which left no row to find, so the first rendering of this pass, over the standing conflict refusals alone, left it
        unrefused while a rebuild refused it; the second review round, 2026-10-09) or rewrites a standing refusal's reason to name the new kept
        record (the verdict stands: nothing counted, no new fact due). The kept record is the smallest key among the records applied, so no
        other record can take its place here; an outcome that says otherwise is a broken invariant and stops the cycle rather than fold on."""
        assert v.table is not None
        key = {c: v.payload[c] for c in R.NATURAL_KEYS[v.table]}
        prefix = self._refiling_prefix(v)
        by_file: Dict[str, Set[str]] = {}
        for sha, uri, _rt, _utc in self.reducer.ordered_records_of(v.table, key):
            if sha != v.sha and "#" in uri:
                by_file.setdefault(uri.rpartition("#")[0], set()).add(sha)
        for file_uri in sorted(by_file):
            for rv in self._validated_by_sha(file_uri, by_file[file_uri]):
                out = self.reducer.apply_ledger_fact(rv)
                self.report.refilings_rejudged += 1
                standing = self.conn.execute("SELECT reason, record_uri FROM record_refusal WHERE record_sha256 = ?", (rv.sha,)).fetchone()
                has_refiling_refusal = standing is not None and str(standing[0]).startswith(prefix)
                if out.status == "duplicate":
                    if has_refiling_refusal:
                        self.conn.execute("DELETE FROM record_refusal WHERE record_sha256 = ?", (rv.sha,))
                        self._move_record_delivery(v.table, rv.sha, "refused", "duplicate", file_uri)    # every file that carried it follows the verdict
                elif out.status == "conflict":
                    reason = self._refiling_reason(rv, out.conflict_fields, out.kept_sha)
                    carrier = self._ordered_carrier(v, rv.sha, rv.uri)              # the index's smallest uri for the record, as every
                    if standing is None:                                             # ordered-fact refusal names it (live equals rebuild)
                        self._refuse(rv.sha, carrier, rv.record_type, R.NATURAL_KEYS[v.table][0], reason, now)
                        self._move_record_delivery(v.table, rv.sha, "accepted", "refused", file_uri)
                    elif (str(standing[0]), str(standing[1])) != (reason, carrier):
                        self.conn.execute("UPDATE record_refusal SET reason = ?, record_uri = ? WHERE record_sha256 = ?", (reason, carrier, rv.sha))
                else:
                    raise RuntimeError(f"ordered ledger fact {v.table} {key}: the indexed record {rv.sha[:12]} ({rv.uri}) sorts before the kept "
                                       f"record {str(out.kept_sha or '')[:12]} ({out.status}); the kept record must be the smallest key applied")

    def _ordered_carrier(self, v: VAL.Validated, sha: str, uri: str) -> str:
        """The uri an ordered-fact refusal row names for ``sha``, a record of ``v``'s natural key: the index's CANONICAL carrier (the
        smallest logical uri over every arrival of the record, ``reducer.Reducer.ordered_carrier``), else ``uri``. The same rule
        the dependency refusal and the record index already keep, applied to the conflict, the displaced and the re-judged refusals alike:
        `record_refusal` is `INSERT OR REPLACE` by sha, so a conflicting filing copied to a smaller
        uri and ingested later left the live row at the arrival's uri while a rebuild, meeting the smaller copy first, wrote the smaller
        one -- the same records, two refusal tables."""
        assert v.table is not None
        key = {c: v.payload[c] for c in R.NATURAL_KEYS[v.table]}
        indexed = self.reducer.ordered_carrier(v.table, key, sha)
        return min(indexed, uri) if indexed is not None else uri

    # -- a filing's children (records.LEDGER_FILING_CHILDREN; the fifth closing pass, 2026-10-09) --------------------------------------
    @staticmethod
    def _parent_key_of(v: VAL.Validated) -> Tuple[str, str, str, Dict[str, object]]:
        """For a child record: ``(the parent table, the binding column naming the parent record's sha, the FK column to the parent, the
        parent's natural key)``."""
        assert v.table is not None
        ptable, col = R.LEDGER_FILING_CHILDREN[v.table]
        fk_col = next(c for c, pt, _pc in S.foreign_keys(v.table) if pt == ptable)
        return ptable, col, fk_col, {R.NATURAL_KEYS[ptable][0]: v.payload[fk_col]}

    def _kept_sha(self, ptable: str, key: Mapping[str, object]) -> Optional[str]:
        """The record sha the parent key's row carries (its kept filing), or None when the key has no row."""
        pk = R.NATURAL_KEYS[ptable]
        where = " AND ".join(f"{c} = ?" for c in pk)
        row = self.conn.execute(f"SELECT {R.LEDGER_ORDERED_TABLES[ptable][1]} FROM {ptable} WHERE {where}", tuple(key[c] for c in pk)).fetchone()
        return str(row[0]) if row is not None else None

    def _authorized_filings(self, ptable: str, key: Mapping[str, object]) -> Set[str]:
        """The filings of a parent key whose children the fold accepts: the record the row carries (the kept filing) and every other
        indexed record of the key with no standing refusal (its duplicates: the same substance under another sha). A conflicting or
        displaced filing stands refused and authorizes nothing."""
        kept = self._kept_sha(ptable, key)
        shas = {sha for sha, _uri, _rt, _utc in self.reducer.ordered_records_of(ptable, key)}
        refused: Set[str] = set()
        if shas:
            marks = ", ".join("?" for _ in shas)
            refused = {str(r[0]) for r in self.conn.execute(f"SELECT record_sha256 FROM record_refusal WHERE record_sha256 IN ({marks})",
                                                             tuple(sorted(shas))).fetchall()}
        return (shas - refused) | ({kept} if kept is not None else set())

    @staticmethod
    def _filing_child_prefix(ptable: str, pid: str) -> str:
        """The opening of every `not kept` refusal reason of the children of one parent key (the table and the key's value)."""
        return f"filing of {ptable} {pid} not kept:"

    @classmethod
    def _filing_child_reason(cls, v: VAL.Validated, ptable: str, pid: str, filing_sha: str, kept_sha: Optional[str]) -> str:
        return (f"{cls._filing_child_prefix(ptable, pid)} this {v.record_type} names the filing {filing_sha[:12]}, which stands refused (a "
                f"conflicting or displaced re-filing of the {ptable}); the inputs and files of an {ptable} are the kept filing's "
                f"({str(kept_sha or '')[:12]}) and its duplicates' alone, on live ingest and on rebuild alike, so a refused filing's report or "
                f"page is never attached to the kept one")

    def _filing_verdict(self, v: VAL.Validated) -> Optional[Tuple[str, str]]:
        """The judgment of a child record's FILING (records.LEDGER_FILING_CHILDREN): None when the header it names authorizes it -- the
        parent key's kept filing or a duplicate of it -- else ``(field, reason)`` of the refusal: a `dependency:` refusal on the header when
        the record index does not hold it (it has not folded; the retry pass re-selects it every cycle, :data:`FILING_PARENT`), a refusal
        by name when the header is indexed under ANOTHER parent key (a hand-written child of the wrong analysis: nothing to wait for), and
        a refusal by name when the header stands refused (a conflicting or displaced re-filing: its inputs and files go down with it).
        The FK parent itself is checked before this (``_missing_parents``), so the parent row exists here."""
        ptable, col, fk_col, key = self._parent_key_of(v)
        filing_sha = str(v.payload.get(col) or "")
        if filing_sha in self._authorized_filings(ptable, key):
            return None
        pk0 = R.NATURAL_KEYS[ptable][0]
        pid = str(key[pk0])
        idx = self.conn.execute(f"SELECT natural_key_json FROM {R.LEDGER_ORDERED_RECORD_INDEX} WHERE table_name = ? AND record_sha256 = ?",
                                (ptable, filing_sha)).fetchone()
        if idx is None:
            return col, filing_dependency_refusal_reason(filing_sha, ptable, pid, v.record_type)
        other = str(json.loads(str(idx[0])).get(pk0))
        if other != pid:
            return col, (f"{self._filing_child_prefix(ptable, pid)} this {v.record_type} names the filing {filing_sha[:12]}, a filing of "
                         f"{ptable} {other}, not of {pid}; correct {col} or {fk_col}")
        return col, self._filing_child_reason(v, ptable, pid, filing_sha, self._kept_sha(ptable, key))

    def _rejudge_filing_children(self, v: VAL.Validated, now: str) -> None:
        """The kept filing of ``v``'s key MOVED (an earlier-sorting header arrived late; ``_rejudge_refilings`` has just judged the other
        headers again): every CHILD of the key (records.LEDGER_FILING_CHILDREN) was accepted or refused against the filings authorized
        BEFORE the move, so the key's children are derived again from the records the fold knows -- every indexed child record of the key
        and every child refused as a non-kept filing's -- each re-read from its immutable file, never from the database, in the log's
        order: the key's child rows and their index rows are dropped and each record applied afresh. One whose filing is authorized now
        folds (inserted, or a duplicate / conflict among the authorized records of its key, judged as the apply always judges them; a
        standing refusal of it withdrawn or re-worded), one whose filing stands refused is refused by name (its row, if it had one, gone
        with it; the refusal names the record's smallest known carrier). Without this a displaced filing's report stayed linked to the
        analysis its earlier-stamped rival now heads while a rebuild -- meeting the earlier filing first -- never linked it: the same
        records, two link tables. A child refused for an absent FK parent or for a filing not yet folded is not touched here: the retry
        pass re-selects it. The kept record of a child key is the smallest authorized record applied, so a `replaced` outcome here is a
        broken invariant and stops the cycle rather than fold on."""
        assert v.table is not None
        children = [t for t, (p, _c) in R.LEDGER_FILING_CHILDREN.items() if p == v.table]
        if not children:
            return
        pk = R.NATURAL_KEYS[v.table]
        key = {c: v.payload[c] for c in pk}
        pid = str(key[pk[0]])
        prefix = self._filing_child_prefix(v.table, pid)
        authorized = self._authorized_filings(v.table, key)
        idx = R.LEDGER_ORDERED_RECORD_INDEX
        cands: Dict[str, str] = {}                      # child record sha -> its smallest known logical uri
        for t in children:
            fk_col = next(c for c, pt, _pc in S.foreign_keys(t) if pt == v.table)
            rt = next(k for k, tb in R.RECORD_TYPE_TABLE.items() if tb == t)
            for sha, uri in self.conn.execute(f"SELECT record_sha256, record_uri FROM {idx} WHERE table_name = ? AND json_extract(natural_key_json, ?) = ?",
                                              (t, f"$.{fk_col}", pid)).fetchall():
                cands[str(sha)] = min(cands.get(str(sha), str(uri)), str(uri))
            for sha, uri in self.conn.execute("SELECT record_sha256, record_uri FROM record_refusal WHERE record_type = ? AND substr(reason, 1, ?) = ?",
                                              (rt, len(prefix), prefix)).fetchall():
                cands[str(sha)] = min(cands.get(str(sha), str(uri)), str(uri))
            # the slate: the key's child rows and their index rows go, derived again below from the records themselves
            self.conn.execute(f"DELETE FROM {t} WHERE {fk_col} = ?", (pid,))
            self.conn.execute(f"DELETE FROM {idx} WHERE table_name = ? AND json_extract(natural_key_json, ?) = ?", (t, f"$.{fk_col}", pid))
        by_file: Dict[str, Set[str]] = {}
        for sha, uri in cands.items():
            if "#" in uri:
                by_file.setdefault(uri.rpartition("#")[0], set()).add(sha)
        recs: List[VAL.Validated] = []
        for file_uri in sorted(by_file):
            recs += self._validated_by_sha(file_uri, by_file[file_uri])
        recs.sort(key=lambda r: (r.written_utc, r.sha))
        kept_sha = self._kept_sha(v.table, key)
        for rv in recs:
            assert rv.table is not None
            self.report.filing_children_rejudged += 1
            _pt, col, _fk, _k = self._parent_key_of(rv)
            filing_sha = str(rv.payload.get(col) or "")
            standing = self.conn.execute("SELECT reason, record_uri FROM record_refusal WHERE record_sha256 = ?", (rv.sha,)).fetchone()
            rv_file = rv.uri.rpartition("#")[0]                 # the file the record is re-read from: its delivery audit follows the verdict
            if filing_sha in authorized:
                out = self.reducer.apply_ledger_fact(rv)
                if out.status in ("inserted", "duplicate"):
                    if standing is not None and (str(standing[0]).startswith(prefix) or str(standing[0]).startswith(self._refiling_prefix(rv))):
                        self.conn.execute("DELETE FROM record_refusal WHERE record_sha256 = ?", (rv.sha,))
                        self._move_record_delivery(rv.table, rv.sha, "refused", "ok" if out.status == "inserted" else "duplicate", rv_file)
                elif out.status == "conflict":
                    reason = self._refiling_reason(rv, out.conflict_fields, out.kept_sha)
                    carrier = self._ordered_carrier(rv, rv.sha, rv.uri)
                    if standing is None:
                        self._refuse(rv.sha, carrier, rv.record_type, R.NATURAL_KEYS[rv.table][0], reason, now)
                        self._move_record_delivery(rv.table, rv.sha, "accepted", "refused", rv_file)
                    elif (str(standing[0]), str(standing[1])) != (reason, carrier):
                        self.conn.execute("UPDATE record_refusal SET reason = ?, record_uri = ? WHERE record_sha256 = ?", (reason, carrier, rv.sha))
                else:
                    raise RuntimeError(f"filing children of {v.table} {key}: the child {rv.sha[:12]} ({rv.uri}) sorts before the kept record "
                                       f"{str(out.kept_sha or '')[:12]} of its key ({out.status}); the children are applied in the log's order")
            else:
                reason = self._filing_child_reason(rv, v.table, pid, filing_sha, kept_sha)
                carrier = min(rv.uri, cands.get(rv.sha, rv.uri), str(standing[1]) if standing is not None else rv.uri)
                if standing is None:
                    self._refuse(rv.sha, carrier, rv.record_type, col, reason, now)
                    self._move_record_delivery(rv.table, rv.sha, "accepted", "refused", rv_file)
                elif (str(standing[0]), str(standing[1])) != (reason, carrier):
                    self.conn.execute("UPDATE record_refusal SET reason = ?, record_uri = ? WHERE record_sha256 = ?", (reason, carrier, rv.sha))

    def _validated_by_sha(self, file_uri: str, want: Set[str]) -> List[VAL.Validated]:
        """The validated records of one ranked file whose shas are wanted, re-read from the file (never from the database)."""
        path = resolve_uri(self.root, file_uri)
        if path is None:
            return []
        out: List[VAL.Validated] = []
        for part in stream_parts(self.root, path, None, ITEM_ROWS_PER_TRANSACTION):
            if part.error or part.retry:
                continue
            out += [pr.result for pr in part.records if pr.sha in want and isinstance(pr.result, VAL.Validated)]
        return out

    def _move_delivery(self, file_uri: str, frm: str, to: str) -> None:
        """The delivery audit of ``file_uri`` (`record_delivery`: the counts and the status the status page's recent-ingests list reads)
        follows a record of the file judged AGAIN after the file's fold -- a filing displaced by an earlier-stamped one, a re-filing or a
        child judged again because the kept filing of its key moved, a dependency refusal accepted on retry: ONE record moves from the
        bucket ``frm`` to the bucket ``to`` (`ok`, `duplicate`, `refused`) and the status is derived again by the fold's own rule
        (:func:`delivery_status`), so the row reads what the record log says of the file, as the row a rebuild writes for it does. ``frm``
        may be `accepted`, the side of a record the fold counted `ok` or `duplicate`: WHICH of the two is not recorded per record, and the
        split between them is the fold's arrival order -- a rebuild meeting the files in another order splits them otherwise too, the
        reason the audit is outside rebuild equality -- so the move takes `n_ok` while the file has any, else `n_duplicate`; `n_refused`,
        `n_ok + n_duplicate` and the status's reading of them are the record log's and agree with a rebuild. ONE file's row: the files a
        record was carried by are :meth:`_move_record_delivery`'s to find (the carrier audit, records.LEDGER_ORDERED_CARRIERS), and every
        judgment of an ordered-fact record goes through it. A file with no delivery row yet (never ranked, or a part-written file whose
        cursor stands) is left alone. The eighth closing pass, 2026-10-09: a filing displaced by an earlier-stamped rival had its header, its
        input and its link refused by name while their three files kept `ok` with `n_ok` 1 and `n_refused` 0 -- a rebuild of the same
        records wrote `refused` -- so the status page listed the rejected filing as ingested."""
        row = self.conn.execute("SELECT n_records, n_ok, n_refused, n_blocked, n_duplicate FROM record_delivery WHERE file_uri = ?",
                                (file_uri,)).fetchone()
        if row is None:
            return
        n = {"ok": int(row[1]), "refused": int(row[2]), "blocked": int(row[3]), "duplicate": int(row[4])}
        if frm == "accepted":
            frm = "ok" if n["ok"] > 0 else "duplicate"
        if frm == to:
            return
        n[frm] = max(0, n[frm] - 1)
        n[to] += 1
        self.conn.execute("UPDATE record_delivery SET n_ok = ?, n_refused = ?, n_blocked = ?, n_duplicate = ?, status = ? WHERE file_uri = ?",
                          (n["ok"], n["refused"], n["blocked"], n["duplicate"], delivery_status(n, int(row[0])), file_uri))
        self.report.deliveries_rejudged += 1

    def _move_record_delivery(self, table: str, sha: str, frm: str, to: str, reread_file: str) -> None:
        """EVERY file that carried the record ``sha`` of the ordered table ``table`` follows its new verdict. The carrier audit
        (`ordered_fact_carrier`, records.LEDGER_ORDERED_CARRIERS: one row per arrival of the record, with the side of its file's delivery
        audit the arrival is counted on) names them: each row standing on the side ``frm`` -- `accepted` (`n_ok` or `n_duplicate`) or
        `refused` -- moves one record of its file to ``to`` (:meth:`_move_delivery`) and is marked on the other side. ``reread_file`` is the
        file the judgment re-read the record from: an accepted record takes ``to`` there as the apply judged it (`ok` or `duplicate`) and
        `duplicate` in every other file that carried it -- the record applied once, so a second copy of it is a duplicate by definition;
        which copy a rebuild counts `ok` is its arrival order, the split the audit does not compare. A record with no carrier row at all
        (applied before the carriers were kept) moves the re-read file alone, as the eighth closing pass did; one whose carriers all stand on
        the ``to`` side already moves nothing (that judgment was recorded). The ninth closing pass, 2026-10-10: the mover followed
        `ordered_fact_record`'s ONE canonical carrier (the smallest uri), so a filing copied to a second uri and then displaced had
        the copy's file moved to `refused` while its original kept `ok` with `n_ok` 1 and `n_refused` 0 -- a rebuild of the same records
        wrote `refused` for both -- and the status page listed one rejected filing file as ingested."""
        side_frm = "refused" if frm == "refused" else "accepted"
        side_to = "refused" if to == "refused" else "accepted"
        rows = self.conn.execute(f"SELECT record_uri, delivery FROM {R.LEDGER_ORDERED_CARRIERS} WHERE table_name = ? AND record_sha256 = ? "
                                 "ORDER BY record_uri", (table, sha)).fetchall()
        if not rows:
            self._move_delivery(reread_file, frm, to)
            return
        for uri, side in rows:
            if str(side) != side_frm:
                continue
            f = str(uri).rpartition("#")[0]
            self._move_delivery(f, frm, to if side_to == "refused" or f == reread_file else "duplicate")
        if side_frm != side_to:
            self.conn.execute(f"UPDATE {R.LEDGER_ORDERED_CARRIERS} SET delivery = ? WHERE table_name = ? AND record_sha256 = ? AND delivery = ?",
                              (side_to, table, sha, side_frm))

    @staticmethod
    def _refiling_prefix(v: VAL.Validated) -> str:
        """The opening of every `conflicting re-filing` refusal reason of ``v``'s natural key (the table and the key's values)."""
        key = ", ".join(str(v.payload.get(c)) for c in R.NATURAL_KEYS[v.table or ""])
        return f"conflicting re-filing of {v.table} {key}:"

    @classmethod
    def _refiling_reason(cls, v: VAL.Validated, fields: Sequence[str], kept_sha: Optional[str]) -> str:
        return (f"{cls._refiling_prefix(v)} {', '.join(fields)} differ from the kept filing {str(kept_sha or '')[:12]}; the "
                f"filing with the smallest (written_utc, record_sha256) is kept -- the log's order, on live ingest and on rebuild alike; a "
                f"changed analysis (its header, an input's role, a file's role or page) is a new analysis_id (its name, code sha or inputs), "
                f"never a re-filing of the old one")

    def _refuse(self, sha: str, uri: str, record_type: Optional[str], field_name: Optional[str], reason: str, now: str) -> None:
        """Record ONE refusal judgment: the current judgment of the record (``record_refusal``, replaced by a later judgment of the same
        sha under another stamp) stamped with the judging code (``judged_with`` = :func:`validator_stamp`), the state sidecar line,
        and the cycle's report. The durable fact of the judgment is derived from this row by the cycle's refusal-fact passes
        (:func:`derive_refusal_fact_records`): at the START of the next cycle at the latest, before anything can replace the row."""
        self.conn.execute(
            "INSERT OR REPLACE INTO record_refusal(record_sha256, record_uri, record_type, field, reason, refused_utc, judged_with) "
            "VALUES (?,?,?,?,?,?,?)", (sha, uri, record_type, field_name, reason, now, validator_stamp()))
        self._unblock(sha)
        self.report.refused += 1
        self.report.refusals.append(f"{uri}: {reason}")
        try:
            with open(os.path.join(state_dir(self.root), "refused.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(canonical_json({"uri": uri, "sha": sha, "record_type": record_type, "field": field_name,
                                         "reason": reason, "refused_utc": now}) + "\n")
        except OSError:
            pass

    def _block(self, sha: str, uri: str, record_type: str, parent_table: str, parent_id: str, now: str,
               extra: Optional[Mapping[str, object]] = None) -> None:
        row = self.conn.execute("SELECT attempts, first_seen_utc, last_checked_utc, record_uri, extra_json FROM blocked_records "
                                "WHERE record_sha256 = ?", (sha,)).fetchone()
        if row:
            # a re-check within the SAME cycle (same `now`) is not a new attempt
            attempts = int(row[0]) + (0 if row[2] == now else 1)
            first = row[1]
            stale = 1 if (R.utc_to_epoch_ms(now) - R.utc_to_epoch_ms(first)) / 1000.0 > BLOCKED_STALE_S else 0
            # the SAME record blocked under another file: the row keeps the canonical (smallest logical URI) carrier and
            # remembers the other files, which the release indexes (live equals replay)
            keep, others = str(row[3]), set(json.loads(row[4] or "{}").get("dup_uris") or [])
            if uri != keep:
                a, b = uri.rpartition("#")[0], keep.rpartition("#")[0]
                if a < b:
                    others.add(b)
                    keep = uri
                else:
                    others.add(a)
            ext = dict(extra or {})
            if others:
                ext["dup_uris"] = sorted(others)
            self.conn.execute(
                "UPDATE blocked_records SET record_uri = ?, missing_parent_table = ?, missing_parent_id = ?, last_checked_utc = ?, "
                "attempts = ?, stale = ?, extra_json = ? WHERE record_sha256 = ?",
                (keep, parent_table, parent_id, now, attempts, stale, canonical_json(ext), sha))
        else:
            self.conn.execute(
                "INSERT INTO blocked_records(record_sha256, record_uri, record_type, missing_parent_table, missing_parent_id, "
                "first_seen_utc, last_checked_utc, attempts, stale, extra_json) VALUES (?,?,?,?,?,?,?,1,0,?)",
                (sha, uri, record_type, parent_table, parent_id, now, now, canonical_json(dict(extra or {}))))
            self.report.blocked += 1
        if self._blocked is not None:
            self._blocked.add(sha)

    def _unblock(self, sha: str, run_ids: Tuple[str, ...] = (), file_uri: Optional[str] = None) -> None:
        """Drop a record from the blocked queue — a DELETE only when the transaction-start set says it is there (F5).
        The same record blocked under ANOTHER file (a duplicate delivery) is applied by this one: that other file still
        mentions the run, so it is indexed here — live and replay then list the same files."""
        if self._blocked is None or sha in self._blocked:
            if run_ids:
                row = self.conn.execute("SELECT record_uri, extra_json FROM blocked_records WHERE record_sha256 = ?", (sha,)).fetchone()
                if row is not None:
                    others = set(json.loads(row[1] or "{}").get("dup_uris") or [])
                    if "#" in str(row[0]):
                        others.add(str(row[0]).rpartition("#")[0])
                    for other in sorted(others):
                        if other != file_uri:
                            self._index_run(run_ids, other)
            self.conn.execute("DELETE FROM blocked_records WHERE record_sha256 = ?", (sha,))
            if self._blocked is not None:
                self._blocked.discard(sha)

    def _blocked_carrier(self, sha: str) -> Optional[str]:
        """The file under which this record is waiting in the blocked queue (None when it is not)."""
        if self._blocked is not None and sha not in self._blocked:
            return None
        row = self.conn.execute("SELECT record_uri FROM blocked_records WHERE record_sha256 = ?", (sha,)).fetchone()
        if row is None or "#" not in str(row[0]):
            return None
        return str(row[0]).rpartition("#")[0]

    def _carrier(self, sha: str, file_uri: str, seq: int, record_file_id: int) -> Tuple[str, int, int]:
        """The provenance an applied record carries: its own file, unless the SAME record waits in the blocked queue under a
        smaller logical URI — then that file, whatever arrived first, so live equals replay."""
        other = self._blocked_carrier(sha)
        if other is None or not (other < file_uri):
            return file_uri, seq, record_file_id
        other_seq = self.ranker.seq_of(other)
        if other_seq is None:
            return file_uri, seq, record_file_id
        return other, other_seq, R.record_file_id_for_uri(other)

    def _index_run(self, run_ids: Tuple[str, ...], file_uri: str) -> None:
        """One record_run_index row per (run, file) — deduplicated in memory per transaction (F5)."""
        for rid in run_ids:
            key = (rid, file_uri)
            if key not in self._indexed:
                self._indexed.add(key)
                self.conn.execute("INSERT OR IGNORE INTO record_run_index(run_id, record_uri) VALUES (?, ?)", (rid, file_uri))

    # -- one record ---------------------------------------------------------------------------
    def apply_validated(self, v: VAL.Validated, seq: int, file_uri: str, record_file_id: int,
                        item_batches: Dict[str, List[Tuple]], item_revisions: Set[str], ex: _Existence, now: str) -> str:
        """Returns ok | blocked | refused | duplicate. ``seq`` is the FILE's ingest_seq. A record of an ordered ledger-fact table
        (records.LEDGER_ORDERED_TABLES) is also entered in the carrier audit (records.LEDGER_ORDERED_CARRIERS) under THIS arrival's uri,
        on the side of its file's delivery audit the verdict counts it on (`refused`, else `accepted`: `ok` or `duplicate`), so a later
        judgment of the record moves every file that carried it (:meth:`_move_record_delivery`); a uri already entered keeps its side (the
        retry pass re-applies a refused record from the same file, and its own move is what flips the side). The ninth closing pass,
        2026-10-10."""
        status = self._apply_validated(v, seq, file_uri, record_file_id, item_batches, item_revisions, ex, now)
        if v.table is not None and v.table in R.LEDGER_ORDERED_TABLES and status != "blocked":
            self.conn.execute(f"INSERT OR IGNORE INTO {R.LEDGER_ORDERED_CARRIERS} (table_name, record_sha256, record_uri, delivery) VALUES (?, ?, ?, ?)",
                              (v.table, v.sha, v.uri, "refused" if status == "refused" else "accepted"))
        return status

    def _apply_validated(self, v: VAL.Validated, seq: int, file_uri: str, record_file_id: int,
                         item_batches: Dict[str, List[Tuple]], item_revisions: Set[str], ex: _Existence, now: str) -> str:
        """:meth:`apply_validated`'s judgment of one record, without the carrier audit's row."""
        missing = self._missing_parents(v, ex)
        if missing:
            ptable, pid = missing[0]
            if v.table is not None and v.table in S.LEDGER_TABLES:
                # a ledger-fact kind (vocab.UNLEDGERED_FACT_TYPES: the analysis kinds) can never WAIT in `blocked_records` -- its
                # record_type CHECK is the ledgered list, and widening it rebuilds both ledgers -- so a parent it names that the registry
                # does not hold is a REFUSAL by name (the FK column, the parent, the id), never a queued wait and never an IntegrityError
                # at the CHECK. `registry analysis file` read-checks every input before it writes, so only a hand-written record gets here;
                # the refusal is a fact, listed on the status page, re-judged under a changed stamp or `ingest --retry-refused`
                col = next((c for c, pt, _pc in S.foreign_keys(v.table) if pt == ptable and str(v.payload.get(c)) == pid), None)
                # the same record under two logical uris (a copied filing): the refusal row -- one per sha -- keeps the SMALLEST carrier
                #, so the retry re-reads the same file whatever the arrival order, live and on rebuild alike (a last-arrival uri
                # left the other carrier's record unretried on one path and the row's carrier differing between the two)
                standing = self.conn.execute("SELECT record_uri FROM record_refusal WHERE record_sha256 = ?", (v.sha,)).fetchone()
                uri = min(v.uri, str(standing[0])) if standing is not None else v.uri
                self._refuse(v.sha, uri, v.record_type, col, dependency_refusal_reason(ptable, pid, v.record_type), now)
                return "refused"
            self._block(v.sha, v.uri, v.record_type, ptable, pid, now)
            return "blocked"
        if v.table is not None and v.table in R.LEDGER_FILING_CHILDREN:
            # a child of a filing (records.LEDGER_FILING_CHILDREN; the fifth closing pass, 2026-10-09): folded only while the header it names
            # is the parent key's kept filing or a duplicate of it; refused by name when that header stands refused (a conflicting re-filing's
            # report or page is never attached to the kept analysis); a dependency refusal the retry pass re-selects every cycle while the
            # header has not folded. The refusal row keeps the record's smallest carrier, as the FK refusal above does
            verdict = self._filing_verdict(v)
            if verdict is not None:
                field_name, reason = verdict
                standing = self.conn.execute("SELECT record_uri FROM record_refusal WHERE record_sha256 = ?", (v.sha,)).fetchone()
                self._refuse(v.sha, min(v.uri, str(standing[0])) if standing is not None else v.uri, v.record_type, field_name, reason, now)
                return "refused"
        applied = False
        carrier_uri, carrier_seq, carrier_rfid = self._carrier(v.sha, file_uri, seq, record_file_id)
        if v.record_type in R.ITEM_SCALE_RECORD_TYPES:
            # an item row touches the database only at flush: no savepoint, no per-row statement (F5)
            table = R.RECORD_TYPE_TABLE[v.record_type]
            item_batches.setdefault(table, []).append(self.reducer.item_row(table, v, carrier_rfid, carrier_seq))
            item_revisions.add(str(v.payload.get("source_revision_id") or ""))
            self._index_run(_run_ids_of(v.payload), file_uri)
            applied = True
        elif v.table is not None and v.table in S.LEDGER_TABLES:
            # a ledger-scale fact (the `record_refusal` kind -> `record_refusal_fact`) folds by its own apply path: INSERT OR IGNORE,
            # no row_history event, no run index, no refold -- the row is the fact and a second record of it is a duplicate; counted apart
            # from `ok` (refusal_fact_records), as bookkeeping the fold itself caused
            try:
                with DB.savepoint(self.conn, "rec"):
                    outcome = self.reducer.apply_ledger_fact(v)
            except (sqlite3.IntegrityError, ValueError, KeyError) as exc:
                self._refuse(v.sha, v.uri, v.record_type, getattr(exc, "field", None), f"{type(exc).__name__}: {exc}", now)
                return "refused"
            self._unblock(v.sha, (), file_uri)
            if outcome.status == "conflict":
                # an ordered ledger fact (records.LEDGER_ORDERED_TABLES) sorting AFTER the kept row with another payload: refused by name,
                # naming the differing columns and the kept filing -- never a silent drop. The refusal row
                # names the record's CANONICAL carrier (the index's smallest uri), the same sha arriving again under another uri included
                self._refuse(v.sha, self._ordered_carrier(v, v.sha, v.uri), v.record_type, R.NATURAL_KEYS[v.table][0],
                             self._refiling_reason(v, outcome.conflict_fields, outcome.kept_sha), now)
                return "refused"
            if outcome.status == "replaced" and outcome.displaced is not None:
                # the incoming record sorts BEFORE the standing row (a late-delivered earlier filing): it took the header, and the filing it
                # displaced is refused by name -- the same judgment a rebuild makes of that filing when it meets the earlier one first
                d_sha, d_uri = outcome.displaced
                d_carrier = self._ordered_carrier(v, d_sha, d_uri)
                self._refuse(d_sha, d_carrier, v.record_type, R.NATURAL_KEYS[v.table][0],
                             self._refiling_reason(v, outcome.conflict_fields, outcome.kept_sha), now)
                # every file that carried it says so too (the eighth closing pass; every carrier, not the index's one, the ninth)
                self._move_record_delivery(v.table, d_sha, "accepted", "refused", d_carrier.rpartition("#")[0])
            if outcome.kept_changed and outcome.status != "inserted" and v.table in R.LEDGER_ORDERED_TABLES:
                # the kept filing of this key moved to an earlier one: every other record of the key was judged against the previous kept
                # filing, so every one the index holds is judged again against this one -- a record identical to it is accepted (its refusal
                # row withdrawn), one that differs is refused by name (a former duplicate of the displaced filing included), a still-conflicting
                # one keeps its verdict with the reason naming the new kept filing. Without this the live refusals named whichever filing was
                # kept when they were made, and a rebuild -- which meets the earliest filing first -- named another: the same records, two
                # refusal tables
                self._rejudge_refilings(v, now)
                # ... and every CHILD of the key (records.LEDGER_FILING_CHILDREN) was accepted or refused against the filings authorized
                # before the move: derived again against the ones authorized now -- the displaced filing's inputs and links withdrawn and
                # refused by name, the new kept filing's accepted (the fifth closing pass, 2026-10-09)
                self._rejudge_filing_children(v, now)
            if not outcome.applied:
                self.report.duplicates += 1
                return "duplicate"
            if v.record_type in V.ANALYSIS_RECORD_TYPES:
                self.report.analysis_records += 1            # a filed analysis's rows: a producer's facts, counted apart from `ok` too
            else:
                self.report.refusal_fact_records += 1
            return "ok"
        else:
            held = self.reducer.lifecycle_regressions_held
            try:
                with DB.savepoint(self.conn, "rec"):
                    for eff in effects_of(v):
                        if self.reducer.apply_run_scale(eff, v, carrier_uri, carrier_seq):
                            applied = True
                        if eff.kind == "full":
                            pk = R.NATURAL_KEYS[eff.table]
                            if len(pk) == 1:
                                ex.note(eff.table, pk[0], eff.key[pk[0]])
                            if eff.table == "source_revision" and int(eff.payload.get("accepted") or 0) == 1:
                                ex.note_accepted(str(eff.key["revision_id"]))
                    self._index_run(_run_ids_of(v.payload), file_uri)
            except LifecycleEvidenceMissing as exc:
                # the refold reached a lifecycle state without an earlier state's utc (reducer.LifecycleEvidenceMissing): the evidence
                # has not arrived in FOLD order, whatever the row shows -- dependency-blocked on it, like _missing_lifecycle_evidence,
                # never refused; the savepoint rolled the event's history row back, and the queue re-attempts it when the evidence lands
                ptable, pid = exc.parent()
                self._block(v.sha, v.uri, v.record_type, ptable, pid, now)
                return "blocked"
            except (sqlite3.IntegrityError, VerifyMismatch, ValueError, KeyError) as exc:
                field_name = getattr(exc, "field", None)
                self._refuse(v.sha, v.uri, v.record_type, field_name, f"{type(exc).__name__}: {exc}", now)
                return "refused"
            self.report.lifecycle_regressions_held += self.reducer.lifecycle_regressions_held - held
        self._unblock(v.sha, _run_ids_of(v.payload), file_uri)
        if not applied:
            self.report.duplicates += 1
            return "duplicate"
        self.report.ok += 1
        return "ok"

    def apply_item(self, pi: ParsedItem, seq: int, file_uri: str, item_batches: Dict[str, List[Tuple]], item_revisions: Set[str],
                   ex: _Existence, now: str) -> str:
        """The worker-built item row: existence check of its parents, then append (the transaction's only work per item)."""
        for ptable, pcol, pid in pi.parents:
            if not ex.exists(ptable, pcol, pid):
                self._block(pi.sha, f"{file_uri}#{pi.line}", pi.table, ptable, pid, now)
                return "blocked"
        rev = revision_of_uri(file_uri)
        if rev and rev != self._activating and not ex.accepted_revision(rev):
            self._block(pi.sha, f"{file_uri}#{pi.line}", pi.table, "source_revision", rev, now)
            return "blocked"
        _, carrier_seq, carrier_rfid = self._carrier(pi.sha, file_uri, seq, R.record_file_id_for_uri(file_uri))
        row = pi.row
        if carrier_seq != seq:
            i = ITEM_PLANS[pi.table].sources.index("record_file_id")
            row = row[:i] + (carrier_rfid,) + row[i + 1:]
        item_batches.setdefault(pi.table, []).append(row + ITEM_PLANS[pi.table].tail(carrier_seq))
        item_revisions.add(pi.source_revision_id)
        self._index_run(pi.run_ids, file_uri)
        self._unblock(pi.sha, pi.run_ids, file_uri)
        self.report.ok += 1
        return "ok"

    def _missing_parents(self, v: VAL.Validated, ex: _Existence) -> List[Tuple[str, str]]:
        out = []
        for ptable, pcol, pid in VAL.parent_references(v):
            if not ex.exists(ptable, pcol, pid):
                out.append((ptable, pid))
        if not out and v.record_type == "item_withdrawal":
            # the rows a producer retracts are parents too: a withdrawal of a row not yet folded waits for it -- blocked,
            # named by the row's canonical key, never refused for it; the rows of this very transaction's item batch are not in
            # the table yet, so a withdrawal delivered beside its rows folds at the cycle's re-evaluation
            for k in v.payload["items"]:
                key = {"run_id": v.payload["run_id"], **k}
                if not ex.item_row_exists(key):
                    out.append(("item_result", canonical_json(key)))
                    break
        if not out and v.record_type == "object" and any(c not in v.payload for c in R.OBJECT_OMITTABLE_WITH_ROW):
            # an object record that omits `status` (a mirror's statement) folds only onto an EXISTING row -- the producer's -- and is
            # dependency-blocked on its own row until one folds (missing parent `object` = its own id), never refused for it; the fold then
            # keeps the folded status (per-column latest statement). The rows of this very chunk are noted as they fold (`ex.note`), so a
            # producer record ordered before the mirror's in the same chunk is seen
            own = str(v.payload.get("object_id"))
            if not ex.exists("object", "object_id", own):
                out.append(("object", own))
        if not out and v.record_type in ("run_state", "run_ingest"):
            out += self._missing_lifecycle_evidence(v)
        if not out and v.record_type != "source_revision":
            # every record of an adapter revision directory waits for the revision's ACCEPTED marker — unless this
            # very transaction is the revision's activation, whose marker comes last
            rev = revision_of_uri(v.uri.rpartition("#")[0])
            if rev and rev != self._activating and not ex.accepted_revision(rev):
                out.append(("source_revision", rev))
        if not out and v.record_type == "source_revision" and int(v.payload.get("accepted") or 0) == 1:
            # the marker's OWN record delivered outside its revision directory (a duplicate copy) must not
            # accept the revision before the directory's activation has staged its constituents — the pending exemption
            # of the tagged rows' foreign keys rests on `accepted = 0` until then — so it waits for that activation; a
            # revision with no directory (plain revision records) folds as before
            rev = str(v.payload.get("revision_id") or "")
            here = revision_of_uri(v.uri.rpartition("#")[0])
            if rev and here != rev and rev != self._activating and not ex.accepted_revision(rev):
                if os.path.isdir(os.path.join(records_dir(self.root), "adapters", str(v.payload.get("adapter") or ""), rev)):
                    out.append(("source_revision", rev))
        return out

    def _missing_lifecycle_evidence(self, v: VAL.Validated) -> List[Tuple[str, str]]:
        """A ``run_state`` (or the harvester's ``run_ingest``, which reaches ``ingested``) whose EARLIER evidence has not
        arrived (``generated`` before ``running``; on replay, ``run_ingest`` in a chunk ahead of the blocked ``judged``)
        is dependency-blocked on that evidence — the same requirement the DDL trigger ``trg_run_state_utc`` enforces —
        and folds when it lands, so live equals replay whatever the delivery order. Named as the missing
        ``run_state_event`` ``<run_id>:<state>``. This reads the ROW; the same requirement in FOLD order -- the evidence must also
        precede the event in the key's history, or the refold holds it as a regression and the upsert's INSERT half carries no
        ``judged_utc`` -- is the reducer's (``reducer.LifecycleEvidenceMissing``, raised by the refold and blocked by
        :meth:`apply_validated` on the same pseudo-parent)."""
        state = "ingested" if v.record_type == "run_ingest" else str(v.payload["state"])
        own = {V.LIFECYCLE_UTC_COLUMN.get(state)}
        if state in V.LIFECYCLE_TERMINAL_FINISHED:
            own.add("finished_utc")
        if v.record_type == "run_ingest" and v.payload.get("judged_utc"):
            own.add("judged_utc")       # a rule-scored run's stamp carries its own scoring time (records.EFFECTS)
        needed = [c for c in V.LIFECYCLE_REQUIRED_UTC.get(state, ()) if c not in own]
        if not needed:
            return []
        row = self.conn.execute(f"SELECT {', '.join(needed)} FROM run WHERE run_id = ?", (v.payload["run_id"],)).fetchone()
        if row is None:
            return [("run", str(v.payload["run_id"]))]
        by_col = {col: st for st, col in V.LIFECYCLE_UTC_COLUMN.items()}
        return [("run_state_event", f"{v.payload['run_id']}:{by_col.get(c, c)}") for c in needed if row[c] is None]

    def flush_items(self, item_batches: Dict[str, List[Tuple]], item_revisions: Set[str]) -> None:
        """Write every table's batch in PRIMARY KEY order (the b-tree is walked once, not seeked per row; F2). After the
        ``item_result`` batch the producer-withdrawal marks of the runs it touched are re-derived: a flushed row carries
        NULL marks, and a row whose order key is below a standing ``item_withdrawal``'s takes the mark back
        (``reducer.remark_item_withdrawals``) -- the marks stay a pure function of the records whatever the delivery order."""
        for table, rows in item_batches.items():
            if not rows:
                continue
            rows.sort(key=operator.itemgetter(*ITEM_PLANS[table].pk_positions))
            sql = self.reducer.item_upsert_sql(table)
            try:
                before = self.conn.total_changes
                DB.executemany_batched(self.conn, sql, rows)
                changed = self.conn.total_changes - before
                dup = max(0, len(rows) - changed)
                self.report.duplicates += dup
                self.report.ok -= dup
            except sqlite3.IntegrityError:
                # isolate the bad rows: one savepoint per row for this batch only
                cols = self.reducer._cols[table]
                for row in rows:
                    try:
                        with DB.savepoint(self.conn, "itm"):
                            self.conn.execute(sql, row)
                    except sqlite3.IntegrityError as exc:
                        sha = row[cols.index("record_sha256")].hex()
                        uri = self.ranker.uri_of(row[cols.index("record_file_id")]) or "?"
                        self.report.ok -= 1
                        self._refuse(sha, uri, table, None, f"IntegrityError: {exc}", R.utc_now())
            if table == "item_result" and ITEM_PLANS[table].trailing:
                run_pos = ITEM_PLANS[table].columns.index("run_id")
                remark_item_withdrawals(self.conn, self._withdrawn_runs({str(r[run_pos]) for r in rows}))
        item_batches.clear()
        item_revisions.clear()

    def _withdrawn_runs(self, run_ids: Set[str]) -> List[str]:
        """The runs among ``run_ids`` that carry an ``item_withdrawal`` row (one indexed probe per chunk of ids)."""
        ids = sorted(run_ids)
        out: List[str] = []
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            out += [str(r[0]) for r in self.conn.execute(
                f"SELECT DISTINCT run_id FROM item_withdrawal WHERE run_id IN ({', '.join('?' for _ in chunk)})", chunk).fetchall()]
        return out

    # -- files ------------------------------------------------------------------------------
    def ingest_parsed(self, parsed: Iterable[ParsedFile], now: Optional[str] = None, stop=None) -> Tuple[Set[str], bool]:
        """Apply parsed files in transactions of ≤ FILES_PER_TRANSACTION files and ≤ ITEM_ROWS_PER_TRANSACTION rows. ``stop`` (a
        callable, the cycle's wall-clock budget) is asked before every chunk; when it answers True the chunk in hand is NOT taken,
        the parse is closed (its worker pool with it) and the files not yet folded wait for the next cycle -- every chunk committed
        before stays committed. Returns ``(the uris of the files taken, whether the budget stopped the pass)``."""
        now = now or R.utc_now()
        taken: Set[str] = set()
        chunks = chunk_parsed(parsed, FILES_PER_TRANSACTION, ITEM_ROWS_PER_TRANSACTION)   # read at call time: overridable
        try:
            for chunk in chunks:
                if stop is not None and stop():
                    return taken, True
                self.ingest_chunk(chunk, now)
                taken.update(pf.uri for pf in chunk)
        finally:
            chunks.close()
            close = getattr(parsed, "close", None)
            if close is not None:
                close()
        return taken, False

    def ingest_chunk(self, chunk: Sequence[ParsedFile], now: str) -> None:
        """ONE transaction: rank the files, fold the run-scale records in order key, existence-check and append the
        item rows, flush in key order, re-rank if a file arrived late, write the delivery audit."""
        self._timed_transaction(lambda: self._ingest_chunk(chunk, now), len(chunk), sum(pf.n_records for pf in chunk), now)

    def apply_dimension_bundle(self, dim_dir: str, files: Sequence[Tuple[str, Optional[str]]], now: str) -> None:
        """Apply every new or changed file of one dimension seed revision directory (``records/dimensions/<rev>/``) in
        ONE transaction, whatever their count or row count: parsed in-process first (the nine files hold a few hundred
        records), then the chunk fold under one ``BEGIN IMMEDIATE``. A reader therefore sees the dimension tables
        entirely before or entirely after the revision (WAL snapshot isolation), never the ``model`` / ``eval`` /
        ``item_set`` rows of the new revision beside the ``instrument`` / ``decode_policy`` / ``object`` rows of the old
        one — which is what the ≤ 50-file chunker produced when the nine files straddled a chunk boundary among the
        cycle's other pending files. The fold, the ledger ranks, the deliveries and the blocked queue are the chunk
        path's own (:meth:`_ingest_chunk`); only the transaction boundary differs. The record files themselves are not
        touched. ``DIMENSION_APPLY_HOOK`` (tests) runs inside the transaction, after the fold and before the commit."""
        parsed = list(iter_parsed(self.root, list(files), 0))
        self.report.dimension_transactions += 1
        t0 = time.monotonic()

        def body() -> None:
            self._ingest_chunk(parsed, now)
            hook = DIMENSION_APPLY_HOOK
            if hook is not None:
                hook(dim_dir)
        self._dimension_bundle = True       # the one changed-file case kept: `dimensions load --force` appends to the revision's files
        try:
            self._timed_transaction(body, len(parsed), sum(pf.n_records for pf in parsed), now, kind="dimensions")
        finally:
            self._dimension_bundle = False
            self.report.dimension_write_s += time.monotonic() - t0

    def _timed_transaction(self, body, files: int, rows, now: str, kind: str = "transaction") -> None:
        """Every write transaction of the cycle — a chunk of plain files, a staging window, an activation, the one
        dense re-rank — goes through here, so ``transactions`` / ``write_s`` / ``longest_transaction_s`` and the
        metrics log see all of them. A body that raises rolls back (and still counts as time under the write
        lock). ``rows`` may be a callable, read after the body for a count only the body knows."""
        t0 = time.monotonic()
        try:
            with DB.transaction(self.conn):
                body()
        finally:
            dt = time.monotonic() - t0
            self.report.transactions += 1
            self.report.write_s += dt
            self.report.longest_transaction_s = max(self.report.longest_transaction_s, dt)
            self._metric({"kind": kind, "files": files, "rows": rows() if callable(rows) else rows, "seconds": round(dt, 4), "utc": now})

    def finalize_ranks(self, now: str) -> int:
        """The ONE dense re-rank of a deferred cycle (``Ranker(defer=True)``), inside one timed transaction logged as
        ``kind = "rerank"`` — always exactly one per cycle, so the metrics log shows the pass even when it had no work.
        In arrival order it has none on the live path: every file the cycle folded is head + 1, so it remaps 0
        rows and touches no seq-bearing table. It still exists for the one file that CAN sort below the head — a
        harvester-originated file stamped by another process (``mirrors``, ``snapshot``) before the head moved past its
        stamp, or a re-keyed file — and remaps once per cycle then (per 50-file chunk before; 3.5 GB read, 4.5 GB
        written, 376–759 s per pass on the live database). Returns the ledger rows remapped."""
        n = [0]

        def body() -> None:
            n[0] = self.ranker.finalize(final=True)
        self._timed_transaction(body, 0, lambda: n[0], now, kind="rerank")
        self.report.rerank_rows += n[0]
        return n[0]

    def _ingest_chunk(self, chunk: Sequence[ParsedFile], now: str) -> None:
        ranker = self.ranker
        ranker.reload()
        ex = _Existence(self.conn)
        self._blocked = {r[0] for r in self.conn.execute("SELECT record_sha256 FROM blocked_records").fetchall()}
        self._indexed = set()
        item_batches: Dict[str, List[Tuple]] = {}
        item_revisions: Set[str] = set()
        todo: List[Tuple[Tuple[str, str], ParsedFile, ParsedRecord]] = []
        file_seqs: Dict[str, int] = {}
        counts: Dict[str, Dict[str, int]] = {}
        type_counts: Dict[str, Dict[str, int]] = {}
        try:
            for pf in chunk:
                if pf.retry:
                    self.report.deferred += 1        # changed under the reader: no ledger row, no delivery, not cached
                    continue
                if pf.part_index == 0:
                    self.report.files += 1
                if pf.uri in self._replaced_refused:
                    continue                             # a later part of a file refused at its first part (below)
                if pf.part_index == 0 and pf.file_sha and not pf.error and not self._dimension_bundle:
                    # an ingested record file is immutable: other bytes at a uri the ledger has ranked are refused by name,
                    # never folded as a changed file. Checked at the file's FIRST part only, against the row a PREVIOUS fold left:
                    # the parts of one parse (provisional sha, then the verified one) and a cursor resume are the same file.
                    # Exempt: a dimension seed revision directory (apply_dimension_bundle), whose `--force` re-publication appends to
                    # the revision's files by design and is re-applied whole in one transaction
                    ingested = self._ingested_sha(pf.uri)
                    if ingested is not None and ingested != pf.file_sha:
                        self._refuse_replaced(pf, ingested, now)
                        continue
                if pf.error:
                    self._deliver(pf, now, "refused", pf.error, {})
                    self.conn.execute("DELETE FROM record_cursor WHERE file_uri = ?", (pf.uri,))
                    self.report.refused += 1
                    self.report.refusals.append(f"{pf.uri}: {pf.error}")
                    continue
                counts[pf.uri] = {"ok": 0, "refused": 0, "blocked": 0, "duplicate": 0}
                type_counts[pf.uri] = {}
                if pf.stat is not None and pf.is_last_part:
                    self.learned[pf.path] = {"size": pf.stat[0], "mtime_ns": pf.stat[1], "ino": pf.stat[2], "sha": pf.file_sha}
                first = pf.first_key
                acc = self._partial.get(pf.uri)
                if acc is None and pf.resumed_from:
                    acc = self._load_cursor(pf.uri, pf.resumed_from)     # the windows an interrupted run committed
                    if acc is not None:
                        self._partial[pf.uri] = acc
                seen_so_far = (acc[3] if acc else 0) + pf.n_records
                if first is not None:
                    n_for_ledger = pf.total_records if pf.last or pf.whole_records is not None else seen_so_far
                    file_seqs[pf.uri] = ranker.assign(pf.uri, pf.file_sha, first[0], n_for_ledger, now)
                elif pf.uri in file_seqs or acc is not None:
                    seq0 = self.ranker.seq_of(pf.uri)
                    if seq0 is not None:
                        file_seqs[pf.uri] = seq0
                self.report.records += pf.n_records
                for pr in pf.records:
                    if isinstance(pr.result, VAL.Malformed):
                        if not self._refused(pr.sha):
                            self._refuse(pr.sha, pr.result.path, pr.result.record_type, pr.result.field, pr.result.reason, now)
                        else:
                            self.report.duplicates += 1
                        continue
                    todo.append((apply_order(pr.result), pf, pr))
            # run-scale records in the fold's order (apply_order = Validated.order_key; the fold is order-independent, this keeps history
            # appends cheap); a revision's marker (source_revision) goes first so the records waiting on it apply in the same transaction
            todo.sort(key=lambda t: t[0])
            for _, pf, pr in todo:
                v: VAL.Validated = pr.result  # type: ignore[assignment]
                status = self.apply_validated(v, file_seqs[pf.uri], pf.uri, R.record_file_id_for_uri(pf.uri), item_batches,
                                              item_revisions, ex, now)
                counts[pf.uri][status] += 1
                type_counts[pf.uri][v.record_type] = type_counts[pf.uri].get(v.record_type, 0) + 1
                if status == "ok" and is_status_page_artifact(v):
                    self.report.status_page_records += 1
                elif status == "ok" and v.record_type == "file_arrival":
                    self.report.arrival_records += 1
            # item rows after the chunk's run-scale rows (their parents may be in this very chunk); order among items
            # is immaterial — the whole-row fold keeps the greatest order key
            for pf in chunk:
                if pf.error or pf.retry or not pf.items or pf.uri in self._replaced_refused:
                    continue
                seq = file_seqs[pf.uri]
                c = counts[pf.uri]
                tc = type_counts[pf.uri]
                for pi in pf.items:
                    c[self.apply_item(pi, seq, pf.uri, item_batches, item_revisions, ex, now)] += 1
                    tc[pi.table] = tc.get(pi.table, 0) + 1
            self.flush_items(item_batches, item_revisions)
            ranker.finalize()
            self.report.reranks += ranker.reranks
            for pf in chunk:
                if pf.error or pf.retry or pf.uri in self._replaced_refused:
                    continue                         # a replaced ingested file was refused whole above: no counts, no delivery row
                c = counts[pf.uri]
                tc = type_counts[pf.uri]
                n_ref = c["refused"] + sum(1 for pr in pf.records if isinstance(pr.result, VAL.Malformed))
                if not pf.is_single:
                    # a file written across transactions: accumulate, deliver once with the last part
                    acc_c, acc_tc, acc_ref, acc_n = self._partial.get(pf.uri, ({"ok": 0, "refused": 0, "blocked": 0, "duplicate": 0}, {}, 0, 0))
                    for k in acc_c:
                        acc_c[k] += c[k]
                    for k, n in tc.items():
                        acc_tc[k] = acc_tc.get(k, 0) + n
                    acc_ref += n_ref
                    acc_n += pf.n_records
                    if not pf.is_last_part:
                        # progress committed WITH this window: a restart resumes after these records, never before
                        self._partial[pf.uri] = (acc_c, acc_tc, acc_ref, acc_n)
                        self.conn.execute(
                            "INSERT OR REPLACE INTO record_cursor(file_uri, file_sha256, records_done, counts_json, updated_utc) "
                            "VALUES (?,?,?,?,?)", (pf.uri, pf.file_sha, acc_n, canonical_json({"counts": acc_c, "types": acc_tc, "n_refused": acc_ref}), now))
                        continue
                    self._partial.pop(pf.uri, None)
                    c, tc, n_ref = acc_c, acc_tc, acc_ref
                    if pf.whole_records is None:
                        pf.whole_records = acc_n
                total = pf.total_records
                status = delivery_status(dict(c, refused=n_ref), total)
                self._deliver(pf, now, status, None, tc, c, n_ref)
                self.conn.execute("DELETE FROM record_cursor WHERE file_uri = ?", (pf.uri,))   # complete: the delivery row takes over
            self.report.changed = self.report.changed or any(counts[u]["ok"] for u in counts)
        finally:
            self._blocked = None
            self._indexed = set()

    def _load_cursor(self, uri: str, resumed_from: int):
        """The committed progress of a file this parse resumed: its accumulated counts, or None when the cursor does
        not match the parse (the file was re-read from the start)."""
        row = self.conn.execute("SELECT records_done, counts_json FROM record_cursor WHERE file_uri = ?", (uri,)).fetchone()
        if row is None or int(row[0]) != int(resumed_from):
            return None
        data = json.loads(row[1] or "{}")
        counts = {"ok": 0, "refused": 0, "blocked": 0, "duplicate": 0}
        counts.update({k: int(v) for k, v in (data.get("counts") or {}).items()})
        return (counts, {k: int(v) for k, v in (data.get("types") or {}).items()}, int(data.get("n_refused") or 0), int(row[0]))

    def _deliver(self, pf: ParsedFile, now: str, status: str, reason: Optional[str], type_counts: Mapping[str, int],
                 c: Optional[Mapping[str, int]] = None, n_ref: int = 0) -> None:
        c = c or {"ok": 0, "blocked": 0, "duplicate": 0}
        self.conn.execute(
            _DELIVERY_SQL,
            (pf.uri, pf.file_sha or hashlib.sha256(b"").hexdigest(), now, now, status, reason, pf.producer_script, pf.producer_session,
             pf.schema_version, canonical_json(type_counts), pf.total_records, c["ok"], n_ref, c["blocked"], c["duplicate"]))

    def _metric(self, obj: Mapping[str, object]) -> None:
        if not self.live:
            return
        try:
            with open(os.path.join(state_dir(self.root), "metrics.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(canonical_json(obj) + "\n")
        except OSError:
            pass

    # -- revision directories -------------------------------------------
    def apply_revision(self, rev_dir: str, now: str, retry: bool = False) -> str:
        """:meth:`_apply_revision` with its outcome on the metrics log: one ``kind = "bundle"`` line per revision directory or
        activation bundle (its uri, ``ok`` / ``blocked`` / ``pending``, its wall time, ``no_op`` when it folded as one, ``retry``
        when the blocked queue re-attempted it), so a cycle's per-bundle progress is readable from ``state/metrics.jsonl`` -- the
        2026-09-29 recovery cycles logged 1,067 ``staging`` and 36 ``activation`` lines and one ``transaction`` in 80 minutes, and
        nothing said how many bundles that was or which. Every bundle's work is committed on its own (the staging windows, the
        activation, or the held path's one transaction), so a cycle killed by its unit bound keeps every bundle this method
        returned from. ``retry`` is the blocked queue's re-attempt (:meth:`reevaluate_blocked`): always a full attempt, never a no-op."""
        t0 = time.monotonic()
        result = "error"
        self._noop_kind = None
        try:
            result = self._apply_revision(rev_dir, now, retry=retry)
            return result
        finally:
            line = {"kind": "bundle", "dir": logical_uri(self.root, rev_dir).rstrip("/"), "result": result,
                    "seconds": round(time.monotonic() - t0, 4), "utc": now}
            if self._noop_kind:
                line["no_op"] = self._noop_kind
            if retry:
                line["retry"] = True
            self._metric(line)

    def _apply_revision(self, rev_dir: str, now: str, retry: bool = False) -> str:
        """Apply one revision (or re-activation) directory in two phases. STAGING: every batch file is
        streamed in bounded windows and its constituents are validated, dependency-checked and written INACTIVE —
        run-scale records into ``revision_staging_record``, item rows into the ``*_staging`` shadows, every row
        tagged with the revision id and read by no view — one transaction of at most ITEM_ROWS_PER_TRANSACTION
        rows per window. ACTIVATION: one small transaction folds the few staged run-scale constituents to a fixed
        point, moves the staged item rows into the live tables with one set-based statement per table, folds the
        ``source_revision`` marker LAST and drops the staging rows. Any malformed constituent (item or run-scale),
        blocked dependency or constraint failure holds the WHOLE revision: the activation rolls back, the
        previous revision stays entirely current, a PENDING marker row (accepted = 0, the reason in ``refusal``)
        shows the state, the constituent and the marker (tagged with the revision id) sit in ``blocked_records``
        and the revision is retried whole — a held revision keeps its staged rows (every completely staged file) and
        activates later without re-reading them. Returns ok | blocked | pending."""
        uri_dir = logical_uri(self.root, rev_dir).rstrip("/") + "/"
        rev = revision_of_uri(uri_dir)
        assert rev is not None, rev_dir
        self._staged_cache = None
        files = self._rev_files(rev_dir)
        marker_path, marker_records = self._find_marker(files, rev)
        if marker_path is None:
            # no marker yet (a crash between staging and publication, or a concurrent publisher): nothing applies; the
            # files wait — one blocked row per file, keyed by the file's logical sha — until the marker arrives
            def unmarked() -> None:
                self._blocked = {r[0] for r in self.conn.execute("SELECT record_sha256 FROM blocked_records").fetchall()}
                for path, _ in files:
                    pf = self._file_summary(path)
                    if pf.error:
                        continue
                    rtype = self._file_record_type(pf.uri)
                    self._block(pf.file_sha, pf.uri, rtype, "source_revision", rev, now, {"revision_id": rev})
                    self._deliver(pf, now, "blocked", None, {rtype: pf.total_records}, {"ok": 0, "blocked": pf.total_records, "duplicate": 0}, 0)
                self._blocked = None
            self._timed_transaction(unmarked, len(files), 0, now, kind="no_marker")
            return "blocked"
        is_bundle = "/_activations/" in uri_dir
        batch_files = [path for path, _ in files if path != marker_path]
        facts = [p for p in batch_files if self._file_record_type(logical_uri(self.root, p)) in FACT_RECORD_TYPES]   # the status files
        measurement = [p for p in batch_files if p not in facts]
        if is_bundle and not retry:
            hold = self._recorded_hold(rev)
            if hold is not None and self._same_measurements(hold[4], measurement):
                # (2026-09-29) a bundle of a revision the harvester HOLDS whose measurement content IS, record for record (stamps
                # aside) and file for file in both directions, the directory or bundle the hold was recorded on: the same records hold
                # the same way, so it folds as a no-op -- delivered blocked under the recorded hold, nothing staged, not attempted while
                # the hold stands (the carrier re-attempts on its behalf; when the carrier activates, this bundle is a bundle of the head
                # and its marker and status facts fold then, in stamp order, below). A bundle whose content differs -- a corrected
                # reader's, a file removed or added -- is attempted in full, here and now, as before this rule
                return self._noop_held_bundle(rev, files, marker_path, marker_records, hold, now)
        duplicates: List[str] = []
        if is_bundle and retry and self._revision_is_head(rev) and self._is_noop_bundle(uri_dir.rstrip("/")) \
                and not self._same_measurements(self._head_applied_dir(rev) or "", measurement):
            # (2026-09-29) a no-op bundle met after its revision activated through a DIFFERENT carrier (a later, correcting read): its
            # measurements are the superseded carrier's and its facts are older than the correction's (a no-op only ever matches the
            # carrier current at its arrival, and carriers arrive in stamp order), so nothing of it can change the database -- folding it
            # would only re-refuse the old content or re-stage rows the correction already replaced. It is a stale copy: delivered
            # duplicate, its rows leave the queue with it. The same on replay, where the correction activates before this bundle is met
            return self._stale_noop_bundle(files, now)
        if is_bundle and self._revision_is_head(rev) and self._same_measurements(self._head_applied_dir(rev) or "", measurement):
            # (2026-09-29) the revision is ALREADY the accepted head and the bundle's measurement files ARE what the head was folded
            # from -- the complete multiset, record for record with written_utc aside, in both directions (_same_measurements): a
            # re-activation that landed twice, or a held revision's identical re-read met after its directory activated. The
            # measurement files are ranked and delivered `duplicate` unstaged (no item row re-inserted, nothing for reclaim to flip);
            # the MARKER and the STATUS files fold normally, so the read's newer facts land -- the head re-observed, its watermarks,
            # source_state, a status row that now reports a source-level refusal (two identical re-reads
            # whose later status said `refuse` folded as a duplicate and the source read healthy). Any measurement difference -- a
            # corrected reader's bundle re-attempted after the revision's directory activated on the same parent (the
            # corrected run_metric marked duplicate and lost), a file the correction removed -- stages the bundle in full
            duplicates = measurement
            batch_files = facts
            self._noop_kind = "head"
            self.report.revisions_noop += 1
        held: Optional[_RevisionHeld] = None
        try:
            self._stage_revision(rev, batch_files, marker_records, logical_uri(self.root, marker_path), now)
            self._activate_revision(rev, files, marker_path, marker_records, now, duplicates=duplicates)
            self.reclaim(now)
            return "ok"
        except _RevisionHeld as exc:
            held = exc
        self.report.revisions_held += 1
        if held.kind == "retry":
            return "pending"          # a file changed under the reader: nothing recorded, the next cycle reads it again
        # held: rank the files (a file's rank is its own), record WHY, leave the previous revision current -- ONE committed
        # transaction (kind = "held" on the metrics log, counted with the cycle's transactions like every other write; before
        # 2026-09-29 it ran outside the timed path, so a cycle that held 36 bundles showed their staging windows and their
        # rolled-back activations and not one line for the commits that recorded the holds)
        summaries = self._revision_summaries(rev, files)
        marker_uri = logical_uri(self.root, marker_path)
        mpf = summaries[marker_uri]
        marker_sha = next((rsha for _, rec, rsha in marker_records if rec.get("record_type") == "source_revision"), mpf.file_sha)
        marker_extra_recorded: List[Optional[Dict[str, object]]] = [None]      # what record_hold put on the marker's queue row (the memo's identity reads it)

        def record_hold() -> None:
            self._blocked = {r[0] for r in self.conn.execute("SELECT record_sha256 FROM blocked_records").fetchall()}
            ranker = self.ranker
            ranker.reload()
            for pf in summaries.values():
                if pf.error or pf.retry:
                    continue
                first = pf.first_key
                if first is not None:
                    # a file with no well-formed record has no place in the ledger (as in a plain chunk) — never the clock,
                    # which ranked a refused one-record constituent differently live and on replay
                    ranker.assign(pf.uri, pf.file_sha, first[0], pf.total_records, now)
                self._deliver(pf, now, "blocked", None, {self._file_record_type(pf.uri): pf.total_records},
                              {"ok": 0, "blocked": pf.total_records, "duplicate": 0}, 0)
            ranker.finalize()
            reason = f"activation pending: {held.parent_table} {held.parent_id} ({held.detail})"
            if held.kind == "refused":
                # names the record: <file_uri>#<line>; the revision's sound files stay staged, the failing one is re-read on retry
                self._refuse(held.sha, held.parent_id, held.record_type, None, held.detail, now)
            elif held.kind == RUN_SET_DROP_KIND:
                # the marker itself is what holds (its membership against the head): its own blocked row, below, carries
                # the reason; every constituent applied and stays staged, the head stays current
                self.report.revisions_held_run_drop += 1
            elif held.kind == RETIREMENT_HEAD_KIND:
                # the marker itself is what holds (a retirement's premise against the live head, registry-hardening 3a round 2): its own
                # blocked row, below, carries the reason; a retirement has no constituents, the head stays current, nothing is retired
                self.report.revisions_held_retirement_head += 1
            else:
                self._block(held.sha, held.detail, held.record_type or "run", held.parent_table, held.parent_id, now, {"revision_id": rev})
                # the other constituents the fixed-point pass could not apply (the run records behind a named run_metric,
                # the dependents of a root run): each in the queue by its own sha, tagged with the revision, so `blocked
                # --stale` / the status page name them and the activation's DELETE by revision_id clears them together
                for sha, uri, rtype, ptable, pid in held.siblings:
                    self._block(sha, uri, rtype or "run", ptable, pid, now, {"revision_id": rev, "held_with": held.detail})
            marker_extra = {"revision_id": rev, "blocked_on": held.detail, "kind": held.kind}
            if held.kind == "refused":
                # the exact hold's key (2026-09-30): the refusing file, the logical sha it was delivered under above, and the stamp of the
                # code that judged it -- the hold is re-attempted only when one of the three moves (_refused_hold_exact); a real attempt
                # rewrites the key (this UPDATE replaces extra_json whole), so the judgment on record is always the latest attempt's
                ref = summaries.get(held.file_uri)
                marker_extra.update(refused_file=held.file_uri, judged_with=validator_stamp(),
                                    refused_file_sha=(ref.file_sha if ref is not None and not ref.error and not ref.retry else None))
            if held.kind == RUN_SET_DROP_KIND:
                marker_extra["rule"] = RUN_SET_DROP_RULE          # the one-line rule beside the reason, the way the input guard's record carries its own
            elif held.kind == RETIREMENT_HEAD_KIND:
                marker_extra["rule"] = RETIREMENT_HEAD_RULE
            self._block(marker_sha, marker_uri, "source_revision", held.parent_table, held.parent_id, now, marker_extra)
            marker_extra_recorded[0] = dict(marker_extra)
            self._placeholder_revision(rev, marker_records, marker_uri, refusal=reason)
            self._blocked = None
        self._timed_transaction(record_hold, len(files), 0, now, kind="held")
        # the identity of what this attempt read -- the directory's listing, the marker's sha, the judging
        # code and (a blocked kind) whether the parent is present -- and when, memoized in state/held_attempts.json (a STATE file, keyed by the
        # directory: never on the marker's queue row, whose extra_json is a pure function of the log's bytes and must replay byte for byte; a
        # clock and a listing's mtimes are not). The re-evaluation skips the directory while the identity stands, re-attempts it when it moves
        # or after `held_sweep_s`, and always under --retry-held
        self._note_held_attempt(rev_dir, self._hold_identity(rev_dir, marker_sha, held.kind, held.parent_table, held.parent_id, marker_extra_recorded[0]), now)
        return "blocked"

    def _revision_is_head(self, rev: str) -> bool:
        """``rev`` is accepted, not superseded, and the LATEST accepted revision of its source by ``(observed_utc, revision_id)`` --
        the head every reader derives visibility from."""
        row = self.conn.execute("SELECT source_id FROM source_revision WHERE revision_id = ? AND accepted = 1 AND superseded_by_revision_id IS NULL",
                                (rev,)).fetchone()
        if row is None:
            return False
        head = self.conn.execute("SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 "
                                 "ORDER BY observed_utc DESC, revision_id DESC LIMIT 1", (str(row[0]),)).fetchone()
        return head is not None and str(head[0]) == rev

    def _head_applied_dir(self, rev: str) -> Optional[str]:
        """The directory whose files the accepted head ``rev`` was folded from: the revision directory, or the activation bundle whose
        marker folded last (``source_revision.record_uri`` names that marker; the fold keeps the latest-stamped marker's carrier)."""
        row = self.conn.execute("SELECT record_uri FROM source_revision WHERE revision_id = ? AND accepted = 1", (rev,)).fetchone()
        if row is None or not row[0]:
            return None
        path = resolve_uri(self.root, str(row[0]))
        return os.path.dirname(path) if path else None

    @staticmethod
    def _content_sha(path: str) -> Optional[str]:
        """A record file's CONTENT identity: sha256 over its records' canonical JSON with ``written_utc`` removed, sorted -- what a
        re-emission of committed bytes (``runner._reemit_committed``: the same records under fresh stamps) reproduces and any change
        to a record's payload, a record added or dropped, does not. None when the file cannot be read as records."""
        try:
            lines = sorted(canonical_json({k: v for k, v in rec.items() if k != "written_utc"})
                           for _, rec, _ in R.read_record_file(path, compute_sha=False) if isinstance(rec, dict))
        except (ValueError, OSError, EOFError, gzip.BadGzipFile):
            return None
        h = hashlib.sha256()
        for line in lines:
            h.update(line.encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()

    def _measurement_multiset(self, directory: str, files: Optional[Sequence[str]] = None) -> Optional[Dict[Tuple[str, str], int]]:
        """The MEASUREMENT content of a revision directory or bundle as a multiset of ``(record type, content sha)`` over its files --
        every file but the marker, the status files and the artifact files (:data:`COMPARED_APART`): the input registrations are fixed
        by the revision id (the same inputs, the same artifacts) and a re-emission of committed bytes omits them by design
        (``runner._REEMIT_SKIP``). ``files`` names the files to read (a bundle's measurement files); None reads the directory. None
        when the directory is gone or a file cannot be read."""
        if files is None:
            if not directory or not os.path.isdir(directory):
                return None
            files = [path for path, _ in self._rev_files(directory)]   # the allowlist applies here too: the head's directory as the replay sees it
        out: Dict[Tuple[str, str], int] = {}
        for path in files:
            rtype = self._file_record_type(logical_uri(self.root, path))
            if rtype in COMPARED_APART:
                continue
            sha = self._content_sha(path)
            if sha is None:
                return None
            out[(rtype, sha)] = out.get((rtype, sha), 0) + 1
        return out

    def _same_measurements(self, carrier_dir: str, files: Sequence[str]) -> bool:
        """Whether a bundle's measurement ``files`` and ``carrier_dir`` (a revision directory or an activation bundle already folded
        or attempted) carry the SAME measurements: equal multisets of ``(record type, content sha)`` in BOTH directions -- a file the
        bundle adds, a file it omits (a correction that removed an erroneous, separately emitted judgment file), a record changed,
        added or dropped, all answer False, and so do an empty measurement set and a carrier that is gone. True is the proof both
        no-op shortcuts need: folding the files would write rows the database already carries (or would hold on exactly what the
        carrier holds on)."""
        have = self._measurement_multiset(carrier_dir)
        got = self._measurement_multiset("", list(files))
        return bool(have) and bool(got) and have == got

    def _light_summary(self, path: str) -> ParsedFile:
        """What a delivery and a ledger row need about a file NOBODY stages this cycle: its logical sha (the cycle's listing computed
        it for every new file; a decompressing hash otherwise), its stat, its record count and its FIRST WRITTEN STAMP -- the ledger's
        ``first_written_utc``, the smallest ``written_utc`` over its records, read with a plain JSON parse of each line and no
        validation, no classification, no item row built (a runner-written file carries one stamp on every record, the pass's).
        A line that is not JSON sends the file through :meth:`_file_summary`, the validating stream, so a torn file is refused by
        name as it always was."""
        uri = logical_uri(self.root, path)
        stat = _stat3(path)
        sha = self.file_shas.get(path) or R.file_logical_sha(path)
        n = 0
        first: Optional[str] = None
        try:
            for _, rec, _ in R.read_record_file(path, compute_sha=False):
                n += 1
                w = rec.get("written_utc") if isinstance(rec, dict) else None
                if isinstance(w, str) and (first is None or w < first):
                    first = w
        except (ValueError, OSError, EOFError, gzip.BadGzipFile):
            return self._file_summary(path)
        pf = ParsedFile(path, uri, sha, stat=stat, whole_records=n)
        if first is not None:
            pf.whole_first_key = (first, "")
        return pf

    def _recorded_hold(self, rev: str) -> Optional[Tuple[str, str, str, str, str]]:
        """The hold the harvester recorded for ``rev`` by an ATTEMPT -- ``(parent_table, parent_id, kind, blocked_on, carrier_dir)`` off
        the ``source_revision``-typed row of ``blocked_records`` tagged with the revision that a real attempt wrote (never a no-op's
        copy), the latest attempted bundle over the revision's directory, by stamp -- or None when the revision is not held (never
        attempted, accepted, or held only by a marker still to arrive). The carrier's directory is what a new bundle's content is
        compared with. Chosen by the rows' uris alone, never by when they were first seen, so live and replay pick the same carrier."""
        row = self.conn.execute(
            "SELECT missing_parent_table, missing_parent_id, json_extract(extra_json, '$.kind'), json_extract(extra_json, '$.blocked_on'), record_uri "
            "FROM blocked_records WHERE record_type = 'source_revision' AND json_extract(extra_json, '$.revision_id') = ? "
            "AND json_extract(extra_json, '$.kind') IN ('blocked', 'refused', ?) AND json_extract(extra_json, '$.no_op') IS NULL "
            "ORDER BY (record_uri LIKE '%/_activations/%') DESC, record_uri DESC LIMIT 1", (rev, RUN_SET_DROP_KIND)).fetchone()
        if row is None or self.conn.execute("SELECT 1 FROM source_revision WHERE revision_id = ? AND accepted = 1", (rev,)).fetchone():
            return None
        carrier = resolve_uri(self.root, str(row[4]))
        return str(row[0]), str(row[1]), str(row[2] or "blocked"), str(row[3] or ""), os.path.dirname(carrier) if carrier else ""

    def _stale_noop_bundle(self, files: Sequence[Tuple[str, Optional[str]]], now: str) -> str:
        """A no-op bundle whose carrier a correcting read superseded before the revision activated (:meth:`_apply_revision`): every file
        -- ranked already, at no-op time -- is delivered ``duplicate`` in one timed transaction (``kind = "stale_noop"``); nothing folds.
        The ``bundle`` line says ``no_op = "stale"``."""
        def body() -> None:
            for path, _ in files:
                pf = self._light_summary(path)
                if pf.error:
                    continue
                rtype = self._file_record_type(pf.uri)
                self._deliver(pf, now, "duplicate", None, {rtype: pf.total_records}, {"ok": 0, "blocked": 0, "duplicate": pf.total_records}, 0)
                if pf.stat is not None:
                    self.learned[path] = {"size": pf.stat[0], "mtime_ns": pf.stat[1], "ino": pf.stat[2], "sha": pf.file_sha}
        self._timed_transaction(body, len(files), 0, now, kind="stale_noop")
        self._noop_kind = "stale"
        self.report.revisions_noop += 1
        return "ok"

    def _noop_held_bundle(self, rev: str, files: Sequence[Tuple[str, Optional[str]]], marker_path: str, marker_records,
                          hold: Tuple[str, str, str, str, str], now: str) -> str:
        """A bundle of a revision the harvester HOLDS whose measurement content repeats the attempted carrier's (2026-09-29), folded in
        the cycle's main pass as a NO-OP. The same records -- a parent nothing has published, a run set against the same head, bytes
        the validator refused -- hold the same way, so staging it would clear the held attempt's staged rows and stage the same rows
        to be held again; the log carried 1,649 such bundles for 99 held revisions (~17 apiece, one per adapter pass, identical to the
        record but for ``written_utc``) when the harvester could not fold 97 of them inside its 45-minute bound. Instead, one timed
        transaction (``kind = "held_noop"``) does what the held path does WITHOUT the staging and the activation: every file takes its
        ledger rank (from a light summary: sha, stat, record count, first stamp; nothing validated) and is delivered ``blocked``; the
        marker takes a blocked row copying the recorded hold, tagged ``no_op``; the pending row keeps its reason. A no-op bundle is
        never attempted: its marker is as exact a hold as the carrier's (:meth:`_held_still_blocked`), and the carrier re-attempts on
        its behalf -- when its parent lands the carrier activates and every bundle of the revision leaves the queue; a bundle whose
        content differs is attempted in full the moment it arrives. Live and replay agree by construction: the decision is a pure
        function of the log's bytes and of the carrier, which both paths pick by uri."""
        ptable, pid, kind, blocked_on, _carrier = hold
        marker_uri = logical_uri(self.root, marker_path)
        marker_sha = next((rsha for _, rec, rsha in marker_records if rec.get("record_type") == "source_revision"), None)
        reason = f"activation pending: {ptable} {pid} ({blocked_on})"
        marker_sha_local: List[Optional[str]] = [None]

        def body() -> None:
            self._blocked = {r[0] for r in self.conn.execute("SELECT record_sha256 FROM blocked_records").fetchall()}
            ranker = self.ranker
            ranker.reload()
            for path, _ in files:
                pf = self._light_summary(path)
                if pf.error:
                    continue
                first = pf.first_key
                if first is not None:
                    ranker.assign(pf.uri, pf.file_sha, first[0], pf.total_records, now)
                rtype = self._file_record_type(pf.uri)
                self._deliver(pf, now, "blocked", None, {rtype: pf.total_records}, {"ok": 0, "blocked": pf.total_records, "duplicate": 0}, 0)
                if pf.stat is not None:
                    self.learned[path] = {"size": pf.stat[0], "mtime_ns": pf.stat[1], "ino": pf.stat[2], "sha": pf.file_sha}
                if path == marker_path:
                    marker_sha_local[0] = pf.file_sha
            ranker.finalize()
            self.report.reranks += ranker.reranks
            extra = {"revision_id": rev, "blocked_on": blocked_on, "kind": kind, "no_op": "held"}
            if kind == RUN_SET_DROP_KIND:
                extra["rule"] = RUN_SET_DROP_RULE
            self._block(marker_sha or marker_sha_local[0], marker_uri, "source_revision", ptable, pid, now, extra)
            self._placeholder_revision(rev, marker_records, marker_uri, refusal=reason)
            self._blocked = None
        self._timed_transaction(body, len(files), 0, now, kind="held_noop")
        self._noop_kind = "held"
        self.report.revisions_noop += 1
        return "blocked"

    # -- staging --
    def _stage_revision(self, rev: str, batch_files: Sequence[str], marker_records, marker_uri: str, now: str) -> None:
        """Stage every batch file of ``rev`` that is not already staged with its current bytes. Pass A streams each
        file for its (few) run-scale constituents and stages them as records; pass B streams each file for its item
        rows and writes them INTO THE ITEM TABLES tagged with the revision — part of their key, so they coexist with
        the current revision's rows and are read by no view until the marker flips — parents checked against
        the live tables PLUS the revision's own staged run-scale keys, one bounded window per transaction, the file
        recorded (sha, counts, envelope, stat) when its last window commits. Raises _RevisionHeld on a malformed
        constituent, a blocked item, a constraint failure or a file that changed under the reader."""
        self._activating = rev
        try:
            staged = {r[0]: (r[1], r[2]) for r in self.conn.execute(
                "SELECT file_uri, file_sha256, stat_json FROM revision_staging_file WHERE revision_id = ?", (rev,)).fetchall()}
            todo: List[Tuple[str, str]] = []
            for path in batch_files:
                furi = logical_uri(self.root, path)
                row = staged.get(furi)
                if row is not None and self._staging_current(path, row[0], row[1]):
                    continue
                todo.append((path, furi))
            if not todo:
                return
            here = bundle_of_uri(todo[0][1])
            if "/_activations/" in here:
                # a bundle is a WHOLE fresh read (or re-emission) of the revision: whatever an earlier bundle of the same
                # revision staged is replaced, never folded beside it — a state for a run the earlier read named and the
                # corrected read no longer does would otherwise hold every later bundle on the earlier one's line (the
                # original directory's staged files stay: they share the revision and the runner's re-read counts on them)
                stale = [r[0] for r in self.conn.execute(
                    "SELECT DISTINCT file_uri FROM (SELECT file_uri FROM revision_staging_file WHERE revision_id = ? "
                    "UNION SELECT file_uri FROM revision_staging_record WHERE revision_id = ?)", (rev, rev)).fetchall()
                    if "/_activations/" in r[0] and not r[0].startswith(here)]
                for furi in stale:
                    self._clear_staged_file(rev, furi, now)
            mpf = self._file_summary(resolve_uri(self.root, marker_uri) or marker_uri)
            mfirst = mpf.first_key

            def prepare() -> None:
                # the marker file takes its ledger rank NOW, so a re-rank it causes (its key sorting before files already
                # ranked) is this bounded transaction's work, never the activation's; the revision ROW (accepted = 0:
                # pending, read by nothing) is the FK target of every tagged row
                self.ranker.reload()
                self.ranker.assign(marker_uri, mpf.file_sha, mfirst[0] if mfirst else now, mpf.total_records, now)
                self.ranker.finalize()
                self.report.reranks += self.ranker.reranks
                if self.conn.execute("SELECT 1 FROM source_revision WHERE revision_id = ?", (rev,)).fetchone() is None:
                    self._placeholder_revision(rev, marker_records, marker_uri)
            self._staging_transaction(prepare, 1, mpf.total_records, now)
            envelope: Dict[str, Tuple] = {}
            census: Dict[str, Tuple[int, ParsedFile]] = {}    # per file after pass A: (item records seen and skipped, the last part)
            for path, furi in todo:
                self._clear_staged_file(rev, furi, now)
                seen = 0
                n_items = 0
                for part in stream_parts(self.root, path, None, ITEM_ROWS_PER_TRANSACTION, want="run_scale", keep_raw=True):
                    self._check_part(part)
                    for pr in part.records:
                        if isinstance(pr.result, VAL.Malformed):
                            raise _RevisionHeld("refused", "record", pr.result.path, furi, pr.sha,
                                                f"{pr.result.field}: {pr.result.reason}", pr.result.record_type)
                    seen += part.n_records
                    n_items += part.skipped
                    if part.last:
                        census[furi] = (n_items, part)
                    if part.producer_script is not None:
                        envelope[furi] = (part.producer_script, part.producer_session, part.schema_version)
                    if not part.records:
                        continue
                    rows = [(rev, furi, pr.line, pr.sha, pr.result.written_utc, canonical_json(pr.raw)) for pr in part.records]
                    n_ledger = part.total_records if part.last else seen
                    first = part.first_key

                    def stage_records(rows=rows, first=first, n_ledger=n_ledger, sha=part.file_sha):
                        self._staged_cache = None
                        self.ranker.reload()
                        if first is not None:
                            self.ranker.assign(furi, sha, first[0], n_ledger, now)
                        self.conn.executemany(
                            "INSERT OR REPLACE INTO revision_staging_record(revision_id, file_uri, line, record_sha256, written_utc, record_json) "
                            "VALUES (?,?,?,?,?,?)", rows)
                        self.ranker.finalize()
                        self.report.reranks += self.ranker.reranks
                    self._staging_transaction(stage_records, 1, len(rows), now)
            # existence for the item pass: the live tables PLUS every staged run-scale key of this revision
            ex = _Existence(self.conn)
            ex.note("source_revision", "revision_id", rev)
            for v, _ in self._staged_records(rev):
                for eff in effects_of(v):
                    if eff.kind == "full":
                        pk = R.NATURAL_KEYS[eff.table]
                        if len(pk) == 1:
                            ex.note(eff.table, pk[0], eff.key[pk[0]])
            for path, furi in todo:
                counts = {"ok": 0, "refused": 0, "blocked": 0, "duplicate": 0}
                types: Dict[str, int] = {}
                seen = 0
                seen_a = census.get(furi)
                if seen_a is not None and seen_a[0] == 0:
                    # pass A read every record of this file and skipped none as an item: the items pass would stream the same
                    # bytes to find nothing. Its one effect — the file's staging row, rank and delivery counts from a last part
                    # with no item — is written from pass A's read instead (the part it would have produced: every record
                    # skipped, no key, the verified sha and stat); a run-scale-only file is thereby read ONCE (F16)
                    last = seen_a[1]
                    part = ParsedFile(path, furi, last.file_sha, stat=last.stat, part_index=0, last=True, whole_records=last.total_records,
                                      skipped=last.total_records, producer_script=last.producer_script, producer_session=last.producer_session,
                                      schema_version=last.schema_version)
                    self._staging_transaction(
                        lambda part=part: self._stage_window(rev, furi, part, ex, counts, types, envelope.get(furi), part.n_records, now),
                        1, part.n_records, now)
                    continue
                for part in self._item_parts(path):
                    self._check_part(part)
                    bad = next((pr for pr in part.records if isinstance(pr.result, VAL.Malformed)), None)   # an items pass files only malformed items here
                    if bad is not None:
                        raise _RevisionHeld("refused", "record", bad.result.path, furi, bad.sha,
                                            f"{bad.result.field}: {bad.result.reason}", bad.result.record_type)
                    seen += part.n_records
                    if part.producer_script is not None and furi not in envelope:
                        envelope[furi] = (part.producer_script, part.producer_session, part.schema_version)
                    if not part.items and not part.last:
                        continue
                    self._staging_transaction(
                        lambda part=part, seen=seen: self._stage_window(rev, furi, part, ex, counts, types, envelope.get(furi), seen, now),
                        1, part.n_records, now)
        finally:
            self._activating = None
            self._blocked = None
            self._indexed = set()

    def _item_parts(self, path: str) -> Iterator[ParsedFile]:
        """The items pass over one batch file, in bounded windows: the prefetched whole-file parse split into parts when
        the replay's pool parsed it ahead (``_Prefetcher``), else streamed in-process. The two agree on every row: the
        same ``_classify`` builds the items; a part's run-scale records (present in a whole-file parse, skipped by the
        streamed items pass) are ignored here — they were staged by pass A — and count toward the file's total either way;
        the file's first key is the minimum over every well-formed record in both, since the ledger keeps the minimum."""
        pf = self.prefetch.take(path) if self.prefetch is not None else None
        if pf is None:
            return stream_parts(self.root, path, None, ITEM_ROWS_PER_TRANSACTION, want="items")
        return iter(pf.parts(ITEM_ROWS_PER_TRANSACTION))

    def _stage_window(self, rev: str, furi: str, part: ParsedFile, ex: _Existence, counts: Dict[str, int], types: Dict[str, int],
                      env: Optional[Tuple], seen: int, now: str) -> None:
        """One window of item rows into the staging shadow (inside its own transaction)."""
        self.ranker.reload()
        self._indexed = set()
        first = part.first_key
        n_ledger = part.total_records if part.last else seen
        if first is not None:
            seq = self.ranker.assign(furi, part.file_sha, first[0], n_ledger, now)
        else:
            seq = self.ranker.seq_of(furi)
            if seq is None:
                seq = self.ranker.assign(furi, part.file_sha, now, n_ledger, now) if part.last else 0
        batches: Dict[str, List[Tuple[Tuple, int]]] = {}
        for pi in part.items:
            for ptable, pcol, pid in pi.parents:
                if not ex.exists(ptable, pcol, pid):
                    raise _RevisionHeld("blocked", ptable, pid, furi, pi.sha, f"{furi}#{pi.line}", pi.table)
            batches.setdefault(pi.table, []).append((pi.row + ITEM_PLANS[pi.table].tail(seq), pi.line))
            counts["ok"] += 1
            types[pi.table] = types.get(pi.table, 0) + 1
            self._index_run(pi.run_ids, furi)
        for table, pairs in batches.items():
            offs = ITEM_PLANS[table].pk_positions
            pairs.sort(key=lambda pr: tuple(pr[0][i] for i in offs))
            rows = [r for r, _ in pairs]
            try:
                self._stage_rows(table, rows)
            except sqlite3.IntegrityError as exc:
                # a constraint failure is a revision-level failure naming the row: replay one by one to find it
                bad = self._first_bad_row(table, rows, [ln for _, ln in pairs])
                if bad is not None:
                    line, sha, err = bad
                    raise _RevisionHeld("refused", "record", f"{furi}#{line}", furi, sha, f"IntegrityError: {err}", table) from exc
        self.ranker.finalize()
        self.report.reranks += self.ranker.reranks
        if part.last:
            row = self.conn.execute("SELECT first_written_utc FROM record_ingest WHERE file_uri = ?", (furi,)).fetchone()
            env = env or (None, None, None)
            stat = {"size": part.stat[0], "mtime_ns": part.stat[1], "ino": part.stat[2]} if part.stat is not None else {}
            self.conn.execute(
                "INSERT OR REPLACE INTO revision_staging_file(revision_id, file_uri, file_sha256, n_records, first_written_utc, counts_json, "
                "type_counts_json, producer_script, producer_session, schema_version, stat_json, staged_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (rev, furi, part.file_sha, part.total_records, row[0] if row else None, canonical_json(counts), canonical_json(types),
                 env[0], env[1], env[2], canonical_json(stat), now))

    def _stage_rows(self, table: str, rows: Sequence[Tuple]) -> None:
        """One window of tagged item rows into the live table (the row-wise fold's upsert; the key carries the revision)."""
        DB.executemany_batched(self.conn, self.reducer.item_upsert_sql(table), rows)

    def _replay_staged_row(self, table: str, sql: str, row: Tuple) -> None:
        with DB.savepoint(self.conn, "stg"):
            self.conn.execute(sql, row)

    def _first_bad_row(self, table: str, rows: Sequence[Tuple], lines: Sequence[int]):
        """After a batch failed: the first row that fails alone — ``(line, sha, error)`` — or None."""
        cols = self.reducer._cols[table]
        sql = self.reducer.item_upsert_sql(table)
        for row, line in zip(rows, lines):
            try:
                self._replay_staged_row(table, sql, row)
            except sqlite3.IntegrityError as exc:
                return (int(line), row[cols.index("record_sha256")].hex(), str(exc))
        return None

    def _staging_transaction(self, body, files: int, rows: int, now: str) -> None:
        self.report.staging_transactions += 1
        self._timed_transaction(body, files, rows, now, kind="staging")

    def _staging_current(self, path: str, sha: str, stat_json: str) -> bool:
        """A staged file still stands for the bytes on disk: same stat as when it was staged (F12), else the same sha."""
        try:
            st = _stat3(path)
        except OSError:
            return False
        stat = json.loads(stat_json or "{}")
        if stat and (stat.get("size"), stat.get("mtime_ns"), stat.get("ino")) == st:
            return True
        try:
            return R.file_logical_sha(path) == sha
        except (OSError, EOFError, gzip.BadGzipFile):
            return False

    def _clear_staged_file(self, rev: str, furi: str, now: str) -> None:
        """Drop whatever an earlier attempt staged for one file of ``rev`` (a partial window set, or stale bytes): the
        tagged item rows that file produced (pending, read by nothing) and the file's ledger rows."""
        if self.ranker.seq_of(furi) is None:
            # never ranked ⇒ no window of it ever committed: every staging window ranks its file in its own transaction
            # (stage_records / _stage_window), so a file absent from the ledger has no tagged row and no staging row to clear —
            # and the three table scans below (their index is dropped during a replay) are not paid once per file (F16)
            return
        self._staged_cache = None
        rfid = R.record_file_id_for_uri(furi)
        have = any(self.conn.execute(f"SELECT 1 FROM {t} WHERE source_revision_id = ? AND record_file_id = ? LIMIT 1", (rev, rfid)).fetchone()
                   for t in S.ITEM_SCALE_TABLES)
        have = have or self.conn.execute("SELECT 1 FROM revision_staging_record WHERE revision_id = ? AND file_uri = ? LIMIT 1", (rev, furi)).fetchone()
        have = have or self.conn.execute("SELECT 1 FROM revision_staging_file WHERE revision_id = ? AND file_uri = ? LIMIT 1", (rev, furi)).fetchone()
        if not have:
            return

        def clear():
            for t in S.ITEM_SCALE_TABLES:
                self.conn.execute(f"DELETE FROM {t} WHERE source_revision_id = ? AND record_file_id = ?", (rev, rfid))
            self.conn.execute("DELETE FROM revision_staging_record WHERE revision_id = ? AND file_uri = ?", (rev, furi))
            self.conn.execute("DELETE FROM revision_staging_file WHERE revision_id = ? AND file_uri = ?", (rev, furi))
        self._staging_transaction(clear, 1, 0, now)

    def _discard_staging(self, rev: str) -> None:
        """The staging ledgers of ``rev`` (inside the caller's transaction) once it activated or its directory is gone;
        the tagged item rows themselves stay — they ARE the revision's rows."""
        self.conn.execute("DELETE FROM revision_staging_record WHERE revision_id = ?", (rev,))
        self.conn.execute("DELETE FROM revision_staging_file WHERE revision_id = ?", (rev,))
        self._staged_cache = None

    def _staged_records(self, rev: str) -> List[Tuple[VAL.Validated, str]]:
        """The revision's staged run-scale constituents, validated again, in order key: ``[(Validated, file_uri)]``.
        Read once per attempt (the staging pass builds its existence notes from it, the activation folds it): the cache
        is dropped whenever a staging row of the revision is written or removed and at the start of every attempt."""
        if self._staged_cache is not None and self._staged_cache[0] == rev:
            return self._staged_cache[1]
        out: List[Tuple[VAL.Validated, str]] = []
        rows = self.conn.execute("SELECT file_uri, line, record_sha256, record_json FROM revision_staging_record WHERE revision_id = ? "
                                 "ORDER BY written_utc, record_sha256", (rev,)).fetchall()
        for furi, line, sha, rec_json in rows:
            where = f"{furi}#{line}"
            res = VAL.validate_record(json.loads(rec_json), where, sha)
            if isinstance(res, VAL.Malformed):
                raise _RevisionHeld("refused", "record", where, furi, sha, f"{res.field}: {res.reason}", res.record_type)
            res.uri = where
            out.append((res, furi))
        self._staged_cache = (rev, out)
        return out

    def _revision_summaries(self, rev: str, files: Sequence[Tuple[str, Optional[str]]]) -> Dict[str, ParsedFile]:
        """Per file of a held revision: what its delivery row needs — from the staging record when the file is staged,
        else from a read (outside any transaction)."""
        staged = {r[0]: r for r in self.conn.execute(
            "SELECT file_uri, file_sha256, n_records, first_written_utc, producer_script, producer_session, schema_version, stat_json "
            "FROM revision_staging_file WHERE revision_id = ?", (rev,)).fetchall()}
        out: Dict[str, ParsedFile] = {}
        for path, _ in files:
            furi = logical_uri(self.root, path)
            r = staged.get(furi)
            if r is None:
                out[furi] = self._file_summary(path)
                continue
            stat = json.loads(r[7] or "{}")
            out[furi] = ParsedFile(path, furi, r[1], producer_script=r[4], producer_session=r[5], schema_version=r[6],
                                   stat=(stat["size"], stat["mtime_ns"], stat["ino"]) if stat else None, whole_records=int(r[2]),
                                   whole_first_key=(r[3], "") if r[3] else None)
        return out

    # -- activation --
    def _activate_revision(self, rev: str, files: Sequence[Tuple[str, Optional[str]]], marker_path: str, marker_records, now: str,
                           duplicates: Sequence[str] = ()) -> None:
        """ONE small transaction — metadata only: the staged run-scale constituents (few) folded to a fixed
        point, the marker's own records LAST (``accepted = 1`` flips the source's head, and with it the visibility of
        every tagged item row), ``observation.current`` recomputed once per source, the deliveries, the staging ledgers
        dropped. No item row is written, moved or marked here. Raises _RevisionHeld (after the rollback) when a
        constituent is blocked or refused. ``duplicates`` are a head bundle's measurement files the head already carries
        (:meth:`_apply_revision`, 2026-09-29): ranked and delivered ``duplicate`` from a light summary in this same
        transaction, never staged."""
        self.report.revision_transactions += 1
        marker_uri = logical_uri(self.root, marker_path)
        staged_files = {r[0]: r for r in self.conn.execute(
            "SELECT file_uri, file_sha256, n_records, counts_json, type_counts_json, producer_script, producer_session, schema_version, stat_json "
            "FROM revision_staging_file WHERE revision_id = ?", (rev,)).fetchall()}
        n_rows = sum(int(r[2]) for r in staged_files.values())

        def body() -> None:
            ranker = self.ranker
            ranker.reload()
            self._blocked = {r[0] for r in self.conn.execute("SELECT record_sha256 FROM blocked_records").fetchall()}
            self._indexed = set()
            self._activating = rev
            self.reducer.deferred_current = set()
            counts: Dict[str, Dict[str, int]] = {}
            types: Dict[str, Dict[str, int]] = {}
            for furi, r in staged_files.items():
                counts[furi] = {"ok": 0, "refused": 0, "blocked": 0, "duplicate": 0}
                counts[furi].update({k: int(v) for k, v in json.loads(r[3] or "{}").items()})
                types[furi] = {k: int(v) for k, v in json.loads(r[4] or "{}").items()}
            # a pending row from an earlier held attempt gives way to the marker's own fold
            self.conn.execute("DELETE FROM source_revision WHERE revision_id = ? AND accepted = 0 AND refusal LIKE 'activation pending:%'", (rev,))
            # the revision ROW must exist before its constituents (item rows and observations reference it by FK), but
            # ACTIVATION is the marker's fold (accepted = 1), applied last: until then the row is a placeholder,
            # accepted = 0, invisible outside this transaction
            self._placeholder_revision(rev, marker_records, marker_uri)
            # pass A — the staged run-scale constituents (few), folded to a fixed point
            pending: List[Tuple[Tuple[int, Tuple[str, int, str]], VAL.Validated, str]] = [(apply_order(v), v, furi) for v, furi in self._staged_records(rev)]
            pending.sort(key=lambda t: t[0])              # the fold's order (apply_order = Validated.order_key), never a second spelling
            waited: Set[str] = set()
            while pending:
                ex = _Existence(self.conn)          # a fresh cache per pass: a parent that landed this pass is no longer "absent"
                ex.note("source_revision", "revision_id", rev)
                left: List[Tuple[Tuple[str, str], VAL.Validated, str]] = []
                for key, v, furi in pending:
                    c = counts.setdefault(furi, {"ok": 0, "refused": 0, "blocked": 0, "duplicate": 0})
                    st = self._apply_constituent(v, ranker.seq_of(furi) or 0, furi, ex, now, c, types.setdefault(furi, {}), hold_blocked=False)
                    if st == "blocked":
                        left.append((key, v, furi))
                        waited.add(v.sha)
                    elif v.sha in waited:
                        waited.discard(v.sha)
                        self.report.blocked = max(0, self.report.blocked - 1)   # it waited for a sibling, then applied
                if len(left) == len(pending):
                    # no constituent applied this pass: the revision is held. Name the ROOT — a blocked constituent whose
                    # missing parent is NOT itself a constituent of this revision — never whichever dependent sorts first:
                    # a run_metric waiting on a run that is staged here and itself waiting on an object made the marker
                    # say `run <id>` (a parent that can never land on its own, so `_held_still_blocked` skipped every
                    # re-attempt) while the run records were nowhere in the queue (r2_export f7858cf8, 2026-09-18). Every
                    # blocked constituent rides out through the exception: this transaction rolls back, and the held
                    # path re-blocks them so the queue shows the `run` rows and what each is missing.
                    staged_keys: Set[Tuple[str, str]] = set()
                    for v2, _ in self._staged_records(rev):
                        for eff in effects_of(v2):
                            if eff.kind == "full" and len(R.NATURAL_KEYS[eff.table]) == 1:
                                staged_keys.add((eff.table, str(eff.key[R.NATURAL_KEYS[eff.table][0]])))
                    blocked: List[Tuple[VAL.Validated, str, str, str]] = []
                    for _, v2, furi2 in left:
                        row = self.conn.execute("SELECT missing_parent_table, missing_parent_id FROM blocked_records WHERE record_sha256 = ?",
                                                (v2.sha,)).fetchone()
                        blocked.append((v2, furi2, str(row[0]) if row else "record", str(row[1]) if row else v2.uri))
                    v, furi, ptable, pid = next((b for b in blocked if (b[2], b[3]) not in staged_keys), blocked[0])
                    raise _RevisionHeld("blocked", ptable, pid, furi, v.sha, v.uri, v.record_type,
                                        siblings=[(b[0].sha, b[0].uri, b[0].record_type, b[2], b[3]) for b in blocked if b[0].sha != v.sha])
                pending = left
            # the marker file LAST: its source_revision record first, then the membership it carries
            ex = _Existence(self.conn)
            ex.note("source_revision", "revision_id", rev)
            counts[marker_uri] = {"ok": 0, "refused": 0, "blocked": 0, "duplicate": 0}
            types[marker_uri] = {}
            marker_valid = []
            for line, rec, rsha in marker_records:
                res = VAL.validate_record(rec, f"{marker_uri}#{line}", rsha)
                if isinstance(res, VAL.Malformed):
                    raise _RevisionHeld("refused", "record", res.path, marker_uri, rsha, f"{res.field}: {res.reason}", res.record_type)
                res.uri = f"{marker_uri}#{line}"
                marker_valid.append(res)
            marker_valid.sort(key=lambda v: (0 if v.record_type == "source_revision" else 1, v.order_key))
            # the retirement head gate (registry-hardening 3a, round 2): an EMPTY revision folds only against the head and inventory it
            # was computed from (extra.retirement_expects); a head that moved under it, or a live input it does not name, holds it here --
            # before the marker's own fold, which is what would withdraw the live head's keys and retire its watermarks
            moved = self._retirement_head_moved(rev, marker_valid)
            if moved is not None:
                mrec, head_rev, detail = moved
                raise _RevisionHeld(RETIREMENT_HEAD_KIND, "source_revision", head_rev or "(none)", marker_uri, mrec.sha, detail, "source_revision")
            # the run-set drop guard: a marker that would move the head and no longer names run ids the head carries,
            # none of them declared, holds the revision here -- after every constituent applied (so the hold is about the
            # membership, not a missing parent) and before the marker's own fold, which is what would flip the head
            drop = self._undeclared_run_drop(rev, marker_valid)
            if drop is not None:
                mrec, head_rev, detail = drop
                raise _RevisionHeld(RUN_SET_DROP_KIND, "source_revision", head_rev, marker_uri, mrec.sha, detail, "source_revision")
            mpf = self._file_summary(marker_path)
            mfirst = mpf.first_key
            mseq = ranker.assign(marker_uri, mpf.file_sha, mfirst[0] if mfirst else now, mpf.total_records, now)
            for v in marker_valid:
                self._apply_constituent(v, mseq, marker_uri, ex, now, counts[marker_uri], types[marker_uri])
            deferred = self.reducer.deferred_current or set()
            self.reducer.deferred_current = None
            for sid in sorted(deferred):
                self.reducer._recompute_current(sid)          # once per source, not once per observation key
            ranker.finalize()
            self.report.reranks += ranker.reranks
            total = mpf.total_records
            for furi, r in staged_files.items():
                path = resolve_uri(self.root, furi) or os.path.join(self.root, furi)
                stat = json.loads(r[8] or "{}")
                if stat:
                    self.learned[path] = {"size": stat["size"], "mtime_ns": stat["mtime_ns"], "ino": stat["ino"], "sha": r[1]}
                pf = ParsedFile(path, furi, r[1], producer_script=r[5], producer_session=r[6], schema_version=r[7], whole_records=int(r[2]))
                self._deliver(pf, now, "ok", None, types[furi], counts[furi], 0)
                total += int(r[2])
            if mpf.stat is not None:
                self.learned[mpf.path] = {"size": mpf.stat[0], "mtime_ns": mpf.stat[1], "ino": mpf.stat[2], "sha": mpf.file_sha}
            self._deliver(mpf, now, "ok", None, types[marker_uri], counts[marker_uri], 0)
            for path in duplicates:
                dpf = self._light_summary(path)
                dfirst = dpf.first_key
                if dfirst is not None:
                    ranker.assign(dpf.uri, dpf.file_sha, dfirst[0], dpf.total_records, now)
                rtype = self._file_record_type(dpf.uri)
                self._deliver(dpf, now, "duplicate", None, {rtype: dpf.total_records}, {"ok": 0, "blocked": 0, "duplicate": dpf.total_records}, 0)
                if dpf.stat is not None:
                    self.learned[path] = {"size": dpf.stat[0], "mtime_ns": dpf.stat[1], "ino": dpf.stat[2], "sha": dpf.file_sha}
                total += dpf.total_records
            self._discard_staging(rev)
            self.report.files += len(files)
            self.report.records += total
            self.report.changed = True

        try:
            self._timed_transaction(body, len(files), n_rows, now, kind="activation")
        finally:
            self.reducer.deferred_current = None
            self._activating = None
            self._blocked = None
            self._indexed = set()

    def reclaim(self, now: str) -> int:
        """``withdrawn_by_revision_id`` is bookkeeping: visibility derives from the head revision, so the
        marks are materialised HERE, after activations, in bounded transactions of RECLAIM_BATCH_ROWS rows — never in the
        activation. Two derived, idempotent passes to a fixed point: (i) the rows of a HEAD revision lose the marks an
        outgoing revision left (a re-activation); (ii) the rows of a SUPERSEDED revision whose measurement key
        its successor no longer carries are marked with the successor (the same keys the runner's withdrawal records
        list — derived from the rows, so a 200k-key withdrawal costs the activation nothing); ``source_revision.reclaimed``
        records a finished (ii) and is reset when the chain moves. An interrupted cycle finishes on the next one."""
        total = 0
        pk = R.NATURAL_KEYS["item_result"]
        cols = ", ".join(pk)
        key = [c for c in pk if c != "source_revision_id"]
        same = " AND ".join(f"h.{c} = i.{c}" for c in key)
        while True:
            row = self.conn.execute(
                "SELECT i.source_revision_id FROM item_result i JOIN source_revision sr ON sr.revision_id = i.source_revision_id "
                "WHERE i.withdrawn_by_revision_id IS NOT NULL AND sr.accepted = 1 AND sr.superseded_by_revision_id IS NULL LIMIT 1").fetchone()
            if row is None:
                break
            n = [0]

            def unmark(rev=str(row[0])) -> None:
                cur = self.conn.execute(
                    f"UPDATE item_result SET withdrawn_by_revision_id = NULL WHERE ({cols}) IN "
                    f"(SELECT {cols} FROM item_result WHERE source_revision_id = ? AND withdrawn_by_revision_id IS NOT NULL LIMIT ?)",
                    (rev, RECLAIM_BATCH_ROWS))
                n[0] = cur.rowcount
            self.report.reclaim_transactions += 1
            self._timed_transaction(unmark, 0, RECLAIM_BATCH_ROWS, now, kind="reclaim")
            total += n[0]
            if n[0] <= 0:
                break
        while True:
            row = self.conn.execute(
                "SELECT revision_id, superseded_by_revision_id FROM source_revision WHERE accepted = 1 "
                "AND superseded_by_revision_id IS NOT NULL AND reclaimed = 0 ORDER BY observed_utc, revision_id LIMIT 1").fetchone()
            if row is None:
                break
            n = [0]

            def mark(rev=str(row[0]), to_rev=str(row[1])) -> None:
                cur = self.conn.execute(
                    f"UPDATE item_result SET withdrawn_by_revision_id = ? WHERE ({cols}) IN "
                    f"(SELECT {cols} FROM item_result i WHERE i.source_revision_id = ? AND i.withdrawn_by_revision_id IS NULL "
                    f" AND NOT EXISTS (SELECT 1 FROM item_result h WHERE {same} AND h.source_revision_id = ?) LIMIT ?)",
                    (to_rev, rev, to_rev, RECLAIM_BATCH_ROWS))
                n[0] = cur.rowcount
                if n[0] < RECLAIM_BATCH_ROWS:
                    self.conn.execute("UPDATE source_revision SET reclaimed = 1 WHERE revision_id = ?", (rev,))
            self.report.reclaim_transactions += 1
            self._timed_transaction(mark, 0, RECLAIM_BATCH_ROWS, now, kind="reclaim")
            total += n[0]
        if total:
            self.report.changed = True
        return total

    def _retirement_head_moved(self, rev: str, marker_valid: Sequence[VAL.Validated]) -> Optional[Tuple[VAL.Validated, Optional[str], str]]:
        """The retirement head gate (registry-hardening 3a), asked inside the activation transaction just before the
        marker folds: ``(marker record, live head revision id or None, detail)`` when ``rev`` is an accepted EMPTY revision -- a
        retirement -- whose :data:`RETIREMENT_EXPECTS_KEY` premise is not the live state (:func:`retirement_head_moved`: the head it was
        computed against is not the source's accepted head, or the live inventory holds a file input it does not name); None for any other
        marker, for a retirement that is the head re-observed, and for a retirement marker that carries no premise (published before the
        rule: judged as the fold of its day judged it -- the rule versions itself by the key's presence, so a replay reproduces the live
        heads). The hold is :data:`RETIREMENT_HEAD_KIND`; the re-attempt rule is the run-drop guard's (:meth:`_held_still_blocked`): the
        directory's bytes never change, so the same comparison trips again while the head it was compared against is still the head, and
        a marker a later accepted read superseded is never re-attempted; a retirement computed against today's head only ever arrives
        as a BUNDLE (the pull's re-activation of the same empty id against the next shard's claim, ``runner._reactivate_retirement``, or
        the grading host path's bundle of a known id), attempted in full -- never a no-op of this hold (:meth:`_recorded_hold` lists this kind
        nowhere), and its own premise judged here."""
        mrec = next((v for v in marker_valid if v.record_type == "source_revision"), None)
        if mrec is None or not int(mrec.payload.get("accepted") or 0):
            return None
        inputs_json_ = mrec.payload.get("inputs_json")
        if inputs_json_ not in ({}, "{}", None) or not isinstance(mrec.extra, Mapping):
            return None                # not a retirement (an empty revision): nothing to compare
        if RETIREMENT_EXPECTS_KEY not in mrec.extra:
            return None                # a retirement published before the rule: judged as the fold of its day judged it (rule versioning)
        source_id = str(mrec.payload.get("source_id") or "")
        moved = retirement_head_moved(self.conn, rev, source_id, mrec.extra.get(RETIREMENT_EXPECTS_KEY))
        if moved is None:
            return None
        head_rev, detail = moved
        return mrec, head_rev, detail

    def _undeclared_run_drop(self, rev: str, marker_valid: Sequence[VAL.Validated]) -> Optional[Tuple[VAL.Validated, str, str]]:
        """The run-set drop guard, asked inside the activation transaction just before the marker folds: ``(marker record,
        head revision id, detail)`` when ``rev`` would become the source's head and no longer names run ids the current head carries
        without declaring them; None otherwise. The head is the source's latest ACCEPTED revision by (observed_utc, revision_id); a
        marker that is that head re-observed (a re-read bundle), or a source with no accepted revision, compares against nothing.
        A held revision whose head has since moved is compared against the NEW head on its re-attempt (a new comparison); one a
        later accepted read has superseded is never re-attempted (:meth:`_held_still_blocked`), so the hold's provenance stands. The revision's
        run ids are read from the marker's ``observation`` records plus the ``observation`` rows it already holds (a re-activation
        bundle re-emits none: its membership is in the table); the head's from its ``observation`` rows; both the way
        :func:`run_ids_of_observation_keys` reads a key. The declaration is the marker record's parked ``declared_run_drops``
        (:func:`declared_run_drops_of`); a malformed one declares nothing and is named in the detail. The rule is a NET SHRINK: the
        revision names fewer run ids than the head and at least one of the head's is undeclared -- the 2026-09-23 bigcodebench / bfcl
        reads (10 -> 0, and 7 -> 3 / 8 -> 2 with every id replaced) all hold; a re-keying that replaces N ids with N others passes.

        VERSIONED (:data:`RUN_SET_DROP_UNDECLARED_SINCE`, :func:`run_set_drop_rule_applies`): the rule applies only to a marker whose
        ``observed_utc`` -- the field in hand here, the marker record's own payload (its ``written_utc`` stands in should the payload lack
        one) -- is at or after the stamp the live harvester began applying the rule (the base checkout's fast-forward past f4ad71fd7, not
        the commit's own date: one live acceptance lies between them). A replay then applies the rule in force when each revision was
        accepted, and reproduces the live heads (the R3 bootstrap held 51 revisions the live fold had accepted before the rule
        Route B). Not a live fold change: every pre-stamp marker was judged before the gate existed and is not re-judged; every live
        hold under the rule is post-stamp (checked read-only; ``scripts/registry/run_set_drop_versioning_check.py``)."""
        mrec = next((v for v in marker_valid if v.record_type == "source_revision"), None)
        if mrec is None or not int(mrec.payload.get("accepted") or 0):
            return None
        if not run_set_drop_rule_applies(str(mrec.payload.get("observed_utc") or mrec.written_utc)):
            return None                # observed before the rule landed: judged as the fold of its day judged it (rule versioning)
        source_id = str(mrec.payload.get("source_id") or "")
        head = self.conn.execute("SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 "
                                 "ORDER BY observed_utc DESC, revision_id DESC LIMIT 1", (source_id,)).fetchone()
        if head is None or str(head[0]) == rev:
            return None
        head_rev = str(head[0])
        prev_ids = run_ids_of_observation_keys(k for (k,) in self.conn.execute(
            "SELECT observation_key FROM observation WHERE source_id = ? AND revision_id = ?", (source_id, head_rev)))
        if not prev_ids:
            return None
        new_keys: List[object] = [k for v in marker_valid if v.record_type == "observation" for k in (v.payload.get("observation_keys") or [])]
        new_keys += [k for (k,) in self.conn.execute("SELECT observation_key FROM observation WHERE source_id = ? AND revision_id = ?", (source_id, rev))]
        new_ids = run_ids_of_observation_keys(new_keys)
        if not (prev_ids - new_ids) or len(new_ids) >= len(prev_ids):
            return None                # nothing dropped, or as many run ids as the head (a re-keying, a superset): not a shrink
        declaration, malformed = declared_run_drops_of(mrec.extra)
        cmp = run_set_drop_comparison(prev_ids, new_ids, declaration)
        if not cmp["tripped"]:
            return None
        named = list(cmp["runs_dropped_undeclared_sample"])
        more = int(cmp["runs_dropped_undeclared"]) - len(named)
        detail = (f"{RUN_SET_DROP_KEY}: {source_id} no longer names {cmp['runs_dropped_undeclared']} of the {cmp['runs_previous']} run ids of "
                  f"the accepted head {head_rev} and declares none of them; the run set shrank {cmp['runs_previous']} -> {cmp['runs_now']} "
                  f"({cmp['runs_added']} added"
                  + (f", {cmp['runs_dropped_declared']} dropped and declared" if cmp["runs_dropped_declared"] else "")
                  + f"); the head stays current. Dropped ({len(named)} of {cmp['runs_dropped_undeclared']} named): " + ", ".join(named)
                  + (f" (+{more} more)" if more > 0 else "") + (f"; {malformed}" if malformed else "")
                  + "; declare them under the adapter's runs_retired and re-read, or release with "
                  + f"`registry revision accept --revision-id {rev} --reason ...`")
        return mrec, head_rev, detail

    def _placeholder_revision(self, rev: str, marker_records, marker_uri: str, refusal: Optional[str] = None) -> None:
        """A ``source_revision`` row for ``rev`` with ``accepted = 0`` built from the marker record (the FK target of the
        revision's item rows and observations); the marker's own fold replaces it, or a held attempt keeps it with
        the reason in ``refusal``."""
        if self.conn.execute("SELECT 1 FROM source_revision WHERE revision_id = ?", (rev,)).fetchone():
            if refusal is not None:
                self.conn.execute("UPDATE source_revision SET accepted = 0, refusal = ? WHERE revision_id = ? AND accepted = 0", (refusal, rev))
            return
        payload = next((rec["payload"] for _, rec, _ in marker_records if rec.get("record_type") == "source_revision"), None)
        if not payload:
            return
        inputs = payload.get("inputs_json")
        row = {"revision_id": rev, "source_id": payload.get("source_id"), "adapter": payload.get("adapter"),
               "observed_utc": payload.get("observed_utc"),
               "inputs_json": canonical_json(inputs) if isinstance(inputs, (dict, list)) else (inputs if inputs is not None else "{}"),
               "accepted": 0, "refusal": refusal, "superseded_by_revision_id": None, "priority": int(payload.get("priority") or 0),
               "field_versions_json": "{}", "extra_json": "{}",
               "record_uri": marker_uri, "ingest_seq": self.ranker.seq_of(marker_uri) or 0}
        cols = list(row)
        self.conn.execute(DB.upsert_sql("source_revision", cols, ("revision_id",)), tuple(row[c] for c in cols))

    def _apply_constituent(self, v: VAL.Validated, seq: int, file_uri: str, ex: _Existence, now: str,
                           counts: Dict[str, int], types: Dict[str, int], hold_blocked: bool = True) -> str:
        """One run-scale constituent of a revision: refused → the revision is held; blocked → held, or (fixed-point
        pass) returned to the caller to retry after its siblings."""
        item_batches: Dict[str, List[Tuple]] = {}
        item_revisions: Set[str] = set()
        st = self.apply_validated(v, seq, file_uri, R.record_file_id_for_uri(file_uri), item_batches, item_revisions, ex, now)
        if st == "blocked":
            if not hold_blocked:
                return st
            row = self.conn.execute("SELECT missing_parent_table, missing_parent_id FROM blocked_records WHERE record_sha256 = ?", (v.sha,)).fetchone()
            raise _RevisionHeld("blocked", str(row[0]), str(row[1]), file_uri, v.sha, v.uri, v.record_type)
        if st == "refused":
            reason = self.conn.execute("SELECT reason FROM record_refusal WHERE record_sha256 = ?", (v.sha,)).fetchone()
            raise _RevisionHeld("refused", "record", v.uri, file_uri, v.sha, str(reason[0]) if reason else "refused", v.record_type)
        if item_batches:
            self.flush_items(item_batches, item_revisions)
        counts[st] += 1
        types[v.record_type] = types.get(v.record_type, 0) + 1
        return st

    @staticmethod
    def _check_part(part: ParsedFile) -> None:
        if part.error:
            raise _RevisionHeld("refused", "record_file", part.uri, part.uri, part.file_sha or "", part.error)
        if part.retry:
            raise _RevisionHeld("retry", "record_file", part.uri, part.uri, part.file_sha or "", "changed while being read")

    def _find_marker(self, files: Sequence[Tuple[str, Optional[str]]], rev: str):
        """The revision directory's marker file — the one holding the ``source_revision`` record for ``rev`` — and its
        records (small: the marker file carries the revision's membership, never its measurements)."""
        for path, _ in files:
            if "source_revision" not in os.path.basename(path):
                continue
            try:
                recs = list(R.read_record_file(path))
            except (ValueError, OSError, EOFError, gzip.BadGzipFile):
                continue
            if any(rec.get("record_type") == "source_revision" and rec.get("payload", {}).get("revision_id") == rev for _, rec, _ in recs):
                return path, recs
        return None, []

    def _file_summary(self, path: str) -> ParsedFile:
        """What a delivery / ledger row needs about one file, from a streamed read: verified sha, stat, envelope, the
        whole record count and THE file's first key — the smallest order key over every well-formed record of every
        window (items included), never the last window's alone and never the clock (live == replay)."""
        last = None
        first: Optional[Tuple[str, str]] = None
        for part in stream_parts(self.root, path, None, ITEM_ROWS_PER_TRANSACTION):
            k = part.first_key
            if k is not None and (first is None or k < first):
                first = k
            last = part
        assert last is not None
        last.whole_first_key = first
        return last

    @staticmethod
    def _file_record_type(uri: str) -> str:
        m = R.RECORD_FILE_RE.match(os.path.basename(uri))
        rt = m.group("type") if m else None
        return rt if rt in R.RECORD_TYPES else "source_revision"

    # -- blocked queue ------------------------------------------------------------------------
    def reevaluate_blocked(self, now: Optional[str] = None, stop=None) -> int:
        """Re-try every blocked record until no further progress; returns the unblocked count. Candidates are
        PARTITIONED before they are materialised: revision directories tagged in ``blocked_records`` are
        retried whole by :meth:`apply_revision`; other blocked records are re-read file by file in bounded windows,
        one transaction per window, to a fixed point. ``stop`` (the cycle's wall-clock budget) is asked before
        every revision re-attempt: once it answers True the remaining re-attempts wait for the next cycle (each attempt
        commits on its own; the queue is the resume point) and the report says ``stopped = "budget"``."""
        now = now or R.utc_now()
        total = 0
        out_of_budget = False
        if self.conn.execute("SELECT 1 FROM source_revision WHERE accepted = 0 AND superseded_by_revision_id IS NULL LIMIT 1").fetchone():
            with DB.transaction(self.conn):
                self._supersede_unaccepted_markers()
        while True:
            rows = self.conn.execute("SELECT record_sha256, record_uri, extra_json, last_checked_utc, record_type, missing_parent_table, missing_parent_id "
                                     "FROM blocked_records ORDER BY record_uri").fetchall()
            if not rows:
                break
            progress = 0
            revisions: Dict[str, str] = {}
            exact: Set[str] = set()      # directories whose refused hold is EXACT this cycle: the stand-down rules see them, the attempt is not made
            memos: Dict[str, Tuple[Mapping[str, object], Optional[str], Optional[str], Optional[str]]] = {}   # the directory's own marker row: (extra, sha, parent table, parent id)
            by_file: Dict[str, Set[str]] = {}
            for sha, uri, extra, checked, rtype, ptable, pid in rows:
                ext = json.loads(extra or "{}")
                tag = ext.get("revision_id")
                rev = revision_of_uri(uri)
                if tag or (rev and "#" not in uri):
                    d = bundle_of_uri(uri).rstrip("/")
                    if rtype == "source_revision" and not ext.get("no_op") and ext.get("kind"):
                        memos[d] = (ext, str(sha), ptable, pid)
                    if ext.get("kind") == "refused" and not ext.get("no_op") and (checked == now or self._refused_hold_exact(ext)):
                        # an EXACT refused hold (2026-09-30): its content refused under THIS code -- this cycle, or an earlier one with the
                        # refusing file's bytes and the judging code unchanged since -- so re-reading the same bytes would only refuse them
                        # again (before this rule a refused kind re-attempted every cycle: 14 held directories re-refused the same records
                        # for ~40 s of every live cycle from 2026-09-20 to 09-30, re-stamping record_refusal and appending to refused.jsonl
                        # each time). It stays in `revisions` so the stand-down rules below still see it, and is skipped at the attempt.
                        # A no-op marker copied that kind from an earlier attempt and its own bytes were never read: _held_still_blocked's
                        exact.add(d)
                    revisions[d] = str(tag or rev)
                    continue
                if "#" in uri:
                    by_file.setdefault(uri.rpartition("#")[0], set()).add(sha)
            self._stand_down_replaced_dirs(revisions)
            for f, want in by_file.items():
                progress += self._reevaluate_file(f, want, now)
            # a revision's directory before its bundles, whatever the revision id's first character sorts against `_activations`
            # (2026-09-29): a bundle attempted after its directory activated is compared with the head (_same_measurements) and, identical,
            # folds as duplicates; attempted first it would have carried the rows itself -- the same order on both paths, so a replay
            # reaches the live database's record_file_id on every item row
            activated: Set[str] = set()          # revisions whose carrier activated in THIS pass: their no-op bundles' facts fold now
            for d in sorted(revisions, key=_revision_then_bundles_key):
                rev_dir = os.path.join(self.root, d)
                if not os.path.isdir(rev_dir):
                    with DB.transaction(self.conn):
                        self._discard_staging(revisions[d])          # its directory is gone: nothing left to activate
                    self._forget_held_attempt(d)
                    continue
                if d in exact:
                    # the hold stands another cycle: the attempt bookkeeping advances as a skipped blocked hold's does, the held count keeps
                    # its meaning (once per cycle; the cycle that recorded the hold counted it at the attempt), no bundle line, no refusal
                    if self._hold_bookkeeping(d, now):
                        self.report.revisions_held += 1
                    continue
                if d in memos and self._hold_memo_stands(d, rev_dir, memos[d], now):
                    # nothing about this held directory moved since the attempt that recorded its identity --
                    # the same files, marker, judging code and parent state, inside the sweep interval -- so re-staging and rolling it back again
                    # would only reproduce the hold (measured: 435-856 held bundles re-attempted every cycle, a ~330-s floor under the write lock
                    # that timed the graders' claims out). Skipped like an exact hold; re-attempted when the identity moves, after held_sweep_s,
                    # or under --retry-held (the window's uncapped cycles)
                    if self._hold_bookkeeping(d, now):
                        self.report.revisions_held += 1
                    self.report.revisions_held_skipped += 1
                    continue
                if self._held_still_blocked(d, now):
                    self.report.revisions_held += 1
                    continue
                if stop is not None and stop() and not (revisions[d] in activated and self._is_noop_bundle(d)):
                    # the budget is spent: this re-attempt and the rest wait for the next cycle's re-evaluation (2026-09-29). A carrier
                    # that activated in this pass and the no-op bundles whose facts fold right behind it are ONE unit of work, never
                    # parted by the budget: their fold is a few rows each, and a cycle's end must be a state a replay reproduces
                    out_of_budget = True
                    self.report.stopped = "budget"
                    self.report.revisions_deferred_budget += 1
                    continue
                before = self.report.ok
                if self.apply_revision(rev_dir, now, retry=True) == "ok":
                    progress += 1
                    activated.add(revisions[d])
                    self._forget_held_attempt(d)
                    # the revision's tagged rows are gone with the activation (a marker row unblocks by sha) -- EXCEPT the markers of the
                    # revision's no-op bundles, whose marker and status facts still fold in the retries behind this one: each leaves the
                    # queue when it is processed (the second DELETE, under its own directory), so a cycle killed between the carrier's
                    # activation and those retries leaves the rows standing for the next cycle to take (2026-09-29)
                    self.conn.execute("DELETE FROM blocked_records WHERE json_extract(extra_json, '$.revision_id') = ? "
                                      "AND json_extract(extra_json, '$.no_op') IS NULL", (revisions[d],))
                    self.conn.execute("DELETE FROM blocked_records WHERE record_uri LIKE ? AND record_uri NOT LIKE '%#%'", (d + "/%",))
                    self.report.blocked = max(0, self.report.blocked - 1)
                    self.report.unblocked += max(0, self.report.ok - before)
            with DB.transaction(self.conn):
                self.conn.execute("UPDATE blocked_records SET stale = 1 WHERE stale = 0 AND "
                                  "(julianday(?) - julianday(first_seen_utc)) * 86400 > ?", (now, BLOCKED_STALE_S))
                self._mark_superseded_blocked()
            total += progress
            if progress == 0 or out_of_budget:
                break
        if total:
            self.report.changed = True
        return total

    def _stand_down_replaced_dirs(self, revisions: Dict[str, str]) -> None:
        """A held directory a later bundle of the same revision has replaced stands down: its rows leave the queue and it is
        not re-attempted. The runner publishes a re-read of a never-accepted directory as an ``_activations/<rev>/<stamp>/``
        bundle carrying the read's records (same natural keys, new shas, the refused record corrected); the original
        directory's bytes are what refused and never change, yet its marker row (``kind = refused``) re-attempted every
        cycle, refused again on the same line and rewrote the shared ``source_revision.refusal`` with that stale reason —
        while the bundle, held on a parent of its own, was skipped by ``_held_still_blocked`` and never refreshed it
        (strength_mtdiff, 2026-09-19: eight sources read ``activation pending: record .../000002-run.jsonl#2`` for a
        revision whose bundles were waiting on a run's judgment). Stands down: the original directory when it is held on a
        refused constituent and a bundle of its revision is queued; every bundle but the latest; and every directory of a
        revision that is already accepted (a bundle activated in this cycle's fold and left the others' rows behind) -- except
        the latest bundle of an accepted revision held on the head's run set or on a moved head (a retirement's
        re-activation, :data:`RETIREMENT_HEAD_KIND`), which is pending work with a reason and stands. The
        shared marker row then names the latest bundle's own hold. Derived from the tables, so live and rebuild reach the
        same fixed point."""
        by_rev: Dict[str, List[str]] = {}
        for d, rev in revisions.items():
            by_rev.setdefault(rev, []).append(d)
        for rev, dirs in by_rev.items():
            bundles = sorted(d for d in dirs if "/_activations/" in d + "/")
            latest = bundles[-1] if bundles else None
            accepted = not self._parent_still_absent("source_revision", rev)
            for d in sorted(dirs):
                under = d.rstrip("/") + "/"
                if d in bundles and self._is_noop_bundle(d):
                    # (2026-09-29) a no-op bundle stands, held or accepted: while the revision is held, an exact hold that costs nothing;
                    # once the carrier activated, the way its marker, status and artifact facts fold -- in stamp order with the revision's
                    # other bundles, as bundles of the head -- and its rows leave the queue only when it is processed
                    continue
                if d == latest:
                    if not accepted:
                        continue
                    if not self._revision_is_head(rev) and self.conn.execute(
                            "SELECT 1 FROM blocked_records WHERE record_type = 'source_revision' AND substr(record_uri, 1, ?) = ? "
                            "AND json_extract(extra_json, '$.kind') IN (?, ?)", (len(under), under, RUN_SET_DROP_KIND, RETIREMENT_HEAD_KIND)).fetchone():
                        # the latest bundle of an ACCEPTED, NON-HEAD revision held on the head's run set or, a retirement, on a head
                        # that moved under it (RETIREMENT_HEAD_KIND; registry-hardening 3a): a re-activation of a non-head
                        # revision that never folded -- pending work with a reason, not the leftovers of an activation; it stands (and the
                        # incremental pass, reading its queue row, re-emits no copy; the reads protocol reads it as the standing hold that
                        # keeps the adapter unread, `retirements_held_for_moved_head`) until a later bundle or the head replaces it -- a
                        # revision that IS the head again (a later bundle folded) has nothing pending, and its older bundles' rows are the
                        # leftovers this stand-down clears. Before round 3 a held re-activation of a retirement accepted once left the queue
                        # here, in the cycle that held it, and no record of the hold survived but the cycle's counter
                        continue
                if not accepted:
                    if not bundles:
                        continue
                    if d not in bundles:
                        if self.conn.execute(
                                "SELECT 1 FROM blocked_records WHERE record_type = 'source_revision' AND substr(record_uri, 1, ?) = ? "
                                "AND json_extract(extra_json, '$.kind') = 'refused'", (len(under), under)).fetchone() is None:
                            continue      # the original directory held on a PARENT re-attempts as before (its parent may land)
                        if self.conn.execute(
                                "SELECT 1 FROM blocked_records WHERE record_type = 'source_revision' AND json_extract(extra_json, '$.revision_id') = ? "
                                "AND record_uri LIKE '%/_activations/%' AND json_extract(extra_json, '$.no_op') IS NULL", (rev,)).fetchone() is None:
                            # (2026-09-29) every queued bundle is a no-op -- the directory's own content under fresh stamps: the directory
                            # stays the carrier and re-attempts on their behalf (a refused kind re-attempts when its file or the judging
                            # code changes, _refused_hold_exact); it stands down for a bundle that was ATTEMPTED, whose content differs
                            # (the corrected read)
                            continue
                with DB.transaction(self.conn):
                    self.conn.execute("DELETE FROM blocked_records WHERE substr(record_uri, 1, ?) = ?", (len(under), under))
                    if not accepted and latest is not None:
                        lunder = latest.rstrip("/") + "/"
                        brow = self.conn.execute(
                            "SELECT missing_parent_table, missing_parent_id, json_extract(extra_json, '$.blocked_on') FROM blocked_records "
                            "WHERE record_type = 'source_revision' AND substr(record_uri, 1, ?) = ?", (len(lunder), lunder)).fetchone()
                        if brow is not None:
                            reason = f"activation pending: {brow[0]} {brow[1]} ({brow[2]})"
                            self.conn.execute("UPDATE source_revision SET refusal = ? WHERE revision_id = ? AND accepted = 0", (reason, rev))
                del revisions[d]
                self.report.changed = True

    #: The accepted revision that moved a source past an UNACCEPTED marker (a refused first read, a held revision): the
    #: earliest accepted revision of the same source observed after it.
    _SUPERSEDING_MARKER_SQL = (
        "SELECT a.revision_id FROM source_revision a WHERE a.source_id = source_revision.source_id AND a.accepted = 1 "
        "AND (a.observed_utc > source_revision.observed_utc "
        "OR (a.observed_utc = source_revision.observed_utc AND a.revision_id > source_revision.revision_id)) "
        "ORDER BY a.observed_utc, a.revision_id LIMIT 1")

    def _supersede_unaccepted_markers(self) -> int:
        """Stamp ``superseded_by_revision_id`` on every unaccepted ``source_revision`` row of a source whose later revision
        was accepted. The marker fold links accepted chains only (the outgoing head takes its successor's id), so a refused
        first read stayed an open refusal forever once its inputs changed and the next read was accepted under a new revision
        id (regrade_twin, 2026-09-17: 269 refused markers still read as the sources' state a day after 85 of those sources had
        an accepted read). Derived from the table on every re-evaluation (idempotent; a rebuild's closing re-evaluation
        stamps the same rows), never from a clock; ``v_source_status`` reads a source's state from its latest revision that
        is not a superseded refusal."""
        # a read first: the write (and its transaction) happens only when a marker is there to stamp, so a cycle with
        # nothing to supersede writes nothing
        if self.conn.execute(f"SELECT 1 FROM source_revision WHERE accepted = 0 AND superseded_by_revision_id IS NULL "
                             f"AND EXISTS ({self._SUPERSEDING_MARKER_SQL}) LIMIT 1").fetchone() is None:
            return 0
        cur = self.conn.execute(
            f"UPDATE source_revision SET superseded_by_revision_id = ({self._SUPERSEDING_MARKER_SQL}) "
            f"WHERE accepted = 0 AND superseded_by_revision_id IS NULL AND EXISTS ({self._SUPERSEDING_MARKER_SQL})")
        n = max(0, cur.rowcount)
        if n:
            self.report.changed = True
        return n

    def _is_noop_bundle(self, bundle_dir: str) -> bool:
        """Whether the queue holds a NO-OP marker row for ``bundle_dir`` (:meth:`_noop_held_bundle`)."""
        under = bundle_dir.rstrip("/") + "/"
        return self.conn.execute("SELECT 1 FROM blocked_records WHERE record_type = 'source_revision' AND substr(record_uri, 1, ?) = ? "
                                 "AND json_extract(extra_json, '$.no_op') IS NOT NULL", (len(under), under)).fetchone() is not None

    def _held_still_blocked(self, bundle_dir: str, now: str) -> bool:
        """A held revision directory whose recorded blocker has not arrived cannot activate: skip the whole re-attempt
        (F16). The held path records, on the marker's blocked row, the parent the ROOT blocked constituent (after the
        fixed-point pass: one whose missing parent is not a constituent of the revision itself) was missing. Parents never
        leave a database, so on a re-attempt every constituent that applied before applies again and that constituent —
        still missing the same external parent — holds the revision on the same reason; the attempt would roll back to exactly this state. When that parent has no row under
        any column a dependency check looks it up by (a revision: no ACCEPTED row), the attempt is skipped and only the
        directory's tagged rows' attempt bookkeeping advances, as a held attempt's would. Keyed by the DIRECTORY (its
        root-relative path): a revision id is shared by its own directory and every re-activation bundle of it, each with
        a marker row of its own. A refused constituent (``kind = refused``) is :meth:`reevaluate_blocked`'s own rule
        (:meth:`_refused_hold_exact`: skipped while the refusing file's bytes and the judging code stand, re-attempted once when
        either moves); a marker still absent (no ``kind``), a pseudo-parent (``run_state_event`` evidence) and a parent that has
        since arrived all re-attempt as before."""
        under = bundle_dir.rstrip("/") + "/"
        noop = self.conn.execute("SELECT json_extract(extra_json, '$.revision_id') FROM blocked_records WHERE record_type = 'source_revision' "
                                 "AND substr(record_uri, 1, ?) = ? AND json_extract(extra_json, '$.no_op') IS NOT NULL", (len(under), under)).fetchone()
        if noop is not None:
            if not self._parent_still_absent("source_revision", str(noop[0] or "")):
                return False            # the carrier activated: this is a bundle of the head now, and its marker and status facts fold
            # a NO-OP bundle's marker (2026-09-29): its content repeats the attempted carrier's, record for record, so the carrier's
            # re-attempt is this bundle's too -- it is skipped whatever its kind (the carrier, a refused kind included, re-attempts as
            # the rules below say), and folds its facts, as a bundle of the head, once the carrier activates
            self.conn.execute(
                "UPDATE blocked_records SET attempts = attempts + CASE WHEN last_checked_utc = ? THEN 0 ELSE 1 END, last_checked_utc = ?, "
                "stale = CASE WHEN (julianday(?) - julianday(first_seen_utc)) * 86400 > ? THEN 1 ELSE stale END "
                "WHERE substr(record_uri, 1, ?) = ?", (now, now, now, BLOCKED_STALE_S, len(under), under))
            return True
        rows = self.conn.execute(
            "SELECT missing_parent_table, missing_parent_id, json_extract(extra_json, '$.kind'), json_extract(extra_json, '$.revision_id') "
            "FROM blocked_records WHERE record_type = 'source_revision' AND substr(record_uri, 1, ?) = ? "
            "AND json_extract(extra_json, '$.kind') IN ('blocked', ?, ?)", (len(under), under, RUN_SET_DROP_KIND, RETIREMENT_HEAD_KIND)).fetchall()
        if len(rows) != 1:
            return False
        ptable, pid, kind, rev = str(rows[0][0]), str(rows[0][1]), rows[0][2], str(rows[0][3] or "")
        if kind in (RUN_SET_DROP_KIND, RETIREMENT_HEAD_KIND):
            # held for dropping the head's run ids undeclared, or a retirement held because the head it was computed against is
            # not the live one (RETIREMENT_HEAD_KIND, registry-hardening 3a round 2): the directory's bytes never change and a declaration
            # -- or a retirement computed against today's head -- only ever arrives as a bundle (its own directory), so the re-attempt is
            # exact -- the same hold -- while the head it was compared against is still the source's head. A held marker a LATER accepted
            # read has superseded is not re-attempted either: it can never become the head, and activating it "already superseded" (what
            # a blocked-parent hold does when its parent lands) would fold its marker and erase the hold's provenance -- the refusal and
            # the queue row are the record of what the reader did (2026-09-23). It is released only by a bundle (`registry revision
            # accept --force` for a run drop; the pull's re-activation against the next shard's claim for a retirement).
            head_same = self.conn.execute("SELECT 1 FROM source_revision WHERE revision_id = ? AND accepted = 1 AND superseded_by_revision_id IS NULL",
                                          (pid,)).fetchone()
            superseded = self.conn.execute("SELECT 1 FROM source_revision WHERE revision_id = ? AND accepted = 0 AND superseded_by_revision_id IS NOT NULL",
                                           (rev,)).fetchone()
            if head_same is None and superseded is None:
                return False
        elif not self._parent_still_absent(ptable, pid):
            return False
        self.conn.execute(
            "UPDATE blocked_records SET attempts = attempts + CASE WHEN last_checked_utc = ? THEN 0 ELSE 1 END, last_checked_utc = ?, "
            "stale = CASE WHEN (julianday(?) - julianday(first_seen_utc)) * 86400 > ? THEN 1 ELSE stale END "
            "WHERE substr(record_uri, 1, ?) = ?", (now, now, now, BLOCKED_STALE_S, len(under), under))
        return True

    def _refused_hold_exact(self, ext: Mapping[str, object]) -> bool:
        """:func:`refused_hold_exact` with this cycle's listing and ``retry_refused`` flag (the rule itself is module-level so the claim
        probe, ``pending.pending_identity_corrections``, applies the same one)."""
        return refused_hold_exact(self.conn, ext, listed_uris=self.listed_uris, retry_refused=self.retry_refused)

    def _hold_identity(self, rev_dir: str, marker_sha: Optional[str], kind: Optional[str], parent_table: Optional[str], parent_id: Optional[str],
                       marker_extra: Optional[Mapping[str, object]] = None) -> str:
        """The attempt identity of a held revision directory (registry follow-ups-4 item 10): sha256 over the directory's sorted listing
        -- every file's root-relative path, size and mtime -- the marker's sha, the hold's own record on the queue row (its kind, reason and
        the refused kind's exact-hold key: a row stripped of its key re-attempts once, as before), the judging code's stamp
        (:func:`validator_stamp`: a landed fix moves it) and, for a blocked kind, whether the parent it waits for is present (its arrival
        moves the identity, so the hold activates on the next cycle as before). Pure in the directory, the row and the tables."""
        return hold_identity(self.root, rev_dir, marker_sha, kind, parent_table, parent_id,
                             parent_present=(None if kind == "refused" or not parent_table else not self._parent_still_absent(str(parent_table), str(parent_id or ""))),
                             marker_extra=marker_extra)

    def _held_attempts(self) -> Dict[str, Dict[str, str]]:
        """The attempt memo (``state/held_attempts.json``: root-relative directory -> {attempt_identity, attempted_utc}), read once per cycle."""
        if self._held_attempts_cache is None:
            self._held_attempts_cache = read_held_attempts(self.root)
        return self._held_attempts_cache

    def _note_held_attempt(self, rev_dir: str, identity: str, now: str) -> None:
        memo = self._held_attempts()
        memo[os.path.relpath(rev_dir, self.root).rstrip("/")] = {"attempt_identity": identity, "attempted_utc": now}
        write_held_attempts(self.root, memo)

    def _forget_held_attempt(self, bundle_dir: str) -> None:
        memo = self._held_attempts()
        if memo.pop(bundle_dir.rstrip("/"), None) is not None:
            write_held_attempts(self.root, memo)

    def _hold_memo_stands(self, bundle_dir: str, rev_dir: str, memo: Tuple[Mapping[str, object], Optional[str], Optional[str], Optional[str]], now: str) -> bool:
        """Whether a held directory's recorded attempt identity STANDS -- the same listing, marker, code and parent state as the attempt that recorded
        it (``state/held_attempts.json``), recorded less than ``held_sweep_s`` ago -- so the re-attempt is skipped this cycle (bookkeeping advances,
        the held count keeps its meaning, ``revisions_held_skipped`` counts it). False under ``retry_held`` / ``retry_refused``, for a directory with
        no memo (a hold recorded before the rule: re-attempted once, which records it), when anything moved, or past the sweep interval."""
        if self.retry_held or self.retry_refused:
            return False
        ext, marker_sha, ptable, pid = memo
        rec = self._held_attempts().get(bundle_dir.rstrip("/")) or {}
        ident, at = rec.get("attempt_identity"), rec.get("attempted_utc")
        if not ident or not at:
            return False
        try:
            age_s = (R.utc_to_epoch_ms(now) - R.utc_to_epoch_ms(str(at))) / 1000.0
        except (TypeError, ValueError):
            return False
        if age_s >= self.held_sweep_s:
            return False
        return self._hold_identity(rev_dir, marker_sha, str(ext.get("kind") or "") or None, ptable, pid, ext) == ident

    def _hold_bookkeeping(self, bundle_dir: str, now: str) -> bool:
        """Advance the attempt bookkeeping of every queue row under ``bundle_dir`` for a re-attempt NOT made because the hold is exact:
        ``attempts`` + 1 and ``last_checked_utc`` = now when the rows were last checked in an EARLIER cycle (the hold stood another
        cycle; returns True), nothing when they were checked this cycle (the attempt that recorded the hold already counted; False);
        ``stale`` past BLOCKED_STALE_S either way -- the statement :meth:`_held_still_blocked` runs for a blocked hold whose parent is
        still absent. Attempts, last_checked_utc and stale are outside rebuild equality by design."""
        under = bundle_dir.rstrip("/") + "/"
        earlier = self.conn.execute("SELECT 1 FROM blocked_records WHERE substr(record_uri, 1, ?) = ? AND last_checked_utc <> ? LIMIT 1",
                                    (len(under), under, now)).fetchone() is not None
        self.conn.execute(
            "UPDATE blocked_records SET attempts = attempts + CASE WHEN last_checked_utc = ? THEN 0 ELSE 1 END, last_checked_utc = ?, "
            "stale = CASE WHEN (julianday(?) - julianday(first_seen_utc)) * 86400 > ? THEN 1 ELSE stale END "
            "WHERE substr(record_uri, 1, ?) = ?", (now, now, now, BLOCKED_STALE_S, len(under), under))
        return earlier

    def _parent_still_absent(self, ptable: str, pid: str) -> bool:
        """:func:`parent_absent` on this connection (module-level so the claim probe applies the queue's own re-attempt condition)."""
        return parent_absent(self.conn, ptable, pid)

    #: The accepted revision that moved a source past a PENDING one: the pending revision's blocked records are tagged with
    #: its id; a later (observed_utc, revision_id) accepted revision of the same source means the pending one can never
    #: become the head — if its dependency ever lands it activates already superseded, changing what no reader sees.
    #: The EARLIEST such revision (the one that moved the source on; the rule ``_SUPERSEDING_MARKER_SQL`` stamps on the marker row),
    #: never the latest: the latest depends on WHEN the stamp is written -- live at the first later head, on replay at the closing
    #: re-evaluation after every later head -- so a pending marker that outlived two heads read differently on the two (found by the
    #: run-set drop guard's held-and-superseded markers, 2026-09-23).
    _SUPERSEDING_ACCEPTED_SQL = (
        "SELECT a.revision_id FROM source_revision p JOIN source_revision a ON a.source_id = p.source_id AND a.accepted = 1 "
        "WHERE p.revision_id = json_extract(blocked_records.extra_json, '$.revision_id') AND p.accepted = 0 "
        "AND (a.observed_utc > p.observed_utc OR (a.observed_utc = p.observed_utc AND a.revision_id > p.revision_id)) "
        "ORDER BY a.observed_utc, a.revision_id LIMIT 1")

    def _mark_superseded_blocked(self) -> int:
        """Mark the blocked records of every superseded pending revision ``stale``, ``extra_json.superseded_by``
        naming the accepted revision that moved the source on — so the blocked queue and the status page stop reporting them
        as pending work. Derived from the tables on every re-evaluation (idempotent; a re-block that reset the flag from the
        record's age is marked again in the same cycle), never from a clock. The records are NOT skipped on retry: should
        the dependency land, the revision activates already superseded, exactly as a replay of the log would have it."""
        cur = self.conn.execute(
            f"UPDATE blocked_records SET stale = 1, extra_json = json_set(extra_json, '$.superseded_by', ({self._SUPERSEDING_ACCEPTED_SQL})) "
            f"WHERE (stale = 0 OR json_extract(extra_json, '$.superseded_by') IS NOT ({self._SUPERSEDING_ACCEPTED_SQL})) "
            f"AND EXISTS ({self._SUPERSEDING_ACCEPTED_SQL})")
        return max(0, cur.rowcount)

    # -- dependency refusals of ledger-fact records -----------------------------------------------------
    def retry_dependency_refusals(self, now: str) -> Tuple[int, int]:
        """Re-select every ledger-fact record refused for a parent the registry did not hold (``dependency_refusal_reason``): its parent is
        looked up again; where the parent now exists the record is re-read from its (immutable) file and applied -- accepted, its
        `record_refusal` row withdrawn (the judgment is replaced by acceptance; the refusal FACT in the log stands as history) -- and where
        it is still absent the refusal stands untouched (no re-judgment, no second fact). Ordinary files are otherwise re-selected only when
        their bytes change, so without this pass a refused child was never re-judged when its parent folded later. Runs after the late
        phase on both paths. Returns ``(retried, resolved)``."""
        rows = self.conn.execute("SELECT record_sha256, record_uri, reason FROM record_refusal WHERE reason LIKE 'dependency: %' "
                                 "ORDER BY record_uri").fetchall()
        by_file: Dict[str, Set[str]] = {}
        retried = 0
        for sha, uri, reason in rows:
            parsed = dependency_refusal_parent(str(reason))
            if parsed is None or "#" not in str(uri):
                continue
            ptable, pid = parsed
            if ptable == FILING_PARENT:
                # a child of a filing whose header had not folded (records.LEDGER_FILING_CHILDREN): the header is looked up in the record
                # index -- indexed means folded, whatever its outcome -- and the re-apply judges the child against the header's standing
                # (accepted under the kept filing or a duplicate of it, refused by name under a refused one)
                retried += 1
                if self.conn.execute(f"SELECT 1 FROM {R.LEDGER_ORDERED_RECORD_INDEX} WHERE record_sha256 = ? LIMIT 1", (pid,)).fetchone() is not None:
                    by_file.setdefault(str(uri).rpartition("#")[0], set()).add(str(sha))
                continue
            if ptable not in S.TABLE_BY_NAME:
                continue
            retried += 1
            pk = R.NATURAL_KEYS[ptable][0]
            if self.conn.execute(f"SELECT 1 FROM {ptable} WHERE {pk} = ? LIMIT 1", (pid,)).fetchone() is not None:
                by_file.setdefault(str(uri).rpartition("#")[0], set()).add(str(sha))
        resolved = 0
        for file_uri, want in by_file.items():
            resolved += self._retry_refused_file(file_uri, want, now)
        self.report.dependency_refusals_retried += retried
        self.report.dependency_refusals_resolved += resolved
        return retried, resolved

    def _retry_refused_file(self, file_uri: str, want: Set[str], now: str) -> int:
        """Re-read one ranked file and apply the wanted (previously refused) records, one transaction per window; a record that applies
        `ok` or `duplicate` has its `record_refusal` row withdrawn and the file's delivery audit moved for THAT record (:meth:`_move_delivery`:
        `n_refused` -> `n_ok` or `n_duplicate` as the apply judged it, the status derived again by the fold's rule -- `ok` once nothing of
        the file is refused or blocked, `partial` while something still is, a wholly refused file with one record now accepted included,
        as a rebuild writes it). One move per accepted record, so a file spanning several parts is never moved by a cumulative count (the
        fifth closing pass, 2026-10-09: a per-part UPDATE subtracted the cumulative count after every part, so a file with one acceptance in
        each of two parts had its second part subtract the first acceptance twice -- `n_refused` 0 and status `ok` while a refusal of the
        file still stood; the eighth closing pass moved the per-part UPDATE into the per-record mover, whose status rule the fold shares).
        Returns the records accepted."""
        path = resolve_uri(self.root, file_uri)
        if path is None:
            return 0
        seq = self.ranker.seq_of(file_uri)
        if seq is None:
            return 0                                           # never ranked (a wholly refused file): nothing to re-read against
        accepted = 0
        for part in stream_parts(self.root, path, None, ITEM_ROWS_PER_TRANSACTION):
            if part.error or part.retry:
                continue
            cands = [pr.result for pr in part.records if pr.sha in want and isinstance(pr.result, VAL.Validated)]
            if not cands:
                continue
            cands.sort(key=lambda v: v.order_key)
            with DB.transaction(self.conn):
                self.ranker.reload()
                ex = _Existence(self.conn)
                self._blocked = {r[0] for r in self.conn.execute("SELECT record_sha256 FROM blocked_records").fetchall()}
                self._indexed = set()
                item_batches: Dict[str, List[Tuple]] = {}
                item_revisions: Set[str] = set()
                part_accepted = 0
                try:
                    for v in cands:
                        st = self.apply_validated(v, seq, file_uri, R.record_file_id_for_uri(file_uri), item_batches, item_revisions, ex, now)
                        if st in ("ok", "duplicate"):
                            self.conn.execute("DELETE FROM record_refusal WHERE record_sha256 = ?", (v.sha,))
                            if v.table is not None and v.table in R.LEDGER_ORDERED_TABLES:
                                self._move_record_delivery(v.table, v.sha, "refused", st, file_uri)   # this record alone, in every file that carried it
                            else:
                                self._move_delivery(file_uri, "refused", st)       # this record alone: the audit follows each verdict once
                            part_accepted += 1
                    accepted += part_accepted
                    self.flush_items(item_batches, item_revisions)
                    self.ranker.finalize()
                finally:
                    self._blocked = None
                    self._indexed = set()
        return accepted

    def _reevaluate_file(self, file_uri: str, want: Set[str], now: str) -> int:
        """Re-read one blocked file in bounded windows; every window of wanted records is its own transaction."""
        path = resolve_uri(self.root, file_uri)
        if path is None:
            return 0
        progress = 0
        seq = self.ranker.seq_of(file_uri)
        if seq is None:
            # every blocked record's file was ranked when it was parsed; a hole here is a corrupted ledger
            raise RuntimeError(f"blocked records name a file with no record_ingest row: {file_uri}")
        for part in stream_parts(self.root, path, None, ITEM_ROWS_PER_TRANSACTION):
            if part.error or part.retry:
                continue
            cands = [pr.result for pr in part.records if pr.sha in want and isinstance(pr.result, VAL.Validated)]
            for pr in part.records:
                if pr.sha in want and isinstance(pr.result, VAL.Malformed):
                    self._refuse(pr.sha, pr.result.path, pr.result.record_type, pr.result.field, pr.result.reason, now)
            items = [pi for pi in part.items if pi.sha in want]
            if not cands and not items:
                continue
            cands.sort(key=lambda v: (0 if v.record_type == "source_revision" else 1, v.order_key))
            with DB.transaction(self.conn):
                self.report.blocked_transactions += 1
                ranker = self.ranker
                ranker.reload()
                ex = _Existence(self.conn)
                self._blocked = {r[0] for r in self.conn.execute("SELECT record_sha256 FROM blocked_records").fetchall()}
                self._indexed = set()
                item_batches: Dict[str, List[Tuple]] = {}
                item_revisions: Set[str] = set()
                try:
                    for v in cands:
                        st = self.apply_validated(v, seq, file_uri, R.record_file_id_for_uri(file_uri), item_batches, item_revisions, ex, now)
                        if st == "ok":
                            progress += 1
                            self.report.blocked = max(0, self.report.blocked - 1)
                            self.report.unblocked += 1
                    for pi in items:
                        st = self.apply_item(pi, seq, file_uri, item_batches, item_revisions, ex, now)
                        if st == "ok":
                            progress += 1
                            self.report.blocked = max(0, self.report.blocked - 1)
                            self.report.unblocked += 1
                    self.flush_items(item_batches, item_revisions)
                    ranker.finalize()
                    self.report.reranks += ranker.reranks
                finally:
                    self._blocked = None
                    self._indexed = set()
        return progress


def _revision_then_bundles_key(d: str) -> Tuple[str, int, str]:
    """The re-evaluation's attempt order: a revision's own directory, then its bundles by stamp; revisions by path."""
    if "/_activations/" in d:
        adapter_dir, rest = d.split("/_activations/", 1)
        return (adapter_dir + "/" + rest.split("/", 1)[0], 1, d)
    return (d.rstrip("/"), 0, d)


class _RevisionHeld(Exception):
    """A constituent of a revision directory could not be applied -- or, ``kind = run_drop``, the marker itself would
    move the head while dropping run ids it carries undeclared -- or, ``kind = retirement_head`` (registry-hardening 3a round 2), the
    marker is a retirement whose premise (the head it was computed against, the inputs it retires) is not the live state: the whole
    revision rolls back. ``siblings`` are the
    OTHER blocked constituents of a held fixed-point pass — ``(sha, uri, record_type, parent_table, parent_id)`` each —
    which the held path re-blocks after the rollback so none of them is silently absent from the queue."""

    def __init__(self, kind: str, parent_table: str, parent_id: str, file_uri: str, sha: str, detail: str,
                 record_type: Optional[str] = None, siblings: Sequence[Tuple[str, str, Optional[str], str, str]] = ()):
        super().__init__(f"{kind}: {parent_table} {parent_id} ({detail})")
        self.kind, self.parent_table, self.parent_id = kind, parent_table, parent_id
        self.file_uri, self.sha, self.detail, self.record_type = file_uri, sha, detail, record_type
        self.siblings = list(siblings)


def delivery_status(c: Mapping[str, int], total: int) -> str:
    """The delivery status of one record file (`record_delivery.status`, vocab.DELIVERY_STATUSES) from its counts -- `ok`, `duplicate`,
    `blocked` and `refused`, the last counting the malformed records too -- and its record total: `duplicate` when every record is one;
    else `ok` when nothing is refused or blocked, `blocked` when something is blocked and nothing refused, `partial` when something is
    refused beside an accepted record, `refused` when nothing was accepted. ONE rule for the fold's delivery of a file and for a record of
    it judged again after the fold (:meth:`Ingestor._move_delivery`), so the status always reads the counts the same way."""
    if total and c["duplicate"] == total:
        return "duplicate"
    if c["blocked"] == 0 and c["refused"] == 0:
        return "ok"
    if c["blocked"] and c["refused"] == 0:
        return "blocked"
    return "partial" if c["ok"] or c["duplicate"] else "refused"


_DELIVERY_SQL = DB.upsert_sql(
    "record_delivery",
    ("file_uri", "file_sha256", "arrived_utc", "ingested_utc", "status", "refusal_reason", "producer_script", "producer_session",
     "schema_version", "record_type_counts_json", "n_records", "n_ok", "n_refused", "n_blocked", "n_duplicate"),
    ("file_uri",), keep=("arrived_utc", "producer_script", "producer_session", "schema_version"),
    update={"n_ok": "n_ok + excluded.n_ok"})


# --------------------------------------------------------------------------------------------
# harvester-originated facts (records BEFORE rows)
# --------------------------------------------------------------------------------------------


def is_status_page_artifact(v: "VAL.Validated") -> bool:
    """The status page registers itself as an ``artifact(kind=page)`` (``status.render``, ``extra.page = status``,
    the harvester's own producer). Ingesting that record is a change the harvester CAUSED by rendering: had
    ``registry ingest --html`` re-rendered on it, the new page would be registered, ingested next cycle, and
    rendered again — one artifact record every 20 s, forever (the live install did exactly that, 2026-09-17).
    ``IngestReport.self_change_only`` names a cycle whose only change is such a record."""
    return bool(v.record_type == "artifact" and v.payload.get("kind") == "page" and v.extra.get("page") == "status"
                and v.producer.get("script") == HARVESTER_PRODUCER_SCRIPT)


#: the producing session's name, the first one set wins: the registry's own variable, then the one the session manager hands
#: every session it starts (the same tuple as ``emit.SESSION_ENV_VARS``; a test pins that they agree)
SESSION_ENV_VARS: Tuple[str, ...] = ("SISTEER_SESSION_NAME", "ROMP_SESSION_NAME")


def harvester_producer(root: str, code_sha: str = "unverified") -> Dict[str, object]:
    import socket
    session = next((v for v in (os.environ.get(var) for var in SESSION_ENV_VARS) if v), None)
    return {"script": HARVESTER_PRODUCER_SCRIPT, "code_sha": code_sha, "session": session,
            "host": socket.gethostname(), "substrate": "grading_host"}


def write_harvester_record(root: str, record_type: str, payload: Mapping[str, object], now: Optional[str] = None,
                           subdir: str = "harvester") -> str:
    """Write one harvester fact as a record file under ``records/<subdir>/<utc-day>/``; deterministic
    id (uuid5 over the payload) so a repeated evaluation is a duplicate, not a second fact."""
    from .identity import adapter_record_id
    now = now or R.utc_now()
    rec = R.make_record(record_type, payload, harvester_producer(root), adapter_record_id(payload, record_type), now)
    sha = R.record_sha(rec)
    day = now[:10]
    path = os.path.join(records_dir(root), subdir, day, f"{record_type}-{sha[:12]}.jsonl")
    if os.path.exists(path):
        return path
    R.write_records(path, [rec])
    return path


#: The headline metric row of a RULE-SCORED run (an instrument without a judge): the eval's headline metric, filed at
#: generation with no judgment (``judgment_id = ''``). Its ``n`` is the coverage the scoring reached.
_RULE_SCORED_HEADLINE_SQL = ("FROM run_metric rm JOIN run r ON r.run_id = rm.run_id JOIN eval e ON e.eval_id = r.eval_id "
                             "WHERE rm.run_id = ? AND rm.metric_id = e.headline_metric_id AND rm.judgment_id = ''")


def instrument_judge_id(conn: sqlite3.Connection, run_id: str) -> Optional[str]:
    """The judge the run's instrument names, or None for a rule-scored instrument (exact match, sandbox: no judge)."""
    row = conn.execute("SELECT i.judge_id FROM run r JOIN instrument i ON i.instrument_id = r.instrument_id WHERE r.run_id = ?",
                       (run_id,)).fetchone()
    return row[0] if row else None


def rule_scored_judged_utc(conn: sqlite3.Connection, run_id: str) -> Optional[str]:
    """The scoring time a rule-scored run's ``run_ingest`` carries as its ``judged_utc``: rule scoring happens at
    generation, so the headline metric's ``computed_utc`` when the producer filed one, else the run's ``generated_utc``.
    None when the run already carries a ``judged_utc`` (nothing to fold) or has not generated."""
    row = conn.execute(f"SELECT r.judged_utc, r.generated_utc, (SELECT MAX(rm.computed_utc) {_RULE_SCORED_HEADLINE_SQL}) AS scored_utc "
                       "FROM run r WHERE r.run_id = ?", (run_id, run_id)).fetchone()
    if row is None or row["judged_utc"]:
        return None
    return row["scored_utc"] or row["generated_utc"] or None


def completion_requirements_hold(conn: sqlite3.Connection, run_id: str) -> Tuple[bool, str]:
    """S6: a run's completion requirements, by the kind of instrument it ran under.

    * An instrument that names a judge (LLM-scored): ``lifecycle_state = judged``, a complete judgment by that judge
      covering the run and bound at claim to the run's CURRENT identity
      (:func:`schema_def.judgment_identity_current_sql`: a complete judgment bound to a since-corrected instrument or
      required judge covers nothing here, as in the worklist and the roster match, so such a run stays ``judged`` and listed for
      re-judging rather than reading ``ingested`` with no current reading), the completion artifact present. Coverage is the schema's one predicate
      (:func:`schema_def.judgment_covered_items_sql`): the items judged + the items the judge declined + the items whose
      judging errored (``judgment.n_errors``, else the judgment's own ``run_metric.n_error_outcomes``) reach the required
      count — each is an outcome for a planned item; an error is a judged-outcome-less item, never a verdict and never a
      gap. Until 2026-09-21 errors did not count and a run whose judge covered every planned item but recorded one error
      stayed ``judged`` forever (38 live runs of two readers). ``n_incomplete`` never counts: no outcome at all.
    * An instrument without a judge (rule-scored: exact match, sandbox, letter-logprob prefill — every ``capability``
      eval today): the run is complete when its generation-time headline metric is — ``lifecycle_state`` is ``generated``
      (or ``judged``, should a judge have filed on it anyway) and a ``run_metric`` row with ``metric_id =
      eval.headline_metric_id`` and ``judgment_id = ''`` reaches the required coverage (``COALESCE(n, 0) >= required``,
      the SAME predicate the pending-judgment and roster-match SQL read). The metric row IS the completion evidence:
      ``store_completion_sha256`` is not required of it — the readers that filed the live capability runs recorded one
      on 287 of 735 (2026-09-20), and a run the reader scored is finished whether or not it also copied the artifact's
      sha — while a run whose headline metric is short or absent stays ``generated``. Such a run never carries a
      judgment row (the schema says a rule-scored eval has no judge); its ``run_ingest`` carries the scoring time as
      ``judged_utc`` (:func:`rule_scored_judged_utc`) so the row reaches ``ingested`` with every reached state's utc.

    Keyed on the CURRENT ``lifecycle_state``, never on ``ingested_utc``: a row whose lifecycle fell behind its own stamp (a
    reader's re-emitted ``run_state`` that folded after the ``run_ingest`` — held by the fold since
    ``reducer.lifecycle_regresses``, but 2,929 live rows predate it) is complete again, and
    :func:`derive_run_ingest_records` stamps it again."""
    r = conn.execute("SELECT lifecycle_state, n_items_generated, n_items_planned, store_completion_sha256 "
                     "FROM run WHERE run_id = ?", (run_id,)).fetchone()
    if r is None:
        return False, "run missing"
    if r["lifecycle_state"] == "ingested":
        return False, "already ingested"
    # required coverage and "covers" are the schema's ONE predicate (min(items generated / planned, item set size);
    # judged + declined items reach it) — the arithmetic v_pending_judgment and v_roster_match read, so the run the
    # grading worklist stops offering is the run the harvester ingests and the paper build counts
    required = conn.execute(f"SELECT {S.required_coverage_sql('r')} FROM run r WHERE r.run_id = ?", (run_id,)).fetchone()[0]
    # the coverage that counts is the INSTRUMENT's judge's; an instrument without a judge is rule-scored
    judge_id = instrument_judge_id(conn, run_id)
    if judge_id is None:
        if r["lifecycle_state"] not in ("generated", "judged"):
            return False, f"lifecycle_state={r['lifecycle_state']}"
        n = conn.execute(f"SELECT MAX(COALESCE(rm.n, 0)) {_RULE_SCORED_HEADLINE_SQL}", (run_id,)).fetchone()[0]
        if n is None:
            return False, "no headline metric row (rule-scored: the generation-time metric is the completion evidence)"
        if required is not None and int(n) < int(required):
            return False, f"headline metric coverage {n} < {required} (rule-scored)"
        return True, "ok"
    if r["lifecycle_state"] != "judged":
        return False, f"lifecycle_state={r['lifecycle_state']}"
    if not r["store_completion_sha256"]:
        return False, "completion artifact absent"
    covered = S.judgment_covered_items_sql("jm")
    # the covering judgment is an identity-current one: a complete judgment bound at claim to a since-corrected run
    # identity covers nothing -- here as in v_pending_judgment and v_roster_match -- so a run whose only complete judgment is
    # stale-bound stays `judged` and listed for re-judging, never `ingested` without a current reading
    j = conn.execute(f"SELECT MAX({covered}) FROM judgment jm JOIN run r ON r.run_id = jm.run_id "
                     "LEFT JOIN instrument i ON i.instrument_id = r.instrument_id "
                     f"WHERE jm.run_id = ? AND jm.status = 'complete' AND jm.judge_id = ? AND {S.judgment_identity_current_sql('jm', 'r', 'i')}",
                     (run_id, judge_id)).fetchone()[0]
    who = f" by the instrument's judge {judge_id[:8]}"
    if j is None:
        n_stale = conn.execute("SELECT COUNT(*) FROM judgment jm WHERE jm.run_id = ? AND jm.status = 'complete' AND jm.judge_id = ?",
                               (run_id, judge_id)).fetchone()[0]
        if n_stale:
            return False, (f"no complete judgment{who} bound to the run's current identity ({n_stale} bound at claim to a since-corrected "
                           f"identity: identity_stale, a fresh grade is wanted)")
        return False, f"no complete judgment{who}"
    if required is not None and int(j) < int(required):
        return False, f"judgment coverage {j} < {required}{who}"
    return True, "ok"


def run_ingest_stamp(conn: sqlite3.Connection, run_id: str, now: str) -> str:
    """The ``run_ingest`` order key must follow every record of the run's own key — a producer stamped after the
    harvester's clock (skew) would otherwise fold AFTER ``ingested`` and leave the row at ``judged`` with an
    ``ingested_utc``. Stamp = max(now, max producer written_utc of the run + 1 ms)."""
    row = conn.execute("SELECT MAX(written_utc) FROM row_history WHERE table_name = 'run' AND natural_key_json = ?",
                       (canonical_json({"run_id": run_id}),)).fetchone()
    latest = row[0] if row else None
    if latest is None or latest < now:
        return now
    return R.epoch_ms_to_utc(R.utc_to_epoch_ms(str(latest)) + 1)


def derive_run_ingest_records(root: str, conn: sqlite3.Connection, now: Optional[str] = None,
                              written_floor: Optional[str] = None) -> List[str]:
    """Write a run_ingest record for every candidate run whose completion requirements hold this cycle — a judged run,
    or a ``generated`` run of a rule-scored instrument (no judge; :func:`completion_requirements_hold`) — stamped after
    the run's own records (:func:`run_ingest_stamp`). ``written_floor`` is the earliest ``written_utc`` the RECORD may
    carry — its file's place in the ledger, which orders a harvester file by its stamp (``ingest_once`` passes the
    head's successor) — while the payload's ``ingested_utc`` stays that stamp against the cycle's clock. A rule-scored
    run's record also carries ``judged_utc`` (its scoring time, :func:`rule_scored_judged_utc`), read from the run's own
    rows at derive time and folded from the record on replay, so live equals replay.

    Candidates are the rows NOT CURRENTLY ``ingested`` whose completion holds — never ``ingested_utc IS NULL``: a row whose
    lifecycle fell behind its own stamp (a reader's re-emitted ``run_state`` folded after the ``run_ingest``; the fold holds
    those now, ``reducer.lifecycle_regresses``, the live rows folded before it do not heal themselves) is a candidate again and
    receives a second ``run_ingest`` with a new stamp — a new record, since :func:`write_harvester_record` ids are
    payload-deterministic. A row at ``ingested`` is never a candidate, so nothing is stamped twice; a ``superseded``,
    ``withdrawn``, ``failed`` or ``cancelled`` row is neither ``judged`` nor ``generated``, so never stamped back."""
    now = now or R.utc_now()
    out = []
    rows = conn.execute("SELECT r.run_id FROM run r JOIN instrument i ON i.instrument_id = r.instrument_id "
                        "WHERE r.lifecycle_state <> 'ingested' AND (r.lifecycle_state = 'judged' "
                        "OR (i.judge_id IS NULL AND r.lifecycle_state = 'generated')) ORDER BY r.run_id").fetchall()
    for (rid,) in rows:
        ok, _ = completion_requirements_hold(conn, rid)
        if ok:
            stamp = run_ingest_stamp(conn, rid, now)
            written = stamp if written_floor is None or written_floor <= stamp else written_floor
            payload: Dict[str, object] = {"run_id": rid, "ingested_utc": stamp}
            if instrument_judge_id(conn, rid) is None:
                judged = rule_scored_judged_utc(conn, rid)
                if judged:
                    payload["judged_utc"] = judged
            out.append(write_harvester_record(root, "run_ingest", payload, written))
    return out


def derive_file_arrival_records(root: str, conn: sqlite3.Connection, now: str, only_uris: Optional[Sequence[str]] = None) -> Tuple[List[str], int]:
    """Write ONE ``file_arrival`` record per ledger file whose arrival the log does not carry yet: every
    ``record_ingest`` row whose ``arrival_utc`` differs from its ``first_written_utc`` (the replay's fallback — a file
    whose arrival IS its stamp, the harvester's own files included, needs no fact) and that has no ``file_arrival`` row.
    All of a cycle's facts go into one file under ``records/harvester/<day>/`` named by the content's sha, with
    deterministic record ids, so a cycle that dies between the fold and this step, or whose facts file is folded twice,
    writes the same facts again and the reducer sees duplicates, never a second arrival. Returns ``([path], n)`` —
    the file written (empty when nothing arrived) and the facts it carries. ``only_uris`` restricts the facts to those
    ledger files (the scoped fold's own files, :func:`fold_only_files`); the cycle passes none and covers every file."""
    from .identity import adapter_record_id
    scope = ""
    params: List[object] = []
    if only_uris is not None:
        if not only_uris:
            return [], 0
        scope = f" AND i.file_uri IN ({','.join('?' * len(only_uris))})"
        params = list(only_uris)
    rows = conn.execute("SELECT i.file_uri, i.file_sha256, i.arrival_utc FROM record_ingest i "
                        "WHERE i.arrival_utc <> i.first_written_utc "
                        "AND NOT EXISTS (SELECT 1 FROM file_arrival f WHERE f.file_uri = i.file_uri)" + scope +
                        " ORDER BY i.ingest_seq", params).fetchall()
    if not rows:
        return [], 0
    prod = harvester_producer(root)
    recs = []
    for uri, sha, arrival in rows:
        payload = {"file_uri": uri, "file_sha256": sha, "arrival_utc": arrival}
        recs.append(R.make_record("file_arrival", payload, prod, adapter_record_id(payload, "file_arrival"), now))
    data = "".join(canonical_json(r) + "\n" for r in recs).encode("utf-8")
    sha12 = hashlib.sha256(data).hexdigest()[:12]
    path = os.path.join(records_dir(root), HARVESTER_SUBDIR, now[:10], f"{ARRIVAL_FILE_PREFIX}{sha12}.jsonl")
    if not os.path.exists(path):
        R.write_records(path, recs)
    return [path], len(recs)


def pending_refusal_fact_files(root: str, conn: sqlite3.Connection) -> List[Tuple[str, Optional[str]]]:
    """The harvester's own refusal-fact files (:data:`REFUSAL_FACT_FILE_PREFIX` under ``records/harvester/``) whose logical uri is not
    in ``record_delivery`` -- a fact file a cycle wrote and died before folding (step 0 of every cycle): ``[(path, logical sha)]``
    in ingest order, for :meth:`Ingestor.ingest_parsed` BEFORE the START derivation, so the facts it carries are rows when the derivation
    looks and no second file for the same (sha, stamp) is ever written. One directory walk of the harvester tree (13,629 files live,
    2026-10-03); nothing is hashed that is not such a file."""
    base = os.path.join(records_dir(root), HARVESTER_SUBDIR)
    if not os.path.isdir(base):
        return []
    known = {r[0] for r in conn.execute("SELECT file_uri FROM record_delivery").fetchall()}
    out: List[Tuple[str, Optional[str]]] = []
    for path in R.iter_record_files(base):
        if not os.path.basename(path).startswith(REFUSAL_FACT_FILE_PREFIX):
            continue
        uri = logical_uri(root, path)
        if uri in known:
            continue
        try:
            out.append((path, R.file_logical_sha(path)))
        except (OSError, EOFError, gzip.BadGzipFile):
            out.append((path, None))                      # unreadable: the parser refuses it by name
    return order_for_ingest(root, out)


#: A refused file larger than this (decompressed) is not scanned for the producer block of a record named without its line: the fact then
#: carries producer_json NULL (its why-null reason), never a stalled cycle.
REFUSED_FILE_SCAN_BYTES = LARGE_FILE_BYTES


def refused_record_producer(root: str, record_uri: str, sha: str) -> Optional[Dict[str, object]]:
    """The producer block of the refused record ``sha`` as its file holds it (the fact re-reads the header by sha), or None when the
    file is gone, unreadable, or does not hold the record. A uri of the form ``<file>#<line>`` reads that one line and accepts it only when
    its sha is the record's; a bare file uri scans the file for the sha, bounded by :data:`REFUSED_FILE_SCAN_BYTES`. Never raises."""
    base, _, frag = str(record_uri).partition("#")
    path = resolve_uri(root, base)
    if path is None:
        return None
    try:
        if frag.isdigit():
            want = int(frag)
            with R.open_record_file(path) as fh:
                for i, line in enumerate(fh, start=1):
                    if i != want:
                        continue
                    line = line.strip()
                    if not line:
                        return None
                    rec = json.loads(line)
                    if isinstance(rec, dict) and R.record_sha(rec) == sha and isinstance(rec.get("producer"), dict):
                        return dict(rec["producer"])
                    return None
            return None
        try:
            if os.path.getsize(path) > REFUSED_FILE_SCAN_BYTES and not path.endswith(".gz"):
                return None
        except OSError:
            return None
        seen = 0
        for _, rec, rsha in R.read_record_file(path):
            seen += 1
            if rsha == sha:
                return dict(rec["producer"]) if isinstance(rec.get("producer"), dict) else None
            if seen >= 200_000:
                return None
    except (OSError, ValueError, EOFError, gzip.BadGzipFile, UnicodeDecodeError):
        return None
    return None


def derive_refusal_fact_records(root: str, conn: sqlite3.Connection, now: str, written_utc: str) -> Tuple[List[str], int]:
    """Write ONE ``record_refusal`` record per refusal judgment the log does not carry yet: every ``record_refusal`` row whose
    current ``(record_sha256, COALESCE(judged_with, 'unverified'))`` has no ``record_refusal_fact`` row. Run at the START of every cycle
    (before the inbox is listed, before any re-attempt could REPLACE a row) -- which materializes the rows refused before the facts
    existed as ``unverified`` facts with ``extra.backfilled = true``, a refusal a crashed cycle committed without its fact, and the previous
    judgment of a row this cycle is about to re-judge -- and again at its END, for the cycle's own refusals and re-judgments. The payload
    carries the row (sha, uri, type, field, reason, stamp, refused_utc), the refused record's producer block re-read from its file by sha
    (:func:`refused_record_producer`; NULL when it cannot be) and ``cycle_utc`` = this cycle's clock; the record id is minted over the
    payload WITHOUT ``cycle_utc`` (:func:`identity.adapter_record_id`), so two cycles deriving the same judgment mint the same id and the
    fact table's key dedupes the row. All of a pass's facts go into one file under ``records/harvester/<day>/`` named by the content's
    sha, stamped ``written_utc`` (the ledger's next arrival, as the arrival facts are). Returns ``([path], n)``, the file written (empty when
    nothing is due) and the facts it carries."""
    from .identity import adapter_record_id
    rows = conn.execute(
        "SELECT r.record_sha256, r.record_uri, r.record_type, r.field, r.reason, r.refused_utc, COALESCE(r.judged_with, 'unverified'), "
        "r.judged_with IS NULL FROM record_refusal r WHERE NOT EXISTS (SELECT 1 FROM record_refusal_fact f WHERE f.record_sha256 = r.record_sha256 "
        "AND f.judged_with = COALESCE(r.judged_with, 'unverified')) ORDER BY r.refused_utc, r.record_sha256").fetchall()
    if not rows:
        return [], 0
    prod = harvester_producer(root)
    recs = []
    for sha, uri, rtype, field_name, reason, refused_utc, stamp, backfilled in rows:
        payload: Dict[str, object] = {"record_sha256": str(sha), "record_uri": str(uri), "reason": str(reason), "judged_with": str(stamp),
                                      "refused_utc": str(refused_utc)}
        if rtype is not None:
            payload["record_type"] = str(rtype)
        if field_name is not None:
            payload["field"] = str(field_name)
        producer = refused_record_producer(root, str(uri), str(sha))
        if producer is not None:
            payload["producer_json"] = producer
        if backfilled:
            payload["extra"] = {"backfilled": True}
        rid = adapter_record_id(payload, "record_refusal")
        payload["cycle_utc"] = now
        recs.append(R.make_record("record_refusal", payload, prod, rid, written_utc))
    data = "".join(canonical_json(r) + "\n" for r in recs).encode("utf-8")
    sha12 = hashlib.sha256(data).hexdigest()[:12]
    path = os.path.join(records_dir(root), HARVESTER_SUBDIR, now[:10], f"{REFUSAL_FACT_FILE_PREFIX}{sha12}.jsonl")
    if not os.path.exists(path):
        R.write_records(path, recs)
    return [path], len(recs)


_UTC_MS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


def read_arrival_facts(root: str, shas: Optional[Dict[str, str]] = None, only: Optional[Collection[str]] = None) -> Dict[str, str]:
    """``{file_uri: arrival_utc}`` from every ``file_arrival`` fact under ``records/harvester/`` — what a replay orders
    files by, read BEFORE the replay ranks anything, since a file's arrival is its ordering key and the fact
    about it is folded after it. The EARLIEST recorded arrival per file wins: the first live fold is the arrival by
    definition, and a later fact for the same file (a live fold of an emptied database, which the docstring of
    :class:`Ranker` names as unsupported) cannot move it. A malformed fact is skipped here (the fold refuses it by name).
    ``only``: the replay's allowlist of logical uris -- only the harvester files IT names are read, so a
    facts file written after the head cannot order the replay (a listed file with no listed fact takes its ``first_written_utc``,
    as the live fold did before the fact existed).
    ``shas``, when given, is filled with ``{file_uri: file_sha256}`` of the same winning facts -- the bytes the live fold
    ranked at each uri, which :func:`prefer_recorded_twins` uses to order two physical files at one uri."""
    base = os.path.join(records_dir(root), HARVESTER_SUBDIR)
    out: Dict[str, str] = {}
    if not os.path.isdir(base):
        return out
    prefix = os.path.join("records", HARVESTER_SUBDIR) + "/"
    if only is not None:
        paths = [p for p in (resolve_uri(root, u) for u in sorted(only) if u.startswith(prefix)) if p is not None]
    else:
        paths = list(R.iter_record_files(base))
    for path in paths:
        try:
            for _, rec, _ in R.read_record_file(path, compute_sha=False):
                if rec.get("record_type") != "file_arrival":
                    continue
                payload = rec.get("payload")
                if not isinstance(payload, dict):
                    continue
                uri, arrival = payload.get("file_uri"), payload.get("arrival_utc")
                if not isinstance(uri, str) or not isinstance(arrival, str) or not _UTC_MS_RE.match(arrival):
                    continue
                if uri not in out or arrival < out[uri]:
                    out[uri] = arrival
                    if shas is not None:
                        sha = payload.get("file_sha256")
                        if isinstance(sha, str) and sha:
                            shas[uri] = sha
                        else:
                            shas.pop(uri, None)
        except (OSError, ValueError, EOFError, gzip.BadGzipFile):
            continue
    return out


def prefer_recorded_twins(root: str, work: Sequence[Tuple[str, Optional[str]]], recorded_shas: Mapping[str, str]) -> List[Tuple[str, Optional[str]]]:
    """``work`` with, among the physical files sharing one logical uri (a plain file beside its ``.gz``), the one whose bytes the log's
    ``file_arrival`` fact recorded (``file_sha256``) moved ahead of the others -- in place, every other position untouched -- so a replay
    ranks the bytes the live fold ranked and refuses the other as the live fold did. A uri with no recorded sha, or none of whose
    files hash to it, keeps the listing's order (the plain name sorts before its ``.gz``). Only twin groups are hashed."""
    groups: Dict[str, List[int]] = collections.defaultdict(list)
    for i, (path, _sha) in enumerate(work):
        groups[logical_uri(root, path)].append(i)
    out = list(work)
    for uri, idx in groups.items():
        want = recorded_shas.get(uri) if len(idx) > 1 else None
        if not want:
            continue

        def rank(w: Tuple[str, Optional[str]]) -> int:
            try:
                return 0 if (w[1] or R.file_logical_sha(w[0])) == want else 1
            except (OSError, EOFError, gzip.BadGzipFile):
                return 2
        items = sorted((work[i] for i in idx), key=rank)       # stable: ties keep the listing's order
        for i, w in zip(idx, items):
            out[i] = w
    return out


# --------------------------------------------------------------------------------------------
# gzip after ingest
# --------------------------------------------------------------------------------------------


def gzip_after_ingest(root: str, conn: sqlite3.Connection, min_age_s: int = GZIP_MIN_AGE_S, now_s: Optional[float] = None,
                      destinations: Optional[Sequence[str]] = None) -> int:
    """Compress record files ingested ``ok``, older than ``min_age_s``, and (when mirrors are configured)
    acked by every destination. The logical sha never changes; ``record_uri`` keeps the plain path."""
    now_s = now_s if now_s is not None else time.time()
    if destinations is None:
        destinations = [d["name"] if isinstance(d, dict) else str(d) for d in (load_config(root).get("mirrors", []) or [])]
    acked_through = None
    if destinations:
        # a destination is compared as its label on both sides (site_aliases.py): the fold writes an ack under the label even when
        # its record spelled the destination the site's own way, and the configuration may still spell it that way
        SA.install(root)
        rows = conn.execute("SELECT destination, acked_through_seq FROM mirror_ack").fetchall()
        acks = {SA.canonical(r[0]): int(r[1]) for r in rows}
        wanted = [SA.canonical(d) for d in destinations]
        if any(d not in acks for d in wanted):
            return 0
        acked_through = min(acks[d] for d in wanted)
    n = 0
    rows = conn.execute("SELECT d.file_uri, i.ingest_seq, d.file_sha256 FROM record_delivery d LEFT JOIN record_ingest i ON i.file_uri = d.file_uri "
                        "WHERE d.status IN ('ok', 'duplicate') ORDER BY d.file_uri").fetchall()
    for uri, seq, delivered_sha in rows:
        path = os.path.join(root, uri)
        if not os.path.exists(path):
            continue
        if os.path.exists(path + ".gz"):
            continue            # a plain twin beside the compressed file: both stay; replacing the .gz with the twin's bytes is never this pass's
        if now_s - os.path.getmtime(path) < min_age_s:
            continue
        if acked_through is not None and seq is not None and int(seq) > acked_through:
            continue
        plain_sha = R.file_logical_sha(path)
        if plain_sha != str(delivered_sha):
            continue            # the bytes on disk are not the ones delivered: the fold judges them first (a replacement is refused by name)
        tmp = path + ".gz.tmp"
        with open(path, "rb") as src, open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as dst:
            shutil.copyfileobj(src, dst)
        os.replace(tmp, path + ".gz")
        if R.file_logical_sha(path + ".gz") != plain_sha:
            os.remove(path + ".gz")
            continue
        os.remove(path)
        n += 1
    return n


# --------------------------------------------------------------------------------------------
# the cycle
# --------------------------------------------------------------------------------------------


def _load_file_stat(root: str) -> Dict[str, Dict[str, object]]:
    p = os.path.join(root, "state", FILE_STAT_NAME)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_file_stat(root: str, entries: Mapping[str, Mapping[str, object]]) -> None:
    p = os.path.join(state_dir(root), FILE_STAT_NAME)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, sort_keys=True, separators=(",", ":"))
    os.replace(tmp, p)


def refused_replacement(conn: sqlite3.Connection, root: str, path: str, sha: Optional[str]) -> bool:
    """Whether ``path`` is a physical file at an INGESTED uri carrying other bytes that this code has already refused by name
    (:meth:`Ingestor._refuse_replaced`): the uri's ledger sha (``record_ingest``) differs from ``sha`` and a ``record_refusal`` row keyed
    by ``sha`` at that uri carries the current judging stamp. Not new work -- the same bytes under the same code refuse again, as
    :func:`refused_hold_exact` reasons for a held revision; a moved stamp re-judges the file once, which re-records the judgment. A file
    with no sha (unreadable) or at a uri never ranked never qualifies. Two point queries per candidate the delivery ledger did not already
    settle, so an idle cycle asks nothing."""
    if not sha:
        return False
    uri = logical_uri(root, path)
    row = conn.execute("SELECT file_sha256 FROM record_ingest WHERE file_uri = ?", (uri,)).fetchone()
    if row is None or str(row[0]) == sha:
        return False
    judged = conn.execute("SELECT judged_with FROM record_refusal WHERE record_sha256 = ? AND record_uri = ?", (sha, uri)).fetchone()
    return judged is not None and judged[0] == validator_stamp()


def new_or_changed_files(root: str, conn: sqlite3.Connection,
                         cache: Optional[Dict[str, Dict[str, object]]] = None) -> List[Tuple[str, Optional[str]]]:
    """``[(path, logical sha)]`` of every record file whose sha is not the one ``record_delivery`` holds, in ingest
    order. F12: a file whose (size, mtime_ns, inode) equal its ``state/file_stat.json`` entry keeps the recorded sha,
    so an idle cycle stats every file and decompresses none; an unreadable file rides along with sha None for the
    parser to refuse by name. A file at an INGESTED uri with other bytes that this code already refused by name is
    not raised again (:func:`refused_replacement`); one with bytes not yet judged is listed, and the fold refuses
    it. A ``cache`` passed in is updated in place (pruned to the files on disk) and left for the caller to save;
    without one the sidecar is loaded and saved here."""
    known = {r[0]: r[1] for r in conn.execute("SELECT file_uri, file_sha256 FROM record_delivery").fetchall()}
    own_cache = cache is None
    cache = _load_file_stat(root) if cache is None else cache
    fresh: Dict[str, Dict[str, object]] = {}
    out: List[Tuple[str, Optional[str]]] = []
    for sub in ("records", "rulings"):
        base = os.path.join(root, sub)
        if not os.path.isdir(base):
            continue
        for path in R.iter_record_files(base):
            try:
                st = os.stat(path)
            except OSError:
                continue
            ent = cache.get(path)
            if ent and ent.get("size") == st.st_size and ent.get("mtime_ns") == st.st_mtime_ns and ent.get("ino") == st.st_ino:
                sha = str(ent["sha"])
            else:
                try:
                    sha = R.file_logical_sha(path)
                    st2 = os.stat(path)
                except (OSError, EOFError, gzip.BadGzipFile):
                    out.append((path, None))
                    continue
                if (st2.st_size, st2.st_mtime_ns, st2.st_ino) != (st.st_size, st.st_mtime_ns, st.st_ino):
                    out.append((path, None))          # changed while being hashed: the parser will settle its sha; cache nothing
                    continue
                ent = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "ino": st.st_ino, "sha": sha}
            fresh[path] = ent
            if known.get(logical_uri(root, path)) != sha and not refused_replacement(conn, root, path, sha):
                out.append((path, sha))
    if fresh != cache:
        cache.clear()
        cache.update(fresh)
        if own_cache:
            try:
                _save_file_stat(root, cache)
            except OSError:
                pass
    return order_for_ingest(root, out)


#: ``state/harvest_cycle.json`` -- the last COMPLETED harvester cycle (:func:`write_cycle_state`): what the status page's harvester row
#: ages. A cycle its unit kills never writes it; a cycle its budget stops does (the stop is a completion with work left for the next).
CYCLE_STATE_NAME = "harvest_cycle.json"


HELD_ATTEMPTS_NAME = "held_attempts.json"


def read_held_attempts(root: str) -> Dict[str, Dict[str, str]]:
    """``state/held_attempts.json`` (registry follow-ups-4 item 10): root-relative held directory -> {attempt_identity, attempted_utc}, the
    memo of what the last attempt read and when. A STATE file beside harvest_cycle.json, never a row: the marker's extra_json must replay byte
    for byte and a clock or a listing's mtimes would not. Absent or torn -> empty (every hold re-attempts once and records itself)."""
    try:
        with open(os.path.join(state_dir(root), HELD_ATTEMPTS_NAME), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return {str(k): dict(v) for k, v in data.items() if isinstance(v, Mapping)} if isinstance(data, Mapping) else {}


def write_held_attempts(root: str, memo: Mapping[str, Mapping[str, str]]) -> None:
    """Atomic (tmp + rename); a failed write never fails the cycle (the next attempt records itself again)."""
    try:
        d = state_dir(root)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, HELD_ATTEMPTS_NAME)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({k: dict(v) for k, v in sorted(memo.items())}, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        pass


def write_cycle_state(root: str, report: IngestReport, now: str) -> None:
    """Record the completed cycle: ``last_completed_cycle_utc`` (the clock at completion), the cycle's own clock, its wall time, how it
    stopped (``budget`` or not at all), what it took and what it left. Atomic (tmp + rename), never fails the cycle."""
    payload = {"last_completed_cycle_utc": report.completed_utc, "cycle_utc": now, "elapsed_s": report.elapsed_s, "stopped": report.stopped,
               "budget_s": report.budget_s, "files": report.files, "ok": report.ok, "refused": report.refused, "refusal_facts": report.refusal_facts,
               "blocked": report.blocked, "files_replaced_refused": report.files_replaced_refused,
               "revisions_applied": report.revisions_applied, "revisions_held": report.revisions_held, "revisions_noop": report.revisions_noop,
               "revisions_held_skipped": report.revisions_held_skipped,
               "revisions_deferred_budget": report.revisions_deferred_budget, "plain_files_deferred_budget": report.plain_files_deferred_budget,
               "transactions": report.transactions}
    try:
        path = os.path.join(state_dir(root), CYCLE_STATE_NAME)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except OSError:
        pass


def read_cycle_state(root: str) -> Optional[Dict[str, object]]:
    """The last completed cycle's record (:func:`write_cycle_state`), or None when no cycle has completed under this code yet."""
    path = os.path.join(state_dir(root), CYCLE_STATE_NAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def fold_only_files(root: str, conn: sqlite3.Connection, files: Sequence[str], report: IngestReport, now: str) -> List[str]:
    """Fold exactly ``files`` -- plain record files under ``root`` (absolute, or relative to it) -- through the chunk path every
    plain file takes (:meth:`Ingestor.ingest_parsed`: validation, the ledger rank, the fold, the delivery audit, a dependency
    block where a parent is missing), and nothing else. Ranks are the eager path's: on the live path a new file is always
    head + 1 (:class:`Ranker`), so no dense pass is needed and none runs. The files' ``file_arrival`` facts ARE written and
    folded here, scoped to these files (:func:`derive_file_arrival_records` with ``only_uris``, stamped at the ranker's next
    arrival exactly as a cycle stamps its facts): a rebuild before the next full cycle would otherwise replay the file at its
    producer stamp and every ``ingest_seq`` after it would differ. The decode-policy stamp (``decode_match.stamp_run_rows``) is
    re-derived for the runs these records name, since a judgment's state effect rewrites its run's live row from the fold's
    stamps -- the cycle does the same for every run. The stat sidecar is not touched (the next cycle hashes the file once and
    finds its delivery row), and ``state/harvest_cycle.json`` is never written. A file whose logical sha
    ``record_delivery`` already holds is not new work and is skipped; a file under a revision directory is refused (``ValueError``:
    a revision activates whole, through :meth:`Ingestor.apply_revision`, the cycle's), as is a path outside ``root`` or a path
    that is not a file. Returns the logical uris folded."""
    root_abs = os.path.abspath(root)
    paths: List[str] = []
    for f in files:
        p = os.path.abspath(f if os.path.isabs(f) else os.path.join(root_abs, f))
        if os.path.commonpath([p, root_abs]) != root_abs:
            raise ValueError(f"only_files: {f!r} is not under the registry root {root_abs}")
        if not os.path.isfile(p):
            raise ValueError(f"only_files: {f!r} is not a record file")
        uri = logical_uri(root, p)
        if revision_of_uri(uri):
            raise ValueError(f"only_files: {uri} belongs to a revision directory, which activates whole in the harvester's cycle")
        paths.append(p)
    if not paths:
        return []
    marks = ",".join("?" * len(paths))
    uris = [logical_uri(root, p) for p in paths]
    known = {r[0]: r[1] for r in conn.execute(f"SELECT file_uri, file_sha256 FROM record_delivery WHERE file_uri IN ({marks})", uris).fetchall()}
    work: List[Tuple[str, Optional[str]]] = []
    for p, uri in zip(paths, uris):
        sha = R.file_logical_sha(p)
        if known.get(uri) == sha:
            continue            # delivered under these very bytes: not new work (a re-fold would only deliver a duplicate)
        work.append((p, sha))
    if not work:
        return []
    # the writers' fold-order check on the caller's connection before anything folds (db.require_fold_order: the code's order recorded,
    # or none -- stamped -- else refused by name); a claim's connection came through open_or_create, which ran it, but this is a fold
    # writer in its own right and checks what it folds into
    DB.require_fold_order(conn, DB.db_path(root))
    work = order_for_ingest(root, work)
    ing = Ingestor(conn, root, report)                      # eager ranks (defer_ranks=False): no end-of-cycle dense pass, none needed
    ing.file_shas = {p: sha for p, sha in work if sha}
    ing.listed_uris = {logical_uri(root, p) for p, _ in work}
    taken, _ = ing.ingest_parsed(iter_parsed(root, work, 0, ITEM_ROWS_PER_TRANSACTION), now)
    if taken:
        # the arrival of each folded file, as a fact in the log, stamped strictly after the head like a cycle's facts
        ing.ranker.reload()
        facts_now = ing.ranker.next_arrival(now)
        fact_paths, n_facts = derive_file_arrival_records(root, conn, facts_now, only_uris=sorted(taken))
        report.arrival_facts += n_facts
        if fact_paths:
            facts = order_for_ingest(root, [(f, R.file_logical_sha(f)) for f in fact_paths])
            ing.ingest_parsed(iter_parsed(root, facts, 0, ITEM_ROWS_PER_TRANSACTION), now)
        # the derived decode-policy stamp on the run rows these records rewrote (a judgment's state effect rewrites its run's
        # live row from the fold's own stamps, dropping the derived one): what the cycle re-derives for every run, for these only
        run_ids = sorted({str(rec["payload"]["run_id"]) for p, _ in work for _n, rec, _s in R.read_record_file(p, compute_sha=False)
                          if isinstance(rec.get("payload"), dict) and rec["payload"].get("run_id")})
        if run_ids:
            with DB.transaction(conn):
                stamped = DM.stamp_run_rows(conn, run_ids=run_ids)
            if stamped:
                ing._metric({"kind": "decode_policy_match", "stamped": stamped, "scoped": len(run_ids), "utc": now})
    return sorted(taken)


def ingest_once(root: str, conn: Optional[sqlite3.Connection] = None, workers: int = 3,
                now: Optional[str] = None, min_free: int = MIN_FREE_BYTES,
                derive_facts: bool = True, gzip_min_age_s: int = GZIP_MIN_AGE_S, budget_s: Optional[float] = None,
                clock=time.monotonic, retry_refused: bool = False, only_files: Optional[Sequence[str]] = None,
                retry_held: bool = False, held_sweep_s: Optional[float] = None) -> IngestReport:
    """One harvester cycle. The caller holds the write lock (``db.write_lock(root)``) or passes
    ``conn=None`` to let this function take it. The status page is the CLI's to render afterwards
    (``registry ingest --html``), outside the lock and with its errors visible.

    ``only_files`` makes this NOT a cycle but a fold of exactly those plain record files (paths under ``root``, absolute
    or relative to it): :func:`fold_only_files` -- no spool listing, no inbox merge, no blocked-queue re-evaluation, no
    harvester facts, no dense re-rank, no gzip pass, no paper head, and ``state/harvest_cycle.json`` is NOT rewritten (a claim is
    not a harvester cycle; the stamp is what the status page ages, see :func:`pending.ingest_now`). The one caller is the claim
    mechanism, which must make its own just-written record visible under the lock and used to pay a FULL cycle for it at every
    claim and every release (a 7-run claim = 7 cycles; a spool backlog made each one longer and the lock hold starved the
    harvester: 8 refusals/h on 2026-10-02).

    DURABLE PROGRESS. Every unit of the fold commits on its own -- a plain chunk (≤ 50 files), a dimension bundle, each staging window
    and the one activation of a revision directory or activation bundle, the held path's one transaction that records a hold -- so a
    cycle killed by its unit's ``TimeoutStartSec`` keeps everything it committed and the next cycle continues from there: a file with a
    delivery row is not new work, a partially staged revision resumes from its staged windows. ``budget_s`` bounds the cycle's wall
    clock from its start (inbox merge included; the cluster pull is :func:`ingest_under_lock`'s, run before the lock and outside this
    budget): once spent, the cycle takes no NEW plain chunk or revision directory /
    bundle (the one in hand finishes; the analysis filings' late phase is budgeted the same way), runs its closing steps (blocked queue,
    facts, the dense pass, gzip, the paper head) and reports ``stopped = "budget"`` with what it left (``revisions_deferred_budget`` /
    ``plain_files_deferred_budget``, the filing files not taken counted among the plain ones); a unit whose budget sits
    below its timeout therefore folds a backlog in bounded, durable cycles instead of dying mid-fold. A completed cycle -- budgeted or
    not -- records itself in ``state/harvest_cycle.json`` (:func:`write_cycle_state`); a killed one does not, which is what the status
    page's harvester row ages. ``clock`` is the monotonic clock the budget and ``elapsed_s`` read (a test's fake clock)."""
    t_cycle0 = clock()
    ensure_layout(root)
    report = IngestReport()
    report.budget_s = budget_s
    now = now or R.utc_now()   # ONE clock reading per cycle: blocked-record attempts count cycles, not calls
    ok, free = free_space_guard(root, min_free, now)
    report.free_bytes = free
    if not ok:
        report.paused = True
        return report

    def over_budget() -> bool:
        return budget_s is not None and (clock() - t_cycle0) > budget_s
    own_conn = conn is None
    if own_conn:
        # the writers' open: refuses (db.SchemaDrift, naming the table / constraint / object) a database whose version or
        # SHAPE is not the code's — nothing is folded into a file whose CHECKs, indexes, triggers or v_* objects differ
        # from the DDL this code was written against (the 2026-09-19 hold: an old paper_label CHECK refusing records
        # mid-revision). The CLI turns it into exit 3 and the status page's banner; a passed-in connection is the caller's
        # to have opened the same way (backfill drops indexes on purpose between its open and this call).
        conn = DB.open_or_create(DB.db_path(root))
    else:
        # the caller opened it; the one check a fold cannot do without is repeated on it (db.require_fold_order: one row read, the
        # bootstrap stamp when the database records no order, a refusal by name when it records another)
        DB.require_fold_order(conn, DB.db_path(root))
    assert conn is not None
    try:
        if only_files is not None:
            fold_only_files(root, conn, only_files, report, now)
            report.elapsed_s = round(clock() - t_cycle0, 3)
            return report
        merged = scan_inbox(root)
        report.conflicts = sum(1 for _, s in merged if s == "conflict")
        t0 = time.monotonic()
        # ranks are provisional within the cycle and made dense ONCE at its end (finalize_ranks); the rank is a pure
        # function of the ledger's keys, so the end-of-cycle ranks equal the eager path's
        ing = Ingestor(conn, root, report, defer_ranks=True)
        ing.retry_refused = retry_refused
        # every held revision re-attempts under --retry-held (the window's uncapped `dimensions load` / `ingest --retry-held`); otherwise a held
        # directory whose attempt identity stands is skipped until the identity moves or `held_sweep_s` passes ([harvest] held_sweep_s, default an hour)
        ing.retry_held = bool(retry_held)
        try:
            ing.held_sweep_s = float(held_sweep_s if held_sweep_s is not None else (CFG.load(root).get("harvest", {}).get("held_sweep_s") or HELD_SWEEP_S))
        except Exception:  # noqa: BLE001 -- a torn config never fails a cycle: the default sweep
            ing.held_sweep_s = HELD_SWEEP_S
        # Step 0: the harvester's OWN refusal-fact files a cycle wrote and died before folding are folded first, so the derivation
        # below sees their rows and never writes a second file for the same (sha, stamp)
        own_facts = pending_refusal_fact_files(root, conn)
        if own_facts:
            ing.ingest_parsed(iter_parsed(root, own_facts, 0, ITEM_ROWS_PER_TRANSACTION), now)
        # The START pass: every refusal judgment with no fact yet becomes a record and a row NOW -- before the listing, the parse, the
        # blocked-queue re-evaluation and any re-attempt that could REPLACE a record_refusal row (the previous judgment of a row about to
        # be re-judged is a fact before the retry; the rows refused before the facts existed are `unverified` facts, backfilled)
        ing.ranker.reload()
        start_paths, n_start = derive_refusal_fact_records(root, conn, now, ing.ranker.next_arrival(now))
        if start_paths:
            ing.ingest_parsed(iter_parsed(root, order_for_ingest(root, [(f, R.file_logical_sha(f)) for f in start_paths]), 0,
                                          ITEM_ROWS_PER_TRANSACTION), now)
        report.refusal_facts += n_start
        stat_cache = _load_file_stat(root)
        work = new_or_changed_files(root, conn, stat_cache)
        plain, rev_dirs, dim_bundles = split_work(root, work)
        plain, late = partition_analysis_files(root, plain)          # the analysis filings fold in the late phase, below
        rev_dirs = group_revision_dirs(root, rev_dirs, conn)   # every source's directories and bundles adjacent, ledger order within (no-ops behind the real attempt)
        ing.file_shas = {path: sha for path, sha in work if sha}      # a no-op bundle's files are delivered from these, never re-hashed
        ing.listed_uris = {logical_uri(root, path) for path, _ in work}   # a refused hold whose file the listing names as new or changed is not exact
        # the dimension seed revisions FIRST, each whole in one transaction: the FK roots of everything the plain files and
        # the revision directories below may reference, and a reader of the paper objects sees the old or the new dimension
        # set, never a mix (apply_dimension_bundle)
        for dim_dir, dim_files in dim_bundles:
            ing.apply_dimension_bundle(dim_dir, dim_files, now)
        # a file an interrupted cycle committed part of resumes after its cursor when its bytes are the ones the cursor
        # belongs to; a moved sha starts the file over
        cursors = {r[0]: (r[1], int(r[2])) for r in conn.execute("SELECT file_uri, file_sha256, records_done FROM record_cursor").fetchall()}
        skips = {path: cursors[logical_uri(root, path)][1] for path, sha in plain
                 if sha and logical_uri(root, path) in cursors and cursors[logical_uri(root, path)][0] == sha}
        # parse (workers, streamed) and write (one transaction per chunk) overlap; parse_s is the time outside transactions
        taken, stopped = ing.ingest_parsed(iter_parsed(root, plain, workers, ITEM_ROWS_PER_TRANSACTION, skips), now, stop=over_budget)
        if stopped:
            report.stopped = "budget"
            report.plain_files_deferred_budget = sum(1 for path, _ in plain if logical_uri(root, path) not in taken)
            report.revisions_deferred_budget = len(rev_dirs)
        else:
            for i, d in enumerate(rev_dirs):
                if over_budget():
                    # the budget is asked BETWEEN bundles, never inside one: the bundle in hand always finishes (its staging windows
                    # and activation are committed on their own either way), the rest wait for the next cycle
                    report.stopped = "budget"
                    report.revisions_deferred_budget = len(rev_dirs) - i
                    break
                ing.apply_revision(d, now)                 # bounded staging + one small activation per revision directory
                report.revisions_applied += 1
        report.parse_s = max(0.0, (time.monotonic() - t0) - report.write_s)
        ing.reevaluate_blocked(now, stop=over_budget)      # the latest bundle of every held revision is attempted here, within the budget
        # the LATE phase: the analysis filings (records/analyses/) fold after every parent they can name has
        # settled -- the plain files, the revision directories and the queue -- so an input whose run rides a revision directory meets its
        # parent, on this path exactly as on a rebuild; then every standing dependency refusal is re-selected and accepted where its
        # parent now exists (retry_dependency_refusals). A cycle its budget stopped defers the late files to the next one, unlisted
        if stopped:
            report.plain_files_deferred_budget += len(late)
        else:
            # the late phase is budgeted like the plain phase: a chunk not taken is the cycle's stop by name, and the filing files it left
            # count with the plain files deferred (the sixth closing pass, 2026-10-09: the pass's result was discarded, so a cycle whose
            # budget ran out before or inside the filings reported no stop and nothing deferred while they stayed unread). The retry runs
            # when the late phase reaches its end, as after a stop in the revision phase: the next cycle re-selects every standing refusal
            late_taken, late_stopped = ing.ingest_parsed(iter_parsed(root, late, workers, ITEM_ROWS_PER_TRANSACTION), now, stop=over_budget)
            if late_stopped:
                report.stopped = "budget"
                report.plain_files_deferred_budget += sum(1 for path, _ in late if logical_uri(root, path) not in late_taken)
            else:
                ing.retry_dependency_refusals(now)
        ing.reclaim(now)
        # the harvester's own facts are ordered by their stamp (Ranker.arrival_for), so they are stamped strictly after
        # the head the cycle's files left — the clock, or head + 1 ms when this cycle's arrivals ran past it
        ing.ranker.reload()
        facts_now = ing.ranker.next_arrival(now)
        derived: List[str] = []
        if derive_facts:
            derived = derive_run_ingest_records(root, conn, now, written_floor=facts_now)
            try:
                from . import paper
                derived += paper.evaluate_rulings(root, conn, now, written_utc=facts_now)
            except Exception as exc:  # the ruling evaluator never kills the cycle
                ing._metric({"kind": "ruling_eval_error", "error": str(exc), "utc": now or R.utc_now()})
            report.harvester_records += len(derived)
        # the arrival of every file this cycle folded goes into the log as a fact, so a rebuild reproduces the
        # ranks — the ledger's own bookkeeping, never optional (a cycle without it leaves an order no replay can find);
        # counted apart from the derived facts above, since a change the fold itself caused is never one to render on
        arrival_paths, n_arrivals = derive_file_arrival_records(root, conn, facts_now)
        report.arrival_facts += n_arrivals
        # The END pass: this cycle's own refusals and re-judgments become facts in the cycle that made them (the START pass of the
        # next cycle is the backstop for a cycle that dies between here and the fold below)
        end_paths, n_end = derive_refusal_fact_records(root, conn, now, facts_now)
        report.refusal_facts += n_end
        facts = derived + arrival_paths + end_paths
        if facts:
            # the facts share one stamp: ranked in the ledger's own (stamp, sha, uri) order like every file
            work = order_for_ingest(root, [(f, R.file_logical_sha(f)) for f in facts])
            ing.ingest_parsed(iter_parsed(root, work, 0, ITEM_ROWS_PER_TRANSACTION), now)
            if derived:
                ing.reevaluate_blocked(now, stop=over_budget)      # a run_ingest / ruling_state can unblock a waiting record; an arrival fact cannot
        # derived run facts recomputed from the cycle's final state (a pure function of run x eval x decode_policy, no clock):
        # the decode-policy match is copied onto the run row (decode_match.stamp_run_rows); a run record re-folded this cycle
        # carries its producer's `no_policy` placeholder until here
        with DB.transaction(conn):
            stamped = DM.stamp_run_rows(conn)
        if stamped:
            ing._metric({"kind": "decode_policy_match", "stamped": stamped, "utc": now})
        if report.lifecycle_regressions_held:
            # one line per cycle, not per row: how many run_state events would have moved a run backwards and were held
            ing._metric({"kind": "lifecycle_regressions_held", "n": report.lifecycle_regressions_held, "utc": now})
        # the cycle's one dense re-rank, after its last fold and before anything that reads final ranks (the gzip pass's
        # acked_through comparison, paper_head, the status page rendered by the CLI afterwards)
        ing.finalize_ranks(now)
        report.gzipped = gzip_after_ingest(root, conn, gzip_min_age_s)
        _write_paper_head(root, conn)
        # every file this cycle parsed (the harvester's own facts included) enters the sidecar with the stat taken
        # before it was read, so the next cycle hashes nothing that did not change
        if ing.learned:
            stat_cache.update(ing.learned)
        try:
            _save_file_stat(root, stat_cache)
        except OSError:
            pass
        # best effort: fold the WAL back into the database now that the writer pass is over (the PASSIVE
        # autocheckpoint never truncates under continuous readers, and a 5 GB WAL turned every table scan into random
        # reads). Bounded by a short busy wait; a reader holding a snapshot leaves busy = 1 and the cycle succeeds anyway.
        try:
            busy, log_frames, done = DB.wal_checkpoint_truncate(conn)
            ing._metric({"kind": "wal_checkpoint", "busy": busy, "log": log_frames, "checkpointed": done, "utc": now})
        except sqlite3.Error as exc:
            ing._metric({"kind": "wal_checkpoint_error", "error": str(exc), "utc": now})
        # the cycle COMPLETED (a budget stop is a completion with work left): one line on the metrics log and the state file the
        # status page ages; a cycle killed before this point leaves neither, on purpose
        report.elapsed_s = round(clock() - t_cycle0, 3)
        report.completed_utc = R.utc_now()
        ing._metric({"kind": "cycle", "seconds": report.elapsed_s, "stopped": report.stopped, "budget_s": budget_s, "files": report.files,
                     "bundles": report.revisions_applied, "bundles_held": report.revisions_held, "bundles_noop": report.revisions_noop,
                     "bundles_held_skipped": report.revisions_held_skipped,
                     "bundles_deferred": report.revisions_deferred_budget, "plain_files_deferred": report.plain_files_deferred_budget,
                     "transactions": report.transactions, "utc": now})
        write_cycle_state(root, report, now)
    finally:
        if own_conn:
            conn.close()
    # every applied record was the previous render's own page artifact or an arrival fact the fold itself caused, and
    # nothing else moved: the CLI's --html must not re-render on this (is_status_page_artifact; the page's own arrival
    # fact would otherwise re-open the loop that guard closed)
    bookkeeping = report.status_page_records + report.arrival_records
    report.self_change_only = bool(report.changed and bookkeeping
                                   and report.ok == bookkeeping and report.harvester_records == 0
                                   and report.unblocked == 0 and report.reranks == 0 and report.conflicts == 0
                                   and report.revision_transactions == 0 and report.reclaim_transactions == 0)
    return report


def _write_paper_head(root: str, conn: sqlite3.Connection) -> None:
    try:
        head = conn.execute("SELECT ingest_seq FROM v_paper_inputs_head").fetchone()[0]
        with open(os.path.join(state_dir(root), "paper_head"), "w", encoding="utf-8") as fh:
            fh.write(str(head or 0) + "\n")
    except (sqlite3.Error, OSError):
        pass


def pull_clusters(root: str, config: Optional[Mapping[str, object]] = None) -> List[str]:
    """Backstop pulls: ``rsync -rt --ignore-existing <host>:<shared>/registry/records/ inbox/<host>/``
    for every configured cluster (``config.json`` → ``pulls: [{host, path, name}]``). Writes under ``inbox/`` only and takes no
    lock: :func:`ingest_under_lock` runs it BEFORE taking ``harvester.lock`` (every rsync failure is swallowed, as before -- the
    push is the primary transport and the next cycle pulls again)."""
    import subprocess
    config = config or load_config(root)
    out = []
    for p in config.get("pulls", []) or []:
        dest = os.path.join(inbox_dir(root), p["name"]) + "/"
        os.makedirs(dest, exist_ok=True)
        cmd = ["rsync", "-rt", "--ignore-existing", "--timeout=60", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=15",
               f"{p['host']}:{p['path'].rstrip('/')}/", dest]
        try:
            subprocess.run(cmd, check=False, capture_output=True, timeout=600)
            out.append(p["name"])
        except (OSError, subprocess.SubprocessError):
            continue
    return out


def ingest_under_lock(root: str, **kw) -> IngestReport:
    """One harvester cycle under ``<root>/harvester.lock`` -- the CLI's ``registry ingest``. ``pull=True`` runs the cluster pull
    backstops (:func:`pull_clusters`) FIRST and OUTSIDE the lock (2026-09-30: the rsync -- ``--timeout=60`` per transfer, a 600-s
    subprocess bound per cluster -- used to run inside it, after the inbox merge, and what it landed was merged one cycle later).
    The pull writes ``inbox/<host>/`` only (rsync's dot-named temp files, finished by rename; :func:`scan_inbox` skips dot-files),
    reads no database and needs no lock; what it lands is merged and folded under the lock by the cycle that follows it, like any
    pushed file -- so a pulled file now folds in the SAME cycle. The free-space floor still gates the pull: a paused registry
    (below ``min_free``) does not keep rsyncing into the inbox; the check here is read-only (``state/paused.json`` stays the
    cycle's guard's to write). ``lock_timeout_s`` (default 30 s) is how long to wait for the lock before :class:`db.LockHeld`."""
    lock_timeout_s = kw.pop("lock_timeout_s", 30.0)
    if kw.pop("pull", False):
        where = root if os.path.exists(root) else (os.path.dirname(root) or ".")
        if free_bytes(where) >= kw.get("min_free", MIN_FREE_BYTES):
            pull_clusters(root)
    with DB.write_lock(root, timeout_s=lock_timeout_s):
        return ingest_once(root, **kw)


# --------------------------------------------------------------------------------------------
# rebuild + verify
# --------------------------------------------------------------------------------------------

#: Tables / columns excluded from rebuild equality (arrival-order dependent by design).
VERIFY_EXCLUDE_TABLES: Tuple[str, ...] = ("record_delivery", "snapshot", "schema_migration") + tuple(S.STAGING_TABLES)
VERIFY_EXCLUDE_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "record_refusal": ("refused_utc", "judged_with"),      # judged_with: the judging CODE's stamp -- a replay under another tree differs on it; the facts carry it
    "blocked_records": ("first_seen_utc", "last_checked_utc", "attempts", "stale"),
}


LOG_NAMESPACES: Tuple[str, ...] = ("records", "rulings")


def rebuild_root(from_dir: str) -> str:
    """The registry root a rebuild replays: ``from_dir`` itself when it holds ``records/``; its parent when it IS a
    ``records/`` directory (every logical URI is root-relative and ``rulings/`` lives beside, so replaying a bare
    records dir as a root would rename every URI and drop the rulings); anything else is refused."""
    from_dir = os.path.abspath(from_dir)
    if os.path.isdir(os.path.join(from_dir, "records")):
        return from_dir
    if os.path.basename(from_dir) == "records" and os.path.isdir(from_dir):
        return os.path.dirname(from_dir)
    raise ValueError(f"rebuild: {from_dir} is neither a registry root (holding records/) nor a records/ directory")


#: Indexes a replay KEEPS while it loads (every other explicit index is dropped and built after the bulk load, spec
#: "write performance"): the partial index over the withdrawn mark, which ``reclaim`` consults after EVERY activation — a
#: full scan of a table that grows to millions of rows without it, and next to free to maintain (a row enters it only
#: once marked). The ``(source_revision_id, record_file_id)`` indexes are NOT kept: measured on 200k rows, maintaining
#: them cost 25–45 % of the insert rate, and their one per-file consumer (``_clear_staged_file``) no longer scans for a
#: file the ledger has never ranked.
REBUILD_KEEP_INDEXES: Tuple[str, ...] = ("idx_item_result_withdrawn",)


def _remove_db_files(path: str) -> None:
    for suffix in ("", "-wal", "-shm", "-journal"):
        try:
            os.remove(path + suffix)
        except FileNotFoundError:
            pass


def rebuild(from_dir: str, into: str, workers: int = 3, records_per_transaction: int = 20000,
            files: Optional[Collection[str]] = None) -> IngestReport:
    """Replay the whole record log at ``from_dir`` (a registry root, or its ``records/`` directory, resolved to the
    root) into a fresh DB; every log namespace present (``records/``, ``rulings/``) is replayed and named in
    ``report.namespaces``. ``files``: an ALLOWLIST of root-relative logical uris --
    exactly these files are replayed, in the ledger's order, and nothing else on disk: not an unlisted plain file, not an
    unlisted file inside a listed revision directory (``Ingestor.allowlist`` filters every directory listing of the revision
    path; ``report.unlisted_ignored`` counts them), not an unlisted ``file_arrival`` fact (``read_arrival_facts(only=)``). A
    listed file that is not on disk refuses the whole replay by name (``FileNotFoundError``) before anything is written: the
    replica's presence check runs first, so this is a guard, not a path.

    The replay is written to ``<into>.building`` (``db.REBUILD_TMP_SUFFIX``) and renamed to ``into`` only once it is
    complete, indexed and fsynced: ``into`` therefore either does not exist or is a whole database, never a torn one. A
    file nobody reads until that rename needs no WAL and no fsync per commit (``journal_mode=MEMORY``, ``synchronous=OFF``
    — the rollback journal stays, a held revision's activation still rolls back whole); the live database is never
    opened this way. Ranks are made dense ONCE at the end (``Ranker(defer=True)``): the ledger's order is a pure function
    of its keys, and re-ranking per staging window was O(transactions × rows) on a replay whose revision directories
    all sort before the plain files applied first. The keys are ARRIVAL order: before anything is ranked the
    ``file_arrival`` facts are read out of the log (``read_arrival_facts``) and every file takes the arrival the live
    registry recorded for it — its ``first_written_utc`` when there is none — so the replay's ``ingest_seq`` equals
    the live database's on every row. A failure removes the ``.building`` file.

    Phase order: the dimension seed bundles, the producers' plain files, the revision directories (source-major), the blocked
    queue, THEN the harvester's own facts (``records/harvester/``, :func:`partition_harvester_files`) and the queue once more --
    the order a live cycle folds them in. Before it, the facts rode the plain phase and a ``run_ingest`` of a run carried by a
    revision directory blocked on it, then was retried at the closing pass over whatever carriers had landed by then; one whose
    first carrier was itself held until that pass refolded ``ingested`` ahead of every ``judged`` and was refused by
    ``trg_run_state_utc`` (128 records of the R3 bootstrap, 0 live). The order is the first half of the fix; the reducer's
    fold-order evidence rule (``reducer.LifecycleEvidenceMissing``) is the second, so the outcome no longer depends on either."""
    if os.path.exists(into):
        raise FileExistsError(f"rebuild: {into} exists; refusing to overwrite")
    root = rebuild_root(from_dir)
    subdirs = [os.path.join(root, ns) for ns in LOG_NAMESPACES if os.path.isdir(os.path.join(root, ns))]
    tmp = into + DB.REBUILD_TMP_SUFFIX
    _remove_db_files(tmp)                     # an interrupted earlier build
    conn = DB.open_fresh(tmp)
    report = IngestReport()
    report.namespaces = [os.path.basename(sd) for sd in subdirs]
    try:
        conn.execute("PRAGMA journal_mode=MEMORY")
        conn.execute("PRAGMA synchronous=OFF")
        t0 = time.monotonic()
        paths: List[str] = []
        allow: Optional[Set[str]] = None
        if files is not None:
            allow = set(files)
            missing = []
            for uri in sorted(allow):
                path = resolve_uri(root, uri)
                if path is None:
                    missing.append(uri)
                else:
                    paths.append(path)
            if missing:
                raise FileNotFoundError(f"rebuild: {len(missing)} allowlisted file(s) are not on disk under {root}: "
                                        + ", ".join(missing[:5]) + (" …" if len(missing) > 5 else ""))
        else:
            for sd in subdirs:
                if os.path.isdir(sd):
                    paths += list(R.iter_record_files(sd))
        recorded_shas: Dict[str, str] = {}
        ing = Ingestor(conn, root, report, live=False, defer_ranks=True, arrivals=read_arrival_facts(root, only=allow, shas=recorded_shas))
        ing.allowlist = allow
        index_sql = DB.drop_indexes(conn, keep=REBUILD_KEEP_INDEXES)
        # files in log order (peeked first stamp) so chunk transactions approximate global order; within a chunk
        # run-scale records are sorted by order key; the ranks stay provisional and are made dense once, below.
        # Two physical files at one uri (a plain twin beside the .gz): the one whose bytes the arrival fact recorded
        # folds first, so the replay ranks what the live fold ranked and refuses the other as live did
        work = prefer_recorded_twins(root, order_for_ingest(root, [(p, None) for p in paths]), recorded_shas)
        plain, rev_dirs, dim_bundles = split_work(root, work)
        rev_dirs = group_revision_dirs(root, rev_dirs, conn)   # as on the live path: a source's bundles behind its first, the rest no-ops
        # the harvester's own facts fold AFTER the revision phase and its blocked-queue pass (partition_harvester_files): a run_ingest
        # whose run rides a revision directory would otherwise block in the plain phase and be retried over a partial history
        plain, facts = partition_harvester_files(root, plain)
        plain, late = partition_analysis_files(root, plain)          # the analysis filings: after the revision phase and its queue pass
        now = R.utc_now()
        for dim_dir, dim_files in dim_bundles:       # the seed revisions first, whole, as on the live path
            ing.apply_dimension_bundle(dim_dir, dim_files, now)
        for chunk in chunk_parsed(iter_parsed(root, plain, workers, records_per_transaction), FILES_PER_TRANSACTION, records_per_transaction):
            ing.ingest_chunk(chunk, now)
        if workers > 0 and rev_dirs:
            # the revision phase is applied one directory at a time; its item files are parsed ahead in the pool (F16)
            with parse_pool(workers) as ex:
                ing.prefetch = _Prefetcher(root, ex, _Prefetcher.item_files(rev_dirs, root=root, only=allow), workers * PARSE_WINDOW_PER_WORKER)
                try:
                    for d in rev_dirs:
                        ing.apply_revision(d, now)
                finally:
                    ing.prefetch = None
        else:
            for d in rev_dirs:
                ing.apply_revision(d, now)
        # the closing pass over the blocked queue FIRST -- the held revisions' re-attempts, so every carrier that can land has landed --
        # then the harvester's facts in ledger order, then the queue once more for whatever a fact released (a live cycle's own order)
        ing.reevaluate_blocked()
        # the late phase, as on the live path: the analysis filings over the settled parents, then the dependency-refusal retry
        for chunk in chunk_parsed(iter_parsed(root, late, workers, records_per_transaction), FILES_PER_TRANSACTION, records_per_transaction):
            ing.ingest_chunk(chunk, now)
        ing.retry_dependency_refusals(now)
        for chunk in chunk_parsed(iter_parsed(root, facts, workers, records_per_transaction), FILES_PER_TRANSACTION, records_per_transaction):
            ing.ingest_chunk(chunk, now)
        report.parse_s = max(0.0, (time.monotonic() - t0) - report.write_s)
        if facts:
            ing.reevaluate_blocked()
        ing.reclaim(now)
        with DB.transaction(conn):
            ing.ranker.finalize(final=True)
            DB.recreate_indexes(conn, index_sql)
            DM.stamp_run_rows(conn)               # the derived decode-policy match, as at the end of every live cycle
        conn.execute("PRAGMA journal_mode=WAL")   # the mode every connection expects the file in (db.connect)
    except BaseException:
        conn.close()
        _remove_db_files(tmp)
        raise
    conn.close()
    DB.fsync_file_and_dir(tmp)
    os.replace(tmp, into)
    DB.fsync_file_and_dir(into)
    return report


def verify_equal(a: sqlite3.Connection, b: sqlite3.Connection) -> Dict[str, int]:
    """Row-level diff per table (symmetric difference size), excluding the arrival ledger."""
    out: Dict[str, int] = {}
    for t in S.TABLES:
        if t.name in VERIFY_EXCLUDE_TABLES:
            continue
        cols = [c for c in t.column_names if c not in VERIFY_EXCLUDE_COLUMNS.get(t.name, ())]
        sql = f"SELECT {', '.join(cols)} FROM {t.name}"
        ra = set(tuple(_norm(v) for v in r) for r in a.execute(sql).fetchall())
        rb = set(tuple(_norm(v) for v in r) for r in b.execute(sql).fetchall())
        out[t.name] = len(ra ^ rb)
    return out


def _norm(v):
    if isinstance(v, bytes):
        return v.hex()
    if isinstance(v, float):
        return round(v, 12)
    return v


# --------------------------------------------------------------------------------------------
# history (§1.8): run-scale from row_history, item-scale folded from the log
# --------------------------------------------------------------------------------------------


def history(root: str, conn: sqlite3.Connection, table: str, key: Mapping[str, object]) -> List[Dict[str, object]]:
    if table in R.RUN_SCALE_TABLES:
        full_key = {c: key.get(c, S.SENTINELS.get(c)) for c in R.NATURAL_KEYS[table]}
        rows = conn.execute(
            "SELECT written_utc, record_sha256, record_type, record_uri, payload_json, folded_state_json FROM row_history "
            f"WHERE table_name = ? AND natural_key_json = ? ORDER BY {RED.ORDER_SQL}",       # the fold's own order (reducer.ORDER_SQL)
            (table, canonical_json(full_key))).fetchall()
        return [{"written_utc": r[0], "record_sha256": r[1], "record_type": r[2], "record_uri": r[3],
                 "payload": json.loads(r[4]), "state": json.loads(r[5])} for r in rows]
    if table not in R.ITEM_SCALE_TABLES:
        raise ValueError(f"history: unknown table {table!r}")
    run_id = key.get("run_id")
    if not run_id:
        raise ValueError("history: an item-scale key needs run_id")
    files = [r[0] for r in conn.execute("SELECT record_uri FROM record_run_index WHERE run_id = ? ORDER BY record_uri", (run_id,)).fetchall()]
    rt = table
    recs = []
    for f in files:
        path = resolve_uri(root, f)
        if path is None:
            continue
        for line, rec, sha in R.read_record_file(path):
            if rec.get("record_type") != rt:
                continue
            if matches_key(table, rec.get("payload", {}), key):
                recs.append((rec, sha, f"{f}#{line}"))
    return fold_item_events(table, recs)
