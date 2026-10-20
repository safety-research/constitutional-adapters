#!/usr/bin/env python3
"""fold_latency.py -- the latency from a producer's COMMITTED store change to its fold.

    python scripts/registry/measure/fold_latency.py --since ISO [--hours 24 | --until ISO] [--root ~/registry] [--state DIR]
                                                     [--tick-min 10] [--authority FILE] [--out latency.jsonl] [--json]

THE CLOCK starts at ``change_utc`` -- the push tool's record of the producer's committed change (``PUBLISHED-<sha8>.json``; for a carried
component the carry's grading host mtime), carried through the operation's READ RECEIPT (``$STATE/reads/receipts/<op_id>.json``: per adapter the
pins ``{slug: sha8}`` and each pin's PUBLISHED digest) to the revisions the receipt pulled (``revisions_pulled``: a revision directory
``<adapter>/<rev16>`` or an activation bundle ``<adapter>/_activations/<rev16>/<stamp>``) and re-activated (``reactivated``: revision ids
whose activation bundle the grading host wrote from committed bytes). Each (adapter, revision, component pin) is ONE SAMPLE -- RECONCILED ACROSS
THE RECEIPTS of the span: a revision held at one tick and corrected through a bundle a later tick pulled is listed by two receipts and is
one sample, carrying every receipt's op id (``op_ids``), the ways it was listed (``ways``: ``pulled`` / ``bundle`` / ``reactivated``) and
the markers each listing binds it to (``bindings``, below), clocked by the pin's PUBLISHED record (one generation has one record,
whichever receipt carries it). The pin is the RECEIPT's (the operation's record of which generation it read); the marker's
``extra.read_copy`` generation is carried beside it
(``marker_generation``) and a difference is noted (``pin_mismatch``), never a key -- a revision re-activated from committed bytes for a
new pin carries the original read's copy, by construction. The class is the pin's publish ``mode`` (``push`` / ``carry`` / ``move`` /
``adopt`` / ``mirror``): a transitional component (``carry``) is expected to FAIL S7 until its producer pushes, and is reported as
failing, never excluded.

FOLD COMPLETION is a fact about the fold's moment, read from the ledger and NEVER re-read from observation-end state. The harvester
delivers every marker file (``<seq>-source_revision.jsonl``) under ITS OWN ``record_delivery`` row (``_deliver`` upserts by ``file_uri``),
so a revision has one marker row per directory it arrived in: its revision directory and each ``_activations/<rev16>/<stamp>/`` bundle. A
held revision's directory marker is delivered ``blocked`` and STAYS so when a later, correcting bundle activates the revision: the
activation delivers the BUNDLE's marker ``ok`` and sets the shared ``source_revision.accepted = 1`` (harvest.py ``_apply_revision`` /
``_activate_revision``; ``_stand_down_replaced_dirs`` clears ``blocked_records``, never a delivery row). EACH PIN IS BOUND TO THE MARKERS
ITS OWN OPERATIONS LISTED and judged over their deliveries only (``bindings``; the revision's deliveries come from two bounded range
queries, the directory's and its bundles', and the binding selects among them): a ``pulled`` listing binds the revision directory's
marker; a ``bundle`` listing binds that bundle's marker (the path the receipt names); a ``reactivated`` listing -- the receipt names the
revision, not the bundle the grading host wrote -- binds the revision's activation bundles STAMPED BY THAT OPERATION: a bundle directory's
name is ``_compact(now)`` of the pass that wrote it (runner.py ``reactivate_by_id``, every bundle producer), so a bundle is this
operation's when its stamp lies between the operation's attestation stamp (the receipt's ``attestation.utc`` -- the shard's admission
check before its read, the earliest stamp of the operation the receipt carries; the pin's ``change_utc`` when the receipt carries none,
``floor_source`` says which) and the receipt's own ``utc``. A marker the pin is not bound to is REFUSED whenever it was delivered
(``n_markers_unbound`` counts them): an earlier generation's directory ``ok`` landing after this pin's change clock, or an earlier
operation's bundle folded late, completes nothing of this pin (otherwise generation B's pin read completes from generation A's marker
delivered between B's change and B's receipt). The receipt's ``utc`` is deliberately NOT a floor on ``ingested_utc``: the receipt is
written AFTER its own operation's inbox merge and re-activations (reads_protocol ``pull_and_receipt``: pull -> inbox_merge -> receipt)
and the harvester is its own timer, so an operation's own marker is routinely delivered before its receipt -- a floor there would censor
the fastest folds. A sample is COMPLETED when a bound marker's delivery is ``ok`` with ``ingested_utc`` at or after the clock AND the
revision's row has ``accepted = 1``; the completion time is the EARLIEST such ``ok`` (the activation: a later ``ok`` of the same revision
is a re-observation of the head, never a second fold). A revision that folded and was LATER superseded by another successful revision
counts as completed at its own ``ingested_utc`` (and the superseding one at its own); ``superseded_by_revision_id`` and ``reclaimed`` are
never consulted (reducer.py recomputes the first when a later revision folds; ``reclaim`` reclaims accepted superseded revisions).

CENSORED at the observation's end, each by name: no ``ok`` since the clock and the newest delivery since it ``blocked`` (a held revision;
its placeholder row has ``accepted = 0``) or ``partial`` / ``refused`` / ``conflict`` / ``duplicate`` (``marker_<status>``; a stale no-op
bundle's ``duplicate`` yields to any other status); NO delivery of a BOUND marker since the clock by the end (``undelivered`` -- the receipt
listed it and the harvester never folded the listed marker for this change; an unbound marker's delivery, an earlier generation's or an
earlier operation's, counts for nothing here); a receipt entry the operation refused WHOLE -- a refusal and no admitted listing beside
it (``refused_operation``, one row per pin). An entry with admitted listings BESIDE a sibling shard's refusal -- cluster_reads
``pull_window`` withholds only the refusing shard's own publications and still pulls the successful shard's (and may replay a trusted
refused-known revision), while ``receipt_from_plan`` records the listings and the adapter-level ``refused`` in one entry -- keeps its
listings as samples judged as usual (each noting the refusal, ``shard_refusal``) and records the refusal as its own row per pin
(``refused_shards``, the part of the adapter the operation did not read, ``shards`` = ok/total), never in place of them: before this a
marker that very operation produced and the harvester folded ``ok`` was absent from the completed samples. A pin whose
receipt carries no PUBLISHED digest (``no_published_record``: no clock to start); the completing or the newest delivery after the observation's
end (``after_end``); an ``ok`` whose row is not accepted (``not_accepted``, a belt); and, with ``--authority FILE``
(``state/store_authority.json``, `check-authority`'s record), a component whose ``current`` generation has no receipt pinning it by the
end (``no_receipt``). Window reads are receipts too, so a window's pull counts like a tick's.

The verdict: ``p50 <= N + 10 min`` and ``p90 <= 2N + 10 min`` over the completed samples (``--tick-min N``, the tick's period, default
10), per class and overall; the censored counts by reason; N recorded. The database is opened READ-ONLY (``file:...?mode=ro``) and
every query is bounded by a revision id or a file-URI prefix range -- never ``row_history``, never a whole-table scan. ``--out`` appends
every sample as one JSON line (``$SPEC/measure/latency.jsonl``). Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_STATE = os.path.expanduser(os.environ.get("REGISTRY_LANDING_STATE") or "~/.local/state/registry-landing")
DEFAULT_ROOT = os.path.expanduser(os.environ.get("REGISTRY_ROOT") or "~/registry")
RECEIPT_SCHEMA = "reads_receipt/1"
AUTHORITY_STATE_SCHEMA = "store_authority_state/1"
MARKER_TOKEN = "source_revision"
#: a receipt's ``revisions_pulled`` entry: ``<adapter>/<rev16>`` (a revision directory) or ``<adapter>/_activations/<rev16>/<stamp>`` (an
#: activation bundle) -- cluster_reads._revision_dirs enumerates both forms
PULLED_DIR_RE = re.compile(r"^(?P<adapter>[^/]+)/(?:_activations/(?P<brev>[0-9a-f]{16})/[^/]+|(?P<rev>[0-9a-f]{16}))/?$")
#: an activation bundle's marker URI: ``records/adapters/<adapter>/_activations/<rev16>/<stamp>/<file>``; the stamp is ``_compact(now)`` of
#: the pass that wrote the bundle (runner.py: every bundle producer), the digits of a registry UTC stamp (``YYYYMMDDHHMMSSmmm``, 17 digits)
BUNDLE_URI_RE = re.compile(r"^records/adapters/(?P<adapter>[^/]+)/_activations/(?P<rev>[0-9a-f]{16})/(?P<stamp>[^/]+)/")
#: every record_delivery status, and the ones that complete / censor a sample
COMPLETING_STATUS = "ok"
CENSOR_STATUSES = ("blocked", "partial", "refused", "conflict", "duplicate")
#: a stale no-op bundle's status: says nothing about the revision's own fold, so it yields to any other delivery when censoring
UNINFORMATIVE_STATUS = "duplicate"
#: S7's bounds as a function of the tick period N (minutes)
P50_BOUND_MIN = lambda n: n + 10          # noqa: E731
P90_BOUND_MIN = lambda n: 2 * n + 10      # noqa: E731


def parse_utc(s: Optional[str]) -> Optional[datetime]:
    """The registry's stamps: ``YYYY-MM-DDTHH:MM:SS[.mmm]Z``; None for anything else."""
    if not s or not isinstance(s, str):
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%MZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z" if dt else None


def receipts_in(rd: str, since: datetime, until: datetime) -> Tuple[List[Dict[str, Any]], int]:
    """The receipts whose ``utc`` lies in [since, until], oldest first, and the count of unreadable / foreign files."""
    d = os.path.join(rd, "receipts")
    out: List[Dict[str, Any]] = []
    bad = 0
    if not os.path.isdir(d):
        return out, 0
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, fn), "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            bad += 1
            continue
        if not isinstance(doc, dict) or doc.get("schema") != RECEIPT_SCHEMA:
            bad += 1
            continue
        t = parse_utc(doc.get("utc"))
        if t is None or t < since or t > until:
            continue
        out.append(doc)
    out.sort(key=lambda r: str(r.get("utc") or ""))
    return out, bad


def open_ro(db_path: str) -> sqlite3.Connection:
    uri = "file:" + os.path.abspath(db_path) + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def _prefix_range(prefix: str) -> Tuple[str, str]:
    """``file_uri >= prefix AND file_uri < prefix + chr(0x10FFFF)``: the primary-key index serves it (a LIKE would not under NOCASE)."""
    return prefix, prefix + "\U0010ffff"


def marker_deliveries(conn: sqlite3.Connection, adapter: str, rev: str) -> List[Dict[str, Any]]:
    """EVERY marker delivery row of a revision: its directory's (``records/adapters/<adapter>/<rev>/<seq>-source_revision.jsonl``) and each
    activation bundle's (``records/adapters/<adapter>/_activations/<rev>/<stamp>/<seq>-source_revision.jsonl``), in file-URI order. The
    harvester delivers each marker under its own row, so a held directory's ``blocked`` row and the correcting bundle's ``ok`` row stand
    side by side; the judgment (:func:`_judge`) reads them together. Two bounded range queries on the primary key."""
    out: List[Dict[str, Any]] = []
    for prefix in (f"records/adapters/{adapter}/{rev}/", f"records/adapters/{adapter}/_activations/{rev}/"):
        lo, hi = _prefix_range(prefix)
        rows = conn.execute("SELECT file_uri, status, ingested_utc, arrived_utc, refusal_reason FROM record_delivery WHERE file_uri >= ? AND file_uri < ? "
                            "ORDER BY file_uri", (lo, hi)).fetchall()
        out.extend(dict(r) for r in rows if MARKER_TOKEN in os.path.basename(str(r["file_uri"])))
    return out


def revision_row(conn: sqlite3.Connection, rev: str) -> Optional[Dict[str, Any]]:
    r = conn.execute("SELECT revision_id, source_id, adapter, observed_utc, accepted, refusal, extra_json FROM source_revision WHERE revision_id = ?",
                     (rev,)).fetchone()
    return dict(r) if r else None


def read_copy_generations(extra_json: Optional[str]) -> Dict[str, str]:
    """``{slug: generation sha8}`` the marker's ``extra.read_copy.components`` recorded (which copy of each component the shard read)."""
    try:
        extra = json.loads(extra_json or "{}")
    except ValueError:
        return {}
    comps = ((extra.get("read_copy") or {}).get("components") or {}) if isinstance(extra, dict) else {}
    out: Dict[str, str] = {}
    for slug, v in comps.items():
        if isinstance(v, dict) and v.get("generation"):
            out[str(slug)] = str(v["generation"])
    return out


def _sample(**kw: Any) -> Dict[str, Any]:
    base = {"op_id": None, "op_ids": [], "owner": None, "receipt_utc": None, "adapter": None, "revision": None, "how": None, "ways": [], "component": None,
            "generation": None, "marker_generation": None, "class": None, "change_utc": None, "published_utc": None, "status": None, "accepted": None,
            "ingested_utc": None, "marker_uri": None, "bindings": [], "n_markers": 0, "n_markers_unbound": 0, "shard_refusal": None, "latency_s": None,
            "outcome": None, "censor": None}
    base.update(kw)
    return base


def listed_revisions(entry: Dict[str, Any]) -> List[Tuple[str, str, Optional[str]]]:
    """The revisions one receipt entry lists, each with the WAY and the PATH the receipt names: ``pulled`` (a revision directory
    ``<adapter>/<rev16>`` under ``revisions_pulled``), ``bundle`` (an ``<adapter>/_activations/<rev16>/<stamp>`` directory under
    ``revisions_pulled``: a corrected re-read or a re-activation the cluster side wrote), ``reactivated`` (``reactivated``: the grading host wrote
    the bundle from committed bytes; the receipt names the revision only, so the path is None and the binding is by stamp). An entry of
    neither form is skipped."""
    out: List[Tuple[str, str, Optional[str]]] = []
    for d in entry.get("revisions_pulled") or []:
        m = PULLED_DIR_RE.match(str(d))
        if m is None:
            continue
        out.append((m.group("brev"), "bundle", str(d).rstrip("/")) if m.group("brev") else (m.group("rev"), "pulled", str(d).rstrip("/")))
    out += [(str(r), "reactivated", None) for r in (entry.get("reactivated") or [])]
    return out


def bundle_stamp(file_uri: str) -> Optional[datetime]:
    """When the bundle whose marker ``file_uri`` is was WRITTEN: its directory name, ``_compact(now)`` of the writing pass (17 digits
    ``YYYYMMDDHHMMSSmmm``; 14 without the milliseconds). None for a revision directory's marker or a bundle whose name is not a stamp."""
    m = BUNDLE_URI_RE.match(file_uri)
    if m is None:
        return None
    digits = "".join(ch for ch in m.group("stamp") if ch.isdigit())
    if digits != m.group("stamp") or len(digits) not in (14, 17):
        return None
    try:
        if len(digits) == 17:
            return datetime.strptime(digits + "000", "%Y%m%d%H%M%S%f").replace(tzinfo=timezone.utc)
        return datetime.strptime(digits, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def binding(adapter: str, rev: str, way: str, listed: Optional[str], op_id: Optional[str], rec: Dict[str, Any],
            change_utc: Optional[str]) -> Dict[str, Any]:
    """What one receipt listing binds a pin to (the module doc's rule): the marker-URI prefix the deliveries must carry -- the revision
    directory's for ``pulled``, the listed bundle's for ``bundle``, the revision's ``_activations/`` for ``reactivated`` -- and, for a
    ``reactivated`` listing, the stamp interval that makes a bundle THIS operation's: from the receipt's ``attestation.utc`` (the shard's
    admission check before its read; the pin's ``change_utc`` when the receipt carries none -- ``floor_source`` records which) to the
    receipt's ``utc`` (the bundle was written before the receipt)."""
    if way == "pulled":
        prefix = f"records/adapters/{adapter}/{rev}/"
    elif way == "bundle":
        prefix = f"records/adapters/{listed}/"
    else:
        prefix = f"records/adapters/{adapter}/_activations/{rev}/"
    b: Dict[str, Any] = {"op_id": op_id, "way": way, "listed": listed, "marker_prefix": prefix}
    if way == "reactivated":
        att = parse_utc(((rec.get("attestation") or {}) if isinstance(rec.get("attestation"), dict) else {}).get("utc"))
        b.update(stamp_floor=(_iso(att) if att is not None else change_utc), floor_source=("attestation" if att is not None else "change_utc"),
                 stamp_ceiling=rec.get("utc"))
    return b


def bound_deliveries(bindings: List[Dict[str, Any]], deliveries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The revision's marker deliveries a sample is bound to, in file-URI order: a delivery whose URI carries a binding's prefix -- and,
    for a ``reactivated`` binding, whose bundle stamp lies in the binding's interval (a bundle with no readable stamp is no operation's
    re-activation). Everything else is REFUSED, whenever it was delivered."""
    out: List[Dict[str, Any]] = []
    seen = set()
    for d in deliveries:
        uri = str(d.get("file_uri"))
        for b in bindings:
            if not uri.startswith(str(b.get("marker_prefix"))):
                continue
            if b.get("way") == "reactivated":
                st = bundle_stamp(uri)
                lo, hi = parse_utc(b.get("stamp_floor")), parse_utc(b.get("stamp_ceiling"))
                if st is None or (lo is not None and st < lo) or (hi is not None and st > hi):
                    continue
            if uri not in seen:
                seen.add(uri)
                out.append(d)
            break
    return out


def samples_for_receipts(conn: Optional[sqlite3.Connection], recs: List[Dict[str, Any]], until: datetime) -> Tuple[List[Dict[str, Any]], int]:
    """Every sample of the span's receipts (oldest first), completed or censored by name (the module doc's rules), and the number of
    receipt LISTINGS they were reconciled from. One sample per (adapter, revision, component pin): a revision two receipts list -- held at
    one tick, its correcting bundle pulled by the next -- is one sample carrying both op ids and both listings' bindings, judged ONCE over
    the deliveries of the markers it is bound to (:func:`binding`, :func:`bound_deliveries`), never over another listing's. A refusal is a
    censored row of its own per pin, per receipt: ``refused_operation`` when the entry lists nothing beside it (the adapter refused whole),
    ``refused_shards`` when admitted listings stand beside it -- those listings are judged as usual, the refusal never replaces them."""
    out: List[Dict[str, Any]] = []
    keyed: Dict[Tuple[str, str, Optional[str], Optional[str]], Dict[str, Any]] = {}
    rows: Dict[str, Optional[Dict[str, Any]]] = {}
    listings = 0
    for rec in recs:
        op = rec.get("op") or {}
        op_id = op.get("op_id")
        for adapter, e in sorted((rec.get("adapters") or {}).items()):
            pins: Dict[str, str] = {str(k): str(v) for k, v in (e.get("pins") or {}).items()}
            pub: Dict[str, Any] = e.get("published") or {}
            common = {"op_id": op_id, "owner": op.get("owner"), "receipt_utc": rec.get("utc"), "adapter": adapter}
            refused = [str(r) for r in (e.get("refused") or [])] or None
            listings_here = listed_revisions(e)
            if refused:
                # the operation refused the adapter WHOLE (nothing listed beside the refusal): nothing it would have folded counts, one
                # censored row per pin so the component is on the record. A refusal BESIDE admitted listings (a sibling shard refused while
                # another shard's publication was pulled, or a trusted refused-known revision replayed): the listings are judged below as
                # usual and the refusal is its own row per pin -- the part of the adapter this operation did not read -- never in place of
                # them (an early `continue` here dropped a marker this very operation produced and the harvester folded ok)
                censor = "refused_operation" if not listings_here else "refused_shards"
                shards = f"{e.get('shards_ok')}/{e.get('shards_total')}" if e.get("shards_total") is not None else None
                for slug, sha8 in sorted(pins.items()) or [(None, None)]:
                    p = pub.get(slug) if slug else None
                    out.append(_sample(**common, op_ids=[op_id], component=slug, generation=sha8, **_clock(p), outcome="censored", censor=censor,
                                       shard_refusal=refused, shards=shards))
            for rev, way, listed in listings_here:
                if rev not in rows:
                    rows[rev] = revision_row(conn, rev) if conn is not None else None
                row = rows[rev]
                gens = read_copy_generations(row.get("extra_json") if row else None) if row else {}
                # the receipt's pins ARE the unit: the operation's record of which generation it read (and the clock's PUBLISHED record is
                # that pin's); the marker's read_copy stands beside each as marker_generation
                comps: Dict[Optional[str], Optional[str]] = dict(pins) or {None: None}
                for slug, sha8 in sorted(comps.items(), key=lambda kv: str(kv[0])):
                    listings += 1
                    key = (adapter, rev, slug, sha8)
                    p = pub.get(slug) if slug else None
                    clock = _clock(p)
                    b = binding(adapter, rev, way, listed, op_id, rec, clock.get("change_utc"))
                    smp = keyed.get(key)
                    if smp is not None:
                        if op_id not in smp["op_ids"]:
                            smp["op_ids"].append(op_id)
                        if way not in smp["ways"]:
                            smp["ways"] = sorted(smp["ways"] + [way])
                        if smp.get("change_utc") is None and isinstance(p, dict):
                            smp.update(**clock)
                            for prior in smp["bindings"]:        # a floor that waited for the clock takes it now
                                if prior.get("way") == "reactivated" and prior.get("floor_source") == "change_utc" and prior.get("stamp_floor") is None:
                                    prior["stamp_floor"] = clock.get("change_utc")
                        if b.get("way") == "reactivated" and b.get("floor_source") == "change_utc" and b.get("stamp_floor") is None:
                            b["stamp_floor"] = smp.get("change_utc")
                        if not any(x.get("op_id") == op_id and x.get("marker_prefix") == b["marker_prefix"] for x in smp["bindings"]):
                            smp["bindings"].append(b)
                        if refused and smp.get("shard_refusal") is None:
                            smp["shard_refusal"] = refused
                        continue
                    smp = _sample(**common, op_ids=[op_id], revision=rev, how=way, ways=[way], component=slug, generation=sha8,
                                  marker_generation=(gens.get(slug) if slug else None), **clock, accepted=(row or {}).get("accepted"), bindings=[b],
                                  shard_refusal=refused)
                    if slug is not None and sha8 is not None and gens.get(slug) not in (None, sha8):
                        smp["pin_mismatch"] = f"receipt pin {sha8} != marker generation {gens.get(slug)}"
                    keyed[key] = smp
                    out.append(smp)
    deliveries: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for (adapter, rev, _slug, _sha8), smp in keyed.items():
        if (adapter, rev) not in deliveries:
            deliveries[(adapter, rev)] = marker_deliveries(conn, adapter, rev) if conn is not None else []
        bound = bound_deliveries(smp["bindings"], deliveries[(adapter, rev)])
        smp["n_markers_unbound"] = len(deliveries[(adapter, rev)]) - len(bound)
        _judge(smp, bound, until)
    return out, listings


def _clock(p: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(p, dict):
        return {"class": None, "change_utc": None, "published_utc": None}
    return {"class": p.get("mode"), "change_utc": p.get("change_utc"), "published_utc": p.get("published_utc")}


def _judge(smp: Dict[str, Any], deliveries: List[Dict[str, Any]], until: datetime) -> None:
    """Completed or censored, by the module doc's rule, over the deliveries of the markers the sample is BOUND to (:func:`bound_deliveries`:
    the directory's or the bundles' its own operations listed, never an earlier generation's); sets ``status`` / ``ingested_utc`` /
    ``marker_uri`` to the delivery the verdict rests on and ``latency_s`` for a completed sample. Only deliveries at or after the clock
    (``change_utc``) count: an earlier ``ok`` is an earlier fold of the same revision."""
    smp["n_markers"] = len(deliveries)
    t0 = parse_utc(smp.get("change_utc"))
    if t0 is None:
        smp.update(outcome="censored", censor="no_published_record")
        return
    since_clock = [(t, d) for d in deliveries for t in [parse_utc(d.get("ingested_utc"))] if t is not None and t >= t0]
    ok = sorted(((t, d) for t, d in since_clock if d.get("status") == COMPLETING_STATUS), key=lambda td: (td[0], str(td[1].get("file_uri"))))
    if ok:
        t1, d = ok[0]          # the activation: the earliest ok; a later ok of the same revision re-observes the head
        smp.update(status=COMPLETING_STATUS, ingested_utc=d.get("ingested_utc"), marker_uri=d.get("file_uri"))
        if t1 > until:
            smp.update(outcome="censored", censor="after_end")
            return
        if int(smp.get("accepted") or 0) != 1:
            smp.update(outcome="censored", censor="not_accepted")
            return
        smp.update(outcome="completed", censor=None, latency_s=round((t1 - t0).total_seconds(), 3))
        return
    if since_clock:
        informative = [td for td in since_clock if td[1].get("status") != UNINFORMATIVE_STATUS] or since_clock
        t1, d = max(informative, key=lambda td: (td[0], str(td[1].get("file_uri"))))
        st = str(d.get("status"))
        smp.update(status=st, ingested_utc=d.get("ingested_utc"), marker_uri=d.get("file_uri"))
        smp.update(outcome="censored", censor=("after_end" if t1 > until else f"marker_{st}"))
        return
    smp.update(status=None, ingested_utc=None, marker_uri=None, outcome="censored", censor="undelivered")


def authority_censoring(path: Optional[str], receipts: List[Dict[str, Any]], until: datetime) -> List[Dict[str, Any]]:
    """With `check-authority`'s record: a component whose ``current`` generation no receipt in the span pinned by the end -- its committed
    change never reached a fold (``no_receipt``); one censored sample per such component, clocked at the record's ``change_utc``."""
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return [_sample(outcome="censored", censor="authority_unreadable", component=path)]
    if not isinstance(doc, dict) or doc.get("schema") != AUTHORITY_STATE_SCHEMA:
        return [_sample(outcome="censored", censor="authority_foreign_schema", component=path)]
    pinned = set()
    for rec in receipts:
        for e in (rec.get("adapters") or {}).values():
            for slug, sha8 in (e.get("pins") or {}).items():
                pinned.add((str(slug), str(sha8)))
    out = []
    for slug, row in sorted((doc.get("components") or {}).items()):
        cur = (row or {}).get("current")
        if not cur:
            continue
        if (str(slug), str(cur)) in pinned:
            continue
        chg = parse_utc((row or {}).get("change_utc"))
        if chg is not None and chg > until:
            continue
        out.append(_sample(component=slug, generation=cur, change_utc=(row or {}).get("change_utc"), published_utc=(row or {}).get("published_utc"),
                           **{"class": (row or {}).get("publish_mode")}, outcome="censored", censor="no_receipt"))
    return out


def percentile(vals: List[float], q: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 3)


def summarize(samples: List[Dict[str, Any]], tick_min: float) -> Dict[str, Any]:
    done = [s for s in samples if s.get("outcome") == "completed"]
    cens = [s for s in samples if s.get("outcome") == "censored"]
    by_reason: Dict[str, int] = {}
    for s in cens:
        by_reason[str(s.get("censor"))] = by_reason.get(str(s.get("censor")), 0) + 1

    def block(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        lat = [float(s["latency_s"]) / 60.0 for s in rows if s.get("latency_s") is not None]
        p50, p90 = percentile(lat, 0.5), percentile(lat, 0.9)
        return {"n": len(lat), "p50_min": p50, "p90_min": p90, "max_min": round(max(lat), 3) if lat else None,
                "p50_bound_min": P50_BOUND_MIN(tick_min), "p90_bound_min": P90_BOUND_MIN(tick_min),
                "s7": (None if not lat else ("pass" if (p50 is not None and p50 <= P50_BOUND_MIN(tick_min) and p90 is not None and p90 <= P90_BOUND_MIN(tick_min))
                                             else "fail"))}
    classes = sorted({str(s.get("class")) for s in done})
    components = sorted({str(s.get("component")) for s in done})
    return {"N": len(samples), "completed": len(done), "censored": len(cens), "censored_by_reason": by_reason, "tick_min": tick_min,
            "overall": block(done), "by_class": {c: block([s for s in done if str(s.get("class")) == c]) for c in classes},
            "by_component": {c: block([s for s in done if str(s.get("component")) == c]) for c in components},
            "censored_by_class": {c: sum(1 for s in cens if str(s.get("class")) == c) for c in sorted({str(s.get("class")) for s in cens})}}


def measure(root: str, state: str, since: datetime, until: datetime, tick_min: float, authority: Optional[str] = None,
            db_path: Optional[str] = None) -> Dict[str, Any]:
    rd = os.path.join(state, "reads")
    recs, bad = receipts_in(rd, since, until)
    db = db_path or os.path.join(root, "registry.sqlite")
    samples: List[Dict[str, Any]] = []
    db_error = None
    conn: Optional[sqlite3.Connection] = None
    if recs:
        try:
            conn = open_ro(db)
        except sqlite3.Error as exc:
            db_error = f"{type(exc).__name__}: {exc}"
    listings = 0
    try:
        samples, listings = samples_for_receipts(conn, recs, until)
    finally:
        if conn is not None:
            conn.close()
    samples.extend(authority_censoring(authority, recs, until))
    if db_error:
        # the fold's fact is UNKNOWN, not absent: every revision sample the ledger would have judged (``undelivered`` is what a None
        # connection leaves) reads database_unreadable; a sample with no clock stays no_published_record (no database would start it)
        for s in samples:
            if s.get("revision") and (s.get("outcome") != "censored" or s.get("censor") == "undelivered"):
                s.update(outcome="censored", censor="database_unreadable", latency_s=None)
    return {"since": _iso(since), "until": _iso(until), "root": root, "db": db, "reads_dir": rd, "receipts": len(recs), "receipts_unreadable": bad,
            "receipt_listings": listings, "db_error": db_error, "summary": summarize(samples, tick_min), "samples": samples}


def render(t: Dict[str, Any]) -> str:
    s = t["summary"]
    o = s["overall"]
    lines = [f'fold latency {t["since"]} -> {t["until"]}: {t["receipts"]} receipt(s) ({t["receipts_unreadable"]} unreadable), '
             f'{t.get("receipt_listings", 0)} revision listing(s) reconciled to N = {s["N"]} sample(s): '
             f'{s["completed"]} completed, {s["censored"]} censored {s["censored_by_reason"] or ""}',
             f'overall (tick N = {s["tick_min"]} min): n {o["n"]}, p50 {o["p50_min"]} min (bound {o["p50_bound_min"]}), p90 {o["p90_min"]} min '
             f'(bound {o["p90_bound_min"]}), max {o["max_min"]} -> S7 {o["s7"] or "no completed sample"}']
    if t.get("db_error"):
        lines.append(f'DATABASE UNREADABLE: {t["db_error"]} -- every revision sample censored (database_unreadable)')
    lines.append("| class | n | p50 min | p90 min | max min | S7 | censored |")
    lines.append("|---|---|---|---|---|---|---|")
    for c, b in s["by_class"].items():
        lines.append(f'| {c} | {b["n"]} | {b["p50_min"]} | {b["p90_min"]} | {b["max_min"]} | {b["s7"]} | {s["censored_by_class"].get(c, 0)} |')
    for c, n in s["censored_by_class"].items():
        if c not in s["by_class"]:
            lines.append(f'| {c} | 0 | — | — | — | — | {n} |')
    lines.append("| component | n | p50 min | p90 min | max min | S7 |")
    lines.append("|---|---|---|---|---|---|")
    for c, b in s["by_component"].items():
        lines.append(f'| {c} | {b["n"]} | {b["p50_min"]} | {b["p90_min"]} | {b["max_min"]} | {b["s7"]} |')
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", required=True, help="the observation's start (UTC ISO)")
    ap.add_argument("--hours", type=float, default=24.0, help="the observation's length (default 24)")
    ap.add_argument("--until", help="the observation's end (UTC ISO; default --since + --hours)")
    ap.add_argument("--root", default=DEFAULT_ROOT, help=f"the registry root (default {DEFAULT_ROOT}); its registry.sqlite is opened read-only")
    ap.add_argument("--db", help="the database to read instead of <root>/registry.sqlite (a replica copy)")
    ap.add_argument("--state", default=DEFAULT_STATE, help=f"REGISTRY_LANDING_STATE (default {DEFAULT_STATE})")
    ap.add_argument("--tick-min", type=float, default=10.0, help="N, the tick's period in minutes (default 10)")
    ap.add_argument("--authority", help="state/store_authority.json: components whose current generation no receipt pinned are censored no_receipt")
    ap.add_argument("--out", help="append every sample as one JSON line to this file")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    since = parse_utc(a.since)
    if since is None:
        print(f"--since: not a UTC stamp: {a.since!r}", file=sys.stderr)
        return 2
    until = parse_utc(a.until) if a.until else since + timedelta(hours=a.hours)
    if until is None:
        print(f"--until: not a UTC stamp: {a.until!r}", file=sys.stderr)
        return 2
    t = measure(a.root, a.state, since, until, a.tick_min, a.authority, a.db)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "a", encoding="utf-8") as fh:
            for s in t["samples"]:
                fh.write(json.dumps(dict(s, since=t["since"], until=t["until"]), sort_keys=True) + "\n")
    if a.json:
        print(json.dumps(t, indent=1, default=str))
    else:
        sys.stdout.write(render(t))
    return 0


if __name__ == "__main__":
    sys.exit(main())
