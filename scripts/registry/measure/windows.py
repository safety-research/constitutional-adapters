#!/usr/bin/env python3
"""windows.py — the landing-window timing table from the chain logs' stage stamps (the homework's Q1.2 method).

    python scripts/registry/measure/windows.py [LOG ...] [--since 2026-09-24] [--json]

Reads ``~/.local/state/registry-landing/windows/chain.log`` and ``/tmp/reg-migrate-chain.log`` by default (both when present; a
window is de-duplicated by its start stamp, so the compat copy window.sh appends is counted once). A window opens at
``=== ff chain start <ISO>`` (or ``=== migrate chain start``), closes at ``=== ... chain end <ISO>``; a stage stamp is a line
``-- <label> HH:MM:SS`` / ``--- <label> HH:MM:SS`` (the time may sit before a parenthesis, as in ``-- dimensions load HH:MM:SS (key)``),
plus ``base at <sha> HH:MM:SS`` and ``writers quiet HH:MM:SS``. The stamp is printed BEFORE its stage runs, so a stage's wall =
the next stamp − its own; the last stage ends at the end marker. A window with no end marker is ABORTED (its total = last stamp − start).

THE END MARKER'S DEFINITION: window.sh writes ``=== ff chain end`` from its EXIT trap, AFTER
every mandatory stage ran and the timer restore was VERIFIED (``timers restored ...`` precedes it) -- so the measured window includes the
restore -- and ``=== ff chain ABORTED (<cause>) <ISO>`` otherwise (a failed restore, a refused adapter, any earlier abort). Historical
logs wrote the end marker BEFORE the restore; such a window is marked ``end_definition = "pre-rev3"`` (the restore line, when present,
sits after the end and is never seen inside the window), a window whose end follows a ``timers restored`` / ``timers NOT restored`` /
``timers LEFT STOPPED`` line ``"rev3"``, so the baseline and the post-cutover windows are never silently compared under two definitions.
``--- ingest end`` is a stage stamp of its own (class ``ingest_end``: the ingest's end to the trap's restore); ``-- reads reservation``
(the window's first act, the reservation taken and the reads-*/attest-* jobs reconciled) is class ``reservation``
and ``-- uninstall units`` (the retired unit files removed after the install) joins ``install``; the trap's
``-- status heavy refresh started once`` stamp now sits INSIDE the window (class ``status_heavy``: the cache initializer's start to the
end marker); ``-- adapter A REFUSED HH:MM:SS (reason)`` is classed ``adapter_refused``; an ABORTED window carries ``abort_cause``.

Prints per window: start, base sha, adapters, total min, the two longest stages; then the summary over completed windows
(n, median, mean, max, min, sum) and the by-stage-class table (adapter reads / dimensions load / final ingest / migrate /
the writers-quiet wait / install / the cluster stages / the skipped cluster block / the store mirror / the acceptance before-phase / the store authority stage / other). Validated against the homework: windows since 2026-09-24 in the pre-repo log
give n = 25 completed, median 16.8 min -- TODAY that baseline is ``--since 2026-09-24 --until 2026-09-28T22:00:00Z`` (the homework
measured on 09-28 before the 22:29Z window; the logs have kept growing since), every window ``pre-rev3``.

S1'S FIRST-TEN RULE AND THE CHANGED-SET COLUMNS: ``--first N`` keeps the first N COMPLETED windows
at or after ``--since`` (``=== ff chain end`` written; an ABORTED window in the span does not count and is listed with its cause under
``summary.aborted_in_span``), so ``--since <cutover ISO> --first 10 --json`` is the acceptance read. The aborts listed are every abort of
the span up to the Nth completion's start once that completion EXISTS, and -- while fewer than N windows have completed, the live read
before the tenth -- every abort of the span through ``--until`` or the latest observed window, an abort after the latest completion
included (it would otherwise vanish until another window completed). ``--until ISO`` bounds the span (windows starting before it). Per window the CHANGED SET -- what the operator computed with ``cluster-reads changed`` and passed as
``ADAPTERS=`` on the start line (``changed_set_n``) -- is joined with the window's READ RECEIPT (``--receipts DIR``, default
``$REGISTRY_LANDING_STATE/reads/receipts``; the window's op id is the ``-- reads reservation HH:MM:SS (op <id>, ...)`` stamp, C7): adapters
read whole / refused, revisions pulled, re-activations, the newest ``change_utc`` among the pins, and the pins' INPUT BYTES resolved from the
grading host manifests (``--root``'s ``state/store_manifests/<slug>/MANIFEST-<sha8>.json``; None when no manifest is on this host). ``records_folded``
/ ``files_folded`` are read from the harvester's summary lines printed inside the ingest stage when the log carries them (``window.sh``
tails the summary, so they may be absent: None, never guessed). ``drain_s`` is the measured drain (start -> ``writers quiet``, the same
number ``quiet_wait_s`` reports). The summary adds ``subpopulations`` (windows with a cluster read / without; by changed-set size) so the
sub-populations' medians stand beside the ten, and the receipts joined / unjoined counts.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

DEFAULT_LOGS = [os.path.expanduser("~/.local/state/registry-landing/windows/chain.log"), "/tmp/reg-migrate-chain.log"]
START_RE = re.compile(r"^=== (?:ff|migrate) chain start (\S+)(?:\s+ADAPTERS=(.*))?\s*$")
END_RE = re.compile(r"^=== (?:ff|migrate) chain end (\S+)")
ABORT_RE = re.compile(r"^=== (?:ff|migrate) chain ABORTED \((.*)\) (\S+)\s*$")
RESTORE_RE = re.compile(r"^timers (?:restored|NOT restored|LEFT STOPPED|never stopped)")
ADAPTER_REFUSED_RE = re.compile(r"^adapter \S+(?: \S+)* REFUSED\b", re.IGNORECASE)
TIME_RE = re.compile(r"(?<![\d:])(\d\d):(\d\d):(\d\d)(?![\d:])")
STAGE_RE = re.compile(r"^(?:--- |-- )(.*)$")
BASE_RE = re.compile(r"^base (?:at|HEAD) (\S+)")
QUIET_RE = re.compile(r"^writers quiet ")
#: `-- reads reservation HH:MM:SS (op <WINDOW_ID>, holder pid N)` (C7): the window's op id, the key its read receipt is filed under
RESERVATION_OP_RE = re.compile(r"\(op (\S+?),")
#: the harvester's summary keys printed inside the ingest stage (`$REG ingest | tail -4` -- present only when the tail reaches them)
INGEST_KEY_RE = re.compile(r'^\s*"(files|records)":\s*(\d+)')
RECEIPT_SCHEMA = "reads_receipt/1"
DEFAULT_RECEIPTS = os.path.join(os.path.expanduser(os.environ.get("REGISTRY_LANDING_STATE") or "~/.local/state/registry-landing"), "reads", "receipts")
DEFAULT_ROOT = os.path.expanduser(os.environ.get("REGISTRY_ROOT") or "~/registry")


def parse_iso(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


#: stage-label prefix → class; the first match wins (the skipped dimensions load before the load itself; the skipped cluster block --
#: one stamped line when the window names no adapter with a cluster policy -- before the cluster stages, which are the
#: reader-state export, the tree export and every `cluster reads <stage>`; the store mirror is its own class: it runs after the timers
#: stop and before the drain, so its wall overlaps the writers-quiet wait; the migration plan's READ -- and the operator's approval wait inside it -- is
#: `migrate_plan`, before `migrate` so the apply's wall stays its own; the reconciliation slot's dry run and apply are `reconcile`;
#: `ingest end`, `uninstall units` (under `install`), `reads reservation` and the trap's `status heavy refresh` are classed too;
#: `config convert` is the configuration conversion before the fast-forward; the acceptance before-phase is `acceptance`; the store authority stage is `authority`;
#: the lifecycle restamp slot's plan and apply (WINDOW_RESTAMP_KEYS) are `restamp`)
STAGE_CLASSES = (("adapter ", "adapter"), ("dimensions load skipped", "dims_skip"), ("dimensions load", "dims"),
                 ("migrate (plan)", "migrate_plan"), ("migrate", "migrate"), ("reconcile", "reconcile"), ("restamp keys", "restamp"),
                 ("ingest end", "ingest_end"), ("ingest", "ingest"), ("install units", "install"), ("uninstall units", "install"),
                 ("reads reservation", "reservation"), ("wal checkpoint", "wal"), ("config convert", "config_convert"), ("roster load", "roster"), ("base at", "base"),
                 ("base head", "base"), ("writers quiet", "quiet"), ("cluster reads skipped", "cluster_skip"), ("cluster reads", "cluster"),
                 ("reader-state export", "cluster"), ("tree export", "cluster"), ("store mirror", "store_mirror"),
                 ("status heavy refresh", "status_heavy"), ("acceptance before-phase", "acceptance"), ("store authority", "authority"))


def stage_class(label: str) -> str:
    low = label.lower()
    if ADAPTER_REFUSED_RE.match(label):
        # `-- adapter A REFUSED HH:MM:SS (reason)` (C6): an adapter recorded refused, never read on the grading host -- its own class
        return "adapter_refused"
    return next((cls for prefix, cls in STAGE_CLASSES if low.startswith(prefix)), "other")


def _label(text: str) -> str:
    """The stage label: the text with its first HH:MM:SS token (and a trailing parenthesised key) removed."""
    m = TIME_RE.search(text)
    if not m:
        return text.strip()
    head, tail = text[: m.start()].strip(), text[m.end():].strip()
    if tail.startswith("("):
        tail = ""
    return (head + (" " + tail if tail else "")).strip()


def parse_log(lines: List[str]) -> List[Dict[str, object]]:
    windows: List[Dict[str, object]] = []
    cur: Optional[Dict[str, object]] = None
    clock: Optional[datetime] = None

    def at(text: str) -> Optional[datetime]:
        """The wall time of a stamp: HH:MM:SS on the window's running date, rolling the date when the clock wraps."""
        nonlocal clock
        m = TIME_RE.search(text)
        if not m or clock is None:
            return None
        t = clock.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=int(m.group(3)))
        if t < clock - timedelta(hours=1):
            t += timedelta(days=1)
        clock = max(clock, t)
        return t

    def close(end: Optional[datetime], aborted: Optional[Tuple[str, Optional[datetime]]] = None) -> None:
        nonlocal cur
        if cur is None:
            return
        stages: List[Dict[str, object]] = cur["stages"]  # type: ignore[assignment]
        # an ABORTED marker ends the last stage (its wall is measured) without completing the window
        last_stamp = aborted[1] if aborted is not None else None
        for i, st in enumerate(stages):
            nxt = stages[i + 1]["at"] if i + 1 < len(stages) else (end if end is not None else last_stamp)
            st["seconds"] = (nxt - st["at"]).total_seconds() if (nxt is not None and st["at"] is not None) else None
        cur["end"] = end.isoformat().replace("+00:00", "Z") if end else None
        cur["completed"] = end is not None
        cur["abort_cause"] = aborted[0] if aborted is not None else None
        cur["aborted_at"] = last_stamp.isoformat().replace("+00:00", "Z") if last_stamp else None
        # the end marker's definition (C6): `rev3` = written by the trap after the timer restore (a restore line precedes it); a historical
        # window ended before its restore (`pre-rev3`); an aborted window is judged by the same rule (an ABORTED marker follows the restore)
        cur["end_definition"] = "rev3" if cur.get("_restore_seen") else "pre-rev3"
        last = end or last_stamp or (stages[-1]["at"] if stages and stages[-1]["at"] else None)
        cur["total_s"] = (last - cur["_start"]).total_seconds() if last else None
        if cur.get("_quiet") is not None:
            cur["quiet_wait_s"] = (cur["_quiet"] - cur["_start"]).total_seconds()
        else:
            cur["quiet_wait_s"] = None
        cur["drain_s"] = cur["quiet_wait_s"]       # C6's honest drain: the flag before the stop, then `writers quiet` -- measured from the start
        cur["changed_set"] = [a for a in str(cur["adapters"]).split() if a]
        cur["changed_set_n"] = len(cur["changed_set"])
        cur["cluster_read"] = any(st["class"] == "cluster" for st in stages)
        cur.pop("_start"); cur.pop("_quiet", None); cur.pop("_restore_seen", None)
        for st in stages:
            st["at"] = st["at"].isoformat().replace("+00:00", "Z") if st["at"] else None
        windows.append(cur)
        cur = None

    for raw in lines:
        line = raw.rstrip("\n")
        m = START_RE.match(line)
        if m:
            close(None)  # a start with no end before it: the previous window aborted
            t0 = parse_iso(m.group(1))
            if t0 is None:
                continue
            clock = t0
            cur = {"start": m.group(1), "_start": t0, "adapters": (m.group(2) or "").strip(), "base": None, "stages": [], "_quiet": None,
                   "_restore_seen": False, "op_id": None, "records_folded": None, "files_folded": None}
            continue
        if cur is None:
            continue
        if cur["stages"] and cur["stages"][-1]["class"] == "ingest":
            # the harvester's summary inside the ingest stage: `"files": N` / `"records": N` when window.sh's tail carries them
            m = INGEST_KEY_RE.match(line)
            if m:
                cur[f"{m.group(1)}_folded"] = int(m.group(2))
        m = END_RE.match(line)
        if m:
            close(parse_iso(m.group(1)) or at(line))
            continue
        m = ABORT_RE.match(line)
        if m:
            close(None, aborted=(m.group(1), parse_iso(m.group(2)) or at(line)))
            continue
        if RESTORE_RE.match(line):
            cur["_restore_seen"] = True
            continue
        m = BASE_RE.match(line)
        if m:
            cur["base"] = m.group(1)
            t = at(line)
            cur["stages"].append({"label": "base at " + m.group(1), "class": "base", "at": t})
            continue
        if QUIET_RE.match(line):
            cur["_quiet"] = at(line)
            continue
        m = STAGE_RE.match(line)
        if m:
            text = m.group(1)
            t = at(text)
            if t is None:
                continue
            label = _label(text)
            if label.startswith("reads reservation"):
                mo = RESERVATION_OP_RE.search(text)
                cur["op_id"] = mo.group(1) if mo else None
            cur["stages"].append({"label": label, "class": stage_class(label), "at": t})
    close(None)
    return windows


def read_logs(paths: List[str]) -> List[Dict[str, object]]:
    seen: Dict[str, Dict[str, object]] = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            for w in parse_log(fh.readlines()):
                prev = seen.get(w["start"])  # type: ignore[index]
                if prev is None or (w["completed"] and not prev["completed"]):
                    seen[w["start"]] = w  # type: ignore[index]
    return [seen[k] for k in sorted(seen)]


def read_receipts(path: str) -> Dict[str, Dict[str, object]]:
    """Every read receipt under ``path`` (``reads_receipt/1``), keyed by its op id (``receipts/<op_id>.json``, written by
    ``cluster-reads receipt`` for a window and by the tick); an unreadable or foreign file is skipped. ``{}`` when the directory is absent."""
    out: Dict[str, Dict[str, object]] = {}
    if not path or not os.path.isdir(path):
        return out
    for fn in sorted(os.listdir(path)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(path, fn), "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("schema") == RECEIPT_SCHEMA:
            op = (doc.get("op") or {}).get("op_id") or doc.get("window")
            if op:
                out[str(op)] = doc
    return out


def manifest_bytes(root: Optional[str], slug: str, sha8: str) -> Tuple[Optional[int], Optional[int]]:
    """``(bytes, n_files)`` of a published generation from the grading host manifest copy ``<root>/state/store_manifests/<slug>/MANIFEST-<sha8>.json``
    (the one the mirror or the push wrote for that content sha); ``(None, None)`` when this host has none for it."""
    if not root:
        return None, None
    p = os.path.join(root, "state", "store_manifests", slug, f"MANIFEST-{sha8}.json")
    try:
        with open(p, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None, None
    if not isinstance(doc, dict):
        return None, None
    b, n = doc.get("bytes"), doc.get("n_files")
    return (int(b) if isinstance(b, (int, float)) else None), (int(n) if isinstance(n, (int, float)) else None)


#: The standing-refusal contract (ONE exact shape, the status page reads the same -- `status.STANDING_REFUSAL`): an adapter's receipt entry carries a
#: list ``standing_refusals`` of ``{"group", "state": "standing_refusal", "refusal", "prior_refusal": {"window", "utc", "digest"}}`` -- each a
#: refusal the cluster read re-confirmed byte for byte against the group's previous recorded refusal under the same inputs and tree, recorded,
#: never an abort. An adapter whose only refusals stand is written ``read_whole: true`` with ``refused`` None (so the tick leaves it alone),
#: so the grouping reads the LIST: a receipt is all-whole only with no standing entry at all, standing-only when every adapter is whole or
#: standing and at least one stands. ``refused_standing`` keeps its one meaning (the revision ids a pull left standing UNREAD,
#: string items) -- and an entry can carry BOTH: a re-confirmed standing refusal beside a revision the pull left unread. The list never
#: outranks the unread ids: an adapter is whole-or-standing only with nothing
#: left unread and every shard ok, the same precedence the status page's ``newest_entry_state`` applies (refused, then a shard missing, then
#: ids left unread, then standing, then whole). Before this the list alone counted the adapter whole-or-standing, so a window with such an
#: entry read ``not_whole_n = 0`` and grouped standing-only: an incomplete read presented as a standing-only outcome in the acceptance summary.
STANDING_REFUSAL = "standing_refusal"
STANDING_REFUSALS_KEY = "standing_refusals"
STANDING_UNREAD_KEY = "refused_standing"


def standing_refusals_n(entry: Dict[str, object]) -> int:
    """How many STANDING refusals an adapter's receipt entry carries: the items of its ``standing_refusals`` list whose ``state`` is the
    standing word. An older receipt carries no list and counts none."""
    return sum(1 for x in (entry.get(STANDING_REFUSALS_KEY) or []) if isinstance(x, dict) and str(x.get("state")) == STANDING_REFUSAL)   # type: ignore[union-attr]


def standing_unread_n(entry: Dict[str, object]) -> int:
    """How many revision ids the pull left standing UNREAD (``refused_standing``: the group was neither read nor committed, so the
    entry is not whole whatever else it says). Every item counts: the key has that one meaning."""
    return len(entry.get(STANDING_UNREAD_KEY) or [])      # type: ignore[arg-type]


def shards_missing(entry: Dict[str, object]) -> bool:
    """A shard of the adapter's read is missing or not ok (``shards_ok`` below ``shards_total``; an adapter routed off the cluster has
    ``shards_total`` 0 and no shard to miss) -- the status page's ``partial``."""
    total = int(entry.get("shards_total") or 0)       # type: ignore[arg-type]
    return total > 0 and int(entry.get("shards_ok") or 0) < total      # type: ignore[arg-type]


def receipt_columns(rec: Optional[Dict[str, object]], root: Optional[str] = None) -> Dict[str, object]:
    """The changed-set columns of one window from its read receipt: adapters in the receipt / read whole / refused / with a STANDING
    refusal (``standing_refusals_n``) / with revision ids left standing UNREAD (``standing_unread_n``) / whole or standing
    (``whole_or_standing_n``: no refusal, nothing left unread, every shard ok, and read whole or standing) / neither whole-or-standing nor
    refused (``not_whole_n``: a shard missing, ids left standing unread, routed off the cluster -- a standing refusal beside any of these is
    still not whole),
    revisions pulled, re-activations, the newest ``change_utc`` and ``published_utc`` among the pins (the fold-latency clock), the pins, and
    the pins' input bytes and file counts resolved from the grading host's manifests (``manifest_bytes``; ``input_bytes`` is None when any
    pin has no manifest here, so a partial sum never reads as the whole). ``receipt: False`` with every column None when the window has no receipt."""
    if not rec:
        return {"receipt": False, "receipt_utc": None, "receipt_adapters_n": None, "read_whole_n": None, "refused_n": None, "standing_refusals_n": None,
                "standing_unread_n": None, "whole_or_standing_n": None, "not_whole_n": None, "revisions_pulled": None,
                "reactivated": None, "change_utc": None, "published_utc": None, "pins": None, "input_bytes": None, "input_files": None,
                "pins_without_manifest": None}
    adapters: Dict[str, Dict[str, object]] = rec.get("adapters") or {}      # type: ignore[assignment]
    pins: Dict[str, str] = {}
    change, published = [], []
    pulled = react = whole = refused = standing = unread = whole_or_standing = 0
    for e in adapters.values():
        pins.update({str(k): str(v) for k, v in (e.get("pins") or {}).items()})      # type: ignore[union-attr]
        for pub in (e.get("published") or {}).values():                             # type: ignore[union-attr]
            if isinstance(pub, dict):
                if pub.get("change_utc"):
                    change.append(str(pub["change_utc"]))
                if pub.get("published_utc"):
                    published.append(str(pub["published_utc"]))
        pulled += len(e.get("revisions_pulled") or [])                               # type: ignore[arg-type]
        react += len(e.get("reactivated") or [])                                    # type: ignore[arg-type]
        is_refused = bool(e.get("refused"))
        n_standing, n_unread = standing_refusals_n(e), standing_unread_n(e)
        whole += 1 if e.get("read_whole") else 0
        refused += 1 if is_refused else 0
        standing += 1 if (not is_refused and n_standing) else 0
        unread += 1 if (not is_refused and n_unread) else 0
        # whole-or-standing: the status page's precedence -- a refusal, a shard missing or an id left unread each make the entry NOT whole,
        # whatever the standing list says; only then does `read_whole` (the writer's own verdict) or a standing refusal count the adapter
        whole_or_standing += 1 if (not is_refused and not n_unread and not shards_missing(e) and (e.get("read_whole") or n_standing)) else 0
    total_b = total_n = 0
    missing: List[str] = []
    for slug, sha8 in sorted(pins.items()):
        b, n = manifest_bytes(root, slug, sha8)
        if b is None:
            missing.append(f"{slug}/{sha8}")
        else:
            total_b += b
            total_n += n or 0
    return {"receipt": True, "receipt_utc": rec.get("utc"), "receipt_adapters_n": len(adapters), "read_whole_n": whole, "refused_n": refused,
            "standing_refusals_n": standing, "standing_unread_n": unread, "whole_or_standing_n": whole_or_standing,
            "not_whole_n": len(adapters) - whole_or_standing - refused,
            "revisions_pulled": pulled, "reactivated": react, "change_utc": max(change) if change else None,
            "published_utc": max(published) if published else None, "pins": pins,
            "input_bytes": total_b if pins and not missing else None, "input_files": total_n if pins and not missing else None,
            "pins_without_manifest": missing}


def join_receipts(windows: List[Dict[str, object]], receipts: Dict[str, Dict[str, object]], root: Optional[str] = None) -> None:
    """Attach :func:`receipt_columns` to every window, by its op id (the reservation stamp); in place."""
    for w in windows:
        op = w.get("op_id")
        w.update(receipt_columns(receipts.get(str(op)) if op else None, root))


def select_span(windows: List[Dict[str, object]], since: Optional[str] = None, until: Optional[str] = None,
                first: Optional[int] = None) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """The windows of the span (start >= since, start < until) and, with ``first``, the first N COMPLETED ones: S1 counts windows
    that END with ``=== ff chain end``; an ABORTED window inside the span is returned separately with its cause and
    never fills a slot -- every abort up to the Nth completion once it exists, every abort of the span while fewer than N have completed.
    Returns ``(selected, aborted_in_span)``; without ``first`` every window of the span is selected."""
    span = [w for w in windows if (not since or str(w["start"]) >= since) and (not until or str(w["start"]) < until)]
    if first is None:
        return span, []
    done = [w for w in span if w["completed"]][:first]
    # the aborts listed beside the N: once the Nth completion EXISTS, every abort of the span up to its start (a later abort is outside
    # the ten); while FEWER than N windows have completed -- the live read before the tenth -- every abort of the span through --until
    # (or the latest observed window). Before this rule the cutoff was the latest completion's start even then, so an abort after the
    # latest completion vanished from the summary until another window completed: exactly the failed windows S1 must list.
    if done and len(done) >= first:
        cutoff = str(done[-1]["start"])
        aborted = [w for w in span if not w["completed"] and str(w["start"]) <= cutoff]
    else:
        aborted = [w for w in span if not w["completed"]]
    return done, aborted


def _stats_min(ws: List[Dict[str, object]]) -> Dict[str, object]:
    t = [float(w["total_s"]) / 60.0 for w in ws if w.get("completed") and w.get("total_s") is not None]       # type: ignore[arg-type]
    return {"n": len(t), "median_min": round(statistics.median(t), 1) if t else None, "mean_min": round(statistics.mean(t), 1) if t else None,
            "max_min": round(max(t), 1) if t else None}


def subpopulations(windows: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    """The sub-populations' medians beside the ten: windows with a cluster read / without one; by changed-set size
    (0 / 1-2 / 3+ adapters named on the start line); windows by what their receipt read -- EVERY adapter whole and no standing entry at
    all (``read_whole_n`` and ``whole_or_standing_n`` both equal the adapters in the receipt; ``refused_n == 0`` alone used
    to group a receipt with an adapter left standing unread, or a shard missing without a refusal, as all-whole), with a refusal,
    standing-only (every adapter whole or standing and at least one ``standing_refusals`` entry; the cluster read writes such an
    adapter ``read_whole: true``, so the list decides -- and nothing left standing unread or missing a shard beside the list: that adapter
    is not whole-or-standing, so the receipt is partial), or partial (no refusal, not every adapter whole or standing). The four are a
    partition of the windows with a receipt."""
    done = [w for w in windows if w.get("completed")]
    out = {"with_cluster_read": _stats_min([w for w in done if w.get("cluster_read")]),
           "without_cluster_read": _stats_min([w for w in done if not w.get("cluster_read")]),
           "changed_set_0": _stats_min([w for w in done if int(w.get("changed_set_n") or 0) == 0]),
           "changed_set_1_2": _stats_min([w for w in done if 1 <= int(w.get("changed_set_n") or 0) <= 2]),
           "changed_set_3_plus": _stats_min([w for w in done if int(w.get("changed_set_n") or 0) >= 3])}
    with_receipt = [w for w in done if w.get("receipt")]
    if with_receipt:
        def n(w, k):
            return int(w.get(k) or 0)

        def all_whole(w):
            return (not w.get("refused_n") and not w.get("standing_refusals_n") and n(w, "read_whole_n") == n(w, "receipt_adapters_n")
                    and n(w, "whole_or_standing_n") == n(w, "receipt_adapters_n"))

        def standing_only(w):
            return not w.get("refused_n") and bool(w.get("standing_refusals_n")) and n(w, "whole_or_standing_n") == n(w, "receipt_adapters_n")
        out["receipt_all_read_whole"] = _stats_min([w for w in with_receipt if all_whole(w)])
        out["receipt_with_refusal"] = _stats_min([w for w in with_receipt if w.get("refused_n")])
        out["receipt_standing_only"] = _stats_min([w for w in with_receipt if standing_only(w)])
        out["receipt_partial"] = _stats_min([w for w in with_receipt if not w.get("refused_n") and not all_whole(w) and not standing_only(w)])
    return out


def summarize(windows: List[Dict[str, object]]) -> Dict[str, object]:
    done = [w for w in windows if w["completed"] and w["total_s"] is not None]
    totals = [w["total_s"] / 60.0 for w in done]  # type: ignore[operator]
    by_class: Dict[str, List[float]] = {}
    for w in windows:
        for st in w["stages"]:  # type: ignore[union-attr]
            if st["seconds"] is not None:
                by_class.setdefault(st["class"], []).append(float(st["seconds"]))
        if w.get("quiet_wait_s") is not None:
            by_class.setdefault("quiet", []).append(float(w["quiet_wait_s"]))  # type: ignore[arg-type]
    classes = {}
    grand = sum(sum(v) for v in by_class.values()) or 1.0
    for k, v in sorted(by_class.items(), key=lambda kv: -sum(kv[1])):
        classes[k] = {"n": len(v), "median_s": round(statistics.median(v), 1), "mean_s": round(statistics.mean(v), 1), "max_s": round(max(v), 1),
                      "sum_min": round(sum(v) / 60.0, 1), "share": round(sum(v) / grand, 3)}
    defs = sorted({str(w.get("end_definition")) for w in done})
    return {"n_started": len(windows), "n_completed": len(done), "n_aborted": len(windows) - len(done),
            "median_min": round(statistics.median(totals), 1) if totals else None, "mean_min": round(statistics.mean(totals), 1) if totals else None,
            "max_min": round(max(totals), 1) if totals else None, "min_min": round(min(totals), 1) if totals else None,
            "sum_min": round(sum(totals), 1) if totals else None, "by_class": classes,
            "end_definitions": defs, "n_with_receipt": sum(1 for w in windows if w.get("receipt")),
            "n_without_receipt": sum(1 for w in windows if w.get("op_id") and not w.get("receipt")),
            "subpopulations": subpopulations(windows)}


def window_rows(windows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for w in windows:
        stages = sorted([s for s in w["stages"] if s["seconds"] is not None], key=lambda s: -s["seconds"])  # type: ignore[index]
        rows.append({"start": w["start"], "base": w["base"], "adapters": w["adapters"], "completed": w["completed"],
                     "end_definition": w.get("end_definition"), "abort_cause": w.get("abort_cause"),
                     "total_min": round(w["total_s"] / 60.0, 1) if w["total_s"] is not None else None,
                     "longest": f'{stages[0]["label"]} {stages[0]["seconds"] / 60:.1f}' if stages else None,
                     "second": f'{stages[1]["label"]} {stages[1]["seconds"] / 60:.1f}' if len(stages) > 1 else None,
                     # the changed-set columns: the set named on the start line, the receipt's reading of it, the
                     # pins' input bytes, revisions pulled, records folded, the measured drain
                     "op_id": w.get("op_id"), "changed_set_n": w.get("changed_set_n"), "cluster_read": w.get("cluster_read"),
                     "receipt": w.get("receipt"), "read_whole_n": w.get("read_whole_n"), "refused_n": w.get("refused_n"),
                     "standing_refusals_n": w.get("standing_refusals_n"), "standing_unread_n": w.get("standing_unread_n"),
                     "whole_or_standing_n": w.get("whole_or_standing_n"), "not_whole_n": w.get("not_whole_n"),
                     "receipt_adapters_n": w.get("receipt_adapters_n"),
                     "revisions_pulled": w.get("revisions_pulled"), "reactivated": w.get("reactivated"), "input_bytes": w.get("input_bytes"),
                     "input_files": w.get("input_files"), "change_utc": w.get("change_utc"), "records_folded": w.get("records_folded"),
                     "files_folded": w.get("files_folded"), "drain_s": w.get("drain_s")})
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="*", help=f"chain logs (default: {' '.join(DEFAULT_LOGS)})")
    ap.add_argument("--since", help="keep windows starting at or after this UTC date/time (YYYY-MM-DD[THH:MM:SSZ])")
    ap.add_argument("--until", help="keep windows starting BEFORE this UTC date/time (the baseline: --since 2026-09-24 --until 2026-09-28T22:00:00Z)")
    ap.add_argument("--first", type=int, help="S1: only the first N COMPLETED windows of the span (an ABORTED window does not count; it is listed -- every "
                                              "abort of the span while fewer than N have completed, up to the Nth completion once it exists)")
    ap.add_argument("--receipts", default=DEFAULT_RECEIPTS, help=f"the read receipts joined per window by op id (default {DEFAULT_RECEIPTS})")
    ap.add_argument("--root", default=DEFAULT_ROOT, help=f"the registry root whose state/store_manifests resolve the pins' input bytes (default {DEFAULT_ROOT})")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    windows = read_logs(a.logs or DEFAULT_LOGS)
    join_receipts(windows, read_receipts(a.receipts), a.root)
    windows, aborted = select_span(windows, a.since, a.until, a.first)
    summary = summarize(windows)
    summary["first"] = a.first
    summary["aborted_in_span"] = [{"start": w["start"], "abort_cause": w.get("abort_cause") or "no end marker", "adapters": w["adapters"]} for w in aborted]
    rows = window_rows(windows)
    if a.json:
        print(json.dumps({"summary": summary, "windows": rows, "stages": [w["stages"] for w in windows]}, indent=1, default=str))
        return 0
    print("| window start (UTC) | end def. | base | ADAPTERS | total min | longest stage (min) | 2nd longest | changed | receipt whole/refused | rev pulled | input MB | drain s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        mark = "" if r["completed"] else (f' (ABORTED: {r["abort_cause"]})' if r.get("abort_cause") else " (ABORTED: no end marker)")
        # whole / refused as before; a receipt with STANDING refusals says so beside them, and one with ids left standing UNREAD
        # says so too, so the column's pinned shape holds and a partial read is never mistaken for a standing-only one
        rcpt = (f'{r["read_whole_n"]}/{r["refused_n"]}' + (f' +{r["standing_refusals_n"]} standing' if r.get("standing_refusals_n") else "")
                + (f' +{r["standing_unread_n"]} unread' if r.get("standing_unread_n") else "")
                if r.get("receipt") else ("—" if not r.get("op_id") else "none"))
        mb = f'{r["input_bytes"] / 1e6:.0f}' if r.get("input_bytes") is not None else "—"
        print(f'| {r["start"]}{mark} | {r.get("end_definition") or "—"} | {r["base"] or "—"} | {r["adapters"] or "—"} | '
              f'{r["total_min"] if r["total_min"] is not None else "—"} | {r["longest"] or "—"} | {r["second"] or "—"} | {r["changed_set_n"]} | {rcpt} | '
              f'{r["revisions_pulled"] if r["revisions_pulled"] is not None else "—"} | {mb} | {int(r["drain_s"]) if r.get("drain_s") is not None else "—"} |')
    s = summary
    print(f'\nCompleted windows n={s["n_completed"]} of {s["n_started"]} started ({s["n_aborted"]} aborted): median {s["median_min"]} min, '
          f'mean {s["mean_min"]}, max {s["max_min"]}, min {s["min_min"]}; sum {s["sum_min"]} min; end definition(s): {", ".join(s["end_definitions"]) or "—"}'
          + (f'; first {a.first} completed windows of the span' if a.first else ""))
    if summary["aborted_in_span"]:
        print("ABORTED in the span (not counted): " + "; ".join(f'{x["start"]} ({x["abort_cause"]})' for x in summary["aborted_in_span"]))
    sp = s["subpopulations"]
    print("sub-populations (completed): " + "; ".join(f'{k} n={v["n"]} median {v["median_min"]} mean {v["mean_min"]} max {v["max_min"]}' for k, v in sp.items()))
    if s["n_with_receipt"] or s["n_without_receipt"]:
        print(f'receipts joined: {s["n_with_receipt"]}; windows with a reservation stamp and no receipt: {s["n_without_receipt"]}')
    print("\n| stage class | n | median s | mean s | max s | sum min | share |")
    print("|---|---|---|---|---|---|---|")
    for k, v in s["by_class"].items():
        print(f'| {k} | {v["n"]} | {v["median_s"]} | {v["mean_s"]} | {v["max_s"]} | {v["sum_min"]} | {v["share"]:.1%} |')
    return 0


if __name__ == "__main__":
    sys.exit(main())
