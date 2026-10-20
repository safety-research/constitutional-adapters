#!/usr/bin/env python3
"""seedrep_petri_watch.py -- the seedrep battery's Petri stall / refusal monitor (D113; stdlib only).

Started from, and crediting, the peer lane's ``~/scripts/watch-petri-stall.sh`` (relu-direct-difference-sweep,
2026-09-19, D33/D33a): the same event vocabulary (``WIDTH-1``, ``REFUSAL-RETRY``, ``STALL``, ``ENGINE-QUIET``,
``LEG-DONE``), the same liveness rule (file MTIME, never the leg log's own timestamps), the same "one line per
NOTABLE TRANSITION, silence is never health" stance -- pointed at :mod:`seedrep_petri_drive`'s state layout and
extended with the D113 triggers (a deterministic refusal LOOP on one conversation; a leg that DIED rather than
hung) and a per-leg sample-progress snapshot read from the Inspect eval logs.

The failure this exists to catch (ReLU's D33, diagnosis CORRECTED in D120): a leg's judge or auditor sits in
inspect_ai's retry belt for hours. TWO different things put it there, told apart ONLY by the record's body text,
never by the HTTP status:
  * ``APIStatusError 200`` + "overloaded" / "internal server error": a STREAMED OVERLOAD. In the pinned inspect_ai
    (``inspect_ai/model/_providers/anthropic.py::should_retry``, :1320-1345; the branch at :1323-1328) an
    APIStatusError whose body carries either phrase is ``RetryDecision.transient`` REGARDLESS of status code
    ("when streaming, anthropic does not set status_code == 529"). The client retries it (max_retries 14, sleep
    capped by PETRI_AUDITOR_RETRY_SLEEP_CAP_S -- the seedrep drive sets 300 s), and the retries CAN succeed. That is
    ``OVERLOAD-RETRY`` here: informational, never an rc-3 trigger by itself; a belt that outlasts --stall-min is
    caught by the mtime STALL.
  * "declined this request with a safety classifier refusal (category: ...)" / ``content_filter``: a CLASSIFIER
    refusal. A refusal RETURNED as output is ``stop_reason == "content_filter"`` -> inspect_scout's RefusalError, and
    the manifest's fallback chain switches legs at once; one that surfaces as a retried exception is deterministic
    on the same body, so its retries cannot succeed. That is ``REFUSAL-RETRY`` / ``REFUSAL-LOOP``.
  * anything else in the retry belt (a 429, a 5xx without those phrases): ``UNKNOWN-RETRY``.
At width 1 one held sample halts its entire leg, the served GPU goes idle, and the drive's heartbeat still says alive.

WHAT IT READS (the drive's layout, :func:`seedrep_petri_drive.state_dir_for` / :func:`attempt_state_dir`):

  <wave state dir>  ~/.local/state/seedrep_battery/petri/<W7>/<wave_uuid8>/<view>[_rerun]/
      drive.log                          UTC-stamped ``[YYYY-MM-DDTHH:MM:SSZ] ...``; ``DRIVE_START ... width=N``,
                                         ``schedule: ... to run: [...]``, ``launched leg_<k>_s<s> (...)``,
                                         ``LEG[leg_<k>_s<s>] COMPLETE|INCOMPLETE|FAILED rc=...``
      attempts/<attempt_id>/
          leg_<k>_s<s>.log               the leg's petri_run stdout+stderr: ``== leg_<k>_s<s> attempt N width=W <utc> ==``
                                         headers, then inspect's Rich-wrapped records stamped ``[MM/DD/YY HH:MM:SS]``
                                         in LOCAL time (PT on the grading host) -- printed verbatim, never compared to a clock
          leg_<k>_s<s>.rc                ``<rc> drive=<id> results=<sha256|absent>`` once the leg RETURNED
          drive_<drive_id>.json          ``schedule`` (record_name -> run_key, tag) + ``out_roots`` (tag -> out root)
  <out root>        ~/.local/state/petri_native/petri_seedrep_<W7>_<wave_uuid8>_<attempt_id>_<tag>/
      inspect_logs/<run_key>/target-seed-<s>/*.eval   zip; ``samples/<id>_epoch_<e>.json`` = one COMPLETED sample
      cells/<run_key>/results.json                    ``audited.expected`` (the leg's sample total)
      cells/<run_key>/petri_auditor_refused.json, petri_judge_refused.json   terminal refusal sidecars

Any directory works as an argument: a wave state dir, an attempt dir, or ReLU's flat ``<dir>/drive.log +
leg_*.log`` layout -- leg logs are collected from ``<dir>/leg_*.log`` and ``<dir>/attempts/*/leg_*.log``.

EVENTS (one line each, ``<UTC> <EVENT> <view>/<leg> ...``; all are transitions except the ``LEG`` snapshot):
  WIDTH-1        one-shot at start: ``DRIVE_START ... width=1`` or a leg header ``width=1`` -- the intended width
                 never took effect (D33); refused by name (rc 3 under --once)
  REFUSAL        a classifier refusal record (``<model> declined this request with a safety classifier refusal
                 (category: X)`` -- its own record with its own sample) appeared for a (sample, model), below --loop-n
  REFUSAL-RETRY  the leg's highest CLASSIFIER ``retry N`` rose; names the model, the sample, the classifier category
  REFUSAL-LOOP   the SAME conversation (sample) CLASSIFIER-refused >= --loop-n times by the SAME model (body text,
                 never the status code) -- deterministic; the D113 trigger to restart the leg on the fallback
  OVERLOAD-RETRY the leg's highest OVERLOAD ``retry N`` rose ("overloaded" / "internal server error" in the body):
                 informational -- model, sample, retry depth, elapsed since the belt's first record; retries can succeed
  UNKNOWN-RETRY  a retry record whose body names neither (a 429, a bare 5xx): said once per new depth
  STALL          neither the leg log nor its inspect logs written for --stall-min -- judged from file MTIMES on every
                 observation, the first included, so a single --once pass detects a stale active leg
  ENGINE-QUIET / ENGINE-UNOBSERVED   a ``serving/vllm_server.log`` under the dir went quiet / none is visible here
                 (the seedrep proxy runs in Modal; engine liveness is NOT observable from this box -- said once)
  LEG-DONE       ``.rc`` appeared with rc 0 and results bytes;  LEG-DIED  ``.rc`` with any other rc or results=absent
                 (ReLU saw ``rc=3 results=absent``: a leg can die, not just hang). BOTH bind the record to the drive
                 being watched: the .rc's ``drive=`` must equal the LATEST ``DRIVE_START``'s ``drive_id`` -- a
                 re-invoked drive appends to the same leg log and leaves its predecessor's .rc until its own leg
                 returns, so an older record is LEG-PRIOR-RC (history; the leg stays live), never terminal
  NO-LEGS-STARTED  no drive wrote a leg log within --grace-min (default 60): said once, the watch continues
  LEG-COMPLETE-PRIOR / LEG-UNSCHEDULED   the watch is bound to the CURRENT drive's schedule (its latest ``to run``
                 list and its drive_<id>.json ``already_complete``) and to its attempt dir (``state=``): a leg the drive
                 reconciled as already complete is complete (its .rc keeps the previous drive id by design), a retained
                 log the drive did not schedule is history -- neither is live, neither can stall (Sol P1, fk2c)
  LEG            the per-leg snapshot: ``samples <done>/<total> last_progress <UTC> refusals auditor=<n> judge=<n>
                 loops=<n> retry_max=<n> rc=<rc|running>`` -- every poll under --once / --status-every, else never
  ALL-LEGS-TERMINAL, WATCH-TIMEOUT

EXIT (``--once``): 0 healthy; 3 when any WIDTH-1 / REFUSAL-LOOP / STALL / LEG-DIED fired -- the launch agent's
in-turn loop acts on 3. Loop mode exits 0 at ALL-LEGS-TERMINAL, 4 at WATCH-TIMEOUT.

ACTING ON A STALLED / LOOPING LEG (documented from the drive's code -- ``leg_attempts`` / ``legs_to_run`` /
``run_leg`` in seedrep_petri_drive.py -- not exercised live by this tool):
  1. Kill the leg's petri_run child:  pkill -f 'scripts/petri_run.py .* --cell-index <k>'  (match the LEG's argv,
     never the drive's). The drive's own three-attempt loop relaunches the leg in place (width throttled down on
     a 429-heavy attempt) and petri_run resumes by run key inside the leg, so completed samples are kept.
  2. If the drive process itself has RETURNED (drive.log ends in a verdict, or every leg has an .rc), re-invoke it
     for the leg(s) by cell index -- a re-invocation drives every leg whose verdict is not COMPLETE and skips a leg
     still locked by a live drive:
       ~/venvs/modal/bin/python scripts/modal/seedrep_petri_drive.py --sha <W 40hex> --wave-json RS/wave.json \\
           --view <view> --legs "<k>" --python ~/venvs/seedrep-petri-drive/bin/python [--width <n>]
     Run it from the W tree (clean, at <W>). The judge/auditor fallback is the MANIFEST's chain
     (``auditor_fallback_models`` / ``judge_fallback_models``), not a drive flag -- the restarted leg runs under
     whatever chain the tree's manifest declares.

Usage:
  seedrep_petri_watch.py <dir> [<dir> ...] [--once] [--poll-s 60] [--stall-min 15] [--loop-n 2] [--max-hours 48]
                         [--expected-samples 60] [--status-every 0] [--judge-model opus] [--auditor-model sonnet,grok]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # the leg logs are stamped in the box's LOCAL time -- PT, not UTC
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]

LEG_LOG_TZ = "America/Los_Angeles"
DEFAULT_EXPECTED_SAMPLES = 60      # 6 behaviors x 10 scenarios x 1 epoch (the frozen bundle 8812ba083bb5473c)
RC_OK = "0"

#: ours ``leg_<k>_s<seed>.log``; a flat layout's ``leg_<k>.log``
LEG_LOG_RE = re.compile(r"^leg_[A-Za-z0-9_]+\.log$")
DRIVE_START_WIDTH_RE = re.compile(r"\bDRIVE_START\b.*?\bwidth=(\d+)\b")
DRIVE_START_VIEW_RE = re.compile(r"\bDRIVE_START\b.*?\bview=(\S+)")
DRIVE_START_ID_RE = re.compile(r"\bDRIVE_START\b.*?\bdrive_id=(\S+)")
DRIVE_START_STATE_RE = re.compile(r"\bDRIVE_START\b.*?\bstate=(\S+)")     # the attempt state dir this drive writes into
#: ours ``== leg_<k>_s<s> attempt N width=W <utc> ==``; a flat layout's ``== leg <k> attempt N <utc> ==`` (no width)
ATTEMPT_HEADER_RE = re.compile(r"^== (leg\S*(?: \d+)?) attempt (\d+)(?: width=(\d+))? (\S+) ==$")
ATTEMPT_RC_RE = re.compile(r"^== (leg_\d+_s\d+) attempt (\d+) rc=(-?\d+) ==$")
TO_RUN_RE = re.compile(r"to run: \[([^\]]*)\]")
LAUNCHED_RE = re.compile(r"\blaunched (leg_\d+_s\d+) \(.*?\bout_root (\S+)\)")
LEG_VERDICT_RE = re.compile(r"\bLEG\[(leg_\d+_s\d+)\] (COMPLETE|INCOMPLETE|FAILED)(?: rc=(-?\d+))? ([0-9a-f]{16})")
#: inspect's Rich console stamps each record ``[MM/DD/YY HH:MM:SS]`` and wraps continuation lines with indentation
INSPECT_STAMP_RE = re.compile(r"^\[(\d{2})/(\d{2})/(\d{2}) (\d{2}):(\d{2}):(\d{2})\] ")
RETRY_RE = re.compile(r"->\s*(\S+)\s+retry\s+(\d+)")
#: a classifier refusal surfaces as ITS OWN record (logger.py:278), carrying its own ``sample=`` and naming the model:
#: ``<model> declined this request with a safety classifier refusal (category: X)`` -- the real leg logs' shape
DECLINED_RE = re.compile(r"(\S+) declined this request with a safety classifier refusal")
RETRY_LOOSE_RE = re.compile(r"\bretry\s+(\d+)\b")
SAMPLE_RE = re.compile(r"\bsample=(\S+)")
CATEGORY_RE = re.compile(r"category:\s*([a-z_]+)")
API_STATUS_RE = re.compile(r"\[APIStatusError (\d+)\]")
CLASSIFIER_TEXT = "safety classifier refusal"
#: the body phrases inspect_ai's anthropic provider retries as TRANSIENT regardless of status (should_retry :1323-1328)
OVERLOAD_TEXTS = ("overloaded", "internal server error")
CLASSIFIER_TEXTS = (CLASSIFIER_TEXT, "content_filter", "category: ")
KIND_CLASSIFIER, KIND_OVERLOAD, KIND_UNKNOWN = "classifier", "overload", "unknown"
JUDGE_REFUSED_TEXT = "Judge model refused to score"
MODEL_RE = re.compile(r"(?:claude|openrouter|openai|anthropic)[-/][A-Za-z0-9./_-]+")
RUN_KEY_IN_LOG_RE = re.compile(r"cells/([0-9a-f]{16})")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_of(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def leg_stamp_to_utc(stamp: str, *, tz_name: str = LEG_LOG_TZ) -> Optional[str]:
    """``[MM/DD/YY HH:MM:SS]`` (inspect's Rich stamp, the box's LOCAL time = PT) -> ISO UTC.
    Pinned: ``[09/19/26 11:06:31]`` is 18:06:31Z, not 11:06Z. Returns None when the
    stamp does not parse or no tz database is available (never a silently wrong conversion)."""
    m = INSPECT_STAMP_RE.match(stamp if stamp.endswith(" ") else stamp + " ")
    if not m or ZoneInfo is None:
        return None
    mo, d, yy, hh, mi, ss = (int(x) for x in m.groups())
    try:
        local = datetime(2000 + yy, mo, d, hh, mi, ss, tzinfo=ZoneInfo(tz_name))
    except Exception:  # an impossible date, a missing tz db
        return None
    return local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- the leg log: Rich-wrapped inspect records ---------------------------------------------------------------------

@dataclass
class LogRecord:
    stamp: str          # the log's own ``[MM/DD/YY HH:MM:SS]`` text, verbatim (LOCAL time)
    text: str           # the record's lines joined with single spaces, indentation stripped


def parse_inspect_records(text: str) -> List[LogRecord]:
    """Group a leg log into inspect records: a line opening with ``[MM/DD/YY HH:MM:SS]`` starts one; indented
    continuation lines belong to it (Rich wraps at 80 cols and may split a token mid-word -- joined with a space,
    so a wrapped model name or sample id reads as two fragments; callers match the unwrapped parts)."""
    records: List[LogRecord] = []
    cur: Optional[LogRecord] = None
    for raw in text.splitlines():
        m = INSPECT_STAMP_RE.match(raw)
        if m:
            cur = LogRecord(stamp=raw[:m.end()].rstrip(), text=raw[m.end():].strip())
            records.append(cur)
        elif cur is not None and raw.startswith(" ") and raw.strip():
            cur.text += " " + raw.strip()
        else:
            cur = None
    return records


@dataclass
class RefusalRecord:
    stamp: str
    sample: str
    model: str
    retry: int
    status: Optional[int]
    category: str
    classifier: bool          # kind == classifier (kept for callers)
    kind: str = KIND_UNKNOWN  # classifier | overload | unknown -- from the BODY TEXT of the record and its neighbours


def record_kind(texts: Sequence[str]) -> str:
    """The kind of one retry record from its own text and its adjacent records' (the exception body rides the record
    next to the ``retry N`` line): classifier text wins, then an overload phrase, else unknown. NEVER the status code:
    a streamed overload reports ``APIStatusError 200`` too (should_retry)."""
    joined = " ".join(texts).lower()
    if any(t in joined for t in CLASSIFIER_TEXTS):
        return KIND_CLASSIFIER
    if any(t in joined for t in OVERLOAD_TEXTS):
        return KIND_OVERLOAD
    return KIND_UNKNOWN


def refusal_records(records: Iterable[LogRecord]) -> List[RefusalRecord]:
    """Every ``retry N`` record -- the retry loop's own trace -- with the model, the sample, the HTTP status
    inspect reported and the classifier category when the text carries one."""
    out: List[RefusalRecord] = []
    records = list(records)
    # the classifier category rides the EXCEPTION record ("...declined this request with a safety classifier
    # refusal (category: X)"), adjacent to the retry record; fill a retry record's category from its nearest
    # neighbour within 3 records when its own text carries none
    categories = [(CATEGORY_RE.search(r.text).group(1) if CATEGORY_RE.search(r.text) else "") for r in records]

    samples = [(SAMPLE_RE.search(r.text).group(1) if SAMPLE_RE.search(r.text) else None) for r in records]

    def neighbours(i: int) -> List[int]:
        """Record i and the exception records that FOLLOW it (inspect logs the ``retry N`` record first, then the
        exception body as sample-less records): walk forward up to 3 records, stopping at the next record that names
        any sample. Never backward -- a sample-less record before i is the PREVIOUS retry record's exception text, so
        reading it would attach one sample's classifier refusal to another sample's (or model's) retry."""
        idx = [i]
        for k in range(1, 4):
            j = i + k
            if not 0 <= j < len(records) or samples[j] is not None:
                break
            idx.append(j)
        return idx

    def nearest_category(i: int) -> str:
        for j in sorted(neighbours(i), key=lambda j: abs(j - i)):
            if categories[j]:
                return categories[j]
        return ""

    def neighbourhood(i: int) -> List[str]:
        return [records[j].text for j in neighbours(i)]

    for i, r in enumerate(records):
        dm = DECLINED_RE.search(r.text)
        if dm and not RETRY_LOOSE_RE.search(r.text):
            # a classifier refusal record in its own right (retry depth 0: it is the refusal, not a retry of one)
            sm = SAMPLE_RE.search(r.text)
            cm_ = CATEGORY_RE.search(r.text)
            out.append(RefusalRecord(stamp=r.stamp, sample=sm.group(1) if sm else "?", model=dm.group(1), retry=0,
                                     status=None, category=cm_.group(1) if cm_ else "", classifier=True, kind=KIND_CLASSIFIER))
            continue
        m = RETRY_RE.search(r.text)
        if m:
            model, retry = m.group(1), int(m.group(2))
        else:
            lm = RETRY_LOOSE_RE.search(r.text)
            if not lm:
                continue
            models = MODEL_RE.findall(r.text)
            model, retry = (models[-1] if models else "?"), int(lm.group(1))
        sm = SAMPLE_RE.search(r.text)
        st = API_STATUS_RE.search(r.text)
        status = int(st.group(1)) if st else None
        kind = record_kind(neighbourhood(i))
        out.append(RefusalRecord(
            stamp=r.stamp, sample=sm.group(1) if sm else "?", model=model, retry=retry, status=status,
            category=nearest_category(i), classifier=(kind == KIND_CLASSIFIER), kind=kind))
    return out


def classify_role(model: str, *, judge_needles: Sequence[str], auditor_needles: Sequence[str]) -> str:
    low = model.lower()
    if any(n and n in low for n in judge_needles):
        return "judge"
    if any(n and n in low for n in auditor_needles):
        return "auditor"
    return "unknown"


def refusal_loops(recs: Sequence[RefusalRecord], *, loop_n: int) -> Dict[Tuple[str, str], int]:
    """``(sample, model) -> count`` of CLASSIFIER refusals (``APIStatusError 200`` / the classifier text) for
    every pair refused at least ``loop_n`` times -- the same conversation refused twice by the same model is a
    deterministic loop, never a transient."""
    counts: Dict[Tuple[str, str], int] = {}
    for r in recs:
        if r.classifier:
            counts[(r.sample, r.model)] = counts.get((r.sample, r.model), 0) + 1
    return {k: v for k, v in counts.items() if v >= loop_n}


# --- the drive's layout ---------------------------------------------------------------------------------------------

@dataclass
class DriveInfo:
    dir: Path
    view: str
    drive_log: Optional[Path]
    width: Optional[int]
    to_run: List[str]
    drive_id: Optional[str] = None      # the LATEST DRIVE_START's (an appended log holds every re-invocation's)
    attempt_dir: Optional[Path] = None  # the LATEST DRIVE_START's ``state=`` -- the attempt whose leg logs are this drive's
    scheduled: bool = False             # a ``to run: [...]`` schedule line was read for the current drive
    #: the CURRENT drive's reconciled already-complete legs (its drive_<drive_id>.json ``already_complete``): the drive
    #: deliberately skips them (seedrep_petri_drive legs_to_run), so their .rc files keep a PREVIOUS drive id -- they
    #: are complete, never "live work whose record is history" (fk2c)
    already_complete: List[str] = field(default_factory=list)
    out_roots_by_leg: Dict[str, Path] = field(default_factory=dict)       # from ``launched ... out_root``
    run_key_by_leg: Dict[str, str] = field(default_factory=dict)          # from drive_<id>.json schedule / LEG[...] lines
    verdicts: Dict[str, Tuple[str, Optional[str]]] = field(default_factory=dict)   # leg -> (COMPLETE|..., rc)


def find_leg_logs(d: Path, attempt_dir: Optional[Path] = None) -> List[Path]:
    """The leg logs to watch under ``d``: ``d/leg_*.log`` plus, when ``d/attempts`` exists, the CURRENT attempt's
    (``attempt_dir``, the latest DRIVE_START's ``state=``) -- logs retained under OLDER attempts are that attempt's
    history, not this drive's work. With no current attempt known, every attempt's logs (the pre-fk2c behaviour)."""
    logs = [p for p in d.glob("leg_*.log") if LEG_LOG_RE.match(p.name)]
    att = d / "attempts"
    if att.is_dir():
        cur = attempt_dir if attempt_dir is not None and attempt_dir.is_dir() and attempt_dir.parent == att else None
        for a in ([cur] if cur is not None else sorted(x for x in att.iterdir() if x.is_dir())):
            logs.extend(p for p in a.glob("leg_*.log") if LEG_LOG_RE.match(p.name))
    return sorted(logs)


def find_drive_log(d: Path) -> Optional[Path]:
    for cand in (d / "drive.log", d.parent / "drive.log", d.parent.parent / "drive.log"):
        if cand.is_file():
            return cand
    return None


def read_drive_info(d: Path) -> DriveInfo:
    info = DriveInfo(dir=d, view=d.name, drive_log=find_drive_log(d), width=None, to_run=[])
    if info.drive_log is not None:
        text = info.drive_log.read_text(errors="replace")
        for line in text.splitlines():
            m = DRIVE_START_WIDTH_RE.search(line)
            if m:
                # an appended drive.log holds EVERY re-invocation's DRIVE_START: the LATEST is the drive being
                # watched (reading the FIRST line reported width 1 for a drive re-run at 60)
                info.width = int(m.group(1))
                vm = DRIVE_START_VIEW_RE.search(line)
                if vm:
                    info.view = vm.group(1)
                im = DRIVE_START_ID_RE.search(line)
                if im:
                    info.drive_id = im.group(1)
                sm = DRIVE_START_STATE_RE.search(line)
                if sm:
                    info.attempt_dir = Path(sm.group(1))
                info.to_run, info.scheduled, info.already_complete = [], False, []   # the schedule that follows supersedes
            tm = TO_RUN_RE.search(line)
            if tm:
                info.to_run = [x.strip().strip("'\"") for x in tm.group(1).split(",") if x.strip()]
                info.scheduled = True
            lm = LAUNCHED_RE.search(line)
            if lm:
                info.out_roots_by_leg[lm.group(1)] = Path(lm.group(2))
            vm2 = LEG_VERDICT_RE.search(line)
            if vm2:
                info.verdicts[vm2.group(1)] = (vm2.group(2), vm2.group(3))
                info.run_key_by_leg.setdefault(vm2.group(1), vm2.group(4))
    # the drive records are the AUTHORITATIVE leg -> run_key -> out root map; the CURRENT drive's record (written right
    # after its schedule) also names its reconciled ``already_complete`` legs
    for rec_dir in [d, *(a for a in (d / "attempts").iterdir() if a.is_dir())] if (d / "attempts").is_dir() else [d]:
        for rec in sorted(rec_dir.glob("drive_*.json")):
            try:
                drec = json.loads(rec.read_text())
            except (OSError, ValueError):
                continue
            if info.drive_id is not None and drec.get("drive_id") == info.drive_id:
                ac = drec.get("already_complete")
                if isinstance(ac, list):
                    info.already_complete = [str(x) for x in ac]
                if not info.scheduled and isinstance(drec.get("schedule"), list):
                    names = [r.get("record_name") for r in drec["schedule"] if isinstance(r, dict) and r.get("record_name")]
                    info.to_run, info.scheduled = [n for n in names if n not in info.already_complete], True
            roots = {k: Path(v) for k, v in (drec.get("out_roots") or {}).items() if isinstance(v, str)}
            for row in drec.get("schedule") or []:
                if not isinstance(row, dict):
                    continue
                name = row.get("record_name") or (
                    f"leg_{row['cell_index']}_s{row['seed_base']}" if "cell_index" in row and "seed_base" in row else None)
                if not name:
                    continue
                rk = row.get("run_key")
                if isinstance(rk, str) and re.fullmatch(r"[0-9a-f]{16}", rk):
                    info.run_key_by_leg[name] = rk
                tag = row.get("tag")
                if isinstance(tag, str) and tag in roots:
                    info.out_roots_by_leg.setdefault(name, roots[tag])
            if isinstance(drec.get("width"), int) and info.width is None:
                info.width = int(drec["width"])
            if info.drive_id is None and isinstance(drec.get("drive_id"), str):
                info.drive_id = drec["drive_id"]
    return info


def leg_run_key(info: DriveInfo, leg: str, log_text: str) -> Optional[str]:
    rk = info.run_key_by_leg.get(leg)
    if rk:
        return rk
    m = RUN_KEY_IN_LOG_RE.findall(log_text)
    return m[-1] if m else None


# --- the Inspect eval logs: completed samples ----------------------------------------------------------------------

def eval_zip_sample_count(path: Path) -> Tuple[Optional[int], float]:
    """``(completed samples, mtime)`` of one ``.eval`` zip: every ``samples/*.json`` member is a COMPLETED
    sample (inspect's incremental recorder). The listing needs no decompression, so a zstd-compressed archive
    (method 93, which this interpreter cannot inflate) still counts. A torn/partial zip counts None."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None, 0.0
    try:
        with zipfile.ZipFile(path) as z:
            n = sum(1 for name in z.namelist() if name.startswith("samples/") and name.endswith(".json"))
    except (zipfile.BadZipFile, OSError):
        return None, mtime
    return n, mtime


def leg_progress(out_root: Optional[Path], run_key: Optional[str]) -> Tuple[Optional[int], Optional[float], Optional[int]]:
    """``(done, newest sample-write mtime, expected)`` for one leg from its ``inspect_logs/<run_key>/`` evals and
    ``cells/<run_key>/results.json``. None where the source is absent -- said as ``?`` in the snapshot, never 0."""
    if out_root is None or run_key is None:
        return None, None, None
    done: Optional[int] = None
    newest: Optional[float] = None
    ldir = out_root / "inspect_logs" / run_key
    if ldir.is_dir():
        done = 0
        for ev in sorted(ldir.rglob("*.eval")):
            n, mt = eval_zip_sample_count(ev)
            if n is not None:
                done += n
            newest = mt if newest is None else max(newest, mt)
    expected: Optional[int] = None
    rj = out_root / "cells" / run_key / "results.json"
    if rj.is_file():
        try:
            exp = (json.loads(rj.read_text()).get("audited") or {}).get("expected")
            expected = int(exp) if exp is not None else None
        except (OSError, ValueError, TypeError):
            expected = None
    return done, newest, expected


def sidecar_refusals(out_root: Optional[Path], run_key: Optional[str]) -> Tuple[int, int]:
    """Terminal refusals by name from the cell's sidecars (``petri_auditor_refused.json``,
    ``petri_judge_refused.json``): ``(auditor, judge)``; 0 when absent (the sidecar rides iff a refusal happened)."""
    if out_root is None or run_key is None:
        return 0, 0
    counts = []
    for name in ("petri_auditor_refused.json", "petri_judge_refused.json"):
        p = out_root / "cells" / run_key / name
        n = 0
        if p.is_file():
            try:
                d = json.loads(p.read_text())
                items = d.get("items") if isinstance(d, dict) else d
                if isinstance(items, list):
                    n = len(items)
                elif isinstance(d, dict) and isinstance(d.get("total"), int):
                    n = int(d["total"])
            except (OSError, ValueError):
                n = 0
        counts.append(n)
    return counts[0], counts[1]


# --- the watch --------------------------------------------------------------------------------------------------------

@dataclass
class LegView:
    key: str                     # ``<view>/<leg>``
    leg: str
    log: Path
    rc_line: Optional[str]
    rc: Optional[str]
    results: Optional[str]
    rc_drive: Optional[str]      # the .rc's ``drive=``
    rc_current: bool             # the .rc belongs to the drive being watched (its drive_id) -- else it is HISTORY
    size: int
    mtime: float
    width_seen: Optional[int]
    attempts: int
    refusals: List[RefusalRecord]
    loops: Dict[Tuple[str, str], int]
    done: Optional[int]
    expected: Optional[int]
    newest_sample: Optional[float]
    side_auditor: int
    side_judge: int
    last_stamp: Optional[str]    # the log's own last stamp (LOCAL), verbatim
    judge_refused_graceful: int  # ``Judge model refused to score`` lines (inspect_scout's graceful path)


def parse_rc(line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """``<rc> drive=<id> results=<sha|absent>`` -> (rc, results, drive)."""
    parts = line.strip().split()
    if not parts:
        return None, None, None
    rc = parts[0]
    results = next((p[len("results="):] for p in parts[1:] if p.startswith("results=")), None)
    drive = next((p[len("drive="):] for p in parts[1:] if p.startswith("drive=")), None)
    return rc, results, drive


def read_leg(info: DriveInfo, log: Path, *, loop_n: int, expected_default: int) -> LegView:
    leg = log.stem
    text = log.read_text(errors="replace")
    st = log.stat()
    records = parse_inspect_records(text)
    refs = refusal_records(records)
    rc_p = log.with_suffix(".rc")
    rc_line = rc_p.read_text(errors="replace").strip() if rc_p.is_file() else None
    rc, results, rc_drive = parse_rc(rc_line) if rc_line else (None, None, None)
    # a re-invoked drive appends to the SAME leg log and leaves the predecessor's .rc in place until its own leg
    # returns (seedrep_petri_drive.run_leg): a return record is terminal ONLY for the drive being watched. With no
    # drive_id known (a flat layout with no DRIVE_START), any .rc counts.
    rc_current = rc_line is not None and (info.drive_id is None or rc_drive is None or rc_drive == info.drive_id)
    headers = [m for m in (ATTEMPT_HEADER_RE.match(l) for l in text.splitlines()) if m]
    widths = [int(m.group(3)) for m in headers if m.group(3)]
    run_key = leg_run_key(info, leg, text)
    out_root = info.out_roots_by_leg.get(leg)
    done, newest, expected = leg_progress(out_root, run_key)
    sa, sj = sidecar_refusals(out_root, run_key)
    return LegView(
        key=f"{info.view}/{leg}", leg=leg, log=log, rc_line=rc_line, rc=rc, results=results, rc_drive=rc_drive,
        rc_current=rc_current, size=st.st_size,
        mtime=st.st_mtime, width_seen=(widths[-1] if widths else None), attempts=len(headers), refusals=refs,
        loops=refusal_loops(refs, loop_n=loop_n), done=done, expected=(expected if expected is not None else expected_default),
        newest_sample=newest, side_auditor=sa, side_judge=sj,
        last_stamp=(records[-1].stamp if records else None),
        judge_refused_graceful=text.count(JUDGE_REFUSED_TEXT))


def snapshot_line(v: LegView, *, judge_needles: Sequence[str], auditor_needles: Sequence[str]) -> str:
    roles = {"auditor": 0, "judge": 0, "unknown": 0}
    for r in v.refusals:
        if r.classifier:
            roles[classify_role(r.model, judge_needles=judge_needles, auditor_needles=auditor_needles)] += 1
    last_prog = v.newest_sample if v.newest_sample is not None else None
    last = utc_of(max(last_prog, v.mtime)) if last_prog is not None else utc_of(v.mtime)
    stamp_part = ""
    if v.last_stamp:
        conv = leg_stamp_to_utc(v.last_stamp)
        stamp_part = f" last_log_stamp={v.last_stamp}(local)={conv or '?'}"
    retry_max = max((r.retry for r in v.refusals if r.kind == KIND_CLASSIFIER), default=0)
    overload_max = max((r.retry for r in v.refusals if r.kind == KIND_OVERLOAD), default=0)
    done = "?" if v.done is None else str(v.done)
    return (f"LEG {v.key} samples {done}/{v.expected} last_progress {last} refusals auditor={roles['auditor'] + v.side_auditor} "
            f"judge={roles['judge'] + v.side_judge + v.judge_refused_graceful} loops={len(v.loops)} retry_max={retry_max} "
            f"overload_retry_max={overload_max} "
            f"rc={v.rc if (v.rc is not None and v.rc_current) else 'running'}"
            f"{'' if v.rc_line is None or v.rc_current else f' prior_rc={v.rc}@{v.rc_drive}'}{stamp_part}")


def tail(path: Path, n: int = 5) -> List[str]:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    return [l.rstrip() for l in lines[-n:]]


class Watch:
    """One monitor over N drive dirs; ``poll()`` returns the event lines of that poll (printed by ``main``)."""

    def __init__(self, dirs: Sequence[Path], *, stall_min: float, loop_n: int, expected_default: int,
                 judge_needles: Sequence[str], auditor_needles: Sequence[str], now: Any = time.time,
                 grace_min: float = 60.0) -> None:
        self.dirs = [Path(d) for d in dirs]
        self.stall_s = stall_min * 60.0
        self.grace_s = grace_min * 60.0
        self.started = now()
        self.seen_any_leg = False
        self.warned_no_legs = False
        self.seen_hist_rc: set = set()
        self.seen_complete_prior: set = set()
        self.seen_refusals: Dict[str, Dict[Tuple[str, str], int]] = {}
        self.seen_unscheduled: set = set()
        self.loop_n = loop_n
        self.expected_default = expected_default
        self.judge_needles = [n.lower() for n in judge_needles]
        self.auditor_needles = [n.lower() for n in auditor_needles]
        self.now = now
        self.seen_retry: Dict[str, int] = {}
        self.seen_loops: Dict[str, set] = {}
        self.seen_rc: set = set()
        self.seen_size: Dict[str, Tuple[int, float]] = {}
        self.stalled: set = set()
        self.warned_width: set = set()
        self.warned_engine: set = set()
        self.bad = False                    # any WIDTH-1 / REFUSAL-LOOP / STALL / LEG-DIED fired

    def _ev(self, out: List[str], text: str) -> None:
        out.append(f"{utc_now()} {text}")

    def start_checks(self) -> List[str]:
        out: List[str] = []
        for d in self.dirs:
            info = read_drive_info(d)
            if info.width == 1 and d not in self.warned_width:
                self.warned_width.add(d)
                self.bad = True
                self._ev(out, f"WIDTH-1 {info.view} -- DRIVE_START width=1 (PETRI_EVAL_MAX_CONNECTIONS=1): one blocked sample "
                              f"halts the whole leg; the intended width never took effect (D33). REFUSED by name: stop the "
                              f"drive and relaunch with --width >= 2 ({info.drive_log})")
        return out

    def poll(self, *, snapshot: bool) -> Tuple[List[str], bool]:
        """One pass over every dir; returns ``(lines, any_leg_live)``."""
        out: List[str] = []
        live = False
        now = self.now()
        for d in self.dirs:
            info = read_drive_info(d)
            logs = find_leg_logs(d, info.attempt_dir)
            for log in logs:
                leg = log.stem
                # bind to the CURRENT drive's schedule (fk2c): a leg it reconciled as already complete is
                # COMPLETE (its .rc keeps the previous drive id by design -- never live work); a leg it did not schedule
                # at all (a retained log of an older run) is history; only its ``to run`` legs are active work
                if info.scheduled and leg not in info.to_run:
                    key = f"{info.view}/{leg}"
                    if leg in info.already_complete:
                        if key not in self.seen_complete_prior:
                            self.seen_complete_prior.add(key)
                            rcl = log.with_suffix(".rc")
                            prior = parse_rc(rcl.read_text(errors="replace")) if rcl.is_file() else (None, None, None)
                            self._ev(out, f"LEG-COMPLETE-PRIOR {key} -- reconciled complete by the current drive "
                                          f"(drive_id={info.drive_id} already_complete; rc={prior[0]}@{prior[2]}); not re-driven")
                        if snapshot:
                            self._ev(out, f"LEG {key} complete (already_complete of drive {info.drive_id}; not this drive's work)")
                    elif key not in self.seen_unscheduled:
                        self.seen_unscheduled.add(key)
                        self._ev(out, f"LEG-UNSCHEDULED {key} -- a retained leg log the current drive (drive_id={info.drive_id}) "
                                      f"did not schedule (to run: {info.to_run}); history, not watched")
                    continue
                v = read_leg(info, log, loop_n=self.loop_n, expected_default=self.expected_default)
                key = v.key
                if v.width_seen == 1 and key not in self.warned_width:
                    self.warned_width.add(key)
                    self.bad = True
                    self._ev(out, f"WIDTH-1 {key} -- leg attempt header width=1: one blocked sample halts the leg (D33); "
                                  f"REFUSED by name")
                if snapshot:
                    self._ev(out, snapshot_line(v, judge_needles=self.judge_needles, auditor_needles=self.auditor_needles))
                if v.rc_line is not None and not v.rc_current:
                    if key not in self.seen_hist_rc:
                        self.seen_hist_rc.add(key)
                        self._ev(out, f"LEG-PRIOR-RC {key} rc={v.rc} drive={v.rc_drive} -- a PREVIOUS drive's return record "
                                      f"(current drive_id={info.drive_id}); history, not this drive's result; the leg is live")
                if v.rc_line is not None and v.rc_current:
                    if key not in self.seen_rc:
                        self.seen_rc.add(key)
                        if v.rc == RC_OK and v.results and v.results != "absent":
                            self._ev(out, f"LEG-DONE {key} rc={v.rc} results={v.results[:12]}")
                        else:
                            self.bad = True
                            self._ev(out, f"LEG-DIED {key} rc={v.rc} results={v.results or '?'} -- the leg RETURNED without "
                                          f"results (not a hang); tail: " + " | ".join(tail(log)))
                    continue
                live = True
                for kind, event, note in ((KIND_CLASSIFIER, "REFUSAL-RETRY", "a safety-classifier refusal; deterministic on this body, retries cannot succeed"),
                                          (KIND_OVERLOAD, "OVERLOAD-RETRY", "a streamed overload / internal server error, retried as transient by inspect_ai "
                                                                            "(should_retry :1323-1328) regardless of the status code; retries can succeed"),
                                          (KIND_UNKNOWN, "UNKNOWN-RETRY", "a retry whose body names neither a classifier refusal nor an overload (a 429 / bare 5xx)")):
                    recs = [r for r in v.refusals if r.kind == kind]
                    hi = max((r.retry for r in recs), default=0)
                    seen_key = (key, kind)
                    if hi > self.seen_retry.get(seen_key, 0):
                        self.seen_retry[seen_key] = hi
                        last = max(recs, key=lambda r: r.retry)
                        # elapsed since this belt's FIRST record: the earliest-stamped record of the SAME sample + model
                        same = [r for r in recs if r.sample == last.sample and r.model == last.model] or recs
                        stamps = [(leg_stamp_to_utc(r.stamp), r) for r in same]
                        conv = leg_stamp_to_utc(last.stamp)
                        conv0 = min((c for c, _r in stamps if c), default=None)
                        elapsed = ""
                        if conv and conv0:
                            t1 = datetime.strptime(conv, "%Y-%m-%dT%H:%M:%SZ"); t0 = datetime.strptime(conv0, "%Y-%m-%dT%H:%M:%SZ")
                            elapsed = f" elapsed_since_first={max(0, int((t1 - t0).total_seconds() // 60))}m"
                        self._ev(out, f"{event} {key} retry={hi} model={last.model} role="
                                      f"{classify_role(last.model, judge_needles=self.judge_needles, auditor_needles=self.auditor_needles)} "
                                      f"sample={last.sample} status={last.status if last.status is not None else '?'} "
                                      f"category={last.category or '?'} at={last.stamp}(local)={conv or '?'}{elapsed} -- {note}")
                counts: Dict[Tuple[str, str], int] = {}
                for r in v.refusals:
                    if r.kind == KIND_CLASSIFIER:
                        counts[(r.sample, r.model)] = counts.get((r.sample, r.model), 0) + 1
                seen_n = self.seen_refusals.setdefault(key, {})
                for (sample, model), n in sorted(counts.items()):
                    if n > seen_n.get((sample, model), 0) and n < self.loop_n:
                        seen_n[(sample, model)] = n
                        cat = next((r.category for r in reversed(v.refusals) if r.kind == KIND_CLASSIFIER and r.sample == sample and r.model == model and r.category), "?")
                        self._ev(out, f"REFUSAL {key} sample={sample} model={model} role="
                                      f"{classify_role(model, judge_needles=self.judge_needles, auditor_needles=self.auditor_needles)} "
                                      f"category={cat} n={n} -- a safety-classifier refusal of this conversation (terminal for the leg's "
                                      f"pinned model; the manifest's fallback chain is what scores it)")
                seen = self.seen_loops.setdefault(key, set())
                for (sample, model), n in sorted(v.loops.items()):
                    if (sample, model) not in seen:
                        seen.add((sample, model))
                        self.bad = True
                        self._ev(out, f"REFUSAL-LOOP {key} sample={sample} model={model} n={n} -- the same conversation refused "
                                      f"{n}x by the same model (classifier, deterministic): restart the leg on the fallback "
                                      f"(--help: ACTING ON A STALLED / LOOPING LEG); tail: " + " | ".join(tail(log)))
                # STALL: neither the leg log nor the leg's inspect logs were written for stall_s, judged from the
                # authoritative file MTIMES on EVERY observation including the FIRST (a fresh --once watcher
                # polls once, so "no growth since the last poll" could never fire and a three-hour-old active leg read
                # healthy). The leg log's own stamps are LOCAL time and are only ever PRINTED.
                newest = max([v.mtime] + ([v.newest_sample] if v.newest_sample is not None else []))
                age = now - newest
                self.seen_size[key] = (v.size, v.newest_sample or 0.0)
                if age > self.stall_s:
                    if key not in self.stalled:
                        self.stalled.add(key)
                        self.bad = True
                        self._ev(out, f"STALL {key} no log write and no new sample for {int(age // 60)}m (size {v.size}B, "
                                      f"samples {v.done if v.done is not None else '?'}/{v.expected}); tail: " + " | ".join(tail(log)))
                elif key in self.stalled:
                    self.stalled.discard(key)
                    self._ev(out, f"RESUMED {key} log written again after a stall")
            # engine: the seedrep proxy runs in Modal -- a vLLM log is only visible if someone mirrored it here
            v_log = next((p for p in (d / "serving" / "vllm_server.log", d.parent / "serving" / "vllm_server.log") if p.is_file()), None)
            if v_log is not None:
                vage = now - v_log.stat().st_mtime
                if vage > self.stall_s:
                    self._ev(out, f"ENGINE-QUIET {info.view} no vLLM log line for {int(vage // 60)}m -- target receiving no traffic")
            elif d not in self.warned_engine:
                self.warned_engine.add(d)
                self._ev(out, f"ENGINE-UNOBSERVED {info.view} no serving/vllm_server.log under {d} -- the proxy runs in Modal; "
                              f"engine liveness is not visible from this box (an auditor-side block shows up as a leg stall instead)")
            # legs the drive has scheduled but not launched yet (bundle waits) keep the watch alive
            launched = {p.stem for p in logs}
            pending = [l for l in info.to_run if l not in launched]
            if pending:
                live = True
            if logs:
                self.seen_any_leg = True
        # startup grace: a drive that has not written a leg log yet -- tunnel wait,
        # bundle materialisation -- is not "all legs terminal"; say NO-LEGS-STARTED once the grace passes and keep
        # watching rather than exiting as if the work had finished
        if not self.seen_any_leg:
            live = True
            if now - self.started > self.grace_s and not self.warned_no_legs:
                self.warned_no_legs = True
                self.bad = True
                self._ev(out, f"NO-LEGS-STARTED after {int((now - self.started) // 60)}m -- no drive wrote a leg log yet "
                              f"(tunnel wait / bundle materialisation exceeded the grace; check drive.log)")
        return out, live


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="drive wave state dir(s) / attempt dir(s) / any dir holding drive.log + leg_*.log")
    ap.add_argument("--once", action="store_true", help="one snapshot + one pass of events, then exit 0 (healthy) / 3 (act)")
    ap.add_argument("--poll-s", type=float, default=60.0)
    ap.add_argument("--stall-min", type=float, default=15.0, help="no leg-log growth AND no new sample for this long = STALL (D113: 15)")
    ap.add_argument("--loop-n", type=int, default=2, help="the same (sample, model) classifier-refused this many times = REFUSAL-LOOP (D113: 2)")
    ap.add_argument("--max-hours", type=float, default=48.0)
    ap.add_argument("--expected-samples", type=int, default=DEFAULT_EXPECTED_SAMPLES,
                    help="the per-leg sample total when the cell's results.json is not there yet (6 behaviors x 10 scenarios)")
    ap.add_argument("--grace-min", type=float, default=60.0, help="startup grace before NO-LEGS-STARTED: no leg log yet is not terminal")
    ap.add_argument("--status-every", type=int, default=0, help="in loop mode, print the LEG snapshot every N polls (0 = transitions only)")
    ap.add_argument("--judge-model", default="opus", help="comma-separated substrings naming the JUDGE model in retry records")
    ap.add_argument("--auditor-model", default="sonnet,grok", help="comma-separated substrings naming the AUDITOR chain's models")
    args = ap.parse_args(argv)
    dirs = [Path(os.path.expanduser(d)) for d in args.dirs]
    missing = [str(d) for d in dirs if not d.is_dir()]
    if missing:
        print(f"{utc_now()} REFUSED no such dir(s): {missing}", flush=True)
        return 2
    w = Watch(dirs, stall_min=args.stall_min, loop_n=args.loop_n, expected_default=args.expected_samples,
              judge_needles=args.judge_model.split(","), auditor_needles=args.auditor_model.split(","),
              grace_min=args.grace_min)
    for line in w.start_checks():
        print(line, flush=True)
    if args.once:
        lines, _live = w.poll(snapshot=True)
        for line in lines:
            print(line, flush=True)
        return 3 if w.bad else 0
    end = time.time() + args.max_hours * 3600.0
    n = 0
    while True:
        if time.time() >= end:
            print(f"{utc_now()} WATCH-TIMEOUT after {args.max_hours}h", flush=True)
            return 4
        n += 1
        lines, live = w.poll(snapshot=(args.status_every > 0 and n % args.status_every == 0))
        for line in lines:
            print(line, flush=True)
        if not live:
            print(f"{utc_now()} ALL-LEGS-TERMINAL", flush=True)
            return 0
        time.sleep(args.poll_s)


if __name__ == "__main__":
    sys.exit(main())
