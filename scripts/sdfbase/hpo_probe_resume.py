"""hpo_probe_resume — the resume line of an INCOMPLETE serving-probe record, derived from the record.

``reproduce/03_hpo.sh --phase probe`` resumes a probe record that ``hpo_probe_status.py`` reports INCOMPLETE_STALLED (a
partial checkpoint, no job) by submitting attempt+1 of the SAME record through ``lib/slurm.sh``. Every input of that
submission comes from a record, never from a re-typed line (a hand-submitted attempt built from the printed template
leaves no ledger line):

* positionals 4-6 (sets, ranks, mode) — the record's ``probe_plan.json`` (the driver refuses a ``--plan-args`` disagreement
  on a resume, so the plan IS the authority);
* the attempt counter and the cap — the carrier's ``probe_attempt.json`` (scripts/sbatch/serving_probe.sbatch writes it at
  the start of every attempt: attempt, max_attempts, gpus, parallel, mode, sets, ranks, job, sha, …); for a record written
  before that file existed, the carrier's per-exit head line ``state=… attempt=N/M … job=J`` (the first line of its
  figure-dir RESULT, ``$SHARED/exp/figs/serving-probe/<record basename>/RESULT`` — written on EVERY exit; the record dir's own
  RESULT is the driver's verdict line, written once, and is NOT an attempt record);
* the width — ``--width record`` (default: the last attempt's ``gpus``), ``auto`` (the sbatch header's) or ``N``; a record
  with no recorded width refuses ``record`` by name (pass ``--probe-width``);
* the ledger — an attempt number the front door already issued (a ``submitted`` / ``registered`` binding for
  ``<lwid>.a<k>`` in ``lib/stage.py``'s bindings dir) is skipped with a note, so a job that died before its carrier wrote
  the attempt record never collides with the funnel's idempotence.

Refusals (exit 2, ``REFUSED: …`` on stderr): no attempt record of either form; the cap reached (attempt N == max: the
carrier's own escalate() would not continue either — a wider cap is a new decision, taken by hand); the plan and the attempt
record disagreeing on sets / ranks / mode; a malformed record; an unrecorded width under ``record``.

    printf '%s' "$(cat probe_plan.json)" | python hpo_probe_resume.py --plan - --attempt-json "$(cat probe_attempt.json)" \\
        --width record --ledger <state>/<model>/submissions.jsonl --lwid-prefix <model>/03_hpo/lora/hpo.lora.probe
    -> attempt=3 / max_attempts=4 / sets=… / ranks=… / mode=tp1 / width=4 / parallel=auto / unit_suffix=a3 / source=… [/ note=…]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

ATTEMPT_FILE = "probe_attempt.json"
LEGACY_HEAD_RE = re.compile(r"\battempt=(\d+)/(\d+)\b")
LEGACY_JOB_RE = re.compile(r"\bjob=(\S+)")
LEGACY_STATE_RE = re.compile(r"\bstate=(\S+)")
SOURCE_ATTEMPT_RECORD = ATTEMPT_FILE
SOURCE_LEGACY_RESULT = "the carrier's RESULT head (figure dir)"
WIDTH_RECORD, WIDTH_AUTO = "record", "auto"
PARALLEL_AUTO = "auto"     # positional 9: the driver derives the concurrency from the allocation
#: the attempt-scoped unit suffix: attempt k of the probe is the logical work id `<probe lwid>.a<k>`
UNIT_SUFFIX = "a{n}"
BOUND_STATES = ("submitted", "registered")


class Refused(Exception):
    pass


#: a head written by a single-shot carrier phase (reverdict / reference) is not the record's attempt counter
SINGLE_SHOT_PHASE_RE = re.compile(r"\bphase=(reverdict|reference)\b")


def parse_legacy_head(line: str) -> Optional[Dict[str, Any]]:
    """The carrier's per-exit head line -> {attempt, max_attempts, job_id, state}; None when the line carries no attempt, or
    was stamped by a single-shot phase (``phase=reverdict|reference``: those preserve the standing counter since 2026-10-01 and
    say so; an older such head is not a counter either)."""
    if SINGLE_SHOT_PHASE_RE.search(line or ""):
        return None
    m = LEGACY_HEAD_RE.search(line or "")
    if not m:
        return None
    job = LEGACY_JOB_RE.search(line); st = LEGACY_STATE_RE.search(line)
    return {"attempt": int(m.group(1)), "max_attempts": int(m.group(2)),
            "job_id": job.group(1) if job else None, "state": st.group(1) if st else None}


def _pos_int(v: Any, what: str) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        raise Refused(f"{what} is not an integer ({v!r})") from None
    if n < 1:
        raise Refused(f"{what} must be >= 1 (got {n})")
    return n


def last_attempt(attempt_json: str, legacy_line: str) -> Tuple[Dict[str, Any], str]:
    """The record's LAST attempt -> (record, source): probe_attempt.json first, else the legacy RESULT head; else a refusal."""
    if (attempt_json or "").strip():
        try:
            rec = json.loads(attempt_json)
        except ValueError as e:
            raise Refused(f"{ATTEMPT_FILE} is not JSON ({e})") from None
        if not isinstance(rec, dict):
            raise Refused(f"{ATTEMPT_FILE} is not a JSON object")
        out = {"attempt": _pos_int(rec.get("attempt"), f"{ATTEMPT_FILE} attempt"),
               "max_attempts": _pos_int(rec.get("max_attempts"), f"{ATTEMPT_FILE} max_attempts"),
               "gpus": (_pos_int(rec["gpus"], f"{ATTEMPT_FILE} gpus") if rec.get("gpus") not in (None, "") else None),
               "parallel": str(rec.get("parallel") or PARALLEL_AUTO), "mode": rec.get("mode"), "sets": rec.get("sets"),
               "ranks": rec.get("ranks"), "job_id": rec.get("job_id"), "sha": rec.get("sha")}
        return out, SOURCE_ATTEMPT_RECORD
    head = parse_legacy_head(legacy_line or "")
    if head is not None:
        return {**head, "gpus": None, "parallel": PARALLEL_AUTO, "mode": None, "sets": None, "ranks": None, "sha": None}, SOURCE_LEGACY_RESULT
    raise Refused(f"the record carries no attempt record: no {ATTEMPT_FILE} (the carrier writes it at the start of every attempt "
                  "since 2026-09-30) and no `attempt=N/M` head in the carrier's figure-dir RESULT — the attempt counter cannot be "
                  "derived, and it is never re-typed; resume this record by hand with the carrier's documented positional line")


def plan_positionals(plan: Dict[str, Any]) -> Tuple[str, str, str]:
    """probe_plan.json -> (sets csv, ranks csv, mode) — the carrier's positionals 4-6."""
    sets = plan.get("sets"); ranks = plan.get("ranks"); tp = plan.get("tp")
    if not isinstance(sets, list) or not sets or not all(isinstance(s, str) and s for s in sets):
        raise Refused(f"probe_plan.json names no module sets ({sets!r})")
    if not isinstance(ranks, list) or not ranks:
        raise Refused(f"probe_plan.json names no ranks ({ranks!r})")
    try:
        tp_i = int(tp)
    except (TypeError, ValueError):
        raise Refused(f"probe_plan.json tp is not an integer ({tp!r})") from None
    if tp_i not in (1, 2):
        raise Refused(f"probe_plan.json tp={tp_i}: the carrier knows tp1 | tp2 only")
    return ",".join(sets), ",".join(str(int(r)) for r in ranks), f"tp{tp_i}"


def check_agreement(last: Dict[str, Any], sets: str, ranks: str, mode: str) -> None:
    """The attempt record (when it names them) must agree with the plan on positionals 4-6 — the driver would refuse anyway."""
    diffs = []
    if last.get("sets") and str(last["sets"]) != sets:
        diffs.append(f"sets {last['sets']} (attempt record) != {sets} (plan)")
    if last.get("ranks") and str(last["ranks"]) != ranks:
        diffs.append(f"ranks {last['ranks']} (attempt record) != {ranks} (plan)")
    if last.get("mode") and str(last["mode"]) != mode:
        diffs.append(f"mode {last['mode']} (attempt record) != {mode} (plan)")
    if diffs:
        raise Refused(f"{ATTEMPT_FILE} and probe_plan.json disagree: " + "; ".join(diffs) + " — a different probe needs a different record")


def resolve_width(width: str, last: Dict[str, Any]) -> str:
    """`record` -> the last attempt's gpus (refused when unrecorded); `auto` -> auto; `N` -> N."""
    w = (width or WIDTH_RECORD).strip()
    if w == WIDTH_RECORD:
        if last.get("gpus") is None:
            raise Refused("the record's last attempt has no recorded width (a record written before probe_attempt.json, or a "
                          "record with no gpus field) — pass --probe-width N|auto (N GPUs on the command line, auto = the sbatch "
                          "header's 8)")
        return str(last["gpus"])
    if w == WIDTH_AUTO:
        return WIDTH_AUTO
    if not re.fullmatch(r"[1-9][0-9]*", w):
        raise Refused(f"--width must be record | auto | a positive integer, got {w!r}")
    return w


def ledger_bound(ledger: Optional[str], lwid: str) -> bool:
    """Whether the front door's ledger already binds `lwid` (submitted / registered) — lib/stage.py's bindings beside the ledger."""
    if not ledger:
        return False
    lib = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "reproduce", "lib")
    if lib not in sys.path:
        sys.path.insert(0, lib)
    import stage as ST  # noqa: E402  (stdlib-only module of the front door)
    return ST.binding_state(ledger, lwid)["state"] in BOUND_STATES


def next_attempt(last: Dict[str, Any], *, ledger: Optional[str], lwid_prefix: Optional[str],
                 max_attempts: Optional[int] = None, cap_reason: Optional[str] = None) -> Tuple[int, List[str]]:
    """attempt+1 of the record, skipping attempt numbers the funnel already issued; refused at the cap. ``max_attempts`` (the front
    door's --probe-max-attempts) is an OPERATOR cap widen: it replaces the record's cap for this resume — refused below the
    next attempt (a cap the next attempt does not fit under widens nothing) — and the carrier writes it into probe_attempt.json, so
    every later derive reads the widened cap from the record."""
    notes: List[str] = []
    n = int(last["attempt"]) + 1
    cap = int(last["max_attempts"])
    if max_attempts is not None:
        if int(max_attempts) < n:
            raise Refused(f"--max-attempts {max_attempts} is below the next attempt {n} (last attempt {last['attempt']}/{cap}"
                          + (f", job {last.get('job_id')}" if last.get("job_id") else "") + ") — a cap widen names a cap the next attempt fits under")
        if int(max_attempts) != cap:
            # a cap raised with no reason leaves a record that authorised more GPU attempts with nothing in it saying
            # who raised the cap or why (an incomplete record mints no receipt; a later attempt rewrote cap_widen to null) — the reason is
            # REQUIRED whenever the cap changes, and the carrier appends it to the record's cap_history
            if not (cap_reason or "").strip():
                raise Refused(f"--max-attempts {max_attempts} changes the record's cap {cap} and no reason was given (--cap-reason / 03_hpo.sh "
                              "--probe-cap-reason) — a cap widen records who raised the cap and why, on the record itself")
            notes.append(f"cap widened {cap} -> {int(max_attempts)} by the operator (--probe-max-attempts): the carrier appends it to {ATTEMPT_FILE}'s "
                         "cap_history (carried forward by every later attempt) and the front door to the reopen receipt's reason when one is minted")
        cap = int(max_attempts)
    if lwid_prefix and ledger:
        while n <= cap and ledger_bound(ledger, f"{lwid_prefix}.{UNIT_SUFFIX.format(n=n)}"):
            notes.append(f"the front door's ledger already binds attempt {n} of this record ({lwid_prefix}.{UNIT_SUFFIX.format(n=n)}: a "
                         f"job that left the queue before its carrier wrote {ATTEMPT_FILE}) — taking attempt {n + 1}")
            n += 1
    if n > cap:
        raise Refused(f"the record's attempt cap {cap} is reached (last attempt {last['attempt']}"
                      + (f", job {last.get('job_id')}" if last.get("job_id") else "") + f"; next would be {n}) — the carrier's own "
                      "escalate() stops at the cap too; raising it is a new decision: resume by hand with positional 8 = a wider cap")
    return n, notes


def derive(plan: Dict[str, Any], *, attempt_json: str, legacy_line: str, width: str, ledger: Optional[str],
           lwid_prefix: Optional[str], max_attempts: Optional[int] = None, cap_reason: Optional[str] = None) -> Dict[str, Any]:
    last, source = last_attempt(attempt_json, legacy_line)
    sets, ranks, mode = plan_positionals(plan)
    check_agreement(last, sets, ranks, mode)
    n, notes = next_attempt(last, ledger=ledger, lwid_prefix=lwid_prefix, max_attempts=max_attempts, cap_reason=cap_reason)
    cap = int(max_attempts) if max_attempts is not None else int(last["max_attempts"])
    cap_widen = f"{int(last['max_attempts'])}->{cap}" if cap != int(last["max_attempts"]) else None
    w = resolve_width(width, last)
    if source == SOURCE_LEGACY_RESULT:
        notes.append(f"the attempt counter was read from {SOURCE_LEGACY_RESULT}: state={last.get('state')} attempt={last['attempt']}/"
                     f"{last['max_attempts']} job={last.get('job_id')} — a record written before {ATTEMPT_FILE} existed")
    return {"attempt": n, "max_attempts": cap, "sets": sets, "ranks": ranks, "mode": mode, "width": w,
            "parallel": PARALLEL_AUTO, "unit_suffix": UNIT_SUFFIX.format(n=n), "source": source,
            "last_attempt": last["attempt"], "last_job_id": last.get("job_id"), "cap_widen": cap_widen, "notes": notes}


def main(argv=None, stdin=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", required=True, help="probe_plan.json (a path, or - for stdin)")
    ap.add_argument("--attempt-json", default="", help=f"the text of the record's {ATTEMPT_FILE} ('' when absent)")
    ap.add_argument("--legacy-result", default="", help="the first line of the carrier's figure-dir RESULT ('' when absent)")
    ap.add_argument("--width", default=WIDTH_RECORD, help="record | auto | N")
    ap.add_argument("--ledger", default=None, help="the front door's submissions.jsonl (its bindings dir sits beside it)")
    ap.add_argument("--lwid-prefix", default=None, help="the probe unit's logical work id; attempt k binds as <prefix>.a<k>")
    ap.add_argument("--max-attempts", default=None, type=int,
                    help="an operator cap widen (03_hpo.sh --probe-max-attempts, D549): the cap for this resume, >= the next attempt")
    ap.add_argument("--cap-reason", default=None, help="why the cap changes (required whenever --max-attempts differs from the record's cap)")
    a = ap.parse_args(argv)
    try:
        raw = (stdin if stdin is not None else sys.stdin).read() if a.plan == "-" else open(a.plan, encoding="utf-8").read()
        try:
            plan = json.loads(raw)
        except ValueError as e:
            raise Refused(f"probe_plan.json is not JSON ({e})") from None
        if not isinstance(plan, dict):
            raise Refused("probe_plan.json is not a JSON object")
        spec = derive(plan, attempt_json=a.attempt_json, legacy_line=a.legacy_result, width=a.width, ledger=a.ledger,
                      lwid_prefix=a.lwid_prefix, max_attempts=a.max_attempts, cap_reason=a.cap_reason)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    for k in ("attempt", "max_attempts", "sets", "ranks", "mode", "width", "parallel", "unit_suffix", "source", "last_attempt", "last_job_id", "cap_widen"):
        print(f"{k}={'' if spec[k] is None else spec[k]}")
    for n in spec["notes"]:
        print(f"note={n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
