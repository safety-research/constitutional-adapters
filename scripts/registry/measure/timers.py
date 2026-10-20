#!/usr/bin/env python3
"""timers.py — the registry timers' own cycle from the user journal.

    python scripts/registry/measure/timers.py [--since=-24h] [--units registry-harvest ...] [--json] [--from FILE]

(``--since=-24h`` with the ``=``: argparse reads a bare ``-24h`` as an option.)

One ``journalctl --user -u <unit>.service -o json`` per unit (or ``--from FILE``: a saved export, one JSON object per line).
Lines are grouped into PASSES by invocation id (``_SYSTEMD_INVOCATION_ID`` on the program's lines, ``USER_INVOCATION_ID`` on
systemd's). Per pass: start (the ``Starting …`` line or the first line), end (``Finished …`` / ``Failed with result '…'``),
outcome (``ok`` / ``exit-code`` / ``timeout`` / ``killed`` / ``running``), whether any line names ``LockHeld``, the program's
single-line ``{"ingest_ok": …}`` and its indent-1 summary JSON reassembled (``{`` … ``}``). Per unit: pass count and outcomes,
duration median / mean / min / max (s), ``LockHeld`` count, timeouts, start-to-start cadence, and the latest summary's keys.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from typing import Dict, Iterable, List, Optional

# registry-cluster-reads (the tick) replaces registry-adapters (retired in C6); the retired unit's summary keys stay
# so a saved export from before the cutover still reads. registry-modal-pull left the set with the Modal transport (C10).
UNITS = ["registry-harvest", "registry-cluster-reads", "registry-status", "registry-status-heavy", "registry-mirrors", "registry-snapshot"]
SUMMARY_KEYS = {"registry-harvest": ["files", "records", "ok", "refused", "blocked", "transactions", "write_s", "parse_s", "lock_hold_s",
                                     "changed", "self_change_only"],
                "registry-adapters": ["adapters", "inputs_seen", "inputs_changed", "groups_refreshed", "groups_held", "groups_refused",
                                      "records_written", "withdrawals"],
                "registry-cluster-reads": ["op", "state", "reservation", "changed", "skipped", "unchanged", "array", "watch", "pulled", "refused",
                                           "receipt", "submit_section_s", "release"],
                "registry-status-heavy": ["skipped", "initialized_now", "head", "head_recorded", "members", "cleared", "joined", "failed"]}


def journal_lines(unit: str, since: str) -> Iterable[Dict[str, object]]:
    cmd = ["journalctl", "--user", "-u", f"{unit}.service", "-o", "json", "--no-pager", "-S", since]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)}: rc={r.returncode}: {r.stderr.strip()[:200]}")
    for raw in r.stdout.splitlines():
        raw = raw.strip()
        if raw:
            try:
                yield json.loads(raw)
            except ValueError:
                continue


def _unit_of(d: Dict[str, object]) -> Optional[str]:
    u = d.get("USER_UNIT") or d.get("UNIT") or d.get("_SYSTEMD_USER_UNIT")
    if isinstance(u, str) and u.endswith(".service"):
        return u[: -len(".service")]
    ident = d.get("SYSLOG_IDENTIFIER")
    return ident if isinstance(ident, str) and ident.startswith("registry-") else None


def passes_from(lines: Iterable[Dict[str, object]], unit: Optional[str] = None) -> Dict[str, List[Dict[str, object]]]:
    """Group journal lines into passes: ``{unit: [pass, ...]}`` sorted by start time."""
    by_inv: Dict[str, Dict[str, object]] = {}
    for d in lines:
        inv = d.get("_SYSTEMD_INVOCATION_ID") or d.get("USER_INVOCATION_ID") or d.get("INVOCATION_ID")
        if not isinstance(inv, str):
            continue
        u = _unit_of(d) or unit
        if u is None:
            continue
        ts = d.get("__REALTIME_TIMESTAMP")
        try:
            t = int(ts) / 1e6  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        msg = str(d.get("MESSAGE") or "")
        p = by_inv.setdefault(inv, {"invocation": inv, "unit": u, "first": t, "last": t, "start": None, "end": None, "outcome": "running",
                                    "lock_held": False, "ingest_ok": None, "summaries": [], "_buf": None, "lines": 0, "exit_status": None})
        p["first"] = min(p["first"], t); p["last"] = max(p["last"], t); p["lines"] += 1
        if d.get("SYSLOG_IDENTIFIER") == "systemd" or d.get("_COMM") == "systemd":
            if msg.startswith("Starting "):
                p["start"] = t
            elif msg.startswith("Finished "):
                p["end"] = t; p["outcome"] = "ok"
            elif "Failed with result" in msg:
                p["end"] = t
                # systemd's UNIT_RESULT field first; else the quoted result in the message; else "failed"
                p["outcome"] = str(d.get("UNIT_RESULT") or (msg.split("'")[1] if "'" in msg else "failed"))
            elif "Main process exited" in msg:
                p["exit_status"] = f'{d.get("EXIT_CODE")}/{d.get("EXIT_STATUS")}'
            continue
        if "LockHeld" in msg:
            p["lock_held"] = True
        stripped = msg.strip()
        if stripped.startswith('{"ingest_ok"'):
            try:
                p["ingest_ok"] = json.loads(stripped)
            except ValueError:
                pass
            continue
        if stripped == "{":
            p["_buf"] = ["{"]
        elif p["_buf"] is not None:
            p["_buf"].append(msg)  # type: ignore[union-attr]
            if stripped == "}":
                try:
                    p["summaries"].append(json.loads("\n".join(p["_buf"])))  # type: ignore[union-attr, arg-type]
                except ValueError:
                    pass
                p["_buf"] = None
        elif stripped.startswith("{") and stripped.endswith("}"):
            try:
                p["summaries"].append(json.loads(stripped))  # type: ignore[union-attr]
            except ValueError:
                pass
    out: Dict[str, List[Dict[str, object]]] = {}
    for p in by_inv.values():
        p.pop("_buf", None)
        start = p["start"] if p["start"] is not None else p["first"]
        end = p["end"] if p["end"] is not None else (p["last"] if p["outcome"] != "running" else None)
        p["start_s"] = start; p["end_s"] = end
        p["duration_s"] = round(end - start, 3) if end is not None else None  # type: ignore[operator]
        out.setdefault(str(p["unit"]), []).append(p)
    for u in out:
        out[u].sort(key=lambda p: p["start_s"])  # type: ignore[arg-type, return-value]
    return out


def _stats(vals: List[float]) -> Dict[str, Optional[float]]:
    if not vals:
        return {"median": None, "mean": None, "min": None, "max": None}
    return {"median": round(statistics.median(vals), 1), "mean": round(statistics.mean(vals), 1), "min": round(min(vals), 1), "max": round(max(vals), 1)}


def unit_table(passes: List[Dict[str, object]], unit: str) -> Dict[str, object]:
    durs = [float(p["duration_s"]) for p in passes if p["duration_s"] is not None]
    outcomes: Dict[str, int] = {}
    for p in passes:
        outcomes[str(p["outcome"])] = outcomes.get(str(p["outcome"]), 0) + 1
    starts = [float(p["start_s"]) for p in passes]
    cadence = [b - a for a, b in zip(starts, starts[1:])]
    latest = next((p for p in reversed(passes) if p["summaries"]), None)
    latest_summary = None
    if latest is not None:
        s = latest["summaries"][-1]  # type: ignore[index]
        keys = SUMMARY_KEYS.get(unit)
        latest_summary = {k: s.get(k) for k in keys if k in s} if keys else s
    return {"unit": unit, "passes": len(passes), "outcomes": outcomes, "ok": outcomes.get("ok", 0),
            "lock_held": sum(1 for p in passes if p["lock_held"]), "timeouts": outcomes.get("timeout", 0),
            "duration_s": _stats(durs), "start_to_start_s": _stats(cadence),
            "ingest_ok_passes": sum(1 for p in passes if p["ingest_ok"] is not None),
            "records_written_gt0": sum(1 for p in passes for s in p["summaries"] if isinstance(s, dict) and (s.get("records_written") or 0) > 0),
            "latest_summary": latest_summary, "first_start": passes[0]["start_s"] if passes else None, "last_start": passes[-1]["start_s"] if passes else None}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", default="-24h", help="journalctl -S (default -24h)")
    ap.add_argument("--units", nargs="*", default=UNITS)
    ap.add_argument("--from", dest="from_file", help="a saved `journalctl -o json` export instead of the live journal")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.from_file:
        with open(a.from_file, "r", encoding="utf-8") as fh:
            lines = [json.loads(l) for l in fh if l.strip()]
        grouped = passes_from(lines)
    else:
        grouped = {}
        for u in a.units:
            grouped.update(passes_from(journal_lines(u, a.since), unit=u))
    tables = {u: unit_table(grouped.get(u, []), u) for u in (a.units if not a.from_file else sorted(grouped))}
    if a.json:
        print(json.dumps({"since": a.since, "units": tables}, indent=1, default=str))
        return 0
    print(f"journal since {a.since}")
    print("| unit | passes | ok | exit-code | timeout | other | LockHeld | duration s median (min–max) | start→start s median | records_written>0 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for u, t in tables.items():
        oc = dict(t["outcomes"])  # type: ignore[arg-type]
        other = sum(v for k, v in oc.items() if k not in ("ok", "exit-code", "timeout"))
        d = t["duration_s"]; c = t["start_to_start_s"]
        print(f'| {u} | {t["passes"]} | {oc.get("ok", 0)} | {oc.get("exit-code", 0)} | {oc.get("timeout", 0)} | {other} | {t["lock_held"]} | '
              f'{d["median"]} ({d["min"]}–{d["max"]}) | {c["median"]} | {t["records_written_gt0"]} |')
    for u, t in tables.items():
        if t["latest_summary"]:
            print(f"\n{u} latest summary: {json.dumps(t['latest_summary'], default=str)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
