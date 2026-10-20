#!/usr/bin/env python
"""seedrep_watch.py — the wave's watcher: one process per leg (`--leg`), plus `--pools` and `--petri-drive`;
polls every 120 s and prints ONE LINE per notable event, exiting at the leg's terminal state — armed with
`Monitor(..., persistent: true)` in the SAME turn as each spawn (the "never leave a running job unwatched" rule).

  STARTED   <standup_id> attempt=<id> host=...                 the first heartbeat of a call
  PROGRESS  <standup_id> rows=<n> rows/min=<r> running=<S_live> phase=<p> eta=<h>   every --progress-min (30) while running
  STALL     <standup_id> no heartbeat for N min | rows flat N min      (--stall-min 20; rows-flat 40 for multi-turn / Petri)
  FAILED    <standup_id> call=<id> <error | RESULT rc!=0> + the last 40 lines of the container's driver log
  PRODUCER_MISSING <leg>/<view>/<stage>  every attempt terminal, none accepted (the queue stays open until a re-spawn)
  DONE      <leg> scope=<base|rungs|all> required=<n> accepted=<n>   every REQUIRED control key of the scope has an accepted attempt -> exit 0
            (the required set is DERIVED from the plan; retries of one key never add coverage)
  WATCH-TIMEOUT                                                 -> exit 3 (--max-hours 24; re-arm if the leg is alive)
  POOL_DOWN / POOL_UP <pool>                                    (--pools) the deployed servers' /health
  DRIVE_STALL / DRIVE_DONE                                      (--petri-drive) the grading host's Petri drive heartbeat file

Inputs: `RS/calls.jsonl` (the spawn ledger `seedrep_served.spawn_row` writes), `RS/launch_plan.json` (the rows + Dict names),
the wave's heartbeats / accepted Dicts, the attempt roots on the volume (RESULT, COMPLETE.json). The classification is a
PURE function (`classify`) over a snapshot the `ModalBackend` gathers, so the tests drive it without Modal.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seedrep_launch_plan as lp  # noqa: E402  (required_accepted_keys: the plan-derived coverage the watcher needs)

#: the battery's run-state directory under the site's state root ($STATE); expanded through sisteer.site after parsing
RS_DEFAULT = "$STATE/romp/candidate-si-refit/run-state/full22/seedrep_battery"
EVENTS = ("STARTED", "PROGRESS", "STALL", "FAILED", "PRODUCER_MISSING", "DONE", "WATCH-TIMEOUT", "POOL_DOWN", "POOL_UP",
          "DRIVE_STALL", "DRIVE_DONE", "WAVE_LAUNCHED", "WAVE_DONE", "HOLD")
#: the three SUMMARY events pushed to the user's phone -- ONE line each, `<EVENT> <json>` (machine-readable)
SUMMARY_EVENTS = ("WAVE_LAUNCHED", "WAVE_DONE", "HOLD")


def summary_line(event: str, payload: Mapping[str, Any]) -> str:
    assert event in SUMMARY_EVENTS, event
    return f"{event} " + json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str)


def s5_bars_of(rs: Optional[Path]) -> Dict[str, Any]:
    """The utilisation bars a WAVE_LAUNCHED line carries: the utilisation thresholds + the S5 record's `bars` block when RS/smokes.json has one."""
    out: Dict[str, Any] = {"thresholds": dict(lp.UTILISATION_BARS), "s5_bars_met": None, "s5_bars_source": None}
    p = Path(rs) / "smokes.json" if rs is not None else None
    if p is not None and p.is_file():
        try:
            bars = ((json.loads(p.read_text()).get("S5") or {}).get("bars")) or {}
            out.update(s5_bars_met=bars.get("bars_met"), s5_bars_source=bars.get("source"))
        except ValueError:
            out["s5_bars_source"] = "RS/smokes.json unreadable"
    return out
ROWS_FLAT_LEGS = ("pair", "crescendo", "actorattack", "petri")


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_utc(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        return time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
    except (ValueError, TypeError):
        return None


def load_calls(rs: Path, leg: str, wave_uuid: str) -> List[dict]:
    p = rs / "calls.jsonl"
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("leg") == leg and rec.get("wave_uuid") == wave_uuid:
            out.append(rec)
    return out


class WatchState:
    """Per-call memory across polls: first heartbeat seen, last heartbeat time, last rows + when they moved, last progress print."""

    def __init__(self) -> None:
        self.started: Dict[str, bool] = {}
        self.last_hb_t: Dict[str, float] = {}
        self.last_rows: Dict[str, int] = {}
        self.rows_moved_t: Dict[str, float] = {}
        self.last_progress_t: Dict[str, float] = {}
        self.reported_terminal: set = set()
        self.first_rows_t: Dict[str, float] = {}
        self.pool_up: Dict[str, bool] = {}
        self.wave_launched: bool = False                  # WAVE_LAUNCHED once per watch
        self.reported_hold: set = set()                   # HOLD once per held standup


def classify(*, now: float, leg: str, calls: List[dict], snapshot: Mapping[str, Any], state: WatchState,
             stall_min: float, rows_flat_min: float, progress_min: float, required_keys: Mapping[str, str],
             scope: str = "base", holds: Optional[Mapping[str, Mapping[str, Any]]] = None, bars: Optional[Mapping[str, Any]] = None,
             wave: Optional[Mapping[str, Any]] = None) -> List[str]:
    """One poll's events. ``snapshot`` = {call_id: {"done": bool, "error": str|None, "summary": dict|None,
    "heartbeat": {"t","rows_written","running","phase","attempt_id"}|None, "result_line": str|None, "log_tail": str,
    "accepted": str|None}}. ``required_keys`` = ``{accepted_key: standup_id}`` DERIVED from the plan for the scope
    (`lp.required_accepted_keys`): DONE fires only when EVERY required key has an accepted attempt; the calls ledger
    contributes attempts, never coverage (four call records for one key are one key). Every rule is stated once here."""
    events: List[str] = []
    covered: set = set()
    terminal_all: Dict[str, List[bool]] = {}
    if calls and not state.wave_launched:                                    # the wave is launched -- once, with the utilisation bars
        state.wave_launched = True
        events.append(summary_line("WAVE_LAUNCHED", {"leg": leg, "scope": scope, "n_calls": len(calls), "required_keys": len(required_keys),
                                                     "W7": (wave or {}).get("W7"), "wave_uuid": (wave or {}).get("wave_uuid"), "bars": dict(bars or {})}))
    for sid_, h in sorted((holds or {}).items()):                             # a HELD standup, once, with its reason
        if h.get("held") and sid_ not in state.reported_hold:
            state.reported_hold.add(sid_)
            events.append(summary_line("HOLD", {"standup_id": sid_, "reason": h.get("hold_reason"), "reattempts": h.get("reattempts"), "held_at": h.get("held_at"),
                                                "leg": leg, "W7": (wave or {}).get("W7"), "wave_uuid": (wave or {}).get("wave_uuid")}))
    for c in calls:
        cid, sid = c["call_id"], c["standup_id"]
        s = snapshot.get(cid) or {}
        hb = s.get("heartbeat")
        if hb and not state.started.get(cid):
            state.started[cid] = True
            # the rows-flat clock starts at STARTED -- a run that never writes a row is flat from its first heartbeat
            state.rows_moved_t.setdefault(cid, now)
            state.last_rows.setdefault(cid, int(hb.get("rows_written") or 0))
            events.append(f"STARTED {sid} attempt={hb.get('attempt_id')} call={cid}")
        if hb:
            t = _parse_utc(hb.get("t")) or now
            state.last_hb_t[cid] = t
            rows = int(hb.get("rows_written") or 0)
            if rows != state.last_rows.get(cid):
                state.last_rows[cid] = rows
                state.rows_moved_t[cid] = now
                state.first_rows_t.setdefault(cid, now)
        # the producer diagnosis is per ACCEPTED KEY (the six axes incl. the shard) -- an accepted sibling shard used to
        # suppress PRODUCER_MISSING for a shard whose every attempt is terminal and none accepted
        akey = c.get("accepted_key") or c["control_key"]
        terminal_all.setdefault(akey, []).append(bool(s.get("done")))
        if s.get("accepted") and c.get("accepted_key") in required_keys:
            covered.add(c["accepted_key"])
        if s.get("done") and cid not in state.reported_terminal:
            state.reported_terminal.add(cid)
            err, summ, res = s.get("error"), s.get("summary") or {}, s.get("result_line") or ""
            failed = bool(err) or (summ and summ.get("status") not in (None, "DONE")) or ("rc=1" in res)
            if failed:
                why = err or summ.get("status") or res
                tail = (s.get("log_tail") or "").strip()
                events.append(f"FAILED {sid} call={cid} {why}" + (f"\n--- log tail ---\n{tail}" if tail else ""))
        if not s.get("done") and state.started.get(cid):
            idle_min = (now - state.last_hb_t.get(cid, now)) / 60
            if idle_min >= stall_min:
                events.append(f"STALL {sid} call={cid} no heartbeat for {idle_min:.0f} min")
            elif leg in ROWS_FLAT_LEGS and cid in state.rows_moved_t and (now - state.rows_moved_t[cid]) / 60 >= rows_flat_min:
                events.append(f"STALL {sid} call={cid} rows flat for {(now - state.rows_moved_t[cid]) / 60:.0f} min (rows={state.last_rows.get(cid)})")
            elif hb and now - state.last_progress_t.get(cid, 0) >= progress_min * 60:
                state.last_progress_t[cid] = now
                rows = state.last_rows.get(cid, 0)
                span = max(1.0, (now - state.first_rows_t.get(cid, now)) / 60)
                events.append(f"PROGRESS {sid} rows={rows} rows/min={rows / span:.1f} running={hb.get('running')} phase={hb.get('phase')}")
    for akey, dones in terminal_all.items():
        mine = [c for c in calls if (c.get("accepted_key") or c["control_key"]) == akey]
        if dones and all(dones) and not any((snapshot.get(c["call_id"]) or {}).get("accepted") for c in mine):
            c0 = mine[0]
            events.append(f"PRODUCER_MISSING {c0['leg']}/{c0['engine']}/{c0['stage']}/shard{c0['shard']} standup={c0['standup_id']} key={akey} "
                          "every attempt terminal, none accepted — re-spawn the row")
    missing = sorted(set(required_keys) - covered)
    if required_keys and not missing:
        events.append(summary_line("WAVE_DONE", {"leg": leg, "scope": scope, "required": len(required_keys), "accepted": len(covered),
                                                 "held": sorted(s_ for s_, h in (holds or {}).items() if h.get("held")),
                                                 "W7": (wave or {}).get("W7"), "wave_uuid": (wave or {}).get("wave_uuid")}))
        events.append(f"DONE {leg} scope={scope} required={len(required_keys)} accepted={len(covered)}")
    return events


class ModalBackend:
    """The live snapshot: call states (`FunctionCall.from_id(...).get(timeout=0)`), the heartbeats / accepted Dicts, the
    attempt roots' RESULT + driver log tails via the Volume API. Every read is best-effort and stamped in the snapshot."""

    def __init__(self, plan: Mapping[str, Any]) -> None:
        import modal
        self.modal = modal
        self.plan = plan
        self.results_root = plan["results_root"]
        self.hb = modal.Dict.from_name(plan["dicts"]["heartbeats"], create_if_missing=True)
        self.acc = modal.Dict.from_name(plan["dicts"]["accepted"], create_if_missing=True)
        self.reatt = modal.Dict.from_name(plan["dicts"]["reattempts"], create_if_missing=True) if plan["dicts"].get("reattempts") else None
        self.vol = modal.Volume.from_name(lp.MODAL_VOLUME_NAME, create_if_missing=False)

    def holds(self, calls: List[dict]) -> Dict[str, Dict[str, Any]]:
        """`{standup_id: ledger entry}` of the calls' standups that the durable `reattempts` Dict marks HELD."""
        out: Dict[str, Dict[str, Any]] = {}
        if self.reatt is None:
            return out
        for sid in sorted({c["standup_id"] for c in calls}):
            try:
                raw = self.reatt.get(sid)
            except Exception:  # noqa: BLE001
                raw = None
            if raw is None:
                continue
            entry = json.loads(raw) if isinstance(raw, str) else dict(raw)
            if entry.get("held"):
                out[sid] = entry
        return out

    def read_text(self, path: str) -> Optional[str]:
        """A file under the Volume mount path (`/vol/...`) through the Volume API (the watcher runs on the grading host, where
        `/vol` is not a directory) -- None when absent."""
        return volume_read_text(self.vol, path)

    def call_state(self, cid: str):
        from modal.exception import OutputExpiredError
        fc = self.modal.FunctionCall.from_id(cid)
        try:
            return True, fc.get(timeout=0), None
        except OutputExpiredError as e:
            return True, None, f"{OUTPUT_EXPIRED}: {e}"
        except TimeoutError:
            return False, None, None
        except Exception as e:  # noqa: BLE001 -- the remote exception IS the failure
            return True, None, f"{type(e).__name__}: {e}"

    def log_tail(self, attempt_root: Optional[str], n: int = 40) -> str:
        if not attempt_root:
            return ""
        try:
            rel = attempt_root[len("/vol"):] + "/children"
            names = [e.path for e in self.vol.listdir(rel)]
            tails = []
            for name in sorted(names)[-3:]:
                data = b"".join(self.vol.read_file(name))
                tails.append(f"== {name}\n" + "\n".join(data.decode(errors='replace').splitlines()[-n:]))
            return "\n".join(tails)
        except Exception as e:  # noqa: BLE001
            return f"(log tail unavailable: {type(e).__name__}: {e})"

    def snapshot(self, calls: List[dict]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for c in calls:
            cid = c["call_id"]
            done, summary, err = self.call_state(cid)
            rec: Dict[str, Any] = {"done": done, "summary": summary, "error": err}
            try:
                raw = self.hb.get(c["control_key"])
                rec["heartbeat"] = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:  # noqa: BLE001
                rec["heartbeat"] = None
            if err and str(err).startswith(OUTPUT_EXPIRED):
                # the call's output is gone from Modal; the attempt's durable RESULT / COMPLETE.json on the volume is the authority
                rec.update(resolve_expired_call(self.read_text, self.results_root, c, rec["heartbeat"]))
            try:
                rec["accepted"] = self.acc.get(c.get("accepted_key") or c["control_key"])
            except Exception:  # noqa: BLE001
                rec["accepted"] = None
            if done and (err or (summary or {}).get("status") not in (None, "DONE")):
                rec["log_tail"] = self.log_tail((summary or {}).get("attempt_root"))
            out[cid] = rec
        return out


OUTPUT_EXPIRED = "OutputExpiredError"
RESULT_LINE_RE = re.compile(r"status=(?P<status>\S+)\s+rc=(?P<rc>\d+)")


def volume_read_text(vol: Any, path: str) -> Optional[str]:
    """`/vol/<rel>` through the Modal Volume API (`read_file` over the mount-relative path); None when the file is absent."""
    rel = path[len(lp.VOL_PATH) + 1:] if path.startswith(lp.VOL_PATH + "/") else path.lstrip("/")
    try:
        return b"".join(vol.read_file(rel)).decode(errors="replace")
    except FileNotFoundError:
        return None
    except Exception as e:  # noqa: BLE001
        name = type(e).__name__
        if "NotFound" in name or "not found" in str(e).lower():
            return None
        raise


EVIDENCE_NAME = "modal_evidence.json"                          # seedrep_served.EVIDENCE_NAME (the watcher cannot import the app: it imports modal)


def post_result_of(evidence_text: Optional[str]) -> Dict[str, Any]:
    """`modal_evidence.json["post_result"]` as a dict ({} when absent / unreadable / not a mapping)."""
    if not evidence_text:
        return {}
    try:
        doc = json.loads(evidence_text)
    except ValueError:
        return {}
    post = doc.get("post_result") if isinstance(doc, Mapping) else None
    return dict(post) if isinstance(post, Mapping) else {}


def resolve_expired_call(read_text: Callable[[str], Optional[str]], results_root: str, call: Mapping[str, Any],
                         heartbeat: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """A `FunctionCall` whose output EXPIRED is resolved from the attempt's DURABLE files -- `RESULT` (`... status=<S>
    rc=<n> ...`) or, failing that, `COMPLETE.json` -- under the attempt root the last heartbeat names (`attempt_id`), never reported as a
    failure for want of Modal's copy. Returns the snapshot fields to update: `summary`, `error`, `result_line`, `resolved_from`."""
    aid = (heartbeat or {}).get("attempt_id")
    if not aid:
        return {"error": f"{OUTPUT_EXPIRED}: no heartbeat names the attempt id, so no durable RESULT can be read", "resolved_from": None}
    root = lp.attempt_root_for(results_root, leg=call["leg"], view=call["engine"], stage=call["stage"], shard=call["shard"], attempt_id=aid,
                               kind=call.get("kind"))
    result = read_text(f"{root}/RESULT")
    if result:
        m = RESULT_LINE_RE.search(result)
        if m:
            status, rc = m.group("status"), int(m.group("rc"))
            out = {"summary": {"status": status, "attempt_root": root, "attempt_id": aid}, "result_line": result.strip(),
                   "error": None if rc == 0 else f"RESULT rc={rc} status={status}", "resolved_from": f"{root}/RESULT"}
            if rc == 0:
                # fo3: a RESULT DONE whose post-RESULT tail timed out BEFORE the acceptance carries its terminal status in
                # the durable evidence (`modal_evidence.json["post_result"]["terminal_status"]`), never in RESULT -- without this read the
                # expired path would report the un-accepted attempt as DONE for as long as the queue stays open
                post = post_result_of(read_text(f"{root}/{EVIDENCE_NAME}"))
                if post.get("terminal_status"):
                    out["summary"] = {**out["summary"], "status": str(post["terminal_status"]), "post_result": post}
                    out["error"] = f"{post['terminal_status']}: the post-RESULT tail did not return (phase {post.get('phase')}, {post.get('elapsed_s')} s)"
                    out["resolved_from"] = f"{root}/{EVIDENCE_NAME}"
            return out
        return {"error": f"{OUTPUT_EXPIRED}: RESULT under {root} is unparseable: {result.strip()[:120]!r}", "resolved_from": f"{root}/RESULT"}
    complete = read_text(f"{root}/COMPLETE.json")
    if complete:
        return {"summary": {"status": "DONE", "attempt_root": root, "attempt_id": aid}, "error": None, "resolved_from": f"{root}/COMPLETE.json"}
    return {"error": f"{OUTPUT_EXPIRED}: no durable RESULT / COMPLETE.json under {root} -- the attempt left no verdict", "resolved_from": None}


def _volume_json_reader() -> Callable[[str], Optional[dict]]:
    """The default pool-record reader: `/vol/attacker-pool/.../endpoint.json` through the Volume API (a LOCAL open of
    `/vol/...` on the grading host read nothing)."""
    import modal
    vol = modal.Volume.from_name(lp.MODAL_VOLUME_NAME, create_if_missing=False)

    def read(path: str) -> Optional[dict]:
        text = volume_read_text(vol, path)
        return json.loads(text) if text else None
    return read


def watch_leg(rs: Path, plan: Mapping[str, Any], leg: str, *, poll_s: int, stall_min: float, rows_flat_min: float,
              progress_min: float, max_hours: float, backend=None, clock=time.time, sleep=time.sleep, scope: str = "base") -> int:
    state = WatchState()
    required = lp.required_accepted_keys(plan, leg, scope)         # the plan's distinct rows of this scope
    if not required:
        raise SystemExit(f"[watch] the plan has no {scope} rows for leg {leg!r}")
    t0 = clock()
    backend = backend or ModalBackend(plan)
    while True:
        calls = load_calls(rs, leg, plan["wave_uuid"])
        if not calls:
            print(f"[watch] no calls for leg {leg} in {rs / 'calls.jsonl'} (wave {plan['wave_uuid'][:8]})", flush=True)
        snap = backend.snapshot(calls)
        holds = backend.holds(calls) if hasattr(backend, "holds") else {}
        for ev in classify(now=clock(), leg=leg, calls=calls, snapshot=snap, state=state, stall_min=stall_min,
                           rows_flat_min=rows_flat_min, progress_min=progress_min, required_keys=required, scope=scope,
                           holds=holds, bars=s5_bars_of(rs), wave={"W7": plan.get("W7"), "wave_uuid": plan.get("wave_uuid")}):
            print(ev, flush=True)
            if ev.startswith("DONE "):
                return 0
        if (clock() - t0) / 3600 >= max_hours:
            print("WATCH-TIMEOUT", flush=True)
            return 3
        sleep(poll_s)


def watch_pools(plan: Mapping[str, Any], *, poll_s: int, max_hours: float, clock=time.time, sleep=time.sleep, probe=None,
                read_record: Optional[Callable[[str], Optional[dict]]] = None) -> int:
    """Every pool's `/health` (bearer-guarded servers answer 401/200 — either is UP; a connection error is DOWN). A pool with no `url`
    in the plan is resolved from its `endpoint_record` on the VOLUME (`read_record`, the Volume API by default)."""
    state = WatchState()
    t0 = clock()
    probe = probe or lp.pool_is_healthy                       # the ONE health rule (lp.POOL_HEALTH_RULE), shared with the entrypoint
    read_record = read_record or _volume_json_reader()
    while True:
        for p in plan["pools"]:
            url = p.get("url")
            if not url:
                try:
                    url = (read_record(p["endpoint_record"]) or {}).get("url")
                except Exception as e:  # noqa: BLE001
                    print(f"POOL_RECORD_UNREADABLE {p['name']} {p['endpoint_record']}: {type(e).__name__}: {e}", flush=True)
                    url = None
            up = bool(url) and probe(url)
            was = state.pool_up.get(p["name"])
            if was is None or up != was:
                print(f"{'POOL_UP' if up else 'POOL_DOWN'} {p['name']} url={url}", flush=True)
            state.pool_up[p["name"]] = up
        if (clock() - t0) / 3600 >= max_hours:
            print("WATCH-TIMEOUT", flush=True)
            return 3
        sleep(poll_s)


#: The drive exit codes (`drive_rc`): 0 released AND published bound to the live proxy attempt (the consumer accepts); 1 legs
#: FAILED (held, or torn down by --teardown-after-failure); 2 released but UNPUBLISHABLE (the tunnel record under the key was missing,
#: unreadable or another attempt's) -- nothing was published, so the proxy sees NOTHING and would run to its deadline: a rc 2 is a
#: distinct failure the grading host watcher names, never a timeout
DRIVE_RC_EVENT = {0: "DRIVE_DONE", 1: "DRIVE_HELD", 2: "DRIVE_UNPUBLISHED"}


def drive_rc_of(record: Mapping[str, Any]) -> int:
    """The drive's own `drive_rc(verdict, completion)` rule over a finished `drive_<id>.json` (pinned equal to it by test)."""
    verdict, completion = record.get("verdict") or {}, record.get("completion") or {}
    if not verdict.get("release"):
        return 1
    return 0 if completion.get("binding") == "bound" and completion.get("published") else 2


def finished_drive_record(state_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    """The newest `drive_<id>.json` under the drive's attempt state dir that carries `finished_at` (the drive writes the record at start and
    updates it at the end); None while the drive runs or when no dir was given."""
    if state_dir is None or not Path(state_dir).is_dir():
        return None
    recs = []
    for p in sorted(Path(state_dir).glob("drive_*.json")):
        try:
            d = json.loads(p.read_text())
        except ValueError:
            continue
        if d.get("finished_at"):
            recs.append((str(d.get("started_at") or ""), p.name, d))
    return max(recs)[2] if recs else None


def watch_petri_drive(heartbeat: Path, done_marker: Path, *, poll_s: int, stall_min: float, max_hours: float,
                      clock=time.time, sleep=time.sleep, drive_state_dir: Optional[Path] = None) -> int:
    """The grading host drive's watch: the drive's finished record decides -- DRIVE_DONE (rc 0), DRIVE_HELD (rc 1: the proxy stays up for the
    re-drive), DRIVE_UNPUBLISHED (rc 2: released but unpublishable -- the proxy saw nothing and will time out; re-drive against the LIVE
    attempt with --resume-from); the legacy DONE marker still ends the watch; a stale heartbeat is DRIVE_STALL; the bound is WATCH-TIMEOUT."""
    t0 = clock()
    while True:
        rec = finished_drive_record(drive_state_dir)
        if rec is not None:
            rc = drive_rc_of(rec)
            comp = rec.get("completion") or {}
            counts = (rec.get("verdict") or {}).get("counts") or {}
            print(f"{DRIVE_RC_EVENT[rc]} rc={rc} drive={rec.get('drive_id')} attempt={rec.get('attempt_id')} counts={counts} binding={comp.get('binding')} "
                  f"published={comp.get('published')}" + ("" if rc != 2 else " -- the proxy received NOTHING and will run to its deadline; re-drive the live attempt with --resume-from"),
                  flush=True)
            return rc
        if done_marker.is_file():
            print(f"DRIVE_DONE {done_marker}", flush=True)
            return 0
        if heartbeat.is_file():
            idle = (clock() - heartbeat.stat().st_mtime) / 60
            if idle >= stall_min:
                print(f"DRIVE_STALL {heartbeat} idle {idle:.0f} min", flush=True)
        if (clock() - t0) / 3600 >= max_hours:
            print("WATCH-TIMEOUT", flush=True)
            return 3
        sleep(poll_s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rs", type=Path, default=RS_DEFAULT)
    ap.add_argument("--plan", type=Path, default=None, help="RS/launch_plan.json (default <rs>/launch_plan.json)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--leg", default=None)
    g.add_argument("--pools", action="store_true")
    g.add_argument("--petri-drive", type=Path, default=None, metavar="HEARTBEAT_FILE")
    ap.add_argument("--done-marker", type=Path, default=None, help="(--petri-drive) the drive's DONE file")
    ap.add_argument("--drive-state-dir", type=Path, default=None,
                    help="(--petri-drive) WS2-F's attempt state dir (~/.local/state/seedrep_battery/petri/<W7>/<wave8>/<view>/attempts/<attempt>): its "
                         "finished drive_<id>.json decides DRIVE_DONE (rc 0) / DRIVE_HELD (rc 1) / DRIVE_UNPUBLISHED (rc 2)")
    ap.add_argument("--scope", choices=lp.WATCH_SCOPES, default="base",
                    help="(--leg) the rows whose ACCEPTED coverage means DONE: base shards | declared rungs | all")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--stall-min", type=float, default=20)
    ap.add_argument("--rows-flat-min", type=float, default=40)
    ap.add_argument("--progress-min", type=float, default=30)
    ap.add_argument("--max-hours", type=float, default=24)
    args = ap.parse_args(argv)
    if args.poll < 30 or args.poll > 300:
        raise SystemExit("--poll must be within [30, 300] s (the poll is the upper bound on how long a dead job sits unnoticed)")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from sisteer import site as _site
    args.rs = _site.expand_path(args.rs, key="--rs")   # an unset $STATE refuses by name
    plan_path = args.plan or (args.rs / "launch_plan.json")
    plan = json.loads(plan_path.read_text())
    if args.leg:
        return watch_leg(args.rs, plan, args.leg, poll_s=args.poll, stall_min=args.stall_min, rows_flat_min=args.rows_flat_min,
                         progress_min=args.progress_min, max_hours=args.max_hours, scope=args.scope)
    if args.pools:
        return watch_pools(plan, poll_s=args.poll, max_hours=args.max_hours)
    return watch_petri_drive(args.petri_drive, args.done_marker or args.petri_drive.with_name("DONE"), poll_s=args.poll,
                             stall_min=args.stall_min, max_hours=args.max_hours, drive_state_dir=args.drive_state_dir)


if __name__ == "__main__":
    sys.exit(main())
