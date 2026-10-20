"""scripts/modal/seedrep_petri_watch.py -- the seedrep battery's Petri stall / refusal monitor.

Every event is pinned against a SYNTHETIC drive (invented sample ids, placeholder run keys, a stdlib-built
``.eval`` zip): the per-leg progress snapshot read from the Inspect eval logs + the drive record; a STALL by file
MTIME (never the leg log's own stamps); the deterministic refusal loop (``APIStatusError 200`` twice on the same
conversation from the same model) that proved to be the real stall; the PT -> UTC conversion of the
leg log's ``[MM/DD/YY HH:MM:SS]`` stamps (reading them as UTC misreads staleness); the ``width=1`` refusal by name;
a leg that DIED (``rc=3 results=absent``) vs one that finished; a flat layout accepted beside ours."""
from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "scripts/modal/seedrep_petri_watch.py"


def _load():
    spec = importlib.util.spec_from_file_location("seedrep_petri_watch", SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod        # dataclasses resolve `from __future__ import annotations` through sys.modules
    spec.loader.exec_module(mod)
    return mod


w = _load()

RK_A = "1111222233334444"
RK_B = "5555666677778888"
def _old() -> float:
    """Three hours ago, taken NOW: past any stall window. A per-test anchor, never a module-level constant: under a
    sharded gate a file can be collected long before its shard runs, and an import-time anchor then reads that much
    older than the test assumed (a ``for 1`` prefix matched ``for 180m`` and missed ``for 205m``)."""
    return time.time() - 3 * 3600


def _for_minutes(line: str, phrase: str) -> int:
    """The minute count the watcher renders after ``phrase`` (``... for 180m ...``): parsed, never prefix-matched."""
    m = re.search(re.escape(phrase) + r" for (\d+)m\b", line)
    assert m, (phrase, line)
    return int(m.group(1))


def _record(stamp: str, sample: str, model: str, retry: int, *, status: int = 200, category: str = "reasoning_extraction",
            secs: str = "1,537", body: str = "classifier") -> str:
    """One inspect WARNING record in Rich's 80-col wrapped shape (the real logs' geometry, synthetic content). ``body``
    picks the adjacent exception record: ``classifier`` (the safety-classifier text), ``overload`` (the streamed
    "Overloaded" body inspect_ai retries as transient), or ``none`` (no exception text at all)."""
    head = (f"{stamp} WARNING  sample={sample:<28} _model.py:2342\n"
            f"                             [11111111222222223333 petri_bloom/bloom_audit/\n"
            f"                             {sample}/1 none/none] ->\n"
            f"                             {model} retry {retry} (retrying in\n"
            f"                             {secs} seconds) [APIStatusError {status}]\n")
    if body == "classifier":
        return head + (f"{stamp} WARNING  the model declined this request with a safety     _model.py:2350\n"
                       f"                             classifier refusal (category: {category}).\n")
    if body == "overload":
        return head + (f"{stamp} WARNING  {{'type': 'error', 'error': {{'type':                _model.py:2350\n"
                       f"                             'overloaded_error', 'message': 'Overloaded'}}}}\n")
    return head


def _eval_zip(path: Path, sample_ids, *, mtime: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("_journal/start.json", json.dumps({"eval": {"task": "bloom_audit"}}))
        for i, sid in enumerate(sample_ids, 1):
            z.writestr(f"samples/{sid}_epoch_1.json", json.dumps({"id": sid, "epoch": 1}))
            z.writestr(f"_journal/summaries/{i}.json", json.dumps([{"id": sid}]))
    path.write_bytes(buf.getvalue())
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def _drive(tmp_path: Path, *, width: int = 3, legs=("leg_0_s1", "leg_1_s1")) -> tuple[Path, Path, Path]:
    """A synthetic seedrep drive: wave state dir + attempt dir + one out root; returns (state, attempt, out_root)."""
    state = tmp_path / "seedrep_battery" / "petri" / "abcdef0" / "11112222" / "s1"
    att = state / "attempts" / "ATTEMPT01"
    att.mkdir(parents=True)
    out = tmp_path / "petri_native" / "petri_seedrep_abcdef0_11112222_ATTEMPT01_qwen35_4b_s1"
    out.mkdir(parents=True)
    lines = [
        f"[2026-09-19T18:12:07Z] DRIVE_START seedrep view=s1 stage=base shard=0 drive_id=20260919T175730Z.1 W={'a' * 40} "
        f"W7=abcdef0 wave_uuid=11112222-3333-4444-5555-666677778888 state={att} width={width} auditor_org=highprio "
        f"target_decode=vendor seed_bases=[1] retry_sleep_cap_s=300",
        f"[2026-09-19T18:12:20Z] schedule: {len(legs)} runs ({len(legs)} cells x 1 seed bases); already complete: 0; to run: "
        + "[" + ", ".join(f"'{l}'" for l in legs) + "]",
    ]
    for l in legs:
        lines.append(f"[2026-09-19T18:12:24Z] launched {l} (cell none vmodel petri__base seed_base 1 width {width} out_root {out})")
    (state / "drive.log").write_text("\n".join(lines) + "\n")
    rks = {legs[0]: RK_A, legs[1]: RK_B} if len(legs) > 1 else {legs[0]: RK_A}
    (att / "drive_20260919T175730Z.1.json").write_text(json.dumps({
        "schema": "seedrep_petri_drive/1", "width": width, "out_roots": {"qwen35_4b_s1": str(out)},
        "schedule": [{"record_name": l, "run_key": rks[l], "tag": "qwen35_4b_s1", "cell_index": i, "seed_base": 1}
                     for i, l in enumerate(legs)],
    }))
    return state, att, out


def _leg_log(att: Path, leg: str, body: str, *, width: int = 3, mtime: float | None = None) -> Path:
    p = att / f"{leg}.log"
    p.write_text(f"== {leg} attempt 1 width={width} 2026-09-19T18:12:24Z ==\n"
                 f"[petri] two-SI registry config: configs/x.yaml\n" + body)
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def _ev(lines, name, key=None):
    """The event lines whose EVENT TOKEN (second field) is ``name`` -- never a substring match, which prose can
    fake (an ENGINE-UNOBSERVED line once said "stall" and read as a STALL event)."""
    out = []
    for l in lines:
        parts = l.split(" ", 3)
        if len(parts) >= 2 and parts[1] == name and (key is None or (len(parts) >= 3 and parts[2] == key)):
            out.append(l)          # a bare event (``<UTC> ALL-LEGS-TERMINAL``) has two fields
    return out


def _watch(dirs, **kw):
    kw.setdefault("stall_min", 15)
    kw.setdefault("loop_n", 2)
    kw.setdefault("expected_default", 60)
    kw.setdefault("judge_needles", ["opus"])
    kw.setdefault("auditor_needles", ["sonnet", "grok"])
    return w.Watch([Path(d) for d in dirs], **kw)   # ``now=`` / ``grace_min=`` pass through


# --- PT -> UTC: the leg log's stamps are LOCAL time (reading them as UTC misreads staleness) ----------------------

def test_leg_log_stamp_is_pt_and_converts_to_utc():
    """``[09/19/26 11:06:31]`` in a leg log is 18:06:31Z (PDT = UTC-7), the value ReLU first misread as 7 h stale.
    A January stamp converts under PST (UTC-8); an unparseable stamp is None, never a silently wrong time."""
    assert w.leg_stamp_to_utc("[09/19/26 11:06:31]") == "2026-09-19T18:06:31Z"
    assert w.leg_stamp_to_utc("[01/15/26 11:06:31]") == "2026-01-15T19:06:31Z"
    assert w.leg_stamp_to_utc("2026-09-19T18:06:31Z") is None
    assert w.leg_stamp_to_utc("[13/45/26 11:06:31]") is None


def test_rich_wrapped_records_are_joined_and_parsed():
    recs = w.parse_inspect_records(_record("[09/19/26 04:52:26]", "canary_case", "claude-opus-5", 10))
    assert len(recs) == 2
    assert recs[0].stamp == "[09/19/26 04:52:26]"
    assert "claude-opus-5 retry 10" in recs[0].text and "[APIStatusError 200]" in recs[0].text
    refs = w.refusal_records(recs)
    assert len(refs) == 1
    r = refs[0]
    assert (r.sample, r.model, r.retry, r.status, r.classifier) == ("canary_case", "claude-opus-5", 10, 200, True)
    assert r.category == "reasoning_extraction"      # filled from the adjacent exception record


def test_a_429_retry_is_not_a_classifier_refusal():
    body = ("[09/19/26 04:52:26] WARNING  sample=rate_case    _model.py:2342\n"
            "                             -> claude-sonnet-5 retry 2 (retrying in 4 seconds) [APIStatusError 429]\n")
    refs = w.refusal_records(w.parse_inspect_records(body))
    assert refs[0].status == 429 and refs[0].classifier is False and refs[0].kind == "unknown"
    assert w.refusal_loops(refs, loop_n=1) == {}


# --- the snapshot: per-leg samples from the Inspect eval logs via the drive record --------------------------------

def test_snapshot_reads_progress_rc_and_sidecars(tmp_path):
    state, att, out = _drive(tmp_path)
    _eval_zip(out / "inspect_logs" / RK_A / "target-seed-1" / "a.eval", [f"s{i}" for i in range(7)])
    _eval_zip(out / "inspect_logs" / RK_A / "target-seed-1" / "b.eval", [f"t{i}" for i in range(3)])
    (out / "cells" / RK_B).mkdir(parents=True)
    (out / "cells" / RK_B / "results.json").write_text(json.dumps({"audited": {"expected": 60, "items": 58}}))
    (out / "cells" / RK_B / "petri_judge_refused.json").write_text(json.dumps({"items": [{"scenario_id": "x", "epoch": 1}, {"scenario_id": "y", "epoch": 1}]}))
    (out / "cells" / RK_B / "petri_auditor_refused.json").write_text(json.dumps({"items": [{"scenario_id": "z", "epoch": 1}]}))
    _eval_zip(out / "inspect_logs" / RK_B / "target-seed-1" / "c.eval", [f"u{i}" for i in range(58)])
    _leg_log(att, "leg_0_s1", "[09/19/26 11:06:31] INFO     sample=s6 finished    logger.py:1\n")
    _leg_log(att, "leg_1_s1", "")
    (att / "leg_1_s1.rc").write_text("0 drive=20260919T175730Z.1 results=" + "f" * 64 + "\n")
    watch = _watch([state])
    lines, live = watch.poll(snapshot=True)
    snap = {l.split(" ", 3)[2]: l for l in _ev(lines, "LEG")}
    assert "samples 10/60" in snap["s1/leg_0_s1"] and "rc=running" in snap["s1/leg_0_s1"]
    assert "last_log_stamp=[09/19/26 11:06:31](local)=2026-09-19T18:06:31Z" in snap["s1/leg_0_s1"]
    assert "samples 58/60" in snap["s1/leg_1_s1"] and "refusals auditor=1 judge=2" in snap["s1/leg_1_s1"] and "rc=0" in snap["s1/leg_1_s1"]
    assert "rc=0 results=ffffffffffff" in _ev(lines, "LEG-DONE", "s1/leg_1_s1")[0]
    assert live is True and watch.bad is False


def test_eval_zip_count_needs_no_decompression_and_tolerates_a_torn_zip(tmp_path):
    p = tmp_path / "x.eval"
    _eval_zip(p, ["a", "b", "c"])
    assert w.eval_zip_sample_count(p)[0] == 3
    torn = tmp_path / "torn.eval"
    torn.write_bytes(p.read_bytes()[:40])
    assert w.eval_zip_sample_count(torn)[0] is None


# --- STALL by mtime, never by the log's own (local-time) stamps -------------------------------------------------

def test_stall_fires_on_the_first_observation_from_mtimes_and_clears_when_the_log_is_written(tmp_path):
    """Staleness is judged from the authoritative file MTIMES on EVERY observation, the first
    included -- a fresh watcher (``--once``) polls exactly once, so a "no growth since the last poll" rule could never
    fire and a three-hour-old active leg read healthy forever. The stamp in the log READS recent-ish; the file has
    not been touched for 3 h: the stall is by mtime."""
    state, att, out = _drive(tmp_path, legs=("leg_0_s1", "leg_1_s1"))
    old = _old()
    log = _leg_log(att, "leg_0_s1", "[09/19/26 11:06:31] INFO     working    logger.py:1\n", mtime=old)
    _eval_zip(out / "inspect_logs" / RK_A / "target-seed-1" / "a.eval", ["a"], mtime=old)
    _leg_log(att, "leg_1_s1", "")            # fresh: never stalls
    watch = _watch([state])
    first, _ = watch.poll(snapshot=False)
    stalls = _ev(first, "STALL", "s1/leg_0_s1")
    assert len(stalls) == 1 and "tail:" in stalls[0]
    assert 180 <= _for_minutes(stalls[0], "no log write and no new sample") < 240     # the 3 h mtime, in minutes
    assert not _ev(first, "STALL", "s1/leg_1_s1")
    assert watch.bad is True
    second, _ = watch.poll(snapshot=False)
    assert not _ev(second, "STALL"), "a stall is a transition, said once"
    with log.open("a") as f:
        f.write("[09/19/26 11:20:00] INFO     more    logger.py:1\n")
    third, _ = watch.poll(snapshot=False)
    assert _ev(third, "RESUMED", "s1/leg_0_s1")


def test_once_returns_3_for_a_stale_active_leg(tmp_path, capsys):
    """The CLI-level pin: repeated ``--once`` invocations against a stale active leg must NOT report
    healthy -- each fresh one-shot watcher classifies the stale leg from mtimes and exits 3."""
    state, att, out = _drive(tmp_path)
    old = _old()
    _leg_log(att, "leg_0_s1", "[09/19/26 08:00:00] INFO     working    logger.py:1\n", mtime=old)
    _leg_log(att, "leg_1_s1", "", mtime=old)
    for _ in range(2):
        assert w.main([str(state), "--once", "--stall-min", "15"]) == 3
        text = capsys.readouterr().out
        assert len(_ev(text.splitlines(), "STALL")) == 2


# --- a re-invoked drive: the predecessor's .rc is history, the latest DRIVE_START is the drive -----------------

def test_a_previous_drives_rc_is_history_not_a_terminal_record(tmp_path, capsys):
    """A per-leg re-invocation appends to the same leg log and leaves the
    predecessor's ``.rc`` (``3 drive=<old> results=absent``) until its own leg returns. Bound to the CURRENT
    drive_id, that record is LEG-PRIOR-RC (said once) and the leg stays live: no LEG-DIED, no ALL-LEGS-TERMINAL."""
    state, att, out = _drive(tmp_path)
    # the second DRIVE_START (a re-invocation of leg 0 at width 2) is appended AFTER the first
    with (state / "drive.log").open("a") as f:
        f.write("[2026-09-19T20:00:00Z] DRIVE_START seedrep view=s1 stage=base shard=0 drive_id=20260919T200000Z.2 W=" + "a" * 40
                + " W7=abcdef0 wave_uuid=11112222-3333-4444-5555-666677778888 width=2 retry_sleep_cap_s=300\n"
                "[2026-09-19T20:00:10Z] schedule: 2 runs (2 cells x 1 seed bases); already complete: 1; to run: ['leg_0_s1']\n")
    _leg_log(att, "leg_0_s1", "== leg_0_s1 attempt 1 rc=3 ==\n== leg_0_s1 attempt 1 width=2 2026-09-19T20:00:20Z ==\n[petri] resuming\n")
    (att / "leg_0_s1.rc").write_text("3 drive=20260919T175730Z.1 results=absent\n")          # the PREDECESSOR's record
    _leg_log(att, "leg_1_s1", "")
    (att / "leg_1_s1.rc").write_text("0 drive=20260919T175730Z.1 results=" + "e" * 64 + "\n")  # completed under the old drive
    watch = _watch([state])
    info = w.read_drive_info(state)
    assert info.drive_id == "20260919T200000Z.2" and info.width == 2 and info.to_run == ["leg_0_s1"]
    assert not watch.start_checks(), "the LATEST DRIVE_START is width 2; the first line's width is not this drive's"
    lines, live = watch.poll(snapshot=True)
    assert not _ev(lines, "LEG-DIED") and not _ev(lines, "LEG-DONE")
    prior = _ev(lines, "LEG-PRIOR-RC", "s1/leg_0_s1")
    assert len(prior) == 1 and "rc=3 drive=20260919T175730Z.1" in prior[0] and "current drive_id=20260919T200000Z.2" in prior[0]
    assert not _ev(lines, "LEG-PRIOR-RC", "s1/leg_1_s1"), "leg 1 is not this drive's work; its old .rc is not 'a live leg's history'"
    assert len(_ev(lines, "LEG-UNSCHEDULED", "s1/leg_1_s1")) == 1
    snap = _ev(lines, "LEG", "s1/leg_0_s1")[0]
    assert "rc=running prior_rc=3@20260919T175730Z.1" in snap
    assert live is True
    again, live2 = watch.poll(snapshot=False)
    assert not _ev(again, "LEG-PRIOR-RC") and live2 is True
    # the loop-mode CLI keeps watching: the second poll must not end in ALL-LEGS-TERMINAL
    (att / "leg_0_s1.rc").write_text("0 drive=20260919T200000Z.2 results=" + "d" * 64 + "\n")   # the NEW drive returns
    done, live3 = watch.poll(snapshot=False)
    assert "rc=0 results=dddddddddddd" in _ev(done, "LEG-DONE", "s1/leg_0_s1")[0]
    # ... and once the one re-driven leg has returned, NOTHING is live: leg 1 (completed under the old drive, not in
    # this drive's `to run`) is history, not work (fk2c -- the assertion the first version of this test lacked)
    assert live3 is False


def test_the_latest_drive_start_wins_for_width(tmp_path, capsys):
    """An appended drive.log's FIRST DRIVE_START said width=1 for a drive re-run at
    60. The latest line is the drive being watched."""
    state, att, out = _drive(tmp_path, width=1)
    with (state / "drive.log").open("a") as f:
        f.write("[2026-09-19T20:00:00Z] DRIVE_START seedrep view=s1 drive_id=20260919T200000Z.2 width=60 retry_sleep_cap_s=300\n")
    _leg_log(att, "leg_0_s1", "", width=60)
    _leg_log(att, "leg_1_s1", "", width=60)
    assert w.main([str(state), "--once"]) == 0
    assert "WIDTH-1" not in capsys.readouterr().out


def test_no_leg_logs_yet_is_not_all_legs_terminal(tmp_path):
    """The startup grace: a drive still in its tunnel wait / bundle materialisation has written no leg log;
    that is not "every leg returned". Within the grace: live, silent. Past it: NO-LEGS-STARTED once, still live."""
    state, att, out = _drive(tmp_path)
    for p in att.glob("leg_*"):
        p.unlink()
    clock = [1_000_000.0]
    watch = _watch([state], now=lambda: clock[0], grace_min=60)
    lines, live = watch.poll(snapshot=False)
    assert live is True and not _ev(lines, "NO-LEGS-STARTED") and watch.bad is False
    clock[0] += 61 * 60
    lines, live = watch.poll(snapshot=False)
    assert live is True and len(_ev(lines, "NO-LEGS-STARTED")) == 1 and watch.bad is True
    lines, live = watch.poll(snapshot=False)
    assert live is True and not _ev(lines, "NO-LEGS-STARTED"), "said once"


# --- the deterministic refusal loop -------------------------------------------------------------------

def test_refusal_retry_and_loop_on_the_same_conversation_and_model(tmp_path):
    state, att, out = _drive(tmp_path)
    body = (_record("[09/19/26 04:52:26]", "canary_case", "claude-opus-5", 10)
            + _record("[09/19/26 05:18:13]", "canary_case", "claude-opus-5", 11)
            + _record("[09/19/26 05:30:00]", "other_case", "claude-sonnet-5", 1, status=529, body="none"))   # a bare 529: not a refusal
    _leg_log(att, "leg_0_s1", body)
    _leg_log(att, "leg_1_s1", "")
    watch = _watch([state])
    lines, _ = watch.poll(snapshot=True)
    retry = _ev(lines, "REFUSAL-RETRY", "s1/leg_0_s1")
    assert len(retry) == 1
    assert ("retry=11 model=claude-opus-5 role=judge sample=canary_case status=200 category=reasoning_extraction "
            "at=[09/19/26 05:18:13](local)=2026-09-19T12:18:13Z") in retry[0]
    assert "retries cannot succeed" in retry[0] and "elapsed_since_first=25m" in retry[0]   # 04:52:26 -> 05:18:13 on canary_case
    loops = _ev(lines, "REFUSAL-LOOP", "s1/leg_0_s1")
    assert len(loops) == 1 and "sample=canary_case model=claude-opus-5 n=2" in loops[0] and "tail:" in loops[0]
    snap = _ev(lines, "LEG", "s1/leg_0_s1")[0]
    assert "refusals auditor=0 judge=2 loops=1 retry_max=11" in snap
    assert watch.bad is True
    again, _ = watch.poll(snapshot=False)
    assert not _ev(again, "REFUSAL-RETRY") and not _ev(again, "REFUSAL-LOOP"), "transitions, said once"


def test_a_single_refusal_is_not_a_loop(tmp_path):
    state, att, out = _drive(tmp_path)
    _leg_log(att, "leg_0_s1", _record("[09/19/26 04:52:26]", "canary_case", "claude-opus-5", 1))
    _leg_log(att, "leg_1_s1", "")
    watch = _watch([state])
    lines, _ = watch.poll(snapshot=False)
    assert _ev(lines, "REFUSAL-RETRY") and not _ev(lines, "REFUSAL-LOOP")
    assert watch.bad is False


# --- width=1 is itself the bug; a leg can DIE ---------------------------------------------------------------------

def test_width_1_is_refused_by_name_with_rc_3(tmp_path, capsys):
    state, att, out = _drive(tmp_path, width=1)
    _leg_log(att, "leg_0_s1", "", width=1)
    _leg_log(att, "leg_1_s1", "", width=1)
    rc = w.main([str(state), "--once"])
    out_text = capsys.readouterr().out
    assert rc == 3
    assert "WIDTH-1 s1 -- DRIVE_START width=1" in out_text and "REFUSED by name" in out_text
    assert out_text.count("WIDTH-1") >= 1


def test_leg_died_vs_leg_done(tmp_path):
    state, att, out = _drive(tmp_path)
    _leg_log(att, "leg_0_s1", "[petri] cell none__lora0 INCOMPLETE: 3 unresolved ledger rows\n== leg_0_s1 attempt 1 rc=3 ==\n")
    (att / "leg_0_s1.rc").write_text("3 drive=20260919T175730Z.1 results=absent\n")
    _leg_log(att, "leg_1_s1", "")
    (att / "leg_1_s1.rc").write_text("0 drive=20260919T175730Z.1 results=" + "e" * 64 + "\n")
    watch = _watch([state])
    lines, live = watch.poll(snapshot=False)
    died = _ev(lines, "LEG-DIED", "s1/leg_0_s1")
    assert len(died) == 1 and "rc=3 results=absent" in died[0] and "INCOMPLETE: 3 unresolved" in died[0]
    assert "rc=0 results=eeeeeeeeeeee" in _ev(lines, "LEG-DONE", "s1/leg_1_s1")[0]
    assert live is False and watch.bad is True


def test_once_is_healthy_rc_0_and_a_missing_dir_is_rc_2(tmp_path, capsys):
    state, att, out = _drive(tmp_path)
    _leg_log(att, "leg_0_s1", "")
    _leg_log(att, "leg_1_s1", "")
    assert w.main([str(state), "--once"]) == 0
    text = capsys.readouterr().out
    assert len(_ev(text.splitlines(), "LEG")) == 2 and _ev(text.splitlines(), "ENGINE-UNOBSERVED", "s1")
    assert w.main([str(tmp_path / "nope"), "--once"]) == 2


def test_engine_unobserved_is_said_once_and_engine_quiet_reads_a_mirrored_vllm_log(tmp_path):
    state, att, out = _drive(tmp_path)
    _leg_log(att, "leg_0_s1", "")
    _leg_log(att, "leg_1_s1", "")
    watch = _watch([state])
    a, _ = watch.poll(snapshot=False)
    b, _ = watch.poll(snapshot=False)
    assert len(_ev(a, "ENGINE-UNOBSERVED")) == 1 and not _ev(b, "ENGINE-UNOBSERVED")
    (state / "serving").mkdir()
    v = state / "serving" / "vllm_server.log"
    v.write_text("INFO Running: 3 reqs, Waiting: 0 reqs\n")
    old = _old()
    os.utime(v, (old, old))
    c, _ = watch.poll(snapshot=False)
    assert 180 <= _for_minutes(_ev(c, "ENGINE-QUIET", "s1")[0], "no vLLM log line") < 240


# --- a flat layout beside ours ---------------------------------------------------------------------------------

def test_flat_layout_with_bare_leg_logs_is_accepted(tmp_path):
    d = tmp_path / "qwen_x_s3"
    d.mkdir()
    (d / "drive.log").write_text("[2026-09-19T17:57:30Z] DRIVE_START sdfbase view=qwen_x_s3 width=1 retry_sleep_cap_s=1800\n")
    (d / "leg_2.log").write_text("== leg 2 attempt 1 2026-09-19T11:01:31Z ==\n"
                                 + _record("[09/19/26 04:52:26]", "canary_case", "claude-opus-5", 10)
                                 + "== leg 2 attempt 1 rc=3 ==\n")
    (d / "leg_2.rc").write_text("3 drive=x results=absent\n")
    watch = _watch([d])
    start = watch.start_checks()
    assert len(start) == 1 and "WIDTH-1 qwen_x_s3" in start[0]
    lines, live = watch.poll(snapshot=True)
    assert "samples ?/60" in _ev(lines, "LEG", "qwen_x_s3/leg_2")[0]
    assert "rc=3 results=absent" in _ev(lines, "LEG-DIED", "qwen_x_s3/leg_2")[0]
    assert live is False


def test_help_names_the_relaunch_command_and_the_pt_rule():
    import subprocess
    out = subprocess.run([sys.executable, str(SRC), "--help"], capture_output=True, text=True, check=True).stdout
    assert "seedrep_petri_drive.py --sha <W 40hex> --wave-json RS/wave.json" in out and '--legs "<k>"' in out
    assert "pkill -f 'scripts/petri_run.py .* --cell-index <k>'" in out
    assert "LOCAL time (PT on the grading host)" in out and "APIStatusError 200" in out
    assert "watch-petri-stall.sh" in out, "credits the peer lane's tool it was started from"
    assert "LEG-PRIOR-RC" in out and "NO-LEGS-STARTED" in out and "the first included" in out


def test_stdlib_only():
    import ast
    tree = ast.parse(SRC.read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    assert mods <= {"argparse", "json", "os", "re", "sys", "time", "zipfile", "dataclasses", "datetime", "pathlib",
                    "typing", "zoneinfo", "__future__"}, mods



def _partial_redrive(tmp_path):
    """A two-leg drive where leg 1 was ALREADY COMPLETE (reconciled, skipped, its .rc
    keeping the previous drive id) and only leg 0 re-ran under the new drive id. Both old files are hours old."""
    state, att, out = _drive(tmp_path)
    new_id = "20260919T200000Z.2"
    with (state / "drive.log").open("a") as f:
        f.write(f"[2026-09-19T20:00:00Z] DRIVE_START seedrep view=s1 stage=base shard=0 drive_id={new_id} W=" + "a" * 40
                + f" W7=abcdef0 wave_uuid=11112222-3333-4444-5555-666677778888 state={att} width=3 retry_sleep_cap_s=300\n"
                "[2026-09-19T20:00:10Z] schedule: 2 runs (2 cells x 1 seed bases); already complete: 1; to run: ['leg_0_s1']\n")
    (att / f"drive_{new_id}.json").write_text(json.dumps({
        "schema": "seedrep_petri_drive/1", "drive_id": new_id, "width": 3, "out_roots": {"qwen35_4b_s1": str(out)},
        "already_complete": ["leg_1_s1"],
        "schedule": [{"record_name": "leg_0_s1", "run_key": RK_A, "tag": "qwen35_4b_s1", "cell_index": 0, "seed_base": 1},
                     {"record_name": "leg_1_s1", "run_key": RK_B, "tag": "qwen35_4b_s1", "cell_index": 1, "seed_base": 1}]}))
    old = _old()
    _leg_log(att, "leg_1_s1", "", mtime=old)                                       # completed under the OLD drive, untouched since
    (att / "leg_1_s1.rc").write_text("0 drive=20260919T175730Z.1 results=" + "e" * 64 + "\n")
    os.utime(att / "leg_1_s1.rc", (old, old))
    _leg_log(att, "leg_0_s1", "== leg_0_s1 attempt 1 rc=3 ==\n== leg_0_s1 attempt 1 width=3 2026-09-19T20:00:20Z ==\n[petri] resuming\n")
    return state, att, out, new_id


def test_partial_redrive_terminates_cleanly_once_the_rerun_leg_returns(tmp_path):
    """After the one re-driven leg completes under the new drive id, the watch is NOT live and the
    already-complete leg (old .rc, hours-old log) is never STALL, never LEG-DIED, never 'a live leg with history'."""
    state, att, out, new_id = _partial_redrive(tmp_path)
    watch = _watch([state])
    info = w.read_drive_info(state)
    assert info.drive_id == new_id and info.to_run == ["leg_0_s1"] and info.already_complete == ["leg_1_s1"]
    assert info.attempt_dir == att
    first, live = watch.poll(snapshot=True)
    assert live is True                                             # leg 0 is running
    assert len(_ev(first, "LEG-COMPLETE-PRIOR", "s1/leg_1_s1")) == 1
    assert "rc=0@20260919T175730Z.1" in _ev(first, "LEG-COMPLETE-PRIOR", "s1/leg_1_s1")[0]
    assert not _ev(first, "STALL") and not _ev(first, "LEG-DIED") and not _ev(first, "LEG-PRIOR-RC")
    assert "complete (already_complete of drive " in _ev(first, "LEG", "s1/leg_1_s1")[0]
    assert watch.bad is False
    (att / "leg_0_s1.rc").write_text(f"0 drive={new_id} results=" + "d" * 64 + "\n")
    second, live2 = watch.poll(snapshot=False)
    assert "rc=0 results=dddddddddddd" in _ev(second, "LEG-DONE", "s1/leg_0_s1")[0]
    assert live2 is False and watch.bad is False
    assert not _ev(second, "STALL") and not _ev(second, "LEG-COMPLETE-PRIOR"), "the complete leg is said once, and never stalls"


def test_partial_redrive_loop_mode_reaches_all_legs_terminal_and_once_is_healthy(tmp_path, capsys):
    """The CLI-level pin: loop mode exits 0 at ALL-LEGS-TERMINAL (before fk2c it could never get there), and a fresh
    --once against the finished partial re-drive is rc 0 with no STALL for the historical leg."""
    state, att, out, new_id = _partial_redrive(tmp_path)
    (att / "leg_0_s1.rc").write_text(f"0 drive={new_id} results=" + "d" * 64 + "\n")
    rc = w.main([str(state), "--poll-s", "0", "--stall-min", "15"])
    text = capsys.readouterr().out
    assert rc == 0 and _ev(text.splitlines(), "ALL-LEGS-TERMINAL") and not _ev(text.splitlines(), "STALL")
    assert w.main([str(state), "--once", "--stall-min", "15"]) == 0
    text = capsys.readouterr().out
    assert not _ev(text.splitlines(), "STALL") and _ev(text.splitlines(), "LEG-DONE", "s1/leg_0_s1")


def test_logs_under_an_older_attempt_are_not_this_drives_work(tmp_path):
    """The attempt binding: the latest DRIVE_START names its attempt state dir (``state=``); a leg log retained under
    an OLDER attempt dir -- hours old, no .rc -- is that attempt's history and never a STALL of this drive."""
    state, att, out = _drive(tmp_path)
    old_att = state / "attempts" / "ATTEMPT00"
    old_att.mkdir()
    _leg_log(old_att, "leg_0_s1", "[09/19/26 01:00:00] INFO     old attempt    logger.py:1\n", mtime=_old())
    _leg_log(att, "leg_0_s1", "")
    _leg_log(att, "leg_1_s1", "")
    watch = _watch([state])
    lines, live = watch.poll(snapshot=True)
    assert live is True and not _ev(lines, "STALL") and len(_ev(lines, "LEG")) == 2
    assert [Path(p).parent.name for p in (str(l) for l in w.find_leg_logs(state, att))] == ["ATTEMPT01", "ATTEMPT01"]
    # with no attempt named (a flat dir, or an old drive.log without state=), every attempt's logs are watched as before
    assert len(w.find_leg_logs(state, None)) == 3



# --- APIStatusError 200 is NOT a classifier refusal; the BODY TEXT decides -----------------------------------

def test_streamed_overload_is_not_a_refusal_loop(tmp_path, capsys):
    """An ``APIStatusError 200`` whose body says "Overloaded" is a streamed overload that
    inspect_ai retries as transient regardless of status (should_retry :1323-1328) and that CAN succeed. Three of them
    on one sample fire OVERLOAD-RETRY (informational), never REFUSAL-LOOP; --once is rc 0."""
    state, att, out = _drive(tmp_path)
    body = (_record("[09/19/26 04:52:26]", "busy_case", "claude-opus-5", 1, body="overload", secs="30")
            + _record("[09/19/26 04:53:00]", "busy_case", "claude-opus-5", 2, body="overload", secs="60")
            + _record("[09/19/26 05:10:00]", "busy_case", "claude-opus-5", 3, body="overload", secs="120"))
    _leg_log(att, "leg_0_s1", body)
    _leg_log(att, "leg_1_s1", "")
    recs = w.refusal_records(w.parse_inspect_records(body))
    assert [r.kind for r in recs] == ["overload"] * 3 and all(r.status == 200 and not r.classifier for r in recs)
    assert w.refusal_loops(recs, loop_n=2) == {}
    watch = _watch([state])
    lines, _ = watch.poll(snapshot=True)
    ov = _ev(lines, "OVERLOAD-RETRY", "s1/leg_0_s1")
    assert len(ov) == 1 and "retry=3 model=claude-opus-5 role=judge sample=busy_case status=200" in ov[0]
    assert "elapsed_since_first=17m" in ov[0] and "retries can succeed" in ov[0]
    assert not _ev(lines, "REFUSAL-LOOP") and not _ev(lines, "REFUSAL-RETRY") and watch.bad is False
    assert "loops=0 retry_max=0 overload_retry_max=3" in _ev(lines, "LEG", "s1/leg_0_s1")[0]
    assert w.main([str(state), "--once"]) == 0
    assert not _ev(capsys.readouterr().out.splitlines(), "REFUSAL-LOOP")


def test_the_same_count_with_the_classifier_text_is_a_refusal_loop(tmp_path, capsys):
    """The control: three ``APIStatusError 200`` records on one sample whose adjacent body carries the classifier text
    ARE a deterministic refusal loop -- REFUSAL-LOOP, --once rc 3."""
    state, att, out = _drive(tmp_path)
    body = "".join(_record(f"[09/19/26 05:0{i}:00]", "canary_case", "claude-opus-5", i + 1) for i in range(3))
    _leg_log(att, "leg_0_s1", body)
    _leg_log(att, "leg_1_s1", "")
    recs = w.refusal_records(w.parse_inspect_records(body))
    assert [r.kind for r in recs] == ["classifier"] * 3
    assert w.main([str(state), "--once"]) == 3
    text = capsys.readouterr().out.splitlines()
    assert len(_ev(text, "REFUSAL-LOOP", "s1/leg_0_s1")) == 1 and "n=3" in _ev(text, "REFUSAL-LOOP", "s1/leg_0_s1")[0]
    assert not _ev(text, "OVERLOAD-RETRY")


def test_a_retry_with_neither_body_is_unknown_and_never_a_loop(tmp_path):
    state, att, out = _drive(tmp_path)
    body = _record("[09/19/26 04:52:26]", "rate_case", "claude-sonnet-5", 2, status=429, body="none")
    _leg_log(att, "leg_0_s1", body)
    _leg_log(att, "leg_1_s1", "")
    recs = w.refusal_records(w.parse_inspect_records(body))
    assert recs[0].kind == "unknown" and recs[0].status == 429
    watch = _watch([state])
    lines, _ = watch.poll(snapshot=False)
    assert len(_ev(lines, "UNKNOWN-RETRY", "s1/leg_0_s1")) == 1 and not _ev(lines, "REFUSAL-LOOP") and watch.bad is False


def test_docstring_cites_should_retry_and_retracts_the_status_code_diagnosis():
    text = SRC.read_text()
    assert "should_retry" in text and ":1323-1328" in text and "REGARDLESS of status code" in text
    assert "a SAFETY-CLASSIFIER refusal returned with HTTP 200" not in text, "the retracted D33 diagnosis must be gone"



def _declined(stamp: str, sample: str, model: str, category: str = "reasoning_extraction") -> str:
    """The REAL shape of a classifier refusal in a leg log (logger.py:278): its own record, its own sample token."""
    return (f"{stamp} WARNING  sample={sample:<28} logger.py:278\n"
            f"                             {model} declined this request with a safety\n"
            f"                             classifier refusal (category: {category}).\n")


def _bare_retry(stamp: str, sample: str, model: str, retry: int, status: int = 200) -> str:
    """The REAL shape of a retry record (_model.py:2342): no exception body in the log at all."""
    return _record(stamp, sample, model, retry, status=status, body="none")


def test_real_shape_one_declined_refusal_is_a_refusal_not_a_loop_and_bare_retries_are_unknown(tmp_path, capsys):
    """The peer lane's live logs (2026-09-19): one ``declined ... safety classifier refusal`` record per leg, each on
    its own sample, and ``retry N [APIStatusError 200]`` records with NO body text. One refusal is a REFUSAL (the
    fallback chain's business), not a loop; the body-less retries are UNKNOWN-RETRY; --once is rc 0."""
    state, att, out = _drive(tmp_path)
    body = (_declined("[09/19/26 12:48:53]", "legal_case", "claude-opus-5")
            + _bare_retry("[09/19/26 13:15:42]", "tax_case", "claude-opus-5", 10)
            + _bare_retry("[09/19/26 13:21:22]", "brand_case", "claude-opus-5", 10))
    _leg_log(att, "leg_0_s1", body)
    _leg_log(att, "leg_1_s1", "")
    recs = w.refusal_records(w.parse_inspect_records(body))
    assert [(r.kind, r.sample, r.retry) for r in recs] == [("classifier", "legal_case", 0), ("unknown", "tax_case", 10), ("unknown", "brand_case", 10)]
    assert recs[0].model == "claude-opus-5" and recs[0].category == "reasoning_extraction"
    watch = _watch([state])
    lines, _ = watch.poll(snapshot=True)
    ref = _ev(lines, "REFUSAL", "s1/leg_0_s1")
    assert len(ref) == 1 and "sample=legal_case model=claude-opus-5 role=judge category=reasoning_extraction n=1" in ref[0]
    assert len(_ev(lines, "UNKNOWN-RETRY", "s1/leg_0_s1")) == 1 and "retry=10" in _ev(lines, "UNKNOWN-RETRY", "s1/leg_0_s1")[0]
    assert not _ev(lines, "REFUSAL-LOOP") and not _ev(lines, "REFUSAL-RETRY") and not _ev(lines, "OVERLOAD-RETRY")
    assert "refusals auditor=0 judge=1 loops=0 retry_max=0 overload_retry_max=0" in _ev(lines, "LEG", "s1/leg_0_s1")[0]
    assert watch.bad is False
    again, _ = watch.poll(snapshot=False)
    assert not _ev(again, "REFUSAL"), "said once"
    assert w.main([str(state), "--once"]) == 0


def test_real_shape_the_same_conversation_declined_twice_by_the_same_model_is_the_loop(tmp_path, capsys):
    """The trigger on the real shape: two ``declined`` records for the same sample from the same model."""
    state, att, out = _drive(tmp_path)
    body = (_declined("[09/19/26 12:48:53]", "legal_case", "claude-opus-5")
            + _declined("[09/19/26 12:58:00]", "other_case", "claude-opus-5", "general_harms")
            + _declined("[09/19/26 13:10:00]", "legal_case", "claude-opus-5"))
    _leg_log(att, "leg_0_s1", body)
    _leg_log(att, "leg_1_s1", "")
    assert w.main([str(state), "--once"]) == 3
    text = capsys.readouterr().out.splitlines()
    loops = _ev(text, "REFUSAL-LOOP", "s1/leg_0_s1")
    assert len(loops) == 1 and "sample=legal_case model=claude-opus-5 n=2" in loops[0]
    assert len(_ev(text, "REFUSAL", "s1/leg_0_s1")) == 1 and "sample=other_case" in _ev(text, "REFUSAL", "s1/leg_0_s1")[0]
