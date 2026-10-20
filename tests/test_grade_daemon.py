"""The grading daemon (`scripts/registry/grade_daemon.py`, behind `reproduce/08_grade.sh --daemon`):
fake pending rows in a synthetic registry → one dispatch per (family, model key) with `--run-id` for exactly the selected runs and the
store map, through a FAKE driver that claims and grades through the real registry worklist; a pending run of another model is never
dispatched; a row another live session holds is left alone; rule-scored families are confirmed, never dispatched; `--dry-run` prints and
dispatches nothing; the run_contrast pass runs once both a steered run and its anchor are complete (through `emit_contrasts` when that
module is in the tree; otherwise the daemon says so and continues)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sisteer.registry import db as DB
from sisteer.registry import emit, identity, synth
from sisteer.registry import harvest as H
from sisteer.registry import records as R

from test_worklist_run_id_filter import OTHER_MODEL, _dims_with_other_model   # the same second model the selector test seeds

REPO = Path(__file__).resolve().parents[1]
GD_RETRYABLE_CAP = 6
DAEMON = REPO / "scripts" / "registry" / "grade_daemon.py"
# the free space every mirror root reports to the daemon under test (grade_daemon.MIRROR_FREE_BYTES_ENV): a stated 1 TB, whatever the host
# has -- a container filesystem reports ~9.2e18 bytes free (its total too, so the clamp in db.free_bytes does not help), which no floor a
# test can name exceeds, and a real host's reading is whatever its disk holds today. 1 TB sits above the 1 MiB floor the tests pass (a
# 2 x 4096-byte group mirrors) and below the 10**18 floor and the 2 x 10**15-byte group the low-disk tests name (both skip, by construction).
# The daemon honours the variable ONLY while PYTEST_CURRENT_TEST is in its environment (pytest sets it for the test's duration; the daemon
# subprocess inherits it through _daemon_argv_env's dict(os.environ, ...)); outside pytest it is a stray, named and ignored, and the real
# reading stands (test_the_free_bytes_override_is_honoured_only_under_pytest_outside_it_the_real_low_reading_pauses_the_group).
FAKE_FREE_BYTES = 10 ** 12

FAKE_DRIVER = r'''
import argparse, json, os, sys
if os.environ.get("FAKE_DRIVER_FAIL_FILE") and os.path.exists(os.environ["FAKE_DRIVER_FAIL_FILE"]):
    print("fake driver: failing on purpose"); sys.exit(1)
if os.environ.get("FAKE_DRIVER_LOCKHELD_FILE") and os.path.exists(os.environ["FAKE_DRIVER_LOCKHELD_FILE"]):   # the grader lost the write lock to the harvester
    os.unlink(os.environ["FAKE_DRIVER_LOCKHELD_FILE"])
    print("Traceback (most recent call last):\n  File \"pending.py\", line 147, in claim\n    with DB.write_lock(root, timeout_s=30.0):", file=sys.stderr)
    print("sisteer.registry.db.LockHeld: lock held: /reg/harvester.lock", file=sys.stderr); sys.exit(1)
if os.environ.get("FAKE_DRIVER_AUTH401_FILE") and os.path.exists(os.environ["FAKE_DRIVER_AUTH401_FILE"]):   # D2: the API rejected the judge's credential (the controls judge's real tail shape)
    print("[served-judge 08:10:09] judging on org=lowprio base_url=https://api.anthropic.com gate=200 cell_parallel=6")
    print("Traceback (most recent call last):\n  File \"judging.py\", line 1868, in judge_all\n    raise JudgingError(", file=sys.stderr)
    print("sisteer.capeval.judging.JudgingError: 450 of 450 judge request(s) are still unverdicted after retries — refusing a partial grade. Failure types: "
          "{\"AuthenticationError: Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'invalid x-api-key'}, 'request_\": 450} "
          "(prompt/response text withheld).", file=sys.stderr)
    print("[served-judge 08:10:28] JUDGE_FAILED: 3 cell(s) failed, 0 done"); sys.exit(1)
if os.environ.get("FAKE_DRIVER_AUTH401_WIF_FILE") and os.path.exists(os.environ["FAKE_DRIVER_AUTH401_WIF_FILE"]):   # the FEDERATED rejection (judge_sonnet's real shape under wif; the token endpoint's OAuth error-string body)
    msg = ("the API rejected the lowprio pool's credential: WorkloadIdentityError (HTTP 401) under SISTEER_SECRETS_MODE=wif -- stopping at the "
           "first credential rejection (D6): no retry, no further request on this credential; fix the federation ids file / the VM identity and re-run")
    # the real driver's ORDER: the stdout line is flushed before SystemExit prints the typed message on stderr, so the typed message is
    # the terminal line (the daemon merges stdout and stderr into one log and reads its last non-blank line)
    print("[judge] CREDENTIAL REJECTED on lowprio at french/steer arm baseline: " + msg + " -- 70 of this arm's 72 requests were never issued, "
          "0 verdict(s) obtained before the stop are cached; no report written (D6)", flush=True)
    print("[judge] " + msg, file=sys.stderr); sys.exit(1)
if os.environ.get("FAKE_DRIVER_ENV_RECORD"):
    open(os.environ["FAKE_DRIVER_ENV_RECORD"], "w").write(os.environ.get("SISTEER_REGISTRY_LOCK_TIMEOUT_S", ""))
import time as _t
if os.environ.get("FAKE_DRIVER_PID_FILE"):
    open(os.environ["FAKE_DRIVER_PID_FILE"], "w").write(str(os.getpid()))
if os.environ.get("FAKE_DRIVER_WRITE_HOLD"):                                                  # g31-fix: the operator writes the hold while this grader runs
    open(os.environ["FAKE_DRIVER_WRITE_HOLD"], "w").write("# written while group A graded\nhold from the fake grader\n")
_i = 0
while os.environ.get("FAKE_DRIVER_HOLD_FILE") and os.path.exists(os.environ["FAKE_DRIVER_HOLD_FILE"]) and _i < 600:   # a long judge, cwd = the repo
    _t.sleep(0.1); _i += 1
if os.environ.get("FAKE_DRIVER_FAIL_AFTER_HOLD_FILE") and os.path.exists(os.environ["FAKE_DRIVER_FAIL_AFTER_HOLD_FILE"]):   # a long judge that then dies
    print("Traceback (most recent call last):\n  File grader.py\nKeyError: 'store'", file=sys.stderr); sys.exit(1)
if os.environ.get("FAKE_DRIVER_IDENTITY_RECORD"):   # g29-fix-3: after the hold, the misalign grader's rows step reads ITS dispatch's identity snapshot
    _d = sys.argv[sys.argv.index("--served-identity-dir") + 1] if "--served-identity-dir" in sys.argv else None
    _p = os.path.join(_d, "cells_full.json") if _d else None
    open(os.environ["FAKE_DRIVER_IDENTITY_RECORD"], "w").write(json.dumps({"served_identity_dir": _d, "cells_full": open(_p).read() if _p and os.path.isfile(_p) else None,
                                                                            "files": sorted(os.listdir(_d)) if _d and os.path.isdir(_d) else None}))
sys.path.insert(0, os.environ["SISTEER_SRC"])
from sisteer.registry import worklist as W
ap = argparse.ArgumentParser()
W.add_worklist_arguments(ap)
ap.add_argument("--record", required=True)
ap.add_argument("--target", default=None)
ap.add_argument("--attacks", default=None)
args, rest = ap.parse_known_args()
with open(args.record, "a") as fh:
    fh.write(json.dumps({"argv": sys.argv[1:], "run_ids": args.run_id, "store_root_map": args.store_root_map, "target": args.target, "attacks": args.attacks}) + "\n")
if "--rows-only" in sys.argv:                          # g29-fix-4: the daemon's rows retry over the retained snapshot -- nothing judged or filed
    _k = 0
    while os.environ.get("FAKE_DRIVER_ROWS_ONLY_HOLD_FILE") and os.path.exists(os.environ["FAKE_DRIVER_ROWS_ONLY_HOLD_FILE"]) and _k < 600:   # g29-fix-5: a long rows step
        _t.sleep(0.1); _k += 1
    if os.environ.get("FAKE_DRIVER_ROWS_FAIL") and not (os.environ.get("FAKE_DRIVER_ROWS_RETRY_OK_FILE") and os.path.exists(os.environ["FAKE_DRIVER_ROWS_RETRY_OK_FILE"])):
        print("[judge] ROWS STEP FAILED: sisteer.capeval.misalign.rows.main returned 2 -- the judgments already filed stay; exit 1", flush=True); sys.exit(1)
    _root = (args.store_root_map or ["==."])[0].split("=", 2)[2]            # the mapped local root IS the judged results root (the fake template passes no --results-root)
    _bins = sorted(str(p) for p in __import__("pathlib").Path(_root).glob("stores/*/store.bin"))   # g29-fix-5: the rows are DERIVED from the kept copies ...
    os.makedirs(os.path.join(_root, "store"), exist_ok=True)                                         # ... into the root's store, as the real rows step files them
    open(os.path.join(_root, "store", "rows.fake.json"), "w").write(json.dumps({"run_ids": args.run_id, "from_store_bins": _bins,
                                                                                "served_identity_dir": sys.argv[sys.argv.index("--served-identity-dir") + 1] if "--served-identity-dir" in sys.argv else None}))
    print("[judge] rows-only: rows filed (fake)", flush=True); sys.exit(0)
wl = W.RegistryWorklist(args.registry_root, args.judge, args.session or "fake-driver", args.campaign, driver="fake_driver.py", **W.selectors_from_args(args))
rows = wl.pending()
def grade(row, scope):
    n_items = int(os.environ.get("FAKE_DRIVER_SHORT_ITEMS") or 200)      # < 200: a complete judgment SHORT of the run's coverage
    scope.items([{"item_id": f"q{i:05d}", "metric_id": "asr", "value": float(i % 2), "label": "harmful" if i % 2 else "safe"} for i in range(n_items)])
    scope.metrics([{"metric_id": "asr", "threshold": 0.5, "value": 0.5, "k": n_items // 2, "n": n_items}])
if os.environ.get("FAKE_DRIVER_UNSCORED_AFTER_N") is not None:
    # the BoN leg's shape (the registry owner's cap ruling of 2026-10-13 23:29): the grader claims every row up front (plan §8.3), files the
    # first N complete, the next row's judging ends with items UNSCORED -- the scope files `partial`, the JB1 line is the terminal line -- and
    # the rows after it are never reached (claimed, never judged); the records are folded before exiting, as the real grader's are by the
    # harvester within a cadence
    _n_ok = int(os.environ["FAKE_DRIVER_UNSCORED_AFTER_N"])
    rows = sorted(rows, key=lambda r: str(r["run_id"]))
    _pos = {str(r["run_id"]): i for i, r in enumerate(rows)}
    def _grade_or_shed(row, scope):
        if _pos[str(row["run_id"])] < _n_ok:
            grade(row, scope); return
        raise RuntimeError("8 judge item(s) still UNSCORED after live grading (bounded retries) — refusing to aggregate a partial grade (JB1)")
    try:
        wl.grade(rows, _grade_or_shed)
    except RuntimeError as _exc:
        wl.finish(fold=True)
        print(f"[fake-bon] BoN: {_exc}. First: dataset=jbb (goal/response text withheld)", file=sys.stderr, flush=True); sys.exit(1)
    wl.finish(fold=True); sys.exit(0)
if os.environ.get("FAKE_DRIVER_NO_FOLD"):
    # the harvester-folds contract on EITHER parent: claim (the claim folds at once, as a real grader's does), then file each judgment
    # through the worklist's scope (records + push) and fold nothing
    wl.claim_all(rows)
    for row in rows:
        scope = wl.scope(row)
        with scope:
            grade(row, scope)
else:
    wl.grade(rows, grade)
    try:
        wl.finish(fold=True)    # main's harvest-lock worklist (f1347e9cf): grade() no longer folds — the tests' premise that a grader's judgments
    except TypeError:           # are folded when it exits is kept explicitly; the pre-merge finish() takes no argument and grade() has already folded
        pass
_j = 0
while os.environ.get("FAKE_DRIVER_HOLD_AFTER_FILE_FILE") and os.path.exists(os.environ["FAKE_DRIVER_HOLD_AFTER_FILE_FILE"]) and _j < 600:   # g29-fix-5: filed, then a long tail (a daemon may die here)
    _t.sleep(0.1); _j += 1
if os.environ.get("FAKE_DRIVER_ROWS_FAIL"):          # g29-fix-3: the judgments are filed and folded; the derived rows step then fails -> the grader exits 1
    print("[judge] ROWS STEP FAILED: sisteer.capeval.misalign.rows.main returned 2 -- the judgments already filed stay; exit 1", flush=True); sys.exit(1)
'''


def _none_baseline_include_ruling():
    """An anchor is a paper baseline row (in_current_paper 1); this world has no roster, so an early INCLUDE ruling over the `none`
    object makes its judged none runs paper rows (the include branch precedes the roster branch of both CASEs in schema_def)."""
    from sisteer.registry.adapters import roster as RO
    raw = {"ruling_id": "r_synth-include-none-baseline", "round_id": "r_synth", "action": "include", "scope_kind": "key",
           "scope_json": {"match": {"object_ids": ["none"]}}, "reason": "synthetic: the none baseline rows are paper rows without a roster",
           "ruled_utc": synth.t(10), "ruled_by": "tests", "source_ref": "tests/test_grade_daemon.py",
           "provenance": {"kind": "user_direct", "message_utc": "2026-10-02T00:00:00Z"}}
    return synth.rec("ruling", RO.normalise_ruling(raw), synth.t(10))


def _root(tmp_path: Path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), _dims_with_other_model() + [_none_baseline_include_ruling()])
    rids = {}
    specs = (("job7@none", "none", "none", synth.MODEL_ID), ("job7@steer", "lora@abcd", synth.OBJECT_SHA, synth.MODEL_ID), ("job8@other", "none", "none", OTHER_MODEL))
    for k, (key, okey, oid, model) in enumerate(specs):
        rid, recs = synth.run_records(key, 100 + 20 * k, object_key=okey, object_id=oid, through="generated", model_id=model,
                                      view_legacy="synthmodel_4b_s1", cluster_job_id="7" if key.startswith("job7") else "8")
        recs[0]["payload"]["store_uri"] = f"/cluster_a/user/exp/stores/{key.replace('@', '_')}"
        R.write_records(os.path.join(root, "records", "c", f"00000{k + 1}-run.jsonl"), recs)
        rids[key] = rid
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.as_dict()
    return root, rids


def _root_many(tmp_path: Path, n: int):
    """`n` pending `generated` runs of the model in ONE dispatch group (one family, model key, campaign and store root), each at its own
    trial_index so that a later-judged one never supersedes a still-pending one (two trials of one object are distinct runs, never
    successors). Returns the root and the run ids in run-id order -- the order the worklist (and the fake driver) reads them in."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), _dims_with_other_model() + [_none_baseline_include_ruling()])
    rids = []
    for k in range(n):
        key = f"job7@cell{k:02d}"
        rid, recs = synth.run_records(key, 100 + 20 * k, object_key="none", object_id="none", through="generated", model_id=synth.MODEL_ID,
                                      view_legacy="synthmodel_4b_s1", cluster_job_id="7", trial_index=k)
        recs[0]["payload"]["store_uri"] = f"/cluster_a/user/exp/stores/{key.replace('@', '_')}"
        R.write_records(os.path.join(root, "records", "c", f"{k + 1:06d}-run.jsonl"), recs)
        rids.append(rid)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.as_dict()
    return root, sorted(rids)


def _daemon_argv_env(root: str, tmp_path: Path, *extra: str, dry_run: bool = False, env_extra: dict | None = None):
    fake = tmp_path / "fake_driver.py"
    fake.write_text(FAKE_DRIVER)
    record = tmp_path / "driver_calls.jsonl"
    template = f"{{python}} {fake} --worklist registry --registry-root {{registry_root}} {{run_id_flags}} {{store_map_flags}} --record {record} {{extra}}"
    env = dict(os.environ, SISTEER_SRC=str(REPO / "src"), PYTHONPATH=str(REPO / "src"), REGISTRY_ROOT=root,
               GRADE_DAEMON_FREE_BYTES=str(FAKE_FREE_BYTES))        # the mirror roots' free space is a stated figure, never the host's disk (env_extra may override);
    # honoured because the daemon inherits this process's PYTEST_CURRENT_TEST with the rest of os.environ -- a daemon outside pytest ignores it
    env.pop(emit.DISABLED_ENV, None)
    env.update(env_extra or {})
    fold = ["--grader-session", "fake-driver"]                        # the FAKE driver's session (a real grader claims under default_session())
    if "--mirror-min-free-bytes" not in extra:                        # the floor is about the grade host's disk, not the test's /tmp (r44: a 92 % root fs skipped every group)
        fold += ["--mirror-min-free-bytes", str(1024 * 1024)]
    argv = [sys.executable, str(DAEMON), "--registry-root", root, "--model-id", synth.MODEL_ID, "--campaign", synth.CAMPAIGN, "--once",
            "--python", sys.executable, "--repo", str(REPO), "--state-dir", str(tmp_path / "state"), *fold,
            "--store-root-map", f"cluster_a=/cluster_a/user/exp={tmp_path / 'mirror'}",
            "--driver", f"static={template}", "--extra", "static=--target {model_key}", "--eval-family", f"{synth.EVAL_ID}=static",
            "--spool", os.path.join(root, "records", "contrasts"), *extra]
    if dry_run:
        argv.append("--dry-run")
    return argv, env


def _run_daemon(root: str, tmp_path: Path, *extra: str, dry_run: bool = False, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    argv, env = _daemon_argv_env(root, tmp_path, *extra, dry_run=dry_run, env_extra=env_extra)
    return subprocess.run(argv, capture_output=True, text=True, env=env, timeout=600, cwd=str(REPO))


def _start_daemon(root: str, tmp_path: Path, *extra: str, env_extra: dict | None = None) -> subprocess.Popen:
    argv, env = _daemon_argv_env(root, tmp_path, *extra, env_extra=env_extra)
    return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=str(REPO))


def _events(tmp_path: Path):
    p = tmp_path / "state" / "grade_daemon.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def _proc_state(pid):
    """The process's state letter from /proc/<pid>/stat (R, S, D, T, Z, ...), or None when the entry is gone or unreadable -- ONE open, no
    exists()-then-open() pair: a grader that exits between the two raised FileNotFoundError under the landing lock's load (sweephold-fix-2's
    first chain, 2026-10-04); the race is closed by reading once and treating a vanished entry as exited."""
    try:
        with open(f"/proc/{int(pid)}/stat", encoding="utf-8") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError, ValueError):
        return None


def _grader_exited(pid) -> bool:
    """Exited = gone from the process table, or a zombie awaiting its parent's reap."""
    return _proc_state(pid) in (None, "Z")


def test_dry_run_prints_the_dispatch_and_touches_nothing(tmp_path):
    root, rids = _root(tmp_path)
    r = _run_daemon(root, tmp_path, dry_run=True, env_extra={"SISTEER_COORDINATION_CHANNEL": "#example-api-coordination"})
    assert r.returncode == 0, r.stderr + r.stdout
    assert "coordination (post to #example-api-coordination;" in r.stdout          # the site's channel, named by the site config
    ev = _events(tmp_path)
    disp = [e for e in ev if e["event"] == "dispatch"]
    assert len(disp) == 1 and disp[0]["dry_run"] is True and disp[0]["family"] == "static" and disp[0]["model_key"] == "synthmodel_4b_s1"
    assert sorted(disp[0]["run_ids"]) == sorted([rids["job7@none"], rids["job7@steer"]])          # the other model's run is not dispatched
    assert f"--run-id {rids['job7@none']}" in disp[0]["cmd"] and "--store-root-map cluster_a=/cluster_a/user/exp=" in disp[0]["cmd"]
    assert "--target synthmodel_4b_s1" in disp[0]["cmd"]                                       # {model_key} substituted into the family's extra args
    assert not (tmp_path / "driver_calls.jsonl").exists()
    with DB.reading(DB.db_path(root)) as conn:
        assert {r[0] for r in conn.execute("SELECT lifecycle_state FROM run").fetchall()} == {"generated"}


def test_once_dispatches_exactly_the_models_pending_runs_and_the_driver_claims_and_grades_them(tmp_path):
    root, rids = _root(tmp_path)
    r = _run_daemon(root, tmp_path)
    assert r.returncode == 0, r.stderr + r.stdout
    calls = [json.loads(l) for l in (tmp_path / "driver_calls.jsonl").read_text().splitlines()]
    assert len(calls) == 1
    assert sorted(calls[0]["run_ids"]) == sorted([rids["job7@none"], rids["job7@steer"]])
    assert calls[0]["store_root_map"] == [f"cluster_a=/cluster_a/user/exp={tmp_path / 'mirror'}"] and calls[0]["target"] == "synthmodel_4b_s1"
    assert H.ingest_once(root, workers=0, min_free=0, derive_facts=False).refused == 0      # the harvester's fold of what the driver spooled (finish() folds nothing, 2026-09-30)
    with DB.reading(DB.db_path(root)) as conn:
        for key in ("job7@none", "job7@steer"):
            assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rids[key],)).fetchone()[0] == "judged"
            assert conn.execute("SELECT COUNT(*) FROM judgment WHERE run_id = ? AND status = 'complete'", (rids[key],)).fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM item_result WHERE run_id = ? AND metric_id = 'asr'", (rids[key],)).fetchone()[0] == 200
        # the other model's run: never claimed, still pending for its judge
        assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rids["job8@other"],)).fetchone()[0] == "generated"
        assert conn.execute("SELECT COUNT(*) FROM judgment WHERE run_id = ?", (rids["job8@other"],)).fetchone()[0] == 0
    ev = _events(tmp_path)
    tick = [e for e in ev if e["event"] == "tick"][-1]
    assert tick["pending"] == 2 and tick["dispatched"] == 2 and tick["failed"] == 0
    # the second tick: nothing pending for the model; the contrast pass files the steered-vs-anchor row when emit_contrasts is in the tree
    r2 = _run_daemon(root, tmp_path)
    assert r2.returncode == 0, r2.stderr + r2.stdout
    ev2 = _events(tmp_path)
    assert [e for e in ev2 if e["event"] == "tick"][-1]["pending"] == 0
    # the contrast pass ran on BOTH ticks and wrote exactly once -- the SECOND tick (2026-09-30): the driver's finish() folds nothing, so
    # the first tick's pass sees the two judgments still on the spool and writes 0; the harvester's fold (the stand-in above) lands them
    # and the second tick's pass files the steered-vs-anchor row (the third finds the record in the spool). A `contrasts_failed` /
    # `contrasts_unavailable` event is a defect in the daemon's call into emit_contrasts, never an acceptable outcome: the pass once
    # called a function the module did not define and logged the AttributeError every tick while an `else` branch here let it through.
    contrast_events = [e for e in ev2 if str(e["event"]).startswith("contrasts")]
    assert [e["event"] for e in contrast_events] == ["contrasts", "contrasts"], contrast_events
    assert sorted(e["written"] for e in contrast_events) == [0, 1], contrast_events   # exactly one contrast across the two ticks, on the tick the fold shows both complete (tick 1 here; tick 2 under a harvester-folded worklist)
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    with DB.reading(DB.db_path(root)) as conn:
        rows = conn.execute("SELECT run_id, anchor_run_id, anchor_kind, n_pairs, delta FROM run_contrast").fetchall()
        assert len(rows) == 1 and rows[0][0] == rids["job7@steer"] and rows[0][1] == rids["job7@none"] and rows[0][3] == 200, rows
        assert rows[0][2] == "same_standup_none"
    # a third tick writes no second contrast (idempotent)
    _run_daemon(root, tmp_path)
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM run_contrast").fetchone()[0] == 1


FAKE_RSYNC = """#!/bin/bash
# a fake rsync: `--dry-run --stats` reports the store's size ($FAKE_RSYNC_SIZE, default 4096) and copies nothing; a real call "mirrors" a
# 4096-byte store file into the destination (the last argument) and records the call; while $FAKE_RSYNC_FAIL_ONCE exists the first real call
# writes the file (the headroom is consumed) and exits 1, removing the flag
case " $* " in *" --dry-run "*) echo "Total transferred file size: ${FAKE_RSYNC_SIZE:-4096} bytes"; exit 0;; esac
dst="${@: -1}"; mkdir -p "$dst"; head -c 4096 /dev/zero > "$dst/store.bin"; echo "$*" >> "${FAKE_RSYNC_LOG:?}"
if [ -n "${FAKE_RSYNC_FAIL_ONCE:-}" ] && [ -e "$FAKE_RSYNC_FAIL_ONCE" ]; then rm -f "$FAKE_RSYNC_FAIL_ONCE"; echo "rsync: connection unexpectedly closed" >&2; exit 12; fi
"""


def _mirror_args(tmp_path: Path):
    """The daemon mirrors the group's stores through a fake rsync into the store map's local root (the grade mirror)."""
    fake = tmp_path / "rsync"; fake.write_text(FAKE_RSYNC); fake.chmod(0o755)
    return ["--mirror", "cluster_a=cluster-a-login", "--rsync-bin", str(fake)]


def test_the_grade_mirror_is_pruned_once_the_registry_shows_the_group_filed_and_the_tick_carries_its_du(tmp_path, monkeypatch):
    """The store directories a tick rsynced for a group are removed — with their bytes in one `mirror_pruned` event each —
    only once the registry shows every run of the group filed (`judged`, or a complete judgment), read from a durable ledger on a later
    pass (here the worklist ingests as it grades, so the same tick's closing pass already confirms it); never the mirror root; the tick
    summary carries each mirror root's du and free bytes and the ledger's depth."""
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, rids = _root(tmp_path)
    mirror = tmp_path / "mirror"
    r = _run_daemon(root, tmp_path, *_mirror_args(tmp_path))
    assert r.returncode == 0, r.stderr + r.stdout
    ev = _events(tmp_path)
    mirrored = [e for e in ev if e["event"] == "mirror"]
    assert sorted(e["run_id"] for e in mirrored) == sorted([rids["job7@none"], rids["job7@steer"]])
    assert len((tmp_path / "rsync_calls.log").read_text().splitlines()) == 2
    dirs = [str(mirror / "stores" / "job7_none"), str(mirror / "stores" / "job7_steer")]
    with DB.reading(DB.db_path(root)) as conn:
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rids[k],)).fetchone()[0] == "judged" for k in ("job7@none", "job7@steer"))
    pruned = [e for e in ev if e["event"] == "mirror_pruned"]
    assert sorted(e["dir"] for e in pruned) == sorted(dirs) and all(e["bytes"] == 4096 for e in pruned) and sorted(e["run_id"] for e in pruned) == sorted([rids["job7@none"], rids["job7@steer"]])
    assert not any(os.path.isdir(d) for d in dirs) and mirror.is_dir()                       # the store dirs are gone; the mirror root stays
    tick = [e for e in ev if e["event"] == "tick"][-1]
    assert tick["mirror_pruned_dirs"] == 2 and tick["mirror_pruned_bytes"] == 8192 and tick["mirror_prune_pending"] == 0 and tick["skipped_low_disk"] == 0
    assert set(tick["mirror_du_bytes"]) == {"cluster_a"} and tick["mirror_du_bytes"]["cluster_a"] == 0 and tick["mirror_free_bytes"]["cluster_a"] > 0
    assert json.loads((tmp_path / "state" / "mirror_prune_pending.json").read_text())["pending"] == []
    # the ledger defers when the registry does not (yet) show the filing: a queued group whose run is still `generated` is kept, untouched
    keep = tmp_path / "mirror" / "stores" / "job8_other"; keep.mkdir(parents=True); (keep / "store.bin").write_bytes(b"x" * 10)
    import grade_daemon as GD  # noqa: E402  (scripts/registry is on sys.path through the front-door test lib in the sibling module)
    GD.mirror_prune_enqueue(str(tmp_path / "state"), family="static", model_key="k", campaign=synth.CAMPAIGN, run_ids=[rids["job8@other"]],
                            mirrored=[{"run_id": rids["job8@other"], "substrate": "cluster_a", "local": str(keep), "root": str(mirror), "rc": 0}], rc=0)
    GD.mirror_prune_enqueue(str(tmp_path / "state"), family="static", model_key="k", campaign=synth.CAMPAIGN, run_ids=[rids["job8@other"]],
                            mirrored=[{"run_id": rids["job8@other"], "substrate": "cluster_a", "local": str(mirror), "root": str(mirror), "rc": 0}], rc=0)   # the ROOT: never queued
    assert len(json.loads((tmp_path / "state" / "mirror_prune_pending.json").read_text())["pending"]) == 1
    pr = GD.mirror_prune_pass(root, str(tmp_path / "state"), dry_run=False, campaign=synth.CAMPAIGN)
    assert pr == {"groups": 0, "dirs": 0, "bytes": 0} and keep.is_dir()
    r2 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path))
    assert r2.returncode == 0
    tick2 = [e for e in _events(tmp_path) if e["event"] == "tick"][-1]
    assert tick2["mirror_prune_pending"] == 1 and tick2["mirror_du_bytes"]["cluster_a"] == 10 and keep.is_dir()


def test_prune_asks_the_worklists_own_question_a_short_or_wrong_judge_judgment_keeps_the_store(tmp_path):
    """A run with lifecycle `judged` or SOME complete judgment can still be pending — the instrument's required judge has not
    covered the required items (`coverage_short`), or the complete judgment is another judge's (`awaiting_second_judge`) — and the worklist
    will ask for its store again. The prune asks the worklist's own question (`v_pending_judgment` through `pending.pending`) and keeps both
    stores; it prunes once the required judge's complete, full-coverage judgments land."""
    from sisteer.registry import identity as I, pending as P
    import grade_daemon as GD
    root, rids = _root(tmp_path)
    a, b = rids["job7@none"], rids["job7@steer"]
    mirror = tmp_path / "mirror"; state = tmp_path / "state"
    dirs = {rid: mirror / "stores" / f"job7_{k}" for rid, k in ((a, "none"), (b, "steer"))}
    for d in dirs.values():
        d.mkdir(parents=True); (d / "store.bin").write_bytes(b"x" * 4096)
    GD.mirror_prune_enqueue(str(state), family="static", model_key="k", campaign=synth.CAMPAIGN, run_ids=[a, b],
                            mirrored=[{"run_id": rid, "substrate": "cluster_a", "local": str(d), "root": str(mirror), "rc": 0} for rid, d in dirs.items()], rc=0)
    # A: the required judge's complete judgment covers 150 of the required 200 (+ run_state judged); B: a complete judgment by ANOTHER judge
    R.write_records(os.path.join(root, "records", "c", "000010-short.jsonl"), [
        synth.rec("judgment", {"judgment_id": I.judgment_id(a, synth.JUDGE_ID), "run_id": a, "judge_id": synth.JUDGE_ID, "n_items_judged": 150, "status": "complete"}, synth.t(300)),
        synth.rec("run_state", {"run_id": a, "state": "judged", "event_utc": synth.t(301), "evidence_kind": "completion_artifact"}, synth.t(301)),
        synth.rec("judgment", {"judgment_id": I.judgment_id(b, synth.JUDGE_ID_2), "run_id": b, "judge_id": synth.JUDGE_ID_2, "n_items_judged": 200, "status": "complete"}, synth.t(302)),
        synth.rec("run_state", {"run_id": b, "state": "judged", "event_utc": synth.t(303), "evidence_kind": "completion_artifact"}, synth.t(303))])
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    with DB.reading(DB.db_path(root)) as conn:
        # what the OLD predicate saw: judged / a complete judgment for both — and what the worklist sees: both still pending
        assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (a,)).fetchone()[0] in ("judged", "ingested")
        assert conn.execute("SELECT COUNT(*) FROM judgment WHERE run_id = ? AND status = 'complete'", (b,)).fetchone()[0] == 1
        reasons = {r["run_id"]: r["pending_reason"] for r in P.pending(conn, None, synth.CAMPAIGN, run_id=(a, b))}
        assert reasons == {a: "coverage_short", b: "awaiting_second_judge"}, reasons
        assert GD.run_pending(conn, a, synth.CAMPAIGN) and GD.run_pending(conn, b, synth.CAMPAIGN)
    pr = GD.mirror_prune_pass(root, str(state), dry_run=False, campaign=synth.CAMPAIGN)
    assert pr == {"groups": 0, "dirs": 0, "bytes": 0} and all(d.is_dir() for d in dirs.values())
    assert len(json.loads((state / "mirror_prune_pending.json").read_text())["pending"]) == 1
    # the required judge completes both with full coverage: no run pending -> pruned
    R.write_records(os.path.join(root, "records", "c", "000011-full.jsonl"), [
        synth.rec("judgment", {"judgment_id": I.judgment_id(a, synth.JUDGE_ID), "run_id": a, "judge_id": synth.JUDGE_ID, "n_items_judged": 200, "status": "complete"}, synth.t(400)),
        synth.rec("judgment", {"judgment_id": I.judgment_id(b, synth.JUDGE_ID), "run_id": b, "judge_id": synth.JUDGE_ID, "n_items_judged": 200, "status": "complete"}, synth.t(401))])
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    with DB.reading(DB.db_path(root)) as conn:
        assert not GD.run_pending(conn, a, synth.CAMPAIGN) and not GD.run_pending(conn, b, synth.CAMPAIGN)
    pr = GD.mirror_prune_pass(root, str(state), dry_run=False, campaign=synth.CAMPAIGN)
    assert pr == {"groups": 1, "dirs": 2, "bytes": 8192} and not any(d.exists() for d in dirs.values()) and mirror.is_dir()


def test_a_failed_mirror_or_dispatch_retires_the_incomplete_copy_and_the_next_tick_recovers_the_space(tmp_path, monkeypatch):
    """An rsync that consumes the headroom and then fails (or a dispatch that fails) must not leave a retained partial copy under
    the floor that every later tick refuses to re-mirror. The tick retires the incomplete copies of the still-pending runs
    (`mirror_retired_incomplete`, with bytes) right after the dispatch returns; the next tick sees the space again, mirrors, dispatches,
    grades and prunes. And the guard is bounded by the stores' probed size: a group larger than free − floor is skipped whole, nothing copied."""
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    fail_rsync = tmp_path / "rsync_fail_once"; fail_rsync.write_text(""); monkeypatch.setenv("FAKE_RSYNC_FAIL_ONCE", str(fail_rsync))
    fail_driver = tmp_path / "driver_fail"; fail_driver.write_text(""); monkeypatch.setenv("FAKE_DRIVER_FAIL_FILE", str(fail_driver))
    root, rids = _root(tmp_path)
    mirror = tmp_path / "mirror"; dirs = [mirror / "stores" / "job7_none", mirror / "stores" / "job7_steer"]
    r = _run_daemon(root, tmp_path, *_mirror_args(tmp_path))
    assert r.returncode == 0, r.stderr + r.stdout
    ev = _events(tmp_path)
    assert [e["rc"] for e in ev if e["event"] == "mirror_failed"] == [12] and [e["rc"] for e in ev if e["event"] == "dispatched"] == [1]
    retired = [e for e in ev if e["event"] == "mirror_retired_incomplete"]
    assert sorted(e["dir"] for e in retired) == sorted(str(d) for d in dirs) and all(e["bytes"] == 4096 for e in retired) and retired[0]["dispatch_rc"] == 1
    assert not any(d.exists() for d in dirs) and mirror.is_dir() and not fail_rsync.exists()
    tick = [e for e in ev if e["event"] == "tick"][-1]
    assert tick["failed"] == 2 and tick["mirror_retired_dirs"] == 2 and tick["mirror_retired_bytes"] == 8192 and tick["mirror_prune_pending"] == 0 and tick["mirror_du_bytes"]["cluster_a"] == 0
    with DB.reading(DB.db_path(root)) as conn:
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rids[k],)).fetchone()[0] == "generated" for k in ("job7@none", "job7@steer"))
    # the failed dispatch quarantined the runs (its own test below); released, the group is larger than the headroom: skipped whole before
    # any copy (the probe bounds the floor check) — a skip is not a failure, nothing is quarantined again
    fail_driver.unlink()
    (tmp_path / "state" / "dispatch_quarantine.release").write_text("*\n")
    monkeypatch.setenv("FAKE_RSYNC_SIZE", str(10 ** 15))
    r2 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path))
    assert r2.returncode == 0, r2.stderr + r2.stdout
    ev2 = _events(tmp_path)
    skipped = [e for e in ev2 if e["event"] == "mirror_skipped_low_disk"]
    assert len(skipped) == 1 and skipped[0]["store_bytes"] == 2 * 10 ** 15 and skipped[0]["unknown_sizes"] == 0 and skipped[0]["free_bytes"] == FAKE_FREE_BYTES
    assert not any(d.exists() and any(d.iterdir()) for d in dirs) and len((tmp_path / "rsync_calls.log").read_text().splitlines()) == 2   # no new copy
    assert [e for e in ev2 if e["event"] == "tick"][-1]["skipped_low_disk"] == 2 and [e for e in ev2 if e["event"] == "tick"][-1]["quarantined"] == 0
    assert len([e for e in ev2 if e["event"] == "quarantine_released" and e["reason"] == "release_file"]) == 2 and json.loads((tmp_path / "state" / "dispatch_quarantine.json").read_text())["runs"] == {}
    # room again, everything healthy: mirrored, dispatched, graded, pruned
    monkeypatch.delenv("FAKE_RSYNC_SIZE")
    r3 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path))
    assert r3.returncode == 0, r3.stderr + r3.stdout
    tick3 = [e for e in _events(tmp_path) if e["event"] == "tick"][-1]
    assert tick3["dispatched"] == 2 and tick3["mirror_pruned_dirs"] == 2 and tick3["mirror_pruned_bytes"] == 8192 and tick3["mirror_prune_pending"] == 0
    assert not any(d.exists() for d in dirs) and len((tmp_path / "rsync_calls.log").read_text().splitlines()) == 4


def _load_daemon_module():
    import importlib.util
    sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(DAEMON.parent))
    spec = importlib.util.spec_from_file_location("grade_daemon_under_test", DAEMON)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)   # type: ignore[union-attr]
    return mod


def test_a_failed_dispatch_quarantines_its_runs_until_the_cause_changes_and_the_tick_raises_a_problem(tmp_path, monkeypatch):
    """The loop of 2026-09-30: a grader that dies at its claim made every tick re-mirror the same stores, launch the same graders and retire
    the copies, with nothing raised. A failed dispatch now QUARANTINES its runs — one `dispatch_failed` event with rc, stderr tail, daemon
    sha and the ledger entry per run — and a later tick neither mirrors nor dispatches them, raising one `problem` (kind
    dispatch_quarantine) and carrying `quarantined` in its summary, until the cause changes: a new daemon code sha, a changed store
    generation, or the release file (consumed when read). The failed tick's copies are still retired."""
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    fail_driver = tmp_path / "driver_fail"; fail_driver.write_text(""); monkeypatch.setenv("FAKE_DRIVER_FAIL_FILE", str(fail_driver))
    root, rids = _root(tmp_path)
    state = tmp_path / "state"; ledger = state / "dispatch_quarantine.json"
    both = sorted(rids[k] for k in ("job7@none", "job7@steer"))
    rsync_lines = lambda: len((tmp_path / "rsync_calls.log").read_text().splitlines()) if (tmp_path / "rsync_calls.log").exists() else 0
    # tick 1: the dispatch fails → recorded, copies retired
    r = _run_daemon(root, tmp_path, *_mirror_args(tmp_path)); assert r.returncode == 0, r.stderr + r.stdout
    ev = _events(tmp_path)
    df = [e for e in ev if e["event"] == "dispatch_failed"]
    assert len(df) == 1 and df[0]["rc"] == 1 and sorted(df[0]["run_ids"]) == both and "failing on purpose" in df[0]["stderr_tail"] and df[0]["daemon_sha"]
    assert os.path.isfile(df[0]["dispatch_log"]) and "failing on purpose" in Path(df[0]["dispatch_log"]).read_text()
    assert [e for e in ev if e["event"] == "dispatched"][-1]["log"] == df[0]["dispatch_log"]
    doc = json.loads(ledger.read_text()); assert sorted(doc["runs"]) == both
    ent = doc["runs"][both[0]]
    assert ent["rc"] == 1 and ent["daemon_sha"] == df[0]["daemon_sha"] and ent["store_generation"].startswith("/cluster_a/user/exp/stores/job7_") and ent["utc"] and "fake_driver.py" in ent["cmd"]
    tick1 = [e for e in ev if e["event"] == "tick"][-1]
    assert tick1["failed"] == 2 and tick1["quarantined"] == 0 and tick1["mirror_retired_dirs"] == 2 and tick1["problems"] == [] and rsync_lines() == 2
    # tick 2: the driver would succeed now, but the runs are held — no mirror, no dispatch, one problem, the summary says so
    fail_driver.unlink()
    r2 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path)); assert r2.returncode == 0, r2.stderr + r2.stdout
    ev2 = _events(tmp_path)[len(ev):]
    assert not [e for e in ev2 if e["event"] in ("dispatch", "dispatched", "mirror")] and rsync_lines() == 2
    prob = [e for e in ev2 if e["event"] == "problem"]
    assert len(prob) == 1 and prob[0]["kind"] == "dispatch_quarantine" and prob[0]["n"] == 2 and sorted(prob[0]["run_ids"]) == both and prob[0]["families"] == {"static": 2}
    tick2 = [e for e in ev2 if e["event"] == "tick"][-1]
    assert tick2["pending"] == 2 and tick2["quarantined"] == 2 and tick2["dispatched"] == 0 and tick2["failed"] == 0 and tick2["problems"] == ["dispatch_quarantine: 2 pending run(s) held after a failed dispatch"]
    # a new daemon code sha releases (the ledger's recorded sha is stale for one run): that run is mirrored, dispatched and graded; the other stays held
    doc = json.loads(ledger.read_text()); doc["runs"][both[0]]["daemon_sha"] = "0" * 40; ledger.write_text(json.dumps(doc))
    r3 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path)); assert r3.returncode == 0, r3.stderr + r3.stdout
    ev3 = _events(tmp_path)[len(ev) + len(ev2):]
    rel = [e for e in ev3 if e["event"] == "quarantine_released"]
    assert [(e["run_id"], e["reason"]) for e in rel] == [(both[0], "daemon_sha")] and rel[0]["was_daemon_sha"] == "0" * 40
    tick3 = [e for e in ev3 if e["event"] == "tick"][-1]
    assert tick3["quarantined"] == 1 and tick3["dispatched"] == 1 and rsync_lines() == 3 and sorted(json.loads(ledger.read_text())["runs"]) == [both[1]]
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (both[0],)).fetchone()[0] == "judged"
        assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (both[1],)).fetchone()[0] == "generated"
    # a changed store generation releases too (in process: the ledger records a stale generation; a dry run releases in memory and saves nothing)
    GD = _load_daemon_module()
    doc = json.loads(ledger.read_text()); doc["runs"][both[1]]["store_generation"] = "stale|0||"; ledger.write_text(json.dumps(doc))
    rows = GD.pending_rows(root, [synth.MODEL_ID], synth.CAMPAIGN, None, {synth.EVAL_ID: "static"})
    eligible, held = GD.quarantine_filter(str(state), root, rows, dry_run=True)
    assert [r["run_id"] for r in eligible] == [both[1]] and held == [] and sorted(json.loads(ledger.read_text())["runs"]) == [both[1]]
    with DB.reading(DB.db_path(root)) as conn:                                   # the true generation back: only the release file can free it now
        doc["runs"][both[1]]["store_generation"] = GD.store_generation(conn, both[1])
    ledger.write_text(json.dumps(doc))
    # the release file names the run: released, consumed, the run graded, the ledger empty
    (state / "dispatch_quarantine.release").write_text("# release\n" + both[1] + "\n")
    n_before = len(_events(tmp_path))                                              # the in-process dry run logged its own release event
    r4 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path)); assert r4.returncode == 0, r4.stderr + r4.stdout
    ev4 = _events(tmp_path)[n_before:]
    assert [(e["run_id"], e["reason"]) for e in ev4 if e["event"] == "quarantine_released"] == [(both[1], "release_file")]
    tick4 = [e for e in ev4 if e["event"] == "tick"][-1]
    assert tick4["quarantined"] == 0 and tick4["dispatched"] == 1 and tick4["problems"] == [] and not (state / "dispatch_quarantine.release").exists()
    assert json.loads(ledger.read_text())["runs"] == {} and rsync_lines() == 4
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (both[1],)).fetchone()[0] == "judged"


def test_a_lock_contention_failure_is_retryable_the_copy_is_kept_nothing_is_quarantined_and_the_next_tick_grades(tmp_path, monkeypatch):
    """The retryable / real split: a grader that lost the registry's write lock to the harvester (`LockHeld`) is not a failure of the run,
    the store or the grader — the `dispatched` record says `outcome: retryable` with the stderr tail, the mirrored copy is KEPT (queued
    for the prune, never retired), nothing is quarantined, and the next tick dispatches the same group and grades it. The grader's env
    carries the daemon's lock wait (SISTEER_REGISTRY_LOCK_TIMEOUT_S), long enough to outlast a harvester cycle."""
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    lockheld = tmp_path / "driver_lockheld"; lockheld.write_text(""); monkeypatch.setenv("FAKE_DRIVER_LOCKHELD_FILE", str(lockheld))
    monkeypatch.setenv("FAKE_DRIVER_ENV_RECORD", str(tmp_path / "driver_env.txt"))
    root, rids = _root(tmp_path)
    mirror = tmp_path / "mirror"; dirs = [mirror / "stores" / "job7_none", mirror / "stores" / "job7_steer"]
    r = _run_daemon(root, tmp_path, *_mirror_args(tmp_path), "--grader-lock-timeout-s", "240"); assert r.returncode == 0, r.stderr + r.stdout
    ev = _events(tmp_path)
    d = [e for e in ev if e["event"] == "dispatched"]
    assert len(d) == 1 and d[0]["rc"] == 1 and d[0]["outcome"] == "retryable" and "LockHeld: lock held:" in d[0]["stderr_tail"]
    assert [e["event"] for e in ev if e["event"] in ("dispatch_retryable", "dispatch_failed", "mirror_retired_incomplete", "problem")] == ["dispatch_retryable"]
    assert all(d_.is_dir() for d_ in dirs) and not (tmp_path / "state" / "dispatch_quarantine.json").exists()
    tick = [e for e in ev if e["event"] == "tick"][-1]
    assert tick["retryable"] == 2 and tick["failed"] == 0 and tick["dispatched"] == 0 and tick["quarantined"] == 0 and tick["mirror_retired_dirs"] == 0 and tick["mirror_prune_pending"] == 1
    # next tick: the lock is free (the fake consumed its flag), the same group is dispatched, graded and pruned; the ledger held ONE entry for
    # the copy (re-queued, never duplicated); the grader's env carried the daemon's lock wait
    r2 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path), "--grader-lock-timeout-s", "240"); assert r2.returncode == 0, r2.stderr + r2.stdout
    tick2 = [e for e in _events(tmp_path) if e["event"] == "tick"][-1]
    assert tick2["dispatched"] == 2 and tick2["retryable"] == 0 and tick2["mirror_pruned_dirs"] == 2 and tick2["mirror_prune_pending"] == 0 and not any(d_.exists() for d_ in dirs)
    assert (tmp_path / "driver_env.txt").read_text() == "240.0"
    ev_all = _events(tmp_path); (r_ev,) = [e for e in ev_all if e["event"] == "dispatch_retryable"]; assert r_ev["consecutive"] == 1 and r_ev["cap"] == GD_RETRYABLE_CAP
    assert json.loads((tmp_path / "state" / "dispatch_retries.json").read_text())["runs"] == {}       # the success cleared the ledger
    with DB.reading(DB.db_path(root)) as conn:
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rids[k],)).fetchone()[0] == "judged" for k in ("job7@none", "job7@steer"))


def test_a_dispatch_that_files_a_cell_is_progress_the_cap_charges_only_the_run_attempted_and_left_incomplete(tmp_path, monkeypatch):
    """The registry owner's ruling of 2026-10-13 23:29 (bon_unscored_investigation.md §4: the BoN grader aborts the dispatch at the first
    cell with an UNSCORED item, so nine dispatches of pure progress -- one cell filed each -- read as six consecutive retryable failures of
    the cells never reached, and the daemon quarantined the battery two cells short). A dispatch that files at least one complete
    judgment is progress: a run it filed RESETS its counter to 0, a run it never reached is NOT charged, only the run it attempted and
    left incomplete increments. A group of eight that files one cell per dispatch over seven dispatches never quarantines -- every
    `dispatch_retryable` event names the filed / attempted / unreached runs and `consecutive` 1, the ledger holds the one attempted run at
    1 and nothing else -- and the eighth dispatch files the last run clean, the ledger empty, all eight judged."""
    monkeypatch.setenv("FAKE_DRIVER_UNSCORED_AFTER_N", "1")
    root, rids = _root_many(tmp_path, 8)
    ledger = tmp_path / "state" / "dispatch_retries.json"
    for k in range(7):
        r = _run_daemon(root, tmp_path); assert r.returncode == 0, r.stderr + r.stdout
        ev = _events(tmp_path)
        tick = [e for e in ev if e["event"] == "tick"][-1]
        assert tick["quarantined"] == 0 and tick["failed"] == 0 and tick["retryable"] == 8 - k and tick["dispatched"] == 0, (k, tick)
        d = [e for e in ev if e["event"] == "dispatch"][-1]; assert d["n"] == 8 - k, (k, d)              # one group: the runs still pending
        rr = [e for e in ev if e["event"] == "dispatch_retryable"][-1]
        assert rr["consecutive"] == 1 and rr["cap"] == GD_RETRYABLE_CAP and "still UNSCORED after live grading (bounded retries)" in rr["terminal_exception"], rr
        assert (rr["filed"], rr["attempted"], rr["unreached"]) == ([rids[k]], [rids[k + 1]], rids[k + 2:]), (k, rr)
        assert json.loads(ledger.read_text())["runs"] == {rids[k + 1]: 1}, (k, ledger.read_text())      # the filed run reset, the unreached never charged
        with DB.reading(DB.db_path(root)) as conn:
            assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rids[k],)).fetchone()[0] == "judged"
    ev = _events(tmp_path)
    assert not [e for e in ev if e["event"] in ("dispatch_failed", "problem")] and not (tmp_path / "state" / "dispatch_quarantine.json").exists()
    r = _run_daemon(root, tmp_path); assert r.returncode == 0, r.stderr + r.stdout                        # the eighth: the last run alone, clean
    tick = [e for e in _events(tmp_path) if e["event"] == "tick"][-1]
    assert tick["dispatched"] == 1 and tick["retryable"] == 0 and tick["quarantined"] == 0 and tick["failed"] == 0, tick
    assert json.loads(ledger.read_text())["runs"] == {}
    with DB.reading(DB.db_path(root)) as conn:
        assert [conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] for rid in rids] == ["judged"] * 8


def test_a_run_attempted_and_left_incomplete_six_times_in_a_row_still_quarantines_and_the_runs_never_reached_are_not_charged(tmp_path, monkeypatch):
    """The other half of the ruling: RETRYABLE_CAP stays 6 for a run the dispatch attempted and did not complete. A group of three whose
    grader leaves the first run's items UNSCORED every time (a partial each dispatch, no complete judgment ever) and never reaches the other
    two: the first run's counter climbs 1..5 while the other two are never charged (absent from the ledger), and the sixth dispatch
    quarantines -- the cap line in the recorded tail, `dispatch_failed`, the quarantine ledger holding the group."""
    monkeypatch.setenv("FAKE_DRIVER_UNSCORED_AFTER_N", "0")
    root, rids = _root_many(tmp_path, 3)
    ledger = tmp_path / "state" / "dispatch_retries.json"
    for k in range(1, GD_RETRYABLE_CAP):
        r = _run_daemon(root, tmp_path); assert r.returncode == 0, r.stderr + r.stdout
        ev = _events(tmp_path)
        tick = [e for e in ev if e["event"] == "tick"][-1]
        assert tick["quarantined"] == 0 and tick["failed"] == 0 and tick["retryable"] == 3, (k, tick)
        rr = [e for e in ev if e["event"] == "dispatch_retryable"][-1]
        assert rr["consecutive"] == k and (rr["filed"], rr["attempted"], rr["unreached"]) == ([], [rids[0]], rids[1:]), (k, rr)
        assert json.loads(ledger.read_text())["runs"] == {rids[0]: k}, (k, ledger.read_text())
    assert not (tmp_path / "state" / "dispatch_quarantine.json").exists()
    n_before = len(_events(tmp_path))
    r = _run_daemon(root, tmp_path); assert r.returncode == 0, r.stderr + r.stdout                        # the sixth attempt: the cap
    ev = _events(tmp_path)[n_before:]
    (d,) = [e for e in ev if e["event"] == "dispatched"]
    assert d["outcome"] == "failed" and f"{GD_RETRYABLE_CAP} consecutive retryable failures (cap {GD_RETRYABLE_CAP})" in d["stderr_tail"], d
    assert [e["event"] for e in ev if e["event"] in ("dispatch_retryable", "dispatch_failed")] == ["dispatch_failed"]
    tick = [e for e in ev if e["event"] == "tick"][-1]; assert tick["failed"] == 3 and tick["retryable"] == 0, tick
    assert json.loads(ledger.read_text())["runs"] == {rids[0]: GD_RETRYABLE_CAP}
    assert sorted(json.loads((tmp_path / "state" / "dispatch_quarantine.json").read_text())["runs"]) == rids
    with DB.reading(DB.db_path(root)) as conn:
        assert [conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] for rid in rids] != ["judged"] * 3


def test_a_claim_left_by_the_daemons_own_dead_grader_is_re_dispatched_and_another_sessions_claim_is_left_alone(tmp_path, monkeypatch):
    """A grader that died between its claim and its filing (the LockHeld wave of 2026-09-30 left 16 runs `judging`) held its own runs out of
    the worklist for `stale_hours`. Rows claimed by the daemon's own graders' session are dispatchable again (`reclaim_own_session`; the
    grader's claim returns the same judgment id); a row another session holds is still left to it."""
    from sisteer.registry import pending as P
    root, rids = _root(tmp_path)
    with DB.reading(DB.db_path(root)) as conn:
        judge = conn.execute("SELECT required_judge_id FROM v_pending_judgment WHERE run_id = ?", (rids["job7@none"],)).fetchone()[0]
    own = "fake-driver"                                                # what the fake driver claims under (see _daemon_argv_env)
    P.claim(root, rids["job7@none"], judge, own)                      # our grader's claim, its process gone
    P.claim(root, rids["job7@steer"], judge, "someone-else@TESTHOST")  # another live session's
    r = _run_daemon(root, tmp_path); assert r.returncode == 0, r.stderr + r.stdout
    ev = _events(tmp_path)
    (rc,) = [e for e in ev if e["event"] == "reclaim_own_session"]; assert rc["run_ids"] == [rids["job7@none"]] and rc["session"] == own
    (d,) = [e for e in ev if e["event"] == "dispatch"]; assert d["run_ids"] == [rids["job7@none"]]
    tick = [e for e in ev if e["event"] == "tick"][-1]; assert tick["reclaimed"] == 1 and tick["dispatched"] == 1 and tick["pending"] == 1
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rids["job7@none"],)).fetchone()[0] == "judged"
        assert tuple(conn.execute("SELECT lifecycle_state, claim_session FROM v_pending_judgment WHERE run_id = ?", (rids["job7@steer"],)).fetchone()) == ("judging", "someone-else@TESTHOST")


def test_dispatch_failure_kind_reads_the_terminal_exception_not_substrings_anywhere():
    """Only a grader whose TERMINAL exception is the registry lock contention is retryable; an early LockHeld
    followed by a later fatal error, or an unrelated EAGAIN, is a defect."""
    GD = _load_daemon_module()
    lockheld = "Traceback (most recent call last):\n  File x\n    fcntl.flock(...)\nBlockingIOError: [Errno 11] Resource temporarily unavailable\n\nDuring handling...\n" \
               "Traceback (most recent call last):\n  File pending.py, line 147, in claim\nsisteer.registry.db.LockHeld: lock held: /reg/harvester.lock\n"
    assert GD.dispatch_failure_kind(1, lockheld) == "retryable" and GD.terminal_exception_line(lockheld).startswith("sisteer.registry.db.LockHeld:")
    later_fatal = lockheld + "[grader] retrying...\nTraceback (most recent call last):\n  File grader.py\nKeyError: 'store'\n"
    assert GD.dispatch_failure_kind(1, later_fatal) == "failed"
    eagain_only = "Traceback (most recent call last):\n  File x\nBlockingIOError: [Errno 11] Resource temporarily unavailable\n"
    assert GD.dispatch_failure_kind(1, eagain_only) == "failed"
    jb1 = "[grade-live:sr] 500/771 graded\n[k] BoN: 12 judge item(s) still UNSCORED after live grading (bounded retries) — refusing to aggregate a partial grade (JB1). First: dataset=jbb\n"
    assert GD.dispatch_failure_kind(1, jb1) == "retryable"                          # the live judge's pool shed, not a defect
    assert GD.dispatch_failure_kind(1, jb1 + "KeyError: 'x'\n") == "failed"        # ...unless something else is the terminal line
    # the registry's SCHEMA refusal at the grader's claim (pending.claim -> db.open_or_create raising db.SchemaDrift) is a kind of its
    # own -- `schema_drift`: held transient, never quarantined -- read from the terminal line by the exception's qualified name; the versions come
    # from db.drift_message's fixed opening clause (the REAL text, never re-typed)
    behind = ("Traceback (most recent call last):\n  File pending.py, line 188, in claim\n    conn = DB.open_or_create(DB.db_path(root))\n"
              f"sisteer.registry.db.SchemaDrift: {DB.drift_message('/reg/registry.sqlite', 21, 22, [])}\n")
    assert GD.dispatch_failure_kind(1, behind) == "schema_drift"
    assert GD.schema_gap_of_terminal(GD.terminal_exception_line(behind)) == {"have": 21, "want": 22, "shape_only": False}
    shape = f"sisteer.registry.db.SchemaDrift: {DB.drift_message('/reg/registry.sqlite', 22, 22, ['index ix_x is absent'])}\n"
    assert GD.dispatch_failure_kind(1, shape) == "schema_drift"
    assert GD.schema_gap_of_terminal(GD.terminal_exception_line(shape)) == {"have": 22, "want": 22, "shape_only": True}
    assert GD.dispatch_failure_kind(1, behind + "KeyError: 'x'\n") == "failed"     # ...unless something else is the terminal line
    assert GD.dispatch_failure_kind(1, "the message mentions SchemaDrift but the terminal line is\nKeyError: 'x'\n") == "failed"
    assert GD.schema_gap_of_terminal("sisteer.registry.db.SchemaDrift: some other text") == {"have": None, "want": None, "shape_only": None}
    # the retry ledger: consecutive retryable failures per run, cleared by any other outcome, the group's max returned
    st = tmp_path_factory = None
    import tempfile
    d = tempfile.mkdtemp()
    assert [GD.retryable_count(d, ["r1", "r2"], failed=True) for _ in range(3)] == [1, 2, 3]
    assert GD.retryable_count(d, ["r1"], failed=False) == 0 and GD.retryable_count(d, ["r2"], failed=True) == 4 and GD.retryable_count(d, ["r1"], failed=True) == 1
    assert GD.dispatch_failure_kind(0, "lock held: whatever") == "ok" and GD.dispatch_failure_kind(1, "") == "failed"


# the real terminal shapes of the demo smoke's 401 incident (2026-10-11), re-typed with the text withheld
_TAIL_401_CONTROLS = ("[served-judge 08:10:09] judging on org=lowprio base_url=https://api.anthropic.com gate=200 cell_parallel=6\n"
                      "Traceback (most recent call last):\n  File judging.py, line 1868, in judge_all\n    raise JudgingError(\n"
                      "sisteer.capeval.judging.JudgingError: 450 of 450 judge request(s) are still unverdicted after retries — refusing a partial grade. Failure types: "
                      "{\"AuthenticationError: Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'invalid x-api-key'}, 'request_\": 450} "
                      "(prompt/response text withheld).\n[served-judge 08:10:28] JUDGE_FAILED: 3 cell(s) failed, 0 done\n")
_TAIL_UNSCORED_BON = ("[grade-live:sr2] 7500/7500 graded (7500 unscored so far) [429s: 747, in-flight now 50]\n"
                      "[qwen35_4b_demo_battery] BoN: 7500 judge item(s) still UNSCORED after live grading (bounded retries) — refusing to aggregate a partial grade (JB1). "
                      "First: dataset=jbb revision=886acc352a31 (goal/response text withheld). — each judgment had its 2 attempts (plan §9.6); no automatic re-issue; "
                      "the successful verdicts are cached and a re-run resumes them; judge client counts: {'judgments': 7500, 'attempts': 8246, 'rate_limited': 747, "
                      "'transient': 6, 'terminal': 7493, 'exhausted': 7}\n")


def test_a_credential_rejection_is_its_own_kind_and_an_unscored_line_whose_counts_say_terminal_is_failed():
    """(a) A tail carrying the API's credential rejection -- 401
    authentication_error / invalid x-api-key / AuthenticationError, 403 PermissionDeniedError -- is `credential`, wherever in the tail the
    grader printed it (the one deliberate exception to the terminal-line rule: a 401 anywhere in a grader's output is never benign), ahead of
    every other kind; a 400 / 429 is not. (b) The JB1 UNSCORED line is retryable only as shedding: one whose embedded judge-client counts say
    terminal > rate_limited (the BoN grader's real line: 7,493 terminal against 747 rate-limited) or whose failure-types histogram names a
    4xx other than 429 is `failed`; the bare line and one whose counts say rate_limited >= terminal stay `retryable`."""
    GD = _load_daemon_module()
    assert GD.dispatch_failure_kind(1, _TAIL_401_CONTROLS) == "credential" and GD.credential_rejected(_TAIL_401_CONTROLS)
    assert GD.credential_org_of_tail(_TAIL_401_CONTROLS) == ("lowprio", True)
    assert GD.dispatch_failure_kind(1, "Traceback...\nanthropic.PermissionDeniedError: Error code: 403 - {'type': 'error', 'error': {'type': 'permission_error'}}\n") == "credential"
    assert GD.dispatch_failure_kind(1, "[judge] item 12: AuthenticationError: Error code: 401\n[judge] done\nKeyError: 'x'\n") == "credential"   # earlier in the tail, another terminal line
    assert GD.dispatch_failure_kind(1, "x\nsisteer.registry.db.LockHeld: lock held: /reg/harvester.lock\n") == "retryable"                      # untouched kinds
    assert GD.dispatch_failure_kind(1, "anthropic.BadRequestError: Error code: 400 - {'type': 'invalid_request_error'}\n") == "failed"
    assert GD.dispatch_failure_kind(1, "anthropic.RateLimitError: Error code: 429 - {'type': 'rate_limit_error'}\n") == "failed"
    assert GD.dispatch_failure_kind(0, _TAIL_401_CONTROLS) == "ok"
    assert GD.credential_org_of_tail("[judge] done: {...}; pool highprio -> highprio; provenance x.json\n") == ("highprio", True)
    assert GD.credential_org_of_tail("'api_pool': {'pool': 'batch', 'auth': 'wif'}") == ("batch", True)
    assert GD.credential_org_of_tail("no org here") == ("lowprio", False)
    # (b)
    jb1 = "[grade-live:sr] 500/771 graded\n[k] BoN: 12 judge item(s) still UNSCORED after live grading (bounded retries) — refusing to aggregate a partial grade (JB1). First: dataset=jbb\n"
    assert GD.dispatch_failure_kind(1, jb1) == "retryable" and not GD.unscored_line_is_terminal(GD.terminal_exception_line(jb1))
    assert GD.dispatch_failure_kind(1, _TAIL_UNSCORED_BON) == "failed" and GD.unscored_line_is_terminal(GD.terminal_exception_line(_TAIL_UNSCORED_BON))
    shed = _TAIL_UNSCORED_BON.replace("'rate_limited': 747", "'rate_limited': 7500").replace("'terminal': 7493", "'terminal': 6")
    assert GD.dispatch_failure_kind(1, shed) == "retryable"                                                                                   # the counts say shedding
    hist = jb1.rstrip("\n") + " Failure types: {\"BadRequestError: Error code: 400 - ...\": 12}\n"
    assert GD.dispatch_failure_kind(1, hist) == "failed"
    hist429 = jb1.rstrip("\n") + " Failure types: {\"RateLimitError: Error code: 429 - ...\": 12}\n"
    assert GD.dispatch_failure_kind(1, hist429) == "retryable"


def test_a_federated_credential_rejection_is_the_credential_kind_whatever_the_token_endpoints_body_shape(monkeypatch):
    """`CREDENTIAL_TAIL_RE` read the keyed SDK class names, `authentication_error` and `Error code: 401`,
    and the FEDERATED rejection carries none of them -- under wif the token exchange refuses the identity and the SDK raises
    `WorkloadIdentityError(status_code=401 | 403)` whose redacted body is an OAuth error string or a non-JSON text, so the typed
    `JudgeCredentialRejected` reads `WorkloadIdentityError (HTTP 401)` with no API error type; the daemon classified such a grader `failed`
    and quarantined its runs. Now the classifier reads the typed rejection's own clause (`rejected the <pool> pool's credential`), the
    `(HTTP 401 / 403)` the typed message and the SDK's own `Token exchange failed (HTTP 40x)` line carry, and the org off the typed
    clause. NOT this: the exchange's 5xx, an unreachable token endpoint (the class name alone never classifies), a 429."""
    from sisteer import vault
    GD = _load_daemon_module()
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "wif")
    typed = {}
    for pool, status in (("lowprio", 401), ("highprio", 403), ("batch", 401)):
        typed[pool] = str(vault.JudgeCredentialRejected(pool, status, "WorkloadIdentityError", api_error_type=None))   # the REAL text
        assert "authentication_error" not in typed[pool] and "AuthenticationError" not in typed[pool] and "Error code" not in typed[pool]
    # judge_sonnet's terminal shape (SystemExit over the typed message), the misalign judge's [pool] line, ReNeLLM's, the sampled grader's
    for tail in ("[judge] CREDENTIAL REJECTED on lowprio at i/m arm a: " + typed["lowprio"] + " -- 70 of this arm's 72 requests were never issued\n[judge] " + typed["lowprio"] + "\n",
                 "[pool] CREDENTIAL REJECTED on highprio: " + typed["highprio"] + " (D6)\nSystemExit: 3\n",
                 "  corpus: CREDENTIAL REJECTED after 2/30 verdicts: " + typed["batch"] + " -- the remaining 28 request(s) are NOT issued (D6)\n[judge] " + typed["batch"] + "\n",
                 "[sampled-grade] CREDENTIAL REJECTED: " + typed["lowprio"] + "\n"):
        assert GD.dispatch_failure_kind(1, tail) == "credential" and GD.credential_rejected(tail), tail
    assert GD.credential_org_of_tail("[judge] " + typed["lowprio"]) == ("lowprio", True)
    assert GD.credential_org_of_tail("[pool] CREDENTIAL REJECTED on highprio: " + typed["highprio"]) == ("highprio", True)
    assert GD.credential_org_of_tail(typed["batch"]) == ("batch", True)
    # a grader that let the SDK's own error through (no typed wrapper): the exchange's 401 / 403 line, JSON or not
    sdk = "anthropic.lib.credentials._workload.WorkloadIdentityError: Token exchange failed (HTTP {s}): {b} Ensure your federation rule matches your identity token.\n"
    for s_, b in ((401, "{'error': 'invalid_grant', 'error_description': 'synthetic'}"), (403, "synthetic non-json"), (401, "{'type': 'error', 'error': {'type': 'authentication_error'}}")):
        assert GD.dispatch_failure_kind(1, "Traceback (most recent call last):\n  File x\n" + sdk.format(s=s_, b=b)) == "credential", (s_, b)
    # NOT this
    for tail in ("anthropic.lib.credentials._workload.WorkloadIdentityError: Failed to reach token endpoint https://api.anthropic.com/v1/oauth/token: synthetic\n",
                 sdk.format(s=500, b="synthetic"), sdk.format(s=529, b="synthetic"),
                 "anthropic.lib.credentials._workload.WorkloadIdentityError: Token endpoint returned non-JSON response (status 502): synthetic\n",
                 "[judge] body 529'd 3x while 2 sibling request(s) succeeded\nRuntimeError: judge failed after retries (last: OverloadedError/529)\n",
                 "the HTTP 4011 gateway\nKeyError: 'x'\n"):
        assert GD.dispatch_failure_kind(1, tail) == "failed", tail
    assert GD.dispatch_failure_kind(0, "[judge] " + typed["lowprio"]) == "ok"
    # the legacy shapes still classify (the regex grew; nothing it read before is lost)
    assert GD.dispatch_failure_kind(1, _TAIL_401_CONTROLS) == "credential"
    assert GD.dispatch_failure_kind(1, "anthropic.PermissionDeniedError: Error code: 403 - {'type': 'error', 'error': {'type': 'permission_error'}}\n") == "credential"


@pytest.mark.parametrize("shape", ["key-401-controls-judge", "wif-exchange-401-judge-sonnet"])
def test_a_credential_rejection_holds_the_family_raises_one_problem_and_quarantines_nothing(tmp_path, monkeypatch, shape):
    """End to end: a grader whose judge requests the API rejected (401) ends the tick with its FAMILY held -- `dispatch.hold.<family>` written
    by the daemon with the reason naming the org, the mode and the log -- ONE `problem` of kind `judge_auth`, the `dispatched` record saying
    `outcome: credential`, the mirrored copy KEPT, no quarantine ledger, no retry count (nothing toward RETRYABLE_CAP) and `credential` in the
    tick summary; the next tick skips the family (`held_family`, the operator-hold path) without a second problem; a standing hold is never
    overwritten; removing the file re-dispatches the family and the credential-fixed grader grades.

    Two tail shapes (Major): the legacy key's (the controls judge's JudgingError histogram naming
    `AuthenticationError: Error code: 401 ... authentication_error`) and the FEDERATED one (judge_sonnet under wif: the token exchange refused
    the identity with an OAuth error-string body, so the typed message reads `WorkloadIdentityError (HTTP 401)` and carries none of the
    legacy words) -- before the fix the second was `failed`, both runs QUARANTINED, and fixing the identity released nothing."""
    wif = shape.startswith("wif")
    mode = "wif" if wif else "op"
    fix_hint = "the federation ids file / the VM identity" if wif else "the vault item"
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    rejected = tmp_path / "driver_auth401"; rejected.write_text("")
    monkeypatch.setenv("FAKE_DRIVER_AUTH401_WIF_FILE" if wif else "FAKE_DRIVER_AUTH401_FILE", str(rejected))
    root, rids = _root(tmp_path)
    both = sorted(rids[k] for k in ("job7@none", "job7@steer"))
    state = tmp_path / "state"; hold = state / "dispatch.hold.static"
    mirror = tmp_path / "mirror"; dirs = [mirror / "stores" / "job7_none", mirror / "stores" / "job7_steer"]
    r = _run_daemon(root, tmp_path, *_mirror_args(tmp_path), env_extra={"SISTEER_SECRETS_MODE": mode}); assert r.returncode == 0, r.stderr + r.stdout
    ev = _events(tmp_path)
    (d,) = [e for e in ev if e["event"] == "dispatched"]
    assert d["rc"] == 1 and d["outcome"] == "credential"
    if wif:
        assert "WorkloadIdentityError (HTTP 401)" in d["stderr_tail"] and "authentication_error" not in d["stderr_tail"] \
            and "AuthenticationError" not in d["stderr_tail"] and "Error code: 401" not in d["stderr_tail"]     # none of the legacy words
    else:
        assert "invalid x-api-key" in d["stderr_tail"]
    (c,) = [e for e in ev if e["event"] == "dispatch_credential"]
    assert sorted(c["run_ids"]) == both and c["org"] == "lowprio" and c["org_named_by_grader"] is True and c["secrets_mode"] == mode
    assert c["hold_file"] == str(hold) and c["hold_written"] is True
    assert c["terminal_exception"].startswith("[judge] the API rejected the lowprio pool's credential: WorkloadIdentityError (HTTP 401)" if wif
                                              else "[served-judge 08:10:28] JUDGE_FAILED")
    prob = [e for e in ev if e["event"] == "problem"]
    assert len(prob) == 1 and prob[0]["kind"] == "judge_auth" and prob[0]["family"] == "static" and prob[0]["org"] == "lowprio" and sorted(prob[0]["run_ids"]) == both
    assert prob[0]["hold_file"] == str(hold) and prob[0]["secrets_mode"] == mode
    assert not [e for e in ev if e["event"] in ("dispatch_failed", "dispatch_retryable", "quarantine_added", "mirror_retired_incomplete")]
    assert hold.is_file() and not (state / "dispatch.hold.static.tmp").exists()
    reason = [ln for ln in hold.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    assert len(reason) == 1 and reason[0].startswith(f"credential: the API rejected the static grader's credential (401 / 403) for org lowprio under SISTEER_SECRETS_MODE={mode}")
    assert fix_hint in reason[0] and "remove this file" in reason[0] and d["log"] in reason[0]
    assert ("WorkloadIdentityError (HTTP 401)" if wif else "JUDGE_FAILED") in reason[0]
    tick = [e for e in ev if e["event"] == "tick"][-1]
    assert tick["credential"] == 2 and tick["failed"] == 0 and tick["retryable"] == 0 and tick["quarantined"] == 0 and tick["dispatched"] == 0
    assert tick["problems"] == [f"judge_auth: the API rejected the static grader's credential (401 / 403) for org lowprio under SISTEER_SECRETS_MODE={mode}: family held ({hold}); "
                                "fix the credential and remove the file"]
    assert all(d_.is_dir() for d_ in dirs) and tick["mirror_retired_dirs"] == 0 and tick["mirror_prune_pending"] == 1      # the copy is kept, as a retryable's
    assert not (state / "dispatch_quarantine.json").exists()
    assert json.loads((state / "dispatch_retries.json").read_text())["runs"] == {}                                        # nothing toward RETRYABLE_CAP
    # tick 2: the family is held (the operator-hold path) -- nothing mirrored or dispatched, no second judge_auth problem, the hold file untouched
    before = hold.read_text()
    r2 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path), env_extra={"SISTEER_SECRETS_MODE": mode}); assert r2.returncode == 0, r2.stderr + r2.stdout
    ev2 = _events(tmp_path)[len(ev):]
    assert not [e for e in ev2 if e["event"] in ("dispatch", "dispatched", "mirror", "dispatch_credential")]
    assert [e["kind"] for e in ev2 if e["event"] == "problem"] == [] and hold.read_text() == before
    (hf,) = [e for e in ev2 if e["event"] == "held_family"]; assert hf["family"] == "static" and hf["reason"] == reason[0] and hf["n_runs"] == 2
    tick2 = [e for e in ev2 if e["event"] == "tick"][-1]
    assert tick2["credential"] == 0 and tick2["dispatched"] == 0 and tick2["held_families"] == ["static"] and tick2["quarantined"] == 0
    # the hold removed, the credential fixed (the fake grades): the family dispatches and grades; the ledgers stay empty
    hold.unlink(); rejected.unlink()
    r3 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path), env_extra={"SISTEER_SECRETS_MODE": mode}); assert r3.returncode == 0, r3.stderr + r3.stdout
    ev3 = _events(tmp_path)[len(ev) + len(ev2):]
    tick3 = [e for e in ev3 if e["event"] == "tick"][-1]
    assert tick3["dispatched"] == 2 and tick3["credential"] == 0 and tick3["problems"] == [] and not hold.exists()
    assert not [e for e in ev3 if e["event"] in ("held_family", "dispatch_credential", "problem")]      # (each --once tick is a fresh process: no `released` memory)
    with DB.reading(DB.db_path(root)) as conn:
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "judged" for rid in both)
    # the writer never overwrites a standing hold (the operator's, or an earlier event's)
    GD = _load_daemon_module()
    hold.write_text("# mine\noperator: keep static held\n")
    assert GD.family_hold_write(str(state), "static", "credential: ...", writer="credential") == (str(hold), False)
    assert GD.family_holds(str(state)) == {"static": "operator: keep static held"}
    assert GD.family_hold_write(str(state), "bon", "credential: the API rejected ...\nsecond line", writer="credential") == (str(state / "dispatch.hold.bon"), True)
    assert GD.family_holds(str(state))["bon"] == "credential: the API rejected ... second line"
    assert GD.family_hold_write(None, "bon", "x", writer="credential") == (None, False)


def test_a_re_pointed_checkout_is_seen_per_tick_and_the_daemon_re_execs_on_the_new_code(tmp_path):
    """The code sha is read fresh before every tick. The daemon runs from a throwaway git checkout of its own two
    files; after its first tick the checkout's HEAD moves (a landing) — the next iteration logs `code_drift` and re-executes itself, and
    the following tick carries the new sha (a `--once` run only reports)."""
    import shutil, time as _t
    co = tmp_path / "co" / "scripts" / "registry"; co.mkdir(parents=True)
    for f in ("grade_daemon.py", "emit_contrasts.py"):
        shutil.copy(REPO / "scripts" / "registry" / f, co / f)
    repo = tmp_path / "co"
    def git(*a):
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True, env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@TESTHOST", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@TESTHOST"))
    git("init", "-q"); git("add", "."); git("commit", "-q", "-m", "one")
    sha1 = git("rev-parse", "HEAD").stdout.strip()
    root, _rids = _root(tmp_path)
    state = tmp_path / "state"
    env = dict(os.environ, PYTHONPATH=str(REPO / "src")); env.pop(emit.DISABLED_ENV, None)
    argv = [sys.executable, str(co / "grade_daemon.py"), "--registry-root", root, "--model-id", "nobody", "--campaign", synth.CAMPAIGN, "--interval", "1",
            "--python", sys.executable, "--repo", str(REPO), "--configs-root", str(REPO / "configs"), "--state-dir", str(state), "--max-ticks", "3", "--driver", "static=true"]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=str(REPO))
    try:
        end = _t.time() + 60
        while _t.time() < end and not any(e["event"] == "tick" for e in _events(tmp_path)):
            _t.sleep(0.2)
        first = [e for e in _events(tmp_path) if e["event"] == "tick"]
        assert first and first[0]["daemon_sha"] == sha1, (proc.poll(), proc.stderr.read() if proc.poll() is not None else "still running")
        git("commit", "-q", "--allow-empty", "-m", "two"); sha2 = git("rev-parse", "HEAD").stdout.strip(); assert sha2 != sha1
        out, err = proc.communicate(timeout=150)
    finally:
        if proc.poll() is None:
            proc.kill()
    ev = _events(tmp_path)
    drift = [e for e in ev if e["event"] == "code_drift"]
    assert drift and drift[0]["imported_sha"] == sha1 and drift[0]["current_sha"] == sha2 and drift[0]["action"] == "re-exec on the new code", err
    started = [e["daemon_sha"] for e in ev if e["event"] == "daemon_started"]; assert started[:2] == [sha1, sha2], (started, err)
    ticks = [e["daemon_sha"] for e in ev if e["event"] == "tick"]; assert ticks[0] == sha1 and sha2 in ticks, (ticks, err)


def test_a_failure_under_a_grader_that_outlived_a_re_point_is_stamped_with_the_code_that_ran_it_and_released_by_the_re_exec(tmp_path):
    """A grader holds the tick while the checkout is re-pointed A -> B; it then fails. The quarantine is stamped with
    A (the code that dispatched and classified it), the daemon re-execs on B at its next loop, and B releases the runs (reason daemon_sha) —
    instead of stamping B and holding them for a fix they never saw."""
    import shutil, time as _t
    co = tmp_path / "co" / "scripts" / "registry"; co.mkdir(parents=True)
    for f in ("grade_daemon.py", "emit_contrasts.py"):
        shutil.copy(REPO / "scripts" / "registry" / f, co / f)
    repo = tmp_path / "co"
    def git(*a):
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True, env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@TESTHOST", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@TESTHOST"))
    git("init", "-q"); git("add", "."); git("commit", "-q", "-m", "A"); sha_a = git("rev-parse", "HEAD").stdout.strip()
    root, rids = _root(tmp_path)
    hold = tmp_path / "hold"; hold.write_text(""); fail_after = tmp_path / "fail_after"; fail_after.write_text("")
    fake = tmp_path / "fake_driver.py"; fake.write_text(FAKE_DRIVER); record = tmp_path / "driver_calls.jsonl"
    template = f"{{python}} {fake} --worklist registry --registry-root {{registry_root}} {{run_id_flags}} {{store_map_flags}} --record {record} {{extra}}"
    pidf = tmp_path / "grader.pid"                                                    # written by the fake BEFORE it holds: "the grader is up"
    env = dict(os.environ, SISTEER_SRC=str(REPO / "src"), PYTHONPATH=str(REPO / "src"), REGISTRY_ROOT=root, FAKE_DRIVER_HOLD_FILE=str(hold),
               FAKE_DRIVER_FAIL_AFTER_HOLD_FILE=str(fail_after), FAKE_DRIVER_PID_FILE=str(pidf))
    env.pop(emit.DISABLED_ENV, None)
    state = tmp_path / "state"
    argv = [sys.executable, str(co / "grade_daemon.py"), "--registry-root", root, "--model-id", synth.MODEL_ID, "--campaign", synth.CAMPAIGN, "--interval", "1",
            "--python", sys.executable, "--repo", str(REPO), "--configs-root", str(REPO / "configs"), "--state-dir", str(state), "--max-ticks", "2",
            "--store-root-map", f"cluster_a=/cluster_a/user/exp={tmp_path / 'mirror'}", "--driver", f"static={template}", "--extra", "static=--target {model_key}",
            "--eval-family", f"{synth.EVAL_ID}=static", "--grader-session", "fake-driver", "--mirror-min-free-bytes", str(1024 * 1024), "--spool", os.path.join(root, "records", "contrasts")]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=str(REPO))
    try:
        end = _t.time() + 60
        while _t.time() < end and not pidf.exists():                                # the grader is up and holding
            _t.sleep(0.2)
        assert pidf.exists(), (proc.poll(), proc.stderr.read() if proc.poll() is not None else "no grader yet")
        git("commit", "-q", "--allow-empty", "-m", "B"); sha_b = git("rev-parse", "HEAD").stdout.strip()   # re-pointed while the grader runs
        hold.unlink()                                                                # the grader proceeds and dies
        out, err = proc.communicate(timeout=180)
    finally:
        if proc.poll() is None:
            proc.kill()
    ev = _events(tmp_path)
    # the failure under the held grader is stamped A; after the re-exec on B the released runs are re-dispatched, the fixture fails again
    # (its fail-after-hold flag still stands) and THAT failure is stamped B — each failure carries the code that ran it
    df = [e for e in ev if e["event"] == "dispatch_failed"]; assert df and df[0]["daemon_sha"] == sha_a, (df, err)
    assert all(e["daemon_sha"] == sha_b for e in df[1:]), (df, err)
    ledger_stamped = [e for e in ev if e["event"] == "tick"][0]["daemon_sha"]; assert ledger_stamped == sha_a
    drift = [e for e in ev if e["event"] == "code_drift"]; assert drift and drift[0]["imported_sha"] == sha_a and drift[0]["current_sha"] == sha_b, err
    rel = [e for e in ev if e["event"] == "quarantine_released"]
    assert sorted(e["run_id"] for e in rel) == sorted(rids[k] for k in ("job7@none", "job7@steer")) and {e["reason"] for e in rel} == {"daemon_sha"} and {e["daemon_sha"] for e in rel} == {sha_b}, (rel, err)
    assert [e["daemon_sha"] for e in ev if e["event"] == "daemon_started"][:2] == [sha_a, sha_b]


def test_a_clean_grader_whose_judgments_are_only_spooled_is_held_from_re_dispatch_until_the_fold_shows(tmp_path, monkeypatch):
    """Under the harvester-folds contract, a grader that exits clean has its judgments in the SPOOL;
    until the harvester folds them the run still reads pending — its own-session claim would even be reclaimed — and the next tick would
    re-dispatch it (paid judging again, rsync again, duplicate records). The daemon writes a durable spooled hold on a clean dispatch and
    honours it: two ticks before any ingest dispatch the group ONCE; the copies are queued, never retired; after the harvester's fold the
    hold clears and the copies are pruned. A hold whose run has no complete judgment in its spool directory is dropped (reclaimable)."""
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log")); monkeypatch.setenv("FAKE_DRIVER_NO_FOLD", "1")
    root, rids = _root(tmp_path)
    mirror = tmp_path / "mirror"; dirs = [mirror / "stores" / "job7_none", mirror / "stores" / "job7_steer"]
    both = sorted(rids[k] for k in ("job7@none", "job7@steer"))
    r = _run_daemon(root, tmp_path, *_mirror_args(tmp_path)); assert r.returncode == 0, r.stderr + r.stdout
    ev = _events(tmp_path); tick = [e for e in ev if e["event"] == "tick"][-1]
    assert tick["dispatched"] == 2 and tick["failed"] == 0 and tick["mirror_retired_dirs"] == 0 and tick["mirror_pruned_dirs"] == 0 and tick["fold_mode"] == "harvester"
    (sp,) = [e for e in ev if e["event"] == "dispatch_spooled"]; assert sorted(sp["run_ids"]) == both
    assert sorted(json.loads((tmp_path / "state" / "dispatch_spooled.json").read_text())["runs"]) == both and all(d.is_dir() for d in dirs)
    with DB.reading(DB.db_path(root)) as conn:                                     # spooled, not folded: still judging (claimed)
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "judging" for rid in both)
    # tick 2, still no ingest: held — no dispatch, no mirror, the copies untouched
    r2 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path)); assert r2.returncode == 0, r2.stderr + r2.stdout
    ev2 = _events(tmp_path)[len(ev):]
    assert not [e for e in ev2 if e["event"] in ("dispatch", "dispatched", "mirror", "reclaim_own_session")], ev2
    (hold,) = [e for e in ev2 if e["event"] == "spooled_hold"]; assert sorted(hold["run_ids"]) == both
    tick2 = [e for e in ev2 if e["event"] == "tick"][-1]; assert tick2["pending"] == 2 and tick2["spooled_hold"] == 2 and tick2["dispatched"] == 0 and tick2["mirror_pruned_dirs"] == 0
    assert len((tmp_path / "rsync_calls.log").read_text().splitlines()) == 2 and all(d.is_dir() for d in dirs)
    # the harvester folds: the hold clears, the copies are pruned, the runs are judged
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    r3 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path)); assert r3.returncode == 0, r3.stderr + r3.stdout
    ev3 = _events(tmp_path)[len(ev) + len(ev2):]
    assert sorted(e["run_id"] for e in ev3 if e["event"] == "spooled_hold_cleared") == both
    tick3 = [e for e in ev3 if e["event"] == "tick"][-1]
    assert tick3["pending"] == 0 and tick3["spooled_hold"] == 0 and tick3["mirror_pruned_dirs"] == 2 and not any(d.exists() for d in dirs)
    assert json.loads((tmp_path / "state" / "dispatch_spooled.json").read_text())["runs"] == {}
    with DB.reading(DB.db_path(root)) as conn:
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] in ("judged", "ingested") for rid in both)
    # the reclaim path stands: a hold whose spool carries no complete judgment (the grader died before filing) is dropped and the run re-dispatched
    GD = _load_daemon_module()
    st = tmp_path / "state"
    GD.spooled_save(str(st), {rids["job8@other"]: {"run_id": rids["job8@other"], "family": "static", "campaign": synth.CAMPAIGN, "spooled_utc": "2026-01-01T00:00:00Z"}})
    rows = GD.pending_rows(root, [OTHER_MODEL], synth.CAMPAIGN, None, {synth.EVAL_ID: "static"})
    eligible, held = GD.spooled_filter(str(st), root, rows)
    assert [r["run_id"] for r in eligible] == [rids["job8@other"]] and held == [] and json.loads((st / "dispatch_spooled.json").read_text())["runs"] == {}
    assert [e for e in _events(tmp_path) if e["event"] == "spooled_hold_dropped"][-1]["run_id"] == rids["job8@other"]


def test_the_spooled_hold_is_bound_to_its_dispatchs_judgments_and_store_generation(tmp_path, monkeypatch):
    """The hold's continued proof was ANY historical complete judgment in the run's spool directory and its only clear
    was `not run_pending` — a grader filing complete with 150 of 200 items (coverage_short after the fold), or a regenerated store beside an
    old complete file, held the run for ever. Now the hold names THIS dispatch's judgment ids and the store generation it graded: a bound
    judgment that FOLDS while the run still pends releases the hold and QUARANTINES the run under a named coverage defect (no blind
    re-dispatch); a changed generation clears it; a bound judgment gone from the spool drops it; a historical complete judgment that is not
    this dispatch's holds nothing; the operator's release file drops it."""
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log")); monkeypatch.setenv("FAKE_DRIVER_NO_FOLD", "1"); monkeypatch.setenv("FAKE_DRIVER_SHORT_ITEMS", "150")
    root, rids = _root(tmp_path)
    both = sorted(rids[k] for k in ("job7@none", "job7@steer"))
    r = _run_daemon(root, tmp_path, *_mirror_args(tmp_path)); assert r.returncode == 0, r.stderr + r.stdout
    ev = _events(tmp_path)
    (sp,) = [e for e in ev if e["event"] == "dispatch_spooled"]; assert sorted(sp["run_ids"]) == both
    ledger = json.loads((tmp_path / "state" / "dispatch_spooled.json").read_text())["runs"]
    for rid in both:                                                                # bound: this dispatch's judgment id + the store generation
        assert len(ledger[rid]["judgment_ids"]) == 1 and ledger[rid]["judgment_ids"] == sp["judgment_ids"][rid] and "|200|" in ledger[rid]["store_generation"]
        assert ledger[rid]["dispatch_utc"] <= ledger[rid]["spooled_utc"]
    # tick 2, no fold yet: held (as before)
    r2 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path)); assert r2.returncode == 0, r2.stderr + r2.stdout
    ev2 = _events(tmp_path)[len(ev):]
    assert [e for e in ev2 if e["event"] == "spooled_hold"] and not [e for e in ev2 if e["event"] in ("dispatch", "dispatched")], ev2
    # the fold lands the SHORT judgments (150 of 200 items): the runs still pend (coverage_short) — the hold releases into a NAMED quarantine
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    with DB.reading(DB.db_path(root)) as conn:
        from sisteer.registry import pending as P
        pend = {x["run_id"]: x for x in P.pending(conn, None, synth.CAMPAIGN, run_id=tuple(both))}
        assert set(pend) == set(both) and {x["pending_reason"] for x in pend.values()} == {"coverage_short"} and {x["items_missing"] for x in pend.values()} == {50}
    r3 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path)); assert r3.returncode == 0, r3.stderr + r3.stdout
    ev3 = _events(tmp_path)[len(ev) + len(ev2):]
    rel = [e for e in ev3 if e["event"] == "spooled_hold_released"]
    assert sorted(e["run_id"] for e in rel) == both and all(e["reason"] == "folded_still_pending" and e["quarantined"] and "coverage_short" in e["defect"]
                                                           and "n_items_judged=150" in e["defect"] and "items_missing=50" in e["defect"] for e in rel), rel
    tick3 = [e for e in ev3 if e["event"] == "tick"][-1]
    assert tick3["dispatched"] == 0 and tick3["spooled_hold"] == 0 and tick3["quarantined"] == 2, tick3
    assert not [e for e in ev3 if e["event"] in ("dispatch", "dispatched")], ev3
    q = json.loads((tmp_path / "state" / "dispatch_quarantine.json").read_text())["runs"]
    assert sorted(q) == both and all(q[rid]["rc"] == "coverage_short" and "n_items_judged=150" in q[rid]["stderr_tail"] and q[rid]["judgment_ids"] == ledger[rid]["judgment_ids"] for rid in both)
    assert json.loads((tmp_path / "state" / "dispatch_spooled.json").read_text())["runs"] == {}
    (prob,) = [e for e in ev3 if e["event"] == "problem" and e["kind"] == "dispatch_quarantine"]; assert prob["n"] == 2
    # the controlled retry is the quarantine's: the release file re-dispatches once (and the daemon's own code sha or a re-filed store would too)
    (tmp_path / "state" / "dispatch_quarantine.release").write_text("*\n")
    monkeypatch.delenv("FAKE_DRIVER_SHORT_ITEMS")
    r4 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path)); assert r4.returncode == 0, r4.stderr + r4.stdout
    ev4 = _events(tmp_path)[len(ev) + len(ev2) + len(ev3):]
    assert sorted(e["run_id"] for e in ev4 if e["event"] == "quarantine_released") == both and [e for e in ev4 if e["event"] == "tick"][-1]["dispatched"] == 2
    # unit: the hold's proof is THIS dispatch's judgment, never a historical complete judgment in the directory
    GD = _load_daemon_module(); st = tmp_path / "state"
    other = rids["job8@other"]
    from sisteer.registry import identity as I
    d = Path(root) / "records" / synth.CAMPAIGN / I.run_dir_name(other); d.mkdir(parents=True, exist_ok=True)
    hist = R.make_record("judgment", {"judgment_id": "J-historical", "run_id": other, "judge_id": "j", "status": "complete", "n_items_judged": 200}, {"script": "t"}, "rec-hist")
    (d / "000900-judgment.jsonl").write_text(json.dumps(hist) + "\n")                # a complete judgment record — not this dispatch's
    gen_now = None
    with DB.reading(DB.db_path(root)) as conn:
        gen_now = GD.store_generation(conn, other, None)
    GD.spooled_save(str(st), {other: {"run_id": other, "family": "static", "campaign": synth.CAMPAIGN, "spooled_utc": "2026-01-01T00:00:00Z",
                                      "judgment_ids": ["J-this-dispatch"], "store_generation": gen_now}})
    rows = GD.pending_rows(root, [OTHER_MODEL], synth.CAMPAIGN, None, {synth.EVAL_ID: "static"})
    eligible, held = GD.spooled_filter(str(st), root, rows)
    assert [r_["run_id"] for r_ in eligible] == [other] and held == []
    assert [e for e in _events(tmp_path) if e["event"] == "spooled_hold_dropped"][-1]["reason"] == "bound_judgment_not_in_spool"
    # unit: a hold whose judgment IS in the spool stands — until the store's generation changes under it
    (d / "000901-judgment.jsonl").write_text(json.dumps(R.make_record("judgment", {"judgment_id": "J-this-dispatch", "run_id": other, "judge_id": "j", "status": "complete",
                                                                                      "n_items_judged": 200}, {"script": "t"}, "rec-this")) + "\n")
    GD.spooled_save(str(st), {other: {"run_id": other, "family": "static", "campaign": synth.CAMPAIGN, "spooled_utc": "2026-01-01T00:00:00Z",
                                      "judgment_ids": ["J-this-dispatch"], "store_generation": gen_now}})
    eligible, held = GD.spooled_filter(str(st), root, rows)
    assert eligible == [] and [r_["run_id"] for r_ in held] == [other]
    GD.spooled_save(str(st), {other: {"run_id": other, "family": "static", "campaign": synth.CAMPAIGN, "spooled_utc": "2026-01-01T00:00:00Z",
                                      "judgment_ids": ["J-this-dispatch"], "store_generation": "an older store|198|deadbeef|2026-01-01T00:00:00Z"}})
    eligible, held = GD.spooled_filter(str(st), root, rows)
    assert [r_["run_id"] for r_ in eligible] == [other] and held == []
    assert [e for e in _events(tmp_path) if e["event"] == "spooled_hold_cleared"][-1]["reason"] == "store_regenerated"
    # unit: the operator's release file (an unresolved spool refusal) drops the hold
    GD.spooled_save(str(st), {other: {"run_id": other, "family": "static", "campaign": synth.CAMPAIGN, "spooled_utc": "2026-01-01T00:00:00Z",
                                      "judgment_ids": ["J-this-dispatch"], "store_generation": gen_now}})
    (st / "dispatch_spooled.release").write_text(f"{other}\n")
    eligible, held = GD.spooled_filter(str(st), root, rows)
    assert [r_["run_id"] for r_ in eligible] == [other] and held == [] and not (st / "dispatch_spooled.release").exists()
    assert [e for e in _events(tmp_path) if e["event"] == "spooled_hold_dropped"][-1]["reason"] == "release_file"


def test_two_instances_over_one_mirror_the_second_never_retires_what_the_first_is_grading_from(tmp_path, monkeypatch):
    """A daemon and a `--once` (or a replacement daemon) overlap on a still-pending run; the first is grading from its
    mirrored copy with a driver whose cwd is the REPO, not the mirror; the second's rsync or driver fails. The per-directory locks (`<dir>.lock`
    beside each store, held from before the copy through dispatch and settlement) make the second skip the group (`mirror_skipped_locked`) —
    no copy over, no retire — while the first grades; once the first files, its directories are pruned and the space is recovered."""
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    import time
    root, rids = _root(tmp_path)
    mirror = tmp_path / "mirror"; dirs = [mirror / "stores" / "job7_none", mirror / "stores" / "job7_steer"]
    hold = tmp_path / "driver_hold"; hold.write_text("")
    fail = tmp_path / "driver_fail"; fail.write_text("")
    first = _start_daemon(root, tmp_path, *_mirror_args(tmp_path), env_extra={"FAKE_DRIVER_HOLD_FILE": str(hold)})
    end = time.time() + 60
    while time.time() < end and not [e for e in _events(tmp_path) if e["event"] == "dispatch"]:
        time.sleep(0.1)
    assert [e for e in _events(tmp_path) if e["event"] == "dispatch"], "the first instance never reached its dispatch"
    assert all((d / "store.bin").exists() for d in dirs) and all((mirror / "stores" / f"{d.name}.lock").exists() for d in dirs)
    n0 = len(_events(tmp_path))
    # the second instance: a `--once` whose driver would fail (and whose rsync would overwrite the copies): it must skip the group
    second = _run_daemon(root, tmp_path, *_mirror_args(tmp_path), env_extra={"FAKE_DRIVER_FAIL_FILE": str(fail)})
    assert second.returncode == 0, second.stderr + second.stdout
    ev2 = _events(tmp_path)[n0:]
    (skipped,) = [e for e in ev2 if e["event"] == "mirror_skipped_locked"]
    assert sorted(skipped["dirs"]) == sorted(str(d) for d in dirs) and sorted(skipped["run_ids"]) == sorted([rids["job7@none"], rids["job7@steer"]])
    assert not [e for e in ev2 if e["event"] in ("mirror", "mirror_failed", "dispatch", "dispatched", "mirror_retired_incomplete", "mirror_pruned")], ev2
    assert [e for e in ev2 if e["event"] == "tick"][-1]["skipped_locked"] == 2
    assert all((d / "store.bin").stat().st_size == 4096 for d in dirs) and len((tmp_path / "rsync_calls.log").read_text().splitlines()) == 2   # untouched
    assert first.poll() is None                                                                     # the first is still grading
    # the first's driver finishes: graded, filed, pruned — the space is recovered
    hold.unlink()
    out, err = first.communicate(timeout=120)
    assert first.returncode == 0, err + out
    ev = _events(tmp_path)
    assert [e for e in ev if e["event"] == "dispatched"][-1]["rc"] == 0 and len([e for e in ev if e["event"] == "mirror_pruned"]) == 2
    assert not any(d.exists() for d in dirs) and mirror.is_dir()
    with DB.reading(DB.db_path(root)) as conn:
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rids[k],)).fetchone()[0] == "judged" for k in ("job7@none", "job7@steer"))
    # the prune pass defers a group whose directories another instance holds, and prunes once released
    import fcntl
    import grade_daemon as GD
    keep = mirror / "stores" / "job8_other"; keep.mkdir(parents=True); (keep / "store.bin").write_bytes(b"y" * 8)
    GD.mirror_prune_enqueue(str(tmp_path / "state"), family="static", model_key="k", campaign=synth.CAMPAIGN, run_ids=[rids["job7@none"]],
                            mirrored=[{"run_id": rids["job7@none"], "substrate": "cluster_a", "local": str(keep), "root": str(mirror), "rc": 0}], rc=0)
    fd = os.open(str(keep) + ".lock", os.O_RDWR | os.O_CREAT); fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        assert GD.mirror_prune_pass(root, str(tmp_path / "state"), dry_run=False, campaign=synth.CAMPAIGN) == {"groups": 0, "dirs": 0, "bytes": 0} and keep.is_dir()
        assert _events(tmp_path)[-1]["event"] == "mirror_prune_deferred" and _events(tmp_path)[-1]["reason"] == "locked by another instance"
    finally:
        os.close(fd)
    assert GD.mirror_prune_pass(root, str(tmp_path / "state"), dry_run=False, campaign=synth.CAMPAIGN) == {"groups": 1, "dirs": 1, "bytes": 8} and not keep.exists()


def test_a_daemon_killed_under_a_running_grader_leaves_the_locks_with_the_grader_and_a_replacement_skips_then_recovers(tmp_path, monkeypatch):
    """The daemon alone dies (SIGKILL of the parent pid) while its grader runs with cwd = the repo. The grader inherited the
    store-lock fds, so the locks live on with it: a replacement `--once` whose dispatch would fail skips the group (`mirror_skipped_locked`),
    defers the dead daemon's dispatch claim (`mirror_claim_deferred`), copies nothing and retires nothing; the grader's inputs stay intact.
    When the grader exits (graded, filed), the next instance reconciles the orphaned claim (`mirror_claim_reconciled`), queues the directories
    and prunes them — the space recovered without the dispatching daemon ever coming back."""
    import fcntl, signal, time
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, rids = _root(tmp_path)
    mirror = tmp_path / "mirror"; dirs = [mirror / "stores" / "job7_none", mirror / "stores" / "job7_steer"]
    hold = tmp_path / "driver_hold"; hold.write_text(""); fail = tmp_path / "driver_fail"; fail.write_text(""); pidf = tmp_path / "grader.pid"
    first = _start_daemon(root, tmp_path, *_mirror_args(tmp_path), env_extra={"FAKE_DRIVER_HOLD_FILE": str(hold), "FAKE_DRIVER_PID_FILE": str(pidf)})
    end = time.time() + 60
    while time.time() < end and not (pidf.exists() and pidf.read_text().strip()):
        time.sleep(0.1)
    grader = int(pidf.read_text().strip())
    claims = json.loads((tmp_path / "state" / "mirror_dispatch_claims.json").read_text())["claims"]
    assert len(claims) == 1 and claims[0]["daemon_pid"] == first.pid and sorted(claims[0]["run_ids"]) == sorted([rids["job7@none"], rids["job7@steer"]])
    # kill ONLY the daemon: the grader survives (reparented), and the locks survive with it
    os.kill(first.pid, signal.SIGKILL); first.wait(timeout=30)
    assert _proc_state(grader) not in (None, "Z")                                       # the grader is alive and not a zombie before the restart
    for d in dirs:
        fd = os.open(str(d) + ".lock", os.O_RDWR)
        with pytest.raises(OSError):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)           # held by the grader's inherited fd, not by the dead daemon
        os.close(fd)
    n0 = len(_events(tmp_path))
    # the replacement: its dispatch would fail; it must not touch the grader's inputs
    second = _run_daemon(root, tmp_path, *_mirror_args(tmp_path), env_extra={"FAKE_DRIVER_FAIL_FILE": str(fail)})
    assert second.returncode == 0, second.stderr + second.stdout
    ev2 = _events(tmp_path)[n0:]
    (deferred,) = [e for e in ev2 if e["event"] == "mirror_claim_deferred"]; assert deferred["daemon_pid"] == first.pid and sorted(deferred["dirs"]) == sorted(str(d) for d in dirs)
    (skipped,) = [e for e in ev2 if e["event"] == "mirror_skipped_locked"]; assert sorted(skipped["dirs"]) == sorted(str(d) for d in dirs)
    assert not [e for e in ev2 if e["event"] in ("mirror", "mirror_failed", "dispatch", "dispatched", "mirror_retired_incomplete", "mirror_pruned")], ev2
    tick2 = [e for e in ev2 if e["event"] == "tick"][-1]; assert tick2["skipped_locked"] == 2 and tick2["mirror_claims_deferred"] == 1 and tick2["mirror_claims_reconciled"] == 0
    assert all((d / "store.bin").stat().st_size == 4096 for d in dirs) and len((tmp_path / "rsync_calls.log").read_text().splitlines()) == 2
    assert len(json.loads((tmp_path / "state" / "mirror_dispatch_claims.json").read_text())["claims"]) == 1
    # the grader finishes (graded, filed) and exits: the locks are released with it
    hold.unlink()
    end = time.time() + 60
    while time.time() < end and not _grader_exited(grader):                            # one read per poll: gone or zombie = exited
        time.sleep(0.1)
    with DB.reading(DB.db_path(root)) as conn:
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rids[k],)).fetchone()[0] == "judged" for k in ("job7@none", "job7@steer"))
    n1 = len(_events(tmp_path))
    third = _run_daemon(root, tmp_path, *_mirror_args(tmp_path))
    assert third.returncode == 0, third.stderr + third.stdout
    ev3 = _events(tmp_path)[n1:]
    (rec,) = [e for e in ev3 if e["event"] == "mirror_claim_reconciled"]; assert rec["daemon_pid"] == first.pid and rec["queued"] == 2 and rec["retired"] == 0
    assert len([e for e in ev3 if e["event"] == "mirror_pruned"]) == 2 and not any(d.exists() for d in dirs) and mirror.is_dir()
    tick3 = [e for e in ev3 if e["event"] == "tick"][-1]
    assert tick3["pending"] == 0 and tick3["mirror_claims_reconciled"] == 1 and tick3["mirror_pruned_dirs"] == 2 and tick3["mirror_prune_pending"] == 0
    assert json.loads((tmp_path / "state" / "mirror_dispatch_claims.json").read_text())["claims"] == []
    # the other face: a dead daemon's claim whose grader ALSO died before filing -> the still-pending copies are retired at reconciliation
    root2, rids2 = _root(tmp_path / "second")
    (tmp_path / "second" / "mirror" / "stores" / "job7_none").mkdir(parents=True); ((tmp_path / "second" / "mirror" / "stores" / "job7_none") / "store.bin").write_bytes(b"z" * 5)
    import grade_daemon as GD
    st2 = tmp_path / "second" / "state"; st2.mkdir()
    GD._claims_save(str(st2), [{"id": "orphan", "daemon_pid": 2 ** 22 - 1, "daemon_start": 1, "family": "static", "model_key": "k", "campaign": synth.CAMPAIGN,
                                "run_ids": [rids2["job7@none"]], "mirrored": [{"run_id": rids2["job7@none"], "substrate": "cluster_a", "local": str(tmp_path / "second" / "mirror" / "stores" / "job7_none"),
                                                                              "root": str(tmp_path / "second" / "mirror"), "rc": 0}], "claimed_utc": "2026-01-01T00:00:00Z"}])
    out = GD.mirror_reconcile_claims(str(st2), root2, synth.CAMPAIGN)
    assert out == {"reconciled": 1, "deferred": 0} and not (tmp_path / "second" / "mirror" / "stores" / "job7_none").exists()
    ev4 = [json.loads(l) for l in (st2 / "grade_daemon.jsonl").read_text().splitlines()]
    assert [e["event"] for e in ev4] == ["mirror_retired_incomplete", "mirror_claim_reconciled"] and ev4[0]["bytes"] == 5 and "died before settling" in ev4[0]["note"]


def test_a_low_disk_mirror_root_pauses_the_groups_grading_and_the_wave_is_untouched(tmp_path, monkeypatch):
    """Below the free-space floor nothing is mirrored and the group's dispatch is skipped for the tick (`mirror_skipped_low_disk`;
    the runs stay pending, unclaimed); the next tick with room dispatches it."""
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, rids = _root(tmp_path)
    r = _run_daemon(root, tmp_path, *_mirror_args(tmp_path), "--mirror-min-free-bytes", str(10 ** 18))
    assert r.returncode == 0, r.stderr + r.stdout
    ev = _events(tmp_path)
    (skipped,) = [e for e in ev if e["event"] == "mirror_skipped_low_disk"]
    assert skipped["substrate"] == "cluster_a" and skipped["mirror_root"] == str(tmp_path / "mirror") and skipped["need_bytes"] == 10 ** 18
    assert skipped["free_bytes"] == FAKE_FREE_BYTES < 10 ** 18                                   # the stated reading (_daemon_argv_env), not the host's disk
    assert sorted(skipped["run_ids"]) == sorted([rids["job7@none"], rids["job7@steer"]])
    assert not [e for e in ev if e["event"] in ("mirror", "dispatch", "dispatched")] and not (tmp_path / "driver_calls.jsonl").exists() and not (tmp_path / "rsync_calls.log").exists()
    tick = [e for e in ev if e["event"] == "tick"][-1]
    assert tick["pending"] == 2 and tick["dispatched"] == 0 and tick["skipped_low_disk"] == 2 and tick["mirror_prune_pending"] == 0
    assert tick["mirror_free_bytes_override"] == FAKE_FREE_BYTES == tick["mirror_free_bytes"]["cluster_a"] and "mirror_free_bytes_override_ignored" not in tick   # the seam, named
    with DB.reading(DB.db_path(root)) as conn:
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rids[k],)).fetchone()[0] == "generated" for k in ("job7@none", "job7@steer"))
        assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 0
    r2 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path))                                 # room again: mirrored, dispatched, graded
    assert r2.returncode == 0, r2.stderr + r2.stdout
    tick2 = [e for e in _events(tmp_path) if e["event"] == "tick"][-1]
    assert tick2["dispatched"] == 2 and tick2["skipped_low_disk"] == 0 and len((tmp_path / "rsync_calls.log").read_text().splitlines()) == 2


def test_the_free_bytes_override_is_honoured_only_under_pytest_outside_it_the_real_low_reading_pauses_the_group(tmp_path, monkeypatch):
    """GRADE_DAEMON_FREE_BYTES (grade_daemon.MIRROR_FREE_BYTES_ENV) is a TEST seam, not a production knob: the daemon honours it only while
    PYTEST_CURRENT_TEST is in its environment. In-process, with the real reading LOW and a large override set but PYTEST_CURRENT_TEST absent
    from the daemon's view, the preflight reads the real disk, emits `mirror_skipped_low_disk` with the REAL figure and copies nothing; a
    value that is not a byte count is ignored there too, and refused by name only once the seam is armed. As a subprocess outside pytest the
    tick summary carries the real readings, names the ignored variable (`mirror_free_bytes_override_ignored`, plus one event per tick) and
    never `mirror_free_bytes_override`; under pytest it carries the stated figure as `mirror_free_bytes_override`, printed with the summary."""
    import grade_daemon as GD
    real_low = 512 * 1024                                                            # the real disk: half the 1 MiB floor
    monkeypatch.setattr(GD.DB, "free_bytes", lambda path: real_low)
    monkeypatch.setattr(GD, "mirror_store_size", lambda rsync_bin, src, dst: 4096)    # the probe; no rsync runs in this process
    monkeypatch.setenv(GD.MIRROR_FREE_BYTES_ENV, str(10 ** 18))                       # the stray: a disk that would read as full of room
    monkeypatch.delenv(GD.PYTEST_CURRENT_TEST_ENV, raising=False)                     # outside pytest, as the daemon sees it
    mirror = tmp_path / "mirror"; state = tmp_path / "state"
    rows = [{"run_id": f"r{i}", "substrate": "cluster_a", "family": "static", "store_uri": f"/cluster_a/user/exp/stores/s{i}",
             "store_path": str(mirror / "stores" / f"s{i}")} for i in (1, 2)]
    assert GD.mirror_free_bytes_override() is None and GD.mirror_free_bytes_override_ignored() == str(10 ** 18) and GD.mirror_free_bytes(str(tmp_path)) == real_low
    out = GD.mirror_stores(rows, {"cluster_a": "cluster-a-login"}, str(tmp_path / "no-such-rsync"), False, str(state), {"cluster_a": str(mirror)}, 1024 * 1024)
    ev = _events(tmp_path)
    assert out is None and [e["event"] for e in ev] == ["mirror_skipped_low_disk"]
    assert ev[0]["free_bytes"] == real_low and ev[0]["store_bytes"] == 8192 and ev[0]["need_bytes"] == 1024 * 1024 and sorted(ev[0]["run_ids"]) == ["r1", "r2"]
    assert mirror.is_dir() and all(not any((mirror / "stores" / f"s{i}").iterdir()) for i in (1, 2))   # the probe's empty dirs only: nothing copied
    monkeypatch.setenv(GD.MIRROR_FREE_BYTES_ENV, "lots")                              # a stray that is not even a number: named, never read
    assert GD.mirror_free_bytes_override() is None and GD.mirror_free_bytes_override_ignored() == "lots" and GD.mirror_free_bytes(str(tmp_path)) == real_low
    monkeypatch.setenv(GD.PYTEST_CURRENT_TEST_ENV, "tests/test_grade_daemon.py::seam (call)")   # the seam armed: refused by name, and a byte count is read
    with pytest.raises(ValueError, match=GD.MIRROR_FREE_BYTES_ENV):
        GD.mirror_free_bytes_override()
    monkeypatch.setenv(GD.MIRROR_FREE_BYTES_ENV, str(10 ** 18))
    assert GD.mirror_free_bytes_override() == 10 ** 18 == GD.mirror_free_bytes(str(tmp_path)) and GD.mirror_free_bytes_override_ignored() is None
    # as a SUBPROCESS (dry runs: nothing is copied or dispatched either way; this is about the tick summary): outside pytest the tick's readings
    # are the real disk's, the stray is named in the summary and in one event, and `mirror_free_bytes_override` is absent; under pytest the
    # stated figure is carried as `mirror_free_bytes_override`, nothing is named as ignored, and both forms print with the summary line
    sub = tmp_path / "sub"; root, _rids = _root(sub); (sub / "mirror").mkdir()
    argv, env = _daemon_argv_env(root, sub, *_mirror_args(sub), dry_run=True, env_extra={GD.MIRROR_FREE_BYTES_ENV: str(10 ** 18)})
    env.pop(GD.PYTEST_CURRENT_TEST_ENV, None)
    r = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=600, cwd=str(REPO))
    assert r.returncode == 0, r.stderr + r.stdout
    ev_sub = _events(sub); tick = [e for e in ev_sub if e["event"] == "tick"][-1]
    assert tick["mirror_free_bytes_override_ignored"] == str(10 ** 18) and "mirror_free_bytes_override" not in tick and tick["mirror_free_bytes"]["cluster_a"] != 10 ** 18
    (ignored,) = [e for e in ev_sub if e["event"] == "mirror_free_bytes_override_ignored"]
    assert ignored["variable"] == GD.MIRROR_FREE_BYTES_ENV == "GRADE_DAEMON_FREE_BYTES" and ignored["value"] == str(10 ** 18) and GD.PYTEST_CURRENT_TEST_ENV in ignored["note"]
    assert f'"mirror_free_bytes_override_ignored": "{10 ** 18}"' in r.stdout
    n = len(ev_sub)
    env[GD.PYTEST_CURRENT_TEST_ENV] = "tests/test_grade_daemon.py::seam (call)"; env[GD.MIRROR_FREE_BYTES_ENV] = str(FAKE_FREE_BYTES)
    r2 = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=600, cwd=str(REPO))
    assert r2.returncode == 0, r2.stderr + r2.stdout
    ev_sub2 = _events(sub)[n:]; tick2 = [e for e in ev_sub2 if e["event"] == "tick"][-1]
    assert tick2["mirror_free_bytes_override"] == FAKE_FREE_BYTES == tick2["mirror_free_bytes"]["cluster_a"] and "mirror_free_bytes_override_ignored" not in tick2
    assert not [e for e in ev_sub2 if e["event"] == "mirror_free_bytes_override_ignored"] and f'"mirror_free_bytes_override": {FAKE_FREE_BYTES}' in r2.stdout


def test_a_row_another_live_session_holds_and_a_rule_scored_family_are_never_dispatched(tmp_path):
    root, rids = _root(tmp_path)
    from sisteer.registry import pending as P
    P.claim(root, rids["job7@none"], synth.JUDGE_ID, "another-session")
    # a rule-scored run (MMLU) of the model: confirm only
    # a rule-scored run (the synthetic capability eval stands for MMLU / GSM8K): its instrument names no judge, so it is pending for nobody,
    # and even when routed to a rule-scored family the daemon confirms it and never dispatches a judge
    rid_m, recs = synth.rule_scored_run_records("job9@mmlu", 200)
    R.write_records(os.path.join(root, "records", "c", "000009-run.jsonl"), recs)
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    r = _run_daemon(root, tmp_path, "--eval-family", f"{synth.CAP_EVAL_ID}=mmlu", dry_run=True)
    assert r.returncode == 0, r.stderr + r.stdout
    disp = [e for e in _events(tmp_path) if e["event"] == "dispatch"]
    assert len(disp) == 1 and disp[0]["run_ids"] == [rids["job7@steer"]]                     # the claimed row is left to its session
    assert not any(rid_m in e.get("run_ids", []) for e in disp)


def test_refuses_a_bad_store_map_and_a_missing_registry(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    r = subprocess.run([sys.executable, str(DAEMON), "--registry-root", str(tmp_path / "nowhere"), "--model-id", "m", "--once"], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "no registry database" in r.stderr
    root, _ = _root(tmp_path)
    r = subprocess.run([sys.executable, str(DAEMON), "--registry-root", root, "--model-id", "m", "--once", "--store-root-map", "nope"], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "store-root-map" in r.stderr


def test_mirror_status_derives_the_per_cluster_ingest_watermark_read_only(tmp_path):
    """`scripts/registry/mirror_status.py`: per substrate the later of the last successful
    `ingest --pull` and the newest folded record file of that substrate's runs; unknown reads null; written only where the caller says."""
    root, rids = _root(tmp_path)
    tool = REPO / "scripts" / "registry" / "mirror_status.py"
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    out = tmp_path / "state" / "mirror_status.json"
    r = subprocess.run([sys.executable, str(tool), "--registry-root", root, "--out", str(out), "--label", "cluster_a=site_a"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    st = json.loads(out.read_text())
    assert st["cluster_a"]["basis"] == "last_file_arrival" and st["cluster_a"]["ingested_through_utc"] is not None    # the synthetic runs are cluster_a's
    # the site label is its own name: the example site spells its clusters cluster_a / cluster_b, like the registry labels
    assert st["site_a"] == dict(st["cluster_a"], substrate="cluster_a")
    assert st["cluster_b"]["ingested_through_utc"] is None and st["cluster_b"]["basis"] == "unknown"                  # nothing from there: never guessed
    assert st["slurm"]["ingested_through_utc"] is None
    # a recorded successful `ingest --pull` later than every arrival covers every cluster
    lr = Path(root) / "state" / "last_run"
    lr.mkdir(parents=True, exist_ok=True)
    (lr / "ingest.json").write_text(json.dumps({"command": "ingest", "utc": "2099-01-01T00:00:00.000Z", "rc": 0, "argv": ["ingest", "--pull"]}))
    r = subprocess.run([sys.executable, str(tool), "--registry-root", root], capture_output=True, text=True, env=env)
    st2 = json.loads(r.stdout)
    assert st2["cluster_b"]["ingested_through_utc"] == "2099-01-01T00:00:00.000Z" and st2["cluster_b"]["basis"] == "last_pull"
    assert st2["cluster_a"]["basis"] == "last_pull"
    # a failed pull is no watermark
    (lr / "ingest.json").write_text(json.dumps({"command": "ingest", "utc": "2099-01-02T00:00:00.000Z", "rc": 3, "argv": ["ingest", "--pull"]}))
    st3 = json.loads(subprocess.run([sys.executable, str(tool), "--registry-root", root], capture_output=True, text=True, env=env).stdout)
    assert st3["cluster_b"]["ingested_through_utc"] is None
    assert not (Path(root) / "state" / "mirror_status.json").exists()      # the tool never writes under the registry root on its own


# --------------------------------------------------------------------------------------------
# Every family's driver gets EVERY argument it requires, derived from the pending run's store
# and an explicit config mapping; the ACTUAL 08_grade.sh -> daemon templates are driven against representative pending runs of every
# family and each driver's own argparse accepts the composed command (a parse-only probe over the real parsers; nothing is graded)
# --------------------------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import assert_no_tokens, run_stage, write_site  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "registry"))
import grade_daemon as GD  # noqa: E402

PAPER_MODELS = ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b")
TEST_MODELS_VIEWS = (("qwen35_4b_demo", "battery"),)      # the demo's one wave is the row's DECLARED `battery` (demo-import M3)
#: the seven dispatched families and the representative pending run of each: (eval_id, eval_category, decode_source, store path
#: RELATIVE to the cluster's shared root, private run); the store layouts are the drivers' own (grade_daemon.FAMILY_LAYOUT)
FAMILY_RUNS = {
    "static": ("pasttense", "static_attack", "generation_config", "exp/sdfbase_full22_native/transcripts_results/sdfbase_full22_{tag}_sampled_t0/pasttense/none"),
    "static2": ("codeattack", "static_attack", "generation_config", "exp/sdfbase_full22_native/transcripts_results/sdfbase_full22_{tag}_sampled_t0/codeattack/none"),
    "padding": ("grok420", "static_attack", "generation_config", "exp/sdfbase_full22_native/static_padding/transcripts_results/pad_static_{padkey}_p0/grok420/none"),   # the hand-listed key's prompted-only family; a front-door key's PER-VIEW family (pad_static_<key>_<view>_p<pad>)
    "bon": ("strongreject_bon", "best_of_n", "generation_config", "exp/sdfbase_full22_native/transcripts_results/sdfbase_full22_{tag}/none"),
    "multiturn": ("crescendo", "multi_turn", "generation_config", "exp/sdfbase_full22_native/transcripts_results/sdfbase_full22_{tag}/crescendo/none"),
    "controls": ("xstest", "over_refusal", "generation_config", "exp/sdfbase_full22_native/sampled/controls_cap512/{tag}_t0/cells/none__xstest"),
    "agentic": ("agentic_leaking", "agentic", "generation_config", "exp/misalign/{tag}/agentic/none"),
    "petri": ("petri_pooled", "petri", "generation_config", "petri_native/petri_full22_{tag}/cells/petri__none"),
}
#: a parse-only run of a driver: its REAL argparse over the daemon's composed argv, then stop (the namespace as JSON on stdout, exit 0);
#: an argparse refusal is argparse's own exit 2 with its message on stderr. No config is read, no store opened, no judge built.
PARSE_PROBE = r'''
import argparse, json, os, runpy, sys
driver, argv = sys.argv[1], sys.argv[2:]
orig = argparse.ArgumentParser.parse_args
def patched(self, args=None, namespace=None):
    ns = orig(self, args, namespace)
    print("PARSED " + json.dumps({k: (str(v) if not isinstance(v, (int, float, bool, list, type(None))) else v) for k, v in vars(ns).items()}))
    raise SystemExit(0)
argparse.ArgumentParser.parse_args = patched
sys.argv = [driver] + argv
sys.path.insert(0, os.path.dirname(driver))
runpy.run_path(driver, run_name="__main__")
'''


def _model_rec(mid, i, family="qwen35", size="4b"):
    return synth.rec("model", {"model_id": mid, "family": family, "size_label": size, "checkpoint_kind": "instruct", "hf_repo": "org/x",
                               "hf_revision": "d" * 40, "chat_template_sha256": synth.SHA256_A, "thinking_channel": "available",
                               "paper_status": "in_paper"}, synth.t(i))


def _family_dims():
    dims = synth.seed_dimensions()
    dims += [_model_rec(m, 40 + k) for k, m in enumerate(("qwen35_4b_it", "qwen35_4b_base"))]
    dims += [_model_rec(m, 42 + k, family="gemma4", size="31b") for k, m in enumerate(("qwen35_4b_demo", "qwen35_4b_demo_base"))]
    i = 60
    seen = set()
    for fam, (eval_id, cat, dec, _rel) in FAMILY_RUNS.items():
        if eval_id in seen:
            continue
        seen.add(eval_id)
        dims.append(synth.rec("eval", {"eval_id": eval_id, "eval_category": cat, "headline_metric_id": "asr", "metric_direction": "lower_is_better",
                                       "attacker_required": 0, "paper_status": "in_paper"}, synth.t(i)))
        dims.append(synth.rec("item_set", {"item_set_id": f"{eval_id}_set", "eval_id": eval_id, "dataset_version": f"synthetic/{eval_id}@v1", "n_items": 200,
                                           "items_sha256": synth.SHA256_B, "source_uri": f"synthetic://items/{eval_id}.jsonl", "item_id_scheme": "q<index>",
                                           "subsample_rule": "full"}, synth.t(i + 1)))
        dims.append(synth.rec("instrument", _instrument_for(fam), synth.t(i + 2)))
        i += 3
    return dims


def _instrument_for(fam):
    eval_id, _cat, dec, _rel = FAMILY_RUNS[fam]
    over = {"eval_id": eval_id, "item_set_id": f"{eval_id}_set", "decode_source": dec}
    if fam == "padding":       # a padded level's instrument: the filler contract beside the level (CHECK instrument_padding_pairing)
        over.update(padding_tokens=0, padding_identity_sha256=synth.SHA256_B)
    return synth.instrument_fields(**over)


def _family_run(fam, key, i0, *, model_id, view_legacy, shared_root, tag, mkey):
    eval_id, _cat, dec, rel = FAMILY_RUNS[fam]
    inst = _instrument_for(fam)
    store = os.path.join(shared_root, rel.format(tag=tag, key=mkey, padkey=(tag if tag != mkey else mkey)))   # a front-door tag `<key>_<view>` pads per view
    p = synth.run_payload(key, "none", "none", i0, model_id=model_id, view_legacy=view_legacy, eval_id=eval_id, item_set_id=f"{eval_id}_set",
                          instrument_id=inst["instrument_id"], decode_source=dec, store_uri=store)
    p["run_id"] = identity.run_id(synth.CAMPAIGN, key, "none", eval_id, str(inst["instrument_id"]))
    rid = str(p["run_id"])
    t = synth.t
    recs = [synth.rec("run", p, t(i0)),
            synth.rec("run_state", {"run_id": rid, "state": "queued", "event_utc": t(i0), "evidence_kind": "submit_wrapper"}, t(i0)),
            synth.rec("run_state", {"run_id": rid, "state": "running", "event_utc": t(i0 + 1), "evidence_kind": "job_log"}, t(i0 + 1)),
            synth.rec("run_state", {"run_id": rid, "state": "generated", "event_utc": t(i0 + 3), "evidence_kind": "completion_artifact",
                                    "evidence_uri": store}, t(i0 + 3)),
            synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_generated": 200}}, t(i0 + 4))]
    return rid, store, recs


def _family_registry(root: str, shared_root: str):
    """A synthetic registry with one pending run PER FAMILY for the paper model qwen35_4b (main view: tag == key) and for the
    configured-not-run front-door model qwen35_4b_demo (view battery: tag qwen35_4b_demo_battery), every store under the neutral cluster's shared root."""
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), _family_dims())
    rids = {}
    i = 0
    for mid, view_legacy, tag, mkey in (("qwen35_4b_it", "qwen35_4b", "qwen35_4b", "qwen35_4b"),
                                        ("qwen35_4b_demo", "qwen35_4b_demo_battery", "qwen35_4b_demo_battery", "qwen35_4b_demo")):
        for fam in FAMILY_RUNS:
            i += 1
            rid, store, recs = _family_run(fam, f"{mkey}_{fam}", 100 + 20 * i, model_id=mid, view_legacy=view_legacy, shared_root=shared_root, tag=tag, mkey=mkey)
            R.write_records(os.path.join(root, "records", "c", f"{i:06d}-run.jsonl"), recs)
            rids[(mid, fam)] = (rid, store)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.as_dict()
    return rids


def _dry_run_argv(stage_out: str):
    """The daemon argv the front door composed: the one `DRY-RUN:` line naming grade_daemon.py (bash `%q`-quoted)."""
    import shlex
    lines = [ln for ln in stage_out.splitlines() if ln.startswith("DRY-RUN:") and "grade_daemon.py" in ln]
    assert len(lines) == 1, stage_out[-3000:]
    return shlex.split(lines[0][len("DRY-RUN:"):])


def _probe(cmd_argv):
    """Run the real driver's argparse over the daemon's composed command; returns (returncode, namespace-or-None, stderr)."""
    py, driver, *args = cmd_argv
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    env.pop(emit.DISABLED_ENV, None)
    r = subprocess.run([py, "-c", PARSE_PROBE, driver, *args], capture_output=True, text=True, env=env, cwd=str(REPO), timeout=600)
    ns = None
    for ln in r.stdout.splitlines():
        if ln.startswith("PARSED "):
            ns = json.loads(ln[len("PARSED "):])
    return r.returncode, ns, r.stderr


def test_config_mapping_resolves_for_the_papers_models_and_names_the_test_models_configs():
    """The explicit model-key -> config mapping (grade_daemon.config_for / target_for / store_root_of) resolves, for EVERY paper model, to
    a config committed in this tree (the sampled t0 static config, the wave config for multi-turn / BoN, the Petri campaign manifest; the
    padded lane's p0 config for the one model the padding generator renders), and for the configured-not-run front-door model to the path
    05_stage_roster.sh --step configs generates (the wave config `configs/sdfbase_full22_<key>_<view>.yaml`); a private run outside a
    family's naming resolves to no config and no target, never a guess."""
    cr = str(REPO / "configs")
    shared = "/scratch/example/shared"
    for key in PAPER_MODELS:
        # static (the sampled lane, trial 0)
        root, pr = GD.store_root_of("static", f"{shared}/exp/sdfbase_full22_native/transcripts_results/sdfbase_full22_{key}_sampled_t0/pasttense/none")
        assert (root, pr) == (f"{shared}/exp/sdfbase_full22_native/transcripts_results", f"sdfbase_full22_{key}_sampled_t0")
        assert GD.target_for("static", pr, key) == key
        cfg = GD.config_for("static", configs_root=cr, private_run=pr)
        assert cfg == f"{cr}/sdfbase_full22_{key}_sampled_t0.yaml" and os.path.isfile(cfg), cfg
        # the escalation rung spelling resolves to its own config too
        assert GD.target_for("static", f"sdfbase_full22_{key}_sampled_t1_cap8192", key) == key
        # multi-turn / BoN (the wave config; BoN's store sits one level shallower)
        root_mt, pr_mt = GD.store_root_of("multiturn", f"{shared}/x/transcripts_results/sdfbase_full22_{key}/crescendo/none")
        root_bon, pr_bon = GD.store_root_of("bon", f"{shared}/x/transcripts_results/sdfbase_full22_{key}/none")
        assert root_mt == root_bon == f"{shared}/x/transcripts_results" and pr_mt == pr_bon == f"sdfbase_full22_{key}"
        for fam, pr_ in (("multiturn", pr_mt), ("bon", pr_bon)):
            cfg = GD.config_for(fam, configs_root=cr, private_run=pr_)
            assert cfg == f"{cr}/sdfbase_full22_{key}.yaml" and os.path.isfile(cfg), (fam, cfg)
            assert GD.target_for(fam, pr_, key) == key
        # petri: the model key's campaign manifest
        cfg = GD.config_for("petri", configs_root=cr, model_key=key)
        assert cfg == f"{cr}/capeval/manifests/petri_full22_{key}.yaml" and os.path.isfile(cfg), cfg
        # agentic / controls: a root, no config
        assert GD.store_root_of("agentic", f"{shared}/exp/misalign/{key}/agentic/none") == (f"{shared}/exp/misalign/{key}", "agentic")
        assert GD.store_root_of("controls", f"{shared}/exp/s/controls_cap512/{key}_t0/cells/st") == (f"{shared}/exp/s/controls_cap512", f"{key}_t0")
        assert GD.config_for("agentic", configs_root=cr, private_run="agentic", model_key=key) is None
        assert GD.config_for("controls", configs_root=cr, private_run=f"{key}_t0", model_key=key) is None
    # padding: the generator's spelling, for the levels and families it renders (qwen35_4b today)
    for pr, name in (("pad_static_qwen35_4b_p0", "sdfbase_full22_qwen35_4b.padstatic_p0.yaml"),
                     ("pad_static_qwen35_4b_obj_p8192", "sdfbase_full22_qwen35_4b.padstatic_obj_p8192.yaml"),
                     ("pad_static_qwen35_4b_obj_s2_p65536", "sdfbase_full22_qwen35_4b.padstatic_obj_s2_p65536.yaml"),
                     # the front door's PER-VIEW steered families: a phase view name, the key split off unambiguously — for the demo's DECLARED wave
                     # `battery` (demo-import M3) by the models table (front_door_views.split_key_view + is_phase_view), never a `battery` literal in the regex
                     ("pad_static_qwen35_4b_demo_battery_p0", "sdfbase_full22_qwen35_4b_demo.padstatic_battery_p0.yaml"),
                     ("pad_static_qwen35_4b_demo_d0p75_relu_p8192", "sdfbase_full22_qwen35_4b_demo.padstatic_d0p75_relu_p8192.yaml"),
                     ("pad_static_qwen35_4b_demo_battery_p32768", "sdfbase_full22_qwen35_4b_demo.padstatic_battery_p32768.yaml"),
                     ("pad_static_qwen35_4b_demo_p0", "sdfbase_full22_qwen35_4b_demo.padstatic_p0.yaml")):
        cfg = GD.config_for("padding", configs_root=cr, private_run=pr)
        assert cfg == f"{cr}/sdfbase/static_padding/{name}", cfg
        key = pr.split("_p")[0][len("pad_static_"):]
        key = "qwen35_4b_demo" if key.startswith("qwen35_4b_demo") else "qwen35_4b" if key.startswith("qwen35_4b") else key.split("_s")[0].split("_d0")[0]
        assert GD.target_for("padding", pr, f"{key}_x") == key, (pr, key)
        if key == "qwen35_4b":
            assert os.path.isfile(cfg), cfg                        # the hand-listed key's configs are committed; a front-door key's are stage 05's
    # the front-door model: the wave config 05_stage_roster.sh --step configs generates for the phase view
    for key, view in TEST_MODELS_VIEWS:
        tag = f"{key}_{view}"
        for fam, rel in (("multiturn", f"transcripts_results/sdfbase_full22_{tag}/crescendo/none"), ("bon", f"transcripts_results/sdfbase_full22_{tag}/none")):
            root, pr = GD.store_root_of(fam, f"{shared}/{rel}")
            assert pr == f"sdfbase_full22_{tag}" and GD.target_for(fam, pr, tag) == tag
            assert GD.config_for(fam, configs_root=cr, private_run=pr) == f"{cr}/sdfbase_full22_{tag}.yaml"
        assert GD.config_for("petri", configs_root=cr, model_key=tag) == f"{cr}/capeval/manifests/petri_full22_{tag}.yaml"
        assert GD.target_for("padding", f"pad_static_{key}_{view}_p8192", tag) == key                       # the single-phase demo pads per view (pad_static_<key>_<view>_p<pad>)
        assert GD.config_for("padding", configs_root=cr, private_run=f"pad_static_{key}_{view}_p8192") == f"{cr}/sdfbase/static_padding/sdfbase_full22_{key}.padstatic_{view}_p8192.yaml"
    # outside a family's naming: no config, no target, and a store too shallow for the layout names no root
    assert GD.config_for("static", configs_root=cr, private_run="stores") is None and GD.target_for("static", "stores", "k") is None
    assert GD.config_for("padding", configs_root=cr, private_run="sdfbase_full22_qwen35_4b") is None
    assert GD.store_root_of("static", "/a/b") == (None, None) and GD.store_root_of("static", None) == (None, None)
    assert GD.store_root_of("unknown", "/a/b/c/d") == (None, None)


def test_front_door_composes_every_familys_required_arguments_and_each_driver_parses_them(tmp_path):
    """The ACTUAL `08_grade.sh` (neutral site, a cluster block with a grade mirror) composes the daemon's argv; the daemon (dry run) turns
    one pending run per family into one dispatch per family whose command carries EVERY argument the family's driver requires; each
    driver's REAL argparse accepts it (PARSE_PROBE), and every `--config` / `--manifest` the daemon resolved exists. Paper model qwen35_4b:
    the committed configs, every family parses. Front-door model qwen35_4b_demo (configured, not run): with the committed configs root its
    config-bearing families are REFUSED by name (`config_missing`: 05_stage_roster.sh has not generated them) and nothing is dispatched into argparse; with
    a configs root holding the configs stage 05 would write, every family is dispatched and the open-target drivers parse — the static
    and padded graders' `--target choices=` are the generators' closed tables (make_sampled_configs.SAMPLED_VIEWS through sampled_tags();
    make_padding_configs.GRADER_TARGETS == output_keys(): the prompted-only leg's KEYS + every single-phase front-door key with a roster-named
    padded family — never KEYS, which a single-phase key never joins), which admit the model exactly when the production roster carries its
    block: on a tree without it the refusal is asserted by name so it flips visibly; on one with it the composed command parses."""
    import shlex
    import sys as _sys
    root = tmp_path
    mirror = root / "mirror"
    site = write_site(root, clusters=("cluster_a",), substrate="cluster_a", extra_cluster=f"CLUSTER_GRADE_MIRROR_ROOT={mirror}",
                      extra_config=f"SISTEER_REGISTRY_CAMPAIGN={synth.CAMPAIGN}")            # the synthetic runs' campaign (default sdfbase_full22)
    shared = str(root / "shared_cluster_a")
    reg = str(root / "registry")
    rids = _family_registry(reg, shared)
    env = {"PYTHONPATH": str(REPO / "src")}

    def compose(model_key):
        r = run_stage("08_grade.sh", "--dry-run", "--once", "--model", model_key, site=site, env=env, timeout=600)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        out = r.stdout + r.stderr
        # the runner's OWN paths (this checkout, its HOME, the tmp root -- pytest's tmp dir carries the login) are not identifiers
        assert_no_tokens(out.replace(str(REPO), "<repo>").replace(str(root), "<tmp>").replace(os.path.expanduser("~"), "<home>"), f"08_grade.sh {model_key}")
        argv = _dry_run_argv(out)
        text = " ".join(argv)
        # the front door's half of the contract: the store map from the cluster block, the graders' root, the configs root, the model ids
        assert f"--store-root-map cluster_a={shared}={mirror}" in text and "--mirror cluster_a=cluster-a-login" in text
        assert f"--grades-root {root / 'state' / model_key / 'grades'}" in text                # SISTEER_GRADES_ROOT, else <state>/grades
        assert f"--configs-root {REPO / 'configs'}" in text and "--dry-run" in text and "--once" in text
        assert "--extra" not in text                                                          # no half-composed driver args ride the front door
        return argv

    def run_daemon(argv, *extra):
        r = subprocess.run(argv + list(extra), capture_output=True, text=True, env=dict(os.environ, **env), cwd=str(REPO), timeout=900)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        return r

    def events(model_key):
        p = root / "state" / model_key / "grade" / "grade_daemon.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []

    def required_ok(fam, ns, cmd):
        text = " ".join(shlex.quote(c) for c in cmd)
        assert ns["worklist"] == "registry" and ns["registry_root"] == reg and ns["run_id"], (fam, text)
        assert ns["store_root_map"] == [f"cluster_a={shared}={mirror}"], (fam, text)
        if fam in ("static", "padding", "multiturn", "bon"):
            for k in ("target", "config", "private_root", "grades_root"):
                assert ns.get(k), (fam, k, text)
            assert os.path.isfile(ns["config"]), (fam, ns["config"])
            assert ns["private_root"].startswith(str(mirror)), (fam, ns["private_root"])        # the mapped store's root, not a site literal
        if fam in ("multiturn", "bon"):
            assert ns["require_engine"] == "lora_native", (fam, text)                              # FD 19: never lens
        if fam == "bon":
            assert ns["attacks"] == "bon" and ns["rubric"] == "v2", text
        if fam == "static":
            assert ns["attacks"] == "codeattack,pasttense", text                                    # both pending statics in ONE dispatch
        if fam == "padding":
            assert ns["attacks"] == "grok420", text                                                 # ONE (level, attack) per invocation
        if fam == "controls":
            assert ns["controls_root"] == f"{mirror}/exp/sdfbase_full22_native/sampled/controls_cap512", text
        if fam == "agentic":
            assert ns["results_root"].endswith("/exp/misalign/" + ns["_tag"]) and ns["no_escalate"] is True, text
        if fam == "petri":
            assert ns["cmd"] == "analyze" and os.path.isfile(ns["manifest"]), text
            assert ns["out_root"] == f"{mirror}/petri_native/petri_full22_{ns['_tag']}", text          # the audit root above cells/<key>

    # ---- the paper model: every family dispatched with the committed configs; every driver's argparse accepts the command
    argv = compose("qwen35_4b")
    run_daemon(argv)
    ev = events("qwen35_4b")
    disp = {e["family"]: e for e in ev if e["event"] == "dispatch"}
    refused = [e for e in ev if e["event"] == "dispatch_refused"]
    assert not refused, refused
    assert set(disp) == {"static", "padding", "bon", "multiturn", "controls", "agentic", "petri"}, sorted(disp)
    tick = [e for e in ev if e["event"] == "tick"][-1]
    assert tick["pending"] == 8 and tick["dispatched"] == 8 and tick["refused"] == 0 and tick["unknown"] == 0, tick
    assert sorted(disp["static"]["run_ids"]) == sorted([rids[("qwen35_4b_it", "static")][0], rids[("qwen35_4b_it", "static2")][0]])
    for fam, e in sorted(disp.items()):
        cmd = shlex.split(e["cmd"])
        rc, ns, err = _probe(cmd)
        assert rc == 0 and ns is not None, (fam, e["cmd"], err[-1500:])
        ns["_tag"] = "qwen35_4b"
        required_ok(fam, ns, cmd)
    # the mirror lines: every mapped store (all eight runs sit under the cluster's shared root) is rsync'd from the cluster block's
    # alias into the mirror before its driver runs
    mirrors = [e for e in ev if e["event"] == "mirror"]
    assert len(mirrors) == 8 and all("cluster-a-login:" in e["cmd"] and str(mirror) in e["cmd"] for e in mirrors), mirrors[:2]

    # ---- the front-door model with THIS tree's configs root: every family whose config stage 05 has not generated here is refused
    # by name (config_missing), nothing dispatched into argparse; a family whose config IS on the tree dispatches (TREE-CONDITIONAL, the
    # fix-1 owner follow-up 10: on the worktree today none of the five exists; on a tree where 05 --step configs ran, all five do)
    argv_t = compose("qwen35_4b_demo")
    run_daemon(argv_t)
    ev_t = events("qwen35_4b_demo")
    ref_t = {e["family"]: e for e in ev_t if e["event"] == "dispatch_refused"}
    disp_t = {e["family"]: e for e in ev_t if e["event"] == "dispatch"}
    generated = {"static": "sdfbase_full22_qwen35_4b_demo_battery_sampled_t0.yaml", "padding": "sdfbase/static_padding/sdfbase_full22_qwen35_4b_demo.padstatic_battery_p0.yaml",
                 "bon": "sdfbase_full22_qwen35_4b_demo_battery.yaml", "multiturn": "sdfbase_full22_qwen35_4b_demo_battery.yaml",
                 "petri": "capeval/manifests/petri_full22_qwen35_4b_demo_battery.yaml"}
    missing = {fam for fam, rel in generated.items() if not (REPO / "configs" / rel).is_file()}
    assert set(ref_t) == missing and set(disp_t) == {"controls", "agentic"} | (set(generated) - missing), (sorted(ref_t), sorted(disp_t))
    for fam, e in ref_t.items():
        assert e["reason"] == "config_missing" and "05_stage_roster.sh" in e["detail"], e
    if missing != set(generated):
        return                                          # the tree already holds the configs: the configs-root branch below is what ran
    assert f"{REPO}/configs/sdfbase_full22_qwen35_4b_demo_battery.yaml" in ref_t["multiturn"]["detail"]
    assert f"{REPO}/configs/sdfbase_full22_qwen35_4b_demo_battery_sampled_t0.yaml" in ref_t["static"]["detail"]
    assert f"{REPO}/configs/sdfbase/static_padding/sdfbase_full22_qwen35_4b_demo.padstatic_battery_p0.yaml" in ref_t["padding"]["detail"]
    assert f"{REPO}/configs/capeval/manifests/petri_full22_qwen35_4b_demo_battery.yaml" in ref_t["petri"]["detail"]
    tick_t = [e for e in ev_t if e["event"] == "tick"][-1]
    assert tick_t["pending"] == 8 and tick_t["refused"] == 6 and tick_t["dispatched"] == 2, tick_t

    # ---- the front-door model with the configs stage 05 would generate (a configs root holding them): every family dispatched
    cfgs = root / "configs"
    (cfgs / "sdfbase" / "static_padding").mkdir(parents=True)
    (cfgs / "capeval" / "manifests").mkdir(parents=True)
    for rel in ("sdfbase_full22_qwen35_4b_demo_battery.yaml", "sdfbase_full22_qwen35_4b_demo_battery_sampled_t0.yaml",
                "sdfbase/static_padding/sdfbase_full22_qwen35_4b_demo.padstatic_battery_p0.yaml", "capeval/manifests/petri_full22_qwen35_4b_demo_battery.yaml"):
        (cfgs / rel).write_text("private_run: synthetic\n")
    (root / "state" / "qwen35_4b_demo" / "grade" / "grade_daemon.jsonl").unlink()
    run_daemon(argv_t, "--configs-root", str(cfgs))                                          # the last --configs-root wins: the same argv otherwise
    ev_t2 = events("qwen35_4b_demo")
    disp_t2 = {e["family"]: e for e in ev_t2 if e["event"] == "dispatch"}
    assert not [e for e in ev_t2 if e["event"] == "dispatch_refused"]
    assert set(disp_t2) == {"static", "padding", "bon", "multiturn", "controls", "agentic", "petri"}, sorted(disp_t2)
    _sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22" / "sampled"))
    _sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22" / "static_padding"))
    import make_padding_configs as _mpc
    import make_sampled_configs as _msc
    closed = {"static": ("qwen35_4b_demo_battery", tuple(_msc.sampled_tags()), "make_sampled_configs.SAMPLED_VIEWS"),
              "padding": ("qwen35_4b_demo", tuple(_mpc.GRADER_TARGETS), "make_padding_configs.GRADER_TARGETS")}   # the grader's gate, never KEYS
    for fam, e in sorted(disp_t2.items()):
        cmd = shlex.split(e["cmd"])
        rc, ns, err = _probe(cmd)
        if fam in closed and closed[fam][0] not in closed[fam][1]:
            # the composed command is complete; the ONE refusal is the generator's closed target table (named here so the day the
            # table admits the model this assertion fails and the branch below takes over)
            assert rc == 2 and ns is None and "--target" in err and "invalid choice" in err and closed[fam][0] in err, (fam, closed[fam][2], err[-800:])
            text = e["cmd"]
            for flag in ("--target", "--config", "--private-root", "--grades-root", "--attacks", "--worklist registry", "--run-id", "--store-root-map"):
                assert flag in text, (fam, flag, text)
            continue
        assert rc == 0 and ns is not None, (fam, e["cmd"], err[-1500:])
        ns["_tag"] = "qwen35_4b_demo_battery"
        required_ok(fam, ns, cmd)
    assert disp_t2["multiturn"]["private_run"] == "sdfbase_full22_qwen35_4b_demo_battery" and disp_t2["bon"]["private_run"] == "sdfbase_full22_qwen35_4b_demo_battery"


def test_every_target_bearing_familys_lane_admits_the_daemons_target_for_the_worked_examples_private_runs(tmp_path):
    """The daemon's `target_for` named the demo key for its padded runs while the padded grader's two gates
    (`--target choices=` and run_identity) were bound to the PROMPTED-ONLY leg's KEYS, which a single-phase key never joins — every padding
    dispatch of the smoke returned argparse's rc 2, and no test fed `target_for`'s output to the lane's parser. The contract: for EVERY
    DEFAULT_TEMPLATES family whose template carries `--target`, every `target_for` output for the worked example's private runs (the
    generators' own spellings — the sampled lane's configs, the padded lane's per-view configs, the wave configs of the declared wave and the
    dose views, each committed on this tree by 05 --step configs) is inside that lane's admitted targets: the sampled and padded lanes'
    `--target choices=` tables (grade_static_sampled.KEYS, grade_padded_static.PaddedStaticLane.targets) and grade_full22_native's
    resolve_targets() (no `choices=`; its post-parse admission). And the daemon's composed command for one private run per (family,
    target) parses through the driver's REAL argparse (PARSE_PROBE) with exactly that target. A family gaining `--target` with no lane
    named here fails by name."""
    import importlib.util
    import shlex
    import sys as _sys

    import yaml
    KEY = "qwen35_4b_demo"
    for p in (REPO / "scripts" / "sdfbase" / "full22" / "sampled", REPO / "scripts" / "sdfbase" / "full22" / "static_padding", REPO / "reproduce" / "lib"):
        if str(p) not in _sys.path:
            _sys.path.insert(0, str(p))
    import make_padding_configs as _mpc
    import make_sampled_configs as _msc
    import models as _M

    def _load(rel, name):
        spec = importlib.util.spec_from_file_location(name, REPO / rel)
        mod = importlib.util.module_from_spec(spec)
        _sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    gss = _load("scripts/sdfbase/full22/sampled/grade_static_sampled.py", "gd_contract_grade_static_sampled")
    gps = _load("scripts/sdfbase/full22/static_padding/grade_padded_static.py", "gd_contract_grade_padded_static")
    twin = _load("scripts/sdfbase/seedrep/grade_full22_native.py", "gd_contract_grade_full22_native")
    native, notes = twin.resolve_targets(REPO / "configs" / "sdfbase_full22_roster.yaml", REPO / "reproduce" / "models.yaml")
    assert notes == [], notes
    lanes = {"static": ("grade_static_sampled.KEYS", set(gss.KEYS)),
             "padding": ("grade_padded_static.PaddedStaticLane.targets", set(gps.PaddedStaticLane.targets)),
             "bon": ("grade_full22_native.resolve_targets", set(native)),
             "multiturn": ("grade_full22_native.resolve_targets", set(native))}
    target_families = {f for f, t in GD.DEFAULT_TEMPLATES.items() if "--target" in t}
    assert target_families == set(lanes), (sorted(target_families), sorted(lanes))        # a family gaining --target names its lane here
    assert gps.PaddedStaticLane.targets == gps.TARGETS == tuple(_mpc.GRADER_TARGETS) == _mpc.output_keys() and KEY not in _mpc.KEYS   # the grader's gate is the generator's output set, never KEYS
    views = _M.phase_views(KEY)
    assert views == ["battery", "d0p75", "d0p75_relu"], views
    # the worked example's private runs, by the generators' spellings: (private_run, eval_id, the store ROOT the daemon hands the driver)
    runs = {fam: [] for fam in lanes}
    for path, tag, _trial, _esc in _msc.outputs(_msc.load_presets()):
        if _msc.key_of(tag) == KEY:
            assert path.is_file() and yaml.safe_load(path.read_text())["private_run"] == path.stem, path
            runs["static"].append((path.stem, "pasttense", "/m/exp/sdfbase_full22_native/transcripts_results"))
    for fam in _mpc.families_of(KEY):
        for pad in _mpc.PADS:
            assert _mpc.padded_config(KEY, pad, fam).is_file(), (fam, pad)
            runs["padding"].append((_mpc.private_run(KEY, pad, fam), "grok420", "/m/exp/sdfbase_full22_native/padded"))
    for v in views:
        pr = f"sdfbase_full22_{KEY}_{v}"
        assert (REPO / "configs" / f"{pr}.yaml").is_file(), pr
        runs["bon"].append((pr, "bon", "/m/exp/sdfbase_full22_native/transcripts_results"))
        runs["multiturn"].append((pr, "crescendo", "/m/exp/sdfbase_full22_native/transcripts_results"))
    assert len(runs["static"]) == 6 * len(views) and len(runs["padding"]) == len(_mpc.PADS) * len(views) and len(runs["bon"]) == len(runs["multiturn"]) == len(views)
    fields = dict(python=sys.executable, repo=str(REPO), registry_root=str(tmp_path), store_root_map_specs=[], extra="", model_key=KEY,
                  configs_root=str(REPO / "configs"), grades_root=str(tmp_path / "g"))
    probed = set()
    for fam, items in sorted(runs.items()):
        name, admitted = lanes[fam]
        for i, (pr, eval_id, root) in enumerate(items):
            target = GD.target_for(fam, pr, KEY)
            want = KEY if fam == "padding" else GD._FULL22_RUN_RE.match(pr).group("tag")
            assert target == want and target.startswith(KEY), (fam, pr, target)
            assert target in admitted, (fam, pr, target, name, sorted(t for t in admitted if t.startswith(KEY)))
            if (fam, target) in probed:
                continue
            probed.add((fam, target))
            rows = [{"run_id": f"11111111-2222-3333-4444-{i:012d}", "eval_id": eval_id, "family": fam, "store_root": root, "private_run": pr,
                     "model_id": KEY, "campaign": synth.CAMPAIGN}]
            cmd = GD.build_command(GD.DEFAULT_TEMPLATES[fam], rows, **fields)
            assert cmd[cmd.index("--target") + 1] == target and os.path.isfile(cmd[cmd.index("--config") + 1]), (fam, cmd)
            rc, ns, err = _probe(cmd)
            assert rc == 0 and ns is not None and ns["target"] == target, (fam, pr, " ".join(shlex.quote(c) for c in cmd), err[-1500:])
    assert {f for f, _t in probed} == set(lanes) and len(probed) == 1 + 3 * len(views)


def test_daemon_refuses_a_group_it_cannot_supply_a_placeholder_for_and_leaves_it_pending(tmp_path):
    """A template placeholder the group cannot supply (a run with no store path on this host; a private run outside the family's naming;
    a missing --grades-root) refuses the group by name with its run ids, claims nothing and leaves the runs pending; a template naming a
    placeholder the daemon does not define is refused as such."""
    root, rids = _root(tmp_path)
    fields = dict(python=sys.executable, repo=str(REPO), registry_root=root, store_root_map_specs=[], extra="", model_key="k")
    rows = [{"run_id": "r1", "eval_id": "pasttense", "family": "static", "store_root": None, "private_run": None, "model_id": "m", "campaign": "c"}]
    with pytest.raises(GD.DispatchRefused) as ei:
        GD.build_command(GD.DEFAULT_TEMPLATES["static"], rows, configs_root=str(REPO / "configs"), grades_root="/g", **fields)
    assert ei.value.reason == "placeholder_unresolved" and "{store_root}" in ei.value.detail and "{config}" in ei.value.detail
    rows2 = [dict(rows[0], store_root="/m/transcripts_results", private_run="sdfbase_full22_qwen35_4b_sampled_t0")]
    with pytest.raises(GD.DispatchRefused) as ei:
        GD.build_command(GD.DEFAULT_TEMPLATES["static"], rows2, configs_root=str(REPO / "configs"), grades_root=None, **fields)
    assert ei.value.reason == "placeholder_unresolved" and "{grades_root}" in ei.value.detail and "--grades-root" in ei.value.detail
    cmd = GD.build_command(GD.DEFAULT_TEMPLATES["static"], rows2, configs_root=str(REPO / "configs"), grades_root="/g", **fields)
    assert cmd[1].endswith("grade_static_sampled.py") and "--config" in cmd and "--private-root" in cmd and "/m/transcripts_results" in cmd
    with pytest.raises(GD.DispatchRefused) as ei:
        GD.build_command(GD.DEFAULT_TEMPLATES["static"], [dict(rows2[0], private_run="stores")], configs_root=str(REPO / "configs"), grades_root="/g", **fields)
    assert ei.value.reason == "placeholder_unresolved" and "{target}" in ei.value.detail
    with pytest.raises(GD.DispatchRefused) as ei:
        GD.build_command("{python} x.py {nope}", rows2, configs_root=str(REPO / "configs"), grades_root="/g", **fields)
    assert ei.value.reason == "unknown_placeholder" and "nope" in ei.value.detail
    # through the daemon: a run whose store the layout cannot place is refused and left pending (nothing claimed)
    r = _run_daemon(root, tmp_path, "--driver", "static=" + GD.DEFAULT_TEMPLATES["static"], "--grades-root", str(tmp_path / "g"), dry_run=True)
    assert r.returncode == 0, r.stderr + r.stdout
    ev = _events(tmp_path)
    ref = [e for e in ev if e["event"] == "dispatch_refused"]
    assert len(ref) == 1 and ref[0]["reason"] == "placeholder_unresolved" and sorted(ref[0]["run_ids"]) == sorted([rids["job7@none"], rids["job7@steer"]]), ref
    assert not [e for e in ev if e["event"] == "dispatch"]
    with DB.reading(DB.db_path(root)) as conn:
        assert {r_[0] for r_ in conn.execute("SELECT lifecycle_state FROM run").fetchall()} == {"generated"}
        assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 0


def test_a_reclaim_row_whose_claims_completion_is_already_on_the_spool_gets_its_hold_back_instead_of_a_re_dispatch(tmp_path, monkeypatch):
    """A run claimed by this daemon's graders' session with no grader running reads as a dead grader's (`reclaim`)
    -- but when the claim's COMPLETE judgment is already on the spool the grader FILED and the daemon died before `spooled_hold_add` (or the
    mirror claim was released first; `mirror_reconcile_claims` settles the mirror only). Re-dispatching it would judge the run again under
    the same id. The tick now re-establishes the hold from the spool, bound to that judgment id and the store generation, and the hold
    filter keeps the run out of the dispatch until the fold; a reclaim row with no such record stays reclaimable."""
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, rids = _root(tmp_path)
    GD = _load_daemon_module(); st = tmp_path / "state"; st.mkdir()
    from sisteer.registry import identity as I
    from sisteer.registry import pending as P
    rid = rids["job7@none"]
    own = "fake-driver"                                                              # this harness's daemon runs its graders under --grader-session fake-driver
    rows0 = GD.pending_rows(root, [synth.MODEL_ID], synth.CAMPAIGN, None, {synth.EVAL_ID: "static"}, grader_session=own)
    (row0,) = [r for r in rows0 if r["run_id"] == rid]
    jid = P.claim(root, rid, str(row0["required_judge_id"]), own)                    # the grader's claim under this daemon's graders' session
    rows = GD.pending_rows(root, [synth.MODEL_ID], synth.CAMPAIGN, None, {synth.EVAL_ID: "static"}, grader_session=own)
    (row,) = [r for r in rows if r["run_id"] == rid]
    assert row.get("reclaim") is True and row["claim_judgment_id"] == jid
    # (a) nothing on the spool: the grader died before filing -- reclaimable, nothing rebuilt, nothing held
    assert GD.spooled_rebuild(str(st), root, rows) == []
    eligible, held = GD.spooled_filter(str(st), root, rows)
    assert rid in [r_["run_id"] for r_ in eligible] and held == []
    # (b) the grader FILED under the claim's id and the hold was never bound: rebuilt from the spool, held
    d = Path(root) / "records" / synth.CAMPAIGN / I.run_dir_name(rid); d.mkdir(parents=True, exist_ok=True)
    (d / "000950-judgment.jsonl").write_text(json.dumps(R.make_record("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": str(row0["required_judge_id"]),
                                                                                      "status": "complete", "n_items_judged": 200, "claimed_by_session": own},
                                                                       {"script": "t"}, "rec-g21")) + "\n")
    assert GD.spooled_rebuild(str(st), root, rows) == [rid]
    ledger = json.loads((st / "dispatch_spooled.json").read_text())["runs"]
    assert ledger[rid]["judgment_ids"] == [jid] and "|200|" in ledger[rid]["store_generation"] and ledger[rid]["dispatch_utc"] == row["claim_utc"]
    ev = [json.loads(l) for l in (st / "grade_daemon.jsonl").read_text().splitlines()]
    (rb,) = [e for e in ev if e["event"] == "spooled_hold_rebuilt"]; assert rb["run_ids"] == [rid] and rb["judgment_ids"] == {rid: [jid]} and rb["dry_run"] is False
    eligible, held = GD.spooled_filter(str(st), root, rows)
    assert [r_["run_id"] for r_ in held] == [rid] and rid not in [r_["run_id"] for r_ in eligible]
    assert GD.spooled_rebuild(str(st), root, rows) == []                               # idempotent: the hold stands, nothing rebuilt twice
    # end to end: a fresh tick (the ledger gone, as after the death) rebuilds the hold itself, reports no reclaim and dispatches only the other run
    (st / "dispatch_spooled.json").unlink()
    n0 = len(_events(tmp_path))
    r = _run_daemon(root, tmp_path, *_mirror_args(tmp_path)); assert r.returncode == 0, r.stderr + r.stdout
    ev2 = _events(tmp_path)[n0:]
    tick = [e for e in ev2 if e["event"] == "tick"][-1]
    assert tick["spooled_rebuilt"] == 1 and tick["spooled_hold"] == 1 and tick["reclaimed"] == 0, tick
    assert [e for e in ev2 if e["event"] == "spooled_hold_rebuilt"][0]["run_ids"] == [rid]
    dispatched_runs = [x for e in ev2 if e["event"] == "dispatch" for x in e.get("run_ids", [])]
    assert rid not in dispatched_runs and rids["job7@steer"] in dispatched_runs, ev2
    assert json.loads((st / "dispatch_spooled.json").read_text())["runs"][rid]["judgment_ids"] == [jid]


def test_the_agentic_dispatch_carries_the_registrys_model_identity_for_the_rows_step(tmp_path):
    """The misalign grader's rows step files model_id / model_revision from the judged root's identity files -- which the
    grading host's mirror does not carry (the 2026-10-02 agentic pass judged 2995 items, then failed its rows step on "no model
    identity" and quarantined its two partial runs). The daemon now passes the registry's identity (the model table's hf_repo /
    hf_revision for the group's model key) on the agentic template; a model the table does not know yields no flags (the grader
    refuses by name as before)."""
    root, rids = _root(tmp_path)
    GD = _load_daemon_module()
    with DB.reading(DB.db_path(root)) as conn:
        hf_repo, hf_rev = conn.execute("SELECT hf_repo, hf_revision FROM model WHERE model_id = ?", (synth.MODEL_ID,)).fetchone()
    assert hf_repo and hf_rev
    rows = GD.pending_rows(root, [synth.MODEL_ID], synth.CAMPAIGN, None, {synth.EVAL_ID: "static"})
    fields = GD.group_fields(rows, python=sys.executable, repo=str(REPO), registry_root=root, store_root_map_specs=[], model_key="k",
                             configs_root=str(REPO / "configs"), grades_root=None)
    assert fields["model_identity_flags"] == f"--identity-model-id {hf_repo} --identity-model-revision {hf_rev}"
    (agentic_template,) = [t["agentic"] for t in vars(GD).values() if isinstance(t, dict) and isinstance(t.get("agentic"), str) and "misalign_judge.py" in t["agentic"]]
    # the flags are the grader's CROSS-CHECK input; the identity it files comes from the root's files, which the mirror now carries
    assert "{model_identity_flags}" in agentic_template and agentic_template.index("{model_identity_flags}") < agentic_template.index("{extra}")
    unknown = [dict(rows[0], model_id="no-such-model")]
    assert GD.group_fields(unknown, python=sys.executable, repo=str(REPO), registry_root=root, store_root_map_specs=[], model_key="k",
                           configs_root=str(REPO / "configs"), grades_root=None)["model_identity_flags"] == ""


def test_the_mirror_copies_the_misalign_roots_identity_files_beside_the_stores(tmp_path, monkeypatch):
    """The misalign rows step files the SERVED identity from the judged root's files (cells_full.json /
    endpoint.snapshot.json / serving.json / serving/endpoint.recorded.json); the mirror copied the rows' store paths only, so the grading
    host had no served identity and the registry's would have been filed in its place. The mirror now rsyncs the root's identity files once
    per root beside the stores (one --files-from call, missing files skipped), logged as mirror_identity; a family without root files
    (static) adds no call."""
    GD = _load_daemon_module(); st = tmp_path / "state"; st.mkdir()
    log = tmp_path / "rsync_calls.log"; monkeypatch.setenv("FAKE_RSYNC_LOG", str(log))
    fake = tmp_path / "rsync"; fake.write_text(FAKE_RSYNC); fake.chmod(0o755)
    mirror = tmp_path / "mirror"
    rows = [{"run_id": "sdfbase_full22:agentic_leaking:aaaa:1111", "family": "agentic", "substrate": "cluster_a",
             "store_uri": "/cluster_a/user/exp/misalign/qwen35_4b_demo_s2/agentic/none", "store_path": f"{mirror}/exp/misalign/qwen35_4b_demo_s2/agentic/none"},
            {"run_id": "sdfbase_full22:agentic_murder:bbbb:1111", "family": "agentic", "substrate": "cluster_a",
             "store_uri": "/cluster_a/user/exp/misalign/qwen35_4b_demo_s2/agentic/none", "store_path": f"{mirror}/exp/misalign/qwen35_4b_demo_s2/agentic/none"},
            {"run_id": "sdfbase_full22:codeattack:cccc:1111", "family": "static", "substrate": "cluster_a",
             "store_uri": "/cluster_a/user/exp/sdfbase_full22_native/sampled/x/codeattack/none", "store_path": f"{mirror}/exp/sdfbase_full22_native/sampled/x/codeattack/none"}]
    out = GD.mirror_stores(rows, {"cluster_a": "cluster-a-login"}, str(fake), False, str(st), {"cluster_a": str(mirror)}, min_free_bytes=0)
    assert [m["rc"] for m in out] == [0, 0, 0]
    calls = log.read_text().splitlines()
    ident = [c for c in calls if "--files-from=-" in c]
    assert len(ident) == 1, calls                                                             # ONE identity copy for the one misalign root (two runs)
    assert "cluster-a-login:/cluster_a/user/exp/misalign/qwen35_4b_demo_s2/" in ident[0] and ident[0].rstrip().endswith(f"{mirror}/exp/misalign/qwen35_4b_demo_s2/")
    assert "--ignore-missing-args" in ident[0] and "sdfbase_full22_native" not in ident[0]
    ev = [json.loads(l) for l in (st / "grade_daemon.jsonl").read_text().splitlines()]
    (mi,) = [e for e in ev if e["event"] == "mirror_identity"]
    assert mi["family"] == "agentic" and mi["files"] == list(GD.ROOT_IDENTITY_FILES["agentic"]) and mi["rc"] == 0 and mi["local_root"] == f"{mirror}/exp/misalign/qwen35_4b_demo_s2"


def test_an_operator_dispatch_hold_leaves_every_pending_row_untouched_and_lifts_when_the_file_goes(tmp_path, monkeypatch):
    """While <state>/dispatch.hold exists a tick dispatches no grader and emits no contrast -- the pending rows are left exactly
    as they are, one `held` event carries the file's reason and the pending count, the summary says held -- and the next tick without the
    file dispatches as before. The static correction op writes the file before re-registering units whose stores already carry
    complete judgments, and removes it after the re-filed judgments fold."""
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, rids = _root(tmp_path)
    st = tmp_path / "state"; st.mkdir()
    (st / "dispatch.hold").write_text("# D1255: static correction op in flight\nre-filing judgments under rung-keyed run ids\n")
    r = _run_daemon(root, tmp_path, *_mirror_args(tmp_path)); assert r.returncode == 0, r.stderr + r.stdout
    ev = _events(tmp_path)
    (held,) = [e for e in ev if e["event"] == "held"]
    assert held["reason"] == "re-filing judgments under rung-keyed run ids" and held["pending"] == 2 and sorted(held["run_ids"]) == sorted(rids[k] for k in ("job7@none", "job7@steer"))
    tick = [e for e in ev if e["event"] == "tick"][-1]
    assert tick["held"] is True and tick["pending"] == 2 and tick["dispatched"] == 0 and tick["reclaimed"] == 0, tick
    assert tick["hold"] == "operator" and tick["held_reason"] == held["reason"] and held["hold"] == "operator"     # the hold's KIND beside the flag
    assert not [e for e in ev if e["event"] in ("dispatch", "dispatched", "mirror", "reclaim_own_session")], ev
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 0                   # nothing claimed, nothing judged
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rids[k],)).fetchone()[0] == "generated" for k in ("job7@none", "job7@steer"))
    (st / "dispatch.hold").unlink()
    n = len(ev)
    r2 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path)); assert r2.returncode == 0, r2.stderr + r2.stdout
    ev2 = _events(tmp_path)[n:]
    tick2 = [e for e in ev2 if e["event"] == "tick"][-1]
    assert tick2["held"] is False and tick2["dispatched"] == 2 and not [e for e in ev2 if e["event"] == "held"], tick2


FAKE_RSYNC_IDENTITY_FAILS = FAKE_RSYNC.replace('dst="${@: -1}"', 'case " $* " in *" --files-from=- "*) echo "rsync: identity transfer failed (synthetic)" >&2; exit 12;; esac\ndst="${@: -1}"')
#: a fake rsync whose identity transfer (`--files-from=-`) copies the named files from $FAKE_RSYNC_IDENTITY_SRC (a directory standing
#: in for the cluster's judged root) into the destination, or fails after the daemon's delete while $FAKE_RSYNC_IDENTITY_FAIL is set
FAKE_RSYNC_IDENTITY_SOURCE = FAKE_RSYNC.replace('dst="${@: -1}"', '''case " $* " in *" --files-from=- "*)
  [ -z "${FAKE_RSYNC_IDENTITY_FAIL:-}" ] || { echo "rsync: identity transfer failed (synthetic)" >&2; exit 12; }
  dst="${@: -1}"; mkdir -p "$dst"; echo "$*" >> "${FAKE_RSYNC_LOG:?}"
  while IFS= read -r f; do [ -z "$f" ] && continue; if [ -f "${FAKE_RSYNC_IDENTITY_SRC:?}/$f" ]; then mkdir -p "$(dirname "$dst/$f")"; cp "${FAKE_RSYNC_IDENTITY_SRC}/$f" "$dst/$f"; fi; done
  exit 0;; esac
dst="${@: -1}"''')


def test_a_failed_or_sourceless_identity_mirror_never_leaves_a_stale_identity_and_blocks_the_groups_dispatch(tmp_path, monkeypatch):
    """mirror_root_identity's rc was ignored and a stale local cells_full.json survived a transfer the
    source no longer carried (--ignore-missing-args), so a root reused by another served revision could file the current stores under the
    old snapshot through rows.resolve_model_identity's candidate order. Now the enumerated local candidates are deleted BEFORE every
    transfer (a failed or sourceless transfer leaves none -- the grader refuses by name rather than reading stale), the rc rides the
    group's mirrored entries, and the tick does not dispatch a group whose identity transfer failed."""
    GD = _load_daemon_module(); st = tmp_path / "state"; st.mkdir()
    log = tmp_path / "rsync_calls.log"; monkeypatch.setenv("FAKE_RSYNC_LOG", str(log))
    ok_rsync = tmp_path / "rsync"; ok_rsync.write_text(FAKE_RSYNC); ok_rsync.chmod(0o755)
    bad_rsync = tmp_path / "rsync_idfail"; bad_rsync.write_text(FAKE_RSYNC_IDENTITY_FAILS); bad_rsync.chmod(0o755)
    mirror = tmp_path / "mirror"; local_root = mirror / "exp" / "misalign" / "qwen35_4b_demo_s2"; local_root.mkdir(parents=True)
    files = GD.ROOT_IDENTITY_FILES["agentic"]
    # (a) a transfer whose source lacks the formerly preferred cells_full.json: the stale local copy is gone afterwards, not kept
    (local_root / "cells_full.json").write_text(json.dumps({"endpoint": {"model": "org/model", "model_revision": "0" * 40}}))
    rc = GD.mirror_root_identity("cluster-a-login", "/cluster_a/user/exp/misalign/qwen35_4b_demo_s2", str(local_root), files, str(ok_rsync), str(st), "agentic")
    assert rc == 0 and not (local_root / "cells_full.json").exists()
    ev = [json.loads(l) for l in (st / "grade_daemon.jsonl").read_text().splitlines()]
    assert ev[-1]["event"] == "mirror_identity" and ev[-1]["removed_stale"] == ["cells_full.json"] and ev[-1]["present"] == []
    # (b) a failed transfer: the stale candidate is gone too, the rc is nonzero
    (local_root / "cells_full.json").write_text(json.dumps({"endpoint": {"model": "org/model", "model_revision": "0" * 40}}))
    rc = GD.mirror_root_identity("cluster-a-login", "/cluster_a/user/exp/misalign/qwen35_4b_demo_s2", str(local_root), files, str(bad_rsync), str(st), "agentic")
    assert rc == 12 and not (local_root / "cells_full.json").exists()
    # (c) through mirror_stores: the identity rc rides every entry of the root, and identity_mirror_blocked names the root -> the tick skips the group
    rows = [{"run_id": "sdfbase_full22:agentic_leaking:aaaa:1111", "family": "agentic", "substrate": "cluster_a",
             "store_uri": "/cluster_a/user/exp/misalign/qwen35_4b_demo_s2/agentic/none", "store_path": f"{local_root}/agentic/none"},
            {"run_id": "sdfbase_full22:agentic_murder:bbbb:1111", "family": "agentic", "substrate": "cluster_a",
             "store_uri": "/cluster_a/user/exp/misalign/qwen35_4b_demo_s2/agentic/none", "store_path": f"{local_root}/agentic/none"}]
    out = GD.mirror_stores(rows, {"cluster_a": "cluster-a-login"}, str(bad_rsync), False, str(st), {"cluster_a": str(mirror)}, min_free_bytes=0)
    assert [m["rc"] for m in out] == [0, 0] and [m["identity_rc"] for m in out] == [12, 12] and {m["identity_root"] for m in out} == {str(local_root)}
    assert GD.identity_mirror_blocked(out) == [str(local_root)]
    out_ok = GD.mirror_stores(rows, {"cluster_a": "cluster-a-login"}, str(ok_rsync), False, str(st), {"cluster_a": str(mirror)}, min_free_bytes=0)
    assert [m["identity_rc"] for m in out_ok] == [0, 0] and GD.identity_mirror_blocked(out_ok) == []
    assert GD.identity_mirror_blocked([{"run_id": "x", "rc": 0}]) == []                      # a family without root files blocks nothing


def test_a_hold_that_appears_while_an_earlier_group_grades_stops_every_later_group_in_the_same_tick(tmp_path, monkeypatch):
    """The hold was read once at tick start; a hold created while group A's grader ran (hours, in
    production) left group B dispatched and the contrasts emitted on the stale flag. Now the hold is re-read before every group and
    again right before each grader starts: the fake grader of group A writes the hold file, group B is not dispatched, the `held` event
    says mid-tick and names the stopped group, the summary says held; the stopped runs keep their generated lifecycle with no judgment.
    Two groups, split by the VIEW (the group key's model key): OTHER_MODEL's run under its own view sorts first (group A, whose fake
    grader writes the hold) and the synthetic model's pair under synthmodel_4b_s1 is group B -- the static layout folds every store
    directory under one mirror into ONE root / private run, so a second store directory alone is not a second group."""
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), _dims_with_other_model())
    rids = {}
    specs = (("job7@none", "none", "none", synth.MODEL_ID, "synthmodel_4b_s1"), ("job7@steer", "lora@abcd", synth.OBJECT_SHA, synth.MODEL_ID, "synthmodel_4b_s1"),
             ("job8@other", "none", "none", OTHER_MODEL, "othermodel_9b_s1"))
    for k, (key, okey, oid, model, view) in enumerate(specs):
        rid, recs = synth.run_records(key, 100 + 20 * k, object_key=okey, object_id=oid, through="generated", model_id=model,
                                      view_legacy=view, cluster_job_id="7" if key.startswith("job7") else "8")
        recs[0]["payload"]["store_uri"] = f"/cluster_a/user/exp/stores/{key.replace('@', '_')}"
        R.write_records(os.path.join(root, "records", "c", f"00000{k + 1}-run.jsonl"), recs)
        rids[key] = rid
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    st = tmp_path / "state"; st.mkdir()
    r = _run_daemon(root, tmp_path, *_mirror_args(tmp_path), "--model-id", OTHER_MODEL, env_extra={"FAKE_DRIVER_WRITE_HOLD": str(st / "dispatch.hold")})
    assert r.returncode == 0, r.stderr + r.stdout
    ev = _events(tmp_path)
    pair = sorted([rids["job7@none"], rids["job7@steer"]])
    # group A (the other model's view) was dispatched and its grader ran to completion -- already-running work finishes
    dispatches = [e for e in ev if e["event"] == "dispatch"]
    assert len(dispatches) == 1 and dispatches[0]["run_ids"] == [rids["job8@other"]] and dispatches[0]["model_key"] == "othermodel_9b_s1", dispatches
    calls = [json.loads(l) for l in (tmp_path / "driver_calls.jsonl").read_text().splitlines()]
    assert len(calls) == 1 and calls[0]["run_ids"] == [rids["job8@other"]], calls
    assert [e["rc"] for e in ev if e["event"] == "dispatched"] == [0], ev
    # group B (the synthetic pair) was stopped by the hold group A's grader wrote: one mid-tick `held` event, seen before B's mirror and claim
    (held,) = [e for e in ev if e["event"] == "held"]
    assert held["mid_tick"] is True and held["reason"] == "hold from the fake grader" and "before" in held["where"], held
    assert [(g["model_key"], sorted(g["run_ids"])) for g in held["stopped_groups"]] == [("synthmodel_4b_s1", pair)] and held["pending"] == 2, held
    tick = [e for e in ev if e["event"] == "tick"][-1]
    assert tick["held"] is True and tick["dispatched"] == 1 and tick["pending"] == 3, tick
    assert [e["run_id"] for e in ev if e["event"] == "mirror"] == [rids["job8@other"]], ev           # B was never mirrored, let alone claimed
    with DB.reading(DB.db_path(root)) as conn:
        for k in ("job7@none", "job7@steer"):
            assert conn.execute("SELECT COUNT(*) FROM judgment WHERE run_id = ?", (rids[k],)).fetchone()[0] == 0
            assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rids[k],)).fetchone()[0] == "generated"
    assert (st / "dispatch.hold").exists()                                                          # the operator's file is never consumed by the daemon


def test_the_agentic_dispatch_names_its_own_identity_snapshot_taken_under_the_roots_identity_lock(tmp_path, monkeypatch):
    """The pieces: the snapshot directory sits beside the dispatch log and is keyed to the
    dispatch (`<log>.<pid>.identity`); the agentic template hands it to the grader as --served-identity-dir (a group without one is refused
    by name); snapshot_served_identity copies the root's identity files with their layout, never over an existing directory; and
    mirror_root_identity runs delete + transfer + snapshot in ONE section under the root's identity lock -- a held lock means `locked`,
    nothing deleted, nothing snapshotted."""
    import fcntl
    GD = _load_daemon_module(); st = tmp_path / "state"; st.mkdir()
    log = tmp_path / "rsync_calls.log"; monkeypatch.setenv("FAKE_RSYNC_LOG", str(log))
    log_path = GD._dispatch_log_path(str(st), "agentic", "synthmodel_4b_s1")
    snap = GD.served_identity_dir_for(log_path)
    assert snap.startswith(log_path[:-len(".log")] + ".") and snap.endswith(f".{os.getpid()}.identity") and os.path.dirname(snap) == str(st / "dispatch_logs")
    root, rids = _root(tmp_path)
    rows = GD.pending_rows(root, [synth.MODEL_ID], synth.CAMPAIGN, None, {synth.EVAL_ID: "agentic"})
    fields = dict(python=sys.executable, repo=str(REPO), registry_root=root, store_root_map_specs=[], extra="", model_key="synthmodel_4b_s1")
    cmd = GD.build_command(GD.DEFAULT_TEMPLATES["agentic"], rows, served_identity_dir=snap, **fields)
    assert cmd[cmd.index("--served-identity-dir") + 1] == snap and cmd.index("--served-identity-dir") > cmd.index("--results-root")
    with pytest.raises(GD.DispatchRefused) as ei:
        GD.build_command(GD.DEFAULT_TEMPLATES["agentic"], rows, **fields)
    assert ei.value.reason == "placeholder_unresolved" and "{served_identity_dir}" in ei.value.detail
    # the snapshot: the root's files present, relative layout kept; an existing directory is another dispatch's -> failed, never reused
    local_root = tmp_path / "mirror" / "exp" / "misalign" / "synthmodel_4b_s1"; (local_root / "serving").mkdir(parents=True)
    cells_full = json.dumps({"endpoint": {"model": "org/synthetic-model", "model_revision": "a" * 40}})
    (local_root / "cells_full.json").write_text(cells_full); (local_root / "serving" / "endpoint.recorded.json").write_text("{}")
    files = GD.ROOT_IDENTITY_FILES["agentic"]
    assert GD.snapshot_served_identity(str(local_root), snap, files, str(st), "agentic") == ["cells_full.json", "serving/endpoint.recorded.json"]
    assert (Path(snap) / "cells_full.json").read_text() == cells_full and (Path(snap) / "serving" / "endpoint.recorded.json").exists()
    assert GD.snapshot_served_identity(str(local_root), snap, files, str(st), "agentic") is None
    ev = [json.loads(l) for l in (st / "grade_daemon.jsonl").read_text().splitlines()]
    assert ev[-1]["event"] == "identity_snapshot_failed" and ev[-1]["snapshot"] == snap
    assert ev[-2]["event"] == "served_identity_snapshot" and ev[-2]["files"] == ["cells_full.json", "serving/endpoint.recorded.json"] and ev[-2]["root"] == str(local_root)
    # mirror_root_identity under a HELD root identity lock: `locked`, the local candidates untouched, no snapshot, no rsync call
    src = tmp_path / "remote_root"; src.mkdir(); (src / "cells_full.json").write_text(cells_full)
    rsync = tmp_path / "rsync_src"; rsync.write_text(FAKE_RSYNC_IDENTITY_SOURCE); rsync.chmod(0o755); monkeypatch.setenv("FAKE_RSYNC_IDENTITY_SRC", str(src))
    snap2 = str(tmp_path / "second.identity")
    fd = os.open(str(local_root) + ".identity.lock", os.O_RDWR | os.O_CREAT); fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        rc = GD.mirror_root_identity("cluster-a-login", "/cluster_a/user/exp/misalign/synthmodel_4b_s1", str(local_root), files, str(rsync), str(st), "agentic", snapshot_dir=snap2)
    finally:
        os.close(fd)
    assert str(rc).startswith("locked") and (local_root / "cells_full.json").read_text() == cells_full and not os.path.exists(snap2) and not log.exists()
    ev = [json.loads(l) for l in (st / "grade_daemon.jsonl").read_text().splitlines()]
    assert ev[-1]["event"] == "mirror_identity" and ev[-1]["removed_stale"] == [] and ev[-1]["snapshot_files"] is None and str(ev[-1]["rc"]).startswith("locked")
    # lock free: delete + transfer + snapshot in one section; the event names the snapshot and its files; the lock is released afterwards
    rc = GD.mirror_root_identity("cluster-a-login", "/cluster_a/user/exp/misalign/synthmodel_4b_s1", str(local_root), files, str(rsync), str(st), "agentic", snapshot_dir=snap2)
    assert rc == 0 and (Path(snap2) / "cells_full.json").read_text() == cells_full and sorted(os.listdir(snap2)) == ["cells_full.json"]
    ev = [json.loads(l) for l in (st / "grade_daemon.jsonl").read_text().splitlines()]
    assert ev[-1]["event"] == "mirror_identity" and ev[-1]["rc"] == 0 and ev[-1]["snapshot"] == snap2 and ev[-1]["snapshot_files"] == ["cells_full.json"]
    assert ev[-1]["removed_stale"] == ["cells_full.json", "serving/endpoint.recorded.json"] and ev[-1]["present"] == ["cells_full.json"]
    assert ev[-2]["event"] == "served_identity_snapshot" and ev[-2]["snapshot"] == snap2
    with GD.root_identity_lock(str(local_root)) as taken:
        assert taken is True


def test_a_concurrent_re_mirror_of_the_shared_root_never_touches_what_a_running_grader_reads_its_own_identity_snapshot(tmp_path, monkeypatch):
    """End to end: two instances over ONE results root. The first is mid-dispatch (its grader
    blocks); the second selects another newly generated cell under the same root -- the first's store is claimed, so the per-store locks do
    not conflict -- and re-mirrors the root's identity files: it deletes the local candidates and its transfer FAILS, leaving the shared
    root with no identity at all. The first grader was handed an immutable per-dispatch snapshot (<state>/dispatch_logs/<dispatch>.<pid>
    .identity/, copied under the root's identity lock in the same section as its own transfer): after the hold its rows step reads that
    snapshot unchanged and the dispatch succeeds; the second's group is skipped (mirror_identity_failed) and nothing of the first's was
    touched."""
    import time
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    src = tmp_path / "remote_root"; src.mkdir()
    cells_full = json.dumps({"endpoint": {"model": "org/synthetic-model", "model_revision": "a" * 40}})
    (src / "cells_full.json").write_text(cells_full)
    root, rids = _root(tmp_path)
    rsync = tmp_path / "rsync_src"; rsync.write_text(FAKE_RSYNC_IDENTITY_SOURCE); rsync.chmod(0o755)
    mirror_args = ["--mirror", "cluster_a=cluster-a-login", "--rsync-bin", str(rsync)]
    fake = tmp_path / "fake_driver.py"; record = tmp_path / "driver_calls.jsonl"
    template = f"{{python}} {fake} --worklist registry --registry-root {{registry_root}} {{run_id_flags}} {{store_map_flags}} --record {record} --served-identity-dir {{served_identity_dir}}"
    agentic = ["--eval-family", f"{synth.EVAL_ID}=agentic", "--driver", f"agentic={template}"]       # the agentic layout: <results-root>/<family>/<cell>
    mirror_root = tmp_path / "mirror"                                                                # ... so the mapped stores' root (<mirror>/stores/<cell>) is the judged root
    hold = tmp_path / "driver_hold"; hold.write_text(""); ident_rec = tmp_path / "identity_read.json"
    first = _start_daemon(root, tmp_path, *mirror_args, *agentic,
                          env_extra={"FAKE_DRIVER_HOLD_FILE": str(hold), "FAKE_DRIVER_IDENTITY_RECORD": str(ident_rec), "FAKE_RSYNC_IDENTITY_SRC": str(src)})
    end = time.time() + 60
    while time.time() < end and not [e for e in _events(tmp_path) if e["event"] == "dispatch"]:
        time.sleep(0.1)
    (disp,) = [e for e in _events(tmp_path) if e["event"] == "dispatch"]
    snap = Path(disp["served_identity_dir"])
    assert snap.is_dir() and snap.parent == tmp_path / "state" / "dispatch_logs" and snap.name.endswith(".identity") and "_agentic_" in snap.name
    assert f"--served-identity-dir {snap}" in disp["cmd"] and sorted(disp["run_ids"]) == sorted([rids["job7@none"], rids["job7@steer"]]) and disp["store_root"] == str(mirror_root)
    assert (snap / "cells_full.json").read_text() == cells_full and (mirror_root / "cells_full.json").read_text() == cells_full
    (ss,) = [e for e in _events(tmp_path) if e["event"] == "served_identity_snapshot"]
    assert ss["snapshot"] == str(snap) and ss["files"] == ["cells_full.json"] and ss["root"] == str(mirror_root)
    (mi,) = [e for e in _events(tmp_path) if e["event"] == "mirror_identity"]
    assert mi["rc"] == 0 and mi["snapshot"] == str(snap) and mi["snapshot_files"] == ["cells_full.json"]
    n0 = len(_events(tmp_path))
    # the second instance: the OTHER model's cell under the same root (a different store directory: no store lock conflict); its identity
    # transfer fails after its delete
    argv, env = _daemon_argv_env(root, tmp_path, *mirror_args, *agentic, env_extra={"FAKE_RSYNC_IDENTITY_SRC": str(src), "FAKE_RSYNC_IDENTITY_FAIL": "1"})
    i = argv.index("--model-id"); argv[i + 1] = OTHER_MODEL
    second = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=600, cwd=str(REPO))
    assert second.returncode == 0, second.stderr + second.stdout
    ev2 = _events(tmp_path)[n0:]
    (mif,) = [e for e in ev2 if e["event"] == "mirror_identity_failed"]
    assert mif["run_ids"] == [rids["job8@other"]] and mif["roots"] == [str(mirror_root)] and mif["rcs"] == {str(mirror_root): 12}
    (mi2,) = [e for e in ev2 if e["event"] == "mirror_identity"]
    assert mi2["rc"] == 12 and mi2["removed_stale"] == ["cells_full.json"] and mi2["present"] == [] and mi2["snapshot_files"] is None
    assert not [e for e in ev2 if e["event"] in ("dispatch", "dispatched", "mirror_skipped_locked", "mirror_retired_incomplete", "served_identity_snapshot", "identity_snapshot_failed")], ev2
    assert [e for e in ev2 if e["event"] == "tick"][-1]["skipped_identity"] == 1
    assert not (mirror_root / "cells_full.json").exists()                     # the SHARED root lost its identity under the running grader ...
    assert (snap / "cells_full.json").read_text() == cells_full                # ... and the grader's own snapshot is untouched
    assert first.poll() is None
    hold.unlink()
    out, err = first.communicate(timeout=120)
    assert first.returncode == 0, err + out
    assert json.loads(ident_rec.read_text()) == {"served_identity_dir": str(snap), "cells_full": cells_full, "files": ["cells_full.json"]}
    ev = _events(tmp_path)
    (done,) = [e for e in ev if e["event"] == "dispatched"]
    assert done["rc"] == 0 and done["outcome"] == "ok"
    calls = [json.loads(l) for l in record.read_text().splitlines()]
    assert len(calls) == 1 and calls[0]["argv"][calls[0]["argv"].index("--served-identity-dir") + 1] == str(snap)
    with DB.reading(DB.db_path(root)) as conn:
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rids[k],)).fetchone()[0] == "judged" for k in ("job7@none", "job7@steer"))
        assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rids["job8@other"],)).fetchone()[0] == "generated"


def test_a_grader_whose_rows_step_fails_after_filing_is_a_failed_dispatch_never_a_held_complete_run(tmp_path, monkeypatch):
    """The daemon's side: the misalign judge now exits 1 when its rows step fails (the judgments it filed stay filed).
    With the fake grader filing, folding and then exiting 1 on the rows step's terminal line: the dispatch is recorded failed (`dispatched`
    outcome failed, `dispatch_failed` carrying that line), the runs are quarantined with the evidence, NO spooled hold is written (the run
    is not held complete with its derived rows missing), and the judgments filed before the rows step are in the registry."""
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, rids = _root(tmp_path)
    r = _run_daemon(root, tmp_path, *_mirror_args(tmp_path), env_extra={"FAKE_DRIVER_ROWS_FAIL": "1"})
    assert r.returncode == 0, r.stderr + r.stdout
    ev = _events(tmp_path)
    (done,) = [e for e in ev if e["event"] == "dispatched"]
    assert done["rc"] == 1 and done["outcome"] == "failed" and done["terminal_exception"].startswith("[judge] ROWS STEP FAILED"), done
    (failed,) = [e for e in ev if e["event"] == "dispatch_failed"]
    assert sorted(failed["run_ids"]) == sorted([rids["job7@none"], rids["job7@steer"]]) and failed["terminal_exception"].startswith("[judge] ROWS STEP FAILED")
    assert not [e for e in ev if e["event"] in ("dispatch_spooled", "dispatch_retryable", "spooled_hold")], ev
    assert not (tmp_path / "state" / "dispatch_spooled.json").exists()
    q = json.loads((tmp_path / "state" / "dispatch_quarantine.json").read_text())["runs"]
    assert set(q) == {rids["job7@none"], rids["job7@steer"]} and all("ROWS STEP FAILED" in q[k]["stderr_tail"] and q[k]["rc"] == 1 for k in q)
    tick = [e for e in ev if e["event"] == "tick"][-1]
    assert tick["failed"] == 2 and tick["dispatched"] == 0 and tick["retryable"] == 0, tick
    with DB.reading(DB.db_path(root)) as conn:          # what the grader filed before its rows step stays filed
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rids[k],)).fetchone()[0] == "judged" for k in ("job7@none", "job7@steer"))
        assert conn.execute("SELECT COUNT(*) FROM judgment WHERE status = 'complete'").fetchone()[0] == 2


def _rows_ledger(tmp_path: Path) -> dict:
    p = tmp_path / "state" / "dispatch_rows_pending.json"
    return json.loads(p.read_text())["runs"] if p.exists() else {}


def _agentic_harness(tmp_path: Path, monkeypatch, env_extra: dict):
    """The agentic family over the synthetic stores (the misalign layout <results-root>/<family>/<cell>: the mapped stores' root <mirror> is the
    judged root), a fake grader handed the dispatch's identity snapshot, and a fake rsync whose identity transfer copies cells_full.json from a
    directory standing in for the cluster root."""
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    src = tmp_path / "remote_root"; src.mkdir(exist_ok=True)
    (src / "cells_full.json").write_text(json.dumps({"endpoint": {"model": "org/synthetic-model", "model_revision": "a" * 40}}))
    rsync = tmp_path / "rsync_src"; rsync.write_text(FAKE_RSYNC_IDENTITY_SOURCE); rsync.chmod(0o755)
    fake = tmp_path / "fake_driver.py"; record = tmp_path / "driver_calls.jsonl"
    template = f"{{python}} {fake} --worklist registry --registry-root {{registry_root}} {{run_id_flags}} {{store_map_flags}} --record {record} --served-identity-dir {{served_identity_dir}}"
    extra = ["--mirror", "cluster_a=cluster-a-login", "--rsync-bin", str(rsync), "--eval-family", f"{synth.EVAL_ID}=agentic", "--driver", f"agentic={template}"]
    env = {"FAKE_RSYNC_IDENTITY_SRC": str(src), **env_extra}
    return extra, env, record


def test_a_failed_rows_step_is_retried_from_the_retained_snapshot_until_it_succeeds_and_only_then_is_the_dispatch_complete(tmp_path, monkeypatch):
    """The misalign grader files its COMPLETE judgments, then its rows step fails (exit 1).
    The run leaves v_pending_judgment at the fold, so the quarantine alone would drop the entry on the next tick and the derived rows would
    stay missing for ever. Tick 1: the dispatch is `dispatch_rows_failed` (not quarantined, not `dispatched`), one DURABLE item per run in
    dispatch_rows_pending.json -- pending, attempt 1, bound to the judgment ids this dispatch filed and to the retained snapshot -- the
    spooled hold is written, the graded store copies are kept (not retired; the prune pass defers the group), the judgments are in the
    registry. Tick 2: the rows step is retried from the SAME snapshot (`--rows-only`, the recorded command), succeeds, the items clear, the
    tick says rows_cleared, and the stores are then pruned."""
    extra, env, record = _agentic_harness(tmp_path, monkeypatch, {"FAKE_DRIVER_ROWS_FAIL": "1"})
    root, rids = _root(tmp_path)
    mirror = tmp_path / "mirror"; dirs = [mirror / "stores" / "job7_none", mirror / "stores" / "job7_steer"]
    both = sorted([rids["job7@none"], rids["job7@steer"]])
    # ---- tick 1: filed, folded, rows failed
    r = _run_daemon(root, tmp_path, *extra, env_extra=env); assert r.returncode == 0, r.stderr + r.stdout
    ev = _events(tmp_path)
    (disp,) = [e for e in ev if e["event"] == "dispatch"]; snap = disp["served_identity_dir"]
    (opened,) = [e for e in ev if e["event"] == "rows_opened"]
    assert sorted(opened["run_ids"]) == both and opened["snapshot_dir"] == snap
    (done,) = [e for e in ev if e["event"] == "dispatched"]
    assert done["rc"] == 1 and done["outcome"] == "failed" and sorted(done["rows_pending"]) == both
    (rf,) = [e for e in ev if e["event"] == "dispatch_rows_failed"]
    assert sorted(rf["run_ids"]) == both and rf["attempt"] == 1 and rf["cap"] == 3 and rf["snapshot_dir"] == snap and rf["terminal_exception"].startswith("[judge] ROWS STEP FAILED")
    assert not [e for e in ev if e["event"] in ("dispatch_failed", "dispatch_retryable", "mirror_retired_incomplete", "rows_retry")], ev
    (sp,) = [e for e in ev if e["event"] == "dispatch_spooled"]; assert sorted(sp["run_ids"]) == both      # the judgments ARE filed: held from re-dispatch
    (pd,) = [e for e in ev if e["event"] == "mirror_prune_deferred"]; assert pd["reason"] == "derived rows pending" and sorted(pd["run_ids"]) == both
    tick1 = [e for e in ev if e["event"] == "tick"][-1]
    assert tick1["dispatched"] == 0 and tick1["failed"] == 0 and tick1["rows_failed"] == 2 and tick1["rows_pending"] == 2 and tick1["rows_retried"] == 0, tick1
    assert tick1["rows_exhausted"] == 0 and tick1["mirror_pruned_dirs"] == 0 and not [p for p in tick1["problems"] if "rows_exhausted" in p]
    ledger = _rows_ledger(tmp_path)
    assert sorted(ledger) == both and all(ledger[k]["state"] == "pending" and ledger[k]["attempts"] == 1 and ledger[k]["snapshot_dir"] == snap for k in ledger)
    assert all(len(ledger[k]["judgment_ids"]) == 1 and ledger[k]["last_error"].startswith("[judge] ROWS STEP FAILED") and ledger[k]["dirs"] == sorted(str(d) for d in dirs) for k in ledger)
    assert all("--served-identity-dir" in ledger[k]["cmd"] and ledger[k]["cmd"][ledger[k]["cmd"].index("--served-identity-dir") + 1] == snap for k in ledger)
    assert not (tmp_path / "state" / "dispatch_quarantine.json").exists()
    assert all((d / "store.bin").exists() for d in dirs) and Path(snap, "cells_full.json").exists()        # kept for the retry
    with DB.reading(DB.db_path(root)) as conn:
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (k,)).fetchone()[0] == "judged" for k in both)
        assert conn.execute("SELECT COUNT(*) FROM judgment WHERE status = 'complete'").fetchone()[0] == 2
    # ---- tick 2: the retry from the retained snapshot succeeds
    ok = tmp_path / "rows_retry_ok"; ok.write_text("")
    n0 = len(ev)
    r = _run_daemon(root, tmp_path, *extra, env_extra={**env, "FAKE_DRIVER_ROWS_RETRY_OK_FILE": str(ok)}); assert r.returncode == 0, r.stderr + r.stdout
    ev2 = _events(tmp_path)[n0:]
    (rt,) = [e for e in ev2 if e["event"] == "rows_retry"]
    assert sorted(rt["run_ids"]) == both and rt["attempt"] == 2 and rt["snapshot_dir"] == snap and rt["cmd"].endswith(" --rows-only") and f"--served-identity-dir {snap}" in rt["cmd"]
    assert Path(rt["log"]).parent == tmp_path / "state" / "dispatch_logs" and "_agentic-rows_" in Path(rt["log"]).name
    (cl,) = [e for e in ev2 if e["event"] == "rows_cleared"]
    assert sorted(cl["run_ids"]) == both and cl["by"] == "retry" and cl["attempt"] == 2
    assert not [e for e in ev2 if e["event"] in ("dispatch", "dispatched", "rows_retry_failed", "rows_exhausted", "dispatch_rows_failed")], ev2
    tick2 = [e for e in ev2 if e["event"] == "tick"][-1]
    assert tick2["rows_retried"] == 2 and tick2["rows_cleared"] == 2 and tick2["rows_pending"] == 0 and tick2["rows_exhausted"] == 0 and tick2["dispatched"] == 0, tick2
    assert tick2["mirror_pruned_dirs"] == 2 and not any(d.exists() for d in dirs) and Path(snap, "cells_full.json").exists()   # the stores go; the snapshot stays
    assert _rows_ledger(tmp_path) == {}
    calls = [json.loads(l) for l in record.read_text().splitlines()]
    assert len(calls) == 2 and "--rows-only" not in calls[0]["argv"] and "--rows-only" in calls[1]["argv"]
    assert calls[1]["argv"][calls[1]["argv"].index("--served-identity-dir") + 1] == snap and sorted(calls[1]["run_ids"]) == both


def test_rows_retries_are_bounded_then_a_named_terminal_state_the_summary_carries_until_the_operator_re_arms(tmp_path, monkeypatch):
    """The rows step keeps failing. Attempt 1 is the grader's own; ticks 2 and 3 retry from the snapshot; at ROWS_ATTEMPT_CAP
    (3) the items enter `rows_exhausted` -- a `rows_exhausted` event, the tick summary's rows_exhausted count and a `problem` -- and stay in
    the ledger: tick 4 retries nothing and still counts them (never a silent drop), the stores stay kept. The release file re-arms them:
    attempts reset, retried, and (the rows step now succeeding) cleared."""
    extra, env, record = _agentic_harness(tmp_path, monkeypatch, {"FAKE_DRIVER_ROWS_FAIL": "1"})
    root, rids = _root(tmp_path)
    mirror = tmp_path / "mirror"; dirs = [mirror / "stores" / "job7_none", mirror / "stores" / "job7_steer"]
    both = sorted([rids["job7@none"], rids["job7@steer"]])
    ticks = []
    for i in range(4):
        n0 = len(_events(tmp_path))
        r = _run_daemon(root, tmp_path, *extra, env_extra=env); assert r.returncode == 0, r.stderr + r.stdout
        ticks.append(_events(tmp_path)[n0:])
    (rf1,) = [e for e in ticks[0] if e["event"] == "dispatch_rows_failed"]; assert sorted(rf1["run_ids"]) == both and rf1["attempt"] == 1
    (rf2,) = [e for e in ticks[1] if e["event"] == "rows_retry_failed"]; assert sorted(rf2["run_ids"]) == both and rf2["attempt"] == 2 and rf2["rc"] == 1
    assert not [e for e in ticks[1] if e["event"] == "rows_exhausted"] and [e for e in ticks[1] if e["event"] == "tick"][-1]["rows_pending"] == 2
    (rf3,) = [e for e in ticks[2] if e["event"] == "rows_retry_failed"]; assert rf3["attempt"] == 3
    (ex,) = [e for e in ticks[2] if e["event"] == "rows_exhausted"]
    assert sorted(ex["run_ids"]) == both and ex["attempts"] == 3 and ex["terminal_exception"].startswith("[judge] ROWS STEP FAILED")
    t3 = [e for e in ticks[2] if e["event"] == "tick"][-1]
    assert t3["rows_exhausted"] == 2 and t3["rows_pending"] == 0 and t3["rows_retried"] == 2 and any(p.startswith("rows_exhausted: 2 run(s)") for p in t3["problems"]), t3
    (pb,) = [e for e in ticks[2] if e["event"] == "problem" and e["kind"] == "rows_exhausted"]; assert sorted(pb["run_ids"]) == both
    # tick 4: nothing retried, still counted, still held in the ledger; the stores still kept
    assert not [e for e in ticks[3] if e["event"] in ("rows_retry", "rows_retry_failed", "rows_cleared", "dispatch", "mirror_pruned")], ticks[3]
    t4 = [e for e in ticks[3] if e["event"] == "tick"][-1]
    assert t4["rows_exhausted"] == 2 and t4["rows_retried"] == 0 and any("rows_exhausted" in p for p in t4["problems"]), t4
    ledger = _rows_ledger(tmp_path)
    assert sorted(ledger) == both and all(v["state"] == "rows_exhausted" and v["attempts"] == 3 for v in ledger.values())
    assert all((d / "store.bin").exists() for d in dirs)
    assert len([json.loads(l) for l in record.read_text().splitlines()]) == 3                                   # the grader + two retries, no more
    # the operator re-arms: release -> attempts reset, retried from the retained snapshot, cleared once the rows step succeeds
    (tmp_path / "state" / "dispatch_rows_pending.release").write_text("*\n")
    ok = tmp_path / "rows_retry_ok"; ok.write_text("")
    n0 = len(_events(tmp_path))
    r = _run_daemon(root, tmp_path, *extra, env_extra={**env, "FAKE_DRIVER_ROWS_RETRY_OK_FILE": str(ok)}); assert r.returncode == 0, r.stderr + r.stdout
    ev5 = _events(tmp_path)[n0:]
    assert sorted(e["run_id"] for e in ev5 if e["event"] == "rows_released") == both
    (rt,) = [e for e in ev5 if e["event"] == "rows_retry"]; assert rt["attempt"] == 1
    (cl,) = [e for e in ev5 if e["event"] == "rows_cleared"]; assert sorted(cl["run_ids"]) == both and cl["by"] == "retry"
    t5 = [e for e in ev5 if e["event"] == "tick"][-1]
    assert t5["rows_exhausted"] == 0 and t5["rows_pending"] == 0 and t5["rows_cleared"] == 2 and not [p for p in t5["problems"] if "rows_exhausted" in p], t5
    assert _rows_ledger(tmp_path) == {} and not (tmp_path / "state" / "dispatch_rows_pending.release").exists() and t5["mirror_pruned_dirs"] == 2


# a fake rsync that ALSO writes the operator's hold file on its real (non-dry-run) call -- the seam between the per-group
# recheck (before the mirror and claim) and the pre-subprocess recheck (after them, before the grader starts)
FAKE_RSYNC_WRITES_HOLD = FAKE_RSYNC.replace(
    'echo "$*" >> "${FAKE_RSYNC_LOG:?}"',
    'echo "$*" >> "${FAKE_RSYNC_LOG:?}"; if [ -n "${FAKE_RSYNC_WRITE_HOLD:-}" ]; then printf \'# written while this group mirrored\\nhold from the fake rsync\\n\' > "$FAKE_RSYNC_WRITE_HOLD"; fi')
assert FAKE_RSYNC_WRITES_HOLD != FAKE_RSYNC


def test_a_hold_that_appears_after_a_groups_mirror_and_claim_stops_it_before_its_grader_and_releases_the_claim(tmp_path, monkeypatch):
    """The per-group recheck (the hold seen BEFORE a group's mirror and claim) is covered elsewhere; the second recheck -- right
    before the grader subprocess starts, after the group was mirrored and its dispatch claimed -- was
    uncovered. Here the fake rsync writes the hold while THIS group mirrors (after the per-group recheck passed), so the pre-subprocess
    recheck is the one that catches it: no grader subprocess is dispatched for the group (no `dispatched` event, no driver call), the
    mirror is settled as not dispatched (the still-pending copies retired), the dispatch claim is released, one mid-tick `held` event names
    the group and where it was caught, the summary says held, and the runs keep their generated lifecycle with no judgment."""
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, rids = _root(tmp_path)
    st = tmp_path / "state"; st.mkdir()
    fake = tmp_path / "rsync_writes_hold"; fake.write_text(FAKE_RSYNC_WRITES_HOLD); fake.chmod(0o755)
    r = _run_daemon(root, tmp_path, "--mirror", "cluster_a=cluster-a-login", "--rsync-bin", str(fake), env_extra={"FAKE_RSYNC_WRITE_HOLD": str(st / "dispatch.hold")})
    assert r.returncode == 0, r.stderr + r.stdout
    ev = _events(tmp_path)
    pair = sorted([rids["job7@none"], rids["job7@steer"]])
    assert sorted(e["run_id"] for e in ev if e["event"] == "mirror") == pair                               # the group WAS mirrored (the per-group recheck had passed)
    (disp,) = [e for e in ev if e["event"] == "dispatch"]
    assert sorted(disp["run_ids"]) == pair and disp["dry_run"] is False                                   # announced and claimed ...
    assert not [e for e in ev if e["event"] == "dispatched"] and not (tmp_path / "driver_calls.jsonl").exists()   # ... but no grader subprocess ran
    (held,) = [e for e in ev if e["event"] == "held"]
    assert held["mid_tick"] is True and held["reason"] == "hold from the fake rsync", held
    assert held["where"] == "after this group's mirror and claim, before its grader", held
    assert [(g["model_key"], sorted(g["run_ids"])) for g in held["stopped_groups"]] == [("synthmodel_4b_s1", pair)] and held["pending"] == 2, held
    tick = [e for e in ev if e["event"] == "tick"][-1]
    assert tick["held"] is True and tick["dispatched"] == 0 and tick["pending"] == 2, tick
    assert json.loads((st / "mirror_dispatch_claims.json").read_text())["claims"] == []                     # the dispatch claim is released
    assert sorted(e["run_id"] for e in ev if e["event"] == "mirror_retired_incomplete") == pair             # settled as not dispatched: the copies retired
    for e in ev:
        if e["event"] == "mirror_retired_incomplete":
            assert "before the grader started" in e["note"] and not os.path.isdir(e["dir"]), e
    assert not [e for e in ev if e["event"] in ("spooled_hold", "dispatch_quarantine", "contrasts")], ev
    with DB.reading(DB.db_path(root)) as conn:
        for k in ("job7@none", "job7@steer"):
            assert conn.execute("SELECT COUNT(*) FROM judgment WHERE run_id = ?", (rids[k],)).fetchone()[0] == 0
            assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rids[k],)).fetchone()[0] == "generated"
    assert (st / "dispatch.hold").exists()                                                                 # the operator's file is never consumed by the daemon
    # the hold lifted: the next tick dispatches the pair as before
    (st / "dispatch.hold").unlink(); n = len(ev)
    r2 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path)); assert r2.returncode == 0, r2.stderr + r2.stdout
    ev2 = _events(tmp_path)[n:]
    tick2 = [e for e in ev2 if e["event"] == "tick"][-1]
    assert tick2["held"] is False and tick2["dispatched"] == 2 and not [e for e in ev2 if e["event"] == "held"], tick2


def test_a_daemon_that_dies_after_its_grader_filed_but_before_the_fold_keeps_the_judged_copies_for_the_rows_retry_across_the_restart(tmp_path, monkeypatch):
    """Part 1: the daemon is SIGKILLed after its misalign grader FILED the complete judgments (spooled, not yet
    folded: the runs still read pending) -- the rows item stays `awaiting_grader`, the mirror claim stays. On the restart the replacement's
    claim reconciliation used to settle the dead daemon's dispatch as an unknown outcome and RETIRE the still-pending copies -- exactly the
    judged.jsonl the adopted rows retry needed. Now the tick runs the rows pass first (the item is adopted from the spool and retried), and
    the reconciliation keeps (queues) every copy an open item references: with the retry failing on the restart tick the copies survive it;
    once the fold shows, the next tick's retry succeeds from the kept copy and snapshot, the item clears and the stores are pruned."""
    import signal, time
    from sisteer.registry import pending as P
    extra, env0, record = _agentic_harness(tmp_path, monkeypatch, {})
    root, rids = _root(tmp_path)
    mirror = tmp_path / "mirror"; dirs = [mirror / "stores" / "job7_none", mirror / "stores" / "job7_steer"]
    both = sorted([rids["job7@none"], rids["job7@steer"]])
    hold = tmp_path / "hold_after_file"; hold.write_text(""); pidf = tmp_path / "grader.pid"
    first = _start_daemon(root, tmp_path, *extra, env_extra={**env0, "FAKE_DRIVER_NO_FOLD": "1", "FAKE_DRIVER_PID_FILE": str(pidf), "FAKE_DRIVER_HOLD_AFTER_FILE_FILE": str(hold)})
    end = time.time() + 60
    while time.time() < end and not (pidf.exists() and pidf.read_text().strip() and all(P.spool_complete_judgments(root, synth.CAMPAIGN, k) for k in both)):
        time.sleep(0.1)
    grader = int(pidf.read_text().strip())
    assert all(P.spool_complete_judgments(root, synth.CAMPAIGN, k) for k in both)                 # filed (spooled), the grader still alive in its tail
    ledger = _rows_ledger(tmp_path)
    assert sorted(ledger) == both and all(v["state"] == "awaiting_grader" and v["daemon_pid"] == first.pid for v in ledger.values())
    os.kill(first.pid, signal.SIGKILL); first.wait(timeout=30)                                      # the daemon dies; the grader exits on its own, nobody settles
    hold.unlink()
    end = time.time() + 60
    while time.time() < end and not _grader_exited(grader):                            # one read per poll: gone or zombie = exited
        time.sleep(0.1)
    with DB.reading(DB.db_path(root)) as conn:                                                      # the claim folded at once (as a real grader's), the judgments did NOT: the runs still read pending
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (k,)).fetchone()[0] in ("judging", "generated") for k in both)
        assert conn.execute("SELECT COUNT(*) FROM judgment WHERE status = 'complete'").fetchone()[0] == 0
    assert len(json.loads((tmp_path / "state" / "mirror_dispatch_claims.json").read_text())["claims"]) == 1
    # ---- the restart: the item is adopted and retried (the retry fails this tick); the reconciliation KEEPS the copies it would have retired
    n0 = len(_events(tmp_path))
    second = _run_daemon(root, tmp_path, *extra, env_extra={**env0, "FAKE_DRIVER_ROWS_FAIL": "1"}); assert second.returncode == 0, second.stderr + second.stdout
    ev2 = _events(tmp_path)[n0:]
    names = [e["event"] for e in ev2]
    assert sorted(e["run_id"] for e in ev2 if e["event"] == "rows_adopted") == both and all(len(e["judgment_ids"]) == 1 for e in ev2 if e["event"] == "rows_adopted")
    (rt,) = [e for e in ev2 if e["event"] == "rows_retry"]; assert sorted(rt["run_ids"]) == both and rt["attempt"] == 2
    (rf,) = [e for e in ev2 if e["event"] == "rows_retry_failed"]; assert rf["attempt"] == 2
    assert names.index("rows_retry") < names.index("mirror_claim_reconciled")                        # the rows pass runs ahead of the reconciliation
    (rec,) = [e for e in ev2 if e["event"] == "mirror_claim_reconciled"]; assert rec["daemon_pid"] == first.pid and rec["retired"] == 0 and rec["queued"] == 2
    kept = [e for e in ev2 if e["event"] == "mirror_retire_deferred"]
    assert len(kept) == 2 and all(e["reason"] == "derived rows pending" for e in kept) and sorted(e["dir"] for e in kept) == sorted(str(d) for d in dirs)
    assert not [e for e in ev2 if e["event"] in ("mirror_retired_incomplete", "dispatch", "dispatched", "rows_dropped", "rows_exhausted", "mirror_pruned")], ev2
    assert [e for e in ev2 if e["event"] == "spooled_hold_rebuilt"]                                   # the spooled completions hold the runs from a re-dispatch
    assert all((d / "store.bin").exists() for d in dirs) and not (mirror / "store" / "rows.fake.json").exists()      # copies kept; no rows built yet
    ledger = _rows_ledger(tmp_path)
    assert sorted(ledger) == both and all(v["state"] == "pending" and v["attempts"] == 2 for v in ledger.values())
    ledger_snapshot = ledger[both[0]]["snapshot_dir"]; assert Path(ledger_snapshot, "cells_full.json").exists()
    tick2 = [e for e in ev2 if e["event"] == "tick"][-1]
    assert tick2["mirror_claims_reconciled"] == 1 and tick2["rows_retried"] == 2 and tick2["rows_pending"] == 2 and tick2["dispatched"] == 0, tick2
    # ---- the fold shows; the next tick's retry succeeds from the kept copy and the retained snapshot: cleared, pruned, complete
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    ok = tmp_path / "rows_retry_ok"; ok.write_text("")
    n1 = len(_events(tmp_path))
    third = _run_daemon(root, tmp_path, *extra, env_extra={**env0, "FAKE_DRIVER_ROWS_RETRY_OK_FILE": str(ok)}); assert third.returncode == 0, third.stderr + third.stdout
    ev3 = _events(tmp_path)[n1:]
    (cl,) = [e for e in ev3 if e["event"] == "rows_cleared"]; assert sorted(cl["run_ids"]) == both and cl["by"] == "retry" and cl["attempt"] == 3
    tick3 = [e for e in ev3 if e["event"] == "tick"][-1]
    assert tick3["rows_cleared"] == 2 and tick3["rows_pending"] == 0 and tick3["rows_exhausted"] == 0 and tick3["mirror_pruned_dirs"] == 2, tick3
    assert _rows_ledger(tmp_path) == {} and not any(d.exists() for d in dirs)
    built = json.loads((mirror / "store" / "rows.fake.json").read_text())                             # the derived rows ARE built, from the kept copies and the retained snapshot
    assert sorted(built["run_ids"]) == both and built["from_store_bins"] == sorted(str(d / "store.bin") for d in dirs) and built["served_identity_dir"] == ledger_snapshot
    with DB.reading(DB.db_path(root)) as conn:                                                      # folded by the harvester: judged (or already ingested)
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (k,)).fetchone()[0] in ("judged", "ingested") for k in both)
    calls = [json.loads(l) for l in record.read_text().splitlines()]
    assert len(calls) == 3 and [("--rows-only" in c["argv"]) for c in calls] == [False, True, True]


def test_mirror_settle_keeps_a_copy_an_open_rows_item_references_and_retires_one_no_item_references(tmp_path, monkeypatch):
    """The rule itself: with the dispatch settled as an unknown outcome (rc None: a dead daemon's claim) and every run still
    pending, a copy referenced by an open rows item -- by its run id, or as one of the item's recorded mirrored store directories, in any
    state -- is kept and queued (`mirror_retire_deferred`, reason "derived rows pending"); a copy with NO open item is retired as before."""
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    GD = _load_daemon_module(); st = tmp_path / "state"; st.mkdir()
    root, rids = _root(tmp_path)
    mirror = tmp_path / "mirror"; a = mirror / "stores" / "job7_none"; b = mirror / "stores" / "job7_steer"; c = mirror / "stores" / "job8_other"
    for d, n in ((a, 4), (b, 5), (c, 6)):
        d.mkdir(parents=True); (d / "store.bin").write_bytes(b"x" * n)
    item = {"run_id": rids["job7@none"], "family": "agentic", "state": "awaiting_grader", "attempts": 0, "dirs": [str(a), str(c)], "snapshot_dir": str(st / "s.identity"),
            "cmd": ["x"], "daemon_pid": 2 ** 22 - 1, "daemon_start": 1}
    GD.rows_update(str(st), lambda runs: runs.__setitem__(rids["job7@none"], item))
    assert (st / "dispatch_rows_pending.json.lock").exists() and _rows_ledger(tmp_path) == {rids["job7@none"]: item}
    mirrored = [{"run_id": rids["job7@none"], "substrate": "cluster_a", "local": str(a), "root": str(mirror), "rc": 0},     # referenced by run id
                {"run_id": rids["job7@steer"], "substrate": "cluster_a", "local": str(b), "root": str(mirror), "rc": 0},    # no item: retired
                {"run_id": rids["job8@other"], "substrate": "cluster_a", "local": str(c), "root": str(mirror), "rc": 0}]    # referenced as a recorded directory
    out = GD.mirror_settle(str(st), root, family="agentic", model_key="k", campaign=synth.CAMPAIGN, run_ids=[m["run_id"] for m in mirrored], mirrored=mirrored, rc=None,
                           reason="the daemon that dispatched this group died before settling it")
    assert out == {"retired": 1, "retired_bytes": 5, "queued": 2} and a.exists() and c.exists() and not b.exists()
    ev = [json.loads(l) for l in (st / "grade_daemon.jsonl").read_text().splitlines()]
    kept = [e for e in ev if e["event"] == "mirror_retire_deferred"]
    assert sorted(e["dir"] for e in kept) == sorted([str(a), str(c)]) and all(e["reason"] == "derived rows pending" for e in kept)
    (ret,) = [e for e in ev if e["event"] == "mirror_retired_incomplete"]; assert ret["dir"] == str(b) and ret["bytes"] == 5
    queued = json.loads((st / "mirror_prune_pending.json").read_text())["pending"]
    assert len(queued) == 1 and sorted(d["local"] for d in queued[0]["dirs"]) == sorted([str(a), str(c)])
    # rows_save writes through a writer-unique temp name: nothing of a fixed `.tmp` is left beside the ledger
    assert not (st / "dispatch_rows_pending.json.tmp").exists() and [p.name for p in st.glob("dispatch_rows_pending.json*")] and all(not p.name.endswith(".tmp") for p in st.glob("*"))


def test_a_long_rows_step_of_one_instance_never_writes_back_a_ledger_snapshot_over_another_instances_item(tmp_path, monkeypatch):
    """Part 2: instance A's rows retry runs for a long time (its rows-only grader blocks) while instance B --
    another model's cell under the same root, its own store locks -- opens its dispatch's rows item and settles it (its rows step fails:
    pending) in the SAME ledger. A used to write back the whole snapshot it loaded before its subprocess, deleting B's item. Now every
    change is one locked read-modify-write over the current file: A's settlement clears A's items only, B's stands; B's own pass deferred
    A's group (store locks busy) rather than retrying it."""
    import time
    extra, env0, record = _agentic_harness(tmp_path, monkeypatch, {})
    root, rids = _root(tmp_path)
    both = sorted([rids["job7@none"], rids["job7@steer"]]); other = rids["job8@other"]
    r = _run_daemon(root, tmp_path, *extra, env_extra={**env0, "FAKE_DRIVER_ROWS_FAIL": "1"}); assert r.returncode == 0, r.stderr + r.stdout   # tick 0: job7 filed, rows failed
    assert sorted(_rows_ledger(tmp_path)) == both
    hold = tmp_path / "rows_only_hold"; hold.write_text(""); ok = tmp_path / "rows_retry_ok"; ok.write_text("")
    n0 = len(_events(tmp_path))
    first = _start_daemon(root, tmp_path, *extra, env_extra={**env0, "FAKE_DRIVER_ROWS_FAIL": "1", "FAKE_DRIVER_ROWS_RETRY_OK_FILE": str(ok), "FAKE_DRIVER_ROWS_ONLY_HOLD_FILE": str(hold)})
    end = time.time() + 60
    while time.time() < end and not [e for e in _events(tmp_path)[n0:] if e["event"] == "rows_retry"]:
        time.sleep(0.1)
    (rt,) = [e for e in _events(tmp_path)[n0:] if e["event"] == "rows_retry"]; assert sorted(rt["run_ids"]) == both      # A is inside its long rows step
    n1 = len(_events(tmp_path))
    argv, env = _daemon_argv_env(root, tmp_path, *extra, env_extra={**env0, "FAKE_DRIVER_ROWS_FAIL": "1"})
    i = argv.index("--model-id"); argv[i + 1] = OTHER_MODEL
    second = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=600, cwd=str(REPO)); assert second.returncode == 0, second.stderr + second.stdout
    ev2 = _events(tmp_path)[n1:]
    (dfr,) = [e for e in ev2 if e["event"] == "rows_retry_deferred"]; assert sorted(dfr["run_ids"]) == both               # B leaves A's group alone
    (op,) = [e for e in ev2 if e["event"] == "rows_opened"]; assert op["run_ids"] == [other]
    (rf,) = [e for e in ev2 if e["event"] == "dispatch_rows_failed"]; assert rf["run_ids"] == [other]
    ledger = _rows_ledger(tmp_path)
    assert sorted(ledger) == sorted(both + [other]) and ledger[other]["state"] == "pending" and ledger[other]["attempts"] == 1 and len(ledger[other]["judgment_ids"]) == 1
    assert first.poll() is None
    hold.unlink()
    out, err = first.communicate(timeout=120); assert first.returncode == 0, err + out
    evA = _events(tmp_path)[n1:]
    (cl,) = [e for e in evA if e["event"] == "rows_cleared" and e.get("by") == "retry"]; assert sorted(cl["run_ids"]) == both and cl["attempt"] == 2
    assert not [e for e in evA if e["event"] == "rows_retry_stale"]
    ledger = _rows_ledger(tmp_path)
    assert sorted(ledger) == [other] and ledger[other]["state"] == "pending" and ledger[other]["attempts"] == 1          # B's item survived A's save
    tickA = [e for e in evA if e["event"] == "tick" and e["rows_cleared"] == 2]
    assert len(tickA) == 1 and tickA[0]["rows_pending"] == 1 and tickA[0]["rows_retried"] == 2, tickA


def test_the_daemon_sweeps_non_current_contrast_anchors_once_per_code_sha_and_the_tick_summary_carries_the_counts(tmp_path, monkeypatch):
    """The campaign-wide re-emission sweep (`emit_contrasts.sweep_noncurrent`) runs ONCE per advance of the daemon's code --
    the first tick under a sha (no <state>/contrasts_sweep.json, or one stamped by another sha) sweeps and records the sha; the next tick
    under the same sha does not. The tick summary carries the block (`contrasts_sweep`: due / emitted / no_anchor / cap_mismatch / deferred,
    dry_run, the shas) and a `contrasts_sweep` event the per-cell table (read-only connection). Tick 1 sweeps an empty registry (zeros);
    tick 2 (same sha) does not sweep and the regular pass files the steered-vs-none row; then that row's anchor is retired (its run re-filed
    with an explicit successor) and the mark carries a stale sha: tick 3 sweeps again and counts the retired anchor without a replacement
    (no_anchor 1) under (model, view, category, same_standup_none); tick 4 (same sha) does not sweep."""
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, rids = _root(tmp_path)
    st = tmp_path / "state"; st.mkdir()
    r = _run_daemon(root, tmp_path, *_mirror_args(tmp_path)); assert r.returncode == 0, r.stderr + r.stdout
    ev = _events(tmp_path)
    tick1 = [e for e in ev if e["event"] == "tick"][-1]
    sha = tick1["daemon_sha"]
    assert tick1["contrasts_sweep"] == {"due": 0, "emitted": 0, "no_anchor": 0, "cap_mismatch": 0, "deferred": 0, "first": 0, "retired": 0, "rejudged": 0,
                                        "dry_run": False, "daemon_sha": sha, "was_daemon_sha": None}, tick1
    (sweep1,) = [e for e in ev if e["event"] == "sweep_noncurrent"]
    assert sweep1["readonly"] is True and sweep1["cells"] == [] and sweep1["daemon_sha"] == sha
    assert json.loads((st / "contrasts_sweep.json").read_text())["daemon_sha"] == sha
    assert H.ingest_once(root, workers=0, min_free=0, derive_facts=False).refused == 0            # the fold of what the fake driver spooled
    n = len(ev)
    r2 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path)); assert r2.returncode == 0, r2.stderr + r2.stdout
    ev2 = _events(tmp_path)[n:]
    tick2 = [e for e in ev2 if e["event"] == "tick"][-1]
    assert "contrasts_sweep" not in tick2 and not [e for e in ev2 if e["event"] == "sweep_noncurrent"], tick2       # same sha: no sweep
    # the regular pass filed the steered-vs-none row exactly once across the two ticks -- on tick 1 when the grader's finish() folds its
    # judgments, on tick 2 under a harvester-folded worklist (the same premise the dispatch test above holds) -- never the sweep
    written = [e["written"] for e in ev if e["event"] == "contrasts"] + [e["written"] for e in ev2 if e["event"] == "contrasts"]
    assert sorted(written) == [0, 1], written
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    with DB.reading(DB.db_path(root)) as conn:
        rows = conn.execute("SELECT run_id, anchor_run_id, anchor_kind FROM run_contrast").fetchall()
        assert [tuple(r) for r in rows] == [(rids["job7@steer"], rids["job7@none"], "same_standup_none")], rows
    # the filed anchor retired (its run re-filed with an explicit successor) and the code "advanced" (a stale sha in the mark)
    p = synth.run_payload("job7@none", "none", "none", 100, model_id=synth.MODEL_ID, view_legacy="synthmodel_4b_s1", cluster_job_id="7",
                          superseded_by_run_id=rids["job8@other"])
    p["store_uri"] = "/cluster_a/user/exp/stores/job7_none"
    R.write_records(os.path.join(root, "records", "c", "000009-supersede.jsonl"), [synth.rec("run", p, synth.t(900))])
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT superseded_by_run_id FROM run WHERE run_id = ?", (rids["job7@none"],)).fetchone()[0] == rids["job8@other"]
    (st / "contrasts_sweep.json").write_text(json.dumps({"daemon_sha": "0" * 40, "utc": "2026-10-02T00:00:00Z"}) + "\n")
    n = len(_events(tmp_path))
    r3 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path)); assert r3.returncode == 0, r3.stderr + r3.stdout
    ev3 = _events(tmp_path)[n:]
    tick3 = [e for e in ev3 if e["event"] == "tick"][-1]
    assert tick3["contrasts_sweep"] == {"due": 0, "emitted": 0, "no_anchor": 1, "cap_mismatch": 0, "deferred": 0, "first": 0, "retired": 1, "rejudged": 0,
                                        "dry_run": False, "daemon_sha": sha, "was_daemon_sha": "0" * 40}, tick3
    (sweep3,) = [e for e in ev3 if e["event"] == "sweep_noncurrent"]
    # the daemon test's spool is <root>/records/contrasts, outside the standard records/<campaign>/ layout: the row's writer class reads `other`
    # (not an adapter's, so still this emitter's to keep or re-file); live, the daemon's rows land under records/sdfbase_full22/ (native)
    assert [(c["uri_class"], c["model_id"], c["view_legacy"], c["eval_category"], c["anchor_kind"], c["no_anchor"], c["due"], c["retired"]) for c in sweep3["cells"]] == \
        [("other", synth.MODEL_ID, "synthmodel_4b_s1", "static_attack", "same_standup_none", 1, 0, 1)], sweep3
    assert json.loads((st / "contrasts_sweep.json").read_text())["daemon_sha"] == sha
    assert [e["written"] for e in ev3 if e["event"] == "contrasts"] == [0]                                          # nothing to file: no current none of the view
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM run_contrast").fetchone()[0] == 1
    n = len(_events(tmp_path))
    r4 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path)); assert r4.returncode == 0, r4.stderr + r4.stdout
    ev4 = _events(tmp_path)[n:]
    assert "contrasts_sweep" not in [e for e in ev4 if e["event"] == "tick"][-1] and not [e for e in ev4 if e["event"] == "sweep_noncurrent"]


def test_a_dead_daemons_awaiting_item_is_kept_while_its_grader_still_runs_then_adopted_once_it_files(tmp_path, monkeypatch):
    """The parent daemon is SIGKILLed after its misalign grader STARTED but before it filed; a
    replacement tick ran while the grader was still grading. rows_retry_pass used to drop the `awaiting_grader` item because the spool held
    no judgment yet -- the grader then filed and failed its rows step with nothing left to retry. Now the item is KEPT while the grader
    holds the dispatch's locks (`rows_awaiting_grader`: the inherited per-dispatch lock and the store locks), the replacement neither
    adopts nor drops it, dispatches nothing (the stores are locked) and defers the dead claim; once the grader has filed, folded and
    failed its rows step, the next tick adopts the item from the spool, retries from the retained snapshot and clears it, the dead claim
    is reconciled and the stores pruned."""
    import signal, time
    from sisteer.registry import pending as P
    extra, env0, record = _agentic_harness(tmp_path, monkeypatch, {})
    root, rids = _root(tmp_path)
    mirror = tmp_path / "mirror"; dirs = [mirror / "stores" / "job7_none", mirror / "stores" / "job7_steer"]
    both = sorted([rids["job7@none"], rids["job7@steer"]])
    hold = tmp_path / "driver_hold"; hold.write_text(""); pidf = tmp_path / "grader.pid"
    first = _start_daemon(root, tmp_path, *extra, env_extra={**env0, "FAKE_DRIVER_HOLD_FILE": str(hold), "FAKE_DRIVER_PID_FILE": str(pidf), "FAKE_DRIVER_ROWS_FAIL": "1"})
    end = time.time() + 60
    while time.time() < end and not (pidf.exists() and pidf.read_text().strip() and len(_rows_ledger(tmp_path)) == 2):
        time.sleep(0.1)
    grader = int(pidf.read_text().strip())
    ledger = _rows_ledger(tmp_path)
    assert sorted(ledger) == both and all(v["state"] == "awaiting_grader" and v["daemon_pid"] == first.pid for v in ledger.values())
    lock_path = ledger[both[0]]["dispatch_lock"]
    assert lock_path and Path(lock_path).exists() and Path(lock_path).parent == tmp_path / "state" / "dispatch_logs" and lock_path.endswith(f".{first.pid}.lock")
    assert not any(P.spool_complete_judgments(root, synth.CAMPAIGN, k) for k in both)                 # nothing filed yet: the grader is blocked before its claim
    os.kill(first.pid, signal.SIGKILL); first.wait(timeout=30)
    assert _proc_state(grader) not in (None, "Z")                                       # the grader is alive and not a zombie before the restart
    # ---- the restart while the grader still runs: the item is KEPT -- not dropped, not adopted; nothing dispatched; the claim deferred
    n0 = len(_events(tmp_path))
    second = _run_daemon(root, tmp_path, *extra, env_extra=env0); assert second.returncode == 0, second.stderr + second.stdout
    ev2 = _events(tmp_path)[n0:]
    (kept,) = [e for e in ev2 if e["event"] == "rows_awaiting_grader"]
    assert sorted(kept["run_ids"]) == both and all(lock_path in locks for locks in kept["locks"].values()) and all(any(str(d) in l for d in dirs) for l in kept["locks"].values())
    assert not [e for e in ev2 if e["event"] in ("rows_dropped", "rows_adopted", "rows_retry", "rows_cleared", "dispatch", "dispatched", "mirror_claim_reconciled", "mirror_retired_incomplete")], ev2
    assert [e for e in ev2 if e["event"] == "mirror_claim_deferred"] and [e for e in ev2 if e["event"] == "mirror_skipped_locked"]
    ledger = _rows_ledger(tmp_path)
    assert sorted(ledger) == both and all(v["state"] == "awaiting_grader" and v["daemon_pid"] == first.pid for v in ledger.values())
    assert all((d / "store.bin").exists() for d in dirs)
    # ---- the grader proceeds on its own: claims, files, folds, fails its rows step, exits 1 (nobody settles: the daemon is dead)
    hold.unlink()
    end = time.time() + 60
    while time.time() < end and not _grader_exited(grader):                            # one read per poll: gone or zombie = exited
        time.sleep(0.1)
    assert all(P.spool_complete_judgments(root, synth.CAMPAIGN, k) for k in both)
    with DB.reading(DB.db_path(root)) as conn:
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (k,)).fetchone()[0] in ("judged", "ingested") for k in both)
    assert not Path(lock_path).exists() or GD_lock_free(lock_path)
    # ---- the next tick: the locks are free -> adopted from the spool, retried from the retained snapshot, cleared; the dead claim reconciled, pruned
    ok = tmp_path / "rows_retry_ok"; ok.write_text("")
    n1 = len(_events(tmp_path))
    third = _run_daemon(root, tmp_path, *extra, env_extra={**env0, "FAKE_DRIVER_ROWS_RETRY_OK_FILE": str(ok)}); assert third.returncode == 0, third.stderr + third.stdout
    ev3 = _events(tmp_path)[n1:]
    assert sorted(e["run_id"] for e in ev3 if e["event"] == "rows_adopted") == both and all(len(e["judgment_ids"]) == 1 for e in ev3 if e["event"] == "rows_adopted")
    (rt,) = [e for e in ev3 if e["event"] == "rows_retry"]; assert sorted(rt["run_ids"]) == both and rt["attempt"] == 2 and rt["snapshot_dir"] == ledger[both[0]]["snapshot_dir"]
    (cl,) = [e for e in ev3 if e["event"] == "rows_cleared"]; assert sorted(cl["run_ids"]) == both and cl["by"] == "retry"
    (rec,) = [e for e in ev3 if e["event"] == "mirror_claim_reconciled"]; assert rec["daemon_pid"] == first.pid and rec["retired"] == 0
    assert not [e for e in ev3 if e["event"] in ("rows_awaiting_grader", "rows_dropped", "rows_exhausted", "mirror_retired_incomplete", "dispatch")], ev3
    built = json.loads((mirror / "store" / "rows.fake.json").read_text())
    assert sorted(built["run_ids"]) == both and built["from_store_bins"] == sorted(str(d / "store.bin") for d in dirs) and built["served_identity_dir"] == ledger[both[0]]["snapshot_dir"]
    tick3 = [e for e in ev3 if e["event"] == "tick"][-1]
    assert tick3["rows_cleared"] == 2 and tick3["rows_pending"] == 0 and tick3["mirror_pruned_dirs"] == 2 and tick3["mirror_claims_reconciled"] == 1, tick3
    assert _rows_ledger(tmp_path) == {} and not any(d.exists() for d in dirs)


def GD_lock_free(path: str) -> bool:
    import fcntl
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); return True
    except OSError:
        return False
    finally:
        os.close(fd)


def test_rows_retry_pass_keeps_an_awaiting_item_under_a_held_dispatch_or_store_lock_and_drops_it_once_both_are_free(tmp_path, monkeypatch):
    """The rule itself (no daemon run): an `awaiting_grader` item of a DEAD daemon with no judgment in the spool is kept while its
    dispatch lock is held, kept while one of its store locks is held, and dropped (the existing path) only once both are free."""
    import fcntl, types
    GD = _load_daemon_module(); st = tmp_path / "state"; st.mkdir(); (st / "dispatch_logs").mkdir()
    root, rids = _root(tmp_path)
    rid = rids["job7@none"]; store = tmp_path / "mirror" / "stores" / "job7_none"; store.mkdir(parents=True)
    lock_path = str(st / "dispatch_logs" / f"x_agentic_k.{2 ** 22 - 1}.lock")
    item = {"run_id": rid, "family": "agentic", "campaign": synth.CAMPAIGN, "state": "awaiting_grader", "attempts": 0, "dirs": [str(store)], "snapshot_dir": str(st / "s.identity"),
            "cmd": ["x"], "daemon_pid": 2 ** 22 - 1, "daemon_start": 1, "dispatch_lock": lock_path, "dispatch_utc": "2026-01-01T00:00:00Z"}
    GD.rows_update(str(st), lambda runs: runs.__setitem__(rid, dict(item)))
    args = types.SimpleNamespace(state_dir=str(st), dry_run=False, registry_root=root, campaign=synth.CAMPAIGN, repo=str(REPO), driver_timeout_s=60.0, grader_lock_timeout_s=1.0)

    def events():
        return [json.loads(l) for l in (st / "grade_daemon.jsonl").read_text().splitlines()] if (st / "grade_daemon.jsonl").exists() else []

    # (a) the dispatch lock held (as the grader's inherited fd holds it): kept
    fd = GD.dispatch_lock_take(lock_path); assert fd is not None
    try:
        GD.rows_retry_pass(args, {}, held=False)
    finally:
        os.close(fd)
    assert _rows_ledger(tmp_path)[rid]["state"] == "awaiting_grader" and events()[-1]["event"] == "rows_awaiting_grader" and events()[-1]["locks"][rid] == [lock_path]
    # (b) a store lock held instead: kept, the store lock named
    fd = os.open(str(store) + ".lock", os.O_RDWR | os.O_CREAT); fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        GD.rows_retry_pass(args, {}, held=False)
    finally:
        os.close(fd)
    assert _rows_ledger(tmp_path)[rid]["state"] == "awaiting_grader" and events()[-1]["event"] == "rows_awaiting_grader" and events()[-1]["locks"][rid] == [str(store)]
    assert not [e for e in events() if e["event"] in ("rows_dropped", "rows_adopted")]
    # (c) both free and nothing filed: dropped, as before
    GD.rows_retry_pass(args, {}, held=False)
    assert _rows_ledger(tmp_path) == {} and events()[-1]["event"] == "rows_dropped" and events()[-1]["run_ids"] == [rid]
    # the dispatch lock file the daemon takes is keyed to the dispatch log and this daemon's pid
    assert GD.dispatch_lock_for("/s/dispatch_logs/20260101T000000Z_agentic_k.log") == f"/s/dispatch_logs/20260101T000000Z_agentic_k.{os.getpid()}.lock"
