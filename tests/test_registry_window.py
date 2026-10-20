"""The landing window (``scripts/registry/landing/window.sh``) rehearsed against a temp state with ``systemctl``, ``git``, ``fuser``, ``systemd-analyze``
and the registry python replaced by PATH shims that record every call: the stage order and its stamps (the shape ``scripts/registry/measure/windows.py``
parses), the restore set (active at stop ∪ newly installed − ``NEVER_START``), the ``NEVER_START`` units never started on any path (the snapshot timer;
the two RETIRED units ``registry-adapters`` and ``registry-modal-pull``; both pairs of unit files and the Modal-era drop-ins uninstalled by the window),
the heavy status service started ONCE after the restore as the cache initializer, exactly
one ingest, the END MARKER written by the trap only after the verified restore (``=== ff chain end``) and ``=== ff chain ABORTED (<cause>)``
otherwise; an install failure aborting BEFORE anything else with the timers still restored; a migrate or ingest failure aborting the stages
after it with the timers restored and the ABORTED marker; the stop and the restore VERIFIED against the shim's active set (a timer that
refuses to stop fails the window before the base moves, exit 13; a restored timer that does not read active fails an otherwise successful
window from the trap, exit 14, ``ABORTED (restore)``). The real ``install_grading_host.sh --units-only`` runs inside (its ``systemctl`` is the
shim, stateful: stop removes from the active set, start / enable --now adds, is-active reads it).

THE GRADING HOST READS NO STORE -- no ``adapters run`` is ever called on the grading host; a named adapter the
cluster did not read is RECORDED REFUSED (``-- adapter A REFUSED HH:MM:SS (reason)``), the window ingests the rest and ends ``ABORTED
(refused: ...)`` with exit 20; the one grading-host run left is the operator's ``--reread --force`` under ``REGISTRY_STORE_READ_OVERRIDE``. THE
HONEST DRAIN: ``state/maintenance.json`` is written BEFORE the timers stop and removed by the trap; ``registry ingest --honor-maintenance``
exits 0 at the door while it exists, before the pull and before the lock, and the claim path never reads it. THE RETIRED READERS:
``registry-adapters.service`` / ``registry-modal-pull.service`` are stopped with the timers and CONFIRMED terminal -- ``is-active`` inactive
or failed, ``show -p MainPID`` 0, the cgroup's ``cgroup.procs`` EMPTY (a PID list, never a command-text match) -- before ``base at``; a
cgroup that never empties fails the window with exit 21 and the PIDs named, and no exit path ever starts a retired unit.

THE READS RESERVATION is the window's first act after the start marker (``cluster-reads reserve
--owner window --op <window> --takeover``: a tick's array cancelled by id and confirmed terminal, the reads-*/attest-* jobs reconciled; a
live window holding it -> exit 22 before the flag, the stop or the base move), ``submit`` and ``pull`` carry ``--op <window>`` (the ownership
re-checked before a job is submitted and before anything is pulled or merged), ``-- cluster reads receipt`` follows the post-pull merge, the
trap releases the reservation BEFORE the timers restore, ``-- uninstall units`` after the install disables and removes the retired unit files
(``registry-adapters.timer`` / ``.service``) and reloads, and ``registry-cluster-reads.timer`` is in the stopped set and the default set.
The migration plan stage: ``-- migrate (plan)`` reads the WHOLE ``registry migrate --plan`` into the log and a file and prints the plan's
TOKEN (the file's sha256) beside it, requires its version line, STOPS on a ``rebuild table`` line before the apply (exit 15, whatever
approval exists; ``WINDOW_ALLOW_REBUILD=1`` lets a reviewed rebuild through and says so), pins the pending list with
``WINDOW_EXPECT_MIGRATION`` (exit 15), and with ``WINDOW_PLAN_APPROVAL`` waits for an approval BOUND to the displayed plan (a file created
while the plan was still being generated used to pass) -- the file must appear after the complete plan is displayed and carry the plan's
token (the log prints the exact command); a file from before the plan or from during it is discarded, one carrying another token is ignored
with a logged line while the wait goes on, the accepted one is consumed -- else exit 16 with nothing applied; a plan with nothing pending
needs no approval. The reconciliation slot (``WINDOW_RECONCILE`` / ``WINDOW_RECONCILE_APPLY``) runs between the dimensions load and the
cluster stages, exit 17 on a failure; ``WINDOW_RECONCILE_APPLY`` is a LIST of the named actions to apply (every other named action a dry
run; the retired global ``1`` and a name ``WINDOW_RECONCILE`` lacks are refused by name before the lock, exit 30), and every applied
correction action runs to a verified fixed point after the ingest.

One window at a time, and the base pinned from the plan to the apply (the review's third round: a second window fast-forwarding the
base while the first waited for its approval would have had the first apply the NEW migration files under an approval of the OLD
plan): an exclusive flock on ``$STATE/windows/.window.lock`` is taken before the start marker and held until exit; a second window is
refused by name at once (exit 18, the holder's line printed, no shim called, no start marker). The base HEAD at the fast-forward, the
migration directory's files with their sha256 and the plan file's sha256 (= the token) are recorded beside the plan file
(``migrate-plan-<window>.pins``) and re-checked immediately before ``registry migrate``: the base advancing, a migration file added or
changed, or the plan file rewritten while the approval was pending refuses the apply (exit 19, both readings printed, nothing applied).
The window id carries the pid by default; the plan file is read once and made read-only. At the merge the slurm-only exit codes that
collided with main's 15-19 moved: 20 a refused adapter, 21 a retired reader still running, 22 the reads reservation refused, 23 the
retired units not uninstalled.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time

import pytest

from test_landing_site_env import SUB_A, write_site
from test_registry_measure import _load


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOW = os.path.join(REPO, "scripts", "registry", "landing", "window.sh")
INSTALL = os.path.join(REPO, "scripts", "registry", "install_grading_host.sh")
UNITS = os.path.join(REPO, "systemd", "user")
OLD_TIMERS = ["registry-harvest.timer", "registry-adapters.timer", "registry-mirrors.timer", "registry-mirrors-full.timer", "registry-modal-pull.timer",
              "registry-status.timer", "registry-snapshot.timer"]
#: the units the window never starts on any path (C6): the snapshot timer and the two retired units
NEVER_START = ["registry-snapshot.timer", "registry-adapters.timer", "registry-modal-pull.timer"]
RETIRED_SERVICES = ["registry-adapters.service", "registry-modal-pull.service"]
# the timers the stacked branches add (status-heavy from Phase 1b, gate-baseline from Phase 2, store-mirror from Phase 3a): the window's
# trap `enable --now`s whichever of them the install stage added this window; derived from the installer so a new unit joins here on its own
NEW_TIMERS = [t for t in re.search(r"^TIMERS=\((.*)\)$", open(INSTALL).read(), re.M).group(1).split()
              if t not in OLD_TIMERS and t not in NEVER_START]
NEW_TIMERS_STR = " ".join(NEW_TIMERS)
STOP_CALL = ("systemctl --user stop registry-harvest.timer registry-cluster-reads.timer registry-adapters.timer registry-modal-pull.timer registry-status.timer "
             "registry-status-heavy.timer registry-store-mirror.timer registry-adapters.service registry-modal-pull.service")
#: the retired unit FILES the window uninstalls after the install stage (C7; the Modal pull's joined in C10): their source files are gone from the repo
RETIRED_UNITS = ["registry-adapters.timer", "registry-adapters.service", "registry-modal-pull.timer", "registry-modal-pull.service"]
#: the Modal-era drop-ins the same stage removes (C10): the modal-pull unit's whole directory, the mirrors unit's clusters-only.conf alone
RETIRED_DROPINS = ["registry-modal-pull.service.d", "registry-mirrors.service.d/clusters-only.conf"]


def _shim(d, name, body):
    p = d / name
    p.write_text("#!/bin/bash\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


DEFAULT_PLAN = ["schema version 17; code 17; pending: none", "shape: matches the code's schema"]


FULL_SHA = "deadbee0" * 5          # the base HEAD the git shim reports (its --short form is the `base at deadbee` stamp)


def _rehearsal(tmp_path, *, active, pre_installed, installer=None, extra_env=None, fail_on=None, adapters="", plan_lines=None,
               shim_touch=None, shim_touch_content=None, state_dir=None, window_id="W1"):
    """Lay the temp world out; returns (env, paths). ``active``: the units the systemctl shim reports active; ``pre_installed``: the
    timer files present in the unit dir before the window; ``fail_on``: a shell glob over the registry python's arguments — the
    shim exits 1 (``boom`` on stderr) for that command, a failing stage; ``adapters``: the window's $ADAPTERS (default none: since C6 a
    named adapter without a cluster read is a recorded refusal, so the Phase-1 rehearsals name none); ``plan_lines``: what the shim prints
    for ``migrate --plan`` (default: nothing pending, the shape matches); ``shim_touch``: a file the shim creates WHILE rendering the plan
    (an approval written before the plan is complete -- the sequence the window must refuse), empty unless ``shim_touch_content`` gives its
    one line; ``state_dir``: the window's state root (default under this home; a second window shares the first one's to meet its lock);
    ``window_id``: the window's operation id (the reservation's ``--op``, the job names, the plan file; None = unset, the window mints its
    pid-bearing default -- for the tests that assert that id or run several windows over one state).

    The systemctl shim also answers ``show -p MainPID|ControlGroup --value UNIT`` from side files (``shim-mainpid-<unit>``,
    ``shim-cgroup-<unit>``; a ``shim-countdown-<unit>`` makes the PID vanish and the cgroup empty after N MainPID queries -- the in-flight
    retired reader of R13) and ``is-active`` reports ``activating`` for the first N queries named in ``shim-activating-<unit>`` (a cycle in
    flight the drain waits out). The cgroup root is ``$WINDOW_CGROUP_ROOT`` = the temp ``cgroup/`` directory. The git shim reports HEAD from
    ``P["head"]`` (rewrite it to advance the base), and the base carries a synthetic migrations directory ``P["mig_dir"]`` with one file
    (the files the pins read)."""
    home = tmp_path / "home"
    bin_ = tmp_path / "bin"
    base = home / "checkout"
    unit_dir = home / ".config" / "systemd" / "user"
    cgroup = tmp_path / "cgroup"
    mig_dir = base / "src" / "sisteer" / "registry" / "migrations"
    state_dir = state_dir or home / ".local" / "state" / "registry-landing"
    for d in (home, bin_, base / "configs" / "registry", unit_dir, home / "registry", cgroup, mig_dir):
        d.mkdir(parents=True, exist_ok=True)
    (base / "configs" / "registry" / "model.yaml").write_text("x: 1\n")
    # the base carries the repository's unit sources, as a real checkout does: an installer that renders them from the site's checkout
    # (SISTEER_REPO, the base here) reads them there
    if not (base / "systemd").exists():
        (base / "systemd").symlink_to(os.path.dirname(UNITS))
    (mig_dir / "0001_synthetic.sql").write_text("CREATE TABLE synthetic (id INTEGER);\n")
    head = tmp_path / "shim-head"
    head.write_text(FULL_SHA + "\n")
    (home / "registry" / "registry.sqlite").write_text("")
    for t in pre_installed:
        (unit_dir / t).write_text("[Timer]\n")
    # the site config the window reads its cluster names from (site_env.sh) and the installer renders the units with
    write_site(home / ".config" / "sisteer", repo=str(base), registry_root=str(home / "registry"),
               clusters={"site_one": {"CLUSTER_SSH": "login-shim", "CLUSTER_SHARED_ROOT": "/shared"}})
    calls = tmp_path / "calls.log"
    pypaths = tmp_path / "pypaths.log"                     # the registry python shim's second record: the PYTHONPATH each call ran under
    state = tmp_path / "shim-active"                       # the units the shim reports active: stop removes, start / enable --now adds
    state.write_text(" ".join(active))
    maint = home / "registry" / "state" / "maintenance.json"
    _shim(bin_, "systemctl", f'''echo "systemctl $*" >> "{calls}"
[ "$1" = "--user" ] && shift
act=" $(cat "{state}") "
cmd="$1"; shift
case "$cmd" in
  is-active)
    quiet=0; [ "$1" = "--quiet" ] && {{ quiet=1; shift; }}
    rc=0
    for u in "$@"; do
      cd="{tmp_path}/shim-activating-$u"
      if [ -f "$cd" ] && [ "$(cat "$cd")" -gt 0 ]; then echo $(( $(cat "$cd") - 1 )) > "$cd"; [ $quiet = 1 ] || echo activating; rc=3; continue; fi
      case "$act" in *" $u "*) [ $quiet = 1 ] || echo active ;; *) [ $quiet = 1 ] || echo inactive; rc=3 ;; esac
    done
    exit $rc ;;
  stop)
    # the drain flag must already be on disk when the timers stop (C6): recorded beside the call
    [ -e "{maint}" ] && echo "maint-at-stop=present" >> "{calls}" || echo "maint-at-stop=absent" >> "{calls}"
    rc=0
    for u in "$@"; do
      case " ${{SHIM_REFUSE_STOP:-}} " in *" $u "*) echo "Failed to stop $u: refused (shim)" >&2; rc=1; continue ;; esac
      act="${{act// $u / }}"
    done
    echo "$act" > "{state}"; exit $rc ;;
  start|enable)
    rc=0
    for u in "$@"; do
      case "$u" in --*) continue ;; esac
      case " ${{SHIM_REFUSE_START:-}} " in *" $u "*) echo "Failed to start $u: refused (shim)" >&2; rc=1; continue ;; esac
      act="$act$u "
    done
    echo "$act" > "{state}"; exit $rc ;;
  show)
    prop=""; unit=""
    while [ $# -gt 0 ]; do case "$1" in -p) prop="$2"; shift 2 ;; --value) shift ;; *) unit="$1"; shift ;; esac; done
    case "$prop" in
      MainPID)
        f="{tmp_path}/shim-mainpid-$unit"; cd="{tmp_path}/shim-countdown-$unit"; cg="{tmp_path}/shim-cgroup-$unit"
        if [ -f "$cd" ]; then
          n=$(cat "$cd")
          if [ "$n" -gt 0 ]; then echo $((n - 1)) > "$cd"; else echo 0 > "$f"; [ -f "$cg" ] && : > "{cgroup}$(cat "$cg")/cgroup.procs"; fi
        fi
        [ -f "$f" ] && cat "$f" || echo 0 ;;
      ControlGroup) f="{tmp_path}/shim-cgroup-$unit"; [ -f "$f" ] && cat "$f" || echo "" ;;
      *) echo "" ;;
    esac
    exit 0 ;;
  *) exit 0 ;;
esac
''')
    _shim(bin_, "git", f'''echo "git $*" >> "{calls}"
case "$*" in
  *"rev-parse --short HEAD"*) head -c 7 "{head}"; echo ;;
  *"rev-parse HEAD"*) cat "{head}" ;;
  *"status --short"*) : ;;
esac
exit 0
''')
    _shim(bin_, "fuser", f'echo "fuser $*" >> "{calls}"; exit 1\n')
    _shim(bin_, "systemd-analyze", f'echo "systemd-analyze $*" >> "{calls}"; exit 0\n')
    fail = f'  {fail_on}) echo "boom" >&2; exit 1 ;;\n' if fail_on else ""
    plan_echo = "; ".join(f'echo "{line}"' for line in (plan_lines or DEFAULT_PLAN))
    touch = (f'printf "%s\\n" "{shim_touch_content}" > "{shim_touch}"; ' if shim_touch_content is not None else f'touch "{shim_touch}"; ') if shim_touch else ""
    _shim(bin_, "regpy", f'''echo "python $*" >> "{calls}"
echo "${{PYTHONPATH:-}} $*" >> "{pypaths}"
case "$*" in
{fail}  *"cluster-reads reserve"*) [ -n "${{SHIM_RESERVE_TOUCH:-}}" ] && : > "$SHIM_RESERVE_TOUCH"; echo "reserve: taken for window $WINDOW_ID (shim)"; exit ${{SHIM_RESERVE_RC:-0}} ;;
  *"cluster-reads release"*) echo "release: released (shim)" ;;
  *"cluster-reads receipt"*) echo "receipt: shim" ;;
  *"migrate --plan"*) {touch}{plan_echo} ;;
  *"stores check-authority"*) n=$(cat "{tmp_path}/shim-authority-n" 2>/dev/null || echo 0); n=$((n + 1)); echo "$n" > "{tmp_path}/shim-authority-n"
    f="{tmp_path}/shim-check-authority-$n.json"
    if [ -f "$f" ]; then cat "$f"; exit "$(cat "{tmp_path}/shim-check-authority-$n.rc" 2>/dev/null || echo 0)"; fi
    echo '{{"ok": true, "n_components": 0, "n_checked": 0, "n_certified": 0, "n_problems": 0, "problems": [], "components": {{}}}}' ;;
  *"stores verify-authority"*) slug=""; while [ $# -gt 0 ]; do [ "$1" = "--component" ] && slug="$2"; shift; done
    echo "{{\\"slug\\": \\"$slug\\", \\"state\\": \\"verified\\"}}"; exit ${{SHIM_VERIFY_AUTHORITY_RC:-0}} ;;
  *"stores mirror"*) echo '{{"summary": true, "components": 1, "ok": 1}}'; exit "$(cat "{tmp_path}/shim-mirror-rc" 2>/dev/null || echo 0)" ;;
  *"adapters run"*) echo '{{"records_written": 0}}'; echo '{{"ingest_ok": 0, "refused": 0, "blocked": 0}}' ;;
  *"reconcile "*" --verify"*) n=$(cat "{tmp_path}/shim-verify-n" 2>/dev/null || echo 0); n=$((n + 1)); echo "$n" > "{tmp_path}/shim-verify-n"
    f="{tmp_path}/shim-verify-$n.out"
    if [ -f "$f" ]; then cat "$f"; else echo "inconsistent 0 run(s); the class is consistent"; fi
    exit "$(cat "{tmp_path}/shim-verify-$n.rc" 2>/dev/null || echo 0)" ;;
  *"reconcile env-comments"*" --check-sources"*) f="{tmp_path}/shim-check-sources.out"
    if [ -f "$f" ]; then cat "$f"; else echo "uncovered 0 run(s); every stamp whose values may carry a comment has its file"; fi
    exit "$(cat "{tmp_path}/shim-check-sources.rc" 2>/dev/null || echo 0)" ;;
  *" ingest"*) echo '{{"files": 0, "ok": 0}}' ;;
  *) : ;;
esac
exit 0
''')
    env = {"PATH": f"{bin_}:{os.environ['PATH']}", "HOME": str(home), "REGISTRY_PY": str(bin_ / "regpy"), "REGISTRY_PYTHONPATH": str(base / "src"),
           "WINDOW_NEW_PY": str(bin_ / "regpy"), "WINDOW_NEW_PYPATH": str(home / "worktree" / "src"), "WINDOW_CONVERT_PY": sys.executable,
           "WINDOW_INSTALLER": installer or INSTALL, "WINDOW_LOG_COMPAT": str(tmp_path / "compat.log"), "WINDOW_QUIET_SLEEP_S": "0",
           "WINDOW_CONFIRM_SLEEP_S": "0", "WINDOW_CGROUP_ROOT": str(cgroup), "ADAPTERS": adapters, "WINDOW_BASE_BUSY_PATTERN": "",
           "CLUSTER_ADAPTERS": "none", "WINDOW_STATE_DIR": str(state_dir)}
    if window_id is not None:
        env["WINDOW_ID"] = window_id
    env.update(extra_env or {})
    return env, {"home": home, "unit_dir": unit_dir, "calls": calls, "pypaths": pypaths, "log": state_dir / "windows" / "chain.log",
                 "state": state_dir, "compat": tmp_path / "compat.log", "maint": maint, "cgroup": cgroup, "tmp": tmp_path, "head": head,
                 "mig_dir": mig_dir, "base": base, "bin": bin_}


def _run(env):
    return subprocess.run(["bash", WINDOW], capture_output=True, text=True, env=env)


def _stamps(log_text):
    """The stage stamps in order, as windows.py reads them (label without the HH:MM:SS; a REFUSED stamp without its reason)."""
    out = []
    for line in log_text.splitlines():
        if line.startswith("=== ") or line.startswith("-- ") or line.startswith("--- ") or line.startswith("base at") or line.startswith("writers quiet") \
                or line.startswith("timers restored"):
            out.append(re.sub(r"\s\d\d:\d\d:\d\d.*$", "", re.sub(r"\s\d{4}-\d\d-\d\dT\S+", "", line)).strip())
    return out


def _calls(P):
    return P["calls"].read_text().splitlines()


def _starts(calls):
    return [c for c in calls if re.search(r"systemctl --user (start|enable --now) ", c)]


def _assert_no_retired_start(calls):
    """No exit path ever starts or enables a retired unit (C6; R13's assertion over the set of every systemctl call)."""
    for c in calls:
        if re.search(r"systemctl --user (start|enable)", c):
            assert "registry-adapters" not in c and "registry-modal-pull" not in c and "registry-snapshot" not in c, c


# --------------------------------------------------------------------------------------------
# Phase 1: the stage order, the restore, the end marker in the trap, the maintenance flag, the retired readers
# --------------------------------------------------------------------------------------------


def test_window_stage_order_restore_set_never_start_the_flag_and_one_ingest_end_with_the_marker_after_the_restore(tmp_path):
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer", "registry-adapters.timer", "registry-status.timer"], pre_installed=OLD_TIMERS)
    r = _run(env)
    assert r.returncode == 0, r.stderr + P["log"].read_text()
    log = P["log"].read_text()
    assert _stamps(log) == ["=== ff chain start ADAPTERS=", "-- reads reservation", "writers quiet", "-- wal checkpoint", "-- config convert", "-- store authority", "base at deadbee", "-- install units", "-- uninstall units",
                            "-- migrate (plan)", "-- migrate (apply)", "-- dimensions load", "--- ingest", "--- ingest end", "timers restored (trap; the set above)",
                            "-- status heavy refresh started once (cache initializer)", "=== ff chain end"], _stamps(log)
    calls = _calls(P)
    sysctl = [c for c in calls if c.startswith("systemctl")]
    # stop first (the timers AND the two retired services); the restore starts the active set minus NEVER_START and enables the newly
    # installed timers; the retired units and the snapshot timer are never started
    assert sysctl[0].startswith("systemctl --user is-active --quiet")
    assert [c for c in sysctl if " stop " in c] == [STOP_CALL]
    starts = _starts(calls)
    assert starts == ["systemctl --user start registry-harvest.timer registry-status.timer",
                      f"systemctl --user enable --now {NEW_TIMERS_STR}",
                      "systemctl --user start --no-block registry-status-heavy.service"], starts
    _assert_no_retired_start(calls)
    assert "systemctl --user daemon-reload" in sysctl and sysctl.index("systemctl --user daemon-reload") < sysctl.index(starts[0])
    # C7: the reservation is the first registry call, before the stop; the trap releases it after the flag and before the restore
    pys = [c for c in calls if c.startswith("python")]
    assert "cluster-reads reserve --owner window --op W1 --takeover --state" in pys[0] and f"--dest {SUB_A}" in pys[0], pys[0]
    # the holder the reservation names is the WINDOW SHELL's pid ($$), not the CLI process in the $(...) -- the
    # same pid every call of this window runs under (the shim records the invoking shell's pid on every call: `$PPID` of the shim is it)
    pid = re.search(r" --pid (\d+)(?: |$)", pys[0])
    assert pid and int(pid.group(1)) > 1, pys[0]
    assert "holder pid " + pid.group(1) in log, "the reservation stamp names the holder pid"
    assert calls.index(pys[0]) < calls.index(STOP_CALL)
    rel = next(c for c in calls if "cluster-reads release --op W1 --state" in c)
    assert calls.index(rel) > calls.index(next(c for c in calls if c.endswith("cli ingest --retry-held"))) and calls.index(rel) < calls.index(starts[0])
    lines_ = log.splitlines()
    assert lines_.index(next(l for l in lines_ if l.startswith("reads reservation released"))) < lines_.index(next(l for l in lines_ if l.startswith("timers restored")))
    # C7: the retired unit files are gone after the uninstall stage (disabled, removed, reloaded) and nothing re-copied them
    assert not (P["unit_dir"] / "registry-adapters.timer").exists() and not (P["unit_dir"] / "registry-adapters.service").exists()
    assert any(c == "systemctl --user disable registry-adapters.timer" for c in calls) and any(c == "systemctl --user disable registry-adapters.service" for c in calls)
    assert sysctl.count("systemctl --user daemon-reload") == 2 and \
        "retired units uninstalled (disabled, files removed: registry-adapters.timer registry-modal-pull.timer; daemon reloaded)" in log
    assert not (P["unit_dir"] / "registry-modal-pull.timer").exists() and not (P["unit_dir"] / "registry-modal-pull.service").exists()
    assert (f"restore set (active at stop ∪ newly installed − registry-snapshot.timer registry-adapters.timer registry-modal-pull.timer): "
            f"registry-harvest.timer registry-status.timer {NEW_TIMERS_STR}") in log
    assert "never started: registry-snapshot.timer registry-adapters.timer registry-modal-pull.timer" in log
    # the units were installed from the repo into the temp unit dir by the real installer
    # (the unit sources are plain files or templates the installer renders, `<unit>.in`)
    sources = sorted(f[:-3] if f.endswith(".in") else f for f in os.listdir(UNITS) if f.endswith(".service") or f.endswith(".service.in"))
    assert sorted(f for f in os.listdir(P["unit_dir"]) if f.endswith(".service")) == sources
    # NO adapter pass on the grading host (C6); exactly one ingest, followed by its end stamp
    assert not any("adapters run" in c for c in calls)
    assert sum(1 for c in calls if c.endswith("sisteer.registry.cli ingest --retry-held")) == 1
    # the git stage ran on the base and the base fast-forwarded before the install
    gits = [c for c in calls if c.startswith("git")]
    assert any("fetch" in c for c in gits) and any("merge -q --ff-only origin/main" in c for c in gits)
    assert calls.index([c for c in calls if "merge -q --ff-only" in c][0]) < calls.index("systemctl --user daemon-reload")
    # the maintenance flag was on disk when the timers stopped, and is gone after the trap; the retired readers were confirmed before the base moved
    assert "maint-at-stop=present" in calls and not P["maint"].exists()
    assert "maintenance flag written" in log and "maintenance flag removed" in log
    lines = log.splitlines()
    i_conf = next(i for i, l in enumerate(lines) if l.startswith("retired readers stopped and confirmed"))
    i_base = next(i for i, l in enumerate(lines) if l.startswith("base at"))
    i_stop = next(i for i, l in enumerate(lines) if l.startswith("maintenance flag written"))
    assert i_stop < i_conf < i_base
    assert any("show -p MainPID --value registry-adapters.service" in c for c in calls) and any("show -p ControlGroup --value registry-modal-pull.service" in c for c in calls)
    # the end marker is the LAST stage line, after `timers restored`: the measured window includes the restore
    assert lines.index(next(l for l in lines if l.startswith("=== ff chain end"))) > lines.index(next(l for l in lines if l.startswith("timers restored")))
    # the compat log received the window's slice as the last act
    assert P["compat"].read_text() == log[-len(P["compat"].read_text()):] and "=== ff chain start" in P["compat"].read_text()
    assert (P["home"] / "registry" / ".dimensions-last-load").exists()


def test_adapters_none_is_an_adapter_less_window_not_an_adapter_named_none(tmp_path):
    """The literal ``none`` is the EMPTY adapter set for ADAPTERS as it is for CLUSTER_ADAPTERS. An adapter-less restamp window run with
    ADAPTERS=none iterated an adapter NAMED none, stamped it ``-- adapter none REFUSED (no cluster policy named for it this window ...)``
    and ended ``ABORTED (refused: none)`` with exit 20 after its work had succeeded, because the fold to the empty set was applied to
    CLUSTER_ADAPTERS alone. Folded for ADAPTERS too: the window runs exactly as with an empty ADAPTERS -- the header reads ``ADAPTERS=``
    (the empty list, as windows.py's start line reads it), no SKIPPED or REFUSED stamp, no ``adapters run``, one ingest,
    ``=== ff chain end``, exit 0."""
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer", "registry-adapters.timer", "registry-status.timer"], pre_installed=OLD_TIMERS,
                        adapters="none")
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    assert "-- adapter none REFUSED" not in log and "ABORTED" not in log and "refused: none" not in log, log
    assert "=== ff chain end" in log and "SKIPPED" not in log
    assert _stamps(log) == ["=== ff chain start ADAPTERS=", "-- reads reservation", "writers quiet", "-- wal checkpoint", "-- config convert", "-- store authority", "base at deadbee", "-- install units", "-- uninstall units",
                            "-- migrate (plan)", "-- migrate (apply)", "-- dimensions load", "--- ingest", "--- ingest end", "timers restored (trap; the set above)",
                            "-- status heavy refresh started once (cache initializer)", "=== ff chain end"], _stamps(log)
    assert not any("adapters run" in c for c in _calls(P))
    start = next(l for l in log.splitlines() if l.startswith("=== ff chain start"))
    assert start.rstrip().endswith(" ADAPTERS="), start                        # the header carries the empty list, never the literal none


def test_install_failure_aborts_before_anything_else_with_the_timers_restored_the_flag_removed_and_the_aborted_marker(tmp_path):
    fail = tmp_path / "fail-install.sh"
    fail.write_text("#!/bin/bash\necho 'boom' >&2\nexit 1\n")
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS, installer=str(fail))
    r = _run(env)
    assert r.returncode == 7, r.stderr + P["log"].read_text()
    log = P["log"].read_text()
    assert _stamps(log) == ["=== ff chain start ADAPTERS=", "-- reads reservation", "writers quiet", "-- wal checkpoint", "-- config convert", "-- store authority", "base at deadbee", "-- install units", "timers restored (trap; the set above)",
                            "=== ff chain ABORTED (install units FAILED rc=1)"], _stamps(log)
    assert "-- uninstall units" not in log                      # the install failed: nothing is uninstalled either
    assert "install units FAILED rc=1; abort" in log and "window ended early (rc=7)" in log and "=== ff chain end" not in log
    calls = _calls(P)
    assert not any("adapters run" in c or c.endswith("cli ingest --retry-held") or "migrate" in c for c in calls)
    # the timers active at stop come back; a "newly installed" timer has no unit file when the install failed, so none is enabled;
    # the heavy refresh is NOT started (nothing was installed); the flag is gone
    assert _starts(calls) == ["systemctl --user start registry-harvest.timer"], _starts(calls)
    assert not any("start --no-block" in c for c in calls) and not P["maint"].exists()
    _assert_no_retired_start(calls)


def test_a_named_adapter_without_a_cluster_read_is_recorded_refused_never_run_on_the_grading_host_and_the_window_ends_aborted(tmp_path):
    """C6, S2: the Phase-1 the grading host pass is gone. Two named adapters with no cluster policy (CLUSTER_ADAPTERS=none) are stamped REFUSED with the
    reason, no `adapters run` is ever called, the ingest of the rest runs, and the window ends `ABORTED (refused: alpha beta)` with exit 20;
    windows.py reads the window as not completed, the cause on the record, the two stamps classed `adapter_refused`."""
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS + NEW_TIMERS, adapters="alpha beta")
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 20, r.stderr + log
    assert _stamps(log) == ["=== ff chain start ADAPTERS=alpha beta", "-- reads reservation", "writers quiet", "-- wal checkpoint", "-- config convert", "-- store authority", "base at deadbee", "-- install units",
                            "-- uninstall units", "-- migrate (plan)", "-- migrate (apply)",
                            "-- dimensions load", "-- cluster reads SKIPPED (no named adapter has a cluster policy)", "-- adapter alpha REFUSED",
                            "-- adapter beta REFUSED", "--- ingest", "--- ingest end", "timers restored (trap; the set above)",
                            "-- status heavy refresh started once (cache initializer)", "=== ff chain ABORTED (refused: alpha beta)"], _stamps(log)
    assert re.search(r"^-- adapter alpha REFUSED \d\d:\d\d:\d\d \(no cluster policy named for it this window \(CLUSTER_ADAPTERS\); the grading host reads no store\)$", log, re.M)
    assert "refused: alpha beta; abort" in log and "window ended early (rc=20)" in log
    calls = _calls(P)
    assert not any("adapters run" in c for c in calls) and sum(1 for c in calls if c.endswith("cli ingest --retry-held")) == 1
    windows = _load("windows").read_logs([str(P["log"])])
    assert len(windows) == 1 and windows[0]["completed"] is False and windows[0]["abort_cause"] == "refused: alpha beta" and windows[0]["end_definition"] == "rev3"
    assert [s["class"] for s in windows[0]["stages"]] == ["reservation", "wal", "config_convert", "authority", "base", "install", "install", "migrate_plan", "migrate", "dims", "cluster_skip", "adapter_refused", "adapter_refused", "ingest", "ingest_end", "status_heavy"]


def test_keep_timers_stopped_leaves_them_stopped_names_the_set_and_still_writes_the_end_marker(tmp_path):
    env, P = _rehearsal(tmp_path, active=["registry-adapters.timer"], pre_installed=OLD_TIMERS, extra_env={"KEEP_TIMERS_STOPPED": "1"})
    r = _run(env)
    assert r.returncode == 0, r.stderr
    log = P["log"].read_text()
    assert "timers LEFT STOPPED by request (KEEP_TIMERS_STOPPED=1); active at stop were: registry-adapters.timer" in log
    assert "=== ff chain end" in log and not P["maint"].exists()
    assert not any(re.search(r"systemctl --user (start|enable --now) ", c) for c in _calls(P))


def test_window_log_is_parsed_by_the_measurement_script_as_a_rev3_window(tmp_path):
    """The stamps window.sh writes are the ones windows.py reads: one completed window, `rev3` (the end after the restore), its stages classed."""
    env, P = _rehearsal(tmp_path, active=[], pre_installed=OLD_TIMERS)
    assert _run(env).returncode == 0
    windows = _load("windows").read_logs([str(P["log"]), str(P["compat"])])
    assert len(windows) == 1 and windows[0]["completed"] and windows[0]["base"] == "deadbee" and windows[0]["adapters"] == ""
    assert windows[0]["end_definition"] == "rev3" and windows[0]["abort_cause"] is None
    classes = [s["class"] for s in windows[0]["stages"]]
    assert classes == ["reservation", "wal", "config_convert", "authority", "base", "install", "install", "migrate_plan", "migrate", "dims", "ingest", "ingest_end", "status_heavy"], classes      # the trap's initializer stamp is inside the window now
    assert windows[0]["quiet_wait_s"] is not None and windows[0]["total_s"] is not None


def test_migrate_failure_aborts_before_dimensions_and_the_ingest_with_the_timers_restored_and_the_aborted_marker(tmp_path):
    """A failing `registry migrate` (the apply) ends the window at once: no dimensions load, no ingest, no end marker but an ABORTED one
    naming the cause; the trap still restores the timers and — the units WERE installed — enables the new timers and starts the initializer."""
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS, fail_on='*"cli migrate"')
    r = _run(env)
    assert r.returncode == 8, r.stderr + P["log"].read_text()
    log = P["log"].read_text()
    assert _stamps(log) == ["=== ff chain start ADAPTERS=", "-- reads reservation", "writers quiet", "-- wal checkpoint", "-- config convert", "-- store authority", "base at deadbee", "-- install units", "-- uninstall units",
                            "-- migrate (plan)", "-- migrate (apply)", "timers restored (trap; the set above)", "-- status heavy refresh started once (cache initializer)",
                            "=== ff chain ABORTED (migrate FAILED rc=1)"], _stamps(log)
    assert "migrate FAILED rc=1; abort" in log and "window ended early (rc=8)" in log and "=== ff chain end" not in log
    calls = _calls(P)
    assert any(c.endswith("sisteer.registry.cli migrate") for c in calls)
    assert "plan: nothing pending, the shape matches; no approval needed" in log
    assert not any("dimensions load" in c or "adapters run" in c or c.endswith("cli ingest --retry-held") for c in calls), calls
    assert _starts(calls) == ["systemctl --user start registry-harvest.timer", f"systemctl --user enable --now {NEW_TIMERS_STR}",
                              "systemctl --user start --no-block registry-status-heavy.service"], _starts(calls)
    _assert_no_retired_start(calls)


def test_the_operator_override_reread_is_the_one_grading_host_run_left_and_is_refused_without_the_override(tmp_path):
    """`--reread --force` reads the store on the grading host: an adapter in $REREAD_FORCE_ADAPTERS runs here ONLY when REGISTRY_STORE_READ_OVERRIDE
    is set (the override the registry logs), stamped as such and carrying --no-ingest --retry-refused; a failing pass is the window's exit 10
    as before. Without the override it is recorded refused with the reason naming the variable, and nothing runs."""
    env, P = _rehearsal(tmp_path, active=[], pre_installed=OLD_TIMERS + NEW_TIMERS, adapters="alpha",
                        extra_env={"REREAD_FORCE_ADAPTERS": "alpha", "REGISTRY_STORE_READ_OVERRIDE": "drill: a corrected adapter"})
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    assert "-- adapter alpha --reread --force [operator override: drill: a corrected adapter]" in log and "REFUSED" not in log and "=== ff chain end" in log
    calls = _calls(P)
    assert any(c.endswith("adapters run --reread --force --no-ingest --retry-refused --adapter alpha") for c in calls), calls
    # a failing override pass: exit 10, ABORTED with the pass named
    d = tmp_path / "fail"
    d.mkdir()
    env2, P2 = _rehearsal(d, active=[], pre_installed=OLD_TIMERS + NEW_TIMERS, adapters="alpha", fail_on='*"--adapter alpha"',
                          extra_env={"REREAD_FORCE_ADAPTERS": "alpha", "REGISTRY_STORE_READ_OVERRIDE": "drill"})
    r2 = _run(env2)
    log2 = P2["log"].read_text()
    assert r2.returncode == 10, r2.stderr + log2
    assert "adapter alpha rc=1" in log2 and "adapter pass FAILED (alpha rc=1); abort" in log2 and "=== ff chain ABORTED (adapter pass FAILED (alpha rc=1))" in log2
    assert not any(c.endswith("cli ingest --retry-held") for c in _calls(P2))
    # without the override: refused by name, nothing runs, the window ends ABORTED (refused: alpha)
    d3 = tmp_path / "noov"
    d3.mkdir()
    env3, P3 = _rehearsal(d3, active=[], pre_installed=OLD_TIMERS + NEW_TIMERS, adapters="alpha", extra_env={"REREAD_FORCE_ADAPTERS": "alpha"})
    r3 = _run(env3)
    log3 = P3["log"].read_text()
    assert r3.returncode == 20, r3.stderr + log3
    assert re.search(r"^-- adapter alpha REFUSED \d\d:\d\d:\d\d \(reread --force reads the store on the grading host; set REGISTRY_STORE_READ_OVERRIDE=<reason>", log3, re.M)
    assert not any("adapters run" in c for c in _calls(P3)) and "=== ff chain ABORTED (refused: alpha)" in log3


def test_ingest_failure_aborts_without_the_end_marker_so_the_window_counts_as_aborted(tmp_path):
    """A failing final ingest exits 12 with `--- ingest` stamped, NO `--- ingest end` and `=== ff chain ABORTED (ingest FAILED rc=1)`: the
    measurement script reads the window as ABORTED with the cause, never as a completed window; the timers are restored and the compat log
    still receives the slice."""
    env, P = _rehearsal(tmp_path, active=["registry-adapters.timer"], pre_installed=OLD_TIMERS + NEW_TIMERS, fail_on='*"cli ingest --retry-held"')
    r = _run(env)
    assert r.returncode == 12, r.stderr + P["log"].read_text()
    log = P["log"].read_text()
    stamps = _stamps(log)
    assert stamps[:13] == ["=== ff chain start ADAPTERS=", "-- reads reservation", "writers quiet", "-- wal checkpoint", "-- config convert", "-- store authority", "base at deadbee", "-- install units", "-- uninstall units",
                           "-- migrate (plan)", "-- migrate (apply)", "-- dimensions load", "--- ingest"], stamps
    assert "--- ingest end" not in stamps and "=== ff chain end" not in log and "timers restored (trap; the set above)" in stamps
    assert stamps[-1] == "=== ff chain ABORTED (ingest FAILED rc=1)"
    assert "ingest FAILED rc=1; abort" in log and "window ended early (rc=12)" in log
    calls = _calls(P)
    assert "systemctl --user start registry-adapters.timer" not in calls          # a retired unit is never started (NEVER_START)
    _assert_no_retired_start(calls)
    assert "=== ff chain start" in P["compat"].read_text()
    windows = _load("windows").read_logs([str(P["log"])])
    assert len(windows) == 1 and windows[0]["completed"] is False and windows[0]["end"] is None and windows[0]["abort_cause"] == "ingest FAILED rc=1"


def test_a_timer_that_does_not_stop_fails_the_window_before_the_base_moves(tmp_path):
    """`systemctl stop` refused for one timer: it still reads active after the stop, so the window exits 13 at once — no quiet wait,
    no fetch, no fast-forward, no install — and the trap restarts the timers that were active (verified active), the flag removed."""
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer", "registry-status.timer"], pre_installed=OLD_TIMERS,
                        extra_env={"SHIM_REFUSE_STOP": "registry-status.timer"})
    r = _run(env)
    assert r.returncode == 13, r.stderr + P["log"].read_text()
    log = P["log"].read_text()
    assert "stop returned rc=1" in log and "timers not stopped (read other than inactive): registry-status.timer" in log
    assert "window ended early (rc=13)" in log and "writers quiet" not in log and "base at" not in log and "-- install units" not in log
    assert "=== ff chain ABORTED (timers not stopped (read other than inactive): registry-status.timer )" in log
    calls = _calls(P)
    assert not any(c.startswith("git") or (c.startswith("python") and "cluster-reads reserve" not in c and "cluster-reads release" not in c) for c in calls), calls
    assert "systemctl --user start registry-harvest.timer registry-status.timer" in calls
    assert "timers restored (trap; the set above)" in log and not any("--no-block" in c for c in calls) and not P["maint"].exists()


def test_a_restored_timer_that_does_not_read_active_fails_the_window_from_the_trap_with_aborted_restore(tmp_path):
    """Every stage succeeded but `systemctl start` is refused for one timer of the restore set: the trap reads it inactive, logs the set,
    writes `=== ff chain ABORTED (restore)` -- NEVER `=== ff chain end` -- and the window exits 14.
    The heavy initializer still starts (the units were installed) and the compat log still receives the slice."""
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer", "registry-status.timer"], pre_installed=OLD_TIMERS,
                        extra_env={"SHIM_REFUSE_START": "registry-status.timer"})
    r = _run(env)
    assert r.returncode == 14, r.stderr + P["log"].read_text()
    log = P["log"].read_text()
    assert "=== ff chain end" not in log and "=== ff chain ABORTED (restore)" in log and "window ended early" not in log
    assert "--- ingest end" in log
    assert "start failed for: registry-harvest.timer registry-status.timer" in log
    assert "timers NOT restored: not active after the start: registry-status.timer (trap)" in log and "timers restored (trap" not in log
    calls = _calls(P)
    assert any("start --no-block registry-status-heavy.service" in c for c in calls)
    assert "=== ff chain start" in P["compat"].read_text() and "timers NOT restored" in P["compat"].read_text()
    windows = _load("windows").read_logs([str(P["log"])])
    assert windows[0]["completed"] is False and windows[0]["abort_cause"] == "restore" and windows[0]["end_definition"] == "rev3"
    # an earlier stage's own code stands over the restore's: a failing ingest AND a refused start exits 12, the marker names both
    env2, P2 = _rehearsal(tmp_path / "second", active=["registry-status.timer"], pre_installed=OLD_TIMERS,
                          extra_env={"SHIM_REFUSE_START": "registry-status.timer"}, fail_on='*"cli ingest --retry-held"')
    r2 = _run(env2)
    assert r2.returncode == 12, r2.stderr + P2["log"].read_text()
    log2 = P2["log"].read_text()
    assert "timers NOT restored" in log2 and "window ended early (rc=12)" in log2 and "=== ff chain ABORTED (ingest FAILED rc=1; restore)" in log2


# --------------------------------------------------------------------------------------------
# an in-flight retired reader is stopped and CONFIRMED by its cgroup PID list before the base moves
# --------------------------------------------------------------------------------------------


def _retired_reader(P, unit="registry-adapters.service", pids=("4242", "4243"), countdown=None):
    """A retired service with a main PID and a populated cgroup; ``countdown`` = the number of MainPID queries after which the shim reports the
    process gone and empties the cgroup (None = it never does)."""
    cg = f"/user.slice/user-1000.slice/user@1000.service/app.slice/{unit}"
    procs = P["cgroup"] / cg.lstrip("/") / "cgroup.procs"
    procs.parent.mkdir(parents=True, exist_ok=True)
    procs.write_text("".join(p + "\n" for p in pids))
    (P["tmp"] / f"shim-mainpid-{unit}").write_text(pids[0] + "\n")
    (P["tmp"] / f"shim-cgroup-{unit}").write_text(cg + "\n")
    if countdown is not None:
        (P["tmp"] / f"shim-countdown-{unit}").write_text(str(countdown))
    return procs


def test_r13_an_in_flight_retired_reader_is_stopped_confirmed_terminal_and_its_cgroup_emptied_before_the_base_moves(tmp_path):
    """R13: `registry-adapters.service` is `activating` with a child process when the window starts. The window stops it with the timers,
    then polls: `is-active` terminal, `show -p MainPID` 0, the cgroup's `cgroup.procs` EMPTY -- here the shim's process lingers for two
    polls and then exits -- and only then `base at`. The set of every systemctl call never starts or enables a retired unit."""
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer", "registry-adapters.timer", "registry-adapters.service"], pre_installed=OLD_TIMERS + NEW_TIMERS)
    procs = _retired_reader(P, countdown=2)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    calls = _calls(P)
    pid_polls = [c for c in calls if "show -p MainPID --value registry-adapters.service" in c]
    assert len(pid_polls) == 3, pid_polls                      # two polls saw the lingering process, the third saw it gone
    assert procs.read_text() == ""                              # the cgroup emptied before the base moved
    lines = log.splitlines()
    i_conf = next(i for i, l in enumerate(lines) if l.startswith("retired readers stopped and confirmed"))
    i_base = next(i for i, l in enumerate(lines) if l.startswith("base at"))
    assert i_conf < i_base
    # the stop named the service; the retired units are started by no path; the restore brings the harvest timer back and nothing retired
    assert STOP_CALL in calls
    _assert_no_retired_start(calls)
    assert _starts(calls)[0] == "systemctl --user start registry-harvest.timer"
    assert "=== ff chain end" in log


def test_r13_a_retired_reader_whose_cgroup_never_empties_fails_the_window_with_the_pids_named_and_nothing_started_or_killed(tmp_path):
    """The negative half: the process never leaves the cgroup. After WINDOW_CONFIRM_TRIES polls the window exits 21 naming the unit and
    the PIDs, BEFORE the quiet wait and the base move; the trap restores the harvest timer and never a retired unit; nothing is killed."""
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer", "registry-adapters.service"], pre_installed=OLD_TIMERS,
                        extra_env={"WINDOW_CONFIRM_TRIES": "3"})
    procs = _retired_reader(P, countdown=None)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 21, r.stderr + log
    assert "retired reader still running after the stop: registry-adapters.service(MainPID=4242) registry-adapters.service(cgroup pids: 4242 4243); abort" in log
    assert "=== ff chain ABORTED (retired reader still running after the stop: registry-adapters.service(MainPID=4242) registry-adapters.service(cgroup pids: 4242 4243))" in log
    assert "writers quiet" not in log and "base at" not in log and "-- install units" not in log
    assert procs.read_text() == "4242\n4243\n"                  # untouched: nothing kills by text
    calls = _calls(P)
    assert not any(c.startswith("git") or (c.startswith("python") and "cluster-reads reserve" not in c and "cluster-reads release" not in c) for c in calls)
    assert _starts(calls) == ["systemctl --user start registry-harvest.timer"], _starts(calls)
    _assert_no_retired_start(calls)
    assert not P["maint"].exists()


# --------------------------------------------------------------------------------------------
# the honest drain: the flag before the stop, a cycle in flight waited out, `ingest --honor-maintenance` at the door
# --------------------------------------------------------------------------------------------


def test_the_drain_waits_out_a_harvest_cycle_in_flight_with_the_flag_already_written(tmp_path):
    """A mid-cycle start: the harvest service reports `activating` for the first three queries after the stop. The flag is on disk before
    the stop (so the NEXT cycle the timer would fire exits at the door), the quiet wait polls until the service is inactive, and only then
    `writers quiet` and the base move."""
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS)
    (P["tmp"] / "shim-activating-registry-harvest.service").write_text("3")
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    calls = _calls(P)
    assert "maint-at-stop=present" in calls
    quiet_polls = [c for c in calls if c.startswith("systemctl --user is-active registry-harvest.service")]
    assert len(quiet_polls) >= 4, quiet_polls                   # three `activating` answers, then inactive
    assert "writers quiet" in log and "=== ff chain end" in log and not P["maint"].exists()


def test_ingest_honor_maintenance_exits_at_the_door_before_the_pull_and_the_lock_and_the_claim_path_ignores_the_flag(tmp_path, monkeypatch, capsys):
    """`registry ingest --honor-maintenance --pull` with state/maintenance.json present: exit 0, `skipped: maintenance` with the flag's content,
    no cluster pull, no write lock; without the flag the cycle runs; without the flag the plain `ingest` runs too; `harvest.ingest_once`
    itself (the claim path's fold, pending.claim / release / ingest_now) never reads the flag."""
    from sisteer.registry import cli as CLI
    from sisteer.registry import db as DB
    from sisteer.registry import harvest as H
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    assert CLI.main(["--root", root, "init"]) == 0
    capsys.readouterr()
    path = H.write_maintenance(root, "landing window (test)", adapters="alpha")
    assert H.maintenance(root)["reason"] == "landing window (test)" and H.maintenance(root)["adapters"] == "alpha"
    monkeypatch.setattr(H, "pull_clusters", lambda *a, **k: pytest.fail("the pull must not run under the maintenance flag"))
    monkeypatch.setattr(DB, "write_lock", lambda *a, **k: pytest.fail("the write lock must not be taken under the maintenance flag"))
    assert CLI.main(["--root", root, "ingest", "--pull", "--honor-maintenance"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ingest"] == "skipped: maintenance" and out["maintenance"]["reason"] == "landing window (test)" and out["flag"] == path
    # the claim path's fold ignores the flag: ingest_once runs a cycle (an empty root: nothing to fold, no refusal)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.paused is False
    # the flag removed: the cycle runs (a real cycle over the fresh root, under the write lock; no pull in this call)
    assert H.clear_maintenance(root) is True and H.maintenance(root) is None and H.clear_maintenance(root) is False
    monkeypatch.undo()
    assert CLI.main(["--root", root, "ingest", "--honor-maintenance"]) == 0
    out2 = capsys.readouterr().out
    assert "skipped: maintenance" not in out2 and '"last_completed_cycle_utc"' in out2
    # an unreadable flag is still a flag (the drain must hold whatever the file says)
    with open(H.maintenance_path(root), "w") as fh:
        fh.write("{not json")
    assert H.maintenance(root)["reason"].startswith("unreadable maintenance flag")
    assert CLI.main(["--root", root, "ingest", "--honor-maintenance"]) == 0 and "skipped: maintenance" in capsys.readouterr().out


# --------------------------------------------------------------------------------------------
# the cluster-read stages (registry-slurm-fast-windows Phase 3b; C6: refusals, never a grading host read): rehearsed with ssh / watch-job /
# mirror-repos shims and a registry python shim that answers the cluster-reads subcommands from the knobs below
# --------------------------------------------------------------------------------------------


def _cluster_rehearsal(tmp_path, *, refusals="", plan_rc=0, stage_rc=0, submit="4242", pull_rc=0, revalidate_out="revalidation: every component still matches its pin",
                       revalidate_rc=0, reader_state_rc=0, tree_out="TREE exported /shared/gates/trees/deadbee (12 files)", tree_rc=0, watch_out="RUNNING 4242 node=n1\\nDONE 4242 state=COMPLETED",
                       cluster_adapters=None, adapters="heavy_one heavy_two", receipt_rc=0, history_rc=0):
    """The world of `_rehearsal` plus the cluster shims. `refusals`: what `cluster-reads refusals --window W` prints (one adapter per line; the
    --json form carries `plan: shim` as each reason); `submit`: what `cluster-reads submit` prints (a job id, or an error line); the *_rc knobs
    fail one stage. `cluster_adapters` (default: every named adapter, as the window defaults since C6) narrows the cluster set."""
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS + NEW_TIMERS, adapters=adapters)
    bin_ = tmp_path / "bin"
    calls = P["calls"]
    _shim(bin_, "ssh", f'''echo "ssh $*" >> "{calls}"
cat >/dev/null
printf "%b\\n" "{tree_out}"
exit {tree_rc}
''')
    _shim(bin_, "watch-job.sh", f'''echo "watch-job.sh $*" >> "{calls}"
printf "%b\\n" "{watch_out}"
exit 0
''')
    _shim(bin_, "mirror-repos.sh", f'echo "mirror-repos.sh $*" >> "{calls}"; exit 0\n')
    _shim(bin_, "mirror_history.sh", f'echo "mirror_history.sh $*" >> "{calls}"; echo "HISTORY current 0 refs on $2 (shim)"; exit {history_rc}\n')
    fb_lines = "\\n".join(refusals.split()) if refusals else ""
    fb_json = "{" + ", ".join(f'"{a}": ["plan: shim"]' for a in refusals.split()) + "}"
    _shim(bin_, "regpy", f'''echo "python $*" >> "{calls}"
case "$*" in
  "-c "*) cat >/dev/null; echo "plan: shim" ;;
  *"cluster-reads reserve"*) echo "reserve: taken for window $WINDOW_ID (shim)"; exit ${{SHIM_RESERVE_RC:-0}} ;;
  *"cluster-reads release"*) echo "release: released (shim)" ;;
  *"cluster-reads receipt"*) echo "receipt: shim -- 2 adapter(s) read whole"; exit {receipt_rc} ;;
  *"migrate --plan"*) echo "schema version 17; code 17; pending: none"; echo "shape: matches the code's schema" ;;
  *"reader-state export"*) echo '{{"rows": {{"source_revision": 3}}, "bytes": 1234}}'; exit {reader_state_rc} ;;
  *"cluster-reads plan"*) echo "plan {tmp_path}/plan.json: heavy_one=cluster, heavy_two=cluster; 2 task(s)"; exit {plan_rc} ;;
  *"cluster-reads stage"*) echo '{{"window_root": "/shared/registry/reads/W", "rulings_entries": 12}}'; exit {stage_rc} ;;
  *"cluster-reads submit"*) echo "{submit}"; [[ "{submit}" =~ ^([0-9]+|none)$ ]] && exit 0 || exit 2 ;;
  *"cluster-reads revalidate"*) echo "{revalidate_out}"; exit {revalidate_rc} ;;
  *"cluster-reads pull"*) echo "pull: 2 revision(s) complete, 0 quarantined, 0 refused-known revision(s) re-activated from committed bytes; no refusal from the pull"; exit {pull_rc} ;;
  *"cluster-reads refusals"*" --json"*) echo '{fb_json}' ;;
  *"cluster-reads refusals"*) printf "%b" "{fb_lines}"; [ -n "{fb_lines}" ] && echo ;;
  *"inbox-merge"*) echo '{{"merged": 7, "duplicate": 0, "conflict": 0}}' ;;
  *"stores check-authority"*) n=$(cat "{tmp_path}/shim-authority-n" 2>/dev/null || echo 0); n=$((n + 1)); echo "$n" > "{tmp_path}/shim-authority-n"
    f="{tmp_path}/shim-check-authority-$n.json"
    if [ -f "$f" ]; then cat "$f"; exit "$(cat "{tmp_path}/shim-check-authority-$n.rc" 2>/dev/null || echo 0)"; fi
    echo '{{"ok": true, "n_components": 0, "n_checked": 0, "n_certified": 0, "n_problems": 0, "problems": [], "components": {{}}}}' ;;
  *"stores verify-authority"*) slug=""; while [ $# -gt 0 ]; do [ "$1" = "--component" ] && slug="$2"; shift; done
    echo "{{\\"slug\\": \\"$slug\\", \\"state\\": \\"verified\\"}}"; exit ${{SHIM_VERIFY_AUTHORITY_RC:-0}} ;;
  *"adapters run"*) echo '{{"records_written": 0}}'; echo '{{"ingest_ok": 0, "refused": 0, "blocked": 0}}' ;;
  *" ingest"*) echo '{{"files": 0, "ok": 0}}' ;;
  *) : ;;
esac
exit 0
''')
    env.update({"ADAPTERS": adapters, "CLUSTER_HOST": "node-shim", "CLUSTER_SHARED": "/shared",
                "CLUSTER_DEST": SUB_A, "CLUSTER_VENV": "/shared/envs/venv", "WINDOW_WATCH_JOB": str(bin_ / "watch-job.sh"),
                "MIRROR_REPOS_SH": str(bin_ / "mirror-repos.sh"), "MIRROR_HISTORY_SH": str(bin_ / "mirror_history.sh"), "WINDOW_MIRROR_WAIT_S": "0"})
    if cluster_adapters is None:
        env.pop("CLUSTER_ADAPTERS", None)                       # the window's default: every named adapter has a cluster policy
    else:
        env["CLUSTER_ADAPTERS"] = cluster_adapters
    return env, P


def _cluster_stamps(log):
    """The stage stamps (the trap's heavy-refresh note excluded: it follows every window whose install ran, Phase 1 tests pin it)."""
    return [s for s in _stamps(log) if (s.startswith("-- ") or s.startswith("--- ") or s.startswith("===")) and "status heavy refresh" not in s]


CLEAN_TAIL = ["--- ingest", "--- ingest end", "=== ff chain end"]


def test_cluster_window_runs_every_stage_in_order_watches_the_array_pulls_merges_ingests_once_and_reads_nothing_on_the_grading_host(tmp_path):
    env, P = _cluster_rehearsal(tmp_path)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    stamps = _cluster_stamps(log)
    assert stamps == ["=== ff chain start ADAPTERS=heavy_one heavy_two", "-- reads reservation", "-- store mirror", "-- wal checkpoint", "-- config convert", "-- store authority", "-- install units", "-- uninstall units",
                      "-- migrate (plan)", "-- migrate (apply)", "-- dimensions load",
                      "-- cluster reads inbox-merge", "-- reader-state export", "-- tree export", "-- cluster reads plan", "-- cluster reads stage",
                      "-- cluster reads submit", "-- cluster reads revalidate", "-- cluster reads pull", "-- cluster reads receipt"] + CLEAN_TAIL, stamps
    calls = _calls(P)
    # the STANDING inbox is merged BEFORE the reader-state export (an earlier window's complete delivery left in the inbox is
    # in records/ when the shards' reader state is exported, so no shard reissues its id), and again after the pull: two inbox-merge calls
    merges = [i for i, c in enumerate(calls) if "inbox-merge" in c]
    export = next(i for i, c in enumerate(calls) if "reader-state export" in c)
    pull = next(i for i, c in enumerate(calls) if "cluster-reads pull" in c)
    assert len(merges) == 2 and merges[0] < export and pull < merges[1], (merges, export, pull)
    # and before the standing merge, the receive-directory recovery: an interrupted pull's complete copies installed first
    recover = [i for i, c in enumerate(calls) if "cluster-reads recover --window" in c]
    assert len(recover) == 1 and recover[0] < merges[0], (recover, merges)
    # the timers stopped BEFORE the store mirror started (a timer firing during a long mirror step used to start a pass the drain
    # then waited out), the stop was verified before the mirror, and the drain came after the mirror finished
    mirror_start = calls.index(next(c for c in calls if "start --no-block registry-store-mirror.service" in c))
    assert calls.index(next(c for c in calls if " stop registry-harvest.timer" in c)) < mirror_start
    assert calls.index(next(c for c in calls if c.startswith("systemctl --user is-active registry-harvest.timer"))) < mirror_start
    assert log.index("store mirror:") < log.index("writers quiet") and log.index("-- store mirror") < log.index("store mirror:")
    # the store-mirror TIMER is among the stopped timers -- stopped and verified inactive before the mirror step and the drain
    assert " registry-store-mirror.timer" in next(c for c in calls if " stop registry-harvest.timer" in c)
    assert calls.index(next(c for c in calls if c.startswith("systemctl --user is-active registry-store-mirror.timer"))) < mirror_start
    # the reader-state export names the cluster adapters and the base as the canonical checkout; the plan carries the tree sha and the destination
    assert any("reader-state export --adapter heavy_one heavy_two --out" in c and "--repo-root" in c for c in calls)
    assert any("cluster-reads plan --window" in c and "--adapter heavy_one heavy_two --tree-sha" in c and f"--dest {SUB_A}" in c for c in calls)
    # the plan reads the standing retirement holds off the sidecar the export above wrote (registry-hardening 3a, closing pass): the same file
    export_out = next(c for c in calls if "reader-state export --adapter heavy_one heavy_two --out" in c).split(" --out ", 1)[1].split()[0]
    assert any("cluster-reads plan --window" in c and f"--reader-state {export_out}" in c for c in calls), calls
    # the tree export went over ssh to the cluster host with the mirror and the trees path; mirror-repos ran first, then the history push
    # (mirror_history.sh with the base checkout and the mirror's URL: the archived branches and tags mirror-repos.sh leaves behind)
    assert any(c.startswith("ssh ") and "node-shim" in c and "/shared/repos/checkout /shared/gates/trees/" in c for c in calls)
    history = next(c for c in calls if c.startswith("mirror_history.sh "))
    assert history.endswith(" ssh://node-shim/shared/repos/checkout") and len(history.split()) == 3, history
    assert calls.index(next(c for c in calls if c.startswith("mirror-repos.sh"))) < calls.index(history) < calls.index(next(c for c in calls if c.startswith("ssh ")))
    assert "mirror_history.sh reported a problem" not in log
    # the watcher ran in the foreground on the array id with the plan's timers
    assert "watch-job.sh --slurm 4242 --host node-shim --poll 30 --stall-min 0 --max-hours 1" in calls
    # order: submit < watch < revalidate < pull < (post-pull) inbox-merge < refusals < ingest; exactly one ingest; NO adapter pass on the grading host
    order = [next(i for i, c in enumerate(calls) if key in c) for key in ("cluster-reads submit", "watch-job.sh", "cluster-reads revalidate", "cluster-reads pull")]
    order += [merges[-1], next(i for i, c in enumerate(calls) if "cluster-reads refusals" in c), next(i for i, c in enumerate(calls) if "cli ingest" in c)]
    assert order == sorted(order), order
    assert sum(1 for c in calls if c.endswith("cli ingest --retry-held")) == 1
    assert not any("adapters run" in c for c in calls), "no adapter is read on the grading host (C6, S2)"
    assert "REFUSED" not in log
    # C7: the reservation first (before the store mirror), submit and pull carry the operation, the receipt follows the post-pull merge
    assert calls.index(next(c for c in calls if "cluster-reads reserve --owner window --op W1 --takeover" in c)) < calls.index(next(c for c in calls if "start --no-block registry-store-mirror.service" in c))
    assert any("cluster-reads submit --window W1" in c and "--op W1 --state" in c for c in calls), [c for c in calls if "submit" in c]
    assert any("cluster-reads pull --window W1 --op W1 --state" in c for c in calls), [c for c in calls if "cluster-reads pull" in c]
    receipt = next(i for i, c in enumerate(calls) if "cluster-reads receipt --window W1 --op W1 --state" in c)
    assert merges[-1] < receipt < next(i for i, c in enumerate(calls) if "cluster-reads refusals" in c)
    assert any("reads-W1" not in c and "cluster-reads recover --window W1" in c for c in calls)


def test_a_refused_component_or_a_failed_shard_records_the_adapter_refused_never_read_and_the_window_ends_aborted(tmp_path):
    """The refusal list the plan + pull + revalidation produce (here: heavy_one refused at plan time) is stamped `-- adapter heavy_one REFUSED
    HH:MM:SS (plan: shim)` BEFORE the one ingest; the ingest runs; the window ends `ABORTED (refused: heavy_one)` with exit 20; no adapter is
    read on the grading host; the other cluster adapter is untouched."""
    env, P = _cluster_rehearsal(tmp_path, refusals="heavy_one")
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 20, r.stderr + log
    stamps = _cluster_stamps(log)
    assert stamps[stamps.index("-- cluster reads pull") + 1:] == ["-- cluster reads receipt", "-- adapter heavy_one REFUSED", "--- ingest", "--- ingest end", "=== ff chain ABORTED (refused: heavy_one)"], stamps
    assert re.search(r"^-- adapter heavy_one REFUSED \d\d:\d\d:\d\d \(plan: shim\)$", log, re.M)
    calls = _calls(P)
    assert not any("adapters run" in c for c in calls) and sum(1 for c in calls if c.endswith("cli ingest --retry-held")) == 1
    assert "refused: heavy_one; abort" in log


def test_a_stale_post_run_revalidation_names_the_adapter_and_it_is_recorded_refused(tmp_path):
    env, P = _cluster_rehearsal(tmp_path, refusals="heavy_two", revalidate_out="revalidation REFUSED: heavy_two (recorded refused; the pull withholds every revision these adapters published this window)")
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 20, r.stderr + log
    assert "revalidation REFUSED: heavy_two" in log and "-- adapter heavy_two REFUSED" in _cluster_stamps(log) and "=== ff chain ABORTED (refused: heavy_two)" in log
    calls = _calls(P)
    assert calls.index(next(c for c in calls if "cluster-reads revalidate" in c)) < calls.index(next(c for c in calls if "cluster-reads pull" in c))


def test_a_failing_history_push_is_a_note_not_a_refusal_the_window_exports_and_reads_as_before(tmp_path):
    """The reads need the tree's code only, and a gate that reuses the tree completes it against the mirror: a mirror_history.sh that fails
    (exit 2) leaves one note in the log and changes nothing else -- the export runs, every stage follows, the window ends clean."""
    env, P = _cluster_rehearsal(tmp_path, history_rc=2)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    assert "mirror_history.sh reported a problem (rc 2): HISTORY current 0 refs on ssh://node-shim/shared/repos/checkout (shim); the exported tree may lack archived branches or tags (the reads need neither)" in log, log
    assert "REFUSED" not in log and "ABORTED" not in log
    stamps = _cluster_stamps(log)
    assert "-- tree export" in stamps and "-- cluster reads submit" in stamps and stamps[-3:] == CLEAN_TAIL, stamps
    calls = _calls(P)
    assert calls.index(next(c for c in calls if c.startswith("mirror_history.sh "))) < calls.index(next(c for c in calls if c.startswith("ssh ")))


def test_a_failing_cluster_stage_records_every_cluster_adapter_refused_and_the_window_ingests_then_aborts(tmp_path):
    for n, knob in enumerate(({"reader_state_rc": 1}, {"tree_rc": 2, "tree_out": "ERR shallow fetch"}, {"plan_rc": 2}, {"stage_rc": 2},
                              {"submit": "cluster-reads submit: sbatch failed"}, {"pull_rc": 2}, {"revalidate_rc": 2})):
        d = tmp_path / f"k{n}"
        d.mkdir()
        env, P = _cluster_rehearsal(d, **knob)
        r = _run(env)
        log = P["log"].read_text()
        assert r.returncode == 20, (knob, r.stderr + log)
        stamps = _cluster_stamps(log)
        assert "-- adapter heavy_one REFUSED" in stamps and "-- adapter heavy_two REFUSED" in stamps, (knob, stamps)
        assert stamps[-3:] == ["--- ingest", "--- ingest end", "=== ff chain ABORTED (refused: heavy_one heavy_two)"], (knob, stamps)
        assert "every cluster adapter is RECORDED REFUSED this window (the grading host reads no store)" in log, knob
        assert re.search(r"^-- adapter heavy_one REFUSED \d\d:\d\d:\d\d \(cluster reads: ", log, re.M), knob
        calls = _calls(P)
        assert sum(1 for c in calls if c.endswith("cli ingest --retry-held")) == 1 and not any("adapters run" in c for c in calls)
        if not ("submit" in knob or "pull_rc" in knob or "revalidate_rc" in knob):
            assert not any("watch-job.sh" in c for c in calls), knob     # nothing submitted: nothing watched
        if "revalidate_rc" in knob:
            # a revalidation that did not RUN leaves no verdict: nothing is pulled
            assert "revalidate FAILED rc=2" in log and not any("cluster-reads pull" in c for c in calls), knob


def test_a_failed_submission_refuses_every_cluster_adapter_even_when_one_was_already_refused_at_admission(tmp_path):
    """heavy_one is on the refusal list from the plan (refused at admission) and the sbatch of heavy_two's array FAILS: before the
    fix the non-numeric result was read as "no cluster task" because the list was non-empty, and heavy_two ran nowhere unnoticed. Now a
    failed submission refuses every cluster adapter, by name, and the window says so."""
    env, P = _cluster_rehearsal(tmp_path, refusals="heavy_one", submit="cluster-reads submit: sbatch failed")
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 20, r.stderr + log
    stamps = _cluster_stamps(log)
    assert "-- adapter heavy_one REFUSED" in stamps and "-- adapter heavy_two REFUSED" in stamps, stamps
    assert "submit FAILED" in log and "no cluster task" not in log and stamps[-3:] == ["--- ingest", "--- ingest end", "=== ff chain ABORTED (refused: heavy_one heavy_two)"]
    calls = _calls(P)
    assert not any("watch-job.sh" in c or "cluster-reads pull" in c for c in calls)


def test_a_plan_with_no_cluster_task_is_not_a_failed_submission_and_its_refusals_are_the_plans(tmp_path):
    """`cluster-reads submit` prints `none` (rc 0) when every adapter was recorded refused at plan time: nothing is watched or pulled,
    nothing is called a failure, and the plan's refusals are what the window stamps."""
    env, P = _cluster_rehearsal(tmp_path, refusals="heavy_one heavy_two", submit="none")
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 20, r.stderr + log
    stamps = _cluster_stamps(log)
    assert "cluster reads submit: no cluster task" in log and "submit FAILED" not in log and "every cluster adapter is RECORDED REFUSED" not in log
    assert "-- adapter heavy_one REFUSED" in stamps and "-- adapter heavy_two REFUSED" in stamps
    calls = _calls(P)
    assert not any("watch-job.sh" in c or "cluster-reads revalidate" in c or "cluster-reads pull" in c for c in calls)


def test_cluster_local_venv_knob_reaches_the_submission_as_an_opt_in_flag(tmp_path):
    """`CLUSTER_LOCAL_VENV=1` adds `--local-venv` to `cluster-reads submit` (the measured ~563 s per task path); unset, the flag is absent."""
    for knob, want in (({"CLUSTER_LOCAL_VENV": "1"}, True), ({}, False)):
        d = tmp_path / ("on" if want else "off")
        d.mkdir()
        env, P = _cluster_rehearsal(d)
        env.update(knob)
        r = _run(env)
        assert r.returncode == 0, r.stderr + P["log"].read_text()
        submit = next(c for c in _calls(P) if "cluster-reads submit" in c)
        assert ("--local-venv" in submit) is want, submit


def test_the_read_policy_reaches_the_plan_and_the_reread_force_adapter_is_the_plans_refusal_not_a_grading_host_pass(tmp_path):
    """$FULL_ADAPTERS / $REREAD_FORCE_ADAPTERS reach `cluster-reads plan` (the per-adapter policy); the plan records the
    reread-force adapter refused (REREAD_FORCE_REASON) and the window stamps it from the refusal list -- no grading-host pass runs for it
    unless the operator override is set in the window's environment, in which case the override run is stamped and the adapter is not
    refused."""
    env, P = _cluster_rehearsal(tmp_path, refusals="heavy_two")
    env.update({"FULL_ADAPTERS": "heavy_one", "REREAD_FORCE_ADAPTERS": "heavy_two"})
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 20, r.stderr + log
    calls = _calls(P)
    assert any("cluster-reads plan" in c and "--full-adapter heavy_one" in c and "--reread-force-adapter heavy_two" in c for c in calls), calls
    assert not any("adapters run" in c for c in calls)
    assert "-- adapter heavy_two REFUSED" in _cluster_stamps(log) and "=== ff chain ABORTED (refused: heavy_two)" in log


def test_cluster_adapters_none_refuses_every_named_adapter_after_the_skipped_line(tmp_path):
    env, P = _cluster_rehearsal(tmp_path, cluster_adapters="none", adapters="alpha beta")
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 20, r.stderr + log
    assert _cluster_stamps(log) == ["=== ff chain start ADAPTERS=alpha beta", "-- reads reservation", "-- wal checkpoint", "-- config convert", "-- store authority", "-- install units", "-- uninstall units", "-- migrate (plan)", "-- migrate (apply)",
                                    "-- dimensions load",
                                    "-- cluster reads SKIPPED (no named adapter has a cluster policy)", "-- adapter alpha REFUSED", "-- adapter beta REFUSED",
                                    "--- ingest", "--- ingest end", "=== ff chain ABORTED (refused: alpha beta)"]
    calls = _calls(P)
    # no cluster-reads STAGE (the registry python's reserve / release are the protocol's, and `is-active registry-cluster-reads.timer` is the timer's name)
    assert not any((c.startswith("python") and "cluster-reads" in c and "reserve" not in c and "release" not in c) or "reader-state" in c
                   or "start --no-block registry-store-mirror" in c or c.startswith("ssh") or "adapters run" in c for c in calls)


def test_a_window_naming_no_adapter_with_a_cluster_policy_skips_the_whole_cluster_block_in_one_line_and_refuses_them(tmp_path):
    """ADAPTERS names none of the adapters with a cluster policy -- the whole cluster block (the store mirror
    included) is absent, ONE stamped `cluster reads SKIPPED` line stands in its place, and every named adapter is recorded refused (no
    grading-host pass). windows.py classes the line `cluster_skip` and the refusals `adapter_refused`."""
    env, P = _cluster_rehearsal(tmp_path, adapters="alpha beta", cluster_adapters="heavy_one heavy_two")
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 20, r.stderr + log
    assert _cluster_stamps(log) == ["=== ff chain start ADAPTERS=alpha beta", "-- reads reservation", "-- wal checkpoint", "-- config convert", "-- store authority", "-- install units", "-- uninstall units", "-- migrate (plan)", "-- migrate (apply)",
                                    "-- dimensions load",
                                    "-- cluster reads SKIPPED (no named adapter has a cluster policy)", "-- adapter alpha REFUSED", "-- adapter beta REFUSED",
                                    "--- ingest", "--- ingest end", "=== ff chain ABORTED (refused: alpha beta)"], _cluster_stamps(log)
    calls = _calls(P)
    assert not any((c.startswith("python") and "cluster-reads" in c and "reserve" not in c and "release" not in c) or "reader-state" in c
                   or "start --no-block registry-store-mirror" in c or c.startswith("ssh") or c.startswith("mirror-repos.sh") or c.startswith("mirror_history.sh")
                   or c.startswith("watch-job.sh") or "adapters run" in c for c in calls), calls
    assert "-- store mirror" not in log
    assert sum(1 for c in calls if c.endswith("cli ingest --retry-held")) == 1
    windows = _load("windows").read_logs([str(P["log"])])
    assert len(windows) == 1 and windows[0]["completed"] is False and windows[0]["abort_cause"] == "refused: alpha beta"
    assert [st["class"] for st in windows[0]["stages"]] == ["reservation", "wal", "config_convert", "authority", "base", "install", "install", "migrate_plan", "migrate", "dims", "cluster_skip", "adapter_refused", "adapter_refused", "ingest", "ingest_end", "status_heavy"]


def test_a_window_naming_one_cluster_adapter_plans_exports_and_refuses_that_adapter_only(tmp_path):
    """ADAPTERS names one of the two adapters with a cluster policy -- the reader-state export and the plan cover THAT adapter
    alone (never the unnamed one), a failing cluster stage refuses that adapter alone, and the unnamed cluster adapter is read nowhere this
    window (a window reads the adapters it names, once)."""
    env, P = _cluster_rehearsal(tmp_path, adapters="heavy_two", cluster_adapters="heavy_one heavy_two")
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    calls = _calls(P)
    assert any("reader-state export --adapter heavy_two --out" in c for c in calls), calls
    assert any("cluster-reads plan --window" in c and "--adapter heavy_two --tree-sha" in c for c in calls), calls
    assert not any("heavy_one" in c for c in calls), [c for c in calls if "heavy_one" in c]
    stamps = _cluster_stamps(log)
    assert stamps[:3] == ["=== ff chain start ADAPTERS=heavy_two", "-- reads reservation", "-- store mirror"] and "-- cluster reads plan" in stamps
    assert stamps[-5:] == ["-- cluster reads pull", "-- cluster reads receipt"] + CLEAN_TAIL, stamps
    # a failing cluster stage refuses the NAMED cluster adapter only
    d = tmp_path / "fail"
    d.mkdir()
    env2, P2 = _cluster_rehearsal(d, adapters="heavy_two", cluster_adapters="heavy_one heavy_two", plan_rc=2)
    r2 = _run(env2)
    log2 = P2["log"].read_text()
    assert r2.returncode == 20, r2.stderr + log2
    assert "-- adapter heavy_two REFUSED" in _cluster_stamps(log2) and "-- adapter heavy_one REFUSED" not in _cluster_stamps(log2)
    assert "=== ff chain ABORTED (refused: heavy_two)" in log2


# --------------------------------------------------------------------------------------------
# the reads reservation and the retired units: the window's first act, the trap's release, the
# refused reservation, the ownership-lost pull, the uninstall stage
# --------------------------------------------------------------------------------------------


def test_a_reservation_held_by_a_live_window_aborts_this_one_before_the_flag_the_stop_or_the_base_move(tmp_path):
    """`cluster-reads reserve` exits 24 (a live window holds the reservation): the window exits 22 at once -- no maintenance flag, no
    `systemctl stop`, no git, no install -- the trap restores nothing it did not stop, releases NOTHING (it never held the reservation) and
    writes the ABORTED marker naming the cause."""
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS, extra_env={"SHIM_RESERVE_RC": "24"})
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 22, r.stderr + log
    assert _stamps(log) == ["=== ff chain start ADAPTERS=", "-- reads reservation", "=== ff chain ABORTED (reads reservation not taken (rc=24): reserve: taken for window W1 (shim))"], _stamps(log)
    calls = _calls(P)
    assert not any(" stop " in c or c.startswith("git") or "cluster-reads release" in c or "install" in c for c in calls), calls
    assert not P["maint"].exists() and "maintenance flag written" not in log and "reads reservation released" not in log
    # the timers were never stopped, so the restore set is empty and nothing is started
    assert _starts(calls) == [], _starts(calls)
    assert "timers never stopped (aborted before the stop): nothing to restore" in log and "timers restored" not in log
    _assert_no_retired_start(calls)
    # windows.py still reads the window as rev3 (the trap's line stands for the restore) and ABORTED with the cause
    windows = _load("windows").read_logs([str(P["log"])])
    assert len(windows) == 1 and windows[0]["completed"] is False and windows[0]["end_definition"] == "rev3" and windows[0]["abort_cause"].startswith("reads reservation not taken")


def _base_without_reserve(P):
    """The cutover's BASE: a registry python whose `cluster-reads` knows no `reserve` / `release` -- the pre-merge main the base
    checkout runs until the window's own fast-forward -- refusing them the way argparse does (exit 2, `invalid choice`, the usage on
    stderr); every other call is answered by the rehearsal's shim. Recorded as `python-base …` in the calls log, so a call that reached
    the base's code is told apart from one that ran under the worktree's (`python …`, with its PYTHONPATH in P["pypaths"])."""
    bin_ = P["tmp"] / "bin"
    return _shim(bin_, "regpy-base", f'''echo "python-base $*" >> "{P['calls']}"
case "$*" in
  *"cluster-reads reserve"*|*"cluster-reads release"*)
    a=$(sed -n 's/.*cluster-reads \\([a-z]*\\).*/\\1/p' <<<"$*")
    echo "usage: registry cluster-reads [-h] {{plan,stage,submit,shard,revalidate,pull,fallback,stamp,status}} ..." >&2
    echo "registry cluster-reads: error: argument action: invalid choice: '$a' (choose from 'plan', 'stage', 'submit', 'shard', 'revalidate', 'pull', 'fallback', 'stamp', 'status')" >&2
    exit 2 ;;
esac
exec "{bin_}/regpy" "$@"
''')


def test_the_reservation_and_its_release_run_under_the_worktrees_cli_when_the_base_has_no_reserve_yet(tmp_path):
    """The cutover window: the base checkout still runs the pre-merge main until `base at`, and that main's `cluster-reads` has no
    `reserve` -- a first attempt ran the reservation through the base's CLI and aborted at its first act ("invalid choice: 'reserve'",
    exit 22). The reservation and its release are registry calls that run BEFORE the base moves, so they run under the WORKTREE's CLI
    (WINDOW_NEW_PY under WINDOW_NEW_PYPATH, like `config --check` and the store authority stage) and a base without them does not abort
    the window; `config --json` (the old loader's validation, the point of the call) and the stages after `base at` still run the base's."""
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS)
    base = _base_without_reserve(P)
    # the fixture fires: the base's CLI refuses `reserve` the way the pre-merge main did
    probe = subprocess.run([str(base), "-m", "sisteer.registry.cli", "cluster-reads", "reserve", "--owner", "window"], capture_output=True, text=True,
                           env={"PATH": env["PATH"], "WINDOW_ID": "W1"})
    assert probe.returncode == 2 and "invalid choice: 'reserve'" in probe.stderr and probe.stdout == ""
    P["calls"].write_text("")
    env["REGISTRY_PY"] = str(base)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    st = _stamps(log)
    assert st[:2] == ["=== ff chain start ADAPTERS=", "-- reads reservation"] and "base at deadbee" in st and st[-1] == "=== ff chain end", st
    assert "invalid choice" not in log and "reads reservation not taken" not in log
    calls = _calls(P)
    reserve = [c for c in calls if "cluster-reads reserve --owner window --op W1 --takeover --state" in c]
    release = [c for c in calls if "cluster-reads release --op W1 --state" in c]
    assert len(reserve) == 1 and len(release) == 1, (reserve, release)
    assert reserve[0].startswith("python -m sisteer.registry.cli cluster-reads reserve") and release[0].startswith("python -m sisteer.registry.cli cluster-reads release"), (reserve, release)
    assert not any(c.startswith("python-base") and "cluster-reads" in c for c in calls), "no cluster-reads call reached the base's code before `base at`"
    # the interpreter and the PYTHONPATH the shim saw: the worktree's for the reservation and the release, the base's for `config --json`
    pyp = P["pypaths"].read_text().splitlines()
    wt_src, base_src = f"{P['home']}/worktree/src", f"{P['base']}/src"
    assert [l.split(" ", 1)[0] for l in pyp if "cluster-reads reserve" in l or "cluster-reads release" in l] == [wt_src, wt_src], pyp
    assert [l.split(" ", 1)[0] for l in pyp if l.endswith(" config --json")] == [base_src], pyp
    assert any(c.startswith("python-base") and c.endswith(" config --json") for c in calls), "the old loader's validation is the base's call, as before"
    assert any(c.startswith("python-base") and "migrate --plan" in c for c in calls), "after `base at` the base IS the landed tree: its CLI runs the stages"
    assert "reads reservation released" in log and log.index("reads reservation released") < log.index("timers restored")


def test_an_abort_before_the_base_moves_releases_the_reservation_under_the_worktrees_cli_too(tmp_path):
    """The release in the trap runs before the base has moved whenever the window aborts early (here `config --check` fails, exit 24): the
    base's CLI has no `release` at the cutover, so the release runs under the worktree's -- else the reservation would stay on disk
    behind a line that says it was released."""
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS, fail_on='*"config --check"*')
    env["REGISTRY_PY"] = str(_base_without_reserve(P))
    (P["home"] / "registry" / "config.toml").write_text("[registry]\nname = 'x'\n")
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 24, r.stderr + log
    assert "base at" not in log and "invalid choice" not in log
    calls = _calls(P)
    release = [c for c in calls if "cluster-reads release --op W1 --state" in c]
    assert release == ["python -m sisteer.registry.cli cluster-reads release --op W1 --state " + str(P["state"])], release
    assert not any(c.startswith("python-base") and "cluster-reads" in c for c in calls)
    assert [l.split(" ", 1)[0] for l in P["pypaths"].read_text().splitlines() if "cluster-reads release" in l] == [f"{P['home']}/worktree/src"]
    assert "release: released (shim)" in log and "reads reservation released" in log


def test_the_reservation_is_released_by_the_trap_on_every_abort_path_before_the_timers_restore(tmp_path):
    """A migrate failure (exit 8) and an ingest failure (exit 12): the trap releases the reservation by name (`cluster-reads release --op W1`)
    after the flag is removed and before the first `systemctl start`, so the next tick finds nothing held."""
    for n, fail_on in enumerate(('*"cli migrate"', '*"cli ingest --retry-held"')):
        d = tmp_path / f"k{n}"
        d.mkdir()
        env, P = _rehearsal(d, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS, fail_on=fail_on)
        r = _run(env)
        log = P["log"].read_text()
        assert r.returncode in (8, 12), (fail_on, r.returncode, log)
        calls = _calls(P)
        rel = next(c for c in calls if "cluster-reads release --op W1 --state" in c)
        assert calls.index(rel) < calls.index(_starts(calls)[0]), (fail_on, calls)
        lines = log.splitlines()
        assert lines.index(next(l for l in lines if l.startswith("maintenance flag removed"))) < lines.index(next(l for l in lines if l.startswith("reads reservation released"))) \
            < lines.index(next(l for l in lines if l.startswith("timers restored")))


def test_an_ownership_lost_pull_refuses_every_cluster_adapter_and_the_window_ends_aborted(tmp_path):
    """`cluster-reads pull --op W1` exits 23 (the reservation stopped naming the window): the pull stage fails, every cluster adapter is
    RECORDED REFUSED, nothing is merged after it, and the window ends ABORTED (refused: ...)."""
    env, P = _cluster_rehearsal(tmp_path, pull_rc=23)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 20, r.stderr + log
    assert "pull FAILED rc=23" in log and "=== ff chain ABORTED (refused: heavy_one heavy_two)" in log
    calls = _calls(P)
    assert any("cluster-reads pull --window W1 --op W1 --state" in c for c in calls)
    assert "-- adapter heavy_one REFUSED" in _cluster_stamps(log) and "-- adapter heavy_two REFUSED" in _cluster_stamps(log)


def test_a_failed_receipt_leaves_the_read_standing(tmp_path):
    """The receipt is the proof for the tick's changed set and S7, not a stage of the read: a non-zero `cluster-reads receipt` is logged and
    the window goes on (the next tick reads these adapters again, having no whole receipt for them)."""
    env, P = _cluster_rehearsal(tmp_path, receipt_rc=2)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    assert "receipt FAILED rc=2 (the read stands" in log and "=== ff chain end" in log


def test_the_uninstall_stage_removes_the_retired_unit_files_the_installer_never_copies_them_back_and_a_failure_is_exit_19(tmp_path):
    """Every retired unit file present before the window (the live grading host's state: both adapters units and both Modal pull units), plus the
    Modal-era drop-ins (`registry-modal-pull.service.d/no-restart.conf`, `registry-mirrors.service.d/clusters-only.conf` beside a
    `no-restart.conf` and a `.bak` that STAY): disabled, removed, the daemon reloaded, the stamp names them; a unit directory the files
    cannot be removed from fails the window with exit 23 after the install stage."""
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS + ["registry-adapters.service", "registry-modal-pull.service"])
    ud = P["unit_dir"]
    (ud / "registry-modal-pull.service.d").mkdir()
    (ud / "registry-modal-pull.service.d" / "no-restart.conf").write_text("[Service]\nRestart=no\n")
    (ud / "registry-mirrors.service.d").mkdir()
    (ud / "registry-mirrors.service.d" / "clusters-only.conf").write_text("[Service]\nExecStart=\nExecStart=x mirrors sync --dest cluster_a --dest cluster_b\n")
    (ud / "registry-mirrors.service.d" / "clusters-only.conf.bak-20260928").write_text("old\n")
    (ud / "registry-mirrors.service.d" / "no-restart.conf").write_text("[Service]\nRestart=no\n")
    (ud / "registry-status.service.d").mkdir()
    (ud / "registry-status.service.d" / "no-restart.conf").write_text("[Service]\nRestart=no\n")
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    for u in RETIRED_UNITS:
        assert not (ud / u).exists(), u
    assert not (ud / "registry-modal-pull.service.d").exists() and not (ud / "registry-mirrors.service.d" / "clusters-only.conf").exists()
    assert (ud / "registry-mirrors.service.d" / "no-restart.conf").exists() and (ud / "registry-mirrors.service.d" / "clusters-only.conf.bak-20260928").exists()
    assert (ud / "registry-status.service.d" / "no-restart.conf").exists(), "only the named drop-ins go"
    assert ("retired units uninstalled (disabled, files removed: registry-adapters.timer registry-adapters.service registry-modal-pull.timer "
            "registry-modal-pull.service registry-modal-pull.service.d/ registry-mirrors.service.d/clusters-only.conf; daemon reloaded)") in log, log
    assert re.search(r"^-- uninstall units \d\d:\d\d:\d\d \(registry-adapters.timer registry-adapters.service registry-modal-pull.timer "
                     r"registry-modal-pull.service; drop-ins registry-modal-pull.service.d registry-mirrors.service.d/clusters-only.conf\)$", log, re.M)
    calls = _calls(P)
    i_inst = calls.index("systemctl --user daemon-reload")
    assert calls.index("systemctl --user disable registry-adapters.timer") > i_inst       # after the install stage's reload
    assert all(f"systemctl --user disable {u}" in calls for u in RETIRED_UNITS)
    _assert_no_retired_start(calls)
    # the repo carries no source file for a retired unit (the installer cannot copy it back), and the installer re-created neither drop-in
    for u in RETIRED_UNITS:
        assert not os.path.exists(os.path.join(UNITS, u)), u
    # (the unit sources are plain files or templates the installer renders, `<unit>.in`)
    assert sorted(f for f in os.listdir(ud) if f.endswith(".service")) == sorted(
        f[:-3] if f.endswith(".in") else f for f in os.listdir(UNITS) if f.endswith(".service") or f.endswith(".service.in"))
    # a failure: the unit file cannot be removed (a directory in its place) -> exit 23 after the install, the timers restored
    d = tmp_path / "fail"
    d.mkdir()
    env2, P2 = _rehearsal(d, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS)
    (P2["unit_dir"] / "registry-adapters.service").mkdir()
    (P2["unit_dir"] / "registry-adapters.service" / "keep").write_text("x")
    r2 = _run(env2)
    log2 = P2["log"].read_text()
    assert r2.returncode == 23, r2.stderr + log2
    assert ("uninstall units FAILED (registry-adapters.timer registry-adapters.service registry-modal-pull.timer registry-modal-pull.service; "
            "drop-ins registry-modal-pull.service.d registry-mirrors.service.d/clusters-only.conf); abort") in log2 and "-- migrate (apply)" not in log2
    assert "timers restored (trap; the set above)" in log2 and "=== ff chain ABORTED (uninstall units FAILED" in log2


# --------------------------------------------------------------------------------------------
# the dimensions-load key: keyed on the SEED FILES the load reads, never every file under configs/registry
# --------------------------------------------------------------------------------------------


def test_the_dims_key_names_exactly_the_seed_files_the_load_reads():
    """window.sh's DIMS_SEED_FILES is the set `registry dimensions load` reads (adapters/dimensions.py SEED_FILES), spelled sorted."""
    from sisteer.registry.adapters import dimensions as DIM
    m = re.search(r'^DIMS_SEED_FILES="([^"]+)"$', open(WINDOW).read(), re.M)
    assert m, "window.sh declares DIMS_SEED_FILES"
    assert m.group(1).split() == sorted(DIM.SEED_FILES.values())


def _dims_stamps(log_text):
    """Per window of the log, the dimensions stamp: 'load' or 'skipped'."""
    out = []
    for chunk in log_text.split("=== ff chain start")[1:]:
        if "-- dimensions load SKIPPED" in chunk:
            out.append("skipped")
        elif "-- dimensions load" in chunk:
            out.append("load")
        else:
            out.append("none")
    return out


def test_a_cluster_reads_yaml_change_never_forces_a_dims_cycle_but_a_seed_change_does(tmp_path):
    """Four windows over one state: the first loads (no stamp yet); the second skips; the third skips although cluster_reads.yaml -- and a
    roster file beside it -- changed in between (neither is an input of the load); the fourth loads because models.yaml changed.
    Four windows over ONE state: each mints its own id (``window_id=None``) -- the plan file is written once per id."""
    env, P = _rehearsal(tmp_path, active=[], pre_installed=OLD_TIMERS + NEW_TIMERS, window_id=None)
    seeds = P["base"] / "configs" / "registry"
    (seeds / "models.yaml").write_text("models: [a]\n")
    (seeds / "cluster_reads.yaml").write_text("schema: cluster_reads/2\n")
    (seeds / "roster_r2_v99.yaml").write_text("roster: []\n")
    assert _run(env).returncode == 0
    assert _run(env).returncode == 0
    (seeds / "cluster_reads.yaml").write_text("schema: cluster_reads/2\nadapters: {x: {shards: 1}}\n")
    (seeds / "roster_r2_v99.yaml").write_text("roster: [b]\n")
    assert _run(env).returncode == 0
    (seeds / "models.yaml").write_text("models: [a, b]\n")
    assert _run(env).returncode == 0
    log = P["log"].read_text()
    assert _dims_stamps(log) == ["load", "skipped", "skipped", "load"], log
    # the stamp carries the key the skip compared: seeds + schema + the database inode
    key = (P["home"] / "registry" / ".dimensions-last-load").read_text().strip()
    assert re.fullmatch(r"seeds=[0-9a-f]{16} schema=17 db=\d+", key), key
    # the load ran exactly twice (the shim records every registry call)
    calls = _calls(P)
    assert sum(1 for c in calls if "dimensions load --seeds" in c) == 2


# --------------------------------------------------------------------------------------------
# the migration plan stage, one window at a time, the pins and the reconciliation slot, under the slurm-only stamps: the reservation and
# uninstall stamps, the trap's ABORTED marker, no grading-host adapter pass, the NEVER_START set
# --------------------------------------------------------------------------------------------

PENDING_PLAN = ["schema version 98; code 99; pending: 0099_synthetic_slug", "shape differs from the code's schema:",
                "  table synthetic_fact: missing", "add column(s) to table object: parent_object_id, named_role (27 rows, none copied; small-scale)",
                "create table synthetic_fact"]
REBUILD_PLAN = PENDING_PLAN + ["rebuild table row_history: 123456 rows to copy (item-scale)"]


def _plan_file(P):
    files = sorted((P["state"] / "windows").glob("migrate-plan-*.txt"))
    assert len(files) == 1, files
    return files[0]


def test_the_plan_is_read_whole_and_a_rebuild_line_stops_the_window_before_the_apply(tmp_path):
    """The STOP the docs state is executable: a `rebuild table` line in `registry migrate --plan` ends the window at the plan stage --
    exit 15, no apply, no dimensions load, no cluster stage, no ingest, the ABORTED marker naming the cause, the timers restored -- with
    the WHOLE plan in the log and in the window's plan file (the stage used to tail six lines and apply at once)."""
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS, plan_lines=REBUILD_PLAN)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 15, r.stderr + log
    stamps = _stamps(log)
    assert stamps[:12] == ["=== ff chain start ADAPTERS=", "-- reads reservation", "writers quiet", "-- wal checkpoint", "-- config convert", "-- store authority", "base at deadbee", "-- install units", "-- uninstall units",
                           "-- migrate (plan)", "timers restored (trap; the set above)", "-- status heavy refresh started once (cache initializer)"], stamps
    assert len(stamps) == 13 and stamps[12].startswith("=== ff chain ABORTED (migration plan has a rebuild line (row_history): a STOP, nothing applied"), stamps
    assert "migration plan has a rebuild line (row_history): a STOP, nothing applied" in log and "WINDOW_ALLOW_REBUILD=1" in log
    assert "window ended early (rc=15)" in log and "=== ff chain end" not in log
    for line in REBUILD_PLAN:
        assert line in log, line                                # the whole plan, not its tail
    assert _plan_file(P).read_text().splitlines() == REBUILD_PLAN
    calls = _calls(P)
    assert any(c.endswith("cli migrate --plan") for c in calls)
    assert not any(c.endswith("cli migrate") or "dimensions load" in c or "adapters run" in c or "cluster-reads plan" in c or c.endswith("cli ingest --retry-held") for c in calls), calls
    assert "systemctl --user start registry-harvest.timer" in calls
    # the reservation was taken first and released by the trap all the same
    assert any("cluster-reads reserve --owner window --op W1" in c for c in calls) and any("cluster-reads release --op W1" in c for c in calls)


def test_window_allow_rebuild_applies_a_reviewed_rebuild_and_says_so(tmp_path):
    env, P = _rehearsal(tmp_path, active=[], pre_installed=OLD_TIMERS + NEW_TIMERS, plan_lines=REBUILD_PLAN, extra_env={"WINDOW_ALLOW_REBUILD": "1"})
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    assert "plan has rebuild line(s) for: row_history -- applied because WINDOW_ALLOW_REBUILD=1 names a reviewed rebuild" in log
    assert "plan: pending 0099_synthetic_slug; shape differs; applying without a pause (WINDOW_PLAN_APPROVAL unset)" in log
    calls = _calls(P)
    assert calls.index(next(c for c in calls if c.endswith("cli migrate --plan"))) < calls.index(next(c for c in calls if c.endswith("cli migrate")))
    assert "=== ff chain end" in log


def test_expect_migration_pins_the_pending_list_to_exactly_one_migration(tmp_path):
    """WINDOW_EXPECT_MIGRATION: the pending list must be exactly that migration (the check migrate_window.sh makes with --expect) -- a
    different list exits 15 with nothing applied; the exact one applies."""
    env, P = _rehearsal(tmp_path, active=[], pre_installed=OLD_TIMERS, plan_lines=PENDING_PLAN, extra_env={"WINDOW_EXPECT_MIGRATION": "0098_another_slug"})
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 15, r.stderr + log
    assert "migration plan pending '0099_synthetic_slug', expected exactly '0098_another_slug' (WINDOW_EXPECT_MIGRATION); nothing applied" in log
    assert not any(c.endswith("cli migrate") for c in _calls(P))
    env2, P2 = _rehearsal(tmp_path / "ok", active=[], pre_installed=OLD_TIMERS, plan_lines=PENDING_PLAN, extra_env={"WINDOW_EXPECT_MIGRATION": "0099_synthetic_slug"})
    r2 = _run(env2)
    assert r2.returncode == 0, r2.stderr + P2["log"].read_text()
    assert any(c.endswith("cli migrate") for c in _calls(P2))


def _token(plan_lines):
    """The plan token the window prints: the sha256 of the plan file, which is the shim's stdout -- one line per echo."""
    return hashlib.sha256(("\n".join(plan_lines) + "\n").encode()).hexdigest()


def test_plan_approval_refuses_a_file_from_before_or_during_the_plan_and_times_out_with_nothing_applied(tmp_path):
    """WINDOW_PLAN_APPROVAL: a plan with something to apply pauses until an approval BOUND to the displayed plan arrives. Three files
    that must NOT count, each ending in exit 16 with nothing applied and the timers restored: one already there when the stage begins
    (discarded before the plan runs), one created WHILE the plan is rendered (the sequence the first review fix accepted; discarded at
    the boundary, after the complete plan is displayed), and one created while the plan is rendered that already carries the RIGHT
    token (the boundary is the event, not the content: nobody had seen the plan whole)."""
    approval = tmp_path / "approve-plan"
    approval.write_text("stale\n")
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS, plan_lines=PENDING_PLAN,
                        extra_env={"WINDOW_PLAN_APPROVAL": str(approval), "WINDOW_PLAN_WAIT_S": "3"})
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 16, r.stderr + log
    assert f"a pre-existing approval at {approval} was discarded: the approval must follow the plan" in log
    assert f"plan token {_token(PENDING_PLAN)} (the sha256 of that file)" in log
    assert f"plan: pending 0099_synthetic_slug; shape differs; waiting up to 3s for the operator's approval of THIS plan -- after reading it, approve with: echo {_token(PENDING_PLAN)} > {approval}" in log
    assert "migration plan not approved within 3s" in log and "nothing applied" in log and "window ended early (rc=16)" in log
    assert not approval.exists()
    calls = _calls(P)
    assert not any(c.endswith("cli migrate") for c in calls) and "systemctl --user start registry-harvest.timer" in calls
    # created while the plan is rendered, empty: discarded at the boundary -- the early-approval sequence no longer passes
    approval2 = tmp_path / "approve-plan-2"
    env2, P2 = _rehearsal(tmp_path / "during", active=[], pre_installed=OLD_TIMERS, plan_lines=PENDING_PLAN, shim_touch=str(approval2),
                          extra_env={"WINDOW_PLAN_APPROVAL": str(approval2), "WINDOW_PLAN_WAIT_S": "3"})
    r2 = _run(env2)
    log2 = P2["log"].read_text()
    assert r2.returncode == 16, r2.stderr + log2
    assert f"an approval at {approval2} written while the plan was being generated was discarded: nobody had seen the plan whole" in log2
    assert "plan approved" not in log2 and "=== ff chain end" not in log2 and not approval2.exists()
    assert not any(c.endswith("cli migrate") for c in _calls(P2))
    # created while the plan is rendered WITH the right token: the boundary discards it all the same
    approval3 = tmp_path / "approve-plan-3"
    env3, P3 = _rehearsal(tmp_path / "during-token", active=[], pre_installed=OLD_TIMERS, plan_lines=PENDING_PLAN, shim_touch=str(approval3),
                          shim_touch_content=_token(PENDING_PLAN), extra_env={"WINDOW_PLAN_APPROVAL": str(approval3), "WINDOW_PLAN_WAIT_S": "3"})
    r3 = _run(env3)
    log3 = P3["log"].read_text()
    assert r3.returncode == 16, r3.stderr + log3
    assert f"an approval at {approval3} written while the plan was being generated was discarded" in log3
    assert "plan approved" not in log3 and not any(c.endswith("cli migrate") for c in _calls(P3))


def _wait_for_log(P, proc, pattern, timeout=60):
    """Poll the window log for ``pattern`` the way an operator tailing it would; the window must still be running."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        text = P["log"].read_text() if P["log"].exists() else ""
        m = re.search(pattern, text)
        if m:
            return m
        assert proc.poll() is None, f"the window ended (rc={proc.returncode}) before the log showed {pattern!r}:\n{text}"
        time.sleep(0.1)
    raise AssertionError(f"no {pattern!r} in the window log within {timeout}s:\n" + (P["log"].read_text() if P["log"].exists() else ""))


def test_plan_approval_bound_to_the_displayed_plan_passes_and_a_wrong_token_is_ignored(tmp_path):
    """The operator's path: the window displays the plan and its token with the exact approve command, then waits (the quiet-poll
    cadence, here 1 s). A file carrying another token is ignored with ONE logged line and the wait goes on; the file rewritten with the
    printed token is accepted, consumed, and only then does `registry migrate` run. The token is the sha256 of the plan file the
    operator can read, so `sha256sum` on that file verifies what the log says. The window mints its own id here (``window_id=None``):
    the plan file name carries the pid beside the second-resolution stamp."""
    approval = tmp_path / "approve-plan"
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS, plan_lines=PENDING_PLAN, window_id=None,
                        extra_env={"WINDOW_PLAN_APPROVAL": str(approval), "WINDOW_PLAN_WAIT_S": "40", "WINDOW_QUIET_SLEEP_S": "1"})
    proc = subprocess.Popen(["bash", WINDOW], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        m = _wait_for_log(P, proc, r"approve with: echo ([0-9a-f]{64}) > (\S+) \d\d:\d\d:\d\d")
        token, path = m.group(1), m.group(2)
        assert path == str(approval) and token == _token(PENDING_PLAN)
        assert token == hashlib.sha256(_plan_file(P).read_bytes()).hexdigest()
        assert not any(c.endswith("cli migrate") for c in _calls(P))                           # nothing applied while waiting
        approval.write_text("deadbeef\n")                                                       # another plan's token, or a typo
        _wait_for_log(P, proc, re.escape(f"approval at {approval} ignored: its content 'deadbeef' is not this plan's token {token} (written before the plan, or for another plan); still waiting"))
        time.sleep(2.5)                                                                           # two more polls: logged once, not per poll
        assert P["log"].read_text().count("ignored: its content 'deadbeef'") == 1
        assert proc.poll() is None and not any(c.endswith("cli migrate") for c in _calls(P))
        approval.write_text(token + "\n")                                                       # the printed command's effect
        out, err = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    log = P["log"].read_text()
    assert proc.returncode == 0, err + log
    assert f"plan approved ({approval} carried plan token {token}; consumed)" in log and "=== ff chain end" in log
    assert not approval.exists()
    calls = _calls(P)
    assert calls.index(next(c for c in calls if c.endswith("cli migrate --plan"))) < calls.index(next(c for c in calls if c.endswith("cli migrate")))
    assert "systemctl --user start registry-harvest.timer" in calls
    # the pins: recorded beside the plan file at the plan stage, re-checked immediately before the apply, and the normal path unchanged
    plan = _plan_file(P)
    assert re.fullmatch(r"migrate-plan-\d{8}T\d{6}Z-\d+\.txt", plan.name), plan.name       # the pid beside the second-resolution stamp
    assert not (plan.stat().st_mode & 0o222), oct(plan.stat().st_mode)                       # written once, then read-only
    pins = plan.with_suffix(".pins").read_text().splitlines()
    mig_sha = hashlib.sha256((P["mig_dir"] / "0001_synthetic.sql").read_bytes()).hexdigest()
    assert pins == [f"base {FULL_SHA}", f"plan {token}", f"{mig_sha}  ./0001_synthetic.sql"], pins
    assert f"pins recorded in {plan.with_suffix('.pins')}: base {FULL_SHA}, plan {token}, 1 migration file(s) under {P['mig_dir']}" in log
    assert f"pin check before the apply: base {FULL_SHA}, 1 migration file(s), plan file sha256 = token; unchanged since the plan" in log
    assert log.index("pin check before the apply") < log.index("-- migrate (apply)")


def test_a_rebuild_line_refuses_the_plan_whatever_approval_exists(tmp_path):
    """A `rebuild table` line is the STOP no approval overrides: with WINDOW_PLAN_APPROVAL set and the RIGHT token for that very plan
    both pre-written and written while the plan is rendered, the window exits 15 at the rebuild check -- before any approval is
    consulted (no approve command printed, no wait), nothing applied."""
    approval = tmp_path / "approve-plan"
    approval.write_text(_token(REBUILD_PLAN) + "\n")
    env, P = _rehearsal(tmp_path, active=[], pre_installed=OLD_TIMERS, plan_lines=REBUILD_PLAN, shim_touch=str(approval),
                        shim_touch_content=_token(REBUILD_PLAN), extra_env={"WINDOW_PLAN_APPROVAL": str(approval), "WINDOW_PLAN_WAIT_S": "3"})
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 15, r.stderr + log
    assert "migration plan has a rebuild line (row_history): a STOP, nothing applied, no approval overrides it" in log
    assert "approve with:" not in log and "waiting up to" not in log and "plan approved" not in log and "=== ff chain end" not in log
    assert not any(c.endswith("cli migrate") for c in _calls(P))


def _waiting_window(tmp_path, **kw):
    """Start a window that pauses for the operator's approval (1 s poll) and read the approve command off its log; returns
    (proc, P, token, approval). The caller manipulates the world, approves, and reads the outcome with ``_finish``. The window mints
    its own pid-bearing id (the lock's holder line and the plan file name are asserted on it)."""
    approval = tmp_path / "approve-plan"
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS, plan_lines=PENDING_PLAN, window_id=None,
                        extra_env={"WINDOW_PLAN_APPROVAL": str(approval), "WINDOW_PLAN_WAIT_S": "40", "WINDOW_QUIET_SLEEP_S": "1"}, **kw)
    proc = subprocess.Popen(["bash", WINDOW], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        m = _wait_for_log(P, proc, r"approve with: echo ([0-9a-f]{64}) > (\S+) \d\d:\d\d:\d\d")
    except BaseException:
        proc.kill(); proc.communicate()
        raise
    assert m.group(2) == str(approval) and m.group(1) == _token(PENDING_PLAN)
    return proc, P, m.group(1), approval


def _finish(proc, timeout=60):
    try:
        out, err = proc.communicate(timeout=timeout)
    finally:
        if proc.poll() is None:
            proc.kill(); proc.communicate()
    return proc.returncode, err


def _no_apply_timers_restored(P):
    calls = _calls(P)
    assert not any(c.endswith("cli migrate") or "dimensions load" in c or c.endswith("cli ingest --retry-held") for c in calls), calls
    assert "systemctl --user start registry-harvest.timer" in calls


def test_a_second_window_is_refused_by_name_while_one_holds_the_lock_and_touches_nothing(tmp_path):
    """One window at a time, enforced: while window A waits for its approval, window B over the same state root is refused at once --
    exit 18, the holder's line (A's id and pid) printed, no start marker (the measurement counts no window), not one shim call (B
    stopped no timer, took no reservation, moved no base). A then approves and completes as if B had never run."""
    proc, P, token, approval = _waiting_window(tmp_path)
    try:
        env_b, P_b = _rehearsal(tmp_path / "b", active=[], pre_installed=OLD_TIMERS, plan_lines=PENDING_PLAN, state_dir=P["state"], window_id=None)
        r_b = _run(env_b)
        log = P["log"].read_text()
        assert r_b.returncode == 18, r_b.stderr + log
        holder = (P["state"] / "windows" / ".window.lock").read_text().strip()
        assert re.fullmatch(rf"window \d{{8}}T\d{{6}}Z-{proc.pid} pid {proc.pid} started \d{{4}}-\d\d-\d\dT\d\d:\d\d:\d\dZ", holder), holder
        assert re.search(rf"window \d{{8}}T\d{{6}}Z-\d+ refused: another window holds {re.escape(str(P['state'] / 'windows' / '.window.lock'))} \({re.escape(holder)}\); one window at a time \d\d:\d\d:\d\d", log), log
        assert log.count("=== ff chain start") == 1 and log.count("-- reads reservation") == 1 and not P_b["calls"].exists()
        assert proc.poll() is None                                                            # A still waiting, untouched
        approval.write_text(token + "\n")
        rc, err = _finish(proc)
    finally:
        if proc.poll() is None:
            proc.kill(); proc.communicate()
    log = P["log"].read_text()
    assert rc == 0, err + log
    assert log.count("=== ff chain end") == 1 and "plan approved" in log and "pin check before the apply" in log


def test_the_base_advancing_while_the_approval_is_pending_refuses_the_apply_by_name(tmp_path):
    """The approval binds the plan read at the fast-forwarded base; if the base moves while the operator reads (another landing's
    fast-forward), the RIGHT token no longer buys an apply: the pin re-check before `registry migrate` names both shas and exits 19
    with nothing applied, the timers restored."""
    proc, P, token, approval = _waiting_window(tmp_path)
    new_sha = "feedface" * 5
    try:
        P["head"].write_text(new_sha + "\n")                                                  # the base advanced under the window
        approval.write_text(token + "\n")
        rc, err = _finish(proc)
    finally:
        if proc.poll() is None:
            proc.kill(); proc.communicate()
    log = P["log"].read_text()
    assert rc == 19, err + log
    assert "plan approved" in log                                                             # the approval itself was in order
    assert f"pins changed between the plan and the apply; base HEAD {FULL_SHA} -> {new_sha}; the approved plan is not what the apply would do" in log
    assert "nothing applied" in log and "window ended early (rc=19)" in log and "-- migrate (apply)" not in log and "=== ff chain end" not in log
    _no_apply_timers_restored(P)


def test_a_migration_file_added_or_changed_while_the_approval_is_pending_refuses_the_apply_by_name(tmp_path):
    """`registry migrate` enumerates the directory's CURRENT files: a migration added (or one rewritten) while the approval was pending
    is not in the approved plan, so the re-check prints the recorded (<) and current (>) file lines and exits 19, nothing applied."""
    proc, P, token, approval = _waiting_window(tmp_path)
    old_sha = hashlib.sha256((P["mig_dir"] / "0001_synthetic.sql").read_bytes()).hexdigest()
    try:
        (P["mig_dir"] / "0002_sneaky_rebuild.sql").write_text("CREATE TABLE sneaky (id INTEGER);\n")
        (P["mig_dir"] / "0001_synthetic.sql").write_text("CREATE TABLE synthetic (id INTEGER, extra TEXT);\n")
        approval.write_text(token + "\n")
        rc, err = _finish(proc)
    finally:
        if proc.poll() is None:
            proc.kill(); proc.communicate()
    log = P["log"].read_text()
    assert rc == 19, err + log
    new_sha = hashlib.sha256((P["mig_dir"] / "0001_synthetic.sql").read_bytes()).hexdigest()
    sneaky_sha = hashlib.sha256((P["mig_dir"] / "0002_sneaky_rebuild.sql").read_bytes()).hexdigest()
    assert f"  < {old_sha}  ./0001_synthetic.sql" in log and f"  > {new_sha}  ./0001_synthetic.sql" in log and f"  > {sneaky_sha}  ./0002_sneaky_rebuild.sql" in log
    assert f"pins changed between the plan and the apply; migration files under {P['mig_dir']} differ (recorded < vs now >):; the approved plan" in log
    assert f"base HEAD {FULL_SHA}" not in log.split("pins changed")[1]                       # only the axis that moved is named
    assert "nothing applied" in log and "-- migrate (apply)" not in log and "=== ff chain end" not in log
    _no_apply_timers_restored(P)


def test_a_plan_file_rewritten_after_the_token_refuses_the_apply_by_name(tmp_path):
    """The plan file is written once and read once; the checks and the token come from those bytes. An external rewrite of the file
    after the token (its read-only bit lifted, as a same-name writer would) breaks the displayed-plan correspondence: the re-check
    reads a different sha256 and exits 19 with both values, nothing applied."""
    proc, P, token, approval = _waiting_window(tmp_path)
    try:
        plan = _plan_file(P)
        os.chmod(plan, 0o644)
        plan.write_text("\n".join(["schema version 98; code 99; pending: 0099_synthetic_slug", "rebuild table row_history: 1 rows (rewritten)"]) + "\n")
        approval.write_text(token + "\n")                                                     # the token of the plan as displayed
        rc, err = _finish(proc)
    finally:
        if proc.poll() is None:
            proc.kill(); proc.communicate()
    log = P["log"].read_text()
    assert rc == 19, err + log
    rewritten = hashlib.sha256(plan.read_bytes()).hexdigest()
    assert f"pins changed between the plan and the apply; plan file sha256 {token} -> {rewritten}; the approved plan is not what the apply would do" in log
    assert "nothing applied" in log and "-- migrate (apply)" not in log and "=== ff chain end" not in log
    _no_apply_timers_restored(P)


def test_a_plan_with_nothing_pending_needs_no_approval(tmp_path):
    approval = tmp_path / "approve-plan"
    env, P = _rehearsal(tmp_path, active=[], pre_installed=OLD_TIMERS, extra_env={"WINDOW_PLAN_APPROVAL": str(approval), "WINDOW_PLAN_WAIT_S": "3"})
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    assert "plan: nothing pending, the shape matches; no approval needed" in log and "waiting up to" not in log and "=== ff chain end" in log


def test_a_plan_without_its_version_line_aborts_before_the_apply(tmp_path):
    """`registry migrate --plan` that did not render (a traceback, a missing database) is a failed migrate stage, exit 8, nothing applied."""
    env, P = _rehearsal(tmp_path, active=[], pre_installed=OLD_TIMERS, plan_lines=["Traceback (most recent call last):", "boom"])
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 8, r.stderr + log
    assert "migrate --plan did not render its version line" in log and "nothing applied" in log
    assert not any(c.endswith("cli migrate") for c in _calls(P))


def test_a_reused_window_id_is_refused_under_the_lock_before_the_base_or_the_timers_move(tmp_path):
    """An operator-supplied WINDOW_ID is honored (the reservation names it, and the tests pin one through `--op`) and the plan file is written
    ONCE per id -- so a retry with an exported id after an earlier window must be refused BEFORE anything moves. A first window under that id
    completes. The base then advances to a head whose plan has a pending migration (the retry's danger: new units installed and the harvester
    timer restored with the migration unapplied; db.open_or_create refuses the old database under new code). The retry under the SAME id exits
    8 at once, under the lock: one refusal line, no second start marker, not one shim call -- no reservation taken (so none released), no
    `systemctl stop` (its active set untouched), no git (the base untouched), no installer (no unit file written), no plan read -- and the first
    window's plan file stands."""
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS)
    assert _run(env).returncode == 0, P["log"].read_text()
    plan = P["state"] / "windows" / "migrate-plan-W1.txt"
    before = plan.read_bytes()
    # the retry's world: its own shims and calls log, the SAME state (the lock, the plan file), the same id W1, a base that now needs a migration
    env2, P2 = _rehearsal(tmp_path / "retry", active=["registry-harvest.timer"], pre_installed=OLD_TIMERS, plan_lines=PENDING_PLAN, state_dir=P["state"])
    P2["head"].write_text("feedface" * 5 + "\n")
    r = _run(env2)
    log = P["log"].read_text()
    assert r.returncode == 8, r.stderr + log
    assert f"window W1 refused: plan file {plan} already exists (a window id collision: an earlier window ran under this id); nothing reserved, stopped, fast-forwarded or installed -- a retry needs a new id" in log
    assert log.count("=== ff chain start") == 1 and log.count("=== ff chain end") == 1 and "ABORTED" not in log
    assert not P2["calls"].exists(), "the retry made no shim call: no reservation, no systemctl, no git, no installer, no plan read"
    assert (P2["tmp"] / "shim-active").read_text().split() == ["registry-harvest.timer"], "the timers' active set is untouched"
    assert not any(f.endswith(".service") for f in os.listdir(P2["unit_dir"])), "no unit was installed"
    assert plan.read_bytes() == before and sorted(p.name for p in (P["state"] / "windows").glob("migrate-plan-*")) == ["migrate-plan-W1.pins", "migrate-plan-W1.txt"]
    # windows.py reads one completed window and no second one
    windows = _load("windows").read_logs([str(P["log"])])
    assert len(windows) == 1 and windows[0]["completed"]


def test_a_plan_file_appearing_after_the_preflight_is_still_refused_at_the_plan_stage_as_the_race_guard(tmp_path):
    """The plan-stage check stays behind the preflight: a file that appears at this id's name AFTER the lock's preflight
    (here planted by the reservation shim -- a hand-written file, a writer outside the lock) is refused before the plan is written: exit 8,
    `nothing applied`, no `-- migrate (apply)`, the timers restored by the trap, the reservation released, the ABORTED marker."""
    state = tmp_path / "home" / ".local" / "state" / "registry-landing"
    planted = state / "windows" / "migrate-plan-W1.txt"
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS, state_dir=state, extra_env={"SHIM_RESERVE_TOUCH": str(planted)})
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 8, r.stderr + log
    assert "-- reads reservation" in log and "-- migrate (plan)" in log and "-- migrate (apply)" not in log
    assert f"plan file {planted} already exists (a window id collision); nothing applied" in log and "=== ff chain ABORTED (plan file" in log
    calls = _calls(P)
    assert any("cluster-reads release --op W1" in c for c in calls) and "systemctl --user start registry-harvest.timer" in calls
    assert not any(c.endswith("cli migrate --plan") or c.endswith("cli migrate") for c in calls)


def test_the_reconcile_slot_runs_the_dry_run_between_the_dimensions_load_and_the_cluster_stages_and_the_apply_only_when_asked(tmp_path):
    """WINDOW_RECONCILE names the `registry reconcile` actions: each dry run is stamped and runs after the dimensions load and before
    the ingest (its records fold in the one ingest; with cluster adapters named it precedes the cluster stages); --apply runs only with
    WINDOW_RECONCILE_APPLY=1; a failing apply (the release's gate, exit 2, included) exits 17 by name. windows.py classes the stamps
    `reconcile`."""
    env, P = _rehearsal(tmp_path, active=[], pre_installed=OLD_TIMERS, extra_env={"WINDOW_RECONCILE": "run-patch-extra"})
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    calls = _calls(P)
    dry = [c for c in calls if "cli reconcile" in c]
    assert dry == ["python -m sisteer.registry.cli reconcile run-patch-extra --dry-run"], dry
    order = [next(i for i, c in enumerate(calls) if key in c) for key in ("dimensions load", "reconcile run-patch-extra --dry-run", "cli ingest")]
    assert order == sorted(order), order
    assert "-- reconcile run-patch-extra (dry run)" in _stamps(log) and "-- reconcile run-patch-extra (apply)" not in _stamps(log)
    windows = _load("windows").read_logs([str(P["log"])])
    assert [st["class"] for st in windows[0]["stages"]] == ["reservation", "wal", "config_convert", "authority", "base", "install", "install", "migrate_plan", "migrate", "dims", "reconcile",
                                                             "ingest", "ingest_end", "status_heavy"]
    env2, P2 = _rehearsal(tmp_path / "apply", active=[], pre_installed=OLD_TIMERS, extra_env={"WINDOW_RECONCILE": "run-patch-extra", "WINDOW_RECONCILE_APPLY": "run-patch-extra"})
    r2 = _run(env2)
    assert r2.returncode == 0, r2.stderr + P2["log"].read_text()
    calls2 = [c for c in _calls(P2) if "cli reconcile" in c]
    assert calls2 == ["python -m sisteer.registry.cli reconcile run-patch-extra --dry-run", "python -m sisteer.registry.cli reconcile run-patch-extra --apply"], calls2
    env3, P3 = _rehearsal(tmp_path / "refused", active=["registry-harvest.timer"], pre_installed=OLD_TIMERS, fail_on='*"run-patch-extra --apply"',
                          extra_env={"WINDOW_RECONCILE": "run-patch-extra", "WINDOW_RECONCILE_APPLY": "run-patch-extra"})
    r3 = _run(env3)
    log3 = P3["log"].read_text()
    assert r3.returncode == 17, r3.stderr + log3
    assert "reconcile run-patch-extra apply FAILED rc=1 (2 = the release's gate refused it, or a row held behind an unfolded record; nothing written for it); abort" in log3
    calls3 = _calls(P3)
    assert not any("adapters run" in c or c.endswith("cli ingest --retry-held") for c in calls3) and "systemctl --user start registry-harvest.timer" in calls3
    assert "=== ff chain end" not in log3 and "=== ff chain ABORTED (reconcile run-patch-extra apply FAILED" in log3
    # with cluster adapters named, the slot sits between the dimensions load and the first cluster stage
    env4, P4 = _cluster_rehearsal(tmp_path / "cluster")
    env4["WINDOW_RECONCILE"] = "run-patch-extra"
    r4 = _run(env4)
    log4 = P4["log"].read_text()
    assert r4.returncode == 0, r4.stderr + log4
    st = _cluster_stamps(log4)
    assert st.index("-- dimensions load") < st.index("-- reconcile run-patch-extra (dry run)") < st.index("-- cluster reads inbox-merge"), st


# --------------------------------------------------------------------------------------------
# the configuration conversion stage and convert_config.py
# --------------------------------------------------------------------------------------------
_TABLE = "modal"   # the table's name, kept out of every literal so this module passes the no-Modal scan it is part of


def _config_with_table() -> str:
    """A configuration shaped like the live one: six tables, an array of tables, comments and spacing -- and the table to remove."""
    return ("# the operating configuration\n\n[registry]\nroot = \"~/registry\"\nuser = \"tester\"   # the cluster login\n\n[paper]\n"
            "checkout = \"~/repo-paper\"\nkeep_failed_builds = 3\n\n[status]\npage_dir = \"~/figs/status\"\n\n[snapshot]\n"
            "in_flight_every_s = 300\nrunlevel_only_in_flight = true\n\n[harvest]\npull_clusters = true\n\n[[clusters]]\nname = \"cluster_a\"\n"
            "host = \"login-a\"\nshared = \"/shared/tester\"\n\n[[clusters]]\nname = \"cluster_b\"\nhost = \"login-b\"\n"
            f"shared = \"/workspace/tester\"\n\n[{_TABLE}]\nenabled = true  # pull + snapshot destinations ON\nvolume = \"twosi-fits\"\n"
            f"python = \"~/venvs/{_TABLE}/bin/python\"\nlog_path = \"/registry/log\"\n\n[repo]\ncheckout = \"~/repo\"\npython = \"~/repo/.venv/bin/python\"\n")


def _convert():
    import importlib.util
    spec = importlib.util.spec_from_file_location("convert_config_under_test", os.path.join(REPO, "scripts", "registry", "landing", "convert_config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_convert_config_strips_the_table_textually_backs_up_byte_for_byte_restores_and_refuses_what_it_cannot_prove(tmp_path):
    """convert_config.py: the table's block is removed from its header to the next header, every other byte kept; the backup under
    state/ hashes as the original and is read-only; the new file re-parses to the old document minus the table; KEY=VALUE and --json
    outputs; a file without the table and a missing file are no-ops; --dry-run writes nothing; --restore puts the bytes back and
    verifies; a backup collision and an unparseable file refuse."""
    import tomllib
    cc = _convert()
    root = tmp_path / "root"
    root.mkdir()
    cfg = root / "config.toml"
    text = _config_with_table()
    cfg.write_text(text)
    old = tomllib.loads(text)
    assert _TABLE in old and old["clusters"][1]["name"] == "cluster_b"
    # the pure strip: header..next header removed, nothing else touched
    new_text, removed = cc.strip_table(text)
    assert removed == 6 and "[" + _TABLE + "]" not in new_text and tomllib.loads(new_text) == {k: v for k, v in old.items() if k != _TABLE}
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l == "[" + _TABLE + "]")
    stop = next(j for j in range(start + 1, len(lines)) if lines[j].startswith("["))
    assert stop - start == 6 and new_text.splitlines() == lines[:start] + lines[stop:], "the block from the header to the next header is gone; every other line is kept"
    # dry run writes nothing
    r = subprocess.run([sys.executable, cc.__file__, "--root", str(root), "--dry-run"], capture_output=True, text=True)
    assert r.returncode == 0 and "CONVERTED=1" in r.stdout and "DRY_RUN=1" in r.stdout and cfg.read_text() == text and not (root / "state").exists(), r.stdout
    # the conversion: backup, strip, re-parse, the KEY=VALUE lines window.sh reads
    r = subprocess.run([sys.executable, cc.__file__, "--root", str(root)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    kv = dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l)
    assert kv["CONVERTED"] == "1" and kv["REMOVED_LINES"] == "6" and kv["CONFIG"] == str(cfg) and kv["BACKUP"].startswith(str(root / "state" / "config.toml.bak-"))
    backup = kv["BACKUP"]
    assert open(backup, "rb").read() == text.encode() and (os.stat(backup).st_mode & 0o222) == 0, "the backup is byte-identical and read-only"
    assert cfg.read_text() == new_text and kv["SHA_BEFORE"] == hashlib.sha256(text.encode()).hexdigest() and kv["SHA_AFTER"] == hashlib.sha256(new_text.encode()).hexdigest()
    assert tomllib.loads(cfg.read_text()) == {k: v for k, v in old.items() if k != _TABLE}
    # a second run: nothing to convert, nothing written, the same sha before and after
    r = subprocess.run([sys.executable, cc.__file__, "--root", str(root), "--json"], capture_output=True, text=True)
    d = json.loads(r.stdout)
    assert r.returncode == 0 and d["CONVERTED"] == 0 and "nothing to convert" in d["REASON"] and d["SHA_BEFORE"] == d["SHA_AFTER"] == kv["SHA_AFTER"] and d["BACKUP"] is None
    assert len(list((root / "state").glob("config.toml.bak-*"))) == 1
    # --restore puts the original bytes back and verifies
    r = subprocess.run([sys.executable, cc.__file__, "--root", str(root), "--restore", backup], capture_output=True, text=True)
    assert r.returncode == 0 and "RESTORED=1" in r.stdout and cfg.read_text() == text, r.stdout + r.stderr
    # a backup collision refuses with nothing written (the stamp is pinned by monkeypatching the module's clock)
    saved = cc.time.strftime
    try:
        stamp = os.path.basename(backup)[len("config.toml.bak-"):]
        cc.time.strftime = lambda fmt, *a: stamp if fmt == "%Y%m%dT%H%M%SZ" else saved(fmt, *a)
        rc = cc.main(["--root", str(root)])
    finally:
        cc.time.strftime = saved
    assert rc == 3 and cfg.read_text() == text
    # no file: a no-op; an unparseable file: refused (exit 2)
    r = subprocess.run([sys.executable, cc.__file__, "--root", str(tmp_path / "empty")], capture_output=True, text=True)
    assert r.returncode == 0 and "CONVERTED=0" in r.stdout and "no config file" in r.stdout
    cfg.write_text("this is = not [toml\n")
    r = subprocess.run([sys.executable, cc.__file__, "--root", str(root)], capture_output=True, text=True)
    assert r.returncode == 2 and "does not parse as TOML" in r.stderr
    # a restore whose source cannot be read refuses (exit 2) and leaves the file alone
    r = subprocess.run([sys.executable, cc.__file__, "--root", str(root), "--restore", str(root / "absent.bak")], capture_output=True, text=True)
    assert r.returncode == 2 and cfg.read_text() == "this is = not [toml\n"


def _config_calls(calls):
    return [c for c in calls if c.startswith("python ") and (" config --check" in c or " config --json" in c)]


def test_config_convert_runs_after_the_writers_are_quiet_before_the_base_moves_and_validates_with_both_loaders(tmp_path):
    """The conversion stage: the table is stripped and backed up, then `config --check` runs under the WORKTREE's loader
    (WINDOW_NEW_PY / WINDOW_NEW_PYPATH) and `config --json` under the base's, both BEFORE the fast-forward; the window goes on to
    `base at` with the converted file in place and the backup under state/."""
    import tomllib
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS)
    cfg = P["home"] / "registry" / "config.toml"
    text = _config_with_table()
    cfg.write_text(text)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stdout + r.stderr + log
    st = _stamps(log)
    assert st.index("writers quiet") < st.index("-- config convert") < st.index("base at deadbee"), st
    assert re.search(r"^config convert: the " + _TABLE + r" table removed \(6 line\(s\)\); backup written; backup .*/state/config\.toml\.bak-\d{8}T\d{6}Z$", log, re.M), log
    assert re.search(r"^config convert: validated by the new loader \(config --check under .*/worktree/src\) and the base's loader \(config --json\); converted=1 \d\d:\d\d:\d\d$", log, re.M), log
    assert "[" + _TABLE + "]" not in cfg.read_text() and tomllib.loads(cfg.read_text()) == {k: v for k, v in tomllib.loads(text).items() if k != _TABLE}
    backups = list((P["home"] / "registry" / "state").glob("config.toml.bak-*"))
    assert len(backups) == 1 and backups[0].read_text() == text
    calls = _calls(P)
    cc = _config_calls(calls)
    assert [c.split(" -m sisteer.registry.cli ", 1)[1] if " -m sisteer.registry.cli " in c else c for c in cc] == \
        [f"--root {P['home']}/registry config --check", "config --json"], cc
    ff = next(i for i, c in enumerate(calls) if c.startswith("git ") and "merge" in c and "--ff-only" in c)
    assert all(calls.index(c) < ff for c in cc), "both validations precede the fast-forward"
    assert (P["state"] / "windows" / "config-check-W1.txt").exists() and (P["state"] / "windows" / "config-json-W1.txt").exists()
    assert "=== ff chain end" in log


def test_a_failed_new_loader_check_restores_the_backup_byte_for_byte_and_aborts_before_the_base_or_the_units_move(tmp_path):
    """The abort path (R12's injected failure): `config --check` under the worktree's loader fails -> the backup is restored (the file
    hashes as before), exit 24, no `base at`, no install, the timers restored under the old code, the ABORTED marker names the cause,
    and no retired unit is ever started."""
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer", "registry-status.timer"], pre_installed=OLD_TIMERS, fail_on='*"config --check"*')
    cfg = P["home"] / "registry" / "config.toml"
    text = _config_with_table()
    cfg.write_text(text)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 24, r.stdout + r.stderr + log
    assert cfg.read_bytes() == text.encode(), "the original bytes are back"
    assert "config convert: backup " in log and "restored byte for byte" in log and "RESTORED=1" in log
    assert "config convert: the configuration failed the new loader's check" in log and "window ended early (rc=24)" in log
    assert "base at" not in log and "-- install units" not in log and "-- migrate (plan)" not in log
    assert re.search(r"^=== ff chain ABORTED \(config convert: the configuration failed the new loader's check.*\) \d{4}-", log, re.M), log
    assert "timers restored (trap; the set above)" in log and not P["maint"].exists()
    calls = _calls(P)
    assert len(_config_calls(calls)) == 1 and not any(" config --json" in c for c in calls), "the base's validation never ran after the first failure"
    starts = _starts(calls)
    assert starts and all("registry-harvest.timer" in c or "registry-status.timer" in c for c in starts), starts
    _assert_no_retired_start(calls)
    assert len(list((P["home"] / "registry" / "state").glob("config.toml.bak-*"))) == 1, "the backup stays for the operator"


def test_a_failed_base_loader_parse_restores_and_aborts_the_same_way(tmp_path):
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS, fail_on='*"config --json"*')
    cfg = P["home"] / "registry" / "config.toml"
    text = _config_with_table()
    cfg.write_text(text)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 24 and cfg.read_bytes() == text.encode(), r.stdout + r.stderr + log
    assert "config convert: the base's loader does not parse the configuration (config --json" in log and "base at" not in log
    assert re.search(r"^=== ff chain ABORTED \(config convert: the base's loader", log, re.M) and "timers restored (trap; the set above)" in log
    assert [c.split(" config ", 1)[1] for c in _config_calls(_calls(P))] == ["--check", "--json"]


def test_config_convert_without_the_table_is_a_no_op_that_still_validates(tmp_path):
    """No table (every window after the cutover): nothing written, no backup, both validations still run, the stage line names it."""
    env, P = _rehearsal(tmp_path, active=[], pre_installed=OLD_TIMERS)
    cfg = P["home"] / "registry" / "config.toml"
    text, _ = _convert().strip_table(_config_with_table())
    cfg.write_text(text)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stdout + r.stderr + log
    assert re.search(r"^config convert: no " + _TABLE + r" table \(nothing to convert\)$", log, re.M) and "converted=0" in log, log
    assert cfg.read_text() == text and not list((P["home"] / "registry" / "state").glob("config.toml.bak-*"))
    assert [c.split(" config ", 1)[1] for c in _config_calls(_calls(P))] == ["--check", "--json"] and "=== ff chain end" in log
    # a failed check with nothing converted: nothing to restore, still an abort before the base moves
    env, P = _rehearsal(tmp_path / "second", active=[], pre_installed=OLD_TIMERS, fail_on='*"config --check"*')
    (P["home"] / "registry" / "config.toml").write_text(text)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 24 and "nothing was converted, nothing to restore" in log and "base at" not in log, log

# --------------------------------------------------------------------------------------------
# the acceptance slot in the Slurm-only window, and the by-name refusal of main's post-ingest pass
# --------------------------------------------------------------------------------------------


def _accept_shim(tmp_path, fail=False):
    """A stand-in for the acceptance script: records its arguments, writes the before-phase file the window tails, exits as told."""
    calls = tmp_path / "accept-calls.log"
    body = f'''echo "accept $*" >> "{calls}"
out=""; while [ $# -gt 0 ]; do [ "$1" = "--out" ] && out="$2"; shift; done
echo "verdict {'FAIL' if fail else 'PASS'}"
exit {1 if fail else 0}
'''
    return _shim(tmp_path / "bin", "accept", body), calls


def test_acceptance_out_runs_the_before_phase_after_the_dimensions_and_forces_the_reader_state_export_before_the_adapters(tmp_path):
    """ACCEPTANCE_OUT: the before-phase runs after `-- dimensions load` and BEFORE `-- reader-state export`, with the window's id, the base as
    the code tree and the bumped list. The window names two adapters with no cluster policy (CLUSTER_ADAPTERS=none), so the export the
    protocol requires runs here over $ADAPTERS (the cluster block's own export runs only for a cluster set). In the branch's world what
    follows is the Slurm-only sequence: the cluster block SKIPPED, both adapters RECORDED REFUSED (no `adapters run` ever), the one ingest,
    and the window ends ABORTED (refused: alpha beta), exit 20 -- the acceptance and the export stand before all of that."""
    env, P = _rehearsal(tmp_path, active=[], pre_installed=OLD_TIMERS, adapters="alpha beta")
    shim, acalls = _accept_shim(tmp_path)
    acc = tmp_path / "acc"
    lst = tmp_path / "bumped.txt"
    lst.write_text("alpha\nbeta\n")
    env.update({"ACCEPTANCE_OUT": str(acc), "ACCEPTANCE_SCRIPT": str(shim), "ACCEPTANCE_LIST": str(lst)})
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 20, r.stderr + log
    assert _stamps(log) == ["=== ff chain start ADAPTERS=alpha beta", "-- reads reservation", "writers quiet", "-- wal checkpoint", "-- config convert", "-- store authority", "base at deadbee",
                            "-- install units", "-- uninstall units", "-- migrate (plan)", "-- migrate (apply)", "-- dimensions load",
                            "-- acceptance before-phase", "-- reader-state export", "-- cluster reads SKIPPED (no named adapter has a cluster policy)",
                            "-- adapter alpha REFUSED", "-- adapter beta REFUSED", "--- ingest", "--- ingest end", "timers restored (trap; the set above)",
                            "-- status heavy refresh started once (cache initializer)", "=== ff chain ABORTED (refused: alpha beta)"], _stamps(log)
    accept = acalls.read_text().splitlines()
    wid = re.search(r"window (\S+) pid", (P["state"] / "windows" / ".window.lock").read_text()).group(1)
    assert accept == [f"accept --root {P['home']}/registry --repo {P['base']} --phase before --window {wid} --out {acc} --bumped-list {lst}"], accept
    assert (acc / "before.txt").read_text() == "verdict PASS\n" and "verdict PASS" in log
    calls = _calls(P)
    export = [c for c in calls if "reader-state export" in c]
    assert export == [f"python -m sisteer.registry.cli reader-state export --adapter alpha beta --out {P['state']}/windows/reader_state-{wid}.sqlite --repo-root {P['base']}"], export
    assert not any("adapters run" in c for c in calls) and sum(1 for c in calls if c.endswith("cli ingest --retry-held")) == 1
    assert calls.index(export[0]) < calls.index(next(c for c in calls if c.endswith("cli ingest --retry-held")))
    # without the knob nothing of this runs: no acceptance stamp, no export (the window names no cluster adapter)
    env2, P2 = _rehearsal(tmp_path / "plain", active=[], pre_installed=OLD_TIMERS)
    assert _run(env2).returncode == 0
    log2 = P2["log"].read_text()
    assert "-- acceptance before-phase" not in log2 and "-- reader-state export" not in log2      # (the temp path itself may spell the test's name)


def test_a_failing_acceptance_before_phase_aborts_the_window_before_the_export_and_the_adapters_with_the_timers_restored(tmp_path):
    """Exit 25: the acceptance abort has its own code -- 20 is the branch's recorded-refusal abort, which never comes here
    because the adapters are never reached. The stamps stop at the acceptance; the restore and the ABORTED marker carry the cause."""
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS, adapters="alpha beta")
    shim, acalls = _accept_shim(tmp_path, fail=True)
    acc = tmp_path / "acc"
    env.update({"ACCEPTANCE_OUT": str(acc), "ACCEPTANCE_SCRIPT": str(shim)})
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 25, r.stderr + log
    assert _stamps(log) == ["=== ff chain start ADAPTERS=alpha beta", "-- reads reservation", "writers quiet", "-- wal checkpoint", "-- config convert", "-- store authority", "base at deadbee",
                            "-- install units", "-- uninstall units", "-- migrate (plan)", "-- migrate (apply)", "-- dimensions load", "-- acceptance before-phase",
                            "timers restored (trap; the set above)", "-- status heavy refresh started once (cache initializer)",
                            f"=== ff chain ABORTED (acceptance before-phase FAILED rc=1 (see {acc}/before.txt))"], _stamps(log)
    assert "acceptance before-phase FAILED rc=1" in log and "window ended early (rc=25)" in log and "=== ff chain end" not in log
    assert "reader-state export" not in log and "-- adapter" not in log and "cluster reads" not in log and " ingest" not in P["calls"].read_text()
    assert "timers restored" in log and "systemctl --user start registry-harvest.timer" in P["calls"].read_text() and not P["maint"].exists()
    assert acalls.read_text().count("accept ") == 1 and "--bumped-list" not in acalls.read_text()       # no list given: the flag is not passed
    windows = _load("windows").read_logs([str(P["log"])])
    assert len(windows) == 1 and windows[0]["completed"] is False and windows[0]["abort_cause"].startswith("acceptance before-phase FAILED rc=1")


def test_post_ingest_adapters_are_refused_by_name_before_the_lock_and_the_reservation_with_nothing_stopped_or_installed(tmp_path):
    """main's POST_INGEST_ADAPTERS ran `adapters run` on the grading host after the first ingest and folded a second time; the grading host
    reads no store, so the knob is a CONFIGURATION refusal, by name, before the lock and the reservation: one line naming the adapters and
    the rule (a second read is the next window's first read), exit 26, no start marker (no window to the measurement), no reservation, no
    stop, no install, no `adapters run`, no ingest, no maintenance flag, the lock never taken."""
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS, extra_env={"POST_INGEST_ADAPTERS": "gamma delta"})
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 26, r.stderr + log
    assert re.search(r"^window W1 refused: POST_INGEST_ADAPTERS=gamma delta names a grading host post-ingest adapter pass, and the grading host reads no store \(C6, S2\); "
                     r"a second read is the next window's first read -- every window exports the frame fresh, so this window's ingest is already in it; "
                     r"nothing reserved, stopped, fast-forwarded or installed \d\d:\d\d:\d\d$", log, re.M), log
    assert _stamps(log) == [] and "=== ff chain" not in log and "window ended early" not in log
    calls = P["calls"].read_text().splitlines() if P["calls"].exists() else []
    assert not any(c.startswith("python ") or " stop " in c or "adapters run" in c or "cli ingest" in c or "cluster-reads reserve" in c for c in calls), calls
    assert _starts(calls) == [] and not P["maint"].exists() and not (P["state"] / "windows" / ".window.lock").exists()
    assert _load("windows").read_logs([str(P["log"])]) == []


def test_the_post_ingest_refusal_has_no_override_and_leaves_the_next_window_to_run_as_the_second_read(tmp_path):
    """The one grading host read left is the operator's `--reread --force` under REGISTRY_STORE_READ_OVERRIDE; the override does NOT open a
    post-ingest pass (the knob is refused with it set: there is no override path for it). The refused attempt leaves the lock free and
    nothing changed, so the next window over the same state (knob unset) runs the plain sequence to its end marker -- the second read IS
    that window's first read -- and the measurement sees exactly that one window."""
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS,
                        extra_env={"POST_INGEST_ADAPTERS": "gamma", "REGISTRY_STORE_READ_OVERRIDE": "operator says so"})
    r = _run(env)
    assert r.returncode == 26, r.stderr + P["log"].read_text()
    assert "window W1 refused: POST_INGEST_ADAPTERS=gamma" in P["log"].read_text() and "=== ff chain" not in P["log"].read_text()
    env2, P2 = _rehearsal(tmp_path / "next", active=["registry-harvest.timer"], pre_installed=OLD_TIMERS, state_dir=P["state"], window_id="W2")
    r2 = _run(env2)
    log = P["log"].read_text()
    assert r2.returncode == 0, r2.stderr + log
    assert _stamps(log) == ["=== ff chain start ADAPTERS=", "-- reads reservation", "writers quiet", "-- wal checkpoint", "-- config convert", "-- store authority", "base at deadbee", "-- install units",
                            "-- uninstall units", "-- migrate (plan)", "-- migrate (apply)", "-- dimensions load", "--- ingest", "--- ingest end",
                            "timers restored (trap; the set above)", "-- status heavy refresh started once (cache initializer)", "=== ff chain end"], _stamps(log)
    assert sum(1 for c in _calls(P2) if c.endswith("cli ingest --retry-held")) == 1 and not any("adapters run" in c for c in _calls(P2))
    windows = _load("windows").read_logs([str(P["log"])])
    assert len(windows) == 1 and windows[0]["completed"] is True


def test_the_acceptance_stamp_is_classed_by_the_measurement_script(tmp_path):
    env, P = _rehearsal(tmp_path, active=[], pre_installed=OLD_TIMERS, adapters="alpha beta")
    shim, _ = _accept_shim(tmp_path)
    env.update({"ACCEPTANCE_OUT": str(tmp_path / "acc"), "ACCEPTANCE_SCRIPT": str(shim)})
    assert _run(env).returncode == 20                      # the adapters, refused by name after the acceptance and the export (the branch's abort)
    windows = _load("windows").read_logs([str(P["log"]), str(P["compat"])])
    assert len(windows) == 1 and windows[0]["completed"] is False and windows[0]["abort_cause"] == "refused: alpha beta"
    assert [s["class"] for s in windows[0]["stages"]] == ["reservation", "wal", "config_convert", "authority", "base", "install", "install", "migrate_plan", "migrate", "dims",
                                                          "acceptance", "cluster", "cluster_skip", "adapter_refused", "adapter_refused", "ingest", "ingest_end", "status_heavy"]


# --------------------------------------------------------------------------------------------
# the store-mirror timer stops with the others; `-- store authority` establishes the certification inside
# the window, after the convert and before the base moves
# --------------------------------------------------------------------------------------------


def _authority_doc(components, strict=False):
    """A check-authority JSON document for the regpy shim: `components` = {slug: row}; the counts derived the way stores.check_authority does."""
    rows = {}
    for slug, row in components.items():
        if row.get("state") == "not_applicable":
            rows[slug] = {"authority": "cluster_a_origin", "state": "not_applicable", "problems": [], "notes": []}
            continue
        r = {"authority": row.get("authority", "grading_host_copy"), "current": row.get("current"), "exists": row.get("exists", bool(row.get("current"))),
             "verified": row.get("verified", False), "sealed": row.get("sealed", False), "published_record": row.get("published_record", False),
             "problems": list(row.get("problems", [])), "notes": list(row.get("notes", []))}
        r["certified"] = bool(r["current"]) and r["exists"] and r["verified"] and r["sealed"] and r["published_record"]
        r["state"] = "ok" if not r["problems"] else r["problems"][0]
        rows[slug] = r
    problems = sorted(f"{slug}: {', '.join(r['problems'])}" for slug, r in rows.items() if r["problems"])
    checked = [r for r in rows.values() if r["state"] != "not_applicable"]
    return {"ok": not problems, "require_certified": strict, "n_components": len(rows), "n_checked": len(checked),
            "n_certified": sum(1 for r in checked if r.get("certified")), "n_problems": len(problems), "problems": problems, "components": rows}


def _authority_side_files(tmp_path, first, second, second_rc=None):
    """What the shim prints for the stage's two check-authority calls (the rc follows `ok` unless given)."""
    for n, doc, rc in ((1, first, None), (2, second, second_rc)):
        (tmp_path / f"shim-check-authority-{n}.json").write_text(json.dumps(doc))
        (tmp_path / f"shim-check-authority-{n}.rc").write_text(f"{rc if rc is not None else (0 if doc['ok'] else 1)}\n")


def _summary(doc):
    return f"certified {doc['n_certified']} of {doc['n_checked']} checked; {doc['n_problems']} problem(s)" + (": " + "; ".join(doc["problems"]) if doc["problems"] else "")


def test_the_store_authority_stage_runs_after_the_convert_before_the_fast_forward_and_verifies_only_the_verified_but_unsealed_or_unrecorded(tmp_path):
    """With the store-mirror timer stopped, `-- store authority` runs under the worktree's new code after `-- config convert` and before
    `base at`: check-authority, then ONE verify-authority for each component whose `current` exists and reads VERIFIED but UNSEALED or
    UNRECORDED (what main's mirror leaves behind), none for a never-published, a dangling or a moved-ahead (unverified) one -- those are
    named as not acted on -- then the strict check; the counts go to the log, both JSON results beside the window's artifacts, and the
    measurement classes the stamp `authority`. WINDOW_REQUIRE_CERTIFIED unset: the problems are a log line and the window goes on."""
    first = _authority_doc({
        "a_unsealed": {"current": "aaaa1111", "verified": True, "sealed": False, "published_record": True, "problems": ["unsealed"]},
        "b_unrecorded": {"current": "bbbb2222", "verified": True, "sealed": True, "published_record": False, "problems": ["unrecorded"]},
        "c_never": {"current": None, "exists": False, "notes": ["never_published"]},
        "d_moved": {"current": "dddd4444", "verified": False, "sealed": True, "published_record": True, "problems": ["unverified"]},
        "e_dangling": {"current": "eeee5555", "exists": False, "problems": ["dangling"]},
        "f_ok": {"current": "ffff6666", "verified": True, "sealed": True, "published_record": True},
        "g_origin": {"state": "not_applicable"}})
    second = _authority_doc({
        "a_unsealed": {"current": "aaaa1111", "verified": True, "sealed": True, "published_record": True},
        "b_unrecorded": {"current": "bbbb2222", "verified": True, "sealed": True, "published_record": True},
        "c_never": {"current": None, "exists": False, "problems": ["not_certified_never_published"]},
        "d_moved": {"current": "dddd4444", "verified": False, "sealed": True, "published_record": True, "problems": ["unverified"]},
        "e_dangling": {"current": "eeee5555", "exists": False, "problems": ["dangling"]},
        "f_ok": {"current": "ffff6666", "verified": True, "sealed": True, "published_record": True},
        "g_origin": {"state": "not_applicable"}}, strict=True)
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer", "registry-store-mirror.timer"], pre_installed=OLD_TIMERS + NEW_TIMERS)
    _authority_side_files(tmp_path, first, second)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    stamps = _stamps(log)
    assert stamps[:7] == ["=== ff chain start ADAPTERS=", "-- reads reservation", "writers quiet", "-- wal checkpoint", "-- config convert", "-- store authority", "base at deadbee"], stamps
    assert stamps[-1] == "=== ff chain end"
    calls = _calls(P)
    checks = [c for c in calls if "stores check-authority" in c]
    verifies = [c for c in calls if "stores verify-authority" in c]
    adir = f"{P['state']}/windows/authority-W1"
    newreg = f"python -m sisteer.registry.cli --root {P['home']}/registry"           # WINDOW_NEW_PY (the shim) under the worktree's PYTHONPATH
    assert checks == [f"{newreg} stores check-authority --dest {SUB_A} --json", f"{newreg} stores check-authority --dest {SUB_A} --require-certified --json"], checks
    # a never-published grading_host_copy component (c_never) is CARRIED under the worktree's code first, then verified like the rest
    carries = [c for c in calls if "stores mirror" in c]
    assert carries == [f"{newreg} stores mirror --dest {SUB_A} --component c_never --bound-s 1800"], carries     # the controller's own bound, never `timeout`
    assert verifies == [f"{newreg} stores verify-authority --dest {SUB_A} --component a_unsealed --out {adir} --json",
                        f"{newreg} stores verify-authority --dest {SUB_A} --component b_unrecorded --out {adir} --json",
                        f"{newreg} stores verify-authority --dest {SUB_A} --component c_never --out {adir} --json"], verifies
    conv = next(i for i, c in enumerate(calls) if "config --check" in c)
    ff = next(i for i, c in enumerate(calls) if c.startswith("git ") and "merge" in c)
    assert conv < calls.index(checks[0]) < calls.index(carries[0]) < calls.index(verifies[0]) < calls.index(verifies[2]) < calls.index(checks[1]) < ff
    assert ("store authority: check-authority rc=1; never published (carried under the worktree's code, then verify-authority): c_never; "
            "verified but unsealed/unrecorded (verify-authority runs): a_unsealed b_unrecorded; "
            "not acted on here (the mirror timer carries them): d_moved(unverified) e_dangling(dangling)") in log, log
    assert "store authority: carry c_never rc=0 (never published; carried under the worktree's code)" in log
    assert "store authority: verify-authority a_unsealed rc=0" in log and "store authority: verify-authority b_unrecorded rc=0" in log and "store authority: verify-authority c_never rc=0" in log
    assert f"store authority: strict check-authority rc=1: {_summary(second)}" in log, log
    assert f"store authority: not every component is certified ({_summary(second)}); WINDOW_REQUIRE_CERTIFIED is unset, the window goes on" in log
    assert json.load(open(f"{adir}/check-authority.json")) == first and json.load(open(f"{adir}/check-authority-strict.json")) == second
    # the store-mirror timer: stopped with the others, before the drain, and restored by the trap
    drain = next(i for i, c in enumerate(calls) if c.startswith("systemctl --user is-active registry-harvest.service"))
    assert [c for c in calls if " stop " in c] == [STOP_CALL] and calls.index(STOP_CALL) < drain
    assert "registry-store-mirror.service" in calls[drain]                                         # the mirror's service joined the drain's set
    assert "systemctl --user start registry-harvest.timer registry-store-mirror.timer" in _starts(calls), _starts(calls)
    windows = _load("windows").read_logs([str(P["log"])])
    assert len(windows) == 1 and windows[0]["completed"] is True
    assert [s["class"] for s in windows[0]["stages"]][:5] == ["reservation", "wal", "config_convert", "authority", "base"]


def test_require_certified_aborts_before_the_base_moves_with_exit_27_when_a_component_stays_uncertified_and_unset_goes_on(tmp_path):
    """The cutover window (WINDOW_REQUIRE_CERTIFIED=1): a component the stage could not certify -- here one whose grading host copy moved ahead of
    `current` (unverified: the mirror timer's to carry, never verify-authority's) -- aborts the window with exit 27 right after the stage,
    before `base at`: no fast-forward, no install, nothing applied; the timers (the store mirror's included) come back and the flag is
    gone. A strict check that could not run aborts the same way. With the knob unset the same reading is a log line and the window goes on."""
    doc = _authority_doc({"x_moved": {"current": "aaaa1111", "verified": False, "sealed": True, "published_record": True, "problems": ["unverified"]},
                          "y_ok": {"current": "bbbb2222", "verified": True, "sealed": True, "published_record": True}})
    strict = _authority_doc({"x_moved": {"current": "aaaa1111", "verified": False, "sealed": True, "published_record": True, "problems": ["unverified"]},
                             "y_ok": {"current": "bbbb2222", "verified": True, "sealed": True, "published_record": True}}, strict=True)
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer", "registry-store-mirror.timer"], pre_installed=OLD_TIMERS, extra_env={"WINDOW_REQUIRE_CERTIFIED": "1"})
    _authority_side_files(tmp_path, doc, strict)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 27, r.stderr + log
    assert _stamps(log) == ["=== ff chain start ADAPTERS=", "-- reads reservation", "writers quiet", "-- wal checkpoint", "-- config convert", "-- store authority",
                            "timers restored (trap; the set above)", f"=== ff chain ABORTED (store authority: not every component is certified ({_summary(strict)}))"], _stamps(log)
    calls = _calls(P)
    assert not any("stores verify-authority" in c for c in calls)                                   # a moved-ahead copy is not verify-authority's to fix
    assert not any(c.startswith("git ") and "merge" in c for c in calls) and "base at" not in log and "-- install units" not in log and not any("migrate" in c for c in calls)
    assert "window ended early (rc=27)" in log and "=== ff chain end" not in log and "the window goes on" not in log
    assert _starts(calls) == ["systemctl --user start registry-harvest.timer registry-store-mirror.timer"], _starts(calls)
    assert not P["maint"].exists()
    _assert_no_retired_start(calls)
    windows = _load("windows").read_logs([str(P["log"])])
    assert len(windows) == 1 and windows[0]["completed"] is False and windows[0]["abort_cause"].startswith("store authority: not every component is certified")
    # a strict check that could not run (rc 2): the cutover window cannot certify, so it aborts the same way
    env2, P2 = _rehearsal(tmp_path / "broken", active=["registry-harvest.timer"], pre_installed=OLD_TIMERS, extra_env={"WINDOW_REQUIRE_CERTIFIED": "1"})
    _authority_side_files(tmp_path / "broken", doc, strict, second_rc=2)
    r2 = _run(env2)
    log2 = P2["log"].read_text()
    assert r2.returncode == 27, r2.stderr + log2
    assert "store authority: the strict check-authority could not run (rc=2:" in log2 and "base at" not in log2 and "=== ff chain end" not in log2
    # the knob unset: the same reading, logged, and the window runs to its end marker
    env3, P3 = _rehearsal(tmp_path / "routine", active=["registry-harvest.timer"], pre_installed=OLD_TIMERS)
    _authority_side_files(tmp_path / "routine", doc, strict)
    r3 = _run(env3)
    log3 = P3["log"].read_text()
    assert r3.returncode == 0, r3.stderr + log3
    assert f"store authority: not every component is certified ({_summary(strict)}); WINDOW_REQUIRE_CERTIFIED is unset, the window goes on" in log3
    assert "base at deadbee" in log3 and "=== ff chain end" in log3


def test_a_verify_authority_that_times_out_or_fails_is_logged_by_name_and_the_strict_check_decides(tmp_path):
    """verify-authority is bounded per component (WINDOW_AUTHORITY_WAIT_S) and its exit is logged by name; the decision is the strict
    check's, not the verify's: a component the verify left certified passes, one it did not (the shim's rc 1, the strict doc still naming
    it) is the strict check's problem."""
    first = _authority_doc({"a_unsealed": {"current": "aaaa1111", "verified": True, "sealed": False, "published_record": True, "problems": ["unsealed"]}})
    second = _authority_doc({"a_unsealed": {"current": "aaaa1111", "verified": True, "sealed": False, "published_record": True, "problems": ["unsealed"]}}, strict=True)
    env, P = _rehearsal(tmp_path, active=[], pre_installed=OLD_TIMERS, extra_env={"WINDOW_REQUIRE_CERTIFIED": "1", "SHIM_VERIFY_AUTHORITY_RC": "1", "WINDOW_AUTHORITY_WAIT_S": "7"})
    _authority_side_files(tmp_path, first, second)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 27, r.stderr + log
    assert "store authority: verify-authority a_unsealed rc=1" in log and f"=== ff chain ABORTED (store authority: not every component is certified ({_summary(second)}))" in log
    assert sum(1 for c in _calls(P) if "stores verify-authority" in c) == 1


# --------------------------------------------------------------------------------------------
# the WAL checkpoint after the drain, before the base moves; never a reason to fail the window
# --------------------------------------------------------------------------------------------


def test_the_wal_checkpoint_runs_after_the_drain_before_the_base_moves_and_never_fails_the_window(tmp_path):
    """Right after `writers quiet` -- the one moment with no writer -- and before the base fast-forwards, the window runs
    `PRAGMA wal_checkpoint(TRUNCATE)` through the registry python with a bounded busy timeout, stamps `-- wal checkpoint`, and logs ONE
    result line (the three-number result and the WAL bytes before / after) without the stage prefix, so the measurement's stage table is
    untouched. The shim answers the call with nothing (its catch-all branch): the result line says so and the window goes on. A python that
    FAILS the checkpoint (a reader holding the WAL, a torn file) fails nothing: the line names the failure and the window lands."""
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    stamps = _stamps(log)
    assert stamps.index("writers quiet") < stamps.index("-- wal checkpoint") < stamps.index("base at deadbee") < stamps.index("-- install units")
    lines = log.splitlines()
    result = [ln for ln in lines if ln.startswith("wal checkpoint: ")]
    assert len(result) == 1 and re.search(r"wal checkpoint: .*; wal bytes before=\d+ after=\d+ \d\d:\d\d:\d\d$", result[0]), result
    assert lines.index(result[0]) > next(i for i, ln in enumerate(lines) if ln.startswith("-- wal checkpoint "))
    assert lines.index(result[0]) < next(i for i, ln in enumerate(lines) if ln.startswith("base at "))
    calls = P["calls"].read_text().splitlines()
    ckpt = [c for c in calls if "PRAGMA wal_checkpoint" in c]        # the PRAGMA, not the test's own tmp path (its name carries the words)
    assert len(ckpt) == 1 and ckpt[0].startswith("python -c ") and "TRUNCATE" in ckpt[0] and "busy_timeout" in ckpt[0], ckpt
    # the checkpoint call comes after the last systemctl is-active probe of the drain and before the base's git fetch
    assert calls.index(ckpt[0]) < calls.index(next(c for c in calls if c.startswith("git") and "fetch" in c))
    # a failing checkpoint: the line names it, the window lands all the same
    env2, P2 = _rehearsal(tmp_path / "two", active=["registry-harvest.timer"], pre_installed=OLD_TIMERS, fail_on="*PRAGMA*wal_checkpoint*")     # the PRAGMA call alone (the tmp path carries the test's name; no space: the shim's case glob)
    r2 = _run(env2)
    log2 = P2["log"].read_text()
    assert r2.returncode == 0, r2.stderr + log2
    result2 = [ln for ln in log2.splitlines() if ln.startswith("wal checkpoint: ")]
    assert len(result2) == 1 and "rc=1" in result2[0] and "never fails the window" in result2[0], result2
    assert "=== ff chain end" in log2


# --------------------------------------------------------------------------------------------
# the lifecycle restamp slot (WINDOW_RESTAMP_KEYS)
# --------------------------------------------------------------------------------------------


def test_the_restamp_slot_plans_then_applies_the_named_keys_after_the_reconcile_slot_and_before_the_cluster_stages_only_when_asked(tmp_path):
    """WINDOW_RESTAMP_KEYS=<keys file> runs `registry replica restamp-keys` TWICE on the live database under the fast-forwarded base --
    the PLAN (`-- restamp keys (plan)`), then the APPLY + VERIFY (`-- restamp keys (apply)`, `--apply`) -- after the dimensions load and the
    reconciliation slot, before the acceptance / cluster stages and the ingest, with the maintenance flag held; windows.py classes both
    stamps `restamp`. Unset: ONE line `restamp keys: skipped`, no stamp, no call. A plan the CLI refuses (an unknown key, a change outside
    the lifecycle class; exit 13) aborts the window with exit 28 BEFORE the apply -- nothing after it runs, the timers are restored, the end
    marker is ABORTED by name; a missing keys file aborts with exit 28 before any CLI call."""
    keys = tmp_path / "keys.json"
    keys.write_text('["alphasteer_f22_rebake:codeattack:0000000000000000:1111111111111111"]\n')
    env, P = _rehearsal(tmp_path, active=[], pre_installed=OLD_TIMERS, extra_env={"WINDOW_RESTAMP_KEYS": str(keys), "WINDOW_RECONCILE": "run-patch-extra"})
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    calls = _calls(P)
    restamp = [c for c in calls if "replica restamp-keys" in c]
    db = str(P["home"] / "registry" / "registry.sqlite")
    out = str(P["state"] / "windows" / "restamp-W1")
    assert restamp == [f"python -m sisteer.registry.cli replica restamp-keys --db {db} --keys {keys} --out {out}",
                       f"python -m sisteer.registry.cli replica restamp-keys --db {db} --keys {keys} --out {out} --apply"], restamp
    order = [next(i for i, c in enumerate(calls) if key in c) for key in ("dimensions load", "reconcile run-patch-extra --dry-run", "restamp-keys --db", "cli ingest")]
    assert order == sorted(order), order
    st = _stamps(log)
    assert st.index("-- reconcile run-patch-extra (dry run)") < st.index("-- restamp keys (plan)") < st.index("-- restamp keys (apply)") < st.index("--- ingest"), st
    assert re.search(r"^-- restamp keys \(plan\) \d\d:\d\d:\d\d \(" + re.escape(str(keys)) + " -> " + re.escape(out) + r"\)$", log, re.M)
    assert "restamp keys: applied and verified against the plan" in log and "restamp keys: skipped" not in log
    # the flag was held when the slot ran: it is written before the stop and removed by the trap, and the slot checks it
    assert "maint-at-stop=present" in calls and "maintenance flag removed" in log
    windows = _load("windows").read_logs([str(P["log"])])
    assert [s["class"] for s in windows[0]["stages"]].count("restamp") == 2
    assert [s["class"] for s in windows[0]["stages"]] == ["reservation", "wal", "config_convert", "authority", "base", "install", "install", "migrate_plan", "migrate",
                                                           "dims", "reconcile", "restamp", "restamp", "ingest", "ingest_end", "status_heavy"]
    # unset: one line, no stamp, no call
    env0, P0 = _rehearsal(tmp_path / "off", active=[], pre_installed=OLD_TIMERS)
    r0 = _run(env0)
    log0 = P0["log"].read_text()
    assert r0.returncode == 0 and "restamp keys: skipped (WINDOW_RESTAMP_KEYS unset)" in log0 and "-- restamp keys" not in log0
    assert not any("restamp-keys" in c for c in _calls(P0)) and "restamp" not in [s["class"] for s in _load("windows").read_logs([str(P0["log"])])[0]["stages"]]
    # a refused plan: exit 28 before the apply; nothing after it runs; the timers restored; ABORTED by name
    env1, P1 = _rehearsal(tmp_path / "refused", active=["registry-harvest.timer"], pre_installed=OLD_TIMERS, fail_on='*"replica restamp-keys"*',
                          extra_env={"WINDOW_RESTAMP_KEYS": str(keys)})
    r1 = _run(env1)
    log1 = P1["log"].read_text()
    assert r1.returncode == 28, r1.stderr + log1
    calls1 = _calls(P1)
    assert sum(1 for c in calls1 if "replica restamp-keys" in c) == 1 and not any("--apply" in c for c in calls1)
    assert not any(c.endswith("cli ingest --retry-held") or "adapters run" in c for c in calls1) and "systemctl --user start registry-harvest.timer" in calls1
    assert "restamp keys plan REFUSED rc=1" in log1 and "nothing applied; abort" in log1 and "=== ff chain ABORTED (restamp keys plan REFUSED rc=1" in log1
    assert "-- restamp keys (apply)" not in _stamps(log1) and "=== ff chain end" not in log1
    # the keys file missing: exit 28, no CLI call at all
    env2, P2 = _rehearsal(tmp_path / "nokeys", active=[], pre_installed=OLD_TIMERS, extra_env={"WINDOW_RESTAMP_KEYS": str(tmp_path / "absent.json")})
    r2 = _run(env2)
    log2 = P2["log"].read_text()
    assert r2.returncode == 28 and "restamp keys: no keys file at" in log2 and not any("restamp-keys" in c for c in _calls(P2))
    # with cluster adapters named, the slot precedes the first cluster stage
    env4, P4 = _cluster_rehearsal(tmp_path / "cluster")
    env4["WINDOW_RESTAMP_KEYS"] = str(keys)
    r4 = _run(env4)
    log4 = P4["log"].read_text()
    assert r4.returncode == 0, r4.stderr + log4
    st4 = _cluster_stamps(log4)
    assert st4.index("-- dimensions load") < st4.index("-- restamp keys (plan)") < st4.index("-- restamp keys (apply)") < st4.index("-- cluster reads inbox-merge"), st4


# --------------------------------------------------------------------------------------------
# a component only the NEW configuration declares is carried under the worktree's code, in-window
# --------------------------------------------------------------------------------------------


def test_a_component_only_the_new_configuration_declares_is_carried_under_the_worktrees_code_and_certified_before_the_strict_check(tmp_path):
    """A census landing declared five new grading_host_copy components; `-- store mirror` only started the BASE's unit, whose
    code and configuration cannot see them, `-- store authority` named them `never_published ... not acted on here (the mirror timer carries
    them)`, and under WINDOW_REQUIRE_CERTIFIED=1 the strict check aborted the window (exit 27) BEFORE the fast-forward the timer needed --
    a pre-fast-forward step must run under the NEW tree. Now: a never-published grading_host_copy component is CARRIED under the
    worktree's CLI (`stores mirror --dest <dest> --component <slug>`, bounded by WINDOW_AUTHORITY_WAIT_S) and then verify-authority'd, all
    before the strict check, which then certifies it and the window goes on to `base at`; a carry that FAILS is logged by name, nothing is
    verified for it, the strict check names it and the window aborts with exit 27 by name as before. With a cluster set, `-- store mirror`
    follows the base unit's quiesce with the worktree's `stores mirror --dest <dest>` over every component the new configuration names."""
    first = _authority_doc({"n_new": {"current": None, "exists": False, "notes": ["never_published"]},
                            "f_ok": {"current": "ffff6666", "verified": True, "sealed": True, "published_record": True}})
    second = _authority_doc({"n_new": {"current": "cccc3333", "verified": True, "sealed": True, "published_record": True},
                             "f_ok": {"current": "ffff6666", "verified": True, "sealed": True, "published_record": True}}, strict=True)
    env, P = _rehearsal(tmp_path, active=["registry-harvest.timer", "registry-store-mirror.timer"], pre_installed=OLD_TIMERS + NEW_TIMERS,
                        extra_env={"WINDOW_REQUIRE_CERTIFIED": "1"})
    _authority_side_files(tmp_path, first, second)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    calls = _calls(P)
    adir = f"{P['state']}/windows/authority-W1"
    newreg = f"python -m sisteer.registry.cli --root {P['home']}/registry"
    checks = [c for c in calls if "stores check-authority" in c]
    carries = [c for c in calls if "stores mirror" in c]
    verifies = [c for c in calls if "stores verify-authority" in c]
    # the commands, in order: the check, the carry under the worktree's code, its verify-authority, the strict check -- all before the fast-forward
    assert carries == [f"{newreg} stores mirror --dest {SUB_A} --component n_new --bound-s 1800"], carries
    assert verifies == [f"{newreg} stores verify-authority --dest {SUB_A} --component n_new --out {adir} --json"], verifies
    ff = next(i for i, c in enumerate(calls) if c.startswith("git ") and "merge" in c)
    assert calls.index(checks[0]) < calls.index(carries[0]) < calls.index(verifies[0]) < calls.index(checks[1]) < ff
    assert ("store authority: check-authority rc=0; never published (carried under the worktree's code, then verify-authority): n_new; "
            "verified but unsealed/unrecorded (verify-authority runs): none; not acted on here (the mirror timer carries them): none") in log, log
    assert "store authority: carry n_new rc=0 (never published; carried under the worktree's code)" in log and "store authority: verify-authority n_new rc=0" in log
    assert f"store authority: strict check-authority rc=0: {_summary(second)}" in log and "base at deadbee" in log and "=== ff chain end" in log
    assert (P["home"] / ".local" / "state" / "registry-landing" / "windows" / "authority-W1" / "to-carry.txt").read_text().strip() == "n_new"
    # no cluster set here: the `-- store mirror` step (and its new-tree carry) did not run; the authority stage alone carried the component
    assert "-- store mirror" not in _stamps(log) and not any(f"stores mirror --dest {SUB_A}" in c and "--component" not in c for c in calls)
    # the negative: the carry FAILS (the shim's rc 1) -- logged by name, nothing verified for it, the strict check names it, exit 27 BEFORE the base moves
    strict_fail = _authority_doc({"n_new": {"current": None, "exists": False, "problems": ["not_certified_never_published"]},
                                  "f_ok": {"current": "ffff6666", "verified": True, "sealed": True, "published_record": True}}, strict=True)
    env2, P2 = _rehearsal(tmp_path / "neg", active=["registry-harvest.timer", "registry-store-mirror.timer"], pre_installed=OLD_TIMERS + NEW_TIMERS,
                          extra_env={"WINDOW_REQUIRE_CERTIFIED": "1"})
    _authority_side_files(tmp_path / "neg", first, strict_fail)
    (tmp_path / "neg" / "shim-mirror-rc").write_text("1\n")
    r2 = _run(env2)
    log2 = P2["log"].read_text()
    assert r2.returncode == 27, r2.stderr + log2
    calls2 = _calls(P2)
    newreg2 = f"python -m sisteer.registry.cli --root {P2['home']}/registry"
    assert [c for c in calls2 if "stores mirror" in c] == [f"{newreg2} stores mirror --dest {SUB_A} --component n_new --bound-s 1800"]
    assert not any("stores verify-authority" in c for c in calls2), "a component whose carry failed has nothing to verify"
    assert "store authority: carry n_new FAILED rc=1 (never published; nothing to verify -- the strict check names it)" in log2
    assert f"=== ff chain ABORTED (store authority: not every component is certified ({_summary(strict_fail)}))" in log2 and "n_new: not_certified_never_published" in log2
    assert "base at" not in log2 and "=== ff chain end" not in log2 and not any(c.startswith("git ") and "merge" in c for c in calls2)
    assert "systemctl --user start registry-harvest.timer registry-store-mirror.timer" in _starts(calls2) and not P2["maint"].exists()
    # with a cluster set, `-- store mirror` quiesces the base unit's pass and THEN carries under the worktree's code, over every declared component
    env4, P4 = _cluster_rehearsal(tmp_path / "cluster")
    r4 = _run(env4)
    log4 = P4["log"].read_text()
    assert r4.returncode == 0, r4.stderr + log4
    calls4 = _calls(P4)
    unit_start = calls4.index(next(c for c in calls4 if "start --no-block registry-store-mirror.service" in c))
    carry_all = [i for i, c in enumerate(calls4) if f"stores mirror --dest {SUB_A}" in c and "--component" not in c]
    first_check = calls4.index(next(c for c in calls4 if "stores check-authority" in c))
    assert len(carry_all) == 1 and unit_start < carry_all[0] < first_check, (unit_start, carry_all, first_check)
    assert calls4[carry_all[0]] == f"python -m sisteer.registry.cli --root {P4['home']}/registry stores mirror --dest {SUB_A} --bound-s 0"
    assert f"store mirror: carry under the worktree's code (stores mirror --dest {SUB_A} --bound-s 0 under" in log4 and "store mirror: new-tree carry rc=0" in log4
    assert not any(c.startswith("timeout ") for c in calls4), "the bound is the controller's (--bound-s), never an outer timeout"
    assert (P4["state"] / "windows" / "store-mirror-W1.jsonl").exists()


# --------------------------------------------------------------------------------------------
# the multi-turn-scope fixed point: verify to a fixed point inside the window
# --------------------------------------------------------------------------------------------

_VERIFY_LINE = "  camp:run:{} (response, every_turn_prefill_and_decode) -> final_header_and_current_decode_only [lora]"


def _verify_fixture(P, n, run_keys):
    """What the registry python shim prints and exits for its n-th `reconcile multi-turn-scope --verify`: the CLI's own shape (one line per
    inconsistent run, the count last) and exit 3 when there are any."""
    (P["tmp"] / f"shim-verify-{n}.out").write_text("multi-turn-scope verify (shim)\n" + "".join(_VERIFY_LINE.format(k) + "\n" for k in run_keys)
                                                  + f"inconsistent {len(run_keys)} run(s)\n")
    (P["tmp"] / f"shim-verify-{n}.rc").write_text("3\n" if run_keys else "0\n")


def test_the_multi_turn_scope_fixed_point_verifies_after_the_ingest_reconciles_once_more_and_fails_by_name_when_it_does_not_converge(tmp_path):
    """With WINDOW_RECONCILE naming multi-turn-scope and WINDOW_RECONCILE_APPLY=1, the verify runs after `--- ingest end`
    (`-- reconcile multi-turn-scope (verify 1)`, the whole text in the log and in scope-verify-<window>-1.txt). Reading zero (exit 0) the
    window goes on: ONE ingest, no second apply, the end marker. Reading runs (exit 3, the ids on its lines -- the race the HIGH names) it
    reconciles ONCE more (`(apply, second pass)`), ingests a second time (`(ingest)`: the one case of two ingests) and verifies again
    (`(verify 2)`); zero passes. A fixture that keeps re-publishing (verify 2 still lists a run) fails the window BY NAME: exit 29, the ids
    in the abort line and in the verify's text above it, the timers restored, no end marker. A verify that itself fails is 17; without the
    apply asked no verify runs. windows.py classes every stamp of the fixed point `reconcile`."""
    scope_env = {"WINDOW_RECONCILE": "multi-turn-scope", "WINDOW_RECONCILE_APPLY": "multi-turn-scope"}
    # (a) consistent after the one ingest: verify 1 reads zero -> one ingest, one apply, one verify, the end marker
    env, P = _rehearsal(tmp_path / "zero", active=[], pre_installed=OLD_TIMERS, extra_env=scope_env)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    calls = _calls(P)
    rc_calls = [c for c in calls if "cli reconcile" in c]
    assert rc_calls == ["python -m sisteer.registry.cli reconcile multi-turn-scope --dry-run", "python -m sisteer.registry.cli reconcile multi-turn-scope --apply",
                        "python -m sisteer.registry.cli reconcile multi-turn-scope --verify"], rc_calls
    assert sum(1 for c in calls if c.endswith("cli ingest --retry-held")) == 1
    st = _stamps(log)
    assert st.index("--- ingest end") < st.index("-- reconcile multi-turn-scope (verify 1)") < st.index("=== ff chain end"), st
    assert "-- reconcile multi-turn-scope (apply, second pass)" not in st and "-- reconcile multi-turn-scope (verify 2)" not in st and "-- reconcile multi-turn-scope (ingest)" not in st
    assert "inconsistent 0 run(s); the class is consistent" in log
    assert (P["state"] / "windows" / "scope-verify-W1-1.txt").read_text().strip() == "inconsistent 0 run(s); the class is consistent"
    assert not (P["state"] / "windows" / "scope-verify-W1-2.txt").exists()
    windows = _load("windows").read_logs([str(P["log"])])
    assert windows[0]["completed"] is True
    assert [s["class"] for s in windows[0]["stages"]] == ["reservation", "wal", "config_convert", "authority", "base", "install", "install", "migrate_plan", "migrate", "dims",
                                                            "reconcile", "reconcile", "ingest", "ingest_end", "reconcile", "status_heavy"]
    # (b) the race: verify 1 lists two runs -> a second apply, a second ingest, verify 2 reads zero -> the window passes, the ids in the log
    env2, P2 = _rehearsal(tmp_path / "converge", active=[], pre_installed=OLD_TIMERS, extra_env=scope_env)
    _verify_fixture(P2, 1, ["a", "b"])
    r2 = _run(env2)
    log2 = P2["log"].read_text()
    assert r2.returncode == 0, r2.stderr + log2
    seq = [c for c in _calls(P2) if "cli reconcile" in c or c.endswith("cli ingest --retry-held")]
    assert seq == ["python -m sisteer.registry.cli reconcile multi-turn-scope --dry-run", "python -m sisteer.registry.cli reconcile multi-turn-scope --apply",
                   "python -m sisteer.registry.cli ingest --retry-held", "python -m sisteer.registry.cli reconcile multi-turn-scope --verify",
                   "python -m sisteer.registry.cli reconcile multi-turn-scope --apply", "python -m sisteer.registry.cli ingest --retry-held",
                   "python -m sisteer.registry.cli reconcile multi-turn-scope --verify"], seq
    st2 = _stamps(log2)
    assert st2[st2.index("--- ingest end"):][:5] == ["--- ingest end", "-- reconcile multi-turn-scope (verify 1)", "-- reconcile multi-turn-scope (apply, second pass)",
                                                     "-- reconcile multi-turn-scope (ingest)", "-- reconcile multi-turn-scope (verify 2)"], st2
    assert _VERIFY_LINE.format("a") in log2 and _VERIFY_LINE.format("b") in log2 and "inconsistent 2 run(s)" in log2 and "=== ff chain end" in log2
    assert (P2["state"] / "windows" / "scope-verify-W1-1.txt").read_text().count("camp:run:") == 2
    assert (P2["state"] / "windows" / "scope-verify-W1-2.txt").read_text().strip() == "inconsistent 0 run(s); the class is consistent"
    windows2 = _load("windows").read_logs([str(P2["log"])])
    assert windows2[0]["completed"] is True
    assert [s["class"] for s in windows2[0]["stages"]][-7:] == ["ingest", "ingest_end", "reconcile", "reconcile", "reconcile", "reconcile", "status_heavy"]
    # (c) no fixed point: a fixture that keeps re-publishing -- verify 2 still lists a run -> exit 29 BY NAME, the timers restored, no end marker
    env3, P3 = _rehearsal(tmp_path / "diverge", active=["registry-harvest.timer"], pre_installed=OLD_TIMERS, extra_env=scope_env)
    _verify_fixture(P3, 1, ["a", "b"])
    _verify_fixture(P3, 2, ["a"])
    r3 = _run(env3)
    log3 = P3["log"].read_text()
    assert r3.returncode == 29, r3.stderr + log3
    assert ("reconcile multi-turn-scope reached no fixed point: after the apply, the ingest, a second apply and a second ingest the verify still lists "
            "inconsistent run(s): camp:run:a; abort") in log3
    marker = [ln for ln in log3.splitlines() if ln.startswith("=== ff chain ABORTED")]
    assert len(marker) == 1 and marker[0].startswith("=== ff chain ABORTED (reconcile multi-turn-scope reached no fixed point") and "camp:run:a" in marker[0]
    assert "=== ff chain end" not in log3 and "systemctl --user start registry-harvest.timer" in _calls(P3) and "window ended early (rc=29)" in log3
    calls3 = _calls(P3)
    assert sum(1 for c in calls3 if c.endswith("cli ingest --retry-held")) == 2 and sum(1 for c in calls3 if c.endswith("multi-turn-scope --verify")) == 2
    assert sum(1 for c in calls3 if c.endswith("multi-turn-scope --apply")) == 2
    assert log3.count(_VERIFY_LINE.format("a")) == 2                        # both verifies' whole text is in the log: nothing hidden
    windows3 = _load("windows").read_logs([str(P3["log"])])
    assert windows3[0]["completed"] is False and windows3[0]["abort_cause"].startswith("reconcile multi-turn-scope reached no fixed point") and "camp:run:a" in windows3[0]["abort_cause"]
    # (d) a verify that itself fails is 17, named
    env4, P4 = _rehearsal(tmp_path / "broken", active=[], pre_installed=OLD_TIMERS, fail_on='*"multi-turn-scope --verify"', extra_env=scope_env)
    r4 = _run(env4)
    log4 = P4["log"].read_text()
    assert r4.returncode == 17 and "reconcile multi-turn-scope verify FAILED rc=1; abort" in log4 and "=== ff chain end" not in log4, r4.stderr + log4
    # (e) a second-pass apply that holds a run (exit 2) is 17 by name, and no second ingest runs
    env5, P5 = _rehearsal(tmp_path / "held", active=[], pre_installed=OLD_TIMERS, extra_env=scope_env)
    _verify_fixture(P5, 1, ["a"])
    _shim(P5["tmp"] / "bin", "regpy", open(P5["tmp"] / "bin" / "regpy").read().split("\n", 1)[1].replace(
        'case "$*" in\n', 'case "$*" in\n  *"multi-turn-scope --apply"*) [ -f "{0}/shim-applied" ] && {{ echo "1 run(s) held" >&2; exit 2; }}; : > "{0}/shim-applied" ;;\n'.format(P5["tmp"]), 1))
    r5 = _run(env5)
    log5 = P5["log"].read_text()
    assert r5.returncode == 17, r5.stderr + log5
    assert "reconcile multi-turn-scope second-pass apply FAILED rc=2 (2 = a run held behind a record not folded yet; nothing written for it); abort" in log5
    assert sum(1 for c in _calls(P5) if c.endswith("cli ingest --retry-held")) == 1 and "-- reconcile multi-turn-scope (verify 2)" not in _stamps(log5)
    # (f) without the apply asked no verify runs: the dry run alone, as before
    env6, P6 = _rehearsal(tmp_path / "dry", active=[], pre_installed=OLD_TIMERS, extra_env={"WINDOW_RECONCILE": "multi-turn-scope"})
    r6 = _run(env6)
    log6 = P6["log"].read_text()
    assert r6.returncode == 0, r6.stderr + log6
    assert [c for c in _calls(P6) if "cli reconcile" in c] == ["python -m sisteer.registry.cli reconcile multi-turn-scope --dry-run"]
    assert "(verify 1)" not in log6 and "=== ff chain end" in log6


# --------------------------------------------------------------------------------------------
# the per-action apply list: WINDOW_RECONCILE_APPLY names the actions to apply; the retired global `1` is refused by name before the lock
# --------------------------------------------------------------------------------------------


def test_the_apply_list_is_per_action_and_the_retired_global_switch_or_an_unnamed_action_is_refused_before_the_lock(tmp_path):
    """WINDOW_RECONCILE names four actions in order; WINDOW_RECONCILE_APPLY lists two of them: the two run dry run THEN apply, the other two
    the dry run alone (run-patch-extra's release gate is never asked for an apply beside a correction -- the configuration that aborted a
    window after its migration had applied). `1` (the retired global switch) and a listed name WINDOW_RECONCILE lacks are refused BY NAME
    with guidance, exit 30, before the lock and the reservation: one line in the log, no start marker, no shim called, no timer touched."""
    four = "run-patch-extra multi-turn-scope env-comments fit-effective-batch"
    env, P = _rehearsal(tmp_path, active=[], pre_installed=OLD_TIMERS, extra_env={"WINDOW_RECONCILE": four, "WINDOW_RECONCILE_APPLY": "env-comments fit-effective-batch"})
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    rc_calls = [c for c in _calls(P) if "cli reconcile" in c]
    assert rc_calls == ["python -m sisteer.registry.cli reconcile run-patch-extra --dry-run",
                        "python -m sisteer.registry.cli reconcile multi-turn-scope --dry-run",
                        "python -m sisteer.registry.cli reconcile env-comments --dry-run", "python -m sisteer.registry.cli reconcile env-comments --apply",
                        "python -m sisteer.registry.cli reconcile fit-effective-batch --dry-run", "python -m sisteer.registry.cli reconcile fit-effective-batch --apply",
                        "python -m sisteer.registry.cli reconcile env-comments --verify", "python -m sisteer.registry.cli reconcile fit-effective-batch --verify"], rc_calls
    st = _stamps(log)
    slot = [x for x in st[st.index("-- dimensions load") + 1:st.index("--- ingest")] if x.startswith("-- reconcile")]
    assert slot == ["-- reconcile run-patch-extra (dry run)", "-- reconcile multi-turn-scope (dry run)",
                    "-- reconcile env-comments (dry run)", "-- reconcile env-comments (apply)",
                    "-- reconcile fit-effective-batch (dry run)", "-- reconcile fit-effective-batch (apply)"], st
    assert "-- reconcile multi-turn-scope (apply)" not in st and "-- reconcile run-patch-extra (apply)" not in st and "-- reconcile multi-turn-scope (verify 1)" not in st
    after = st[st.index("--- ingest end"):st.index("=== ff chain end")]
    assert [x for x in after if x.startswith("-- reconcile")] == ["-- reconcile env-comments (verify 1)", "-- reconcile fit-effective-batch (verify 1)"], after
    assert (P["state"] / "windows" / "verify-env-comments-W1-1.txt").read_text().strip() == "inconsistent 0 run(s); the class is consistent"
    assert (P["state"] / "windows" / "verify-fit-effective-batch-W1-1.txt").exists() and not (P["state"] / "windows" / "scope-verify-W1-1.txt").exists()
    windows = _load("windows").read_logs([str(P["log"])])
    assert windows[0]["completed"] is True
    assert [s_["class"] for s_ in windows[0]["stages"]][-9:] == ["reconcile"] * 4 + ["ingest", "ingest_end", "reconcile", "reconcile", "status_heavy"]
    # the old OFF switch still means "apply nothing": every named action a dry run, no refusal
    env0, P0 = _rehearsal(tmp_path / "off", active=[], pre_installed=OLD_TIMERS, extra_env={"WINDOW_RECONCILE": "multi-turn-scope env-comments", "WINDOW_RECONCILE_APPLY": "0"})
    r0 = _run(env0)
    assert r0.returncode == 0, r0.stderr + P0["log"].read_text()
    assert [c for c in _calls(P0) if "cli reconcile" in c] == ["python -m sisteer.registry.cli reconcile multi-turn-scope --dry-run",
                                                               "python -m sisteer.registry.cli reconcile env-comments --dry-run"]
    # the retired global switch: refused by name before anything moves
    for bad, needle in (("1", "the retired global switch"), ("multi-turn-scope env-comments bogus-action", "names 'bogus-action', which WINDOW_RECONCILE")):
        env2, P2 = _rehearsal(tmp_path / f"refused-{bad[:1]}", active=["registry-harvest.timer"], pre_installed=OLD_TIMERS,
                              extra_env={"WINDOW_RECONCILE": "multi-turn-scope env-comments", "WINDOW_RECONCILE_APPLY": bad})
        r2 = _run(env2)
        log2 = P2["log"].read_text()
        assert r2.returncode == 30, (bad, r2.stderr, log2)
        assert "window W1 refused: WINDOW_RECONCILE_APPLY" in log2 and needle in log2 and "nothing reserved, stopped, fast-forwarded or installed" in log2
        assert "=== ff chain" not in log2 and not P2["calls"].exists(), log2
        if bad == "1":
            assert 'e.g. WINDOW_RECONCILE_APPLY="multi-turn-scope env-comments"' in log2        # the guidance names the list the operator should write


def test_a_correction_actions_fixed_point_runs_per_action_with_its_own_verify_file_and_fails_by_name_with_the_action(tmp_path):
    """env-comments listed for apply alone (multi-turn-scope named, dry run only): after the ingest only env-comments verifies; a verify 1
    that lists runs makes the second pass for THAT action (`(apply, second pass)`, `(ingest)`, `(verify 2)`), its files named
    verify-env-comments-<window>-<n>.txt; verify 2 still listing a run fails by name with the action (exit 29)."""
    env, P = _rehearsal(tmp_path / "converge", active=[], pre_installed=OLD_TIMERS,
                        extra_env={"WINDOW_RECONCILE": "multi-turn-scope env-comments", "WINDOW_RECONCILE_APPLY": "env-comments"})
    _verify_fixture(P, 1, ["a", "b"])
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    seq = [c for c in _calls(P) if "cli reconcile" in c or c.endswith("cli ingest --retry-held")]
    assert seq == ["python -m sisteer.registry.cli reconcile multi-turn-scope --dry-run",
                   "python -m sisteer.registry.cli reconcile env-comments --dry-run", "python -m sisteer.registry.cli reconcile env-comments --apply",
                   "python -m sisteer.registry.cli ingest --retry-held", "python -m sisteer.registry.cli reconcile env-comments --verify",
                   "python -m sisteer.registry.cli reconcile env-comments --apply", "python -m sisteer.registry.cli ingest --retry-held",
                   "python -m sisteer.registry.cli reconcile env-comments --verify"], seq
    st = _stamps(log)
    assert st[st.index("--- ingest end"):][:5] == ["--- ingest end", "-- reconcile env-comments (verify 1)", "-- reconcile env-comments (apply, second pass)",
                                                   "-- reconcile env-comments (ingest)", "-- reconcile env-comments (verify 2)"], st
    assert (P["state"] / "windows" / "verify-env-comments-W1-1.txt").read_text().count("camp:run:") == 2
    assert (P["state"] / "windows" / "verify-env-comments-W1-2.txt").read_text().strip() == "inconsistent 0 run(s); the class is consistent"
    assert "(verify 1)" in log and "multi-turn-scope (verify" not in log
    env3, P3 = _rehearsal(tmp_path / "diverge", active=["registry-harvest.timer"], pre_installed=OLD_TIMERS,
                          extra_env={"WINDOW_RECONCILE": "env-comments", "WINDOW_RECONCILE_APPLY": "env-comments"})
    _verify_fixture(P3, 1, ["a"])
    _verify_fixture(P3, 2, ["a"])
    r3 = _run(env3)
    log3 = P3["log"].read_text()
    assert r3.returncode == 29, r3.stderr + log3
    assert "reconcile env-comments reached no fixed point: after the apply, the ingest, a second apply and a second ingest the verify still lists inconsistent run(s): camp:run:a; abort" in log3
    assert "=== ff chain ABORTED (reconcile env-comments reached no fixed point" in log3 and "systemctl --user start registry-harvest.timer" in _calls(P3)


def test_the_env_comments_sources_knob_rides_every_env_comments_and_run_patch_extra_invocation_and_a_missing_path_is_refused_before_the_lock(tmp_path):
    """WINDOW_ENV_COMMENTS_SOURCES names env-comments' overrides files (a file, a directory): each rides as `--source` on the action's dry
    run, apply and verify -- the pass restates a stamp from its file, matched by sha256, and its verify needs the same files -- and on
    run-patch-extra's dry run, whose comparison of a legacy run_overrides stamp with the live one reads the same files (without them a
    stamp env-comments restated would read as a conflict against its legacy record); on no other action's call. A named path that is not
    there is refused by name before the lock (exit 30): one line, no start marker, no shim called."""
    src_a = tmp_path / "pipeline.env"
    src_a.write_text("OVERRIDE_ID=pipeline_v1\n")
    src_dir = tmp_path / "overrides"
    src_dir.mkdir()
    env, P = _rehearsal(tmp_path / "sources", active=[], pre_installed=OLD_TIMERS,
                        extra_env={"WINDOW_RECONCILE": "run-patch-extra multi-turn-scope env-comments", "WINDOW_RECONCILE_APPLY": "env-comments",
                                   "WINDOW_ENV_COMMENTS_SOURCES": f"{src_a} {src_dir}"})
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    srcs = f"--source {src_a} --source {src_dir}"
    assert [c for c in _calls(P) if "cli reconcile" in c] == [f"python -m sisteer.registry.cli reconcile run-patch-extra {srcs} --dry-run",
                                                              "python -m sisteer.registry.cli reconcile multi-turn-scope --dry-run",
                                                              f"python -m sisteer.registry.cli reconcile env-comments {srcs} --dry-run",
                                                              f"python -m sisteer.registry.cli reconcile env-comments {srcs} --apply",
                                                              f"python -m sisteer.registry.cli reconcile env-comments {srcs} --verify"]
    assert (P["state"] / "windows" / "verify-env-comments-W1-1.txt").read_text().strip() == "inconsistent 0 run(s); the class is consistent"
    env2, P2 = _rehearsal(tmp_path / "missing", active=["registry-harvest.timer"], pre_installed=OLD_TIMERS,
                          extra_env={"WINDOW_RECONCILE": "env-comments", "WINDOW_RECONCILE_APPLY": "env-comments",
                                     "WINDOW_ENV_COMMENTS_SOURCES": f"{src_a} {tmp_path / 'nowhere.env'}"})
    r2 = _run(env2)
    log2 = P2["log"].read_text()
    assert r2.returncode == 30, r2.stderr + log2
    assert f"window W1 refused: WINDOW_ENV_COMMENTS_SOURCES names '{tmp_path / 'nowhere.env'}', which is not a file or a directory here" in log2
    assert "nothing reserved, stopped, fast-forwarded or installed" in log2 and "=== ff chain" not in log2 and not P2["calls"].exists(), log2


def test_env_comments_listed_for_apply_has_its_sources_coverage_checked_before_the_lock_and_an_uncovered_stamp_refuses_the_window_by_name(tmp_path):
    """When env-comments is listed for apply, `registry reconcile env-comments --check-sources` runs under the WORKTREE's CLI before the
    lock and the reservation, with the knob's sources: a source list that covers no file for a stamp whose values may carry a comment --
    the knob unset, or an existing directory without the files the stamps were read from -- is refused by name (exit 30): one line naming
    the knob and the way out, the check's whole text in env-comments-sources-<window>.txt, no start marker, the check the ONLY call (no
    reservation, no timer stopped). A check that reads clean is one line and the window runs, the check the first registry call, before the
    reservation. WINDOW_ENV_COMMENTS_LEAVE_UNSOURCED=1 is the operator's word: no check, and `--leave-unsourced` rides env-comments' dry
    run, apply and verify (not run-patch-extra's); any other value is refused by name. An env-comments named but not listed for apply is
    not checked (it applies nothing)."""
    empty = tmp_path / "empty_overrides"
    empty.mkdir()
    listing = ("env-comments sources (read-only, the folded run table): every stamped run row whose values may carry a comment has its overrides file among the sources\n"
               "sources: 1 file(s) by digest\nstamped run rows 1365; with a source 0\n"
               "uncovered (no file given for this stamp; values that may carry a comment: OVERRIDE_EFFECTIVE_BATCH, OVERRIDE_LORA_RANK): pipeline_v1 c03ba00f 528 run(s)\n"
               "  camp:run:a unsourced: no file given for this stamp (pipeline_v1 c03ba00f); values that may carry a comment: OVERRIDE_EFFECTIVE_BATCH, OVERRIDE_LORA_RANK\n"
               "uncovered 528 run(s); name their files with --source, or leave them as stored by name with --leave-unsourced\n")
    for name, sources in (("empty-dir", str(empty)), ("unset", None)):
        extra = {"WINDOW_RECONCILE": "run-patch-extra env-comments", "WINDOW_RECONCILE_APPLY": "env-comments"}
        if sources is not None:
            extra["WINDOW_ENV_COMMENTS_SOURCES"] = sources
        env, P = _rehearsal(tmp_path / name, active=["registry-harvest.timer"], pre_installed=OLD_TIMERS, extra_env=extra)
        (P["tmp"] / "shim-check-sources.out").write_text(listing)
        (P["tmp"] / "shim-check-sources.rc").write_text("3\n")
        r = _run(env)
        log = P["log"].read_text()
        assert r.returncode == 30, (name, r.stderr, log)
        refused = [ln for ln in log.splitlines() if ln.startswith("window W1 refused:")]
        assert len(refused) == 1 and f'WINDOW_ENV_COMMENTS_SOURCES="{sources or ""}" covers no file for run-overrides stamps whose values may carry a comment' in refused[0], log
        assert "(uncovered 528 run(s); name their files with --source, or leave them as stored by name with --leave-unsourced)" in refused[0]
        assert "env-comments is listed for apply and would restate nothing for them" in refused[0] and "WINDOW_ENV_COMMENTS_LEAVE_UNSOURCED=1 to leave them as stored by name" in refused[0]
        assert "nothing reserved, stopped, fast-forwarded or installed" in refused[0] and "=== ff chain" not in log, log
        check_file = P["state"] / "windows" / "env-comments-sources-W1.txt"
        assert check_file.read_text() == listing and str(check_file) in refused[0]
        # the check is the only call: under the worktree's CLI, with the knob's sources (none when unset), nothing else ran
        calls = _calls(P)
        want = "python -m sisteer.registry.cli --root " + str(P["home"] / "registry") + " reconcile env-comments " + (f"--source {sources} " if sources else "") + "--check-sources"
        assert calls == [want], (calls, want)
        assert P["pypaths"].read_text().splitlines()[0].startswith(str(P["home"] / "worktree" / "src") + " "), P["pypaths"].read_text()
    # a check that could not run (rc 1) refuses the same way, by name, with its rc
    env1, P1 = _rehearsal(tmp_path / "broken", active=["registry-harvest.timer"], pre_installed=OLD_TIMERS,
                          extra_env={"WINDOW_RECONCILE": "env-comments", "WINDOW_RECONCILE_APPLY": "env-comments", "WINDOW_ENV_COMMENTS_SOURCES": str(empty)})
    (P1["tmp"] / "shim-check-sources.out").write_text("refused: no database at the root\n")
    (P1["tmp"] / "shim-check-sources.rc").write_text("1\n")
    r1 = _run(env1)
    log1 = P1["log"].read_text()
    assert r1.returncode == 30 and "env-comments' sources check could not run (registry reconcile env-comments --check-sources rc=1" in log1 and "=== ff chain" not in log1, log1
    assert len(_calls(P1)) == 1
    # a clean check: one line in the log with the check's last line, the window runs, the check before the reservation
    env2, P2 = _rehearsal(tmp_path / "covered", active=[], pre_installed=OLD_TIMERS,
                          extra_env={"WINDOW_RECONCILE": "run-patch-extra env-comments", "WINDOW_RECONCILE_APPLY": "env-comments", "WINDOW_ENV_COMMENTS_SOURCES": str(empty)})
    r2 = _run(env2)
    log2 = P2["log"].read_text()
    assert r2.returncode == 0, r2.stderr + log2
    assert "env-comments sources: uncovered 0 run(s); every stamp whose values may carry a comment has its file (the check's text in" in log2
    pys = [c for c in _calls(P2) if c.startswith("python")]
    assert pys[0].endswith(f"reconcile env-comments --source {empty} --check-sources") and "cluster-reads reserve" in pys[1], pys[:2]
    assert log2.index("env-comments sources:") < log2.index("=== ff chain start")
    assert [c for c in _calls(P2) if "cli reconcile" in c] == [f"python -m sisteer.registry.cli reconcile run-patch-extra --source {empty} --dry-run",
                                                               f"python -m sisteer.registry.cli reconcile env-comments --source {empty} --dry-run",
                                                               f"python -m sisteer.registry.cli reconcile env-comments --source {empty} --apply",
                                                               f"python -m sisteer.registry.cli reconcile env-comments --source {empty} --verify"]
    windows = _load("windows").read_logs([str(P2["log"])])
    assert len(windows) == 1 and windows[0]["completed"] is True
    # the operator's word: no check; --leave-unsourced on env-comments' every invocation and on no other action's
    env3, P3 = _rehearsal(tmp_path / "left", active=[], pre_installed=OLD_TIMERS,
                          extra_env={"WINDOW_RECONCILE": "run-patch-extra env-comments", "WINDOW_RECONCILE_APPLY": "env-comments",
                                     "WINDOW_ENV_COMMENTS_SOURCES": str(empty), "WINDOW_ENV_COMMENTS_LEAVE_UNSOURCED": "1"})
    (P3["tmp"] / "shim-check-sources.rc").write_text("3\n")             # would refuse, were it asked
    r3 = _run(env3)
    log3 = P3["log"].read_text()
    assert r3.returncode == 0, r3.stderr + log3
    assert not any("--check-sources" in c for c in _calls(P3)) and "env-comments sources:" not in log3
    assert [c for c in _calls(P3) if "cli reconcile" in c] == [f"python -m sisteer.registry.cli reconcile run-patch-extra --source {empty} --dry-run",
                                                               f"python -m sisteer.registry.cli reconcile env-comments --source {empty} --leave-unsourced --dry-run",
                                                               f"python -m sisteer.registry.cli reconcile env-comments --source {empty} --leave-unsourced --apply",
                                                               f"python -m sisteer.registry.cli reconcile env-comments --source {empty} --leave-unsourced --verify"]
    # env-comments named but not listed for apply: no check (it applies nothing); `0` reads as the default
    env4, P4 = _rehearsal(tmp_path / "dry", active=[], pre_installed=OLD_TIMERS,
                          extra_env={"WINDOW_RECONCILE": "env-comments", "WINDOW_RECONCILE_APPLY": "0", "WINDOW_ENV_COMMENTS_LEAVE_UNSOURCED": "0"})
    (P4["tmp"] / "shim-check-sources.rc").write_text("3\n")
    r4 = _run(env4)
    assert r4.returncode == 0, r4.stderr + P4["log"].read_text()
    assert not any("--check-sources" in c for c in _calls(P4)) and [c for c in _calls(P4) if "cli reconcile" in c] == ["python -m sisteer.registry.cli reconcile env-comments --dry-run"]
    # any other value of the word is refused by name before the lock
    env5, P5 = _rehearsal(tmp_path / "badword", active=["registry-harvest.timer"], pre_installed=OLD_TIMERS,
                          extra_env={"WINDOW_RECONCILE": "env-comments", "WINDOW_RECONCILE_APPLY": "env-comments", "WINDOW_ENV_COMMENTS_LEAVE_UNSOURCED": "yes"})
    r5 = _run(env5)
    log5 = P5["log"].read_text()
    assert r5.returncode == 30 and "window W1 refused: WINDOW_ENV_COMMENTS_LEAVE_UNSOURCED=yes is neither 1" in log5 and "=== ff chain" not in log5 and not P5["calls"].exists(), log5
