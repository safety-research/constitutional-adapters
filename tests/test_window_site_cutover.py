"""The site cutover inside a landing window (``WINDOW_SITE_CUTOVER``; scripts/registry/landing/window.sh and site_cutover.py), rehearsed
with the PATH-shim harness of tests/test_registry_window.py and invented old labels (``oldsite_a`` / ``oldsite_b``) that the prepared
alias table maps to the two registry labels.

The six rehearsals: an in-flight pull (a harvest pass holding harvester.lock and writing into the old inbox's receive directory is waited
out by the drain, and nothing is written under the old name after the rename), an interrupted delivery (refused by name, nothing touched,
the timers back on the old configuration), a mid-pull reads tick (the reservation waits for the critical section, and the cutover refuses
by name if it is still held), a non-terminal op record (refused by name), a mirrors pass mid-sync (stopped at the timer, waited out, and
confirmed terminal before the cutover) and a failure after the install (a planted preflight refusal: every file restored byte for byte, the
inboxes moved back, and only then the timers restored). Then a failure after the fast-forward (the units, the base checkout and the cutover
all put back before any timer restarts), the preflight alone (no lock, no reservation, nothing installed), and site_cutover.py by itself."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from test_landing_site_env import SUB_A, SUB_B, write_site
from test_registry_window import NEW_TIMERS, OLD_TIMERS, _calls, _rehearsal, _run, _shim, _stamps, _starts

REPO = Path(__file__).resolve().parents[1]
SITE_CUTOVER = REPO / "scripts" / "registry" / "landing" / "site_cutover.py"
SELECTOR = "sel_one"                 # the one cluster name the stub repo-mirror tool knows
SUCCESS_STAMPS = ["=== ff chain start ADAPTERS=", "-- reads reservation", "writers quiet", "-- wal checkpoint", "-- config convert", "-- site cutover",
                  "-- preflight", "-- store authority", "base at deadbee", "-- install units", "-- uninstall units", "-- migrate (plan)",
                  "-- migrate (apply)", "-- dimensions load", "--- ingest", "--- ingest end", "timers restored (trap; the set above)",
                  "-- status heavy refresh started once (cache initializer)", "=== ff chain end"]


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _cutover_world(tmp_path, *, selector=SELECTOR, active=("registry-harvest.timer", "registry-status.timer"), extra_env=None, plan_lines=None,
                   replica_aliases=True, remote_src_present=True):
    """The window world of test_registry_window plus: the live registry config and two inboxes under the old labels, a prepared set (new
    registry config, alias table, site config) with its MANIFEST.tsv, a stub repo-mirror tool that knows only SELECTOR, an ssh shim over a
    scratch "remote" tree (the clone's src, the replica root's alias table), and a registry-python wrapper that answers the preflight's
    and the cutover's two snippets."""
    env, P = _rehearsal(tmp_path, active=list(active), pre_installed=OLD_TIMERS + NEW_TIMERS, extra_env=extra_env, plan_lines=plan_lines)
    home, bin_, calls = P["home"], P["bin"], P["calls"]
    reg = home / "registry"
    site = home / ".config" / "sisteer"
    (reg / "config.toml").write_text('[[clusters]]\nname = "oldsite_a"\nhost = "login-shim"\nshared = "/shared"\n')
    inbox_a, inbox_b = reg / "inbox" / "oldsite_a", reg / "inbox" / "oldsite_b"
    (inbox_a / "rev0").mkdir(parents=True)
    (inbox_a / "rev0" / "r.jsonl").write_text("{}\n")
    inbox_b.mkdir(parents=True)
    # the prepared set: the same site label (its file name), the registry label, the mirror selector, the clone named explicitly
    s = tmp_path / "prepared"
    write_site(s, repo=str(P["base"]), registry_root=str(reg),
               clusters={"site_one": {"CLUSTER_SSH": "login-shim", "CLUSTER_SHARED_ROOT": "/shared", "CLUSTER_MIRROR_SELECTOR": selector,
                                      "CLUSTER_REPO_MIRROR": "/shared/repos/checkout"}})
    (s / "config.toml").write_text(f'[[clusters]]\nname = "{SUB_A}"\nhost = "login-shim"\nshared = "/shared"\nremote_src = "/shared/repos/checkout/src"\n'
                                   'qos = "high"\n')
    (s / "site_aliases.toml").write_text(f'[site_aliases]\noldsite_a = "{SUB_A}"\noldsite_b = "{SUB_B}"\n')
    rows = [("config.toml", reg / "config.toml"), ("site_aliases.toml", reg / "site_aliases.toml"), ("config.env", site / "config.env"),
            ("clusters/site_one.env", site / "clusters" / "site_one.env")]
    lines = ["# kind\tsource\ttarget\tbefore\tafter"]
    for src, target in rows:
        lines.append(f"file\t{src}\t{target}\t{_sha(target) if target.exists() else 'absent'}\t{_sha(s / src)}")
    lines += [f"rename\t{inbox_a}\t{reg / 'inbox' / SUB_A}", f"rename\t{inbox_b}\t{reg / 'inbox' / SUB_B}"]
    (s / "MANIFEST.tsv").write_text("\n".join(lines) + "\n")
    before = {str(t): (_sha(t) if t.exists() else None) for _, t in rows}
    # the cluster as the preflight sees it over ssh
    remote = tmp_path / "remote"
    if remote_src_present:
        (remote / "shared/repos/checkout/src/sisteer/registry").mkdir(parents=True)
        (remote / "shared/repos/checkout/src/sisteer/registry/__init__.py").write_text("")
    if replica_aliases:
        (remote / "shared/registry/replica").mkdir(parents=True)
        (remote / "shared/registry/replica/site_aliases.toml").write_bytes(
            (s / "site_aliases.toml").read_bytes() if replica_aliases is True else replica_aliases.encode())
    _shim(bin_, "ssh", f'''echo "ssh $*" >> "{calls}"
while [ $# -gt 0 ]; do case "$1" in -o) shift 2 ;; *) break ;; esac; done
shift
case "$1" in
  git) [ -e "{remote}/.off-mirror" ] && exit 1; exit 0 ;;
  test) [ "$2" = -f ] && [ -f "{remote}$3" ] && exit 0; exit 1 ;;
  cat) cat "{remote}$2" ;;
  *) cat >/dev/null; exit 0 ;;
esac
''')
    _shim(bin_, "git", f'''echo "git $*" >> "{calls}"
case "$*" in
  *"rev-parse --short origin/main"*|*"rev-parse --short HEAD"*) head -c 7 "{P['head']}"; echo ;;
  *"rev-parse origin/main"*|*"rev-parse HEAD"*) cat "{P['head']}" ;;
  *"status --short"*) : ;;
esac
exit 0
''')
    _shim(bin_, "mirror-repos.sh", f'''echo "mirror-repos.sh $*" >> "{calls}"
[ "$1" = --cluster ] && [ "$2" = "{SELECTOR}" ] && exit 0
echo "mirror-repos.sh: unknown cluster '$2' (known: {SELECTOR})" >&2; exit 2
''')
    _shim(bin_, "fuser", f'''echo "fuser $*" >> "{calls}"; [ -e "{tmp_path}/shim-harvest-running" ] && exit 0; exit 1
''')
    os.rename(bin_ / "regpy", bin_ / "regpy.base")
    _shim(bin_, "regpy", f'''if [ "${{1:-}}" = -c ]; then
  snippet="$2"; shift 2
  echo "python -c <snippet> $*" >> "{calls}"
  case "$snippet" in
    *mirror_destinations*) python3 -c 'import sys, tomllib
d = tomllib.load(open(sys.argv[1] + "/config.toml", "rb"))
for c in d.get("clusters", []):
    print("CLUSTER", c["name"], c["host"], c.get("remote_src") or c["shared"] + "/repos/checkout/src")' "$1" ;;
    *ROOT_DIGEST*) echo "ROOT_DIGEST=$(sha256sum < "$1/site_aliases.toml" | cut -d' ' -f1)"; echo "SET_DIGEST=$(sha256sum < "$2" | cut -d' ' -f1)" ;;
    *DIGEST*) for p in "$@"; do echo "DIGEST $(sha256sum < "$p" | cut -d' ' -f1) $p"; done ;;
  esac
  exit 0
fi
case "$*" in
  *"cluster-reads reserve"*) mkdir -p "$WINDOW_STATE_DIR/reads"; [ -n "${{SHIM_RESERVE_NOWAIT:-}}" ] || flock "$WINDOW_STATE_DIR/reads/critical.lock" true
    python3 -c 'import time; print(time.time_ns())' > "{tmp_path}/reserve-at" ;;
  *"cli migrate --plan"*) if [ -e "{tmp_path}/shim-migrated" ]; then echo "python $*" >> "{calls}"
      echo "schema version 32; code 32; pending: none"; echo "shape differs from the code's schema:"; echo "  index shim_idx: differs"; exit 0; fi ;;
  *"cli migrate") case "${{SHIM_MIGRATE:-}}" in
      commits-then-fails) echo "python $*" >> "{calls}"; touch "{tmp_path}/shim-migrated"; echo "applied [32]; schema version 32"
        echo "the shape still differs (shim)" >&2; exit 1 ;;
      fails) echo "python $*" >> "{calls}"; echo "migration failed (rolled back): shim" >&2; exit 1 ;;
    esac ;;
esac
exec "{bin_}/regpy.base" "$@"
''')
    env.update({"WINDOW_SITE_CUTOVER": str(s), "MIRROR_REPOS_SH": str(bin_ / "mirror-repos.sh")})
    P.update({"set": s, "reg": reg, "site": site, "inbox_a": inbox_a, "inbox_b": inbox_b, "before": before, "remote": remote,
              "inbox_new_a": reg / "inbox" / SUB_A, "inbox_new_b": reg / "inbox" / SUB_B})
    return env, P


def _ledger(P) -> dict:
    return json.loads((P["state"] / "windows" / "site-cutover-W1" / "cutover.json").read_text())


def _assert_untouched(P):
    """Every live file as it was before the window (or still absent) and both inboxes under their old names."""
    for target, sha in P["before"].items():
        assert (Path(target).exists() and _sha(target) == sha) if sha else not Path(target).exists(), target
    assert (P["inbox_a"] / "rev0" / "r.jsonl").is_file() and P["inbox_b"].is_dir()
    assert not P["inbox_new_a"].exists() and not P["inbox_new_b"].exists()


def _assert_installed(P):
    for src, target in (("config.toml", P["reg"] / "config.toml"), ("site_aliases.toml", P["reg"] / "site_aliases.toml"),
                        ("config.env", P["site"] / "config.env"), ("clusters/site_one.env", P["site"] / "clusters" / "site_one.env")):
        assert _sha(target) == _sha(P["set"] / src), target
    assert not P["inbox_a"].exists() and not P["inbox_b"].exists()
    assert (P["inbox_new_a"] / "rev0" / "r.jsonl").is_file() and P["inbox_new_b"].is_dir()


def _line_index(log: str, prefix: str) -> int:
    lines = log.splitlines()
    return next(i for i, l in enumerate(lines) if l.startswith(prefix))


# --------------------------------------------------------------------------------------------
# the cutover that succeeds, and the in-flight pull it waits out (rehearsal 1)
# --------------------------------------------------------------------------------------------


HARVEST_PY = r'''
import fcntl, os, sys, time
root, inbox, log, marker = sys.argv[1:5]
fd = os.open(os.path.join(root, "harvester.lock"), os.O_RDWR | os.O_CREAT, 0o644)
fcntl.flock(fd, fcntl.LOCK_EX)
recv = os.path.join(inbox, ".recv-W0-1-ab")
os.makedirs(recv)
open(marker, "w").close()
with open(log, "a") as fh:
    for i in range(5):
        p = os.path.join(recv, f"f{i}.jsonl")
        open(p, "w").write("{}\n")
        fh.write(f"{time.time_ns()} write {p}\n"); fh.flush()
        time.sleep(0.3)
    rev = os.path.join(inbox, "rev1")
    os.makedirs(rev)
    for i in range(5):
        os.rename(os.path.join(recv, f"f{i}.jsonl"), os.path.join(rev, f"f{i}.jsonl"))
    os.rmdir(recv)
    fh.write(f"{time.time_ns()} merged {rev}\n"); fh.flush()
os.unlink(marker)
fcntl.flock(fd, fcntl.LOCK_UN)
os.close(fd)
'''


def test_an_in_flight_pull_is_waited_out_then_the_cutover_installs_renames_and_the_window_runs_on_the_new_names(tmp_path):
    env, P = _cutover_world(tmp_path, extra_env={"WINDOW_QUIET_SLEEP_S": "0.2"})
    marker, wlog = tmp_path / "shim-harvest-running", tmp_path / "harvest-writes.log"
    harvest = subprocess.Popen([sys.executable, "-c", HARVEST_PY, str(P["reg"]), str(P["inbox_a"]), str(wlog), str(marker)])
    try:
        for _ in range(200):
            if marker.exists():
                break
            time.sleep(0.05)
        assert marker.exists(), "the harvest pass is in flight when the window starts"
        r = _run(env)
    finally:
        harvest.wait(timeout=60)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    assert _stamps(log) == SUCCESS_STAMPS, _stamps(log)
    _assert_installed(P)
    # every file the pass wrote is under the new name, and the pass wrote nothing after the old name went away
    assert sorted(os.listdir(P["inbox_new_a"] / "rev1")) == [f"f{i}.jsonl" for i in range(5)]
    writes = [int(l.split()[0]) for l in wlog.read_text().splitlines()]
    renamed = {e["old"]: e["rename_started_ns"] for e in _ledger(P)["renames"]}
    assert max(writes) < renamed[str(P["inbox_a"])], (max(writes), renamed)
    # the reservation ran under the base's code with the cluster's current label (the alias table read backwards)
    calls = _calls(P)
    reserve = next(c for c in calls if "cluster-reads reserve" in c and c.startswith("python "))
    assert "--dest oldsite_a" in reserve, reserve
    pyp = P["pypaths"].read_text().splitlines()
    assert any(l.startswith(f"{P['base']}/src ") and "cluster-reads reserve" in l for l in pyp), pyp[:3]
    # after the cutover every name is the new one: the store authority's destination, the mirror selector, the clone the commit is on
    assert any(f"stores check-authority --dest {SUB_A}" in c for c in calls)
    assert f"mirror-repos.sh --cluster {SELECTOR} checkout" in calls
    assert any(c.startswith("ssh ") and "test -f /shared/repos/checkout/src/sisteer/registry/__init__.py" in c for c in calls)
    assert "preflight passed: the window moves to " in log and "site cutover: installed and validated" in log
    assert f"the installed site config ({P['site'] / 'config.env'}) reads the prepared names" in log
    # the timers come back under the new configuration; nothing undid the cutover
    assert "undoing it" not in log and "timers restored (trap; the set above)" in log


# --------------------------------------------------------------------------------------------
# rehearsals 2, 4 and the cutover's own refusals: nothing touched, the old timers back
# --------------------------------------------------------------------------------------------


def _assert_refused_before_anything_moved(r, P, words):
    log = P["log"].read_text()
    assert r.returncode == 31, r.stderr + log
    assert words in log, log
    assert "site cutover refused or failed rc=31" in log and "=== ff chain ABORTED (site cutover refused or failed rc=31" in log
    _assert_untouched(P)
    calls = _calls(P)
    assert not any("merge -q --ff-only" in c for c in calls) and not any(c.startswith("python") and " migrate" in c for c in calls)
    assert "-- install units" not in log and "-- preflight" not in log and "undoing it" not in log
    assert "systemctl --user start registry-harvest.timer registry-status.timer" in calls, "the old timers come back on the old configuration"


def test_an_interrupted_delivery_under_an_old_inbox_name_is_refused_by_name_and_nothing_moves(tmp_path):
    env, P = _cutover_world(tmp_path)
    recv = P["inbox_a"] / ".recv-W0-1-ab"
    recv.mkdir()
    (recv / "RECEIVING.json").write_text('{"schema": "receiving/1"}\n')
    r = _run(env)
    _assert_refused_before_anything_moved(r, P, f"REFUSED: an interrupted delivery is in {P['inbox_a']}: {recv}")
    assert (recv / "RECEIVING.json").is_file(), "never moved by hand"


def test_a_reads_operation_record_that_is_not_terminal_is_refused_by_name_and_the_windows_own_is_not(tmp_path):
    env, P = _cutover_world(tmp_path)
    ops = P["state"] / "reads" / "ops"
    ops.mkdir(parents=True)
    (ops / "W1.json").write_text(json.dumps({"op_id": "W1", "owner": "window", "state": "reserved"}))          # this window's own: live by design
    (ops / "tick-old.json").write_text(json.dumps({"op_id": "tick-old", "owner": "tick", "state": "receipted"}))
    (ops / "tick-live.json").write_text(json.dumps({"op_id": "tick-live", "owner": "tick", "state": "submitted"}))
    r = _run(env)
    _assert_refused_before_anything_moved(r, P, "REFUSED: the reads operation tick-live is not terminal (state 'submitted', owner 'tick'")
    log = P["log"].read_text()
    assert "tick-old" not in log and "operation W1 " not in log


def test_a_live_file_edited_after_the_set_was_prepared_and_an_existing_inbox_target_are_refused_by_name(tmp_path):
    env, P = _cutover_world(tmp_path)
    (P["reg"] / "config.toml").write_text('[[clusters]]\nname = "oldsite_a"\nhost = "login-shim"\nshared = "/shared"\n# edited\n')
    P["before"][str(P["reg"] / "config.toml")] = _sha(P["reg"] / "config.toml")
    P["inbox_new_b"].mkdir()
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 31, log
    assert f"REFUSED: {P['reg'] / 'config.toml'} changed after the set was prepared" in log
    assert f"REFUSED: the inbox target {P['inbox_new_b']} already exists" in log
    P["inbox_new_b"].rmdir()
    _assert_untouched(P)


# --------------------------------------------------------------------------------------------
# rehearsal 3: a reads tick mid-pull when the reservation is taken over
# --------------------------------------------------------------------------------------------


TICK_PY = r'''
import fcntl, os, sys, time
lock, inbox, log, marker, release = sys.argv[1:6]
os.makedirs(os.path.dirname(lock), exist_ok=True)
fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
fcntl.flock(fd, fcntl.LOCK_EX)
open(marker, "w").close()
d = os.path.join(inbox, "tick-rev")
os.makedirs(d)
open(os.path.join(d, "t.jsonl"), "w").write("{}\n")
if release == "-":
    time.sleep(1.5)
else:
    for _ in range(1200):
        if os.path.exists(release):
            break
        time.sleep(0.05)
open(log, "w").write(f"{time.time_ns()}\n")
fcntl.flock(fd, fcntl.LOCK_UN)
os.close(fd)
'''


def _tick(tmp_path, P, release="-"):
    marker = tmp_path / "tick-holding"
    p = subprocess.Popen([sys.executable, "-c", TICK_PY, str(P["state"] / "reads" / "critical.lock"), str(P["inbox_a"]), str(tmp_path / "tick-released"),
                          str(marker), release])
    for _ in range(200):
        if marker.exists():
            break
        time.sleep(0.05)
    assert marker.exists()
    return p


def test_a_reads_tick_mid_pull_is_waited_for_by_the_takeover_and_its_delivery_moves_with_the_inbox(tmp_path):
    env, P = _cutover_world(tmp_path)
    tick = _tick(tmp_path, P)
    try:
        r = _run(env)
    finally:
        tick.wait(timeout=60)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    released = int((tmp_path / "tick-released").read_text())
    reserved = int((tmp_path / "reserve-at").read_text())
    assert reserved > released, "the reservation was taken only once the tick's pull had left the critical section"
    assert (P["inbox_new_a"] / "tick-rev" / "t.jsonl").is_file() and _stamps(log) == SUCCESS_STAMPS


def test_a_critical_section_still_held_at_the_cutover_is_refused_by_name(tmp_path):
    """The takeover is what waits for a tick mid-pull; should anything still hold the critical section when the cutover runs, the cutover
    refuses by name rather than rename an inbox under a pull."""
    env, P = _cutover_world(tmp_path, extra_env={"SHIM_RESERVE_NOWAIT": "1"})
    release = tmp_path / "release-tick"
    tick = _tick(tmp_path, P, release=str(release))
    try:
        r = _run(env)
    finally:
        release.write_text("")
        tick.wait(timeout=60)
    # the tick's own delivery stays where it wrote it: under the old name
    assert (P["inbox_a"] / "tick-rev" / "t.jsonl").is_file()
    (P["inbox_a"] / "tick-rev" / "t.jsonl").unlink()
    (P["inbox_a"] / "tick-rev").rmdir()
    _assert_refused_before_anything_moved(r, P, f"REFUSED: {P['state'] / 'reads' / 'critical.lock'} is held by another process")


# --------------------------------------------------------------------------------------------
# rehearsal 5: a mirrors pass mid-sync when the window starts
# --------------------------------------------------------------------------------------------


def _mirrors_in_flight(P, countdown: str | None):
    cg = "/user.slice/registry-mirrors.service"
    (P["cgroup"] / cg.lstrip("/")).mkdir(parents=True)
    (P["cgroup"] / cg.lstrip("/") / "cgroup.procs").write_text("777\n")
    (P["tmp"] / "shim-activating-registry-mirrors.service").write_text("3")
    (P["tmp"] / "shim-mainpid-registry-mirrors.service").write_text("777")
    (P["tmp"] / "shim-cgroup-registry-mirrors.service").write_text(cg)
    if countdown is not None:
        (P["tmp"] / "shim-countdown-registry-mirrors.service").write_text(countdown)


def test_a_mirrors_pass_mid_sync_is_stopped_at_its_timer_waited_out_and_confirmed_terminal_before_the_cutover(tmp_path):
    active = ("registry-harvest.timer", "registry-status.timer", "registry-mirrors.timer", "registry-mirrors-full.timer")
    env, P = _cutover_world(tmp_path, active=active)
    _mirrors_in_flight(P, countdown="2")
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 0, r.stderr + log
    calls = _calls(P)
    stop = next(c for c in calls if c.startswith("systemctl --user stop "))
    assert stop.split()[3:] == ["registry-harvest.timer", "registry-cluster-reads.timer", "registry-adapters.timer", "registry-modal-pull.timer",
                                "registry-status.timer", "registry-status-heavy.timer", "registry-store-mirror.timer", "registry-mirrors.timer",
                                "registry-mirrors-full.timer", "registry-adapters.service", "registry-modal-pull.service"], stop
    assert _line_index(log, "writers quiet") < _line_index(log, "mirrors services confirmed terminal") < _line_index(log, "-- site cutover")
    assert any("show -p ControlGroup --value registry-mirrors.service" in c for c in calls)
    # both mirrors timers were active at stop and come back after the window, under the new configuration
    assert any(c.startswith("systemctl --user start ") and "registry-mirrors.timer" in c and "registry-mirrors-full.timer" in c for c in _starts(calls))
    _assert_installed(P)


def test_a_mirrors_service_still_running_after_the_drain_refuses_the_window_before_the_cutover(tmp_path):
    env, P = _cutover_world(tmp_path, active=("registry-harvest.timer", "registry-mirrors.timer"))
    _mirrors_in_flight(P, countdown=None)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 31, log
    assert "a mirrors service is still running after the drain: registry-mirrors.service(MainPID=777) registry-mirrors.service(cgroup pids: 777)" in log
    assert "-- site cutover" not in log
    _assert_untouched(P)


# --------------------------------------------------------------------------------------------
# rehearsal 6 and the stubbed selector: a failure after the install restores everything before any timer restarts
# --------------------------------------------------------------------------------------------


def test_a_mirror_selector_the_tool_does_not_know_fails_the_preflight_and_everything_is_restored_before_the_timers(tmp_path):
    """No CLUSTER_MIRROR_SELECTOR in the prepared block: the selector defaults to the site label, which the stub tool refuses. The preflight
    fails BEFORE `-- migrate (plan)` and before anything irreversible (no fast-forward, no units, no migration); every installed file is
    restored byte for byte, the inboxes move back, and only then do the timers start."""
    env, P = _cutover_world(tmp_path, selector=None)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 32, r.stderr + log
    assert "preflight: the repo-mirror tool refused '--cluster site_one checkout' (rc 2: mirror-repos.sh: unknown cluster 'site_one'" in log, log
    assert "set CLUSTER_MIRROR_SELECTOR in that cluster's block" in log
    _assert_untouched(P)
    calls = _calls(P)
    assert not any("merge -q --ff-only" in c for c in calls) and not any(c.startswith("python") and " migrate" in c for c in calls)
    for stage in ("-- migrate (plan)", "-- install units", "base at", "-- store authority"):
        assert stage not in log, stage
    assert _line_index(log, "site cutover restored byte for byte") < _line_index(log, "timers restored")
    assert _stamps(log)[-1] == "=== ff chain ABORTED (preflight: the repo-mirror tool refused '--cluster site_one checkout' (rc 2: mirror-repos.sh: unknown cluster 'site_one' (known: sel_one)); set CLUSTER_MIRROR_SELECTOR in that cluster's block to the name that tool knows the cluster by)"


def test_a_failure_after_the_fast_forward_puts_back_the_units_the_base_and_the_cutover_before_any_timer_restarts(tmp_path):
    env, P = _cutover_world(tmp_path, plan_lines=["schema version 31; code 32; pending: 0032_synthetic", "rebuild table run: 1 row(s)"])
    units_before = {f: (P["unit_dir"] / f).read_bytes() for f in os.listdir(P["unit_dir"])}
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 15, r.stderr + log
    assert "base at deadbee" in log and "-- install units" in log and "-- migrate (apply)" not in log
    i_undo = _line_index(log, "site cutover: undoing it before any timer restarts")
    assert i_undo < _line_index(log, "site cutover: unit files back as they were") < _line_index(log, "site cutover: base checkout back at")
    assert _line_index(log, "site cutover restored byte for byte") < _line_index(log, "timers restored")
    assert {f: (P["unit_dir"] / f).read_bytes() for f in os.listdir(P["unit_dir"])} == units_before
    calls = _calls(P)
    assert f"git -C {P['base']} reset -q --keep {'deadbee0' * 5}" in calls
    i_reset = calls.index(f"git -C {P['base']} reset -q --keep {'deadbee0' * 5}")
    assert i_reset < calls.index(next(c for c in calls if c.startswith("systemctl --user start ")))
    _assert_untouched(P)


def _tree(root: Path) -> dict:
    """Every entry under ``root``: a file's bytes, a symlink's target, a directory as itself."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for n in dirnames + filenames:
            p = Path(dirpath) / n
            rel = str(p.relative_to(root))
            out[rel] = ("link", os.readlink(p)) if p.is_symlink() else ("dir",) if p.is_dir() else ("file", p.read_bytes())
    return out


def test_an_abort_after_the_uninstall_stage_puts_back_the_drop_ins_and_enablement_links_before_any_timer(tmp_path):
    """The install and the uninstall stage change more than unit files: the uninstall removes a retired unit, its enablement link and a
    retired drop-in. A plan refusal after both stages (a rebuild line, before the migration) puts the unit directory back exactly as it
    was, drop-ins and enablement links included, before any timer starts."""
    env, P = _cutover_world(tmp_path, plan_lines=["schema version 31; code 32; pending: 0032_synthetic", "rebuild table run: 1 row(s)"])
    ud, bin_ = P["unit_dir"], P["bin"]
    (ud / "registry-mirrors.service.d").mkdir()
    (ud / "registry-mirrors.service.d" / "clusters-only.conf").write_text("[Service]\nEnvironment=SHIM_ONLY=1\n")
    (ud / "registry-mirrors.service.d" / "site.conf").write_text("[Service]\nEnvironment=SHIM_SITE=1\n")
    (ud / "registry-adapters.timer").write_text("[Timer]\n")
    (ud / "timers.target.wants").mkdir()
    os.symlink("../registry-adapters.timer", ud / "timers.target.wants" / "registry-adapters.timer")
    os.symlink("../registry-harvest.timer", ud / "timers.target.wants" / "registry-harvest.timer")
    # `systemctl --user disable` removes the unit's enablement link, as the real one does
    os.rename(bin_ / "systemctl", bin_ / "systemctl.base")
    _shim(bin_, "systemctl", f'''if [ "$1" = --user ] && [ "$2" = disable ]; then rm -f "{ud}"/*.wants/"$3"; fi
exec "{bin_}/systemctl.base" "$@"
''')
    before = _tree(ud)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 15, r.stderr + log
    uninstalled = next(l for l in log.splitlines() if l.startswith("retired units uninstalled"))
    assert "registry-adapters.timer" in uninstalled and "registry-mirrors.service.d/clusters-only.conf" in uninstalled, uninstalled
    assert _tree(ud) == before
    assert _line_index(log, "site cutover: unit files back as they were before the install, with their drop-ins and enablement links") \
        < _line_index(log, "timers restored")
    _assert_untouched(P)


def test_a_migration_that_commits_and_then_fails_is_not_rolled_back_and_every_timer_stays_stopped(tmp_path):
    """`registry migrate` can commit and then exit non-zero (its shape check runs after the commit). The window reads the schema version
    again the way the plan read it: it moved, so the migration counts as applied, the cutover, the units and the base checkout stay as
    they are, and no timer starts, by name -- the old code refuses this database."""
    env, P = _cutover_world(tmp_path, plan_lines=["schema version 31; code 32; pending: 0032_synthetic", "shape: matches the code's schema"])
    env["SHIM_MIGRATE"] = "commits-then-fails"
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 8, r.stderr + log
    assert "migrate: the migration committed (schema version 31 -> 32) and registry migrate exited 1; the site cutover stays installed" in log, log
    assert "timers LEFT STOPPED: the migration committed (schema version 31 -> 32)" in log
    assert "site cutover: undoing it" not in log
    calls = _calls(P)
    assert _starts(calls) == [] and not any(" reset -q --keep " in c for c in calls), calls
    _assert_installed(P)


def test_a_migration_that_fails_without_committing_rolls_the_cutover_back_before_the_timers(tmp_path):
    env, P = _cutover_world(tmp_path, plan_lines=["schema version 31; code 32; pending: 0032_synthetic", "shape: matches the code's schema"])
    env["SHIM_MIGRATE"] = "fails"
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 8, r.stderr + log
    assert "migrate: the schema version is still 31 after the failed apply, so nothing committed" in log, log
    assert _line_index(log, "site cutover: undoing it before any timer restarts") < _line_index(log, "site cutover restored byte for byte") \
        < _line_index(log, "timers restored")
    assert "timers LEFT STOPPED" not in log
    _assert_untouched(P)


def _convert_wrapper(P, *, restore_fails: bool) -> str:
    """WINDOW_CONVERT_PY as a wrapper: `site_cutover.py apply` runs for real and then exits 1, as a crash after the install would (files
    installed, the ledger written, no word on what happened); with ``restore_fails`` a `site_cutover.py restore` exits 3 and restores
    nothing. Every other call runs the interpreter as it is."""
    w = P["bin"] / "convert-py"
    on_restore = "echo 'RESTORE PROBLEM: a simulated restore that cannot finish'; exit 3" if restore_fails else ":"
    w.write_text(f'''#!/bin/bash
case "$*" in
  *"site_cutover.py apply"*) "{sys.executable}" "$@"; echo "a simulated stop after the install"; exit 1 ;;
  *"site_cutover.py restore"*) {on_restore} ;;
esac
exec "{sys.executable}" "$@"
''')
    w.chmod(0o755)
    return str(w)


def test_a_cutover_that_stops_without_saying_after_its_install_is_undone_by_the_trap_before_any_timer(tmp_path):
    """The cutover tool ends with a code that is neither success nor its refusal after it installed the set: the ledger it wrote says files
    may be installed, so the window treats it as installed, the trap runs the restore again before any timer starts, and the old timers
    come back on the old configuration once that restore finishes."""
    env, P = _cutover_world(tmp_path)
    env["WINDOW_CONVERT_PY"] = _convert_wrapper(P, restore_fails=False)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 31, r.stderr + log
    assert "site cutover FAILED rc=1 with files possibly installed" in log, log
    assert _line_index(log, "site cutover: undoing it before any timer restarts") < _line_index(log, "site cutover restored byte for byte") \
        < _line_index(log, "timers restored")
    _assert_untouched(P)
    for stage in ("-- preflight", "base at", "-- install units", "-- migrate (plan)"):
        assert stage not in log, stage


#: site_cutover.py run under injected faults: the module is loaded from argv[1] and its main() runs on argv[2:], with its ledger writes
#: and the process's renames wrapped. Each knob counts within this one process:
#:   SC_FAIL_SAVE=N          the N-th ledger write raises ENOSPC before it writes (SC_FAIL_ON=1: so does every later one, a disk that stays full)
#:   SC_CRASH_SAVE=N         the N-th ledger write ends the process before it writes (no exception handler, no restore: a kill)
#:   SC_FAIL_AFTER_RENAME=K  the first ledger write after the K-th rename raises ENOSPC, once (the rename ran, its done mark did not)
#:   SC_CRASH_AFTER_RENAME=K the first ledger write after the K-th rename ends the process
#:   SC_CRASH_RENAME=N       the N-th rename ends the process before it runs (a restore stopped half way)
#:   SC_CRASH_REPLACE=N      the N-th install of a target (an os.replace onto anything but the ledger) ends the process before it runs,
#:                           leaving the temporary file beside the target
#:   SC_COUNT=1              print SAVES=<n> RENAMES=<n> at the end
FAULT_WRAPPER = r'''import errno, importlib.util, os, sys
spec = importlib.util.spec_from_file_location("site_cutover", sys.argv[1])
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)
knob = lambda k: int(os.environ.get(k) or 0)
FAIL_SAVE, CRASH_SAVE, FAIL_AFTER_RENAME = knob("SC_FAIL_SAVE"), knob("SC_CRASH_SAVE"), knob("SC_FAIL_AFTER_RENAME")
CRASH_AFTER_RENAME, CRASH_RENAME, CRASH_REPLACE = knob("SC_CRASH_AFTER_RENAME"), knob("SC_CRASH_RENAME"), knob("SC_CRASH_REPLACE")
FAIL_ON = os.environ.get("SC_FAIL_ON") == "1"
n = {"save": 0, "rename": 0, "replace": 0, "failed_after_rename": False}
real_save, real_rename, real_replace = sc.save_ledger, os.rename, os.replace
def enospc():
    return OSError(errno.ENOSPC, "No space left on device (injected)")
def save(backup, ledger):
    n["save"] += 1
    if n["save"] == CRASH_SAVE or (CRASH_AFTER_RENAME and n["rename"] == CRASH_AFTER_RENAME):
        os._exit(9)
    if FAIL_SAVE and (n["save"] == FAIL_SAVE or (FAIL_ON and n["save"] > FAIL_SAVE)):
        raise enospc()
    if FAIL_AFTER_RENAME and n["rename"] == FAIL_AFTER_RENAME and not n["failed_after_rename"]:
        n["failed_after_rename"] = True
        raise enospc()
    real_save(backup, ledger)
def rename(a, b):
    n["rename"] += 1
    if n["rename"] == CRASH_RENAME:
        os._exit(9)
    real_rename(a, b)
def replace(a, b):
    if os.path.basename(str(b)) != sc.LEDGER:
        n["replace"] += 1
        if n["replace"] == CRASH_REPLACE:
            os._exit(9)
    real_replace(a, b)
sc.save_ledger = save
os.rename, os.replace = rename, replace
rc = sc.main(sys.argv[2:])
if os.environ.get("SC_COUNT"):
    print(f"SAVES={n['save']} RENAMES={n['rename']}")
sys.exit(rc)
'''


def _fault_wrapper(tmp_path) -> Path:
    w = tmp_path / "fault_wrapper.py"
    if not w.exists():
        w.write_text(FAULT_WRAPPER)
    return w


def _fault_convert(P, tmp_path, **faults) -> str:
    """WINDOW_CONVERT_PY as a wrapper: `site_cutover.py apply` runs under the fault knobs, every other call (the restore the trap runs
    included) as it is."""
    w = P["bin"] / "convert-py-faults"
    knobs = " ".join(f"{k}={v}" for k, v in faults.items())
    w.write_text(f'''#!/bin/bash
case "$*" in
  *"site_cutover.py apply"*) exec env {knobs} "{sys.executable}" "{_fault_wrapper(tmp_path)}" "$@" ;;
esac
exec "{sys.executable}" "$@"
''')
    w.chmod(0o755)
    return str(w)


@pytest.mark.parametrize("fault", [{"SC_FAIL_AFTER_RENAME": 1}, {"SC_CRASH_AFTER_RENAME": 1}], ids=["ledger-write-fails", "killed"])
def test_an_inbox_renamed_without_its_done_mark_is_moved_back_before_the_old_timers_restart(tmp_path, fault):
    """The first inbox rename runs and the ledger write that marks it done does not: the disk is full (the cutover's own restore runs and
    the cutover exits 31), or the cutover is killed (it exits 9 and the trap runs the restore). Either way the restore reads the
    filesystem, not the done mark: the inbox is back under its old name with its records, every file is as it was, and only then do the
    old timers restart."""
    env, P = _cutover_world(tmp_path)
    env["WINDOW_CONVERT_PY"] = _fault_convert(P, tmp_path, **fault)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 31, r.stderr + log
    if "SC_FAIL_AFTER_RENAME" in fault:
        assert "FAILED: [Errno 28] No space left on device (injected)" in log and "RESTORED=1" in log, log
        assert "site cutover refused or failed rc=31" in log, log
    else:
        assert "site cutover FAILED rc=9 with files possibly installed" in log, log
        assert _line_index(log, "site cutover: undoing it before any timer restarts") < _line_index(log, "site cutover restored byte for byte")
    _assert_untouched(P)
    assert _line_index(log, "timers restored") > _line_index(log, "-- site cutover"), log
    led = _ledger(P)
    assert len(led["renames"]) == 1 and led["restored"] is True, led
    for stage in ("-- preflight", "base at", "-- install units", "-- migrate (plan)"):
        assert stage not in log, stage


def test_a_cutover_whose_restore_cannot_finish_leaves_every_timer_stopped_by_name(tmp_path):
    """The same stop, and the trap's restore cannot finish either: no timer is started on a configuration that is neither the old one nor
    the new one, the window says so by name and exits 31, and the files stay as the cutover left them for the operator."""
    env, P = _cutover_world(tmp_path)
    env["WINDOW_CONVERT_PY"] = _convert_wrapper(P, restore_fails=True)
    r = _run(env)
    log = P["log"].read_text()
    assert r.returncode == 31, r.stderr + log
    assert "site cutover: the restore did not finish (rc=3" in log, log
    assert "timers LEFT STOPPED: the site cutover could not be restored" in log, log
    assert _starts(_calls(P)) == [], _starts(_calls(P))
    _assert_installed(P)


# --------------------------------------------------------------------------------------------
# the preflight alone
# --------------------------------------------------------------------------------------------


def _preflight_only(env):
    e = dict(env, WINDOW_PREFLIGHT_ONLY="1")
    return subprocess.run(["bash", str(REPO / "scripts" / "registry" / "landing" / "window.sh")], capture_output=True, text=True, env=e)


def test_the_preflight_alone_reads_the_prepared_set_takes_no_lock_and_installs_nothing(tmp_path):
    env, P = _cutover_world(tmp_path)
    r = _preflight_only(env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "preflight only: prepared set" in r.stdout and "preflight PASSED" in r.stdout
    assert (P["state"] / "windows" / "preflight-W1.txt").read_text() == r.stdout
    calls = _calls(P)
    assert not any(c.startswith("systemctl") for c in calls) and not any("cluster-reads" in c for c in calls)
    assert not (P["state"] / "windows" / ".window.lock").exists() and not P["log"].exists() and not P["maint"].exists()
    # the prepared names: the selector, the clone, the prepared registry config's remote_src, the prepared alias table
    assert f"mirror-repos.sh --cluster {SELECTOR} checkout" in calls
    assert any("test -f /shared/repos/checkout/src/sisteer/registry/__init__.py" in c for c in calls)
    assert any(c.startswith("python -c <snippet> ") and str(P["set"] / "site_aliases.toml") in c for c in calls)
    _assert_untouched(P)


def test_the_preflight_alone_names_each_failing_check(tmp_path):
    env, P = _cutover_world(tmp_path / "a", selector=None)
    r = _preflight_only(env)
    assert r.returncode == 32 and "preflight FAILED: the repo-mirror tool refused '--cluster site_one checkout'" in r.stdout, r.stdout
    env, P = _cutover_world(tmp_path / "b", replica_aliases="[site_aliases]\noldsite_a = \"other\"\n")
    r = _preflight_only(env)
    assert r.returncode == 32 and "preflight FAILED: site-alias digests differ" in r.stdout, r.stdout
    env, P = _cutover_world(tmp_path / "c", replica_aliases=False)
    r = _preflight_only(env)
    assert r.returncode == 32 and "site_aliases.toml is under only one of" in r.stdout, r.stdout
    env, P = _cutover_world(tmp_path / "d", remote_src_present=False)
    r = _preflight_only(env)
    assert r.returncode == 32 and f"the [[clusters]] entry {SUB_A}: login-shim:/shared/repos/checkout/src/sisteer/registry/__init__.py does not exist" in r.stdout
    env, P = _cutover_world(tmp_path / "e")
    (P["remote"] / ".off-mirror").write_text("")
    r = _preflight_only(env)
    assert r.returncode == 32 and "is not on login-shim:/shared/repos/checkout and a direct push did not put it there" in r.stdout, r.stdout
    assert any(" push -q ssh://login-shim/shared/repos/checkout " in c for c in _calls(P)), "the commit was pushed by object id first"
    # a [[clusters]] entry that names no qos: its jobs would take the code's default QoS, so the preflight refuses by name
    env, P = _cutover_world(tmp_path / "f")
    (P["set"] / "config.toml").write_text((P["set"] / "config.toml").read_text().replace('qos = "high"\n', ""))
    r = _preflight_only(env)
    assert r.returncode == 32 and f"preflight FAILED: the [[clusters]] entr(ies) {SUB_A} in {P['set']}/config.toml name no qos" in r.stdout, r.stdout


def _second_cluster(P, selector):
    """A block for the second cluster in the prepared set's clusters/ directory that SISTEER_CLUSTERS (the front door's default fan-out)
    does not list, and the replica root's alias table on that cluster."""
    (P["set"] / "clusters" / "site_two.env").write_text(
        f"CLUSTER_SSH=login-two\nCLUSTER_SHARED_ROOT=/shared_two\nCLUSTER_PARTITION=general\nCLUSTER_QOS=high\nCLUSTER_SUBSTRATE={SUB_B}\n"
        f"CLUSTER_LABEL=site_two\nCLUSTER_MIRROR_SELECTOR={selector}\nCLUSTER_REPO_MIRROR=/shared_two/repos/checkout\n")
    (P["remote"] / "shared_two/registry/replica").mkdir(parents=True)
    (P["remote"] / "shared_two/registry/replica/site_aliases.toml").write_bytes((P["set"] / "site_aliases.toml").read_bytes())


def test_the_preflight_addresses_the_cluster_cluster_dest_names_with_every_name_from_its_own_block(tmp_path):
    """CLUSTER_DEST names the landing's cluster by registry label. The preflight reaches that cluster's login host, hands the repo-mirror
    tool that cluster's selector and checks that cluster's clone and replica root, all from its block in the prepared set, though
    SISTEER_CLUSTERS lists only the primary: never the primary's host or clone beside the named label."""
    env, P = _cutover_world(tmp_path / "a")
    _second_cluster(P, SELECTOR)
    r = _preflight_only(dict(env, CLUSTER_DEST=SUB_B))
    assert r.returncode == 0 and "preflight PASSED" in r.stdout, r.stdout + r.stderr
    assert (f"preflight: login host login-two, registry label {SUB_B}, mirror selector {SELECTOR}, clone /shared_two/repos/checkout"
            in r.stdout), r.stdout
    ssh = [c for c in _calls(P) if c.startswith("ssh ")]
    assert any(" login-two git -C /shared_two/repos/checkout cat-file -e " in c for c in ssh), ssh
    assert any(" login-two test -f /shared_two/registry/replica/site_aliases.toml" in c for c in ssh), ssh
    assert not any("/shared/repos/checkout cat-file" in c or " login-shim test -f /shared/registry/replica" in c for c in ssh), ssh
    # the second cluster's own selector, which the tool does not know: refused by name, pointing at that cluster's block
    env, P = _cutover_world(tmp_path / "b")
    _second_cluster(P, "sel_two")
    r = _preflight_only(dict(env, CLUSTER_DEST=SUB_B))
    assert r.returncode == 32 and "preflight FAILED: the repo-mirror tool refused '--cluster sel_two checkout'" in r.stdout, r.stdout
    assert "set CLUSTER_MIRROR_SELECTOR in that cluster's block" in r.stdout, r.stdout


# --------------------------------------------------------------------------------------------
# site_cutover.py by itself
# --------------------------------------------------------------------------------------------


def _set(tmp_path, second_inbox=False):
    live, s = tmp_path / "live", tmp_path / "set"
    (live / "cfg").mkdir(parents=True)
    (live / "inbox" / "old_a" / "rev").mkdir(parents=True)
    (live / "inbox" / "old_a" / "rev" / "r.jsonl").write_text("{}\n")
    if second_inbox:
        (live / "inbox" / "old_b").mkdir(parents=True)
        (live / "inbox" / "old_b" / "b.jsonl").write_text("{}\n")
    s.mkdir()
    (live / "cfg" / "config.toml").write_text("old\n")
    (live / "cfg" / "config.env").write_text("SECRET_MODE=env\n")
    (live / "cfg" / "config.env").chmod(0o600)
    (s / "config.toml").write_text("new\n")
    (s / "config.env").write_text("SECRET_MODE=env\nNEW=1\n")
    (s / "site_aliases.toml").write_text('[site_aliases]\nold_a = "label_a"\n')
    (s / "MANIFEST.tsv").write_text(
        f"file\tconfig.toml\t{live / 'cfg' / 'config.toml'}\t{_sha(live / 'cfg' / 'config.toml')}\t{_sha(s / 'config.toml')}\n"
        f"file\tconfig.env\t{live / 'cfg' / 'config.env'}\t{_sha(live / 'cfg' / 'config.env')}\t{_sha(s / 'config.env')}\n"
        f"file\tsite_aliases.toml\t{live / 'cfg' / 'site_aliases.toml'}\tabsent\t{_sha(s / 'site_aliases.toml')}\n"
        f"rename\t{live / 'inbox' / 'old_a'}\t{live / 'inbox' / 'label_a'}\n"
        + (f"rename\t{live / 'inbox' / 'old_b'}\t{live / 'inbox' / 'label_b'}\n" if second_inbox else ""))
    return live, s


def _sc(*args):
    return subprocess.run([sys.executable, str(SITE_CUTOVER), *map(str, args)], capture_output=True, text=True)


def test_site_cutover_apply_then_restore_returns_every_byte_mode_and_name(tmp_path):
    live, s = _set(tmp_path)
    before = {p: (p.read_bytes(), p.stat().st_mode) for p in (live / "cfg" / "config.toml", live / "cfg" / "config.env")}
    r = _sc("apply", "--set", s, "--root", live / "cfg", "--state", tmp_path / "state", "--backup", tmp_path / "bak")
    assert r.returncode == 0 and "INSTALLED=1" in r.stdout, r.stdout
    assert (live / "cfg" / "config.toml").read_text() == "new\n" and (live / "inbox" / "label_a" / "rev" / "r.jsonl").is_file()
    assert (live / "cfg" / "config.env").stat().st_mode & 0o777 == 0o600, "a private site config stays private"
    again = _sc("apply", "--set", s, "--root", live / "cfg", "--state", tmp_path / "state", "--backup", tmp_path / "bak")
    assert again.returncode == 31 and "already exists" in again.stdout
    r = _sc("restore", "--backup", tmp_path / "bak")
    assert r.returncode == 0 and "RESTORED=1" in r.stdout, r.stdout
    assert {p: (p.read_bytes(), p.stat().st_mode) for p in before} == before
    assert not (live / "cfg" / "site_aliases.toml").exists() and (live / "inbox" / "old_a" / "rev" / "r.jsonl").is_file()
    assert not (live / "inbox" / "label_a").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="a read-only directory does not stop root, so the restore would not meet an error")
def test_site_cutover_restore_names_a_step_it_cannot_undo_and_still_undoes_the_rest(tmp_path):
    """A restore that meets an operating-system error on one step (the config directory read-only, so no temporary file can be written beside
    a target and the new file cannot be removed) names each such step, still moves the inbox back, and exits 3: never a traceback, which the
    window would read as an unknown state. Run again once the directory is writable, the same restore finishes."""
    live, s = _set(tmp_path)
    r = _sc("apply", "--set", s, "--root", live / "cfg", "--state", tmp_path / "state", "--backup", tmp_path / "bak")
    assert r.returncode == 0, r.stdout
    (live / "cfg").chmod(0o500)
    try:
        r = _sc("restore", "--backup", tmp_path / "bak")
    finally:
        (live / "cfg").chmod(0o755)
    assert r.returncode == 3 and "Traceback" not in r.stderr, r.stdout + r.stderr
    assert "RESTORED=0" in r.stdout and r.stdout.count("could not be restored") == 3, r.stdout
    assert (live / "inbox" / "old_a" / "rev" / "r.jsonl").is_file() and not (live / "inbox" / "label_a").exists()
    r = _sc("restore", "--backup", tmp_path / "bak")
    assert r.returncode == 0 and "RESTORED=1" in r.stdout, r.stdout
    assert (live / "cfg" / "config.toml").read_text() == "old\n" and not (live / "cfg" / "site_aliases.toml").exists()
    assert (live / "cfg" / "config.env").stat().st_mode & 0o777 == 0o600


def test_site_cutover_rehearses_on_copies_and_the_planted_failure_is_restored(tmp_path):
    live, s = _set(tmp_path)
    snapshot = {p: p.read_bytes() for p in (live / "cfg").iterdir()}
    for extra, scratch in (([], tmp_path / "scratch1"), (["--plant-failure"], tmp_path / "scratch2")):
        r = _sc("rehearse", "--set", s, "--scratch", scratch, *extra)
        assert r.returncode == 0 and "REHEARSAL=PASS" in r.stdout, r.stdout
    assert "FAILED: a planted failure after the install" in r.stdout and "RESTORED=1" in r.stdout
    assert {p: p.read_bytes() for p in (live / "cfg").iterdir()} == snapshot, "a rehearsal touches only its scratch copies"
    assert (live / "inbox" / "old_a").is_dir()


def test_site_cutover_check_refuses_each_unsafe_state_by_name(tmp_path):
    live, s = _set(tmp_path)
    state = tmp_path / "state"
    ok = _sc("check", "--set", s, "--root", live / "cfg", "--state", state)
    assert ok.returncode == 0 and "CHECKED=1" in ok.stdout, ok.stdout
    (s / "config.toml").write_text("newer\n")
    (live / "cfg" / "site_aliases.toml").write_text("")
    (live / "inbox" / "label_a").mkdir()
    (live / "inbox" / "old_a" / "rev" / "RECEIVING.json").write_text("{}")
    ops = state / "reads" / "ops"
    ops.mkdir(parents=True)
    (ops / "w.json").write_text(json.dumps({"op_id": "w", "state": "pulled"}))
    holder = subprocess.Popen([sys.executable, "-c", "import fcntl, os, sys, time\nfd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT)\n"
                               "fcntl.flock(fd, fcntl.LOCK_EX)\nopen(sys.argv[2], 'w').close()\ntime.sleep(30)",
                               str(live / "cfg" / "harvester.lock"), str(tmp_path / "held")])
    try:
        for _ in range(200):
            if (tmp_path / "held").exists():
                break
            time.sleep(0.05)
        r = _sc("check", "--set", s, "--root", live / "cfg", "--state", state)
    finally:
        holder.kill()
        holder.wait()
    out = r.stdout
    assert r.returncode == 31, out
    for words in ("is not the one the manifest records", f"{live / 'cfg' / 'site_aliases.toml'} exists, and the manifest says it is absent",
                  f"the inbox target {live / 'inbox' / 'label_a'} already exists", "an interrupted delivery is in",
                  "the reads operation w is not terminal (state 'pulled'", f"{live / 'cfg' / 'harvester.lock'} is held by another process"):
        assert words in out, (words, out)
    # the window's own operation is live by design; a malformed manifest is refused before anything is read
    assert _sc("check", "--set", s, "--root", live / "cfg", "--state", state, "--own-op", "w").stdout.count("reads operation") == 0
    (s / "MANIFEST.tsv").write_text("file\tonly-two-columns\n")
    bad = _sc("check", "--set", s, "--root", live / "cfg", "--state", state)
    assert bad.returncode == 31 and "not a `file` row (5 columns) or a `rename` row (3 columns)" in bad.stdout


def test_site_cutover_refuses_a_unit_that_is_still_running(tmp_path):
    live, s = _set(tmp_path)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "systemctl").write_text('#!/bin/bash\n[ "$3" = "registry-mirrors.service" ] && echo activating || echo inactive\n')
    (bin_ / "systemctl").chmod(0o755)
    r = subprocess.run([sys.executable, str(SITE_CUTOVER), "check", "--set", str(s), "--root", str(live / "cfg"), "--state", str(tmp_path / "state"),
                        "--unit", "registry-harvest.service", "--unit", "registry-mirrors.service"], capture_output=True, text=True,
                       env=dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}"))
    assert r.returncode == 31 and "REFUSED: the unit registry-mirrors.service reads activating" in r.stdout and "registry-harvest" not in r.stdout


# --------------------------------------------------------------------------------------------
# site_cutover.py under injected faults: the ledger is written ahead and the restore reads the filesystem
# --------------------------------------------------------------------------------------------

#: the ledger writes of a clean apply of _set's manifest (three files, one rename): the opening record, then an intent and a done mark per
#: install and per rename. The parametrized faults below cover every one of them; test_a_clean_apply_makes_the_ledger_writes_the_faults_cover
#: fails if a step is added without them.
SET_SAVES = 9


def _sf(tmp_path, *args, **faults):
    env = dict(os.environ, **{k: str(v) for k, v in faults.items()})
    return subprocess.run([sys.executable, str(_fault_wrapper(tmp_path)), str(SITE_CUTOVER), *map(str, args)], capture_output=True, text=True, env=env)


def _apply_args(tmp_path, live, s):
    return ("apply", "--set", s, "--root", live / "cfg", "--state", tmp_path / "state", "--backup", tmp_path / "bak")


def _assert_set_back(live, *, second_inbox=False):
    """_set's live tree as it was before the cutover: both files' bytes and the private mode, the absent file absent, every inbox under its
    old name with its records, no new name, no temporary file left beside a target."""
    cfg = live / "cfg"
    assert (cfg / "config.toml").read_text() == "old\n"
    assert (cfg / "config.env").read_text() == "SECRET_MODE=env\n" and (cfg / "config.env").stat().st_mode & 0o777 == 0o600
    assert not (cfg / "site_aliases.toml").exists()
    assert sorted(p.name for p in cfg.iterdir()) == ["config.env", "config.toml"], sorted(p.name for p in cfg.iterdir())
    assert (live / "inbox" / "old_a" / "rev" / "r.jsonl").is_file() and not (live / "inbox" / "label_a").exists()
    if second_inbox:
        assert (live / "inbox" / "old_b" / "b.jsonl").is_file() and not (live / "inbox" / "label_b").exists()


def test_a_clean_apply_makes_the_ledger_writes_the_faults_cover(tmp_path):
    live, s = _set(tmp_path)
    r = _sf(tmp_path, *_apply_args(tmp_path, live, s), SC_COUNT=1)
    assert r.returncode == 0 and "INSTALLED=1" in r.stdout and f"SAVES={SET_SAVES} RENAMES=1" in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("n", range(1, SET_SAVES + 1))
def test_a_ledger_write_that_fails_at_any_step_is_restored_from_what_the_filesystem_shows(tmp_path, n):
    """The n-th ledger write of the apply fails (a full disk): the cutover's own restore puts every step that ran back, the ones whose done
    mark was never written included (the absent file's install at write 7, the inbox rename at write 9), and exits 31 with RESTORED=1."""
    live, s = _set(tmp_path)
    r = _sf(tmp_path, *_apply_args(tmp_path, live, s), SC_FAIL_SAVE=n)
    assert r.returncode == 31 and "FAILED: [Errno 28] No space left on device (injected)" in r.stdout and "RESTORED=1" in r.stdout, r.stdout + r.stderr
    _assert_set_back(live)


def test_a_disk_that_stays_full_leaves_everything_back_and_says_the_ledger_could_not_record_it(tmp_path):
    """Every ledger write from the rename's done mark on fails: the restore still moves the inbox back and puts every file back, then names
    the ledger it could not write and exits 3 (the window keeps the timers stopped). Once the ledger can be written, the restore run again
    finds everything in place and finishes."""
    live, s = _set(tmp_path)
    r = _sf(tmp_path, *_apply_args(tmp_path, live, s), SC_FAIL_SAVE=SET_SAVES, SC_FAIL_ON=1)
    assert r.returncode == 3 and "RESTORED=0" in r.stdout and "could not record the restore" in r.stdout, r.stdout + r.stderr
    _assert_set_back(live)
    again = _sc("restore", "--backup", tmp_path / "bak")
    assert again.returncode == 0 and "RESTORED=1" in again.stdout, again.stdout
    _assert_set_back(live)
    assert json.loads((tmp_path / "bak" / "cutover.json").read_text())["restored"] is True


@pytest.mark.parametrize("n", range(1, SET_SAVES + 1))
def test_a_cutover_killed_before_any_ledger_write_is_put_back_by_a_later_restore(tmp_path, n):
    """The apply is killed just before its n-th ledger write (no handler runs, so its own restore does not either), as the trap finds it
    when the cutover exits with a code it does not know: `restore` alone puts back every step that ran, the rename killed before its done
    mark included. Killed before the first write, nothing was touched and there is no ledger to restore from (the window then restarts
    the timers on the untouched configuration)."""
    live, s = _set(tmp_path)
    r = _sf(tmp_path, *_apply_args(tmp_path, live, s), SC_CRASH_SAVE=n)
    assert r.returncode == 9, r.stdout + r.stderr
    rr = _sc("restore", "--backup", tmp_path / "bak")
    if n == 1:
        assert rr.returncode == 3 and "REFUSED: no cutover.json" in rr.stdout, rr.stdout
    else:
        assert rr.returncode == 0 and "RESTORED=1" in rr.stdout, rr.stdout
    _assert_set_back(live)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_a_cutover_killed_inside_an_install_leaves_no_temporary_file_after_the_restore(tmp_path, n):
    """Killed between writing a target's temporary file and renaming it into place (the n-th of the three installs): the restore removes
    the temporary file and leaves every target as it was."""
    live, s = _set(tmp_path)
    r = _sf(tmp_path, *_apply_args(tmp_path, live, s), SC_CRASH_REPLACE=n)
    assert r.returncode == 9, r.stdout + r.stderr
    assert any(".tmp-site-cutover-" in p.name for p in (live / "cfg").iterdir())
    rr = _sc("restore", "--backup", tmp_path / "bak")
    assert rr.returncode == 0 and "RESTORED=1" in rr.stdout, rr.stdout
    _assert_set_back(live)


@pytest.mark.parametrize("stop", [{"SC_CRASH_RENAME": 2}, {"SC_CRASH_SAVE": 1}], ids=["between-the-two-inboxes", "before-the-ledger-records-it"])
def test_a_restore_stopped_half_way_is_finished_by_the_next_restore(tmp_path, stop):
    """A restore is killed after it moved one of two inboxes back (or after it put everything back and before its ledger said so): the
    next restore finds what is already in place, moves the rest, and finishes."""
    live, s = _set(tmp_path, second_inbox=True)
    assert _sc(*_apply_args(tmp_path, live, s)).returncode == 0
    r = _sf(tmp_path, "restore", "--backup", tmp_path / "bak", **stop)
    assert r.returncode == 9, r.stdout + r.stderr
    if "SC_CRASH_RENAME" in stop:
        assert (live / "inbox" / "old_b").is_dir() and (live / "inbox" / "label_a").is_dir(), "stopped between the two inboxes"
    rr = _sc("restore", "--backup", tmp_path / "bak")
    assert rr.returncode == 0 and "RESTORED=1" in rr.stdout, rr.stdout
    _assert_set_back(live, second_inbox=True)


@pytest.mark.parametrize("after", ["the-old-name-recreated", "both-names-gone"])
def test_an_inbox_restore_that_cannot_tell_where_the_records_are_refuses_by_name_and_moves_nothing(tmp_path, after):
    """Killed after the rename and before its done mark, and then the old name is created again (a writer that started meanwhile), or the
    new name is gone too: the restore cannot move the inbox back without losing or mixing records. It names the inbox, moves neither
    name, still puts every file back, and exits 3, so the window keeps the timers stopped."""
    live, s = _set(tmp_path)
    assert _sf(tmp_path, *_apply_args(tmp_path, live, s), SC_CRASH_AFTER_RENAME=1).returncode == 9
    old, new = live / "inbox" / "old_a", live / "inbox" / "label_a"
    assert new.is_dir() and not old.exists()
    if after == "the-old-name-recreated":
        old.mkdir()
        (old / "late.jsonl").write_text("{}\n")
    else:
        (new / "rev" / "r.jsonl").unlink()
        (new / "rev").rmdir()
        new.rmdir()
    rr = _sc("restore", "--backup", tmp_path / "bak")
    assert rr.returncode == 3 and "RESTORED=0" in rr.stdout, rr.stdout
    if after == "the-old-name-recreated":
        assert f"RESTORE PROBLEM: both {old} and {new} exist" in rr.stdout, rr.stdout
        assert (old / "late.jsonl").is_file() and (new / "rev" / "r.jsonl").is_file(), "neither name is moved"
    else:
        assert f"RESTORE PROBLEM: neither {old} nor {new} exists" in rr.stdout, rr.stdout
    cfg = live / "cfg"
    assert (cfg / "config.toml").read_text() == "old\n" and not (cfg / "site_aliases.toml").exists(), "every file is still put back"
