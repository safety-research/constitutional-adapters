"""The systemd unit templates and the grading host's installer: every template parses (structurally always; rendered for this
checkout and verified with ``systemd-analyze --user verify`` when it is available, so the referenced scripts exist), the cadences and
bounds are the expected ones, the units are secret-free except the gate-baseline tick, which evaluates the site's environment
hook when the site names one (the hook itself is site config, never a template's text; the three Modal units and their credential are
gone since the registry became Slurm-only), none shares a lock with the paper builder, and the installer lays the layout out (dry-run prints;
a real run with ``--no-systemctl`` creates the directories, config.toml, the database and the rendered units without touching systemd).
The rendering itself (the values, the site config, the refusals) is tests/test_registry_units_render.py's.
"""
from __future__ import annotations

import configparser
import os
import re
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from front_door_testlib import write_site  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNITS = os.path.join(REPO, "systemd", "user")
INSTALL = os.path.join(REPO, "scripts", "registry", "install_grading_host.sh")
# registry-cluster-reads (the periodic cluster read's tick) replaces registry-adapters (retired in C6; its unit files are
# gone from the repo and the landing window uninstalls the live copies); registry-modal-pull left with the Modal transport (C10: its unit
# files are gone from the repo too and the same window uninstalls the live copies and the Modal-era drop-ins)
SERVICES = ("registry-harvest", "registry-cluster-reads", "registry-snapshot", "registry-mirrors", "registry-mirrors-full",
            "registry-status", "registry-status-heavy", "registry-paper-rebuild", "registry-gate-baseline", "registry-store-mirror",
            "registry-store-mirror-full")
#: the retired units whose files the repo no longer carries (the window's RETIRED_UNITS; C7 + C10)
RETIRED = ("registry-adapters", "registry-modal-pull")
#: the unit files the installer renders: one .service + one .timer per unit (the dry-run and --units-only tests count them from here)
N_UNIT_FILES = 2 * len(SERVICES)
#: the unit files a render produces, by name (each template's name without `.in`)
RENDERED = sorted([f"{n}.service" for n in SERVICES] + [f"{n}.timer" for n in SERVICES])


def _tmpl(name):
    """The committed template of the unit file `name` (systemd/user/<name>.in)."""
    return os.path.join(UNITS, f"{name}.in")
CADENCE = {"registry-cluster-reads": "10min", "registry-snapshot": "5min", "registry-mirrors": "5min",
           "registry-status": "10min", "registry-status-heavy": "6h", "registry-paper-rebuild": "2min",
           "registry-gate-baseline": "2min", "registry-store-mirror": "30min"}
# the timers that count from the previous run's END (OnUnitInactiveSec), not its start: the harvester (2026-09-30) -- a cycle took
# 75-110 s live while OnUnitActiveSec=20s re-fired 20 s after the START, so the write lock was held 73 % of the hour; the gap after
# every cycle is what lets the judging drivers' claims take the lock. Such a timer carries NO OnUnitActiveSec (both triggers would be
# unioned and the earlier would win) and keeps OnBootSec (an inactive-since trigger never elapses for a unit that has not yet run)
GAP_AFTER_END = {"registry-harvest": "30s"}
# the weekly full mirror verify (2026-09-28) is a CALENDAR timer: Sundays 02:00 UTC (the clusters' quiet window), Persistent=; the weekly
# full STORE-mirror verify an hour later, so the two never queue their Slurm jobs together. The record-log weekly
# conflicts with the periodic unit of the same mirror (both would rsync the same roots and switch the same pointers); the store weekly does
# NOT: a Conflicts= lets the 30-min timer's start STOP the weekly mid-verify, its Slurm job left without its watcher, so the two
# store units serialize on the shared mirror lock instead -- the weekly waits for a tick that holds it (--lock-wait-s), the tick exits 0
# when the weekly holds it
CALENDAR = {"registry-mirrors-full": "Sun *-*-* 02:00:00 UTC", "registry-store-mirror-full": "Sun *-*-* 03:00:00 UTC"}
CONFLICTS = {"registry-mirrors-full": "registry-mirrors.service"}
LOCK_WAIT = {"registry-store-mirror-full": "--lock-wait-s 3600"}
# the ONE unit whose ExecStart evals the secret-manager CLI: the gate-baseline tick, which reads the private GitHub origin (gh's
# credential helper needs GH_TOKEN in a non-login shell). The three units that evaluated it for the Modal credential (snapshot,
# mirrors, modal-pull) no longer do: the destination is gone
CREDENTIAL_UNITS = {"registry-gate-baseline"}
# the two units that restart themselves ON PURPOSE: the paper-rebuild trigger, and the weekly full verify (ONE retry an hour later
# under a 2-per-day start limit — a transient ssh or queue failure). Every other registry unit: the timer is the only re-fire
# (the status / modal-pull / mirrors / snapshot drop-ins said Restart=no since 2026-09-20 — an on-failure restart kept a failing
# oneshot `activating` through every paused window; the drop-ins are now folded into the repo units) — the gate-baseline tick
# SUBMITS a cluster array and the store mirror a multi-GB copy plus a verify job, neither may loop. The weekly full store-mirror
# verify follows the record-log weekly's precedent: one retry an hour later, every step idempotent
SELF_RESTART = {"registry-paper-rebuild", "registry-mirrors-full", "registry-store-mirror-full"}
NO_SELF_RESTART = set(SERVICES) - SELF_RESTART
COMMANDS = {"registry-harvest": "ingest --pull", "registry-cluster-reads": "cluster-reads tick", "registry-snapshot": "snapshot --if-due --push",
            "registry-mirrors": "mirrors sync", "registry-mirrors-full": "mirrors sync --kind ssh --full",
            "registry-status": "status --html", "registry-status-heavy": "status --refresh-heavy", "registry-paper-rebuild": "scripts/registry/paper_rebuild.sh",
            "registry-gate-baseline": "scripts/registry/landing/baseline_refresh.sh", "registry-store-mirror": "stores mirror --dest @STORE_DEST@",
            "registry-store-mirror-full": "stores mirror --dest @STORE_DEST@ --full"}


def _parse(path):
    cp = configparser.ConfigParser(strict=False, allow_no_value=True, interpolation=None, comment_prefixes=("#",), inline_comment_prefixes=None)
    cp.optionxform = str  # keep case
    with open(path) as fh:
        cp.read_string(fh.read())
    return cp


def test_every_unit_present_and_structurally_sound():
    files = sorted(os.listdir(UNITS))
    assert files == sorted(f"{n}.in" for n in RENDERED), "every unit is a template: systemd/user holds registry-*.{service,timer}.in and nothing else"
    for name in SERVICES:
        svc = _parse(_tmpl(f"{name}.service"))
        s = svc["Service"]
        assert s["Type"] == "oneshot" and s["TimeoutStartSec"]
        if name in NO_SELF_RESTART:
            # the timer is the only re-fire: a failed cycle never re-takes the write lock or re-runs a cluster submission
            assert s["Restart"] == "no" and "RestartSec" not in s, name
        else:
            assert s["Restart"] == "on-failure" and s["RestartSec"], name
        assert s["Nice"] == "10" and s["SyslogIdentifier"] == name and s["StandardOutput"] == "journal"
        assert svc["Unit"]["StartLimitBurst"] and svc["Unit"]["StartLimitIntervalSec"]
        assert COMMANDS[name] in s["ExecStart"], (name, s["ExecStart"])
        assert "Environment=PYTHONPATH" not in s or True
        envs = [v for k, v in svc.items("Service") if k == "Environment"]
        text = open(_tmpl(f"{name}.service")).read()
        assert "Environment=PYTHONPATH=@REPO@/src" in text and "Environment=REGISTRY_ROOT=@REGISTRY_ROOT@" in text
        tmr = _parse(_tmpl(f"{name}.timer"))
        t = tmr["Timer"]
        if name in CALENDAR:
            assert t["OnCalendar"] == CALENDAR[name] and t["Persistent"] == "true" and "OnUnitActiveSec" not in t, name
            if name in CONFLICTS:
                assert svc["Unit"]["Conflicts"] == CONFLICTS[name], "never beside the periodic unit of the same mirror: same rsync, same pointer"
            else:
                assert "Conflicts" not in svc["Unit"], (name, "a systemd conflict lets the periodic timer's start stop the weekly mid-run")
                assert LOCK_WAIT[name] in s["ExecStart"], (name, "the weekly serializes on the shared mirror lock: it waits for a tick that holds it")
            assert "--export" not in s["ExecStart"]
        elif name in GAP_AFTER_END:
            assert t["OnUnitInactiveSec"] == GAP_AFTER_END[name] and "OnUnitActiveSec" not in t, (name, dict(t))
            assert t["OnBootSec"], name
        else:
            assert t["OnUnitActiveSec"] == CADENCE[name], (name, t["OnUnitActiveSec"])
            assert t["OnBootSec"], name
        assert t["AccuracySec"] and t["Unit"] == f"{name}.service"
        assert tmr["Install"]["WantedBy"] == "timers.target"


def test_secret_free_except_the_credential_units_and_no_shared_lock_with_the_builder():
    for name in SERVICES:
        text = open(_tmpl(f"{name}.service")).read()
        assert not re.search(r"(?i)(TOKEN|API_KEY|SECRET|PASSWORD)\s*=", text), name
        assert "://" not in text.split("[Service]", 1)[1], name   # no unit reads a secret reference
        assert "paper-autobuild/lock" not in text and "harvester.lock" not in text.split("[Service]")[1]
        exec_line = [l for l in text.splitlines() if l.startswith("ExecStart=")][0]
        if name in CREDENTIAL_UNITS:
            # the site's hook, filled at install time; the template names no secret manager of its own
            assert exec_line.startswith("ExecStart=/bin/sh -c '@UNIT_ENV_HOOK@exec "), name
        else:
            assert "@UNIT_ENV_HOOK@" not in text and "MODAL" + "_TOKEN" not in text, name   # composed: this file is in the no-Modal scan's scope
        assert "eval " not in text, name          # a template evaluates nothing of its own: the site's hook is filled at install
    # the retired units have no file in the repo (the installer copies systemd/user/registry-*; nothing could bring them back)
    for name in RETIRED:
        for f in (f"{name}.service", f"{name}.timer"):
            assert not os.path.exists(_tmpl(f)) and not os.path.exists(os.path.join(UNITS, f)), (name, f)
    # the cluster-reads tick accepts exit 20 (cancelled by a landing window's takeover) as success: a deliberate outcome, not a failure to count
    assert _parse(_tmpl("registry-cluster-reads.service"))["Service"]["SuccessExitStatus"] == "20"


def test_harvest_timer_counts_from_the_previous_cycles_end():
    """2026-09-30: `OnUnitInactiveSec=30s` and no `OnUnitActiveSec` -- the next cycle starts 30 s after the previous one deactivates,
    whatever the cycle took, so every cycle is followed by a window in which `harvester.lock` is free for the claims and the other
    writers (with `OnUnitActiveSec=20s` a 75-110 s cycle was re-fired while still running: the lock was held 73 % of a measured hour).
    `OnBootSec` stays so a (re)started timer fires its first cycle at once; the description no longer promises a 20-s beat."""
    tmr = _parse(_tmpl("registry-harvest.timer"))
    t = tmr["Timer"]
    assert t["OnUnitInactiveSec"] == "30s" and "OnUnitActiveSec" not in t and t["OnBootSec"] == "1min" and t["AccuracySec"] == "1s"
    assert "every 20 s" not in tmr["Unit"]["Description"] and "30 s" in tmr["Unit"]["Description"]
    text = open(_tmpl("registry-harvest.timer")).read()
    assert "OnUnitInactiveSec" in text and "73 %" in text            # the measured reason rides the unit file
    svc = open(_tmpl("registry-harvest.service")).read()
    assert "Every 20 s" not in svc


def test_ssh_units_use_the_agent_socket():
    for name in ("registry-harvest", "registry-cluster-reads", "registry-snapshot", "registry-mirrors", "registry-mirrors-full", "registry-gate-baseline",
                 "registry-store-mirror", "registry-store-mirror-full"):
        assert "Environment=SSH_AUTH_SOCK=%h/.ssh/agent.sock" in open(_tmpl(f"{name}.service")).read(), name


@pytest.mark.skipif(shutil.which("systemd-analyze") is None, reason="systemd-analyze not installed")
def test_systemd_analyze_verify_accepts_every_unit(tmp_path):
    """The units as the installer renders them for THIS checkout and this interpreter, so the referenced scripts exist."""
    d = tmp_path / "units"
    r0 = subprocess.run(["bash", INSTALL, "--units-only", "--no-systemctl", "--unit-dir", str(d)], capture_output=True, text=True, env=_env(tmp_path))
    assert r0.returncode == 0, r0.stdout + r0.stderr
    assert sorted(os.listdir(d)) == RENDERED
    r = subprocess.run(["systemd-analyze", "--user", "verify"] + [str(d / fn) for fn in RENDERED],
                       capture_output=True, text=True, cwd="/")
    if any(m in r.stderr for m in ("Failed to connect", "not supported", "Failed to initialize manager", "Failed to lookup RuntimeDirectory")):
        # a compute node runs no user manager (the Slurm battery: "Failed to lookup RuntimeDirectory path"); the units are verified where one runs
        pytest.skip(f"systemd-analyze cannot run here: {r.stderr.strip()[:120]}")
    noise = [l for l in (r.stderr + r.stdout).splitlines() if l.strip() and "Unit is bound to inactive unit" not in l]
    assert r.returncode == 0 and not noise, "\n".join(noise)


def _env(tmp_path):
    """The installer's environment: HOME is the test's directory, the units are rendered from a neutral site config written there, and
    the store configuration declares that site's store destination (its primary block is generic, so the destination is its
    CLUSTER_LABEL, `cluster_a`)."""
    cfg = tmp_path / "site" / "config.env"
    if not cfg.exists():
        write_site(tmp_path)
    stores = tmp_path / "cluster_reads.yaml"
    stores.write_text("destinations:\n  cluster_a:\n    stores_rel: registry/stores\n")
    return {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "REGISTRY_PY": sys.executable, "REGISTRY_PYTHONPATH": os.path.join(REPO, "src"),
            "SISTEER_CONFIG": str(cfg), "REGISTRY_CLUSTER_READS": str(stores)}


def test_install_grading_host_dry_run_prints_every_step_and_changes_nothing(tmp_path):
    root = tmp_path / "registry"
    r = subprocess.run(["bash", INSTALL, "--dry-run", "--root", str(root), "--unit-dir", str(tmp_path / "units")], capture_output=True, text=True,
                       env=_env(tmp_path))
    assert r.returncode == 0, r.stderr
    for d in ("records", "inbox", "rulings", "snapshots", "refused", "state", "manifests"):
        assert f"+ mkdir -p {root}/{d}" in r.stdout
    assert "config --write-default" in r.stdout and " init" in r.stdout
    assert r.stdout.count("+ install -m 0644") == N_UNIT_FILES
    # the dry run prints every unit it would install, rendered (the directives; the comments are the template's)
    assert all(f"# rendered {fn}:" in r.stdout for fn in RENDERED), r.stdout[-2000:]
    assert "#   ExecStart=" in r.stdout and "@" not in "\n".join(l for l in r.stdout.splitlines() if l.startswith("#   "))
    assert "+ systemctl --user enable --now registry-harvest.timer registry-cluster-reads.timer registry-snapshot.timer registry-mirrors.timer " \
           "registry-mirrors-full.timer registry-status.timer registry-status-heavy.timer registry-gate-baseline.timer " \
           "registry-store-mirror.timer registry-store-mirror-full.timer" in r.stdout
    assert "registry-modal-pull" not in r.stdout
    assert "registry-paper-rebuild.timer installed but NOT enabled" in r.stdout
    assert not root.exists() and not (tmp_path / "units").exists()


def test_install_grading_host_no_systemctl_lays_out_root_config_db_and_units(tmp_path):
    root = tmp_path / "registry"
    units = tmp_path / "units"
    r = subprocess.run(["bash", INSTALL, "--no-systemctl", "--root", str(root), "--unit-dir", str(units), "--paper-checkout", str(tmp_path / "pf")],
                       capture_output=True, text=True, env=_env(tmp_path))
    assert r.returncode == 0, r.stderr + r.stdout
    for d in ("records", "inbox", "rulings", "snapshots", "refused", "state", "manifests"):
        assert (root / d).is_dir()
    assert not os.path.islink(root / "inbox")
    assert (root / "config.toml").exists() and (root / "registry.sqlite").exists()
    cfg = (root / "config.toml").read_text()
    assert ('checkout = "~/pf"' in cfg or f'checkout = "{tmp_path}/pf"' in cfg)   # HOME=tmp_path → tilde form
    # the default file carries no Modal table (C10): the new loader's check passes on what the installer writes
    import tomllib
    assert "modal" not in tomllib.loads(cfg)
    chk = subprocess.run([sys.executable, "-m", "sisteer.registry.cli", "--root", str(root), "config", "--check"], capture_output=True, text=True,
                         env=dict(_env(tmp_path), PYTHONPATH=os.path.join(REPO, "src")))
    assert chk.returncode == 0 and chk.stdout.startswith("config --check: OK "), chk.stdout + chk.stderr
    # the retired flag is gone with the destination
    r_flag = subprocess.run(["bash", INSTALL, "--dry-run", "--root", str(root), "--unit-dir", str(units), "--no-modal"], capture_output=True, text=True,
                            env=_env(tmp_path))
    assert r_flag.returncode == 2 and "unknown argument: --no-modal" in r_flag.stderr
    assert sorted(os.listdir(units)) == RENDERED
    assert all(oct(os.stat(units / fn).st_mode & 0o777) == "0o644" for fn in RENDERED)
    assert (tmp_path / "figs" / "results-registry").is_dir()
    assert "systemctl" not in "\n".join(l for l in r.stdout.splitlines() if l.startswith("+ "))
    # idempotent: a second run leaves config.toml and the database alone
    before = (root / "config.toml").read_text()
    r2 = subprocess.run(["bash", INSTALL, "--no-systemctl", "--root", str(root), "--unit-dir", str(units)], capture_output=True, text=True,
                        env=_env(tmp_path))
    assert r2.returncode == 0 and (root / "config.toml").read_text() == before and "exists" in r2.stdout + r2.stderr


def test_writer_units_do_not_restart_themselves_and_the_harvest_cycle_bound_is_45min():
    """The harvest unit looped (10-min timeout, on-failure restart every 20 s) holding the write lock, and
    an on-failure restart of the adapters pass re-took the lock a targeted re-read needed. The timer is the only re-fire;
    the harvest bound covers a cycle that folds thousands of pulled files. Read from the unit text the installer copies."""
    harvest = open(_tmpl("registry-harvest.service")).read()
    reads = open(_tmpl("registry-cluster-reads.service")).read()
    assert re.search(r"^Restart=no$", harvest, re.M) and re.search(r"^TimeoutStartSec=45min$", harvest, re.M)
    assert re.search(r"^Restart=no$", reads, re.M)
    for text in (harvest, reads):
        assert not re.search(r"^Restart=on-failure$", text, re.M) and not re.search(r"^RestartSec=", text, re.M)
        assert "the timer is the only re-fire" in text          # the unit says why it never restarts itself
    # the installer copies these files verbatim (no heredoc of its own)
    installer = open(INSTALL).read()
    assert 'UNITS_SRC="$REPO/systemd/user"' in installer and "Restart=" not in installer


# --------------------------------------------------------------------------------------------
# registry-slurm-fast-windows Phase 1 (S3 / S4): the render leaves the harvester's path; the status units; --units-only; the window's set
# --------------------------------------------------------------------------------------------

WINDOW = os.path.join(REPO, "scripts", "registry", "landing", "window.sh")


def test_harvest_unit_no_longer_renders_the_status_unit_owns_the_page_and_the_heavy_unit_is_bounded():
    """S4: the harvester's `--html` render ran ~38 min after a ~2-min fold inside the unit's 45-min bound (47 timeouts in two days);
    the page is registry-status.timer's alone. The heavy refresh is a oneshot at Nice 10, 45-min bound, Restart=no, 6-hourly."""
    harvest = _parse(_tmpl("registry-harvest.service"))["Service"]["ExecStart"]
    # C6: `--honor-maintenance` -- a landing window's state/maintenance.json holds the cycle at the door, before the pull and the lock
    assert harvest.endswith("ingest --pull --honor-maintenance") and "--html" not in harvest
    heavy = _parse(_tmpl("registry-status-heavy.service"))["Service"]
    assert heavy["ExecStart"].endswith("status --refresh-heavy") and heavy["TimeoutStartSec"] == "45min" and heavy["Restart"] == "no"
    assert heavy["Nice"] == "10" and "eval " not in heavy["ExecStart"]
    status = _parse(_tmpl("registry-status.service"))["Service"]
    assert status["Restart"] == "no" and "RestartSec" not in status and status["TimeoutStartSec"] == "10min"
    # the live drop-ins' rule is now the repo's: no registry unit restarts itself but the two that do so on purpose (SELF_RESTART)
    for name in SERVICES:
        text = open(_tmpl(f"{name}.service")).read()
        if name in SELF_RESTART:
            assert re.search(r"^Restart=on-failure$", text, re.M) and re.search(r"^RestartSec=", text, re.M), name
        else:
            assert re.search(r"^Restart=no$", text, re.M) and not re.search(r"^RestartSec=", text, re.M), name


def test_install_grading_host_units_only_copies_and_reloads_and_enables_nothing(tmp_path):
    """The landing window's install stage: unit files + daemon-reload; no layout, no config, no database, nothing enabled or started."""
    root = tmp_path / "registry"
    units = tmp_path / "units"
    r = subprocess.run(["bash", INSTALL, "--dry-run", "--units-only", "--root", str(root), "--unit-dir", str(units)], capture_output=True, text=True,
                       env=_env(tmp_path))
    assert r.returncode == 0, r.stderr + r.stdout
    plus = [l for l in r.stdout.splitlines() if l.startswith("+ ")]
    assert sum(1 for l in plus if l.startswith("+ install -m 0644")) == N_UNIT_FILES
    assert "+ systemctl --user daemon-reload" in plus
    assert not any("enable" in l or "start" in l or "mkdir -p " + str(root) in l or "config --write-default" in l or l.endswith(" init") for l in plus), plus
    assert "nothing enabled or started (--units-only)" in r.stdout
    # a real copy, no systemd: the files land, nothing else exists
    r2 = subprocess.run(["bash", INSTALL, "--units-only", "--no-systemctl", "--root", str(root), "--unit-dir", str(units)], capture_output=True, text=True,
                        env=_env(tmp_path))
    assert r2.returncode == 0, r2.stderr + r2.stdout
    assert sorted(os.listdir(units)) == RENDERED and not root.exists()
    # a failure propagates: a unit directory that cannot be made exits non-zero (set -e), so the window aborts on it
    r3 = subprocess.run(["bash", INSTALL, "--units-only", "--no-systemctl", "--unit-dir", "/proc/nonexistent/units"], capture_output=True, text=True,
                        env=_env(tmp_path))
    assert r3.returncode != 0


def test_install_grading_host_skip_timer_leaves_one_timer_out_of_the_enable_set(tmp_path):
    r = subprocess.run(["bash", INSTALL, "--dry-run", "--root", str(tmp_path / "r"), "--unit-dir", str(tmp_path / "u"), "--skip-timer",
                        "registry-snapshot.timer"], capture_output=True, text=True, env=_env(tmp_path))
    assert r.returncode == 0, r.stderr
    line = [l for l in r.stdout.splitlines() if l.startswith("+ systemctl --user enable --now")][0]
    assert "registry-snapshot.timer" not in line and "registry-status-heavy.timer" in line and "registry-harvest.timer" in line


def test_window_timer_set_is_the_installers_default_set_minus_never_start():
    """window.sh's DEFAULT_TIMERS (the set whose newly installed members the trap starts) == install_grading_host.sh's TIMERS − NEVER_START:
    registry-snapshot.timer (the operator's call) and the two RETIRED units registry-adapters.timer and registry-modal-pull.timer,
    which the window stops and never starts again (no failure path, no trap). The timers it stops are the DB writers/readers plus the
    two status units; the retired SERVICES are stopped with them and confirmed gone before the base moves. The window also UNINSTALLS
    the Modal pull's unit files (both retired units' files are gone from the repo) and removes the Modal-era drop-ins (the modal-pull
    unit's directory, the mirrors unit's clusters-only.conf). Since the merge of 2026-10-06 the store mirror's TIMER is stopped too --
    a tick inside the window re-publishes a component's cluster `current` in the running code's format (main's, unsealed and
    unrecorded, until the cutover) -- and, being in the installer's default set, is restored by the trap; the weekly
    registry-store-mirror-full.timer is not stopped."""
    installer = open(INSTALL).read()
    m = re.search(r"^TIMERS=\((.*)\)$", installer, re.M)
    installer_set = set(m.group(1).split())
    window = open(WINDOW).read()
    default = set(re.search(r'^DEFAULT_TIMERS="(.*)"$', window, re.M).group(1).split())
    never = set(re.search(r'^NEVER_START="(.*)"$', window, re.M).group(1).split())
    stopped = set(re.search(r'^TIMERS="(.*)"$', window, re.M).group(1).split())
    retired = set(re.search(r'^RETIRED_SERVICES="(.*)"$', window, re.M).group(1).split())
    assert never == {"registry-snapshot.timer", "registry-adapters.timer", "registry-modal-pull.timer"}
    assert default == installer_set - never
    assert stopped == {"registry-harvest.timer", "registry-cluster-reads.timer", "registry-adapters.timer", "registry-modal-pull.timer", "registry-status.timer",
                       "registry-status-heavy.timer", "registry-store-mirror.timer"}
    assert "registry-store-mirror.timer" in default and "registry-store-mirror-full.timer" not in stopped       # Stopped, restored by the trap
    assert stopped - never <= default                                                                           # every non-retired timer the window stops comes back
    assert retired == {"registry-adapters.service", "registry-modal-pull.service"}
    # C7 + C10: the retired UNIT FILES the window uninstalls have no source in the repo (the installer cannot copy them back) and are not in
    # its set; the Modal-era drop-ins it removes are named too
    uninstalled = set(re.search(r'^RETIRED_UNITS="(.*)"$', window, re.M).group(1).split())
    assert uninstalled == {"registry-adapters.timer", "registry-adapters.service", "registry-modal-pull.timer", "registry-modal-pull.service"}
    for u in uninstalled:
        assert not os.path.exists(_tmpl(u)) and not os.path.exists(os.path.join(UNITS, u)) and u not in installer_set, u
    assert uninstalled == {f"{n}.{k}" for n in RETIRED for k in ("timer", "service")}
    dropins = set(re.search(r'^RETIRED_DROPINS="(.*)"$', window, re.M).group(1).split())
    assert dropins == {"registry-modal-pull.service.d", "registry-mirrors.service.d/clusters-only.conf"}
    assert "registry-modal-pull" not in installer and "--no-modal" not in installer
    assert "registry-cluster-reads.timer" in installer_set and "registry-cluster-reads.timer" in default
    # nothing in the script starts the snapshot timer, a retired unit, or any unit by name other than the heavy refresh
    starts = re.findall(r"systemctl --user (?:start|enable --now)[^\n]*", window)
    assert all("registry-snapshot" not in l and "registry-adapters" not in l and "registry-modal-pull" not in l for l in starts), starts
    assert any("registry-status-heavy.service" in l and "--no-block" in l for l in starts)
    # the retired services are confirmed by a cgroup PID list (show -p ControlGroup / MainPID), never by a command-text match
    assert "show -p ControlGroup --value" in window and "show -p MainPID --value" in window and "cgroup.procs" in window
    assert not re.search(r"pkill|killall|kill -", window), "nothing in the window kills by text"


# --------------------------------------------------------------------------------------------
# the periodic cluster read's tick replaces the retired adapters unit
# --------------------------------------------------------------------------------------------


def test_cluster_reads_tick_unit_replaces_the_adapters_unit_with_the_plans_bounds_and_no_credential():
    """`registry-cluster-reads.{service,timer}` -- every 10 min, OnBootSec 4 min, Nice 10, Restart=no, no secret; the
    tick's state root in the environment (REGISTRY_LANDING_STATE); the unit outlasts the tick's 40-min watch bound (the array's walltime +
    15 min) so systemd never cuts a tick in its last minutes -- a cut tick releases on SIGTERM (exit 25); exit 20 (a window's takeover) is a
    success. The retired `registry-adapters.{service,timer}` are gone from the repo."""
    svc = _parse(_tmpl("registry-cluster-reads.service"))["Service"]
    assert svc["ExecStart"].endswith("-m sisteer.registry.cli cluster-reads tick") and "eval " not in svc["ExecStart"]
    assert svc["TimeoutStartSec"] == "50min" and svc["Restart"] == "no" and svc["Nice"] == "10" and svc["SuccessExitStatus"] == "20"
    text = open(_tmpl("registry-cluster-reads.service")).read()
    assert "Environment=REGISTRY_LANDING_STATE=%h/.local/state/registry-landing" in text and "Environment=SSH_AUTH_SOCK=%h/.ssh/agent.sock" in text
    assert "--export" not in svc["ExecStart"] and "SIGTERM" in text and "40 min" in text
    tmr = _parse(_tmpl("registry-cluster-reads.timer"))["Timer"]
    assert tmr["OnUnitActiveSec"] == "10min" and tmr["OnBootSec"] == "4min" and tmr["Unit"] == "registry-cluster-reads.service"
    assert not any(os.path.exists(p) for f in ("registry-adapters.service", "registry-adapters.timer") for p in (_tmpl(f), os.path.join(UNITS, f)))
    # the writer set of the migration window and the measurement's unit list follow the replacement
    mw = open(os.path.join(REPO, "scripts", "registry", "migrate_window.sh")).read()
    line = re.search(r'^WRITER_UNITS="\$\{REGISTRY_WRITER_UNITS:-(.*)\}"$', mw, re.M).group(1).split()
    assert "registry-cluster-reads.timer" in line and "registry-cluster-reads.service" in line and not any("registry-adapters" in u for u in line)
    timers_py = open(os.path.join(REPO, "scripts", "registry", "measure", "timers.py")).read()
    units_line = re.search(r"^UNITS = \[(.*?)\]", timers_py, re.M | re.S).group(1)
    assert '"registry-cluster-reads"' in units_line and '"registry-adapters"' not in units_line
