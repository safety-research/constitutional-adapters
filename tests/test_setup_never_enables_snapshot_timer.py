"""00_setup.sh --what registry: the installer runs with --no-enable, then an EXPLICIT allowlist
enable that NEVER names registry-snapshot.timer (it stays stopped) and preserves a timer that is already
enabled. Run under a fake systemctl / systemd-analyze / registry python.

After the Modal pull's unit files were deleted the allowlist still named its timer, and the enable loop had no failure check,
so on a correctly cut-over installation `systemctl` failed and the stage still printed its done line (the fake systemctl of the
time accepted any unit). Now (1) the allowlist is pinned a subset of install_grading_host.sh's TIMERS, disjoint from the window's
RETIRED_UNITS and NEVER_START, with a unit file per member; (2) a member without a unit file refuses the stage by name before
the installer and before any systemctl; (3) the fake systemctl REFUSES a unit the installer did not install, the way systemd
does; (4) a failed enable fails the stage, and nothing after it is enabled."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, assert_no_tokens, run_stage, scrub_site, write_site  # noqa: E402

# Every stage run here moves HOME to the fake home (_fakes / _env) and checks the units where systemd looks for them, the
# installer's default ~/.config/systemd/user under that home: the suite's empty unit directory ($REGISTRY_UNIT_DIR,
# tests/conftest.py) is lifted for these tests.
pytestmark = pytest.mark.usefixtures("home_site_default")

SETUP = REPRO / "00_setup.sh"
INSTALLER = REPO / "scripts" / "registry" / "install_grading_host.sh"
WINDOW = REPO / "scripts" / "registry" / "landing" / "window.sh"
UNITS = REPO / "systemd" / "user"

#: the fake systemctl's refusal of a unit it has no file for, in systemd's own words
_MISSING = 'echo "Failed to enable unit: Unit file $u does not exist." >&2; exit 1'


def _fakes(tmp_path: Path, enable_case: str) -> tuple[Path, Path, Path]:
    """A fake bin with systemctl (recording every call to systemctl.log; `is-enabled` says registry-harvest.timer is already
    enabled and every other unit disabled; `enable --now` runs `enable_case`), systemd-analyze and the registry python.
    Returns (bin, the call log, the fake HOME whose ~/.config/systemd/user the installer copies the unit files into)."""
    fake = tmp_path / "bin"; fake.mkdir()
    rec = tmp_path / "systemctl.log"
    (fake / "systemctl").write_text(f'''#!/bin/bash
echo "$*" >> {rec}
case "$*" in
  *is-enabled*registry-harvest.timer*) echo enabled; exit 0;;     # already on: must be LEFT alone
  *is-enabled*) echo disabled; exit 1;;
  *"enable --now "*) all="$*"; u="${{all##* }}"; {enable_case};;
  *) exit 0;;
esac
'''); (fake / "systemctl").chmod(0o755)
    (fake / "systemd-analyze").write_text("#!/bin/bash\nexit 0\n"); (fake / "systemd-analyze").chmod(0o755)
    (fake / "python").write_text("#!/bin/bash\nexit 0\n"); (fake / "python").chmod(0o755)
    home = tmp_path / "home"; (home / ".config/systemd/user").mkdir(parents=True)
    return fake, rec, home


def _env(fake: Path, home: Path) -> dict:
    """The stage's environment: the fake bin first, and a store configuration declaring the neutral site's store destination (its
    primary block is generic, so the destination is its CLUSTER_LABEL, `cluster_a`)."""
    stores = fake / "cluster_reads.yaml"
    stores.write_text("destinations:\n  cluster_a:\n    stores_rel: registry/stores\n")
    return {"PATH": f"{fake}:{os.environ['PATH']}", "HOME": str(home), "REGISTRY_PY": str(fake / "python"),
            "REGISTRY_CLUSTER_READS": str(stores)}


def _allowlist() -> set[str]:
    m = re.search(r"^\s*ALLOW=\((.*)\)$", SETUP.read_text(), re.M)
    assert m, "00_setup.sh has no ALLOW=(...) line"
    return set(m.group(1).split())


def test_registry_setup_allowlist(tmp_path):
    """The stub refuses a unit without a file under the fake HOME's unit directory -- the installer (--no-enable) rendered the
    repo's unit templates there a moment earlier, so every allowlisted enable succeeds and a retired unit would have been refused."""
    site = write_site(tmp_path)
    fake, rec, home = _fakes(tmp_path, f'[ -f "$HOME/.config/systemd/user/$u" ] || {{ {_MISSING}; }}; exit 0')
    r = run_stage("00_setup.sh", "--what", "registry", site=site, env=_env(fake, home))
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    calls = rec.read_text().splitlines()
    enables = [c for c in calls if "enable" in c and "is-enabled" not in c]
    assert enables, calls
    assert not any("registry-snapshot.timer" in c for c in enables), enables
    assert not any("registry-harvest.timer" in c for c in enables), enables          # already enabled: preserved
    assert any("registry-cluster-reads.timer" in c for c in enables), enables      # the tick (C7) in place of the retired adapters timer
    assert not any("registry-adapters.timer" in c for c in enables), enables
    # C10: the Modal pull's timer is retired -- never enabled, never probed, never named to systemctl at all
    assert not any("registry-modal-pull" in c for c in calls), calls
    # every allowlisted member was either probed enabled or enabled; the set of enables is exactly the allowlist minus the preserved one
    enabled_units = {c.split()[-1] for c in enables}
    assert enabled_units == _allowlist() - {"registry-harvest.timer"}, (enabled_units, _allowlist())
    # the unit files the stub checked against came from the installer's render step
    for t in _allowlist():
        assert (home / ".config/systemd/user" / t).is_file(), t
    assert "registry-snapshot.timer: NOT touched" in r.stdout
    assert "00_setup done" in r.stdout


def test_a_failed_enable_fails_the_stage(tmp_path):
    """systemd refuses `enable --now registry-mirrors-full.timer` (no unit file, the shape of the C10 defect): the stage exits
    non-zero with a REFUSED line naming the unit, enables nothing after it, and prints no done line."""
    site = write_site(tmp_path)
    fake, rec, home = _fakes(tmp_path, f'[ "$u" = registry-mirrors-full.timer ] && {{ {_MISSING}; }}; exit 0')
    r = run_stage("00_setup.sh", "--what", "registry", site=site, env=_env(fake, home))
    assert r.returncode != 0, r.stdout[-1500:]
    assert "REFUSED:" in r.stderr and "registry-mirrors-full.timer" in r.stderr and "systemctl --user enable --now" in r.stderr, r.stderr[-1500:]
    assert "00_setup done" not in r.stdout
    calls = rec.read_text().splitlines()
    enables = [c.split()[-1] for c in calls if "enable" in c and "is-enabled" not in c]
    assert "registry-mirrors-full.timer" in enables and "registry-status.timer" not in enables, enables   # stopped AT the failure
    assert enables.index("registry-cluster-reads.timer") < enables.index("registry-mirrors-full.timer"), enables


def test_a_retired_unit_in_the_allowlist_refuses_before_the_installer_and_before_systemctl(tmp_path):
    """A copy of reproduce/ whose allowlist names the Modal pull's timer (the C10 defect re-created) refuses by name before the
    installer runs (no registry root laid out) and before any systemctl call (an empty call log) -- so a future retirement that
    forgets this list fails loudly instead of printing a done line over a failed enable."""
    site = write_site(tmp_path)
    fake, rec, home = _fakes(tmp_path, "exit 0")
    copy_repo = tmp_path / "repo"; copy_repo.mkdir()
    shutil.copytree(REPRO, copy_repo / "reproduce", ignore=shutil.ignore_patterns("__pycache__"))
    (copy_repo / "src" / "sisteer").mkdir(parents=True); shutil.copy(REPO / "src" / "sisteer" / "site.py", copy_repo / "src" / "sisteer")   # clusters.py parses with it
    (copy_repo / "systemd").symlink_to(UNITS.parent, target_is_directory=True)      # the real unit templates; the retired one is absent
    stage = copy_repo / "reproduce" / "00_setup.sh"
    text = stage.read_text()
    assert "registry-modal-pull" not in text
    injected = text.replace("ALLOW=(registry-harvest.timer", "ALLOW=(registry-modal-pull.timer registry-harvest.timer", 1)
    assert injected != text
    stage.write_text(injected)
    e = dict(os.environ); e.pop("SISTEER_SITE_DIR", None); e.pop("SISTEER_CONFIG", None)
    e["SISTEER_STATE_DIR"] = str(tmp_path / "state"); e.update(_env(fake, home))
    r = subprocess.run(["bash", str(stage), "--what", "registry", "--config", str(site / "config.env")], capture_output=True, text=True,
                       env=e, cwd=str(REPO), timeout=300)
    assert r.returncode != 0, r.stdout[-1500:]
    assert "REFUSED:" in r.stderr and "registry-modal-pull.timer" in r.stderr and "no unit file template" in r.stderr, r.stderr[-1500:]
    assert not rec.exists(), rec.read_text() if rec.exists() else None              # no systemctl call of any kind
    assert not (tmp_path / "registry").exists()                                    # the installer never ran (no layout)
    assert "00_setup done" not in r.stdout
    # the same check passes --dry-run too: a dry run of a stale allowlist refuses just the same, so the defect never hides behind it
    r2 = subprocess.run(["bash", str(stage), "--what", "registry", "--dry-run", "--config", str(site / "config.env")], capture_output=True,
                        text=True, env=e, cwd=str(REPO), timeout=300)
    assert r2.returncode != 0 and "registry-modal-pull.timer" in r2.stderr, r2.stdout[-800:] + r2.stderr[-800:]


def test_allowlist_is_a_subset_of_the_installers_set_with_a_unit_file_per_member():
    """The static pin: 00_setup.sh's ALLOW is a non-empty subset of install_grading_host.sh's TIMERS, disjoint from window.sh's
    RETIRED_UNITS and NEVER_START (so registry-snapshot.timer and every retired unit are out), every member has a unit file in
    systemd/user/ (its template, <name>.in), and no retired unit's name appears anywhere in the stage's text (install_grading_host.sh is
    held to the same)."""
    allow = _allowlist()
    installer = INSTALLER.read_text()
    timers = set(re.search(r"^TIMERS=\((.*)\)$", installer, re.M).group(1).split())
    window = WINDOW.read_text()
    retired = set(re.search(r'^RETIRED_UNITS="(.*)"$', window, re.M).group(1).split())
    never = set(re.search(r'^NEVER_START="(.*)"$', window, re.M).group(1).split())
    assert allow, "an empty allowlist enables nothing"
    assert allow <= timers, allow - timers
    assert not (allow & retired), allow & retired
    assert not (allow & never), allow & never
    assert "registry-snapshot.timer" not in allow and "registry-cluster-reads.timer" in allow
    for t in sorted(allow):
        assert (UNITS / f"{t}.in").is_file(), f"{t}: in the allowlist, no unit template under systemd/user/"
    setup = SETUP.read_text()
    for stem in ("registry-modal-pull", "registry-adapters"):
        assert stem not in setup, stem
    assert "registry-modal-pull" not in installer      # the same pin tests/test_registry_systemd_units.py holds the installer to
    # the stage checks every member's unit file BEFORE the installer line and makes a failed installer and a failed enable fatal
    i_check = setup.index('[ -f "$REPRO_REPO/systemd/user/$t.in" ]')
    i_install = setup.index("install_grading_host.sh\" --no-enable")
    assert i_check < i_install
    # the installer renders from the site config this stage loaded, and runs under --dry-run too (in its own --dry-run): never repro_run
    assert re.search(r'install_grading_host\.sh" --no-enable --root "\$ROOT" --config "\$SISTEER_CONFIG" \$\(repro_dry_flag\) \\\n\s*\|\| repro_fatal', setup)
    assert not re.search(r"repro_run[^\n]*install_grading_host", setup)
    assert re.search(r'repro_run systemctl --user enable --now "\$t" \\\n\s*\|\| repro_fatal', setup)


def test_registry_dry_run_on_a_neutral_site_prints_the_rendered_units_and_writes_nothing(tmp_path):
    """`00_setup.sh --what registry --dry-run`: the installer runs in its own --dry-run, so the units it would install are printed
    RENDERED for this site (the store-mirror destination is the primary cluster's, which the store configuration declares), every enable is printed and none
    runs, registry-snapshot.timer is never named to an enable, and nothing is laid out or installed."""
    site = write_site(tmp_path)
    fake, rec, home = _fakes(tmp_path, "exit 0")
    r = run_stage("00_setup.sh", "--what", "registry", "--dry-run", site=site, env=_env(fake, home))
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    assert "dry_run=1" in r.stdout and "store_dest=cluster_a" in r.stdout, r.stdout[-2500:]      # a generic block: its CLUSTER_LABEL
    assert "# rendered registry-store-mirror.service:" in r.stdout and "#   ExecStart=" in r.stdout
    assert any(l.startswith("#   ExecStart=") and l.endswith(" stores mirror --dest cluster_a") for l in r.stdout.splitlines()), r.stdout[-2500:]
    assert "DRY-RUN: systemctl --user is-enabled registry-cluster-reads.timer || systemctl --user enable --now registry-cluster-reads.timer" in r.stdout
    assert not any("registry-snapshot.timer" in l for l in r.stdout.splitlines() if "enable --now" in l and not l.startswith("#")), r.stdout[-2500:]
    assert not rec.exists(), "a dry run calls no systemctl"
    assert not (tmp_path / "registry").exists() and not any((home / ".config/systemd/user").iterdir()), "a dry run writes nothing"
    assert_no_tokens(scrub_site(out, tmp=tmp_path), "00_setup.sh --what registry --dry-run")


def test_a_site_the_units_cannot_be_rendered_from_refuses_at_the_dry_run(tmp_path):
    """A registry root no unit file can carry as written (a space in it) passes the site loader but not the installer: the dry run
    refuses by name, before any enable is printed, instead of printing a command that would fail for real."""
    site = write_site(tmp_path, extra_config=f"SISTEER_REGISTRY_ROOT='{tmp_path / 'with space' / 'registry'}'")
    fake, rec, home = _fakes(tmp_path, "exit 0")
    r = run_stage("00_setup.sh", "--what", "registry", "--dry-run", site=site, env=_env(fake, home))
    assert r.returncode != 0, r.stdout[-1500:]
    assert "REFUSED: install:" in r.stderr and "SISTEER_REGISTRY_ROOT=" in r.stderr and "REFUSED: registry: install_grading_host.sh failed" in r.stderr, r.stderr[-1500:]
    assert "systemctl --user enable" not in r.stdout and "00_setup done" not in r.stdout
