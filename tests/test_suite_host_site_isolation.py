"""The suite runs without the HOST's site config and without the host's installed registry units (tests/_host_site_isolation.py,
tests/conftest.py). Pins:

* the rule itself (``isolate``), on synthetic environments and a synthetic host site under a moved HOME: by default the host's
  ``$SISTEER_CONFIG`` / ``$SISTEER_SITE_DIR`` are dropped, the site directory is an empty one, and every suite default is exported
  even for a key the host's site config sets (the defect: the conftest stood down for those keys, so a test that moved HOME lost
  ``$STATE`` and ``$STAGE`` altogether); a value the environment sets stands; ``SISTEER_TEST_HOST_SITE=1`` restores the host's
  site and its keys, as before;
* that THIS run is isolated: no site resolves in-process or in a child, and the suite defaults expand;
* ``home_site_default``: a test of the ``~/.config/sisteer`` default under a moved HOME gets that default back;
* the installer's unit directory: ``install_grading_host.sh`` takes it from ``--unit-dir``, else ``$REGISTRY_UNIT_DIR``, else
  ``~/.config/systemd/user``; a host's installed hook-evaluating unit refuses a site that names no hook (what stage 00's dry run
  met on the grading host), the suite's empty ``$REGISTRY_UNIT_DIR`` keeps it out, and ``--unit-dir`` wins over both.

Synthetic sites, homes and units only; nothing reads the real ``~/.config``."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _host_site_isolation as HSI  # noqa: E402
from front_door_testlib import REPO, VENV_PY, write_site  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
from sisteer import site as S  # noqa: E402

INSTALL = REPO / "scripts" / "registry" / "install_grading_host.sh"
DEFAULTS = {"SISTEER_STATE_ROOT": "~/.local/state", "SISTEER_STAGE_DIR": "~/stage", "SISTEER_WORK_DIR": "~/work",
            "SISTEER_GRADE_DIR": "~/grade"}
GATE = "registry-gate-baseline.service"
#: an installed gate-baseline unit that evaluates an environment hook before its tick (the shape the grading host's units have)
HOOKED_EXEC = "/bin/sh -c 'eval \"`%h/.local/bin/secret-env`\"; exec /bin/true'"


def _host_site(home: Path, **keys: str) -> Path:
    """A host site config at <home>/.config/sisteer/config.env setting `keys` plus one cluster (the README's install location)."""
    d = home / ".config" / "sisteer"
    (d / "clusters").mkdir(parents=True)
    lines = [f"{k}='{v}'" for k, v in keys.items()] + ["SISTEER_CLUSTERS='hostcluster'", "SISTEER_PRIMARY_CLUSTER='hostcluster'"]
    (d / "config.env").write_text("\n".join(lines) + "\n")
    (d / "clusters" / "hostcluster.env").write_text("CLUSTER_SSH='login-host'\nCLUSTER_SHARED_ROOT='/host/shared'\nCLUSTER_LABEL='hostcluster'\n")
    return d / "config.env"


def _maker(tmp_path: Path):
    made = []

    def make() -> str:
        d = tmp_path / f"suite-site-{len(made)}"
        d.mkdir()
        made.append(d)
        return str(d)
    return make, made


# --------------------------------------------------------------------------------------------- the rule


def test_by_default_the_host_site_is_dropped_and_every_suite_default_is_exported(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _host_site(home, SISTEER_STATE_ROOT="/host/state", SISTEER_STAGE_DIR="/host/stage", SISTEER_REPO="/host/checkout")
    monkeypatch.setenv("HOME", str(home))                   # the host's site sits at the default location under this HOME
    assert HSI.host_site_keys({}) >= {"SISTEER_STATE_ROOT", "SISTEER_STAGE_DIR", "SISTEER_REPO", "SISTEER_CLUSTERS"}
    env = {"PATH": "/usr/bin", "HOME": str(home)}
    make, made = _maker(tmp_path)
    got = HSI.isolate(env, DEFAULTS, make)
    assert got == {"mode": "isolated", "site_dir": str(made[0]), "dropped": {}, "defaulted": list(DEFAULTS)}
    assert env[HSI.SITE_DIR_ENV] == str(made[0]) and HSI.CONFIG_ENV not in env
    assert {k: env[k] for k in DEFAULTS} == DEFAULTS       # the host file sets two of them: they are exported all the same
    # the reader finds no site at all: no cluster, no host value, the fresh-clone state
    site = S.load_site(env=env)
    assert site.path is None and dict(site.config) == {} and site.labels == []
    assert S.setting("SISTEER_REPO", env=env) is None
    assert S.expand("$STATE/x", env=env) == "~/.local/state/x" and S.expand("$STAGE/y", env=env) == "~/stage/y"
    with pytest.raises(S.PlaceholderError, match=r"\$SHARED"):
        S.expand("$SHARED/z", env=env)
    # and a test that then moves HOME keeps $STATE, under ITS home (the defect: no SISTEER_STATE_ROOT anywhere, a PlaceholderError)
    moved = tmp_path / "relocated-home"
    monkeypatch.setenv("HOME", str(moved))
    assert S.expand_path("$STATE/lane/run-state", env=env) == moved / ".local/state/lane/run-state"


def test_the_old_behaviour_is_the_opt_in_and_it_is_what_lost_state_under_a_moved_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _host_site(home, SISTEER_STATE_ROOT="/host/state", SISTEER_STAGE_DIR="/host/stage")
    monkeypatch.setenv("HOME", str(home))
    env = {"PATH": "/usr/bin", "HOME": str(home), HSI.OPT_IN_ENV: "1"}
    make, made = _maker(tmp_path)
    got = HSI.isolate(env, DEFAULTS, make)
    assert got == {"mode": "host", "site_dir": None, "dropped": {}, "defaulted": ["SISTEER_WORK_DIR", "SISTEER_GRADE_DIR"]}
    assert not made and HSI.SITE_DIR_ENV not in env and "SISTEER_STATE_ROOT" not in env and "SISTEER_STAGE_DIR" not in env
    assert S.load_site(env=env).labels == ["hostcluster"] and S.expand("$STATE/x", env=env) == "/host/state/x"   # the host's site wins
    monkeypatch.setenv("HOME", str(tmp_path / "relocated-home"))                                                # ...until HOME moves
    with pytest.raises(S.PlaceholderError, match=r"\$STATE .*SISTEER_STATE_ROOT"):
        S.expand("$STATE/x", env=env)


def test_a_value_the_environment_sets_stands_and_a_caller_named_site_is_dropped(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    named = _host_site(tmp_path / "elsewhere", SISTEER_STATE_ROOT="/named/state")
    env = {"SISTEER_STAGE_DIR": "/mine/stage", "SISTEER_WORK_DIR": "", HSI.CONFIG_ENV: str(named), HSI.SITE_DIR_ENV: str(named.parent)}
    make, made = _maker(tmp_path)
    got = HSI.isolate(env, DEFAULTS, make)
    assert got["dropped"] == {HSI.CONFIG_ENV: str(named), HSI.SITE_DIR_ENV: str(named.parent)}
    assert env[HSI.SITE_DIR_ENV] == str(made[0]) and HSI.CONFIG_ENV not in env
    assert env["SISTEER_STAGE_DIR"] == "/mine/stage"                               # set by the caller: it stands
    assert env["SISTEER_WORK_DIR"] == "~/work"                                     # empty counts as unset
    assert got["defaulted"] == ["SISTEER_STATE_ROOT", "SISTEER_WORK_DIR", "SISTEER_GRADE_DIR"]


# --------------------------------------------------------------------------------------------- this run


def _isolated_run() -> bool:
    return os.environ.get(HSI.OPT_IN_ENV) != "1"


@pytest.mark.skipif(not _isolated_run(), reason="this run opted into the host's site (SISTEER_TEST_HOST_SITE=1)")
def test_this_run_resolves_no_site_in_process_or_in_a_child():
    sd = Path(os.environ[HSI.SITE_DIR_ENV])
    assert HSI.CONFIG_ENV not in os.environ
    assert sd.is_dir() and list(sd.iterdir()) == [] and sd != Path(os.path.expanduser(HSI.DEFAULT_SITE_DIR))
    if os.geteuid() != 0:
        assert not os.access(sd, os.W_OK)                                          # a write into "the site" fails loudly
    site = S.load_site()
    assert site.path is None and dict(site.config) == {} and site.labels == []
    assert S.config_path() == str(sd / "config.env")
    for k, v in DEFAULTS.items():
        assert os.environ.get(k), k                                                # exported (or the caller's own value)
    # a child resolves the same: no site config, the defaults expand, $SHARED refuses by name
    r = subprocess.run([VENV_PY, "-m", "sisteer.site", "show"], capture_output=True, text=True, timeout=60,
                       env=dict(os.environ, PYTHONPATH=str(REPO / "src")))
    assert r.returncode == 0, r.stderr
    rows = dict(ln.split("\t", 1) for ln in r.stdout.splitlines() if "\t" in ln)
    if not (os.environ.get("SISTEER_SHARED_ROOT") or os.environ.get("CLUSTER_SHARED_ROOT")):   # a job env may export the root itself
        assert rows["$SHARED"].startswith("UNSET:") and "none on this host" in rows["$SHARED"]
    assert not rows["$STATE"].startswith("UNSET") and not rows["$GRADE"].startswith("UNSET")


def test_home_site_default_gives_a_moved_home_its_own_default_site(home_site_default, tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    cfg = _host_site(home, SISTEER_STATE_ROOT="/home-site/state")
    for var in (HSI.CONFIG_ENV, HSI.SITE_DIR_ENV, "REGISTRY_UNIT_DIR"):
        assert var not in os.environ, var
    site = S.load_site()
    assert site.path == str(cfg) and site.labels == ["hostcluster"] and site.config["SISTEER_STATE_ROOT"] == "/home-site/state"
    assert S.config_path() == str(cfg)


@pytest.mark.skipif(not _isolated_run(), reason="this run opted into the host's site (SISTEER_TEST_HOST_SITE=1)")
def test_this_runs_unit_directory_is_an_empty_one():
    units = Path(os.environ["REGISTRY_UNIT_DIR"])
    assert units.is_dir() and list(units.iterdir()) == [] and units != Path(os.path.expanduser("~/.config/systemd/user"))


# --------------------------------------------------------------------------------------------- the installer's unit directory


def _hooked_home(home: Path) -> Path:
    """`home` with an installed gate-baseline unit whose ExecStart evaluates an environment hook (an earlier install's)."""
    units = home / ".config" / "systemd" / "user"
    units.mkdir(parents=True)
    (units / GATE).write_text(f"[Unit]\nDescription=gate baseline\n\n[Service]\nType=oneshot\nExecStart={HOOKED_EXEC}\n")
    return units


def _installer_env(tmp_path: Path, home: Path, **extra: str) -> dict:
    """The installer's environment as stage 00 gives it: a synthetic site that names NO environment hook, the store configuration
    declaring its destination, HOME the synthetic home, and nothing else (no unit directory unless `extra` names one)."""
    site = write_site(tmp_path / "s")
    stores = tmp_path / "cluster_reads.yaml"
    stores.write_text("destinations:\n  cluster_a:\n    stores_rel: registry/stores\n")
    env = {"PATH": os.environ["PATH"], "HOME": str(home), "REGISTRY_PY": VENV_PY, "REGISTRY_PYTHONPATH": str(REPO / "src"),
           "SISTEER_CONFIG": str(site / "config.env"), "REGISTRY_CLUSTER_READS": str(stores)}
    env.update(extra)
    return env


def _dry_install(env: dict, *args: str, root: Path) -> subprocess.CompletedProcess:
    # stage 00's call: --no-enable --root --config, and its own --dry-run
    return subprocess.run(["bash", str(INSTALL), "--no-enable", "--root", str(root), "--config", env["SISTEER_CONFIG"], "--dry-run", *args],
                          capture_output=True, text=True, env=env, timeout=120, cwd="/")


def test_the_installer_reads_its_unit_directory_from_the_flag_then_the_variable_then_home(tmp_path):
    home = tmp_path / "home"
    hooked = _hooked_home(home)
    before = (hooked / GATE).read_bytes()
    root = tmp_path / "registry"
    # no flag, no variable: the HOME default, whose installed hooked unit refuses a site that names no hook -- what the grading host's
    # own units did to every test that dry-ran stage 00 against a synthetic site
    r = _dry_install(_installer_env(tmp_path, home), root=root)
    assert r.returncode == 2 and "REFUSED: install:" in r.stderr and str(hooked / GATE) in r.stderr and "SISTEER_UNIT_ENV_HOOK" in r.stderr, \
        r.stdout[-1200:] + r.stderr[-1200:]
    # $REGISTRY_UNIT_DIR at an empty directory (the suite's belt): nothing installed there, so nothing refuses; the plan names it
    empty = tmp_path / "empty-units"; empty.mkdir()
    r = _dry_install(_installer_env(tmp_path, home, REGISTRY_UNIT_DIR=str(empty)), root=root)
    assert r.returncode == 0, r.stdout[-1200:] + r.stderr[-1200:]
    assert f"units={empty} " in r.stdout and f"+ install -m 0644" in r.stdout and str(hooked) not in r.stdout
    # the variable naming the hooked directory refuses the same way the HOME default did
    r = _dry_install(_installer_env(tmp_path, tmp_path / "other-home", REGISTRY_UNIT_DIR=str(hooked)), root=root)
    assert r.returncode == 2 and str(hooked / GATE) in r.stderr, r.stderr[-1200:]
    # --unit-dir wins over the variable
    flag = tmp_path / "flag-units"
    r = _dry_install(_installer_env(tmp_path, home, REGISTRY_UNIT_DIR=str(hooked)), "--unit-dir", str(flag), root=root)
    assert r.returncode == 0 and f"units={flag} " in r.stdout, r.stdout[-1200:] + r.stderr[-1200:]
    # a dry run wrote nothing anywhere
    assert (hooked / GATE).read_bytes() == before and sorted(p.name for p in hooked.iterdir()) == [GATE]
    assert not root.exists() and list(empty.iterdir()) == [] and not flag.exists()
