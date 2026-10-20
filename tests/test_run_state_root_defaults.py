"""The seed-replicate battery grader and the derived-chain submitter take their run-state root from an option, their override
variable, or the wave's run-state tree under the site state root ($STATE, which sisteer.site expands from SISTEER_STATE_ROOT in
the environment, else in the site config); with none of them they refuse by name. Neither script carries a built-in home path.

grade_battery.sh stops at its --org check and submit_derived_chain.sh at its --sha check, both right after the root is resolved,
so nothing runs.

grade_battery.sh --detach re-runs the script as a user-manager unit, which inherits none of the invoking shell's variables: the
unit must plan under the root the invoking shell resolved, however that shell named it.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import write_fake_systemd_run  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SEEDREP = REPO / "scripts" / "sdfbase" / "seedrep"
BATTERY = SEEDREP / "grade_battery.sh"
CHAIN = SEEDREP / "submit_derived_chain.sh"
CASES = {
    # script: (argv after the script, the override variable, the check that follows the root, the refusal's words)
    "battery": (BATTERY, ["static", "--org", "bogus"], "SEEDREP_RS", "--org must be lowprio or highprio", "no run-state root"),
    "chain": (CHAIN, ["synthetic_key", "synthetic_bank"], "RS", "--sha must be the 40-hex commit", "no RS root"),
}


def _run(tmp_path: Path, which: str, *extra: str, **env_extra: str) -> subprocess.CompletedProcess:
    script, argv, override, _, _ = CASES[which]
    env = {k: v for k, v in os.environ.items()
           if k not in ("SISTEER_STATE_ROOT", "SISTEER_CONFIG", "SISTEER_SITE_DIR", "SEEDREP_RS", "RS")}
    env["SISTEER_SITE_DIR"] = str(tmp_path / "no-site")
    env["HOME"] = str(tmp_path / "home")
    env.update(env_extra)
    return subprocess.run(["bash", str(script), *argv, *extra], capture_output=True, text=True, env=env, timeout=60)


@pytest.mark.parametrize("which", sorted(CASES))
def test_no_root_anywhere_refuses_by_name(tmp_path, which):
    r = _run(tmp_path, which)
    assert r.returncode == 2 and CASES[which][4] in r.stderr and "SISTEER_STATE_ROOT" in r.stderr, r.stderr


@pytest.mark.parametrize("which", sorted(CASES))
def test_the_environment_names_the_root(tmp_path, which):
    r = _run(tmp_path, which, SISTEER_STATE_ROOT=str(tmp_path / "state"))
    assert CASES[which][4] not in r.stderr and CASES[which][3] in r.stderr, r.stderr


@pytest.mark.parametrize("which", sorted(CASES))
def test_the_site_config_names_the_root(tmp_path, which):
    cfg = tmp_path / "site" / "config.env"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("SISTEER_STATE_ROOT='~/state'\n")
    r = _run(tmp_path, which, SISTEER_CONFIG=str(cfg))
    assert CASES[which][4] not in r.stderr and CASES[which][3] in r.stderr, r.stderr


@pytest.mark.parametrize("which", sorted(CASES))
def test_the_option_and_the_override_need_no_site(tmp_path, which):
    assert CASES[which][3] in _run(tmp_path, which, "--rs", str(tmp_path / "rs")).stderr
    assert CASES[which][3] in _run(tmp_path, which, **{CASES[which][2]: str(tmp_path / "rs")}).stderr


@pytest.mark.parametrize("script", [BATTERY, CHAIN])
def test_no_built_in_home_path(script):
    assert ".local/state" not in script.read_text()


# --detach: the fake systemd-run runs the unit's command as the user manager would, with only PATH, HOME and the --setenv
# values, in the unit's working directory (the worktree). The stub interpreter records the run-state root the unit's planner
# receives and refuses, so the unit stops at its first planner call.
DETACH_STAMP = "20260101-000000"
DETACH_CASES = ["site config", "default site names another root", "state root", "override variable", "relative option"]


def _detach(tmp_path: Path, case: str):
    """Run `grade_battery.sh prepare --detach` with the root named one way in the invoking shell. Returns the launcher's result,
    the fake's records, the roots the unit's planner received, and the root the launcher must resolve: exactly `want`, or
    (when `want` is None) one under the site state root the shell named."""
    home = tmp_path / "home"
    home.mkdir()
    state = tmp_path / "state"
    want = None                                            # the site cases: under the state root, wherever the launcher resolved
    site = tmp_path / "site" / "config.env"
    site.parent.mkdir()
    site.write_text(f"SISTEER_STATE_ROOT='{state}'\n")
    rec = tmp_path / "systemd.jsonl"
    fake = write_fake_systemd_run(tmp_path / "bin", rec, run=True)
    seen = tmp_path / "planner_saw.txt"
    stub = tmp_path / "py"
    stub.write_text(f'#!/bin/bash\n[ -n "${{BAT_RS:-}}" ] && printf "%s\\n" "$BAT_RS" >> "{seen}"\nexit 2\n')
    stub.chmod(0o755)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SISTEER_", "ANTHROPIC_", "BAT_")) and k not in ("SEEDREP_RS", "RS")}
    env.update(PATH=f"{fake.parent}:{os.environ['PATH']}", HOME=str(home))
    argv = []
    if case == "site config":
        env["SISTEER_CONFIG"] = str(site)
    elif case == "default site names another root":       # the unit, reading the default site, would grade another tree
        env["SISTEER_CONFIG"] = str(site)
        default = home / ".config" / "sisteer"
        default.mkdir(parents=True)
        (default / "config.env").write_text(f"SISTEER_STATE_ROOT='{tmp_path / 'other'}'\n")
    elif case == "state root":
        env["SISTEER_STATE_ROOT"] = str(state)
    elif case == "override variable":
        want = tmp_path / "rs"
        env["SEEDREP_RS"] = str(want)
    elif case == "relative option":                        # relative to the invoking shell, not to the unit's worktree
        want = tmp_path / "rel" / "rs"
        argv = ["--rs", "rel/rs"]
    r = subprocess.run(["bash", str(BATTERY), "prepare", "--detach", "--python", str(stub), "--stamp", DETACH_STAMP, *argv],
                       capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=120)
    records = [json.loads(line) for line in rec.read_text().splitlines()] if rec.exists() else []
    planned = seen.read_text().splitlines() if seen.exists() else []
    return r, records, planned, state, want


@pytest.mark.parametrize("case", DETACH_CASES)
def test_detached_unit_plans_under_the_root_the_shell_resolved(tmp_path, case):
    r, records, planned, state, want = _detach(tmp_path, case)
    assert r.returncode == 0 and "detached: unit=" in r.stdout, r.stdout + r.stderr
    m = re.search(rf"log=(/\S+)/grades/logs/orchestrator-prepare-{DETACH_STAMP}\.log", r.stdout)
    assert m, r.stdout                                      # the launcher reports an absolute log path under the root it resolved
    root = Path(m.group(1))
    if want is not None:
        assert root == want, root
    else:
        assert str(root).startswith(f"{state}{os.sep}"), root
    log = root / "grades" / "logs" / f"orchestrator-prepare-{DETACH_STAMP}.log"
    unit_out = log.read_text() if log.exists() else "(no log)"
    assert [x.get("exit") for x in records[1:]] == [2], (records, unit_out)     # the unit ran and stopped at the stub planner
    assert planned == [str(root)], (planned, unit_out)      # the unit planned under the launcher's root, not its own lookup
    assert "no run-state root" not in unit_out, unit_out
