"""grade_sampled.sh's analysis root: --dest, else F22S_DEST, else the wave's run-state tree under the site state root ($STATE,
which sisteer.site expands from SISTEER_STATE_ROOT in the environment, else in the site config file); with none of them the
script refuses by name. The script has no built-in home path.

Each run stops at the --org check, which comes right after the root is resolved, so no grading starts.

--detach re-runs the script as a user-manager unit, which inherits none of the invoking shell's variables: the unit must plan
under the root the invoking shell resolved, however that shell named it.
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
SCRIPT = REPO / "scripts" / "sdfbase" / "full22" / "sampled" / "grade_sampled.sh"
REFUSAL = "no analysis root"
PAST_ROOT = "--org must be lowprio or highprio"


def _run(tmp_path: Path, *args: str, **env_extra: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items()
           if k not in ("F22S_DEST", "SISTEER_STATE_ROOT", "SISTEER_CONFIG", "SISTEER_SITE_DIR")}
    env["SISTEER_SITE_DIR"] = str(tmp_path / "no-site")      # no site config unless a test names one
    env["HOME"] = str(tmp_path / "home")
    env.update(env_extra)
    return subprocess.run(["bash", str(SCRIPT), "static", "--org", "bogus", *args],
                          capture_output=True, text=True, env=env, timeout=60)


def test_no_root_anywhere_refuses_by_name(tmp_path):
    r = _run(tmp_path)
    assert r.returncode == 2 and REFUSAL in r.stderr and "SISTEER_STATE_ROOT" in r.stderr, r.stderr


def test_the_environment_names_the_state_root(tmp_path):
    r = _run(tmp_path, SISTEER_STATE_ROOT=str(tmp_path / "state"))
    assert r.returncode == 2 and REFUSAL not in r.stderr and PAST_ROOT in r.stderr, r.stderr


def test_the_site_config_names_the_state_root(tmp_path):
    cfg = tmp_path / "site" / "config.env"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("SISTEER_STATE_ROOT='~/state'\n")
    r = _run(tmp_path, SISTEER_CONFIG=str(cfg))
    assert r.returncode == 2 and REFUSAL not in r.stderr and PAST_ROOT in r.stderr, r.stderr


def test_dest_and_the_override_need_no_site(tmp_path):
    assert PAST_ROOT in _run(tmp_path, "--dest", str(tmp_path / "d")).stderr
    assert PAST_ROOT in _run(tmp_path, F22S_DEST=str(tmp_path / "d")).stderr


def test_no_built_in_home_path():
    text = SCRIPT.read_text()
    assert ".local/state" not in text


# --detach: the fake systemd-run runs the unit's command as the user manager would, with only PATH, HOME and the --setenv
# values, in the unit's working directory (the worktree). The stub interpreter records the analysis root the unit's planner
# receives (its third argument after the `-` that starts the planner) and refuses, so the unit stops at its first plan.
DETACH_STAMP = "20260101-000000"
DETACH_CASES = ["site config", "default site names another root", "state root", "override variable", "relative option"]


def _detach(tmp_path: Path, case: str):
    """Run `grade_sampled.sh static --detach` with the root named one way in the invoking shell. Returns the launcher's
    result, the fake's records, the roots the unit's planner received, and the root the launcher must resolve: exactly `want`, or
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
    stub.write_text(f'#!/bin/bash\n[ "${{1:-}}" = - ] && printf "%s\\n" "$4" >> "{seen}"\nexit 2\n')
    stub.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SISTEER_", "ANTHROPIC_")) and k != "F22S_DEST"}
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
        want = tmp_path / "d"
        env["F22S_DEST"] = str(want)
    elif case == "relative option":                        # relative to the invoking shell, not to the unit's worktree
        want = tmp_path / "rel" / "d"
        argv = ["--dest", "rel/d"]
    r = subprocess.run(["bash", str(SCRIPT), "static", "--detach", "--python", str(stub), "--stamp", DETACH_STAMP, *argv],
                       capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=120)
    records = [json.loads(line) for line in rec.read_text().splitlines()] if rec.exists() else []
    planned = seen.read_text().splitlines() if seen.exists() else []
    return r, records, planned, state, want


@pytest.mark.parametrize("case", DETACH_CASES)
def test_detached_unit_plans_under_the_root_the_shell_resolved(tmp_path, case):
    r, records, planned, state, want = _detach(tmp_path, case)
    assert r.returncode == 0 and "detached: unit=" in r.stdout, r.stdout + r.stderr
    m = re.search(rf"log=(/\S+)/logs/orchestrator-static-{DETACH_STAMP}\.log", r.stdout)
    assert m, r.stdout                                      # the launcher reports an absolute log path under the root it resolved
    root = Path(m.group(1))
    if want is not None:
        assert root == want, root
    else:
        assert str(root).startswith(f"{state}{os.sep}"), root
    log = root / "logs" / f"orchestrator-static-{DETACH_STAMP}.log"
    unit_out = log.read_text() if log.exists() else "(no log)"
    assert [x.get("exit") for x in records[1:]] == [2], (records, unit_out)     # the unit ran and stopped at the stub planner
    assert planned == [str(root)], (planned, unit_out)      # the unit planned under the launcher's root, not its own lookup
    assert REFUSAL not in unit_out, unit_out
