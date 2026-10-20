"""The SHELL half of the two-modes contract (the env-mode rule): the four real grading launchers under
SISTEER_SECRETS_MODE=env with the secret-manager CLI ABSENT from PATH, through a fake `systemd-run` that records argv
and the environment the unit would receive and EXECUTES the unit command with exactly that environment.
  * launch_judge.sh: name-only `--setenv=ANTHROPIC_API_KEY` (never `=value`), the key present in the unit env, absent from
    argv and from the log, CONSUMED by the unit (`: "${ANTHROPIC_API_KEY:?}"` passes and the stub judge sees it); unset -> a
    by-name failure.
  * grade_sampled.sh / grade_battery.sh: their `--detach` re-exec line carries the same name-only imports; their unit-script
    generators emit `: "${ANTHROPIC_API_KEY:?…}"` under env mode and the secret-manager line under op mode (pinned from the
    generator source, since running their planners needs a whole world).
  * petri_drive_full22.sh's env-mode key check is covered by tests/test_front_door_bootstrap_portability.py.
  * lib/secrets.sh: secrets_unit_flags carries only the secret-manager CLI's path under op mode (SISTEER_SECRET_CMD, a value, never
    a key; nothing when the setting is unset) and name-only imports under env mode.
The Python half (sisteer.vault) is tests/test_vault_env_mode.py."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import write_python3_shim  # noqa: E402
from front_door_testlib import LIB, REPO, write_fake_systemd_run  # noqa: E402

F22 = REPO / "scripts/sdfbase/full22"
#: the command a generated op-mode key line runs: the secret-manager CLI the unit's environment names, refused by name when unset
KEY_CMD = '"${SISTEER_SECRET_CMD:?SISTEER_SECRET_CMD must name the secret-manager CLI in the unit environment}"'


def _fake_cli(d: Path, marker: Path = None) -> Path:
    """A fake secret-manager CLI in ``d`` (a neutral name; a marker line per call when ``marker`` is given)."""
    d.mkdir(parents=True, exist_ok=True)
    f = d / "fake-secret-cmd"
    f.write_text("#!/bin/sh\n" + (f'echo "ran $*" >> {marker}\n' if marker else "") + 'printf "%s" never-a-real-value\n'); f.chmod(0o755)
    return f


def _env(tmp_path, **extra):
    fake = tmp_path / "bin"
    e = dict(os.environ, PATH=f"{fake}:/usr/bin:/bin", HOME=str(tmp_path / "home"), SISTEER_SECRETS_MODE="env")
    for v in ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_LOWPRIO", "OPENROUTER_API_KEY", "SISTEER_SECRET_CMD"):
        e.pop(v, None)
    e.update(extra)
    (tmp_path / "home").mkdir(exist_ok=True)
    # a `python3` on this restricted PATH is the interpreter running the tests: /usr/bin:/bin need not carry one (the CPU-suite
    # container's lives elsewhere), and the surfaces under test may resolve it by name; no secret-manager CLI is named
    write_python3_shim(fake)
    assert ("SISTEER_SECRET_CMD" in e) == ("SISTEER_SECRET_CMD" in extra)  # unless a test names one
    return e


def _wt(tmp_path):
    """A worktree stub with the REAL launch_judge.sh + lib/secrets.sh and a stub judge that records whether it saw the key."""
    wt = tmp_path / "wt"; (wt / "scripts/sdfbase/full22/judge").mkdir(parents=True); (wt / "reproduce/lib").mkdir(parents=True); (wt / ".venv/bin").mkdir(parents=True)
    shutil.copy(F22 / "judge/launch_judge.sh", wt / "scripts/sdfbase/full22/judge/launch_judge.sh")
    shutil.copy(LIB / "secrets.sh", wt / "reproduce/lib/secrets.sh")
    (wt / "scripts/served_capability_judge.py").write_text("")
    seen = tmp_path / "judge_saw.txt"
    (wt / ".venv/bin/python").write_text(f'#!/bin/bash\n# stub judge: record whether the key reached the process, never its value\nif [ -n "${{ANTHROPIC_API_KEY:-}}" ]; then echo present > {seen}; else echo absent > {seen}; fi\nexit 0\n')
    (wt / ".venv/bin/python").chmod(0o755)
    return wt, seen


def test_launch_judge_env_mode_name_only_import_consumed_by_the_unit(tmp_path):
    wt, seen = _wt(tmp_path)
    rec = tmp_path / "systemd.log"; write_fake_systemd_run(tmp_path / "bin", rec, run=True)
    root = tmp_path / "grade"; (root / "logs").mkdir(parents=True)
    env = _env(tmp_path, ANTHROPIC_API_KEY="synthetic-secret-value", F22_GRADE_ROOT=str(root))
    r = subprocess.run(["bash", str(wt / "scripts/sdfbase/full22/judge/launch_judge.sh"), "--targets", "x"], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    calls = [json.loads(l) for l in rec.read_text().splitlines()]
    launch = calls[0]
    assert "--setenv=ANTHROPIC_API_KEY" in launch["argv"] and "--setenv=SISTEER_SECRETS_MODE" in launch["argv"]
    assert not any(a.startswith("--setenv=ANTHROPIC_API_KEY=") for a in launch["argv"])          # NAME-ONLY, never the value
    assert "synthetic-secret-value" not in json.dumps(launch["argv"])
    assert launch["unit_env_has_key"] is True and "ANTHROPIC_API_KEY" in launch["unit_env_keys"]
    assert calls[1]["exit"] == 0                                                                  # the unit command ran to completion
    assert seen.read_text().strip() == "present"                                                 # CONSUMED by the judge
    logs = [p for p in (root / "logs").glob("judge-*.log") if not p.is_symlink()]
    assert logs and "synthetic-secret-value" not in logs[0].read_text()
    assert "synthetic-secret-value" not in r.stdout + r.stderr


def test_launch_judge_env_mode_unset_key_fails_by_name(tmp_path):
    wt, seen = _wt(tmp_path)
    rec = tmp_path / "systemd.log"; write_fake_systemd_run(tmp_path / "bin", rec, run=True)
    root = tmp_path / "grade"; (root / "logs").mkdir(parents=True)
    env = _env(tmp_path, F22_GRADE_ROOT=str(root))                       # ANTHROPIC_API_KEY unset
    r = subprocess.run(["bash", str(wt / "scripts/sdfbase/full22/judge/launch_judge.sh"), "--targets", "x"], capture_output=True, text=True, env=env, timeout=60)
    calls = [json.loads(l) for l in rec.read_text().splitlines()]
    assert calls[0]["unit_env_has_key"] is False and calls[1]["exit"] != 0
    assert not seen.exists()                                              # the judge never started
    # the by-name refusal fires BEFORE the unit's log redirect, so it reaches the unit's stderr (the fake inherits ours)
    assert "ANTHROPIC_API_KEY must be set in the unit environment" in r.stderr, r.stderr[-600:]


def test_launch_judge_op_mode_carries_only_the_cli_path(tmp_path):
    """op mode: the unit's line runs the secret-manager CLI its environment names; the launcher carries that CLI's ABSOLUTE path
    (a bare name resolved on PATH) and nothing else, and carries nothing when the setting is unset (the unit then refuses by name)."""
    wt, _ = _wt(tmp_path)
    rec = tmp_path / "systemd.log"; write_fake_systemd_run(tmp_path / "bin", rec, run=False)
    root = tmp_path / "grade"; (root / "logs").mkdir(parents=True)
    env = _env(tmp_path, F22_GRADE_ROOT=str(root)); env.pop("SISTEER_SECRETS_MODE")
    subprocess.run(["bash", str(wt / "scripts/sdfbase/full22/judge/launch_judge.sh")], capture_output=True, text=True, env=env, timeout=60)
    launch = json.loads(rec.read_text().splitlines()[0])
    assert not any(a.startswith("--setenv=") for a in launch["argv"])   # op mode, no CLI named: no flags at all
    assert f'export ANTHROPIC_API_KEY="$({KEY_CMD} "$F22_ORG")"' in " ".join(launch["cmd"])
    cli = _fake_cli(tmp_path / "bin")
    for named in (str(cli), cli.name):                                   # an absolute path, and a bare name on PATH
        rec.unlink()
        subprocess.run(["bash", str(wt / "scripts/sdfbase/full22/judge/launch_judge.sh")], capture_output=True, text=True,
                       env=dict(env, SISTEER_SECRET_CMD=named), timeout=60)
        launch = json.loads(rec.read_text().splitlines()[0])
        assert [a for a in launch["argv"] if a.startswith("--setenv=")] == [f"--setenv=SISTEER_SECRET_CMD={cli}"], launch["argv"]


@pytest.mark.parametrize("script,phase", [("scripts/sdfbase/full22/sampled/grade_sampled.sh", "static"), ("scripts/sdfbase/seedrep/grade_battery.sh", "static")])
def test_detach_relaunch_carries_name_only_imports(tmp_path, script, phase):
    rec = tmp_path / "systemd.log"; write_fake_systemd_run(tmp_path / "bin", rec, run=False)
    env = _env(tmp_path, ANTHROPIC_API_KEY="synthetic-secret-value")
    stub_py = tmp_path / "py"; stub_py.write_text("#!/bin/bash\nexit 0\n"); stub_py.chmod(0o755)
    r = subprocess.run(["bash", str(REPO / script), phase, "--detach", "--python", str(stub_py), "--dest" if "sampled" in script else "--rs", str(tmp_path / "dest")],
                       capture_output=True, text=True, env=env, cwd=str(REPO), timeout=120)
    assert "detached: unit=" in r.stdout, r.stdout + r.stderr
    launch = json.loads(rec.read_text().splitlines()[0])
    assert "--setenv=ANTHROPIC_API_KEY" in launch["argv"] and "--setenv=SISTEER_SECRETS_MODE" in launch["argv"]
    assert not any("synthetic-secret-value" in a for a in launch["argv"])
    assert any(a.startswith("--setenv=PATH=") for a in launch["argv"])   # the existing PATH forward is untouched


@pytest.mark.parametrize("script", ["scripts/sdfbase/full22/sampled/grade_sampled.sh", "scripts/sdfbase/seedrep/grade_battery.sh"])
def test_unit_script_generators_emit_the_env_mode_line(script):
    src = (REPO / script).read_text()
    assert '\': "${ANTHROPIC_API_KEY:?SISTEER_SECRETS_MODE=env: ANTHROPIC_API_KEY must be set in the unit environment (one key serves every org role)}"; unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN\'' in src
    assert 'export {env_var}="$("${{SISTEER_SECRET_CMD:?' in src                               # op mode: the org's own variable, through the CLI
    assert "ANTHROPIC_API_KEY_LOWPRIO" in src and src.count('uflags="$(secrets_unit_flags)"') == 2      # both systemd-run sites capture the flags…
    assert src.count("secrets_unit_flags_cleanup") >= 3                                        # …and unlink the EnvironmentFile after each


def test_launch_judge_envfile_fallback_is_consumed_then_unlinked(tmp_path):
    """Finding 12: on systemd < 247 the key rides a 0600 EnvironmentFile under $XDG_RUNTIME_DIR; the unit reads it at start and the
    launcher unlinks it right after — a secret-bearing file never outlives the unit's start."""
    wt, seen = _wt(tmp_path)
    rec = tmp_path / "systemd.log"; write_fake_systemd_run(tmp_path / "bin", rec, run=True)
    root = tmp_path / "grade"; (root / "logs").mkdir(parents=True)
    rt = tmp_path / "rt"; rt.mkdir()
    env = _env(tmp_path, ANTHROPIC_API_KEY="synthetic-secret-value", F22_GRADE_ROOT=str(root), SISTEER_SYSTEMD_FORCE_ENVFILE="1", XDG_RUNTIME_DIR=str(rt))
    r = subprocess.run(["bash", str(wt / "scripts/sdfbase/full22/judge/launch_judge.sh"), "--targets", "x"], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    launch = json.loads(rec.read_text().splitlines()[0])
    assert launch["props"]["EnvironmentFile"].startswith(str(rt)) and launch["props"]["envfile_present_at_start"] is True
    assert not any(a.startswith("--setenv=ANTHROPIC") for a in launch["argv"])                # the fallback, not the name-only import
    assert launch["unit_env_has_key"] is True and seen.read_text().strip() == "present"        # consumed by the judge
    assert list(rt.glob("sisteer-unit-env.*")) == []                                          # and unlinked once the unit started
    assert "synthetic-secret-value" not in r.stdout + r.stderr


@pytest.mark.parametrize("script", ["scripts/sdfbase/full22/sampled/grade_sampled.sh", "scripts/sdfbase/seedrep/grade_battery.sh"])
def test_detach_relaunch_envfile_fallback_is_unlinked(tmp_path, script):
    rec = tmp_path / "systemd.log"; write_fake_systemd_run(tmp_path / "bin", rec, run=False)
    rt = tmp_path / "rt"; rt.mkdir()
    env = _env(tmp_path, ANTHROPIC_API_KEY="synthetic-secret-value", SISTEER_SYSTEMD_FORCE_ENVFILE="1", XDG_RUNTIME_DIR=str(rt))
    stub_py = tmp_path / "py"; stub_py.write_text("#!/bin/bash\nexit 0\n"); stub_py.chmod(0o755)
    r = subprocess.run(["bash", str(REPO / script), "static", "--detach", "--python", str(stub_py), "--dest" if "sampled" in script else "--rs", str(tmp_path / "dest")],
                       capture_output=True, text=True, env=env, cwd=str(REPO), timeout=120)
    assert "detached: unit=" in r.stdout, r.stdout + r.stderr
    launch = json.loads(rec.read_text().splitlines()[0])
    assert launch["props"]["EnvironmentFile"].startswith(str(rt)) and launch["props"]["envfile_present_at_start"] is True
    assert list(rt.glob("sisteer-unit-env.*")) == []


def test_secrets_sh_unit_flags(tmp_path):
    lib = LIB / "secrets.sh"
    # secrets_unit_flags reads the systemd version off `systemd-run --version` (secrets_systemd_version); the recording fake answers
    # 255 ahead of whatever the host carries, so the name-only branch is exercised wherever the suite runs (a host with no systemd at
    # all would otherwise take the EnvironmentFile fallback and refuse for want of XDG_RUNTIME_DIR); the forced fallback below is the
    # same fake
    fake_path = f"{write_fake_systemd_run(tmp_path / 'bin', tmp_path / 'systemd.log', run=False).parent}:{os.environ['PATH']}"
    base = {k: v for k, v in os.environ.items() if k != "SISTEER_SECRET_CMD"}
    op = subprocess.run(["bash", "-c", f"source {lib}; secrets_unit_flags; echo \"[$(secrets_unit_key_line lowprio)]\""], capture_output=True, text=True,
                        env=dict(base, SISTEER_SECRETS_MODE="op")).stdout
    assert op.startswith(f'[export ANTHROPIC_API_KEY_LOWPRIO="$({KEY_CMD} lowprio)"') and "--setenv" not in op
    cli = _fake_cli(tmp_path / "cli")
    op = subprocess.run(["bash", "-c", f"source {lib}; secrets_unit_flags"], capture_output=True, text=True,
                        env=dict(base, SISTEER_SECRETS_MODE="op", SISTEER_SECRET_CMD=str(cli))).stdout
    assert op == f"--setenv=SISTEER_SECRET_CMD={cli}", op               # the CLI's path, a value: never a key
    e = dict(os.environ, SISTEER_SECRETS_MODE="env", ANTHROPIC_API_KEY="k", OPENROUTER_API_KEY="o")
    e["PATH"] = fake_path
    env_out = subprocess.run(["bash", "-c", f"source {lib}; secrets_unit_flags; echo; secrets_unit_key_line batch"], capture_output=True, text=True, env=e).stdout
    assert env_out.splitlines()[0] == "--setenv=SISTEER_SECRETS_MODE --setenv=ANTHROPIC_API_KEY --setenv=OPENROUTER_API_KEY"
    assert env_out.splitlines()[1].startswith(': "${ANTHROPIC_API_KEY:?')
    assert "k" not in env_out.splitlines()[0]
    # the systemd < 247 fallback: a 0600 file under XDG_RUNTIME_DIR, never under the tree
    rt = tmp_path / "rt"; rt.mkdir()
    fb = subprocess.run(["bash", "-c", f"source {lib}; secrets_unit_flags"], capture_output=True, text=True,
                        env=dict(e, SISTEER_SYSTEMD_FORCE_ENVFILE="1", XDG_RUNTIME_DIR=str(rt))).stdout
    m = re.match(r"-p EnvironmentFile=(\S+)", fb); assert m, fb
    f = Path(m.group(1)); assert f.parent == rt and oct(f.stat().st_mode & 0o777) == "0o600" and "ANTHROPIC_API_KEY=k" in f.read_text()
    # env mode: every org role -> the one key (secret_anthropic); the secret-manager CLI never consulted
    for org in ("lowprio", "highprio", "batch"):
        v = subprocess.run(["bash", "-c", f"source {lib}; secret_anthropic {org}"], capture_output=True, text=True, env=dict(e, PATH="/usr/bin:/bin")).stdout
        assert v == "k", org


# ------------------------------------------------------------------------------------------------- wif (round 2 on 4574e103c)
SECRET_SENTINEL = "synthetic-secret-value-must-not-travel"


def _env_wif(tmp_path, **extra):
    """_env's restricted PATH under SISTEER_SECRETS_MODE=wif, with a fake secret-manager CLI ON the PATH and at ~/.local/bin (the two
    places a unit resolves it) that leaves a marker when invoked -- the assertion is that nothing invokes it -- and the host's
    ANTHROPIC_* / SISTEER_* / HF_* / OPENROUTER_* / OP_* scrubbed."""
    fake = tmp_path / "bin"
    e = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "SISTEER_", "HF_", "OPENROUTER_", "OP_"))}
    e.update(PATH=f"{fake}:/usr/bin:/bin", HOME=str(tmp_path / "home"), SISTEER_SECRETS_MODE="wif")
    e.update(extra)
    write_python3_shim(fake)
    marker = tmp_path / "secret-cli-ran"
    for d in (fake, tmp_path / "home" / ".local" / "bin"):
        _fake_cli(d, marker)
    e["SISTEER_SECRET_CMD"] = str(fake / "fake-secret-cmd")                 # named, so a call would leave the marker
    return e, marker


def test_launch_judge_wif_mode_carries_the_mode_and_no_key_and_never_runs_the_secret_cli(tmp_path):
    """Under SISTEER_SECRETS_MODE=wif the unit once received NO mode (secrets_unit_flags printed nothing), so its bash defaulted to op
    -- ran the secret-manager CLI for a key that no longer exists and exported it -- and its vault ran under op too. Now the flags carry
    `--setenv=SISTEER_SECRETS_MODE=wif` (a value, not a secret; the site's ids directory rides the same way), the unit's line only
    SCRUBS the inherited ANTHROPIC_*, the secret-manager CLI is never invoked, and the judge runs under wif with no key. A non-op mode
    with the library unreadable is refused before systemd-run."""
    wt, seen = _wt(tmp_path)
    (wt / ".venv/bin/python").write_text('#!/bin/bash\n# stub judge: the mode it runs under and whether a key reached it, never a value\n'
                                         'echo "mode=${SISTEER_SECRETS_MODE:-unset} key=$([ -n "${ANTHROPIC_API_KEY+x}" ] && echo set || echo unset) '
                                         'dir=${SISTEER_WIF_ENV_DIR:-unset}" > ' + str(seen) + '\nexit 0\n')
    (wt / ".venv/bin/python").chmod(0o755)
    rec = tmp_path / "systemd.log"; write_fake_systemd_run(tmp_path / "bin", rec, run=True)
    root = tmp_path / "grade"; (root / "logs").mkdir(parents=True)
    env, marker = _env_wif(tmp_path, ANTHROPIC_API_KEY=SECRET_SENTINEL, F22_GRADE_ROOT=str(root))
    r = subprocess.run(["bash", str(wt / "scripts/sdfbase/full22/judge/launch_judge.sh"), "--targets", "x"], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    calls = [json.loads(l) for l in rec.read_text().splitlines()]
    launch = calls[0]
    assert "--setenv=SISTEER_SECRETS_MODE=wif" in launch["argv"]
    assert not any(a.startswith("--setenv=ANTHROPIC") for a in launch["argv"]) and SECRET_SENTINEL not in json.dumps(launch["argv"])
    assert launch["unit_env_has_key"] is False and "ANTHROPIC_API_KEY" not in launch["unit_env_keys"] and "SISTEER_SECRETS_MODE" in launch["unit_env_keys"]
    assert calls[1]["exit"] == 0                                                                  # the unit ran to completion
    assert seen.read_text().strip() == "mode=wif key=unset dir=unset"                           # the judge ran under wif, keyless
    assert not marker.exists(), "the secret-manager CLI was invoked under wif: " + (marker.read_text() if marker.exists() else "")
    assert f"--setenv=SISTEER_SECRET_CMD={tmp_path / 'bin' / 'fake-secret-cmd'}" in launch["argv"]   # the CLI's path rides along for the
    # unit's non-Anthropic secrets, which resolve as under op
    assert '= wif ]; then unset "${!ANTHROPIC_@}"' in " ".join(launch["cmd"])                  # the keyless branch is in the rendered unit (every ANTHROPIC_* name, round 10)
    logs = [p for p in (root / "logs").glob("judge-*.log") if not p.is_symlink()]
    assert logs and SECRET_SENTINEL not in logs[0].read_text() and SECRET_SENTINEL not in r.stdout + r.stderr
    # the site's ids directory rides along as a value (dir=unset above: the host's never leaks)
    rec.unlink(); seen.unlink()
    env2, _ = _env_wif(tmp_path, F22_GRADE_ROOT=str(root), SISTEER_WIF_ENV_DIR=str(tmp_path / "ids"))
    r = subprocess.run(["bash", str(wt / "scripts/sdfbase/full22/judge/launch_judge.sh"), "--targets", "x"], capture_output=True, text=True, env=env2, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    launch = json.loads(rec.read_text().splitlines()[0])
    assert f"--setenv=SISTEER_WIF_ENV_DIR={tmp_path / 'ids'}" in launch["argv"] and "--setenv=SISTEER_SECRETS_MODE=wif" in launch["argv"]
    assert seen.read_text().strip() == f"mode=wif key=unset dir={tmp_path / 'ids'}"
    assert not marker.exists()
    # the library unreadable under a non-op mode: refused by name, nothing launched (the stub fallback used to launch under op's silence)
    rec.unlink(); (wt / "reproduce/lib/secrets.sh").unlink()
    r = subprocess.run(["bash", str(wt / "scripts/sdfbase/full22/judge/launch_judge.sh"), "--targets", "x"], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 2 and "REFUSED: SISTEER_SECRETS_MODE=wif" in r.stderr and "secrets.sh" in r.stderr, r.stderr
    assert not rec.exists() and not marker.exists()


@pytest.mark.parametrize("script,phase", [("scripts/sdfbase/full22/sampled/grade_sampled.sh", "static"), ("scripts/sdfbase/seedrep/grade_battery.sh", "static")])
def test_detach_relaunch_carries_the_wif_mode_and_no_key(tmp_path, script, phase):
    """The `--detach` re-exec of the two planners under wif: the orchestrator unit receives the mode (it plans and launches the
    judging units under it) and no key; the secret-manager CLI is never invoked on the way."""
    rec = tmp_path / "systemd.log"; write_fake_systemd_run(tmp_path / "bin", rec, run=False)
    env, marker = _env_wif(tmp_path, ANTHROPIC_API_KEY=SECRET_SENTINEL)
    stub_py = tmp_path / "py"; stub_py.write_text("#!/bin/bash\nexit 0\n"); stub_py.chmod(0o755)
    r = subprocess.run(["bash", str(REPO / script), phase, "--detach", "--python", str(stub_py), "--dest" if "sampled" in script else "--rs", str(tmp_path / "dest")],
                       capture_output=True, text=True, env=env, cwd=str(REPO), timeout=120)
    assert "detached: unit=" in r.stdout, r.stdout + r.stderr
    launch = json.loads(rec.read_text().splitlines()[0])
    assert "--setenv=SISTEER_SECRETS_MODE=wif" in launch["argv"]
    assert not any(a.startswith("--setenv=ANTHROPIC") for a in launch["argv"]) and SECRET_SENTINEL not in json.dumps(launch["argv"])
    assert any(a.startswith("--setenv=PATH=") for a in launch["argv"])                     # the existing PATH forward is untouched
    assert not marker.exists()


def _unit_script_source(script: str) -> str:
    """The planner's `unit_script` extracted from the bash-embedded Python (running the planner needs a whole world); bound by
    the caller to a namespace carrying the globals it reads: LOGS, WT, q, org / ORG (grade_sampled / grade_battery) and E."""
    src = (REPO / script).read_text()
    start = src.index("\ndef unit_script("); end = src.index("\ndef render_cmd(", start)
    return src[start:end]


@pytest.mark.parametrize("script", ["scripts/sdfbase/full22/sampled/grade_sampled.sh", "scripts/sdfbase/seedrep/grade_battery.sh"])
def test_unit_script_generators_render_a_keyless_unit_under_wif_that_runs(tmp_path, monkeypatch, script):
    """The generators once rendered the secret-manager export line for EVERY non-env mode, wif included (the CLI ran inside
    the unit for a key that does not exist, and the unit's vault, handed no mode, ran under op). Rendered here under each mode: wif
    re-asserts the mode and only scrubs the inherited ANTHROPIC_*; env requires the one key; op exports the org's own variable,
    byte for byte as before. The wif unit is then EXECUTED under a caller that carries a key and no mode: its probe step sees no
    ANTHROPIC_API_KEY and the mode, and the unit completes."""
    import shlex
    battery = "battery" in script
    wt = tmp_path / "wt"; wt.mkdir(); logs = tmp_path / "logs"; (logs / "static").mkdir(parents=True)

    def render(mode, steps):
        ns = {"os": os, "Path": Path, "q": shlex.quote, "LOGS": logs, "WT": wt, "org": "lowprio", "ORG": "lowprio", "E": {"SISTEER_SECRETS_MODE": mode}}
        monkeypatch.setenv("SISTEER_SECRETS_MODE", mode)
        exec(_unit_script_source(script), ns)
        return ns["unit_script"]("u1", "static", steps, needs_key=True) if battery else ns["unit_script"]("u1", steps, needs_key=True)

    # the probe: under wif the unit carries NO ANTHROPIC_* name at all -- the key, and (round 10) the custom-headers variable the SDK
    # would parse into every request whatever credential it is given
    probe = [{"name": "probe", "argv": ["bash", "-c", 'if [ -z "${ANTHROPIC_API_KEY+x}" ] && [ -z "$(compgen -A variable | grep "^ANTHROPIC_")" ] && [ "${SISTEER_SECRETS_MODE:-}" = wif ]; then echo PROBE-OK; else echo PROBE-BAD; exit 1; fi']}]
    wif_u, env_u, op_u = render("wif", probe), render("env", probe), render("op", probe)
    assert 'export SISTEER_SECRETS_MODE=wif; unset "${!ANTHROPIC_@}"' in wif_u                 # every ANTHROPIC_* name by name (vault.WIF_SHELL_SCRUB)
    assert "SISTEER_SECRET_CMD" not in wif_u and ":?" not in wif_u and "export ANTHROPIC" not in wif_u
    assert ': "${ANTHROPIC_API_KEY:?' in env_u and "SISTEER_SECRET_CMD" not in env_u and "SISTEER_SECRETS_MODE=wif" not in env_u
    assert f'export ANTHROPIC_API_KEY_LOWPRIO="$({KEY_CMD} lowprio)"' in op_u and "SISTEER_SECRETS_MODE=wif" not in op_u
    script_p = tmp_path / "u1.sh"; script_p.write_text(wif_u); script_p.chmod(0o700)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "SISTEER_"))}
    env.update(PATH="/usr/bin:/bin", HOME=str(tmp_path), ANTHROPIC_API_KEY=SECRET_SENTINEL,
               ANTHROPIC_CUSTOM_HEADERS="X-Api-Key: " + SECRET_SENTINEL)                       # the inherited static header (round 10)
    r = subprocess.run(["bash", str(script_p)], capture_output=True, text=True, env=env, timeout=60)
    log_dir = logs / "static" if battery else logs
    log = (log_dir / "u1.log").read_text()
    assert r.returncode == 0, r.stderr + log
    assert "PROBE-OK" in log and "PROBE-BAD" not in log and "[unit] done" in log and (log_dir / "EXIT-u1").read_text().strip() == "0"
    assert SECRET_SENTINEL not in log
