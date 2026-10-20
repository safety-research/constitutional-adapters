"""The startup config assertion (wave-autopsy 2026-08-22, lesson 1).

The 2026-08-22 GPU wave ran all night on stale arm widths: the launch
tarball predated the config commit, a running scheduler never re-reads its
yaml (so shipping the corrected file changed nothing), and no fit ever
printed what it had actually resolved. Two guards close that class:

- the ``[config-resolved]`` banner — the FIRST line of the scheduler
  (run_parallel_fit) and of every arm task (cmd_arm_task) names the config
  file's byte identity (sha256 of the FILE BYTES) plus the RESOLVED
  widths/epochs/leg, so a human diffs a running fit against intent from its
  first log lines instead of hours later;
- ``--expect-config-sha`` — the driver-side CHECK: a wave driver computes
  the sha of the config it INTENDS and passes it, and a stale tarball then
  refuses in milliseconds — before any pair or model load, before the
  world-size pin. The scheduler forwards a pinned sha to every arm task it
  spawns (the driver's, else its own), because an arm task re-reads the
  config FILE, possibly hours after launch.

CPU-only, no model loads: the scheduler paths are exercised the way
tests/test_ddp_fit.py exercises them (bare argparse Namespaces +
SISTEER_FIT_GPU_POOL), so every helper must survive minimal cfg dicts and
Namespaces that predate the flag (getattr/.get defaults throughout).
"""

from __future__ import annotations

import hashlib
import inspect
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_two_si_objects as tt

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "train_two_si_objects.py"


def _ns(**kw):
    """A bare Namespace, the way the existing driver tests build them —
    deliberately missing every attribute not passed, since programmatic
    callers predate the new flag."""
    return type("A", (), kw)()


def _write_cfg(tmp_path: Path) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump({"data_parallel": {"gpus": 3}}))
    return p


# --------------------------------------------------------------------------- #
# The sha: file-byte identity, resolved like _load_cfg                          #
# --------------------------------------------------------------------------- #


def test_config_sha_is_the_file_byte_identity(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("data_parallel:\n  gpus: 4\n")
    want = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    assert tt._config_sha(str(p)) == want
    assert tt._config_sha(str(p)) == want, "must be stable across calls"
    # A yaml-invisible edit (a comment) still changes the identity: the sha
    # is of the FILE BYTES, because a stale tarball is a byte question, not
    # a parse question.
    p.write_text("data_parallel:\n  gpus: 4\n# touched\n")
    assert tt._config_sha(str(p)) != want
    # Unreadable -> None, never a crash: the driver tests pass placeholder
    # config paths like "x".
    assert tt._config_sha(str(tmp_path / "missing.yaml")) is None
    assert tt._config_sha("") is None


def test_config_sha_resolves_relative_paths_like_load_cfg():
    """A relative path hashes the repo-rooted file _load_cfg parses — the
    SAME resolution, so the banner's sha is the identity of the bytes the
    fit actually loaded, not of some cwd-relative namesake."""
    rel = "configs/two_si.yaml"
    want = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()[:16]
    assert tt._config_sha(rel) == want
    assert tt._config_sha(str(REPO / rel)) == want


# --------------------------------------------------------------------------- #
# The banner                                                                    #
# --------------------------------------------------------------------------- #


def test_banner_prints_resolved_values_on_one_greppable_line():
    cfg = {"data_parallel": {"gpus": 4,
                             "arm_gpus": {"bidpo": 2, "diffmean": 4}},
           "train_kwargs": {"default": {"epochs": 3, "positions": "all"}}}
    sha = "ab12" * 4
    line = tt._resolved_banner(cfg, sha, "all", "configs/two_si.yaml")
    assert "\n" not in line and line.startswith("[config-resolved] ")
    assert f"sha256={sha}" in line
    assert "gpus=4" in line
    assert "arm_gpus=bidpo:2,diffmean:4" in line, \
        "mixed widths must print per arm, sorted"
    assert "epochs=3" in line
    assert "positions=all" in line
    assert f"config={REPO / 'configs/two_si.yaml'}" in line


def test_banner_compacts_uniform_arm_widths():
    cfg = {"data_parallel": {"gpus": 4,
                             "arm_gpus": {"bidpo": 1, "diffmean": 1}}}
    assert "arm_gpus=all:1" in tt._resolved_banner(cfg, None, "all", "")
    # No override map at all: widths derive from per-arm batch sizes.
    assert "arm_gpus=auto" in tt._resolved_banner({}, None, "all", "")


def test_banner_tolerates_minimal_cfg_and_missing_file():
    """The world-size-pin test's minimal cfg shape must flow through the
    banner unharmed — .get defaults everywhere, in-code defaults where the
    fit itself has them."""
    line = tt._resolved_banner({}, None, "all", "")
    assert line.startswith("[config-resolved] ")
    assert "sha256=unreadable" in line
    assert "gpus=1" in line
    assert "epochs=6" in line, "train_si_objects' in-code default cap"


def test_banner_positions_is_the_post_override_leg():
    """cmd_arm_task applies --positions via _cfg_with_positions BEFORE the
    banner: the printed leg is what the task will FIT, never merely what
    the config file said."""
    cfg = {"train_kwargs": {"default": {"positions": "all"}}}
    over = tt._cfg_with_positions(cfg, "response")
    line = tt._resolved_banner(
        over, "0" * 16,
        tt._effective_positions(_ns(positions="response"), over), "")
    assert "positions=response" in line
    # ... and with no flag, the config's own leg.
    line = tt._resolved_banner(
        cfg, "0" * 16, tt._effective_positions(_ns(), cfg), "")
    assert "positions=all" in line


# --------------------------------------------------------------------------- #
# The check                                                                     #
# --------------------------------------------------------------------------- #


def test_expect_sha_mismatch_refuses_naming_both_and_the_path(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    actual = tt._config_sha(str(cfg_path))
    with pytest.raises(SystemExit) as ei:
        tt._assert_expected_config_sha(
            _ns(config=str(cfg_path), expect_config_sha="0" * 16))
    msg = str(ei.value)
    assert "0" * 16 in msg, "the refusal must name the EXPECTED sha"
    assert actual in msg, "the refusal must name the ACTUAL sha"
    assert str(cfg_path) in msg, "the refusal must name the config path"


def test_expect_sha_match_proceeds(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    full = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    # 16-hex (the banner value), the full sha, and case-insensitively.
    for given in (full[:16], full, full[:16].upper()):
        got = tt._assert_expected_config_sha(
            _ns(config=str(cfg_path), expect_config_sha=given))
        assert got == full[:16]
    # No flag — or a bare pre-flag Namespace — is never a gate.
    assert tt._assert_expected_config_sha(
        _ns(config=str(cfg_path))) == full[:16]
    assert tt._assert_expected_config_sha(_ns()) is None


def test_expect_sha_refusal_fires_before_the_world_size_pin(
        tmp_path, monkeypatch):
    """Ordering: a stale config must die on the SHA, not on whatever the
    stale widths happen to do to the pool assertion — the check is cheaper
    and more fundamental than the world-size pin."""
    monkeypatch.setenv("SISTEER_FIT_GPU_POOL", "0,1")
    cfg_path = _write_cfg(tmp_path)  # gpus: 3 — the pin WOULD also refuse
    args = _ns(si="s", overwrite=False, config=str(cfg_path), pairs="y",
               out="", arms="", expect_config_sha="f" * 16)
    with pytest.raises(SystemExit) as ei:
        tt.run_parallel_fit(args, yaml.safe_load(cfg_path.read_text()),
                            ["diffmean"], tmp_path)
    msg = str(ei.value)
    assert "sha256" in msg and "f" * 16 in msg
    assert "PINNED" not in msg, \
        "the sha refusal must fire BEFORE the world-size pin"


def test_scheduler_banner_is_first_output_and_match_proceeds(
        tmp_path, monkeypatch, capsys):
    """run_parallel_fit with the CORRECT sha proceeds past the check (the
    world-size pin then fires on the 2-vs-3 pool), and its FIRST output is
    the banner — carrying the real sha and the flag's positions leg."""
    monkeypatch.setenv("SISTEER_FIT_GPU_POOL", "0,1")
    cfg_path = _write_cfg(tmp_path)
    actual = tt._config_sha(str(cfg_path))
    args = _ns(si="s", overwrite=False, config=str(cfg_path), pairs="y",
               out="", arms="", positions="response",
               expect_config_sha=actual)
    with pytest.raises(SystemExit) as ei:
        tt.run_parallel_fit(args, yaml.safe_load(cfg_path.read_text()),
                            ["diffmean"], tmp_path)
    assert "PINNED" in str(ei.value), "a matching sha must proceed"
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith("[config-resolved] ")
    assert f"sha256={actual}" in first
    assert "gpus=3" in first
    assert "positions=response" in first


def test_arm_task_banner_first_and_post_override(tmp_path, monkeypatch,
                                                 capsys):
    """The real arm-task entrypoint, up to its first refusal past the
    banner (the width pin — no model load happens): the banner is this
    rank's first output and prints the post-override leg. (The width pin
    is the data-parallel gate, _enter_fit_process_group: a planned
    width of 2 with no launcher behind it refuses by name — "not under a
    2-rank launcher" — before any import heavier than the banner's. Same
    contract: banner first, width pin next, nothing heavier.)"""
    for var in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR",
                "MASTER_PORT"):
        monkeypatch.delenv(var, raising=False)
    cfg_path = _write_cfg(tmp_path)
    args = _ns(config=str(cfg_path), positions="response",
               expected_width="2", si="s", arm="diffmean",
               pairs=str(tmp_path / "absent.jsonl"), out="",
               overwrite=False)
    with pytest.raises(SystemExit) as ei:
        tt.cmd_arm_task(args, yaml.safe_load(cfg_path.read_text()))
    assert "not under a 2-rank launcher" in str(ei.value), \
        "expected to get past the banner to the width pin, nothing heavier"
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith("[config-resolved] ")
    assert f"sha256={tt._config_sha(str(cfg_path))}" in first
    assert "positions=response" in first


# --------------------------------------------------------------------------- #
# The scheduler forwards the pin to every arm task                              #
# --------------------------------------------------------------------------- #


class _FakePopen:
    calls: list[list[str]] = []

    def __init__(self, cmd, **kw):
        _FakePopen.calls.append(list(cmd))

    def poll(self):
        return 0


def _spawn_argv(monkeypatch, tmp_path, args):
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    _FakePopen.calls = []
    tt._spawn_arm_task(args, "diffmean", 1, ["0"], tmp_path)
    (argv,) = _FakePopen.calls
    return argv


def test_spawn_argv_forwards_expect_config_sha(tmp_path, monkeypatch):
    """The scheduler forwards the driver's flag to every arm task it
    spawns, the way it forwards --positions."""
    cfg_path = _write_cfg(tmp_path)
    argv = _spawn_argv(monkeypatch, tmp_path, _ns(
        config=str(cfg_path), si="s", pairs="p.jsonl", out="",
        overwrite=False, expect_config_sha="d" * 16))
    i = argv.index("--expect-config-sha")
    assert argv[i + 1] == "d" * 16


def test_spawn_autopins_the_schedulers_own_config_sha(tmp_path, monkeypatch):
    """Even with no driver flag, every arm task is pinned to the sha of the
    file THIS scheduler resolved: an arm task re-reads the config file —
    possibly hours after launch for a queued arm — so a mid-wave config
    edit must refuse rather than silently fit a different world than the
    scheduler planned."""
    cfg_path = _write_cfg(tmp_path)
    argv = _spawn_argv(monkeypatch, tmp_path, _ns(
        config=str(cfg_path), si="s", pairs="p.jsonl", out="",
        overwrite=False))
    i = argv.index("--expect-config-sha")
    assert argv[i + 1] == tt._config_sha(str(cfg_path))
    # An unreadable config pins nothing — there is nothing to pin against.
    argv = _spawn_argv(monkeypatch, tmp_path, _ns(
        config=str(tmp_path / "gone.yaml"), si="s", pairs="p.jsonl",
        out="", overwrite=False))
    assert "--expect-config-sha" not in argv


def test_arm_task_source_order_and_spawn_pattern():
    """Pinned order inside cmd_arm_task — sha check, THEN the positions
    override, THEN the banner — so a stale task refuses before doing any
    work and the banner prints the post-override leg. Source-level, the
    test_arm_task_inherits_positions_and_bidpo_pool pattern."""
    src = inspect.getsource(tt.cmd_arm_task)
    i_check = src.index("_assert_expected_config_sha")
    i_over = src.index("_cfg_with_positions")
    i_banner = src.index("_resolved_banner")
    assert i_check < i_over < i_banner
    assert "--expect-config-sha" in inspect.getsource(tt._spawn_arm_task), \
        "scheduler does not forward --expect-config-sha"


# --------------------------------------------------------------------------- #
# The real argv surface                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sub", ["train", "arm-task"])
def test_cli_refuses_stale_config_before_any_load(tmp_path, sub):
    """Both subcommands through the real CLI: a wrong --expect-config-sha
    exits nonzero naming both shas and the config path — a refusal, not a
    crash, and BEFORE the pairs file (deliberately absent here) or any
    model is touched."""
    cfg_path = _write_cfg(tmp_path)
    actual = tt._config_sha(str(cfg_path))
    cmd = [sys.executable, str(SCRIPT), sub, "--config", str(cfg_path),
           "--si", "s", "--pairs", str(tmp_path / "absent.jsonl"),
           "--expect-config-sha", "0" * 16]
    if sub == "arm-task":
        cmd += ["--arm", "diffmean"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert "0" * 16 in r.stderr and actual in r.stderr
    assert str(cfg_path) in r.stderr
    assert "Traceback" not in r.stderr, "a refusal, not a crash"
    assert "absent.jsonl" not in r.stderr, \
        "must refuse before ever looking at the pairs file"
