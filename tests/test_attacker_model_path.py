"""The attacker a job script serves is the config's pinned snapshot, resolved on the node and present whole, or the job
refuses by name before any GPU work.

The configs write ``attacker.path`` as ``$HF_HOME/hub/.../snapshots/<revision>`` (one as ``$SHARED/...``). The job scripts
that serve the configured attacker hand that value to ``vllm serve``, and a shell never expands a ``$NAME`` that a Python
one-liner printed. So each of them takes the value from ``scripts/attacker_model_path.py``, which expands it through
``sisteer.site``, checks that the directory is the pinned revision with its config and every weight file on this host, and
exits 2, printing nothing, when it is not.

* the helper: expansion from the environment, ``$NAME`` and ``${NAME}`` alike; a refusal by name for an unset or unknown
  placeholder, a snapshot that is absent, incomplete or another revision, and a pinned snapshot with no path; the repo id
  only for a config that pins no snapshot
* every committed config's ``attacker.path``: it resolves under a test site to an absolute directory ending in the config's
  pinned ``attacker.snapshot``
* each job script's own ``MODEL_PATH=`` statement, run in bash against a placeholder-valued config: the real path, or the
  job exits 1 naming the placeholder or the missing snapshot, never the literal text
* each job script assigns ``MODEL_PATH`` once and passes that variable to every ``vllm_attacker_serve.sh`` call, so no later
  line can swap in the repo id; the MT wave and the SWP pool run end to end, from the assignment through the serve call
  (a stub that records its arguments), with the snapshot present and absent
* no job script under ``scripts/`` or ``reproduce/`` reads ``attacker.path`` from a config itself
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

REPO = Path(__file__).resolve().parents[1]
HELPER = REPO / "scripts" / "attacker_model_path.py"
sys.path.insert(0, str(HELPER.parent))

import attacker_model_path as AMP  # noqa: E402

#: the job scripts that serve a config's attacker (vllm serve), each reading the model path once
LAUNCHERS = (
    "scripts/sbatch/attacker_pool.sbatch",
    "scripts/sbatch/swp_attacker_pool.sbatch",
    "scripts/sdfbase/mt/attack_wave.sbatch",
    "scripts/sbatch/jailbreak_vllm_measure.sbatch",
)
#: the launchers whose lines between the assignment and the serve call run without a GPU, a venv or a target model
END_TO_END = ("scripts/sbatch/swp_attacker_pool.sbatch", "scripts/sdfbase/mt/attack_wave.sbatch")
MODEL = "example-org/attacker-27b"
SNAP = "27502c8717fd5a2f8c0c77188c10c243fd4f672e"
REL_SNAP = f"hub/models--example-org--attacker-27b/snapshots/{SNAP}"
CFG_TEXT = f"attacker:\n  model: {MODEL}\n  snapshot: {SNAP}\n  path: $HF_HOME/{REL_SNAP}\n"


def _snapshot(root: Path, rel: str = REL_SNAP, layout: str = "single") -> Path:
    """A fake cached snapshot under ``root``: ``single`` (config + one weight file), ``sharded`` (an index and both shards),
    or a broken one: ``no-config``, ``no-weights``, ``missing-shard``, ``dangling-shard`` (a cache link to no blob)."""
    snap = root / rel
    snap.mkdir(parents=True, exist_ok=True)
    if layout != "no-config":
        (snap / "config.json").write_text("{}", encoding="utf-8")
    if layout in ("single", "no-config"):
        (snap / "model.safetensors").write_bytes(b"\0")
    if layout in ("sharded", "missing-shard", "dangling-shard"):
        shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        (snap / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": shards[0], "b": shards[1]}}), encoding="utf-8")
        (snap / shards[0]).write_bytes(b"\0")
        if layout == "sharded":
            (snap / shards[1]).write_bytes(b"\0")
        elif layout == "dangling-shard":
            (snap / shards[1]).symlink_to(root / "blobs" / "absent")
    return snap


def _env(tmp_path: Path, **extra) -> dict:
    """A process environment with no site config and no placeholder variable, plus ``extra``."""
    env = {"PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}",
           "HOME": str(tmp_path / "home"), "SISTEER_SITE_DIR": str(tmp_path / "no-site-here"), "LANG": "C.UTF-8"}
    env.update(extra)
    return env


def _run_helper(tmp_path: Path, cfg_text: str, **env) -> subprocess.CompletedProcess:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(cfg_text, encoding="utf-8")
    return subprocess.run([sys.executable, str(HELPER), str(cfg)], env=_env(tmp_path, **env), capture_output=True,
                          text=True, timeout=60)


def _assert_refused(r: subprocess.CompletedProcess, *needles: str) -> None:
    assert r.returncode == 2 and r.stdout == "", (r.returncode, r.stdout, r.stderr)
    assert "REFUSED" in r.stderr, r.stderr
    for n in needles:
        assert n in r.stderr, (n, r.stderr)


# --- the helper ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("layout", ["single", "sharded"])
def test_the_helper_expands_hf_home_and_prints_a_complete_snapshot(tmp_path, layout):
    _snapshot(tmp_path / "hf", layout=layout)
    r = _run_helper(tmp_path, CFG_TEXT, HF_HOME=str(tmp_path / "hf"))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == f"{tmp_path}/hf/{REL_SNAP}"
    braced = CFG_TEXT.replace("$HF_HOME", "${HF_HOME}")
    assert _run_helper(tmp_path, braced, HF_HOME=str(tmp_path / "hf")).stdout == r.stdout


def test_the_helper_refuses_an_unset_placeholder_by_name_and_prints_nothing(tmp_path):
    _assert_refused(_run_helper(tmp_path, CFG_TEXT), "$HF_HOME", "attacker.path")


def test_the_helper_refuses_a_placeholder_outside_the_grammar(tmp_path):
    _assert_refused(_run_helper(tmp_path, CFG_TEXT.replace("$HF_HOME", "$HFHOME"), HF_HOME=str(tmp_path / "hf")), "$HFHOME")


@pytest.mark.parametrize("layout", ["absent", "no-config", "no-weights", "missing-shard", "dangling-shard"])
def test_the_helper_refuses_a_snapshot_that_is_not_here_whole(tmp_path, layout):
    if layout != "absent":
        _snapshot(tmp_path / "hf", layout=layout)
    r = _run_helper(tmp_path, CFG_TEXT, HF_HOME=str(tmp_path / "hf"))
    _assert_refused(r, f"{tmp_path}/hf/{REL_SNAP}", f"hf download {MODEL} --revision {SNAP}")
    if layout in ("missing-shard", "dangling-shard"):
        assert "model-00002-of-00002.safetensors" in r.stderr and "incomplete" in r.stderr


def test_the_helper_refuses_a_path_that_names_another_revision(tmp_path):
    other = REL_SNAP.replace(SNAP, "0" * 40)
    _snapshot(tmp_path / "hf", rel=other)
    r = _run_helper(tmp_path, CFG_TEXT.replace(REL_SNAP, other), HF_HOME=str(tmp_path / "hf"))
    _assert_refused(r, "does not end in the pinned snapshot", SNAP)


def test_the_helper_expands_the_shared_root_and_serves_a_repo_id_only_without_a_pin(tmp_path):
    shared = CFG_TEXT.replace("$HF_HOME", "$SHARED/hf_cache")
    _snapshot(tmp_path / "shared" / "hf_cache")
    r = _run_helper(tmp_path, shared, SISTEER_SHARED_ROOT=str(tmp_path / "shared"))
    assert r.returncode == 0 and r.stdout.strip() == f"{tmp_path}/shared/hf_cache/{REL_SNAP}", r.stderr
    r = _run_helper(tmp_path, f"attacker:\n  model: {MODEL}\n")
    assert r.returncode == 0 and r.stdout.strip() == MODEL, r.stderr
    _assert_refused(_run_helper(tmp_path, f"attacker:\n  model: {MODEL}\n  snapshot: {SNAP}\n"), "attacker.snapshot",
                    "unpinned revision")
    _assert_refused(_run_helper(tmp_path, "attacker: {}\n"), "attacker.model")


def test_every_committed_attacker_path_resolves_to_its_pinned_snapshot():
    import yaml

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    env = {"HF_HOME": "/site/hf", "SISTEER_SHARED_ROOT": "/site/shared", "SISTEER_SITE_DIR": "/nonexistent/site"}
    seen, bad = 0, []
    for p in sorted((REPO / "configs").rglob("*.yaml")):
        text = p.read_text(encoding="utf-8", errors="replace")
        if "attacker:" not in text or "registry" in p.relative_to(REPO / "configs").parts[:1]:
            continue
        cfg = yaml.load(text, Loader=loader)
        att = cfg.get("attacker") if isinstance(cfg, dict) else None
        if not isinstance(att, dict) or not att.get("path"):
            continue
        seen += 1
        try:
            got = AMP.attacker_model_path(cfg, env=env)
        except ValueError as exc:
            bad.append(f"{p.relative_to(REPO)}: {exc}")
            continue
        snap = str(att.get("snapshot") or "")
        if not got.startswith("/site/") or not snap or PurePosixPath(got).name != snap:
            bad.append(f"{p.relative_to(REPO)}: {att['path']!r} -> {got!r} (snapshot {snap!r})")
    assert seen > 0, "no committed config names attacker.path"
    assert not bad, f"{len(bad)} of {seen} attacker paths do not end in their snapshot: " + "; ".join(bad[:10])


# --- the job scripts -----------------------------------------------------------------------------------------------------

def _statement(script: Path, first_marker: str, last_marker: str = "", span: int = 12) -> str:
    """The lines of ``script`` from the first one starting ``first_marker`` through the first complete command at or after
    the first later line containing ``last_marker`` (the start line when none), checked complete with ``bash -n``."""
    lines = script.read_text(encoding="utf-8").splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith(first_marker)]
    assert starts, f"{script.relative_to(REPO)} has no {first_marker!r} line"
    lo = starts[0]
    if last_marker:
        hits = [i for i in range(lo + 1, len(lines)) if last_marker in lines[i]]
        assert hits, f"{script.relative_to(REPO)} has no {last_marker!r} line after {first_marker!r}"
        lo = hits[0]
    for end in range(lo, min(lo + span, len(lines))):
        if lines[end].rstrip().endswith("\\"):
            continue
        text = "\n".join(lines[starts[0]:end + 1])
        if subprocess.run(["bash", "-n", "-c", text], capture_output=True).returncode == 0:
            return text
    raise AssertionError(f"{script.relative_to(REPO)}: no complete command within {span} lines")


def _checkout(tmp_path: Path) -> Path:
    """A checkout with the helper, a placeholder-valued config, a serve stub that records its arguments, and a python3."""
    repo = tmp_path / "checkout"
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    (repo / "configs").mkdir(exist_ok=True)
    link = repo / "scripts" / HELPER.name
    if not link.exists():
        link.symlink_to(HELPER)
    (repo / "configs" / "attacker.yaml").write_text(CFG_TEXT, encoding="utf-8")
    stub = repo / "scripts" / "vllm_attacker_serve.sh"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$SERVE_RECORD"\n', encoding="utf-8")
    (tmp_path / "bin").mkdir(exist_ok=True)
    if not (tmp_path / "bin" / "python3").exists():
        (tmp_path / "bin" / "python3").symlink_to(sys.executable)
    return repo


def _run_lines(tmp_path: Path, text: str, **env) -> subprocess.CompletedProcess:
    """Run launcher lines the way the job does: from the checkout, PY / MPY / python3 a Python with PyYAML, the job's
    other inputs set to scratch values, ``sleep`` a no-op (the wave backs off between serve attempts)."""
    repo = _checkout(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    prelude = (f'set -uo pipefail\nlog(){{ echo "[log] $*" >&2; }}\nsleep(){{ :; }}\n'
               f'PY={sys.executable!s}\nMPY={sys.executable!s}\nREPO_DIR={repo!s}\nCFG=configs/attacker.yaml\n'
               f'SERVE_DIR={scratch!s}\nPOOL_DIR={scratch!s}\nVLLM_VENV={scratch / "venv"!s}\nSLURM_JOB_ID=4242\n'
               f'GPU=0\nGPU_A=0\nPORT=8000\nNODE_IP=127.0.0.1\ncd "$REPO_DIR"\n')
    return subprocess.run(["bash", "-c", prelude + text + '\necho "MODEL_PATH=$MODEL_PATH"\n'],
                          env=_env(tmp_path, SERVE_RECORD=str(tmp_path / "served.txt"), **env),
                          capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_the_job_script_serves_the_expanded_snapshot(launcher, tmp_path):
    _snapshot(tmp_path / "hf")
    r = _run_lines(tmp_path, _statement(REPO / launcher, "MODEL_PATH="), HF_HOME=str(tmp_path / "hf"))
    want = f"{tmp_path}/hf/{REL_SNAP}"
    assert r.returncode == 0 and f"MODEL_PATH={want}" in r.stdout.splitlines(), (launcher, r.stdout, r.stderr)


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_the_job_script_refuses_an_unset_placeholder_before_serving(launcher, tmp_path):
    r = _run_lines(tmp_path, _statement(REPO / launcher, "MODEL_PATH="))
    assert r.returncode != 0, (launcher, r.stdout)
    assert "MODEL_PATH=" not in r.stdout, f"{launcher} went on with {r.stdout.strip()!r}"
    assert "$HF_HOME" in r.stderr and "FATAL" in r.stderr, (launcher, r.stderr)


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_the_job_script_refuses_a_missing_snapshot_before_serving(launcher, tmp_path):
    r = _run_lines(tmp_path, _statement(REPO / launcher, "MODEL_PATH="), HF_HOME=str(tmp_path / "hf"))
    assert r.returncode != 0, (launcher, r.stdout)
    assert "MODEL_PATH=" not in r.stdout, f"{launcher} went on with {r.stdout.strip()!r}"
    assert "FATAL" in r.stderr and f"{tmp_path}/hf/{REL_SNAP}" in r.stderr, (launcher, r.stderr)


#: an assignment to MODEL_PATH anywhere on a line (a later `[ -e ... ] || MODEL_PATH=...` fallback included)
ASSIGN = re.compile(r"(^|[\s;&|(])(export\s+)?MODEL_PATH=")


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_the_job_script_serves_the_value_it_assigned(launcher):
    text = (REPO / launcher).read_text(encoding="utf-8")
    assigns = [i for i, ln in enumerate(text.splitlines(), 1) if ASSIGN.search(ln)]
    assert len(assigns) == 1, f"{launcher} assigns MODEL_PATH on lines {assigns}; a second assignment can serve a fallback"
    calls = re.findall(r"scripts/vllm_attacker_serve\.sh(?:[^\n]*\\\n)*[^\n]*", text)
    assert calls, f"{launcher} has no vllm_attacker_serve.sh call"
    for call in calls:
        assert '--model "$MODEL_PATH"' in call, f"{launcher}: a serve call does not pass the assigned path: {call[:200]}"


@pytest.mark.parametrize("launcher", END_TO_END)
def test_the_wave_and_the_pool_serve_the_pinned_snapshot_or_nothing(launcher, tmp_path):
    lines = _statement(REPO / launcher, "MODEL_PATH=", "scripts/vllm_attacker_serve.sh", span=40)
    served = tmp_path / "served.txt"
    r = _run_lines(tmp_path, lines, HF_HOME=str(tmp_path / "hf"))
    assert r.returncode != 0 and "FATAL" in r.stderr, (launcher, r.stdout, r.stderr)
    assert not served.exists(), f"{launcher} reached the serve call without its snapshot: {served.read_text()!r}"
    _snapshot(tmp_path / "hf")
    r = _run_lines(tmp_path, lines, HF_HOME=str(tmp_path / "hf"))
    calls = served.read_text(encoding="utf-8").splitlines() if served.exists() else []
    assert calls, (launcher, r.stdout, r.stderr)
    for call in calls:
        assert f"--model {tmp_path}/hf/{REL_SNAP} " in call + " ", (launcher, call)


#: a job script reading attacker.path out of a config itself (the value would reach vllm or a test unexpanded)
RAW_READ = re.compile(r"""\[['"]attacker['"]\]\s*(\.get\(\s*['"]path['"]|\[\s*['"]path['"]\s*\])""")


def test_no_job_script_reads_the_attacker_path_itself():
    hits = []
    for top in ("scripts", "reproduce"):
        for p in sorted((REPO / top).rglob("*")):
            if p.is_file() and p.suffix in (".sh", ".sbatch"):
                for i, ln in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if RAW_READ.search(ln):
                        hits.append(f"{p.relative_to(REPO)}:{i}")
    assert not hits, ("these job scripts read attacker.path from the config themselves; call "
                      "scripts/attacker_model_path.py instead: " + ", ".join(hits))
    for launcher in LAUNCHERS:
        assert "scripts/attacker_model_path.py" in _statement(REPO / launcher, "MODEL_PATH="), launcher
