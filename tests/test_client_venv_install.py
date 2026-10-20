"""The tinycap CLIENT venv's hashed install (common.sh f22_client_venv, and its in-container copy seedrep_served.ensure_client_venv).

The client lock (configs/capeval/tinycap-client-requirements.txt) is fully hash-pinned and compiled with `--no-config` under its
own cutoff. Both bootstraps run uv from inside the checkout, where uv reads pyproject.toml's [tool.uv] table. A command-line
`--exclude-newer` replaces only the table's global cutoff; its `exclude-newer-package` entries still apply, and the anthropic entry
(2026-08-21) refuses the lock's anthropic 1.1.0 (published 2026-08-26). So every first build of a client venv failed at the
client-venv guard (the Nemotron backend probe died there on a node with no client venv). The install therefore reads no project
configuration. These tests run the function's own install line for real (a uv dry run, from inside the checkout), show the line
without --no-config is refused there while the table carries such an entry, and hold the in-container copy to the same flag."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:          # Python 3.10
    tomllib = None

REPO = Path(__file__).resolve().parents[1]
F22 = REPO / "scripts" / "sdfbase" / "full22" / "common.sh"
SEEDREP = REPO / "scripts" / "modal" / "seedrep_served.py"
LOCK = REPO / "configs" / "capeval" / "tinycap-client-requirements.txt"
VCS = re.compile(r"^(instruction-following-eval) @ ")
#: what a uv that cannot reach its index says; only then does a dry run skip (any other failure fails the test)
NETWORK = re.compile(r"(?i)failed to fetch|error sending request|dns error|tcp connect|connection (refused|reset|timed out)"
                     r"|network is unreachable|operation timed out|offline mode")
#: uv's refusal of a pin its per-package cutoff filters out
REFUSAL = re.compile(r"there is no version of (\S+)==")


def _func(name: str) -> str:
    sh = F22.read_text()
    start = sh.index(f"{name}(){{")
    return sh[start:sh.index("\n}\n", start) + 3]


def _install_line() -> str:
    """The hashed install exactly as f22_client_venv runs it (its continuation lines joined), less the trailing `|| exit 1`."""
    body = _func("f22_client_venv")
    m = re.search(r"^\s*(uv pip install [^\n]*--require-hashes \\\n[^\n]*-r \"\$hashed\") \|\| exit 1$", body, re.M)
    assert m, "f22_client_venv carries no hashed `uv pip install ... --require-hashes ... -r \"$hashed\"` line"
    return re.sub(r"\\\n\s*", "", m.group(1))


def _uv_env() -> dict:
    """The caller's environment minus the switches that change uv's configuration discovery, so a dry run reads the checkout
    exactly as a job does."""
    return {k: v for k, v in os.environ.items() if k not in ("UV_NO_CONFIG", "UV_CONFIG_FILE", "UV_EXCLUDE_NEWER")}


def _lock_pins() -> dict:
    out = {}
    for ln in LOCK.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]+\])?==([^ \\]+)", ln)
        if m:
            out[m.group(1).lower()] = m.group(2)
    return out


# --------------------------------------------------------------------------- statically
def test_the_hashed_client_install_reads_no_project_configuration():
    line = _install_line()
    assert line.startswith("uv pip install --no-config ") and "--no-deps --require-hashes" in line
    # the rolling one-week cutoff stays: --no-config drops the table's cutoff, it must not drop the >= 1-week-old rule
    assert '--exclude-newer "$(date -u -d "-7 days" +%Y-%m-%dT00:00:00Z)"' in line
    # the lock is compiled the same way (its header), so install and compile read the same (no) configuration
    assert "uv pip compile --no-config " in LOCK.read_text().splitlines()[1]


def test_the_in_container_copy_carries_the_same_flag():
    src = SEEDREP.read_text()
    start = src.index("def ensure_client_venv(")
    body = src[start:src.index("\ndef ", start + 1)]
    hashed = [ln for ln in re.findall(r'steps\.append\(\[([^\]]*"--require-hashes"[^\]]*)\]', body, re.S)]
    assert len(hashed) == 1 and '"--no-config"' in hashed[0] and '"--exclude-newer", cutoff' in hashed[0], hashed


# --------------------------------------------------------------------------- run (a uv dry run from inside the checkout)
@pytest.fixture(scope="module")
def client_dry(tmp_path_factory):
    uv = shutil.which("uv")
    if not uv:
        pytest.skip("uv is not on PATH: the dry run needs the uv the bootstraps call")
    if not shutil.which("bash"):
        pytest.skip("bash is not on PATH")
    root = tmp_path_factory.mktemp("client-dry")
    venv = root / "venv"
    r = subprocess.run([uv, "venv", "-q", "--no-config", "--python", "3.12", str(venv)], capture_output=True, text=True,
                       env=_uv_env(), timeout=300)
    if r.returncode != 0:
        pytest.skip(f"uv cannot make a Python 3.12 venv here (the client lock is resolved for 3.12): {r.stderr.strip()[-200:]}")
    hashed = root / "hashed.txt"
    hashed.write_text("".join(ln + "\n" for ln in LOCK.read_text().splitlines() if not VCS.match(ln)))
    return venv, hashed


def _dry_run(line: str, venv: Path, hashed: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", f"{line} --dry-run", "_"], cwd=REPO, capture_output=True, text=True, timeout=900,
                          env={**_uv_env(), "CLIENT_VENV": str(venv), "hashed": str(hashed)})


def test_the_client_lock_resolves_from_inside_the_checkout(client_dry):
    """f22_client_venv's own install line, --dry-run appended, from the checkout root: every hashed pin of the lock resolves."""
    venv, hashed = client_dry
    r = _dry_run(_install_line(), venv, hashed)
    out = r.stdout + r.stderr
    if r.returncode != 0 and NETWORK.search(out):
        pytest.skip(f"the package index is unreachable from here, so uv cannot resolve: {out.strip()[-300:]}")
    assert r.returncode == 0 and not REFUSAL.search(out), out[-2000:]
    would = {m.group(1).lower(): m.group(2) for m in re.finditer(r"^\s*\+\s*([A-Za-z0-9_.\-]+)==(\S+)", out, re.M)}
    pins = {k: v for k, v in _lock_pins().items() if k != "instruction-following-eval"}
    assert {k.replace("_", "-"): v for k, v in would.items()} == {k.replace("_", "-"): v for k, v in pins.items()}, out[-2000:]


def test_the_line_without_no_config_is_refused_from_inside_the_checkout(client_dry):
    """The control: the same line with --no-config removed, from the checkout root, is refused by name while the project table
    carries an exclude-newer-package entry that predates a pin of the lock, so the test above exercises that table."""
    if tomllib is None:
        pytest.skip("tomllib (Python 3.11+) is needed to read pyproject.toml's [tool.uv] table")
    table = tomllib.loads((REPO / "pyproject.toml").read_text()).get("tool", {}).get("uv", {}).get("exclude-newer-package", {})
    pinned = set(_lock_pins()) & {k.lower() for k in table}
    if "anthropic" not in pinned:
        pytest.skip("the project table no longer cuts off a package the client lock pins (anthropic): the refusal cannot occur")
    venv, hashed = client_dry
    line = _install_line().replace("uv pip install --no-config ", "uv pip install ", 1)
    r = _dry_run(line, venv, hashed)
    out = r.stdout + r.stderr
    if r.returncode != 0 and NETWORK.search(out) and not REFUSAL.search(out):
        pytest.skip(f"the package index is unreachable from here: {out.strip()[-300:]}")
    version = subprocess.run(["uv", "--version"], capture_output=True, text=True).stdout.strip()
    m = REFUSAL.search(out)
    assert r.returncode != 0 and m and m.group(1).lower() in pinned, (version, out[-1500:])
