"""The serving venv's KERNEL OVERLAY install (common.sh f22_overlay_install) and the job environment's NCCL interface.

The overlay (configs/capeval/vllm-requirements-{gdn,nemotron}.txt) installs with `--no-build-isolation --no-deps` on top of the
base serving venv, and every bootstrap runs with the checkout as its working directory (f22_init's cd). There uv reads
pyproject.toml's [tool.uv] table, whose extra-build-dependencies give causal-conv1d and mamba-ssm a `torch` build dependency
with match-runtime, and uv refuses any resolution holding either kernel without torch, which a --no-deps overlay always is:
every first build of a GDN or Nemotron serving venv failed that way. The overlay therefore installs with --no-config and the
base install's cutoff passed explicitly. These tests run that command for real (a uv dry run, from inside the checkout), show
the unfixed command is refused there, run the bootstrap's completion logic against a stubbed uv (a venv whose overlay failed
is never taken as complete), and hold the job scripts to exporting no NCCL socket interface (the site's nccl.conf decides)."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:          # Python 3.10
    tomllib = None

REPO = Path(__file__).resolve().parents[1]
F22 = REPO / "scripts" / "sdfbase" / "full22" / "common.sh"
CERT = REPO / "scripts" / "sdfbase" / "multilora_cert.sbatch"
FIT_ENV = REPO / "scripts" / "sbatch" / "sdfbase_fit_env.sbatch"
OVERLAYS = {
    "gdn": REPO / "configs" / "capeval" / "vllm-requirements-gdn.txt",
    "nemotron": REPO / "configs" / "capeval" / "vllm-requirements-nemotron.txt",
}
#: uv's refusal, as the failed builds logged it
REFUSAL = "was declared as an extra build dependency with `match-runtime = true`, but was not found in the resolution"
#: what a uv that cannot reach its index says; only then does the dry run skip (any other failure fails the test)
NETWORK = re.compile(r"(?i)failed to fetch|error sending request|dns error|tcp connect|connection (refused|reset|timed out)"
                     r"|network is unreachable|operation timed out|offline mode")


def _pins(path: Path) -> dict:
    out = {}
    for ln in path.read_text().splitlines():
        ln = ln.split("#", 1)[0].strip()
        if ln and "==" in ln:
            name, ver = ln.split("==", 1)
            out[name.strip().lower()] = ver.strip()
    return out


def _func(name: str) -> str:
    sh = F22.read_text()
    start = sh.index(f"{name}(){{")
    return sh[start:sh.index("\n}\n", start) + 3]


def _uv_env() -> dict:
    """The caller's environment minus the switches that would change uv's configuration discovery, so the dry run reads the
    checkout exactly as a job does."""
    return {k: v for k, v in os.environ.items() if k not in ("UV_NO_CONFIG", "UV_CONFIG_FILE")}


# --------------------------------------------------------------------------- the one overlay command, statically
def test_both_bootstraps_install_the_overlay_through_the_one_command():
    install = _func("f22_overlay_install")
    assert "--no-config" in install and "--no-build-isolation --no-deps" in install and '-r "$overlay"' in install
    assert "CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE" in install
    # the cutoff the project table used to supply is the base install's own, passed explicitly
    venv = _func("f22_serving_venv")
    base_cut = re.search(r"--exclude-newer (\S+) -r \"\$reqs\"", venv)
    over_cut = re.search(r"--exclude-newer (\S+)", install)
    assert base_cut and over_cut and base_cut.group(1) == over_cut.group(1) == "2026-08-03T00:00:00Z"
    # the base install itself is unchanged: it reads the table (its exclude-newer-package entries shape the base)
    assert "--no-config" not in venv
    assert venv.count('f22_overlay_install "$SERVE_VENV/bin/python" "$overlay" || exit 1') == 1
    cert = CERT.read_text()
    assert 'source "$F22_COMMON"' in cert and cert.count('f22_overlay_install "$VENV/bin/python" "$OVERLAY" || exit 1') == 1
    # no second spelling anywhere: the only --no-deps kernel install in either bootstrap is the shared one (comments may name it)
    for text in (venv, cert):
        code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
        assert not [ln for ln in code if "--no-build-isolation" in ln], code
    # the fit env's builder reads no uv configuration either (a submission from inside a checkout would hit the same refusal)
    assert 'uvpi() { uv pip install --no-config --python "$PYBIN" "$@"; }' in FIT_ENV.read_text()


# --------------------------------------------------------------------------- the one overlay command, run
@pytest.fixture(scope="module")
def empty_venv(tmp_path_factory):
    uv = shutil.which("uv")
    if not uv:
        pytest.skip("uv is not on PATH: the overlay dry run needs the uv the bootstraps call")
    venv = tmp_path_factory.mktemp("overlay-dry") / "venv"
    r = subprocess.run([uv, "venv", "-q", "--no-config", "--python", sys.executable, str(venv)], capture_output=True, text=True,
                       env=_uv_env(), timeout=300)
    assert r.returncode == 0, r.stderr
    return venv / "bin" / "python"


def _dry_run(cmd: str, py: Path, overlay: Path) -> subprocess.CompletedProcess:
    script = f'source "{F22}"\n{cmd}'
    return subprocess.run(["bash", "-c", script, "_", str(py), str(overlay)], cwd=REPO, capture_output=True, text=True,
                          env=_uv_env(), timeout=600)


@pytest.mark.parametrize("variant", sorted(OVERLAYS))
def test_the_overlay_resolves_its_pins_from_inside_the_checkout(variant, empty_venv):
    """f22_overlay_install itself (common.sh, sourced), with --dry-run appended, from the checkout root: resolves exactly the
    overlay file's pins, and uv never reaches the refusal."""
    overlay = OVERLAYS[variant]
    r = _dry_run('f22_overlay_install "$1" "$2" --dry-run', empty_venv, overlay)
    out = r.stdout + r.stderr
    if r.returncode != 0 and NETWORK.search(out):
        pytest.skip(f"the package index is unreachable from here, so uv cannot resolve: {out.strip()[-300:]}")
    assert r.returncode == 0 and REFUSAL not in out, out
    would = dict(m.groups() for m in re.finditer(r"^\s*\+\s*([A-Za-z0-9_.\-]+)==(\S+)", out, re.M))
    assert {k.lower(): v for k, v in would.items()} == _pins(overlay), out


def test_the_unfixed_command_is_refused_from_inside_the_checkout(empty_venv):
    """The control: the overlay command as it stood (no --no-config, no cutoff), from the checkout root, is refused by name
    while pyproject.toml declares the match-runtime torch build dependency, so the test above exercises that table."""
    if tomllib is None:
        pytest.skip("tomllib (Python 3.11+) is needed to read pyproject.toml's [tool.uv] table")
    table = tomllib.loads((REPO / "pyproject.toml").read_text()).get("tool", {}).get("uv", {}).get("extra-build-dependencies", {})
    if not any(isinstance(d, dict) and d.get("match-runtime") for deps in table.values() for d in deps):
        pytest.skip("pyproject.toml declares no match-runtime build dependency: the refusal this guards against cannot occur")
    r = _dry_run('CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE uv pip install --python "$1" --no-build-isolation '
                 '--no-deps -r "$2" --dry-run', empty_venv, OVERLAYS["nemotron"])
    out = r.stdout + r.stderr
    if r.returncode != 0 and NETWORK.search(out) and REFUSAL not in out:
        pytest.skip(f"the package index is unreachable from here: {out.strip()[-300:]}")
    version = subprocess.run(["uv", "--version"], capture_output=True, text=True).stdout.strip()
    assert r.returncode != 0 and REFUSAL in out, (version, out)


# --------------------------------------------------------------------------- a venv whose overlay failed is not complete
_STUB = r'''
set -u
source "$F22"
f22_log(){ echo "LOG $*"; }
f22_fatal(){ echo "FATAL $*"; exit 1; }
REPO_DIR="$REPO"
# the stub uv: venv makes an interpreter that answers the import checks from marker files; the base install marks BASE;
# the overlay install (the --no-deps call) marks OVERLAY, or fails when FAIL_OVERLAY=1
uv(){
  echo "uv $*" >> "$UVLOG"
  case "$1" in
    venv) local d="${@: -1}"; mkdir -p "$d/bin"; cp "$FAKEPY" "$d/bin/python"; chmod +x "$d/bin/python" ;;
    pip) case " $* " in
           *" --no-deps "*) [ "${FAIL_OVERLAY:-0}" = 1 ] && return 1; touch "$SERVE_VENV/OVERLAY" ;;
           *) touch "$SERVE_VENV/BASE" ;;
         esac ;;
  esac
}
f22_serving_venv gdn
echo "SENT=$(cat "$F22_SERVING_VENV_ROOT/.$(basename "$SERVE_VENV").vllm-bootstrap" 2>/dev/null) SHA=${SERVE_REQS_SHA:0:16}"
'''
_FAKEPY = r'''#!/bin/bash
v="$(dirname "$(dirname "$0")")"
case "${2:-}" in
  *"import fla"*|*"import mamba_ssm"*) [ -e "$v/OVERLAY" ] || exit 1; echo "kernel overlay OK (stub)" ;;
  *) [ -e "$v/BASE" ] || exit 1; echo "serving venv OK (stub)" ;;
esac
'''


def test_a_venv_whose_overlay_failed_is_rebuilt_and_never_served(tmp_path):
    """The bootstrap's completion marker (the sentinel) is written only after the base AND the overlay installed: a failed
    overlay leaves the base in place and no sentinel (the state the failed builds left on two nodes), the next call re-enters
    the bootstrap without recreating the venv and installs the overlay, the call after that reuses it, and a venv marked
    complete whose kernels are missing is refused by the overlay import check, never served on the pure-torch path."""
    if not shutil.which("flock"):
        pytest.skip("flock (util-linux) is not on PATH: the bootstrap holds its lock with it")
    cuda = tmp_path / "cuda"; (cuda / "bin").mkdir(parents=True)
    nvcc = cuda / "bin" / "nvcc"; nvcc.write_text("#!/bin/sh\nexit 0\n"); nvcc.chmod(0o755)
    fakepy = tmp_path / "fakepy"; fakepy.write_text(_FAKEPY); fakepy.chmod(0o755)
    root = tmp_path / "venvs"; uvlog = tmp_path / "uv.log"
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "USER": "tester", "F22": str(F22), "REPO": str(REPO),
           "F22_SERVING_VENV_ROOT": str(root), "CUDA_HOME": str(cuda), "FAKEPY": str(fakepy), "UVLOG": str(uvlog)}

    def call(**extra):
        uvlog.write_text("")
        r = subprocess.run(["bash", "-c", _STUB], capture_output=True, text=True, env={**env, **extra}, timeout=120)
        return r, uvlog.read_text().splitlines()

    # 1. the overlay fails: the venv keeps its base, carries no sentinel, and the job dies by name
    r, uv = call(FAIL_OVERLAY="1")
    assert r.returncode == 1 and "FATAL serving venv bootstrap failed (rc=1)" in r.stdout, (r.stdout, r.stderr)
    venv = next(root.glob("sisteer-petri-vllm-gdn-*"))
    assert (venv / "BASE").exists() and not (venv / "OVERLAY").exists()
    assert not list(root.glob(".*.vllm-bootstrap"))
    assert [l.split()[1] for l in uv] == ["venv", "pip", "pip"] and "--no-config" in uv[2]
    # 2. the next call takes that partial venv as INCOMPLETE: no new venv, the base re-audited, the overlay installed, sentinel
    r, uv = call()
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "already bootstrapped" not in r.stdout and "kernel overlay OK (stub)" in r.stdout
    assert [l.split()[1] for l in uv] == ["pip", "pip"] and "--no-config" in uv[1] and "--no-deps" in uv[1]
    sent = re.search(r"SENT=(\S*) SHA=(\S+)", r.stdout)
    assert sent and sent.group(1) == sent.group(2)
    # 3. complete: reused as is, uv never runs
    r, uv = call()
    assert r.returncode == 0 and "serving venv already bootstrapped (variant-gdn)" in r.stdout and uv == []
    # 4. marked complete but the kernels are gone: refused by the import check, loudly, never served without them
    (venv / "OVERLAY").unlink()
    r, uv = call()
    assert r.returncode == 1 and "FATAL serving venv variant-gdn: fla / causal_conv1d do not import" in r.stdout and uv == []


# --------------------------------------------------------------------------- NCCL: the site's interface, never the job's
#: the overlay interface name the job scripts used to export, assembled here so this file does not spell it
_SITE_IFACE = "vx" + "lan"
_ASSIGN = (re.compile(r"NCCL_SOCKET_IFNAME\s*="),                              # shell: export X=..., X=... cmd
           re.compile(r"""["']NCCL_SOCKET_IFNAME["']\s*(\]\s*=|:)"""),           # python: env["X"] = ..., {"X": ...}
           re.compile(r"""setdefault\(\s*["']NCCL_SOCKET_IFNAME"""))


def _code_files():
    roots = ("scripts", "reproduce", "corpus_generator", "src")
    try:
        names = subprocess.run(["git", "ls-files", "--", *roots], cwd=REPO, capture_output=True, text=True,
                               check=True).stdout.split("\n")
        paths = [REPO / n for n in names if n]
    except (OSError, subprocess.CalledProcessError):          # an exported tree without git
        paths = [p for r in roots for p in (REPO / r).rglob("*") if p.is_file()]
    for p in paths:
        if p.is_file() and p.suffix in ("", ".sh", ".sbatch", ".py", ".bash", ".env", ".txt", ".md", ".yaml", ".yml"):
            try:
                yield p, p.read_text()
            except UnicodeDecodeError:
                continue


def test_no_job_script_exports_or_names_an_nccl_socket_interface():
    """NCCL's socket interface is the site's to choose (its nccl.conf, else NCCL's own detection). An exported
    NCCL_SOCKET_IFNAME overrides the site's setting, and a name the node lacks fails the first collective with "no socket
    interface found", so no job script, template or helper sets it, and none names the interface the scripts used to export."""
    hits = []
    for p, text in _code_files():
        rel = p.relative_to(REPO)
        for n, line in enumerate(text.splitlines(), 1):
            if any(rx.search(line) for rx in _ASSIGN) or _SITE_IFACE in line.lower():
                hits.append(f"{rel}:{n}: {line.strip()[:160]}")
    assert not hits, "\n".join(hits)
