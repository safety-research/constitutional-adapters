"""peft is REQUIRED in every serving venv: `configs/capeval/vllm-requirements.txt` pins `peft==0.20.0` (the repo's
uv.lock pin), and the three serving-venv bootstraps — the probe sbatch's guard,
`reproduce/02_prepare_model.sh serving_venv_sh`, and `scripts/sdfbase/full22/common.sh f22_serving_venv` — FAIL LOUD
when the install did not land or peft does not import, instead of proceeding on an interpreter that merely exists.
The verdict side (a pair without HF+PEFT evidence is never certified) is covered in tests/test_serving_probe_smoke.py.

`f22_serving_venv` is RUN under bash with `uv` / `flock` stubbed, not grepped: the fail-before case is `uv venv`
succeeding (bin/python exists) and `uv pip install` failing — the old function tested only `[ -x "$SERVE_VENV/bin/python" ]`
after the bootstrap subshell and went on.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REQS = REPO / "configs/capeval/vllm-requirements.txt"
COMMON = REPO / "scripts/sdfbase/full22/common.sh"
SBATCH = REPO / "scripts/sbatch/serving_probe.sbatch"
STAGE02 = REPO / "reproduce/02_prepare_model.sh"


def _pins(text: str) -> dict:
    pins = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        m = re.fullmatch(r"([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?==(\S+)", line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def test_requirements_pin_peft_to_the_lockfiles_release():
    pins = _pins(REQS.read_text())
    assert pins["peft"] == "0.20.0"
    lock = (REPO / "uv.lock").read_text()
    m = re.search(r'name = "peft"\nversion = "([^"]+)"', lock)
    assert m and m.group(1) == pins["peft"], "peft must be the lockfile's pin (a pinned release >= a week old)"
    # the comment cites the lock, and the pin sits with the serving stack, before the harness deps
    txt = REQS.read_text()
    assert "uv.lock" in txt[txt.index("# peft:"):txt.index("peft==0.20.0")]
    assert txt.index("transformers==5.14.1") < txt.index("peft==0.20.0") < txt.index("evalplus==")


def test_every_serving_venv_bootstrap_imports_peft_as_a_hard_guard():
    # the probe sbatch: peft in the SAME guard as vllm/torch, no tolerated ImportError
    sb = SBATCH.read_text()
    guard = sb[sb.index("\"$VENV/bin/python\" -c 'import vllm"):sb.index("unset VLLM_PLUGINS")]
    assert "import peft" in guard and "except ImportError" not in guard and "FINAL_RC=3; exit 3" in guard
    # stage 02's remote bootstrap string
    s2 = STAGE02.read_text()
    assert 'python" -c "import vllm, torch, peft;' in s2 and "REFUSED: vllm / torch / peft do not all import" in s2
    # common.sh f22_serving_venv: the subshell's rc decides, then the import guard
    fn = _func("f22_serving_venv")
    assert "local brc=$?" in fn and 'f22_fatal "serving venv bootstrap failed (rc=$brc)' in fn
    assert "import vllm, torch, peft" in fn and "peft==0.20.0 is REQUIRED" in fn
    # order: the rc check, the interpreter check, the import guard, THEN the overlay's own import check (the LAST
    # `if [ -n "$overlay" ]` — the first one is the overlay INSTALL inside the bootstrap subshell)
    assert fn.index("local brc=$?") < fn.index('[ -x "$SPY" ]') < fn.index("import vllm, torch, peft") < fn.rindex('if [ -n "$overlay" ]')


def _func(name: str) -> str:
    sh = COMMON.read_text()
    start = sh.index(f"{name}(){{")
    return sh[start:sh.index("\n}\n", start) + 3]


def _run_f22_serving_venv(tmp_path: Path, *, install_rc: int, python_body: str) -> subprocess.CompletedProcess:
    """f22_serving_venv under bash: `uv venv` installs a stub bin/python (a bash script standing in for the interpreter,
    body = `python_body`), `uv pip install` exits `install_rc`; flock is a no-op; REPO_DIR is the real checkout (the
    requirements bytes name the venv); the venv root is re-pointed from /home/$USER/venvs to a tmp dir through the function's
    own F22_SERVING_VENV_ROOT."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    fn = _func("f22_serving_venv")
    assert '"${F22_SERVING_VENV_ROOT:-/home/$USER/venvs}/' in fn          # the default root is unchanged
    fn = f"F22_SERVING_VENV_ROOT={tmp_path}/venvs\n{fn}"
    stub = tmp_path / "python_stub"
    stub.write_text(python_body)
    script = f"""set -u
f22_log(){{ echo "[log] $*"; }}
f22_fatal(){{ echo "FATAL: $*"; exit 1; }}
flock(){{ :; }}
uv(){{
  case "$1" in
    venv) local v="${{@: -1}}"; mkdir -p "$v/bin"; cp "{stub}" "$v/bin/python"; chmod +x "$v/bin/python"; return 0;;
    pip) return {install_rc};;
  esac
}}
REPO_DIR={REPO}; USER=tester
{fn}
f22_serving_venv
echo "SPY=$SPY"
"""
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=120)


PY_OK = "#!/bin/bash\necho 'serving venv OK 0.25.1 2.11.0 peft 0.20.0'"                       # every import succeeds
PY_NO_PEFT = "#!/bin/bash\ncase \"$*\" in *peft*) echo 'ModuleNotFoundError: peft' >&2; exit 1;; esac; echo ok"


def test_f22_serving_venv_fails_when_the_install_fails_after_the_venv_exists(tmp_path):
    """Fail-before: `uv venv` left bin/python behind, `uv pip install` failed, and the old function passed its only check
    (`[ -x bin/python ]`) — a half-built venv went on to serve. Now the subshell's rc is fatal."""
    r = _run_f22_serving_venv(tmp_path, install_rc=1, python_body=PY_OK)
    assert r.returncode == 1 and "FATAL: serving venv bootstrap failed (rc=1)" in r.stdout, r.stdout + r.stderr
    assert "SPY=" not in r.stdout
    h = hashlib.sha256(REQS.read_bytes()).hexdigest()[:8]
    assert (tmp_path / f"venvs/sisteer-petri-vllm-{h}/bin/python").is_file()          # the interpreter DID exist


def test_f22_serving_venv_fails_when_peft_does_not_import_and_passes_when_it_does(tmp_path):
    # a venv whose install landed (sentinel written) but where peft does not import: FATAL every time it is entered
    r = _run_f22_serving_venv(tmp_path / "nopeft", install_rc=0, python_body=PY_NO_PEFT)
    assert r.returncode == 1 and "FATAL: serving venv guard: vllm / torch / peft do not all import" in r.stdout, r.stdout + r.stderr
    assert "peft==0.20.0 is REQUIRED" in r.stdout
    r = _run_f22_serving_venv(tmp_path / "nopeft", install_rc=1, python_body=PY_OK)      # re-entered: still fatal, at the guard
    assert r.returncode == 1 and "already bootstrapped" in r.stdout and "do not all import" in r.stdout, r.stdout + r.stderr
    # a venv where every import succeeds: SPY is the hash-named interpreter
    r = _run_f22_serving_venv(tmp_path / "ok", install_rc=0, python_body=PY_OK)
    assert r.returncode == 0, r.stdout + r.stderr
    h = hashlib.sha256(REQS.read_bytes()).hexdigest()[:8]
    assert f"SPY={tmp_path}/ok/venvs/sisteer-petri-vllm-{h}/bin/python" in r.stdout
    # the sentinel was written on the successful path: a rerun says "already bootstrapped", runs the guard, passes
    r2 = _run_f22_serving_venv(tmp_path / "ok", install_rc=1, python_body=PY_OK)
    assert r2.returncode == 0 and "already bootstrapped" in r2.stdout, r2.stdout + r2.stderr
