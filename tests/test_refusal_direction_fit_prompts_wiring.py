"""Stage 04's refusal direction fits on the paper fit's own training draw by default, and the job checks the installed copy first.

The committed refusal-direction objects were fit on data/arditi/fit_prompts.json (identity in data/arditi/fit_prompts.manifest.json),
whose text is not distributed; ``sisteer.data.arditi_fit_prompts`` re-creates it from the pinned upstream splits and
``00_setup.sh --what third_party`` installs it under each cluster's shared root. Pinned here: 04 hands the job
``arditi:harmful_train`` / ``arditi:harmless_train`` unless SISTEER_REFUSAL_HARMFUL / _HARMLESS say otherwise, the sbatch defaults
to the same two sets, and its preflight block, EXECUTED in bash over the invented stand-in upstream, passes on a verified
installed copy, fails (pre_ok=0, reason logged) on an absent or altered one, defaults the third-party root to the shared root's
third_party/ when the cluster's helper did not export it, never fetches, and is not consulted for sets that are not the paper's.
The end-to-end dry run of 04's submission line is in tests/test_refusal_direction_pin.py."""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

import arditi_fit_prompts_standin as st

REPO = Path(__file__).resolve().parents[1]
SBATCH = REPO / "scripts" / "sbatch" / "fit_refusal_direction.sbatch"
FRONT = REPO / "reproduce" / "04_fit_objects.sh"
sys.path.insert(0, str(REPO / "scripts" / "arditi"))
import fit_refusal_direction as frd  # noqa: E402

#: stand-in for ``"$PY" -m sisteer.data.arditi_fit_prompts``: the module's own main() with the stand-in's manifest and a
#: network that refuses (a compute node has no egress), every invocation logged
_DRIVER = """\
import os, sys
sys.path[:0] = [{src!r}, {tests!r}]
from pathlib import Path
from sisteer.data import arditi_fit_prompts as m
import arditi_fit_prompts_standin as st
m.MANIFEST_PATH = st.write_manifest(Path(os.environ["STANDIN_MANIFEST"]))
m._open_url = st.refuse_network
sys.exit(m.main(sys.argv[1:]))
"""


def test_04_and_the_sbatch_default_to_the_paper_fits_training_draw():
    sb, front = SBATCH.read_text(), FRONT.read_text()
    assert 'HARMFUL="${REFUSAL_HARMFUL:-arditi:harmful_train}"' in sb and 'HARMLESS="${REFUSAL_HARMLESS:-arditi:harmless_train}"' in sb
    assert ('REFUSAL_HARMFUL=${SISTEER_REFUSAL_HARMFUL:-arditi:harmful_train} '
            'REFUSAL_HARMLESS=${SISTEER_REFUSAL_HARMLESS:-arditi:harmless_train}"') in front
    for src in (sb, front):
        assert ":-harmbench_val}" not in src and ":-alpaca}" not in src
    assert {"arditi:harmful_train", "arditi:harmless_train"} <= set(frd.ARDITI_SETS)


def _preflight_block() -> str:
    m = re.search(r'(# an arditi:\* set is the paper fit\'s own draw.*?\nesac\n)', SBATCH.read_text(), re.S)
    assert m, "no arditi preflight block in the sbatch"
    return m.group(1)


def _world(tmp_path: Path) -> dict:
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER.format(src=str(REPO / "src"), tests=str(REPO / "tests")))
    log = tmp_path / "py.log"
    py = tmp_path / "venv" / "bin" / "python"; py.parent.mkdir(parents=True)
    py.write_text("#!/bin/bash\n"
                  f'echo "$*" >> {shlex.quote(str(log))}\n'
                  f'if [ "$1" = -m ] && [ "$2" = sisteer.data.arditi_fit_prompts ]; then shift 2; exec {shlex.quote(sys.executable)} {shlex.quote(str(driver))} "$@"; fi\n'
                  f'exec {shlex.quote(sys.executable)} "$@"\n')
    py.chmod(0o755)
    shared = tmp_path / "shared"
    dest = shared / "third_party" / "refusal_direction" / st.COMMIT / "fit_prompts.json"
    return {"py": py, "log": log, "shared": shared, "dest": dest, "manifest": tmp_path / "manifest.json"}


def _run(w: dict, harmful: str = "arditi:harmful_train", harmless: str = "arditi:harmless_train", env: dict | None = None):
    body = (f'log(){{ echo "LOG $*"; }}\nPY={shlex.quote(str(w["py"]))}\nSHARED={shlex.quote(str(w["shared"]))}\n'
            f'HARMFUL={shlex.quote(harmful)}\nHARMLESS={shlex.quote(harmless)}\npre_ok=1\n'
            f'{_preflight_block()}echo "pre_ok=$pre_ok dir=$SISTEER_THIRD_PARTY_DIR offline=$SISTEER_THIRD_PARTY_OFFLINE"\n')
    base = {"PATH": os.environ["PATH"], "HOME": str(w["shared"].parent), "STANDIN_MANIFEST": str(w["manifest"])}
    return subprocess.run(["bash", "-c", body], capture_output=True, text=True, env={**base, **(env or {})}, timeout=120)


def _install(w: dict, data: bytes | None = None) -> None:
    w["dest"].parent.mkdir(parents=True, exist_ok=True)
    w["dest"].write_bytes(st.expected_document() if data is None else data)


def test_the_preflight_passes_on_the_verified_installed_copy_at_the_shared_root(tmp_path):
    w = _world(tmp_path)
    _install(w)
    r = _run(w)                                            # the helper did not export the root: the shared root's third_party/
    assert r.returncode == 0 and r.stdout.strip().endswith(f"pre_ok=1 dir={w['shared']}/third_party offline=1"), r.stdout + r.stderr
    assert "-m sisteer.data.arditi_fit_prompts fetch --offline" in w["log"].read_text()
    # a root and an offline flag the helper exported stand
    other = tmp_path / "elsewhere"
    (other / "refusal_direction" / st.COMMIT).mkdir(parents=True)
    (other / "refusal_direction" / st.COMMIT / "fit_prompts.json").write_bytes(st.expected_document())
    r = _run(w, env={"SISTEER_THIRD_PARTY_DIR": str(other), "SISTEER_THIRD_PARTY_OFFLINE": "1"})
    assert r.stdout.strip().endswith(f"pre_ok=1 dir={other} offline=1"), r.stdout + r.stderr


@pytest.mark.parametrize("state", ["absent", "altered"])
def test_the_preflight_fails_on_an_absent_or_altered_copy_and_names_the_setup_step(tmp_path, state):
    w = _world(tmp_path)
    if state == "altered":
        _install(w, st.expected_document().replace(b"Stand-in", b"Stand-ln", 1))
    r = _run(w)
    assert r.returncode == 0 and "pre_ok=0" in r.stdout, r.stdout + r.stderr
    assert "LOG PREFLIGHT the refusal-direction fit prompts (arditi:harmful_train / arditi:harmless_train) are not installed and verified" in r.stdout
    assert "reproduce/00_setup.sh --what third_party installs them" in r.stdout and "REFUSED: " in r.stderr
    assert not (w["dest"].exists() and state == "absent")             # nothing was fetched or written


def test_the_preflight_is_not_consulted_for_sets_that_are_not_the_papers(tmp_path):
    w = _world(tmp_path)
    r = _run(w, harmful="harmbench_val", harmless="alpaca")
    assert r.returncode == 0 and "pre_ok=1" in r.stdout, r.stdout + r.stderr
    assert not w["log"].exists()                                       # the interpreter was never invoked
    r = _run(w, harmful="/data/my_harmful.jsonl", harmless="arditi:harmless_train")   # one side is enough to check it
    assert "pre_ok=0" in r.stdout and w["log"].exists(), r.stdout + r.stderr
