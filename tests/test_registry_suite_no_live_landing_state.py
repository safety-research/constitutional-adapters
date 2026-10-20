"""No test reads the host's LIVE landing state by default.

``sisteer.registry.reads_protocol.reads_dir`` resolves the cluster-read namespace -- the reservation, the op records and the receipts
the status page's "Cluster reads, store authority and the replica" section lists -- from ``$REGISTRY_LANDING_STATE``, default
``~/.local/state/registry-landing``, and on the grading host that default IS the landing chain's live state: every tick leaves a receipt there
naming the store paths it read. A status-page test that rendered without naming a state directory read those receipts into its page, and
a word ban over the rendered text failed on a store path a live receipt carried -- on the host that runs the landing chain and nowhere
else (``tests/test_registry_bon_decode_precedence.py``, the presquash-fu8 review). ``tests/conftest.py`` now points the variable at an
EMPTY directory for the whole session, in-process and in every child a test spawns, the shape of the no-live-spool belt beside it.
Pinned here: the namespace resolves under the suite's state directory and never under the live default, in-process and in a child;
the status page's section over it is the empty, green shape; a state directory a test names, or hands the section as ``state_dir``,
wins; and the known-bad shape, hermetically: with the variable deleted the namespace is the home default's, and a reservation standing
there is what the page shows -- what the fixture prevents."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sisteer.registry import reads_protocol as RP
from sisteer.registry import status

REPO = Path(__file__).resolve().parents[1]
LIVE_READS = os.path.realpath(os.path.join(RP.DEFAULT_STATE, RP.READS_SUBDIR))
FIXTURE_DIR_PREFIX = "no-live-landing-state"


def _suite_state() -> str:
    state = os.environ.get(RP.STATE_ENV, "")
    if not state or not os.path.basename(state).startswith(FIXTURE_DIR_PREFIX):
        pytest.skip(f"the caller named its own landing state ({RP.STATE_ENV}={state!r}): the session fixture yields to it")
    return state


def test_the_suite_points_the_namespace_at_an_empty_state_directory_never_the_live_default():
    state = _suite_state()
    assert os.path.isdir(state)
    rd = RP.reads_dir()
    assert rd == os.path.join(os.path.abspath(state), RP.READS_SUBDIR) and RP.reads_dir(None) == rd
    assert os.path.realpath(rd) != LIVE_READS and not os.path.realpath(rd).startswith(os.path.realpath(RP.DEFAULT_STATE))
    # nothing of a live chain under it: no reservation, no receipts, no op records
    assert not os.path.exists(RP.reservation_path(rd))
    assert RP.receipts(rd) == [] and not os.path.isdir(os.path.join(rd, RP.OPS_SUBDIR))


def test_the_status_pages_section_over_the_suite_state_is_the_empty_green_shape(tmp_path):
    _suite_state()
    so = status.slurm_only_section(str(tmp_path / "root"))          # no state_dir: the environment's, i.e. the suite's
    assert so["reservation"] == {"held": False, "reads_dir": RP.reads_dir(), "level": "green"}
    rc = so["receipts"]
    assert rc.get("rows") == [] and not rc.get("error") and rc.get("level") == "green", rc
    assert not any(p.startswith(("reservation", "receipts")) for p in so["problems"]), so["problems"]


def test_a_state_a_test_names_wins_and_so_does_the_sections_own_argument(tmp_path, monkeypatch):
    _suite_state()
    named = tmp_path / "named-state"
    monkeypatch.setenv(RP.STATE_ENV, str(named))
    assert RP.reads_dir() == str(named / RP.READS_SUBDIR)
    handed = tmp_path / "handed-state"
    so = status.slurm_only_section(str(tmp_path / "root"), state_dir=str(handed))
    assert so["reservation"]["reads_dir"] == str(handed / RP.READS_SUBDIR)


def test_the_known_bad_shape_hermetically_the_variable_deleted_reads_the_home_default_and_what_stands_there(tmp_path, monkeypatch):
    _suite_state()
    monkeypatch.delenv(RP.STATE_ENV)
    assert os.path.realpath(RP.reads_dir()) == LIVE_READS                 # the live default -- what the fixture prevents
    # the same shape over a synthetic home default carrying a reservation: the page reads it, naming it held
    home_state = tmp_path / "home" / ".local" / "state" / "registry-landing"
    rd = os.path.join(str(home_state), RP.READS_SUBDIR)
    os.makedirs(rd)
    with open(RP.reservation_path(rd), "w", encoding="utf-8") as fh:
        json.dump({"schema": RP.RESERVATION_SCHEMA, "owner": "tester", "op_id": "op-000001", "host": "TESTHOST", "pid": 2 ** 22 - 7,
                   "started_utc": "2026-10-10T00:00:00.000Z", "tree_sha": "c" * 40}, fh)
    monkeypatch.setattr(RP, "DEFAULT_STATE", str(home_state))
    assert RP.reads_dir() == rd
    so = status.slurm_only_section(str(tmp_path / "root"))
    assert so["reservation"]["held"] is True and so["reservation"]["reads_dir"] == rd
    # the variable back at an empty directory: the same reservation is invisible again
    monkeypatch.setenv(RP.STATE_ENV, str(tmp_path / "empty-state"))
    so = status.slurm_only_section(str(tmp_path / "root"))
    assert so["reservation"] == {"held": False, "reads_dir": str(tmp_path / "empty-state" / RP.READS_SUBDIR), "level": "green"}


def test_a_child_resolves_the_suite_state_too():
    state = _suite_state()
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    out = subprocess.run([sys.executable, "-c", "from sisteer.registry import reads_protocol as RP; print(RP.reads_dir())"],
                         env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == os.path.join(os.path.abspath(state), RP.READS_SUBDIR), out.stdout + out.stderr
    env.pop(RP.STATE_ENV)                                                 # a child whose env a test strips: the home default again
    out = subprocess.run([sys.executable, "-c", "from sisteer.registry import reads_protocol as RP; import os; print(os.path.realpath(RP.reads_dir()))"],
                         env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == LIVE_READS, out.stdout + out.stderr


def test_the_fixture_is_the_sessions_and_yields_to_the_callers_state():
    src = (REPO / "tests" / "conftest.py").read_text()
    i = src.index("def _no_live_landing_state_for_the_suite(tmp_path_factory):")
    assert src[:i].rstrip().endswith('@pytest.fixture(scope="session", autouse=True)')
    body = src[i:i + 3200]
    assert 'var = "REGISTRY_LANDING_STATE"' in body and "if os.environ.get(var):" in body
    assert f'tmp_path_factory.mktemp("{FIXTURE_DIR_PREFIX}")' in body and "os.environ[var] = str(state)" in body
    assert "os.environ.pop(var, None)" in body
