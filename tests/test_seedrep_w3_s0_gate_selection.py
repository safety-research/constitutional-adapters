"""Launch F-D: every pytest gate of the S0 runner's table SELECTS at least one test on the real tree.

The earlier table's G7 ran ``tests/test_seedrep_ladder_protocol.py -k superseded`` -> pytest rc 5, 15 deselected, 0 selected -> the gate
FAILED by construction (the superseded-attempt closure tests live in ``tests/test_seedrep_ladder_attempts.py``,
``test_seedrep_attempts.py`` renamed at the cut 8ffaebd4). ``pytest_verdict`` already refuses an empty selection at run time; this test
refuses it on the tree BEFORE S0, gate by gate, through pytest's own collection (``--collect-only -q``) -- the same argv shape the
runner builds (``-o addopts= -p no:cacheprovider`` + the table's targets), so a renamed test file or an unmatched ``-k`` fails here by
gate id, not at S0. Fails before the table fix (G7 collects nothing); passes after it.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _s0():
    spec = importlib.util.spec_from_file_location("seedrep_s0_gates_w3_selection", REPO / "scripts/modal/seedrep_s0_gates.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


S0 = _s0()
PYTEST_GATES = [g for g in S0.GATES if "pytest" in g]


def _collect(targets):
    """pytest's own collection of the gate's targets on THIS tree -> (rc, node ids)."""
    argv = [sys.executable, "-m", "pytest", "-o", "addopts=", "-p", "no:cacheprovider", "--collect-only", "-q"]
    for t in targets:
        argv += S0._split_target(t)
    env = {**os.environ, "PYTHONPATH": f"{REPO}/src" + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
    r = subprocess.run(argv, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=900)
    nodes = [ln.strip() for ln in r.stdout.splitlines() if "::" in ln and not ln.startswith(("=", "-", " "))]
    return r.returncode, nodes, (r.stdout + r.stderr)[-1500:]


def test_the_table_has_pytest_gates_and_every_target_file_exists():
    assert [g["id"] for g in PYTEST_GATES] == ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G9", "G10", "ARGV"]
    for g in PYTEST_GATES:
        for t in g["pytest"]:
            f = REPO / S0._split_target(t)[0]
            assert f.is_file(), (g["id"], t)


def _k_of(target: str):
    parts = S0._split_target(target)
    return parts[parts.index("-k") + 1] if "-k" in parts else None


def test_a_gate_carries_at_most_one_k_expression():
    """The runner concatenates every target into ONE pytest argv (`run_pytest_gate`: `argv += _split_target(t)`), and pytest keeps the LAST
    `-k` -- so two different `-k` filters in one gate silently deselect the earlier target's tests (a G9 spelled `-k logprobs` + `-k readiness`
    would drop the argv gate test itself). One expression per gate, or none."""
    for g in PYTEST_GATES:
        ks = {k for k in (_k_of(t) for t in g["pytest"]) if k is not None}
        assert len(ks) <= 1, (g["id"], g["pytest"])


@pytest.mark.parametrize("gate", PYTEST_GATES, ids=[g["id"] for g in PYTEST_GATES])
def test_every_pytest_gate_collects_at_least_one_test_on_the_real_tree(gate):
    rc, nodes, tail = _collect(gate["pytest"])
    assert rc == 0, f"{gate['id']}: collection rc {rc} (5 = nothing selected) for {gate['pytest']}\n{tail}"
    assert nodes, f"{gate['id']}: {gate['pytest']} selects no test on this tree\n{tail}"
    ks = {k for k in (_k_of(t) for t in gate["pytest"]) if k is not None}
    if ks:
        needle = ks.pop()
        assert all(needle in n for n in nodes), (gate["id"], needle, [n for n in nodes if needle not in n][:5])
    for t in gate["pytest"]:
        parts = S0._split_target(t)
        if "-k" not in parts:          # a bare file target must itself contribute (a `-k` target's share is the gate's, above)
            assert any(n.startswith(parts[0]) for n in nodes), (gate["id"], t)


def test_g7_names_the_superseded_attempt_files_not_the_protocol_file():
    """The F-D spelling: G7 selects the superseded-attempt closure tests (ladder_attempts) and the accepted-union publication test."""
    g7 = next(g for g in PYTEST_GATES if g["id"] == "G7")
    files = {S0._split_target(t)[0] for t in g7["pytest"]}
    assert "tests/test_seedrep_ladder_attempts.py" in files and "tests/test_seedrep_ladder_protocol.py" not in files
    rc, nodes, _ = _collect(["tests/test_seedrep_ladder_protocol.py -k superseded"])
    assert rc == 5 and nodes == []          # the old spelling still selects nothing: the fix is the table, not the tests
    rc, nodes, _ = _collect(g7["pytest"])
    names = {n.rsplit("::", 1)[-1] for n in nodes}
    assert rc == 0 and {"test_accepted_producer_dies_after_acceptance_then_a_superseded_reexecution_publishes_afterwards",
                        "test_the_pull_refuses_a_shard_with_no_accepted_attempt_and_a_later_complete_attempt_is_superseded",
                        "test_two_shards_plus_a_superseded_and_an_unaccepted_attempt_publish_exactly_the_accepted_union"} <= names
    assert S0.selftest() == 0
