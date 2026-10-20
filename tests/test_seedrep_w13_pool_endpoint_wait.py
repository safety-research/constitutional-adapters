"""The grading host's attacker-pool tools waited 600 s for the Server's `endpoint.json` while the 27B attacker was still
loading ~14 min after deploy, so every pool 'failed' before it could publish. The wait is ONE constant that must exceed the
Server's own readiness bound in spirit (>= 30 min) and every Modal tool that waits on it must outlive the wait."""
from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "scripts" / "modal" / "seedrep_attacker_pool.py"


def _module():
    return ast.parse(SRC.read_text())


def _const(name: str) -> int:
    for node in ast.walk(_module()):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return int(ast.literal_eval(node.value))
    raise AssertionError(f"{name} not defined")


def test_the_endpoint_publish_wait_covers_a_27b_model_load():
    assert _const("ENDPOINT_PUBLISH_WAIT_S") >= 1800, "a 27B attacker took > 14 min to load on H200:2 at W12; 600 s was the defect"


def test_every_endpoint_wait_derives_from_the_one_constant():
    src = SRC.read_text()
    assert "deadline = time.time() + 600" not in src, "a hard-coded 600 s endpoint deadline survived"
    assert "def _wait_endpoint_file(timeout_s: float = ENDPOINT_PUBLISH_WAIT_S)" in src
    assert len(re.findall(r"deadline = time\.time\(\) \+ ENDPOINT_PUBLISH_WAIT_S", src)) >= 1


def test_every_tool_that_waits_on_the_endpoint_outlives_the_wait():
    wait = _const("ENDPOINT_PUBLISH_WAIT_S")
    bad = []
    for node in ast.walk(_module()):
        if not isinstance(node, ast.FunctionDef):
            continue
        body_src = ast.unparse(node)
        if "_wait_endpoint_file(" not in body_src and "ENDPOINT_PUBLISH_WAIT_S" not in body_src:
            continue
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == "function":
                t = next((k.value for k in dec.keywords if k.arg == "timeout"), None)
                tv = ast.literal_eval(t) if t is not None else None
                if tv is None or tv <= wait:
                    bad.append(f"{node.name}: Modal timeout {tv} does not exceed ENDPOINT_PUBLISH_WAIT_S={wait}")
    assert not bad, bad
