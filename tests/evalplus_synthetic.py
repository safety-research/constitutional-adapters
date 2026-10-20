"""A synthetic two-task EvalPlus sample set and the results file shaped like EvalPlus 0.3.1's output.

The format tests of the capability harness need one samples file and one results file to pin the shapes the
harness exchanges with EvalPlus. These stand-ins replace the benchmark-derived fixture pair: the task ids keep
EvalPlus's id grammar, which the harness validates, but the solutions are invented and were never executed.
Task 0's results say it passes the base and plus tests, task 1's say it fails both, and the dataset hash is a
placeholder.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

SAMPLES: tuple[dict, ...] = (
    {"task_id": "HumanEval/0", "solution": "def stand_in_that_passes():\n    return True\n"},
    {"task_id": "HumanEval/1", "solution": "def stand_in_that_fails():\n    return False\n"},
)
_STATUS = {"HumanEval/0": "pass", "HumanEval/1": "fail"}
RESULTS: dict = {
    "date": "2026-08-15 00:00:00",
    "hash": "1111111122222222333333334444444455555555666666667777777788888888",
    "eval": {s["task_id"]: [{"task_id": s["task_id"], "solution": s["solution"],
                             "base_status": _STATUS[s["task_id"]], "plus_status": _STATUS[s["task_id"]]}]
             for s in SAMPLES},
}

SAMPLES_NAME = "samples.jsonl"
RESULTS_NAME = "eval_results_synthetic.json"


def samples() -> list[dict]:
    """A fresh copy of the two sample rows."""
    return copy.deepcopy(list(SAMPLES))


def samples_jsonl() -> str:
    """The samples as EvalPlus reads them: one JSON object per line, newline-terminated."""
    return "".join(json.dumps(s) + "\n" for s in SAMPLES)


def eval_results() -> dict:
    """A fresh copy of the results document."""
    return copy.deepcopy(RESULTS)


def write(directory: Path) -> tuple[Path, Path]:
    """Write ``samples.jsonl`` and ``eval_results_synthetic.json`` into ``directory``; return both paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    samples_path = directory / SAMPLES_NAME
    results_path = directory / RESULTS_NAME
    samples_path.write_text(samples_jsonl())
    results_path.write_text(json.dumps(RESULTS, indent=2) + "\n")
    return samples_path, results_path
