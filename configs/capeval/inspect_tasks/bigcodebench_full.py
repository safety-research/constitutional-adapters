"""``bigcodebench_full`` — inspect_evals' stock BigCodeBench over the FULL
``v0.1.2`` complete split (all 1,140 task_ids, BigCodeBench-Hard INCLUDED),
executed in OWNED Modal Sandboxes.

Loaded by PATH from :mod:`sisteer.capeval.tiers.inspect_served`
(``load_repo_task``), like the sibling thin tasks. This file is the 200 leg's
builder (``bigcodebench_200.narrowed_bigcodebench`` — the stock task, its
solver, its ``verify()`` scorer, its pinned compose image, the position == id
assertion, the ``modal_owned`` provider) over the FULL pin
(``configs/capeval/tasks/bcb_full.json``, read through
``contracts.load_bcb_full_pin``, which refuses a pin that does not regenerate
from its rule: every complete-split id, no seed). Nothing here is a second
BigCodeBench, and nothing here is a second builder: the sibling module is
imported by path and its ONE builder is called with the full pin.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional, Sequence

from inspect_ai import Task

from sisteer.capeval.tinycap import contracts as C


def _sibling():
    """The 200 leg's module (``bigcodebench_200.py``, beside this file), by path —
    the SAME builder, the SAME compose check, the SAME docker-listing substitution."""
    path = Path(__file__).with_name("bigcodebench_200.py")
    spec = importlib.util.spec_from_file_location("sisteer_inspect_tasks.bigcodebench_200", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def bigcodebench_full(ids: Optional[Sequence[str]] = None, sandbox: str = C.BCB_SANDBOX_PROVIDER,
                      repo_root: Optional[str | Path] = None, pin: Optional[dict] = None, *,
                      generation_only: bool = False) -> "Task | tuple[Task, dict]":
    """The stock BigCodeBench task over the FULL pin (or a subset of it in pin
    order — a smoke head), executed in ``sandbox``. ``generation_only`` forwards
    to the ONE builder: the same task built and checked with its sandbox, then
    the sandbox dropped and ``(task, compose facts)`` returned — the GPU phase
    of the two-phase BigCodeBench contract."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    doc = pin if pin is not None else C.load_bcb_full_pin(root)
    return _sibling().narrowed_bigcodebench(C.CONTROL_BIGCODEBENCH_FULL, doc, ids, sandbox,
                                            generation_only=generation_only)


def selected_ids(task: Task) -> list[str]:
    """The pinned ids of a built task, in order (``BigCodeBench/<n>``)."""
    return [C.bcb_pinned_id(str(s.id)) for s in task.dataset]


def compose_facts(task: Task) -> dict:
    """The compose facts the driver records (for the manifest's route block)."""
    return _sibling().compose_facts(task)
