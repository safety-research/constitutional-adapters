"""The PROMPT RENDERER for the canonical IFEval battery.

The 541 prompts of inspect_evals' stock ``ifeval`` task, rendered EXACTLY as the served task sends them: the stock
dataset (``google/IFEval`` @ ``966cd895``, split ``train``, ``record_to_sample`` — the vendor's own row ``key`` as the
sample id, the ``prompt`` verbatim as the ONE user turn) through the task's own solver chain (``[generate()]``) with
``generate`` a no-op (``inspect_served._render_task_messages``), the prompted cell's SI folded into ONE leading system
turn the way the provider collapses it (``collapse_system_messages_dicts``).

WHY NOT ``inspect_evals.ifeval.ifeval()``: the stock task function asserts the optional dependency
``instruction_following_eval`` (the scorer's checker) BEFORE it loads the dataset, and the CPU venvs that measure prompt
lengths do not carry the checker. Rendering needs the dataset and the solver only, so this module builds
``Task(dataset=hf_dataset(...record_to_sample...), solver=[generate()])`` from the stock task's OWN constants and
``record_to_sample`` — byte-identical messages, no checker. Merged: ``inspect_served.ifeval_messages`` (through the
real ``_plan_task`` branch, the stock task built) is the PRODUCTION renderer and :func:`ifeval_messages` is a pure
delegate to it; :func:`_local_ifeval_messages` is the checker-free CROSS-CHECK construction, and
``tests/test_ifeval_messages.py`` pins that the two agree message for message.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from sisteer.capeval.tinycap import ifeval_contracts as IC

MESSAGES_SOURCE = ("sisteer.capeval.tiers.ifeval_messages.ifeval_messages (inspect_evals ifeval: google/IFEval @ "
                   f"{IC.IFEVAL_DATASET_REVISION[:8]} through the task's own solver chain [generate()], generate a no-op)")
PROMPTED_MESSAGES_SOURCE = MESSAGES_SOURCE + " + the cell's system turn folded into the leading system message (collapse_system_messages_dicts)"


class IfevalRenderError(RuntimeError):
    """The stock dataset or task is not what this instrument pins."""


def stock_module() -> Any:
    """The stock task MODULE (``inspect_evals.ifeval.ifeval``) — imported by name because the package's ``__init__``
    re-exports the ``ifeval`` task FUNCTION under the same attribute."""
    return importlib.import_module("inspect_evals.ifeval.ifeval")


def ifeval_render_task() -> Any:
    """``Task(dataset=<the stock dataset at the pinned revision>, solver=[generate()])`` — the stock task minus its scorer
    (and minus the checker dependency the scorer imports). Refuses a dataset path / revision the module does not spell as
    this instrument pins, or a dataset of another size than 541."""
    from inspect_ai import Task
    from inspect_ai.solver import generate
    from inspect_evals.utils.huggingface import hf_dataset

    mod = stock_module()
    if mod.DATASET_PATH != IC.IFEVAL_DATASET_PATH or mod.IFEVAL_DATASET_REVISION != IC.IFEVAL_DATASET_REVISION:
        raise IfevalRenderError(
            f"the installed inspect_evals ifeval task pins {mod.DATASET_PATH} @ {mod.IFEVAL_DATASET_REVISION[:8]}, not this "
            f"instrument's {IC.IFEVAL_DATASET_PATH} @ {IC.IFEVAL_DATASET_REVISION[:8]} — the dataset moved")
    dataset = hf_dataset(path=mod.DATASET_PATH, split="train", sample_fields=mod.record_to_sample,
                         revision=mod.IFEVAL_DATASET_REVISION)
    n = len(dataset)
    if n != IC.IFEVAL_N:
        raise IfevalRenderError(f"the IFEval dataset has {n} records, not the frozen {IC.IFEVAL_N}")
    return Task(dataset=dataset, solver=[generate()], scorer=None)


def ifeval_ids(task: Any = None) -> List[str]:
    """The 541 sample ids (``str(record['key'])``), in dataset order — the item keys of the seed rule."""
    task = task if task is not None else ifeval_render_task()
    ids = [str(s.id) for s in task.dataset]
    if len(set(ids)) != len(ids):
        raise IfevalRenderError("the IFEval dataset repeats a sample id")
    return ids


def _local_ifeval_messages(repo_root: Optional[Path] = None, *, system: Optional[str] = None
                           ) -> List[List[Dict[str, str]]]:
    from sisteer.capeval.tiers import inspect_served as isv

    task = ifeval_render_task()
    ids = ifeval_ids(task)
    plan = isv.TaskPlan(row_task=IC.IFEVAL_TASK, name=IC.IFEVAL_TASK_NAME, task=task, sample_ids=ids,
                        expected_n=IC.IFEVAL_N, n_run=IC.IFEVAL_N, item_set="ifeval541")
    rendered = isv._render_task_messages(plan, ids, what=IC.IFEVAL_TASK_NAME)
    for msgs, iid in zip(rendered, ids):
        if [m["role"] for m in msgs] != ["user"]:
            raise IfevalRenderError(f"ifeval {iid}: the stock render is ONE user turn; got roles {[m['role'] for m in msgs]}")
    if system:
        rendered = [isv.collapse_system_messages_dicts(msgs, system) for msgs in rendered]
    return rendered


def ifeval_messages(repo_root: Optional[Path] = None, *, system: Optional[str] = None) -> List[List[Dict[str, str]]]:
    """The exact chat messages the IFEval plan sends for each of its 541 items, in dataset order — one OpenAI-shaped
    ``{"role", "content"}`` list per item — plus, for a PROMPTED cell, the cell's ``system`` text as the ONE leading
    system turn. A pure delegate to the production renderer ``inspect_served.ifeval_messages`` (the stock task
    through its own ``_plan_task`` branch); :func:`_local_ifeval_messages` is the checker-free cross-check."""
    from sisteer.capeval.tiers import inspect_served as isv

    if repo_root is not None:
        return isv.ifeval_messages(repo_root, system=system)
    return isv.ifeval_messages(system=system)


def messages_source(*, prompted: bool) -> str:
    return PROMPTED_MESSAGES_SOURCE if prompted else MESSAGES_SOURCE


__all__ = ["IfevalRenderError", "MESSAGES_SOURCE", "PROMPTED_MESSAGES_SOURCE", "ifeval_ids", "ifeval_messages",
           "ifeval_render_task", "messages_source", "stock_module"]
