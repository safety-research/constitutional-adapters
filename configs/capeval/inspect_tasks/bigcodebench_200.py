"""``bigcodebench_200`` — inspect_evals' stock BigCodeBench over the pinned
200 non-Hard ids, executed in OWNED Modal Sandboxes.

Loaded by PATH from
:mod:`sisteer.capeval.tiers.inspect_served` (``load_repo_task``), like the
sibling thin tasks. Nothing here is a second BigCodeBench: the task is the
stock ``inspect_evals.bigcodebench.bigcodebench(subset=..., version="v0.1.2",
sandbox="modal_owned")`` — its instruction prompt, its ``generate()`` solver,
its ``verify()`` scorer and its own compose (the pinned execution image) — and
this file only

* NARROWS it to the pin (``configs/capeval/tasks/bcb_200.json``, read through
  ``contracts.load_bcb_pin``, which refuses a pin that does not regenerate
  from its rule) or to any caller-supplied SUBSET of the pin in pin order (a
  smoke head, the truncation ladder's selection);
* asserts, per sample, that the stock builder's position-indexed ``subset``
  returned the id the pin names (``sample.id == int(suffix)``; the split's ids
  are contiguous, and this is where that assumption is checked);
* names the ``modal_owned`` provider (``sisteer.capeval.tiers.sandbox_owned``,
  imported here so Inspect's registry knows the name), and REFUSES unless the
  task's compose declares exactly one service whose image is the pinned digest
  (``contracts.BCB_SANDBOX_IMAGE``) with ``network_mode: none`` — a compose
  that moved is another verifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from inspect_ai import Task

from sisteer.capeval.tinycap import contracts as C


def images_from_compose_yaml(compose_path: str | Path) -> list:
    """The ``image:`` of every service in a compose file, read from the YAML —
    what ``docker compose -f <file> config`` prints for a compose that
    declares images only (the stock BigCodeBench one), without the docker CLI."""
    import yaml

    doc = yaml.safe_load(Path(compose_path).read_text(encoding="utf-8"))
    return [str(svc["image"]) for svc in (doc.get("services") or {}).values() if svc.get("image")]


def _substitute_docker_image_listing(module) -> None:
    """inspect_evals 0.18.0's ``bigcodebench()`` constructor calls
    ``get_images_from_compose`` UNCONDITIONALLY (before the docker-handling
    branch), and that helper shells out to ``docker compose … config`` — a
    binary the Modal serving container does not have, so the stock constructor
    cannot even be called there (verified 2026-09-10: ``FileNotFoundError:
    'docker'`` from ``bigcodebench.py:113``; the plan's "DEFAULT runs no docker
    subprocess" was true of ``force_build_or_pull_docker_image`` alone). The
    ONE substitution this thin task makes: the module-level name is rebound to
    a YAML read of the same compose, which returns the same ``image:`` list;
    everything else in the constructor — the dataset load, the samples, the
    solver, the ``verify()`` scorer, the sandbox tuple, the version metadata —
    is the stock code, unchanged. Idempotent; pinned by test against the stock
    source so a pin bump that drops the docker call is noticed."""
    if getattr(module.get_images_from_compose, "__name__", "") != "images_from_compose_yaml":
        module.get_images_from_compose = images_from_compose_yaml


def _check_compose(compose_path: str) -> dict:
    """The stock compose, held to the contracts: one service, the pinned
    image, no network."""
    from inspect_ai.util import parse_compose_yaml

    cfg = parse_compose_yaml(compose_path, multiple_services=False)
    services = dict(cfg.services)
    if len(services) != 1:
        raise ValueError(f"bigcodebench_200: compose {compose_path} declares {len(services)} services, not 1")
    (name, svc), = services.items()
    if svc.image != C.BCB_SANDBOX_IMAGE:
        raise ValueError(
            f"bigcodebench_200: the compose image {svc.image!r} is not the pinned "
            f"contracts.BCB_SANDBOX_IMAGE {C.BCB_SANDBOX_IMAGE!r} — a moved verifier image is another "
            "instrument; re-pin deliberately")
    if svc.network_mode != "none":
        raise ValueError(f"bigcodebench_200: compose network_mode {svc.network_mode!r} != 'none'")
    if svc.build:
        raise ValueError("bigcodebench_200: the compose declares a build — only the pinned image is served")
    return {"service": name, "image": svc.image, "network_mode": svc.network_mode,
            "cpus": svc.cpus, "mem_limit": svc.mem_limit}


def narrowed_bigcodebench(label: str, doc: dict, ids: Optional[Sequence[str]], sandbox: str, *,
                          generation_only: bool = False) -> "Task | tuple[Task, dict[str, Any]]":
    """The stock BigCodeBench task narrowed to ``ids`` — the pin ``doc`` (default)
    or a SUBSET of it in pin order (refused otherwise: a foreign or re-ordered id
    list is not this leg) — executed in ``sandbox`` (``modal_owned``; the parent
    ``modal`` provider is accepted for a hand run that WANTS untagged sandboxes
    and says so; anything else refuses). ONE builder for the 200 leg and its FULL
    sibling (capeval-full-tasks, 2026-09-23): ``label`` names the leg in every
    refusal; ``doc`` is the leg's VERIFIED pin.

    ``generation_only`` (the GPU
    half of BigCodeBench verification as a CPU phase): the task is built EXACTLY as
    above — with ``sandbox``, the pin-order / position == id / ``task_id_metadata``
    assertions, the sandbox type check and the compose check all run — then the
    compose facts are captured and ``task.sandbox`` is set to ``None`` (a plain
    attribute on inspect_ai 0.3.260's ``Task``), so ``eval_async(score=False)``
    allocates NO sandbox for any sample; the stock ``generate()`` solver and the
    ``verify()`` scorer STAY on the task (the scorer never runs under
    ``score=False``, and its name is what the plan's primary-scorer check reads).
    Returns ``(task, facts)`` in that mode — the facts are what ``compose_facts``
    would have read off the sandboxed task, which a sandbox-less task cannot
    supply. The default returns the task alone, byte-identical to before."""
    # lazy, like the sibling thin tasks: a config file should not pull the
    # eval stack in just to be read
    import importlib

    import sisteer.capeval.tiers.sandbox_owned  # noqa: F401  (registers `modal_owned`)

    # the MODULE (the package's __init__ re-exports the function under the
    # same name, so `from inspect_evals.bigcodebench import bigcodebench` is
    # the function, not the module whose global the substitution rebinds)
    _bcb_module = importlib.import_module("inspect_evals.bigcodebench.bigcodebench")
    _substitute_docker_image_listing(_bcb_module)
    bigcodebench = _bcb_module.bigcodebench
    pinned = [str(i) for i in doc["ids"]]
    if ids is None:
        chosen = list(pinned)
    else:
        chosen = [str(i) for i in ids]
        if len(set(chosen)) != len(chosen):
            raise ValueError(f"{label}: duplicate ids in the selection")
        foreign = [i for i in chosen if i not in pinned]
        if foreign:
            raise ValueError(f"{label}: ids outside the pin: {foreign[:5]} — the leg is the pin "
                             "or a subset of it, never another item set")
        order = {i: k for k, i in enumerate(pinned)}
        if chosen != sorted(chosen, key=order.__getitem__):
            raise ValueError(f"{label}: the selection is not in pin order")
    if not chosen:
        raise ValueError(f"{label}: an empty selection")
    if sandbox not in (C.BCB_SANDBOX_PROVIDER, "modal"):
        raise ValueError(f"{label}: sandbox {sandbox!r} is not {C.BCB_SANDBOX_PROVIDER!r} "
                         "(the owned provider) — refusing an unknown provider")
    positions = [int(C.bcb_log_id(i)) for i in chosen]
    task = bigcodebench(subset=positions, version=C.BCB_SPLIT, sandbox=sandbox)
    # the stock builder indexes the split by POSITION; the pin's ids are the
    # split's task_ids, contiguous, so position == id suffix — asserted per sample
    got = [s.id for s in task.dataset]
    if got != positions:
        raise ValueError(f"{label}: the stock builder returned ids {got[:5]}... for positions "
                         f"{positions[:5]}... — the split's ids are not contiguous; refusing")
    for s in task.dataset:
        want = f"BigCodeBench/{s.id}"
        if s.metadata.get("task_id_metadata") != want:
            raise ValueError(f"{label}: sample {s.id} carries task_id {s.metadata.get('task_id_metadata')!r}"
                             f" != {want!r}")
    stype, compose = task.sandbox.type, task.sandbox.config
    if stype != sandbox or not isinstance(compose, str):
        raise ValueError(f"{label}: the task's sandbox is {(stype, compose)!r}, expected "
                         f"({sandbox!r}, <compose path>)")
    facts = _check_compose(compose)
    if generation_only:
        # every check above ran against the SANDBOXED build; only now
        # is the sandbox dropped, so the generation-only task is the coupled task
        # minus its sandbox — same dataset, same ids, same metadata, same solver,
        # same scorer (kept, never run), same version
        task.sandbox = None
        return task, facts
    return task


def bigcodebench_200(ids: Optional[Sequence[str]] = None, sandbox: str = C.BCB_SANDBOX_PROVIDER,
                     repo_root: Optional[str | Path] = None, pin: Optional[dict] = None, *,
                     generation_only: bool = False) -> "Task | tuple[Task, dict[str, Any]]":
    """The stock BigCodeBench task narrowed to the 200 PIN (or a subset of it in
    pin order), executed in ``sandbox`` — :func:`narrowed_bigcodebench` over
    ``contracts.load_bcb_pin``. ``generation_only`` forwards."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    doc = pin if pin is not None else C.load_bcb_pin(root)
    return narrowed_bigcodebench("bigcodebench_200", doc, ids, sandbox, generation_only=generation_only)


def selected_ids(task: Task) -> list[str]:
    """The pinned ids of a built task, in order (``BigCodeBench/<n>``)."""
    return [C.bcb_pinned_id(str(s.id)) for s in task.dataset]


def compose_facts(task: Task) -> dict[str, Any]:
    """The compose facts the driver records (for the manifest's route block). A
    GENERATION-ONLY task carries no sandbox to read them off — its
    builder returned them beside the task; asking here refuses by name rather than
    failing on ``None.config``."""
    if task.sandbox is None:
        raise ValueError("compose_facts: the task carries no sandbox (a generation-only build, plan §2.2.1) — "
                         "the compose facts are the second element `narrowed_bigcodebench(..., "
                         "generation_only=True)` returned")
    return _check_compose(task.sandbox.config)
