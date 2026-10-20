"""Capability metrics for a DefenseCell — the WHOLE defense, or a loud refusal.

The raw capability entries (``capability.mmlu_accuracy``, the
``capability_ext`` panel) take an intervention object and an optional
``system`` prompt as SEPARATE arguments, so nothing stops a caller handing
them HALF a defense — the object of a stacked cell without its SI, or the SI
without its object — and labelling the number as the full cell. A silent
partial serve is indistinguishable from a correct one downstream; an
uncommitted serving path once served an object alone under a stacked label
and every downstream check read it as valid. This module closes that hole
for the capability surface: hand it the CELL, and it either serves
everything the cell names or refuses loudly.

(The full generation+judging pipeline for cells is
``scripts/cell_controls_run.py`` via ``eval.cell_controls.cell_serving``,
which threads one serving resolution into both its parity and judged legs —
this wrapper is the direct capability-metric twin for the entries here.)

Dispatch is CLOSED-WORLD on ``cell.class_`` — an unknown class refuses by
name, never falls through to a partial defense:

- ``baseline`` with no ``si_id`` (the ``none`` cell)  -> no object, no system;
- ``baseline`` with ``si_id`` (``prompted__<si>``)    -> system only;
- ``A`` / ``B`` (a compiled object cell)              -> object only;
- ``stacked`` (``stacked__<si>__<arm>``)              -> BOTH the object and
  the SI system prompt, together.

The ``stacked`` class is defined on the stacked-defense registry; cells are
duck-typed here (the same convention ``cell_controls`` uses), so this
wrapper serves those cells as soon as that registry produces them. Every
half is resolved from the cell itself — the SI text through
``jailbreak.resolve_instruction`` (the same resolution the JB serving legs
use) and the object through ``methods.load_object`` from
``cell.object_path`` — so a caller never assembles halves by hand.
"""

from __future__ import annotations

from ..methods import load_object
from .capability_ext import (
    gpqa_diamond_accuracy,
    gsm8k_accuracy,
    mmlu_pro_accuracy,
    mmlu_under_obj,
)
from .jailbreak import resolve_instruction

#: Cell classes this wrapper knows how to serve IN FULL. Anything else —
#: including a class added by a future registry — refuses by name below.
SERVABLE_CLASSES = ("baseline", "A", "B", "stacked")

#: The cell-servable capability metrics: the object-generic entries whose
#: ``obj`` an artifact loaded by ``methods.load_object`` satisfies.
#: ``capability.mmlu_accuracy`` is deliberately absent — its ``sv`` is a raw
#: SteeringVector, not an InterventionObject, so a loaded cell artifact
#: cannot serve through it; ``mmlu_under_obj`` is its object-generic twin.
CAPABILITY_METRICS = {
    "mmlu": mmlu_under_obj,
    "gsm8k": gsm8k_accuracy,
    "gpqa_diamond": gpqa_diamond_accuracy,
    "mmlu_pro": mmlu_pro_accuracy,
}


def capability_serving_for_cell(cell, model, tok, *, load=load_object):
    """``(obj, system)`` for one cell — the FULL defense it names, or a loud
    ``ValueError``. Mirrors ``jailbreak.build_defense_cell`` /
    ``cell_controls.cell_serving`` dispatch and SI resolution; ``load`` is a
    test seam (defaults to the same ``methods.load_object`` the JB serving
    path uses).

    Refusals happen BEFORE any half is materialised, so a caller can never
    end up holding half a stack: an unknown class, a compiled/stacked cell
    with no ``object_path``, or a stacked cell with no ``si_id`` all raise
    with nothing served."""
    cls = getattr(cell, "class_", None)
    cid = getattr(cell, "id", "<no id>")
    if cls not in SERVABLE_CLASSES:
        raise ValueError(
            f"[capability-cell] cell {cid!r} has class {cls!r}, which this "
            f"wrapper does not know how to serve (servable: "
            f"{SERVABLE_CLASSES}). Refusing rather than guessing at a "
            "partial defense — a silent partial serve is indistinguishable "
            "from a correct one downstream.")
    if cls == "baseline":
        if cell.si_id is None:  # the undefended `none` cell
            return None, None
        return None, resolve_instruction(cell.si_id).text
    # A compiled half is involved (A / B / stacked): validate the WHOLE cell
    # before serving any part of it.
    if cell.object_path is None:
        raise ValueError(
            f"[capability-cell] {cls} cell {cid!r} has no object_path "
            "(registry built without a defenses_dir?) — cannot serve its "
            "object half, refusing to serve anything.")
    system = None
    if cls == "stacked":
        if cell.si_id is None:
            raise ValueError(
                f"[capability-cell] stacked cell {cid!r} has no si_id — "
                "cannot serve its system half, refusing to serve the object "
                "alone.")
        system = resolve_instruction(cell.si_id).text
    obj = load(cell.object_path, model, tok)
    return obj, system


def capability_for_cell(model, tok, cell, metric="mmlu", *,
                        scale: float = 1.0, load=load_object,
                        **metric_kwargs):
    """One capability number for one cell, with the whole defense served.

    ``metric`` is a name in :data:`CAPABILITY_METRICS`, or a callable with
    the ``capability_ext`` signature
    ``fn(model, tok, obj, *, scale, system, **kw)`` (the callable form is a
    test seam). ``metric_kwargs`` (``rows=``, ``n=``, ``seed=``,
    ``batch_size=``, ...) forward to the metric unchanged — but never
    ``obj``/``system``, which only the cell may supply (passing either here
    is a loud ``TypeError``)."""
    if callable(metric):  # resolved BEFORE serving: a bad metric name must
        fn = metric       # not load an object first
    else:
        try:
            fn = CAPABILITY_METRICS[metric]
        except KeyError:
            raise ValueError(
                f"[capability-cell] unknown metric {metric!r}; known: "
                f"{tuple(CAPABILITY_METRICS)}") from None
    obj, system = capability_serving_for_cell(cell, model, tok, load=load)
    return fn(model, tok, obj, scale=scale, system=system, **metric_kwargs)
