"""The thinking-header / attestation steps a lens STANDUP performs, factored
out of ``scripts/capctrl_serving_up.py`` so the cluster standup
(``scripts/sdfbase/mt_serving_up.py``) runs the SAME code: ``mt_serving_up``
stays the cluster standup, capctrl's four export calls are ported through
this module rather than re-typed.

Torch-free and import-light on purpose: every ``sisteer`` import here is
lazy, so the module loads in the client venv, in the serving venv and in the
grading host test venv alike. Nothing here derives a header differently from the
functions it wraps — it only names the ORDER and the SHAPE the two standups
share:

1. :func:`lens_resolved_specs` — the resolved specs of the canonical lens
   cells (the ones a serve declaration can ride).
2. :func:`declares_thinking` / :func:`thinking_on_header_ids` — whether any
   exported cell declares the thinking-ON render (``@thinking``), and the
   SECOND header the registry then needs. On a header-INVARIANT template
   (gemma-4: the thinking switch leaves the assistant header in place) the
   ON header EQUALS the OFF header and is recorded as such — the declaration
   is admitted and the boundary is the same under both renders. A header that
   does not move is no reason not to serve or steer thinking-ON.
3. :func:`standup_thinking_header` — the fact for the routing table /
   endpoint.json / the proxy: HARD for a declaring cell, non-fatal otherwise.
4. the three ``attested_*_at`` readers over the workers' steering attestation
   (reduction mode, resumed-prefill policy, serve declarations) — each the
   ONE strict reader ``lens_worker`` exports, never a local re-read.
5. :func:`backend_timeout_kwargs` — the ``HTTPBackend(timeout_s=)`` kwarg only
   when a standup was given ``--upstream-timeout-s``.
6. :func:`served_positions_passthrough` — the ``lens_served_positions``
   routing-table stamp handed to ``RoutingProxy`` WHEN the constructor
   on this tree accepts it, recorded either way so a reader never has to
   infer whether the proxy could enforce the served-positions rule.
"""
from __future__ import annotations

import inspect
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: the routing-table key AND the ``RoutingProxy`` kwarg pinned for the
#: per-cell SERVED-positions stamp: symmetric, matching the
#: ``thinking_header`` convention; ``lens_`` marks a lens-route fact like
#: ``lens_serve_declarations``. Read here with ``.get`` — the key exists only
#: once the proxy refinement has landed.
LENS_SERVED_POSITIONS_KEY = "lens_served_positions"
#: endpoint.json's record of whether the stamp REACHED the proxy: ``passed``
#: (the constructor accepted the kwarg — the served-positions rule is live),
#: else the reason it could not be handed over (this tree's constructor lacks
#: the kwarg: the proxy then fails closed to ``thinking_on_undeclared`` on
#: every undeclared lens-route thinking-ON request, exactly today's behaviour).
LENS_SERVED_POSITIONS_PASSTHROUGH_KEY = "lens_served_positions_passthrough"
PASSTHROUGH_PASSED = "passed"


# --------------------------------------------------------------------------- #
# the workers' attestation, read through lens_worker's own readers            #
# --------------------------------------------------------------------------- #


def _load_attestation(path) -> tuple[Any, str | None]:
    """The steering attestation at ``path`` as ``(document, None)``, or
    ``(None, "<Error>: <why>")`` when the file is missing or unreadable — the
    readers below record that in endpoint.json, never guess."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def attested_resumed_prefill_policy_cells_at(path):
    """``(mapping, error)`` — the per-cell RESUMED-PREFILL POLICY the launched
    workers attest (``resumed_prefill_policy_cells`` on every top-level install
    ``worker_reports`` entry of the steering attestation at ``path``: cell id ->
    ``never`` | ``always`` | ``conditional``), through
    ``lens_worker.resumed_prefill_policy_cells_attested`` — the ONE strict
    reader (every worker must carry it and they must agree). Handed to the
    proxy, which mints a CELL-STABLE cache salt for an ``always`` cell's
    steered legs so their prefills resume, and the request-unique salt for
    every other cell. A missing or unreadable file, or a tree that predates the
    landing, is ``(None, "<why>")`` — recorded in endpoint.json, never guessed,
    and the proxy then shares nothing."""
    from sisteer.capeval.serving import lens_worker as lw
    reader = getattr(lw, "resumed_prefill_policy_cells_attested", None)
    if reader is None:
        return None, ("lens_worker carries no resumed_prefill_policy_cells_attested "
                      "— a tree that predates the resumed-prefill landing")
    att, error = _load_attestation(path)
    if error is not None:
        return None, error
    return reader(att)


def attested_serve_declarations_at(path):
    """``(mapping, error)`` — the per-cell SERVE DECLARATIONS the launched
    workers attest (``lens_serve_declarations`` on every top-level install
    ``worker_reports`` entry of the steering attestation at ``path``: harness
    cell id (variant spelling) -> ``{"serve_out_of_support": ["thinking"]}`` |
    ``{"serve_experiment": <name>}``, declaration-carrying cells only, the
    ``none__lens0`` twin under its own id when its donor declares), through
    ``lens_worker.serve_declarations_attested`` — the ONE strict reader. An
    attested EMPTY map is ``({}, None)`` (a standup with no declared cell). A
    missing or unreadable file, or a tree that predates the landing, is
    ``(None, "<why>")`` — recorded in endpoint.json, never guessed, and the
    proxy then refuses every thinking-ON declaration (fail closed)."""
    from sisteer.capeval.serving import lens_worker as lw
    reader = getattr(lw, "serve_declarations_attested", None)
    if reader is None:
        return None, ("lens_worker carries no serve_declarations_attested — a tree "
                      "that predates the fit-support-transfer landing")
    att, error = _load_attestation(path)
    if error is not None:
        return None, error
    return reader(att)


def attested_reduction_mode_at(path):
    """``(mode, error)`` — the REDUCTION MODE the launched workers attest
    (``lens_reduction_mode`` in every ``scheduler_geometry.worker_reports``
    entry of the steering attestation at ``path``: ``per_row`` | ``stacked``),
    through ``lens_worker.attested_reduction_mode`` — the ONE strict reader
    (every worker must carry it and they must agree on a known mode). A
    missing or unreadable file is ``(None, "<why>")`` like any other failure:
    recorded in endpoint.json, never guessed."""
    from sisteer.capeval.serving import lens_worker as lw
    att, error = _load_attestation(path)
    if error is not None:
        return None, error
    return lw.attested_reduction_mode(att)


# --------------------------------------------------------------------------- #
# the declaration and the two headers                                         #
# --------------------------------------------------------------------------- #


def _lens_resolved_spec_items(cells: Any, source: Any):
    """``(cell_id, resolved_spec)`` for every CANONICAL cell the standup serves
    on the LENS path (``serving_identity.path == "lens"``) whose cell-source
    entry carries a resolved spec, in canonical order."""
    from sisteer.capeval import manifest as cm
    for c in cells.canonical:
        if cm.parse_serving_identity(c.serving_identity).path != "lens":
            continue
        entry = source.get(c.cell_id)
        rspec = getattr(entry, "resolved_spec", None) if entry is not None else None
        if isinstance(rspec, Mapping):
            yield c.cell_id, rspec


def lens_resolved_specs(cells: Any, source: Any) -> list[Mapping[str, Any]]:
    """The ``resolved_spec`` of every CANONICAL cell the standup serves on
    the LENS path (``serving_identity.path == "lens"``), read from the cell
    source — the specs a serve declaration (``@thinking``) or an
    ``all_turns`` position can ride. Base and prompted cells carry none."""
    return [spec for _, spec in _lens_resolved_spec_items(cells, source)]


def lens_resolved_specs_by_cell(cells: Any, source: Any) -> dict[str, Mapping[str, Any]]:
    """:func:`lens_resolved_specs` keyed by the canonical cell id (the harness
    spelling, variant included) — for refusals that must NAME the cell."""
    return dict(_lens_resolved_spec_items(cells, source))


def vacuous_declarations(specs_by_cell: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """The cells whose thinking-ON declaration is VACUOUS — declared
    (``serve_out_of_support: ["thinking"]``) while NO served op is
    header-located: the served position mode read through
    ``defense_cells.served_positions_of`` (a stacked cell's is its object leg's)
    is outside ``proxy.HEADER_LOCATED_MODES``, or is absent. This is the lens's
    rule 3 (``lens_adapters.attach_thinking_header``: "the declaration is
    VACUOUS on an all / last cell, whose mask does not move with the render"),
    applied at STANDUP time so a standup refuses by name before a GPU minute
    is spent rather than at the worker's install — the same predicate the proxy
    decides header-locatedness from (served positions, stacked unwrapped;
    absent reads as NOT header-located here because a declaration on a mask the
    exporter cannot even locate certifies nothing). An undeclared cell is
    never listed, whatever its positions."""
    from sisteer.capeval.petri.defense_cells import served_positions_of
    from sisteer.capeval.serving import serve_declaration as sd
    from sisteer.capeval.serving.proxy import HEADER_LOCATED_MODES
    out: list[str] = []
    for cell_id, spec in specs_by_cell.items():
        declared = sd.OUT_OF_SUPPORT_THINKING in sd.declarations_of(spec).get(sd.SERVE_OUT_OF_SUPPORT_KEY, [])
        if not declared:
            continue
        mode = served_positions_of(spec)
        if mode not in HEADER_LOCATED_MODES:
            out.append(cell_id)
    return out


def declares_thinking(lens_specs: Sequence[Mapping[str, Any]]) -> bool:
    """Whether ANY exported lens cell declares the thinking-ON render
    (``serve_out_of_support: ["thinking"]`` in its hashed resolved spec — the
    ``@thinking`` spelling, ``serve_declaration.apply_variant``)."""
    from sisteer.capeval.serving import serve_declaration as sd
    return any(
        sd.OUT_OF_SUPPORT_THINKING
        in sd.declarations_of(s).get(sd.SERVE_OUT_OF_SUPPORT_KEY, []) for s in lens_specs)


def thinking_on_header_ids(tok: Any, header_ids: tuple[int, ...], *,
                           declares_thinking: bool) -> tuple[int, ...] | None:
    """The SECOND header the worker registry needs when a cell declares the
    thinking-ON render — ``generation_header_ids_under(tok,
    enable_thinking=True)`` — or ``None`` when no cell declares (rule 5: a
    second header is never handed to an UNDECLARED cell; a standup with no
    declaring cell exports the OFF header alone, so a template whose ON header
    cannot be derived — Nemotron-3 — still stands up). On a template whose
    switch does NOT move the header (gemma-4) the derived ON header EQUALS the
    OFF header and is returned as the second header all the same: the
    declaration is admitted, the registry records the equal header, and the
    worker locates the same boundary under either render (the rule 2026-09-10;
    plan P9's "vacuous" refusal is withdrawn — a header that does not move is
    no reason not to serve or steer thinking-ON)."""
    if not declares_thinking:
        return None
    from sisteer.chat_markers import generation_header_ids_under
    return tuple(generation_header_ids_under(tok, enable_thinking=True))


def standup_thinking_header(tok, *, declares_thinking: bool):
    """``(thinking_header, thinking_header_error)`` for the routing table,
    endpoint.json and the proxy.

    A standup with a cell DECLARING the thinking-ON render (``@thinking``)
    derives the whole fact HARD (``chat_markers.thinking_header_facts``): the
    declaration cannot be served without a certified ON header, so a template
    that refuses it refuses the standup — as before. Every OTHER standup (an
    ordinary ``--cell none`` or unsuffixed thinking-OFF cell — the protected
    legacy path) derives it NON-FATALLY (``thinking_header_facts_or_error``):
    a template with a switch whose ON header cannot be derived (the pinned
    Nemotron-3: its ON generation prompt ends in an OPEN ``<think>`` where the
    completed turn carries ``<think></think>``, failing chat_markers' prefix
    rule) keeps ``switch`` and ``off_ids``, records ``on_ids: None`` and
    ``moves: None``, and the reason rides beside it. Thinking-OFF serving
    proceeds; the proxy refuses every declared thinking-ON request on a lens
    route ``thinking_on_unattested`` (fail closed) and accepts the base
    route's ON render as §2.5.5 says. Selector validation and chat_markers'
    own checks are untouched — nothing here derives a header differently,
    it only stops an unused derivation from killing the launch. The standup
    NEVER substitutes a measurement table's ids for a null ``on_ids``: an
    attestation is what the lens can locate, not what a table says."""
    from sisteer.chat_markers import (
        thinking_header_facts,
        thinking_header_facts_or_error,
    )
    if declares_thinking:
        return dict(thinking_header_facts(tok)), None
    return thinking_header_facts_or_error(tok)


# --------------------------------------------------------------------------- #
# the proxy's construction                                                    #
# --------------------------------------------------------------------------- #


def backend_timeout_kwargs(upstream_timeout_s: Any) -> dict[str, float]:
    """The ``HTTPBackend`` kwarg for the proxy's UPSTREAM (proxy -> vLLM)
    total timeout — ONLY when the standup was given one
    (``--upstream-timeout-s``); ``{}`` otherwise, so the proxy's own default
    stands and every caller that never passed the flag builds its backend
    byte for byte as before. A non-positive or non-numeric value refuses: a
    proxy that "times out" at 0 s would 504 every draw."""
    if upstream_timeout_s is None:
        return {}
    if isinstance(upstream_timeout_s, bool) or not isinstance(upstream_timeout_s, (int, float, str)):
        raise ValueError(f"--upstream-timeout-s must be a number of seconds, got {upstream_timeout_s!r}")
    value = float(upstream_timeout_s)
    if not value > 0:
        raise ValueError(f"--upstream-timeout-s must be > 0 seconds, got {value!r}")
    return {"timeout_s": value}


def served_positions_passthrough(table: Mapping[str, Any], routing_proxy_cls: Any
                                 ) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(kwargs, record)`` for the ``lens_served_positions`` stamp:
    ``kwargs`` is ``{"lens_served_positions": table.get(KEY)}`` when the
    ``RoutingProxy`` constructor on this tree accepts the kwarg (the proxy
    refinement has landed), else ``{}``; ``record`` is what endpoint.json
    carries either way — the stamp as read from the FRESHLY loaded table
    (``None`` when the table has none: a pre-refinement exporter) and
    :data:`LENS_SERVED_POSITIONS_PASSTHROUGH_KEY` = ``passed`` or the reason the
    proxy could not be handed it. The dry run (``--stage prod``) and the sbatch
    REQUIRE the key in the fresh table before any request (the stale-table
    hazard); this function never manufactures the map."""
    value = table.get(LENS_SERVED_POSITIONS_KEY) if isinstance(table, Mapping) else None
    try:
        params = inspect.signature(routing_proxy_cls.__init__).parameters
        accepts = LENS_SERVED_POSITIONS_KEY in params
    except (TypeError, ValueError):
        accepts = False
    record: dict[str, Any] = {LENS_SERVED_POSITIONS_KEY: value}
    if accepts:
        record[LENS_SERVED_POSITIONS_PASSTHROUGH_KEY] = PASSTHROUGH_PASSED
        return {LENS_SERVED_POSITIONS_KEY: value}, record
    record[LENS_SERVED_POSITIONS_PASSTHROUGH_KEY] = (
        "not passed: RoutingProxy.__init__ on this tree has no lens_served_positions kwarg "
        "(Evals' proxy refinement not merged) — an undeclared thinking-ON request on a lens "
        "route fails closed as thinking_on_undeclared")
    return {}, record


def routing_proxy_attestation_kwargs(*, attestation_path: Path, table: Mapping[str, Any],
                                     thinking_header_error: str | None,
                                     lens_salt_policy: str,
                                     expected_digests: Mapping[str, str],
                                     routing_proxy_cls: Any
                                     ) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(kwargs, facts)`` — the attestation-derived ``RoutingProxy`` kwargs
    both standups hand over after ``launch_server`` (capctrl ``bring_up``'s
    order, verbatim): the per-cell resumed-prefill policy map, the serve
    declarations map (EVEN WHEN EMPTY: an attested ``{}`` is the stated "no
    declared cell", so an in-support lens ON request passes the
    ``thinking_on_unattested`` check that fires only on ``None``) with its
    error only when the map is absent, the routing table's ``thinking_header``
    with its error, the salt policy, the per-cell artifact digests the registry
    export was held to, and the served-positions stamp when the constructor
    takes it. ``facts`` is the SAME information for endpoint.json / the
    standup's own record (``check_serve_attestation``'s keys)."""
    rp_cells, rp_error = attested_resumed_prefill_policy_cells_at(attestation_path)
    decls, decls_error = attested_serve_declarations_at(attestation_path)
    positions_kwargs, positions_record = served_positions_passthrough(table, routing_proxy_cls)
    kwargs: dict[str, Any] = dict(
        resumed_prefill_policy_cells=rp_cells,
        lens_serve_declarations=decls,
        lens_serve_declarations_error=(decls_error if decls is None else None),
        thinking_header=table.get("thinking_header"),
        thinking_header_error=thinking_header_error,
        lens_salt_policy=lens_salt_policy,
        object_artifact_sha256_by_cell=dict(expected_digests),
        **positions_kwargs)
    facts: dict[str, Any] = {
        "resumed_prefill_policy_cells": rp_cells,
        "resumed_prefill_policy_error": rp_error,
        "lens_serve_declarations_error": (decls_error if decls is None else None),
        "thinking_header": table.get("thinking_header"),
        "thinking_header_error": thinking_header_error,
        "lens_salt_policy": lens_salt_policy,
        "object_artifact_sha256_by_cell": dict(expected_digests),
        **positions_record,
    }
    if decls is not None:
        facts["lens_serve_declarations"] = decls
    return kwargs, facts
