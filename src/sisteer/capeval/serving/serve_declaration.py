"""Hashed, cell-level SERVE DECLARATIONS and the ``@<variant>`` cell-id
grammar.

Every fitted steering object in the campaign was fit under thinking-OFF,
one-turn renders, and is served under its fit's own position regime. The
two transfer experiments serve EXISTING objects, unchanged, outside that
support: experiment 1 under a thinking-ON render (a different assistant
header), experiment 2 in a multi-turn conversation under two position modes
the fit never saw (every assistant turn; every position including the
attacker's user turns — the fit/serve guard's known collapse corner). Each
such serving is a TEST, never a supported mode, so it must be declared
where three consumers can see it and none can miss it:

- in the cell's **resolved spec**, as one of two closed-vocabulary keys —
  :data:`SERVE_OUT_OF_SUPPORT_KEY` (a render the object was not fit under;
  today exactly ``["thinking"]``) or :data:`SERVE_EXPERIMENT_KEY` (a NAMED
  experiment from :data:`SERVE_EXPERIMENTS`). The intervention hash covers
  the resolved spec, so a declared cell is a DISTINCT served identity from
  its base; ``lens_adapters._served_positions`` reads the served mode from
  ``resolved_spec["positions"]`` ("the mode belongs to the CELL"), the
  fit/serve guard reads the experiment name to admit exactly the pair it
  describes, and the worker's install report attests the declarations per
  cell;
- in the cell's **NAME**, as ``<object cell>@<variant>`` with ``<variant>``
  from :data:`SERVE_VARIANTS` — so vmodels (``petri__si_safety__psr_kl_all
  @thinking``), transcript dirs and manifests cannot confuse the settings;
- in the **attestation** (``lens_serve_declarations``), keyed by
  the harness id in its variant spelling, one closed shape per cell.

Each variant is ONE overlay (:func:`apply_variant`) over the artifact-
derived resolved spec: it writes exactly one declaration key and, for the
position variants, REPLACES the artifact's ``positions`` default (the
campaign's PSR base specs already carry ``positions="response"``, so the
presence of ``positions`` is never a refusal trigger — only a declaration
key already present is: a second application, or a hand-edited base). The
overlay never touches the artifact bytes, their sha256, or the fit regime
the guard reads (``recorded_fit_positions`` reads BYTES, never the spec).

The vocabulary is CLOSED: an unknown variant, token or experiment name
refuses at every door (:func:`split_variant`, :func:`apply_variant`,
:func:`declarations_of`); widening it is a code change, never a runtime
value. This module is torch-free and stdlib-only on purpose — the proxy,
the torch-free ``petri.defense_cells`` grammar and the bare launch venvs
all import it. The position literals it carries are pinned against
``lens_adapters``' constants by ``tests/test_serve_declaration.py`` (which
CAN import torch), not by importing them here.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Tuple


class ServeDeclarationError(ValueError):
    """A serve-declaration refusal (an unknown variant/token/experiment, a
    malformed declaration value, a double overlay) — loud, never a default."""


# --- the closed vocabulary ----------------------------------------------------

#: resolved-spec key: a list of renders the object was NOT fit under and is
#: nevertheless served under (today exactly ``["thinking"]``).
SERVE_OUT_OF_SUPPORT_KEY = "serve_out_of_support"
#: resolved-spec key: the NAME of a deliberate fit/serve transfer experiment
#: (:data:`SERVE_EXPERIMENTS`) — the fit/serve guard's positive, named
#: carve-out.
SERVE_EXPERIMENT_KEY = "serve_experiment"
#: the POSITIVE markers every variant writes — exactly one per variant; their
#: presence on a spec is what :func:`apply_variant` refuses to overlay twice.
DECLARATION_KEYS = (SERVE_OUT_OF_SUPPORT_KEY, SERVE_EXPERIMENT_KEY)

#: the one out-of-support render token: served under a thinking-ON render
#: (the alternate assistant header) — experiment 1.
OUT_OF_SUPPORT_THINKING = "thinking"
OUT_OF_SUPPORT_TOKENS = (OUT_OF_SUPPORT_THINKING,)

#: experiment 2 setting (2): a response-fit object served on EVERY assistant
#: turn's span (``positions="response_all_turns"``).
EXPERIMENT_FIT_RESPONSE_SERVE_ALL_TURNS = "fit_response_serve_all_turns"
#: experiment 2 setting (3): a response-fit object served at ALL positions,
#: the attacker's user-turn tokens included (``positions="all"``) — the
#: guard's collapse corner, admitted ONLY under this name.
EXPERIMENT_FIT_RESPONSE_SERVE_ALL_POSITIONS = "fit_response_serve_all_positions"
SERVE_EXPERIMENTS = (EXPERIMENT_FIT_RESPONSE_SERVE_ALL_TURNS,
                     EXPERIMENT_FIT_RESPONSE_SERVE_ALL_POSITIONS)

#: the resolved-spec key ``lens_adapters._served_positions`` reads the SERVED
#: mode from — the field the position variants replace.
POSITIONS_KEY = "positions"

#: Each named experiment describes exactly ONE ``(recorded fit regime,
#: served positions)`` pair: the guard passes a named cell iff
#: the artifact's recorded fit AND the ops' served positions are this pair,
#: and refuses a name that does not describe the deployment. The literals
#: are ``lens_adapters.POSITIONS_RESPONSE`` / ``POSITIONS_RESPONSE_ALL_TURNS``
#: / ``POSITIONS_ALL`` — pinned by test, not imported (torch-free module).
EXPERIMENT_FIT_SERVE_PAIRS: Mapping[str, Tuple[str, str]] = {
    EXPERIMENT_FIT_RESPONSE_SERVE_ALL_TURNS: ("response", "response_all_turns"),
    EXPERIMENT_FIT_RESPONSE_SERVE_ALL_POSITIONS: ("response", "all"),
}

#: the cell-id suffix separator: ``<object cell>@<variant>``. Never occurred
#: in a cell id before this grammar, so no existing id changes meaning.
VARIANT_SEP = "@"
VARIANT_THINKING = "thinking"
VARIANT_ALL_TURNS = "all_turns"
VARIANT_ALL_POSITIONS = "all_positions"
#: variant name -> the overlay it writes onto the base resolved spec. Every
#: variant carries a DECLARATION (none is attested by its ``positions``
#: alone); the position variants also replace the served ``positions``.
SERVE_VARIANTS: Mapping[str, Mapping[str, Any]] = {
    # experiment 1: the same response-mode cell under the thinking-ON header
    VARIANT_THINKING: {
        SERVE_OUT_OF_SUPPORT_KEY: [OUT_OF_SUPPORT_THINKING],
    },
    # experiment 2 setting (2)
    VARIANT_ALL_TURNS: {
        POSITIONS_KEY: EXPERIMENT_FIT_SERVE_PAIRS[
            EXPERIMENT_FIT_RESPONSE_SERVE_ALL_TURNS][1],
        SERVE_EXPERIMENT_KEY: EXPERIMENT_FIT_RESPONSE_SERVE_ALL_TURNS,
    },
    # experiment 2 setting (3)
    VARIANT_ALL_POSITIONS: {
        POSITIONS_KEY: EXPERIMENT_FIT_SERVE_PAIRS[
            EXPERIMENT_FIT_RESPONSE_SERVE_ALL_POSITIONS][1],
        SERVE_EXPERIMENT_KEY: EXPERIMENT_FIT_RESPONSE_SERVE_ALL_POSITIONS,
    },
}


def _known_variant(variant: Any) -> str:
    if not isinstance(variant, str) or variant not in SERVE_VARIANTS:
        raise ServeDeclarationError(
            f"unknown serve variant {variant!r} — the vocabulary is closed: "
            f"{sorted(SERVE_VARIANTS)} (widening it is a code change in "
            "serve_declaration.SERVE_VARIANTS, never a runtime value)")
    return variant


def variant_overlay(variant: str) -> Dict[str, Any]:
    """A fresh copy of the overlay :data:`SERVE_VARIANTS` writes for
    ``variant`` (every key, ``positions`` included). Unknown -> refuse."""
    return copy.deepcopy(dict(SERVE_VARIANTS[_known_variant(variant)]))


def variant_declarations(variant: str) -> Dict[str, Any]:
    """The DECLARATION-key subset of ``variant``'s overlay — the exact
    value the cell is attested under in ``lens_serve_declarations``:
    ``{"serve_out_of_support": ["thinking"]}`` for ``thinking``,
    ``{"serve_experiment": <name>}`` for the two position variants."""
    overlay = variant_overlay(variant)
    return {k: overlay[k] for k in DECLARATION_KEYS if k in overlay}


# --- the id grammar -----------------------------------------------------------


def split_variant(cell_id: Any) -> Tuple[str, Optional[str]]:
    """``"<base>@<variant>"`` -> ``(base, variant)``; a variant-free id ->
    ``(cell_id, None)``. At most one separator; an empty base, an empty
    variant, or a variant outside :data:`SERVE_VARIANTS` refuses. Says
    nothing about whether ``base`` is a legal cell id or whether a variant
    is legal on it — that is the grammar's call
    (``defense_cells.parse_cell_id``: object cells only)."""
    if not isinstance(cell_id, str) or not cell_id:
        raise ServeDeclarationError(
            f"cell id must be a non-empty string, got {cell_id!r}")
    if VARIANT_SEP not in cell_id:
        return cell_id, None
    base, _sep, variant = cell_id.partition(VARIANT_SEP)
    if VARIANT_SEP in variant:
        raise ServeDeclarationError(
            f"cell id {cell_id!r} carries more than one {VARIANT_SEP!r} — a "
            "cell wears at most ONE serve variant ('<cell>@<variant>')")
    if not base:
        raise ServeDeclarationError(
            f"cell id {cell_id!r} has an empty base before {VARIANT_SEP!r}")
    if not variant:
        raise ServeDeclarationError(
            f"cell id {cell_id!r} has an empty serve variant after "
            f"{VARIANT_SEP!r} (known: {sorted(SERVE_VARIANTS)})")
    return base, _known_variant(variant)


def join_variant(base_id: str, variant: Optional[str]) -> str:
    """The inverse of :func:`split_variant`: ``base_id`` unchanged when
    ``variant`` is None, else ``"<base_id>@<variant>"``. Refuses a base that
    already carries the separator (it is not a base) and an unknown
    variant."""
    if not isinstance(base_id, str) or not base_id:
        raise ServeDeclarationError(
            f"base cell id must be a non-empty string, got {base_id!r}")
    if VARIANT_SEP in base_id:
        raise ServeDeclarationError(
            f"base cell id {base_id!r} already carries {VARIANT_SEP!r} — a "
            "cell wears at most ONE serve variant; split it first")
    if variant is None:
        return base_id
    return f"{base_id}{VARIANT_SEP}{_known_variant(variant)}"


# --- the overlay and its strict reader -----------------------------------------


def apply_variant(resolved_spec: Mapping[str, Any], variant: str
                  ) -> Dict[str, Any]:
    """The ONE overlay: a NEW dict = ``resolved_spec`` + ``variant``'s
    overlay (:data:`SERVE_VARIANTS`), the base object untouched.

    Refuses iff the base spec ALREADY carries a declaration key
    (:data:`DECLARATION_KEYS`) — a second application, or a base artifact
    someone hand-edited — whatever that key's value. ``positions`` is
    never a refusal trigger: the campaign's PSR base specs carry
    ``positions="response"`` (``registry_adapter._resolve_torch_blob``
    copies every non-tensor blob field), and the position variants
    REPLACE that artifact-derived default with theirs; that is the
    sanctioned seam (``lens_adapters._served_positions``: the mode
    belongs to the CELL, the spec overrides the artifact's default),
    while the recorded FIT regime the guard reads comes from the artifact
    BYTES and cannot be reached from here."""
    if not isinstance(resolved_spec, Mapping):
        raise ServeDeclarationError(
            f"apply_variant: resolved_spec must be a mapping, got "
            f"{type(resolved_spec).__name__}")
    overlay = variant_overlay(variant)
    already = [k for k in DECLARATION_KEYS if k in resolved_spec]
    if already:
        raise ServeDeclarationError(
            f"apply_variant({variant!r}): the spec already carries serve "
            f"declaration key(s) {already} "
            f"({ {k: resolved_spec[k] for k in already} }) — a variant is "
            "applied exactly once to an artifact-derived base spec; a "
            "second overlay (or a hand-edited base artifact) is refused, "
            "never merged")
    out: Dict[str, Any] = dict(resolved_spec)
    out.update(overlay)
    return out


def declarations_of(resolved_spec: Mapping[str, Any]) -> Dict[str, Any]:
    """``{}`` for an undeclared spec, else the validated declaration — exactly
    ONE of the two closed shapes, as a fresh dict:
    ``{"serve_out_of_support": [<tokens>]}`` (a non-empty, duplicate-free
    list of :data:`OUT_OF_SUPPORT_TOKENS`) or ``{"serve_experiment":
    <name>}`` (a :data:`SERVE_EXPERIMENTS` member). Everything else refuses:
    a non-list / empty / duplicated / unknown-token ``serve_out_of_support``,
    a non-string or unknown ``serve_experiment``, or BOTH keys at once (no
    variant combines them)."""
    if not isinstance(resolved_spec, Mapping):
        raise ServeDeclarationError(
            f"declarations_of: resolved_spec must be a mapping, got "
            f"{type(resolved_spec).__name__}")
    out: Dict[str, Any] = {}
    if SERVE_OUT_OF_SUPPORT_KEY in resolved_spec:
        value = resolved_spec[SERVE_OUT_OF_SUPPORT_KEY]
        if not isinstance(value, list) or not value:
            raise ServeDeclarationError(
                f"{SERVE_OUT_OF_SUPPORT_KEY} must be a NON-EMPTY list of "
                f"render tokens from {list(OUT_OF_SUPPORT_TOKENS)}, got "
                f"{value!r}")
        for tok in value:
            if not isinstance(tok, str) or tok not in OUT_OF_SUPPORT_TOKENS:
                raise ServeDeclarationError(
                    f"{SERVE_OUT_OF_SUPPORT_KEY} carries {tok!r}, outside the "
                    f"closed vocabulary {list(OUT_OF_SUPPORT_TOKENS)}")
        if len(set(value)) != len(value):
            raise ServeDeclarationError(
                f"{SERVE_OUT_OF_SUPPORT_KEY} carries a duplicate token: "
                f"{value!r}")
        out[SERVE_OUT_OF_SUPPORT_KEY] = list(value)
    if SERVE_EXPERIMENT_KEY in resolved_spec:
        value = resolved_spec[SERVE_EXPERIMENT_KEY]
        if not isinstance(value, str) or value not in SERVE_EXPERIMENTS:
            raise ServeDeclarationError(
                f"{SERVE_EXPERIMENT_KEY} must name one of "
                f"{list(SERVE_EXPERIMENTS)}, got {value!r} — the vocabulary "
                "is closed; a new experiment is a code change")
        out[SERVE_EXPERIMENT_KEY] = value
    if len(out) > 1:
        raise ServeDeclarationError(
            f"a cell declares exactly ONE of {list(DECLARATION_KEYS)}, never "
            f"both — got {out!r} (no serve variant combines a render "
            "declaration with a named experiment; plan §2.3)")
    return out


# --- the thinking declaration as a value (the sampled-rerun standup's spelling) ---------


def thinking_declaration() -> Dict[str, Any]:
    """The thinking-ON declaration as a VALUE — exactly what a cell wearing the
    ``thinking`` variant is attested under in ``lens_serve_declarations``
    (:func:`variant_declarations`): ``{"serve_out_of_support": ["thinking"]}``.
    A fresh dict on every call. The ``mt_serving_up --thinking-on`` standup
    (which declares every header-dependent lens cell of a wave in one stroke,
    rather than one ``--cell <id>@thinking`` spelling at a time) reads its
    expected attestation from here, so the two spellings can never drift."""
    return variant_declarations(VARIANT_THINKING)


def declare_thinking(resolved_spec: Mapping[str, Any]) -> Dict[str, Any]:
    """:func:`apply_variant` with the ``thinking`` variant: a NEW spec carrying
    the thinking-ON declaration, the base untouched. Refuses (the overlay's
    rule) a spec that already carries ANY declaration key — a second
    application, or a base that already names a ``serve_experiment``."""
    return apply_variant(resolved_spec, VARIANT_THINKING)


def declares_thinking(resolved_spec: Mapping[str, Any]) -> bool:
    """Whether ``resolved_spec``'s validated declaration names the thinking-ON
    render (:func:`declarations_of`; a malformed declaration refuses rather
    than reading as False)."""
    decl = declarations_of(resolved_spec)
    return OUT_OF_SUPPORT_THINKING in decl.get(SERVE_OUT_OF_SUPPORT_KEY, [])
