"""Axis-agnostic engine-identity comparison — the refusing primitive.

Two observations produced on different engines — a different serving
backend, a different attacker engine, a different determinism mode — must
never be compared as if only the intervention differed: the engine is a
confound, and a delta computed across engines silently measures the
engine swap. This module is that rule as a CALLABLE INVARIANT. It exists
because an invariant that REFUSES at comparison time beats a sidecar
artifact someone has to remember to check: a recorded engine column is
only as good as the reader who joins on it, and the reader who forgets
gets a plausible-looking number.

The primitive is deliberately axis-agnostic and vocabulary-free: an
identity is an opaque string (or a flat mapping of named string
components), and the AXIS is whatever the caller is guarding — "target
serving", "attacker engine", …. It knows nothing of petri's cells,
tiers, control legs, or certificates.

The petri pairing rules (``sisteer.capeval.petri.defense_cells
.check_pairing``) are a CALLER of this primitive — they layer the §3
two-tier cell semantics (which component may differ, under which
certificate) on top of the comparison done here. Do not collapse the
two: this module is the generic "same engine or refuse" check; petri's
is one policy built on it.

Comparison semantics: EXACT string equality per component — no case
folding, no whitespace trimming, no version-aware parsing. An identity
is a label of record; two labels that differ in any byte were written by
different worlds until proven otherwise. Malformed inputs (empty
identities, non-string components, a string compared against a mapping)
REFUSE loudly rather than reporting a non-match: malformed identity
metadata can no more license a comparison than a mismatched one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple, Union

__all__ = [
    "OPAQUE",
    "EngineIdentityError",
    "IdentityMatch",
    "check_same_identity",
    "identity_diff",
    "render_identity",
]

#: an identity: an opaque string, or a flat {component: value} mapping.
Identity = Union[str, Mapping[str, str]]

#: the component name ``identity_diff`` reports when two OPAQUE (string)
#: identities differ — string identities have no finer structure to name.
OPAQUE = "identity"


class EngineIdentityError(ValueError):
    """A REFUSED comparison: the two observations' engine identities
    differ on the caller's axis (or the identity metadata itself is
    malformed — either way, nothing licenses the comparison).

    Structured fields for callers that map the refusal onto their own
    domain prose: ``axis``, ``identity_a`` / ``identity_b`` (rendered),
    and ``differing`` (the component names that differ; empty on a
    malformed-input refusal, where no comparison was performed)."""

    def __init__(self, message: str, *, axis: Optional[str] = None,
                 identity_a: Optional[str] = None,
                 identity_b: Optional[str] = None,
                 differing: Tuple[str, ...] = ()):
        super().__init__(message)
        self.axis = axis
        self.identity_a = identity_a
        self.identity_b = identity_b
        self.differing = tuple(differing)


@dataclass(frozen=True)
class IdentityMatch:
    """An ACCEPTED comparison: both observations carry the same engine
    identity on ``axis``. Identities are carried in rendered form."""

    axis: str
    identity_a: str
    identity_b: str


def _refuse(message: str, **kw: Any) -> None:
    raise EngineIdentityError(message, **kw)


def _validate_identity(identity: Any, label: str) -> None:
    if isinstance(identity, str):
        if not identity:
            _refuse(f"engine identity {label} is an empty string — an "
                    "unlabelled observation can never license a comparison")
        return
    if isinstance(identity, Mapping):
        if not identity:
            _refuse(f"engine identity {label} is an empty mapping — an "
                    "identity with no components identifies nothing")
        for k, v in identity.items():
            if not isinstance(k, str) or not k:
                _refuse(f"engine identity {label} has a non-string or empty "
                        f"component name: {k!r} — components are named by "
                        "non-empty strings")
            if not isinstance(v, str) or not v:
                _refuse(f"engine identity {label} component {k!r} has a "
                        f"non-string or empty value: {v!r} — render every "
                        "component to a non-empty string before comparing; "
                        "exact string equality is the whole contract")
        return
    _refuse(f"engine identity {label} must be a string or a flat mapping of "
            f"string components, got {type(identity).__name__}: {identity!r}")


def render_identity(identity: Identity) -> str:
    """A human-readable rendering of a (valid) identity, as used in
    refusal messages and :class:`IdentityMatch`."""
    _validate_identity(identity, "identity")
    if isinstance(identity, str):
        return identity
    return ", ".join(f"{k}={v!r}" for k, v in identity.items())


def identity_diff(identity_a: Identity,
                  identity_b: Identity) -> Tuple[str, ...]:
    """The comparison core: the names of the components that DIFFER,
    ``()`` when the identities are identical. Opaque strings compare as
    one component (:data:`OPAQUE`); mappings compare per component, in
    ``identity_a``'s component order (then ``identity_b``'s extras), and
    a component present on one side only counts as differing. Malformed
    or mixed-representation inputs refuse loudly — never a silent
    non-match, never a silent match."""
    _validate_identity(identity_a, "identity_a")
    _validate_identity(identity_b, "identity_b")
    a_is_str = isinstance(identity_a, str)
    if a_is_str != isinstance(identity_b, str):
        _refuse("engine identities use mixed representations (an opaque "
                f"string vs a component mapping): "
                f"{render_identity(identity_a)!r} vs "
                f"{render_identity(identity_b)!r} — comparing across "
                "representations would have to guess the component "
                "structure; refused")
    if a_is_str:
        return () if identity_a == identity_b else (OPAQUE,)
    keys = list(identity_a)
    keys += [k for k in identity_b if k not in identity_a]
    return tuple(k for k in keys
                 if identity_a.get(k) != identity_b.get(k))


def check_same_identity(identity_a: Identity, identity_b: Identity, *,
                        axis: str) -> IdentityMatch:
    """THE primitive: refuse unless both observations carry the SAME
    engine identity on ``axis``.

    Returns :class:`IdentityMatch`, or raises :class:`EngineIdentityError`
    whose message names the axis, both identities, and (for structured
    identities) exactly which components differed."""
    if not isinstance(axis, str) or not axis:
        _refuse("axis must be a non-empty string naming what is being "
                f"guarded (e.g. 'attacker engine'), got {axis!r}")
    differing = identity_diff(identity_a, identity_b)
    ra, rb = render_identity(identity_a), render_identity(identity_b)
    if differing:
        detail = ("" if differing == (OPAQUE,)
                  else f" (differing component(s): {', '.join(differing)})")
        _refuse(f"{axis} mismatch: {ra!r} vs {rb!r}{detail} — the two "
                "observations were produced on different engines; refusing "
                "to compare them (a delta across engines measures the "
                "engine swap, not the intervention)",
                axis=axis, identity_a=ra, identity_b=rb, differing=differing)
    return IdentityMatch(axis=axis, identity_a=ra, identity_b=rb)
