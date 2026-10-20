"""Site-placeholder forms of recorded values, for the readers that match them (stdlib only).

A committed value that names site storage is written in placeholder form (``$SHARED/...``, ``$STATE/...``; the grammar and the
closed set are :mod:`sisteer.site`'s). A reader that compares such a value with bytes recorded on a site compares it with its
expansion there as well: the expansion is exactly the literal the site's runs recorded, so a placeholder spelling and its
expansion name the same thing. Expansion is exact or absent, never a guess: a placeholder this host does not resolve has no
expansion here, and the placeholder form still matches itself. A site config file that is named but missing refuses, as it does in
:mod:`sisteer.site`.

:func:`expansion` gives that text; :func:`with_expansions` extends a spelling list with it; :func:`config_value` reads a key from
the environment, then from the site config file, the order :mod:`sisteer.site` documents for its own keys.
"""
from __future__ import annotations

import os
from typing import Iterable, List, Optional


def _site():
    """The site module, or None in a checkout that does not carry it (such a checkout has no site placeholders to expand)."""
    try:
        from .. import site
    except ImportError:
        return None
    return site


def expansion(value: object) -> Optional[str]:
    """The text ``value`` stands for on this host when it holds a site placeholder, else None (no placeholder, or one this host
    does not resolve). ``~`` is left as written."""
    if not isinstance(value, str) or "$" not in value:
        return None
    site = _site()
    if site is None:
        return None
    try:
        got = site.expand(value)
    except site.PlaceholderError:
        return None
    return got if got != value else None


def with_expansions(spellings: Iterable[object]) -> List[str]:
    """``spellings`` as strings, each followed by its expansion on this host when it has one (order kept, no duplicates)."""
    out: List[str] = []
    for s in spellings:
        for v in (str(s), expansion(str(s))):
            if v is not None and v not in out:
                out.append(v)
    return out


def config_value(key: str) -> Optional[str]:
    """``key`` from the environment, else from this host's site config file, else None."""
    if not key:
        return None
    if os.environ.get(key):
        return os.environ[key]
    site = _site()
    if site is None:
        return None
    return site.load_site().config.get(key) or None
