"""Site paths for scripts that must not import the ``sisteer`` package of this checkout.

Some scripts put other trees on ``sys.path`` (a campaign's checkout, staged analysis modules, a pinned tree whose modules a
script requires to be the only ``sisteer`` it imports), so importing the package here would change which tree's modules they
get, or make them refuse. This module loads this checkout's ``src/sisteer/site.py`` by file path instead (the module is
stdlib only and imports nothing from its package), so such a script resolves a site placeholder (``$STAGE``, the staging
directory; ``$STATE``, the state root) through the site config, refused by name when it is unset, without importing the
package. :func:`main_checkout` is the checkout a campaign's other worktrees sit under: the parent of git's common directory
for the tree this file is in.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SITE_FILE = _HERE.parent / "src" / "sisteer" / "site.py"
_NAME = "_scripts_sisteer_site"


def _site():
    mod = sys.modules.get(_NAME)
    if mod is None:
        spec = importlib.util.spec_from_file_location(_NAME, _SITE_FILE)
        if spec is None or spec.loader is None:
            raise SystemExit(f"site_paths: cannot load {_SITE_FILE}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_NAME] = mod   # a dataclass module must be importable by name while it executes
        spec.loader.exec_module(mod)
    return mod


class SitePathError(SystemExit):
    """A site placeholder that does not resolve on this host; the message names it and what to set. Uncaught, it ends the
    run with that message; a caller that has its own refusal catches it by name."""


def site_path(value, key: str) -> Path:
    """``value`` with its site placeholder expanded (and ``~``), as a path; a value without one comes back as a path
    unchanged, and an unset placeholder raises :class:`SitePathError`."""
    mod = _site()
    try:
        return mod.expand_path(str(value), key=key)
    except (mod.PlaceholderError, mod.SiteConfigError) as exc:
        raise SitePathError(f"{key}: {exc}") from exc


def main_checkout() -> Path:
    """The main checkout of the repository this file is in (the parent of git's common directory); ends the run by name
    when this file is not in a git checkout."""
    got = subprocess.run(["git", "-C", str(_HERE), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                         capture_output=True, text=True)
    if got.returncode != 0 or not got.stdout.strip():
        raise SystemExit(f"site_paths: {_HERE} is not in a git checkout, so the main checkout is unknown")
    return Path(got.stdout.strip()).parent
