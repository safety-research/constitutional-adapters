"""The test suite runs without the HOST's site config (``tests/conftest.py`` applies :func:`isolate` at import, before
anything is collected).

The README installs a site config at ``~/.config/sisteer/config.env`` (``$SISTEER_SITE_DIR``, ``$SISTEER_CONFIG``;
``sisteer.site``, ``reproduce/lib/common.sh``, ``scripts/registry/landing/site_env.sh``), and every reader falls back to that
file when nothing names another. On a host that has one (the grading host, or any machine that followed the README), the
suite used to read it: a test that built its own cluster world inherited the host's clusters, mirror and repository names, and
the conftest's neutral ``$STATE`` / ``$STAGE`` / ``$WORK`` / ``$GRADE`` values stood down for every key the host file set, so a
test that moved ``HOME`` lost both (measured: 23 tests in 5 modules red with a site config installed, all green without
one). The Modal CPU suite and the Slurm runner have no host site, so the gate never saw it.

So, by default, a run sees NO host site config, exactly as the gate does:

- ``$SISTEER_CONFIG`` is removed and ``$SISTEER_SITE_DIR`` names an EMPTY directory the conftest makes for the run (read-only,
  so a test that writes into "the site" fails loudly instead of leaving a file for every later test to find), in-process and
  in every child a test spawns. A reader then finds no ``config.env``: an empty site, the state of a fresh clone.
- every suite default (``SUITE_SITE_DEFAULTS`` in the conftest) is exported unless the environment itself sets that key.

A test that builds its own site names it the way the code under test reads one: ``SISTEER_CONFIG=<its file>`` (or
``--config``), a ``SISTEER_SITE_DIR`` of its own, or, for a test of the ``~/.config/sisteer`` default under a moved
``HOME``, the ``home_site_default`` fixture, which removes both variables for that test alone.

``SISTEER_TEST_HOST_SITE=1`` opts a whole run INTO the host's site (the behaviour before this module): nothing is removed, and
a suite default stands down for every key the host's site config sets. Pinned by ``tests/test_suite_host_site_isolation.py``.
Stdlib only, and no side effect at import: the conftest imports it before anything of sisteer loads.
"""
from __future__ import annotations

import os
import re
import shlex
from typing import Callable, Dict, List, Mapping, MutableMapping, Optional, Set

#: ``=1`` lets a run read the host's site config (and keeps a suite default off every key that file sets)
OPT_IN_ENV = "SISTEER_TEST_HOST_SITE"
#: the two variables every site reader consults before ``~/.config/sisteer`` (sisteer.site.config_path; common.sh repro_config_path)
CONFIG_ENV = "SISTEER_CONFIG"
SITE_DIR_ENV = "SISTEER_SITE_DIR"
DEFAULT_SITE_DIR = "~/.config/sisteer"   # sisteer.site.DEFAULT_SITE_DIR, spelled out: nothing of sisteer is imported here

_ASSIGN = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)=(.*?)\s*$")


def host_site_keys(environ: Mapping[str, str]) -> Set[str]:
    """The keys the site config this environment resolves to sets to a non-empty value (the file ``sisteer.site.load_site``
    reads: ``$SISTEER_CONFIG``, else ``${SISTEER_SITE_DIR:-~/.config/sisteer}/config.env``); empty when there is none."""
    path = environ.get(CONFIG_ENV) or os.path.join(environ.get(SITE_DIR_ENV) or os.path.expanduser(DEFAULT_SITE_DIR), "config.env")
    keys: Set[str] = set()
    try:
        with open(os.path.expanduser(path), encoding="utf-8") as fh:
            for ln in fh:
                m = _ASSIGN.match(ln)
                if not m:
                    continue
                try:
                    words = shlex.split(m.group(2), comments=True)
                except ValueError:
                    continue
                if words:
                    keys.add(m.group(1))
    except OSError:
        pass
    return keys


def isolate(environ: MutableMapping[str, str], defaults: Mapping[str, str], make_site_dir: Callable[[], str]) -> Dict[str, object]:
    """Apply the suite's site isolation to ``environ`` IN PLACE and say what was done.

    Default: drop ``$SISTEER_CONFIG``, point ``$SISTEER_SITE_DIR`` at ``make_site_dir()`` (an empty directory), and export each
    of ``defaults`` the environment does not set (non-empty). With ``SISTEER_TEST_HOST_SITE=1``: nothing is dropped and a default
    is exported only where neither the environment nor the host's site config sets the key.

    Returns ``{"mode": "isolated"|"host", "site_dir": <dir or None>, "dropped": {var: value}, "defaulted": [keys]}``."""
    if environ.get(OPT_IN_ENV) == "1":
        host = host_site_keys(environ)
        defaulted: List[str] = [k for k in defaults if not environ.get(k) and k not in host]
        for k in defaulted:
            environ[k] = defaults[k]
        return {"mode": "host", "site_dir": None, "dropped": {}, "defaulted": defaulted}
    dropped: Dict[str, str] = {}
    for var in (CONFIG_ENV, SITE_DIR_ENV):
        if var in environ:
            dropped[var] = environ.pop(var)
    site_dir: Optional[str] = make_site_dir()
    environ[SITE_DIR_ENV] = site_dir
    defaulted = [k for k in defaults if not environ.get(k)]
    for k in defaulted:
        environ[k] = defaults[k]
    return {"mode": "isolated", "site_dir": site_dir, "dropped": dropped, "defaulted": defaulted}
