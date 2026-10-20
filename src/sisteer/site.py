"""The site configuration, and the placeholder grammar for path values in committed configs (stdlib only).

Committed configs never name a host, a user or a storage root. A value that points at site storage carries one
placeholder from a closed set, and the code that reads the value expands it just before it opens or writes the path:

    placeholder      stands for                                 resolved from (first set wins)
    $SHARED          the cluster's shared storage root           env SISTEER_SHARED_ROOT, env CLUSTER_SHARED_ROOT,
                                                                 else the cluster block's CLUSTER_SHARED_ROOT
    $HF_HOME         the Hugging Face cache                      env HF_HOME, env SISTEER_HF_HOME,
                                                                 else the cluster block's CLUSTER_HF_HOME
    $STATE           the per-user state root                     SISTEER_STATE_ROOT
    $RUNSTATE        the per-user run-state tree of the lanes    SISTEER_RUN_STATE_ROOT
    $STAGE           the grading host's staging directory        SISTEER_STAGE_DIR
    $WORK            the grading host's working-data root        SISTEER_WORK_DIR
    $GRADE           the grading host's grade root               SISTEER_GRADE_DIR
    $WANDB_PROJECT   the W&B project                             env WANDB_PROJECT, SISTEER_WANDB_PROJECT

``$NAME`` and ``${NAME}`` are the same placeholder. A key named for the site config (SISTEER_STATE_ROOT and the rest) is
read from the environment first and then from the site config file. ``$STATE`` has its own key, SISTEER_STATE_ROOT,
because SISTEER_STATE_DIR already names the front door's own state directory (``<root>/sisteer-reproduce`` by default),
which is a different directory.

The cluster block for ``$SHARED`` and ``$HF_HOME`` is the one the caller names, else the job's SISTEER_CLUSTER_LABEL,
else the site's SISTEER_PRIMARY_CLUSTER, else the site's only cluster. A caller that names a cluster gets that cluster's
block even when the environment carries another root, so a grading-host process can compose a path on either cluster.

A config's identity is its placeholder form: readers hash file bytes, never expanded values.

:func:`expand` substitutes text and nothing else: a site value written ``~/.local/state`` stays ``~/.local/state/...``
after it replaces ``$STATE``, so an expanded value is exactly the literal the placeholder replaced. A reader that opens
the path uses :func:`expand_path`, which expands ``~`` as well.

Every failure raises :class:`PlaceholderError` naming the placeholder and the variable to set: a placeholder that
resolves to nothing, a ``$NAME`` outside the closed set, a bare or unterminated ``$``, and a value that still holds a
placeholder after expansion. Nothing falls back to a guessed path, so no run reads from or writes to a literal
``$SHARED`` directory.

The site config is ``$SISTEER_CONFIG``, else ``$SISTEER_SITE_DIR/config.env`` (default directory ``~/.config/sisteer``),
plus one ``clusters/<label>.env`` block per label its SISTEER_CLUSTERS names. SISTEER_CLUSTERS is the default set (the
front door's fan-out, :attr:`Site.labels`); a caller that names a cluster outside it still gets that cluster's block, read
from the ``clusters/<label>.env`` file beside the config, as ``reproduce/lib/clusters.py``'s lookups by label do.
``reproduce/lib/clusters.py`` reads the same files with this module's parser and validates the blocks in full; this module
reads only what expansion needs.

    python -m sisteer.site expand [--cluster LABEL] VALUE...   # print each value expanded, or refuse by name
    python -m sisteer.site show [--cluster LABEL]              # every placeholder and what it resolves to here
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

#: Each placeholder: (environment variables, then site-config keys, then the cluster-block key), each tried in order.
#: The environment always wins over the site file; the cluster block is the last source.
PLACEHOLDERS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...], Optional[str]]] = {
    "SHARED": (("SISTEER_SHARED_ROOT", "CLUSTER_SHARED_ROOT"), (), "CLUSTER_SHARED_ROOT"),
    "HF_HOME": (("HF_HOME", "SISTEER_HF_HOME"), (), "CLUSTER_HF_HOME"),
    "STATE": (("SISTEER_STATE_ROOT",), ("SISTEER_STATE_ROOT",), None),
    "RUNSTATE": (("SISTEER_RUN_STATE_ROOT",), ("SISTEER_RUN_STATE_ROOT",), None),
    "STAGE": (("SISTEER_STAGE_DIR",), ("SISTEER_STAGE_DIR",), None),
    "WORK": (("SISTEER_WORK_DIR",), ("SISTEER_WORK_DIR",), None),
    "GRADE": (("SISTEER_GRADE_DIR",), ("SISTEER_GRADE_DIR",), None),
    "WANDB_PROJECT": (("WANDB_PROJECT", "SISTEER_WANDB_PROJECT"), ("SISTEER_WANDB_PROJECT",), None),
}
NAMES: Tuple[str, ...] = tuple(PLACEHOLDERS)

DEFAULT_SITE_DIR = "~/.config/sisteer"

# `${anything}`, `$NAME`, or a bare `$` (the empty alternative): every `$` in a value is one of the three
_TOKEN = re.compile(r"\$(?:\{([^}]*)\}|([A-Za-z_][A-Za-z0-9_]*)|)")
_ASSIGN = re.compile(r"""^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)=(.*?)\s*$""")
_MAX_DEPTH = 4   # a site value may itself hold a placeholder (`'$SHARED/hf_cache'`); deeper than this is a cycle
#: a site label, the name of a cluster block file (reproduce/lib/clusters.py LABEL_RE); a name outside it is never a file name
_LABEL_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class SiteConfigError(ValueError):
    """The site config cannot be read: a named file is missing, or a line does not parse."""


class PlaceholderError(ValueError):
    """A value holds a placeholder that cannot be expanded here; the message names it and the variable to set."""


# --- the site config files ---------------------------------------------------------------------------------------------

def parse_env_file(path: Union[str, os.PathLike]) -> Dict[str, str]:
    """``KEY=value`` lines of a site file. Values are bash-quoted words joined by one space; ``$SHARED`` and every other
    placeholder stay literal (single quotes in the file keep bash from expanding them when the front door sources it)."""
    out: Dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            m = _ASSIGN.match(ln)
            if not m:
                continue
            key, raw = m.group(1), m.group(2)
            try:
                parts = shlex.split(raw, comments=True)
            except ValueError as exc:
                raise SiteConfigError(f"{path}: cannot parse {key}: {exc}") from exc
            out[key] = " ".join(parts) if parts else ""
    return out


def substitute_shared(value: str, root: str) -> str:
    """``$SHARED`` and ``${SHARED}`` in a cluster-block value, replaced by that cluster's own root."""
    return re.sub(r"\$\{SHARED\}|\$SHARED(?![A-Za-z0-9_])", lambda _m: root, value)


def _read_block(path: str) -> Dict[str, str]:
    """One cluster block file, ``$SHARED`` in its values replaced by its own CLUSTER_SHARED_ROOT."""
    block = parse_env_file(path)
    root = block.get("CLUSTER_SHARED_ROOT", "")
    if root:
        block = {k: substitute_shared(v, root) for k, v in block.items()}
    return block


_BLOCK_CACHE: Dict[Tuple[str, int, int], Dict[str, str]] = {}


def site_dir(env: Optional[Mapping[str, str]] = None) -> str:
    """The site directory: the directory of ``$SISTEER_CONFIG`` when set, else ``$SISTEER_SITE_DIR``, else
    ``~/.config/sisteer``."""
    env = os.environ if env is None else env
    explicit = env.get("SISTEER_CONFIG")
    if explicit:
        return os.path.dirname(os.path.abspath(os.path.expanduser(explicit)))
    return os.path.expanduser(env.get("SISTEER_SITE_DIR") or DEFAULT_SITE_DIR)


def config_path(env: Optional[Mapping[str, str]] = None) -> str:
    """The site config file a process with no explicit file reads: ``$SISTEER_CONFIG``, else
    ``<site dir>/config.env``."""
    env = os.environ if env is None else env
    explicit = env.get("SISTEER_CONFIG")
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    return os.path.join(site_dir(env), "config.env")


@dataclass(frozen=True)
class Site:
    """One host's site config: the scalars of the config file and each named cluster's block (``$SHARED`` in a block
    already replaced by that block's root). ``path`` is None when no site config exists on this host."""
    path: Optional[str] = None
    config: Mapping[str, str] = field(default_factory=dict)
    clusters: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    @property
    def labels(self) -> List[str]:
        return list(self.clusters)

    @property
    def primary(self) -> Optional[str]:
        p = self.config.get("SISTEER_PRIMARY_CLUSTER") or ""
        if p:
            return p
        return self.labels[0] if len(self.labels) == 1 else None

    def block(self, label: str) -> Optional[Mapping[str, str]]:
        """The block of the cluster whose site label is ``label``: the one SISTEER_CLUSTERS names, else the file
        ``clusters/<label>.env`` beside the site config when it exists (a cluster outside the default set, reached only
        by naming it), else None. Read on first use, so an unlisted block no caller names is never opened."""
        got = self.clusters.get(label)
        if got is not None or not self.path or not _LABEL_RE.match(label):
            return got
        bp = os.path.join(os.path.dirname(self.path), "clusters", f"{label}.env")
        try:
            st = os.stat(bp)
        except OSError:
            return None
        key = (bp, st.st_mtime_ns, st.st_size)
        if key not in _BLOCK_CACHE:
            _BLOCK_CACHE[key] = _read_block(bp)
        return _BLOCK_CACHE[key]


_SITE_CACHE: Dict[Tuple[str, int, int], Site] = {}


def load_site(path: Optional[Union[str, os.PathLike]] = None, *, env: Optional[Mapping[str, str]] = None) -> Site:
    """Read the site config: ``path``, else ``$SISTEER_CONFIG``, else ``<site dir>/config.env``, plus the
    ``clusters/<label>.env`` block of every label its SISTEER_CLUSTERS names (the blocks live beside the file).

    A file that was asked for by name (``path`` or ``$SISTEER_CONFIG``) and is missing raises
    :class:`SiteConfigError`, and so does a named cluster with no block. When nothing was asked for and the default file
    is absent (a cluster node, a fresh checkout), the result is an empty :class:`Site`: expansion then reads the
    environment only and refuses by name whatever it cannot resolve."""
    env = os.environ if env is None else env
    explicit = path is not None or bool(env.get("SISTEER_CONFIG"))
    p = os.path.abspath(os.path.expanduser(os.fspath(path))) if path is not None else config_path(env)
    try:
        st = os.stat(p)
    except OSError:
        if explicit:
            raise SiteConfigError(f"no site config at {p}")
        return Site()
    key = (p, st.st_mtime_ns, st.st_size)
    if key in _SITE_CACHE:
        return _SITE_CACHE[key]
    config = parse_env_file(p)
    clusters: Dict[str, Dict[str, str]] = {}
    base = os.path.dirname(p)
    for label in (config.get("SISTEER_CLUSTERS") or "").split():
        bp = os.path.join(base, "clusters", f"{label}.env")
        if not os.path.isfile(bp):
            raise SiteConfigError(f"cluster {label!r} is named in SISTEER_CLUSTERS of {p} but {bp} does not exist")
        clusters[label] = _read_block(bp)
    site = Site(path=p, config=config, clusters=clusters)
    _SITE_CACHE[key] = site
    return site


def setting(key: str, default: Optional[str] = None, *, site: Optional[Site] = None,
            env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """One site setting: the environment's ``key``, else the site config's, else ``default``. An empty value counts as
    unset, and a host with no site config file is no error (only a file named by ``$SISTEER_CONFIG`` and missing is).

    The value is returned as written: a caller that opens it as a path expands it with :func:`expand_path`."""
    env = os.environ if env is None else env
    if env.get(key):
        return env[key]
    site = load_site(env=env) if site is None else site
    got = site.config.get(key)
    return got if got else default


def state_path(key: str, *parts: str, site: Optional[Site] = None, env: Optional[Mapping[str, str]] = None) -> Path:
    """A per-user state path a site may move: the setting ``key`` (the environment, then the site config) expanded with
    :func:`expand_path`, else ``$STATE/<parts>`` when SISTEER_STATE_ROOT is set, else ``~/.local/state/<parts>``.

    A ``key`` or SISTEER_STATE_ROOT that is set but does not expand refuses (:class:`PlaceholderError`); only a host that
    sets neither gets the default under ``~/.local/state``."""
    env = os.environ if env is None else env
    site = load_site(env=env) if site is None else site
    got = setting(key, site=site, env=env)
    if got:
        return expand_path(got, site=site, env=env, key=key)
    rel = "/".join(parts)
    if setting("SISTEER_STATE_ROOT", site=site, env=env):
        return expand_path("$STATE/" + rel, site=site, env=env, key=key)
    return Path(os.path.expanduser("~/.local/state/" + rel))


# --- expansion ---------------------------------------------------------------------------------------------------------

def placeholders_in(value: str) -> List[str]:
    """Every ``$`` token of ``value`` as written: a name for ``$NAME`` / ``${NAME}``, the braces' raw text for an
    invalid ``${...}``, and ``""`` for a bare or unterminated ``$``."""
    out = []
    for m in _TOKEN.finditer(value):
        out.append(m.group(1) if m.group(1) is not None else (m.group(2) or ""))
    return out


def has_placeholder(value: object) -> bool:
    """True when ``value`` is a string that holds any ``$`` token (a site placeholder or not)."""
    return isinstance(value, (str, os.PathLike)) and "$" in os.fspath(value)


def _where(key: Optional[str]) -> str:
    return f" ({key})" if key else ""


def _sources_text(name: str) -> str:
    envs, keys, ckey = PLACEHOLDERS[name]
    parts = []
    env_only = [v for v in envs if v not in keys]
    if env_only:
        parts.append(" or ".join(env_only) + " in the environment")
    if keys:
        parts.append(" or ".join(keys) + " in the environment or the site config")
    if ckey:
        parts.append(f"{ckey} in the cluster's block")
    return ", or ".join(parts)


def _unknown(name: str, value: str, key: Optional[str], braced: bool = False) -> PlaceholderError:
    spelled = "${" + name + "}" if braced else "$" + name
    return PlaceholderError(f"{value or spelled!r}{_where(key)} holds {spelled}, which is not a site placeholder "
                            f"(known: {', '.join('$' + n for n in NAMES)})")


def resolve(name: str, cluster: Optional[str] = None, *, site: Optional[Site] = None,
            env: Optional[Mapping[str, str]] = None, value: str = "", key: Optional[str] = None) -> str:
    """The value one placeholder stands for on this host, or :class:`PlaceholderError` naming what to set.

    For ``$SHARED`` and ``$HF_HOME`` with a named ``cluster``: that cluster's block when the site config has it (listed in
    SISTEER_CLUSTERS or a ``clusters/<label>.env`` file beside it, :meth:`Site.block`); a site config that describes other
    clusters but not this one refuses; with no block (a cluster node carries no site config) the environment answers only
    when the job's SISTEER_CLUSTER_LABEL is unset or names the same cluster."""
    if name not in PLACEHOLDERS:
        raise _unknown(name, value, key)
    env = os.environ if env is None else env
    site = load_site(env=env) if site is None else site
    envs, keys, ckey = PLACEHOLDERS[name]
    shown = value or "$" + name
    if ckey and cluster:
        block = site.block(cluster)
        if block is not None:
            if block.get(ckey):
                return block[ckey]
        elif site.clusters:
            raise PlaceholderError(f"${name} in {shown!r}{_where(key)} is asked for cluster {cluster!r}, which the site "
                                   f"config {site.path} does not name (SISTEER_CLUSTERS: {' '.join(site.labels)}; no "
                                   f"clusters/{cluster}.env beside it)")
        job_label = env.get("SISTEER_CLUSTER_LABEL")
        if job_label and job_label != cluster:
            raise PlaceholderError(f"${name} in {shown!r}{_where(key)} is asked for cluster {cluster!r}, but this job runs "
                                   f"on {job_label!r} and no site config here names {cluster!r}")
    for var in envs:
        if env.get(var):
            return env[var]
    for k in keys:
        if site.config.get(k):
            return site.config[k]
    if ckey and not cluster:
        label = env.get("SISTEER_CLUSTER_LABEL") or site.primary
        block = site.block(label) if label else None
        if block and block.get(ckey):
            return block[ckey]
    raise PlaceholderError(f"${name} in {shown!r}{_where(key)} is not set on this host: set {_sources_text(name)} "
                           f"(site config: {site.path or 'none on this host'})")


def _expand(value: str, cluster: Optional[str], site: Site, env: Mapping[str, str], key: Optional[str],
            depth: int, chain: Tuple[str, ...]) -> str:
    def one(m: "re.Match[str]") -> str:
        braced, bare = m.group(1), m.group(2)
        if braced is None and bare is None:
            raise PlaceholderError(f"{value!r}{_where(key)} holds a bare or unterminated '$' at offset {m.start()}; "
                                   f"a placeholder is one of {', '.join('$' + n for n in NAMES)}")
        name = braced if braced is not None else bare
        if name not in PLACEHOLDERS:
            raise _unknown(name, value, key, braced is not None)
        if name in chain or depth >= _MAX_DEPTH:
            raise PlaceholderError(f"${name} in {value!r}{_where(key)} resolves through itself "
                                   f"({' -> '.join('$' + c for c in chain + (name,))}); fix the site config")
        got = resolve(name, cluster, site=site, env=env, value=value, key=key)
        if "$" in got:
            got = _expand(got, cluster, site, env, key, depth + 1, chain + (name,))
        return got

    return _TOKEN.sub(one, value)


def expand(value, cluster: Optional[str] = None, *, site: Optional[Site] = None,
           env: Optional[Mapping[str, str]] = None, key: Optional[str] = None):
    """``value`` with every placeholder substituted by its text on this host (``~`` is left as written; see
    :func:`expand_path`).

    None passes through, so ``expand(cfg.get(k))`` keeps an absent key absent; a value with no ``$`` is returned as it
    is. Anything else that is not a string or a path refuses. ``key`` names the config key in the error message."""
    if value is None:
        return None
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str):
        raise TypeError(f"expand(){_where(key)}: a path value must be a string, got {type(value).__name__}")
    if "$" not in value:
        return value
    env = os.environ if env is None else env
    site = load_site(env=env) if site is None else site
    return _expand(value, cluster, site, env, key, 0, ())


def expand_path(value, cluster: Optional[str] = None, *, site: Optional[Site] = None,
                env: Optional[Mapping[str, str]] = None, key: Optional[str] = None) -> Optional[Path]:
    """:func:`expand`, then ``~`` expansion of the whole value, as a :class:`Path` (None passes through)."""
    got = expand(value, cluster, site=site, env=env, key=key)
    return None if got is None else Path(os.path.expanduser(got))


def require_expanded(value, *, key: Optional[str] = None):
    """Refuse a path that still holds a ``$`` token, the guard before a write. Returns ``value`` unchanged."""
    if value is not None and has_placeholder(value):
        names = ", ".join("$" + (n or "(bare)") for n in placeholders_in(os.fspath(value)))
        raise PlaceholderError(f"{os.fspath(value)!r}{_where(key)} still holds {names}: expand it with sisteer.site.expand "
                               "before opening it, never use it as a literal path")
    return value


def expand_keys(mapping: Mapping, keys: Sequence[str], cluster: Optional[str] = None, *,
                site: Optional[Site] = None, env: Optional[Mapping[str, str]] = None, prefix: str = "",
                paths: bool = False) -> dict:
    """A copy of ``mapping`` with the string values of ``keys`` expanded (absent and None keys are left alone). With
    ``paths``, each expanded value also has ``~`` expanded (still a string), for keys a reader opens or writes."""
    out = dict(mapping)
    for k in keys:
        if out.get(k) is not None:
            got = expand(out[k], cluster, site=site, env=env, key=f"{prefix}{k}")
            out[k] = os.path.expanduser(got) if paths else got
    return out


# --- command line ------------------------------------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sisteer.site", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("expand", help="print each value with its placeholders expanded")
    e.add_argument("--cluster", default=None, help="the cluster whose block resolves $SHARED and $HF_HOME (its site label; any "
                   "block under the site's clusters/, listed in SISTEER_CLUSTERS or not)")
    e.add_argument("--config", default=None, help="the site config file (default: $SISTEER_CONFIG, else <site dir>/config.env)")
    e.add_argument("values", nargs="+")
    s = sub.add_parser("show", help="every placeholder and what it resolves to on this host")
    s.add_argument("--cluster", default=None)
    s.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    try:
        site = load_site(args.config) if args.config else load_site()
        if args.cmd == "expand":
            for v in args.values:
                print(expand(v, args.cluster, site=site))
            return 0
        for n in NAMES:
            try:
                print(f"${n}\t{resolve(n, args.cluster, site=site)}")
            except PlaceholderError as exc:
                print(f"${n}\tUNSET: {exc}")
        return 0
    except (PlaceholderError, SiteConfigError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
