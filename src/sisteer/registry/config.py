"""The registry's operating configuration: ``<root>/config.toml``.

One file, no secrets — paths and names only. It names the paper checkout the rebuild trigger builds
from, the publish and status-page directories, the two clusters (login host + shared root), and the
snapshot cadence. Everything operational derives from it: the mirror destinations
(``<shared>/registry/log`` per cluster), the pull sources (``<shared>/registry/records``), the snapshot
destinations (``<shared>/registry/snapshot``), the remote ``PYTHONPATH`` the one stdlib verifier runs under
on a cluster (``<shared>/repos/<checkout name>/src``, the site's repo mirror, unless the cluster entry names ``remote_src``).

The ``[status]`` block carries the page directory and the render's three budgets (``section_budget_s`` /
``render_budget_s`` / ``heavy_budget_s``; :mod:`sisteer.registry.status`).

**The Modal destination was removed.** Until 2026-10 a ``modal`` table named
a volume the inbox pull, the snapshot push and (opt-in) the record-log mirror reached. The transport is gone, not
relocated, and the loader is FAIL-CLOSED about it: :func:`load` refuses a configuration that still carries that
table, and :func:`check_destinations` (run by ``harvest.load_config`` whenever a legacy ``config.json`` is read, and by
``registry config --check``) refuses a ``kind: modal`` entry anywhere in ``config.json``'s ``mirrors`` / ``pulls`` /
``snapshot_destinations`` — both by name (:class:`ConfigError`), never a silent skip, because a destination the code
cannot reach must not look configured. The cutover window converts the live file BEFORE the first new-code invocation
(``scripts/registry/landing/convert_config.py``: a byte-identical backup under ``state/``, the table stripped) and
validates the result with ``registry config --check``.

``harvest.load_config`` reads this file first and overlays the older ``config.json`` (whose
``mirrors`` / ``pulls`` lists the tests write directly), so both spellings keep working.
Python 3.11+ ``tomllib`` reads it; :func:`render_default` writes the file ``install_grading_host.sh`` lays
down (a TOML subset written by hand so no writer dependency is added).
"""
from __future__ import annotations

import getpass
import os
from typing import Any, Dict, List, Mapping, Optional

CONFIG_NAME = "config.toml"
#: The configuration table the loader REFUSES (the removed Modal destination) and the destination
#: kind a legacy ``config.json`` may still spell for it. Both are checked by name, never skipped.
REMOVED_TABLE = "modal"
REMOVED_KIND = "modal"
REMOVAL_REASON = "the Modal destination was removed (registry-slurm-only S3)"
CONVERTER = "scripts/registry/landing/convert_config.py"
#: A table this file never carries: the site's aliases live in ``site_aliases.toml`` beside it (``site_aliases.py``), and
#: :func:`load` refuses the table here by name, so there is one source.
SITE_ALIASES_TABLE = "site_aliases"
#: The destination lists a legacy ``config.json`` may carry (the shape :func:`as_json_config` derives from the TOML).
DESTINATION_LISTS = ("mirrors", "pulls", "snapshot_destinations")


class ConfigError(ValueError):
    """A configuration the loader refuses BY NAME: a table or a destination the code no longer has a transport for. Fail-closed:
    nothing derives from such a file, no silent fallback — the message names the file, the entry and the
    converter that fixes it."""


#: Every key with its default; ``~`` is expanded at load time for the path-valued ones.
DEFAULTS: Dict[str, Dict[str, Any]] = {
    "registry": {"root": "~/registry", "user": None},
    "paper": {
        "checkout": "~/constitutional-adapters",
        "publish_dir": "~/sisteer-pages/paper",
        "build_lock": "~/.local/state/paper-autobuild/lock",
        "python": "~/constitutional-adapters/.venv/bin/python",
        "keep_failed_builds": 3,
    },
    "status": {
        "page_dir": "~/sisteer-pages/status",
        # the render's budgets (registry-slurm-fast-windows, Phase 1 / S4): a section query past `section_budget_s` is
        # interrupted and rendered from its last good rows (state/status_sections/<name>.json) with a visible "as of" mark, and
        # joins the cached-with-refresh set (state/status_heavy.json) that `registry status --refresh-heavy` recomputes exactly,
        # each member under `heavy_budget_s`; `render_budget_s` bounds the whole render — sections it cannot reach are served
        # from cache and marked "skipped this render"
        "section_budget_s": 20,
        "render_budget_s": 120,
        "heavy_budget_s": 900,
    },
    "snapshot": {
        # while runs are in flight, a run-level set every 5 min; a FULL set (every table's Parquet) daily
        # or on demand; an idle registry whose head moved republishes hourly
        "in_flight_every_s": 300,
        "idle_every_s": 3600,
        "full_every_s": 86400,
        "runlevel_only_in_flight": True,
    },
    "harvest": {
        "pull_clusters": True,
        # the expected wall time between two COMPLETED harvester cycles: the timer's 30-s gap after a cycle ends plus a healthy cycle's fold. The
        # status page's harvester row turns red when the last completed cycle is older than 3x this (a unit killed at its
        # TimeoutStartSec completes nothing; four in a row froze the paper frame for four hours, 2026-09-29)
        "cadence_s": 1200,
        # a held revision whose attempt identity stands (the same files, marker, judging code and parent
        # state) is re-attempted at most once per this many seconds; the window's uncapped cycles and `ingest --retry-held` re-attempt every one
        "held_sweep_s": 3600,
    },
    "repo": {"checkout": "~/constitutional-adapters", "python": "~/constitutional-adapters/.venv/bin/python"},
}
PATH_KEYS = {("registry", "root"), ("paper", "checkout"), ("paper", "publish_dir"), ("paper", "build_lock"), ("paper", "python"),
             ("status", "page_dir"), ("repo", "checkout"), ("repo", "python")}
CLUSTER_DEFAULTS = [
    {"name": "cluster_a", "host": "cluster-a-login", "shared": "/scratch/example/{user}", "qos": "normal"},
    {"name": "cluster_b", "host": "cluster-b-login", "shared": "/data/example/{user}", "qos": "normal"},
]
#: A cluster entry without `qos`: the default entry's of the same name, else `normal`. A site whose QoS names differ sets `qos` on
#: every entry (the live configuration names each one).
DEFAULT_CLUSTER_QOS = {c["name"]: c["qos"] for c in CLUSTER_DEFAULTS}
#: The full-verify job's shape and the per-cycle sample when the entry does not name them (sisteer.registry.mirrors constants).
CLUSTER_VERIFY_DEFAULTS = {"slurm": True, "partition": "general", "verify_cpus": 4, "verify_mem": "16G", "verify_sample": 2000}


def config_path(root: str) -> str:
    return os.path.join(root, CONFIG_NAME)


def _expand(v: Any) -> Any:
    return os.path.expanduser(v) if isinstance(v, str) and v.startswith("~") else v


def removed_table_refusal(path: str, table: Mapping[str, Any], root: Optional[str] = None) -> str:
    """The refusal :func:`load` raises for a file that still carries the removed table: the file, the reason, the keys it found
    and the converter that strips it (the cutover's `-- config convert` stage)."""
    keys = ", ".join(sorted(str(k) for k in table)) or "no keys"
    where = f" --root {root}" if root else " --root <registry root>"
    return (f"{path}: {REMOVAL_REASON}: the file still carries a `{REMOVED_TABLE}` table ({keys}); nothing is derived from it. "
            f"Convert the file first — `python3 {CONVERTER}{where}` strips the table after a byte-identical backup under state/ — "
            f"then validate with `registry config --check`")


def load(root: str, path: Optional[str] = None) -> Dict[str, Any]:
    """The merged configuration (defaults ← file); absent file → defaults. Path values are expanded.

    Raises :class:`ConfigError` — by name, before anything derives from the file — when the file still carries the removed
    ``modal`` table (:func:`removed_table_refusal`), or a ``[site_aliases]`` table, which belongs in
    ``site_aliases.toml`` beside it."""
    import tomllib
    cfg: Dict[str, Any] = {k: dict(v) for k, v in DEFAULTS.items()}
    cfg["clusters"] = []
    p = path or config_path(root)
    if os.path.exists(p):
        with open(p, "rb") as fh:
            data = tomllib.load(fh)
        if REMOVED_TABLE in data:
            table = data[REMOVED_TABLE]
            raise ConfigError(removed_table_refusal(p, table if isinstance(table, Mapping) else {}, root))
        if SITE_ALIASES_TABLE in data:
            # the site's alias table lives in its own file beside this one, which no tool rewrites (site_aliases.py)
            from . import site_aliases as SA
            raise SA.SiteAliasError(SA.config_table_refusal(p))
        for section, values in data.items():
            if section == "clusters":
                cfg["clusters"] = [dict(c) for c in values]
            elif isinstance(values, Mapping):
                cfg.setdefault(section, {}).update(values)
            else:
                cfg[section] = values
    # the root the caller resolved (--root / $REGISTRY_ROOT / the default) IS the root; the file's value is informational
    cfg["registry"]["root_declared"] = _expand(cfg["registry"].get("root"))
    cfg["registry"]["root"] = os.path.abspath(root)
    if not cfg["registry"].get("user"):
        cfg["registry"]["user"] = getpass.getuser()
    for section, key in PATH_KEYS:
        if section in cfg and key in cfg[section]:
            cfg[section][key] = _expand(cfg[section][key])
    for c in cfg["clusters"]:
        c["shared"] = str(c.get("shared", "")).format(user=cfg["registry"]["user"])
    cfg["_path"] = p
    cfg["_present"] = os.path.exists(p)
    return cfg


def check_destinations(cfg_json: Mapping[str, Any], source: str) -> None:
    """Refuse, by name, a destination of the removed kind anywhere in the ``config.json`` shape (``mirrors`` / ``pulls`` /
    ``snapshot_destinations``): ``harvest.load_config`` calls this over the merged document whenever a legacy ``config.json``
    overlaid the TOML (an entry written there is taken verbatim, so it is the one place such an entry can still appear), and
    ``registry config --check`` reports the same refusal. Raises :class:`ConfigError`; returns None when clean."""
    for key in DESTINATION_LISTS:
        entries = cfg_json.get(key)
        if not isinstance(entries, list):
            continue
        for i, d in enumerate(entries):
            if not isinstance(d, Mapping):
                continue
            kind = str(d.get("kind", "") or "").lower()
            if kind == REMOVED_KIND:
                raise ConfigError(f"{source}: {REMOVAL_REASON}: {key}[{i}] (name {str(d.get('name', '?'))!r}) has kind {kind!r}; "
                                  f"remove the entry — no transport exists for it, and nothing is derived from a file that names one")


def get(cfg: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            raise KeyError(dotted)
    return cur


# --------------------------------------------------------------------------------------------
# derived destinations
# --------------------------------------------------------------------------------------------


def checkout_name(cfg: Optional[Mapping[str, Any]] = None) -> str:
    """The directory name of ``[repo] checkout``: the repo-mirror tool keys each clone on a cluster by it (``<shared>/repos/<name>``),
    and the landing scripts derive the same name from the same checkout (scripts/registry/landing/site_env.sh)."""
    checkout = ((cfg or {}).get("repo") or {}).get("checkout") or DEFAULTS["repo"]["checkout"]
    return os.path.basename(os.path.expanduser(str(checkout)).rstrip("/"))


def remote_src(cluster: Mapping[str, Any], cfg: Optional[Mapping[str, Any]] = None) -> str:
    """The cluster's clone of this repository, as its ``src``: the entry's ``remote_src``, else ``<shared>/repos/<checkout name>/src``."""
    return cluster.get("remote_src") or os.path.join(cluster["shared"], "repos", checkout_name(cfg), "src")


def mirror_destinations(cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """``mirrors`` entries for :mod:`sisteer.registry.mirrors`: the complete log to each cluster's ``<shared>/registry/log``
    over ssh — the record log's copies are the grading host and the clusters (the 2026-09-28 ruling; the Modal copy is gone with
    its transport)."""
    out: List[Dict[str, Any]] = []
    for c in cfg.get("clusters", []):
        d = {"name": c["name"], "kind": "ssh", "host": c["host"], "path": os.path.join(c["shared"], "registry", "log"),
             "remote_src": remote_src(c, cfg), "qos": c.get("qos") or DEFAULT_CLUSTER_QOS.get(c["name"], "normal")}
        # the FULL verify runs as a Slurm step (`slurm = false` keeps it on the login node — a fresh mirror's bootstrap only)
        # and the incremental verify re-hashes `verify_sample` carried entries per cycle; defaults unless the entry names them
        for key, default in CLUSTER_VERIFY_DEFAULTS.items():
            d[key] = c.get(key, default)
        out.append(d)
    return out


def pull_sources(cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """``pulls`` entries for ``harvest.pull_clusters``: each cluster's producer spool."""
    return [{"name": c["name"], "host": c["host"], "path": os.path.join(c["shared"], "registry", "records")}
            for c in cfg.get("clusters", [])]


def snapshot_destinations(cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """``snapshot_destinations`` entries for :func:`sisteer.registry.snapshot.push`: each cluster's ``<shared>/registry/snapshot``."""
    out: List[Dict[str, Any]] = []
    for c in cfg.get("clusters", []):
        out.append({"name": c["name"], "kind": "ssh", "host": c["host"], "path": os.path.join(c["shared"], "registry", "snapshot"),
                    "remote_src": remote_src(c, cfg)})
    return out


def as_json_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``config.json`` shape (``mirrors`` / ``pulls`` / ``snapshot_destinations``) derived from the TOML."""
    return {"mirrors": mirror_destinations(cfg), "pulls": pull_sources(cfg), "snapshot_destinations": snapshot_destinations(cfg)}


# --------------------------------------------------------------------------------------------
# the file install_grading_host.sh writes
# --------------------------------------------------------------------------------------------


def render_default(root: str, user: Optional[str] = None, paper_checkout: Optional[str] = None) -> str:
    user = user or getpass.getuser()
    home = os.path.expanduser("~")

    def tilde(p: str) -> str:
        p = os.path.abspath(os.path.expanduser(p))
        return "~" + p[len(home):] if p.startswith(home + os.sep) or p == home else p
    d = DEFAULTS
    lines = [
        "# ~/registry/config.toml — the results registry's operating configuration.",
        "# Paths and names only; NO secrets live here (the registry needs none: ssh goes through the agent socket).",
        "",
        "[registry]",
        f'root = "{tilde(root)}"',
        f'user = "{user}"                     # the cluster login; <shared> roots below are per user',
        "",
        "[paper]",
        f'checkout = "{tilde(paper_checkout or d["paper"]["checkout"])}"   # paper_rebuild.sh builds paper/build.sh from here',
        f'publish_dir = "{tilde(d["paper"]["publish_dir"])}"',
        f'build_lock = "{tilde(d["paper"]["build_lock"])}"       # the builder\'s OWN lock, taken non-blocking; never the harvester\'s',
        f'python = "{tilde(d["paper"]["python"])}"',
        f'keep_failed_builds = {d["paper"]["keep_failed_builds"]}',
        "",
        "[status]",
        f'page_dir = "{tilde(d["status"]["page_dir"])}"',
        f'section_budget_s = {d["status"]["section_budget_s"]}          # a section query past this is interrupted and rendered from its last good rows',
        f'render_budget_s = {d["status"]["render_budget_s"]}          # the whole render; sections it cannot reach are served from cache, marked',
        f'heavy_budget_s = {d["status"]["heavy_budget_s"]}          # `registry status --refresh-heavy`: each cached section recomputed exactly under this',
        "",
        "[snapshot]",
        f'in_flight_every_s = {d["snapshot"]["in_flight_every_s"]}     # run-level set while runs are in flight',
        f'idle_every_s = {d["snapshot"]["idle_every_s"]}         # an idle registry whose head moved republishes hourly',
        f'full_every_s = {d["snapshot"]["full_every_s"]}        # the FULL Parquet set (every table) daily; `registry snapshot` on demand',
        f'runlevel_only_in_flight = {"true" if d["snapshot"]["runlevel_only_in_flight"] else "false"}',
        "",
        "[harvest]",
        "pull_clusters = true            # rsync each cluster's spool into the inbox every cycle (the push's backstop)",
        f'cadence_s = {d["harvest"]["cadence_s"]}               # expected seconds between COMPLETED cycles; the status page reddens the harvester row past 3x',
        f'held_sweep_s = {d["harvest"]["held_sweep_s"]}            # a held revision whose attempt identity stands is re-attempted at most once per this many seconds (D961)',
        "",
    ]
    for c in CLUSTER_DEFAULTS:
        lines += ["[[clusters]]", f'name = "{c["name"]}"', f'host = "{c["host"]}"', f'shared = "{c["shared"].format(user=user)}"',
                  f'qos = "{c["qos"]}"                    # the full record-log verify runs as a CPU-only Slurm step at this QoS', ""]
    lines += [
        "[repo]",
        f'checkout = "{tilde(d["repo"]["checkout"])}"',
        f'python = "{tilde(d["repo"]["python"])}"',
        "",
    ]
    return "\n".join(lines)
