#!/usr/bin/env python3
"""reproduce/lib/clusters.py — the one reader of the site's cluster blocks (stdlib only).

The site config (`$SISTEER_SITE_DIR/config.env`, or the file `--config` / `SISTEER_CONFIG` names in that directory) names
its clusters (`SISTEER_CLUSTERS="cluster_a cluster_b"`) and one
file per label under `clusters/<label>.env` describes each: ssh alias, shared root, partition, QoS, account, extra
sbatch flags, HF cache, HF token source, repo mirror, registry substrate + label, `high` budget, the grading host's
mirror root, the prebuilt fit venv's name (CLUSTER_FIT_ENV_NAME). Nothing in the repo hard-codes a host: every consumer — the shell library (`cluster_var`), the stage
scripts, `stage_strength.py`'s routing, `petri_drive_full22.sh`'s host lookup — comes through here.

    python reproduce/lib/clusters.py --site-dir DIR --sh          # `CLUSTER__<label>__<VAR>='…'` assignments for `eval`
    python reproduce/lib/clusters.py --config DIR/other.env --sh  # the same, for the labels THAT config file names (its directory is the site dir)
    python reproduce/lib/clusters.py --site-dir DIR --json        # {label: {VAR: value}}
    python reproduce/lib/clusters.py --site-dir DIR --get LABEL VAR   # one value; LABEL is a site label or a registry label
                                                                      # (exit 3 when no block matches)
    python reproduce/lib/clusters.py --job-paths LABEL            # `export SISTEER_PROJECT_ROOT=…` (and the eval-repo, Petri and
                                                                  # repo-mirror roots) for a job submitted by hand

A site directory may hold more than one config file (an experiment's beside config.env), and a run reads ONE of them: the
file it was given decides the scalars AND the cluster list, never config.env beside it (site_config_file; submit_twin.Site
and common.sh repro_load_config both go through it).

SISTEER_CLUSTERS is the set of clusters the front door fans out to by default (`--sh`, `--json`, `--labels`, load_all). A
lookup BY LABEL (`--get`, `--job-paths`, block_for, load_addressable) reads every block in the site's `clusters/` directory,
listed or not, so a cluster that only data names (a second cluster the site does not submit to by default, named by the
registry label on its records) still resolves to its own login host and roots. Every block a lookup can reach is validated
in full, and a `.env` file there whose name is not a site label refuses.

Each cluster has three names, each with one job:
  * the site label: the `clusters/<label>.env` file name, the label SISTEER_CLUSTERS lists, and CLUSTER_LABEL. The front
    door's per-cluster loops use it, and records carry CLUSTER_LABEL as extra_json.cluster_label.
  * the registry label, CLUSTER_SUBSTRATE: a value of the registry's substrate vocabulary. It is the cluster's
    `[[clusters]].name` in the registry config, every `--dest` the landing tools pass, and the substrate of the records
    the cluster's jobs emit. The generic `slurm` serves any site whose clusters are not the registry's named ones, and it
    requires CLUSTER_LABEL. Anything else refuses by name.
  * the mirror selector, CLUSTER_MIRROR_SELECTOR (optional, default the site label): only the argument the site's
    repo-mirror tool knows this cluster by.

Values are parsed by src/sisteer/site.py's parser, the same one the placeholder expansion reads the site config with.
`$SHARED` (or `${SHARED}`) inside a block means that cluster's own CLUSTER_SHARED_ROOT, which is itself a literal path. Any
other `$` in a block refuses by name at load: another site placeholder, a name outside the closed set (a misspelt `$SHARE`
among them) and a bare `$` alike, since nothing would expand it later and it would reach a command as written. A value of the
site config file may hold the site placeholders and no other `$`: an environment variable ($HOME, $USER, ...) refuses by key
at load too, because the front door's shell expands it when it sources the file while every Python reader (submit_twin.py
--config, site.py) keeps it as written, so one file would name two different paths. common.sh refuses what bash left in a
value after sourcing; this module reads the file as written and refuses the rest.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import sys

REQUIRED = ("CLUSTER_SSH", "CLUSTER_SHARED_ROOT", "CLUSTER_PARTITION", "CLUSTER_QOS", "CLUSTER_SUBSTRATE", "CLUSTER_LABEL")
OPTIONAL_DEFAULTS = {
    "CLUSTER_ACCOUNT": "",
    "CLUSTER_SBATCH_EXTRA": "",
    "CLUSTER_HF_HOME": "$SHARED/hf_cache",
    "CLUSTER_HF_TOKEN_SOURCE": "none",
    "CLUSTER_REPO_MIRROR": "$SHARED/repos/constitutional-adapters",
    "CLUSTER_EVAL_REPO_ROOT": "$SHARED/sisteer-eval-repo",
    "CLUSTER_PETRI_REPO_ROOT": "$SHARED/sisteer-petri-repo",
    # the project directory cluster jobs write and read: <root>/<dir> and the per-experiment siblings <root>-<suffix>
    "CLUSTER_PROJECT_ROOT": "$SHARED/sisteer",
    # the argument the site's repo-mirror tool knows this cluster by; empty means the site label (filled by load_cluster)
    "CLUSTER_MIRROR_SELECTOR": "",
    "CLUSTER_HIGH_BUDGET_GPUS": "16",
    "CLUSTER_GRADE_MIRROR_ROOT": "",
    "CLUSTER_LOG_DIR": "$SHARED/exp/logs",
    "CLUSTER_DEV_PARTITION": "",
    "CLUSTER_DEV_QOS": "",
    "CLUSTER_HELPERS_DIR": "$SHARED/scripts",
    "CLUSTER_ENVS_DIR": "$SHARED/envs",
    # the prebuilt fit venv NAME under CLUSTER_ENVS_DIR (scripts/sbatch/sdfbase_fit_env.sbatch <name>); empty -> the site's global
    # SISTEER_FIT_ENV_NAME, else `sdfbase-fit` (common.sh cluster_fit_env_name / submit_twin.Site.fit_env_name — the ONE resolution
    # both faces share: a job on the other cluster must run on THAT cluster's venv name)
    "CLUSTER_FIT_ENV_NAME": "",
}
KNOWN_SUBSTRATES = ("cluster_a", "cluster_b", "slurm")
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
LABEL_RE = re.compile(r"^[a-z][a-z0-9_]*$")


# the job-environment names of the cluster's directory roots: the front door's job env (lib/slurm.sh slurm_job_env,
# submit_twin.job_env) carries them, and cluster job scripts refuse by name without them
JOB_PATHS = (("SISTEER_PROJECT_ROOT", "CLUSTER_PROJECT_ROOT"), ("SISTEER_EVAL_REPO_ROOT", "CLUSTER_EVAL_REPO_ROOT"),
             ("SISTEER_PETRI_REPO_ROOT", "CLUSTER_PETRI_REPO_ROOT"), ("SISTEER_REPO_MIRROR", "CLUSTER_REPO_MIRROR"))


class ClusterConfigError(SystemExit):
    def __init__(self, msg: str):
        super().__init__(f"REFUSED: {msg}")


def _load_site_module():
    """src/sisteer/site.py of the checkout this file belongs to, loaded by path: the front door's interpreter need not
    have the package installed, and an installed copy from another checkout must not answer for this one."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "src", "sisteer", "site.py")
    path = os.path.normpath(path)
    if not os.path.isfile(path):
        raise ClusterConfigError(f"{path} is missing: reproduce/lib/clusters.py parses the site files with it (the checkout "
                                 "must hold src/ beside reproduce/)")
    name = "_sisteer_site_for_clusters"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # registered first: site.py's dataclasses look their module up while it executes
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


_SITE = None


def _site():
    global _SITE
    if _SITE is None:
        _SITE = _load_site_module()
    return _SITE


def site_module():
    """src/sisteer/site.py of this checkout (parser, placeholder expansion), for front-door helpers whose interpreter may not
    have the sisteer package installed."""
    return _site()


def parse_env_file(path: str) -> dict:
    """KEY=value lines (bash-quoted values accepted; `$SHARED` kept literal for later expansion), by site.py's parser."""
    S = _site()
    try:
        return S.parse_env_file(path)
    except S.SiteConfigError as exc:
        raise ClusterConfigError(str(exc))


def _spelled(names) -> str:
    """`$` tokens as placeholders_in reports them, spelled as written: `$NAME`, `${raw}` for an invalid braced name, and a bare
    or unterminated `$` by that description."""
    out = []
    for n in names:
        out.append("a bare or unterminated '$'" if n == "" else ("$" + n if _NAME_RE.match(n) else "${" + n + "}"))
    return ", ".join(out)


def _known_text() -> str:
    return ", ".join("$" + n for n in _site().NAMES)


def check_site_values(config: dict, path: str) -> dict:
    """Refuse, by key, a site config value holding a `$` token that is not a site placeholder. A misspelt `$SHARE` or a bare `$`
    would reach a path or a command as written; an environment variable ($HOME, $USER) would be expanded by the front door's shell
    when it sources the file and kept as written by every Python reader, so the same file would name two paths. Returns `config`."""
    S = _site()
    for k, v in config.items():
        bad = [n for n in S.placeholders_in(v) if n not in S.NAMES]
        if bad:
            raise ClusterConfigError(f"{path}: {k}={v!r} holds {_spelled(bad)}, which is not a site placeholder ({_known_text()}); "
                                     "a site value names no environment variable, since the shell that sources this file would expand "
                                     "it and the Python readers would not: write the path out (`~` for the home directory)")
    return config


def site_config(site_dir: str) -> dict:
    cfg = os.path.join(site_dir, "config.env")
    if not os.path.isfile(cfg):
        raise ClusterConfigError(f"no config.env under {site_dir}")
    return check_site_values(parse_env_file(cfg), cfg)


def site_config_file(path: str) -> dict:
    """The site config at `path` — the file a run was GIVEN (`--config`, `SISTEER_CONFIG`; `config.env` by default), parsed as
    site_config parses config.env. A site directory may hold more than one (an experiment's file beside config.env), and every
    reader of the file's scalars — the registry campaign, the primary cluster, the cluster LIST — reads the one the run was given,
    never the directory's config.env beside it: submit_twin.Site read config.env and composed every job under ITS campaign while
    the stage, which sources the file it is given, read done-ness under the other's (demo-import integration round 2). A
    missing file refuses by its own name."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise ClusterConfigError(f"no site config at {path}")
    return check_site_values(parse_env_file(path), path)


def cluster_labels(site_dir: str, config: dict | None = None) -> list:
    config = config if config is not None else site_config(site_dir)
    labels = (config.get("SISTEER_CLUSTERS") or "").split()
    if not labels:
        raise ClusterConfigError("the site config sets no SISTEER_CLUSTERS")
    for lab in labels:
        if not LABEL_RE.match(lab):
            raise ClusterConfigError(f"cluster label {lab!r} must match {LABEL_RE.pattern}")
    return labels


def load_cluster(site_dir: str, label: str) -> dict:
    path = os.path.join(site_dir, "clusters", f"{label}.env")
    if not os.path.isfile(path):
        raise ClusterConfigError(f"cluster {label!r} is named in SISTEER_CLUSTERS but {path} does not exist")
    block = parse_env_file(path)
    for k in REQUIRED:
        if not block.get(k):
            raise ClusterConfigError(f"{path} must set {k}")
    for k, v in OPTIONAL_DEFAULTS.items():
        block.setdefault(k, v)
    sub = block["CLUSTER_SUBSTRATE"]
    if sub not in KNOWN_SUBSTRATES:
        raise ClusterConfigError(f"{path}: CLUSTER_SUBSTRATE={sub!r} is not a registry cluster label ({'|'.join(KNOWN_SUBSTRATES)}); "
                                 "the registry label is a substrate value, and the site label stays in CLUSTER_LABEL")
    if sub == "slurm" and not block.get("CLUSTER_LABEL"):
        raise ClusterConfigError(f"{path}: CLUSTER_SUBSTRATE=slurm requires CLUSTER_LABEL (carried as extra_json.cluster_label)")
    if not block["CLUSTER_SHARED_ROOT"].startswith("/"):
        raise ClusterConfigError(f"{path}: CLUSTER_SHARED_ROOT must be absolute, got {block['CLUSTER_SHARED_ROOT']!r}")
    try:
        int(block["CLUSTER_HIGH_BUDGET_GPUS"])
    except ValueError:
        raise ClusterConfigError(f"{path}: CLUSTER_HIGH_BUDGET_GPUS must be an integer")
    if not block.get("CLUSTER_MIRROR_SELECTOR"):
        block["CLUSTER_MIRROR_SELECTOR"] = label
    # `$SHARED` / `${SHARED}` inside a cluster block means that cluster's own root; any other `$` refuses by name
    S = _site()
    root = block["CLUSTER_SHARED_ROOT"]
    if "$" in root:
        raise ClusterConfigError(f"{path}: CLUSTER_SHARED_ROOT={root!r} holds {_spelled(S.placeholders_in(root))}; the shared root "
                                 "is what $SHARED stands for, so write it as a literal absolute path")
    for k, v in list(block.items()):
        if isinstance(v, str) and "$" in v:
            v = S.substitute_shared(v, root)
            left = S.placeholders_in(v)
            other = [n for n in left if n in S.PLACEHOLDERS]
            if other:
                raise ClusterConfigError(f"{path}: {k}={v!r} holds {_spelled(other)}; a cluster block "
                                         "expands only $SHARED (its own CLUSTER_SHARED_ROOT), so write the path out")
            if left:
                raise ClusterConfigError(f"{path}: {k}={v!r} holds {_spelled(left)}, which is not a site placeholder ({_known_text()}); "
                                         "a cluster block expands only $SHARED (its own CLUSTER_SHARED_ROOT)")
            block[k] = v
    return block


def load_all(site_dir: str, config: dict | None = None) -> dict:
    """Every cluster block the site names: `config` (the parsed site config the caller was GIVEN — site_config_file) names the labels
    and the primary; None reads `<site_dir>/config.env`. The blocks themselves are always `<site_dir>/clusters/<label>.env`."""
    config = config if config is not None else site_config(site_dir)
    labels = cluster_labels(site_dir, config)
    out = {lab: load_cluster(site_dir, lab) for lab in labels}
    primary = config.get("SISTEER_PRIMARY_CLUSTER", "")
    if primary and primary not in out:
        raise ClusterConfigError(f"SISTEER_PRIMARY_CLUSTER={primary!r} is not one of {labels}")
    return out


def block_file_labels(site_dir: str) -> list:
    """The site label of every block file in `<site_dir>/clusters/` (`<label>.env`), sorted. A `.env` file there whose name is
    not a site label refuses by name: no lookup could reach it, and it may be the block a lookup is looking for."""
    d = os.path.join(site_dir, "clusters")
    if not os.path.isdir(d):
        return []
    labels = []
    for name in sorted(os.listdir(d)):
        if not name.endswith(".env"):
            continue
        lab = name[: -len(".env")]
        if not LABEL_RE.match(lab):
            raise ClusterConfigError(f"{os.path.join(d, name)}: a cluster block's file name is its site label, which must match "
                                     f"{LABEL_RE.pattern}; rename the file or move it out of {d}")
        labels.append(lab)
    return labels


def load_addressable(site_dir: str, config: dict | None = None) -> dict:
    """Every cluster block a lookup BY LABEL reaches: the blocks the site config names (load_all, listed first, the primary
    checked), then every other `clusters/<label>.env` in the site directory, each validated as load_cluster validates a named
    one. SISTEER_CLUSTERS stays the set the front door fans out to by default; a block outside it is reached only when a caller
    names it, by its site label or by the registry label data carries (a record's substrate, a pins source, a store
    destination). One block that fails validation refuses every lookup, since it may carry the label asked for."""
    out = load_all(site_dir, config)
    for lab in block_file_labels(site_dir):
        if lab not in out:
            out[lab] = load_cluster(site_dir, lab)
    return out


def resolve_label(blocks: dict, label: str) -> str:
    """The site label of the block `label` names. `label` is a site label, or a registry label (CLUSTER_SUBSTRATE) that
    exactly one of `blocks` carries. Data names a cluster by its registry label (a record's substrate, a pins source
    prefix, a store destination), while the block file keeps the site's own name, so the two differ on a site whose
    cluster names are private. A caller that holds a registry label reads the cluster's ssh alias, shared root and QoS
    through this; pass it load_addressable's blocks so a cluster outside the default fan-out resolves too. Refuses by name
    when no block matches, when more than one block carries the registry label, and when the label is one block's site
    label and another block's registry label."""
    hits = [lab for lab, b in blocks.items() if b.get("CLUSTER_SUBSTRATE") == label]
    if label in blocks:
        others = [lab for lab in hits if lab != label]
        if others:
            raise ClusterConfigError(f"{label!r} is the site label of one cluster block ({label}) and the registry label of "
                                     f"another ({' '.join(others)}); rename the block file so the two cannot be confused")
        return label
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise ClusterConfigError(f"registry label {label!r} is carried by more than one cluster block ({' '.join(hits)}); "
                                 "name the cluster by its site label")
    regs = sorted({str(b.get("CLUSTER_SUBSTRATE", "")) for b in blocks.values()} - {""})
    raise ClusterConfigError(f"cluster {label!r} is neither a site label ({' '.join(blocks) or 'none configured'}) nor the "
                             f"registry label of one ({' '.join(regs) or 'none'}): add a block for it under the site's clusters/ "
                             f"directory, a <site label>.env file with CLUSTER_SUBSTRATE={label}")


def block_for(label: str, config_path: str | None = None) -> dict:
    """The block of the cluster `label` names (a site label or a registry label, as resolve_label reads it), from the site
    config this process uses: `config_path`, else $SISTEER_CONFIG, else `<site dir>/config.env`, among every block in that
    directory's clusters/ (load_addressable: listed in SISTEER_CLUSTERS or not). Its CLUSTER_SSH is the login host's ssh alias,
    CLUSTER_SHARED_ROOT the shared root and CLUSTER_QOS the QoS, so a caller never derives a host from the label. Refuses by
    name when the site config is missing or describes no such cluster."""
    path = os.path.abspath(config_path) if config_path else default_config_path()
    config = site_config_file(path)
    blocks = load_addressable(os.path.dirname(path), config)
    return blocks[resolve_label(blocks, label)]


def default_site_dir() -> str:
    """$SISTEER_CONFIG's directory, else $SISTEER_SITE_DIR, else ~/.config/sisteer (site.py's resolution)."""
    return _site().site_dir()


def default_config_path() -> str:
    """The site config FILE a process with no --config reads: $SISTEER_CONFIG (a stage exports the file it loaded), else
    `<default site dir>/config.env` — the same resolution as common.sh repro_config_path and site.py's config_path."""
    return _site().config_path()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--site-dir", default=None, help="the site directory (clusters/<label>.env; its config.env names the labels unless --config does)")
    ap.add_argument("--config", default=None, help="the site config FILE whose SISTEER_CLUSTERS / SISTEER_PRIMARY_CLUSTER are read (default: "
                    "$SISTEER_CONFIG, else <site dir>/config.env); its directory is the site dir unless --site-dir says otherwise")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sh", action="store_true")
    g.add_argument("--json", action="store_true")
    g.add_argument("--get", nargs=2, metavar=("LABEL", "VAR"), help="one value of the block LABEL names (a site or registry label; "
                   "every block under clusters/, listed in SISTEER_CLUSTERS or not)")
    g.add_argument("--labels", action="store_true")
    g.add_argument("--job-paths", metavar="LABEL", help="export lines for the job-environment roots of LABEL (a site or registry label; "
                   "every block under clusters/, listed in SISTEER_CLUSTERS or not)")
    args = ap.parse_args(argv)
    # the config FILE decides the labels: --config, else (with no --site-dir) $SISTEER_CONFIG / the default site dir's config.env;
    # --site-dir alone keeps reading that directory's config.env
    cfg_path = os.path.abspath(args.config) if args.config else (None if args.site_dir else default_config_path())
    site_dir = args.site_dir or os.path.dirname(cfg_path)
    try:
        config = site_config_file(cfg_path) if cfg_path else None
        if args.get or args.job_paths:
            label, var = args.get or (args.job_paths, None)
            all_ = load_addressable(site_dir, config)
            try:
                site_label = resolve_label(all_, label)
            except ClusterConfigError as exc:
                print(f"{str(exc.code).removeprefix('REFUSED: ')} (site dir {site_dir})", file=sys.stderr)
                return 3
            if var is None:
                for env_name, key in JOB_PATHS:
                    print(f"export {env_name}={shlex.quote(str(all_[site_label][key]))}")
            else:
                print(all_[site_label].get(var, ""))
            return 0
        blocks = load_all(site_dir, config)
    except ClusterConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.labels:
        print("\n".join(blocks))
    elif args.json:
        print(json.dumps(blocks, indent=2, sort_keys=True))
    else:
        for lab, block in blocks.items():
            for k, v in sorted(block.items()):
                print(f"export CLUSTER__{lab}__{k}={shlex.quote(str(v))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
