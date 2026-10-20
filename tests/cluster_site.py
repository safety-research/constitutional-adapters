"""A two-cluster site config for tests of scripts that reach a cluster over ssh, rsync or a printed command.

Each cluster gets three different names: a site label (the clusters/<label>.env file: site_one, site_two), a registry label
(CLUSTER_SUBSTRATE, the name data carries: REGISTRY["a"], REGISTRY["b"]) and an ssh alias (login-a, login-b). Its directories
sit under its own shared root and its QoS is qos-a / qos-b, so a value read from the right block is recognisably that
cluster's, and a script that still derived a host from the registry label would show the label where the alias belongs.
Its partitions (part-a / part-b) and CPU-step partition and QoS (devpart-a / devqos-a, ...) differ the same way."""
from __future__ import annotations

from pathlib import Path

REGISTRY = {"a": "cluster_a", "b": "cluster_b"}       # the registry labels (CLUSTER_SUBSTRATE)
SITE_LABEL = {"a": "site_one", "b": "site_two"}         # the site's own names (block file names, CLUSTER_LABEL)
LOGIN = {"a": "login-a", "b": "login-b"}                # the ssh aliases (CLUSTER_SSH)


def write_site(root: Path, keys=("a", "b"), extra: dict | None = None) -> Path:
    """Write the site under `root/site` naming the clusters `keys` (a subset of "a", "b"); return its config.env.
    `extra[k]` adds keys to cluster k's block (CLUSTER_SBATCH_EXTRA, say)."""
    site = root / "site"
    (site / "clusters").mkdir(parents=True, exist_ok=True)
    (site / "config.env").write_text(f'SISTEER_CLUSTERS="{" ".join(SITE_LABEL[k] for k in keys)}"\n'
                                     f"SISTEER_PRIMARY_CLUSTER={SITE_LABEL[keys[0]]}\n")
    for k in keys:
        shared = root / f"shared_{k}"
        shared.mkdir(parents=True, exist_ok=True)
        lines = [f"CLUSTER_SSH={LOGIN[k]}", f"CLUSTER_SHARED_ROOT={shared}", f"CLUSTER_PARTITION=part-{k}", f"CLUSTER_QOS=qos-{k}",
                 f"CLUSTER_SUBSTRATE={REGISTRY[k]}", f"CLUSTER_LABEL={SITE_LABEL[k]}", f"CLUSTER_DEV_QOS=devqos-{k}",
                 f"CLUSTER_DEV_PARTITION=devpart-{k}",
                 "CLUSTER_EVAL_REPO_ROOT='$SHARED/evalrepo'", "CLUSTER_ENVS_DIR='$SHARED/envs'", "CLUSTER_LOG_DIR='$SHARED/logs'"]
        lines += [f"{kk}='{vv}'" for kk, vv in ((extra or {}).get(k) or {}).items()]
        (site / "clusters" / f"{SITE_LABEL[k]}.env").write_text("\n".join(lines) + "\n")
    return site / "config.env"


def expect(root: Path, k: str) -> dict:
    """What a script should read for cluster k from the site write_site(root) wrote. `sbatch` is the flags every sbatch
    line for cluster k carries before its --time (the partition, QoS and log file the site names)."""
    shared = root / f"shared_{k}"
    return {"host": LOGIN[k], "shared": str(shared), "eval": f"{shared}/evalrepo", "envs": f"{shared}/envs",
            "logs": f"{shared}/logs", "qos": f"qos-{k}", "part": f"part-{k}", "label": REGISTRY[k],
            "sbatch": f"--qos=qos-{k} --partition=part-{k} --output={shared}/logs/%x_%j.out"}
