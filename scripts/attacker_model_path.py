#!/usr/bin/env python3
"""Print the attacker model a config serves: the pinned snapshot directory ``attacker.path`` names, its site placeholder
expanded, checked complete on this host; else ``attacker.model``.

The configs write the attacker snapshot as ``$HF_HOME/hub/.../snapshots/<revision>`` (or ``$SHARED/...``), the placeholder
grammar of ``sisteer.site``. A job script that hands the value to ``vllm serve`` calls this instead of reading the YAML
itself: a shell never expands a ``$NAME`` that a Python one-liner printed, so a raw read hands the literal text on.

The value printed is always the pinned revision, or nothing. The placeholder resolves on this host (``HF_HOME`` from the
job's environment, the shared root from SISTEER_SHARED_ROOT or the site config's cluster block), and the directory it names
must end in ``attacker.snapshot`` and hold that snapshot whole: ``config.json`` and every weight file (each shard a
``*.safetensors.index.json`` names, else at least one ``*.safetensors`` or ``*.bin``), a dangling cache link counting as
absent. A config that pins ``attacker.snapshot`` without a path refuses too: serving the repo id would load whatever
revision a cache or the hub holds. Each refusal names the key and the reason, exits 2 and prints nothing on stdout, so a
caller that checks the exit status never serves a literal ``$HF_HOME`` directory or another revision.

    attacker_model_path.py CONFIG     # the snapshot directory to serve, or the repo id of a config that pins no snapshot

PyYAML and the standard library; ``sisteer.site`` is imported from this checkout's ``src``, so a job's venv needs no
install of the package (``sisteer/site.py`` is stdlib-only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sisteer import site  # noqa: E402


def _attacker_block(cfg: Mapping) -> Mapping:
    if not isinstance(cfg, Mapping):
        raise ValueError(f"the config is {type(cfg).__name__}, not a mapping")
    att = cfg.get("attacker") or {}
    if not isinstance(att, Mapping):
        raise ValueError(f"attacker is {type(att).__name__}, not a mapping")
    return att


def attacker_model_path(cfg: Mapping, env: Optional[Mapping[str, str]] = None) -> str:
    """The attacker model of ``cfg``: ``attacker.path`` expanded through :func:`sisteer.site.expand_path` (``~`` too),
    else ``attacker.model``. Text only: nothing on disk is read. Raises :class:`sisteer.site.PlaceholderError` for a
    placeholder with no value here and ValueError for a config with neither key."""
    att = _attacker_block(cfg)
    path = att.get("path")
    if path:
        return str(site.expand_path(path, env=env, key="attacker.path"))
    model = att.get("model")
    if not model:
        raise ValueError("the config names neither attacker.path nor attacker.model")
    return str(model)


def _missing_weights(snap: Path) -> list:
    """The weight files ``snap`` lacks: the shards its safetensors index names that are not regular files here, or a
    placeholder entry when it holds no index and no ``*.safetensors`` / ``*.bin`` file at all."""
    indexes = sorted(snap.glob("*.safetensors.index.json"))
    if indexes:
        missing = []
        for idx in indexes:
            try:
                weight_map = json.loads(idx.read_text(encoding="utf-8")).get("weight_map") or {}
            except (OSError, ValueError) as exc:
                return [f"{idx.name} (unreadable: {exc})"]
            missing += sorted(s for s in set(weight_map.values()) if not (snap / s).is_file())
        return missing
    if any(p.is_file() for pat in ("*.safetensors", "*.bin") for p in snap.glob(pat)):
        return []
    return ["any *.safetensors or *.bin file"]


def served_attacker_model(cfg: Mapping, env: Optional[Mapping[str, str]] = None) -> str:
    """What a job serves for ``cfg``: :func:`attacker_model_path`, checked on this host. A path must end in
    ``attacker.snapshot`` (when the config pins one) and hold that snapshot whole; a pinned snapshot with no path refuses.
    Raises ValueError (PlaceholderError is one) with the reason."""
    att = _attacker_block(cfg)
    got = attacker_model_path(cfg, env=env)
    pinned = str(att.get("snapshot") or "")
    model = str(att.get("model") or "the attacker")
    if not att.get("path"):
        if pinned:
            raise ValueError(f"attacker.snapshot pins {pinned} but the config names no attacker.path; serving the repo id "
                             f"{got} would load an unpinned revision")
        return got
    snap = Path(got)
    if pinned and snap.name != pinned:
        raise ValueError(f"attacker.path {got} does not end in the pinned snapshot {pinned}")
    fetch = (f"fetch revision {pinned} of {model} into this host's cache first (hf download {model} --revision {pinned})"
             if pinned else f"fetch {model} into {got} first")
    if not (snap / "config.json").is_file():
        raise ValueError(f"attacker.path: the pinned snapshot is not on this host: {got} has no config.json; {fetch}")
    missing = _missing_weights(snap)
    if missing:
        raise ValueError(f"attacker.path: the pinned snapshot at {got} is incomplete, it lacks {len(missing)} weight "
                         f"file(s) ({', '.join(missing[:3])}); {fetch}")
    return got


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", help="a run config with an attacker block (YAML)")
    args = ap.parse_args(argv)
    import yaml

    try:
        with open(args.config, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        print(served_attacker_model(cfg))
    except (OSError, ValueError, yaml.YAMLError) as exc:   # PlaceholderError and SiteConfigError are ValueErrors
        print(f"REFUSED: {args.config}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
