"""W&B run resolution for fits.

Spec "Out": no wider W&B ingestion. This module fills ONLY ``fit_wandb_url`` (and hands back ``config`` /
``summary`` for ``extra_json``) for run ids a sidecar or a trial log ALREADY names — never a run the
registry did not know about. Two read-only sources, in order:

1. the offline mirror ``<mirror>/{offline-run,run}-*-<id>/files/wandb-metadata.json`` (present for a run only when the
   mirror kept its metadata);
2. ``wandb.Api()``, ``api.run(entity/project/id)`` over the candidate projects, reading ``run.url`` / ``run.config`` /
   ``run.summary`` only.

The entity, the candidate projects and the mirror are the site's: the registry configuration's ``[wandb]`` table
(``entity``, ``projects``, ``mirror``; :func:`settings`), never a default in code. An unset entity, an unreachable API, a
disabled ``WANDB_MODE`` or an id no project knows leaves the URL null with ``field_confidence_json.fit_wandb_url =
unverified`` and the reason; an unset mirror is not consulted. Nothing is written to W&B.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .base import AdapterContext, Input, RecordBatch, SourceStateRow, check_input, store_listdir

HOME = os.path.expanduser("~")
#: The registry configuration's table this module reads (``config.toml``): ``entity`` (the W&B entity), ``projects`` (the
#: candidate projects, in lookup order) and ``mirror`` (the offline mirror's ``wandb`` directory, ``~`` expanded).
CONFIG_TABLE = "wandb"


@dataclass(frozen=True)
class Settings:
    entity: Optional[str] = None
    projects: Tuple[str, ...] = ()
    mirror: Optional[str] = None


def settings(root: Optional[str] = None, cfg: Optional[Mapping[str, object]] = None) -> Settings:
    """The ``[wandb]`` table of the registry configuration: ``cfg`` when given, else ``config.load`` of ``root`` (default
    ``$REGISTRY_ROOT``, else ``~/registry``, the registry's own default). Absent keys stay unset."""
    if cfg is None:
        from .. import config as CFG
        cfg = CFG.load(os.path.abspath(root or os.environ.get("REGISTRY_ROOT") or os.path.join(HOME, "registry")))
    table = cfg.get(CONFIG_TABLE) if isinstance(cfg, Mapping) else None
    table = table if isinstance(table, Mapping) else {}
    entity = str(table.get("entity") or "").strip() or None
    raw = table.get("projects") or ()
    projects = tuple(str(p).strip() for p in ([raw] if isinstance(raw, str) else raw) if str(p).strip())
    mirror = str(table.get("mirror") or "").strip()
    return Settings(entity, projects, os.path.expanduser(mirror) if mirror else None)


@dataclass
class Resolved:
    run_id: str
    url: Optional[str] = None
    project: Optional[str] = None
    entity: Optional[str] = None
    source: str = ""                       # mirror:<path> | api:<entity>/<project> | unresolved
    reason: Optional[str] = None           # why unresolved
    config: Dict[str, object] = field(default_factory=dict)
    summary: Dict[str, object] = field(default_factory=dict)
    metadata_path: Optional[str] = None

    def confidence(self) -> Dict[str, str]:
        if self.url:
            return {"confidence": "recorded", "source": self.source}
        return {"confidence": "unverified", "source": self.reason or "unresolved", "reason": self.reason or "unresolved"}


def _mirror_dirs(mirror: str) -> Dict[str, str]:
    """``{run_id: dir}`` for every run directory in the offline mirror (one level)."""
    out: Dict[str, str] = {}
    try:
        names = sorted(store_listdir(mirror))
    except OSError:
        return out
    for n in names:
        if (n.startswith("offline-run-") or n.startswith("run-")) and "-" in n:
            rid = n.rsplit("-", 1)[-1]
            out[rid] = os.path.join(mirror, n)
    return out


def _from_mirror(rid: str, run_dir: str, entity: str) -> Optional[Resolved]:
    meta_path = os.path.join(run_dir, "files", "wandb-metadata.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return None
    project = meta.get("project") if isinstance(meta, Mapping) else None
    ent = meta.get("entity") if isinstance(meta, Mapping) and meta.get("entity") else entity
    if not isinstance(project, str) or not project:
        return None
    r = Resolved(rid, url=f"https://wandb.ai/{ent}/{project}/runs/{rid}", project=project, entity=ent,
                 source=f"mirror:{meta_path}", metadata_path=meta_path)
    cfg_path = os.path.join(run_dir, "files", "config.yaml")
    if os.path.isfile(cfg_path):
        try:
            import yaml  # type: ignore
            with open(cfg_path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            r.config = {k: (v.get("value") if isinstance(v, Mapping) and "value" in v else v) for k, v in cfg.items() if not k.startswith("_")}
        except Exception:  # noqa: BLE001
            pass
    return r


def _api_runs(entity: str, projects: Sequence[str], ids: Sequence[str], timeout_s: float) -> Dict[str, Resolved]:
    """Look each id up through ``wandb.Api()``; a failure of the API as a whole marks every id unresolved."""
    out: Dict[str, Resolved] = {}
    if os.environ.get("WANDB_MODE", "").lower() in ("disabled", "offline"):
        for rid in ids:
            out[rid] = Resolved(rid, reason=f"WANDB_MODE={os.environ.get('WANDB_MODE')}: the API is not consulted")
        return out
    try:
        import wandb  # type: ignore
        api = wandb.Api(timeout=int(timeout_s))
    except Exception as exc:  # noqa: BLE001
        for rid in ids:
            out[rid] = Resolved(rid, reason=f"wandb API unavailable ({type(exc).__name__}: {str(exc)[:120]})")
        return out
    for rid in ids:
        last = None
        for project in projects:
            try:
                run = api.run(f"{entity}/{project}/{rid}")
                url = getattr(run, "url", None) or f"https://wandb.ai/{entity}/{project}/runs/{rid}"
                r = Resolved(rid, url=url, project=project, entity=entity, source=f"api:{entity}/{project}")
                try:
                    r.config = {k: v for k, v in dict(run.config).items() if not str(k).startswith("_")}
                except Exception:  # noqa: BLE001
                    pass
                try:
                    r.summary = {k: v for k, v in dict(run.summary).items() if not str(k).startswith("_")}
                except Exception:  # noqa: BLE001
                    pass
                out[rid] = r
                break
            except Exception as exc:  # noqa: BLE001 — a missing run in one project is expected
                last = f"{type(exc).__name__}: {str(exc)[:100]}"
        if rid not in out:
            out[rid] = Resolved(rid, reason=f"no project in {list(projects)} under {entity} knows run {rid}" + (f" (last error {last})" if last else ""))
    # second pass: the entity's OTHER projects (the tune_arm sweeps take their project from the fit config)
    missing = [rid for rid, r in out.items() if not r.url]
    if missing:
        try:
            others = [p.name for p in api.projects(entity) if p.name not in projects]
        except Exception as exc:  # noqa: BLE001
            others = []
            for rid in missing:
                out[rid].reason = (out[rid].reason or "") + f"; listing the entity's projects failed ({type(exc).__name__})"
        for rid in missing:
            for project in others:
                try:
                    run = api.run(f"{entity}/{project}/{rid}")
                except Exception:  # noqa: BLE001
                    continue
                url = getattr(run, "url", None) or f"https://wandb.ai/{entity}/{project}/runs/{rid}"
                r = Resolved(rid, url=url, project=project, entity=entity, source=f"api:{entity}/{project} (found by listing the entity's projects)")
                try:
                    r.config = {k: v for k, v in dict(run.config).items() if not str(k).startswith("_")}
                    r.summary = {k: v for k, v in dict(run.summary).items() if not str(k).startswith("_")}
                except Exception:  # noqa: BLE001
                    pass
                out[rid] = r
                break
            if not out[rid].url and others:
                out[rid].reason = f"no project of {entity} ({len(projects) + len(others)} tried) knows run {rid}"
    return out


def resolve(run_ids: Iterable[str], entity: Optional[str] = None, projects: Optional[Sequence[str]] = None,
            mirror: Optional[str] = None, use_api: bool = True, timeout_s: float = 20.0, root: Optional[str] = None) -> Dict[str, Resolved]:
    """Resolve every named run id: mirror first, API second; every id gets a :class:`Resolved`. ``entity`` / ``projects`` /
    ``mirror`` default to the registry configuration's (:func:`settings` of ``root``); with no entity the API is not consulted."""
    if entity is None or projects is None or mirror is None:
        site = settings(root)
        entity = site.entity if entity is None else entity
        projects = site.projects if projects is None else projects
        mirror = site.mirror if mirror is None else mirror
    ids = sorted({str(r) for r in run_ids if r})
    out: Dict[str, Resolved] = {}
    mirror_dirs = _mirror_dirs(mirror) if mirror else {}
    pending: List[str] = []
    for rid in ids:
        r = _from_mirror(rid, mirror_dirs[rid], entity or "") if rid in mirror_dirs else None
        if r:
            out[rid] = r
        else:
            pending.append(rid)
    if pending:
        if use_api and not entity:
            for rid in pending:
                out[rid] = Resolved(rid, reason=f"no W&B entity is configured ([{CONFIG_TABLE}] entity in the registry's config.toml): "
                                                "the API is not consulted")
        elif use_api:
            out.update(_api_runs(str(entity), tuple(projects), pending, timeout_s))
        else:
            for rid in pending:
                out[rid] = Resolved(rid, reason="not in the offline mirror and the API was not consulted (use_api=False)")
    return out


def apply_to_objects(objects: Sequence[Dict[str, object]], resolved: Mapping[str, Resolved]) -> int:
    """Fill ``fit_wandb_url`` (+ ``extra.wandb``) on every object payload that names a run id; returns the count filled."""
    n = 0
    for o in objects:
        rid = o.get("fit_wandb_run_id")
        if not rid:
            continue
        r = resolved.get(str(rid))
        fc = o.setdefault("field_confidence_json", {})
        if r is None:
            fc["fit_wandb_url"] = {"confidence": "unverified", "source": "run id not resolved", "reason": "run id not resolved"}
            continue
        fc["fit_wandb_url"] = r.confidence()
        if r.url:
            o["fit_wandb_url"] = r.url
            ex = dict(o.get("extra") or {})
            ex["wandb"] = {"project": r.project, "entity": r.entity, "source": r.source,
                           "config_keys": sorted(r.config)[:64], "summary_keys": sorted(r.summary)[:64]}
            o["extra"] = ex
            n += 1
    return n


class WandbRunsAdapter:
    """Registers the mirror metadata files of the runs the registry names as artifacts; the URL fill itself
    happens inside the object / trial records (there is no partial-update record for ``object``), through
    :func:`resolve` + :func:`apply_to_objects` at read time in ``objects_bank`` / the scan."""
    name = "wandb_runs"
    covers_sources: Tuple[str, ...] = ()
    rng_scope = "none"
    allow_symlinks = False
    priority = 0                           # A resolver that emits no measurement of its own

    def __init__(self, mirror: Optional[str] = None, run_ids: Sequence[str] = ()):
        """``mirror`` default: the registry configuration's ``[wandb] mirror`` (:func:`settings`); with none the adapter declares no
        root and discovers nothing."""
        self.mirror = mirror if mirror is not None else settings().mirror
        self.roots = (self.mirror,) if self.mirror else ()
        self.run_ids = tuple(run_ids)

    def discover(self) -> List[Input]:
        out: List[Input] = []
        if not self.mirror:
            return out
        dirs = _mirror_dirs(self.mirror)
        for rid in self.run_ids:
            d = dirs.get(rid)
            if d:
                p = os.path.join(d, "files", "wandb-metadata.json")
                if os.path.isfile(p):
                    out.append(check_input(p, role="report", allow_symlink=False))
        return out

    def comparison_group(self, inp: Input) -> str:
        return os.path.dirname(os.path.dirname(inp.uri))

    def read(self, ctx: AdapterContext, group: str, inputs: Sequence[Input]) -> Iterator[RecordBatch]:
        return iter(())

    def source_states(self, ctx: AdapterContext, inputs: Sequence[Input]) -> List[SourceStateRow]:
        return []


def register() -> None:
    """The standalone pass sees this adapter too (every legacy source). Constructed bare it names no run
    id and so discovers nothing — the spec's "no wider W&B ingestion" — while the URL fill for the ids the sidecars
    and trial logs name still rides ``objects_bank`` / ``objects scan`` through :func:`resolve`."""
    from .base import register as _register
    _register(WandbRunsAdapter.name, lambda: WandbRunsAdapter())
