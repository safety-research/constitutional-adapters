"""Frozen-scenario management for the Petri Bloom audit.

The scenario set (seeds + judge dimensions, generated ONCE by `bloom
scenarios`) is the audit's shared measurement instrument: all ~24 served
legs are audited against the identical frozen set. This module owns the
freeze/verify/materialize lifecycle — it deliberately does NOT run
`bloom scenarios` (a live Claude-API action deferred to the launch
phase) and imports neither ``inspect_ai`` nor ``petri_bloom``:

- **Content hash** — :func:`bundle_content_hash` over every file under
  ``scenarios/seeds/`` + ``scenarios/dimensions/`` of every behavior
  (the §6 contract; the same value ``petri_twosi.yaml`` pins as
  ``scenario_bundle_hash`` and ``petri-scenarios.lock.yaml`` records).
- **Durable bundle (M12)** — :func:`freeze_scenarios` copies the
  generated ``scenarios/`` trees into a content-addressed, mode-0700,
  read-only bundle dir ``<bundle_root>/<content-hash>/``. The production
  root is cluster A's ``$SHARED/petri/scenarios`` (NFS, honors modes);
  :func:`validate_bundle_root` REFUSES cluster B's ``/workspace``
  (MooseFS ignores chmod — a containment store there would be silently
  world-readable).
- **Seed-config lock** — :func:`freeze_lock` records hashes into
  ``configs/capeval/petri-scenarios.lock.yaml`` (no seed text in the lock
  itself) and the M14 concurrency-cap record;
  :func:`assert_generation_allowed` refuses regeneration once the lock is
  frozen (re-running the audit never regenerates scenarios).
- **The published copy** — the bundle the paper's Petri runs used ships in
  the repository, byte for byte as frozen (for exact reproduction),
  at ``data/petri/scenarios/<content hash>/``
  (:func:`repo_bundle_dir`). A git checkout keeps no modes beyond the
  executable bit, so the copy is checked by :func:`verify_bundle_content`
  (every file against the manifest, the content hash recomputed — the M12
  mode check left out), and :func:`install_bundle` lays it into a host's
  0700 store, where :func:`verify_bundle` holds it to the full contract.
- **Read-only materialization (M12/H6)** — :func:`materialize` builds
  the run-time behavior dirs (canonical + the ``PROMPTED_SAFETY`` /
  ``PROMPTED_CONSTITUTION`` SI-prefix variants) from the ONE bundle:
  BEHAVIOR.md from the version-controlled config dir, ``scenarios/``
  hash-verified from the bundle, write bits stripped. Variant dirs hold
  no scenario bytes of their own, so the variant↔canonical scenario-id
  bijection holds by construction.

Every failure is a loud :class:`ScenarioError` refusal, never a silent
default (repo rule).
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import yaml

from sisteer.capeval import manifest as cm


class ScenarioError(ValueError):
    """A frozen-scenario refusal (hash mismatch, regeneration attempt,
    forbidden bundle root, tampered materialization, …)."""


# --- Layout constants --------------------------------------------------

#: the two subdirs of ``scenarios/`` that the content hash covers (§6).
#: (``understanding.md`` and any other file still travel in the bundle and
#: are pinned by the bundle manifest's full file map — nothing can drift —
#: but the KEY hash is over seeds + dimensions, the plan's stated inputs.)
HASHED_SUBDIRS = ("seeds", "dimensions")

#: SI-prefix variant behavior dirs (H8): share the ONE frozen bundle, add
#: only a verified ``target_sysprompt_prefix``. si_id -> dir name.
SI_VARIANT_DIRS = {"safety": "PROMPTED_SAFETY",
                   "constitution": "PROMPTED_CONSTITUTION",
                   "irrelevant": "PROMPTED_IRRELEVANT",
                   "constitution_full": "PROMPTED_CONSTITUTION_FULL"}
# PROMPTABLE SIs only: the object-only pseudo-instructions
# (cm.PETRI_OBJECT_ONLY_SI_IDS) have no prompted baseline, hence no
# variant dir (sdf-base-steering, 2026-09-04).
VARIANT_DIRS = tuple(SI_VARIANT_DIRS[si] for si in cm.PETRI_PROMPTABLE_SI_IDS)

#: per-behavior modality: tool-use self-preservation guarantees the
#: agent path; the other five behaviors are conversation.
BEHAVIOR_MODALITIES = {
    b: ("agent" if b == "self_preservation" else "conversation")
    for b in cm.PETRI_BEHAVIORS
}

BUNDLE_MANIFEST_NAME = "bundle_manifest.json"
LOCK_STATUS_UNFROZEN = "UNFROZEN"
LOCK_STATUS_FROZEN = "frozen"

#: cluster B's MooseFS shared root: chmod is silently ignored there, so a
#: "0700" bundle would be world-readable. NEVER a bundle root (M12).
_FORBIDDEN_ROOT_PREFIX = "/workspace/"

#: where the repository carries a frozen bundle: ``<repo>/data/petri/scenarios/<content hash>/`` (the same
#: ``petri/scenarios/<hash>`` tail as the cluster store the lock names and the grading host's store)
REPO_BUNDLE_ROOT_REL = Path("data") / "petri" / "scenarios"


def _repo_root() -> Path:
    # src/sisteer/capeval/petri/scenarios.py -> the repo root is parents[4]
    return Path(__file__).resolve().parents[4]


def repo_bundle_dir(content_hash: str, repo_root=None) -> Path:
    """``<repo>/data/petri/scenarios/<content_hash>``: where a checkout carries the frozen bundle (it may not exist; the caller
    checks)."""
    if not isinstance(content_hash, str) or not content_hash or "/" in content_hash or content_hash.startswith("."):
        _refuse(f"repo_bundle_dir: {content_hash!r} is not a bundle content hash")
    return Path(repo_root if repo_root is not None else _repo_root()) / REPO_BUNDLE_ROOT_REL / content_hash


def _durable_root(cluster: Optional[str] = None) -> Path:
    """The durable production root: the cluster's shared root (``$SHARED``, resolved through the site config by
    :mod:`sisteer.site`: the job env's SISTEER_SHARED_ROOT, else the named or primary cluster's CLUSTER_SHARED_ROOT).
    Refused by name when the site names none."""
    from sisteer import site
    try:
        return site.expand_path("$SHARED", cluster)
    except (site.PlaceholderError, site.SiteConfigError) as exc:
        _refuse(f"no durable bundle root: {exc}")


def _refuse(msg: str) -> None:
    raise ScenarioError(msg)


# --- Bundle roots -------------------------------------------------------------


def default_bundle_root(cluster: Optional[str] = None) -> Path:
    """The durable production bundle root: cluster A's
    ``$SHARED/petri/scenarios`` (NFS, honors mode 0700 — M12). This is
    the CLUSTER path recorded in the lock; the grading host's working
    copy is an ephemeral hash-verified materialization."""
    return _durable_root(cluster) / "petri" / "scenarios"


def validate_bundle_root(root) -> Path:
    """Refuse bundle roots that cannot hold a contained 0700 store:
    relative paths, and anything under cluster B's ``/workspace/``
    (MooseFS ignores chmod; every file there is world-readable — M12).
    Non-cluster absolute paths (e.g. a tmpdir in tests) are allowed; the
    PRODUCTION root is :func:`default_bundle_root`."""
    p = Path(root)
    if not p.is_absolute():
        _refuse(f"bundle root must be an absolute path, got {root!r}")
    posix = p.as_posix()
    if posix == _FORBIDDEN_ROOT_PREFIX.rstrip("/") or posix.startswith(_FORBIDDEN_ROOT_PREFIX):
        _refuse(
            f"bundle root {posix} is on cluster B /workspace (MooseFS): "
            "chmod is silently ignored there, so the 0700 containment "
            "store would be world-readable — use the shared root of a "
            "cluster whose storage honours modes ($SHARED)")
    return p


# --- Hashing ------------------------------------------------------------------


def _walk_files(root: Path) -> Dict[str, Path]:
    """{relative posix path: absolute path} for every regular file under
    ``root``, sorted. Symlinks anywhere are a refusal (a symlink can
    smuggle mutable content past the content hash)."""
    out: Dict[str, Path] = {}
    if not root.is_dir():
        _refuse(f"not a directory: {root}")
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        for name in list(dirnames) + list(filenames):
            if (dp / name).is_symlink():
                _refuse(f"symlink inside a scenario tree: {dp / name} — "
                        "refusing (content must be plain files)")
        for name in filenames:
            p = dp / name
            out[p.relative_to(root).as_posix()] = p
    return dict(sorted(out.items()))


def behavior_file_map(scenarios_dir) -> Dict[str, str]:
    """{relpath: sha256_hex16} over one behavior's HASHED subdirs
    (``scenarios/seeds/`` + ``scenarios/dimensions/``). A missing or
    empty subdir is a refusal — the frozen set must actually exist."""
    scenarios_dir = Path(scenarios_dir)
    out: Dict[str, str] = {}
    for sub in HASHED_SUBDIRS:
        subdir = scenarios_dir / sub
        if not subdir.is_dir():
            _refuse(f"{scenarios_dir}: missing scenarios/{sub}/ — not a "
                    "generated scenario set")
        files = _walk_files(subdir)
        if not files:
            _refuse(f"{scenarios_dir}: scenarios/{sub}/ is empty — not a "
                    "generated scenario set")
        for rel, path in files.items():
            out[f"{sub}/{rel}"] = cm.sha256_hex16(path.read_bytes())
    return out


def behavior_scenarios_hash(scenarios_dir) -> str:
    """16-hex content hash of one behavior's frozen seeds + dimensions."""
    return cm.sha256_hex16(
        cm.canonical_json(behavior_file_map(scenarios_dir)).encode("utf-8"))


def bundle_content_hash(per_behavior_file_maps: Mapping[str, Mapping[str, str]]) -> str:
    """The BUNDLE content hash (the ``scenario_bundle_hash`` every run
    keys on): 16-hex sha256 over canonical_json of
    ``{behavior: {relpath: file_hash}}``."""
    if not per_behavior_file_maps:
        _refuse("bundle_content_hash: no behaviors")
    return cm.sha256_hex16(cm.canonical_json(
        {b: dict(m) for b, m in per_behavior_file_maps.items()}
    ).encode("utf-8"))


def scenario_ids(scenarios_dir) -> Tuple[str, ...]:
    """The canonical scenario ids of one behavior's frozen set: the
    sorted stems of ``scenarios/seeds/*`` (Bloom uses the scenario name
    as the seed filename). Duplicate stems refuse."""
    seeds = Path(scenarios_dir) / "seeds"
    if not seeds.is_dir():
        _refuse(f"{scenarios_dir}: no seeds/ dir")
    stems = sorted(Path(rel).stem for rel in _walk_files(seeds))
    if len(stems) != len(set(stems)):
        dups = sorted({s for s in stems if stems.count(s) > 1})
        _refuse(f"{scenarios_dir}: duplicate scenario ids {dups} in seeds/")
    if not stems:
        _refuse(f"{scenarios_dir}: seeds/ is empty")
    return tuple(stems)


# --- Freeze -------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenBundle:
    """One frozen, content-addressed scenario bundle."""

    bundle_dir: Path
    content_hash: str
    behavior_hashes: Dict[str, str]
    #: {behavior: {relpath under scenarios/: sha256_hex16}} — HASHED
    #: subdirs only (the content-hash inputs).
    file_maps: Dict[str, Dict[str, str]]
    #: {behavior: {"scenarios/<relpath>": sha256_hex16}} — the manifest's
    #: FULL per-behavior file map (understanding.md and every other bundled
    #: file included). Runtime materializations are verified against THIS,
    #: not just the hash inputs, so no un-hashed bundled file can drift
    #: (runner review 2026-08-18: editing a materialized understanding.md
    #: was previously accepted because only seeds/dimensions were compared).
    full_file_maps: Dict[str, Dict[str, str]]


def _chmod_readonly_tree(root: Path) -> None:
    """Strip write bits: files 0400, dirs 0500 (root itself stays 0700 so
    the owner can later remove/inspect the tree)."""
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        for name in filenames:
            os.chmod(Path(dirpath) / name, 0o400)
        for name in dirnames:
            os.chmod(Path(dirpath) / name, 0o500)
    os.chmod(root, 0o700)


def _verify_tree_modes(root: Path, *, what: str) -> None:
    """Permission drift invalidates a frozen tree (runner review
    2026-08-18): the root must be EXACTLY 0700 (the M12 containment mode)
    and every contained directory and file must carry no write bits and no
    group/world access — a chmod that leaves every hash intact still voids
    the read-only/0700 containment guarantee, so it is a refusal, never a
    pass."""
    mode = stat.S_IMODE(os.stat(root).st_mode)
    if mode != 0o700:
        _refuse(f"{root}: {what} root mode {oct(mode)} != 0700 — "
                "permission drift on the containment root (M12); refusing")
    for dirpath, dirnames, filenames in os.walk(root):
        for name in list(dirnames) + list(filenames):
            p = Path(dirpath) / name
            m = stat.S_IMODE(os.stat(p).st_mode)
            if m & 0o222:
                _refuse(f"{p}: writable (mode {oct(m)}) inside a read-only "
                        f"{what} — permission drift (M12); refusing")
            if m & 0o077:
                _refuse(f"{p}: group/world access (mode {oct(m)}) inside a "
                        f"contained {what} — permission drift (M12); "
                        "refusing")


def _copy_tree(src: Path, dst: Path) -> Dict[str, str]:
    """Copy every regular file under ``src`` to ``dst`` (refusing
    symlinks) and return {relpath: sha256_hex16} of the COPIES."""
    out: Dict[str, str] = {}
    for rel, path in _walk_files(src).items():
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        out[rel] = cm.sha256_hex16(target.read_bytes())
    return out


def freeze_scenarios(source_root, bundle_root, *,
                     behaviors: Sequence[str]) -> FrozenBundle:
    """Freeze the one generated scenario set into the durable bundle.

    ``source_root/<behavior>/scenarios/`` (the post-``bloom scenarios``
    behavior dirs) is copied to ``<bundle_root>/<content-hash>/
    <behavior>/scenarios/``, mode-0700 at the top, every file read-only,
    with a full-file-map ``bundle_manifest.json`` so later verification
    catches ANY drift (including files outside the hashed subdirs).

    Idempotent by content-addressing: if the bundle dir for this content
    already exists it is VERIFIED and returned — never rewritten, never
    regenerated."""
    source_root = Path(source_root)
    bundle_root = validate_bundle_root(bundle_root)
    if not behaviors:
        _refuse("freeze_scenarios: behaviors must be non-empty")
    unknown = sorted(set(behaviors) - set(cm.PETRI_BEHAVIORS))
    if unknown:
        _refuse(f"freeze_scenarios: unknown behavior(s) {unknown}")

    file_maps = {b: behavior_file_map(source_root / b / "scenarios")
                 for b in behaviors}
    content_hash = bundle_content_hash(file_maps)
    bundle_dir = bundle_root / content_hash

    if bundle_dir.exists():
        # already frozen — verify, never rewrite (no-regeneration).
        return verify_bundle(bundle_dir, expected_hash=content_hash)

    tmp = bundle_root / f".{content_hash}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    os.chmod(tmp, 0o700)
    full_maps: Dict[str, Dict[str, str]] = {}
    for b in behaviors:
        full_maps[b] = _copy_tree(source_root / b / "scenarios",
                                  tmp / b / "scenarios")
    # re-derive the hash FROM THE COPY — a copy error must not freeze
    copied_maps = {b: behavior_file_map(tmp / b / "scenarios")
                   for b in behaviors}
    copied_hash = bundle_content_hash(copied_maps)
    if copied_hash != content_hash:
        _refuse(f"freeze_scenarios: copied bundle hashes to {copied_hash} "
                f"but the source hashed to {content_hash} — copy corrupted; "
                "refusing to freeze")
    manifest = {
        "version": 1,
        "content_hash": content_hash,
        "behaviors": {b: cm.sha256_hex16(
            cm.canonical_json(copied_maps[b]).encode("utf-8"))
            for b in behaviors},
        # FULL file map (understanding.md included) — tamper detection
        # over everything in the bundle, not just the hash inputs.
        "files": {b: {f"scenarios/{rel}": h for rel, h in full_maps[b].items()}
                  for b in behaviors},
        "hashed_subdirs": list(HASHED_SUBDIRS),
        "frozen_at": time.time(),
    }
    (tmp / BUNDLE_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    _chmod_readonly_tree(tmp)
    os.replace(tmp, bundle_dir)
    return FrozenBundle(bundle_dir=bundle_dir, content_hash=content_hash,
                        behavior_hashes=dict(manifest["behaviors"]),
                        file_maps={b: dict(m) for b, m in copied_maps.items()},
                        full_file_maps={b: dict(m)
                                        for b, m in manifest["files"].items()})


def verify_bundle(bundle_dir, *, expected_hash: Optional[str] = None) -> FrozenBundle:
    """Verify a frozen bundle end-to-end and return it. Refusals: missing
    or unparseable bundle manifest; the dir name not matching the
    recorded content hash (content-addressing); any file added, removed,
    or modified relative to the FULL recorded file map; a recomputed
    content hash differing from the recorded one; ``expected_hash``
    (when given) differing from the recomputed hash; and PERMISSION DRIFT
    anywhere in the tree (root not exactly 0700, or any contained dir/file
    writable or group/world-accessible — runner review 2026-08-18: a bare
    chmod leaves every hash intact but voids the M12 containment, so it is
    a refusal too)."""
    return _verify_bundle(bundle_dir, expected_hash=expected_hash, contained=True)


def verify_bundle_content(bundle_dir, *, expected_hash: str) -> FrozenBundle:
    """Every check of :func:`verify_bundle` but the M12 mode check, for the published copy a checkout carries
    (:func:`repo_bundle_dir`): git records no mode beyond the executable bit, so a clone's files read 0644 and say nothing about
    containment, while their BYTES are still the frozen bundle's. The manifest, the dir name, the full file map and the recomputed content
    hash all hold, and ``expected_hash`` is required (a published copy is only ever read as the bundle a lock or a manifest pins). The
    run-time materialization made from it is held to the full contract (:func:`materialize`, :func:`install_bundle`)."""
    if not isinstance(expected_hash, str) or not expected_hash:
        _refuse("verify_bundle_content: expected_hash is required — a published copy is read only as the bundle a lock pins")
    return _verify_bundle(bundle_dir, expected_hash=expected_hash, contained=False)


def _verify_bundle(bundle_dir, *, expected_hash: Optional[str], contained: bool) -> FrozenBundle:
    bundle_dir = Path(bundle_dir)
    mpath = bundle_dir / BUNDLE_MANIFEST_NAME
    if not mpath.is_file():
        _refuse(f"{bundle_dir}: no {BUNDLE_MANIFEST_NAME} — not a frozen "
                "scenario bundle")
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        _refuse(f"{mpath}: unparseable bundle manifest: {e}")
    recorded_hash = manifest.get("content_hash")
    behaviors = sorted(manifest.get("files", {}))
    if not behaviors:
        _refuse(f"{mpath}: bundle manifest records no behaviors")
    if bundle_dir.name != recorded_hash:
        _refuse(f"{bundle_dir}: dir name != recorded content hash "
                f"{recorded_hash!r} — the bundle store is content-addressed; "
                "refusing")

    # FULL file-map verification (tamper detection over every file).
    for b in behaviors:
        recorded_files = manifest["files"][b]
        on_disk = {f"scenarios/{rel}": cm.sha256_hex16(p.read_bytes())
                   for rel, p in _walk_files(bundle_dir / b / "scenarios").items()}
        if on_disk != recorded_files:
            extra = sorted(set(on_disk) - set(recorded_files))
            missing = sorted(set(recorded_files) - set(on_disk))
            changed = sorted(k for k in set(on_disk) & set(recorded_files)
                             if on_disk[k] != recorded_files[k])
            _refuse(f"{bundle_dir}/{b}: bundle drift — extra {extra}, "
                    f"missing {missing}, modified {changed} (the frozen "
                    "bundle is immutable; refusing)")

    file_maps = {b: behavior_file_map(bundle_dir / b / "scenarios")
                 for b in behaviors}
    content_hash = bundle_content_hash(file_maps)
    if content_hash != recorded_hash:
        _refuse(f"{bundle_dir}: recomputed content hash {content_hash} != "
                f"recorded {recorded_hash} — refusing")
    if expected_hash is not None and content_hash != expected_hash:
        _refuse(f"{bundle_dir}: content hash {content_hash} != expected "
                f"{expected_hash} — the audit is pinned to a different "
                "frozen set (acceptance 8); refusing")
    if contained:
        _verify_tree_modes(bundle_dir, what="frozen bundle")
    return FrozenBundle(
        bundle_dir=bundle_dir, content_hash=content_hash,
        behavior_hashes={b: cm.sha256_hex16(
            cm.canonical_json(file_maps[b]).encode("utf-8"))
            for b in behaviors},
        file_maps={b: dict(m) for b, m in file_maps.items()},
        full_file_maps={b: dict(manifest["files"][b]) for b in behaviors})


def install_bundle(source_dir, bundle_root, *, expected_hash: str) -> Tuple[FrozenBundle, bool]:
    """Lay the published copy of a frozen bundle (``source_dir``, normally :func:`repo_bundle_dir`) into a host's store as
    ``<bundle_root>/<expected_hash>/``: mode 0700 at the top, every file read-only, the bytes those of the source, the bundle manifest
    included, so the store is the bundle as it was frozen. Returns ``(bundle, installed)``: ``installed`` False when the store already
    held it.

    The source is checked first (:func:`verify_bundle_content`), and only the manifest and the files its map names are copied. The
    result is held to the full contract (:func:`verify_bundle`, modes included) before it is returned. An existing store is verified,
    never rewritten: a store that does not verify is a refusal to clear by hand, never a repair in place. ``bundle_root`` passes
    :func:`validate_bundle_root` (never cluster B's ``/workspace``)."""
    src = verify_bundle_content(source_dir, expected_hash=expected_hash)
    bundle_root = validate_bundle_root(bundle_root)
    dest = bundle_root / expected_hash
    if dest.exists():
        return verify_bundle(dest, expected_hash=expected_hash), False
    bundle_root.mkdir(parents=True, exist_ok=True)
    tmp = bundle_root / f".{expected_hash}.tmp-install-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    os.chmod(tmp, 0o700)
    try:
        rels = [BUNDLE_MANIFEST_NAME] + [f"{b}/{rel}" for b in sorted(src.full_file_maps) for rel in src.full_file_maps[b]]
        for rel in rels:
            target = tmp / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src.bundle_dir / rel, target)
        _chmod_readonly_tree(tmp)
        try:
            os.replace(tmp, dest)
        except OSError:
            if not dest.exists():
                raise
            # another install finished first: its store is the one verified below, this copy goes
            _remove_tree(tmp)
            return verify_bundle(dest, expected_hash=expected_hash), False
    except BaseException:
        if tmp.exists():
            _remove_tree(tmp)
        raise
    return verify_bundle(dest, expected_hash=expected_hash), True


def _remove_tree(root: Path) -> None:
    """rmtree a read-only tree of our own making (write bits restored first)."""
    for dirpath, dirnames, filenames in os.walk(root):
        os.chmod(dirpath, 0o700)
    shutil.rmtree(root)


# --- The seed-config lock (hashes only) ----------------------------------------


def read_lock(lock_path) -> Dict[str, Any]:
    """Load + minimally validate ``petri-scenarios.lock.yaml``."""
    lock_path = Path(lock_path)
    if not lock_path.is_file():
        _refuse(f"lock file missing: {lock_path} — the seed-config lock is "
                "a committed deliverable (plan §6); refusing to proceed "
                "without it")
    raw = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        _refuse(f"{lock_path}: lock document is not a mapping")
    status = raw.get("status")
    if status not in (LOCK_STATUS_UNFROZEN, LOCK_STATUS_FROZEN):
        _refuse(f"{lock_path}: unknown status {status!r} (expected "
                f"{LOCK_STATUS_UNFROZEN!r} or {LOCK_STATUS_FROZEN!r})")
    for key in ("generation", "bundle", "behaviors"):
        if not isinstance(raw.get(key), Mapping):
            _refuse(f"{lock_path}: missing/malformed {key!r} block")
    return dict(raw)


def assert_generation_allowed(lock_path) -> None:
    """Refuse scenario (re)generation once the lock is frozen: the audit
    path never regenerates a frozen set (`bloom scenarios`
    without --overwrite refuses on existing output, and the audit path
    never passes --overwrite — this guard is the belt to that suspender)."""
    lock = read_lock(lock_path)
    if lock["status"] == LOCK_STATUS_FROZEN:
        _refuse(
            f"{lock_path}: the scenario set is FROZEN (content hash "
            f"{lock['bundle'].get('content_hash')!r}) — re-running the "
            "audit never regenerates scenarios (acceptance 8). A new set "
            "is a deliberate new freeze: bump the lock version and record "
            "why, never --overwrite in the audit path.")


def freeze_lock(lock_path, bundle: FrozenBundle, *,
                actual_max_connections: int,
                ran_at: str,
                coordination_posted: bool = False,
                durable_root: Optional[str] = None) -> Dict[str, Any]:
    """Record the one generation pass into the lock — HASHES ONLY, no
    seed text (M12/M14). Refusals: a bundle OUTSIDE the durable root (a
    production freeze may only record a bundle under the cluster's shared root —
    runner review 2026-08-18: a ``/tmp`` bundle could be marked frozen and
    then vanish on reboot, leaving an unreproducible audit); a lock
    already frozen on a DIFFERENT bundle (idempotent on the same one); a
    behavior-set mismatch between lock and bundle;
    ``actual_max_connections`` above the recorded cap without
    ``coordination_posted=True`` (M14: exceeding the cap requires the
    coordination-channel post FIRST).

    ``durable_root`` is the EXPLICIT test-only seam: production callers
    leave it None (-> the cluster's shared storage); tests inject their
    tmp root. Never point it anywhere ephemeral for a real production
    record."""
    lock_path = Path(lock_path)
    lock = read_lock(lock_path)
    required_root = Path(durable_root) if durable_root is not None \
        else _durable_root()
    bundle_dir = Path(bundle.bundle_dir)
    try:
        bundle_dir.relative_to(required_root)
    except ValueError:
        _refuse(
            f"freeze_lock: bundle {bundle_dir} is outside the durable root "
            f"{required_root} — a production freeze records only "
            "bundles under the cluster's shared root; an ephemeral bundle marked "
            "frozen would leave the audit unreproducible after cleanup. "
            "Tests inject their own root via the explicit durable_root "
            "seam; production never overrides it.")
    if lock["status"] == LOCK_STATUS_FROZEN:
        if lock["bundle"].get("content_hash") == bundle.content_hash:
            return lock  # idempotent re-freeze of the same content
        _refuse(f"{lock_path}: already frozen on content hash "
                f"{lock['bundle'].get('content_hash')!r}, refusing to "
                f"re-freeze onto {bundle.content_hash!r} — a new set is a "
                "deliberate new lock version, never an overwrite")
    lock_behaviors = set(lock["behaviors"])
    bundle_behaviors = set(bundle.behavior_hashes)
    if lock_behaviors != bundle_behaviors:
        _refuse(f"{lock_path}: lock behaviors {sorted(lock_behaviors)} != "
                f"bundle behaviors {sorted(bundle_behaviors)} — the freeze "
                "must cover exactly the locked behavior set")
    cap = lock["generation"].get("max_connections_cap")
    if not isinstance(cap, int) or cap < 1:
        _refuse(f"{lock_path}: generation.max_connections_cap must be a "
                f"positive int, got {cap!r}")
    if not isinstance(actual_max_connections, int) or actual_max_connections < 1:
        _refuse(f"freeze_lock: actual_max_connections must be a positive "
                f"int, got {actual_max_connections!r}")
    if actual_max_connections > cap and not coordination_posted:
        _refuse(
            f"freeze_lock: actual_max_connections={actual_max_connections} "
            f"exceeds the recorded cap {cap} — exceeding the cap requires "
            "the coordination-channel post FIRST (M14); "
            "pass coordination_posted=True only after it is out")
    if not isinstance(ran_at, str) or not ran_at:
        _refuse("freeze_lock: ran_at must be a non-empty ISO-8601 string")

    updated = dict(lock)
    updated["status"] = LOCK_STATUS_FROZEN
    updated["generation"] = {
        **dict(lock["generation"]),
        "actual_max_connections": actual_max_connections,
        "coordination_posted": bool(coordination_posted),
        "ran_at": ran_at,
    }
    updated["bundle"] = {
        **dict(lock["bundle"]),
        "location": str(bundle.bundle_dir),
        "content_hash": bundle.content_hash,
    }
    updated["behaviors"] = dict(bundle.behavior_hashes)

    header = (
        "# Petri Bloom frozen-scenario seed-config manifest (petri plan §6;\n"
        "# M12/M14). This lock holds hashes only; FROZEN by the one\n"
        "# `bloom scenarios` pass, and the runner verifies the recorded content\n"
        "# hash and refuses on mismatch (acceptance 8). `location` is the\n"
        "# durable 0700 store; a bundle published with the repository sits at\n"
        "# data/petri/scenarios/<content_hash>/, byte for byte as frozen.\n")
    tmp = lock_path.with_name(lock_path.name + ".tmp")
    tmp.write_text(header + yaml.safe_dump(updated, sort_keys=True,
                                           allow_unicode=True),
                   encoding="utf-8")
    os.replace(tmp, lock_path)
    return updated


# --- BEHAVIOR.md handling -------------------------------------------------------


def parse_behavior_md(data) -> Tuple[Dict[str, Any], str]:
    """Parse a BEHAVIOR.md into (frontmatter dict, description body).
    Strict: the file must open with a ``---`` line, close the frontmatter
    with a bare ``---`` line, and the frontmatter must be a YAML mapping.
    (Block-scalar frontmatter values are indented, so a bare ``---`` can
    only be the terminator — the authored SI-prefix variants keep that
    invariant and the tests pin it.)"""
    text = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)
    if not text.startswith("---\n"):
        _refuse("BEHAVIOR.md must open with a '---' frontmatter line")
    end = text.find("\n---\n", 3)
    if end < 0:
        _refuse("BEHAVIOR.md frontmatter is unterminated (no closing '---')")
    front = yaml.safe_load(text[4:end + 1])
    if not isinstance(front, Mapping):
        _refuse("BEHAVIOR.md frontmatter is not a YAML mapping")
    body = text[end + len("\n---\n"):].strip()
    return dict(front), body


def behavior_files(config_root, behaviors: Sequence[str], *,
                   variants: Sequence[str] = VARIANT_DIRS) -> Dict[str, bytes]:
    """The ``{dir_key: BEHAVIOR.md bytes}`` mapping that
    ``manifest.petri_harness_config_hash`` hashes: the
    canonical ``<behavior>/BEHAVIOR.md`` plus every SI-prefix variant
    ``<behavior>/<VARIANT>/BEHAVIOR.md``, all read from the
    version-controlled config root. Every file must exist — a missing
    behavior definition is a refusal, never a silently narrower hash."""
    config_root = Path(config_root)
    if not behaviors:
        _refuse("behavior_files: behaviors must be non-empty")
    out: Dict[str, bytes] = {}
    for b in behaviors:
        if b not in cm.PETRI_BEHAVIORS:
            _refuse(f"behavior_files: unknown behavior {b!r}")
        keys = [(b, config_root / b / "BEHAVIOR.md")]
        keys += [(f"{b}/{v}", config_root / b / v / "BEHAVIOR.md")
                 for v in variants]
        for key, path in keys:
            if not path.is_file():
                _refuse(f"behavior_files: missing {path} — every behavior "
                        "definition in play is a harness-hash input (H10)")
            data = path.read_bytes()
            if not data:
                _refuse(f"behavior_files: {path} is empty")
            out[key] = data
    return out


# --- Read-only materialization ---------------------------------------------------


def materialize(bundle_dir, dest_root, *,
                expected_bundle_hash: str,
                config_root,
                behaviors: Optional[Sequence[str]] = None,
                variants: Sequence[str] = VARIANT_DIRS) -> Dict[str, Path]:
    """Materialize the run-time behavior dirs from the ONE frozen bundle
    (M12): for every behavior, the canonical dir and each SI-prefix
    variant dir get BEHAVIOR.md from ``config_root`` and a hash-verified
    copy of the bundle's ``scenarios/`` tree — READ-ONLY (write bits
    stripped), under a mode-0700 ``dest_root``. Returns
    ``{dir_key: behavior dir}`` with the same dir keys the harness hash
    uses (``<behavior>`` / ``<behavior>/<VARIANT>``).

    Idempotent: an existing, verifying materialization is returned as-is
    (no rewrite). ANY divergence — a hash mismatch against
    ``expected_bundle_hash``, tampered dest content (ANY file of the full
    frozen tree, ``understanding.md`` included), an extra or missing file,
    a partial previous materialization, or permission drift anywhere in
    the tree — is a loud refusal, never a repair-in-place (runner review
    2026-08-18)."""
    bundle = verify_bundle(bundle_dir, expected_hash=expected_bundle_hash)
    behaviors = list(behaviors) if behaviors is not None else sorted(bundle.behavior_hashes)
    missing_b = sorted(set(behaviors) - set(bundle.behavior_hashes))
    if missing_b:
        _refuse(f"materialize: behavior(s) {missing_b} not in bundle "
                f"{bundle.bundle_dir} (has {sorted(bundle.behavior_hashes)})")
    bf = behavior_files(config_root, behaviors, variants=variants)

    dest_root = Path(dest_root)
    dir_keys = [k for b in behaviors
                for k in ([b] + [f"{b}/{v}" for v in variants])]

    if dest_root.exists():
        _verify_materialized_tree(dest_root, dir_keys, bf, bundle)
        return {key: dest_root / key for key in dir_keys}

    tmp = dest_root.with_name(dest_root.name + ".tmp-materialize")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    os.chmod(tmp, 0o700)
    for key in dir_keys:
        b = key.split("/", 1)[0]
        dest = tmp / key
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "BEHAVIOR.md").write_bytes(bf[key])
        copied = _copy_tree(bundle.bundle_dir / b / "scenarios",
                            dest / "scenarios")
        # the FULL copied tree verified against the bundle manifest's
        # per-behavior file map — understanding.md included, never just
        # the hash-input subdirs (M12; runner review 2026-08-18)
        got_full = {f"scenarios/{rel}": h for rel, h in copied.items()}
        if got_full != bundle.full_file_maps[b]:
            _refuse(f"materialize: copied scenario tree for {key!r} does "
                    "not match the bundle's full file map — copy corrupted; "
                    "refusing")
    _chmod_readonly_tree(tmp)
    os.replace(tmp, dest_root)
    return {key: dest_root / key for key in dir_keys}


def _verify_materialized_tree(dest_root: Path, dir_keys: Sequence[str],
                              bf: Mapping[str, bytes],
                              bundle: FrozenBundle) -> None:
    """An existing materialization must match EXACTLY, over the ENTIRE
    tree (runner review 2026-08-18): every dir key's BEHAVIOR.md bytes,
    the FULL frozen scenario tree per behavior (``understanding.md`` and
    every other bundled file, not just the hashed seeds/dimensions), no
    extra and no missing files anywhere under ``dest_root``, and no
    permission drift (root exactly 0700; every contained dir/file
    read-only with no group/world access)."""
    expected: Dict[str, str] = {}
    for key in dir_keys:
        b = key.split("/", 1)[0]
        expected[f"{key}/BEHAVIOR.md"] = cm.sha256_hex16(bytes(bf[key]))
        for rel, h in bundle.full_file_maps[b].items():
            expected[f"{key}/{rel}"] = h
    on_disk = {rel: cm.sha256_hex16(p.read_bytes())
               for rel, p in _walk_files(dest_root).items()}
    missing = sorted(set(expected) - set(on_disk))
    if missing:
        _refuse(f"materialize: {len(missing)} path(s) missing from an "
                f"existing materialization under {dest_root} (first: "
                f"{missing[:5]}) — partial/foreign dest; remove the "
                "ephemeral working copy and re-materialize")
    extra = sorted(set(on_disk) - set(expected))
    if extra:
        _refuse(f"materialize: unexpected file(s) {extra[:5]} inside the "
                f"existing materialization {dest_root} — stale/tampered/"
                "foreign materialization (M12); remove the working copy "
                "and re-materialize")
    changed = sorted(k for k in expected if on_disk[k] != expected[k])
    if changed:
        _refuse(f"materialize: {changed[:5]} differ from the frozen bundle "
                f"{bundle.content_hash} / the version-controlled behavior "
                "definitions — stale/tampered materialization; refusing")
    _verify_tree_modes(dest_root, what="materialization")


# --- Command line: the published copy ----------------------------------------------------


def default_lock_path() -> Path:
    """``configs/capeval/petri-scenarios.lock.yaml`` of the checkout this module belongs to."""
    return _repo_root() / "configs" / "capeval" / "petri-scenarios.lock.yaml"


def frozen_lock_hash(lock_path=None) -> str:
    """The content hash a FROZEN lock pins (refused by name for an unfrozen or malformed lock)."""
    lock_path = Path(lock_path) if lock_path is not None else default_lock_path()
    lock = read_lock(lock_path)
    h = lock["bundle"].get("content_hash")
    if lock["status"] != LOCK_STATUS_FROZEN or not isinstance(h, str) or not h:
        _refuse(f"{lock_path}: not a frozen lock with a bundle.content_hash (status {lock['status']!r})")
    return h


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``verify`` checks the published copy against the lock (content, no modes); ``install`` lays it into a host's 0700 store
    (``--dest <bundle root>/<content hash>``), or verifies a store already there. Prints paths, hashes and counts, never file text.
    Exit 0 done, 2 refused."""
    import argparse
    import sys
    ap = argparse.ArgumentParser(prog="python -m sisteer.capeval.petri.scenarios",
                                 description="the frozen Petri scenario bundle the repository publishes (data/petri/scenarios/<hash>)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="check a published copy against the lock's content hash (every file, the hash recomputed; no mode check)")
    v.add_argument("--dir", default=None, help="the copy (default: data/petri/scenarios/<the lock's content hash> in this checkout)")
    v.add_argument("--lock", default=None, help="the lock (default: configs/capeval/petri-scenarios.lock.yaml in this checkout)")
    i = sub.add_parser("install", help="install the published copy into a 0700 store, or verify the store already there")
    i.add_argument("--dest", required=True, help="the store's bundle dir, <bundle root>/<the lock's content hash>")
    i.add_argument("--from", dest="source", default=None, help="the published copy (default: this checkout's)")
    i.add_argument("--lock", default=None, help="the lock (default: this checkout's)")
    a = ap.parse_args(argv)
    try:
        h = frozen_lock_hash(a.lock)
        if a.cmd == "verify":
            b = verify_bundle_content(a.dir or repo_bundle_dir(h), expected_hash=h)
            n = sum(len(m) for m in b.full_file_maps.values())
            print(f"verified {b.bundle_dir}: bundle {h}, {len(b.full_file_maps)} behaviours, {n} scenario files (content checked; modes not)")
            return 0
        dest = Path(os.path.expanduser(a.dest))
        if dest.name != h:
            _refuse(f"install: --dest {dest} must be named for the lock's content hash {h} (the store is content-addressed)")
        b, installed = install_bundle(a.source or repo_bundle_dir(h), dest.parent, expected_hash=h)
        n = sum(len(m) for m in b.full_file_maps.values())
        print(f"{'installed' if installed else 'verified the store already at'} {b.bundle_dir}: bundle {h}, "
              f"{len(b.full_file_maps)} behaviours, {n} scenario files, mode 0700")
        return 0
    except ScenarioError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
