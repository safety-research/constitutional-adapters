"""Manifest schema, validation, and hashing.

Stage-0 FROZEN interfaces. Bodies are ``NotImplementedError`` until the
implementation lands; the dataclasses and signatures are the binding
contract, pinned by ``tests/test_capeval_protocol.py``.

Validation stance: loaded from YAML and validated LOUDLY —
unknown keys, missing revisions, and absent artifacts are refusals, not
warnings. Mode/target contract (F20): ``mode: sweep`` REQUIRES ``target``;
``mode: control`` and ``mode: calibration`` FORBID it. ``confirmatory:
false`` is the v0 default AND the only value the v0 loader accepts.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields as dc_fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import yaml


class ManifestError(ValueError):
    """A manifest refusal (loud, never a warning)."""

# --- Schema constants (frozen) ---------------------------------------------

MODES = ("sweep", "control", "calibration")  # F20

# --- Petri serving identity (ADDITIVE) --------------------------------------
# Proxy-served Petri rows must be unmixable with in-process capability rows,
# so every Petri row's `engine` column carries a structured serving identity
# "<backend>+<path>+<mode>". The in-process capability values ("hf", "vllm")
# are unchanged; the identity strings widen the ENGINES enum (a deliberate,
# recorded widening of an Evals-adopted enum; the protocol-test pin moves
# with it). Helpers: serving_identity() / parse_serving_identity() /
# is_serving_identity() further down.
SERVING_BACKENDS = ("proxy_vllm", "hf_shim")
SERVING_MODES = ("interleaved", "exact_replay")  # the determinism mode
# valid paths per backend: the routing proxy serves lens-steered, native
# multi-LoRA, and passthrough legs; hf_shim exists ONLY as the §10 gate's
# fp32 reference oracle (B1): a kind the proxy cannot serve is a hard gate
# failure, never an HF-served audit cell — the enum keeps hf_shim so gate
# tooling has an identity and foreign rows refuse loudly at pairing.
SERVING_PATHS = {
    "proxy_vllm": ("lens", "lora", "passthrough"),
    "hf_shim": ("apply",),
}
SERVING_IDENTITIES = tuple(
    f"{backend}+{path}+{mode}"
    for backend in SERVING_BACKENDS
    for path in SERVING_PATHS[backend]
    for mode in SERVING_MODES
)

# v0 in-process values first ("vllm" accepts kind:none arms only — that
# refusal in validate_manifest stays scoped to the in-process value); the
# Petri serving identities widen the enum.
ENGINES = ("hf", "vllm") + SERVING_IDENTITIES
# Manifest arm kinds the v0 loader accepts. `control:*` is a namespace:
# any kind starting with CONTROL_KIND_PREFIX is a control arm (§8).
STEERING_KINDS = ("steering", "sisteer_object")
NONE_KIND = "none"
CONTROL_KIND_PREFIX = "control:"
# All hashes are sha256, stored/compared as 16-hex prefixes (§2.1).
HASH_HEX_LEN = 16


# --- Dataclasses (frozen field sets — see test_capeval_protocol.py) --------


@dataclass
class PrimarySpec:
    """One target metric (``target.primaries[*]``), resolved via the
    targets.py plugin registry."""

    metric: str  # e.g. "si_compliance"
    plugin: str  # e.g. "sisteer_compliance"
    params: Dict[str, Any]
    direction: str  # "up" | "down"
    min_effect: float


@dataclass
class CorpusRef:
    """A pinned corpus reference + slice spec (``target.relevant_corpus``).

    Either ``path`` (a file on shared storage) or ``repo_id``+``revision``
    (a hub dataset at a full SHA) — exactly one addressing mode, refusal
    otherwise. The RESOLVED slice is content-hashed into
    ``harness_config_hash`` (F20); §4.5's kl_localization reads it.
    """

    slice: Dict[str, Any]  # e.g. {"first": 200, "field": "prompt"}
    path: Optional[str] = None
    repo_id: Optional[str] = None
    revision: Optional[str] = None  # full SHA, required with repo_id


@dataclass
class TargetSpec:
    """The target/off-target partition (REQUIRED for mode: sweep, F20)."""

    primaries: List[PrimarySpec]
    relevant_corpus: CorpusRef
    secondary: List[PrimarySpec] = field(default_factory=list)
    # Domains legitimately moved by the target (excluded from the gating
    # set, still run + reported), e.g. instruction_following for S2 SIs.
    off_target_exclusions: List[str] = field(default_factory=list)


@dataclass
class AblationSpec:
    """One manifested arm (``ablations[*]``)."""

    name: str
    # "steering" | "sisteer_object" | "none" | "control:<name>"
    kind: str
    # REQUIRED for steering kinds; None for "none" and control:* kinds.
    artifact: Optional[str] = None
    # REQUIRED for steering kinds; validated against the artifact's JSON
    # sidecar at resolve time — a mismatch is a refusal (F9).
    layer: Optional[int] = None
    # A DECLARED multi-layer object (a LoRA fitted on several layers, e.g. the
    # Nemotron hybrid's four attention layers; hand-ported from the tinycap-full22
    # lane's cd8927d6 by bcb-bfcl-200-widen S1-f). When set, F9 requires this set
    # to EQUAL the object's / sidecar's layers — every layer, nothing else —
    # instead of the single-layer membership rule on ``layer``; a layer is never
    # picked to pass. The identity hash already carries the object's own
    # ``layers`` (resolved_spec), so the declaration changes no hash. Read from
    # the serving config's ``multi_layer_arms`` (cell_controls.arm_spec_for_cell).
    layers: Optional[List[int]] = None
    # A single coeff = fixed-dose audit mode; lists >1 are ladders
    # (controls use them). Empty only for kind "none".
    coeffs: List[float] = field(default_factory=list)
    positions: str = "all"
    norm_relative: bool = False
    # control:* kinds only — hashed via canonical_json({kind, seed,
    # params}) since there is no artifact (§2.1).
    seed: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Manifest:
    """One sweep/control/calibration run spec."""

    sweep: str
    mode: str  # "sweep" | "control" | "calibration" (F20)
    base: str  # model id, e.g. "meta-llama/Llama-3.1-8B-Instruct"
    base_revision: str  # full 40-hex SHA, required
    engine: str  # "hf" | "vllm"
    tiers: List[int]
    ablations: List[AblationSpec]
    # v0 default AND the only value the v0 loader accepts.
    confirmatory: bool = False
    # REQUIRED for mode "sweep", FORBIDDEN otherwise (F20).
    target: Optional[TargetSpec] = None
    # Per-domain m_d overrides, raw units (domain name -> margin).
    margins: Dict[str, float] = field(default_factory=dict)


# --- Loading / validation ---------------------------------------------------


def _refuse(msg: str) -> None:
    raise ManifestError(msg)


def _check_unknown_keys(raw: Mapping[str, Any], allowed: tuple, where: str) -> None:
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        _refuse(f"unknown key(s) {unknown} in {where} (allowed: {sorted(allowed)})")


def _dc_field_names(dc) -> tuple:
    return tuple(f.name for f in dc_fields(dc))


def _build_manifest(raw: Mapping[str, Any], where: str) -> Manifest:
    """Raw YAML dict -> Manifest, refusing unknown keys at every level."""
    if not isinstance(raw, Mapping):
        _refuse(f"{where}: manifest document is not a mapping")
    _check_unknown_keys(raw, _dc_field_names(Manifest), where)
    for req in ("sweep", "mode", "base", "base_revision", "engine", "tiers", "ablations"):
        if req not in raw:
            _refuse(f"{where}: missing required key {req!r}")

    target_raw = raw.get("target")
    target = None
    if target_raw is not None:
        if not isinstance(target_raw, Mapping):
            _refuse(f"{where}: target is not a mapping")
        _check_unknown_keys(target_raw, _dc_field_names(TargetSpec), f"{where}.target")
        if "primaries" not in target_raw or "relevant_corpus" not in target_raw:
            _refuse(f"{where}.target: missing primaries and/or relevant_corpus")

        def _primary(p: Mapping[str, Any], pwhere: str) -> PrimarySpec:
            if not isinstance(p, Mapping):
                _refuse(f"{pwhere}: primary entry is not a mapping")
            _check_unknown_keys(p, _dc_field_names(PrimarySpec), pwhere)
            for k in ("metric", "plugin", "direction", "min_effect"):
                if k not in p:
                    _refuse(f"{pwhere}: missing required key {k!r}")
            return PrimarySpec(
                metric=p["metric"], plugin=p["plugin"],
                params=dict(p.get("params", {})),
                direction=p["direction"], min_effect=p["min_effect"],
            )

        rc_raw = target_raw["relevant_corpus"]
        if not isinstance(rc_raw, Mapping):
            _refuse(f"{where}.target.relevant_corpus is not a mapping")
        _check_unknown_keys(rc_raw, _dc_field_names(CorpusRef),
                            f"{where}.target.relevant_corpus")
        if "slice" not in rc_raw:
            _refuse(f"{where}.target.relevant_corpus: missing required key 'slice'")
        target = TargetSpec(
            primaries=[_primary(p, f"{where}.target.primaries[{i}]")
                       for i, p in enumerate(target_raw["primaries"] or [])],
            secondary=[_primary(p, f"{where}.target.secondary[{i}]")
                       for i, p in enumerate(target_raw.get("secondary") or [])],
            off_target_exclusions=list(target_raw.get("off_target_exclusions") or []),
            relevant_corpus=CorpusRef(**rc_raw),
        )

    ablations = []
    for i, a in enumerate(raw["ablations"] or []):
        awhere = f"{where}.ablations[{i}]"
        if not isinstance(a, Mapping):
            _refuse(f"{awhere}: ablation entry is not a mapping")
        _check_unknown_keys(a, _dc_field_names(AblationSpec), awhere)
        for k in ("name", "kind"):
            if k not in a:
                _refuse(f"{awhere}: missing required key {k!r}")
        # bcb-bfcl-200-widen code review: the DECLARED multi-layer set
        # is CARRIED (a manifest declaring `layers` used to load with layers=None
        # and fall back to single-layer membership) and typed here — a non-empty
        # list of distinct ints, or absent; never a string, a bool or a scalar
        layers = a.get("layers")
        if layers is not None:
            if (not isinstance(layers, list) or not layers
                    or not all(isinstance(x, int) and not isinstance(x, bool) for x in layers)):
                _refuse(f"{awhere}.layers must be a non-empty list of ints (the object's own "
                        f"target_layers), got {layers!r}")
            if len(set(layers)) != len(layers):
                _refuse(f"{awhere}.layers repeats a layer: {layers!r}")
            layers = [int(x) for x in layers]
        ablations.append(AblationSpec(
            name=a["name"], kind=a["kind"], artifact=a.get("artifact"),
            layer=a.get("layer"), layers=layers, coeffs=list(a.get("coeffs", [])),
            positions=a.get("positions", "all"),
            norm_relative=a.get("norm_relative", False),
            seed=a.get("seed"), params=dict(a.get("params", {})),
        ))

    return Manifest(
        sweep=raw["sweep"], mode=raw["mode"], base=raw["base"],
        base_revision=raw["base_revision"], engine=raw["engine"],
        tiers=list(raw["tiers"]), ablations=ablations,
        confirmatory=raw.get("confirmatory", False), target=target,
        margins=dict(raw.get("margins", {})),
    )


def _resolve_artifact_path(artifact: str, manifest_dir: Optional[Path]) -> Path:
    """Absolute paths stand verbatim; relative paths resolve against the
    MANIFEST FILE'S DIRECTORY — never the cwd (review-fix contract,
    2026-08-15: the manifest is the one stable anchor; cwd is whatever
    the caller happened to be in, so a cwd-relative path silently changes
    meaning between the loader, the driver, and a Slurm job). The result
    is absolute whenever a manifest_dir is known; the loader stores it
    back into the spec so every downstream consumer uses it verbatim.
    A site placeholder (``$SHARED/...``) is expanded first through
    ``sisteer.site``; an unset one refuses by name, so a placeholder is
    never anchored to the manifest's directory as a relative path."""
    from sisteer import site as _site
    try:
        p = _site.expand_path(artifact, key="artifact")
    except _site.PlaceholderError as exc:
        _refuse(str(exc))
    if p.is_absolute():
        return p
    if manifest_dir is None:
        # validate_manifest-only path: an in-memory Manifest has no
        # directory to anchor to; existence checks are loader-side.
        return p
    return (Path(manifest_dir) / p).resolve()


def sidecar_path(artifact: Union[str, Path]) -> Path:
    """The JSON sidecar next to a steering artifact (the path
    SteeringVector.save writes: ``<artifact>.with_suffix('.json')``)."""
    return Path(artifact).with_suffix(".json")


def _sidecar_layers(sidecar: Mapping[str, Any], where: str) -> List[int]:
    """Layer(s) recorded in a sidecar: either ``layer: 15`` or
    ``layers: [15]`` (SteeringVector.save writes the latter)."""
    if "layer" in sidecar:
        return [int(sidecar["layer"])]
    if "layers" in sidecar:
        return [int(x) for x in sidecar["layers"]]
    _refuse(f"{where}: sidecar has neither 'layer' nor 'layers' — cannot "
            "validate the manifest layer against it (F9)")


def f9_layer_problem(arm: AblationSpec, observed: Sequence[int], *,
                     observed_what: str = "the artifact sidecar's layer(s)") -> Optional[str]:
    """The ONE F9 rule (cd8927d6, hand-ported by bcb-bfcl-200-widen S1-f). A
    DECLARED multi-layer set (``arm.layers``) must EQUAL the observed layers —
    every layer, nothing else; otherwise the manifest's single ``layer`` must
    appear in them. Returns the refusal text naming both sides, or None."""
    obs = [int(x) for x in observed]
    declared = getattr(arm, "layers", None)
    if declared:
        want = sorted({int(x) for x in declared})
        if want != sorted(set(obs)):
            return (f"arm {arm.name!r}: declared layer set {want} does not equal {observed_what} "
                    f"{obs} (F9 refusal) — a multi-layer declaration must be the object's own "
                    "target_layers, every one of them and nothing else; correct the declaration, "
                    "never trim it to pass")
        return None
    if arm.layer not in obs:
        return (f"arm {arm.name!r}: manifest layer {arm.layer} does not match "
                f"{observed_what} {obs} (F9 refusal)")
    return None


def validate_layer_against_sidecar(arm: AblationSpec, sidecar: Mapping[str, Any]) -> None:
    """F9: the manifest's explicit ``layer`` must appear in the artifact
    sidecar's recorded layer(s) — or, for a DECLARED multi-layer arm, the declared
    set must equal them; a mismatch is a refusal naming both (:func:`f9_layer_problem`)."""
    layers = _sidecar_layers(sidecar, f"arm {arm.name!r}")
    problem = f9_layer_problem(arm, layers)
    if problem:
        _refuse(problem)


def _validate_artifacts_on_disk(manifest: Manifest, manifest_dir: Optional[Path]) -> None:
    """Loader-side filesystem refusals: absent artifacts; absent sidecar
    for kind 'steering'; sidecar-layer mismatch (F9). For
    'sisteer_object' (torch blobs) a sidecar is optional — validated when
    present; the blob's own recorded kind/layer is re-validated by the
    engine at load_object time.

    Also RESOLVES every steering-kind artifact path to absolute (relative
    paths anchor to the manifest file's directory) and stores the
    resolved path back into the spec — downstream (runtime/engine) uses
    ``arm.artifact`` verbatim, so the resolution happens exactly once,
    here (review-fix contract, 2026-08-15)."""
    for arm in manifest.ablations:
        if arm.kind not in STEERING_KINDS:
            continue
        path = _resolve_artifact_path(arm.artifact, manifest_dir)
        if not path.is_file():
            _refuse(f"arm {arm.name!r}: artifact does not exist: {path} "
                    "(relative paths resolve against the manifest's own "
                    "directory)")
        arm.artifact = str(path)
        sc = sidecar_path(path)
        if arm.kind == "steering":
            if not sc.is_file():
                _refuse(f"arm {arm.name!r}: kind 'steering' requires the JSON "
                        f"sidecar next to the artifact; missing: {sc}")
        if sc.is_file():
            try:
                sidecar = json.loads(sc.read_text())
            except json.JSONDecodeError as e:
                _refuse(f"arm {arm.name!r}: unparseable sidecar {sc}: {e}")
            validate_layer_against_sidecar(arm, sidecar)


def load_manifest(path: Union[str, Path]) -> Manifest:
    """Load + validate a manifest YAML. Refusals (never warnings):
    unknown keys at any level; mode not in MODES; ``confirmatory: true``
    (v0); missing/short ``base_revision``; engine not in ENGINES
    (and engine "vllm" with any arm kind other than "none"); mode/target
    contract violations (F20); steering-kind arms missing artifact/layer/
    coeffs or whose artifact path does not exist; control:* arms missing
    ``seed``."""
    path = Path(path)
    docs = list(yaml.safe_load_all(path.read_text()))
    if len(docs) != 1:
        _refuse(f"{path}: expected exactly one YAML document, found "
                f"{len(docs)} — use load_manifests() for multi-document "
                "manifest files (the s2_rediscovery convention)")
    manifest = _build_manifest(docs[0], str(path))
    validate_manifest(manifest)
    _validate_artifacts_on_disk(manifest, path.parent)
    return manifest


def load_manifests(path: Union[str, Path]) -> List[Manifest]:
    """Load a MULTI-document manifest YAML (one Manifest per document,
    sharing a sweep name) — the s2_rediscovery convention: the schema's
    one-target-per-manifest contract (F20) meets seven cells with
    per-cell SIs by giving each cell its own document. Added without
    touching the frozen functions."""
    path = Path(path)
    docs = list(yaml.safe_load_all(path.read_text()))
    if not docs:
        _refuse(f"{path}: no YAML documents found")
    out = []
    for i, doc in enumerate(docs):
        manifest = _build_manifest(doc, f"{path}[doc {i}]")
        validate_manifest(manifest)
        _validate_artifacts_on_disk(manifest, path.parent)
        out.append(manifest)
    sweeps = {m.sweep for m in out}
    if len(sweeps) > 1:
        _refuse(f"{path}: documents disagree on sweep name: {sorted(sweeps)}")
    names = [a.name for m in out for a in m.ablations]
    if len(names) != len(set(names)):
        _refuse(f"{path}: arm names collide across documents: {names}")
    return out


def _is_full_sha(s: Any) -> bool:
    return (isinstance(s, str) and len(s) == 40
            and all(c in "0123456789abcdef" for c in s.lower()))


_CELL_KEY_UNSAFE = ("__", "/")


def _validate_arm(arm: AblationSpec) -> None:
    where = f"arm {arm.name!r}"
    if not isinstance(arm.name, str) or not arm.name:
        _refuse("ablation with empty/non-string name")
    for bad in _CELL_KEY_UNSAFE:
        if bad in arm.name:
            _refuse(f"{where}: name contains {bad!r}, which the §2.3 cell-key "
                    "format reserves")
    if not isinstance(arm.positions, str) or not arm.positions:
        _refuse(f"{where}: positions must be a non-empty string")
    if not isinstance(arm.norm_relative, bool):
        _refuse(f"{where}: norm_relative must be a bool")
    if not all(isinstance(c, (int, float)) and not isinstance(c, bool)
               for c in arm.coeffs):
        _refuse(f"{where}: coeffs must all be numbers, got {arm.coeffs!r}")

    if arm.kind in STEERING_KINDS:
        if not arm.artifact:
            _refuse(f"{where}: kind {arm.kind!r} requires an artifact path")
        if not isinstance(arm.layer, int) or isinstance(arm.layer, bool) or arm.layer < 0:
            _refuse(f"{where}: kind {arm.kind!r} requires an explicit "
                    f"non-negative integer layer (F9), got {arm.layer!r}")
        if not arm.coeffs:
            _refuse(f"{where}: kind {arm.kind!r} requires a non-empty coeff "
                    "ladder (a single coeff = fixed-dose audit mode)")
        if arm.seed is not None:
            _refuse(f"{where}: seed is only meaningful on control:* kinds")
        if arm.params:
            _refuse(f"{where}: params is only meaningful on control:* kinds")
    elif arm.kind == NONE_KIND:
        if arm.artifact is not None or arm.layer is not None:
            _refuse(f"{where}: kind 'none' takes no artifact/layer")
        if arm.coeffs:
            _refuse(f"{where}: kind 'none' takes no coeffs (base run)")
        if arm.seed is not None:
            _refuse(f"{where}: kind 'none' takes no seed")
        # params.trials is the ONE schema-blessed extension on kind none:
        # the calibration trial count (default 6; it grows in batches of
        # 3 to max 12 — the loader must accept the grown manifest, and
        # the vllm bring-up manifest uses trials: 2). Anything else in
        # params stays a refusal.
        extra_params = sorted(set(arm.params) - {"trials"})
        if extra_params:
            _refuse(f"{where}: kind 'none' params allows only 'trials' "
                    f"(calibration trial count); got {extra_params}")
        if "trials" in arm.params:
            t = arm.params["trials"]
            if not isinstance(t, int) or isinstance(t, bool) or not 1 <= t <= 12:
                _refuse(f"{where}: params.trials must be an int in 1..12 "
                        f"(D5/D10 cap), got {t!r}")
    elif arm.kind.startswith(CONTROL_KIND_PREFIX):
        if len(arm.kind) <= len(CONTROL_KIND_PREFIX):
            _refuse(f"{where}: empty control kind name")
        if arm.artifact is not None:
            _refuse(f"{where}: control:* arms carry no top-level artifact "
                    "(an input artifact, e.g. control:zero_coeff's vector, "
                    "goes in params — §8)")
        if arm.layer is not None:
            _refuse(f"{where}: control:* arms carry layer inside params")
        if not isinstance(arm.seed, int) or isinstance(arm.seed, bool):
            _refuse(f"{where}: control:* arms require an integer seed (§2.1)")
        if not arm.coeffs:
            _refuse(f"{where}: control:* arms require a non-empty coeff ladder")
    else:
        _refuse(f"{where}: unknown kind {arm.kind!r} (allowed: "
                f"{STEERING_KINDS + (NONE_KIND,)} or {CONTROL_KIND_PREFIX}<name>)")


def validate_manifest(manifest: Manifest) -> None:
    """Validate an in-memory Manifest under the same refusal rules as
    load_manifest (for manifests constructed in code). Structural rules
    only — artifact-on-disk checks are loader-side (load_manifest), since
    an in-memory Manifest has no directory to resolve relative paths
    against."""
    m = manifest
    if m.mode not in MODES:
        _refuse(f"mode {m.mode!r} not in {MODES}")
    if m.confirmatory is not False:
        _refuse("confirmatory: false is the only value the v0 loader accepts "
                f"(D4/F1); got {m.confirmatory!r}. Verdict-bearing runs are "
                "v1 (confirmatory tier set).")
    if not isinstance(m.sweep, str) or not m.sweep:
        _refuse("sweep must be a non-empty string")
    if not isinstance(m.base, str) or not m.base:
        _refuse("base must be a non-empty model id")
    if not _is_full_sha(m.base_revision):
        _refuse(f"base_revision must be a full 40-hex commit SHA, got "
                f"{m.base_revision!r}")
    if m.engine not in ENGINES:
        _refuse(f"engine {m.engine!r} not in {ENGINES}")
    if (not isinstance(m.tiers, list) or not m.tiers
            or any(t not in (0, 1) for t in m.tiers)
            or len(set(m.tiers)) != len(m.tiers)):
        _refuse(f"tiers must be a non-empty duplicate-free subset of [0, 1] "
                f"(v0 implements Tiers 0-1 only), got {m.tiers!r}")

    # F20 mode/target contract.
    if m.mode == "sweep" and m.target is None:
        _refuse("mode: sweep REQUIRES a target spec (F20)")
    if m.mode in ("control", "calibration") and m.target is not None:
        _refuse(f"mode: {m.mode} FORBIDS a target spec (F20)")

    if m.target is not None:
        t = m.target
        if not t.primaries:
            _refuse("target.primaries must be non-empty")
        for p in list(t.primaries) + list(t.secondary):
            if p.direction not in ("up", "down"):
                _refuse(f"primary {p.metric!r}: direction {p.direction!r} "
                        "not in ('up', 'down')")
            if not isinstance(p.min_effect, (int, float)) or p.min_effect <= 0:
                _refuse(f"primary {p.metric!r}: min_effect must be > 0")
            if not isinstance(p.params, dict):
                _refuse(f"primary {p.metric!r}: params must be a mapping")
        rc = t.relevant_corpus
        has_path, has_repo = rc.path is not None, rc.repo_id is not None
        if has_path == has_repo:
            _refuse("target.relevant_corpus: exactly one of path / repo_id "
                    f"(got path={rc.path!r}, repo_id={rc.repo_id!r})")
        if has_repo and not _is_full_sha(rc.revision):
            _refuse("target.relevant_corpus: repo_id requires a full 40-hex "
                    f"revision, got {rc.revision!r}")
        if has_path and rc.revision is not None:
            _refuse("target.relevant_corpus: revision is meaningless with path")
        if not isinstance(rc.slice, dict) or not rc.slice:
            _refuse("target.relevant_corpus.slice must be a non-empty mapping")

    if not isinstance(m.margins, dict):
        _refuse("margins must be a mapping")
    for dom, val in m.margins.items():
        if not isinstance(val, (int, float)) or isinstance(val, bool) or val < 0:
            _refuse(f"margins[{dom!r}] must be a non-negative number, got {val!r}")

    if not m.ablations:
        _refuse("ablations must be non-empty (calibration manifests carry "
                "one kind: none arm)")
    names = [a.name for a in m.ablations]
    if len(names) != len(set(names)):
        _refuse(f"duplicate arm names: {sorted(n for n in names if names.count(n) > 1)}")
    for arm in m.ablations:
        _validate_arm(arm)
    if m.engine == "vllm":
        bad = [a.name for a in m.ablations if a.kind != NONE_KIND]
        if bad:
            _refuse(f"engine 'vllm' accepts kind: none arms only in v0 "
                    f"(base-only bring-up, §6.2); offending arms: {bad}")


# --- Hashing (all sha256, stored as 16-hex prefixes) ------------------------


def sha256_hex16(data: bytes) -> str:
    """sha256(data) as a HASH_HEX_LEN(=16)-hex prefix — the storage/
    comparison form of every capeval hash (§2.1)."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        _refuse(f"sha256_hex16 takes bytes, got {type(data).__name__}")
    return hashlib.sha256(bytes(data)).hexdigest()[:HASH_HEX_LEN]


def _first_non_finite(obj: Any, path: str = "") -> Optional[str]:
    """The JSON key path of the FIRST non-finite float in ``obj``
    (``a.b[3].c``), or None. Depth-first in the order ``json.dumps`` would
    walk with ``sort_keys=True``, so the path names the value that actually
    tripped the encoder. Pure; bools are not floats here (``json`` writes
    them as ``true``/``false``)."""
    if isinstance(obj, float):
        return None if math.isfinite(obj) else (path or "<root>")
    if isinstance(obj, Mapping):
        for k in sorted(obj, key=str):
            hit = _first_non_finite(obj[k], f"{path}.{k}" if path else str(k))
            if hit is not None:
                return hit
        return None
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            hit = _first_non_finite(v, f"{path}[{i}]")
            if hit is not None:
                return hit
    return None


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization used by every hash: sorted keys,
    no insignificant whitespace, stable across dict insertion order.
    Hash-stability across key order is pinned by the protocol test.
    Non-JSON-serializable values and NaN/inf are refusals (a hash over a
    guessed serialization is a silent-drift channel).

    A NaN/inf refusal NAMES THE KEY PATH (§PW-nanfix-1). The bare encoder
    message ("Out of range float values are not JSON compliant: nan") says
    only that the payload holds one somewhere, which cost a wave of Llama
    cells a whole diagnosis: the value was a scorer's NaN sentinel three
    dict levels down in a row's ``extra``, and nothing in the traceback said
    so. The path is derived by walking the SAME object the encoder rejected,
    never guessed."""
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except ValueError as e:
        where = _first_non_finite(obj)
        if where is not None:
            raise ManifestError(
                f"canonical_json: non-finite float at key path {where!r} — a hash "
                "(or a stored payload) over NaN/inf is not JSON and never a fact; "
                f"the producer of that key must ship None with its counts ({e})") from e
        raise ManifestError(f"canonical_json: unserializable object: {e}") from e
    except TypeError as e:
        raise ManifestError(f"canonical_json: unserializable object: {e}") from e


def intervention_hash(
    resolved_spec: Mapping[str, Any],
    artifact_bytes: Optional[bytes] = None,
    sidecar_bytes: Optional[bytes] = None,
) -> str:
    """The §2.1/F9 intervention hash, 16-hex.

    Steering kinds: H(artifact_bytes || sidecar_bytes ||
    canonical_json(resolved_spec)), where ``resolved_spec`` is the
    canonical RESOLVED intervention spec — every effective field after
    load_object has applied its defaults (layers, positions, skip_bos,
    norm_relative, coeff, converter_version) — so the hash captures what
    actually runs, not what was typed. Both byte arguments are REQUIRED
    for steering kinds (refusal if absent). A blob artifact with no
    on-disk sidecar passes ``sidecar_bytes=b""`` — explicitly, so the
    absence is a stated input, never an accidental default.

    control:* kinds: H(canonical_json({kind, seed, params})) with both
    byte arguments None (refusal if provided).

    kind "none": H(canonical_json(resolved_spec)), no byte arguments —
    the base arm's hash is a pure function of its resolved spec.
    """
    kind = resolved_spec.get("kind")
    if kind is None:
        _refuse("intervention_hash: resolved_spec has no 'kind'")

    if kind in STEERING_KINDS:
        if artifact_bytes is None or sidecar_bytes is None:
            _refuse(f"intervention_hash: kind {kind!r} requires BOTH "
                    "artifact_bytes and sidecar_bytes (F9; pass b'' for a "
                    "blob kind with no sidecar file)")
        payload = (bytes(artifact_bytes) + bytes(sidecar_bytes)
                   + canonical_json(dict(resolved_spec)).encode("utf-8"))
        return sha256_hex16(payload)

    if artifact_bytes is not None or sidecar_bytes is not None:
        _refuse(f"intervention_hash: kind {kind!r} takes no byte arguments")

    if isinstance(kind, str) and kind.startswith(CONTROL_KIND_PREFIX):
        if "seed" not in resolved_spec or resolved_spec["seed"] is None:
            _refuse(f"intervention_hash: control kind {kind!r} requires a seed")
        payload = canonical_json({
            "kind": kind,
            "seed": resolved_spec["seed"],
            "params": dict(resolved_spec.get("params", {})),
        }).encode("utf-8")
        return sha256_hex16(payload)

    if kind == NONE_KIND:
        return sha256_hex16(canonical_json(dict(resolved_spec)).encode("utf-8"))

    _refuse(f"intervention_hash: unknown kind {kind!r}")


# ============================================================================
# Petri Bloom — ADDITIVE, guarded.
#
# Everything below is the Petri misalignment audit's manifest surface. It
# touches NOTHING above: the capability Manifest schema, its loaders, and
# intervention_hash are unchanged (frozen-schema discipline; the existing
# capeval suite must stay green). Phase-2 (serving/, tiers/petri_runner.py,
# scripts/petri_run.py) builds against these contracts.
# ============================================================================

#: The six audited Petri behaviors (all Petri builtins).
#: domains.yaml registers task `petri/<behavior>` for each, all in the ONE
#: display-kind `petri_behavior` domain.
PETRI_BEHAVIORS = (
    "cyber_bio_chem_capabilities",
    "self_preservation",
    "delusion_sycophancy",
    "defer_to_users",
    "reasoning_unfaithfulness",
    "instruction_evasion",
)

#: The prompted-baseline kind: unmodified base weights + a verified
#: SI-prefix. Its intervention identity IS the SI bytes.
PETRI_PROMPTED_KIND = "prompted"
#: The STACKED kind: a verified SI system prefix AND a steered defense
#: object served simultaneously on one cell (the prompt+object headline
#: combination). Its intervention identity covers BOTH legs: the object
#: artifact/sidecar bytes plus a wrapper resolved spec that names the
#: exact prefix bytes by sha16 (the SI text itself stays out of worker
#: registries — containment). Built by
#: ``defense_cells.stacked_resolved_spec`` — the ONE wrapper
#: construction, so no two callers can hash different shapes.
PETRI_STACKED_KIND = "stacked"
#: The system instructions the prompted baselines and the Class A
#: `si_<si>__<method>` cells compile (two-SI spec). "irrelevant" joined on
#: 2026-08-28: the off-target CONTROL SI (`si_irrelevant` — a length-matched,
#: task-irrelevant text) was previously OUTSIDE the audit on purpose, but the
#: control is compared on the SAME serving engine as everything else, so it is
#: a first-class SI id now. Purely additive: no existing spec's hash inputs
#: change — the hash reads spec CONTENT (cells / prompted_prefix_hashes),
#: never this tuple.
#: The PROMPTABLE SI ids: the three system instructions the prompted
#: baselines render and the two-SI Class A cells compile.
#: "constitution_full" joined on 2026-09-10: the FULL constitution text
#: (`si_constitution_full`, 39 sections / 28,977 words) as a prompted baseline —
#: `constitution` is and stays the compressed EXCERPT. Purely additive like
#: `irrelevant`: no existing spec's hash inputs change.
PETRI_PROMPTABLE_SI_IDS = ("safety", "constitution", "irrelevant", "constitution_full")
#: OBJECT-ONLY SI ids (2026-09-04, the multi-turn + Petri audit on the
#: full-corpus cells): the document-corpus
#: PSEUDO-instructions the base-fit cells are named through —
#: `constv2-mix-full` (the constitutional corpus) and `constv2ctl2-full`
#: (the register-matched control corpus), translated by
#: petri.registry_adapter.HARNESS_TO_TRAINTEST_SI. A pseudo-instruction's
#: "text" is never a system prompt (defense_cell refuses a prompted cell
#: over a corpus_cell), so these carry NO prompted baseline and NO
#: SI-variant behavior dir: only `si_<si>__<method>` is a legal id for
#: them (defense_cells.parse_cell_id refuses `prompted__` / `stacked__`
#: up front). Purely additive, like `irrelevant` was: the harness hash
#: reads spec CONTENT, never these tuples, so no completed audit's run
#: keys move.
PETRI_OBJECT_ONLY_SI_IDS = ("constv2_mix_full", "constv2ctl2_full")
#: PROMPTED-ONLY SI ids (2026-09-10): the FULL constitution is a prompted baseline and
#: nothing else — no fitted object exists or will exist for it, so
#: `si_constitution_full__<method>` / `stacked__constitution_full__…` are not cells the
#: vocabulary can express (parse_cell_id refuses them, the mirror image of the object-only
#: refusal) and the canonical Class A matrix does not expect them. Purely additive: the hash
#: reads spec CONTENT, never these tuples.
PETRI_PROMPTED_ONLY_SI_IDS = ("constitution_full",)
PETRI_SI_IDS = PETRI_PROMPTABLE_SI_IDS + PETRI_OBJECT_ONLY_SI_IDS
#: the PROMPTABLE SI ids a Class A cell of a PROMPTABLE method (`si_<si>__<method>`, defense_cells.PROMPTABLE_CLASS_A_METHODS)
#: may name: every promptable SI except the prompted-only ones (constitution_full). The object-only SIs
#: (PETRI_OBJECT_ONLY_SI_IDS) are NOT here: their cells are the object-only methods'
#: (petri_stats.object_only_cell_ids), never a promptable method's.
PETRI_CLASS_A_SI_IDS = tuple(si for si in PETRI_PROMPTABLE_SI_IDS if si not in PETRI_PROMPTED_ONLY_SI_IDS)
#: si_id -> the SI-prefix variant behavior directory it audits under.
#: AUTHORITATIVE here so the harness hash can derive the
#: full expected behavior-directory inventory; petri.scenarios mirrors it
#: (SI_VARIANT_DIRS) and a test pins the two equal.
PETRI_SI_VARIANT_DIRS = {"safety": "PROMPTED_SAFETY",
                         "constitution": "PROMPTED_CONSTITUTION",
                         "irrelevant": "PROMPTED_IRRELEVANT",
                         "constitution_full": "PROMPTED_CONSTITUTION_FULL"}
#: manifest cell-id prefix of the prompted baselines — the harness hash
#: derives the in-play SI-variant dirs from it (H10).
PETRI_PROMPTED_CELL_PREFIX = "prompted__"
#: the two prompt roles whose exact BYTES bind the harness hash (H10).
PETRI_PROMPT_ROLES = ("auditor", "judge")
#: Sentinel for a hash-like PetriSpec field whose value is not frozen yet
#: (scenario bundle before the one `bloom scenarios` pass, the chat
#: template / serving commit before Phase-2 pins them, SI-prefix hashes
#: before the two-SI landing). The LOADER accepts it so the manifest can be
#: committed now; every HASH computation REFUSES it loudly — no run can
#: ever be keyed on an unfrozen input (loud refusal, never a silent
#: placeholder hash).
PETRI_UNFROZEN = "UNFROZEN"


@dataclass(frozen=True)
class ServingIdentity:
    """A parsed ``<backend>+<path>+<mode>`` serving identity."""

    backend: str  # proxy_vllm | hf_shim
    path: str  # lens | lora | passthrough (proxy_vllm); apply (hf_shim)
    mode: str  # interleaved | exact_replay


def is_serving_identity(engine: Any) -> bool:
    """True iff ``engine`` is a (syntactically) Petri serving identity —
    the in-process capability values ("hf", "vllm") never contain '+'."""
    return isinstance(engine, str) and "+" in engine


def parse_serving_identity(engine: Any) -> ServingIdentity:
    """Parse + validate a serving-identity string. Malformed or unknown
    components are refusals, never defaults."""
    if not isinstance(engine, str) or not engine:
        _refuse(f"parse_serving_identity: not a string: {engine!r}")
    parts = engine.split("+")
    if len(parts) != 3 or not all(parts):
        _refuse(f"parse_serving_identity: {engine!r} is not "
                "'<backend>+<path>+<mode>' (plan §3)")
    backend, path, mode = parts
    if backend not in SERVING_BACKENDS:
        _refuse(f"parse_serving_identity: backend {backend!r} not in "
                f"{SERVING_BACKENDS}")
    if mode not in SERVING_MODES:
        _refuse(f"parse_serving_identity: mode {mode!r} not in {SERVING_MODES}")
    if path not in SERVING_PATHS[backend]:
        _refuse(f"parse_serving_identity: path {path!r} invalid for backend "
                f"{backend!r} (allowed: {SERVING_PATHS[backend]})")
    return ServingIdentity(backend=backend, path=path, mode=mode)


def serving_identity(backend: str, path: str, mode: str) -> str:
    """Build a validated ``<backend>+<path>+<mode>`` identity string."""
    ident = f"{backend}+{path}+{mode}"
    parse_serving_identity(ident)  # loud refusal on any bad component
    return ident


# --- PetriSpec: the Petri manifest block ------------------------------------


_PETRI_PROVENANCE_PACKAGE = "package-pinned@"
_PETRI_PROVENANCE_OVERRIDE = "override@"


@dataclass
class BloomOptions:
    """The ``bloom_audit`` task options that shape the elicitation, plus
    the evaluation epoch count and seed — ALL of them harness-hash
    inputs. Without them, a cell run once with ``realism_filter: false``
    and then edited to ``true`` kept its ``petri_run_key`` and resumed
    old samples under the new configuration. Manifest-owned and
    canonical: the runner passes exactly these values into ``eval_set``
    and stamps/verifies them in every ``.eval`` log. All fields REQUIRED
    — no defaults, so an unstated option can never silently diverge from
    what ran."""

    max_turns: int  # auditor conversation-turn budget per scenario
    enable_prefill: bool
    enable_rollback: bool
    compaction: bool  # transcript compaction on long audits
    realism_filter: bool
    epochs: int  # evaluation epochs per scenario (>= 1)
    seed: int  # the ONE audit-wide evaluation seed


#: The Petri TARGET-role decode contract: ``harness`` is today's behaviour
#: byte for byte -- the
#: target request carries ONLY ``max_tokens`` (``petri_runner.target_response_
#: cap``) and the server fills every sampler key; ``vendor`` sends the
#: registry's vendor-preset row on EVERY target request (the six sampler keys
#: incl. ``presence_penalty``, which no server default can carry) plus a
#: per-scenario-turn ``seed``. A manifest declaring ``vendor`` is a different
#: instrument: the block enters the harness hash (ADDITIVE-WHEN-ABSENT, the
#: ``auditor_fallbacks`` pattern), so every existing Petri store keeps its
#: ``petri_run_key`` and a vendor audit gets a fresh namespace.
PETRI_TARGET_DECODE_HARNESS = "harness"
PETRI_TARGET_DECODE_VENDOR = "vendor"
PETRI_TARGET_DECODE_MODES = (PETRI_TARGET_DECODE_HARNESS,
                             PETRI_TARGET_DECODE_VENDOR)
#: the per-request seed rule a vendor target draws under (the closure lives in
#: ``petri_runner.petri_target_seed``): ``sha256(f"{seed_base}|petri|
#: {behaviour}|{scenario_id}|{turn}")[:4]`` -- the static rule's shape, keyed
#: on the Petri item identity (scenario, target turn) instead of (goal,
#: sample). Named here (torch-free) so the hash input and the row record share
#: one string.
PETRI_TARGET_SEED_RULE = "per_scenario_turn"
#: the registry eval category a Petri vendor row must carry
PETRI_DECODE_EVAL_CATEGORY = "petri"
#: the registry mode a Petri vendor row must carry
PETRI_DECODE_VENDOR_MODE = "vendor_preset"


@dataclass
class PetriTargetDecode:
    """The manifest's ``target_decode:`` block.

    ``mode`` is one of :data:`PETRI_TARGET_DECODE_MODES`. Under ``vendor``,
    ``policy`` is the EXACT registered decode-policy ref WITH its version
    (``qwen35_4b_it|petri|off|vendor_preset@v1``): resolved by
    ``PolicyFile.by_ref`` only -- never ``resolve(mode=...)`` (which ignores
    the eval and returns the model-wide sampling-mode row, the IFEval row
    for Qwen) and never a default lookup. Under
    ``harness`` ``policy`` must be None.

    ``seed_base`` is the RUN's replicate label (the literal training seed s
    of the object served), set by the driver's
    ``--target-seed-base`` and NEVER declared in a manifest: it is not a
    harness input -- one standup serves every seed base (the Arditi proxy
    carries three seed-runs on one manifest), and the standup's
    ``cells_served.json`` must agree with the client's enumeration on
    ``harness_config_hash`` / ``petri_run_key`` cell by cell -- so it is
    excluded from the hash and instead stamped on the cell manifest, every
    row, the eval-log metadata, and the Inspect log-dir namespace
    (``petri_runner``)."""

    mode: str
    policy: Optional[str] = None
    seed_base: Optional[int] = None


@dataclass
class PetriSpec:
    """The Petri audit manifest (``petri:`` block of
    configs/capeval/manifests/petri_twosi.yaml). All fields except the
    additive ``auditor_fallback_models`` and ``judge_fallback_models`` are
    required; omitting the auditor field selects the documented default
    chain (Grok 4.6), omitting the judge field selects NO judge chain.

    ``cells`` lists the CANONICAL DefenseCell ids only (the 22); the
    path-matched no-op control legs (``none__lens0``, ``none__lora0``)
    are DERIVED by petri.defense_cells.enumerate_cells, never
    manifest inputs."""

    audit: str  # audit/sweep name, e.g. "petri_twosi"
    behaviors: List[str]  # subset of PETRI_BEHAVIORS, non-empty
    cells: List[str]  # canonical DefenseCell ids (grammar checked by petri.defense_cells)
    scenario_bundle_hash: str  # 16-hex content hash of the frozen bundle, or UNFROZEN
    auditor_model: str  # inspect_ai model string, e.g. anthropic/claude-sonnet-5
    judge_model: str
    scenarios_model: str
    # "package-pinned@<version/commit>" (auditor/judge prompts ship inside
    # the pinned petri-bloom/inspect_ai packages) or "override@<16-hex>"
    # (the exact bytes-hash of a local prompt override). Either way an
    # auditor/judge prompt edit changes harness_config_hash.
    prompt_provenance: str
    target_model: str  # e.g. meta-llama/Llama-3.1-8B-Instruct
    target_revision: str  # full 40-hex SHA (model + tokenizer revision)
    # The committed production primary is proxy-vLLM (B1) — the validator
    # REFUSES anything else; hf_shim is the §10 gate's fp32 reference
    # oracle only and NEVER serves an audit cell: an unservable kind is a
    # hard gate failure, not an HF fallback.
    serving_backend: str
    serving_mode: str  # interleaved | exact_replay; production pinned exact_replay (§3 H5)
    serving_packages: Dict[str, str]  # package -> pinned version (vllm, vllm-lens, torch, …)
    serving_launch_flags: List[str]  # vllm serve flags (part of harness identity, §5)
    chat_template_hash: str  # 16-hex over the pinned tokenizer's rendered template, or UNFROZEN
    tool_parser: str  # vLLM tool-call parser name (e.g. llama3_json)
    serving_impl_commit: str  # full 40-hex serving-implementation commit, or UNFROZEN
    prompted_prefix_hashes: Dict[str, str]  # si_id -> 16-hex over the exact SI bytes, or UNFROZEN
    # bloom_audit task options + epochs + seed (harness-hash inputs, H10).
    bloom_options: BloomOptions
    # Ordered fallback auditors after auditor_model. This changes the
    # measurement instrument, so a NON-EMPTY chain is a harness-hash input
    # and is stamped into every eval log's metadata. Empty disables
    # fallback and is ADDITIVE: it enters neither the hash nor the log
    # metadata, so a manifest declaring `[]` (the candidate-si-refit
    # manifests, whose recorded legs ran Sonnet-only) keys and resumes
    # exactly as before the field existed (INT-1 stage 6). Omitting
    # the field keeps main's documented default: Grok 4.6, then the local
    # abliterated Qwen. Omission is the ONLY way to the default: an
    # explicitly present value must already be a list of non-empty model
    # strings (load_petri_spec checks the raw YAML value before any
    # normalization; `false` / `null` / "" / {} / a scalar are refusals).
    auditor_fallback_models: List[str] = field(default_factory=lambda: [
        "openrouter/x-ai/grok-4.6",
    ])
    # The TARGET-role decode contract (:class:`PetriTargetDecode`). None or
    # ``mode: harness`` is today's audit byte for byte and enters neither the
    # hash nor the log metadata; ``mode: vendor`` re-keys (a different
    # instrument) and is reachable only when the manifest asks for it.
    target_decode: Optional[PetriTargetDecode] = None
    # Ordered fallback JUDGES after judge_model (Grok 4.6 is the fallback
    # for BOTH roles). Same semantics as the auditor chain -- a NON-EMPTY
    # chain is a harness-hash input (role_models.judge_fallbacks) and is
    # stamped into every eval log's metadata; an empty chain is ADDITIVE
    # and enters neither -- with ONE deliberate difference: OMISSION means
    # NO chain. The auditor's default-on-omission is the trap that kept
    # re-instating the abliterated-Qwen leg from a generator that never
    # spelled it; a judge chain exists only where a manifest spells it.
    # Every manifest written before this field keys and resumes exactly as
    # before. An explicitly present value must be a list of non-empty model
    # strings (`false` / `null` / "" / {} / a scalar are refusals, checked
    # on the raw YAML value by load_petri_spec).
    judge_fallback_models: List[str] = field(default_factory=list)
    # The PER-REQUEST refusal fallback of each role:
    # the Anthropic API's server-side fallback (inspect GenerateConfig.
    # fallback_models) -- a request the role's model declines with a safety-
    # classifier refusal is served ONCE by the next model in the list, and the
    # next request is at the default again. The role's DEFAULT model is the
    # instrument and stays (Sonnet 5 / Opus 5); this is a remedy for requests
    # that would otherwise be refused, with provenance on every served message
    # (ChatMessageAssistant.model = the serving model; EvalSample.model_fallbacks
    # counts them). So it is ADDITIVE: NOT a harness-hash input (the cells keep
    # their petri_run_key and every completed sample resumes), stamped into the
    # eval-log metadata when non-empty, and compared on resume by PROVENANCE
    # (petri_runner.verify_log_metadata: every completed sample's auditor/judge
    # serving models must lie in default + this list). First-party Claude ids
    # only (`anthropic/...`): the server-side fallback cannot name another
    # provider. Omission or [] = no refusal fallback (today's audit byte for
    # byte). The sticky client-side chains above are untouched by it.
    auditor_refusal_fallback_models: List[str] = field(default_factory=list)
    judge_refusal_fallback_models: List[str] = field(default_factory=list)


#: the only provider the per-request refusal fallback can name (the Anthropic API serves it server-side)
PETRI_REFUSAL_FALLBACK_PROVIDER = "anthropic/"
#: The ONE role whose per-request refusal-fallback list may END with a non-Claude leg, and that leg's provider --
#: Grok 4.6 over OpenRouter (`openrouter/x-ai/grok-4.6`, the id the auditor's sticky chain already reaches as an
#: alias; the leg is declared as the IMMUTABLE dated build `openrouter/x-ai/grok-4.6-20260810` -- OpenRouter
#: accepts the canonical slug as a request id and routes it to that build): the terminal judge after every Claude leg
#: declines a transcript with a safety-classifier refusal. Served CLIENT-side by auditor_fallback.make_refusal_fallback_model
#: like every other leg (the same rubric prompt; its structured answer is the score; a Grok refusal or non-answer is the
#: terminal judge refusal filed under Grok, B4; a Grok HTTP/transport error is a hard error (B2)). ADDITIVE like the
#: rest of the list: never a harness-hash input. The auditor's list stays first-party.
PETRI_REFUSAL_FALLBACK_TERMINAL_ROLE = "judge"
PETRI_REFUSAL_FALLBACK_TERMINAL_PROVIDER = "openrouter/"


def _is_hex16(s: Any) -> bool:
    return (isinstance(s, str) and len(s) == HASH_HEX_LEN
            and all(c in "0123456789abcdef" for c in s))


def _check_hash_or_unfrozen(value: Any, field_name: str) -> None:
    if value == PETRI_UNFROZEN:
        return
    if not _is_hex16(value):
        _refuse(f"petri.{field_name} must be a {HASH_HEX_LEN}-hex hash or "
                f"the literal {PETRI_UNFROZEN!r}, got {value!r}")


def validate_petri_spec(spec: PetriSpec) -> None:
    """Loud structural validation of a PetriSpec (mirrors
    validate_manifest's stance: refusals, never warnings). UNFROZEN
    sentinels are legal HERE (the manifest is committed before the bundle/
    serving pins freeze) and refused at hash time."""
    s = spec
    if not isinstance(s.audit, str) or not s.audit:
        _refuse("petri.audit must be a non-empty string")
    for bad in _CELL_KEY_UNSAFE:
        if bad in s.audit:
            _refuse(f"petri.audit contains {bad!r}, reserved by the cell-key "
                    "format (§2.3)")
    if (not isinstance(s.behaviors, list) or not s.behaviors
            or len(set(s.behaviors)) != len(s.behaviors)):
        _refuse(f"petri.behaviors must be a non-empty duplicate-free list, "
                f"got {s.behaviors!r}")
    unknown_b = sorted(set(s.behaviors) - set(PETRI_BEHAVIORS))
    if unknown_b:
        _refuse(f"petri.behaviors: unknown behavior(s) {unknown_b} "
                f"(registered: {list(PETRI_BEHAVIORS)})")
    if (not isinstance(s.cells, list) or not s.cells
            or not all(isinstance(c, str) and c for c in s.cells)):
        _refuse("petri.cells must be a non-empty list of cell-id strings")
    if len(set(s.cells)) != len(s.cells):
        dups = sorted({c for c in s.cells if s.cells.count(c) > 1})
        _refuse(f"petri.cells: duplicate cell id(s) {dups}")
    _check_hash_or_unfrozen(s.scenario_bundle_hash, "scenario_bundle_hash")
    for f in ("auditor_model", "judge_model", "scenarios_model",
              "target_model", "tool_parser"):
        v = getattr(s, f)
        if not isinstance(v, str) or not v:
            _refuse(f"petri.{f} must be a non-empty string, got {v!r}")
    if (not isinstance(s.auditor_fallback_models, list)
            or not all(isinstance(m, str) and m
                       for m in s.auditor_fallback_models)):
        _refuse("petri.auditor_fallback_models must be a list of non-empty "
                "inspect model strings")
    full_chain = [s.auditor_model, *s.auditor_fallback_models]
    if len(full_chain) != len(set(full_chain)):
        _refuse(f"petri auditor chain repeats a model: {full_chain}")
    if (not isinstance(s.judge_fallback_models, list)
            or not all(isinstance(m, str) and m
                       for m in s.judge_fallback_models)):
        _refuse("petri.judge_fallback_models must be a list of non-empty "
                "inspect model strings")
    judge_chain = [s.judge_model, *s.judge_fallback_models]
    if len(judge_chain) != len(set(judge_chain)):
        _refuse(f"petri judge chain repeats a model: {judge_chain}")
    for role, default, chain in (("auditor", s.auditor_model, s.auditor_refusal_fallback_models),
                                 ("judge", s.judge_model, s.judge_refusal_fallback_models)):
        fname = f"petri.{role}_refusal_fallback_models"
        if not isinstance(chain, list) or not all(isinstance(m, str) and m for m in chain):
            _refuse(f"{fname} must be a list of non-empty inspect model strings")
        foreign = [i for i, m in enumerate(chain) if not m.startswith(PETRI_REFUSAL_FALLBACK_PROVIDER)]
        if foreign:
            # the JUDGE's list may END with exactly ONE OpenRouter leg (Grok 4.6, the terminal judge after every Claude
            # leg refused); two non-Claude legs, a non-Claude leg before a Claude one, another provider, or any non-Claude
            # AUDITOR leg refuse by name
            terminal_ok = (role == PETRI_REFUSAL_FALLBACK_TERMINAL_ROLE and len(foreign) == 1
                           and foreign[0] == len(chain) - 1
                           and chain[-1].startswith(PETRI_REFUSAL_FALLBACK_TERMINAL_PROVIDER))
            if terminal_ok and not (len(chain[-1]) > 9 and chain[-1][-9] == "-" and chain[-1][-8:].isdigit()):
                # The leg names an IMMUTABLE DATED build (OpenRouter's canonical slug, accepted as a request id and routed
                # to that build alone), never the mutable alias an OpenRouter retarget could move under one judge_id
                _refuse(f"{fname}: the terminal {PETRI_REFUSAL_FALLBACK_TERMINAL_PROVIDER} leg must name an IMMUTABLE DATED build "
                        f"(…-YYYYMMDD, e.g. openrouter/x-ai/grok-4.6-20260810), not a mutable alias (D896); got {chain[-1]!r}")
            if not terminal_ok:
                why = ("two non-Claude legs" if len(foreign) > 1
                       else "a non-Claude leg before a Claude one (the non-Claude leg must be TERMINAL)"
                       if foreign[0] != len(chain) - 1
                       else f"a terminal leg off {PETRI_REFUSAL_FALLBACK_TERMINAL_PROVIDER}"
                       if role == PETRI_REFUSAL_FALLBACK_TERMINAL_ROLE else "a non-Claude auditor leg")
                _refuse(f"{fname}: the per-request refusal fallback is the Anthropic API's server-side "
                        f"fallback and names first-party Claude ids only ({PETRI_REFUSAL_FALLBACK_PROVIDER}...), except ONE "
                        f"TERMINAL {PETRI_REFUSAL_FALLBACK_TERMINAL_PROVIDER} leg for the {PETRI_REFUSAL_FALLBACK_TERMINAL_ROLE} "
                        f"(D869); got {chain}: {why}")
        if len(set(chain)) != len(chain) or default in chain:
            _refuse(f"{fname} repeats a model or names the role's default: {[default, *chain]}")
    pp = s.prompt_provenance
    if not isinstance(pp, str) or not (
            (pp.startswith(_PETRI_PROVENANCE_PACKAGE)
             and len(pp) > len(_PETRI_PROVENANCE_PACKAGE))
            or (pp.startswith(_PETRI_PROVENANCE_OVERRIDE)
                and _is_hex16(pp[len(_PETRI_PROVENANCE_OVERRIDE):]))):
        _refuse(f"petri.prompt_provenance must be "
                f"'{_PETRI_PROVENANCE_PACKAGE}<version/commit>' or "
                f"'{_PETRI_PROVENANCE_OVERRIDE}<{HASH_HEX_LEN}-hex>', "
                f"got {pp!r} (plan §5 H10)")
    if not _is_full_sha(s.target_revision):
        _refuse(f"petri.target_revision must be a full 40-hex SHA, got "
                f"{s.target_revision!r}")
    if s.serving_backend != "proxy_vllm":
        _refuse(f"petri.serving_backend must be 'proxy_vllm' — the committed "
                f"production primary (plan B1; hf_shim is the gate's fp32 "
                f"reference oracle and never serves an audit cell); got "
                f"{s.serving_backend!r}")
    if s.serving_mode not in SERVING_MODES:
        _refuse(f"petri.serving_mode {s.serving_mode!r} not in {SERVING_MODES}")
    if (not isinstance(s.serving_packages, dict) or not s.serving_packages
            or not all(isinstance(k, str) and k and isinstance(v, str) and v
                       for k, v in s.serving_packages.items())):
        _refuse("petri.serving_packages must be a non-empty "
                "{package: version} mapping of non-empty strings")
    if (not isinstance(s.serving_launch_flags, list)
            or not all(isinstance(f, str) and f for f in s.serving_launch_flags)):
        _refuse("petri.serving_launch_flags must be a list of non-empty "
                "strings")
    _check_hash_or_unfrozen(s.chat_template_hash, "chat_template_hash")
    if s.serving_impl_commit != PETRI_UNFROZEN and not _is_full_sha(s.serving_impl_commit):
        _refuse(f"petri.serving_impl_commit must be a full 40-hex commit SHA "
                f"or {PETRI_UNFROZEN!r}, got {s.serving_impl_commit!r}")
    if not isinstance(s.prompted_prefix_hashes, dict):
        _refuse("petri.prompted_prefix_hashes must be a mapping")
    prompted_in_play = [c for c in s.cells if isinstance(c, str) and (
        c.startswith(PETRI_PROMPTED_CELL_PREFIX) or c.startswith("stacked__"))]
    if not s.prompted_prefix_hashes and prompted_in_play:
        # the SI bytes ARE the prompted intervention (H8): a prompted or
        # stacked cell with no recorded prefix hash can never be verified
        _refuse("petri.prompted_prefix_hashes must be a non-empty mapping when "
                f"prompted/stacked cells are in play ({prompted_in_play})")
    # an EMPTY mapping is legal for an audit with no prompted baseline (the
    # sdfbase-mt wave: `none` + object-only cells, 2026-09-04) — there is
    # nothing to verify; petri_expected_behavior_keys adds no variant dir
    # a prefix hash is the PROMPTED intervention's identity (H8), so only a
    # PROMPTABLE SI may carry one — an object-only pseudo-instruction has no
    # system-prompt bytes to hash
    unknown_si = sorted(set(s.prompted_prefix_hashes) - set(PETRI_PROMPTABLE_SI_IDS))
    if unknown_si:
        _refuse(f"petri.prompted_prefix_hashes: unknown or non-promptable SI "
                f"id(s) {unknown_si} (promptable: {list(PETRI_PROMPTABLE_SI_IDS)})")
    for si, h in s.prompted_prefix_hashes.items():
        _check_hash_or_unfrozen(h, f"prompted_prefix_hashes[{si!r}]")
    _validate_bloom_options(s.bloom_options)
    _validate_target_decode(s.target_decode)


def _validate_bloom_options(bo: Any) -> None:
    if not isinstance(bo, BloomOptions):
        _refuse(f"petri.bloom_options must be a BloomOptions (the bloom_audit "
                f"task options + epochs + seed are canonical, manifest-owned "
                f"harness-hash inputs — H10), got {type(bo).__name__}")
    for name in ("enable_prefill", "enable_rollback", "compaction",
                 "realism_filter"):
        v = getattr(bo, name)
        if not isinstance(v, bool):
            _refuse(f"petri.bloom_options.{name} must be a bool, got {v!r}")
    for name, minimum in (("max_turns", 1), ("epochs", 1)):
        v = getattr(bo, name)
        if not isinstance(v, int) or isinstance(v, bool) or v < minimum:
            _refuse(f"petri.bloom_options.{name} must be an int >= {minimum}, "
                    f"got {v!r}")
    if not isinstance(bo.seed, int) or isinstance(bo.seed, bool):
        _refuse(f"petri.bloom_options.seed must be an int, got {bo.seed!r}")


_TARGET_DECODE_MANIFEST_KEYS = ("mode", "policy")


def _validate_target_decode(td: Any) -> None:
    """Structural validation of ``PetriSpec.target_decode`` (None = harness):
    a known mode; ``vendor`` carries a versioned policy ref (``...@v<int>``,
    four ``|``-separated segments, the eval segment ``petri``), ``harness``
    carries none; ``seed_base`` is None or a non-negative int (a bool is
    refused). The registry row itself is resolved at hash time
    (:func:`petri_target_decode_identity`), not here."""
    if td is None:
        return
    if not isinstance(td, PetriTargetDecode):
        _refuse(f"petri.target_decode must be a PetriTargetDecode, got "
                f"{type(td).__name__}")
    if td.mode not in PETRI_TARGET_DECODE_MODES:
        _refuse(f"petri.target_decode.mode {td.mode!r} not in "
                f"{list(PETRI_TARGET_DECODE_MODES)}")
    if td.mode == PETRI_TARGET_DECODE_HARNESS:
        if td.policy is not None:
            _refuse(f"petri.target_decode: mode {PETRI_TARGET_DECODE_HARNESS!r} "
                    f"carries policy {td.policy!r} -- the harness mode sends no "
                    "sampler row; drop the policy or declare mode "
                    f"{PETRI_TARGET_DECODE_VENDOR!r}")
    else:
        pol = td.policy
        if not isinstance(pol, str) or not pol:
            _refuse(f"petri.target_decode: mode {PETRI_TARGET_DECODE_VENDOR!r} "
                    "requires policy = the exact registered decode-policy ref "
                    "WITH its version (e.g. "
                    "'qwen35_4b_it|petri|off|vendor_preset@v1'); got "
                    f"{pol!r}")
        ref, sep, ver = pol.rpartition("@v")
        if not sep or not ver.isdigit():
            _refuse(f"petri.target_decode.policy {pol!r} carries no '@v<int>' "
                    "version -- the ref must pin the exact row version (D9)")
        segs = ref.split("|")
        if len(segs) != 4 or not all(segs):
            _refuse(f"petri.target_decode.policy {pol!r}: the ref before '@v' "
                    "must be '<model_id>|<eval>|<thinking>|<mode>' (four "
                    "non-empty segments)")
        if segs[1] != PETRI_DECODE_EVAL_CATEGORY:
            _refuse(f"petri.target_decode.policy {pol!r} names eval {segs[1]!r}, "
                    f"not {PETRI_DECODE_EVAL_CATEGORY!r} -- a Petri target draws "
                    "under the Petri row only, never another eval's")
    sb = td.seed_base
    if sb is not None and (not isinstance(sb, int) or isinstance(sb, bool)
                           or sb < 0):
        _refuse(f"petri.target_decode.seed_base must be None or a non-negative "
                f"int, got {sb!r}")


def petri_target_decode_of(spec: Any) -> Optional[PetriTargetDecode]:
    """``spec.target_decode`` -- None for a spec (or a caller-built namespace
    that predates the field, the ``serving_path_of`` convention) carrying
    none: the harness audit every such caller had."""
    return getattr(spec, "target_decode", None)


def _vendor_target_decode(spec: Any) -> Optional[PetriTargetDecode]:
    """The VALIDATED ``target_decode`` block iff the spec declares ``vendor``;
    None under harness (no block, an explicit harness block, or a namespace
    predating the field). Validated on every read, not only at load: specs
    reach the hash and the runner through ``dataclasses.replace`` and direct
    construction too, and every reader below routes through here."""
    td = petri_target_decode_of(spec)
    if td is None:
        return None
    _validate_target_decode(td)
    return td if td.mode == PETRI_TARGET_DECODE_VENDOR else None


def petri_target_decode_mode(spec: PetriSpec) -> str:
    """``harness`` (None or an explicit harness block) | ``vendor``."""
    return (PETRI_TARGET_DECODE_HARNESS if _vendor_target_decode(spec) is None
            else PETRI_TARGET_DECODE_VENDOR)


def petri_target_decode_record(spec: PetriSpec) -> Optional[Dict[str, Any]]:
    """The MANIFEST-level declaration ``{"mode", "policy"}`` a vendor audit
    carries (the enumeration record's additive key; NO seed_base -- the record
    belongs to the manifest, and a seed-run is a run of it). None under
    harness, so every existing record compares byte-identically."""
    td = _vendor_target_decode(spec)
    return None if td is None else {"mode": td.mode, "policy": td.policy}


def _leading_int(text: Any) -> Optional[int]:
    """The leading integer of a registry prose rule (``"1024 per turn"`` ->
    1024); None when the rule does not start with one."""
    if not isinstance(text, str):
        return None
    head = text.strip().split(" ", 1)[0].replace(",", "")
    return int(head) if head.isdigit() else None


def petri_target_decode_identity(spec: PetriSpec, *,
                                 policy_file: Any = None) -> Optional[Dict[str, Any]]:
    """The vendor target's decode identity -- the harness-hash input, the cell
    manifest's ``target_decode`` record and what every row records:
    ``{mode, policy_id, request_params (the six sampler keys AS SENT),
    max_tokens, seed_rule}``. None under ``harness`` (nothing enters the hash).

    Resolution is the ONE authority the registry allows:
    ``PolicyFile.by_ref(<policy>)`` -- the exact ref with its version -- then
    the row is held to: not superseded; the manifest's ``target_model``
    (``pf.model_id`` normalises the HF repo / legacy key spellings); eval
    category :data:`PETRI_DECODE_EVAL_CATEGORY`; thinking ``off``; mode
    :data:`PETRI_DECODE_VENDOR_MODE`; ALL six ``decode_policy.PARAM_KEYS``
    SENT (``presence_penalty`` included -- a row that leaves one to the
    server is not the contract); a ``max_new_tokens_rule`` whose leading
    integer is the per-turn cap. An absent row is ``NoPolicy`` (never
    invented); every other disagreement is a refusal naming it. ``policy_file``
    injects a loaded ``PolicyFile`` (tests); default: ``decode_policy.load()``
    (``$SISTEER_DECODE_POLICY`` > the package copy > the repo copy)."""
    td = _vendor_target_decode(spec)
    if td is None:
        return None
    from sisteer.registry import decode_policy as dp

    pf = policy_file if policy_file is not None else dp.load()
    pol = pf.by_ref(td.policy)  # NoPolicy when absent: refuse, never invent
    where = f"petri.target_decode.policy {td.policy!r}"
    if pol.superseded_by_policy_id:
        _refuse(f"{where} is SUPERSEDED by {pol.superseded_by_policy_id!r} -- a "
                "superseded row never drives a run (plan §20 P1-4)")
    want_model = pf.model_id(spec.target_model)
    if pol.model_id != want_model:
        _refuse(f"{where} is {pol.model_id!r}'s row but the manifest's "
                f"target_model {spec.target_model!r} is {want_model!r} -- a "
                "target draws under its own model's row only")
    if pol.eval_category != PETRI_DECODE_EVAL_CATEGORY:
        _refuse(f"{where} carries eval_category {pol.eval_category!r}, not "
                f"{PETRI_DECODE_EVAL_CATEGORY!r}")
    if pol.thinking_mode != "off":
        _refuse(f"{where} carries thinking_mode {pol.thinking_mode!r}; the Petri "
                "target runs thinking OFF (spec fixed decision 2)")
    if pol.mode != PETRI_DECODE_VENDOR_MODE:
        _refuse(f"{where} carries mode {pol.mode!r}, not "
                f"{PETRI_DECODE_VENDOR_MODE!r}")
    sent = pol.request_params()
    missing = [k for k in dp.PARAM_KEYS if sent.get(k) is None]
    if missing:
        _refuse(f"{where} does not SEND {missing} (params_sent / stated values) "
                "-- the vendor contract sends all six sampler keys on every "
                "request, presence_penalty included (no server default carries "
                "it); refusing")
    extra = sorted(set(sent) - set(dp.PARAM_KEYS))
    if extra:
        _refuse(f"{where} sends non-sampler key(s) {extra} through request_params")
    cap = _leading_int(pol.max_new_tokens_rule)
    if cap is None or cap <= 0:
        _refuse(f"{where} max_new_tokens_rule {pol.max_new_tokens_rule!r} does not "
                "start with the per-turn cap as an integer (expected e.g. "
                "'1024 per turn'); refusing to guess the cap")
    params = {k: (int(sent[k]) if k == "top_k" else float(sent[k]))
              for k in dp.PARAM_KEYS}
    return {
        "mode": PETRI_TARGET_DECODE_VENDOR,
        "policy_id": pol.policy_id,
        "request_params": params,
        "max_tokens": int(cap),
        "seed_rule": PETRI_TARGET_SEED_RULE,
    }


def load_petri_spec(path: Union[str, Path]) -> PetriSpec:
    """Load + validate a Petri manifest YAML (one document, exactly one
    top-level ``petri:`` block; unknown keys anywhere are refusals)."""
    path = Path(path)
    docs = list(yaml.safe_load_all(path.read_text()))
    if len(docs) != 1:
        _refuse(f"{path}: expected exactly one YAML document, found {len(docs)}")
    raw = docs[0]
    if not isinstance(raw, Mapping):
        _refuse(f"{path}: manifest document is not a mapping")
    _check_unknown_keys(raw, ("petri",), str(path))
    if "petri" not in raw:
        _refuse(f"{path}: missing the top-level 'petri' block")
    block = raw["petri"]
    if not isinstance(block, Mapping):
        _refuse(f"{path}: 'petri' block is not a mapping")
    where = f"{path}.petri"
    _check_unknown_keys(block, _dc_field_names(PetriSpec), where)
    optional = {"auditor_fallback_models", "judge_fallback_models", "target_decode",
                "auditor_refusal_fallback_models", "judge_refusal_fallback_models"}
    missing = [f for f in _dc_field_names(PetriSpec)
               if f not in block and f not in optional]
    if missing:
        _refuse(f"{where}: missing required key(s) {missing} (all PetriSpec "
                "fields except auditor_fallback_models / judge_fallback_models / "
                "target_decode / auditor_refusal_fallback_models / "
                "judge_refusal_fallback_models are required)")
    # The auditor fallback chain's RAW value is type-checked BEFORE the
    # dataclass is built and before any normalization: only OMITTING the
    # key selects the dataclass default (main's documented chain). Every
    # explicitly present value must already be a list of non-empty model
    # strings -- `false`, `null`, "", {} and a bare scalar string are
    # refusals, never coerced. Until 2026-09-06 the normalization below
    # ran `list(value or [])` on the raw field, so those declarations
    # silently became `[]` (a scalar string: a list of its characters),
    # hashed identically to an explicit `[]`, and ran with the bare
    # auditor instead of failing validation.
    if "auditor_fallback_models" in block:
        raw_chain = block["auditor_fallback_models"]
        if (not isinstance(raw_chain, list)
                or not all(isinstance(m, str) and m for m in raw_chain)):
            _refuse(f"{where}.auditor_fallback_models must be a list of "
                    "non-empty model strings -- declare [] for no fallback, "
                    "or omit the key for the default chain; got "
                    f"{raw_chain!r}")
    # The TARGET decode block's RAW value is type-checked before the dataclass
    # is built: omitting the key is today's harness audit; an explicitly
    # present value must be a mapping over exactly ``mode`` (+ ``policy``
    # under vendor) -- ``null`` / "" / a scalar / a list are refusals, never
    # coerced, and ``seed_base`` is refused by name (a run knob, never a
    # manifest input: PetriTargetDecode docstring).
    td_block: Optional[PetriTargetDecode] = None
    if "target_decode" in block:
        raw_td = block["target_decode"]
        td_where = f"{where}.target_decode"
        if not isinstance(raw_td, Mapping) or not raw_td:
            _refuse(f"{td_where} must be a non-empty mapping "
                    f"{{mode: harness|vendor[, policy: <ref@v>]}} -- omit the "
                    f"key for today's harness audit; got {raw_td!r}")
        if "seed_base" in raw_td:
            _refuse(f"{td_where}: seed_base is a RUN knob (`petri_run.py run "
                    "--target-seed-base`), never a manifest input -- one "
                    "standup serves every seed base; remove it")
        _check_unknown_keys(raw_td, _TARGET_DECODE_MANIFEST_KEYS, td_where)
        if "mode" not in raw_td:
            _refuse(f"{td_where}: missing required key 'mode'")
        td_block = PetriTargetDecode(mode=raw_td["mode"],
                                     policy=raw_td.get("policy"))
    # the judge chain's RAW value, the same rule; here omission and [] both
    # mean no chain (the field's docstring says why they must)
    if "judge_fallback_models" in block:
        raw_judge_chain = block["judge_fallback_models"]
        if (not isinstance(raw_judge_chain, list)
                or not all(isinstance(m, str) and m
                           for m in raw_judge_chain)):
            _refuse(f"{where}.judge_fallback_models must be a list of "
                    "non-empty model strings -- declare [] or omit the key "
                    f"for no judge fallback; got {raw_judge_chain!r}")
    # the per-request refusal fallbacks' RAW values, the judge chain's rule (omission and [] both mean none)
    for rf in ("auditor_refusal_fallback_models", "judge_refusal_fallback_models"):
        if rf in block:
            raw_rf = block[rf]
            if (not isinstance(raw_rf, list)
                    or not all(isinstance(m, str) and m for m in raw_rf)):
                _refuse(f"{where}.{rf} must be a list of non-empty model strings "
                        f"(`anthropic/...`) -- declare [] or omit the key for no "
                        f"refusal fallback; got {raw_rf!r}")
    spec = PetriSpec(**{f: block[f] for f in _dc_field_names(PetriSpec)
                        if f in block and f != "target_decode"},
                     target_decode=td_block)
    # normalize the containers to the declared types before validation
    spec.behaviors = list(spec.behaviors or [])
    spec.cells = list(spec.cells or [])
    spec.serving_launch_flags = list(spec.serving_launch_flags or [])
    # a plain copy: the raw check above guarantees a list (declared, or the
    # dataclass default on omission), so no `or []` coercion may run here
    spec.auditor_fallback_models = list(spec.auditor_fallback_models)
    spec.judge_fallback_models = list(spec.judge_fallback_models)
    spec.auditor_refusal_fallback_models = list(spec.auditor_refusal_fallback_models)
    spec.judge_refusal_fallback_models = list(spec.judge_refusal_fallback_models)
    spec.serving_packages = dict(spec.serving_packages or {})
    spec.prompted_prefix_hashes = dict(spec.prompted_prefix_hashes or {})
    bo_where = f"{where}.bloom_options"
    if not isinstance(spec.bloom_options, Mapping):
        _refuse(f"{bo_where}: must be a mapping of the bloom_audit task "
                "options + epochs + seed (H10)")
    _check_unknown_keys(spec.bloom_options, _dc_field_names(BloomOptions),
                        bo_where)
    missing_bo = [f for f in _dc_field_names(BloomOptions)
                  if f not in spec.bloom_options]
    if missing_bo:
        _refuse(f"{bo_where}: missing required key(s) {missing_bo} (every "
                "bloom_audit option that shapes the elicitation is a stated "
                "harness-hash input — no defaults, H10)")
    spec.bloom_options = BloomOptions(
        **{f: spec.bloom_options[f] for f in _dc_field_names(BloomOptions)})
    validate_petri_spec(spec)
    return spec


# --- The three separated Petri keys -----------------------------------------
#
# 1. petri_intervention_hash — ONLY the defense (artifact + resolved defense
#    spec + prompted-SI bytes). No serving input, so the same intervention
#    hashes identically in-process, via the proxy, or via the HF shim.
# 2. petri_harness_config_hash — everything about the MEASUREMENT, inputs
#    enumerated exhaustively below (H10).
# 3. petri_run_key = H(harness_config_hash, cell_id, intervention_hash) —
#    the run/resume/log_dir key. Any change to harness or intervention
#    yields a fresh namespace; stale completed samples can never be reused.


def petri_intervention_hash(
    resolved_spec: Mapping[str, Any],
    *,
    artifact_bytes: Optional[bytes] = None,
    sidecar_bytes: Optional[bytes] = None,
    si_prefix_bytes: Optional[bytes] = None,
) -> str:
    """The Petri intervention hash, 16-hex.

    Kinds:
    - ``prompted`` — REQUIRES non-empty ``si_prefix_bytes`` (the exact SI
      text bytes; they ARE the intervention); byte-artifact arguments are
      refused. H(si_prefix_bytes || canonical_json(resolved_spec)).
    - ``stacked`` — the prompt+object cell (both legs ARE the
      intervention): REQUIRES ``artifact_bytes`` + ``sidecar_bytes`` for
      the object leg, and a wrapper spec carrying ``si_id`` /
      ``si_prefix_sha16`` / ``object`` (``defense_cells.
      stacked_resolved_spec``); ``si_prefix_bytes``, when given, is
      VERIFIED against the recorded sha rather than hashed raw.
      H(artifact_bytes || sidecar_bytes || canonical_json(wrapper)).
    - ``none`` / ``control:*`` / the capability STEERING_KINDS — delegated
      verbatim to :func:`intervention_hash` (one hashing implementation,
      identical digests across serving paths).
    - any other kind — a TrainTest defense-object kind (the DefenseCell
      registry vocabulary is validated by petri.defense_cells, not here):
      REQUIRES both ``artifact_bytes`` and ``sidecar_bytes`` (pass ``b""``
      explicitly for a missing sidecar — a stated input, never a default);
      H(artifact_bytes || sidecar_bytes || canonical_json(resolved_spec)),
      the same payload layout as the steering kinds.

    Serving identity is deliberately NOT an input — control legs and HF
    exception repeats share their canonical counterpart's hash by
    construction (same intervention, different serving)."""
    kind = resolved_spec.get("kind")
    if not isinstance(kind, str) or not kind:
        _refuse("petri_intervention_hash: resolved_spec has no 'kind'")
    # the sentinel is refused ANYWHERE in the semantic spec — nested fields
    # included (a resolved spec carrying {"scale": "UNFROZEN"} must never
    # receive a stable hash). Raw artifact/sidecar/prompt BYTES are exempt:
    # byte payloads are opaque content, never sentinels.
    _reject_unfrozen_nested(dict(resolved_spec),
                            "petri_intervention_hash resolved_spec")

    if kind == PETRI_PROMPTED_KIND:
        if si_prefix_bytes is None or len(bytes(si_prefix_bytes)) == 0:
            _refuse("petri_intervention_hash: kind 'prompted' requires "
                    "non-empty si_prefix_bytes — the SI text IS the "
                    "intervention (plan §5 H4)")
        if artifact_bytes is not None or sidecar_bytes is not None:
            _refuse("petri_intervention_hash: kind 'prompted' takes no "
                    "artifact/sidecar bytes")
        payload = (bytes(si_prefix_bytes)
                   + canonical_json(dict(resolved_spec)).encode("utf-8"))
        return sha256_hex16(payload)

    if kind == PETRI_STACKED_KIND:
        # BOTH legs are the intervention under stacked routing: the object
        # bytes enter raw; the SI prefix enters via its sha16 INSIDE the
        # wrapper spec, so a worker re-verifying the registry needs no SI
        # text (containment) while the hash still binds the exact prefix.
        # A caller HOLDING the bytes (enumeration, export) must pass them
        # so the recorded sha is checked against the real text — a wrapper
        # spec carrying a wrong/stale sha refuses here rather than keying
        # a lying identity.
        spec = dict(resolved_spec)
        allowed = {"kind", "si_id", "si_prefix_sha16", "object"}
        unknown = sorted(set(spec) - allowed)
        if unknown:
            _refuse(f"petri_intervention_hash: stacked spec carries "
                    f"unknown key(s) {unknown} (allowed: {sorted(allowed)})")
        # the prompt leg IS a system prompt: only a PROMPTABLE SI can be
        # stacked (an object-only pseudo-instruction has no text to hash
        # — the grammar refuses the id; the hash boundary refuses the spec)
        if spec.get("si_id") not in PETRI_PROMPTABLE_SI_IDS:
            _refuse(f"petri_intervention_hash: stacked spec si_id "
                    f"{spec.get('si_id')!r} not in the promptable SIs "
                    f"{PETRI_PROMPTABLE_SI_IDS}")
        sha = spec.get("si_prefix_sha16")
        if not _is_hex16(sha):
            _refuse("petri_intervention_hash: stacked spec needs "
                    f"si_prefix_sha16 (a {HASH_HEX_LEN}-hex hash of the "
                    f"exact SI prefix bytes), got {sha!r}")
        obj = spec.get("object")
        if not isinstance(obj, Mapping) or not obj:
            _refuse("petri_intervention_hash: stacked spec needs a "
                    "non-empty 'object' mapping (the object leg's own "
                    "resolved spec)")
        okind = obj.get("kind")
        if (not isinstance(okind, str) or not okind
                or okind in (NONE_KIND, PETRI_PROMPTED_KIND,
                             PETRI_STACKED_KIND)
                or okind.startswith(CONTROL_KIND_PREFIX)):
            _refuse(f"petri_intervention_hash: stacked object leg has kind "
                    f"{okind!r} — it must be a defense-object kind (never "
                    "none/prompted/control, and never a nested stack)")
        if si_prefix_bytes is not None:
            got = sha256_hex16(bytes(si_prefix_bytes))
            if got != sha:
                _refuse(f"petri_intervention_hash: stacked si_prefix_bytes "
                        f"hash to {got} but the wrapper spec records "
                        f"{sha!r} — the spec does not name the prefix "
                        "actually in hand; refusing to key a lying identity")
        if artifact_bytes is None or sidecar_bytes is None:
            _refuse("petri_intervention_hash: kind 'stacked' requires BOTH "
                    "artifact_bytes and sidecar_bytes for its object leg "
                    "(pass b'' for a missing sidecar — a stated input, "
                    "never a default)")
        payload = (bytes(artifact_bytes) + bytes(sidecar_bytes)
                   + canonical_json(spec).encode("utf-8"))
        return sha256_hex16(payload)

    if si_prefix_bytes is not None:
        _refuse(f"petri_intervention_hash: kind {kind!r} takes no "
                "si_prefix_bytes (only 'prompted' — and 'stacked', as its "
                "verification input — do)")

    if kind == NONE_KIND or kind in STEERING_KINDS or kind.startswith(CONTROL_KIND_PREFIX):
        return intervention_hash(resolved_spec, artifact_bytes=artifact_bytes,
                                 sidecar_bytes=sidecar_bytes)

    # a defense-object kind (the defense-cell vocabulary; same rule as
    # the steering kinds: both byte inputs stated explicitly)
    if artifact_bytes is None or sidecar_bytes is None:
        _refuse(f"petri_intervention_hash: object kind {kind!r} requires "
                "BOTH artifact_bytes and sidecar_bytes (pass b'' for a "
                "missing sidecar — a stated input, never a default)")
    payload = (bytes(artifact_bytes) + bytes(sidecar_bytes)
               + canonical_json(dict(resolved_spec)).encode("utf-8"))
    return sha256_hex16(payload)


def _refuse_unfrozen(value: str, field_name: str) -> None:
    if value == PETRI_UNFROZEN:
        _refuse(f"petri harness hash: {field_name} is {PETRI_UNFROZEN} — "
                "freeze it before keying any run (loud refusal, never a "
                "placeholder hash; plan §5/§6)")


def _reject_unfrozen_nested(obj: Any, where: str) -> None:
    """Recursively refuse the exact UNFROZEN sentinel anywhere inside a
    SEMANTIC hash-input structure (mappings, sequences, strings — keys and
    values alike). Bytes are deliberately exempt: raw artifact/prompt/
    sidecar payloads are opaque content, never subject to string-sentinel
    interpretation."""
    if isinstance(obj, str):
        if obj == PETRI_UNFROZEN:
            _refuse(f"petri hash: {where} carries the {PETRI_UNFROZEN} "
                    "sentinel — freeze it before keying any run (loud "
                    "refusal, never a placeholder hash; plan §5/§6)")
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            _reject_unfrozen_nested(k, f"{where} key {k!r}")
            _reject_unfrozen_nested(v, f"{where}[{k!r}]")
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for i, v in enumerate(obj):
            _reject_unfrozen_nested(v, f"{where}[{i}]")


def _prompt_file_hashes(prompt_files: Any) -> Dict[str, str]:
    """{role: sha256_hex16(bytes)} over the EXACT auditor/judge prompt
    bytes in use — refusing missing/extra roles and empty payloads."""
    if not isinstance(prompt_files, Mapping):
        _refuse("petri harness hash: prompt_files must be a "
                "{role: prompt bytes} mapping over the roles "
                f"{PETRI_PROMPT_ROLES} — the exact auditor/judge prompt "
                "BYTES in use bind the harness hash (plan §5 H10)")
    if set(prompt_files) != set(PETRI_PROMPT_ROLES):
        _refuse(f"petri harness hash: prompt_files must carry exactly the "
                f"roles {PETRI_PROMPT_ROLES}, got {sorted(prompt_files)}")
    out: Dict[str, str] = {}
    for role in PETRI_PROMPT_ROLES:
        data = prompt_files[role]
        if not isinstance(data, (bytes, bytearray, memoryview)) or not bytes(data):
            _refuse(f"petri harness hash: prompt_files[{role!r}] must be "
                    "non-empty bytes — the exact prompt content in use, "
                    "resolved from the local override or the installed "
                    "pinned package (H10)")
        out[role] = sha256_hex16(bytes(data))
    return out


def petri_prompt_override_hash(prompt_files: Mapping[str, bytes]) -> str:
    """The 16-hex value an ``override@<hex>`` prompt provenance DECLARES:
    sha256_hex16 over canonical_json({role: sha256_hex16(bytes)}) of the
    exact auditor and judge prompt bytes. ``petri_harness_config_inputs``
    verifies the declared value against the loaded bytes — an override can
    never run with prompt bytes other than the ones it declares (H10)."""
    return sha256_hex16(
        canonical_json(_prompt_file_hashes(prompt_files)).encode("utf-8"))


def petri_expected_behavior_keys(spec: PetriSpec) -> List[str]:
    """The FULL expected behavior-directory inventory implied by the spec:
    every canonical ``<behavior>`` dir, plus ``<behavior>/<VARIANT>`` for
    every SI whose prompted baseline is in ``spec.cells`` (the SI text
    reaches the audit only through those variant dirs). The
    harness hash requires ``behavior_files`` to carry EXACTLY this key set,
    so an SI-variant BEHAVIOR.md can neither be silently omitted from the
    hash nor smuggled in unrecognized (H10)."""
    in_play_sis = []
    for cell_id in spec.cells:
        if isinstance(cell_id, str) and cell_id.startswith(
                PETRI_PROMPTED_CELL_PREFIX):
            si = cell_id[len(PETRI_PROMPTED_CELL_PREFIX):]
            if si not in PETRI_SI_IDS:
                _refuse(f"petri harness hash: cell {cell_id!r} names unknown "
                        f"prompted SI {si!r} (known: {list(PETRI_SI_IDS)})")
            if si in PETRI_OBJECT_ONLY_SI_IDS:
                _refuse(f"petri harness hash: cell {cell_id!r} prompts an "
                        f"OBJECT-ONLY SI {si!r} (a document-corpus pseudo-"
                        "instruction has no system-prompt text and no "
                        "SI-variant behavior dir)")
            if si not in in_play_sis:
                in_play_sis.append(si)
    keys = list(spec.behaviors)
    for b in spec.behaviors:
        for si in in_play_sis:
            keys.append(f"{b}/{PETRI_SI_VARIANT_DIRS[si]}")
    return sorted(keys)


def petri_harness_config_inputs(
    spec: PetriSpec,
    *,
    behavior_files: Mapping[str, bytes],
    prompt_files: Mapping[str, bytes],
    serving_identity: str,
) -> Dict[str, Any]:
    """The EXHAUSTIVELY ENUMERATED harness_config_hash inputs. Nothing
    that shapes the elicitation or the scoring may be missing — the
    enumerated list is the contract:

    - the exact BEHAVIOR.md evaluation bytes of the FULL behavior-dir
      inventory (:func:`petri_expected_behavior_keys`: every canonical
      ``<behavior>`` plus every in-play SI-variant ``<behavior>/<VARIANT>``
      implied by the spec's prompted cells), keyed by dir and hashed per
      file — a missing required key and an unrecognized key both refuse;
    - the frozen scenario bundle content hash (§6);
    - the auditor / judge / scenarios role model IDs, plus each DECLARED
      (non-empty) fallback chain -- ``auditor_fallbacks`` and
      ``judge_fallbacks`` -- since a chain changes the instrument (an
      empty chain is absent from the inputs, so pre-field keys stand);
    - the auditor/judge PROMPT BYTES actually in use (``prompt_files``,
      hashed per role) plus the provenance label. The caller resolves the
      bytes from wherever the provenance declares — the local override
      files, or the installed pinned package's prompt resources. For an
      ``override@<hex>`` provenance the declared hex is VERIFIED against
      the loaded bytes (:func:`petri_prompt_override_hash`); either way,
      editing a judge or auditor prompt changes this hash, hence
      petri_run_key, hence the Inspect log_dir — the provenance STRING
      alone is not the binding (H10);
    - the bloom_audit task options + epochs + seed (``spec.bloom_options``);
    - the target model + tokenizer revision;
    - the serving package versions + launch flags + chat template hash +
      tool parser + serving implementation commit;
    - the serving identity + determinism mode (§3);
    - the prompted-prefix hashes in play.

    ``serving_identity`` is the PER-CELL identity (its path varies by cell:
    lens / lora / passthrough / apply); its MODE must equal the spec's
    pinned mode — the audit runs only in the mode the gate certified
    (H5 mode binding), so a mismatch is a refusal. UNFROZEN sentinels
    refuse here — named fields first for clear messages, then a recursive
    sweep over the WHOLE input mapping, so no nested field can carry the
    sentinel into a hash. Raw behavior/prompt BYTES are hashed, never
    sentinel-scanned."""
    validate_petri_spec(spec)
    ident = parse_serving_identity(serving_identity)
    if ident.mode != spec.serving_mode:
        _refuse(f"petri harness hash: serving identity {serving_identity!r} "
                f"carries mode {ident.mode!r} but the audit is pinned to "
                f"{spec.serving_mode!r} — the audit runs only in the mode "
                "the gate certified (plan §3 H5)")
    _refuse_unfrozen(spec.scenario_bundle_hash, "scenario_bundle_hash")
    _refuse_unfrozen(spec.chat_template_hash, "chat_template_hash")
    _refuse_unfrozen(spec.serving_impl_commit, "serving_impl_commit")
    for si, h in spec.prompted_prefix_hashes.items():
        _refuse_unfrozen(h, f"prompted_prefix_hashes[{si!r}]")

    if not isinstance(behavior_files, Mapping) or not behavior_files:
        _refuse("petri harness hash: behavior_files must be a non-empty "
                "{dir_key: BEHAVIOR.md bytes} mapping")
    expected_keys = petri_expected_behavior_keys(spec)
    missing_b = sorted(set(expected_keys) - set(behavior_files))
    if missing_b:
        _refuse(f"petri harness hash: no BEHAVIOR.md bytes for required "
                f"behavior dir(s) {missing_b} — the FULL inventory (canonical "
                "+ every in-play SI-variant dir) is a hash input; an omitted "
                "variant would let its evaluation file drift unhashed (H10)")
    extra_b = sorted(set(behavior_files) - set(expected_keys))
    if extra_b:
        _refuse(f"petri harness hash: unrecognized behavior_files key(s) "
                f"{extra_b} — keys must be exactly the expected inventory "
                f"{expected_keys} (canonical <behavior> and in-play "
                "<behavior>/<VARIANT> dirs; H10)")
    for key, data in behavior_files.items():
        if not isinstance(data, (bytes, bytearray, memoryview)) or not bytes(data):
            _refuse(f"petri harness hash: behavior_files[{key!r}] must be "
                    "non-empty bytes (the exact BEHAVIOR.md evaluation bytes)")

    prompt_hashes = _prompt_file_hashes(prompt_files)
    pp = spec.prompt_provenance
    if pp.startswith(_PETRI_PROVENANCE_OVERRIDE):
        declared = pp[len(_PETRI_PROVENANCE_OVERRIDE):]
        actual = sha256_hex16(canonical_json(prompt_hashes).encode("utf-8"))
        if declared != actual:
            _refuse(f"petri harness hash: prompt_provenance declares "
                    f"override@{declared} but the loaded auditor/judge prompt "
                    f"bytes hash to {actual} (petri_prompt_override_hash) — "
                    "an override may never run with prompt bytes other than "
                    "the ones it declares (H10)")

    inputs = {
        "behavior_files": {k: sha256_hex16(bytes(v))
                           for k, v in sorted(behavior_files.items())},
        "scenario_bundle_hash": spec.scenario_bundle_hash,
        "role_models": {
            "auditor": spec.auditor_model,
            # a NON-EMPTY fallback chain changes the instrument and re-keys;
            # an empty chain is ADDITIVE — the input is absent, so a manifest
            # declaring `[]` hashes byte-identically to one that predates the
            # field (the frozen candidate-si-refit run keys)
            **({"auditor_fallbacks": list(spec.auditor_fallback_models)}
               if spec.auditor_fallback_models else {}),
            "judge": spec.judge_model,
            # the judge chain, the identical ADDITIVE rule (2026-09-19): a
            # declared chain re-keys, an empty or omitted one is absent here
            **({"judge_fallbacks": list(spec.judge_fallback_models)}
               if spec.judge_fallback_models else {}),
            # the PER-REQUEST refusal fallbacks (auditor_refusal_fallback_models /
            # judge_refusal_fallback_models) are DELIBERATELY ABSENT: the
            # role's default model is the instrument; a request-level remedy with
            # per-message provenance never re-keys a cell (field docstring)
            "scenarios": spec.scenarios_model,
        },
        "prompt_provenance": spec.prompt_provenance,
        "prompt_files": prompt_hashes,
        "bloom_options": {f.name: getattr(spec.bloom_options, f.name)
                          for f in dc_fields(BloomOptions)},
        "target_model": spec.target_model,
        "target_revision": spec.target_revision,
        "serving_packages": dict(spec.serving_packages),
        "serving_launch_flags": list(spec.serving_launch_flags),
        "chat_template_hash": spec.chat_template_hash,
        "tool_parser": spec.tool_parser,
        "serving_impl_commit": spec.serving_impl_commit,
        "serving_identity": serving_identity,
        "serving_mode": spec.serving_mode,
        "prompted_prefix_hashes": dict(spec.prompted_prefix_hashes),
    }
    # The TARGET decode contract: a VENDOR target is a different
    # instrument and re-keys under its resolved identity (policy id, the
    # six keys AS SENT, the per-turn cap, the seed rule -- NOT the seed
    # base, PetriTargetDecode docstring); the HARNESS mode is ADDITIVE --
    # the key is absent, so a manifest without the block (every existing
    # Petri store) hashes byte-identically to before the field existed
    # (the ``auditor_fallbacks`` pattern above).
    target_identity = petri_target_decode_identity(spec)
    if target_identity is not None:
        inputs["target_decode"] = target_identity
    # the backstop: NOTHING in the final input mapping may carry the
    # sentinel, however deeply nested (auditor_model, a launch flag, …).
    _reject_unfrozen_nested(inputs, "harness inputs")
    return inputs


def petri_harness_config_hash(
    spec: PetriSpec,
    *,
    behavior_files: Mapping[str, bytes],
    prompt_files: Mapping[str, bytes],
    serving_identity: str,
) -> str:
    """16-hex sha256 over canonical_json of
    :func:`petri_harness_config_inputs`."""
    inputs = petri_harness_config_inputs(
        spec, behavior_files=behavior_files, prompt_files=prompt_files,
        serving_identity=serving_identity)
    return sha256_hex16(canonical_json(inputs).encode("utf-8"))


def petri_run_key(harness_config_hash: str, cell_id: str,
                  intervention_hash: str) -> str:
    """The composite run/resume key:
    H(harness_config_hash, cell_id, intervention_hash), 16-hex. Keys the
    store cell content-addressing, the resume ledger, and the Inspect
    log_dir namespace — any change to harness or intervention yields
    a fresh, empty namespace."""
    for label, h in (("harness_config_hash", harness_config_hash),
                     ("intervention_hash", intervention_hash)):
        if not _is_hex16(h):
            _refuse(f"petri_run_key: {label} must be a {HASH_HEX_LEN}-hex "
                    f"hash, got {h!r}")
    if not isinstance(cell_id, str) or not cell_id:
        _refuse("petri_run_key: cell_id must be a non-empty string")
    if cell_id == PETRI_UNFROZEN:
        _refuse(f"petri_run_key: cell_id is the {PETRI_UNFROZEN} sentinel — "
                "never a cell id; no run is ever keyed on an unfrozen input "
                "(plan §5/§6)")
    payload = canonical_json({
        "cell_id": cell_id,
        "harness_config_hash": harness_config_hash,
        "intervention_hash": intervention_hash,
    }).encode("utf-8")
    return sha256_hex16(payload)
