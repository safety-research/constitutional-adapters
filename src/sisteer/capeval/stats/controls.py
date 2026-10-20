"""Control-arm constructors + the control acceptance checker.

Five controls, all on the anchor model, HF engine, manifested as
``kind: control:<name>`` with ``seed`` + ``params`` hashed (no artifact
bytes — manifest.intervention_hash's control branch):

1. ``control:zero_coeff`` — a REAL S2 artifact (``params.artifact`` is a
   real path) loaded through the full ``load_object`` path, applied at
   coeff 0.0: tests the whole apply/ctx/hook machinery is a no-op at
   zero. The engine refuses any other coeff for this kind.
2. ``control:sham_hook`` — a forward hook on ``params.layer`` that
   returns the output unchanged (exercises the registration/dtype path
   without the steering math).
3. ``control:random_vector`` — a seeded Gaussian direction at
   ``params.layer``, rescaled to ``params.norm_target`` (the median
   layer-norm of the rediscovery vectors, computed from their JSON
   sidecars via :func:`median_sidecar_norm` — no GPU needed). The
   manifest coeff ladder multiplies that norm.
4. ``control:noise_lora`` — hand-rolled rank-``params.rank`` (default
   16) ΔW = α·B·Aᵀ on ``o_proj`` of EVERY decoder layer, A/B seeded
   Gaussian, α set per layer so ‖ΔW‖_F/‖W‖_F = ε; the manifest coeff IS
   ε. In-place weight add with exact restore from pinned CPU copies.
5. ``control:weight_noise`` — ε·N(0,1)·std(W) added to every 2-D weight
   matrix (single rung, known broad damage); coeff is ε; seeded,
   exact-restore.

Every constructor is deterministic in (seed, params): the same seed
rebuilds bit-identical perturbations (a fresh ``torch.Generator`` per
tensor, keyed on seed + a stable per-tensor index). Weight-mutating
controls build every perturbation ONCE at CONSTRUCTION and cache it —
along with original-weight copies — on the parameter's own device
(score_corpus enters a fresh ctx per item forward, so per-apply
regeneration/CPU-GPU transfer of full-model tensors is terabytes of
waste over a real corpus); every context exit restores bit-exactly
from the cached originals; the driver additionally isolates such arms
one-arm-per-process (§11.1).

The acceptance checker consumes per-arm
fingerprint-style observations: no-ops must sit within the calibrated
null, trip nothing, and show EXACTLY-ZERO coarsened KL (the
exact-zero certificate — a no-op through the full serving path replays
the base logits bit-identically, so any nonzero KL indicts the HARNESS;
hard-stop); damage ladders must be DETECTABLE at their top rung by
Tier-0 evidence — a Tier-0 domain harm beyond its null band or a
confirmed canary failure. KL magnitudes are annotations and are NEVER
detection evidence (a perturbation whose only footprint is KL is
sub-behavioral — not damage). Monotonicity across rungs is deliberately
NOT required.
"""

from __future__ import annotations

import contextlib
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from sisteer.capeval.manifest import AblationSpec, CONTROL_KIND_PREFIX
from sisteer.methods import VectorObject, load_object
from sisteer.models import get_decoder_layers, hidden_size
from sisteer.vectors import SteeringVector


def _site_path(value, key: str) -> Path:
    """A manifest path with any site placeholder (``$SHARED/...``) expanded through ``sisteer.site``;
    an unset one is a ControlError naming it, never a literal ``$SHARED`` path."""
    from sisteer import site as _site
    try:
        return _site.expand_path(value, key=key)
    except _site.PlaceholderError as exc:
        raise ControlError(str(exc)) from None


class ControlError(ValueError):
    """A control-arm construction/acceptance refusal."""


CONTROL_KINDS = (
    "control:zero_coeff",
    "control:sham_hook",
    "control:random_vector",
    "control:noise_lora",
    "control:weight_noise",
)
NOOP_KINDS = ("control:zero_coeff", "control:sham_hook")
DAMAGE_KINDS = ("control:random_vector", "control:noise_lora",
                "control:weight_noise")

# distinct odd multipliers decorrelate per-tensor generator seeds
_SEED_STRIDE = 100_003


# ---------------------------------------------------------------------------
# constructors
# ---------------------------------------------------------------------------


def build_control(spec: AblationSpec, model, tok) -> Any:
    """Dispatch one manifested control arm to its constructor. The
    returned object speaks the sisteer object protocol the engine
    wraps: ``apply(model, scale)`` (a FRESH context manager per call)
    and ``inject`` (identity — no control is a token kind)."""
    if not spec.kind.startswith(CONTROL_KIND_PREFIX):
        raise ControlError(f"build_control got non-control kind {spec.kind!r}")
    if spec.kind not in CONTROL_KINDS:
        raise ControlError(f"unknown control kind {spec.kind!r} — v0 builds "
                           f"{CONTROL_KINDS}")
    if spec.seed is None:
        raise ControlError(f"control arm {spec.name!r} requires a seed "
                           "(hashed into the intervention hash, §2.1)")
    params = dict(spec.params)
    # tier1_coeffs is an ENUMERATION-side key (print-cells picks which
    # rungs run Tier 1 from it); construction ignores it. It
    # stays in spec.params for hashing — popping the local copy only.
    params.pop("tier1_coeffs", None)

    if spec.kind == "control:zero_coeff":
        artifact = params.pop("artifact", None)
        # declarative in the manifest: the artifact's sidecar is the
        # authoritative layer source; the param is documentation/hashing
        params.pop("layer", None)
        _refuse_unknown(spec, params)
        if not artifact:
            raise ControlError(
                f"{spec.name!r}: control:zero_coeff requires params.artifact "
                "— a REAL artifact path, so the zero-dose no-op exercises the "
                "exact load/apply path the real arms use (§8)")
        artifact = _site_path(artifact, f"{spec.name} params.artifact")
        if not artifact.is_file():
            raise ControlError(f"{spec.name!r}: artifact does not exist: "
                               f"{artifact}. The shipped manifest names a "
                               "recorded stage-2 vector the repository cannot "
                               "rebuild (its instruction set was removed); any "
                               "saved steering object stands in, since the "
                               "zero dose only exercises the load/apply path: "
                               "point params.artifact at one of yours")
        return load_object(artifact, model, tok)

    if spec.kind == "control:sham_hook":
        layer = params.pop("layer", None)
        _refuse_unknown(spec, params)
        if layer is None:
            raise ControlError(f"{spec.name!r}: control:sham_hook requires "
                               "params.layer")
        return ShamHookObject(int(layer))

    if spec.kind == "control:random_vector":
        layer = params.pop("layer", None)
        norm_target = params.pop("norm_target", None)
        norm_spec = params.pop("norm", None)
        d = params.pop("d", None)
        _refuse_unknown(spec, params)
        if norm_target is None and norm_spec is not None:
            # the manifest's declarative form (§8): resolve the target
            # from the named steering-vector sidecars' stored norms
            source = norm_spec.get("source")
            if source != "median_sidecar_norm":
                raise ControlError(
                    f"{spec.name!r}: unknown params.norm.source {source!r} "
                    "(only 'median_sidecar_norm' is implemented)")
            norm_target = median_sidecar_norm(
                norm_spec["sidecars"], layer=norm_spec.get("layer", layer))
        if layer is None or norm_target is None:
            raise ControlError(f"{spec.name!r}: control:random_vector "
                               "requires params.layer and either "
                               "params.norm_target or params.norm")
        return random_vector_object(model, seed=int(spec.seed),
                                    layer=int(layer),
                                    norm_target=float(norm_target),
                                    d=int(d) if d is not None else None)

    if spec.kind == "control:noise_lora":
        rank = int(params.pop("rank", 16))
        module = params.pop("module", "o_proj")
        layers = params.pop("layers", "all")
        _refuse_unknown(spec, params)
        if module != "o_proj" or layers != "all":
            # declarative in the manifest; only the §8 configuration is
            # implemented — refuse anything else rather than mis-build
            raise ControlError(
                f"{spec.name!r}: control:noise_lora supports only "
                f"module=o_proj, layers=all (got {module!r}, {layers!r})")
        return NoiseLoRAObject(model, rank=rank, seed=int(spec.seed))

    # control:weight_noise
    _refuse_unknown(spec, params)
    return WeightNoiseObject(model, seed=int(spec.seed))


def _refuse_unknown(spec: AblationSpec, leftover: Mapping[str, Any]) -> None:
    if leftover:
        raise ControlError(f"control arm {spec.name!r}: unknown params "
                           f"{sorted(leftover)}")


def median_sidecar_norm(sidecar_paths: Sequence[Any],
                        layer: Optional[int] = None) -> float:
    """Median steering-vector norm across artifact JSON sidecars — the
    §8 norm-matching input for control:random_vector, computed from the
    sidecars' stored ``norms`` (no GPU, no artifact bytes). ``layer``
    picks that layer's norm from multi-layer sidecars; a single-norm
    sidecar needs no layer."""
    norms: List[float] = []
    for p in sidecar_paths:
        p = _site_path(p, "params.norm.sidecars")
        try:
            sidecar = json.loads(p.read_text(encoding="utf-8"))
        except OSError as e:
            raise ControlError(
                f"{p}: cannot read this norm sidecar ({type(e).__name__}). The "
                "shipped manifest names recorded stage-2 sidecars that no "
                "included code can produce (their instruction set was removed); "
                "set params.norm_target to a norm, or point params.norm.sidecars "
                "at sidecars of your own objects") from e
        stored = sidecar.get("norms")
        if stored is None and "norm" in sidecar:  # single-norm sidecar
            norms.append(float(sidecar["norm"]))
            continue
        if not isinstance(stored, Mapping) or not stored:
            raise ControlError(f"{p}: sidecar has no usable 'norms' mapping "
                               "(or scalar 'norm')")
        if layer is not None:
            key = str(layer)
            if key not in stored:
                raise ControlError(f"{p}: no stored norm for layer {layer} "
                                   f"(has {sorted(stored)})")
            norms.append(float(stored[key]))
        elif len(stored) == 1:
            norms.append(float(next(iter(stored.values()))))
        else:
            raise ControlError(f"{p}: multi-layer sidecar needs an explicit "
                               "layer argument")
    if not norms:
        raise ControlError("median_sidecar_norm: no sidecars given")
    return float(statistics.median(norms))


def random_vector_object(model, *, seed: int, layer: int,
                         norm_target: float, d: Optional[int] = None
                         ) -> VectorObject:
    """A seeded Gaussian direction rescaled to ``norm_target``, wrapped
    in VectorObject so it runs through EXACTLY the real vector arms'
    apply path (hooks.steering, skip_bos=True S2 semantics)."""
    dim = d if d is not None else hidden_size(model)
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g, dtype=torch.float32)
    v = v / v.norm() * norm_target
    sv = SteeringVector(vectors={layer: v}, method="control_random_vector",
                        model_name=getattr(model.config, "name_or_path", ""),
                        instruction_id="control",
                        metadata={"seed": seed, "norm_target": norm_target})
    return VectorObject(sv)


class ShamHookObject:
    """A forward hook on one decoder layer that returns the output
    unchanged — rebuilt (registered/removed) FRESH per apply() call."""

    kind = "control:sham_hook"

    def __init__(self, layer: int):
        self.layer = layer

    def inject(self, rendered_prompt: str) -> str:
        return rendered_prompt

    @contextlib.contextmanager
    def apply(self, model, scale: float = 1.0):
        layers = get_decoder_layers(model)
        if not (0 <= self.layer < len(layers)):
            raise ControlError(f"sham_hook layer {self.layer} out of range "
                               f"(model has {len(layers)} layers)")

        def hook(module, args, output):
            # reconstruct the output container around the SAME tensor:
            # exercises the replace path bit-identically
            if isinstance(output, tuple):
                return (output[0],) + output[1:]
            return output

        h = layers[self.layer].register_forward_hook(hook)
        try:
            yield
        finally:
            h.remove()


class _WeightMutatingControl:
    """Shared cache-once/exact-restore machinery.

    At CONSTRUCTION, per touched tensor: an original-weight copy and the
    unit-scale (ε=1) float32 perturbation, both cached on the tensor's
    own device. Every apply() then only does device-local arithmetic —
    ``w += (scale · Δ_unit).cast(dtype)`` on entry, a bit-exact
    ``copy_`` from the cached original on exit (crash-safe via
    try/finally). Nothing is regenerated and nothing crosses the
    CPU/GPU boundary per forward — score_corpus enters a fresh ctx per
    item, so per-apply regeneration was the r1 BLOCKER."""

    def __init__(self, model, touched_names: Sequence[str]):
        params = dict(model.named_parameters())
        missing = [n for n in touched_names if n not in params]
        if missing:
            raise ControlError(f"{type(self).__name__}: parameters not found "
                               f"on the model: {missing[:3]}...")
        self._touched = list(touched_names)
        with torch.no_grad():
            self._originals = {
                n: params[n].detach().clone() for n in self._touched}
            self._deltas = {
                n: self._build_unit_delta(i, params[n])
                for i, n in enumerate(self._touched)}

    def inject(self, rendered_prompt: str) -> str:
        return rendered_prompt

    def _build_unit_delta(self, index: int, w: torch.Tensor) -> torch.Tensor:
        """The ε=1 float32 perturbation for one tensor, on w's device.
        Seeded on (seed, index) — construction-time only."""
        raise NotImplementedError  # pragma: no cover

    def _mutate(self, model, scale: float) -> None:
        if scale < 0:
            raise ControlError(f"{type(self).__name__}: ε must be >= 0")
        if scale == 0:
            return
        params = dict(model.named_parameters())
        for n in self._touched:
            w = params[n]
            w.data.add_((scale * self._deltas[n]).to(dtype=w.dtype))

    @contextlib.contextmanager
    def apply(self, model, scale: float = 1.0):
        params = dict(model.named_parameters())
        try:
            with torch.no_grad():
                self._mutate(model, float(scale))
            yield
        finally:
            with torch.no_grad():
                for n in self._touched:
                    params[n].data.copy_(self._originals[n])


def _o_proj_names(model) -> List[str]:
    layers = get_decoder_layers(model)
    names = []
    for i, layer in enumerate(layers):
        o_proj = getattr(getattr(layer, "self_attn", None), "o_proj", None)
        if o_proj is None or not hasattr(o_proj, "weight"):
            raise ControlError(
                f"control:noise_lora: decoder layer {i} has no "
                "self_attn.o_proj.weight — the v0 control is defined on the "
                "Llama-family anchor (§8); extend deliberately for other "
                "architectures")
        names.append(_param_name(model, o_proj.weight))
    return names


def _param_name(model, param: torch.Tensor) -> str:
    for n, p in model.named_parameters():
        if p is param:
            return n
    raise ControlError("parameter not found on model")


class NoiseLoRAObject(_WeightMutatingControl):
    """Rank-r ΔW = α·B·Aᵀ on o_proj of every decoder layer with
    Frobenius targeting: α_i = ε·‖W_i‖_F / ‖B_i·A_iᵀ‖_F, so that
    ‖ΔW_i‖_F/‖W_i‖_F = ε exactly (in float32) on every layer. ε is the
    apply() scale — the manifest coeff ladder IS the ε ladder (§8).
    ΔW at ε=1 is built once at construction and cached on-device."""

    kind = "control:noise_lora"

    def __init__(self, model, rank: int = 16, seed: int = 0):
        self.rank = int(rank)
        self.seed = int(seed)
        super().__init__(model, _o_proj_names(model))

    def _build_unit_delta(self, index: int, w: torch.Tensor) -> torch.Tensor:
        d_out, d_in = w.shape
        g = torch.Generator().manual_seed(self.seed * _SEED_STRIDE + index)
        a = torch.randn(d_in, self.rank, generator=g, dtype=torch.float32)
        b = torch.randn(d_out, self.rank, generator=g, dtype=torch.float32)
        delta = b @ a.T
        alpha_unit = float(w.detach().float().norm()) / float(delta.norm())
        return (alpha_unit * delta).to(device=w.device)


class WeightNoiseObject(_WeightMutatingControl):
    """ε·N(0,1)·std(W) added to every 2-D weight matrix (§8 rung 5,
    known broad damage). ε is the apply() scale; per-tensor seeded
    generators keyed on the SORTED parameter-name index, so iteration
    order can never change the noise. N(0,1)·std(W) is built once at
    construction and cached on-device."""

    kind = "control:weight_noise"

    def __init__(self, model, seed: int = 0):
        self.seed = int(seed)
        touched = sorted(n for n, p in model.named_parameters() if p.dim() == 2)
        if not touched:
            raise ControlError("weight_noise: model has no 2-D weight matrices")
        super().__init__(model, touched)

    def _build_unit_delta(self, index: int, w: torch.Tensor) -> torch.Tensor:
        std = float(w.detach().float().std())
        g = torch.Generator().manual_seed(self.seed * _SEED_STRIDE + index)
        noise = torch.randn(w.shape, generator=g, dtype=torch.float32)
        return (std * noise).to(device=w.device)


# ---------------------------------------------------------------------------
# acceptance checker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlObservation:
    """One control arm's harness readout, assembled by the report step
    from the fingerprint JSONs (synthetic in tests).

    ``domain_harms`` are harm-oriented deltas (positive = worse) per
    registry domain; ``kl_coarse_lb_max`` is the largest per-corpus
    coarsened-KL mean (nats/pos; None when the rung filed no KL rows) —
    the no-op exact-zero certificate input, and otherwise an
    annotation; ``canary_tripped``/``canary_confirmed`` come from the
    §4.4 rows.

    ``tier0_domain_harms`` is the Tier-0-only slice of ``domain_harms``:
    damage-control ACCEPTANCE evidence is restricted to it (plus the
    confirmed canary; KL is never evidence), so Tier-1 damage can never
    satisfy the required Tier-0 detection. The full map still feeds the
    no-op within-null check (stricter is safe there)."""

    arm: str
    kind: str
    coeff: Optional[float]
    domain_harms: Dict[str, float]
    canary_tripped: int = 0
    canary_confirmed: bool = False
    kl_coarse_lb_max: Optional[float] = None
    tier0_domain_harms: Dict[str, float] = field(default_factory=dict)
    # Tier-0 damage scalars for the dose-response figure (report §12),
    # e.g. {"kl_coarse_lb": <mean nats/pos over neutral corpora>} —
    # display payload only, never an acceptance input (integrator,
    # item-7 shape freeze).
    tier0_damage: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ControlArmResult:
    arm: str
    kind: str
    role: str  # "noop" | "damage"
    passed: bool
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class ControlReport:
    """The §8 pass/fail control report report/build.py consumes.
    ``hard_stop`` means a NO-OP failed: the harness indicts itself and
    the sweep must not proceed (§8)."""

    passed: bool
    hard_stop: bool
    arms: Tuple[ControlArmResult, ...]

    def to_json(self) -> str:
        return json.dumps({
            "passed": self.passed,
            "hard_stop": self.hard_stop,
            "arms": [{"arm": a.arm, "kind": a.kind, "role": a.role,
                      "passed": a.passed, "reasons": list(a.reasons)}
                     for a in self.arms],
        }, indent=1, sort_keys=True)


def observations_from_store(
    control_df: Any,
    ref_df: Any,
    registry: Any,
    *,
    model: str,
    tier0_domains: Sequence[str],
    b: int = 1000,
    seed: int = 0,
) -> List[ControlObservation]:
    """Assemble ControlObservations from a CONTROL sweep's consolidated
    rows (one observation per (arm, coeff) rung), paired against the
    calibration trial-0 reference rows:

    * domain_harms — point-estimate harms from the registered §7.2
      aggregation (stats.noninferiority.statuses_for_arm; the bootstrap
      CI bounds are computed but only the point estimate feeds the
      acceptance check, so a small ``b`` is fine).
    * tier0_domain_harms — the ``tier0_domains`` slice of domain_harms.
      The filter is an EXPLICIT parameter so the caller names which
      domains are Tier-0 evidence — never inferred here, never allowed
      to drift into counting Tier-1 harm as Tier-0 detection.
    * kl_coarse_lb_max — max per-corpus kl_coarse_lb_corpus mean (the
      no-op exact-zero certificate input; None without KL rows). Rows
      with a stored flag_kl_gt_kappa are deliberately IGNORED — the κ
      flag is retired.
    * canary_tripped / canary_confirmed — the §4.4 rows
      (canary/tripped_count at attempt 0; flag_canary_confirmed).
    * tier0_damage — {"kl_coarse_lb": mean of kl_coarse_lb_corpus} per
      rung, the dose-response display series.
    """
    from sisteer.capeval.stats.noninferiority import statuses_for_arm

    tier0_set = set(tier0_domains)
    registered = {d.name for d in registry.domains}
    unknown = sorted(tier0_set - registered)
    if unknown:
        raise ControlError(
            f"observations_from_store: tier0_domains {unknown} are not in "
            "the registry — refusing a filter that silently matches nothing")
    ctl = control_df[control_df["kind"].astype(str).str.startswith(
        CONTROL_KIND_PREFIX)]
    if ctl.empty:
        raise ControlError("observations_from_store: no control:* rows in "
                           "the sweep frame")
    obs: List[ControlObservation] = []
    for (arm, coeff, kind), g in sorted(
            ctl.groupby(["arm_name", "coeff", "kind"], dropna=False),
            key=lambda kv: (kv[0][0], kv[0][1])):
        a0 = g[g["attempt"] == 0]
        # allow_arm_task_subset: tier-0-only rungs legitimately carry
        # none of the Tier-1 domains' tasks while the trial-0 reference
        # carries all of them — those domains are simply not measured in
        # the rung (pair_domain seam, integrator 2026-08-15). Real-arm
        # paths keep the strict default.
        statuses = statuses_for_arm(a0, ref_df, registry, model=model,
                                    b=b, seed=seed,
                                    allow_arm_task_subset=True)
        harms = {name: st.harm for name, st in statuses.items()}
        kl_corpus = a0[a0["metric"] == "kl_coarse_lb_corpus"]["value"]
        tier0_damage = ({"kl_coarse_lb": float(kl_corpus.mean())}
                        if len(kl_corpus) else {})
        tripped = a0[a0["metric"] == "canary/tripped_count"]["value"]
        confirmed = bool((g["metric"] == "flag_canary_confirmed").any())
        obs.append(ControlObservation(
            arm=str(arm), kind=str(kind), coeff=float(coeff),
            domain_harms=harms,
            canary_tripped=int(tripped.max()) if len(tripped) else 0,
            canary_confirmed=confirmed,
            kl_coarse_lb_max=(float(kl_corpus.max()) if len(kl_corpus)
                              else None),
            tier0_domain_harms={n: h for n, h in harms.items()
                                if n in tier0_set},
            tier0_damage=tier0_damage))
    return obs


def null_bands_from_calibration(calibration: Any) -> Dict[str, float]:
    """Per-domain |Δ| acceptance band (§8 "within the calibrated p99 null
    band"): the symmetric envelope max(|p01|, |p99|) of the calibrated
    null-contrast tails — conservative for asymmetric nulls."""
    bands = {}
    for name, dc in calibration.domains.items():
        try:
            bands[name] = max(abs(dc.tails["p01"]), abs(dc.tails["p99"]))
        except KeyError as e:
            raise ControlError(
                f"calibration domain {name!r} lacks tail {e} — rebuild "
                "calibration.json (stats.calibrate)") from e
    return bands


def controls_acceptance_payload(
    observations: Sequence[ControlObservation],
    report: ControlReport,
) -> Dict[str, Any]:
    """The controls_acceptance.json payload — THE one frozen shape both
    sides use (integrator item 7, 2026-08-15): report/build.py's
    dose_response_figure reads ``arms.<name>.coeffs`` +
    ``arms.<name>.tier0_damage.{metric: [per-coeff values]}``, and the
    acceptance readout reads ``arms.<name>.acceptance`` ("pass"/"fail")
    + ``reasons``. Ladders are ordered by ascending coeff; single-rung
    arms carry one-element lists. Serialize with json.dumps and write to
    <sweep_dir>/controls_acceptance.json."""
    by_arm: Dict[str, List[ControlObservation]] = {}
    for o in observations:
        by_arm.setdefault(o.arm, []).append(o)
    results_by_arm: Dict[str, List[ControlArmResult]] = {}
    for r in report.arms:
        results_by_arm.setdefault(r.arm, []).append(r)
    arms: Dict[str, Any] = {}
    for name, obs in sorted(by_arm.items()):
        obs = sorted(obs, key=lambda o: (o.coeff is not None, o.coeff))
        rs = results_by_arm.get(name, [])
        if not rs:
            raise ControlError(f"controls_acceptance_payload: arm {name!r} "
                               "has observations but no check_controls result")
        metrics = sorted({m for o in obs for m in o.tier0_damage})
        arms[name] = {
            "kind": obs[0].kind,
            "role": rs[0].role,
            "acceptance": "pass" if all(r.passed for r in rs) else "fail",
            "reasons": sorted({reason for r in rs for reason in r.reasons}),
            "coeffs": [o.coeff for o in obs],
            "tier0_damage": {
                m: [o.tier0_damage.get(m) for o in obs] for m in metrics},
        }
    return {"passed": report.passed, "hard_stop": report.hard_stop,
            "arms": arms}


def check_controls(
    observations: Sequence[ControlObservation],
    null_bands: Mapping[str, float],
    top_rungs: Mapping[str, float],
    *,
    required_kinds: Sequence[str] = CONTROL_KINDS,
) -> ControlReport:
    """Evaluate control acceptance, mechanically:

    * NO-OPS (zero_coeff, sham_hook): every domain harm within the
      calibrated null band (|Δ| <= p99 band, over ALL reported domains —
      stricter is safe for a no-op), ZERO canary indicators tripped,
      and EXACTLY-ZERO coarsened KL when KL rows exist (the exact-zero
      certificate: a genuine no-op replays the base logits
      bit-identically). Any failure is harness-indicting: ``hard_stop``
      is set.
    * DAMAGE families (random_vector, noise_lora, weight_noise): the TOP
      RUNG — the coeff PINNED in ``top_rungs[kind]``, from the manifest,
      never re-derived from whatever cells happen to exist — must be
      DETECTABLE at Tier 0: some TIER-0 domain harm
      (``tier0_domain_harms``) beyond its null band, OR a confirmed
      canary failure. Tier-1 harm is never acceptance evidence, and
      neither is a KL magnitude (KL is an annotation; a perturbation
      whose only footprint is KL is sub-behavioral, not damage).
      Dose-response monotonicity is NOT required (§7.3): sub-top rungs
      may sit anywhere without failing.

    Completeness is required: every kind in
    ``required_kinds`` (default: the full v0 §8 set) must have
    observations, and every required damage family must have a pinned
    top rung WITH an observation at it — an incomplete or partially
    consolidated sweep is a refusal, never a pass.

    A domain in an observation with no calibrated band is a refusal —
    silently skipping it would let damage hide in an unbanded domain."""
    if not observations:
        raise ControlError("check_controls: no observations")
    required = list(required_kinds)
    stray_required = sorted(set(required) - set(CONTROL_KINDS))
    if stray_required:
        raise ControlError(f"check_controls: required_kinds {stray_required} "
                           f"are not control kinds (v0 builds {CONTROL_KINDS})")
    present = {o.kind for o in observations}
    absent = sorted(set(required) - present)
    if absent:
        raise ControlError(
            f"check_controls: required control kind(s) {absent} have no "
            "observations — an incomplete controls sweep cannot pass "
            "acceptance #1 (r1 finding 7); consolidate every manifested "
            "cell first")
    results: List[ControlArmResult] = []
    hard_stop = False

    for obs in observations:
        unbanded = sorted(
            (set(obs.domain_harms) | set(obs.tier0_domain_harms))
            - set(null_bands))
        if unbanded:
            raise ControlError(
                f"arm {obs.arm!r}: domains {unbanded} have no calibrated "
                "null band — refusing to evaluate around them")

    for obs in (o for o in observations if o.kind in NOOP_KINDS):
        reasons = []
        for dom, harm in sorted(obs.domain_harms.items()):
            if abs(harm) > null_bands[dom]:
                reasons.append(f"{dom}: |Δ|={abs(harm):.4g} exceeds the "
                               f"calibrated null band {null_bands[dom]:.4g}")
        if obs.canary_tripped:
            reasons.append(f"canary tripped {obs.canary_tripped} indicator(s) "
                           "on a no-op")
        if obs.kl_coarse_lb_max is not None and obs.kl_coarse_lb_max > 0.0:
            reasons.append(f"kl_coarse_lb {obs.kl_coarse_lb_max:.4g} nats/pos "
                           "nonzero on a no-op — the exact-zero certificate "
                           "(D22) requires bit-identical base replay")
        passed = not reasons
        if not passed:
            hard_stop = True  # a no-op failure indicts the harness (§8)
        results.append(ControlArmResult(obs.arm, obs.kind, "noop", passed,
                                        tuple(reasons)))

    for kind in DAMAGE_KINDS:
        family = [o for o in observations if o.kind == kind]
        if not family:
            continue  # not manifested AND not required (checked above)
        if any(o.coeff is None for o in family):
            raise ControlError(f"{kind}: damage-ladder observations must "
                               "carry a coeff")
        if not top_rungs or kind not in top_rungs:
            raise ControlError(
                f"{kind}: no pinned top rung — top_rungs must name every "
                "damage family's top coefficient from the MANIFEST; deriving "
                "it from the observed cells would let a failed/missing top "
                "cell redefine a lower rung as 'top' and pass (r1 finding 7)")
        top_coeff = top_rungs[kind]
        tops = [o for o in family if o.coeff == top_coeff]
        if not tops:
            raise ControlError(
                f"{kind}: pinned top rung {top_coeff} has no observation "
                f"(coeffs seen: {sorted(o.coeff for o in family)})")
        top = tops[0]
        evidence = []
        # Tier-0 evidence ONLY: tier0_domain_harms + the confirmed
        # canary — a Tier-1 harm must never satisfy the check, and
        # neither may a KL magnitude (annotation, not damage).
        for dom, harm in sorted(top.tier0_domain_harms.items()):
            if harm > null_bands[dom]:
                evidence.append(f"{dom}: harm {harm:.4g} beyond null band "
                                f"{null_bands[dom]:.4g}")
        if top.canary_confirmed:
            evidence.append("confirmed canary failure")
        passed = bool(evidence)
        reasons = (tuple(evidence) if passed else
                   (f"no Tier-0 damage detectable at the top rung "
                    f"(coeff {top_coeff}) — the harness cannot see known "
                    "damage (§8)",))
        for o in family:
            role_pass = passed if o is top else True  # sub-rungs never fail
            results.append(ControlArmResult(
                o.arm, o.kind, "damage", role_pass,
                reasons if o is top else ("sub-top rung: no requirement "
                                          "(monotonicity not required)",)))

    known = set(NOOP_KINDS) | set(DAMAGE_KINDS)
    stray = [o.arm for o in observations if o.kind not in known]
    if stray:
        raise ControlError(f"check_controls: observations with non-control "
                           f"kinds: {stray}")

    return ControlReport(passed=all(r.passed for r in results),
                         hard_stop=hard_stop, arms=tuple(results))
