"""Target-metric plugins. v0 ships ``sisteer_compliance``.

A target plugin scores ONE leg (base when ``arm is None``, else the arm)
of a target primary on a fixed prompt slice, returning per-item values
the driver files as store rows (domain = store.TARGET_DOMAIN — the
target metric is DESIGN §2.1's own axis, never a capability domain).
Pairing, deltas, CIs, and the established/excluded/indeterminate call
are stats-side (stats.fingerprint.target_status), per-item by item_id
against the same slice's base leg — the base leg is generated once per
sweep (shared cell, §9) because generations are SI-independent; only the
checkers differ.

Registry contract: plugins are looked up by ``PrimarySpec.plugin`` via
``get_target_plugin`` — an unknown name is a refusal listing what is
registered. The plugin callable signature is::

    plugin(engine, arm, primary, prompts, *, gen_texts=None) -> TargetLegScores

- ``engine``/``arm``: the §2.4 Engine and an ArmHandle (None = base leg).
- ``prompts``: the resolved prompt slice (corpora.CorpusSlice).
- ``gen_texts``: precomputed generations for this leg (the shared base
  cell hands its texts to every SI's scoring pass instead of
  regenerating); when given, the engine is never called.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from sisteer.capeval.corpora import CorpusSlice
from sisteer.capeval.manifest import Manifest, PrimarySpec, TargetSpec


class TargetError(ValueError):
    """A target-plugin refusal (unknown plugin, bad params, …)."""


# Generation config every v0 target plugin uses (greedy, chat-templated
# by the engine — the engine owns the tokenizer). Keys are the §2.4
# gen_cfg contract shared with the canary path.
def default_gen_cfg(max_new_tokens: int) -> Dict[str, Any]:
    return {"max_new_tokens": int(max_new_tokens), "greedy": True,
            "chat": True}


@dataclass(frozen=True)
class TargetLegScores:
    """Per-item scores for one leg of one target primary."""

    metric: str
    plugin: str
    arm_name: str  # "none" for the base leg
    item_ids: Tuple[str, ...]
    values: Tuple[float, ...]
    params_resolved: Dict[str, Any]  # archived into the cell manifest

    def __post_init__(self) -> None:
        if len(self.item_ids) != len(self.values):
            raise TargetError("TargetLegScores: item_ids/values length mismatch")


TargetPlugin = Callable[..., TargetLegScores]

_PLUGINS: Dict[str, TargetPlugin] = {}


def register_target(name: str) -> Callable[[TargetPlugin], TargetPlugin]:
    def deco(fn: TargetPlugin) -> TargetPlugin:
        if name in _PLUGINS:
            raise TargetError(f"target plugin {name!r} registered twice")
        _PLUGINS[name] = fn
        return fn
    return deco


def get_target_plugin(name: str) -> TargetPlugin:
    if name not in _PLUGINS:
        raise TargetError(f"unknown target plugin {name!r}; registered: "
                          f"{sorted(_PLUGINS)}")
    return _PLUGINS[name]


def resolve_target(manifest: Manifest) -> list:
    """The mode/target contract (F20), re-checked at the point of use:
    ``mode: sweep`` returns [(primary, plugin)] bindings (target
    REQUIRED); control/calibration return [] (target FORBIDDEN). Unknown
    plugin names refuse here, before any GPU time is spent."""
    if manifest.mode == "sweep":
        if manifest.target is None:
            raise TargetError("mode: sweep REQUIRES a target spec (F20)")
        return [(p, get_target_plugin(p.plugin))
                for p in manifest.target.primaries]
    if manifest.target is not None:
        raise TargetError(f"mode: {manifest.mode} FORBIDS a target spec (F20)")
    return []


def relevant_slice(target: TargetSpec) -> CorpusSlice:
    """Resolve ``target.relevant_corpus`` (the §4.5 kl_localization
    slice) — thin, hashed delegation to corpora.resolve_corpus_ref."""
    from sisteer.capeval.corpora import resolve_corpus_ref

    return resolve_corpus_ref(target.relevant_corpus)


# --------------------------------------------------------------------------
# sisteer_compliance (§9): strict SI compliance via the vendored checkers
# --------------------------------------------------------------------------


@register_target("sisteer_compliance")
def sisteer_compliance(
    engine: Any,
    arm: Optional[Any],
    primary: PrimarySpec,
    prompts: CorpusSlice,
    *,
    gen_texts: Optional[Sequence[str]] = None,
) -> TargetLegScores:
    """Strict SI-compliance rate on pinned alpaca-holdout prompts (§9).

    Renders ``params.n_prompts`` prompts (plain user turns — the SI text
    is NEVER shown; the intervention must induce the behavior), generates
    greedily ``params.max_new_tokens`` tokens under the arm's ctx (token
    kinds get ``arm.transform_prompt`` — this path renders its own
    prompts, §5.1), and scores each generation with the AUTHORITATIVE
    sisteer checker for the SI: ``data.instructions.resolve_si`` →
    ``eval.checkers.check`` (strict, the repo's programmatic checkers)."""
    from sisteer.data.instructions import resolve_si
    from sisteer.eval import checkers

    params = dict(primary.params)
    unknown = sorted(set(params) - {"si", "n_prompts", "max_new_tokens"})
    if unknown:
        raise TargetError(f"sisteer_compliance: unknown param(s) {unknown}")
    if "si" not in params:
        raise TargetError("sisteer_compliance: params.si is required")
    instr = resolve_si(params["si"])  # loud KeyError on an unknown SI
    n_prompts = int(params.get("n_prompts", 100))
    max_new_tokens = int(params.get("max_new_tokens", 200))
    if prompts.kind != "prompts":
        raise TargetError(f"sisteer_compliance: prompt slice kind must be "
                          f"'prompts', got {prompts.kind!r}")
    if len(prompts) < n_prompts:
        raise TargetError(f"sisteer_compliance: slice has {len(prompts)} "
                          f"prompts, params ask for {n_prompts}")
    texts_in = list(prompts.items[:n_prompts])
    ids = list(prompts.item_ids[:n_prompts])

    if gen_texts is None:
        rendered = ([arm.transform_prompt(p) for p in texts_in]
                    if arm is not None else texts_in)
        outs = engine.generate(rendered, arm, default_gen_cfg(max_new_tokens))
        gen_texts = [o.text for o in outs]
    if len(gen_texts) != n_prompts:
        raise TargetError(f"sisteer_compliance: got {len(gen_texts)} "
                          f"generations for {n_prompts} prompts")

    values = tuple(
        1.0 if checkers.check(instr.checker, t, **instr.checker_kwargs)
        else 0.0
        for t in gen_texts)
    return TargetLegScores(
        metric=primary.metric,
        plugin="sisteer_compliance",
        arm_name=(arm.name if arm is not None else "none"),
        item_ids=tuple(ids),
        values=values,
        params_resolved={
            "si": params["si"], "instruction_id": instr.id,
            "checker": instr.checker, "checker_kwargs": instr.checker_kwargs,
            "n_prompts": n_prompts, "max_new_tokens": max_new_tokens,
            "prompt_slice_hash": prompts.slice_hash,
        },
    )
