"""Task→domain registry, loaded from configs/capeval/domains.yaml (§2.2).

Stage-0 FROZEN interfaces. Bodies are ``NotImplementedError`` until the
implementation lands; the dataclasses/signatures and the checked-in
domains.yaml are the binding contract, pinned by
``tests/test_capeval_protocol.py``.

Registered-metric rule (DESIGN §3): every emitted metric must resolve to
exactly one domain or the run refuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import yaml


class RegistryError(ValueError):
    """A domains.yaml / task-resolution refusal (loud, never a default)."""


# v0 implements exactly these transforms/estimators; anything else in the
# registry is a refusal at load time (better than a silent wrong aggregate).
TRANSFORMS = ("identity", "relative_nll")
ESTIMATORS = ("paired_bootstrap",)

# --- Schema constants (frozen) ---------------------------------------------

DOMAIN_KINDS = ("accuracy", "nll", "gate", "display")
DIRECTIONS = ("up", "down")
# `role: acceptance` domains get full non-inferiority statuses but are
# EXCLUDED from the gating set and from screen status; their CIs are
# plain one-sided 95% percentile-bootstrap, own family, UNADJUSTED.
# Absent role (None) means plain gating:/kind: semantics.
ROLES = (None, "acceptance")
# F14: every accuracy/nll entry must carry ALL FIVE of these fields —
# the loader refuses an entry missing any of them.
F14_FIELDS = ("unit", "transform", "task_weight", "item_weight", "estimator")
ITEM_WEIGHTS = ("items", "tokens")


@dataclass(frozen=True)
class Domain:
    """One domains.yaml row.

    Aggregation semantics (F14): domain delta = task_weight-weighted mean
    of per-task deltas, each task aggregated per its item_weight
    ("items": unweighted mean over items; "tokens": Σ numerator / Σ
    denominator from per-item nll_sum/token_count rows). The bootstrap
    resamples items WITH their weights (stratified by task) and recomputes
    the registered aggregate from resampled numerators/denominators —
    never from cached scalars.
    """

    name: str
    kind: str  # accuracy | nll | gate | display
    tasks: Tuple[str, ...]
    direction: Optional[str] = None  # accuracy/nll only: up | down
    margin: Optional[float] = None  # accuracy/nll only, raw units
    gating: bool = False
    role: Optional[str] = None  # None | "acceptance"
    #: ``retired: true`` marks a row whose instrument no longer runs forward
    #: (math_s2parity, PW3): the row stays so existing stores' rows still
    #: resolve and the frozen analyses still read them; the flag is an
    #: annotation for readers/reports — it changes no gating, status or
    #: aggregation semantics.
    retired: bool = False
    # F14 fields — required (non-None) for accuracy/nll kinds:
    unit: Optional[str] = None  # e.g. fraction_correct | nats_per_token
    transform: Optional[str] = None  # e.g. identity | relative_nll
    task_weight: Optional[Union[float, Dict[str, float]]] = None
    item_weight: Optional[str] = None  # items | tokens
    estimator: Optional[str] = None  # e.g. paired_bootstrap


@dataclass(frozen=True)
class Registry:
    """The loaded registry. ``version`` bumps the harness version."""

    version: int
    domains: Tuple[Domain, ...]

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Registry":
        """Load + validate domains.yaml. Refusals: a task mapped to two
        domains; any accuracy/nll entry missing one of the five F14
        fields; unknown kind/direction/role/item_weight values; a
        gating:true entry with role: acceptance."""
        path = Path(path)
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, Mapping):
            raise RegistryError(f"{path}: not a mapping")
        unknown = sorted(set(raw) - {"version", "domains"})
        if unknown:
            raise RegistryError(f"{path}: unknown top-level key(s) {unknown}")
        version = raw.get("version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise RegistryError(f"{path}: version must be an int, got {version!r}")
        doms_raw = raw.get("domains")
        if not isinstance(doms_raw, Mapping) or not doms_raw:
            raise RegistryError(f"{path}: domains must be a non-empty mapping")

        allowed_keys = {
            "kind", "tasks", "direction", "margin", "gating", "role",
            "unit", "transform", "task_weight", "item_weight", "estimator",
            "retired",
        }
        domains: List[Domain] = []
        seen_tasks: Dict[str, str] = {}
        for name, spec in doms_raw.items():
            where = f"{path}:domains.{name}"
            if not isinstance(spec, Mapping):
                raise RegistryError(f"{where}: entry is not a mapping")
            unknown = sorted(set(spec) - allowed_keys)
            if unknown:
                raise RegistryError(f"{where}: unknown key(s) {unknown}")
            kind = spec.get("kind")
            if kind not in DOMAIN_KINDS:
                raise RegistryError(f"{where}: kind {kind!r} not in {DOMAIN_KINDS}")
            retired = spec.get("retired", False)
            if not isinstance(retired, bool):
                raise RegistryError(f"{where}: retired must be a bool, got {retired!r}")
            tasks = spec.get("tasks")
            if (not isinstance(tasks, list) or not tasks
                    or not all(isinstance(t, str) and t for t in tasks)):
                raise RegistryError(f"{where}: tasks must be a non-empty list "
                                    "of task names")
            for t in tasks:
                if t in seen_tasks:
                    raise RegistryError(
                        f"{where}: task {t!r} already mapped to domain "
                        f"{seen_tasks[t]!r} — every task maps to exactly one "
                        "domain (DESIGN §3)")
                seen_tasks[t] = name
            role = spec.get("role")
            if role not in ROLES:
                raise RegistryError(f"{where}: role {role!r} not in {ROLES}")
            gating = spec.get("gating", False)
            if not isinstance(gating, bool):
                raise RegistryError(f"{where}: gating must be a bool")
            if gating and role == "acceptance":
                raise RegistryError(f"{where}: gating: true with role: "
                                    "acceptance is a contradiction (r2-F6)")

            if kind in ("accuracy", "nll"):
                missing = [f for f in F14_FIELDS if spec.get(f) is None]
                if missing:
                    raise RegistryError(f"{where}: {kind} entry missing F14 "
                                        f"field(s) {missing}")
                direction = spec.get("direction")
                if direction not in DIRECTIONS:
                    raise RegistryError(f"{where}: direction {direction!r} "
                                        f"not in {DIRECTIONS}")
                margin = spec.get("margin")
                if not isinstance(margin, (int, float)) or isinstance(margin, bool) \
                        or margin <= 0:
                    raise RegistryError(f"{where}: margin must be a positive "
                                        f"number, got {margin!r}")
                if spec["item_weight"] not in ITEM_WEIGHTS:
                    raise RegistryError(f"{where}: item_weight "
                                        f"{spec['item_weight']!r} not in "
                                        f"{ITEM_WEIGHTS}")
                if spec["transform"] not in TRANSFORMS:
                    raise RegistryError(f"{where}: transform "
                                        f"{spec['transform']!r} not in "
                                        f"{TRANSFORMS} (v0)")
                if spec["estimator"] not in ESTIMATORS:
                    raise RegistryError(f"{where}: estimator "
                                        f"{spec['estimator']!r} not in "
                                        f"{ESTIMATORS} (v0)")
                # v0 aggregation implements exactly these two pairings —
                # a mismatch would silently mis-aggregate (F14).
                if kind == "nll" and spec["item_weight"] != "tokens":
                    raise RegistryError(f"{where}: nll domains aggregate "
                                        "Σnll_sum/Σtoken_count — item_weight "
                                        "must be 'tokens'")
                if kind == "accuracy" and spec["item_weight"] != "items":
                    raise RegistryError(f"{where}: accuracy domains aggregate "
                                        "unweighted over items — item_weight "
                                        "must be 'items'")
                domains.append(Domain(
                    name=name, kind=kind, tasks=tuple(tasks),
                    direction=direction, margin=float(margin), gating=gating,
                    role=role, retired=retired,
                    unit=spec["unit"], transform=spec["transform"],
                    task_weight=spec["task_weight"],
                    item_weight=spec["item_weight"],
                    estimator=spec["estimator"],
                ))
            else:  # gate | display
                forbidden = [f for f in ("direction", "margin", *F14_FIELDS)
                             if spec.get(f) is not None]
                if forbidden:
                    raise RegistryError(f"{where}: {kind} entry carries "
                                        f"accuracy/nll-only field(s) {forbidden}")
                if gating:
                    raise RegistryError(f"{where}: {kind} domains are never "
                                        "non-inferiority gating rows (they "
                                        "feed κ / the canary rule instead)")
                if role is not None:
                    raise RegistryError(f"{where}: role is only meaningful on "
                                        "accuracy/nll entries")
                domains.append(Domain(name=name, kind=kind, tasks=tuple(tasks),
                                      gating=False, retired=retired))
        return cls(version=version, domains=tuple(domains))

    def domain_of(self, task: str) -> Domain:
        """Resolve a task name to its ONE domain. An unregistered task is
        a refusal (registered-metric rule, DESIGN §3), never a default."""
        for d in self.domains:
            if task in d.tasks:
                return d
        raise RegistryError(
            f"task {task!r} is not registered in any domain — every emitted "
            "metric must resolve to exactly one domain (DESIGN §3); "
            f"registered tasks: {sorted(t for d in self.domains for t in d.tasks)}")

    def domain_named(self, name: str) -> Domain:
        """Domain row by name; unregistered name is a refusal."""
        for d in self.domains:
            if d.name == name:
                return d
        raise RegistryError(f"no domain named {name!r}; registered: "
                            f"{[d.name for d in self.domains]}")

    def gating_set(
        self,
        model: str,
        power_table: Mapping[Tuple[str, str], float],
        exclusions: Sequence[str],
    ) -> List[Domain]:
        """The gating domains for one run: gating:true rows, minus
        ``exclusions`` (the manifest's off_target_exclusions), minus
        role:acceptance rows (structurally excluded, r2-F6). The power
        table — Power(preserved | harm=0) keyed (model, domain) from
        stats/power.py — annotates underpowered domains; on v0
        (non-confirmatory) runs it never removes a domain from the
        fingerprint (§7.4), so it is accepted (and validated) but does
        not filter here; power-based removal belongs to the v1
        confirmatory path."""
        known = {d.name for d in self.domains}
        bad = sorted(set(exclusions) - known)
        if bad:
            raise RegistryError(f"gating_set: unknown exclusion domain(s) "
                                f"{bad}; registered: {sorted(known)}")
        for key in power_table:
            if (not isinstance(key, tuple) or len(key) != 2):
                raise RegistryError(f"gating_set: power_table keys are "
                                    f"(model, domain) tuples, got {key!r}")
        return [d for d in self.domains
                if d.gating and d.role != "acceptance"
                and d.name not in set(exclusions)]
