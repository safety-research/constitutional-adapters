"""Petri Bloom: the ONE display-kind ``petri_behavior`` domain — every
``petri/<behavior>`` task resolves to it, it is excluded from the gating
set, and ``statuses_for_arm`` skips it.

CPU-offline; no inspect_ai / petri_bloom imports anywhere in Phase 1."""

from pathlib import Path

import pytest

from sisteer.capeval.domains import Registry, RegistryError
from sisteer.capeval.manifest import PETRI_BEHAVIORS
from sisteer.capeval.stats import noninferiority as ni

ROOT = Path(__file__).parents[1]
DOMAINS_YAML = ROOT / "configs" / "capeval" / "domains.yaml"

GATING_9 = {"lm_fidelity_raw", "lm_fidelity_chat", "knowledge", "commonsense",
            "truthfulness", "math", "code", "instruction_following",
            "reasoning"}
REV = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load(DOMAINS_YAML)


def test_petri_tasks_resolve_to_one_display_domain(registry):
    assert len(PETRI_BEHAVIORS) == 6
    for behavior in PETRI_BEHAVIORS:
        dom = registry.domain_of(f"petri/{behavior}")
        assert dom.name == "petri_behavior"
        assert dom.kind == "display"
        assert dom.gating is False
        assert dom.role is None
    dom = registry.domain_named("petri_behavior")
    assert dom.tasks == tuple(f"petri/{b}" for b in PETRI_BEHAVIORS)


def test_unregistered_petri_task_refuses(registry):
    # the registered-metric rule stays loud: a typo'd behavior never
    # resolves to a default domain
    with pytest.raises(RegistryError, match="not registered"):
        registry.domain_of("petri/not_a_behavior")


def test_petri_domain_excluded_from_gating_set(registry):
    names = {d.name for d in registry.gating_set("m", {}, [])}
    assert "petri_behavior" not in names
    # and the capability gating set is exactly what it was before the
    # addition — the petri registration changes nothing there
    assert names == GATING_9
    # display domains are not even valid gating exclusions to worry about:
    # excluding a gating domain still never surfaces petri_behavior
    names2 = {d.name for d in registry.gating_set("m", {}, ["math"])}
    assert "petri_behavior" not in names2 and "math" not in names2


def _row(*, arm, task, domain, metric, item, value, trial):
    return dict(
        sweep="synthetic", run_id="r", model_id="m", model_revision=REV,
        arm_name=arm, kind="steering" if arm != "none" else "none",
        coeff=2.0 if arm != "none" else None, intervention_hash="ab" * 8,
        engine="hf", tier=1, task=task, task_version="v1", domain=domain,
        item_id=item, trial_id=trial, seed=trial, attempt=0,
        position_order=None, metric=metric, value=float(value),
        extra_json=None, harness_config_hash="cd" * 8, env_digest="ef" * 8,
        slurm_job_id=None, created_at=0.0, wall_clock_s=None)


def _rows(arm, trial):
    rows = []
    # a real accuracy domain (math), 6 items
    for j, v in enumerate([1, 0, 1, 1, 0, 1]):
        rows.append(_row(arm=arm, task="gsm8k_platinum_cap", domain="math",
                         metric="acc", item=f"i{j}", value=v, trial=trial))
    # petri display rows sitting in the same frame (M13 metric names)
    for j, v in enumerate([7.0, 2.0, 9.0]):
        rows.append(_row(arm=arm, task="petri/self_preservation",
                         domain="petri_behavior", metric="self_preservation",
                         item=f"s{j}", value=v, trial=trial))
        rows.append(_row(arm=arm, task="petri/self_preservation",
                         domain="petri_behavior", metric="eval_awareness",
                         item=f"s{j}", value=3.0, trial=trial))
    return rows


def test_statuses_for_arm_skips_petri(registry):
    """statuses_for_arm computes non-inferiority for accuracy/nll domains
    only — petri rows in the frame produce NO petri_behavior status (the
    two-SI Petri contrast is bespoke petri_stats)."""
    arm = _rows("arm", trial=1)
    ref = _rows("none", trial=0)
    statuses = ni.statuses_for_arm(arm, ref, registry, model="m", b=50)
    assert "math" in statuses
    assert "petri_behavior" not in statuses
    assert all(s.kind in ("accuracy", "nll") for s in statuses.values())
