"""Simulation-based power tests — the rule-of-three
discordance floor (an empty discordance sample never yields power 1),
the HIERARCHICAL contrast-then-items simulation (run-level effects
don't average away as 1/√n), per-task simulation with task weights,
seeded goldens, the encoded unpowered-tiny expectation, the
NLL analog without the spread/SE scale factor, and resolved-margin
plumbing."""

import numpy as np
import pytest

from sisteer.capeval.domains import Registry
from sisteer.capeval.stats import calibrate as cal
from sisteer.capeval.stats import power as pw
from sisteer.capeval.stats.calibrate import TaskCalibration
from sisteer.capeval.stats.power import PowerError

from pathlib import Path

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def registry():
    return Registry.load(ROOT / "configs" / "capeval" / "domains.yaml")


# --------------------------------------------------------------------------
# rule of three
# --------------------------------------------------------------------------


def test_rule_of_three_floor():
    assert pw.rule_of_three_floor(500) == pytest.approx(3 / 500)
    with pytest.raises(PowerError):
        pw.rule_of_three_floor(0)


def test_floored_discordance_rate():
    rate, floored = pw.floored_discordance_rate(0, 500)
    assert rate == pytest.approx(3 / 500) and floored
    rate, floored = pw.floored_discordance_rate(50, 500)
    assert rate == pytest.approx(0.1) and not floored
    # an observed rate below the floor is floored too
    rate, floored = pw.floored_discordance_rate(1, 1000)
    assert rate == pytest.approx(3 / 1000) and floored


# --------------------------------------------------------------------------
# accuracy: hierarchical simulation helpers
# --------------------------------------------------------------------------


def _acc_task(task, ref, arms_by_contrast):
    ref = np.asarray(ref, dtype=float)
    return TaskCalibration(
        task=task, n_items=len(ref), base_mean=float(ref.mean()),
        ref_values=[float(v) for v in ref],
        arm_values_by_contrast=[[float(v) for v in a]
                                for a in arms_by_contrast])


def _iid_flip_contrasts(ref, rate, n_contrasts, rng):
    """Null contrasts with symmetric per-item flips at ``rate`` — item
    noise only, no run-level effect."""
    ref = np.asarray(ref, dtype=float)
    p = ref.mean()
    out = []
    for _ in range(n_contrasts):
        p1 = min(1.0, rate / (2 * p))
        p0 = min(1.0, rate / (2 * (1 - p)))
        flip = np.where(ref >= 0.5, rng.random(ref.size) < p1,
                        rng.random(ref.size) < p0)
        out.append(np.where(flip, 1 - ref, ref))
    return out


def test_empty_discordance_never_gives_power_one(registry):
    """r2-F3: zero observed discordance floors the flip rate at
    3/(N·contrasts), so the harm=0 simulation always carries noise and
    Power(preserved|h0) = 1 exactly is impossible by construction."""
    dom = registry.domain_named("math")
    rng = np.random.default_rng(0)
    ref = (rng.random(100) < 0.7).astype(float)
    tasks = [_acc_task("gsm8k_platinum_cap", ref, [ref] * 5)]  # NO noise
    sim = pw.simulate_power_accuracy(dom, tasks, margin=0.01, family_k=9,
                                     n_sims=300, b_boot=300, seed=0)
    assert sim["discordance_floored"] is True
    assert sim["discordance_rate"] == pytest.approx(3 / 500)
    assert 0.0 <= sim["power_preserved_h0"] < 1.0


def test_power_seeded_reproducible(registry):
    dom = registry.domain_named("math")
    rng = np.random.default_rng(1)
    ref = (rng.random(200) < 0.7).astype(float)
    tasks = [_acc_task("gsm8k_platinum_cap", ref,
                       _iid_flip_contrasts(ref, 0.05, 5, rng))]
    kw = dict(margin=0.02, family_k=9, n_sims=100, b_boot=200)
    a = pw.simulate_power_accuracy(dom, tasks, seed=42, **kw)
    b = pw.simulate_power_accuracy(dom, tasks, seed=42, **kw)
    assert a == b
    c = pw.simulate_power_accuracy(dom, tasks, seed=43, **kw)
    assert a != c  # the simulation is genuinely stochastic


def test_unpowered_tiny_expectation_encoded(registry):
    """§16.4: a ~100-item domain at m=2pt with real discordance is
    UNPOWERED for preserved-certification — the honest annotation the
    §7.4 rule exists to make."""
    dom = registry.domain_named("math")
    rng = np.random.default_rng(2)
    ref = (rng.random(100) < 0.65).astype(float)
    tasks = [_acc_task("gsm8k_platinum_cap", ref,
                       _iid_flip_contrasts(ref, 0.15, 5, rng))]
    sim = pw.simulate_power_accuracy(dom, tasks, margin=0.02, family_k=9,
                                     n_sims=200, b_boot=300, seed=0)
    assert sim["power_preserved_h0"] < pw.POWER_THRESHOLD


def test_large_n_low_noise_is_powered_both_ways(registry):
    dom = registry.domain_named("math")
    rng = np.random.default_rng(3)
    ref = (rng.random(3000) < 0.7).astype(float)
    tasks = [_acc_task("gsm8k_platinum_cap", ref,
                       _iid_flip_contrasts(ref, 0.004, 5, rng))]
    sim = pw.simulate_power_accuracy(dom, tasks, margin=0.05, family_k=9,
                                     n_sims=100, b_boot=300, seed=0)
    assert sim["power_preserved_h0"] >= 0.8
    assert sim["power_degraded_h2m"] >= 0.8


def test_saturated_base_acc_refused(registry):
    dom = registry.domain_named("math")
    for acc in (0.0, 1.0):
        ref = np.full(100, acc)
        tasks = [_acc_task("gsm8k_platinum_cap", ref, [ref] * 3)]
        with pytest.raises(PowerError, match="base_acc"):
            pw.simulate_power_accuracy(dom, tasks, margin=0.02, family_k=9)


def test_run_level_effects_do_not_average_away(registry):
    """r1 stats-core #4: five contrasts carrying COHERENT run-level
    ±2·margin shifts must defeat preserved-certification even at n=2000
    — a flat per-item flip model would launder the same discordance into
    1/√n noise and report high power."""
    dom = registry.domain_named("math")
    rng = np.random.default_rng(4)
    n = 2000
    ref = (rng.random(n) < 0.7).astype(float)
    arms = []
    for shift in (-0.04, -0.04, 0.0, 0.04, 0.04):
        arm = ref.copy()
        k = int(round(abs(shift) * n))
        if shift < 0:  # arm loses k correct items: harm +|shift|
            idx = rng.choice(np.flatnonzero(ref >= 0.5), size=k,
                             replace=False)
            arm[idx] = 0.0
        elif shift > 0:  # arm gains k items: harm −|shift|
            idx = rng.choice(np.flatnonzero(ref < 0.5), size=k,
                             replace=False)
            arm[idx] = 1.0
        arms.append(arm)
    tasks = [_acc_task("gsm8k_platinum_cap", ref, arms)]
    sim = pw.simulate_power_accuracy(dom, tasks, margin=0.02, family_k=9,
                                     n_sims=150, b_boot=300, seed=0)
    # with n=2000 the CI hugs the drawn run effect, so preserved fires
    # only on the harm ≤ 0 contrasts (3 of 5 ⇒ power ≈ 0.6). A flat
    # item-pooled model launders the same contrasts into mean-zero 1/√n
    # noise (SE ≈ 0.004) and certifies essentially always (power ≈ 1) —
    # the upper bound is the discriminating assertion.
    assert 0.4 < sim["power_preserved_h0"] < pw.POWER_THRESHOLD


def test_multi_task_domains_simulate_per_task(registry):
    """r1 stats-core #6: a 100-item noisy task at 50% task_weight
    dominates the domain CI even next to a 1000-item quiet task — an
    item-pooled simulation would dilute its noise 10:1 and overstate
    power."""
    dom = registry.domain_named("commonsense")  # two tasks, equal weight
    rng = np.random.default_rng(5)
    small_ref = (rng.random(100) < 0.6).astype(float)
    big_ref = (rng.random(1000) < 0.6).astype(float)
    tasks = [
        _acc_task("tinyHellaswag", small_ref,
                  _iid_flip_contrasts(small_ref, 0.20, 5, rng)),  # noisy
        _acc_task("tinyWinogrande", big_ref, [big_ref] * 5),  # noiseless
    ]
    sim = pw.simulate_power_accuracy(dom, tasks, margin=0.03, family_k=9,
                                     n_sims=150, b_boot=300, seed=0)
    # per-task weighting: the small task's SE (≈0.045, weighted 0.5)
    # keeps the α/9 upper bound above m=0.03 most replicates. Pooled-1100
    # would have SE ≈ 0.004 and certify essentially always.
    assert sim["power_preserved_h0"] < 0.5


# --------------------------------------------------------------------------
# NLL analog (hierarchical, no spread/SE scale factor — r1 #5)
# --------------------------------------------------------------------------


def _nll_task(task, base_nll, deltas_by_contrast, n_items):
    return TaskCalibration(
        task=task, n_items=n_items, base_mean=base_nll,
        deltas_by_contrast=[[float(x) for x in d]
                            for d in deltas_by_contrast])


def test_nll_power_basic(registry):
    dom = registry.domain_named("lm_fidelity_chat")
    rng = np.random.default_rng(0)
    deltas = [rng.normal(0, 0.01, size=200) for _ in range(5)]
    tasks = [_nll_task("tier0/nll_ultrachat_masked", 2.0, deltas, 200)]
    sim = pw.simulate_power_nll(dom, tasks, margin=0.03, family_k=2,
                                n_sims=100, b_boot=200, seed=0)
    # shift at 2m = 0.12 nats vs 0.01-nat noise: both powers high
    assert sim["power_preserved_h0"] >= 0.8
    assert sim["power_degraded_h2m"] >= 0.8
    pooled = np.concatenate(deltas)
    assert sim["noise_spread"] == pytest.approx(float(pooled.std(ddof=1)))


def test_nll_run_level_shifts_defeat_certification(registry):
    """r1 #4/#5 for NLL: whole-contrast mean shifts of ±2·margin·base
    survive the item resample — power reflects the run effect, with no
    spread/SE scale factor anywhere."""
    dom = registry.domain_named("lm_fidelity_chat")
    rng = np.random.default_rng(1)
    base = 2.0
    deltas = [rng.normal(mu * 2 * 0.03 * base, 0.005, size=300)
              for mu in (-1, -1, 0, 1, 1)]
    tasks = [_nll_task("tier0/nll_ultrachat_masked", base, deltas, 300)]
    sim = pw.simulate_power_nll(dom, tasks, margin=0.03, family_k=2,
                                n_sims=150, b_boot=200, seed=0)
    # preserved fires only on the harm ≤ 0 contrasts (3 of 5 ⇒ ≈ 0.6);
    # pooling the deltas would zero the mean and certify always
    assert 0.4 < sim["power_preserved_h0"] < pw.POWER_THRESHOLD


def test_nll_zero_spread_needs_rule_of_three_inputs(registry):
    dom = registry.domain_named("lm_fidelity_chat")
    tasks = [_nll_task("tier0/nll_ultrachat_masked", 2.0,
                       [[0.0] * 50] * 5, 50)]
    with pytest.raises(PowerError, match="rule-of-three"):
        pw.simulate_power_nll(dom, tasks, margin=0.03, family_k=2,
                              n_sims=20, b_boot=50)
    # with the scale, it simulates ±scale at the floored rate — never σ=0
    sim = pw.simulate_power_nll(dom, tasks, margin=0.03, family_k=2,
                                n_sims=50, b_boot=100, seed=0,
                                discretization_scale=0.005)
    assert sim["degenerate_null"] is True
    assert 0.0 <= sim["power_preserved_h0"] <= 1.0
    assert sim["power_degraded_h2m"] > 0.5  # 0.12-nat shift is detectable


def test_nll_empty_null_refused(registry):
    dom = registry.domain_named("lm_fidelity_chat")
    tasks = [TaskCalibration(task="tier0/nll_ultrachat_masked", n_items=10,
                             base_mean=2.0, deltas_by_contrast=[])]
    with pytest.raises(PowerError, match="deltas_by_contrast"):
        pw.simulate_power_nll(dom, tasks, margin=0.03, family_k=1)


# --------------------------------------------------------------------------
# calibration-driven table + resolved margins (r1 #13)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_calibration(registry):
    rng = np.random.default_rng(3)
    rows = []

    def acc_rows(task, domain, trial, vals):
        return [dict(
            sweep="s", run_id="r", model_id="m",
            model_revision="0" * 40, arm_name="none", kind="none",
            coeff=None, intervention_hash="ab" * 8, engine="hf", tier=1,
            task=task, task_version="v1", domain=domain, item_id=f"i{j}",
            trial_id=trial, seed=trial, attempt=0, position_order=None,
            metric="acc", value=float(v), extra_json=None,
            harness_config_hash="cd" * 8, env_digest="ef" * 8,
            slurm_job_id=None, created_at=0.0, wall_clock_s=None)
            for j, v in enumerate(vals)]

    base = (rng.random(60) < 0.7).astype(float)
    for trial in range(4):
        vals = base if trial == 0 else np.where(
            rng.random(60) < 0.08, 1 - base, base)
        rows += acc_rows("gsm8k_platinum_cap", "math", trial, vals)
    return cal.build_calibration(rows, registry, b_boot=300, seed=0)


def test_domain_power_and_table(registry, synthetic_calibration):
    c = synthetic_calibration
    table = pw.power_table(c, registry, model="m", n_sims=60, b_boot=150,
                           seed=0)
    assert set(table) == {("m", "math")}  # only calibrated domains appear
    res = table[("m", "math")]
    assert res.kind == "accuracy" and res.family_k == 9
    assert res.n_items == 60 and res.margin == 0.02
    assert res.n_contrasts == 3
    assert 0.0 <= res.power_preserved_h0 <= 1.0
    assert res.unpowered == (res.power_preserved_h0 < pw.POWER_THRESHOLD)
    assert res.discordance_rate is not None
    # gate/display domains refuse power outright
    with pytest.raises(PowerError, match="accuracy/nll"):
        pw.domain_power(registry.domain_named("dist_shift"),
                        c.domains["math"], family_k=9)


def test_power_uses_resolved_margins(registry, synthetic_calibration):
    """r1 stats-core #13: a manifest margin override must reach the power
    simulation — the SAME value statuses_for_arm classifies with, never
    domains.yaml's default."""
    c = synthetic_calibration
    table = pw.power_table(c, registry, model="m", margins={"math": 0.05},
                           n_sims=60, b_boot=150, seed=0)
    res = table[("m", "math")]
    assert res.margin == 0.05  # the override, not domains.yaml's 0.02
    default = pw.power_table(c, registry, model="m", n_sims=60, b_boot=150,
                             seed=0)[("m", "math")]
    assert default.margin == 0.02
    # a wider margin can only help preserved-certification
    assert res.power_preserved_h0 >= default.power_preserved_h0
    # unknown override names refuse (same rule as statuses_for_arm)
    with pytest.raises(PowerError, match="unregistered"):
        pw.power_table(c, registry, model="m", margins={"mystery": 0.1})
    # domain_power takes the override directly
    r2 = pw.domain_power(registry.domain_named("math"), c.domains["math"],
                         family_k=9, margin=0.05, n_sims=30, b_boot=100)
    assert r2.margin == 0.05


def test_power_table_threads_nll_discretization_scale(registry,
                                                      synthetic_calibration):
    """A fully deterministic replay leaves an NLL domain's null spread at
    exactly zero (the k=12 Llama-3.1-8B HF calibration did this for both
    lm_fidelity domains). power_table must (a) still refuse loudly
    without a scale — never silently skip the domain — and (b) hand
    ``nll_discretization_scale`` through to simulate_power_nll so the
    rule-of-three treatment can run."""
    import copy

    c = copy.deepcopy(synthetic_calibration)
    c.domains["lm_fidelity_chat"] = cal.DomainCalibration(
        domain="lm_fidelity_chat", kind="nll", sigma_d=0.0, sigma_d_se=0.0,
        tails={}, n_contrasts=5, n_items=50, needs_more_trials=False,
        tasks={"tier0/nll_ultrachat_masked": _nll_task(
            "tier0/nll_ultrachat_masked", 2.0, [[0.0] * 50] * 5, 50)})
    with pytest.raises(PowerError, match="rule-of-three"):
        pw.power_table(c, registry, model="m", n_sims=20, b_boot=50, seed=0)
    table = pw.power_table(c, registry, model="m", n_sims=50, b_boot=100,
                           seed=0, nll_discretization_scale=0.005)
    res = table[("m", "lm_fidelity_chat")]
    assert res.kind == "nll"
    assert res.noise_spread == 0.005      # the scale, not σ=0
    assert res.power_degraded_h2m > 0.5   # 2m·base = 0.12-nat shift detectable
    # the accuracy domain is untouched by the NLL-only knob
    assert table[("m", "math")].discordance_rate is not None


def test_power_refuses_calibration_without_task_samples(registry):
    """A pre-#4 calibration record (no per-task null samples) cannot be
    power-simulated honestly — refuse, never approximate."""
    dc = cal.DomainCalibration(
        domain="math", kind="accuracy", sigma_d=0.01, sigma_d_se=0.001,
        tails={}, n_contrasts=5, n_items=100, needs_more_trials=False)
    with pytest.raises(PowerError, match="per-task"):
        pw.domain_power(registry.domain_named("math"), dc, family_k=9)
