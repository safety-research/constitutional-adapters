"""Stats tests — pairing + engine-matched
refusals, the registered F14 aggregate, paired-bootstrap machinery,
Bonferroni quantile inversion + monotonicity, McNemar display-only
goldens, the trial-0-reference conditional null (5-contrast structure),
canary thresholds, the unfloored null-KL annotation, calibration
persistence, target status, and the end-to-end synthetic fingerprint."""

from pathlib import Path

import numpy as np
import pytest

from sisteer.capeval.domains import Registry
from sisteer.capeval.manifest import PrimarySpec
from sisteer.capeval.stats import calibrate as cal
from sisteer.capeval.stats import fingerprint as fp
from sisteer.capeval.stats import noninferiority as ni
from sisteer.capeval.stats.noninferiority import StatsError

ROOT = Path(__file__).parents[1]
REV = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load(ROOT / "configs" / "capeval" / "domains.yaml")


def _row(*, arm="arm", task="gsm8k_platinum_cap", domain="math", item="i0",
         metric="acc", value=1.0, trial=1, engine="hf", tier=1):
    return dict(
        sweep="synthetic", run_id="r", model_id="m", model_revision=REV,
        arm_name=arm, kind="steering" if arm not in ("none",) else "none",
        coeff=2.0 if arm != "none" else None, intervention_hash="ab" * 8,
        engine=engine, tier=tier, task=task, task_version="v1", domain=domain,
        item_id=item, trial_id=trial, seed=trial, attempt=0,
        position_order=None, metric=metric, value=float(value),
        extra_json=None, harness_config_hash="cd" * 8,
        env_digest="ef" * 8, slurm_job_id=None, created_at=0.0,
        wall_clock_s=None)


def _acc_rows(arm, values, *, task="gsm8k_platinum_cap", domain="math",
              trial=1, engine="hf"):
    return [_row(arm=arm, task=task, domain=domain, item=f"i{j}",
                 value=v, trial=trial, engine=engine)
            for j, v in enumerate(values)]


def _nll_rows(arm, sums, toks, *, task="tier0/nll_ultrachat_masked",
              domain="lm_fidelity_chat", trial=1):
    rows = []
    for j, (s, t) in enumerate(zip(sums, toks)):
        rows.append(_row(arm=arm, task=task, domain=domain, item=f"w{j}",
                         metric="nll_sum", value=s, trial=trial, tier=0))
        rows.append(_row(arm=arm, task=task, domain=domain, item=f"w{j}",
                         metric="token_count", value=t, trial=trial, tier=0))
    return rows


# --------------------------------------------------------------------------
# pairing + engine matching
# --------------------------------------------------------------------------


def test_engine_mismatch_refused(registry):
    arm = _acc_rows("arm", [1, 0, 1, 1])
    ref = _acc_rows("none", [1, 1, 1, 1], trial=0, engine="vllm")
    with pytest.raises(StatsError, match="engine-matched"):
        ni.statuses_for_arm(arm, ref, registry, model="m")


def test_item_set_mismatch_refused(registry):
    dom = registry.domain_named("math")
    arm = ni.rows_frame(_acc_rows("arm", [1, 0, 1]))
    ref = ni.rows_frame(_acc_rows("none", [1, 1, 1, 1], trial=0))
    with pytest.raises(StatsError, match="item sets differ"):
        ni.pair_task(arm, ref, "gsm8k_platinum_cap", dom.kind)


def test_duplicate_item_rows_refused(registry):
    rows = _acc_rows("arm", [1, 0]) + _acc_rows("arm", [1])
    with pytest.raises(StatsError, match="duplicate item_id"):
        ni.pair_task(ni.rows_frame(rows),
                     ni.rows_frame(_acc_rows("none", [1, 1], trial=0)),
                     "gsm8k_platinum_cap", "accuracy")


def test_aggregate_pseudo_item_excluded_from_pairing(registry):
    """2026-08-17 incident: the s2parity probes filed a per-(task, trial)
    summary row under item_id "__aggregate__" with the SAME metric name
    as the per-item rows ("acc"), and it entered the paired stats as a
    201st item (PARITY_N = 200) for both parity domains. The point delta
    was provably unchanged (the pseudo-item equals its own leg's mean),
    but the bootstrap resampled it like an item. Stored campaign rows
    still carry that row, so the PAIRING side must drop it — and the
    result must be bit-identical to pairing rows that never had it."""
    dom = registry.domain_named("math")
    arm_vals, ref_vals = [1, 0, 1, 1], [1, 1, 0, 1]
    arm = _acc_rows("arm", arm_vals)
    ref = _acc_rows("none", ref_vals, trial=0)
    agg_arm = [_row(arm="arm", item="__aggregate__",
                    value=float(np.mean(arm_vals)))]
    agg_ref = [_row(arm="none", item="__aggregate__",
                    value=float(np.mean(ref_vals)), trial=0)]

    clean = ni.pair_task(ni.rows_frame(arm), ni.rows_frame(ref),
                         "gsm8k_platinum_cap", dom.kind)
    with_agg = ni.pair_task(ni.rows_frame(arm + agg_arm),
                            ni.rows_frame(ref + agg_ref),
                            "gsm8k_platinum_cap", dom.kind)
    assert with_agg.n_items == len(arm_vals) == clean.n_items
    assert with_agg.item_ids == clean.item_ids
    assert np.array_equal(with_agg.arm_values, clean.arm_values)
    assert np.array_equal(with_agg.ref_values, clean.ref_values)

    # the full status — point harm, bounds, n_items — is bit-identical
    kw = dict(margin=0.05, k_family=1, in_gating_set=True, b=500, seed=7)
    st_clean = ni.domain_status(ni.rows_frame(arm), ni.rows_frame(ref),
                                dom, **kw)
    st_agg = ni.domain_status(ni.rows_frame(arm + agg_arm),
                              ni.rows_frame(ref + agg_ref), dom, **kw)
    assert st_agg.as_dict() == st_clean.as_dict()
    assert st_agg.n_items == len(arm_vals)

    # aggregate-only rows are a summary, not per-item data — refuse
    with pytest.raises(StatsError, match="no per-item"):
        ni.pair_task(ni.rows_frame(agg_arm), ni.rows_frame(agg_ref),
                     "gsm8k_platinum_cap", dom.kind)


def test_one_leg_task_refused(registry):
    dom = registry.domain_named("math")
    arm = ni.rows_frame(_acc_rows("arm", [1, 0]))
    ref = ni.rows_frame(_acc_rows("none", [1, 1], trial=0,
                                  task="tinyMMLU", domain="knowledge"))
    with pytest.raises(StatsError, match="one leg is incomplete"):
        ni.pair_domain(arm, ref, dom)


def test_rows_frame_schema_drift_refused():
    rows = [dict(_row(), stray=1)]
    with pytest.raises(StatsError, match="schema drift"):
        ni.rows_frame(rows)


def test_non_finite_values_refused(registry):
    """r1 stats-core #1: a NaN metric would ride astype(float) into NaN
    bootstrap quantiles, classify as indeterminate, and read NaN > margin
    as False — a corrupted domain presenting as SCREEN-PASS. Refuse at
    pairing instead."""
    dom = registry.domain_named("math")
    ref = ni.rows_frame(_acc_rows("none", [1, 1, 1], trial=0))
    for bad in (float("nan"), float("inf")):
        arm = ni.rows_frame(_acc_rows("arm", [1, 0, bad]))
        with pytest.raises(StatsError, match="non-finite"):
            ni.pair_task(arm, ref, "gsm8k_platinum_cap", dom.kind)
    # accuracy outside [0, 1] is a scorer failure too
    arm = ni.rows_frame(_acc_rows("arm", [1, 0, 1.5]))
    with pytest.raises(StatsError, match=r"outside \[0, 1\]"):
        ni.pair_task(arm, ref, "gsm8k_platinum_cap", dom.kind)


def test_bad_token_counts_refused(registry):
    """r1 stats-core #1: token_count = 0 makes Σnll/Σtok a NaN/inf
    channel; fractional counts are a windowing failure."""
    dom = registry.domain_named("lm_fidelity_chat")
    ref = ni.rows_frame(_nll_rows("none", [200.0, 100.0], [100, 50], trial=0))
    arm = ni.rows_frame(_nll_rows("arm", [220.0, 110.0], [100, 0]))
    with pytest.raises(StatsError, match="non-positive or non-integral"):
        ni.pair_domain(arm, ref, dom)
    arm = ni.rows_frame(_nll_rows("arm", [220.0, 110.0], [100, 50.5]))
    with pytest.raises(StatsError, match="non-positive or non-integral"):
        ni.pair_domain(arm, ref, dom)


def test_model_and_task_version_matching_refused(registry):
    """r1 stats-core #2: same item IDs under a different model_id,
    model_revision, or task config/version is a different instrument —
    pairing refuses, naming both sides."""
    ref = _acc_rows("none", [1, 1, 1, 1], trial=0)
    # model_id mismatch
    arm = [dict(r, model_id="other-model") for r in _acc_rows("arm", [1, 0, 1, 1])]
    with pytest.raises(StatsError, match="model_id"):
        ni.statuses_for_arm(arm, ref, registry, model="m", b=50)
    # model_revision mismatch
    arm = [dict(r, model_revision="f" * 40) for r in _acc_rows("arm", [1, 0, 1, 1])]
    with pytest.raises(StatsError, match="model_revision"):
        ni.statuses_for_arm(arm, ref, registry, model="m", b=50)
    # task_version mismatch: the task config changed between legs
    dom = registry.domain_named("math")
    arm = ni.rows_frame([dict(r, task_version="v2")
                         for r in _acc_rows("arm", [1, 0, 1, 1])])
    with pytest.raises(StatsError, match="task_version"):
        ni.pair_task(arm, ni.rows_frame(ref), "gsm8k_platinum_cap", dom.kind)
    # mixed task_version WITHIN one leg is a refusal too
    rows = _acc_rows("arm", [1, 0, 1, 1])
    rows[0]["task_version"] = "v2"
    with pytest.raises(StatsError, match="mixed task_version"):
        ni.pair_task(ni.rows_frame(rows), ni.rows_frame(ref),
                     "gsm8k_platinum_cap", dom.kind)


# --------------------------------------------------------------------------
# the registered aggregate (F14)
# --------------------------------------------------------------------------


def test_domain_harm_accuracy_hand_example(registry):
    dom = registry.domain_named("math")
    pairs = [ni.pair_task(ni.rows_frame(_acc_rows("arm", [1, 0, 0, 0])),
                          ni.rows_frame(_acc_rows("none", [1, 1, 0, 1],
                                                  trial=0)),
                          "gsm8k_platinum_cap", "accuracy")]
    # harm = mean(ref - arm) = (0 + 1 + 0 + 1)/4 = 0.5
    assert ni.domain_harm(dom, pairs) == pytest.approx(0.5)


def test_domain_harm_nll_hand_example(registry):
    dom = registry.domain_named("lm_fidelity_chat")
    arm = ni.rows_frame(_nll_rows("arm", [220.0, 110.0], [100, 50]))
    ref = ni.rows_frame(_nll_rows("none", [200.0, 100.0], [100, 50], trial=0))
    pairs = ni.pair_domain(arm, ref, dom)
    # nll_arm = 330/150 = 2.2; nll_ref = 300/150 = 2.0; harm = 0.1 (F14:
    # Σnum/Σden recomputed from rows, not a mean of per-window ratios)
    assert ni.domain_harm(dom, pairs) == pytest.approx(0.1)


def test_domain_harm_task_weighting(registry):
    dom = registry.domain_named("commonsense")  # two tasks, equal weight
    arm = ni.rows_frame(
        _acc_rows("arm", [0, 0], task="tinyHellaswag", domain="commonsense")
        + _acc_rows("arm", [1, 1], task="tinyWinogrande", domain="commonsense"))
    ref = ni.rows_frame(
        _acc_rows("none", [1, 1], task="tinyHellaswag", domain="commonsense",
                  trial=0)
        + _acc_rows("none", [1, 1], task="tinyWinogrande",
                    domain="commonsense", trial=0))
    pairs = ni.pair_domain(arm, ref, dom)
    # per-task harms 1.0 and 0.0, equal task_weight -> 0.5
    assert ni.domain_harm(dom, pairs) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Bonferroni quantile inversion + statuses
# --------------------------------------------------------------------------


def test_bonferroni_quantile_inversion_hand_example(registry):
    """The k-family bound is the 1−α/k empirical quantile of the harm
    bootstrap — verified against numpy on the same seeded draws."""
    dom = registry.domain_named("math")
    arm_vals = [1] * 60 + [0] * 40
    ref_vals = [1] * 80 + [0] * 20
    arm = ni.rows_frame(_acc_rows("arm", arm_vals))
    ref = ni.rows_frame(_acc_rows("none", ref_vals, trial=0))
    pairs = ni.pair_domain(arm, ref, dom)
    rng = np.random.default_rng(0)
    boots = ni.bootstrap_harms(dom, pairs, b=4000, rng=rng)
    st = ni.domain_status(arm, ref, dom, margin=0.02, k_family=5,
                          in_gating_set=True, alpha=0.05, b=4000, seed=0)
    assert st.upper_bound == pytest.approx(float(np.quantile(boots, 0.99)))
    assert st.lower_bound == pytest.approx(float(np.quantile(boots, 0.01)))
    assert st.harm == pytest.approx(0.2)
    assert st.status == "degraded"  # lower(1%) > 0 and 0.2 > 0.02
    assert st.bound_method.startswith("percentile-bootstrap approximate")
    # unadjusted (k=1) bounds are strictly inside the k=5 bounds
    st1 = ni.domain_status(arm, ref, dom, margin=0.02, k_family=1,
                           in_gating_set=False, alpha=0.05, b=4000, seed=0)
    assert st1.upper_bound <= st.upper_bound
    assert st1.lower_bound >= st.lower_bound


def test_status_classification_rules():
    assert ni.classify(harm=0.0, upper=0.01, lower=-0.01, margin=0.02) \
        == "preserved"
    assert ni.classify(harm=0.05, upper=0.08, lower=0.02, margin=0.02) \
        == "degraded"
    # CI spans the margin -> indeterminate
    assert ni.classify(harm=0.015, upper=0.04, lower=-0.01, margin=0.02) \
        == "indeterminate"
    # point estimate below margin blocks "degraded" even with lower > 0
    assert ni.classify(harm=0.015, upper=0.019, lower=0.001, margin=0.02) \
        == "preserved"
    assert ni.classify(harm=0.015, upper=0.03, lower=0.001, margin=0.02) \
        == "indeterminate"


def test_statuses_for_arm_families_and_roles(registry):
    arm_rows = (_acc_rows("arm", [1, 0, 0, 0, 1, 0])
                + _acc_rows("arm", [1, 1, 1, 0, 1, 1], task="tinyMMLU",
                            domain="knowledge")
                + _acc_rows("arm", [0, 0, 0, 0, 0, 1],
                            task="s2parity/gsm8k_0shot",
                            domain="math_s2parity"))
    ref_rows = (_acc_rows("none", [1, 1, 1, 1, 1, 1], trial=0)
                + _acc_rows("none", [1, 1, 1, 0, 1, 1], task="tinyMMLU",
                            domain="knowledge", trial=0)
                + _acc_rows("none", [1, 1, 1, 1, 1, 1],
                            task="s2parity/gsm8k_0shot",
                            domain="math_s2parity", trial=0))
    statuses = ni.statuses_for_arm(arm_rows, ref_rows, registry, model="m",
                                   b=500, seed=0)
    assert set(statuses) == {"math", "knowledge", "math_s2parity"}
    # gating rows carry the family size (9 gating domains), acceptance
    # rows are UNADJUSTED (own family, k=1, r3-3)
    assert statuses["math"].k_family == 9
    assert statuses["math"].in_gating_set
    assert statuses["math_s2parity"].k_family == 1
    assert not statuses["math_s2parity"].in_gating_set
    assert statuses["math_s2parity"].role == "acceptance"
    # an excluded gating domain drops from the family and goes unadjusted
    st2 = ni.statuses_for_arm(arm_rows, ref_rows, registry, model="m",
                              exclusions=["math"], b=500, seed=0)
    assert st2["math"].k_family == 1 and not st2["math"].in_gating_set
    assert st2["knowledge"].k_family == 8


def test_monotonicity_domains_do_not_interact(registry):
    """Tightening one domain's data never flips another's status at fixed
    inputs (§7.2): domain statuses are functions of their own rows given
    a fixed family size."""
    arm_a = _acc_rows("arm", [1, 0, 1, 0, 1, 0])
    arm_b = _acc_rows("arm", [1, 1, 0, 1, 1, 1], task="tinyMMLU",
                      domain="knowledge")
    ref_a = _acc_rows("none", [1, 1, 1, 1, 1, 1], trial=0)
    ref_b = _acc_rows("none", [1, 1, 1, 1, 1, 1], task="tinyMMLU",
                      domain="knowledge", trial=0)
    before = ni.statuses_for_arm(arm_a + arm_b, ref_a + ref_b, registry,
                                 model="m", b=500, seed=3)
    # improve domain A's data to perfect agreement
    arm_a2 = _acc_rows("arm", [1, 1, 1, 1, 1, 1])
    after = ni.statuses_for_arm(arm_a2 + arm_b, ref_a + ref_b, registry,
                                model="m", b=500, seed=3)
    assert before["knowledge"].status == after["knowledge"].status
    assert before["knowledge"].upper_bound == after["knowledge"].upper_bound


def test_mcnemar_goldens():
    def mk(ref_vals, arm_vals):
        arm = ni.rows_frame(_acc_rows("arm", arm_vals))
        ref = ni.rows_frame(_acc_rows("none", ref_vals, trial=0))
        return ni.pair_domain(
            arm, ref, Registry.load(
                ROOT / "configs" / "capeval" / "domains.yaml"
            ).domain_named("math"))

    # b=3, c=1: p = 2 * (C(4,0)+C(4,1)) / 2^4 = 0.625
    pairs = mk([1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 0])
    assert ni.mcnemar_exact_p(pairs) == pytest.approx(0.625)
    # no discordants: p = 1
    pairs = mk([1, 0, 1], [1, 0, 1])
    assert ni.mcnemar_exact_p(pairs) == pytest.approx(1.0)
    # b=5, c=0: p = 2 * C(5,0)/2^5 = 0.0625
    pairs = mk([1] * 5 + [0], [0] * 5 + [0])
    assert ni.mcnemar_exact_p(pairs) == pytest.approx(0.0625)


# --------------------------------------------------------------------------
# calibration: the trial-0-reference conditional null (§7.1)
# --------------------------------------------------------------------------


def _calib_rows(accs_by_trial, *, task="gsm8k_platinum_cap", domain="math"):
    rows = []
    for trial, accs in enumerate(accs_by_trial):
        rows.extend(_acc_rows("none", accs, task=task, domain=domain,
                              trial=trial))
    return rows


@pytest.fixture(scope="module")
def calib(registry):
    """6-trial synthetic calibration over math (accuracy, with real
    discordance) + lm_fidelity_chat (NLL, with per-window jitter)."""
    rng = np.random.default_rng(7)
    base = (rng.random(24) < 0.7).astype(float)
    trials = [base]
    for _ in range(5):
        flip = rng.random(24) < 0.10
        trials.append(np.where(flip, 1 - base, base))
    rows = _calib_rows(trials)
    sums0 = 2.0 * np.ones(16) * 100
    for trial in range(6):
        jitter = rng.normal(0, 0.4, size=16) if trial else np.zeros(16)
        rows.extend(_nll_rows("none", (sums0 + jitter).tolist(),
                              [100] * 16, trial=trial))
    return cal.build_calibration(rows, registry, b_boot=800, seed=0), trials


def test_calibration_structure_and_estimand(calib, registry):
    c, trials = calib
    assert c.estimand == "conditional_on_reference"
    assert c.reference_trial == 0 and c.n_contrasts == 5
    assert c.model_revision == REV and c.engine == "hf"
    assert c.harness_version == cal.harness_version(registry) == "domains-v1"
    assert set(c.domains) == {"math", "lm_fidelity_chat"}
    # config-identity fields (r1 stats-core #2)
    assert c.model_id == "m"
    assert c.task_versions == {"gsm8k_platinum_cap": "v1",
                               "tier0/nll_ultrachat_masked": "v1"}
    assert c.eval_config_digest is not None and len(c.eval_config_digest) == 16
    from sisteer.capeval.store import eval_config_digest
    assert c.eval_config_digest == eval_config_digest(
        c.task_versions, template_hash=None, gen_params=None)


def _evalplus_rows(*, trials, version, metric_def, task="evalplus/humaneval_plus",
                   arm="none", n_items=4):
    """One EvalPlus task's rows (``pass@1_base`` / ``pass@1_plus`` / ``acc``)
    under one raw ``task_version``; ``metric_def`` rides in ``extra_json`` on
    the plus/acc rows (None = absent, the retained rows), ``pass@1_base``
    carries none by design. Values are constant so the calibration's
    accuracy machinery is untouched — the domain is a display domain and
    only the version map is under test."""
    import json as _json

    rows = []
    for trial in trials:
        for j in range(n_items):
            for metric in ("pass@1_base", "pass@1_plus", "acc"):
                extra = None
                if metric != "pass@1_base" and metric_def is not None:
                    extra = _json.dumps({"metric_def": metric_def, "source_metric": "pass@1_plus"})
                r = _row(arm=arm, task=task, domain="evalplus_served", item=f"HumanEval/{j}",
                         metric=metric, value=1.0, trial=trial)
                r.update(task_version=version, extra_json=extra)
                rows.append(r)
    return rows


def test_effective_task_version_reads_the_evalplus_column_with_the_row_metadata(registry):
    """PW14 (E22 H1): ``_registry_task_versions`` / ``eval_config_digest`` /
    ``build_fingerprint`` / ``pair_task`` read only the ``task_version``
    column, so an old plus-only calibration (bare ``evalplus-0.3.1``, no
    ``metric_def``) and a PW12-era base-and-plus run (the SAME bare version,
    ``metric_def: base_and_plus`` on its rows) got one eval-config identity
    and pooled. Every pooling reader now goes through
    ``noninferiority.effective_task_version``: bare + explicit
    ``base_and_plus`` IS the suffixed version (same digest as rows written
    under it), bare + absent is plus-only (a different version — never
    pooled with the suffixed one), and a mixed or contradictory group refuses,
    named — before calibrating or digesting."""
    from sisteer.capeval.store import eval_config_digest
    from sisteer.capeval.tinycap import contracts as tc

    he = "evalplus/humaneval_plus"
    BARE, SUFFIXED = tc.EVALPLUS_TASK_VERSION, tc.evalplus_row_task_version(tc.EVALPLUS_PLUS_DEF_BASE_AND_PLUS)
    BP, PO = tc.EVALPLUS_PLUS_DEF_BASE_AND_PLUS, tc.EVALPLUS_PLUS_DEF_PLUS_ONLY
    rng = np.random.default_rng(3)
    base = (rng.random(24) < 0.7).astype(float)
    trials = [base] + [np.where(rng.random(24) < 0.10, 1 - base, base) for _ in range(5)]
    math_rows = _calib_rows(trials)

    def calibration(version, metric_def):
        rows = math_rows + _evalplus_rows(trials=range(6), version=version, metric_def=metric_def)
        return cal.build_calibration(rows, registry, b_boot=200, seed=0)

    # bare + explicit base_and_plus (PW12-era rows) -> the suffixed version, the SAME identity as suffixed rows
    pw12 = calibration(BARE, BP)
    suffixed = calibration(SUFFIXED, None)
    assert pw12.task_versions[he] == SUFFIXED == suffixed.task_versions[he]
    assert pw12.eval_config_digest == suffixed.eval_config_digest == eval_config_digest(
        pw12.task_versions, template_hash=None, gen_params=None)
    assert calibration(SUFFIXED, BP).task_versions[he] == SUFFIXED                # the column and the rows agree
    # bare + absent (the retained rows on the volume) -> plus-only: a DIFFERENT version and digest, never pooled
    retained = calibration(BARE, None)
    assert retained.task_versions[he] == BARE and retained.eval_config_digest != pw12.eval_config_digest
    with pytest.raises(StatsError, match="absent from the calibration|task_version"):
        cal.check_calibration(retained, model_revision=REV, engine="hf",
                              harness_version=cal.harness_version(registry), task_versions={he: SUFFIXED})
    # the mixed group: two bare rows with CONFLICTING metadata -> refused, named, before any digest
    rows = math_rows + _evalplus_rows(trials=range(6), version=BARE, metric_def=None)
    rows[-1]["extra_json"] = '{"metric_def": "base_and_plus"}'                   # one PW12-era row among retained ones
    with pytest.raises(StatsError, match="MIXED metric_def"):
        cal.build_calibration(rows, registry, b_boot=200, seed=0)
    # bare-absent rows beside suffixed rows: two effective versions in one calibration -> the mixed refusal
    rows = math_rows + _evalplus_rows(trials=range(6), version=BARE, metric_def=None)
    rows += _evalplus_rows(trials=range(6), version=SUFFIXED, metric_def=None, task="evalplus/humaneval_plus")
    with pytest.raises(StatsError, match="mixed task_version"):
        cal.build_calibration(rows, registry, b_boot=200, seed=0)
    # ...while bare+base_and_plus rows beside suffixed rows are ONE instrument (the PW12 label bug), pooled
    rows = math_rows + _evalplus_rows(trials=range(6), version=BARE, metric_def=BP)
    rows += _evalplus_rows(trials=range(6), version=SUFFIXED, metric_def=None)
    assert cal.build_calibration(rows, registry, b_boot=200, seed=0).task_versions[he] == SUFFIXED
    # a suffixed version whose rows say plus_only: contradictory -> refused
    rows = math_rows + _evalplus_rows(trials=range(6), version=SUFFIXED, metric_def=PO)
    with pytest.raises(StatsError, match="contradictory"):
        cal.build_calibration(rows, registry, b_boot=200, seed=0)
    # the same reader at the pairing seam: legs under one EFFECTIVE version pair; plus-only vs suffixed refuse
    dom = registry.domain_named("evalplus_served")
    arm_pw12 = ni.rows_frame(_evalplus_rows(trials=[1], version=BARE, metric_def=BP, arm="arm"))
    ref_suffixed = ni.rows_frame(_evalplus_rows(trials=[1], version=SUFFIXED, metric_def=None))
    ref_retained = ni.rows_frame(_evalplus_rows(trials=[1], version=BARE, metric_def=None))
    assert ni.effective_task_version(arm_pw12, he, "arm") == SUFFIXED
    assert ni.effective_task_version(ref_retained, he, "reference") == BARE
    assert len(ni.pair_task(arm_pw12, ref_suffixed, he, "accuracy").item_ids) == 4
    with pytest.raises(StatsError, match="task_version"):
        ni.pair_task(arm_pw12, ref_retained, he, "accuracy")
    with pytest.raises(StatsError, match="task_version"):
        ni.pair_task(ni.rows_frame(_evalplus_rows(trials=[1], version=BARE, metric_def=None, arm="arm")),
                     ref_suffixed, he, "accuracy")
    # a non-EvalPlus task is untouched by the metadata reading
    assert ni.effective_task_version(ni.rows_frame(_acc_rows("arm", [1, 0, 1, 1])), "gsm8k_platinum_cap", "arm") == "v1"
    assert dom.kind == "display"


def test_calibration_per_task_null_samples(calib):
    """r1 stats-core #4/#6: the calibration persists per-contrast,
    per-task null samples with run-level structure intact — the §7.4
    hierarchical simulation's inputs."""
    c, trials = calib
    tc = c.domains["math"].tasks["gsm8k_platinum_cap"]
    assert tc.n_items == 24
    # pair_task sorts by item_id (lexically), so compare through that
    # permutation — what matters is that ref/arm stay index-ALIGNED
    perm = sorted(range(24), key=lambda k: f"i{k}")
    assert tc.ref_values == [float(trials[0][k]) for k in perm]
    assert len(tc.arm_values_by_contrast) == 5
    for j, arm_vals in enumerate(tc.arm_values_by_contrast, start=1):
        assert arm_vals == [float(trials[j][k]) for k in perm]
    assert tc.base_mean == pytest.approx(float(np.mean(trials[0])))
    # the observed per-contrast domain harms are recorded
    harms = [float(np.mean(trials[0] - trials[j])) for j in range(1, 6)]
    assert c.domains["math"].contrast_harms == pytest.approx(harms)
    # nll: 5 contrasts of 16 per-window deltas in nats/token
    ntc = c.domains["lm_fidelity_chat"].tasks["tier0/nll_ultrachat_masked"]
    assert len(ntc.deltas_by_contrast) == 5
    assert all(len(d) == 16 for d in ntc.deltas_by_contrast)
    assert ntc.base_mean == pytest.approx(2.0)


def test_sigma_d_is_spread_of_observed_contrasts(calib):
    c, trials = calib
    # σ_d must equal the SD of the 5 OBSERVED contrast harms exactly
    harms = [float(np.mean(trials[0] - trials[j])) for j in range(1, 6)]
    assert c.domains["math"].sigma_d == pytest.approx(
        float(np.std(harms, ddof=1)))
    assert c.domains["math"].sigma_d_se > 0
    tails = c.domains["math"].tails
    assert set(tails) == {"p01", "p05", "p50", "p95", "p99"}
    assert tails["p01"] <= tails["p50"] <= tails["p99"]


def test_calibration_discordance_counts(calib):
    c, trials = calib
    dc = c.domains["math"]
    expected = sum(int(np.sum(trials[j] != trials[0])) for j in range(1, 6))
    assert dc.null_discordant == expected
    assert dc.null_pairs == 24 * 5
    assert dc.null_discordance_rate == pytest.approx(expected / (24 * 5))
    assert dc.base_mean == pytest.approx(float(np.mean(trials[0])))


def test_calibration_nll_sample(calib):
    c, _ = calib
    dc = c.domains["lm_fidelity_chat"]
    assert dc.null_window_deltas_sample is not None
    assert len(dc.null_window_deltas_sample) == 16 * 5
    assert dc.base_mean == pytest.approx(2.0)


def test_calibration_refusals(registry):
    # missing reference trial
    rows = _calib_rows([[1, 1], [1, 0]])
    rows = [r for r in rows if r["trial_id"] != 0]
    rows += _calib_rows([[]])  # no-op filler to keep list non-empty
    with pytest.raises(StatsError, match="reference"):
        cal.build_calibration([r for r in _calib_rows([[1, 1], [1, 0], [0, 1]])
                               if r["trial_id"] != 0] or rows,
                              registry, b_boot=50, seed=0)
    # fewer than 2 contrasts
    with pytest.raises(StatsError, match=">= 2"):
        cal.build_calibration(_calib_rows([[1, 1], [1, 0]]), registry,
                              b_boot=50, seed=0)
    # mixed engines
    mixed = _calib_rows([[1, 1], [1, 0], [0, 1]])
    mixed[0]["engine"] = "vllm"
    with pytest.raises(StatsError, match="mixed"):
        cal.build_calibration(mixed, registry, b_boot=50, seed=0)
    # a domain covered in some contrasts but missing from one
    rows = _calib_rows([[1, 1], [1, 0], [0, 1]])
    rows += _acc_rows("none", [1, 1], task="tinyMMLU", domain="knowledge",
                      trial=0)
    rows += _acc_rows("none", [1, 0], task="tinyMMLU", domain="knowledge",
                      trial=1)  # missing for trial 2
    with pytest.raises(StatsError, match="incomplete"):
        cal.build_calibration(rows, registry, b_boot=50, seed=0)


def test_growth_rule(registry):
    """Wildly different contrasts on a gating domain trip the SE(σ_d)
    rule; growth_needed lists it."""
    rng = np.random.default_rng(1)
    trials = [(rng.random(8) < 0.5).astype(float) for _ in range(4)]
    c = cal.build_calibration(_calib_rows(trials), registry, b_boot=400,
                              seed=0)
    dc = c.domains["math"]
    assert dc.needs_more_trials == (dc.sigma_d_se > 0.25 * dc.sigma_d)
    assert cal.growth_needed(c) == (["math"] if dc.needs_more_trials else [])


def test_growth_rule_fires_at_sigma_d_zero(registry):
    """The growth rule is SE(σ_d) > 0.25·σ_d, applied directly.
    Contrasts whose observed harms are all identical (σ_d = 0) but whose
    item bootstrap gives SE > 0 are an UNSTABLE zero — growth must fire,
    not be suppressed by a σ_d > 0 guard."""
    base = np.array([1.0] * 10 + [0.0] * 10)
    trials = [base]
    for j in range(3):
        t = base.copy()
        t[j] = 0.0        # one correct item flips down…
        t[10 + j] = 1.0   # …and one incorrect item flips up: harm_j = 0
        trials.append(t)
    c = cal.build_calibration(_calib_rows(trials), registry, b_boot=400,
                              seed=0)
    dc = c.domains["math"]
    assert dc.sigma_d == 0.0
    assert dc.sigma_d_se > 0.0
    assert dc.needs_more_trials is True
    assert cal.growth_needed(c) == ["math"]


def test_canary_thresholds_from_values():
    per_trial = {
        "max_rep_run_p95": [4.0, 5.0, 4.0, 6.0, 4.0, 5.0],  # harmful up
        "distinct3": [0.80, 0.78, 0.81, 0.79, 0.80, 0.80],  # harmful down
    }
    th = cal.canary_thresholds_from_values(per_trial)
    up = th["max_rep_run_p95"]
    assert up["worst_base"] == 6.0
    # p99 of harmful deltas [1, 0, 2, 0, 1] ≈ max = 2 -> fail-line ≈ 8
    assert up["fail_line"] == pytest.approx(6.0 + up["null_spread_p99"])
    assert up["null_spread_p99"] == pytest.approx(
        float(np.quantile([1.0, 0.0, 2.0, 0.0, 1.0], 0.99)))
    down = th["distinct3"]
    assert down["worst_base"] == 0.78
    assert down["fail_line"] <= 0.78  # pushed DOWN in the harmful direction
    with pytest.raises(StatsError, match="direction"):
        cal.canary_thresholds_from_values({"mystery_indicator": [1, 2]})


def test_kappa_is_the_unfloored_null_p99_annotation():
    """κ is a recorded annotation — the RAW p99 of the null
    contrasts' kl_coarse_lb, no floor. On a deterministic stack the null
    is exactly zero and the honest value is 0.0 (the old 1e-4 floor is
    what made the retired flag fire on every active arm)."""
    def kl_rows(vals_by_trial):
        rows = []
        for trial, vals in vals_by_trial.items():
            for j, v in enumerate(vals):
                rows.append(_row(arm="none", task="tier0/kl_coarse_wikitext103",
                                 domain="dist_shift", item=f"w{j}",
                                 metric="kl_coarse_lb", value=v, trial=trial,
                                 tier=0))
        return rows

    # degenerate zero null (deterministic replay) -> exactly 0.0, no floor
    rows = kl_rows({0: [0.0] * 4, 1: [0.0] * 4, 2: [0.0] * 4})
    assert cal.derive_kappa(rows, b_boot=200, seed=0) == 0.0
    # tiny null KLs -> tracks the p99 of contrast means, NOT a 1e-4 floor
    rows = kl_rows({0: [0.0] * 4, 1: [1e-7] * 4, 2: [2e-7] * 4})
    kappa = cal.derive_kappa(rows, b_boot=200, seed=0)
    assert 1e-7 <= kappa <= 2e-7
    # larger null KLs -> still the p99 of contrast means
    rows = kl_rows({0: [0.0] * 4, 1: [0.01, 0.02, 0.01, 0.02],
                    2: [0.015] * 4})
    kappa = cal.derive_kappa(rows, b_boot=200, seed=0)
    assert 0.01 <= kappa <= 0.02
    # +inf in the NULL contrasts is harness breakage -> refusal
    rows = kl_rows({0: [0.0] * 2, 1: [0.01, float("inf")], 2: [0.01, 0.01]})
    with pytest.raises(StatsError, match="inf"):
        cal.derive_kappa(rows, b_boot=100, seed=0)


def test_calibration_persistence_and_check(calib, tmp_path, registry):
    c, _ = calib
    path = cal.save_calibration(c, tmp_path / "calibration.json")
    back = cal.load_calibration(path)
    assert back.model_revision == c.model_revision
    assert back.domains["math"].sigma_d == c.domains["math"].sigma_d
    assert back.estimand == "conditional_on_reference"
    # the per-task null samples round-trip as dataclasses
    assert back.domains["math"].tasks["gsm8k_platinum_cap"] \
        == c.domains["math"].tasks["gsm8k_platinum_cap"]
    assert back.model_id == "m" and back.task_versions == c.task_versions
    assert back.eval_config_digest == c.eval_config_digest
    # missing file: the loud run-calibration-first error
    with pytest.raises(StatsError, match="refuses to run"):
        cal.load_calibration(tmp_path / "nope.json")
    # matching check passes; any key off refuses
    cal.check_calibration(back, model_revision=REV, engine="hf",
                          harness_version="domains-v1")
    for kw in ({"model_revision": "f" * 40}, {"engine": "vllm"},
               {"harness_version": "domains-v2"}):
        args = dict(model_revision=REV, engine="hf",
                    harness_version="domains-v1")
        args.update(kw)
        with pytest.raises(StatsError, match="calibration mismatch"):
            cal.check_calibration(back, **args)


def test_check_calibration_config_identity(calib):
    """r1 stats-core #2: model_id, per-task task_versions, and the
    eval_config_digest are config identity — a run naming any of them
    must match, and a task the null never covered refuses."""
    c, _ = calib
    ok = dict(model_revision=REV, engine="hf", harness_version="domains-v1")
    cal.check_calibration(c, **ok, model_id="m")
    with pytest.raises(StatsError, match="model_id"):
        cal.check_calibration(c, **ok, model_id="other-model")
    # a subset of the calibrated tasks at matching versions is fine
    cal.check_calibration(c, **ok,
                          task_versions={"gsm8k_platinum_cap": "v1"})
    # a changed task version refuses
    with pytest.raises(StatsError, match="task_version"):
        cal.check_calibration(c, **ok,
                              task_versions={"gsm8k_platinum_cap": "v2"})
    # a task the calibration never covered refuses (foreign instrument)
    with pytest.raises(StatsError, match="absent from the calibration"):
        cal.check_calibration(c, **ok, task_versions={"tinyMMLU": "v1"})
    # digest checked when the caller supplies one
    cal.check_calibration(c, **ok, eval_config_digest=c.eval_config_digest)
    with pytest.raises(StatsError, match="eval_config_digest"):
        cal.check_calibration(c, **ok, eval_config_digest="0" * 16)


# --------------------------------------------------------------------------
# target status (DESIGN §2.1)
# --------------------------------------------------------------------------


def _primary(min_effect=0.10, direction="up", metric="si_compliance"):
    return PrimarySpec(metric=metric, plugin="sisteer_compliance", params={},
                       direction=direction, min_effect=min_effect)


def _leg(values):
    """values -> the {item_id: value} mapping target_status pairs by."""
    return {f"p{j}": float(v) for j, v in enumerate(values)}


def test_target_status_established():
    base = _leg([0.0] * 50)
    arm = _leg([1.0] * 20 + [0.0] * 30)  # delta 0.40 >> 0.10
    status, results = fp.target_status([(_primary(), base, arm)], b=2000,
                                       seed=0)
    assert status is fp.TargetStatus.ESTABLISHED
    assert results[0].delta == pytest.approx(0.4)
    assert results[0].holm_lower >= 0.10
    assert 0.0 < results[0].p_value < 0.05


def test_target_status_excluded_and_indeterminate():
    base = _leg([0.0] * 50)
    status, _ = fp.target_status([(_primary(), base, _leg([0.0] * 50))],
                                 b=2000, seed=0)
    assert status is fp.TargetStatus.EXCLUDED  # upper < min_effect
    # small n, borderline effect -> indeterminate
    status, _ = fp.target_status(
        [(_primary(), _leg([0.0] * 8),
          _leg([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]))],
        b=2000, seed=0)
    assert status is fp.TargetStatus.INDETERMINATE


def test_target_status_direction_down_and_holm():
    # direction down: delta = base - arm
    base, arm = _leg([1.0] * 40), _leg([0.0] * 40)
    status, res = fp.target_status(
        [(_primary(direction="down"), base, arm)], b=1000, seed=0)
    assert status is fp.TargetStatus.ESTABLISHED and res[0].delta == 1.0
    # Holm across two primaries: one strong + one flat -> NOT established,
    # and the flat one's plain upper bound < min_effect -> excluded
    strong = (_primary(metric="p1"), _leg([0.0] * 40), _leg([1.0] * 40))
    flat = (_primary(metric="p2"), _leg([0.0] * 40), _leg([0.0] * 40))
    status, res = fp.target_status([strong, flat], b=1000, seed=0)
    assert status is fp.TargetStatus.EXCLUDED


def test_target_status_pairs_by_item_id(registry):
    """r1 stats-core #7: legs pair BY ITEM ID — the same values keyed in
    a different order give the identical paired delta, mismatched item
    sets refuse, and bare positional sequences refuse outright."""
    from sisteer.capeval.targets import TargetLegScores

    rng = np.random.default_rng(5)
    ids = [f"it{j}" for j in range(30)]
    base_vals = (rng.random(30) < 0.3).astype(float)
    arm_vals = (rng.random(30) < 0.8).astype(float)
    base = dict(zip(ids, base_vals))
    arm = dict(zip(ids, arm_vals))
    status1, res1 = fp.target_status([(_primary(), base, arm)], b=500, seed=0)
    # arm leg handed over in REVERSED insertion order: identical result
    arm_shuffled = dict(reversed(list(arm.items())))
    status2, res2 = fp.target_status([(_primary(), base, arm_shuffled)],
                                     b=500, seed=0)
    assert status1 is status2 and res1[0].delta == res2[0].delta
    # TargetLegScores legs are first-class
    mk = lambda name, m: TargetLegScores(  # noqa: E731
        metric="si_compliance", plugin="sisteer_compliance", arm_name=name,
        item_ids=tuple(m), values=tuple(m.values()), params_resolved={})
    status3, res3 = fp.target_status(
        [(_primary(), mk("none", base), mk("arm", arm_shuffled))],
        b=500, seed=0)
    assert status3 is status1 and res3[0].delta == res1[0].delta
    # mismatched item sets refuse, naming the differences
    arm_missing = {k: v for k, v in arm.items() if k != "it3"}
    arm_missing["it99"] = 1.0
    with pytest.raises(StatsError, match="different item sets"):
        fp.target_status([(_primary(), base, arm_missing)], b=100, seed=0)
    # bare positional sequences are the r1 #7 hazard -> refusal
    with pytest.raises(StatsError, match="mapping"):
        fp.target_status([(_primary(), [0.0] * 3, [1.0] * 3)], b=100, seed=0)


def test_holm_orders_by_p_value_not_bound_magnitude():
    """r1 stats-core #8: Holm must order by one-sided evidence against
    each primary's OWN min_effect. A large-scale primary with a big raw
    lower bound but BORDERLINE evidence must take the LENIENT level;
    ordering by bound magnitude would hand it the strict α/p level and
    flip an establishable pair to indeterminate."""
    rng = np.random.default_rng(11)
    # primary A: big units, evidence tuned to z ≈ 1.8 against min_effect 1
    # -> one-sided p ≈ 0.036: clears α = 0.05, FAILS α/2 = 0.025
    a_deltas = rng.normal(0.0, 20.0, size=44)
    a_deltas -= a_deltas.mean()  # exact zero mean
    se = float(a_deltas.std(ddof=1)) / np.sqrt(44)
    a_deltas += 1.0 + 1.8 * se
    # primary B: small units, overwhelming evidence vs min_effect 0.05
    b_deltas = rng.normal(0.5, 0.02, size=44)
    a = (_primary(metric="big_units", min_effect=1.0),
         _leg(np.zeros(44)), _leg(a_deltas))
    bb = (_primary(metric="small_units", min_effect=0.05),
          _leg(np.zeros(44)), _leg(b_deltas))
    status, res = fp.target_status([a, bb], b=4000, seed=3)
    by_metric = {r.metric: r for r in res}
    big, small = by_metric["big_units"], by_metric["small_units"]
    # the discriminating construction holds: big has the LARGER raw lower
    # bound (bound-magnitude ordering would test it first, at α/2, where
    # its borderline p fails) but the WEAKER evidence
    assert big.lower_95 > small.lower_95
    assert small.p_value < 0.01
    assert 0.025 < big.p_value <= 0.05
    # p-ordered Holm: small tested at α/2 (clears), big at α (clears)
    assert status is fp.TargetStatus.ESTABLISHED


# --------------------------------------------------------------------------
# end-to-end synthetic fingerprint (v0 output shape)
# --------------------------------------------------------------------------


def test_build_fingerprint_end_to_end(calib, registry):
    c, trials = calib
    ref_rows = (_acc_rows("none", trials[0].tolist(), trial=0)
                + _nll_rows("none", [200.0] * 16, [100] * 16, trial=0))
    # arm: math badly degraded, chat NLL clean
    arm_rows = (_acc_rows("arm", (trials[0] * 0).tolist())
                + _nll_rows("arm", [200.5] * 16, [100] * 16))
    out = fp.build_fingerprint(
        arm_rows, ref_rows, registry, c, model="m", arm_name="arm",
        b=500, seed=0)
    assert out["schema_version"] == 2  # readers refuse unknown versions
    assert out["model_id"] == "m"
    assert out["verdict"] is None  # v0 NEVER fills this
    assert out["kl_coarse_lb"] is None  # no KL rows in this cell
    assert out["screen"] == "SCREEN-FAIL"  # math point-harm >> margin
    assert out["domains"]["math"]["status"] == "degraded"
    assert out["domains"]["math"]["z_display"] is not None
    # the writer's bound field names are the frozen reader contract
    # (simplify-bug #6): upper_bound/lower_bound, never renamed silently
    assert {"upper_bound", "lower_bound"} <= set(out["domains"]["math"])
    assert out["domains"]["lm_fidelity_chat"]["harm"] < 0.03
    # every uncovered registered domain is named in the gate log
    logged = " ".join(out["gate_log"])
    for name in ("knowledge", "code", "reasoning", "math_s2parity"):
        assert name in logged
    # calibration mismatch refuses (foreign engine null)
    import dataclasses as dcs
    foreign = dcs.replace(c, engine="vllm")
    with pytest.raises(StatsError, match="calibration mismatch"):
        fp.build_fingerprint(arm_rows, ref_rows, registry, foreign,
                             model="m", arm_name="arm", b=50, seed=0)
    # a foreign model_id null refuses even at matching revision/engine
    foreign_model = dcs.replace(c, model_id="other-model")
    with pytest.raises(StatsError, match="model_id"):
        fp.build_fingerprint(arm_rows, ref_rows, registry, foreign_model,
                             model="m", arm_name="arm", b=50, seed=0)
    # a null derived under a different task_version refuses (r1 #2)
    foreign_tv = dcs.replace(
        c, task_versions={**c.task_versions, "gsm8k_platinum_cap": "v0"})
    with pytest.raises(StatsError, match="task_version"):
        fp.build_fingerprint(arm_rows, ref_rows, registry, foreign_tv,
                             model="m", arm_name="arm", b=50, seed=0)


def test_build_fingerprint_screen_pass_when_clean(calib, registry):
    c, trials = calib
    ref_rows = _acc_rows("none", trials[0].tolist(), trial=0)
    arm_rows = _acc_rows("arm", trials[0].tolist())  # identical
    out = fp.build_fingerprint(arm_rows, ref_rows, registry, c, model="m",
                               arm_name="arm", b=500, seed=0)
    assert out["screen"] == "SCREEN-PASS"
    assert out["domains"]["math"]["harm"] == pytest.approx(0.0)
    # confirmed tier-0 (canary) flag flips the screen regardless of
    # domain harms — the caller derives this bit from the canary alone,
    # never from KL flags
    out2 = fp.build_fingerprint(arm_rows, ref_rows, registry, c, model="m",
                                arm_name="arm", tier0_flag_confirmed=True,
                                b=500, seed=0)
    assert out2["screen"] == "SCREEN-FAIL"


def test_build_fingerprint_kl_annotation_is_reported_not_screened(calib,
                                                                  registry):
    """kl_coarse_lb_corpus rows surface as a per-arm ANNOTATION
    (per-corpus magnitudes + mean) and leave the screen untouched — a
    huge KL magnitude on a behaviorally-clean arm still SCREEN-PASSes."""
    import dataclasses as dcs

    c, trials = calib
    kl_task = "tier0/kl_coarse_wikitext103"
    c = dcs.replace(c, task_versions={**c.task_versions, kl_task: "v1"})
    ref_rows = _acc_rows("none", trials[0].tolist(), trial=0)
    arm_rows = _acc_rows("arm", trials[0].tolist())  # behaviorally clean
    arm_rows.append(_row(arm="arm", task=kl_task, domain="dist_shift",
                         item="_corpus", metric="kl_coarse_lb_corpus",
                         value=0.86, tier=0))
    out = fp.build_fingerprint(arm_rows, ref_rows, registry, c, model="m",
                               arm_name="arm", b=500, seed=0)
    ann = out["kl_coarse_lb"]
    assert ann["per_corpus"] == {kl_task: pytest.approx(0.86)}
    assert ann["mean"] == pytest.approx(0.86)
    assert ann["unit"] == "nats_per_position"
    assert "annotation" in ann["note"]
    assert out["screen"] == "SCREEN-PASS"  # KL magnitude never screens


# --------------------------------------------------------------------------
# t0only-rung pairing seam (controls acceptance; integrator 2026-08-15)
# --------------------------------------------------------------------------


def test_pair_domain_whole_domain_absent_in_arm_needs_the_flag(registry):
    """A tier-0-only control rung carries none of a Tier-1 domain's tasks
    while the trial-0 reference carries all of them: default = strict
    refusal (real arms); allow_arm_task_subset=True = domain unmeasured
    (returns [])."""
    arm = ni.rows_frame(_nll_rows("rung", [10.0, 12.0], [100, 110]))
    ref = ni.rows_frame(
        _nll_rows("none", [10.0, 11.0], [100, 108], trial=0)
        + _acc_rows("none", [1, 1, 1], trial=0))
    math_dom = registry.domain_named("math")
    with pytest.raises(StatsError, match="reference rows only"):
        ni.pair_domain(arm, ref, math_dom)
    assert ni.pair_domain(arm, ref, math_dom, allow_arm_task_subset=True) == []
    # the nll domain both legs carry still pairs under the flag
    chat = registry.domain_named("lm_fidelity_chat")
    assert len(ni.pair_domain(arm, ref, chat, allow_arm_task_subset=True)) == 1


def test_statuses_for_arm_t0only_rung_scores_tier0_domains_only(registry):
    arm = _nll_rows("rung", [10.0, 12.0], [100, 110])
    ref = (_nll_rows("none", [10.0, 11.0], [100, 108], trial=0)
           + _acc_rows("none", [1, 1, 1], trial=0))
    with pytest.raises(StatsError, match="reference rows only"):
        ni.statuses_for_arm(arm, ref, registry, model="m", b=50)
    statuses = ni.statuses_for_arm(arm, ref, registry, model="m", b=50,
                                   allow_arm_task_subset=True)
    assert set(statuses) == {"lm_fidelity_chat"}  # only Tier-0 evidence


def test_partial_task_subset_still_refuses_even_with_the_flag(registry):
    """The flag is a DOMAIN-level allowance: an arm leg carrying only part
    of a domain's task set is a silently-missing-task hazard and refuses
    with or without it (a full arm with a missing task keeps the strict
    behavior)."""
    common = registry.domain_named("commonsense")
    arm = ni.rows_frame(_acc_rows("arm", [1, 0, 1], task="tinyHellaswag",
                                  domain="commonsense"))
    ref = ni.rows_frame(
        _acc_rows("none", [1, 1, 1], task="tinyHellaswag",
                  domain="commonsense", trial=0)
        + _acc_rows("none", [1, 1, 1], task="tinyWinogrande",
                    domain="commonsense", trial=0))
    for kw in ({}, {"allow_arm_task_subset": True}):
        with pytest.raises(StatsError, match="reference rows only"):
            ni.pair_domain(arm, ref, common, **kw)
    # and an ARM-only task (incomplete reference) always refuses too
    ref_missing = ni.rows_frame(_acc_rows("none", [1, 1, 1], trial=0,
                                          task="tinyWinogrande",
                                          domain="commonsense"))
    arm_only = ni.rows_frame(
        _acc_rows("arm", [1, 0, 1], task="tinyHellaswag", domain="commonsense")
        + _acc_rows("arm", [1, 0, 1], task="tinyWinogrande", domain="commonsense"))
    for kw in ({}, {"allow_arm_task_subset": True}):
        with pytest.raises(StatsError, match="arm rows only"):
            ni.pair_domain(arm_only, ref_missing, common, **kw)
