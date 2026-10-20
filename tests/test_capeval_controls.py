"""Control-arm tests (seeded reproducibility of
constructed arms; norm targeting; ladder manifesting; the acceptance-
checker logic on synthetic fingerprints). Base lane; model = tiny_llama.
"""

import json
from pathlib import Path

import pytest
import torch

from sisteer.capeval.manifest import AblationSpec
from sisteer.capeval.stats import controls as cc
from sisteer.methods import VectorObject


def _spec(kind, seed=7, **params):
    return AblationSpec(name=f"t_{kind.split(':')[1]}", kind=kind, seed=seed,
                        params=params, coeffs=[1.0])


# ---------------------------------------------------------------------------
# constructors: seeded reproducibility + norm targeting
# ---------------------------------------------------------------------------


def test_random_vector_seeded_and_norm_matched(tiny_llama):
    model, tok = tiny_llama
    a = cc.build_control(_spec("control:random_vector", seed=7, layer=1,
                               norm_target=4.0), model, tok)
    b = cc.build_control(_spec("control:random_vector", seed=7, layer=1,
                               norm_target=4.0), model, tok)
    c = cc.build_control(_spec("control:random_vector", seed=8, layer=1,
                               norm_target=4.0), model, tok)
    assert isinstance(a, VectorObject)  # the REAL vector-arm apply path
    va, vb, vc = (o.sv.vectors[1] for o in (a, b, c))
    assert torch.equal(va, vb)          # same seed -> bit-identical
    assert not torch.equal(va, vc)      # different seed -> different arm
    assert float(va.norm()) == pytest.approx(4.0, rel=1e-6)  # norm targeting
    assert a.skip_bos is True           # S2-exact apply semantics


def test_random_vector_dimension_defaults_to_hidden_size(tiny_llama):
    model, tok = tiny_llama
    obj = cc.build_control(_spec("control:random_vector", layer=1,
                                 norm_target=2.0), model, tok)
    assert obj.sv.vectors[1].shape == (16,)  # tiny hidden size


def test_noise_lora_frobenius_targeting_and_reproducibility(tiny_llama):
    model, tok = tiny_llama
    obj = cc.build_control(_spec("control:noise_lora", rank=4), model, tok)
    eps = 1e-2
    params = dict(model.named_parameters())
    before = {n: params[n].detach().clone() for n in obj._touched}
    with obj.apply(model, eps):
        for n in obj._touched:
            delta = params[n].detach() - before[n]
            ratio = float(delta.norm()) / float(before[n].norm())
            assert ratio == pytest.approx(eps, rel=1e-4), n  # ‖ΔW‖F/‖W‖F = ε
        during = {n: params[n].detach().clone() for n in obj._touched}
    # bit-exact restore
    for n in obj._touched:
        assert torch.equal(params[n].detach(), before[n])
    # same seed reproduces the identical mutation
    obj2 = cc.build_control(_spec("control:noise_lora", rank=4), model, tok)
    with obj2.apply(model, eps):
        for n in obj2._touched:
            assert torch.equal(params[n].detach(), during[n]), n


def test_noise_lora_touches_o_proj_of_every_layer(tiny_llama):
    model, tok = tiny_llama
    obj = cc.build_control(_spec("control:noise_lora"), model, tok)
    assert len(obj._touched) == model.config.num_hidden_layers
    assert all("o_proj" in n for n in obj._touched)
    assert obj.rank == 16  # §8 default


def test_weight_noise_touches_every_matrix_seeded(tiny_llama):
    model, tok = tiny_llama
    obj = cc.build_control(_spec("control:weight_noise", seed=3), model, tok)
    expected = sorted(n for n, p in model.named_parameters() if p.dim() == 2)
    assert obj._touched == expected  # every weight matrix, stable order
    params = dict(model.named_parameters())
    before = {n: params[n].detach().clone() for n in expected}
    with obj.apply(model, 3e-3):
        changed = {n: params[n].detach().clone() for n in expected}
        assert all(not torch.equal(before[n], changed[n]) for n in expected)
    for n in expected:
        assert torch.equal(params[n].detach(), before[n])
    obj2 = cc.build_control(_spec("control:weight_noise", seed=3), model, tok)
    with obj2.apply(model, 3e-3):
        for n in expected:
            assert torch.equal(params[n].detach(), changed[n])


def test_weight_mutation_cached_at_construction_never_regenerated(
        tiny_llama, monkeypatch):
    """The perturbation is built ONCE at
    construction — score_corpus enters a fresh ctx per item forward, so
    per-apply regeneration (torch.randn over ~all weights, B·Aᵀ per
    layer) is terabytes of waste on a real corpus. After construction,
    entering the ctx must not generate any randomness and must apply the
    bit-identical cached mutation."""
    model, tok = tiny_llama
    lora = cc.build_control(_spec("control:noise_lora", rank=4), model, tok)
    noise = cc.build_control(_spec("control:weight_noise"), model, tok)
    params = dict(model.named_parameters())
    first = {}
    for obj, eps in ((lora, 1e-2), (noise, 3e-3)):
        name = obj._touched[0]
        with obj.apply(model, eps):
            first[obj.kind] = params[name].detach().clone()

    def boom(*a, **kw):
        raise AssertionError(
            "torch.randn called inside apply() — the perturbation must be "
            "cached at construction (r1 finding 2)")

    monkeypatch.setattr(torch, "randn", boom)
    for obj, eps in ((lora, 1e-2), (noise, 3e-3)):
        name = obj._touched[0]
        with obj.apply(model, eps):
            assert torch.equal(params[name].detach(), first[obj.kind])


def test_ladder_manifesting_scales_linearly(tiny_llama):
    """One constructed arm serves the whole ε ladder: apply(scale=ε_i)
    per rung, ΔW scaling linearly with ε (the coeff IS ε, §8)."""
    model, tok = tiny_llama
    obj = cc.build_control(_spec("control:noise_lora", rank=2), model, tok)
    params = dict(model.named_parameters())
    name = obj._touched[0]
    base = params[name].detach().clone()
    norms = []
    for eps in (1e-4, 1e-3, 1e-2):
        with obj.apply(model, eps):
            norms.append(float((params[name].detach() - base).norm()))
    assert norms[1] == pytest.approx(norms[0] * 10, rel=1e-3)
    assert norms[2] == pytest.approx(norms[0] * 100, rel=1e-3)


def test_zero_coeff_requires_real_artifact(tiny_llama, tmp_path):
    model, tok = tiny_llama
    with pytest.raises(cc.ControlError, match="params.artifact"):
        cc.build_control(_spec("control:zero_coeff"), model, tok)
    with pytest.raises(cc.ControlError, match="does not exist"):
        cc.build_control(_spec("control:zero_coeff",
                               artifact=str(tmp_path / "nope.safetensors")),
                         model, tok)
    # a REAL artifact loads through the full load_object path
    from sisteer.vectors import SteeringVector
    p = tmp_path / "real.safetensors"
    SteeringVector(vectors={1: torch.randn(16)}, method="distill_soft",
                   model_name="m", instruction_id="i").save(p)
    obj = cc.build_control(_spec("control:zero_coeff", artifact=str(p)),
                           model, tok)
    assert isinstance(obj, VectorObject)


def test_build_control_refusals(tiny_llama):
    model, tok = tiny_llama
    with pytest.raises(cc.ControlError, match="unknown control kind"):
        cc.build_control(_spec("control:quantize"), model, tok)
    with pytest.raises(cc.ControlError, match="seed"):
        cc.build_control(AblationSpec(name="x", kind="control:sham_hook",
                                      params={"layer": 1}), model, tok)
    with pytest.raises(cc.ControlError, match="unknown params"):
        cc.build_control(_spec("control:sham_hook", layer=1, extra=2),
                         model, tok)
    with pytest.raises(cc.ControlError, match="params.layer"):
        cc.build_control(_spec("control:sham_hook"), model, tok)


def test_median_sidecar_norm(tmp_path):
    paths = []
    for i, norms in enumerate([{"15": 3.0}, {"15": 5.0}, {"15": 9.0}]):
        p = tmp_path / f"s{i}.json"
        p.write_text(json.dumps({"layers": [15], "norms": norms}))
        paths.append(p)
    assert cc.median_sidecar_norm(paths, layer=15) == 5.0
    assert cc.median_sidecar_norm(paths) == 5.0  # single-norm sidecars
    scalar = tmp_path / "scalar.json"
    scalar.write_text(json.dumps({"layer": 15, "norm": 7.0}))
    assert cc.median_sidecar_norm([scalar]) == 7.0
    multi = tmp_path / "multi.json"
    multi.write_text(json.dumps({"norms": {"10": 1.0, "15": 2.0}}))
    with pytest.raises(cc.ControlError, match="explicit layer"):
        cc.median_sidecar_norm([multi])
    with pytest.raises(cc.ControlError, match="no stored norm for layer 20"):
        cc.median_sidecar_norm(paths, layer=20)
    with pytest.raises(cc.ControlError, match="no sidecars"):
        cc.median_sidecar_norm([])


# ---------------------------------------------------------------------------
# acceptance checker on synthetic fingerprints
# ---------------------------------------------------------------------------

BANDS = {"math": 0.03, "knowledge": 0.02, "lm_fidelity_raw": 0.01}
# the pinned-from-the-manifest top rungs: the checker
# refuses to derive them from whatever cells happen to exist
RV_TOP = {"control:random_vector": 4.0}
NOOP_RV_KINDS = ("control:zero_coeff", "control:sham_hook",
                 "control:random_vector")


def _obs(arm, kind, coeff, harms, tripped=0, confirmed=False, kl=None,
         tier0=None):
    return cc.ControlObservation(arm=arm, kind=kind, coeff=coeff,
                                 domain_harms=harms, canary_tripped=tripped,
                                 canary_confirmed=confirmed,
                                 kl_coarse_lb_max=kl,
                                 tier0_domain_harms=dict(tier0 or {}))


def _clean_noops():
    return [_obs("zc", "control:zero_coeff", 0.0,
                 {"math": 0.001, "knowledge": -0.002, "lm_fidelity_raw": 0.0}),
            _obs("sham", "control:sham_hook", 1.0,
                 {"math": -0.001, "knowledge": 0.0, "lm_fidelity_raw": 0.001})]


def _detectable_ladder():
    return [_obs("rv_05", "control:random_vector", 0.5, {"math": 0.0}),
            _obs("rv_4", "control:random_vector", 4.0,
                 {"math": 0.20, "lm_fidelity_raw": 0.10}, kl=0.55,
                 tier0={"lm_fidelity_raw": 0.10})]


def test_checker_passes_clean_controls():
    rep = cc.check_controls(_clean_noops() + _detectable_ladder(), BANDS,
                            RV_TOP, required_kinds=NOOP_RV_KINDS)
    assert rep.passed and not rep.hard_stop
    noop_results = [a for a in rep.arms if a.role == "noop"]
    assert all(a.passed for a in noop_results)


def test_checker_noop_out_of_band_is_harness_indicting():
    bad = _obs("zc", "control:zero_coeff", 0.0, {"math": 0.10})
    sham = _obs("sham", "control:sham_hook", 1.0, {"math": 0.0})
    rep = cc.check_controls([bad, sham] + _detectable_ladder(), BANDS,
                            RV_TOP, required_kinds=NOOP_RV_KINDS)
    assert not rep.passed and rep.hard_stop  # hard-stop the sweep (§8)
    [zc] = [a for a in rep.arms if a.arm == "zc"]
    assert not zc.passed and "math" in zc.reasons[0]


def test_checker_noop_canary_trip_is_harness_indicting():
    bad = _obs("sham", "control:sham_hook", 1.0, {"math": 0.0}, tripped=1)
    zc = _obs("zc", "control:zero_coeff", 0.0, {"math": 0.0})
    rep = cc.check_controls([bad, zc] + _detectable_ladder(), BANDS,
                            RV_TOP, required_kinds=NOOP_RV_KINDS)
    assert rep.hard_stop


def test_checker_noop_nonzero_kl_fails_the_exact_zero_certificate():
    """No-op role (i): a genuine no-op replays the base logits
    bit-identically, so ANY nonzero kl_coarse_lb on a no-op indicts the
    harness — even a value far below the retired κ floor. Exactly-zero
    KL (and no KL rows at all, for tier-less synthetic cells) passes."""
    bad = _obs("sham", "control:sham_hook", 1.0, {"math": 0.0}, kl=1e-6)
    zc = _obs("zc", "control:zero_coeff", 0.0, {"math": 0.0}, kl=0.0)
    rep = cc.check_controls([bad, zc] + _detectable_ladder(), BANDS,
                            RV_TOP, required_kinds=NOOP_RV_KINDS)
    assert not rep.passed and rep.hard_stop
    [sham] = [a for a in rep.arms if a.arm == "sham"]
    assert "exact-zero certificate" in sham.reasons[0]
    [zero] = [a for a in rep.arms if a.arm == "zc"]
    assert zero.passed


def test_checker_damage_undetectable_at_top_rung_fails():
    flat = [_obs("rv_05", "control:random_vector", 0.5, {"math": 0.0}),
            _obs("rv_4", "control:random_vector", 4.0,
                 {"math": 0.001, "lm_fidelity_raw": 0.0},
                 tier0={"lm_fidelity_raw": 0.0})]
    rep = cc.check_controls(_clean_noops() + flat, BANDS, RV_TOP,
                            required_kinds=NOOP_RV_KINDS)
    assert not rep.passed
    assert not rep.hard_stop  # damage-miss indicts sensitivity, not sanity
    [top] = [a for a in rep.arms if a.arm == "rv_4"]
    assert "cannot see known damage" in top.reasons[0]


def test_checker_tier1_harm_never_satisfies_tier0_detection():
    """A Tier-1 harm (e.g. MMLU beyond its band) at the top rung must
    NOT pass the checker — the requirement is detection AT TIER 0
    (tier0_domain_harms + KL/canary flags only)."""
    tier1_only = [_obs("rv_4", "control:random_vector", 4.0,
                       {"math": 0.20, "knowledge": 0.15},  # both Tier-1
                       tier0={})]
    rep = cc.check_controls(
        _clean_noops() + tier1_only, BANDS, RV_TOP,
        required_kinds=NOOP_RV_KINDS)
    assert not rep.passed
    [top] = [a for a in rep.arms if a.arm == "rv_4"]
    assert "cannot see known damage" in top.reasons[0]
    # the same harm IN a Tier-0 domain is evidence
    tier0_seen = [_obs("rv_4", "control:random_vector", 4.0,
                       {"math": 0.20, "lm_fidelity_raw": 0.05},
                       tier0={"lm_fidelity_raw": 0.05})]
    assert cc.check_controls(_clean_noops() + tier0_seen, BANDS, RV_TOP,
                             required_kinds=NOOP_RV_KINDS).passed


def test_checker_does_not_require_monotonicity():
    """§7.3: dose-response monotonicity is NOT required — a mid rung
    showing LESS damage than a lower rung must not fail the family."""
    wobbly = [
        _obs("nl_1", "control:noise_lora", 1e-4, {"math": 0.05}),   # up
        _obs("nl_2", "control:noise_lora", 3e-4, {"math": 0.0}),    # down!
        _obs("nl_3", "control:noise_lora", 3e-3, {"lm_fidelity_raw": 0.30},
             tier0={"lm_fidelity_raw": 0.30}),                      # top: seen
    ]
    rep = cc.check_controls(
        _clean_noops() + wobbly, BANDS, {"control:noise_lora": 3e-3},
        required_kinds=("control:zero_coeff", "control:sham_hook",
                        "control:noise_lora"))
    assert rep.passed


def test_checker_top_rung_canary_is_evidence_kl_never_is():
    """Confirmed canary alone satisfies top-rung detection; a KL
    magnitude alone — however large — never does: KL is an
    annotation, and a KL-only footprint is sub-behavioral, not damage.
    Formerly this test's `by_kl` case PASSED on κ flags over 2 corpora;
    the inversion is the accepted consequence of that rule."""
    kinds = ("control:zero_coeff", "control:sham_hook",
             "control:weight_noise")
    tops = {"control:weight_noise": 3e-3}
    by_canary = [_obs("wn", "control:weight_noise", 3e-3, {"math": 0.0},
                      confirmed=True)]
    assert cc.check_controls(_clean_noops() + by_canary, BANDS, tops,
                             required_kinds=kinds).passed
    by_kl_only = [_obs("wn", "control:weight_noise", 3e-3, {"math": 0.0},
                       kl=0.86)]
    rep = cc.check_controls(_clean_noops() + by_kl_only, BANDS, tops,
                            required_kinds=kinds)
    assert not rep.passed
    assert not rep.hard_stop  # sensitivity miss, not harness insanity
    [top] = [a for a in rep.arms if a.arm == "wn"]
    assert "cannot see known damage" in top.reasons[0]


def test_checker_requires_every_manifested_kind():
    """An absent required control kind (a failed or unconsolidated
    cell) is a refusal, never a silent skip — by default the full v0 §8
    five-kind set is required."""
    with pytest.raises(cc.ControlError, match="noise_lora"):
        cc.check_controls(_clean_noops() + _detectable_ladder(), BANDS,
                          RV_TOP)  # default required_kinds = all five
    with pytest.raises(cc.ControlError, match="zero_coeff"):
        cc.check_controls(_detectable_ladder(), BANDS, RV_TOP,
                          required_kinds=NOOP_RV_KINDS)


def test_checker_requires_pinned_top_rung_never_max_fallback():
    """With the 4.0 cell missing, the old max() fallback re-defined 0.5
    as 'top' and could pass the family — both an unpinned family and a
    pinned-but-unobserved rung must refuse."""
    only_low = _clean_noops() + [
        _obs("rv_05", "control:random_vector", 0.5, {"math": 0.0})]
    with pytest.raises(cc.ControlError, match="no pinned top rung"):
        cc.check_controls(only_low, BANDS, {},
                          required_kinds=NOOP_RV_KINDS)
    with pytest.raises(cc.ControlError, match="has no observation"):
        cc.check_controls(only_low, BANDS, RV_TOP,
                          required_kinds=NOOP_RV_KINDS)


def test_checker_refusals():
    with pytest.raises(cc.ControlError, match="no observations"):
        cc.check_controls([], BANDS, RV_TOP)
    with pytest.raises(cc.ControlError, match="no calibrated null band"):
        cc.check_controls([_obs("zc", "control:zero_coeff", 0.0,
                                {"mystery_domain": 0.0})], BANDS, {},
                          required_kinds=("control:zero_coeff",))
    with pytest.raises(cc.ControlError, match="pinned top rung"):
        cc.check_controls(_detectable_ladder(), BANDS,
                          top_rungs={"control:random_vector": 8.0},
                          required_kinds=("control:random_vector",))
    with pytest.raises(cc.ControlError, match="non-control"):
        cc.check_controls([cc.ControlObservation(
            arm="x", kind="steering", coeff=1.0, domain_harms={})], BANDS,
            RV_TOP, required_kinds=())
    with pytest.raises(cc.ControlError, match="not control kinds"):
        cc.check_controls(_clean_noops(), BANDS, {},
                          required_kinds=("control:quantize",))


def test_checker_report_json_shape():
    rep = cc.check_controls(_clean_noops() + _detectable_ladder(), BANDS,
                            RV_TOP, required_kinds=NOOP_RV_KINDS)
    payload = json.loads(rep.to_json())
    assert set(payload) == {"passed", "hard_stop", "arms"}
    assert all(set(a) == {"arm", "kind", "role", "passed", "reasons"}
               for a in payload["arms"])


def test_observations_from_store_tier0_filter_is_explicit(monkeypatch):
    """The builder carries a separate Tier-0-only harm map, filtered by
    an EXPLICIT tier0_domains parameter — and refuses names outside the
    registry (a typo must not silently filter to nothing)."""
    import pandas as pd
    from types import SimpleNamespace

    from sisteer.capeval.domains import Registry
    from sisteer.capeval.stats import noninferiority as ni

    registry = Registry.load(
        Path(__file__).parents[1] / "configs" / "capeval" / "domains.yaml")
    monkeypatch.setattr(
        ni, "statuses_for_arm",
        lambda *a, **kw: {"math": SimpleNamespace(harm=0.2),
                          "lm_fidelity_raw": SimpleNamespace(harm=0.05)})
    df = pd.DataFrame.from_records([
        {"kind": "control:weight_noise", "arm_name": "wn", "coeff": 3e-3,
         "attempt": 0, "metric": m, "value": v}
        for m, v in [("kl_coarse_lb_corpus", 0.5), ("flag_kl_gt_kappa", 1.0),
                     ("flag_kl_gt_kappa", 1.0), ("canary/tripped_count", 2.0)]
    ] + [{"kind": "control:weight_noise", "arm_name": "wn", "coeff": 3e-3,
          "attempt": 1, "metric": "flag_canary_confirmed", "value": 1.0}])

    [obs] = cc.observations_from_store(
        df, df, registry, model="m", tier0_domains=("lm_fidelity_raw",
                                                    "lm_fidelity_chat"))
    assert obs.domain_harms == {"math": 0.2, "lm_fidelity_raw": 0.05}
    assert obs.tier0_domain_harms == {"lm_fidelity_raw": 0.05}
    # the max per-corpus KL magnitude is carried (annotation + the
    # no-op certificate input); the stored legacy flag_kl_gt_kappa rows
    # are deliberately ignored
    assert obs.kl_coarse_lb_max == pytest.approx(0.5)
    assert obs.canary_tripped == 2 and obs.canary_confirmed

    with pytest.raises(cc.ControlError, match="not in the registry"):
        cc.observations_from_store(df, df, registry, model="m",
                                   tier0_domains=("lm_fidelity_typo",))


# ---------------------------------------------------------------------------
# THE regression the 115650/115651 failures exposed (2026-08-16): nothing
# ever constructed the arms of the REAL checked-in controls.yaml through
# build_control, so manifest params the constructor didn't pop (norm,
# tier1_coeffs, module, layers) refused at cluster runtime. Construct
# every real-yaml arm param-wise here (artifact/sidecar file access is
# stubbed to fixtures — the cluster paths don't exist on the grading host; the
# point is PARAM ACCEPTANCE and correct wiring, not artifact bytes).
# ---------------------------------------------------------------------------

def test_build_control_accepts_every_checked_in_arm(tiny_llama, monkeypatch,
                                                    tmp_path):
    import json

    from sisteer.capeval import manifest as cm
    from sisteer.capeval.stats import controls as ctl

    model, tok = tiny_llama
    repo = Path(__file__).parents[1]
    m = cm.load_manifest(repo / "configs" / "capeval" / "manifests"
                         / "controls.yaml")

    # stand-ins for cluster-only files: a REAL loadable SteeringVector
    # (layer_<k> keys, tiny-model dims) + a norms sidecar
    import torch
    from safetensors.torch import save_file

    d = model.config.hidden_size
    art = tmp_path / "vector.safetensors"
    save_file({"layer_1": torch.zeros(d)}, str(art))
    sidecar = tmp_path / "vector.json"
    sidecar.write_text(json.dumps(
        {"method": "test", "model_name": "tiny", "instruction_id": "t",
         "layers": [1], "norms": {"1": 7.5}, "metadata": {}}))

    built = []
    for ab in m.ablations:
        params = dict(ab.params)
        if "artifact" in params:
            params["artifact"] = str(art)
        if "norm" in params:
            params = {**params,
                      "norm": {**params["norm"], "layer": 1,
                               "sidecars": [str(sidecar)] * len(
                                   params["norm"]["sidecars"])}}
        if "d" in params:
            params["d"] = model.config.hidden_size  # tiny model dims
        spec = cm.AblationSpec(**{**ab.__dict__, "params": params})
        obj = ctl.build_control(spec, model, tok)
        assert obj is not None
        built.append(ab.kind)
    assert sorted(built) == sorted([
        "control:zero_coeff", "control:sham_hook", "control:random_vector",
        "control:noise_lora", "control:weight_noise"])
