"""Integration-seam tests: sisteer.capeval.runtime (the WS merge module)
+ the item-7 controls_acceptance payload shape + CellStore.append_ts.

Base lane only — no model, no heavy deps: resolve_cell is deliberately
CPU/metadata-only (it must key and resume-check a cell BEFORE the model
load pays off), so everything here runs on synthetic manifests and
the checked-in protocol fixtures.
"""

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "capeval"

spec = importlib.util.spec_from_file_location(
    "capeval_run_rt", REPO / "scripts" / "capeval_run.py")
capeval_run = importlib.util.module_from_spec(spec)
sys.modules["capeval_run_rt"] = capeval_run
spec.loader.exec_module(capeval_run)

from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval import runtime as rt  # noqa: E402
from sisteer.capeval import store as cs  # noqa: E402
from sisteer.capeval.tiers.tier0 import _TEMPLATE_KEYS  # noqa: E402

REV = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def _no_tokenizer(monkeypatch):
    # harness_config_hash covers the chat template; tests must not load
    # a real tokenizer (network/HF-cache) — pin a constant.
    monkeypatch.setattr(rt, "_chat_template", lambda m: "TEST-TEMPLATE")
    monkeypatch.delenv("CAPEVAL_CALIBRATION_JSON", raising=False)


def _controls_manifest():
    return cm.Manifest(
        sweep="rt_controls", mode="control", base="test-org/tiny",
        base_revision=REV, engine="hf", tiers=[0, 1], confirmatory=False,
        target=None, margins={},
        ablations=[
            cm.AblationSpec(name="sham", kind="control:sham_hook",
                            coeffs=[1.0], seed=0, params={"layer": 15}),
            cm.AblationSpec(name="nl", kind="control:noise_lora",
                            coeffs=[1e-4, 3e-3], seed=29,
                            params={"tier1_coeffs": [3e-3]}),
        ])


def _steering_manifest():
    return cm.Manifest(
        sweep="rt_sweep", mode="sweep", base="test-org/tiny",
        base_revision=REV, engine="hf", tiers=[0, 1], confirmatory=False,
        margins={},
        target=cm.TargetSpec(
            primaries=[cm.PrimarySpec(metric="si_compliance",
                                      plugin="sisteer_compliance",
                                      params={"si": "number_words"},
                                      direction="up", min_effect=0.1)],
            secondary=[], off_target_exclusions=[],
            relevant_corpus=cm.CorpusRef(
                slice={"first": 2, "field": "prompt"},
                path=str(FIXTURES / "pairs_synthetic.jsonl"))),
        ablations=[cm.AblationSpec(
            name="arm-a", kind="steering",
            artifact=str(FIXTURES / "vector_synthetic.safetensors"),
            layer=15, coeffs=[2.0])])


# ------------------------------------------------------------- resolve_cell


def test_resolve_cell_control_rungs_get_distinct_keys(tmp_path):
    m = _controls_manifest()
    cells = capeval_run.enumerate_cells(m)
    # noise_lora ladder: 2 t0only rungs + 1 tier1 rung share {kind,seed,
    # params} -> coeff-free control intervention hash; the tier token
    # must keep their cell dirs distinct.
    keys = set()
    for cell in cells:
        store, resolved = rt.resolve_cell(m, cell, 0, tmp_path, seed=100 + cell.index)
        keys.add(store.key)
        assert resolved["row_constants"] == {k: resolved[k] for k in _TEMPLATE_KEYS}
        assert json.loads(store.manifest_path.read_text())["row_constants"] \
            == resolved["row_constants"]
    assert len(keys) == len(cells)


def test_tier_token_format_pinned():
    [full, t0a, t0b, t1] = capeval_run.enumerate_cells(_controls_manifest())
    assert rt._tier_token(full) == "tier01-c1p0"        # sham, coeff 1.0
    assert rt._tier_token(t0a) == "tier0-c0p0001"       # 1e-4 rung
    assert rt._tier_token(t0b) == "tier0-c0p003"        # 3e-3 rung
    assert rt._tier_token(t1) == "tier1-c0p003"


def test_coeff_token_is_lossless():
    """Review tier1-runtime #6: %g kept six significant digits, so two
    valid distinct rungs collided on one cell dir. repr() round-trips
    every float; the mapped token must stay injective and shell-inert."""
    coeffs = [0.1234564, 0.12345649, 3e-3, 1e-4, 4.0, 1.0, 2.5e-7,
              123456789.0, -0.5]
    tokens = [rt._coeff_token(c) for c in coeffs]
    assert len(set(tokens)) == len(coeffs)  # the #6 pair no longer collides
    import re as _re

    for tok in tokens:
        assert _re.fullmatch(r"[0-9a-zA-Z_pme]+", tok), tok  # glob/shell-inert


def test_resolve_cell_resume_identity_and_done(tmp_path):
    m = _controls_manifest()
    cell = capeval_run.enumerate_cells(m)[0]
    store, resolved = rt.resolve_cell(m, cell, 0, tmp_path, seed=7)
    assert not store.is_done()
    store.write_results({"ok": True})
    # same identity resumes onto the same cell dir, now done
    store2, _ = rt.resolve_cell(m, cell, 0, tmp_path, seed=7)
    assert store2.key == store.key and store2.is_done()
    # a config edit re-keys the cell (chash8 in the key, F8)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rt, "_chat_template", lambda m: "DIFFERENT-TEMPLATE")
        store3, _ = rt.resolve_cell(m, cell, 0, tmp_path, seed=7)
    assert store3.key != store.key and not store3.is_done()


def test_append_ts_files_schema_complete_meta_rows(tmp_path):
    m = _controls_manifest()
    cell = capeval_run.enumerate_cells(m)[0]
    store, _ = rt.resolve_cell(m, cell, 0, tmp_path, seed=7)
    store.append_ts("job_start", 1755000000.0)
    store.append_ts("model_ready", 1755000010.0)
    [r0, r1] = store._parsed_rows()
    assert r0["metric"] == "ts_job_start" and r0["value"] == 1755000000.0
    assert r0["domain"] == cs.META_DOMAIN and r0["task"] == "_cell"
    assert set(r0) == set(cs.ROW_COLUMNS)  # fully schema'd
    assert r1["metric"] == "ts_model_ready"
    # a store built WITHOUT row_constants refuses (loud, not silent)
    bare = cs.CellStore(tmp_path, store.key)
    with pytest.raises(cs.StoreError):
        bare.append_ts("x", 0.0)


def test_steering_static_hash_moves_with_coeff_and_validates_sidecar(tmp_path):
    m = _steering_manifest()
    ab = m.ablations[0]
    h2 = rt.cell_intervention_hash(ab, 2.0)
    h4 = rt.cell_intervention_hash(ab, 4.0)
    assert h2 != h4 and len(h2) == cm.HASH_HEX_LEN
    # F9 at resolve time: manifest layer vs sidecar layer mismatch refuses
    bad = dataclasses.replace(ab, layer=14)
    with pytest.raises(Exception) as ei:
        rt.cell_intervention_hash(bad, 2.0)
    assert "F9" in str(ei.value) or "layer" in str(ei.value).lower()


def test_resolve_cell_sweep_uses_owning_document(tmp_path):
    m = _steering_manifest()
    cell = capeval_run.enumerate_cells(m)[0]
    store, resolved = rt.resolve_cell(m, cell, 0, tmp_path, seed=100)
    assert resolved["kind"] == "steering" and resolved["coeff"] == 2.0
    assert resolved["static_intervention_spec"]["skip_bos"] is True
    assert resolved["intervention_hash"] == rt.cell_intervention_hash(
        m.ablations[0], 2.0)


def test_calibration_missing_refuses_for_non_calibration_modes(tmp_path):
    # §7.1 authoritative-source rule: control/sweep Tier-0 without the
    # calibration null is a refusal, never a silent default.
    with pytest.raises(rt.RuntimeError_) as ei:
        rt._tier0_thresholds(_controls_manifest(), tmp_path / "rt_controls")
    assert "calibration" in str(ei.value)
    # calibration mode itself runs uncalibrated by design (its output);
    # only the canary thresholds are returned — κ feeds nothing
    calib_m = cm.Manifest(
        sweep="calibration", mode="calibration", base="test-org/tiny",
        base_revision=REV, engine="hf", tiers=[0, 1], confirmatory=False,
        target=None, margins={},
        ablations=[cm.AblationSpec(name="none", kind="none")])
    assert rt._tier0_thresholds(calib_m, tmp_path / "calibration") is None


# ------------------------------------------------- item 7: controls payload


def test_controls_acceptance_payload_matches_report_consumer(tmp_path):
    from sisteer.capeval.report import build as rb
    from sisteer.capeval.stats import controls as cc

    obs = [
        # the no-op: exactly-zero KL (the exact-zero certificate)
        cc.ControlObservation(arm="sham", kind="control:sham_hook", coeff=1.0,
                              domain_harms={"math": 0.001},
                              kl_coarse_lb_max=0.0,
                              tier0_damage={"kl_coarse_lb": 0.0}),
    ] + [
        # damage ladder: KL magnitudes ride as the dose-response DISPLAY
        # series; detection at the top rung comes from the confirmed
        # canary (KL is never detection evidence)
        cc.ControlObservation(arm="rand-vec", kind="control:random_vector",
                              coeff=c, domain_harms={"math": h},
                              canary_confirmed=(c == 4.0),
                              kl_coarse_lb_max=d,
                              tier0_damage={"kl_coarse_lb": d})
        for c, h, d in [(0.5, 0.0, 1e-4), (1.0, 0.005, 5e-4),
                        (2.0, 0.01, 4e-3), (4.0, 0.08, 0.03)]
    ]
    report = cc.check_controls(
        obs, null_bands={"math": 0.02},
        top_rungs={"control:random_vector": 4.0},
        required_kinds=("control:sham_hook", "control:random_vector"))
    payload = cc.controls_acceptance_payload(obs, report)
    assert set(payload) == {"passed", "hard_stop", "arms"}
    rv = payload["arms"]["rand-vec"]
    assert rv["coeffs"] == [0.5, 1.0, 2.0, 4.0]
    assert rv["tier0_damage"]["kl_coarse_lb"] == [1e-4, 5e-4, 4e-3, 0.03]
    assert rv["acceptance"] == "pass" and rv["role"] == "damage"
    assert payload["arms"]["sham"]["acceptance"] == "pass"
    assert payload["hard_stop"] is False
    # the report's dose-response figure consumes the payload UNCHANGED
    png = rb.dose_response_figure(payload, tmp_path / "dose.png")
    assert png.is_file() and png.stat().st_size > 1000
    # and it round-trips through json (the file on disk)
    json.loads(json.dumps(payload))


# ----------------------------------------------------- unavailability path


# ----------------------------------------------------------- controls-accept


def _ctl_row(arm, kind, coeff, task, domain, item, metric, value, *,
             attempt=0, trial=0, tier=1):
    return cs.ResultRow(
        sweep="ctl", run_id="r", model_id="meta-llama/Llama-3.1-8B-Instruct",
        model_revision="0e9e39f249a16976918f6564b8830bc894c89659",
        arm_name=arm, kind=kind, coeff=coeff,
        intervention_hash="a1b2c3d4e5f60718", engine="hf", tier=tier,
        task=task, task_version="v1", domain=domain, item_id=item,
        trial_id=trial, seed=trial, attempt=attempt, position_order=None,
        metric=metric, value=value, extra_json=None,
        harness_config_hash="0f1e2d3c4b5a6978", env_digest="99aabbccddeeff00",
        slurm_job_id=None, created_at=1755000000.0, wall_clock_s=None)


def _write_cell(root, key, rows):
    store = cs.CellStore(root, key)
    for r in rows:
        store.append_row(r)
    store.write_results({"n": len(rows)})
    return store


def _damage_top_rows(arm, kind, coeff):
    """Rows making a damage family's top rung Tier-0-DETECTABLE: a
    CONFIRMED canary failure (KL magnitudes are annotations, never
    detection evidence) + kl_coarse_lb_corpus as the dose-response
    display series. A stray legacy flag_kl_gt_kappa row rides along to
    pin that stored-campaign rows keep loading and stay ignored."""
    return [_ctl_row(arm, kind, coeff, f"tier0/kl_coarse_{c}", "dist_shift",
                     "_corpus", "kl_coarse_lb_corpus", 0.05, tier=0)
            for c in ("wikitext103", "stack_py")] + [
        _ctl_row(arm, kind, coeff, "tier0/kl_coarse_wikitext103",
                 "dist_shift", "_corpus", "flag_kl_gt_kappa", 1.0, tier=0),
        _ctl_row(arm, kind, coeff, "tier0/canary", "coherence_canary",
                 "_run", "canary/tripped_count", 2.0, tier=0),
        _ctl_row(arm, kind, coeff, "tier0/canary", "coherence_canary",
                 "_run", "flag_canary_confirmed", 1.0, attempt=1, tier=0),
    ]


def test_controls_accept_end_to_end(tmp_path):
    """Synthetic control sweep + calibration reference -> controls-accept
    writes controls_acceptance.json in the frozen shape; every §8 kind
    observed, every damage top rung (pinned from the manifest) detectable
    at Tier 0, the no-ops within the null, exit code 0."""
    from sisteer.capeval.domains import Registry
    from sisteer.capeval.stats import calibrate as cal

    ctl_root = tmp_path / "controls"
    cal_root = tmp_path / "calibration"
    items = [f"i{k}" for k in range(8)]

    # calibration trial 0 (reference leg): knowledge accuracy = all correct
    _write_cell(cal_root, "none__tier01__t0__a0__a1b2c3d4__0f1e2d3c",
                [_ctl_row("none", "none", None, "tinyMMLU", "knowledge", i,
                          "acc", 1.0) for i in items])

    # no-ops: identical accuracy, quiet tier-0
    _write_cell(ctl_root, "zero__tier01-c0__t0__a0__a1b2c3d4__0f1e2d3c",
                [_ctl_row("zero", "control:zero_coeff", 0.0, "tinyMMLU",
                          "knowledge", i, "acc", 1.0) for i in items])
    _write_cell(ctl_root, "sham__tier01-c1__t0__a0__a1b2c3d4__0f1e2d3c",
                [_ctl_row("sham", "control:sham_hook", 1.0, "tinyMMLU",
                          "knowledge", i, "acc", 1.0) for i in items]
                + [_ctl_row("sham", "control:sham_hook", 1.0,
                            "tier0/kl_coarse_wikitext103", "dist_shift",
                            "_corpus", "kl_coarse_lb_corpus", 0.0, tier=0),
                   _ctl_row("sham", "control:sham_hook", 1.0, "tier0/canary",
                            "coherence_canary", "_run", "canary/tripped_count",
                            0.0, tier=0)])
    # random_vector ladder: a TIER-0-ONLY bottom rung first (no accuracy
    # tasks at all while the trial-0 reference carries tinyMMLU — the
    # pair_domain allow_arm_task_subset seam, integrator 2026-08-15);
    # low rung quiet; TOP rung (manifest: 4.0) breaks accuracy + kl
    # flags on 2 corpora
    _write_cell(ctl_root, "rv__tier0-c0p5__t0__a0__a1b2c3d4__0f1e2d3c",
                [_ctl_row("rv", "control:random_vector", 0.5,
                          "tier0/kl_coarse_wikitext103", "dist_shift",
                          "_corpus", "kl_coarse_lb_corpus", 2e-5, tier=0)])
    _write_cell(ctl_root, "rv__tier01-c1__t0__a0__a1b2c3d4__0f1e2d3c",
                [_ctl_row("rv", "control:random_vector", 1.0, "tinyMMLU",
                          "knowledge", i, "acc", 1.0) for i in items]
                + [_ctl_row("rv", "control:random_vector", 1.0,
                            "tier0/kl_coarse_wikitext103", "dist_shift",
                            "_corpus", "kl_coarse_lb_corpus", 1e-4, tier=0)])
    _write_cell(ctl_root, "rv__tier01-c4__t0__a0__a1b2c3d4__0f1e2d3c",
                [_ctl_row("rv", "control:random_vector", 4.0, "tinyMMLU",
                          "knowledge", i, "acc", 0.0) for i in items]
                + _damage_top_rows("rv", "control:random_vector", 4.0))
    # the other damage families' pinned tops (manifest: 3e-3 each) — in
    # the real sweep every top rung also carries Tier-1 rows (noise_lora
    # 3e-3 is a tier1_coeffs rung; weight_noise is a full cell)
    _write_cell(ctl_root, "nl__tier01-c0p003__t0__a0__a1b2c3d4__0f1e2d3c",
                [_ctl_row("nl", "control:noise_lora", 3e-3, "tinyMMLU",
                          "knowledge", i, "acc", 1.0) for i in items]
                + _damage_top_rows("nl", "control:noise_lora", 3e-3))
    _write_cell(ctl_root, "wn__tier01-c0p003__t0__a0__a1b2c3d4__0f1e2d3c",
                [_ctl_row("wn", "control:weight_noise", 3e-3, "tinyMMLU",
                          "knowledge", i, "acc", 1.0) for i in items]
                + _damage_top_rows("wn", "control:weight_noise", 3e-3))

    registry = Registry.load(REPO / "configs" / "capeval" / "domains.yaml")
    hv = cal.harness_version(registry)
    tails = {"p01": -0.02, "p05": -0.01, "p50": 0.0, "p95": 0.01, "p99": 0.02}
    calib = cal.Calibration(
        model_revision="0e9e39f249a16976918f6564b8830bc894c89659",
        engine="hf", harness_version=hv, estimand="conditional_on_reference",
        reference_trial=0, n_contrasts=5, b_boot=100, seed=0,
        created_at=1755000000.0,
        domains={d.name: cal.DomainCalibration(
            domain=d.name, kind=d.kind, sigma_d=0.01, sigma_d_se=0.002,
            tails=dict(tails), n_contrasts=5, n_items=8,
            needs_more_trials=False)
            for d in registry.domains if d.kind in ("accuracy", "nll")},
        canary_thresholds=None, kappa=1e-3)
    cal.save_calibration(calib, cal_root / "calibration.json")

    rc = capeval_run.main([
        "--manifest",
        str(REPO / "configs" / "capeval" / "manifests" / "controls.yaml"),
        "controls-accept", "--out-root", str(ctl_root),
        "--calibration-root", str(cal_root), "--b", "50"])
    assert rc == 0
    payload = json.loads((ctl_root / "controls_acceptance.json").read_text())
    assert set(payload) == {"passed", "hard_stop", "arms"}
    assert payload["passed"] is True and payload["hard_stop"] is False
    sham, rv = payload["arms"]["sham"], payload["arms"]["rv"]
    assert sham == {
        "kind": "control:sham_hook", "role": "noop", "acceptance": "pass",
        "reasons": [], "coeffs": [1.0],
        # exactly 0.0 — the no-op certificate (a nonzero value here
        # now hard-stops; see the checker tests)
        "tier0_damage": {"kl_coarse_lb": [0.0]}}
    assert rv["role"] == "damage" and rv["acceptance"] == "pass"
    assert rv["coeffs"] == [0.5, 1.0, 4.0]
    assert rv["tier0_damage"]["kl_coarse_lb"] == [2e-5, 1e-4, 0.05]
    assert set(payload["arms"]) == {"zero", "sham", "rv", "nl", "wn"}
    # the report consumes it unchanged
    from sisteer.capeval.report import build as rb
    png = rb.dose_response_figure(payload, tmp_path / "dose.png")
    assert png.is_file()


def test_controls_accept_noop_failure_hard_stops(tmp_path):
    """A no-op beyond the null band -> hard_stop, exit code 2, payload
    still written (a failing acceptance is a recorded result)."""
    from sisteer.capeval.domains import Registry
    from sisteer.capeval.stats import calibrate as cal

    ctl_root = tmp_path / "controls"
    cal_root = tmp_path / "calibration"
    items = [f"i{k}" for k in range(8)]
    _write_cell(cal_root, "none__tier01__t0__a0__a1b2c3d4__0f1e2d3c",
                [_ctl_row("none", "none", None, "tinyMMLU", "knowledge", i,
                          "acc", 1.0) for i in items])
    # the FAILING no-op: accuracy collapses under a sham hook
    _write_cell(ctl_root, "sham__tier01-c1__t0__a0__a1b2c3d4__0f1e2d3c",
                [_ctl_row("sham", "control:sham_hook", 1.0, "tinyMMLU",
                          "knowledge", i, "acc", 0.0) for i in items])
    # completeness: every other §8 kind observed (check_controls refuses
    # an incomplete sweep before evaluating anything)
    _write_cell(ctl_root, "zero__tier01-c0__t0__a0__a1b2c3d4__0f1e2d3c",
                [_ctl_row("zero", "control:zero_coeff", 0.0, "tinyMMLU",
                          "knowledge", i, "acc", 1.0) for i in items])
    _write_cell(ctl_root, "rv__tier01-c4__t0__a0__a1b2c3d4__0f1e2d3c",
                [_ctl_row("rv", "control:random_vector", 4.0, "tinyMMLU",
                          "knowledge", i, "acc", 1.0) for i in items]
                + _damage_top_rows("rv", "control:random_vector", 4.0))
    _write_cell(ctl_root, "nl__tier01-c0p003__t0__a0__a1b2c3d4__0f1e2d3c",
                [_ctl_row("nl", "control:noise_lora", 3e-3, "tinyMMLU",
                          "knowledge", i, "acc", 1.0) for i in items]
                + _damage_top_rows("nl", "control:noise_lora", 3e-3))
    _write_cell(ctl_root, "wn__tier01-c0p003__t0__a0__a1b2c3d4__0f1e2d3c",
                [_ctl_row("wn", "control:weight_noise", 3e-3, "tinyMMLU",
                          "knowledge", i, "acc", 1.0) for i in items]
                + _damage_top_rows("wn", "control:weight_noise", 3e-3))

    registry = Registry.load(REPO / "configs" / "capeval" / "domains.yaml")
    calib = cal.Calibration(
        model_revision="0e9e39f249a16976918f6564b8830bc894c89659",
        engine="hf", harness_version=cal.harness_version(registry),
        estimand="conditional_on_reference", reference_trial=0,
        n_contrasts=5, b_boot=100, seed=0, created_at=1755000000.0,
        domains={d.name: cal.DomainCalibration(
            domain=d.name, kind=d.kind, sigma_d=0.01, sigma_d_se=0.002,
            tails={"p01": -0.02, "p05": -0.01, "p50": 0.0, "p95": 0.01,
                   "p99": 0.02},
            n_contrasts=5, n_items=8, needs_more_trials=False)
            for d in registry.domains if d.kind in ("accuracy", "nll")},
        canary_thresholds=None, kappa=1e-3)
    cal.save_calibration(calib, cal_root / "calibration.json")

    rc = capeval_run.main([
        "--manifest",
        str(REPO / "configs" / "capeval" / "manifests" / "controls.yaml"),
        "controls-accept", "--out-root", str(ctl_root),
        "--calibration-root", str(cal_root), "--b", "50"])
    assert rc == 2
    payload = json.loads((ctl_root / "controls_acceptance.json").read_text())
    assert payload["hard_stop"] is True
    assert payload["arms"]["sham"]["acceptance"] == "fail"
