"""Jailbreak Antidote arm (arXiv 2410.02298): PCA/sign/mask unit tests
on synthetic clusters, artifact save/load round trip, base_alpha
calibration logic (midpoint, bounds, loud failures), and the arm wiring
(registry position, jailbreak defense axis, configs, sbatch index decode).

Ported from a development branch for the two-SI bring-in, with the
wiring assertions adapted to THIS branch's registry: antidote is appended at
index 17 in methods.ARMS (after the 17 pre-existing arms), followed by
alphasteer and lofit. The S2-matrix wiring tests (stage2_common.sh S2_ARMS
index pins, stage2_llama8b.yaml symmetry) arrived with the stage2-evals
merge, whose config wiring pins antidote at S2_ARMS[14] — a positional enum independent of the
methods.ARMS registry index. The jailbreak defense axis (COMPILED /
DEFENSES / configs/stage2_jailbreak.yaml) order is asserted against the
live registry in test_jailbreak_run.py.
"""

import re
from pathlib import Path

import pytest
import torch
import yaml

from sisteer.extraction.antidote import (
    DEFAULT_ALPHA_LADDER,
    AntidoteObject,
    antidote_direction,
    calibrate_base_alpha,
    sparsify,
    train_antidote,
)
from sisteer.methods import (
    ARMS,
    VECTOR_SAFETENSOR_ARMS,
    load_object,
    object_artifact_name,
)
from sisteer.models import hidden_size
from sisteer.vectors import SteeringVector

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- direction


def _clusters(d=64, n=200, sep=8.0, noise=0.1, seed=0):
    """Two Gaussian clusters separated along a known unit axis."""
    g = torch.Generator().manual_seed(seed)
    axis = torch.zeros(d)
    axis[3] = 1.0
    base = torch.randn(n, d, generator=g) * noise
    target = base + sep / 2 * axis
    other = torch.randn(n, d, generator=g) * noise - sep / 2 * axis
    return target, other, axis


def test_pca_direction_recovers_synthetic_axis_signed_toward_target():
    target, other, axis = _clusters()
    pc1, info = antidote_direction(target, other)
    assert abs(float(pc1.norm()) - 1.0) < 1e-5  # unit norm by construction
    # the separation axis dominates the pooled covariance, and the sign
    # convention points +alpha toward the TARGET cluster
    assert float(pc1 @ axis) > 0.99
    assert info["explained_variance_ratio_pc1"] > 0.9
    # sign_cos is the PRE-flip evidence (sign_flipped says whether the raw
    # PC pointed away); its magnitude is what certifies the sign choice
    assert abs(info["sign_cos"]) > 0.9
    assert info["warnings"] == []


def test_pca_direction_sign_flips_with_swapped_clusters():
    target, other, axis = _clusters()
    pc1_fwd, info_fwd = antidote_direction(target, other)
    pc1_rev, info_rev = antidote_direction(other, target)
    # same principal subspace, opposite sign: +alpha must move toward
    # whichever cluster is TARGET
    assert float(pc1_fwd @ pc1_rev) < -0.99
    assert info_fwd["sign_flipped"] != info_rev["sign_flipped"]


def test_pca_direction_warns_on_weak_pc1_and_null_sign():
    # isotropic noise, no separation: PC1 explains ~1/d of the variance and
    # the mean difference barely projects onto it — both warnings must be
    # recorded in the metadata trail (loud, never silent)
    g = torch.Generator().manual_seed(1)
    target = torch.randn(300, 128, generator=g)
    other = torch.randn(300, 128, generator=g)
    _, info = antidote_direction(target, other)
    joined = " ".join(info["warnings"])
    assert "PC1 explains only" in joined
    assert "sign projection is near zero" in joined


def test_pca_direction_rejects_too_few_examples():
    with pytest.raises(ValueError, match=">= 2 pooled examples"):
        antidote_direction(torch.randn(1, 8), torch.randn(0, 8))


# ---------------------------------------------------------------- sparsify


def test_sparsify_keeps_top_k_by_abs_without_renormalizing():
    d = 100
    direction = torch.linspace(-1.0, 1.0, d)
    direction = direction / direction.norm()  # unit, spread magnitudes
    masked, info = sparsify(direction, k_frac=0.05)
    assert info["k_dims"] == 5  # int(0.05 * 100), panda-guard's floor
    nz = masked.nonzero().flatten()
    assert len(nz) == 5
    # kept dims are exactly the 5 largest |components|
    top = torch.topk(direction.abs(), 5).indices
    assert set(nz.tolist()) == set(top.tolist())
    # NOT renormalized: the masked norm is the kept mass, < 1
    expected = float(direction[top].norm())
    assert abs(info["masked_norm"] - expected) < 1e-6
    assert float(masked.norm()) < 0.999
    # untouched surviving components: same values, not rescaled
    assert torch.equal(masked[top], direction[top])


def test_sparsify_validates_k_frac():
    v = torch.randn(16)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="k_frac"):
            sparsify(v, bad)
    # k_frac tiny -> still keeps at least one dim
    masked, info = sparsify(v, 1e-6)
    assert info["k_dims"] == 1


# ---------------------------------------------------------------- calibration


def _patch_calib(monkeypatch, distinct_values):
    """Route generation + metrics through stubs: distinct-3 returns the
    given per-rung values in order; generation and perplexity are inert."""
    calls = {"n": 0}

    def fake_generate(model, tok, prompts, **kw):
        return ["stub text one", "stub text two"][: len(prompts)]

    def fake_distinct(texts, n=3):
        v = distinct_values[calls["n"]]
        calls["n"] += 1
        return v

    def fake_ppl(model, tok, texts=None, **kw):
        return 10.0

    import sisteer.eval.capability as cap
    import sisteer.generate as gen

    monkeypatch.setattr(gen, "generate_batch", fake_generate)
    monkeypatch.setattr(cap, "distinct_ngram_ratio", fake_distinct)
    monkeypatch.setattr(cap, "perplexity", fake_ppl)


def test_calibration_stops_at_first_degenerate_and_takes_midpoint(
    tiny_llama, monkeypatch
):
    model, tok = tiny_llama
    d = hidden_size(model)
    # rungs: 0.5 coherent, 1.0 coherent, 2.0 DEGENERATE -> coherent range
    # [0, 1.0], base_alpha = 0.5, and the scan never reaches 4.0
    _patch_calib(monkeypatch, distinct_values=[0.9, 0.8, 0.1])
    calib = calibrate_base_alpha(
        model, tok, ["p1", "p2"], torch.randn(d), 1,
        alpha_ladder=[0.5, 1.0, 2.0, 4.0], rep_threshold=0.35,
    )
    assert calib["base_alpha"] == pytest.approx(0.5)
    assert calib["alpha_bounds"] == [0.0, 1.0]
    assert [r["alpha"] for r in calib["sweep"]] == [0.5, 1.0, 2.0]
    assert calib["sweep"][-1]["degenerate"] is True
    assert calib["warnings"] == []  # the bound WAS bracketed


def test_calibration_warns_when_bound_not_bracketed(tiny_llama, monkeypatch):
    model, tok = tiny_llama
    d = hidden_size(model)
    _patch_calib(monkeypatch, distinct_values=[0.9, 0.9])
    calib = calibrate_base_alpha(
        model, tok, ["p1"], torch.randn(d), 1, alpha_ladder=[1.0, 2.0],
    )
    assert calib["base_alpha"] == pytest.approx(1.0)
    assert calib["alpha_bounds"] == [0.0, 2.0]
    assert any("not bracketed" in w for w in calib["warnings"])


def test_calibration_fails_loudly_when_no_coherent_alpha(
    tiny_llama, monkeypatch
):
    model, tok = tiny_llama
    d = hidden_size(model)
    _patch_calib(monkeypatch, distinct_values=[0.1])
    with pytest.raises(RuntimeError, match="NO coherent alpha"):
        calibrate_base_alpha(
            model, tok, ["p1"], torch.randn(d), 1, alpha_ladder=[0.5, 1.0],
        )


def test_calibration_rejects_bad_ladder(tiny_llama):
    model, tok = tiny_llama
    with pytest.raises(ValueError, match="alpha_ladder"):
        calibrate_base_alpha(model, tok, ["p"], torch.randn(4), 1,
                             alpha_ladder=[])
    with pytest.raises(ValueError, match="alpha_ladder"):
        calibrate_base_alpha(model, tok, ["p"], torch.randn(4), 1,
                             alpha_ladder=[-1.0, 2.0])


# ---------------------------------------------------------------- end to end


def _tiny_pairs():
    from sisteer.data.pairs import Pair

    prompts = ["say hello", "count to three", "name a color", "what is two"]
    return [Pair(prompt=p, pos="ok", neg="no", instruction_id="french")
            for p in prompts]


def _train_tiny(model, tok):
    from sisteer.data.instructions import BY_ID

    return train_antidote(
        model, tok, BY_ID["french"], _tiny_pairs(), layer=1,
        k_frac=0.05, alpha_ladder=[0.5, 1.0], calib_prompts=2,
        calib_max_new_tokens=4, rep_threshold=0.0, ppl_guard_factor=1e9,
        batch_size=2, epochs=1, max_len=64,
    )


def test_train_antidote_end_to_end_metadata_and_apply(tiny_llama):
    model, tok = tiny_llama
    d = hidden_size(model)
    obj = _train_tiny(model, tok)
    assert obj.kind == "antidote"
    sv = obj.sv
    assert sv.method == "antidote"
    assert sv.layers == [1]
    vec = sv.vectors[1]
    meta = sv.metadata
    assert meta["k_frac"] == 0.05
    assert meta["k_dims"] == max(1, int(0.05 * d))
    assert int((vec != 0).sum()) == meta["k_dims"]
    assert float(vec.norm()) <= 1.0 + 1e-5  # masked unit PC, no renorm
    # sign evidence + PC1 strength recorded (the WARN-loudly audit trail)
    for key in ("explained_variance_ratio_pc1", "sign_projection",
                "sign_cos", "sign_flipped", "warnings"):
        assert key in meta
    # calibration: ladder [0.5, 1.0] all coherent -> hi 1.0, base 0.5
    assert meta["base_alpha"] == pytest.approx(0.5)
    assert meta["alpha_bounds"] == [0.0, 1.0]
    assert meta["positions"] == "last"
    assert meta["contrast"] == "si_prompted_vs_unprompted_prompt_last_token"
    assert obj.base_alpha == pytest.approx(0.5)
    # applies through the shared protocol
    enc = tok(obj.inject("the quick brown fox"), return_tensors="pt")
    with torch.no_grad(), obj.apply(model, scale=1.0):
        model(**enc)


def test_antidote_roundtrip_and_load_object_dispatch(tiny_llama, tmp_path):
    model, tok = tiny_llama
    obj = _train_tiny(model, tok)
    path = tmp_path / "object.safetensors"
    obj.save(path)
    loaded = load_object(path, model, tok)
    assert isinstance(loaded, AntidoteObject)  # stored method, not suffix
    assert loaded.base_alpha == pytest.approx(obj.base_alpha)
    assert loaded.sv.metadata["alpha_bounds"] == obj.sv.metadata["alpha_bounds"]
    enc = tok("the quick brown fox", return_tensors="pt")
    with torch.no_grad():
        base = model(**enc).logits
        with obj.apply(model, scale=2.0):
            orig = model(**enc).logits
        with loaded.apply(model, scale=2.0):
            rehydrated = model(**enc).logits
    assert torch.allclose(orig, rehydrated, atol=1e-6)
    # the re-hydrated object still honors positions="last": every
    # non-final position is bit-identical to the unsteered forward
    assert torch.equal(base[:, :-1], rehydrated[:, :-1])
    assert not torch.allclose(base[:, -1], rehydrated[:, -1])


def test_load_object_antidote_without_base_alpha_fails_loudly(
    tiny_llama, tmp_path
):
    model, tok = tiny_llama
    d = hidden_size(model)
    sv = SteeringVector(vectors={1: torch.randn(d)}, method="antidote",
                        model_name="m", instruction_id="french",
                        metadata={})  # uncalibrated: no base_alpha
    path = tmp_path / "object.safetensors"
    sv.save(path)
    with pytest.raises(ValueError, match="base_alpha"):
        load_object(path, model, tok)


def test_antidote_object_rejects_wrong_method():
    sv = SteeringVector(vectors={1: torch.randn(8)}, method="distill_soft",
                        model_name="m", instruction_id="i",
                        metadata={"base_alpha": 1.0})
    with pytest.raises(ValueError, match="method='antidote'"):
        AntidoteObject(sv)


def test_train_antidote_rejects_empty_pairs(tiny_llama):
    from sisteer.data.instructions import BY_ID

    model, tok = tiny_llama
    with pytest.raises(ValueError, match="empty pairs"):
        train_antidote(model, tok, BY_ID["french"], [], layer=1)


# ---------------------------------------------------------------- wiring


def test_antidote_arm_wiring():
    """Registry membership + artifact naming on THIS branch: antidote is
    APPENDED after the pre-existing arms, so no positional decode shifts
 — index 18 since the 2026-08-28 lora_r4 alias joined at index
    4 next to its lora_r1_all sibling (the one audited non-append
    insertion, see test_methods_s2's order test) — with this lane's full
    core suite set and the SteeringVector safetensors container."""
    assert list(ARMS)[18] == "antidote"
    assert ARMS["antidote"].suites == frozenset({"ifeval"})
    assert "antidote" in VECTOR_SAFETENSOR_ARMS
    assert object_artifact_name("antidote") == "object.safetensors"
    # the jailbreak defense axis picks up the brought-in arms in registry
    # order (the loader resolves the artifact name through
    # methods.object_artifact_name — jailbreak_run's _defense_object_path —
    # so the container choice above is load-bearing); the alphakl rungs
    # were appended after the defense trio, then the diffmean
    # floor after those (cast / circuit_breaker briefly sat between until
    # their 2026-08-20 registry removal), then the stage2-evals and cheap-estimator
    # arms at the tail — so antidote sits at 18-20 and
    # neither it nor the alphakl rungs are last.
    from sisteer.eval.jailbreak import COMPILED, DEFENSES

    assert COMPILED[18:21] == ("antidote", "alphasteer", "lofit")
    assert COMPILED[21:24] == ("alphakl_null", "alphakl_soft", "alphakl_fisher")
    # the 2026-08-20 floor, then the 2026-08-21 bidpo_lora cell, then the
    # 2026-08-21 caa_static_r Class B floor, then the stage2-evals
    # did/adasteer and cheap-estimator caa/actadd/actadd_front arms — appended last
    # in sequence (indices +1 since the lora_r4 slot at 4), so the wired
    # arms' relative order is unmoved
    assert COMPILED[24:27] == ("diffmean", "bidpo_lora", "caa_static_r")
    # positional, no longer the tail: the sdf-base-steering CE arms
    # (psr_ce_all/lora_r4_ce, 2026-09-01) are appended after these.
    assert COMPILED[27:32] == ("did", "adasteer", "caa", "actadd",
                               "actadd_front")
    # the tail in RELATIVE order (a literal tail slice moves at every
    # append; appending is the rule) — did/adasteer, the cheap-estimator arms, the
    # transferred psr_xfer_* arms (merged 2026-09-02), the sdf-base DERIVED
    # LoRA arm (2026-09-10), then the UNDISTILLED paired write as the FINAL
    # registered arm (relu-differenced-native-qwen, 2026-09-17)
    _tail = ("did", "adasteer", "caa", "actadd", "actadd_front",
             "psr_xfer_from_e2b", "psr_xfer_from_e2b_rw1",
             "psr_xfer_from_e4b", "psr_xfer_from_e4b_rw1",
             "lora_r4_ce_minus_ctl", "psr_ce_all_minus_ctl")
    _idx = [COMPILED.index(a) for a in _tail]
    assert _idx == sorted(_idx) and len(set(_idx)) == len(_idx)
    assert COMPILED[-2:] == ("lora_r4_ce_minus_ctl", "psr_ce_all_minus_ctl")
    assert DEFENSES[20:23] == ("antidote", "alphasteer", "lofit")
    assert DEFENSES[:2] == ("none", "prompted")
    # the driver's artifact-name set mirrors methods (loader agreement)
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import stage2_run as s2run
    finally:
        sys.path.pop(0)
    assert "antidote" in s2run.VECTOR_SAFETENSOR_ARMS


def test_antidote_arm_position_and_config_symmetry():
    """The antidote arm is pinned at S2_ARMS[14] — existing decodes never
    shift (the appended-index rule) — and the config mirrors it."""
    sh = (REPO / "scripts" / "sbatch" / "stage2_common.sh").read_text()

    def bash_array(name):
        m = re.search(rf"{name}=\(([^)]*)\)", sh)
        assert m, name
        return m.group(1).split()

    arms = bash_array("S2_ARMS")
    # antidote pinned at 14; the arms after it mirror the methods.ARMS
    # registry order since the 2026-08-19 reorder — alphasteer, lofit, the
    # alphakl ladder, then did/adasteer at the tail (len 15→18→19→22,
    # unchanged by the reorder)
    assert arms[14] == "antidote" and len(arms) == 22
    cfg = yaml.safe_load(
        (REPO / "configs" / "stage2_llama8b.yaml").read_text())
    assert cfg["arms"][14] == "antidote"
    assert cfg["arms"] == arms
    tk = cfg["train_kwargs"]["antidote"]
    assert tk["k_frac"] == 0.05  # the paper default, pinned in provenance


def test_default_ladder_is_positive_ascending():
    assert list(DEFAULT_ALPHA_LADDER) == sorted(DEFAULT_ALPHA_LADDER)
    assert DEFAULT_ALPHA_LADDER[0] > 0
