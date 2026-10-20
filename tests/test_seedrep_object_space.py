"""Object-space seed stability on SYNTHETIC blobs: identical objects → cosine 1.0,
sign-flipped → −1.0, orthogonal → 0; rank-4 LoRA and 32-layer PSR shapes; the
subtraction blob composes to the explicit difference; an end-to-end run over a
synthetic bank writes tables, verdicts and figures; an incomplete triplet is
`unavailable` with seed stability FAILED and no survivor statistic.

Three later contracts (interim_analysis.md §9 item 9; O23):
* FLOAT64 statistics — a pair of ΔW FACTORS whose exact cosine is 1/2 (and exact relnorm
  2/√3, exact norm α√n) over 10^7 elements: float32 ``dot``/``vector_norm`` (the old path)
  drift by ~1e-3, the module's float64 accumulation lands within 1e-9; a cosine past
  1 + 1e-9 is refused, never clamped;
* the CLASS ROSTER is derived from the roster's replicate views + the RS admission files —
  a class the admission refused on every seed reads ``unavailable: incomplete replicate
  triplet (0/3 admitted; missing: …)`` and nothing refuses;
* the batch-8 seed-0 reference column is OPT-IN: by default no ``base__*`` / ``derived__*``
  object is loaded and no vs-original pair exists; ``--include-batch8-reference`` restores it.
"""
from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
SEEDREP = REPO / "scripts" / "sdfbase" / "seedrep"
if str(SEEDREP) not in sys.path:
    sys.path.insert(0, str(SEEDREP))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SEEDREP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


os_ = _load("object_space")
tg = sys.modules["triplet_gate"]

LAYERS_LORA = [3, 7, 11, 15]
D_IN, D_OUT, RANK = 24, 16, 4
LAYERS_PSR = list(range(32))
D_PSR = 12
DISTILLED = "constv2-mix-full__psr_ce_all_minus_ctl_distilled"
LORA_DIFF = "constv2-mix-full__lora_r4_ce_minus_ctl"


# ---------------------------------------------------------------- synthetic blob builders (the real schemas)
def lora_blob(seed: int, layers=LAYERS_LORA, rank=RANK, scale=1.0, A_override=None) -> dict:
    g = torch.Generator().manual_seed(seed)
    A = {str(l): (A_override[l] if A_override else torch.randn(D_IN, rank, generator=g) / D_IN ** 0.5) for l in layers}
    B = {str(l): scale * torch.randn(D_OUT, rank, generator=g) for l in layers}
    return {"kind": "lora_all", "layers": list(layers), "rank": rank, "A": A, "B": B, "positions": "all",
            "response_header_len": None}


def subtraction_blob(pos: dict, neg: dict) -> dict:
    return {"kind": "lora_subtraction", "layers": list(pos["layers"]), "rank": pos["rank"],
            "A_pos": {k: v.clone() for k, v in pos["A"].items()}, "B_pos": {k: v.clone() for k, v in pos["B"].items()},
            "A_neg": {k: v.clone() for k, v in neg["A"].items()}, "B_neg": {k: v.clone() for k, v in neg["B"].items()},
            "provenance": {"kind": "lora_subtraction", "derived_id": "synthetic"}, "positions": "all",
            "response_header_len": None}


def psr_blob(seed: int, layers=LAYERS_PSR, d=D_PSR) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {"kind": "psr", "layers": list(layers), "train_direction": True, "positions": "all", "response_header_len": None,
            "directions": {str(l): torch.randn(d, generator=g) for l in layers},
            "gate_w": {str(l): 0.1 * torch.randn(d, generator=g) for l in layers},
            "gate_b": {str(l): torch.ones(1) + 0.01 * torch.randn(1, generator=g) for l in layers}}


def flipped(blob: dict) -> dict:
    out = json.loads(json.dumps({k: v for k, v in blob.items() if not isinstance(v, dict) or k in ("provenance",)}))
    for key in ("A", "B", "A_pos", "B_pos", "A_neg", "B_neg", "directions", "gate_w", "gate_b"):
        if key in blob:
            out[key] = {k: (-v if key in ("B", "B_pos", "B_neg", "directions", "gate_w") else v.clone())
                        for k, v in blob[key].items()}
    return out


# ---------------------------------------------------------------- vector extraction + pair statistics
def test_lora_delta_is_B_A_transpose_and_shapes_match_rank4():
    b = lora_blob(1)
    v = os_.vectors_from_blob(b)
    assert v.kind == "lora_all" and v.layers == LAYERS_LORA and list(v.comps) == ["dW"]
    for l in LAYERS_LORA:
        expect = b["B"][str(l)] @ b["A"][str(l)].T
        assert expect.shape == (D_OUT, D_IN)
        assert torch.allclose(v.comps["dW"][l], expect.reshape(-1))
        assert v.comps["dW"][l].dtype == torch.float32
    assert v.stacked("dW").numel() == len(LAYERS_LORA) * D_OUT * D_IN


def test_rank1_legacy_vectors_compose_as_outer_product():
    A, B = torch.randn(D_IN), torch.randn(D_OUT)
    assert torch.allclose(os_.lora_delta(A, B), torch.outer(B, A))


def test_psr_shapes_32_layers():
    v = os_.vectors_from_blob(psr_blob(1))
    assert v.kind == "psr" and v.layers == LAYERS_PSR
    assert set(v.comps) == {"direction", "gate_w", "gate_b"}
    assert v.comps["direction"][0].shape == (D_PSR,) and v.comps["gate_b"][31].shape == (1,)
    assert v.stacked("direction").numel() == 32 * D_PSR


def test_identical_objects_cosine_one_relnorm_zero():
    a = os_.vectors_from_blob(lora_blob(1))
    b = os_.vectors_from_blob(lora_blob(1))
    st = os_.pair_stats(a, b, "lora")["components"]["dW"]
    assert all(abs(c - 1.0) < 1e-6 for c in st["per_layer_cosine"].values())
    assert abs(st["stacked_cosine"] - 1.0) < 1e-6
    assert all(abs(r) < 1e-6 for r in st["per_layer_relnorm"].values()) and abs(st["stacked_relnorm"]) < 1e-6
    p = os_.vectors_from_blob(psr_blob(2))
    stp = os_.pair_stats(p, os_.vectors_from_blob(psr_blob(2)), "psr")
    assert abs(stp["components"]["direction"]["stacked_cosine"] - 1.0) < 1e-6
    assert abs(stp["components"]["gate_w"]["stacked_cosine"] - 1.0) < 1e-6
    assert all(d == 0.0 for d in stp["gate_b_abs_diff"].values())


def test_sign_flipped_objects_cosine_minus_one_relnorm_two():
    a = os_.vectors_from_blob(lora_blob(3))
    b = os_.vectors_from_blob(flipped(lora_blob(3)))
    st = os_.pair_stats(a, b, "lora")["components"]["dW"]
    assert all(abs(c + 1.0) < 1e-6 for c in st["per_layer_cosine"].values())
    assert abs(st["stacked_cosine"] + 1.0) < 1e-6
    assert all(abs(r - 2.0) < 1e-5 for r in st["per_layer_relnorm"].values())
    p = os_.vectors_from_blob(psr_blob(4))
    q = os_.vectors_from_blob(flipped(psr_blob(4)))
    stp = os_.pair_stats(p, q, "psr")
    assert abs(stp["components"]["direction"]["stacked_cosine"] + 1.0) < 1e-6
    assert abs(stp["components"]["gate_w"]["stacked_cosine"] + 1.0) < 1e-6


def test_orthogonal_objects_cosine_zero():
    # rank-1 factors on disjoint coordinates: ΔW_a = e0 ⊗ f0, ΔW_b = e1 ⊗ f1 → vec() orthogonal
    layers = [3]
    A_a, A_b = torch.zeros(D_IN, 1), torch.zeros(D_IN, 1)
    A_a[0, 0], A_b[1, 0] = 1.0, 1.0
    B_a, B_b = torch.zeros(D_OUT, 1), torch.zeros(D_OUT, 1)
    B_a[0, 0], B_b[1, 0] = 1.0, 1.0
    a = {"kind": "lora_all", "layers": layers, "rank": 1, "A": {"3": A_a}, "B": {"3": B_a}}
    b = {"kind": "lora_all", "layers": layers, "rank": 1, "A": {"3": A_b}, "B": {"3": B_b}}
    st = os_.pair_stats(os_.vectors_from_blob(a), os_.vectors_from_blob(b), "lora")["components"]["dW"]
    assert abs(st["per_layer_cosine"][3]) < 1e-7 and abs(st["stacked_cosine"]) < 1e-7
    # relnorm of two orthogonal unit vectors is √2
    assert abs(st["stacked_relnorm"] - 2 ** 0.5) < 1e-6
    # PSR: orthogonal directions
    p = psr_blob(5, layers=[0, 1], d=4)
    q = psr_blob(6, layers=[0, 1], d=4)
    for l in ("0", "1"):
        p["directions"][l] = torch.tensor([1.0, 0.0, 0.0, 0.0]); q["directions"][l] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    stp = os_.pair_stats(os_.vectors_from_blob(p), os_.vectors_from_blob(q), "psr")["components"]["direction"]
    assert abs(stp["stacked_cosine"]) < 1e-7


def test_subtraction_blob_equals_explicit_difference_of_parents():
    pos, neg = lora_blob(7), lora_blob(8)
    sub = os_.vectors_from_blob(subtraction_blob(pos, neg))
    vp, vn = os_.vectors_from_blob(pos), os_.vectors_from_blob(neg)
    assert sub.kind == "lora_subtraction" and sub.layers == LAYERS_LORA
    for l in LAYERS_LORA:
        assert torch.allclose(sub.comps["dW"][l], vp.comps["dW"][l] - vn.comps["dW"][l], atol=1e-6)
    # and the composition is NOT the difference of the factors: B_pos A_posᵀ − B_neg A_negᵀ ≠ (B_pos−B_neg)(A_pos−A_neg)ᵀ
    l = LAYERS_LORA[0]
    naive = (pos["B"][str(l)] - neg["B"][str(l)]) @ (pos["A"][str(l)] - neg["A"][str(l)]).T
    assert not torch.allclose(sub.comps["dW"][l], naive.reshape(-1), atol=1e-3)


def test_layer_mismatch_and_unknown_kind_refuse():
    with pytest.raises(ValueError):
        os_.pair_stats(os_.vectors_from_blob(lora_blob(1)), os_.vectors_from_blob(lora_blob(1, layers=[3, 7])), "lora")
    with pytest.raises(ValueError):
        os_.vectors_from_blob({"kind": "vector_all"})
    bad = lora_blob(1)
    bad["layers"] = [3, 7]
    with pytest.raises(ValueError):
        os_.vectors_from_blob(bad)


# ---------------------------------------------------------------- float64 accumulation (interim_analysis §9 item 9)
N_BIG_OUT, N_BIG_IN = 2500, 4000            # 10^7-element ΔW per layer
ALPHA32 = float(torch.tensor(1e-3, dtype=torch.float32))     # the float32 value every element carries


def _big_factor_pair():
    """Rank-1 ΔW FACTORS with a known exact geometry: ``ΔW_a = α·1·1ᵀ`` (every element α) and ``ΔW_b = α·e_S·1ᵀ``
    with ``e_S`` the indicator of the first quarter of the rows — so vec(ΔW_b) equals vec(ΔW_a) on the first n/4
    elements and is 0 elsewhere. Exactly: cos = √(1/4) = 1/2, ‖a‖ = α√n, ‖a − b‖ = α√(3n/4), relnorm = 2/√3.
    Multiplying by 1.0 is exact, so the float32 ΔW elements are exactly α or 0 and the closed forms hold for the
    float32 vectors themselves."""
    A = torch.ones(N_BIG_IN, 1)
    B_a = torch.full((N_BIG_OUT, 1), ALPHA32)
    B_b = torch.zeros(N_BIG_OUT, 1)
    B_b[: N_BIG_OUT // 4, 0] = ALPHA32
    a = os_.vectors_from_blob({"kind": "lora_all", "layers": [0], "rank": 1, "A": {"0": A}, "B": {"0": B_a}})
    b = os_.vectors_from_blob({"kind": "lora_all", "layers": [0], "rank": 1, "A": {"0": A}, "B": {"0": B_b}})
    return a, b


def _old_float32_cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    """The path the module took before the fix: float32 dot / float32 norms."""
    x, y = x.reshape(-1).to(torch.float32), y.reshape(-1).to(torch.float32)
    return float(torch.dot(x, y) / (torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)))


def test_float64_cosine_and_relnorm_exact_where_float32_drifts():
    a, b = _big_factor_pair()
    x, y = a.comps["dW"][0], b.comps["dW"][0]
    assert x.numel() == 10_000_000 and x.dtype == torch.float32
    # the old float32 accumulation is visibly wrong on this pair (measured ~5e-4 low; the real bank read +0.01..0.02)
    assert abs(_old_float32_cosine(x, y) - 0.5) > 1e-5
    assert abs(float(torch.linalg.vector_norm(x - y)) - ALPHA32 * math.sqrt(0.75 * x.numel())) > 1e-5 * ALPHA32 * math.sqrt(x.numel())
    # the module's float64 accumulation lands on the exact values
    assert abs(os_.cosine(x, y) - 0.5) < 1e-9
    assert abs(os_.relnorm(x, y) - 2 / math.sqrt(3)) < 1e-9
    # pair_stats (per layer AND stacked, the sum-of-layer-sums path) agrees with the direct call
    st = os_.pair_stats(a, b, "lora")["components"]["dW"]
    assert abs(st["per_layer_cosine"][0] - 0.5) < 1e-9 and abs(st["stacked_cosine"] - 0.5) < 1e-9
    assert abs(st["per_layer_relnorm"][0] - 2 / math.sqrt(3)) < 1e-9 and abs(st["stacked_relnorm"] - 2 / math.sqrt(3)) < 1e-9
    assert abs(st["stacked_cosine"] - os_.cosine(a.stacked("dW"), b.stacked("dW"))) < 1e-9
    # chunking is an implementation detail: any chunk size gives the same answer
    for chunk in (1 << 10, 1 << 20, 10_000_000, 1 << 30):
        s = os_._f64_sums(x, y, chunk=chunk)
        assert abs(os_.bounded_cosine(s.ab, s.aa, s.bb) - 0.5) < 1e-9


def test_float64_norm_exact_where_float32_drifts(tmp_path):
    a, _ = _big_factor_pair()
    x = a.comps["dW"][0]
    exact = ALPHA32 * math.sqrt(x.numel())
    # float32 vector_norm drifts (measured ~1e-3 relative on this build; ~1.2 % on the real 84 M-element vectors)
    assert abs(float(torch.linalg.vector_norm(x)) - exact) / exact > 1e-5
    got = os_.norms(a)["dW"][0]
    assert abs(got - exact) / exact < 1e-9
    assert abs(os_.vector_norm(x) - exact) / exact < 1e-9
    # the per-layer norm table written beside every object (`_describe`) is the float64 one
    d = tmp_path / "cell"
    d.mkdir()
    torch.save({"kind": "lora_all", "layers": [0], "rank": 1}, d / "object.pt")
    desc = os_._describe(d, a)
    assert abs(desc["per_layer_norms"]["dW"]["0"] - exact) / exact < 1e-9 and desc["n_layers"] == 1


def test_cosine_beyond_one_is_refused_never_clamped():
    assert os_.bounded_cosine(1.0, 1.0, 1.0) == 1.0
    assert abs(os_.bounded_cosine(1.0 + 1e-12, 1.0, 1.0) - 1.0) < 1e-11       # float64 rounding is admitted …
    with pytest.raises(ValueError, match="exceeds 1"):
        os_.bounded_cosine(1.0 + 1e-6, 1.0, 1.0)                              # … an impossible value is not
    with pytest.raises(ValueError, match="exceeds 1"):
        os_.bounded_cosine(-(1.0 + 1e-6), 1.0, 1.0)
    assert math.isnan(os_.bounded_cosine(0.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        os_._f64_sums(torch.ones(4), torch.ones(5))


# ---------------------------------------------------------------- end to end on a synthetic bank
def _write_cell(bank: Path, staged_id: str, blob: dict, admitted: bool | None = None) -> Path:
    d = bank / staged_id
    d.mkdir(parents=True)
    torch.save(blob, d / "object.pt")
    if admitted is not None:
        (d / "ADMISSION.json").write_text(json.dumps({"admitted": admitted, "reasons": ["synthetic reason"]}))
    return d


def _make_bank(bank: Path, *, drop_seed: int | None = None, not_admitted_seed: int | None = None,
               skip_classes: tuple[str, ...] = (), skip_seeds: dict[str, tuple[int, ...]] | None = None,
               write_original: bool = True) -> None:
    """Six classes × (s1, s2, s3, orig). Parents: seed-perturbed LoRA/PSR fits; the LoRA
    difference is the true subtraction of the same-seed parents; the PSR difference is a
    fresh PSR blob (a distilled object is a PSR blob). ``skip_classes`` writes no replicate of those
    classes (the admission refused them: never staged); ``skip_seeds`` drops named seeds of a class."""
    for s in (1, 2, 3, 0):
        # correlated replicates: a shared component + a per-seed perturbation
        base_A = {l: torch.randn(D_IN, RANK, generator=torch.Generator().manual_seed(100 + l)) / D_IN ** 0.5 for l in LAYERS_LORA}
        const = lora_blob(10 + s, A_override=base_A)
        ctl = lora_blob(20 + s, A_override=base_A, scale=0.5)
        for l in LAYERS_LORA:      # make the seeds correlated on B (the trained factor)
            shared = torch.randn(D_OUT, RANK, generator=torch.Generator().manual_seed(300 + l))
            const["B"][str(l)] = shared + 0.3 * const["B"][str(l)]
            ctl["B"][str(l)] = 0.5 * shared + 0.3 * ctl["B"][str(l)]
        diff = subtraction_blob(const, ctl)
        pconst, pctl, pdiff = psr_blob(30 + s), psr_blob(40 + s), psr_blob(50 + s)
        for l in LAYERS_PSR:
            shared = torch.randn(D_PSR, generator=torch.Generator().manual_seed(500 + l))
            pconst["directions"][str(l)] = shared + 0.4 * pconst["directions"][str(l)]
            pctl["directions"][str(l)] = shared + 0.6 * pctl["directions"][str(l)]
        for cls, blob in ((os_.CLASS_BY_ID["constv2-mix-full__lora_r4_ce"], const),
                          (os_.CLASS_BY_ID["constv2ctl2-full__lora_r4_ce"], ctl),
                          (os_.CLASS_BY_ID[LORA_DIFF], diff),
                          (os_.CLASS_BY_ID["constv2-mix-full__psr_ce_all"], pconst),
                          (os_.CLASS_BY_ID["constv2ctl2-full__psr_ce_all"], pctl),
                          (os_.CLASS_BY_ID[DISTILLED], pdiff)):
            if s == 0:
                if write_original:
                    _write_cell(bank, cls.original_id(), blob)
            else:
                if cls.cell_id in skip_classes or s in (skip_seeds or {}).get(cls.cell_id, ()):
                    continue
                if drop_seed == s and cls.role == "difference":
                    continue
                adm = False if (not_admitted_seed == s and cls.derived) else None
                _write_cell(bank, cls.staged_id(s), blob, admitted=adm)


def test_staged_ids_follow_the_lane_grammar():
    c = os_.CLASS_BY_ID[LORA_DIFF]
    assert c.staged_id(2) == "derived_base64s2__constv2-mix-full__lora_r4_ce_minus_ctl"
    assert c.original_id() == "derived__constv2-mix-full__lora_r4_ce_minus_ctl"
    f = os_.CLASS_BY_ID["constv2ctl2-full__psr_ce_all"]
    assert f.staged_id(3) == "base64s3__constv2ctl2-full__psr_ce_all" and f.original_id() == "base__constv2ctl2-full__psr_ce_all"
    assert os_.ORIGINAL_LABEL == "orig_b8_s0"


def test_end_to_end_complete_bank_writes_tables_verdicts_and_figures(tmp_path):
    bank, out = tmp_path / "cells", tmp_path / "out" / "object_space"
    _make_bank(bank)
    res = os_.run(bank, out, include_batch8_reference=True)      # the reference column, opted in
    assert all(t["status"] == "ok" and t["criterion_5"] == "PASS" and t["admitted_count"] == 3 for t in res["tables"])
    for cls in os_.CLASSES:
        assert (out / "tables" / f"{cls.cell_id}.json").is_file() and (out / "tables" / f"{cls.cell_id}.md").is_file()
    t = json.loads((out / "tables" / f"{LORA_DIFF}.json").read_text())
    assert t["original"]["label"] == "orig_b8_s0" and t["original"]["eligible"] and t["original"]["included"]
    kinds = [p["kind"] for p in t["pairs"]]
    assert kinds.count("replicate") == 3 and kinds.count("vs_original") == 3
    assert {p["label"] for p in t["pairs"] if p["kind"] == "replicate"} == {"s1–s2", "s1–s3", "s2–s3"}
    for p in t["pairs"]:
        c = p["components"]["dW"]
        assert set(map(int, c["per_layer_cosine"])) == set(LAYERS_LORA)
        assert -1.0 - 1e-9 <= c["stacked_cosine"] <= 1.0 + 1e-9 and c["stacked_relnorm"] >= 0
    # per-layer norm tables for every column incl. the original
    for col in ("s1", "s2", "s3"):
        assert set(t["replicates"][col]["per_layer_norms"]["dW"]) == {str(l) for l in LAYERS_LORA}
    assert set(t["original"]["per_layer_norms"]["dW"]) == {str(l) for l in LAYERS_LORA}
    # the replicate summary carries mean/min/max over exactly three pairs
    s = t["summary"]["dW"]["replicate"]
    assert s["n_pairs"] == 3 and s["min_stacked_cosine"] <= s["mean_stacked_cosine"] <= s["max_stacked_cosine"]
    # PSR table has direction + gate_w cosines and gate_b abs diffs on 32 layers
    tp = json.loads((out / "tables" / "constv2-mix-full__psr_ce_all.json").read_text())
    assert set(tp["summary"]) >= {"direction", "gate_w", "gate_b"}
    assert len(tp["pairs"][0]["gate_b_abs_diff"]) == 32
    # verdicts per arm, each naming the difference vs both parents with min/max
    v = res["verdicts"]
    for arm in ("lora", "psr"):
        assert v[arm]["criterion_5"] == "PASS"
        assert "seed-stable" in v[arm]["verdict"] and "min" in v[arm]["verdict"]
        assert set(v[arm]["classes"]) == {"constitution", "control", "difference"}
    assert (out / "verdict.md").is_file() and (out / "index.json").is_file()
    idx = json.loads((out / "index.json").read_text())
    assert idx["include_batch8_reference"] is True and idx["class_roster"]["source"] == "declared"
    # figures: cosine + relnorm per class + the summary
    figs = res["figures"]
    assert "seed_stability_summary" in figs and Path(figs["seed_stability_summary"]).is_file()
    for cls in os_.CLASSES:
        assert Path(figs[f"seed_cosine_{cls.cell_id}"]).is_file() and Path(figs[f"seed_relnorm_{cls.cell_id}"]).is_file()
    assert Path(figs["seed_cosine_constv2-mix-full__psr_ce_all"]).parent == tmp_path / "out" / "figs"
    md = (out / "tables" / f"{LORA_DIFF}.md").read_text()
    assert "orig_b8_s0 (reference, never a replicate)" in md and "criterion 5:** PASS" in md


def test_verdict_direction_is_computed_from_means():
    def tbl(role, arm, vals):
        return {"class": f"{role}-{arm}", "arm": arm, "role": role, "status": "ok",
                "summary": {os_.PRIMARY_COMPONENT[arm]: {"replicate": {"mean_stacked_cosine": sum(vals) / 3,
                                                                          "min_stacked_cosine": min(vals), "max_stacked_cosine": max(vals),
                                                                          "stacked_cosine": vals, "n_pairs": 3}}}}
    v = os_.verdicts([tbl("constitution", "lora", [0.9, 0.9, 0.9]), tbl("control", "lora", [0.8, 0.8, 0.8]),
                      tbl("difference", "lora", [0.95, 0.96, 0.97]),
                      tbl("constitution", "psr", [0.9, 0.9, 0.9]), tbl("control", "psr", [0.8, 0.8, 0.8]),
                      tbl("difference", "psr", [0.85, 0.85, 0.85])])
    assert "MORE seed-stable than both parents" in v["lora"]["verdict"]
    assert "between its parents" in v["psr"]["verdict"] and "more seed-stable than the control" in v["psr"]["verdict"]
    v2 = os_.verdicts([tbl("constitution", "lora", [0.9] * 3), tbl("control", "lora", [0.8] * 3), tbl("difference", "lora", [0.5] * 3)])
    assert "LESS seed-stable than both parents" in v2["lora"]["verdict"]
    assert v2["psr"]["criterion_5"] == "FAILED" and v2["psr"]["verdict"].startswith("unavailable")


def test_incomplete_triplet_is_unavailable_and_no_survivor_statistic(tmp_path):
    bank, out = tmp_path / "cells", tmp_path / "out" / "object_space"
    _make_bank(bank, drop_seed=3)           # the two difference objects lack s3
    res = os_.run(bank, out, make_figs=False, include_batch8_reference=True)
    by = {t["class"]: t for t in res["tables"]}
    for cid in (LORA_DIFF, DISTILLED):
        t = by[cid]
        assert t["criterion_5"] == "FAILED"
        assert t["status"].startswith("unavailable: incomplete replicate triplet") and "s3 — no store" in t["status"]
        assert "(2/3 admitted; missing: s3 — no store)" in t["status"] and t["admitted_count"] == 2
        assert t["pairs"] == [] and t["summary"] == {}          # nothing computed over the s1/s2 survivors
        assert t["replicates"]["s3"] == {"present": False, "eligible": False, "reason": "no store"}
        assert t["original"]["eligible"]                        # the opted-in reference column still loads (its norms are shown)
    assert by["constv2-mix-full__lora_r4_ce"]["criterion_5"] == "PASS"
    for arm in ("lora", "psr"):
        assert res["verdicts"][arm]["criterion_5"] == "FAILED"
        assert res["verdicts"][arm]["verdict"].startswith("unavailable")
    md = (out / "tables" / f"{LORA_DIFF}.md").read_text()
    assert "no pair statistic is computed over the survivors" in md and "admitted:** 2/3" in md


def test_o20_no_distilled_object_for_any_seed_reads_unavailable_and_the_five_admitted_classes_are_unaffected(tmp_path):
    """decisions O20 (2026-09-17): the distilled PSR difference FAILED the lane's admission for EVERY seed and is on no roster view,
    so the serving bank holds no `derived_base64s<s>__…_distilled` directory at all. The class is still on the list — through the
    DECLARED `CLASSES` on the library path used here, and through its RS admission files on the CLI's roster+admissions path
    (`test_missing_class_is_unavailable_with_the_admitted_count_not_a_refusal`; the roster alone would drop the row, which is why the
    admission files are the second source) — and reads `unavailable: incomplete replicate triplet (0/3 admitted; missing: s1 — no
    store, s2 — no store, s3 — no store)` with seed stability FAILED; nothing is refused, no other class moves: the LoRA arm's three
    classes and the two PSR parents are complete and PASS, and the PSR arm's verdict is unavailable because its difference class is.
    The batch-8 reference column is opt-in (O23): not loaded by default, loaded only with `include_batch8_reference=True`."""
    bank, out = tmp_path / "cells", tmp_path / "out" / "object_space"
    _make_bank(bank)
    for s in (1, 2, 3):
        shutil.rmtree(bank / os_.CLASS_BY_ID[DISTILLED].staged_id(s))
    assert not any(p.name.endswith(DISTILLED) and p.name.startswith("derived_base64s") for p in bank.iterdir())
    res = os_.run(bank, out, make_figs=False)
    by = {t["class"]: t for t in res["tables"]}
    assert set(by) == {c.cell_id for c in os_.CLASSES}                       # the declared six, the dropped class included
    t = by[DISTILLED]
    assert t["status"] == "unavailable: incomplete replicate triplet (0/3 admitted; missing: s1 — no store, s2 — no store, s3 — no store)"
    assert t["criterion_5"] == "FAILED" and t["pairs"] == [] and t["summary"] == {} and t["admitted_count"] == 0
    assert sorted((m["seed"], m["reason"]) for m in t["missing"]) == [(1, "no store"), (2, "no store"), (3, "no store")]
    assert all(t["replicates"][f"s{s}"] == {"present": False, "eligible": False, "reason": "no store"} for s in (1, 2, 3))
    assert t["original"]["included"] is False and t["original"]["eligible"] is False      # opt-in: not loaded by default
    for cid in [c.cell_id for c in os_.CLASSES if c.cell_id != DISTILLED]:
        assert by[cid]["status"] == "ok" and by[cid]["criterion_5"] == "PASS", cid
    assert res["verdicts"]["lora"]["criterion_5"] == "PASS" and "seed-stable" in res["verdicts"]["lora"]["verdict"]
    assert res["verdicts"]["psr"]["criterion_5"] == "FAILED" and res["verdicts"]["psr"]["verdict"].startswith("unavailable")
    assert (out / "tables" / f"{DISTILLED}.md").is_file() and (out / "tables" / f"{DISTILLED}.json").is_file()
    assert "unavailable: incomplete replicate triplet" in (out / "tables" / f"{DISTILLED}.md").read_text()
    # opted in, the reference column still loads for the dropped class (its norms are shown beside the unavailable status)
    res_on = os_.run(bank, tmp_path / "on", make_figs=False, include_batch8_reference=True)
    t_on = next(t for t in res_on["tables"] if t["class"] == DISTILLED)
    assert t_on["original"]["included"] and t_on["original"]["eligible"] and t_on["criterion_5"] == "FAILED"


def test_not_admitted_derived_object_is_unavailable_with_the_admission_reason(tmp_path):
    bank, out = tmp_path / "cells", tmp_path / "out" / "object_space"
    _make_bank(bank, not_admitted_seed=2)
    res = os_.run(bank, out, make_figs=False)
    t = next(t for t in res["tables"] if t["class"] == DISTILLED)
    assert t["criterion_5"] == "FAILED" and "s2 — not admitted: synthetic reason" in t["status"]
    assert t["replicates"]["s2"]["present"] and not t["replicates"]["s2"]["eligible"]
    assert t["admitted_count"] == 2 and "(2/3 admitted;" in t["status"]


def test_replicate_records_count_rows_and_feed_the_gate(tmp_path):
    bank = tmp_path / "cells"
    _make_bank(bank)
    cls = os_.CLASS_BY_ID["constv2-mix-full__lora_r4_ce"]
    recs = os_.replicate_records(bank, cls)
    assert [r.seed for r in recs] == [1, 2, 3] and all(r.eligible for r in recs)
    trip = tg.require_triplet(recs)
    assert Path(trip[1]).name == cls.staged_id(1)


# ---------------------------------------------------------------- the class roster: roster views + admission files
SIX = ("constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__psr_ce_all",
       "constv2ctl2-full__lora_r4_ce", DISTILLED, LORA_DIFF)


def _roster_yaml(path: Path, *, cells: dict[str, tuple[str, ...]], pending: dict[str, tuple[str, ...]],
                 target: str = "qwen35_4b", views=(1, 2, 3), staged_id=None) -> Path:
    """A synthetic full22 roster: `targets.<target>.views.s<s>.{cells, pending_fit}` mapping cell id → staged id
    (the lane grammar unless ``staged_id`` overrides), plus a non-replicate view that must be ignored."""
    def sid(cid, s):
        if staged_id:
            return staged_id(cid, s)
        return os_.CLASS_BY_ID[cid].staged_id(s) if cid in os_.CLASS_BY_ID else f"base64s{s}__{cid}"
    vws = {}
    for s in views:
        vws[f"s{s}"] = {"config": f"configs/x_s{s}.yaml", "caps": {"bon": 4096},
                        "cells": ({c: sid(c, s) for c in cells.get(f"s{s}", ())} or None),
                        "pending_fit": ({c: sid(c, s) for c in pending.get(f"s{s}", ())} or None)}
    vws["main"] = {"config": "configs/x.yaml", "cells": {"constv2-mix-full__lora_r4_ce": "base__constv2-mix-full__lora_r4_ce",
                                                          "prompted__si_constitution": None}}
    doc = {"targets": {target: {"model": "Qwen/Qwen3.5-4B", "views": vws}}, "lane_flags": {}}
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def _admission_dir(path: Path, verdicts: dict[str, dict[int, tuple[bool, list[str]]]], *, target: str = "qwen35_4b") -> Path:
    """`RS/reseed/admission/<key>__<cell>.json` files: {cell: {seed: (admitted, reasons)}}."""
    path.mkdir(parents=True, exist_ok=True)
    for cell, per in verdicts.items():
        for s, (ok, reasons) in per.items():
            key = f"{target}_base_eb64_s{s}"
            (path / f"{key}__{cell}.json").write_text(json.dumps({
                "admission_doc_kind": "seedrep_derived_admission", "admitted": ok, "reasons": reasons, "cell": cell,
                "derived_id": cell, "key": key, "expected_key": key, "expected_seed": s, "seed": s,
                "kind": "psr_residual" if "distilled" in cell else "lora_subtraction", "checked_at": "2026-01-01T00:00:00+00:00"}))
    return path


REFUSED = ["recovery_verdict 'not recovered' != 'recovered'", "gate liveness min over layers 0.21 < floor 0.81"]


def _post_staging_world(tmp_path: Path):
    """The roster after staging: parents + the LoRA difference under `cells`, the refused distilled id left under
    `pending_fit`; admission files admit the LoRA difference and refuse the distilled object on every seed; the
    bank holds the 15 staged objects and no distilled replicate."""
    roster = _roster_yaml(tmp_path / "roster.yaml",
                          cells={f"s{s}": tuple(c for c in SIX if c != DISTILLED) for s in (1, 2, 3)},
                          pending={f"s{s}": (DISTILLED,) for s in (1, 2, 3)})
    adm = _admission_dir(tmp_path / "admission", {LORA_DIFF: {s: (True, []) for s in (1, 2, 3)},
                                                  DISTILLED: {s: (False, REFUSED) for s in (1, 2, 3)}})
    bank = tmp_path / "cells"
    _make_bank(bank, skip_classes=(DISTILLED,))
    return roster, adm, bank


def test_class_roster_derives_six_classes_from_roster_and_admissions(tmp_path):
    roster, adm, _ = _post_staging_world(tmp_path)
    cr = os_.class_roster(roster, adm)
    assert [c.cell_id for c in cr.classes] == ["constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce", LORA_DIFF,
                                              "constv2-mix-full__psr_ce_all", "constv2ctl2-full__psr_ce_all", DISTILLED]
    assert tuple(cr.classes) == os_.CLASSES                      # the derived roster IS the campaign grammar's six
    assert cr.roster_path == str(roster) and cr.admission_dir == str(adm)
    # the distilled class is on the roster only via pending_fit + the admission files; its admissions say why
    assert {s: (a.admitted, list(a.reasons)) for s, a in cr.for_class(os_.CLASS_BY_ID[DISTILLED]).items()} == \
        {1: (False, REFUSED), 2: (False, REFUSED), 3: (False, REFUSED)}
    assert all(a.admitted for a in cr.for_class(os_.CLASS_BY_ID[LORA_DIFF]).values())
    assert cr.for_class(os_.CLASS_BY_ID["constv2-mix-full__lora_r4_ce"]) == {}
    assert cr.listed[DISTILLED][2] == ("pending_fit", f"derived_base64s2__{DISTILLED}")
    assert cr.listed[LORA_DIFF][1] == ("cells", f"derived_base64s1__{LORA_DIFF}")
    # a distilled id absent from the ROSTER entirely is still a class through its admission files
    roster2 = _roster_yaml(tmp_path / "roster2.yaml", cells={f"s{s}": tuple(c for c in SIX if c != DISTILLED) for s in (1, 2, 3)},
                           pending={})
    assert [c.cell_id for c in os_.class_roster(roster2, adm).classes] == [c.cell_id for c in cr.classes]
    # … and one absent from BOTH is simply not a class (no refusal; the verdict then says `difference: no table`)
    adm2 = _admission_dir(tmp_path / "admission2", {LORA_DIFF: {s: (True, []) for s in (1, 2, 3)}})
    assert [c.cell_id for c in os_.class_roster(roster2, adm2).classes] == [c.cell_id for c in cr.classes if c.cell_id != DISTILLED]


def test_missing_class_is_unavailable_with_the_admitted_count_not_a_refusal(tmp_path):
    roster, adm, bank = _post_staging_world(tmp_path)
    out = tmp_path / "out" / "object_space"
    cr = os_.class_roster(roster, adm)
    res = os_.run(bank, out, make_figs=False, roster=cr)          # must not raise
    by = {t["class"]: t for t in res["tables"]}
    t = by[DISTILLED]
    assert t["status"] == ("unavailable: incomplete replicate triplet (0/3 admitted; missing: "
                           + ", ".join(f"s{s} — not admitted: {'; '.join(REFUSED)}" for s in (1, 2, 3)) + ")")
    assert t["criterion_5"] == "FAILED" and t["admitted_count"] == 0 and t["pairs"] == [] and t["summary"] == {}
    for s in ("s1", "s2", "s3"):
        r = t["replicates"][s]
        assert r["present"] is False and r["eligible"] is False and r["reason"].startswith("not admitted: recovery_verdict")
    for cid in SIX:
        if cid != DISTILLED:
            assert by[cid]["status"] == "ok" and by[cid]["admitted_count"] == 3
    assert res["verdicts"]["lora"]["criterion_5"] == "PASS"
    assert res["verdicts"]["psr"]["criterion_5"] == "FAILED"
    assert res["verdicts"]["psr"]["verdict"].startswith("unavailable: difference: unavailable: incomplete replicate triplet (0/3 admitted;")
    idx = json.loads((out / "index.json").read_text())
    assert idx["class_roster"] == {"source": "roster+admissions", "roster": str(roster), "admission_dir": str(adm)}
    assert idx["admitted_count"][DISTILLED] == 0 and idx["admitted_count"][LORA_DIFF] == 3
    md = (out / "tables" / f"{DISTILLED}.md").read_text()
    assert "admitted:** 0/3" in md and "(0/3 admitted; missing:" in md
    # an ADMITTED derived object whose staged dir is missing reads `no store: admitted … but not staged`
    bank2 = tmp_path / "cells2"
    _make_bank(bank2, skip_classes=(DISTILLED,), skip_seeds={LORA_DIFF: (2,)})
    res2 = os_.run(bank2, tmp_path / "out2", make_figs=False, roster=cr)
    t2 = next(t for t in res2["tables"] if t["class"] == LORA_DIFF)
    assert t2["admitted_count"] == 2 and "(2/3 admitted; missing: s2 — no store: admitted by " in t2["status"] \
        and "but not staged in the bank)" in t2["status"]
    # a class on neither source: no table, and the run still does not raise
    adm3 = _admission_dir(tmp_path / "admission3", {LORA_DIFF: {s: (True, []) for s in (1, 2, 3)}})
    roster3 = _roster_yaml(tmp_path / "roster3.yaml", cells={f"s{s}": tuple(c for c in SIX if c != DISTILLED) for s in (1, 2, 3)}, pending={})
    res3 = os_.run(bank, tmp_path / "out3", make_figs=False, roster=os_.class_roster(roster3, adm3))
    assert len(res3["tables"]) == 5 and res3["verdicts"]["psr"]["verdict"] == "unavailable: difference: no table"


def test_class_roster_refuses_grammar_drift_and_bad_inputs(tmp_path):
    adm = _admission_dir(tmp_path / "admission", {})           # no admission files: the roster alone is under test
    # a difference id staged without the derived_ prefix
    bad = _roster_yaml(tmp_path / "bad.yaml", cells={"s1": (LORA_DIFF,)}, pending={},
                       staged_id=lambda cid, s: f"base64s{s}__{cid}")
    with pytest.raises(ValueError, match="staged-id prefix and the cell grammar disagree"):
        os_.class_roster(bad, adm)
    # a staged id outside the lane grammar
    bad2 = _roster_yaml(tmp_path / "bad2.yaml", cells={"s1": ("constv2-mix-full__lora_r4_ce",)}, pending={},
                        staged_id=lambda cid, s: f"base64s{s}__other")
    with pytest.raises(ValueError, match="not the lane grammar"):
        os_.class_roster(bad2, adm)
    # an SI with no declared role
    bad3 = _roster_yaml(tmp_path / "bad3.yaml", cells={"s1": ("safety_refusal__lora_r4_ce",)}, pending={})
    with pytest.raises(ValueError, match="no declared role"):
        os_.class_roster(bad3, adm)
    # an admission file for a cell the roster stages as a fit
    adm4 = _admission_dir(tmp_path / "admission4", {"constv2-mix-full__lora_r4_ce": {1: (True, [])}})
    ok = _roster_yaml(tmp_path / "ok.yaml", cells={"s1": ("constv2-mix-full__lora_r4_ce",)}, pending={})
    with pytest.raises(ValueError, match="stages it as a fit"):
        os_.class_roster(ok, adm4)
    # an absent admission dir is a fault, not an empty roster
    with pytest.raises(FileNotFoundError):
        os_.class_roster(ok, tmp_path / "nope")
    # a roster with no such target
    with pytest.raises(ValueError, match="names no target"):
        os_.class_roster(ok, adm, target="other")
    # classify_cell on its own
    assert os_.classify_cell("constv2ctl2-full__psr_ce_all", derived=False) == os_.CLASS_BY_ID["constv2ctl2-full__psr_ce_all"]
    assert os_.classify_cell(DISTILLED, derived=True) == os_.CLASS_BY_ID[DISTILLED]
    with pytest.raises(ValueError, match="neither a psr_"):
        os_.classify_cell("constv2-mix-full__vector_all", derived=False)
    # admission files of another target are ignored; a duplicate for one (cell, seed) refuses
    d = _admission_dir(tmp_path / "admission5", {LORA_DIFF: {1: (True, [])}}, target="llama8b")
    assert os_.read_admissions(d) == {}
    (d / "dup.json").write_text(json.dumps({"admitted": True, "reasons": [], "cell": LORA_DIFF, "key": "qwen35_4b_base_eb64_s1"}))
    (d / "dup2.json").write_text(json.dumps({"admitted": True, "reasons": [], "cell": LORA_DIFF, "key": "qwen35_4b_base_eb64_s1"}))
    with pytest.raises(ValueError, match="two admission files"):
        os_.read_admissions(d)


def test_the_repo_roster_and_grammar_agree(tmp_path):
    """The checked-in roster's replicate views name a subset of the campaign grammar's six cells with the lane's staged ids
    (whether under `cells` or `pending_fit`), always the four parents, each on all three seeds. A derived id the admission
    refused may be on no view at all (the post-staging roster, decisions O20) — it enters the class list through its
    admission files, so roster + admission files for both derived ids derive exactly the declared `CLASSES`."""
    listed = os_.roster_cells(os_.load_roster(os_.DEFAULT_ROSTER))
    declared = {c.cell_id for c in os_.CLASSES}
    assert set(listed) <= declared and {c.cell_id for c in os_.CLASSES if not c.derived} <= set(listed)
    for cid, per in listed.items():
        assert set(per) == {1, 2, 3}, cid
        for s, (_, sid) in per.items():
            assert sid == os_.CLASS_BY_ID[cid].staged_id(s)
    adm = _admission_dir(tmp_path / "admission", {LORA_DIFF: {s: (True, []) for s in (1, 2, 3)},
                                                  DISTILLED: {s: (False, REFUSED) for s in (1, 2, 3)}})
    assert tuple(os_.class_roster(os_.DEFAULT_ROSTER, adm).classes) == os_.CLASSES
    # with NO admission files the roster alone still yields every class it lists, in the grammar's order
    none = _admission_dir(tmp_path / "none", {})
    assert [c.cell_id for c in os_.class_roster(os_.DEFAULT_ROSTER, none).classes] == [c.cell_id for c in os_.CLASSES if c.cell_id in listed]


# ---------------------------------------------------------------- the batch-8 reference column is opt-in (O23)
def test_batch8_reference_column_absent_by_default_present_only_with_the_flag(tmp_path, monkeypatch):
    bank = tmp_path / "cells"
    _make_bank(bank)                                              # the bank DOES hold base__/derived__ originals
    loaded: list[str] = []
    real = os_.load_blob
    monkeypatch.setattr(os_, "load_blob", lambda p: (loaded.append(Path(p).parent.name), real(p))[1])
    res = os_.run(bank, tmp_path / "off" / "object_space", make_figs=True)
    assert not any(n.startswith(("base__", "derived__")) for n in loaded), loaded
    assert len(loaded) == 18                                        # 6 classes × 3 replicates, nothing else
    for t in res["tables"]:
        assert t["status"] == "ok" and t["include_batch8_reference"] is False
        assert t["original"] == {"label": "orig_b8_s0", "included": False, "eligible": False, "reason": os_.ORIGINAL_NOT_LOADED}
        assert [p["kind"] for p in t["pairs"]] == ["replicate"] * 3
        for comp, kinds in t["summary"].items():
            if comp != "gate_b":
                assert kinds["vs_original"] is None and kinds["replicate"]["n_pairs"] == 3
        if t["arm"] == "psr":
            assert set(t["summary"]["gate_b"]) == {"replicate"}
    md = (tmp_path / "off" / "object_space" / "tables" / f"{LORA_DIFF}.md").read_text()
    assert "orig_b8_s0 (reference, never a replicate)" not in md and "not loaded: the batch-8 seed-0 reference column is opt-in" in md
    assert "| orig_b8_s0 |" not in md                                # no norm-table column either
    idx = json.loads((tmp_path / "off" / "object_space" / "index.json").read_text())
    assert idx["include_batch8_reference"] is False and Path(res["figures"]["seed_stability_summary"]).is_file()
    # with the flag: the six originals load, three vs-original pairs per class, the column is tabled
    loaded.clear()
    res_on = os_.run(bank, tmp_path / "on" / "object_space", make_figs=False, include_batch8_reference=True)
    assert sorted(n for n in loaded if n.startswith(("base__", "derived__"))) == sorted(c.original_id() for c in os_.CLASSES)
    for t in res_on["tables"]:
        assert t["original"]["included"] and t["original"]["eligible"]
        assert [p["kind"] for p in t["pairs"]].count("vs_original") == 3
        assert t["summary"][t["primary_component"]]["vs_original"]["n_pairs"] == 3
    md_on = (tmp_path / "on" / "object_space" / "tables" / f"{LORA_DIFF}.md").read_text()
    assert "orig_b8_s0 (reference, never a replicate)" in md_on and "| orig_b8_s0 |" in md_on
    # the replicate statistics are identical either way — the reference column never touches them
    for t_off, t_on in zip(res["tables"], res_on["tables"]):
        assert [p["components"] for p in t_off["pairs"]] == [p["components"] for p in t_on["pairs"] if p["kind"] == "replicate"]


def test_main_cli_derives_the_roster_and_defaults_the_reference_column_off(tmp_path, capsys):
    roster, adm, bank = _post_staging_world(tmp_path)
    out = tmp_path / "out" / "object_space"
    base = ["--bank", str(bank), "--out", str(out), "--no-figures", "--roster", str(roster), "--admission-dir", str(adm)]
    assert os_.main(base) == 0
    idx = json.loads((out / "index.json").read_text())
    assert idx["include_batch8_reference"] is False
    assert idx["class_roster"] == {"source": "roster+admissions", "roster": str(roster), "admission_dir": str(adm)}
    assert idx["classes"] == [c.cell_id for c in os_.CLASSES] and idx["admitted_count"][DISTILLED] == 0
    printed = capsys.readouterr().out
    assert "class roster: 6 class(es)" in printed and "not loaded (opt-in)" in printed and "(0/3 admitted;" in printed
    t = json.loads((out / "tables" / f"{LORA_DIFF}.json").read_text())
    assert t["original"]["included"] is False and all(p["kind"] == "replicate" for p in t["pairs"])
    assert os_.main(base + ["--include-batch8-reference"]) == 0
    idx2 = json.loads((out / "index.json").read_text())
    assert idx2["include_batch8_reference"] is True
    t2 = json.loads((out / "tables" / f"{LORA_DIFF}.json").read_text())
    assert t2["original"]["included"] and t2["original"]["eligible"] and [p["kind"] for p in t2["pairs"]].count("vs_original") == 3
    assert "INCLUDED (--include-batch8-reference)" in capsys.readouterr().out
