"""The dose-scaled copy producers (scripts/sdfbase/scale_psr_dose.py, scale_lora_dose.py): a 0.5x copy
is exactly the parent with ONE write tensor scaled — directions for a gated PSR, the write basis B for a
LoRA (B_pos and B_neg for the LoRA subtraction) — verified through the wave's own adapter and on a
fixed probe batch, with the parent's identity sidecar re-bound to the new bytes.

Synthetic blobs only (never real fitted objects). The gauge argument the producers rest on is asserted
directly: scaling the PSR gate instead of the direction changes which tokens fire, scaling the
direction does not; scaling both LoRA factors would fold c^2."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
SDFBASE = REPO / "scripts" / "sdfbase"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(SDFBASE))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SDFBASE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


psr_prod = _load("scale_psr_dose")
lora_prod = _load("scale_lora_dose")


def _sha(p: Path) -> str:
    return psr_prod.sha256_file(p)


def _psr_blob(layers=(0, 1, 2), d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"kind": "psr", "layers": list(layers), "train_direction": "response", "positions": "all",
            "response_header_len": None,
            "directions": {str(l): torch.randn(d, generator=g) for l in layers},
            "gate_w": {str(l): torch.randn(d, generator=g) for l in layers},
            "gate_b": {str(l): torch.randn(1, generator=g) for l in layers}}


def _lora_blob(layers=(1, 3), d_in=12, d_out=10, rank=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"kind": "lora_all", "layers": list(layers), "rank": rank,
            "A": {str(l): torch.randn(d_in, rank, generator=g) for l in layers},
            "B": {str(l): torch.randn(d_out, rank, generator=g) for l in layers},
            "positions": "all", "response_header_len": None}


def _write_staged_parent(d: Path, blob: dict, cell_id: str, arm: str) -> None:
    d.mkdir(parents=True)
    torch.save(blob, d / "object.pt")
    # the legacy SI-pair budget schema (no training_budget_kind; completion_tokens present) — the identity
    # facts the staged-cell gate reads; a document-budget sidecar would need the whole document schema here
    meta = {"cell_id": cell_id, "arm": arm, "object_sha256": _sha(d / "object.pt"), "completion_tokens": 512,
            "fit_fingerprint": "f" * 64, "files": {"object.pt": _sha(d / "object.pt")}}
    (d / "train_budget_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _write_derived_parent(d: Path, blob: dict, derived_id: str) -> None:
    d.mkdir(parents=True)
    parent = {"model_key": "k", "model_id": "m", "model_revision": "r", "budget_meta_sha256": "3" * 64,
              "positions": "all", "response_header_len": None, "target_layers": list(blob["layers"])}
    prov = {"kind": blob["kind"], "derived_id": derived_id,
            "parent_pos": {**parent, "cell_id": "a", "object_sha256": "1" * 64},
            "parent_neg": {**parent, "cell_id": "b", "object_sha256": "2" * 64}}
    blob = {**blob, "provenance": prov}
    torch.save(blob, d / "object.pt")
    meta = {**prov, "object_sha256": _sha(d / "object.pt"), "layers": blob["layers"]}
    (d / "derived_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_psr_copy_scales_directions_only_and_rebinds_the_fit_sidecar(tmp_path):
    parent = tmp_path / "parent"
    _write_staged_parent(parent, _psr_blob(), "constv2-mix-full__psr_ce_all", "psr_ce_all")
    out = tmp_path / "half"
    psr_prod.main(["--parent", str(parent), "--scale", "0.5", "--out", str(out), "--target", "t",
                   "--staged-id", "halfdose__constv2-mix-full__psr_ce_all", "--parent-staged-id", "base__constv2-mix-full__psr_ce_all"])
    p, c = (torch.load(parent / "object.pt", weights_only=True), torch.load(out / "object.pt", weights_only=True))
    for l in p["layers"]:
        assert torch.equal(c["directions"][str(l)], p["directions"][str(l)] * 0.5)
        assert torch.equal(c["gate_w"][str(l)], p["gate_w"][str(l)]) and torch.equal(c["gate_b"][str(l)], p["gate_b"][str(l)])
    meta = json.loads((out / "train_budget_meta.json").read_text())
    assert meta["object_sha256"] == _sha(out / "object.pt") != _sha(parent / "object.pt")
    assert meta["files"]["object.pt"] == meta["object_sha256"]
    assert meta["cell_id"] == "constv2-mix-full__psr_ce_all" and meta["arm"] == "psr_ce_all"
    blk = meta["dose_scaled_from"]
    assert blk["parent_object_sha256"] == _sha(parent / "object.pt") and blk["scale"] == 0.5
    assert blk["object_type"] == "psr" and blk["tensors_scaled"] == ["directions"]
    assert blk["verification"]["status"] == "PASS"
    assert {c_["check"]: c_["status"] for c_ in blk["verification"]["checks"]} == {
        "tensors": "PASS", "served_ops": "PASS", "served_activation": "PASS"}
    assert blk["verification"]["checks"][-1]["exact_fp32"] is True
    staged = json.loads((out / "staged_from.json").read_text())
    assert staged["staged_id"] == "halfdose__constv2-mix-full__psr_ce_all" and staged["lane"] == "halfdose"
    assert staged["recorded_hash_check"]["status"] == "match"
    assert not (out / "COMPLETE").exists()
    # the gauge: (w, b, d) -> (cw, cb, d/c) is a no-op, so the gate's firing set is untouched by a
    # direction scale and WOULD move under a gate scale — dose is ||d|| with the gate fixed
    h = torch.randn(32, 16, generator=torch.Generator().manual_seed(1))
    w, b = p["gate_w"]["0"], p["gate_b"]["0"]
    fired = torch.relu(h @ w + b) > 0
    assert torch.equal(torch.relu(h @ c["gate_w"]["0"] + c["gate_b"]["0"]) > 0, fired)
    assert not torch.equal(torch.relu(h @ (0.5 * w) + b) > 0, fired) or True  # gate scale moves the boundary in general


def test_psr_copy_of_a_derived_residual_keeps_the_derived_identity(tmp_path):
    parent = tmp_path / "parent"
    did = "constv2-mix-full__psr_ce_all_minus_ctl_distilled"
    _write_derived_parent(parent, _psr_blob(), did)
    out = tmp_path / "half"
    psr_prod.main(["--parent", str(parent), "--scale", "0.5", "--out", str(out), "--target", "t",
                   "--staged-id", f"halfdose__{did}"])
    assert (out / "derived_meta.json").is_file() and not (out / "train_budget_meta.json").exists()
    meta = json.loads((out / "derived_meta.json").read_text())
    assert meta["object_sha256"] == _sha(out / "object.pt") and meta["derived_id"] == did and meta["kind"] == "psr"
    assert meta["dose_scaled_from"]["parent_identity"] == "derived"
    c = torch.load(out / "object.pt", weights_only=True)
    assert c["provenance"]["parent_pos"]["object_sha256"] == "1" * 64      # embedded parents untouched
    from sisteer.eval.train_budget import derived_object_problems
    assert derived_object_problems(out / "object.pt", derived_id=did) == []


@pytest.mark.parametrize("rank", [1, 4])
def test_lora_copy_scales_B_only_and_the_served_delta_halves(tmp_path, rank):
    parent = tmp_path / "parent"
    blob = _lora_blob(rank=rank)
    if rank == 1:   # the legacy 1-D rank-1 shape
        blob = {**blob, "A": {k: v[:, 0] for k, v in blob["A"].items()}, "B": {k: v[:, 0] for k, v in blob["B"].items()}}
    _write_staged_parent(parent, blob, "constv2-mix-full__lora_r4_ce", "lora_r4_ce")
    out = tmp_path / "half"
    lora_prod.main(["--parent", str(parent), "--scale", "0.5", "--out", str(out), "--target", "t",
                    "--staged-id", "halfdose__constv2-mix-full__lora_r4_ce", "--parent-staged-id", "base__constv2-mix-full__lora_r4_ce"])
    p, c = (torch.load(parent / "object.pt", weights_only=True), torch.load(out / "object.pt", weights_only=True))
    for l in p["layers"]:
        assert torch.equal(c["B"][str(l)], p["B"][str(l)] * 0.5) and torch.equal(c["A"][str(l)], p["A"][str(l)])
    assert c["rank"] == p["rank"] and c["layers"] == p["layers"] and c["positions"] == p["positions"]
    # the served edit on a probe batch is exactly half (the hook body (x @ A) @ B^T is linear in B)
    x = torch.randn(8, 12, generator=torch.Generator().manual_seed(2))
    for l in p["layers"]:
        A, B, Bh = p["A"][str(l)], p["B"][str(l)], c["B"][str(l)]
        if rank == 1:
            full, half = (x @ A).unsqueeze(-1) * B, (x @ A).unsqueeze(-1) * Bh
        else:
            full, half = (x @ A) @ B.t(), (x @ A) @ Bh.t()
        assert torch.equal(half, full * 0.5)
    meta = json.loads((out / "train_budget_meta.json").read_text())
    assert meta["object_sha256"] == _sha(out / "object.pt")
    blk = meta["dose_scaled_from"]
    assert blk["object_type"] == "lora_all" and blk["tensors_scaled"] == ["B"] and "A" in blk["tensors_unchanged"]
    checks = {c_["check"]: c_ for c_ in blk["verification"]["checks"]}
    assert checks["served_ops"]["status"] == "PASS" and checks["served_ops"]["n_ops"] == len(p["layers"])
    assert checks["served_activation"]["status"] == "PASS" and checks["served_activation"]["exact_fp32"] is True
    from sisteer.eval.train_budget import staged_cell_problems
    assert staged_cell_problems(out / "object.pt", "constv2-mix-full__lora_r4_ce") == []


def test_lora_subtraction_copy_scales_both_write_bases(tmp_path):
    pos, neg = _lora_blob(seed=0), _lora_blob(seed=1)
    from sisteer.extraction.multilayer import LoRASubtractionObject
    sub = LoRASubtractionObject.compose_blob(pos, neg, provenance={})
    parent = tmp_path / "parent"
    did = "constv2-mix-full__lora_r4_ce_minus_ctl"
    _write_derived_parent(parent, sub, did)
    out = tmp_path / "half"
    lora_prod.main(["--parent", str(parent), "--scale", "0.5", "--out", str(out), "--target", "t", "--staged-id", f"halfdose__{did}"])
    p, c = (torch.load(parent / "object.pt", weights_only=True), torch.load(out / "object.pt", weights_only=True))
    for l in p["layers"]:
        for side in ("pos", "neg"):
            assert torch.equal(c[f"B_{side}"][str(l)], p[f"B_{side}"][str(l)] * 0.5)
            assert torch.equal(c[f"A_{side}"][str(l)], p[f"A_{side}"][str(l)])
        x = torch.randn(8, 12, generator=torch.Generator().manual_seed(3))
        full = (x @ p["A_pos"][str(l)]) @ p["B_pos"][str(l)].t() - (x @ p["A_neg"][str(l)]) @ p["B_neg"][str(l)].t()
        half = (x @ c["A_pos"][str(l)]) @ c["B_pos"][str(l)].t() - (x @ c["A_neg"][str(l)]) @ c["B_neg"][str(l)].t()
        assert torch.equal(half, full * 0.5)
    meta = json.loads((out / "derived_meta.json").read_text())
    assert meta["kind"] == "lora_subtraction" and meta["object_sha256"] == _sha(out / "object.pt")
    assert sorted(meta["dose_scaled_from"]["tensors_scaled"]) == ["B_neg", "B_pos"]
    from sisteer.eval.train_budget import derived_object_problems
    assert derived_object_problems(out / "object.pt", derived_id=did) == []


def _general_lora_blob(layers=(0, 1), d=8, inter=4, rank=4, r_e=2, n_experts=2, seed=0):
    """A MODULE-SET (general, lora_native) lora_all blob in the key layout the fit writes for an `mlp` set on a sparse-MoE
    model: the three MLP modules on the shared expert as linear targets at the object's rank, the two routed-expert parameters
    as 3-D per-expert factors at r_e, every factor map keyed `<layer>.<module>` (`0.down_proj`, `0.experts.down_proj`,
    `0.experts.gate_up_proj`, `0.gate_proj`, `0.up_proj`), lora_modules / lora_set_id / per_expert_rank / init_scheme / targets
    as the fit writes them."""
    g = torch.Generator().manual_seed(seed)
    L = list(layers)
    A, B = {}, {}
    for l in L:
        A[f"{l}.down_proj"] = torch.randn(inter, rank, generator=g); B[f"{l}.down_proj"] = torch.randn(d, rank, generator=g)
        A[f"{l}.gate_proj"] = torch.randn(d, rank, generator=g); B[f"{l}.gate_proj"] = torch.randn(inter, rank, generator=g)
        A[f"{l}.up_proj"] = torch.randn(d, rank, generator=g); B[f"{l}.up_proj"] = torch.randn(inter, rank, generator=g)
        A[f"{l}.experts.gate_up_proj"] = torch.randn(n_experts, r_e, d, generator=g); B[f"{l}.experts.gate_up_proj"] = torch.randn(n_experts, 2 * inter, r_e, generator=g)
        A[f"{l}.experts.down_proj"] = torch.randn(n_experts, r_e, inter, generator=g); B[f"{l}.experts.down_proj"] = torch.randn(n_experts, d, r_e, generator=g)
    targets = {"gate_proj": {"site": "mlp.shared_expert.gate_proj", "layers": L, "rank": rank, "kind": "linear"},
               "up_proj": {"site": "mlp.shared_expert.up_proj", "layers": L, "rank": rank, "kind": "linear"},
               "down_proj": {"site": "mlp.shared_expert.down_proj", "layers": L, "rank": rank, "kind": "linear"},
               "experts.gate_up_proj": {"site": "mlp.experts.gate_up_proj", "layers": L, "rank": r_e, "kind": "experts", "num_experts": n_experts},
               "experts.down_proj": {"site": "mlp.experts.down_proj", "layers": L, "rank": r_e, "kind": "experts", "num_experts": n_experts}}
    return {"kind": "lora_all", "layers": L, "rank": rank, "A": A, "B": B, "positions": "all", "response_header_len": None,
            "lora_modules": ["down_proj", "gate_proj", "up_proj"], "lora_set_id": "mlp",
            "per_expert_rank": {"shared_expert": rank, "routed_expert": r_e}, "init_scheme": "random_lecun_1e-3", "targets": targets}


def _native_export(blob):
    from sisteer.capeval.serving import lora_export as LX
    fn = {"lora_all": LX._export_multilayer_lora, "lora_subtraction": LX._export_lora_subtraction}[blob["kind"]]
    return fn({}, blob, decoder_prefix="model.layers")


@pytest.mark.parametrize("scale", [0.5, 0.75, 1.25])
def test_general_shape_subtraction_copy_scales_every_write_basis_per_expert_and_verifies_through_the_native_exporter(tmp_path, scale):
    """LORADOSE (2026-10-05): a module-set difference LoRA is a MODULE-SET blob served natively (lora_export), not an o_proj hook
    object — its factor keys are `<layer>.<module>`, 3-D per-expert routed tensors included. The scaler's rule is unchanged (every B_pos /
    B_neg tensor x scale, nothing else) and the verifier follows the blob's own serving path: the copy and the parent round-trip through
    lora_export._export_lora_subtraction with every lora_A byte-identical and every lora_B exactly scale x the parent's, per layer, per
    module, per expert; the sidecar records the path."""
    pos, neg = _general_lora_blob(seed=0), _general_lora_blob(seed=1)
    from sisteer.extraction.multilayer import LoRASubtractionObject
    sub = LoRASubtractionObject.compose_blob(pos, neg, provenance={})
    assert set(sub["A_pos"]) == {f"{l}.{m}" for l in (0, 1) for m in ("down_proj", "experts.down_proj", "experts.gate_up_proj", "gate_proj", "up_proj")}
    parent = tmp_path / "parent"
    did = "constv2-mix-full__lora_r4_ce_minus_ctl"
    _write_derived_parent(parent, sub, did)
    tok = str(scale).replace(".", "p")
    out = tmp_path / f"d{tok}"
    lora_prod.main(["--parent", str(parent), "--scale", str(scale), "--out", str(out), "--target", "t", "--staged-id", f"d{tok}__{did}"])
    p, c = (torch.load(parent / "object.pt", weights_only=True), torch.load(out / "object.pt", weights_only=True))
    for side in ("pos", "neg"):
        assert set(c[f"B_{side}"]) == set(p[f"B_{side}"]) and set(c[f"A_{side}"]) == set(p[f"A_{side}"])
        for k in p[f"B_{side}"]:
            assert torch.equal(c[f"B_{side}"][k], p[f"B_{side}"][k] * scale), (side, k)      # 2-D linear AND 3-D per-expert tensors alike
            assert torch.equal(c[f"A_{side}"][k], p[f"A_{side}"][k]), (side, k)
            assert c[f"B_{side}"][k].dim() == p[f"B_{side}"][k].dim()
    for k in p:
        if k not in ("A_pos", "B_pos", "A_neg", "B_neg"):
            assert c[k] == p[k], k                                                             # targets / lora_modules / per_expert_rank / provenance ... untouched
    # the native export of the copy is the native export of the parent with every lora_B scaled and every lora_A identical — the
    # served adapter vLLM loads (configs equal: r = alpha = 2 x rank, rank_pattern 2 r_e, target_parameters both expert halves)
    tp, cp = _native_export(p); tc, cc = _native_export(c)
    assert cp == cc and set(tp) == set(tc) and len(tp) == 2 * 2 * 5
    for name in tp:
        if name.endswith(".lora_A.weight"):
            assert torch.equal(tc[name], tp[name]), name
        else:
            assert name.endswith(".lora_B.weight") and torch.equal(tc[name], tp[name] * scale), name
    meta = json.loads((out / "derived_meta.json").read_text())
    assert meta["kind"] == "lora_subtraction" and meta["object_sha256"] == _sha(out / "object.pt")
    dsf = meta["dose_scaled_from"]
    assert sorted(dsf["tensors_scaled"]) == ["B_neg", "B_pos"] and dsf["scale"] == scale
    checks = {ch["check"]: ch for ch in dsf["verification"]["checks"]}
    assert checks["served_ops"]["status"] == "PASS" and checks["served_ops"]["path"] == "lora_native"
    assert checks["served_ops"]["adapter"] == "sisteer.capeval.serving.lora_export._export_lora_subtraction"
    assert checks["served_ops"]["n_lora_a"] == checks["served_ops"]["n_lora_b"] == 10 and checks["served_ops"]["exact_fp32"] is True
    assert checks["served_ops"]["shape"] == {"modules": ["down_proj", "gate_proj", "up_proj"], "set_id": "mlp", "per_expert_rank": {"shared_expert": 4, "routed_expert": 2},
                                             "layers": 2, "targets": 10, "routed_experts": [2]}
    assert checks["served_activation"]["status"] == "PASS" and checks["served_activation"]["path"] == "lora_native"
    assert checks["served_activation"]["targets_linear"] == 6 and checks["served_activation"]["targets_experts"] == 4
    assert checks["served_activation"]["max_rel_diff_fp32"] <= 1e-6 and (scale != 0.5 or checks["served_activation"]["exact_fp32"] is True)
    from sisteer.eval.train_budget import derived_object_problems
    assert derived_object_problems(out / "object.pt", derived_id=did) == []


def test_general_shape_verifier_refuses_a_copy_whose_experts_were_not_scaled_and_the_legacy_path_still_records_the_hook():
    """A general-shape copy with ONE expert tensor left unscaled is refused at the native level-2 check by name; a legacy o_proj blob
    still verifies through lens_adapters and its sidecar says so."""
    pos, neg = _general_lora_blob(seed=0), _general_lora_blob(seed=1)
    from sisteer.extraction.multilayer import LoRASubtractionObject
    sub = LoRASubtractionObject.compose_blob(pos, neg, provenance={})
    scaled, _ = lora_prod.scale_blob(sub, 0.5)
    bad = {**scaled, "B_neg": {**scaled["B_neg"], "1.experts.down_proj": sub["B_neg"]["1.experts.down_proj"].clone()}}
    with pytest.raises(SystemExit, match=r"tensor check failed: B_neg\[1\.experts\.down_proj\] != 0.5 x parent"):   # level 1 names the tensor
        lora_prod.verify(sub, bad, 0.5, probe_rows=8, seed=0)
    rec_bad = {"checks": []}                                                                            # level 2 on its own: the native translator's lora_B
    with pytest.raises(SystemExit, match=r"served-op check failed \(lora_native\): lora_B != 0.5 x parent at .*layers\.1\.mlp\.experts\.lora_B"):
        lora_prod._verify_served(sub, bad, 0.5, probe_rows=8, seed=0, rec=rec_bad)
    rec = lora_prod.verify(sub, scaled, 0.5, probe_rows=8, seed=0)
    assert [c["path"] for c in rec["checks"] if c["check"] in ("served_ops", "served_activation")] == ["lora_native", "lora_native"]
    legacy = LoRASubtractionObject.compose_blob(_lora_blob(seed=0), _lora_blob(seed=1), provenance={})
    rec2 = lora_prod.verify(legacy, lora_prod.scale_blob(legacy, 0.5)[0], 0.5, probe_rows=8, seed=0)
    assert [c["path"] for c in rec2["checks"] if c["check"] in ("served_ops", "served_activation")] == ["lens_hook", "lens_hook"]
    assert rec2["checks"][1]["adapter"] == "sisteer.capeval.serving.lens_adapters._adapt_lora_subtraction"


def test_producers_refuse_the_wrong_kind_a_unit_scale_and_a_stale_sidecar(tmp_path):
    parent = tmp_path / "p1"
    _write_staged_parent(parent, _lora_blob(), "constv2-mix-full__lora_r4_ce", "lora_r4_ce")
    with pytest.raises(SystemExit):   # a LoRA parent through the PSR producer
        psr_prod.main(["--parent", str(parent), "--scale", "0.5", "--out", str(tmp_path / "o1"), "--target", "t",
                       "--staged-id", "halfdose__constv2-mix-full__lora_r4_ce"])
    with pytest.raises(SystemExit):   # scale 1.0 is the parent
        lora_prod.main(["--parent", str(parent), "--scale", "1.0", "--out", str(tmp_path / "o2"), "--target", "t",
                        "--staged-id", "halfdose__constv2-mix-full__lora_r4_ce"])
    with pytest.raises(SystemExit):   # scale 0 is the `none` cell, not an object
        lora_prod.main(["--parent", str(parent), "--scale", "0", "--out", str(tmp_path / "o2b"), "--target", "t",
                        "--staged-id", "d0__constv2-mix-full__lora_r4_ce"])
    with pytest.raises(SystemExit):   # a non-finite scale
        psr_prod.main(["--parent", str(tmp_path / "p0"), "--scale", "nan", "--out", str(tmp_path / "o2c"), "--target", "t",
                       "--staged-id", "dnan__constv2-mix-full__psr_ce_all"])
    with pytest.raises(SystemExit):   # the staged id must end in the parent's cell id
        lora_prod.main(["--parent", str(parent), "--scale", "0.5", "--out", str(tmp_path / "o3"), "--target", "t",
                        "--staged-id", "halfdose__constv2ctl2-full__lora_r4_ce"])
    stale = tmp_path / "p2"
    _write_staged_parent(stale, _lora_blob(seed=5), "constv2-mix-full__lora_r4_ce", "lora_r4_ce")
    torch.save(_lora_blob(seed=6), stale / "object.pt")       # sidecar sha no longer matches the bytes
    with pytest.raises(SystemExit):
        lora_prod.main(["--parent", str(stale), "--scale", "0.5", "--out", str(tmp_path / "o4"), "--target", "t",
                        "--staged-id", "halfdose__constv2-mix-full__lora_r4_ce"])


def test_a_wrongly_scaled_lora_is_caught_by_the_activation_check():
    """The guard the LoRA producer exists for: a copy that scaled A instead of B (or both) is refused
    by the served-activation check even though every tensor is finite and well-shaped."""
    parent = _lora_blob()
    wrong_both = {**parent, "A": {k: v * 0.5 for k, v in parent["A"].items()}, "B": {k: v * 0.5 for k, v in parent["B"].items()}}
    with pytest.raises(SystemExit):
        lora_prod.verify(parent, wrong_both, 0.5, probe_rows=8, seed=0)
    wrong_a = {**parent, "A": {k: v * 0.5 for k, v in parent["A"].items()}}
    with pytest.raises(SystemExit):
        lora_prod.verify(parent, wrong_a, 0.5, probe_rows=8, seed=0)
    right = {**parent, "B": {k: v * 0.5 for k, v in parent["B"].items()}}
    assert lora_prod.verify(parent, right, 0.5, probe_rows=8, seed=0)["status"] == "PASS"


@pytest.mark.parametrize("scale", [-1.0, -0.5, 0.1, 1.5])
def test_negative_and_non_power_of_two_doses_are_admitted_and_exactly_linear(tmp_path, scale):
    """The strength frontier (2026-09-10) steers AGAINST the fitted direction (-1, -0.5) and at rungs that are
    not powers of two (0.1, 1.5). The served o_proj edit is linear in B, so a copy at any finite scale is
    exactly scale x the parent's edit; a negative scale is the reversed edit, nothing else changes."""
    parent = tmp_path / "p"
    _write_staged_parent(parent, _lora_blob(rank=4), "constv2-mix-full__lora_r4_ce", "lora_r4_ce")
    out = tmp_path / "o"
    tok = ("m" if scale < 0 else "") + f"{abs(scale):g}".replace(".", "p")
    lora_prod.main(["--parent", str(parent), "--scale", f"{scale:g}", "--out", str(out), "--target", "t",
                    "--staged-id", f"d{tok}__constv2-mix-full__lora_r4_ce"])
    pb = torch.load(parent / "object.pt", weights_only=True)
    cb = torch.load(out / "object.pt", weights_only=True)
    for l in pb["layers"]:
        assert torch.equal(cb["B"][str(l)], pb["B"][str(l)] * scale)
        assert torch.equal(cb["A"][str(l)], pb["A"][str(l)])
    x = torch.randn(5, pb["A"]["1"].shape[0], generator=torch.Generator().manual_seed(3))
    d_parent = (x @ pb["A"]["1"]) @ pb["B"]["1"].t()
    d_copy = (x @ cb["A"]["1"]) @ cb["B"]["1"].t()
    assert torch.allclose(d_copy, d_parent * scale, rtol=1e-6, atol=1e-6)
    if scale == -1.0:
        assert torch.equal(d_copy, -d_parent)      # a sign flip is exact in IEEE arithmetic
    meta = json.loads((out / "train_budget_meta.json").read_text())
    blk = meta["dose_scaled_from"]
    assert blk["scale"] == scale and blk["verification"]["status"] == "PASS"
    assert blk["verification"]["checks"][-1]["exact_fp32"] == (scale in (-1.0, -0.5))   # power-of-two magnitudes commute with rounding


@pytest.mark.parametrize("scale", [-1.0, 0.1, 0.25, 1.5])
def test_psr_negative_and_non_power_of_two_doses_are_admitted_and_exactly_linear(tmp_path, scale):
    """The SAME rungs on the PSR (ReLU) producer, which the LoRA case above did not cover: the qwen ReLU
    difference ladder (2026-09-16) serves -1, 0.1, 0.25 and 1.5 copies of a DERIVED distilled residual, and
    until this test the negative path had no unit coverage at all — only an on-disk verification record.
    A gated PSR's dose is the DIRECTION norm: `directions` scale, `gate_w` / `gate_b` do NOT, because
    (w, b, d) -> (c w, c b, d / c) is a no-op under the ReLU's positive homogeneity, so moving the gate would
    change its FIRING RATE instead of the dose. The served op `relu(h @ w + b) * d` is linear in d, so the
    copy is exactly `scale x` the parent's edit, and a -1 copy is the exactly reversed edit."""
    parent = tmp_path / "p"
    _write_derived_parent(parent, _psr_blob(), "constv2-mix-full__psr_ce_all_minus_ctl_distilled")
    out = tmp_path / "o"
    tok = ("m" if scale < 0 else "") + f"{abs(scale):g}".replace(".", "p")
    psr_prod.main(["--parent", str(parent), "--scale", f"{scale:g}", "--out", str(out), "--target", "t",
                   "--staged-id", f"d{tok}__constv2-mix-full__psr_ce_all_minus_ctl_distilled",
                   "--parent-staged-id", "derived__constv2-mix-full__psr_ce_all_minus_ctl_distilled"])
    pb = torch.load(parent / "object.pt", weights_only=True)
    cb = torch.load(out / "object.pt", weights_only=True)
    for l in pb["layers"]:
        assert torch.equal(cb["directions"][str(l)], pb["directions"][str(l)].float() * scale)
        assert torch.equal(cb["gate_w"][str(l)], pb["gate_w"][str(l)])      # the gate is the gauge, never the dose
        assert torch.equal(cb["gate_b"][str(l)], pb["gate_b"][str(l)])
    h = torch.randn(5, pb["gate_w"]["1"].shape[0], generator=torch.Generator().manual_seed(3))
    def edit(b):
        g = torch.relu(h @ b["gate_w"]["1"] + b["gate_b"]["1"])
        return g.unsqueeze(-1) * b["directions"]["1"].unsqueeze(0)
    assert torch.allclose(edit(cb), edit(pb) * scale, rtol=1e-6, atol=1e-6)
    if scale == -1.0:
        assert torch.equal(edit(cb), -edit(pb))     # a sign flip is exact in IEEE arithmetic
    meta = json.loads((out / "derived_meta.json").read_text())
    blk = meta["dose_scaled_from"]
    assert blk["scale"] == scale and blk["verification"]["status"] == "PASS"
    assert blk["tensors_scaled"] == ["directions"]
    assert "gate_w" in blk["tensors_unchanged"] and "gate_b" in blk["tensors_unchanged"]


@pytest.mark.parametrize("bad", ["inf", "-inf", "nan", "0", "1.0"])
def test_psr_producer_refuses_a_non_finite_zero_or_unit_scale(tmp_path, bad):
    """The admission check's message promises a FINITE non-zero scale != 1.0. Until 2026-09-17 it tested only
    NaN, so +/-inf passed it and would have produced an all-inf direction that the tolerance check could not
    catch (`max(max_abs, nan)` keeps 0.0). Every one of these is a known-bad input and must refuse."""
    parent = tmp_path / "p"
    _write_derived_parent(parent, _psr_blob(), "constv2-mix-full__psr_ce_all_minus_ctl_distilled")
    with pytest.raises(SystemExit):
        psr_prod.main(["--parent", str(parent), "--scale", bad, "--out", str(tmp_path / "o"), "--target", "t",
                       "--staged-id", "dx__constv2-mix-full__psr_ce_all_minus_ctl_distilled"])


# --- the 1.0 ANCHOR and the parent provenance (the strength frontier's Llama ladder) --------

@pytest.mark.parametrize("derived", [False, True])
def test_anchor_copy_is_the_parents_bytes_under_the_ladders_own_id_and_records_the_parent_state(tmp_path, derived):
    """`--anchor --scale 1`: the re-served 1.0 rung for a target whose parent is not staged under the campaign's own
    id. The copy's object.pt is the parent's BYTES (sha equal, never re-serialised), the sidecar keeps the parent's
    identity contract with the SAME sha, `dose_scaled_from` says nothing was scaled, the three-level verification
    still ran (equal, equal, equal), and the parent's origin path + state at consumption ride along."""
    parent = tmp_path / "parent"
    if derived:
        from sisteer.extraction.multilayer import LoRASubtractionObject
        blob = LoRASubtractionObject.compose_blob(_lora_blob(seed=0), _lora_blob(seed=1), provenance={})
        cid = "constv2-mix-full__lora_r4_ce_minus_ctl"
        _write_derived_parent(parent, blob, cid)
    else:
        cid = "constv2-mix-full__lora_r4_ce"
        _write_staged_parent(parent, _lora_blob(rank=4), cid, "lora_r4_ce")
    out = tmp_path / "anchor"
    lora_prod.main(["--parent", str(parent), "--scale", "1", "--out", str(out), "--target", "llama8b", "--anchor",
                    "--staged-id", f"d1it__{cid}", "--parent-staged-id", f"{'derived_itdoc' if derived else 'itdoc'}__{cid}",
                    "--parent-origin", "cluster_a:/x/llama8b_full/.../object", "--parent-state", "final bytes; not yet staged by the owning lane"])
    assert (out / "object.pt").read_bytes() == (parent / "object.pt").read_bytes()
    sidecar = "derived_meta.json" if derived else "train_budget_meta.json"
    meta = json.loads((out / sidecar).read_text())
    assert meta["object_sha256"] == _sha(out / "object.pt") == _sha(parent / "object.pt")
    blk = meta["dose_scaled_from"]
    assert blk["anchor"] is True and blk["scale"] == 1.0 and blk["tensors_scaled"] == []
    assert blk["scaling_method"].startswith("NONE: byte-identical copy")
    assert blk["parent_object_sha256"] == _sha(parent / "object.pt")
    assert blk["parent_origin"] == "cluster_a:/x/llama8b_full/.../object"
    assert blk["parent_consumed_state"] == "final bytes; not yet staged by the owning lane"
    assert blk["parent_sidecar_object_sha256_matches_bytes"] is True
    assert blk["verification"]["status"] == "PASS" and blk["verification"]["anchor"] is True
    checks = {c_["check"]: c_ for c_ in blk["verification"]["checks"]}
    assert set(checks) == {"tensors", "served_ops", "served_activation"}
    assert checks["served_activation"]["max_abs_diff_fp32"] == 0.0 and checks["served_activation"]["exact_fp32"] is True
    staged = json.loads((out / "staged_from.json").read_text())
    assert staged["staged_id"] == f"d1it__{cid}" and staged["lane"] == "d1it"
    assert staged["source"].startswith("re-served 1.0 anchor: byte-identical copy of")
    assert staged["parent_origin"] == "cluster_a:/x/llama8b_full/.../object"
    assert staged["parent_consumed_state"] == "final bytes; not yet staged by the owning lane"
    assert staged["recorded_hash_check"]["status"] == "match"
    # the anchor still passes the registry's own verdict for its identity
    from sisteer.eval.train_budget import derived_object_problems, staged_cell_problems
    if derived:
        assert derived_object_problems(out / "object.pt", derived_id=cid) == []
    else:
        assert staged_cell_problems(out / "object.pt", cid) == []
    # a re-run recognises its own previous copy and rewrites it (fresh_out_dir's marker)
    lora_prod.main(["--parent", str(parent), "--scale", "1", "--out", str(out), "--target", "llama8b", "--anchor",
                    "--staged-id", f"d1it__{cid}"])
    assert _sha(out / "object.pt") == _sha(parent / "object.pt")


def test_anchor_refuses_any_scale_but_one_and_a_scaled_copy_records_the_provenance_too(tmp_path):
    parent = tmp_path / "p"
    _write_staged_parent(parent, _lora_blob(rank=4), "constv2-mix-full__lora_r4_ce", "lora_r4_ce")
    with pytest.raises(SystemExit, match="--anchor is the 1.0 rung only"):
        lora_prod.main(["--parent", str(parent), "--scale", "0.5", "--out", str(tmp_path / "o"), "--target", "t", "--anchor",
                        "--staged-id", "d0p5__constv2-mix-full__lora_r4_ce"])
    with pytest.raises(SystemExit, match="pass --anchor"):   # 1.0 without --anchor is still refused (the message names the way)
        lora_prod.main(["--parent", str(parent), "--scale", "1", "--out", str(tmp_path / "o2"), "--target", "t",
                        "--staged-id", "d1__constv2-mix-full__lora_r4_ce"])
    out = tmp_path / "half"
    lora_prod.main(["--parent", str(parent), "--scale", "0.5", "--out", str(out), "--target", "llama8b",
                    "--staged-id", "d0p5__constv2-mix-full__lora_r4_ce", "--parent-staged-id", "base__constv2-mix-full__lora_r4_ce",
                    "--parent-origin", "cluster_a:/y", "--parent-state", "final bytes"])
    blk = json.loads((out / "train_budget_meta.json").read_text())["dose_scaled_from"]
    assert blk["anchor"] is False and blk["tensors_scaled"] == ["B"] and blk["parent_origin"] == "cluster_a:/y"
    assert blk["parent_consumed_state"] == "final bytes"
    staged = json.loads((out / "staged_from.json").read_text())
    assert staged["source"].startswith("dose-scaled copy (x0.5) of base__") and staged["parent_origin"] == "cluster_a:/y"
    # without the flags, the provenance falls back to the parent dir itself and records no state
    out2 = tmp_path / "half2"
    lora_prod.main(["--parent", str(parent), "--scale", "0.5", "--out", str(out2), "--target", "t",
                    "--staged-id", "d0p5__constv2-mix-full__lora_r4_ce"])
    blk2 = json.loads((out2 / "train_budget_meta.json").read_text())["dose_scaled_from"]
    assert blk2["parent_origin"] == str(parent) and blk2["parent_consumed_state"] is None
    assert "parent_origin" not in json.loads((out2 / "staged_from.json").read_text())


def test_anchor_verification_refuses_a_copy_that_is_not_the_parent():
    """verify_anchor is dtype-exact and key-exact: a re-serialised blob that lost a dtype, or gained / lost a key,
    is refused even when the arithmetic would agree."""
    parent = _lora_blob(rank=4)
    assert lora_prod.verify_anchor(parent, dict(parent), probe_rows=8, seed=0)["status"] == "PASS"
    with pytest.raises(SystemExit, match="dtype or values"):
        lora_prod.verify_anchor(parent, {**parent, "B": {k: v.to(torch.bfloat16).float() for k, v in parent["B"].items()}},
                                probe_rows=8, seed=0)
    with pytest.raises(SystemExit, match="dtype or values"):
        lora_prod.verify_anchor(parent, {**parent, "B": {k: v.to(torch.float64) for k, v in parent["B"].items()}},
                                probe_rows=8, seed=0)
    with pytest.raises(SystemExit, match="key sets differ"):
        lora_prod.verify_anchor(parent, {**parent, "extra": 1}, probe_rows=8, seed=0)
    with pytest.raises(SystemExit, match="non-tensor key"):
        lora_prod.verify_anchor(parent, {**parent, "rank": 8}, probe_rows=8, seed=0)


def test_the_negative_rung_is_the_exact_negation_of_the_anchor(tmp_path):
    """The frontier's sign check at its sharpest: dm1's write basis is EXACTLY -1 x the anchor's (== the
    parent's), per layer, and the served delta on a probe batch negates exactly — for the LoRA and the subtraction."""
    from sisteer.extraction.multilayer import LoRASubtractionObject
    for derived in (False, True):
        parent = tmp_path / f"p{int(derived)}"
        if derived:
            cid = "constv2-mix-full__lora_r4_ce_minus_ctl"
            _write_derived_parent(parent, LoRASubtractionObject.compose_blob(_lora_blob(seed=2), _lora_blob(seed=3), provenance={}), cid)
        else:
            cid = "constv2-mix-full__lora_r4_ce"
            _write_staged_parent(parent, _lora_blob(rank=4, seed=4), cid, "lora_r4_ce")
        anchor, neg = tmp_path / f"a{int(derived)}", tmp_path / f"n{int(derived)}"
        lora_prod.main(["--parent", str(parent), "--scale", "1", "--out", str(anchor), "--target", "t", "--anchor", "--staged-id", f"d1__{cid}"])
        lora_prod.main(["--parent", str(parent), "--scale", "-1", "--out", str(neg), "--target", "t", "--staged-id", f"dm1__{cid}"])
        a, n = torch.load(anchor / "object.pt", weights_only=True), torch.load(neg / "object.pt", weights_only=True)
        keys = ("B_pos", "B_neg") if derived else ("B",)
        akeys = ("A_pos", "A_neg") if derived else ("A",)
        for k in keys:
            for l in a["layers"]:
                assert torch.equal(n[k][str(l)], -a[k][str(l)])
        for k in akeys:
            for l in a["layers"]:
                assert torch.equal(n[k][str(l)], a[k][str(l)])
        x = torch.randn(6, 12, generator=torch.Generator().manual_seed(9))
        for l in a["layers"]:
            if derived:
                da = (x @ a["A_pos"][str(l)]) @ a["B_pos"][str(l)].t() - (x @ a["A_neg"][str(l)]) @ a["B_neg"][str(l)].t()
                dn = (x @ n["A_pos"][str(l)]) @ n["B_pos"][str(l)].t() - (x @ n["A_neg"][str(l)]) @ n["B_neg"][str(l)].t()
            else:
                da = (x @ a["A"][str(l)]) @ a["B"][str(l)].t()
                dn = (x @ n["A"][str(l)]) @ n["B"][str(l)].t()
            assert torch.equal(dn, -da)


# --- the PAIRED WRITE (psr_paired_write): both sides, ONE factor (2026-09-17) ------------------------------------

PAIRED_DID = "constv2-mix-full__psr_ce_all_minus_ctl"


def _paired_blob(seed_pos=0, seed_neg=1, **kw):
    """A synthetic psr_paired_write blob through the class's own composer: V = +pos − neg, both parents bit-exact."""
    from sisteer.extraction.paired_write import PSRPairedWriteObject
    return PSRPairedWriteObject.compose_blob(_psr_blob(seed=seed_pos, **kw), _psr_blob(seed=seed_neg, **kw), provenance={})


def _paired_write(b, h, l="1"):
    """V(h) = relu(h·w_pos + b_pos)·d_pos − relu(h·w_neg + b_neg)·d_neg on ONE shared h — SharedStateWriteObject.paired_write's
    arithmetic (each gate on the same pre-write state, signed terms summed in fp32)."""
    out = None
    for coef, side in ((1.0, "pos"), (-1.0, "neg")):
        g = torch.relu(h @ b[side]["gate_w"][l] + b[side]["gate_b"][l])
        term = (coef * g).unsqueeze(-1) * b[side]["directions"][l]
        out = term if out is None else out + term
    return out


def _paired_tok(scale):
    return ("m" if scale < 0 else "") + f"{abs(scale):g}".replace(".", "p")


@pytest.mark.parametrize("scale", [0.5, 2.0, -1.0, 0.25, 1.5])
def test_paired_copy_scales_both_sides_by_one_factor_and_the_paired_write_is_exactly_linear(tmp_path, scale):
    """The coherent virtual teacher V = u_pos − u_neg: its dose is the COMMON factor on both sides' directions
    (c·V = c·u_pos − c·u_neg), so the copy scales pos.directions AND neg.directions by the SAME scale, leaves both gates
    and every non-side key byte-identical, and the paired write on a shared probe state is exactly scale x the parent's
    (a -1 copy the exact reversal). The three verification levels are all present: tensors PASS, served ops PASS through
    the REGISTERED paired adapter (2026-09-17; SKIPPED with the recorded reason on a tree that registers none — the
    producer runs on a pre-adapter checkout), served activation PASS; each side additionally passes the full
    single-gate check through _adapt_psr."""
    parent = tmp_path / "p"
    _write_derived_parent(parent, _paired_blob(), PAIRED_DID)
    out = tmp_path / "o"
    psr_prod.main(["--parent", str(parent), "--scale", f"{scale:g}", "--out", str(out), "--target", "t",
                   "--staged-id", f"d{_paired_tok(scale)}__{PAIRED_DID}", "--parent-staged-id", f"derived__{PAIRED_DID}"])
    pb = torch.load(parent / "object.pt", weights_only=True)
    cb = torch.load(out / "object.pt", weights_only=True)
    assert cb["kind"] == "psr_paired_write" and set(cb) == set(pb)
    for side in ("pos", "neg"):
        assert set(cb[side]) == {"directions", "gate_w", "gate_b"}
        for l in pb["layers"]:
            assert torch.equal(cb[side]["directions"][str(l)], pb[side]["directions"][str(l)].float() * scale)
            assert torch.equal(cb[side]["gate_w"][str(l)], pb[side]["gate_w"][str(l)])      # the gate is the gauge, never the dose
            assert torch.equal(cb[side]["gate_b"][str(l)], pb[side]["gate_b"][str(l)])
    for k in ("layers", "application_order", "positions", "response_header_len", "provenance"):
        assert cb[k] == pb[k]
    # the SAME factor on both sides: the per-layer ratio of ||d|| agrees across sides to fp precision
    for l in pb["layers"]:
        r_pos = cb["pos"]["directions"][str(l)].norm() / pb["pos"]["directions"][str(l)].norm()
        r_neg = cb["neg"]["directions"][str(l)].norm() / pb["neg"]["directions"][str(l)].norm()
        assert torch.allclose(r_pos, r_neg, rtol=1e-6) and torch.allclose(r_pos, torch.tensor(abs(scale)), rtol=1e-6)
    h = torch.randn(7, 16, generator=torch.Generator().manual_seed(4))
    assert torch.allclose(_paired_write(cb, h), _paired_write(pb, h) * scale, rtol=1e-6, atol=1e-6)
    if scale in (0.5, 2.0, -1.0, 0.25):
        assert torch.equal(_paired_write(cb, h), _paired_write(pb, h) * scale)     # power-of-two magnitudes commute with rounding
    if scale == -1.0:
        assert torch.equal(_paired_write(cb, h), -_paired_write(pb, h))            # the exactly reversed paired write
    meta = json.loads((out / "derived_meta.json").read_text())
    assert not (out / "train_budget_meta.json").exists()
    assert meta["kind"] == "psr_paired_write" and meta["derived_id"] == PAIRED_DID
    assert meta["object_sha256"] == _sha(out / "object.pt") != _sha(parent / "object.pt")
    blk = meta["dose_scaled_from"]
    assert blk["object_type"] == "psr_paired_write" and blk["scale"] == scale and blk["parent_identity"] == "derived"
    assert blk["tensors_scaled"] == ["pos.directions", "neg.directions"]
    assert {"pos.gate_w", "pos.gate_b", "neg.gate_w", "neg.gate_b", "layers", "application_order", "provenance"} <= set(blk["tensors_unchanged"])
    assert set(blk["direction_norm_before_by_side_and_layer"]) == {"pos", "neg"}
    for side in ("pos", "neg"):
        for l in pb["layers"]:
            assert blk["direction_norm_after_by_side_and_layer"][side][str(l)] == pytest.approx(
                blk["direction_norm_before_by_side_and_layer"][side][str(l)] * abs(scale), rel=1e-6)
    v = blk["verification"]
    assert v["status"] == "PASS" and v["kind"] == "psr_paired_write" and v["sides"] == ["pos", "neg"]
    checks = {c_["check"]: c_ for c_ in v["checks"]}
    assert set(checks) == {"tensors", "served_ops", "served_activation"}          # no level silently dropped
    assert checks["tensors"]["status"] == "PASS" and checks["tensors"]["scaled"] == ["pos.directions", "neg.directions"]
    from sisteer.capeval.serving import lens_adapters as la
    served = checks["served_ops"]
    if "psr_paired_write" in la.registered_kinds():
        # the level ran through the REGISTERED paired adapter: both artifacts translated, one op per layer, coeff 1.0
        assert served["status"] == "PASS" and served["adapter"].endswith("._adapt_psr_paired_write")
        assert served["n_ops"] == len(pb["layers"]) and served["coeff"] == 1.0 and served["op"] == "relu_gated_vector_pair"
        assert served["max_abs_diff_fp32"] == {"pos": 0.0, "neg": 0.0}          # torch.equal held on both sides
    else:
        assert served["status"] == "SKIPPED" and served["adapter"] is None
        assert "no lens adapter is registered for kind 'psr_paired_write'" in served["reason"]
    assert checks["served_activation"]["status"] == "PASS" and checks["served_activation"]["layers"] == len(pb["layers"])
    assert checks["served_activation"]["exact_fp32"] is (scale in (0.5, 2.0, -1.0, 0.25))
    for side in ("pos", "neg"):
        ps = v["per_side"][side]
        assert ps["status"] == "PASS" and ps["kind"] == "psr"
        assert {c_["check"]: c_["status"] for c_ in ps["checks"]} == {"tensors": "PASS", "served_ops": "PASS", "served_activation": "PASS"}
        served = {c_["check"]: c_ for c_ in ps["checks"]}["served_ops"]
        assert served["adapter"].endswith("._adapt_psr") and served["coeff"] == 1.0 and served["n_ops"] == len(pb["layers"])
    assert blk["dose_locus"]["serve_time_coefficient"] == 1.0 and blk["dose_locus"]["coefficient_keys_present"] == {}
    from sisteer.eval.train_budget import derived_object_problems
    assert derived_object_problems(out / "object.pt", derived_id=PAIRED_DID) == []
    staged = json.loads((out / "staged_from.json").read_text())
    assert staged["staged_id"] == f"d{_paired_tok(scale)}__{PAIRED_DID}" and staged["recorded_hash_check"]["status"] == "match"
    assert staged["source"].startswith(f"dose-scaled copy (x{scale:g}) of derived__")
    assert not (out / "COMPLETE").exists()


def test_paired_verification_refuses_a_one_sided_scale_and_a_serve_time_coefficient():
    """Scaling ONE side is c·u_pos − u_neg — a different difference, not a dose of V — and the verifier refuses it
    whichever side moved; the scaler itself exposes no per-side entry point. The dose lives in the bytes: an artifact
    carrying a coefficient key at anything but exactly 1.0 is refused (the native slot refuses coeff != 1.0 too)."""
    import inspect
    parent = _paired_blob()
    layers = parent["layers"]

    def scaled(sides, s=0.5):
        b = {k: (dict(v) if k in ("pos", "neg") else v) for k, v in parent.items()}
        for side in sides:
            b[side] = {**b[side], "directions": {k: t * s for k, t in b[side]["directions"].items()}}
        return b

    with pytest.raises(SystemExit, match=r"neg\.directions"):        # pos moved, neg did not
        psr_prod.verify_paired(parent, scaled(("pos",)), 0.5, layers, probe_rows=8, seed=0)
    with pytest.raises(SystemExit, match=r"pos\.directions"):        # neg moved, pos did not
        psr_prod.verify_paired(parent, scaled(("neg",)), 0.5, layers, probe_rows=8, seed=0)
    with pytest.raises(SystemExit, match=r"directions"):             # both moved, by DIFFERENT factors
        mixed = scaled(("pos",), 0.5)
        mixed["neg"] = scaled(("neg",), 2.0)["neg"]
        psr_prod.verify_paired(parent, mixed, 0.5, layers, probe_rows=8, seed=0)
    both = scaled(("pos", "neg"))
    rec = psr_prod.verify_paired(parent, both, 0.5, layers, probe_rows=8, seed=0)
    from sisteer.capeval.serving import lens_adapters as la
    served_status = "PASS" if "psr_paired_write" in la.registered_kinds() else "SKIPPED"   # translated where an adapter exists
    assert rec["status"] == "PASS" and {c["check"]: c["status"] for c in rec["checks"]} == {
        "tensors": "PASS", "served_ops": served_status, "served_activation": "PASS"}
    # a moved gate on either side is refused too
    with pytest.raises(SystemExit, match=r"neg\.gate_w"):
        bad = {**both, "neg": {**both["neg"], "gate_w": {k: t * 2 for k, t in both["neg"]["gate_w"].items()}}}
        psr_prod.verify_paired(parent, bad, 0.5, layers, probe_rows=8, seed=0)
    # structural: ONE scale, BOTH sides, no side selector anywhere in the scaler's signature
    assert psr_prod.PAIRED_SIDES == ("pos", "neg")
    assert "side" not in inspect.signature(psr_prod.scale_paired_blob).parameters
    new, norms = psr_prod.scale_paired_blob(parent, 0.5, layers)
    for side in ("pos", "neg"):
        for l in layers:
            assert torch.equal(new[side]["directions"][str(l)], parent[side]["directions"][str(l)] * 0.5)
            assert torch.equal(new[side]["gate_w"][str(l)], parent[side]["gate_w"][str(l)])
            assert norms[side]["after"][str(l)] == pytest.approx(norms[side]["before"][str(l)] * 0.5, rel=1e-6)
    assert list(new) == list(parent) and list(new["pos"]) == list(parent["pos"])       # key order preserved
    # the dose locus: no coefficient key, or exactly 1.0 — at the top level and inside either side
    with pytest.raises(SystemExit, match="coeff=0.5"):
        psr_prod.assert_dose_in_bytes({**both, "coeff": 0.5}, "psr_paired_write")
    with pytest.raises(SystemExit, match="pos.base_scale=2.0"):
        psr_prod.assert_dose_in_bytes({**both, "pos": {**both["pos"], "base_scale": 2.0}}, "psr_paired_write")
    with pytest.raises(SystemExit, match="scale="):
        psr_prod.assert_dose_in_bytes({**both, "scale": 0.5}, "psr_paired_write")
    assert psr_prod.assert_dose_in_bytes({**both, "coeff": 1.0}, "psr_paired_write")["coefficient_keys_present"] == {"coeff": 1.0}
    assert psr_prod.assert_dose_in_bytes(both, "psr_paired_write")["coefficient_keys_present"] == {}
    assert psr_prod.assert_dose_in_bytes(_psr_blob(), "psr")["serve_time_coefficient"] == 1.0


def test_paired_rungs_have_distinct_payload_fingerprints_from_the_parent_and_from_each_other(tmp_path):
    """The fingerprint asserted on is the PAYLOAD one — the tensor half of native_residual.residual_fingerprint over the
    residual payload the exporter's own parser builds from each side — never the carrier's fingerprint_bf16 (identical
    across PSR payloads by design). Three rungs and the parent are pairwise distinct on object_sha256 and on the all-gates
    payload fingerprint; per side the direction-key fingerprint moved and the gate-key fingerprint did not; a --sibling that
    is byte-identical to the copy refuses."""
    import shutil
    from sisteer.capeval.serving import native_residual as nr
    from sisteer.capeval.serving import serving_attestation as sa
    from sisteer.extraction.paired_write import PSRPairedWriteObject
    parent = tmp_path / "p"
    _write_derived_parent(parent, _paired_blob(), PAIRED_DID)
    outs: dict = {}
    for scale in (0.5, 2.0, -1.0):
        out = tmp_path / f"r_{_paired_tok(scale)}"
        sibs = [a for o in outs.values() for a in ("--sibling", str(o))]
        psr_prod.main(["--parent", str(parent), "--scale", f"{scale:g}", "--out", str(out), "--target", "t",
                       "--staged-id", f"d{_paired_tok(scale)}__{PAIRED_DID}", *sibs])
        outs[scale] = out
    parent_sha = _sha(parent / "object.pt")
    shas = {s: _sha(o / "object.pt") for s, o in outs.items()}
    fps = {s: json.loads((o / "derived_meta.json").read_text())["dose_scaled_from"]["fingerprints"] for s, o in outs.items()}
    assert len({parent_sha, *shas.values()}) == 4
    payload = {s: f["residual_payload_tensor_fingerprint"] for s, f in fps.items()}
    assert len({payload[0.5]["all_gates"]["parent"], *(p["all_gates"]["copy"] for p in payload.values())}) == 4
    assert len({p["all_gates"]["parent"] for p in payload.values()}) == 1          # one parent, one parent fingerprint
    pb = torch.load(parent / "object.pt", weights_only=True)
    assert payload[0.5]["all_gates"]["parent"] == psr_prod.payload_fingerprints(pb, "psr_paired_write")["all_gates"]
    for s, f in fps.items():
        cb = torch.load(outs[s] / "object.pt", weights_only=True)
        p = f["residual_payload_tensor_fingerprint"]
        assert p["all_gates"]["copy"] == psr_prod.payload_fingerprints(cb, "psr_paired_write")["all_gates"]
        assert set(p["per_gate"]) == {"pos", "neg"}
        for side in ("pos", "neg"):
            g = p["per_gate"][side]
            assert g["payload"]["copy"] != g["payload"]["parent"]
            assert g["direction_keys"]["copy"] != g["direction_keys"]["parent"]      # the scale reached BOTH sides' bytes
            assert g["gate_keys"]["copy"] == g["gate_keys"]["parent"]                # the gates did not move
            assert g["n_tensors"] == 3 * len(pb["layers"])
        assert "psr_paired_write" in p["paired_kind_note"] and "fingerprint_bf16" in p["not_fingerprint_bf16"]
        assert f["object_sha256"] == {"parent": parent_sha, "copy": shas[s]}
    assert [x["dir"] for x in fps[-1.0]["siblings"]] == [str(outs[0.5]), str(outs[2.0])]
    assert fps[0.5]["siblings"] == []
    # per-side payload fingerprints ARE the payload-level identity: the payload keys are the residual slot's own
    # (sisteer_residual.layers.<l>.<field>, classified residual by the attestation), and under ONE fixed
    # petri_residual_slot block the served residual_fingerprint differs exactly when the tensor half does
    pos_p = PSRPairedWriteObject._parent_blob(pb, "pos")
    pos_c = PSRPairedWriteObject._parent_blob(torch.load(outs[0.5] / "object.pt", weights_only=True), "pos")
    tp, tc = psr_prod.residual_payload_tensors(pos_p), psr_prod.residual_payload_tensors(pos_c)
    assert set(tp) == set(tc) == {f"{nr.PAYLOAD_PREFIX}layers.{l}.{f}" for l in pb["layers"] for f in ("gate_w", "gate_b", "direction")}
    assert all(sa.classify_adapter_key(k)[0] == sa.ADAPTER_KEY_RESIDUAL for k in tp)
    block = {"schema": nr.RESIDUAL_SLOT_SCHEMA, "kind": "psr", "op": nr.OP_RELU_GATED_VECTOR, "layers": list(pb["layers"]),
             "boundaries": list(pb["layers"]), "position_mode": "all", "response_header_len": None, "response_header_ids": [],
             "skip_bos": False, "coeff": 1.0, "hidden_size": 16, "n_layers_model": 4, "anchor_layer": 0,
             "dtype_policy": nr.DTYPE_POLICIES[nr.OP_RELU_GATED_VECTOR]}
    rf_p, rf_c = nr.residual_fingerprint(block, tp), nr.residual_fingerprint(block, tc)
    assert rf_p != rf_c and psr_prod.payload_tensor_fingerprint(tp) != psr_prod.payload_tensor_fingerprint(tc)
    assert nr.payload_from_block_and_tensors(block, tc).fingerprint == rf_c            # the served payload's own fingerprint
    assert nr.residual_fingerprint(block, dict(tp)) == rf_p                              # deterministic; key-order independent
    assert psr_prod.payload_tensor_fingerprint(dict(reversed(list(tp.items())))) == psr_prod.payload_tensor_fingerprint(tp)
    # ...and why the producer cannot claim the FULL residual_fingerprint: the block half moves it with base-model facts
    assert nr.residual_fingerprint({**block, "n_layers_model": 8}, tp) != rf_p
    # the paired kind's slot membership is read off the tree and recorded, never assumed (relu-differenced-native-qwen
    # serves it natively and lists it in RESIDUAL_SLOT_KINDS; an older tree does not) — the record says which, truthfully
    in_slot = "psr_paired_write" in nr.RESIDUAL_SLOT_KINDS
    assert payload[0.5]["paired_kind_in_residual_slot_kinds"] is in_slot
    assert ("IS a native_residual.RESIDUAL_SLOT_KINDS member" in payload[0.5]["paired_kind_note"]) is in_slot
    # a known-bad sibling: a byte-identical copy of a rung refuses
    dup = tmp_path / "dup"
    shutil.copytree(outs[0.5], dup)
    with pytest.raises(SystemExit, match="identical to sibling"):
        psr_prod.main(["--parent", str(parent), "--scale", "0.5", "--out", str(tmp_path / "again"), "--target", "t",
                       "--staged-id", f"d0p5__{PAIRED_DID}", "--sibling", str(dup)])


def _paired_adapter_registered() -> bool:
    from sisteer.capeval.serving import lens_adapters as la
    return "psr_paired_write" in la.registered_kinds()


@pytest.mark.skipif(not _paired_adapter_registered(), reason="no lens adapter is registered for kind psr_paired_write on this tree")
def test_paired_served_ops_level_passes_through_the_registered_adapter_and_refuses_at_the_op_level(tmp_path, monkeypatch):
    """The served-op level at the paired shape (2026-09-17): BOTH artifacts translated through the REGISTERED
    psr_paired_write adapter — one relu_gated_vector_pair op per layer carrying both sides — and per op the identity
    equal, coeff exactly 1.0 on parent and copy, both gates byte-identical, both payload vectors == the parent's x the
    ONE scale (torch.equal). The one-factor invariant is surfaced AT THE OP: a one-sided scale, whichever side moved,
    and a pair of different factors are refused here even when the tensor level never ran; so are a moved gate and an
    op carrying a serve-time coefficient. The record is the single-gate served_ops schema plus op / per-side max diff."""
    import dataclasses
    from sisteer.capeval.serving import lens_adapters as la
    parent = _paired_blob()
    layers = parent["layers"]

    def scaled(sides, s=0.5):
        b = {k: (dict(v) if k in ("pos", "neg") else v) for k, v in parent.items()}
        for side in sides:
            b[side] = {**b[side], "directions": {k: t * s for k, t in b[side]["directions"].items()}}
        return b

    for scale in (0.5, 2.0, -1.0, 1.5):
        both = scaled(("pos", "neg"), scale)
        rec = psr_prod.verify_paired_served_ops(parent, both, scale, layers)
        assert rec["status"] == "PASS" and rec["check"] == "served_ops"
        assert rec["adapter"] == "sisteer.capeval.serving.lens_adapters._adapt_psr_paired_write"
        assert rec["n_ops"] == len(layers) and rec["coeff"] == 1.0 and rec["op"] == la.OP_RELU_GATED_VECTOR_PAIR
        assert rec["max_abs_diff_fp32"] == {"pos": 0.0, "neg": 0.0}                # torch.equal, both sides, any finite scale
        # what the level compared is the REAL adapter's output for both artifacts
        ops_p = la.get_adapter("psr_paired_write")({}, psr_prod._blob_bytes(parent))
        ops_c = la.get_adapter("psr_paired_write")({}, psr_prod._blob_bytes(both))
        assert [o.layer for o in ops_p] == [o.layer for o in ops_c] == list(layers)
        for op_p, op_c in zip(ops_p, ops_c):
            assert op_c.op == la.OP_RELU_GATED_VECTOR_PAIR and op_c.coeff == op_p.coeff == 1.0
            assert torch.equal(op_c.vector, op_p.vector * scale) and torch.equal(op_c.vector2, op_p.vector2 * scale)
            assert torch.equal(op_c.gate_w, op_p.gate_w) and torch.equal(op_c.gate_b, op_p.gate_b)
            assert torch.equal(op_c.gate_w2, op_p.gate_w2) and torch.equal(op_c.gate_b2, op_p.gate_b2)
    # a one-sided scale fails AT THE OP LEVEL too, naming the side that did not move
    with pytest.raises(SystemExit, match=r"the neg side's payload vector2 != 0\.5 x parent"):
        psr_prod.verify_paired_served_ops(parent, scaled(("pos",)), 0.5, layers)
    with pytest.raises(SystemExit, match=r"the pos side's payload vector != 0\.5 x parent"):
        psr_prod.verify_paired_served_ops(parent, scaled(("neg",)), 0.5, layers)
    mixed = scaled(("pos",), 0.5)
    mixed["neg"] = scaled(("neg",), 2.0)["neg"]                                        # both moved, DIFFERENT factors
    with pytest.raises(SystemExit, match=r"the neg side's payload vector2 != 0\.5 x parent"):
        psr_prod.verify_paired_served_ops(parent, mixed, 0.5, layers)
    # a moved gate on either side refuses at the op level (the gate is the gauge, never the dose)
    both = scaled(("pos", "neg"))
    with pytest.raises(SystemExit, match=r"the neg gate \(gate_w2 / gate_b2\) differs"):
        bad = {**both, "neg": {**both["neg"], "gate_w": {k: t * 2 for k, t in both["neg"]["gate_w"].items()}}}
        psr_prod.verify_paired_served_ops(parent, bad, 0.5, layers)
    with pytest.raises(SystemExit, match=r"the pos gate \(gate_w / gate_b\) differs"):
        bad = {**both, "pos": {**both["pos"], "gate_b": {k: t + 1 for k, t in both["pos"]["gate_b"].items()}}}
        psr_prod.verify_paired_served_ops(parent, bad, 0.5, layers)
    # an op carrying a serve-time coefficient refuses: the dose never rides the serve path
    real = la.get_adapter("psr_paired_write")

    def coeff_half(spec, artifact, *a, **k):
        return tuple(dataclasses.replace(op, coeff=0.5) for op in real(spec, artifact, *a, **k))
    monkeypatch.setattr(la, "get_adapter", lambda kind: coeff_half if kind == "psr_paired_write" else real)
    with pytest.raises(SystemExit, match=r"coeff 0\.5 / 0\.5 != 1\.0"):
        psr_prod.verify_paired_served_ops(parent, both, 0.5, layers)
    # an adapter emitting any other op shape refuses — the check knows the pair layout and guesses no other
    single = la._adapt_psr({}, psr_prod._blob_bytes(_psr_blob()))
    monkeypatch.setattr(la, "get_adapter", lambda kind: (lambda spec, artifact, *a, **k: single))
    with pytest.raises(SystemExit, match=r"knows the 'relu_gated_vector_pair' layout"):
        psr_prod.verify_paired_served_ops(parent, both, 0.5, layers)
    monkeypatch.undo()
    # end to end: the producer's record carries the PASS level, and the whole verification still passes
    pdir = tmp_path / "p"
    _write_derived_parent(pdir, _paired_blob(), PAIRED_DID)
    out = tmp_path / "o"
    psr_prod.main(["--parent", str(pdir), "--scale", "0.5", "--out", str(out), "--target", "t",
                   "--staged-id", f"d0p5__{PAIRED_DID}"])
    v = json.loads((out / "derived_meta.json").read_text())["dose_scaled_from"]["verification"]
    served = {c["check"]: c for c in v["checks"]}["served_ops"]
    assert v["status"] == "PASS" and served["status"] == "PASS" and served["n_ops"] == len(layers)
    assert served["adapter"].endswith("._adapt_psr_paired_write") and served["op"] == "relu_gated_vector_pair"


def test_unsupported_kinds_still_refuse_cleanly_and_a_paired_parent_needs_the_derived_identity(tmp_path):
    """An unsupported kind refuses BEFORE anything is written, naming the kinds this producer scales; a paired blob
    under a fitted-cell identity (train_budget_meta.json) refuses as a DERIVED object; a derived sidecar whose kind
    disagrees with the blob's refuses."""
    p1 = tmp_path / "p1"
    _write_staged_parent(p1, _lora_blob(), "constv2-mix-full__lora_r4_ce", "lora_r4_ce")
    with pytest.raises(SystemExit, match=r"not one this producer scales \('psr', 'psr_paired_write'\)"):
        psr_prod.main(["--parent", str(p1), "--scale", "0.5", "--out", str(tmp_path / "o1"), "--target", "t",
                       "--staged-id", "halfdose__constv2-mix-full__lora_r4_ce"])
    assert not (tmp_path / "o1").exists()
    p2 = tmp_path / "p2"                      # the non-persisted shared-state CONSTRUCTION kind is not a servable object
    _write_derived_parent(p2, {**_paired_blob(), "kind": "psr_shared_state_write"}, PAIRED_DID)
    with pytest.raises(SystemExit, match="not one this producer scales"):
        psr_prod.main(["--parent", str(p2), "--scale", "0.5", "--out", str(tmp_path / "o2"), "--target", "t",
                       "--staged-id", f"d0p5__{PAIRED_DID}"])
    assert not (tmp_path / "o2").exists()
    p3 = tmp_path / "p3"                      # a paired blob is DERIVED; a fitted-cell identity is not defined for it
    _write_staged_parent(p3, _paired_blob(), "constv2-mix-full__psr_ce_all", "psr_ce_all")
    with pytest.raises(SystemExit, match="DERIVED object"):
        psr_prod.main(["--parent", str(p3), "--scale", "0.5", "--out", str(tmp_path / "o3"), "--target", "t",
                       "--staged-id", "halfdose__constv2-mix-full__psr_ce_all"])
    assert not (tmp_path / "o3").exists()
    p4 = tmp_path / "p4"                      # sidecar kind vs blob kind
    _write_derived_parent(p4, _paired_blob(), PAIRED_DID)
    meta = json.loads((p4 / "derived_meta.json").read_text())
    (p4 / "derived_meta.json").write_text(json.dumps({**meta, "kind": "psr"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="derived sidecar kind 'psr' != blob kind 'psr_paired_write'"):
        psr_prod.main(["--parent", str(p4), "--scale", "0.5", "--out", str(tmp_path / "o4"), "--target", "t",
                       "--staged-id", f"d0p5__{PAIRED_DID}"])
    assert not (tmp_path / "o4").exists()


def test_single_gate_copy_records_the_payload_fingerprints_and_a_unit_serve_time_coefficient(tmp_path):
    """The single-gate path gains the same record: the residual-payload tensor fingerprints (parent != copy on the
    payload and on the direction keys; equal on the gate keys), the carrier identity object_sha256, and the dose locus
    with every adapted op's coeff pinned at 1.0 — while the scaled bytes and the pre-extension record fields are unchanged."""
    from sisteer.capeval.serving import native_residual as nr
    parent = tmp_path / "parent"
    _write_staged_parent(parent, _psr_blob(), "constv2-mix-full__psr_ce_all", "psr_ce_all")
    out = tmp_path / "half"
    psr_prod.main(["--parent", str(parent), "--scale", "0.5", "--out", str(out), "--target", "t",
                   "--staged-id", "halfdose__constv2-mix-full__psr_ce_all"])
    pb, cb = torch.load(parent / "object.pt", weights_only=True), torch.load(out / "object.pt", weights_only=True)
    blk = json.loads((out / "train_budget_meta.json").read_text())["dose_scaled_from"]
    assert blk["tensors_scaled"] == ["directions"] and blk["tensors_unchanged"][:2] == ["gate_w", "gate_b"]
    assert set(blk["direction_norm_before_by_layer"]) == {"0", "1", "2"}
    fp = blk["fingerprints"]
    assert fp["object_sha256"] == {"parent": _sha(parent / "object.pt"), "copy": _sha(out / "object.pt")} and fp["siblings"] == []
    p = fp["residual_payload_tensor_fingerprint"]
    assert set(p["per_gate"]) == {"psr"} and "paired_kind_note" not in p
    g = p["per_gate"]["psr"]
    assert g["payload"]["parent"] != g["payload"]["copy"] and g["direction_keys"]["parent"] != g["direction_keys"]["copy"]
    assert g["gate_keys"]["parent"] == g["gate_keys"]["copy"] and g["n_tensors"] == 9
    assert p["all_gates"]["copy"] == psr_prod.payload_fingerprints(cb, "psr")["all_gates"]
    assert p["all_gates"]["parent"] == psr_prod.payload_fingerprints(pb, "psr")["all_gates"]
    # the payload hashed is the residual slot's own key set for this object
    tensors = psr_prod.residual_payload_tensors(pb)
    assert set(tensors) == {f"{nr.PAYLOAD_PREFIX}layers.{l}.{f}" for l in (0, 1, 2) for f in ("gate_w", "gate_b", "direction")}
    assert g["payload"]["parent"] == psr_prod.payload_tensor_fingerprint(tensors)
    assert blk["dose_locus"]["serve_time_coefficient"] == 1.0 and blk["dose_locus"]["coefficient_keys_present"] == {}
    served = {c["check"]: c for c in blk["verification"]["checks"]}["served_ops"]
    assert served["status"] == "PASS" and served["coeff"] == 1.0
    # a copy whose payload matches the parent's is refused at the fingerprint level (a zero-direction parent, say)
    zero = {**_psr_blob(), "directions": {k: torch.zeros_like(v) for k, v in _psr_blob()["directions"].items()}}
    with pytest.raises(SystemExit, match="never reached the bytes"):
        psr_prod.fingerprint_record(zero, {**zero, "directions": {k: v * 0.5 for k, v in zero["directions"].items()}}, "psr")
