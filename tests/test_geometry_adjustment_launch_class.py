"""The shared adjustment validator (geometry.geometry_adjustment_problem) accepts the `launch_class` step by RE-DERIVING it from the pinned
site record — never by a whitelist entry (geometry-fix-2). Before: the chain knew lora_kernel_index_ceiling
and kv_shrink alone and refused `geometry.adjustments[0].kind='launch_class' is not one of [...]`, the kernel step was assumed at
adjustments[0] and its ceiling re-derived from the identity S — so the S=1 -> 14 step the s2r pair target needs would have been rejected
by every reader (the driver's manifest copy, the capctrl smoke, the math / GSM8K / IFEval validators).

Pinned here on a synthetic classed record (the s2r shape in miniature): the GENERATED step (native_standup.launch_class_geometry) passes
the validator alone and followed by a kernel step re-derived from the CLASS-adjusted S at the kernel step's own index; a forged to_S, forged
launch ids, a window that is not the run's, a class other than the attested one, a step that re-derives no change, and a kernel step placed
before the class step are refused by name; the KV step may follow the class step; the pre-existing chains ([] and the identity) are
unchanged; the driver passes the standup's attested lora_config."""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

from sisteer.capeval.serving import geometry as g  # noqa: E402
from sisteer.capeval.serving import native_standup as ns  # noqa: E402

KEY = "qwen35_4b_demo"
GEN = "f" * 64
L = 16384
WO, QKVO = ("o_proj",), ("k_proj", "o_proj", "q_proj", "v_proj")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in (g.SITE_GEOMETRY_DIR_ENV, g.SITE_DIR_ENV, g.SHARED_ROOT_ENV):
        monkeypatch.delenv(v, raising=False)


def _budget(S, Lb=L) -> g.ModelBudget:
    return g.ModelBudget(model_key=KEY, batched_token_budget=S * Lb, certified=True, source=g.MEASURED_BUDGET_SOURCE,
                         basis=g.BudgetBasis(method=f"{g.SITE_GEOMETRY_METHOD}: test", max_model_len=Lb, max_num_seqs=S, gpu_memory_utilization=0.9,
                                             kv_tokens_reported=S * Lb + 16, probe_job_id="11111111", measured_on="2026-10-02", source="test",
                                             first_refused=None, note="synthetic"))


def _launch(Ll, S, rank, loras):
    return {"L": Ll, "S": S, "batched_tokens": S * Ll, "max_lora_rank": rank, "max_loras": loras}


def _set(sid, mods, D, launches, budget_S) -> dict:
    first = next(iter(launches))
    return {"lora_modules": list(mods), "lora_max_module_dim": D, "budget": _budget(budget_S), "allowed_pairs": [[4, sid]], "certified_by": [first],
            "module_dim_launches": {first: D}, "budget_launches": dict(launches),
            "adapter_parity": g.adapter_parity_block([{"rank": 4, "set": sid, "status": g.PARITY_VERIFIED, "basis": "hf_peft", "mismatched_adapters": []}])}


def _install(tmp_path: Path, monkeypatch, *, wide: bool) -> Path:
    """The record: at 16,384 the r4/cap8 class admitted 14 (qkvo) / 16 (wo) and the r64/cap128 class 1; at 49,152 the r4 class 4-5 and the
    cap64 class 4. `wide` = the zero adapter's class (wo) carries D 9216 and qkvo's r4 launch admits 16 too, so the class step reaches 16
    and the LoRA kernel ceiling (2^31-1 // 9216 // 16384 = 14) then lowers it — the class + kernel chain."""
    D_wo = 9216 if wide else 4096
    q16 = 16 if wide else 14
    sets = {"wo": _set("wo", WO, D_wo, {"wo_r4_L16384": _launch(L, 16, 8, 4), "wo_r64_L16384": _launch(L, 1, 128, 128),
                                         "wo_r4_L49152": _launch(49152, 5, 8, 4), "wo_r64_L49152": _launch(49152, 4, 64, 64)}, 16),
            "qkvo": _set("qkvo", QKVO, 9216, {"q_r4_L16384": _launch(L, q16, 8, 4), "q_r64_L16384": _launch(L, 1, 128, 128),
                                              "q_r4_L49152": _launch(49152, 4, 8, 4), "q_r64_L49152": _launch(49152, 4, 64, 64)}, q16)}
    rec = g.site_geometry_record(model_key=KEY, hf_id="org/toy", revision="0" * 40, sets=sets, written_utc="2026-10-02T00:00:00Z",
                                 provenance={"probe_job_id": "11111111", "probe_record": "/shared/exp/multilora_cert/toy_probe_11111111",
                                             "probe_cluster": "cluster_a", g.RESULT_GENERATION_KEY: GEN,
                                             "attestation_sha256": {"wo_r4_L16384": {"file": None, "sha256": "ab" * 32}}, "host": "TESTHOST"})
    rec[g.STAGING_VERIFICATION_KEY] = {"verified_utc": "2026-10-02T00:00:01Z", "record_generation": GEN, "verifier_host": "TESTHOST"}
    d = tmp_path / "geometry"; d.mkdir(parents=True, exist_ok=True)
    (d / f"{KEY}.json").write_text(json.dumps(rec))
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(d))
    return d


def _standup_block(served_lora=(1, 3)):
    """What the native standup writes: the budget-derived geometry, the class step, then the kernel step from the class-adjusted S."""
    geom0 = g.resolve_geometry("multiturn_attacks", max_model_len=L, model_key=KEY, strict_identity=True)
    geom, rec = ns.launch_class_geometry(geom0, served_lora=served_lora, eval_name="multiturn_attacks", model_key=KEY)
    adjustments = [rec] if rec else []
    S_served = int(geom.max_num_seqs)
    k = g.kernel_ceiling_adjustment(KEY, L, S_served)
    if k:
        adjustments.append(k); S_served = int(k["to_S"])
    block = {**geom.as_dict(), "max_num_seqs": S_served, "adjustments": adjustments}
    S_id = g.recorded_identity_seqs({"max_model_len": L, "budget": geom.budget.batched_token_budget, "model_key": KEY})
    return geom0, geom, rec, k, block, S_id, S_served


def _check(block, S_id, S_served, **kw):
    return g.geometry_adjustment_problem(block, model_key=KEY, L=L, S_identity=S_id, S_served=S_served, native=True, **kw)


def test_the_chain_order_and_the_fields_are_pinned():
    assert g.ADJUSTMENT_KINDS == (g.ADJUSTMENT_LAUNCH_CLASS, g.ADJUSTMENT_KERNEL, g.ADJUSTMENT_KV) == ("launch_class", "lora_kernel_index_ceiling", "kv_shrink")
    assert g.LAUNCH_CLASS_ADJUSTMENT_FIELDS == ("kind", "from_S", "to_S", "max_lora_rank", "max_loras", "L", "window_seqs_source", "launches")
    assert set(g.launch_class_adjustment(from_S=1, to_S=14, served_lora=(1, 3), window=L, window_seqs_source="x", launches=["a"])) == set(g.LAUNCH_CLASS_ADJUSTMENT_FIELDS)


def test_the_generated_class_step_alone_passes_the_shared_validator_and_forgeries_are_refused_by_name(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, wide=False)
    geom0, geom, rec, k, block, S_id, S_served = _standup_block()
    assert (S_id, geom0.max_num_seqs, geom.max_num_seqs, S_served) == (1, 1, 14, 14) and k is None
    assert rec == {"kind": "launch_class", "from_S": 1, "to_S": 14, "max_lora_rank": 1, "max_loras": 3, "L": L, "window_seqs_source": "launch_class:1x3", "launches": ["q_r4_L16384"]}
    assert _check(block, S_id, S_served) is None and _check(block, S_id, S_served, served_lora=(1, 3)) is None
    # the pre-fix refusal is gone
    assert "is not one of" not in str(_check(block, S_id, S_served))
    forged = lambda **over: {**block, "adjustments": [dict(rec, **over)]}
    # a forged to_S, stated consistently as the served S: the class re-derivation refuses before the served-S check
    assert "re-derives 14" in _check({**block, "max_num_seqs": 13, "adjustments": [dict(rec, to_S=13)]}, S_id, 13)
    assert "launches that set S=14" in _check(forged(launches=["nope"]), S_id, S_served)
    assert "not this run's" in _check(forged(L=49152), S_id, S_served)
    assert "window_seqs_source" in _check(forged(window_seqs_source="record-wide"), S_id, S_served)
    assert "is not the identity S=1" in _check(forged(from_S=2), S_id, S_served)
    assert "not exactly" in _check({**block, "adjustments": [{k_: v for k_, v in rec.items() if k_ != "launches"}]}, S_id, S_served)
    # the attested class is held when the caller knows it; a class that re-derives differently is refused on its arithmetic
    assert "the standup attests 4 x 3" in _check(block, S_id, S_served, served_lora=(4, 3))
    assert "is not a pair of positive ints" in _check(forged(max_lora_rank=0), S_id, S_served)
    # a class that re-derives NO change is not a step: rank cap 64 x 64 adapters is dominated at every probed window (by the cap128 launches at
    # 16,384 and the cap64 ones at 49,152) and its tightest dominating S at 16,384 is the record-wide 1
    wide_class = dict(rec, max_lora_rank=64, max_loras=64, to_S=1)
    assert "re-derives the same S 1" in _check({**block, "max_num_seqs": 1, "adjustments": [wide_class]}, S_id, 1)
    # a class nothing dominates at some window is refused by the overlay, named
    assert "no probed launch at window 49152 dominates" in _check({**block, "max_num_seqs": 1, "adjustments": [dict(rec, max_lora_rank=128, max_loras=128, to_S=1)]}, S_id, 1)
    # the served S must be the last step's (a block that states no max_num_seqs of its own: the chain's end decides)
    assert "is not the last adjustment's to_S 14" in _check({k_: v for k_, v in block.items() if k_ != "max_num_seqs"}, S_id, 13)
    # no model key -> the step cannot be re-derived
    assert "no model key" in g.geometry_adjustment_problem(block, model_key=None, L=L, S_identity=S_id, S_served=S_served)


def test_the_class_step_followed_by_a_kernel_step_re_derived_from_the_class_adjusted_S_passes_and_misorders_are_refused(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, wide=True)
    geom0, geom, rec, k, block, S_id, S_served = _standup_block()
    assert (S_id, geom0.max_num_seqs, geom.max_num_seqs) == (1, 1, 16) and rec["to_S"] == 16 and rec["launches"] == ["q_r4_L16384", "wo_r4_L16384"]
    assert k == {"kind": g.ADJUSTMENT_KERNEL, "from_S": 16, "to_S": 14, "D": 9216, "ceiling": (2**31 - 1) // 9216} and S_served == 14
    assert [a["kind"] for a in block["adjustments"]] == ["launch_class", "lora_kernel_index_ceiling"]
    assert _check(block, S_id, S_served, served_lora=(1, 3)) is None
    # a kernel step placed BEFORE the class step is the wrong order; a kernel step re-derived from the identity S (1 -> nothing) is refused
    assert "the contract's order is" in _check({**block, "adjustments": [k, rec]}, S_id, S_served)
    assert "the formula gives" in _check({**block, "adjustments": [rec, dict(k, from_S=1)]}, S_id, S_served)
    # a native record that omits the kernel step the class-adjusted S requires is refused naming it from the class-adjusted S
    assert "must record the 'lora_kernel_index_ceiling' step (16 -> 14" in _check({**block, "max_num_seqs": 16, "adjustments": [rec]}, S_id, 16)
    # a KV step may follow (class, kernel) — it is held to kv_shrink_problem against the previous step's S first (a bare stub is refused as
    # an incomplete shrink record), so the chain order admits it and the record's own fields decide
    kv_stub = {"kind": g.ADJUSTMENT_KV, "from_S": 14, "to_S": 12}
    assert "'kv_shrink' step does not hold" in str(_check({**block, "max_num_seqs": 12, "adjustments": [rec, k, kv_stub]}, S_id, 12))
    assert "the contract's order is" not in str(_check({**block, "max_num_seqs": 12, "adjustments": [rec, k, kv_stub]}, S_id, 12))


def test_the_pre_existing_chains_are_unchanged_and_the_driver_passes_the_attested_class(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, wide=False)
    # an identity native record: [] and the identity S — accepted exactly as before; a bare smaller S still refused
    identity = {"max_model_len": L, "max_num_seqs": 1, "adjustments": [], "identity": True, "model_key": KEY}
    assert _check(identity, 1, 1) is None
    assert "a smaller S is accepted only through a named, re-derivable step" in _check({**identity, "max_num_seqs": 1}, 2, 1)
    src = (REPO / "scripts/served_capability_run.py").read_text()
    assert 'def _native_geometry_problem(geom: Mapping, lora_config: "Mapping | None" = None)' in src
    assert 'problem = _native_geometry_problem(block, endpoint.get("lora_config"))' in src
    assert "served_lora=served_lora" in src
