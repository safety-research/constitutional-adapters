"""The steered smoke's PETRI-geometry certification seams
(scripts/petri_steered_smoke.py) — CPU-only, no vllm.

Job 227308 certified the audit's max_num_seqs at max_model_len 2048
only, while the audit's servers run build_server_launch's defaults (16k
floor) — and at that geometry the requested concurrency is not free:
vLLM resolved 486,944 KV tokens = 29.72x for a requested 32 at the
default utilization (job 227309's production server), silently
scheduling fewer rather than erroring. Jobs 229194/229390 then measured
that 32 x 16384 cannot fit one H200 under no-chunking at ANY
utilization (KV coverage and warmup headroom are zero-sum), so the
production default and the certified PETRI geometry are 24 x 16384.
Three contracts hold that gap closed:

1. the smoke's PETRI geometry constants ARE build_server_launch's
   defaults (a drift between them re-opens the original bug: a gate
   certifying a geometry production does not run);
2. ``kv_coverage_problem`` is the gate-shaped verdict on whether the
   resolved KV cache actually backs the full no-chunking budget —
   coverage passes, a shortfall names both numbers, and an UNREADABLE
   value refuses rather than passing. Since the hybrid fix it DELEGATES
   to the launcher's shared ``vllm_server.kv_coverage_problem`` (this
   was the formula's third copy): a uniform cache keeps the original
   num_gpu_blocks x block_size arithmetic verbatim, a hybrid cache is
   held to the allocator's own max-concurrency verdict instead — a
   healthy hybrid PASSES; only unreadable structure refuses;
3. the fidelity child's argv carries the geometry (and the utilization
   that makes it coverable), so the phase the orchestrator asks for is
   the phase the worker builds.
"""

import argparse
import importlib.util
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location(
    "_petri_steered_smoke_geom", ROOT / "scripts" / "petri_steered_smoke.py")
smoke = importlib.util.module_from_spec(_spec)
sys.modules["_petri_steered_smoke_geom"] = smoke
_spec.loader.exec_module(smoke)


# --- 1. the certified geometry IS the production geometry ----------------------


def test_petri_geometry_matches_build_server_launch_defaults():
    from sisteer.capeval.serving import vllm_server as vs

    params = inspect.signature(vs.build_server_launch).parameters
    assert smoke.PETRI_MAX_MODEL_LEN == params["max_model_len"].default, (
        "the smoke's PETRI-geometry gate certifies a max_model_len that "
        "build_server_launch does not serve by default — the gate would "
        "again certify a geometry production does not run")
    assert smoke.PETRI_MAX_NUM_SEQS == vs.DEFAULT_MAX_NUM_SEQS, (
        "the smoke's PETRI-geometry gate certifies a max_num_seqs that is "
        "not the composer's concurrent-mode default")


def test_petri_budget_is_the_exact_no_chunking_product():
    assert (smoke.PETRI_MAX_NUM_BATCHED_TOKENS
            == smoke.PETRI_MAX_NUM_SEQS * smoke.PETRI_MAX_MODEL_LEN)
    # the identity, not just the product: the 2048 gate's budget is the
    # same construction, and neither is inherited from the other.
    assert (smoke.MAX_NUM_BATCHED_TOKENS
            == smoke.MAX_NUM_SEQS * smoke.MAX_MODEL_LEN)


def test_petri_utilization_is_raised_above_the_smoke_default():
    # the PETRI budget's profiling forward eats far more headroom than
    # the 65,536-token engines'; the constant exists to be HIGHER than
    # the smoke default so full KV coverage resolves.
    assert (smoke.PETRI_GPU_MEMORY_UTILIZATION
            > smoke.SMOKE_GPU_MEMORY_UTILIZATION)


def test_the_negative_result_stands_24_not_32():
    # 32 x 16384 measurably does NOT fit one H200 under the no-chunking
    # identity AT VLLM_BATCH_INVARIANT=1 (jobs 229194/229390:
    # full-coverage KV 64.0 GiB + a
    # 14.0 GiB full-budget MLP warmup intermediate + ~15 GiB weights
    # against 139.81 GiB, with KV and warmup headroom zero-sum in
    # utilization — 0.95 under-covered KV, 0.97 covered it and OOMed).
    # 24 x 16384 = 393,216 is the widest that fits with real slack.
    # This test exists so a future "just put it back to 32" trips over
    # the measurement before it trips over the cluster. With
    # exact_replay dropped (2026-08-27) production runs BI=0, and the
    # smoke's --petri32-probe-utils re-measure took the question there
    # too (job 230089, utils 0.90/0.95/0.97): SAME WALL — 0.95 resolved
    # 520,736 KV tokens vs BI=1's 521,520, essentially identical, and
    # 0.97 could not init. The limit is structural, closed at BOTH BI
    # settings; re-measure only on a stack change that plausibly moves
    # the memory profile (new vLLM pin, new model), and move
    # PETRI_MAX_NUM_SEQS, DEFAULT_MAX_NUM_SEQS and this test together,
    # citing the new job.
    assert smoke.PETRI_MAX_NUM_SEQS == 24
    assert smoke.PETRI_MAX_NUM_BATCHED_TOKENS == 393216


# --- 2. kv_coverage_problem: covered / short / unreadable -----------------------


def _uniform_facts(n_blocks, block_size=16):
    """One worker's facts in the REAL producer's shape — since the
    hybrid fix, _worker_kv_cache_facts always merges the group fragment
    (a facts dict with NO group count means the worker read failed, and
    it refuses)."""
    return {"num_gpu_blocks": n_blocks, "block_size": block_size,
            "kv_num_cache_groups": 1}


def _hybrid_facts(mc):
    """A Qwen3.5-shaped hybrid: full-attention + gated-delta-net
    groups, with the allocator's own max-concurrency verdict."""
    return {"num_gpu_blocks": 8192, "block_size": 16,
            "kv_num_cache_groups": 2,
            "kv_cache_groups": [
                {"spec": "FullAttentionSpec", "block_size": 16,
                 "num_layers": 8, "page_size_bytes": 4096},
                {"spec": "MambaSpec", "block_size": 64,
                 "num_layers": 24, "page_size_bytes": 1 << 20}],
            "kv_max_concurrency": mc}


def test_full_coverage_passes():
    # 33,000 blocks x 16 = 528,000 tokens >= 32 x 16,384 = 524,288
    assert smoke.kv_coverage_problem(
        _uniform_facts(33000),
        max_num_seqs=32, max_model_len=16384) is None


def test_the_measured_production_shortfall_fails_and_names_both_numbers():
    # job 227309's production server: 486,944 KV tokens against the
    # 524,288-token budget = 29.72x. The verdict must name the resolved
    # tokens, the budget, and the actually-backed concurrency.
    problem = smoke.kv_coverage_problem(
        _uniform_facts(30434),
        max_num_seqs=32, max_model_len=16384)
    assert problem is not None
    assert "486944" in problem
    assert "524288" in problem
    assert "29.72" in problem


def test_exact_coverage_is_coverage():
    # 32,768 blocks x 16 = 524,288 tokens == the budget, exactly.
    assert smoke.kv_coverage_problem(
        _uniform_facts(32768),
        max_num_seqs=32, max_model_len=16384) is None


def test_the_measured_gate_refusal_point_fails_by_a_hair():
    # job 229194 ran the PETRI phase at gpu_memory_utilization 0.95 and
    # the gate refused on exactly this resolution: 32,595 blocks x 16 =
    # 521,520 tokens = 31.83x — 2,768 tokens short. The refusal is the
    # mechanism working (a 31.83-way "32" must not certify), and it is
    # why PETRI_GPU_MEMORY_UTILIZATION is calibrated above 0.95.
    problem = smoke.kv_coverage_problem(
        _uniform_facts(32595),
        max_num_seqs=32, max_model_len=16384)
    assert problem is not None
    assert "521520" in problem
    assert "31.83" in problem


def test_unreadable_facts_refuse_rather_than_pass():
    for facts in (
            # the group fragment is missing entirely — the worker read
            # failed (the RPC's read_error path), or a pre-hybrid facts
            # shape; either way coverage cannot be verified
            {"num_gpu_blocks": 33000, "block_size": 16},
            {},
            # group structure readable, the SIZE is not (the uniform
            # branch's own refusal, wording preserved)
            _uniform_facts(None),
            _uniform_facts(33000, block_size=None),
            _uniform_facts(0),
            _uniform_facts(-4),
            _uniform_facts(True)):
        problem = smoke.kv_coverage_problem(
            facts, max_num_seqs=32, max_model_len=16384)
        assert problem is not None, (
            f"unreadable KV facts {facts!r} must refuse, never pass — a "
            "coverage that cannot be verified must not certify")
        assert "could not be read" in problem


def test_hybrid_coverage_is_the_allocator_verdict_not_the_formula():
    # a healthy hybrid PASSES — never refused for being hybrid
    assert smoke.kv_coverage_problem(
        _hybrid_facts(24.0), max_num_seqs=24, max_model_len=16384) is None
    # under-covered refuses on the allocator's number even where the
    # uniform arithmetic (8,192 x 16 = 131,072 tokens for an 8 x 16,384
    # budget) would certify — the quiet-miscount shape the shared
    # verdict exists to kill; the refusal names the group structure
    problem = smoke.kv_coverage_problem(
        _hybrid_facts(6.5), max_num_seqs=8, max_model_len=16384)
    assert problem is not None
    assert "6.50" in problem and "MambaSpec" in problem


def test_smoke_verdict_is_the_shared_launcher_verdict():
    # SINGLE SOURCE: the smoke's verdict IS the launcher's readiness
    # verdict, key-adapted — byte-identical strings across uniform and
    # hybrid, pass and refuse (three drifting copies of this formula is
    # how the hybrid miscount survived two fixes)
    from sisteer.capeval.serving import vllm_server as vs
    for facts in (_uniform_facts(33000), _uniform_facts(30434),
                  _uniform_facts(None), {},
                  _hybrid_facts(24.0), _hybrid_facts(2.5),
                  _hybrid_facts(None)):
        rep = dict(facts)
        rep["kv_num_gpu_blocks"] = facts.get("num_gpu_blocks")
        rep["kv_block_size"] = facts.get("block_size")
        assert smoke.kv_coverage_problem(
            facts, max_num_seqs=24, max_model_len=16384) == \
            vs.kv_coverage_problem(rep, max_num_seqs=24,
                                   max_model_len=16384), facts


def test_worker_kv_cache_facts_merges_the_group_report(monkeypatch):
    import types

    class FullAttentionSpec:
        block_size = 16
        page_size_bytes = 4096

    class Group:
        def __init__(self, spec, n):
            self.kv_cache_spec = spec
            self.layer_names = [f"l{i}" for i in range(n)]

    kv_cfg = types.SimpleNamespace(
        kv_cache_groups=[Group(FullAttentionSpec(), 32)], num_blocks=8192)
    worker = types.SimpleNamespace(
        vllm_config=types.SimpleNamespace(
            cache_config=types.SimpleNamespace(num_gpu_blocks=8192,
                                               block_size=16)),
        model_runner=types.SimpleNamespace(kv_cache_config=kv_cfg))
    # the allocator's function, faked at its REAL import path (no vLLM
    # on this box) — kv_group_report's lazy default import must find it
    util = types.ModuleType("vllm.v1.core.kv_cache_utils")
    util.get_max_concurrency_for_kv_cache_config = (
        lambda vllm_config, kv_cache_config: 48.0)
    for name, mod in (("vllm", types.ModuleType("vllm")),
                      ("vllm.v1", types.ModuleType("vllm.v1")),
                      ("vllm.v1.core", types.ModuleType("vllm.v1.core")),
                      ("vllm.v1.core.kv_cache_utils", util)):
        monkeypatch.setitem(sys.modules, name, mod)

    facts = smoke._worker_kv_cache_facts(worker)
    assert facts["num_gpu_blocks"] == 8192 and facts["block_size"] == 16
    assert facts["kv_num_cache_groups"] == 1
    assert facts["kv_max_concurrency"] == 48.0
    assert facts["kv_cache_groups"][0]["spec"] == "FullAttentionSpec"
    # and the merged facts certify through the smoke's own verdict
    assert smoke.kv_coverage_problem(
        facts, max_num_seqs=8, max_model_len=16384) is None


# --- 3. the child argv carries the geometry -------------------------------------


def _ns() -> argparse.Namespace:
    return argparse.Namespace(model="m/m", revision="deadbeef")


def test_fidelity_child_argv_carries_the_petri_geometry():
    argv = smoke._fidelity_child_argv(
        _ns(), Path("reg"), Path("arms.json"), Path("out.json"),
        max_model_len=smoke.PETRI_MAX_MODEL_LEN,
        max_num_seqs=smoke.PETRI_MAX_NUM_SEQS,
        gpu_memory_utilization=smoke.PETRI_GPU_MEMORY_UTILIZATION)

    def val(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    assert val("--max-model-len") == str(smoke.PETRI_MAX_MODEL_LEN)
    assert val("--max-num-seqs") == str(smoke.PETRI_MAX_NUM_SEQS)
    assert (val("--gpu-memory-utilization")
            == str(smoke.PETRI_GPU_MEMORY_UTILIZATION))


def test_fidelity_child_argv_defaults_to_the_2048_gate_unchanged():
    argv = smoke._fidelity_child_argv(
        _ns(), Path("reg"), Path("arms.json"), Path("out.json"))

    def val(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    assert val("--max-model-len") == str(smoke.MAX_MODEL_LEN)
    assert val("--max-num-seqs") == str(smoke.MAX_NUM_SEQS)
    assert (val("--gpu-memory-utilization")
            == str(smoke.SMOKE_GPU_MEMORY_UTILIZATION))
