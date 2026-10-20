"""The serving probe's plan estimator prices the PREFILL batch.

The defect (found on attempt 2 of the first serving probe): the plan lowered S only by the
packed-stride int32 ceiling and the KV budget, never by the batch the harness's no-chunking invariant forces the engine
to prefill in one go (`max_num_batched_tokens = S x L`). The sets the stride did not bind kept S = 21 / 15 / 8 / 32 at
L = 49,152 / 65,536 / 98,304 / 16,384 (0.5-1.03 M tokens) and the engine died in `_initialize_kv_caches` while the largest
batch that ever came up was 262,144 (a 4,096 window at S 64). `write_verdict` read those as `oom_at_R=<cap>_at_L=<L>`
-> `ceiling_S<1` -> the (down|mlp, 4|16) pairs EXCLUDED on false OOMs.

The fix, pinned here on a dense toy whose stride and KV terms never bind (so the requested production concurrency composes
the prefill batch, exactly the shape the defect had) against a synthetic record of passes and OOMs:
* S_m = min(S_stride, S_kv, requested, floor(batched_token_budget / L)) — `launch_admission`, `build_plan(batched_token_budget=)`;
* the budget is MEASURED (`measured_budget`): the largest S x L a launch of the record came up at is the floor, the
  smallest start-time OOM above it the ceiling; method and launch ids ride the plan, `probe_progress.json` and the
  geometry record's provenance; never a typed constant;
* `write_verdict` reads an `oom` through `oom_kind`: a start-time prefill OOM above the budget is `replan_required`
  (undecided, never an exclusion); a serving OOM, a start-time OOM at S = 1, or one within the budget is genuine;
* `run --resume --replan` re-derives S for the outstanding AND the replan_required launches with stable ids (the id
  encodes manifest / cap / L / tp, never S), moves the superseded results aside (never deletes) and re-runs them;
  idempotent — a second --replan changes nothing and moves nothing.

Pinned to the SPLIT topology (the 2026-10-01 ruling): every (set, rank) pair is served by TWO servers at every window —
`S1cc` (control + constitution + zero at cap rank_cap(r); the carriers ride it for `wo`) and `S1d` (difference + zero at
cap rank_cap(2r)) — plus the shared `S2` for every set but `wo`; the ids follow `launch_id(manifest, cap, L)`
(`down_r4_S1cc_cap8_L4096`, `down_r16_S1d_cap32_L49152`, ...). The record below is the defect's shape on that topology:
both servers of a pair at a window share the planned S, so each row is two launches.
CPU only: fake engines through the `Deps` seam; synthetic model ids; no cluster path, no host name."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from sisteer.capeval.serving import standup_manifest as sm
from sisteer.extraction.lora_modules import QWEN35_4B_DIMS as Q4

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
sys.path.insert(0, str(REPO / "tests"))
import hpo_probe_status as hps  # noqa: E402
import serving_probe as sp  # noqa: E402
from front_door_testlib import assert_no_tokens  # noqa: E402
from test_serving_probe_parallel import _fake_deps  # noqa: E402  (the serving-probe suite's recording fake engine)
from test_serving_probe_smoke import MODEL, MOE_TOY, REV, TOY, FakeExporter, _passed  # noqa: E402

SBATCH = REPO / "scripts/sbatch/serving_probe.sbatch"
FLOOR = 262_144            # S 64 x 4,096 — the largest batch the record proved
CEILING = 1_572_864        # down_r{4,16} / mlp_r{4,16} at 49,152, S 32 — the smallest start-time OOM above it
START_TAIL = ("(EngineCore pid=1) ERROR core.py:1231 EngineCore failed to start.\n"
              "    kv_cache_config = self._initialize_kv_caches(vllm_config)\n"
              "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 63.00 GiB.\n")


def _servers(set_id, rank, L):
    """The pair's two servers' launch ids at window ``L`` under the split topology: ``S1cc`` (control + constitution, cap
    rank_cap(r)) then ``S1d`` (difference, cap rank_cap(2r)) — the order the plan sorts them in."""
    return [sp.launch_id(f"{set_id}_r{rank}_{sm.TOPOLOGY_S1CC}", sm.rank_cap(rank), L),
            sp.launch_id(f"{set_id}_r{rank}_{sm.TOPOLOGY_S1D}", sm.rank_cap(2 * rank), L)]


#: the synthetic record on the toy dims: per (set, rank, window) BOTH servers of the pair, (status, launch_S, engine_S | None,
#: ready_s | None). On the toy neither the stride nor the KV term binds, so every pass at 4,096 rides the requested S 64 (the plan's
#: own S for its launch) and every launch at 49,152 was composed at the requested S 32 (1.57 M tokens): the (down|mlp, 4|16) pairs
#: died at start there (the defect), the rank-64 servers died at S 1 (buffers + weights do not fit: genuine), two 4,096 launches
#: were refused by the composer, and the Q/K/V-bearing pairs at 49,152 have not run.
PAIR_RECORD = {
    ("attn_mlp", 4, 4096): ("passed", 64, 64, 771.0), ("attn_mlp", 16, 4096): ("passed", 64, 64, 190.0),
    ("attn_mlp", 64, 4096): ("passed", 64, 64, 249.0), ("attn_mlp", 64, 49152): ("oom", 1, None, None),
    ("down", 4, 4096): ("passed", 64, 64, 195.0), ("down", 4, 49152): ("oom", 32, None, None),
    ("down", 16, 4096): ("refused", 64, None, None), ("down", 16, 49152): ("oom", 32, None, None),
    ("down", 64, 4096): ("passed", 64, 64, 169.0), ("down", 64, 49152): ("oom", 1, None, None),
    ("mlp", 4, 4096): ("passed", 64, 64, 209.0), ("mlp", 4, 49152): ("oom", 32, None, None),
    ("mlp", 16, 4096): ("refused", 64, None, None), ("mlp", 16, 49152): ("oom", 32, None, None),
    ("mlp", 64, 4096): ("passed", 64, 64, 207.0), ("mlp", 64, 49152): ("oom", 1, None, None),
    ("qkvo", 4, 4096): ("passed", 64, 64, 175.0), ("qkvo", 16, 4096): ("passed", 64, 64, 179.0),
    ("qkvo", 64, 4096): ("passed", 64, 64, 215.0),
    ("wo", 4, 4096): ("passed", 64, 64, 205.0), ("wo", 16, 4096): ("passed", 64, 64, 175.0),
    ("wo", 64, 4096): ("passed", 64, 64, 179.0),
}
#: launch id -> (status, launch_S, engine_S | None, ready_s | None): 44 launches over the 22 (set, rank, window) rows
RECORD = {lid: row for (s, r, L), row in PAIR_RECORD.items() for lid in _servers(s, r, L)}
#: the OLD plan's S at the long windows (the requested production concurrency: nothing else binds on the toy) and the re-derived
#: floor // L
OLD_S = {"L49152": 32, "L65536": 16, "L98304": 8, "L16384": 32}
NEW_S = {"L49152": 5, "L65536": 4, "L98304": 2, "L16384": 16}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    from sisteer.capeval.serving import geometry as g
    for v in (g.SITE_GEOMETRY_DIR_ENV, g.SITE_DIR_ENV, g.SHARED_ROOT_ENV, "SISTEER_MODELS_TABLE",
              sp.VISIBLE_DEVICES_ENV, "SLURM_JOB_ID"):
        monkeypatch.delenv(v, raising=False)


def _toy_plan(**kw):
    return sp.build_plan(TOY, model_key=MODEL, model_id="org/toy", revision=REV, **kw)


def _oom(rec, S, *, ready=None, tail=START_TAIL):
    """An ``oom`` launch record as the driver writes one: ``launch_S``, ``ready_s`` None when the engine never came up."""
    return {"launch_id": rec["id"], "manifest": rec["manifest"], "serves": rec["serves"], "L": rec["L"], "status": "oom",
            "launch_S": int(S), "ready_s": ready, "failure": {"class": "oom", "message": "cuda_out_of_memory", "log_tail": tail}}


def _record_results(plan):
    """The RECORD table as result records over the toy plan (passes through the smoke suite's `_passed`)."""
    by = {r["id"]: r for r in plan["launches"]}
    out = {}
    for lid, (status, S, eng, ready) in RECORD.items():
        rec = by[lid]
        m = sm.Manifest.from_dict(plan["manifests"][rec["manifest"]])
        if status == "passed":
            res = _passed(rec, m, engine_S=eng)
            res["launch_S"] = S
            res["ready_s"] = ready
        elif status == "oom":
            res = _oom(rec, S)
        else:
            res = {"launch_id": lid, "manifest": rec["manifest"], "serves": rec["serves"], "L": rec["L"], "status": "refused",
                   "launch_S": S, "ready_s": None, "failure": {"class": "refused", "message": "refused:composer"}}
        out[lid] = res
    return out


# --------------------------------------------------------------------------- the S formula


class TestSFormula:
    def test_the_shared_admission_carries_the_term_and_the_estimate_prices_the_prefill_when_measured(self):
        """The term lives in standup_manifest (production standups size through it too): the kwarg caps S at
        floor(budget / L) after stride / KV / requested; without it the record keeps today's shape and S. The estimate
        prices the no-chunking prefill ONLY from a measured per-token cost — never a typed constant — and otherwise
        reports the batch with `prefill_activations` None."""
        cls = sm.target_class_for(TOY, "down")
        old = sm.admitted_max_num_seqs(TOY, cls, 49152, kv_budget_bytes=1e12, requested_S=32)
        assert old["admitted_S"] == 32 and old["binding"] == "requested" and old["budget_S"] is None
        assert old["batched_token_budget"] is None and old["batched_tokens"] == CEILING
        new = sm.admitted_max_num_seqs(TOY, cls, 49152, kv_budget_bytes=1e12, requested_S=32, batched_token_budget=FLOOR)
        assert new["admitted_S"] == 5 and new["binding"] == "batched_tokens" and new["budget_S"] == 5 and new["batched_tokens"] == 245_760
        assert {k: v for k, v in new.items() if k not in ("admitted_S", "binding", "budget_S", "batched_tokens", "batched_token_budget")} == \
               {k: v for k, v in old.items() if k not in ("admitted_S", "binding", "budget_S", "batched_tokens", "batched_token_budget")}
        # on the worked example's Qwen3.5-4B dims the Q/K/V set is stride-bound at 49,152 (the packed 8192 + 2 x 1024 stride:
        # S 4 = 196,608 tokens): the budget term does not move it
        qk = sm.admitted_max_num_seqs(Q4, sm.target_class_for(Q4, "qkvo"), 49152, kv_budget_bytes=1e12, requested_S=32, batched_token_budget=FLOOR)
        assert qk["admitted_S"] == 4 and qk["binding"] == "stride" and qk["budget_S"] == 5
        # a routed-experts toy: the SAME prefill term over the expert-bearing class (the strides read the MoE wrappers)
        mcls = sm.target_class_for(MOE_TOY, "down")
        assert "experts" in mcls and set(sm.executing_strides(MOE_TOY, mcls)) == {"mlp.shared_expert.down_proj", "mlp.experts.w13", "mlp.experts.w2"}
        moe = sm.admitted_max_num_seqs(MOE_TOY, mcls, 49152, kv_budget_bytes=1e12, requested_S=32, batched_token_budget=FLOOR)
        assert moe["admitted_S"] == 5 and moe["binding"] == "batched_tokens" and moe["budget_S"] == 5 and moe["D"] == 64
        m = sm.build_manifests(TOY, set_id="down", rank=4)[0]
        est0 = sm.estimated_vram_bytes(m, 8, TOY, kv_per_seq_bytes=1.0, S=5)
        assert est0["batched_tokens"] is None and est0["prefill_activations"] is None and est0["prefill_bytes_per_token"] is None
        est1 = sm.estimated_vram_bytes(m, 8, TOY, kv_per_seq_bytes=1.0, S=5, batched_tokens=245_760)
        assert est1["batched_tokens"] == 245_760 and est1["prefill_activations"] is None and est1["total"] == est0["total"]
        est2 = sm.estimated_vram_bytes(m, 8, TOY, kv_per_seq_bytes=1.0, S=5, batched_tokens=245_760, prefill_bytes_per_token=1000.0)
        assert est2["prefill_activations"] == 245_760_000.0 and est2["total"] == est0["total"] + 245_760_000.0
        assert est2["headroom"] == est0["headroom"] - 245_760_000.0

    def test_a_moe_plan_carries_the_budget_term_and_its_expert_ranks_beside_it(self):
        """The routed-experts option rides the same admission: a MoE toy plan under a measured floor re-derives S = floor // L
        on every long-window launch, and each launch still names its server's ``moe_lora_rank`` (the bytes law that priced it)."""
        plan = sp.build_plan(MOE_TOY, model_key=MODEL, model_id="org/toy", revision=REV, sets=("down",), ranks=(4,),
                             windows=(4096, 49152), ladder_caps=(),
                             batched_token_budget={"method": sp.BUDGET_METHOD_RECORD_FLOOR, "tokens": FLOOR, "floor_launch": "x"})
        by = {r["id"]: r for r in plan["launches"]}
        assert plan["is_moe"] is True and plan["pairs"][0]["per_expert_rank"] == {"shared_expert": 4, "routed_expert": 2}
        assert by["down_r4_S1cc_cap8_L49152"]["admitted_S"] == 5 and by["down_r4_S1cc_cap8_L49152"]["admission"]["binding"] == "batched_tokens"
        assert by["down_r4_S1cc_cap8_L49152"]["moe_lora_rank"] == 4 == by["down_r4_S1d_cap8_L49152"]["moe_lora_rank"]
        assert by["down_r4_S2_cap8_L49152"]["moe_lora_rank"] is None and by["down_r4_S1cc_cap8_L4096"]["admitted_S"] == 64
        assert by["down_r4_S1cc_cap8_L49152"]["expected_lora_bytes"]["rank_by_family"] == {"shared_expert": 8, "routed_experts": 4}

    def test_launch_admission_adds_the_batched_token_term_after_stride_kv_and_requested(self):
        m = sm.build_manifests(TOY, set_id="down", rank=4)[0]           # D = 64: the stride never binds at these windows
        kv = 1e12
        adm = sp.launch_admission(TOY, m, 49152, requested_S=32, kv_budget_bytes=kv, recurrent_state_bytes=0, batched_token_budget=None)
        assert adm["admitted_S"] == 32 and adm["binding"] == "requested" and adm["budget_S"] is None and adm["batched_token_budget"] is None
        assert adm["batched_tokens"] == 32 * 49152 == CEILING             # the OLD plan's composed prefill batch
        adm = sp.launch_admission(TOY, m, 49152, requested_S=32, kv_budget_bytes=kv, recurrent_state_bytes=0, batched_token_budget=FLOOR)
        assert adm["admitted_S"] == 5 == FLOOR // 49152 and adm["binding"] == "batched_tokens" and adm["budget_S"] == 5
        assert adm["batched_tokens"] == 5 * 49152 <= FLOOR and adm["batched_token_budget"] == FLOOR
        # a budget that does not bind leaves the stride / requested binding alone
        adm = sp.launch_admission(TOY, m, 4096, requested_S=64, kv_budget_bytes=kv, recurrent_state_bytes=0, batched_token_budget=FLOOR)
        assert adm["admitted_S"] == 64 and adm["binding"] == "requested" and adm["batched_tokens"] == FLOOR
        # a budget smaller than one window: S 0 is recorded (the driver launches it at 1; the verdict reads S = 1 OOMs as genuine)
        adm = sp.launch_admission(TOY, m, 98304, requested_S=8, kv_budget_bytes=kv, recurrent_state_bytes=0, batched_token_budget=90_000)
        assert adm["admitted_S"] == 0 and adm["binding"] == "batched_tokens"

    def test_every_exposed_launch_comes_out_at_or_under_the_measured_floor_and_the_passes_keep_their_S(self):
        """Every launch at the long windows (both servers of each pair, the pair's union — the served launch class — and the
        shared S2, 184 of the plan's 240) composed a prefill above the floor under the old plan; against the record's floor they
        all come out at floor // L."""
        old = {r["id"]: r for r in _toy_plan()["launches"]}
        new_plan = _toy_plan(batched_token_budget={"method": sp.BUDGET_METHOD_RECORD_FLOOR, "tokens": FLOOR,
                                                   "floor_launch": "attn_mlp_r16_S1cc_cap16_L4096"})
        new = {r["id"]: r for r in new_plan["launches"]}
        assert list(old) == list(new) and len(new) == 240                       # ids and order are stable
        exposed = [lid for lid, r in old.items() if r["L"] in (16384, 49152, 65536, 98304)]
        assert len(exposed) == 184 and set(RECORD) <= set(new)
        assert new_plan["batched_token_budget"]["tokens"] == FLOOR and new_plan["batched_token_budget"]["method"] == "record_floor"
        # nothing in the plan exceeds the floor any more
        assert [i for i, r in new.items() if r["admitted_S"] * r["L"] > FLOOR] == []
        for lid in exposed:
            Lkey = lid.rsplit("_", 1)[1]
            assert old[lid]["admitted_S"] == OLD_S[Lkey] and old[lid]["admission"]["binding"] == "requested", lid   # the defect, as recorded
            assert old[lid]["admitted_S"] * old[lid]["L"] >= 524_288
            assert new[lid]["admitted_S"] == NEW_S[Lkey] and new[lid]["admission"]["binding"] == "batched_tokens", lid
            assert new[lid]["admitted_S"] * new[lid]["L"] <= FLOOR
        # the record's passes are at or under the floor by construction and keep exactly their S
        for lid, (status, S, eng, _r) in RECORD.items():
            if status == "passed":
                exp = old[lid]["admitted_S"]
                assert new[lid]["admitted_S"] == exp == S, lid
        # the estimate rides the re-derived S and records the prefill batch it would compose
        assert new["down_r4_S1cc_cap8_L49152"]["estimated_vram"]["S"] == 5 and old["down_r4_S1cc_cap8_L49152"]["estimated_vram"]["S"] == 32
        assert new["down_r4_S1cc_cap8_L49152"]["estimated_vram"]["batched_tokens"] == 245_760
        assert old["down_r4_S1cc_cap8_L49152"]["estimated_vram"]["batched_tokens"] == CEILING
        # no budget: the plan says so instead of carrying a typed constant
        b = _toy_plan()["batched_token_budget"]
        assert b["method"] == "unmeasured" and b["tokens"] is None and "note" in b


# --------------------------------------------------------------------------- the measured budget and the OOM kinds


class TestMeasuredBudget:
    def test_the_record_measures_its_floor_and_ceiling_by_launch_id(self):
        plan = _toy_plan()
        results = _record_results(plan)
        b = sp.measured_budget(plan, results)
        assert b["method"] == sp.BUDGET_METHOD_RECORD_FLOOR and b["tokens"] == FLOOR
        assert b["floor_launch"] == "attn_mlp_r16_S1cc_cap16_L4096"           # the first of the 262,144 ties in id order
        assert b["ceiling_tokens"] == CEILING and b["ceiling_launch"] == "down_r16_S1cc_cap16_L49152"
        assert b["n_passed"] == 26 and b["n_start_time_oom"] == 14
        # the S = 1 start-time OOMs (the rank-64 servers at 49,152: buffers + weights do not fit) are never a ceiling candidate
        only_min = {k: v for k, v in results.items() if v["status"] != "oom" or v["launch_S"] == 1}
        b2 = sp.measured_budget(plan, only_min)
        assert b2["tokens"] == FLOOR and b2["ceiling_tokens"] is None and b2["n_start_time_oom"] == 6
        # no pass at all: unmeasured, tokens None, the ceiling still reported
        b3 = sp.measured_budget(plan, {k: v for k, v in results.items() if v["status"] == "oom"})
        assert b3["method"] == sp.BUDGET_METHOD_UNMEASURED and b3["tokens"] is None and b3["ceiling_tokens"] == CEILING
        assert sp.measured_budget(plan, {})["tokens"] is None

    def test_oom_kind_reads_the_four_kinds_and_replan_required_lists_only_the_over_budget_ones(self):
        plan = _toy_plan()
        by = {r["id"]: r for r in plan["launches"]}
        results = _record_results(plan)
        rec = by["down_r4_S1cc_cap8_L49152"]
        assert sp.oom_kind(rec, results["down_r4_S1cc_cap8_L49152"], FLOOR) == sp.OOM_START_OVER_BUDGET
        assert sp.oom_kind(rec, results["down_r4_S1cc_cap8_L49152"], None) == sp.OOM_START_OVER_BUDGET   # no floor yet: still re-plan
        # the difference server died the same way at the same batch: the same kind
        assert sp.oom_kind(by["down_r4_S1d_cap8_L49152"], results["down_r4_S1d_cap8_L49152"], FLOOR) == sp.OOM_START_OVER_BUDGET
        # the genuine KV-overs (the rank-64 servers at 49,152, S 1) stay exclusions
        assert sp.oom_kind(by["attn_mlp_r64_S1cc_cap64_L49152"], results["attn_mlp_r64_S1cc_cap64_L49152"], FLOOR) == sp.OOM_START_MIN_S
        assert sp.oom_kind(by["down_r64_S1d_cap128_L49152"], results["down_r64_S1d_cap128_L49152"], FLOOR) == sp.OOM_START_MIN_S
        assert sp.oom_kind(by["mlp_r64_S1d_cap128_L49152"], _oom(by["mlp_r64_S1d_cap128_L49152"], 1), FLOOR) == sp.OOM_START_MIN_S
        assert sp.oom_kind(rec, _oom(rec, 4), FLOOR) == sp.OOM_START_WITHIN_BUDGET        # 196,608 <= the floor: genuine
        assert sp.oom_kind(rec, _oom(rec, 21, ready=120.0), FLOOR) == sp.OOM_SERVING        # the engine was up: genuine
        assert sp.oom_kind(rec, results["down_r4_S1cc_cap8_L4096"], FLOOR) is None          # not an oom
        assert sp.OOM_START_OVER_BUDGET in sp.REPLAN_OOM_KINDS and sp.OOM_START_MIN_S not in sp.REPLAN_OOM_KINDS
        over = {lid for s, r in (("down", 4), ("mlp", 4), ("down", 16), ("mlp", 16)) for lid in _servers(s, r, 49152)}
        assert len(over) == 8
        assert sp.replan_required_launches(plan, results) == [r["id"] for r in plan["launches"] if r["id"] in over]   # plan order
        # the checkpoint carries the budget and the replan set, and is NOT complete while they stand
        prog = sp.progress_record(plan, results)
        assert prog["replan_required"] == sp.replan_required_launches(plan, results) and prog["complete"] is False
        assert prog["batched_token_budget"] == {"method": "record_floor", "tokens": FLOOR, "floor_launch": "attn_mlp_r16_S1cc_cap16_L4096",
                                                "ceiling_tokens": CEILING, "ceiling_launch": "down_r16_S1cc_cap16_L49152"}
        assert prog["plan_batched_token_budget"] is None
        # hpo_probe_status names the re-plan set in its detail (03 reads it before ever fetching a verdict)
        st, detail = hps.probe_state(prog, allowed_present=False, job_active=False)
        assert st == hps.INCOMPLETE_STALLED and "need `run --resume --replan`" in detail and "down_r4_S1cc_cap8_L49152" in detail

    def test_replan_launches_re_derives_the_outstanding_and_the_replan_set_with_stable_ids(self):
        plan = _toy_plan()
        results = _record_results(plan)
        budget = sp.measured_budget(plan, results)
        new_plan, changes = sp.replan_launches(plan, results, budget)
        old = {r["id"]: r for r in plan["launches"]}
        new = {r["id"]: r for r in new_plan["launches"]}
        assert list(old) == list(new)                                           # ids / order stable (no S in an id)
        assert new_plan["batched_token_budget"]["tokens"] == FLOOR
        moved = {c["id"]: c for c in changes}
        exposed = [lid for lid, r in old.items() if r["L"] in (16384, 49152, 65536, 98304)]
        for lid in exposed:
            if lid in RECORD and RECORD[lid][0] == "oom" and RECORD[lid][1] == 1:
                continue                                                          # a genuine S = 1 death: terminal, never re-planned
            assert lid in moved and moved[lid]["new_S"] == new[lid]["admitted_S"] and moved[lid]["new_S"] * new[lid]["L"] <= FLOOR, lid
        for lid in ("down_r4_S1cc_cap8_L49152", "down_r4_S1d_cap8_L49152", "mlp_r16_S1cc_cap16_L49152", "mlp_r16_S1d_cap32_L49152"):
            assert moved[lid]["replan_required"] is True and moved[lid]["old_S"] == 32 and moved[lid]["new_S"] == 5
        # launches that ran to a terminal state and are NOT replan_required keep their recorded S (history)
        for lid, (status, S, _e, _r) in RECORD.items():
            if lid not in moved:
                assert new[lid] == old[lid], lid
        assert "attn_mlp_r64_S1cc_cap64_L49152" not in moved and "down_r16_S1cc_cap16_L4096" not in moved
        # idempotent: the re-derived plan re-derives to itself
        again, changes2 = sp.replan_launches(new_plan, results, budget)
        assert changes2 == [] and again["launches"] == new_plan["launches"]


# --------------------------------------------------------------------------- the verdict


# each pair's three engines at 49,152: the two split servers and the pair's UNION (the served launch class, the probe owner's ruling of
# 2026-10-11 — it carries the same adapters in one engine, so a fixture that kills or passes the servers does the same to it)
DOWN_49152 = ["down_r4_S1cc_cap8_L49152", "down_r4_S1d_cap8_L49152", "down_r4_single_cap8_L49152"]
WO_49152 = ["wo_r4_S1cc_cap8_L49152", "wo_r4_S1d_cap8_L49152", "wo_r4_single_cap8_L49152"]


class TestVerdictClassification:
    def _plan(self):
        # toy dims, down (S1cc + S1d + S2 + the union) and wo (S1cc + S1d + the union), rank 4, two windows: 14 launches
        plan = _toy_plan(sets=("down", "wo"), ranks=(4,), windows=(4096, 49152), ladder_caps=())
        assert plan["n_launches"] == 14
        return plan, {r["id"]: r for r in plan["launches"]}

    def _base_results(self, plan, by):
        def ok(lid, S):
            return _passed(by[lid], sm.Manifest.from_dict(plan["manifests"][by[lid]["manifest"]]), engine_S=S)
        res = {lid: ok(lid, 64) for lid in ("down_r4_S1cc_cap8_L4096", "down_r4_S1d_cap8_L4096", "down_r4_S2_cap8_L4096",
                                            "down_r4_single_cap8_L4096", "wo_r4_S1cc_cap8_L4096", "wo_r4_S1d_cap8_L4096",
                                            "wo_r4_single_cap8_L4096")}
        res["down_r4_S2_cap8_L49152"] = ok("down_r4_S2_cap8_L49152", 5)
        # the defect: the down pair's two servers at 49,152 composed at the requested S 32 (1.57 M tokens) and died at start
        for lid in DOWN_49152:
            res[lid] = _oom(by[lid], 32)
        return res

    @pytest.mark.parametrize("wo_oom,kind", [
        (dict(S=1), sp.OOM_START_MIN_S),                  # died at start at S = 1: nothing smaller exists
        (dict(S=5), sp.OOM_START_WITHIN_BUDGET),          # 245,760 <= the floor: buffers / KV, genuine
        (dict(S=10, ready=90.0), sp.OOM_SERVING),         # the engine was up and OOM'd under load
    ])
    def test_a_prefill_oom_above_the_budget_replans_while_a_genuine_oom_excludes(self, tmp_path, wo_oom, kind):
        plan, by = self._plan()
        results = self._base_results(plan, by)
        for lid in WO_49152:                              # both of the wo pair's servers died the same way at 49,152
            results[lid] = _oom(by[lid], **wo_oom)
        verdict, allowed = sp.write_verdict(plan, results, probe_job_id="1", out=tmp_path)
        # (4, down): NOT excluded — undecided on the replan launches; the verdict is partial and writes no menu
        assert verdict["status"] == "partial" and verdict["certifying"] is False
        assert verdict["replan_required"] == DOWN_49152 and allowed["replan_required"] == DOWN_49152
        assert verdict["undecided_pairs"] == [[4, "down"]]
        down = next(e for e in allowed["excluded"] if e["set"] == "down")
        assert down["decided"] is False and down["reason"] == "not_run"
        assert not any("oom_at_R" in r for r in next(e for e in verdict["excluded"] if e["set"] == "down")["reasons"])
        assert not (tmp_path / sp.ALLOWED_PAIRS_FILE).exists()
        # (4, wo): the genuine OOM excludes, with evidence, exactly as before — the control server's reason leads, the
        # difference server's names its own server (the role-keyed rule), the died servers' S 0 is the ceiling reason, and the
        # pair's union (the served launch class, 2026-10-11) died with them: the class it serves is undominated at 49,152, by name
        wo = next(e for e in allowed["excluded"] if e["set"] == "wo")
        assert wo["decided"] is True and wo["reason"] == "oom_at_R=8_at_L=49152"
        assert next(e for e in verdict["excluded"] if e["set"] == "wo")["reasons"] == \
            ["oom_at_R=8_at_L=49152", "difference_on_S1d:oom_at_R=8_at_L=49152", "ceiling_S<1_at_L=49152",
             "served_class_undominated_at_L=49152:wo_r4_single_cap8_L49152:oom_at_R=8"]
        assert next(e for e in verdict["excluded"] if e["set"] == "wo")["served_class"]["certified_by_window"] == \
            {"4096": "wo_r4_single_cap8_L4096", "49152": None}
        # the launches carry their kind and batch; the budget is the record's, by launch id
        L = {l["id"]: l for l in verdict["launches"]}
        for lid in DOWN_49152:
            assert L[lid]["oom_kind"] == sp.OOM_START_OVER_BUDGET and L[lid]["batched_tokens"] == CEILING
        for lid in WO_49152:
            assert L[lid]["oom_kind"] == kind
        assert L["down_r4_S1cc_cap8_L4096"]["oom_kind"] is None and L["down_r4_S1cc_cap8_L4096"]["batched_tokens"] == FLOOR
        b = verdict["batched_token_budget"]
        assert b["tokens"] == FLOOR and b["floor_launch"] == "down_r4_S1cc_cap8_L4096" and b["ceiling_launch"] == "down_r4_S1cc_cap8_L49152"
        assert verdict["plan_batched_token_budget"]["method"] == "unmeasured"
        # the CLI: exit 4, the RESULT line names the re-plan set and the budget, the memo carries the banner
        sp.write_json_atomic(tmp_path / sp.PLAN_FILE, plan)
        for lid, res in results.items():
            sp.write_json_atomic(sp.launch_result_path(tmp_path, lid), res)
        rc = sp.main(["verdict", "--model-key", MODEL, "--out", str(tmp_path), "--probe-job-id", "1"])
        assert rc == sp.EXIT_INCOMPLETE
        line = (tmp_path / sp.RESULT_FILE).read_text()
        assert line.startswith("state=PARTIAL certifying=false") and "replan_required=3" in line
        assert f"budget=record_floor:{FLOOR}" in line and f"replan={DOWN_49152!r}" in line
        assert sp.main(["memo", "--model-key", MODEL, "--out", str(tmp_path)]) == 0
        memo = (tmp_path / sp.MEMO_FILE).read_text()
        assert "RE-PLAN REQUIRED (D108)" in memo and all(lid in memo for lid in DOWN_49152) and "`run --resume --replan`" in memo

    def test_the_old_reading_is_gone_and_a_re_run_at_the_re_derived_S_certifies(self, tmp_path):
        """Fail-before: the same record read by the pre-fix verdict excluded (4, down) with `oom_at_R=8_at_L=49152` and
        `ceiling_S<1_at_L=49152`. After: undecided; and once the launches re-run at S 5 and pass, the pair certifies."""
        plan, by = self._plan()
        results = self._base_results(plan, by)
        for lid in WO_49152:
            m_wo = sm.Manifest.from_dict(plan["manifests"][by[lid]["manifest"]])
            results[lid] = _passed(by[lid], m_wo, engine_S=5)
        v, allowed = sp.write_verdict(plan, results, probe_job_id="1")
        assert v["status"] == "partial" and [(p["rank"], p["set"]) for p in allowed["pairs"]] == [(4, "wo")]
        assert v["undecided_pairs"] == [[4, "down"]]
        # the re-plan + re-run: S 32 -> 5 on the down pair's three engines, passed
        new_plan, changes = sp.replan_launches(plan, results, sp.measured_budget(plan, results))
        assert [c["id"] for c in changes] == DOWN_49152 and all(c["new_S"] == 5 for c in changes)
        for lid in DOWN_49152:
            rec = next(r for r in new_plan["launches"] if r["id"] == lid)
            results[lid] = _passed(rec, sm.Manifest.from_dict(plan["manifests"][rec["manifest"]]), engine_S=5)
        v, allowed = sp.write_verdict(new_plan, results, probe_job_id="1", out=tmp_path)
        assert v["status"] == "complete" and v["replan_required"] == []
        assert sorted((p["rank"], p["set"]) for p in allowed["pairs"]) == [(4, "down"), (4, "wo")]
        down = next(p for p in allowed["pairs"] if p["set"] == "down")
        assert down["admitted_max_num_seqs"] == {"4096": 64, "49152": 5}
        assert down["topology"] == "S1ccS1dS2" and down["caps_by_server"] == {"S1cc": 8, "S1d": 8}
        wo = next(p for p in allowed["pairs"] if p["set"] == "wo")
        assert wo["topology"] == "S1ccS1d" and wo["admitted_max_num_seqs"] == {"4096": 64, "49152": 5}
        # the geometry record's provenance says how S was sized: the measured budget and the plan's copy of it
        rec_doc = json.loads((tmp_path / sp.GEOMETRY_RECORD_FILE).read_text())
        prov = rec_doc["provenance"]["batched_token_budget"]
        assert prov["measured"]["tokens"] == FLOOR and prov["measured"]["method"] == "record_floor"
        assert prov["measured"]["floor_launch"] == "down_r4_S1cc_cap8_L4096" and prov["plan"]["tokens"] == FLOOR
        assert v["geometry_fold"]["status"] == "record_only"


# --------------------------------------------------------------------------- `run --resume --replan`


class TestReplanRun:
    def _record(self, tmp_path):
        """A toy record the fake engine can serve: the wo r4 pair's three engines (S1cc, S1d and the pair's union — the served
        launch class) at 4,096 passed (S 64 = the floor); at 49,152 all three died at start at S 32."""
        plan = _toy_plan(sets=("wo",), ranks=(4,), windows=(4096, 49152), ladder_caps=())
        by = {r["id"]: r for r in plan["launches"]}
        first = ["wo_r4_S1cc_cap8_L4096", "wo_r4_S1d_cap8_L4096", "wo_r4_single_cap8_L4096"]
        lost = ["wo_r4_S1cc_cap8_L49152", "wo_r4_S1d_cap8_L49152", "wo_r4_single_cap8_L49152"]
        assert [r["id"] for r in plan["launches"]] == first + lost
        assert all(by[l]["admitted_S"] == 64 for l in first) and all(by[l]["admitted_S"] == 32 for l in lost)   # 262,144 and 1,572,864 tokens
        sp.write_json_atomic(tmp_path / sp.PLAN_FILE, plan)
        sp.build_adapter_dirs(plan, tmp_path, FakeExporter())
        for lid in first:
            m = sm.Manifest.from_dict(plan["manifests"][by[lid]["manifest"]])
            sp.write_json_atomic(sp.launch_result_path(tmp_path, lid), _passed(by[lid], m, engine_S=64))
        for lid in lost:
            sp.write_json_atomic(sp.launch_result_path(tmp_path, lid), _oom(by[lid], 32))
            (tmp_path / sp.LAUNCHES_DIR / lid).mkdir(parents=True)
            (tmp_path / sp.LAUNCHES_DIR / lid / sp.SERVER_LOG_FILE).write_text(START_TAIL)
        sp.write_progress(plan, sp.load_results(tmp_path), tmp_path)
        return plan, first, lost

    def test_replan_re_derives_moves_aside_re_runs_only_the_lost_launches_and_is_idempotent(self, tmp_path, monkeypatch, capsys):
        plan, first, lost = self._record(tmp_path)
        prog = sp.load_progress(tmp_path)
        assert prog["complete"] is False and prog["outstanding"] == [] and prog["replan_required"] == lost
        assert prog["batched_token_budget"]["tokens"] == FLOOR and prog["batched_token_budget"]["floor_launch"] == first[0]
        monkeypatch.setattr(sp, "ProductionExporter", lambda *a, **k: FakeExporter())
        deps, seen = _fake_deps(plan)
        monkeypatch.setattr(sp.Deps, "production", classmethod(lambda cls: deps))
        base = ["run", "--model-key", MODEL, "--out", str(tmp_path), "--no-throughput"]
        with pytest.raises(SystemExit, match="--replan re-derives an EXISTING record"):
            sp.main(base + ["--replan"])
        # a plain resume over this record has nothing outstanding to run and exits INCOMPLETE naming the re-plan
        rc = sp.main(base + ["--resume"])
        assert rc == sp.EXIT_INCOMPLETE and seen["stand_up"] == {}
        assert "replan_required" in capsys.readouterr().out
        # --resume --replan: the plan re-derives S 32 -> 5 for the lost launches, moves their results + dirs aside, re-runs
        # exactly them (the pair is certified at 4,096 and its servers and union are already planned at 49,152: the extension adds nothing)
        rc = sp.main(base + ["--resume", "--replan"])
        assert rc == 0
        assert list(seen["stand_up"]) == lost
        new_plan = sp.load_plan(tmp_path)
        by = {r["id"]: r for r in new_plan["launches"]}
        assert [r["id"] for r in new_plan["launches"]] == [r["id"] for r in plan["launches"]]
        assert "extensions" not in new_plan
        for lid in lost:
            assert by[lid]["admitted_S"] == 5 == FLOOR // 49152 and by[lid]["admission"]["binding"] == "batched_tokens"
        for lid in first:
            assert by[lid]["admitted_S"] == 64                                        # history: the passed launches keep their S
        assert new_plan["batched_token_budget"]["tokens"] == FLOOR and new_plan["batched_token_budget"]["floor_launch"] == first[0]
        assert new_plan["batched_token_budget"]["method"] == "record_floor" and "replanned_utc" in new_plan
        sup = sorted(p.name for p in (tmp_path / sp.LAUNCHES_DIR / sp.SUPERSEDED_DIR).iterdir())
        assert len(sup) == 6 and all(any(n.startswith(lid + ".") for lid in lost) for n in sup)
        assert sum(1 for n in sup if n.endswith(".json")) == 3
        for p in (tmp_path / sp.LAUNCHES_DIR / sp.SUPERSEDED_DIR).glob("*.json"):
            old = json.loads(p.read_text())
            assert old["status"] == "oom" and old["launch_S"] == 32                   # nothing deleted
        res = sp.load_results(tmp_path)
        for lid in lost:
            assert res[lid]["status"] == "passed" and res[lid]["launch_S"] == 5
        prog = sp.load_progress(tmp_path)
        assert prog["complete"] is True and prog["replan_required"] == [] and prog["plan_batched_token_budget"] == FLOOR
        # idempotent: a second --resume --replan changes nothing, moves nothing, stands nothing up
        plan_bytes = (tmp_path / sp.PLAN_FILE).read_bytes()
        deps2, seen2 = _fake_deps(plan)
        monkeypatch.setattr(sp.Deps, "production", classmethod(lambda cls: deps2))
        rc = sp.main(base + ["--resume", "--replan"])
        assert rc == 0 and seen2["stand_up"] == {}
        assert (tmp_path / sp.PLAN_FILE).read_bytes() == plan_bytes
        assert sorted(p.name for p in (tmp_path / sp.LAUNCHES_DIR / sp.SUPERSEDED_DIR).iterdir()) == sup
        # the verdict now certifies the pair at the re-derived S and folds the budget into the geometry record
        monkeypatch.setenv("SISTEER_GEOMETRY_DIR", str(tmp_path / "geom"))
        rc = sp.main(["verdict", "--model-key", MODEL, "--out", str(tmp_path), "--probe-job-id", "2"])
        assert rc == 0
        allowed = json.loads((tmp_path / sp.ALLOWED_PAIRS_FILE).read_text())
        assert [(p["rank"], p["set"]) for p in allowed["pairs"]] == [(4, "wo")] and allowed["replan_required"] == []
        # the menu carries the ENGINE's resolved S where a launch ran (the fake engine reports 4), the plan's 5 is what it was launched at
        assert allowed["pairs"][0]["admitted_max_num_seqs"] == {"4096": 64, "49152": 4}
        assert allowed["pairs"][0]["topology"] == "S1ccS1d" and allowed["pairs"][0]["caps_by_server"] == {"S1cc": 8, "S1d": 8}
        prov = json.loads((tmp_path / "geom" / f"{MODEL}.json").read_text())["provenance"]["batched_token_budget"]
        assert prov["measured"]["tokens"] == FLOOR and prov["plan"]["tokens"] == FLOOR and prov["plan"]["floor_launch"] == first[0]
        assert f"budget=record_floor:{FLOOR}" in (tmp_path / sp.RESULT_FILE).read_text()

    def test_replan_with_no_measured_floor_re_derives_nothing_and_says_so(self, tmp_path, monkeypatch, capsys):
        plan = _toy_plan(sets=("wo",), ranks=(4,), windows=(4096, 49152), ladder_caps=())
        by = {r["id"]: r for r in plan["launches"]}
        second = "wo_r4_S1cc_cap8_L49152"
        sp.write_json_atomic(tmp_path / sp.PLAN_FILE, plan)
        sp.write_json_atomic(sp.launch_result_path(tmp_path, second), _oom(by[second], 32))
        results = sp.load_results(tmp_path)
        assert sp.measured_budget(plan, results)["tokens"] is None
        rp = sp.replan_record(tmp_path, plan, results)
        assert rp["budget"]["method"] == "unmeasured" and rp["changes"] == [] and rp["superseded"] == []
        assert rp["replan_required"] == [second] and rp["plan_written"] is False
        assert sp.load_plan(tmp_path)["launches"] == plan["launches"]                 # untouched
        assert not (tmp_path / sp.LAUNCHES_DIR / sp.SUPERSEDED_DIR).exists()
        assert "budget UNMEASURED" in capsys.readouterr().out
        prog = sp.progress_record(plan, results)
        assert prog["complete"] is False and prog["replan_required"] == [second] and prog["batched_token_budget"]["tokens"] is None


# --------------------------------------------------------------------------- the carrier


def test_the_carrier_resumes_with_replan_and_documents_it():
    text = SBATCH.read_text()
    assert subprocess.run(["bash", "-n", str(SBATCH)], capture_output=True, text=True).returncode == 0
    assert text.count("RESUME=(--resume --replan)") == 2 and "RESUME=(--resume)\n" not in text
    assert "`run --resume --replan`" in text and "D108" in text
    assert re.search(r"^[^#\n]*--export=", text, re.M) is None
    assert_no_tokens(text, str(SBATCH))
