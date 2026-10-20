"""The serving probe over LEGACY and MIXED records (the 2026-10-01 topology change): the role-keyed verdict over a record
written with the four-slot ``S1`` (+ its S1a/S1b/S1c fallbacks) and ``S2``, the reopen's plan extension with the current
``S1cc`` / ``S1d`` servers, the parity reading by adapter name, the record's memory of its parity record, the reopen receipt a
COMPLETE record's extension needs, the parity record's at-rest reading on a plan mismatch, and the carrier's attempt head.
CPU only: the fixtures are hand-built on the worked example's Qwen3.5-4B dims (``QWEN35_4B_DIMS``, the dims table's one
onboarded entry), a dense toy and a routed-experts toy (the MoE serving option: a legacy MoE record's ``moe_lora_rank``
derives from its slots, the extension's new servers carry theirs); synthetic ids; no cluster path, no host name.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

from sisteer.capeval.serving import standup_manifest as sm
from sisteer.extraction.lora_modules import QWEN35_4B_DIMS as Q4
from sisteer.extraction.lora_modules import ArchDims

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts", REPO / "scripts" / "sdfbase"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import serving_probe as sp  # noqa: E402

#: a dense toy: hidden 64, one attention block (q_out 128, kv_out 16, o_in 64), two dense MLP blocks of intermediate 32
TOY = ArchDims(hidden=64, attention_layers=1, q_out=128, kv_out=16, o_in=64, dense_mlp_layers=2, dense_intermediate=32)
#: a 2-MoE-block toy: the same attention block, 4 routed experts (2 active) of I = 32 plus a shared expert of 32
MOE_TOY = ArchDims(hidden=64, attention_layers=1, q_out=128, kv_out=16, o_in=64, moe_layers=2,
                   num_experts=4, active_experts=2, moe_intermediate=32, shared_intermediate=32)
REV = "0123456789abcdef0123456789abcdef01234567"
#: the worked example's served key — the dims table's one onboarded entry (lora_modules.ARCH_DIMS_BY_MODEL_KEY)
MODEL = "qwen35_4b_demo"


# --------------------------------------------------------------------------- the probe: legacy / mixed records, the extension


def _legacy_pair_plan(dims=Q4, set_id="mlp", rank=64, windows=(4096, 49152), fallbacks=True) -> Dict[str, Any]:
    """A record written BEFORE the topology change: the four-slot S1 (+ the S1a/S1b/S1c fallbacks) and S2, hand-built. On a
    routed-experts dims every slot states its per-expert rank the way the older records did (FD 6's rule on the slot's
    own r, the zero adapter's on the cap) and the record's bytes law priced the experts at the CAP (vLLM's own sizing);
    the record carries no ``moe_lora_rank`` (the key did not exist), so the readers derive it."""
    modules = sm.MODULE_SETS[set_id]
    cap = sm.rank_cap(2 * rank)
    target = sm.target_class_for(dims, modules)
    routed = sm.adapts_routed_experts(dims, modules)
    r_e = (lambda r: sm.per_expert_rank(r, dims.active_experts)) if routed else (lambda r: None)  # noqa: E731
    def slot(name, kind, r, role):
        return sm.AdapterSlot(name=name, kind=kind, lora_modules=modules, rank=r, role=role, per_expert_rank=r_e(r))
    cells = (slot("control__lora", "genuine", rank, "control"), slot("constitution__lora", "genuine", rank, "constitution"),
             slot("difference__lora", "genuine", 2 * rank, "difference"))
    zero = lambda c: slot("none__lora0", "zero", c, "none")  # noqa: E731
    s1 = sm.Manifest(name=f"{set_id}_r{rank}_S1", set_id=set_id, rank=rank, topology="S1", adapters=cells + (zero(cap),),
                     target_class=target, max_loras=4, max_lora_rank=cap)
    ms = [s1, sm.build_s2_manifest(set_id, rank)]
    if fallbacks:
        for cell in cells:
            top = sm.S1_FALLBACK_BY_ROLE[cell.role]
            fcap = sm.rank_cap(cell.rank)
            ms.append(sm.Manifest(name=f"{set_id}_r{rank}_{top}", set_id=set_id, rank=rank, topology=top,
                                  adapters=(cell, zero(fcap)), target_class=target, max_loras=2, max_lora_rank=fcap,
                                  fallback_of=s1.name))
    budget = {"base_weights_bytes": sm.DEFAULT_BASE_WEIGHTS_BYTES, "graphs_activations_bytes": sm.DEFAULT_GRAPHS_ACTIVATIONS_BYTES,
              "gpu_memory_utilization": 0.9, "total_vram_bytes": sm.H200_TOTAL_BYTES, "recurrent_state_per_seq_bytes": 0}
    by_name = {m.name: m for m in ms}
    launches = [sp._launch_record(m, L, "fallback" if m.topology in sm.S1_FALLBACKS else "natural", sp._launch_priority(m, L, 4096),
                                  dims=dims, tp=1, requested_S=sp.REQUESTED_S_BY_WINDOW[L], budget_tokens=None, budget=budget,
                                  manifests_by_name=by_name) for m in ms for L in windows]
    for rec in launches:
        m = by_name[rec["manifest"]]
        # the legacy law: the experts were at the cap (vLLM's own sizing) — the record says so (a dense dims: unchanged)
        rec["expected_lora_bytes"] = sm.expected_lora_bytes(m, m.max_lora_rank, dims, moe_lora_rank=m.max_lora_rank if m.wraps_experts else None)
        rec.pop("roles", None)                      # a record written before the plan recorded the roles a launch serves
        rec.pop("moe_lora_rank", None)              # ...or the expert-buffer rank (added later)
    import dataclasses as dc
    return {"version": 1, "written_utc": sp.utc_now(), "model_key": MODEL, "model_id": "org/toy", "served_hf_id": "org/toy",
            "revision": REV, "dims": dc.asdict(dims), "is_moe": bool(dims.moe_layers), "sets": [set_id], "ranks": [rank],
            "windows": list(windows),
            "ladder_window": 4096, "ladder_caps": [], "tp": 1, "requested_S_by_window": {str(k): v for k, v in sp.REQUESTED_S_BY_WINDOW.items()},
            "budget": budget, "synthetic_scale": 100.0,
            "pairs": [{"set": set_id, "rank": rank, "manifests": [m.name for m in ms], "trainable_params": sm.trainable_params(dims, set_id, rank),
                       "per_expert_rank": sm.pair_per_expert_rank(dims, set_id, rank), "lora_modules": list(modules), "natural_cap": cap}],
            "manifests": {m.name: {k: v for k, v in m.as_dict().items() if k != "moe_lora_rank"} for m in ms},
            "synthetic_scale_by_set": None, "parity_pass": False, "batched_token_budget": {"method": "unmeasured", "tokens": None},
            "launches": launches, "n_launches": len(launches)}


REF_OK = {"status": "pass", "basis": sp.BASIS_HF_PEFT,
          "per_adapter": {"control__lora": {"status": "pass"}, "constitution__lora": {"status": "pass"}, "difference__lora": {"status": "pass"}}}


def _passed(rec, m, *, S: int = 4, reference=REF_OK, dims=Q4):
    per = {a.name: {"kind": a.kind, "role": a.role, "rank": a.rank, "pass": True} for a in m.adapters}
    D = sm.packed_stride_ceiling(dims, m.target_class)["D"]
    return {"launch_id": rec["id"], "manifest": m.name, "serves": [m.name], "set": m.set_id, "rank": m.rank, "topology": m.topology,
            "status": "passed", "ready_s": 100.0, "engine": {"max_num_seqs_engine": S}, "launch_S": S,
            "checks": {"pass": True, "failed": [], "adapters": per, "executing_stride": {"status": "measured", "reported_D": D},
                       "buffer_bytes": {"pass": True}}, "reference_check": reference}


def _dead(rec, status="oom", S=None):
    """A start-time death at ``S`` (default: the plan's admitted S) — inside the measured floor, so the OOM is genuine,
    never a replan."""
    return {"launch_id": rec["id"], "status": status, "ready_s": None, "launch_S": (int(S) if S else max(1, rec["admitted_S"])),
            "failure": {"class": status, "message": "cuda_out_of_memory" if status == "oom" else "refused:KvCoverageShortfall"}}


def _by_id(plan):
    return {r["id"]: r for r in plan["launches"]}


def _s2_passed(plan, S=64):
    return {r["id"]: _passed(r, sp.plan_manifest(plan, r["manifest"]), S=S) for r in plan["launches"] if r["topology"] == "S2"}


class TestRoleKeyedVerdict:
    def test_a_legacy_record_reads_as_before_s1_certifies_every_role(self):
        plan = _legacy_pair_plan(windows=(4096, 49152), fallbacks=False)
        by = _by_id(plan)
        m = sp.plan_manifest(plan, "mlp_r64_S1")
        results = {"mlp_r64_S1_cap128_L4096": _passed(by["mlp_r64_S1_cap128_L4096"], m, S=2),
                   "mlp_r64_S1_cap128_L49152": _passed(by["mlp_r64_S1_cap128_L49152"], m, S=1), **_s2_passed(plan)}
        v, allowed = sp.write_verdict(plan, results, probe_job_id="1")
        p = allowed["pairs"][0]
        assert p["topology"] == "S1S2" and p["caps_by_server"] == {"S1": 128}
        assert p["moe_lora_rank_by_server"] == {"S1": None} and p["per_expert_rank"] is None      # a dense model: no expert rank anywhere
        assert p["admitted_max_num_seqs"] == {"4096": 2, "49152": 1}
        assert p["certifying_by_window"] == {"4096": {"control": "mlp_r64_S1_cap128_L4096", "difference": "mlp_r64_S1_cap128_L4096"},
                                             "49152": {"control": "mlp_r64_S1_cap128_L49152", "difference": "mlp_r64_S1_cap128_L49152"}}
        assert v["status"] == "complete" and v["serving_certified_pairs"] == [[64, "mlp"]]

    def test_the_live_records_shape_s1_oom_fallbacks_a_and_b_pass_c_dies_excludes_the_difference_by_name(self):
        plan = _legacy_pair_plan(windows=(4096, 49152))
        by = _by_id(plan)
        mf = lambda n: sp.plan_manifest(plan, n)  # noqa: E731
        results = {"mlp_r64_S1_cap128_L4096": _passed(by["mlp_r64_S1_cap128_L4096"], mf("mlp_r64_S1"), S=2),
                   "mlp_r64_S1_cap128_L49152": _dead(by["mlp_r64_S1_cap128_L49152"], S=5),
                   "mlp_r64_S1a_cap64_L49152": _passed(by["mlp_r64_S1a_cap64_L49152"], mf("mlp_r64_S1a"), S=5),
                   "mlp_r64_S1b_cap64_L49152": _passed(by["mlp_r64_S1b_cap64_L49152"], mf("mlp_r64_S1b"), S=5),
                   "mlp_r64_S1c_cap128_L49152": _dead(by["mlp_r64_S1c_cap128_L49152"], S=5), **_s2_passed(plan)}
        for lid in ("mlp_r64_S1a_cap64_L4096", "mlp_r64_S1b_cap64_L4096", "mlp_r64_S1c_cap128_L4096"):
            results[lid] = {"launch_id": lid, "status": "skipped"}
        v, allowed = sp.write_verdict(plan, results, probe_job_id="1")
        assert allowed["pairs"] == []
        e = allowed["excluded"][0]
        assert e["decided"] is True and e["reason"] == "difference_on_S1c:oom_at_R=128_at_L=49152"
        assert "ceiling_S<1_at_L=49152" in v["excluded"][0]["reasons"]
        # the role-keyed status the extension reads: 4096 certified by S1 (S 2), 49152 not (the difference died)
        st = sp.pair_serving_status(plan, results, plan["pairs"][0])
        assert st["windows"][4096] == {"certified": True, "measured_S": 2, "undecided": False,
                                       "by_role": {"control": "passed", "difference": "passed"}}
        assert st["windows"][49152]["certified"] is False and st["windows"][49152]["by_role"] == {"control": "passed", "difference": "down"}
        # ...and the legacy fallback wording at the LADDER window when S1 and S1c both die there
        results2 = dict(results)
        results2["mlp_r64_S1_cap128_L4096"] = _dead(by["mlp_r64_S1_cap128_L4096"], S=2)
        results2["mlp_r64_S1a_cap64_L4096"] = _passed(by["mlp_r64_S1a_cap64_L4096"], mf("mlp_r64_S1a"), S=8)
        results2["mlp_r64_S1c_cap128_L4096"] = _dead(by["mlp_r64_S1c_cap128_L4096"], S=2)
        _v, allowed2 = sp.write_verdict(plan, results2, probe_job_id="1")
        assert "fallback_S1c:oom_at_R=128" in allowed2["excluded"][0]["reason"]

    def test_the_reopen_extends_the_plan_with_the_current_servers_where_the_pair_is_uncertified_or_starved(self):
        plan = _legacy_pair_plan(windows=(4096, 49152))
        by = _by_id(plan)
        mf = lambda n: sp.plan_manifest(plan, n)  # noqa: E731
        results = {"mlp_r64_S1_cap128_L4096": _passed(by["mlp_r64_S1_cap128_L4096"], mf("mlp_r64_S1"), S=2),
                   "mlp_r64_S1_cap128_L49152": _dead(by["mlp_r64_S1_cap128_L49152"], S=5),
                   "mlp_r64_S1a_cap64_L49152": _passed(by["mlp_r64_S1a_cap64_L49152"], mf("mlp_r64_S1a"), S=5),
                   "mlp_r64_S1b_cap64_L49152": _passed(by["mlp_r64_S1b_cap64_L49152"], mf("mlp_r64_S1b"), S=5),
                   "mlp_r64_S1c_cap128_L49152": _dead(by["mlp_r64_S1c_cap128_L49152"], S=5), **_s2_passed(plan, S=64)}
        for lid in ("mlp_r64_S1a_cap64_L4096", "mlp_r64_S1b_cap64_L4096", "mlp_r64_S1c_cap128_L4096"):
            results[lid] = {"launch_id": lid, "status": "skipped"}
        new_plan, ext = sp.extend_plan_for_uncertified_pairs(plan, results, Q4, tree_sha="t" * 40)
        assert ext["pairs"] == [[64, "mlp"]] and ext["manifests"] == ["mlp_r64_S1cc", "mlp_r64_S1d"]
        # 4096: certified by S1 at S = 2, but the current servers admit 28 there (the packed gate_up stride, 2 x 9,216, binds
        # under the floor S2's 64 x 4,096 sets): starved -> added
        # 49152: the difference is uncertified -> added
        assert sorted(ext["added"]) == ["mlp_r64_S1cc_cap64_L4096", "mlp_r64_S1cc_cap64_L49152",
                                        "mlp_r64_S1d_cap128_L4096", "mlp_r64_S1d_cap128_L49152"]
        assert ext["why"]["mlp_r64_S1cc_cap64_L4096"] == "measured_S=2<planned_S=28" and ext["why"]["mlp_r64_S1d_cap128_L49152"] == "role_uncertified"
        assert new_plan["n_launches"] == plan["n_launches"] + 4 and new_plan["extensions"][0]["tree_sha"] == "t" * 40
        assert "mlp_r64_S1cc" in new_plan["pairs"][0]["manifests"] and new_plan["manifests"]["mlp_r64_S1d"]["max_lora_rank"] == 128
        assert new_plan["manifests"]["mlp_r64_S1d"]["moe_lora_rank"] is None                   # a dense class states no expert rank
        added = {r["id"]: r for r in new_plan["launches"] if r["id"] in ext["added"]}
        assert added["mlp_r64_S1cc_cap64_L4096"]["cap"] == 64 and added["mlp_r64_S1cc_cap64_L4096"]["roles"] == ["control", "constitution"]
        assert added["mlp_r64_S1cc_cap64_L4096"]["moe_lora_rank"] is None
        # the difference server's dense gate_up + down buffers at cap 128 over the 32 blocks, two slots
        assert added["mlp_r64_S1d_cap128_L49152"]["expected_lora_bytes"]["total"] == 2 * 32 * ((2 * 128 * 2560 + 2 * 9216 * 128) + (128 * 9216 + 2560 * 128)) * 2
        # idempotent: a second extension adds nothing; the original launches are untouched
        again, ext2 = sp.extend_plan_for_uncertified_pairs(new_plan, results, Q4)
        assert ext2["added"] == [] and again["n_launches"] == new_plan["n_launches"]
        assert [r["id"] for r in new_plan["launches"] if r["id"] in by] and all(by[r["id"]] == r for r in new_plan["launches"] if r["id"] in by)
        # a parity-pass plan extends on uncertified roles alone (never on admission)
        pp = json.loads(json.dumps(plan)); pp["parity_pass"] = True; pp["windows"] = [4096]
        pp["launches"] = [r for r in pp["launches"] if r["L"] == 4096]; pp["n_launches"] = len(pp["launches"])
        _np, ext3 = sp.extend_plan_for_uncertified_pairs(pp, results, Q4)
        assert ext3["added"] == []
        # once the new servers PASS where the legacy ones died, the MIXED record's ROLES certify: legacy S1 at 4096 is out-ranked by
        # the current servers (first in order), the difference at 49152 comes from S1d — and the pair is still EXCLUDED, decided
        # by the served launch class: the (128, 4) class a steered standup of (64, mlp) serves was
        # certified at 4,096 by the legacy S1 and DIED at 49,152 with it; the split servers (64, 3) and (128, 2) never dominate it,
        # the extension adds no second copy of the shape that died there, and `least_admitted_by_window` would refuse the standup at
        # 49,152 — the verdict says so first, by the window and the launch
        res2 = dict(results)
        for lid in ext["added"]:
            rec = added[lid]
            res2[lid] = _passed(rec, sp.plan_manifest(new_plan, rec["manifest"]), S=(64 if rec["L"] == 4096 else 5))
        v, allowed = sp.write_verdict(new_plan, res2, probe_job_id="2")
        assert v["status"] == "complete" and allowed["pairs"] == [] and v["serving_certified_pairs"] == [] and v["undecided_pairs"] == []
        st2 = sp.pair_serving_status(new_plan, res2, new_plan["pairs"][0])
        assert st2["certified_all"] and st2["windows"][49152]["by_role"] == {"control": "passed", "difference": "passed"}
        e = v["excluded"][0]
        assert e["reasons"] == [f"{sp.REASON_SERVED_CLASS}_at_L=49152:mlp_r64_S1_cap128_L49152:oom_at_R=128"] and e["decided"] is True
        assert e["topology"] == "S1ccS1dS2" and e["caps_by_server"] == {"S1cc": 64, "S1d": 128} and e["max_lora_rank_cap"] == 128
        assert e["moe_lora_rank_by_server"] == {"S1cc": None, "S1d": None}
        assert e["certifying_by_window"]["49152"] == {"control": "mlp_r64_S1cc_cap64_L49152", "difference": "mlp_r64_S1d_cap128_L49152"}
        assert e["certifying_by_window"]["4096"] == {"control": "mlp_r64_S1cc_cap64_L4096", "difference": "mlp_r64_S1d_cap128_L4096"}
        # the class's own numbers ride the trail: at 4,096 the legacy S1 certified the class at
        # its engine's S 2 — what a standup there is sized from — while the current servers that certify the ROLES there admit 64;
        # nothing where the class died; this fixture records no throughput
        assert e["served_class"] == {"max_lora_rank": 128, "max_loras": 4, "target_class": ["down_proj", "gate_proj", "up_proj"], "manifest": None,
                                     "certified_by_window": {"4096": "mlp_r64_S1_cap128_L4096", "49152": None},
                                     "measured_S_by_window": {"4096": 2, "49152": None}, "throughput_by_window": {"4096": None, "49152": None}}
        assert res2["mlp_r64_S1cc_cap64_L4096"]["engine"]["max_num_seqs_engine"] == 64 > 2        # the roles' S is not the class's
        assert sp.extend_plan_for_uncertified_pairs(new_plan, res2, Q4)[1]["added"] == []
        # had the legacy S1 come up at 49,152 too, the same record certifies: the class is covered by S1 at both windows
        res3 = dict(res2)
        res3["mlp_r64_S1_cap128_L49152"] = _passed(by["mlp_r64_S1_cap128_L49152"], mf("mlp_r64_S1"), S=1)
        v3, allowed3 = sp.write_verdict(new_plan, res3, probe_job_id="2")
        p = allowed3["pairs"][0]
        assert v3["status"] == "complete" and p["topology"] == "S1ccS1dS2" and p["caps_by_server"] == {"S1cc": 64, "S1d": 128}
        assert p["admitted_max_num_seqs"] == {"4096": 64, "49152": 5}
        assert p["certifying_by_window"]["49152"] == {"control": "mlp_r64_S1cc_cap64_L49152", "difference": "mlp_r64_S1d_cap128_L49152"}
        assert p["served_class"]["certified_by_window"] == {"4096": "mlp_r64_S1_cap128_L4096", "49152": "mlp_r64_S1_cap128_L49152"}

    def test_a_legacy_moe_record_derives_its_expert_rank_and_the_extension_carries_the_new_servers_ranks(self):
        """The same two readings on a routed-experts toy (the MoE serving option): a legacy four-slot S1 at cap 128 carries
        no ``moe_lora_rank`` key, so the manifest derives it from its slots (the zero adapter's 128 // 2 = 64 = the difference's
        r_e(128)); its launch records priced the experts at the cap (the legacy law) and KEEP that pricing; the verdict reports
        the derived rank per server; the reopen's S1cc / S1d manifests and launches carry THEIR ranks (32: the single-rank
        adapters' and the zero's 64 // 2; 64: the difference's 2 x 32 and the zero's 128 // 2) and price the experts there."""
        plan = _legacy_pair_plan(dims=MOE_TOY, windows=(4096, 49152), fallbacks=False)
        assert plan["is_moe"] is True and plan["pairs"][0]["per_expert_rank"] == {"shared_expert": 64, "routed_expert": 32}
        assert "moe_lora_rank" not in plan["manifests"]["mlp_r64_S1"] and all("moe_lora_rank" not in r for r in plan["launches"])
        s1 = sp.plan_manifest(plan, "mlp_r64_S1")
        assert s1.wraps_experts and s1.moe_lora_rank == 64 and [a.per_expert_rank for a in s1.adapters] == [32, 32, 64, 64]
        by = _by_id(plan)
        legacy_bytes = by["mlp_r64_S1_cap128_L4096"]["expected_lora_bytes"]
        assert legacy_bytes["moe_lora_rank"] == 128 and legacy_bytes["rank_by_family"]["routed_experts"] == 128
        assert legacy_bytes["per_slot"]["routed_experts"] == 2 * 4 * 128 * (64 + 64 + 32 + 64) * 2
        results = {"mlp_r64_S1_cap128_L4096": _passed(by["mlp_r64_S1_cap128_L4096"], s1, S=2, dims=MOE_TOY),
                   "mlp_r64_S1_cap128_L49152": _dead(by["mlp_r64_S1_cap128_L49152"], S=5),
                   **{r["id"]: _passed(r, sp.plan_manifest(plan, r["manifest"]), S=64, dims=MOE_TOY) for r in plan["launches"] if r["topology"] == "S2"}}
        v, allowed = sp.write_verdict(plan, results, probe_job_id="1")
        assert v["status"] == "complete" and allowed["pairs"] == []                 # the difference died at 49,152: decided
        new_plan, ext = sp.extend_plan_for_uncertified_pairs(plan, results, MOE_TOY)
        assert ext["manifests"] == ["mlp_r64_S1cc", "mlp_r64_S1d"]
        cc, d = sp.plan_manifest(new_plan, "mlp_r64_S1cc"), sp.plan_manifest(new_plan, "mlp_r64_S1d")
        assert (cc.moe_lora_rank, d.moe_lora_rank) == (32, 64) and new_plan["manifests"]["mlp_r64_S1cc"]["moe_lora_rank"] == 32
        added = {r["id"]: r for r in new_plan["launches"] if r["id"] in ext["added"]}
        assert added["mlp_r64_S1cc_cap64_L4096"]["moe_lora_rank"] == 32 and added["mlp_r64_S1d_cap128_L49152"]["moe_lora_rank"] == 64
        assert added["mlp_r64_S1d_cap128_L49152"]["expected_lora_bytes"]["rank_by_family"] == {"shared_expert": 128, "routed_experts": 64}
        assert added["mlp_r64_S1d_cap128_L49152"]["expected_lora_bytes"]["per_slot"]["routed_experts"] == legacy_bytes["per_slot"]["routed_experts"] // 2
        # the original launches keep the legacy law byte for byte
        assert all(by[r["id"]] == r for r in new_plan["launches"] if r["id"] in by)
        res2 = dict(results)
        for lid, rec in added.items():
            res2[lid] = _passed(rec, sp.plan_manifest(new_plan, rec["manifest"]), S=(64 if rec["L"] == 4096 else 5), dims=MOE_TOY)
        v, allowed = sp.write_verdict(new_plan, res2, probe_job_id="2")
        # the roles certify on the new servers at their ranks — and the pair stays EXCLUDED, decided, for the (128, 4) class that
        # died with the legacy S1 at 49,152 (the served launch class); the per-server ranks ride the
        # excluded pair's audit trail
        assert v["status"] == "complete" and allowed["pairs"] == [] and v["serving_certified_pairs"] == []
        p = v["excluded"][0]
        assert p["reasons"] == [f"{sp.REASON_SERVED_CLASS}_at_L=49152:mlp_r64_S1_cap128_L49152:oom_at_R=128"] and p["decided"] is True
        assert p["topology"] == "S1ccS1dS2" and p["caps_by_server"] == {"S1cc": 64, "S1d": 128}
        assert p["moe_lora_rank_by_server"] == {"S1cc": 32, "S1d": 64} and p["per_expert_rank"] == {"shared_expert": 64, "routed_expert": 32}
        assert p["served_class"]["certified_by_window"] == {"4096": "mlp_r64_S1_cap128_L4096", "49152": None}
        assert p["served_class"]["target_class"] == ["down_proj", "experts", "gate_proj", "up_proj"]
        # the memo's expert law comes from the dims, per slot per unit of moe_lora_rank, beside the per-class law
        mi = v["memo_inputs"]
        assert mi["buffer_law"]["expert_bytes_per_R_per_slot"] == 2 * 4 * (64 + 64 + 32 + 64) * 2
        assert mi["native_support_per_class"]["down_proj,experts,gate_proj,up_proj"]["wraps_experts"] is True
        assert mi["native_support_per_class"]["o_proj"]["wraps_experts"] is False
        text = sp.render_memo(v)
        assert "| wraps experts |" in text and "the routed-expert family alone" in text

    def test_the_parity_record_is_read_by_adapter_name_when_it_never_planned_the_new_manifest(self):
        plan = _legacy_pair_plan(windows=(4096,), fallbacks=False)
        # the legacy server came up STARVED (S = 1 against the 28 the current servers admit under S2's floor): the extension adds
        # S1cc / S1d on admission, the pair's served class (128, 4) stays covered by the S1 (a dead S1 would exclude the pair for the
        # class instead — the served-class pins of the neighbouring tests), and this test keeps to its own question: the parity
        # record is read by ADAPTER NAME for manifests it never planned
        results = {"mlp_r64_S1_cap128_L4096": _passed(_by_id(plan)["mlp_r64_S1_cap128_L4096"], sp.plan_manifest(plan, "mlp_r64_S1"), S=1),
                   **_s2_passed(plan, S=64)}
        new_plan, ext = sp.extend_plan_for_uncertified_pairs(plan, results, Q4)
        assert ext["manifests"] == ["mlp_r64_S1cc", "mlp_r64_S1d"] and ext["why"]["mlp_r64_S1cc_cap64_L4096"].startswith("measured_S=1<planned_S=")
        added = {r["id"]: r for r in new_plan["launches"] if r["id"] in ext["added"]}
        res = {**results, **{lid: _passed(rec, sp.plan_manifest(new_plan, rec["manifest"]), S=64, reference=None) for lid, rec in added.items()}}
        res.update({r["id"]: _passed(r, sp.plan_manifest(new_plan, r["manifest"]), S=64, reference=None)
                    for r in new_plan["launches"] if r["topology"] == "S2"})
        # the parity record ran the LEGACY four-slot S1 and judged the three adapters by name
        parity_plan = _legacy_pair_plan(windows=(4096,), fallbacks=False)
        parity_plan["parity_pass"] = True
        parity_plan["synthetic_scale_by_set"] = {"mlp": 0.05}
        prec = _by_id(parity_plan)["mlp_r64_S1_cap128_L4096"]
        judged = {"status": "fail", "basis": sp.BASIS_HF_PEFT, "undecidable": {},
                  "per_adapter": {"control__lora": {"status": "pass"}, "constitution__lora": {"status": "pass"},
                                  "difference__lora": {"status": "fail"}}}
        parity = {"dir": "/parity", "plan": parity_plan, "results": {prec["id"]: _passed(prec, sp.plan_manifest(parity_plan, "mlp_r64_S1"), reference=judged)},
                  "generation": {"generation": "a" * 64, "n_results": 1}}
        sources = dict(sp.parity_reference_sources(new_plan, [added["mlp_r64_S1cc_cap64_L4096"]["id"], added["mlp_r64_S1d_cap128_L4096"]["id"]], parity))
        cc_label = next(k for k in sources if "mlp_r64_S1cc" in k)
        d_label = next(k for k in sources if "mlp_r64_S1d" in k)
        assert sources[cc_label]["status"] == "pass" and sources[cc_label]["restricted_to"] == ["control__lora", "constitution__lora"]
        assert sources[d_label]["status"] == "fail" and sources[d_label]["donor_launch"] == prec["id"]
        v, allowed = sp.write_verdict(new_plan, res, probe_job_id="3", parity=parity)
        assert allowed["pairs"] == [] and allowed["excluded"][0]["reason"] == "reference_mismatch:difference__lora"
        assert v["serving_certified_pairs"] == [[64, "mlp"]]       # Serving certified, parity a separate fact
        # with no judged parity launch of that (set, rank) the pair is reference_missing, exactly as before
        empty = {**parity, "results": {}}
        v2, allowed2 = sp.write_verdict(new_plan, res, probe_job_id="3", parity=empty)
        assert v2["status"] == "uncertified" and allowed2["excluded"][0]["reason"].startswith("reference_missing:")


class TestRecordMemory:
    def test_the_plan_remembers_its_parity_record_and_verdict_defaults_to_it(self, tmp_path, monkeypatch):
        plan = _legacy_pair_plan(windows=(4096,), fallbacks=False)
        sp.write_json_atomic(tmp_path / sp.PLAN_FILE, plan)
        (tmp_path / sp.VERDICT_FILE).write_text(json.dumps({sp.PARITY_RECORD_KEY: {"dir": "/some/parity"}}))
        assert sp._remember_parity_record(tmp_path, plan) == "/some/parity"
        assert sp.load_plan(tmp_path)["parity_record_dir"] == "/some/parity"
        # an explicit dir is remembered when the plan has none, and an explicit dir REPLACES the one the plan carries (round 2 on
        # 83da932b8: the operator's `--parity-record B` is durable — this line pinned the write-once rule for the explicit binding
        # until the finding; the implicit read of the standing verdict's stamp still never overwrites a remembered dir)
        plan2 = _legacy_pair_plan(windows=(4096,), fallbacks=False)
        sp.write_json_atomic(tmp_path / "b" / sp.PLAN_FILE, plan2)
        assert sp._remember_parity_record(tmp_path / "b", plan2, parity_dir="/x") == "/x"
        assert sp._remember_parity_record(tmp_path / "b", sp.load_plan(tmp_path / "b"), parity_dir="/y") == "/y"
        assert sp.load_plan(tmp_path / "b")["parity_record_dir"] == "/y"
        (tmp_path / "b" / sp.VERDICT_FILE).write_text(json.dumps({sp.PARITY_RECORD_KEY: {"dir": "/z"}}))
        assert sp._remember_parity_record(tmp_path / "b", sp.load_plan(tmp_path / "b")) == "/y"                    # implicit: write-once
        assert sp._remember_parity_record(tmp_path / "b", sp.load_plan(tmp_path / "b"), parity_dir="/y") == "/y"   # the same dir: no rewrite
        assert sp.load_plan(tmp_path / "b")["parity_record_dir"] == "/y"
        # cmd_verdict reads the remembered record (and refuses by name when it is not a parity record)
        seen = {}
        def fake_load(path, main_plan):
            seen["path"] = str(path)
            raise ValueError("not a parity record (test)")
        monkeypatch.setattr(sp, "load_parity_record", fake_load)
        import argparse
        with pytest.raises(SystemExit, match="not a parity record"):
            sp.cmd_verdict(argparse.Namespace(model_key=MODEL, out=tmp_path, probe_job_id="1", verdict_uri=None,
                                              geometry_dir=None, probe_cluster=None, parity_record=None))
        assert seen["path"] == "/some/parity"


class TestReopenNeedsTheReceiptForAnExtension:
    def test_a_complete_record_whose_only_work_is_the_extension_refuses_without_a_receipt_and_mutates_nothing(self, tmp_path, monkeypatch):
        """The receipt check used only the OLD outstanding / replan-required ids, so a COMPLETE legacy record
        whose every old result is terminal but whose pair is excluded (a dead S1c) had `work_pre == []`, needed no receipt, and
        `run --resume --replan` would have archived its menu / verdict / overlay and started the S1cc / S1d launches with no
        revoke marker on any cluster copy. Now the extension is computed BEFORE the receipt check (pure, no write) and its ids
        are work: without the front door's receipt the run REFUSES, and nothing on the record moves."""
        plan = _legacy_pair_plan(windows=(4096,), fallbacks=True)
        by = _by_id(plan)
        mf = lambda n: sp.plan_manifest(plan, n)  # noqa: E731
        results = {"mlp_r64_S1_cap128_L4096": _dead(by["mlp_r64_S1_cap128_L4096"], S=2),
                   "mlp_r64_S1a_cap64_L4096": _passed(by["mlp_r64_S1a_cap64_L4096"], mf("mlp_r64_S1a"), S=8),
                   "mlp_r64_S1b_cap64_L4096": _passed(by["mlp_r64_S1b_cap64_L4096"], mf("mlp_r64_S1b"), S=8),
                   "mlp_r64_S1c_cap128_L4096": _dead(by["mlp_r64_S1c_cap128_L4096"], S=2), **_s2_passed(plan)}
        out = tmp_path / "rec"
        sp.write_json_atomic(out / sp.PLAN_FILE, plan)
        for lid, res in results.items():
            sp.write_json_atomic(sp.launch_result_path(out, lid), res)
        monkeypatch.delenv("SISTEER_GEOMETRY_DIR", raising=False)
        v, allowed = sp.write_verdict(plan, results, probe_job_id="1", out=out)
        assert v["status"] == "complete" and allowed["pairs"] == []            # decided: the difference server died
        prog = sp.write_progress(plan, results, out)
        assert prog["complete"] is True and sp.outstanding_launches(plan, results) == [] \
            and sp.replan_required_launches(plan, results) == []
        # the extension IS work: the pair is uncertified, so the current servers would be added
        _np, ext = sp.extend_plan_for_uncertified_pairs(plan, results, Q4)
        assert sorted(ext["added"]) == ["mlp_r64_S1cc_cap64_L4096", "mlp_r64_S1d_cap128_L4096"]
        before = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
        superseded_before = sorted((out / sp.LAUNCHES_DIR / sp.SUPERSEDED_DIR).glob("*")) if (out / sp.LAUNCHES_DIR / sp.SUPERSEDED_DIR).is_dir() else []
        with pytest.raises(SystemExit) as ei:
            sp.main(["run", "--model-key", MODEL, "--out", str(out), "--resume", "--replan",
                     "--plan-args", "--sets mlp --ranks 64 --tp 1"])
        msg = str(ei.value)
        assert "receipt" in msg.lower() and "2 launch(es)" in msg
        after = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
        assert after == before                                                # plan, verdict, menu, checkpoint: byte-identical
        assert (out / sp.ALLOWED_PAIRS_FILE).is_file() and (out / sp.VERDICT_FILE).is_file()
        assert "extensions" not in sp.load_plan(out) and sp.load_plan(out)["n_launches"] == plan["n_launches"]
        superseded_after = sorted((out / sp.LAUNCHES_DIR / sp.SUPERSEDED_DIR).glob("*")) if (out / sp.LAUNCHES_DIR / sp.SUPERSEDED_DIR).is_dir() else []
        assert superseded_after == superseded_before                          # nothing archived
        # the consult may create reopen_receipts/ (it lists it); it must leave no consumed / in-progress / lock receipt behind
        assert not list((out / sp.RECEIPTS_DIR).glob("*.json")) if (out / sp.RECEIPTS_DIR).is_dir() else True


class TestParityStatusCarriesTheRecordsStateOnPlanMismatch:
    def test_a_replanned_complete_parity_record_reads_plan_mismatch_with_its_completeness_and_live_generation(self, tmp_path):
        """The front door's PARITY_STALE remedy (the receipted `reverdict --parity-record`) may go out only over a parity record
        that is COMPLETE and at rest, judged from the status's `parity_record.complete`. A re-planned parity record (a reopen that
        EXTENDED its plan: 25 -> 29 launches — the live one, 2026-10-01) is `plan_mismatch`, and that branch returned BEFORE reading
        the record's checkpoint, so `complete` was None and the remedy was refused with `complete=`. Now the state is read first."""
        import hpo_probe_status as hps
        main = tmp_path / "main"; parity = tmp_path / "parity"
        (parity / "launches").mkdir(parents=True); main.mkdir()
        pplan = {"parity_pass": True, "model_key": "k", "model_id": "org/toy", "revision": REV, "n_launches": 29, "windows": [4096],
                 "synthetic_scale_by_set": {"mlp": 0.05}}
        (parity / hps.PLAN_FILE).write_text(json.dumps(pplan))
        for i in range(3):
            (parity / "launches" / f"l{i}.json").write_text(json.dumps({"launch_id": f"l{i}", "status": "passed"}))
        live = hps.record_generation(parity)
        (parity / hps.PROGRESS_FILE).write_text(json.dumps({"complete": True, "attempt": None, "hold": None, "n_done": 29,
                                                            "result_generation": live["generation"]}))
        (main / hps.PLAN_FILE).write_text(json.dumps({"model_key": "k", "model_id": "org/toy", "revision": REV}))
        stamp = {"dir": str(parity), hps.GENERATION_KEY: "a" * 64, "n_results": 25, "plan_launches": 25, "windows": [4096],
                 "synthetic_scale_by_set": {"mlp": 0.05}}
        info = hps.verify_parity_record(main, {hps.PARITY_RECORD_KEY: stamp}, {hps.PARITY_RECORD_KEY: stamp})
        assert info["parity_check"] == "plan_mismatch" and "29 launches, the stamp says 25" in info["parity_why"]
        rec = info["parity_record"]
        assert rec["complete"] is True and rec["attempt"] is None and rec["hold"] is None
        assert rec["live_generation"] == live["generation"] and rec["live_n_results"] == 3
        # a re-planned record whose driver still holds it says so through the same fields (the remedy must wait)
        (parity / hps.PROGRESS_FILE).write_text(json.dumps({"complete": False, "attempt": {"started_utc": "t", "job_id": "1"}, "hold": None}))
        info2 = hps.verify_parity_record(main, {hps.PARITY_RECORD_KEY: stamp}, {hps.PARITY_RECORD_KEY: stamp})
        assert info2["parity_check"] == "plan_mismatch" and info2["parity_record"]["complete"] is False \
            and info2["parity_record"]["attempt"] == {"started_utc": "t", "job_id": "1"}
        # an unchanged plan keeps the ordinary precedence: in_progress outranks the generation comparison
        pplan["n_launches"] = 25; (parity / hps.PLAN_FILE).write_text(json.dumps(pplan))
        info3 = hps.verify_parity_record(main, {hps.PARITY_RECORD_KEY: stamp}, {hps.PARITY_RECORD_KEY: stamp})
        assert info3["parity_check"] == "in_progress"


# --------------------------------------------------------------------------- the parity record must be AT REST


class TestParityRecordAtRest:
    def _records(self, tmp_path, *, n_launches=29, progress=None, own_verdict=True, own_ref=True):
        import hpo_probe_status as hps
        main = tmp_path / "main"; parity = tmp_path / "parity"
        (parity / "launches").mkdir(parents=True); main.mkdir(exist_ok=True)
        (parity / hps.PLAN_FILE).write_text(json.dumps({"parity_pass": True, "model_key": "k", "model_id": "org/toy", "revision": REV,
                                                       "n_launches": n_launches, "windows": [4096], "synthetic_scale_by_set": {"mlp": 0.05}}))
        for i in range(3):
            (parity / "launches" / f"l{i}.json").write_text(json.dumps({"launch_id": f"l{i}", "status": "passed"}))
        live = hps.record_generation(parity)
        prog = {"complete": True, "attempt": None, "hold": None, "n_done": n_launches, "result_generation": live["generation"]}
        prog.update(progress or {})
        (parity / hps.PROGRESS_FILE).write_text(json.dumps(prog))
        if own_verdict:
            (parity / hps.VERDICT_FILE).write_text(json.dumps({"status": "complete"}))
        if own_ref:
            (parity / hps.REFERENCE_CHECK_FILE).write_text(json.dumps({"status": "done"}))
        (main / hps.PLAN_FILE).write_text(json.dumps({"model_key": "k", "model_id": "org/toy", "revision": REV}))
        stamp = {"dir": str(parity), hps.GENERATION_KEY: "a" * 64, "n_results": 25, "plan_launches": 25, "windows": [4096],
                 "synthetic_scale_by_set": {"mlp": 0.05}}
        return hps, main, parity, stamp, live

    def test_a_replanned_record_is_at_rest_only_with_no_hold_a_current_checkpoint_and_its_own_artefacts(self, tmp_path):
        hps, main, parity, stamp, live = self._records(tmp_path)
        info = hps.verify_parity_record(main, {hps.PARITY_RECORD_KEY: stamp}, {hps.PARITY_RECORD_KEY: stamp})
        rec = info["parity_record"]
        assert info["parity_check"] == "plan_mismatch" and rec["complete"] is True and rec["at_rest"] is True
        assert rec["checkpoint_generation_matches_live"] is True and rec["own_verdict_present"] and rec["own_reference_check_present"]

    def test_plan_mismatch_with_complete_true_but_own_artefacts_absent_is_not_at_rest(self, tmp_path):
        """`complete` is written on the last RESULT before the reference / verdict phases: a re-planned pass read in that window
        has complete=true and NO verdict / reference_check of its own — the remedy must not see it as eligible."""
        hps, main, parity, stamp, live = self._records(tmp_path, own_verdict=False, own_ref=False)
        rec = hps.verify_parity_record(main, {hps.PARITY_RECORD_KEY: stamp}, {hps.PARITY_RECORD_KEY: stamp})["parity_record"]
        assert rec["complete"] is True and rec["at_rest"] is False and rec["own_verdict_present"] is False and rec["own_reference_check_present"] is False
        # a checkpoint whose stamped generation is stale (written before the last result) is not at rest either
        hps2, main2, parity2, stamp2, _ = self._records(tmp_path / "b", progress={"result_generation": "f" * 64})
        rec2 = hps2.verify_parity_record(main2, {hps2.PARITY_RECORD_KEY: stamp2}, {hps2.PARITY_RECORD_KEY: stamp2})["parity_record"]
        assert rec2["complete"] is True and rec2["checkpoint_generation_matches_live"] is False and rec2["at_rest"] is False

    def test_a_held_record_is_not_at_rest(self, tmp_path):
        hps, main, parity, stamp, live = self._records(tmp_path, progress={"complete": False, "attempt": {"started_utc": "t", "job_id": "9"}})
        rec = hps.verify_parity_record(main, {hps.PARITY_RECORD_KEY: stamp}, {hps.PARITY_RECORD_KEY: stamp})["parity_record"]
        assert rec["at_rest"] is False and rec["attempt"] == {"started_utc": "t", "job_id": "9"}
        hps, main, parity, stamp, live = self._records(tmp_path / "h", progress={"hold": "reference_only_in_progress"})
        rec = hps.verify_parity_record(main, {hps.PARITY_RECORD_KEY: stamp}, {hps.PARITY_RECORD_KEY: stamp})["parity_record"]
        assert rec["at_rest"] is False and rec["hold"] == "reference_only_in_progress"

    def test_the_front_doors_parity_stale_remedy_tests_at_rest_never_complete_alone(self):
        """The remedy's guard, executed: the PARITY_STALE branch's one guard line reads `at_rest` off the status's parity_record and
        refuses otherwise — run with json_field stubbed over a status object of each shape."""
        import subprocess
        src = (REPO / "reproduce" / "03_hpo.sh").read_text(encoding="utf-8")
        guards = [ln for ln in src.splitlines() if "--reopen on PARITY_STALE: the parity record" in ln and "repro_fatal" in ln]
        assert len(guards) == 1 and 'json_field "$prec" at_rest' in guards[0] and 'json_field "$prec" complete)" = true' not in guards[0]
        line = guards[0].strip()
        def run(prec: dict) -> int:
            script = ("json_field(){ python3 -c 'import json,sys; v=json.loads(sys.argv[1]).get(sys.argv[2]); "
                      "print(\"true\" if v is True else \"false\" if v is False else (\"\" if v is None else v))' \"$1\" \"$2\"; }\n"
                      "repro_fatal(){ echo \"REFUSED: $*\" >&2; exit 2; }\n"
                      f"prec='{json.dumps(prec)}'; pdir=/p\n" + line + "\necho PROCEED")
            return subprocess.run(["bash", "-c", script], capture_output=True, text=True).returncode
        assert run({"complete": True, "at_rest": True}) == 0
        assert run({"complete": True, "at_rest": False}) == 2            # complete alone never admits the remedy
        assert run({"complete": True}) == 2


# --------------------------------------------------------------------------- the attempt head: single-shot phases never rewrite it


class TestAttemptHeadSurvivesSingleShotPhases:
    def test_the_carrier_preserves_the_standing_counter_in_reverdict_and_reference_and_stamps_the_phase(self, tmp_path):
        """write_result, EXECUTED out of the carrier with its inputs stubbed: a reverdict / reference phase keeps the standing head's
        attempt=N/M (job 818826 wrote 1/1 over 805161's 3/5); the full phase writes its own; every head names its phase."""
        import re
        import subprocess
        src = (REPO / "scripts" / "sbatch" / "serving_probe.sbatch").read_text(encoding="utf-8")
        m = re.search(r"^write_result\(\)\{\n(.*?)^\}\n", src, re.S | re.M)
        assert m, "write_result() not found"
        fn = m.group(0)
        calls = [0]
        def run(phase, attempt, maxa, prior):
            calls[0] += 1
            figs = tmp_path / f"figs-{calls[0]}-{phase}"; out = tmp_path / f"out-{calls[0]}-{phase}"
            figs.mkdir(); out.mkdir()
            if prior:
                (figs / "RESULT").write_text(prior + "\n")
            (out / "RESULT").write_text("state=VERDICT pairs_allowed=9\n")
            script = (f"PHASE={phase}; ATTEMPT={attempt}; MAX_ATTEMPTS={maxa}; SECONDS=5; MODEL_KEY=k; MODE=tp1; SLURM_JOB_ID=1; SUCCESSOR=; "
                      f"FIGS_DIR='{figs}'; OUT='{out}'\n" + fn + "\nwrite_result DONE 0\nhead -1 \"$FIGS_DIR/RESULT\"")
            r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
            return r.stdout.strip()
        prior = "state=DONE exit=0 attempt=3/5 elapsed=8976 host=h model=k mode=tp1 job=805161"
        rv = run("reverdict", 1, 1, prior)
        assert "attempt=3/5" in rv and "phase=reverdict" in rv and "job=1" in rv
        rf = run("reference", 1, 1, prior)
        assert "attempt=3/5" in rf and "phase=reference" in rf
        full = run("full", 4, 5, prior)
        assert "attempt=4/5" in full and "phase=full" in full
        assert "attempt=1/2" in run("reverdict", 1, 2, None)                 # no standing head: its own, honestly stamped

    def test_the_resume_derivation_ignores_a_single_shot_head(self):
        import hpo_probe_resume as hpr
        assert hpr.parse_legacy_head("state=DONE exit=0 attempt=3/5 elapsed=1 host=h model=k mode=tp1 job=805161") == {
            "attempt": 3, "max_attempts": 5, "job_id": "805161", "state": "DONE"}
        assert hpr.parse_legacy_head("state=DONE exit=0 attempt=1/1 phase=reverdict elapsed=1 host=h model=k mode=tp1 job=818826") is None
        assert hpr.parse_legacy_head("state=DONE exit=0 attempt=1/1 phase=reference elapsed=1 job=1") is None
        assert hpr.parse_legacy_head("state=DONE exit=0 attempt=4/5 phase=full elapsed=1 job=1")["attempt"] == 4


class TestServedClassOnALegacyRecord:
    def test_a_legacy_four_slot_launch_already_dominates_the_pairs_union_so_the_extension_adds_none(self):
        """The probe owner's ruling of 2026-10-11, read against a record written BEFORE the split: the legacy four-slot ``S1`` at
        cap 128 (4 slots) IS the union class of (64, mlp) — cells + zero at cap 2r — so its planned launch dominates the served
        class at every window it was planned at and the extension's third reason adds nothing; where the legacy server DIED at a
        window the extension adds the current servers for the uncertified role, never a second copy of the shape that died."""
        plan = _legacy_pair_plan(windows=(4096, 49152), fallbacks=False)
        by = _by_id(plan)
        s1 = sp.plan_manifest(plan, "mlp_r64_S1")
        assert sp._union_class(sm.build_union_manifest(Q4, set_id="mlp", rank=64)) == (128, 4) == (s1.max_lora_rank, s1.max_loras)
        results = {"mlp_r64_S1_cap128_L4096": _passed(by["mlp_r64_S1_cap128_L4096"], s1, S=8),
                   "mlp_r64_S1_cap128_L49152": _passed(by["mlp_r64_S1_cap128_L49152"], s1, S=4), **_s2_passed(plan, S=64)}
        union = sm.build_union_manifest(Q4, set_id="mlp", rank=64)
        for L in (4096, 49152):
            # the legacy S1 covers the class (same target class, same cap and slots); the shared o_proj S2 never counts
            assert sp._served_class_dominated(plan, plan["pairs"][0], union, L, {n: sp.plan_manifest(plan, n) for n in plan["manifests"]}) \
                == [f"mlp_r64_S1_cap128_L{L}"]
        # the legacy S1 passed at both windows: the third reason adds nothing (the servers' own starvation reason may — S 8 at
        # 4,096 is below the current servers' admission there — but never a union, and never under its name)
        new_plan, ext = sp.extend_plan_for_uncertified_pairs(plan, results, Q4)
        assert sp.REASON_SERVED_CLASS not in ext["why"].values() and "mlp_r64_single" not in new_plan["manifests"]
        assert not any("_single_" in lid for lid in ext["added"]) and set(ext["manifests"]) <= {"mlp_r64_S1cc", "mlp_r64_S1d"}
        again, ext2 = sp.extend_plan_for_uncertified_pairs(new_plan, results, Q4)
        assert ext2["added"] == [] and again["n_launches"] == new_plan["n_launches"]
        # the legacy server DIED at 49,152: the current servers come in for the uncertified roles there, no union (its planned
        # twin, the dead S1, is the evidence that the shape cannot come up at that window)
        dead = dict(results)
        dead["mlp_r64_S1_cap128_L49152"] = _dead(by["mlp_r64_S1_cap128_L49152"], S=4)
        new_plan, ext = sp.extend_plan_for_uncertified_pairs(plan, dead, Q4)
        assert {"mlp_r64_S1cc_cap64_L49152", "mlp_r64_S1d_cap128_L49152"} <= set(ext["added"])
        assert ext["why"]["mlp_r64_S1cc_cap64_L49152"] == "role_uncertified" == ext["why"]["mlp_r64_S1d_cap128_L49152"]
        assert sp.REASON_SERVED_CLASS not in ext["why"].values() and "mlp_r64_single" not in new_plan["manifests"]
