"""Serving-probe smoke (`tests/test_serving_probe_smoke.py`): the
manifest builder, the expected-bytes law, the packed-stride ceiling, the
plan's one-launch-per-(manifest, cap, topology, window) rule, the adapter-dir
builder under a monkeypatched exporter, the verdict writer's `allowed_pairs.json`
with the documented exclusion reasons, and the `--dry-run` CLI — all on CPU,
no vllm, no peft, on a 2-block dense toy dims object, a 2-MoE-block toy (the routed-experts
serving OPTION — none of the paper models routes experts) and the worked example's
Qwen3.5-4B dims.

The buffer-bytes law and the executing stride are HARD gates and a missing
required measurement fails; a time-budget stop is INCOMPLETE (exit 4),
checkpoints the launch list, `run --resume` continues it and `verdict` refuses
to certify while a decision-relevant launch is outstanding — no
`allowed_pairs.json` over unrun pairs; the HF/PEFT reference prompt is the
pinned thinking-off render the native server also applies; the planned S at
4,096 is pinned by value, not by itself; the sbatch carrier honours
SISTEER_SHARED_ROOT / SISTEER_HF_HOME with no one-cluster fallback and
requeues itself on the incomplete exit.

The 2026-10-01 topology ruling, which these pins follow: every (set, rank)
pair is served by ``S1cc`` (control r + constitution r + zero [+ the four rank-1
carriers for ``wo``] at cap ``rank_cap(r)``) and ``S1d`` (difference 2r + zero at cap
``rank_cap(2r)``), plus the shared ``S2`` for every set but ``wo``; on a routed-experts
model the expert buffers are priced at each server's ``moe_lora_rank``; the verdict
certifies the control and difference ROLES window by window. No four-slot ``S1``
and no S1a/S1b/S1c VRAM fallback is planned any more (reading OLD records that
carry them is tests/test_serving_probe_legacy_records.py's job; the worker patch
/ composer / manifest pins of the MoE option are tests/test_moe_lora_buffer_rank.py's).

The served launch class (the probe owner's ruling of 2026-10-11, ``TestServedClassUnion``):
beside each pair's servers the plan carries the pair's UNION manifest — the ``single``
shape, cells + zero at cap >= 2r [+ the carriers for ``wo``]: the class a steered
standup SERVES — as one natural launch at EVERY window; the fold writes its engine
class so ``geometry.least_admitted_by_window`` sizes the served class at every window;
the roles stay certified by S1cc / S1d; the reopen's extension adds the union to an
existing record where no planned launch dominates it, and ``run --resume --replan
--dry-run`` previews exactly that."""
from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from sisteer.capeval.serving import standup_manifest as sm
from sisteer.extraction.lora_modules import QWEN35_4B_DIMS as Q4
from sisteer.extraction.lora_modules import ArchDims

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
sys.path.insert(0, str(REPO / "tests"))
import serving_probe as sp  # noqa: E402
from front_door_testlib import MATCHERS, SCAN, assert_no_tokens, token_hits  # noqa: E402  — the ONE token table the publication scan also reads

#: one tier of the publication scan's token table, matched as the scan matches it (the tokens are never spelled here)
STORAGE_TOKENS = SCAN.Matcher([r for r in MATCHERS["rows"] if r.tier == "storage"])

#: a 2-block dense toy: hidden 64, one attention block (q_out = 2 heads x 32 x 2
#: gate = 128, kv_out 16, o_in 64), two dense MLP blocks of intermediate 32
TOY = ArchDims(hidden=64, attention_layers=1, q_out=128, kv_out=16, o_in=64, dense_mlp_layers=2, dense_intermediate=32)
#: a 2-MoE-block toy with the same attention block: 4 routed experts (2 active) of I = 32 plus a shared expert of 32
MOE_TOY = ArchDims(hidden=64, attention_layers=1, q_out=128, kv_out=16, o_in=64, moe_layers=2,
                   num_experts=4, active_experts=2, moe_intermediate=32, shared_intermediate=32)
REV = "0123456789abcdef0123456789abcdef01234567"
#: the probe's subject: the worked example's served key, the one onboarded model of the dims table
#: (lora_modules.ARCH_DIMS_BY_MODEL_KEY: hidden 2560, 8 full-attention blocks, 32 dense MLP blocks of 9,216)
MODEL = "qwen35_4b_demo"
MODEL_HF_ID = "Qwen/Qwen3.5-4B"
#: a PASSING HF+PEFT reference record (the certifying evidence a pair needs)
REF_OK = {"status": "pass", "basis": sp.BASIS_HF_PEFT, "per_adapter": {"control__lora": {"status": "pass"}}}


@pytest.fixture(autouse=True)
def _no_site_overlay(monkeypatch):
    """No site geometry overlay reaches these tests from the shell (a stale overlay would flip the resolvers'
    answers); the fold tests set the directory explicitly."""
    from sisteer.capeval.serving import geometry as g
    for v in (g.SITE_GEOMETRY_DIR_ENV, g.SITE_DIR_ENV, g.SHARED_ROOT_ENV, "SISTEER_MODELS_TABLE"):
        monkeypatch.delenv(v, raising=False)


def _plan(dims=TOY, sets=("wo", "mlp"), ranks=(4, 16), windows=(4096, 49152), **kw):
    return sp.build_plan(dims, model_key=MODEL, model_id="org/toy", revision=REV, sets=sets, ranks=ranks,
                         windows=windows, **kw)


# --------------------------------------------------------------------------- manifests


class TestManifests:
    def test_wo_is_two_standups_a_seven_slot_s1cc_and_a_two_slot_s1d_on_the_o_proj_class(self):
        """The ruling: the single-rank adapters never share a cap with the 2r difference. ``wo`` is S1cc (control r +
        constitution r + zero + the four rank-1 carriers = 7 slots, cap rank_cap(r)) and S1d (difference 2r + zero =
        2 slots, cap rank_cap(2r)) — no ``single`` server, nothing but o_proj is wrapped (no expert on any dims), nothing is
        a fallback."""
        for dims in (TOY, MOE_TOY, Q4):
            cc, d = sm.build_manifests(dims, set_id="wo", rank=4)
            assert (cc.topology, d.topology) == ("S1cc", "S1d") and (cc.name, d.name) == ("wo_r4_S1cc", "wo_r4_S1d")
            assert cc.max_loras == 7 == len(cc.adapters) and d.max_loras == 2 == len(d.adapters)
            assert cc.target_class == d.target_class == ("o_proj",)
            assert sorted(a.kind for a in cc.adapters) == ["carrier"] * 4 + ["genuine"] * 2 + ["zero"]
            assert {a.role for a in cc.adapters if a.kind == "carrier"} == set(sm.CARRIER_ROLES)
            assert cc.genuine_roles == ("control", "constitution") and d.genuine_roles == ("difference",)
            assert cc.carries_role("control") and not cc.carries_role("difference") and d.carries_role("difference")
            assert cc.max_lora_rank == 8 == sm.rank_cap(4) and d.max_lora_rank == 8 == sm.rank_cap(8)
            assert cc.moe_lora_rank is None and d.moe_lora_rank is None and not cc.wraps_experts
            assert cc.fallback_of is None and d.fallback_of is None
        cc16, d16 = sm.build_manifests(Q4, set_id="wo", rank=16)
        assert (cc16.max_lora_rank, d16.max_lora_rank) == (16, 32)       # cap r on S1cc, cap 2r on S1d

    @pytest.mark.parametrize("set_id", ["qkvo", "attn_mlp", "mlp", "down"])
    def test_mlp_bearing_and_qkvo_sets_produce_s1cc_s1d_and_s2(self, set_id):
        ms = sm.build_manifests(TOY, set_id=set_id, rank=4)
        assert [m.topology for m in ms] == ["S1cc", "S1d", "S2"], [m.name for m in ms]
        s1cc, s1d, s2 = ms
        assert s1cc.max_loras == 3 and [a.role for a in s1cc.adapters] == ["control", "constitution", "none"]
        assert s1d.max_loras == 2 and [a.role for a in s1d.adapters] == ["difference", "none"]
        expected_class = tuple(sorted(sm.MODULE_SETS[set_id]))
        assert s1cc.target_class == s1d.target_class == expected_class
        assert s1cc.max_lora_rank == 8 == sm.rank_cap(4) and s1d.max_lora_rank == 8 == sm.rank_cap(8)
        assert all(m.fallback_of is None for m in ms)
        assert s2.target_class == ("o_proj",) and s2.max_lora_rank == 8 and s2.max_loras == 5
        assert s2.passthrough_cells == ("si_safety", "si_constitution")
        assert all(a.rank == 1 and a.lora_modules == ("o_proj",) for a in s2.adapters if a.kind == "carrier")
        assert not any(a.kind == "carrier" for m in (s1cc, s1d) for a in m.adapters)    # the carriers ride S2 here
        assert {a.rank for a in s1cc.genuine} == {4} and {a.rank for a in s1d.genuine} == {8}
        # a dense model states no expert rank anywhere
        assert all(a.per_expert_rank is None for m in ms for a in m.adapters) and all(m.moe_lora_rank is None for m in ms)

    @pytest.mark.parametrize("set_id", ["qkvo", "attn_mlp", "mlp", "down"])
    def test_on_a_routed_experts_model_the_mlp_bearing_classes_wrap_the_experts_and_price_their_rank(self, set_id):
        """The MoE serving option beside the dense shape above: on the MoE toy every MLP-bearing set's class gains ``experts``
        (qkvo's does not), FD 6's per-expert rank rides every genuine slot (r_e = max(1, r // active) on the single-rank
        adapters, 2 x r_e on the difference — its two parents stacked along the per-expert axis, never r_e(2r)), and each
        server's ``moe_lora_rank`` is its adapters' largest routed rank (the zero adapter's cap // active binds on
        both toy servers at r = 4)."""
        s1cc, s1d, s2 = sm.build_manifests(MOE_TOY, set_id=set_id, rank=4)
        expected_class = tuple(sorted(sm.MODULE_SETS[set_id] + (("experts",) if set_id != "qkvo" else ())))
        assert s1cc.target_class == s1d.target_class == expected_class and s2.target_class == ("o_proj",)
        if set_id != "qkvo":
            assert s1cc.wraps_experts and s1d.wraps_experts
            assert {a.per_expert_rank for a in s1cc.genuine} == {2} and {a.per_expert_rank for a in s1d.genuine} == {4}
            assert s1d.genuine[0].per_expert_rank == sm.difference_per_expert_rank(MOE_TOY, sm.MODULE_SETS[set_id], 4) == 4
            assert s1cc.moe_lora_rank == 4 == s1cc.zero.per_expert_rank and s1d.moe_lora_rank == 4
            assert "|moe4|" in s1cc.launch_key and s1cc.as_dict()["moe_lora_rank"] == 4
        else:
            assert s1cc.moe_lora_rank is None and s1d.moe_lora_rank is None and not s1cc.wraps_experts
        assert s2.moe_lora_rank is None

    @pytest.mark.parametrize("rank,cap_cc,cap_d", [(4, 8, 8), (16, 16, 32), (64, 64, 128)])
    def test_the_difference_2r_drives_the_s1d_cap_r_the_s1cc_cap_and_each_zero_pins_its_server(self, rank, cap_cc, cap_d):
        for set_id in ("wo", "mlp"):
            cc, d = sm.build_manifests(TOY, set_id=set_id, rank=rank)[:2]
            diff = next(a for a in d.adapters if a.role == "difference")
            assert diff.rank == 2 * rank
            assert d.max_lora_rank == cap_d == sm.rank_cap(2 * rank)
            assert d.zero.rank == cap_d                 # fitted --max-lora-rank follows the largest declared r
            assert not any(a.role == "difference" for a in cc.adapters) and {a.rank for a in cc.genuine} == {rank}
            assert cc.max_lora_rank == cap_cc == sm.rank_cap(rank) and cc.zero.rank == cap_cc

    def test_real_dims_rank64_mlp_sets_split_into_s1cc_s1d_s2_with_no_vram_fallback(self):
        """Before the ruling the four-slot S1 at r = 64 padded every buffer to the difference's cap 128 and the builder
        appended the three-server S1a/S1b/S1c fallback on the real dims; now the split IS the smaller server: the
        MLP-bearing sets at r = 64 are S1cc (cap 64) + S1d (cap 128) + S2, and no manifest is ever a fallback of
        another (the fallback constants remain only to read old records). On the MoE toy the same split carries the
        expert buffers at R = 32 (S1cc: 64 // 2) and R = 64 (S1d: 2 x 32 = 128 // 2)."""
        for set_id in ("attn_mlp", "mlp", "down"):
            ms = sm.build_manifests(Q4, set_id=set_id, rank=64)
            assert [m.topology for m in ms] == ["S1cc", "S1d", "S2"], [m.name for m in ms]
            assert all(m.fallback_of is None and m.topology not in sm.S1_FALLBACKS for m in ms)
            cc, d = ms[0], ms[1]
            assert (cc.max_lora_rank, cc.moe_lora_rank, cc.max_loras) == (64, None, 3)
            assert (d.max_lora_rank, d.moe_lora_rank, d.max_loras) == (128, None, 2)
            assert [a.role for a in d.adapters] == ["difference", "none"]
            assert d.genuine[0].rank == 128 == d.zero.rank and cc.genuine[0].rank == 64 == cc.zero.rank
            mcc, md, _ = sm.build_manifests(MOE_TOY, set_id=set_id, rank=64)
            assert (mcc.max_lora_rank, mcc.moe_lora_rank, mcc.max_loras) == (64, 32, 3)
            assert (md.max_lora_rank, md.moe_lora_rank, md.max_loras) == (128, 64, 2)
            assert md.genuine[0].per_expert_rank == 64 == md.zero.per_expert_rank      # 2 x 32, and 128 // 2
            assert mcc.genuine[0].per_expert_rank == 32 == mcc.zero.per_expert_rank    # 64 // 2 on both
        # every rank, set and dims takes the same shape; the VRAM budget arguments change nothing (no fit question remains)
        assert [m.topology for m in sm.build_manifests(Q4, set_id="mlp", rank=16)] == ["S1cc", "S1d", "S2"]
        assert [m.topology for m in sm.build_manifests(Q4, set_id="qkvo", rank=64)] == ["S1cc", "S1d", "S2"]
        assert [m.topology for m in sm.build_manifests(TOY, set_id="mlp", rank=64)] == ["S1cc", "S1d", "S2"]
        assert [m.topology for m in sm.build_manifests(MOE_TOY, set_id="mlp", rank=64)] == ["S1cc", "S1d", "S2"]
        assert sm.build_manifests(Q4, set_id="mlp", rank=64, kv_budget_bytes=1.0) == sm.build_manifests(Q4, set_id="mlp", rank=64)

    def test_moe_lora_rank_is_none_off_the_experts_and_the_adapters_largest_routed_rank_on_them(self):
        """``Manifest.moe_lora_rank`` is the R the server's fused-MoE LoRA buffers are allocated at — None when the
        class wraps no expert (``wo``, ``qkvo``, every S2, every dense dims), else the largest per-expert rank over its
        adapters: on the 2-active toy r = 4 -> S1cc 4 / S1d 4, r = 16 -> 8 / 16, r = 64 -> 32 / 64."""
        for set_id in ("wo", "qkvo"):
            for rank in (4, 16, 64):
                for dims in (TOY, MOE_TOY, Q4):
                    assert all(m.moe_lora_rank is None for m in sm.build_manifests(dims, set_id=set_id, rank=rank)), (set_id, rank)
        for rank, (cc_R, d_R) in {4: (4, 4), 16: (8, 16), 64: (32, 64)}.items():
            for set_id in ("mlp", "attn_mlp", "down"):
                cc, d, s2 = sm.build_manifests(MOE_TOY, set_id=set_id, rank=rank)
                assert (cc.moe_lora_rank, d.moe_lora_rank, s2.moe_lora_rank) == (cc_R, d_R, None), (set_id, rank)
                assert cc.moe_lora_rank == max(a.per_expert_rank for a in cc.adapters)
                assert d.moe_lora_rank == max(a.per_expert_rank for a in d.adapters)
                assert cc.moe_lora_rank <= cc.max_lora_rank and d.moe_lora_rank <= d.max_lora_rank
                assert all(m.moe_lora_rank is None for m in sm.build_manifests(TOY, set_id=set_id, rank=rank))
        # the rank rides the engine key (servers that differ only in their expert rank are different engines) and the record
        cc, d, _ = sm.build_manifests(MOE_TOY, set_id="mlp", rank=16)
        assert "|moe8|" in cc.launch_key and "|moe16|" in d.launch_key and d.as_dict()["moe_lora_rank"] == 16

    def test_every_s2_shares_one_engine_key_and_round_trips(self):
        keys = {sm.build_manifests(dims, set_id=s, rank=r)[2].launch_key for dims in (TOY, MOE_TOY) for s in ("qkvo", "mlp", "down") for r in (4, 16)}
        assert len(keys) == 1
        for dims in (Q4, MOE_TOY):
            for m in sm.build_manifests(dims, set_id="attn_mlp", rank=64):
                back = sm.Manifest.from_dict(json.loads(json.dumps(m.as_dict())))
                assert back == m and back.moe_lora_rank == m.moe_lora_rank

    def test_refusals(self):
        with pytest.raises(sm.ManifestError):
            sm.build_manifests(TOY, set_id="nope", rank=4)
        with pytest.raises(sm.ManifestError):
            sm.rank_cap(513)
        cc = sm.build_manifests(TOY, set_id="wo", rank=4)[0]             # 7 slots
        with pytest.raises(sm.ManifestError):
            dataclasses.replace(cc, max_loras=6)
        with pytest.raises(sm.ManifestError):
            dataclasses.replace(cc, adapters=tuple(a for a in cc.adapters if a.kind != "zero"), max_loras=6)
        # an adapter above the cap is refused by name (cap 1 is legal and below the rank-4 adapters)
        with pytest.raises(sm.ManifestError, match="> cap"):
            dataclasses.replace(cc, max_lora_rank=1)
        # the expert-buffer rank is refused below an adapter's routed rank, above the cap, and on a class without experts
        mcc = sm.build_manifests(MOE_TOY, set_id="mlp", rank=4)[0]            # moe_lora_rank 4 at cap 8
        with pytest.raises(sm.ManifestError, match="never truncated"):
            dataclasses.replace(mcc, moe_lora_rank=2)
        with pytest.raises(sm.ManifestError, match="> cap"):
            dataclasses.replace(mcc, moe_lora_rank=16)
        with pytest.raises(sm.ManifestError, match="wraps no expert"):
            dataclasses.replace(cc, moe_lora_rank=4)


# --------------------------------------------------------------------------- the bytes law


class TestExpectedBytes:
    """Hand counts on the worked example's Qwen3.5-4B dims at each manifest's natural cap, bf16:
    o_proj per full-attention block ``R x 4096 + 2560 x R``; packed qkv ``3 x R x 2560``
    (one A per slice) + ``(8192 + 2 x 1024) x R``; the dense MLP's packed gate_up
    ``2 x R x 2560 + 2 x 9216 x R`` + down ``R x 9216 + 2560 x R`` per block; 8 attention
    blocks, 32 MLP blocks; every family at the cap R; everything x max_loras. The MoE toy's
    hand counts (the shared expert at the cap, the routed experts at the server's
    ``moe_lora_rank``) sit beside the dense ones."""

    def test_wo_r4_cap8(self):
        cc, d = sm.build_manifests(Q4, set_id="wo", rank=4)
        e = sm.expected_lora_bytes(cc, 8, Q4)
        assert e["per_slot"] == {"attention_o_proj": 8 * 8 * (4096 + 2560) * 2}
        assert e["per_slot_total"] == 851_968 and e["total"] == 851_968 * 7 == 5_963_776    # 7 slots: cells + zero + 4 carriers
        assert abs(e["per_slot_mib_per_R"]["attention_o_proj"] - 0.1015625) < 1e-9
        assert e["R"] == 8 and e["max_loras"] == 7 and e["target_class"] == ["o_proj"]
        assert e["moe_lora_rank"] is None and e["rank_by_family"] == {"attention_o_proj": 8}
        ed = sm.expected_lora_bytes(d, 8, Q4)
        assert ed["per_slot"] == e["per_slot"] and ed["total"] == 851_968 * 2 == 1_703_936   # the difference server: 2 slots

    def test_qkvo_r4_cap8(self):
        cc, d = sm.build_manifests(Q4, set_id="qkvo", rank=4)[:2]
        e = sm.expected_lora_bytes(cc, 8, Q4)
        assert e["per_slot"]["attention_qkv_packed"] == 8 * (3 * 8 * 2560 + 8 * (8192 + 2 * 1024)) * 2 == 2_293_760
        assert e["per_slot"]["attention_o_proj"] == 851_968
        assert e["per_slot_total"] == 3_145_728 and e["total"] == 3_145_728 * 3 == 9_437_184
        assert abs(e["per_slot_total"] / 8 / 2**20 - 0.375) < 1e-9   # 0.375 MiB x R
        assert sm.expected_lora_bytes(d, 8, Q4)["total"] == 3_145_728 * 2 == 6_291_456

    def test_mlp_r4_cap8_prices_the_dense_mlp_family_at_the_cap(self):
        """The MLP sets wrap ONE family on a dense model: the packed gate_up (two slices, one A each) and down over the 32
        blocks, at the cap R = 8 on both servers."""
        cc, d, _s2 = sm.build_manifests(Q4, set_id="mlp", rank=4)
        e = sm.expected_lora_bytes(cc, 8, Q4)
        assert e["per_slot"] == {"dense_mlp": 32 * ((2 * 8 * 2560 + 2 * 9216 * 8) + (8 * 9216 + 2560 * 8)) * 2}
        assert e["per_slot_total"] == 18_087_936 and e["total"] == 18_087_936 * 3 == 54_263_808
        assert abs(e["per_slot_mib_per_R"]["dense_mlp"] - 2.15625) < 1e-9
        # the S1d manifest: the same cap at r = 4 (rank_cap(8) = 8), two slots
        ed = sm.expected_lora_bytes(d, 8, Q4)
        assert ed["per_slot"] == e["per_slot"] and ed["total"] == 18_087_936 * 2 == 36_175_872

    def test_mlp_r4_on_the_moe_toy_prices_the_experts_at_the_servers_routed_rank(self):
        """On the MoE toy: the shared expert at the cap R = 8, the routed experts at the server's ``moe_lora_rank`` (4 on
        both toy servers at r = 4: the zero adapter's 8 // 2) — ``E x R_e x (d + 2I + I + d) x 2 B`` per MoE block; the
        explicit override prices an older record, whose experts were padded to the dense cap."""
        cc, d, _s2 = sm.build_manifests(MOE_TOY, set_id="mlp", rank=4)
        e = sm.expected_lora_bytes(cc, 8, MOE_TOY)
        assert e["moe_lora_rank"] == 4 == cc.moe_lora_rank and e["rank_by_family"] == {"shared_expert": 8, "routed_experts": 4}
        assert e["per_slot"]["routed_experts"] == 2 * 4 * 4 * (64 + 64 + 32 + 64) * 2 == 14_336
        assert e["per_slot"]["shared_expert"] == 2 * ((2 * 8 * 64 + 2 * 32 * 8) + (8 * 32 + 64 * 8)) * 2 == 9_216
        assert e["per_slot_total"] == 23_552 and e["total"] == 23_552 * 3 == 70_656
        assert abs(e["per_slot_mib_per_R"]["routed_experts"] - 14_336 / 4 / 2**20) < 1e-12    # per unit of R_e, not of R
        assert sm.expected_lora_bytes(d, 8, MOE_TOY)["total"] == 23_552 * 2 == 47_104
        assert sm.expected_lora_bytes(cc, 8, MOE_TOY, moe_lora_rank=8)["per_slot"]["routed_experts"] == 2 * 14_336
        # the down set wraps the FULL expert buffers (both w13 and w2) and only the shared down projection
        m = sm.build_manifests(MOE_TOY, set_id="down", rank=4)[0]
        ed = sm.expected_lora_bytes(m, 8, MOE_TOY)
        assert ed["per_slot"]["routed_experts"] == 14_336 and ed["per_slot"]["shared_expert"] == 2 * (8 * 32 + 64 * 8) * 2

    def test_attn_mlp_r64_s1cc_cap64_and_s1d_cap128(self):
        cc, d, _s2 = sm.build_manifests(Q4, set_id="attn_mlp", rank=64)
        e = sm.expected_lora_bytes(cc, 64, Q4)
        assert e["per_slot"] == {"attention_qkv_packed": 18_350_080, "attention_o_proj": 6_815_744, "dense_mlp": 144_703_488}
        assert e["per_slot_total"] == 169_869_312 and e["total"] == 169_869_312 * 3 == 509_607_936
        ed = sm.expected_lora_bytes(d, 128, Q4)
        assert ed["per_slot"] == {"attention_qkv_packed": 36_700_160, "attention_o_proj": 13_631_488, "dense_mlp": 289_406_976}
        assert ed["per_slot_total"] == 339_738_624 and ed["total"] == 339_738_624 * 2 == 679_477_248
        assert ed["per_slot"]["dense_mlp"] == 2 * e["per_slot"]["dense_mlp"]          # linear in the cap: 128 vs 64
        # the MoE toy at r = 64: experts at R_e 32 on S1cc and 64 on S1d — a quarter / a half of the padded cap-128 pricing
        mcc, md, _ = sm.build_manifests(MOE_TOY, set_id="attn_mlp", rank=64)
        em, emd = sm.expected_lora_bytes(mcc, 64, MOE_TOY), sm.expected_lora_bytes(md, 128, MOE_TOY)
        assert (em["moe_lora_rank"], emd["moe_lora_rank"]) == (32, 64)
        assert em["per_slot"]["routed_experts"] == 2 * 4 * 32 * 224 * 2 and emd["per_slot"]["routed_experts"] == 2 * 4 * 64 * 224 * 2
        assert sm.expected_lora_bytes(md, 128, MOE_TOY, moe_lora_rank=128)["per_slot"]["routed_experts"] == 2 * emd["per_slot"]["routed_experts"]

    def test_down_wraps_only_the_dense_down_projection(self):
        m = sm.build_manifests(Q4, set_id="down", rank=4)[0]
        e = sm.expected_lora_bytes(m, 8, Q4)
        assert e["per_slot"] == {"dense_mlp": 32 * (8 * 9216 + 2560 * 8) * 2} and e["per_slot_total"] == 6_029_312

    def test_toy_hand_count(self):
        cc, d, _s2 = sm.build_manifests(TOY, set_id="mlp", rank=4)
        e = sm.expected_lora_bytes(cc, 8, TOY)
        assert e["per_slot"] == {"dense_mlp": 2 * ((2 * 8 * 64 + 2 * 32 * 8) + (8 * 32 + 64 * 8)) * 2} and e["per_slot_total"] == 9_216
        assert e["total"] == 9_216 * 3 == 27_648
        assert sm.expected_lora_bytes(d, 8, TOY)["total"] == 9_216 * 2 == 18_432

    def test_buffer_bytes_within_five_percent(self):
        assert sm.buffer_bytes_within(1000, 1040)["pass"] and not sm.buffer_bytes_within(1000, 1060)["pass"]
        assert sm.buffer_bytes_within(0, 0)["pass"]


# --------------------------------------------------------------------------- strides / admission


class TestStrideCeiling:
    def test_the_qkv_adapting_set_executes_at_the_packed_qkv_stride(self):
        """On the Qwen3.5-4B dims the packed qkv wrapper's ``sum(output_slices)`` is 8192 + 2 x 1024 (the gated query's
        doubled width plus two 4-head KV slices): 209,715 tokens under int32, S <= 25 at 8,192 and S <= 4 at 49,152."""
        c = sm.packed_stride_ceiling(Q4, sm.target_class_for(Q4, "qkvo"))
        assert c["D"] == 8192 + 2 * 1024 and c["widest"] == "self_attn.qkv_proj" and c["ceiling"] == 209_715
        assert c["ceiling"] // 8192 == 25 and c["ceiling"] // 49152 == 4
        assert sm.admitted_max_num_seqs(Q4, sm.target_class_for(Q4, "qkvo"), 8192)["admitted_S"] == 25
        assert sm.admitted_max_num_seqs(Q4, sm.target_class_for(Q4, "qkvo"), 49152)["admitted_S"] == 4

    def test_wo_keeps_the_o_proj_4096_stride(self):
        c = sm.packed_stride_ceiling(Q4, ("o_proj",))
        assert c["D"] == 4096 and c["ceiling"] == 524_287
        assert sm.admitted_max_num_seqs(Q4, ("o_proj",), 8192)["admitted_S"] == 63

    def test_mlp_bearing_sets_execute_at_the_packed_gate_up_width(self):
        """On a dense model the MLP sets' widest executing wrapper is the packed gate_up (2 x 9,216 = 18,432 on the Qwen3.5-4B
        dims), wider than the packed qkv; ``down`` alone executes at the intermediate width."""
        for s in ("mlp", "attn_mlp"):
            c = sm.packed_stride_ceiling(Q4, sm.target_class_for(Q4, s))
            assert c["D"] == 18_432 and c["widest"] == "mlp.gate_up_proj" and c["ceiling"] == 116_508
            assert set(c["strides"]) >= {"mlp.gate_up_proj", "mlp.down_proj"}
        c = sm.packed_stride_ceiling(Q4, sm.target_class_for(Q4, "down"))
        assert c["D"] == 9_216 and c["widest"] == "mlp.down_proj" and c["ceiling"] == 233_016 and set(c["strides"]) == {"mlp.down_proj"}

    def test_expert_bearing_sets_execute_at_hidden_on_the_moe_toy(self):
        """The MoE option's strides: the routed experts' w13 / w2 and the shared expert's projections are the class's executing
        wrappers (no dense ``mlp.gate_up_proj`` on a MoE block); on the toy they all stop at hidden = 64, so the int32 ceiling
        is the toy's 33,554,431 and nothing binds."""
        c = sm.packed_stride_ceiling(MOE_TOY, sm.target_class_for(MOE_TOY, "mlp"))
        assert c["D"] == 64 and set(c["strides"]) == {"mlp.experts.w13", "mlp.experts.w2", "mlp.shared_expert.gate_up_proj",
                                                      "mlp.shared_expert.down_proj"}
        assert c["strides"]["mlp.experts.w13"] == {"in": 64, "out": 64} and c["strides"]["mlp.experts.w2"] == {"in": 32, "out": 64}
        c = sm.packed_stride_ceiling(MOE_TOY, sm.target_class_for(MOE_TOY, "attn_mlp"))
        assert c["D"] == 160 and c["widest"] == "self_attn.qkv_proj"      # the toy's packed qkv (128 + 2 x 16) is its widest wrapper
        assert sm.mlp_intermediate_elements(MOE_TOY)["elements"] == 2 * 32 + 32 and sm.mlp_intermediate_elements(Q4)["elements"] == 9_216

    def test_kv_facts_and_the_kv_binding(self):
        assert sm.kv_bytes_per_token(Q4) == 8 * 2 * 1024 * 2 == 32_768
        assert abs(sm.kv_bytes_per_seq(Q4, 49152) / 2**30 - 1.5) < 1e-9
        rec = sm.admitted_max_num_seqs(Q4, ("o_proj",), 49152, kv_budget_bytes=3 * sm.kv_bytes_per_seq(Q4, 49152),
                                       requested_S=32)
        assert rec["admitted_S"] == 3 and rec["binding"] == "kv"
        rec = sm.admitted_max_num_seqs(Q4, ("o_proj",), 4096, kv_budget_bytes=1e12, requested_S=200)
        assert rec["admitted_S"] == 127 and rec["binding"] == "stride"
        rec = sm.admitted_max_num_seqs(Q4, ("o_proj",), 4096, kv_budget_bytes=1e12, requested_S=64)
        assert rec["admitted_S"] == 64 and rec["binding"] == "requested"
        # a hybrid's per-sequence recurrent state rides the KV term as a constant (derived from its config by the caller)
        assert sm.kv_bytes_per_seq(Q4, 49152, recurrent_state_bytes=1_000) == sm.kv_bytes_per_seq(Q4, 49152) + 1_000

    def test_estimated_vram_the_split_r64_servers_fit_beside_one_49k_sequence(self):
        """The split servers' LoRA buffers at r = 64 are 414 MiB (S1cc: 3 slots x 138 MiB) and 552 MiB (S1d: 2 x 276 MiB) on the
        Qwen3.5-4B dims and both fit an H200 beside the base-weights default and one 49,152-token sequence of KV (1.5 GiB), as
        do the r = 16 ones; 36 such sequences fit beside the r = 64 difference server, 37 do not."""
        kv = sm.kv_bytes_per_seq(Q4, 49152)
        for rank, lora in ((64, (434_110_464, 578_813_952)), (16, (108_527_616, 144_703_488))):
            cc, d, _s2 = sm.build_manifests(Q4, set_id="mlp", rank=rank)
            est_cc = sm.estimated_vram_bytes(cc, cc.max_lora_rank, Q4, kv_per_seq_bytes=None, S=None, kv_budget_bytes=kv)
            est_d = sm.estimated_vram_bytes(d, d.max_lora_rank, Q4, kv_per_seq_bytes=None, S=None, kv_budget_bytes=kv)
            assert est_cc["fits"] and est_d["fits"]
            assert (est_cc["lora_buffers"], est_d["lora_buffers"]) == lora
        d64 = sm.build_manifests(Q4, set_id="mlp", rank=64)[1]
        assert sm.estimated_vram_bytes(d64, 128, Q4, kv_per_seq_bytes=kv, S=36)["fits"]
        assert not sm.estimated_vram_bytes(d64, 128, Q4, kv_per_seq_bytes=kv, S=37)["fits"]
        # the MoE toy's estimate prices the experts at the server's routed rank (the counterfactual at the cap is 2x on S1d)
        mcc, md, _ = sm.build_manifests(MOE_TOY, set_id="mlp", rank=64)
        est = sm.estimated_vram_bytes(md, 128, MOE_TOY, kv_per_seq_bytes=1.0, S=1)
        assert est["lora_buffers"] == sm.expected_lora_bytes(md, 128, MOE_TOY)["total"] < sm.expected_lora_bytes(md, 128, MOE_TOY, moe_lora_rank=128)["total"]

    def test_trainable_params_from_the_formula(self):
        assert sm.trainable_params(Q4, "mlp", 4) == 4_521_984
        assert sm.trainable_params(Q4, "attn_mlp", 64) == 84_934_656
        assert sm.trainable_params(TOY, "mlp", 4) == 2_304 and sm.trainable_params(TOY, "wo", 4) == 512
        # the MoE toy's count carries the routed experts at r_e (FD 6) and the pair records the map
        assert sm.trainable_params(MOE_TOY, "mlp", 4) == sm.trainable_params(TOY, "mlp", 4) + 2 * 4 * 2 * (64 + 2 * 32 + 32 + 64)
        assert sm.pair_per_expert_rank(MOE_TOY, "mlp", 4) == {"shared_expert": 4, "routed_expert": 2} and sm.pair_per_expert_rank(Q4, "mlp", 4) is None


# --------------------------------------------------------------------------- the plan


class TestPlan:
    def test_exactly_one_launch_per_manifest_cap_topology_window(self):
        plan = sp.build_plan(Q4, model_key=MODEL, model_id=MODEL_HF_ID, revision=REV)
        seen = {}
        for rec in plan["launches"]:
            for name in rec["serves"]:
                m = plan["manifests"][name]
                key = (name, m["max_lora_rank"], m["topology"], rec["L"])
                assert key not in seen, f"{key} in both {seen[key]} and {rec['id']}"
                seen[key] = rec["id"]
                assert rec["cap"] == m["max_lora_rank"] and rec["topology"] == m["topology"]
        ids = [r["id"] for r in plan["launches"]]
        assert len(ids) == len(set(ids))
        # every manifest appears at every window; the ladder points at the ladder window only
        pair_manifests = {p for pr in plan["pairs"] for p in pr["manifests"]}
        for name, m in plan["manifests"].items():
            windows = sorted(L for (n, _c, _t, L) in seen if n == name)
            if name not in pair_manifests:   # a ladder point
                assert windows == [plan["ladder_window"]], name
            else:
                assert windows == plan["windows"], name
        # the S2 engine launches once per window for the 12 (set, rank) S2 manifests
        s2 = [r for r in plan["launches"] if r["topology"] == "S2"]
        assert len(s2) == len(plan["windows"]) and all(len(r["serves"]) == 12 for r in s2)

    def test_the_rank_cap_ladder_is_separate_launches_above_the_natural_cap(self):
        plan = _plan(sets=("mlp",), ranks=(4, 16, 64), windows=(4096,), dims=Q4)
        ladder = sorted((r["cap"], r["manifest"]) for r in plan["launches"] if r["kind"] == "ladder")
        # the ladder keys on the manifest carrying the CONTROL role (S1cc; natural caps 8 (r4), 16 (r16), 64 (r64)):
        # 32 rides r16 and 128 rides r64; 8 / 16 / 64 ARE natural S1cc launches, and the S1d caps 8 / 32 / 128 are the
        # difference server's natural launches, never ladder points
        assert ladder == [(32, "mlp_r16_S1cc_cap32"), (128, "mlp_r64_S1cc_cap128")]
        for cap, name in ladder:
            m = sm.Manifest.from_dict(plan["manifests"][name])
            assert m.topology == "S1cc" and m.max_lora_rank == cap and m.zero.rank == cap
            assert all(a.rank < cap for a in m.genuine)
            assert m.zero.per_expert_rank is None and m.moe_lora_rank is None          # a dense model: no expert rank to re-pin
        natural_cc = sorted(r["cap"] for r in plan["launches"] if r["kind"] == "natural" and r["topology"] == "S1cc")
        natural_d = sorted(r["cap"] for r in plan["launches"] if r["kind"] == "natural" and r["topology"] == "S1d")
        assert natural_cc == [8, 16, 64] and natural_d == [8, 32, 128]
        assert all(r["roles"] == ["control", "constitution"] for r in plan["launches"] if r["kind"] == "ladder")
        # on the MoE toy the re-pinned zero adapter's routed rank follows the cap, and the ladder server's expert buffers with
        # it: cap // 2 active
        moe_plan = _plan(sets=("mlp",), ranks=(4, 16, 64), windows=(4096,), dims=MOE_TOY)
        moe_ladder = sorted((r["cap"], r["manifest"]) for r in moe_plan["launches"] if r["kind"] == "ladder")
        assert moe_ladder == ladder
        for cap, name in moe_ladder:
            m = sm.Manifest.from_dict(moe_plan["manifests"][name])
            assert m.zero.per_expert_rank == cap // 2 == m.moe_lora_rank
            assert next(r for r in moe_plan["launches"] if r["manifest"] == name)["moe_lora_rank"] == cap // 2

    def test_the_difference_server_is_unconditional_and_no_fallback_launch_is_planned(self):
        """Before the ruling the three-server S1a/S1b/S1c fallback rode behind the four-slot S1 (a `condition` on its
        OOM); now S1d is a natural launch in its own right: it runs whether or not S1cc came up, nothing is conditional,
        and next_launch walks S1cc, S1d, S2, then the ladder point."""
        plan = _plan(sets=("mlp",), ranks=(64,), windows=(4096,), dims=Q4)
        # the split servers, the pair's UNION (the served launch class, the probe owner's ruling of 2026-10-11: cells + zero at
        # cap 128, 4 slots — natural, after the servers in id order), the shared S2, then the ladder point
        assert [r["kind"] for r in plan["launches"]] == ["natural", "natural", "natural", "natural", "ladder"]
        assert [r["id"] for r in plan["launches"]] == ["mlp_r64_S1cc_cap64_L4096", "mlp_r64_S1d_cap128_L4096",
                                                       "mlp_r64_single_cap128_L4096", "mlp_r64_S2_cap8_L4096",
                                                       "mlp_r64_S1cc_cap128_L4096"]
        assert all(r["condition"] is None for r in plan["launches"])
        assert not any(r["topology"] in sm.S1_FALLBACKS for r in plan["launches"])
        cc, d, union, s2, ladder = [r["id"] for r in plan["launches"]]
        assert sp.next_launch(plan, {})["id"] == cc
        # a dead S1cc skips nothing: the difference server is next, then the union, then S2, then the ladder point
        oom = {cc: {"status": "oom"}}
        assert sp.next_launch(plan, oom)["id"] == d and not sp.next_launch(plan, oom).get("_skip")
        assert [r["id"] for r in sp.outstanding_launches(plan, oom)] == [d, union, s2, ladder]
        done = {cc: {"status": "oom"}, d: {"status": "passed"}}
        assert sp.next_launch(plan, done)["id"] == union
        done[union] = {"status": "passed"}
        assert sp.next_launch(plan, done)["id"] == s2
        done[s2] = {"status": "passed"}
        assert sp.next_launch(plan, done)["id"] == ladder
        done[ladder] = {"status": "passed"}
        assert sp.next_launch(plan, done) is None and sp.outstanding_launches(plan, done) == []
        # the plan records what each launch serves — the roles and the expert-buffer rank (None on a dense model) — never
        # re-derived from the label
        by = {r["id"]: r for r in plan["launches"]}
        assert by[cc]["roles"] == ["control", "constitution"] and by[cc]["cap"] == 64 and by[cc]["moe_lora_rank"] is None
        assert by[d]["roles"] == ["difference"] and by[d]["cap"] == 128 and by[d]["moe_lora_rank"] is None
        assert by[union]["roles"] == ["control", "constitution", "difference"] and by[union]["cap"] == 128 and by[union]["max_loras"] == 4
        assert by[union]["topology"] == "single" and by[union]["moe_lora_rank"] is None
        assert by[s2]["roles"] == [] and by[s2]["cap"] == 8 and by[s2]["moe_lora_rank"] is None
        assert plan["is_moe"] is False and plan["pairs"][0]["per_expert_rank"] is None
        # the MoE toy's plan carries each server's expert-buffer rank and the pair's per-expert map (the union's: the
        # difference's 2 x 32 = the zero's 128 // 2)
        moe = _plan(sets=("mlp",), ranks=(64,), windows=(4096,), dims=MOE_TOY)
        mby = {r["id"]: r for r in moe["launches"]}
        assert [r["id"] for r in moe["launches"]] == [cc, d, union, s2, ladder] and moe["is_moe"] is True
        assert (mby[cc]["moe_lora_rank"], mby[d]["moe_lora_rank"], mby[union]["moe_lora_rank"], mby[s2]["moe_lora_rank"]) == (32, 64, 64, None)
        assert moe["pairs"][0]["per_expert_rank"] == {"shared_expert": 64, "routed_expert": 32}

    def test_admission_and_estimates_ride_every_launch(self):
        """On the Qwen3.5-4B dims the packed-qkv stride binds before the KV term (32 KiB per token leaves room for hundreds
        of 8,192-token sequences beside the base-weights default, the buffers and the graphs; the int32 ceiling admits 25 at
        8,192 and 4 at 49,152), and the record says which term bound."""
        plan = _plan(sets=("qkvo",), ranks=(4,), windows=(8192, 49152), dims=Q4, requested_S={8192: 64, 49152: 32})
        by = {r["id"]: r for r in plan["launches"]}
        for top, n_slots in (("S1cc", 3), ("S1d", 2)):
            assert by[f"qkvo_r4_{top}_cap8_L8192"]["admitted_S"] == 25 and by[f"qkvo_r4_{top}_cap8_L49152"]["admitted_S"] == 4
            assert by[f"qkvo_r4_{top}_cap8_L8192"]["admission"]["binding"] == "stride" and by[f"qkvo_r4_{top}_cap8_L8192"]["admission"]["kv_S"] > 25
            assert by[f"qkvo_r4_{top}_cap8_L49152"]["admission"]["binding"] == "stride" and by[f"qkvo_r4_{top}_cap8_L49152"]["admission"]["stride_S"] == 4
            assert by[f"qkvo_r4_{top}_cap8_L8192"]["expected_lora_bytes"]["total"] == 3_145_728 * n_slots
            assert by[f"qkvo_r4_{top}_cap8_L8192"]["estimated_vram"]["fits"]
        assert by["qkvo_r4_S1cc_cap8_L8192"]["expected_lora_bytes"]["total"] == 9_437_184

    def test_dry_run_cli_prints_and_touches_nothing(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sp, "default_out", lambda key: tmp_path / "out")
        rc = sp.main(["plan", "--model-key", MODEL, "--model-id", MODEL_HF_ID,
                      "--revision", REV, "--sets", "wo,mlp", "--ranks", "4", "--windows", "4096", "--dry-run"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "probe plan:" in out and "dry run: nothing written" in out
        for lid in ("wo_r4_S1cc_cap8_L4096", "wo_r4_S1d_cap8_L4096", "mlp_r4_S1cc_cap8_L4096", "mlp_r4_S1d_cap8_L4096",
                    "mlp_r4_S2_cap8_L4096", "wo_r4_single_cap8_L4096", "mlp_r4_single_cap8_L4096"):
            assert lid in out, lid
        # the only `single` is each pair's UNION (the served launch class); no legacy four-slot S1, no S1a/S1b/S1c fallback
        assert out.count("_single_") == 2 and "_S1_" not in out and "S1a" not in out
        assert list(tmp_path.iterdir()) == []

    def test_plan_cli_writes_the_plan_file(self, tmp_path):
        rc = sp.main(["plan", "--model-key", MODEL, "--model-id", "org/toy", "--revision", REV,
                      "--sets", "wo", "--ranks", "4", "--windows", "4096", "--ladder-caps", "8", "--out", str(tmp_path)])
        assert rc == 0
        plan = json.loads((tmp_path / sp.PLAN_FILE).read_text())
        # S1cc + S1d + the pair's union (the served launch class); cap 8 is natural, so the one ladder cap adds no point
        assert plan["n_launches"] == len(plan["launches"]) == 3 and plan["revision"] == REV
        assert [r["id"] for r in plan["launches"]] == ["wo_r4_S1cc_cap8_L4096", "wo_r4_S1d_cap8_L4096", "wo_r4_single_cap8_L4096"]
        # a synthetic id has no cached config: the recurrent state is priced 0 (never a typed constant) and a declared value wins
        assert plan["budget"]["recurrent_state_per_seq_bytes"] == 0
        rc = sp.main(["plan", "--model-key", MODEL, "--model-id", "org/toy", "--revision", REV, "--sets", "wo", "--ranks", "4",
                      "--windows", "4096", "--ladder-caps", "8", "--recurrent-state-mib", "2", "--out", str(tmp_path / "declared")])
        assert rc == 0 and json.loads((tmp_path / "declared" / sp.PLAN_FILE).read_text())["budget"]["recurrent_state_per_seq_bytes"] == 2 * 2**20


# --------------------------------------------------------------------------- adapters


class FakeExporter:
    """Writes a PEFT-shaped adapter_config.json per call and counts calls."""

    def __init__(self, refuse_kinds=()):
        self.calls = []
        self.refuse_kinds = set(refuse_kinds)

    def _write(self, out_dir, modules, rank, per_expert_rank):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = {"r": int(rank), "lora_alpha": int(rank), "target_modules": list(modules), "peft_type": "LORA"}
        if per_expert_rank is not None:       # the exporter's experts config (lora_export._general_config): PEFT names + the routed rank
            cfg["target_parameters"] = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
            cfg["rank_pattern"] = {".*\\.experts\\..*": int(per_expert_rank)}
        (out_dir / "adapter_config.json").write_text(json.dumps(cfg))
        (out_dir / "adapter_model.safetensors").write_bytes(b"\0" * 8)

    def export_synthetic(self, out_dir, *, modules, rank, per_expert_rank, seed):
        if "genuine" in self.refuse_kinds:
            raise sp.ExporterUnavailable("lora_export.synthetic_module_set_blob is not on this tree")
        self.calls.append(("genuine", tuple(modules), rank, per_expert_rank, seed))
        self._write(out_dir, modules, rank, per_expert_rank)

    def export_zero(self, out_dir, *, modules, rank, per_expert_rank, bank):
        if "zero" in self.refuse_kinds:
            raise sp.ExporterUnavailable("lora_export.export_zero_adapter_for_class is not on this tree")
        self.calls.append(("zero", tuple(modules), rank, per_expert_rank, None))
        self._write(out_dir, modules, rank, per_expert_rank)

    def target_class(self, adapter_dir):
        cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
        cls = list(cfg["target_modules"]) + (["experts"] if cfg.get("target_parameters") else [])
        return tuple(sorted(cls))


class TestAdapterDirs:
    @pytest.mark.parametrize("dims", [TOY, MOE_TOY], ids=["dense", "moe"])
    def test_builder_writes_every_slots_adapter_config_from_the_manifests(self, tmp_path, dims):
        plan = _plan(sets=("wo", "mlp"), ranks=(4,), windows=(4096,), dims=dims)
        ex = FakeExporter()
        layout = sp.build_adapter_dirs(plan, tmp_path, ex)
        assert not layout["refused"]
        for name, m in plan["manifests"].items():
            for slot in m["adapters"]:
                d = sp.slot_dir(tmp_path, name, slot["name"])
                cfg = json.loads((d / "adapter_config.json").read_text())
                assert cfg["r"] == slot["rank"] and cfg["lora_alpha"] == slot["rank"]
                assert sorted(cfg["target_modules"]) == sorted(slot["lora_modules"])
                assert "experts" not in cfg["target_modules"]   # PEFT semantics, never the vLLM wrapper word
                rec = layout["manifests"][name]["slots"][slot["name"]]
                assert rec["adapter_config"]["r"] == slot["rank"] and rec["inferred_target_class"] == m["target_class"]
                assert rec["per_expert_rank"] == slot["per_expert_rank"]
                if slot["per_expert_rank"] is not None:       # the MoE toy's MLP slots: the exporter was handed the routed rank
                    assert cfg["target_parameters"] == ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
                    assert rec["adapter_config"]["rank_pattern"] == {".*\\.experts\\..*": slot["per_expert_rank"]}
                    assert "experts" in m["target_class"]
                else:
                    assert "target_parameters" not in cfg and rec["adapter_config"]["rank_pattern"] is None
        if dims is MOE_TOY:
            assert any(c[3] is not None for c in ex.calls)     # the routed rank reached the exporter
        else:
            assert all(c[3] is None for c in ex.calls)
        # content-addressed bank: the four rank-1 o_proj carriers are ONE export; the same
        # genuine (modules, rank, r_e, seed) is exported once across manifests
        genuine = [c for c in ex.calls if c[0] == "genuine"]
        assert len(genuine) == len({c[1:] for c in genuine})
        carriers = [c for c in ex.calls if c == ("zero", ("o_proj",), 1, None, None)]
        assert len(carriers) == 1
        assert (tmp_path / sp.ADAPTERS_DIR / "layout.json").is_file()
        # idempotent: a second pass exports nothing new
        n = len(ex.calls)
        sp.build_adapter_dirs(plan, tmp_path, ex)
        assert len(ex.calls) == n

    def test_an_absent_exporter_is_refused_by_name_and_recorded(self, tmp_path):
        plan = _plan(sets=("mlp",), ranks=(4,), windows=(4096,))
        layout = sp.build_adapter_dirs(plan, tmp_path, FakeExporter(refuse_kinds={"zero"}))
        assert layout["refused"] and all("export_zero_adapter_for_class" in v for v in layout["refused"].values())
        for name in ("mlp_r4_S1cc", "mlp_r4_S1d"):
            m = sm.Manifest.from_dict(plan["manifests"][name])
            assert sp.missing_adapter_dirs(tmp_path, m) == ["none__lora0"]

    def test_compare_adapter_layouts(self):
        ours = {"a.mlp.gate_proj.lora_A.weight": {"shape": [8, 64], "dtype": "F32"},
                "a.mlp.down_proj.lora_A.weight": {"shape": [8, 32], "dtype": "F32"}}
        assert sp.compare_adapter_layouts(ours, dict(ours))["pass"]
        assert sp.compare_adapter_layouts(ours, dict(reversed(list(ours.items()))))["pass"]      # names and shapes, not order
        bad = {**ours, "a.mlp.down_proj.lora_A.weight": {"shape": [4, 32], "dtype": "F32"}}
        r = sp.compare_adapter_layouts(ours, bad)
        assert not r["pass"] and r["shape_mismatch"] == ["a.mlp.down_proj.lora_A.weight"]
        missing = {k: v for k, v in ours.items() if "down_proj" not in k}
        r = sp.compare_adapter_layouts(ours, missing)
        assert not r["pass"] and r["only_ours"] == ["a.mlp.down_proj.lora_A.weight"] and r["n_peft"] == 1
        assert r["expert_header_order_agrees"] is True                       # no expert pair in a dense layout: vacuously agreed
        # the MoE layout: the expert pairs' HEADER ORDER is part of the agreement (experts.base_layer = gate_up before experts = down)
        moe = {"a.mlp.experts.base_layer.lora_A.weight": {"shape": [8, 64], "dtype": "F32"},
               "a.mlp.experts.lora_A.weight": {"shape": [8, 32], "dtype": "F32"}}
        assert sp.compare_adapter_layouts(moe, dict(moe))["pass"]
        swapped = dict(reversed(list(moe.items())))
        r = sp.compare_adapter_layouts(moe, swapped)
        assert not r["pass"] and not r["expert_header_order_agrees"] and not r["only_ours"] and not r["shape_mismatch"]
        bad = {**moe, "a.mlp.experts.lora_A.weight": {"shape": [4, 32], "dtype": "F32"}}
        r = sp.compare_adapter_layouts(moe, bad)
        assert not r["pass"] and r["shape_mismatch"] == ["a.mlp.experts.lora_A.weight"]


# --------------------------------------------------------------------------- checks


#: a clean engine log: nothing the checks read as a failure
ENGINE_LOG = "INFO engine up\n"


def _probe(first_id=5, first_lp=-0.10, second=(7, -2.0), n=3):
    rows = [{"token": f"token_id:{first_id}", "logprob": first_lp,
             "top": [[f"token_id:{first_id}", first_lp], [f"token_id:{second[0]}", second[1]]]}]
    rows += [{"token": f"token_id:{i}", "logprob": -0.5, "top": [[f"token_id:{i}", -0.5]]} for i in range(n - 1)]
    return {"token_ids": [first_id] + list(range(n - 1)), "rows": rows, "finish_reason": "length", "text": "x"}


class TestEvaluateChecks:
    def _run(self, *, log_text, zero_probe, genuine_probe, report, set_id="mlp", dims=TOY, **kw):
        plan = _plan(sets=(set_id,), ranks=(4,), windows=(4096,), dims=dims)
        rec = next(r for r in plan["launches"] if r["topology"] == "S1cc")     # the server carrying the control adapter
        m = sm.Manifest.from_dict(plan["manifests"][rec["manifest"]])
        probes = {a.name: (zero_probe if a.kind != "genuine" else genuine_probe) for a in m.adapters}
        return m, sp.evaluate_checks(manifest=m, launch_rec=rec, dims=sp._Dims(plan["dims"]), log_text=log_text,
                                     models_listed=[a.name for a in m.adapters] + ["org/toy"],
                                     base_probe=_probe(), adapter_probes=probes, worker_report=report, **kw)

    def test_buffer_bytes_and_executing_stride_are_hard_gates(self):
        """A measured buffer outside the 5 % law, a stride wider than the manifest's, and — required by default — a
        MISSING or PARTIAL measurement all FAIL the launch (before: computed, never in `failed`; a buffer-law change
        could certify a pair and size the H200 wrongly)."""
        plan = _plan(sets=("mlp",), ranks=(4,), windows=(4096,))
        exp = next(r for r in plan["launches"] if r["topology"] == "S1cc")["expected_lora_bytes"]["total"]
        strides = {"L0.mlp.gate_up_proj": {"in": 64, "out": 64}}
        ok = dict(log_text=ENGINE_LOG, zero_probe=_probe(), genuine_probe=_probe(first_lp=-0.3))
        # the injected mismatch: 20 % over the law
        m, c = self._run(**ok, report={"lora_execution_module_lora_buffer_bytes": int(exp * 1.2),
                                       "lora_execution_module_strides": strides})
        assert not c["pass"] and c["failed"] == ["buffer_bytes"]
        assert c["buffer_bytes"]["status"] == "measured" and c["buffer_bytes"]["pass"] is False
        assert "buffer law" in c["buffer_bytes"]["reason"]
        # within the law: passes, and nothing else fails
        m, c = self._run(**ok, report={"lora_execution_module_lora_buffer_bytes": int(exp * 1.03),
                                       "lora_execution_module_strides": strides})
        assert c["pass"] and c["failed"] == [] and c["buffer_bytes"]["pass"] is True
        # a MISSING measurement fails when required (the default)
        m, c = self._run(**ok, report={"lora_execution_module_strides": strides})
        assert c["failed"] == ["buffer_bytes"] and c["buffer_bytes"]["status"] == "not_measured"
        assert c["buffer_bytes"]["pass"] is False and c["buffer_bytes"]["required"] is True
        # a PARTIAL measurement (a wrapper's buffers unread) fails too — the total does not cover the manifest
        m, c = self._run(**ok, report={"lora_execution_module_lora_buffer_bytes": exp,
                                       "lora_execution_module_lora_buffer_unread": ["L1.mlp.down_proj"],
                                       "lora_execution_module_strides": strides})
        assert c["failed"] == ["buffer_bytes"] and c["buffer_bytes"]["status"] == "partially_measured"
        # a wrapper executing wider than the manifest's widest stride fails
        m, c = self._run(**ok, report={"lora_execution_module_lora_buffer_bytes": exp,
                                       "lora_execution_module_strides": {"x": {"in": 64, "out": 4096}}})
        assert c["failed"] == ["executing_stride"] and c["executing_stride"]["reported_D"] == 4096
        assert c["executing_stride"]["reported_D"] > c["executing_stride"]["expected_D"]
        # missing strides fail when required
        m, c = self._run(**ok, report={"lora_execution_module_lora_buffer_bytes": exp})
        assert c["failed"] == ["executing_stride"] and c["executing_stride"]["status"] == "not_reported"
        # both missing: both named
        m, c = self._run(**ok, report={})
        assert c["failed"] == ["buffer_bytes", "executing_stride"]
        # not required: a missing measurement is None, never a pass, never a failure
        m, c = self._run(**ok, report={}, require_measurements=False)
        assert c["pass"] and c["failed"] == [] and c["buffer_bytes"]["pass"] is None and c["executing_stride"]["pass"] is None
        # and the verdict carries a buffer-law failure as the pair's evidence-backed exclusion (the difference server unrun:
        # a failed check on the control's server is evidence, so the pair is decided without it)
        rec = next(r for r in plan["launches"] if r["topology"] == "S1cc")
        res = _passed(rec, m)
        res["checks"]["buffer_bytes"] = {"status": "measured", "measured": int(exp * 1.2), "pass": False}
        res["checks"]["failed"] = ["buffer_bytes"]
        res["checks"]["pass"] = False
        res["status"] = "failed"
        # the pair's union (the served launch class) carries the control adapter too: the same buffer-law failure on it — else the
        # role stays UNDECIDED while a launch carrying it could still certify it (the role-keyed rule), never excluded
        urec = next(r for r in plan["launches"] if r["topology"] == sm.TOPOLOGY_SINGLE and r["L"] == rec["L"])
        ures = _passed(urec, sp.plan_manifest(plan, urec["manifest"]))
        ures["checks"].update(buffer_bytes={"status": "measured", "measured": int(exp * 1.2), "pass": False}, failed=["buffer_bytes"])
        ures["checks"]["pass"] = False
        ures["status"] = "failed"
        s2 = [r["id"] for r in plan["launches"] if r["topology"] == "S2"]
        results = {rec["id"]: res, urec["id"]: ures, **_fixture_results(plan, outcomes={i: ("passed",) for i in s2})}
        v, allowed = sp.write_verdict(plan, results, probe_job_id="1")
        assert allowed["pairs"] == [] and allowed["excluded"][0]["reason"] == "failed:buffer_bytes"
        assert allowed["excluded"][0]["decided"] is True and v["status"] == "complete"

    def test_all_assertions_pass_on_a_clean_launch(self):
        plan = _plan(sets=("mlp",), ranks=(4,), windows=(4096,))
        exp = next(r for r in plan["launches"] if r["topology"] == "S1cc")["expected_lora_bytes"]["total"]
        report = {"lora_execution_module_lora_buffer_bytes": {"L0.mlp.gate_up_proj": int(exp * 1.02)},
                  "kv_num_gpu_blocks": 100, "kv_block_size": 16, "max_num_seqs": 64,
                  "lora_execution_module_strides": {"x": {"in": 64, "out": 64}}}
        m, c = self._run(log_text=ENGINE_LOG, zero_probe=_probe(), genuine_probe=_probe(first_lp=-0.30), report=report)
        assert c["pass"] and c["failed"] == []
        assert c["buffer_bytes"]["status"] == "measured" and c["buffer_bytes"]["pass"]
        assert c["memory"]["kv_capacity_tokens"] == 1600 and c["executing_stride"]["pass"]
        assert c["log_no_unwrapped"]["pass"] and c["models_listed"]["pass"]
        assert set(c) == {"log_moe_lora_line", "log_no_unwrapped", "log_triton_backend", "models_listed", "adapters", "buffer_bytes",
                          "memory", "executing_stride", "launch_class", "mixed_rank", "companion_check", "failed", "pass"}
        # the attested launch class (2026-10-11): this report names no lora_max_lora_rank / lora_max_loras — not judged here, the
        # fold refuses the class; never a hard failure for want of the pair (TestServedClassUnion pins the equality check)
        assert c["launch_class"]["status"] == "not_reported" and c["launch_class"]["pass"] is None
        # a dense launch expects no fused-MoE line and no expert-only check applies
        assert c["log_moe_lora_line"] == {"expected": False, "found": False, "pass": True}
        assert c["log_triton_backend"]["applicable"] is False and c["mixed_rank"]["applicable"] is False
        assert c["companion_check"] == {"applicable": False, "pass": None}

    def test_all_assertions_pass_on_a_clean_moe_launch_and_the_moe_line_is_a_hard_gate_there(self):
        """The MoE option's checks: on a routed-experts plan the engine's fused-MoE LoRA line is a HARD check, the Triton
        evidence is recorded (never a gate), the MLP sets' genuine adapters ARE the mixed-rank variant (attention/shared R,
        experts r_e) and the per-adapter record names each one's routed rank."""
        plan = _plan(sets=("mlp",), ranks=(4,), windows=(4096,), dims=MOE_TOY)
        exp = next(r for r in plan["launches"] if r["topology"] == "S1cc")["expected_lora_bytes"]["total"]
        report = {"lora_execution_module_lora_buffer_bytes": {"L0.mlp.experts": int(exp * 1.02)},
                  "kv_num_gpu_blocks": 100, "kv_block_size": 16, "max_num_seqs": 64,
                  "lora_execution_module_strides": {"L0.mlp.experts": {"in": 64, "out": 64}}}
        m, c = self._run(log_text=f"INFO {sp.LOG_MOE_LORA_LINE}\nINFO fused MoE experts on the triton backend\n", zero_probe=_probe(),
                         genuine_probe=_probe(first_lp=-0.30), report=report, dims=MOE_TOY, is_moe=True)
        assert c["pass"] and c["failed"] == [] and m.wraps_experts
        assert c["log_moe_lora_line"] == {"expected": True, "found": True, "pass": True}
        assert c["log_triton_backend"]["applicable"] and c["log_triton_backend"]["found"] and c["log_triton_backend"]["pass"]
        assert c["mixed_rank"]["applicable"] and c["mixed_rank"]["pass"]
        assert c["mixed_rank"]["adapters"] == [{"name": a.name, "attention_shared_rank": 4, "expert_rank": 2} for a in m.genuine]
        assert all(c["adapters"][a.name]["per_expert_rank"] == a.per_expert_rank for a in m.adapters)
        assert c["companion_check"] == {"applicable": False, "pass": None}         # the companion check is the down set's
        # the fused-MoE line missing on a MoE plan fails by name; a dense plan never expects it
        m, c = self._run(log_text=ENGINE_LOG, zero_probe=_probe(), genuine_probe=_probe(first_lp=-0.3), report=report, dims=MOE_TOY, is_moe=True)
        assert "log_moe_lora_line" in c["failed"] and c["log_moe_lora_line"] == {"expected": True, "found": False, "pass": False}
        m, c = self._run(log_text=ENGINE_LOG, zero_probe=_probe(), genuine_probe=_probe(first_lp=-0.3), report={}, is_moe=False)
        assert c["log_moe_lora_line"]["pass"] and "log_moe_lora_line" not in c["failed"]

    def test_down_set_companion_check_rides_the_class_zero_on_the_moe_toy(self):
        """(7) the down set's zero gate_up companion changes no logit: the class zero adapter (zero companion pair + zero down) is
        bit-identical to base and the genuine down adapter differs; a moving zero fails the check by name. Dense ``down`` has no
        companion (no expert pair): not applicable."""
        m, c = self._run(log_text=sp.LOG_MOE_LORA_LINE, zero_probe=_probe(), genuine_probe=_probe(first_lp=-0.3),
                         report={}, set_id="down", dims=MOE_TOY, is_moe=True)
        assert c["companion_check"]["applicable"] and c["companion_check"]["pass"] and "companion_check" not in c["failed"]
        assert c["companion_check"]["basis"].startswith("class zero adapter")
        m, c = self._run(log_text=sp.LOG_MOE_LORA_LINE, zero_probe=_probe(first_lp=-0.11),
                         genuine_probe=_probe(first_lp=-0.3), report={}, set_id="down", dims=MOE_TOY, is_moe=True)
        assert not c["companion_check"]["pass"] and "companion_check" in c["failed"]
        m, c = self._run(log_text=ENGINE_LOG, zero_probe=_probe(), genuine_probe=_probe(first_lp=-0.3), report={}, set_id="down")
        assert c["companion_check"] == {"applicable": False, "pass": None}

    def test_a_silent_no_op_adapter_and_a_moving_zero_fail(self):
        m, c = self._run(log_text=ENGINE_LOG, zero_probe=_probe(), genuine_probe=_probe(), report={})
        assert not c["pass"] and any(f.startswith("adapters/control__lora") for f in c["failed"])
        m, c = self._run(log_text=ENGINE_LOG, zero_probe=_probe(first_lp=-0.1000001),
                         genuine_probe=_probe(first_lp=-0.3), report={})
        assert "adapters/none__lora0" in c["failed"]

    def test_an_unwrapped_warning_fails_and_absent_bytes_are_not_measured(self):
        m, c = self._run(log_text="", zero_probe=_probe(), genuine_probe=_probe(first_lp=-0.3), report={})
        assert c["log_no_unwrapped"]["pass"] and c["buffer_bytes"]["status"] == "not_measured" and c["failed"] == ["buffer_bytes", "executing_stride"]
        m, c = self._run(log_text=ENGINE_LOG + "WARNING module x could not be wrapped\n",
                         zero_probe=_probe(), genuine_probe=_probe(first_lp=-0.3), report={})
        assert "log_no_unwrapped" in c["failed"] and c["log_no_unwrapped"]["found"] == ["WARNING module x could not be wrapped"]

    def test_classify_failure(self):
        assert sp.classify_failure("torch.OutOfMemoryError: CUDA out of memory", None)[0] == "oom"
        assert sp.classify_failure("ValueError: LoRA rank 16 is greater than max_lora_rank 8.", None) == (
            "refused", "LoRA rank 16 is greater than max_lora_rank 8")
        assert sp.classify_failure("", sp.LaunchRefused("composer_target_class_mismatch: x"))[0] == "refused"
        assert sp.classify_failure("", RuntimeError("boom"))[0] == "error"
        # the worker's routed-rank refusal (native_moe_lora.REFUSAL_PHRASE) is a REFUSED launch naming the ranks
        from sisteer.capeval.serving import native_moe_lora as nml
        msg = nml.REFUSAL_PHRASE.format(incoming=8, routed=4) + " on Fake slot 1"
        assert sp.classify_failure(f"ERROR {msg}\n", None) == ("refused", nml.REFUSAL_PHRASE.format(incoming=8, routed=4))
        assert sp.classify_failure("", RuntimeError(msg))[0] == "refused"


# --------------------------------------------------------------------------- verdict


def _passed(rec, m, *, control=True, difference=True, engine_S=None, reference=REF_OK, stride=True, dims=TOY):
    """A passed launch record as the driver writes one: the per-adapter checks, the MEASURED buffer bytes and executing
    stride (both hard gates — the fold reads the stride), the engine's resolved S, and the HF+PEFT reference
    (``reference=None`` = none recorded, the refusal case)."""
    per = {}
    for a in m.adapters:
        ok = {"control": control, "difference": difference}.get(a.role, True)
        per[a.name] = {"kind": a.kind, "role": a.role, "rank": a.rank, "pass": ok,
                       "reason": None if ok else f"{a.role} failed"}
    strides = sm.executing_strides(dims, m.target_class)
    D = max(max(v["in"], v["out"]) for v in strides.values())
    res = {"launch_id": rec["id"], "manifest": m.name, "serves": rec["serves"], "status": "passed",
           "launch_S": max(1, int(rec["admitted_S"])),
           "checks": {"adapters": per, "failed": [f"adapters/{n}" for n, v in per.items() if not v["pass"]],
                      "pass": all(v["pass"] for v in per.values()),
                      "buffer_bytes": {"status": "measured", "measured": rec["expected_lora_bytes"]["total"], "pass": True},
                      "executing_stride": ({"expected_D": D, "reported_D": D, "pass": True, "n_wrappers": len(strides),
                                            "status": "measured"} if stride else
                                           {"expected_D": D, "reported_D": None, "status": "not_reported", "pass": None}),
                      "memory": {"kv_capacity_tokens": 4096}},
           # the engine's attested class rides beside its S (the fold reads both; 53bd76ed0): the launch's own rank cap and its manifest's
           # adapter count — what a real lora-native worker report carries as lora_max_lora_rank / lora_max_loras
           "engine": {"max_num_seqs_engine": engine_S if engine_S is not None else rec["admitted_S"],
                      "max_lora_rank": int(rec["cap"]), "max_loras": int(m.max_loras)}}
    if not res["checks"]["pass"]:
        res["status"] = "failed"
    if reference is not None:
        res["reference_check"] = reference
    return res


def _fixture_results(plan, *, outcomes, dims=TOY):
    """outcomes: launch id -> ("passed", kw) | ("oom"|"refused",) ; unlisted launches are not run."""
    results = {}
    for rec in plan["launches"]:
        o = outcomes.get(rec["id"])
        if o is None:
            continue
        m = sm.Manifest.from_dict(plan["manifests"][rec["manifest"]])
        if o[0] == "passed":
            results[rec["id"]] = _passed(rec, m, dims=dims, **(o[1] if len(o) > 1 else {}))
        else:
            results[rec["id"]] = {"launch_id": rec["id"], "manifest": m.name, "status": o[0],
                                  "failure": {"class": o[0], "message": (o[1] if len(o) > 1 else o[0])}}
    return results


def _with_union(plan, outcomes, default=("passed",)):
    """``outcomes`` plus ``default`` for every UNION launch (the pair's served launch class, the probe owner's ruling of 2026-10-11)
    the fixture did not name: the union carries the same adapters as the pair's servers in one engine, so a fixture that names
    what the servers did names the union's outcome explicitly where it differs (a dead server's twin dies, a failed adapter fails)
    and lets it pass beside passing servers otherwise."""
    out = dict(outcomes)
    for r in plan["launches"]:
        if r["topology"] == sm.TOPOLOGY_SINGLE and r["id"] not in out:
            out[r["id"]] = default
    return out


class TestVerdict:
    def test_allowed_pairs_schema_and_the_documented_exclusions(self, tmp_path):
        plan = _plan(sets=("wo", "mlp"), ranks=(4, 16), windows=(4096, 49152))
        s2_ids = [r["id"] for r in plan["launches"] if r["topology"] == "S2"]
        outcomes = {
            # wo r4: both servers passed at both windows -> allowed, S1ccS1d, caps 8 / 8
            "wo_r4_S1cc_cap8_L4096": ("passed",), "wo_r4_S1d_cap8_L4096": ("passed",),
            "wo_r4_S1cc_cap8_L49152": ("passed", {"engine_S": 10}), "wo_r4_S1d_cap8_L49152": ("passed", {"engine_S": 10}),
            # wo r16: the difference adapter failed on ITS server -> excluded
            "wo_r16_S1cc_cap16_L4096": ("passed",), "wo_r16_S1cc_cap16_L49152": ("passed",),
            "wo_r16_S1d_cap32_L4096": ("passed", {"difference": False}), "wo_r16_S1d_cap32_L49152": ("passed",),
            # mlp r4: S1cc + S1d at both windows + S2 -> allowed S1ccS1dS2
            "mlp_r4_S1cc_cap8_L4096": ("passed",), "mlp_r4_S1d_cap8_L4096": ("passed",),
            "mlp_r4_S1cc_cap8_L49152": ("passed", {"engine_S": 21}), "mlp_r4_S1d_cap8_L49152": ("passed", {"engine_S": 21}),
            # mlp r16: the control server came up, the difference server OOMs at cap 32 -> excluded naming S1d
            "mlp_r16_S1cc_cap16_L4096": ("passed",), "mlp_r16_S1d_cap32_L4096": ("oom",),
        }
        for sid in s2_ids:
            outcomes[sid] = ("passed",)
        # each pair's UNION (the served launch class, 2026-10-11) beside its servers: the same adapters in one engine — wo r16's
        # difference fails on it too, mlp r16's dies with its difference server at cap 32 and stays unrun at 49,152 like them
        outcomes.update({"wo_r4_single_cap8_L49152": ("passed", {"engine_S": 10}), "mlp_r4_single_cap8_L49152": ("passed", {"engine_S": 21}),
                         "wo_r16_single_cap32_L4096": ("passed", {"difference": False}), "mlp_r16_single_cap32_L4096": ("oom",)})
        outcomes = _with_union(plan, outcomes)
        del outcomes["mlp_r16_single_cap32_L49152"]
        results = _fixture_results(plan, outcomes=outcomes)
        verdict, allowed = sp.write_verdict(plan, results, probe_job_id="777", out=tmp_path, verdict_uri="file://v")
        assert set(allowed) >= {"model_id", "served_hf_id", "revision", "probe_job_id", "pairs", "excluded", "verdict_uri", "memo_inputs"}
        assert allowed["probe_job_id"] == "777" and allowed["verdict_uri"] == "file://v"
        pairs = {(p["rank"], p["set"]): p for p in allowed["pairs"]}
        assert set(pairs) == {(4, "wo"), (4, "mlp")}
        wo = pairs[(4, "wo")]
        assert set(wo) >= {"rank", "set", "lora_modules", "per_expert_rank", "max_lora_rank_cap", "topology",
                           "trainable_params", "admitted_max_num_seqs", "reference_check_basis"}
        assert wo["topology"] == "S1ccS1d" and wo["max_lora_rank_cap"] == 8 and wo["lora_modules"] == ["o_proj"] and wo["per_expert_rank"] is None
        assert wo["caps_by_server"] == {"S1cc": 8, "S1d": 8} and wo["moe_lora_rank_by_server"] == {"S1cc": None, "S1d": None}
        # the role-keyed rule's audit trail: at every window, the launch that certified each required role
        assert wo["certifying_by_window"] == {"4096": {"control": "wo_r4_S1cc_cap8_L4096", "difference": "wo_r4_S1d_cap8_L4096"},
                                              "49152": {"control": "wo_r4_S1cc_cap8_L49152", "difference": "wo_r4_S1d_cap8_L49152"}}
        assert wo["certified_by"] == ["wo_r4_S1cc_cap8_L4096", "wo_r4_S1d_cap8_L4096", "wo_r4_S1cc_cap8_L49152", "wo_r4_S1d_cap8_L49152"]
        assert wo["warnings"] == []
        # the planned S at 4,096 is pinned BY VALUE — on the toy no stride or KV pressure binds at 4,096, so the admitted
        # S is the requested production concurrency (64); the 49,152 value is the fixture's measured engine S.
        # (Before: `{"4096": x if False else wo[...]["4096"]}` compared the value with itself.)
        wo_4096 = next(r for r in plan["launches"] if r["id"] == "wo_r4_S1cc_cap8_L4096")
        assert wo_4096["admission"]["binding"] == "requested" and wo_4096["admitted_S"] == 64
        assert wo["admitted_max_num_seqs"] == {"4096": 64, "49152": 10}
        assert verdict["status"] == "complete" and verdict["certifying"] is True and verdict["undecided_pairs"] == []
        assert wo["windows_measured"] == [4096, 49152] and wo["reference_check_basis"] == sp.BASIS_HF_PEFT
        assert verdict["unreferenced_pairs"] == [] and allowed["unreferenced_pairs"] == []
        # a complete, certifying verdict FOLDS the geometry: no site dir configured here, so the
        # record lands beside the verdict only, and says so
        fold = verdict["geometry_fold"]
        assert fold["status"] == "record_only" and fold["site_path"] is None and "SISTEER_GEOMETRY_DIR" in fold["reason"]
        assert fold["record"] == str(tmp_path / sp.GEOMETRY_RECORD_FILE) and (tmp_path / sp.GEOMETRY_RECORD_FILE).is_file()
        # PER SET: each allowed set's D and budget — the budget = the largest engine S x L over THAT
        # set's allowed pairs' manifests (S2 engines included — every one is an engine of this model that came up with usable
        # KV); derived from the fixture, never a literal. The summary is reading-only.
        assert sorted(fold["sets"]) == ["mlp", "wo"]
        for sid in ("wo", "mlp"):
            products = {lid: r["engine"]["max_num_seqs_engine"] * next(l["L"] for l in plan["launches"] if l["id"] == lid)
                        for lid, r in results.items() if r["status"] == "passed"
                        and any(m in r["serves"] for pr in plan["pairs"] if (pr["rank"], pr["set"]) == (4, sid) for m in pr["manifests"])}
            fs = fold["sets"][sid]
            assert fs["batched_token_budget"] == max(products.values()) and products[fs["basis_launch"]] == fs["batched_token_budget"]
            assert fs["lora_max_module_dim"] == 64 and fs["allowed_pairs"] == [[4, sid]]    # the toy's o_in / hidden = 64 (TOY dims)
        assert fold["sets"]["mlp"]["batched_token_budget"] >= 21 * 49152           # at least the mlp S1's measured 21 x 49,152
        assert fold["summary"]["lora_max_module_dim_max"] == 64 and "never sizes a standup" in fold["summary"]["note"]
        mlp = pairs[(4, "mlp")]
        assert mlp["topology"] == "S1ccS1dS2" and mlp["lora_modules"] == ["down_proj", "gate_proj", "up_proj"]
        assert mlp["caps_by_server"] == {"S1cc": 8, "S1d": 8} and mlp["moe_lora_rank_by_server"] == {"S1cc": None, "S1d": None}
        assert mlp["per_expert_rank"] is None                                   # the dense toy: no routed expert anywhere
        assert mlp["certified_by"][-1] == next(i for i in s2_ids if i.endswith("_L4096"))     # S2 at the ladder window closes it
        assert mlp["admitted_max_num_seqs"]["49152"] == 21
        assert mlp["trainable_params"] == sm.trainable_params(TOY, "mlp", 4)
        excluded = {(e["rank"], e["set"]): e["reason"] for e in allowed["excluded"]}
        assert excluded[(16, "wo")].startswith("difference_failed:")
        assert excluded[(16, "mlp")] == "difference_on_S1d:oom_at_R=32"
        assert (tmp_path / sp.ALLOWED_PAIRS_FILE).is_file() and (tmp_path / sp.VERDICT_FILE).is_file()
        assert verdict["reference_check_basis"] == sp.BASIS_HF_PEFT
        assert [l["status"] for l in verdict["launches"] if l["id"] == "mlp_r16_S1d_cap32_L49152"] == ["not_run"]

    def test_not_run_ceiling_and_s2_failures_exclude(self, tmp_path):
        plan = _plan(sets=("qkvo",), ranks=(4,), windows=(4096, 49152))
        s2 = [r["id"] for r in plan["launches"] if r["topology"] == "S2"]
        cc, d = "qkvo_r4_S1cc_cap8", "qkvo_r4_S1d_cap8"
        # nothing run -> not_run
        _v, allowed = sp.write_verdict(plan, {}, probe_job_id="1")
        assert allowed["pairs"] == [] and allowed["excluded"][0]["reason"] == "not_run"
        # both servers passed at 4096 but the plan admits S < 1 at 49152 (unrun) -> ceiling
        plan2 = json.loads(json.dumps(plan))
        for rec in plan2["launches"]:
            if rec["id"] in (f"{cc}_L49152", f"{d}_L49152"):
                rec["admitted_S"] = 0
        results = _fixture_results(plan2, outcomes={f"{cc}_L4096": ("passed",), f"{d}_L4096": ("passed",),
                                                    **{i: ("passed",) for i in s2}})
        _v, allowed = sp.write_verdict(plan2, results, probe_job_id="1")
        assert allowed["excluded"][0]["reason"] == "ceiling_S<1_at_L=49152"
        # S2 refused -> the three-server pair is excluded by its residual server
        results = _fixture_results(plan, outcomes={f"{cc}_L4096": ("passed",), f"{d}_L4096": ("passed",),
                                                   f"{cc}_L49152": ("passed",), f"{d}_L49152": ("passed",),
                                                   s2[0]: ("refused", "refused:mamba_cache_blocks")})
        _v, allowed = sp.write_verdict(plan, results, probe_job_id="1")
        assert allowed["excluded"][0]["reason"].startswith("s2_refused:")

    @pytest.mark.parametrize("dims,moe_by_server,per_expert", [
        (Q4, {"S1cc": None, "S1d": None}, None),
        (MOE_TOY, {"S1cc": 32, "S1d": 64}, {"shared_expert": 64, "routed_expert": 32}),
    ], ids=["dense", "moe"])
    def test_the_split_servers_certify_a_rank64_mlp_pair_and_a_dead_difference_server_excludes_it_by_name(self, tmp_path, dims,
                                                                                                           moe_by_server, per_expert):
        """The role-keyed rule at r = 64: S1cc (cap 64) certifies the control role, S1d (cap 128) the difference, S2 the
        carriers -> allowed as S1ccS1dS2 with the caps — and, on the MoE toy, the expert-buffer ranks (32 / 64) — per
        server and the admitted S = the min over the two engines; S1d dying names the difference's server. (Before the
        ruling this shape was the S1a/S1b/S1c VRAM fallback behind a four-slot S1 padded to the difference's cap.)"""
        plan = _plan(sets=("mlp",), ranks=(64,), windows=(4096,), dims=dims)
        s2 = [r["id"] for r in plan["launches"] if r["topology"] == "S2"][0]
        cc, d, u = "mlp_r64_S1cc_cap64_L4096", "mlp_r64_S1d_cap128_L4096", "mlp_r64_single_cap128_L4096"
        # the pair's union (cells + zero at cap 128, 4 slots) passed beside the servers: it certifies the CLASS, never a role here
        results = _fixture_results(plan, outcomes={cc: ("passed", {"engine_S": 8}), d: ("passed", {"engine_S": 2}), s2: ("passed",),
                                                   u: ("passed", {"engine_S": 2})}, dims=dims)
        _v, allowed = sp.write_verdict(plan, results, probe_job_id="1")
        assert len(allowed["pairs"]) == 1
        p = allowed["pairs"][0]
        assert p["topology"] == "S1ccS1dS2" and p["max_lora_rank_cap"] == 128
        assert p["caps_by_server"] == {"S1cc": 64, "S1d": 128} and p["moe_lora_rank_by_server"] == moe_by_server
        assert p["per_expert_rank"] == per_expert
        assert p["certifying_by_window"] == {"4096": {"control": cc, "difference": d}} and p["certified_by"] == [cc, d, s2]
        assert p["admitted_max_num_seqs"] == {"4096": 2} and p["warnings"] == []
        # the difference's server dying excludes the pair naming that server (the control's launch passed); the union, which
        # carries the same difference adapter in a larger engine, dies with it
        results[d] = {"launch_id": d, "status": "oom", "failure": {"class": "oom", "message": "oom"}}
        results[u] = {"launch_id": u, "status": "oom", "failure": {"class": "oom", "message": "oom"}}
        _v, allowed = sp.write_verdict(plan, results, probe_job_id="1")
        assert allowed["pairs"] == [] and allowed["excluded"][0]["reason"] == "difference_on_S1d:oom_at_R=128"
        assert allowed["excluded"][0]["decided"] is True

    def test_reference_check_basis_and_a_mismatch(self, tmp_path):
        plan = _plan(sets=("wo",), ranks=(4,), windows=(4096,), ladder_caps=())
        cc, d, u = "wo_r4_S1cc_cap8_L4096", "wo_r4_S1d_cap8_L4096", "wo_r4_single_cap8_L4096"
        results = _fixture_results(plan, outcomes={cc: ("passed", {"reference": REF_OK}), d: ("passed", {"reference": REF_OK}),
                                                   u: ("passed", {"reference": REF_OK})})
        v, allowed = sp.write_verdict(plan, results, probe_job_id="1")
        assert allowed["pairs"][0]["reference_check_basis"] == sp.BASIS_HF_PEFT and v["reference_check_basis"] == sp.BASIS_HF_PEFT
        bad = {"status": "fail", "basis": sp.BASIS_HF_PEFT, "per_adapter": {"control__lora": {"status": "fail"}}}
        results = _fixture_results(plan, outcomes={cc: ("passed", {"reference": bad}), d: ("passed", {"reference": REF_OK}),
                                                   u: ("passed", {"reference": REF_OK})})
        _v, allowed = sp.write_verdict(plan, results, probe_job_id="1")
        assert allowed["excluded"][0]["reason"] == "reference_mismatch:control__lora"

    def test_no_reference_evidence_is_uncertified_never_a_vllm_side_certification(self, tmp_path):
        """A pair that passed every vLLM-side check but has NO HF+PEFT reference evidence is `reference_missing`, the
        verdict is `uncertified` and non-certifying, allowed_pairs.json is NOT written, the CLI exits EXIT_UNCERTIFIED
        (6, never the requeue code 4) with RESULT state=UNCERTIFIED, and the memo banners it. Fail-before: the pair was
        allowed with reference_check_basis=vllm_side_checks_1_3_5_7."""
        plan = _plan(sets=("wo",), ranks=(4,), windows=(4096,), ladder_caps=())
        cc, d, u = "wo_r4_S1cc_cap8_L4096", "wo_r4_S1d_cap8_L4096", "wo_r4_single_cap8_L4096"
        # (a) no reference_check recorded at all (run --no-reference): every CERTIFYING launch is named — the pair's union passed
        # beside the servers but certifies no role, so the reason never names it
        results = _fixture_results(plan, outcomes={cc: ("passed", {"reference": None}), d: ("passed", {"reference": None}),
                                                   u: ("passed", {"reference": None})})
        v, allowed = sp.write_verdict(plan, results, probe_job_id="1", out=tmp_path)
        assert v["status"] == "uncertified" and v["certifying"] is False and v["unreferenced_pairs"] == [[4, "wo"]]
        missing = "no reference_check recorded (run --no-reference, or the reference phase never ran)"
        assert allowed["pairs"] == [] and allowed["excluded"] == [{
            "rank": 4, "set": "wo", "decided": True, "reason": f"reference_missing:{cc}: {missing}; {d}: {missing}"}]
        assert not (tmp_path / sp.ALLOWED_PAIRS_FILE).exists() and (tmp_path / sp.VERDICT_FILE).is_file()
        # the pair's SERVING is certified whatever the reference said — the geometry IS folded, flagged `unverified` (no
        # overlay dir here: record_only), while the menu stays unwritten; the verdict and the fold say the same
        assert v["geometry_fold"]["status"] == "record_only" and v["geometry_fold"][sp.ADAPTER_PARITY_KEY] == "unverified"
        assert v[sp.ADAPTER_PARITY_KEY]["status"] == "unverified" and v["serving_certified_pairs"] == [[4, "wo"]]
        assert json.loads((tmp_path / sp.GEOMETRY_RECORD_FILE).read_text())[sp.ADAPTER_PARITY_KEY]["status"] == "unverified"
        assert v["reference_check_basis"] == sp.BASIS_VLLM_SIDE                   # the diagnostic label of what ran; never certifying
        # (b) the phase ran and RECORDED unavailability / an error: the reason rides the exclusion
        err = {"status": "error", "basis": None, "reason": "RuntimeError: peft is not importable in the serving venv"}
        results = _fixture_results(plan, outcomes={cc: ("passed", {"reference": err}), d: ("passed", {"reference": err}),
                                                   u: ("passed", {"reference": err})})
        v, allowed = sp.write_verdict(plan, results, probe_job_id="1", out=tmp_path / "b")
        assert v["status"] == "uncertified"
        why = "reference error — RuntimeError: peft is not importable in the serving venv"
        assert allowed["excluded"][0]["reason"] == f"reference_missing:{cc}: {why}; {d}: {why}"
        # (c) the CLI: exit 6, RESULT state=UNCERTIFIED naming the pair, no allowed_pairs.json; the memo banners it
        for lid, res in results.items():
            sp.write_json_atomic(sp.launch_result_path(tmp_path / "c", lid), res)
        sp.write_json_atomic(tmp_path / "c" / sp.PLAN_FILE, plan)
        rc = sp.main(["verdict", "--model-key", MODEL, "--out", str(tmp_path / "c"), "--probe-job-id", "5"])
        assert rc == sp.EXIT_UNCERTIFIED == 6 and rc != sp.EXIT_INCOMPLETE
        result = (tmp_path / "c" / sp.RESULT_FILE).read_text()
        assert result.startswith("state=UNCERTIFIED certifying=false reason=reference_missing") and "[4, 'wo']" in result
        assert not (tmp_path / "c" / sp.ALLOWED_PAIRS_FILE).exists()
        assert sp.main(["memo", "--model-key", MODEL, "--out", str(tmp_path / "c")]) == 0
        memo = (tmp_path / "c" / sp.MEMO_FILE).read_text()
        assert "UNCERTIFIED — NOT CERTIFYING" in memo and "PARTIAL" not in memo
        # (d) an excluded-on-evidence pair needs no reference to be decided (the reference is moot for it): the difference
        # adapter fails on ITS server
        results = _fixture_results(plan, outcomes={cc: ("passed", {"reference": None}),
                                                   d: ("passed", {"reference": None, "difference": False}),
                                                   u: ("passed", {"reference": None, "difference": False})})
        v, allowed = sp.write_verdict(plan, results, probe_job_id="1")
        assert v["status"] == "complete" and allowed["excluded"][0]["reason"].startswith("difference_failed:")
        # (e) outstanding launches take precedence: partial first, the reference question comes after the resume
        plan2 = _plan(sets=("wo",), ranks=(4,), windows=(4096, 49152), ladder_caps=())
        results = _fixture_results(plan2, outcomes={cc: ("passed", {"reference": None}), d: ("passed", {"reference": None}),
                                                    u: ("passed", {"reference": None})})
        v, _a = sp.write_verdict(plan2, results, probe_job_id="1")
        assert v["status"] == "partial" and v["unreferenced_pairs"] == []

    def test_the_geometry_fold_writes_the_site_overlay_the_resolvers_and_the_padded_preflight_read(self, tmp_path, monkeypatch):
        """End to end: a complete verdict folds the MEASURED widest executing
        module dim and the largest S x L an engine came up at into geometry_record.json AND <geometry-dir>/<model_key>.json
        with provenance; geometry.rows_for / budget_for / kernel_ceiling_adjustment then read that row BEFORE the built-in
        tables (the Llama-certified default 393,216 and the missing width), and padding_preflight.geometry_expectation sizes the
        model from it — while without the overlay it REFUSES the non-measured row by name. Fail-before: the probe wrote nothing
        the resolvers read; kernel_ceiling_adjustment refused the onboarded model after a successful probe."""
        from sisteer.capeval.serving import geometry as g
        sys.path.insert(0, str(REPO / "scripts/sdfbase/full22/static_padding"))
        import padding_preflight as pf
        plan = _plan(sets=("wo",), ranks=(4,), windows=(4096, 49152), ladder_caps=())
        cc, d = "wo_r4_S1cc_cap8", "wo_r4_S1d_cap8"
        outcomes = {f"{cc}_L4096": ("passed",), f"{d}_L4096": ("passed",),
                    f"{cc}_L49152": ("passed", {"engine_S": 9}), f"{d}_L49152": ("passed", {"engine_S": 9}),
                    # the pair's union (the served launch class) came up beside them at the same S: folded, never certifying a role
                    "wo_r4_single_cap8_L4096": ("passed",), "wo_r4_single_cap8_L49152": ("passed", {"engine_S": 9})}
        # every window's certifying launch, control then difference — the fold reads the measured D off each of them
        certifying = [f"{cc}_L4096", f"{d}_L4096", f"{cc}_L49152", f"{d}_L49152"]
        results = _fixture_results(plan, outcomes=outcomes)
        out, site = tmp_path / "rec", tmp_path / "site" / "geometry"
        sp.write_json_atomic(out / sp.PLAN_FILE, plan)
        for lid, res in results.items():
            sp.write_json_atomic(sp.launch_result_path(out, lid), res)
        # the SELECTED set comes from the models table's promoted pick (the one source) — a synthetic copy selecting wo
        table = tmp_path / "models.yaml"
        table.write_text(f"schema: 1\nmodels:\n  {MODEL}:\n    lora_architecture: {{rank: 4, modules: [o_proj], per_expert_rank: null, "
                         "selected_by: manual, selection_ruling_id: r1}\n")
        monkeypatch.setenv(pf.MODELS_TABLE_ENV, str(table))
        # BEFORE the fold: the built-in tables — no row of its own (the resolver hands back the Llama-certified default, a
        # non-measurement), no width, the preflight refuses by name
        assert g.budget_for(MODEL, "wo").source == g.DEFAULT_BUDGET_SOURCE and g.lora_max_module_dim_for(MODEL, "wo") is None
        with pytest.raises(g.UnknownModelKey, match="no pinned LoRA max module dim"):
            g.kernel_ceiling_adjustment(MODEL, 49152, 8, module_set="wo")
        with pytest.raises(ValueError, match="default_llama_certified.*not a measurement.*serving probe.*selected module set 'wo'"):
            pf.geometry_expectation(MODEL, 49152)
        # the CLI verdict with --geometry-dir: exit 0, RESULT names the fold, both files exist
        rc = sp.main(["verdict", "--model-key", MODEL, "--out", str(out), "--probe-job-id", "4242",
                      "--verdict-uri", "file://v", "--geometry-dir", str(site)])
        assert rc == 0, (out / sp.RESULT_FILE).read_text()
        result = (out / sp.RESULT_FILE).read_text()
        assert result.startswith("state=VERDICT pairs_allowed=1") and f"geometry=written:wo(D=64,budget={9 * 49152}),site={site / f'{MODEL}.json'}" in result
        assert (out / sp.ALLOWED_PAIRS_FILE).is_file() and (out / sp.GEOMETRY_RECORD_FILE).is_file() and (site / f"{MODEL}.json").is_file()
        record = json.loads((site / f"{MODEL}.json").read_text())
        assert record == json.loads((out / sp.GEOMETRY_RECORD_FILE).read_text())
        # ONE serializer: the record beside the verdict and the overlay copy are byte-identical, one trailing newline
        assert (out / sp.GEOMETRY_RECORD_FILE).read_bytes() == (site / f"{MODEL}.json").read_bytes() == g.site_geometry_bytes(record)
        assert record["version"] == g.SITE_GEOMETRY_VERSION == 2 and record["model_key"] == MODEL
        assert record["hf_id"] == "org/toy" and record["revision"] == REV and sorted(record["sets"]) == ["wo"]
        wo_rec = record["sets"]["wo"]
        assert wo_rec["lora_modules"] == ["o_proj"] and wo_rec["lora_max_module_dim"] == 64 and wo_rec["allowed_pairs"] == [[4, "wo"]]
        assert wo_rec["module_dim_launches"] == {lid: 64 for lid in certifying} and wo_rec["certified_by"] == certifying
        b = wo_rec["budget"]
        assert b["batched_token_budget"] == 9 * 49152 and b["certified"] is True and b["source"] == g.MEASURED_BUDGET_SOURCE
        assert b["basis"]["max_num_seqs"] == 9 and b["basis"]["max_model_len"] == 49152 and b["basis"]["probe_job_id"] == "4242"
        assert b["basis"]["kv_tokens_reported"] == 4096 and b["basis"]["method"].startswith(g.SITE_GEOMETRY_METHOD + ": module set 'wo'")
        assert record["summary"]["lora_max_module_dim_max"] == 64 and record["summary"]["sets"] == ["wo"]
        prov = record["provenance"]
        assert prov["probe_job_id"] == "4242" and prov["probe_record"] == str(out) and prov["verdict_uri"] == "file://v"
        assert prov["allowed_pairs"] == [[4, "wo"]] and prov["reference_check_basis"] == sp.BASIS_HF_PEFT
        # the attestation sha256: the launch record file, hashed (no serving_attestation.json under a fake launch)
        for lid in certifying:
            att = prov["attestation_sha256"][lid]
            assert att["file"] == str(sp.launch_result_path(out, lid))
            assert att["sha256"] == __import__("hashlib").sha256(Path(att["file"]).read_bytes()).hexdigest()
        # AFTER the fold, with the overlay configured: every resolver reads the SELECTED set's row before the built-in tables
        monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(site))
        rows = g.rows_for(MODEL, "wo")
        assert rows.site is not None and rows.site.path == str(site / f"{MODEL}.json") and rows.budget.certified is True
        rb = g.budget_for(MODEL, "wo")
        assert rb.source == g.MEASURED_BUDGET_SOURCE and rb.measured and rb.batched_token_budget == 9 * 49152 and rb.entry.certified
        assert g.lora_max_module_dim_for(MODEL, "wo") == 64
        assert g.kernel_ceiling_adjustment(MODEL, 49152, 9, module_set="wo") is None      # D=64: the ceiling admits 682 rows
        assert g.kernel_ceiling_adjustment(MODEL, 49152, 100_000, module_set="wo")["D"] == 64
        # no set named (an UNSTEERED caller): the record's conservative row — the one certified set's 9 x 49,152 — never the provisional
        # row beside a record; a named set still takes its own row
        rb0 = g.budget_for(MODEL)
        assert rb0.measured and rb0.batched_token_budget == 9 * 49152 and rb0.module_set == "wo" and rb0.module_set_source == g.MODULE_SET_SOURCE_UNSTEERED
        assert dict(g.LORA_MAX_MODULE_DIM_BY_MODEL) == {"gemma4e4b": 4096, "qwen35_4b": 4096, "nemotron3nano4b": 5120, "llama8b": 4096}   # untouched
        geo = pf.geometry_expectation(MODEL, 49152)
        assert geo["module_set"] == "wo" and geo["budget_source"] == "measured" and geo["budget_from_site_overlay"] is True and geo["budget_certified"] is True
        assert geo["lora_max_module_dim"] == 64 and geo["identity_S"] == 9 and geo["served_S"] == 9 and geo["batched_tokens"] == 9 * 49152
        # a launch record under a real record dir also folds through the CLI without --geometry-dir when the env names the dir
        monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(tmp_path / "site2"))
        rc = sp.main(["verdict", "--model-key", MODEL, "--out", str(out), "--probe-job-id", "4243"])
        assert rc == 0 and (tmp_path / "site2" / f"{MODEL}.json").is_file()
        # a certifying launch WITHOUT a measured stride: the menu still certifies, the fold REFUSES (never a guessed D), exit 6
        results_bad = _fixture_results(plan, outcomes={**outcomes, f"{cc}_L4096": ("passed", {"stride": False})})
        out2 = tmp_path / "rec2"
        sp.write_json_atomic(out2 / sp.PLAN_FILE, plan)
        for lid, res in results_bad.items():
            sp.write_json_atomic(sp.launch_result_path(out2, lid), res)
        rc = sp.main(["verdict", "--model-key", MODEL, "--out", str(out2), "--probe-job-id", "1", "--geometry-dir", str(tmp_path / "site3")])
        assert rc == sp.EXIT_UNCERTIFIED
        assert (out2 / sp.ALLOWED_PAIRS_FILE).is_file() and not (tmp_path / "site3").exists() and not (out2 / sp.GEOMETRY_RECORD_FILE).exists()
        assert f"geometry=refused:{cc}_L4096: no MEASURED executing stride" in (out2 / sp.RESULT_FILE).read_text()

    def test_a_broad_probe_a_narrower_pick_the_real_native_ceiling_and_both_pins_end_to_end(self, tmp_path, monkeypatch):
        """End to end on the worked example's Qwen3.5-4B dims: a probe that certifies wo
        (o_proj, D 4,096) AND qkvo (the packed 8192 + 2 x 1024 stride) folds PER SET; the HPO pick (the models table's promoted
        lora_architecture, the one source) selects the NARROWER wo; the padded preflight derives S from wo's record; the REAL
        native standup ceiling (native_standup.lora_kernel_ceiling_geometry over real safetensors headers of wo adapters, the
        function bring_up_native calls) attests exactly that S; BOTH padded launchers' pins accept the attested record — while the
        model-wide maximum (qkvo's D = q_out + 2·kv_out = 8192 + 2·1024 on the 4B's dims, the summary) would have demanded a smaller S and refused every steered padded unit
        after standup. A pick naming an uncertified set refuses by name; the REAL models
        table's promoted pick (the demo key's o_proj architecture) is followed."""
        import dataclasses as _dc
        import torch
        from safetensors.torch import save_file
        from sisteer.capeval.serving import geometry as g, native_standup as ns, serving_attestation as sa
        sys.path.insert(0, str(REPO / "scripts/sdfbase/full22/static_padding"))
        import padding_preflight as pf
        KEY, L = MODEL, 98304
        D_WO, D_QKVO = Q4.o_in, Q4.q_out + 2 * Q4.kv_out
        plan = _plan(dims=Q4, sets=("wo", "qkvo"), ranks=(4,), windows=(4096, L), ladder_caps=())
        # every planned launch passed; the wo engines (S1cc and S1d) at 98,304 came up at S 8 (above the wo ceiling's 5 rows, so
        # the kernel step binds for THIS design); the qkvo set's three engines at 98,304 at S 3 (one above the packed stride's own
        # ceiling of 2 there — the least admitted S of the record at that window, which the unsteered row below inherits)
        wo_at_L = {f"wo_r4_S1cc_cap8_L{L}", f"wo_r4_S1d_cap8_L{L}", f"wo_r4_single_cap8_L{L}"}            # the pair's union came up at the same S
        qkvo_at_L = {f"qkvo_r4_S1cc_cap8_L{L}", f"qkvo_r4_S1d_cap8_L{L}", f"qkvo_r4_S2_cap8_L{L}", f"qkvo_r4_single_cap8_L{L}"}
        outcomes = {r["id"]: ("passed", {"engine_S": 8} if r["id"] in wo_at_L else {"engine_S": 3} if r["id"] in qkvo_at_L else {})
                    for r in plan["launches"]}
        results = _fixture_results(plan, outcomes=outcomes, dims=Q4)
        out, site = tmp_path / "rec", tmp_path / "site"
        v, allowed = sp.write_verdict(plan, results, probe_job_id="4242", out=out, verdict_uri="file://v", geometry_dir=site)
        assert v["status"] == "complete" and sorted((p["set"] for p in allowed["pairs"])) == ["qkvo", "wo"]
        fold = v["geometry_fold"]
        assert fold["status"] == "written" and fold["sets"]["wo"]["lora_max_module_dim"] == D_WO and fold["sets"]["qkvo"]["lora_max_module_dim"] == D_QKVO
        assert fold["sets"]["wo"]["batched_token_budget"] == 8 * L and fold["sets"]["qkvo"]["batched_token_budget"] == 3 * L
        assert fold["summary"]["lora_max_module_dim_max"] == D_QKVO
        monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(site))
        # the pick selects wo (the one source: the models table)
        table = tmp_path / "models.yaml"
        table.write_text(f"schema: 1\nmodels:\n  {KEY}:\n    lora_architecture: {{rank: 4, modules: [o_proj], per_expert_rank: null, "
                         "selected_by: manual, selection_ruling_id: r1}\n")
        monkeypatch.setenv(pf.MODELS_TABLE_ENV, str(table))
        geo = pf.geometry_expectation(KEY, L, steered=True)                       # the STEERED preflight follows the pick
        assert geo["module_set"] == "wo" and geo["identity_S"] == 8 and geo["lora_max_module_dim"] == D_WO
        assert geo["kernel_adjustment"] == {"kind": g.ADJUSTMENT_KERNEL, "from_S": 8, "to_S": 5, "D": D_WO, "ceiling": (2**31 - 1) // D_WO}
        assert geo["served_S"] == 5
        # the model-wide maximum would have said S 2 — the number a round-2 fold pinned the arms to
        assert g.kernel_ceiling_adjustment(KEY, L, 8, module_set="qkvo")["to_S"] == 2
        # THE REAL NATIVE STANDUP CEILING over real safetensors headers of the wo class (the zero adapter + one steered adapter)
        def wo_adapter(d: Path, r: int) -> None:
            d.mkdir(parents=True)
            (d / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": r, "target_modules": ["o_proj"], "peft_type": "LORA"}))
            save_file({"base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight": torch.zeros(r, Q4.o_in),
                       "base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight": torch.zeros(Q4.o_in, r)},
                      str(d / "adapter_model.safetensors"))
        wo_adapter(tmp_path / "bank/none__lora0", 1); wo_adapter(tmp_path / "bank/si_x__lora_r4", 4)

        @_dc.dataclass(frozen=True)
        class Geom:
            max_model_len: int
            max_num_seqs: int
            seqs_source: str = "model_budget"

        blocks = {"none__lora0": {"dir": str(tmp_path / "bank/none__lora0")}, "si_x__lora_r4": {"dir": str(tmp_path / "bank/si_x__lora_r4")}}
        record, lowered = ns.lora_kernel_ceiling_geometry(blocks, Geom(L, geo["identity_S"]), eval_name=None, model_key=KEY,
                                                          allow_over_budget=False, override_reason=None)
        assert record["D"] == D_WO and record["ceiling"] == (2**31 - 1) // D_WO and lowered is not None and lowered["S"] == 5 == record["S"]
        assert lowered["reason"] == sa.LORA_KERNEL_CEILING_REASON and record["S"] == geo["served_S"]        # the standup attests what the preflight derived
        ep = {"engine": "lora_native", "id_name_map": {"1": "none__lora0", "2": "si_x__lora_r4"},
              "geometry": {"adjustments": [{"kind": lowered["reason"], "from_S": geo["identity_S"], "to_S": record["S"], "D": record["D"],
                                            "ceiling": record["ceiling"]}]}}
        ep_path = tmp_path / "endpoint.json"; ep_path.write_text(json.dumps(ep))

        def pin(name: str, extra, *, table_path, S=record["S"]):
            src = (REPO / "scripts/sdfbase/full22/static_padding" / name).read_text()
            i = src.index('"$MPY" - "$SERVE_DIR/static/endpoint.json" "$KEY" "$WINDOW" "$TARGET_S"')
            st = src.index("\n", src.index("<<'PY'", i)) + 1
            prog = src[st:src.index("\nPY\n", st)]
            env = {"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin", g.SITE_GEOMETRY_DIR_ENV: str(site)}
            if table_path is not None:
                env[pf.MODELS_TABLE_ENV] = str(table_path)
            r = subprocess.run([sys.executable, "-", str(ep_path), KEY, str(L), str(S)] + extra, input=prog, capture_output=True, text=True,
                               cwd=REPO, env=env)
            return r.returncode, r.stdout + r.stderr

        for name, extra in (("target_static_padded.sbatch", []), ("target_static_padded_obj.sbatch", [""])):
            steered = name.endswith("_obj.sbatch")
            if steered:
                rc, o = pin(name, extra, table_path=table)
                assert rc == 0 and "[geometry-pin] OK" in o and f"kernel ceiling 5 at D {D_WO} for module set 'wo'" in o, (name, o)
            else:
                # the UNSTEERED launcher: no pick drives it — the record's CONSERVATIVE row sizes the zero-adapter standup
                # (the qkvo set came up at only 3 x 49,152, so identity S 3; wo's kernel ceiling does not bind), and a standup that attested
                # the steered wo S 5 is refused
                rc, o = pin(name, extra, table_path=table, S=3)
                assert rc == 0 and "[geometry-pin] OK" in o and "unsteered standup: module set 'wo' (unsteered_default)" in o and "identity 3" in o, (name, o)
                rc, o = pin(name, extra, table_path=table)
                assert rc == 1 and "attested max_num_seqs 5 != the identity S 3" in o and "Traceback" not in o, (name, o)
            # a pick naming a set the probe did not certify, and no pick (the REAL table, pending): the STEERED pin refuses by name; the
            # UNSTEERED one reports the pick and sizes from the record regardless
            mlp = tmp_path / "mlp.yaml"
            mlp.write_text(f"schema: 1\nmodels:\n  {KEY}:\n    lora_architecture: {{rank: 4, modules: [down_proj, gate_proj, up_proj], "
                           "per_expert_rank: null, selected_by: manual, selection_ruling_id: r2}\n")
            rc, o = pin(name, extra, table_path=mlp, **({} if steered else {"S": 3}))
            if steered:
                assert rc == 1 and "selected module set 'mlp' has NO certified geometry record" in o and "Traceback" not in o, (name, o)
            else:
                assert rc == 0 and "[geometry-pin] OK" in o and "'module_set': 'mlp'" in o, (name, o)
            # the REAL models table: the demo key's promoted pick is the paper's o_proj architecture at rank 4 (reproduce/models.yaml) —
            # the STEERED pin follows it to the certified wo record; the UNSTEERED pin reports the pick and sizes from the record
            # regardless (the pick never drives an unsteered size)
            rc, o = pin(name, extra, table_path=None, **({} if steered else {"S": 3}))
            if steered:
                assert rc == 0 and "[geometry-pin] OK" in o and f"kernel ceiling 5 at D {D_WO} for module set 'wo'" in o, (name, o)
            else:
                assert rc == 0 and "[geometry-pin] OK" in o and "promoted pick {'module_set': 'wo', 'rank': 4" in o and "promoted pick None" not in o, (name, o)
            # the counterfactual the CLOSE named: holding this wo standup to the model-wide (qkvo) width would refuse it
            qkvo = tmp_path / "qkvo.yaml"
            qkvo.write_text(f"schema: 1\nmodels:\n  {KEY}:\n    lora_architecture: {{rank: 4, modules: [k_proj, o_proj, q_proj, v_proj], "
                            "per_expert_rank: null, selected_by: manual, selection_ruling_id: r3}\n")
            rc, o = pin(name, extra, table_path=qkvo)
            if steered:
                # qkvo's row: identity 3 lowered to 2 by its own packed-stride ceiling — the attested wo S 5 is another class's number
                assert rc == 1 and "attested max_num_seqs 5 != 2" in o and f"certifies D={D_QKVO}" in o and "another class" in o, (name, o)
            else:
                # the UNSTEERED launcher reports the qkvo pick but keeps the zero adapter's class: the steered S 5 is refused in the identity form
                assert rc == 1 and "attested max_num_seqs 5 != the identity S 3" in o and "'module_set': 'qkvo'" in o and "Traceback" not in o, (name, o)

    def test_memo_renders_from_the_verdict(self, tmp_path):
        plan = _plan(sets=("wo", "mlp"), ranks=(4,), windows=(4096,))
        s2 = [r["id"] for r in plan["launches"] if r["topology"] == "S2"][0]
        # each pair's union beside its servers: wo's passed, mlp's died with its control server (the same adapters, a larger engine)
        results = _fixture_results(plan, outcomes={"wo_r4_S1cc_cap8_L4096": ("passed",), "wo_r4_S1d_cap8_L4096": ("passed",),
                                                   "wo_r4_single_cap8_L4096": ("passed",), "mlp_r4_single_cap8_L4096": ("oom",),
                                                   "mlp_r4_S1cc_cap8_L4096": ("oom",), s2: ("passed",)})
        verdict, _a = sp.write_verdict(plan, results, probe_job_id="9", out=tmp_path)
        text = sp.render_memo(verdict, {"verdict": "fits", "s_per_step": 1.4})
        for h in ("## 1. Native support", "## 2. The buffer law", "## 3. Per-server manifests", "## 4. KV and stride",
                  "## 5. Tensor parallelism", "## 6. Verdict", "## 7. Fit throughput", "## 8. HPO cost"):
            assert h in text
        assert "oom_at_R=8" in text and "| 4 | wo | S1ccS1d | 8 |" in text and "s_per_step" in text
        mi = verdict["memo_inputs"]
        # the dense law per class, per slot per unit R: o_proj (64 + 64) x 2 B over one block; the mlp class's gate_up (two slices) +
        # down over the toy's two blocks
        assert mi["buffer_law"]["bytes_per_R_per_slot_by_class"] == {"o_proj": 1 * (64 + 64) * 2,
                                                                     "down_proj,gate_proj,up_proj": 2 * ((2 * 64 + 2 * 32) + (32 + 64)) * 2}
        assert "`o_proj` 256 B" in text and "standup_manifest.expected_lora_bytes" in text
        # a dense plan: no expert law, no class wraps the experts (the column still renders)
        assert mi["buffer_law"]["expert_bytes_per_R_per_slot"] is None and "the routed-expert family alone" not in text
        assert all(c["wraps_experts"] is False for c in mi["native_support_per_class"].values()) and "| wraps experts |" in text
        assert mi["tp_needed"] == {"target": False, "attacker_27b": False, "reason": mi["tp_needed"]["reason"]}
        rc = sp.main(["memo", "--model-key", MODEL, "--out", str(tmp_path)])
        assert rc == 0 and (tmp_path / sp.MEMO_FILE).read_text().startswith("# Scaling memo inputs")

    def test_memo_carries_the_expert_law_on_a_moe_plan(self, tmp_path):
        """The MoE option's memo inputs beside the dense ones: the routed-expert family's per-slot bytes per unit of
        ``moe_lora_rank`` from the dims (E x (d + 2I + I + d) x 2 B x MoE blocks), each class's ``wraps_experts``, every ladder
        point's ``moe_lora_rank``, and the memo text names them."""
        plan = _plan(sets=("wo", "mlp"), ranks=(4,), windows=(4096,), dims=MOE_TOY)
        s2 = [r["id"] for r in plan["launches"] if r["topology"] == "S2"][0]
        results = _fixture_results(plan, outcomes=_with_union(plan, {"wo_r4_S1cc_cap8_L4096": ("passed",), "wo_r4_S1d_cap8_L4096": ("passed",),
                                                                     "mlp_r4_S1cc_cap8_L4096": ("passed",), "mlp_r4_S1d_cap8_L4096": ("passed",),
                                                                     s2: ("passed",)}),
                                   dims=MOE_TOY)
        verdict, allowed = sp.write_verdict(plan, results, probe_job_id="9", out=tmp_path)
        mi = verdict["memo_inputs"]
        assert mi["buffer_law"]["expert_bytes_per_R_per_slot"] == 2 * 4 * (64 + 64 + 32 + 64) * 2 == 3_584
        assert abs(mi["buffer_law"]["expert_mib_per_R_per_slot"] - 3_584 / 2**20) < 1e-12
        assert mi["native_support_per_class"]["down_proj,experts,gate_proj,up_proj"]["wraps_experts"] is True
        assert mi["native_support_per_class"]["o_proj"]["wraps_experts"] is False
        assert "routed-expert family" in mi["buffer_law"]["law"] and all("moe_lora_rank" in pt for pt in mi["buffer_law"]["measured_points"])
        text = sp.render_memo(verdict)
        assert "| `down_proj,experts,gate_proj,up_proj` | True |" in text and "the routed-expert family alone 3584 B" in text
        mlp = next(p for p in allowed["pairs"] if p["set"] == "mlp")
        assert mlp["moe_lora_rank_by_server"] == {"S1cc": 4, "S1d": 4} and mlp["per_expert_rank"] == {"shared_expert": 4, "routed_expert": 2}


# --------------------------------------------------------------------------- run loop (fakes)


class TestRunLoopWithFakes:
    """The per-launch driver over a fake engine: adapters missing -> refused;
    a fake stand-up raising an OOM -> oom with the log tail; a clean fake ->
    passed with the checks, resumable (a second run re-launches nothing)."""

    def _deps(self, tmp_path, *, fail=None, plan=None, ref_mode="pass"):
        calls = {"stand_up": 0, "reference": 0}

        def reference(**kw):
            # the HF+PEFT phase over the pending adapters: a PASS per adapter dir (the certifying evidence), or the two
            # non-certifying failure shapes — a crash, and a recorded unavailability
            calls["reference"] += 1
            if ref_mode == "crash":
                raise RuntimeError("the base model failed to load")
            if ref_mode == "unavailable":
                return {"status": "unavailable", "reason": "peft not importable", "per_adapter": {}}
            return {"status": "done", "peft_version": "0.20.0", "layouts": {},
                    "per_adapter": {item["dir"]: {"status": "pass"} for item in kw["pending"]}}

        def worker_report(launch, d):
            # a real lora-native report carries the stacked-buffer bytes and every wrapper's stride;
            # the fake reports the plan's own expectation for the launch under `d` (its dir is the launch id)
            rep = {"max_num_seqs": 4, "kv_num_gpu_blocks": 10, "kv_block_size": 16,
                   "lora_execution_module_strides": {"L0.self_attn.o_proj": {"in": 64, "out": 64}}}
            if plan is not None:
                rec = next(r for r in plan["launches"] if r["id"] == Path(d).name)
                rep["lora_execution_module_lora_buffer_bytes"] = int(rec["expected_lora_bytes"]["total"])
                # the attested class (worker_reports: lora_max_lora_rank / lora_max_loras) — the launch's rank cap and its manifest's
                # adapter count, as a real lora-native engine reports them; the fold refuses a passed launch without them (53bd76ed0)
                rep["lora_max_lora_rank"] = int(rec["cap"])
                rep["lora_max_loras"] = int(plan["manifests"][rec["manifest"]]["max_loras"])
            return rep

        class L:
            model = "org/toy"
            base_url = "http://127.0.0.1:1"
            auth_headers = {}
            cmd = ("vllm", "serve")
            max_num_seqs = 4

        def stand_up(**kw):
            calls["stand_up"] += 1
            (Path(kw["out_dir"]) / "vllm_server.log").write_text(
                (ENGINE_LOG if fail is None else "torch.OutOfMemoryError: CUDA out of memory\n"))
            if fail is not None:
                raise RuntimeError(fail)
            return L(), object()

        def chat(launch, body):
            genuine = body["model"] not in ("org/toy", "none__lora0") and "carrier" not in body["model"]
            p = _probe(first_lp=-0.3 if genuine else -0.1)
            return {"choices": [{"token_ids": p["token_ids"], "finish_reason": "length", "message": {"content": "x"},
                                 "logprobs": {"content": [{"token": r["token"], "logprob": r["logprob"],
                                                           "top_logprobs": [{"token": t, "logprob": lp} for t, lp in r["top"]]}
                                                          for r in p["rows"]]}}]}

        deps = sp.Deps(stand_up=stand_up, shut_down=lambda l, p: None, chat=chat,
                       list_models=lambda l: ["org/toy", "control__lora", "constitution__lora", "difference__lora",
                                              "none__lora0"] + [f"{r}__carrier" for r in sm.CARRIER_ROLES],
                       worker_report=worker_report,
                       nvidia_smi=lambda devices=None: {"used_mib": 1},   # devices=: the pinned subset, None here
                       throughput=lambda l, **kw: [{"requested_concurrency": 32}],
                       read_log=sp._read_log_tail, wait_gpu_free=lambda devices=None: {"drained": True},
                       reference=reference)
        return deps, calls

    def test_missing_adapters_refuse_then_a_clean_run_passes_and_resumes(self, tmp_path):
        plan = _plan(sets=("wo",), ranks=(4,), windows=(4096,), ladder_caps=())
        # two standups per view since the ruling, plus the pair's union (the served launch class, 2026-10-11)
        assert [r["topology"] for r in plan["launches"]] == ["S1cc", "S1d", "single"]
        rec = plan["launches"][0]
        deps, calls = self._deps(tmp_path, plan=plan)
        res = sp.run_one_launch(plan, rec, tmp_path, deps, gpu_mem_util=0.9, ready_timeout_s=1, throughput=True)
        assert res["status"] == "refused" and res["failure"]["message"].startswith("refused:adapters not exported")
        sp.build_adapter_dirs(plan, tmp_path, FakeExporter())
        results = sp.run_launches(plan, tmp_path, deps, gpu_mem_util=0.9, ready_timeout_s=1, time_budget_s=None,
                                  max_launches=None, throughput=True)
        assert all(results[r["id"]]["status"] == "passed" for r in plan["launches"]) and calls["stand_up"] == 3
        assert results[rec["id"]]["throughput"] == [{"requested_concurrency": 32}]
        # the driver measures throughput on EVERY control-carrying natural launch at a throughput window: the union's too (the served
        # class's pace is what the serving rows price from — round 2 on the served-class work), never the difference server's
        assert results["wo_r4_single_cap8_L4096"]["throughput"] == [{"requested_concurrency": 32}]
        assert "throughput" not in results["wo_r4_S1d_cap8_L4096"] and "throughput_error" not in results["wo_r4_S1d_cap8_L4096"]
        # every passed launch attests the class its manifest intended (the fake reports the plan's cap and slot count)
        assert all(results[r["id"]]["checks"]["launch_class"]["pass"] is True for r in plan["launches"])
        assert all((tmp_path / sp.LAUNCHES_DIR / f"{r['id']}.json").is_file() for r in plan["launches"])
        # resume: nothing re-launches
        sp.run_launches(plan, tmp_path, deps, gpu_mem_util=0.9, ready_timeout_s=1, time_budget_s=None,
                        max_launches=None, throughput=True)
        assert calls["stand_up"] == 3
        # the reference phase judges every genuine adapter on the HF+PEFT basis and writes it into EVERY passed launch's
        # result (both servers certify the pair, so both carry the evidence the verdict reads)
        sp.reference_phase(plan, tmp_path, deps)
        for r in plan["launches"]:
            res = sp.load_results(tmp_path)[r["id"]]
            assert res["reference_check"]["basis"] == sp.BASIS_HF_PEFT and res["reference_check"]["status"] == "pass"
            assert set(res["reference_check"]["per_adapter"]) == {a.name for a in sm.Manifest.from_dict(plan["manifests"][r["manifest"]]).adapters if a.kind == sm.KIND_GENUINE}
        # the launch-list checkpoint says complete
        prog = sp.load_progress(tmp_path)
        assert prog["complete"] is True and prog["outstanding"] == [] and sorted(prog["done"]) == sorted(r["id"] for r in plan["launches"])

    def test_a_report_without_the_measurements_fails_the_launch(self, tmp_path):
        """End to end: the same clean fake engine, but a worker report lacking the buffer bytes -> failed."""
        plan = _plan(sets=("wo",), ranks=(4,), windows=(4096,), ladder_caps=())
        sp.build_adapter_dirs(plan, tmp_path, FakeExporter())
        deps, _c = self._deps(tmp_path)          # plan=None: no buffer bytes in the report
        res = sp.run_one_launch(plan, plan["launches"][0], tmp_path, deps, gpu_mem_util=0.9, ready_timeout_s=1,
                                throughput=False)
        assert res["status"] == "failed" and res["checks"]["failed"] == ["buffer_bytes"]

    def test_a_time_budget_stop_is_incomplete_checkpoints_and_resume_finishes_it(self, tmp_path, monkeypatch):
        """The time budget stops the run with launches outstanding ->
        `probe_progress.json` names them, `run` exits EXIT_INCOMPLETE, `write_verdict` is PARTIAL and writes no
        allowed_pairs.json, `verdict` refuses (exit 4, RESULT state=PARTIAL); a bare `run` over the partial record
        refuses; `run --resume` continues exactly the outstanding launch and then `verdict` certifies."""
        plan = _plan(sets=("wo",), ranks=(4,), windows=(4096, 49152), ladder_caps=())
        assert [r["kind"] for r in plan["launches"]] == ["natural"] * 6            # S1cc + S1d + the union at each of the two windows
        sp.write_json_atomic(tmp_path / sp.PLAN_FILE, plan)
        sp.build_adapter_dirs(plan, tmp_path, FakeExporter())
        deps, calls = self._deps(tmp_path, plan=plan)
        clock = {"t": 0.0}

        def now():
            # every read of the clock is 1,000 s later; the budget is checked BEFORE each launch: the first check
            # sees 1,000 s (under the 1,500 s budget) and the launch runs (three more reads), the second sees 5,000 s
            clock["t"] += 1000.0
            return clock["t"]

        slow = dataclasses.replace(deps, now=now)
        results = sp.run_launches(plan, tmp_path, slow, gpu_mem_util=0.9, ready_timeout_s=1, time_budget_s=1500.0,
                                  max_launches=None, throughput=False)
        first, rest = plan["launches"][0]["id"], [r["id"] for r in plan["launches"][1:]]
        assert calls["stand_up"] == 1 and set(results) == {first} and results[first]["status"] == "passed"
        prog = sp.load_progress(tmp_path)
        assert prog["complete"] is False and prog["stopped_reason"] == "time_budget"
        assert [o["id"] for o in prog["outstanding"]] == rest and prog["decision_outstanding"] == rest
        # the verdict: PARTIAL, non-certifying, the pair undecided, NO allowed_pairs.json
        verdict, allowed = sp.write_verdict(plan, sp.load_results(tmp_path), probe_job_id="7", out=tmp_path)
        assert verdict["status"] == "partial" and verdict["certifying"] is False
        assert verdict["undecided_pairs"] == [[4, "wo"]]
        assert [o["id"] for o in verdict["outstanding"] if o["decision_relevant"]] == rest
        assert allowed["pairs"] == [] and allowed["excluded"] == [{"rank": 4, "set": "wo", "reason": "not_run", "decided": False}]
        assert (tmp_path / sp.VERDICT_FILE).is_file() and not (tmp_path / sp.ALLOWED_PAIRS_FILE).exists()
        # the CLI refuses to certify, says so in RESULT, and never writes P
        rc = sp.main(["verdict", "--model-key", MODEL, "--out", str(tmp_path), "--probe-job-id", "7"])
        assert rc == sp.EXIT_INCOMPLETE == 4
        result = (tmp_path / sp.RESULT_FILE).read_text()
        assert result.startswith("state=PARTIAL certifying=false") and all(lid in result for lid in rest)
        assert not (tmp_path / sp.ALLOWED_PAIRS_FILE).exists()
        # the memo still renders — flagged PARTIAL, NOT CERTIFYING
        assert sp.main(["memo", "--model-key", MODEL, "--out", str(tmp_path)]) == 0
        assert "PARTIAL — NOT CERTIFYING" in (tmp_path / sp.MEMO_FILE).read_text()
        # `run` over the partial record: refused without --resume; --resume with disagreeing plan args refused
        monkeypatch.setattr(sp, "ProductionExporter", lambda *a, **k: FakeExporter())
        monkeypatch.setattr(sp.Deps, "production", classmethod(lambda cls: deps))
        base = ["run", "--model-key", MODEL, "--out", str(tmp_path), "--no-throughput"]
        with pytest.raises(SystemExit, match="INCOMPLETE record .* pass --resume"):
            sp.main(base)
        with pytest.raises(SystemExit, match="disagree with the existing plan"):
            sp.main(base + ["--resume", "--plan-args", "--sets mlp --ranks 4"])
        with pytest.raises(SystemExit, match="nothing to resume"):
            sp.main(["run", "--model-key", MODEL, "--out", str(tmp_path / "empty"), "--resume"])
        assert calls["stand_up"] == 1
        # --resume runs exactly the outstanding launches, exits 0, and the verdict now certifies the pair
        rc = sp.main(base + ["--resume", "--plan-args", "--sets wo --ranks 4 --tp 1"])
        assert rc == 0 and calls["stand_up"] == 6
        prog = sp.load_progress(tmp_path)
        assert prog["complete"] is True and prog["outstanding"] == [] and sorted(prog["done"]) == sorted([first] + rest)
        rc = sp.main(["verdict", "--model-key", MODEL, "--out", str(tmp_path), "--probe-job-id", "8"])
        assert rc == 0 and (tmp_path / sp.ALLOWED_PAIRS_FILE).is_file()
        allowed = json.loads((tmp_path / sp.ALLOWED_PAIRS_FILE).read_text())
        assert allowed["status"] == "complete" and [(p["rank"], p["set"]) for p in allowed["pairs"]] == [(4, "wo")]
        assert allowed["pairs"][0]["reference_check_basis"] == sp.BASIS_HF_PEFT and calls["reference"] >= 1
        assert (tmp_path / sp.RESULT_FILE).read_text().startswith("state=VERDICT pairs_allowed=1")
        # an incomplete run through the CLI returns EXIT_INCOMPLETE (a fresh record, budget spent at once)
        out2 = tmp_path / "second"
        plan2 = _plan(sets=("wo",), ranks=(4,), windows=(4096,), ladder_caps=())
        sp.write_json_atomic(out2 / sp.PLAN_FILE, plan2)
        sp.build_adapter_dirs(plan2, out2, FakeExporter())
        monkeypatch.setattr(sp.Deps, "production", classmethod(lambda cls: dataclasses.replace(deps, now=now)))
        rc = sp.main(["run", "--model-key", MODEL, "--out", str(out2), "--no-throughput", "--time-budget-s", "0"])
        assert rc == sp.EXIT_INCOMPLETE and sp.load_progress(out2)["complete"] is False
        # `run --no-reference` completes its launches (exit 0) but the verdict then REFUSES certification: exit 6, never 4
        # (nothing to resume), RESULT state=UNCERTIFIED, no allowed_pairs.json
        out3 = tmp_path / "third"
        sp.write_json_atomic(out3 / sp.PLAN_FILE, plan2)
        sp.build_adapter_dirs(plan2, out3, FakeExporter())
        monkeypatch.setattr(sp.Deps, "production", classmethod(lambda cls: deps))
        rc = sp.main(["run", "--model-key", MODEL, "--out", str(out3), "--no-throughput", "--no-reference"])
        assert rc == 0 and sp.load_progress(out3)["complete"] is True
        assert "reference_check" not in sp.load_results(out3)[plan2["launches"][0]["id"]]
        rc = sp.main(["verdict", "--model-key", MODEL, "--out", str(out3), "--probe-job-id", "9"])
        assert rc == sp.EXIT_UNCERTIFIED and not (out3 / sp.ALLOWED_PAIRS_FILE).exists()
        assert (out3 / sp.RESULT_FILE).read_text().startswith("state=UNCERTIFIED certifying=false reason=reference_missing")

    def test_a_reference_phase_crash_is_recorded_and_the_verdict_refuses_by_name(self, tmp_path, monkeypatch):
        """The HF+PEFT phase raising (the base model failed to load, peft missing) no longer
        leaves the pair certifiable on the vLLM-side checks — reference_check.json records status=error with the
        exception, every judged launch carries `reference_check.status=error, basis=None`, and the verdict is
        `uncertified` naming the error. A recorded `unavailable` is the same refusal. And the production phase itself
        RAISES when peft does not import (never a silent `unavailable`)."""
        plan = _plan(sets=("wo",), ranks=(4,), windows=(4096,), ladder_caps=())
        ids = [r["id"] for r in plan["launches"]]          # S1cc, S1d, the union: every launch carries the error
        certifying = ids[:2]                               # S1cc, S1d certify the pair's roles: the refusal names them, never the union
        rid = ids[0]
        sp.build_adapter_dirs(plan, tmp_path, FakeExporter())
        deps, calls = self._deps(tmp_path, plan=plan, ref_mode="crash")
        sp.run_launches(plan, tmp_path, deps, gpu_mem_util=0.9, ready_timeout_s=1, time_budget_s=None, max_launches=None,
                        throughput=False)
        rec = sp.reference_phase(plan, tmp_path, deps)
        assert rec["status"] == "error" and "RuntimeError: the base model failed to load" in rec["reason"]
        assert json.loads((tmp_path / "reference_check.json").read_text())["status"] == "error"
        res = sp.load_results(tmp_path)[rid]
        assert res["reference_check"] == {"status": "error", "reason": rec["reason"], "basis": None}
        v, allowed = sp.write_verdict(plan, sp.load_results(tmp_path), probe_job_id="1", out=tmp_path)
        assert v["status"] == "uncertified" and allowed["pairs"] == [] and not (tmp_path / sp.ALLOWED_PAIRS_FILE).exists()
        assert allowed["excluded"][0]["reason"] == "reference_missing:" + "; ".join(f"{lid}: reference error — {rec['reason']}" for lid in certifying)
        assert sp.load_results(tmp_path)[ids[2]]["reference_check"]["status"] == "error"          # the union carries it too, unnamed
        # a recorded unavailability: the same refusal, the reason carried
        out2 = tmp_path / "u"
        sp.build_adapter_dirs(plan, out2, FakeExporter())
        deps_u, _c = self._deps(out2, plan=plan, ref_mode="unavailable")
        sp.run_launches(plan, out2, deps_u, gpu_mem_util=0.9, ready_timeout_s=1, time_budget_s=None, max_launches=None,
                        throughput=False)
        sp.reference_phase(plan, out2, deps_u)
        assert sp.load_results(out2)[rid]["reference_check"] == {"status": "unavailable", "reason": "peft not importable", "basis": None}
        v, allowed = sp.write_verdict(plan, sp.load_results(out2), probe_job_id="1")
        assert v["status"] == "uncertified" and "reference unavailable — peft not importable" in allowed["excluded"][0]["reason"]
        # the PRODUCTION phase with peft unimportable: a RuntimeError naming peft and the requirements, before any model load
        monkeypatch.setitem(sys.modules, "peft", None)
        with pytest.raises(RuntimeError, match="peft is not importable.*peft==0.20.0"):
            sp._production_reference_phase(model_id="org/toy", revision=REV, pending=[], out=tmp_path)

    def test_a_decided_pair_does_not_hold_the_verdict_and_ladder_launches_never_do(self, tmp_path):
        """The `decision-relevant` rule: a pair excluded on EVIDENCE (an OOM that ran) is decided even with its
        other windows unrun; a ladder launch outstanding never blocks; an estimate-only exclusion (unrun S<1) does."""
        plan = _plan(sets=("wo", "mlp"), ranks=(4, 16), windows=(4096, 49152))
        s2 = [r["id"] for r in plan["launches"] if r["topology"] == "S2"]
        outcomes = {"wo_r4_S1cc_cap8_L4096": ("passed",), "wo_r4_S1d_cap8_L4096": ("passed",),
                    "wo_r4_S1cc_cap8_L49152": ("passed",), "wo_r4_S1d_cap8_L49152": ("passed",),
                    "wo_r16_S1cc_cap16_L4096": ("passed",), "wo_r16_S1cc_cap16_L49152": ("passed",),
                    "wo_r16_S1d_cap32_L4096": ("passed", {"difference": False}), "wo_r16_S1d_cap32_L49152": ("passed",),
                    "mlp_r4_S1cc_cap8_L4096": ("passed",), "mlp_r4_S1d_cap8_L4096": ("passed",),
                    "mlp_r4_S1cc_cap8_L49152": ("passed",), "mlp_r4_S1d_cap8_L49152": ("passed",),
                    "mlp_r16_S1cc_cap16_L4096": ("passed",), "mlp_r16_S1d_cap32_L4096": ("oom",), **{i: ("passed",) for i in s2},
                    # each pair's union (the same adapters in one engine): wo r16's difference fails on it too, mlp r16's dies with
                    # its difference server, and mlp r16's at 49,152 stays unrun like its servers
                    "wo_r16_single_cap32_L4096": ("passed", {"difference": False}), "mlp_r16_single_cap32_L4096": ("oom",)}
        outcomes = _with_union(plan, outcomes)
        del outcomes["mlp_r16_single_cap32_L49152"]
        results = _fixture_results(plan, outcomes=outcomes)
        assert any(r["kind"] == "ladder" for r in sp.outstanding_launches(plan, results))
        v, allowed = sp.write_verdict(plan, results, probe_job_id="1", out=tmp_path)
        assert v["status"] == "complete" and (tmp_path / sp.ALLOWED_PAIRS_FILE).is_file()
        assert all(o["decision_relevant"] is False for o in v["outstanding"])
        assert {(e["rank"], e["set"]): e["decided"] for e in allowed["excluded"]} == {(16, "wo"): True, (16, "mlp"): True}
        # an allowed pair with a production window UNRUN (here the difference server's — and the union that carries the same
        # difference adapter, which would otherwise certify the role) is undecided -> partial
        del results["mlp_r4_S1d_cap8_L49152"]
        del results["mlp_r4_single_cap8_L49152"]
        v, allowed = sp.write_verdict(plan, results, probe_job_id="1", out=tmp_path / "p")
        assert v["status"] == "partial" and v["undecided_pairs"] == [[4, "mlp"]]
        assert not (tmp_path / "p" / sp.ALLOWED_PAIRS_FILE).exists()
        assert next(e for e in allowed["excluded"] if e["set"] == "mlp" and e["rank"] == 4) == {
            "rank": 4, "set": "mlp", "reason": "not_run", "decided": False}
        # an ESTIMATED S<1 at an unrun window is not evidence: undecided until that launch runs
        plan2 = json.loads(json.dumps(plan))
        # the pair's three engines at 49,152 (the union shares the servers' class, estimate and measurement)
        wo4_at_49152 = ("wo_r4_S1cc_cap8_L49152", "wo_r4_S1d_cap8_L49152", "wo_r4_single_cap8_L49152")
        for rec in plan2["launches"]:
            if rec["id"] in wo4_at_49152:
                rec["admitted_S"] = 0
        results2 = _fixture_results(plan2, outcomes={k: v_ for k, v_ in outcomes.items() if k not in wo4_at_49152})
        v, allowed = sp.write_verdict(plan2, results2, probe_job_id="1")
        e = next(e for e in allowed["excluded"] if (e["rank"], e["set"]) == (4, "wo"))
        assert e["reason"] == "ceiling_S<1_at_L=49152" and e["decided"] is False and v["status"] == "partial"
        # a MEASURED engine S<1 at that window would decide it
        results3 = _fixture_results(plan2, outcomes={**outcomes, **{lid: ("passed", {"engine_S": 0}) for lid in wo4_at_49152}})
        v, allowed = sp.write_verdict(plan2, results3, probe_job_id="1")
        e = next(e for e in allowed["excluded"] if (e["rank"], e["set"]) == (4, "wo"))
        assert e["reason"] == "ceiling_S<1_at_L=49152" and e["decided"] is True
        # ...and the served class says so for ITSELF: its launch came up admitting nothing, so the class is not certified there
        # (round 2 on the served-class work: a certified class is one a standup can be sized from) — a decided serving exclusion
        ex = next(x for x in v["excluded"] if (x["rank"], x["set"]) == (4, "wo"))
        assert f"{sp.REASON_SERVED_CLASS}_at_L=49152:wo_r4_single_cap8_L49152:ceiling_S<1" in ex["reasons"]
        assert ex["served_class"]["certified_by_window"]["49152"] is None and ex["served_class"]["measured_S_by_window"]["49152"] is None
        assert ex["served_class"]["certified_by_window"]["4096"] == "wo_r4_single_cap8_L4096"
        assert ex["served_class"]["measured_S_by_window"]["4096"] == results3["wo_r4_single_cap8_L4096"]["engine"]["max_num_seqs_engine"] >= 1

    def test_an_oom_at_stand_up_is_classified_from_the_log(self, tmp_path):
        plan = _plan(sets=("wo",), ranks=(4,), windows=(4096,), ladder_caps=())
        sp.build_adapter_dirs(plan, tmp_path, FakeExporter())
        deps, _c = self._deps(tmp_path, fail="engine died")
        res = sp.run_one_launch(plan, plan["launches"][0], tmp_path, deps, gpu_mem_util=0.9, ready_timeout_s=1,
                                throughput=False)
        assert res["status"] == "oom" and "CUDA out of memory" in res["failure"]["log_tail"]
        assert res["launch_S"] == plan["launches"][0]["admitted_S"]


# --------------------------------------------------------------------------- the reference prompt


def test_reference_prompt_is_the_pinned_thinking_off_render_the_server_applies():
    """The HF/PEFT reference renders PROBE_PROMPT through the fit side's
    `apply_chat_template_folding` under the tokenizer's pinned kwargs, i.e. the SAME thinking-off header the native
    server pins with `--default-chat-template-kwargs` (vllm_server.THINKING_OFF_TEMPLATE_KWARGS); a bare
    `apply_chat_template` (the old code) rendered Qwen's thinking-ON default and failed parity for correct adapters."""
    from sisteer.capeval.serving import vllm_server as vs

    class Tok:
        chat_template = ("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}"
                         "<|assistant|>{% if not enable_thinking %}<think></think>{% endif %}")

        def __init__(self, pinned):
            if pinned:
                self._sisteer_template_kwargs = {"enable_thinking": False}

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
            body = "".join(f"<|{m['role']}|>{m['content']}" for m in messages) + ("<|assistant|>" if add_generation_prompt else "")
            return body + ("<think></think>" if kw.get("enable_thinking") is False else "")

    tok = Tok(pinned=True)
    served = tok.apply_chat_template(list(sp.PROBE_PROMPT), tokenize=False, add_generation_prompt=True,
                                     **json.loads(vs.THINKING_OFF_TEMPLATE_KWARGS))
    bare = tok.apply_chat_template(list(sp.PROBE_PROMPT), tokenize=False, add_generation_prompt=True)
    assert served.endswith("<|assistant|><think></think>") and bare.endswith("<|assistant|>") and served != bare
    assert sp.reference_prompt_text(tok) == served
    # an unpinned tokenizer on a thinking-capable template refuses (the two renders would differ)
    with pytest.raises(RuntimeError, match="pins no thinking-off render"):
        sp.reference_prompt_text(Tok(pinned=False))

    class Plain(Tok):
        chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"

    plain = Plain(pinned=False)
    assert sp.reference_prompt_text(plain) == plain.apply_chat_template(list(sp.PROBE_PROMPT), tokenize=False,
                                                                        add_generation_prompt=True)
    # and the production phase renders through it (source pin: never a bare apply_chat_template on the reference path)
    src = Path(sp.__file__).read_text()
    body = src[src.index("def _production_reference_phase"):src.index("def classify_failure")]
    assert "reference_prompt_text(tok)" in body and "tok.apply_chat_template(" not in body


# --------------------------------------------------------------------------- the sbatch carrier (findings 3, 5)


SBATCH = REPO / "scripts/sbatch/serving_probe.sbatch"


def _prelude(text: str) -> str:
    """The carrier's positional + shared-root + HF block (up to the offline pin), runnable without Slurm."""
    start = text.index('MODEL_KEY="${1:?')
    end = text.index("export HF_HUB_OFFLINE=1")
    return text[start:end]


def _run_prelude(tmp_path, env, *args):
    script = _prelude(SBATCH.read_text()) + '\necho "SHARED=$SHARED HF_HOME=${HF_HOME:-unset} ATTEMPT=$ATTEMPT MAX=$MAX_ATTEMPTS"\n'
    e = {"PATH": os.environ["PATH"], "USER": "nobody-testhost", "HOME": str(tmp_path)}
    e.update(env)
    return subprocess.run(["bash", "-c", script, "prelude", *args], capture_output=True, text=True, env=e, timeout=60)


class TestSbatchCarrier:
    ARGS = (MODEL, "org/toy", "0" * 40)

    def test_shared_root_and_hf_cache_come_from_the_site_env_first(self, tmp_path):
        """SISTEER_SHARED_ROOT is honoured like every other carrier; SISTEER_HF_HOME wins over hf_env.sh;
        with neither the job REFUSES instead of hard-coding one cluster's cache path."""
        shared = tmp_path / "shared"
        shared.mkdir()
        r = _run_prelude(tmp_path, {"SISTEER_SHARED_ROOT": str(shared), "SISTEER_HF_HOME": str(tmp_path / "hf")}, *self.ARGS)
        assert r.returncode == 0, r.stdout + r.stderr
        assert f"SHARED={shared} HF_HOME={tmp_path / 'hf'} ATTEMPT=1 MAX=4" in r.stdout
        # the shared copy of hf_env.sh derives the cache when the site passes no SISTEER_HF_HOME
        (shared / "scripts").mkdir()
        (shared / "scripts/hf_env.sh").write_text("export HF_HOME=/site/derived/cache\n")
        r = _run_prelude(tmp_path, {"SISTEER_SHARED_ROOT": str(shared)}, *self.ARGS, "wo", "4", "tp1", "3", "5")
        assert r.returncode == 0 and f"SHARED={shared} HF_HOME=/site/derived/cache ATTEMPT=3 MAX=5" in r.stdout
        # neither: a refusal naming both, never the shared model cache
        (shared / "scripts/hf_env.sh").unlink()
        r = _run_prelude(tmp_path, {"SISTEER_SHARED_ROOT": str(shared)}, *self.ARGS)
        assert r.returncode == 2 and "no SISTEER_HF_HOME" in r.stdout and "hf_env.sh" in r.stdout
        assert not token_hits(r.stdout, STORAGE_TOKENS), token_hits(r.stdout, STORAGE_TOKENS)
        # no shared root in the job environment: a refusal by name (nothing is probed for)
        r = _run_prelude(tmp_path, {}, *self.ARGS)
        assert r.returncode == 2 and "no SISTEER_SHARED_ROOT" in r.stdout

    def test_carrier_text_contract(self):
        text = SBATCH.read_text()
        roots = [text[e:e + 6] for _r, _s, e in STORAGE_TOKENS.find(text)]
        assert all(r.startswith(("/$USER", "/%u/")) for r in roots), roots   # no one-cluster fallback: every shared root is per user
        assert 'SISTEER_SHARED_ROOT' in text and '[ -d "/' not in text   # the job's site value; no storage path is probed for
        # the positional attempt counter, --no-requeue, the sentinel, the self-resubmission carrying the
        # SAME record (SERVING_PROBE_OUT) + --resume, on the driver's incomplete exit and on SIGTERM
        assert 'ATTEMPT="${7:-1}"' in text and 'MAX_ATTEMPTS="${8:-4}"' in text and "#SBATCH --no-requeue" in text
        # BOTH sentinels — the record's own and the stop helper's contract location — and
        # escalate() consults them through stop_requested (never the record's alone)
        assert 'STOP_FILE="$OUT/STOP"' in text and 'HELPER_STOP_FILE="$SHARED/exp/training/${SLURM_JOB_NAME:-serving-probe}/STOP"' in text
        assert 'if [ -e "$STOP_FILE" ]' in text and 'if [ -e "$HELPER_STOP_FILE" ]' in text
        esc = text[text.index("escalate(){"):text.index("WORK=\"\"\nhandled=0")]
        assert 'if sentinel="$(stop_requested)"' in esc and "not resubmitting" in esc
        # peft is REQUIRED by the venv guard — no tolerated ImportError, no "recorded unavailable"
        guard = text[text.index("\"$VENV/bin/python\" -c 'import vllm"):text.index("unset VLLM_PLUGINS")]
        assert "import peft" in guard and "except ImportError" not in guard and "recorded unavailable" not in guard
        assert "peft==0.20.0 is REQUIRED" in guard
        # the geometry fold's site overlay dir is exported for the driver's verdict
        assert 'export SISTEER_GEOMETRY_DIR="${SISTEER_GEOMETRY_DIR:-$SHARED/exp/reproduce/geometry}"' in text
        assert text.index('export SISTEER_GEOMETRY_DIR=') < text.index("serving_probe.py verdict")
        assert "geometry_record.json" in text[text.index("for f in probe_plan.json"):text.index("mirror-figs.sh")]
        # EXIT_UNCERTIFIED (6) is a terminal state, never requeued: only 4 escalates
        assert "6) write_result UNCERTIFIED 6" in text and '[ "$RC" -eq 6 ]' not in text
        # the successor carries positional 9 (parallel) after the attempt counter and cap
        assert 'SERVING_PROBE_OUT="$OUT"' in text and '"$((ATTEMPT + 1))" "$MAX_ATTEMPTS" "$PARALLEL")' in text
        # a resumed attempt also re-plans (S re-derived from the record's measured batched-token budget)
        assert 'RESUME=(--resume --replan)' in text and '"${RESUME[@]}"' in text
        assert '[ "$RC" -eq 4 ] && escalate' in text and "FINAL_RC=143" in text and "escalate \"SIGTERM\"" in text
        assert "4) write_result INCOMPLETE 4" in text and "143) write_result INTERRUPTED 143" in text
        # the SIGTERM handler disarms FIRST (the re-entry measured 2026-09-11)
        on_term = text[text.index("on_term(){"):text.index("trap on_term SIGTERM")]
        assert on_term.strip().splitlines()[1].strip() == "trap '' SIGTERM"
        # hard rules: never an --export= on a command, CUDA_VISIBLE_DEVICES never SET (the tp2 guard READS Slurm's value
        # into a log line, which is fine), no --mem=0, the sha pin, the srun wrap
        assert re.search(r"^[^#\n]*--export=", text, re.M) is None
        assert re.search(r"^\s*(export\s+)?CUDA_VISIBLE_DEVICES=", text, re.M) is None and "export CUDA_VISIBLE_DEVICES" not in text
        assert "--mem=0" not in text
        # pin-fix: the resubmission carries the job's WHOLE comment — the sha field the submitter validated (== HEAD, checked at start) AND the
        # funnel's nonce, so the successor stays adoptable by its positive identity; a re-minted `sha:$HEAD_SHA` alone would drop the nonce
        assert '--comment="$COMMENT"' in text and '--comment="sha:$HEAD_SHA"' not in text
        assert "srun --ntasks=1" in text
        assert_no_tokens(text, str(SBATCH))                 # the shared token table (tests/front_door_testlib.py)

    @staticmethod
    def _run_escalate(tmp_path, *, sentinel):
        """Run the carrier's OWN sentinel definitions + escalate() under bash with `sbatch` stubbed: returns (log, sbatch calls).
        `sentinel`: None | "record" ($OUT/STOP) | "helper" ($SHARED/exp/training/<job-name>/STOP)."""
        text = SBATCH.read_text()
        frag = (text[text.index('STOP_FILE="$OUT/STOP"'):text.index("log(){ echo")] + "\n"
                + text[text.index("escalate(){"):text.index("WORK=\"\"\nhandled=0")])
        (tmp_path / "frag.sh").write_text(frag)
        bin_dir = tmp_path / "bin"; bin_dir.mkdir(exist_ok=True)
        calls = tmp_path / "sbatch_calls"
        (bin_dir / "sbatch").write_text(f'#!/bin/bash\necho "$*" >> "{calls}"; echo "999;cluster"\n'); (bin_dir / "sbatch").chmod(0o755)
        out, shared = tmp_path / "out", tmp_path / "shared"
        out.mkdir(exist_ok=True); shared.mkdir(exist_ok=True)
        (out / "STOP").unlink(missing_ok=True)
        helper = shared / "exp/training/serving-probe-x/STOP"; helper.unlink(missing_ok=True)
        if sentinel == "record":
            (out / "STOP").touch()
        elif sentinel == "helper":
            helper.parent.mkdir(parents=True, exist_ok=True); helper.touch()
        calls.unlink(missing_ok=True)
        script = (f'set -u; OUT="{out}"; SHARED="{shared}"; SLURM_JOB_NAME=serving-probe-x; ATTEMPT=1; MAX_ATTEMPTS=4; '
                  f'SCRIPT_PATH="{tmp_path / "frag.sh"}"; ESCALATE_QOS=high; TIMELIMIT=04:00:00; TP=1; NGPU=8; PARALLEL=auto; STDOUT_PATH=; HEAD_SHA=abc; '
                  'COMMENT=sha:abc,nonce:0123456789ab; '      # the job's own comment (pin + the funnel's nonce) — the successor carries it whole (pin-fix)
                  'MODEL_KEY=k; SERVED_ID=o/t; REVISION=r; SETS=wo; RANKS=4; MODE=tp1; SUCCESSOR=; log(){ echo "[log] $*"; }; '
                  f'source "{tmp_path / "frag.sh"}"; escalate "why"; echo "SUCCESSOR=$SUCCESSOR"')
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                           env={"PATH": f"{bin_dir}:{os.environ['PATH']}", "USER": "nobody-testhost", "HOME": str(tmp_path)})
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout, (calls.read_text().splitlines() if calls.exists() else [])

    def test_escalate_stands_down_on_either_stop_sentinel(self, tmp_path):
        """RUN not grepped: the carrier's escalate() resubmits attempt+1 with no
        sentinel, and stands down — no sbatch call, the log naming the file — when EITHER $OUT/STOP or the stop helper's
        $SHARED/exp/training/<job-name>/STOP is present. Fail-before: the helper's sentinel was ignored and a successor
        was queued, defeating the cancellation."""
        log, calls = self._run_escalate(tmp_path, sentinel=None)
        assert len(calls) == 1 and "successor job 999 queued" in log and "SUCCESSOR=999" in log
        # positional 9 (parallel) rides the successor and its --gres is THIS job's GPU count, not the per-standup tp
        assert calls[0].endswith("k o/t r wo 4 tp1 2 4 auto") and "--comment=sha:abc,nonce:0123456789ab" in calls[0] and "--export" not in calls[0]
        assert "--gres=gpu:8" in calls[0] and "--gres=gpu:1" not in calls[0]
        for which, path in (("helper", tmp_path / "shared/exp/training/serving-probe-x/STOP"), ("record", tmp_path / "out/STOP")):
            log, calls = self._run_escalate(tmp_path, sentinel=which)
            assert calls == [] and f"STOP sentinel {path} present" in log and "not resubmitting" in log and "SUCCESSOR=" in log
            assert "queued" not in log


# --------------------------------------------------------------------------- the served launch class (2026-10-11)


W5 = tuple(sm.PRODUCTION_WINDOWS)


def _without_union(plan):
    """A record as the tree BEFORE the ruling planned it: the pair's split servers and ladder alone (the live smoke record's shape)."""
    old = json.loads(json.dumps(plan))
    union = {n for n, m in old["manifests"].items() if m["topology"] == sm.TOPOLOGY_SINGLE}
    old["manifests"] = {n: m for n, m in old["manifests"].items() if n not in union}
    old["launches"] = [r for r in old["launches"] if r["manifest"] not in union]
    old["n_launches"] = len(old["launches"])
    for pr in old["pairs"]:
        pr["manifests"] = [n for n in pr["manifests"] if n not in union]
    return old


class TestServedClassUnion:
    """The probe owner's ruling of 2026-10-11: the battery's steered ``wo`` standups serve (rank cap 8, 8 adapters) — control r4,
    constitution r4, difference r8, the four carriers, ``none__lora0`` — and the paper-pair probe had folded only the split servers'
    (8, 7) and (8, 2), so every multi-adapter stage-06 standup refused at ``launch-class geometry`` (no probed launch at window 4096
    dominates the served class). The plan now carries each pair's UNION manifest (``standup_manifest.build_union_manifest``) as one
    natural launch at every window; the fold writes its engine class; the extension adds it to an existing record where nothing
    planned dominates it; the engine's attested class must equal the planned one."""

    def test_the_union_manifest_is_the_served_class_of_a_steered_standup(self):
        u = sm.build_union_manifest(Q4, set_id="wo", rank=4)
        assert (u.name, u.topology, u.max_lora_rank, u.max_loras, u.target_class) == ("wo_r4_single", "single", 8, 8, ("o_proj",))
        assert [(a.name, a.kind, a.rank) for a in u.adapters] == [
            ("control__lora", "genuine", 4), ("constitution__lora", "genuine", 4), ("difference__lora", "genuine", 8), ("none__lora0", "zero", 8),
            ("refusal_direction__carrier", "carrier", 1), ("relu_control__carrier", "carrier", 1), ("relu_constitution__carrier", "carrier", 1),
            ("relu_difference__carrier", "carrier", 1)]
        assert u.genuine_roles == ("control", "constitution", "difference") and sm.UNION_NOTE in u.notes and u.moe_lora_rank is None
        # the union of the pair's servers: every genuine adapter of S1cc and S1d, the zero at S1d's cap, S1cc's carriers
        cc, d = sm.build_manifests(Q4, set_id="wo", rank=4)
        assert {a.name for a in u.adapters} == {a.name for a in cc.adapters} | {a.name for a in d.adapters}
        assert u.max_lora_rank == d.max_lora_rank == sm.rank_cap(8) and u.launch_key not in (cc.launch_key, d.launch_key)
        # the other sets: cells + zero, 4 slots (the carriers ride S2); on the MoE toy the expert rank follows the adapters
        m = sm.build_union_manifest(TOY, set_id="mlp", rank=4)
        assert (m.name, m.max_lora_rank, m.max_loras) == ("mlp_r4_single", 8, 4) and not any(a.kind == "carrier" for a in m.adapters)
        moe = sm.build_union_manifest(MOE_TOY, set_id="mlp", rank=4)
        assert moe.wraps_experts and moe.moe_lora_rank == 4 == max(a.per_expert_rank for a in moe.adapters)
        assert sm.build_union_manifest(Q4, set_id="wo", rank=64).max_lora_rank == 128
        with pytest.raises(sm.ManifestError, match="unknown module set"):
            sm.build_union_manifest(Q4, set_id="nope", rank=4)
        # the classes the geometry rule compares: (8, 7) and (8, 2) never dominate (8, 8); the union does
        assert sp._union_class(u) == (8, 8) and sp._union_class(cc) == (8, 7) and sp._union_class(d) == (8, 2)
        assert sp._is_union(u) and not sp._is_union(cc) and not sp._is_union(d)

    def test_build_plan_carries_the_union_as_a_natural_launch_at_every_window_after_the_servers(self):
        plan = _plan(sets=("wo",), ranks=(4,), windows=W5, dims=Q4)
        # the live smoke record's 14 launches (10 natural servers + 4 ladder points) + one union launch per window
        assert plan["n_launches"] == 19
        union = [r for r in plan["launches"] if r["topology"] == "single"]
        assert sorted(r["L"] for r in union) == sorted(W5) and len(union) == 5
        for r in union:
            assert r["id"] == f"wo_r4_single_cap8_L{r['L']}" and r["kind"] == "natural" and r["manifest"] == "wo_r4_single"
            assert (r["cap"], r["max_loras"], r["roles"], r["condition"], r["target_class"]) == (8, 8, ["control", "constitution", "difference"], None, ["o_proj"])
            assert r["priority"] == sp._launch_priority(sp.plan_manifest(plan, "wo_r4_S1cc"), r["L"], 4096)       # the servers' priority at that window
        # AFTER the servers in the pair's manifests: the role order reads S1cc / S1d first, the union last
        assert plan["pairs"][0]["manifests"] == ["wo_r4_S1cc", "wo_r4_S1d", "wo_r4_single"]
        role_m = sp._role_manifests([sp.plan_manifest(plan, n) for n in plan["pairs"][0]["manifests"]])
        assert [m.name for m in role_m["control"]] == ["wo_r4_S1cc", "wo_r4_single"] and [m.name for m in role_m["difference"]] == ["wo_r4_S1d", "wo_r4_single"]
        # at each window the servers sort before the union; the ladder still rides S1cc and has no union point
        at = {L: [r["id"] for r in plan["launches"] if r["L"] == L and r["kind"] == "natural"] for L in W5}
        assert all(at[L] == [f"wo_r4_S1cc_cap8_L{L}", f"wo_r4_S1d_cap8_L{L}", f"wo_r4_single_cap8_L{L}"] for L in W5)
        ladder = sorted((r["cap"], r["manifest"]) for r in plan["launches"] if r["kind"] == "ladder")
        assert ladder == [(16, "wo_r4_S1cc_cap16"), (32, "wo_r4_S1cc_cap32"), (64, "wo_r4_S1cc_cap64"), (128, "wo_r4_S1cc_cap128")]
        assert not any("single" in n for _c, n in ladder)
        # every other set gets its 4-slot union too, and the shared S2 is untouched
        mlp = _plan(sets=("mlp",), ranks=(4, 16), windows=(4096,))
        assert sorted(r["id"] for r in mlp["launches"] if r["topology"] == "single") == ["mlp_r16_single_cap32_L4096", "mlp_r4_single_cap8_L4096"]
        assert all(r["max_loras"] == 4 for r in mlp["launches"] if r["topology"] == "single")
        assert [r["serves"] for r in mlp["launches"] if r["topology"] == "S2"] == [["mlp_r4_S2", "mlp_r16_S2"]]
        # a PARITY PASS plans no union: it folds no geometry and judges adapters by name; the union adds no adapter
        pp = _plan(sets=("wo", "mlp"), ranks=(4,), windows=(4096,), ladder_caps=(), parity_pass=True, parity_scales=sp.PARITY_SCALES_BY_SET)
        assert pp["n_launches"] == 5 and not any(r["topology"] == "single" for r in pp["launches"])
        assert all("single" not in n for pr in pp["pairs"] for n in pr["manifests"])

    def test_the_fold_writes_the_union_class_and_the_served_class_picks_it_at_every_window(self, tmp_path, monkeypatch):
        from sisteer.capeval.serving import geometry as g
        plan = _plan(sets=("wo",), ranks=(4,), windows=W5, dims=Q4, ladder_caps=())
        # every launch passed; the union came up one sequence below its servers at every window, so the pick is visible
        outcomes = {r["id"]: ("passed", {"engine_S": max(1, int(r["admitted_S"]) - 1)} if r["topology"] == "single" else {}) for r in plan["launches"]}
        results = _fixture_results(plan, outcomes=outcomes, dims=Q4)
        out, site = tmp_path / "rec", tmp_path / "site"
        v, allowed = sp.write_verdict(plan, results, probe_job_id="4242", out=out, verdict_uri="file://v", geometry_dir=site)
        assert v["status"] == "complete" and v["certifying"] is True and [(p["rank"], p["set"]) for p in allowed["pairs"]] == [(4, "wo")]
        p = allowed["pairs"][0]
        # the roles stay certified by the split servers: certified_by and the audit trail never name the union
        assert p["topology"] == "S1ccS1d" and p["caps_by_server"] == {"S1cc": 8, "S1d": 8}
        assert p["certified_by"] == [lid for L in W5 for lid in (f"wo_r4_S1cc_cap8_L{L}", f"wo_r4_S1d_cap8_L{L}")]
        assert p["certifying_by_window"] == {str(L): {"control": f"wo_r4_S1cc_cap8_L{L}", "difference": f"wo_r4_S1d_cap8_L{L}"} for L in W5}
        assert p["admitted_max_num_seqs"] == {str(L): results[f"wo_r4_S1cc_cap8_L{L}"]["engine"]["max_num_seqs_engine"] for L in W5}
        # the served class's OWN S per window is the union launch's — one below the roles' at every window here — the number the
        # serving rows price from (round 2 on the served-class work); this fixture records no throughput
        assert p["served_class"]["measured_S_by_window"] == {str(L): results[f"wo_r4_single_cap8_L{L}"]["engine"]["max_num_seqs_engine"] for L in W5}
        assert all(p["served_class"]["measured_S_by_window"][str(L)] == p["admitted_max_num_seqs"][str(L)] - 1 for L in W5)
        assert p["served_class"]["throughput_by_window"] == {str(L): None for L in W5}
        # ...and the STANDUP resolution of the class over the fold: one certified set, so the all-set rule is the union's S at
        # every window inside the record's budget // L (the record's largest S x L; a union S x L never exceeds it, so the quotient
        # binds nowhere below the union here), the union launch named at each window, geometry's source label — on the pair record
        # and in the fold's own block
        prod = max(results[r["id"]]["engine"]["max_num_seqs_engine"] * int(r["L"]) for r in plan["launches"] if r["manifest"] in plan["pairs"][0]["manifests"])
        st = p["served_class"]["standup"]
        assert st == {"S_by_window": {str(L): min(prod // L, results[f"wo_r4_single_cap8_L{L}"]["engine"]["max_num_seqs_engine"]) for L in W5},
                      "launches_by_window": {str(L): [f"wo_r4_single_cap8_L{L}"] for L in W5},
                      "source": "launch_class:8x8", "batched_token_budget": prod, "refused": None}
        assert st["S_by_window"] == p["served_class"]["measured_S_by_window"] and v["geometry_fold"][sp.STANDUP_S_KEY] == {"4:wo": st}
        # the fold wrote the union's engine class (8, 8) beside S / L at every window
        assert v["geometry_fold"]["status"] == "written"
        monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(site))
        rec = g.load_site_geometry(MODEL).sets["wo"]
        bl = dict(rec.budget_launches)
        for L in W5:
            S = results[f"wo_r4_single_cap8_L{L}"]["engine"]["max_num_seqs_engine"]
            assert bl[f"wo_r4_single_cap8_L{L}"] == {"S": S, "L": L, "batched_tokens": S * L, "max_lora_rank": 8, "max_loras": 8}
            assert bl[f"wo_r4_S1cc_cap8_L{L}"]["max_loras"] == 7 and bl[f"wo_r4_S1d_cap8_L{L}"]["max_loras"] == 2
        assert sorted(rec.certified_by) == sorted(p["certified_by"])                       # the union certifies no role
        # least_admitted_by_window(served_lora=(8, 8)) picks the union at EVERY window; (8, 7) still reads the S1cc launches
        ws, src, chosen = g.least_admitted_by_window(MODEL, bl, (8, 8))
        assert src == "launch_class:8x8" and chosen == {L: (f"wo_r4_single_cap8_L{L}",) for L in W5}
        assert ws == {L: results[f"wo_r4_single_cap8_L{L}"]["engine"]["max_num_seqs_engine"] for L in W5}
        ws7, _src7, chosen7 = g.least_admitted_by_window(MODEL, bl, (8, 7))
        assert chosen7 == {L: (f"wo_r4_S1cc_cap8_L{L}",) for L in W5} and all(ws7[L] == ws[L] + 1 for L in W5)
        # ...and through the resolver a steered (8, 8) standup is sized from them at every window
        for L in W5:
            rb = g.budget_for(MODEL, served_lora=(8, 8))
            assert rb.entry.window_seqs_launches[L] == (f"wo_r4_single_cap8_L{L}",) and rb.entry.seqs_at(L) == ws[L]
            assert st["S_by_window"][str(L)] == rb.entry.seqs_at(L)          # the verdict's standup number IS the resolver's
        # a record that folded the servers alone (the live smoke record) refuses the served class by name at the first window
        servers_only = {lid: b for lid, b in bl.items() if "_single_" not in lid}
        with pytest.raises(g.SiteGeometryError, match=r"no probed launch at window 4096 dominates the served adapter class \(rank cap 8, 8 adapters\)"):
            g.least_admitted_by_window(MODEL, servers_only, (8, 8))

    def test_the_extension_adds_exactly_the_union_launches_to_a_certified_record_and_changes_nothing_else(self):
        """The live smoke record: wo r4's servers and ladder passed at every window, certified, folded — and no launch covers the
        (8, 8) class. The reopen's extension adds the five union launches (one per window, `served_class_undominated`), is
        idempotent, leaves every old launch record byte-identical, and the role selection and the parity binding read the same."""
        old = _without_union(_plan(sets=("wo",), ranks=(4,), windows=W5, dims=Q4))
        assert old["n_launches"] == 14 and old["pairs"][0]["manifests"] == ["wo_r4_S1cc", "wo_r4_S1d"]
        results = _fixture_results(old, outcomes={r["id"]: ("passed",) for r in old["launches"]}, dims=Q4)
        st0 = sp.pair_serving_status(old, results, old["pairs"][0])
        assert st0["certified_all"] and sp.outstanding_launches(old, results) == [] and sp.replan_required_launches(old, results) == []
        new, ext = sp.extend_plan_for_uncertified_pairs(old, results, Q4, tree_sha="t" * 40)
        assert ext["added"] == [f"wo_r4_single_cap8_L{L}" for L in W5]                   # window order; nothing else
        assert ext["why"] == {lid: sp.REASON_SERVED_CLASS for lid in ext["added"]} and sp.REASON_SERVED_CLASS == "served_class_undominated"
        assert ext["manifests"] == ["wo_r4_single"] and ext["pairs"] == [[4, "wo"]]
        assert new["n_launches"] == 19 and new["pairs"][0]["manifests"] == ["wo_r4_S1cc", "wo_r4_S1d", "wo_r4_single"]
        e = new["extensions"][0]
        assert e["tree_sha"] == "t" * 40 and e["served_class"] is True and e["launches"] == ext["added"] and "served launch class" in e["rule"]
        added = {r["id"]: r for r in new["launches"] if r["id"] in ext["added"]}
        assert all((r["kind"], r["cap"], r["max_loras"], r["topology"], r["roles"]) == ("natural", 8, 8, "single", ["control", "constitution", "difference"])
                   for r in added.values())
        assert sm.Manifest.from_dict(new["manifests"]["wo_r4_single"]) == sm.build_union_manifest(Q4, set_id="wo", rank=4)
        # completed launches are canonical: every old record is untouched, byte for byte
        by_old = {r["id"]: r for r in old["launches"]}
        assert all(by_old[r["id"]] == r for r in new["launches"] if r["id"] in by_old) and len(by_old) == 14
        # idempotent: the union now dominates itself at every window
        again, ext2 = sp.extend_plan_for_uncertified_pairs(new, results, Q4)
        assert ext2 == {"added": [], "manifests": [], "pairs": [], "why": {}} and again == new
        U = sm.build_union_manifest(Q4, set_id="wo", rank=4)
        for L in W5:
            assert sp._served_class_dominated(new, new["pairs"][0], U, L, {n: sp.plan_manifest(new, n) for n in new["manifests"]}) == [f"wo_r4_single_cap8_L{L}"]
            assert sp._served_class_dominated(old, old["pairs"][0], U, L, {n: sp.plan_manifest(old, n) for n in old["manifests"]}) == []
        # the role selection reads the same before and after. The OLD record's verdict: the roles are certified at every window, and
        # the pair is EXCLUDED, decided, for the class nothing planned ever served (a completed probe never
        # certifies a class it never served — the standup would refuse it at `launch-class geometry` anyway); the reopen's
        # extension is the answer to exactly that reason. With the union UNRUN the reopened record is PARTIAL — the five union
        # launches are the pair's decision-relevant work (a fold must wait for them), nothing else is outstanding; once they pass,
        # the certification is the servers' one, field for field, the union named per window, and the parity binding reads the
        # same manifests
        assert sp.pair_serving_status(new, results, new["pairs"][0]) == st0
        v0, a0 = sp.write_verdict(old, results, probe_job_id="1")
        assert v0["status"] == "complete" and a0["pairs"] == [] and v0["serving_certified_pairs"] == [] and v0["undecided_pairs"] == []
        e0 = v0["excluded"][0]
        assert (e0["rank"], e0["set"], e0["decided"], e0["outstanding"]) == (4, "wo", True, [])
        assert e0["reasons"] == [f"served_class_unplanned_at_L={L}:(rank cap 8, 8 adapters)" for L in W5] and e0["reason"] == e0["reasons"][0]
        assert a0["excluded"] == [{"rank": 4, "set": "wo", "reason": e0["reason"], "decided": True}]
        assert e0["served_class"] == {"max_lora_rank": 8, "max_loras": 8, "target_class": ["o_proj"], "manifest": None,
                                      "certified_by_window": {str(L): None for L in W5},
                                      "measured_S_by_window": {str(L): None for L in W5}, "throughput_by_window": {str(L): None for L in W5}}
        assert e0["certified_by"] == [lid for L in W5 for lid in (f"wo_r4_S1cc_cap8_L{L}", f"wo_r4_S1d_cap8_L{L}")]
        assert e0["certifying_by_window"] == {str(L): {"control": f"wo_r4_S1cc_cap8_L{L}", "difference": f"wo_r4_S1d_cap8_L{L}"} for L in W5}
        assert e0["caps_by_server"] == {"S1cc": 8, "S1d": 8} and e0["topology"] == "S1ccS1d"
        assert v0["geometry_fold"]["status"] == "not_applicable"                              # nothing folded: no site record for the class
        v_unrun, a_unrun = sp.write_verdict(new, results, probe_job_id="1")
        assert v_unrun["status"] == "partial" and v_unrun["undecided_pairs"] == [[4, "wo"]] and a_unrun["pairs"] == []
        assert sorted(o["id"] for o in v_unrun["outstanding"] if o["decision_relevant"]) == sorted(ext["added"])
        assert sorted(o["id"] for o in v_unrun["outstanding"]) == sorted(ext["added"])
        assert v_unrun["excluded"][0]["reasons"] == ["not_run"] and v_unrun["excluded"][0]["served_class"]["certified_by_window"] == {str(L): None for L in W5}
        res_run = {**results, **_fixture_results(new, outcomes={lid: ("passed",) for lid in ext["added"]}, dims=Q4)}
        v1, a1 = sp.write_verdict(new, res_run, probe_job_id="1")
        assert v1["status"] == "complete" and v1["outstanding"] == [] and v1["serving_certified_pairs"] == [[4, "wo"]]
        p1 = a1["pairs"][0]
        for k in ("certified_by", "certifying_by_window", "caps_by_server", "topology"):
            assert p1[k] == e0[k], k                                                         # the roles' certification: the old one, field for field
        assert p1["admitted_max_num_seqs"] == {str(L): results[f"wo_r4_S1cc_cap8_L{L}"]["engine"]["max_num_seqs_engine"] for L in W5}
        assert p1["reference_check_basis"] == sp.BASIS_HF_PEFT
        # the standup resolution over the fold (an in-memory verdict: the fold is record_only, the resolution runs on the record
        # it built): one set, so the union's S at each window inside the record's largest S x L // L
        prod = max(res_run[r["id"]]["engine"]["max_num_seqs_engine"] * int(r["L"]) for r in new["launches"] if r["manifest"] in new["pairs"][0]["manifests"])
        assert v1["geometry_fold"]["status"] == "record_only"
        assert p1["served_class"] == {"max_lora_rank": 8, "max_loras": 8, "target_class": ["o_proj"], "manifest": "wo_r4_single",
                                      "certified_by_window": {str(L): f"wo_r4_single_cap8_L{L}" for L in W5},
                                      # the class's own S: the union launches' engines (the plan's admitted S here); no throughput recorded
                                      "measured_S_by_window": {str(L): res_run[f"wo_r4_single_cap8_L{L}"]["engine"]["max_num_seqs_engine"] for L in W5},
                                      "throughput_by_window": {str(L): None for L in W5},
                                      "standup": {"S_by_window": {str(L): min(prod // L, res_run[f"wo_r4_single_cap8_L{L}"]["engine"]["max_num_seqs_engine"]) for L in W5},
                                                  "launches_by_window": {str(L): [f"wo_r4_single_cap8_L{L}"] for L in W5},
                                                  "source": "launch_class:8x8", "batched_token_budget": prod, "refused": None}}
        parity = {"plan": {"launches": []}, "results": {}}                           # a pass that planned nothing: the labels name the manifests read
        assert sp.parity_reference_sources(new, p1["certified_by"], parity) == sp.parity_reference_sources(old, e0["certified_by"], parity)
        # a union launch whose start-time OOM is a re-plan holds the pair undecided exactly as a server's would
        over = dict(res_run)
        over["wo_r4_single_cap8_L98304"] = {"launch_id": "wo_r4_single_cap8_L98304", "manifest": "wo_r4_single", "status": "oom", "launch_S": 64,
                                            "ready_s": None, "failure": {"class": "oom", "message": "cuda_out_of_memory", "log_tail": "EngineCore failed to start"}}
        v_over, a_over = sp.write_verdict(new, over, probe_job_id="1")
        assert v_over["status"] == "partial" and v_over["replan_required"] == ["wo_r4_single_cap8_L98304"] and a_over["pairs"] == []
        # a record whose union is planned but DEAD at a window gets no second copy (the twin would die the same way) — and its verdict
        # EXCLUDES the pair, decided, naming the window and the launch: the servers passed everywhere, the (8, 8)
        # class died at 98,304, and nothing certifies it there
        dead = dict(res_run)
        dead["wo_r4_single_cap8_L98304"] = {"launch_id": "wo_r4_single_cap8_L98304", "status": "oom", "failure": {"class": "oom", "message": "oom"}}
        assert sp.extend_plan_for_uncertified_pairs(new, dead, Q4)[1]["added"] == []
        v_dead, a_dead = sp.write_verdict(new, dead, probe_job_id="1")
        assert v_dead["status"] == "complete" and a_dead["pairs"] == [] and v_dead["serving_certified_pairs"] == [] and v_dead["undecided_pairs"] == []
        e_dead = v_dead["excluded"][0]
        assert e_dead["reasons"] == [f"{sp.REASON_SERVED_CLASS}_at_L=98304:wo_r4_single_cap8_L98304:oom_at_R=8"] and e_dead["decided"] is True
        assert e_dead["served_class"]["certified_by_window"] == {**{str(L): f"wo_r4_single_cap8_L{L}" for L in W5}, "98304": None}
        assert e_dead["certifying_by_window"] == p1["certifying_by_window"]                 # the roles still read certified by the servers
        # a parity-pass record gets none
        pp = json.loads(json.dumps(old)); pp["parity_pass"] = True; pp["windows"] = [4096]
        pp["launches"] = [r for r in pp["launches"] if r["L"] == 4096]; pp["n_launches"] = len(pp["launches"])
        assert sp.extend_plan_for_uncertified_pairs(pp, results, Q4)[1]["added"] == []

    def test_run_resume_replan_dry_run_prints_the_five_union_launches_and_touches_nothing(self, tmp_path, capsys):
        old = _without_union(_plan(sets=("wo",), ranks=(4,), windows=W5, dims=Q4))
        results = _fixture_results(old, outcomes={r["id"]: ("passed",) for r in old["launches"]}, dims=Q4)
        out = tmp_path / "rec"
        sp.write_json_atomic(out / sp.PLAN_FILE, old)
        for lid, res in results.items():
            sp.write_json_atomic(sp.launch_result_path(out, lid), res)
        assert sp.write_progress(old, results, out)["complete"] is True
        before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
        rc = sp.main(["run", "--model-key", MODEL, "--out", str(out), "--resume", "--replan", "--dry-run"])
        text = capsys.readouterr().out
        assert rc == 0, text
        assert "DRY RUN of run --resume --replan" in text and "plan 14 launch(es), 14 done, checkpoint COMPLETE" in text
        assert "plan extension (5 launch(es), 1 new manifest(s), pairs [[4, 'wo']])" in text
        for L in W5:
            assert re.search(rf"\+ wo_r4_single_cap8_L{L}\s+served_class_undominated\s+class=\(cap 8, 8 slots\) topology=single L={L} ", text), L
        assert "work: 5 launch(es) would run" in text and "this resume is a REOPEN and needs the front door's receipt" in text
        assert "dry run: nothing written, no lock taken, no receipt consumed" in text
        assert {p: p.read_bytes() for p in out.rglob("*") if p.is_file()} == before                   # byte-identical, no receipt, no archive
        assert not (out / sp.RECEIPTS_DIR).is_dir() or not list((out / sp.RECEIPTS_DIR).glob("*.json"))
        # without --replan the extension is not part of the work and the dry run says so; without --resume it refuses
        rc = sp.main(["run", "--model-key", MODEL, "--out", str(out), "--resume", "--dry-run"])
        text = capsys.readouterr().out
        assert rc == 0 and "plan extension (0 launch(es)" in text and "nothing would run: a reopen on this tree would mint a receipt for no work" in text
        with pytest.raises(SystemExit, match="previews the work of a RESUME"):
            sp.main(["run", "--model-key", MODEL, "--out", str(out), "--dry-run"])
        assert {p: p.read_bytes() for p in out.rglob("*") if p.is_file()} == before

    def test_the_attested_launch_class_must_equal_the_planned_one(self, tmp_path):
        """(a)'s assertion: the engine's attested class (worker_reports' lora_max_lora_rank / lora_max_loras) equals the manifest's cap
        and slot count — a HARD check at run time, and the fold refuses a passed launch whose attested class is not the planned one."""
        u = sm.build_union_manifest(TOY, set_id="wo", rank=4)
        rec = {"expected_lora_bytes": {"total": 1, "per_slot": {}}}
        base = {"rows": [], "token_ids": None}

        def checks(report):
            return sp.evaluate_checks(manifest=u, launch_rec=rec, dims=TOY, log_text="", models_listed=[a.name for a in u.adapters],
                                      base_probe=base, adapter_probes={}, worker_report=report, require_measurements=False)
        ok = checks({"lora_max_lora_rank": 8, "lora_max_loras": 8})
        assert ok["launch_class"] == {"status": "measured", "expected": {"max_lora_rank": 8, "max_loras": 8},
                                      "reported": {"max_lora_rank": 8, "max_loras": 8}, "pass": True} and "launch_class" not in ok["failed"]
        bad = checks({"lora_max_lora_rank": 8, "lora_max_loras": 7})
        assert bad["launch_class"]["pass"] is False and "launch_class" in bad["failed"]
        assert "(rank cap 8, 7 adapters)" in bad["launch_class"]["reason"] and "planned at (rank cap 8, 8 adapters)" in bad["launch_class"]["reason"]
        assert checks({"lora_max_lora_rank": 16, "lora_max_loras": 8})["launch_class"]["pass"] is False
        none = checks({})
        assert none["launch_class"]["status"] == "not_reported" and none["launch_class"]["pass"] is None and "launch_class" not in none["failed"]
        # the fold: a passed union launch whose engine attested (8, 7) is refused by name, never folded as (8, 8)
        plan = _plan(sets=("wo",), ranks=(4,), windows=(4096,), ladder_caps=())
        results = _fixture_results(plan, outcomes={r["id"]: ("passed",) for r in plan["launches"]})
        v_ok, a_ok = sp.write_verdict(plan, results, probe_job_id="1")
        assert v_ok["serving_certified_pairs"] == [[4, "wo"]] and a_ok["pairs"][0]["served_class"]["certified_by_window"] == {"4096": "wo_r4_single_cap8_L4096"}
        results["wo_r4_single_cap8_L4096"]["engine"]["max_loras"] = 7
        # the verdict first: a passed launch attesting another class than the plan's never covers the served
        # class — the pair is excluded by name, nothing is folded for it
        v, a = sp.write_verdict(plan, results, probe_job_id="1", out=tmp_path)
        assert v["status"] == "complete" and a["pairs"] == [] and v["serving_certified_pairs"] == []
        assert v["excluded"][0]["reasons"] == [f"{sp.REASON_SERVED_CLASS}_at_L=4096:wo_r4_single_cap8_L4096:attested_class=(8, 7)!=planned=(8, 8)"]
        assert v["excluded"][0]["decided"] is True and v["geometry_fold"]["status"] == "not_applicable"
        # ...and the fold's own gate stands behind it: handed the pair anyway, it refuses the mis-attested launch by name
        fold = sp.geometry_fold(plan, results, a_ok["pairs"], out=tmp_path, probe_job_id="1", verdict_uri=None)
        assert fold["status"] == "refused" and "wo_r4_single_cap8_L4096: the engine attested launch class" in fold["reason"]
        assert "{'max_lora_rank': 8, 'max_loras': 7}" in fold["reason"] and "the plan intended {'max_lora_rank': 8, 'max_loras': 8}" in fold["reason"]

    def test_a_union_launch_that_died_at_one_window_excludes_the_pair_whose_class_it_never_served(self, tmp_path, monkeypatch):
        """On the served-class work: a wo / r4 two-window record whose split servers PASSED
        at both windows and whose union passed at 4,096 and OOM'd (a genuine death, inside the measured floor) at 49,152 — the verdict
        used to return COMPLETE with the pair allowed and `serving_certified`, the fold wrote only (8, 7) / (8, 2) at 49,152, and
        `least_admitted_by_window((8, 8))` refused the standup there. Now the pair is EXCLUDED, decided, by the window and the launch;
        nothing is folded; no site record claims the class."""
        from sisteer.capeval.serving import geometry as g
        plan = _plan(sets=("wo",), ranks=(4,), windows=(4096, 49152), dims=Q4, ladder_caps=())
        outcomes = {r["id"]: ("passed",) for r in plan["launches"]}
        results = _fixture_results(plan, outcomes=outcomes, dims=Q4)
        rec = next(r for r in plan["launches"] if r["id"] == "wo_r4_single_cap8_L49152")
        results[rec["id"]] = {"launch_id": rec["id"], "manifest": "wo_r4_single", "status": "oom", "ready_s": None,
                              "launch_S": max(1, int(rec["admitted_S"])), "failure": {"class": "oom", "message": "cuda_out_of_memory"}}
        assert sp.oom_kind(rec, results[rec["id"]], sp.measured_budget(plan, results)["tokens"]) in (sp.OOM_START_WITHIN_BUDGET, sp.OOM_START_MIN_S)
        assert sp.replan_required_launches(plan, results) == [] and sp.outstanding_launches(plan, results) == []
        out, site = tmp_path / "rec", tmp_path / "site"
        v, allowed = sp.write_verdict(plan, results, probe_job_id="4242", out=out, verdict_uri="file://v", geometry_dir=site)
        assert v["status"] == "complete" and v["certifying"] is True                     # decided: the record has no launch left to run
        assert allowed["pairs"] == [] and v["serving_certified_pairs"] == [] and allowed["serving_certified_pairs"] == []
        e = v["excluded"][0]
        assert e["reasons"] == [f"{sp.REASON_SERVED_CLASS}_at_L=49152:wo_r4_single_cap8_L49152:oom_at_R=8"] and e["decided"] is True
        assert e["served_class"] == {"max_lora_rank": 8, "max_loras": 8, "target_class": ["o_proj"], "manifest": "wo_r4_single",
                                     "certified_by_window": {"4096": "wo_r4_single_cap8_L4096", "49152": None},
                                     "measured_S_by_window": {"4096": results["wo_r4_single_cap8_L4096"]["engine"]["max_num_seqs_engine"], "49152": None},
                                     "throughput_by_window": {"4096": None, "49152": None}}
        assert e["certifying_by_window"] == {str(L): {"control": f"wo_r4_S1cc_cap8_L{L}", "difference": f"wo_r4_S1d_cap8_L{L}"} for L in (4096, 49152)}
        assert allowed["excluded"] == [{"rank": 4, "set": "wo", "reason": e["reason"], "decided": True}]
        assert v["geometry_fold"]["status"] == "not_applicable" and not (out / sp.GEOMETRY_RECORD_FILE).exists()
        monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(site))
        assert g.load_site_geometry(MODEL) is None                                        # no overlay: nothing sizes the class anywhere
        # the status tool reads it as a SERVING exclusion (launches to run on a reopen, never a parity verdict)
        import hpo_probe_status as hps
        assert hps.serving_excluded(allowed) == {"serving_excluded": [[4, "wo", e["reason"][:120]]], "serving_excluded_n": 1}
        # the extension adds no second copy of the shape that died there; a verdict with the union UNRUN at 49,152 is undecided
        assert sp.extend_plan_for_uncertified_pairs(plan, results, Q4)[1]["added"] == []
        unrun = {k: r for k, r in results.items() if k != rec["id"]}
        v_u, a_u = sp.write_verdict(plan, unrun, probe_job_id="1")
        assert v_u["status"] == "partial" and v_u["undecided_pairs"] == [[4, "wo"]] and a_u["pairs"] == []
        assert v_u["excluded"][0]["reasons"] == ["not_run"] and v_u["excluded"][0]["served_class"]["certified_by_window"] == {"4096": "wo_r4_single_cap8_L4096", "49152": None}

    def test_the_shared_s2_never_stands_in_for_an_mlp_pairs_union_on_a_reopened_split_era_record(self, tmp_path, monkeypatch):
        """On the served-class work: on an existing split-topology mlp / r4 record (S1cc + S1d + the shared S2, no
        union) the S2 is an o_proj-only (cap 8, five-slot) launch at every window — it out-counts the mlp union (cap 8, four slots)
        on (cap, slots) alone, but serves another module class with a lighter adapter workload. The extension used to count it as
        dominating and added NO union; the site record then sized the (8, 4) mlp class from S2's S (the tightest dominating footprint,
        with no union folded). Now: the extension adds the union at both windows; the verdict on the old record excludes the pair for
        the unplanned class; with the union passed, the fold carries S2 and the union and `least_admitted_by_window((8, 4))` picks the
        union at every window — where the servers-only fold would have picked S2."""
        from sisteer.capeval.serving import geometry as g
        old = _without_union(_plan(sets=("mlp",), ranks=(4,), windows=(4096, 49152), ladder_caps=()))
        pair = old["pairs"][0]
        assert pair["manifests"] == ["mlp_r4_S1cc", "mlp_r4_S1d", "mlp_r4_S2"] and old["n_launches"] == 6
        s2 = {r["id"]: r for r in old["launches"] if r["topology"] == "S2"}
        assert all((r["cap"], r["max_loras"], r["target_class"]) == (8, 5, ["o_proj"]) for r in s2.values()) and len(s2) == 2
        U = sm.build_union_manifest(TOY, set_id="mlp", rank=4)
        assert (sp._union_class(U), U.target_class) == ((8, 4), sm.target_class_for(TOY, sm.MODULE_SETS["mlp"])) and U.target_class != ("o_proj",)
        by_name = {n: sp.plan_manifest(old, n) for n in old["manifests"]}
        for L in (4096, 49152):
            # (8, 5) >= (8, 4) on the numbers alone — and S2 is NOT a candidate: another target class
            assert sp._served_class_dominated(old, pair, U, L, by_name) == []
            assert [r["id"] for r in sp._served_class_candidates(old, pair, U, L, by_name)] == []
        results = _fixture_results(old, outcomes={r["id"]: ("passed", {"engine_S": (40 if r["topology"] == "S2" else 20)}) for r in old["launches"]})
        assert sp.pair_serving_status(old, results, pair)["certified_all"] and sp.outstanding_launches(old, results) == []
        new, ext = sp.extend_plan_for_uncertified_pairs(old, results, TOY, tree_sha="t" * 40)
        assert ext["added"] == ["mlp_r4_single_cap8_L4096", "mlp_r4_single_cap8_L49152"] and ext["manifests"] == ["mlp_r4_single"]
        assert ext["why"] == {lid: sp.REASON_SERVED_CLASS for lid in ext["added"]} and ext["pairs"] == [[4, "mlp"]]
        assert new["pairs"][0]["manifests"] == ["mlp_r4_S1cc", "mlp_r4_S1d", "mlp_r4_S2", "mlp_r4_single"]
        added = {r["id"]: r for r in new["launches"] if r["id"] in ext["added"]}
        assert all((r["cap"], r["max_loras"], r["topology"], r["target_class"]) == (8, 4, "single", list(U.target_class)) for r in added.values())
        assert sp.extend_plan_for_uncertified_pairs(new, results, TOY)[1]["added"] == []    # idempotent: the union covers itself
        # the old record's verdict: roles certified, the (8, 4) class never planned -> excluded, decided, by name
        v0, a0 = sp.write_verdict(old, results, probe_job_id="1")
        assert v0["status"] == "complete" and a0["pairs"] == [] and v0["serving_certified_pairs"] == []
        assert v0["excluded"][0]["reasons"] == [f"served_class_unplanned_at_L={L}:(rank cap 8, 4 adapters)" for L in (4096, 49152)]
        assert v0["excluded"][0]["served_class"]["target_class"] == list(U.target_class) and v0["excluded"][0]["served_class"]["manifest"] is None
        # with the union passed (one sequence below the servers, so the pick is visible) the pair certifies and the fold writes BOTH
        # S2 (8, 5) and the union (8, 4) at every window; the served class is sized from the union, never from S2
        res = {**results, **_fixture_results(new, outcomes={lid: ("passed", {"engine_S": 19}) for lid in ext["added"]})}
        out, site = tmp_path / "rec", tmp_path / "site"
        v1, a1 = sp.write_verdict(new, res, probe_job_id="1", out=out, verdict_uri="file://v", geometry_dir=site)
        assert v1["status"] == "complete" and [(p["rank"], p["set"]) for p in a1["pairs"]] == [(4, "mlp")] and v1["geometry_fold"]["status"] == "written"
        assert a1["pairs"][0]["served_class"]["certified_by_window"] == {"4096": "mlp_r4_single_cap8_L4096", "49152": "mlp_r4_single_cap8_L49152"}
        assert a1["pairs"][0]["certified_by"] == ["mlp_r4_S1cc_cap8_L4096", "mlp_r4_S1d_cap8_L4096", "mlp_r4_S1cc_cap8_L49152", "mlp_r4_S1d_cap8_L49152",
                                                  "mlp_r4_S2_cap8_L4096"]
        # round 2 on the served-class work: the roles' S (the servers' 20) and the CLASS's S (the union's 19, what a standup gets and what
        # the serving rows price from) are two numbers on the pair record, and the memo's verdict table shows both
        assert a1["pairs"][0]["admitted_max_num_seqs"] == {"4096": 20, "49152": 20}
        assert a1["pairs"][0]["served_class"]["measured_S_by_window"] == {"4096": 19, "49152": 19}
        memo = sp.render_memo(v1)
        assert "| admitted S by window (roles) | served-class S by window | standup S by window |" in memo
        assert '| {"4096": 20, "49152": 20} | {"4096": 19, "49152": 19} |' in memo
        # the STANDUP resolution of the class over this record REFUSES — production's launch_class_geometry resolves with no module
        # set (geometry.unsteered_site_set), which needs the record to certify the zero adapter's class `wo`, and this record certifies mlp
        # alone — recorded in geometry's words on the pair and in the memo, so a reader (and replan_compute) sees that no steered standup of
        # the pair is sized from this record as it stands, rather than a number that nothing would launch at
        st = a1["pairs"][0]["served_class"]["standup"]
        assert st["S_by_window"] is None and st["launches_by_window"] is None and st["source"] is None and st["batched_token_budget"] is None
        assert "does not certify 'wo'" in st["refused"] and "certified module sets ['mlp']" in st["refused"]
        assert v1["geometry_fold"][sp.STANDUP_S_KEY] == {"4:mlp": st} and "| refused: " in memo and "does not certify 'wo'" in memo
        monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(site))
        bl = dict(g.load_site_geometry(MODEL).sets["mlp"].budget_launches)
        for L in (4096, 49152):
            assert bl[f"mlp_r4_S2_cap8_L{L}"]["max_loras"] == 5 and bl[f"mlp_r4_single_cap8_L{L}"] == {"S": 19, "L": L, "batched_tokens": 19 * L, "max_lora_rank": 8, "max_loras": 4}
        ws, src, chosen = g.least_admitted_by_window(MODEL, bl, (8, 4))
        assert src == "launch_class:8x4" and chosen == {4096: ("mlp_r4_single_cap8_L4096",), 49152: ("mlp_r4_single_cap8_L49152",)} and ws == {4096: 19, 49152: 19}
        # the fold the old record would have left: S2 alone "dominates" (8, 4) and the class is sized from its lighter workload
        servers_only = {lid: b for lid, b in bl.items() if "_single_" not in lid}
        ws0, _src0, chosen0 = g.least_admitted_by_window(MODEL, servers_only, (8, 4))
        assert chosen0 == {4096: ("mlp_r4_S2_cap8_L4096",), 49152: ("mlp_r4_S2_cap8_L49152",)} and ws0 == {4096: 40, 49152: 40}
