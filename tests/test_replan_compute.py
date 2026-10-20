"""reproduce/tools/replan_compute.py on the memprobe record's REAL layout (schema 1: `rungs` + `verdict`).

The fixtures under tests/fixtures/replan/ are FD 23 records measured on one H200 (2026-09-30) and relabelled with a SYNTHETIC
model (`example_moe`): every identifier scrubbed, the numbers kept — the ReLU steer fits at micro-batch 1 only (49 s/step over
three cold steps = 167 tok/s), the LoRA r4 fits at micro-batch 2 (13.1 s/step = 1,251 tok/s). What is tested is the READER over
the layout and its arithmetic, never a fact about a model. What is pinned: the pace comes from the widest
FITTING rung, `--tok-s ARM=V` overrides it (the ReLU arm's warm pace is unmeasured), the fold's micro-batch is cross-checked
against the verdict, `--probe` is optional (the serving-derived rows are `pending`, excluded from the total, `complete` false),
the fit width divides the wall (GPU-hours unchanged) and must divide accum_steps, the legacy `microbatch` layout is refused by
name, and a does-not-fit verdict refuses.

The serving rows: the probe verdict the tool reads is PRODUCED HERE by the real emitter's writer
(`scripts/sdfbase/serving_probe.py write_verdict` over the probe smoke test's toy plan + launch-result builders), never a
hand-written dict — the multi-turn row from the certified pair's admitted S at 49,152 and its measured aggregate tok/s, the
padding row from the certified S at 98,304 with the pace transfer stated; a partial / uncertified verdict, a `replan_required`
set, an uncertified production pair, an unmeasured window and a missing throughput point each REFUSE (nothing written), so
`probe_status: present` is only ever a fully priced plan; `repro_require_compute_plan <model> complete` refuses the pending one.

The served class: the S a row is priced at is the pair's `served_class.standup`
— the standup resolution over the folded record (every certified set's launches, the least S among the tightest launches dominating
the class, inside the record's minimum budget // L: `geometry.unsteered_row_of`, what `launch_class_geometry` sizes a steered standup
to) — never the role servers' `admitted_max_num_seqs` and never the selected pair's own union alone (a down union at S 20 beside
an mlp union of the same class at S 9 stands the down pair up at 9); the pace is the union's measured aggregate tok/s at 49,152
rescaled by (standup S / measured S), never up. A standup S below 2 at a priced window, for a class of two or more adapters, REFUSES
before pricing: native_standup refuses that standup before launch — the multi-LoRA routing control needs two distinct selections
in one engine step — so a row priced at S 1 would price a standup that cannot come up.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures" / "replan"
for _p in (REPO / "reproduce" / "tools", REPO / "scripts" / "sdfbase", REPO / "scripts", REPO / "src", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import replan_compute as rc  # noqa: E402
import serving_probe as sp  # noqa: E402
from test_serving_probe_smoke import _fixture_results, _plan  # noqa: E402  (the serving probe's builders; the writer is the real one)

MODEL = "example_moe"
WINDOWS = (4096, 49152, 98304)
#: the EFFECTIVE BATCH is the FITS CONFIG's (tune_arm config_effective_batch's rule): an UNFOLDED config names it in its
#: accum sentinel (PENDING_MEMPROBE_EFF<n>_ACCUM), a folded one realises batch_size_by_arm x accum_steps. Two synthetic configs in the
#: generator's layout, differing in the batch alone — eb 64 (the demo lane's, unfolded) and eb 8 (the paper lanes', folded at the fixtures'
#: micro-batches 1 / 2) — so every derivation is pinned on both and nothing in the planner carries a batch of its own.
EB64_BATCH_KEYS = """batch_size_by_arm:
  psr_ce_all: MEMPROBE
  lora_r4_ce: MEMPROBE
"""
EB8_BATCH_KEYS = """batch_size_by_arm:
  psr_ce_all: 1
  lora_r4_ce: 2
"""
#: a SYNTHETIC fits config in the generator's layout: `fit_width: 4` (the planner's width source) and a lora_r4_ce fit_selection that is a
#: folded pick (rank 16 / mlp), NOT the toy verdict's (4, "wo") — the pricing tests name the toy pair explicitly (--serving-pair 4:wo) and the
#: planner's default read of the production pair from the config is exercised as the NEGATIVE case below (certified ≠ production → refuse)
FITS_CFG_TEXT = """model: example-org/example-moe-base
model_revision: 1111111111111111111111111111111111111111
fit_width: 4
si_ids:
- s
""" + EB64_BATCH_KEYS + """train_kwargs:
  default:
    heldout_pool: /vol/pool/example_moe_base/pairs_constv2-heldout.jsonl
  psr_ce_all:
    accum_steps: PENDING_MEMPROBE_EFF64_ACCUM
  lora_r4_ce:
    accum_steps: PENDING_MEMPROBE_EFF64_ACCUM
    fit_selection:
      lora_rank: 16
      lora_modules:
      - gate_proj
      - up_proj
      - down_proj
"""
EB8_FITS_CFG_TEXT = """model: example-org/example-moe-base
model_revision: 1111111111111111111111111111111111111111
fit_width: 4
si_ids:
- s
""" + EB8_BATCH_KEYS + """train_kwargs:
  default:
    heldout_pool: /vol/pool/example_moe_base/pairs_constv2-heldout.jsonl
  psr_ce_all:
    accum_steps: 8
  lora_r4_ce:
    accum_steps: 4
    fit_selection:
      lora_rank: 16
      lora_modules:
      - gate_proj
      - up_proj
      - down_proj
"""
CFG_TEXT_BY_EB = {64: FITS_CFG_TEXT, 8: EB8_FITS_CFG_TEXT}


@pytest.fixture(scope="module")
def fits_cfg(tmp_path_factory) -> str:
    p = tmp_path_factory.mktemp("replan_cfg") / "sdfbase_example_moe_base_fits.yaml"
    p.write_text(FITS_CFG_TEXT)
    return str(p)


def _cfg_at(tmp_path, eb: int) -> str:
    """The synthetic fits config at effective batch `eb`, written under the test's dir."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / f"sdfbase_example_moe_base_eb{eb}_fits.yaml"
    p.write_text(CFG_TEXT_BY_EB[eb])
    return str(p)


def _tp(points):
    """Throughput points as the driver records them (serving_probe._production_throughput): one per closed-loop B."""
    return [{"requested_concurrency": B, "engine_max_num_seqs": S, "server_tok_s": tok, "client_tok_s": tok * 0.97,
             "errors": 0, "throughput_source": "server_metrics", "elapsed_s": 30.0, "window_s": 30.0} for B, S, tok in points]


def _real_verdict(tmp_path, monkeypatch, *, S=(64, 5, 2), tok_s_49152=900.0, drop_throughput=False, drop_window=None,
                  oom_49152=False, no_reference=False, union_S=None, union_tok_s_49152=None, drop_union_throughput=False):
    """A verdict WRITTEN BY THE EMITTER over the toy plan (one pair, rank 4 / wo, the three windows the rows need): the
    certified S per window come from the engine, the throughput from the control-carrying natural launches at 4,096 and
    49,152 — the control's server S1cc AND the pair's union, exactly the launches the driver measures (``run_one_launch``:
    kind natural, a window in THROUGHPUT_WINDOWS, a manifest carrying the control role; the union carries it). ``union_S`` /
    ``union_tok_s_49152`` give the union its OWN capacity and pace (default: the servers'), ``drop_union_throughput`` leaves the
    union unmeasured while S1cc keeps its points."""
    from sisteer.capeval.serving import geometry as g
    for v in (g.SITE_GEOMETRY_DIR_ENV, g.SITE_DIR_ENV, g.SHARED_ROOT_ENV, "SISTEER_MODELS_TABLE"):
        monkeypatch.delenv(v, raising=False)
    windows = tuple(w for w in WINDOWS if w != drop_window)
    plan = _plan(sets=("wo",), ranks=(4,), windows=windows)
    # the split topology (the 2026-10-01 ruling): the pair's roles are certified per window by S1cc (control +
    # constitution + the carriers) AND S1d (the rank-2r difference) — both must come up; a start-time OOM lands on the
    # control's server (S1cc), the one the ladder keys on
    outcomes = {}
    for w, s_eng, s_union in zip(WINDOWS, S, union_S or S):
        if w not in windows:
            continue
        ref = {"reference": None} if no_reference else {}
        passed = ("passed", {"engine_S": s_eng, **ref})
        outcomes[f"wo_r4_S1cc_cap8_L{w}"] = ("oom",) if (oom_49152 and w == 49152) else passed
        outcomes[f"wo_r4_S1d_cap8_L{w}"] = passed
        # the pair's union (the served launch class, 2026-10-11) carries the same adapters in one engine: it did what the control
        # server did at that window, at ITS OWN engine's S — it certifies the class a steered standup serves (what the serving rows
        # price from), never a role
        outcomes[f"wo_r4_single_cap8_L{w}"] = ("oom",) if (oom_49152 and w == 49152) else ("passed", {"engine_S": s_union, **ref})
    results = _fixture_results(plan, outcomes=outcomes)
    if not drop_throughput:
        for w, tok in ((4096, 3000.0), (49152, tok_s_49152)):
            # the driver measures throughput on every control-carrying natural launch at the throughput windows: S1cc and the union
            tok_union = union_tok_s_49152 if (union_tok_s_49152 is not None and w == 49152) else tok
            for topo, tok_here in (("S1cc", tok), ("single", tok_union)):
                if topo == "single" and drop_union_throughput:
                    continue
                lid = f"wo_r4_{topo}_cap8_L{w}"
                if lid in results and results[lid]["status"] == "passed":
                    S_w = results[lid]["engine"]["max_num_seqs_engine"]
                    results[lid]["throughput"] = _tp([(32, S_w, tok_here), (64, S_w, tok_here * 1.02), (128, S_w, tok_here * 1.03)])
    out = tmp_path / "probe"
    out.mkdir(parents=True, exist_ok=True)
    sp.write_json_atomic(out / sp.PLAN_FILE, plan)          # the record dir as the probe leaves it: the plan beside the verdict
    verdict, _allowed = sp.write_verdict(plan, results, probe_job_id="4242", out=out, geometry_dir=tmp_path / "geom")
    return out / sp.VERDICT_FILE, verdict

PSR = str(FIX / "memprobe.psr_ce_all.json")
LORA = str(FIX / "memprobe.lora_r4_ce.json")
FOLD = str(FIX / "fold_memprobe.json")
LORA_TOK_S = 1250.9856157292897


def _run(tmp_path, *extra, model=MODEL, eb: int | None = 64):
    """The planner over the two fixture records; `--fits-config` is REQUIRED to price them (the effective batch is the config's), so the
    eb-64 synthetic config rides along unless the call names one (`--fits-config` in extra) or asks for none (eb=None)."""
    out = tmp_path / "compute_plan.json"
    cfg = [] if (eb is None or "--fits-config" in extra) else ["--fits-config", _cfg_at(tmp_path, eb)]
    rc_ = rc.main(["--model", model, "--memprobe", PSR, "--memprobe", LORA, "--out", str(out), *cfg, *extra])
    return rc_, (json.loads(out.read_text()) if out.exists() else None)


@pytest.mark.parametrize("eb", [64, 8])
def test_real_layout_is_read_and_the_plan_is_written_with_the_probe_pending(tmp_path, eb):
    rc_, plan = _run(tmp_path, "--fold", FOLD, "--fit-width", "4", "--tok-s", f"psr_ce_all={LORA_TOK_S}", eb=eb)
    assert rc_ == 0 and plan is not None
    assert plan["schema"] == 3 and plan["probe"] is None and plan["probe_status"] == "pending" and plan["complete"] is False
    arms = plan["arms"]
    # accum_steps = ceil(the FITS CONFIG's effective batch / micro-batch): 64 / 32 on the eb-64 config, 8 / 4 on the eb-8 one — the
    # config's batch rides each arm's record; nothing else in the plan moves with it
    assert arms["psr_ce_all"]["fits"] is True and arms["psr_ce_all"]["micro_batch"] == 1 and arms["psr_ce_all"]["accum_steps"] == eb // 1
    assert arms["lora_r4_ce"]["fits"] is True and arms["lora_r4_ce"]["micro_batch"] == 2 and arms["lora_r4_ce"]["accum_steps"] == eb // 2
    assert arms["psr_ce_all"]["effective_batch"] == arms["lora_r4_ce"]["effective_batch"] == eb
    assert plan["fits_config"].endswith(f"_eb{eb}_fits.yaml")
    # the pace: LoRA from its widest fitting rung (micro-batch 2), ReLU overridden with the cold bound stated beside it
    assert arms["lora_r4_ce"]["tokens_per_s"] == pytest.approx(LORA_TOK_S) and "micro-batch 2" in arms["lora_r4_ce"]["pace_source"]
    assert arms["psr_ce_all"]["pace_source"].startswith("override") and arms["psr_ce_all"]["measured_tokens_per_s"] == pytest.approx(167.18, abs=0.01)
    assert arms["psr_ce_all"]["s_per_step_at_min_micro_batch"] == pytest.approx(49.0, abs=0.01)
    fits = plan["rows"]["04_fits"]
    assert fits["fit_width"] == 4
    c = fits["cells"]["constitution_lora"]
    assert c["h_per_pass_1gpu"] == pytest.approx(9.5, abs=0.01) and c["gpu_h"] == pytest.approx(152.0, abs=0.1)
    assert c["wall_h"] == pytest.approx(152.0 / 4 + 5 / 60, abs=0.1)          # the width divides the wall, not the GPU-hours
    assert fits["critical_path_wall_h"] == pytest.approx(38.1, abs=0.1)
    assert fits["gpu_h"] == pytest.approx(2 * 152.0 + 2 * 28.5, abs=0.2)
    assert plan["rows"]["06_arms_multiturn"]["status"] == "pending" and plan["rows"]["06_arms_padding"]["status"] == "pending"
    assert set(plan["pending_rows"]) == {"06_arms_multiturn", "06_arms_padding"}
    assert plan["fit_replan_triggered"] is True and "fit_width=4" in plan["decision"] and "pending" in plan["decision"]
    # the total counts only the rows with a number
    assert plan["total_gpu_h"] == pytest.approx(sum(float(v["gpu_h"]) for v in plan["rows"].values() if v.get("gpu_h") is not None), abs=0.2)
    assert plan["rows"]["03_relu_trials"]["h_per_trial_1gpu"] == pytest.approx(3 * 1e7 / LORA_TOK_S / 3600 + 5 / 60, abs=0.01)


def test_without_the_override_the_relu_arm_plans_on_its_own_cold_bound(tmp_path):
    rc_, plan = _run(tmp_path, "--fit-width", "1")
    assert rc_ == 0
    relu = plan["arms"]["psr_ce_all"]
    assert relu["tokens_per_s"] == pytest.approx(167.18, abs=0.01) and "micro-batch 1" in relu["pace_source"]
    c = plan["rows"]["04_fits"]["cells"]["constitution_relu"]
    assert c["gpu_h"] == pytest.approx(1137.4, abs=0.2) and c["wall_h"] == pytest.approx(c["gpu_h"] + 5 / 60, abs=0.1)


def test_a_verdict_written_by_the_emitter_prices_both_serving_rows(tmp_path, monkeypatch, fits_cfg):
    pv_path, verdict = _real_verdict(tmp_path, monkeypatch)
    assert verdict["status"] == "complete" and verdict["certifying"] and verdict["replan_required"] == []
    # the real schema the tool reads: no top-level `windows` / `padding`; admission + throughput live here
    assert "windows" not in verdict and "padding" not in verdict
    pair = verdict["pairs"][0]
    assert pair["admitted_max_num_seqs"] == {"4096": 64, "49152": 5, "98304": 2} and sorted(pair["windows_measured"]) == list(WINDOWS)
    assert pair["topology"] == "S1ccS1d" and pair["caps_by_server"] == {"S1cc": 8, "S1d": 8}
    assert sorted(pair["certifying_by_window"]["49152"].values()) == ["wo_r4_S1cc_cap8_L49152", "wo_r4_S1d_cap8_L49152"]
    srv = next(s for s in verdict["servers"] if s["pair"] == [4, "wo"] and s["manifest"] == "wo_r4_S1cc")
    assert srv["admitted_by_window"]["49152"]["throughput"][0]["server_tok_s"] == 900.0
    # the SERVED CLASS (the pair's union, one engine over every adapter) carries its own S and pace per window on the pair record —
    # here equal to the servers', so every priced number below is the same as before the rows read the class (round 2)
    sc = pair["served_class"]
    assert (sc["max_lora_rank"], sc["max_loras"], sc["manifest"]) == (8, 8, "wo_r4_single")
    assert sc["certified_by_window"] == {str(w): f"wo_r4_single_cap8_L{w}" for w in WINDOWS}
    assert sc["measured_S_by_window"] == {"4096": 64, "49152": 5, "98304": 2}
    assert sc["throughput_by_window"]["49152"][0]["server_tok_s"] == 900.0 and sc["throughput_by_window"]["98304"] is None
    # ...and the STANDUP resolution of the class over the folded record: one certified set, so the all-set rule yields the
    # union's S at every window (the record's budget 64 x 4,096 buys 5 at 49,152 and 2 at 98,304 — the quotient never binds below the
    # union), the union launch named at each window, geometry's own source label
    assert sc["standup"] == {"S_by_window": {"4096": 64, "49152": 5, "98304": 2},
                             "launches_by_window": {str(w): [f"wo_r4_single_cap8_L{w}"] for w in WINDOWS},
                             "source": "launch_class:8x8", "batched_token_budget": 64 * 4096, "refused": None}
    assert verdict["geometry_fold"]["standup_S"] == {"4:wo": sc["standup"]}
    rc_, plan = _run(tmp_path, "--fits-config", fits_cfg, "--serving-pair", "4:wo", "--tok-s", f"psr_ce_all={LORA_TOK_S}", "--probe", str(pv_path))
    assert rc_ == 0 and plan["probe_status"] == "present" and plan["pending_rows"] == [] and plan["complete"] is True
    assert plan["fit_width"] == 4                                     # from the fits config
    mt = plan["rows"]["06_arms_multiturn"]
    assert mt["basis"] == "certified" and mt["pair"] == {"rank": 4, "set": "wo", "lora_modules": ["o_proj"],
                                                          "topology": pair["topology"], "certified_by": pair["certified_by"]}
    assert mt["admitted_S"] == 5 and mt["window"] == 49152
    # the S priced is the standup's, the pace was measured at that same S: no transfer on the multi-turn row
    assert mt["standup_S_launches"] == ["wo_r4_single_cap8_L49152"] and mt["standup_S_source"].startswith("launch_class:8x8 over the record's certified sets")
    assert mt["measured_S_at_pace_window"] == 5 and mt["s_ratio_scale"] == 1.0 and mt["tok_s_measured"] == 900.0 and "no transfer" in mt["assumption"]
    # the saturating point: the smallest requested concurrency >= S (32 at S 5) -> 900 tok/s, server counter — read on the served
    # class's launch at 49,152, named on the row beside the roles' provenance
    assert mt["tok_s"] == 900.0 and mt["throughput_point"]["requested_concurrency"] == 32 and mt["throughput_point"]["source"] == "server_tok_s"
    assert mt["throughput_point"]["manifest"] == "wo_r4_single" and mt["throughput_point"]["launch"] == "wo_r4_single_cap8_L49152"
    assert mt["served_class"] == {"max_lora_rank": 8, "max_loras": 8, "target_class": ["o_proj"], "manifest": "wo_r4_single",
                                  "launch": "wo_r4_single_cap8_L49152"}
    assert plan["rows"]["06_arms_padding"]["served_class"] == dict(mt["served_class"], launch="wo_r4_single_cap8_L98304",
                                                                   launch_at_pace_window="wo_r4_single_cap8_L49152")
    assert plan["probe_verdict"]["served_class"]["measured_S_by_window"] == {"4096": 64, "49152": 5, "98304": 2}
    assert plan["probe_verdict"]["served_class"]["standup"] == sc["standup"]                           # what the rows were priced at
    assert plan["probe_verdict"]["admitted_max_num_seqs"] == {"4096": 64, "49152": 5, "98304": 2}       # the roles', kept as a record
    assert mt["h_per_attack_arm"] == pytest.approx(259 * 10 * 1500 / 900 / 3600, abs=0.01)
    assert mt["gpu_h"] == pytest.approx(3 * 10 * 259 * 10 * 1500 / 900 / 3600, abs=0.1)
    pad = plan["rows"]["06_arms_padding"]
    assert pad["admitted_S"] == 2 and pad["admitted_S_at_pace_window"] == 5 and pad["s_ratio_scale"] == pytest.approx(0.4)
    assert pad["measured_S_at_pace_window"] == 5 and pad["measured_S"] == 2 and pad["standup_S_launches"] == ["wo_r4_single_cap8_L98304"]
    assert pad["tok_s_assumed"] == pytest.approx(360.0) and pad["levels"] == [0, 8192, 16384, 32768, 65536]
    assert pad["cap"] == 16384 and pad["reserve"] == 8192 and "make_padding_configs" in pad["constants_source"]
    exp = sum(10 * 2590 * (lvl + 8192 + 16384) / 360.0 / 3600 for lvl in (0, 8192, 16384, 32768, 65536))
    assert pad["gpu_h"] == pytest.approx(exp, abs=0.2) and "assumption" in pad
    assert plan["probe_verdict"]["probe_job_id"] == "4242" and plan["probe_verdict"]["replan_required"] == []
    assert "batched_token_budget" in plan["probe_verdict"]
    # the total now counts both rows
    assert plan["total_gpu_h"] == pytest.approx(sum(float(v["gpu_h"]) for v in plan["rows"].values()), abs=0.3)


@pytest.mark.parametrize("union_S, union_tok", [((60, 4, 3), 800.0),      # the one engine admits LESS where the pace is measured and runs SLOWER
                                                 ((64, 8, 3), 1200.0)])   # ...or MORE and FASTER: the class's numbers either way, never a min
# (the LESS case admits 3, not 1, at 98,304: an 8-adapter standup at S 1 REFUSES before launch — the planner's gate refuses it too —
#  and S 2 would equal the servers' there, so 3 is the least width that both launches and differs)
def test_the_serving_rows_price_the_served_class_never_the_role_servers(tmp_path, monkeypatch, fits_cfg, union_S, union_tok):
    """Round-2 finding on the served-class work: the split servers S1cc / S1d certify the ROLES on two engines, each lighter than the one
    engine a steered standup runs (every adapter of the pair in one class); `geometry.least_admitted_by_window` sizes that standup from
    the union's launch (the tightest dominating footprint), so a row priced from the servers' `admitted_max_num_seqs` and pace priced an
    engine nobody serves. Servers at S (64, 5, 2) / 900 tok/s; the union at its own S and pace: the rows read the union's S at both
    windows and its pace at 49,152, the roles' provenance stays on the row, and the pair record keeps the roles' S beside the class's."""
    pv_path, verdict = _real_verdict(tmp_path, monkeypatch, union_S=union_S, union_tok_s_49152=union_tok)
    pair = verdict["pairs"][0]
    assert verdict["status"] == "complete" and verdict["certifying"] and [(p["rank"], p["set"]) for p in verdict["pairs"]] == [(4, "wo")]
    # the role provenance is the servers', field for field, with the servers' S — unchanged by the union's numbers
    assert pair["admitted_max_num_seqs"] == {"4096": 64, "49152": 5, "98304": 2} and pair["topology"] == "S1ccS1d"
    assert pair["certified_by"] == [lid for w in WINDOWS for lid in (f"wo_r4_S1cc_cap8_L{w}", f"wo_r4_S1d_cap8_L{w}")]
    assert pair["certifying_by_window"] == {str(w): {"control": f"wo_r4_S1cc_cap8_L{w}", "difference": f"wo_r4_S1d_cap8_L{w}"} for w in WINDOWS}
    # the served class carries ITS launch's S and pace per window
    sc = pair["served_class"]
    assert sc["measured_S_by_window"] == {str(w): s for w, s in zip(WINDOWS, union_S)}
    assert sc["throughput_by_window"]["49152"][0]["server_tok_s"] == union_tok and sc["throughput_by_window"]["4096"][0]["server_tok_s"] == 3000.0
    srv_cc = next(s for s in verdict["servers"] if s["pair"] == [4, "wo"] and s["manifest"] == "wo_r4_S1cc")
    assert srv_cc["admitted_by_window"]["49152"]["throughput"][0]["server_tok_s"] == 900.0           # the servers' pace is still on the memo
    rc_, plan = _run(tmp_path, "--fits-config", fits_cfg, "--serving-pair", "4:wo", "--tok-s", f"psr_ce_all={LORA_TOK_S}", "--probe", str(pv_path))
    assert rc_ == 0 and plan["probe_status"] == "present" and plan["complete"] is True
    S_mt, S_pad = union_S[1], union_S[2]
    # one certified set: the standup resolution (every set's launches, the tightest dominating the class, inside the record's minimum
    # budget // L) is the union's own S at every window — the record's budget is the largest S x L of the set's launches, the union's
    # own in both cases (8 x 49,152 = 393,216 buys 8 / 4 at 49,152 / 98,304, above the union's 3; 3 x 98,304 = 294,912 buys 72 / 6 / 3,
    # at or above the union's 60 / 4 / 3) — a union's S x L never exceeds the record's largest product; so the priced S is the union's,
    # not the split servers' 5 / 2
    assert sc["standup"]["S_by_window"] == {str(w): s for w, s in zip(WINDOWS, union_S)}
    assert sc["standup"]["launches_by_window"] == {str(w): [f"wo_r4_single_cap8_L{w}"] for w in WINDOWS} and sc["standup"]["refused"] is None
    assert sc["standup"]["batched_token_budget"] == max(64 * 4096, max(s * w for w, s in zip(WINDOWS, union_S)))
    mt, pad = plan["rows"]["06_arms_multiturn"], plan["rows"]["06_arms_padding"]
    assert mt["admitted_S"] == S_mt and mt["tok_s"] == union_tok and mt["throughput_point"]["launch"] == "wo_r4_single_cap8_L49152"
    assert mt["measured_S_at_pace_window"] == S_mt and mt["s_ratio_scale"] == 1.0 and mt["tok_s_measured"] == union_tok   # measured at the standup's S
    assert mt["standup_S_launches"] == ["wo_r4_single_cap8_L49152"]
    assert mt["throughput_point"]["manifest"] == "wo_r4_single" and mt["throughput_point"]["requested_concurrency"] == 32
    assert mt["gpu_h"] == pytest.approx(3 * 10 * 259 * 10 * 1500 / union_tok / 3600, abs=0.1)
    assert mt["pair"]["certified_by"] == pair["certified_by"] and mt["served_class"]["launch"] == "wo_r4_single_cap8_L49152"
    assert f"S {S_mt}" in mt["note"] and f"{union_tok:.0f} tok/s" in mt["note"] and "wo_r4_single_cap8_L49152" in mt["note"]
    scale = min(1.0, S_pad / S_mt)
    assert pad["admitted_S"] == S_pad and pad["admitted_S_at_pace_window"] == S_mt and pad["s_ratio_scale"] == pytest.approx(round(scale, 3))
    assert pad["measured_S_at_pace_window"] == S_mt and pad["measured_S"] == S_pad and pad["standup_S_launches"] == ["wo_r4_single_cap8_L98304"]
    assert pad["tok_s_assumed"] == pytest.approx(round(union_tok * scale, 1)) and pad["served_class"]["launch"] == "wo_r4_single_cap8_L98304"
    exp = sum(10 * 2590 * (lvl + 8192 + 16384) / (union_tok * scale) / 3600 for lvl in (0, 8192, 16384, 32768, 65536))
    assert pad["gpu_h"] == pytest.approx(exp, abs=0.2)
    # nothing priced reads the servers' 5 / 2 / 900 when the union differs: the split S is on the record of the roles alone
    if union_S[1] != 5:
        assert mt["admitted_S"] != 5 and mt["measured_S_at_pace_window"] != 5 and pad["admitted_S_at_pace_window"] != 5
    if union_S[2] != 2:
        assert pad["admitted_S"] != 2
    assert mt["tok_s"] != 900.0 and pad["tok_s_at_pace_window"] != 900.0
    assert plan["probe_verdict"]["served_class"]["measured_S_by_window"] == sc["measured_S_by_window"]
    assert plan["probe_verdict"]["served_class"]["standup"] == sc["standup"]
    assert plan["probe_verdict"]["admitted_max_num_seqs"] == {"4096": 64, "49152": 5, "98304": 2}


def test_a_served_class_the_verdict_cannot_price_refuses_by_name(tmp_path, monkeypatch, capsys, fits_cfg):
    """The rows read the served class or refuse: a union the driver never measured at 49,152 (the servers' points do not stand in), a
    pair record with no `served_class` block (a verdict written before the class was certified), a class certified at 49,152 but not at
    the padding window, a class whose launch admitted S < 1, a standup resolution sized at S 1 for a class of 8 adapters (the standup's
    own pre-launch refusal) and a served class with no adapter count — each refused by name, rc 3, nothing written."""
    common = ["--fits-config", fits_cfg, "--serving-pair", "4:wo", "--tok-s", f"psr_ce_all={LORA_TOK_S}"]
    # the union carries no throughput at 49,152 while S1cc does: the servers' pace never prices the class
    pv, v = _real_verdict(tmp_path / "a", monkeypatch, drop_union_throughput=True)
    assert v["status"] == "complete" and v["pairs"][0]["served_class"]["throughput_by_window"]["49152"] is None
    srv_cc = next(s for s in v["servers"] if s["manifest"] == "wo_r4_S1cc")
    assert srv_cc["admitted_by_window"]["49152"]["throughput"]                                 # measured on the control's server...
    assert _run(tmp_path / "a", *common, "--probe", str(pv))[0] == 3
    err = capsys.readouterr().err
    assert "no measured throughput for the served class" in err and "wo_r4_single_cap8_L49152" in err and "window 49152" in err
    assert not (tmp_path / "a" / "compute_plan.json").exists()
    # a pair record without the block: the verdict predates the served-class certification
    pv, v = _real_verdict(tmp_path / "b", monkeypatch)
    doc = json.loads(pv.read_text()); del doc["pairs"][0]["served_class"]; pv.write_text(json.dumps(doc))
    assert _run(tmp_path / "b", *common, "--probe", str(pv))[0] == 3
    err = capsys.readouterr().err
    assert "no `served_class` block" in err and "re-run `serving_probe.py verdict`" in err
    # the class not certified at the padding window (its launch there is not the certifying one)
    pv, v = _real_verdict(tmp_path / "c", monkeypatch)
    doc = json.loads(pv.read_text())
    doc["pairs"][0]["served_class"]["certified_by_window"]["98304"] = None
    doc["pairs"][0]["served_class"]["measured_S_by_window"]["98304"] = None
    pv.write_text(json.dumps(doc))
    assert _run(tmp_path / "c", *common, "--probe", str(pv))[0] == 3
    err = capsys.readouterr().err
    assert "the served class (rank cap 8, 8 adapters) at the padding window 98304 was not MEASURED" in err
    # a class whose certifying launch admitted nothing at 49,152 (a hand-edited record; the verdict itself excludes such a pair)
    pv, v = _real_verdict(tmp_path / "d", monkeypatch)
    doc = json.loads(pv.read_text()); doc["pairs"][0]["served_class"]["measured_S_by_window"]["49152"] = 0; pv.write_text(json.dumps(doc))
    assert _run(tmp_path / "d", *common, "--probe", str(pv))[0] == 3
    assert "admitted S 0 < 1 at window 49152" in capsys.readouterr().err
    # the standup resolution is what the rows are priced at — a verdict without it (written before the resolution rode the pair
    # record) refuses by name, the union's own S never standing in
    pv, v = _real_verdict(tmp_path / "e", monkeypatch)
    doc = json.loads(pv.read_text()); del doc["pairs"][0]["served_class"]["standup"]; pv.write_text(json.dumps(doc))
    assert _run(tmp_path / "e", *common, "--probe", str(pv))[0] == 3
    err = capsys.readouterr().err
    assert "no `served_class.standup` block" in err and "re-run `serving_probe.py verdict`" in err and "never one set's union alone" in err
    assert not (tmp_path / "e" / "compute_plan.json").exists()
    # a class the record cannot size (the standup would refuse at launch-class geometry): refused with geometry's words
    pv, v = _real_verdict(tmp_path / "f", monkeypatch)
    doc = json.loads(pv.read_text())
    doc["pairs"][0]["served_class"]["standup"] = {"S_by_window": None, "launches_by_window": None, "source": None, "batched_token_budget": None,
                                                  "refused": "qwen35_4b_demo: no probed launch at window 49152 dominates the served adapter class"}
    pv.write_text(json.dumps(doc))
    assert _run(tmp_path / "f", *common, "--probe", str(pv))[0] == 3
    err = capsys.readouterr().err
    assert "would REFUSE at launch-class geometry" in err and "no probed launch at window 49152 dominates" in err
    # a resolution that sizes a window at nothing, and one that lacks the padding window
    pv, v = _real_verdict(tmp_path / "g", monkeypatch)
    doc = json.loads(pv.read_text()); doc["pairs"][0]["served_class"]["standup"]["S_by_window"]["49152"] = 0; pv.write_text(json.dumps(doc))
    assert _run(tmp_path / "g", *common, "--probe", str(pv))[0] == 3 and "sizes window 49152 at S 0 < 1" in capsys.readouterr().err
    pv, v = _real_verdict(tmp_path / "h", monkeypatch)
    doc = json.loads(pv.read_text()); del doc["pairs"][0]["served_class"]["standup"]["S_by_window"]["98304"]; pv.write_text(json.dumps(doc))
    assert _run(tmp_path / "h", *common, "--probe", str(pv))[0] == 3 and "carries no S at the padding window 98304" in capsys.readouterr().err
    # a resolution that sizes the multi-turn window at ONE sequence for a class of 8 adapters: the standup refuses that width before launch
    # (native_standup's `max_num_seqs < 2` with two or more adapters), so the planner refuses the row by the same rule — S 1 is not S 0's
    # case (the gate read S < 1 only and priced the row)
    pv, v = _real_verdict(tmp_path / "i", monkeypatch)
    doc = json.loads(pv.read_text()); doc["pairs"][0]["served_class"]["standup"]["S_by_window"]["49152"] = 1; pv.write_text(json.dumps(doc))
    assert _run(tmp_path / "i", *common, "--probe", str(pv))[0] == 3
    err = capsys.readouterr().err
    assert "sizes the multi-turn window 49152 at S 1 with 8 adapters" in err and "REFUSES before launch" in err and "selection_honored" in err
    assert "max_num_seqs=1 with 8 adapters" in err and not (tmp_path / "i" / "compute_plan.json").exists()
    # ...and a served class with no adapter count cannot be gated: refused by name, never priced
    pv, v = _real_verdict(tmp_path / "j", monkeypatch)
    doc = json.loads(pv.read_text()); del doc["pairs"][0]["served_class"]["max_loras"]; pv.write_text(json.dumps(doc))
    assert _run(tmp_path / "j", *common, "--probe", str(pv))[0] == 3 and "carries no adapter count (`max_loras`)" in capsys.readouterr().err


def _three_set_verdict(tmp_path, monkeypatch, *, S_overrides=None):
    """A verdict WRITTEN BY THE EMITTER over a THREE-set plan (wo / down / mlp, rank 4, the three windows the rows need) in which two
    sets' unions share one launch class — down_r4_single and mlp_r4_single are both (rank cap 8, 4 adapters) — with UNEQUAL capacities:
    at 49,152 the down union admits 20 and the mlp union 9 (at 98,304: 10 and 4). The record's minimum batched-token budget is wo's
    (20 x 49,152 = 983,040; the shared S2 at 40 x 49,152 lifts down's and mlp's to 1,966,080), whose quotients — 240 / 20 / 10 — bind
    nowhere below the unions, so the per-window S is the launch-class rule's alone. Every launch passed; the pace is measured on the
    control-carrying natural launches at 4,096 and 49,152 — the down union at 1,000 tok/s on its engine admitting 20, the mlp union at 500.
    ``S_overrides`` replaces entries of the engine-S table by the same ``(set | '<set>_union' | 'S2', window)`` keys — the S=1 regression
    starves the mlp union at one priced window and leaves every other launch as above."""
    from sisteer.capeval.serving import geometry as g
    for v in (g.SITE_GEOMETRY_DIR_ENV, g.SITE_DIR_ENV, g.SHARED_ROOT_ENV, "SISTEER_MODELS_TABLE"):
        monkeypatch.delenv(v, raising=False)
    plan = _plan(sets=("wo", "down", "mlp"), ranks=(4,), windows=WINDOWS, ladder_caps=())
    S_at = {  # (set, topology, window) -> engine S; the S2 (shared by down and mlp) under its own key
        ("wo", 49152): 20, ("wo", 98304): 10,
        ("down", 49152): 24, ("down", 98304): 12, ("down_union", 49152): 20, ("down_union", 98304): 10,
        ("mlp", 49152): 12, ("mlp", 98304): 6, ("mlp_union", 4096): 120, ("mlp_union", 49152): 9, ("mlp_union", 98304): 4,
        ("S2", 49152): 40, ("S2", 98304): 20}
    S_at.update(S_overrides or {})
    outcomes = {}
    for r in plan["launches"]:
        key = ("S2" if r["topology"] == "S2" else f"{r['set']}_union" if r["topology"] == "single" else r["set"], int(r["L"]))
        s_eng = S_at.get(key, S_at.get((r["set"], int(r["L"])), 64))
        outcomes[r["id"]] = ("passed", {"engine_S": s_eng})
    results = _fixture_results(plan, outcomes=outcomes)
    pace = {"down_r4_single": 1000.0, "mlp_r4_single": 500.0, "wo_r4_single": 700.0, "down_r4_S1cc": 1100.0, "mlp_r4_S1cc": 600.0, "wo_r4_S1cc": 750.0}
    for lid, res in results.items():
        rec = next(r for r in plan["launches"] if r["id"] == lid)
        if rec["manifest"] in pace and int(rec["L"]) in (4096, 49152):
            tok = 3000.0 if int(rec["L"]) == 4096 else pace[rec["manifest"]]
            S_w = res["engine"]["max_num_seqs_engine"]
            res["throughput"] = _tp([(32, S_w, tok), (64, S_w, tok * 1.02), (128, S_w, tok * 1.03)])
    out = tmp_path / "probe"
    out.mkdir(parents=True, exist_ok=True)
    sp.write_json_atomic(out / sp.PLAN_FILE, plan)
    verdict, _allowed = sp.write_verdict(plan, results, probe_job_id="4343", out=out, geometry_dir=tmp_path / "geom")
    return out / sp.VERDICT_FILE, verdict


def test_two_certified_sets_of_one_class_price_the_least_served_concurrency_the_standup_gets(tmp_path, monkeypatch):
    """Production's `native_standup.launch_class_geometry` resolves WITHOUT a module set —
`geometry.unsteered_site_set` merges every certified set's folded launches and `least_admitted_by_window` takes the MINIMUM S among
the tightest launches dominating the served class — so a down union at S 20 beside an mlp union of the same (8, 4) class at S 9
stands the down pair up at 9. The rows priced the selected pair's own union (20); now they price the standup's S (9 at 49,152,
4 at 98,304), the pace measured at 20 rescaled by 9/20 (never up), and the verdict carries the resolution by geometry's own
function — equal to what the resolver hands a steered (8, 4) standup over the staged record."""
    from sisteer.capeval.serving import geometry as g
    pv_path, verdict = _three_set_verdict(tmp_path, monkeypatch)
    assert verdict["status"] == "complete" and verdict["certifying"] and verdict["geometry_fold"]["status"] == "written"
    pairs = {(p["rank"], p["set"]): p for p in verdict["pairs"]}
    assert set(pairs) == {(4, "wo"), (4, "down"), (4, "mlp")}
    down, mlp, wo = pairs[(4, "down")], pairs[(4, "mlp")], pairs[(4, "wo")]
    # each pair's served class measured on ITS union: down 20 / 10, mlp 9 / 4 (the roles' S beside them: 24 / 12 and 12 / 6)
    assert down["served_class"]["measured_S_by_window"] == {"4096": 64, "49152": 20, "98304": 10} and down["admitted_max_num_seqs"] == {"4096": 64, "49152": 24, "98304": 12}
    assert mlp["served_class"]["measured_S_by_window"] == {"4096": 120, "49152": 9, "98304": 4} and mlp["admitted_max_num_seqs"] == {"4096": 64, "49152": 12, "98304": 6}
    assert down["served_class"]["throughput_by_window"]["49152"][0] == {**down["served_class"]["throughput_by_window"]["49152"][0], "server_tok_s": 1000.0, "engine_max_num_seqs": 20}
    # the STANDUP resolution of the down pair's (8, 4) class: the mlp union — same class, lower S — sets 49,152 and 98,304; at 4,096 the
    # down union's 64 is below the mlp union's 120; the record's minimum budget is wo's 983,040 (its quotients 240 / 20 / 10 bind nowhere)
    st = down["served_class"]["standup"]
    assert st == {"S_by_window": {"4096": 64, "49152": 9, "98304": 4},
                  "launches_by_window": {"4096": ["down_r4_single_cap8_L4096"], "49152": ["mlp_r4_single_cap8_L49152"], "98304": ["mlp_r4_single_cap8_L98304"]},
                  "source": "launch_class:8x4", "batched_token_budget": 20 * 49152, "refused": None}
    assert mlp["served_class"]["standup"]["S_by_window"] == {"4096": 64, "49152": 9, "98304": 4}            # the same class: the same standup
    # the wo pair's (8, 8) class is dominated by its union alone: 20 / 10, its own measured S (the (8, 4) unions do not dominate it)
    assert wo["served_class"]["standup"]["S_by_window"] == {"4096": 64, "49152": 20, "98304": 10}
    assert wo["served_class"]["standup"]["launches_by_window"]["49152"] == ["wo_r4_single_cap8_L49152"] and wo["served_class"]["measured_S_by_window"]["49152"] == 20
    assert verdict["geometry_fold"]["standup_S"] == {"4:down": st, "4:mlp": mlp["served_class"]["standup"], "4:wo": wo["served_class"]["standup"]}
    # ...and it IS what the resolver hands a steered standup over the staged record: budget_for with no module set, by served class
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(tmp_path / "geom"))
    probe_model = verdict["model_key"]                     # the probe's (the smoke lane's onboarded key), not the planner's synthetic --model
    for w in WINDOWS:
        rb = g.budget_for(probe_model, served_lora=(8, 4))
        assert rb.entry.seqs_at(w) == st["S_by_window"][str(w)] and list(rb.entry.window_seqs_launches[w]) == st["launches_by_window"][str(w)]
        assert g.budget_for(probe_model, served_lora=(8, 8)).entry.seqs_at(w) == wo["served_class"]["standup"]["S_by_window"][str(w)]
    monkeypatch.delenv(g.SITE_GEOMETRY_DIR_ENV)
    # the rows for the down pair price 9 and 4, the pace measured at 20 rescaled (1,000 x 9/20 = 450; 1,000 x 4/20 = 200)
    rc_, plan = _run(tmp_path, "--serving-pair", "4:down", "--tok-s", f"psr_ce_all={LORA_TOK_S}", "--probe", str(pv_path))
    assert rc_ == 0 and plan["probe_status"] == "present" and plan["complete"] is True
    mt, pad = plan["rows"]["06_arms_multiturn"], plan["rows"]["06_arms_padding"]
    assert mt["pair"]["set"] == "down" and mt["served_class"]["launch"] == "down_r4_single_cap8_L49152"
    assert mt["admitted_S"] == 9 and mt["measured_S_at_pace_window"] == 20 and mt["s_ratio_scale"] == pytest.approx(0.45)
    assert mt["tok_s_measured"] == 1000.0 and mt["tok_s"] == pytest.approx(450.0) and mt["throughput_point"]["requested_concurrency"] == 32
    assert mt["standup_S_launches"] == ["mlp_r4_single_cap8_L49152"] and mt["standup_S_source"].startswith("launch_class:8x4 over the record's certified sets")
    assert mt["gpu_h"] == pytest.approx(3 * 10 * 259 * 10 * 1500 / 450.0 / 3600, abs=0.1)
    assert "scaled by 9/20" in mt["assumption"] and "S 9" in mt["note"] and "mlp_r4_single_cap8_L49152" in mt["note"]
    assert pad["admitted_S"] == 4 and pad["admitted_S_at_pace_window"] == 9 and pad["measured_S_at_pace_window"] == 20 and pad["measured_S"] == 10
    assert pad["s_ratio_scale"] == pytest.approx(0.2) and pad["tok_s_assumed"] == pytest.approx(200.0) and pad["tok_s_at_pace_window"] == 1000.0
    assert pad["standup_S_launches"] == ["mlp_r4_single_cap8_L98304"] and "scaled by 4/20" in pad["assumption"]
    exp = sum(10 * 2590 * (lvl + 8192 + 16384) / 200.0 / 3600 for lvl in (0, 8192, 16384, 32768, 65536))
    assert pad["gpu_h"] == pytest.approx(exp, abs=0.2)
    # nothing prices the selected pair's own union alone (20 / 10 / 1,000) nor its roles (24 / 12)
    assert 20 not in (mt["admitted_S"], pad["admitted_S"], pad["admitted_S_at_pace_window"]) and pad["admitted_S"] != 10
    assert mt["tok_s"] != 1000.0 and pad["tok_s_assumed"] != pytest.approx(500.0) and 24 not in (mt["admitted_S"], pad["admitted_S_at_pace_window"])
    assert plan["probe_verdict"]["served_class"]["standup"] == st and plan["probe_verdict"]["admitted_max_num_seqs"] == down["admitted_max_num_seqs"]
    # the mlp pair prices the same S (its own union is the binding one) at its own pace, unscaled: measured at 9, stood up at 9
    rc2, plan2 = _run(tmp_path / "mlp", "--serving-pair", "4:mlp", "--tok-s", f"psr_ce_all={LORA_TOK_S}", "--probe", str(pv_path))
    mt2 = plan2["rows"]["06_arms_multiturn"]
    assert rc2 == 0 and mt2["admitted_S"] == 9 and mt2["measured_S_at_pace_window"] == 9 and mt2["s_ratio_scale"] == 1.0 and mt2["tok_s"] == 500.0
    assert plan2["rows"]["06_arms_padding"]["admitted_S"] == 4 and plan2["rows"]["06_arms_padding"]["s_ratio_scale"] == pytest.approx(round(4 / 9, 3))


@pytest.mark.parametrize("window", [49152, 98304])
def test_a_standup_sized_at_one_sequence_for_a_multi_adapter_class_refuses_before_pricing(tmp_path, monkeypatch, capsys, window):
    """Round-1 finding on the standup pricing: native_standup refuses a steered standup of two or more adapters at `max_num_seqs < 2`
    BEFORE launch, right after launch_class_geometry (the multi-LoRA routing control, selection_honored, needs two distinct selections
    in ONE engine step and cannot pass at one sequence per step) — so when a certified launch of the class admits ONE sequence at a priced
    window, the all-set resolution sizes every pair of that class at S 1 there and the standup the rows would price cannot come up. The
    planner's gate read S < 1 only: the rows were priced at S 1 and a serve-only plan read `complete`. Now: the VERDICT is untouched
    (admission is geometry's — the mlp union that PASSED at S 1 certifies the class there, the resolution hands 1, equal to what the
    resolver hands a steered (8, 4) standup over the staged record, and the memo's standup column names the refusal beside the S), and
    the planner REFUSES by name — the window, S 1, the adapter count, the launch that set it, the standup's own words — for EVERY pair of
    the class, rc 3, nothing written, through `--probe` and `--serve-only` alike; the wo pair's (8, 8) class, dominated by its own union
    alone, still prices at its own S. Parametrized over both priced windows: the multi-turn row's and the padding row's."""
    from sisteer.capeval.serving import geometry as g
    other = {49152: 98304, 98304: 49152}[window]
    pv_path, verdict = _three_set_verdict(tmp_path, monkeypatch, S_overrides={("mlp_union", window): 1})
    assert verdict["status"] == "complete" and verdict["certifying"] and verdict["geometry_fold"]["status"] == "written"
    pairs = {(p["rank"], p["set"]): p for p in verdict["pairs"]}
    assert set(pairs) == {(4, "wo"), (4, "down"), (4, "mlp")} and verdict["excluded"] == []      # a launch that PASSED at S 1 certifies
    down, mlp, wo = pairs[(4, "down")], pairs[(4, "mlp")], pairs[(4, "wo")]
    assert mlp["served_class"]["measured_S_by_window"][str(window)] == 1
    assert down["served_class"]["measured_S_by_window"][str(window)] == {49152: 20, 98304: 10}[window]     # the down union itself admits more
    for p in (down, mlp):
        st = p["served_class"]["standup"]
        assert p["served_class"]["max_loras"] == 4 and st["refused"] is None and st["source"] == "launch_class:8x4"
        assert st["S_by_window"][str(window)] == 1 and st["launches_by_window"][str(window)] == [f"mlp_r4_single_cap8_L{window}"]
        assert st["S_by_window"][str(other)] == {49152: 9, 98304: 4}[other]                      # the other priced window as before
    assert wo["served_class"]["standup"]["S_by_window"] == {"4096": 64, "49152": 20, "98304": 10}
    # admission is untouched: the resolver hands a steered (8, 4) standup the same S 1 at the window over the staged record
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(tmp_path / "geom"))
    assert g.budget_for(verdict["model_key"], served_lora=(8, 4)).entry.seqs_at(window) == 1
    assert g.budget_for(verdict["model_key"], served_lora=(8, 8)).entry.seqs_at(window) == {49152: 20, 98304: 10}[window]
    monkeypatch.delenv(g.SITE_GEOMETRY_DIR_ENV)
    # the memo's standup column names what the standup does at that width, beside the S
    memo = sp.render_memo(verdict)
    assert f"a standup of 4 adapters REFUSES S < 2 at L={window} (native_standup" in memo and "two distinct selections in one engine step" in memo
    assert memo.count("REFUSES S < 2") == 2                                                  # the down row and the mlp row; never the wo row
    # the planner: every pair of the class refuses by name, nothing written — the row's window, S 1, the adapter count, the standup's rule
    what = {49152: "multi-turn", 98304: "padding"}[window]
    for set_id in ("down", "mlp"):
        rc_, plan = _run(tmp_path / set_id, "--serving-pair", f"4:{set_id}", "--tok-s", f"psr_ce_all={LORA_TOK_S}", "--probe", str(pv_path))
        err = capsys.readouterr().err
        assert rc_ == 3 and plan is None, (set_id, err)
        assert f"pair (rank 4, {set_id})" in err and f"sizes the {what} window {window} at S 1 with 4 adapters" in err
        assert f"mlp_r4_single_cap8_L{window}" in err and "REFUSES before launch" in err and "max_num_seqs=1 with 4 adapters" in err
        assert "selection_honored" in err and "cannot come up" in err
    # ...and a serve-only plan is never `complete` on it
    rc_so, plan_so = _serve_only(tmp_path / "so", "--probe", str(pv_path), "--serving-pair", "4:down")
    err = capsys.readouterr().err
    assert rc_so == 3 and plan_so is None and f"sizes the {what} window {window} at S 1 with 4 adapters" in err
    assert not (tmp_path / "so" / "compute_plan.json").exists()
    # the wo pair's class is sized by its own union alone (20 / 10): it still prices, at its own S, the pace unscaled
    rc_wo, plan_wo = _run(tmp_path / "wo", "--serving-pair", "4:wo", "--tok-s", f"psr_ce_all={LORA_TOK_S}", "--probe", str(pv_path))
    assert rc_wo == 0 and plan_wo["complete"] is True and plan_wo["probe_status"] == "present"
    assert plan_wo["rows"]["06_arms_multiturn"]["admitted_S"] == 20 and plan_wo["rows"]["06_arms_padding"]["admitted_S"] == 10
    assert plan_wo["rows"]["06_arms_multiturn"]["s_ratio_scale"] == 1.0 and plan_wo["rows"]["06_arms_multiturn"]["tok_s"] == 700.0


def test_a_verdict_that_cannot_price_a_row_refuses_instead_of_reporting_present(tmp_path, monkeypatch, capsys, fits_cfg):
    common = ["--fits-config", fits_cfg, "--serving-pair", "4:wo", "--tok-s", f"psr_ce_all={LORA_TOK_S}"]
    # partial: the 49,152 natural launch OOMed -> the pair is excluded / undecided, the verdict not complete
    pv, v = _real_verdict(tmp_path / "a", monkeypatch, oom_49152=True)
    assert v["status"] != "complete" or not v["pairs"]
    assert _run(tmp_path / "a", *common, "--probe", str(pv))[0] == 3
    err = capsys.readouterr().err
    assert ("only a complete, certifying verdict" in err) or ("not a CERTIFIED pair" in err)
    assert not (tmp_path / "a" / "compute_plan.json").exists()
    # uncertified: no HF+PEFT reference evidence
    pv, v = _real_verdict(tmp_path / "b", monkeypatch, no_reference=True)
    assert v["status"] == "uncertified"
    assert _run(tmp_path / "b", *common, "--probe", str(pv))[0] == 3 and "uncertified" in capsys.readouterr().err
    # the padding window was never in the plan -> its S was not measured
    pv, v = _real_verdict(tmp_path / "c", monkeypatch, drop_window=98304)
    assert v["status"] == "complete"
    assert _run(tmp_path / "c", *common, "--probe", str(pv))[0] == 3 and "padding window 98304 was not MEASURED" in capsys.readouterr().err
    # no throughput point at 49,152
    pv, v = _real_verdict(tmp_path / "d", monkeypatch, drop_throughput=True)
    assert v["status"] == "complete"
    assert _run(tmp_path / "d", *common, "--probe", str(pv))[0] == 3 and "no measured throughput" in capsys.readouterr().err
    # a production pair the verdict did not certify (rank 16 / mlp is not in the toy plan)
    pv, v = _real_verdict(tmp_path / "e", monkeypatch)
    assert _run(tmp_path / "e", "--serving-pair", "16:mlp", "--tok-s", f"psr_ce_all={LORA_TOK_S}", "--probe", str(pv))[0] == 3
    assert "not a CERTIFIED pair" in capsys.readouterr().err
    # a replan_required set refuses even on an otherwise complete document
    pv, v = _real_verdict(tmp_path / "f", monkeypatch)
    doc = json.loads(pv.read_text()); doc["replan_required"] = ["wo_r4_S1cc_cap8_L49152"]; pv.write_text(json.dumps(doc))
    assert _run(tmp_path / "f", *common, "--probe", str(pv))[0] == 3 and "replan_required" in capsys.readouterr().err
    # the production pair the planner reads from the fits config BY DEFAULT (its lora_r4_ce fit_selection — a folded pick, rank 16 / mlp)
    # is not the toy verdict's (4, wo): refused, the refusal naming BOTH pairs — a verdict certifying another pair never prices
    # production, and the default read is what a live --probe run exercises
    pv, v = _real_verdict(tmp_path / "h", monkeypatch)
    sel = yaml.safe_load(Path(fits_cfg).read_text())["train_kwargs"]["lora_r4_ce"]["fit_selection"]
    assert (sel["lora_rank"], sel["lora_modules"]) != (4, ["o_proj"]), "the synthetic config must not sit on the paper default"
    assert _run(tmp_path / "h", "--fits-config", fits_cfg, "--tok-s", f"psr_ce_all={LORA_TOK_S}", "--probe", str(pv))[0] == 3
    err = capsys.readouterr().err
    assert f"rank {sel['lora_rank']}" in err and "not a CERTIFIED pair" in err and "(4, 'wo')" in err, err
    # --probe without a way to name the production pair: a fits config (REQUIRED for the memprobe rows) that carries no
    # lora_r4_ce fit_selection, and no --serving-pair
    pv, v = _real_verdict(tmp_path / "g", monkeypatch)
    nosel = tmp_path / "g" / "sdfbase_nosel_fits.yaml"
    nosel.write_text(FITS_CFG_TEXT.split("    fit_selection:\n", 1)[0])
    assert "fit_selection" not in nosel.read_text() and "PENDING_MEMPROBE_EFF64_ACCUM" in nosel.read_text()
    assert _run(tmp_path / "g", "--fits-config", str(nosel), "--tok-s", f"psr_ce_all={LORA_TOK_S}", "--probe", str(pv))[0] == 3
    assert "--serving-pair" in capsys.readouterr().err
    # and with no fits config at all the memprobe rows cannot be priced in the first place
    assert _run(tmp_path / "g2", "--tok-s", f"psr_ce_all={LORA_TOK_S}", "--probe", str(pv), eb=None)[0] == 3
    assert "--fits-config is required" in capsys.readouterr().err


def test_the_complete_gate_in_common_sh_refuses_a_pending_plan(tmp_path):
    """`repro_require_compute_plan <model> complete` (lib/common.sh): a plan with pending serving rows is refused by name; the
    bare form (existence) still passes it — the stages that price steered waves ask for `complete`."""
    import os
    import subprocess

    state = tmp_path / "state" / "m"
    state.mkdir(parents=True)
    rc_, plan = _run(tmp_path, "--fit-width", "4", "--tok-s", f"psr_ce_all={LORA_TOK_S}")
    assert rc_ == 0 and plan["complete"] is False
    (state / "compute_plan.json").write_text(json.dumps(plan))
    common = REPO / "reproduce" / "lib" / "common.sh"
    env = dict(os.environ, SISTEER_STATE_DIR=str(tmp_path / "state"), SISTEER_PYTHON=sys.executable)
    prog = f'source {common} 2>/dev/null; repro_require_compute_plan m && echo BARE_OK; repro_require_compute_plan m complete && echo COMPLETE_OK'
    r = subprocess.run(["bash", "-c", prog], capture_output=True, text=True, env=env)
    assert "BARE_OK" in r.stdout and "COMPLETE_OK" not in r.stdout, (r.stdout, r.stderr)
    assert "INCOMPLETE" in (r.stdout + r.stderr) and "06_arms_multiturn" in (r.stdout + r.stderr)
    plan["complete"] = True
    (state / "compute_plan.json").write_text(json.dumps(plan))
    r = subprocess.run(["bash", "-c", prog], capture_output=True, text=True, env=env)
    assert "COMPLETE_OK" in r.stdout, (r.stdout, r.stderr)


@pytest.mark.parametrize("eb", [64, 8])
def test_fit_width_comes_from_the_fits_config_when_not_given(tmp_path, eb):
    cfg = tmp_path / "sdfbase_x_fits.yaml"
    cfg.write_text(CFG_TEXT_BY_EB[eb])                                                        # fit_width: 4
    rc_, plan = _run(tmp_path, "--fits-config", str(cfg), "--tok-s", f"psr_ce_all={LORA_TOK_S}")
    assert rc_ == 0 and plan["fit_width"] == 4 and plan["fits_config"] == str(cfg) and plan["arms"]["lora_r4_ce"]["accum_steps"] == eb // 2
    cfg.write_text(CFG_TEXT_BY_EB[eb].replace("fit_width: 4\n", ""))                          # absent -> 1
    rc_, plan = _run(tmp_path, "--fits-config", str(cfg))
    assert rc_ == 0 and plan["fit_width"] == 1 and plan["arms"]["psr_ce_all"]["accum_steps"] == eb


def test_the_effective_batch_is_the_fits_configs_own_by_the_trainers_rule(tmp_path, capsys):
    """The planner carries NO effective batch of its own — `effective_batch_of(config, arm)` is tune_arm
    config_effective_batch's rule (batch_size_by_arm x accum_steps when both are ints, else the PENDING_MEMPROBE_EFF<n>_ACCUM sentinel's
    n), read from the generated layout; the COMMITTED configs resolve to their lanes' batches (the paper lane 8, the demo lane and the seed
    replicate 64); no config, an arm stating neither key, or a value naming no batch REFUSES by name (rc 3, nothing written)."""
    assert rc.effective_batch_of(_cfg_at(tmp_path / "a", 64), "psr_ce_all") == rc.effective_batch_of(_cfg_at(tmp_path / "a", 64), "lora_r4_ce") == 64
    assert rc.effective_batch_of(_cfg_at(tmp_path / "b", 8), "psr_ce_all") == 1 * 8 and rc.effective_batch_of(_cfg_at(tmp_path / "b", 8), "lora_r4_ce") == 2 * 4
    cfgs = REPO / "configs"
    assert rc.effective_batch_of(str(cfgs / "sdfbase_qwen35_4b_base_fits.yaml"), "lora_r4_ce") == 8 == rc.effective_batch_of(str(cfgs / "sdfbase_qwen35_4b_base_fits.yaml"), "psr_ce_all")
    assert rc.effective_batch_of(str(cfgs / "sdfbase_qwen35_4b_demo_base_fits.yaml"), "lora_r4_ce") == 64 == rc.effective_batch_of(str(cfgs / "sdfbase_qwen35_4b_demo_base_fits.yaml"), "psr_ce_all")
    assert rc.effective_batch_of(str(cfgs / "sdfbase_qwen35_4b_base_eb64_s1_fits.yaml"), "lora_r4_ce") == 64
    # a folded micro-batch beside an unfolded accum sentinel: the sentinel names the batch (tune_arm: both ints, else the capture)
    mixed = tmp_path / "mixed.yaml"
    mixed.write_text(FITS_CFG_TEXT.replace("  psr_ce_all: MEMPROBE\n", "  psr_ce_all: 2\n"))
    assert rc.effective_batch_of(str(mixed), "psr_ce_all") == 64
    with pytest.raises(rc.Refused, match="--fits-config is required"):
        rc.effective_batch_of(None, "psr_ce_all")
    nokeys = tmp_path / "nokeys.yaml"; nokeys.write_text("model: a/b\nfit_width: 4\nsi_ids:\n- s\n")
    with pytest.raises(rc.Refused, match="states no effective batch for psr_ce_all"):
        rc.effective_batch_of(str(nokeys), "psr_ce_all")
    bad = tmp_path / "bad.yaml"; bad.write_text(FITS_CFG_TEXT.replace("PENDING_MEMPROBE_EFF64_ACCUM", "PENDING"))
    with pytest.raises(rc.Refused, match="names no effective batch"):
        rc.effective_batch_of(str(bad), "lora_r4_ce")
    # through the CLI: no --fits-config refuses before any record is priced; a config without the keys refuses by name
    rc_, plan = _run(tmp_path / "none", "--fit-width", "4", "--tok-s", f"psr_ce_all={LORA_TOK_S}", eb=None)
    assert rc_ == 3 and plan is None and "--fits-config is required" in capsys.readouterr().err
    rc_, plan = _run(tmp_path / "nokeys", "--fits-config", str(nokeys), "--fit-width", "4")
    assert rc_ == 3 and plan is None and "states no effective batch" in capsys.readouterr().err


def test_refusals_by_name(tmp_path, capsys):
    # the width must divide accum_steps (eb 64: ReLU 64, LoRA 32; eb 8: ReLU 8, LoRA 4) — 3 divides none; the refusal names the config's batch
    for eb in (64, 8):
        assert _run(tmp_path / f"w3_{eb}", "--fit-width", "3", "--tok-s", f"psr_ce_all={LORA_TOK_S}", eb=eb)[0] == 3
        err = capsys.readouterr().err
        assert "does not divide accum_steps" in err and f"the fits config's effective batch {eb}" in err, err
    # a fold that disagrees with the verdict's largest fitting rung
    bad_fold = tmp_path / "fold.json"
    bad_fold.write_text(json.dumps({"example_moe_base": {"lora_r4_ce": 4, "psr_ce_all": 1}}))
    assert _run(tmp_path, "--fold", str(bad_fold))[0] == 3
    assert "fold says micro-batch 4" in capsys.readouterr().err
    # --tok-s spellings
    assert _run(tmp_path, "--tok-s", "psr_ce_all")[0] == 3 and "ARM=TOKENS_PER_S" in capsys.readouterr().err
    assert _run(tmp_path, "--tok-s", "vector=100")[0] == 3
    assert _run(tmp_path, "--tok-s", "psr_ce_all=0")[0] == 3
    # the legacy `microbatch` layout, refused by name
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"arm": "psr_ce_all", "microbatch": {"1": {"fits": True, "s_per_step": 1.4}}}))
    cfg64 = _cfg_at(tmp_path, 64)
    assert rc.main(["--model", "m", "--memprobe", str(legacy), "--fits-config", cfg64, "--out", str(tmp_path / "o.json")]) == 3
    assert "`microbatch` mapping" in capsys.readouterr().err
    # a does-not-fit verdict
    nofit = json.loads(Path(PSR).read_text())
    nofit["verdict"]["fits_at_min_micro_batch"] = False
    for r in nofit["rungs"]:
        r["fits"] = False if r["fits"] is not None else None
    (tmp_path / "nofit.json").write_text(json.dumps(nofit))
    assert rc.main(["--model", "m", "--memprobe", str(tmp_path / "nofit.json"), "--fits-config", cfg64, "--out", str(tmp_path / "o.json")]) == 3
    assert "does not fit" in capsys.readouterr().err
    # the same arm twice
    assert rc.main(["--model", "m", "--memprobe", PSR, "--memprobe", PSR, "--fits-config", cfg64, "--out", str(tmp_path / "o.json")]) == 3


def test_dry_run_writes_nothing(tmp_path, capsys):
    out = tmp_path / "plan.json"
    assert rc.main(["--model", "m", "--memprobe", PSR, "--memprobe", LORA, "--fits-config", _cfg_at(tmp_path, 64), "--out", str(out), "--dry-run"]) == 0
    assert not out.exists() and "DRY-RUN: write" in capsys.readouterr().out


def test_exclude_row_removes_a_priced_row_from_the_total_and_keeps_the_plan_complete(tmp_path, monkeypatch, capsys, fits_cfg):
    """The padding ladder is EXCLUDED from the steered waves — `--exclude-row 06_arms_padding=<reason>` keeps the
    row (gpu_h null, excluded, the given reason, its priced figure beside it), lists it under excluded_rows (never pending_rows: `complete`
    stays true and lib/common.sh's `complete` gate passes), and the total omits it; an unknown row, a bare name or a repeat refuses by name."""
    import os
    import subprocess

    pv_path, verdict = _real_verdict(tmp_path, monkeypatch)
    common = ["--fits-config", fits_cfg, "--serving-pair", "4:wo", "--tok-s", f"psr_ce_all={LORA_TOK_S}", "--probe", str(pv_path)]
    rc0, full = _run(tmp_path / "full", *common)
    assert rc0 == 0 and full["complete"] is True and full["rows"]["06_arms_padding"]["gpu_h"]
    reason = "USER RULING D889: the padding ladder is excluded from the steered waves for these objects"
    rc_, plan = _run(tmp_path / "ex", *common, "--exclude-row", f"06_arms_padding={reason}")
    assert rc_ == 0 and plan["complete"] is True and plan["pending_rows"] == [] and plan["excluded_rows"] == ["06_arms_padding"] and plan["probe_status"] == "present"
    row = plan["rows"]["06_arms_padding"]
    assert row["excluded"] is True and row["status"] == "excluded" and row["gpu_h"] is None and row["excluded_reason"] == reason
    assert row["gpu_h_if_run"] == pytest.approx(full["rows"]["06_arms_padding"]["gpu_h"])                    # the priced figure stands beside the exclusion
    assert plan["total_gpu_h"] == pytest.approx(full["total_gpu_h"] - full["rows"]["06_arms_padding"]["gpu_h"], abs=0.2)
    assert plan["total_gpu_h"] == pytest.approx(sum(float(v["gpu_h"]) for v in plan["rows"].values() if v.get("gpu_h") is not None), abs=0.2)
    assert "rows EXCLUDED by ruling (not counted): 06_arms_padding (" in plan["decision"] and "pending" not in plan["decision"].split("EXCLUDED")[0].split("critical path")[-1]
    assert {k: v for k, v in plan["rows"].items() if k != "06_arms_padding"} == {k: v for k, v in full["rows"].items() if k != "06_arms_padding"}   # nothing else moves
    # lib/common.sh's `complete` gate passes the excluded plan (06 prices the steered wave on it)
    state = tmp_path / "state" / "m"; state.mkdir(parents=True); (state / "compute_plan.json").write_text(json.dumps(plan))
    env = dict(os.environ, SISTEER_STATE_DIR=str(tmp_path / "state"), SISTEER_PYTHON=sys.executable)
    prog = f'source {REPO / "reproduce" / "lib" / "common.sh"} 2>/dev/null; repro_require_compute_plan m complete && echo COMPLETE_OK'
    r = subprocess.run(["bash", "-c", prog], capture_output=True, text=True, env=env)
    assert "COMPLETE_OK" in r.stdout, (r.stdout, r.stderr)
    # (an UNPRICED row cannot be excluded at all — the no-probe padding exclusion REFUSES; see
    # test_exclude_row_refuses_an_unpriced_row_so_an_exclusion_never_stands_in_for_the_probe)
    # an unknown row, a bare name, an empty reason, or a repeat: refused by name, nothing written
    for bad, want in (("06_arms_nope=why", "not a row of the plan"), ("06_arms_padding", "expected ROW=REASON"), ("06_arms_padding=", "expected ROW=REASON")):
        rc3, plan3 = _run(tmp_path / f"bad_{abs(hash(bad))}", *common, "--exclude-row", bad)
        assert rc3 == 3 and plan3 is None and want in capsys.readouterr().err, (bad, want)
    rc4, plan4 = _run(tmp_path / "twice", *common, "--exclude-row", f"06_arms_padding={reason}", "--exclude-row", "06_arms_padding=again")
    assert rc4 == 3 and plan4 is None and "named twice" in capsys.readouterr().err


def test_exclude_row_refuses_an_unpriced_row_so_an_exclusion_never_stands_in_for_the_probe(tmp_path, capsys):
    """With no --probe both serving rows are pending; excluding one or both must not make the plan complete — an
    exclusion is a PRICED row set aside with its price beside it (gpu_h_if_run), never a way past the certified-serving gate. The padding row
    unpriced refuses the same way; a pending plan without exclusions is still written incomplete."""
    reason = "USER RULING D889: a test exclusion"
    for rows in (["06_arms_padding"], ["06_arms_multiturn"], ["06_arms_padding", "06_arms_multiturn"], ["06_arms_multiturn", "06_arms_padding"]):
        args = ["--fit-width", "4", "--tok-s", f"psr_ce_all={LORA_TOK_S}"]
        for r in rows:
            args += ["--exclude-row", f"{r}={reason}"]
        rc_, plan = _run(tmp_path / "_".join(rows), *args)
        err = capsys.readouterr().err
        assert rc_ == 3 and plan is None, (rows, err)
        assert f"--exclude-row {rows[0]}: the row is UNPRICED (gpu_h null" in err and "run the probe first, then exclude" in err, (rows, err)
    rc_, plan = _run(tmp_path / "plain", "--fit-width", "4", "--tok-s", f"psr_ce_all={LORA_TOK_S}")
    assert rc_ == 0 and plan["complete"] is False and set(plan["pending_rows"]) == {"06_arms_multiturn", "06_arms_padding"} and plan["excluded_rows"] == []


# ------------------------------------------------------------------------------------------ --serve-only (demo-import M6; Q3 ruled)
#: the DEMO's committed fits config: its lora_r4_ce fit_selection IS the toy verdict's pair (rank 4 / [o_proj] = `wo`), so the planner's
#: default read of the production pair from the config prices the one-pair verdict — the live --serve-only invocation's shape
DEMO_CFG = str(REPO / "configs" / "sdfbase_qwen35_4b_demo_base_fits.yaml")
SERVE_ONLY_IMPORTED = sorted(["02_memprobe", "03_recon", "03_relu_trials", "03_lora_trials", "04_fits", "04_derived"])   # the document lists them sorted


def _serve_only(tmp_path, *extra, model="qwen35_4b_demo"):
    out = tmp_path / "compute_plan.json"
    rc_ = rc.main(["--model", model, "--serve-only", "--out", str(out), *extra])
    return rc_, (json.loads(out.read_text()) if out.exists() else None)


def test_serve_only_with_a_certifying_one_pair_probe_is_complete_and_imports_the_fit_rows_but_never_the_probe(tmp_path, monkeypatch):
    """A model that imports its objects prices NO fit-side row — `--serve-only` with no --memprobe marks
    02_memprobe / 03_recon / 03_relu_trials / 03_lora_trials / 04_fits / 04_derived `imported` (gpu_h null, never pending, not counted)
    — while the two serving rows go through the SAME gate as a fitting lane's: the real emitter's certifying one-pair verdict, the pair
    read from the committed demo fits config's lora_r4_ce fit_selection (rank 4 / [o_proj] -> wo). `03_serving_probe` is NOT imported:
    the probe actually runs (its baseline literal stands). The plan is `complete`, lib/common.sh's gate passes it, and the document says
    serve_only / imported_rows / imported_objects (schema 3)."""
    import os
    import subprocess

    pv_path, verdict = _real_verdict(tmp_path, monkeypatch)
    sel = yaml.safe_load(Path(DEMO_CFG).read_text())["train_kwargs"]["lora_r4_ce"]["fit_selection"]
    assert (sel["lora_rank"], sel["lora_modules"]) == (4, ["o_proj"]), "the demo config's declared pair is the toy verdict's (wo, 4)"
    rc_, plan = _serve_only(tmp_path, "--probe", str(pv_path), "--fits-config", DEMO_CFG)
    assert rc_ == 0 and plan is not None, plan
    assert plan["schema"] == 3 and plan["complete"] is True and plan["pending_rows"] == [] and plan["probe_status"] == "present"
    assert plan["serve_only"] is True and plan["imported_rows"] == SERVE_ONLY_IMPORTED and plan["imported_objects"] == "objects/DEMO.json"
    assert "03_serving_probe" not in plan["imported_rows"] and plan["rows"]["03_serving_probe"] == rc.BASELINE["03_serving_probe"]
    assert "imported" not in plan["rows"]["03_serving_probe"]
    for row in SERVE_ONLY_IMPORTED:
        r = plan["rows"][row]
        assert r["status"] == "imported" and r["imported"] is True and r["gpu_h"] is None and "objects/DEMO.json" in r["reason"], (row, r)
        if rc.BASELINE[row].get("gpu_h") is not None:
            assert r["gpu_h_if_run"] == rc.BASELINE[row]["gpu_h"]                       # the baseline figure stands beside the import
    assert plan["memprobe"] == [] and plan["arms"] == {} and plan["fit_width"] is None and plan["fit_replan_triggered"] is False
    # the serving rows: the SAME pricing as a fitting lane's (test_a_verdict_written_by_the_emitter_prices_both_serving_rows)
    mt, pad = plan["rows"]["06_arms_multiturn"], plan["rows"]["06_arms_padding"]
    assert mt["basis"] == "certified" and mt["pair"]["rank"] == 4 and mt["pair"]["set"] == "wo" and mt["admitted_S"] == 5 and mt["tok_s"] == 900.0
    assert mt["gpu_h"] == pytest.approx(3 * 10 * 259 * 10 * 1500 / 900 / 3600, abs=0.1)
    assert pad["admitted_S"] == 2 and pad["tok_s_assumed"] == pytest.approx(360.0) and "assumption" in pad
    assert plan["probe_verdict"]["probe_job_id"] == "4242" and plan["probe_verdict"]["pair"]["set"] == "wo"
    # the total counts the priced rows alone: the imported rows contribute nothing (the baseline's 2 + 3 + 1 GPU-h are not in it)
    assert plan["total_gpu_h"] == pytest.approx(sum(float(v["gpu_h"]) for v in plan["rows"].values() if v.get("gpu_h") is not None), abs=0.2)
    assert plan["rows"]["03_serving_probe"]["gpu_h"] == 6 and plan["total_gpu_h"] >= 6
    assert "IMPORTED" in plan["decision"] and "no memprobe read" in plan["decision"] and "pending" not in plan["decision"]
    # lib/common.sh's `complete` gate (06 / 09) passes the serve-only plan
    state = tmp_path / "state" / "m"; state.mkdir(parents=True); (state / "compute_plan.json").write_text(json.dumps(plan))
    env = dict(os.environ, SISTEER_STATE_DIR=str(tmp_path / "state"), SISTEER_PYTHON=sys.executable)
    prog = f'source {REPO / "reproduce" / "lib" / "common.sh"} 2>/dev/null; repro_require_compute_plan m complete && echo COMPLETE_OK'
    r = subprocess.run(["bash", "-c", prog], capture_output=True, text=True, env=env)
    assert "COMPLETE_OK" in r.stdout, (r.stdout, r.stderr)
    # the same through the documented invocation shape with an explicit --serving-pair (replan's option, not 03's)
    rc2, plan2 = _serve_only(tmp_path / "pair", "--probe", str(pv_path), "--serving-pair", "4:wo")
    assert rc2 == 0 and plan2["complete"] is True and plan2["rows"]["06_arms_multiturn"]["gpu_h"] == mt["gpu_h"]


def test_serve_only_without_a_certifying_probe_pends_exactly_the_two_serving_rows(tmp_path, monkeypatch, capsys):
    """The probe gate is UNCHANGED under --serve-only (no --probe-imported, no shape pricing): with no --probe the plan is written
    INCOMPLETE with exactly the two serving rows pending (the imported rows are not pending), lib/common.sh's `complete` gate refuses it,
    and a verdict that cannot price a row — partial / uncertified / replan_required — REFUSES (rc 3, nothing written) exactly as for a
    fitting lane."""
    import os
    import subprocess

    rc_, plan = _serve_only(tmp_path / "none")
    assert rc_ == 0 and plan["complete"] is False and plan["probe_status"] == "pending" and plan["probe"] is None
    assert plan["pending_rows"] == ["06_arms_multiturn", "06_arms_padding"] and plan["imported_rows"] == SERVE_ONLY_IMPORTED
    assert plan["rows"]["06_arms_multiturn"]["status"] == "pending" and plan["rows"]["06_arms_padding"]["status"] == "pending"
    state = tmp_path / "state" / "m"; state.mkdir(parents=True); (state / "compute_plan.json").write_text(json.dumps(plan))
    env = dict(os.environ, SISTEER_STATE_DIR=str(tmp_path / "state"), SISTEER_PYTHON=sys.executable)
    prog = f'source {REPO / "reproduce" / "lib" / "common.sh"} 2>/dev/null; repro_require_compute_plan m complete && echo COMPLETE_OK'
    r = subprocess.run(["bash", "-c", prog], capture_output=True, text=True, env=env)
    assert "COMPLETE_OK" not in r.stdout and "INCOMPLETE" in (r.stdout + r.stderr) and "06_arms_multiturn" in (r.stdout + r.stderr)
    # a non-certifying verdict refuses by name — the gate's refusals are the fitting lane's, word for word
    pv, v = _real_verdict(tmp_path / "a", monkeypatch, oom_49152=True)
    assert _serve_only(tmp_path / "a", "--probe", str(pv), "--fits-config", DEMO_CFG)[0] == 3
    err = capsys.readouterr().err
    assert ("only a complete, certifying verdict" in err) or ("not a CERTIFIED pair" in err)
    assert not (tmp_path / "a" / "compute_plan.json").exists()
    pv, v = _real_verdict(tmp_path / "b", monkeypatch, no_reference=True)
    assert v["status"] == "uncertified"
    assert _serve_only(tmp_path / "b", "--probe", str(pv), "--fits-config", DEMO_CFG)[0] == 3 and "uncertified" in capsys.readouterr().err
    pv, v = _real_verdict(tmp_path / "c", monkeypatch)
    doc = json.loads(pv.read_text()); doc["replan_required"] = ["wo_r4_S1cc_cap8_L49152"]; pv.write_text(json.dumps(doc))
    assert _serve_only(tmp_path / "c", "--probe", str(pv), "--fits-config", DEMO_CFG)[0] == 3 and "replan_required" in capsys.readouterr().err
    pv, v = _real_verdict(tmp_path / "d", monkeypatch, drop_window=98304)
    assert _serve_only(tmp_path / "d", "--probe", str(pv), "--fits-config", DEMO_CFG)[0] == 3 and "padding window 98304 was not MEASURED" in capsys.readouterr().err


def test_serve_only_refuses_a_verdict_whose_pair_is_not_the_fits_configs(tmp_path, monkeypatch, capsys, fits_cfg):
    """Identity rejection: the serve-only plan prices the PRODUCTION pair the fits config declares; a verdict certifying another pair
    (the toy's (4, wo) against the synthetic config's rank 16 / mlp) refuses naming both — a probe of the wrong pair never prices the
    demo's serving rows. A config with no fit_selection and no --serving-pair refuses too."""
    pv, v = _real_verdict(tmp_path, monkeypatch)
    sel = yaml.safe_load(Path(fits_cfg).read_text())["train_kwargs"]["lora_r4_ce"]["fit_selection"]
    assert (sel["lora_rank"], sel["lora_modules"]) != (4, ["o_proj"])
    assert _serve_only(tmp_path / "h", "--probe", str(pv), "--fits-config", fits_cfg)[0] == 3
    err = capsys.readouterr().err
    assert f"rank {sel['lora_rank']}" in err and "not a CERTIFIED pair" in err and "(4, 'wo')" in err, err
    assert not (tmp_path / "h" / "compute_plan.json").exists()
    nosel = tmp_path / "sdfbase_nosel_fits.yaml"
    nosel.write_text(FITS_CFG_TEXT.split("    fit_selection:\n", 1)[0])
    assert _serve_only(tmp_path / "g", "--probe", str(pv), "--fits-config", str(nosel))[0] == 3 and "--serving-pair" in capsys.readouterr().err
    assert _serve_only(tmp_path / "g2", "--probe", str(pv))[0] == 3 and "--serving-pair" in capsys.readouterr().err


def test_serve_only_refuses_every_fit_side_input_and_a_fitting_lane_still_needs_its_memprobes(tmp_path, capsys):
    """`--serve-only` beside --memprobe / --fold / --tok-s / --fit-width refuses by name (nothing written): a plan cannot half-import and
    half-price the same objects. Without --serve-only, `--memprobe` is still required — the refusal names both ways out."""
    for extra, flag in ((["--memprobe", PSR], "--memprobe"), (["--fold", FOLD], "--fold"), (["--tok-s", f"psr_ce_all={LORA_TOK_S}"], "--tok-s"),
                        (["--fit-width", "4"], "--fit-width")):
        rc_, plan = _serve_only(tmp_path / flag.strip("-"), *extra, "--fits-config", DEMO_CFG)
        err = capsys.readouterr().err
        assert rc_ == 3 and plan is None and f"--serve-only takes no {flag}" in err, (flag, err)
    out = tmp_path / "plain.json"
    assert rc.main(["--model", "m", "--out", str(out), "--fits-config", DEMO_CFG]) == 3 and not out.exists()
    err = capsys.readouterr().err
    assert "no --memprobe record" in err and "--serve-only" in err
    # a dry run of the serve-only plan writes nothing and prints the document
    assert rc.main(["--model", "m", "--serve-only", "--out", str(out), "--dry-run"]) == 0 and not out.exists()
    printed = capsys.readouterr().out
    assert "DRY-RUN: write" in printed and '"serve_only": true' in printed


def test_serve_only_keeps_the_exclude_row_guard_an_exclusion_never_stands_in_for_the_probe(tmp_path, monkeypatch, capsys):
    """Under --serve-only, with no --probe the serving rows are UNPRICED and cannot be excluded — the plan never becomes
    complete through an exclusion; an imported row cannot be excluded either (it has no price to set aside); with a certifying probe
    the exclusion of the priced padding row works as for a fitting lane, the plan staying complete."""
    reason = "USER RULING D889: a test exclusion"
    for rows in (["06_arms_padding"], ["06_arms_multiturn"], ["06_arms_padding", "06_arms_multiturn"]):
        args = []
        for r in rows:
            args += ["--exclude-row", f"{r}={reason}"]
        rc_, plan = _serve_only(tmp_path / "_".join(rows), *args)
        err = capsys.readouterr().err
        assert rc_ == 3 and plan is None and f"--exclude-row {rows[0]}: the row is UNPRICED (gpu_h null" in err and "run the probe first, then exclude" in err, (rows, err)
    rc_, plan = _serve_only(tmp_path / "imp", "--exclude-row", f"04_fits={reason}")
    assert rc_ == 3 and plan is None and "--exclude-row 04_fits: the row is IMPORTED" in capsys.readouterr().err
    pv_path, verdict = _real_verdict(tmp_path, monkeypatch)
    rc0, full = _serve_only(tmp_path / "full", "--probe", str(pv_path), "--fits-config", DEMO_CFG)
    rc1, plan = _serve_only(tmp_path / "ex", "--probe", str(pv_path), "--fits-config", DEMO_CFG, "--exclude-row", f"06_arms_padding={reason}")
    assert rc0 == 0 and rc1 == 0 and plan["complete"] is True and plan["excluded_rows"] == ["06_arms_padding"] and plan["pending_rows"] == []
    assert plan["rows"]["06_arms_padding"]["gpu_h_if_run"] == pytest.approx(full["rows"]["06_arms_padding"]["gpu_h"])
    assert plan["imported_rows"] == SERVE_ONLY_IMPORTED and plan["total_gpu_h"] == pytest.approx(full["total_gpu_h"] - full["rows"]["06_arms_padding"]["gpu_h"], abs=0.2)


def test_plan_document_carries_the_serve_only_keys_for_a_fitting_lane_too(tmp_path):
    """The writer's one shape: a fitting lane's plan (and the integration tests' synthetic plans built from plan_document with the old
    keywords) carry serve_only false / imported_rows [] / imported_objects null — the keys exist on every document, never only on the
    serve-only one."""
    rc_, plan = _run(tmp_path, "--fit-width", "4", "--tok-s", f"psr_ce_all={LORA_TOK_S}")
    assert rc_ == 0 and plan["serve_only"] is False and plan["imported_rows"] == [] and plan["imported_objects"] is None
    assert not any(v.get("imported") for v in plan["rows"].values())
    doc = rc.plan_document(model="m", memprobe=[], fold=[], probe=None, probe_status="present", probe_rec=None, fit_width=1, fits_config=None,
                           step_threshold_s=0.0, replan=False, arms={}, plan={}, pending=[], excluded={}, total=0.0, decision=["synthetic"])
    assert doc["serve_only"] is False and doc["imported_rows"] == [] and doc["imported_objects"] is None and doc["complete"] is True
