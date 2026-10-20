"""fm2 / fm2b: the seedrep attacker pools are ONE H200 each (TP 1), with
``max_num_seqs`` and ``max_model_len`` DERIVED from the measured attacker traffic and the production fan-out instead of
the S x L identity that had pushed them onto two cards -- the Slurm precedent's shape, the contract's ``peak_mean`` rule:
S = the production peak x 1.25, the KV checked at the MEAN sequence, vLLM's scheduler queuing / preempting on the tail,
the p99 fit recorded BESIDE it and never as the bound. Pinned here, each with its known-bad twin:

  * the derivation and its inputs (the four pools' vLLM ``/metrics`` histograms of 2026-09-19, the attacker's 1,024 reply
    cap, the plan's pinned children per pool) -> crescendo S 64 / actorattack 158 / pair 64 at a 65,536 window, with
    ``p99_fit_S`` 24 / 149 / 64 beside them
  * the KV-fit arithmetic refuses an (S, L) that does not fit one card BY NAME (an S whose S x mean exceeds the KV budget; a
    window below the attack's p99 sequence); the contract's coverage check applies the same rule to the live engine's KV
    line and records the p99 coverage fraction beside it
  * the plan's pool rows and capacity ledger: 4 pools x 1 GPU -> 134 containers / 134 GPUs
  * ``alloc_conf_env(1)`` keeps ``expandable_segments:True`` (the campaign pool's one-card setting; F-M's IPC constraint is
    the custom all-reduce's, which exists only above one card)
  * the endpoint record / snapshot carries ``gpu == "H200"``, ``tensor_parallel_size == 1``, the sizing arithmetic and the
    ``watch_metrics`` the launch agent reads on the S4/S5 rows
  * the one-card endpoint publish wait is derived from the measured TP-2 startup and stays above the 1,800 s floor."""
from __future__ import annotations

import ast
import collections
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts" / "modal"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from test_seedrep_attacker_pool import POOL_ENV, _load  # noqa: E402  (the stub-modal loader)
from sisteer.eval.attacks import attacker_pool as c  # noqa: E402


@pytest.fixture(scope="module")
def pool():
    return _load(name="_seedrep_attacker_pool_fm2")


@pytest.fixture(scope="module")
def lp():
    spec = importlib.util.spec_from_file_location("_seedrep_launch_plan_fm2", REPO / "scripts/modal/seedrep_launch_plan.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- the derivation, pinned with its inputs

def test_the_measured_inputs_are_the_pools_metrics_and_the_reply_cap_is_the_drivers_1024(pool):
    m = pool.MEASURED_ATTACKER_CALLS
    assert set(m) == set(pool.ATTACKS) == {"crescendo", "actorattack", "pair"}
    # every histogram is cumulative, ends at +Inf with the call count, and names the pool it was read from
    for attack, rec in m.items():
        for key in ("prompt_buckets", "generation_buckets", "max_tokens_buckets"):
            b = rec[key]
            assert b[-1][0] == float("inf") and b[-1][1] == rec["n_calls"], (attack, key)
            assert all(b[i][1] <= b[i + 1][1] for i in range(len(b) - 1)), (attack, key)
        assert rec["pool"].startswith("seedrep-attacker-") and rec["waiting_max"] == 0
    # the reply cap is the drivers' attacks.attacker_max_new_tokens (1024), NOT the target's 4,096: every generation call sits
    # in the (1,000, 2,000] max_tokens bucket and every 8-token scorer call in (5, 10]
    assert pool.ATTACKER_REPLY_CAP_TOKENS == 1_024
    cfg = (REPO / "configs/sdfbase_full22_qwen35_4b_s1.yaml").read_text()
    assert "attacker_max_new_tokens: 1024" in cfg
    for attack, rec in m.items():
        mt = dict(rec["max_tokens_buckets"])
        assert mt[1_000] < mt[2_000] == rec["n_calls"] and mt[10] == mt[1_000], attack
    # crescendo is the long-context attack: 3.4 % of its prompts above 20,000 tokens, none above 50,000; pair none above
    # 20,000; actorattack none above 5,000
    pb = {a: dict(r["prompt_buckets"]) for a, r in m.items()}
    assert pb["crescendo"][20_000] == 14_950 and pb["crescendo"][50_000] == 15_478 == m["crescendo"]["n_calls"]
    assert pb["pair"][20_000] == 10_320 and pb["pair"][10_000] == 10_318 and pb["actorattack"][5_000] == 5_375


def test_histogram_quantile_is_the_prometheus_estimator_with_its_known_bad_inputs(pool):
    hq = pool.histogram_quantile
    b = ((10, 5), (20, 10), (float("inf"), 10))
    assert hq(b, 0.5) == 10.0 and hq(b, 0.99) == pytest.approx(19.8) and hq(b, 1.0) == 20.0
    assert hq(((10, 0), (20, 10), (float("inf"), 12)), 0.99) == 20.0            # the quantile in the +Inf bucket -> its lower bound
    assert hq(pool.MEASURED_ATTACKER_CALLS["crescendo"]["prompt_buckets"], 0.99) == pytest.approx(41_205.68, abs=0.01)
    with pytest.raises(ValueError, match="quantile"):
        hq(b, 0.0)
    with pytest.raises(ValueError, match=r"\+Inf"):
        hq(((10, 5), (20, 10)), 0.5)
    with pytest.raises(ValueError, match="empty"):
        hq(((10, 0), (float("inf"), 0)), 0.5)


def test_the_derived_geometry_per_attack_and_the_window(pool):
    # p99 sequence = interpolated p99 prompt + the reply cap; mean sequence = the histogram's mean prompt + the cap
    p = {a: pool.p99_sequence_tokens(a) for a in pool.ATTACKS}
    assert (p["crescendo"]["p99_prompt_tokens"], p["crescendo"]["p99_seq_tokens"], p["crescendo"]["p99_seq_tokens_ceiling"]) == (41_206, 42_230, 51_024)
    assert (p["actorattack"]["p99_prompt_tokens"], p["actorattack"]["p99_seq_tokens"]) == (4_190, 5_214)
    assert (p["pair"]["p99_prompt_tokens"], p["pair"]["p99_seq_tokens"]) == (6_818, 7_842)
    m = {a: pool.mean_sequence_tokens(a) for a in pool.ATTACKS}
    assert (m["crescendo"]["mean_prompt_tokens"], m["crescendo"]["mean_seq_tokens"]) == (4_756, 5_780)
    assert (m["actorattack"]["mean_seq_tokens"], m["pair"]["mean_seq_tokens"]) == (1_864, 2_691)
    for fn in (pool.p99_sequence_tokens, pool.mean_sequence_tokens):
        with pytest.raises(pool.PoolParamError, match="no measured attacker calls"):
            fn("tap")
    # the window: the largest observed prompt bucket + the cap, chunk-aligned -> 65,536 (the ruling's assumed 4,096 cap lands the same)
    w = pool.WINDOW_DERIVATION
    assert (w["max_prompt_bucket"], w["reply_cap"], w["align"], w["window"]) == (50_000, 1_024, 16_384, 65_536)
    assert -(-(50_000 + 1_024) // 16_384) * 16_384 == 65_536 == -(-(50_000 + 4_096) // 16_384) * 16_384 == pool.DEFAULT_MAX_MODEL_LEN
    # the peak: pinned children x 1.25 IS the launched S (peak_mean); the mean bound S_kv_mean must hold it
    assert pool.PRODUCTION_PEAK_CHILDREN == {"crescendo": 51, "actorattack": 126, "pair": 51} and pool.PEAK_HEADROOM == 1.25
    assert (pool.peak_slots("crescendo"), pool.peak_slots("actorattack"), pool.peak_slots("pair")) == (64, 158, 64)
    with pytest.raises(pool.PoolParamError, match="below the peak"):
        pool.peak_slots("pair", headroom=0.5)
    assert pool.POOL_KV_RULE == "peak_mean" and pool.WATCH_METRIC_NAMES == ("vllm:num_requests_waiting", "vllm:num_preemptions_total",
                                                                          "vllm:request_queue_time_seconds", "vllm:e2e_request_latency_seconds")
    d = {a: pool.derive_pool_sizing(a, 65_536) for a in pool.ATTACKS}
    for a in pool.ATTACKS:
        assert d[a]["kv_rule"] == "peak_mean" and d[a]["slots_cover_peak"] is True and d[a]["S"] == d[a]["S_peak"] <= d[a]["S_kv_mean"] and d[a]["L"] == 65_536
        assert d[a]["watch_metrics"] == list(pool.WATCH_METRIC_NAMES) and d[a]["identity_S"] == 16      # what the S x L identity would have given
    assert (d["crescendo"]["S"], d["crescendo"]["S_kv_mean"], d["crescendo"]["mean_seq_tokens"], d["crescendo"]["p99_fit_S"], d["crescendo"]["S_kv_p99"], d["crescendo"]["p99_seq_tokens"]) == (64, 139, 5_780, 24, 24, 42_230)
    assert (d["actorattack"]["S"], d["actorattack"]["S_kv_mean"], d["actorattack"]["mean_seq_tokens"], d["actorattack"]["p99_fit_S"], d["actorattack"]["S_kv_p99"]) == (158, 269, 1_864, 149, 149)
    assert (d["pair"]["S"], d["pair"]["S_kv_mean"], d["pair"]["mean_seq_tokens"], d["pair"]["p99_fit_S"], d["pair"]["S_kv_p99"]) == (64, 225, 2_691, 64, 110)
    # the arithmetic behind crescendo: 64 x 5,780 = 369,920 KV tokens x 64 KiB = 22.6 GiB inside the ~66 GiB the budget leaves after
    # weights 51.7 + state 64 x 0.14 + activations 2.5 + reserve 2 (37.8 GiB headroom); the model's KV holds 14 % of a 64 x p99 peak
    assert d["crescendo"]["kv_tokens_at_S"] == 64 * 5_780 == 369_920 and c.QWEN36_27B.kv_bytes_per_token == 65_536
    assert d["crescendo"]["model_headroom_gib_at_S"] == pytest.approx(37.76, abs=0.05) and d["crescendo"]["p99_kv_coverage_model"] == pytest.approx(0.1369, abs=0.001)
    # the module-level geometry the Server declares IS the crescendo derivation (pool 0 = crescendo)
    assert (pool.P.gpu, pool.P.tensor_parallel, pool.MAX_MODEL_LEN, pool.MAX_NUM_SEQS, pool.GEOM.seqs_source) == ("H200", 1, 65_536, 64, "peak_mean")
    assert (pool.GEOM.kv_rule, pool.GEOM.kv_seq_tokens) == ("peak_mean", 5_780) and pool.SIZING["p99_fit_S"] == 24
    m2 = _load({"SEEDREP_ATTACKER_POOL_INDEX": "2", "SEEDREP_ATTACKER_ATTACK": "actorattack"}, name="_seedrep_attacker_pool_fm2_aa")
    m3 = _load({"SEEDREP_ATTACKER_POOL_INDEX": "3", "SEEDREP_ATTACKER_ATTACK": "pair"}, name="_seedrep_attacker_pool_fm2_pair")
    assert (m2.MAX_NUM_SEQS, m2.GEOM.seqs_source, m3.MAX_NUM_SEQS, m3.GEOM.seqs_source) == (158, "peak_mean", 64, "peak_mean")
    for m in (pool, m2, m3):
        kw = {a.name: a for a in m._fake_modal.App.instances}[m.APP_NAME].server_kwargs
        assert kw["gpu"] == "H200" and kw["target_concurrency"] == m.MAX_NUM_SEQS and kw["cpu"] == 16 and kw["memory"] == 131_072


def test_the_pinned_children_per_pool_are_the_plans(pool, lp):
    inputs = lp.load_inputs(REPO)
    d = lp.build_plan(W="2" * 40, wave_uuid="22222222-3333-4444-8444-555555555555", reqs_sha256=inputs["reqs_sha256"], roster=inputs["roster"],
                      pins=inputs["pins"], arditi_pins=inputs["arditi_pins"], arms=inputs["arms"], materialisation="draft",
                      generated_at="2026-09-18T00:00:00Z")
    by_pool = collections.Counter()
    for r in d["standups"]:
        if r.get("pool"):
            by_pool[r["pool"]] += len(r["children"])
    attack_of = {p["name"]: p["attacks"][0] for p in d["pools"]}
    assert dict(by_pool) == {"seedrep-attacker-0": 51, "seedrep-attacker-1": 51, "seedrep-attacker-2": 126, "seedrep-attacker-3": 51}
    for name, n in by_pool.items():
        assert pool.PRODUCTION_PEAK_CHILDREN[attack_of[name]] == n, name
    # the plan's pool rows are one H200 each and the ledger counts 4 pool GPUs -> 134 / 134
    assert all((p["gpu"], p["gpus"], p["max_model_len"]) == ("H200", 1, 65_536) for p in d["pools"]) and len(d["pools"]) == 4
    led = d["capacity_ledger"]
    assert led["pools"] == {"containers": 4, "gpus": 4} and led["targets_plus_pools"] == {"containers": 78, "gpus": 78}
    assert led["unstaged_peak"] == {"containers": 134, "gpus": 134} == led["unstaged_peak_without_dose"]
    assert lp.S4_DEFAULTS["pools"]["gpu"] == "H200" == pool.DEFAULT_GPU
    # a two-card spec in the S4 fields still counts its cards (the row's `gpus` follows its `gpu` spec)
    rows = lp.pool_rows({"pools": {**lp.S4_DEFAULTS["pools"], "gpu": "H200:2"}}, "2222222", "22222222")
    assert all(r["gpus"] == 2 for r in rows)


# --------------------------------------------------------------------------- the KV-fit refusals (pool tool + contract)

def test_an_s_or_l_that_does_not_fit_one_card_is_refused_by_name(pool, monkeypatch):
    # an explicit S whose S x mean exceeds the KV budget: refused, naming the peak_mean rule and the largest S that fits
    with pytest.raises(c.PoolGeometryUnfit, match=r"peak_mean rule \(S x 5780 sequence tokens.*max_num_seqs <= 139"):
        pool.pool_geometry(pool.pool_params({**POOL_ENV, "SEEDREP_ATTACKER_MAX_NUM_SEQS": "140"}))
    # a production peak the mean bound cannot hold refuses the DERIVATION by name (400 children x 1.25 = 500 > 139)
    monkeypatch.setitem(pool.PRODUCTION_PEAK_CHILDREN, "crescendo", 400)
    with pytest.raises(c.PoolGeometryUnfit, match=r"peak_mean rule.*max_num_seqs <= 139"):
        pool.derive_pool_sizing("crescendo", 65_536)
    monkeypatch.setitem(pool.PRODUCTION_PEAK_CHILDREN, "crescendo", 51)
    # a window that cannot hold the attack's p99 sequence is the WRONG WINDOW, refused by name with the derived one
    with pytest.raises(pool.PoolParamError, match=r"MAX_MODEL_LEN=32768 cannot hold crescendo's p99 sequence of 42,230 tokens.*the window, not S, is wrong.*65,536"):
        pool.derive_pool_sizing("crescendo", 32_768)
    # an explicit S that fits is kept (explicit), still under the peak_mean rule -- the two-card era's 44 included
    g = pool.pool_geometry(pool.pool_params({**POOL_ENV, "SEEDREP_ATTACKER_MAX_NUM_SEQS": "44"}))
    assert (g.max_num_seqs, g.seqs_source, g.kv_rule, g.kv_seq_tokens, g.fits) == (44, "explicit", "peak_mean", 5_780, True)
    # the contract: the KV term under a named rule, its validation, and the identity untouched for every other caller
    assert c.kv_tokens_needed(65_536, 64, 5_780) == 369_920 and c.kv_tokens_needed(65_536, 1, 5_780) == 65_536 and c.kv_tokens_needed(65_536, 24) == 24 * 65_536
    ident = c.resolve_pool_geometry(65_536, None, chunked_prefill=True, chunk_tokens=16_384)
    assert (ident.max_num_seqs, ident.seqs_source, ident.kv_seq_tokens, ident.kv_rule) == (16, "kv_fit", None, "identity")
    assert c.resolve_pool_geometry(19200, 37, chunked_prefill=True, chunk_tokens=16_384).kv_rule == "identity"     # the campaign pool
    p99 = c.resolve_pool_geometry(65_536, None, chunked_prefill=True, chunk_tokens=16_384, kv_seq_tokens=42_230, kv_rule="p99_fit")
    assert (p99.max_num_seqs, p99.seqs_source, p99.kv_rule) == (25, "kv_fit", "p99_fit")
    with pytest.raises(ValueError, match="kv_seq_tokens must be a positive int"):
        c.fit(65_536, 1, kv_seq_tokens=0, kv_rule="peak_mean")
    with pytest.raises(ValueError, match="exceeds the window"):
        c.fit(32_768, 1, kv_seq_tokens=42_230, kv_rule="p99_fit")
    with pytest.raises(ValueError, match="must be NAMED"):
        c.fit(65_536, 1, kv_seq_tokens=5_780)
    with pytest.raises(ValueError, match="needs kv_seq_tokens"):
        c.fit(65_536, 1, kv_rule="peak_mean")
    assert ident.as_dict()["kv_rule"] == "identity" and ident.as_dict()["kv_seq_tokens"] is None


def test_the_live_coverage_check_applies_the_launched_rule_and_records_the_p99_fraction(pool):
    # the Server attests coverage under GEOM's rule: the live KV line must hold S x mean (and one full window)
    live_ok = {"kv_tokens_reported": 1_060_000, "concurrency_window": 65_536, "max_concurrency_reported": 16.17}
    assert c.coverage_problem(live_ok, max_num_seqs=64, max_model_len=65_536, kv_seq_tokens=5_780, kv_rule="peak_mean") is None
    # KNOWN-BAD: the identity check would REFUSE this very engine at S 64 (16.17x < 64) -- the rule change is what lets one card advertise
    assert "below the launched max_num_seqs 64" in c.coverage_problem(live_ok, max_num_seqs=64, max_model_len=65_536)
    # KNOWN-BAD: an engine short of the mean need is refused, naming the rule and the S that would fit live
    short = {"kv_tokens_reported": 300_000, "concurrency_window": 65_536, "max_concurrency_reported": 4.58}
    msg = c.coverage_problem(short, max_num_seqs=64, max_model_len=65_536, kv_seq_tokens=5_780, kv_rule="peak_mean")
    assert "300,000 KV tokens" in msg and "369,920 the peak_mean rule needs" in msg and "relaunch at S <= 51" in msg
    # the p99 coverage FRACTION recorded beside the verdict: 1,060,000 / (64 x 42,230) = 0.39 of a worst-case peak; > 1 at the mean
    assert c.coverage_fraction(live_ok, max_num_seqs=64, max_model_len=65_536, seq_tokens=42_230) == pytest.approx(0.3922, abs=1e-4)
    assert c.coverage_fraction(live_ok, max_num_seqs=64, max_model_len=65_536, seq_tokens=5_780) == pytest.approx(2.8655, abs=1e-4)
    assert c.coverage_fraction({}, max_num_seqs=64, max_model_len=65_536, seq_tokens=42_230) is None
    with pytest.raises(ValueError, match="seq_tokens must be a positive int"):
        c.coverage_fraction(live_ok, max_num_seqs=64, max_model_len=65_536, seq_tokens=0)
    # without the KV-size line the concurrency line stands in (conc x L); an unnamed rule, a window mismatch and an unattested log refuse
    assert c.coverage_problem({"concurrency_window": 65_536, "max_concurrency_reported": 5.65}, max_num_seqs=64, max_model_len=65_536, kv_seq_tokens=5_780, kv_rule="peak_mean") is None
    assert "must be NAMED" in c.coverage_problem(live_ok, max_num_seqs=64, max_model_len=65_536, kv_seq_tokens=5_780)
    assert "not the launched window" in c.coverage_problem(live_ok, max_num_seqs=64, max_model_len=49_152, kv_seq_tokens=5_780, kv_rule="peak_mean")
    assert "unattested" in c.coverage_problem({}, max_num_seqs=64, max_model_len=65_536, kv_seq_tokens=5_780, kv_rule="peak_mean")
    # the Server's _attest_coverage: the verdict under GEOM's rule, and the rule / sequence / p99 fraction recorded on the coverage block
    log = "GPU KV cache size: 1,060,000 tokens\nMaximum concurrency for 65,536 tokens per request: 16.17x\n"
    cov = pool._attest_coverage(log)
    assert (cov["kv_rule"], cov["kv_seq_tokens"], cov["p99_seq_tokens"], cov["p99_coverage_fraction"]) == ("peak_mean", 5_780, 42_230, pytest.approx(0.3922, abs=1e-4))
    with pytest.raises(RuntimeError, match="peak_mean rule needs"):
        pool._attest_coverage("GPU KV cache size: 300,000 tokens\nMaximum concurrency for 65,536 tokens per request: 4.58x\n")


def test_an_explicit_width_that_fits_launches_when_the_automatic_recommendation_does_not(lp):
    """fm2b (reproduced): crescendo, H200, utilization 0.60, explicit SEEDREP_ATTACKER_MAX_NUM_SEQS=8 -- the pool
    resolver accepts S 8, yet fm2b refused at import while deriving the UNUSED automatic recommendation (S_peak 64, which does not
    fit at 0.60), and following the refusal's advice re-checked S 64 again; SIZING and launched_sizing() invoked the same
    derivation. Now the automatic launch keeps refusing when its peak cannot fit; an explicit launch validates its OWN S and records
    the unfittable recommendation as metadata (recommended.fits false + reason) -- through module initialization AND endpoint-record
    production, with the certificate reader accepting the record."""
    over = {"SEEDREP_ATTACKER_GPU_MEM_UTIL": "0.60", "SEEDREP_ATTACKER_MAX_NUM_SEQS": "8"}
    m = _load(over, name="_seedrep_attacker_pool_fm2c_explicit")
    # module initialization: GEOM is the explicit S 8 under peak_mean, SIZING is the (unfittable) recommendation kept as metadata
    assert (m.MAX_NUM_SEQS, m.GEOM.seqs_source, m.GEOM.kv_rule, m.GEOM.fits, m.P.gpu_memory_utilization) == (8, "explicit", "peak_mean", True, 0.60)
    assert m.SIZING["S"] == 64 and m.SIZING["fits"] is False and "does not fit this card at util 0.6" in m.SIZING["unfit_reason"]
    bound = m.SIZING["S_kv_mean"]
    assert 8 <= bound < 64 and f"largest S that fits under the peak_mean rule: {bound}" in m.SIZING["unfit_reason"]
    assert "NOTE the automatic recommendation S_peak 64" in m.GEOM.note and "launched at the explicit S 8 instead" in m.GEOM.note
    kw = {a.name: a for a in m._fake_modal.App.instances}[m.APP_NAME].server_kwargs
    assert kw["target_concurrency"] == 8
    # the pure derivation: strict (the automatic launch) refuses by name; non-strict (an explicit launch) records fits False
    with pytest.raises(c.PoolGeometryUnfit, match=rf"peak_mean rule.*max_num_seqs <= {bound}"):
        m.derive_pool_sizing("crescendo", 65_536, util=0.60)
    rec = m.derive_pool_sizing("crescendo", 65_536, util=0.60, strict=False)
    assert (rec["S"], rec["S_peak"], rec["fits"], rec["S_kv_mean"]) == (64, 64, False, bound) and rec["unfit_reason"]
    assert m.derive_pool_sizing("crescendo", 65_536)["fits"] is True and m.derive_pool_sizing("crescendo", 65_536)["unfit_reason"] is None
    # endpoint-record production carries the launched width and the recommendation's verdict; the certificate reader accepts it
    live = {"kind": "modal", "app_name": "seedrep-attacker-0", "app_id": "ap-11111111", "server_name": "qwen36-27b-abliterated-vllm",
            "server_id": "sv-11111111", "call_id": None, "container_id": "ta-22222222"}
    ep = m.endpoint_record_fields(
        m.P, m.GEOM, coverage={"kv_tokens_reported": 400_000, "concurrency_window": 65_536, "max_concurrency_reported": 6.1},
        serving_fields={"attacker_quantization": "none", "prefix_caching": True}, url="https://TESTHOST--seedrep-attacker-0.modal.direct",
        liveness=live, started="2026-09-19T21:00:00Z", call_id=None, server_id="sv-11111111", container_id="ta-22222222", source_sha="a" * 40)
    sz = ep["sizing"]
    assert ep["max_num_seqs"] == ep["geometry"]["max_num_seqs"] == sz["S"] == 8 and sz["explicit_override"] is True
    assert sz["slots_cover_peak"] is False and sz["kv_resident_at"]["mean"]["fits"] is True and sz["kv_resident_at"]["mean"]["S_fit"] == bound
    assert (sz["recommended"]["S"], sz["recommended"]["fits"]) == (64, False) and sz["recommended"]["unfit_reason"] == m.SIZING["unfit_reason"]
    text = json.dumps(ep, indent=2, sort_keys=True) + "\n"
    row = {"standup_id": "crescendo__s1__base__shard00", "leg": "crescendo", "pool": "seedrep-attacker-0"}
    facts = lp._smoke_check("pool_endpoint", text, path="/x/pool_endpoint.json", plan={"W": "a" * 40}, accepted_attempts=["a"], row=row, aid="a",
                            root="/x", producer=_bound(lp, text))
    assert (facts["max_num_seqs"], facts["sizing_S"], facts["slots_cover_peak"], facts["recommended_S"]) == (8, 8, False, 64)
    # the refusal's advice is about the ACTUAL S and can be followed: S = the bound imports, S = bound + 1 refuses naming the bound
    ok = _load({**over, "SEEDREP_ATTACKER_MAX_NUM_SEQS": str(bound)}, name="_seedrep_attacker_pool_fm2c_at_bound")
    assert ok.MAX_NUM_SEQS == bound and ok.GEOM.fits
    with pytest.raises(c.PoolGeometryUnfit, match=rf"peak_mean rule.*max_num_seqs <= {bound}"):
        _load({**over, "SEEDREP_ATTACKER_MAX_NUM_SEQS": str(bound + 1)}, name="_seedrep_attacker_pool_fm2c_past_bound")
    # KNOWN-BAD (unchanged): the AUTOMATIC launch at 0.60 still refuses at import -- its S IS the recommendation
    with pytest.raises(c.PoolGeometryUnfit, match=rf"peak_mean rule.*max_num_seqs <= {bound}"):
        _load({"SEEDREP_ATTACKER_GPU_MEM_UTIL": "0.60"}, name="_seedrep_attacker_pool_fm2c_auto")


# --------------------------------------------------------------------------- allocator, endpoint record, publish wait

def test_one_card_keeps_expandable_segments_and_the_default_image_bakes_it(pool):
    assert pool.alloc_conf_env(1) == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"} and pool.alloc_conf_env(2) == {}
    assert pool._IMAGE_ENV["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True" and pool._IMAGE_ENV["SEEDREP_ATTACKER_GPU"] == "H200"
    assert pool.serve_env_problem(pool._IMAGE_ENV, pool.P.tensor_parallel) is None
    # the F-M constraint is unchanged above one card (a two-card pool still bakes nothing and refuses the setting)
    assert pool.serve_env_problem({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, 2) is not None


def test_the_endpoint_record_carries_one_card_and_the_sizing_and_the_snapshot_check_reads_it(pool, lp):
    live = {"kind": "modal", "app_name": "seedrep-attacker-0", "app_id": "ap-11111111", "server_name": "qwen36-27b-abliterated-vllm",
            "server_id": "sv-11111111", "call_id": None, "container_id": "ta-22222222"}
    rec = pool.endpoint_record_fields(
        pool.P, pool.GEOM, coverage={"kv_tokens_reported": 1_060_000, "concurrency_window": 65_536, "max_concurrency_reported": 16.17},
        serving_fields={"attacker_quantization": "none", "prefix_caching": True}, url="https://TESTHOST--seedrep-attacker-0.modal.direct",
        liveness=live, started="2026-09-19T20:00:00Z", call_id=None, server_id="sv-11111111", container_id="ta-22222222", source_sha="a" * 40)
    assert (rec["gpu"], rec["tensor_parallel_size"], rec["max_model_len"], rec["max_num_seqs"], rec["pytorch_cuda_alloc_conf"]) == ("H200", 1, 65_536, 64, "expandable_segments:True")
    assert rec["geometry"]["kv_rule"] == "peak_mean" and rec["geometry"]["kv_seq_tokens"] == 5_780 and rec["geometry"]["seqs_source"] == "peak_mean"
    s = rec["sizing"]
    assert {k: s[k] for k in ("kv_rule", "S", "S_peak", "S_kv_mean", "p99_fit_S", "mean_seq_tokens", "p99_seq_tokens", "L", "slots_cover_peak", "explicit_override")} == {
        "kv_rule": "peak_mean", "S": 64, "S_peak": 64, "S_kv_mean": 139, "p99_fit_S": 24, "mean_seq_tokens": 5_780, "p99_seq_tokens": 42_230, "L": 65_536,
        "slots_cover_peak": True, "explicit_override": False}
    # slot coverage and KV residency are two facts: 64 slots admit the 51-child peak; the KV keeps 64 sequences resident at the mean, not at the p99
    assert s["kv_resident_at"] == {"mean": {"seq_tokens": 5_780, "fits": True, "S_fit": 139}, "p99": {"seq_tokens": 42_230, "fits": False, "S_fit": 24}}
    assert "covers_peak" not in s and s["kv_tokens_at_S"] == 64 * 5_780
    r = s["recommended"]
    assert (r["attack"], r["S"], r["reply_cap"], r["pinned_children"], r["S_kv_p99"], r["identity_S"], r["slots_cover_peak"]) == ("crescendo", 64, 1_024, 51, 24, 16, True)
    # the watch rules: a gauge, a counter DELTA, two latency histogram deltas; each with a window, a corrective threshold and an action
    wm = {m["metric"]: m for m in rec["watch_metrics"]}
    assert list(wm) == list(pool.WATCH_METRIC_NAMES) and rec["watch_metrics"] == [dict(m) for m in pool.WATCH_METRICS]
    assert wm["vllm:num_requests_waiting"]["read"] == "gauge" and wm["vllm:num_preemptions_total"]["read"] == "counter_delta"
    assert wm["vllm:request_queue_time_seconds"]["read"] == wm["vllm:e2e_request_latency_seconds"]["read"] == "histogram_delta_p95"
    for m in rec["watch_metrics"]:
        assert m["window_s"] == pool.WATCH_WINDOW_S == 600 and m["threshold"] and "lower S" in m["action"] and "add a pool" in m["action"]
    assert "delta > 0" in wm["vllm:num_preemptions_total"]["threshold"] and "p95 > 30 s" in wm["vllm:request_queue_time_seconds"]["threshold"]
    assert "timeout" in wm["vllm:e2e_request_latency_seconds"]["threshold"] and "10 %" in wm["vllm:num_requests_waiting"]["threshold"]
    # the rule's caveat rides on the record: the call-weighted mean is an ESTIMATE of resident occupancy; slots are not KV residency
    assert s["caveat"] == pool.PEAK_MEAN_CAVEAT and "ESTIMATE" in s["caveat"] and "not a measured resident-request mean" in s["caveat"]
    assert "long-context phases together" in s["caveat"] and "slots_cover_peak" in s["caveat"] and "kv_resident_at" in s["caveat"]
    ep = c.PoolEndpoint.from_dict(rec)
    assert ep.identity()["gpu"] == "H200" and ep.identity()["tensor_parallel_size"] == 1 and ep.max_num_seqs == 64
    # every key the certificate's pool_endpoint check requires is present, and the check reads the one-card facts
    assert not [k for k in lp.POOL_ENDPOINT_KEYS if k not in rec]
    text = json.dumps(rec, indent=2, sort_keys=True) + "\n"
    row = {"standup_id": "crescendo__s1__base__shard00", "leg": "crescendo", "pool": "seedrep-attacker-0"}
    facts = lp._smoke_check("pool_endpoint", text, path="/x/pool_endpoint.json", plan={"W": "a" * 40}, accepted_attempts=["a"], row=row, aid="a",
                            root="/x", producer={lp.POOL_ENDPOINT_SNAPSHOT_REL: {"sha256": __import__("hashlib").sha256(text.encode()).hexdigest(),
                                                                                  "writer": lp.PRODUCER_WRITERS["pool_endpoint"], "tree_sha": "a" * 40, "rc": 0}})
    assert (facts["gpu"], facts["tensor_parallel_size"], facts["max_num_seqs"], facts["max_model_len"], facts["pytorch_cuda_alloc_conf"]) == ("H200", 1, 64, 65_536, "expandable_segments:True")
    assert (facts["sizing_S"], facts["kv_rule"], facts["slots_cover_peak"], facts["recommended_S"]) == (64, "peak_mean", True, 64)


def _bound(lp, text: str) -> dict:
    return {lp.POOL_ENDPOINT_SNAPSHOT_REL: {"sha256": __import__("hashlib").sha256(text.encode()).hexdigest(), "writer": lp.PRODUCER_WRITERS["pool_endpoint"],
                                            "tree_sha": "a" * 40, "rc": 0}}


def test_an_operator_override_records_the_launched_width_and_the_reader_refuses_a_record_that_disagrees(lp):
    """fm2 (reproduced): with SEEDREP_ATTACKER_MAX_NUM_SEQS=20 on the actorattack pool the Server launched 20 slots while the
    record's `sizing` came from the automatic default (S 158, covers_peak true against 126 children) and the certificate accepted the
    contradiction. Now: `sizing` is the LAUNCHED geometry's, the recommendation rides under `recommended`, and the reader refuses a
    record whose launched width disagrees with its sizing.S (or its geometry.max_num_seqs)."""
    m = _load({"SEEDREP_ATTACKER_POOL_INDEX": "2", "SEEDREP_ATTACKER_ATTACK": "actorattack", "SEEDREP_ATTACKER_MAX_NUM_SEQS": "20"},
              name="_seedrep_attacker_pool_fm2_override")
    assert (m.MAX_NUM_SEQS, m.GEOM.seqs_source, m.GEOM.kv_rule) == (20, "explicit", "peak_mean")
    live = {"kind": "modal", "app_name": "seedrep-attacker-2", "app_id": "ap-11111111", "server_name": "qwen36-27b-abliterated-vllm",
            "server_id": "sv-11111111", "call_id": None, "container_id": "ta-22222222"}
    rec = m.endpoint_record_fields(
        m.P, m.GEOM, coverage={"kv_tokens_reported": 1_060_000, "concurrency_window": 65_536, "max_concurrency_reported": 16.17},
        serving_fields={"attacker_quantization": "none", "prefix_caching": True}, url="https://TESTHOST--seedrep-attacker-2.modal.direct",
        liveness=live, started="2026-09-19T20:00:00Z", call_id=None, server_id="sv-11111111", container_id="ta-22222222", source_sha="a" * 40)
    s = rec["sizing"]
    assert rec["max_num_seqs"] == rec["geometry"]["max_num_seqs"] == s["S"] == 20 and s["explicit_override"] is True and s["seqs_source"] == "explicit"
    assert s["slots_cover_peak"] is False and s["pinned_children"] == 126          # 20 slots do NOT admit the 126-child peak -- said so
    assert s["kv_resident_at"]["mean"]["fits"] is True and s["kv_resident_at"]["p99"]["fits"] is True      # 20 x p99 (5,214) is resident too
    assert s["kv_tokens_at_S"] == max(20 * 1_864, 65_536) == 65_536                # never below one full window
    assert (s["recommended"]["S"], s["recommended"]["S_peak"], s["recommended"]["slots_cover_peak"]) == (158, 158, True)
    # the reader accepts the honest record and carries the launched facts
    text = json.dumps(rec, indent=2, sort_keys=True) + "\n"
    row = {"standup_id": "actorattack__s1__base__shard00", "leg": "actorattack", "pool": "seedrep-attacker-2"}
    facts = lp._smoke_check("pool_endpoint", text, path="/x/pool_endpoint.json", plan={"W": "a" * 40}, accepted_attempts=["a"], row=row, aid="a",
                            root="/x", producer=_bound(lp, text))
    assert (facts["max_num_seqs"], facts["sizing_S"], facts["slots_cover_peak"], facts["recommended_S"]) == (20, 20, False, 158)
    # KNOWN-BAD (the P2 shape): the same record with the automatic sizing under a 20-slot launch is refused by name
    bad = json.dumps({**rec, "sizing": {**s, "S": 158, "slots_cover_peak": True}}, indent=2, sort_keys=True) + "\n"
    with pytest.raises(ValueError, match="launched max_num_seqs 20 disagrees with its sizing.S 158"):
        lp._smoke_check("pool_endpoint", bad, path="/x/pool_endpoint.json", plan={"W": "a" * 40}, accepted_attempts=["a"], row=row, aid="a", root="/x", producer=_bound(lp, bad))
    # KNOWN-BAD: a geometry block of another width; a sizing that is not an object
    bad2 = json.dumps({**rec, "geometry": {**rec["geometry"], "max_num_seqs": 158}}, indent=2, sort_keys=True) + "\n"
    with pytest.raises(ValueError, match="disagrees with geometry.max_num_seqs 158"):
        lp._smoke_check("pool_endpoint", bad2, path="/x/pool_endpoint.json", plan={"W": "a" * 40}, accepted_attempts=["a"], row=row, aid="a", root="/x", producer=_bound(lp, bad2))
    bad3 = json.dumps({**rec, "sizing": "peak_mean"}, indent=2, sort_keys=True) + "\n"
    with pytest.raises(ValueError, match="sizing is not an object"):
        lp._smoke_check("pool_endpoint", bad3, path="/x/pool_endpoint.json", plan={"W": "a" * 40}, accepted_attempts=["a"], row=row, aid="a", root="/x", producer=_bound(lp, bad3))
    # a record without the sizing block still reads, its sizing facts None -- a carried row is not refused for predating the block
    old = {k: v for k, v in rec.items() if k != "sizing"}
    old_text = json.dumps(old, indent=2, sort_keys=True) + "\n"
    f_old = lp._smoke_check("pool_endpoint", old_text, path="/x/pool_endpoint.json", plan={"W": "a" * 40}, accepted_attempts=["a"], row=row, aid="a", root="/x", producer=_bound(lp, old_text))
    assert f_old["sizing_S"] is None and f_old["max_num_seqs"] == 20


def test_the_endpoint_publish_wait_is_sized_for_one_card_above_the_w13_floor(pool):
    src = (REPO / "scripts/modal/seedrep_attacker_pool.py").read_text()
    tree = ast.parse(src)
    consts = {t.id: ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and t.id in ("ENDPOINT_PUBLISH_WAIT_S", "ONE_CARD_STARTUP_ESTIMATE_S")}
    assert consts["ENDPOINT_PUBLISH_WAIT_S"] == pool.ENDPOINT_PUBLISH_WAIT_S == 2400 == 2 * consts["ONE_CARD_STARTUP_ESTIMATE_S"] + 200
    assert 1800 <= pool.ENDPOINT_PUBLISH_WAIT_S < 3600           # the floor; below the Server's own readiness bound
    # the estimate is from the measured TP-2 startup (903 s ready; weights 97.9 s, compile 329 s, engine init 741 s) with the one-card
    # weight load doubled -- recorded beside the constant
    assert "903 s" in src and "97.9 s" in src and "329 s" in src
