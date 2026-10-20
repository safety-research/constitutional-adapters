"""scripts/served_decode_sweep.py + scripts/geom_sweep_analyze.py +
scripts/sbatch/geom_sweep.sbatch — CPU-only (no vllm, aiohttp, torch, GPU).

The per-model budget sweep. Covered: the model
table equals geometry.MODEL_KEYS and every row's config exists, serves that HF
id, and is either pinned to the row's revision or left null (llama3b), in
which case ``resolve_serving_config`` materializes a pinned copy and refuses
a disagreeing pin; the budget walk (ladder upward to the first refusal, then
exactly ONE bisection; a first-rung failure closes with no pass); launch
failure classification; the batch ladder and permitted S; the knee and the
summary builder on synthetic rows (tok/s at the old S = 24, speedup,
duty); the Prometheus parser + histogram quantile + in-window client stats +
dmon parsing; ``fit_repeats`` against a fake tokenizer; the analyzer's paste
block round-trips through geometry's constructors; the sbatch: ``bash -n``,
the hard rules, the array header, the MODELS mirror, the guards and traps;
the standup's ``bring_up``/``Standup`` extraction keeps ``main``'s CLI.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sisteer.capeval.serving import geometry as sg

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sw():
    return _load("_served_decode_sweep", "scripts/served_decode_sweep.py")


@pytest.fixture(scope="module")
def an():
    return _load("_geom_sweep_analyze", "scripts/geom_sweep_analyze.py")


SBATCH = REPO / "scripts/sbatch/geom_sweep.sbatch"


# --------------------------------------------------------------------------- #
# the model table                                                              #
# --------------------------------------------------------------------------- #


def test_model_table_matches_geometry_keys_and_configs(sw):
    assert tuple(sw.MODELS) == sg.MODEL_KEYS == sw.MODEL_ORDER
    for key, ms in sw.MODELS.items():
        assert ms.key == key
        cfg_path = REPO / ms.cfg
        assert cfg_path.is_file(), ms.cfg
        cfg = yaml.safe_load(cfg_path.read_text())
        assert cfg["model"] == ms.hf, (key, cfg["model"], ms.hf)
        assert re.fullmatch(r"[0-9a-f]{40}", ms.rev), (key, ms.rev)
        assert cfg.get("model_revision") in (None, ms.rev), (key, cfg.get("model_revision"))
        # the cell the steered arm prefers must be an arm the config declares
        assert "psr_kl_all" in cfg.get("class_a_arms", []), key
        assert set(cfg["si_ids"]) >= {"safety_refusal", "si_constitution"}, key
    # llama3b is THE null-pinned row (its config pins at first fetch)
    assert yaml.safe_load((REPO / sw.MODELS["llama3b"].cfg).read_text())["model_revision"] is None
    assert sw.hf_snapshot_relpath(sw.MODELS["qwen35_4b"]) == (
        "hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")


def test_resolve_serving_config_pins_null_and_refuses_disagreement(sw, tmp_path):
    ms = sw.MODELS["llama3b"]
    out = sw.resolve_serving_config(ms, REPO, tmp_path)
    assert out.parent == tmp_path and "pinned" in out.name
    pinned = yaml.safe_load(out.read_text())
    assert pinned["model_revision"] == ms.rev and pinned["model"] == ms.hf
    base = yaml.safe_load((REPO / ms.cfg).read_text())
    assert {k: v for k, v in pinned.items() if k != "model_revision"} == \
           {k: v for k, v in base.items() if k != "model_revision"}
    # a pinned config is handed back as-is
    ms8 = sw.MODELS["llama8b"]
    assert sw.resolve_serving_config(ms8, REPO, tmp_path) == REPO / ms8.cfg
    # a row whose config names another model / pin refuses
    import dataclasses
    with pytest.raises(ValueError):
        sw.resolve_serving_config(dataclasses.replace(ms8, hf="other/model"), REPO, tmp_path)
    with pytest.raises(ValueError):
        sw.resolve_serving_config(dataclasses.replace(ms8, rev="0" * 40), REPO, tmp_path)


def test_shell_env_lines_are_eval_safe(sw):
    lines = sw.shell_env_lines(sw.MODELS["gemma4e2b"], Path("/r"))
    keys = [ln.split("=", 1)[0] for ln in lines]
    assert keys == ["GEOM_MODEL_KEY", "GEOM_HF_ID", "GEOM_HF_REV", "GEOM_CONFIG", "GEOM_BANK",
                    "GEOM_HF_SNAPSHOT_REL"]
    proc = subprocess.run(["bash", "-c", "\n".join(lines) + '\necho "$GEOM_HF_ID|$GEOM_CONFIG|$GEOM_HF_SNAPSHOT_REL"'],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0
    assert proc.stdout.strip() == (
        "google/gemma-4-E2B-it|/r/configs/two_si_gemma4e2b_fits.yaml|"
        "hub/models--google--gemma-4-E2B-it/snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7")


def test_select_steered_cell_prefers_constitution_then_safety(sw, tmp_path):
    assert sw.select_steered_cell(None) is None
    assert sw.select_steered_cell(tmp_path / "absent") is None
    bank = tmp_path / "bank"
    (bank / "safety_refusal__psr_kl_all").mkdir(parents=True)
    assert sw.select_steered_cell(bank) is None           # dir without an artifact
    (bank / "safety_refusal__psr_kl_all" / "object.pt").write_bytes(b"x")
    assert sw.select_steered_cell(bank) == "safety_refusal__psr_kl_all"
    (bank / "si_constitution__psr_kl_all").mkdir()
    (bank / "si_constitution__psr_kl_all" / "object.pt").write_bytes(b"x")
    assert sw.select_steered_cell(bank) == "si_constitution__psr_kl_all"


# --------------------------------------------------------------------------- #
# the budget walk                                                              #
# --------------------------------------------------------------------------- #


def _walk(sw, wall: int, L: int = 8192, ladder=None):
    ladder = ladder or sw.BUDGET_LADDER
    probes = []
    while (n := sw.next_budget_probe(probes, ladder, L)) is not None:
        probes.append({"product": n, "max_num_seqs": n // L, "max_model_len": L,
                       "status": "pass" if n < wall else "kv_under_coverage"})
    return probes


def test_budget_ladder_is_the_seven_products_at_S_48_to_192(sw):
    assert sw.BUDGET_LADDER == (393_216, 524_288, 655_360, 786_432, 1_048_576, 1_310_720, 1_572_864)
    assert [sw.ladder_seqs(p, 8192) for p in sw.BUDGET_LADDER] == [48, 64, 80, 96, 128, 160, 192]
    assert sw.BUDGET_LADDER[0] == sg.BATCHED_TOKEN_BUDGET
    with pytest.raises(ValueError):
        sw.ladder_seqs(393_216, 10000)


def test_budget_walk_to_first_refusal_then_one_bisection(sw):
    # a llama-like wall between 48 and 64 (the certified 393,216 passes, 524,288 refuses)
    probes = _walk(sw, wall=524_288)
    assert [(p["product"], p["status"]) for p in probes] == [
        (393_216, "pass"), (524_288, "kv_under_coverage"),
        (56 * 8192, "pass")]                                 # the ONE bisection at S=56
    br = sw.budget_bracket(probes)
    assert br["largest_passing"]["product"] == 458_752 and br["first_refused"] == 524_288
    assert br["wall_reached"] and br["n_probes"] == 3
    # the wall just above the bisection point: 56 x 8192 refuses, 48 stands
    probes = _walk(sw, wall=458_752)
    assert [(p["product"], p["status"]) for p in probes] == [
        (393_216, "pass"), (524_288, "kv_under_coverage"), (458_752, "kv_under_coverage")]
    br = sw.budget_bracket(probes)
    assert br["largest_passing"]["product"] == 393_216 and br["first_refused"] == 458_752
    # a qwen-like model: the wall above 1,441,792 -> bisect between 160 and 192, and it passes
    probes = _walk(sw, wall=1_500_000)
    prods = [p["product"] for p in probes]
    assert prods[:6] == list(sw.BUDGET_LADDER[:6]) and prods[6] == 1_572_864
    assert prods[7] == 176 * 8192                               # (160 + 192) // 2
    assert probes[7]["status"] == "pass" and len(probes) == 8  # once, even though it passed
    br = sw.budget_bracket(probes)
    assert br["largest_passing"]["product"] == 176 * 8192 and br["first_refused"] == 1_572_864
    # everything passes: the ladder top, no wall, no bisection
    probes = _walk(sw, wall=10 ** 9)
    assert [p["product"] for p in probes] == list(sw.BUDGET_LADDER)
    br = sw.budget_bracket(probes)
    assert br["largest_passing"]["product"] == 1_572_864 and not br["wall_reached"]
    # the first rung fails: closed with no passing probe (nothing is guessed)
    probes = _walk(sw, wall=1)
    assert len(probes) == 1 and sw.budget_bracket(probes)["largest_passing"] is None
    # adjacent S (no midpoint) -> no bisection
    probes = _walk(sw, wall=50 * 8192, ladder=(48 * 8192, 49 * 8192, 50 * 8192))
    assert [p["product"] for p in probes] == [48 * 8192, 49 * 8192, 50 * 8192]


def test_paste_record_and_classification(sw):
    probes = _walk(sw, wall=1_500_000)
    for p in probes:
        p["kv_tokens"] = p["product"] + 5000
        p["kv_kind"] = "hybrid_max_concurrency_x_L"
        p["log_max_concurrency_x"] = 200.5
    br = sw.budget_bracket(probes)
    rec = sw.paste_record(model_key="qwen35_4b", bracket=br, max_model_len=8192, gpu_mem_util=0.9,
                          job_id="123_5", measured_on="2026-09-04", results_dir="/r/qwen35_4b",
                          node="gpu-3")
    assert rec["batched_token_budget"] == 176 * 8192 and rec["certified"] is False
    assert rec["basis"]["max_num_seqs"] == 176 and rec["basis"]["first_refused"] == 1_572_864
    assert rec["basis"]["kv_tokens_reported"] == 176 * 8192 + 5000
    # ...and it constructs a real ModelBudget
    b = sg.ModelBudget(model_key=rec["model_key"], batched_token_budget=rec["batched_token_budget"],
                       certified=rec["certified"], basis=sg.BudgetBasis(**rec["basis"]))
    assert b.seqs_at(16384) == 88
    assert sw.paste_record(model_key="x", bracket={"largest_passing": None}, max_model_len=8192,
                           gpu_mem_util=0.9, job_id=None, measured_on="d", results_dir="r",
                           node="n") is None
    # failure classes
    assert sw.classify_launch_failure("worker 0: KV cache holds 400 tokens < the 500-token budget") == "kv_under_coverage"
    assert sw.classify_launch_failure("hybrid KV cache supports only 30.5-way concurrency") == "kv_under_coverage"
    assert sw.classify_launch_failure("did not become ready within 900s", "torch.OutOfMemoryError: CUDA out of memory") == "oom"
    assert sw.classify_launch_failure("vllm serve did not become ready within 900s", "") == "not_ready"
    assert sw.classify_launch_failure("something else", "") == "launch_error"


def test_kv_facts_and_log_scrape(sw):
    uni = [{"kv_num_cache_groups": 1, "kv_num_gpu_blocks": 36000, "kv_block_size": 16}]
    f = sw.kv_facts_from_reports(uni, 16384)
    assert f["kv_tokens"] == 576_000 and f["kv_kind"] == "uniform_blocks_x_block_size"
    assert abs(f["kv_max_concurrency_at_L"] - 576_000 / 16384) < 1e-9
    hyb = [{"kv_num_cache_groups": 2, "kv_max_concurrency": 128.9}]
    f = sw.kv_facts_from_reports(hyb, 16384)
    assert f["kv_tokens"] == int(128.9 * 16384) and f["kv_kind"] == "hybrid_max_concurrency_x_L"
    assert sw.kv_facts_from_reports([], 16384)["kv_tokens"] is None
    assert sw.kv_facts_from_reports([{"kv_num_cache_groups": 1}], 16384)["kv_kind"] == "unreadable"
    log = ("INFO Memory profiling takes 5.35 seconds. Total non-KV-cache memory: 21.44GiB; "
           "torch peak memory increase: 4.81GiB; non-torch forward increase memory: 0.69GiB; "
           "weights memory: 15.94GiB.\nINFO Available KV cache memory: 104.39 GiB\n"
           "INFO GPU KV cache size: 2,113,536 tokens\n"
           "INFO Maximum concurrency for 16,384 tokens per request: 129.00x\nother line\n")
    s = sw.scrape_server_log(log)
    assert s["log_kv_cache_tokens"] == 2_113_536 and s["log_max_concurrency_x"] == 129.0
    assert s["log_max_concurrency_len"] == 16384 and s["log_kv_memory_gib"] == 104.39
    assert s["log_weights_gib"] == 15.94 and s["log_torch_peak_gib"] == 4.81
    assert s["log_non_kv_gib"] == 21.44 and not s["log_oom"]
    assert len(s["memory_lines"]) == 4
    assert sw.scrape_server_log("torch.OutOfMemoryError: CUDA out of memory")["log_oom"]


# --------------------------------------------------------------------------- #
# the sweep ladder, knee, summary                                              #
# --------------------------------------------------------------------------- #


def test_batch_ladder_permitted_seqs_and_windows(sw):
    assert sw.BATCH_LADDER == (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
    assert sw.OLD_S in sw.BATCH_LADDER                        # the old S=24 comparison rung
    assert sw.batch_ladder(45) == (8, 16, 24, 32)
    assert sw.batch_ladder(24) == (8, 16, 24)
    assert sw.batch_ladder(5) == (5,)
    assert sw.batch_ladder(600) == sw.BATCH_LADDER
    assert sw.permitted_seqs(393_216, 16384) == 24
    assert sw.permitted_seqs(2_113_536, 1536) == 512          # the cap
    assert sw.permitted_seqs(2_113_536, 16384) == 129
    assert sw.permitted_seqs(1000, 16384) == 1
    assert sw.DEFAULT_WINDOWS == (8704, 7424, 11520, 16384, 1536)
    # every default window IS a table window, resolvable by the standup
    for L in sw.DEFAULT_WINDOWS:
        labels = sw.window_labels(L)
        assert labels, L
        key = sw.eval_key_for_window(L)
        assert sg.geometry_for(key).max_model_len == L
        assert not sg.geometry_for(key).requires_recertification
    assert sw.eval_key_for_window(7424) == "capability"       # top-level beats sub-entry
    assert sw.eval_key_for_window(1536) == "capability:short"
    assert sw.eval_key_for_window(16384) == "petri"
    assert sw.eval_key_for_window(12345) is None


def _rows(sw, L, arm, S, curve):
    return [{"model": "m", "window_L": L, "window_labels": list(sw.window_labels(L)), "S": S,
             "budget": 1_000_000, "arm": arm, "cell_id": ("none" if arm == "passthrough" else "c"),
             "B": b, "tok_s": t, "req_s": t / 512, "lat_p50_s": b / 20, "lat_p95_s": b / 10,
             "ttft_p50_s_hist": 0.1, "running_mean": b * 0.95, "running_max": b,
             "preemptions_delta": (1 if b >= 128 else 0), "sm_mean": 50 + b / 4, "sm_p10": 40,
             "engine_gen_tok_s": t * 1.01, "n_completed": 10, "n_errors": 0}
            for b, t in curve]


def test_knee_and_summary_on_synthetic_rows(sw):
    curve = [(8, 1000.0), (16, 1900.0), (24, 2600.0), (32, 3200.0), (48, 4000.0),
             (64, 4600.0), (96, 5000.0), (128, 5100.0)]
    # 64 -> 4600 >= 0.9 x 5100 = 4590 is the FIRST rung at >= 90% of the max
    assert sw.knee_batch(curve) == 64
    assert sw.knee_batch(curve, 0.90) == 64
    assert sw.knee_batch(curve, 0.95) == 96
    assert sw.knee_batch([]) is None
    assert sw.knee_batch([(8, None), (16, 5.0)]) == 16
    rows = _rows(sw, 7424, "passthrough", 128, curve) + _rows(sw, 7424, "steered", 128, [
        (8, 900.0), (16, 1700.0), (24, 2300.0), (32, 2800.0)])
    summ = sw.summarize_rows(rows, budget=1_000_000)
    g = summ["groups"]["7424:passthrough"]
    assert g["knee_B"] == 64 and g["tok_s_at_knee"] == 4600.0 and g["max_tok_s"] == 5100.0
    assert g["tok_s_at_B24"] == 2600.0
    assert abs(g["speedup_knee_vs_B24"] - 4600.0 / 2600.0) < 1e-9
    assert g["duty_at_knee"] == 50 + 64 / 4 and g["S"] == 128
    assert g["preemptions_total"] == 1 and g["n_points"] == 8
    assert [t["B"] for t in g["table"]] == [8, 16, 24, 32, 48, 64, 96, 128]
    assert g["window_labels"] == ["capability", "families_prompted", "bon"]
    st = summ["groups"]["7424:steered"]
    # 4 of the 8 rungs at S=128: INCOMPLETE -> a provisional knee only (review R3)
    assert st["complete"] is False and st["missing_B"] == [48, 64, 96, 128]
    assert st["knee_B"] is None and st["knee_provisional_B"] == 32
    assert st["tok_s_at_B24"] == 2300.0 and st["cell_id"] == "c" and st["speedup_knee_vs_B24"] is None
    assert g["complete"] and g["missing_B"] == []
    assert summ["windows"] == [7424] and summ["arms"] == ["passthrough", "steered"]
    assert summ["n_rows"] == 12 and summ["partial"] and summ["groups_incomplete"] == ["7424:steered"]
    assert summ["old_S"] == 24
    # no B=24 rung run -> no speedup, not an error
    summ = sw.summarize_rows(_rows(sw, 16384, "passthrough", 12, [(8, 1.0), (12, 2.0)]),
                             partial=True, missing=[(16384, "steered", 8)])
    g = summ["groups"]["16384:passthrough"]
    assert g["tok_s_at_B24"] is None and g["speedup_knee_vs_B24"] is None and g["knee_B"] == 12
    assert summ["partial"] and summ["missing_points"] == [[16384, "steered", 8]]


def test_prometheus_histogram_and_window_stats(sw):
    text = """# HELP vllm:num_requests_running x
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m"} 45.0
vllm:num_requests_waiting{model_name="m"} 3
vllm:num_preemptions_total{model_name="m"} 7
vllm:generation_tokens_total{model_name="m"} 123456
vllm:kv_cache_usage_perc{model_name="m"} 0.83
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="m"} 10
vllm:time_to_first_token_seconds_bucket{le="0.5",model_name="m"} 30
vllm:time_to_first_token_seconds_bucket{le="+Inf",model_name="m"} 40
vllm:time_to_first_token_seconds_sum{model_name="m"} 12.5
vllm:time_to_first_token_seconds_count{model_name="m"} 40
"""
    p = sw.parse_prometheus(text)
    assert sw.metric_sum(p, "vllm:num_requests_running") == 45.0
    assert sw.metric_sum(p, "vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc") == 0.83
    assert sw.metric_sum(p, "vllm:nothing") is None
    b = sw.hist_buckets(p, "vllm:time_to_first_token_seconds")
    assert b[0] == (0.1, 10.0) and b[1] == (0.5, 30.0) and b[-1][1] == 40.0
    # median of 40 obs: rank 20 lies in the (0.1, 0.5] bucket at 10/20 -> 0.3
    assert abs(sw.hist_quantile(b, 0.5) - 0.3) < 1e-9
    assert sw.hist_quantile(b, 0.99) == 0.5                    # the +Inf bucket -> its lower edge
    assert sw.hist_quantile([], 0.5) is None and sw.hist_quantile([(0.1, 0.0), (float("inf"), 0.0)], 0.5) is None
    # client-side in-window stats: only completions inside [w0, w1] count
    recs = [(0.0, 5.0, 512, True, None), (1.0, 12.0, 512, True, None), (2.0, 15.0, 0, False, "boom"),
            (3.0, 18.0, 512, True, None), (4.0, 31.0, 512, True, None)]
    st = sw.in_window_stats(recs, 10.0, 30.0)
    assert st["n_completed"] == 2 and st["n_errors"] == 1 and st["completion_tokens"] == 1024
    assert abs(st["tok_s"] - 1024 / 20) < 1e-9 and abs(st["req_s"] - 0.1) < 1e-9
    assert st["lat_p50_s"] == 11.0 and st["lat_p95_s"] == 15.0 and st["first_error"] == "boom"
    assert sw.percentile([3, 1, 2], 0.5) == 2 and sw.percentile([], 0.5) is None
    # dmon rows
    assert sw.parse_dmon_line("# gpu   sm   mem") is None
    assert sw.parse_dmon_line("    3    97    41     0     0") == (3, 97)
    assert sw.parse_dmon_line("    3     -     -     0     0") is None


def test_fit_repeats_lands_at_or_under_the_target(sw):
    calls = []

    def count_fn(k):
        calls.append(k)
        return 7 + 19 * k                                       # 7 template tokens + 19 per repeat

    k, c = sw.fit_repeats(count_fn, 1000)
    assert c <= 1000 and 7 + 19 * (k + 1) > 1000 and k == 52 and c == 995
    assert len(calls) <= 8
    k, c = sw.fit_repeats(lambda k: 7 + 19 * k, 30)
    assert (k, c) == (1, 26)
    with pytest.raises(ValueError):
        sw.fit_repeats(count_fn, 0)
    assert sw.gpu_selector.__doc__ and "never set" in sw.gpu_selector.__doc__


# --------------------------------------------------------------------------- #
# the analyzer                                                                 #
# --------------------------------------------------------------------------- #


def _paste(key, budget, L=8192):
    return {"model_key": key, "batched_token_budget": budget, "certified": False,
            "basis": {"method": "largest S x L passing", "max_model_len": L,
                      "max_num_seqs": budget // L, "gpu_memory_utilization": 0.9,
                      "kv_tokens_reported": budget + 4096, "probe_job_id": "555_3",
                      "measured_on": "2026-09-04", "source": "/r/budget.json",
                      "first_refused": budget + 8192 * 16, "note": "kv facts: uniform; 'quoted'"}}


def test_paste_block_round_trips_through_geometry_constructors(an):
    block = an.format_paste_block([_paste("qwen35_4b", 1_441_792), _paste("gemma4e4b", 786_432)])
    assert block.index("gemma4e4b") < block.index("qwen35_4b")   # MODEL_KEYS order
    ns = {"ModelBudget": sg.ModelBudget, "BudgetBasis": sg.BudgetBasis}
    entries = eval("(\n" + block + "\n)", ns)                    # exactly what geometry.py holds
    assert [e.model_key for e in entries] == ["gemma4e4b", "qwen35_4b"]
    q = entries[1]
    assert q.batched_token_budget == 1_441_792 and q.basis.max_num_seqs == 176
    assert q.basis.kv_tokens_reported == 1_441_792 + 4096 and q.basis.first_refused == 1_441_792 + 131_072
    assert q.basis.probe_job_id == "555_3" and q.basis.note == "kv facts: uniform; 'quoted'"
    assert not q.certified and q.seqs_at(16384) == 88
    assert "1_441_792" in block                                  # readable literals
    full = an.paste_block_with_markers([_paste("qwen35_4b", 1_441_792)])
    assert full.startswith(an.PASTE_BEGIN) and full.rstrip().endswith(an.PASTE_END)
    assert "_MEASURED_BUDGETS: Tuple[ModelBudget, ...] = (" in full
    # the markers are the ones geometry.py carries
    src = (REPO / "src/sisteer/capeval/serving/geometry.py").read_text()
    assert an.PASTE_BEGIN in src and an.PASTE_END in src
    assert "<no measured entries yet>" in an.format_paste_block([])


def test_analyzer_tables_and_loading(an, sw, tmp_path):
    root = tmp_path / "geom-sweep"
    d = root / "qwen35_4b"
    d.mkdir(parents=True)
    curve = [(8, 1000.0), (16, 1900.0), (24, 2600.0), (32, 3200.0), (48, 4000.0), (64, 4600.0)]
    rows = _rows(sw, 7424, "passthrough", 64, curve) + _rows(sw, 16384, "passthrough", 24, curve[:3])
    summ = sw.summarize_rows(rows, budget=1_441_792)
    summ["model_key"] = "qwen35_4b"
    import json
    (d / "sweep_summary.json").write_text(json.dumps(summ))
    (d / "budget.json").write_text(json.dumps({
        "model_key": "qwen35_4b", "max_model_len": 8192, "job_id": "555_3", "node": "gpu-3",
        "measured_on": "2026-09-04", "complete": True, "first_refused": 1_572_864,
        "largest_passing": {"product": 1_441_792, "max_num_seqs": 176, "kv_tokens": 1_500_000},
        "probes": [], "paste": _paste("qwen35_4b", 1_441_792)}))
    (root / "glm46vflash").mkdir()                              # a model with no results yet
    res = an.load_results(root)
    assert set(res) == {"qwen35_4b", "glm46vflash"}
    assert res["glm46vflash"] == {"budget": None, "summary": None, "rows": []}
    md = an.markdown_tables(res)
    assert "## qwen35_4b" in md and "**1,441,792**" in md and "x3.67" in md   # vs 393,216
    line = next(l for l in md.splitlines() if l.startswith("| 7424 |"))
    cells = [c.strip() for c in line.strip("|").split("|")]
    # window, evals, arm, S permitted, old S, knee B, tok/s@knee, tok/s@B=24, speedup, duty, lat, preempt, points
    assert cells[0] == "7424" and cells[2] == "passthrough" and cells[3] == "64" and cells[4] == "52"
    assert cells[5] == "64" and cells[6] == "4,600" and cells[7] == "2,600" and cells[8] == "x1.77"
    assert "## glm46vflash" in md and "(no budget.json)" in md
    # the CLI end to end (paste-only, no figure)
    rc = an.main(["--results-root", str(root), "--paste-only"])
    assert rc == 0
    try:
        import matplotlib  # noqa: F401
    except Exception:
        return
    p = an.figure_grid(res, tmp_path / "grid.png")
    assert p is not None and p.is_file() and p.stat().st_size > 1000


# --------------------------------------------------------------------------- #
# the sbatch                                                                   #
# --------------------------------------------------------------------------- #


def test_sbatch_parses_and_keeps_the_hard_rules():
    proc = subprocess.run(["bash", "-n", str(SBATCH)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    txt = SBATCH.read_text()
    directives = [l for l in txt.splitlines() if l.startswith("#SBATCH")]
    active = directives + [l for l in txt.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    assert not any("--export=" in l for l in active)              # hard rule 1
    assert not any("--mem=0" in l for l in active)                # hard rule 2
    assert not re.search(r"^\s*(export\s+)?CUDA_VISIBLE_DEVICES\s*=", txt, re.M)  # hard rule 3 (read only)
    assert not any(re.search(r"--qos=\S*dev", l) for l in directives)
    assert re.search(r"^#SBATCH --job-name=geom-sweep-\d{4}-\d{2}-\d{2}$", txt, re.M)
    assert "#SBATCH --qos=high" in txt and "#SBATCH --gres=gpu:1" in txt
    assert "#SBATCH --cpus-per-task=16" in txt and "#SBATCH --mem=128G" in txt
    assert re.search(r"^#SBATCH --time=0-03:00:00", txt, re.M)
    assert re.search(r"^#SBATCH --array=0-7\b", txt, re.M)
    assert "#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out" in txt
    assert "#SBATCH --open-mode=append" in txt and "#SBATCH --no-requeue" in txt
    assert "#SBATCH --signal=B:TERM@120" in txt
    assert not re.search(r"^\s*for .*sbatch", txt, re.M)          # never a submit loop


def test_sbatch_body_scaffolding_and_guards(sw):
    txt = SBATCH.read_text()
    body = [l for l in txt.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    assert 'source "$SHARED/scripts/hf_env.sh"' in txt
    for env in ("NCCL_NVLS_ENABLE=0",
                'PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"', "HF_HUB_OFFLINE=1"):
        assert env in txt, env
    assert "NCCL_SOCKET_IFNAME=" not in txt     # the socket interface is the site's (its nccl.conf), never exported
    assert "cleanup() { _gs_handled=1; kill -TERM -$$ 2>/dev/null; wait; }" in txt
    assert "trap cleanup SIGTERM SIGINT SIGQUIT" in txt and "trap finish EXIT" in txt
    assert 'export FIGS_DIR="$SHARED/exp/figs/$SLURM_JOB_NAME/$MODEL"' in txt
    assert '"$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR"' in txt
    assert '> "$FIGS_DIR/RESULT"' in txt and "elapsed_s=$SECONDS" in txt
    # the MODELS array mirrors the Python table, in order (array index decode)
    m = re.search(r"^MODELS=\(([^)]*)\)", txt, re.M)
    assert m and m.group(1).split() == list(sw.MODELS) == list(sg.MODEL_KEYS)
    assert 'MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"' in txt
    assert "SLURM_ARRAY_TASK_ID is unset" in txt
    # the node-local requirements-hash venv, bootstrapped under flock, out-of-lockfile tagged
    assert 'UV_PROJECT_ENVIRONMENT="/home/$USER/venvs/sisteer-geom-sweep-vllm-$REQ_HASH"' in txt
    assert "flock -x 9" in txt and "geom-sweep-requirements.txt" in txt
    assert "# audit-exempt: out-of-lockfile-venv" in txt
    assert all("--no-sync" in l for l in body if re.search(r"\buv\s+run\b", l))
    # fail-fast guards on every external input
    for guard in ('[ -s "$SHARED/scripts/hf_env.sh" ]', '[ -s "$GEOM_CONFIG" ]',
                  '[ -s "$REPO_DIR/configs/capeval/manifests/petri_twosi.yaml" ]',
                  '[ -s "$HF_HOME/$GEOM_HF_SNAPSHOT_REL/config.json" ]',
                  '[ -s "$VLLM_REQS" ]', '[ -s "$GEOM_REQS" ]',
                  '[ -s "$RESULTS_DIR/budget.json" ]', '[ -s "$RESULTS_DIR/sweep_summary.json" ]',
                  '[ -d "$REPO_DIR/src/sisteer" ]'):
        assert guard in txt, guard
    assert "--phase env" in txt and "--phase preflight" in txt
    # preflight, then budget, then sweep, in that order, through the one script
    # (scan the BODY: the header comments mention the phases too)
    body_txt = "\n".join(body)
    assert body_txt.index("--phase preflight") < body_txt.index("--phase budget") \
        < body_txt.index("--phase sweep")
    assert 'RESULTS_ROOT="$SHARED/exp/geom-sweep"' in txt
    assert 'OBJECTS_ROOT="$SHARED/exp/objects/scale-12"' in txt
    # the requirements include: the lane + matplotlib, hashed together
    reqs = (REPO / "configs/capeval/geom-sweep-requirements.txt").read_text()
    assert "-r vllm-requirements.txt" in reqs and re.search(r"^matplotlib==\d", reqs, re.M)
    assert 'cat "$VLLM_REQS" "$GEOM_REQS" | sha256sum' in txt


# --------------------------------------------------------------------------- #
# the standup extraction                                                       #
# --------------------------------------------------------------------------- #


def test_standup_exposes_bring_up_and_keeps_its_cli():
    up = _load("_capctrl_serving_up_bring_up", "scripts/capctrl_serving_up.py")
    import inspect
    assert callable(up.bring_up) and inspect.isclass(up.Standup)
    params = inspect.signature(up.bring_up).parameters
    for name in ("repo", "config_path", "cell", "defenses_dir", "out", "geom", "gpu_mem_util",
                 "max_logprobs", "ready_timeout_s", "install_timeout_s", "petri_agent_loop",
                 "tool_calls", "manifest"):
        assert name in params and params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    fields = {f.name for f in __import__("dataclasses").fields(up.Standup)}
    assert {"launch", "server", "proxy_url", "table_path", "harness_cell", "geometry"} <= fields
    for meth in ("tear_down", "endpoint", "server_rc", "attestation_path"):
        assert hasattr(up.Standup, meth), meth
    src = (REPO / "scripts/capctrl_serving_up.py").read_text()
    # main still owns the CLI contract the geometry tests pin
    for flag in ("--eval", "--max-model-len", "--max-num-seqs", "--allow-over-budget",
                 "--gpu-mem-util", "--max-logprobs", "--stop-file", "--tool-calls", "--manifest"):
        assert f'"{flag}"' in src, flag
    assert 'tmp.rename(out / "endpoint.json")' in src            # atomic publish kept
    assert "su = bring_up(" in src and "su.tear_down()" in src
    # the sweep goes through the SAME bring-up, never its own composer call
    sweep = (REPO / "scripts/served_decode_sweep.py").read_text()
    assert "standup.bring_up(" in sweep and "build_server_launch(" not in sweep
    assert "allow_over_budget=True" in sweep                     # probing past the budget is the point


# --------------------------------------------------------------------------- #
# E50 #7 — a profiling measurement's identity includes the OBJECT's bytes      #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def hp():
    return _load("_hookprof_served", "scripts/hookprof_served.py")


def _bank(tmp_path, cell="safety_refusal__psr_kl_all", *, artifact=b"OBJECT-A",
          sidecar=None, unserved=None):
    """A model's object bank as the standup resolves it: ``<bank>/<cell>/`` with
    the arm's own artifact filename (``psr_kl_all`` -> ``object.pt``,
    ``defense_cell.object_path_for``) and, optionally, its ``object.json``
    sidecar — the ``defenses_dir`` the standup is handed. ``unserved`` drops a
    further ``(name, bytes)`` into the dir that the export does NOT copy."""
    d = tmp_path / "bank" / cell
    d.mkdir(parents=True, exist_ok=True)
    (d / "object.pt").write_bytes(artifact)
    if sidecar is not None:
        (d / "object.json").write_bytes(sidecar)
    if unserved is not None:
        (d / unserved[0]).write_bytes(unserved[1])
    return tmp_path / "bank"


def test_object_identity_digests_the_bytes_the_standup_would_export(sw, tmp_path):
    """E50 #7: these probe standups pass ``require_object_digest=False``
    (they file timings, never rows, and have no preflight to bind
    to), so a cell NAME is the only thing tying a measurement to an object, and
    a bank path re-fitted in place serves different bytes under that same name.
    The identity is the bytes: the artifact's entry is exactly the sha256
    ``lens_steer.export_worker_registry`` computes over the bytes it copies, and
    the sidecar rides along because the served intervention hash is taken over
    (spec, artifact, sidecar)."""
    import hashlib

    cell = "safety_refusal__psr_kl_all"
    bank = _bank(tmp_path, cell)
    ident = sw.object_identity(bank, cell)
    assert ident == {"artifact": hashlib.sha256(b"OBJECT-A").hexdigest()}
    assert sw.select_steered_cell(bank, (cell,)) == cell            # the sweep's own resolution
    # no steered arm: no object identity, and a sayable label either way
    assert sw.object_identity(bank, None) is None
    assert sw.object_identity(None, None) is None
    assert sw.object_identity_id(None) == "none" and len(sw.object_identity_id(ident)) == 16
    # a sidecar is part of what is served, so it is part of the identity
    with_side = sw.object_identity(_bank(tmp_path / "s", cell, sidecar=b'{"layers": [15]}'), cell)
    assert set(with_side) == {"artifact", "sidecar"}
    assert sw.object_identity_id(with_side) != sw.object_identity_id(ident)
    # the SAME bytes always give the same label; different bytes never do
    assert sw.object_identity_id(sw.object_identity(_bank(tmp_path / "same", cell), cell)) == \
        sw.object_identity_id(ident)
    other = sw.object_identity(_bank(tmp_path / "b", cell, artifact=b"OBJECT-B"), cell)
    assert sw.object_identity_id(other) != sw.object_identity_id(ident)
    # a named cell with nothing on disk is a refusal, never an empty identity
    (bank / cell / "object.pt").unlink()
    with pytest.raises(SystemExit, match="cannot be identified by a cell name alone"):
        sw.object_identity(bank, cell)


def _sweep_ctx(sw, tmp_path, bank, cell):
    (tmp_path / "res").mkdir(exist_ok=True)
    return sw.SweepContext(
        ms=sw.MODELS["gemma4e4b"], repo=REPO, results_dir=tmp_path / "res",
        figs_dir=tmp_path / "figs", config_path=REPO / sw.MODELS["gemma4e4b"].cfg,
        objects_dir=bank, steered_cell=cell, gpu_mem_util="0.90", ready_timeout_s=10,
        manifest="configs/capeval/manifests/petri_twosi.yaml", deadline_epoch=None,
        deadline_margin_s=0.0, job_id="1", node="TESTHOST")


def test_the_sweep_fingerprint_moves_when_the_object_does(sw, tmp_path):
    """E50 #7: the fingerprint decides what a resume may reuse, and before this
    it named the cell but not the object — so re-fitting in place and re-running
    resumed onto the previous object's probes and rows under one label."""
    cell = "safety_refusal__psr_kl_all"
    bank = _bank(tmp_path, cell)
    kw = dict(phase="budget", budget_len=8192, ladder=[393_216])
    fp1 = sw.measurement_fingerprint(_sweep_ctx(sw, tmp_path, bank, cell), **kw)
    assert fp1["steered_cell"] == cell
    assert fp1["steered_object"] == sw.object_identity(bank, cell)
    # the same object, twice: one measurement
    assert sw.fingerprint_id(fp1) == sw.fingerprint_id(
        sw.measurement_fingerprint(_sweep_ctx(sw, tmp_path, bank, cell), **kw))
    # the object is re-fitted IN PLACE — same bank, same cell name, new bytes
    (bank / cell / "object.pt").write_bytes(b"OBJECT-REFIT")
    fp2 = sw.measurement_fingerprint(_sweep_ctx(sw, tmp_path, bank, cell), **kw)
    assert sw.fingerprint_id(fp2) != sw.fingerprint_id(fp1)
    assert [k for k, _a, _b in sw.fingerprint_diff(fp1, fp2)] == ["steered_object"]
    # a passthrough-only sweep is not made unrunnable by any of this
    fp0 = sw.measurement_fingerprint(_sweep_ctx(sw, tmp_path, bank, None), **kw)
    assert fp0["steered_object"] is None and sw.fingerprint_id(fp0) != sw.fingerprint_id(fp1)


def test_a_budget_resume_reuses_probes_of_the_same_object_and_refuses_a_swapped_one(sw, tmp_path):
    """The consequence, at the gate that acts on it: a recorded budget walk
    resumes when the object is the one it measured, and is refused — never
    silently extended — when the bytes under that cell name changed."""
    cell = "safety_refusal__psr_kl_all"
    bank = _bank(tmp_path, cell)
    ctx = _sweep_ctx(sw, tmp_path, bank, cell)
    L, ladder = 8192, [393_216]
    fp = sw.measurement_fingerprint(ctx, phase="budget", budget_len=L, ladder=ladder)
    probe = {"product": 393_216, "max_num_seqs": 48, "max_model_len": L, "status": "pass",
             "tag": "budget_L8192xS48"}
    (ctx.results_dir / "budget.json").write_text(json.dumps(
        {"fingerprint": fp, "measurement_id": sw.fingerprint_id(fp), "probes": [probe]}))
    # SAME object: the recorded probe is reused, the walk is already closed, and
    # no standup is attempted (this test has no GPU and never reaches one)
    doc = sw._run_budget_locked(ctx, budget_len=L, ladder=ladder)
    assert doc["complete"] and doc["largest_passing"]["product"] == 393_216
    assert doc["measurement_id"] == sw.fingerprint_id(fp)
    # SWAPPED object under the same cell name: refused, naming the field
    (bank / cell / "object.pt").write_bytes(b"OBJECT-REFIT")
    with pytest.raises(SystemExit) as ei:
        sw._run_budget_locked(_sweep_ctx(sw, tmp_path, bank, cell), budget_len=L, ladder=ladder)
    why = str(ei.value)
    assert "DIFFERENT measurement" in why and "steered_object" in why
    assert "refusing to reuse" in why and str(ctx.results_dir) in why


def test_hookprof_points_are_partitioned_by_the_object_they_measured(hp, sw, tmp_path):
    """E50 #7: ``points.jsonl`` is APPENDED to and its standup carries no object
    digest, so the file itself is where the partition has to live. Every row
    states the object it measured; a run refuses to append into a file holding
    another object's points (including pre-partition rows, whose object is
    simply unknown), and the analyzer refuses to summarize a file that holds
    more than one."""
    cell = "safety_refusal__psr_kl_all"
    bank = _bank(tmp_path, cell)
    a = sw.object_identity(bank, cell)
    a_id = sw.object_identity_id(a)
    b_id = sw.object_identity_id(sw.object_identity(_bank(tmp_path / "b", cell,
                                                          artifact=b"OBJECT-B"), cell))
    assert hp.OBJECT_ID_FIELD == "object_id" and hp.OBJECT_FIELD == "steered_object"
    rows_path = tmp_path / "points.jsonl"
    assert hp.read_points(rows_path) == [] and hp.points_object_ids([]) == []
    hp.assert_points_object(rows_path, a_id, a)          # a fresh file: nothing to clash with

    def _append(oid):
        with rows_path.open("a") as fh:
            fh.write(json.dumps({"arm": "steered", "B": 8, hp.OBJECT_ID_FIELD: oid,
                                 hp.OBJECT_FIELD: a}) + "\n")

    _append(a_id)
    hp.assert_points_object(rows_path, a_id, a)          # the same object: appending is a resume
    assert hp.points_object_ids(hp.read_points(rows_path)) == [a_id]
    with pytest.raises(SystemExit) as ei:
        hp.assert_points_object(rows_path, b_id, {"artifact": "b"})
    assert a_id in str(ei.value) and b_id in str(ei.value) and "E50 #7" in str(ei.value)
    # a pre-partition row (written before the key existed) is its own identity:
    # unknown, and therefore never this run's
    old = tmp_path / "old.jsonl"
    old.write_text(json.dumps({"arm": "steered", "B": 8}) + "\n")
    with pytest.raises(SystemExit, match="no object recorded"):
        hp.assert_points_object(old, a_id, a)
    # the analyzer refuses a mixed file rather than averaging two measurements
    mixed = tmp_path / "gemma4e4b"
    mixed.mkdir()
    (mixed / "points.jsonl").write_text(json.dumps({"arm": "steered", "B": 8,
                                                    hp.OBJECT_ID_FIELD: a_id}) + "\n"
                                        + json.dumps({"arm": "steered", "B": 8,
                                                      hp.OBJECT_ID_FIELD: b_id}) + "\n")
    args = SimpleNamespace(out=str(tmp_path), model="gemma4e4b", speedscope=None, window=400,
                           thresh=0.05, min_len=50, top_n=5)
    with pytest.raises(SystemExit, match="different fitted objects"):
        hp.analyze(args)


def test_the_profiling_probes_keep_their_library_only_digest_exemption():
    """The deviation stands: these two scripts state
    ``require_object_digest=False`` rather than omitting the flag, and neither
    grows a preflight requirement. E50 #7 asked for an honest measurement
    identity, not for the exemption to be reversed."""
    for rel in ("scripts/served_decode_sweep.py", "scripts/hookprof_served.py"):
        src = (REPO / rel).read_text()
        assert "require_object_digest=False" in src, rel
        assert "expect_object_sha256" not in src, rel
        assert "object_identity(" in src, rel


# --------------------------------------------------------------------------- #
# E54 #5 — the identity is RE-ATTESTED against the bytes the standup loaded    #
# --------------------------------------------------------------------------- #

#: The registry entry key is the HARNESS cell id, which is NOT the bank
#: directory name — a lookup that confused the two would pass this file's
#: fixtures by accident, so they are deliberately different strings.
HARNESS = "petri__safety_refusal__psr_kl_all"


def _registry(out_dir, cell_id=HARNESS, *, artifact=b"OBJECT-A", sidecar=None,
              sha16=None):
    """A standup's exported worker registry, shaped as
    ``lens_steer.export_worker_registry`` writes it: ``<out>/lens_registry/``
    holding ``registry.json`` plus the blob COPIES the export hashes and the
    server is then launched against. ``sha16`` overrides the recorded digest so
    a registry that disagrees with its own bytes can be built."""
    import hashlib

    reg = Path(out_dir) / "lens_registry"
    (reg / cell_id).mkdir(parents=True, exist_ok=True)
    (reg / cell_id / "artifact.bin").write_bytes(artifact)
    entry = {"cell_id": cell_id, "kind": "psr", "artifact": f"{cell_id}/artifact.bin",
             "sidecar": None, "artifact_sha16": sha16 or hashlib.sha256(artifact).hexdigest()[:16],
             "artifact_intervention_hash": "ab" * 16}
    if sidecar is not None:
        (reg / cell_id / "sidecar.bin").write_bytes(sidecar)
        entry["sidecar"] = f"{cell_id}/sidecar.bin"
    (reg / "registry.json").write_text(json.dumps({"version": 1, "entries": [entry]}))
    return reg


def test_the_attestation_reads_the_bytes_the_standup_actually_loaded(sw, tmp_path):
    """E54 #5: ``object_identity`` hashes the BANK before the standup, and the
    standup then runs unbound to any object digest (the seam is kept), so
    nothing tied the fingerprint to the bytes served. The attestation re-reads
    the artifact and sidecar COPIES out of the registry the standup exported —
    what the worker loads — and returns that identity for the measurement to
    record."""
    import hashlib

    cell = "safety_refusal__psr_kl_all"
    side = b'{"layers": [15]}'
    bank = _bank(tmp_path, cell, sidecar=side)
    fp_obj = sw.object_identity(bank, cell)
    out = tmp_path / "standup"
    _registry(out, artifact=b"OBJECT-A", sidecar=side)
    att = sw.attest_object_identity(out, HARNESS, fp_obj, where="unit")
    assert att["cell_id"] == HARNESS and att["intervention_hash"] == "ab" * 16
    assert att["artifact_sha256"] == hashlib.sha256(b"OBJECT-A").hexdigest()
    assert att["artifact_sha16"] == att["artifact_sha256"][:16]
    # the bank names object.pt / object.json; the registry names artifact.bin /
    # sidecar.bin — the SAME bytes under different filenames, which is why both
    # sides are keyed by the ROLE the export copied them under (E55 #4), with
    # the registry's own filenames kept beside them for the refusal messages
    assert set(att["digests"]) == {"artifact", "sidecar"} == set(fp_obj)
    assert att["digests"] == dict(fp_obj)
    assert att["blobs"] == {"artifact": f"{HARNESS}/artifact.bin",
                            "sidecar": f"{HARNESS}/sidecar.bin"}
    assert str(out) in att["registry"]
    # a passthrough-only run has nothing to attest and is not made to refuse
    assert sw.attest_object_identity(out, None, None, where="unit") is None
    assert sw.attest_object_identity(out, "none", None, where="unit") is None
    assert sw.loaded_object_identity(out, None) is None


@pytest.mark.parametrize("served,side,what", [
    (b"OBJECT-B", b'{"layers": [15]}', "artifact"),
    (b"OBJECT-A", b'{"layers": [20]}', "sidecar"),
])
def test_bytes_swapped_between_the_fingerprint_and_the_standup_refuse(
        sw, tmp_path, served, side, what):
    """E54 #5, the finding itself: an in-place re-fit landing between the
    fingerprint and the bring-up made the run record B's measurements under A's
    identity — or, with only one of the two blobs changed, under a hybrid
    identity that never existed on any disk. Both halves now refuse, naming
    BOTH sides. The sidecar case matters on its own: the served intervention
    hash is taken over (spec, artifact, sidecar), so a sidecar-only edit changes
    what is served while the artifact bytes stand still."""
    import hashlib

    cell = "safety_refusal__psr_kl_all"
    fingerprinted = b'{"layers": [15]}'
    bank = _bank(tmp_path, cell, sidecar=fingerprinted)
    fp_obj = sw.object_identity(bank, cell)
    out = tmp_path / "standup"
    _registry(out, artifact=served, sidecar=side)
    with pytest.raises(SystemExit) as ei:
        sw.attest_object_identity(out, HARNESS, fp_obj, where="budget probe L8192xS48")
    why = str(ei.value)
    assert "budget probe L8192xS48" in why and "E54 #5" in why
    assert "not the bytes the measurement was fingerprinted on" in why
    changed = served if what == "artifact" else side
    kept = b"OBJECT-A" if what == "artifact" else fingerprinted
    assert hashlib.sha256(changed).hexdigest() in why          # what was LOADED
    assert hashlib.sha256(kept).hexdigest() in why             # what was FINGERPRINTED
    # ... and the run does not proceed to file anything under either identity
    assert sw.object_identity_id(fp_obj) != sw.object_identity_id(
        {"artifact": hashlib.sha256(served).hexdigest(),
         "sidecar": hashlib.sha256(side).hexdigest()})


def test_an_unreadable_or_self_contradicting_registry_refuses(sw, tmp_path):
    """The attestation's own failure modes are refusals, never a shrug: a
    measurement that cannot say which bytes it loaded must not be filed. All
    four shapes name the file."""
    cell = "safety_refusal__psr_kl_all"
    bank = _bank(tmp_path, cell)
    fp_obj = sw.object_identity(bank, cell)
    # (a) no registry at all
    with pytest.raises(SystemExit, match="missing or unreadable"):
        sw.attest_object_identity(tmp_path / "empty", HARNESS, fp_obj, where="u")
    # (b) a registry that does not serve the cell this measurement claims
    out = tmp_path / "s2"
    _registry(out, cell_id="petri__something_else")
    with pytest.raises(SystemExit, match="no entry for the served cell"):
        sw.attest_object_identity(out, HARNESS, fp_obj, where="u")
    # (c) the recorded digest disagrees with the blob on disk
    out3 = tmp_path / "s3"
    _registry(out3, artifact=b"OBJECT-A", sha16="0" * 16)
    with pytest.raises(SystemExit, match="disagrees with its own bytes"):
        sw.attest_object_identity(out3, HARNESS, fp_obj, where="u")
    # (d) the blob the registry names is gone
    out4 = tmp_path / "s4"
    reg = _registry(out4, artifact=b"OBJECT-A")
    (reg / HARNESS / "artifact.bin").unlink()
    with pytest.raises(SystemExit, match="cannot be re-hashed"):
        sw.attest_object_identity(out4, HARNESS, fp_obj, where="u")
    # (e) the two sides disagree about whether there IS an object at all
    out5 = tmp_path / "s5"
    _registry(out5, artifact=b"OBJECT-A")
    with pytest.raises(SystemExit, match="does not name"):
        sw.attest_object_identity(out5, HARNESS, None, where="u")
    with pytest.raises(SystemExit, match="exported no lens leg"):
        sw.attest_object_identity(out5, "none", fp_obj, where="u")


# --------------------------------------------------------------------------- #
# E55 #4 — the attestation is COMPLETE over the exporter's served projection   #
# --------------------------------------------------------------------------- #


def test_the_fingerprint_names_the_exporters_served_projection_not_the_bank_dir(sw, tmp_path):
    """E55 #4: completeness is only definable against a KNOWN set of served
    components, so the fingerprint is taken over the projection
    ``lens_steer.export_worker_registry`` actually copies — the arm's own
    artifact filename (``defense_cell.object_path_for``, which
    ``registry_adapter`` cross-checks the registry against) plus the
    ``<artifact>.json`` sidecar when the cell declares one — never every
    ``object.*`` file in the directory."""
    cell = "safety_refusal__psr_kl_all"                    # a torch-blob arm: object.pt
    bank = _bank(tmp_path, cell, sidecar=b'{"layers": [15]}',
                 unserved=("object.safetensors", b"A-STALE-BLOB-NOBODY-SERVES"))
    proj = sw.served_projection(bank, cell)
    assert set(proj) == {"artifact", "sidecar"} == set(sw.SERVED_ROLES)
    assert proj["artifact"].name == "object.pt" and proj["sidecar"].name == "object.json"
    assert set(sw.object_identity(bank, cell)) == {"artifact", "sidecar"}
    # the stale blob is in the dir and in no way in the identity
    assert "A-STALE-BLOB-NOBODY-SERVES" not in json.dumps(sw.object_identity(bank, cell))
    # a cell that declares no sidecar projects the artifact alone
    assert set(sw.served_projection(_bank(tmp_path / "nos", cell), cell)) == {"artifact"}
    # a SteeringVector-container arm's loader reads the sidecar, so the export
    # declares it whether or not it exists — missing is refused here, before a
    # GPU minute, rather than by the export dying on the read
    vec = "safety_refusal__antidote"
    d = tmp_path / "vec" / "bank" / vec
    d.mkdir(parents=True)
    (d / "object.safetensors").write_bytes(b"VEC")
    with pytest.raises(SystemExit, match="serves a sidecar"):
        sw.served_projection(tmp_path / "vec" / "bank", vec)
    (d / "object.json").write_bytes(b"{}")
    assert set(sw.served_projection(tmp_path / "vec" / "bank", vec)) == {"artifact", "sidecar"}
    # no steered cell projects nothing; a named cell with no objects dir refuses
    assert sw.served_projection(bank, None) == {}
    with pytest.raises(SystemExit, match="cannot be identified by a cell name alone"):
        sw.served_projection(None, cell)


def test_a_fingerprinted_component_that_was_never_served_refuses(sw, tmp_path):
    """The finding: the comparison was subset membership over an unordered set
    of digests, so a sidecar that DISAPPEARED between the fingerprint and the
    export left the registry serving the artifact alone, every loaded digest
    was still accounted for, and the attestation passed — filing timings under
    a ``measurement_id`` whose fingerprint named bytes that were never served,
    while the sidecar is part of the served intervention hash. Missing, extra
    and mismatched now each refuse, naming the ROLE and both digests."""
    import hashlib

    cell = "safety_refusal__psr_kl_all"
    side = b'{"layers": [15]}'
    art_sha = hashlib.sha256(b"OBJECT-A").hexdigest()
    side_sha = hashlib.sha256(side).hexdigest()

    # (a) MISSING: fingerprinted with a sidecar, exported without one
    bank = _bank(tmp_path, cell, sidecar=side)
    fp_obj = sw.object_identity(bank, cell)
    out = tmp_path / "no_sidecar"
    _registry(out, artifact=b"OBJECT-A")                   # artifact only
    with pytest.raises(SystemExit) as ei:
        sw.attest_object_identity(out, HARNESS, fp_obj, where="budget probe L8192xS48")
    why = str(ei.value)
    assert "MISSING sidecar" in why and side_sha in why    # the role, and what was expected
    assert "budget probe L8192xS48" in why and "E55 #4" in why
    assert "MISSING artifact" not in why                   # the artifact was fine; say so

    # (b) EXTRA: fingerprinted without a sidecar, exported with one
    bank2 = _bank(tmp_path / "b", cell)
    out2 = tmp_path / "extra"
    _registry(out2, artifact=b"OBJECT-A", sidecar=side)
    with pytest.raises(SystemExit) as ei2:
        sw.attest_object_identity(out2, HARNESS, sw.object_identity(bank2, cell), where="u")
    assert "EXTRA sidecar" in str(ei2.value) and side_sha in str(ei2.value)

    # (c) an unserved extra file in the BANK is not a component, so the same
    #     standup that refused (a) proceeds here: projection, not directory
    bank3 = _bank(tmp_path / "c", cell, sidecar=side,
                  unserved=("object.safetensors", b"A-STALE-BLOB-NOBODY-SERVES"))
    out3 = tmp_path / "ok"
    _registry(out3, artifact=b"OBJECT-A", sidecar=side)
    att = sw.attest_object_identity(out3, HARNESS, sw.object_identity(bank3, cell), where="u")
    assert att["digests"] == {"artifact": art_sha, "sidecar": side_sha}

    # (d) a fingerprint written by the OLD sweep — bank filenames, not roles —
    #     cannot be compared role-wise and is refused rather than half-checked
    with pytest.raises(SystemExit, match="not served roles"):
        sw.attest_object_identity(out3, HARNESS,
                                  {"object.pt": art_sha, "object.json": side_sha}, where="u")


def _fake_standup(sw, monkeypatch, *, served: bytes, torn: list):
    """A standup that never touches a GPU: it writes the registry a real one
    would export (with ``served`` as the loaded artifact bytes) and reports the
    harness cell. This is the "fake standup that reports a different digest"
    the finding asks the refusal to be driven by."""

    class _Su:
        def __init__(self, out):
            self.out = out
            self.harness_cell = HARNESS
            self.attestation_path = out / "attestation.json"

        def tear_down(self):
            torn.append(True)

    def _up(ctx, *, max_model_len, max_num_seqs, tag, eval_key):
        out = sw.standup_dir(ctx, tag)
        out.mkdir(parents=True, exist_ok=True)
        _registry(out, artifact=served)
        return _Su(out)

    monkeypatch.setattr(sw, "_bring_up", _up)
    monkeypatch.setattr(sw, "wait_gpu_free", lambda *a, **k: 0)


def test_a_budget_probe_records_the_attested_identity_and_refuses_a_swap(
        sw, tmp_path, monkeypatch):
    """The wiring, through the real ``probe_geometry``: the attestation happens
    right after the bring-up and BEFORE the probe is recorded, its result rides
    on the probe record, and a swap aborts the walk (a SystemExit, not a
    ``launch_error`` status — a re-fitted object is not evidence about the
    geometry) with the server still torn down."""
    import hashlib

    cell = "safety_refusal__psr_kl_all"
    bank = _bank(tmp_path, cell)
    ctx = _sweep_ctx(sw, tmp_path, bank, cell)
    fp = sw.measurement_fingerprint(ctx, phase="budget", budget_len=8192, ladder=[393_216])
    torn: list = []

    _fake_standup(sw, monkeypatch, served=b"OBJECT-A", torn=torn)
    rec = sw.probe_geometry(ctx, max_model_len=8192, max_num_seqs=48,
                            expect_object=fp["steered_object"])
    assert rec["status"] == "pass" and torn == [True]
    att = rec[sw.ATTESTED_OBJECT_FIELD]
    assert att["artifact_sha256"] == hashlib.sha256(b"OBJECT-A").hexdigest()
    assert att["digests"] == fp["steered_object"]       # complete, not a subset

    # the same walk, with the object re-fitted under it between the fingerprint
    # and this probe's standup
    torn.clear()
    _fake_standup(sw, monkeypatch, served=b"OBJECT-REFIT", torn=torn)
    (tmp_path / "two").mkdir()
    ctx2 = _sweep_ctx(sw, tmp_path / "two", bank, cell)
    with pytest.raises(SystemExit, match="not the bytes the measurement was fingerprinted on"):
        sw.probe_geometry(ctx2, max_model_len=8192, max_num_seqs=48,
                          expect_object=fp["steered_object"])
    assert torn == [True]                      # the engine is not left on the card
    # a probe with no expectation to check against is unchanged (a passthrough
    # sweep), and the field is filed as null rather than omitted
    torn.clear()
    monkeypatch.setattr(sw, "_bring_up", lambda ctx_, **kw: _NoLensSu(
        sw.standup_dir(ctx_, kw["tag"]), torn))
    (tmp_path / "three").mkdir()
    rec0 = sw.probe_geometry(_sweep_ctx(sw, tmp_path / "three", bank, None),
                             max_model_len=8192, max_num_seqs=48, expect_object=None)
    assert rec0[sw.ATTESTED_OBJECT_FIELD] is None and rec0["status"] == "pass"


class _NoLensSu:
    """A passthrough standup: no fitted leg, so nothing to attest."""

    def __init__(self, out, torn):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.harness_cell = "none"
        self.attestation_path = self.out / "attestation.json"
        self._torn = torn

    def tear_down(self):
        self._torn.append(True)


def test_every_persisted_probe_measurement_carries_the_attested_identity(sw, hp):
    """A structural pin for the two sites a unit test cannot reach without a
    GPU: the sweep's window loop and hookprof's point loop both attest BEFORE
    they measure, and both put the attested identity on the row they file. The
    deviation is untouched by all of it — neither script gains a preflight
    requirement."""
    sweep_src = (REPO / "scripts/served_decode_sweep.py").read_text()
    hook_src = (REPO / "scripts/hookprof_served.py").read_text()
    assert sweep_src.count("attest_object_identity(") >= 3      # def + budget + window
    assert "attested = attest_object_identity(" in sweep_src
    assert "ATTESTED_OBJECT_FIELD: attested," in sweep_src
    assert "attested = sds.attest_object_identity(" in hook_src
    assert "sds.ATTESTED_OBJECT_FIELD: attested," in hook_src
    # one owner for the field name; hookprof reads it off the sweep
    assert sw.ATTESTED_OBJECT_FIELD == "steered_object_attested"
    assert "ATTESTED_OBJECT_FIELD =" not in hook_src
    assert getattr(hp, "ATTESTED_OBJECT_FIELD", None) is None
    for src in (sweep_src, hook_src):
        assert "require_object_digest=False" in src and "expect_object_sha256" not in src


# --------------------------------------------------------------------------- #
# the budget-only window sbatch (the MATH-500 17,664 re-measure)               #
# --------------------------------------------------------------------------- #

BUDGET_WINDOW_SBATCH = REPO / "scripts/sbatch/geom_budget_window.sbatch"


def _active_lines(txt: str):
    return [l for l in txt.splitlines() if l.strip() and not l.lstrip().startswith("#")]


def test_budget_window_sbatch_parses_and_keeps_the_hard_rules():
    """One model, one explicit window, one explicit S ladder, the budget phase
    alone — the same cluster hygiene as geom_sweep.sbatch, positional args in
    place of the array index."""
    proc = subprocess.run(["bash", "-n", str(BUDGET_WINDOW_SBATCH)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    txt = BUDGET_WINDOW_SBATCH.read_text()
    directives = [l for l in txt.splitlines() if l.startswith("#SBATCH")]
    active = directives + _active_lines(txt)
    assert not any("--export=" in l for l in active)              # hard rule 1
    assert not any("--mem=0" in l for l in active)                # hard rule 2
    assert not re.search(r"^\s*(export\s+)?CUDA_VISIBLE_DEVICES\s*=", txt, re.M)  # hard rule 3 (read only)
    assert not any(re.search(r"--qos=\S*dev", l) for l in directives)
    assert "#SBATCH --job-name=geom-budget-window" in txt
    assert "#SBATCH --qos=high" in txt and "#SBATCH --gres=gpu:1" in txt
    assert "#SBATCH --cpus-per-task=16" in txt and "#SBATCH --mem=200G" in txt
    assert re.search(r"^#SBATCH --time=0-02:30:00", txt, re.M)
    assert not any("--array" in l for l in directives)            # one model per submission, by argument
    assert "#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out" in txt
    assert "#SBATCH --open-mode=append" in txt and "#SBATCH --no-requeue" in txt
    assert "#SBATCH --signal=B:TERM@120" in txt
    assert not re.search(r"^\s*for .*sbatch", txt, re.M)          # never a submit loop


def test_budget_window_sbatch_body_takes_the_window_and_ladder_by_argument(sw):
    txt = BUDGET_WINDOW_SBATCH.read_text()
    body = _active_lines(txt)
    body_txt = "\n".join(body)
    # the four positional args, every one REQUIRED (no default sha, model, window or ladder)
    for arg in ('GEOM_SHA="${1:?', 'MODEL="${2:?', 'BUDGET_LEN="${3:?', 'SEQS_LADDER="${4:?'):
        assert arg in txt, arg
    assert "SLURM_ARRAY_TASK_ID" not in body_txt
    # the ladder is S values, tiled into S x L products for --budget-ladder; block-aligned, ascending
    assert '[ $((BUDGET_LEN % 16)) -eq 0 ]' in txt
    assert 'PRODUCTS="${PRODUCTS:+$PRODUCTS,}$((S * BUDGET_LEN))"' in txt
    assert 'strictly ascending' in txt
    assert '--budget-len "$BUDGET_LEN" --budget-ladder "$PRODUCTS" --gpu-mem-util 0.90' in txt
    # the same scaffolding as geom_sweep.sbatch
    assert 'source "$SHARED/scripts/hf_env.sh"' in txt
    for env in ("NCCL_NVLS_ENABLE=0",
                'PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"', "HF_HUB_OFFLINE=1"):
        assert env in txt, env
    assert "NCCL_SOCKET_IFNAME=" not in txt     # the socket interface is the site's (its nccl.conf), never exported
    assert "cleanup() { trap '' SIGTERM; _gb_handled=1; kill -TERM -$$ 2>/dev/null; wait; }" in txt
    assert "trap cleanup SIGTERM SIGINT SIGQUIT" in txt and "trap finish EXIT" in txt
    assert 'export FIGS_DIR="$SHARED/exp/figs/$SLURM_JOB_NAME/$MODEL"' in txt
    assert '"$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR"' in txt
    assert '> "$FIGS_DIR/RESULT"' in txt and "elapsed_s=$SECONDS" in txt
    # the SAME requirements-hash venv as geom_sweep.sbatch (one venv per requirements set, per node)
    assert 'UV_PROJECT_ENVIRONMENT="/home/$USER/venvs/sisteer-geom-sweep-vllm-$REQ_HASH"' in txt
    assert 'cat "$VLLM_REQS" "$GEOM_REQS" | sha256sum' in txt
    assert "flock -x 9" in txt and "geom-sweep-requirements.txt" in txt and "geom-sweep-requirements-tf5141.txt" in txt
    assert "# audit-exempt: out-of-lockfile-venv" in txt
    assert all("--no-sync" in l for l in body if re.search(r"\buv\s+run\b", l))
    # fail-fast guards on every external input
    for guard in ('[ -s "$SHARED/scripts/hf_env.sh" ]', '[ -s "$GEOM_CONFIG" ]',
                  '[ -s "$REPO_DIR/configs/capeval/manifests/petri_twosi.yaml" ]',
                  '[ -s "$HF_HOME/$GEOM_HF_SNAPSHOT_REL/config.json" ]',
                  '[ -s "$VLLM_REQS" ]', '[ -s "$GEOM_REQS" ]', '[ -s "$RESULTS_DIR/budget.json" ]',
                  '[ -d "$REPO_DIR/src/sisteer" ]'):
        assert guard in txt, guard
    # preflight, then the budget phase, and NO sweep phase; a per-WINDOW results root so the
    # L=8192 record (fingerprint-locked to its own ladder) is never extended by this walk
    assert body_txt.index("--phase preflight") < body_txt.index("--phase budget")
    assert "--phase sweep" not in body_txt
    assert 'RESULTS_ROOT="$SHARED/exp/geom-sweep-L$BUDGET_LEN"' in txt
    assert 'OBJECTS_ROOT="$SHARED/exp/objects/scale-12"' in txt
    # the positive artifact: a COMPLETE bracket with a passing probe, else the job fails
    assert 'd.get("complete") is True and lp is not None and not d.get("aborted") and not d.get("stopped_for_deadline")' in txt
    assert "budget.json is not a complete measurement" in txt
    # the ladder the sbatch hands the sweep tiles the window exactly (the sweep refuses otherwise)
    for S in (20, 32, 40, 44, 48, 56, 64, 80):
        assert sw.ladder_seqs(S * 17664, 17664) == S
