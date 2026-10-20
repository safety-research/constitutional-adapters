"""Canary 373464 (2026-09-03, qwen35_4b): the sweep's 5th window, 469 x 1536,
was REFUSED by the readiness gate (KV coverage) and the exception propagated
out of ``run_sweep`` — the phase exited 1 with 44 good rows and no summary.

Two fixes, pinned here CPU-only with a fake ``_bring_up`` that refuses above
a threshold: (1) a refused window is RECORDED (``window_refused`` row) and
RETRIED at the next lower ladder S, at most 3 times, then ``window_skipped``,
and the phase CONTINUES to the next window and still writes the summary, which
lists refused / skipped / not-measured windows explicitly; (2) the budget is a
TOKEN budget measured at ONE S — for a hybrid-recurrent model the per-slot
recurrent state makes the feasible product shrink with S, so the per-window S
is capped at 256 for those families (the same detection vllm_server uses),
and the analyzer rebuilds a per-window table from ``sweep.jsonl`` alone when
the summary is absent.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.capeval.serving import geometry as sg
from sisteer.capeval.serving.vllm_server import VllmServerError

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import served_decode_sweep as sw  # noqa: E402  (the real module name: the analyzer imports it)


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# FIX 2: the hybrid per-slot cap                                               #
# --------------------------------------------------------------------------- #


def test_hybrid_detection_matches_vllm_server_and_caps_S():
    assert sw.S_MAX_HYBRID == 256 and sw.MAX_SEQS_CAP == 512
    assert sw.S_MAX_HYBRID == sg.HYBRID_SEQS_CAP                       # one cap, two homes
    assert set(sg.HYBRID_MODEL_KEYS) == {k for k, ms in sw.MODELS.items() if sw.is_hybrid_model(ms)}
    hybrid = {k: sw.is_hybrid_model(ms) for k, ms in sw.MODELS.items()}
    assert hybrid == {"llama8b": False, "llama3b": False, "gemma4e4b": False,
                      "gemma4e2b": False, "qwen35_9b": True, "qwen35_4b": True,
                      "glm46vflash": False, "nemotron3nano4b": True}
    assert sw.seqs_cap(True) == 256 and sw.seqs_cap(False) == 512
    # the canary's window: 720,896 // 1536 = 469 -> capped to 256 for qwen
    assert sw.permitted_seqs(720_896, 1536, cap=sw.seqs_cap(True)) == 256
    assert sw.permitted_seqs(720_896, 1536, cap=sw.seqs_cap(False)) == 469
    assert sw.permitted_seqs(720_896, 16384, cap=sw.seqs_cap(True)) == 44   # unchanged below the cap


def test_next_lower_seqs_walks_the_ladder_down():
    assert sw.next_lower_seqs(469) == 384
    assert sw.next_lower_seqs(384) == 256
    assert sw.next_lower_seqs(256) == 192
    assert sw.next_lower_seqs(44) == 32 and sw.next_lower_seqs(32) == 24
    assert sw.next_lower_seqs(9) == 8 and sw.next_lower_seqs(8) is None
    assert sw.next_lower_seqs(1) is None


def test_geometry_states_the_limitation():
    src = (REPO / "src/sisteer/capeval/serving/geometry.py").read_text()
    assert "STATED LIMITATION" in src and "469 x 1536" in src and "S_MAX_HYBRID" in src
    assert "STATED LIMITATION" in sg.ModelBudget.__doc__
    assert "upper bound" in sg.ModelBudget.seqs_at.__doc__.lower()


# --------------------------------------------------------------------------- #
# FIX 1: per-window resilience                                                 #
# --------------------------------------------------------------------------- #


class FakeStandup:
    def __init__(self, L, S):
        self.L, self.S = L, S
        self.proxy_url = "http://127.0.0.1:1/v1"
        self.table_path = Path("/nonexistent/routing_table.json")
        #: E54 #5: the window loop re-attests the object the standup LOADED off
        #: the registry under this dir. These fixtures are passthrough-only
        #: (``harness_cell`` "none"), so the attestation returns None without
        #: reading anything — but ``out`` is a real Standup field and a fake
        #: missing it would only ever fail for the wrong reason.
        self.out = Path("/nonexistent/standup")
        self.harness_cell = "none"
        self.launch = SimpleNamespace(host="127.0.0.1", port=1, api_key="k")
        self.torn_down = False

    def server_rc(self):
        return None

    def tear_down(self):
        self.torn_down = True


class FakeSampler:
    def __init__(self, *a, **k):
        pass

    def start(self):
        return self

    def stop(self):
        pass

    def window(self, w0, w1):
        return {}


def _ctx(tmp_path, *, hybrid):
    (tmp_path / "res").mkdir()
    return sw.SweepContext(
        ms=sw.MODELS["qwen35_4b"], repo=REPO, results_dir=tmp_path / "res",
        figs_dir=tmp_path / "figs", config_path=REPO / "configs/two_si_qwen35_4b.yaml",
        objects_dir=None, steered_cell=None, gpu_mem_util="0.90", ready_timeout_s=10,
        manifest="configs/capeval/manifests/petri_twosi.yaml", deadline_epoch=None,
        deadline_margin_s=0.0, job_id="373464", node="TESTHOST", hybrid=hybrid)


@pytest.fixture
def sweep_env(monkeypatch, tmp_path):
    """run_sweep with every GPU / network seam replaced; ``refuse(L, S)`` decides
    which standups the fake readiness gate refuses."""
    calls = []
    standups = []
    policy = {"refuse": lambda L, S: False}

    def fake_bring_up(ctx, *, max_model_len, max_num_seqs, tag, eval_key):
        calls.append((max_model_len, max_num_seqs, tag, eval_key))
        if policy["refuse"](max_model_len, max_num_seqs):
            raise VllmServerError(
                "resolved scheduler state does not match the composed no-chunking "
                "invariant — a silent clamp, a long-prefill threshold, or a speculative "
                "config re-introduces scheduled shapes the plan lattice refuses; the server "
                f"must not pass readiness:\n- worker 0: KV cache holds 600000 tokens < the "
                f"{max_num_seqs * max_model_len}-token budget (= {max_num_seqs} x "
                f"{max_model_len}): only 390.62-way concurrency at full length is actually "
                "KV-backed, so the requested max_num_seqs exists in argv but not in memory")
        su = FakeStandup(max_model_len, max_num_seqs)
        standups.append(su)
        return su

    async def fake_closed_loop(proxy_url, vmodel, messages, *, B, max_tokens, warmup_s, measure_s):
        import time
        t0 = time.monotonic()
        w0, w1 = t0, t0 + 1.0
        recs = [(w0 + 0.1, w0 + 0.5, max_tokens, True, None) for _ in range(B)]
        return recs, (w0, w1)

    monkeypatch.setattr(sw, "_bring_up", fake_bring_up)
    monkeypatch.setattr(sw, "_closed_loop", fake_closed_loop)
    monkeypatch.setattr(sw, "_load_routing_table",
                        lambda path: {"routes": [{"cell_id": "none", "virtual_model": "petri__base"}]})
    monkeypatch.setattr(sw, "build_prompt",
                        lambda proxy_url, vmodel, n: ([{"role": "user", "content": "x"}], n, 7))
    monkeypatch.setattr(sw, "MetricsSampler", FakeSampler)
    monkeypatch.setattr(sw, "DmonSampler", FakeSampler)
    monkeypatch.setattr(sw, "wait_gpu_free", lambda *a, **k: None)
    monkeypatch.setattr(sw, "SAMPLER_SETTLE_S", 0.0)
    monkeypatch.setattr(sw, "_figures", lambda *a, **k: [])
    return SimpleNamespace(calls=calls, standups=standups, policy=policy, tmp_path=tmp_path)


def _budget_json(ctx, product=720_896):
    (ctx.results_dir / "budget.json").write_text(json.dumps({
        "model_key": "qwen35_4b", "max_model_len": 8192,
        "largest_passing": {"product": product, "max_num_seqs": product // 8192,
                            "max_model_len": 8192, "kv_tokens": 2_000_000}}))


def test_refused_window_is_retried_then_skipped_and_the_phase_continues(sweep_env, tmp_path, capsys):
    ctx = _ctx(tmp_path, hybrid=True)
    _budget_json(ctx)
    # the fake gate: refuses EVERYTHING at 16384, and S > 200 at 1536 (the canary's shape)
    sweep_env.policy["refuse"] = lambda L, S: (L == 16384) or (L == 1536 and S > 200)
    summ = sw.run_sweep(ctx, windows=(16384, 1536), budget=None, prompt_tokens=1000,
                        max_tokens=512, warmup_s=0.0, measure_s=1.0)
    out = capsys.readouterr().out
    # 16384: S_target = min(720,896 // 16384, 256) = 44 -> refused; retried 32, 24, 16; skipped
    # 1536: S_target = 256 (the hybrid cap, not 469) -> refused; retried at 192 -> up
    assert [(L, S) for L, S, _, _ in sweep_env.calls] == [
        (16384, 44), (16384, 32), (16384, 24), (16384, 16), (1536, 256), (1536, 192)]
    assert sweep_env.calls[0][3] == "petri" and sweep_env.calls[4][3] == "capability:short"
    assert len(sweep_env.standups) == 1 and sweep_env.standups[0].torn_down
    # the exact log lines of a refused-then-retried window, and of the skip
    assert ("WINDOW L=1536 (capability:short) REFUSED at S=256 (kv_under_coverage): "
            "VllmServerError: resolved scheduler state does not match") in out
    assert "— retry 1/3 at S=192" in out
    assert "WINDOW L=1536: standing at S=192 after 1 refusal(s) (target was 256)" in out
    # the 16,384 window's label names EVERY table key served at it: the certified footprint AND, since llama-paper-rows
    # (2026-09-12), the constitution-prompted GSM8K cells' 8,192-pass entry (served at exactly that window)
    assert ("WINDOW L=16384 (petri,capability:gen_long_gsm8k_longsi) SKIPPED after 4 refusals (S tried 44, 32, 24, 16; "
            "last kv_under_coverage; retries exhausted) — continuing with the next window") in out
    assert "WINDOW L=16384 (petri,capability:gen_long_gsm8k_longsi) REFUSED at S=44 (kv_under_coverage)" in out
    assert "— retry 3/3 at S=16" in out
    assert "hybrid per-slot cap" in out and "cap 256" in out
    # rows: 5 refusals + 1 skip + the 9 points of the 1536 window at S=192
    rows = [json.loads(l) for l in (ctx.results_dir / "sweep.jsonl").read_text().splitlines()]
    kinds = [r.get("kind") for r in rows]
    assert kinds.count(sw.ROW_KIND_REFUSED) == 5 and kinds.count(sw.ROW_KIND_SKIPPED) == 1
    points = [r for r in rows if sw.is_point_row(r)]
    assert [r["B"] for r in points] == [8, 16, 24, 32, 48, 64, 96, 128, 192]
    assert all(r["S"] == 192 and r["S_target"] == 256 and r["S_cap"] == 256 and r["hybrid"]
               and r["retries"] == 1 and r["window_L"] == 1536 for r in points)
    assert points[0]["tok_s"] == 8 * 512 and points[-1]["tok_s"] == 192 * 512
    refused = [r for r in rows if r.get("kind") == sw.ROW_KIND_REFUSED]
    assert [(r["window_L"], r["S_tried"], r["attempt"]) for r in refused] == [
        (16384, 44, 1), (16384, 32, 2), (16384, 24, 3), (16384, 16, 4), (1536, 256, 1)]
    assert all(r["status"] == "kv_under_coverage" and "KV cache holds" in r["error"]
               and r["eval_key"] in ("petri", "capability:short") for r in refused)
    skipped = [r for r in rows if r.get("kind") == sw.ROW_KIND_SKIPPED][0]
    assert skipped["window_L"] == 16384 and skipped["S_tried"] == [44, 32, 24, 16]
    assert skipped["reason"] == "retries exhausted" and skipped["last_status"] == "kv_under_coverage"
    # the summary was written and lists everything explicitly
    on_disk = json.loads((ctx.results_dir / "sweep_summary.json").read_text())
    assert on_disk["groups"].keys() == {"1536:passthrough"} == summ["groups"].keys()
    g = summ["groups"]["1536:passthrough"]
    assert g["S"] == 192 and g["knee_B"] == 192 and g["tok_s_at_B24"] == 24 * 512
    assert summ["windows_requested"] == [16384, 1536] and summ["windows_not_measured"] == [16384]
    assert summ["windows"] == [1536] and summ["n_rows"] == 9
    assert len(summ["window_refusals"]) == 5 and summ["window_refusals"][4]["S_tried"] == 256
    assert summ["skipped_windows"] == [{k: skipped[k] for k in summ["skipped_windows"][0]}]
    assert summ["hybrid"] is True and summ["S_cap"] == 256 and not summ["partial"]
    assert "COMPLETE WITH SKIPPED WINDOWS" in out and "skipped windows [16384]" in out


def test_no_refusal_is_the_plain_path_and_a_dense_model_keeps_the_512_cap(sweep_env, tmp_path, capsys):
    ctx = _ctx(tmp_path, hybrid=False)
    _budget_json(ctx, product=1_572_864)
    summ = sw.run_sweep(ctx, windows=(1536,), budget=None, prompt_tokens=1000,
                        max_tokens=512, warmup_s=0.0, measure_s=1.0)
    assert [(L, S) for L, S, _, _ in sweep_env.calls] == [(1536, 512)]   # 1,572,864 // 1536 = 1024 -> 512
    assert summ["window_refusals"] == [] and summ["skipped_windows"] == []
    assert summ["windows_not_measured"] == [] and summ["groups"]["1536:passthrough"]["S"] == 512
    assert summ["n_rows"] == len(sw.BATCH_LADDER)
    assert "SWEEP COMPLETE:" in capsys.readouterr().out
    # rerun: every point present -> no standup at all, summary rewritten
    sweep_env.calls.clear()
    sw.run_sweep(ctx, windows=(1536,), budget=None, prompt_tokens=1000, max_tokens=512,
                 warmup_s=0.0, measure_s=1.0)
    assert sweep_env.calls == []


def test_a_window_with_no_lower_rung_is_skipped_at_once(sweep_env, tmp_path):
    ctx = _ctx(tmp_path, hybrid=False)
    _budget_json(ctx, product=8 * 16384)                     # S = 8 at 16384: nothing below 8
    sweep_env.policy["refuse"] = lambda L, S: True
    summ = sw.run_sweep(ctx, windows=(16384,), budget=None, prompt_tokens=1000,
                        max_tokens=512, warmup_s=0.0, measure_s=1.0)
    assert [(L, S) for L, S, _, _ in sweep_env.calls] == [(16384, 8)]
    assert summ["skipped_windows"][0]["reason"] == "no smaller S on the ladder"
    assert summ["windows_not_measured"] == [16384] and summ["groups"] == {}


def test_summarize_rows_ignores_event_rows_and_names_unmeasured_windows():
    pt = {"kind": "point", "window_L": 7424, "arm": "passthrough", "B": 24, "S": 97, "tok_s": 10.0}
    ev = {"kind": "window_refused", "window_L": 1536, "S_tried": 469, "status": "kv_under_coverage",
          "error": "KV cache holds", "attempt": 1, "eval_key": "capability:short"}
    legacy = {"window_L": 8704, "arm": "passthrough", "B": 8, "S": 82, "tok_s": 5.0}   # pre-kind row
    summ = sw.summarize_rows([pt, ev, legacy], windows_requested=sw.DEFAULT_WINDOWS)
    assert set(summ["groups"]) == {"7424:passthrough", "8704:passthrough"} and summ["n_rows"] == 2
    assert summ["windows_not_measured"] == [11520, 16384, 1536]
    assert summ["window_refusals"] == [{"window_L": 1536, "eval_key": "capability:short",
                                        "S_tried": 469, "attempt": 1,
                                        "status": "kv_under_coverage", "error": "KV cache holds"}]
    assert summ["skipped_windows"] == []


# --------------------------------------------------------------------------- #
# the analyzer on a crashed task's rows (no sweep_summary.json)               #
# --------------------------------------------------------------------------- #


def test_analyzer_rebuilds_the_table_from_rows_alone_and_reports_the_missing_window(tmp_path):
    an = _load("_geom_sweep_analyze_retry", "scripts/geom_sweep_analyze.py")
    root = tmp_path / "geom-sweep"
    d = root / "qwen35_4b"
    d.mkdir(parents=True)
    (d / "budget.json").write_text(json.dumps({
        "model_key": "qwen35_4b", "max_model_len": 8192, "job_id": "373464", "node": "TESTHOST",
        "measured_on": "2026-09-03", "complete": True, "first_refused": 786_432,
        "largest_passing": {"product": 720_896, "max_num_seqs": 88, "kv_tokens": 2_000_000},
        "probes": [], "paste": None}))
    # the canary's shape: 4 windows of points (pre-kind rows, as 0f9a814 wrote them), then death
    rows = []
    for L, S in ((8704, 82), (7424, 97), (11520, 62), (16384, 44)):
        for B in sw.batch_ladder(S):
            rows.append({"model": "qwen35_4b", "window_L": L, "window_labels": list(sw.window_labels(L)),
                         "S": S, "arm": "passthrough", "B": B, "tok_s": 100.0 * B ** 0.5,
                         "lat_p50_s": 1.0, "sm_mean": 80.0, "preemptions_delta": 0})
    lines = [json.dumps(r) for r in rows] + ['{"torn": tru']       # a torn last line
    (d / "sweep.jsonl").write_text("\n".join(lines) + "\n")
    res = an.load_results(root)
    s = res["qwen35_4b"]["summary"]
    assert s is not None and s["rebuilt_from_rows"] is True
    assert s["budget"] == 720_896 and s["n_rows"] == len(rows)
    assert sorted(s["windows"]) == [7424, 8704, 11520, 16384]
    assert s["windows_not_measured"] == [1536] and s["window_refusals"] == []
    assert s["groups"]["16384:passthrough"]["S"] == 44
    md = an.markdown_tables(res)
    assert "summary REBUILT from sweep.jsonl" in md
    assert "NOT MEASURED: window L=1536 (capability:short) — no point rows" in md
    assert "| 16384 | petri,capability:gen_long_gsm8k_longsi | passthrough | 44 | 24 |" in md      # both keys served at 16,384
    # an older summary lacking the bookkeeping gets it filled from the rows
    (d / "sweep_summary.json").write_text(json.dumps(
        {"groups": {}, "old_S": 24, "windows_requested": [16384, 1536]}))
    s2 = an.load_results(root)["qwen35_4b"]["summary"]
    assert s2["windows_not_measured"] == [1536]
    assert s2["rebuilt_from_rows"] and s2["stored_summary_present"]   # rows are the truth (review R3)
    # with a refused-window event in the rows, the report names it
    rows.append({"kind": "window_refused", "model": "qwen35_4b", "window_L": 1536,
                 "eval_key": "capability:short", "S_tried": 256, "S_target": 256, "attempt": 1,
                 "status": "kv_under_coverage", "error": "KV cache holds 600000 tokens"})
    rows.append({"kind": "window_skipped", "model": "qwen35_4b", "window_L": 1536,
                 "S_tried": [256, 192, 128, 96], "reason": "retries exhausted",
                 "last_status": "kv_under_coverage"})
    (d / "sweep.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / "sweep_summary.json").unlink()
    md = an.markdown_tables(an.load_results(root))
    assert "REFUSED standup: L=1536 at S=256 (target 256, attempt 1): kv_under_coverage" in md
    assert "SKIPPED window: L=1536 after S tried [256, 192, 128, 96] (retries exhausted" in md
