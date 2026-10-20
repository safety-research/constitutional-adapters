"""The timing harness for the native residual slot.

CPU-only (no GPU, no network, no vLLM): the closed-loop module is the certification driver's copy
(signature, source and prompt bytes pinned); the sampler runs against a fake aiohttp session + fake
/metrics; the ms/step helpers, the row builder and the hook-only subtraction over fake stats; the
harness's pure helpers (the standup kwargs, the composer binding for prefix caching OFF, the recorded
launch facts + the protocol refusals with KNOWN-BAD inputs, the vector arm ABSENT by name); the
sbatch's cluster hard rules + shape; the transcribed lens rows held to the doc's own lines (with a
KNOWN-BAD perturbation); the report over synthetic per-target JSONs (markdown into tmp, one PNG per
target; never a file under the repository's docs/).
"""
from __future__ import annotations

import asyncio
import inspect
import json
import re
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import multilora_cert as mc  # noqa: E402
import native_timing_72 as nt  # noqa: E402
import native_timing_72_report as rep  # noqa: E402
from sisteer.capeval.serving import closed_loop as cl  # noqa: E402

SBATCH = REPO / "scripts" / "sdfbase" / "native_timing_72.sbatch"
LENS_JSON = REPO / "configs" / "timing" / "lens_stacked_72_psr_rows.json"
DOC = REPO / "docs" / "lens_stacked_reductions_scoping.md"
PSR_CELL = "constv2-mix-full__psr_ce_all"
PSR_H = "si_constv2_mix_full__psr_ce_all"   # a synthetic harness id for the fixtures (never resolved here)


# --------------------------------------------------------------------------- #
# 1. closed_loop is the certification driver's copy                            #
# --------------------------------------------------------------------------- #
def test_closed_loop_is_the_certs_copy():
    assert inspect.signature(cl.closed_loop_via_proxy) == inspect.signature(mc.closed_loop_via_proxy)
    assert inspect.getsource(cl.closed_loop_via_proxy) == inspect.getsource(mc.closed_loop_via_proxy)
    assert inspect.signature(cl.read_metrics) == inspect.signature(mc.read_metrics)
    assert inspect.getsource(cl.read_metrics) == inspect.getsource(mc.read_metrics)
    assert cl.FILLER_PROMPT.encode() == mc.FILLER_PROMPT.encode()
    assert cl.METRIC_RE.pattern == mc.METRIC_RE.pattern and cl.METRIC_RE.flags == mc.METRIC_RE.flags
    assert asyncio.iscoroutinefunction(cl.closed_loop_via_proxy)


def test_gate_filler_and_targets_are_the_gates_text():
    """The prompt recipe copies the lens gate's FILLER passage and token ladder verbatim (the doc's
    ~800-token prompt is the 5th rung)."""
    gate = (REPO / "scripts" / "lens_vectorize_gate.py").read_text(encoding="utf-8")
    assert "The river runs quietly past the old stone mill while the miller counts sacks " in gate
    assert "of grain and the afternoon light moves slowly across the wooden floor. " in gate
    assert cl.GATE_FILLER == ("The river runs quietly past the old stone mill while the miller counts sacks "
                              "of grain and the afternoon light moves slowly across the wooden floor. ")
    assert "PROMPT_TOKEN_TARGETS = (48, 96, 200, 400, 800, 1200)" in gate
    assert cl.GATE_PROMPT_TOKEN_TARGETS == (48, 96, 200, 400, 800, 1200)
    assert cl.GATE_TIMING_PROMPT_TARGET == 800 and cl.GATE_TIMING_PROMPT_INDEX == 4
    # the gate's implied_step_ms is the formula behind every §7.2 number — mirrored exactly
    assert "return 1000.0 * float(running_mean) / float(gen_tok_s)" in gate
    assert cl.ms_per_step_gate(60.0, 2000.0) == pytest.approx(30.0)


class _FakeResp:
    def __init__(self, text: str, status: int = 200):
        self._t, self.status = text, status

    def read(self):
        return self._t.encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_read_metrics_sums_labelled_series_and_names_http_status(monkeypatch):
    text = ("# HELP vllm:generation_tokens_total x\n"
            "vllm:generation_tokens_total{model_name=\"a\"} 100.0\n"
            "vllm:generation_tokens_total{model_name=\"b\"} 23\n"
            "vllm:num_requests_running{model_name=\"a\"} 7\n"
            "vllm:iteration_tokens_total_count{model_name=\"a\"} 4100\n"
            "vllm:iteration_tokens_total_bucket{le=\"8\",model_name=\"a\"} 12\n"
            "python_gc_objects_collected_total{generation=\"0\"} 5\n")
    monkeypatch.setattr(cl.urllib.request, "urlopen", lambda req, timeout=10: _FakeResp(text))
    status, m = cl.read_metrics("http://x/metrics", {"Authorization": "Bearer t"})
    assert status == 200
    assert m["vllm:generation_tokens_total"] == pytest.approx(123.0)
    assert m["vllm:num_requests_running"] == 7
    assert m[cl.ITERATION_COUNTER_METRIC] == 4100
    assert "python_gc_objects_collected_total" not in m

    import urllib.error

    def boom(req, timeout=10):
        raise urllib.error.HTTPError(req.full_url, 401, "nope", hdrs=None, fp=None)
    monkeypatch.setattr(cl.urllib.request, "urlopen", boom)
    assert cl.read_metrics("http://x/metrics", {}) == (401, {})


# --------------------------------------------------------------------------- #
# 2. the sampler against a fake session + fake /metrics                        #
# --------------------------------------------------------------------------- #
class _FakeAiohttp(types.ModuleType):
    """Just enough of aiohttp for closed_loop_via_proxy: every post is a 200 with the exact-length usage
    after a short await; the bodies are captured so the bound prompt is observable."""

    def __init__(self, posted, per_request_s: float = 0.01):
        super().__init__("aiohttp")
        mod = self

        class ClientTimeout:
            def __init__(self, total=None):
                self.total = total

        class TCPConnector:
            def __init__(self, limit=None):
                self.limit = limit

        class _Resp:
            def __init__(self, body):
                self.status = 200
                self._body = body

            async def json(self):
                return {"usage": {"completion_tokens": int(self._body["max_tokens"])}, "choices": [{"message": {"content": "x"}}]}

            async def text(self):
                return ""

            async def __aenter__(self):
                await asyncio.sleep(per_request_s)
                return self

            async def __aexit__(self, *a):
                return False

        class ClientSession:
            def __init__(self, connector=None, timeout=None):
                self.connector = connector

            def post(self, url, json=None, headers=None, timeout=None):
                posted.append({"url": url, "json": dict(json), "headers": dict(headers or {})})
                return _Resp(json)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        mod.ClientTimeout, mod.TCPConnector, mod.ClientSession = ClientTimeout, TCPConnector, ClientSession


def _fake_metrics_reader(tok_per_s: float = 1000.0, running: int = 3, steps_per_s: float = 50.0):
    t0 = time.monotonic()

    def read(url, headers):
        dt = time.monotonic() - t0
        return 200, {cl.GENERATION_COUNTER_METRIC: tok_per_s * dt, "vllm:num_requests_running": running,
                     "vllm:num_requests_waiting": 0, cl.ITERATION_COUNTER_METRIC: steps_per_s * dt}
    return read


def test_closed_loop_via_proxy_runs_against_fakes_and_sends_the_bound_prompt(monkeypatch):
    posted = []
    monkeypatch.setitem(sys.modules, "aiohttp", _FakeAiohttp(posted))
    monkeypatch.setattr(cl, "read_metrics", _fake_metrics_reader())
    with cl.prompt_bound("Passage 5. the timing passage"):
        stats, gauges = asyncio.run(cl.closed_loop_via_proxy(
            "http://proxy/v1", "tok", "vm-psr", metrics_url="http://engine/metrics", metrics_headers={"Authorization": "Bearer k"},
            B=3, max_tokens=16, warmup_s=0.05, window_s=0.3, exact_length=True, sample_s=0.02))
    assert cl.FILLER_PROMPT == mc.FILLER_PROMPT          # restored after the block
    assert posted and all(p["json"]["messages"][0]["content"] == "Passage 5. the timing passage" for p in posted)
    assert all(p["json"]["min_tokens"] == 16 and p["json"]["ignore_eos"] is True and p["json"]["model"] == "vm-psr" for p in posted)
    assert all(p["headers"] == {"Authorization": "Bearer tok"} for p in posted)
    assert stats["errors"] == 0 and stats["requests_done"] > 0
    assert stats["server_tokens"] is not None and stats["server_tokens"] > 0
    assert stats["server_tok_s"] == pytest.approx(1000.0, rel=0.25)
    assert stats["running_mean"] == pytest.approx(3.0)
    assert 0.25 <= stats["window_s"] <= 0.6
    assert gauges and all("t" in g and g["running"] == 3 for g in gauges)
    ms = cl.ms_per_step_all(stats, 3)
    assert ms["gate"] == pytest.approx(3.0, rel=0.3) and ms["nominal"] == pytest.approx(3.0, rel=0.3) and ms["counter"] is None


def test_step_counter_sampler_aligns_to_the_loops_window():
    reader = _fake_metrics_reader(steps_per_s=100.0)
    s = cl.StepCounterSampler("http://engine/metrics", {}, sample_s=0.02, reader=reader).start()
    time.sleep(0.05)
    t_start = time.monotonic()
    time.sleep(0.3)
    t_end = time.monotonic()
    time.sleep(0.05)
    samples = s.stop()
    gauges = [{"t": t_start, "running": 3}, {"t": t_end, "running": 3}]
    span = cl.counter_span_over_gauges(samples, gauges)
    assert span["steps"] is not None and span["span_s"] is not None and span["n_samples"] >= 3
    assert span["steps"] / span["span_s"] == pytest.approx(100.0, rel=0.2)
    assert cl.ms_per_step_from_counter(span["steps"], span["span_s"]) == pytest.approx(10.0, rel=0.2)
    # an engine without the histogram: None with the reason, never a guess
    none_samples = [(t, None, g) for t, _, g in samples]
    span2 = cl.counter_span_over_gauges(none_samples, gauges)
    assert span2["steps"] is None and "not exposed" in span2["reason"]
    assert cl.counter_span_over_gauges(samples, [])["steps"] is None


def test_ms_per_step_helpers_three_ways_and_none_cases():
    assert cl.ms_per_step_gate(60.0, 2000.0) == pytest.approx(30.0)      # the gate: running_mean-based
    assert cl.ms_per_step_nominal(64, 2000.0) == pytest.approx(32.0)     # the cert: B-based
    assert cl.ms_per_step_from_counter(900, 45.0) == pytest.approx(50.0)  # the engine's iteration counter
    for bad in ((None, 2000.0), (0, 2000.0), (60.0, None), (60.0, 0.0), (60.0, -1.0)):
        assert cl.ms_per_step_gate(*bad) is None
    assert cl.ms_per_step_nominal(0, 2000.0) is None and cl.ms_per_step_nominal(8, None) is None
    assert cl.ms_per_step_from_counter(0, 45.0) is None and cl.ms_per_step_from_counter(10, None) is None
    all3 = cl.ms_per_step_all({"running_mean": 60.0, "server_tok_s": 2000.0}, 64, counter_steps=900, counter_span_s=45.0)
    assert all3 == {"gate": pytest.approx(30.0), "nominal": pytest.approx(32.0), "counter": pytest.approx(50.0)}
    # a window whose engine counter was unreadable carries None everywhere (the client count is never promoted)
    assert cl.ms_per_step_all({"running_mean": 60.0, "server_tok_s": None, "client_tok_s": 1000.0}, 64) == {"gate": None, "nominal": None, "counter": None}


def test_prompt_bound_restores_on_exception_and_refuses_empty():
    before = cl.FILLER_PROMPT
    with pytest.raises(RuntimeError):
        with cl.prompt_bound("x"):
            assert cl.FILLER_PROMPT == "x"
            raise RuntimeError("boom")
    assert cl.FILLER_PROMPT == before
    with pytest.raises(ValueError):
        with cl.prompt_bound(""):
            pass


def test_build_timing_prompt_fits_the_gates_800_token_target():
    per_repeat = 31   # a filler repeat ~ 31 tokens on the real tokenizers; any linear count works here

    def count(text: str) -> int:
        return 6 + per_repeat * text.count(cl.GATE_FILLER)
    p = cl.build_timing_prompt(count)
    assert p["text"].startswith("Passage 5. ") and p["target_tokens"] == 800
    assert p["tokens"] <= 800 < count(p["text"] + cl.GATE_FILLER)
    assert p["tokens"] == count(p["text"]) and p["repeats"] == p["text"].count(cl.GATE_FILLER)
    assert cl.build_timing_prompt(count, target_tokens=200)["tokens"] <= 200


# --------------------------------------------------------------------------- #
# 3. the harness's pure helpers                                                 #
# --------------------------------------------------------------------------- #
def test_vector_arm_recorded_absent_by_name_when_no_vector_cell():
    for t in ("llama8b", "gemma4e4b", "qwen35_4b", "nemotron3nano4b"):
        assert nt.vector_arm_record(t, None) == {"status": "absent",
                                                 "reason": f"no all-positions vector-kind object is staged on {t} (D4)"}
    assert nt.vector_arm_record("gemma4e4b", "x__vector_all") == {"status": "present", "cell": "x__vector_all"}


def test_config_paths_follow_f22_resolve_config():
    cfg, man = nt.config_paths(Path("/r"), "gemma4e4b", "psrcert")
    assert cfg == Path("/r/configs/sdfbase_full22_gemma4e4b_psrcert.yaml")
    assert man == Path("/r/configs/capeval/manifests/petri_full22_gemma4e4b_psrcert.yaml")
    assert nt.config_paths(Path("/r"), "llama8b", "main")[0] == Path("/r/configs/sdfbase_full22_llama8b.yaml")


def test_standup_kwargs_is_the_certs_shape_at_the_72_geometry():
    kw = nt.standup_kwargs(repo_dir=Path("/r"), config=Path("/r/c.yaml"), manifest=Path("/r/m.yaml"),
                           defenses_dir=Path("/v"), out=Path("/o/standup"), target="gemma4e4b", cells=[PSR_CELL],
                           expect_object_sha256=[f"{PSR_CELL}=abc"], adapter_bank=None)
    assert kw["cell"] == [PSR_CELL] and kw["model_key"] == "gemma4e4b" and kw["eval"] == "families_served:short"
    assert kw["max_model_len"] == 4096 and kw["max_num_seqs"] == 64
    assert kw["geometry_override_reason"] and "4096 x 64" in kw["geometry_override_reason"]
    assert kw["expect_object_sha256"] == [f"{PSR_CELL}=abc"] and "adapter_bank" not in kw
    assert kw["probe_max_tokens"] == 48 and kw["gpu_mem_util"] == "0.9"
    # every keyword is one bring_up_lora accepts (the seam is the library, not a subprocess)
    import lora_serving_up as lsu
    accepted = set(inspect.signature(lsu.bring_up_lora).parameters)
    assert set(kw) <= accepted, set(kw) - accepted
    kw2 = nt.standup_kwargs(repo_dir=Path("/r"), config=Path("c"), manifest=Path("m"), defenses_dir=Path("v"), out=Path("o"),
                            target="llama8b", cells=[PSR_CELL, "x__vector_all"], expect_object_sha256=[], adapter_bank=Path("/bank"))
    assert kw2["adapter_bank"] == "/bank" and kw2["cell"] == [PSR_CELL, "x__vector_all"]
    with pytest.raises(nt.TimingRefused):
        nt.standup_kwargs(repo_dir=Path("/r"), config=Path("c"), manifest=Path("m"), defenses_dir=Path("v"), out=Path("o"),
                          target="llama8b", cells=[], expect_object_sha256=[], adapter_bank=None)


def test_prefix_caching_off_binds_the_composer_and_restores_it():
    seen = []

    def build_server_launch(spec, **kw):
        seen.append(dict(kw))
        return ("launch", kw.get("prefix_caching"))
    fake = types.SimpleNamespace(build_server_launch=build_server_launch)
    orig = fake.build_server_launch
    with nt.prefix_caching_off(fake) as calls:
        assert fake.build_server_launch is not orig
        out = fake.build_server_launch("spec", lora_native=True, max_model_len=4096, max_num_seqs=64)
        assert out == ("launch", False)
        assert seen[-1]["prefix_caching"] is False and seen[-1]["lora_native"] is True
        # even a caller asking for ON is bound OFF (the protocol owns the choice for the standup's duration)
        fake.build_server_launch("spec", prefix_caching=True)
        assert seen[-1]["prefix_caching"] is False
        with pytest.raises(nt.TimingRefused):
            fake.build_server_launch("spec", exact_replay_strategy="batch_invariant")
    assert fake.build_server_launch is orig
    assert calls == [{"prefix_caching": False, "exact_replay_strategy": None, "max_model_len": 4096, "max_num_seqs": 64, "lora_native": True},
                     {"prefix_caching": False, "exact_replay_strategy": None, "max_model_len": None, "max_num_seqs": None, "lora_native": None}]
    # the real module attribute is what native_standup's composer calls (a local `import vllm_server as vs`)
    src = (REPO / "src" / "sisteer" / "capeval" / "serving" / "native_standup.py").read_text(encoding="utf-8")
    assert "from sisteer.capeval.serving import vllm_server as vs" in src and "vs.build_server_launch(" in src


def _good_fixture():
    launch = types.SimpleNamespace(cmd=("vllm", "serve", "m", "--no-enforce-eager", "--no-enable-prefix-caching", "--enable-lora"),
                                   env={"VLLM_PLUGINS": ""}, prefix_caching=False)
    endpoint = {"engine": "lora_native", "serving_engine": "lora_native", "attested_enforce_eager": False,
                "attested_cudagraph_mode": "FULL_AND_PIECEWISE", "output_diagnostics": False,
                "residual_slot": {"armed": True, "cells": {PSR_H: {"kind": "psr", "position_mode": "all", "residual_fingerprint": "f" * 64}}},
                "lora_cells": [PSR_H], "cells_skipped": [], "max_model_len": 4096, "max_num_seqs": 64, "attested_max_num_seqs": 64,
                "geometry": {"adjustments": []}, "geometry_override_reason": nt.GEOMETRY_OVERRIDE_REASON, "geometry_lowered": None,
                "adapters": {PSR_H: {"name": "a1", "r": 1, "petri_source_kind": "psr", "residual_fingerprint": "f" * 64, "position_mode": "all"}},
                "vmodels": {"none": "m", PSR_H: "m--psr", "none__lora0": "m--zero"}}
    attestation = {"schema_version": 2, "engine": {"enforce_eager": False, "cudagraph_mode": "FULL_AND_PIECEWISE", "enable_prefix_caching": False},
                   "scheduler_geometry": {"workers": [{"batch_invariant_env": False, "batch_invariant_active": False, "batch_invariant_surface": "vllm.envs"}]}}
    return launch, endpoint, attestation


def test_launch_facts_record_what_was_composed_and_attested():
    launch, endpoint, att = _good_fixture()
    facts = nt.launch_facts(launch=launch, endpoint=endpoint, attestation=att, composed_calls=[{"prefix_caching": False}],
                            harness_env={}, versions={"vllm": "0.25.1"})
    assert facts["prefix_caching"]["composed"] is False and facts["prefix_caching"]["argv_off_token"] is True
    assert facts["prefix_caching"]["argv_on_token"] is False
    assert facts["prefix_caching"]["attested_enable_prefix_caching"] == [False]
    assert facts["batch_invariance"] == {"harness_env": None, "launch_env": None, "attested_env": [False], "attested_active": [False],
                                         "effective": 0, "mechanism": nt.BATCH_INVARIANCE_MECHANISM}
    assert facts["engine"]["residual_slot"]["armed"] is True and facts["engine"]["geometry"] == {
        "L": 4096, "S": 64, "attested_S": 64, "adjustments": [], "override_reason": nt.GEOMETRY_OVERRIDE_REASON, "lowered": None}
    assert facts["engine"]["adapters"][PSR_H]["petri_source_kind"] == "psr" and facts["versions"] == {"vllm": "0.25.1"}
    assert nt.protocol_problems(facts, psr_harness_cell=PSR_H) == []


@pytest.mark.parametrize("mutate, needle", [
    (lambda l, e, a, env: setattr(l, "prefix_caching", True), "composed prefix_caching"),
    (lambda l, e, a, env: setattr(l, "cmd", ("vllm", "serve", "--enable-prefix-caching")), "--no-enable-prefix-caching"),
    (lambda l, e, a, env: a["engine"].__setitem__("enable_prefix_caching", True), "attested enable_prefix_caching"),
    (lambda l, e, a, env: a["engine"].__delitem__("enable_prefix_caching"), "no enable_prefix_caching fact"),
    (lambda l, e, a, env: env.__setitem__("VLLM_BATCH_INVARIANT", "1"), "harness environment sets VLLM_BATCH_INVARIANT"),
    (lambda l, e, a, env: l.env.__setitem__("VLLM_BATCH_INVARIANT", "1"), "launch env sets VLLM_BATCH_INVARIANT"),
    (lambda l, e, a, env: a["scheduler_geometry"]["workers"][0].__setitem__("batch_invariant_active", True), "attest batch invariance ON"),
    (lambda l, e, a, env: e["residual_slot"].__setitem__("armed", False), "residual_slot.armed"),
    (lambda l, e, a, env: e["residual_slot"].__setitem__("cells", {}), "does not carry the psr cell"),
    (lambda l, e, a, env: e.__setitem__("output_diagnostics", True), "output_diagnostics"),
    (lambda l, e, a, env: e.__setitem__("attested_enforce_eager", True), "not graphs-on"),
    (lambda l, e, a, env: e.__setitem__("attested_cudagraph_mode", "NONE"), "not graphs-on"),
    (lambda l, e, a, env: e.__setitem__("max_num_seqs", 32), "served geometry"),
    (lambda l, e, a, env: e.__setitem__("attested_max_num_seqs", 48), "attested max_num_seqs"),
    (lambda l, e, a, env: e["geometry"].__setitem__("adjustments", [{"kind": "kernel", "from_S": 64, "to_S": 32}]), "adjusted"),
    (lambda l, e, a, env: e.__setitem__("cells_skipped", [{"cell_id": PSR_H}]), "cells_skipped"),
    (lambda l, e, a, env: e.__setitem__("engine", "lens"), "engine"),
])
def test_protocol_problems_known_bad(mutate, needle):
    launch, endpoint, att = _good_fixture()
    env = {}
    mutate(launch, endpoint, att, env)
    facts = nt.launch_facts(launch=launch, endpoint=endpoint, attestation=att, composed_calls=[], harness_env=env)
    problems = nt.protocol_problems(facts, psr_harness_cell=PSR_H)
    assert problems and any(needle in p for p in problems), problems


def _stats(running_mean: float, tok_s: float, **extra):
    base = {"window_s": 45.0, "server_tokens": tok_s * 45.0, "server_tok_s": tok_s, "client_tokens": tok_s * 45.0,
            "client_tok_s": tok_s, "requests_done": 40, "running_mean": running_mean, "running_max": int(running_mean) + 1,
            "waiting_max": 0, "latency_p50_s": 3.0, "latency_p95_s": 3.5, "errors": 0, "error_sample": None}
    base.update(extra)
    return base


def test_build_row_and_hook_only_from_the_same_run():
    eng = {"enforce_eager": False, "cudagraph_mode": "FULL_AND_PIECEWISE"}
    prompt = {"tokens": 799, "repeats": 26, "target_tokens": 800, "sha256": "0" * 64, "served_prompt_tokens": 803}
    common = dict(target="gemma4e4b", prompt=prompt, node="gpu-3", sha="abc123def", date="2026-09-14T12:00:00+00:00",
                  engine=eng, standup_out="/o/standup")
    none8 = nt.build_row(arm="none", cell=None, harness_cell="none", vmodel="m", B=8, stats=_stats(8.0, 400.0),
                         counter={"steps": 2250.0, "span_s": 45.0}, **common)
    psr8 = nt.build_row(arm="psr", cell=PSR_CELL, harness_cell=PSR_H, vmodel="m--psr", B=8, stats=_stats(8.0, 320.0),
                        counter={"steps": 1800.0, "span_s": 45.0}, **common)
    none32 = nt.build_row(arm="none", cell=None, harness_cell="none", vmodel="m", B=32, stats=_stats(30.0, 1000.0),
                          counter={"steps": None, "span_s": None, "reason": "not exposed"}, **common)
    psr32 = nt.build_row(arm="psr", cell=PSR_CELL, harness_cell=PSR_H, vmodel="m--psr", B=32, stats=_stats(30.0, 750.0),
                         counter={"steps": None, "span_s": None, "reason": "not exposed"}, **common)
    psr64 = nt.build_row(arm="psr", cell=PSR_CELL, harness_cell=PSR_H, vmodel="m--psr", B=64, stats=_stats(60.0, 1500.0),
                         counter={"steps": None, "span_s": None}, **common)
    # the §7.2 fields
    for r in (none8, psr8):
        assert r["geometry"] == {"L": 4096, "S": 64} and r["window_s"] == 45.0 and r["warmup_s"] == 10.0
        assert r["max_tokens"] == 128 and r["exact_length"] is True and r["prefix_caching"] is False and r["batch_invariance"] == 0
        assert r["label"] == "native (measured)" and r["node"] == "gpu-3" and r["sha"] == "abc123def" and r["date"]
        assert r["engine"] == eng and r["prompt"] == prompt and r["standup_out"] == "/o/standup"
    # ms/step both ways (and the counter): none@8 -> 20 ms (gate == nominal at running_mean == B), counter 20 ms
    assert none8["ms_per_step"] == {"gate": pytest.approx(20.0), "nominal": pytest.approx(20.0), "counter": pytest.approx(20.0)}
    assert psr8["ms_per_step"] == {"gate": pytest.approx(25.0), "nominal": pytest.approx(25.0), "counter": pytest.approx(25.0)}
    # the gate reading follows the ENGINE's running mean, the nominal reading the client B
    assert none32["ms_per_step"]["gate"] == pytest.approx(30.0) and none32["ms_per_step"]["nominal"] == pytest.approx(32.0)
    assert none32["ms_per_step"]["counter"] is None and none8["ms_per_step_primary"] == "gate"
    rows = nt.fill_hook_only([none8, psr8, none32, psr32, psr64])
    assert psr8["passthrough_floor_ms"] == none8["ms_per_step"]
    assert psr8["hook_only_ms"] == {"gate": pytest.approx(5.0), "nominal": pytest.approx(5.0), "counter": pytest.approx(5.0)}
    assert none8["hook_only_ms"] == {"gate": pytest.approx(0.0), "nominal": pytest.approx(0.0), "counter": pytest.approx(0.0)}
    assert psr32["hook_only_ms"] == {"gate": pytest.approx(10.0), "nominal": pytest.approx(42.6667 - 32.0, abs=1e-3), "counter": None}
    assert psr64["passthrough_floor_ms"] is None and psr64["hook_only_ms"] is None   # no floor at B=64 in this run: never borrowed
    assert rows[1] is psr8


def test_window_plan_pairs_the_floor_and_the_arm_at_each_B():
    assert nt.window_plan([8, 32, 64]) == [(8, "none"), (8, "psr"), (32, "none"), (32, "psr"), (64, "none"), (64, "psr")]
    assert nt.window_plan([8], ["none", "psr", "vector"]) == [(8, "none"), (8, "psr"), (8, "vector")]
    assert nt.parse_batches("8,32,64") == [8, 32, 64]
    for bad in ("", "0,8", "a"):
        with pytest.raises((nt.TimingRefused, ValueError)):
            nt.parse_batches(bad)


def test_cli_defaults_are_the_72_protocol():
    ap = nt.build_parser()
    a = ap.parse_args(["--target", "gemma4e4b", "--view", "psrcert", "--out", "/o", "--defenses-dir", "/v"])
    assert a.cell == PSR_CELL and a.batches == "8,32,64" and a.window_s == 45.0 and a.warmup_s == 10.0
    assert a.max_tokens == 128 and a.max_model_len == 4096 and a.max_num_seqs == 64 and a.prompt_tokens == 800
    assert a.vector_cell is None and a.adapter_bank is None and a.expect_object_sha256 == []
    a2 = ap.parse_args(["--target", "llama8b", "--view", "psrcert", "--out", "/o", "--defenses-dir", "/v",
                        "--expect-object-sha256", f"{PSR_CELL}=ab", "--vector-cell", "x__vector_all", "--adapter-bank", "/b"])
    assert a2.expect_object_sha256 == [f"{PSR_CELL}=ab"] and a2.vector_cell == "x__vector_all"


def test_exact_length_probe_refuses_a_short_completion(monkeypatch):
    def urlopen(req, timeout=600.0):
        body = json.loads(req.data)
        assert body["min_tokens"] == body["max_tokens"] == 128 and body["ignore_eos"] is True
        return _FakeResp(json.dumps({"usage": {"prompt_tokens": 803, "completion_tokens": 128}}))
    monkeypatch.setattr(nt.urllib.request, "urlopen", urlopen)
    usage = nt.exact_length_probe("http://p/v1", "k", "m", "prompt text", max_tokens=128)
    assert usage == {"prompt_tokens": 803, "completion_tokens": 128}
    monkeypatch.setattr(nt.urllib.request, "urlopen",
                        lambda req, timeout=600.0: _FakeResp(json.dumps({"usage": {"prompt_tokens": 803, "completion_tokens": 5}})))
    with pytest.raises(nt.TimingRefused):
        nt.exact_length_probe("http://p/v1", "k", "m", "prompt text", max_tokens=128)


def test_tokenizer_count_fn_handles_list_and_dict_renders():
    class Tok:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize, return_dict):
            assert add_generation_prompt and tokenize and not return_dict
            return list(range(len(messages[0]["content"]) // 4))
    assert nt.tokenizer_count_fn(Tok())("x" * 40) == 10

    class TokDict(Tok):
        def apply_chat_template(self, *a, **k):
            return {"input_ids": [1, 2, 3]}
    assert nt.tokenizer_count_fn(TokDict())("abc") == 3


def test_harness_main_refuses_a_batch_invariant_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with pytest.raises(nt.TimingRefused):
        nt.main(["--target", "gemma4e4b", "--view", "psrcert", "--out", str(tmp_path), "--defenses-dir", str(tmp_path)])


# --------------------------------------------------------------------------- #
# 4. the sbatch                                                                  #
# --------------------------------------------------------------------------- #
def test_sbatch_bash_n():
    subprocess.run(["bash", "-n", str(SBATCH)], check=True)


def test_sbatch_hard_rules_and_shape():
    txt = SBATCH.read_text(encoding="utf-8")
    # the cluster hard rules (each has caused a real outage) — test_sbatch_lint's pattern
    assert not re.search(r"^#SBATCH --export", txt, re.M)
    assert "--export=" not in txt and "--export " not in txt
    assert not re.search(r"^#SBATCH --mem=0", txt, re.M) and "--mem=0" not in txt
    assert "CUDA_VISIBLE_DEVICES" not in txt
    assert not re.search(r"^#SBATCH --qos=dev", txt, re.M)
    # the header the plan names
    for line in ("#SBATCH --partition=general", "#SBATCH --qos=high", "#SBATCH --gres=gpu:1", "#SBATCH --cpus-per-task=16",
                 "#SBATCH --mem=128G", "#SBATCH --time=01:00:00", "#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out",
                 "#SBATCH --open-mode=append", "#SBATCH --no-requeue", "#SBATCH --signal=B:TERM@120"):
        assert re.search("^" + re.escape(line) + r"\s*$", txt, re.M), line
    # positional args only: <target> <view> <sha> [psr_cell] [vector_cell]
    assert 'KEY="${1:?target}"' in txt and 'VIEW="${2:?view}"' in txt and 'WAVE_SHA="${3:?wave sha}"' in txt
    assert 'PSR_CELL="${4:-constv2-mix-full__psr_ce_all}"' in txt and 'VECTOR_CELL="${5:-}"' in txt
    # the per-sha checkout's common.sh + the f22 scaffolding, the nemotron serving-venv variant, the bank view
    assert 'source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"' in txt
    for tok in ('f22_init "$WAVE_SHA"', 'f22_resolve_config "$KEY" "$VIEW"', "f22_driver_venv", "f22_serving_venv nemotron",
                'f22_bank_view "$MPY"', "f22_result ", "$FIGS_DIR", "VIEW_MANIFEST.json", "--expect-object-sha256"):
        assert tok in txt, tok
    # the harness under srun from the checkout in the serving venv, at the §7.2 protocol, records under exp/native_timing_72
    assert 'srun --ntasks=1 "$SPY" scripts/sdfbase/native_timing_72.py' in txt
    assert "--batches 8,32,64 --window-s 45 --warmup-s 10 --max-tokens 128 --max-model-len 4096 --max-num-seqs 64" in txt
    assert 'OUT="$SHARED/exp/native_timing_72/${KEY}_${SLURM_JOB_ID:-manual}"' in txt
    assert "mt_serving_up.py" not in txt          # the library seam, never a subprocess of the mt front
    assert "unset VLLM_BATCH_INVARIANT" in txt    # BI unset is the protocol
    # the disarming cleanup: `trap '' SIGTERM` is the handler's FIRST statement; the trap covers EXIT + the signals
    m = re.search(r"^cleanup\(\)\{\n(.*?)\n\}", txt, re.M | re.S)
    assert m and m.group(1).splitlines()[0].strip() == "trap '' SIGTERM"
    assert "trap cleanup EXIT SIGTERM SIGINT SIGQUIT" in txt
    assert "kill -TERM -$$ 2>/dev/null; wait" in m.group(1)
    assert "glm46vflash" in txt and "EXCLUDED" in txt


# --------------------------------------------------------------------------- #
# 5. the reused lens rows vs the doc                                             #
# --------------------------------------------------------------------------- #
def test_lens_rows_json_matches_the_doc_lines_and_known_bad_is_caught():
    lens = json.loads(LENS_JSON.read_text(encoding="utf-8"))
    doc = DOC.read_text(encoding="utf-8")
    assert rep.check_lens_rows_against_doc(lens, doc) == []
    rows = lens["rows"]
    assert {(r["target"], r["B"]) for r in rows} == {(t, B) for t in ("llama8b", "gemma4e4b", "qwen35_4b", "nemotron3nano4b") for B in (8, 32, 64)}
    assert all(r["label"] == "lens (reused, §7.2, 2026-09-04)" for r in rows) and lens["label"] == "lens (reused, §7.2, 2026-09-04)"
    assert lens["measurement"]["stacked"]["gate_run"] == 382416 and lens["measurement"]["per_row"]["gate_run"] == 384162
    assert lens["measurement"]["date"] == "2026-09-04"
    cited = {r["source"] for r in rows}
    want_lines = {*range(809, 812), *range(821, 824), *range(830, 833), *range(842, 845)}
    assert {int(s.rsplit(" ", 1)[1]) for s in cited} == want_lines
    assert all(s.startswith("docs/lens_stacked_reductions_scoping.md §7.2.1 line ") for s in cited)
    # a few spot values straight off the doc (llama8b B=8: 63.6 / 57.1 | 31.4 | 17.2 | ... | 42.6 / 16.9 / 4.5)
    r = next(x for x in rows if x["target"] == "llama8b" and x["B"] == 8)
    assert r["legacy_ms_per_step"] == {"382416": 63.6, "384162": 57.1} and r["per_row_ms_per_step"] == 31.4
    assert r["stacked_ms_per_step"] == 17.2 and r["hook_only_ms"] == {"legacy": 42.6, "per_row": 16.9, "stacked": 4.5}
    r = next(x for x in rows if x["target"] == "nemotron3nano4b" and x["B"] == 64)
    assert r["stacked_ms_per_step"] == 39.0 and r["hook_only_ms"]["stacked"] == 16.8 and r["legacy_ms_per_step"]["384162"] == 513.3
    # KNOWN-BAD: one number perturbed in memory -> the check names the row and the field
    bad = json.loads(json.dumps(lens))
    bad["rows"][5]["stacked_ms_per_step"] += 0.1
    probs = rep.check_lens_rows_against_doc(bad, doc)
    assert len(probs) == 1 and "stacked" in probs[0] and f"B={bad['rows'][5]['B']}" in probs[0] and bad["rows"][5]["target"] in probs[0]
    bad2 = json.loads(json.dumps(lens))
    bad2["rows"][0]["hook_only_ms"]["per_row"] = 0.0
    assert any("hook_only per_row" in p for p in rep.check_lens_rows_against_doc(bad2, doc))
    bad3 = json.loads(json.dumps(lens))
    bad3["rows"][0]["source"] = "docs/lens_stacked_reductions_scoping.md §7.2.1 line 812"   # the lora_all row: identity differs
    assert any("identity differs" in p for p in rep.check_lens_rows_against_doc(bad3, doc))
    bad4 = json.loads(json.dumps(lens))
    bad4["rows"][0]["B"] = 16
    assert rep.check_lens_rows_against_doc(bad4, doc)


def test_parse_doc_row_refuses_non_rows():
    with pytest.raises(rep.ReportRefused):
        rep.parse_doc_row("| model | case | B |")
    d = rep.parse_doc_row("| gemma4e4b | psr response | 32 | 218.1 / 242.6 | 59.9 | 31.1 | 0.25 | 0.14 | 1.93 | 217.1 / 34.4 / 9.0 | 3.82 |")
    assert d == {"target": "gemma4e4b", "case": "psr response", "B": 32, "legacy": {"382416": 218.1, "384162": 242.6},
                 "per_row": 59.9, "stacked": 31.1, "hook_only": {"legacy": 217.1, "per_row": 34.4, "stacked": 9.0}}


# --------------------------------------------------------------------------- #
# 6. the report over synthetic per-target JSONs                                  #
# --------------------------------------------------------------------------- #
def _native_record(target: str, per_B: dict, *, status: str = "complete", drop=(), sha: str = "deadbeef", node: str = "gpu-9",
                   cell: str = PSR_CELL):
    """A synthetic per-target record in the harness's shape — VALID under the one validator
    (``native_timing_72.validate_timing_record``): per B ``(none_ms, psr_ms)`` at running_mean == B,
    error-free windows with an engine tok/s, the §7.2 protocol declared, and launch facts the
    protocol check re-derives to []. The known-bad tests below perturb ONE fact of this record at a time."""
    psr_h = "si_constv2_mix_full__psr_ce_all" if cell == PSR_CELL else cell     # the constitution cell spells the same in both
    mode = "all" if cell == PSR_CELL else "response"
    rows = []
    for B, (none_ms, psr_ms) in per_B.items():
        for arm, ms in (("none", none_ms), ("psr", psr_ms)):
            if (arm, B) in drop:
                continue
            tok_s = 1000.0 * B / ms
            rows.append({"target": target, "arm": arm, "cell": None if arm == "none" else cell,
                         "harness_cell": None if arm == "none" else psr_h, "B": B,
                         "label": "native (measured)", "geometry": {"L": 4096, "S": 64}, "window_s": 45.0, "warmup_s": 10.0,
                         "max_tokens": 128, "exact_length": True, "prefix_caching": False, "batch_invariance": 0,
                         "ms_per_step": {"gate": ms, "nominal": ms, "counter": None}, "ms_per_step_primary": "gate",
                         "passthrough_floor_ms": None, "hook_only_ms": None, "tok_s": tok_s, "running_mean": float(B),
                         "requests_done": 40, "errors": 0, "error_sample": None, "measured_window_s": 45.2,
                         "sha": sha, "node": node})
    nt.fill_hook_only(rows)
    return {"schema": 1, "target": target, "view": "psrcert", "cell": cell, "status": status, "node": node, "sha": sha,
            "date": "2026-09-14T12:00:00+00:00", "rows": rows,
            "protocol": {**nt.PROTOCOL, "batches": [8, 32, 64]},
            "protocol_problems": [],
            "arms": {"none": {"cell": None, "harness_cell": "none"}, "psr": {"cell": cell, "harness_cell": psr_h},
                     "vector": nt.vector_arm_record(target, None)},
            "prompt": {"tokens": 799, "served_prompt_tokens": 803},
            "launch": {"prefix_caching": {"composed": False, "argv_off_token": True, "argv_on_token": False,
                                          "attested_enable_prefix_caching": [False]},
                       "batch_invariance": {"harness_env": None, "launch_env": None, "attested_env": [], "attested_active": [],
                                            "effective": 0},
                       "engine": {"engine": "lora_native", "enforce_eager": False, "cudagraph_mode": "FULL_AND_PIECEWISE",
                                  "residual_slot": {"armed": True, "cells": {psr_h: {"kind": "psr", "position_mode": mode}}},
                                  "output_diagnostics": False, "cells_skipped": [],
                                  "geometry": {"L": 4096, "S": 64, "attested_S": 64, "adjustments": None, "lowered": None}}}}


def test_the_one_validator_accepts_the_valid_record_and_names_every_perturbation():
    """The validator (producer, post-check, report) requires complete status, the declared
    §7.2 protocol, the launch facts re-derived clean, and error-free engine-counted windows at 0.5 B
    concurrency — every KNOWN-BAD below keeps every numeric row and changes ONE fact."""
    import copy
    good = _native_record("gemma4e4b", {8: (20.0, 24.0), 32: (30.0, 36.0), 64: (40.0, 50.0)})
    v = nt.validate_timing_record(good)
    assert v["valid"] is True and v["problems"] == [] and v["n_rows"] == 6 and v["arms"] == ["none", "psr"], v
    def bad(mutate, needle):
        r = copy.deepcopy(good); mutate(r); out = nt.validate_timing_record(r)
        assert out["valid"] is False and any(needle in q for q in out["problems"]), (needle, out["problems"])
        return out
    bad(lambda r: r["rows"][3].update(errors=17, error_sample="HTTP 500"), "17 request error(s)")
    bad(lambda r: r.update(status="partial"), "status 'partial'")
    bad(lambda r: r.update(refused="the standup refused"), "carries refused")
    bad(lambda r: r["protocol"].update(window_s=5.0), "declared protocol window_s=5.0")
    bad(lambda r: [x.update(window_s=5.0) for x in r["rows"]], "window_s=5.0 != 45.0")
    bad(lambda r: r["protocol"].update(L=2048), "declared protocol L=2048")
    bad(lambda r: r["launch"]["engine"]["geometry"].update(L=2048), "served geometry 2048 x 64")
    bad(lambda r: r["launch"]["prefix_caching"].update(composed=True), "composed prefix_caching is True")
    bad(lambda r: r["launch"]["batch_invariance"].update(effective=1), "effective batch invariance 1")
    bad(lambda r: r["launch"]["engine"]["residual_slot"].update(armed=False), "residual_slot.armed is False")
    bad(lambda r: r["launch"]["engine"].update(output_diagnostics=True), "output_diagnostics is True")
    bad(lambda r: r["rows"].pop(2), "rows missing")
    bad(lambda r: r["rows"][0].update(tok_s=None), "no engine tok/s")
    bad(lambda r: r["rows"][1].update(running_mean=3.0), "running_mean 3.0 < 0.5 x B")
    bad(lambda r: r["rows"][1].update(measured_window_s=5.0), "measured window 5.0s outside")
    bad(lambda r: r["rows"][5].update(sha="other"), "sha 'other' != the record's")
    bad(lambda r: r.update(protocol_problems=["composed prefix_caching is True"]), "protocol_problems recorded")
    bad(lambda r: r.pop("launch"), "no launch facts recorded")
    bad(lambda r: r.update(sha=None), "no tree sha")
    bad(lambda r: r["protocol"].update(batches=[8, 32]), "declared batches [8, 32]")


def test_the_report_refuses_to_grade_invalid_evidence(tmp_path):
    """The report recomputes the validator's verdict and states NON-EVALUABLE — no PASS, no FAIL —
    for a record with a single errored window, a partial status, or a 5-second window, while a valid
    record IS graded (the known-bad in each direction)."""
    import copy
    good = _native_record("gemma4e4b", {8: (20.0, 24.0), 32: (30.0, 36.0), 64: (40.0, 50.0)})
    lens = json.loads(nt_lens_rows_path().read_text()) if False else rep.load_lens_rows(rep.DEFAULT_LENS_ROWS)
    md, summary = rep.render_markdown({"gemma4e4b": good}, lens, targets=["gemma4e4b"])
    c3 = summary["gemma4e4b"]["criterion3"]
    assert summary["gemma4e4b"]["validation"]["valid"] is True and c3["evaluable"] is True
    assert isinstance(c3["ms_per_step_pass"], bool) and isinstance(c3["hook_only_pass"], bool)
    assert "Criterion 3 — native psr < lens stacked" in md and "NON-EVALUABLE" not in md
    for mutate, needle in ((lambda r: r["rows"][3].update(errors=17), "17 request error(s)"),
                           (lambda r: r["rows"][3].pop("errors"), "not an integer count"),          # absent != zero
                           # the timed object is ONE name — a re-labelled top-level cell (a T5 record
                           # claiming the lens rows' object over rows measured on another) is INVALID
                           (lambda r: r.update(cell="si_constitution__psr_kl_all"), "!= arms.psr.cell"),
                           (lambda r: r["arms"]["psr"].update(cell="other__psr"), "!= arms.psr.cell"),
                           (lambda r: r["rows"][3].update(cell="other__psr"), "row identity"),
                           (lambda r: r["rows"][3].update(harness_cell="si_other__psr"), "row identity"),
                           # cell + arms.psr.cell + every psr row re-labelled TOGETHER, the harness id untouched:
                           # the static translation still refuses the lie
                           (lambda r: (r.update(cell="safety_refusal__psr_kl_all"), r["arms"]["psr"].update(cell="safety_refusal__psr_kl_all"),
                                       [row.update(cell="safety_refusal__psr_kl_all") for row in r["rows"] if row["arm"] == "psr"]),
                            "translates to harness id"),
                           (lambda r: r["rows"][3].update(errors=None), "not an integer count"),
                           (lambda r: r.update(status="partial"), "status 'partial'"),
                           (lambda r: [x.update(window_s=5.0) for x in r["rows"]], "window_s=5.0"),
                           (lambda r: r["launch"]["engine"]["geometry"].update(L=2048, S=32), "served geometry")):
        r = copy.deepcopy(good); mutate(r)
        md_bad, s_bad = rep.render_markdown({"gemma4e4b": r}, lens, targets=["gemma4e4b"])
        c3b = s_bad["gemma4e4b"]["criterion3"]
        assert c3b["evaluable"] is False and c3b["ms_per_step_pass"] is None and c3b["hook_only_pass"] is None
        assert "NON-EVALUABLE" in md_bad and "**PASS**" not in md_bad and "**FAIL**" not in md_bad
        assert any(needle in q for q in c3b["invalid_evidence"]), (needle, c3b["invalid_evidence"])
        # an invalid record gets NEITHER a caveat nor a table — its section is the heading +
        # NON-EVALUABLE + the validator's reasons only (a re-labelled record must never read "cross-run only")
        sec = md_bad[md_bad.index("## gemma4e4b"):]                  # the record's section (the preamble names both phrases)
        assert "| 8 |" not in sec and "**Caveat:**" not in sec and "cross-run only" not in sec
        assert s_bad["gemma4e4b"]["caveat"] is None and needle in sec


def test_known_bad_with_wandb_an_invalid_record_publishes_only_non_evaluable_and_reasons(tmp_path, monkeypatch):
    """With --wandb the report used to upload EVERY summary table, the invalid
    (re-labelled) record's computed comparison included. Through a stubbed wandb: the invalid record publishes
    only `<key>/non_evaluable` + its reasons — no table, no image — while the valid sibling publishes both; a
    record's target alias is not logged twice."""
    import sys, types
    calls = {"log": [], "summary": {}}

    class _Run:
        url = "https://wandb.invalid/run"
        summary = calls["summary"]

        def log(self, d):
            calls["log"].append(dict(d))

        def finish(self):
            calls["finished"] = True

    fake = types.SimpleNamespace(init=lambda **kw: _Run(), Table=lambda columns, data: ("table", tuple(columns), len(data)),
                                 Image=lambda p: ("image", str(p)))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    tdir, figs, md = tmp_path / "timing", tmp_path / "figs", tmp_path / "docs" / "native_psr_timing.md"
    tdir.mkdir()
    bad = _native_record("gemma4e4b", {8: (12.0, 13.9), 32: (14.0, 15.55), 64: (16.0, 17.4)})
    bad["cell"] = rep.LENS_ROWS_OBJECT_CELL
    (tdir / "relabelled.json").write_text(json.dumps(bad))
    (tdir / "qwen.json").write_text(json.dumps(_native_record("qwen35_4b", {8: (10.0, 21.0), 32: (12.0, 20.0), 64: (14.0, 30.0)})))
    rc = rep.main(["--timing-dir", str(tdir), "--out-md", str(md), "--figs", str(figs), "--lens-rows", str(LENS_JSON), "--doc", str(DOC),
                   "--targets", "gemma4e4b,qwen35_4b", "--wandb"])
    assert rc == 0 and calls.get("finished") is True
    logged = {k for d in calls["log"] for k in d}
    bad_key, good_key = rep.record_key("gemma4e4b", rep.LENS_ROWS_OBJECT_CELL), rep.record_key("qwen35_4b", PSR_CELL)
    assert f"{bad_key}/timing_table" not in logged and f"{bad_key}/ms_per_step" not in logged
    assert calls["summary"][f"{bad_key}/non_evaluable"] is True
    assert any("!= arms.psr.cell" in p for p in calls["summary"][f"{bad_key}/invalid_evidence"])
    assert calls["summary"][f"{bad_key}/criterion3_ms_per_step_pass"] is None
    assert f"{good_key}/timing_table" in logged and f"{good_key}/ms_per_step" in logged
    assert isinstance(calls["summary"][f"{good_key}/criterion3_ms_per_step_pass"], bool)
    assert not any(k.startswith("gemma4e4b/") or k.startswith("qwen35_4b/") for k in logged | set(calls["summary"]))


def test_known_bad_a_relabelled_record_gets_no_caveat_and_no_figure_only_non_evaluable(tmp_path):
    """A T5-shaped record whose top-level `cell` alone is re-labelled to the lens rows'
    object is INVALID (the one-cell binding) — and the report must give it NEITHER the cell-derived
    "cross-run only" caveat NOR a comparison figure beside the invalid-evidence warning: the section states
    NON-EVALUABLE and the validator's reasons only; a valid sibling record in the same run is rendered in full."""
    tdir, figs, md = tmp_path / "timing", tmp_path / "figs", tmp_path / "docs" / "native_psr_timing.md"
    tdir.mkdir()
    bad = _native_record("gemma4e4b", {8: (12.0, 13.9), 32: (14.0, 15.55), 64: (16.0, 17.4)})
    bad["cell"] = rep.LENS_ROWS_OBJECT_CELL                      # arms.psr / rows / attestation still name PSR_CELL
    (tdir / "relabelled.json").write_text(json.dumps(bad))
    (tdir / "qwen.json").write_text(json.dumps(_native_record("qwen35_4b", {8: (10.0, 21.0), 32: (12.0, 20.0), 64: (14.0, 30.0)})))
    rc = rep.main(["--timing-dir", str(tdir), "--out-md", str(md), "--figs", str(figs), "--lens-rows", str(LENS_JSON), "--doc", str(DOC),
                   "--targets", "gemma4e4b,qwen35_4b"])
    assert rc == 0
    text = md.read_text(encoding="utf-8")
    g = text.index(f"## gemma4e4b — cell `{rep.LENS_ROWS_OBJECT_CELL}`")
    sec = text[g:text.index("## qwen35_4b")]
    assert "Criterion 3 — **NON-EVALUABLE**" in sec and "!= arms.psr.cell" in sec
    assert "**Caveat:**" not in sec and "cross-run only" not in sec and "cross-object" not in sec
    assert "| 8 |" not in sec and "![" not in sec and "**PASS**" not in sec and "**FAIL**" not in sec
    assert not (figs / rep.figure_name("gemma4e4b", rep.LENS_ROWS_OBJECT_CELL)).exists()
    assert not any(p.name.startswith("gemma4e4b") for p in figs.iterdir())
    # the valid sibling is rendered in full: caveat, table, verdict, figure
    q = text[text.index("## qwen35_4b"):]
    assert "**Caveat:**" in q and "| 8 |" in q and "![" in q and (figs / rep.figure_name("qwen35_4b", PSR_CELL)).exists()


def test_report_over_synthetic_jsons_writes_tables_figures_and_verdicts(tmp_path):
    tdir, figs, md = tmp_path / "timing", tmp_path / "figs", tmp_path / "docs" / "native_psr_timing.md"
    tdir.mkdir()
    # gemma: faster than the lens stacked row at every B (17.2/22.3/34.7-style lens numbers are gemma's 27.8/31.1/34.8)
    (tdir / "gemma4e4b.json").write_text(json.dumps(_native_record("gemma4e4b", {8: (12.0, 13.9), 32: (14.0, 15.55), 64: (16.0, 17.4)})))
    # llama: slower than lens stacked at B=64 and a missing psr@64 window (partial) -> not evaluable
    (tdir / "llama8b.json").write_text(json.dumps(_native_record("llama8b", {8: (10.0, 12.0), 32: (12.0, 14.0), 64: (30.0, 40.0)},
                                                                 status="partial", drop={("psr", 64)})))
    # qwen: complete but FAILS the hook-only inequality at B=8 (native hook-only 11.0 > lens stacked 10.7)
    (tdir / "qwen35_4b.json").write_text(json.dumps(_native_record("qwen35_4b", {8: (10.0, 21.0), 32: (12.0, 20.0), 64: (14.0, 30.0)})))
    # T5b: a SECOND gemma record on the lens rows' own object — keyed by (target, cell), never by file name
    (tdir / "gemma4e4b_constitution.json").write_text(json.dumps(_native_record(
        "gemma4e4b", {8: (12.0, 14.5), 32: (14.0, 16.0), 64: (16.0, 18.0)}, cell=rep.LENS_ROWS_OBJECT_CELL)))
    repo_doc = REPO / "docs" / "native_psr_timing.md"                 # the report's former in-repo path; absent from the tree, and the report must never create it
    before = repo_doc.read_bytes() if repo_doc.exists() else None
    rc = rep.main(["--timing-dir", str(tdir), "--out-md", str(md), "--figs", str(figs), "--lens-rows", str(LENS_JSON), "--doc", str(DOC)])
    assert rc == 0 and md.exists()
    text = md.read_text(encoding="utf-8")
    # the test never writes the repo doc: the committed report's bytes are exactly what they were
    assert (repo_doc.read_bytes() if repo_doc.exists() else None) == before
    for t in ("llama8b", "gemma4e4b", "qwen35_4b"):
        assert f"## {t} — cell `{PSR_CELL}`" in text
    for t in ("gemma4e4b", "qwen35_4b"):
        assert (figs / rep.figure_name(t, PSR_CELL)).exists()
    assert not (figs / rep.figure_name("llama8b", PSR_CELL)).exists()      # INVALID (partial) evidence: no figure
    assert "## nemotron3nano4b\n" in text and (figs / "nemotron3nano4b_ms_per_step.png").exists()   # no record: the lens rows alone
    # BOTH gemma records: their own sections, figures and caveats — the constitution one is the like-for-like row
    assert f"## gemma4e4b — cell `{rep.LENS_ROWS_OBJECT_CELL}`" in text and (figs / rep.figure_name("gemma4e4b", rep.LENS_ROWS_OBJECT_CELL)).exists()
    c = text.index(f"## gemma4e4b — cell `{rep.LENS_ROWS_OBJECT_CELL}`")
    assert "cross-run only" in text[c:c + 1200] and "cross-object" not in text[c:c + 1200]
    g0 = text.index(f"## gemma4e4b — cell `{PSR_CELL}`")
    assert "cross-run + cross-object" in text[g0:g0 + 1200] and f"on `{PSR_CELL}`" in text[g0:g0 + 1200]
    assert "| B | native `none` floor ms/step | native psr ms/step | native hook-only ms | lens stacked ms/step (reused) | lens stacked hook-only ms (reused) | native psr / lens stacked |" in text
    # gemma's B=8 row: none 12.0, psr 13.9, hook-only 1.9, lens stacked 27.8, lens hook-only 6.8, ratio 0.50
    assert "| 8 | 12.0 | 13.9 | 1.9 | 27.8 | 6.8 | 0.50 |" in text
    assert "| 64 | 16.0 | 17.4 | 1.4 | 34.8 | 13.0 | 0.50 |" in text
    g = text.index(f"## gemma4e4b — cell `{PSR_CELL}`")
    assert "native psr < lens stacked at every B: **PASS**" in text[g:g + 3000]
    assert "native hook-only <= lens stacked hook-only at every B: **PASS**" in text[g:g + 3000]
    q = text.index("## qwen35_4b")
    assert "native psr < lens stacked at every B: **PASS**" in text[q:q + 3000]
    assert "native hook-only <= lens stacked hook-only at every B: **FAIL**" in text[q:q + 3000]
    l8 = text.index("## llama8b")
    l8_end = text.index("\n## ", l8 + 1)                            # the llama section alone (gemma's follows it)
    # a PARTIAL record with a missing row is INVALID evidence: the section is the heading + NON-EVALUABLE + the
    # validator's reasons ONLY — no caveat, no table, no verdict, no figure
    sec = text[l8:l8_end]
    assert "Criterion 3 — **NON-EVALUABLE**" in sec and "rows missing" in sec
    assert "| 64 | 30.0 |" not in sec and "**Caveat:**" not in sec and "![" not in sec and "**PASS**" not in sec and "**FAIL**" not in sec
    n = text.index("## nemotron3nano4b")
    assert "No native record" in text[n:n + 600] and "| 8 | — | — | — | 18.5 | 5.5 | — |" in text[n:n + 2000]
    # both sides labelled; the reused rows cited to their doc lines; the absent vector arm named; the caveats printed
    assert "lens (reused, §7.2, 2026-09-04)" in text and "native (measured)" in text
    assert "docs/lens_stacked_reductions_scoping.md §7.2.1 line 821" in text
    assert "no all-positions vector-kind object is staged on gemma4e4b (D4)" in text
    assert "cross-object" in text and "cross-run" in text and "never re-run" in text
    # the figure is a real PNG
    assert (figs / rep.figure_name("gemma4e4b", PSR_CELL)).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # the loader keys by (target, cell) and refuses a duplicate of the SAME pair; a file name is not a key
    recs = rep.load_records(sorted(tdir.glob("*.json")))
    assert set(recs) >= {("gemma4e4b", PSR_CELL), ("gemma4e4b", rep.LENS_ROWS_OBJECT_CELL), ("qwen35_4b", PSR_CELL)}
    (tdir / "dup.json").write_text((tdir / "qwen35_4b.json").read_text())
    with pytest.raises(rep.ReportRefused, match="two records for qwen35_4b"):
        rep.load_records(sorted(tdir.glob("*.json")))
    # the record's caveat is DERIVED: the same cell as the lens rows -> cross-run only; another -> cross-object too
    assert "cross-run only" in rep.record_caveat({"cell": rep.LENS_ROWS_OBJECT_CELL})
    assert "cross-object" in rep.record_caveat({"cell": PSR_CELL})
    # a lens rows file that does not name its object refuses (the caveat is never assumed)
    bad = tmp_path / "lens_bad.json"
    doc_ = json.loads(LENS_JSON.read_text()); doc_["what"] = "rows"; doc_["caveats"] = ["cross-run"]
    bad.write_text(json.dumps(doc_))
    with pytest.raises(rep.ReportRefused, match="does not name its object"):
        rep.load_lens_rows(bad)


def test_report_table_and_criterion3_pure():
    lens = json.loads(LENS_JSON.read_text(encoding="utf-8"))
    by = rep.lens_rows_by_target(lens)["gemma4e4b"]
    rec = _native_record("gemma4e4b", {8: (12.0, 13.9), 32: (14.0, 15.55), 64: (16.0, 17.4)})
    table = rep.build_table(rec, by)
    assert [r["B"] for r in table] == [8, 32, 64]
    assert table[0]["native_none_ms"] == 12.0 and table[0]["native_psr_ms"] == 13.9
    assert table[0]["native_hook_only_ms"] == pytest.approx(1.9) and table[0]["lens_stacked_ms"] == 27.8
    assert table[0]["lens_hook_only_ms"] == 6.8 and table[0]["ratio_native_psr_over_lens_stacked"] == pytest.approx(13.9 / 27.8)
    assert table[0]["lens_label"] == "lens (reused, §7.2, 2026-09-04)" and table[0]["native_label"] == "native (measured)"
    c3 = rep.criterion3(table)
    assert c3 == {"complete": True, "evaluable": True, "invalid_evidence": [], "ms_per_step_pass": True, "hook_only_pass": True,
                  "per_B": [{"B": 8, "ms_per_step_pass": True, "hook_only_pass": True}, {"B": 32, "ms_per_step_pass": True, "hook_only_pass": True},
                            {"B": 64, "ms_per_step_pass": True, "hook_only_pass": True}]}
    # a missing native record: the lens side alone, no verdict
    empty = rep.build_table(None, by)
    assert all(r["native_psr_ms"] is None and r["lens_stacked_ms"] is not None for r in empty)
    assert rep.criterion3(empty)["complete"] is False and rep.criterion3(empty)["ms_per_step_pass"] is None
    # a slower native psr at ONE B fails the ms/step inequality (every B is required)
    slow = rep.build_table(_native_record("gemma4e4b", {8: (12.0, 13.9), 32: (14.0, 15.55), 64: (16.0, 40.0)}), by)
    assert rep.criterion3(slow)["ms_per_step_pass"] is False and rep.criterion3(slow)["hook_only_pass"] is False


def test_report_refuses_drifted_lens_rows(tmp_path):
    lens = json.loads(LENS_JSON.read_text(encoding="utf-8"))
    lens["rows"][3]["per_row_ms_per_step"] = 1.0
    p = tmp_path / "lens.json"
    p.write_text(json.dumps(lens))
    with pytest.raises(rep.ReportRefused):
        rep.main(["--lens-rows", str(p), "--doc", str(DOC), "--out-md", str(tmp_path / "x.md")])
    assert not (tmp_path / "x.md").exists()


def test_render_figure_titles_are_two_words_at_most(tmp_path):
    lens = json.loads(LENS_JSON.read_text(encoding="utf-8"))
    table = rep.build_table(_native_record("llama8b", {8: (10.0, 12.0), 32: (12.0, 14.0), 64: (14.0, 20.0)}),
                            rep.lens_rows_by_target(lens)["llama8b"])
    out = rep.render_figure("llama8b", table, tmp_path / "f.png")
    assert out.exists() and out.stat().st_size > 1000
    src = inspect.getsource(rep.render_figure)
    for title in re.findall(r'set_title\("([^"]+)"', src) + [f"llama8b ms/step"]:
        assert len(title.split()) <= 2, title
    assert 'fig.suptitle(f"{target} ms/step"' in src
    # fixed-order series colors from the reference palette; the caveat is a footnote, not a title
    assert list(rep.SERIES_COLORS) == ["native none floor", "native psr", "lens stacked (reused)"]
    assert rep.SERIES_COLORS["native none floor"] == "#2a78d6" and rep.SERIES_COLORS["native psr"] == "#eb6834"
    assert "fig.text(0.02, 0.005, foot" in src


# --------------------------------------------------------------------------- #
# 7. main() end to end against fakes (the wiring a GPU job would otherwise find) #
# --------------------------------------------------------------------------- #
class _FakeTok:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize, return_dict):
        return list(range(6 + 31 * messages[0]["content"].count(cl.GATE_FILLER)))


def _wire_fakes(monkeypatch, tmp_path, *, attested_prefix_caching=False):
    """A fake standup behind lora_serving_up.bring_up_lora: it calls the (bound) composer once, writes
    endpoint.json + serving_attestation.json under its out dir, and exposes the attributes main() reads."""
    import lora_serving_up as lsu
    from sisteer.capeval.petri import registry_adapter as ra
    from sisteer.capeval.serving import run_identity, vllm_server as vs

    state = {"bring_up_kw": None, "torn_down": 0, "composer_kw": None, "windows": [], "probe": None}

    def stub_build_server_launch(spec, **kw):
        state["composer_kw"] = dict(kw)
        return types.SimpleNamespace(cmd=("vllm", "serve", "m", "--no-enforce-eager",
                                          "--no-enable-prefix-caching" if kw.get("prefix_caching") is False else "--enable-prefix-caching"),
                                     env={}, prefix_caching=kw.get("prefix_caching"), base_url="http://127.0.0.1:9", api_key="k",
                                     auth_headers={"Authorization": "Bearer k"})
    monkeypatch.setattr(vs, "build_server_launch", stub_build_server_launch)

    def fake_bring_up_lora(**kw):
        state["bring_up_kw"] = dict(kw)
        launch = vs.build_server_launch("spec", lora_native=True, max_model_len=kw["max_model_len"], max_num_seqs=kw["max_num_seqs"])
        out = Path(kw["out"]); out.mkdir(parents=True, exist_ok=True)
        _l, endpoint, att = _good_fixture()
        att["engine"]["enable_prefix_caching"] = attested_prefix_caching
        endpoint["proxy_url"] = "http://127.0.0.1:8/v1"; endpoint["proxy_token"] = None
        (out / "endpoint.json").write_text(json.dumps(endpoint)); (out / "serving_attestation.json").write_text(json.dumps(att))

        class Standup:
            endpoint_path, attestation_path = out / "endpoint.json", out / "serving_attestation.json"
            proxy_url, tokenizer = "http://127.0.0.1:8/v1", _FakeTok()
            base_vmodel, zero_vmodel = "m", "m--zero"
            vmodels = {"none": "m", PSR_H: "m--psr", "none__lora0": "m--zero"}

            def tear_down(self):
                state["torn_down"] += 1
        Standup.launch = launch
        return Standup()
    monkeypatch.setattr(lsu, "bring_up_lora", fake_bring_up_lora)
    monkeypatch.setattr(nt.sa, "read_serving_attestation", lambda p: json.loads(Path(p).read_text()))
    monkeypatch.setattr(ra, "harness_cell_id", lambda tt: PSR_H if tt == PSR_CELL else tt)
    monkeypatch.setattr(run_identity, "serving_versions", lambda *a, **k: {"vllm": "0.25.1", "torch": "2.x"})
    monkeypatch.setattr(run_identity, "tree_sha", lambda repo: "cafe0123456")

    def fake_probe(proxy_url, api_key, vmodel, prompt, *, max_tokens, timeout_s=600.0):
        state["probe"] = {"vmodel": vmodel, "prompt_head": prompt[:11], "max_tokens": max_tokens}
        return {"prompt_tokens": 803, "completion_tokens": max_tokens}
    monkeypatch.setattr(nt, "exact_length_probe", fake_probe)
    monkeypatch.setattr(cl, "read_metrics", _fake_metrics_reader(steps_per_s=40.0))

    async def fake_loop(proxy_url, api_key, vmodel, *, metrics_url, metrics_headers, B, max_tokens, warmup_s, window_s,
                        exact_length, sample_s=2.0):
        assert exact_length and cl.FILLER_PROMPT.startswith("Passage 5. ") and metrics_headers == {"Authorization": "Bearer k"}
        state["windows"].append((vmodel, B, max_tokens, warmup_s, window_s))
        t = time.monotonic()
        await asyncio.sleep(0.05)
        tok_s = {"m": 1000.0, "m--psr": 800.0}[vmodel] * B / 8
        return (_stats(float(B), tok_s), [{"t": t, "running": B}, {"t": time.monotonic(), "running": B}])
    monkeypatch.setattr(cl, "closed_loop_via_proxy", fake_loop)
    return state


def _fake_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "configs" / "capeval" / "manifests").mkdir(parents=True)
    (repo / "configs" / "sdfbase_full22_gemma4e4b_psrcert.yaml").write_text("explicit_cells: [none]\n")
    (repo / "configs" / "capeval" / "manifests" / "petri_full22_gemma4e4b_psrcert.yaml").write_text("x: 1\n")
    return repo


def test_main_end_to_end_against_fakes(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    state = _wire_fakes(monkeypatch, tmp_path)
    repo, out = _fake_repo(tmp_path), tmp_path / "out"
    rc = nt.main(["--target", "gemma4e4b", "--view", "psrcert", "--out", str(out), "--repo-dir", str(repo),
                  "--defenses-dir", str(tmp_path), "--expect-object-sha256", f"{PSR_CELL}=ab12", "--window-s", "0.1", "--warmup-s", "0.01"])
    # a run OFF the §7.2 protocol (a 0.1 s window here) completes but its evidence is INVALID —
    # the producer stamps it and exits RC_INVALID_EVIDENCE; every numeric row is kept for diagnosis
    assert rc == nt.RC_INVALID_EVIDENCE and state["torn_down"] == 1
    rec_short = json.loads((out / "gemma4e4b.json").read_text())
    assert rec_short["status"] == "complete" and rec_short["validation"]["valid"] is False
    assert any("window_s=0.1 != 45.0" in p for p in rec_short["validation"]["problems"]) and len(rec_short["rows"]) == 6
    # the SAME fakes under the protocol's window and warm-up: valid, rc 0
    state = _wire_fakes(monkeypatch, tmp_path)
    out = tmp_path / "out_protocol"
    rc = nt.main(["--target", "gemma4e4b", "--view", "psrcert", "--out", str(out), "--repo-dir", str(repo),
                  "--defenses-dir", str(tmp_path), "--expect-object-sha256", f"{PSR_CELL}=ab12", "--window-s", "45", "--warmup-s", "10"])
    assert rc == 0 and state["torn_down"] == 1
    kw = state["bring_up_kw"]
    assert kw["cell"] == [PSR_CELL] and kw["max_model_len"] == 4096 and kw["max_num_seqs"] == 64
    assert kw["geometry_override_reason"] == nt.GEOMETRY_OVERRIDE_REASON and kw["expect_object_sha256"] == [f"{PSR_CELL}=ab12"]
    assert kw["config"].endswith("sdfbase_full22_gemma4e4b_psrcert.yaml") and kw["out"] == str(out / "standup")
    assert state["composer_kw"]["prefix_caching"] is False            # the composer was BOUND off
    assert state["probe"] == {"vmodel": "m", "prompt_head": "Passage 5. ", "max_tokens": 128}
    assert state["windows"] == [(vm, B, 128, 10.0, 45.0) for B in (8, 32, 64) for vm in ("m", "m--psr")]
    rec = json.loads((out / "gemma4e4b.json").read_text())
    assert rec["status"] == "complete" and rec["target"] == "gemma4e4b" and rec["sha"] == "cafe0123456"
    assert rec["validation"]["valid"] is True and rec["validation"]["problems"] == []
    assert rec["protocol_problems"] == [] and rec["launch"]["prefix_caching"]["composed"] is False
    assert rec["launch"]["prefix_caching"]["composer_calls"] == [{"prefix_caching": False, "exact_replay_strategy": None,
                                                                   "max_model_len": 4096, "max_num_seqs": 64, "lora_native": True}]
    assert rec["launch"]["batch_invariance"]["effective"] == 0 and rec["launch"]["versions"] == {"vllm": "0.25.1", "torch": "2.x"}
    assert rec["arms"]["vector"] == {"status": "absent", "reason": "no all-positions vector-kind object is staged on gemma4e4b (D4)"}
    assert rec["arms"]["psr"] == {"cell": PSR_CELL, "vmodel": "m--psr", "harness_cell": PSR_H} and rec["zero_vmodel"] == "m--zero"
    assert rec["prompt"]["tokens"] <= 800 and rec["prompt"]["served_prompt_tokens"] == 803 and "text" not in rec["prompt"]
    assert (out / "timing_prompt.txt").read_text().startswith("Passage 5. ")
    rows = rec["rows"]
    assert [(r["arm"], r["B"]) for r in rows] == [(a, B) for B in (8, 32, 64) for a in ("none", "psr")]
    for r in rows:
        assert r["ms_per_step"]["gate"] is not None and r["ms_per_step"]["nominal"] is not None and r["label"] == "native (measured)"
        assert r["prefix_caching"] is False and r["batch_invariance"] == 0 and r["geometry"] == {"L": 4096, "S": 64}
        assert r["engine"]["residual_slot"]["armed"] is True and r["node"] and r["date"]
    psr8 = next(r for r in rows if r["arm"] == "psr" and r["B"] == 8)
    assert psr8["ms_per_step"]["gate"] == pytest.approx(10.0) and psr8["passthrough_floor_ms"]["gate"] == pytest.approx(8.0)
    assert psr8["hook_only_ms"]["gate"] == pytest.approx(2.0)
    assert sorted(p.name for p in (out / "windows").iterdir()) == sorted(f"gauges_{a}_B{B}.json" for a in ("none", "psr") for B in (8, 32, 64))
    assert cl.FILLER_PROMPT == mc.FILLER_PROMPT                       # the prompt binding was released


def test_main_refuses_a_standup_that_attests_prefix_caching_on_and_still_tears_down(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    state = _wire_fakes(monkeypatch, tmp_path, attested_prefix_caching=True)
    repo, out = _fake_repo(tmp_path), tmp_path / "out"
    rc = nt.main(["--target", "gemma4e4b", "--view", "psrcert", "--out", str(out), "--repo-dir", str(repo),
                  "--defenses-dir", str(tmp_path), "--expect-object-sha256", f"{PSR_CELL}=ab12"])
    assert rc == 2 and state["torn_down"] == 1 and state["windows"] == []
    rec = json.loads((out / "gemma4e4b.json").read_text())
    assert rec["status"] == "refused" and any("enable_prefix_caching" in p for p in rec["protocol_problems"])
    assert "attested enable_prefix_caching [True]" in rec["refused"]


def test_main_refuses_a_missing_config(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    with pytest.raises(nt.TimingRefused):
        nt.main(["--target", "gemma4e4b", "--view", "nope", "--out", str(tmp_path / "o"), "--repo-dir", str(tmp_path),
                 "--defenses-dir", str(tmp_path)])


def test_prefix_caching_off_keeps_the_composers_signature_and_geometry_imports_under_it():
    """KNOWN-BAD of T1-T3 (jobs 509129-509131): `sisteer.capeval.serving.geometry` reads
    `inspect.signature(vs.build_server_launch).parameters["max_model_len"]` at import and the
    standup imports it lazily — a bare wrapper hid the signature (KeyError). Under the binding
    the composer's signature is the original's and the geometry module imports."""
    import importlib
    import inspect
    import sys
    from sisteer.capeval.serving import vllm_server as vs
    orig_sig = inspect.signature(vs.build_server_launch)
    with nt.prefix_caching_off(vs):
        assert inspect.signature(vs.build_server_launch) == orig_sig
        assert "max_model_len" in inspect.signature(vs.build_server_launch).parameters
        # a FRESH execution of the geometry module's source under a PRIVATE name — never a pop of the
        # canonical `sisteer.capeval.serving.geometry` from sys.modules: that re-import gave every later
        # importer a second module object (the Modal merge-tree suite on 20a6064e: 7 tests of
        # tests/test_served_geometry.py failed isinstance(ServedGeometry) / read the production budget
        # table instead of the patched one after this file was collected first)
        import importlib.util
        from sisteer.capeval.serving import geometry as canonical
        # (a dotted private name inside the real package, so the module's relative imports resolve; the
        # private module is never registered in sys.modules)
        spec = importlib.util.spec_from_file_location("sisteer.capeval.serving._geometry_under_the_composer_binding",
                                                      canonical.__file__)
        geom = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = geom                    # its dataclasses resolve their module through sys.modules
        try:
            spec.loader.exec_module(geom)
        finally:
            sys.modules.pop(spec.name, None)
        assert geom.PETRI_MAX_MODEL_LEN == orig_sig.parameters["max_model_len"].default
        assert sys.modules["sisteer.capeval.serving.geometry"] is canonical
    assert vs.build_server_launch.__wrapped__ if hasattr(vs.build_server_launch, "__wrapped__") else True
    assert inspect.signature(vs.build_server_launch) == orig_sig
