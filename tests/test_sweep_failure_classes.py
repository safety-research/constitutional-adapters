"""The fan-out's real launch failures (2026-09-03), pinned by their EXACT
production strings:

(a) nemotron3nano4b at 128 x 8192, engine init: ``ValueError: No available
    memory for the cache blocks. Try increasing gpu_memory_utilization ...`` —
    a genuine memory WALL. It was filed as ``launch_error`` (which aborts the
    walk); it must classify as ``oom``, and a RESUME must re-read that recorded
    rung as the wall and bisect 96 <-> 128.
(b) glm46vflash on a busy node: ``torch.AcceleratorError: CUDA error:
    CUDA-capable device(s) is/are busy or unavailable`` — INFRA, not geometry
    and not our code: the new ``gpu_unavailable`` status; the budget walk
    aborts with a distinct SystemExit ("GPU busy/unavailable on <node> —
    resubmit excluding it"), no wall recorded; a window standup with it skips
    the retry ladder and re-raises the same SystemExit.

Plus the per-family requirements selection: gemma-4 cannot serve under
transformers 5.15.0 (AmbiguousGlobalPerLayerAttributeError on 'head_dim'), so
the sbatch bootstraps a second venv from the -tf5141 variant for the two
gemma keys and stamps the versions it serves with.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import served_decode_sweep as sw  # noqa: E402
from test_sweep_window_retry import _budget_json, _ctx, sweep_env  # noqa: E402,F401  (fixture)

OOM_CACHE_BLOCKS = ("ValueError: No available memory for the cache blocks. Try increasing "
                    "gpu_memory_utilization when initializing the engine.")
GPU_BUSY = "torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable"
SBATCH = REPO / "scripts/sbatch/geom_sweep.sbatch"


# --------------------------------------------------------------------------- #
# classification                                                               #
# --------------------------------------------------------------------------- #


def test_no_available_memory_for_cache_blocks_is_an_oom_wall():
    # the message lives in the server log; the launcher's exception is generic
    assert sw.classify_launch_failure("vllm serve did not become ready within 1500s — see log",
                                      OOM_CACHE_BLOCKS) == "oom"
    assert sw.classify_launch_failure("vllm serve exited with 1", "INFO ...\n" + OOM_CACHE_BLOCKS) == "oom"
    assert sw.classify_launch_failure(OOM_CACHE_BLOCKS, "") == "oom"
    assert sw.is_wall({"product": 1_048_576, "status": "oom"})
    assert sw.scrape_server_log("x\n" + OOM_CACHE_BLOCKS + "\ny")["log_oom"] is True
    assert OOM_CACHE_BLOCKS in sw.scrape_server_log(OOM_CACHE_BLOCKS)["memory_lines"][0]


def test_busy_or_unavailable_gpu_is_infra_not_a_wall():
    assert sw.classify_launch_failure("vllm serve exited with 1 right after spawn", GPU_BUSY) \
        == sw.STATUS_GPU_UNAVAILABLE == "gpu_unavailable"
    assert sw.classify_launch_failure(GPU_BUSY, "") == "gpu_unavailable"
    # even when the log ALSO shows an OOM-looking line, a busy card is the cause
    assert sw.classify_launch_failure("did not become ready", GPU_BUSY + "\nout of memory") \
        == "gpu_unavailable"
    assert not sw.is_wall({"product": 393_216, "status": "gpu_unavailable"})
    assert "gpu_unavailable" not in sw.WALL_STATUSES
    # the other classes are untouched
    assert sw.classify_launch_failure("worker 0: KV cache holds 1 tokens < the 2-token budget") \
        == "kv_under_coverage"
    assert sw.classify_launch_failure("did not become ready within 900s", "") == "not_ready"
    assert sw.classify_launch_failure("AttributeError: 'dict' object has no attribute", "") \
        == "launch_error"


def test_resume_reclassifies_the_nemotron_rung_and_bisects_96_128(tmp_path):
    L = 8192
    probes = [{"product": S * L, "max_num_seqs": S, "max_model_len": L, "status": "pass",
               "tag": f"budget_L{L}xS{S}"} for S in (48, 64, 80, 96)]
    # as the old classifier filed it: launch_error, the message only in the scraped log lines
    bad = {"product": 128 * L, "max_num_seqs": 128, "max_model_len": L, "status": "launch_error",
           "tag": f"budget_L{L}xS128", "error": "VllmServerError: vllm serve exited with 1 — see log",
           "memory_lines": ["INFO Memory profiling takes 4.1 seconds", OOM_CACHE_BLOCKS]}
    # before: the launch_error is not a wall -> the walk would re-probe 128
    assert sw.next_budget_probe(probes + [bad], sw.BUDGET_LADDER, L) == 128 * L
    fixed = sw.reclassify_probe(bad, tmp_path)
    assert fixed["status"] == "oom" and fixed["reclassified_from"] == "launch_error"
    assert sw.next_budget_probe(probes + [fixed], sw.BUDGET_LADDER, L) == 112 * L   # bisect 96 <-> 128
    br = sw.budget_bracket(probes + [fixed])
    assert br["largest_passing"]["product"] == 96 * L and br["first_refused"] == 128 * L
    # the message can also be recovered from the previous job's server log on disk
    bad2 = {k: v for k, v in bad.items() if k != "memory_lines"}
    assert sw.reclassify_probe(bad2, tmp_path)["status"] == "launch_error"   # nothing to go on
    logdir = tmp_path / "standups" / "budget_L8192xS128__job65057"
    logdir.mkdir(parents=True)
    (logdir / "vllm_server.log").write_text("boot...\n" + OOM_CACHE_BLOCKS + "\n")
    assert sw.reclassify_probe(bad2, tmp_path)["status"] == "oom"
    # a passing probe is never touched
    assert sw.reclassify_probe(probes[0], tmp_path) == probes[0]
    # a recorded wall with NO error text / log to re-read keeps its verdict (a resume
    # once demoted such walls to launch_error and re-probed them)
    wall = {"product": 160 * L, "max_num_seqs": 160, "max_model_len": L, "status": "kv_under_coverage",
            "tag": f"budget_L{L}xS160"}                        # a tag with no log on disk
    assert sw.reclassify_probe(wall, tmp_path) == wall


# --------------------------------------------------------------------------- #
# the budget walk and the window standup on a busy GPU                         #
# --------------------------------------------------------------------------- #


def test_budget_walk_aborts_on_a_busy_gpu_without_a_wall(monkeypatch, tmp_path):
    ctx = _ctx(tmp_path, hybrid=False)

    def fake_probe(ctx_, *, max_model_len, max_num_seqs, expect_object=None):
        # E54 #5, as in test_review_provenance: run_budget forwards the phase
        # fingerprint's ``steered_object`` to probe_geometry, so the stub takes
        # it and pins that it is the fingerprint's own value rather than
        # swallowing whatever arrives (None here — passthrough-only context).
        assert expect_object == sw.object_identity(ctx_.objects_dir, ctx_.steered_cell)
        return {"product": max_num_seqs * max_model_len, "max_num_seqs": max_num_seqs,
                "max_model_len": max_model_len, "tag": f"budget_L{max_model_len}xS{max_num_seqs}",
                "status": sw.classify_launch_failure("vllm serve exited with 1", GPU_BUSY),
                "error": GPU_BUSY}

    monkeypatch.setattr(sw, "probe_geometry", fake_probe)
    monkeypatch.setattr(sw, "_nvidia_smi", lambda *a: "H200")
    with pytest.raises(SystemExit) as ei:
        sw.run_budget(ctx, budget_len=8192, ladder=sw.BUDGET_LADDER)
    msg = str(ei.value)
    assert "GPU busy/unavailable on TESTHOST — resubmit excluding it" in msg
    assert "budget_L8192xS48" in msg and "no wall recorded" in msg
    assert "PROBE-CODE error" not in msg                      # distinct from launch_error's
    doc = json.loads((ctx.results_dir / "budget.json").read_text())
    assert doc["aborted"] == "gpu_unavailable" and len(doc["probes"]) == 1
    assert doc["probes"][0]["status"] == "gpu_unavailable"
    assert doc.get("first_refused") is None and doc.get("largest_passing") is None
    assert doc["complete"] is False
    assert "transformers_version" in doc and "requirements_file" in doc


def test_window_standup_on_a_busy_gpu_skips_the_ladder_and_reraises(sweep_env, monkeypatch,
                                                                     tmp_path, capsys):
    ctx = _ctx(tmp_path, hybrid=True)
    _budget_json(ctx)

    def busy_bring_up(ctx_, *, max_model_len, max_num_seqs, tag, eval_key):
        sweep_env.calls.append((max_model_len, max_num_seqs, tag, eval_key))
        raise RuntimeError(GPU_BUSY)

    monkeypatch.setattr(sw, "_bring_up", busy_bring_up)   # over the fixture's fake
    with pytest.raises(SystemExit) as ei:
        sw.run_sweep(ctx, windows=(1536, 8704), budget=None, prompt_tokens=1000,
                     max_tokens=512, warmup_s=0.0, measure_s=1.0)
    assert "GPU busy/unavailable on TESTHOST — resubmit excluding it" in str(ei.value)
    assert [(L, S) for L, S, _, _ in sweep_env.calls] == [(1536, 256)]   # ONE attempt, no ladder
    rows = [json.loads(l) for l in (ctx.results_dir / "sweep.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["kind"] == "window_refused"
    assert rows[0]["status"] == "gpu_unavailable" and rows[0]["retried"] is False
    assert rows[0]["S_tried"] == 256 and GPU_BUSY in rows[0]["error"]
    out = capsys.readouterr().out
    assert ("WINDOW L=1536 (capability:short) standup at S=256: GPU busy/unavailable on TESTHOST "
            "— no retry ladder; aborting the phase for a resubmit elsewhere") in out
    assert not (ctx.results_dir / "sweep_summary.json").exists()


def test_version_stamps():
    v = sw.transformers_version()
    assert v is None or re.match(r"\d+\.\d+", v)
    import os
    old = os.environ.get("GEOM_REQS")
    os.environ["GEOM_REQS"] = "/r/configs/capeval/geom-sweep-requirements-tf5141.txt"
    try:
        assert sw.requirements_file() == "geom-sweep-requirements-tf5141.txt"
    finally:
        if old is None:
            del os.environ["GEOM_REQS"]
        else:
            os.environ["GEOM_REQS"] = old


# --------------------------------------------------------------------------- #
# the per-family requirements selection in the sbatch                          #
# --------------------------------------------------------------------------- #


def _select(model: str) -> str:
    txt = SBATCH.read_text()
    m = re.search(r'^case "\$MODEL" in\n.*?^esac$', txt, re.M | re.S)
    assert m, "no case \"$MODEL\" selection block in the sbatch"
    proc = subprocess.run(
        ["bash", "-c", f'MODEL={model}; REPO_DIR=/r\n{m.group(0)}\necho "$GEOM_REQS|$GEOM_TF_PIN"'],
        capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_sbatch_selects_the_tf5141_requirements_for_gemma_and_the_default_otherwise():
    for key in sw.MODELS:
        reqs, pin = _select(key).split("|")
        if key in ("gemma4e4b", "gemma4e2b"):
            assert reqs == "/r/configs/capeval/geom-sweep-requirements-tf5141.txt", key
            assert "5.14.1" in pin
        else:
            assert reqs == "/r/configs/capeval/geom-sweep-requirements.txt", key
            assert "unpinned" in pin
    txt = SBATCH.read_text()
    assert '  gemma4e4b|gemma4e2b) GEOM_REQS="$REPO_DIR/configs/capeval/geom-sweep-requirements-tf5141.txt"' in txt
    assert "export GEOM_REQS" in txt
    # the header line names the file and the pin; the post-bootstrap line the resolved versions
    assert 'reqs=$(basename "$GEOM_REQS") ($GEOM_TF_PIN)' in txt
    assert "m.version('transformers')" in txt and "m.version('vllm')" in txt
    assert 'echo "[geom-sweep] serving venv $UV_PROJECT_ENVIRONMENT from $(basename "$GEOM_REQS"): $VENV_VERSIONS"' in txt
    # the hash covers the SELECTED file, so the two venvs never collide
    assert 'cat "$VLLM_REQS" "$GEOM_REQS" | sha256sum' in txt
    assert subprocess.run(["bash", "-n", str(SBATCH)], capture_output=True).returncode == 0


def test_tf5141_variant_is_the_default_plus_the_exact_pin():
    default = REPO / "configs/capeval/geom-sweep-requirements.txt"
    variant = REPO / "configs/capeval/geom-sweep-requirements-tf5141.txt"
    assert default.is_file() and variant.is_file()

    def lines(p):
        return [l.strip() for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]

    assert lines(variant) == lines(default) + ["transformers==5.14.1"]
    assert "AmbiguousGlobalPerLayerAttributeError" in variant.read_text()
    assert not any(l.startswith("transformers") for l in lines(default))   # the default floats
