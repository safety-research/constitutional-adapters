"""Attacker-only prefix-cache and quantization identity tests."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from sisteer.eval.attacks import attacker_serving as serving
from sisteer.eval.attacks import vllm_attacker

REPO = Path(__file__).resolve().parents[1]


def test_checked_in_default_is_bf16_with_prefix_cache_on():
    options = serving.load_defaults()
    assert options == serving.AttackerServingOptions("none", True)
    assert options.vllm_argv() == [
        "--enable-prefix-caching", "--mamba-cache-mode", "align"
    ]
    assert options.as_dict()["kv_cache_dtype"] == "auto"
    assert options.as_dict()["enable_thinking"] is False


def test_thinking_mode_has_one_source_of_truth():
    assert vllm_attacker.ATTACKER_ENABLE_THINKING is serving.ATTACKER_ENABLE_THINKING
    assert serving.ATTACKER_ENABLE_THINKING is False


def test_fp8_is_opt_in_and_does_not_enable_fp8_kv_cache():
    options, sources = serving.resolve("fp8", "off")
    assert sources == {"quantization": "flag", "prefix_caching": "flag"}
    argv = options.vllm_argv()
    assert argv == ["--quantization", "fp8", "--no-enable-prefix-caching"]
    assert "--kv-cache-dtype" not in argv
    assert options.as_dict()["kv_cache_dtype"] == "auto"


def test_attacker_options_refuse_target_composition():
    options = serving.AttackerServingOptions("none", True)
    with pytest.raises(serving.AttackerServingError, match="steered target"):
        options.vllm_argv(model_role="target")


def test_frozen_target_standups_do_not_consume_attacker_serving_seam():
    # Composition backstop: the target launch path must never import this seam or
    # consume its defaults file. This deliberately reads source rather than
    # importing Modal/vLLM modules into the CPU test environment.
    target_files = [
        REPO / "src/sisteer/capeval/serving/vllm_server.py",
        REPO / "src/sisteer/capeval/serving/proxy.py",
        REPO / "scripts/capctrl_serving_up.py",
        REPO / "scripts/petri_steered_smoke.py",
        REPO / "scripts/sbatch/petri_steered_smoke.sbatch",
        REPO / "scripts/modal/petri_agentloop_control.py",
    ]
    for path in target_files:
        text = path.read_text()
        assert "attacker_serving" not in text, path
        assert "configs/attacker_serving.json" not in text, path


def test_combined_modal_target_standup_does_not_consume_attacker_options():
    # grokmt_served contains both roles in one file, so a whole-file import grep
    # would prevent the attacker from using the seam. Pin the target composition
    # functions instead.
    path = REPO / "scripts/modal/grokmt_served.py"
    text = path.read_text()
    tree = ast.parse(text)
    functions = {
        node.name: ast.get_source_segment(text, node) or ""
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in ("_geometry_argv", "_target_standup_cmd", "_start_target"):
        body = functions[name]
        assert "attacker_serving" not in body
        assert "configs/attacker_serving.json" not in body


def _engine_log(*, quantization="None", prefix=True, kv="auto") -> str:
    return (
        "INFO Initializing a V1 LLM engine (v0.25.1) with config: "
        f"model='m', dtype=torch.bfloat16, max_model_len=8192, "
        f"quantization={quantization}, kv_cache_dtype={kv}, "
        f"enable_prefix_caching={prefix}\n"
        "INFO Mamba cache mode is set to 'align'\n"
        "INFO GPU KV cache size: 123,456 tokens\n"
        "INFO Prefix cache hit rate: 77.5%\n"
    )


def test_engine_log_is_verified_before_manifest_is_written(tmp_path):
    engine = serving.parse_engine_log(_engine_log())
    options = serving.AttackerServingOptions("none", True)
    assert serving.verify_engine(options, engine) == []
    manifest = serving.write_manifest(
        tmp_path / "serve.json",
        opts=options,
        sources={"quantization": "defaults_file", "prefix_caching": "defaults_file"},
        engine=engine,
        model="m",
        served_name="m",
        endpoint="http://127.0.0.1:8000",
        log_path="serve.log",
        argv=options.vllm_argv(),
    )
    fields = serving.driver_fields(manifest, manifest_path="serve.json")
    assert fields == {
        "attacker_quantization": "none",
        "attacker_prefix_caching": True,
        "attacker_kv_cache_dtype": "auto",
        "attacker_mamba_cache_mode": "align",
        "attacker_enable_thinking": False,
        "attacker_vllm_version": "0.25.1",
        "attacker_serving_manifest": "serve.json",
        "attacker_serving_source": "manifest",
    }


def test_read_manifest_reverifies_recorded_identity(tmp_path):
    options = serving.AttackerServingOptions("none", True)
    path = tmp_path / "serve.json"
    manifest = serving.write_manifest(
        path, opts=options, sources={}, engine=serving.parse_engine_log(_engine_log()),
        model="m", served_name="m", endpoint=None, log_path=None,
        argv=options.vllm_argv(),
    )
    assert serving.read_manifest(path) == manifest

    manifest["engine"]["enable_prefix_caching"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(serving.AttackerServingError, match="no longer verifies"):
        serving.read_manifest(path)

    manifest["engine"]["enable_prefix_caching"] = True
    manifest["requested"]["mamba_cache_mode"] = "all"
    path.write_text(json.dumps(manifest))
    with pytest.raises(serving.AttackerServingError, match="mamba_cache_mode"):
        serving.read_manifest(path)

    manifest["requested"]["mamba_cache_mode"] = "align"
    manifest["requested"]["quantization"] = "fp8"
    path.write_text(json.dumps(manifest))
    with pytest.raises(serving.AttackerServingError, match="contradictory"):
        serving.read_manifest(path)


def test_manifest_refuses_unseen_or_contradictory_engine(tmp_path):
    options = serving.AttackerServingOptions("fp8", True)
    for engine in (
        serving.parse_engine_log(""),
        serving.parse_engine_log(_engine_log(quantization="None")),
        serving.parse_engine_log(_engine_log(quantization="fp8", prefix=False)),
        serving.parse_engine_log(_engine_log(quantization="fp8", kv="fp8")),
        serving.parse_engine_log(_engine_log(quantization="fp8").replace(
            ", kv_cache_dtype=auto", ""
        )),
        serving.parse_engine_log(_engine_log(quantization="fp8").replace(
            "INFO Mamba cache mode is set to 'align'\n", ""
        )),
    ):
        with pytest.raises(serving.AttackerServingError, match="could not be verified"):
            serving.write_manifest(
                tmp_path / "bad.json", opts=options, sources={}, engine=engine,
                model="m", served_name="m", endpoint=None, log_path=None,
                argv=options.vllm_argv(),
            )


def test_quantization_identity_refuses_mixed_run():
    bf16 = {"requested": {"attacker_quantization": "none"}}
    fp8 = {"requested": {"attacker_quantization": "fp8"}}
    assert serving.assert_uniform_quantization([bf16, bf16]) == "none"
    with pytest.raises(serving.AttackerServingError, match="refusing to mix"):
        serving.assert_uniform_quantization([bf16, fp8])


def test_cli_reads_the_one_defaults_file():
    command = [
        sys.executable,
        str(REPO / "src/sisteer/eval/attacks/attacker_serving.py"),
        "argv",
    ]
    result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == [
        "--enable-prefix-caching", "--mamba-cache-mode", "align"
    ]


def test_defaults_reject_unpublished_ablation_quantization(tmp_path):
    path = tmp_path / "defaults.json"
    path.write_text(json.dumps({"quantization": "fp8_per_channel", "prefix_caching": True}))
    with pytest.raises(serving.AttackerServingError, match="attacker_quantization"):
        serving.load_defaults(path)


def test_http_status_reports_the_answer_or_zero_when_nothing_listens(tmp_path, capsys):
    """The launcher's readiness probe (the `health` subcommand) is stdlib-only:
    a live server's status comes back as-is (200 for /health, 404 for a
    missing path); a closed port or a malformed URL is 0, printed as `000`
    the way the launcher compares it, with exit 0 either way. The probe
    exists because the Modal images ship no curl -- the previous curl loop
    read 000 against a healthy server (2026-09-06)."""
    import functools
    import http.server
    import socket
    import threading

    (tmp_path / "health").write_text("ok\n")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert serving.http_status(f"http://127.0.0.1:{port}/health") == 200
        assert serving.http_status(f"http://127.0.0.1:{port}/missing") == 404
        assert serving._cli(["health", "--endpoint", f"http://127.0.0.1:{port}/health"]) == 0
        assert capsys.readouterr().out.strip() == "200"
    finally:
        srv.shutdown()
        srv.server_close()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        closed = s.getsockname()[1]
    assert serving.http_status(f"http://127.0.0.1:{closed}/health", timeout=2) == 0
    assert serving.http_status("not a url") == 0
    # the script route the launcher uses: three digits, exit 0 on a refusal too
    r = subprocess.run(
        [sys.executable, str(REPO / "src/sisteer/eval/attacks/attacker_serving.py"),
         "health", "--endpoint", f"http://127.0.0.1:{closed}/health", "--timeout", "2"],
        text=True, capture_output=True)
    assert r.returncode == 0 and r.stdout.strip() == "000", r.stdout + r.stderr
