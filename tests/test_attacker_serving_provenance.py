"""The attacker's serving identity (bf16 vs opt-in FP8, prefix caching) is
recorded in ``attacker_engine.json`` and refused when mixed -- the way the
engine itself already is -- and it can only ever be composed for the attacker.

CPU-only: no vLLM, no model, no network. The launcher test runs
``scripts/vllm_attacker_serve.sh`` against a FAKE ``vllm`` binary that echoes
the engine's resolved-config line and serves ``/health`` with the stdlib
http.server, so the launcher's flag composition, manifest write, and
contradiction kill path run end to end.
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest

from sisteer.eval.attacks import attacker_serving as serving

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import jailbreak_run  # noqa: E402

ATTACKER = "huihui-ai/Huihui-Qwen3.6-27B-abliterated"


def _engine_log(*, quantization="None", prefix=True, kv="auto") -> str:
    # the exact shape of vllm 0.25.1's line (max_seq_len=, quantization=None)
    return (
        "INFO Initializing a V1 LLM engine (v0.25.1) with config: model='m', "
        f"dtype=torch.bfloat16, max_seq_len=49152, quantization={quantization}, "
        f"kv_cache_dtype={kv}, enable_prefix_caching={prefix}\n"
        + ("INFO Mamba cache mode is set to 'align'\n" if prefix else "")
        + "INFO GPU KV cache size: 735,142 tokens\nINFO Prefix cache hit rate: 76.5%\n"
    )


def _manifest(tmp_path, *, quantization="none", prefix=True, served=ATTACKER):
    opts = serving.AttackerServingOptions(quantization, prefix)
    engine = serving.parse_engine_log(
        _engine_log(quantization="None" if quantization == "none" else "fp8", prefix=prefix))
    path = tmp_path / f"{quantization}_{'on' if prefix else 'off'}.serving.json"
    serving.write_manifest(path, opts=opts, sources={"quantization": "flag",
                                                     "prefix_caching": "flag"},
                           engine=engine, model="/snap", served_name=served,
                           endpoint="http://127.0.0.1:8000", log_path="serve.log",
                           argv=opts.vllm_argv())
    return path


def _cfg():
    return {"attacker": {"model": ATTACKER, "snapshot": "27502c87"}, "dtype": "bfloat16"}


# --- the driver records the serving identity ---------------------------------


def test_max_seq_len_is_the_recorded_window():
    engine = serving.parse_engine_log(_engine_log())
    assert engine["max_model_len"] == 49152
    assert engine["quantization"] == "none" and engine["kv_cache_dtype"] == "auto"


def test_manifest_records_fp8_and_prefix_caching_from_the_serving_sidecar(tmp_path):
    sidecar = _manifest(tmp_path, quantization="fp8", prefix=True)
    args = SimpleNamespace(attacker_engine="vllm", attacker_endpoint="http://127.0.0.1:8000",
                           attacker_serving=str(sidecar))
    jailbreak_run._write_engine_manifest(tmp_path / "run", _cfg(), args, "pair")
    m = json.loads((tmp_path / "run" / "attacker_engine.json").read_text())
    assert m["attacker_engine"] == "vllm"
    assert m["attacker_quantization"] == "fp8"
    assert m["attacker_prefix_caching"] is True
    assert m["attacker_kv_cache_dtype"] == "auto"          # never an FP8 KV cache
    assert m["attacker_mamba_cache_mode"] == "align"
    assert m["attacker_enable_thinking"] is False
    assert m["attacker_vllm_version"] == "0.25.1"
    assert m["attacker_serving_manifest"] == str(sidecar)


def test_manifest_without_sidecar_records_unrecorded_not_a_guess(tmp_path):
    args = SimpleNamespace(attacker_engine="vllm", attacker_endpoint="http://127.0.0.1:8000")
    jailbreak_run._write_engine_manifest(tmp_path / "a", _cfg(), args, "pair")
    m = json.loads((tmp_path / "a" / "attacker_engine.json").read_text())
    assert m["attacker_quantization"] == "unrecorded"
    assert m["attacker_prefix_caching"] is None and m["attacker_serving_manifest"] is None
    # the in-process HF attacker is unquantized; a hosted one is not ours to record
    jailbreak_run._write_engine_manifest(tmp_path / "b", _cfg(),
                                         SimpleNamespace(attacker_engine="hf"), "pair")
    assert json.loads((tmp_path / "b" / "attacker_engine.json").read_text())[
        "attacker_quantization"] == "none"
    jailbreak_run._write_engine_manifest(
        tmp_path / "c", _cfg(),
        SimpleNamespace(attacker_engine="openrouter", attacker_endpoint="https://x/api",
                        attacker_model="x-ai/grok-4.6"), "pair")
    assert json.loads((tmp_path / "c" / "attacker_engine.json").read_text())[
        "attacker_quantization"] == "hosted"


def test_manifest_refuses_a_sidecar_for_another_server_or_engine(tmp_path):
    other = _manifest(tmp_path, served="some/other-model")
    args = SimpleNamespace(attacker_engine="vllm", attacker_endpoint="http://127.0.0.1:8000",
                           attacker_serving=str(other))
    with pytest.raises(SystemExit, match="another server"):
        jailbreak_run._write_engine_manifest(tmp_path / "run", _cfg(), args, "pair")
    with pytest.raises(SystemExit, match="meaningless"):
        jailbreak_run._write_engine_manifest(
            tmp_path / "run2", _cfg(),
            SimpleNamespace(attacker_engine="hf", attacker_serving=str(other)), "pair")
    # a tampered sidecar (engine no longer verifies) is refused, not attached
    tampered = tmp_path / "tampered.json"
    doc = json.loads(_manifest(tmp_path).read_text())
    doc["engine"]["enable_prefix_caching"] = False
    tampered.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="no longer verifies"):
        jailbreak_run._write_engine_manifest(
            tmp_path / "run3", _cfg(),
            SimpleNamespace(attacker_engine="vllm", attacker_endpoint="http://127.0.0.1:8000",
                            attacker_serving=str(tampered)), "pair")


def test_generate_parser_accepts_attacker_serving_and_defaults_none():
    import argparse
    p = argparse.ArgumentParser()
    jailbreak_run._add_attacker_engine_args(p)
    ns = p.parse_args([])
    assert ns.attacker_serving is None and ns.attacker_engine == "vllm"
    assert p.parse_args(["--attacker-serving", "/x.json"]).attacker_serving == "/x.json"


# --- the aggregate refuses to mix modes ---------------------------------------


def _root_with(tmp_path, quantizations, *, engine="vllm", prefix=True):
    """Shards with a RECORDED identity (quantization + cache setting) for each
    non-None entry; a None entry is a pre-field sidecar recording neither."""
    from sisteer.capeval.engine_identity import EngineIdentityError

    root = tmp_path / "pair" / "none"
    for i, q in enumerate(quantizations):
        d = root / f"shard{i:02d}"
        d.mkdir(parents=True)
        doc = {"attacker_engine": engine, "attacker_model": ATTACKER}
        if q is not None:
            doc["attacker_quantization"] = q
            if prefix is not None:
                doc["attacker_prefix_caching"] = prefix
        (d / "attacker_engine.json").write_text(json.dumps(doc))
    return root, EngineIdentityError


def test_mixed_attacker_quantization_root_is_refused(tmp_path):
    root, err = _root_with(tmp_path, ["none", "fp8"])
    with pytest.raises(err) as e:
        jailbreak_run._assert_uniform_attacker_engine(root)
    assert "attacker_quantization" in e.value.differing


def test_uniform_recorded_root_passes_and_any_unknown_identity_refuses(tmp_path):
    root, err = _root_with(tmp_path, ["fp8", "fp8"])
    jailbreak_run._assert_uniform_attacker_engine(root)          # uniform AND known: fine
    # both pre-field ('unrecorded' on every shard) used to pass as "uniform"; an
    # unknown precision shared by every shard is uniformly unverifiable, refused
    root2, _ = _root_with(tmp_path / "old", [None, None])
    with pytest.raises(err, match="unrecorded") as e:
        jailbreak_run._assert_uniform_attacker_engine(root2)
    assert "attacker_quantization" in e.value.differing
    root3, _ = _root_with(tmp_path / "mix", [None, "none"])
    with pytest.raises(err) as e:                                # unverifiable vs verified
        jailbreak_run._assert_uniform_attacker_engine(root3)
    assert "attacker_quantization" in e.value.differing
    # a vLLM shard with a recorded precision but NO cache setting is unknown too
    root4, _ = _root_with(tmp_path / "nocache", ["none", "none"], prefix=None)
    with pytest.raises(err, match="no attacker_prefix_caching") as e:
        jailbreak_run._assert_uniform_attacker_engine(root4)
    assert "attacker_prefix_caching" in e.value.differing
    # the cache setting is an axis of the vLLM server only: HF shards record none
    root5, _ = _root_with(tmp_path / "hf", ["none", "none"], engine="hf", prefix=None)
    jailbreak_run._assert_uniform_attacker_engine(root5)


def test_all_unrecorded_aggregate_is_refused_before_grading(tmp_path):
    from sisteer.capeval.engine_identity import EngineIdentityError
    cfg = {"defenses": ["none", "si_constitution__psr_kl_all"], "private_run": "run",
           "prereg": {"threshold_grid": [0.25, 0.5, 0.75]}}
    for defense in cfg["defenses"]:
        for i in range(2):
            d = tmp_path / "run" / "pair" / defense / f"shard{i:02d}"
            d.mkdir(parents=True)
            (d / "attacker_engine.json").write_text(json.dumps(
                {"attacker_engine": "vllm", "attacker_model": ATTACKER}))   # pre-cutover sidecars

    def grade_fn(*a, **k):
        pytest.fail("grading must not start on an aggregate whose precision is unknown everywhere")

    with pytest.raises(EngineIdentityError, match="unrecorded"):
        jailbreak_run._aggregate_gated("pair", cfg, tmp_path, {}, grade_fn, {}, [],
                                       expected_goal_ids=[])


# --- the Modal smoke composes the seam for the ATTACKER only --------------------


def test_modal_smoke_attacker_standup_composes_the_seam_and_target_does_not():
    path = REPO / "scripts/modal/grokmt_served.py"
    text = path.read_text()
    tree = ast.parse(text)
    functions = {
        node.name: ast.get_source_segment(text, node) or ""
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "attacker_serving" in functions["_attacker_serving_options"]
    assert "model_role=\"attacker\"" in functions["_start_attacker"]
    for name in ("_geometry_argv", "_target_standup_cmd", "_start_target"):
        assert "attacker_serving" not in functions[name], name
        assert "prefix-caching" not in functions[name], name
        assert "quantization" not in functions[name], name


# --- the shell launcher, end to end against a fake vllm -------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _fake_venv(tmp_path) -> pathlib.Path:
    """A venv-shaped dir whose ``bin/vllm`` records its argv, prints the engine
    line vllm 0.25.1 would (honouring the flags it was given, unless
    FAKE_ENGINE_PREFIX overrides what it REPORTS), then serves /health."""
    venv = tmp_path / "fakevenv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(sys.executable)
    serve_dir = tmp_path / "www"
    serve_dir.mkdir()
    (serve_dir / "health").write_text("ok\n")
    fake = venv / "bin" / "vllm"
    fake.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        printf '%s\\n' "$@" > "$FAKE_ARGV_FILE"
        q=None; pc=False; port=8000
        while [ $# -gt 0 ]; do
          case "$1" in
            --quantization) q="$2"; shift 2;;
            --enable-prefix-caching) pc=True; shift;;
            --no-enable-prefix-caching) pc=False; shift;;
            --port) port="$2"; shift 2;;
            *) shift;;
          esac
        done
        [ -n "${{FAKE_ENGINE_PREFIX:-}}" ] && pc="$FAKE_ENGINE_PREFIX"
        echo "INFO Initializing a V1 LLM engine (v0.25.1) with config: model='m', dtype=torch.bfloat16, max_seq_len=8192, quantization=$q, kv_cache_dtype=auto, enable_prefix_caching=$pc"
        [ "$pc" = True ] && echo "INFO Mamba cache mode is set to 'align'"
        echo "INFO GPU KV cache size: 1,000 tokens"
        exec {sys.executable} -m http.server "$port" --bind 127.0.0.1 --directory {serve_dir}
        """))
    fake.chmod(0o755)
    return venv


def _run_launcher(tmp_path, venv, *extra, env_extra=None):
    port = _free_port()
    log = tmp_path / f"serve_{port}.log"
    argv_file = tmp_path / f"argv_{port}.txt"
    env = dict(os.environ, FAKE_ARGV_FILE=str(argv_file), **(env_extra or {}))
    r = subprocess.run(
        ["bash", str(REPO / "scripts/vllm_attacker_serve.sh"), "--venv", str(venv),
         "--model", "/snap/m", "--served-name", ATTACKER, "--port", str(port),
         "--log", str(log), "--wait", "60", *extra],
        env=env, text=True, capture_output=True, timeout=120)
    pid = None
    for line in r.stdout.splitlines():
        if line.startswith("VLLM_SERVER_PID="):
            pid = int(line.split("=", 1)[1])
    return r, pid, log, argv_file


def _kill(pid):
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def _alive(pid) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@pytest.mark.parametrize("extra, want_flags, want_q, want_pc", [
    ([], ["--enable-prefix-caching", "--mamba-cache-mode", "align"], "none", True),
    (["--prefix-caching", "off"], ["--no-enable-prefix-caching"], "none", False),
    (["--quantization", "fp8"],
     ["--quantization", "fp8", "--enable-prefix-caching", "--mamba-cache-mode", "align"],
     "fp8", True),
])
def test_launcher_composes_flags_from_the_seam_and_writes_a_verified_manifest(
        tmp_path, extra, want_flags, want_q, want_pc):
    venv = _fake_venv(tmp_path)
    r, pid, log, argv_file = _run_launcher(tmp_path, venv, *extra)
    try:
        assert r.returncode == 0, r.stdout + r.stderr
        assert pid and _alive(pid)
        argv = argv_file.read_text().splitlines()
        assert argv[:2] == ["serve", "/snap/m"]
        for flag in want_flags:
            assert flag in argv
        assert "--kv-cache-dtype" not in argv           # never an FP8 KV cache
        manifest_line = [l for l in r.stdout.splitlines() if l.startswith("VLLM_SERVING_MANIFEST=")]
        assert manifest_line, r.stdout
        manifest = serving.read_manifest(manifest_line[0].split("=", 1)[1])
        assert manifest["requested"]["attacker_quantization"] == want_q
        assert manifest["requested"]["prefix_caching"] is want_pc
        assert manifest["engine"]["enable_prefix_caching"] is want_pc
        assert manifest["engine"]["kv_cache_dtype"] == "auto"
        assert manifest["engine"]["max_model_len"] == 8192
        assert manifest["served_model_name"] == ATTACKER
        assert manifest["engine_verified"] is True
        # the driver-side fields are derivable from it
        fields = serving.driver_fields(manifest, manifest_path="x")
        assert fields["attacker_quantization"] == want_q
        assert fields["attacker_prefix_caching"] is want_pc
    finally:
        _kill(pid)


def test_launcher_kills_a_server_whose_engine_contradicts_the_request(tmp_path):
    venv = _fake_venv(tmp_path)
    # the engine REPORTS prefix caching off although it was asked for on
    r, pid, log, argv_file = _run_launcher(tmp_path, venv,
                                           env_extra={"FAKE_ENGINE_PREFIX": "False"})
    assert r.returncode == 1, r.stdout + r.stderr
    assert "contradicts" in r.stderr
    assert "REFUSED" in r.stderr and "prefix_caching" in r.stderr
    assert pid is None                                  # the server was never handed out
    # the fake server it started must be gone (it was killed on the contradiction)
    argv = argv_file.read_text().splitlines()
    port = int(argv[argv.index("--port") + 1])
    deadline = time.time() + 15
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                break
        time.sleep(0.5)
    else:
        raise AssertionError("contradicting server still listening")


def _path_without(tmp_path, tool: str) -> str:
    """A PATH equivalent to the current one with exactly ``tool`` removed:
    every other executable on PATH is symlinked into one dir (first PATH hit
    wins, as in real resolution), so a shell run under it lacks that one
    tool and nothing else."""
    farm = tmp_path / f"path-without-{tool}"
    farm.mkdir()
    for d in os.environ.get("PATH", os.defpath).split(os.pathsep):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            if name == tool or os.path.lexists(farm / name):
                continue
            (farm / name).symlink_to(pathlib.Path(d) / name)
    assert not os.path.lexists(farm / tool)
    return str(farm)


def test_launcher_readiness_probe_needs_no_curl(tmp_path):
    """The Modal images (debian_slim, nvidia/cuda) ship no curl. Before the
    probe moved into the serving module, the launcher's /health loop read
    `000` against a HEALTHY fake server for the whole --wait and then killed
    it -- the four launcher tests above passed on a workstation and failed
    only in the container (2026-09-06). Same launcher, same fake, a PATH with
    every tool but curl: it must come up healthy and hand the server out."""
    venv = _fake_venv(tmp_path)
    r, pid, log, argv_file = _run_launcher(
        tmp_path, venv, env_extra={"PATH": _path_without(tmp_path, "curl")})
    try:
        assert r.returncode == 0, r.stdout + r.stderr
        assert pid and _alive(pid)
        assert "HEALTHY" in r.stdout and "last=000" not in r.stderr
        assert any(l.startswith("VLLM_SERVING_MANIFEST=") for l in r.stdout.splitlines()), r.stdout
    finally:
        _kill(pid)


# --- the Modal smoke hands the VERIFIED manifest to every per-job Qwen driver --


def _grokmt():
    import importlib.util
    from test_grokmt_served_resume import _stub_modal
    sys.modules.setdefault("modal", _stub_modal())
    spec = importlib.util.spec_from_file_location(
        "_grokmt_attacker_serving", REPO / "scripts/modal/grokmt_served.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_DRIVER_KW = dict(attack="crescendo", cell="none", sub=0, subshards=3, ids_path="ids",
                  private_root="pr", proxy_url="http://127.0.0.1:1/v1", rt="rt.json",
                  attacker_port=1234, target_window=8704)


def test_driver_cmd_carries_the_serving_manifest_to_the_per_job_qwen_driver_only(tmp_path):
    g = _grokmt()
    qwen = g._driver_cmd("/tree", arm="qwen", attacker_serving="/run/srv/attacker_vllm.serving.json",
                         **_DRIVER_KW)
    assert qwen[qwen.index("--attacker-serving") + 1] == "/run/srv/attacker_vllm.serving.json"
    assert qwen[qwen.index("--attacker-engine") + 1] == "vllm"
    # the grok arm's attacker is hosted (its scorer is Qwen): no serving manifest
    grok = g._driver_cmd("/tree", arm="grok", attacker_serving="/m.json", **_DRIVER_KW)
    assert "--attacker-serving" not in grok and "--scorer-endpoint" in grok
    # the pooled branch takes its serving identity from --attacker-pool, never a manifest
    pj = tmp_path / "attacker_pool.json"
    pj.write_text(json.dumps({"url": "http://10.0.0.9:8000"}))
    pooled = g._driver_cmd("/tree", arm="qwen", attacker_pool=str(pj), attacker_serving="/m.json",
                           **_DRIVER_KW)
    assert "--attacker-pool" in pooled and "--attacker-serving" not in pooled
    # and BOTH _driver_cmd call sites in the smoke thread the manifest through
    text = (REPO / "scripts/modal/grokmt_served.py").read_text()
    tree = ast.parse(text)
    impl = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run_smoke_impl")
    calls = [n for n in ast.walk(impl)
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_driver_cmd"]
    assert len(calls) == 2
    for call in calls:
        assert "attacker_serving" in {kw.arg for kw in call.keywords}, ast.get_source_segment(text, call)
    # the smoke waits for the manifest after the attacker's health, before any driver
    src = ast.get_source_segment(text, impl)
    assert src.index("_wait_attacker_serving_manifest(") < src.index("_driver_cmd(")
    assert "ATTACKER_SERVING_UNVERIFIED" in src


def test_smoke_waits_for_the_verified_manifest_or_fails_typed(tmp_path):
    g = _grokmt()
    log = tmp_path / "attacker_vllm.log"
    log.write_text("")
    alive = SimpleNamespace(poll=lambda: None, returncode=None)
    # verified manifest present -> its path
    (tmp_path / "attacker_vllm.serving.json").write_text("{}")
    assert g._wait_attacker_serving_manifest(log, alive, timeout_s=1) == tmp_path / "attacker_vllm.serving.json"
    (tmp_path / "attacker_vllm.serving.json").unlink()
    # the writer's contradiction record -> typed, carrying the reason
    (tmp_path / "attacker_vllm.serving.error").write_text("FATAL: requested bf16 weights but engine quantization=fp8\n")
    with pytest.raises(g.AttackerServingUnverified) as e:
        g._wait_attacker_serving_manifest(log, alive, timeout_s=1)
    assert "quantization=fp8" in str(e.value)
    (tmp_path / "attacker_vllm.serving.error").unlink()
    # a dead attacker -> typed
    dead = SimpleNamespace(poll=lambda: 1, returncode=1)
    with pytest.raises(g.AttackerServingUnverified, match="exited"):
        g._wait_attacker_serving_manifest(log, dead, timeout_s=1)
    # nothing at all -> typed on the timeout, never a driver with 'unrecorded'
    with pytest.raises(g.AttackerServingUnverified, match="after 0s"):
        g._wait_attacker_serving_manifest(log, alive, timeout_s=0, poll_s=0.01)


# --- a POOLED run records the pool's advertised precision -----------------------


def _attested_serving(*, quantization="None", prefix="True"):
    opts = serving.AttackerServingOptions("none" if quantization == "None" else quantization,
                                          prefix == "True")
    engine = serving.parse_engine_log(_engine_log(quantization=quantization, prefix=prefix == "True"))
    manifest = {"schema": serving.MANIFEST_SCHEMA, "requested": opts.as_dict(), "engine": engine,
                "engine_verified": True}
    return serving.attested_serving_record(manifest, manifest_path="/pool/serving.json")


def test_pooled_run_copies_only_an_engine_attested_serving_record():
    attested = _attested_serving()
    args = SimpleNamespace(attacker_engine="vllm", attacker_serving=None, attacker_model=ATTACKER,
                           _attacker_pool={"identity": {"url": "http://10.0.0.9:8000"},
                                           "serving": attested, "declared_quantization": "none"})
    f = jailbreak_run._attacker_serving_provenance(args, engine="vllm", served=ATTACKER)
    assert f["attacker_quantization"] == "none" and f["attacker_prefix_caching"] is True
    assert f["attacker_mamba_cache_mode"] == "align" and f["attacker_kv_cache_dtype"] == "auto"
    assert f["attacker_serving_source"] == "attacker_pool"
    assert f["attacker_serving_manifest"] == "/pool/serving.json"
    assert f["attacker_vllm_version"] == "0.25.1"
    assert f["attacker_enable_thinking"] is serving.ATTACKER_ENABLE_THINKING
    # the attested path records what the ENGINE LINE said: an fp8 pool advertises fp8
    args._attacker_pool = {"identity": {"url": "u"}, "serving": _attested_serving(quantization="fp8"),
                           "declared_quantization": None}
    assert jailbreak_run._attacker_serving_provenance(args, engine="vllm", served=ATTACKER)[
        "attacker_quantization"] == "fp8"
    # a bare DECLARATION (the first pool's literal) with no attested block is refused
    args._attacker_pool = {"identity": {"url": "u"}, "serving": None, "declared_quantization": "none"}
    with pytest.raises(SystemExit, match="refusing to copy a declaration"):
        jailbreak_run._attacker_serving_provenance(args, engine="vllm", served=ATTACKER)
    # an unattested block is refused too
    args._attacker_pool = {"identity": {"url": "u"}, "serving": {"attacker_quantization": "none"},
                           "declared_quantization": None}
    with pytest.raises(SystemExit, match="not engine-attested"):
        jailbreak_run._attacker_serving_provenance(args, engine="vllm", served=ATTACKER)
    # an attested block whose advertised value contradicts its own engine line is refused
    tampered = dict(_attested_serving()); tampered["attacker_quantization"] = "fp8"
    args._attacker_pool = {"identity": {"url": "u"}, "serving": tampered, "declared_quantization": None}
    with pytest.raises(SystemExit, match="contradict"):
        jailbreak_run._attacker_serving_provenance(args, engine="vllm", served=ATTACKER)
    # a pool that advertises nothing about its serving is refused: a shared pool must
    # attest, so a record without the block is stale, stripped or malformed
    args._attacker_pool = {"identity": {"url": "http://10.0.0.9:8000"}, "serving": None,
                           "declared_quantization": None}
    with pytest.raises(SystemExit, match="no engine-attested serving record at all"):
        jailbreak_run._attacker_serving_provenance(args, engine="vllm", served=ATTACKER)
    # 'unrecorded' remains the per-job vllm-without-manifest outcome only
    plain = SimpleNamespace(attacker_engine="vllm", attacker_serving=None, attacker_model=ATTACKER)
    assert jailbreak_run._attacker_serving_provenance(plain, engine="vllm", served=ATTACKER)[
        "attacker_quantization"] == "unrecorded"
    # a manifest alongside the pool is a second source: refused
    args.attacker_serving = "/m.json"
    with pytest.raises(SystemExit, match="second source"):
        jailbreak_run._attacker_serving_provenance(args, engine="vllm", served=ATTACKER)


def test_resolve_attacker_pool_records_the_attested_block_and_the_declaration():
    from sisteer.eval.attacks import attacker_pool as apool
    attested = _attested_serving()
    base = {"url": "http://10.0.0.9:8000", "model": ATTACKER, "snapshot": "27502c87" * 5,
            "max_model_len": 19200, "max_num_seqs": 37, "api_key_env_name": "VLLM_API_KEY",
            "job_id": "fc-1", "node": "ta-1", "started_at": "2026-09-05T00:00:00Z",
            "attacker_quantization": "none", "prefix_caching": True, "serving": attested}
    ep = apool.PoolEndpoint.from_dict(base)
    args = SimpleNamespace(attacker_pool="/pool/endpoint.json", attacker_engine="vllm", attacker_model=None)
    rec = jailbreak_run._resolve_attacker_pool(
        args, _cfg(), connect=lambda spec, **kw: (ep, "k" * 43, {"liveness": {"kind": "modal"}}))
    assert rec["serving"] == attested and rec["declared_quantization"] == "none"
    assert args.attacker_endpoint == "http://10.0.0.9:8000"
    # the first pool's record shape (a literal, no attested block) is carried as a declaration only
    ep2 = apool.PoolEndpoint.from_dict({k: v for k, v in base.items() if k != "serving"})
    rec2 = jailbreak_run._resolve_attacker_pool(
        args, _cfg(), connect=lambda spec, **kw: (ep2, "k" * 43, {}))
    assert rec2["serving"] is None and rec2["declared_quantization"] == "none"


# --- the aggregate compares attacker identity ACROSS its defenses ---------------


def _aggregate_root(tmp_path, per_defense: dict):
    """``{defense: quantization}`` -> a private root where each defense's shards
    are internally uniform."""
    cfg = {"defenses": list(per_defense), "private_run": "run",
           "prereg": {"threshold_grid": [0.25, 0.5, 0.75]}}
    for defense, q in per_defense.items():
        for i in range(2):
            d = tmp_path / "run" / "pair" / defense / f"shard{i:02d}"
            d.mkdir(parents=True)
            (d / "attacker_engine.json").write_text(json.dumps(
                {"attacker_engine": "vllm", "attacker_model": ATTACKER, "attacker_quantization": q,
                 "attacker_prefix_caching": True}))
    return cfg


def test_aggregate_refuses_a_bf16_defense_beside_an_fp8_defense_each_internally_uniform(tmp_path):
    from sisteer.capeval.engine_identity import EngineIdentityError
    cfg = _aggregate_root(tmp_path, {"none": "none", "si_constitution__psr_kl_all": "fp8"})
    roots = [jailbreak_run._gated_defense_root(cfg, tmp_path, "pair", d) for d in cfg["defenses"]]
    for r in roots:
        jailbreak_run._assert_uniform_attacker_engine(r)        # each root alone: uniform, passes
    with pytest.raises(EngineIdentityError) as e:               # across the aggregate: refused
        jailbreak_run._assert_uniform_attacker_engine(*roots)
    assert "attacker_quantization" in e.value.differing

    def grade_fn(*a, **k):
        pytest.fail("grading must not start on a mixed-precision aggregate")

    with pytest.raises(EngineIdentityError):
        jailbreak_run._aggregate_gated("pair", cfg, tmp_path, {}, grade_fn, {}, [],
                                       expected_goal_ids=[])
    # the same layout with ONE precision passes the cross-defense check
    cfg2 = _aggregate_root(tmp_path / "ok", {"none": "none", "si_constitution__psr_kl_all": "none"})
    jailbreak_run._assert_uniform_attacker_engine(
        *[jailbreak_run._gated_defense_root(cfg2, tmp_path / "ok", "pair", d) for d in cfg2["defenses"]])


def test_prefix_cache_setting_is_part_of_the_compared_identity(tmp_path):
    from sisteer.capeval.engine_identity import EngineIdentityError
    root = tmp_path / "pair" / "none"
    for i, pc in enumerate([True, False]):
        d = root / f"shard{i:02d}"
        d.mkdir(parents=True)
        (d / "attacker_engine.json").write_text(json.dumps(
            {"attacker_engine": "vllm", "attacker_model": ATTACKER,
             "attacker_quantization": "none", "attacker_prefix_caching": pc}))
    with pytest.raises(EngineIdentityError) as e:
        jailbreak_run._assert_uniform_attacker_engine(root)
    assert "attacker_prefix_caching" in e.value.differing


# --- every consumed shard must carry ONE readable identity (fail closed) --------


def _shard(root, name, manifest=None, *, raw=None, with_rows=True):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    if with_rows:
        (d / "responses.jsonl").write_text(json.dumps({"goal_id": "g1", "sample_idx": 0, "response": "x"}) + "\n")
        (d / "progress.jsonl").write_text(json.dumps({"goal_id": "g1", "outcome": "done"}) + "\n")
    if raw is not None:
        (d / "attacker_engine.json").write_text(raw)
    elif manifest is not None:
        (d / "attacker_engine.json").write_text(json.dumps(manifest))
    return d


_BF16 = {"attacker_engine": "vllm", "attacker_model": ATTACKER,
         "attacker_quantization": "none", "attacker_prefix_caching": True}


def test_shard_with_rows_but_no_manifest_is_refused(tmp_path):
    from sisteer.capeval.engine_identity import EngineIdentityError
    root = tmp_path / "pair" / "none"
    _shard(root, "shard00", _BF16)
    _shard(root, "shard01")                       # responses + progress, no identity
    with pytest.raises(EngineIdentityError, match="no attacker_engine.json"):
        jailbreak_run._assert_uniform_attacker_engine(root)


def test_corrupt_manifest_is_refused_not_skipped(tmp_path):
    # the reviewer's reproduction: one valid bf16 manifest + one broken one used to pass
    from sisteer.capeval.engine_identity import EngineIdentityError
    root = tmp_path / "pair" / "none"
    _shard(root, "shard00", _BF16)
    _shard(root, "shard01", raw="{not json")
    with pytest.raises(EngineIdentityError, match="not readable JSON"):
        jailbreak_run._assert_uniform_attacker_engine(root)
    # parseable but naming no engine/model is just as unknown
    root2 = tmp_path / "pair" / "psr"
    _shard(root2, "shard00", _BF16)
    _shard(root2, "shard01", {"attacker_quantization": "fp8"})
    with pytest.raises(EngineIdentityError, match="names no attacker_engine"):
        jailbreak_run._assert_uniform_attacker_engine(root2)
    root3 = tmp_path / "pair" / "list"
    _shard(root3, "shard00", raw="[1, 2]")
    with pytest.raises(EngineIdentityError, match="names no attacker_engine"):
        jailbreak_run._assert_uniform_attacker_engine(root3)


def test_zero_manifest_root_is_refused_never_vacuously_passed(tmp_path):
    from sisteer.capeval.engine_identity import EngineIdentityError
    root = tmp_path / "pair" / "none"
    _shard(root, "shard00")
    _shard(root, "shard01")
    with pytest.raises(EngineIdentityError, match="no attacker_engine.json"):
        jailbreak_run._assert_uniform_attacker_engine(root)
    # a root with no shard at all cannot establish an identity either
    empty = tmp_path / "pair" / "empty"
    empty.mkdir(parents=True)
    with pytest.raises(EngineIdentityError, match="no shard\\* directory"):
        jailbreak_run._assert_uniform_attacker_engine(empty)
    # and the cross-defense form refuses on the FIRST unknown shard before grading
    good = tmp_path / "pair" / "good"
    _shard(good, "shard00", _BF16)
    with pytest.raises(EngineIdentityError):
        jailbreak_run._assert_uniform_attacker_engine(good, root)
    jailbreak_run._assert_uniform_attacker_engine(good)          # the complement still passes


# --- a pooled smoke refuses an unattested pool BEFORE any target-side work ------


def _pool_record(**over):
    d = {"url": "https://test--attacker-pool.modal.run", "model": ATTACKER, "snapshot": "27502c87" * 5,
         "max_model_len": 19200, "max_num_seqs": 37, "api_key_env_name": "VLLM_API_KEY",
         "job_id": "fc-test", "node": "ta-pool", "started_at": "2026-09-05T00:00:00Z",
         "runtime": "modal_server", "liveness": {"kind": "modal", "app_name": "a", "app_id": "ap-1",
                                                 "server_name": "s", "server_id": "sv-1",
                                                 "call_id": "fc-test", "container_id": "ta-pool"}}
    d.update(over)
    return d


@pytest.mark.parametrize("label, record", [
    ("nothing advertised", _pool_record()),
    ("bare declaration", _pool_record(attacker_quantization="none")),
    ("unattested block", _pool_record(serving={"attacker_quantization": "none", "prefix_caching": True})),
    ("contradictory block", _pool_record(serving=dict(_attested_serving(), attacker_quantization="fp8"))),
])
def test_pooled_smoke_refuses_an_unattested_pool_before_the_hf_probe_and_the_target(
        tmp_path, monkeypatch, label, record):
    g = _grokmt()
    tree = tmp_path / "tree"
    (tree / "src" / "sisteer").mkdir(parents=True)
    monkeypatch.setattr(g, "_tree", lambda sha: str(tree))
    monkeypatch.setattr(g, "resolve_target_geometry", lambda *a, **k: SimpleNamespace(
        describe=lambda: "geom", as_dict=lambda: {}, max_model_len=8704, max_num_seqs=8))
    monkeypatch.setattr(g, "_sync", lambda syncer: None)
    monkeypatch.setattr(g, "RunSyncer", lambda run_local, dest: None)
    monkeypatch.setattr(g.subprocess, "run",
                        lambda *a, **k: pytest.fail(f"[{label}] the HF timing probe ran on an unattested pool"))
    monkeypatch.setattr(g, "_start_target",
                        lambda *a, **k: pytest.fail(f"[{label}] the target server was stood up on an unattested pool"))
    monkeypatch.setattr(g, "_http_ok", lambda *a, **k: pytest.fail(f"[{label}] a network probe ran"))
    monkeypatch.setattr(g, "_start_attacker", lambda *a, **k: pytest.fail(f"[{label}] a per-job attacker spawned"))
    summary = g._run_smoke_impl("s" * 40, ["g1"], f"t-{abs(hash(label))}", ["qwen"], hf_probe=True,
                                attacker_pool=record)
    assert summary["status"] == "ATTACKER_POOL_REFUSED", label
    assert "PoolUnattested" in summary["attacker_pool_err"], label
    assert "attacker_pool_verified" not in summary["timeline"] and "target_up" not in summary["timeline"]


# --- the canary path is held to the same two rules -------------------------------


def _canary_cfg_for(tmp_path):
    return {**_cfg(), "private_run": "run", "out_dir": "out", "model": "m", "dtype": "bfloat16",
            "attacks": {"canary": {"defenses": ["none", "prompted"], "n_goals": 2, "xstest_n": 1}},
            "prereg": {"threshold_grid": [0.25, 0.5, 0.75]},
            "datasets": {"xstest": {"revision": "r"}}}


def test_canary_generate_refuses_a_block_less_pool_before_loading_the_target(tmp_path, monkeypatch):
    import types
    endpoint = tmp_path / "pool" / "endpoint.json"
    endpoint.parent.mkdir()
    endpoint.write_text(json.dumps(_pool_record()))          # nothing advertised about its serving
    loaded = []
    models = types.ModuleType("sisteer.models")
    models.load_model_and_tokenizer = lambda *a, **k: loaded.append(a) or pytest.fail(
        "the canary loaded the target before the pool was verified")
    monkeypatch.setitem(sys.modules, "sisteer.models", models)
    jb = types.ModuleType("sisteer.eval.jailbreak")
    jb.COMPILED, jb.build_defense = set(), lambda *a, **k: pytest.fail("a defense was built")
    monkeypatch.setitem(sys.modules, "sisteer.eval.jailbreak", jb)
    args = SimpleNamespace(attacker_pool=str(endpoint), attacker_engine="vllm", attacker_model=None,
                           attacker_serving=None, private_root=str(tmp_path / "priv"), attack="pair",
                           n_goals=None, xstest_n=None, scale=1.0, positions="all")
    with pytest.raises(SystemExit, match="PoolUnattested") as e:
        jailbreak_run.cmd_canary_generate(args, _canary_cfg_for(tmp_path))
    assert "no engine-attested serving record" in str(e.value) and loaded == []
    # and the order is pinned in the source: the pool is resolved before the target loads
    text = (REPO / "scripts/jailbreak_run.py").read_text()
    src = ast.get_source_segment(text, next(
        n for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef) and n.name == "cmd_canary_generate"))
    assert src.index("_resolve_attacker_pool(args, cfg)") < src.index("load_model_and_tokenizer(")


def test_canary_calibrate_refuses_an_all_unrecorded_canary_before_grading(tmp_path, monkeypatch):
    import types
    from sisteer.capeval.engine_identity import EngineIdentityError
    cfg = _canary_cfg_for(tmp_path)
    root = tmp_path / "priv"
    for defense in cfg["attacks"]["canary"]["defenses"]:
        d = jailbreak_run._canary_root(cfg, root, "pair", defense)     # the flat layout, as generate writes it
        (d / "attacker_engine.json").write_text(json.dumps(
            {"attacker_engine": "vllm", "attacker_model": ATTACKER}))   # pre-cutover sidecar: unrecorded
        (d / "responses.jsonl").write_text("")
    for name, attrs in (("judge_batch", {"grade_xstest": lambda *a, **k: pytest.fail("graded")}),
                        ("sisteer.eval.attacks.grading", {"BatchStrongREJECTGrader": lambda **k: pytest.fail("grader built")}),
                        ("sisteer.eval.jailbreak_data", {"load_xstest_safe": lambda **k: []})):
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(jailbreak_run, "_grade_judges",
                        lambda *a, **k: pytest.fail("judges were built for an all-unrecorded canary"))
    args = SimpleNamespace(private_root=str(root), attack="pair", n_goals=None, poll_interval=1)
    with pytest.raises(EngineIdentityError, match="unrecorded"):
        jailbreak_run.cmd_canary_calibrate(args, cfg)
    # a canary whose defenses differ in precision is refused the same way
    dirs = [jailbreak_run._canary_root(cfg, root, "pair", d) for d in cfg["attacks"]["canary"]["defenses"]]
    for d, q in zip(dirs, ("none", "fp8")):
        (d / "attacker_engine.json").write_text(json.dumps(
            {"attacker_engine": "vllm", "attacker_model": ATTACKER, "attacker_quantization": q,
             "attacker_prefix_caching": True}))
    with pytest.raises(EngineIdentityError) as e:
        jailbreak_run.cmd_canary_calibrate(args, cfg)
    assert "attacker_quantization" in e.value.differing
    # the complement: a known, uniform identity passes the flat check itself
    for d in dirs:
        (d / "attacker_engine.json").write_text(json.dumps(
            {"attacker_engine": "vllm", "attacker_model": ATTACKER, "attacker_quantization": "none",
             "attacker_prefix_caching": True}))
    jailbreak_run._assert_uniform_attacker_identities(dirs, axis="canary attacker engine")
    with pytest.raises(EngineIdentityError, match="nothing to check"):
        jailbreak_run._assert_uniform_attacker_identities([])


# --- the manifest writer is resume-safe: no relabelling of existing work ----------


def _prior_work(d, manifest, *, responses=True):
    d.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        (d / "attacker_engine.json").write_text(manifest if isinstance(manifest, str) else json.dumps(manifest, indent=2))
    if responses:
        (d / "responses.jsonl").write_text(json.dumps({"goal_id": "g1", "sample_idx": 0, "response": "old"}) + "\n")
        (d / "progress.jsonl").write_text(json.dumps({"goal_id": "g1", "outcome": "done"}) + "\n")
    return (d / "attacker_engine.json").read_bytes() if manifest is not None else None


_FP8_VLLM = {"attacker_engine": "vllm", "attacker_model": ATTACKER,
             "attacker_quantization": "fp8", "attacker_prefix_caching": True}


def _hf_args():
    return SimpleNamespace(attacker_engine="hf", attacker_serving=None)


@pytest.mark.parametrize("layout", ["canary_flat", "sharded"])
def test_manifest_writer_refuses_to_relabel_existing_work_under_a_changed_identity(tmp_path, layout):
    from sisteer.capeval.engine_identity import EngineIdentityError
    d = (tmp_path / "run" / "canary" / "pair" / "none" if layout == "canary_flat"
         else tmp_path / "run" / "pair" / "none" / "shard00")
    before = _prior_work(d, _FP8_VLLM)
    # the reviewer's reproduction: FP8-vLLM responses survive, the run restarts as HF
    with pytest.raises(EngineIdentityError) as e:
        jailbreak_run._write_engine_manifest(d, _cfg(), _hf_args(), "pair")
    assert "attacker_engine" in e.value.differing and "attacker_quantization" in e.value.differing
    assert (d / "attacker_engine.json").read_bytes() == before          # untouched
    assert (d / "responses.jsonl").read_text().count("old") == 1        # untouched
    assert not (d / jailbreak_run._RESUME_RECORDS).exists()
    # a changed serving identity with the SAME engine is refused too (bf16 manifest onto fp8 work)
    bf16 = SimpleNamespace(attacker_engine="vllm", attacker_model=ATTACKER,
                           attacker_serving=str(_manifest(tmp_path, quantization="none")))
    with pytest.raises(EngineIdentityError) as e:
        jailbreak_run._write_engine_manifest(d, _cfg(), bf16, "pair")
    assert e.value.differing == ("attacker_quantization",)
    assert (d / "attacker_engine.json").read_bytes() == before


def test_manifest_writer_refuses_work_with_a_missing_corrupt_or_unknown_manifest(tmp_path):
    from sisteer.capeval.engine_identity import EngineIdentityError
    hf = _hf_args()
    d = tmp_path / "missing"; _prior_work(d, None)
    with pytest.raises(EngineIdentityError, match="no attacker_engine.json"):
        jailbreak_run._write_engine_manifest(d, _cfg(), hf, "pair")
    assert not (d / "attacker_engine.json").exists()                    # nothing was created
    d = tmp_path / "corrupt"; before = _prior_work(d, "{not json")
    with pytest.raises(EngineIdentityError, match="not readable JSON"):
        jailbreak_run._write_engine_manifest(d, _cfg(), hf, "pair")
    assert (d / "attacker_engine.json").read_bytes() == before
    d = tmp_path / "unknown"; before = _prior_work(d, {"attacker_engine": "hf", "attacker_model": ATTACKER})
    with pytest.raises(EngineIdentityError, match="cannot be resumed under any identity") as e:
        jailbreak_run._write_engine_manifest(d, _cfg(), hf, "pair")      # even the SAME engine
    assert e.value.differing == ("attacker_quantization",)
    assert (d / "attacker_engine.json").read_bytes() == before


def test_manifest_writer_leaves_a_matching_identity_untouched_and_records_the_resume(tmp_path):
    d = tmp_path / "run" / "pair" / "none" / "shard00"
    manifest_path = _manifest(tmp_path, quantization="fp8")
    args = SimpleNamespace(attacker_engine="vllm", attacker_model=ATTACKER, attacker_serving=str(manifest_path))
    jailbreak_run._write_engine_manifest(d, _cfg(), args, "pair")        # an empty run: created
    before = (d / "attacker_engine.json").read_bytes()
    assert json.loads(before)["attacker_quantization"] == "fp8"
    (d / "responses.jsonl").write_text("{}\n")                           # ... then work lands
    jailbreak_run._write_engine_manifest(d, _cfg(), args, "pair")        # the resume: same identity
    assert (d / "attacker_engine.json").read_bytes() == before           # byte-identical
    records = (d / jailbreak_run._RESUME_RECORDS).read_text().splitlines()
    assert len(records) == 1 and json.loads(records[0])["manifest"]["attacker_quantization"] == "fp8"
    # the sharded generate path writes the manifest before generation starts
    text = (REPO / "scripts/jailbreak_run.py").read_text()
    src = ast.get_source_segment(text, next(
        n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "cmd_generate"))
    assert src.index("_write_engine_manifest(out_dir") < src.index("_run_gated_attack(")


def test_interrupted_canary_resume_under_a_changed_pool_is_refused_before_generation(tmp_path, monkeypatch):
    import types
    from sisteer.capeval.engine_identity import EngineIdentityError
    cfg = _canary_cfg_for(tmp_path)
    root = tmp_path / "priv"
    first = jailbreak_run._canary_root(cfg, root, "pair", "none")
    before = _prior_work(first, _FP8_VLLM)                               # completed FP8-pool work
    (first / "local_scores.jsonl").write_text('{"goal_id": "g1", "scores": [0.9]}\n')
    scores_before = (first / "local_scores.jsonl").read_bytes()
    models = types.ModuleType("sisteer.models"); models.load_model_and_tokenizer = lambda *a, **k: (None, None)
    monkeypatch.setitem(sys.modules, "sisteer.models", models)
    jb = types.ModuleType("sisteer.eval.jailbreak"); jb.COMPILED, jb.build_defense = set(), lambda *a, **k: object()
    monkeypatch.setitem(sys.modules, "sisteer.eval.jailbreak", jb)
    monkeypatch.setattr(jailbreak_run, "_build_attacker_seams", lambda cfg, args: (None, None, None))
    monkeypatch.setattr(jailbreak_run, "_eval_goals", lambda cfg, **k: [])
    for name in ("_run_gated_attack", "_rescore_local", "_xstest_generate"):
        monkeypatch.setattr(jailbreak_run, name,
                            lambda *a, _n=name, **k: pytest.fail(f"{_n} ran on relabelled work"))
    # the restart: the same root, the attacker now HF (no pool, bf16)
    args = SimpleNamespace(attacker_pool=None, attacker_engine="hf", attacker_model=None, attacker_serving=None,
                           private_root=str(root), attack="pair", n_goals=None, xstest_n=None,
                           scale=1.0, positions="all")
    with pytest.raises(EngineIdentityError) as e:
        jailbreak_run.cmd_canary_generate(args, cfg)
    assert "attacker_engine" in e.value.differing
    assert (first / "attacker_engine.json").read_bytes() == before      # not overwritten
    assert (first / "local_scores.jsonl").read_bytes() == scores_before  # not rescored
    assert (first / "responses.jsonl").read_text().count("old") == 1


# --- persisted work is ANYTHING beside the manifest files (denylist, not allowlist) --


@pytest.mark.parametrize("evidence", [
    "attacker_calls.jsonl",       # the reviewer's ActorAttack case: a call ledger, no responses yet
    "scorer_calls.jsonl",
    "target_timing.jsonl",
    "served_target.json",
    "foo.bin",                    # an artifact no list knows about
    "subdir/",                    # a subdirectory
])
def test_any_entry_beside_the_manifest_files_is_persisted_work(tmp_path, evidence):
    from sisteer.capeval.engine_identity import EngineIdentityError
    d = tmp_path / "run" / "actorattack" / "none" / "shard00"
    before = _prior_work(d, _FP8_VLLM, responses=False)
    if evidence.endswith("/"):
        (d / evidence.rstrip("/")).mkdir()
        ledger_before = None
    else:
        (d / evidence).write_text('{"role": "attacker", "latency_s": 1.0}\n')
        ledger_before = (d / evidence).read_bytes()
    with pytest.raises(EngineIdentityError) as e:
        jailbreak_run._write_engine_manifest(d, _cfg(), _hf_args(), "actorattack")
    assert "attacker_engine" in e.value.differing
    assert (d / "attacker_engine.json").read_bytes() == before            # manifest untouched
    if ledger_before is not None:
        assert (d / evidence).read_bytes() == ledger_before                # evidence untouched
    assert not (d / jailbreak_run._RESUME_RECORDS).exists()
    # the same evidence with NO manifest beside it: refused, nothing created
    d2 = tmp_path / "run" / "actorattack" / "none" / "shard01"
    _prior_work(d2, None, responses=False)
    if evidence.endswith("/"):
        (d2 / evidence.rstrip("/")).mkdir()
    else:
        (d2 / evidence).write_text("x\n")
    with pytest.raises(EngineIdentityError, match="no attacker_engine.json"):
        jailbreak_run._write_engine_manifest(d2, _cfg(), _hf_args(), "actorattack")
    assert not (d2 / "attacker_engine.json").exists()


def test_only_the_manifest_files_do_not_count_as_persisted_work(tmp_path):
    """The manifest-class files -- the attacker manifest, its resume records,
    and the served target's per-process counters ledger
    (``served_target.attempts.jsonl``: a record ABOUT attempts, like the
    resume records, never persisted work) --
    are the ONLY files that do not count as evidence on resume. A dir holding
    nothing but them is owned by the run that starts now, so its manifest is
    (re)written, not refused. Pins the set itself so a new manifest-class
    file cannot join the writer's denylist without this test following."""
    assert jailbreak_run._MANIFEST_FILES == {
        "attacker_engine.json", "attacker_engine.resumes.jsonl", "served_target.attempts.jsonl"}
    # an empty dir: created
    d = tmp_path / "run" / "pair" / "none" / "shard00"
    d.mkdir(parents=True)
    jailbreak_run._write_engine_manifest(d, _cfg(), _hf_args(), "pair")
    assert json.loads((d / "attacker_engine.json").read_text())["attacker_engine"] == "hf"
    # a dir holding only the manifest-class files (the manifest, a resume record
    # and the served target's counters ledger) holds no evidence to relabel: the
    # run that starts now owns it, so the manifest is written for that run, not refused
    d2 = tmp_path / "run" / "pair" / "none" / "shard01"
    _prior_work(d2, _FP8_VLLM, responses=False)
    (d2 / jailbreak_run._RESUME_RECORDS).write_text("{}\n")
    (d2 / "served_target.attempts.jsonl").write_text(
        json.dumps({"stage": "final", "all_turns_check": {"n_refused_total": 0}}) + "\n")
    jailbreak_run._write_engine_manifest(d2, _cfg(), _hf_args(), "pair")
    assert json.loads((d2 / "attacker_engine.json").read_text())["attacker_engine"] == "hf"
    assert sorted(p.name for p in d2.iterdir()) == sorted(jailbreak_run._MANIFEST_FILES)
    # the counters ledger is left as it was (append-only; never deleted by the writer)
    assert (d2 / "served_target.attempts.jsonl").read_text().startswith('{"stage": "final"')
    # the guard reads the directory, not the list: the list is for messages only
    text = (REPO / "scripts/jailbreak_run.py").read_text()
    src = ast.get_source_segment(text, next(
        n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "_write_engine_manifest"))
    assert "out_dir.iterdir()" in src and "_MANIFEST_FILES" in src
    assert "_PERSISTED_WORK" not in src
