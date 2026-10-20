"""CPU-only contract tests for the Modal lens and full-suite launchers.

The launch modules are imported with a synthetic Modal stub: these tests cover
pure argv, environment, pin, identity, and report parsing logic without making
network calls or starting containers.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from front_door_testlib import MATCHERS, SCAN, token_hits

#: one tier of the publication scan's token table, matched as the scan matches it (the tokens are never spelled here)
STORAGE_TOKENS = SCAN.Matcher([r for r in MATCHERS["rows"] if r.tier == "storage"])

REPO = Path(__file__).resolve().parents[1]


class _Chain:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: self


class _App:
    def __init__(self, name=None):
        self.name = name
        self.description = name
        self.app_id = "ap-TEST"

    def function(self, **_kwargs):
        return lambda fn: fn

    def local_entrypoint(self, **_kwargs):
        return lambda fn: fn


class _Volume:
    @classmethod
    def from_name(cls, *_args, **_kwargs):
        return cls()

    def commit(self):
        return None

    def reload(self):
        return None


def _modal_stub(*, is_local: bool = True) -> types.ModuleType:
    mod = types.ModuleType("modal")
    mod.is_local = lambda: is_local
    mod.App = _App
    mod.Volume = _Volume
    mod.Secret = types.SimpleNamespace(from_name=lambda name: {"secret": name})
    mod.Image = types.SimpleNamespace(from_registry=lambda *_a, **_k: _Chain(),
                                      debian_slim=lambda *_a, **_k: _Chain())
    mod.FunctionCall = types.SimpleNamespace(from_id=lambda _id: None)
    return mod


def _load(name: str, rel: str, *, stub_modal: bool = False,
          modal_local: bool = True):
    old = sys.modules.get("modal")
    if stub_modal:
        sys.modules["modal"] = _modal_stub(is_local=modal_local)
    try:
        path = Path(rel) if Path(rel).is_absolute() else REPO / rel
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if stub_modal:
            if old is None:
                sys.modules.pop("modal", None)
            else:
                sys.modules["modal"] = old


@pytest.fixture(scope="module")
def gate():
    return _load("_modal_lens_gate_test", "scripts/modal/lens_vectorize_gate.py",
                 stub_modal=True)


@pytest.fixture(scope="module")
def reviewed_gate():
    return _load("_reviewed_lens_gate_test", "scripts/lens_vectorize_gate.py")


@pytest.fixture(scope="module")
def cpu_suite():
    return _load("_modal_cpu_suite_test", "scripts/modal/cpu_suite.py",
                 stub_modal=True)


def test_lens_modal_mirrors_reviewed_model_table(gate, reviewed_gate):
    assert tuple(gate.MODELS) == reviewed_gate.MODEL_ORDER
    assert {
        key: (spec["hf"], spec["rev"], spec["cfg"])
        for key, spec in gate.MODELS.items()
    } == {
        key: (spec.hf, spec.rev, spec.cfg)
        for key, spec in reviewed_gate.MODELS.items()
    }
    assert gate.DEFAULT_REFERENCE_SHA == reviewed_gate.DEFAULT_REFERENCE_SHA


def test_lens_modal_uses_transformers_split_and_one_function_per_family(gate):
    assert gate.TF_VARIANT == {
        "llama8b": "tf5150", "gemma4e4b": "tf5141", "qwen35_4b": "tf5150",
        "glm46vflash": "tf5150", "nemotron3nano4b": "tf5150",
    }
    assert gate.TRANSFORMERS_PIN == {"tf5141": "5.14.1", "tf5150": "5.15.0"}
    # Both images resolve with uv from the same certified input/cutoff as the
    # former Slurm gate, rather than a second hand-copied dependency list.
    assert gate.UV_VERSION == "0.12.1"
    assert gate.SERVING_EXCLUDE_NEWER == "2026-08-03T00:00:00Z"
    assert gate.TRANSFORMERS_EXCLUDE_NEWER == {
        "tf5141": None, "tf5150": "2026-08-11T00:00:00Z",
    }
    vllm_reqs = gate.VLLM_REQUIREMENTS.read_text()
    assert "vllm-0.25.1+cu129" in vllm_reqs
    assert ("#sha256=9e206f370c934a2d4b6b1f05d3d09708d344e05d80260189ef19f60755709431"
            in vllm_reqs)
    for pin in ("torch==2.11.0+cu129", "torchvision==0.26.0+cu129",
                "torchaudio==2.11.0+cu129", "vllm-lens==1.2.1"):
        assert pin in vllm_reqs
    src = (REPO / "scripts/modal/lens_vectorize_gate.py").read_text()
    assert "uv pip install --python $(command -v python)" in src
    assert "SECONDARY_PINS" not in src
    assert set(gate.GATE_FN) == set(gate.MODELS)
    assert len({id(fn) for fn in gate.GATE_FN.values()}) == len(gate.MODELS)
    assert gate.GATE_FN["llama8b"].__name__ == "gate_llama8b"
    assert gate.GATE_FN["gemma4e4b"].__name__ == "gate_gemma4e4b"
    assert gate.GATE_TIMEOUT_S >= 8 * 3600


def test_lens_modal_gate_argv_preserves_sbatch_seams_and_default_mode(gate):
    argv = gate.gate_argv(
        "/usr/bin/python", "/tmp/candidate", "/tmp/reference",
        model="llama8b", ref_sha="a" * 40, results_root="/tmp/results",
        cases=["case:a", "case:b"], batches="8,32", timing_batches="8,32,64",
        measure_s=12, legacy_ab=False, reduction_mode=None, ready_timeout_s=2700,
    )
    joined = " ".join(argv)
    assert argv[:3] == ["/usr/bin/python", "/tmp/candidate/scripts/lens_vectorize_gate.py",
                        "--model"]
    assert argv[argv.index("--batch-invariant") + 1] == "0"
    assert argv[argv.index("--batches") + 1] == "8,32"
    assert argv[argv.index("--cases") + 1] == "case:a,case:b"
    assert "--reduction-mode" not in argv
    assert "SISTEER_LENS_STACKED_REDUCTIONS" not in joined
    # the served geometry rides along exactly like --timing-batches, at the gate's own defaults when unspecified
    assert argv[argv.index("--timing-batches") + 1] == "8,32,64"
    assert argv[argv.index("--max-num-seqs") + 1] == str(gate.DEFAULT_MAX_NUM_SEQS)
    assert argv[argv.index("--max-model-len") + 1] == str(gate.DEFAULT_MAX_MODEL_LEN)
    assert f"{gate.EXTRA_OBJECTS.keys().__iter__().__next__()}=llama8b" in argv
    assert not [x for x in argv if token_hits(x, STORAGE_TOKENS)], argv


def test_lens_modal_explicit_mode_and_env_scrubbing(gate):
    argv = gate.gate_argv(
        "python", "/tmp/c", "/tmp/r", model="qwen35_4b", ref_sha="b" * 40,
        results_root="/tmp/out", cases=None, batches="1,8,32",
        timing_batches="8,32,64,128,256", measure_s=1, legacy_ab=True,
        reduction_mode="per_row", ready_timeout_s=1800, max_num_seqs=256, max_model_len=2048,
    )
    assert argv[argv.index("--reduction-mode") + 1] == "per_row"
    assert "--legacy-ab" in argv
    # a TIMING-ONLY row at 2048 x 256 with the wide ladder: both geometry flags carry the caller's values
    assert argv[argv.index("--timing-batches") + 1] == "8,32,64,128,256"
    assert argv[argv.index("--max-num-seqs") + 1] == "256" and argv[argv.index("--max-model-len") + 1] == "2048"
    base = {key: "inherited" for key in gate.SCRUB_ENV}
    base["KEEP"] = "yes"
    env = gate.gate_env(base, "/tmp/c")
    assert all(key not in env for key in gate.SCRUB_ENV)
    assert env["KEEP"] == "yes"
    assert env["PYTHONPATH"] == "/tmp/c/src:/tmp/c/scripts"
    assert env["HF_HOME"] == "/vol/hf"


def _gate_parser_defaults() -> dict:
    """The reviewed gate's own ``--max-num-seqs`` / ``--max-model-len`` defaults,
    READ from its argparse calls (never retyped)."""
    tree = ast.parse((REPO / "scripts/lens_vectorize_gate.py").read_text())
    out = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant) and node.args[0].value in ("--max-num-seqs", "--max-model-len")):
            for kw in node.keywords:
                if kw.arg == "default":
                    out[node.args[0].value] = ast.literal_eval(kw.value)
    assert set(out) == {"--max-num-seqs", "--max-model-len"}
    return out


def _wrapper_main_defaults() -> dict:
    """The Modal wrapper's ``main()`` parameter defaults, read from its source
    (the local_entrypoint decorator is stubbed in this test, so the AST is the
    authoritative reading either way)."""
    tree = ast.parse((REPO / "scripts/modal/lens_vectorize_gate.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main")
    args = fn.args.args
    defaults = [None] * (len(args) - len(fn.args.defaults)) + list(fn.args.defaults)
    # literal defaults are read; a named default (ref_sha=DEFAULT_REFERENCE_SHA) is kept as its name
    return {a.arg: (ast.literal_eval(d) if isinstance(d, ast.Constant) else (d.id if isinstance(d, ast.Name) else None))
            for a, d in zip(args, defaults)}


def test_lens_modal_geometry_defaults_equal_the_gates_and_are_recorded(gate):
    """The served geometry threads through the wrapper (main -> spawn -> gate
    argv) with defaults IDENTICAL to the gate's own, so the certification run is
    unchanged; a timing-only row can widen it; the state file and the
    ``[lens-gate] app=`` line record what was launched."""
    gd = _gate_parser_defaults()
    wd = _wrapper_main_defaults()
    assert wd["max_num_seqs"] == gd["--max-num-seqs"] == gate.DEFAULT_MAX_NUM_SEQS
    assert wd["max_model_len"] == gd["--max-model-len"] == gate.DEFAULT_MAX_MODEL_LEN
    assert wd["timing_batches"] == "8,32,64" and wd["batches"] == "1,8,32"           # the ladder seams as before
    # the argv without explicit geometry equals the argv with the gate's defaults spelled out
    common = dict(model="gemma4e4b", ref_sha="c" * 40, results_root="/tmp/out", cases=None, batches="1,8,32",
                  timing_batches="8,32,64", measure_s=45.0, legacy_ab=False, reduction_mode=None, ready_timeout_s=2700)
    assert gate.gate_argv("py", "/c", "/r", **common) == gate.gate_argv(
        "py", "/c", "/r", **common, max_num_seqs=gd["--max-num-seqs"], max_model_len=gd["--max-model-len"])
    # every family function and the shared runners accept the geometry with the same defaults
    import inspect
    for fn_name in ("gate_llama8b", "gate_gemma4e4b", "gate_qwen35_4b", "gate_glm46vflash", "gate_nemotron3nano4b",
                    "_run_named_family", "_gate_family"):
        sig = inspect.signature(getattr(gate, fn_name))
        assert sig.parameters["max_num_seqs"].default == gd["--max-num-seqs"], fn_name
        assert sig.parameters["max_model_len"].default == gd["--max-model-len"], fn_name
    # the state file records the geometry and the timing ladder next to the call ids
    st = gate.launch_state(run_name="lens-gate-timing-abc", sha="a" * 40, ref_sha="b" * 40, case_list=["case:x"],
                           batches="1,8,32", timing_batches="8,32,64,128,256", max_num_seqs=256, max_model_len=2048,
                           models=["gemma4e4b", "qwen35_4b"], calls={"gemma4e4b": "fc-1", "qwen35_4b": "fc-2"})
    assert st["max_num_seqs"] == 256 and st["max_model_len"] == 2048 and st["timing_batches"] == "8,32,64,128,256"
    assert st["calls"] == {"gemma4e4b": "fc-1", "qwen35_4b": "fc-2"} and st["app"] == "lens-gate-timing-abc"
    assert json.loads(json.dumps(st)) == st
    # ... and the launch banner names them
    line = gate.banner_line(run_name="lens-gate-timing-abc", app_id="ap-1", sha="a" * 40, ref_sha="b" * 40,
                            models=["gemma4e4b"], case_list=None, batches="1,8,32", timing_batches="8,32,64,128,256",
                            measure_s=45.0, max_num_seqs=256, max_model_len=2048)
    assert line.startswith("[lens-gate] app=lens-gate-timing-abc app_id=ap-1 ")
    assert "timing_batches=8,32,64,128,256" in line and "max_num_seqs=256 max_model_len=2048" in line
    # the spawn passes them positionally after run_name, in main's order
    src = (REPO / "scripts/modal/lens_vectorize_gate.py").read_text()
    assert "int(ready_timeout_s), run_name,\n                            int(max_num_seqs), int(max_model_len))" in src


def test_lens_modal_aggregates_gate_exit_codes(gate):
    assert gate.modal_exit_code([{"gate_rc": 0, "status": "DONE"}]) == 0
    assert gate.modal_exit_code([{"gate_rc": 0, "status": "DONE"},
                                 {"gate_rc": 3, "status": "INCONCLUSIVE"}]) == 3
    assert gate.modal_exit_code([{"gate_rc": 3, "status": "INCONCLUSIVE"},
                                 {"gate_rc": 2, "status": "GATE_FAIL"}]) == 2
    assert gate.modal_exit_code([{"status": "SETUP_FAILED"}]) == 2


def test_modal_volume_client_paths_strip_mount_root(gate, cpu_suite):
    assert gate.volume_client_path("/vol/repos") == "/repos"
    assert gate.volume_client_path("/vol/lens-gate") == "/lens-gate"
    assert cpu_suite.volume_client_path("/vol/repos") == "/repos"
    with pytest.raises(ValueError):
        gate.volume_client_path("/tmp/not-the-volume")


def test_mirror_accepts_volume_paths_without_leading_slash(gate, monkeypatch, tmp_path):
    sha = "3" * 40
    src = f"/lens-gate/{sha}/llama8b"
    entries = [
        types.SimpleNamespace(path=f"lens-gate/{sha}/llama8b/report.md",
                              type=types.SimpleNamespace(name="FILE")),
        types.SimpleNamespace(path=f"/lens-gate/{sha}/llama8b/RESULT",
                              type=types.SimpleNamespace(name="FILE")),
        types.SimpleNamespace(path=f"lens-gate/{sha}/llama8b/standups",
                              type=types.SimpleNamespace(name="DIRECTORY")),
        types.SimpleNamespace(path=f"lens-gate/{sha}/llama8b/server_auth.json",
                              type=types.SimpleNamespace(name="FILE")),
    ]
    blobs = {
        f"{src}/report.md": b"report",
        f"{src}/RESULT": b"exit 3",
    }
    monkeypatch.setattr(gate.VOL, "listdir",
                        lambda path, recursive: entries if path == src and recursive else [],
                        raising=False)
    reads = []
    monkeypatch.setattr(gate, "_vol_read",
                        lambda path: reads.append(path) or blobs[path])

    got = gate.mirror_family(sha, "llama8b", None, tmp_path)

    assert sorted(got["pulled"]) == ["RESULT", "report.md"]
    assert sorted(reads) == sorted(blobs)
    assert (tmp_path / "llama8b" / "report.md").read_bytes() == b"report"
    assert (tmp_path / "llama8b" / "RESULT").read_bytes() == b"exit 3"


def test_preflight_provenance_excludes_adapter_probe(gate):
    cases = {
        "real": {"sources": ["adapter-probe", "/vol/objects/cell/object.pt"]},
        "synthetic": {"sources": ["adapter-probe"]},
        "empty": {"sources": []},
    }
    assert gate.inventory_real_sources(cases) == {
        "real": ["/vol/objects/cell/object.pt"],
    }


def test_lens_modal_results_and_summary_report_real_vs_synthetic(gate):
    cases = ["kind:op:shape:all:skipbos0:both", "multigroup"]
    assert gate.results_rel("llama8b", None) == "llama8b"
    assert gate.results_rel("llama8b", cases).startswith("llama8b/subset/")
    doc = {"exit_code": 3, "complete": False, "cases": [
        {"case_id": "a", "objects": "real(bank: x)", "verdict": "PASS", "checks": {}},
        {"case_id": "b", "objects": "synthetic(kind)", "verdict": "PASS", "checks": {}},
    ]}
    summary = gate.summarize_gate_json(doc)
    assert summary["n_real_objects"] == 1
    assert summary["n_synthetic_objects"] == 1


def test_lens_modal_requires_exact_descriptive_run_name(gate):
    sha = "1" * 40
    gate.app.description = gate.DEFAULT_APP_NAME
    with pytest.raises(SystemExit, match="modal run -n lens-gate-stacked-1111111"):
        gate.main(sha=sha)


def test_cpu_suite_dependency_hashes_are_baked_from_repo(cpu_suite):
    assert cpu_suite.DEPENDENCY_HASHES == {
        name: cpu_suite.sha256_file(REPO / name)
        for name in cpu_suite.DEPENDENCY_FILES
    }
    assert cpu_suite.UV_VERSION


def test_modal_launchers_import_from_flat_container_path(tmp_path):
    for rel in ("scripts/modal/lens_vectorize_gate.py", "scripts/modal/cpu_suite.py"):
        src = REPO / rel
        flat = Path("/tmp") / f"{tmp_path.name}-{src.name}"
        try:
            flat.write_bytes(src.read_bytes())
            mod = _load(f"_flat_{src.stem}", str(flat), stub_modal=True,
                        modal_local=False)
            assert mod.LOCAL_REPO == Path("/repo")
        finally:
            flat.unlink(missing_ok=True)


def test_cpu_suite_runs_full_tests_path_and_scrubs_gpu_env(cpu_suite, tmp_path):
    argv = cpu_suite.pytest_argv("python", "/tmp/junit.xml")
    assert argv == ["python", "-m", "pytest", "tests", "--tb=short", "--continue-on-collection-errors",
                    "--junitxml=/tmp/junit.xml"]
    # spool_root: suite_env mints the throwaway registry spool under it; tmp_path so the test leaves nothing
    env = cpu_suite.suite_env({"CUDA_VISIBLE_DEVICES": "0", "KEEP": "yes"},
                              Path("/tmp/tree"), spool_root=tmp_path)
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert env["PYTHONPATH"] == "/tmp/tree/src"
    assert env["USER"] == cpu_suite.SUITE_USER
    assert env["LOGNAME"] == cpu_suite.SUITE_USER
    assert env["KEEP"] == "yes"
    assert env["WANDB_MODE"] == "disabled"
    # the gate never emits into the volume's live registry spool (2,727 tiny-model objects did, 2026-09-18)
    # the gate spools registry emission into a throwaway dir instead of a global off
    # switch; the switch is popped from the composed env (its own tests pin the spool contract)
    assert "SISTEER_REGISTRY_DISABLED" not in env
    # an inherited container identity never reaches the suite
    env = cpu_suite.suite_env({"USER": "root", "LOGNAME": "root"}, Path("/tmp/tree"), spool_root=tmp_path)
    assert env["USER"] == cpu_suite.SUITE_USER and env["LOGNAME"] == cpu_suite.SUITE_USER


def test_cpu_suite_identity_cannot_false_positive_the_username_leak_tests(cpu_suite):
    """test_launchers_use_model_budget and test_review_provenance treat any
    USER of 3+ characters as a username that must not appear in generated
    text; ``root`` (the 2026-09-04 identity) is prose in the text they scan."""
    user = cpu_suite.SUITE_USER
    assert user != "root" and len(user) > 2
    scanned = (REPO / "src/sisteer/capeval/serving/geometry.py").read_text()
    assert user not in scanned
    assert user not in (REPO / "scripts/geom_sweep_analyze.py").read_text()


def test_cpu_suite_parses_actual_junit_failures_without_allowlist(cpu_suite, tmp_path):
    p = tmp_path / "junit.xml"
    p.write_text('''<?xml version="1.0"?>
<testsuites tests="3" failures="1" errors="1" skipped="0">
  <testsuite name="pytest" tests="3" failures="1" errors="1" skipped="0">
    <testcase classname="tests.test_alpha.TestA" name="test_pass" />
    <testcase classname="tests.test_beta" name="test_fail"><failure message="assert 1 == 2">trace</failure></testcase>
    <testcase classname="tests.test_gamma" name="test_error"><error message="boom">trace</error></testcase>
  </testsuite>
</testsuites>''')
    got = cpu_suite.parse_junit(p)
    assert got["counts"] == {"tests": 3, "failures": 1, "errors": 1, "skipped": 0}
    assert [f["node"] for f in got["failures"]] == [
        "tests.test_beta::test_fail", "tests.test_gamma::test_error",
    ]
    # The launcher's CODE names no historical failure (nothing is forgiven or
    # filtered); its module docstring may DESCRIBE the measured Modal baseline
    # so the next reader does not compare against the Slurm list.
    src = (REPO / "scripts/modal/cpu_suite.py").read_text()
    tree = ast.parse(src)
    doc_node = tree.body[0] if isinstance(tree.body[0], ast.Expr) else None
    assert doc_node is not None and ast.get_docstring(tree)
    code = "\n".join(src.splitlines()[doc_node.end_lineno:])
    for historical in ("test_constitution_si", "test_irrelevant_si",
                       "test_grading_live", "test_stacked_defense",
                       "test_capeval_stage2_common"):
        assert historical not in code
    assert "test_constitution_si" in ast.get_docstring(tree)   # the baseline is documented


def test_cpu_suite_requires_exact_descriptive_run_name(cpu_suite):
    cpu_suite.app.description = cpu_suite.DEFAULT_APP_NAME
    with pytest.raises(SystemExit, match="modal run -n cpu-suite-2222222"):
        cpu_suite.main(sha="2" * 40)


# ------------------------------------------------------------------------------------------ cpu_suite: sharded run
def _fake_tree(tmp_path: Path, sizes: dict) -> Path:
    tree = tmp_path / "tree"
    (tree / "tests").mkdir(parents=True)
    for name, n in sizes.items():
        (tree / "tests" / name).write_text("x" * n)
    (tree / "tests" / "conftest.py").write_text("# not a test file\n")
    (tree / "tests" / "helper_test_data.json").write_text("{}")
    return tree


def test_cpu_suite_shard_files_is_a_deterministic_name_stable_partition(cpu_suite, tmp_path):
    """The shards' union is every tests/test_*.py exactly once (a partition), the dealing is deterministic and
    keyed on NAMES only — a file growing does not move any file between shards, so two commits with the same
    file list compare like-for-like — and nothing that is not a test file rides along (2026-09-16: the gate
    container was preempted four times in a row; shards are the fix; the first size-dealt cut re-shuffled the
    layout on every edit)."""
    sizes = {"test_big.py": 1000, "test_b.py": 300, "test_c.py": 300, "test_d.py": 200, "test_e.py": 100, "test_f.py": 50, "test_g.py": 50}
    tree = _fake_tree(tmp_path, sizes)
    every = {f"tests/{n}" for n in sizes}
    for count in (1, 2, 3, 7, 10):
        dealt = [cpu_suite.shard_files(tree, i, count) for i in range(count)]
        flat = [f for d in dealt for f in d]
        assert sorted(flat) == sorted(every), (count, flat)                  # a cover ...
        assert len(flat) == len(set(flat)), (count, flat)                      # ... with no file twice
        assert dealt == [cpu_suite.shard_files(tree, i, count) for i in range(count)]   # deterministic
        if count <= len(sizes):
            assert all(d for d in dealt), (count, dealt)                       # no empty shard while files remain
    names = sorted(every)
    assert [cpu_suite.shard_files(tree, i, 3) for i in range(3)] == [names[0::3], names[1::3], names[2::3]]
    assert cpu_suite.shard_files(tree, 0, 1) == names
    before = [cpu_suite.shard_files(tree, i, 3) for i in range(3)]
    (tree / "tests" / "test_g.py").write_text("x" * 100_000)                  # a file grows: the layout must not move
    assert [cpu_suite.shard_files(tree, i, 3) for i in range(3)] == before
    with pytest.raises(ValueError):
        cpu_suite.shard_files(tree, 2, 2)
    with pytest.raises(ValueError):
        cpu_suite.shard_files(tree, 0, 0)


def test_cpu_suite_pytest_argv_with_files_runs_exactly_those_files(cpu_suite):
    argv = cpu_suite.pytest_argv("python", "/tmp/j.xml", ["tests/test_b.py", "tests/test_a.py"])
    assert argv == ["python", "-m", "pytest", "tests/test_b.py", "tests/test_a.py", "--tb=short", "--continue-on-collection-errors",
                    "--junitxml=/tmp/j.xml"]
    assert "tests" not in argv                                                 # a shard never widens to the full suite
    assert cpu_suite.pytest_argv("python", "/tmp/j.xml", None)[3] == "tests"
    assert cpu_suite.pytest_argv("python", "/tmp/j.xml", [])[3] == "tests"     # (the runner records an empty shard instead)


_DEFAULT_COUNTS = object()


def _shard_rec(i, n, status="PASS", rc=0, counts=_DEFAULT_COUNTS, failures=(), elapsed=10.0, **extra):
    return {"sha": "a" * 40, "app": "cpu-suite-aaaaaaa", "status": status, "pytest_rc": rc, "elapsed_s": elapsed,
            "command": ["python", "-m", "pytest", f"tests/test_{i}.py"], "tree": {"ok": True}, "dependency_hashes": {"uv.lock": "h"},
            "versions": {"pytest": "8"}, "counts": {"tests": 10, "failures": 0, "errors": 0, "skipped": 1} if counts is _DEFAULT_COUNTS else counts,
            "failures": list(failures), "shard": {"index": i, "count": n, "n_files": 1, "files": [f"tests/test_{i}.py"]},
            "results_dir": f"/vol/cpu-suite/{'a' * 40}/shards/{i}-of-{n}",
            **extra}


def test_cpu_suite_merge_shard_records_sums_counts_unions_failures_and_takes_the_worst_status(cpu_suite):
    sha = "a" * 40
    ok = _shard_rec(0, 3)
    bad = _shard_rec(1, 3, status="FAIL", rc=1, counts={"tests": 20, "failures": 2, "errors": 0, "skipped": 0},
                     failures=[{"node": "tests.test_z::test_q", "kind": "failure", "message": "m"},
                               {"node": "tests.test_a::test_p", "kind": "failure", "message": "m"}], elapsed=40.0)
    also = _shard_rec(2, 3, status="FAIL", rc=1, counts={"tests": 5, "failures": 1, "errors": 0, "skipped": 2},
                      failures=[{"node": "tests.test_m::test_r", "kind": "error", "message": "m"}], elapsed=25.0)
    rec = cpu_suite.merge_shard_records([ok, bad, also], sha, "cpu-suite-aaaaaaa")
    assert rec["status"] == "FAIL" and rec["pytest_rc"] == 1 and rec["shards"] == 3
    assert rec["shard_layout"]["count"] == 3 and rec["shard_layout"]["rule"] == cpu_suite.SHARD_RULE
    assert rec["shard_layout"]["map_sha256"] == cpu_suite.shard_layout([ok, bad, also])["map_sha256"]
    # the layout stamp separates like from unlike: the same files dealt differently is a different map
    swapped = [dict(ok, shard={"index": 0, "count": 3, "n_files": 1, "files": ["tests/test_1.py"]}),
               dict(bad, shard={"index": 1, "count": 3, "n_files": 1, "files": ["tests/test_0.py"]}), also]
    assert cpu_suite.shard_layout(swapped)["map_sha256"] != rec["shard_layout"]["map_sha256"]
    assert cpu_suite.shard_layout([ok, bad, also]) == cpu_suite.shard_layout([also, ok, bad])   # order of gathering is irrelevant
    assert rec["counts"] == {"tests": 35, "failures": 3, "errors": 0, "skipped": 3}
    assert [f["node"] for f in rec["failures"]] == ["tests.test_a::test_p", "tests.test_m::test_r", "tests.test_z::test_q"]
    assert rec["elapsed_s"] == 40.0 and len(rec["shard_records"]) == 3 and rec["shard_records"][1]["status"] == "FAIL"
    assert rec["tree"] == {"ok": True} and rec["dependency_hashes"] == {"uv.lock": "h"}
    # every shard green -> PASS, rc 0
    green = cpu_suite.merge_shard_records([_shard_rec(i, 2) for i in range(2)], sha, "cpu-suite-aaaaaaa")
    assert (green["status"], green["pytest_rc"], green["counts"]["tests"], green["failures"]) == ("PASS", 0, 20, [])
    # one shard timed out -> TIMEOUT, whatever the others did
    to = cpu_suite.merge_shard_records([_shard_rec(0, 2), _shard_rec(1, 2, status="TIMEOUT", rc=-9, counts=None)], sha, "x")
    assert to["status"] == "TIMEOUT" and to["pytest_rc"] == -9 and to["counts"] is None
    # a shard that never ran pytest is the whole record's refusal, never a pass
    ref = cpu_suite.merge_shard_records([_shard_rec(0, 2), {"sha": sha, "status": "DEPENDENCY_MISMATCH", "error": "lock differs"}], sha, "x")
    assert ref["status"] == "DEPENDENCY_MISMATCH" and ref["pytest_rc"] is None and "lock differs" in ref["error"]
    assert ref["shards"] == 2 and ref["shard_layout"]["count"] == 2
