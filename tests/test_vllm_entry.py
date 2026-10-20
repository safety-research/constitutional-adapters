"""``sisteer.capeval.serving.vllm_entry`` — the clamp-override wrapper entrypoint,
on a FAKE ``vllm`` (no wheel on the grading host): the flag
is parsed off the argv front, the wheel's pinned ``_MAX_TEMP`` is asserted, the
global is set, the fork method is asserted and exported, ``sys.argv`` is rebuilt as
``["vllm", "serve", …]`` and the stock ``main`` is called exactly once; every
refusal names its reason and starts no server. Plus the launch composer's half
(``vllm_server.apply_sampling_clamp_override`` / ``build_server_launch``'s kwarg):
the wrapper argv iff the override is asked, byte-identical otherwise.
"""
from __future__ import annotations

import io
import os
import sys
import types
from pathlib import Path

import pytest

from sisteer.capeval.serving import vllm_entry as VE
from sisteer.capeval.serving import vllm_server as vs
from sisteer.capeval.tinycap import contracts as C


@pytest.fixture()
def fake_vllm(monkeypatch):
    """A fake ``vllm`` package: ``sampling_params._MAX_TEMP`` at the wheel's value
    and a recording ``entrypoints.cli.main.main``."""
    vllm = types.ModuleType("vllm")
    sp = types.ModuleType("vllm.sampling_params")
    sp._MAX_TEMP = 1e-2
    ep = types.ModuleType("vllm.entrypoints")
    cli = types.ModuleType("vllm.entrypoints.cli")
    cm = types.ModuleType("vllm.entrypoints.cli.main")
    calls = []

    def main():
        calls.append(list(sys.argv))
        return 0

    cm.main = main
    # the process-creation seam (0.25.1: vllm.utils.system_utils.get_mp_context) — a FORK context
    # by default; tests flip `ctx_method` to exercise the guard
    utils = types.ModuleType("vllm.utils")
    su = types.ModuleType("vllm.utils.system_utils")
    state = types.SimpleNamespace(ctx_method="fork", ctx_calls=[])

    def get_mp_context():
        state.ctx_calls.append(state.ctx_method)
        return types.SimpleNamespace(get_start_method=lambda: state.ctx_method)

    su.get_mp_context = get_mp_context
    for name, m in (("vllm", vllm), ("vllm.sampling_params", sp), ("vllm.entrypoints", ep),
                    ("vllm.entrypoints.cli", cli), ("vllm.entrypoints.cli.main", cm),
                    ("vllm.utils", utils), ("vllm.utils.system_utils", su)):
        monkeypatch.setitem(sys.modules, name, m)
    vllm.sampling_params = sp
    saved = list(sys.argv)
    yield types.SimpleNamespace(sp=sp, calls=calls, su=su, utils=utils, state=state)
    sys.argv = saved


def test_the_wrapper_lifts_the_pinned_clamp_and_calls_the_stock_cli_once(fake_vllm):
    out = io.StringIO()
    env: dict = {}
    rc = VE.main(["--sisteer-max-temp", "0.001", "serve", "m", "--port", "1"], environ=env, stdout=out)
    assert rc == 0
    assert fake_vllm.calls == [["vllm", "serve", "m", "--port", "1"]]          # the SAME tail, once
    assert fake_vllm.sp._MAX_TEMP == 0.001 == C.BFCL_PROTOCOL_TEMPERATURE == VE.PROTOCOL_MAX_TEMP
    assert env[VE.MULTIPROC_ENV] == "fork"                                        # the default, made explicit
    lines = out.getvalue().strip().splitlines()
    assert lines[0] == "[sisteer] vllm _MAX_TEMP 0.01 -> 0.001 (BFCL protocol, D13)"
    assert lines[1] == ("[sisteer] vllm get_mp_context guarded (fork) in ['vllm.utils.system_utils']; override mechanism: "
                        + VE.MECHANISM_IN_PROCESS)
    assert VE.PINNED_MAX_TEMP == C.VLLM_MAX_TEMP_PINNED == 0.01
    assert VE.FLAG_MAX_TEMP == C.FLAG_VLLM_ENTRY_MAX_TEMP == "--sisteer-max-temp"
    # a LEAF: stdlib + vllm only, never sisteer (it runs in the serving venv, from the staged tree)
    src = Path(VE.__file__).read_text()
    assert "import sisteer" not in src and "from sisteer" not in src


@pytest.mark.parametrize("argv,why", [
    (["serve", "m"], "must start with"),
    (["--sisteer-max-temp", "x", "serve"], "not a float"),
    (["--sisteer-max-temp", "0.02", "serve"], "must lie in"),
    (["--sisteer-max-temp", "0", "serve"], "must lie in"),
    (["--sisteer-max-temp", "0.005", "serve"], "above the protocol value"),
    (["--sisteer-max-temp", "0.001", "bench"], "must start with `serve`"),
    (["--sisteer-max-temp", "0.001"], "must start with `serve`"),
])
def test_a_malformed_argv_refuses_before_any_import(fake_vllm, argv, why):
    with pytest.raises(VE.ClampOverrideRefused, match=why):
        VE.main(argv, environ={}, stdout=io.StringIO())
    assert fake_vllm.calls == [] and fake_vllm.sp._MAX_TEMP == 1e-2


def test_a_spawn_environment_refuses_and_fork_is_accepted(fake_vllm):
    with pytest.raises(VE.ClampOverrideRefused, match="spawn"):
        VE.main(["--sisteer-max-temp", "0.001", "serve", "m"], environ={VE.MULTIPROC_ENV: "spawn"},
                stdout=io.StringIO())
    assert fake_vllm.calls == [] and fake_vllm.sp._MAX_TEMP == 1e-2
    env = {VE.MULTIPROC_ENV: "fork"}
    VE.main(["--sisteer-max-temp", "0.001", "serve", "m"], environ=env, stdout=io.StringIO())
    assert env[VE.MULTIPROC_ENV] == "fork" and len(fake_vllm.calls) == 1


def test_another_wheels_clamp_refuses(fake_vllm):
    """The wheel's value is asserted BEFORE it is changed: 0.02 (or an already
    lifted 0.001) is not the 0.25.1 wheel — a different instrument."""
    for have in (0.02, 0.001, None):
        if have is None:
            del fake_vllm.sp._MAX_TEMP
        else:
            fake_vllm.sp._MAX_TEMP = have
        with pytest.raises(VE.ClampOverrideRefused, match="not the pinned"):
            VE.main(["--sisteer-max-temp", "0.001", "serve", "m"], environ={}, stdout=io.StringIO())
        assert fake_vllm.calls == []


def test_the_pure_helpers(fake_vllm):
    assert VE.parse_argv(["--sisteer-max-temp", "0.001", "serve", "m"]) == (0.001, ["serve", "m"])
    rec = VE.apply_override(fake_vllm.sp, 0.001)
    assert rec == {"from": 0.01, "to": 0.001} and fake_vllm.sp._MAX_TEMP == 0.001
    assert VE.assert_fork({}) == "fork"


# --------------------------------------------------------------------------- #
# the topologies the fork assertion does not cover are REFUSED,                #
# and the start method is asserted at process creation                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tail,why", [
    (["serve", "m", "--api-server-count", "2"], "api-server-count 2"),
    (["serve", "m", "--api-server-count=4"], "api-server-count 4"),
    (["serve", "m", "--api-server-count", "x"], "not an int"),
    (["serve", "m", "--numa-bind"], "numa-bind"),
    (["serve", "m", "--distributed-executor-backend", "ray"], "launched by ray"),
    (["serve", "m", "--distributed-executor-backend=external_launcher"], "external_launcher"),
    (["serve", "m", "--data-parallel-backend", "ray"], "data-parallel-backend ray"),
])
def test_a_spawn_forcing_topology_refuses_at_parse_and_at_compose(fake_vllm, serving_venv, tail, why):
    """The pinned wheel spawns frontends above ``--api-server-count 1`` and forces
    spawn under ``--numa-bind`` AFTER the env assertion; Ray-launched workers are
    never forked. Each is refused by the entrypoint (no server) AND by the
    launch composer (no argv)."""
    assert why.split()[0] in (VE.spawn_topology_reason(tail) or "")
    with pytest.raises(VE.ClampOverrideRefused, match=why.split()[0]):
        VE.main(["--sisteer-max-temp", "0.001", *tail], environ={}, stdout=io.StringIO())
    assert fake_vllm.calls == [] and fake_vllm.sp._MAX_TEMP == 1e-2
    with pytest.raises(vs.VllmServerError, match="launch topology"):
        vs.apply_sampling_clamp_override(["vllm", *tail], {}, 0.001)
    # the single-frontend forked topology is what the override is certified for
    assert VE.spawn_topology_reason(["serve", "m", "--api-server-count", "1", "--distributed-executor-backend", "mp"]) is None
    assert VE.parse_argv(["--sisteer-max-temp", "0.001", "serve", "m", "--api-server-count=1"])[1] == ["serve", "m", "--api-server-count=1"]


def test_the_start_method_is_asserted_at_process_creation(fake_vllm):
    """``get_mp_context`` is wrapped in the wheel's module: a fork context passes
    through; a spawn context, or an environment flipped to spawn AFTER the
    assertion (``_maybe_force_spawn``), refuses before any child exists."""
    env: dict = {}
    VE.main(["--sisteer-max-temp", "0.001", "serve", "m"], environ=env, stdout=io.StringIO())
    guarded = fake_vllm.su.get_mp_context
    assert getattr(guarded, "__sisteer_fork_guard__", False) and not hasattr(fake_vllm.utils, "get_mp_context")
    ctx = guarded()
    assert ctx.get_start_method() == "fork" and fake_vllm.state.ctx_calls == ["fork"]
    fake_vllm.state.ctx_method = "spawn"                       # the wheel hands out a spawn context
    with pytest.raises(VE.ClampOverrideRefused, match="'spawn' multiprocessing context"):
        guarded()
    fake_vllm.state.ctx_method = "fork"
    env[VE.MULTIPROC_ENV] = "spawn"                            # _maybe_force_spawn flipped the env behind the assertion
    with pytest.raises(VE.ClampOverrideRefused, match="at process creation"):
        guarded()
    # idempotent (a second wrap never stacks), and both known modules are patched when both define it
    fake_vllm.utils.get_mp_context = lambda: types.SimpleNamespace(get_start_method=lambda: "fork")
    assert VE.guard_mp_context({"vllm.utils.system_utils": fake_vllm.su, "vllm.utils": fake_vllm.utils}, {VE.MULTIPROC_ENV: "fork"}) == ["vllm.utils"]
    assert fake_vllm.su.get_mp_context is guarded
    # a wheel exposing the seam in neither module refuses: the guard MUST reach process creation
    with pytest.raises(VE.ClampOverrideRefused, match="cannot reach vLLM's process creation"):
        VE.guard_mp_context({"vllm.utils.system_utils": types.ModuleType("x")}, {VE.MULTIPROC_ENV: "fork"})
    del fake_vllm.su.get_mp_context
    del fake_vllm.utils.get_mp_context                         # (guarded above — a guarded seam in EITHER module counts as reached)
    fake_vllm.sp._MAX_TEMP = 1e-2                              # the wheel as shipped again
    with pytest.raises(VE.ClampOverrideRefused, match="cannot reach"):
        VE.main(["--sisteer-max-temp", "0.001", "serve", "m"], environ={}, stdout=io.StringIO())
    assert len(fake_vllm.calls) == 1                           # the stock CLI ran once (the first call), never here


# --------------------------------------------------------------------------- #
# the composer's half: vllm_server.apply_sampling_clamp_override                #
# --------------------------------------------------------------------------- #


@pytest.fixture()
def serving_venv(tmp_path, monkeypatch):
    """A fake serving venv: a ``vllm`` console script with a sibling ``python`` and a site-packages
    the composer drops the shim's ``.pth`` into (asked of the serving python live; here a tmp dir)."""
    (tmp_path / "vllm").write_text("#!/bin/sh\n")
    (tmp_path / "python").write_text("")
    (tmp_path / "site-packages").mkdir()
    monkeypatch.setattr(vs.shutil, "which", lambda name, *a, **k: str(tmp_path / "vllm") if name == "vllm" else None)
    monkeypatch.setattr(vs, "serving_purelib", lambda python: tmp_path / "site-packages")
    return tmp_path


def test_apply_sampling_clamp_override_wraps_the_argv_and_leads_pythonpath_with_the_live_src(serving_venv, tmp_path):
    import sisteer
    src = str(Path(sisteer.__file__).resolve().parents[1])
    shim = str(VE.shim_dir())
    cmd, env, rec = vs.apply_sampling_clamp_override(["vllm", "serve", "m", "--x", "1"], {"PYTHONPATH": "/other"}, 0.001)
    assert cmd == [str(serving_venv / "python"), "-m", C.VLLM_ENTRY_MODULE, C.FLAG_VLLM_ENTRY_MAX_TEMP, "0.001",
                   "serve", "m", "--x", "1"]
    # round 6 / round 9: src FIRST and the override value in the environment; exactly ONE delivery route — the .pth
    # (preferred) leaves the shim dir OFF PYTHONPATH (both routes at once exec the shim twice and their finders recursed)
    assert env["PYTHONPATH"] == os.pathsep.join([src, "/other"]) and env[VE.SHIM_ENV] == "0.001"
    assert env[VE.SHIM_ROUTE_ENV] == VE.ROUTE_PTH == "pth" and "SISTEER_VLLM_SHIM_PTH_REFUSED" not in env
    # round 8: the .pth boot dropped into the SERVING interpreter's site-packages, named in the env; idempotent
    pth = Path(env[VE.SHIM_PTH_ENV])
    assert pth == serving_venv / "site-packages" / VE.SHIM_PTH_NAME and pth.is_file()
    assert pth.read_text() == VE.pth_line(Path(shim) / VE.SHIM_FILENAME) + "\n" and pth.read_text().startswith("import ")
    assert "\n" not in pth.read_text().rstrip("\n") and VE.pth_target(pth) == str(Path(shim) / VE.SHIM_FILENAME)
    before = pth.stat().st_mtime_ns
    vs.apply_sampling_clamp_override(["vllm", "serve", "m"], {}, 0.001)
    assert pth.stat().st_mtime_ns == before                                    # the same line is left as is
    assert (VE.shim_dir() / VE.SHIM_FILENAME).is_file() and VE.shim_dir() == Path(src) / "sisteer/capeval/serving/vllm_shim"
    assert rec == dict(C.SAMPLING_CLAMP_OVERRIDE) == {"from": 0.01, "to": 0.001, "why": "BFCL protocol (D13)"}
    # already present but not first: moved to the front; an inherited shim dir is DROPPED (the .pth is the route)
    _, env2, _ = vs.apply_sampling_clamp_override(["vllm", "serve", "m"], {"PYTHONPATH": "/a:" + src + ":" + shim}, 0.001)
    assert env2["PYTHONPATH"] == os.pathsep.join([src, "/a"])
    _, env3, _ = vs.apply_sampling_clamp_override(["vllm", "serve", "m"], {}, 0.001)
    assert env3["PYTHONPATH"] == src
    # an inherited shim dir of ANOTHER copy (the staged tree's, round 8's live topology) is dropped as well — by signature
    staged = tmp_path / "staged" / "vllm_shim"; staged.mkdir(parents=True)
    (staged / VE.SHIM_FILENAME).write_text((Path(shim) / VE.SHIM_FILENAME).read_text())
    plain = tmp_path / "plain"; plain.mkdir(); (plain / VE.SHIM_FILENAME).write_text("# not the shim\n")
    _, env5, _ = vs.apply_sampling_clamp_override(["vllm", "serve", "m"], {"PYTHONPATH": f"{staged}:{plain}"}, 0.001)
    assert env5["PYTHONPATH"] == os.pathsep.join([src, str(plain)]) and env5[VE.SHIM_ROUTE_ENV] == VE.ROUTE_PTH
    assert vs.clamp_override_mechanism(env) == "in-process (vllm_entry) + sitecustomize shim (SISTEER_VLLM_MAX_TEMP) via pth"
    # the FALLBACK route, only when the .pth cannot be written: the shim dir on PYTHONPATH, the refusal recorded
    import unittest.mock as _mock
    with _mock.patch.object(vs, "drop_shim_pth", side_effect=vs.VllmServerError("read-only site-packages")):
        _, env4, _ = vs.apply_sampling_clamp_override(["vllm", "serve", "m"], {"PYTHONPATH": "/other"}, 0.001)
    assert env4["PYTHONPATH"] == os.pathsep.join([src, shim, "/other"]) and VE.SHIM_PTH_ENV not in env4
    assert env4[VE.SHIM_ROUTE_ENV] == VE.ROUTE_PYTHONPATH and env4["SISTEER_VLLM_SHIM_PTH_REFUSED"].startswith("read-only")
    assert vs.clamp_override_mechanism(env4).endswith("via pythonpath-sitecustomize")


@pytest.mark.parametrize("bad", [0.01, 0.0, 0.002, 0.0005, -1, True, "0.001"])
def test_only_the_pinned_override_value_is_composed(serving_venv, bad):
    """The record is hashed identity: any value but the pinned 0.001 refuses."""
    with pytest.raises(vs.VllmServerError):
        vs.apply_sampling_clamp_override(["vllm", "serve", "m"], {}, bad)


def test_the_override_needs_a_vllm_serve_argv_and_a_serving_interpreter(serving_venv, monkeypatch):
    with pytest.raises(vs.VllmServerError, match="vllm serve"):
        vs.apply_sampling_clamp_override(["python", "-m", "x", "serve"], {}, 0.001)
    monkeypatch.setattr(vs.shutil, "which", lambda *a, **k: None)
    with pytest.raises(vs.VllmServerError, match="not on PATH"):
        vs.apply_sampling_clamp_override(["vllm", "serve", "m"], {}, 0.001)


def test_build_server_launch_composes_the_wrapper_iff_asked_and_is_byte_identical_otherwise(serving_venv, monkeypatch):
    """The card launch's argv and env are EXACTLY what a compose without the kwarg
    gives; the protocol launch swaps only the head and records the override."""
    import inspect as _inspect

    sig = _inspect.signature(vs.build_server_launch)
    assert "sampling_clamp_override" in sig.parameters and sig.parameters["sampling_clamp_override"].default is None
    src = Path(vs.__file__).read_text()
    body = src[src.index("def build_server_launch("):src.index("def _canonical_option_name(")]
    # a self-contained APPENDED step over the finished argv (the later merge of main is mechanical)
    assert body.index("api_key = secrets.token_urlsafe(32)") < body.index("cmd, env, clamp_record = apply_sampling_clamp_override(")
    assert "sampling_clamp_override=clamp_record" in body
    assert "sampling_clamp_override_mechanism=(clamp_override_mechanism(env) if clamp_record is not None else None)" in body
    # the ServerLaunch record field defaults to None (every hand-built launch unchanged)
    import dataclasses
    f = {x.name: x for x in dataclasses.fields(vs.ServerLaunch)}["sampling_clamp_override"]
    assert f.default is None
    assert {x.name: x for x in dataclasses.fields(vs.ServerLaunch)}["sampling_clamp_override_mechanism"].default is None


# --------------------------------------------------------------------------- #
# the spawn-safe SHIM — vLLM forces `spawn` for the Qwen3.5 server            #
# --------------------------------------------------------------------------- #


def _fake_wheel(tmp_path: Path) -> Path:
    """A fake `vllm` package with the wheel's clamp semantics, for a FRESH interpreter to import."""
    pkg = tmp_path / "wheel" / "vllm"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "sampling_params.py").write_text(
        "_MAX_TEMP = 1e-2\n"
        "class SamplingParams:\n"
        "    def __init__(self, temperature=1.0):\n"
        "        self.temperature = float(temperature)\n"
        "        if 0.0 < self.temperature < _MAX_TEMP:\n"
        "            self.temperature = _MAX_TEMP\n")
    return pkg.parent


def test_the_shim_patches_the_clamp_in_a_real_spawned_child_only_when_armed(tmp_path, monkeypatch):
    """The live failure: vLLM forced `spawn` for qwen35_4b ("CUDA is initialized") and the fork-only
    guard refused the standup. A spawned child is a fresh interpreter; the shim (`vllm_shim/
    sitecustomize.py`, on PYTHONPATH, armed by SISTEER_VLLM_MAX_TEMP) patches `_MAX_TEMP` when that
    child imports `vllm.sampling_params`. Real `multiprocessing.get_context("spawn")` children,
    against a fake wheel on the child's PYTHONPATH; unarmed = the wheel's clamp, untouched."""
    wheel = _fake_wheel(tmp_path)
    src = str(Path(VE.__file__).resolve().parents[3])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(wheel), src]))         # the child's sitecustomize comes off PYTHONPATH …
    monkeypatch.syspath_prepend(str(wheel))                                       # … its sys.path is the PARENT's (multiprocessing hands
    monkeypatch.delitem(sys.modules, "vllm", raising=False)                       # it down); the real server has vllm on sys.path
    monkeypatch.delitem(sys.modules, "vllm.sampling_params", raising=False)
    monkeypatch.delenv(VE.SHIM_ENV, raising=False)
    armed = VE.spawned_child_control(armed=True, timeout_s=120)
    assert armed["ok"] is True and armed["armed"] is True, armed
    assert armed["temperature"] == 0.001 and armed["max_temp"] == 0.001 and armed["shim_from"] == 0.01 and armed["shim_applied"] == 0.001
    assert Path(armed["sitecustomize"]).resolve() == (VE.shim_dir() / VE.SHIM_FILENAME).resolve()   # OUR sitecustomize ran in the child
    unarmed = VE.spawned_child_control(armed=False, timeout_s=120)
    assert unarmed["ok"] is True and unarmed["armed"] is False, unarmed
    assert unarmed["temperature"] == 0.01 and unarmed["max_temp"] == 0.01 and unarmed["shim_applied"] is None   # the wheel's clamp
    # the parent's environment is restored either way
    assert VE.SHIM_ENV not in os.environ and os.environ["PYTHONPATH"] == os.pathsep.join([str(wheel), src])
    # the shim's own API, in-process: unarmed installs nothing; armed installs the finder once and patches an
    # already-imported module too (recording the wheel's value)
    import importlib.util
    spec = importlib.util.spec_from_file_location("sitecustomize_under_test", VE.shim_dir() / VE.SHIM_FILENAME)
    shim = importlib.util.module_from_spec(spec)
    monkeypatch.delenv(VE.SHIM_ENV, raising=False)
    spec.loader.exec_module(shim)
    assert shim.ARMED is False and not any(isinstance(f, shim._ClampShimFinder) for f in sys.meta_path)
    fake = types.ModuleType("vllm.sampling_params")
    fake._MAX_TEMP = 1e-2
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", fake)
    assert shim.install({VE.SHIM_ENV: "0.001"}) is True and fake._MAX_TEMP == 0.001 and fake._SISTEER_MAX_TEMP_FROM == 0.01
    assert shim.install({VE.SHIM_ENV: "0.001"}) is True and sum(shim.is_clamp_finder(f) for f in sys.meta_path) == 1
    assert shim.install({VE.SHIM_ENV: "not-a-float"}) is False and shim.install({}) is False
    sys.meta_path[:] = [f for f in sys.meta_path if not shim.is_clamp_finder(f)]
    # a stdlib-only module (it runs where sisteer is never importable)
    src_text = (VE.shim_dir() / VE.SHIM_FILENAME).read_text()
    assert "sisteer" not in [ln.split()[1].split(".")[0] for ln in src_text.splitlines() if ln.startswith(("import ", "from "))]


def test_the_guard_permits_spawn_only_with_the_shim_armed_and_importable(fake_vllm, monkeypatch):
    """Two faces: the shim armed (env) AND importable by a child (its directory on PYTHONPATH, no
    other sitecustomize shadowing it) -> a spawn context passes the guard and `assert_fork` leaves
    the method alone; anything less -> refused as before."""
    shim = str(VE.shim_dir())
    monkeypatch.delitem(sys.modules, "sitecustomize", raising=False)
    # round 8: importability is what a CHILD of this interpreter would import (`find_spec("sitecustomize")` on this
    # process's sys.path — a child inherits it); in the server the PYTHONPATH entries ARE on sys.path
    monkeypatch.syspath_prepend(shim)
    # (1) armed + importable: spawn permitted, the env left as vLLM wants it, the mechanism says shim
    env = {VE.SHIM_ENV: "0.001", "PYTHONPATH": shim + os.pathsep + "/x", VE.MULTIPROC_ENV: "spawn"}
    assert VE.shim_armed(env, 0.001) and VE.shim_importable(env) == str((VE.shim_dir() / VE.SHIM_FILENAME).resolve())
    assert VE.spawn_permitted(env, 0.001) and VE.assert_fork(env, 0.001) == "spawn" and env[VE.MULTIPROC_ENV] == "spawn"
    assert VE.override_mechanism(env, 0.001) == f"{VE.MECHANISM_SHIM} [route: {VE.ROUTE_PYTHONPATH}]"
    assert VE.override_mechanism({**env, VE.SHIM_ROUTE_ENV: "pth"}, 0.001).endswith("[route: pth]")
    VE.main(["--sisteer-max-temp", "0.001", "serve", "m"], environ=env, stdout=io.StringIO())
    fake_vllm.state.ctx_method = "spawn"
    assert fake_vllm.su.get_mp_context().get_start_method() == "spawn"          # permitted at process creation
    # (2) armed but NOT importable (the shim dir off the child's path): refused, the diagnostics in the message
    sys.path.remove(shim)
    env2 = {VE.SHIM_ENV: "0.001", "PYTHONPATH": "/x", VE.MULTIPROC_ENV: "spawn"}
    assert VE.shim_importable(env2) is None and not VE.spawn_permitted(env2)
    with pytest.raises(VE.ClampOverrideRefused, match="diagnostics: .*armed_env.*find_spec_sitecustomize"):
        VE.assert_fork(env2, 0.001)
    monkeypatch.syspath_prepend(shim)
    # (3) importable but not armed: refused; a wrong value: refused
    env3 = {"PYTHONPATH": shim, VE.MULTIPROC_ENV: "spawn"}
    assert VE.shim_importable(env3) and not VE.spawn_permitted(env3)
    with pytest.raises(VE.ClampOverrideRefused):
        VE.assert_fork(env3, 0.001)
    assert not VE.spawn_permitted({VE.SHIM_ENV: "0.005", "PYTHONPATH": shim}, 0.001)
    # (4) another sitecustomize loaded in THIS process shadows the shim: not importable
    other = types.ModuleType("sitecustomize")
    other.__file__ = "/usr/lib/python3/dist-packages/sitecustomize.py"
    monkeypatch.setitem(sys.modules, "sitecustomize", other)
    assert VE.shim_importable(env) is None
    monkeypatch.delitem(sys.modules, "sitecustomize")
    # (5) fork stays the default without the shim: exported, mechanism in-process
    env5: dict = {}
    assert VE.assert_fork(env5) == "fork" and env5[VE.MULTIPROC_ENV] == "fork" and VE.override_mechanism(env5) == VE.MECHANISM_IN_PROCESS
    # the guard at process creation under the shim: a spawn context passes (the env flipped to spawn by vLLM too)
    fake_vllm.sp._MAX_TEMP = 1e-2
    env6 = {VE.SHIM_ENV: "0.001", "PYTHONPATH": shim}
    out = io.StringIO()
    VE.main(["--sisteer-max-temp", "0.001", "serve", "m"], environ=env6, stdout=out)
    env6[VE.MULTIPROC_ENV] = "spawn"                                            # _maybe_force_spawn flips it
    fake_vllm.state.ctx_method = "spawn"
    assert fake_vllm.su.get_mp_context().get_start_method() == "spawn"
    assert "override mechanism: in-process (vllm_entry.apply_override) + sitecustomize shim" in out.getvalue()
    # apply_override accepts an override the SHIM already applied in this interpreter (the marker names the wheel)
    fake_vllm.sp._MAX_TEMP = 0.001
    fake_vllm.sp._SISTEER_MAX_TEMP_FROM = 1e-2
    fake_vllm.sp._SISTEER_MAX_TEMP_SHIM = 0.001
    assert VE.apply_override(fake_vllm.sp, 0.001) == {"from": 0.01, "to": 0.001}
    fake_vllm.sp._SISTEER_MAX_TEMP_FROM = 0.02                                   # the shim patched another wheel
    with pytest.raises(VE.ClampOverrideRefused, match="two overrides disagree, or the wheel"):
        VE.apply_override(fake_vllm.sp, 0.001)


def _decoy_sitecustomize(tmp_path: Path) -> Path:
    d = tmp_path / "decoy"
    d.mkdir()
    (d / "sitecustomize.py").write_text("DECOY = True\n")
    return d


def test_the_shim_is_recognised_by_content_not_path_and_a_decoy_sitecustomize_is_named(tmp_path, monkeypatch):
    """Round 8, the live false negative: the GPU container's server imported `sisteer` from the /root
    mount (cwd precedes PYTHONPATH under `python -m`) while PYTHONPATH carried the STAGED tree's shim
    dir — the staged COPY loaded as sitecustomize and a path-equality check refused it. A copy is the
    shim by its signature line, wherever it lives; a decoy earlier on the path is named in the
    diagnostics and does not pass route (b); the .pth route (a) passes regardless of the decoy."""
    import importlib.util
    import shutil as _sh
    import sysconfig
    staged = tmp_path / "staged" / "vllm_shim"
    staged.mkdir(parents=True)
    _sh.copy(VE.shim_dir() / VE.SHIM_FILENAME, staged / VE.SHIM_FILENAME)
    assert VE.is_shim_file(staged / VE.SHIM_FILENAME) and not VE.is_shim_file(tmp_path / "nope.py")
    # (b) another copy on PYTHONPATH, imported as this process's sitecustomize: OURS -> importable
    spec = importlib.util.spec_from_file_location("sitecustomize", staged / VE.SHIM_FILENAME)
    other_copy = importlib.util.module_from_spec(spec)
    monkeypatch.delenv(VE.SHIM_ENV, raising=False)
    spec.loader.exec_module(other_copy)                                        # unarmed here: installs nothing
    monkeypatch.setitem(sys.modules, "sitecustomize", other_copy)
    monkeypatch.syspath_prepend(str(staged))
    env = {VE.SHIM_ENV: "0.001", "PYTHONPATH": str(staged)}
    got = VE.shim_importable(env)
    assert got and Path(got).resolve() == (staged / VE.SHIM_FILENAME).resolve() and VE.spawn_permitted(env, 0.001)
    d = VE.shim_diagnostics(env)
    assert d["loaded_sitecustomize"]["ours"] and d["find_spec_sitecustomize"]["ours"] and d["armed_env"] == "0.001"
    assert d["pythonpath_sitecustomize"] == [{"entry": str(staged), "ours": True}] and d["pth"]["present"] is False
    # round 9: the diagnostics name the route (inferred here — the composer's SISTEER_VLLM_SHIM_ROUTE wins when set)
    # and count the clamp finders this interpreter carries (the unarmed copy above installed none)
    assert d["route"] == VE.ROUTE_PYTHONPATH and d["clamp_finders_on_meta_path"] == 0
    assert VE.shim_diagnostics({**env, VE.SHIM_ROUTE_ENV: VE.ROUTE_PTH})["route"] == VE.ROUTE_PTH
    # a DECOY imported as sitecustomize (a system / venv layer's): route (b) refused, the decoy named
    decoy = _decoy_sitecustomize(tmp_path)
    dspec = importlib.util.spec_from_file_location("sitecustomize", decoy / "sitecustomize.py")
    dmod = importlib.util.module_from_spec(dspec)
    dspec.loader.exec_module(dmod)
    monkeypatch.setitem(sys.modules, "sitecustomize", dmod)
    sys.path.remove(str(staged))
    monkeypatch.syspath_prepend(str(decoy))
    env2 = {VE.SHIM_ENV: "0.001", "PYTHONPATH": str(decoy) + os.pathsep + str(staged)}
    assert VE.shim_importable(env2) is None
    d2 = VE.shim_diagnostics(env2)
    assert d2["loaded_sitecustomize"] == {"file": str(decoy / "sitecustomize.py"), "ours": False}
    assert d2["find_spec_sitecustomize"]["ours"] is False and [e["ours"] for e in d2["pythonpath_sitecustomize"]] == [False, True]
    with pytest.raises(VE.ClampOverrideRefused, match="diagnostics: .*loaded_sitecustomize.*ours.*false"):
        VE.assert_fork({**env2, VE.MULTIPROC_ENV: "spawn"}, 0.001)
    # (a) the .pth route: in THIS interpreter's site-packages, its target a shim copy -> importable despite the decoy
    purelib = Path(sysconfig.get_paths()["purelib"])
    pth = purelib / VE.SHIM_PTH_NAME
    assert not pth.exists(), "a leftover .pth in the test interpreter's site-packages — remove it"
    try:
        pth.write_text(VE.pth_line(staged / VE.SHIM_FILENAME) + "\n")
        env3 = {**env2, VE.SHIM_PTH_ENV: str(pth)}
        assert VE.shim_importable(env3) == str(pth) and VE.spawn_permitted(env3, 0.001)
        d3 = VE.shim_diagnostics(env3)
        assert d3["pth"] == {"path": str(pth), "present": True, "in_this_interpreters_site_packages": True, "purelib": str(purelib),
                             "target": str(staged / VE.SHIM_FILENAME), "target_ours": True}
        assert d3["route"] == VE.ROUTE_PTH                                    # inferred from what shim_importable found
        # a .pth OUTSIDE this interpreter's site-packages, or pointing at a non-shim, does not count
        stray = tmp_path / VE.SHIM_PTH_NAME
        stray.write_text(VE.pth_line(staged / VE.SHIM_FILENAME) + "\n")
        assert VE.shim_importable({**env2, VE.SHIM_PTH_ENV: str(stray)}) is None
        pth.write_text(VE.pth_line(decoy / "sitecustomize.py") + "\n")
        assert VE.shim_importable(env3) is None
    finally:
        pth.unlink(missing_ok=True)


def test_the_pth_boot_patches_a_real_child_despite_a_decoy_sitecustomize_first_on_its_path(tmp_path, monkeypatch):
    """A REAL fresh interpreter (the system python3 with a private user site — venvs disable the
    user site) whose sys.path holds a DECOY sitecustomize FIRST and our .pth in a site dir: `site`
    imports the decoy as `sitecustomize` AND executes the .pth's boot line, so the fake wheel's
    clamp is patched when armed and untouched when not."""
    import shutil as _sh
    import subprocess as _sp
    py = "/usr/bin/python3" if Path("/usr/bin/python3").is_file() else _sh.which("python3")
    if not py:
        pytest.skip("no system python3")
    userbase = tmp_path / "userbase"
    r = _sp.run([py, "-c", "import site; print(site.getusersitepackages()); print(site.ENABLE_USER_SITE)"],
                env={"PYTHONUSERBASE": str(userbase), "PATH": os.environ.get("PATH", "")}, text=True, capture_output=True)
    user_site, enabled = r.stdout.strip().splitlines()[:2]
    if enabled != "True":
        pytest.skip(f"{py} has the user site disabled")
    Path(user_site).mkdir(parents=True)
    (Path(user_site) / VE.SHIM_PTH_NAME).write_text(VE.pth_line(VE.shim_dir() / VE.SHIM_FILENAME) + "\n")
    wheel = _fake_wheel(tmp_path)
    decoy = _decoy_sitecustomize(tmp_path)
    probe = ("import sys, sitecustomize, vllm.sampling_params as sp; "
             "print(sitecustomize.__file__); print(getattr(sitecustomize, 'DECOY', None)); "
             "print(sp._MAX_TEMP); print(sp.SamplingParams(temperature=0.001).temperature); "
             "print(getattr(sp, '_SISTEER_MAX_TEMP_SHIM', None), getattr(sp, '_SISTEER_MAX_TEMP_FROM', None))")
    base_env = {"PYTHONUSERBASE": str(userbase), "PYTHONPATH": os.pathsep.join([str(decoy), str(wheel)]), "PATH": os.environ.get("PATH", "")}
    armed = _sp.run([py, "-c", probe], env={**base_env, VE.SHIM_ENV: "0.001"}, text=True, capture_output=True)
    assert armed.returncode == 0, armed.stderr
    lines = armed.stdout.strip().splitlines()
    assert Path(lines[0]) == decoy / "sitecustomize.py" and lines[1] == "True"      # the DECOY is the child's sitecustomize …
    assert lines[2] == "0.001" and lines[3] == "0.001" and lines[4] == "0.001 0.01"  # … and the .pth still lifted the clamp
    unarmed = _sp.run([py, "-c", probe], env=base_env, text=True, capture_output=True)
    assert unarmed.returncode == 0, unarmed.stderr
    ulines = unarmed.stdout.strip().splitlines()
    assert ulines[2] == "0.01" and ulines[3] == "0.01" and ulines[4] == "None None"  # unarmed: the wheel's clamp, untouched


def _exec_shim_copy(name: str):
    """Exec the shim as a FRESH module copy (its own classes) — what a `.pth` boot and a sitecustomize
    import do in one interpreter when both routes are active."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, VE.shim_dir() / VE.SHIM_FILENAME)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_two_shim_copies_in_one_interpreter_register_one_finder_and_never_recurse(tmp_path, monkeypatch):
    """Round 9 (reproduced at b6c4b403): the `.pth` boot AND the sitecustomize import each exec'd the
    shim as its own module copy; the `isinstance` de-dup missed the other copy's class, BOTH finders registered,
    and on `import vllm.sampling_params` each delegated to the other -> RecursionError. The finder now carries
    a stable marker, de-dup keys on it, and a finder never delegates to another marked finder."""
    wheel = _fake_wheel(tmp_path)
    monkeypatch.syspath_prepend(str(wheel))
    for name in ("vllm", "vllm.sampling_params"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    saved_meta = list(sys.meta_path)
    monkeypatch.setenv(VE.SHIM_ENV, "0.001")
    try:
        a = _exec_shim_copy("sisteer_shim_copy_a")          # route 1 (the .pth boot): armed at exec
        b = _exec_shim_copy("sisteer_shim_copy_b")          # route 2 (the sitecustomize import): a DIFFERENT class object
        assert a.ARMED and b.ARMED and a._ClampShimFinder is not b._ClampShimFinder
        finders = [f for f in sys.meta_path if a.is_clamp_finder(f)]
        assert len(finders) == 1 and b.is_clamp_finder(finders[0])                 # one finder, recognised by both copies
        assert not isinstance(finders[0], b._ClampShimFinder)                        # ... which isinstance could not see
        import vllm.sampling_params as sp                                            # the real import machinery, once
        assert sp._MAX_TEMP == 0.001 and sp._SISTEER_MAX_TEMP_FROM == 0.01 and sp.SamplingParams(temperature=0.001).temperature == 0.001
        # a SECOND marked finder forced in (a copy that predates the de-dup): still no recursion — neither delegates to the other
        rogue = b._ClampShimFinder(0.001)
        sys.meta_path.insert(1, rogue)
        monkeypatch.delitem(sys.modules, "vllm.sampling_params")
        import vllm.sampling_params as sp2
        assert sp2._MAX_TEMP == 0.001
    finally:
        sys.meta_path[:] = saved_meta
        for name in ("vllm", "vllm.sampling_params"):
            sys.modules.pop(name, None)


def test_both_routes_active_in_a_real_fresh_interpreter_patch_once_without_recursion(tmp_path):
    """The combined case the round-8 composer produced and the qwen smoke at b6c4b403 died of: a REAL fresh
    interpreter with the shim's .pth in a site dir AND the shim dir on PYTHONPATH as `sitecustomize`, armed.
    Expected: `vllm.sampling_params` imports once, exactly one marked finder, clamp 0.001, no RecursionError."""
    import shutil as _sh
    import subprocess as _sp
    py = "/usr/bin/python3" if Path("/usr/bin/python3").is_file() else _sh.which("python3")
    if not py:
        pytest.skip("no system python3")
    userbase = tmp_path / "userbase"
    r = _sp.run([py, "-c", "import site; print(site.getusersitepackages()); print(site.ENABLE_USER_SITE)"],
                env={"PYTHONUSERBASE": str(userbase), "PATH": os.environ.get("PATH", "")}, text=True, capture_output=True)
    user_site, enabled = r.stdout.strip().splitlines()[:2]
    if enabled != "True":
        pytest.skip(f"{py} has the user site disabled")
    Path(user_site).mkdir(parents=True)
    (Path(user_site) / VE.SHIM_PTH_NAME).write_text(VE.pth_line(VE.shim_dir() / VE.SHIM_FILENAME) + "\n")
    wheel = _fake_wheel(tmp_path)
    probe = ("import sys, sitecustomize, vllm.sampling_params as sp; "
             "print(sitecustomize.__file__); "
             "print(sum(1 for f in sys.meta_path if getattr(f, '__sisteer_clamp_shim__', None) == 'sisteer-vllm-clamp-override-shim/1')); "
             "print(sp._MAX_TEMP); print(sp.SamplingParams(temperature=0.001).temperature)")
    env = {"PYTHONUSERBASE": str(userbase), "PYTHONPATH": os.pathsep.join([str(VE.shim_dir()), str(wheel)]),
           "PATH": os.environ.get("PATH", ""), VE.SHIM_ENV: "0.001"}
    both = _sp.run([py, "-c", probe], env=env, text=True, capture_output=True)
    assert both.returncode == 0 and "RecursionError" not in both.stderr, both.stderr[-2000:]
    lines = both.stdout.strip().splitlines()
    assert Path(lines[0]).resolve() == (VE.shim_dir() / VE.SHIM_FILENAME).resolve()   # route 2 loaded as sitecustomize ...
    assert lines[1] == "1" and lines[2] == "0.001" and lines[3] == "0.001"             # ... one finder, patched once, no recursion
