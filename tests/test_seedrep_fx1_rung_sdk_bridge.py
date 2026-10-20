"""fx1: the rung consumer reaches the Modal SDK / the mounted volume instead of the shadowing ``scripts/modal`` package.

The launch tree's 11 seed-1 rungs all died ``DRIVERS_FAILED`` inside ``rung_redraw.py consume``: ``_consume_volume`` did a bare
``import modal`` under the DRIVER venv (``uv sync --frozen --extra cu128``; the ``modal`` distribution is in neither pyproject nor
uv.lock) with ``PYTHONPATH=<tree>/src:<tree>/scripts``, so the import answered with the repo's ``scripts/modal/`` NAMESPACE package
(no ``__init__.py``; no ``Volume``, no ``Dict``) and ``modal.Volume`` raised ``AttributeError: module 'modal' has no attribute
'Volume'`` -- the ModuleNotFoundError it would otherwise have been, spelled misleadingly. ``_kv`` (``ModalDictKV.from_name``) would
have failed the same way one line later. The fix: the consumer reaches the SDK through ONE helper process under the interpreter
the served app names in ``$SEEDREP_MODAL_SDK_PYTHON`` (its own ``sys.executable``, the container's only SDK-bearing python), run in
isolated mode so ``import modal`` there is unambiguous; the grading host paths keep the direct import, now guarded by name.

fx1c: under the container's FLAT mount (/root/seedrep_served.py) `_HERE.parent.parent` is `/`, which fx1b
excluded -- a prefix every absolute path is under -- so the roots came out empty; fx1c gates the `_HERE` exclusions on the real
<repo>/scripts/modal/ layout, never excludes `/` or a single segment, compares paths path-aware, and names an empty result in the log.

fx1b: inside a container the SDK is NOT in the interpreter's site-packages -- modal 1.5.4 mounts it at ``/pkg``
and ``/pkg/sitecustomize.py`` adds ``/__modal/deps`` -- so a ``-I`` helper (no PYTHONPATH, no site customisation) saw neither and died
``No module named 'modal'`` at ``hello``. The app now computes the SDK IMPORT ROOTS from its live interpreter (``sdk_import_roots``: the
parent of ``modal``'s package dir first, then its non-repo ``sys.path`` entries) and hands them to the consumer child in
``$SEEDREP_MODAL_SDK_PATH``; the helper prepends them before ``import modal`` (still ``-I``) and refuses in-band, naming the roots tried.
The tests below follow that reproduction shape: a fake-but-REGULAR ``modal`` package reachable ONLY through an import root (a temp dir
holding ``modal/__init__.py`` with Dict / Volume stubs, never site-packages), under the real ``sys.executable``.

The pytest venv is the same shape as the driver venv here: no ``modal`` distribution, ``scripts/`` on ``sys.path`` -- so the
subprocess probes below ARE the container's import resolution. Everything synthetic (placeholder tree id / wave uuid, ``TESTHOST``-free).
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _seedrep_testlib as T  # noqa: E402
from test_seedrep_ladder_review_fixes import _closed_static_queue, _axes, _tally  # noqa: E402
from _env import host_site_path  # noqa: E402  (tests/ helper: a host-only data path through the site config)

L = T.load_ladder()
RR = T.load_rung_redraw()
REPO = T.REPO
HELPER = REPO / "scripts" / "sdfbase" / "seedrep" / "modal_sdk_bridge.py"
SERVED = REPO / "scripts" / "modal" / "seedrep_served.py"
ENV = "SEEDREP_MODAL_SDK_PYTHON"
PATH_ENV = "SEEDREP_MODAL_SDK_PATH"

#: the consumer's argv, in the plan's shape (`rung_consumer_argv`): --vol-root + --volume, no --memory-kv-dir
CONSUME_ARGV = ["consume", "--vol-root", "/vol", "--volume", "twosi-fits", "--W7", T.W7, "--wave-uuid", T.WAVE, "--leg", "renellm", "--view", "s1",
                "--kind", "static", "--cap-next", "16384", "--producers", "0", "--attempt-id", "a1", "--call-id", "fc-1",
                "--proxy-url", "http://127.0.0.1:1/v1", "--width", "21", "--shard", "0", "--attempt-root-local", "/workspace/attempt"]


def _container_shaped_probe(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` the way the rung container runs the consumer: the venv's interpreter (no ``modal`` distribution) with the tree's
    ``src`` and ``scripts`` on PYTHONPATH and nothing else of ours, ``$SEEDREP_MODAL_SDK_PYTHON`` unset."""
    env = {k: v for k, v in os.environ.items() if k not in (ENV, "PYTHONPATH")}
    env["PYTHONPATH"] = f"{REPO / 'src'}:{REPO / 'scripts'}"
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=120, env=env, cwd=str(REPO))


def _venv_has_no_sdk() -> bool:
    r = subprocess.run([sys.executable, "-I", "-c", "import importlib.util; print(importlib.util.find_spec('modal') is None)"], capture_output=True, text=True)
    return r.stdout.strip() == "True"


needs_sdk_free_venv = pytest.mark.skipif(not _venv_has_no_sdk(), reason="this interpreter carries the Modal SDK; the shadow only bites without it")


# --------------------------------------------------------------------------- #
# the known-bad, in the container's own shape                                  #
# --------------------------------------------------------------------------- #

@needs_sdk_free_venv
def test_known_bad_a_bare_import_under_the_driver_venv_shape_answers_with_the_scripts_modal_namespace_package():
    r = _container_shaped_probe("""
        import json, modal
        print(json.dumps({"volume": hasattr(modal, "Volume"), "dict": hasattr(modal, "Dict"), "file": getattr(modal, "__file__", None),
                          "path": [str(p) for p in getattr(modal, "__path__", [])]}))
    """)
    assert r.returncode == 0, r.stderr[-800:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["volume"] is False and got["dict"] is False and got["file"] is None          # a namespace package: no __init__, no SDK surface
    assert any(Path(p) == REPO / "scripts" / "modal" for p in got["path"]), got             # ... and it is the repo's scripts/modal


@needs_sdk_free_venv
def test_the_consumers_sdk_route_without_an_sdk_interpreter_refuses_by_name_never_the_containers_attributeerror():
    """The container's traceback ended ``AttributeError: module 'modal' has no attribute 'Volume'`` at ``_consume_volume``. The fixed
    consumer refuses BY NAME there instead: the message says the import answered with the repo's scripts/modal, that this interpreter
    carries no Modal SDK, and names the env var that routes it to one."""
    r = _container_shaped_probe(f"""
        import importlib.util, json, sys
        spec = importlib.util.spec_from_file_location("rr_probe", {str(REPO / 'scripts/sdfbase/seedrep/rung_redraw.py')!r})
        RR = importlib.util.module_from_spec(spec); sys.modules["rr_probe"] = RR; spec.loader.exec_module(RR)
        ap = RR.build_parser(); a = ap.parse_args({CONSUME_ARGV!r})
        try:
            RR._consume_volume(ap, a)
            print(json.dumps({{"raised": None}}))
        except BaseException as e:
            print(json.dumps({{"raised": type(e).__name__, "msg": str(e)}}))
    """)
    assert r.returncode == 0, r.stderr[-1200:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["raised"] is not None, "the SDK route silently succeeded with no SDK in this interpreter"
    assert got["raised"] != "AttributeError", f"the container's failure reproduced verbatim: {got}"
    msg = got["msg"]
    assert "scripts/modal" in msg and "Modal SDK" in msg and ENV in msg, msg


def test_the_guarded_import_in_the_ladder_module_names_a_namespace_shadow_and_returns_a_real_sdk(monkeypatch):
    shadow = types.ModuleType("modal")
    shadow.__path__ = [str(REPO / "scripts" / "modal")]                     # what the container saw
    monkeypatch.setitem(sys.modules, "modal", shadow)
    with pytest.raises(L.LadderRefusal, match="scripts/modal.*Modal SDK") as ei:
        L.modal_sdk()
    assert RR.SDK_PYTHON_ENV in str(ei.value)
    with pytest.raises(L.LadderRefusal):
        L.ModalDictKV.from_name("any")                                        # the Dict half fails by name too (it used to be one line behind)
    real = types.ModuleType("modal")
    real.Volume = types.SimpleNamespace(from_name=lambda n: n)
    real.Dict = types.SimpleNamespace(from_name=lambda n, create_if_missing=True: {"n": n})
    monkeypatch.setitem(sys.modules, "modal", real)
    assert L.modal_sdk() is real
    assert L.ModalDictKV.from_name("d").name == "d"
    # each caller names ITS surface: the lease rule's double (FunctionCall + exception, no Dict / Volume) passes its check and fails the
    # consumer's; the namespace shadow fails every check, naming what is missing
    lease = types.ModuleType("modal")
    lease.FunctionCall = types.SimpleNamespace(from_id=lambda cid: None)
    lease.exception = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "modal", lease)
    assert L.modal_sdk("FunctionCall", "exception") is lease
    with pytest.raises(L.LadderRefusal, match=r"missing \['Dict', 'Volume'\]"):
        L.modal_sdk()
    monkeypatch.setitem(sys.modules, "modal", shadow)
    with pytest.raises(L.LadderRefusal, match=r"missing \['FunctionCall', 'exception'\]"):
        L.modal_sdk("FunctionCall", "exception")


def test_no_bare_modal_import_survives_in_the_consumer_or_the_protocol_module():
    """The docstring rule (rung_redraw.py line 5) is now mechanical: outside the ONE guarded accessor, neither module spells
    ``import modal`` -- the consumer routes through the bridge or ``L.modal_sdk()``."""
    import re
    for rel in ("scripts/sdfbase/seedrep/rung_redraw.py", "scripts/modal/seedrep_ladder.py"):
        src = (REPO / rel).read_text()
        hits = [m.start() for m in re.finditer(r"^\s*(import modal\b|from modal\b)", src, re.M)]
        lines = [src[:h].count("\n") + 1 for h in hits]
        if rel.endswith("seedrep_ladder.py"):
            assert len(hits) == 1, f"{rel}: bare modal imports at lines {lines}; the one allowed sits inside modal_sdk()"
            body = src[src.index("def modal_sdk("):]
            body = body[:body.index("\ndef ", 1)]
            assert "import modal" in body
        else:
            assert hits == [], f"{rel}: bare modal imports at lines {lines}"


# --------------------------------------------------------------------------- #
# the helper: the SDK side of the bridge, over a stub SDK                       #
# --------------------------------------------------------------------------- #

class _StubDict(L.MemoryKV):
    pass


def _stub_sdk(dicts: dict, volumes: dict):
    m = types.ModuleType("modal")
    m.__version__ = "0.0-stub"
    m.__file__ = "<stub>/modal/__init__.py"
    m.Dict = types.SimpleNamespace(from_name=lambda name, create_if_missing=True: dicts.setdefault(name, _StubDict(name)))

    class _Vol:
        def __init__(self, name):
            self.name, self.commits, self.reloads = name, 0, 0

        def commit(self):
            self.commits += 1

        def reload(self):
            self.reloads += 1

    m.Volume = types.SimpleNamespace(from_name=lambda name: volumes.setdefault(name, _Vol(name)))
    return m


def _load_helper():
    spec = importlib.util.spec_from_file_location("modal_sdk_bridge_under_test", HELPER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modal_sdk_bridge_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _drive(helper, sdk, requests: list, roots=()) -> list:
    inp = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    out = io.StringIO()
    rc = helper.serve(inp, out, sdk=sdk, roots=list(roots))
    assert rc == 0
    return [json.loads(ln) for ln in out.getvalue().splitlines()]


def test_the_helper_serves_the_kv_protocol_and_the_volume_verbs_over_the_sdk():
    helper = _load_helper()
    dicts: dict = {}
    volumes: dict = {}
    sdk = _stub_sdk(dicts, volumes)
    replies = _drive(helper, sdk, [
        {"id": 1, "op": "hello"},
        {"id": 2, "op": "dict.put", "name": "d", "key": "k", "value": {"a": 1}, "skip_if_exists": True},
        {"id": 3, "op": "dict.put", "name": "d", "key": "k", "value": {"a": 2}, "skip_if_exists": True},   # held -> False, no write
        {"id": 4, "op": "dict.get", "name": "d", "key": "k"},
        {"id": 5, "op": "dict.get", "name": "d", "key": "missing", "default": "dflt"},
        {"id": 6, "op": "dict.contains", "name": "d", "key": "k"},
        {"id": 7, "op": "dict.contains", "name": "d", "key": "missing"},
        {"id": 8, "op": "dict.put", "name": "d", "key": "k2", "value": "v2"},
        {"id": 9, "op": "dict.items", "name": "d"},
        {"id": 10, "op": "dict.pop", "name": "d", "key": "missing"},                                     # KeyError, by type
        {"id": 11, "op": "dict.pop", "name": "d", "key": "missing", "default": None},
        {"id": 12, "op": "dict.pop", "name": "d", "key": "k2"},
        {"id": 13, "op": "volume.reload", "name": "twosi-fits"},
        {"id": 14, "op": "volume.commit", "name": "twosi-fits"},
        {"id": 15, "op": "volume.commit", "name": "twosi-fits"},
        {"id": 16, "op": "no.such.op"},
    ])
    by = {r["id"]: r for r in replies}
    assert by[1]["ok"] and by[1]["result"]["modal"] == "0.0-stub" and by[1]["result"]["python"] == sys.executable and by[1]["result"]["roots"] == []
    assert by[2]["result"] is True and by[3]["result"] is False and by[4]["result"] == {"a": 1}
    assert by[5]["result"] == "dflt" and by[6]["result"] is True and by[7]["result"] is False
    assert sorted(map(tuple, by[9]["result"])) == [("k", {"a": 1}), ("k2", "v2")]
    assert by[10]["ok"] is False and by[10]["type"] == "KeyError"
    assert by[11]["ok"] and by[11]["result"] is None and by[12]["result"] == "v2"
    assert by[13]["ok"] and by[14]["ok"] and by[15]["ok"]
    assert volumes["twosi-fits"].reloads == 1 and volumes["twosi-fits"].commits == 2
    assert by[16]["ok"] is False and "no.such.op" in by[16]["error"]
    assert set(dicts) == {"d"} and dict(dicts["d"].items()) == {"k": {"a": 1}}


def test_the_helper_answers_a_malformed_line_by_name_and_keeps_serving():
    helper = _load_helper()
    sdk = _stub_sdk({}, {})
    inp = io.StringIO("this is not json\n" + json.dumps({"id": 2, "op": "hello"}) + "\n" + json.dumps({"op": "hello"}) + "\n")
    out = io.StringIO()
    assert helper.serve(inp, out, sdk=sdk) == 0
    replies = [json.loads(ln) for ln in out.getvalue().splitlines()]
    assert replies[0]["ok"] is False and replies[0]["id"] is None and "JSON" in replies[0]["error"]
    assert replies[1]["id"] == 2 and replies[1]["ok"]
    assert replies[2]["id"] is None and replies[2]["ok"]                     # an id-less request is answered with a null id


def test_the_helper_refuses_a_shadowing_module_as_the_sdk():
    helper = _load_helper()
    shadow = types.ModuleType("modal")
    shadow.__path__ = ["/somewhere/scripts/modal"]
    with pytest.raises(RuntimeError, match=r"not the Modal SDK.*roots tried \(SEEDREP_MODAL_SDK_PATH\): \['/r1'\]"):
        helper.checked_sdk(shadow, ["/r1"])
    assert helper.SDK_PATH_ENV == PATH_ENV
    assert helper.sdk_roots_from_env({PATH_ENV: os.pathsep.join(["/a", "", "/b"])}) == ["/a", "/b"] and helper.sdk_roots_from_env({}) == []


# --------------------------------------------------------------------------- #
# the client: the consumer end to end over a real helper process               #
# --------------------------------------------------------------------------- #

FAKE_SDK = '''
"""A stand-in `modal` package for the helper PROCESS: Dicts persisted as JSON files, Volume verbs appended to a log, so the test
(another process) can read what the consumer did through the bridge."""
import json, os, pathlib
ROOT = pathlib.Path(os.environ["FX1_FAKE_SDK_ROOT"])
__version__ = "0.0-fake-process"

class _Dict:
    def __init__(self, name):
        self.name = name
        self.path = ROOT / "dicts" / (name + ".json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
    def _load(self):
        return json.loads(self.path.read_text()) if self.path.exists() else {}
    def _save(self, d):
        self.path.write_text(json.dumps(d, sort_keys=True))
    def put(self, key, value, *, skip_if_exists=False):
        die = int(os.environ.get("FX1_FAKE_SDK_DIE_ON_PUT", "0") or 0)
        if die:
            n = int((ROOT / "puts").read_text() or 0) + 1 if (ROOT / "puts").exists() else 1
            (ROOT / "puts").write_text(str(n))
            if n >= die:
                os._exit(7)                                      # the helper PROCESS dies mid-verb: the test of the client's fail-closed path
        d = self._load()
        if skip_if_exists and key in d:
            return False
        d[key] = value
        self._save(d)
        return True
    def get(self, key, default=None):
        return self._load().get(key, default)
    def pop(self, key, *a):
        d = self._load()
        if key not in d:
            if a:
                return a[0]
            raise KeyError(key)
        v = d.pop(key)
        self._save(d)
        return v
    def contains(self, key):
        return key in self._load()
    def items(self):
        return list(self._load().items())

class Dict:
    @staticmethod
    def from_name(name, create_if_missing=True):
        return _Dict(name)

class _Vol:
    def __init__(self, name):
        self.name = name
    def _log(self, verb):
        with (ROOT / "volume_ops.log").open("a") as f:
            f.write(verb + "\\n")
    def commit(self):
        self._log("commit")
    def reload(self):
        self._log("reload")

class Volume:
    @staticmethod
    def from_name(name):
        return _Vol(name)
'''


def _fake_sdk_root(tmp_path: Path) -> Path:
    """The shape: a fake-but-REGULAR ``modal`` package (``__init__.py`` with Dict / Volume) reachable ONLY through an import ROOT -- a temp
    directory, never site-packages -- the way a container's /pkg is. The SDK python is the real ``sys.executable``; the root reaches the
    helper through ``$SEEDREP_MODAL_SDK_PATH`` exactly as the app hands it to the consumer child."""
    sdk_dir = tmp_path / "fakesdk" / "modal"
    sdk_dir.mkdir(parents=True)
    (sdk_dir / "__init__.py").write_text(FAKE_SDK)
    return tmp_path / "fakesdk"


def _route_via_env(monkeypatch, tmp_path: Path) -> Path:
    """The consumer child's two variables, as the app sets them: the SDK python and the import roots."""
    root = _fake_sdk_root(tmp_path)
    monkeypatch.setenv(ENV, sys.executable)
    monkeypatch.setenv(PATH_ENV, str(root))
    return root


def _helper_hello(env_extra: dict, root_env: bool = True) -> dict:
    """Run the REAL helper as the consumer does (``<sys.executable> -I <helper>``), with a scrubbed environment plus ``env_extra``; return the
    ``hello`` reply."""
    env = {k: v for k, v in os.environ.items() if k not in (ENV, PATH_ENV, "PYTHONPATH")}
    env.update(env_extra)
    r = subprocess.run([sys.executable, "-I", str(HELPER)], input=json.dumps({"id": 1, "op": "hello"}) + "\n", capture_output=True, text=True,
                       timeout=60, env=env)
    assert r.returncode == 0, r.stderr[-800:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@needs_sdk_free_venv
def test_sol_a_the_helper_under_isolated_mode_with_no_roots_refuses_by_name_in_band(tmp_path):
    """The reproduction: -I ignores PYTHONPATH and site customisation, so the helper sees neither /pkg nor /__modal/deps -- nor, here, a fake
    root handed only through PYTHONPATH. The refusal is the ``hello`` reply (never a dead pipe), naming the roots tried (none)."""
    root = _fake_sdk_root(tmp_path)
    got = _helper_hello({"PYTHONPATH": str(root)})                              # a PYTHONPATH root is invisible under -I: the container's shape
    assert got["ok"] is False and got["type"] == "RuntimeError"
    assert "No module named 'modal'" in got["error"] and f"roots tried ({PATH_ENV}): []" in got["error"], got["error"]
    # ... and through the consumer's client it is the handshake's named error, before any Dict is opened
    env = {k: v for k, v in os.environ.items() if k not in (PATH_ENV,)}
    with pytest.raises(RR.SdkBridgeError, match=r"hello.*No module named 'modal'.*roots tried"):
        RR.SdkBridge(sys.executable, popen=lambda argv, **kw: subprocess.Popen(argv, env=env, **kw))


@needs_sdk_free_venv
def test_sol_b_with_the_sdk_path_env_the_helper_prepends_the_root_and_handshakes(tmp_path):
    root = _fake_sdk_root(tmp_path)
    (tmp_path / "state").mkdir()
    state = {"FX1_FAKE_SDK_ROOT": str(tmp_path / "state")}                   # the fake package's own store (read at its import)
    got = _helper_hello({PATH_ENV: str(root), **state})
    assert got["ok"] is True, got
    assert got["result"]["roots"] == [str(root)] and got["result"]["modal"] == "0.0-fake-process"
    assert Path(got["result"]["file"]) == root / "modal" / "__init__.py"            # the package at the ROOT, not a site-packages copy
    # several roots, os.pathsep-joined, in order; an empty entry is dropped; a non-SDK first root does not shadow the SDK root
    got = _helper_hello({PATH_ENV: os.pathsep.join([str(tmp_path / "deps-first"), "", str(root)]), **state})
    assert got["ok"] is True and got["result"]["roots"] == [str(tmp_path / "deps-first"), str(root)]


def test_sol_c_the_app_derives_the_sdk_import_roots_from_its_live_interpreter_and_hands_them_to_the_consumer_child(srv, tmp_path, monkeypatch):
    """`sdk_import_roots`: the parent of modal's package directory FIRST, then this process's sys.path entries in order -- minus the repo tree
    (src / scripts and anything under it), the app's own checkout, the CWD, empties, non-directories, duplicates -- and the consumer child's env
    carries them os.pathsep-joined under the variable the helper reads."""
    pkg = tmp_path / "pkg"                                                    # the container's /pkg: the SDK's package root
    (pkg / "modal").mkdir(parents=True)
    (pkg / "modal" / "__init__.py").write_text("Dict = Volume = object\n")
    deps = tmp_path / "__modal" / "deps"                                       # the container's /__modal/deps (sitecustomize)
    deps.mkdir(parents=True)
    site = tmp_path / "usr" / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    tree = tmp_path / "workspace" / "seedrep" / "tree"
    (tree / "src").mkdir(parents=True)
    (tree / "scripts").mkdir()
    cwd = tmp_path / "root"
    cwd.mkdir()
    fake = types.ModuleType("modal")
    fake.__file__ = str(pkg / "modal" / "__init__.py")
    sys_path = ["", str(tree / "src"), str(tree / "scripts"), str(tree), str(cwd), str(deps), str(pkg), str(site), str(tmp_path / "gone"), str(deps), "."]
    roots = srv.sdk_import_roots(tree=str(tree), modal_module=fake, sys_path=sys_path, cwd=str(cwd))
    assert roots == [str(pkg), str(deps), str(site)]                          # pkg first (from modal.__file__), then the app's order; the rest excluded
    assert not any(str(tree) in r for r in roots) and str(cwd) not in roots
    # the live default: this process's modal (a stub here) and sys.path -- the repo tree of THIS clone is excluded
    monkeypatch.setitem(sys.modules, "modal", fake)
    live = srv.sdk_import_roots(tree=str(REPO))
    assert live and live[0] == str(pkg) and not any(r == str(REPO) or r.startswith(str(REPO) + "/") for r in live)
    # the consumer child carries both variables
    row = {"standup_id": "renellm__s1__16384__shard00", "leg": "renellm", "eval_legs": ["renellm"], "seed_bases": [1]}
    kid = srv.rung_consumer_child(row, ["{driver_python}", "x.py", "consume"], attempt_root="/workspace/attempt", attempt_id="a1", tree=str(REPO))
    assert kid["env"][ENV] == sys.executable and kid["env"][srv.RUNG_SDK_PATH_ENV].split(os.pathsep) == live
    assert srv.RUNG_SDK_PATH_ENV == PATH_ENV
    # the call site passes the staged tree
    src = SERVED.read_text()
    assert 'rung_consumer_child(row, row["rung_consumer"], attempt_root=str(root), attempt_id=attempt_id, tree=str(tree))' in src


def test_sol_d_under_the_containers_flat_mount_the_roots_keep_pkg_and_modal_deps(srv):
    """The defect at 0f47cefa2 (fx1b): the image installs the app FLAT at /root/seedrep_served.py, so `_HERE.parent.parent` is `/` -- fx1b
    excluded it, a prefix every absolute path is under, and the roots came out EMPTY. fx1c: `_HERE`-derived exclusions apply ONLY in the
    <repo>/scripts/modal/ layout (`grading_host_repo_src`'s rule); `/` and any single-segment prefix are never exclusions; the prefix test is
    path-aware. The shape here: here=/root, sys.path with /pkg and /__modal/deps + stdlib + site-packages, the staged tree, CWD /root."""
    here = Path("/root")
    tree = "/workspace/seedrep/si-steering-eval-repo/e526edf801f5dafddbb8df210c3de3b2ab7abed7"
    sys_path = ["/pkg", "/root", "/usr/local/lib/python312.zip", "/usr/local/lib/python3.12", "/usr/local/lib/python3.12/lib-dynload",
                "/usr/local/lib/python3.12/site-packages", "/__modal/deps", f"{tree}/src", f"{tree}/scripts", "/src", ""]
    present = {"/pkg", "/root", "/usr/local/lib/python3.12", "/usr/local/lib/python3.12/lib-dynload", "/usr/local/lib/python3.12/site-packages", "/__modal/deps",
               f"{tree}/src", f"{tree}/scripts"}
    fake = types.ModuleType("modal")
    fake.__file__ = "/pkg/modal/__init__.py"
    excl = srv.sdk_import_exclusions(tree=tree, cwd="/root", here=here)
    assert excl == [tree]                                                     # the staged tree only: no "/", no "/root", nothing _HERE-derived
    roots = srv.sdk_import_roots(tree=tree, modal_module=fake, sys_path=sys_path, cwd="/root", exists=lambda p: p in present, here=here)
    assert roots[0] == "/pkg" and "/__modal/deps" in roots, roots
    assert roots.index("/pkg") < roots.index("/__modal/deps") and roots == [r for r in roots if r in present]      # order preserved, existing only
    assert not any(r.startswith(tree) for r in roots) and "/src" not in roots and "" not in roots
    # a sibling prefix is NOT a parent (path-aware, never string-prefix): /pkg2 survives an exclusion of /pkg-like names, /pkgx is not under /pkg
    roots2 = srv.sdk_import_roots(tree="/w/tree", modal_module=fake, sys_path=["/w/tree-extra", "/w/tree", "/w/tree/src"], cwd="/root",
                                  exists=lambda p: True, here=here)
    assert roots2 == ["/pkg", "/w/tree-extra"]
    # the grading host layout (this clone) still excludes its checkout and scripts/, and the exclusions never contain "/" or a single segment
    live = srv.sdk_import_exclusions(tree="/w/tree", cwd="/root")
    assert str(REPO) in live and str(REPO / "scripts") in live and "/" not in live and "/root" not in live
    assert all(len(Path(x).parts) >= 3 for x in live)


def test_sol_d_empty_roots_are_named_in_the_container_log_at_the_call_site(srv, monkeypatch, capsys):
    """An empty root set is never silent: the consumer child's env carries "" and the app prints WHY (modal's file, sys.path, the exclusions, here,
    tree) so the helper's `hello` refusal has its cause one line above it in the container log."""
    monkeypatch.setattr(srv, "sdk_import_roots", lambda **kw: [])
    row = {"standup_id": "renellm__s1__16384__shard00", "leg": "renellm", "eval_legs": ["renellm"], "seed_bases": [1]}
    kid = srv.rung_consumer_child(row, ["{driver_python}", "x.py", "consume"], attempt_root="/workspace/attempt", attempt_id="a1", tree="/w/tree")
    assert kid["env"][PATH_ENV] == ""
    out = capsys.readouterr().out
    assert "WARNING: no Modal SDK import roots derived" in out and "sys.path=" in out and "excluded=" in out and "here=" in out and "tree='/w/tree'" in out
    # the healthy path prints nothing and carries the roots
    monkeypatch.setattr(srv, "sdk_import_roots", lambda **kw: ["/pkg", "/__modal/deps"])
    kid = srv.rung_consumer_child(row, ["{driver_python}", "x.py", "consume"], attempt_root="/workspace/attempt", attempt_id="a1", tree="/w/tree")
    assert kid["env"][PATH_ENV] == os.pathsep.join(["/pkg", "/__modal/deps"]) and capsys.readouterr().out == ""


def test_the_bridge_client_runs_the_consumer_end_to_end_over_a_helper_process(tmp_path, monkeypatch, capsys):
    """The P1-1 consume run (test_seedrep_ladder_review_fixes) again, with NO `modal` module in this process at all: the consumer
    finds $SEEDREP_MODAL_SDK_PYTHON, spawns the helper under it, and every Dict verb and Volume commit/reload crosses the bridge."""
    monkeypatch.delitem(sys.modules, "modal", raising=False)
    fake_root = tmp_path / "sdk-state"
    fake_root.mkdir()
    monkeypatch.setenv("FX1_FAKE_SDK_ROOT", str(fake_root))
    sdk_root = _route_via_env(monkeypatch, tmp_path)
    vol_root = tmp_path / "vol"
    vol = L.LocalVolume(vol_root)
    spec = T.static_spec(L, shards=("0",))
    acc = L.MemoryKV(L.dict_name(T.W7, T.WAVE, "accepted"))
    recs = [L.static_queue_record(spec, arm="none", s=1, family="pasttense", row=r, base_store="b", producer=spec.producers[0],
                                  producer_attempt_id="a1", vmodel="petri__base", sampling=T.PRESET) for r in T.rows3()]
    _closed_static_queue(vol, spec, recs, acc)
    # the acceptance the producer took lands in the HELPER's Dict store (the wave's `accepted` Dict, by its name)
    (fake_root / "dicts").mkdir()
    (fake_root / "dicts" / f"{acc.name}.json").write_text(json.dumps(dict(acc.items()), sort_keys=True))
    proxy = T.StubProxy()
    try:
        rc = RR.main(["consume", "--vol-root", str(vol_root), "--volume", "twosi-fits", *_axes(spec), "--attempt-id", "rung-1", "--call-id", "fc-1",
                      "--no-request-evidence", "--proxy-url", proxy.url, "--width", "2", "--rung-shard", "0", "--rung-shards", "0", "--finish-producer"])
    finally:
        proxy.close()
    out = capsys.readouterr()
    assert rc == 0, out.err[-1500:]
    tally = _tally(out)
    assert (tally["drawn"], tally["won"], tally["closed"], tally["covered"]) == (3, 3, True, True)
    assert tally["sdk_bridge"]["python"] == sys.executable and tally["sdk_bridge"]["modal"] == "0.0-fake-process"
    assert tally["sdk_bridge"]["roots"] == [str(sdk_root)] and Path(tally["sdk_bridge"]["file"]) == sdk_root / "modal" / "__init__.py"
    assert "modal" not in sys.modules                                          # nothing was imported into THIS process
    ops = (fake_root / "volume_ops.log").read_text().split()
    assert ops.count("reload") >= 1 and ops.count("commit") >= 3              # the mount's real verbs, through the helper
    names = {p.stem for p in (fake_root / "dicts").glob("*.json")}
    assert names == {L.dict_name(T.W7, T.WAVE, p) for p in ("accepted", "claims", "results")}
    results = json.loads((fake_root / "dicts" / f"{L.dict_name(T.W7, T.WAVE, 'results')}.json").read_text())
    claims = json.loads((fake_root / "dicts" / f"{L.dict_name(T.W7, T.WAVE, 'claims')}.json").read_text())
    assert len(results) == 3 and len(claims) == 3 and len(proxy.bodies) == 3
    assert tally["producer"]["state"] == "accepted"


def test_the_bridge_client_surfaces_a_dead_or_sdk_less_helper_by_name(tmp_path):
    dead = tmp_path / "dead.py"
    dead.write_text("import sys; print('helper: no SDK here', file=sys.stderr); sys.exit(3)\n")
    with pytest.raises(RR.SdkBridgeError, match="exited rc=3"):
        RR.SdkBridge(sys.executable, helper=dead)
    missing = tmp_path / "nope" / "python"
    with pytest.raises(RR.SdkBridgeError, match="could not start"):
        RR.SdkBridge(str(missing), helper=HELPER)


def test_the_bridge_kv_keeps_the_memory_kv_semantics_through_the_client(tmp_path, monkeypatch):
    """put(skip_if_exists) -> False on a held key; get default; pop KeyError vs default; contains; items -- the client raises what the
    KV protocol's callers expect (a KeyError, not a bridge error)."""
    fake_root = tmp_path / "s"
    fake_root.mkdir()
    monkeypatch.setenv("FX1_FAKE_SDK_ROOT", str(fake_root))
    _route_via_env(monkeypatch, tmp_path)
    b = RR.SdkBridge(sys.executable, helper=HELPER)
    try:
        kv = b.kv("d")
        assert kv.put("k", {"a": 1}, skip_if_exists=True) is True and kv.put("k", {"a": 2}, skip_if_exists=True) is False
        assert kv.get("k") == {"a": 1} and kv.get("zz", 7) == 7 and kv.contains("k") and not kv.contains("zz")
        assert kv.put("k2", "v") is True and sorted(kv.items()) == [("k", {"a": 1}), ("k2", "v")]
        with pytest.raises(KeyError):
            kv.pop("zz")
        assert kv.pop("zz", None) is None and kv.pop("k2") == "v" and dict(kv.items()) == {"k": {"a": 1}}
        commit, reload = b.volume_verbs("twosi-fits")
        reload(); commit()
        assert (fake_root / "volume_ops.log").read_text().split() == ["reload", "commit"]
    finally:
        b.close()
    with pytest.raises(RR.SdkBridgeError, match="closed"):
        kv.get("k")


def test_a_helper_that_dies_mid_consume_fails_the_consumer_closed_by_name_and_never_spins_on_the_queue(tmp_path, monkeypatch, capsys):
    """The helper process exits during the second Dict verb: the consumer raises `SdkBridgeError` naming the exit (rc=7) on that verb --
    `dispatch` re-raises a worker's first exception and the round loop's reload / Dict scans are unguarded -- so the container dies on it
    instead of polling the queue to its deadline; every later bridge call is refused as closed."""
    monkeypatch.delitem(sys.modules, "modal", raising=False)
    fake_root = tmp_path / "sdk-state"
    fake_root.mkdir()
    monkeypatch.setenv("FX1_FAKE_SDK_ROOT", str(fake_root))
    monkeypatch.setenv("FX1_FAKE_SDK_DIE_ON_PUT", "2")
    _route_via_env(monkeypatch, tmp_path)
    vol_root = tmp_path / "vol"
    vol = L.LocalVolume(vol_root)
    spec = T.static_spec(L, shards=("0",))
    acc = L.MemoryKV(L.dict_name(T.W7, T.WAVE, "accepted"))
    recs = [L.static_queue_record(spec, arm="none", s=1, family="pasttense", row=r, base_store="b", producer=spec.producers[0],
                                  producer_attempt_id="a1", vmodel="petri__base", sampling=T.PRESET) for r in T.rows3()]
    _closed_static_queue(vol, spec, recs, acc)
    (fake_root / "dicts").mkdir()
    (fake_root / "dicts" / f"{acc.name}.json").write_text(json.dumps(dict(acc.items()), sort_keys=True))
    proxy = T.StubProxy()
    t0 = time.time()
    try:
        with pytest.raises(RR.SdkBridgeError, match=r"exited rc=7 during 'dict\.put'"):
            RR.main(["consume", "--vol-root", str(vol_root), "--volume", "twosi-fits", *_axes(spec), "--attempt-id", "rung-1", "--call-id", "fc-1",
                     "--no-request-evidence", "--proxy-url", proxy.url, "--width", "1", "--rung-shard", "0", "--rung-shards", "0",
                     "--deadline-s", "600", "--poll-s", "60"])
    finally:
        proxy.close()
    assert time.time() - t0 < 60                                              # the 600 s deadline / 60 s poll never came into it
    out = capsys.readouterr()
    assert "Modal SDK bridge:" in out.out and "CLOSED and covered" not in out.out and "INCOMPLETE" not in out.err


# --------------------------------------------------------------------------- #
# the carry: none of fx1's files gates the six completed seed-1 rows           #
# --------------------------------------------------------------------------- #

FX1_FILES = {"scripts/modal/seedrep_served.py", "scripts/modal/seedrep_ladder.py", "scripts/sdfbase/seedrep/rung_redraw.py",
             "scripts/sdfbase/seedrep/modal_sdk_bridge.py"}
LAUNCH_W = "e526edf801f5dafddbb8df210c3de3b2ab7abed7"
RS = host_site_path("$STATE/romp/candidate-si-refit/run-state/full22/seedrep_battery")
PEEK = Path("/tmp/launch_vol_peek")
#: (leg, prior tree, standup id, the pulled root holding modal_evidence.json) -- the launch tree's own seed-1 bases (prev tree = the launch tree;
#: bon has no pulled root: derived from the row alone) and the three priors carried into it (the fj35 triples)
SIX = [("renellm", None, "renellm__s1__base__shard00", RS / "stores/e526edf/renellm/01M30KG5XMNHT6KF6MBBBP6B5R-ce97651fbfc6"),
       ("codeattack", None, "codeattack__s1__base__shard00", RS / "stores/e526edf/codeattack/01M30KG6YJZNBHSEX8KW3JCTXR-da8b75d4644d"),
       ("grok420", None, "grok420__s1__base__shard00", RS / "stores/e526edf/grok420/01M30KG7YBN94P4M7EWR3DYX8R-e8208aa7d2b7"),
       ("bon", None, "bon__s1__base__shard00", None),
       ("controls", "6fdd5e5", "controls__s1__base__shard00", PEEK / "6fdd5e5/stores/S5_controls/01M2YNWDRFVSAT7XFYMEXEPPQR-dc4bd2f90303/01M2YNWDRFVSAT7XFYMEXEPPQR-dc4bd2f90303"),
       ("pasttense", "56aff57", "pasttense__s1__base__shard00", PEEK / "56aff57/stores/S5_pasttense/01M2XD1H6WDGXZ65BS0NRNMZRR-ccf6feacddc8/01M2XD1H6WDGXZ65BS0NRNMZRR-ccf6feacddc8"),
       ("misalign", "56aff57", "misalign__s1__base__shard00", PEEK / "56aff57/stores/S5_misalign/01M2XD1H6W9CT4XFNSTZ7VED8R-b4cbfce42621/01M2XD1H6W9CT4XFNSTZ7VED8R-b4cbfce42621")]


def _sha_exists(sha):
    return subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True).returncode == 0


def _archives_present():
    return ((RS / "launch_plan.json").is_file() and all((RS / "prior" / w7 / "launch_plan_draft.json").is_file() for _l, w7, _s, _r in SIX if w7)
            and all(root is None or (root / "modal_evidence.json").is_file() for _l, _w, _s, root in SIX) and _sha_exists(LAUNCH_W))


@pytest.mark.skipif(not _archives_present(), reason="the launch archive, the prior archives and the pulled roots are not on this machine")
def test_the_six_completed_seed1_rows_carry_onto_this_tree_with_no_fx1_file_gating():
    """The real-object re-derivation fj35 runs (archived rows, the pulled roots' evidence, this clone's git) for the six completed seed-1 rows the
    fixed tree must carry: NONE of fx1's four files is in any row's gating set (seedrep_served.py / seedrep_ladder.py are CONTAINER_MODULES --
    recorded, not gating; rung_redraw.py enters only through a rung row's consumer, the drivers load it BY PATH so clause (b)'s import walk never
    resolves it; the helper is spawned by path, never imported), none of them is a differing producer file for any row, and every row's verdict
    against HEAD is ok. So no admission is needed for fx1. (fx1d: the property is stated as claimed -- NOT as set equality with the launch
    tree's verdict: on the landing head other files differ legitimately under their own admissions.)"""
    for _p in (REPO / "scripts" / "modal",):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    import seedrep_carry_over as co
    import seedrep_launch_plan as lp
    git = co.default_git(REPO)
    head = git(["rev-parse", "HEAD"])[1].strip()
    plan = json.loads((RS / "launch_plan.json").read_text())
    assert plan["W"] == LAUNCH_W
    rows = {r["standup_id"]: r for r in plan["standups"]}
    for leg, w7, sid, root in SIX:
        if w7 is None:
            row, prev = rows[sid], LAUNCH_W
        else:
            prior = co.load_prior_source(RS / "prior" / w7)
            prow = co.archived_row(prior, sid)
            s5 = (prior.get("smoke_waves") or {}).get("S5")
            row = lp.smoke_row(prow, lp.smoke_identity(prior["plan"], "S5", s5), max_children=len(prow["children"]), max_timeout_s=int(prow["modal"]["timeout_s"]))
            prev = prior["W"]
        ev = json.loads((root / "modal_evidence.json").read_text()) if root is not None else None
        pset = co.producer_set(row, ev, W=prev)
        imports = co.direct_imports(git, prev, pset["all"])
        gating = sorted((set(pset["all"]) | {p for ps in imports.values() for p in ps}) - set(co.CONTAINER_MODULES))
        assert gating and not (set(gating) & FX1_FILES), (sid, sorted(set(gating) & FX1_FILES))
        v_head = co.producer_set_verdict(gating, prev_sha=prev, this_sha=head, git=git, leg=leg)
        assert v_head["ok"] is True and v_head.get("why") is None, (sid, v_head.get("why"))
        differing = sorted(p["path"] for p in v_head["files"] if not p["identical"])
        assert not set(differing) & FX1_FILES, (sid, sorted(set(differing) & FX1_FILES))     # no fx1 file is a differing producer file
        assert not {a["path"] for a in v_head["admitted_differences"]} & FX1_FILES, sid       # ... nor an admitted one
        assert {c["path"] for c in v_head["container_modules"]} >= {"scripts/modal/seedrep_served.py", "scripts/modal/seedrep_ladder.py"}


# --------------------------------------------------------------------------- #
# the app: the consumer child is handed the app's own interpreter              #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def srv():
    from test_seedrep_w9_container_paths import _stub_modal
    spec = importlib.util.spec_from_file_location("seedrep_served_fx1_under_test", SERVED)
    mod = importlib.util.module_from_spec(spec)
    prior = sys.modules.get("modal")
    sys.modules["modal"] = _stub_modal()
    try:
        spec.loader.exec_module(mod)
    finally:
        if prior is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prior
    return mod


def test_the_app_hands_the_consumer_child_its_own_interpreter_under_the_env_the_consumer_reads(srv):
    assert srv.RUNG_SDK_PYTHON_ENV == RR.SDK_PYTHON_ENV == ENV
    row = {"standup_id": "renellm__s1__16384__shard00", "leg": "renellm", "eval_legs": ["renellm"], "seed_bases": [1]}
    kid = srv.rung_consumer_child(row, ["{driver_python}", "x.py", "consume"], attempt_root="/workspace/attempt", attempt_id="a1", tree="/tree")
    assert kid["env"][ENV] == sys.executable                                  # the served app's python: the container's SDK-bearing one
    assert PATH_ENV in kid["env"]                                             # ... and the import roots its -I helper prepends (fx1b)
    assert kid["arm"] == srv.RUNG_CONSUMER_ARM
    # the consumer's helper ships in the tree beside the consumer, stdlib + modal only (it runs where the driver venv's packages are not)
    assert HELPER.is_file() and RR.SDK_BRIDGE_HELPER == HELPER
    import re
    tops = {m.group(1).split(".")[0] for m in re.finditer(r"^(?:from|import)\s+([\w\.]+)", HELPER.read_text(), re.M)}
    assert tops <= set(sys.stdlib_module_names) | {"modal"}, tops
