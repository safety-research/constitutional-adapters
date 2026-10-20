"""The shared sampled-controls module's OWN contract (sisteer/eval/sampled_controls.py):
the driver seam, the repo locator, the two re-exports, and the driver's use of it.

The module is imported by two drivers (main's ``scripts/served_capability_run.py`` and
a lane pinned to an older serving stack), so what is pinned here is what makes ONE
module safe for both:

  * the four call-time driver globals are ``_Unbound`` sentinels until
    ``bind_driver_globals`` runs — every use RAISES, naming the name and the seam
    (never a silent ``None`` that flows into a store), a rebind to a different object
    is refused BEFORE anything is validated, every offender is reported in one error, and
    a refused bind leaves EVERY name a sentinel (the bind is atomic);
  * ``REPO`` comes from the module's own file (first parent holding
    ``pyproject.toml``), never from a caller or the working directory, and
    ``set_repo_root`` is the explicit override; ``_ladder_decide`` follows it;
  * ``controls_cap`` / ``CONTROLS_CAP_KEY`` are ``cell_controls``' objects re-exported
    (the controls' own cap, with the pre-key fallback), and
    ``FIT_PROVENANCE_TEMPLATE_CHECK_KEY`` is ``sweep``'s object, never a copy;
  * the driver defines none of the moved names, its names ARE the module's objects,
    and it binds the seam before its ``__main__`` guard, so a script run binds
    exactly as an import does.

Offline; no GPU, no network. Synthetic paths only.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE = REPO / "src" / "sisteer" / "eval" / "sampled_controls.py"
DRIVER = REPO / "scripts" / "served_capability_run.py"
sys.path.insert(0, str(REPO / "scripts"))

from sisteer.eval import cell_controls as cc  # noqa: E402
from sisteer.eval import sampled_controls as scp  # noqa: E402
from sisteer.eval import sweep as sweep_mod  # noqa: E402
from sisteer.eval.cell_controls import SampledDecodeError  # noqa: E402

#: the 32 definitions that left the driver for the module (10 ladder + 18 transitive
#: dependencies + 4 escalation-wiring), by name
MOVED = (
    "sampling_seed", "floor_rank_key", "floor_sample_ids", "_decode_of", "_decode_block",
    "GenMeta", "Escalation", "_apply_items", "_apply_escalation", "_run_judged_served",
    "CONTROLS_CAP_LADDER", "FLOOR_SAMPLE_DEFAULT", "FLOOR_SAMPLE_RULE", "META_SUFFIX",
    "META_ROW_KEYS", "META_REGEN_KEYS", "META_COPIED_KEYS", "META_EXCLUDED_KEY",
    "META_EXCLUSIONS", "_text_sha", "_first_divergence", "_fork_fields", "_meta_stats",
    "SourceControl", "_ladder_decide", "_read_items_files", "_locate_source_store",
    "_decoder_of",
    "_validate_source_identity", "_read_source_spool", "_load_source_control",
    "_SOURCE_IDENTITY_KEYS",
)
assert len(MOVED) == 32 and len(set(MOVED)) == 32
FORBIDDEN_IMPORT_PREFIXES = ("sisteer.capeval.tinycap", "sisteer.capeval.serving")


def _fresh(name: str = "sampled_controls_fresh_under_test"):
    """A private, UNBOUND load of the module from its file: seam state is per module
    object, and the shared import is already bound by the driver in this process.
    Registered in sys.modules for the length of the exec (dataclasses resolve a
    class's module through it), then removed."""
    spec = importlib.util.spec_from_file_location(name, MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


def _top_level_defs(path: Path) -> set:
    out = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            out.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
    return out


def _subprocess_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


# ------------------------------------------------------------------ the driver seam
def test_an_unbound_seam_raises_on_every_use_and_names_itself():
    fresh = _fresh()
    assert set(fresh.driver_globals_bound()) == set(fresh.DRIVER_GLOBALS) == {
        "_chat_payload", "_fanout", "ccr", "DRIVER_VERSION"}
    assert "REPO" not in fresh.DRIVER_GLOBALS  # the locator's, not the seam's
    for name in fresh.DRIVER_GLOBALS:
        obj = getattr(fresh, name)
        assert isinstance(obj, fresh._Unbound)
        pattern = rf"sampled_controls\.{name} is UNBOUND.*bind_driver_globals"
        with pytest.raises(RuntimeError, match=pattern):
            obj("anything")                       # a call (the builder / the fan-out)
        with pytest.raises(RuntimeError, match=pattern):
            _ = obj.DOMAIN_OF                     # an attribute (ccr.DOMAIN_OF)
        with pytest.raises(RuntimeError, match=pattern):
            _ = {"driver": "x"}.get("driver") != obj   # the identity compare (DRIVER_VERSION)
        with pytest.raises(RuntimeError, match=pattern):
            _ = obj == "served_capability_run_v2"
        with pytest.raises(RuntimeError, match=pattern):
            bool(obj)                             # a truth test
        with pytest.raises(RuntimeError, match=pattern):
            str(obj)                              # an f-string / print
        with pytest.raises(RuntimeError, match=pattern):
            _ = obj / "scripts"                   # a path join (the REPO shape, had it been in the seam)
        # repr never raises: tracebacks and logging must be able to show the sentinel
        assert repr(obj) == f"<unbound sampled_controls.{name}>"


def _valid_bindings():
    def builder(*a, **k):
        return {"built": True}

    async def fanout(post, work, *, concurrency):
        return {}, [], {}, {}

    class CCR:
        DOMAIN_OF = {"xstest": "x"}

    return builder, fanout, CCR


def test_a_refused_bind_leaves_every_name_a_sentinel_and_names_every_offender():
    """THE assertion that would have caught the original defect: the first form validated
    and assigned name by name, so a refusal at a later name left the earlier ones bound.
    Asserted on EVERY name of DRIVER_GLOBALS, never just the first."""
    fresh = _fresh()
    builder, fanout, CCR = _valid_bindings()
    every = list(fresh.DRIVER_GLOBALS)

    def all_unbound(when):
        assert fresh.driver_globals_bound() == every, when
        for name in every:
            assert isinstance(getattr(fresh, name), fresh._Unbound), f"{name} bound {when}"

    # three offenders in ONE call -> ONE error naming all three (and not the valid ccr)
    with pytest.raises(RuntimeError) as ei:
        fresh.bind_driver_globals(_chat_payload=fresh._chat_payload, _fanout=42, ccr=CCR,
                                  DRIVER_VERSION=7)
    msg = str(ei.value)
    assert "refused, nothing was bound" in msg
    assert "_chat_payload: an unbound sentinel" in msg
    assert "_fanout: not callable (int)" in msg
    assert "DRIVER_VERSION: not a non-empty str (7)" in msg
    assert "ccr" not in msg.split("—", 1)[1]
    all_unbound("after a three-offender refusal")

    # the original defect's exact shape: three valid names, the sentinel LAST
    with pytest.raises(RuntimeError, match="DRIVER_VERSION: an unbound sentinel"):
        fresh.bind_driver_globals(_chat_payload=builder, _fanout=fanout, ccr=CCR,
                                  DRIVER_VERSION=fresh.DRIVER_VERSION)
    all_unbound("after a last-position sentinel refusal")

    # a ccr without the one attribute the bodies read; an empty DRIVER_VERSION
    class NoDomain:
        pass

    with pytest.raises(RuntimeError) as ei:
        fresh.bind_driver_globals(_chat_payload=builder, _fanout=fanout, ccr=NoDomain,
                                  DRIVER_VERSION="")
    assert "ccr: NoDomain has no DOMAIN_OF" in str(ei.value)
    assert "DRIVER_VERSION: not a non-empty str ('')" in str(ei.value)
    all_unbound("after a wrong-type refusal")

    # then the valid bind lands WHOLE: fully unbound -> fully bound, nothing between
    fresh.bind_driver_globals(_chat_payload=builder, _fanout=fanout, ccr=CCR,
                              DRIVER_VERSION="served_capability_run_v2")
    assert fresh.driver_globals_bound() == []
    assert fresh._chat_payload is builder and fresh._fanout is fanout and fresh.ccr is CCR
    assert fresh.DRIVER_VERSION == "served_capability_run_v2"
    assert fresh._chat_payload("m", "u") == {"built": True}      # bound: the real object answers


def test_bind_is_idempotent_for_the_same_objects_and_refuses_a_different_one():
    fresh = _fresh()
    builder, fanout, CCR = _valid_bindings()
    fresh.bind_driver_globals(_chat_payload=builder, _fanout=fanout, ccr=CCR,
                              DRIVER_VERSION="served_capability_run_v2")
    # the same objects again: idempotent (a second import of the same driver is not a conflict)
    fresh.bind_driver_globals(_chat_payload=builder, _fanout=fanout, ccr=CCR,
                              DRIVER_VERSION="served_capability_run_v2")
    # an EQUAL value under a different object is the same binding (strings compare by value)
    fresh.bind_driver_globals(_chat_payload=builder, _fanout=fanout, ccr=CCR,
                              DRIVER_VERSION="served_capability_run_" + "v2")

    async def other_fanout(post, work, *, concurrency):
        return {}, [], {}, {}

    with pytest.raises(RuntimeError, match=r"refusing to rebind sampled_controls\._fanout .*nothing was changed"):
        fresh.bind_driver_globals(_chat_payload=builder, _fanout=other_fanout, ccr=CCR,
                                  DRIVER_VERSION="served_capability_run_v2")
    assert fresh._fanout is fanout                                 # the first binding stands
    # every clashing name is named, not just the first
    with pytest.raises(RuntimeError) as ei:
        fresh.bind_driver_globals(_chat_payload=builder, _fanout=other_fanout, ccr=CCR,
                                  DRIVER_VERSION="served_capability_run_v3")
    assert "sampled_controls._fanout" in str(ei.value) and "sampled_controls.DRIVER_VERSION" in str(ei.value)
    assert fresh._fanout is fanout and fresh.DRIVER_VERSION == "served_capability_run_v2"


def test_the_rebind_check_runs_before_validation_and_a_refused_call_changes_nothing():
    fresh = _fresh()
    builder, fanout, CCR = _valid_bindings()
    fresh.bind_driver_globals(_chat_payload=builder, _fanout=fanout, ccr=CCR,
                              DRIVER_VERSION="served_capability_run_v2")

    async def other_fanout(post, work, *, concurrency):
        return {}, [], {}, {}

    # a clash AND an invalid value in one call: the rebind refusal fires (step 1 before step 2)
    with pytest.raises(RuntimeError, match="refusing to rebind") as ei:
        fresh.bind_driver_globals(_chat_payload=builder, _fanout=other_fanout, ccr=CCR,
                                  DRIVER_VERSION=7)
    assert "not a non-empty str" not in str(ei.value)
    # a BOUND module handed a sentinel: reported as an offender, never compared (a sentinel
    # compares by raising, and a reflected != would otherwise surface as a bare UNBOUND error)
    with pytest.raises(RuntimeError, match="DRIVER_VERSION: an unbound sentinel"):
        fresh.bind_driver_globals(_chat_payload=builder, _fanout=fanout, ccr=CCR,
                                  DRIVER_VERSION=fresh._Unbound("DRIVER_VERSION"))
    assert fresh._chat_payload is builder and fresh._fanout is fanout and fresh.ccr is CCR
    assert fresh.DRIVER_VERSION == "served_capability_run_v2"
    assert fresh.driver_globals_bound() == []


def test_the_shared_import_and_a_fresh_load_have_independent_seam_state():
    # the driver bound the shared module in this process; a fresh load starts unbound —
    # the seam is per module object, which is what lets the tests above run beside the driver
    import served_capability_run  # noqa: F401  (binds scp)
    assert scp.driver_globals_bound() == []
    assert set(_fresh().driver_globals_bound()) == set(scp.DRIVER_GLOBALS)


# ------------------------------------------------------------------ the repo locator
def test_repo_root_is_located_from_the_modules_own_file_not_the_caller_or_cwd(tmp_path):
    assert scp.REPO == REPO
    assert scp.repo_root() == REPO
    assert scp.repo_root(MODULE) == REPO
    assert scp.repo_root(REPO / "scripts" / "sdfbase") == REPO   # any start inside the checkout
    with pytest.raises(RuntimeError, match="no pyproject.toml above"):
        scp.repo_root(Path("/"))
    # a process whose cwd is an unrelated directory (and whose argv[0] is elsewhere) still
    # resolves THIS checkout: the locator reads the module's file, nothing about the caller
    r = subprocess.run(
        [sys.executable, "-c",
         "from sisteer.eval import sampled_controls as m; print(m.REPO)"],
        cwd=str(tmp_path), capture_output=True, text=True, env=_subprocess_env())
    assert r.returncode == 0, r.stderr[-1500:]
    assert Path(r.stdout.strip()) == REPO


def test_ladder_decide_loads_from_repo_and_set_repo_root_overrides_it(tmp_path):
    fresh = _fresh()
    ld = fresh._ladder_decide()
    want = REPO / "scripts" / "sdfbase" / "full22" / "sampled" / "ladder_decide.py"
    assert Path(ld.__file__).resolve() == want.resolve()
    assert callable(ld.read_items_file) and issubclass(ld.Refusal, Exception)  # what _read_items_files needs

    # the override refuses a directory that is not a checkout root
    with pytest.raises(RuntimeError, match="holds no pyproject.toml"):
        fresh.set_repo_root(tmp_path)
    assert fresh.REPO == REPO                                     # refused: unchanged

    # a pinned / relocated tree: pyproject.toml + its own ladder_decide.py
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'pinned-tree'\n")
    fake = tmp_path / "scripts" / "sdfbase" / "full22" / "sampled"
    fake.mkdir(parents=True)
    (fake / "ladder_decide.py").write_text("SENTINEL = 'the override tree'\n")
    assert fresh.set_repo_root(tmp_path) == tmp_path.resolve()
    assert fresh.REPO == tmp_path.resolve()
    assert fresh._ladder_decide().SENTINEL == "the override tree"  # the loader followed REPO
    # the override is per module object: the shared import the driver bound is untouched
    assert scp.REPO == REPO


# ------------------------------------------------------------------ the re-exports
def test_controls_cap_is_cell_controls_accessor_reads_the_new_key_and_falls_back():
    assert scp.controls_cap is cc.controls_cap                    # re-exported, not copied
    assert scp.CONTROLS_CAP_KEY is cc.CONTROLS_CAP_KEY == "controls_max_new_tokens"
    # the controls' OWN key wins over the multi-turn target's (4,096 on the wave configs)
    assert scp.controls_cap({"attacks": {"controls_max_new_tokens": 512,
                                         "target_max_new_tokens": 4096}}) == 512
    # a pre-key config: the old coupling to target_max_new_tokens, byte-identical to before
    assert scp.controls_cap({"attacks": {"target_max_new_tokens": 4096}}) == 4096
    assert scp.controls_cap({"attacks": {"target_max_new_tokens": 128},
                             "completion_tokens": 64}) == 128
    assert scp.controls_cap({"completion_tokens": 64}) == 64
    assert scp.controls_cap({}) == 512
    assert scp.controls_cap({"attacks": {}}) == 512
    for bad in (0, -1, True, "512", 512.0, None):
        with pytest.raises(SampledDecodeError, match="controls_max_new_tokens must be a positive int"):
            scp.controls_cap({"attacks": {"controls_max_new_tokens": bad}})
    # and the ladder's own config reader goes through it (the escalation refusals compare on it)
    assert cc.controls_gen_config({"attacks": {"controls_max_new_tokens": 512,
                                               "target_max_new_tokens": 4096}})["max_new_tokens"] == 512


def test_fit_provenance_key_is_sweeps_object_and_the_module_copies_neither_re_export():
    assert scp.FIT_PROVENANCE_TEMPLATE_CHECK_KEY is sweep_mod.FIT_PROVENANCE_TEMPLATE_CHECK_KEY
    assert scp._SOURCE_IDENTITY_KEYS[-1] == sweep_mod.FIT_PROVENANCE_TEMPLATE_CHECK_KEY
    defined = _top_level_defs(MODULE)
    for name in ("FIT_PROVENANCE_TEMPLATE_CHECK_KEY", "controls_cap", "CONTROLS_CAP_KEY"):
        assert name not in defined, f"sampled_controls.py DEFINES {name}; it must import it"
    assert set(MOVED) <= defined


def test_the_module_imports_nothing_from_tinycap_or_the_serving_tree():
    # the property that makes the module shareable with a tree pinned to an older serving
    # stack: none of these definitions needs tinycap or proxy_engine
    for node in ast.walk(ast.parse(MODULE.read_text())):
        mods = []
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
        for m in mods:
            assert not m.startswith(FORBIDDEN_IMPORT_PREFIXES), f"sampled_controls.py imports {m}"


# ------------------------------------------------------------------ the driver's use of it
def test_the_driver_defines_none_of_the_moved_names_and_its_names_are_the_modules_objects():
    defined = _top_level_defs(DRIVER)
    still = [n for n in MOVED if n in defined]
    assert not still, f"the driver still DEFINES {still}: two definitions of one name is the drift this module ends"
    import served_capability_run as srun
    for n in MOVED:
        assert getattr(srun, n) is getattr(scp, n), n            # the same object, not a copy
    assert srun.REPO == scp.REPO
    # the seam is bound to THIS driver's objects
    assert scp.driver_globals_bound() == []
    assert scp._chat_payload is srun._chat_payload
    assert scp._fanout is srun._fanout
    assert scp.ccr is srun.ccr
    assert scp.DRIVER_VERSION == srun.DRIVER_VERSION == "served_capability_run_v2"


def test_the_driver_binds_the_seam_before_its_main_guard_so_a_script_run_is_bound():
    body = ast.parse(DRIVER.read_text()).body
    binds = [i for i, n in enumerate(body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
             and getattr(n.value.func, "attr", None) == "bind_driver_globals"]
    assert len(binds) == 1, "the driver binds the seam exactly once"
    i_main = next(i for i, n in enumerate(body) if isinstance(n, ast.If) and "__main__" in ast.dump(n.test))
    assert binds[0] < i_main, "bind_driver_globals must run before the __main__ guard calls main()"
    # behaviour, in a process that runs the file AS __main__ (the production shape)
    probe = (
        "import runpy, sys\n"
        f"sys.argv = [{str(DRIVER)!r}, '--help']\n"
        "try:\n"
        f"    runpy.run_path({str(DRIVER)!r}, run_name='__main__')\n"
        "except SystemExit:\n"
        "    pass\n"
        "from sisteer.eval import sampled_controls as scp\n"
        "print('UNBOUND:', scp.driver_globals_bound())\n"
        "sys.exit(1 if scp.driver_globals_bound() else 0)\n")
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                       env=_subprocess_env(), cwd=str(REPO))
    assert r.returncode == 0, f"seam unbound when the driver runs as __main__:\n{r.stdout}\n{r.stderr[-1500:]}"
