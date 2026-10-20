"""The tinycap control contracts after the tiny loglikelihood battery was removed.

The ``tiny_ll`` family (five tinyBenchmarks tasks scored by continuation loglikelihood), its
third-party tier-1 harness and the vendored task files it ran on are gone. These tests pin what is
left: the contracts module imports and exports only names that exist, every retained control
expands and dispatches, every control-keyed map covers exactly the retained controls, the removed
family is refused by name, nothing under ``src/``, ``scripts/`` or ``reproduce/`` imports the
removed modules, and every remaining reference in the tree is one the reference inventory keeps.
CPU only; no network.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import re
import sys
import warnings
from pathlib import Path

import pytest

from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]

#: The removed family, whole and by each former sub-task.
REMOVED_TOKENS = ("tiny_ll", "tiny_ll:tinyMMLU", "tiny_ll:tinyArc", "tiny_ll:tinyHellaswag",
                  "tiny_ll:tinyWinogrande", "tiny_ll:tinyTruthfulQA")
#: Contract names that went with it.
REMOVED_NAMES = ("TINY_LL_TASKS", "CONTROL_TINY_LL", "SERVED_LL_TASK_PREFIX", "SERVED_LL_TASK_VERSION",
                 "served_ll_task", "SERVED_LL_TASKS", "SERVED_LL_EXTRA_FIELDS", "SERVED_LL_CANDIDATE_FIELDS",
                 "DOMAIN_SERVED_LL", "LM_EVAL_PIN", "TINYBENCHMARKS_PIN", "TINYBENCHMARKS_GIT",
                 "TINYBENCHMARKS_REV", "FRAMEWORK_VCS_PINS", "PREFILL_SCORED_TASKS", "PREFILL_SAMPLES_PER_ITEM",
                 "EVIDENCE_CONTINUATION_BITES_KEY", "CONTAINER_HF_SPOTCHECK", "METRIC_ACC_NORM",
                 "METRIC_MC2_TRUE_MASS", "METRIC_ACC_NORM_AGGREGATE", "METRIC_IRT_GPIRT_AGGREGATE")
#: Modules nothing may import any more.
BANNED_MODULES = ("lm_eval", "tinyBenchmarks", "sisteer.capeval.tiers.served_tiny",
                  "sisteer.capeval.tiers.lmeval_runner", "sisteer.capeval.serving.vllm_inproc")


def test_contracts_import_and_every_exported_name_resolves():
    mod = importlib.import_module("sisteer.capeval.tinycap.contracts")
    missing = [n for n in mod.__all__ if not hasattr(mod, n)]
    assert not missing, f"__all__ names that do not resolve: {missing}"
    for name in REMOVED_NAMES:
        assert name not in mod.__all__ and not hasattr(mod, name), name


def test_every_retained_control_and_subtask_expands_and_the_removed_family_is_refused():
    atoms = []
    for control in C.TINYCAP_CONTROLS:
        expanded = C.expand_control(control)
        assert expanded and all(a in C.CONTROL_ATOMS for a in expanded), (control, expanded)
        for atom in expanded:
            assert C.expand_control(atom) == (atom,)
            assert C.control_family(atom) == control
        atoms += expanded
    assert tuple(atoms) == C.CONTROL_ATOMS
    for sub in C.INSPECT_CMP_TASKS:
        token = f"{C.CONTROL_INSPECT_CMP}{C.CONTROL_SUBTASK_SEP}{sub}"
        assert C.expand_control(token) == (token,)
    for spelling, canonical in C.RETAINED_CONTROL_SPELLINGS.items():
        assert C.expand_control(spelling) == C.expand_control(canonical)
    for token in REMOVED_TOKENS:
        with pytest.raises(C.ControlNameError, match=re.escape(f"unknown control {token!r}")):
            C.expand_control(token)


def test_every_dispatch_module_imports():
    assert set(C.DISPATCH_MODULE_OF_CONTROL) == set(C.TINYCAP_CONTROLS)
    for control, module in C.DISPATCH_MODULE_OF_CONTROL.items():
        assert module not in BANNED_MODULES, (control, module)
        importlib.import_module(module)


def test_control_keyed_maps_cover_exactly_the_retained_controls():
    controls = set(C.TINYCAP_CONTROLS)
    assert set(C.TASKS_OF_CONTROL) == set(C.CONTROL_ATOMS)
    assert {C.control_family(a) for a in C.TASKS_OF_CONTROL} == controls
    assert set(C.GEOMETRY_CLASS_OF_CONTROL) == controls
    assert C.GEOMETRY_CLASS_TINY not in C.GEOMETRY_CLASS_OF_CONTROL.values()
    retired = {C.inspect_task(n) for n in C.RETIRED_INSPECT_TASK_NAMES}
    produced = {t for tasks in C.TASKS_OF_CONTROL.values() for t in tasks}
    assert produced | retired == set(C.TINYCAP_TASKS) and not produced & retired
    # the thinking-mode map and the other row-task maps are total over the row tasks, nothing more
    for name in ("THINKING_BY_TASK", "TINYCAP_DOMAIN_OF", "SAMPLES_BY_TASK"):
        assert set(getattr(C, name)) == set(C.TINYCAP_TASKS), name
    assert not any(t.startswith("served_ll/") for t in C.TINYCAP_TASKS)
    assert set(C.TINYCAP_DOMAIN_OF.values()) == set(C.TINYCAP_DOMAINS) == {"inspect_gen", "evalplus_served"}
    for battery in (C.STANDARD_BATTERY_PROMPTED, C.STANDARD_BATTERY_FITTED, C.IN_DEPTH_EXTRAS):
        for token in battery:
            assert set(C.expand_control(token)) <= set(C.CONTROL_ATOMS), token
            assert C.control_family(C.expand_control(token)[0]) in controls, token
    # the client lock's pins and the frameworks the driver enforces name no removed package
    assert set(C.FRAMEWORK_PINS) == {"inspect_ai", "inspect_evals", "evalplus"}
    assert set(C.FRAMEWORK_PINS) == set(C.FRAMEWORKS_BLOCK_FIELDS) - {"client_lock_sha256"}
    for gone in ("lm-eval", "lm_eval", "tinybenchmarks"):
        assert gone not in C.CLIENT_TOP_LEVEL_PINS
        assert not any(line.startswith(f"{gone} @ ") for line in C.CLIENT_VCS_PINS)


def _imported_modules(node: ast.AST):
    """Every module an import statement names, with ``from a import b`` also as ``a.b``."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield alias.name
    elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
        yield node.module
        for alias in node.names:
            yield f"{node.module}.{alias.name}"


_CODE_IMPORT = re.compile(r"\bimport\s+([A-Za-z_][\w.]*(?:\s*,\s*[A-Za-z_][\w.]*)*)")
_CODE_FROM = re.compile(r"\bfrom\s+([A-Za-z_][\w.]*)\s+import\s+([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)")


def _string_imports(text: str):
    """The modules an import statement embedded in a string literal names (code run with ``-c``)."""
    for m in _CODE_IMPORT.finditer(text):
        for name in m.group(1).split(","):
            yield name.strip()
    for m in _CODE_FROM.finditer(text):
        yield m.group(1)
        for name in m.group(2).split(","):
            yield f"{m.group(1)}.{name.strip()}"


def _banned(module: str) -> bool:
    return any(module == b or module.startswith(b + ".") for b in BANNED_MODULES)


def test_nothing_imports_the_removed_harness_or_its_modules():
    offenders = []
    for root in ("src", "scripts", "reproduce"):
        for path in sorted((REPO / root).rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            for node in ast.walk(tree):
                names = list(_imported_modules(node))
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    names += list(_string_imports(node.value))
                offenders += [(rel, getattr(node, "lineno", 0), n) for n in names if _banned(n)]
    assert not offenders, offenders
    launcher = ast.parse((REPO / "scripts" / "modal" / "tinycap_served.py").read_text(encoding="utf-8"))
    assigned = {t.id for n in ast.walk(launcher) if isinstance(n, (ast.Assign, ast.AnnAssign))
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(t, ast.Name)}
    assert "TINY_LOADER_CODE" not in assigned
    for rel in ("src/sisteer/capeval/tiers/served_tiny.py", "src/sisteer/capeval/tiers/lmeval_runner.py",
                "src/sisteer/capeval/serving/vllm_inproc.py", "configs/capeval/tier1.yaml",
                "configs/capeval/tasks/capeval_tiny_utils.py", "configs/capeval/tasks/tinyBenchmarks.pkl"):
        assert not (REPO / rel).exists(), rel


def test_the_import_scanner_catches_each_import_shape():
    """The scanner above is only as good as its parsing: each way the removed modules were reached
    is caught, and a mention in prose is not an import."""
    cases = {
        "import lm_eval": True, "import lm_eval.api.model": True, "from lm_eval import simple_evaluate": True,
        "from sisteer.capeval.tiers import served_tiny as st": True, "import a, tinyBenchmarks, b": True,
        "from sisteer.capeval.serving.vllm_inproc import VllmInprocEngine": True,
        "import inspect_ai, inspect_evals, evalplus": False, "the lm_eval harness was removed": False,
        "from sisteer.capeval.tiers import evalplus_runner": False,
    }
    for src, want in cases.items():
        found = [n for n in _string_imports(src) if _banned(n)]
        assert bool(found) is want, (src, found)
        tree = ast.parse(src) if not src.startswith("the ") else ast.parse(repr(src))
        found_ast = [n for node in ast.walk(tree) for n in _imported_modules(node) if _banned(n)]
        assert bool(found_ast) is (want and not src.startswith("the ")), (src, found_ast)


def _inventory_module():
    name = "_release_inventory_refs"
    if name in sys.modules:
        return sys.modules[name]
    path = REPO / "scripts" / "release" / "inventory_refs.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # registered before it runs: its dataclasses look their module up there
    spec.loader.exec_module(mod)
    return mod


def test_every_remaining_reference_is_one_the_inventory_keeps():
    inv = _inventory_module()
    rows = inv.inventory(REPO)
    assert rows, "the inventory found nothing at all, so it scanned nothing"
    bad = [(h.path, h.lineno, cls, h.text.strip()[:120]) for h, cls, _ in rows if cls not in inv.KEEP_CLASSES]
    assert not bad, bad


def test_the_inventory_classifies_against_a_target_and_matches_at_identifier_boundaries():
    inv = _inventory_module()
    files = {"a.py": b"x = CONTROL_TINY_LL\ny = 'tiny_llama'\nz = 'served_ll/tinyArc'\n",
             "configs/registry/evals.yaml": b"- eval_id: tiny_ll\n"}
    hits = list(inv.scan(files))
    assert [(h.path, h.lineno) for h in hits] == [("a.py", 1), ("a.py", 3), ("configs/registry/evals.yaml", 1)]
    assert hits[1].tokens == ("tinyArc", "served_ll/")
    target = {"a.py": b"x = CONTROL_TINY_LL\n", "configs/registry/evals.yaml": b"- eval_id: tiny_ll\n"}
    classes = [inv.classify(h, target, {})[0] for h in hits]
    assert classes == ["unclassified", "edit", "keep-history"]
    assert inv.classify(hits[0], {}, {})[0] == "delete"
