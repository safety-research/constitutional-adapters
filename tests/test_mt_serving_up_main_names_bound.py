"""Every name `_main` LOADS in scripts/sdfbase/mt_serving_up.py is bound somewhere it can see — a mini-pyflakes for the
served standup's entry point. Regression for sdfbase-llama8b: the origin/main merge dropped
`want = list(cfg["explicit_cells"])` from `_main`, and every served leg died at standup with
`NameError: name 'want' is not defined` — after the load check and preflight passed, before the first request. The
unit tests never import `_main`'s body, so only a run on hardware showed it; this test reads the function's AST instead."""
from __future__ import annotations

import ast
import builtins
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "scripts" / "sdfbase" / "mt_serving_up.py"


def _bound_names(fn: ast.FunctionDef, module: ast.Module) -> set[str]:
    names: set[str] = set(dir(builtins))
    for node in module.body:  # module-level bindings visible to the function
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t in ast.walk(node):
                if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                    names.add(t.id)
    names |= {a.arg for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs}
    if fn.args.vararg:
        names.add(fn.args.vararg.arg)
    if fn.args.kwarg:
        names.add(fn.args.kwarg.arg)
    for node in ast.walk(fn):  # every binding anywhere inside the function (order-insensitive: a NameError is what we hunt)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def test_every_name_loaded_in__main_is_bound():
    module = ast.parse(SRC.read_text(encoding="utf-8"))
    fn = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "_main")
    bound = _bound_names(fn, module)
    loaded = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    unbound = sorted(loaded - bound)
    assert not unbound, f"_main loads names it never binds: {unbound}"


def test_the_pool_select_cells_draws_from_is_the_configs_explicit_cells():
    """The binding, by content: `want` is bound from cfg['explicit_cells'] inside `_main`."""
    module = ast.parse(SRC.read_text(encoding="utf-8"))
    fn = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "_main")
    binds = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "want" for t in n.targets)]
    assert binds, "_main never binds `want`"
    src = ast.get_source_segment(SRC.read_text(encoding="utf-8"), binds[0])
    assert 'cfg["explicit_cells"]' in src, src
