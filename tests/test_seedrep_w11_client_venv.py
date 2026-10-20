"""The Inspect-side steps run under the tinycap CLIENT venv, never the driver venv.

The GSM8K children (served_capability_run.py -> inspect_served -> inspect_ai / inspect_evals.gsm8k) and the G-FIT budget tool
(tinycap_oos_prompt_budget.py imports inspect_ai) were composed with {driver_python}; the driver venv (`uv sync --frozen --extra cu128`)
carries no inspect_ai (only the `petri` dependency group does), so every one of them would have died at import in the container. Now a
child's own `task` (an `inspect/...` row task) or `budget_argv` says it needs the CLIENT venv: the plan names {client_python}, the app
builds the venv (f22_client_venv, ported) for such a row, `resolve_spawn` substitutes it and refuses the old resolution by name.
Synthetic ids only.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _seedrep_testlib as T  # noqa: E402
from test_seedrep_w9_container_paths import _stub_modal  # noqa: E402

REPO = T.REPO
DRIVER = REPO / "scripts" / "modal" / "seedrep_served.py"
_ABSENT = object()
sys.path.insert(0, str(REPO / "scripts" / "modal"))
import seedrep_launch_plan as lp  # noqa: E402

W = "2" * 40


@pytest.fixture(scope="module")
def srv():
    spec = importlib.util.spec_from_file_location("seedrep_served_w11_client_venv", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    prior = sys.modules.get("modal", _ABSENT)
    sys.modules["modal"] = _stub_modal()
    try:
        spec.loader.exec_module(mod)
    finally:
        if prior is _ABSENT:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prior
    return mod


def _ctx(**kw):
    base = dict(attempt_root="/workspace/attempt", attempt_id="ta-1", standup_id="sid", plans_root="/vol/p", proxy_url="http://127.0.0.1:9/v1",
                routing_table="/t.json", driver_python="/home/seedrep/venvs/f22-x/bin/python", S=21, ncells=8, bank_view="/b", base_env={})
    base.update(kw)
    return lp.SpawnContext(**base)


def _gsm8k_child(argv0):
    return {"eval_leg": "gsm8k", "arm": "none", "cell": "none", "s": 1, "task": lp.GSM8K_ROW_TASK, "concurrency_rule": "S", "cap": 8192,
            "argv": [argv0, "/t/scripts/served_capability_run.py", "generate", "--out-root", f"{lp.ATTEMPT_ROOT_PLACEHOLDER}/exp/x"]}


def _static_child(argv0=lp.DRIVER_PY_PLACEHOLDER):
    return {"eval_leg": "pasttense", "arm": "none", "cell": "none", "s": 1, "task": "static/pasttense", "concurrency_rule": "one_request_each_S_drivers",
            "argv": [argv0, "/t/scripts/jailbreak_run.py", "generate"], "cap": 4096}


class TestPlanNamesTheClientPython:
    def test_the_generated_gsm8k_child_and_budget_argv_name_the_client_placeholder(self):
        b = lp.gsm8k_budget_argv(W=W, cap=8192, geometry_eval="capability:x", config_rel="configs/c.yaml", cell="none", budget_rel="exp/budgets/b.json")
        assert b[0] == lp.CLIENT_PY_PLACEHOLDER and b[1].endswith("/scripts/tinycap_oos_prompt_budget.py")
        src = inspect.getsource(lp._gsm8k_children)
        assert 'argv = [CLIENT_PY_PLACEHOLDER, f"{tree}/scripts/served_capability_run.py", "generate"' in src
        assert lp.CLIENT_PY_PLACEHOLDER in lp.SPAWN_PLACEHOLDERS and lp.CLIENT_PY_PLACEHOLDER != lp.DRIVER_PY_PLACEHOLDER

    def test_the_need_is_read_off_the_child_not_a_leg_list(self):
        assert lp.child_needs_client_python(_gsm8k_child(lp.CLIENT_PY_PLACEHOLDER)) and not lp.child_needs_client_python(_static_child())
        assert lp.row_needs_client_venv({"leg": "gsm8k", "children": [_gsm8k_child(lp.CLIENT_PY_PLACEHOLDER)]})
        assert lp.row_needs_client_venv({"leg": "anything", "children": [{**_static_child(), "budget_argv": [lp.CLIENT_PY_PLACEHOLDER, "tool"]}]})
        assert not lp.row_needs_client_venv({"leg": "pasttense", "children": [_static_child()]})
        assert not lp.row_needs_client_venv({"leg": "gsm8k", "children": []})

    def test_known_bad_a_pre_w11_plan_is_a_named_problem_and_validate_plan_carries_it(self):
        pre = {"standups": [{"standup_id": "gsm8k__s1__base__shard00", "children": [{**_gsm8k_child(lp.DRIVER_PY_PLACEHOLDER),
                                                                                     "budget_argv": [lp.DRIVER_PY_PLACEHOLDER, "/t/scripts/tinycap_oos_prompt_budget.py"]}]}],
               "rungs": [], "s5_probe_rows": []}
        probs = lp.client_python_problems(pre)
        assert len(probs) == 2 and "argv[0] is '{driver_python}', not {client_python}" in probs[0] and "budget_argv[0] is '{driver_python}'" in probs[1]
        assert lp.client_python_problems({"standups": [{"standup_id": "x", "children": [_gsm8k_child(lp.CLIENT_PY_PLACEHOLDER), _static_child()]}]}) == []
        assert "problems += client_python_problems(plan)" in inspect.getsource(lp.validate_plan)


class TestResolveSpawn:
    def test_a_gsm8k_child_resolves_to_the_client_python(self):
        argv, _ = lp.resolve_spawn(_gsm8k_child(lp.CLIENT_PY_PLACEHOLDER), _ctx(client_python="/home/seedrep/venvs/tinycap-client-abcd1234/bin/python"))
        assert argv[0] == "/home/seedrep/venvs/tinycap-client-abcd1234/bin/python" and "{" not in " ".join(argv)

    def test_known_bad_without_a_client_python_the_resolution_refuses_by_name(self):
        with pytest.raises(ValueError, match="runs under the tinycap CLIENT venv .* the spawn context carries no client python"):
            lp.resolve_spawn(_gsm8k_child(lp.CLIENT_PY_PLACEHOLDER), _ctx())

    def test_known_bad_the_old_resolution_to_the_driver_python_is_refused_by_name(self):
        """37762b566..bc1325393's plans: {driver_python} on an `inspect/` child -> the driver venv, which has no inspect_ai."""
        with pytest.raises(ValueError, match="resolved to the DRIVER python .* carries no inspect_ai"):
            lp.resolve_spawn(_gsm8k_child(lp.DRIVER_PY_PLACEHOLDER), _ctx(client_python="/cpy"))

    def test_a_static_child_still_resolves_to_the_driver_python_without_a_client_python(self):
        argv, _ = lp.resolve_spawn(_static_child(), _ctx())
        assert argv[0] == "/home/seedrep/venvs/f22-x/bin/python"

    def test_child_argv_env_threads_the_client_python(self, srv):
        argv, _ = srv.child_argv_env(_gsm8k_child(lp.CLIENT_PY_PLACEHOLDER), attempt_root="/workspace/attempt", plans_root="/vol/p", proxy_url="u",
                                     routing_table="t", driver_python="/dpy", S=21, ncells=8, bank_view="/b", base_env={}, client_python="/cpy")
        assert argv[0] == "/cpy"


class TestTheAppBuildsAndUsesTheClientVenv:
    def test_run_standup_row_builds_the_client_venv_for_such_a_row_before_the_budget_prestep(self, srv):
        src = inspect.getsource(srv._run_standup_row)
        i = src.index("if lp.row_needs_client_venv(row):")
        assert "ensure_client_venv(tree)" in src[i:i + 400] and 'return fail("CLIENT_VENV_FAILED"' in src[i:i + 500]
        assert i < src.index("gsm8k_budget_prestep(kids") < src.index("spawn_context = lp.SpawnContext(")
        assert "client_python=client_python" in src[src.index("gsm8k_budget_prestep(kids"):src.index("gsm8k_budget_prestep(kids") + 400]
        assert "client_python=client_python" in src[src.index("spawn_context = lp.SpawnContext("):src.index("spawn_context = lp.SpawnContext(") + 600]

    def test_budget_prestep_runs_the_tool_under_the_client_python_and_refuses_without_one(self, srv, tmp_path):
        calls = []

        def fake_run(argv, **kw):
            calls.append(list(argv))
            out = Path(argv[argv.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            Path(str(out) + ".fit.json").write_text(json.dumps({"fits": True, "cap": 8192, "geometry_eval": "capability:x", "arm": "on", "tree_sha": W}))
            return types.SimpleNamespace(returncode=0)
        kid = {"arm": "none", "cell": "none", "s": 1, "cap": 8192, "budget_rel": "exp/budgets/b.json",
               "budget_argv": lp.gsm8k_budget_argv(W=W, cap=8192, geometry_eval="capability:x", config_rel="c.yaml", cell="none", budget_rel="exp/budgets/b.json")}
        probs = srv.gsm8k_budget_prestep([kid], root=tmp_path, driver_python="/dpy", tree=REPO, env={}, W=W, geometry_eval="capability:x", run=fake_run,
                                         client_python="/cpy")
        assert calls and calls[0][0] == "/cpy" and "/dpy" not in " ".join(calls[0])
        assert not probs or all("budget tool rc" not in p_ for p_ in probs)          # the sidecar check may name fields; the TOOL ran under the client python
        calls.clear()
        probs = srv.gsm8k_budget_prestep([kid], root=tmp_path / "b", driver_python="/dpy", tree=REPO, env={}, W=W, geometry_eval="capability:x", run=fake_run)
        assert calls == [] and len(probs) == 1 and "needs the tinycap CLIENT venv python" in probs[0]

    def test_the_client_venv_guard_imports_inspect_under_the_client_python_not_the_driver(self, srv, tmp_path, monkeypatch):
        """A driver venv without inspect_ai (the stub) is never consulted: the guard is run under the CLIENT venv's python."""
        monkeypatch.setattr(srv, "VENVS_LOCAL", str(tmp_path / "venvs"))
        seen = []

        def fake_run(argv, **kw):
            seen.append(list(argv))
            if argv[:2] == ["uv", "venv"]:
                Path(argv[-1], "bin").mkdir(parents=True)
                Path(argv[-1], "bin", "python").write_text("")
            if argv[1:2] == ["-c"] and "inspect_ai" in argv[2] and "tinycap-client-" not in argv[0]:
                return types.SimpleNamespace(returncode=1, stdout="", stderr="ModuleNotFoundError: No module named 'inspect_ai'")
            return types.SimpleNamespace(returncode=0, stdout="client venv OK\n", stderr="")
        out = srv.ensure_client_venv(REPO, run=fake_run)
        guard = [a for a in seen if a[1:2] == ["-c"]]
        assert out["built"] and len(guard) == 1 and "tinycap-client-" in guard[0][0] and guard[0][0].endswith("/bin/python")
        assert srv.client_venv_python(REPO) == guard[0][0]
