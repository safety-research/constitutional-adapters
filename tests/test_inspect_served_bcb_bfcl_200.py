"""bcb-bfcl-200 in the Inspect provider: the two pinned legs' plans
(the thin BigCodeBench task narrowed to the pin / a selection / a smoke head; the
stock BFCL narrowed by sample_id to the pin), the tool-call guards widened to
``bfcl_200``, the BigCodeBench row facts classified by RECONSTRUCTING the pinned
scorer's exact explanation grammar (fixtures built by the scorer's own
constructor), the summary counts, ``max_sandboxes`` reaching ``eval_async`` on
a sandboxed plan only, the strict errored-log refusal with its diagnostics
file, and the ladder's plan-level hooks (``cap_override`` / ``selection``).

Needs inspect_ai + inspect_evals (the client lock)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("inspect_ai", reason="tinycap client venv (hashed lock)")

from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402
from inspect_ai.scorer import Score  # noqa: E402

from sisteer.capeval.tiers import inspect_served as isv  # noqa: E402
from sisteer.capeval.tinycap import completion  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402

from test_inspect_served import (  # noqa: E402
    FIXTURE_ROUTES, LENS_ROUTE, _Emitter, _Log, _Sample, _Score, _Store, _fake_eval, _model_output, _plan)

REPO = isv.REPO_ROOT
PIN_B = C.load_bcb_pin(REPO)
PIN_F = C.load_bfcl_pin(REPO)
LOG_IDS_B = [C.bcb_log_id(i) for i in PIN_B["ids"]]


# --------------------------------------------------------------------------- #
# guards and tables                                                            #
# --------------------------------------------------------------------------- #


def test_the_tool_call_guards_cover_bfcl_200():
    # the full `bfcl` retired 2026-09-11; the pinned FULL sibling was added 2026-09-23
    assert isv.TOOL_CALL_TASK_NAMES == ("bfcl_200", "bfcl_full")
    assert isv._bfcl_requested(["bfcl_200"]) and not isv._bfcl_requested(["math"])
    ok = dict(agent_loop_entry=True, tool_calls=True)
    isv.assert_route_can_serve(["bfcl_200", "bigcodebench_200"], LENS_ROUTE, **ok)
    isv.assert_route_can_serve(["bigcodebench_200"], LENS_ROUTE)                    # no tools: nothing consulted
    with pytest.raises(isv.InspectServedError, match="composed no --tool-calls"):
        isv.assert_route_can_serve(["bfcl_200"], FIXTURE_ROUTES[1], agent_loop_entry=True, tool_calls=False)
    with pytest.raises(isv.InspectServedError, match="agent-loop"):
        isv.assert_route_can_serve(["bfcl_200"], LENS_ROUTE, agent_loop_entry=False, tool_calls=True)
    assert isv.PRIMARY_SCORER["bigcodebench_200"] == "verify"
    # the contracts' pin roster IS the battery's single-turn category list (the pin loader holds the pin to it)
    assert isv.bfcl_categories() == list(C.BFCL_CATEGORIES)
    assert isv.SANDBOXED_TASK_NAMES == ("bigcodebench_200", "bigcodebench_full")
    assert isv.PRIMARY_SCORER["bigcodebench_full"] == "verify"
    assert isv.task_max_tokens("bigcodebench_full", thinking="off") == (4096, "off_table")
    assert isv.task_max_tokens("bfcl_full", thinking="off") == (1024, "off_table")
    assert isv.task_max_tokens("bigcodebench_200", thinking="off") == (4096, "off_table")
    assert isv.task_max_tokens("bfcl_200", thinking="off") == (1024, "off_table")
    assert isv.task_max_tokens("bigcodebench_200", thinking="on") == (8192, "on_table")


# --------------------------------------------------------------------------- #
# the plans                                                                    #
# --------------------------------------------------------------------------- #


def _fake_task(ids):
    """A built Inspect-shaped task over ``ids`` (the shape _dataset_ids reads)."""
    from types import SimpleNamespace
    return SimpleNamespace(dataset=MemoryDataset([Sample(id=i, input="x") for i in ids]),
                           sandbox=("modal_owned", "/x/compose.yaml"), scorer=None)


def _stub_bcb_builder(monkeypatch, seen):
    def _build(ids=None, sandbox=C.BCB_SANDBOX_PROVIDER, repo_root=None, pin=None):
        seen.append({"ids": list(ids or []), "sandbox": sandbox, "pin_n": (pin or {}).get("n")})
        return _fake_task([int(C.bcb_log_id(i)) for i in ids])
    monkeypatch.setattr(isv, "load_repo_task",
                        lambda name: _build if name == "bigcodebench_200" else pytest.fail(name))


def test_bigcodebench_200_plans_the_pin_a_smoke_head_and_a_selection(monkeypatch):
    seen: List[Dict[str, Any]] = []
    _stub_bcb_builder(monkeypatch, seen)
    kw = dict(grader_model=None, thinking="off", repo_root=REPO)
    plan = isv.plan_task("bigcodebench_200", **kw)
    assert plan.row_task == "inspect/bigcodebench_200" and plan.expected_n == 200 == plan.n_run
    assert plan.sample_ids == LOG_IDS_B and plan.limit is None and plan.selection is None
    assert plan.max_tokens == 4096 and plan.cap_source == "off_table" and plan.primary_scorer == "verify"
    assert plan.item_set == "pinned200" and plan.samples_per_item == 1
    assert plan.sandbox == {"provider": "modal_owned", "image": C.BCB_SANDBOX_IMAGE, "cpu": 1.0,
                            "memory_mib": 512, "block_network": True, "timeout_s": 10_800,
                            "verify_timeout_s": 30, "app_name": "inspect_modal_sandbox",
                            "binding_version": C.CPU_VERIFY_BINDING_VERSION}      # the generation declares its binding contract (#441)
    assert seen[-1]["ids"] == PIN_B["ids"] and seen[-1]["sandbox"] == "modal_owned" and seen[-1]["pin_n"] == 200
    # a smoke cap heads the PIN before building (fewer sandboxes created), n_run set, limit_plan a no-op
    smoke = isv.plan_task("bigcodebench_200", inspect_limit=5, **kw)
    assert smoke.sample_ids == LOG_IDS_B[:5] and smoke.n_run == 5 and smoke.expected_n == 200 and smoke.limit is None
    assert seen[-1]["ids"] == PIN_B["ids"][:5]
    assert isv.limit_plan(smoke, 5) == smoke
    # the ladder's selection: a subset of the pin, re-ordered to pin order, at the override cap
    sel = [LOG_IDS_B[7], LOG_IDS_B[2], LOG_IDS_B[40]]
    rung = isv.plan_task("bigcodebench_200", cap_override=16384, selection=sel, **kw)
    assert rung.sample_ids == [LOG_IDS_B[2], LOG_IDS_B[7], LOG_IDS_B[40]] == rung.selection
    assert rung.n_run == 3 and rung.expected_n == 200 and rung.max_tokens == 16384
    assert rung.cap_source == isv.CAP_SOURCE_OVERRIDE == "cap_override"
    assert "ladder selection of 3 capped items at cap 16384" in rung.note
    # refusals: a selection without an override; a foreign id; an override at or below the table
    with pytest.raises(isv.InspectServedError, match="without a cap override"):
        isv.plan_task("bigcodebench_200", selection=sel, **kw)
    with pytest.raises(isv.InspectServedError, match="not a set of pinned ids"):
        isv.plan_task("bigcodebench_200", cap_override=16384, selection=["1"], **kw)
    with pytest.raises(isv.InspectServedError, match="not above the table's cap"):
        isv.plan_task("bigcodebench_200", cap_override=4096, selection=sel, **kw)
    with pytest.raises(isv.InspectServedError, match="truncation ladder's knobs"):
        isv.plan_task("gsm8k_paired200", cap_override=16384, **dict(kw, verify_pairs=False))


def _bfcl_module():
    """The MODULE inspect_evals.bfcl.bfcl (the package's __init__ re-exports the
    task function under the same name, so the dotted string resolves to the
    function — the plan imports the module's attribute, so that is what to stub)."""
    import importlib
    return importlib.import_module("inspect_evals.bfcl.bfcl")


def test_bfcl_200_plans_the_pin_by_sample_id_and_verifies_every_id(monkeypatch):
    population = [i for ids in PIN_F["population"].values() for i in ids]
    built: List[List[str]] = []

    def _bfcl_task(categories):
        built.append(list(categories))
        return _fake_task(population)
    monkeypatch.setattr(_bfcl_module(), "bfcl", _bfcl_task)
    kw = dict(grader_model=None, thinking="off", repo_root=REPO)
    plan = isv.plan_task("bfcl_200", **kw)
    assert plan.row_task == "inspect/bfcl_200" and plan.sample_ids == PIN_F["ids"]
    # an id-selected plan: limit_plan at limit 0 sets n_run to the full count, no eval-level limit
    assert plan.expected_n == 200 and plan.n_run == 200 and plan.limit is None
    assert plan.max_tokens == 1024 and plan.sandbox is None and plan.item_set == "pinned200"
    assert built[-1] == isv.bfcl_categories() and len(built[-1]) == 18
    # a smoke cap takes the FIRST N pinned ids (limit_plan on an id-selected plan)
    smoke = isv.plan_task("bfcl_200", inspect_limit=20, **kw)
    assert smoke.sample_ids == PIN_F["ids"][:20] and smoke.n_run == 20 and smoke.expected_n == 200
    # the ladder selection, in pin order
    sel = [PIN_F["ids"][5], PIN_F["ids"][1]]
    rung = isv.plan_task("bfcl_200", cap_override=16384, selection=sel, **kw)
    assert rung.sample_ids == [PIN_F["ids"][1], PIN_F["ids"][5]] and rung.max_tokens == 16384
    # a dataset drift (population != the pin's) refuses before any selection check
    monkeypatch.setattr(_bfcl_module(), "bfcl", lambda categories: _fake_task(population[:-1]))
    with pytest.raises(isv.InspectServedError, match="not the pin's population 3981"):
        isv.plan_task("bfcl_200", **kw)
    # a pin id absent from the built task refuses by name
    monkeypatch.setattr(_bfcl_module(), "bfcl",
                        lambda categories: _fake_task([i for i in population if i != PIN_F["ids"][0]] + ["extra_0"]))
    with pytest.raises(isv.InspectServedError, match="only 199/200 pinned ids"):
        isv.plan_task("bfcl_200", **kw)


def test_plans_for_atoms_threads_the_ladder_maps_per_task_and_refuses_strays(monkeypatch):
    seen: List[Dict[str, Any]] = []
    _stub_bcb_builder(monkeypatch, seen)
    population = [i for ids in PIN_F["population"].values() for i in ids]
    monkeypatch.setattr(_bfcl_module(), "bfcl", lambda categories: _fake_task(population))
    kw = dict(grader_model=None, thinking="off", repo_root=REPO)
    plans = isv.plans_for_atoms(["bigcodebench_200", "bfcl_200"], **kw)
    assert [p.row_task for p in plans] == ["inspect/bigcodebench_200", "inspect/bfcl_200"]
    assert all(p.cap_source == "off_table" for p in plans)
    ladder = isv.plans_for_atoms(
        ["bigcodebench_200", "bfcl_200"],
        cap_overrides_by_task={"inspect/bigcodebench_200": 16384},
        item_selection_by_task={"inspect/bigcodebench_200": LOG_IDS_B[:3]}, **kw)
    assert ladder[0].max_tokens == 16384 and ladder[0].selection == LOG_IDS_B[:3] and ladder[0].n_run == 3
    assert ladder[1].max_tokens == 1024 and ladder[1].selection is None       # the other leg untouched
    with pytest.raises(isv.InspectServedError, match="which this run does not plan"):
        isv.plans_for_atoms(["bfcl_200"], cap_overrides_by_task={"inspect/bigcodebench_200": 16384}, **kw)
    with pytest.raises(isv.InspectServedError, match="with no cap override"):
        isv.plans_for_atoms(["bfcl_200"], item_selection_by_task={"inspect/bfcl_200": PIN_F["ids"][:2]}, **kw)


# --------------------------------------------------------------------------- #
# the BigCodeBench row facts — fixtures from the scorer's OWN constructor       #
# --------------------------------------------------------------------------- #

TEST_SRC = "import unittest\nclass T(unittest.TestCase):\n    def test_x(self):\n        self.assertEqual(f(), 1)\n"


def _scorer_explanation(answer: str, test: str, stderr: str | None) -> str:
    """inspect_evals 0.18.0's ``verify()`` explanation, built the way the scorer
    builds it (the code list joined, the fence, then the verdict)."""
    code = [answer, "\n", test, "\n", "\nif __name__ == '__main__':\n    unittest.main()"]
    explanation = "".join(["The following verification code was executed:\n\n"] + ["```python\n\n"] + code + ["\n```\n"])
    if stderr is None:
        return explanation + "The submission was correct."
    return explanation + f"The submission was incorrect with the following error: \n\n{stderr}"


def _bcb_sample(sid, completion_text, answer, stderr=None, *, test=TEST_SRC):
    s = _Sample(str(sid), {"verify": Score(value="C" if stderr is None else "I", answer=answer,
                                           explanation=_scorer_explanation(answer, test, stderr))},
                output=_model_output(completion_text))
    s.metadata = {"test": test, "libs": "['os']", "task_id_metadata": f"BigCodeBench/{sid}"}
    return s


def test_the_grammar_constants_are_the_pinned_scorers_own():
    """The reconstruction is pinned against the installed scorer's SOURCE, so a
    pin bump that moves a literal fails here, not on a wrong number."""
    import importlib
    import inspect as _inspect
    m = importlib.import_module("inspect_evals.bigcodebench.bigcodebench")

    src = _inspect.getsource(m.verify)
    for lit in ('"The following verification code was executed:\\n\\n"', '"```python\\n\\n"',
                '"\\n```\\n"', '"The submission was correct."',
                '"The submission was incorrect with the following error: \\n\\n{result.stderr}"',
                '"Verification timed out."', "if __name__ == '__main__':", "unittest.main()"):
        assert lit in src, lit
    assert m.VERIFY_TIMEOUT == C.BCB_VERIFY_TIMEOUT_S == 30
    assert isv.bcb_verify_preamble("x=1", "T") == "The following verification code was executed:\n\n```python\n\nx=1\nT\n\nif __name__ == '__main__':\n    unittest.main()\n```\n"
    # the stock constructor's docker dependency the thin task substitutes (bigcodebench_200.py)
    assert "get_images_from_compose(default_compose_file)" in _inspect.getsource(m.bigcodebench)


def test_bcb_row_facts_classify_every_production_shape_and_refuse_a_moved_grammar():
    ok = _bcb_sample(4, "```python\ndef f():\n    return 1\n```", "def f():\n    return 1\n")
    facts = isv.bcb_row_facts(ok, ok.scores["verify"])
    assert facts == {"code_source": "python_fence", "sandbox_status": "ok", "verifier_explanation": None}
    tb = _bcb_sample(8, "```\ndef f():\n    return 2\n```", "def f():\n    return 2\n",
                     stderr="Traceback (most recent call last):\n  ...\nAssertionError: 2 != 1\n")
    facts = isv.bcb_row_facts(tb, tb.scores["verify"])
    assert facts["code_source"] == "bare_fence" and facts["sandbox_status"] == "ok"
    assert facts["verifier_explanation"].endswith("AssertionError: 2 != 1\n")
    empty = _bcb_sample(18, "def f():\n    return 3", "def f():\n    return 3", stderr="")
    assert isv.bcb_row_facts(empty, empty.scores["verify"]) == {
        "code_source": "whole_completion", "sandbox_status": "ok", "verifier_explanation": ""}
    timeout = _bcb_sample(20, "```python\nwhile True: pass\n```", "while True: pass\n",
                          stderr="Verification timed out.")
    assert isv.bcb_row_facts(timeout, timeout.scores["verify"])["sandbox_status"] == "verify_timeout"
    # unfenced Python is valid code, not an extraction failure
    unfenced = _bcb_sample(40, "import os\ndef f():\n    return 1\n", "import os\ndef f():\n    return 1\n")
    assert isv.bcb_row_facts(unfenced, unfenced.scores["verify"])["code_source"] == "whole_completion"
    # an ADVERSARIAL answer that embeds the verdict sentence inside its own code: the
    # preamble reconstruction pins the boundary, so it still classifies by the tail
    sneaky_code = "x = 'The submission was correct.'\n"
    sneaky = _bcb_sample(41, f"```python\n{sneaky_code}```", sneaky_code, stderr="NameError: f\n")
    assert isv.bcb_row_facts(sneaky, sneaky.scores["verify"])["sandbox_status"] == "ok"
    # a preamble mismatch (the scorer's fence moved) refuses, never guesses
    moved = _bcb_sample(42, "```python\ny=1\n```", "y=1\n")
    moved.scores["verify"].explanation = moved.scores["verify"].explanation.replace("```python\n\n", "```py\n\n")
    with pytest.raises(isv.InspectServedError, match="does not begin with the verification program"):
        isv.bcb_row_facts(moved, moved.scores["verify"])
    # a verdict tail that is neither literal refuses
    odd = _bcb_sample(43, "```python\nz=1\n```", "z=1\n")
    odd.scores["verify"].explanation = isv.bcb_verify_preamble("z=1\n", TEST_SRC) + "The submission was fine."
    with pytest.raises(isv.InspectServedError, match="neither the correct suffix"):
        isv.bcb_row_facts(odd, odd.scores["verify"])
    # a score without an answer / a sample without the test source cannot be classified
    bare = _bcb_sample(44, "```python\nq=1\n```", "q=1\n")
    bare.metadata = {}
    with pytest.raises(isv.InspectServedError, match="cannot reconstruct"):
        isv.bcb_row_facts(bare, bare.scores["verify"])
    # code_source: an answer from neither fence nor the whole completion means find_code moved
    with pytest.raises(isv.InspectServedError, match="find_code grammar moved"):
        isv.bcb_code_source("```python\na=1\n```", "b=2\n")
    # ifeval-canonical-battery (P5): the IFEval task files its five values + ids the same way; nothing else joins
    # capeval-full-tasks: the full sibling reads the same three facts through the same reader
    assert isv.ROW_FACTS_OF_TASK == {"bigcodebench_200": isv.bcb_row_facts, "bigcodebench_full": isv.bcb_row_facts,
                                     C.IFEVAL_TASK_NAME: isv.ifeval_row_facts}


def test_results_and_summary_carry_the_facts_counts_and_max_sandboxes(tmp_path, monkeypatch):
    log = _Log([_bcb_sample(4, "```python\ndef f():\n    return 1\n```", "def f():\n    return 1\n"),
                _bcb_sample(8, "def f():\n    return 2", "def f():\n    return 2", stderr="Verification timed out.")])
    res = isv.results_from_log(log, primary_scorer="verify", row_facts=isv.bcb_row_facts)
    assert [r.row_facts["sandbox_status"] for r in res] == ["ok", "verify_timeout"]
    extra = res[0].extra(thinking="off", max_tokens=4096)
    assert extra["code_source"] == "python_fence" and extra["sandbox_status"] == "ok" and extra["verifier_explanation"] is None
    assert set(C.INSPECT_EXTRA_FIELDS) <= set(extra)
    counts = isv.bcb_summary_counts(res)
    assert counts == {"n_sandbox_ok": 1, "n_verify_timeout": 1, "n_code_source_python_fence": 1,
                      "n_code_source_bare_fence": 0, "n_code_source_whole_completion": 1}
    # through run_one_task on a SANDBOXED plan: max_sandboxes = the served S, sandbox_cleanup on
    seen: List[Any] = []

    async def _run(task, **kwargs):
        seen.append(kwargs)
        return [log]
    monkeypatch.setattr("inspect_ai.eval_async", _run)
    plan = _plan(row_task="inspect/bigcodebench_200", name="bigcodebench_200", max_tokens=4096,
                 primary_scorer="verify", sample_ids=["4", "8"], expected_n=200, n_run=2,
                 sandbox={"provider": "modal_owned"}, limit=None)
    summary = asyncio.run(isv.run_one_task(plan, model=None, store=_Store(tmp_path / "a"), template={},
                                           thinking="off", concurrency=18, emitter=_Emitter(),
                                           limits={"inspect_limit": 2}))
    assert seen[0]["max_sandboxes"] == 18 and seen[0]["sandbox_cleanup"] is True
    assert summary["n_sandbox_ok"] == 1 and summary["n_verify_timeout"] == 1
    assert summary["n_code_source_whole_completion"] == 1 and set(summary) <= set(C.SUMMARY_PAYLOAD_FIELDS) | {
        "logs_reused", "n_paired", "note", "render", "reasoning_parser", "n_content_empty_with_reasoning",
        "n_retried", "retry_max_tokens", "unparseable_rate", "at_cap_rate", "n_scored", "n_unscored",
        "n_scored_samples", "n_unscored_samples", "mean_completion_tokens", "p99_completion_tokens",
        "render_regime", "salt_regime", "resumed_policy", "n_render_regimes", "n_salt_regimes", "n_resumed_policies"}
    # a NON-sandboxed plan's eval call is byte-identical to before: no sandbox kwargs
    asyncio.run(isv.run_one_task(_plan(max_tokens=512), model=None, store=_Store(tmp_path / "b"), template={},
                                 thinking="off", concurrency=18, emitter=_Emitter()))
    assert "max_sandboxes" not in seen[-1] and "sandbox_cleanup" not in seen[-1]
    assert "n_sandbox_ok" not in completion.read_task_summary(tmp_path / "b", "inspect/math")


def test_a_sandboxed_tasks_errored_log_is_refused_and_its_diagnostics_written(tmp_path, monkeypatch):
    """Strict: an errored sample refuses the task (no row, no
    marker) — and, on a sandboxed task, the errored items land in
    diagnostics/sandbox_failures.json first, appended per attempt."""
    from test_inspect_served import _errored_sample

    bad = _Log([_bcb_sample(4, "```python\nx=1\n```", "x=1\n"), _errored_sample("8")], status="error")
    monkeypatch.setattr("inspect_ai.eval_async", _fake_eval({0: bad}))
    store = _Store(tmp_path / "c")
    plan = _plan(row_task="inspect/bigcodebench_200", name="bigcodebench_200", max_tokens=4096,
                 primary_scorer="verify", sample_ids=["4", "8"], expected_n=200, n_run=2,
                 sandbox={"provider": "modal_owned"}, limit=None)
    with pytest.raises(isv.InspectServedError, match="produced NO generation"):
        asyncio.run(isv.run_one_task(plan, model=None, store=store, template={}, thinking="off",
                                     concurrency=4, emitter=_Emitter(), limits={"inspect_limit": 2}))
    diag = json.loads((tmp_path / "c" / "diagnostics" / "sandbox_failures.json").read_text())
    assert len(diag["attempts"]) == 1 and diag["attempts"][0]["errors"][0]["item_id"] == "8"
    assert diag["attempts"][0]["task"] == "inspect/bigcodebench_200" and diag["attempts"][0]["log_status"] == "error"
    assert not (tmp_path / "c" / C.PARTIAL_DIR).exists() and not (tmp_path / "c" / C.DONE_DIR).exists()
    # a second refused attempt appends
    with pytest.raises(isv.InspectServedError):
        asyncio.run(isv.run_one_task(plan, model=None, store=store, template={}, thinking="off",
                                     concurrency=4, emitter=_Emitter(), limits={"inspect_limit": 2}))
    assert len(json.loads((tmp_path / "c" / "diagnostics" / "sandbox_failures.json").read_text())["attempts"]) == 2
    # a non-sandboxed task's refusal writes no diagnostics file
    monkeypatch.setattr("inspect_ai.eval_async", _fake_eval({0: _Log([_errored_sample("a")], status="error")}))
    with pytest.raises(isv.InspectServedError):
        asyncio.run(isv.run_one_task(_plan(expected_n=1), model=None, store=_Store(tmp_path / "d"), template={},
                                     thinking="off", concurrency=4, emitter=_Emitter()))
    assert not (tmp_path / "d" / "diagnostics").exists()


def test_the_pre_standup_selection_probe_derives_both_pins(tmp_path):
    ids = isv.bcb200_selection_ids(REPO)
    assert ids == LOG_IDS_B and all(i.isdigit() for i in ids)
    assert isv.bfcl200_selection_ids(REPO) == PIN_F["ids"]
    assert isv._SELECTION_IDS_OF_TASK["inspect/bigcodebench_200"] is isv.bcb200_selection_ids
    assert isv._SELECTION_IDS_OF_TASK["inspect/bfcl_200"] is isv.bfcl200_selection_ids
    # the probe refuses a foreign-id retained log for a pinned leg (fail closed), by name
    d = tmp_path / "logs"
    d.mkdir()
    spec = {"repo_root": str(REPO), "checks": [{"log_dir": str(d), "row_task": "inspect/bigcodebench_200",
                                                  "expected_n": 200}]}
    why = isv.retained_logs_report(spec)[str(d)]
    assert why is not None and "no retained .eval log" in why


# --------------------------------------------------------------------------- #
# bcb-bfcl-200: a ladder rung through run_one_task                             #
# --------------------------------------------------------------------------- #


def test_a_ladder_rung_evaluates_at_the_override_cap_and_ends_partial_by_the_selection(tmp_path, monkeypatch):
    """`_fake_eval` sees `max_tokens=16384` and exactly the selection's ids; the
    task ends with a PARTIAL marker whose limits are {"item_selection": 1},
    `n_items` = the selection's size, `n_expected_full` = 200, the summary and
    the outcome record at the override cap — what `ladder_partial_reason`
    (launcher) and `check_rung2_store` (slice C) read."""
    import inspect as _inspect
    ids = PIN_F["ids"][:2]
    seen: List[Any] = []
    log = _Log([_Sample(ids[0], {"s": _Score("C")}), _Sample(ids[1], {"s": _Score("I")})])
    monkeypatch.setattr("inspect_ai.eval_async", _fake_eval({0: log}, seen))
    plan = _plan(row_task="inspect/bfcl_200", name="bfcl_200", max_tokens=16384, cap_source=isv.CAP_SOURCE_OVERRIDE,
                 sample_ids=list(ids), selection=list(ids), expected_n=200, n_run=2, limit=None, item_set="pinned200")
    store = _Store(tmp_path / "rung2")
    limits = completion.active_limits(item_selection=1)
    summary = asyncio.run(isv.run_one_task(plan, model=None, store=store, template={}, thinking="off",
                                           concurrency=16, emitter=_Emitter(), limits=limits))
    assert seen and seen[0][2] == 16384 and list(seen[0][3]) == list(ids) and seen[0][4] is None
    assert summary["max_tokens"] == 16384 and summary["n"] == 2
    marker = json.loads(completion.partial_marker_path(store.cell_dir, "inspect/bfcl_200").read_text())
    assert set(marker) == set(C.PARTIAL_MARKER_FIELDS)
    assert marker["limits"] == {"item_selection": 1} and marker["n_items"] == 2 and marker["n_expected_full"] == 200
    assert not completion.done_marker_path(store.cell_dir, "inspect/bfcl_200").is_file()
    outcomes = completion.read_task_outcomes(store.cell_dir, "inspect/bfcl_200")
    assert outcomes["max_tokens"] == 16384 and {it["item_id"] for it in outcomes["items"]} == set(ids)
    # the dispatch entry point threads the two maps to plans_for_atoms, as given
    params = _inspect.signature(isv.run_inspect_tasks).parameters
    assert "cap_overrides_by_task" in params and "item_selection_by_task" in params
    src = Path(isv.__file__).read_text()
    assert "cap_overrides_by_task=cap_overrides_by_task,\n                            item_selection_by_task=item_selection_by_task," in src


def test_a_ladder_task_already_validly_filed_is_skipped_at_dispatch_without_a_request(tmp_path, monkeypatch):
    """A resumed two-leg rung hands the driver the FULL plan; the leg
    already filed as a valid PARTIAL over exactly the selection is
    skipped by the ONE strict validator (no eval call, the prior summary
    returned); a PARTIAL over ANOTHER selection is not a skip."""
    from sisteer.capeval.tinycap import rung_probe as RP
    ids = list(PIN_F["ids"][:2])
    seen: List[Any] = []
    log = _Log([_Sample(ids[0], {"s": _Score("C")}), _Sample(ids[1], {"s": _Score("I")})])
    monkeypatch.setattr("inspect_ai.eval_async", _fake_eval({0: log}, seen))
    plan = _plan(row_task="inspect/bfcl_200", name="bfcl_200", max_tokens=16384, cap_source=isv.CAP_SOURCE_OVERRIDE,
                 sample_ids=list(ids), selection=list(ids), expected_n=200, n_run=2, limit=None, item_set="pinned200")
    store = _Store(tmp_path / "rung2")
    limits = completion.active_limits(item_selection=1)
    first = asyncio.run(isv.run_one_task(plan, model=None, store=store, template={}, thinking="off",
                                         concurrency=16, emitter=_Emitter(), limits=limits))
    assert len(seen) == 1 and RP.ladder_task_partial_reason(store.cell_dir, "inspect/bfcl_200", log_ids=ids,
                                                             cap=16384, n_expected_full=200) is None
    # the relaunch: the same plan over the same store -> skipped, nothing requested
    again = asyncio.run(isv.run_one_task(plan, model=None, store=store, template={}, thinking="off",
                                         concurrency=16, emitter=_Emitter(), limits=limits))
    assert len(seen) == 1 and again["skipped"].startswith("ladder rung complete") and again["n"] == first["n"] == 2
    # another selection over the same store is NOT a skip (the validator names the mismatch) — it runs
    other = list(PIN_F["ids"][1:3])
    log2 = _Log([_Sample(other[0], {"s": _Score("C")}), _Sample(other[1], {"s": _Score("I")})])
    monkeypatch.setattr("inspect_ai.eval_async", _fake_eval({0: log2}, seen))
    plan2 = _plan(row_task="inspect/bfcl_200", name="bfcl_200", max_tokens=16384, cap_source=isv.CAP_SOURCE_OVERRIDE,
                  sample_ids=list(other), selection=list(other), expected_n=200, n_run=2, limit=None, item_set="pinned200")
    why = RP.ladder_task_partial_reason(store.cell_dir, "inspect/bfcl_200", log_ids=other, cap=16384, n_expected_full=200)
    assert why and "outcomes items != the selection" in why
    asyncio.run(isv.run_one_task(plan2, model=None, store=_Store(store.cell_dir), template={}, thinking="off",
                                 concurrency=16, emitter=_Emitter(), limits=limits))
    assert len(seen) == 2                                              # it DID request
    assert RP.ladder_task_partial_reason(store.cell_dir, "inspect/bfcl_200", log_ids=other, cap=16384,
                                         n_expected_full=200) is None  # ... and the store now files THAT selection
    # a non-ladder plan never consults the validator
    asyncio.run(isv.run_one_task(_plan(max_tokens=512), model=None, store=_Store(tmp_path / "b"), template={},
                                 thinking="off", concurrency=16, emitter=_Emitter()))
    assert len(seen) == 3
    # a summary edited out of agreement with its outcomes is NOT skipped at dispatch —
    # the validator's canonical verifier refuses, and the task runs again
    sp = completion.task_summary_path(store.cell_dir, "inspect/bfcl_200")
    doc = json.loads(sp.read_text())
    doc["n_scored_samples"] = doc["n_scored_samples"] + 1
    sp.write_text(json.dumps(doc))
    why = RP.ladder_task_partial_reason(store.cell_dir, "inspect/bfcl_200", log_ids=other, cap=16384, n_expected_full=200)
    assert why and "outcomes disagree with the summary" in why
    asyncio.run(isv.run_one_task(plan2, model=None, store=_Store(store.cell_dir), template={}, thinking="off",
                                 concurrency=16, emitter=_Emitter(), limits=limits))
    assert len(seen) == 4


# --------------------------------------------------------------------------- #
# the two FULL siblings' plans                                                  #
# --------------------------------------------------------------------------- #

PIN_B_FULL = C.load_bcb_full_pin(REPO)
PIN_F_FULL = C.load_bfcl_full_pin(REPO)
LOG_IDS_B_FULL = [C.bcb_log_id(i) for i in PIN_B_FULL["ids"]]


def test_bigcodebench_full_plans_the_whole_split_in_the_same_sandbox_and_refuses_the_ladder(monkeypatch):
    seen: List[Dict[str, Any]] = []

    def _build(ids=None, sandbox=C.BCB_SANDBOX_PROVIDER, repo_root=None, pin=None):
        seen.append({"ids": list(ids or []), "sandbox": sandbox, "pin_n": (pin or {}).get("n"), "pin_task": (pin or {}).get("task")})
        return _fake_task([int(C.bcb_log_id(i)) for i in ids])
    monkeypatch.setattr(isv, "load_repo_task",
                        lambda name: _build if name == "bigcodebench_full" else pytest.fail(name))
    kw = dict(grader_model=None, thinking="off", repo_root=REPO)
    plan = isv.plan_task("bigcodebench_full", **kw)
    assert plan.row_task == "inspect/bigcodebench_full" and plan.expected_n == 1140 == plan.n_run
    assert plan.sample_ids == LOG_IDS_B_FULL and plan.limit is None and plan.selection is None
    assert plan.max_tokens == 4096 and plan.cap_source == "off_table" and plan.primary_scorer == "verify"
    assert plan.item_set == "full" and plan.samples_per_item == 1 and plan.thinking == "off"
    # the SAME sandbox facts as the 200 leg (one verifier)
    assert plan.sandbox == C.bcb_sandbox_facts()
    assert seen[-1]["ids"] == PIN_B_FULL["ids"] and seen[-1]["sandbox"] == "modal_owned"
    assert seen[-1]["pin_n"] == 1140 and seen[-1]["pin_task"] == "bigcodebench_full"
    assert set(LOG_IDS_B) < set(LOG_IDS_B_FULL) and "Hard included: 148" in plan.note
    assert "FULL_TASK_LIMITS" in plan.note
    # a smoke cap heads the FULL pin before building
    smoke = isv.plan_task("bigcodebench_full", inspect_limit=7, **kw)
    assert smoke.sample_ids == LOG_IDS_B_FULL[:7] and smoke.n_run == 7 and smoke.expected_n == 1140
    assert seen[-1]["ids"] == PIN_B_FULL["ids"][:7]
    # the truncation ladder's knobs are the 200 legs' — refused on the full sibling by name
    with pytest.raises(isv.InspectServedError, match="truncation ladder's knobs"):
        isv.plan_task("bigcodebench_full", cap_override=16384, selection=LOG_IDS_B_FULL[:2], **kw)
    with pytest.raises(isv.InspectServedError, match="truncation ladder's knobs"):
        isv.plan_task("bfcl_full", cap_override=16384, selection=PIN_F_FULL["ids"][:2], **kw)
    assert "inspect/bigcodebench_full" not in C.LADDER_ROW_TASKS and "inspect/bfcl_full" not in C.LADDER_ROW_TASKS


def test_bfcl_full_plans_every_id_by_sample_id_and_refuses_a_drifted_population(monkeypatch):
    population = [i for ids in PIN_F_FULL["population"].values() for i in ids]
    built: List[List[str]] = []

    def _bfcl_task(categories):
        built.append(list(categories))
        return _fake_task(population)
    monkeypatch.setattr(_bfcl_module(), "bfcl", _bfcl_task)
    kw = dict(grader_model=None, thinking="off", repo_root=REPO)
    plan = isv.plan_task("bfcl_full", **kw)
    assert plan.row_task == "inspect/bfcl_full" and plan.sample_ids == PIN_F_FULL["ids"]
    assert plan.expected_n == 3981 and plan.n_run == 3981 and plan.limit is None
    assert plan.max_tokens == 1024 and plan.sandbox is None and plan.item_set == "full"
    assert built[-1] == isv.bfcl_categories() and len(built[-1]) == 18
    assert set(PIN_F["ids"]) < set(plan.sample_ids) and "needs a --tool-calls standup" in plan.note
    # the full sibling is a tool-calling task: the route guards cover it
    assert isv._bfcl_requested(["bfcl_full"])
    with pytest.raises(isv.InspectServedError, match="composed no --tool-calls"):
        isv.assert_route_can_serve(["bfcl_full"], FIXTURE_ROUTES[1], agent_loop_entry=True, tool_calls=False)
    # a dataset drift (one id short) refuses by name
    monkeypatch.setattr(_bfcl_module(), "bfcl", lambda categories: _fake_task(population[:-1]))
    with pytest.raises(isv.InspectServedError, match="bfcl_full: the task built over 18 categories exposes 3980"):
        isv.plan_task("bfcl_full", **kw)
    # a population of the right size missing a pinned id refuses too
    swapped = [*population[:-1], "not_a_pinned_id"]
    monkeypatch.setattr(_bfcl_module(), "bfcl", lambda categories: _fake_task(swapped))
    with pytest.raises(isv.InspectServedError, match="only 3980/3981 pinned ids"):
        isv.plan_task("bfcl_full", **kw)
