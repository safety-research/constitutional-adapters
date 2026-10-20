"""BigCodeBench verification as a CPU phase — the DRIVER half, on the PINNED inspect-ai
client (0.3.260, the ``inspect260`` mark; the module importorskips ``inspect_ai``, so
the repo ``.venv`` skips it).

* the generation-only task construction: the coupled build minus its sandbox — same ids,
  metadata, version, a scorer named ``verify``, equal compose facts; the fake ``eval_async``
  receives ``score=False`` and NO sandbox kwarg; a non-sandboxed task refuses the knob;
* ``generate`` writes the UNSCORED log under the store, the generation sidecar (exactly
  ``cpu_verify.GENERATION_SIDECAR_FIELDS``) and the pending marker (exactly
  ``contracts.PENDING_MARKER_FIELDS``) into the row dir, files NOTHING; a re-run reuses the
  log; a coupled run over the pending task refuses by name;
* the BINDING: a vehicle produced SHUFFLED at ``max_connections=4`` through the ``replay``
  provider binds by ``(id, epoch)``; every original ``model_usage`` key, the ``output`` bytes,
  ``messages``, ``metadata`` and ``events`` are kept; ``sample_usage`` reads the original
  target key off the merged log; a differing completion, a missing id, a duplicate id, a
  foreign id, a missing score each REFUSE by name;
* the two-phase filing (generate -> verify) produces outcome items
  (``TASK_OUTCOME_ITEM_FIELDS``), a summary and rows EQUAL to the coupled path's on the same
  fixture; the coupled path through the lifted ``_file_task_results`` is byte-identical to a
  golden captured from the pre-lift tail;
* the three mechanisms' mechanics: M1 raises (no sandbox at re-score time — the §0
  prediction), M1b scores through ``sandboxenv_context`` (a LOCAL sandbox here), M2 through
  the replay vehicle; the verifier pointer's fields are the harness's ``POINTER_FIELDS`` +
  ``phase`` (pinned by reading the harness source, no Modal import).

The fixture ``tests/fixtures/tinycap_verification/unscored_12.eval`` is SYNTHETIC (built by
``build_fixture.py`` beside it: twelve invented tasks and completions served by a fixture
provider registered under the production name, so the log's shape is production's). The
``verify`` scorer here is the stock scorer's logic verbatim (``find_code``, ``sandbox().exec``
of answer + test, the explanation grammar) over a ``local`` sandbox, so ``bcb_row_facts``
reads real verdicts.
"""

from __future__ import annotations

import ast
import functools
import asyncio
import json
import os
import re
import shutil
import sys
import types
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("inspect_ai", reason="tinycap client venv (hashed lock; the inspect260 mark)")
# the stock scorer's code-block extraction is imported at module level below: a venv with inspect_ai but
# without inspect_evals (the repo .venv) must SKIP this module, not fail its collection
pytest.importorskip("inspect_evals", reason="tinycap client venv (inspect_evals rides the hashed lock; the inspect260 mark)")

import inspect_ai  # noqa: E402

pytestmark = [
    pytest.mark.inspect260,
    pytest.mark.skipif(inspect_ai.__version__ != "0.3.260",
                       reason=f"the pinned client is inspect-ai 0.3.260; this venv has {inspect_ai.__version__}"),
]

os.environ.setdefault("INSPECT_DISPLAY", "none")

from inspect_ai import Task  # noqa: E402
from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from inspect_ai.scorer import Score, mean, scorer, std  # noqa: E402
from inspect_ai.solver import generate  # noqa: E402
from inspect_ai.util import ExecResult, sandbox  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tinycap_golden as golden  # noqa: E402
from test_inspect_served import _Emitter, _Store  # noqa: E402

from sisteer.capeval.tiers import inspect_served as isv  # noqa: E402
from sisteer.capeval.tinycap import completion  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer.capeval.tinycap import cpu_verify as CV  # noqa: E402
from sisteer.capeval.tinycap import sandbox_scope as scope  # noqa: E402

REPO = isv.REPO_ROOT
FIXTURE_DIR = REPO / "tests" / "fixtures" / "tinycap_verification"
FIXTURE = FIXTURE_DIR / "unscored_12.eval"
GOLDEN = FIXTURE_DIR / "coupled_filing_golden.json"
NAME = C.CONTROL_BIGCODEBENCH_FULL
ROW_TASK = C.inspect_task(NAME)
CELL = "none"
VMODEL = "petri__base"
TARGET_KEY = C.inspect_model_name(VMODEL)
GEN_FCID = "fc-01TESTGENERATE000000000000"
VERIFY_FCID = "fc-01TESTVERIFY0000000000000"
N = 12
IDS = [str(i) for i in range(N)]
#: the store root's basename — the row dir's generation files are keyed by it (the in-run repeat of the same
#: --cell runs into another store); every store in this module lives under <tmp>/<STORE>/cells/k
STORE = "bcbw-synthetic-shared"
#: the fixture's verdicts: ids 1, 4, 7, 10 return 3 * x (wrong); everything else 2 * x
WRONG = {"1", "4", "7", "10"}
M1, M1B, M2 = C.CPU_VERIFY_MECHANISMS


# --------------------------------------------------------------------------- #
# the stock scorer's logic, over a local sandbox                                #
# --------------------------------------------------------------------------- #


from inspect_evals.bigcodebench.bigcodebench import find_code  # noqa: E402  (the stock code-block extraction the scorer uses)


@scorer(metrics=[mean(), std()])
def verify():
    """The stock ``verify`` scorer's body (the sandbox exec of answer + test, the exact explanation
    grammar ``inspect_served.bcb_verify_preamble`` reconstructs) — registered under the stock NAME."""

    async def score(state, target):
        answer = find_code(state.output.completion)
        code = "".join([answer, "\n", state.metadata["test"], "\n", isv.BCB_EXPLANATION_MAIN])
        try:
            result = await sandbox().exec(cmd=[sys.executable, "-c", code], timeout=30)
        except TimeoutError:
            result = ExecResult(False, 1, "", "Verification timed out.")
        explanation = isv.bcb_verify_preamble(answer, state.metadata["test"])
        explanation += (isv.BCB_CORRECT_SUFFIX if result.success else isv.BCB_INCORRECT_PREFIX + result.stderr)
        return Score(value="C" if result.success else "I", answer=answer, explanation=explanation)

    return score


def fixture_log():
    return read_eval_log(str(FIXTURE))


def synthetic_task(original, *, sandbox_spec="local", ids=None) -> Task:
    """The fixture's twelve samples (or the ``ids`` named, in log vocabulary) as a Task in the thin
    task's shape: the stock ``generate()`` solver, the ``verify`` scorer, a LOCAL sandbox (the
    vehicle's / M1b's ``python -c`` runs here)."""
    keep = None if ids is None else {str(i) for i in ids}
    samples = [Sample(id=s.id, input=s.input, target=s.target, metadata=dict(s.metadata or {}))
               for s in original.samples if keep is None or str(s.id) in keep]
    return Task(dataset=MemoryDataset(samples), solver=[generate()], scorer=verify(), sandbox=sandbox_spec,
                version=C.BCB_SPLIT, name="bigcodebench")


def _plan(original, *, generation_only: bool, task=None, expected_n: int = N, limits_n: int = N) -> isv.TaskPlan:
    """A BigCodeBench-full plan over the fixture's ids (the fields the planner would set on the
    12-item head of the full pin, with a 12-item "full" so the task ends DONE)."""
    return isv.TaskPlan(
        row_task=ROW_TASK, name=NAME, task=task if task is not None else synthetic_task(original),
        sample_ids=list(IDS), samples_per_item=1, limit=None, max_tokens=4096, primary_scorer="verify",
        expected_n=expected_n, n_run=limits_n, temperature=0.0, top_p=1.0, top_k=0, item_set="full",
        thinking=C.THINKING_OFF, cap_source="off_table", sandbox=C.bcb_sandbox_facts(),
        generation_only=generation_only, note="synthetic fixture (tests/fixtures/tinycap_verification)")


def _fake_eval_copying(src: Path, seen: List[Dict[str, Any]]):
    """A stand-in ``eval_async`` that copies ``src`` into the log dir and returns it read back —
    what a real eval leaves behind — recording the kwargs it was called with."""

    async def _run(task, **kwargs):
        seen.append(dict(kwargs))
        log_dir = Path(kwargs["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        dst = log_dir / src.name
        shutil.copyfile(src, dst)
        return [read_eval_log(str(dst))]

    return _run


def _gen_phase(row_dir: Path, *, fcid=GEN_FCID, concurrency=32) -> CV.GenerationPhase:
    return CV.GenerationPhase(row_dir=row_dir, cell=CELL, generation_fcid=fcid, concurrency=concurrency, store=STORE)


def _run_generate(tmp_path, monkeypatch, *, seen=None, row_dir=None, store_dir=None, limits=None, em=None,
                  fcid=GEN_FCID):
    """One generation-only ``run_one_task`` over the fixture. The fake ``eval_async`` lives on a LOCAL
    MonkeyPatch and is undone on return, so a real eval (the vehicle) can follow in the same test."""
    original = fixture_log()
    store = _Store(store_dir if store_dir is not None else tmp_path / STORE / "cells" / "k")
    row_dir = row_dir if row_dir is not None else tmp_path / "row"
    seen = seen if seen is not None else []
    em = em if em is not None else _Emitter()
    mp = pytest.MonkeyPatch()
    mp.setattr("inspect_ai.eval_async", _fake_eval_copying(FIXTURE, seen))
    try:
        out = asyncio.run(isv.run_one_task(
            _plan(original, generation_only=True), model=None, store=store, template={}, thinking=C.THINKING_OFF,
            concurrency=32, emitter=em, route_virtual_model=VMODEL, limits=limits or {},
            generation=_gen_phase(row_dir, fcid=fcid)))
    finally:
        mp.undo()
    return out, store, row_dir, seen, em


def _vehicle(original, task, tmp_path, *, shuffle=3, max_connections=4, mechanism=M2, max_sandboxes=4):
    return asyncio.run(CV.run_mechanism(mechanism, original=original, coupled_task=task, max_sandboxes=max_sandboxes,
                                        log_dir=tmp_path / f"vehicle_{mechanism}", max_connections=max_connections,
                                        sample_shuffle=shuffle))


# --------------------------------------------------------------------------- #
# §2.2.1 the generation-only task construction                                 #
# --------------------------------------------------------------------------- #


COMPOSE = """services:
  default:
    image: {image}
    init: true
    command: tail -f /dev/null
    cpus: 1.0
    mem_limit: 0.5gb
    network_mode: none
"""


def _stub_sandbox_owned(monkeypatch) -> types.ModuleType:
    """The owned sandbox provider module, stubbed: it needs Modal (absent from the client venv), and the
    tests here exercise the planning, the filing and the binding, never the sandbox exec."""
    stub = types.ModuleType("sisteer.capeval.tiers.sandbox_owned")
    stub.OwnedModalSandboxEnvironment = type("OwnedModalSandboxEnvironment", (), {})
    stub.PROVIDER_NAME = C.BCB_SANDBOX_PROVIDER
    monkeypatch.setitem(sys.modules, "sisteer.capeval.tiers.sandbox_owned", stub)
    return stub


def _stock_builder_faked(monkeypatch, tmp_path):
    """The stock ``inspect_evals`` constructor faked (position-indexed ids, the pinned compose) and
    the owned provider module stubbed (its import needs Modal, which the client venv lacks) — every
    check in ``narrowed_bigcodebench`` still runs against the faked build."""
    import importlib

    compose = tmp_path / "compose.yaml"
    compose.write_text(COMPOSE.format(image=C.BCB_SANDBOX_IMAGE))
    stock = importlib.import_module("inspect_evals.bigcodebench.bigcodebench")

    def fake_bigcodebench(*, subset, version, sandbox):
        samples = [Sample(id=p, input=f"prompt {p}", target=f"test {p}",
                          metadata={"test": f"test {p}", "libs": "[]", "task_id_metadata": f"BigCodeBench/{p}"})
                   for p in subset]
        return Task(dataset=MemoryDataset(samples), solver=[generate()], scorer=verify(),
                    sandbox=(sandbox, str(compose)), version=version, name="bigcodebench")

    monkeypatch.setattr(stock, "bigcodebench", fake_bigcodebench)
    _stub_sandbox_owned(monkeypatch)
    return compose


def test_the_generation_only_task_is_the_coupled_task_minus_its_sandbox(tmp_path, monkeypatch):
    """With ``generation_only=True`` the thin task is built EXACTLY as the coupled one
    (every assertion and the compose check run), then the sandbox is dropped and ``(task, facts)``
    returned — same ids, same metadata, same version, the scorer named ``verify`` still on it, the
    facts equal to ``compose_facts(coupled)``; ``compose_facts`` on the sandbox-less task refuses."""
    _stock_builder_faked(monkeypatch, tmp_path)
    build = isv.load_repo_task(NAME)
    module = types.SimpleNamespace(**build.__globals__)          # the thin task's module namespace (loaded by path)
    pin = C.load_bcb_full_pin(REPO)
    ids = [C.bcb_pinned_id(i) for i in IDS]
    coupled = build(ids=ids, sandbox=C.BCB_SANDBOX_PROVIDER, repo_root=REPO, pin=pin)
    built = build(ids=ids, sandbox=C.BCB_SANDBOX_PROVIDER, repo_root=REPO, pin=pin, generation_only=True)
    assert isinstance(built, tuple) and len(built) == 2
    gen, facts = built
    assert gen.sandbox is None
    assert coupled.sandbox.type == C.BCB_SANDBOX_PROVIDER and isinstance(coupled.sandbox.config, str)
    assert [s.id for s in gen.dataset] == [s.id for s in coupled.dataset] == list(range(N))
    assert [s.metadata for s in gen.dataset] == [s.metadata for s in coupled.dataset]
    assert gen.version == coupled.version == C.BCB_SPLIT
    assert isv._scorer_names(gen) == isv._scorer_names(coupled) == ["verify"]
    assert facts == module.compose_facts(coupled)
    assert facts["image"] == C.BCB_SANDBOX_IMAGE and facts["network_mode"] == "none"
    with pytest.raises(ValueError, match="generation-only build"):
        module.compose_facts(gen)
    # the default is byte-identical: a Task, not a tuple
    assert isinstance(coupled, Task)
    # the 200 leg's own entry point forwards the knob through the ONE builder too
    sibling = isv.load_repo_task("bigcodebench_200")
    pin200 = C.load_bcb_pin(REPO)
    g200, f200 = sibling(ids=list(pin200["ids"][:3]), sandbox=C.BCB_SANDBOX_PROVIDER, repo_root=REPO, pin=pin200,
                         generation_only=True)
    assert g200.sandbox is None and f200 == facts


def test_the_planner_builds_a_generation_only_plan_for_the_sandboxed_legs_alone(tmp_path, monkeypatch):
    """``plan_task(..., generation_only=True)``: the plan's task carries no sandbox while
    ``plan.sandbox`` STAYS the leg's facts (the filing tail reads them) and ``generation_only``
    is set; the coupled plan is unchanged; a non-sandboxed task refuses the knob before any build;
    ``plans_for_atoms`` marks exactly the named row tasks and refuses a stray name."""
    _stock_builder_faked(monkeypatch, tmp_path)
    kw = dict(grader_model=None, thinking=C.THINKING_OFF, inspect_limit=N, samples=1, repo_root=REPO,
              verify_math500=False)
    gen = isv.plan_task(NAME, generation_only=True, **kw)
    coupled = isv.plan_task(NAME, **kw)
    assert gen.generation_only is True and gen.task.sandbox is None
    assert gen.sandbox == C.bcb_sandbox_facts() == coupled.sandbox
    assert coupled.generation_only is False and coupled.task.sandbox is not None
    assert gen.sample_ids == coupled.sample_ids == IDS and gen.expected_n == coupled.expected_n == 1140
    assert gen.primary_scorer == coupled.primary_scorer == "verify"
    assert replace(gen, task=None, generation_only=False) == replace(coupled, task=None)
    with pytest.raises(isv.InspectServedError, match="generation_only is the sandboxed legs"):
        isv.plan_task("gsm8k_paired200", generation_only=True, **kw)
    plans = isv.plans_for_atoms([NAME], generation_only_tasks=[ROW_TASK], thinking=C.THINKING_OFF, grader_model=None,
                                inspect_limit=N, samples=1, repo_root=REPO, verify_math500=False)
    assert [p.generation_only for p in plans] == [True]
    plans = isv.plans_for_atoms([NAME], thinking=C.THINKING_OFF, grader_model=None, inspect_limit=N, samples=1,
                                repo_root=REPO, verify_math500=False)
    assert [p.generation_only for p in plans] == [False]
    with pytest.raises(isv.InspectServedError, match="does not plan"):
        isv.plans_for_atoms([NAME], generation_only_tasks=["inspect/bfcl_full"], thinking=C.THINKING_OFF,
                            grader_model=None, inspect_limit=N, samples=1, repo_root=REPO, verify_math500=False)
    # plan_facts round-trips the generation-only plan (the sidecar carries it; the verifier rebuilds it)
    facts = isv.plan_facts(gen)
    assert set(facts) == {f for f in gen.__dataclass_fields__} - set(isv.PLAN_FACTS_EXCLUDED)
    back = isv.plan_from_facts(json.loads(json.dumps(facts)), task=gen.task)
    assert back == gen
    with pytest.raises(isv.InspectServedError, match="missing \\['sample_ids'\\]"):
        isv.plan_from_facts({k: v for k, v in facts.items() if k != "sample_ids"}, task=gen.task)


# --------------------------------------------------------------------------- #
# §2.2.3 generate: the unscored log, the sidecar, the marker — and nothing filed #
# --------------------------------------------------------------------------- #


def test_generate_writes_the_unscored_log_the_sidecar_and_the_pending_marker_and_files_nothing(tmp_path, monkeypatch):
    out, store, row_dir, seen, em = _run_generate(tmp_path, monkeypatch)
    # the fake eval saw score=False and NO sandbox kwarg
    assert len(seen) == 1
    assert seen[0]["score"] is False
    assert "max_sandboxes" not in seen[0] and "sandbox_cleanup" not in seen[0]
    assert seen[0]["sample_id"] == IDS and seen[0]["max_tokens"] == 4096
    # the return value is a GENERATED block, never an OK status
    assert out["status"] == C.REC_STATUS_GENERATED and out["status"] not in C.REC_OK_STATUSES
    assert out["n_items"] == N and out["logs_reused"] == 0 and out["verification_phase"] == C.VERIFICATION_PHASE_GENERATE
    # the unscored log under the store's task log dir
    log_dir = C.inspect_log_dir(store.cell_dir, ROW_TASK, 0)
    logs = sorted(log_dir.glob("*.eval"))
    assert len(logs) == 1 and Path(out["log"]) == logs[0].resolve()
    assert all(not s.scores for s in read_eval_log(str(logs[0])).samples)
    # the pending marker: EXACTLY the contract's fields
    marker = CV.read_pending_marker(row_dir, CELL, ROW_TASK, STORE)
    assert tuple(sorted(marker)) == tuple(sorted(C.PENDING_MARKER_FIELDS))
    assert marker["task"] == ROW_TASK and marker["cell"] == CELL and marker["n_items"] == N
    assert marker["n_expected_full"] == N and marker["generation_fcid"] == GEN_FCID
    assert Path(marker["log"]) == logs[0].resolve()
    assert Path(out["pending_marker"]) == CV.pending_marker_path(row_dir, CELL, ROW_TASK, STORE)
    # the sidecar: EXACTLY cpu_verify's fields, the generation-side facts
    sidecar = CV.read_generation_sidecar(row_dir, CELL, ROW_TASK, STORE)
    assert tuple(sorted(sidecar)) == tuple(sorted(CV.GENERATION_SIDECAR_FIELDS))
    assert sidecar["target_model"] == TARGET_KEY and sidecar["route_virtual_model"] == VMODEL
    assert sidecar["sample_ids"] == IDS and sidecar["concurrency"] == 32 and sidecar["logs_reused"] == 0
    # the generated log's bytes, recorded — the verifier holds the store's log (or the archive) to them
    assert sidecar["log_sha256"] == CV.sha256_file(logs[0]) == CV.sha256_file(FIXTURE)
    # the generation side's request accounting, carried into the filing (the coupled tail's formula: the pass's
    # items + the truncation retry's requests — 0 under the k = 1, no-retry contract, carried as a fact)
    assert sidecar["n_retried_requests"] == 0 and sidecar["n_requests"] == N == out["n_requests"]
    assert sidecar["plan"]["generation_only"] is True and sidecar["plan"]["max_tokens"] == 4096
    assert sidecar["regimes"] == isv.regime_summary_fields(None, expected=None, task=ROW_TASK, out_of_support=False,
                                                           prior=None)
    assert sidecar["client"] == isv.client_record(None) and sidecar["n_http_rejected_retried"] is None
    assert sidecar["inspect_ai_version"] == inspect_ai.__version__
    for k in C.PENDING_MARKER_FIELDS:
        assert sidecar[k] == marker[k], k
    # NOTHING filed: no rows, no summary, no outcomes, no marker in the store
    assert em.rows == []
    for sub in ("done", "summary", "outcomes", "partial"):
        assert not (store.cell_dir / sub).exists(), sub
    assert not completion.is_done(store.cell_dir, ROW_TASK)
    # the row dir holds exactly the two generation files
    assert sorted(p.name for p in (row_dir / CV.GENERATION_DIR).iterdir()) == [
        f"{CELL}__inspect__bigcodebench_full__{STORE}.json", f"pending_{CELL}__inspect__bigcodebench_full__{STORE}.json"]
    assert sidecar["store"] == STORE


def test_a_re_run_reuses_the_unscored_log_and_a_coupled_run_over_the_pending_task_refuses(tmp_path, monkeypatch):
    """The unscored log is this plan's evidence: a second `generate` REUSES it (no eval call) and
    rewrites the sidecar with ``logs_reused`` 1 — never a re-generation. A COUPLED plan over the
    same store + row dir refuses by name: the verifier owns the task. A generation-only plan
    handed no generation facts refuses before any eval."""
    out, store, row_dir, seen, _em = _run_generate(tmp_path, monkeypatch)
    first = CV.read_generation_sidecar(row_dir, CELL, ROW_TASK, STORE)
    marker_path = CV.pending_marker_path(row_dir, CELL, ROW_TASK, STORE)
    marker_before, stat_before = marker_path.read_bytes(), marker_path.stat()
    # the re-run comes from ANOTHER container (a new fcid): the door reuses the log and keeps the
    # STANDING marker — the generation happened once, by the first fcid, at the first time
    out2, _s, _r, seen2, _e = _run_generate(tmp_path, monkeypatch, seen=seen, row_dir=row_dir,
                                            store_dir=store.cell_dir, fcid="fc-01TESTGENERATE000000000002")
    assert len(seen2) == 1                                   # no second eval call
    assert out2["logs_reused"] == 1 and out2["log"] == out["log"]
    second = CV.read_generation_sidecar(row_dir, CELL, ROW_TASK, STORE)
    assert second["logs_reused"] == 1 and second["generated_calls"] == 0 and second["log"] == first["log"]
    assert second["regimes"] == first["regimes"]
    assert second["generated_at"] == first["generated_at"] and second["generation_fcid"] == first["generation_fcid"] == GEN_FCID
    assert second["log_sha256"] == first["log_sha256"]
    # no new marker: the file is byte for byte the first run's and was not rewritten
    assert marker_path.read_bytes() == marker_before
    assert (marker_path.stat().st_mtime_ns, marker_path.stat().st_ino) == (stat_before.st_mtime_ns, stat_before.st_ino)
    assert Path(out2["pending_marker"]) == marker_path
    original = fixture_log()
    gen = _gen_phase(row_dir)
    with pytest.raises(isv.InspectServedError, match="pending marker .* awaits its CPU verifier"):
        asyncio.run(isv.run_one_task(_plan(original, generation_only=False), model=None, store=store, template={},
                                     thinking=C.THINKING_OFF, concurrency=32, emitter=_Emitter(),
                                     route_virtual_model=VMODEL, limits={}, generation=gen))
    with pytest.raises(isv.InspectServedError, match="generation-only plan needs the row's generation facts"):
        asyncio.run(isv.run_one_task(_plan(original, generation_only=True), model=None, store=_Store(tmp_path / "x"),
                                     template={}, thinking=C.THINKING_OFF, concurrency=32, emitter=_Emitter(),
                                     route_virtual_model=VMODEL, limits={}))
    assert len(seen2) == 1
    # a plan the contract does not cover refuses too (k > 1)
    with pytest.raises(isv.InspectServedError, match="k = 1"):
        asyncio.run(isv.run_one_task(replace(_plan(original, generation_only=True), samples_per_item=2), model=None,
                                     store=_Store(tmp_path / "y"), template={}, thinking=C.THINKING_OFF,
                                     concurrency=32, emitter=_Emitter(), route_virtual_model=VMODEL, limits={},
                                     generation=gen))


def test_a_retained_unscored_log_without_its_marker_or_sidecar_is_torn_and_refuses_rather_than_re_marking(
        tmp_path, monkeypatch):
    """The door (the lost-verifier rule): a store the harness parks as torn — the unscored
    log retained but the pending marker (or the sidecar) gone — is never healed in place by a re-run of
    `generate`: refused by name, no eval call, no marker re-created. The remedy is a deliberate regeneration."""
    out, store, row_dir, seen, _em = _run_generate(tmp_path, monkeypatch)
    marker_path = CV.pending_marker_path(row_dir, CELL, ROW_TASK, STORE)
    sidecar_path = CV.generation_sidecar_path(row_dir, CELL, ROW_TASK, STORE)
    # (a) the marker gone, the sidecar standing
    marker_path.unlink()
    with pytest.raises(isv.InspectServedError, match="torn") as ei:
        _run_generate(tmp_path, monkeypatch, seen=seen, row_dir=row_dir, store_dir=store.cell_dir)
    assert "pending marker" in str(ei.value) and str(marker_path) in str(ei.value)
    assert len(seen) == 1 and not marker_path.exists()          # nothing generated, nothing re-marked
    # (b) the marker standing, the sidecar gone
    sidecar_bytes = sidecar_path.read_bytes()
    CV.write_generation_pending(row_dir, sidecar=json.loads(sidecar_bytes),
                                marker={k: v for k, v in json.loads(sidecar_bytes).items() if k in C.PENDING_MARKER_FIELDS})
    sidecar_path.unlink()
    with pytest.raises(isv.InspectServedError, match="torn"):
        _run_generate(tmp_path, monkeypatch, seen=seen, row_dir=row_dir, store_dir=store.cell_dir)
    assert len(seen) == 1 and not sidecar_path.exists()
    # (c) both standing but the sidecar naming another log's bytes: torn too
    sidecar_path.write_bytes(sidecar_bytes)
    doc = json.loads(sidecar_bytes)
    doc["log_sha256"] = "0" * 64
    sidecar_path.write_text(json.dumps(doc))
    with pytest.raises(isv.InspectServedError, match="torn"):
        _run_generate(tmp_path, monkeypatch, seen=seen, row_dir=row_dir, store_dir=store.cell_dir)
    assert len(seen) == 1
    # the log itself was never touched
    assert CV.sha256_file(Path(out["log"])) == CV.sha256_file(FIXTURE)


def test_generate_refuses_a_log_whose_model_key_is_not_the_targets(tmp_path, monkeypatch):
    """The sidecar's ``target_model`` is what the verifier's ``sample_usage`` reads — a log recorded
    under another provider key would mis-attribute every row's tokens; refused at generation end."""
    original = fixture_log()
    seen: List[Dict[str, Any]] = []
    monkeypatch.setattr("inspect_ai.eval_async", _fake_eval_copying(FIXTURE, seen))
    gen = _gen_phase(tmp_path / "row", fcid=None, concurrency=8)
    with pytest.raises(isv.InspectServedError, match="model key .* is not the target's"):
        asyncio.run(isv.run_one_task(_plan(original, generation_only=True), model=None,
                                     store=_Store(tmp_path / "s"), template={}, thinking=C.THINKING_OFF,
                                     concurrency=8, emitter=_Emitter(), route_virtual_model="petri__other",
                                     limits={}, generation=gen))
    assert not (tmp_path / "row").exists()


def test_run_inspect_tasks_gates_the_phase_before_any_model_is_built():
    """``verify`` is never a dispatch value; ``generate`` needs the generation facts."""
    with pytest.raises(isv.InspectServedError, match="is not a GPU-side phase"):
        asyncio.run(isv.run_inspect_tasks(atoms=[NAME], route=None, routes=[], system=None, store=None, template={},
                                          thinking=C.THINKING_OFF, base_url="http://127.0.0.1:1/v1",
                                          verification_phase=C.VERIFICATION_PHASE_VERIFY))
    with pytest.raises(isv.InspectServedError, match="needs the row's generation facts"):
        asyncio.run(isv.run_inspect_tasks(atoms=[NAME], route=None, routes=[], system=None, store=None, template={},
                                          thinking=C.THINKING_OFF, base_url="http://127.0.0.1:1/v1",
                                          verification_phase=C.VERIFICATION_PHASE_GENERATE))
    gen = _gen_phase(Path("/nowhere"), fcid=None, concurrency=8)
    with pytest.raises(isv.InspectServedError, match="plan no sandboxed task"):
        asyncio.run(isv.run_inspect_tasks(atoms=["gsm8k"], route=None, routes=[], system=None, store=None, template={},
                                          thinking=C.THINKING_OFF, base_url="http://127.0.0.1:1/v1",
                                          verification_phase=C.VERIFICATION_PHASE_GENERATE, generation=gen))


# --------------------------------------------------------------------------- #
# §2.2.3 the binding                                                             #
# --------------------------------------------------------------------------- #


def test_the_merge_binds_by_id_and_epoch_over_a_shuffled_concurrent_vehicle(tmp_path):
    original = fixture_log()
    task = synthetic_task(original)
    vehicle, vpath = _vehicle(original, task, tmp_path, shuffle=3, max_connections=4)
    assert vpath.is_file() and vehicle.status == "success"
    assert vehicle.eval.model == f"{CV.REPLAY_PROVIDER}/{VMODEL}"
    # the vehicle RAN shuffled and 4-wide (the log records both; its samples are stored by id) — the
    # binding must not depend on the order the vehicle scored them in
    assert sorted(s.id for s in vehicle.samples) == list(range(N))
    assert vehicle.eval.config.sample_shuffle == 3 and vehicle.plan.config.max_connections == 4
    merged = CV.bind_scores(original, vehicle, primary_scorer="verify")
    assert [s.id for s in merged.samples] == list(range(N))          # the original's order
    for o, m in zip(original.samples, merged.samples):
        assert CV.sample_key(o) == CV.sample_key(m)
        # every original model_usage key and value, the output bytes, messages, metadata, events: untouched
        assert set(m.model_usage) == set(o.model_usage) == {TARGET_KEY}
        assert m.model_usage[TARGET_KEY] == o.model_usage[TARGET_KEY]
        assert m.output.model_dump_json() == o.output.model_dump_json()
        assert m.output.model == VMODEL
        assert m.messages == o.messages and m.metadata == o.metadata and m.events == o.events
        assert m.model_dump(exclude={"scores"}) == o.model_dump(exclude={"scores"})
        assert set(m.scores) == {"verify"}
    verdicts = {str(s.id): s.scores["verify"].value for s in merged.samples}
    assert {k for k, v in verdicts.items() if v == "I"} == WRONG
    # the original is not mutated by the bind; the header takes the vehicle's metrics
    assert all(not s.scores for s in original.samples)
    assert merged.eval.model == TARGET_KEY and merged.stats == original.stats
    assert [sc.name for sc in merged.results.scores] == ["verify"]
    assert merged.results.completed_samples == N
    # sample_usage reads the ORIGINAL target key off the merged log
    assert isv.sample_usage(merged.samples[3], TARGET_KEY) == (203, 43)
    res = isv.results_from_log(merged, primary_scorer="verify", target_model=TARGET_KEY,
                               row_facts=isv.bcb_row_facts, row_task=ROW_TASK)
    assert [r.item_id for r in res] == IDS
    assert {r.item_id for r in res if r.value == 0.0} == WRONG
    assert {r.row_facts["code_source"] for r in res} == {"python_fence", "bare_fence", "whole_completion"}
    assert {r.row_facts["sandbox_status"] for r in res} == {"ok"}
    assert [(r.prompt_tokens, r.completion_tokens) for r in res][:2] == [(200, 40), (201, 41)]
    assert [r.stop_reason for r in res if r.item_id == "7"] == ["max_tokens"]


def test_a_mismatched_missing_duplicate_or_foreign_sample_refuses_the_binding(tmp_path):
    original = fixture_log()
    task = synthetic_task(original)
    vehicle, _p = _vehicle(original, task, tmp_path, shuffle=None, max_connections=2)

    def mutate(fn):
        v = vehicle.model_copy(deep=True)
        fn(v)
        return v

    def _completion(v):
        v.samples[0].output.completion = v.samples[0].output.completion + " # drift"

    def _missing(v):
        del v.samples[5]

    def _duplicate(v):
        v.samples.append(v.samples[2].model_copy(deep=True))

    def _foreign(v):
        v.samples[1].id = 999

    def _no_score(v):
        v.samples[0].scores = {}

    def _model(v):
        v.samples[4].output.model = "another_model"

    def _errored(v):
        from inspect_ai.log import EvalError
        v.samples[6].error = EvalError(message="boom", traceback="", traceback_ansi="")

    for fn, needle in [(_completion, r"sample \('0', 1\): output.completion differs"),
                       (_missing, r"original sample \('5', 1\) is missing from the vehicle"),
                       (_duplicate, r"vehicle sample \('2', 1\) appears 2 times \(duplicate\)"),
                       (_foreign, r"vehicle sample \('999', 1\) is not in the original \(foreign\)"),
                       (_no_score, r"sample \('0', 1\): the vehicle carries no score"),
                       (_model, r"sample \('4', 1\): output.model differs"),
                       (_errored, r"vehicle sample \('6', 1\) errored")]:
        v = mutate(fn)
        problems = CV.binding_problems(original, v, primary_scorer="verify")
        assert any(re.search(needle, p) for p in problems), (needle, problems)
        with pytest.raises(CV.CpuVerifyError, match="refusing to bind"):
            CV.bind_scores(original, v, primary_scorer="verify")
    # the foreign id is ALSO a missing one (id 1 vanished with it): both named
    problems = CV.binding_problems(original, mutate(_foreign), primary_scorer="verify")
    assert any("('1', 1) is missing" in p for p in problems)
    # the untouched vehicle binds
    assert CV.binding_problems(original, vehicle, primary_scorer="verify") == []
    # a scored log is NOT an unscored generation log
    assert CV.unscored_problems(original) == []
    assert CV.unscored_problems(vehicle) and "already carry scores" in CV.unscored_problems(vehicle)[0]
    stripped = CV.strip_scores(vehicle)
    assert CV.unscored_problems(stripped) == [] and stripped.results.scores == []
    assert vehicle.samples[0].scores                                   # the strip is a copy


# --------------------------------------------------------------------------- #
# §2.2.3 the two-phase filing equals the coupled path's; the coupled path is byte-identical #
# --------------------------------------------------------------------------- #


_TIMING = re.compile(r"Ran (\d+) tests? in \d+\.\d+s")


def _stable(o):
    """unittest prints ``Ran N tests in 0.001s`` into the stderr the verifier keeps as
    ``verifier_explanation`` — the one run-to-run volatile byte of a row; masked."""
    if isinstance(o, dict):
        return {k: _stable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_stable(v) for v in o]
    if isinstance(o, str):
        return _TIMING.sub(r"Ran \1 tests in <T>s", o)
    return o


def _normalise_rows(rows):
    return [_stable({k: v for k, v in r.items() if k != "wall_clock_s"}) for r in rows]


def _no_modal_transport(monkeypatch):
    """The coupled path's transport install imports the owned provider, which needs Modal (absent
    from the client venv): the provider module is stubbed and the install made a no-op — the
    transport is the sandbox exec's concern, not the filing's."""
    from sisteer.capeval.tinycap import verifier_transport as VT

    _stub_sandbox_owned(monkeypatch)
    monkeypatch.setattr(VT, "install_exec_spill", lambda *a, **k: None)


def _coupled_filing(tmp_path, monkeypatch, *, coupled_log_path: Path):
    """The coupled path over a scored log (a fake eval returning it), through run_one_task."""
    original = fixture_log()
    store = _Store(tmp_path / "coupled" / "cells" / "k")
    seen: List[Dict[str, Any]] = []
    _no_modal_transport(monkeypatch)
    em = _Emitter()
    mp = pytest.MonkeyPatch()
    mp.setattr("inspect_ai.eval_async", _fake_eval_copying(coupled_log_path, seen))
    try:
        summary = asyncio.run(isv.run_one_task(
            _plan(original, generation_only=False), model=None, store=store, template={}, thinking=C.THINKING_OFF,
            concurrency=32, emitter=em, route_virtual_model=VMODEL, limits={}))
    finally:
        mp.undo()
    assert seen[0].get("score", True) is True and seen[0]["max_sandboxes"] == 32 and seen[0]["sandbox_cleanup"] is True
    return summary, store, em


def _two_phase_filing(tmp_path, monkeypatch):
    """generate (run_one_task, generation-only) -> the mechanism over a SHUFFLED vehicle -> the binding
    -> the merged log in place -> results_from_log + the lifted filing tail with the sidecar."""
    original = fixture_log()
    em = _Emitter()
    _out, store, row_dir, _seen, em = _run_generate(tmp_path, monkeypatch, em=em, store_dir=tmp_path / STORE / "cells" / "k")
    sidecar = CV.read_generation_sidecar(row_dir, CELL, ROW_TASK, STORE)
    log_path = Path(sidecar["log"])
    unscored = read_eval_log(str(log_path))
    task = synthetic_task(unscored)
    vehicle, _vp = _vehicle(unscored, task, tmp_path, shuffle=5, max_connections=4)
    merged = CV.bind_scores(unscored, vehicle, primary_scorer="verify")
    archive = CV.archived_log_path(row_dir, CELL, ROW_TASK, STORE)
    digests = CV.write_merged_log(merged, log_path=log_path, archive_path=archive)
    assert digests["original_sha256"] == CV.sha256_file(FIXTURE) == CV.sha256_file(archive)
    merged_disk = read_eval_log(str(log_path))
    plan = replace(isv.plan_from_facts(sidecar["plan"], task=task), generation_only=False)
    per_sample = {0: isv.results_from_log(merged_disk, sample_idx=0, primary_scorer=plan.primary_scorer,
                                          paired_ids=plan.paired_ids, attempt=1, max_tokens_used=plan.max_tokens,
                                          target_model=sidecar["target_model"], **isv._plan_readers(plan))}
    summary = isv._file_task_results(
        plan, store=store, task=ROW_TASK, per_sample=per_sample, em=em, thinking=sidecar["thinking"],
        ident=dict(sidecar["row_identity"]), sampling_ref=sidecar["sampling_ref"], reused=sidecar["logs_reused"],
        n_retried_requests=int(sidecar["n_retried_requests"]), t0=None, model=None,
        route_virtual_model=sidecar["route_virtual_model"],
        reasoning_parser=sidecar["reasoning_parser"], limits=dict(sidecar["limits"]),
        out_of_support=sidecar["out_of_support"], ledger_path=None, attempt_scope=None,
        generated={"n": sidecar["generated_calls"]}, generation=sidecar)
    return summary, store, em, sidecar, row_dir


def test_the_two_phase_filing_equals_the_coupled_paths_item_for_item(tmp_path, monkeypatch):
    """At the unit level: the SAME completions filed through generate -> verify produce
    the same outcome items (``TASK_OUTCOME_ITEM_FIELDS``), the same summary (its wall clock the GPU
    phase's), the same rows and a DONE marker, as the coupled path on the same fixture."""
    original = fixture_log()
    # the coupled log: the fixture's completions SCORED under the target provider's own key (M1b's
    # in-place scoring over the local sandbox) — the log a coupled run leaves behind
    coupled_log, coupled_path = _vehicle(original, synthetic_task(original), tmp_path / "c", mechanism=M1B)
    assert set(coupled_log.samples[0].model_usage) == {TARGET_KEY}
    summary_a, store_a, em_a = _coupled_filing(tmp_path, monkeypatch, coupled_log_path=coupled_path)
    summary_b, store_b, em_b, sidecar, _row_dir = _two_phase_filing(tmp_path, monkeypatch)
    fn = C.marker_filename(ROW_TASK)
    oa = json.loads((store_a.cell_dir / C.TASK_OUTCOMES_DIR / fn).read_text())
    ob = json.loads((store_b.cell_dir / C.TASK_OUTCOMES_DIR / fn).read_text())
    assert oa["items"] == ob["items"] and len(oa["items"]) == N
    assert set(oa["items"][0]) == set(C.TASK_OUTCOME_ITEM_FIELDS)
    assert {k: v for k, v in oa.items() if k != "items"} == {k: v for k, v in ob.items() if k != "items"}
    assert {i["item_id"] for i in oa["items"] if i["value"] == 0.0} == WRONG
    assert {k: v for k, v in summary_a.items() if k != "wall_clock_s"} == {k: v for k, v in summary_b.items() if k != "wall_clock_s"}
    assert summary_b["wall_clock_s"] == sidecar["wall_clock_s"] and summary_b["logs_reused"] == 0
    assert summary_a[C.METRIC_ACC] == summary_b[C.METRIC_ACC] == (N - len(WRONG)) / N
    # the request accounting is the GENERATION side's, carried through the sidecar: the same field set AND value as
    # the coupled path's (and as the pre-lift golden's)
    assert summary_b["n_requests"] == summary_a["n_requests"] == N == sidecar["n_requests"]
    assert json.loads(GOLDEN.read_text())["summary"]["n_requests"] == N
    assert summary_a["n_sandbox_ok"] == N and summary_a["n_code_source_python_fence"] == 8
    assert _normalise_rows(em_a.rows) == _normalise_rows(em_b.rows)
    assert len(em_b.rows) == N + 2                                     # N per-item acc rows + the two aggregates
    assert completion.is_done(store_a.cell_dir, ROW_TASK) and completion.is_done(store_b.cell_dir, ROW_TASK)
    sa = json.loads((store_a.cell_dir / C.SUMMARY_DIR / fn).read_text())
    sb = json.loads((store_b.cell_dir / C.SUMMARY_DIR / fn).read_text())
    assert {k: v for k, v in sa.items() if k != "wall_clock_s"} == {k: v for k, v in sb.items() if k != "wall_clock_s"}
    da = json.loads((store_a.cell_dir / C.DONE_DIR / fn).read_text())
    db = json.loads((store_b.cell_dir / C.DONE_DIR / fn).read_text())
    volatile = {"written_at", "ts", "wall_clock_s", "started_at", "finished_at"}
    assert {k: v for k, v in da.items() if k not in volatile} == {k: v for k, v in db.items() if k not in volatile}


def test_the_coupled_path_through_the_lifted_tail_is_byte_identical_to_the_pre_lift_golden(tmp_path, monkeypatch):
    """The refactor's own pin: the coupled path's outcome record, summary and rows on this fixture
    are compared to a golden captured from the pre-lift tail (the landing branch's base); a
    departure names the leaf. Rewriting the golden is a deliberate filing-shape change."""
    original = fixture_log()
    _log, coupled_path = _vehicle(original, synthetic_task(original), tmp_path / "c", mechanism=M1B)
    summary, store, em = _coupled_filing(tmp_path, monkeypatch, coupled_log_path=coupled_path)
    fn = C.marker_filename(ROW_TASK)
    got = json.loads(json.dumps({"summary": {k: v for k, v in summary.items() if k != "wall_clock_s"},
                                 "outcomes": json.loads((store.cell_dir / C.TASK_OUTCOMES_DIR / fn).read_text()),
                                 "rows": _normalise_rows(em.rows)}, default=str))
    # captured from the PRE-LIFT tree by tests/fixtures/tinycap_verification/capture_coupled_golden.py
    golden.compare_or_write(GOLDEN, got)


# --------------------------------------------------------------------------- #
# §2.2.2 (i) the mechanisms' mechanics                                           #
# --------------------------------------------------------------------------- #


def test_m1_raises_without_a_sandbox_and_m1b_scores_inside_a_sandbox_context(tmp_path):
    """The §0 prediction: a plain ``score_async`` finds no sandbox for the stock scorer; M1b's
    per-sample ``sandboxenv_context`` gives it one (a LOCAL sandbox here) and the mutation is in
    place — the identical binding assertions then hold trivially."""
    original = fixture_log()
    task = synthetic_task(original)
    with pytest.raises(Exception, match="(?i)sandbox"):
        asyncio.run(CV.run_mechanism(M1, original=original, coupled_task=task, max_sandboxes=2, log_dir=tmp_path / "m1"))
    scored, path = asyncio.run(CV.run_mechanism(M1B, original=original, coupled_task=task, max_sandboxes=2,
                                                log_dir=tmp_path / "m1b"))
    assert path.is_file() and path.name == f"vehicle_{M1B}.eval"
    assert CV.binding_problems(original, scored, primary_scorer="verify") == []
    assert {str(s.id) for s in scored.samples if s.scores["verify"].value == "I"} == WRONG
    merged = CV.bind_scores(original, scored, primary_scorer="verify")
    assert [s.scores["verify"].value for s in merged.samples] == [s.scores["verify"].value for s in scored.samples]
    assert all(not s.scores for s in original.samples)
    # M2 through the replay provider serves EVERY sample exactly once from the stored outputs
    model = CV.replay_model(original)
    assert str(model) == f"{CV.REPLAY_PROVIDER}/{VMODEL}" and len(model.api.outputs) == N
    with pytest.raises(CV.CpuVerifyError, match="not one of"):
        asyncio.run(CV.run_mechanism("M3", original=original, coupled_task=task, max_sandboxes=1, log_dir=tmp_path))


# --------------------------------------------------------------------------- #
# the contracts this half spells                                                  #
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=None)
def _harness_src() -> str:
    """The harness's SOURCE, read once per module (importing it needs Modal)."""
    return (REPO / "scripts" / "modal" / "tinycap_served.py").read_text(encoding="utf-8")


def _harness_pointer_fields() -> tuple:
    """The harness's ``POINTER_FIELDS`` read off its SOURCE."""
    src = _harness_src()
    m = re.search(r"^POINTER_FIELDS: Tuple\[str, \.\.\.\] = (\([^)]*\))", src, re.M)
    assert m, "tinycap_served.POINTER_FIELDS not found"
    return tuple(ast.literal_eval(m.group(1)))


def _harness_literal(name: str):
    """The harness's ``name``, read off its SOURCE (importing it needs Modal): since landing 2 the harness binds it
    to the contracts' spelling (``NAME = C.NAME``) — a literal of its own on that line is the drift this catches."""
    src = _harness_src()
    m = re.search(rf"^{name}(?:: Tuple\[str, \.\.\.\])? = (.+)$", src, re.M)
    assert m, f"tinycap_served.{name} not found"
    assert m.group(1) == f"C.{name}", f"tinycap_served.{name} = {m.group(1)} — not the contracts' spelling"
    return getattr(C, name)


def test_the_result_grammar_constants_are_the_harness_literals():
    """The driver writes RESULT lines since landing 2, so the harness's literals are spelled in the
    contracts and the harness binds its names to them — held by reading the harness source (importing
    it needs Modal)."""
    assert C.LADDER_REC_STATUS == _harness_literal("LADDER_REC_STATUS")
    assert C.REC_OK_STATUSES == _harness_literal("REC_OK_STATUSES")
    assert C.RESULT_FILENAME == _harness_literal("RESULT_FILENAME")
    assert C.RESULT_COMPLETE == _harness_literal("RESULT_COMPLETE")
    assert C.REC_STATUS_GENERATED not in C.REC_OK_STATUSES
    for name in ("LADDER_REC_STATUS", "REC_OK_STATUSES", "RESULT_FILENAME", "RESULT_COMPLETE"):
        assert name in C.__all__


def test_the_verifier_pointer_is_the_harness_pointer_plus_phase_and_never_a_row_pointer(tmp_path):
    assert CV.VERIFY_POINTER_FIELDS == (*_harness_pointer_fields(), "phase")
    owner = scope.Owner("run-1", CELL, VERIFY_FCID)
    p = CV.write_verify_pointer(tmp_path / "row", owner=owner, name="n", model="m", mode="thinking_off",
                                out_dir="/vol/out", controls=[NAME], tree_sha="ab" * 20)
    doc = json.loads(p.read_text())
    assert p == tmp_path / "row" / "watch" / f"attempt_{VERIFY_FCID}.json"
    assert set(doc) == set(CV.VERIFY_POINTER_FIELDS) and doc["phase"] == C.VERIFICATION_PHASE_VERIFY
    assert doc["cell"] == CELL and doc["function_call_id"] == VERIFY_FCID and doc["run_id"] == "run-1"
    assert "cells" not in doc and "row_attempt" not in doc                # one owner: never a row pointer
    assert doc["sandbox_record"] == str(tmp_path / "row" / "watch" / f"sandboxes_{VERIFY_FCID}__{CELL}.jsonl")
    assert doc["sandbox_app_name"] == C.BCB_SANDBOX_APP_NAME
    # the harness exports the record path for the verifier process; an exported path is what the pointer names
    p2 = CV.write_verify_pointer(tmp_path / "row2", owner=owner, name="n", model="m", mode="thinking_off",
                                 out_dir="/vol/out", controls=[NAME], tree_sha="ab" * 20, sandbox_record="/vol/r.jsonl")
    assert json.loads(p2.read_text())["sandbox_record"] == "/vol/r.jsonl"


def test_the_verified_block_the_pending_block_and_the_result_lines_are_the_contracts_spellings():
    block = CV.verified_block(mechanism=M2, function_call_id=VERIFY_FCID, started_at=1.0, finished_at=2.0,
                              sandbox_facts=C.bcb_sandbox_facts(), max_sandboxes=8, n_items=N,
                              vehicle_log_sha256="0" * 64)
    assert tuple(block) == C.VERIFICATION_PASS_VERIFIED_FIELDS and block["state"] == C.VERIFICATION_PASS_VERIFIED
    # tinycap-verify-rate-limit (2026-09-30): the two tree shas are RECORDED facts (None when unknown), the field set grew by
    # them, and the readers accept the pre-change nine-key set as a verified block too
    assert block["verifier_tree_sha"] is None and block["generation_tree_sha"] is None
    stamped = CV.verified_block(mechanism=M2, function_call_id=VERIFY_FCID, started_at=1.0, finished_at=2.0,
                                sandbox_facts=C.bcb_sandbox_facts(), max_sandboxes=4, n_items=N, vehicle_log_sha256="0" * 64,
                                verifier_tree_sha="cd" * 20, generation_tree_sha="ab" * 20)
    assert tuple(stamped) == C.VERIFICATION_PASS_VERIFIED_FIELDS and stamped["verifier_tree_sha"] == "cd" * 20 and stamped["generation_tree_sha"] == "ab" * 20
    assert C.VERIFICATION_PASS_VERIFIED_FIELDS == C.VERIFICATION_PASS_VERIFIED_FIELDS_V1 + (
        "verifier_tree_sha", "generation_tree_sha", "executed_image", "executed_verify_timeout_s", "binding_version")
    assert block["executed_image"] is None and block["executed_verify_timeout_s"] is None and block["binding_version"] is None
    legacy = {k: v for k, v in block.items() if k in C.VERIFICATION_PASS_VERIFIED_FIELDS_V1}
    assert C.verified_block_keys_problem(block) is None and C.verified_block_keys_problem(legacy) is None
    assert "'stray'" in C.verified_block_keys_problem({**legacy, "stray": 1}) and "legacy set" in C.verified_block_keys_problem({"state": "verified"})
    assert C.VERIFICATION_PASS_VERIFIED_FIELD_SETS == (C.VERIFICATION_PASS_VERIFIED_FIELDS, C.VERIFICATION_PASS_VERIFIED_FIELDS_V1)
    assert C.validate_row_engine_record({
        "row_engine": True, "standup_id": "s", "engine_cells": [CELL], "max_loras": 2, "residual_slot_armed": True,
        "compile_cache": {"hit": True, "key": "k", "vllm_cache_dir": None}, "row_dir": "/r",
        "verification_pass": block})["verification_pass"] == block
    with pytest.raises(CV.CpuVerifyError, match="not one of"):
        CV.verified_block(mechanism="M9", function_call_id="f", started_at=1.0, finished_at=2.0, sandbox_facts={},
                          max_sandboxes=1, n_items=1, vehicle_log_sha256="x")
    assert CV.pending_block([ROW_TASK]) == {"state": C.VERIFICATION_PASS_PENDING, "tasks": [ROW_TASK]}
    with pytest.raises(CV.CpuVerifyError):
        CV.pending_block([])
    gen_line = CV.result_line(name=CELL, status=C.REC_STATUS_GENERATED, tree_sha="ab" * 20,
                              extra={"phase": C.VERIFICATION_PHASE_GENERATE, "tasks": ROW_TASK})
    assert gen_line.startswith(f"{CV.RESULT_TAG} name={CELL} rc=0 result={C.REC_STATUS_GENERATED} "
                               f"status={C.REC_STATUS_GENERATED} sha={'ab' * 20} at=")
    assert gen_line.endswith(f" phase={C.VERIFICATION_PHASE_GENERATE} tasks={ROW_TASK}")
    ok_line = CV.result_line(name=CELL, status="OK", tree_sha="ab" * 20, extra={"verified_by": VERIFY_FCID})
    assert f" result={C.RESULT_COMPLETE} status=OK " in ok_line and ok_line.endswith(f" verified_by={VERIFY_FCID}")
    assert C.REC_STATUS_GENERATED not in C.REC_OK_STATUSES
    assert CV.default_max_sandboxes(32) == 8 and CV.default_max_sandboxes(6) == 6 and CV.default_max_sandboxes(1) == 1
    with pytest.raises(CV.CpuVerifyError):
        CV.default_max_sandboxes(0)
    assert CV.DEFAULT_MECHANISM == M2


def test_the_sidecar_and_marker_readers_hold_the_exact_key_sets(tmp_path):
    row = tmp_path / "row"
    sidecar = {k: None for k in CV.GENERATION_SIDECAR_FIELDS}
    sidecar.update(task=ROW_TASK, cell=CELL, store=STORE, log="/x.eval", log_sha256="0" * 64, n_items=1, n_expected_full=1,
                   generated_at=1.0, generation_fcid=None, regimes={}, client={}, limits={}, row_identity={}, plan={},
                   sample_ids=["0"], n_requests=1, n_retried_requests=0)
    marker = {k: sidecar[k] for k in C.PENDING_MARKER_FIELDS}
    sc, mk = CV.write_generation_pending(row, sidecar=sidecar, marker=marker)
    assert sc == CV.generation_sidecar_path(row, CELL, ROW_TASK, STORE) and mk == CV.pending_marker_path(row, CELL, ROW_TASK, STORE)
    assert sc.name == f"{CELL}__inspect__bigcodebench_full__{STORE}.json"
    assert mk.name == f"pending_{CELL}__inspect__bigcodebench_full__{STORE}.json"
    assert CV.read_pending_marker(row, CELL, ROW_TASK, STORE) == marker
    assert CV.read_generation_sidecar(row, CELL, ROW_TASK, STORE) == sidecar
    assert CV.pending_marker_exists(row, CELL, ROW_TASK, STORE)
    # another store of the SAME cell (the in-run repeat) has its own files — no collision
    assert not CV.pending_marker_exists(row, CELL, ROW_TASK, STORE + "-rep2")
    with pytest.raises(CV.CpuVerifyError, match="store root's basename"):
        CV.pending_marker_path(row, CELL, ROW_TASK, "a/b")
    with pytest.raises(CV.CpuVerifyError, match="unknown \\['extra'\\]"):
        CV.write_generation_pending(row, sidecar={**sidecar, "extra": 1}, marker=marker)
    with pytest.raises(CV.CpuVerifyError, match="missing \\['log'\\]"):
        CV.write_generation_pending(row, sidecar=sidecar, marker={k: v for k, v in marker.items() if k != "log"})
    with pytest.raises(CV.CpuVerifyError, match="disagree on n_items"):
        CV.write_generation_pending(row, sidecar=sidecar, marker={**marker, "n_items": 2})
    mk.write_text(json.dumps({**marker, "stray": 1}))
    with pytest.raises(CV.CpuVerifyError, match="unknown \\['stray'\\]"):
        CV.read_pending_marker(row, CELL, ROW_TASK, STORE)
    with pytest.raises(CV.CpuVerifyError, match="no pending marker"):
        CV.read_pending_marker(row, "other", ROW_TASK, STORE)
    with pytest.raises(CV.CpuVerifyError, match="names \\('none', 'inspect/bigcodebench_full', 'bcbw-synthetic-shared'\\), not \\('other'"):
        shutil.copyfile(sc, CV.generation_sidecar_path(row, "other", ROW_TASK, STORE))
        CV.read_generation_sidecar(row, "other", ROW_TASK, STORE)
    assert CV.read_generation_sidecar_if_present(row, "nobody", ROW_TASK, STORE) is None


# --------------------------------------------------------------------------- #
# §2.2.2 (i) the G5 probe, locally over a synthetic scored store                 #
# --------------------------------------------------------------------------- #


def test_the_cpu_verify_probe_runs_the_three_mechanisms_and_chooses_m2(tmp_path):
    """``scripts/modal/tinycap_cpu_verify_probe.py --local <store>`` over a store whose one BigCodeBench log
    is the fixture SCORED (the store's own verdicts): M1 fails (recorded, the §0 prediction), M1b and M2
    reproduce every verdict on a scores-stripped copy AND bind; both pass, so M2 (public API) is chosen;
    the record carries every mechanism's facts under ``cpu_verify_probe/1``."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("tinycap_cpu_verify_probe",
                                                  REPO / "scripts" / "modal" / "tinycap_cpu_verify_probe.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    original = fixture_log()
    scored, _p = _vehicle(original, synthetic_task(original), tmp_path / "s", mechanism=M1B)
    store_root = tmp_path / "store"
    log_dir = C.inspect_log_dir(store_root / "cells" / "k", ROW_TASK, 0)
    log_dir.mkdir(parents=True)
    from inspect_ai.log import write_eval_log
    write_eval_log(scored, str(log_dir / "scored.eval"), format="eval")
    out = tmp_path / "logs" / "cpu_verify_probe.json"
    record = probe.probe_store(store_root, n=10, out_path=out, repo_root=REPO,
                               task_factory=lambda name, *, log_ids, repo_root, sandbox: synthetic_task(original, ids=log_ids),
                               log=lambda *_a, **_k: None)
    assert out.is_file() and json.loads(out.read_text())["chosen"] == record["chosen"] == M2
    assert record["schema"] == probe.PROBE_SCHEMA == "cpu_verify_probe/1" and record["n"] == 10
    assert record["ids"] == IDS[:10] and record["max_sandboxes"] == 8
    by = {r["mechanism"]: r for r in record["mechanisms"]}
    assert list(by) == list(C.CPU_VERIFY_MECHANISMS)
    assert by[M1]["ok"] is False and "sandbox" in by[M1]["error"].lower()
    for m in (M1B, M2):
        assert by[m]["ok"] and by[m]["verdicts_reproduced"] and by[m]["binding_ok"] and by[m]["n_mismatched"] == 0
        assert re.fullmatch(r"[0-9a-f]{64}", by[m]["vehicle_log_sha256"])
    assert record["reason"].startswith("both pass; M2")
    # a task factory that ignores the ids (a 12-sample task over a 10-sample head) is caught by the binding
    # test: M2's vehicle carries two FOREIGN samples, so M2 fails the probe and M1b (in place) is the choice
    wide = probe.probe_store(store_root, n=10, out_path=None, repo_root=REPO,
                             task_factory=lambda name, *, log_ids, repo_root, sandbox: synthetic_task(original),
                             log=lambda *_a, **_k: None)
    assert wide["chosen"] == M1B and wide["reason"] == "the only mechanism to pass both tests"
    m2 = next(r for r in wide["mechanisms"] if r["mechanism"] == M2)
    assert m2["ok"] and m2["binding_ok"] is False and any("foreign" in p for p in m2["binding_problems"])
    assert {k for k, v in record["reference_verdicts"].items() if v == "I"} == {f"('{i}', 1)" for i in WRONG if int(i) < 10}
    # the store was read only: its one log is untouched
    assert sorted(q.name for q in log_dir.iterdir()) == ["scored.eval"]
    # the CLI path (`--local` over the same store) builds the REAL thin task (inspect_evals + the pinned pin): not run here


def test_the_probes_modal_image_installs_the_pinned_client_lock_and_never_falls_back_to_a_bare_image(tmp_path):
    """Simplify FYI (landing 2): the probe's Modal image pointed at `configs/capeval/client-requirements.txt`,
    which does not exist (the lock is `contracts.CLIENT_LOCK_TXT`), and fell back to a bare debian_slim — a
    container without inspect_ai. The image is built from the lock, and a tree without the lock REFUSES."""
    import importlib.util
    import types

    path = REPO / "scripts" / "modal" / "tinycap_cpu_verify_probe.py"
    spec = importlib.util.spec_from_file_location("tinycap_cpu_verify_probe", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    calls: List[Any] = []

    class FakeImage:
        def __init__(self, tag):
            self.tag = tag

        @staticmethod
        def debian_slim(python_version):
            calls.append(("debian_slim", python_version))
            return FakeImage("slim")

        def pip_install_from_requirements(self, requirements):
            calls.append(("pip_install_from_requirements", requirements))
            return FakeImage("client")

    fake_modal = types.SimpleNamespace(Image=FakeImage)
    image = probe.client_image(fake_modal, REPO)
    assert image.tag == "client"
    assert calls == [("debian_slim", "3.12"), ("pip_install_from_requirements", str(REPO / C.CLIENT_LOCK_TXT))]
    assert (REPO / C.CLIENT_LOCK_TXT).is_file() and C.CLIENT_LOCK_TXT.endswith("tinycap-client-requirements.txt")
    # no lock -> a named refusal, never a bare image
    calls.clear()
    with pytest.raises(FileNotFoundError, match="CLIENT_LOCK_TXT"):
        probe.client_image(fake_modal, tmp_path)
    assert calls == []
    # the source pin: the wrong literal is gone, the image is built by the ONE function
    src = path.read_text(encoding="utf-8")
    assert "client-requirements.txt" not in src.replace("tinycap-client-requirements.txt", "")
    assert "C.CLIENT_LOCK_TXT" in src and "_image = client_image(modal, REPO)" in src
