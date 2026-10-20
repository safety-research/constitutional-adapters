"""The Petri legs stop at the FIRST credential rejection.

The federated inspect provider's latch (``sisteer.eval.anthropic_wif``, the WIF driver sweep) made every later request of the leg a
typed error before the wire -- and the leg went on: inspect walked every remaining scenario as an instant error under
``fail_on_error=False``, ``_inspect_driver`` launched the next eval group, converted the logs and FILED the scenarios judged before the
rejection as if the leg had ended well, and the leg's terminal line was the INCOMPLETE / hard-error diagnosis ("fix the REQUEST"), not
the credential. Now (1) the first rejection aborts the running inspect tasks through inspect's own cancel handle
(tests/test_credential_stop_sweep.py), (2) the driver reads the latch after each eval_set, launches no later group, converts no log and
raises the typed error, (3) the runner's H9 ledger names the credential stop and files no row, and (4) ``petri_run.py cmd_run`` ends the
leg process with the typed text as its terminal line (``vault.credential_exit``), never a traceback and never a results.json. CPU only:
inspect's seams are stubbed as in test_petri_auditor_fallback._drive_inspect; the latch is set by hand; no network, no key."""
from __future__ import annotations

import sys
import types
from dataclasses import replace
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from sisteer import vault
from sisteer.capeval import store as cs
from sisteer.capeval.tiers import petri_runner as pr
from sisteer.eval import anthropic_wif as w
from test_petri_auditor_fallback import _cell_under
from test_petri_store_shape import BEHAVIORS, PROMPT_FILES, SIDS, world  # noqa: F401 -- the fixture
import test_petri_overflow_rerun as tor


def _credential_error(status: int = 401):
    """The API's credential rejection on a request: body shape only, no key anywhere."""
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    api_type = "authentication_error" if status == 401 else "permission_error"
    body = {"type": "error", "error": {"type": api_type, "message": "synthetic"}}
    cls = anthropic.AuthenticationError if status == 401 else anthropic.PermissionDeniedError
    return cls(f"Error code: {status} - {body}", response=httpx.Response(status, request=req), body=body)


def _log_stub():
    """An eval_set log the converter would accept (``_dumped_log`` reads ``samples``, ``model_dump`` and ``location``) -- what the
    driver must NOT convert once the credential is rejected."""
    return SimpleNamespace(location="synthetic.eval", samples=[], model_dump=lambda mode="json": {"eval": {"metadata": {}}})


def _inspect_seams(monkeypatch, eval_set):
    """test_petri_auditor_fallback._drive_inspect's seam setup with a caller-supplied ``eval_set``; returns the engine."""
    inspect_pkg = types.ModuleType("inspect_ai")
    inspect_pkg.eval_set = eval_set
    inspect_pkg.task_with = lambda task, *, model_roles, metadata: task
    log_mod = types.ModuleType("inspect_ai.log"); log_mod.read_eval_log = lambda location: None
    model_mod = types.ModuleType("inspect_ai.model")

    class FakeGenerateConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def merge(self, other):
            return self
    model_mod.GenerateConfig = FakeGenerateConfig
    model_mod.get_model = lambda name, **kwargs: SimpleNamespace(name=name, config=None)
    bloom = types.ModuleType("petri_bloom"); bloom.bloom_audit = lambda **kwargs: object()
    for name, mod in (("inspect_ai", inspect_pkg), ("inspect_ai.log", log_mod), ("inspect_ai.model", model_mod), ("petri_bloom", bloom)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(pr, "_require_inspect", lambda: inspect_pkg)
    monkeypatch.setattr(pr, "require_agent_loop_entry", lambda engine, cell: None)
    monkeypatch.setattr(pr, "eval_concurrency", lambda spec, engine: 4)
    monkeypatch.setattr(pr, "install_retry_sleep_cap", lambda: None)
    monkeypatch.setattr(pr, "eval_set_bounds", lambda: {})
    return SimpleNamespace(petri_target=lambda cell: SimpleNamespace(config=None))


def _bare(world):
    return replace(world.spec, auditor_fallback_models=[], judge_fallback_models=[],
                   auditor_refusal_fallback_models=[], judge_refusal_fallback_models=[])


def _two_groups(tmp_path):
    """Two tasks with different resume subsets -> two eval groups (the driver groups by ``scenario_ids``)."""
    common = dict(log_dir=tmp_path / "logs", metadata={}, options={})
    return [pr.EvalTaskSpec(behavior_id=BEHAVIORS[0], dir_key=BEHAVIORS[0], behavior_dir=tmp_path / "b0", scenario_ids=None, **common),
            pr.EvalTaskSpec(behavior_id=BEHAVIORS[1], dir_key=BEHAVIORS[1], behavior_dir=tmp_path / "b1", scenario_ids=(SIDS[0],), **common)]


def _latch_now(monkeypatch, status: int = 401) -> vault.CredentialLatch:
    """What a federated provider client does inside the eval when the API rejects the credential: the process-wide latch holds it."""
    latch = vault.CredentialLatch("lowprio")
    latch.rejection(_credential_error(status))
    monkeypatch.setattr(w, "_LATCH", latch)
    return latch


# --- (2) the driver ------------------------------------------------------------------------------------------------------------


def test_the_driver_ends_the_leg_at_the_group_whose_eval_met_the_rejection_launching_no_later_group_and_converting_no_log(
        world, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(w, "_LATCH", None)
    spec = _bare(world); cell = _cell_under(spec, world)
    calls = []

    def eval_set(group, **kwargs):
        calls.append(len(group))
        _latch_now(monkeypatch)                            # a federated client met the rejection during THIS eval
        return True, [_log_stub() for _ in group]          # fail_on_error=False: the eval_set "succeeded"; the logs must NOT be converted
    engine = _inspect_seams(monkeypatch, eval_set)
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        pr._inspect_driver(_two_groups(tmp_path), spec=spec, cell=cell, engine=engine)
    assert ei.value is w.credential_rejected() and ei.value.pool == "lowprio" and ei.value.status == 401
    assert calls == [1]                                    # the second group never launched
    out = capsys.readouterr().out
    assert "CREDENTIAL REJECTED" in out and "1 of 2 eval group(s) launched" in out and "no row filed" in out


def test_the_driver_types_a_rejection_latched_during_an_eval_set_that_raised(world, tmp_path, monkeypatch):
    """inspect ends a cancelled task's eval_set as a KeyboardInterrupt: the credential stop is read FIRST and the typed error carries
    what eval_set raised as its cause."""
    monkeypatch.setattr(w, "_LATCH", None)
    spec = _bare(world); cell = _cell_under(spec, world)
    calls = []

    def eval_set(group, **kwargs):
        calls.append(len(group))
        _latch_now(monkeypatch, 403)
        raise KeyboardInterrupt
    engine = _inspect_seams(monkeypatch, eval_set)
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        pr._inspect_driver(_two_groups(tmp_path), spec=spec, cell=cell, engine=engine)
    assert isinstance(ei.value.__cause__, KeyboardInterrupt) and ei.value.status == 403 and calls == [1]


def test_the_driver_is_unchanged_when_no_credential_was_rejected(world, tmp_path, monkeypatch):
    monkeypatch.setattr(w, "_LATCH", None)
    spec = _bare(world); cell = _cell_under(spec, world)
    calls = []

    def eval_set(group, **kwargs):
        calls.append(len(group))
        return True, [_log_stub() for _ in group]
    engine = _inspect_seams(monkeypatch, eval_set)
    out = pr._inspect_driver(_two_groups(tmp_path), spec=spec, cell=cell, engine=engine)
    assert calls == [1, 1] and len(out) == 2 and all(o["location"] == "synthetic.eval" for o in out)
    # and an eval_set that raises for another reason still propagates as itself
    def exploding(group, **kwargs):
        raise RuntimeError("auditor API meltdown")
    engine = _inspect_seams(monkeypatch, exploding)
    with pytest.raises(RuntimeError, match="meltdown"):
        pr._inspect_driver(_two_groups(tmp_path), spec=spec, cell=cell, engine=engine)


# --- (3) the runner's ledger -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("stop", [
    vault.JudgeCredentialRejected("lowprio", 401, "AuthenticationError", api_error_type="authentication_error"),
    _credential_error(403),                               # the SDK's own rejection out of a driver is the same class of stop
])
def test_a_credential_stop_out_of_the_driver_registers_the_ledger_with_the_credential_reason_files_no_row_and_writes_no_results(
        world, tmp_path, stop):
    cell = world.cells["none"]

    def rejected(tasks, *, spec, cell, engine):
        raise stop
    store = cs.CellStore(tmp_path / "sweep", pr.petri_cell_key(cell))
    with pytest.raises(type(stop)) as ei:
        pr._run(object(), world.spec, cell, store, seed=world.spec.bloom_options.seed, driver=rejected,
                config_root=world.config_root, bundle_dir=world.bundle.bundle_dir,
                work_root=tmp_path / "work", log_root=tmp_path / "logs", prompt_files=PROMPT_FILES)
    assert ei.value is stop
    assert pr.is_credential_stop(stop) and not pr.is_credential_stop(RuntimeError("meltdown"))
    entries = pr.read_failure_ledger(store)["unresolved"]
    assert len(entries) == len(BEHAVIORS) * len(SIDS) * 3                      # every missing required dimension, registered (H9)
    assert all(e["reason"].startswith(pr.CREDENTIAL_STOP_REASON_PREFIX) for e in entries)
    assert all(("rejected the lowprio pool's credential" in e["reason"]) or ("Error code: 403" in e["reason"]) for e in entries)
    assert not any("driver/eval failure" in e["reason"] for e in entries)
    assert store._parsed_rows() == [] and not store.results_path.exists() and not store.is_done()


# --- (4) the leg process ---------------------------------------------------------------------------------------------------------


def test_cmd_run_ends_the_leg_with_the_typed_credential_line_and_writes_no_results(tmp_path, monkeypatch):
    """The terminal line of the leg's log is the typed text (``[petri] the API rejected the <pool> pool's credential: ...``) -- what
    the drive's leg log carries and the grading daemon's `credential` kind reads -- never a traceback; every other failure passes."""
    typed = vault.JudgeCredentialRejected("lowprio", 401, "WorkloadIdentityError")

    def rejected_run_petri(_engine, _spec, _cell, _store, _seed):
        raise typed
    args, spec, cell_dir = tor._wire_cmd_run(tmp_path, monkeypatch, rejected_run_petri)
    tor._write_ledger(cell_dir, [tor._ledger_entry(pr.overflow_reason("maximum context length is"), sid="scenario_a")])
    with pytest.raises(SystemExit) as ei:
        tor.prun.cmd_run(args, spec)
    assert str(ei.value) == f"[petri] {typed}" and "rejected the lowprio pool's credential" in str(ei.value)
    assert not (cell_dir / "results.json").exists()
    # the SDK's own rejection out of the runner: the same typed line
    raw = _credential_error(403)

    def raw_run_petri(_engine, _spec, _cell, _store, _seed):
        raise raw
    (tmp_path / "raw").mkdir()
    args, spec, cell_dir = tor._wire_cmd_run(tmp_path / "raw", monkeypatch, raw_run_petri)
    tor._write_ledger(cell_dir, [tor._ledger_entry(pr.overflow_reason("maximum context length is"), sid="scenario_a")])
    with pytest.raises(SystemExit) as ei:
        tor.prun.cmd_run(args, spec)
    assert str(ei.value).startswith("[petri] the API rejected the ") and "PermissionDeniedError (HTTP 403, permission_error)" in str(ei.value)
    # every other failure passes as before
    def exploding_run_petri(_engine, _spec, _cell, _store, _seed):
        raise RuntimeError("auditor API meltdown")
    (tmp_path / "other").mkdir()
    args, spec, cell_dir = tor._wire_cmd_run(tmp_path / "other", monkeypatch, exploding_run_petri)
    tor._write_ledger(cell_dir, [tor._ledger_entry(pr.overflow_reason("maximum context length is"), sid="scenario_a")])
    with pytest.raises(RuntimeError, match="meltdown"):
        tor.prun.cmd_run(args, spec)
