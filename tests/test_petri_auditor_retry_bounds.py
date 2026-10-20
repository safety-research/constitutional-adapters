"""The Petri driver bounds a wedged auditor/judge API call and continues
past an errored sample.

Background: inspect-ai 0.3.251 retries a transiently-failing model call
FOREVER by default (GenerateConfig.max_retries=None, timeout=None; the
backoff sleeps cap at 30 min). Eight live new-fitted legs sat 10+ h in
``claude-sonnet-5 retry N (retrying in 1,800 seconds) [APIStatusError
200]`` -- a mid-stream ``overloaded_error`` on a streaming response, which
the anthropic provider classifies as transient. The remedy is three
eval_set kwargs (petri_runner.eval_set_bounds): max_retries 14, a 7200 s
whole-request timeout, and fail_on_error=False so the exhausted sample is
filed as a sample error (failure ledger -> auditor abstention under the
scoring rule) and the leg completes.

CPU-only. The inspect_ai-backed checks skip when it is not installed; they
never call an API (mockllm + config resolution only)."""

from inspect import signature

import pytest

from sisteer.capeval.tiers import petri_runner as pr
from test_petri_store_shape import world  # noqa: F401  (session fixture)


def test_bounds_are_the_d186_values():
    b = pr.eval_set_bounds()
    assert b == {"fail_on_error": False, "max_retries": 14, "timeout": 7200}
    assert pr.AUDITOR_MAX_RETRIES == 14
    assert pr.AUDITOR_REQUEST_TIMEOUT_S == 7200


def test_timeout_binds_before_max_retries():
    # inspect's backoff: wait_exponential_jitter(initial=3, max=1800): sleeps
    # 3, 6, 12, ... capped at 1800 s (jitter aside). The 2 h budget is what
    # ends a wedged call; 14 retries is the belt behind it.
    sleeps = [min(3 * 2 ** k, 1800) for k in range(pr.AUDITOR_MAX_RETRIES)]
    cumulative = [sum(sleeps[:i + 1]) for i in range(len(sleeps))]
    # 3+6+...+1536 = 3069 s after 10 retries, then +1800 s each: 4869,
    # 6669, 8469 -> the 7200 s budget expires during the 13th sleep, one
    # short of the 14-retry belt (the live logs showed exactly this ladder:
    # "retrying in 1,536 seconds" at retry 10, 1,800 from retry 11 on).
    assert cumulative[9] == 3069
    first_over = next(i + 1 for i, c in enumerate(cumulative)
                      if c >= pr.AUDITOR_REQUEST_TIMEOUT_S)
    assert first_over == 13
    assert first_over <= pr.AUDITOR_MAX_RETRIES


def test_errored_sample_goes_to_the_ledger_not_the_rows(world):
    # With fail_on_error=False inspect logs the sample with `error` set and
    # no scores; the converter files it as a sample error — after the
    # overflow check, before the score branch (convert_eval_log's inlined
    # disposition order; main has no separate _sample_disposition helper)
    # — on EVERY required dimension, never as a scored row.
    from test_petri_store_shape import SIDS, convert, sample
    cell = world.cells["si_safety__psr_kl_all"]
    b = "self_preservation"
    err = {"message": "APIStatusError 200 overloaded", "traceback": "",
           "traceback_ansi": ""}
    rows, ledger = convert(world, cell, b, [sample(SIDS[0], None, error=err)])
    assert rows == []
    assert {e.key() for e in ledger} == {
        (b, SIDS[0], 1, dim) for dim in pr.required_metrics(b)}
    assert all(e.reason.startswith("sample error (auditor/judge failure)")
               and "APIStatusError 200 overloaded" in e.reason
               for e in ledger)


def test_bounds_are_accepted_by_inspect_eval_set():
    inspect_ai = pytest.importorskip("inspect_ai")
    from inspect_ai.model import GenerateConfig
    params = signature(inspect_ai.eval_set).parameters
    assert "fail_on_error" in params
    fields = GenerateConfig.model_fields
    assert "max_retries" in fields and "timeout" in fields
    cfg = GenerateConfig(**{k: v for k, v in pr.eval_set_bounds().items()
                            if k != "fail_on_error"})
    assert (cfg.max_retries, cfg.timeout) == (14, 7200)


def test_bounds_reach_role_models_as_operational_config():
    """The auditor/judge/target are ROLE models (never the active model):
    inspect copies max_retries/timeout from the active eval-level config
    onto them in Model._resolve_config. Exercised with mockllm, no API."""
    pytest.importorskip("inspect_ai")
    from inspect_ai.model import GenerateConfig, get_model
    from inspect_ai.model._model import set_active_generate_config
    bounds = {k: v for k, v in pr.eval_set_bounds().items()
              if k != "fail_on_error"}
    set_active_generate_config(GenerateConfig(**bounds))
    try:
        role_model = get_model("mockllm/model")
        resolved = role_model._resolve_config(None)
        assert resolved.max_retries == 14
        assert resolved.timeout == 7200
    finally:
        set_active_generate_config(GenerateConfig())


def test_bounds_do_not_change_eval_set_task_identity():
    """A restarted leg must RESUME its prior logs (re-running only the
    scenarios it had not completed): the bounds must therefore be outside
    inspect's eval-set task identity. Pinned to the private exclusion set
    so a rename fails loudly here rather than silently re-running every
    scenario of every restarted cell."""
    pytest.importorskip("inspect_ai")
    from inspect_ai._eval.evalset import _GENERATE_CONFIG_FIELDS_TO_EXCLUDE
    assert {"max_retries", "timeout", "max_connections"} <= set(
        _GENERATE_CONFIG_FIELDS_TO_EXCLUDE)


# --- the sleep ceiling as a tunable -----------------------------------------
#
# Legs were observed parked in 1,800 s sleeps (retry 10-19) for up to
# 30 min AFTER the shed had cleared. inspect hardcodes the ceiling inside
# model_retry_config; the runner records it as a tunable and can install a
# shorter one per leg.


def test_sleep_cap_defaults_to_inspects_and_reads_the_env(monkeypatch):
    monkeypatch.delenv(pr.PETRI_RETRY_SLEEP_CAP_ENV, raising=False)
    assert pr.retry_sleep_cap_s() == pr.INSPECT_RETRY_SLEEP_CAP_S == 1800
    monkeypatch.setenv(pr.PETRI_RETRY_SLEEP_CAP_ENV, "300")
    assert pr.retry_sleep_cap_s() == 300
    monkeypatch.setenv(pr.PETRI_RETRY_SLEEP_CAP_ENV, "2")
    with pytest.raises(pr.PetriRunnerError, match="below"):
        pr.retry_sleep_cap_s()
    monkeypatch.setenv(pr.PETRI_RETRY_SLEEP_CAP_ENV, "five")
    with pytest.raises(pr.PetriRunnerError, match="whole number"):
        pr.retry_sleep_cap_s()


def test_ladder_shows_which_bound_binds_under_each_cap():
    ladder = pr.retry_backoff_ladder(pr.INSPECT_RETRY_SLEEP_CAP_S)
    assert ladder[:11] == [3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 1800]
    cumulative = [sum(ladder[:i + 1]) for i in range(len(ladder))]
    assert cumulative[9] == 3069  # the docstring's number
    assert next(i + 1 for i, c in enumerate(cumulative)
                if c >= pr.AUDITOR_REQUEST_TIMEOUT_S) == 13  # timeout binds first
    short = pr.retry_backoff_ladder(300)
    assert short[:8] == [3, 6, 12, 24, 48, 96, 192, 300]
    assert sum(short) == 381 + 7 * 300 < pr.AUDITOR_REQUEST_TIMEOUT_S  # 14 retries bind first


def test_install_cap_reaches_inspects_model_module(monkeypatch):
    pytest.importorskip("inspect_ai")
    from inspect_ai.model import _model as im
    from inspect_ai.model import _retry as ir
    from tenacity import wait_exponential_jitter
    original = getattr(ir.model_retry_config, "__petri_wrapped__", None) \
        or ir.model_retry_config
    try:
        monkeypatch.delenv(pr.PETRI_RETRY_SLEEP_CAP_ENV, raising=False)
        assert pr.install_retry_sleep_cap() is None  # default cap: nothing installed
        assert ir.model_retry_config is original

        assert pr.install_retry_sleep_cap(300) == 300
        # Model._generate calls the name it imported into _model: that one
        # must be the wrapper, and a call without `wait` gets the capped ladder
        assert im.model_retry_config is ir.model_retry_config
        cfg = im.model_retry_config("m", 14, 7200, lambda e: True,
                                    lambda e: None, lambda n, rs: None)
        assert cfg["wait"].max == 300 and cfg["wait"].initial == 3
        # an explicit wait is left alone
        mine = wait_exponential_jitter(initial=1, max=7)
        cfg = im.model_retry_config("m", 14, 7200, lambda e: True,
                                    lambda e: None, lambda n, rs: None,
                                    wait=mine)
        assert cfg["wait"] is mine
        # idempotent: re-installing wraps the ORIGINAL, never the wrapper
        pr.install_retry_sleep_cap(120)
        assert ir.model_retry_config.__petri_wrapped__ is original
        assert im.model_retry_config("m", 1, None, lambda e: True,
                                     lambda e: None,
                                     lambda n, rs: None)["wait"].max == 120
        # the env is honoured when no cap is passed
        monkeypatch.setenv(pr.PETRI_RETRY_SLEEP_CAP_ENV, "600")
        assert pr.install_retry_sleep_cap() == 600
    finally:
        ir.model_retry_config = original
        im.model_retry_config = original
    # inspect's own default is what the constant says it is
    cfg = original("m", None, None, lambda e: True, lambda e: None,
                   lambda n, rs: None)
    assert cfg["wait"].max == pr.INSPECT_RETRY_SLEEP_CAP_S
