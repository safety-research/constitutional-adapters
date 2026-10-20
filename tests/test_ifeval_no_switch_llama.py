"""ifeval-canonical-battery (P2): Llama's ``not_available`` is a POLICY, not vocabulary.

Before this ``THINKING_NOT_AVAILABLE`` existed only as a row-vocabulary literal: the provider always wrote
``chat_template_kwargs.enable_thinking`` from the boolean arm, the driver hashed ``thinking_by_task`` as ``off`` and
every Llama row was stamped ``off`` — an inert switch on the wire, recorded as if the template had one. Now (the Llama
lane's PR-A design, under the frozen matrix's ``--thinking off`` line — decisions X6):

* ``contracts.effective_arm_for_family`` / ``apply_no_switch_policy`` derive the per-task ARM from the mode and the served
  FAMILY: a family in ``NO_THINKING_CHANNEL_FAMILIES`` (llama) runs its OFF arm as ``not_available`` (its ON arm refuses:
  no channel); every other family is untouched. ``sends_thinking_kwargs`` / ``chat_template_kwargs`` (``{}``) /
  ``enable_thinking`` (False) / ``thinking_on_declaration`` (``{}``) / ``expected_render_regime`` (``unspecified``) read the
  no-switch arm; ``THINKING_MODES`` (the driver MODES) and ``RENDER_REGIME_BY_ARM`` (the switch arms' table) are unchanged;
* the provider sends NO ``chat_template_kwargs`` under it and refuses a caller's own switch; the runner stamps the arm on
  every row, the summary and the outcome header;
* the request-evidence gate requires the ABSENCE of the switch on every task call
  (``request_evidence_no_switch_absent``; ``request_evidence_reconciliation(expected_enable_thinking=None)``);
* the launcher's expectation (``thinking_by_task_of(..., family=)`` in ``expected_identity``) hashes the same map, on BOTH
  Llama views, so a Llama store resumes and a qwen / gemma store is untouched.
The driver-level end-to-end run (the real provider on a loopback proxy under a llama8b world) is asserted in
``test_ifeval_draw_subset_producer.py``'s regression.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, List

import pytest

from sisteer.capeval.tiers import inspect_served as isv
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C

import ifeval_synthetic_store as G
from test_inspect_served import _Emitter, _Log, _Store, _fake_eval
from test_inspect_served_ifeval import _IfSample, _VALUE
from test_tinycap_launcher_port import L, _plan  # noqa: F401  (the launcher under test)

IFEVAL = C.IFEVAL_TASK
NA = C.THINKING_NOT_AVAILABLE
LLAMA_MAIN, LLAMA_IT = C.IFEVAL_SLUGS[("llama8b", "main")], C.IFEVAL_SLUGS[("llama8b", "it")]
QWEN_MAIN = C.IFEVAL_SLUGS[("qwen35_4b", "main")]


# --------------------------------------------------------------------------- #
# 1. the contracts: the policy and what reads the no-switch arm                 #
# --------------------------------------------------------------------------- #
def test_the_no_switch_arm_is_derived_from_the_off_mode_and_the_family_and_the_mode_vocabulary_is_unchanged():
    assert C.THINKING_SWITCH_ARMS == ("on", "off") and C.THINKING_ARMS == ("on", "off", NA) and NA == "not_available"
    # the no-switch arm is a full driver arm; the regime table carries its row
    assert C.THINKING_MODES == C.THINKING_ARMS and set(C.RENDER_REGIME_BY_ARM) == {"on", "off", NA}
    assert C.NO_THINKING_CHANNEL_FAMILIES == ("llama",) and C.family_of_model_key("llama8b") == "llama"
    assert C.effective_arm_for_family("off", "llama") == NA
    assert C.effective_arm_for_family("off", "qwen35") == "off" and C.effective_arm_for_family("off", None) == "off"
    assert C.effective_arm_for_family("on", "gemma4") == "on" and C.effective_arm_for_family(NA, "llama") == NA
    with pytest.raises(C.NoThinkingChannel, match="has no thinking channel"):
        C.effective_arm_for_family("on", "llama")
    with pytest.raises(ValueError, match="thinking arm must be one of"):
        C.effective_arm_for_family(C.THINKING_PER_TASK, "llama")
    assert C.apply_no_switch_policy({IFEVAL: "off", "inspect/math": "on"}, "qwen35") == {IFEVAL: "off", "inspect/math": "on"}
    assert C.apply_no_switch_policy({IFEVAL: "off"}, "llama") == {IFEVAL: NA}
    assert C.run_thinking_by_task([C.CONTROL_IFEVAL], "off", family="llama") == {IFEVAL: NA}
    assert C.run_thinking_by_task([C.CONTROL_IFEVAL], "off") == {IFEVAL: "off"} == C.run_thinking_by_task([C.CONTROL_IFEVAL], "off", family="gemma4")
    # what reads the arm
    assert C.enable_thinking(NA) is False and C.sends_thinking_kwargs(NA) is False
    assert C.sends_thinking_kwargs("off") is True and C.sends_thinking_kwargs("on") is True
    assert C.chat_template_kwargs(NA) == {} and C.chat_template_kwargs("off") == {"chat_template_kwargs": {"enable_thinking": False}}
    assert C.check_thinking_arm(NA) == NA
    with pytest.raises(ValueError):
        C.check_thinking_arm(C.THINKING_PER_TASK)
    assert C.check_thinking_arg(NA) == NA                                  # a MODE label: the `--thinking not_available` spelling
    assert C.check_thinking_arg_for_family(NA, "llama") == NA
    with pytest.raises(ValueError, match="has a thinking switch"):
        C.check_thinking_arg_for_family(NA, "qwen3_5")                    # a mislabel on a switched family
    assert C.thinking_on_declaration(lens=True, arm=NA, header_located=True) == {}
    assert C.expected_render_regime(NA) == C.RENDER_REGIME_UNSPECIFIED and C.expected_render_regime("off") == C.RENDER_REGIME_THINKING_OFF
    assert C.RENDER_REGIME_BY_ARM[NA] == C.RENDER_REGIME_UNSPECIFIED       # the table's third row
    base = C.expected_stamp_regimes(lens=False, arm=NA)
    assert base == C.StampRegimes(C.SALT_REGIME_CELL_STABLE, C.POLICY_NOT_CONSULTED, C.RENDER_REGIME_UNSPECIFIED)
    lens = C.expected_stamp_regimes(lens=True, arm=NA, policy=C.RESUMED_PREFILL_CONDITIONAL)
    off = C.expected_stamp_regimes(lens=True, arm="off", policy=C.RESUMED_PREFILL_CONDITIONAL)
    assert (lens.salt_regime, lens.resumed_policy) == (off.salt_regime, off.resumed_policy)      # an OFF-row leg ...
    assert lens.render_regime == C.RENDER_REGIME_UNSPECIFIED != off.render_regime                 # ... rendered unspecified
    assert C.authorised_stamp_policy(lens=True, arm=NA) == C.authorised_stamp_policy(lens=True, arm="off")
    assert NA in C.ROW_THINKING_VALUES


def _rec(kind="task", *, present, value=None, task=IFEVAL, sample="1000/1", declared=False):
    return {"request_evidence_version": C.REQUEST_EVIDENCE_VERSION, "kind": kind, "task": task, "sample": sample,
            "petri_thinking_on_present": declared, "enable_thinking_present": present, "enable_thinking_value": value}


def test_the_evidence_gate_requires_the_absence_of_the_switch_under_the_no_switch_arm():
    ok = [_rec(present=False), _rec(present=False, sample="1001/0")]
    assert C.request_evidence_no_switch_absent(ok) is None
    why = C.request_evidence_no_switch_absent([_rec(present=False), _rec(present=True, value=False, sample="1001/0")])
    assert why and "1 of 2 task request(s) CARRIED chat_template_kwargs.enable_thinking" in why and "'1001/0'" in why
    assert "the no-switch arm 'not_available' sends none" in why
    assert C.request_evidence_no_switch_absent([]) and "never counts as demonstrated absence of the switch" in C.request_evidence_no_switch_absent([])
    assert "CARRIED petri_thinking_on" in C.request_evidence_no_switch_absent([_rec(present=False, declared=True)])
    # a driver-side PROBE may carry the switch (as under OFF); only TASK calls are held to its absence
    assert C.request_evidence_no_switch_absent([_rec(present=False), _rec("probe", present=True, value=False, task="probe/render", sample="v/1")]) is None
    # the reconciliation dispatches: None = the no-switch reading; False = the OFF value on every task call
    assert C.request_evidence_reconciliation([_rec(present=True, value=False)], [], expected_enable_thinking=None) is not None
    assert "CARRIED" in C.request_evidence_reconciliation([_rec(present=True, value=False)], [], expected_enable_thinking=None)
    assert "did not render thinking-OFF" in C.request_evidence_reconciliation([_rec(present=False)], [], expected_enable_thinking=False)
    # ... and past that step the ledger is required, as for every arm
    assert "the usage ledger holds no rows" in C.request_evidence_reconciliation(ok, [], expected_enable_thinking=None)


# --------------------------------------------------------------------------- #
# 2. the provider and the runner                                               #
# --------------------------------------------------------------------------- #
def _provider(**kw):
    from inspect_ai.model import GenerateConfig
    cls = isv._register_provider()
    route = {"cell_id": "none", "virtual_model": "petri__none", "serving_identity": "proxy_vllm+passthrough+interleaved",
             "intervention_hash": "0" * 16}
    api = cls(model_name=C.inspect_model_name(route["virtual_model"]), base_url="http://127.0.0.1:1/v1",
              api_key="synthetic", config=GenerateConfig(), si=None, route=route, thinking=C.THINKING_OFF, **kw)
    return api, GenerateConfig


def test_the_provider_sends_no_switch_under_the_no_switch_arm_and_refuses_a_callers_own():
    api, GenerateConfig = _provider(thinking_by_task={IFEVAL: NA})
    tok = isv._TASK.set(IFEVAL)
    try:
        assert api.request_thinking() == NA
        cfg = api.tinycap_config(GenerateConfig())
        assert C.CHAT_TEMPLATE_KWARGS_KEY not in (cfg.extra_body or {}) and C.OOS_THINKING_DECLARATION_KEY not in (cfg.extra_body or {})
        # a caller's other template kwargs ride; its switch does not
        cfg2 = api.tinycap_config(GenerateConfig(extra_body={C.CHAT_TEMPLATE_KWARGS_KEY: {"custom": 1}}))
        assert cfg2.extra_body[C.CHAT_TEMPLATE_KWARGS_KEY] == {"custom": 1}
        with pytest.raises(isv.InspectServedError, match="has no thinking switch"):
            api.tinycap_config(GenerateConfig(extra_body={C.CHAT_TEMPLATE_KWARGS_KEY: {C.ENABLE_THINKING_KEY: False}}))
        assert api.expected_regimes(NA).render_regime == C.RENDER_REGIME_UNSPECIFIED
    finally:
        isv._TASK.reset(tok)
    # the OFF arm is byte-identical: the switch rides as before
    api_off, _ = _provider(thinking_by_task={IFEVAL: C.THINKING_OFF})
    tok = isv._TASK.set(IFEVAL)
    try:
        assert api_off.tinycap_config(GenerateConfig()).extra_body[C.CHAT_TEMPLATE_KWARGS_KEY] == {C.ENABLE_THINKING_KEY: False}
    finally:
        isv._TASK.reset(tok)
    # the map's values are arms: the mode token still refuses, with the pinned phrase
    with pytest.raises(isv.InspectServedError, match="the map's VALUES are arms"):
        _provider(thinking_by_task={IFEVAL: C.THINKING_PER_TASK})


def test_the_runner_stamps_not_available_on_every_row_the_summary_and_the_outcome_header(tmp_path, monkeypatch):
    seen: List[Any] = []
    logs = {0: _Log([_IfSample("1000", _VALUE), _IfSample("1001", _VALUE)]),
            1: _Log([_IfSample("1000", _VALUE), _IfSample("1001", _VALUE)])}
    monkeypatch.setattr("inspect_ai.eval_async", _fake_eval(logs, seen))
    em, store = _Emitter(), _Store(tmp_path / "cell")
    plan = isv.TaskPlan(row_task=IFEVAL, name=C.IFEVAL_TASK_NAME, task=object(), expected_n=2, samples_per_item=2,
                        primary_scorer="instruction_following", max_tokens=2048, thinking=NA,
                        item_set="ifeval541", sample_ids=["1000", "1001"], note="llama pass 1")
    summary = asyncio.run(isv.run_one_task(plan, model=None, store=store, template={}, concurrency=4, emitter=em,
                                           sampling_ref="abc123def456"))
    rows = [r for r in em.rows if r["item_id"] != C.AGG_ITEM_ID]
    assert rows and all(r["extra"]["thinking"] == NA for r in rows)
    assert all(r["extra"]["thinking"] in C.ROW_THINKING_VALUES for r in rows)
    agg = em.by(C.METRIC_ACC_AGGREGATE)[0]
    assert agg["extra"]["thinking"] is False                              # the aggregate row's boolean: it did not think
    assert summary["thinking"] == NA and summary["render_regime"] is None      # no provider accumulator here: null
    outcomes = completion.read_task_outcomes(store.cell_dir, IFEVAL)
    assert outcomes["thinking"] == NA and outcomes["n_attempted_samples"] == 4
    # the cap rule takes the arm with OFF semantics (the table's 2,048; the OFF setter admits it)
    assert isv.task_max_tokens(C.IFEVAL_TASK_NAME, thinking=NA) == isv.task_max_tokens(C.IFEVAL_TASK_NAME, thinking="off")
    assert isv.task_max_tokens(C.IFEVAL_TASK_NAME, thinking=NA, off_cap_setter={C.IFEVAL_TASK_NAME: 4096}) == (4096, C.CAP_SOURCE_OFF_CAP_EXPLICIT)
    with pytest.raises(isv.InspectServedError, match="run_one_task was given the 'per-task' MODE"):
        asyncio.run(isv.run_one_task(plan, model=None, store=_Store(tmp_path / "x"), template={}, thinking=C.THINKING_PER_TASK))


# --------------------------------------------------------------------------- #
# 3. the launcher's expectation: both Llama views, qwen untouched               #
# --------------------------------------------------------------------------- #
def test_the_launchers_expectation_hashes_not_available_on_both_llama_views_and_off_elsewhere(L):
    for slug in (LLAMA_MAIN, LLAMA_IT):
        (cfg,) = _plan(L, slug)
        assert L.thinking_by_task_of(cfg["mode"], cfg["controls"], family="llama") == {IFEVAL: NA}
        exp = L.expected_identity(cfg, L.configuration_geometry(cfg, 1), pass_no=1, controls=cfg["controls"], samples=2,
                                  reduction_mode=C.REDUCTION_MODE_STACKED, sampling_mode=C.SAMPLING_VENDOR_PRESET)
        # the launcher expects the HASHED mode label the driver writes — the resolved arm
        assert exp.thinking_by_task == {IFEVAL: NA} and exp.mode == NA and cfg["mode"] == C.THINKING_OFF
        assert exp.as_dict()[C.IDENTITY_KEY_THINKING_BY_TASK] == {IFEVAL: NA}
        assert L.thinking_by_task_of_cfg(cfg) == {IFEVAL: NA}
        # the family-resolved map equals the driver's derivation (contracts.run_thinking_by_task with the family)
        assert exp.thinking_by_task == C.run_thinking_by_task(cfg["controls"], cfg["mode"], family=C.model_row(slug).family)
    (q,) = _plan(L, QWEN_MAIN)
    qexp = L.expected_identity(q, L.configuration_geometry(q, 1), pass_no=1, controls=q["controls"], samples=2,
                               reduction_mode=C.REDUCTION_MODE_STACKED, sampling_mode=C.SAMPLING_VENDOR_PRESET)
    assert qexp.thinking_by_task == {IFEVAL: C.THINKING_OFF} and L.thinking_by_task_of_cfg(q) == {IFEVAL: C.THINKING_OFF}
    # the no-family reading is the legacy one (every pre-existing caller)
    assert L.thinking_by_task_of(C.THINKING_OFF, [C.CONTROL_IFEVAL]) == {IFEVAL: C.THINKING_OFF}
    # the frozen matrix still says `off` on every llama row: that is the MODE (the driver line), and the family policy
    # is what turns it into the recorded arm — the documentary field is not the hashed one
    rows = [r for r in G.matrix_rows() if r["model_key"] == "llama8b"]
    assert rows and all(r["thinking"] == C.THINKING_OFF for r in rows)
