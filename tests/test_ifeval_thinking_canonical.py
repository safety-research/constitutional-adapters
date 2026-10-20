"""ifeval-canonical-battery: the no-switch arm is CANONICALISED to the `not_available` spelling.

One spelling accepted and hashed the Llama mode `not_available` via `--thinking not_available`; the other rejected
that CLI value and hashed `off` while resolving the task arm to `not_available` — two
`tinycap_identity` hashes for otherwise identical llama inputs, and two such cell directories under one out root are
refused by the analysis. Now the tables and gate are the contracts' (`THINKING_MODES` / `LAUNCHER_THINKING_MODES` with
the no-switch arm LAST, `check_thinking_arg_for_family`, `RENDER_REGIME_BY_ARM[not_available] = unspecified`,
`CERT_CLASS_*` + `cert_class_for`, `RenderFact.thinking_switch`), `effective_arm_for_family` stays the ONE derivation
(an explicit `off` on a no-switch family resolves to `not_available`; `not_available` on a switched family refuses by
name), and BOTH the driver's hashed mode label and the launcher's expected identity read the resolved arm — so the frozen
matrix line and the explicit arm hash one identity (the loopback regression resumes a store across the two spellings).
"""
from __future__ import annotations

import pytest

from sisteer.capeval.tinycap import contracts as C

from test_tinycap_launcher_port import GEMMA, L, LLAMA, QWEN, _plan  # noqa: F401

NA = C.THINKING_NOT_AVAILABLE
LLAMA_MAIN = C.IFEVAL_SLUGS[("llama8b", "main")]


def test_the_tables_are_pr_as():
    assert NA == "not_available"
    assert C.THINKING_MODES == ("on", "off", NA) == C.THINKING_ARMS == C.ROW_THINKING_VALUES
    assert C.THINKING_SWITCH_ARMS == ("on", "off")
    assert C.LAUNCHER_THINKING_MODES == ("off", "on", "per-task", NA)          # the no-switch arm LAST
    assert C.RENDER_REGIME_BY_ARM == {"on": "thinking_on", "off": "thinking_off", NA: "unspecified"}
    assert C.expected_render_regime(NA) == C.RENDER_REGIME_UNSPECIFIED == C.RENDER_REGIME_BY_ARM[NA]
    assert C.enable_thinking(NA) is False and C.sends_thinking_kwargs(NA) is False and C.chat_template_kwargs(NA) == {}
    assert C.check_thinking_arg(NA) == NA and C.check_thinking_arm(NA) == NA
    assert (C.CERT_CLASS_SWITCH, C.CERT_CLASS_NO_SWITCH) == ("switch", "no_switch") and "CERT_CLASS_NO_SWITCH" in C.NO_SWITCH_LENS_ON_BITE_REASON
    with pytest.raises(ValueError):
        C.enable_thinking(C.THINKING_PER_TASK)                                # per-task is a MODE, not an arm — unchanged


def test_the_family_gate_and_the_one_derivation():
    assert C.check_thinking_arg_for_family(NA, "llama") == NA
    for arm in ("on", "off"):
        assert C.check_thinking_arg_for_family(arm, "qwen3_5") == arm and C.check_thinking_arg_for_family(arm, "llama") == arm
    with pytest.raises(ValueError, match="has a thinking switch"):
        C.check_thinking_arg_for_family(NA, "qwen3_5")
    with pytest.raises(ValueError, match="has a thinking switch"):
        C.check_thinking_arg_for_family(NA, None)
    # the ONE derivation: `off` and `not_available` on a no-switch family are the same arm; `on` refuses; a switched
    # family keeps its arms and refuses the mislabel; an unknown family keeps the arm
    assert C.effective_arm_for_family("off", "llama") == NA == C.effective_arm_for_family(NA, "llama")
    with pytest.raises(C.NoThinkingChannel):
        C.effective_arm_for_family("on", "llama")
    assert C.effective_arm_for_family("off", "qwen3_5") == "off" and C.effective_arm_for_family("on", "gemma4") == "on"
    with pytest.raises(ValueError, match="has a thinking switch"):
        C.effective_arm_for_family(NA, "qwen3_5")
    assert C.effective_arm_for_family(NA, None) == NA and C.effective_arm_for_family("off", None) == "off"
    assert C.apply_no_switch_policy({"inspect/ifeval": "off"}, "llama") == {"inspect/ifeval": NA}
    assert C.run_thinking_by_task([C.CONTROL_IFEVAL], NA) == {C.IFEVAL_TASK: NA} == C.run_thinking_by_task([C.CONTROL_IFEVAL], "off", family="llama")


def test_cert_class_for_and_the_render_fact_switch_field():
    fact = C.render_fact_for("gemma4")
    assert fact is not None and fact.thinking_switch is True and fact.as_dict()["thinking_switch"] is True
    assert C.cert_class_for("on", fact) == C.CERT_CLASS_SWITCH == C.cert_class_for("off", fact)
    with pytest.raises(ValueError, match="carries a thinking switch"):
        C.cert_class_for(NA, fact)
    with pytest.raises(ValueError, match="is absent"):
        C.cert_class_for(NA, None)
    no_switch = fact._replace(family="llama", thinking_switch=False)
    assert C.cert_class_for(NA, no_switch) == C.CERT_CLASS_NO_SWITCH
    with pytest.raises(ValueError, match="no thinking-ON class"):
        C.cert_class_for("off", no_switch)
    with pytest.raises(ValueError, match="arm must be one of"):
        C.cert_class_for(C.THINKING_PER_TASK, fact)


def test_the_launcher_hashes_the_resolved_mode_for_llama_and_admits_pr_as_explicit_arm(L, tmp_path):
    # the frozen matrix line (`--thinking off`) and PR-A's explicit arm plan to the SAME expected identity on llama
    (off_cfg,) = _plan(L, LLAMA_MAIN, cell="none", out_dir=str(tmp_path / "a"))
    (na_cfg,) = L.plan_configurations(cells=["none"], modes=[NA], controls=[C.CONTROL_IFEVAL], model=LLAMA_MAIN,
                                      sampling_mode=C.SAMPLING_VENDOR_PRESET, samples=C.IFEVAL_SAMPLES_PER_ITEM,
                                      reduction_mode=C.REDUCTION_MODE_STACKED, pass_no=1, out_dir=str(tmp_path / "a"))
    assert off_cfg["mode"] == "off" and na_cfg["mode"] == NA                     # the REQUESTED modes differ ...
    assert off_cfg["thinking_by_task"] == na_cfg["thinking_by_task"] == {C.IFEVAL_TASK: NA}
    geom = L.configuration_geometry(off_cfg, 1)
    e_off = L.expected_identity(off_cfg, geom, pass_no=1, controls=[C.CONTROL_IFEVAL], samples=2,
                                reduction_mode=C.REDUCTION_MODE_STACKED, sampling_mode=C.SAMPLING_VENDOR_PRESET)
    e_na = L.expected_identity(na_cfg, L.configuration_geometry(na_cfg, 1), pass_no=1, controls=[C.CONTROL_IFEVAL], samples=2,
                               reduction_mode=C.REDUCTION_MODE_STACKED, sampling_mode=C.SAMPLING_VENDOR_PRESET)
    assert e_off.mode == e_na.mode == NA                                          # ... the HASHED mode label does not
    assert e_off.thinking_by_task == e_na.thinking_by_task == {C.IFEVAL_TASK: NA}
    assert e_off.as_dict() == e_na.as_dict()
    # a switched family keeps `off` and refuses the mislabel at plan time
    (q_cfg,) = _plan(L, QWEN, cell="none", out_dir=str(tmp_path / "q"))
    e_q = L.expected_identity(q_cfg, L.configuration_geometry(q_cfg, 1), pass_no=1, controls=[C.CONTROL_IFEVAL], samples=2,
                              reduction_mode=C.REDUCTION_MODE_STACKED, sampling_mode=C.SAMPLING_VENDOR_PRESET)
    assert e_q.mode == "off" and e_q.thinking_by_task == {C.IFEVAL_TASK: "off"}
    with pytest.raises(ValueError, match="has a thinking switch"):
        L.plan_configurations(cells=["none"], modes=[NA], controls=[C.CONTROL_IFEVAL], model=QWEN,
                              sampling_mode=C.SAMPLING_VENDOR_PRESET, samples=C.IFEVAL_SAMPLES_PER_ITEM,
                              reduction_mode=C.REDUCTION_MODE_STACKED, pass_no=1)
    assert NA in L.MODE_PLAN_ORDER and L.MODE_PLAN_ORDER[-1] == NA
