"""The canonical IFEval battery — the OFF-arm cap SETTER ``--off-cap-by-task``
and the thinking-OFF server pin.

The ON setter refuses any OFF task by design; IFEval runs OFF, so its 4,096 /
8,192 rungs come through a flag of their own, bounded by
``OFF_CAP_BY_TASK_ALLOWED`` and ``IFEVAL_CAP_CEILING``, consulted by
``task_max_tokens`` in the OFF branch after the authorised comparator and before
the OFF table, recorded unhashed while the executed cap hashes as every cap does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))     # served_capability_run lives in scripts/ (as every driver test inserts)

from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tiers import inspect_served as isv
from sisteer.capeval.serving import vllm_server as vs

IFEVAL = C.IFEVAL_TASK_NAME


def test_the_off_setter_shares_the_statements_grammar_and_names_its_own_flag():
    assert C.FLAG_OFF_CAP_BY_TASK == "--off-cap-by-task" and C.CAP_SOURCE_OFF_CAP_EXPLICIT == "off_cap_explicit"
    assert C.MANIFEST_KEY_OFF_CAP_BY_TASK == "off_cap_by_task"
    assert C.OFF_CAP_BY_TASK_ALLOWED == {IFEVAL: (4096, 8192)}
    assert C.parse_off_cap_by_task(["ifeval=4096"]) == {IFEVAL: 4096}
    assert C.parse_off_cap_by_task(["ifeval=8192"]) == {IFEVAL: 8192}
    assert C.parse_off_cap_by_task([]) == {}
    for bad in (["ifeval"], ["ifeval=x"], ["ifeval=0"], ["ifeval=4096", "ifeval=8192"]):
        with pytest.raises(ValueError, match=r"--off-cap-by-task"):
            C.parse_off_cap_by_task(bad)
    assert issubclass(C.OffCapSetterRefused, ValueError)


def test_the_off_setter_is_bounded_by_the_table_the_arm_the_resolution_and_the_ceiling():
    # admitted: the two rungs, under OFF, above the 2,048 table resolution
    for cap in (4096, 8192):
        C.check_off_cap_setter(IFEVAL, cap, arm=C.THINKING_OFF, resolved_cap=2048, cap_source="off_table")
    # refused, each by name
    with pytest.raises(C.OffCapSetterRefused, match="not a task the plan lets the OFF setter raise"):
        C.check_off_cap_setter("gsm8k_paired200", 4096, arm=C.THINKING_OFF, resolved_cap=512, cap_source="off_table")
    with pytest.raises(C.OffCapSetterRefused, match="not one the plan allows"):
        C.check_off_cap_setter(IFEVAL, 2048, arm=C.THINKING_OFF, resolved_cap=2048, cap_source="off_table")
    with pytest.raises(C.OffCapSetterRefused, match="not one the plan allows"):
        C.check_off_cap_setter(IFEVAL, 16384, arm=C.THINKING_OFF, resolved_cap=2048, cap_source="off_table")
    with pytest.raises(C.OffCapSetterRefused, match="runs under the 'on' arm — the setter is a thinking-OFF cap"):
        C.check_off_cap_setter(IFEVAL, 4096, arm=C.THINKING_ON, resolved_cap=2048, cap_source="on_floor")
    with pytest.raises(C.OffCapSetterRefused, match="pins its own cap"):
        C.check_off_cap_setter(IFEVAL, 4096, arm=C.THINKING_OFF, resolved_cap=None, cap_source="solver")
    with pytest.raises(C.OffCapSetterRefused, match="not ABOVE the off_table resolution 4096"):
        C.check_off_cap_setter(IFEVAL, 4096, arm=C.THINKING_OFF, resolved_cap=4096, cap_source="off_table")
    assert C.IFEVAL_CAP_CEILING == 8192 == max(C.OFF_CAP_BY_TASK_ALLOWED[IFEVAL])
    # the ON setter still refuses the OFF task, and never learned the IFEval task
    with pytest.raises(C.OnCapSetterRefused, match="not a task the plan lets the setter raise"):
        C.check_on_cap_setter(IFEVAL, 4096, arm=C.THINKING_ON, resolved_cap=2048, cap_source="on_floor")


def test_task_max_tokens_consults_the_setter_after_the_comparator_and_before_the_table():
    assert isv.task_max_tokens(IFEVAL, thinking=C.THINKING_OFF) == (2048, "off_table")
    assert isv.task_max_tokens(IFEVAL, thinking=C.THINKING_OFF, off_cap_setter={IFEVAL: 4096}) == (
        4096, C.CAP_SOURCE_OFF_CAP_EXPLICIT)
    assert isv.task_max_tokens(IFEVAL, thinking=C.THINKING_OFF, off_cap_setter={IFEVAL: 8192}) == (
        8192, C.CAP_SOURCE_OFF_CAP_EXPLICIT)
    # a map not naming the task: byte-identical
    assert isv.task_max_tokens(IFEVAL, thinking=C.THINKING_OFF, off_cap_setter={"gsm8k_paired200": 4096}) == (
        2048, "off_table")
    assert isv.task_max_tokens("gsm8k_paired200", thinking=C.THINKING_OFF, off_cap_setter={IFEVAL: 4096}) == (
        512, "off_table")
    # refusals surface as the adapter's error, naming the flag
    with pytest.raises(isv.InspectServedError, match="--off-cap-by-task.*not one the plan allows"):
        isv.task_max_tokens(IFEVAL, thinking=C.THINKING_OFF, off_cap_setter={IFEVAL: 2048})
    with pytest.raises(isv.InspectServedError, match="--off-cap-by-task.*not one the plan allows"):
        isv.task_max_tokens(IFEVAL, thinking=C.THINKING_OFF, off_cap_setter={IFEVAL: 16384})
    with pytest.raises(isv.InspectServedError, match="thinking-OFF cap"):
        isv.task_max_tokens(IFEVAL, thinking=C.THINKING_ON, off_cap_setter={IFEVAL: 4096})
    with pytest.raises(isv.InspectServedError, match="not a task the plan lets the OFF setter raise"):
        isv.task_max_tokens("gsm8k_paired200", thinking=C.THINKING_OFF, off_cap_setter={"gsm8k_paired200": 4096})
    # the ON setter naming the OFF task still refuses by ITS name
    with pytest.raises(isv.InspectServedError, match="--on-cap-by-task"):
        isv.task_max_tokens(IFEVAL, thinking=C.THINKING_OFF, on_cap_override={IFEVAL: 4096})


def test_the_two_off_overrides_are_mutually_exclusive_by_name(monkeypatch):
    """The authorised comparator's raised OFF cap (`off_cap_override`, a bool) and
    the rung setter never combine on one task."""
    monkeypatch.setattr(C, "off_cap_override_max_tokens", lambda name: 8192 if name == IFEVAL else None)
    assert isv.task_max_tokens(IFEVAL, thinking=C.THINKING_OFF, off_cap_override=True) == (8192, "offcap_override")
    with pytest.raises(isv.InspectServedError, match="mutually exclusive by name"):
        isv.task_max_tokens(IFEVAL, thinking=C.THINKING_OFF, off_cap_override=True, off_cap_setter={IFEVAL: 4096})


def test_the_plan_carries_the_rung_cap_and_its_source(monkeypatch):
    """`plan_task(off_cap_setter=)` plans the rung cap with `cap_source`
    off_cap_explicit — the same rule the driver's hashed `max_tokens_by_task`
    derivation applies, so the identity's cap and the executed cap agree."""
    from test_inspect_served_ifeval import _ifeval_cached
    if not _ifeval_cached():
        pytest.skip("google/IFEval is not in the local HF cache")
    plan = isv.plan_task(IFEVAL, grader_model=None, thinking=C.THINKING_OFF, off_cap_setter={IFEVAL: 4096})
    assert plan.max_tokens == 4096 and plan.cap_source == C.CAP_SOURCE_OFF_CAP_EXPLICIT
    plans = isv.plans_for_atoms([C.CONTROL_IFEVAL], grader_model=None, thinking=C.THINKING_OFF,
                                off_cap_setter={IFEVAL: 8192})
    assert [p.max_tokens for p in plans] == [8192]
    with pytest.raises(isv.InspectServedError, match="thinking-OFF cap"):
        isv.plan_task(IFEVAL, grader_model=None, thinking=C.THINKING_ON, off_cap_setter={IFEVAL: 4096},
                      family="qwen3_5", reasoning_parser="qwen3", require_certified_fact=False)


def test_the_driver_takes_the_flag_under_identity_v2_only_and_records_it_unhashed():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts" / "served_capability_run.py").read_text()
    assert 'g.add_argument(tc.FLAG_OFF_CAP_BY_TASK, dest="off_cap_by_task", action="append"' in src
    assert "off_cap_stmt = tc.parse_off_cap_by_task(" in src
    assert "(tc.FLAG_OFF_CAP_BY_TASK, bool(off_cap_stmt))" in src              # the identity-v2 gate
    assert "off_cap_setter=(dict(off_cap_stmt) if off_cap_stmt else None)" in src
    assert "manifest[tc.MANIFEST_KEY_OFF_CAP_BY_TASK] = dict(off_cap_stmt)" in src
    # the item-subset binding names the OFF setter for an IFEval rung, the ON setter for GSM8K
    assert "flag = tc.FLAG_OFF_CAP_BY_TASK if name in tc.OFF_CAP_BY_TASK_ALLOWED else tc.FLAG_ON_CAP_BY_TASK" in src


def test_the_item_subset_binding_names_the_off_setter_for_an_ifeval_rung(tmp_path):
    import served_capability_run as srun
    doc = {"task": C.inspect_task(IFEVAL), "cap": 4096, "source_out_root": str(tmp_path / "src"),
           "source_identity_hash": "0" * 16, "source_cap": 2048}
    why = srun._item_subset_problem(doc, executed_caps={C.inspect_task(IFEVAL): 2048}, out_root=tmp_path / "rung")
    assert "state the rung cap with --off-cap-by-task ifeval=4096" in why
    doc_g = {**doc, "task": C.inspect_task("gsm8k_paired200"), "cap": 16384, "source_cap": 8192}
    why = srun._item_subset_problem(doc_g, executed_caps={C.inspect_task("gsm8k_paired200"): 8192},
                                    out_root=tmp_path / "rung")
    assert "state the rung cap with --on-cap-by-task gsm8k_paired200=16384" in why


# --------------------------------------------------------------------------- #
# thinking OFF: the composed server pin + the per-request kwarg                  #
# --------------------------------------------------------------------------- #


def test_the_thinking_off_server_pin_is_the_static_waves_and_composes_only_where_the_template_switches():
    """`build_server_launch` composes `--default-chat-template-kwargs
    THINKING_OFF_TEMPLATE_KWARGS` whenever the pinned template's bytes carry
    `enable_thinking` (Qwen3.5, Nemotron) and nothing for gemma / Llama — BY
    CONSTRUCTION the static full22 wave's pin (the same composer, wrapper-owned);
    every IFEval request also carries chat_template_kwargs.enable_thinking false."""
    import inspect
    assert vs.THINKING_OFF_TEMPLATE_KWARGS == '{"enable_thinking": false}'
    src = inspect.getsource(vs.build_server_launch)
    assert 'if b"enable_thinking" in template_bytes:' in src
    # main's thinking-ON standup (the sampled-rerun lineage, merged 2026-09-12) made the pin a CHOICE: the OFF
    # literal unless the standup asks for ON (`thinking_on=True`); every IFEval standup leaves the default, so the
    # composed pin is THINKING_OFF_TEMPLATE_KWARGS exactly as before
    assert "default_chat_template_kwargs = (THINKING_ON_TEMPLATE_KWARGS if thinking_on" in src
    assert "else THINKING_OFF_TEMPLATE_KWARGS)" in src
    assert inspect.signature(vs.build_server_launch).parameters["thinking_on"].default is False
    assert '"--default-chat-template-kwargs", default_chat_template_kwargs' in src
    assert "--default-chat-template-kwargs" in vs._WRAPPER_OWNED_OPTIONS
    # the per-request kwarg on the OFF arm, the fragment every IFEval request carries
    assert C.chat_template_kwargs(C.THINKING_OFF) == {"chat_template_kwargs": {"enable_thinking": False}}
    assert C.THINKING_BY_TASK[C.inspect_task(IFEVAL)] == C.THINKING_OFF
    # the OFF plan needs no reasoning parser and no render fact; the split mode is record-only
    plan_off = isv.task_max_tokens(IFEVAL, thinking=C.THINKING_OFF)
    assert plan_off == (2048, "off_table")
    assert C.think_split_mode(arm=C.THINKING_OFF, reasoning_parser=None, prompt_opened_think=False) == \
        C.THINK_SPLIT_RECORD_ONLY


def test_header_located_safety_cells_are_served_undeclared_under_off():
    """P15: the GSM8K leg declared the two header-located safety cells `@thinking`
    because it ran ON; IFEval is OFF = the fit render, so those cells are served
    at `response` positions with NO declaration — `cell_spellings(declared=False)`
    keeps the bare cell as the standup cell, and the expected stamp regimes are
    the OFF row's."""
    for cell in ("safety_refusal__psr_kl_all", "safety_refusal__lora_r4"):
        sp = C.cell_spellings(cell, declared=False)
        # undeclared: every door's spelling is the bare cell — no `@thinking` variant anywhere
        assert sp.traintest == cell and "@thinking" not in sp.harness and "@thinking" not in sp.vmodel
        declared = C.cell_spellings(cell, declared=True)
        assert declared.traintest != sp.traintest and "thinking" in declared.traintest   # the GSM8K ON spelling
        assert C.route_kind_of_cell(cell) == C.ROUTE_KIND_LENS
    regimes = C.expected_stamp_regimes(lens=True, arm=C.THINKING_OFF, policy=C.RESUMED_PREFILL_CONDITIONAL)
    assert isinstance(regimes, C.StampRegimes) and C.RENDER_REGIME_BY_ARM[C.THINKING_OFF] in tuple(regimes)
    assert C.RENDER_REGIME_BY_ARM[C.THINKING_ON] not in tuple(regimes)
    with pytest.raises(ValueError):
        C.expected_stamp_regimes(lens=True, arm=C.THINKING_PER_TASK)
