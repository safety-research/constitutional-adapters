"""ifeval-canonical-battery, lane 1 — the ``inspect/ifeval`` row task on the served
adapter.

The plan branch builds inspect_evals 0.18.0's STOCK ``ifeval`` task untouched
(the dataset pinned at ``google/IFEval @ 966cd895``, the stock
``instruction_following`` scorer whose value is a DICT of five numbers); the
row's ``acc`` reads ``PRIMARY_VALUE_KEY["ifeval"]`` and the five values + the
instruction ids ride as row facts on every row and outcome item. The stock task
is built for real here (from the local HF cache, offline) so the 541-id count,
the scorer name and the rendered prompts are the installed task's own; the
runner tests use synthetic logs shaped like inspect_ai's.
"""

from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tiers import inspect_served as isv
from sisteer.eval.cell_controls import _assert_counts_only

# the sibling module's synthetic-log helpers (pytest puts tests/ on sys.path)
from test_inspect_served import (  # noqa: E402
    _Emitter, _Log, _Sample, _Score, _Store, _fake_eval, _model_output, _plan)

REPO = Path(__file__).resolve().parents[1]
IFEVAL = C.inspect_task(C.IFEVAL_TASK_NAME)

# the local HF cache holds the pinned snapshot + the arrow cache; the task is
# built offline so a test never reaches the hub. Scoped to THIS module: the
# module-level `os.environ.setdefault` this replaced set offline mode for the
# whole pytest process at collection time and never restored it (datasets /
# huggingface_hub also read the flag at import and hold their own copy), so
# every later test module inherited it — a leak the merged tree's union run made
# visible; the fixture sets, and restores, both the env and the libraries' copies.
@pytest.fixture(scope="module", autouse=True)
def _offline_for_this_module():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HF_HUB_OFFLINE", "1")
        mp.setenv("HF_DATASETS_OFFLINE", "1")
        # the libraries' import-time copies of the flag, so the module is offline
        # whatever imported them first — and restored with the env afterwards
        try:
            import huggingface_hub.constants as _hc
            mp.setattr(_hc, "HF_HUB_OFFLINE", True, raising=False)
        except ImportError:  # pragma: no cover
            pass
        try:
            import datasets.config as _dc
            mp.setattr(_dc, "HF_HUB_OFFLINE", True, raising=False)
            mp.setattr(_dc, "HF_DATASETS_OFFLINE", True, raising=False)
        except ImportError:  # pragma: no cover
            pass
        yield


def _ifeval_cached() -> bool:
    home = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    return (home / "hub" / "datasets--google--IFEval").is_dir()


needs_dataset = pytest.mark.skipif(not _ifeval_cached(), reason="google/IFEval is not in the local HF cache")


@pytest.fixture(scope="module")
def real_plan():
    """The stock task planned ONCE per module (a ~10 s dataset load)."""
    if not _ifeval_cached():
        pytest.skip("google/IFEval is not in the local HF cache")
    return isv.plan_task(C.IFEVAL_TASK_NAME, grader_model=None, thinking=C.THINKING_OFF, repo_root=REPO)


# --------------------------------------------------------------------------- #
# 1. contracts: the names the plan defines (§1.1)                              #
# --------------------------------------------------------------------------- #


def test_the_ifeval_atom_and_row_task_are_in_the_grammar_under_one_atom():
    assert C.CONTROL_IFEVAL == "ifeval" and C.CONTROL_IFEVAL in C.TINYCAP_CONTROLS
    assert C.CONTROL_IFEVAL in C.TINYCAP_CONTROLS and C.TINYCAP_CONTROLS[-2:] == (C.CONTROL_BIGCODEBENCH_FULL, C.CONTROL_BFCL_FULL)   # ifeval was appended LAST at the time; the two full-population controls followed it (capeval-full-tasks)
    assert C.expand_control("ifeval") == ("ifeval",)
    assert C.TASKS_OF_CONTROL[C.CONTROL_IFEVAL] == (IFEVAL,) == ("inspect/ifeval",)
    assert sum(IFEVAL in ts for ts in C.TASKS_OF_CONTROL.values()) == 1    # exactly one atom
    assert C.DISPATCH_MODULE_OF_CONTROL[C.CONTROL_IFEVAL] == "sisteer.capeval.tiers.inspect_served"
    assert C.IFEVAL_TASK_NAME == "ifeval" and C.IFEVAL_TASK_NAME in C.INSPECT_TASK_NAMES      # the full-population siblings (bigcodebench_full, bfcl_full) were appended after it (capeval-full-tasks)
    assert C.INSPECT_TASK_NAMES[-2:] == (C.CONTROL_BIGCODEBENCH_FULL, C.CONTROL_BFCL_FULL)
    assert IFEVAL in C.INSPECT_TASKS and IFEVAL in C.TINYCAP_TASKS and IFEVAL in C.DECODING_TASKS
    assert C.TINYCAP_DOMAIN_OF[IFEVAL] == C.DOMAIN_INSPECT_GEN
    assert C.THINKING_BY_TASK[IFEVAL] == C.THINKING_OFF
    assert C.SAMPLES_BY_TASK[IFEVAL] == C.IFEVAL_SAMPLES_PER_ITEM == 2
    # P22: in no default battery list
    for battery in C.STANDARD_BATTERY_OF_KIND.values():
        assert C.CONTROL_IFEVAL not in battery
    assert C.CONTROL_IFEVAL not in C.IN_DEPTH_EXTRAS


def test_the_frozen_facts_equal_the_installed_task_and_the_pins_file():
    import yaml

    pins = yaml.safe_load((REPO / "configs" / "capeval" / "pins.yaml").read_text())
    entry = next(v for k, v in _walk(pins) if k == "ifeval" and isinstance(v, dict) and "repo_id" in v)
    assert entry["repo_id"] == "google/IFEval"
    assert entry["revision"] == C.IFEVAL_DATASET_REVISION == "966cd89545d6b6acfd7638bc708b98261ca58e84"
    mod = _ifeval_module()
    assert mod.IFEVAL_DATASET_REVISION == C.IFEVAL_DATASET_REVISION
    assert mod.DATASET_PATH == "google/IFEval"
    meta = yaml.safe_load((Path(mod.__file__).parent / "eval.yaml").read_text())
    assert str(meta["version"]) == C.IFEVAL_TASK_VERSION == "2-A"
    assert meta["tasks"][0]["dataset_samples"] == C.IFEVAL_N == 541
    # P20: the framework string stays inspect_evals-0.18.0; the task version rides beside it
    assert C.INSPECT_TASK_VERSION == "inspect_evals-0.18.0"
    assert C.IFEVAL_LANGDETECT_SEED == 0
    assert C.IFEVAL_MAX_TOKENS_OFF == 2048 and C.IFEVAL_CAP_LADDER == (2048, 4096, 8192)
    assert C.IFEVAL_CAP_CEILING == 8192


def _ifeval_module():
    """The ``inspect_evals.ifeval.ifeval`` MODULE (the package re-exports the
    task function under the same name, shadowing the submodule attribute)."""
    import importlib
    return importlib.import_module("inspect_evals.ifeval.ifeval")


def _walk(node, prefix=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield k, v
            yield from _walk(v, prefix + (k,))


def test_the_samples_flag_is_explicit_for_math_and_ifeval_only():
    card, greedy = C.SAMPLING_CARD, C.SAMPLING_GREEDY
    assert C.SAMPLES_FLAG_TASKS == (C.inspect_task("math"), IFEVAL)
    assert C.samples_of_task(card, IFEVAL) == 2 and C.samples_of_task(greedy, IFEVAL) == 1
    assert C.samples_of_task(card, IFEVAL, explicit_samples=3) == 3
    assert C.samples_of_task(card, "inspect/math", explicit_samples=3) == 3
    assert C.samples_of_task(card, "inspect/gsm8k", explicit_samples=3) == 1        # ignored elsewhere
    assert C.samples_of_task(card, IFEVAL, math_samples=3) == 3                        # the alias
    with pytest.raises(ValueError, match="not both"):
        C.samples_of_task(card, IFEVAL, explicit_samples=2, math_samples=2)
    for bad in (0, -1, True, 2.0):
        with pytest.raises(ValueError, match="positive int"):
            C.samples_of_task(card, IFEVAL, explicit_samples=bad)
    m = C.run_samples_by_task([IFEVAL, "inspect/math", "inspect/gsm8k_paired200"], card, explicit_samples=2)
    assert m == {"inspect/gsm8k_paired200": 1, IFEVAL: 2, "inspect/math": 2}
    # the pre-IFEval keyword still works, so no existing caller moves
    assert C.run_samples_by_task(["inspect/math"], card, math_samples=4) == {"inspect/math": 4}


def test_the_row_facts_and_outcome_shape_and_summary_fields_are_named_once():
    assert C.IFEVAL_ROW_FACT_FIELDS == ("prompt_level_strict", "prompt_level_loose", "inst_level_strict",
                                        "inst_level_loose", "num_instructions", "instruction_id_list")
    # the ifeval_contracts fallback shape: the version-2 set, then the six row facts
    # the producer's order: the v1 fields, the six facts, sampling_ref LAST (it closes every shape that carries it)
    assert C.TASK_OUTCOME_ITEM_FIELDS_IFEVAL == (*C.TASK_OUTCOME_ITEM_FIELDS, *C.IFEVAL_ROW_FACT_FIELDS, C.SAMPLING_REF_FIELD)
    assert C.IFEVAL_OUTCOME_ITEM_FIELDS == C.TASK_OUTCOME_ITEM_FIELDS_IFEVAL
    assert C.TASK_OUTCOME_ITEM_FIELDS_IFEVAL in C.TASK_OUTCOME_ITEM_FIELD_SETS
    # a no-switch family's IFEval store (llama): the facts + the `none` split's close-tag stray, sampling_ref LAST
    assert C.IFEVAL_OUTCOME_ITEM_FIELDS_NO_SWITCH == C.TASK_OUTCOME_ITEM_FIELDS_IFEVAL_NO_SWITCH \
        == (*C.TASK_OUTCOME_ITEM_FIELDS_IFEVAL[:-1], "stray_close_tag", C.SAMPLING_REF_FIELD)
    assert C.TASK_OUTCOME_ITEM_FIELDS_IFEVAL_NO_SWITCH in C.TASK_OUTCOME_ITEM_FIELD_SETS
    assert set(C.IFEVAL_SUMMARY_FIELDS) <= set(C.SUMMARY_PAYLOAD_FIELDS)
    assert C.ITEM_SUBSET_TASKS == (C.GSM8K_PAIRED_TASK_NAME, C.IFEVAL_TASK_NAME)
    assert isv.PRIMARY_SCORER["ifeval"] == "instruction_following"
    assert isv.PRIMARY_VALUE_KEY == {"ifeval": "prompt_level_strict"}
    assert isv.MAX_TOKENS_OFF["ifeval"] == C.IFEVAL_MAX_TOKENS_OFF
    assert isv.ROW_FACTS_OF_TASK["ifeval"] is isv.ifeval_row_facts
    assert "ifeval" in isv.TASK_MESSAGES_RENDERERS


# --------------------------------------------------------------------------- #
# 2. the plan branch on the REAL stock task                                    #
# --------------------------------------------------------------------------- #


@needs_dataset
def test_the_plan_builds_the_stock_task_with_541_ids_the_pinned_scorer_and_k_2(real_plan):
    plan = real_plan
    assert plan.row_task == IFEVAL and plan.name == "ifeval"
    assert plan.expected_n == C.IFEVAL_N == 541 == plan.n_expected_run
    assert plan.sample_ids is not None and len(plan.sample_ids) == 541
    assert plan.sample_ids == sorted(plan.sample_ids) and plan.sample_ids[0] == "1000"
    assert plan.samples_per_item == 2                                        # the contract k (P2)
    assert plan.max_tokens == 2048 and plan.cap_source == "off_table"
    assert plan.primary_scorer == "instruction_following"
    assert isv._scorer_names(plan.task) == ["instruction_following"]
    assert plan.thinking == C.THINKING_OFF and plan.item_set == "ifeval541"
    assert plan.think_split_mode == C.THINK_SPLIT_RECORD_ONLY
    assert plan.grader_max_tokens is None                                    # no model grader
    assert plan.limit is None
    assert "2-A" in plan.note and "966cd895" in plan.note and "SEED_RULE_IFEVAL" in plan.note
    assert "langdetect seed 0" in plan.note
    import langdetect
    assert langdetect.DetectorFactory.seed == 0                              # P6
    # the sample id IS the vendor's own row key (the seed rule's item_key)
    sample = next(s for s in plan.task.dataset)
    assert str(sample.id) == "1000" and "instruction_id_list" in sample.metadata


@needs_dataset
def test_the_plan_refuses_a_drifted_dataset_or_a_moved_scorer(monkeypatch, real_plan):
    mod = _ifeval_module()

    class _Short:
        def __init__(self, task):
            self.dataset = list(task.dataset)[:-1]
            self.scorer = task.scorer

    monkeypatch.setattr(mod, "ifeval", lambda: _Short(real_plan.task))
    with pytest.raises(isv.InspectServedError, match="540 samples, not the frozen 541"):
        isv.plan_task("ifeval", grader_model=None, thinking=C.THINKING_OFF, repo_root=REPO)

    class _OtherScorer:
        def __init__(self, task):
            self.dataset = task.dataset
            self.scorer = [isv._gsm8k_scorer(C.THINK_SPLIT_RECORD_ONLY)]

    monkeypatch.setattr(mod, "ifeval", lambda: _OtherScorer(real_plan.task))
    with pytest.raises(isv.InspectServedError, match="the pinned scorer moved"):
        isv.plan_task("ifeval", grader_model=None, thinking=C.THINKING_OFF, repo_root=REPO)


@needs_dataset
def test_a_smoke_cap_heads_the_ids_and_the_item_subset_narrows_the_plan(real_plan, monkeypatch):
    mod = _ifeval_module()
    monkeypatch.setattr(mod, "ifeval", lambda: real_plan.task)   # skip the dataset reload
    smoke = isv.plan_task("ifeval", grader_model=None, thinking=C.THINKING_OFF, repo_root=REPO, inspect_limit=5)
    assert smoke.sample_ids == real_plan.sample_ids[:5] and smoke.n_run == 5 and smoke.expected_n == 541
    assert smoke.limit is None and isv.limit_plan(smoke, 5) is smoke
    subset = [real_plan.sample_ids[i] for i in (7, 3, 11)]
    rung = isv.plan_task("ifeval", grader_model=None, thinking=C.THINKING_OFF, repo_root=REPO,
                         item_subset=subset)
    assert rung.sample_ids == sorted(subset) and rung.expected_n == rung.n_run == 3
    assert "RUNG" in rung.note and rung.limit is None and rung.item_set == "ifeval541"
    with pytest.raises(isv.InspectServedError, match="not in the task's 541 sample ids"):
        isv.plan_task("ifeval", grader_model=None, thinking=C.THINKING_OFF, repo_root=REPO,
                      item_subset=subset + ["not-an-id"])
    with pytest.raises(isv.InspectServedError, match="names no ids"):
        isv.plan_task("ifeval", grader_model=None, thinking=C.THINKING_OFF, repo_root=REPO, item_subset=[])
    with pytest.raises(isv.InspectServedError, match="takes no smoke cap"):
        isv.plan_task("ifeval", grader_model=None, thinking=C.THINKING_OFF, repo_root=REPO,
                      item_subset=subset, inspect_limit=2)
    # the run's k map wins over the contract default
    k3 = isv.plan_task("ifeval", grader_model=None, thinking=C.THINKING_OFF, repo_root=REPO, samples_per_item=3)
    assert k3.samples_per_item == 3


@needs_dataset
def test_ifeval_messages_render_541_single_user_turns_and_fold_a_si_into_one_system_turn(real_plan, monkeypatch):
    mod = _ifeval_module()
    monkeypatch.setattr(mod, "ifeval", lambda: real_plan.task)
    bare = isv.ifeval_messages(REPO)
    assert len(bare) == 541
    assert all([m["role"] for m in msgs] == ["user"] for msgs in bare)
    assert all(isinstance(msgs[0]["content"], str) and msgs[0]["content"] for msgs in bare)
    prompted = isv.ifeval_messages(REPO, system="SI TEXT")
    assert len(prompted) == 541
    assert all([m["role"] for m in msgs] == ["system", "user"] for msgs in prompted)
    assert all(msgs[0]["content"] == "SI TEXT" for msgs in prompted)
    assert [m[-1]["content"] for m in prompted] == [m[0]["content"] for m in bare]
    # the renderer table routes the task name to it
    assert isv.task_messages("ifeval", repo_root=REPO, system="X")[0][0] == {"role": "system", "content": "X"}


# --------------------------------------------------------------------------- #
# 3. the dict-valued score: value key, row facts, flattening (P5)              #
# --------------------------------------------------------------------------- #

_VALUE = {"prompt_level_strict": True, "inst_level_strict": 2, "prompt_level_loose": True,
          "inst_level_loose": 2, "num_instructions": 2}


def test_score_to_value_reads_the_named_key_and_refuses_without_one():
    assert isv.score_to_value(_Score(_VALUE), value_key="prompt_level_strict") == 1.0
    assert isv.score_to_value(_Score({**_VALUE, "prompt_level_strict": False}), value_key="prompt_level_strict") == 0.0
    assert isv.score_to_value(_Score(_VALUE), value_key="inst_level_strict") == 2.0
    with pytest.raises(isv.InspectServedError, match="no single numeric reading"):
        isv.score_to_value(_Score(_VALUE))                                     # a dict with no key still refuses
    with pytest.raises(isv.InspectServedError, match="carries no 'nope'"):
        isv.score_to_value(_Score(_VALUE), value_key="nope")
    with pytest.raises(isv.InspectServedError, match="is not a dict, but the task names the value key"):
        isv.score_to_value(_Score("C"), value_key="prompt_level_strict")
    assert isv.score_to_value(_Score("C")) == 1.0                              # the scalar reading, untouched


class _IfSample(_Sample):
    def __init__(self, sid, value, ids=("keywords:existence", "length_constraints:number_words"),
                 completion="Some answer text.", stop_reason="stop", **kw):
        super().__init__(sid, {"instruction_following": _Score(value, answer=completion)},
                         output=_model_output(completion, stop_reason=stop_reason), **kw)
        self.metadata = {"prompt": "p", "instruction_id_list": list(ids), "kwargs": {}}


def test_ifeval_row_facts_read_the_five_keys_and_the_ids_and_refuse_a_moved_grammar():
    s = _IfSample("1000", _VALUE)
    facts = isv.ifeval_row_facts(s, s.scores["instruction_following"])
    assert facts == {"prompt_level_strict": True, "prompt_level_loose": True, "inst_level_strict": 2,
                     "inst_level_loose": 2, "num_instructions": 2,
                     "instruction_id_list": ["keywords:existence", "length_constraints:number_words"]}
    assert tuple(facts) == C.IFEVAL_ROW_FACT_FIELDS
    bad = [({**_VALUE, "extra": 1}, "not exactly the five pinned keys"),
           ({k: v for k, v in _VALUE.items() if k != "num_instructions"}, "not exactly the five"),
           ({**_VALUE, "prompt_level_strict": 1}, "not a bool"),
           ({**_VALUE, "inst_level_strict": True}, "not a non-negative int"),
           ({**_VALUE, "inst_level_strict": 3}, "not coherent"),
           ("C", "not exactly the five pinned keys")]
    for value, why in bad:
        s = _IfSample("1000", value)
        with pytest.raises(isv.InspectServedError, match=why):
            isv.ifeval_row_facts(s, s.scores["instruction_following"])
    s = _IfSample("1000", _VALUE, ids=("keywords:existence",))
    with pytest.raises(isv.InspectServedError, match="1 instruction ids but num_instructions 2"):
        isv.ifeval_row_facts(s, s.scores["instruction_following"])
    s = _IfSample("1000", _VALUE)
    s.metadata = {}
    with pytest.raises(isv.InspectServedError, match="instruction_id_list"):
        isv.ifeval_row_facts(s, s.scores["instruction_following"])


def test_results_from_log_files_the_facts_the_flattened_scores_and_unparseable_iff_empty():
    log = _Log([_IfSample("1000", _VALUE),
                _IfSample("1001", {**_VALUE, "prompt_level_strict": False, "inst_level_strict": 1}),
                _IfSample("1005", {**_VALUE, "prompt_level_strict": False, "prompt_level_loose": False,
                                   "inst_level_strict": 0, "inst_level_loose": 0},
                          completion="", stop_reason="max_tokens")])
    res = isv.results_from_log(log, primary_scorer="instruction_following",
                               row_facts=isv.ifeval_row_facts, primary_value_key="prompt_level_strict")
    assert [r.value for r in res] == [1.0, 0.0, 0.0]                          # acc == prompt_level_strict
    assert res[0].scores == {"instruction_following/inst_level_loose": 2.0,
                             "instruction_following/inst_level_strict": 2.0,
                             "instruction_following/num_instructions": 2.0,
                             "instruction_following/prompt_level_loose": 1.0,
                             "instruction_following/prompt_level_strict": 1.0}
    assert res[0].row_facts["instruction_id_list"] == ["keywords:existence", "length_constraints:number_words"]
    assert res[1].row_facts["inst_level_strict"] == 1
    # unparseable := empty completion (P5), never a judgement about the instructions
    assert [r.unparseable for r in res] == [False, False, True]
    assert [r.at_cap for r in res] == [False, False, True] and res[2].stop_reason == "max_tokens"
    extra = res[0].extra(thinking="off")
    for k in C.IFEVAL_ROW_FACT_FIELDS:
        assert k in extra
    # the legacy scalar path is byte-identical (no key named): a dict still refuses
    with pytest.raises(isv.InspectServedError, match="no single numeric reading"):
        isv.results_from_log(log, primary_scorer="instruction_following")


# --------------------------------------------------------------------------- #
# 4. the runner at k = 2: per-sample rows, the mean row, the outcome record,   #
#    the summary's IFEval fields (§1.2)                                        #
# --------------------------------------------------------------------------- #


def _ifeval_log(idx: int) -> _Log:
    # trial 0: both items pass strictly; trial 1: item 1001 fails one instruction
    v_pass = dict(_VALUE)
    v_fail = {**_VALUE, "prompt_level_strict": False, "inst_level_strict": 1}
    return _Log([_IfSample("1000", v_pass), _IfSample("1001", v_pass if idx == 0 else v_fail)])


def _ifeval_plan(**kw):
    base = dict(row_task=IFEVAL, name="ifeval", task=object(), expected_n=2, samples_per_item=2,
                primary_scorer="instruction_following", max_tokens=2048, thinking=C.THINKING_OFF,
                item_set="ifeval541", sample_ids=["1000", "1001"], note="ifeval test plan")
    base.update(kw)
    return isv.TaskPlan(**base)


def test_run_one_task_at_k_2_files_acc_rows_per_trial_the_mean_row_and_an_ifeval_shaped_record(tmp_path, monkeypatch):
    seen: List[Any] = []
    monkeypatch.setattr("inspect_ai.eval_async", _fake_eval({0: _ifeval_log(0), 1: _ifeval_log(1)}, seen))
    em, store = _Emitter(), _Store(tmp_path / "cell")
    # production IFEval runs are identity v2 and carry a sampling_ref on every row / item
    summary = asyncio.run(isv.run_one_task(_ifeval_plan(), model=None, store=store, template={},
                                           concurrency=4, emitter=em, sampling_ref="abc123def456"))
    assert [s[1] for s in seen] == [0, 1] and {s[2] for s in seen} == {2048}
    for k in range(2):
        rows = em.by(C.sample_metric(C.METRIC_ACC, k))
        assert [r["item_id"] for r in rows] == ["1000", "1001"]
        for r in rows:
            assert r["extra"]["thinking"] == "off" and r["extra"]["max_tokens"] == 2048
            assert r["extra"]["sampling_ref"] == "abc123def456"
            for f in C.IFEVAL_ROW_FACT_FIELDS:
                assert f in r["extra"]
    means = {r["item_id"]: r["value"] for r in em.by(C.METRIC_ACC)}
    assert means == {"1000": 1.0, "1001": 0.5}
    mean_row = next(r for r in em.by(C.METRIC_ACC) if r["item_id"] == "1001")
    assert mean_row["extra"]["samples_per_item"] == 2 and mean_row["extra"]["values"] == [1.0, 0.0]
    assert len(mean_row["extra"]["seeds"]) == 2
    agg = em.by(C.METRIC_ACC_AGGREGATE)[0]
    assert agg["value"] == pytest.approx(0.75) and agg["extra"]["samples_per_item"] == 2
    assert agg["extra"]["ifeval_task_version"] == "2-A" and agg["extra"]["langdetect_seed"] == 0
    # the summary: the payload rules hold, the IFEval fields are present, per trial
    _assert_counts_only({"ifeval": summary})
    assert set(summary) <= set(C.SUMMARY_PAYLOAD_FIELDS) | {
        "logs_reused", "n_paired", "render", "reasoning_parser", "n_content_empty_with_reasoning",
        C.SAMPLING_REF_FIELD}
    assert summary[C.SAMPLING_REF_FIELD] == "abc123def456"
    assert summary["task"] == IFEVAL and summary["item_set"] == "ifeval541" and summary["samples_per_item"] == 2
    assert summary["ifeval_task_version"] == "2-A" and summary["langdetect_seed"] == 0
    assert summary["prompt_strict_mean_by_trial"] == [1.0, 0.5]
    assert summary["prompt_loose_mean_by_trial"] == [1.0, 1.0]
    assert summary["inst_strict_mean_by_trial"] == [1.0, 0.75]          # (2+1) / (2+2) followed / total
    assert summary["inst_loose_mean_by_trial"] == [1.0, 1.0]
    assert summary[C.METRIC_ACC] == pytest.approx(0.75) and summary["n"] == 2
    # the outcome record: EXACTLY the IFEval item shape, the facts on every item
    from sisteer.capeval.tinycap import completion
    outcomes = completion.read_task_outcomes(store.cell_dir, IFEVAL)
    assert outcomes["samples_per_item"] == 2 and outcomes["n_attempted_samples"] == 4
    for item in outcomes["items"]:
        assert set(item) == set(C.TASK_OUTCOME_ITEM_FIELDS_IFEVAL)
        assert item["sampling_ref"] == "abc123def456"
    by = {(i["item_id"], i["sample_idx"]): i for i in outcomes["items"]}
    assert by[("1001", 1)]["inst_level_strict"] == 1 and by[("1001", 1)]["prompt_level_strict"] is False
    assert by[("1001", 1)]["value"] == 0.0 and by[("1000", 0)]["value"] == 1.0
    assert completion.is_done(store.cell_dir, IFEVAL)


def test_a_non_ifeval_task_files_no_ifeval_fields(tmp_path, monkeypatch):
    """Every existing task's summary, aggregate row and outcome items are byte-identical."""
    log = _Log([_Sample("math_aaaa1111", {"s": _Score("C")}), _Sample("math_bbbb2222", {"s": _Score("I")})])
    monkeypatch.setattr("inspect_ai.eval_async", _fake_eval({0: log}))
    em, store = _Emitter(), _Store(tmp_path / "cell")
    summary = asyncio.run(isv.run_one_task(_plan(primary_scorer=None, max_tokens=64), model=None, store=store,
                                           template={}, thinking=C.THINKING_OFF, concurrency=2, emitter=em))
    assert not set(C.IFEVAL_SUMMARY_FIELDS) & set(summary)
    assert not set(C.IFEVAL_SUMMARY_FIELDS) & set(em.by(C.METRIC_ACC_AGGREGATE)[0]["extra"])
    from sisteer.capeval.tinycap import completion
    for item in completion.read_task_outcomes(store.cell_dir, "inspect/math")["items"]:
        assert set(item) == set(C.TASK_OUTCOME_ITEM_FIELDS)
    assert isv.ifeval_summary_fields(_plan(), []) == {}


def test_an_ifeval_result_without_its_facts_refuses_the_outcome_record():
    r = isv.ItemResult(item_id="1000", value=1.0, sample_idx=0, seed=1, answer="x", prompt_tokens=1,
                       completion_tokens=1, paired_id=None, scores={}, stop_reason="stop")
    plan = _ifeval_plan()
    with pytest.raises(isv.InspectServedError, match="carries no IFEval row facts"):
        isv.outcome_record([r], plan=plan, task=IFEVAL, thinking="off",
                           summary={"n_scored": 1, "n_unscored": 0, "n_scored_samples": 1, "n_unscored_samples": 0,
                                    "n_stop_reason": 1, "n_at_cap": 0},
                           split_counts=isv.think_split_counts([r], plan=plan), producer_tree_sha=None)


def test_unscored_draws_leave_the_per_trial_means_and_a_missing_trial_reads_null():
    plan = _ifeval_plan()
    ok = isv.results_from_log(_ifeval_log(0), sample_idx=0, primary_scorer="instruction_following",
                              row_facts=isv.ifeval_row_facts, primary_value_key="prompt_level_strict")
    nan = isv.ItemResult(item_id="1000", value=math.nan, sample_idx=1, seed=1, answer="", prompt_tokens=1,
                         completion_tokens=1, paired_id=None, scores={}, row_facts=ok[0].row_facts)
    fields = isv.ifeval_summary_fields(plan, [*ok, nan])
    assert fields["prompt_strict_mean_by_trial"] == [1.0, None]
    assert fields["inst_strict_mean_by_trial"] == [1.0, None]
