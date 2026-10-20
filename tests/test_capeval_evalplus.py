"""EvalPlus generation-seam tests.

BASE lane: the fixture-pinned wire formats — samples.jsonl shape,
eval-results -> per-item row conversion (the interface the execution
wave consumes), and the generation flow's F8 ordering with a stubbed
engine/store (evalplus itself monkeypatched out).
HEAVY lane (the capeval venv on a cluster host): the real 0.3.1 imports —
make_raw_chat_prompt convention, sanitize, problem data. Skips are loud and
name the lane.
"""

import json
from pathlib import Path

import pytest

from sisteer.capeval.serving import GenOut
from sisteer.capeval.tiers import evalplus_runner as er
import evalplus_synthetic as EP  # noqa: E402  (tests/ helper: the synthetic EvalPlus sample pair)

ROOT = Path(__file__).parents[1]

HEAVY_SKIP = (
    "evalplus not importable — HEAVY-LANE test: run in the capeval venv on a cluster host "
    "(/home/$USER/venvs/sisteer-capeval, py3.12)"
)


# ---------------------------------------------------------------------------
# BASE LANE
# ---------------------------------------------------------------------------


def test_eval_results_to_rows_matches_frozen_fixture():
    results = EP.eval_results()
    rows = er.eval_results_to_rows(results)
    by = {(r["item_id"], r["metric"]): r for r in rows}
    assert by[("HumanEval/0", "pass@1_base")]["value"] == 1.0
    assert by[("HumanEval/0", "pass@1_plus")]["value"] == 1.0
    assert by[("HumanEval/1", "pass@1_base")]["value"] == 0.0
    assert by[("HumanEval/1", "pass@1_plus")]["value"] == 0.0
    # simplify-pass #7: the normalized GATING row the code domain pairs
    # on — metric="acc", value = pass@1_plus, source named in extra_json
    assert by[("HumanEval/0", "acc")]["value"] == 1.0
    assert by[("HumanEval/1", "acc")]["value"] == 0.0
    assert json.loads(by[("HumanEval/0", "acc")]["extra_json"]) == {
        "source_metric": "pass@1_plus", "metric_def": "base_and_plus"}
    # PW12 (tinycap E19): the pass@1_plus DEFINITION is named on the rows that carry it (plus and
    # the gating acc); pass@1_base needs none — its definition never changed
    assert json.loads(by[("HumanEval/0", "pass@1_plus")]["extra_json"]) == {"metric_def": "base_and_plus"}
    assert by[("HumanEval/0", "pass@1_base")]["extra_json"] is None
    assert er.PLUS_METRIC_DEF == "base_and_plus"
    for r in rows:
        assert r["task"] == "humaneval_plus" and r["domain"] == "code"
        assert r["metric"] in er.METRICS + (er.GATING_ROW_METRIC,)
    # v0 is one greedy sample per task: extra samples refuse, never average
    doubled = {**results, "eval": {
        k: v * 2 for k, v in results["eval"].items()
    }}
    with pytest.raises(ValueError, match="one greedy sample"):
        er.eval_results_to_rows(doubled)
    # a missing status field refuses, never defaults to fail (tier1 #4)
    broken = {**results, "eval": {
        k: [{kk: vv for kk, vv in v[0].items() if kk != "plus_status"}]
        for k, v in results["eval"].items()}}
    with pytest.raises(ValueError, match="plus_status"):
        er.eval_results_to_rows(broken)


def test_eval_results_to_rows_plus_requires_base_too():
    """PW10 (tinycap, 2026-09-04): ``pass@1_plus`` is evalplus's own headline
    — a sample passes the plus set only when it passes the BASE tests AND
    the plus tests (evalplus 0.3.1 ``evaluate.py``: ``base_status ==
    plus_status == PASS``). A base-fail / plus-pass item therefore scores
    0.0 on ``pass@1_plus`` and on the gating ``acc``; ``pass@1_base`` is
    ``base_status`` alone."""
    results = EP.eval_results()
    mixed = {**results, "eval": {
        "HumanEval/0": [{**results["eval"]["HumanEval/0"][0],
                         "base_status": "fail", "plus_status": "pass"}],
        "HumanEval/1": [{**results["eval"]["HumanEval/1"][0],
                         "base_status": "pass", "plus_status": "fail"}]}}
    by = {(r["item_id"], r["metric"]): r["value"] for r in er.eval_results_to_rows(mixed)}
    assert by[("HumanEval/0", "pass@1_base")] == 0.0
    assert by[("HumanEval/0", "pass@1_plus")] == 0.0 == by[("HumanEval/0", "acc")]
    assert by[("HumanEval/1", "pass@1_base")] == 1.0
    assert by[("HumanEval/1", "pass@1_plus")] == 0.0 == by[("HumanEval/1", "acc")]


def test_eval_results_to_rows_enforces_pinned_task_ids():
    """Review tier1 #4: the conversion path refuses a partial eval when
    handed the pinned ID set — an inflated pass@1 never records."""
    results = EP.eval_results()
    with pytest.raises(er.EvalPlusSeamError, match="164"):
        er.eval_results_to_rows(results,
                                expected_task_ids=er.EXPECTED_TASK_IDS)
    # matching set passes
    two = frozenset(results["eval"])
    rows = er.eval_results_to_rows(results, expected_task_ids=two)
    assert len(rows) == 2 * 3  # base + plus + acc per task


def test_validate_task_ids_pins_the_164_set():
    assert len(er.EXPECTED_TASK_IDS) == 164
    er.validate_task_ids([f"HumanEval/{i}" for i in range(164)], where="t")
    with pytest.raises(er.EvalPlusSeamError, match="163"):
        er.validate_task_ids([f"HumanEval/{i}" for i in range(163)], where="t")
    with pytest.raises(er.EvalPlusSeamError, match="duplicate"):
        er.validate_task_ids(["HumanEval/0"] * 2 +
                             [f"HumanEval/{i}" for i in range(1, 163)],
                             where="t")


def test_samples_jsonl_roundtrip_and_shape(tmp_path):
    fixture_rows = er.read_samples_jsonl(EP.write(tmp_path)[0])
    assert [set(s) for s in fixture_rows] == [{"task_id", "solution"}] * len(fixture_rows)


def test_write_samples_jsonl_refuses_extra_keys(tmp_path):
    with pytest.raises(ValueError, match="task_id"):
        er.write_samples_jsonl(
            tmp_path / "s.jsonl",
            [{"task_id": "HumanEval/0", "solution": "x", "extra": 1}],
        )


class _StubStore:
    """Frozen CellStore surface used by generate_samples: cell_dir only —
    generation_complete.json belongs to the DRIVER now (review tier1 #3)."""

    def __init__(self, root: Path):
        self.cell_dir = root
        self.events = []

    def write_generation_complete(self):  # pragma: no cover - must not run
        self.events.append("generation_complete")


class _StubEngine:
    def __init__(self, texts):
        self.texts = texts
        self.calls = []

    def generate(self, prompts, arm, gen_cfg):
        self.calls.append((list(prompts), arm, dict(gen_cfg)))
        return [GenOut(text=t, gen_token_logprobs=[-0.1], ended_with_eos=True)
                for t in self.texts[: len(prompts)]]


TWO_IDS = frozenset({"HumanEval/0", "HumanEval/1"})


def test_generate_samples_order_and_format(monkeypatch, tmp_path):
    prompts = [
        er.RenderedPrompt("HumanEval/0", "PROMPT0", "has_close_elements"),
        er.RenderedPrompt("HumanEval/1", "PROMPT1", "separate_paren_groups"),
    ]
    monkeypatch.setattr(er, "render_prompts", lambda tok, problems=None, limit=None: prompts)
    monkeypatch.setattr(er, "sanitize_solution", lambda text, ep: f"# {ep}\n{text}")
    store = _StubStore(tmp_path)
    engine = _StubEngine(["def a(): pass", "def b(): pass"])
    samples = er.generate_samples(
        engine=engine, arm=None, tokenizer=object(),
        gen_cfg={"do_sample": False, "max_new_tokens": 8},
        store=store, problems={"synthetic": {}},
        expected_task_ids=TWO_IDS,
    )
    # engine got the rendered prompts, arm passed through, greedy cfg intact
    assert engine.calls[0][0] == ["PROMPT0", "PROMPT1"]
    assert engine.calls[0][2]["do_sample"] is False
    # samples.jsonl written atomically with the fixture shape; NO marker
    # from here (the driver writes it after results_pending — tier1 #3)
    on_disk = er.read_samples_jsonl(tmp_path / "samples.jsonl")
    assert on_disk == samples
    assert [set(s) for s in on_disk] == [{"task_id", "solution"}] * 2
    assert store.events == []
    assert not (tmp_path / "samples.jsonl.tmp").exists()  # atomic replace
    # sanitize ran per sample with the problem's entry point
    assert on_disk[0]["solution"].startswith("# has_close_elements")


def test_generate_samples_bounds_the_generation_batch(monkeypatch, tmp_path):
    """OOM guard (job 115651_14): the stock HumanEval+ generation cfg
    (``DEFAULT_GEN_CFG``) carries no batch_size, and the engine's no-batch_size default is
    ONE chunk of all 164 problems — whose retained fp32 per-step scores
    + KV cache OOM'd an H200 next to control:weight_noise's on-GPU
    weight caches. generate_samples must hand the engine a bounded
    default batch_size; an explicitly configured one still wins."""
    prompts = [
        er.RenderedPrompt("HumanEval/0", "P0", "f"),
        er.RenderedPrompt("HumanEval/1", "P1", "g"),
    ]
    monkeypatch.setattr(er, "render_prompts",
                        lambda tok, problems=None, limit=None: prompts)
    monkeypatch.setattr(er, "sanitize_solution", lambda text, ep: text)

    # the stock cfg's shape — no batch_size — gets the bounded default
    engine = _StubEngine(["def a(): pass", "def b(): pass"])
    er.generate_samples(
        engine=engine, arm=None, tokenizer=object(),
        gen_cfg={"do_sample": False, "max_new_tokens": 8},
        store=_StubStore(tmp_path), problems={"synthetic": {}},
        expected_task_ids=TWO_IDS,
    )
    assert engine.calls[0][2]["batch_size"] == 16

    # an explicit batch_size is respected, never overridden
    engine2 = _StubEngine(["def a(): pass", "def b(): pass"])
    er.generate_samples(
        engine=engine2, arm=None, tokenizer=object(),
        gen_cfg={"do_sample": False, "max_new_tokens": 8, "batch_size": 4},
        store=_StubStore(tmp_path / "b"), problems={"synthetic": {}},
        expected_task_ids=TWO_IDS,
    )
    assert engine2.calls[0][2]["batch_size"] == 4


def test_generate_samples_refuses_partial_task_set(monkeypatch, tmp_path):
    """Review tier1 #4: a rendered set that is not the pinned ID set
    refuses BEFORE any generation cost, and nothing lands on disk."""
    prompts = [er.RenderedPrompt("HumanEval/0", "P0", "f")]
    monkeypatch.setattr(er, "render_prompts",
                        lambda tok, problems=None, limit=None: prompts)
    engine = _StubEngine(["def a(): pass"])
    with pytest.raises(er.EvalPlusSeamError, match="164"):
        er.generate_samples(engine=engine, arm=None, tokenizer=object(),
                            gen_cfg={}, store=_StubStore(tmp_path),
                            problems={"synthetic": {}})
    assert engine.calls == []  # refused before generating
    assert not (tmp_path / "samples.jsonl").exists()


def test_generate_samples_length_mismatch_is_loud(monkeypatch, tmp_path):
    prompts = [er.RenderedPrompt("HumanEval/0", "P0", "f")]
    monkeypatch.setattr(er, "render_prompts", lambda tok, problems=None, limit=None: prompts)

    class ShortEngine:
        def generate(self, p, a, g):
            return []

    with pytest.raises(er.EvalPlusSeamError, match="0 outputs"):
        er.generate_samples(
            engine=ShortEngine(), arm=None, tokenizer=object(), gen_cfg={},
            store=_StubStore(tmp_path), problems={"synthetic": {}},
            expected_task_ids=frozenset({"HumanEval/0"}),
        )


def test_read_samples_jsonl_validates_when_asked(tmp_path):
    p = tmp_path / "s.jsonl"
    er.write_samples_jsonl(p, [{"task_id": "HumanEval/0", "solution": "x"}])
    assert er.read_samples_jsonl(p) == [{"task_id": "HumanEval/0", "solution": "x"}]
    with pytest.raises(er.EvalPlusSeamError, match="164"):
        er.read_samples_jsonl(p, expected_task_ids=er.EXPECTED_TASK_IDS)


def test_prefixes_are_the_evalplus_031_defaults():
    # provenance pin (codegen.py:176-177 at 0.3.1) — a drifted copy would
    # silently change every HumanEval+ prompt
    assert er.INSTRUCTION_PREFIX.startswith("Please provide a self-contained")
    assert er.RESPONSE_PREFIX.startswith("Below is a Python script")
    assert er.GATING_METRIC == "pass@1_plus" and er.METRICS == (
        "pass@1_base", "pass@1_plus",
    )


# ---------------------------------------------------------------------------
# HEAVY LANE (evalplus importable — the cluster's capeval venv)
# ---------------------------------------------------------------------------


def test_make_raw_chat_prompt_importable_at_pin():
    pytest.importorskip("evalplus", reason=HEAVY_SKIP)
    # the resolved §5.5 [impl-verify]: import the helper, don't vendor it
    from evalplus.provider.utility import make_raw_chat_prompt  # noqa: F401
    import evalplus

    assert evalplus.__version__ == "0.3.1"


def test_render_prompts_refuses_templateless_tokenizer(tiny_llama):
    pytest.importorskip("evalplus", reason=HEAVY_SKIP)
    _, tok = tiny_llama
    if getattr(tok, "chat_template", None) is not None:
        pytest.skip("tiny tokenizer unexpectedly has a chat template")
    with pytest.raises(er.EvalPlusSeamError, match="chat_template"):
        er.render_prompts(tok, problems={"HumanEval/0": {"prompt": "def f():\n", "entry_point": "f"}})


def test_sanitize_extracts_fenced_function():
    pytest.importorskip("evalplus", reason=HEAVY_SKIP)
    gen = (
        "def add(a, b):\n    return a + b\n```\n"
        "That's my solution, hope it helps!"
    )
    out = er.sanitize_solution(gen, "add")
    assert "return a + b" in out and "hope it helps" not in out
