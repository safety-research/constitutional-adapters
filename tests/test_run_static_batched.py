"""Batched run_static: deterministic fixed-size chunks through ONE
defense.generate call each, rows appended in the unbatched order, a goal's
progress record written only after ALL its variants complete (so per-goal
resume is unchanged), a failed batch terminal-errors exactly the goals in
it, and the batch size is recorded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from sisteer.eval.attacks.static_common import run_static
from sisteer.eval.jailbreak import build_defense


@dataclass(frozen=True)
class _Goal:
    goal_id: str
    text: str
    category: str = "benign_synthetic"


def _goals(n):
    return [_Goal(goal_id=f"g{i:02d}", text=f"topic {i}") for i in range(n)]


def _real_pairs():
    from sisteer.data.pairs import Pair
    return [Pair(prompt="Say hi.", pos="Bonjour", neg="Hello", instruction_id="french"),
            Pair(prompt="Describe a tree.", pos="Un arbre", neg="A tree",
                 instruction_id="french")]


def _variants(goal, seed=0):
    # 2 deterministic variants per goal, like artprompt.
    return [("v0", f"{goal.text} / a"), ("v1", f"{goal.text} / b")]


def _defense(batch_size, calls, fail_on_call=None):
    """A real LLMHFSteered through the generate_fn seam; records each generate
    call's size and can fail a specific call."""
    def seam(rendered, **kw):
        calls.append(len(rendered))
        if fail_on_call is not None and len(calls) == fail_on_call:
            raise RuntimeError("boom")
        return [f"resp:{r[-20:]}" for r in rendered]
    return build_defense(None, None, mode="none", positions="all",
                         batch_size=batch_size, generate_fn=seam)


def _rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def test_batches_are_deterministic_fixed_size_and_order_preserved(tmp_path):
    calls: list[int] = []
    d = _defense(4, calls)
    res = run_static(d, _goals(5), out_dir=tmp_path / "r", family="fam",
                     variants_fn=_variants)
    # 5 goals x 2 variants = 10 prompts -> consecutive chunks of 4: [4, 4, 2]
    assert calls == [4, 4, 2]
    assert res["batch_size"] == 4 and res["n_batches"] == 3
    rows = _rows(res["responses_path"])
    # rows appended in the exact unbatched order (goal-major, variant order)
    assert [(r["goal_id"], r["sample_idx"]) for r in rows] == \
        [(f"g{i:02d}", j) for i in range(5) for j in range(2)]
    # a re-run over the same inputs reproduces the same batches
    calls2: list[int] = []
    d2 = _defense(4, calls2)
    run_static(d2, _goals(5), out_dir=tmp_path / "r2", family="fam",
               variants_fn=_variants)
    assert calls2 == calls


def test_progress_written_only_after_all_variants_and_accounting(tmp_path):
    calls: list[int] = []
    d = _defense(3, calls)
    res = run_static(d, _goals(3), out_dir=tmp_path / "r", family="fam",
                     variants_fn=_variants)
    recs = res["per_goal"]
    assert [r["goal_id"] for r in recs] == ["g00", "g01", "g02"]  # goal order
    assert all(r["outcome"] == "done" and r["n_variants"] == 2
               and r["n_target_forwards"] == 2 for r in recs)
    # the JB7 counter counted every prompt exactly once
    assert res["n_target_forwards_total"] == 6


def test_resume_skips_covered_goals_and_drops_partial_rows(tmp_path):
    out = tmp_path / "r"
    calls: list[int] = []
    d = _defense(4, calls)
    run_static(d, _goals(2), out_dir=out, family="fam", variants_fn=_variants)
    # simulate a crash mid-batch on a THIRD goal: partial row, no progress record
    with open(out / "responses.jsonl", "a") as f:
        f.write(json.dumps({"goal_id": "g02", "sample_idx": 0, "trial": 0,
                            "variant_key": "0:0", "variant_name": "v0",
                            "family": "fam", "variant": "x",
                            "attack_prompt": "x", "response": "partial"}) + "\n")
    calls2: list[int] = []
    d2 = _defense(4, calls2)
    res = run_static(d2, _goals(3), out_dir=out, family="fam",
                     variants_fn=_variants)
    assert res["n_goals_run"] == 1                      # only g02 redone
    assert calls2 == [2]                                # its 2 variants, batched
    rows = _rows(out / "responses.jsonl")
    g02 = [r for r in rows if r["goal_id"] == "g02"]
    assert [r["sample_idx"] for r in g02] == [0, 1]      # partial row dropped
    assert not any(r["response"] == "partial" for r in g02)


def test_failed_batch_terminal_errors_exactly_its_goals(tmp_path):
    # batch_size=3 over 3 goals x 2 variants: chunks [g0v0,g0v1,g1v0],
    # [g1v1,g2v0,g2v1]. Fail the FIRST chunk -> g00+g01 error (g01's remaining
    # variant skipped); the second chunk then serves only g02's variants.
    calls: list[int] = []
    d = _defense(3, calls, fail_on_call=1)
    res = run_static(d, _goals(3), out_dir=tmp_path / "r", family="fam",
                     variants_fn=_variants)
    by_id = {r["goal_id"]: r for r in res["per_goal"]}
    assert by_id["g00"]["outcome"].startswith("error:RuntimeError")
    assert by_id["g01"]["outcome"].startswith("error:RuntimeError")
    assert by_id["g02"]["outcome"] == "done"
    assert calls == [3, 2]                               # g01's tail skipped
    rows = _rows(res["responses_path"])
    assert {r["goal_id"] for r in rows} == {"g02"}
    # errored goals are RETRIED on resume (2026-08-26: a terminal-error
    # skip made a total failure permanent and invisible — job 206660);
    # the successful goal stays covered.
    calls2: list[int] = []
    d2 = _defense(3, calls2)
    res2 = run_static(d2, _goals(3), out_dir=tmp_path / "r", family="fam",
                      variants_fn=_variants)
    assert res2["n_goals_run"] == 2 and calls2 == [3, 1]
    by_id2 = {r["goal_id"]: r for r in res2["per_goal"]}
    assert by_id2["g00"]["outcome"] == "done"
    assert by_id2["g01"]["outcome"] == "done"
    rows2 = _rows(res2["responses_path"])
    assert {r["goal_id"] for r in rows2} == {"g00", "g01", "g02"}


def test_batch_size_defaults_to_the_defense_and_floors_at_one(tmp_path):
    calls: list[int] = []
    d = _defense(8, calls)
    res = run_static(d, _goals(2), out_dir=tmp_path / "r", family="fam",
                     variants_fn=_variants)
    assert res["batch_size"] == 8 and calls == [4]       # 4 prompts, one batch
    calls2: list[int] = []
    d2 = _defense(8, calls2)
    res2 = run_static(d2, _goals(2), out_dir=tmp_path / "r2", family="fam",
                      variants_fn=_variants, batch_size=1)
    assert res2["batch_size"] == 1 and calls2 == [1, 1, 1, 1]  # unbatched mode


def test_stamp_records_batch_size_not_a_pairing_axis():
    from sisteer.eval import sweep
    c = sweep.Condition(label="c", cell="none", adaptive=())
    st = c.held_out_stamp(batch_size=8)
    assert st["batch_size"] == 8
    # NOT a pairing axis: two stamps differing only in batch_size still pair.
    a = sweep.engine_stamp(target_engine="hf", served_template_hash="t",
                           batch_size=8)
    b = sweep.engine_stamp(target_engine="hf", served_template_hash="t",
                           batch_size=1)
    assert sweep.assert_pairable(a, b, what="batch size differs")


def test_batched_vs_unbatched_equivalence_on_real_generation(tiny_llama, tmp_path):
    """The equivalence bar: NOT byte-identity (bf16 batching nondeterminism
    makes that a known-fail on GPU) — structural equality (same rows, same
    order, same coverage) plus a REPORTED per-item divergence rate, so the
    size of the batching effect is measured rather than discovered later."""
    model, tok = tiny_llama
    goals = _goals(3)
    outs = {}
    for label, bs in (("unbatched", 1), ("batched", 3)):
        d = build_defense(model, tok, mode="none", positions="all",
                          batch_size=bs)
        res = run_static(d, goals, out_dir=tmp_path / label, family="fam",
                         variants_fn=_variants, max_new_tokens=4)
        assert res["batch_size"] == bs
        outs[label] = _rows(res["responses_path"])
    keys_u = [(r["goal_id"], r["sample_idx"]) for r in outs["unbatched"]]
    keys_b = [(r["goal_id"], r["sample_idx"]) for r in outs["batched"]]
    assert keys_u == keys_b                     # same items, same order
    n = len(keys_u)
    diverged = sum(1 for a, b in zip(outs["unbatched"], outs["batched"])
                   if a["response"] != b["response"])
    # reported, not gated: fp32 CPU is usually identical; bf16 GPU will not be.
    print(f"[equivalence] per-item divergence: {diverged}/{n}")


def test_batch_reaches_model_generate_as_one_tensor(tiny_llama):
    """A chunk must arrive at model.generate as ONE [B, S] tensor, not a
    per-item loop. Instruments the REAL path (no generate_fn seam):
    run_static -> defense.generate(chunk) -> generate_with_object ->
    generate_batch -> model.generate. A per-item loop (the original defect's
    shape) would show B distinct [1, S] calls and would make the batching a
    no-op that only adds overhead — exactly the signature to rule out.
    Checked for BOTH the no-object (none) and compiled-object paths."""
    model, tok = tiny_llama
    from sisteer.methods import ARMS, object_artifact_name
    from sisteer.data.instructions import BY_ID

    def _capture_batch_dims(defense, n_prompts, bs, tmp):
        calls = []
        orig = model.generate

        def spy(*a, **kw):
            ids = kw.get("input_ids", a[0] if a else None)
            calls.append(tuple(ids.shape))
            return orig(*a, **kw)
        model.generate = spy
        try:
            run_static(defense, _goals(n_prompts // 2), out_dir=tmp,
                       family="fam", variants_fn=_variants, max_new_tokens=3,
                       batch_size=bs)
        finally:
            model.generate = orig
        return [shape[0] for shape in calls]  # batch dim of each call

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        # none arm: 4 goals x 2 variants = 8 prompts, bs=8 -> ONE [8, S] call.
        d_none = build_defense(model, tok, mode="none", positions="all",
                               batch_size=8)
        dims = _capture_batch_dims(d_none, 8, 8, d + "/none")
        assert dims == [8], f"expected one batch of 8, got {dims} (per-item loop?)"

    # a compiled object arm through the obj.apply chunk path.
    import tempfile as _t
    with _t.TemporaryDirectory() as d:
        from pathlib import Path
        path = Path(d) / "vector" / object_artifact_name("vector")
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = ARMS["vector"].train(model, tok, BY_ID["french"], _real_pairs(),
                                   layer=1, epochs=1, batch_size=2, max_len=64)
        obj.save(path)
        d_obj = build_defense(model, tok, mode="vector", object_path=path,
                              positions="all", batch_size=8)
        dims = _capture_batch_dims(d_obj, 8, 8, d + "/run")
        assert dims == [8], f"compiled arm: expected one batch of 8, got {dims}"
