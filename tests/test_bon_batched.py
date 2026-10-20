"""Batched BoN generation. BoN's n_max variants per
goal are DIFFERENT augmented prompts, so batching them is plain prompt batching;
it changes decode numerics but not what is computed. These pin the invariants
required before launch:

- variant IDENTITY + ORDER preserved (sample_idx -> the right augmentation),
  so the ASR@N prefix property holds;
- per-goal grouping and resume/dedup on (goal_id, sample_idx, trial);
- a FIXED width with the ragged final chunk PADDED, and pad rows NEVER written
  and NEVER counted as forwards (no silent n-inflation);
- a goal whose variants span multiple chunks still reassembles;
- the DEFAULT width is DEFAULT_BON_BATCH_SIZE (64, the production-proven
  value); an EXPLICIT width 1 stays byte-for-byte the legacy one-at-a-time
  path;
- the batch reaches the model as ONE tensor (not a per-item loop) on the real
  generation path.
"""

from __future__ import annotations

import json

from sisteer.eval.attacks.bon import (DEFAULT_BON_BATCH_SIZE,
                                       _chunk_sampling_seed, augment,
                                       bon_generate, bon_seed,
                                       load_bon_responses)
from sisteer.eval.jailbreak import LLMHFSteered, build_defense
from sisteer.eval.jailbreak_data import Goal


def _goals(n):
    return [Goal(f"g{i}", f"benign goal number {i} about topic {i}", "harmful",
                 "harmbench", "rev") for i in range(n)]


def _seam_defense(calls, batch_size=8):
    """A real LLMHFSteered through the generate_fn seam: records each call's
    prompt list + the sampling_seed, and echoes a per-prompt response tagged
    with its content so ordering is checkable."""
    def seam(rendered, *, sampling_seed=None, **kw):
        calls.append({"n": len(rendered), "seed": sampling_seed,
                      "prompts": list(rendered)})
        return [f"resp[{sampling_seed}]:{r[-24:]}" for r in rendered]
    return LLMHFSteered(model=None, tok=None, mode="none",
                        batch_size=batch_size, generate_fn=seam)


def _rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


# -- identity + ordering + grouping ------------------------------------------


def test_variant_identity_and_order_preserved_across_widths(tmp_path):
    goals = _goals(3)
    n_max = 8
    keys = {}
    variants = {}
    for label, bs in (("w1", 1), ("w4", 4), ("w16", 16), ("wdef", None)):
        calls: list[dict] = []
        d = _seam_defense(calls, batch_size=8)
        jsonl = bon_generate(d, goals, out_dir=tmp_path / label, n_max=n_max,
                             batch_size=bs)
        rows = _rows(jsonl)
        keys[label] = [(r["goal_id"], r["sample_idx"]) for r in rows]
        variants[label] = {(r["goal_id"], r["sample_idx"]): r["variant"]
                           for r in rows}
    # same (goal, sample) rows in the same goal-major, sample-order flat order
    expected = [(f"g{i}", j) for i in range(3) for j in range(n_max)]
    assert keys["w1"] == expected
    assert keys["w4"] == expected
    assert keys["w16"] == expected
    assert keys["wdef"] == expected
    # the AUGMENTATION for a given (goal, sample) is identical across widths —
    # width changes decode numerics, never the variant it decodes
    assert variants["w1"] == variants["w4"] == variants["w16"] == variants["wdef"]
    # and the augmentation is exactly what the unbatched contract computes
    for (gid, sidx), var in variants["w16"].items():
        g = next(g for g in goals if g.goal_id == gid)
        import random
        assert var == augment(g.text, 0.4, random.Random(bon_seed(gid, sidx, 0)))


def test_per_goal_grouping_reassembles_in_sample_order(tmp_path):
    goals = _goals(2)
    calls: list[dict] = []
    d = _seam_defense(calls)
    jsonl = bon_generate(d, goals, out_dir=tmp_path / "r", n_max=5, batch_size=4)
    grouped = load_bon_responses(jsonl)
    assert set(grouped) == {"g0", "g1"}
    assert all(len(v) == 5 for v in grouped.values())


# -- fixed width + padding ----------------------------------------------------


def test_fixed_width_and_ragged_final_chunk_is_padded(tmp_path):
    # 2 goals x 3 = 6 work units at width 4 -> chunks [4, 4]; the second is
    # padded from 2 real to 4. Every GPU call is EXACTLY width 4 (one shape).
    goals = _goals(2)
    calls: list[dict] = []
    d = _seam_defense(calls)
    jsonl = bon_generate(d, goals, out_dir=tmp_path / "r", n_max=3, batch_size=4)
    assert [c["n"] for c in calls] == [4, 4]          # fixed width, padded tail
    rows = _rows(jsonl)
    # exactly 6 REAL rows — the 2 pad slots never became rows
    assert len(rows) == 6
    assert [(r["goal_id"], r["sample_idx"]) for r in rows] == \
        [(f"g{i}", j) for i in range(2) for j in range(3)]
    assert all(r["bon_batch_size"] == 4 for r in rows)


def test_pad_forwards_not_counted(tmp_path):
    # 2 goals x 3 = 6 real variants; width 4 pads the tail with 2 units. The
    # JB7 forward counter must read 6 (real), not 8 (padded).
    goals = _goals(2)
    calls: list[dict] = []
    d = _seam_defense(calls)
    bon_generate(d, goals, out_dir=tmp_path / "r", n_max=3, batch_size=4)
    assert d.n_target_forwards == 6


def test_no_padding_when_width_divides_work(tmp_path):
    # 2 goals x 8 = 16 at width 8 -> [8, 8], no ragged chunk, no pad.
    goals = _goals(2)
    calls: list[dict] = []
    d = _seam_defense(calls)
    bon_generate(d, goals, out_dir=tmp_path / "r", n_max=8, batch_size=8)
    assert [c["n"] for c in calls] == [8, 8]
    assert d.n_target_forwards == 16


def test_goal_spanning_multiple_chunks(tmp_path):
    # 1 goal x 10 variants at width 4 -> chunks [4, 4, 2->pad4]: one goal's
    # variants span three chunks and must still reassemble 0..9 in order.
    goals = _goals(1)
    calls: list[dict] = []
    d = _seam_defense(calls)
    jsonl = bon_generate(d, goals, out_dir=tmp_path / "r", n_max=10, batch_size=4)
    assert [c["n"] for c in calls] == [4, 4, 4]        # last padded 2->4
    grouped = load_bon_responses(jsonl)
    assert len(grouped["g0"]) == 10
    rows = _rows(jsonl)
    assert [r["sample_idx"] for r in rows] == list(range(10))
    assert d.n_target_forwards == 10                   # 2 pad forwards uncounted


# -- resume -------------------------------------------------------------------


def test_resume_no_duplicates_no_gaps(tmp_path):
    goals = _goals(3)
    out = tmp_path / "r"
    # first pass: only the first goal's variants (simulate an interrupt after g0)
    d1 = _seam_defense([])
    bon_generate(d1, goals[:1], out_dir=out, n_max=6, batch_size=4)
    first = _rows(out / "bon_responses.jsonl")
    assert len(first) == 6
    # resume over ALL goals: g0 already covered, only g1+g2 regenerate
    calls2: list[dict] = []
    d2 = _seam_defense(calls2)
    bon_generate(d2, goals, out_dir=out, n_max=6, batch_size=4)
    rows = _rows(out / "bon_responses.jsonl")
    keys = [(r["goal_id"], r["sample_idx"]) for r in rows]
    # complete set, no duplicates, no gaps
    assert sorted(keys) == sorted((f"g{i}", j) for i in range(3) for j in range(6))
    assert len(keys) == len(set(keys)) == 18
    # only the 12 uncovered units were regenerated (g0 skipped)
    assert d2.n_target_forwards == 12


def test_resume_all_covered_is_noop(tmp_path):
    goals = _goals(2)
    out = tmp_path / "r"
    bon_generate(_seam_defense([]), goals, out_dir=out, n_max=4, batch_size=4)
    calls2: list[dict] = []
    d2 = _seam_defense(calls2)
    bon_generate(d2, goals, out_dir=out, n_max=4, batch_size=4)
    assert calls2 == [] and d2.n_target_forwards == 0


# -- the DEFAULT width is 64; explicit width 1 is the legacy path -------------


def test_default_width_is_64(tmp_path):
    # No batch_size -> DEFAULT_BON_BATCH_SIZE (64): one fixed width-64 call
    # (6 real units + 58 pads), the defense PINNED to 64 for the leg and
    # RESTORED after, per-CHUNK seeding, and width 64 recorded on every row.
    # NOTE the default change (1 -> 64) changes WHICH samples are drawn:
    # width-1 cells do not reproduce under the default — see bon_generate.
    assert DEFAULT_BON_BATCH_SIZE == 64
    goals = _goals(2)
    calls: list[dict] = []
    holder: dict = {}

    def seam(rendered, *, sampling_seed=None, **kw):
        calls.append({"n": len(rendered), "seed": sampling_seed,
                      "pinned_bs": holder["d"].batch_size})
        return [f"resp[{sampling_seed}]:{r[-24:]}" for r in rendered]

    d = LLMHFSteered(model=None, tok=None, mode="none", batch_size=8,
                     generate_fn=seam)
    holder["d"] = d
    jsonl = bon_generate(d, goals, out_dir=tmp_path / "r", n_max=3)
    assert [c["n"] for c in calls] == [64]            # ONE fixed-width chunk
    # pinned to the default width DURING the leg, restored after (finally)
    assert calls[0]["pinned_bs"] == 64 and d.batch_size == 8
    # batched seeding: the chunk's key-hash seed, NOT any variant's bon_seed
    chunk = [(f"g{i}", j, None) for i in range(2) for j in range(3)]
    assert calls[0]["seed"] == _chunk_sampling_seed(chunk, 0)
    rows = _rows(jsonl)
    assert len(rows) == 6                             # pads never became rows
    assert [(r["goal_id"], r["sample_idx"]) for r in rows] == \
        [(f"g{i}", j) for i in range(2) for j in range(3)]
    assert all(r["bon_batch_size"] == 64 for r in rows)
    assert d.n_target_forwards == 6                   # 58 pad forwards uncounted


def test_explicit_width1_is_legacy_path(tmp_path):
    # batch_size=1 (EXPLICIT — 64 is the default now): one prompt per call,
    # each seeded by its OWN bon_seed (byte-identical to the pre-batching
    # loop), no padding, no defense re-pin.
    goals = _goals(2)
    calls: list[dict] = []
    d = _seam_defense(calls)
    bon_generate(d, goals, out_dir=tmp_path / "r", n_max=3, batch_size=1)
    assert [c["n"] for c in calls] == [1, 1, 1, 1, 1, 1]      # one at a time
    # each call's sampling seed is the variant's own bon_seed
    expected_seeds = [bon_seed(f"g{i}", j, 0) for i in range(2) for j in range(3)]
    assert [c["seed"] for c in calls] == expected_seeds
    assert d.n_target_forwards == 6
    rows = _rows(tmp_path / "r" / "bon_responses.jsonl")
    assert all(r["bon_batch_size"] == 1 for r in rows)


# -- the real generation path: one tensor, not a per-item loop ---------------


def test_batch_reaches_model_generate_as_one_tensor(tiny_llama, tmp_path):
    """The chunk must arrive at model.generate as ONE [B, S] tensor, not a
    per-item loop (which would make batching a no-op that only adds overhead).
    Instruments the REAL path: bon_generate -> defense.generate ->
    generate_with_object -> generate_batch -> model.generate."""
    model, tok = tiny_llama
    seen = {"batch_dims": []}
    orig = model.generate

    def spy(*a, **kw):
        ids = kw.get("input_ids")
        if ids is None and a:
            ids = a[0]
        seen["batch_dims"].append(int(ids.shape[0]))
        return orig(*a, **kw)

    model.generate = spy
    try:
        d = build_defense(model, tok, mode="none", positions="all",
                          batch_size=8)
        bon_generate(d, _goals(2), out_dir=tmp_path / "r", n_max=4,
                     batch_size=8, max_new_tokens=4)
    finally:
        model.generate = orig
    # 2 goals x 4 = 8 units at width 8 -> ONE call of batch dim 8 (not 8 of 1)
    assert seen["batch_dims"] == [8]


def test_batched_vs_unbatched_structural_equivalence(tiny_llama, tmp_path):
    """The equivalence bar: NOT token identity (bf16 batching numerics differ; on
    fp32 CPU they usually match) — same rows, same order, same coverage, same
    VARIANTS, with a reported per-item divergence rate so the batching effect is
    measured, not discovered."""
    model, tok = tiny_llama
    goals = _goals(3)
    out = {}
    for label, bs in (("unbatched", 1), ("batched", 4)):
        d = build_defense(model, tok, mode="none", positions="all", batch_size=8)
        jsonl = bon_generate(d, goals, out_dir=tmp_path / label, n_max=4,
                             batch_size=bs, max_new_tokens=4)
        out[label] = _rows(jsonl)
    ku = [(r["goal_id"], r["sample_idx"]) for r in out["unbatched"]]
    kb = [(r["goal_id"], r["sample_idx"]) for r in out["batched"]]
    assert ku == kb                                   # same items, same order
    vu = {(r["goal_id"], r["sample_idx"]): r["variant"] for r in out["unbatched"]}
    vb = {(r["goal_id"], r["sample_idx"]): r["variant"] for r in out["batched"]}
    assert vu == vb                                   # same augmentations decoded
    n = len(ku)
    diverged = sum(1 for a, b in zip(out["unbatched"], out["batched"])
                   if a["response"] != b["response"])
    print(f"[bon-equivalence] per-item response divergence: {diverged}/{n}")
