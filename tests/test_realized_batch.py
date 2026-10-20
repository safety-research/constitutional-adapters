"""The REALIZED effective-batch vector.

``fit_record.realized_eff_batch`` is a LIST (one entry per optimizer step
of the first epoch completed in the process). The trainer's grouping is
exact and seeded: micro-batches are the length-sorted rows chunked by
``batch_size`` (the short chunk LAST), each epoch's order is shuffled by
``random.Random(seed)`` with no draw before the first shuffle, and groups
of ``accum_steps`` are taken along that order. So the micro-batch carrying
the row deficit lands in ANY group — a rule expecting it last would reject
a valid run — and the vector is exactly reproducible from
``(n_docs, bs, accum, seed)`` when the run was not resumed.

Pinned here:
  (a) the trainer's own loop (tiny model, 7 rows, bs 2, accum 2) records
      exactly ``group_row_counts`` of its own shuffled order — the helper
      is driven THROUGH the trainer, not re-derived beside it;
  (b) (17,073, 2, 32, seed 0): 267 groups, sum 17,073, exactly one 63,
      last in {50, 49}, validates; the seed-0 position of the 63 is named;
  (c) (17,352, 1, 64) -> 271 x 64 + 8 and (4,102, 1, 64) -> 64 x 64 + 6;
  (d) the one-document gate cell -> [1] with steps_per_epoch 1;
  (e) a PERMUTED valid vector passes the semantic floor, fails the exact
      check only when unresumed, and passes outright when resumed;
  (f) a dropped row, two deficits, a wrong count, a wrong steps_per_epoch
      each refuse — with every problem named, never an exception;
  and ``train_object_ce``'s ``seed`` default is 0 (the trainer default the
  validator assumes when a config declares none).
"""

from __future__ import annotations

import inspect
import math
import random

import pytest

from sisteer.extraction import realized_batch as rb
from sisteer.extraction.objects import (group_row_counts,
                                        shuffled_epoch_order, train_object_ce)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def test_micro_rows_and_allowed_sizes():
    assert rb.micro_rows(7, 2) == [2, 2, 2, 1]
    assert rb.micro_rows(8, 2) == [2, 2, 2, 2]
    assert rb.micro_rows(1, 1) == [1]
    assert rb.micro_rows(5, 8) == [5]
    with pytest.raises(ValueError):
        rb.micro_rows(0, 2)
    s = rb.allowed_group_sizes(17_073, 2, 32)
    assert (s["n_micro"], s["G"], s["m_last"], s["r"], s["deficit"]) == \
        (8_537, 267, 25, 1, 1)
    assert s["full"] == {64, 63} and s["last"] == {50, 49}
    s = rb.allowed_group_sizes(17_352, 1, 64)
    assert (s["G"], s["m_last"], s["r"]) == (272, 8, 0)
    assert s["full"] == {64} and s["last"] == {8}
    with pytest.raises(ValueError):
        rb.allowed_group_sizes(10, 1, 0)


def test_shuffled_order_is_a_pure_function_of_seed_and_count():
    a = shuffled_epoch_order(random.Random(0), 100)
    b = shuffled_epoch_order(random.Random(0), 100)
    c = shuffled_epoch_order(random.Random(1), 100)
    assert a == b and sorted(a) == list(range(100)) and a != c
    # the trainer's own two lines: range(n) shuffled in place
    rng = random.Random(0)
    ref = list(range(100))
    rng.shuffle(ref)
    assert a == ref


def test_group_row_counts_groups_along_the_order_and_honours_skip():
    rows = [2, 2, 2, 1]
    assert group_row_counts(rows, [0, 1, 2, 3], 2) == [4, 3]
    assert group_row_counts(rows, [3, 0, 1, 2], 2) == [3, 4]   # deficit FIRST
    assert group_row_counts(rows, [0, 3, 1, 2], 3) == [5, 2]   # ragged last
    # a resumed epoch's suffix: skip whole groups already done
    full = group_row_counts(rows, [0, 1, 2, 3], 2)
    assert group_row_counts(rows, [0, 1, 2, 3], 2, skip=2) == full[1:]
    assert group_row_counts(rows, [0, 1, 2, 3], 2, skip=4) == []


# --------------------------------------------------------------------------
# (a) driven THROUGH the trainer
# --------------------------------------------------------------------------

_DOCS = [
    ("Sys one\n\n", "A short doc."),
    ("", "A much longer story body that runs on for quite a few more tokens "
         "than the short one does, deliberately, on purpose."),
    ("Sys two\n\n", "Medium length advice body follows here for the test."),
    ("", "Tiny."),
    ("Sys three\n\n", "One more medium advice body for good measure here."),
    ("", "Final story text, middling length overall, nothing more."),
    ("Sys four\n\n", "Seventh document so the chunking is ragged at bs 2."),
]


def _examples(tok, docs):
    from sisteer.data.pairs import Pair, docs_to_ce_examples

    return docs_to_ce_examples(tok, [
        Pair(prompt=p, pos=pos, neg="", instruction_id="constv2-mix-1k")
        for p, pos in docs])


@pytest.mark.parametrize("seed", [0, 3])
def test_trainer_records_exactly_group_row_counts_of_its_own_order(
        tiny_llama, seed):
    from sisteer.extraction.psr import PSRObject

    model, tok = tiny_llama
    obj = PSRObject(model, [0, 1], train_direction=True)
    diag: dict = {}
    train_object_ce(model, tok, obj, _examples(tok, _DOCS), lr=1e-3,
                    epochs=1, batch_size=2, max_len=128, accum_steps=2,
                    seed=seed, heldout=_examples(tok, _DOCS[:2]),
                    diagnostics=diag)
    vec = diag["realized_eff_batch"]
    assert diag["steps_per_epoch"] == 2 == math.ceil(math.ceil(7 / 2) / 2)
    assert diag["resumed_from_step"] is None
    assert diag["optimizer_steps_run"] == 2
    # 7 docs at bs 2: micro rows (2, 2, 2, 1); accum 2 -> two groups whose
    # row counts depend on WHERE the shuffle put the 1-row chunk
    expected = group_row_counts(rb.micro_rows(7, 2),
                                shuffled_epoch_order(random.Random(seed), 4), 2)
    assert vec == expected == rb.expected_realized_vector(7, 2, 2, seed)
    assert sorted(vec) == [3, 4] and sum(vec) == 7
    v = rb.validate_realized_vector(vec, n_docs=7, bs=2, accum=2, seed=seed,
                                    steps_per_epoch=diag["steps_per_epoch"],
                                    resumed_from_step=None)
    assert v.ok and v.exact_checked and v.exact_ok, v.problems
    assert v.deficit_positions == [vec.index(3)]


def test_train_object_ce_seed_defaults_to_zero():
    assert inspect.signature(train_object_ce).parameters["seed"].default == 0


# --------------------------------------------------------------------------
# (b) the ctl2-full shape at bs 2 / accum 32
# --------------------------------------------------------------------------

def test_ctl2_full_at_bs2_accum32_puts_the_63_anywhere_and_validates():
    vec = rb.expected_realized_vector(17_073, 2, 32, seed=0)
    assert len(vec) == 267
    assert sum(vec) == 17_073
    assert vec.count(63) == 1
    assert vec[-1] in (50, 49)
    pos = vec.index(63)
    # the reviewer's read-only reproduction placed the 63 at group 129
    # (0-based) with a 50 last; if the trainer's seed or sort ever changes,
    # this line says where the helper now puts it instead of assuming.
    assert pos == 129, f"seed-0 order puts the 63-row group at {pos}, last {vec[-1]}"
    assert vec[-1] == 50
    assert set(vec[:-1]) <= {64, 63}
    v = rb.validate_realized_vector(vec, n_docs=17_073, bs=2, accum=32,
                                    seed=0, steps_per_epoch=267,
                                    resumed_from_step=None)
    assert v.ok, v.problems
    assert v.exact_checked and v.exact_ok and v.expected == vec
    assert v.deficit_positions == [pos]
    assert (v.G, v.m_last) == (267, 25)
    d = v.as_dict()
    assert d["ok"] is True and d["n_docs"] == 17_073


# --------------------------------------------------------------------------
# (c), (d) the other production shapes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_docs,bs,accum,shape", [
    (17_352, 1, 64, [64] * 271 + [8]),      # mix-full at bs 1
    (4_102, 1, 64, [64] * 64 + [6]),        # the 10M search reference cell
    (17_073, 1, 64, [64] * 266 + [49]),     # ctl2-full at bs 1
    (1, 1, 64, [1]),                        # the one-document gate cell
])
def test_divisible_shapes_are_order_independent(n_docs, bs, accum, shape):
    # bs 1 -> no row deficit, so every order gives the same vector
    for seed in (0, 1, 7):
        assert rb.expected_realized_vector(n_docs, bs, accum, seed) == shape
    v = rb.validate_realized_vector(shape, n_docs=n_docs, bs=bs, accum=accum,
                                    seed=0, steps_per_epoch=len(shape),
                                    resumed_from_step=None)
    assert v.ok, v.problems
    assert v.deficit_positions == []


def test_the_gate_cell_is_one_row_one_step_not_sixty_four():
    v = rb.validate_realized_vector([1], n_docs=1, bs=1, accum=64, seed=0,
                                    steps_per_epoch=1, resumed_from_step=None)
    assert v.ok and v.G == 1 and v.m_last == 1
    bad = rb.validate_realized_vector([64], n_docs=1, bs=1, accum=64, seed=0,
                                      steps_per_epoch=1, resumed_from_step=None)
    assert not bad.ok
    assert any("sum 64 != n_docs 1" in p for p in bad.problems)


# --------------------------------------------------------------------------
# (e) permuted: floor passes; exact fails only when unresumed
# --------------------------------------------------------------------------

def test_permuted_vector_passes_the_floor_and_fails_only_the_exact_check():
    vec = rb.expected_realized_vector(17_073, 2, 32, seed=0)
    pos = vec.index(63)
    perm = list(vec)
    perm[pos], perm[5] = perm[5], perm[pos]      # the 63 moves to group 5
    assert perm != vec and sum(perm) == 17_073 and perm[-1] == 50
    fresh = rb.validate_realized_vector(perm, n_docs=17_073, bs=2, accum=32,
                                        seed=0, steps_per_epoch=267,
                                        resumed_from_step=None)
    assert not fresh.ok
    assert fresh.exact_checked and fresh.exact_ok is False
    assert fresh.deficit_positions == [5]
    assert [p for p in fresh.problems if "not the seed-0 reproduction" in p]
    assert len(fresh.problems) == 1                   # the floor itself passed
    assert "first difference at group 5" in fresh.problems[0]
    resumed = rb.validate_realized_vector(perm, n_docs=17_073, bs=2, accum=32,
                                          seed=0, steps_per_epoch=267,
                                          resumed_from_step=4_000)
    assert resumed.ok and not resumed.exact_checked and resumed.exact_ok is None
    # a resumed process records the SUFFIX of its epoch: shorter, smaller sum
    suffix = vec[100:]
    v = rb.validate_realized_vector(suffix, n_docs=17_073, bs=2, accum=32,
                                    seed=0, steps_per_epoch=267,
                                    resumed_from_step=100)
    assert v.ok, v.problems
    # ...but the same suffix claiming to be a fresh epoch refuses on count+sum
    v = rb.validate_realized_vector(suffix, n_docs=17_073, bs=2, accum=32,
                                    seed=0, steps_per_epoch=267,
                                    resumed_from_step=None)
    assert not v.ok
    assert any("len 167 != G 267" in p for p in v.problems)
    assert any("sum" in p and "!= n_docs" in p for p in v.problems)


def test_the_deficit_may_land_last_and_a_different_seed_moves_it():
    vec = rb.expected_realized_vector(17_073, 2, 32, seed=0)
    last_deficit = list(vec)
    p = last_deficit.index(63)
    last_deficit[p], last_deficit[-1] = 64, 49       # 63 -> 64, 50 -> 49
    assert sum(last_deficit) == 17_073
    v = rb.validate_realized_vector(last_deficit, n_docs=17_073, bs=2,
                                    accum=32, seed=0, steps_per_epoch=267,
                                    resumed_from_step=7)
    assert v.ok, v.problems
    assert v.deficit_positions == [266]
    # some other seed places it elsewhere: the vector is seed-specific
    others = {rb.expected_realized_vector(17_073, 2, 32, s).index(63)
              for s in range(1, 6)}
    assert others != {p}


# --------------------------------------------------------------------------
# (f) refusals, every problem named
# --------------------------------------------------------------------------

def _validate(vec, **over):
    kw = dict(n_docs=17_073, bs=2, accum=32, seed=0, steps_per_epoch=267,
              resumed_from_step=None)
    kw.update(over)
    return rb.validate_realized_vector(vec, **kw)


def test_refusals_name_every_problem():
    vec = rb.expected_realized_vector(17_073, 2, 32, seed=0)
    pos = vec.index(63)

    dropped = list(vec)                      # one row silently dropped
    dropped[10] = 63
    v = _validate(dropped)
    assert not v.ok
    assert any("sum 17072 != n_docs 17073" in p for p in v.problems)
    assert any("at most ONE group may carry it" in p for p in v.problems)

    two = list(vec)                          # the 63 AND a 49
    two[-1] = 49
    v = _validate(two)
    assert not v.ok
    assert any("appears in 2 groups" in p for p in v.problems)
    assert v.deficit_positions == [pos, 266]

    short = vec[:-1]                         # wrong group count
    v = _validate(short)
    assert not v.ok and any("len 266 != G 267" in p for p in v.problems)

    v = _validate(vec, steps_per_epoch=266)  # the record's own count is off
    assert not v.ok
    assert any("steps_per_epoch 266 != ceil" in p for p in v.problems)

    odd = list(vec)                          # an impossible group size
    odd[3] = 61
    v = _validate(odd)
    assert not v.ok
    assert any("outside the allowed sizes [63, 64]" in p for p in v.problems)

    tail = list(vec)                         # a full-size last group, unresumed
    tail[-1] = 64
    v = _validate(tail)
    assert not v.ok
    assert any("is not a last-group size [49, 50]" in p for p in v.problems)

    no_deficit = [64] * 266 + [50]           # sum 17,074: no group short a row
    v = _validate(no_deficit)
    assert not v.ok
    assert any("in exactly one group; found 0" in p for p in v.problems)

    v = _validate([])
    assert not v.ok and any("empty" in p for p in v.problems)

    # the verdict never raises on a bad vector: every problem is listed
    assert all(isinstance(p, str) for p in v.problems)


# --------------------------------------------------------------------------
# (g) the fit RECORD: absent provenance is not a fresh run
# --------------------------------------------------------------------------

def test_validate_fit_record_distinguishes_absent_provenance_from_fresh():
    vec = rb.expected_realized_vector(17_073, 2, 32, seed=0)
    kw = {"n_docs": 17_073, "bs": 2, "accum": 32, "seed": 0}
    # an explicit fresh-run marker: the exact reproduction applies
    fresh = {"realized_eff_batch": vec, "steps_per_epoch": 267,
             "resumed_from_step": None}
    v = rb.validate_fit_record(fresh, **kw)
    assert v.ok and v.exact_checked and v.exact_ok, v.problems
    assert v.resume_provenance == "fresh"
    # a resumed process: the suffix, floor only
    resumed = {"realized_eff_batch": vec[100:], "steps_per_epoch": 267,
               "resumed_from_step": 100}
    v = rb.validate_fit_record(resumed, **kw)
    assert v.ok and not v.exact_checked, v.problems
    assert v.resume_provenance == "resumed"
    # NO marker at all: unknown provenance — a healthy resumed suffix must
    # not be judged as a fresh epoch (length/sum/reproduction refusals),
    # and a fresh vector must not pass on an assumption; refused by name
    absent = {"realized_eff_batch": vec[100:], "steps_per_epoch": 267}
    v = rb.validate_fit_record(absent, **kw)
    assert not v.ok
    assert v.resume_provenance == "absent"
    assert any("resumed_from_step" in p for p in v.problems)
    assert not any("len 167" in p for p in v.problems)   # no fresh-run verdicts
    v = rb.validate_fit_record({"realized_eff_batch": vec,
                                "steps_per_epoch": 267}, **kw)
    assert not v.ok and v.resume_provenance == "absent"
    # no vector at all
    v = rb.validate_fit_record({"resumed_from_step": None}, **kw)
    assert not v.ok and any("realized_eff_batch" in p for p in v.problems)
    # the verdict serialises with the provenance
    assert rb.validate_fit_record(fresh, **kw).as_dict()["resume_provenance"] == "fresh"
