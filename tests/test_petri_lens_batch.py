"""lens_batch: flattened-forward segmentation, per-request HookContext
construction, and the post-run prefill ledger — CPU-only, synthetic
metadata (invented shapes and token ids, no GPU, no vLLM).

The load-bearing tests are the two ledger refusals: a preemption-
recompute (in flight, including the length-collision variant where the
recompute consumes another prompt's credit and that prompt's own
prefill later over-draws) and the 1-token final chunk of a chunked
prefill (finish time) — both are bit-for-bit legal shapes to every
per-forward check, so a silent pass here means wrong numbers that look
right.
"""

import pytest
import torch

from sisteer.capeval.serving import lens_adapters as la
from sisteer.capeval.serving import lens_batch as lb

HIDDEN = 8

# The canonical mixed batch: two complete prefills interleaved with two
# decode steps, as (q, computed) per segment.
MIXED = ((5, 0), (1, 3), (7, 0), (1, 19))


def _metadata(segs):
    """(query_start_loc, seq_lens, num_actual_tokens) for (q, computed)
    segment shapes."""
    qsl = [0]
    seq_lens = []
    for q, computed in segs:
        qsl.append(qsl[-1] + q)
        seq_lens.append(computed + q)
    return qsl, seq_lens, qsl[-1]


def _plan(segs=MIXED):
    return lb.derive_batch_plan(*_metadata(segs))


def _positions(segs=MIXED):
    out = []
    for q, computed in segs:
        out.extend(range(computed, computed + q))
    return torch.tensor(out, dtype=torch.long)


def _row_local_op():
    return la.HookOp(op=la.OP_ADD_VECTOR, layer=0,
                     vector=torch.ones(HIDDEN),
                     positions=la.POSITIONS_ALL, skip_bos=False,
                     phase=la.PHASE_BOTH)


# --------------------------------------------------------------------------
# Part A — derive_batch_plan: the happy path and the lattice
# --------------------------------------------------------------------------


@pytest.mark.parametrize("as_tensor", [False, True],
                         ids=["lists", "tensors"])
def test_mixed_batch_happy_path(as_tensor):
    qsl, seq_lens, nat = _metadata(MIXED)
    if as_tensor:
        qsl = torch.tensor(qsl, dtype=torch.int32)
        seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        nat = torch.tensor(nat)
    plan = lb.derive_batch_plan(qsl, seq_lens, nat)

    assert plan.num_tokens == 14
    assert len(plan.segments) == 4
    assert plan.n_decode == 2
    assert plan.full_prefills == (plan.segments[0], plan.segments[2])

    s0, s1, s2, s3 = plan.segments
    assert (s0.start, s0.end, s0.q, s0.computed, s0.seq_len) == (0, 5, 5, 0, 5)
    assert (s1.start, s1.end, s1.q, s1.computed, s1.seq_len) == (5, 6, 1, 3, 4)
    assert (s2.start, s2.end, s2.q, s2.computed, s2.seq_len) == (6, 13, 7, 0, 7)
    assert (s3.start, s3.end, s3.q, s3.computed, s3.seq_len) == (13, 14, 1,
                                                                 19, 20)
    assert s0.is_prefill and s2.is_prefill
    assert not s1.is_prefill and not s3.is_prefill
    assert s2.rows == slice(6, 13)


def test_lattice_refuses_chunked_prefill_continuation_or_prefix_cache_hit():
    # q=3 with 2 tokens already computed: the continuation of a chunked
    # prefill — indistinguishable from a prefix-cache hit; both refuse.
    with pytest.raises(lb.LensBatchError,
                       match="chunked-prefill continuation or a "
                             "prefix-cache hit"):
        lb.derive_batch_plan([0, 3], [5], 3)


def test_lattice_refuses_one_token_prefill():
    with pytest.raises(lb.LensBatchError,
                       match="1-token prefill.*[Rr]efusing to guess"):
        lb.derive_batch_plan([0, 1], [1], 1)


@pytest.mark.parametrize(
    "qsl, seq_lens, nat, match",
    [
        pytest.param([], [], 0, "no requests", id="empty-qsl"),
        pytest.param([0], [], 0, "no requests", id="one-entry-qsl"),
        pytest.param([0, 5], [5, 4], 5, "metadata mismatch",
                     id="seq-lens-count"),
        pytest.param([1, 6], [5], 6, "must start at 0", id="nonzero-start"),
        pytest.param([0, 5], [5], 6, "num_actual_tokens",
                     id="padded-or-truncated"),
        pytest.param([0, 3, 3], [3, 1], 3, "non-increasing",
                     id="zero-length-segment"),
        pytest.param([0, 3, 2], [3, 3], 2, "non-increasing",
                     id="decreasing-qsl"),
        pytest.param([0, 5], [3], 5, "seq_len 3 < query length 5",
                     id="negative-computed"),
    ])
def test_metadata_shape_refusals(qsl, seq_lens, nat, match):
    with pytest.raises(lb.LensBatchError, match=match):
        lb.derive_batch_plan(qsl, seq_lens, nat)


@pytest.mark.parametrize(
    "qsl, seq_lens, nat, match",
    [
        pytest.param(None, [5], 5, "query_start_loc is required",
                     id="none-qsl"),
        pytest.param(torch.zeros((2, 2), dtype=torch.long), [5], 5,
                     "must be a 1-D tensor", id="2d-qsl"),
        pytest.param(torch.tensor([0.0, 5.0]), [5], 5, "integer tensor",
                     id="float-qsl"),
        pytest.param([0, True], [1], 1, "bool", id="bool-element"),
        pytest.param([0, 5], [5], 5.0, "must be an int", id="float-nat"),
        pytest.param("05", [5], 5, "list/tuple", id="string-qsl"),
    ])
def test_metadata_type_refusals(qsl, seq_lens, nat, match):
    with pytest.raises(lb.LensBatchError, match=match):
        lb.derive_batch_plan(qsl, seq_lens, nat)


# --------------------------------------------------------------------------
# Part A — the positions cross-check (REQUIRED, loud)
# --------------------------------------------------------------------------


def test_cross_check_passes_on_agreeing_metadata():
    assert lb.cross_check_positions(_plan(), _positions()) is None


def test_cross_check_accepts_a_plain_list():
    assert lb.cross_check_positions(_plan(),
                                    _positions().tolist()) is None


def test_cross_check_fires_on_segment_start_mismatch():
    pos = _positions()
    pos[5] = 2  # the decode row claims position 2; metadata says 3
    with pytest.raises(lb.LensBatchError,
                       match="segment 1 row 5 sits at position 2, "
                             "expected 3"):
        lb.cross_check_positions(_plan(), pos)


def test_cross_check_fires_on_interior_prefill_row():
    # The deliberate strengthening over the original backend's first-row-only check:
    # an interior row of a prefill segment out of place also refuses.
    pos = _positions()
    pos[8] = 7  # segment 2's row at offset 2 should be position 2
    with pytest.raises(lb.LensBatchError,
                       match="segment 2 row 8 sits at position 7"):
        lb.cross_check_positions(_plan(), pos)


def test_cross_check_refuses_missing_positions():
    with pytest.raises(lb.LensBatchError,
                       match="positions is required and got None"):
        lb.cross_check_positions(_plan(), None)


def test_cross_check_refuses_wrong_rank():
    with pytest.raises(lb.LensBatchError, match="must be a 1-D tensor"):
        lb.cross_check_positions(_plan(),
                                 _positions().reshape(2, 7))


def test_cross_check_refuses_float_positions():
    with pytest.raises(lb.LensBatchError, match="integer tensor"):
        lb.cross_check_positions(_plan(), _positions().float())


def test_cross_check_refuses_too_few_rows():
    with pytest.raises(lb.LensBatchError, match="13 rows.*14 actual"):
        lb.cross_check_positions(_plan(), _positions()[:-1])


def test_cross_check_tolerates_cudagraph_padding_rows():
    padded = torch.cat([_positions(),
                        torch.zeros(3, dtype=torch.long)])
    assert lb.cross_check_positions(_plan(), padded) is None


# --------------------------------------------------------------------------
# Part B — per-request HookContext construction
# --------------------------------------------------------------------------


def test_contexts_cover_the_segments_in_order():
    scs = lb.build_segment_contexts(_plan(), _positions())
    assert len(scs) == 4
    assert [sc.segment for sc in scs] == list(_plan().segments)
    assert [sc.rows for sc in scs] == [slice(0, 5), slice(5, 6),
                                       slice(6, 13), slice(13, 14)]

    # complete prefills: phase="prefill", positions 0..q-1
    assert scs[0].ctx.phase == la.PHASE_PREFILL
    assert scs[0].ctx.positions == (0, 1, 2, 3, 4)
    assert scs[2].ctx.phase == la.PHASE_PREFILL
    assert scs[2].ctx.positions == (0, 1, 2, 3, 4, 5, 6)

    # decode steps: phase="decode", exactly ONE absolute position
    assert scs[1].ctx.phase == la.PHASE_DECODE
    assert scs[1].ctx.positions == (3,)
    assert scs[3].ctx.phase == la.PHASE_DECODE
    assert scs[3].ctx.positions == (19,)

    # no token-ids tensor supplied -> None on every context
    assert all(sc.ctx.token_ids is None for sc in scs)


def test_contexts_slice_token_ids_like_the_rows():
    ids = torch.arange(1000, 1014, dtype=torch.long)  # invented ids
    scs = lb.build_segment_contexts(_plan(), _positions(), token_ids=ids)
    assert scs[0].ctx.token_ids == tuple(range(1000, 1005))
    assert scs[1].ctx.token_ids == (1005,)
    assert scs[2].ctx.token_ids == tuple(range(1006, 1013))
    assert scs[3].ctx.token_ids == (1013,)
    for sc in scs:
        assert len(sc.ctx.token_ids) == len(sc.ctx.positions)


def test_contexts_accept_token_ids_as_a_list():
    ids = list(range(2000, 2014))
    scs = lb.build_segment_contexts(_plan(), _positions(), token_ids=ids)
    assert scs[2].ctx.token_ids == tuple(range(2006, 2013))


def test_contexts_refuse_malformed_token_ids_instead_of_degrading():
    with pytest.raises(lb.LensBatchError, match="must be a 1-D tensor"):
        lb.build_segment_contexts(
            _plan(), _positions(),
            token_ids=torch.zeros((2, 7), dtype=torch.long))
    with pytest.raises(lb.LensBatchError,
                       match="13 rows.*never degraded to token_ids=None"):
        lb.build_segment_contexts(_plan(), _positions(),
                                  token_ids=list(range(13)))


def test_contexts_run_the_positions_cross_check():
    pos = _positions()
    pos[0] = 9  # first prefill row off its absolute position
    with pytest.raises(lb.LensBatchError,
                       match="disagrees with attention metadata"):
        lb.build_segment_contexts(_plan(), pos)


def test_contexts_refuse_missing_positions():
    with pytest.raises(lb.LensBatchError,
                       match="positions is required and got None"):
        lb.build_segment_contexts(_plan(), None)


# --------------------------------------------------------------------------
# Part C — the prefill ledger
# --------------------------------------------------------------------------


def test_ledger_clean_run_leaves_nothing_and_disarms():
    ledger = lb.PrefillLedger()
    assert not ledger.armed
    ledger.begin_run([5, 7])
    assert ledger.armed
    # forward 1: both complete prefills; forward 2: their decode steps
    ledger.observe(_plan(((5, 0), (7, 0))))
    ledger.observe(_plan(((1, 5), (1, 7))))
    ledger.finish_run()  # nothing leftover: passes
    assert not ledger.armed


def test_ledger_two_prompts_same_length_need_two_prefills():
    ledger = lb.PrefillLedger()
    ledger.begin_run([6, 6])
    ledger.observe(_plan(((6, 0),)))
    ledger.observe(_plan(((6, 0),)))
    ledger.finish_run()


def test_ledger_refuses_preemption_recompute_in_flight():
    # (b): a preempted request re-runs prompt+generated (5+3 tokens) as
    # one fresh prefill — structurally perfect, but no submitted prompt
    # has length 8, so it must refuse ON THAT FORWARD.
    ledger = lb.PrefillLedger()
    ledger.begin_run([5])
    with pytest.raises(lb.LensBatchError,
                       match=r"complete prefill of 8 tokens.*preemption "
                             r"recompute.*DISCARDED"):
        ledger.observe(_plan(((8, 0),)))


def test_ledger_refuses_length_collision_at_the_over_draw():
    # (b) collision variant: prompts of length 5 and 9 submitted; a
    # recompute of some other request happens to be 5 tokens long and
    # silently consumes the length-5 credit — the REAL length-5 prefill
    # then over-draws and the refusal fires there.
    ledger = lb.PrefillLedger()
    ledger.begin_run([5, 9])
    ledger.observe(_plan(((5, 0),)))  # the recompute, indistinguishable
    with pytest.raises(lb.LensBatchError,
                       match=r"complete prefill of 5 tokens.*DISCARDED"):
        ledger.observe(_plan(((5, 0),)))  # the real prompt's prefill


def test_ledger_refuses_one_token_final_chunk_at_finish():
    # (a): the length-6 prompt was chunked 5+1 — its 5-token first chunk
    # drew the (colliding) length-5 credit and its 1-token remnant is
    # decode-shaped, so nothing fires in flight; finish_run catches the
    # never-consumed length-6 credit.
    ledger = lb.PrefillLedger()
    ledger.begin_run([5, 6])
    ledger.observe(_plan(((5, 0),)))  # actually prompt 6's first chunk
    ledger.observe(_plan(((1, 5),)))  # its remnant, decode-shaped
    with pytest.raises(lb.LensBatchError,
                       match="never observed as a complete "
                             "prefill.*DISCARDED") as exc:
        ledger.finish_run()
    # +Counter dropped the consumed length-5 entry: only 6 is owed.
    assert "{6: 1}" in str(exc.value)
    assert "5:" not in str(exc.value)
    assert not ledger.armed  # disarmed even on refusal


def test_ledger_refuses_observe_without_begin_run():
    with pytest.raises(lb.LensBatchError, match="no armed run"):
        lb.PrefillLedger().observe(_plan(((5, 0),)))


def test_ledger_refuses_finish_without_begin_run():
    with pytest.raises(lb.LensBatchError, match="no armed run"):
        lb.PrefillLedger().finish_run()


def test_ledger_refuses_double_finish():
    ledger = lb.PrefillLedger()
    ledger.begin_run([5])
    ledger.observe(_plan(((5, 0),)))
    ledger.finish_run()
    with pytest.raises(lb.LensBatchError, match="no armed run"):
        ledger.finish_run()


def test_ledger_refuses_rearming_an_armed_run():
    ledger = lb.PrefillLedger()
    ledger.begin_run([5])
    with pytest.raises(lb.LensBatchError, match="already armed"):
        ledger.begin_run([7])


def test_ledger_refuses_sub_2_token_prompt_at_arm_time():
    with pytest.raises(lb.LensBatchError,
                       match="length 1 < 2 can NEVER be observed"):
        lb.PrefillLedger().begin_run([5, 1])


def test_ledger_disarm_clears_without_checking():
    """The exception-path cleanup: an aborted run's leftovers are
    dropped WITHOUT the run-end refusal, so the next begin_run works —
    and disarm is a safe no-op when nothing is armed."""
    ledger = lb.PrefillLedger()
    ledger.disarm()                 # unarmed: no-op, no raise
    ledger.begin_run([5, 9])
    ledger.observe(_plan(((5, 0),)))
    ledger.disarm()                 # 9 still owed — dropped, not refused
    assert not ledger.armed
    ledger.begin_run([4])           # would be "already armed" without it
    ledger.disarm()


# --------------------------------------------------------------------------
# Part C' — the per-request prefill invariants (check_tracked_segment):
# the ALWAYS-ON defense, valid with no run lifecycle at all — what the
# serving path relies on, since HTTP requests overlap and never arm the
# ledger.
# --------------------------------------------------------------------------


def _seg(q, computed, start=0):
    return lb.Segment(start=start, end=start + q, q=q, computed=computed,
                      seq_len=computed + q)


def test_tracked_complete_prefill_matching_prompt_passes():
    lb.check_tracked_segment(_seg(5, 0), prompt_len=5, req_id="req-a")


def test_tracked_recompute_refuses():
    # (b): prompt N=5, g=3 generated, re-run as one fresh 8-token prefill.
    with pytest.raises(lb.LensBatchError,
                       match=r"complete prefill of 8 tokens.*preemption.*"
                             r"DISCARDED"):
        lb.check_tracked_segment(_seg(8, 0), prompt_len=5, req_id="req-a")


def test_tracked_first_chunk_refuses():
    # the lattice cannot tell a 5-token FIRST CHUNK of a 9-token prompt
    # from a legitimate 5-token complete prefill; the prompt length can.
    with pytest.raises(lb.LensBatchError, match="SHORTER"):
        lb.check_tracked_segment(_seg(5, 0), prompt_len=9, req_id="req-a")


def test_tracked_decode_at_or_past_prompt_end_passes():
    lb.check_tracked_segment(_seg(1, 5), prompt_len=5, req_id="req-a")
    lb.check_tracked_segment(_seg(1, 11), prompt_len=5, req_id="req-a")


def test_tracked_one_token_final_chunk_refuses():
    # (a): prompt N=5 chunked 4+1 — the remnant is decode-shaped at
    # computed == 4 < 5, i.e. a PROMPT row about to be steered as a
    # generated one.
    with pytest.raises(lb.LensBatchError, match="INSIDE"):
        lb.check_tracked_segment(_seg(1, 4), prompt_len=5, req_id="req-a")


def test_tracked_fully_cached_prompt_refuses():
    # a full prefix-cache hit schedules the prompt's last token as its
    # first computed one: decode-shaped, computed == N-1 — same refusal.
    with pytest.raises(lb.LensBatchError, match="INSIDE"):
        lb.check_tracked_segment(_seg(1, 8), prompt_len=9, req_id="req-a")


def test_tracked_sub_2_token_prompt_refuses():
    with pytest.raises(lb.LensBatchError, match="< 2"):
        lb.check_tracked_segment(_seg(1, 1), prompt_len=1, req_id="req-a")


def test_tracked_prompt_len_is_validated_loudly():
    with pytest.raises(lb.LensBatchError, match="prompt_len"):
        lb.check_tracked_segment(_seg(5, 0), prompt_len=None, req_id="req-a")
    with pytest.raises(lb.LensBatchError, match="prompt_len"):
        lb.check_tracked_segment(_seg(5, 0), prompt_len=5.5, req_id="req-a")


# --------------------------------------------------------------------------
# Part D — when the plan machinery applies (drives off the ops)
# --------------------------------------------------------------------------


def test_row_local_ops_skip_the_plan():
    assert lb.ops_need_plan([_row_local_op()]) is False
    assert lb.ops_need_plan([]) is False


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(
            la.HookOp(op=la.OP_ADD_VECTOR, layer=0,
                      vector=torch.ones(HIDDEN),
                      positions=la.POSITIONS_LAST),
            id="positions-last"),
        pytest.param(
            la.HookOp(op=la.OP_ADD_VECTOR, layer=0,
                      vector=torch.ones(HIDDEN), skip_bos=True),
            id="bos-skip"),
        pytest.param(
            la.HookOp(op=la.OP_ADD_VECTOR, layer=0,
                      vector=torch.ones(HIDDEN),
                      phase=la.PHASE_DECODE),
            id="phase-restricted"),
        pytest.param(
            la.HookOp(op=la.OP_PREFILL_LAST_MATMUL, layer=1,
                      matrix=torch.eye(HIDDEN),
                      phase=la.PHASE_PREFILL),
            id="alphasteer-prefill-matmul"),
    ])
def test_boundary_dependent_ops_need_the_plan(op):
    assert lb.ops_need_plan([op]) is True
    # one boundary-dependent op among row-local ones is enough
    assert lb.ops_need_plan([_row_local_op(), op]) is True
    # and the predicate is exactly lens_adapters.op_needs_context, reused
    assert lb.ops_need_plan([op]) == la.op_needs_context(op)
