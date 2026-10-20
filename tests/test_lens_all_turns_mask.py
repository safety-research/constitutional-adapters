"""THE ALL-TURNS MASK — ``positions="response_all_turns"``, the served
position mode that steers EVERY assistant-turn span of a render (each
assistant header through that turn's END SCAFFOLD: terminator + the glue
before the next turn's opener) plus every generated position — and its
resumed-prefill policy: ``conditional`` under the SAME two clauses as
``response`` — a complete earlier header in the cached region refuses, a
cache boundary inside a header refuses. The mode ships as a MASK with no
new cache-sharing claim; the clause-set machinery that would let its
admission narrow is exercised here with explicit clause sets, and the
writer/reader tests record which boundaries the masks would allow, so the
future narrowing flips assertions rather
than writing tests. CPU-only, synthetic tensors, invented token ids; the
real gemma-4 template where it is cached.

Every expected mask here is DERIVED from the render's construction (the
spans a builder laid down), never a literal copied from the output.

THE METHODOLOGY THE REVIEW ASKED FOR (its P1-1/P1-2/P1-3 all hid from a
cold-vs-own-suffix comparison): a cache hazard is about a DIFFERENT
request having written the shared prefix under a different mask. So the
``TestWriterReader`` classes build a WRITER request A and a READER request
B sharing a token-identical prefix, compute each one's full cold mask
independently, assert they agree on every shared position where the
admission predicate SERVES, and assert the predicate REFUSES exactly the
boundaries where they disagree.
"""

import io
import json
import sys
from pathlib import Path

import pytest
import torch

from sisteer.capeval import manifest as cm
from sisteer.capeval.serving import lens_adapters as la
from sisteer.capeval.serving import lens_steer as ls
from sisteer.capeval.serving import lens_worker as lw
from sisteer.capeval.serving import registry_shim as rs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_petri_lens_worker import HIDDEN, _forward, _obj, _registry, _sel  # noqa: E402

D = 8
#: invented ids standing in for a generation-prompt header whose OPENER is
#: shared with user turns (gemma-4: <|turn> model \n vs <|turn> user \n)
OPENER = 900
HEADER = (OPENER, 901, 902)
USER_HEADER = (OPENER, 903, 902)
#: the turn TERMINATOR and the GLUE the template puts after it (gemma-4's
#: <turn|> and \n); together they are the turn-END SCAFFOLD
TERM = 990
GLUE = 77
TURN_END = (TERM, GLUE)
ALL = la.POSITIONS_RESPONSE_ALL_TURNS
RESP = la.POSITIONS_RESPONSE


def _vec(seed: int, dim: int = D) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(dim, generator=g, dtype=torch.float32)


def _hidden(rows: int, seed: int = 7, dim: int = D) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, dim, generator=g, dtype=torch.float32)


def _all_turns_op(**kw) -> la.HookOp:
    kw.setdefault("vector", _vec(1))
    kw.setdefault("layer", 0)
    kw.setdefault("header_ids", HEADER)
    kw.setdefault("turn_end_ids", TURN_END)
    return la.HookOp(op=la.OP_ADD_VECTOR, positions=ALL, **kw)


def _response_op(**kw) -> la.HookOp:
    kw.setdefault("vector", _vec(1))
    kw.setdefault("layer", 0)
    kw.setdefault("header_ids", HEADER)
    return la.HookOp(op=la.OP_ADD_VECTOR, positions=RESP, **kw)


def _gated(positions, **kw) -> la.HookOp:
    g = torch.Generator().manual_seed(3)
    base = dict(op=la.OP_RELU_GATED_VECTOR, layer=0,
                vector=torch.randn(D, generator=g),
                gate_w=torch.randn(D, generator=g), gate_b=torch.zeros(1),
                positions=positions, header_ids=HEADER)
    if positions == ALL:
        base["turn_end_ids"] = TURN_END
    base.update(kw)
    return la.HookOp(**base)


def _prefill_ctx(ids, **facts) -> la.HookContext:
    ids = tuple(ids)
    return la.HookContext(phase=la.PHASE_PREFILL,
                          positions=tuple(range(len(ids))), token_ids=ids,
                          **facts)


def _resumed_ctx(ids, L, *, with_cached=True, **facts) -> la.HookContext:
    ids = tuple(ids)
    return la.HookContext(phase=la.PHASE_PREFILL,
                          positions=tuple(range(L, len(ids))),
                          token_ids=ids[L:], cached=L,
                          cached_ids=(ids[:L] if with_cached else None),
                          **facts)


def _user_turn(*content) -> list:
    return list(USER_HEADER) + list(content) + list(TURN_END)


def _conversation(n_prior: int, *, closed_tail=False):
    """A render with ``n_prior`` COMPLETED assistant turns, each preceded
    by a user turn, then either a final user turn + the generation-prompt
    header (the ordinary shape) or — ``closed_tail`` — a render that ENDS
    right after the last assistant turn's scaffold (gemma-4's Petri
    ``content_and_tool_response`` shape, served under a declaration).
    Returns ``(ids, assistant_spans)`` with the spans read off the
    CONSTRUCTION: each earlier span is header + content + the turn-end
    scaffold; the final span is the open generating turn (to the prompt
    end)."""
    ids, spans = [], []
    for k in range(n_prior):
        ids += _user_turn(100 + 3 * k, 101 + 3 * k)
        lo = len(ids)
        ids += list(HEADER) + [200 + 3 * k, 201 + 3 * k] + list(TURN_END)
        spans.append((lo, len(ids)))
    if closed_tail:
        assert n_prior >= 1
        lo, _ = spans.pop()
        spans.append((lo, len(ids)))     # the closed final turn IS the generating span
        return tuple(ids), tuple(spans)
    ids += _user_turn(300, 301)
    lo = len(ids)
    ids += list(HEADER)
    spans.append((lo, len(ids)))
    return tuple(ids), tuple(spans)


def _continued(a):
    """Turn N's request from turn N-1's: A's whole prompt (which ended at
    the generating header), the assistant's answer + scaffold, a user turn,
    the new generating header — with the spans read off the construction."""
    a = tuple(a)
    lo0 = len(a) - len(HEADER)
    b = list(a) + [200, 201] + list(TURN_END)
    hi0 = len(b)
    b += _user_turn(400, 401)
    lo1 = len(b)
    b += list(HEADER)
    return tuple(b), ((lo0, hi0), (lo1, len(b)))


def _indices(spans) -> set:
    return {i for lo, hi in spans for i in range(lo, hi)}


def _touched(out: torch.Tensor, h: torch.Tensor) -> set:
    return {i for i, b in enumerate((out != h).any(dim=-1).tolist()) if b}


def _mask_list(op, ids, **facts) -> list:
    """One request's FULL cold mask, as a list of bools."""
    ids = tuple(ids)
    return la._row_selection(op, _prefill_ctx(ids, **facts),
                             _hidden(len(ids))).tolist()


def _mask_set(op, ctx, h) -> set:
    sel = la._row_selection(op, ctx, h)
    return {i for i, b in enumerate(sel.tolist()) if b}


def _shared_prefix_len(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# ============================================================================
# A — the mask
# ============================================================================


class TestTheMask:
    def test_all_turns_masks_every_assistant_span_and_response_only_the_last(self):
        """THE HEADLINE: a 3-assistant-turn render — under ``response``
        only the final turn's header rows are edited; under
        ``response_all_turns`` all three spans are, and nothing else. Both
        expectations are derived from the builder's spans."""
        ids, spans = _conversation(2)
        assert len(spans) == 3
        P = len(ids)
        h = _hidden(P)
        ctx = _prefill_ctx(ids)
        out_all = la.apply_hook_ops(h, (_all_turns_op(),), 1.0, layer=0, ctx=ctx)
        out_resp = la.apply_hook_ops(h, (_response_op(),), 1.0, layer=0, ctx=ctx)
        assert _touched(out_all, h) == _indices(spans)
        assert _touched(out_resp, h) == _indices(spans[-1:])
        assert _indices(spans[-1:]) == set(range(P - len(HEADER), P))
        for op, want in ((_all_turns_op(), _indices(spans)),
                         (_response_op(), _indices(spans[-1:]))):
            dense = la._row_mask(op, ctx, h)
            sel = la._row_selection(op, ctx, h)
            assert torch.equal(dense, sel)
            assert {i for i, b in enumerate(sel.tolist()) if b} == want

    def test_earlier_spans_end_after_the_scaffold(self):
        """An earlier span runs through the turn-end SCAFFOLD — terminator
        AND glue — and stops before the next turn's opener."""
        ids, spans = _conversation(3)
        got = _mask_set(_all_turns_op(), _prefill_ctx(ids), _hidden(len(ids)))
        for lo, hi in spans[:-1]:
            assert ids[hi - 2:hi] == TURN_END and {hi - 2, hi - 1} <= got
            assert ids[hi] == OPENER and hi not in got          # the USER turn's opener
            assert ids[lo:lo + len(HEADER)] == HEADER and lo in got

    def test_user_turns_and_their_scaffold_stay_unmasked(self):
        ids, spans = _conversation(3)
        got = _mask_set(_all_turns_op(), _prefill_ctx(ids), _hidden(len(ids)))
        inside = _indices(spans)
        assert not (got - inside)
        user_terms = [i for i, t in enumerate(ids) if t == TERM and i not in inside]
        assert len(user_terms) == 4 and not (set(user_terms) & got)
        user_openers = [i for i in range(len(ids) - 2) if ids[i:i + 3] == USER_HEADER]
        assert len(user_openers) == 4 and not (set(user_openers) & got)

    def test_single_turn_render_is_the_response_mask(self):
        ids, spans = _conversation(0)
        P, H = len(ids), len(HEADER)
        ctx = _prefill_ctx(ids)
        h = _hidden(P)
        a = la._row_selection(_all_turns_op(), ctx, h)
        r = la._row_selection(_response_op(), ctx, h)
        assert torch.equal(a, r)
        assert a.tolist() == [False] * (P - H) + [True] * H

    def test_closed_tail_declared_render_steers_through_its_scaffold(self):
        """The supported gemma-4 shape behind P1-2: a declared render that
        ENDS with a closed assistant turn steers header .. scaffold as its
        FINAL span (to the prompt end), earlier turns included."""
        ids, spans = _conversation(2, closed_tail=True)
        assert ids[-2:] == TURN_END
        got = _mask_set(_all_turns_op(), _prefill_ctx(ids, agent_loop=True), _hidden(len(ids)))
        assert got == _indices(spans)
        assert spans[-1][1] == len(ids)

    def test_every_decode_row_is_steered(self):
        for ctx in (la.HookContext(phase=la.PHASE_DECODE, positions=(40,), token_ids=(4242,)),
                    la.DecodeRows()):
            h = _hidden(1)
            out = la.apply_hook_ops(h, (_all_turns_op(),), 1.0, layer=0, ctx=ctx)
            assert not torch.equal(out, h)
            assert la._row_selection(_all_turns_op(), ctx, h) is None

    def test_skip_bos_composes_with_the_all_turns_mask(self):
        ids = HEADER
        out = la.apply_hook_ops(_hidden(len(ids)), (_all_turns_op(skip_bos=True),), 1.0,
                                layer=0, ctx=_prefill_ctx(ids))
        assert (out != _hidden(len(ids))).any(dim=-1).tolist() == [False, True, True]

    def test_scale_zero_is_bit_identical(self):
        ids, _ = _conversation(2)
        h = _hidden(len(ids))
        assert la.apply_hook_ops(h, (_all_turns_op(),), 0.0, layer=0, ctx=_prefill_ctx(ids)) is h

    def test_gated_op_edits_only_inside_the_spans(self):
        ids, spans = _conversation(2)
        h = _hidden(len(ids))
        out = la.apply_hook_ops(h, (_gated(ALL),), 1.0, layer=0, ctx=_prefill_ctx(ids))
        assert _touched(out, h) <= _indices(spans) and _touched(out, h) & _indices(spans)

    def test_declared_seeded_render_masks_earlier_turns_header_and_seed(self):
        ids, spans = _conversation(1)
        seeded = ids + (555, 556)
        P = len(seeded)
        got = _mask_set(_all_turns_op(), _prefill_ctx(seeded, response_prefill=True), _hidden(P))
        assert got == _indices(spans[:-1]) | set(range(spans[-1][0], P))
        assert _mask_set(_all_turns_op(), _prefill_ctx(seeded, agent_loop=True), _hidden(P)) == got
        for op in (_all_turns_op(), _response_op()):
            with pytest.raises(la.AdapterError, match="does not end at the header"):
                la._row_selection(op, _prefill_ctx(seeded), _hidden(P))

    def test_completed_agent_loop_turn_is_one_span(self):
        """gemma-4 renders a completed tool-using model turn as ONE open
        turn — header, tool call, tool response, text, one scaffold — so
        its span is header through that scaffold (invented control ids
        stand in for the tool scaffold)."""
        ids = _user_turn(100, 101)
        lo = len(ids)
        ids += list(HEADER) + [48, 601, 49, 50, 602, 51, 603] + list(TURN_END)
        hi = len(ids)
        ids += _user_turn(110, 111)
        final = len(ids)
        ids += list(HEADER)
        got = _mask_set(_all_turns_op(), _prefill_ctx(ids), _hidden(len(ids)))
        assert got == set(range(lo, hi)) | set(range(final, len(ids)))


class TestTheRefusals:
    def test_missing_token_ids_refuses(self):
        ids, _ = _conversation(1)
        ctx = la.HookContext(phase=la.PHASE_PREFILL, positions=tuple(range(len(ids))))
        with pytest.raises(la.AdapterError, match="no token ids"):
            la.apply_hook_ops(_hidden(len(ids)), (_all_turns_op(),), 1.0, layer=0, ctx=ctx)

    def test_an_unclosed_earlier_turn_refuses(self):
        """An earlier header with NO scaffold before the next header:
        refuse, never guess the span's end. (The documented limit: with a
        LATER turn's scaffold present the two markers cannot tell an
        unclosed turn from a longer one — a shape no template produces;
        scaffold ids from another template find no occurrence and refuse
        here, a wrong registry refuses at startup.)"""
        ids, spans = _conversation(1)
        lo, hi = spans[0]
        broken = list(ids[:hi - 2]) + [GLUE] + list(USER_HEADER) + [110, 111, GLUE] + list(HEADER)
        with pytest.raises(la.AdapterError, match="not closed by the turn-end scaffold"):
            la._row_selection(_all_turns_op(), _prefill_ctx(broken), _hidden(len(broken)))
        with pytest.raises(la.AdapterError, match="not closed by the turn-end scaffold"):
            la._row_selection(_all_turns_op(turn_end_ids=(991, 77)), _prefill_ctx(ids),
                              _hidden(len(ids)))

    def test_the_final_boundary_refusals_are_the_response_ones(self):
        for ids, msg in ((HEADER[:2], "shorter than its own header"),
                         (tuple(range(10, 16)) + (901, 902, 903), "not the assistant header")):
            for op in (_all_turns_op(), _response_op()):
                with pytest.raises(la.AdapterError, match=msg):
                    la._row_selection(op, _prefill_ctx(ids), _hidden(len(ids)))

    def test_declared_render_with_no_header_anywhere_refuses(self):
        ids = (1, 2, 3, 4, 5, 6)
        with pytest.raises(la.AdapterError, match="occur NOWHERE"):
            la._row_selection(_all_turns_op(), _prefill_ctx(ids, agent_loop=True), _hidden(len(ids)))

    def test_misaligned_rows_refuse(self):
        ids, _ = _conversation(1)
        with pytest.raises(la.AdapterError, match="mis-aligned"):
            la._all_turns_prefill_mask(_all_turns_op(), _prefill_ctx(ids), len(ids) + 1)

    def test_all_turns_ops_need_a_context(self):
        assert la.op_needs_context(_all_turns_op())
        with pytest.raises(la.AdapterError, match="needs a HookContext"):
            la.apply_hook_ops(_hidden(4), (_all_turns_op(),), 1.0, layer=0)


class TestQuotedDelimiterFailsClosed:
    """Finding 2 of the re-review and its mirror from the third review (a
    correctness bug in the mask itself): a complete turn-end scaffold
    INSIDE assistant content — text quoting the template's literal
    delimiter — would end a HISTORICAL span early (the rest of that turn
    silently unsteered, no request failure) and would run a FINAL span
    past it to the prompt end. Both positions fail closed: the generating
    turn refuses a scaffold followed by more content; a historical turn
    refuses a scaffold followed by anything but a turn opener. Expected
    spans here come from the render's CONSTRUCTION, never from the
    scanner, and the residual limit (a fully reproduced turn boundary in
    content) is precondition (b) in lens_adapters' module docstring."""

    def _quoting_writer(self):
        ids = _user_turn(100, 101)
        lo = len(ids)
        ids += list(HEADER) + [200, 201] + list(TURN_END) + [202, 203]   # the assistant quotes the delimiter, then goes on
        return tuple(ids), lo

    def test_generating_turn_with_a_quoted_delimiter_refuses(self):
        a, lo = self._quoting_writer()
        for facts in ({"agent_loop": True}, {"response_prefill": True}):
            with pytest.raises(la.AdapterError, match="complete turn-end scaffold") as ei:
                la._row_selection(_all_turns_op(), _prefill_ctx(a, **facts), _hidden(len(a)))
            assert "quoting the template's own turn delimiter" in str(ei.value)
        # and the same render is refused at the span level, naming the position
        with pytest.raises(la.AdapterError, match=f"at position {lo + len(HEADER) + 2}"):
            la.assistant_turn_spans(a, HEADER, TURN_END, lo)

    def test_closed_tail_is_not_the_quoting_shape(self):
        """A scaffold exactly at the prompt end is the closed-tail render:
        both readings end the span there, so it is served."""
        a, spans = _conversation(1, closed_tail=True)
        assert a[-2:] == TURN_END
        assert _mask_set(_all_turns_op(), _prefill_ctx(a, agent_loop=True), _hidden(len(a))) == _indices(spans)

    def test_quoted_delimiter_in_an_earlier_turn_refuses_cold(self):
        """The third review's cold counterexample: the quoted delimiter in
        an EARLIER assistant turn with a user follow-up, submitted cold
        (undeclared, ends at the generating header). The scanner's reading
        would end that turn at the quote and leave the rest of it
        unsteered with no request failure; the render REFUSES, naming the
        quoted scaffold's position and the content token that follows it.
        The refusal reaches the apply too (no silent partial serve)."""
        a, lo = self._quoting_writer()
        b = tuple(list(a) + list(TURN_END) + _user_turn(400, 401) + list(HEADER))
        assert _shared_prefix_len(a, b) == len(a)
        with pytest.raises(la.AdapterError, match="complete turn-end scaffold"):
            _mask_list(_all_turns_op(), a, agent_loop=True)               # A: the generating-turn rule
        quote_at = lo + len(HEADER) + 2
        with pytest.raises(la.AdapterError, match=f"scaffold .* at position {quote_at}, by tokens .*202.* at position "
                                                  f"{quote_at + len(TURN_END)} rather than a complete role header") as ei:
            _mask_list(_all_turns_op(), b)                                # B: the historical rule (cold)
        assert "quoting the template's own turn delimiter" in str(ei.value)
        with pytest.raises(la.AdapterError, match="rather than a complete role header"):
            la.apply_hook_ops(_hidden(len(b)), (_all_turns_op(),), 1.0, layer=0, ctx=_prefill_ctx(b))
        # the message-derived span the mask SHOULD have covered: header through the REAL scaffold
        real_end = len(a) + len(TURN_END)
        assert b[real_end - len(TURN_END):real_end] == TURN_END and b[real_end] == OPENER
        # response is untouched by any of this: it masks the final header only
        assert _mask_list(_response_op(), b) == [i >= len(b) - len(HEADER) for i in range(len(b))]

    def test_scaffold_plus_opener_plus_prose_refuses(self):
        """The fourth review's bypass of the opener-only rule: assistant
        text quoting the delimiter AND the opener ("the tokens are <turn|>\\n
        <|turn> and then prose"), a user follow-up, submitted cold. An
        opener token alone is not a genuine next turn — the complete role
        header (opener, one role token, the tail) must follow the scaffold,
        or the next header itself. Refused, naming the tokens found; the
        message-derived span the mask should have covered is the whole
        turn through its real scaffold."""
        ids = _user_turn(100)
        lo = len(ids)
        ids += list(HEADER) + [200] + list(TURN_END) + [OPENER, 555, 556] + list(TURN_END)   # ... <turn|>\n <|turn> prose
        real_end = len(ids)
        ids += _user_turn(300) + list(HEADER)
        ids = tuple(ids)
        quote_at = lo + len(HEADER) + 1
        with pytest.raises(la.AdapterError, match="rather than a complete role header") as ei:
            la.apply_hook_ops(_hidden(len(ids)), (_all_turns_op(),), 1.0, layer=0, ctx=_prefill_ctx(ids))
        assert f"at position {quote_at}" in str(ei.value) and str([OPENER, 555, 556]) in str(ei.value)
        assert ids[real_end - len(TURN_END):real_end] == TURN_END and ids[real_end:real_end + 3] == USER_HEADER
        # the nested variant: opener + one token + something that is not the tail (a role-shaped fragment)
        nested = list(ids[:quote_at + len(TURN_END)]) + [OPENER, 903, 444, 557] + list(TURN_END) + _user_turn(300) + list(HEADER)
        with pytest.raises(la.AdapterError, match="rather than a complete role header"):
            la._row_selection(_all_turns_op(), _prefill_ctx(nested), _hidden(len(nested)))
        # and a quoted scaffold followed by a header-SHAPED fake (opener, any token, tail) is the residual:
        # a complete turn boundary the token stream cannot vouch against — served, documented (docs §7)
        forged = list(ids[:quote_at + len(TURN_END)]) + [OPENER, 903, 902] + [558] + list(TURN_END) \
            + _user_turn(300) + list(HEADER)
        la._row_selection(_all_turns_op(), _prefill_ctx(forged), _hidden(len(forged)))
        # response is untouched
        assert _mask_list(_response_op(), ids) == [i >= len(ids) - len(HEADER) for i in range(len(ids))]

    def test_a_scaffold_followed_by_the_opener_is_a_genuine_turn_end(self):
        """The rule's positive case, derived from construction: a genuine
        scaffold is followed by the next turn's opener (user or model), and
        the span ends exactly there — consecutive model turns included."""
        ids, spans = _conversation(2)
        assert la.assistant_turn_spans(ids, HEADER, TURN_END, spans[-1][0]) == spans
        consecutive = list(HEADER) + [200] + list(TURN_END) + list(HEADER) + [201] + list(TURN_END) \
            + _user_turn(300) + list(HEADER)
        got = la.assistant_turn_spans(consecutive, HEADER, TURN_END, len(consecutive) - len(HEADER))
        first_end = len(HEADER) + 1 + len(TURN_END)
        assert got == ((0, first_end), (first_end, 2 * first_end), (len(consecutive) - len(HEADER), len(consecutive)))

    def test_bare_terminator_without_its_glue_is_content(self):
        """A lone terminator (no glue after it) is not the scaffold and
        ends nothing — in an earlier turn and in the generating turn."""
        ids = _user_turn(100)
        lo = len(ids)
        ids += list(HEADER) + [200, TERM, 201] + list(TURN_END)     # bare terminator inside an earlier turn
        hi = len(ids)
        ids += _user_turn(300)
        final = len(ids)
        ids += list(HEADER) + [202, TERM, 203]                       # ... and inside the declared generating turn
        got = _mask_set(_all_turns_op(), _prefill_ctx(ids, agent_loop=True), _hidden(len(ids)))
        assert got == set(range(lo, hi)) | set(range(final, len(ids)))


class TestAssistantTurnSpans:
    def test_spans_are_the_builders(self):
        ids, spans = _conversation(3)
        assert la.assistant_turn_spans(ids, HEADER, TURN_END, spans[-1][0]) == spans

    def test_final_start_must_hold_the_header(self):
        ids, spans = _conversation(1)
        with pytest.raises(la.AdapterError, match="does not hold"):
            la.assistant_turn_spans(ids, HEADER, TURN_END, spans[-1][0] - 1)
        with pytest.raises(la.AdapterError, match="non-empty"):
            la.assistant_turn_spans(ids, HEADER, (), spans[-1][0])

    def test_single_id_scaffold(self):
        """A template with no glue (Llama-3.1: [<|eot_id|>]) ends the span
        right after the terminator — which, in a genuine render, the next
        turn's opener follows."""
        ids = [1, 2] + list(HEADER) + [201, TERM] + [OPENER, 903, 902, 3, TERM] + list(HEADER)
        assert la.assistant_turn_spans(ids, HEADER, (TERM,), len(ids) - len(HEADER)) == \
            ((2, 2 + len(HEADER) + 2), (len(ids) - len(HEADER), len(ids)))
        # content right after the bare terminator is the quoted-delimiter shape: refused
        with pytest.raises(la.AdapterError, match="rather than a complete role header"):
            la.assistant_turn_spans([1, 2] + list(HEADER) + [201, TERM, 3] + list(HEADER), HEADER, (TERM,),
                                    2 + len(HEADER) + 3)


# ============================================================================
# B — response stays byte-identical (the pins the mode must not move)
# ============================================================================


class TestResponseUnchanged:
    def test_single_turn_response_mask_and_arithmetic(self):
        ids, _ = _conversation(0)
        P, H = len(ids), len(HEADER)
        ctx = _prefill_ctx(ids)
        h = _hidden(P)
        vec = _vec(3)
        op = _response_op(vector=vec, coeff=2.0)
        out = la.apply_hook_ops(h, (op,), 0.5, layer=0, ctx=ctx)
        mask = torch.zeros(P, dtype=torch.float32)
        mask[P - H:] = 1.0
        assert torch.equal(out, h + (vec * (0.5 * 2.0)) * mask.unsqueeze(-1))
        assert la._row_mask(op, ctx, h).tolist() == [False] * (P - H) + [True] * H
        assert la._row_selection(op, ctx, h).tolist() == [False] * (P - H) + [True] * H
        assert la._response_start(op, ctx, P) == P - H

    def test_multi_turn_response_masks_only_the_final_header(self):
        ids, spans = _conversation(3)
        P, H = len(ids), len(HEADER)
        got = _mask_set(_response_op(), _prefill_ctx(ids), _hidden(P))
        assert got == set(range(P - H, P)) and not (got & _indices(spans[:-1]))

    def test_response_resumed_rules_unchanged(self):
        ids, spans = _conversation(1)
        P = len(ids)
        op = _response_op()
        h = _hidden(P)
        earlier_lo, earlier_hi = spans[0]
        L = earlier_lo
        warm = la.apply_hook_ops(h[L:], (op,), 1.0, layer=0, ctx=_resumed_ctx(ids, L))
        cold = la.apply_hook_ops(h, (op,), 1.0, layer=0, ctx=_prefill_ctx(ids))
        assert torch.equal(warm, cold[L:])
        with pytest.raises(la.ResumedPrefillRefusal, match="prior writer's response boundary"):
            la.apply_hook_ops(h[earlier_hi:], (op,), 1.0, layer=0, ctx=_resumed_ctx(ids, earlier_hi))
        with pytest.raises(la.ResumedPrefillRefusal, match="INSIDE the assistant header"):
            la.apply_hook_ops(h[earlier_lo + 1:], (op,), 1.0, layer=0,
                              ctx=_resumed_ctx(ids, earlier_lo + 1))
        with pytest.raises(la.ResumedPrefillRefusal, match="no cached_ids"):
            la.apply_hook_ops(h[L:], (op,), 1.0, layer=0, ctx=_resumed_ctx(ids, L, with_cached=False))

    def test_response_conflict_predicate_default_is_both_clauses(self):
        """The pre-existing predicate, called the pre-existing way, keeps
        both clauses."""
        assert la.cached_region_header_conflict(HEADER, (1, 2) + HEADER + (5,)) is not None
        assert la.cached_region_header_conflict(HEADER, (1, 2, OPENER)) is not None
        assert la.cached_region_header_conflict(HEADER, (1, 2, 3)) is None
        assert la.conflict_clauses_for(RESP) == la.RESPONSE_CONFLICT_CLAUSES == set(la.CONFLICT_CLAUSES)


# ============================================================================
# C — the op vocabulary
# ============================================================================


class TestHookOpVocabulary:
    def test_mode_constants(self):
        assert la.POSITIONS_RESPONSE_ALL_TURNS == "response_all_turns"
        assert la.POSITIONS_RESPONSE_ALL_TURNS in la.POSITION_MODES
        assert la.POSITIONS_RESPONSE_ALL_TURNS not in la.FIT_POSITION_MODES
        assert la.FIT_POSITION_MODES == (la.POSITIONS_ALL, la.POSITIONS_RESPONSE)
        assert set(la.RESPONSE_MASK_MODES) == {RESP, ALL}
        assert la._SERVED_POSITION_MODES["response_all_turns"] == ALL

    def test_all_turns_requires_both_markers(self):
        with pytest.raises(la.AdapterError, match="requires header_ids"):
            la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=_vec(1), positions=ALL, turn_end_ids=TURN_END)
        with pytest.raises(la.AdapterError, match="requires turn_end_ids"):
            la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=_vec(1), positions=ALL, header_ids=HEADER)
        with pytest.raises(la.AdapterError, match="turn_end_ids is empty"):
            _all_turns_op(turn_end_ids=())
        with pytest.raises(la.AdapterError, match="turn_end_ids must be non-negative ints"):
            _all_turns_op(turn_end_ids=(1, -2))
        op = _all_turns_op(turn_end_ids=[TERM, GLUE])
        assert op.turn_end_ids == TURN_END and op.header_ids == HEADER

    def test_other_modes_refuse_turn_end_ids(self):
        with pytest.raises(la.AdapterError, match="does not take turn_end_ids"):
            _response_op(turn_end_ids=TURN_END)
        with pytest.raises(la.AdapterError, match="does not take turn_end_ids"):
            la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=_vec(1), turn_end_ids=TURN_END)
        with pytest.raises(la.AdapterError, match="does not take header_ids"):
            la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=_vec(1), header_ids=HEADER)
        with pytest.raises(la.AdapterError, match="requires header_ids"):
            la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=_vec(1), positions=RESP)

    def test_lora_oproj_admits_the_mode(self):
        g = torch.Generator().manual_seed(1)
        a, b = torch.randn(2, D, generator=g), torch.randn(D, 2, generator=g)
        op = la.HookOp(op=la.OP_LORA_OPROJ, layer=0, lowrank_a=a, lowrank_b=b,
                       positions=ALL, header_ids=HEADER, turn_end_ids=TURN_END)
        assert op.positions == ALL
        with pytest.raises(la.AdapterError, match="'all' or 'response'"):
            la.HookOp(op=la.OP_LORA_OPROJ, layer=0, lowrank_a=a, lowrank_b=b, positions=la.POSITIONS_LAST)
        ids, spans = _conversation(1)
        P = len(ids)
        x_in, x_out = _hidden(P, seed=1), _hidden(P, seed=2)
        out = la.apply_oproj_ops(x_in, x_out, (op,), 1.0, layer=0, ctx=_prefill_ctx(ids))
        assert _touched(out, x_out) == _indices(spans)


# ============================================================================
# D — the resumed-prefill policy: CONDITIONAL, boundary clause only
# ============================================================================


class TestPolicy:
    def test_all_turns_is_conditional_under_the_boundary_clause_only(self):
        assert la.op_resumed_prefill_policy(_all_turns_op()) == la.RESUMED_CONDITIONAL
        assert la.op_resumed_prefill_policy(_response_op()) == la.RESUMED_CONDITIONAL
        assert la.resumed_prefill_ok([_all_turns_op()]) is False          # never "always"
        assert la.resumed_prefill_policy([_all_turns_op(), _gated(ALL)]) == la.RESUMED_CONDITIONAL
        never = la.HookOp(op=la.OP_PREFILL_LAST_MATMUL, layer=0, matrix=torch.eye(D), phase=la.PHASE_PREFILL)
        assert la.resumed_prefill_policy([_all_turns_op(), never]) == la.RESUMED_NEVER
        assert la.op_resumed_prefill_policy(_all_turns_op(phase=la.PHASE_DECODE)) == la.RESUMED_NEVER
        # the clause sets: the SAME full set for both modes today (no new cache-sharing claim)
        assert la.conflict_clauses_for(ALL) == la.ALL_TURNS_CONFLICT_CLAUSES == la.RESPONSE_CONFLICT_CLAUSES
        assert la.conflict_clauses_for(RESP) == la.RESPONSE_CONFLICT_CLAUSES == set(la.CONFLICT_CLAUSES)
        with pytest.raises(la.AdapterError, match="not a header-located mode"):
            la.conflict_clauses_for(la.POSITIONS_ALL)
        assert la.resumed_conflict_clauses([_all_turns_op()]) == la.RESPONSE_CONFLICT_CLAUSES
        assert la.resumed_conflict_clauses([_all_turns_op(), _response_op()]) == la.RESPONSE_CONFLICT_CLAUSES
        assert la.resumed_conflict_clauses([_gated(la.POSITIONS_ALL, header_ids=None)]) == frozenset()
        assert la.response_header_ids([_all_turns_op()]) == HEADER

    def test_the_two_clauses_factored(self):
        """The machinery a future narrowing would flip, exercised with
        EXPLICIT clause sets; the shipped per-mode sets are both full."""
        region_full = (1, 2) + HEADER + (201, TERM, GLUE, 5)      # a full earlier header inside
        region_partial = (1, 2) + HEADER[:1]                       # the boundary inside a header
        region_partial2 = (1, 2) + HEADER[:2]
        both, bnd = la.RESPONSE_CONFLICT_CLAUSES, frozenset({la.CONFLICT_BOUNDARY_INSIDE_HEADER})
        assert la.cached_region_header_conflict(HEADER, region_full, both) is not None
        assert la.cached_region_header_conflict(HEADER, region_full, bnd) is None      # what narrowing WOULD admit
        for r in (region_partial, region_partial2):
            assert "INSIDE the assistant header" in la.cached_region_header_conflict(HEADER, r, both)
            assert "INSIDE the assistant header" in la.cached_region_header_conflict(HEADER, r, bnd)  # KEPT
        assert la.cached_region_header_conflict(HEADER, (1, 2, 3), bnd) is None
        with pytest.raises(la.AdapterError, match="non-empty subset"):
            la.cached_region_header_conflict(HEADER, (1,), frozenset())
        with pytest.raises(la.AdapterError, match="non-empty subset"):
            la.cached_region_header_conflict(HEADER, (1,), frozenset({"bogus"}))
        # the cell helper picks the clause set from the ops' modes — the full set for both today
        assert la.resumed_header_conflict([_all_turns_op()], region_full) is not None
        assert la.resumed_header_conflict([_response_op()], region_full) is not None
        assert la.resumed_header_conflict([_all_turns_op()], region_partial) is not None
        assert la.resumed_header_conflict([_all_turns_op(), _response_op()], region_full) is not None


# ============================================================================
# E — WRITER / READER: the shape a cache hazard actually takes
# ============================================================================


def _agree_on_shared(mask_a, mask_b, n) -> bool:
    return mask_a[:n] == mask_b[:n]


class TestWriterReader:
    """Writer request A and reader request B share a token-identical prefix
    ``[0, n)``; each cold mask is computed INDEPENDENTLY on its own full
    prompt. Where the masks agree on the shared prefix, a resume of A's
    cache by B (for any cache boundary ``L <= n`` the predicate admits) is
    correct position by position; where they DISAGREE at some shared
    position ``p``, every ``L > p`` is a poison resume and the predicate
    must refuse it. The predicate may refuse MORE than the masks require
    (it does today: the shipped all-turns clause set is ``response``'s),
    and the tests below record both facts separately — what the masks
    would allow, and what is admitted — so the future narrowing flips
    assertions rather than writing tests."""

    def _check_pair(self, op, a, b, *, facts_a=None, facts_b=None):
        ma = _mask_list(op, a, **(facts_a or {}))
        mb = _mask_list(op, b, **(facts_b or {}))
        n = _shared_prefix_len(a, b)
        disagree = [p for p in range(n) if ma[p] != mb[p]]
        clauses = la.conflict_clauses_for(op.positions)
        admitted, refused = [], []
        for L in range(1, n + 1):
            why = la.cached_region_header_conflict(op.header_ids, tuple(b[:L]), clauses)
            (admitted if why is None else refused).append(L)
        # SOUNDNESS: every admitted boundary lies at or before the first disagreement
        first_bad = min(disagree) if disagree else n
        assert all(L <= first_bad for L in admitted), (disagree, admitted)
        # and an admitted resume reproduces the reader's own cold mask on the slice
        for L in admitted:
            if L < len(b):
                ctx = _resumed_ctx(b, L, **(facts_b or {}))
                sel = la._row_selection(op, ctx, _hidden(len(b) - L))
                assert sel.tolist() == mb[L:], L
        return n, disagree, admitted, refused

    def test_turn_n_from_turn_n_minus_1_masks_agree_but_admission_is_held(self):
        """A = turn 1 (one header, at the tail); B = turn 2 (A's whole
        prompt + a user turn + the generating header) — the motivating
        case. The MASKS agree on every shared position (the all-turns cold
        prefill of B steers A's turn too), which is what would justify
        admitting the boundary at A's prompt end. It is NOT admitted today:
        A's header is a complete occurrence in B's cached region and the
        shipped clause set is ``response``'s (docs §7 for what unlocks it)."""
        a, _ = _conversation(0)
        b, spans_b = _continued(a)
        n = _shared_prefix_len(a, b)
        assert n == len(a)                              # A's prompt is a prefix of B's
        assert _mask_list(_all_turns_op(), b) == [i in _indices(spans_b) for i in range(len(b))]
        na, disagree, admitted, refused = self._check_pair(_all_turns_op(), a, b)
        assert not disagree                             # the masks would allow every shared boundary
        assert len(a) in refused                        # ... but the turn boundary is REFUSED today (full occurrence)
        assert all(L in refused for L in range(len(a) - len(HEADER) + 1, len(a)))   # inside A's header
        assert len(a) - len(HEADER) in admitted         # right before A's header: header-free region, served
        why = la.cached_region_header_conflict(HEADER, tuple(b[:len(a)]), la.ALL_TURNS_CONFLICT_CLAUSES)
        assert why is not None and "prior writer's response boundary" in why
        # what the narrowing would admit, recorded against the masks' agreement
        assert la.cached_region_header_conflict(HEADER, tuple(b[:len(a)]),
                                                frozenset({la.CONFLICT_BOUNDARY_INSIDE_HEADER})) is None

    def test_second_user_turn_is_the_boundary_inside_header_hazard(self):
        """P1-1's counterexample: A = [user, generating header]; B = [user,
        a SECOND user turn, ..., header]. They share the header's OPENER
        (shared with user turns). A steers that opener (it is A's header);
        B does not (it is a user turn's). The predicate must refuse every
        boundary past the opener — and it does, under the boundary clause
        alone, because the region then ends with header[:1]."""
        a, _ = _conversation(0)                         # ... user turn, HEADER
        prefix = list(a[:-len(HEADER)])                 # everything before A's header
        b = tuple(prefix + _user_turn(400, 401) + list(HEADER))
        n = _shared_prefix_len(a, b)
        assert n == len(prefix) + 1                     # exactly the shared opener
        na, disagree, admitted, refused = self._check_pair(_all_turns_op(), a, b)
        assert disagree == [len(prefix)]                # the opener: masked by A, not by B
        assert n in refused and (n - 1) in admitted

    def test_closed_tail_glue_pair_p1_2(self):
        """P1-2's counterexample: A = a declared closed-tail render whose
        final span runs through its scaffold (terminator + glue); B = the
        same conversation with a user turn following. They share A's whole
        prompt, glue included. With the scaffold as the span end the masks
        agree everywhere shared (a span stopping at the terminator would
        leave the glue unsteered in B: same ids, different mask)."""
        a, spans_a = _conversation(1, closed_tail=True)
        assert a[-2:] == TURN_END
        b, spans_b = _conversation(2)
        n = _shared_prefix_len(a, b)
        assert n == len(a)
        ma = _mask_list(_all_turns_op(), a, agent_loop=True)
        mb = _mask_list(_all_turns_op(), b)
        assert ma == mb[:n]
        assert ma[-1] and ma[-2]                        # A steers its terminator AND glue
        # what a terminator-only historical span would have done, for the record:
        span_end_at_term = spans_b[0][1] - 1            # ... excluding the glue
        assert mb[span_end_at_term] is True             # the glue IS steered in B
        na, disagree, admitted, refused = self._check_pair(
            _all_turns_op(), a, b, facts_a={"agent_loop": True})
        assert not disagree and len(a) in refused        # agree; the boundary is still refused today (full header cached)

    def test_open_turn_pairs_agree(self):
        """The two open-turn agent-loop shapes: A's final span covers a
        mid-turn prompt end; in B that turn is historical and closed.
        Agreement on the shared prefix, and every non-header-internal
        boundary admitted."""
        b, spans_b = _conversation(1)
        lo, hi = spans_b[0]
        for cut in (lo + len(HEADER) + 1, hi - 2):       # mid-content; right before the scaffold
            a = tuple(b[:cut])
            na, disagree, admitted, refused = self._check_pair(
                _all_turns_op(), a, b, facts_a={"agent_loop": True})
            assert not disagree
            assert lo in admitted                                            # right before the header: header-free
            assert {lo + 1, lo + 2, lo + len(HEADER), len(a)} <= set(refused)  # inside / past the header: refused today

    def test_boundary_inside_the_scaffold_is_sound_the_span_end_asymmetry(self):
        """The multi-token turn-end scaffold needs NO partial-boundary
        clause, unlike the header: a header is a span START (a completion
        that is not the header leaves the cached opener unmasked for one
        request and masked for the other), the scaffold is a span END (a
        completion that is not the scaffold EXTENDS the span). So writer A
        and a reader B sharing a prefix that ends between the terminator and
        the glue agree on the cached terminator whatever follows it in B —
        the template's glue (the real case) or, in a deliberately malformed
        reader, anything else."""
        a, spans_a = _conversation(1)
        lo, hi = spans_a[0]
        term_at = hi - len(TURN_END)                     # the earlier turn's terminator
        L = term_at + 1                                  # boundary INSIDE the scaffold
        assert a[term_at] == TERM and a[L] == GLUE
        # the real reader: the template's glue follows — identical prefix, same masks, boundary admitted
        b_real, _ = _continued(a)
        assert _shared_prefix_len(a, b_real) == len(a)
        n, disagree, admitted, refused = self._check_pair(_all_turns_op(), a, b_real)
        assert not disagree and L in refused             # agree; refused today (a complete header precedes L)
        assert la.cached_region_header_conflict(HEADER, b_real[:L], frozenset({la.CONFLICT_BOUNDARY_INSIDE_HEADER})) is None
        # a malformed reader: content follows the terminator instead of the glue, then the turn is
        # closed properly later — the span EXTENDS, the shared terminator stays masked, no disagreement
        b_bad = tuple(list(a[:L]) + [606, 607] + list(TURN_END) + _user_turn(400, 401) + list(HEADER))
        assert _shared_prefix_len(a, b_bad) == L
        ma, mb = _mask_list(_all_turns_op(), a), _mask_list(_all_turns_op(), b_bad)
        assert ma[:L] == mb[:L] and mb[term_at] is True and mb[L] is True and mb[L + 1] is True
        assert la.cached_region_header_conflict(HEADER, b_bad[:L], frozenset({la.CONFLICT_BOUNDARY_INSIDE_HEADER})) is None
        # the shipped set refuses (a complete header precedes the boundary); the mask itself is consistent
        assert la.cached_region_header_conflict(HEADER, b_bad[:L], la.ALL_TURNS_CONFLICT_CLAUSES) is not None

    def test_response_pairs_for_comparison(self):
        """Under response the same writer/reader pairs disagree wherever B
        has a historical turn A steered as final — and response's clause
        set refuses every boundary past the first shared header."""
        a, _ = _conversation(0)
        b, spans_b = _continued(a)
        na, disagree, admitted, refused = self._check_pair(_response_op(), a, b)
        assert disagree == list(range(spans_b[0][0], len(a)))      # A's header rows
        lo = spans_b[0][0]
        # only boundaries before A's header — minus the one ending on the shared opener (conservative)
        assert admitted == [L for L in range(1, lo + 1) if b[L - 1] != OPENER]
        assert max(admitted) == lo

    def test_declared_resumed_reader_locates_over_the_full_prompt_p1_3(self):
        """A DECLARED reader resumed with the cache boundary before,
        across, and after its final header locates that header over the
        FULL prompt, never over the suffix. Includes the collision case:
        headers at absolute 4 and 20, cached = 16 — a suffix-relative
        search returns 4, which lands on the EARLIER header and would mask
        the user rows between the two."""
        ids = [1, 2, 3, 4] + list(HEADER) + [200, 201] + list(TURN_END)     # earlier turn: header@4, scaffold through 10
        ids += list(USER_HEADER) + [300, 301, 302, 303] + list(TURN_END)    # user turn 11..19
        assert len(ids) == 20
        ids += list(HEADER) + [555, 556]                                    # final header @20, seeded
        ids = tuple(ids)
        P = len(ids)
        op = _all_turns_op()
        cold = _mask_list(op, ids, response_prefill=True)
        assert cold == [i in set(range(4, 11)) | set(range(20, P)) for i in range(P)]
        narrow = frozenset({la.CONFLICT_BOUNDARY_INSIDE_HEADER})
        for L in (3, 4, 5, 11, 12, 16, 19, 20, 21, 23, 24):                # before / inside / across / after the final header
            ctx = _resumed_ctx(ids, L, response_prefill=True)
            why = la.cached_region_header_conflict(HEADER, ids[:L], la.ALL_TURNS_CONFLICT_CLAUSES)
            if why is not None:
                with pytest.raises(la.ResumedPrefillRefusal):
                    la._row_selection(op, ctx, _hidden(P - L))
                # the locator itself is exercised through the narrow set (the future admission): a
                # served slice must equal the cold mask's slice — the P1-3 collision would break this
                if la.cached_region_header_conflict(HEADER, ids[:L], narrow) is None:
                    full = ids[:L] + ids[L:]
                    assert la._locate_final_header(op, ctx, full, P) == 20, L
                continue
            assert la._row_selection(op, ctx, _hidden(P - L)).tolist() == cold[L:], L
        # the collision, spelled out: the suffix-relative offset of the final header equals an earlier header's absolute position
        assert la._rfind_header(ids[16:], HEADER) == 4 and ids[4:4 + len(HEADER)] == HEADER
        assert la._locate_final_header(op, _prefill_ctx(ids, response_prefill=True), ids, P) == 20
        # a boundary before any header is served, declared or not (the tail rule applies undeclared)
        assert la._row_selection(op, _resumed_ctx(ids, 3, agent_loop=True), _hidden(P - 3)).tolist() == cold[3:]
        with pytest.raises(la.AdapterError, match="does not end at the header"):
            la._row_selection(op, _resumed_ctx(ids, 3), _hidden(P - 3))


class TestResumedApply:
    """The admitted resume is exactly the tail of the cold apply; the
    boundary inside a header is refused (contained); the payoff boundary
    is admitted where response refuses it."""

    def _shapes(self):
        ids, spans = _conversation(2)
        (lo0, hi0), (lo1, hi1), (lo2, _) = spans
        served = {
            "before-any-header": lo0,                      # the only header-free region shape
            "before-any-header-minus-one": lo0 - 1,
        }
        full = {                                           # a complete earlier header in the region: refused today
            "earlier-header-fully-cached": hi0,
            "boundary-inside-earlier-content": lo0 + len(HEADER) + 1,
            "two-earlier-turns-cached": hi1 + 3,
            "boundary-inside-scaffold": hi0 - 1,
        }
        partial = {                                        # the boundary inside a header: refused by both clauses sets
            "boundary-inside-earlier-header": lo0 + 1,
            "boundary-inside-earlier-header-2": lo0 + 2,
            "boundary-inside-final-header": lo2 + 1,
        }
        return ids, spans, served, full, partial

    @pytest.mark.parametrize("op_name", ["add_vector", "relu_gated"])
    def test_admitted_shapes_equal_the_cold_tail(self, op_name):
        ids, spans, served, _, _ = self._shapes()
        P = len(ids)
        op = _all_turns_op() if op_name == "add_vector" else _gated(ALL)
        h = _hidden(P)
        cold = la.apply_hook_ops(h, (op,), 1.0, layer=0, ctx=_prefill_ctx(ids))
        full_mask = [i in _indices(spans) for i in range(P)]
        for name, L in served.items():
            ctx = _resumed_ctx(ids, L)
            warm = la.apply_hook_ops(h[L:], (op,), 1.0, layer=0, ctx=ctx)
            if op_name == "add_vector":
                assert torch.equal(warm, cold[L:]), name
            else:
                assert torch.allclose(warm, cold[L:], atol=1e-5, rtol=1e-5), name
            sel = la._row_selection(op, ctx, h[L:])
            assert sel.tolist() == full_mask[L:], name
            assert torch.equal(la._row_mask(op, ctx, h[L:]), sel), name

    def test_boundary_inside_a_header_refuses_contained(self):
        """A boundary inside a header refuses. Inside the EARLIER header the
        boundary clause names it; inside the FINAL header a complete earlier
        header precedes it, and the full-occurrence clause (scanned first)
        names that instead — either way, refused and contained."""
        ids, _, _, _, partial = self._shapes()
        P = len(ids)
        for name, L in partial.items():
            with pytest.raises(la.ResumedPrefillRefusal) as ei:
                la.apply_hook_ops(_hidden(P)[L:], (_all_turns_op(),), 1.0, layer=0, ctx=_resumed_ctx(ids, L))
            msg = str(ei.value)
            assert "resumed_prefill_policy is conditional" in msg, name
            if name.startswith("boundary-inside-earlier"):
                assert "INSIDE the assistant header" in msg, name
            else:
                assert "prior writer's response boundary" in msg, name
            # the boundary clause on its own names the header-internal boundary in every case
            assert "INSIDE the assistant header" in la.cached_region_header_conflict(
                HEADER, ids[:L], frozenset({la.CONFLICT_BOUNDARY_INSIDE_HEADER})), name
            with pytest.raises(la.ResumedPrefillRefusal):
                la._row_selection(_all_turns_op(), _resumed_ctx(ids, L), _hidden(P - L))

    def test_full_occurrence_shapes_refuse_under_both_modes_today(self):
        """The held-back payoff, recorded: every shape with a complete
        earlier header in the cached region is refused on the all-turns
        cell exactly as on the response cell — while the all-turns MASK on
        those slices equals the cold mask's slice, which is what the
        narrowing (docs §7) would rely on."""
        ids, spans, _, full, _ = self._shapes()
        P = len(ids)
        h = _hidden(P)
        full_mask = [i in _indices(spans) for i in range(P)]
        for name, L in full.items():
            for op in (_all_turns_op(), _response_op()):
                with pytest.raises(la.ResumedPrefillRefusal, match="prior writer's response boundary"):
                    la.apply_hook_ops(h[L:], (op,), 1.0, layer=0, ctx=_resumed_ctx(ids, L))
            # the mask the narrowing would serve on that slice: the spans located over the FULL prompt
            spans_full = la.assistant_turn_spans(ids, HEADER, TURN_END, spans[-1][0])
            assert [i in _indices(spans_full) for i in range(P)][L:] == full_mask[L:], name

    def test_fully_cached_prompt_last_row(self):
        """vLLM leaves the prompt's last token to be computed: the one-row
        resume of a fully-cached multi-turn prompt has its boundary inside
        the final header AND a complete earlier header in the region —
        refused (the full-occurrence clause names it first); on a
        single-turn prompt the boundary clause names it."""
        ids, _ = _conversation(1)
        P = len(ids)
        with pytest.raises(la.ResumedPrefillRefusal, match="prior writer's response boundary"):
            la._row_selection(_all_turns_op(), _resumed_ctx(ids, P - 1), _hidden(1))
        single, _ = _conversation(0)
        with pytest.raises(la.ResumedPrefillRefusal, match="INSIDE the assistant header"):
            la._row_selection(_all_turns_op(), _resumed_ctx(single, len(single) - 1), _hidden(1))

    def test_no_cached_ids_refuses_never_guesses(self):
        ids, spans = _conversation(1)
        L = spans[0][1]
        with pytest.raises(la.ResumedPrefillRefusal, match="no cached_ids"):
            la.apply_hook_ops(_hidden(len(ids) - L), (_all_turns_op(),), 1.0, layer=0,
                              ctx=_resumed_ctx(ids, L, with_cached=False))

    def test_scale_zero_stays_bit_identical(self):
        ids, spans = _conversation(1)
        L = spans[0][1]
        h = _hidden(len(ids) - L)
        assert la.apply_hook_ops(h, (_all_turns_op(),), 0.0, layer=0, ctx=_resumed_ctx(ids, L)) is h


# ============================================================================
# F — the fit/serve guard
# ============================================================================


HDR3 = (7, 8, 9)
END3 = (99, 11)


def _psr_blob(extra=None) -> bytes:
    layers = (0, 2)
    g = torch.Generator().manual_seed(9)
    blob = {"kind": "psr", "layers": list(layers),
            "directions": {str(l): torch.randn(HIDDEN, generator=g) * 0.05 for l in layers},
            "gate_w": {str(l): torch.randn(HIDDEN, generator=g) * 0.3 for l in layers},
            "gate_b": {str(l): torch.randn(1, generator=g) * 0.5 for l in layers}}
    blob.update(extra or {})
    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue()


def _vector_all_blob(positions) -> bytes:
    buf = io.BytesIO()
    torch.save({"kind": "vector_all", "vectors": {"0": torch.ones(4)},
                "positions": positions, "skip_bos": True}, buf)
    return buf.getvalue()


def _lora_blob(positions) -> bytes:
    g = torch.Generator().manual_seed(4)
    buf = io.BytesIO()
    torch.save({"kind": "lora", "layer": 1, "A": torch.randn(HIDDEN, generator=g),
                "B": torch.randn(HIDDEN, generator=g), "positions": positions,
                "response_header_len": len(HDR3)}, buf)
    return buf.getvalue()


class TestFitServeGuard:
    def test_response_fit_serves_under_both_modes(self):
        for kind, blob in (("vector_all", _vector_all_blob("response")),
                           ("psr", _psr_blob({"positions": "response", "response_header_len": 3})),
                           ("lora", _lora_blob("response"))):
            resp = la.get_adapter(kind)({"kind": kind, "positions": "response"}, blob, b"",
                                        header_ids=HDR3, turn_end_ids=END3)
            assert {o.positions for o in resp} == {RESP}
            assert all(o.header_ids == HDR3 and o.turn_end_ids is None for o in resp)
            allt = la.get_adapter(kind)({"kind": kind, "positions": "response_all_turns"}, blob, b"",
                                        header_ids=HDR3, turn_end_ids=END3)
            assert {o.positions for o in allt} == {ALL}, kind
            assert all(o.header_ids == HDR3 and o.turn_end_ids == END3 for o in allt), kind
            dflt = la.get_adapter(kind)({"kind": kind}, blob, b"", header_ids=HDR3, turn_end_ids=END3)
            assert {o.positions for o in dflt} == {RESP}

    def test_response_fit_under_all_still_refuses_with_the_existing_message(self):
        with pytest.raises(la.AdapterError, match="DECODE-ONLY") as ei:
            la.get_adapter("vector_all")({"kind": "vector_all", "positions": "all"}, _vector_all_blob("response"), b"")
        assert "wider-position approximation" in str(ei.value) and "response_all_turns" in str(ei.value)
        with pytest.raises(la.AdapterError, match="DECODE-ONLY"):
            la.get_adapter("psr")({"kind": "psr", "positions": "all"},
                                  _psr_blob({"positions": "response", "response_header_len": 3}), b"")

    def test_all_fit_under_all_turns_refuses(self):
        with pytest.raises(la.AdapterError, match="KNOWN 'response' fit only"):
            la.get_adapter("vector_all")({"kind": "vector_all", "positions": "response_all_turns"},
                                         _vector_all_blob("all"), b"", header_ids=HDR3, turn_end_ids=END3)
        ops = la.get_adapter("vector_all")({"kind": "vector_all", "positions": "response"},
                                           _vector_all_blob("all"), b"", header_ids=HDR3)
        assert [o.positions for o in ops] == [RESP]

    def test_absent_record_under_all_turns_refuses(self):
        with pytest.raises(la.AdapterError, match="KNOWN 'response' fit only") as ei:
            la.get_adapter("psr")({"kind": "psr", "positions": "response_all_turns"}, _psr_blob(), b"",
                                  header_ids=HDR3, turn_end_ids=END3)
        assert "None" in str(ei.value)
        with pytest.raises(la.AdapterError, match="KNOWN 'response' fit only"):
            la._check_fit_serve_positions(
                "vector_all", {"kind": "vector_all", "positions": "response_all_turns"}, _vector_all_blob("all"), b"",
                (la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=torch.ones(4), positions=ALL,
                           header_ids=HDR3, turn_end_ids=END3),))

    def test_width_cross_check_applies_to_the_all_turns_serve_too(self):
        blob = _psr_blob({"positions": "response", "response_header_len": 4})
        with pytest.raises(la.AdapterError, match="DIFFERENT regions"):
            la.get_adapter("psr")({"kind": "psr", "positions": "response_all_turns"}, blob, b"",
                                  header_ids=HDR3, turn_end_ids=END3)

    def test_all_turns_without_a_scaffold_refuses_at_adapt(self):
        with pytest.raises(la.AdapterError, match="no turn-end scaffold"):
            la.get_adapter("vector_all")({"kind": "vector_all", "positions": "response_all_turns"},
                                         _vector_all_blob("response"), b"", header_ids=HDR3)
        with pytest.raises(la.AdapterError, match="supplied no") as ei:
            la.get_adapter("vector_all")({"kind": "vector_all", "positions": "response_all_turns"},
                                         _vector_all_blob("response"), b"", turn_end_ids=END3)
        assert "header" in str(ei.value)

    def test_stacked_wrapper_threads_the_scaffold(self):
        spec = {"kind": la.STACKED_KIND, "si_id": "si", "si_prefix_sha16": "0" * 16,
                "object": {"kind": "vector_all", "positions": "response_all_turns"}}
        ops = la.get_adapter(la.STACKED_KIND)(spec, _vector_all_blob("response"), b"",
                                              header_ids=HDR3, turn_end_ids=END3)
        assert [o.positions for o in ops] == [ALL] and ops[0].turn_end_ids == END3


# ============================================================================
# G — the worker, the registry, the startup check, the shim
# ============================================================================


def _all_turns_cell(cell_id="cell-all-turns", layer=1):
    return _obj(cell_id, [la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=layer, vector=torch.ones(HIDDEN),
                                    gate_w=torch.ones(HIDDEN), gate_b=torch.zeros(1), positions=ALL,
                                    header_ids=HEADER, turn_end_ids=TURN_END)])


def _response_cell(cell_id="cell-response", layer=1):
    return _obj(cell_id, [la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=layer, vector=torch.ones(HIDDEN),
                                    gate_w=torch.ones(HIDDEN), gate_b=torch.zeros(1), positions=RESP,
                                    header_ids=HEADER)])


def _slots(arm, segs, tokens, prompt_lens, *, extra=None, allow_resumed=True):
    qsl, seq_lens, nat, pos = _forward(segs)
    reg = _registry(arm, *(extra or ()))
    sels = [(_sel(o, scale=(0.0 if o.effective_scale == 0.0 else None)) if o is not None else None)
            for o in [arm] + list(extra or ())][:len(tokens)]
    return lw.build_request_slots(
        query_start_loc=qsl, seq_lens=seq_lens, num_actual_tokens=nat, positions=pos,
        req_ids=list(tokens), raw_selectors=sels,
        token_lookup=tokens.__getitem__, prompt_len_lookup=prompt_lens.__getitem__,
        registry=reg, allow_resumed=allow_resumed)


class TestWorker:
    def test_resume_decision(self):
        ids, spans = _conversation(1)
        P = len(ids)
        L = spans[0][1]                                   # the earlier turn fully cached
        for cell in (_all_turns_cell(), _response_cell()):        # the same admission today
            d = lw.resume_decision(cell, computed=L, q=P - L, seq_len=P, prompt_len=P, tokens=list(ids), req_id="r")
            assert d.kind == lw.RESUME_REFUSE_CONTAINED and d.policy == la.RESUMED_CONDITIONAL
            assert "prior writer's response boundary" in d.reason
        Lf = spans[0][0]                                  # header-free region: served with the cached ids
        d = lw.resume_decision(_all_turns_cell(), computed=Lf, q=P - Lf, seq_len=P, prompt_len=P, tokens=list(ids), req_id="r")
        assert d.kind == lw.RESUME_SERVE and d.policy == la.RESUMED_CONDITIONAL and d.cached_ids == ids[:Lf]
        Lb = spans[0][0] + 1                              # the boundary inside the earlier header
        d3 = lw.resume_decision(_all_turns_cell(), computed=Lb, q=P - Lb, seq_len=P, prompt_len=P, tokens=list(ids), req_id="r")
        assert d3.kind == lw.RESUME_REFUSE_CONTAINED and "INSIDE the assistant header" in d3.reason
        assert isinstance(d3.refusal, la.ResumedPrefillRefusal) and "conditional" in str(d3.refusal)
        plain = _obj("plain", [la.HookOp(op=la.OP_ADD_VECTOR, layer=1, vector=torch.ones(HIDDEN), skip_bos=True)])
        d4 = lw.resume_decision(plain, computed=L, q=P - L, seq_len=P, prompt_len=P, tokens=list(ids), req_id="r")
        assert d4.kind == lw.RESUME_SERVE and d4.policy == la.RESUMED_ALWAYS and d4.cached_ids is None
        with pytest.raises(lw.LensWorkerError, match="behind the cache state"):
            lw.resume_decision(_all_turns_cell(), computed=L, q=P - L, seq_len=P, prompt_len=P, tokens=list(ids[:L - 1]), req_id="r")

    def test_build_request_slots_header_free_served_turn_boundary_contained(self):
        ids, spans = _conversation(1)
        P = len(ids)
        L = spans[0][0]                                   # header-free region (the user turn): served
        q = P - L
        arm = _all_turns_cell()
        fwd = _slots(arm, [(q, L)], {"r": list(ids)}, {"r": P})
        (slot,) = fwd.slots
        assert slot.refusal is None and slot.ctx.is_resumed and slot.ctx.cached == L
        assert slot.ctx.cached_ids == ids[:L]
        assert slot.resumed == lw.ResumedPrefill(cached=L, computed=q)
        h = _hidden(P, dim=HIDDEN)
        cold = la.apply_hook_ops(h, arm.ops, 1.0, layer=1, ctx=_prefill_ctx(ids))
        warm = la.apply_hook_ops(h[L:], arm.ops, 1.0, layer=1, ctx=slot.ctx)
        assert torch.allclose(warm, cold[L:], atol=1e-5, rtol=1e-5)
        # the turn boundary (turn N from turn N-1): CONTAINED on both cells today
        Lt = spans[0][1]
        for cell in (arm, _response_cell()):
            fwd2 = _slots(cell, [(P - Lt, Lt)], {"r": list(ids)}, {"r": P})
            assert isinstance(fwd2.slots[0].refusal, la.ResumedPrefillRefusal) and fwd2.slots[0].ctx is None
            assert "prior writer's response boundary" in str(fwd2.slots[0].refusal)
        # the all-turns cell with the boundary inside the earlier header: contained too
        Lb = spans[0][0] + 1
        fwd3 = _slots(arm, [(P - Lb, Lb)], {"r": list(ids)}, {"r": P})
        assert isinstance(fwd3.slots[0].refusal, la.ResumedPrefillRefusal)
        assert "INSIDE the assistant header" in str(fwd3.slots[0].refusal) and fwd3.slots[0].ctx is None

    def test_registry_round_trip_with_turn_end(self, tmp_path):
        blob = _psr_blob({"positions": "response", "response_header_len": len(HDR3)})
        resolved = {"kind": "psr", "coeff": 1.0, "positions": "response_all_turns"}
        ih = cm.petri_intervention_hash(resolved, artifact_bytes=blob, sidecar_bytes=b"")
        entry = {"cell_id": "arm__c1", "kind": "psr", "resolved_spec": resolved, "expect_hash": ih,
                 "artifact": "arm/artifact.bin", "sidecar": None, "artifact_sha16": cm.sha256_hex16(blob),
                 "artifact_intervention_hash": ih, "forced_scale": 1.0, "donor_cell": None, "validated_by": None}
        for with_end in (True, False):
            reg = tmp_path / ("with" if with_end else "without")
            (reg / "arm").mkdir(parents=True)
            (reg / "arm" / "artifact.bin").write_bytes(blob)
            index = {"version": 1, "assistant_header_ids": list(HDR3), "entries": [entry]}
            if with_end:
                index["assistant_turn_end_ids"] = list(END3)
            (reg / ls.REGISTRY_INDEX).write_text(json.dumps(index))
            if not with_end:
                with pytest.raises(la.AdapterError, match="no turn-end scaffold"):
                    ls.WorkerObjectRegistry.load(reg)
                continue
            loaded = ls.WorkerObjectRegistry.load(reg)
            (obj,) = loaded.objects()
            assert {o.positions for o in obj.ops} == {ALL} and all(o.turn_end_ids == END3 for o in obj.ops)
            assert loaded.assistant_turn_end_ids == END3
            assert loaded.resumed_prefill_policy_by_cell() == {"arm__c1": la.RESUMED_CONDITIONAL}
            assert loaded.resumed_prefill_ok_by_cell() == {"arm__c1": False}
            assert loaded.materialize(device="cpu", dtype=torch.float32).assistant_turn_end_ids == END3

    def test_validated_turn_end_ids(self):
        assert ls._validated_turn_end_ids(None) is None
        assert ls._validated_turn_end_ids([106, 107]) == (106, 107)
        with pytest.raises(ls.LensSteerError, match="assistant_turn_end_ids is empty"):
            ls._validated_turn_end_ids([])
        with pytest.raises(ls.LensSteerError, match="non-negative"):
            ls._validated_turn_end_ids([-1])

    def test_ops_must_express_the_spec_mode(self):
        resp_op, all_op = _response_op(), _all_turns_op()
        ls._assert_ops_express_spec_positions("c", {"resolved_spec": {"positions": "response_all_turns"}}, [all_op])
        ls._assert_ops_express_spec_positions("c", {"resolved_spec": {"positions": "response"}}, [resp_op])
        with pytest.raises(ls.LensSteerError, match="dropped the recorded mask"):
            ls._assert_ops_express_spec_positions("c", {"resolved_spec": {"positions": "response_all_turns"}}, [resp_op])
        with pytest.raises(ls.LensSteerError, match="dropped the recorded mask"):
            ls._assert_ops_express_spec_positions("c", {"resolved_spec": {"positions": "response"}}, [all_op])

    def test_startup_turn_end_verifier(self, monkeypatch):
        from sisteer import chat_markers as cmk
        assert lw.DEFAULT_TURN_END_VERIFIER is ls.verify_assistant_turn_end
        ls.verify_assistant_turn_end(object(), None)          # nothing recorded: the tokenizer is never touched
        monkeypatch.setattr(cmk, "assistant_turn_end_ids", lambda tok: (106, 107))
        monkeypatch.setattr(cmk, "turn_opener_id", lambda tok: 105)
        ls.verify_assistant_turn_end(object(), (106, 107))
        with pytest.raises(ls.LensSteerError, match="turn-end scaffold mismatch"):
            ls.verify_assistant_turn_end(object(), (106,))
        # the opener invariant is verified at startup too
        def bad(tok):
            raise cmk.ChatMarkerError("opens user and model turns differently")
        monkeypatch.setattr(cmk, "turn_opener_id", bad)
        with pytest.raises(ls.LensSteerError, match="turn-opener check failed"):
            ls.verify_assistant_turn_end(object(), (106, 107))
        monkeypatch.setattr(cmk, "assistant_turn_end_ids", lambda tok: None)
        with pytest.raises(ls.LensSteerError, match="successor-bounded"):
            ls.verify_assistant_turn_end(object(), (106, 107))


class TestResponseOnlyStartup:
    """THE FIFTH REVIEW'S REGRESSION, through the production preflight with
    the DEFAULT verifiers (no stubs): a registry serving ONLY
    ``positions="response"`` ops must preflight on every template exactly
    as it did before the all-turns mode existed — with no scaffold
    metadata, AND with stale metadata an older build recorded — including
    templates whose role headers do not share one shape (the fake Nemotron,
    whose think-suffixed assistant header fails the all-turns check; the
    stock tiny templates, where "assistant" splits). The all-turns
    certification is gated on LOADED all-turns ops: the same registries
    with an all-turns op refuse at startup on those templates."""

    def _registry(self, tok, positions, *, stale_scaffold):
        from sisteer.chat_markers import assistant_turn_end_ids, generation_header_ids
        header = tuple(generation_header_ids(tok))
        scaffold = tuple(assistant_turn_end_ids(tok))      # the template's real scaffold (derivable on all three)
        op_kw = dict(op=la.OP_ADD_VECTOR, layer=1, vector=torch.ones(HIDDEN), positions=positions,
                     header_ids=header)
        if positions == ALL:
            op_kw["turn_end_ids"] = scaffold
        obj = _obj("cell", [la.HookOp(**op_kw)])
        return ls.WorkerObjectRegistry({obj.cell_id: obj}, assistant_header_ids=header,
                                       assistant_turn_end_ids=(scaffold if (stale_scaffold or positions == ALL) else None))

    def _preflight(self, tok, reg):
        from test_petri_lens_worker import FakeLayer, FakeView
        view = FakeView(layers=[FakeLayer() for _ in range(4)])
        steering = lw.LensWorkerSteering(view=view, registry=reg)
        steering.preflight(tok)                        # DEFAULT verifiers: the real cross-checks

    def _tokenizers(self):
        from test_generation_header_ids import GEMMA_SHAPE_TEMPLATE, NSR_SHAPE_TEMPLATE, _tiny_tok_with
        from test_nemotron_chat_markers import FakeNemoTok
        return {"nemotron": FakeNemoTok(), "nsr-shape": _tiny_tok_with(NSR_SHAPE_TEMPLATE),
                "gemma-shape": _tiny_tok_with(GEMMA_SHAPE_TEMPLATE)}

    def test_response_only_registries_preflight_on_every_template(self):
        from sisteer import chat_markers as cmk
        for name, tok in self._tokenizers().items():
            with pytest.raises(cmk.ChatMarkerError):     # these templates FAIL the all-turns shape check ...
                cmk.successor_role_header_ids(tok)
            for stale in (False, True):
                reg = self._registry(tok, RESP, stale_scaffold=stale)
                assert not reg.serves_all_turns()
                self._preflight(tok, reg)                 # ... and a response-only registry does not care

    def test_all_turns_registries_refuse_on_those_templates_at_startup(self):
        for name, tok in self._tokenizers().items():
            reg = self._registry(tok, ALL, stale_scaffold=False)
            assert reg.serves_all_turns()
            with pytest.raises(ls.LensSteerError, match="turn-opener check failed for an all-turns registry"):
                self._preflight(tok, reg)             # the scaffold matches; the ROLE-HEADER shape check refuses

    def test_verifier_gating_is_on_loaded_ops_not_metadata(self):
        """The unit of the fix: preflight invokes the turn-end verifier iff
        a loaded op is all-turns, whatever the metadata says."""
        calls = []
        tok = self._tokenizers()["nemotron"]
        from test_petri_lens_worker import FakeLayer, FakeView
        for positions, stale in ((RESP, False), (RESP, True), (ALL, False)):
            reg = self._registry(tok, positions, stale_scaffold=stale)
            steering = lw.LensWorkerSteering(view=FakeView(layers=[FakeLayer() for _ in range(4)]), registry=reg)
            steering.preflight(tok, header_verifier=lambda t, r: None,
                               turn_end_verifier=lambda t, r, p=positions: calls.append((p, r)))
        from sisteer.chat_markers import assistant_turn_end_ids
        assert calls == [(ALL, tuple(assistant_turn_end_ids(tok)))]


def cmk_header(tok):
    from sisteer.chat_markers import generation_header_ids
    return tuple(generation_header_ids(tok))


def _shim_condition(tmp, art_bytes, *, kind, coeff, cell_id="cell0", spec_extra=None):
    art = tmp / f"{cell_id}.pt"
    art.write_bytes(art_bytes)
    spec = {"kind": kind}
    spec.update(spec_extra or {})
    return {"cell_id": cell_id, "artifact_path": str(art), "resolved_spec": spec,
            "coefficient": coeff, "object_sha256": cm.sha256_hex16(art_bytes)}


def _shim_vector_all_blob(positions="response") -> bytes:
    g = torch.Generator().manual_seed(2)
    buf = io.BytesIO()
    torch.save({"kind": "vector_all", "vectors": {"3": torch.randn(6, generator=g)},
                "positions": positions, "skip_bos": True}, buf)
    return buf.getvalue()


class TestShim:
    def test_all_turns_condition_needs_a_scaffold(self, tmp_path):
        cond = _shim_condition(tmp_path, _shim_vector_all_blob(), kind="vector_all", coeff=1.0,
                               spec_extra={"positions": "response_all_turns"})
        with pytest.raises(rs.RegistryShimError, match="assistant_header_ids"):
            rs.build_lens_registry([cond], tmp_path / "reg0")
        with pytest.raises(rs.RegistryShimError, match="assistant_turn_end_ids"):
            rs.build_lens_registry([cond], tmp_path / "reg1", assistant_header_ids=list(HDR3))

    @pytest.mark.parametrize("coeff", [1.0, 0.5])
    def test_all_turns_condition_builds_and_reloads(self, tmp_path, coeff):
        cond = _shim_condition(tmp_path, _shim_vector_all_blob(), kind="vector_all", coeff=coeff,
                               spec_extra={"positions": "response_all_turns"})
        res = rs.build_lens_registry([cond], tmp_path / "reg", assistant_header_ids=list(HDR3),
                                     assistant_turn_end_ids=list(END3))
        assert res.assistant_header_ids == HDR3 and res.assistant_turn_end_ids == END3
        loaded = ls.WorkerObjectRegistry.load(Path(res.registry_dir))
        (obj,) = loaded.objects()
        assert {o.positions for o in obj.ops} == {ALL} and all(o.turn_end_ids == END3 for o in obj.ops)
        index = json.loads((Path(res.registry_dir) / ls.REGISTRY_INDEX).read_text())
        assert index["assistant_turn_end_ids"] == list(END3)

    def test_response_only_registry_records_no_scaffold(self, tmp_path):
        cond = _shim_condition(tmp_path, _shim_vector_all_blob(), kind="vector_all", coeff=1.0,
                               spec_extra={"positions": "response"})
        res = rs.build_lens_registry([cond], tmp_path / "reg", assistant_header_ids=list(HDR3))
        assert res.assistant_turn_end_ids is None
        assert json.loads((Path(res.registry_dir) / ls.REGISTRY_INDEX).read_text())["assistant_turn_end_ids"] is None

    def test_marker_gates_admit_every_all_turns_shape_and_exclude_response_only(self, tmp_path):
        """THE SIXTH REVIEW'S P2 and the discipline it asked for: the scaffold
        gate decides from the mode a condition actually SERVES — for a
        STACKED condition that is the object leg's ``positions`` — so every
        shape that needs the markers gets them, tokenizer-derived, with no
        explicit ids: plain all-turns, stacked-only all-turns, stacked
        all-turns + plain response, plain + stacked all-turns; explicit
        ids still win; and the one excluded shape — response-only, plain
        or stacked — records no scaffold and reloads. Each case builds
        through the real shim and RELOADS through the worker registry
        (where the P2 build failed)."""
        from test_generation_header_ids import _tiny_tok_with
        tok = _tiny_tok_with(EQ_ROLE_TEMPLATE)
        eot = (tok.convert_tokens_to_ids("<|eot_id|>"),)

        def stacked(positions, cell_id):
            spec = {"kind": la.STACKED_KIND, "si_id": "safety", "si_prefix_sha16": "0" * 16,
                    "object": {"kind": "vector_all", "positions": positions}}   # a real SI id: the hash checks it
            return _shim_condition(tmp_path, _shim_vector_all_blob(), kind=la.STACKED_KIND, coeff=1.0,
                                   cell_id=cell_id, spec_extra=spec)

        def plain(positions, cell_id):
            return _shim_condition(tmp_path, _shim_vector_all_blob(), kind="vector_all", coeff=1.0,
                                   cell_id=cell_id, spec_extra={"positions": positions})

        admitted = {
            "plain-all-turns": [plain(ALL, "p_all")],
            "stacked-only-all-turns": [stacked(ALL, "s_all")],
            "stacked-all-turns+plain-response": [stacked(ALL, "s_all"), plain(RESP, "p_resp")],
            "plain+stacked-all-turns": [plain(ALL, "p_all"), stacked(ALL, "s_all")],
            "stacked-all-turns+stacked-response": [stacked(ALL, "s_all"), stacked(RESP, "s_resp")],
        }
        for name, conds in admitted.items():
            res = rs.build_lens_registry(conds, tmp_path / f"reg_{name}", tokenizer=tok)
            assert res.assistant_turn_end_ids == eot, name                 # derived, no explicit ids
            loaded = ls.WorkerObjectRegistry.load(Path(res.registry_dir))   # the P2 failure point
            assert loaded.serves_all_turns(), name
            modes = {c["cell_id"]: {o.positions for o in loaded.select(c["cell_id"], r.selector["expect_hash"]).ops}
                     for c, r in zip(conds, res.conditions)}
            assert modes[conds[0]["cell_id"]] == {ALL}, name
        # explicit ids win, stacked included
        res = rs.build_lens_registry([stacked(ALL, "s_all")], tmp_path / "reg_explicit",
                                     assistant_header_ids=list(cmk_header(tok)), assistant_turn_end_ids=[5])
        assert res.assistant_turn_end_ids == (5,)
        # the excluded shape: response-only, plain or stacked — no scaffold, reloads, serves response
        for name, conds in {"plain-response": [plain(RESP, "p_resp")],
                            "stacked-response": [stacked(RESP, "s_resp")],
                            "stacked+plain-response": [stacked(RESP, "s_resp"), plain(RESP, "p_resp")]}.items():
            res = rs.build_lens_registry(conds, tmp_path / f"reg_{name}", tokenizer=tok)
            assert res.assistant_turn_end_ids is None, name
            loaded = ls.WorkerObjectRegistry.load(Path(res.registry_dir))
            assert not loaded.serves_all_turns(), name
            assert {o.positions for obj in loaded.objects() for o in obj.ops} == {RESP}, name
        # and the header gate unwraps the same way: a stacked response condition with neither ids nor
        # tokenizer refuses at BUILD (not at reload), like a plain one
        with pytest.raises(rs.RegistryShimError, match="assistant_header_ids"):
            rs.build_lens_registry([stacked(RESP, "s_resp")], tmp_path / "reg_nohdr")
        with pytest.raises(rs.RegistryShimError, match="assistant_turn_end_ids"):
            rs.build_lens_registry([stacked(ALL, "s_all")], tmp_path / "reg_noend",
                                   assistant_header_ids=list(cmk_header(tok)))

    def test_response_only_build_from_a_tokenizer_records_no_scaffold(self, tmp_path):
        """The regression's build-side half: with a real tokenizer, a
        response-only build derives the header and NOTHING else — no
        scaffold metadata for a later startup check to trip on."""
        from test_generation_header_ids import _tiny_tok_with
        tok = _tiny_tok_with(EQ_ROLE_TEMPLATE)                      # a template where derivation WOULD succeed
        cond = _shim_condition(tmp_path, _shim_vector_all_blob(), kind="vector_all", coeff=1.0,
                               spec_extra={"positions": "response"})
        res = rs.build_lens_registry([cond], tmp_path / "reg", tokenizer=tok)
        assert res.assistant_header_ids is not None and res.assistant_turn_end_ids is None
        assert json.loads((Path(res.registry_dir) / ls.REGISTRY_INDEX).read_text())["assistant_turn_end_ids"] is None
        # and the same tokenizer DOES yield the scaffold for an all-turns build
        cond2 = _shim_condition(tmp_path, _shim_vector_all_blob(), kind="vector_all", coeff=1.0, cell_id="cell1",
                                spec_extra={"positions": "response_all_turns"})
        res2 = rs.build_lens_registry([cond2], tmp_path / "reg2", tokenizer=tok)
        assert res2.assistant_turn_end_ids == (tok.convert_tokens_to_ids("<|eot_id|>"),)

    def test_scaffold_derived_from_the_tokenizer_or_refused(self, monkeypatch):
        from sisteer import chat_markers as cmk
        conds_all = [{"resolved_spec": {"kind": "vector_all", "positions": "response_all_turns"}}]
        conds_resp = [{"resolved_spec": {"kind": "vector_all", "positions": "response"}}]
        monkeypatch.setattr(cmk, "assistant_turn_end_ids", lambda tok: (106, 107))
        monkeypatch.setattr(cmk, "turn_opener_id", lambda tok: 105)
        assert rs._resolve_turn_end(conds_all, None, object()) == (106, 107)
        assert rs._resolve_turn_end(conds_all, [5], object()) == (5,)
        # a RESPONSE-ONLY build derives and records NOTHING, tokenizer or not (the fifth review's
        # regression: recorded-but-unused scaffold metadata tripped the all-turns startup check)
        def never(tok):
            raise AssertionError("a response-only build must not probe the template for a scaffold")
        monkeypatch.setattr(cmk, "assistant_turn_end_ids", never)
        monkeypatch.setattr(cmk, "turn_opener_id", never)
        assert rs._resolve_turn_end(conds_resp, None, object()) is None
        assert rs._resolve_turn_end(conds_resp, [5], object()) == (5,)        # explicit ids are the caller's
        monkeypatch.setattr(cmk, "assistant_turn_end_ids", lambda tok: (106, 107))
        # the role-header shape is verified at build when an all-turns condition exists ...
        def bad(tok):
            raise cmk.ChatMarkerError("do not share one shape")
        monkeypatch.setattr(cmk, "turn_opener_id", bad)
        with pytest.raises(rs.RegistryShimError, match="turn-opener check"):
            rs._resolve_turn_end(conds_all, None, object())
        monkeypatch.setattr(cmk, "turn_opener_id", lambda tok: 105)
        monkeypatch.setattr(cmk, "assistant_turn_end_ids", lambda tok: None)
        with pytest.raises(rs.RegistryShimError, match="SUCCESSOR-BOUNDED"):
            rs._resolve_turn_end(conds_all, None, object())
        assert rs._resolve_turn_end(conds_resp, None, None) is None
        with pytest.raises(rs.RegistryShimError, match="empty"):
            rs._resolve_turn_end(conds_resp, [], None)


# ============================================================================
# H — chat_markers.assistant_turn_end_ids on synthetic templates
# ============================================================================


#: the synthetic Llama-3.1-shaped template whose role TAGS are single tokens
#: on the tiny tokenizer ("U" / "A"), so its user and assistant headers share
#: one shape the way the real gemma-4 and Llama-3.1 templates' do
EQ_ROLE_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + ('U' if message['role'] == 'user' else 'A') + '<|end_header_id|>' }}"
    "{{ message['content'] }}{{ '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>A<|end_header_id|>' }}{% endif %}")


def _assistant_terminator_index(ids, markers) -> int:
    """The ASSISTANT turn's terminator in a [user, assistant, user] render:
    the first terminator after the located assistant header (the first
    terminator in the render closes the USER turn)."""
    head = tuple(markers.header)
    h = next(i for i in range(len(ids) - len(head) + 1) if tuple(ids[i:i + len(head)]) == head)
    return next(i for i in range(h + len(head), len(ids)) if ids[i] == markers.terminator)


class TestTurnEndProbe:
    def test_nousresearch_shape_drops_the_unconditional_trailing_header(self):
        """The reason the scaffold is LOCATED in a multi-turn probe rather
        than read off ``closing``: on a flag-ignoring template the closing
        scaffold carries the trailing assistant header, which follows only
        the LAST turn of a render — a historical turn is followed by
        [<|eot_id|>, <|start_header_id|>, user ...]."""
        from test_generation_header_ids import NSR_SHAPE_TEMPLATE, _tiny_tok_with
        from sisteer import chat_markers as cmk
        tok = _tiny_tok_with(NSR_SHAPE_TEMPLATE)
        m = cmk.derive_assistant_markers(tok)
        eot = tok.convert_tokens_to_ids("<|eot_id|>")
        assert m.terminator == eot and len(m.closing) > 1          # closing = eot + the trailing header
        end = cmk.assistant_turn_end_ids(tok, m)
        assert end == (eot,)                                        # the scaffold between turns is the eot alone
        assert tuple(m.closing[:len(end)]) == end and end[m.glue] == m.terminator   # closing's head; terminator at glue
        # raw closing never occurs after a HISTORICAL turn on this template — the reason it is located, not copied
        ids = cmk.render_ids(tok, [{"role": "user", "content": "a"}, {"role": "assistant", "content": "bb"},
                                   {"role": "user", "content": "c"}])
        k = _assistant_terminator_index(ids, m)
        assert tuple(ids[k:k + len(m.closing)]) != tuple(m.closing)
        assert tuple(ids[k:k + len(end)]) == end

    def test_gemma_shape_keeps_the_post_terminator_glue(self):
        from test_generation_header_ids import GEMMA_SHAPE_TEMPLATE, _tiny_tok_with
        from sisteer import chat_markers as cmk
        tok = _tiny_tok_with(GEMMA_SHAPE_TEMPLATE)
        m = cmk.derive_assistant_markers(tok)
        end = cmk.assistant_turn_end_ids(tok, m)
        assert end[0] == m.terminator == tok.convert_tokens_to_ids("<|eot_id|>")
        assert len(end) > 1 and end == tuple(m.closing)             # eot + the newline glue, up to the next opener
        assert end[m.glue] == m.terminator
        # and it is exactly what follows a historical assistant turn in a real multi-turn render
        ids = cmk.render_ids(tok, [{"role": "user", "content": "a"}, {"role": "assistant", "content": "bb"},
                                   {"role": "user", "content": "c"}])
        k = _assistant_terminator_index(ids, m)
        assert tuple(ids[k:k + len(end)]) == end
        assert ids[k + len(end)] == tok.convert_tokens_to_ids("<|start_header_id|>")

    def test_successor_bounded_template_has_no_turn_end(self):
        from sisteer import chat_markers as cmk
        m = cmk.AssistantMarkers(header=(1, 2), terminator=None, closing=())
        assert cmk.assistant_turn_end_ids(object(), m) is None       # never touches the tokenizer
        assert cmk.turn_opener_id(object(), m) is None

    def test_successor_role_header_is_located_and_shape_checked(self, monkeypatch):
        """The historical rule validates a successor header by the
        generation header's SHAPE (opener, one role token, tail); the probe
        locates the template's user header and certifies it has that
        shape. On the tiny test tokenizer "assistant" splits into two
        tokens while "user" is one, so the two stock synthetic templates
        REFUSE (roles of unequal shape) — the same refusal Nemotron's
        think-suffixed header and Llama-3.1's two-token ``ipython`` role
        would meet; a template whose role tags are single tokens passes."""
        from test_generation_header_ids import GEMMA_SHAPE_TEMPLATE, NSR_SHAPE_TEMPLATE, _tiny_tok_with
        from sisteer import chat_markers as cmk
        for template in (NSR_SHAPE_TEMPLATE, GEMMA_SHAPE_TEMPLATE):
            tok = _tiny_tok_with(template)
            assert cmk.assistant_turn_end_ids(tok) is not None            # the scaffold is fine ...
            with pytest.raises(cmk.ChatMarkerError, match="do not share one shape"):
                cmk.successor_role_header_ids(tok)                        # ... the role headers are not
            with pytest.raises(cmk.ChatMarkerError, match="do not share one shape"):
                cmk.turn_opener_id(tok)
        tok = _tiny_tok_with(EQ_ROLE_TEMPLATE)
        header = tuple(cmk.generation_header_ids(tok))
        uh = cmk.successor_role_header_ids(tok)
        assert uh is not None and len(uh) == len(header) and uh[0] == header[0] and uh[2:] == header[2:]
        assert uh[1] != header[1]                                          # the role token differs ...
        assert tok.decode([uh[1]]).strip() == "U" and tok.decode([header[1]]).strip() == "A"   # ... and is the role tag
        assert uh[0] == tok.convert_tokens_to_ids("<|start_header_id|>") and uh[2] == tok.convert_tokens_to_ids("<|end_header_id|>")
        assert cmk.turn_opener_id(tok) == header[0]
        real = cmk.generation_header_ids
        monkeypatch.setattr(cmk, "generation_header_ids", lambda t, m=None: (12345,) + tuple(real(t, m))[1:])
        with pytest.raises(cmk.ChatMarkerError, match="opens user and model turns differently"):
            cmk.turn_opener_id(tok)


# ============================================================================
# I — the REAL gemma-4 template (skipped unless the pinned tokenizer is cached)
# ============================================================================

GEMMA = "google/gemma-4-E4B-it"
GEMMA_REV = "ee0ef6023621cff504d758262d4e04895a5af4a2"


def _gemma_tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(GEMMA, revision=GEMMA_REV, local_files_only=True)
    except Exception as e:  # not cached here — skip, never fetch
        pytest.skip(f"pinned {GEMMA} tokenizer not in the local HF cache: {type(e).__name__}")


def _gemma_render(tok, msgs, tools=None):
    ids = tok.apply_chat_template(list(msgs), tokenize=True, tools=tools, add_generation_prompt=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return tuple(int(i) for i in ids)


_G_TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
_G_USER = {"role": "user", "content": "Hi there. Please check the weather."}
_G_TOOL_RESULT = {"role": "tool", "tool_call_id": "call1", "content": "Sunny, 72F."}


def _g_tc(content):
    return {"role": "assistant", "content": content,
            "tool_calls": [{"id": "call1", "type": "function",
                            "function": {"name": "get_weather", "arguments": {"city": "Boston"}}}]}


class TestRealGemma4:
    def test_turn_end_scaffold_is_terminator_plus_glue(self):
        from sisteer.chat_markers import assistant_turn_end_ids, derive_assistant_markers, generation_header_ids
        tok = _gemma_tok()
        m = derive_assistant_markers(tok)
        end = assistant_turn_end_ids(tok, m)
        assert end == (tok.convert_tokens_to_ids("<turn|>"), tok.convert_tokens_to_ids("\n"))
        assert end == tuple(m.closing)                       # on gemma-4 the closing IS the inter-turn scaffold
        assert end[m.glue] == m.terminator and m.glue == 0
        from sisteer.chat_markers import successor_role_header_ids, turn_opener_id
        assert turn_opener_id(tok, m) == tok.convert_tokens_to_ids("<|turn>")
        hdr = tuple(generation_header_ids(tok, m))
        assert successor_role_header_ids(tok, m) == (hdr[0], tok.convert_tokens_to_ids("user"), hdr[2])
        header = tuple(generation_header_ids(tok, m))
        assert header[0] == tok.convert_tokens_to_ids("<|turn>")
        # the shared-opener fact behind P1-1: the same control token opens a user turn
        user_turn = _gemma_render(tok, [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                                        {"role": "user", "content": "c"}])
        opener_positions = [i for i, t in enumerate(user_turn) if t == header[0]]
        assert len(opener_positions) == 4                    # user, model, user, the generating model header
        assert user_turn[opener_positions[0] + 1] != header[1]   # the first is a USER turn's: same opener, other role
        assert user_turn[opener_positions[1] + 1] == header[1]

    def test_three_turn_render_masks_exactly_the_model_turns(self):
        from sisteer.chat_markers import assistant_turn_end_ids, generation_header_ids
        tok = _gemma_tok()
        header = tuple(generation_header_ids(tok))
        end = assistant_turn_end_ids(tok)
        msgs = [{"role": "user", "content": "Hi there"}, {"role": "assistant", "content": "Hello friend"},
                {"role": "user", "content": "Second ask"}, {"role": "assistant", "content": "Second reply"},
                {"role": "user", "content": "Third ask"}]
        ids = _gemma_render(tok, msgs)
        op = la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=torch.ones(D), positions=ALL,
                       header_ids=header, turn_end_ids=end)
        got = _mask_set(op, _prefill_ctx(ids), _hidden(len(ids)))
        # derive the expectation from the template's structure: every model header through its <turn|>\n
        want = set()
        H = len(header)
        for s in [i for i in range(len(ids) - H + 1) if ids[i:i + H] == header]:
            e = s + H
            while e < len(ids) and ids[e:e + len(end)] != end:
                e += 1
            hi = e + len(end) if e < len(ids) else len(ids)
            want |= set(range(s, hi))
        assert got == want
        model_word = tok.convert_tokens_to_ids("model")
        user_word = tok.convert_tokens_to_ids("user")
        assert all(ids[i] != user_word for i in got)          # no user-role token inside
        assert sum(1 for i in got if ids[i] == model_word) == 3

    def test_closed_tail_writer_and_continued_reader_agree_on_the_shared_prefix(self):
        """P1-2 on the REAL template: writer A = the Petri
        ``content_and_tool_response`` shape (ends ``<turn|>\\n``, declared
        agent_loop); reader B = the same conversation with a USER turn
        following. B's render is token-identical to A's whole prompt; A
        steers its glue as part of its final span and B steers it as part
        of the historical span — the masks agree on every shared position,
        so a resume at any admitted boundary is sound."""
        from sisteer.chat_markers import assistant_turn_end_ids, generation_header_ids
        tok = _gemma_tok()
        header = tuple(generation_header_ids(tok))
        end = assistant_turn_end_ids(tok)
        op = la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=torch.ones(D), positions=ALL,
                       header_ids=header, turn_end_ids=end)
        writer_msgs = [_G_USER, _g_tc("Let me check that for you."), _G_TOOL_RESULT]
        a = _gemma_render(tok, writer_msgs, tools=_G_TOOLS)
        assert a[-2:] == end                                  # the closed tail the review named
        b = _gemma_render(tok, writer_msgs + [{"role": "user", "content": "Thanks, and Rome?"}], tools=_G_TOOLS)
        n = _shared_prefix_len(a, b)
        assert n == len(a)                                    # the whole writer prompt, glue included
        ma = _mask_list(op, a, agent_loop=True)
        mb = _mask_list(op, b)
        assert ma == mb[:n]
        assert ma[-1] and ma[-2]                              # glue and terminator steered by both
        # every boundary the predicate admits reproduces B's cold mask on the slice
        for L in range(1, n + 1):
            why = la.cached_region_header_conflict(header, b[:L], la.ALL_TURNS_CONFLICT_CLAUSES)
            if why is None and L < len(b):
                assert la._row_selection(op, _resumed_ctx(b, L), _hidden(len(b) - L)).tolist() == mb[L:], L
        # the boundary at A's prompt end — the turn boundary — is what the masks would allow ...
        assert la.cached_region_header_conflict(header, b[:n], frozenset({la.CONFLICT_BOUNDARY_INSIDE_HEADER})) is None
        # ... and what the shipped admission refuses today, on both modes (a complete header in the region)
        assert la.cached_region_header_conflict(header, b[:n], la.ALL_TURNS_CONFLICT_CLAUSES) is not None
        assert la.cached_region_header_conflict(header, b[:n]) is not None

    def test_open_turn_writers_and_continued_readers_agree(self):
        from sisteer.chat_markers import assistant_turn_end_ids, generation_header_ids
        tok = _gemma_tok()
        header = tuple(generation_header_ids(tok))
        end = assistant_turn_end_ids(tok)
        op = la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=torch.ones(D), positions=ALL,
                       header_ids=header, turn_end_ids=end)
        follow = [{"role": "assistant", "content": "It is sunny."}, {"role": "user", "content": "Thanks."}]
        for msgs in ([_G_USER, _g_tc("")], [_G_USER, _g_tc(""), _G_TOOL_RESULT]):
            a = _gemma_render(tok, msgs, tools=_G_TOOLS)
            b = _gemma_render(tok, msgs + follow, tools=_G_TOOLS)
            n = _shared_prefix_len(a, b)
            assert n == len(a)
            assert _mask_list(op, a, agent_loop=True) == _mask_list(op, b)[:n]

    def test_quoted_delimiter_in_the_generating_turn_refuses_on_the_real_template(self):
        """Finding 2 on the real tokenizer: assistant text containing the
        literal ``<turn|>`` + newline (the tokenizer parses the special
        token) inside an open tool turn — the render is a legitimate
        request, and the generating-turn reading fails closed."""
        from sisteer.chat_markers import assistant_turn_end_ids, generation_header_ids
        tok = _gemma_tok()
        header = tuple(generation_header_ids(tok))
        end = assistant_turn_end_ids(tok)
        op = la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=torch.ones(D), positions=ALL,
                       header_ids=header, turn_end_ids=end)
        quoting = [_G_USER, _g_tc("The delimiter is <turn|>\n and then I continue."), _G_TOOL_RESULT]
        a = _gemma_render(tok, quoting, tools=_G_TOOLS)
        inner = [i for i in range(len(a) - len(end) + 1) if a[i:i + len(end)] == end]
        assert inner and inner[0] + len(end) < len(a)           # a complete scaffold inside, more content after
        with pytest.raises(la.AdapterError, match="complete turn-end scaffold"):
            la._row_selection(op, _prefill_ctx(a, agent_loop=True), _hidden(len(a)))
        # the same turn historical (a user turn follows), submitted COLD: the mirror rule refuses —
        # the quoted scaffold is followed by content, not by the turn opener
        b = _gemma_render(tok, quoting + [{"role": "user", "content": "Thanks."}], tools=_G_TOOLS)
        with pytest.raises(la.AdapterError, match="rather than a complete role header"):
            la._row_selection(op, _prefill_ctx(b), _hidden(len(b)))
        with pytest.raises(la.AdapterError, match="rather than a complete role header"):
            la.apply_hook_ops(_hidden(len(b)), (op,), 1.0, layer=0, ctx=_prefill_ctx(b))

    def test_reviewers_cold_counterexample_on_the_real_template(self):
        """The third review's executed case: a quoted ``<turn|>\n`` in an
        EARLIER plain assistant turn, a user follow-up, submitted cold. The
        render is a legitimate request (the tokenizer parses the literal
        delimiter as the control token); it refuses, and ``response`` still
        serves it unchanged."""
        from sisteer.chat_markers import assistant_turn_end_ids, generation_header_ids
        tok = _gemma_tok()
        header = tuple(generation_header_ids(tok))
        end = assistant_turn_end_ids(tok)
        op = la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=torch.ones(D), positions=ALL,
                       header_ids=header, turn_end_ids=end)
        msgs = [{"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "The delimiter is <turn|>\n and then I continue for a while."},
                {"role": "user", "content": "ok"}]
        ids = _gemma_render(tok, msgs)
        scaffolds = [i for i in range(len(ids) - len(end) + 1) if ids[i:i + len(end)] == end]
        assert len(scaffolds) == 4                                 # user, QUOTED, model, user
        assert ids[-len(header):] == header                        # an ordinary cold render
        with pytest.raises(la.AdapterError, match="rather than a complete role header") as ei:
            la.apply_hook_ops(_hidden(len(ids)), (op,), 1.0, layer=0, ctx=_prefill_ctx(ids))
        assert f"at position {scaffolds[1]}" in str(ei.value)
        resp = la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=torch.ones(D), positions=RESP, header_ids=header)
        assert _mask_list(resp, ids) == [i >= len(ids) - len(header) for i in range(len(ids))]

    def test_scaffold_plus_opener_plus_prose_refuses_on_the_real_template(self):
        """The fourth review's executed bypass on the pinned tokenizer:
        ordinary assistant content explaining the template's tokens, with
        the scaffold immediately followed by the opener and then prose, a
        user follow-up, cold — refused (was: served, 108 rows unsteered)."""
        from sisteer.chat_markers import assistant_turn_end_ids, generation_header_ids, successor_role_header_ids
        tok = _gemma_tok()
        header = tuple(generation_header_ids(tok))
        end = assistant_turn_end_ids(tok)
        assert successor_role_header_ids(tok) == (header[0], tok.convert_tokens_to_ids("user"), header[2])
        op = la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=torch.ones(D), positions=ALL,
                       header_ids=header, turn_end_ids=end)
        text = "the literal boundary tokens are <turn|>\n<|turn> and then prose, not a role header. More follows."
        msgs = [{"role": "user", "content": "Explain the template."}, {"role": "assistant", "content": text},
                {"role": "user", "content": "thanks"}]
        ids = _gemma_render(tok, msgs)
        q = [i for i in range(len(ids) - 2) if ids[i:i + 2] == end and ids[i + 2] == header[0]]
        assert q and ids[q[0] + 3] != tok.convert_tokens_to_ids("user")     # opener then prose, not a role header
        with pytest.raises(la.AdapterError, match="rather than a complete role header"):
            la.apply_hook_ops(_hidden(len(ids)), (op,), 1.0, layer=0, ctx=_prefill_ctx(ids))
        # the same text in the GENERATING (declared) turn: the generating rule refuses
        gen = _gemma_render(tok, [{"role": "user", "content": "Explain."}, _g_tc(text), _G_TOOL_RESULT], tools=_G_TOOLS)
        with pytest.raises(la.AdapterError, match="complete turn-end scaffold"):
            la._row_selection(op, _prefill_ctx(gen, agent_loop=True), _hidden(len(gen)))

    def test_production_matrix_serves_with_message_derived_masks(self):
        """Every production shape still serves, and the served mask equals
        one DERIVED FROM MESSAGE STRUCTURE (prefix renders of the message
        list — the incremental-render machinery chat_markers uses), never
        from the lexical scanner: the three Petri agent-loop shapes
        (including the closed tail), ordinary single- and multi-turn
        renders, the declared seeded render, and a bare terminator without
        its glue in an earlier and in the generating turn."""
        from sisteer.chat_markers import assistant_turn_end_ids, generation_header_ids
        tok = _gemma_tok()
        header = tuple(generation_header_ids(tok))
        H = len(header)
        end = assistant_turn_end_ids(tok)
        op = la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=torch.ones(D), positions=ALL,
                       header_ids=header, turn_end_ids=end)

        def expected(msgs, tools, ids, *, open_final):
            """Model turns = maximal runs of assistant/tool messages; each
            closed run spans [header start .. end of its render]; an open
            final run (or the generation prompt) spans to the prompt end."""
            def R(k, agp):
                out = tok.apply_chat_template(list(msgs[:k]), tokenize=True, tools=tools, add_generation_prompt=agp)
                if hasattr(out, "keys"):
                    out = out["input_ids"]
                if out and isinstance(out[0], list):
                    out = out[0]
                return tuple(int(i) for i in out)
            want = set()
            runs, i = [], 0
            while i < len(msgs):
                if msgs[i]["role"] in ("assistant", "tool"):
                    j = i
                    while j + 1 < len(msgs) and msgs[j + 1]["role"] in ("assistant", "tool"):
                        j += 1
                    runs.append((i, j))
                    i = j + 1
                else:
                    i += 1
            for i, j in runs:
                pre = R(i, True)
                assert ids[:len(pre)] == pre, "prefix render not a prefix of the full render"
                lo = len(pre) - H
                assert ids[lo:lo + H] == header
                if j == len(msgs) - 1 and open_final:
                    hi = len(ids)
                else:
                    post = R(j + 1, False)
                    assert ids[:len(post)] == post
                    hi = len(post)
                want |= set(range(lo, hi))
            if not (runs and runs[-1][1] == len(msgs) - 1 and open_final):
                want |= set(range(len(ids) - H, len(ids)))        # the generation prompt
            return want

        cases = [
            ("pending_tool_call", [_G_USER, _g_tc("")], _G_TOOLS, {"agent_loop": True}),
            ("tool_response_continuation", [_G_USER, _g_tc(""), _G_TOOL_RESULT], _G_TOOLS, {"agent_loop": True}),
            ("content_and_tool_response", [_G_USER, _g_tc("Let me check that for you."), _G_TOOL_RESULT], _G_TOOLS,
             {"agent_loop": True}),
            ("single_turn", [{"role": "system", "content": "Be safe."}, _G_USER], None, {}),
            ("multi_turn", [_G_USER, {"role": "assistant", "content": "Hello."}, {"role": "user", "content": "More"},
                            {"role": "assistant", "content": "Sure."}, {"role": "user", "content": "Go on"}], None, {}),
            ("bare_terminator_earlier", [_G_USER, {"role": "assistant", "content": "a bare <turn|>x mid-sentence"},
                                          {"role": "user", "content": "ok"}], None, {}),
            ("bare_terminator_final", [_G_USER, _g_tc("a bare <turn|>x mid-sentence"), _G_TOOL_RESULT], _G_TOOLS,
             {"agent_loop": True}),
        ]
        for name, msgs, tools, facts in cases:
            ids = _gemma_render(tok, msgs, tools=tools)
            open_final = bool(facts)
            want = expected(msgs, tools, ids, open_final=open_final)
            got = _mask_set(op, _prefill_ctx(ids, **facts), _hidden(len(ids)))
            assert got == want, name
        # the declared seeded render: the generation prompt + a seed, response_prefill
        base = _gemma_render(tok, [_G_USER])
        seed = tuple(tok.encode(" Sure, here is", add_special_tokens=False))
        ids = base + seed
        got = _mask_set(op, _prefill_ctx(ids, response_prefill=True), _hidden(len(ids)))
        assert got == set(range(len(base) - H, len(ids)))

    def test_a_merged_continuation_shares_no_glue(self):
        """The template MERGES a follow-up assistant message into the same
        model turn (probed 2026-09-07): the writer's ``<turn|>\\n`` is then
        never a shared position, so that continuation cannot resume the
        glue block at all — the hazard needs a USER turn to follow."""
        tok = _gemma_tok()
        writer_msgs = [_G_USER, _g_tc("Let me check that for you."), _G_TOOL_RESULT]
        a = _gemma_render(tok, writer_msgs, tools=_G_TOOLS)
        b = _gemma_render(tok, writer_msgs + [{"role": "assistant", "content": "It is sunny."}], tools=_G_TOOLS)
        assert _shared_prefix_len(a, b) == len(a) - 2


# ============================================================================
# J — the span arithmetic moved to the
# torch-free span_locator (one implementation, re-exported here), the named
# all-turns experiment binds to the loaded ops, and thinking never combines
# with the all-turns mask
# ============================================================================


class TestSpanLocatorReexports:
    def test_one_implementation(self):
        from sisteer.capeval.serving import span_locator as sl
        assert la._find_all is sl.find_all and la._find_first is sl.find_first
        assert la._rfind_header is sl.rfind
        assert la.assistant_turn_spans.__wrapped__ is sl.assistant_turn_spans
        ids, spans = _conversation(2)
        assert la.assistant_turn_spans(ids, HEADER, TURN_END, spans[-1][0]) == \
            sl.assistant_turn_spans(ids, HEADER, TURN_END, spans[-1][0]) == spans
        assert sl.locate_final_header(ids, HEADER, declared=False) == spans[-1][0]

    def test_the_wrapper_keeps_the_adapter_error_and_its_message(self):
        from sisteer.capeval.serving import span_locator as sl
        ids, spans = _conversation(1)
        quoted = list(ids[:spans[-1][0] + len(HEADER)]) + [201, TERM, GLUE, 3]  # scaffold + content in the final turn
        with pytest.raises(sl.SpanLocatorError) as e_sl:
            sl.assistant_turn_spans(quoted, HEADER, TURN_END, spans[-1][0])
        with pytest.raises(la.AdapterError, match="complete turn-end scaffold") as e_la:
            la.assistant_turn_spans(quoted, HEADER, TURN_END, spans[-1][0])
        assert str(e_la.value) == str(e_sl.value)
        assert e_sl.value.reason == sl.REASON_QUOTED_SCAFFOLD_IN_FINAL

    def test_the_named_all_turns_experiment_binds_to_the_ops(self):
        from sisteer.capeval.serving import serve_declaration as sd
        spec = sd.apply_variant({"kind": "psr", "positions": "response"}, "all_turns")
        raw = {"resolved_spec": spec}
        assert ls._assert_ops_express_declarations("c", raw, [_all_turns_op()]) == \
            {"serve_experiment": "fit_response_serve_all_turns"}
        with pytest.raises(ls.LensSteerError, match="fit_response_serve_all_turns"):
            ls._assert_ops_express_declarations("c", raw, [_response_op()])
        # thinking never combines with the all-turns mask
        with pytest.raises(la.AdapterError, match="never combined"):
            la.attach_thinking_header(
                [_all_turns_op()], sd.apply_variant({"kind": "psr", "positions": "response"}, "thinking"),
                (OPENER, 901, 950))
        # an all-turns registry serves no thinking-ON cell
        reg = _registry(_obj("c", [_all_turns_op()]), header=HEADER)
        assert reg.serves_all_turns() and not reg.serves_thinking_on()
        assert reg.serve_declarations_by_cell() == {}
