"""Pure, torch-free SPAN LOCATION over a rendered token-id sequence — the
one implementation behind the lens adapters' response / all-turns masks
AND the proxy's per-request render check.

The hook side (``lens_adapters``) locates a request's assistant-turn
spans in the ENGINE's token ids and masks them; the proxy side has the
MESSAGES and the served tokenizer, and can recompute the same spans on
the same ids to check them against the message-derived expectation
BEFORE the request is forwarded. Both must run the SAME arithmetic, or
the check would compare one locator against another. So the arithmetic
lives here, importable where torch is not (the proxy process, the bare
launch venvs), and ``lens_adapters`` re-exports these functions under its
historical names: the ``_find_all`` / ``_find_first`` / ``_rfind_header``
helpers are these objects, and ``lens_adapters.assistant_turn_spans`` /
``_locate_final_header`` wrap the two locators below to raise the
adapters' ``AdapterError`` with their own verbatim per-op messages
(``tests/test_all_turns_render_check.py`` pins the identity and the
wrapping, so there is one implementation, never two).

Every refusal is a :class:`SpanLocatorError` carrying a ``reason`` code so
a caller can map it onto its own vocabulary without parsing prose.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

#: :attr:`SpanLocatorError.reason` codes of :func:`locate_final_header`.
REASON_PROMPT_SHORTER_THAN_HEADER = "prompt_shorter_than_header"
REASON_HEADER_NOWHERE = "header_nowhere"
REASON_NOT_AT_TAIL = "not_at_tail"
#: ... and of :func:`assistant_turn_spans`.
REASON_BAD_MARKERS = "bad_markers"
REASON_FINAL_NOT_A_HEADER = "final_not_a_header"
REASON_TURN_UNTERMINATED = "turn_unterminated"
REASON_SUCCESSOR_UNCERTIFIED = "successor_uncertified"
REASON_QUOTED_SCAFFOLD_IN_FINAL = "quoted_scaffold_in_final"


class SpanLocatorError(ValueError):
    """A span-location refusal: the render and the marker ids disagree
    about where an assistant turn starts or ends. ``reason`` is one of the
    ``REASON_*`` codes; ``detail`` carries the numbers the message names
    (so a wrapper can re-render the message in its own words)."""

    def __init__(self, message: str, *, reason: str, **detail):
        super().__init__(message)
        self.reason = reason
        self.detail = dict(detail)


def rfind(ids: Sequence[int], needle: Sequence[int]) -> int:
    """Index of the LAST occurrence of ``needle`` in ``ids``, or -1 —
    ``parity._rfind_subseq``'s semantics, scanned from the tail (the
    header sits just before a short seed, so the scan ends in a few
    steps on every real seeded render)."""
    want = tuple(int(t) for t in needle)
    n, m = len(ids), len(want)
    for start in range(n - m, -1, -1):
        if tuple(int(t) for t in ids[start:start + m]) == want:
            return start
    return -1


def find_all(ids: Sequence[int], needle: Sequence[int]) -> Tuple[int, ...]:
    """Start index of every NON-OVERLAPPING occurrence of ``needle`` in
    ``ids``, scanned left to right."""
    want = tuple(int(t) for t in needle)
    n, m = len(ids), len(want)
    out: List[int] = []
    i = 0
    while i <= n - m:
        if tuple(int(t) for t in ids[i:i + m]) == want:
            out.append(i)
            i += m
        else:
            i += 1
    return tuple(out)


def find_first(ids: Sequence[int], needle: Sequence[int], lo: int, hi: int
               ) -> int:
    """Start index of the first occurrence of ``needle`` lying ENTIRELY
    inside ``ids[lo:hi]``, or -1."""
    want = tuple(int(t) for t in needle)
    m = len(want)
    for start in range(lo, hi - m + 1):
        if tuple(int(t) for t in ids[start:start + m]) == want:
            return start
    return -1


def locate_final_header(ids: Sequence[int], header: Sequence[int], *,
                        declared: bool) -> int:
    """The start of the GENERATING turn's assistant header in ``ids`` —
    the response mask's boundary rules, in order: a header longer than the
    prompt refuses; a DECLARED non-tail render (seeded / agent-loop) takes
    the LAST header occurrence and refuses when none exists; an ordinary
    render must END at the header and refuses otherwise. Returns the
    header's start index."""
    hdr = tuple(int(t) for t in header)
    H, T = len(hdr), len(ids)
    if H == 0:
        raise SpanLocatorError(
            "locate_final_header needs a non-empty header id sequence",
            reason=REASON_BAD_MARKERS)
    if H > T:
        raise SpanLocatorError(
            f"the assistant header is {H} tokens but the prompt is only {T} "
            "— a prompt shorter than its own header means the render and the "
            "served chat template disagree",
            reason=REASON_PROMPT_SHORTER_THAN_HEADER, H=H, T=T)
    if declared:
        start = rfind(ids, hdr)
        if start < 0:
            raise SpanLocatorError(
                f"the request declares a non-tail render but the assistant "
                f"header ids {list(hdr)} occur NOWHERE in its prompt tokens",
                reason=REASON_HEADER_NOWHERE, H=H, T=T)
        return start
    start = T - H
    got = tuple(int(t) for t in ids[start:])
    if got != hdr:
        raise SpanLocatorError(
            f"the last {H} prompt tokens are {list(got)}, not the assistant "
            f"header {list(hdr)} — this prompt does not end at the header",
            reason=REASON_NOT_AT_TAIL, H=H, T=T, got=got)
    return start


def assistant_turn_spans(ids: Sequence[int], header_ids: Sequence[int],
                         turn_end_ids: Sequence[int], final_start: int
                         ) -> Tuple[Tuple[int, int], ...]:
    """Every assistant-turn span ``(lo, hi)`` (``hi`` exclusive) of a FULL
    prompt ``ids`` under THE ALL-TURNS MASK (``lens_adapters`` module
    docstring): for each assistant-header occurrence before ``final_start``
    (the generating turn's header, located by :func:`locate_final_header`)
    the span runs from the header THROUGH the first occurrence of the
    turn-END SCAFFOLD after it — the turn's own terminator plus the glue
    the template puts after it (gemma-4 ``<turn|>\\n``;
    ``sisteer.chat_markers.assistant_turn_end_ids``) — and the generating
    turn's span runs from ``final_start`` to the end of the prompt (open:
    every generated position is masked). An earlier header that is not
    closed by the scaffold before the next header REFUSES; a complete
    scaffold followed by anything but the next header or a COMPLETE role
    header of the template's certified shape (opener, one role token, the
    assistant header's tail) REFUSES; a complete scaffold inside the
    GENERATING turn followed by more content REFUSES. The rules and their
    reasons are stated at length on the ``lens_adapters`` wrapper of this
    function (THE RESIDUAL LIMIT included); this is the arithmetic."""
    header = tuple(int(t) for t in header_ids)
    scaffold = tuple(int(t) for t in turn_end_ids)
    H, K = len(header), len(scaffold)
    if H == 0 or K == 0:
        raise SpanLocatorError(
            "assistant_turn_spans needs non-empty header and turn-end id "
            "sequences", reason=REASON_BAD_MARKERS)
    P = len(ids)
    if not 0 <= final_start <= P - H \
            or tuple(int(t) for t in ids[final_start:final_start + H]) != header:
        raise SpanLocatorError(
            f"assistant_turn_spans: position {final_start} of a {P}-token "
            f"prompt does not hold the {H}-token assistant header — the "
            "generating turn's boundary must be LOCATED first "
            "(_locate_final_header); refusing",
            reason=REASON_FINAL_NOT_A_HEADER, final_start=final_start, P=P, H=H)
    # a genuine successor turn begins with a complete ROLE HEADER of the
    # template's one shape: the opener, exactly one role token, then the
    # assistant header's tail — certified for the template's user header
    # at registry build and worker startup
    # (chat_markers.successor_role_header_ids); the assistant header itself
    # has it by construction (consecutive model turns)
    opener, tail = header[0], header[2:]
    earlier = [st for st in find_all(ids, header) if st < final_start]
    spans: List[Tuple[int, int]] = []
    for i, st in enumerate(earlier):
        nxt = earlier[i + 1] if i + 1 < len(earlier) else final_start
        e = find_first(ids, scaffold, st + H, nxt)
        if e < 0:
            raise SpanLocatorError(
                f"the assistant turn opened by the header at position {st} "
                f"is not closed by the turn-end scaffold {list(scaffold)} "
                f"before the next assistant header at position {nxt} — the "
                "render and the served chat template disagree about where "
                "assistant turns end (an unterminated earlier turn, or "
                "scaffold ids from a different template). Refusing rather "
                "than guessing the span's end, which would steer the user's "
                "turn", reason=REASON_TURN_UNTERMINATED, start=st, next=nxt)
        after = e + K
        # FAIL CLOSED on a quoted delimiter in a HISTORICAL turn — the
        # mirror of the generating-turn rule below: after a genuine
        # scaffold comes the next header itself, or a COMPLETE role header
        # of the template's shape (the next user / model turn begins). An
        # opener token alone is not proof of a turn — "the tokens are
        # <turn|>\n<|turn> and then prose" passed an opener-only rule and
        # left 108 rows unsteered (the fourth review). Anything else there
        # means the scaffold at ``e`` was quoted inside the assistant's
        # text and did not end the turn; trusting it would leave the rest
        # of that turn unsteered while an eval records a successful serve.
        if after != nxt:
            found = tuple(int(t) for t in ids[after:after + H])
            genuine = (after + H <= nxt and len(found) == H
                       and found[0] == opener and found[2:] == tail)
            if not genuine:
                raise SpanLocatorError(
                    f"the assistant turn opened by the header at position "
                    f"{st} is followed, after a complete turn-end scaffold "
                    f"{list(scaffold)} at position {e}, by tokens "
                    f"{list(found)} at position {after} rather than a "
                    f"complete role header of the certified shape (opener "
                    f"{opener}, one role token, tail {list(tail)}) or the "
                    "next assistant header — a successor header this checker "
                    "cannot certify. Two shapes produce this and the token "
                    "stream cannot tell them apart: content quoting the "
                    "template's own turn delimiter (the scaffold did not end "
                    "the turn, and reading it as the end would leave the "
                    "rest of the assistant's turn unsteered), or a genuine "
                    "turn of a role whose header has another shape (a "
                    "multi-token role such as Llama-3.1's ipython), which "
                    "this mode's certification covers only for the user "
                    "header and so can only refuse here, at apply. Refusing "
                    "rather than trusting a boundary the stream cannot vouch "
                    "for; the mode is EXPERIMENTAL and UNSUPPORTED until "
                    "boundaries are declared from the request's message structure",
                    reason=REASON_SUCCESSOR_UNCERTIFIED, start=st,
                    scaffold_at=e, after=after)
        spans.append((st, after))
    # FAIL CLOSED on a quoted delimiter in the GENERATING turn: a complete
    # scaffold inside its content, followed by more content. As a
    # historical turn the same tokens would end the span at that scaffold
    # (the loop above), while the final span runs to the prompt end — two
    # readings of one token sequence. The lexical locator cannot tell a
    # quoted scaffold from a real one, so the only reading that is not
    # silently wrong is a refusal. (A scaffold exactly at the prompt end is
    # the closed-tail render and both readings agree.)
    quoted = find_first(ids, scaffold, final_start + H, P)
    if quoted >= 0 and quoted + K < P:
        raise SpanLocatorError(
            f"the generating assistant turn (header at position {final_start}) "
            f"contains a complete turn-end scaffold {list(scaffold)} at "
            f"position {quoted} followed by {P - quoted - K} more prompt "
            "token(s) — content quoting the template's own turn delimiter, or "
            "a render whose final message is not the open turn. Read as the "
            "final turn its span would run to the prompt end; read as a "
            "historical turn the same tokens would end it at that scaffold, "
            "so the mask would depend on whether the turn is final. Refusing "
            "rather than choosing one reading silently",
            reason=REASON_QUOTED_SCAFFOLD_IN_FINAL, final_start=final_start,
            scaffold_at=quoted, P=P)
    spans.append((final_start, P))
    return tuple(spans)


def expected_all_turns_spans(content_spans: Sequence[Tuple[int, int]], *,
                             header_len: int, turn_end_len: int,
                             final_start: int, prompt_len: int
                             ) -> Tuple[Tuple[int, int], ...]:
    """The MESSAGE-DERIVED all-turns spans: each
    assistant message's content span ``(lo, hi)`` from
    ``chat_markers.assistant_content_spans`` widened to header + content +
    end scaffold — ``(lo - header_len, hi + turn_end_len)``, the all-turns
    mask's own definition of a historical span — plus the generating turn
    ``(final_start, prompt_len)``. Pure arithmetic; the caller supplies the
    spans and the marker widths."""
    out = [(int(lo) - int(header_len), int(hi) + int(turn_end_len))
           for lo, hi in content_spans]
    out.append((int(final_start), int(prompt_len)))
    return tuple(out)


__all__ = [
    "SpanLocatorError", "rfind", "find_all", "find_first",
    "locate_final_header", "assistant_turn_spans", "expected_all_turns_spans",
    "REASON_PROMPT_SHORTER_THAN_HEADER", "REASON_HEADER_NOWHERE",
    "REASON_NOT_AT_TAIL", "REASON_BAD_MARKERS", "REASON_FINAL_NOT_A_HEADER",
    "REASON_TURN_UNTERMINATED", "REASON_SUCCESSOR_UNCERTIFIED",
    "REASON_QUOTED_SCAFFOLD_IN_FINAL",
]
