"""Marker-LOCATED assistant-content spans in tokenized chat renders.

The reference implementation for finding where assistant CONTENT lives in
``apply_chat_template`` output. Three lanes independently hit the same
template trap and grew three private workarounds (a header-length H=4
derivation, an H==0 hard error, and tier0's suffix arithmetic — the one
that silently corrupted); this module replaces all of them. Import from
here rather than adding a fourth implementation::

    from sisteer.chat_markers import assistant_content_spans

The rule this module encodes (converged on by the tier0, jailbreak and
train/test lanes, 2026-08-21):

    **Derive template structure by LOCATING it in a render — never by
    reading it off a flag's effect. Where two renders must be compared,
    assert the difference is non-zero rather than letting 0 flow onward
    as a valid answer, and verify the extracted structure round-trips.**

The trap, measured on 2026-08-21 (transformers 5.14.1): the
``NousResearch/Meta-Llama-3.1-8B-Instruct`` template @ d10aef79 appends
``<|start_header_id|>assistant<|end_header_id|>\\n\\n`` UNCONDITIONALLY at
the end of every render — ``add_generation_prompt`` is never consulted (the
name does not appear in its 348-char template). Two consequences:

* a probe that infers the header by DIFFING the ``add_generation_prompt``
  True/False renders measures 0 added tokens on exactly the template where
  the header is ALWAYS present, and "confirms" the flag is ignored — a diff
  artifact, not the flag's behaviour (both renders are identical);
* suffix arithmetic that takes ``render(msgs[:k+1])`` minus the
  ``render(msgs[:k], add_generation_prompt=True)`` prefix, then strips
  trailing SPECIAL tokens, places the whole trailing scaffold — the
  terminal ``<|eot_id|>`` PLUS the unconditional trailing header — inside
  the assistant CONTENT span (the strip halts at the header's trailing
  newline token, which is not special). The content comes first and the
  scaffold trails it, so the prefix boundary is right and the SUFFIX
  boundary is what corrupts.

This module instead derives, once per tokenizer, the markers bounding
assistant content from a content-substitution probe
(:func:`derive_assistant_markers`):

* ``closing`` — the content-independent scaffold from the end of assistant
  content to the end of a render whose final message is an assistant turn
  (meta-llama Llama-3.1: ``[<|eot_id|>]``; NousResearch: ``[<|eot_id|>]``
  plus the unconditional trailing header; Llama-2-style templates:
  a glue space token then ``</s>``);
* ``terminator`` — the first CONTROL token inside ``closing``, the token
  that actually closes the turn;
* ``header`` — the token sequence introducing an assistant turn, when the
  template structure lets it be located (Llama-3.1 mirrors:
  ``[<|start_header_id|>, assistant, <|end_header_id|>, ĊĊ]``, H=4 —
  matching the 4-token ``add_generation_prompt`` diff on the meta-llama
  template, so marker location reproduces what a conventional template
  reports rather than working around it).

Some templates have NO closing scaffold at all — GLM-4.x chat renders an
assistant turn as ``<|assistant|>\\n<think></think>\\n{content}`` and then
nothing: generation stops on the model's EOS set (which contains the NEXT
turn's role marker ``<|user|>``), and in a multi-turn render the next
turn's opening control token is the only bound on the content. These
SUCCESSOR-BOUNDED templates get ``closing == ()`` and ``terminator is
None`` — accepted only after a second probe proves that appending a
follow-up user turn (a) leaves every token through the content end
untouched and (b) puts a control token immediately at the content end.
Anything weaker keeps the hard refusal.

:func:`assistant_content_spans` then verifies, for every assistant message:
the generation-prompt render is a token prefix of the message render (so
that boundary sits exactly at this message's content start — the probe
proved only that the generation prompt lands INSIDE the header, which is
strictly weaker on a template whose generation prompt stops short), the
``header`` marker sits immediately before the span when one was derivable,
the span ends exactly where the probe-derived ``closing`` scaffold begins
(so the terminal EOT and any unconditional trailing header can never leak
into it), the full-conversation render agrees with the incremental render
through the terminator, and no control token survives inside the span — so
a span can never cross a turn boundary. Every violated invariant raises
:class:`ChatMarkerError` — fail loudly, never guess.

Note on "special" tokens: ``tok.all_special_ids`` on the Llama-3.1 mirrors
is only ``[<|begin_of_text|>, <|eot_id|>]``. The header tokens
(``<|start_header_id|>`` etc.) are added CONTROL tokens
(``added_tokens_decoder[id].special is True``) but are NOT special ids, so
any guard built on ``all_special_ids`` alone waves scaffold through — the
measured empty-assistant corruption was four pure-scaffold positions
containing zero special ids. Use :func:`control_token_ids`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


class ChatMarkerError(ValueError):
    """A chat-template structure invariant failed; refusing to guess."""


# --------------------------------------------------------------------------
# render + token helpers
# --------------------------------------------------------------------------


def render_ids(tok, messages: Sequence[Mapping[str, str]],
               add_generation_prompt: bool = False) -> List[int]:
    """``apply_chat_template`` normalized to a plain ``list[int]`` across
    transformers versions (4.x list, 5.x BatchEncoding / nested lists)."""
    out = tok.apply_chat_template(
        list(messages), tokenize=True,
        add_generation_prompt=add_generation_prompt,
        **getattr(tok, "_sisteer_template_kwargs", {}),
    )
    if hasattr(out, "keys"):  # transformers>=5 returns a BatchEncoding
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return [int(i) for i in out]


def divergence(a: Sequence[int], b: Sequence[int]) -> int:
    """Index of the first differing token (min length if one is a prefix)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def control_token_ids(tok) -> frozenset:
    """Every token id that is template scaffold rather than content:
    ``all_special_ids`` UNION the added tokens flagged special
    (``added_tokens_decoder``). On the Llama-3.1 mirrors the union is what
    catches ``<|start_header_id|>``/``<|end_header_id|>``/role headers —
    none of which appear in ``all_special_ids``."""
    ids = set(getattr(tok, "all_special_ids", None) or [])
    added = getattr(tok, "added_tokens_decoder", None) or {}
    for tid, added_tok in added.items():
        if getattr(added_tok, "special", False):
            ids.add(int(tid))
    return frozenset(ids)


# --------------------------------------------------------------------------
# marker derivation (the content-substitution probe)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AssistantMarkers:
    """Probe-derived markers bounding assistant content in a render.

    ``closing``: the content-independent scaffold from the end of assistant
    content to the end of a render whose final message is an assistant turn
    — including any non-control glue before the turn terminator and any
    scaffold the template appends unconditionally after it. ``terminator``:
    the first control token inside ``closing``, the token that closes the
    turn. ``header``: token ids introducing an assistant turn (immediately
    before content), or ``()`` when the template structure does not let one
    be located (no terminator occurrence precedes the assistant turn, as in
    Llama-2-style ``[INST]`` templates whose user turns are unterminated).

    Successor-bounded templates (GLM-4.x: assistant turns end bare, bounded
    only by the next turn's opening control token or the end of the render)
    carry ``closing == ()`` and ``terminator is None`` — see the module
    docstring; :func:`derive_assistant_markers` grants that shape only
    after the successor probe verifies it.

    ``generation_header``: the ids ``add_generation_prompt`` APPENDS — the
    serve-time response boundary's id vocabulary, derived and cross-checked
    by the same probe (:func:`_appended_generation_header`) so there is one
    derivation rather than two. ``()`` means the template appends nothing
    because it emits the header unconditionally (the NousResearch mirror),
    where :func:`generation_header_ids` serves the located ``header``
    instead; ``None`` marks markers built by hand rather than derived."""

    header: Tuple[int, ...]
    terminator: Optional[int]
    closing: Tuple[int, ...]
    generation_header: Optional[Tuple[int, ...]] = None

    @property
    def glue(self) -> int:
        """Tokens of non-control glue in ``closing`` before ``terminator``
        (0 on a successor-bounded template — there is no closing scaffold)."""
        if self.terminator is None:
            return 0
        return self.closing.index(self.terminator)


_PROBE_USER = "Hi"
_SENTINEL_A = "Quartz zebra ostrich"
_SENTINEL_B = "Marble falcon indigo"


def _render_regime_declared(tok) -> bool:
    """Whether this tokenizer's chat-template render regime is PINNED —
    the same seam :func:`render_ids` renders through and
    ``models.pin_reasoning_template_off`` writes
    (``tok._sisteer_template_kwargs``)."""
    return bool(getattr(tok, "_sisteer_template_kwargs", None))


def _reasoning_template(tok) -> bool:
    """Whether the chat template branches on a thinking switch — the
    IDENTICAL signal ``models.pin_reasoning_template_off`` keys on, kept
    inline because ``models`` imports THIS module (never the reverse)."""
    return "enable_thinking" in (getattr(tok, "chat_template", None) or "")


def _appended_generation_header(tok, head: Sequence[int], pre: Sequence[int],
                                full_a: Sequence[int], s: int,
                                ) -> Tuple[int, ...]:
    """The ids ``add_generation_prompt`` appends, cross-checked against the
    assistant-turn scaffold the content-substitution probe located.

    ``head`` / ``pre`` are the probe's user turn rendered WITHOUT / WITH the
    generation prompt; ``full_a`` is the probe render whose assistant
    content begins at token ``s``. Both scaffolds here are HEAD-ANCHORED —
    they start where the user turn's own render ends — so the inter-turn
    glue that :class:`AssistantMarkers`'s terminator-anchored ``header``
    absorbs (gemma-4's post-``<turn|>`` newline) sits inside ``head`` and
    drops out of the comparison.

    The rule, stated on TEXT so a token merge at the boundary cannot break
    it:

        the appended header's text must be a PREFIX of the text a
        COMPLETED assistant turn renders in the same place.

    Equality is the ordinary case (gemma-4 H=3, Llama-3.1 H=4, Qwen3.5
    pinned thinking-OFF H=7). Anchoring both spans at the end of ``head``
    and comparing TEXT is what keeps a token merge from deciding: the old
    rule compared ``pre`` to ``full_a`` as TOKEN sequences across the
    whole header, so Qwen3.5's ``\n`` merging with the header's ``\n``
    into ``\n\n`` refused the template outright and reported the merge
    as the fault. (Equal text with a different split cannot actually
    occur at this seam — both spans start at the same character offset
    and end on a token edge, so equal text forces equal ids. The merge is
    always a symptom of a genuine length difference, which is what the
    prefix relation below then classifies.)

    A STRICT prefix means the generation prompt stops INSIDE the header
    and the model is expected to emit the rest — Qwen3.5 thinking-ON
    leaves ``<think>\n`` open (H=5) where a completed turn carries the
    closed empty block (H=7). Which width a caller wants is then a
    property of the RENDER REGIME, not of the template, so that shape
    derives only when the regime is pinned; unpinned, BOTH widths are
    derivable and neither is the answer, so it refuses and names the pin.
    That refusal is the one this module already raised for the whole
    thinking-ON family (lens_worker's install preflight, job 255399
    2026-08-30; the tinycap served driver, 2026-09-06) — but it said
    "boundary retokenization", which is the SYMPTOM (Qwen3.5's ``\n``
    merging with the header's ``\n`` into ``\n\n``) and sent both
    diagnoses to the wrong layer.

    Refuses (:class:`ChatMarkerError`) on every other shape: a
    generation prompt that overshoots the content start (seeded
    content), one whose text diverges from the turn scaffold, and a
    template that ignores the flag WITHOUT emitting the header
    unconditionally."""
    if full_a[:len(head)] != list(head):
        d = divergence(head, full_a)
        raise ChatMarkerError(
            "probe: the add_generation_prompt=False render of the user "
            "turn is not a token prefix of the probe render — boundary "
            f"retokenization at token {d} "
            f"({head[d:d + 3]!r} vs {full_a[d:d + 3]!r}); refusing to "
            "guess assistant markers")
    if pre[:len(head)] != list(head):
        d = divergence(head, pre)
        raise ChatMarkerError(
            "the add_generation_prompt=False render is not a token prefix "
            f"of the True render — boundary retokenization at token {d} "
            f"({head[d:d + 3]!r} vs {pre[d:d + 3]!r}); the appended "
            "header cannot be read off the renders")
    appended = tuple(pre[len(head):])
    scaffold = tuple(full_a[len(head):s])
    if not appended:
        if scaffold:
            raise ChatMarkerError(
                "probe: the add_generation_prompt render ends at token "
                f"{len(pre)} but assistant content begins at token {s} — "
                "the generation-prompt boundary does not sit at the "
                "content start (a template that ignores the flag without "
                "appending the header unconditionally, or boundary "
                "retokenization merging across it); refusing to guess "
                "assistant markers")
        return appended  # flag-ignoring: the header is unconditional
    app_txt = tok.decode(list(appended))
    scaf_txt = tok.decode(list(scaffold))
    if not scaf_txt.startswith(app_txt):
        d = divergence(pre, full_a)
        raise ChatMarkerError(
            "probe: the add_generation_prompt render is not a prefix of "
            f"the probe render — boundary retokenization at token {d} "
            f"({pre[d:d + 3]!r} vs {full_a[d:d + 3]!r}): the generation "
            f"prompt appends {app_txt!r} where a completed assistant turn "
            f"renders {scaf_txt!r}; refusing to guess assistant markers")
    if app_txt != scaf_txt and not _render_regime_declared(tok):
        if _reasoning_template(tok):
            raise ChatMarkerError(
                "probe: the generation prompt stops INSIDE the assistant "
                f"header — it appends {app_txt!r} ({len(appended)} tokens) "
                "where a completed assistant turn renders "
                f"{scaf_txt!r} ({len(scaffold)} tokens) — and this "
                "template branches on a thinking switch whose render "
                "regime is NOT pinned, so both widths are derivable and "
                "neither is the answer. Pin the regime first: "
                "models.pin_reasoning_template_off(tok) for the fit/serve "
                "thinking-OFF render, or set tok._sisteer_template_kwargs "
                "explicitly; refusing to guess assistant markers")
        raise ChatMarkerError(
            "probe: the add_generation_prompt render ends at token "
            f"{len(pre)} but assistant content begins at token {s} — the "
            "generation-prompt boundary does not sit at the content start: "
            f"it appends {app_txt!r} where a completed assistant turn "
            f"renders {scaf_txt!r}; refusing to guess assistant markers")
    return appended


def derive_assistant_markers(tok) -> AssistantMarkers:
    """Derive :class:`AssistantMarkers` from a content-substitution probe.

    Renders ``[user: Hi, assistant: <sentinel>]`` twice with two sentinels
    sharing no words; the tokens that CHANGE between the two renders are the
    assistant content, everything constant around them is scaffold. This is
    the one sanctioned way to compare two renders: the probes differ only in
    CONTENT (never in a flag), the difference is asserted non-empty, and the
    extracted span must decode back to the sentinel exactly — so a
    boundary-retokenization merge or an off-by-scaffold boundary fails
    loudly here instead of leaking into a mask. (The FORBIDDEN comparison —
    diffing ``add_generation_prompt`` True vs False and reading the header
    off the flag's effect — measures 0 on templates that append the header
    unconditionally, exactly where the header is always present.)

    The ``add_generation_prompt=True`` render of the user turn must land
    INSIDE the assistant header the probe located and go no further —
    checked on TEXT, so a boundary merge does not by itself refuse
    (:func:`_appended_generation_header`, which also records the appended
    header on the returned markers). A template that ignores the flag
    *without* appending the header unconditionally, one whose generation
    prompt overshoots the content start, and one whose generation prompt
    stops short on an UNPINNED reasoning template all fail HERE, loudly,
    instead of producing a scaffold-contaminated span or a guessed width.
    Raises :class:`ChatMarkerError` on any structural surprise."""

    def probe(sentinel):
        return [{"role": "user", "content": _PROBE_USER},
                {"role": "assistant", "content": sentinel}]

    full_a = render_ids(tok, probe(_SENTINEL_A))
    full_b = render_ids(tok, probe(_SENTINEL_B))
    if full_a == full_b:
        raise ChatMarkerError(
            "probe: substituting the assistant content changed nothing in "
            "the render — the difference must be non-zero; cannot derive "
            "assistant markers")
    s = divergence(full_a, full_b)  # common-prefix length = content start
    q = 0                           # common-suffix length = closing scaffold
    while (q < min(len(full_a), len(full_b)) - s
           and full_a[len(full_a) - 1 - q] == full_b[len(full_b) - 1 - q]):
        q += 1
    e = len(full_a) - q
    content = full_a[s:e]
    dec = tok.decode(content).strip()
    if dec != _SENTINEL_A:
        raise ChatMarkerError(
            f"probe: the substituted content span decodes to {dec!r}, not "
            f"the sentinel {_SENTINEL_A!r} — boundary retokenization merged "
            "content with scaffold; refusing to guess assistant markers")
    ctrl = control_token_ids(tok)
    if any(t in ctrl for t in content):
        raise ChatMarkerError(
            "probe: control tokens inside the substituted content span "
            f"({content!r}); cannot derive assistant markers")

    head = render_ids(tok, probe(_SENTINEL_A)[:1])
    pre = render_ids(tok, probe(_SENTINEL_A)[:1], add_generation_prompt=True)
    generation_header = _appended_generation_header(tok, head, pre, full_a, s)

    closing = full_a[e:]
    term_at = [i for i, t in enumerate(closing) if t in ctrl]
    if not closing:
        # Nothing at all follows the assistant content — a candidate
        # SUCCESSOR-BOUNDED template (GLM-4.x: turns end bare; the next
        # turn's role marker doubles as an EOS id and, mid-conversation,
        # as the only bound on the content). Accept only what a second
        # probe can prove: appending a follow-up user turn must leave
        # every token through the content end untouched AND put a
        # control token immediately at the content end.
        ext = render_ids(tok, probe(_SENTINEL_A)
                         + [{"role": "user", "content": _PROBE_USER}])
        if ext[:e] != full_a[:e]:
            d = divergence(ext, full_a)
            raise ChatMarkerError(
                "probe: appending a successor turn re-tokenized the "
                f"render at token {d} ({ext[d:d + 3]!r} vs "
                f"{full_a[d:d + 3]!r}); refusing to guess where "
                "assistant turns end")
        if len(ext) <= e or ext[e] not in ctrl:
            raise ChatMarkerError(
                "probe: no control token closes the assistant turn and "
                "the successor turn does not open with one at the "
                f"content end (next tokens {ext[e:e + 3]!r}); refusing "
                "to guess where assistant turns end")
        # The header is anchored INCLUSIVELY at the last control token
        # before the content — there is no previous-turn terminator to
        # anchor after. GLM-4.6V under the thinking-off pin:
        # [<|assistant|>, \n, <think>, </think>, \n]. The
        # generation_header_ids tail cross-check still guards a bad
        # anchor, and assistant_content_spans re-verifies the header
        # sits immediately before every span.
        anchors = [i for i in range(s) if full_a[i] in ctrl]
        header = full_a[anchors[-1]:s] if anchors else []
        return AssistantMarkers(header=tuple(header), terminator=None,
                                closing=(),
                                generation_header=generation_header)
    if not term_at:
        raise ChatMarkerError(
            "probe: no control token closes the assistant turn (closing "
            f"scaffold {closing!r}); refusing to guess where assistant "
            "turns end")
    terminator = closing[term_at[0]]

    # The header marker is located, not required: templates whose user
    # turns are unterminated (Llama-2-style [INST]) have no terminator
    # occurrence before the content, so no header can be located — the
    # probe boundary check above still guards the content start.
    before = [i for i in range(s) if full_a[i] == terminator]
    header = full_a[before[-1] + 1:s] if before else []
    return AssistantMarkers(header=tuple(header), terminator=terminator,
                            closing=tuple(closing),
                            generation_header=generation_header)


def generation_header_ids(tok, markers: Optional[AssistantMarkers] = None,
                           ) -> Tuple[int, ...]:
    """The GENERATION-PROMPT assistant header: the token ids that
    ``add_generation_prompt`` appends to a render — the id-form of
    ``sisteer.hooks.response_header_len``'s width, and the ONE id
    vocabulary for the serve-time response boundary.

    This is deliberately NOT ``derive_assistant_markers(tok).header``.
    The located header is anchored at the PREVIOUS turn's terminator and
    absorbs whatever scaffold the template puts after it: on gemma-4 the
    user turn closes with ``<turn|>\n``, so the located header carries a
    leading ``\n`` — a token that is PROMPT content, present in both
    ``add_generation_prompt`` renders and inside the fit-time mask's
    EXCLUDED region. The fit-time mask (``hooks.response_header_len``,
    the single width home) counts only the appended tokens, so a serve
    boundary built from the located header would steer one prompt-side
    token OUTSIDE the fit support — a train/serve support mismatch, not
    a wider margin (found on the scale-12 gemma wave, 2026-08-28: fits
    recorded 3, the located header measured 4, and the old guard refused
    the agreement of two different definitions). Llama-3.1 hides the
    difference — its user turn ends with the special ``<|eot_id|>`` and
    no trailing scaffold, so located == appended (H=4) — which is why
    the disagreement first surfaced on gemma.

    Derivation mirrors ``response_header_len``'s two paths at the id
    level, and every path is cross-checked against LOCATED structure so
    this never degrades into bare width arithmetic. Both paths are
    derived ONCE, by the probe in :func:`derive_assistant_markers`
    (:func:`_appended_generation_header`), and read off
    ``AssistantMarkers.generation_header`` here — this function is the
    named entry point, not a second implementation:

    * flag-honoring templates: the ``add_generation_prompt=True`` render
      minus the ``False`` render of a probe user turn — verified to be a
      non-empty TEXT PREFIX of the scaffold a completed assistant turn
      renders in the same place (seeded content and a divergent
      generation prompt refuse; a generation prompt that stops short
      inside the header, as Qwen3.5 thinking-ON does, derives only when
      the render regime is pinned);
    * flag-ignoring templates (the NousResearch mirror): the diff is
      empty exactly where the header is appended unconditionally — there
      the ``False`` render already ends with the full header, and the
      located header IS the appended header, so it serves (matching
      ``response_header_len``'s marker-fallback count of 4 on that
      mirror).

    Raises :class:`ChatMarkerError` when no header can be derived —
    templates with no locatable assistant header (Llama-2-style
    ``[INST]``) cannot serve an id-located response boundary."""
    if markers is None:
        markers = derive_assistant_markers(tok)
    if not markers.header:
        raise ChatMarkerError(
            "the template exposes no locatable assistant header — the "
            "generation-prompt header ids cannot be derived, so an "
            "id-located response boundary cannot be served")
    appended = markers.generation_header
    if appended is None:  # markers built by hand, never probe-derived
        appended = derive_assistant_markers(tok).generation_header
    if not appended:
        # flag-ignoring template: the header is appended unconditionally
        # (the probe in derive_assistant_markers proved the render ends
        # at content start), so the located header IS the appended one.
        return tuple(markers.header)
    return tuple(appended)


def _successor_turn_probe(tok, markers: AssistantMarkers
                          ) -> Tuple[Tuple[int, ...], int, Tuple[int, ...]]:
    """(inter-turn scaffold, the successor turn's opening control token,
    the successor USER turn's complete role header), all LOCATED in a
    multi-turn probe render — the content-substitution probe with a
    successor user turn appended, rendered twice with two sentinels so the
    user header's end is where the content begins — and cross-checked
    against the closing scaffold. Shared by :func:`assistant_turn_end_ids`,
    :func:`successor_role_header_ids` and :func:`turn_opener_id`; requires
    a terminator (the caller handles the successor-bounded case). Raises
    :class:`ChatMarkerError` on any structural surprise, never a guess."""
    probe = [{"role": "user", "content": _PROBE_USER},
             {"role": "assistant", "content": _SENTINEL_A}]
    full_a = render_ids(tok, probe)
    ext = render_ids(tok, probe + [{"role": "user", "content": _SENTINEL_A}])
    ext_b = render_ids(tok, probe + [{"role": "user", "content": _SENTINEL_B}])
    closing = list(markers.closing)
    e = len(full_a) - len(closing)          # content end (the probe's rule)
    if e < 0 or full_a[e:] != closing:
        raise ChatMarkerError(
            "turn end: the probe render does not end with the template's "
            f"closing scaffold {closing!r}; refusing to locate the turn end")
    if ext[:e] != full_a[:e]:
        d = divergence(ext, full_a)
        raise ChatMarkerError(
            "turn end: appending a successor turn re-tokenized the render "
            f"at token {d} ({ext[d:d + 3]!r} vs {full_a[d:d + 3]!r}); "
            "refusing to locate the turn end")
    t = e + markers.glue                    # the terminator's index
    if t >= len(ext) or ext[t] != markers.terminator:
        raise ChatMarkerError(
            f"turn end: expected the terminator {markers.terminator} at "
            f"token {t} of the multi-turn render, found "
            f"{ext[t:t + 3]!r}; refusing to locate the turn end")
    ctrl = control_token_ids(tok)
    nxt = [i for i in range(t + 1, len(ext)) if ext[i] in ctrl]
    if not nxt:
        raise ChatMarkerError(
            "turn end: no control token opens the successor turn after the "
            f"terminator (tokens {ext[t + 1:t + 4]!r}); refusing to locate "
            "where the assistant turn's scaffold ends")
    scaffold = tuple(int(x) for x in ext[e:nxt[0]])
    if list(scaffold) != closing[:len(scaffold)]:
        raise ChatMarkerError(
            f"turn end: the inter-turn scaffold {list(scaffold)} is not the "
            f"head of the closing scaffold {closing!r} — the template ends "
            "a historical assistant turn differently from a final one; "
            "refusing to guess which the mask should follow")
    # the successor USER turn's role header: from its opener to where its
    # content begins (the two sentinel renders diverge exactly there)
    c = divergence(ext, ext_b)
    if c <= nxt[0] or ext_b[:nxt[0] + 1] != ext[:nxt[0] + 1]:
        raise ChatMarkerError(
            "turn end: substituting the successor turn's content changed the "
            f"render at token {c}, at or before that turn's opener at "
            f"{nxt[0]}; refusing to locate the successor role header")
    user_header = tuple(int(x) for x in ext[nxt[0]:c])
    ctrl_inside = [t for t in user_header[1:] if t in ctrl and t != ext[nxt[0]]]
    if not user_header or any(t in ctrl for t in ext[c:c + 1]):
        raise ChatMarkerError(
            "turn end: the successor turn's content does not begin right "
            f"after its role header ({user_header!r} then {ext[c:c + 3]!r}); "
            "refusing to locate the successor role header")
    del ctrl_inside
    return scaffold, int(ext[nxt[0]]), user_header


def assistant_turn_end_ids(tok, markers: Optional[AssistantMarkers] = None,
                           ) -> Optional[Tuple[int, ...]]:
    """The assistant turn's END SCAFFOLD as it appears BETWEEN turns: the
    terminator plus the non-control glue the template emits after it,
    up to (not including) the next turn's opening control token —
    gemma-4 ``[<turn|>, \\n]``, Llama-3.1 ``[<|eot_id|>]``. This is
    :attr:`AssistantMarkers.closing` restricted to what follows a
    HISTORICAL turn: equal to ``closing`` whenever the template appends
    no unconditional trailing header, and always ``closing``'s head
    (``closing[:len(result)]``) with the terminator at offset
    :attr:`AssistantMarkers.glue`. ``None`` for a SUCCESSOR-BOUNDED
    template (``closing == ()``, no terminator: GLM-4.x), whose turns have
    no locatable end — a structural fact of the template, not a lookup
    failure.

    This is the serve-time ALL-TURNS mask's second marker (the lens
    ``positions="response_all_turns"``): an earlier assistant turn's
    steered span runs from its header THROUGH this scaffold, so that a
    turn's mask does not change when it stops being the final one — a
    closed-tail render (gemma-4's Petri ``content_and_tool_response``
    shape) ends in exactly this scaffold and steers it as part of the
    generating turn's span, and a resumed prefill that shares those
    positions must see the same mask.

    Deliberately NOT ``derive_assistant_markers(tok).closing``: the
    closing scaffold is read off a render whose assistant turn is LAST,
    and on a flag-ignoring template (the NousResearch Llama-3.1 mirror) it
    carries the unconditional trailing assistant header, which follows
    only the last turn of a render and never a historical one. So the
    scaffold is LOCATED in a multi-turn probe render instead
    (:func:`_successor_turn_probe`) and cross-checked against the closing
    scaffold's head. Raises :class:`ChatMarkerError` on any structural
    surprise (never a guess)."""
    if markers is None:
        markers = derive_assistant_markers(tok)
    if markers.terminator is None:
        return None
    scaffold, _opener, _user_header = _successor_turn_probe(tok, markers)
    return scaffold


def successor_role_header_ids(tok, markers: Optional[AssistantMarkers] = None,
                              ) -> Optional[Tuple[int, ...]]:
    """The complete ROLE HEADER of the turn that follows an assistant turn
    — the user turn's header, gemma-4 ``[<|turn>, user, \\n]``, Llama-3.1
    ``[<|start_header_id|>, user, <|end_header_id|>, \\n\\n]`` — LOCATED
    in the multi-turn probe (:func:`_successor_turn_probe`) and
    cross-checked to have THE SAME SHAPE as the generation header
    (:func:`generation_header_ids`): the same opener, exactly ONE role
    token in place of the assistant's, and the same tail after it.
    ``None`` for a successor-bounded template.

    This is what the ALL-TURNS mask's historical rule relies on. The op
    carries only the assistant header, so it validates the header that
    follows a turn-end scaffold BY SHAPE — opener, one role token, the
    assistant header's tail — and this function certifies, once per
    template, that the user header actually has that shape, so a shape
    match is a genuine successor header for every role the rule admits
    (and the assistant itself for consecutive model turns). An
    opener token alone is NOT a genuine next turn: prose such as "the
    tokens are <turn|>\\n<|turn> and then text" produced an opener after a
    quoted scaffold and would have passed an opener-only rule (the
    2026-09-07 fourth review: 108 rows unsteered, no request failure).

    Raises :class:`ChatMarkerError` when the template's role headers do
    not share one shape — a multi-token role (Llama-3.1's ``ipython`` is
    two tokens; the tiny test tokenizer splits ``assistant``), an
    assistant header with a suffix the user header lacks (Nemotron's
    thinking-off ``<think></think>``) — because the rule would then refuse
    genuine turns or admit forged ones; the all-turns mode refuses such a
    template rather than guess."""
    if markers is None:
        markers = derive_assistant_markers(tok)
    if markers.terminator is None:
        return None
    _scaffold, opener, user_header = _successor_turn_probe(tok, markers)
    header = tuple(int(t) for t in generation_header_ids(tok, markers))
    if not header or int(header[0]) != opener:
        raise ChatMarkerError(
            f"turn opener: the successor turn opens with control token "
            f"{opener} but the generation header starts with "
            f"{list(header)[:1]} — this template opens user and model turns "
            "differently, so a genuine turn end cannot be told from a quoted "
            "one by the token that follows it; the all-turns mask's "
            "historical rule has no ground here")
    if len(user_header) != len(header) or user_header[2:] != header[2:]:
        raise ChatMarkerError(
            f"role header shape: the successor user header {list(user_header)} "
            f"and the generation header {list(header)} do not share one shape "
            "(same opener, exactly one role token, the same tail) — the "
            "all-turns mask validates a successor header by that shape, so on "
            "this template it would refuse genuine turns or admit forged "
            "ones; the mode cannot be served here")
    return user_header


def turn_opener_id(tok, markers: Optional[AssistantMarkers] = None,
                   ) -> Optional[int]:
    """The control token that OPENS a turn on this template — gemma-4
    ``<|turn>``, Llama-3.1 ``<|start_header_id|>``, ChatML ``<|im_start|>``
    — the first id of :func:`successor_role_header_ids`, which performs
    the full shape cross-check (same opener as the generation header, one
    role token, the same tail) and raises on a template whose role
    headers do not share one shape. ``None`` for a successor-bounded
    template. Kept as the verifiers' entry point; the all-turns rule
    itself validates the COMPLETE successor header, never the opener
    alone."""
    user_header = successor_role_header_ids(tok, markers)
    return None if user_header is None else int(user_header[0])


# --------------------------------------------------------------------------
# content spans
# --------------------------------------------------------------------------


def assistant_content_spans(
    tok, messages: Sequence[Mapping[str, str]],
    markers: Optional[AssistantMarkers] = None,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """(full-conversation token ids, one ``(lo, hi)`` span per assistant
    message — ``hi`` exclusive, empty content ⇒ ``lo == hi``).

    Invariants, each verified per assistant message and raising
    :class:`ChatMarkerError` when violated (fail loudly, never guess):

    * the ``add_generation_prompt=True`` render of the preceding messages
      is a token prefix of the incremental render (boundary retokenization
      check) — which is what pins the boundary at this message's content
      start: the probe proved only that the generation prompt lands
      inside the assistant header, so a template whose generation prompt
      stops short of recorded content (Qwen3.5 thinking-ON) refuses HERE
      rather than at derivation;
    * when a ``header`` marker was locatable, it sits immediately before
      the span — the span is introduced by a LOCATED assistant header;
    * the span ends exactly where the probe-derived ``closing`` scaffold
      begins (``hi = len(render) - len(closing)``, verified token-for-token
      against ``closing``), so the terminal EOT, any glue, and any
      unconditional trailing header can never leak into the span — this is
      the suffix-end fix for the NousResearch corruption;
    * the full-conversation render agrees with the incremental render
      through the turn terminator, so spans index the same tokens in
      ``full``;
    * no control token id (:func:`control_token_ids`) survives inside the
      span — which also means a span can never cross a turn boundary.
    """
    full = render_ids(tok, messages)
    ctrl = control_token_ids(tok)
    spans: List[Tuple[int, int]] = []
    for k, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        if markers is None:
            markers = derive_assistant_markers(tok)
        head, closing = list(markers.header), list(markers.closing)
        pre = render_ids(tok, messages[:k], add_generation_prompt=True)
        post = render_ids(tok, messages[:k + 1])
        if post[:len(pre)] != pre:
            d = divergence(pre, post)
            raise ChatMarkerError(
                f"assistant message {k}: the add_generation_prompt render "
                "is not a token prefix of the message render — boundary "
                f"retokenization at token {d} "
                f"({pre[d:d + 3]!r} vs {post[d:d + 3]!r}); refusing to "
                "guess an assistant span")
        lo = len(pre)
        if head and (len(pre) < len(head) or pre[-len(head):] != head):
            raise ChatMarkerError(
                f"assistant message {k}: the tokens before the content span "
                f"are {pre[-len(head):]!r}, not the assistant header marker "
                f"{head!r}; refusing an unmarked assistant boundary")
        hi = len(post) - len(closing)
        if hi < lo:
            raise ChatMarkerError(
                f"assistant message {k}: the closing scaffold "
                f"({len(closing)} tokens) overlaps the content start at "
                f"token {lo}; refusing to guess an assistant span")
        if post[hi:] != closing:
            d = divergence(closing, post[hi:])
            raise ChatMarkerError(
                f"assistant message {k}: the render does not end with the "
                f"template's closing scaffold — divergence at token "
                f"{hi + d} ({post[hi + d:hi + d + 3]!r} vs "
                f"{closing[d:d + 3]!r}) — assistant content may have merged "
                "with scaffold, or the template's closing structure "
                "changed; refusing to guess")
        # content (+ glue + terminator, when the template has a closing
        # scaffold) must agree with the full render
        n = hi if markers.terminator is None else hi + markers.glue + 1
        if full[:n] != post[:n]:
            d = divergence(post, full)
            raise ChatMarkerError(
                f"assistant message {k}: the incremental render is not a "
                "token prefix of the full-conversation render over the "
                f"content span (divergence at token {d}); refusing to "
                "guess an assistant span")
        if markers.terminator is None and len(full) > hi \
                and full[hi] not in ctrl:
            raise ChatMarkerError(
                f"assistant message {k}: successor-bounded template, but "
                "the token after the content span in the full render is "
                f"not a control token ({full[hi:hi + 3]!r}) — the span "
                "cannot be trusted to end at the turn boundary; refusing "
                "to guess an assistant span")
        bad = [(p, post[p]) for p in range(lo, hi) if post[p] in ctrl]
        if bad:
            raise ChatMarkerError(
                f"assistant message {k}: control tokens survived inside "
                f"the content span: {bad!r}; refusing a "
                "scaffold-contaminated span")
        spans.append((lo, hi))
    return full, spans


# --------------------------------------------------------------------------
# the render REGIME's header: thinking-ON vs thinking-OFF
# --------------------------------------------------------------------------

_TEMPLATE_KWARGS_ATTR = "_sisteer_template_kwargs"
_UNSET = object()


def generation_header_ids_under(tok, *, enable_thinking: bool) -> Tuple[int, ...]:
    """:func:`generation_header_ids` derived with the render regime PINNED
    to ``enable_thinking`` for the length of the derivation — the same seam
    ``models.pin_reasoning_template_off`` writes (``tok._sisteer_template_
    kwargs``) and every render site reads — and the tokenizer's own pin
    restored afterwards, whatever it was (absent included).

    Qwen3.5's assistant header is a property of the RENDER REGIME, not of
    the template (:func:`_appended_generation_header`): pinned OFF the
    generation prompt is the 7-token header ending in a closed empty think
    block; pinned ON it is the 5-token header ending in an OPEN ``<think>``.
    A serving stack that declares a thinking-ON render for a response-fit
    object needs the ON header beside the OFF one, and both must come from
    THIS one derivation so neither can drift from the other.

    ``enable_thinking=True`` on a template with no thinking switch REFUSES
    (:class:`ChatMarkerError`): there is no ON render to derive, and a
    declaration of one would be inert (the proxy's ``thinking_kwargs_inert``
    refusal is the request-time twin). ``enable_thinking=False`` on such a
    template derives the plain header with the pin cleared — the switch is
    not passed to a template that cannot read it."""
    if not isinstance(enable_thinking, bool):
        raise ChatMarkerError(
            f"generation_header_ids_under: enable_thinking must be a bool, "
            f"got {enable_thinking!r} — the render regime is a declared fact, "
            "never a truthy coercion")
    has_switch = _reasoning_template(tok)
    if enable_thinking and not has_switch:
        raise ChatMarkerError(
            "generation_header_ids_under(enable_thinking=True): this chat "
            "template has no thinking switch (no 'enable_thinking' in the "
            "template), so there is no thinking-ON render whose header could "
            "be derived — a thinking-ON declaration on this template would be "
            "inert; refusing")
    saved = getattr(tok, _TEMPLATE_KWARGS_ATTR, _UNSET)
    try:
        if has_switch:
            setattr(tok, _TEMPLATE_KWARGS_ATTR, {"enable_thinking": enable_thinking})
        elif saved is not _UNSET:
            delattr(tok, _TEMPLATE_KWARGS_ATTR)
        return tuple(int(t) for t in generation_header_ids(tok))
    finally:
        if saved is _UNSET:
            if hasattr(tok, _TEMPLATE_KWARGS_ATTR):
                delattr(tok, _TEMPLATE_KWARGS_ATTR)
        else:
            setattr(tok, _TEMPLATE_KWARGS_ATTR, saved)


def thinking_header_facts(tok) -> Dict[str, Any]:
    """The per-template THINKING-HEADER FACT the proxy validates requests
    against and the harness cross-checks its own header measurement with:
    a JSON-able dict

    ``{"switch": bool, "moves": bool | None, "off_ids": [...], "on_ids": [...] | None}``

    - ``switch`` — whether the template branches on ``enable_thinking``
      (:func:`_reasoning_template`; Llama-3.1 False, gemma-4 and Qwen3.5
      True);
    - ``off_ids`` — the thinking-OFF generation header
      (:func:`generation_header_ids_under` at ``enable_thinking=False``);
    - ``on_ids`` — the thinking-ON generation header, ``None`` when there is
      no switch (no ON render exists);
    - ``moves`` — ``on_ids != off_ids``: whether the thinking switch MOVES
      the assistant header (Qwen3.5: 7 -> 5 tokens, True; gemma-4: the ON
      render adds a leading system block and leaves the header at 3, False);
      ``None`` when there is no switch.

    Derived from the served tokenizer exactly as the registry's headers
    are, so the three surfaces (registry index, proxy fact, harness
    cross-check) share one derivation. The tokenizer's own render pin is
    left exactly as it was."""
    switch = bool(_reasoning_template(tok))
    off_ids = list(generation_header_ids_under(tok, enable_thinking=False))
    if not switch:
        return {"switch": False, "moves": None, "off_ids": off_ids,
                "on_ids": None}
    on_ids = list(generation_header_ids_under(tok, enable_thinking=True))
    return {"switch": True, "moves": on_ids != off_ids, "off_ids": off_ids,
            "on_ids": on_ids}


def thinking_header_facts_or_error(tok) -> Tuple[Dict[str, Any], Optional[str]]:
    """:func:`thinking_header_facts` with the thinking-ON derivation made
    NON-FATAL — ``(facts, None)`` when every field derives, else
    ``(partial, reason)``.

    A template can carry a thinking switch whose ON generation header this
    module cannot certify: Nemotron-3's ON generation prompt ends in an OPEN
    ``<think>\n`` where its completed assistant turn carries the closed
    ``<think></think>``, so :func:`_appended_generation_header`'s prefix rule
    refuses (:class:`ChatMarkerError`). A standup that serves thinking-OFF
    only has no use for that header, and must not die deriving it
    (fit-support-transfer code review X1) — so here the ON certification is
    recorded as UNAVAILABLE rather than raised: ``partial`` keeps ``switch``
    (True) and ``off_ids`` exactly as :func:`thinking_header_facts` would
    and sets ``on_ids: None`` and ``moves: None`` (unknown, not False —
    nothing here says the header stays put), while ``reason`` is
    ``f"{type(e).__name__}: {e}"`` for the recorder to carry beside the fact.
    The proxy then refuses every declared thinking-ON request on a lens route
    ``thinking_on_unattested`` (fail closed) and serves OFF as before.

    Only :class:`ChatMarkerError` is caught, and only around the ON
    derivation: the switch detection and the OFF header still raise (a
    template with no OFF header serves nothing), and any other exception is
    a defect, not an unavailable certification. A caller that NEEDS the ON
    header (a cell declaring the thinking render) keeps calling
    :func:`thinking_header_facts`, whose behaviour is unchanged."""
    switch = bool(_reasoning_template(tok))
    off_ids = list(generation_header_ids_under(tok, enable_thinking=False))
    if not switch:
        return ({"switch": False, "moves": None, "off_ids": off_ids,
                 "on_ids": None}, None)
    try:
        on_ids = list(generation_header_ids_under(tok, enable_thinking=True))
    except ChatMarkerError as e:
        return ({"switch": True, "moves": None, "off_ids": off_ids,
                 "on_ids": None}, f"{type(e).__name__}: {e}")
    return ({"switch": True, "moves": on_ids != off_ids, "off_ids": off_ids,
             "on_ids": on_ids}, None)


# --- the thinking-ON CERTIFICATION (the sampled-rerun standup's `--thinking-on`) ----------
# Ported from the sampled-rerun lineage: `mt_serving_up --thinking-on` certifies the served
# template's ON header per template BEFORE any GPU minute (`certify_thinking`), while the
# per-cell `@thinking` declaration path above keeps its own derivation
# (`generation_header_ids_under` / `thinking_header_facts`). Additive: nothing above changes.

_REASON_A = "Copper heron lattice"
_REASON_B = "Velvet condor prism"


class _pinned_regime:
    """Pin ``tok``'s render regime (``_sisteer_template_kwargs``) for the
    length of a ``with`` block and restore it EXACTLY afterwards — absent
    included — so a derivation under one regime leaves the tokenizer
    rendering the other, as every existing render site expects."""

    def __init__(self, tok, kwargs: Optional[Mapping[str, object]]):
        self._tok = tok
        self._kwargs = kwargs
        self._saved = _UNSET

    def __enter__(self):
        self._saved = getattr(self._tok, _TEMPLATE_KWARGS_ATTR, _UNSET)
        if self._kwargs is None:
            if hasattr(self._tok, _TEMPLATE_KWARGS_ATTR):
                delattr(self._tok, _TEMPLATE_KWARGS_ATTR)
        else:
            setattr(self._tok, _TEMPLATE_KWARGS_ATTR, dict(self._kwargs))
        return self._tok

    def __exit__(self, *exc):
        if self._saved is _UNSET:
            if hasattr(self._tok, _TEMPLATE_KWARGS_ATTR):
                delattr(self._tok, _TEMPLATE_KWARGS_ATTR)
        else:
            setattr(self._tok, _TEMPLATE_KWARGS_ATTR, self._saved)
        return False


def _find_all(ids: Sequence[int], needle: Sequence[int]) -> Tuple[int, ...]:
    """Start index of every NON-OVERLAPPING occurrence of ``needle`` in
    ``ids`` (the lens adapters' own scan, kept local: this module imports
    nothing of the serving package)."""
    want = tuple(int(t) for t in needle)
    n, m = len(ids), len(want)
    out: List[int] = []
    i = 0
    while m and i <= n - m:
        if tuple(int(t) for t in ids[i:i + m]) == want:
            out.append(i)
            i += m
        else:
            i += 1
    return tuple(out)


def _thinking_turn_scaffold(tok, head: Sequence[int]
                            ) -> Tuple[Optional[Tuple[int, ...]], Optional[str]]:
    """The THINKING-turn probe: the scaffold a COMPLETED THINKING turn —
    one carrying ``reasoning_content`` — renders between the end of the
    user turn and the start of its reasoning, located by substituting two
    reasoning sentinels with the content held fixed. ``(scaffold_ids,
    None)`` or ``(None, reason)`` when the template renders no reasoning
    (both probes identical) or re-tokenizes the head.

    This is the thinking-ON regime's CANONICAL completed turn (the model
    thinks, then answers), against which the ON generation prompt is
    certified; the plain completed turn (empty reasoning) is the regime's
    DEGENERATE case and renders differently per template — Qwen3.5 keeps
    an empty block ``<think>\\n\\n</think>\\n\\n``, Nemotron-3 collapses it to
    ``<think></think>`` (where the ON prompt opens ``<think>\\n``), GLM-4.6V
    to ``<think></think>\\n``."""
    def probe(reason):
        return [{"role": "user", "content": _PROBE_USER},
                {"role": "assistant", "content": _SENTINEL_A,
                 "reasoning_content": reason}]
    try:
        full_ra = render_ids(tok, probe(_REASON_A))
        full_rb = render_ids(tok, probe(_REASON_B))
    except Exception as e:  # noqa: BLE001 — a template that cannot render the key
        return None, (f"the template cannot render a reasoning_content-bearing "
                      f"assistant turn: {type(e).__name__}: {e}")
    if full_ra == full_rb:
        return None, ("the template renders no reasoning_content (substituting "
                      "the reasoning changed nothing), so no completed THINKING "
                      "turn can be located")
    if full_ra[:len(head)] != list(head):
        d = divergence(head, full_ra)
        return None, ("the reasoning-bearing probe render is not a token "
                      f"continuation of the user turn (divergence at token {d})")
    r = divergence(full_ra, full_rb)  # common prefix = reasoning start
    return tuple(full_ra[len(head):r]), None


#: Whether the thinking-ON certification ALSO requires the plain-turn rule
#: (:func:`_appended_generation_header` against the EMPTY-reasoning completed
#: turn — the fit-support-transfer landing's certification, which refuses
#: Nemotron-3: its ON prompt opens ``<think>\n`` where its empty-reasoning
#: turn collapses to ``<think></think>``). ``False``: the THINKING-turn rule
#: (:func:`_thinking_turn_scaffold`) is the required certification and the
#: plain-turn verdict is RECORDED beside it (``plain_turn_prefix`` in the
#: certification). Flip to ``True`` to reproduce the landing's verdicts
#: (Qwen3.5 / GLM-4.6V certified, Nemotron-3 refused).
THINKING_ON_REQUIRES_PLAIN_TURN_PREFIX = False


def thinking_on_certification(tok) -> Dict[str, Any]:
    """CERTIFY the served template's THINKING-ON generation header, or say
    exactly why it cannot be — the one derivation behind
    :func:`generation_header_ids_under`, :func:`thinking_header_facts` and
    the standup's ``thinking_certification`` record. Returns a JSON-able
    dict:

    ``switch``
        whether the template branches on ``enable_thinking``
        (:func:`_reasoning_template`). ``False`` -> nothing else is derived:
        there is no ON render to certify (``verdict: "no_switch"``).
    ``off_ids``
        the thinking-OFF generation header — ``generation_header_ids`` under
        the OFF pin, the served path's own derivation, untouched.
    ``on_ids``
        the thinking-ON generation header: the ids the ON render's
        ``add_generation_prompt`` appends after the user turn's own render
        (a clean token suffix, non-empty), CERTIFIED as header structure by
        the text-prefix rule against a completed turn — ``None`` when the
        certification fails.
    ``moves``
        ``on_ids != off_ids``: whether the thinking switch MOVES the
        assistant header (Qwen3.5 7 -> 5, GLM-4.6V 5 -> 2, Nemotron-3
        6 -> 6 with a different last token: True; gemma-4, whose ON render
        adds a leading system block and leaves the header at 3: False).
        ``None`` when ``on_ids`` is None.
    ``checks``
        the individual certification facts: ``clean_suffix`` (the ON
        generation prompt continues the user turn's render token for token
        and appends something), ``plain_turn_prefix`` (the landing's rule:
        the appended text is a prefix of the EMPTY-reasoning completed
        turn's scaffold), ``thinking_turn_prefix`` (the appended text is a
        prefix of the REASONING-bearing completed turn's scaffold — the ON
        regime's canonical completed turn), ``off_header_absent_in_on_render``
        (the OFF header occurs nowhere in the ON generation render — so the
        two regimes can never be confused at the tail; ``None`` where the
        header does not move, the OFF header then IS the ON header),
        ``tail`` (the ON render ends with ``on_ids`` — the response
        locator's tail rule holds under the declared header).
    ``verdict``
        ``"certified"`` (``on_ids`` derived and every required check
        holds, header moves), ``"trivially_in_support"`` (``on_ids`` derived
        and equal to ``off_ids`` — the mask needs no second header; gemma-4),
        ``"no_switch"``, or ``"refused"`` with ``reason`` naming the failed
        check.

    The tokenizer's own render pin is left EXACTLY as it was. Nothing here
    changes the OFF derivation: ``off_ids`` is ``generation_header_ids``'s
    own answer under the OFF pin."""
    out: Dict[str, Any] = {"switch": bool(_reasoning_template(tok)),
                           "off_ids": None, "on_ids": None, "moves": None,
                           "checks": {}, "verdict": None, "reason": None,
                           "requires_plain_turn_prefix":
                               bool(THINKING_ON_REQUIRES_PLAIN_TURN_PREFIX)}
    out["off_ids"] = list(generation_header_ids_under(tok, enable_thinking=False))
    if not out["switch"]:
        out["verdict"] = "no_switch"
        out["reason"] = ("the chat template has no thinking switch (no "
                         "'enable_thinking' in the template), so there is no "
                         "thinking-ON render whose header could be derived")
        return out
    checks = out["checks"]
    with _pinned_regime(tok, {"enable_thinking": True}):
        probe_user = [{"role": "user", "content": _PROBE_USER}]
        head = render_ids(tok, probe_user, add_generation_prompt=False)
        pre = render_ids(tok, probe_user, add_generation_prompt=True)
        clean = pre[:len(head)] == head and len(pre) > len(head)
        checks["clean_suffix"] = bool(clean)
        if not clean:
            out["verdict"] = "refused"
            out["reason"] = (
                "the thinking-ON generation prompt is not a clean non-empty "
                "token suffix of the user turn's own render (divergence at "
                f"token {divergence(head, pre)}; {len(pre)} vs {len(head)} "
                "tokens) — the ON header cannot be read off the renders")
            return out
        appended = tuple(int(t) for t in pre[len(head):])
        # the PLAIN-turn probe (the landing's rule), recorded either way
        full_a = render_ids(tok, probe_user + [{"role": "assistant",
                                                "content": _SENTINEL_A}])
        full_b = render_ids(tok, probe_user + [{"role": "assistant",
                                                "content": _SENTINEL_B}])
        plain_reason: Optional[str] = None
        if full_a == full_b:
            plain_ok = False
            plain_reason = ("substituting the assistant content changed nothing "
                            "in the render")
        else:
            s = divergence(full_a, full_b)
            try:
                got = _appended_generation_header(tok, head, pre, full_a, s)
                plain_ok = tuple(got) == appended
                if not plain_ok:
                    plain_reason = (f"the plain-turn rule derived {list(got)}, "
                                    f"not the appended {list(appended)}")
            except ChatMarkerError as e:
                plain_ok, plain_reason = False, str(e)
        checks["plain_turn_prefix"] = bool(plain_ok)
        checks["plain_turn_reason"] = plain_reason
        # the THINKING-turn probe: the ON regime's canonical completed turn
        scaffold_r, think_reason = _thinking_turn_scaffold(tok, head)
        if scaffold_r is None:
            think_ok = False
        else:
            app_txt = tok.decode(list(appended))
            scaf_txt = tok.decode(list(scaffold_r))
            think_ok = scaf_txt.startswith(app_txt)
            if not think_ok:
                think_reason = (f"the ON generation prompt appends {app_txt!r} "
                                f"where a completed THINKING turn renders "
                                f"{scaf_txt!r} before its reasoning — not a "
                                "prefix, so the appended tokens are not the "
                                "thinking turn's opening scaffold")
        checks["thinking_turn_prefix"] = bool(think_ok)
        checks["thinking_turn_reason"] = think_reason
        checks["tail"] = tuple(pre[-len(appended):]) == appended
    off = tuple(out["off_ids"])
    moves = appended != off
    checks["off_header_absent_in_on_render"] = (
        (len(_find_all(pre, off)) == 0) if moves else None)
    required = [("thinking_turn_prefix", think_ok), ("tail", checks["tail"])]
    if THINKING_ON_REQUIRES_PLAIN_TURN_PREFIX:
        required.append(("plain_turn_prefix", plain_ok))
    if moves:
        required.append(("off_header_absent_in_on_render",
                         checks["off_header_absent_in_on_render"]))
    failed = [name for name, ok in required if not ok]
    if failed:
        out["verdict"] = "refused"
        reasons = {"thinking_turn_prefix": think_reason,
                   "plain_turn_prefix": plain_reason,
                   "tail": "the ON render does not end with its own appended header",
                   "off_header_absent_in_on_render":
                       "the thinking-OFF header occurs inside the thinking-ON "
                       "render, so the two regimes' tails cannot be told apart"}
        out["reason"] = "; ".join(f"{name}: {reasons.get(name)}" for name in failed)
        return out
    out["on_ids"] = list(appended)
    out["moves"] = bool(moves)
    out["verdict"] = "certified" if moves else "trivially_in_support"
    return out
