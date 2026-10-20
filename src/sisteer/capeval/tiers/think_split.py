"""The think-block split a scorer applies before extracting an answer, under an
explicit MODE (``contracts.think_split_mode``) — never inferred from content.

WHY a mode: Qwen3.5's thinking-ON template opens ``<think>`` INSIDE THE PROMPT
(the header ends ``<think>\\n`` — ``contracts.QWEN35_HEADER_IDS_ON``), so a draw
that ran out of cap inside its thoughts is tag-free; ``sampling.engine.strip_think``
would return it unchanged and a numeric matcher would score an intermediate
reasoning number as the answer. No content rule can tell it from a genuine
tag-free answer.

* ``record_only`` — the completion IS the answer; tags are recorded, nothing is
  stripped (thinking-OFF; a parser-separated ON arm).
* ``implicit_open`` — the block is open at index 0 by construction of the prompt:
  text after the FIRST ``</think>`` is the answer; none means UNTERMINATED.
* ``explicit_tags`` — complete blocks are removed, an unclosed ``<think>`` is
  UNTERMINATED, tag-free text is unchanged (equal to ``strip_think`` on tagged
  content; its ``</think>``-only safety net is deliberately not reproduced).
* ``none`` — NO reasoning channel exists (a no-thinking-channel family served
  under ``contracts.THINKING_NOT_AVAILABLE``; llama-paper-rows P8 / P19): the
  content is the answer BYTE FOR BYTE, nothing is stripped, nothing is ever
  unterminated, and any think-tag literal is recorded as a STRAY
  (``stray_open_tag`` / ``stray_close_tag``) — an anomaly the reader fails a
  primary leg closed on, never a block to fold or discard.

:data:`UNTERMINATED_POLICY` is the ONE constant selecting how an unterminated
draw scores: ``unscored`` (primary) or ``incorrect`` (sensitivity).
:func:`split_think` is pure Python; :func:`think_aware_scorer` imports
inspect_ai lazily.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from sisteer.capeval.tinycap import contracts as C

__all__ = [
    "CENSORED_REASONING_ONLY_FIELD", "LIFTED_CLOSE_FIELD", "NO_THINK_BLOCK_FIELD", "SCORER_WRAPPER_FIELD", "SPLIT_METADATA_FIELDS", "SPLIT_METADATA_FIELDS_NONE", "THINK_CLOSE",
    "THINK_OPEN", "UNTERMINATED_POLICY", "WRAPPER_NAME", "ThinkSplit", "inner_scorer_name",
    "message_channels", "output_channels", "split_metadata", "split_think", "think_aware_scorer",
]

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
#: The policy in force for an UNTERMINATED think block: the spec's rule.
UNTERMINATED_POLICY = C.UNTERMINATED_UNSCORED
#: The per-row facts the scorer writes into ``Score.metadata`` — read back by
#: ``inspect_served.results_from_log`` onto the row and the outcome record.
SPLIT_METADATA_FIELDS = ("think_block_present", "think_block_unterminated",
                         "think_split_mode", "stray_open_tag")
#: GSM8K-SCORER-fix: under ``implicit_open`` a completion that STOPPED (not at the cap) without ever emitting ``</think>`` is a
#: response with NO think block — the model reasoned inline and ended its turn — scored by the plain final-answer parse, never filed
#: unscored as an "unterminated block" (50 of 200 on the LeCun constitution ReLU cell read NaN with a parseable `ANSWER: <n>` line).
#: Recorded per row in ``Score.metadata`` under this key (True on exactly those draws; absent elsewhere, so every other row's metadata
#: is byte-identical); the summary counts them as ``n_no_think_block``.
#: G8KSCORERFIX2: the rule is an OPT-IN of the wrap (``think_aware_scorer(no_think_block_when_stopped=True)`` — the GSM8K
#: wrap alone; MATH-500's wrap keeps filing such draws unterminated -> unscored, its paper reader classifies them from the raw body
#: as ``eos_in_thoughts`` / ``lifted_close`` and requires every implicit_open draw present), it admits a NATURAL stop only
#: (``stop_reason == "stop"``; ``model_length`` — a context overflow — and ``unknown`` — no finish reason — stay unterminated), and
#: it reads the output channels BEFORE claiming no block: see :data:`LIFTED_CLOSE_FIELD`.
#: G8KSCORERFIX3: "no think block" means no ``<think>`` ANYWHERE in the text — a stopped draw whose text carries an unclosed
#: ``<think>`` (``stray_open_tag``) with no lift is a block the model opened and never closed: unterminated, never this key.
NO_THINK_BLOCK_FIELD = "no_think_block"
#: inspect_ai's client-side lift (``parse_content_with_reasoning``, grammar ``<think([^>]*)>…</think>``)
#: moves a COMPLETE think block out of the assistant text into a ``ContentReasoning`` part, so a draw whose model closed its block
#: and answered reaches the scorer with NO ``</think>`` in its text and a NON-EMPTY reasoning channel. Under the opt-in such a draw
#: is a PRESENT, closed block (``think_block_present`` True, unterminated False) scored on its text — recorded under this key (True
#: on exactly those draws; absent elsewhere), NEVER as ``no_think_block``, so ``n_no_think_block`` counts inline answers only and
#: ``n_think_block_present`` counts the lifted close as the block it is. Row metadata only: no summary count (the GSM8K paper claim
#: is the no-think count; a lifted close is an ordinary present block to it).
#: The reasoning channel proves the lift of A complete pair, NOT that the remaining text holds no open block —
#: a stray ``<think>`` the split still sees in the text (``stray_open_tag``) after the lift is a block the model opened again and
#: never closed, so the lifted-close reading applies ONLY when ``stray_open_tag`` is False; otherwise the draw is unterminated
#: (unscored / INCORRECT by policy, neither this key nor ``no_think_block``, counted by ``n_unterminated_not_at_cap``).
LIFTED_CLOSE_FIELD = "lifted_close"
#: the ``none`` split's ADDITIONAL fact (llama-paper-rows P19 / P21): a literal
#: ``</think>`` in a completion of a family with NO reasoning channel — recorded
#: under ``none`` ONLY, so every other mode's metadata is byte-identical
SPLIT_METADATA_FIELDS_NONE = (*SPLIT_METADATA_FIELDS, "stray_close_tag")
_COMPLETE_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

# --- the MATH answer's DELIMITERS (math500-thinking-on-full22) ------------------------------------------------
#: GLM-4.5V / GLM-4.6V wrap the committed answer in an answer-boxing pair. It is output FORMAT, like a code fence: a
#: scorer must see the boxed text, not the box — the same edit as ``strip_box_markers`` on the grading path
#: (66cf6472), marker by marker so an opener whose closer fell past the cap leaves a truncated answer, never an
#: empty one. Contrast ``<think>``: reasoning that is REMOVED, whose unterminated form means "no answer".
BOX_MARKERS = ("<|begin_of_box|>", "<|end_of_box|>")
#: ``\( x \)`` / ``\[ x \]`` are the LaTeX presentation delimiters inspect_evals' ``normalize_final_answer`` does
#: not read: it extracts ``$…$`` only (Lewkowycz et al. 2022 appendix D, copied there verbatim), so ``ANSWER: \( 5 \)``
#: scored INCORRECT against target 5 through the real scorer while ``ANSWER: $5$`` and ``ANSWER: 5`` scored CORRECT.
#: The delimiter TOKENS are REMOVED from the answer line's payload — never rewritten to ``$…$``: the extractor keeps
#: only the FIRST ``$…$`` span of a line, so a span rewrite would credit ``ANSWER: \(2\) + \(3\)`` as ``2`` (the
#: reviewer's case); with the tokens gone the whole expression ``2 + 3`` reaches sympy. A ``\text{ANSWER: }`` label
#: inside display math becomes the plain label the extractor reads. Measured on the GLM diagnostic smoke: 4 of 11
#: finished thinking-ON answers used ``\( \)``, 2 the box — every one INCORRECT against its correct target before.
_ANSWER_LINE = re.compile(r"(?im)^(.*?ANSWER\s*:\s*)(.*)$")      # the extractor's line grammar: the payload after the label
_PRESENTATION_DELIMS = re.compile(r"\\[()\[\]]")                 # the four tokens \( \) \[ \]
_TEXT_ANSWER_LABEL = re.compile(r"\\text(?:bf)?\{\s*ANSWER\s*:\s*\}", re.IGNORECASE)
#: a display block CLOSED on its own line after the answer (``\\[\n\\text{ANSWER: } x\n\\]``): the extractor anchors the
#: ANSWER line to the END of the completion (``[^\n]+\s*\Z``), so a trailing line holding nothing but the closing
#: delimiter hides the answer — 1 of 18 well-formed Nemotron smoke answers (G-PARSER); the lone token is dropped
_TRAILING_DELIM_LINE = re.compile(r"\n[ \t]*\\[\])][ \t]*(?=\s*\Z)")
#: the answer normalizations a wrap may apply to the text the INNER scorer reads (``Score.metadata[
#: ANSWER_NORMALIZATION_FIELD] = {"rule": <name>, "changed": <bool>}`` records which, and whether it changed the text)
ANSWER_NORMALIZATION_NONE = "none"                 # the GSM8K wrap: the inner scorer reads the text byte-identically
ANSWER_NORMALIZATION_MATH = "math_delimiters"      # the MATH-500 wrap (inspect_served._math_scorer)
ANSWER_NORMALIZATION_FIELD = "answer_normalization"


def normalize_math_answer(text: str) -> str:
    """The MATH answer text with GLM's box markers dropped (content kept), a ``\\text{ANSWER: }`` label made plain,
    and the presentation delimiters ``\\( \\) \\[ \\]`` REMOVED from every ``ANSWER:`` line's payload — the
    mathematics is untouched and complete: ``ANSWER: \\(2\\) + \\(3\\)`` becomes ``ANSWER: 2 + 3``, never ``$2$ + $3$``
    (a span the extractor would truncate to its first term). Lines without the label are left alone, and so is an
    answer line that already carries a literal ``$`` — there the upstream first-span selection is in force and removing
    delimiters beside it could manufacture a parseable fragment; such a line scores exactly as before. A trailing
    line holding only a closing delimiter (a display block closed under the answer) is dropped, since the extractor reads
    the ANSWER line only at the end of the text."""
    for m in BOX_MARKERS:
        text = text.replace(m, "")
    text = _TEXT_ANSWER_LABEL.sub("ANSWER: ", text)
    text = _TRAILING_DELIM_LINE.sub("", text)

    def strip_line(m):
        # an answer line that already carries a literal `$` is left EXACTLY as written (the reviewer's second case):
        # inspect_evals' normalizer selects the first `$…$` span of the line, so removing delimiters next to literal
        # dollar signs could turn the fragment it selects into a parseable partial expression (`$\\(5\\) + $\\(3\\)` ->
        # `$5 + $3` -> `5 +` -> 5); untouched, such a line scores exactly as it did before this normalization
        if "$" in m.group(2):
            return m.group(0)
        return m.group(1) + _PRESENTATION_DELIMS.sub("", m.group(2))

    return _ANSWER_LINE.sub(strip_line, text)


ANSWER_NORMALIZATIONS: dict[str, Callable[[str], str]] = {ANSWER_NORMALIZATION_NONE: lambda t: t,
                                                          ANSWER_NORMALIZATION_MATH: normalize_math_answer}


@dataclass(frozen=True)
class ThinkSplit:
    """One completion's split: the text the scorer should read, and the two
    facts the row records. ``close_tag_index`` is the index of the FIRST
    ``</think>`` in the raw content (``None`` when there is none);
    ``stray_open_tag`` is set under ``implicit_open`` when a literal
    ``<think>`` appears inside a completion whose block the PROMPT already
    opened (an anomaly count; the rule is unchanged by it)."""

    mode: str
    answer_text: str
    present: bool
    unterminated: bool
    close_tag_index: Optional[int]
    stray_open_tag: bool = False
    #: ``none`` only (llama-paper-rows): a literal ``</think>`` in a completion
    #: of a family with NO reasoning channel — a stray, recorded, never acted on
    stray_close_tag: bool = False


def _check_mode(mode: str) -> str:
    if mode not in C.THINK_SPLIT_MODES:
        raise ValueError(f"think split mode must be one of {C.THINK_SPLIT_MODES}, got {mode!r}")
    return mode


def split_think(content: str, *, mode: str) -> ThinkSplit:
    """Split ``content`` under ``mode`` (one of ``contracts.THINK_SPLIT_MODES``).

    The mode is a RENDER fact handed in by the plan, never inferred here — see
    the module docstring for the three rules. ``record_only`` returns the
    content byte-identical; the two splitting modes return the answer text
    stripped of surrounding whitespace (as ``strip_think`` does), and ``""``
    for an unterminated block."""
    _check_mode(mode)
    text = content if isinstance(content, str) else ""
    first_close = text.find(THINK_CLOSE)
    close_index = first_close if first_close >= 0 else None
    if mode == C.THINK_SPLIT_NONE:
        # llama-paper-rows (P19): NO reasoning channel — the content IS the answer,
        # byte for byte; a tag literal is a STRAY (both flags), never a block: present
        # False, unterminated False, nothing stripped
        return ThinkSplit(mode=mode, answer_text=text, present=False, unterminated=False,
                          close_tag_index=close_index, stray_open_tag=THINK_OPEN in text,
                          stray_close_tag=close_index is not None)
    if mode == C.THINK_SPLIT_RECORD_ONLY:
        last_open = text.rfind(THINK_OPEN)
        present = last_open >= 0 or close_index is not None
        unterminated = last_open >= 0 and text.find(THINK_CLOSE, last_open) < 0
        return ThinkSplit(mode=mode, answer_text=text, present=present,
                          unterminated=unterminated, close_tag_index=close_index)
    if mode == C.THINK_SPLIT_IMPLICIT_OPEN:
        # the block is open at index 0 by construction of the prompt: the
        # completion either closes it (answer follows) or never does
        stray = THINK_OPEN in text
        if close_index is None:
            return ThinkSplit(mode=mode, answer_text="", present=True, unterminated=True,
                              close_tag_index=None, stray_open_tag=stray)
        answer = text[close_index + len(THINK_CLOSE):].strip()
        return ThinkSplit(mode=mode, answer_text=answer, present=True, unterminated=False,
                          close_tag_index=close_index, stray_open_tag=stray)
    # explicit_tags: the strip_think rule for content that carries <think>,
    # minus its </think>-only safety net (that case is implicit_open's)
    if THINK_OPEN not in text:
        return ThinkSplit(mode=mode, answer_text=text, present=False, unterminated=False,
                          close_tag_index=close_index)
    out = _COMPLETE_BLOCK.sub("", text)
    if THINK_OPEN in out:
        return ThinkSplit(mode=mode, answer_text="", present=True, unterminated=True,
                          close_tag_index=close_index)
    return ThinkSplit(mode=mode, answer_text=out.strip(), present=True, unterminated=False,
                      close_tag_index=close_index)


def split_metadata(split: ThinkSplit) -> dict[str, Any]:
    """The ``Score.metadata`` entries for one split (:data:`SPLIT_METADATA_FIELDS`;
    under ``none`` :data:`SPLIT_METADATA_FIELDS_NONE` — the close-tag stray is
    recorded there ONLY, so every other mode's metadata is byte-identical)."""
    out = {"think_block_present": bool(split.present),
           "think_block_unterminated": bool(split.unterminated),
           "think_split_mode": split.mode,
           "stray_open_tag": bool(split.stray_open_tag)}
    if split.mode == C.THINK_SPLIT_NONE:
        out["stray_close_tag"] = bool(split.stray_close_tag)
    return out


#: The wrapper's LEGACY registered name (the GSM8K wrap; pinned by test).
WRAPPER_NAME = "think_aware_scorer"
#: ``Score.metadata["scorer_wrapper"]`` — the wrapper's own identity, recorded
#: SEPARATELY from the score key: a MATH wrap is registered under the INNER
#: scorer's name so the log's score key stays ``PRIMARY_SCORER["math"]``, and
#: this field says a think-aware wrapper scored it.
SCORER_WRAPPER_FIELD = "scorer_wrapper"
#: the censored-representation metadata: a parser-separated
#: (``record_only``) cap-bound output whose reasoning channel is non-empty and
#: whose content is empty is an UNTERMINATED think block, never an empty answer
CENSORED_REASONING_ONLY_FIELD = "censored_reasoning_only"
#: inspect_ai's normalised at-cap ``stop_reason`` (``inspect_served.STOP_REASON_AT_CAP``)
_AT_CAP = "max_tokens"
#: inspect_ai's normalised NATURAL stop (``as_stop_reason``: the provider's ``stop`` / ``eos``) — the ONE stop reason the
#: no-think-block opt-in admits (G8KSCORERFIX2): ``model_length`` is a context overflow and ``unknown`` a missing finish
#: reason, neither a model that ended its turn, and ``inspect_served`` already tells the three apart on the row
_STOPPED = "stop"

_FACTORIES: dict[str, Callable[..., Any]] = {}


def _factory(name: str = WRAPPER_NAME) -> Callable[..., Any]:
    """The registered Inspect scorer factory under ``name``, built on first use
    so this module imports without inspect_ai — ONE per registered name (the
    legacy ``think_aware_scorer`` for GSM8K; the inner scorer's own name for
    MATH)."""
    cached = _FACTORIES.get(name)
    if cached is not None:
        return cached
    from inspect_ai.scorer import INCORRECT, Score, Target, accuracy, scorer, stderr
    from inspect_ai.solver import TaskState

    # metrics: accuracy + stderr — the two `match` declares; the wrapper wraps
    # `match` on the GSM8K tasks only in this tree (the plan's S2.4), and the
    # metrics decide Inspect's headline, never a row: rows come off the score
    @scorer(metrics=[accuracy(), stderr()], name=name)
    def think_aware(inner: Any, *, mode: str, unterminated_policy: str,
                    answer_normalization: str = ANSWER_NORMALIZATION_NONE,
                    no_think_block_when_stopped: bool = False) -> Any:
        normalize = ANSWER_NORMALIZATIONS[answer_normalization]

        async def score(state: TaskState, target: Target) -> Score:
            raw = state.output.completion
            split = split_think(raw, mode=mode)
            meta = split_metadata(split)
            meta[SCORER_WRAPPER_FIELD] = f"think_aware/{mode}"
            meta[ANSWER_NORMALIZATION_FIELD] = {"rule": answer_normalization, "changed": False}
            if mode == C.THINK_SPLIT_NONE:
                # llama-paper-rows (P19 / P21): no reasoning channel — the inner scorer reads the
                # UNTOUCHED state (the text the model returned, byte for byte; the provider's
                # no-switch policy restored it before any lift); the strays ride the metadata
                # and the reader refuses a primary leg on them. No normalization applies
                # (GSM8K's `none` rule; a MATH wrap under this mode would be a new decision).
                return _with_metadata(await inner(state, target), meta)
            if mode == C.THINK_SPLIT_RECORD_ONLY:
                # channel-aware censoring for a PARSER-SEPARATED output.
                # A cap-bound draw whose reasoning channel is non-empty and whose
                # content is EMPTY ran out of cap inside its thoughts: an explicit
                # censored representation (present, unterminated, no answer) ->
                # unscored under the primary policy, INCORRECT under the
                # sensitivity branch — never an empty answer scored as wrong in
                # both. Reasoning + non-empty content at the cap is case (c):
                # scored as-is, at-cap.
                content, reasoning = output_channels(state.output)
                if reasoning and not content and _stop_reason(state.output) == _AT_CAP:
                    meta.update({"think_block_present": True, "think_block_unterminated": True,
                                 CENSORED_REASONING_ONLY_FIELD: True})
                    if unterminated_policy == C.UNTERMINATED_UNSCORED:
                        return Score.unscored(answer=None, explanation=raw, metadata=meta)
                    return Score(value=INCORRECT, answer="", explanation=raw, metadata=meta)
                if reasoning:
                    meta["think_block_present"] = True
                meta[CENSORED_REASONING_ONLY_FIELD] = False
                # byte-identical scoring to today under the `none` normalization: the
                # inner scorer reads the UNTOUCHED state; only the metadata is added.
                # A normalization that changes the text hands the inner scorer a
                # shallow copy — the raw output and the .eval log keep the full text
                normed = normalize(raw)
                if normed == raw:
                    inner_score = await inner(state, target)
                else:
                    meta[ANSWER_NORMALIZATION_FIELD]["changed"] = True
                    shadow = copy.copy(state)
                    shadow.output = state.output.model_copy(update={"completion": normed})
                    inner_score = await inner(shadow, target)
                return _with_metadata(inner_score, meta)
            if (split.unterminated and mode == C.THINK_SPLIT_IMPLICIT_OPEN and no_think_block_when_stopped
                    and _stop_reason(state.output) == _STOPPED):
                # The GSM8K scorer fix, which the wrap OPTED INTO (the GSM8K wrap alone): the prompt opened
                # the block, the model never closed it in its TEXT and ended its turn NATURALLY (`stop`; an at-cap draw is the policy case
                # below, and `model_length` / `unknown` stay unterminated too). The channels decide what that is: a NON-EMPTY reasoning
                # channel means inspect_ai's client-side lift moved a complete `<think>…</think>` out of the text (the lifted_close
                # class) — the block IS present and closed and the text is the answer; an EMPTY one means a response with NO think
                # block — the completion IS the answer. Either way the inner scorer reads the text whole (its final `ANSWER: <n>` line
                # parses; an unparseable one is INCORRECT, never NaN), and only the no-block draw is counted as `n_no_think_block`.
                _content, reasoning = output_channels(state.output)
                if split.stray_open_tag:
                    # G8KSCORERFIX3: a stray `<think>` still in the TEXT is a block the model opened and never closed,
                    # whatever the reasoning channel says — beside a lift (`<think>example</think>\n<think>still reasoning\nANSWER: 43`:
                    # inspect_ai lifts `example` and leaves the unclosed `<think>`; the channel proves the lift of A complete pair, not a
                    # closed text) or with no lift at all (`<think>still reasoning\nANSWER: 43`, reasoning None). Unterminated —
                    # the same policy as the unterminated branch below, applied here; neither lifted_close nor no_think_block;
                    # n_unterminated_not_at_cap counts it. A text with NO `<think>` anywhere is the no-block case below, as before.
                    if unterminated_policy == C.UNTERMINATED_UNSCORED:
                        return Score.unscored(answer=None, explanation=raw, metadata=meta)
                    return Score(value=INCORRECT, answer="", explanation=raw, metadata=meta)
                if reasoning:
                    meta.update({"think_block_present": True, "think_block_unterminated": False, LIFTED_CLOSE_FIELD: True})
                else:
                    meta.update({"think_block_present": False, "think_block_unterminated": False, NO_THINK_BLOCK_FIELD: True})
                answer = normalize(raw.strip())
                meta[ANSWER_NORMALIZATION_FIELD]["changed"] = answer != raw.strip()
                shadow = copy.copy(state)
                shadow.output = state.output.model_copy(update={"completion": answer})
                return _with_metadata(await inner(shadow, target), meta)
            if split.unterminated:
                if unterminated_policy == C.UNTERMINATED_UNSCORED:
                    return Score.unscored(answer=None, explanation=raw, metadata=meta)
                return Score(value=INCORRECT, answer="", explanation=raw, metadata=meta)
            # a SHALLOW copy of the state whose output text is the answer: the
            # raw output is untouched, so the row's `content` and the .eval log
            # keep the full text
            answer = normalize(split.answer_text)
            meta[ANSWER_NORMALIZATION_FIELD]["changed"] = answer != split.answer_text
            shadow = copy.copy(state)
            shadow.output = state.output.model_copy(update={"completion": answer})
            inner_score = await inner(shadow, target)
            return _with_metadata(inner_score, meta)

        return score

    _FACTORIES[name] = think_aware
    return think_aware


def message_channels(message: Any) -> tuple[str, Optional[str]]:
    """``(content, reasoning)`` of one assistant message as inspect_ai
    0.3.260 represents a split response: ``content`` is the message's
    ``text`` (the TEXT parts joined — exactly what ``ModelOutput.completion``
    and every scorer read; ``""`` when the answer was empty or null),
    ``reasoning`` the ``ContentReasoning`` parts joined (None when there
    are none). Pure — the ONE reader of that shape (``inspect_served``
    re-exports it for its log adapter and provider)."""
    parts = [str(c.reasoning) for c in message.content_list
             if getattr(c, "type", None) == "reasoning" and getattr(c, "reasoning", None)]
    return str(message.text or ""), ("\n".join(parts) if parts else None)


def output_channels(output: Any) -> tuple[Optional[str], Optional[str]]:
    """``(content, reasoning)`` of a sample's ``output`` (``ModelOutput``);
    ``(None, None)`` when there is no output or it has no choices — an
    absent message, distinct from an empty one."""
    choices = list(getattr(output, "choices", None) or []) if output is not None else []
    if not choices:
        return None, None
    return message_channels(choices[0].message)


def _stop_reason(output: Any) -> Optional[str]:
    return getattr(output, "stop_reason", None) if output is not None else None


def _with_metadata(score: Any, meta: dict[str, Any]) -> Any:
    merged = {**(dict(score.metadata) if score.metadata else {}), **meta}
    return score.model_copy(update={"metadata": merged})


def think_aware_scorer(inner: Any, *, mode: str,
                       unterminated_policy: str = UNTERMINATED_POLICY,
                       name: Optional[str] = None,
                       answer_normalization: str = ANSWER_NORMALIZATION_NONE,
                       no_think_block_when_stopped: bool = False) -> Any:
    """Wrap an Inspect scorer to score the ANSWER of a completion under ``mode`` and
    record the split in ``Score.metadata`` (:data:`SPLIT_METADATA_FIELDS`).
    ``record_only`` delegates to ``inner`` on the untouched state; otherwise
    ``inner`` scores a shallow copy whose completion is the split's answer (the
    raw ``state.output`` and the ``.eval`` log keep the full text). An
    UNTERMINATED draw files ``Score.unscored()`` under ``unscored`` (answer
    ``None``, so the row's ``unparseable`` reads ``None``) or ``INCORRECT`` with an
    empty answer under ``incorrect``; the policy in force is
    :data:`UNTERMINATED_POLICY`.

    ``name`` is the REGISTERED name of the returned scorer — the log's score
    key. Default :data:`WRAPPER_NAME` (``think_aware_scorer``, the GSM8K wrap,
    unchanged); MATH passes the inner scorer's own unqualified name so
    ``PRIMARY_SCORER["math"]`` survives the wrap; the wrapper's identity rides
    in ``metadata["scorer_wrapper"]``.

    Channel-aware censoring: under ``record_only`` a cap-bound output with a
    non-empty reasoning channel and EMPTY content is filed as an unterminated
    block (unscored / incorrect by policy), never as an empty wrong answer.

    ``answer_normalization`` names the rewrite of the ANSWER text the inner
    scorer reads: ``none`` (default; GSM8K: byte-identical) or ``math_delimiters``
    (:func:`normalize_math_answer`, the MATH-500 wrap). Recorded per row in
    ``Score.metadata[ANSWER_NORMALIZATION_FIELD]`` with whether it changed the text.

    ``no_think_block_when_stopped`` (an OPT-IN; default False — the GSM8K wrap alone
    passes True): under ``implicit_open`` a draw whose text never closed the
    prompt-opened block and whose stop reason is the NATURAL ``stop`` is scored whole
    on its text — a PRESENT closed block when the reasoning channel is non-empty
    (inspect_ai lifted the complete block client-side; :data:`LIFTED_CLOSE_FIELD`), a
    response with NO think block otherwise (:data:`NO_THINK_BLOCK_FIELD`, counted as
    ``n_no_think_block``). ``max_tokens``, ``model_length`` and ``unknown`` stay
    unterminated under the policy. Off, such a draw is unterminated exactly as it was
    before this option existed (MATH-500's wrap: its paper reader classifies these
    from the raw body and requires every implicit_open draw present)."""
    _check_mode(mode)
    if unterminated_policy not in C.UNTERMINATED_POLICIES:
        raise ValueError(
            f"unterminated policy must be one of {C.UNTERMINATED_POLICIES}, "
            f"got {unterminated_policy!r}")
    if not callable(inner):
        raise TypeError("think_aware_scorer needs an inner Inspect scorer to delegate to")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError(f"think_aware_scorer: name must be a non-empty registered name, got {name!r}")
    if answer_normalization not in ANSWER_NORMALIZATIONS:
        raise ValueError(f"think_aware_scorer: answer_normalization must be one of {sorted(ANSWER_NORMALIZATIONS)}, "
                         f"got {answer_normalization!r}")
    if not isinstance(no_think_block_when_stopped, bool):
        raise ValueError(f"think_aware_scorer: no_think_block_when_stopped must be a bool, got {no_think_block_when_stopped!r}")
    return _factory(name or WRAPPER_NAME)(inner, mode=mode, unterminated_policy=unterminated_policy,
                                          answer_normalization=answer_normalization,
                                          no_think_block_when_stopped=no_think_block_when_stopped)


def inner_scorer_name(inner: Any) -> str:
    """The UNQUALIFIED registered name of an Inspect scorer object (the key
    the ``.eval`` log files its score under) — what a MATH wrap is registered
    as. Imports inspect_ai lazily."""
    from inspect_ai._util.registry import registry_unqualified_name

    return str(registry_unqualified_name(inner))
