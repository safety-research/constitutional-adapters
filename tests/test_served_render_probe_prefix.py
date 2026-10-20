"""The driver's predicate-(A) render probe carries the cell's SI text as the first message on a PROMPTED route
(math500-thinking-on-full22): the proxy verifies every request on a route with a registered SI prefix against
that prefix (H8) and refuses a bare user turn — the first production prompted cell died on exactly this probe. A bare or
object cell's probe is unchanged (no system turn: such routes refuse an unexpected prefix).

The ON arm's draws are CLASSIFIED against the family's render fact — a finished draw with no reasoning and no
think tag is the MODEL declining to think on that draw (gemma-4 answered the arithmetic probe "43" with no thought
block), inconclusive, and the probe re-asks with a problem that needs the working under a fresh seed, up to three
draws in all; a leaked tag or a draw carrying the shape is conclusive at once; the certification reads the last draw
and a refusal after declines names them."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import served_capability_run as srun  # noqa: E402
from sisteer.capeval.tinycap import contracts as tc  # noqa: E402


def _fake_chat(bodies: list):
    def chat(body):
        bodies.append(body)
        return {"choices": [{"message": {"content": "ok", "reasoning_content": None}, "finish_reason": "stop"}],
                tc.STAMP_BLOCK_KEY: {tc.STAMP_KEY_RENDER_REGIME: "thinking_off"}}
    return chat


def test_a_prompted_route_probe_leads_with_the_cells_system_text_on_both_arms():
    bodies: list = []
    out = srun._render_probe(_fake_chat(bodies), "petri__prompted__safety", lens=False, header_located=False,
                             system="SI PREFIX TEXT")
    assert [b["model"] for b in bodies] == ["petri__prompted__safety"] * 2
    for b in bodies:
        assert b["messages"][0] == {"role": "system", "content": "SI PREFIX TEXT"}     # the H8 prefix, first
        assert b["messages"][1] == {"role": "user", "content": srun.RENDER_PROBE_PROMPT} and len(b["messages"]) == 2
        assert b["max_tokens"] == srun.RENDER_PROBE_MAX_TOKENS == 512 and b["temperature"] == 0.0     # no row: greedy
    assert set(out["regimes"]) == {tc.THINKING_OFF, tc.THINKING_ON} and set(out["responses"]) == set(out["regimes"])


def test_an_on_reply_cut_at_the_cap_with_neither_channel_is_retried_once_at_the_larger_budget():
    """gemma's greedy 64-token probe came back with no content and no reasoning at the cap — inconclusive, not a
    shape; the probe retries the ON arm once at 2,048 and keeps the retry's reply; the OFF arm is never retried."""
    bodies: list = []
    seen_on = {"n": 0}

    def chat(body):
        bodies.append(body)
        on = body.get("chat_template_kwargs", {}).get("enable_thinking") is True or body.get("enable_thinking") is True
        if on:
            seen_on["n"] += 1
            if seen_on["n"] == 1:      # the first ON attempt: cut at the cap, channel-less
                return {"choices": [{"message": {"content": None, "reasoning_content": None}, "finish_reason": "length"}],
                        tc.STAMP_BLOCK_KEY: {tc.STAMP_KEY_RENDER_REGIME: "thinking_on"}}
            return {"choices": [{"message": {"content": "43", "reasoning_content": "17 + 26 = 43"}, "finish_reason": "stop"}],
                    tc.STAMP_BLOCK_KEY: {tc.STAMP_KEY_RENDER_REGIME: "thinking_on"}}
        return {"choices": [{"message": {"content": "43", "reasoning_content": None}, "finish_reason": "stop"}],
                tc.STAMP_BLOCK_KEY: {tc.STAMP_KEY_RENDER_REGIME: "thinking_off"}}

    out = srun._render_probe(chat, "petri__base", lens=False, header_located=False)
    on_bodies = [b for b in bodies if (b.get("chat_template_kwargs", {}).get("enable_thinking") is True or b.get("enable_thinking") is True)]
    assert [b["max_tokens"] for b in on_bodies] == [srun.RENDER_PROBE_MAX_TOKENS, srun.RENDER_PROBE_RETRY_MAX_TOKENS]
    assert len(bodies) == 3                                              # OFF once, ON twice
    assert out["responses"][tc.THINKING_ON]["reasoning"] == "17 + 26 = 43" and out["responses"][tc.THINKING_ON]["finish_reason"] == "stop"
    # a SECOND channel-less cap-cut stands (no third attempt) and is handed to the certification as it is
    bodies.clear(); seen_on["n"] = 0

    def chat2(body):
        bodies.append(body)
        on = body.get("chat_template_kwargs", {}).get("enable_thinking") is True or body.get("enable_thinking") is True
        return {"choices": [{"message": {"content": None, "reasoning_content": None}, "finish_reason": "length" if on else "stop"}],
                tc.STAMP_BLOCK_KEY: {tc.STAMP_KEY_RENDER_REGIME: "thinking_on" if on else "thinking_off"}}

    out2 = srun._render_probe(chat2, "petri__base", lens=False, header_located=False)
    assert len(bodies) == 3 and out2["responses"][tc.THINKING_ON]["finish_reason"] == "length"
    assert not out2["responses"][tc.THINKING_ON]["content"] and out2["responses"][tc.THINKING_ON]["reasoning"] is None


def test_a_bare_or_object_route_probe_sends_no_system_turn():
    for system in (None, ""):
        bodies: list = []
        srun._render_probe(_fake_chat(bodies), "petri__base", lens=False, header_located=False, system=system)
        assert all(b["messages"] == [{"role": "user", "content": srun.RENDER_PROBE_PROMPT}] for b in bodies)
    bodies = []
    srun._render_probe(_fake_chat(bodies), "petri__base", lens=False, header_located=False)      # the default
    assert all(len(b["messages"]) == 1 and b["messages"][0]["role"] == "user" for b in bodies)


def test_the_generate_call_site_hands_the_probe_the_cells_system_text():
    src = (REPO / "scripts/served_capability_run.py").read_text()
    i = src.index("probe = _render_probe(_probe_chat(tc.PROBE_KIND_RENDER)")
    assert "system=system" in src[i:i + 300]
    assert "system = ccr._cell_system(cell, cfg)" in src        # the same text the real requests carry


def test_the_probe_decodes_under_the_cells_sampling_row_when_given():
    """The probe decodes under the production row (temperature / top_p / top_k, optional penalties only when set),
    seeded — never argmax, which looped on the opening think tag on a steered route."""
    bodies: list = []
    row = tc.SamplingRow(temperature=0.6, top_p=0.95, top_k=20, source="test", presence_penalty=1.5)
    srun._render_probe(_fake_chat(bodies), "petri__si_x", lens=True, header_located=False, sampling_row=row)
    for b in bodies:
        assert (b["temperature"], b["top_p"], b["top_k"], b["presence_penalty"], b["seed"]) == (0.6, 0.95, 20, 1.5, 0)
        assert "min_p" not in b and "repetition_penalty" not in b                # None fields are not sent
    bodies.clear()
    srun._render_probe(_fake_chat(bodies), "petri__base", lens=False, header_located=False,
                       sampling_row=tc.SamplingRow(temperature=1.0, top_p=None, top_k=None, source="test"))
    assert all(b["temperature"] == 1.0 and "top_p" not in b and "top_k" not in b for b in bodies)


def test_the_generate_call_site_hands_the_probe_the_sampling_row():
    src = (REPO / "scripts/served_capability_run.py").read_text()
    i = src.index("probe = _render_probe(_probe_chat(tc.PROBE_KIND_RENDER)")
    assert "sampling_row=sampling.row" in src[i:i + 400]


def test_the_probe_reply_reader_accepts_the_reasoning_channel_under_either_key():
    """This vLLM returns the parser-separated channel as `reasoning`; older servers as `reasoning_content`."""
    old = {"choices": [{"message": {"content": "43", "reasoning_content": "17 + 26 = 43"}, "finish_reason": "stop"}]}
    new = {"choices": [{"message": {"content": "43", "reasoning": "17 + 26 = 43"}, "finish_reason": "stop"}]}
    none = {"choices": [{"message": {"content": "43"}, "finish_reason": "stop"}]}
    assert srun._probe_reply(old)["reasoning"] == "17 + 26 = 43"
    assert srun._probe_reply(new)["reasoning"] == "17 + 26 = 43"
    assert srun._probe_reply(none)["reasoning"] is None and srun._probe_reply(none)["content"] == "43"


# --------------------------------------------------------------------------- #
# the ON arm's classification and its retries                                  #
# --------------------------------------------------------------------------- #

_ON_STAMP = {tc.STAMP_BLOCK_KEY: {tc.STAMP_KEY_RENDER_REGIME: tc.RENDER_REGIME_THINKING_ON}}
_OFF_STAMP = {tc.STAMP_BLOCK_KEY: {tc.STAMP_KEY_RENDER_REGIME: tc.RENDER_REGIME_THINKING_OFF}}


def _is_on(body) -> bool:
    return body.get("chat_template_kwargs", {}).get("enable_thinking") is True or body.get("enable_thinking") is True


def _reply(content, reasoning=None, finish="stop", *, on: bool) -> dict:
    return {"choices": [{"message": {"content": content, "reasoning": reasoning}, "finish_reason": finish}],
            **(_ON_STAMP if on else _OFF_STAMP)}


def _scripted_chat(bodies: list, on_replies: list):
    """The OFF arm answers "43" once; the ON arm answers ``on_replies`` in order (a shortfall fails the test loudly)."""
    on_replies = list(on_replies)

    def chat(body):
        bodies.append(body)
        if not _is_on(body):
            return _reply("43", on=False)
        assert on_replies, "the probe drew more ON completions than the script provides"
        return on_replies.pop(0)
    return chat


def _on_bodies(bodies):
    return [b for b in bodies if _is_on(b)]


def _certify(out, family: str):
    """Predicate (A) THROUGH the driver's wrapper over ``contracts.render_certified`` with the family's own header ids
    measured live and the right regime stamps, so the shape (and the declined note) is the only thing that can refuse."""
    fact = tc.FAMILY_RENDER_FACTS[family]
    ids = {tc.THINKING_OFF: list(fact.off_ids), tc.THINKING_ON: list(fact.on_ids)}
    return srun._render_certification(out, family=family, header_ids_by_arm=ids, measured_live=True)


GEMMA_DECLINED = _reply("43", None, "stop", on=True)                       # the production reply, verbatim in shape
GEMMA_THOUGHT = _reply("36", "Monday 96 + 105 = 201; Tuesday 72 + 165 = 237; 237 - 201 = 36.", "stop", on=True)


def test_a_gemma_probe_whose_first_on_draw_declines_is_retried_with_the_reasoning_prompt_and_certified():
    """content-only / stop on the arithmetic prompt is the model declining, not the render — the second draw asks
    the reasoning prompt under seed 1, carries the parser-separated channel, and certifies; both draws are on record."""
    bodies: list = []
    out = srun._render_probe(_scripted_chat(bodies, [GEMMA_DECLINED, GEMMA_THOUGHT]), "petri__base", lens=False,
                             header_located=False, family="gemma4",
                             sampling_row=tc.SamplingRow(temperature=1.0, top_p=0.95, top_k=64, source="test"))
    on = _on_bodies(bodies)
    assert len(bodies) == 3 and len(on) == 2                                        # OFF once, ON twice
    assert [b["messages"][-1]["content"] for b in on] == [srun.RENDER_PROBE_PROMPT, srun.RENDER_PROBE_REASONING_PROMPT]
    assert [b["seed"] for b in on] == [0, 1] and [b["max_tokens"] for b in on] == [512, 512]
    assert all((b["temperature"], b["top_p"], b["top_k"]) == (1.0, 0.95, 64) for b in on)   # the cell's row, every draw
    attempts = out["on_attempts"]
    assert [a["class"] for a in attempts] == [srun.PROBE_CLASS_NO_THINKING_EMITTED, srun.PROBE_CLASS_THINK_SHAPE]
    assert [(a["attempt"], a["prompt_id"], a["seed"], a["finish_reason"]) for a in attempts] == [
        (1, srun.RENDER_PROBE_PROMPT_ARITHMETIC, 0, "stop"), (2, srun.RENDER_PROBE_PROMPT_REASONING, 1, "stop")]
    assert attempts[0]["reasoning_chars"] == 0 and attempts[1]["reasoning_chars"] > 0 and attempts[1]["content_chars"] == 2
    assert out["responses"][tc.THINKING_ON]["content"] == "36"                     # the CERTIFYING draw is the second
    record, error = _certify(out, "gemma4")
    assert error is None and record["think_shape_certified"] is True and record["render_certified"] is True


def test_three_declining_draws_are_refused_with_the_declined_message_naming_the_prompts():
    bodies: list = []
    out = srun._render_probe(_scripted_chat(bodies, [GEMMA_DECLINED] * 3), "petri__base", lens=False,
                             header_located=False, family="gemma4")
    on = _on_bodies(bodies)
    assert len(on) == 3 == srun.RENDER_PROBE_MAX_ATTEMPTS and len(bodies) == 4
    assert [b["seed"] for b in on] == [0, 1, 2]                                    # a fresh, deterministic seed per draw
    assert [b["messages"][-1]["content"] for b in on] == [
        srun.RENDER_PROBE_PROMPT, srun.RENDER_PROBE_REASONING_PROMPT, srun.RENDER_PROBE_REASONING_PROMPT]
    assert [a["class"] for a in out["on_attempts"]] == [srun.PROBE_CLASS_NO_THINKING_EMITTED] * 3
    record, error = _certify(out, "gemma4")
    assert record["render_certified"] is False and record["think_shape_certified"] is False
    assert error.startswith(tc.REC_THINKING_UNSERVABLE) and "does not carry the family's think shape" in error
    assert "the model declined to open a thought block on 3 draws" in error
    assert repr(srun.RENDER_PROBE_PROMPT) in error and repr(srun.RENDER_PROBE_REASONING_PROMPT) in error   # the prompts, named
    assert srun.RENDER_PROBE_PROMPT_ARITHMETIC in error and srun.RENDER_PROBE_PROMPT_REASONING in error


def test_a_leaked_tag_on_the_first_on_draw_is_refused_at_once_with_no_retry():
    """record_only (gemma): a raw channel marker in content means the server parser missed a block — conclusive, refused,
    never retried; implicit_open (qwen): a stray opening tag inside the prompt-opened block, the same."""
    gemma_leak = _reply("<|channel>thought\nlet me add<channel|>43", None, "stop", on=True)
    bodies: list = []
    out = srun._render_probe(_scripted_chat(bodies, [gemma_leak]), "petri__base", lens=False, header_located=False,
                             family="gemma4")
    assert len(_on_bodies(bodies)) == 1 and [a["class"] for a in out["on_attempts"]] == [srun.PROBE_CLASS_LEAKED_TAG]
    record, error = _certify(out, "gemma4")
    assert error is not None and "does not carry the family's think shape" in error and "declined" not in error
    assert record["think_shape_certified"] is False

    qwen_stray = _reply("<think>\nagain</think>\n43", None, "stop", on=True)
    bodies.clear()
    out = srun._render_probe(_scripted_chat(bodies, [qwen_stray]), "petri__base", lens=False, header_located=False,
                             family="qwen3_5")
    assert len(_on_bodies(bodies)) == 1 and [a["class"] for a in out["on_attempts"]] == [srun.PROBE_CLASS_LEAKED_TAG]
    error = _certify(out, "qwen3_5")[1]
    assert error is not None and "does not carry the family's think shape" in error and "declined" not in error


def test_the_cap_cut_retry_path_is_unchanged_under_a_family_fact_and_counts_toward_the_draws():
    """The cap-cut retry: a channel-less cap-cut re-sends the SAME prompt and seed at 2,048 (the same draw with room); a
    second channel-less cap-cut stands with no third draw. A cap-cut retry that then DECLINES still gets the reasoning
    prompt, within the three-draw total."""
    at_cap = _reply(None, None, "length", on=True)
    bodies: list = []
    out = srun._render_probe(_scripted_chat(bodies, [at_cap, GEMMA_THOUGHT]), "petri__base", lens=False,
                             header_located=False, family="gemma4")
    on = _on_bodies(bodies)
    assert [b["max_tokens"] for b in on] == [srun.RENDER_PROBE_MAX_TOKENS, srun.RENDER_PROBE_RETRY_MAX_TOKENS]
    assert [b["seed"] for b in on] == [0, 0] and all(b["messages"][-1]["content"] == srun.RENDER_PROBE_PROMPT for b in on)
    assert [a["class"] for a in out["on_attempts"]] == [srun.PROBE_CLASS_AT_CAP_CHANNEL_LESS, srun.PROBE_CLASS_THINK_SHAPE]
    assert _certify(out, "gemma4")[1] is None

    bodies.clear()
    out = srun._render_probe(_scripted_chat(bodies, [at_cap, at_cap]), "petri__base", lens=False, header_located=False,
                             family="gemma4")
    assert len(_on_bodies(bodies)) == 2 and out["responses"][tc.THINKING_ON]["finish_reason"] == "length"
    assert [a["class"] for a in out["on_attempts"]] == [srun.PROBE_CLASS_AT_CAP_CHANNEL_LESS] * 2
    error = _certify(out, "gemma4")[1]
    assert error is not None and "declined" not in error                            # a cap-cut is not a decline

    bodies.clear()
    out = srun._render_probe(_scripted_chat(bodies, [at_cap, GEMMA_DECLINED, GEMMA_THOUGHT]), "petri__base",
                             lens=False, header_located=False, family="gemma4")
    on = _on_bodies(bodies)
    assert [(b["max_tokens"], b["seed"]) for b in on] == [(512, 0), (2048, 0), (512, 2)]
    assert [b["messages"][-1]["content"] for b in on] == [srun.RENDER_PROBE_PROMPT] * 2 + [srun.RENDER_PROBE_REASONING_PROMPT]
    assert [a["class"] for a in out["on_attempts"]] == [srun.PROBE_CLASS_AT_CAP_CHANNEL_LESS,
                                                        srun.PROBE_CLASS_NO_THINKING_EMITTED, srun.PROBE_CLASS_THINK_SHAPE]
    assert _certify(out, "gemma4")[1] is None


def test_a_first_draw_carrying_the_shape_is_exactly_one_draw_for_every_family():
    """qwen / nemotron (implicit_open: the closing tag), glm (explicit_tags: the opening tag), gemma (record_only: the
    reasoning channel) — one ON draw on the arithmetic prompt at seed 0, as before."""
    shaped = {"qwen3_5": _reply("17 + 26 = 43</think>\n43", None, "stop", on=True),
              "nemotron_h": _reply("Adding.</think>\n\n43", None, "stop", on=True),
              "glm4v": _reply("<think>17 + 26 = 43</think>43", None, "stop", on=True),
              "gemma4": _reply("43", "17 + 26 = 43", "stop", on=True)}
    for family, reply in shaped.items():
        bodies: list = []
        out = srun._render_probe(_scripted_chat(bodies, [reply]), "petri__base", lens=False, header_located=False,
                                 family=family)
        on = _on_bodies(bodies)
        assert len(bodies) == 2 and len(on) == 1, family
        assert on[0]["seed"] == 0 and on[0]["max_tokens"] == 512 and on[0]["messages"][-1]["content"] == srun.RENDER_PROBE_PROMPT
        assert [a["class"] for a in out["on_attempts"]] == [srun.PROBE_CLASS_THINK_SHAPE], family
        assert _certify(out, family)[1] is None, family


def test_the_off_probe_is_one_arithmetic_draw_whatever_the_on_arm_does():
    bodies: list = []
    out = srun._render_probe(_scripted_chat(bodies, [GEMMA_DECLINED] * 3), "petri__prompted__safety", lens=False,
                             header_located=False, system="SI PREFIX TEXT", family="gemma4")
    off = [b for b in bodies if not _is_on(b)]
    assert len(off) == 1 and off[0]["seed"] == 0 and off[0]["max_tokens"] == 512
    assert off[0]["messages"] == [{"role": "system", "content": "SI PREFIX TEXT"},
                                  {"role": "user", "content": srun.RENDER_PROBE_PROMPT}]
    assert set(out["responses"][tc.THINKING_OFF]) == {"content", "reasoning", "finish_reason", "render_regime"}
    assert out["responses"][tc.THINKING_OFF]["content"] == "43" and out["regimes"][tc.THINKING_OFF] == tc.RENDER_REGIME_THINKING_OFF
    assert all(b["messages"][0] == {"role": "system", "content": "SI PREFIX TEXT"} for b in bodies)    # H8, every draw


def test_the_on_draw_classifier_reads_the_shape_by_the_certifications_rule():
    C = srun._classify_on_probe_reply
    gemma, qwen, glm = (tc.FAMILY_RENDER_FACTS[f] for f in ("gemma4", "qwen3_5", "glm4v"))
    def rec(content, reasoning=None, finish="stop"):
        return {"content": content, "reasoning": reasoning, "finish_reason": finish}
    # record_only (gemma)
    assert C(gemma, rec("43", "thinking")) == srun.PROBE_CLASS_THINK_SHAPE
    assert C(gemma, rec("43", "partial thought", "length")) == srun.PROBE_CLASS_THINK_SHAPE     # an open block at the cap yields reasoning
    assert C(gemma, rec("43")) == srun.PROBE_CLASS_NO_THINKING_EMITTED
    assert C(gemma, rec("", None, "stop")) == srun.PROBE_CLASS_NO_THINKING_EMITTED                # finished empty: declined too
    assert C(gemma, rec(None, None, "length")) == srun.PROBE_CLASS_AT_CAP_CHANNEL_LESS
    assert C(gemma, rec("", None, "max_tokens")) == srun.PROBE_CLASS_AT_CAP_CHANNEL_LESS
    assert C(gemma, rec("a long answer with no thought", None, "length")) == srun.PROBE_CLASS_NO_THINK_SHAPE
    assert C(gemma, rec("<channel|>43")) == srun.PROBE_CLASS_LEAKED_TAG
    assert C(gemma, rec("<|channel>thought\nx<channel|>43", "y")) == srun.PROBE_CLASS_LEAKED_TAG  # the leak outranks the channel
    # implicit_open (qwen, nemotron): the prompt opened the block — a finished draw without the close is the model
    # ending its turn INSIDE the block (gsm8k-paired200 rows 27 / 30), inconclusive like a decline, and retried
    assert C(qwen, rec("x</think>\n43")) == srun.PROBE_CLASS_THINK_SHAPE
    assert C(qwen, rec("still thinking", None, "length")) == srun.PROBE_CLASS_THINK_SHAPE          # cut inside the block
    assert C(qwen, rec("43")) == srun.PROBE_CLASS_UNCLOSED_BLOCK
    assert C(qwen, rec("", None, "stop")) == srun.PROBE_CLASS_UNCLOSED_BLOCK                       # finished empty: unclosed too
    assert C(qwen, rec("<think>\nx</think>\n43")) == srun.PROBE_CLASS_LEAKED_TAG
    assert C(qwen, rec("<think>\nx", None, "stop")) == srun.PROBE_CLASS_LEAKED_TAG                 # a stray open outranks unclosed
    nemo = tc.FAMILY_RENDER_FACTS["nemotron_h"]
    assert nemo.split_mode == tc.THINK_SPLIT_IMPLICIT_OPEN
    assert C(nemo, rec("17 plus 26 is 43. 43.")) == srun.PROBE_CLASS_UNCLOSED_BLOCK                 # row 30, verbatim shape
    assert C(nemo, rec("Alright, the user wants the answer to 17 plus 26" + " x" * 500)) == srun.PROBE_CLASS_UNCLOSED_BLOCK  # row 27
    assert C(nemo, rec("Okay, 17 plus 26 is 43.\n</think>\n43")) == srun.PROBE_CLASS_THINK_SHAPE     # row 28
    assert srun.PROBE_CLASS_UNCLOSED_BLOCK in srun.PROBE_CLASSES
    assert set(srun.PROBE_DECLINED_CLASSES) == {srun.PROBE_CLASS_NO_THINKING_EMITTED, srun.PROBE_CLASS_UNCLOSED_BLOCK}
    # explicit_tags (glm): the model opens the block itself, so a tag-free finished draw is a decline
    assert C(glm, rec("<think>x</think>43")) == srun.PROBE_CLASS_THINK_SHAPE
    assert C(glm, rec("43")) == srun.PROBE_CLASS_NO_THINKING_EMITTED
    assert C(glm, rec("</think>43")) == srun.PROBE_CLASS_NO_THINK_SHAPE
    # no fact: only the mode-free cap-cut class is readable
    assert C(None, rec(None, None, "length")) == srun.PROBE_CLASS_AT_CAP_CHANNEL_LESS
    assert C(None, rec("43")) == srun.PROBE_CLASS_NO_RENDER_FACT
    assert set(srun.PROBE_CLASSES) >= {srun.PROBE_CLASS_THINK_SHAPE, srun.PROBE_CLASS_LEAKED_TAG,
                                       srun.PROBE_CLASS_NO_THINKING_EMITTED, srun.PROBE_CLASS_AT_CAP_CHANNEL_LESS}


def test_the_declined_note_is_empty_unless_the_certifying_draw_declined():
    assert srun._render_probe_declined_note({"on_attempts": []}) == ""
    assert srun._render_probe_declined_note({"on_attempts": [{"class": srun.PROBE_CLASS_AT_CAP_CHANNEL_LESS,
                                                              "prompt_id": "arithmetic"}]}) == ""
    mixed = {"on_attempts": [{"class": srun.PROBE_CLASS_AT_CAP_CHANNEL_LESS, "prompt_id": "arithmetic"},
                             {"class": srun.PROBE_CLASS_NO_THINKING_EMITTED, "prompt_id": "arithmetic"},
                             {"class": srun.PROBE_CLASS_NO_THINKING_EMITTED, "prompt_id": "reasoning"}]}
    note = srun._render_probe_declined_note(mixed)
    assert "the model declined to open a thought block on 2 of 3 draws" in note and "D143" in note
    unclosed = {"on_attempts": [{"class": srun.PROBE_CLASS_UNCLOSED_BLOCK, "prompt_id": "arithmetic"},
                                {"class": srun.PROBE_CLASS_UNCLOSED_BLOCK, "prompt_id": "reasoning"},
                                {"class": srun.PROBE_CLASS_UNCLOSED_BLOCK, "prompt_id": "reasoning"}]}
    note = srun._render_probe_declined_note(unclosed)
    assert "ended its turn inside the prompt-opened thought block without closing it on 3 draws" in note
    assert "rule 1" in note and "arithmetic" in note and "reasoning" in note
    # the certifying draw decides which note: a decline after unclosed draws reads as the decline
    assert "declined to open" in srun._render_probe_declined_note({"on_attempts": [
        {"class": srun.PROBE_CLASS_UNCLOSED_BLOCK, "prompt_id": "arithmetic"},
        {"class": srun.PROBE_CLASS_NO_THINKING_EMITTED, "prompt_id": "reasoning"}]})
    assert srun._render_probe_declined_note({"on_attempts": [{"class": srun.PROBE_CLASS_NO_THINK_SHAPE,
                                                              "prompt_id": "arithmetic"}]}) == ""


# --------------------------------------------------------------------------- #
# gsm8k-paired200 rows 27 / 30: an implicit-open draw that FINISHED inside the block  #
# --------------------------------------------------------------------------- #
#
# The three real ON draws of the nemotron standups on one node (evidence.json, `render_probe.on`): row 27 rambled
# 1,103 chars inside the prompt-opened block and hit `stop` with no `</think>`; row 30 answered "17 plus 26 is 43. 43."
# and stopped the same way; row 28 closed the block and answered. Each was ONE draw: the classifier filed the first two
# as `no_think_shape` (no retry) and the certification refused. The probe decodes under the cell's SAMPLING ROW on the
# STEERED route (T 1.0 / top_p 0.95 for nemotron), so the draw differs per object and per standup — not greedy.

NEMO_UNCLOSED_RAMBLE = _reply("Alright, the user wants the answer to 17 plus 26 and said to reply with just the number. "
                              "Simple arithmetic here — 17 plus 26 is 43. " + "Still better check than skip. " * 30,
                              None, "stop", on=True)                                            # row 27's shape
NEMO_UNCLOSED_SHORT = _reply("17 plus 26 is 43. 43.", None, "stop", on=True)                     # row 30, verbatim
NEMO_CLOSED = _reply("Okay, the user is asking for a simple arithmetic sum: 17 plus 26. 10+20=30, 7+6=13, 30+13=43. "
                     "Deliver 43 only.\n</think>\n43", None, "stop", on=True)                     # row 28's shape
NEMO_CLOSED_REASONING = _reply("Monday: 12*8 + 7*15 = 96 + 105 = 201. Tuesday: 9*8 + 11*15 = 72 + 165 = 237. "
                               "237 - 201 = 36.\n</think>\n36", None, "stop", on=True)


def _nemo_ids():
    fact = tc.FAMILY_RENDER_FACTS["nemotron_h"]
    return {tc.THINKING_OFF: list(fact.off_ids), tc.THINKING_ON: list(fact.on_ids)}


def test_a_nemotron_draw_that_finished_inside_the_block_is_retried_with_the_reasoning_prompt_and_certified():
    """Row 30's shape on the first draw, a closed block on the second: two ON draws, the second under the reasoning
    prompt at seed 1 and the base budget, classes [unclosed_block, think_shape], certified — where the pinned tree
    refused after one draw."""
    bodies: list = []
    out = srun._render_probe(_scripted_chat(bodies, [NEMO_UNCLOSED_SHORT, NEMO_CLOSED_REASONING]), "petri__si_x",
                             lens=True, header_located=False, family="nemotron_h")
    on = _on_bodies(bodies)
    assert [b["messages"][-1]["content"] for b in on] == [srun.RENDER_PROBE_PROMPT, srun.RENDER_PROBE_REASONING_PROMPT]
    assert [b["seed"] for b in on] == [0, 1] and [b["max_tokens"] for b in on] == [srun.RENDER_PROBE_MAX_TOKENS] * 2
    assert [a["class"] for a in out["on_attempts"]] == [srun.PROBE_CLASS_UNCLOSED_BLOCK, srun.PROBE_CLASS_THINK_SHAPE]
    assert out["on_attempts"][0] == {"attempt": 1, "prompt_id": "arithmetic", "seed": 0, "max_tokens": 512,
                                     "finish_reason": "stop", "reasoning_chars": 0, "content_chars": 21,
                                     "class": srun.PROBE_CLASS_UNCLOSED_BLOCK}
    assert out["responses"][tc.THINKING_ON]["content"].endswith("</think>\n36")            # the certifying draw = the last
    record, error = srun._render_certification(out, family="nemotron_h", header_ids_by_arm=_nemo_ids(), measured_live=True)
    assert error is None and record["think_shape_certified"] is True and record["render_certified"] is True
    # row 27's rambling shape is the same class and the same retry
    bodies2: list = []
    out2 = srun._render_probe(_scripted_chat(bodies2, [NEMO_UNCLOSED_RAMBLE, NEMO_CLOSED]), "petri__si_x",
                              lens=True, header_located=False, family="nemotron_h")
    assert [a["class"] for a in out2["on_attempts"]] == [srun.PROBE_CLASS_UNCLOSED_BLOCK, srun.PROBE_CLASS_THINK_SHAPE]
    assert out2["on_attempts"][0]["content_chars"] > 1000 and len(_on_bodies(bodies2)) == 2


def test_three_unclosed_nemotron_draws_are_refused_with_the_unclosed_note_and_no_fourth_draw():
    """`think_shape` stays the ONLY accepting class: three unclosed draws exhaust the ladder (arithmetic seed 0,
    reasoning seed 1, reasoning seed 2), the certification refuses by name, and the note says what happened."""
    bodies: list = []
    out = srun._render_probe(_scripted_chat(bodies, [NEMO_UNCLOSED_SHORT, NEMO_UNCLOSED_RAMBLE, NEMO_UNCLOSED_SHORT]),
                             "petri__si_x", lens=True, header_located=False, family="nemotron_h")
    on = _on_bodies(bodies)
    assert len(on) == srun.RENDER_PROBE_MAX_ATTEMPTS == 3 and [b["seed"] for b in on] == [0, 1, 2]
    assert [b["messages"][-1]["content"] for b in on] == [srun.RENDER_PROBE_PROMPT] + [srun.RENDER_PROBE_REASONING_PROMPT] * 2
    assert [a["class"] for a in out["on_attempts"]] == [srun.PROBE_CLASS_UNCLOSED_BLOCK] * 3
    record, error = srun._render_certification(out, family="nemotron_h", header_ids_by_arm=_nemo_ids(), measured_live=True)
    assert record["render_certified"] is False and record["think_shape_certified"] is False
    assert error is not None and tc.REC_THINKING_UNSERVABLE in error
    assert "does not carry the family's think shape" in error
    assert "ended its turn inside the prompt-opened thought block without closing it on 3 draws" in error
    assert "not a render defect" in error and "no draws left" in error


def test_a_first_nemotron_draw_that_closes_is_one_draw_and_a_cap_cut_inside_the_block_is_the_shape():
    """Unchanged behaviour on the draws that already certified: row 28's closed block is conclusive at once, and a
    draw cut at the cap inside the block is the shape (no retry, no 'unclosed' reading)."""
    bodies: list = []
    out = srun._render_probe(_scripted_chat(bodies, [NEMO_CLOSED]), "petri__si_x", lens=True, header_located=False,
                             family="nemotron_h")
    assert len(_on_bodies(bodies)) == 1 and [a["class"] for a in out["on_attempts"]] == [srun.PROBE_CLASS_THINK_SHAPE]
    at_cap_inside = _reply("still working through the sum " * 40, None, "length", on=True)
    bodies2: list = []
    out2 = srun._render_probe(_scripted_chat(bodies2, [at_cap_inside]), "petri__si_x", lens=True, header_located=False,
                              family="nemotron_h")
    assert len(_on_bodies(bodies2)) == 1 and [a["class"] for a in out2["on_attempts"]] == [srun.PROBE_CLASS_THINK_SHAPE]
    _, error = srun._render_certification(out2, family="nemotron_h", header_ids_by_arm=_nemo_ids(), measured_live=True)
    assert error is None


NEMO_AT_CAP_CHANNEL_LESS = _reply(None, None, "length", on=True)                 # nothing in either channel, cut at the cap


def test_the_accepting_class_is_the_only_path_to_a_certified_render():
    """Review P2-1: `contracts._think_shape` reads an implicit-open completion cut at the cap as "inside the block"
    whatever the channels hold, so [unclosed_block, at_cap_channel_less, at_cap_channel_less] — reachable now that
    an unclosed first draw is retried — would certify nemotron's render on NO text. The certification gates on the
    certifying draw's recorded class: `think_shape` certifies, anything else refuses by name."""
    # (a) unclosed -> channel-less cap-cut at 512 -> the budget retry at 2,048, channel-less again: REFUSED
    bodies: list = []
    out = srun._render_probe(_scripted_chat(bodies, [NEMO_UNCLOSED_SHORT, NEMO_AT_CAP_CHANNEL_LESS, NEMO_AT_CAP_CHANNEL_LESS]),
                             "petri__si_x", lens=True, header_located=False, family="nemotron_h")
    on = _on_bodies(bodies)
    assert [a["class"] for a in out["on_attempts"]] == [srun.PROBE_CLASS_UNCLOSED_BLOCK,
                                                        srun.PROBE_CLASS_AT_CAP_CHANNEL_LESS, srun.PROBE_CLASS_AT_CAP_CHANNEL_LESS]
    assert [b["max_tokens"] for b in on] == [512, 512, 2048] and [b["seed"] for b in on] == [0, 1, 1]
    assert out["responses"][tc.THINKING_ON]["finish_reason"] == "length"          # the draw contracts._think_shape accepts
    record, error = srun._render_certification(out, family="nemotron_h", header_ids_by_arm=_nemo_ids(), measured_live=True)
    assert record["render_certified"] is False and record["think_shape_certified"] is False
    assert record[srun.CERTIFYING_DRAW_CLASS_FIELD] == srun.PROBE_CLASS_AT_CAP_CHANNEL_LESS
    assert error is not None and tc.REC_THINKING_UNSERVABLE in error
    assert "certifying ON draw's class is 'at_cap_channel_less', not 'think_shape'" in error
    assert "nothing emitted in either channel" in error
    # (b) unclosed -> closed: CERTIFIED, the class recorded
    bodies_b: list = []
    out_b = srun._render_probe(_scripted_chat(bodies_b, [NEMO_UNCLOSED_SHORT, NEMO_CLOSED_REASONING]), "petri__si_x",
                               lens=True, header_located=False, family="nemotron_h")
    record_b, error_b = srun._render_certification(out_b, family="nemotron_h", header_ids_by_arm=_nemo_ids(), measured_live=True)
    assert error_b is None and record_b["render_certified"] is True
    assert record_b[srun.CERTIFYING_DRAW_CLASS_FIELD] == srun.PROBE_CLASS_THINK_SHAPE
    # (c) channel-less cap-cuts only: the ladder takes exactly TWO such draws (512, then 2,048; no third) and REFUSES
    bodies_c: list = []
    out_c = srun._render_probe(_scripted_chat(bodies_c, [NEMO_AT_CAP_CHANNEL_LESS] * 3), "petri__si_x", lens=True,
                               header_located=False, family="nemotron_h")
    assert len(_on_bodies(bodies_c)) == 2 and [a["class"] for a in out_c["on_attempts"]] == [srun.PROBE_CLASS_AT_CAP_CHANNEL_LESS] * 2
    record_c, error_c = srun._render_certification(out_c, family="nemotron_h", header_ids_by_arm=_nemo_ids(), measured_live=True)
    assert record_c["render_certified"] is False and error_c is not None and "at_cap_channel_less" in error_c
    # ... and a hand-built record of three such draws refuses the same way (the gate reads the LAST class)
    three = {**out_c, "on_attempts": [{"class": srun.PROBE_CLASS_AT_CAP_CHANNEL_LESS, "prompt_id": "arithmetic"}] * 3}
    record_3, error_3 = srun._render_certification(three, family="nemotron_h", header_ids_by_arm=_nemo_ids(), measured_live=True)
    assert record_3["render_certified"] is False and error_3 is not None
    # (d) a probe that recorded NO ON draw certifies nothing
    none_rec, none_err = srun._render_certification({**out_b, "on_attempts": []}, family="nemotron_h",
                                                    header_ids_by_arm=_nemo_ids(), measured_live=True)
    assert none_rec["render_certified"] is False and none_err is not None and "no ON draw was recorded" in none_err
    # (e) a whitespace-only cap-cut is channel-less too (gemma looped on whitespace at the cap)
    C = srun._classify_on_probe_reply
    nemo = tc.FAMILY_RENDER_FACTS["nemotron_h"]
    assert C(nemo, {"content": " \n\n ", "reasoning": None, "finish_reason": "length"}) == srun.PROBE_CLASS_AT_CAP_CHANNEL_LESS
    assert C(nemo, {"content": "still working", "reasoning": None, "finish_reason": "length"}) == srun.PROBE_CLASS_THINK_SHAPE


def test_the_unclosed_retry_is_implicit_open_only_and_qwen_shares_it():
    """qwen3_5 is the other implicit-open family and takes the same retry; the record_only (gemma) and
    explicit_tags (glm) ladders are exactly as before (a finished tag-free draw there is `no_thinking_emitted`)."""
    bodies: list = []
    qwen_unclosed = _reply("43", None, "stop", on=True)
    qwen_closed = _reply("x</think>\n36", None, "stop", on=True)
    out = srun._render_probe(_scripted_chat(bodies, [qwen_unclosed, qwen_closed]), "petri__base", lens=False,
                             header_located=False, family="qwen3_5")
    assert [a["class"] for a in out["on_attempts"]] == [srun.PROBE_CLASS_UNCLOSED_BLOCK, srun.PROBE_CLASS_THINK_SHAPE]
    bodies_g: list = []
    out_g = srun._render_probe(_scripted_chat(bodies_g, [GEMMA_DECLINED, GEMMA_THOUGHT]), "petri__base", lens=False,
                               header_located=False, family="gemma4")
    assert [a["class"] for a in out_g["on_attempts"]] == [srun.PROBE_CLASS_NO_THINKING_EMITTED, srun.PROBE_CLASS_THINK_SHAPE]
    bodies_glm: list = []
    glm_declined = _reply("43", None, "stop", on=True)
    glm_thought = _reply("<think>working</think>36", None, "stop", on=True)
    out_glm = srun._render_probe(_scripted_chat(bodies_glm, [glm_declined, glm_thought]), "petri__base", lens=False,
                                 header_located=False, family="glm4v")
    assert [a["class"] for a in out_glm["on_attempts"]] == [srun.PROBE_CLASS_NO_THINKING_EMITTED, srun.PROBE_CLASS_THINK_SHAPE]


def test_the_probe_prompts_and_budgets():
    """The reasoning prompt takes several steps to a bare number (the working can only go into the thought channel),
    differs from the arithmetic prompt, and neither carries any family's think delimiter."""
    assert srun.RENDER_PROBE_MAX_ATTEMPTS == 3 and srun.RENDER_PROBE_MAX_TOKENS == 512 == srun.RENDER_PROBE_RETRY_MAX_TOKENS // 4
    assert srun.RENDER_PROBE_REASONING_PROMPT != srun.RENDER_PROBE_PROMPT
    assert srun.RENDER_PROBE_REASONING_PROMPT.endswith("Reply with just the number.")
    assert 9 * 8 + 11 * 15 - (12 * 8 + 7 * 15) == 36                                    # the problem's answer, as documented
    for text in (srun.RENDER_PROBE_PROMPT, srun.RENDER_PROBE_REASONING_PROMPT):
        for fact in tc.FAMILY_RENDER_FACTS.values():
            # a no-thinking-channel family (llama-paper-rows) has EMPTY delimiters — nothing to be absent
            for delim in (fact.think_open, fact.think_close):
                assert not delim or delim not in text, (fact.family, delim)
    assert srun.RENDER_PROBE_PROMPTS == {srun.RENDER_PROBE_PROMPT_ARITHMETIC: srun.RENDER_PROBE_PROMPT,
                                         srun.RENDER_PROBE_PROMPT_REASONING: srun.RENDER_PROBE_REASONING_PROMPT}


def test_the_generate_call_site_hands_the_probe_the_family_and_records_every_on_draw():
    src = (REPO / "scripts/served_capability_run.py").read_text()
    i = src.index("probe = _render_probe(_probe_chat(tc.PROBE_KIND_RENDER)")
    site = src[i:i + 900]
    assert "family=family" in site
    assert "_render_certification(" in site                                                    # the wrapper, not tc directly
    assert 'evidence["render_probe"] = {**probe["responses"], "on_attempts": probe["on_attempts"]}' in site
