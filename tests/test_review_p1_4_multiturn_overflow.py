"""Review round 1 (c9eaeb73..fa9eb1c4 — "no ship"): the six findings, each reproduced here first.

P1-4  the vendor-sampled multi-turn chat() path clamped on overflow (sent 512 then 204, recorded
      n_prompt_over_window=0) — the fixed no-clamped-multi-turn policy says: raise PromptOverWindow.
P1-3  switching samplers silently reused and relabelled results (BoN coverage keyed on (goal, sample,
      trial) only; served_target.json overwritten with the new decode before covered goals were skipped).
P1-1  every fullconst load check probed `none` with the registered prefix — the real proxy refuses that.
P1-2  the probe combined pooled p99 prompts and table caps without the run's cap or the window: a
      guaranteed 400 read as a "measured limit".
P1-5  the cap-ladder verdict did not govern the controls escalation loop.
P2-6  XSTest generation still included the 200 unsafe contrast prompts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from sisteer.capeval.serving import served_window as sw
from sisteer.capeval.serving.proxy import ProxyRefusal
from sisteer.eval.attacks import bon_served as bs
from sisteer.eval.attacks import served_target as st

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "tests", REPO / "scripts" / "sdfbase" / "full22"):
    sys.path.insert(0, str(p))

PRESET = {"temperature": 0.6, "top_p": 0.9}


def _ctx400(window, prompt, requested):
    return ProxyRefusal(
        f"This model's maximum context length is {window} tokens. However, you requested {requested} "
        f"output tokens and your prompt contains {prompt} input tokens, for a total of {prompt + requested} "
        "tokens. Please reduce the length of the input prompt or the number of requested output tokens.",
        http_status=400, code="context_length")


class _OverflowChat:
    """A proxy stand-in: the FIRST request at max_tokens > room is a context-length 400; a request that
    fits (max_tokens <= room) is answered — the clamped-retry path would succeed here, which is exactly
    the silent shortening the multi-turn policy forbids."""

    def __init__(self, window=8704, prompt=8500):
        self.window, self.prompt, self.payloads = window, prompt, []

    def __call__(self, payload):
        self.payloads.append(dict(payload))
        if self.prompt + int(payload["max_tokens"]) > self.window:
            raise _ctx400(self.window, self.prompt, int(payload["max_tokens"]))
        return {"choices": [{"message": {"content": "short reply"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": self.prompt, "completion_tokens": 3}}


# --------------------------------------------------------------------------- P1-4

def test_p1_4_vendor_sampled_multiturn_chat_raises_on_overflow_never_clamps(tmp_path):
    chat = _OverflowChat(window=8704, prompt=8500)
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x", timing_log=tmp_path / "t.jsonl",
                            max_model_len=8704)
    msgs = [{"role": "user", "content": "x"}]
    with pytest.raises(sw.PromptOverWindow) as ei:
        t.chat(msgs, max_new_tokens=512, do_sample=True, temperature=0.6, sampling_seed=7, decode=PRESET)
    assert ei.value.requested_max_tokens == 512
    assert [p["max_tokens"] for p in chat.payloads] == [512]         # ONE send, never a 204 retry
    assert t.n_prompt_over_window == 1 and t.n_clamped == 0
    # the greedy multi-turn path holds the same policy
    chat2 = _OverflowChat(window=8704, prompt=8500)
    t2 = st.ServedChatTarget(chat2, virtual_model="petri__x", cell_id="x", max_model_len=8704)
    with pytest.raises(sw.PromptOverWindow):
        t2.chat(msgs, max_new_tokens=512)
    assert [p["max_tokens"] for p in chat2.payloads] == [512] and t2.n_clamped == 0
    # the single-shot generate() keeps its deliberately supported once-only clamp (the static families' path)
    chat3 = _OverflowChat(window=8704, prompt=8500)
    t3 = st.ServedChatTarget(chat3, virtual_model="petri__x", cell_id="x", max_model_len=8704)
    out = t3.generate(["g"], max_new_tokens=512)
    assert out == ["short reply"] and [p["max_tokens"] for p in chat3.payloads] == [512, 204] and t3.n_clamped == 1
    # the manifest says which policy the multi-turn path ran under
    man = json.loads(st.write_target_manifest(tmp_path, t).read_text())
    assert man["window_clamp"]["multiturn_overflow_policy"] == "strict"
    assert man["window_clamp"]["n_prompt_over_window"] == 1


