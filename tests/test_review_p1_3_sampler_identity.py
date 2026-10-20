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


# --------------------------------------------------------------------------- P1-3

def _bon_world(tmp_path):
    import test_bon_served as tb
    backend = tb.RecordingBackend()
    proxy, routes = tb._proxy(tmp_path, backend)
    defense = bs.ServedDefense.over_proxy(proxy, virtual_model="petri__base", cell_id="none")
    return tb, backend, routes, defense


def test_p1_3_bon_refuses_to_resume_a_store_drawn_under_another_sampler(tmp_path):
    tb, backend, routes, defense = _bon_world(tmp_path)
    goals = tb._goals(2)
    out = tmp_path / "store"
    kw = dict(out_dir=out, routes=routes, n_max=3, max_concurrency=4, decode_label=bs.BON_HARNESS_DEFAULT)
    bs.bon_generate_served(defense, goals, **kw)                                   # the grandfathered row by name, complete
    manifest_before = (out / bs.BON_SERVED_MANIFEST).read_text()
    n_calls = backend.n_calls
    with pytest.raises(bs.BonServedError, match="drawn under a different sampler"):
        bs.bon_generate_served(defense, goals, decode=PRESET, **{**kw, "decode_label": "vendor"})
    # ... and the vendor-preset DEFAULT (2026-09-17) is another sampler too: it never resumes a T=1 store
    with pytest.raises(bs.BonServedError, match="drawn under a different sampler"):
        bs.bon_generate_served(defense, goals, model="gemma4e4b", **{**kw, "decode_label": None})
    assert backend.n_calls == n_calls                                              # nothing generated
    assert (out / bs.BON_SERVED_MANIFEST).read_text() == manifest_before           # nothing relabelled
    # the same sampler resumes idempotently
    bs.bon_generate_served(defense, goals, **kw)
    assert backend.n_calls == n_calls
    # a PARTIAL store under another sampler refuses too (no mixture): drop the last row and retry with vendor
    rows = (out / "bon_responses.jsonl").read_text().splitlines()
    (out / "bon_responses.jsonl").write_text("\n".join(rows[:-1]) + "\n")
    with pytest.raises(bs.BonServedError, match="drawn under a different sampler"):
        bs.bon_generate_served(defense, goals, decode=PRESET, **{**kw, "decode_label": "vendor"})


def test_p1_3_bon_legacy_unrecorded_rows_are_never_relabelled_as_a_preset(tmp_path):
    tb, backend, routes, defense = _bon_world(tmp_path)
    goals = tb._goals(1)
    out = tmp_path / "legacy"
    kw = dict(out_dir=out, routes=routes, n_max=2, max_concurrency=2, decode_label=bs.BON_HARNESS_DEFAULT)
    bs.bon_generate_served(defense, goals, **kw)
    # a legacy store (before 2026-09-10) recorded NO sampler and NO seed: strip every record key (a row
    # carrying a seed but no sampler, or a sampler but no seed, is UNVERIFIABLE — not legacy)
    rows = [json.loads(l) for l in (out / "bon_responses.jsonl").read_text().splitlines() if l.strip()]
    for r in rows:
        for k in ("decode", "bon_decode_preset", "sampling_seed", "seed_rule"):
            r.pop(k)
    (out / "bon_responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(bs.BonServedError, match="legacy"):
        bs.bon_generate_served(defense, goals, decode=PRESET, **{**kw, "decode_label": "vendor"})
    # the harness default may resume it (the same sampler those rows were drawn under), and the manifest
    # says how many rows carry no record instead of relabelling them
    bs.bon_generate_served(defense, goals, **kw)
    man = json.loads((out / bs.BON_SERVED_MANIFEST).read_text())
    assert man["n_legacy_unrecorded_rows"] == 2 and man["bon_decode_preset"] == "harness_default"


def test_p1_3_multiturn_store_refuses_a_different_target_decode_before_any_manifest_write(tmp_path):
    chat = _OverflowChat(window=8704, prompt=10)
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x", max_model_len=8704)
    vendor = {"do_sample": True, **PRESET, "cap": 512, "arm": "vendor_recommended", "seed_base": 1, "trial": 0}
    st.write_target_manifest(tmp_path, t, stage="final", decode=vendor)
    (tmp_path / "progress.jsonl").write_text(json.dumps({"goal_id": "g0", "outcome": "success"}) + "\n")
    # same sampler: fine; a different preset / greedy over a vendor store: refused, manifest untouched
    st.assert_target_decode_compatible(tmp_path, vendor)
    before = (tmp_path / "served_target.json").read_text()
    with pytest.raises(st.ServedTargetError, match="different target decode"):
        st.assert_target_decode_compatible(tmp_path, {**vendor, "temperature": 1.0})
    with pytest.raises(st.ServedTargetError, match="different target decode"):
        st.assert_target_decode_compatible(tmp_path, None)
    assert (tmp_path / "served_target.json").read_text() == before
    # a legacy store (no decode block, covered goals) refuses a preset — never relabelled
    st.write_target_manifest(tmp_path, t, stage="final")
    with pytest.raises(st.ServedTargetError, match="LEGACY"):
        st.assert_target_decode_compatible(tmp_path, vendor)
    st.assert_target_decode_compatible(tmp_path, None)          # greedy over a legacy greedy store: fine
    # an empty store (no progress) accepts any sampler
    (tmp_path / "progress.jsonl").unlink()
    st.assert_target_decode_compatible(tmp_path, vendor)
