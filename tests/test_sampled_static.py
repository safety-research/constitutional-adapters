"""The full22 SAMPLED re-run's STATIC lane (scripts/sdfbase/full22/sampled/DESIGN.md):
the static one-shot families replayed under a served model's vendor sampler preset.

CPU only, NO server, no GPU: stub chat transports stand in for the routing proxy.
What must hold:

1. the request body carries EXACTLY the preset's keys + the per-item seed (do_sample
   semantics), refuses any key outside the sampler set, and the greedy surfaces are
   byte-identical when no preset is given;
2. the detailed surfaces (ServedChatTarget / ServedStaticDefense.generate_detailed)
   return finish_reason / usage / max_tokens / sampling_seed per row, clamp fields
   only when the once-only window clamp fired;
3. run_static under ``decode`` writes the DESIGN row schema with the DESIGN seed, is
   BYTE-IDENTICAL without it, and refuses to mix decoders in one store;
4. jailbreak_run validates ``sampled_decode`` (trial / cap / keys / applies_to), threads
   it into the family ONLY when present, and the static served_target.json carries the
   decode block;
5. at_cap_static counts across shards and REFUSES a greedy store rather than printing 0;
6. escalate_static copies terminated rows, re-draws at-cap rows at the new cap with the
   SAME seed, records the divergence fields, is idempotent, and refuses an incomplete or
   greedy base.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import jailbreak_run  # noqa: E402
from sisteer.capeval.serving.proxy import FORBIDDEN_CLIENT_KEYS  # noqa: E402
from sisteer.eval.attacks import bon_served as bs  # noqa: E402
from sisteer.eval.attacks import served_target as st  # noqa: E402
from sisteer.eval.attacks import static_common as sc  # noqa: E402
from godmode_standin import godmode_standin  # noqa: E402,F401  (grok420 renders the invented G0DM0D3 stand-in)

SAMPLED_DIR = REPO / "scripts" / "sdfbase" / "full22" / "sampled"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, SAMPLED_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: the Qwen preset of DESIGN.md (every key present) and a gemma-like one (no penalties)
QWEN = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 1.5, "repetition_penalty": 1.0}
GEMMA = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
ARM = "vendor_recommended"
SEED_BASE = 20260909
DECODE = {"arm": ARM, **QWEN}


@dataclass(frozen=True)
class _Goal:
    goal_id: str
    text: str
    category: str = "harmful"


def _goals(n):
    return [_Goal(goal_id=f"g{i}", text=f"do thing {i}") for i in range(n)]


def _variants(goal, seed=0):
    return [("v0", f"{goal.text} / a"), ("v1", f"{goal.text} / b")]


def _rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


class _StubChat:
    """A sync chat transport: records payloads; the reply text, finish_reason and
    completion_tokens come from ``decide(payload) -> (text, finish_reason, ntok)``."""

    def __init__(self, decide=None, usage=True):
        self.payloads = []
        self.decide = decide or (lambda p: (f"reply:{p['messages'][-1]['content'][-12:]}",
                                            "stop", 5))
        self.usage = usage

    def __call__(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        text, fr, ntok = self.decide(payload)
        out = {"choices": [{"message": {"content": text}, "finish_reason": fr}],
               "petri_intervention_hash": "cc" * 8}
        if self.usage:
            out["usage"] = {"prompt_tokens": 11, "completion_tokens": ntok}
        return out


def _target(chat, **kw):
    kw.setdefault("virtual_model", "petri__x")
    kw.setdefault("cell_id", "none")
    return st.ServedChatTarget(chat, **kw)


# --------------------------------------------------------------------------- #
# 1. payload composition                                                        #
# --------------------------------------------------------------------------- #

def test_sampler_keys_never_overlap_the_proxys_reserved_keys():
    assert not (set(st.SAMPLER_KEYS) & set(FORBIDDEN_CLIENT_KEYS))
    assert "n" not in st.SAMPLER_KEYS            # every draw is its own request
    assert st.SAMPLER_KEYS is bs.SAMPLER_KEYS    # ONE definition


def test_payload_sends_exactly_the_preset_keys_plus_seed():
    chat = _StubChat()
    t = _target(chat)
    p = t._payload([{"role": "user", "content": "q"}], max_new_tokens=4096,
                   do_sample=True, temperature=1.0, sampling_seed=42, decode=QWEN)
    assert set(p) == {"model", "messages", "max_tokens", "temperature", "seed",
                      "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"}
    assert p["temperature"] == 0.7 and p["seed"] == 42 and p["max_tokens"] == 4096
    assert p["presence_penalty"] == 1.5 and p["top_k"] == 20 and p["min_p"] == 0.0
    # a preset without penalties sends NO penalty keys (exactly the keys present)
    g = t._payload([{"role": "user", "content": "q"}], max_new_tokens=8,
                   do_sample=True, temperature=1.0, sampling_seed=1, decode=GEMMA)
    assert set(g) == {"model", "messages", "max_tokens", "temperature", "seed",
                      "top_p", "top_k"}
    assert g["temperature"] == 1.0 and g["top_k"] == 64
    assert not (set(p) | set(g)) & set(FORBIDDEN_CLIENT_KEYS)


def test_payload_refusals():
    t = _target(_StubChat())
    msgs = [{"role": "user", "content": "q"}]

    def call(**kw):
        base = dict(max_new_tokens=8, do_sample=True, temperature=1.0, sampling_seed=1,
                    decode=QWEN)
        base.update(kw)
        return t._payload(msgs, **base)

    with pytest.raises(st.ServedTargetError, match="outside the sampler set"):
        call(decode={**QWEN, "n": 2})
    with pytest.raises(st.ServedTargetError, match="outside the sampler set"):
        call(decode={**QWEN, "cache_salt": "x"})       # a proxy-owned key never rides
    with pytest.raises(st.ServedTargetError, match="do_sample=True"):
        call(do_sample=False)
    with pytest.raises(st.ServedTargetError, match="sampling_seed"):
        call(sampling_seed=None)
    with pytest.raises(st.ServedTargetError, match="not a number"):
        call(decode={**QWEN, "top_k": True})
    with pytest.raises(st.ServedTargetError, match="no temperature"):
        call(decode={"top_p": 0.9})
    with pytest.raises(st.ServedTargetError, match="mapping"):
        call(decode=[("temperature", 0.7)])


def test_greedy_surfaces_are_byte_identical_without_a_preset():
    chat = _StubChat()
    t = _target(chat)
    t.chat([{"role": "user", "content": "hi"}], max_new_tokens=7)
    t.generate(["a", "b"], max_new_tokens=9)
    t.generate_detailed(["c"], max_new_tokens=3)               # no decode: greedy
    for p in chat.payloads:
        assert set(p) == {"model", "messages", "max_tokens", "temperature"}
        assert p["temperature"] == 0.0
    assert [p["max_tokens"] for p in chat.payloads] == [7, 9, 9, 3]
    assert t.n_target_forwards == 4 and t.n_sampled_requests == 0


# --------------------------------------------------------------------------- #
# 2. the detailed surfaces                                                      #
# --------------------------------------------------------------------------- #

def test_chat_target_generate_detailed_rows_and_counters():
    chat = _StubChat(decide=lambda p: ("out", "length" if "b" in p["messages"][-1]["content"]
                                       else "stop", 4096 if "b" in p["messages"][-1]["content"]
                                       else 12))
    t = _target(chat)
    recs = t.generate_detailed(["a", "b"], max_new_tokens=4096, decode=QWEN, seeds=[7, 8])
    assert [r["sampling_seed"] for r in recs] == [7, 8]
    assert [r["finish_reason"] for r in recs] == ["stop", "length"]
    assert [r["completion_tokens"] for r in recs] == [12, 4096]
    assert all(r["prompt_tokens"] == 11 and r["max_tokens"] == 4096 and r["text"] == "out"
               for r in recs)
    assert all("max_tokens_clamped_from" not in r and "window_mismatch" not in r for r in recs)
    assert [p["seed"] for p in chat.payloads] == [7, 8]
    assert all(p["temperature"] == 0.7 and p["presence_penalty"] == 1.5 for p in chat.payloads)
    assert t.n_target_forwards == 2 and t.n_sampled_requests == 2
    with pytest.raises(st.ServedTargetError, match="one sampling seed per prompt"):
        t.generate_detailed(["a"], max_new_tokens=8, decode=QWEN)
    with pytest.raises(st.ServedTargetError, match="aligned"):
        t.generate_detailed(["a", "b"], max_new_tokens=8, decode=QWEN, seeds=[1])


def test_chat_target_detailed_row_carries_clamp_fields_when_the_clamp_fires():
    """A context-length 400 is retried once with max_tokens clamped; the detailed
    row then says so (max_tokens actually sent + max_tokens_clamped_from)."""
    body = ("This model's maximum context length is 4864 tokens. However, you requested "
            "4096 output tokens and your prompt contains 1000 input tokens, for a total of "
            "5096 tokens. Please reduce the length of the input prompt or the number of "
            "requested output tokens.")
    err_payload = {"error": {"message": f"upstream vLLM returned 400: {body}",
                             "type": "invalid_request_error", "code": "upstream_error"}}

    class _Windowed(_StubChat):
        def __call__(self, payload):
            if payload["max_tokens"] > 3864:
                self.payloads.append(dict(payload))
                raise RuntimeError(f"proxy 400: {json.dumps(err_payload)[:400]}")
            return super().__call__(payload)

    chat = _Windowed()
    t = _target(chat, max_model_len=4864)
    (rec,) = t.generate_detailed(["long"], max_new_tokens=4096, decode=GEMMA, seeds=[3])
    assert rec["max_tokens"] == 3864 and rec["max_tokens_clamped_from"] == 4096
    assert rec["finish_reason"] == "stop" and rec["sampling_seed"] == 3
    assert "window_mismatch" not in rec
    assert [p["max_tokens"] for p in chat.payloads] == [4096, 3864]
    assert t.n_clamped == 1 and t.n_sampled_requests == 1


class _AsyncStub:
    """An async chat transport for ServedDefense (the fan-out's seam)."""

    def __init__(self, decide=None, fail_if=None):
        self.payloads = []
        self.decide = decide or (lambda p: ("r:" + p["messages"][-1]["content"], "stop", 3))
        self.fail_if = fail_if

    async def __call__(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        if self.fail_if is not None and self.fail_if(payload):
            raise RuntimeError("synthetic upstream failure")
        await asyncio.sleep(0)
        text, fr, ntok = self.decide(payload)
        return {"choices": [{"index": 0, "finish_reason": fr,
                             "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": ntok}}


def _static_defense(achat, loop, **kw):
    sd = bs.ServedDefense(achat, virtual_model="petri__base", cell_id="none")
    return bs.ServedStaticDefense(sd, loop=loop, max_concurrency=4, **kw)


def test_served_static_defense_detailed_is_concurrent_seeded_and_ordered():
    achat = _AsyncStub(decide=lambda p: ("r:" + p["messages"][-1]["content"],
                                         "length" if p["messages"][-1]["content"] == "p1"
                                         else "stop", 9))
    loop = asyncio.new_event_loop()
    try:
        d = _static_defense(achat, loop)
        recs = d.generate_detailed(["p0", "p1", "p2"], max_new_tokens=64, decode=QWEN,
                                   seeds=[10, 11, 12])
        # greedy surface unchanged: seed 0, temperature 0.0, no preset keys
        texts = d.generate(["q0", "q1"], max_new_tokens=8)
    finally:
        loop.close()
    assert [r["text"] for r in recs] == ["r:p0", "r:p1", "r:p2"]      # prompt order
    assert [r["sampling_seed"] for r in recs] == [10, 11, 12]
    assert [r["finish_reason"] for r in recs] == ["stop", "length", "stop"]
    assert all(r["max_tokens"] == 64 and r["completion_tokens"] == 9 for r in recs)
    by_prompt = {p["messages"][-1]["content"]: p for p in achat.payloads}
    assert by_prompt["p1"]["seed"] == 11 and by_prompt["p1"]["temperature"] == 0.7
    assert by_prompt["p1"]["top_k"] == 20 and by_prompt["p1"]["repetition_penalty"] == 1.0
    assert texts == ["r:q0", "r:q1"]
    assert by_prompt["q0"]["seed"] == 0 and by_prompt["q0"]["temperature"] == 0.0
    assert set(by_prompt["q0"]) == {"model", "messages", "temperature", "max_tokens", "seed"}
    assert d.n_target_forwards == 5


def test_served_static_defense_detailed_refusals_and_error_policy():
    loop = asyncio.new_event_loop()
    try:
        d = _static_defense(_AsyncStub(), loop)
        with pytest.raises(bs.BonServedError, match="one sampling seed per prompt"):
            d.generate_detailed(["p"], max_new_tokens=8, decode=QWEN)
        with pytest.raises(bs.BonServedError, match="aligned"):
            d.generate_detailed(["p", "q"], max_new_tokens=8, decode=QWEN, seeds=[1])
        with pytest.raises(bs.BonServedError, match="outside the sampler set"):
            d.generate_detailed(["p"], max_new_tokens=8, decode={**QWEN, "n": 3}, seeds=[1])
        # default policy: a failed request RAISES after the gather
        bad = _static_defense(_AsyncStub(fail_if=lambda p: p["messages"][-1]["content"] == "x"),
                              loop, max_retries=0)
        with pytest.raises(bs.BonServedError, match="request"):
            bad.generate_detailed(["ok", "x"], max_new_tokens=8, decode=QWEN, seeds=[1, 2])
        # record mode: a placeholder record with ``error`` set, never raised
        rec_mode = _static_defense(_AsyncStub(fail_if=lambda p: p["messages"][-1]["content"] == "x"),
                                   loop, max_retries=0, raise_on_error=False)
        recs = rec_mode.generate_detailed(["ok", "x"], max_new_tokens=8, decode=QWEN, seeds=[1, 2])
        assert recs[0]["error"] is None if "error" in recs[0] else "error" not in recs[0]
        assert recs[1]["error"].startswith("RuntimeError") and recs[1]["finish_reason"] is None
        assert recs[1]["text"].startswith("__SERVED_ERROR__") and rec_mode.n_errors == 1
    finally:
        loop.close()


# --------------------------------------------------------------------------- #
# 3. run_static under a preset                                                  #
# --------------------------------------------------------------------------- #

class _DetailedDefense:
    """A run_static defense with the served target's detailed surface (stub)."""

    def __init__(self, at_cap_if=None, finish_reason="stop", fail=False, error_record=False):
        self.calls = []
        self.n_target_forwards = 0
        self.batch_size = 4
        self.at_cap_if = at_cap_if or (lambda prompt: False)
        self.finish_reason = finish_reason
        self.fail = fail
        self.error_record = error_record

    def generate(self, prompts, *, max_new_tokens=512):
        self.n_target_forwards += len(prompts)
        return [f"greedy:{p}" for p in prompts]

    def generate_detailed(self, prompts, *, max_new_tokens=512, decode=None, seeds=None, labels=None):
        self.calls.append({"prompts": list(prompts), "max_new_tokens": max_new_tokens,
                           "decode": dict(decode) if decode else None, "seeds": list(seeds)})
        if self.fail:
            raise RuntimeError("upstream down")
        self.n_target_forwards += len(prompts)
        out = []
        for p, s in zip(prompts, seeds):
            at_cap = self.at_cap_if(p)
            fr = "length" if at_cap else self.finish_reason
            rec = {"text": f"sampled:{p}", "finish_reason": fr,
                   "completion_tokens": max_new_tokens if at_cap else 7,
                   "prompt_tokens": 3, "max_tokens": max_new_tokens, "sampling_seed": s}
            if self.error_record:
                rec["error"] = "RuntimeError: boom"
            out.append(rec)
        return out


class _GreedyOnly:
    def __init__(self):
        self.n_target_forwards = 0
        self.batch_size = 8

    def generate(self, prompts, *, max_new_tokens=512):
        self.n_target_forwards += len(prompts)
        return [f"greedy:{p}" for p in prompts]


def test_run_static_sampled_rows_carry_the_design_schema_and_seed(tmp_path):
    d = _DetailedDefense(at_cap_if=lambda p: p.endswith("/ b"))
    res = sc.run_static(d, _goals(2), out_dir=tmp_path / "s", family="pasttense",
                        variants_fn=_variants, trial=1, max_new_tokens=4096,
                        decode=DECODE, seed_base=SEED_BASE)
    rows = _rows(res["responses_path"])
    assert [(r["goal_id"], r["sample_idx"]) for r in rows] == \
        [("g0", 0), ("g0", 1), ("g1", 0), ("g1", 1)]
    block = {"do_sample": True, **QWEN, "cap": 4096, "arm": ARM, "seed_base": SEED_BASE,
             "trial": 1}
    for r in rows:
        assert r["decode"] == block
        assert r["sampling_seed"] == sc.static_sampling_seed(SEED_BASE, 1, "pasttense",
                                                             r["goal_id"], r["sample_idx"])
        assert r["max_tokens"] == 4096 and r["prompt_tokens"] == 3 and r["trial"] == 1
        assert r["at_cap"] == (r["finish_reason"] == "length")
        assert r["response"].startswith("sampled:")
    assert [r["at_cap"] for r in rows] == [False, True, False, True]
    assert [r["completion_tokens"] for r in rows] == [7, 4096, 7, 4096]
    # the ORDER of the added keys is stable (byte-stable rows for the analysis)
    assert list(rows[0]) == ["goal_id", "sample_idx", "trial", "variant_key", "variant_name",
                             "family", "variant", "attack_prompt", "response", "decode",
                             "sampling_seed", "finish_reason", "completion_tokens",
                             "prompt_tokens", "max_tokens", "at_cap"]
    # the request carried EXACTLY the preset (no arm) and the DESIGN seeds
    (call,) = d.calls
    assert call["decode"] == QWEN and call["max_new_tokens"] == 4096
    assert call["seeds"] == [r["sampling_seed"] for r in rows]
    assert res["decode"] == block and res["n_target_forwards_total"] == 4
    # the DESIGN formula, spelled out
    import hashlib
    key = f"{SEED_BASE}|1|pasttense|g0|1".encode()
    assert rows[1]["sampling_seed"] == int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def test_run_static_without_decode_is_byte_identical(tmp_path):
    """The greedy path's rows, progress and summary are exactly what they were:
    nine keys per row in this order, no decode anywhere."""
    d = _GreedyOnly()
    res = sc.run_static(d, _goals(1), out_dir=tmp_path / "g", family="fam",
                        variants_fn=_variants, max_new_tokens=16)
    lines = (tmp_path / "g" / "responses.jsonl").read_text().splitlines()
    assert lines == [
        json.dumps({"goal_id": "g0", "sample_idx": 0, "trial": 0, "variant_key": "0:0",
                    "variant_name": "v0", "family": "fam", "variant": "do thing 0 / a",
                    "attack_prompt": "do thing 0 / a", "response": "greedy:do thing 0 / a"}),
        json.dumps({"goal_id": "g0", "sample_idx": 1, "trial": 0, "variant_key": "1:0",
                    "variant_name": "v1", "family": "fam", "variant": "do thing 0 / b",
                    "attack_prompt": "do thing 0 / b", "response": "greedy:do thing 0 / b"}),
    ]
    prog = (tmp_path / "g" / "progress.jsonl").read_text().splitlines()
    assert prog == [json.dumps({"goal_id": "g0", "category": "harmful", "family": "fam",
                                "n_variants": 2, "n_target_forwards": 2, "outcome": "done",
                                "capped": False})]
    assert "decode" not in res
    assert set(res) == {"responses_path", "progress_path", "per_goal", "n_goals_run",
                        "n_goals_total", "n_goals_with_variants", "n_goals_zero_variants",
                        "n_target_forwards_total", "batch_size", "n_batches",
                        "n_goals_errored"}
    # a detailed-capable defense driven WITHOUT decode still takes the greedy path
    d2 = _DetailedDefense()
    sc.run_static(d2, _goals(1), out_dir=tmp_path / "g2", family="fam",
                  variants_fn=_variants, max_new_tokens=16)
    assert d2.calls == [] and _rows(tmp_path / "g2" / "responses.jsonl")[0]["response"] \
        .startswith("greedy:")


def test_run_static_refuses_to_mix_decoders_in_one_store(tmp_path):
    out = tmp_path / "mix"
    # greedy store first
    sc.run_static(_GreedyOnly(), _goals(1), out_dir=out, family="fam", variants_fn=_variants)
    with pytest.raises(sc.StaticStoreIncompleteError, match="GREEDY row"):
        sc.run_static(_DetailedDefense(), _goals(2), out_dir=out, family="fam",
                      variants_fn=_variants, decode=DECODE, seed_base=SEED_BASE)
    # sampled store: a greedy run refuses, a different preset refuses, the same block resumes
    out2 = tmp_path / "sampled"
    sc.run_static(_DetailedDefense(), _goals(1), out_dir=out2, family="fam",
                  variants_fn=_variants, max_new_tokens=64, decode=DECODE, seed_base=SEED_BASE)
    with pytest.raises(sc.StaticStoreIncompleteError, match="SAMPLED row"):
        sc.run_static(_GreedyOnly(), _goals(2), out_dir=out2, family="fam",
                      variants_fn=_variants, max_new_tokens=64)
    with pytest.raises(sc.StaticStoreIncompleteError, match="not this run's"):
        sc.run_static(_DetailedDefense(), _goals(2), out_dir=out2, family="fam",
                      variants_fn=_variants, max_new_tokens=64,
                      decode={"arm": ARM, **GEMMA}, seed_base=SEED_BASE)
    with pytest.raises(sc.StaticStoreIncompleteError, match="not this run's"):   # cap differs
        sc.run_static(_DetailedDefense(), _goals(2), out_dir=out2, family="fam",
                      variants_fn=_variants, max_new_tokens=8192, decode=DECODE,
                      seed_base=SEED_BASE)
    with pytest.raises(sc.StaticStoreIncompleteError, match="not this run's"):   # trial differs
        sc.run_static(_DetailedDefense(), _goals(2), out_dir=out2, family="fam",
                      variants_fn=_variants, max_new_tokens=64, decode=DECODE,
                      seed_base=SEED_BASE, trial=1)
    res = sc.run_static(_DetailedDefense(), _goals(2), out_dir=out2, family="fam",
                        variants_fn=_variants, max_new_tokens=64, decode=DECODE,
                        seed_base=SEED_BASE)
    assert res["n_goals_run"] == 1                              # g0 covered, g1 generated
    assert len(_rows(out2 / "responses.jsonl")) == 4


def test_run_static_sampled_refusals_are_loud(tmp_path):
    with pytest.raises(ValueError, match="seed_base"):
        sc.run_static(_DetailedDefense(), _goals(1), out_dir=tmp_path / "a", family="fam",
                      variants_fn=_variants, decode=DECODE)
    with pytest.raises(TypeError, match="generate_detailed"):        # never an HF leg
        sc.run_static(_GreedyOnly(), _goals(1), out_dir=tmp_path / "b", family="fam",
                      variants_fn=_variants, decode=DECODE, seed_base=SEED_BASE)
    with pytest.raises(ValueError, match="arm"):
        sc.run_static(_DetailedDefense(), _goals(1), out_dir=tmp_path / "c", family="fam",
                      variants_fn=_variants, decode=dict(QWEN), seed_base=SEED_BASE)
    with pytest.raises(ValueError, match="outside the sampler set"):
        sc.run_static(_DetailedDefense(), _goals(1), out_dir=tmp_path / "d", family="fam",
                      variants_fn=_variants, decode={**DECODE, "n": 2}, seed_base=SEED_BASE)
    # a failed sampled chunk RAISES and finalizes nothing (goals stay uncovered)
    with pytest.raises(RuntimeError, match="upstream down"):
        sc.run_static(_DetailedDefense(fail=True), _goals(1), out_dir=tmp_path / "e",
                      family="fam", variants_fn=_variants, decode=DECODE, seed_base=SEED_BASE)
    assert not (tmp_path / "e" / "progress.jsonl").exists()
    assert not (tmp_path / "e" / "responses.jsonl").exists()
    with pytest.raises(RuntimeError, match="error placeholder"):
        sc.run_static(_DetailedDefense(error_record=True), _goals(1), out_dir=tmp_path / "f",
                      family="fam", variants_fn=_variants, decode=DECODE, seed_base=SEED_BASE)
    with pytest.raises(RuntimeError, match="finish_reason"):
        sc.run_static(_DetailedDefense(finish_reason=None), _goals(1), out_dir=tmp_path / "h",
                      family="fam", variants_fn=_variants, decode=DECODE, seed_base=SEED_BASE)
    assert not (tmp_path / "h" / "progress.jsonl").exists()


def test_run_static_sampled_through_a_served_chat_target_stamps_the_engine(tmp_path):
    """THE CALLER'S HALF of the engine stamp: the sampled-static lane hands
    ``run_static`` the ``ServedChatTarget`` ``build_served_target`` returns — NOT
    ``bon_served.ServedStaticDefense`` — and ``run_static`` reads the tag OFF THE DEFENSE
    (``getattr(defense, "engine_tag", None)``), so a served target class without the attribute
    writes rows with neither ``engine`` nor ``serving_engine`` while its multi-turn sibling
    (``serving_engine_of(defense)``, keyed on ``serving_engine`` alone) stamps. The writer tests
    drive ``run_static`` with the adapter that carries the tag by construction; this one drives
    it with the class the sampled lane actually passes."""
    chat = _StubChat()
    target = _target(chat)
    out = tmp_path / "s"
    res = sc.run_static(target, _goals(2), out_dir=out, family="pasttense", variants_fn=_variants,
                        decode=DECODE, seed_base=SEED_BASE)
    rows = _rows(res["responses_path"])
    assert len(rows) == 4
    assert all(r["engine"] == bs.SERVED_ENGINE for r in rows), [sorted(r) for r in rows]
    # a scripted target with no proxy identity block is lens BY CONSTRUCTION (serving_engine_from_identity)
    assert all(r[st.SERVING_ENGINE_FIELD] == st.SERVING_ENGINE_LENS for r in rows)
    assert all(r["sampling_seed"] == sc.static_sampling_seed(SEED_BASE, 0, "pasttense", r["goal_id"], r["sample_idx"])
               and r["decode"]["arm"] == ARM for r in rows)
    # the tag arms the resume guard: a second run of the same target over its own store is admitted ...
    res2 = sc.run_static(target, _goals(3), out_dir=out, family="pasttense", variants_fn=_variants,
                         decode=DECODE, seed_base=SEED_BASE)
    rows2 = _rows(res2["responses_path"])
    assert len(rows2) == 6 and all(r["engine"] == bs.SERVED_ENGINE for r in rows2)
    # ... and the stamped tag is what the guard compares: a defense of ANOTHER engine tag, same decode,
    # is refused off this store before the decode compare (nothing appended)
    class _OtherEngine(_DetailedDefense):
        engine_tag = "other_engine"
        serving_engine = st.SERVING_ENGINE_LENS
    with pytest.raises(sc.StaticStoreIncompleteError, match="not 'other_engine'"):
        sc.run_static(_OtherEngine(), _goals(4), out_dir=out, family="pasttense", variants_fn=_variants,
                      decode=DECODE, seed_base=SEED_BASE)
    assert len(_rows(res2["responses_path"])) == 6


def test_served_chat_target_declares_its_engine_tag_structurally():
    """The tag is a CLASS attribute of ``ServedChatTarget`` — the same shape as
    ``bon_served.ServedStaticDefense.engine_tag`` — so every caller that builds a served target
    (jailbreak_run's static / sampled dispatch, the escalation driver, a future one) hands
    ``run_static`` a tagged defense without remembering to; never a per-call-site patch."""
    assert "engine_tag" in vars(st.ServedChatTarget), "engine_tag must be declared on the class"
    assert st.ServedChatTarget.engine_tag == bs.SERVED_ENGINE == bs.ServedStaticDefense.engine_tag
    assert _target(_StubChat()).engine_tag == bs.SERVED_ENGINE


@pytest.mark.parametrize("family", ["pasttense", "codeattack", "grok420"])
def test_family_entry_points_thread_decode_into_run_static(tmp_path, family, godmode_standin):
    from sisteer.eval.attacks import get_attack

    d = _DetailedDefense()
    res = get_attack(family)(d, _goals(1), out_dir=tmp_path / family, max_new_tokens=32,
                             decode=DECODE, seed_base=SEED_BASE)
    rows = _rows(res["responses_path"])
    assert rows and all(r["decode"]["arm"] == ARM and r["decode"]["cap"] == 32 for r in rows)
    assert all(r["sampling_seed"] == sc.static_sampling_seed(SEED_BASE, 0, family, r["goal_id"],
                                                             r["sample_idx"]) for r in rows)


def test_all_four_sampled_families_accept_the_passthrough():
    from sisteer.eval.attacks import get_attack

    for fam in ("pasttense", "codeattack", "renellm", "grok420"):
        params = inspect.signature(get_attack(fam)).parameters
        assert "decode" in params and "seed_base" in params, fam
        assert params["decode"].default is None and params["seed_base"].default is None


# --------------------------------------------------------------------------- #
# 4. jailbreak_run: the sampled_decode block                                    #
# --------------------------------------------------------------------------- #

def _sampled_block(**over):
    block = {"arm": ARM, "trial": 0, "n_trials": 2, "seed_base": SEED_BASE, **QWEN,
             "source": "model card", "source_kind": "model_card",
             "applies_to": ["static_families", "controls"], "static_cap": 4096,
             "controls_cap": 512}
    block.update(over)
    return block


def _cfg(**over):
    cfg = {"model": "org/model", "model_revision": "r1", "dtype": "bfloat16",
           "private_run": "run_sampled_t0",
           "attacks": {"static_max_new_tokens": 4096, "target_max_new_tokens": 512},
           "sampled_decode": _sampled_block()}
    cfg.update(over)
    return cfg


def test_sampled_decode_validator_happy_path_and_absence():
    assert jailbreak_run._sampled_decode({"attacks": {}}, 0) is None
    got = jailbreak_run._sampled_decode(_cfg(), 0)
    assert got == {"decode": {"arm": ARM, **QWEN}, "seed_base": SEED_BASE, "trial": 0}
    # a preset without penalties keeps exactly its keys
    g = jailbreak_run._sampled_decode(
        _cfg(sampled_decode={k: v for k, v in _sampled_block().items()
                             if k not in ("min_p", "presence_penalty", "repetition_penalty")}), 0)
    assert set(g["decode"]) == {"arm", "temperature", "top_p", "top_k"}
    # the escalation config's extra key is allowed
    esc = _cfg(sampled_decode=_sampled_block(static_cap=8192, escalated_from={
        "kind": "static", "cap": 4096, "private_run": "run_sampled_t0"}),
        attacks={"static_max_new_tokens": 8192})
    assert jailbreak_run._sampled_decode(esc, 0)["decode"]["temperature"] == 0.7
    # every provenance kind the lanes emit (Nemotron's user-supplied preset is a
    # vendor_recommendation), and GLM's repetition_penalty 1.1 rides the preset
    for kind in ("generation_config", "model_card", "vendor_recommendation"):
        blk = _sampled_block(source_kind=kind, repetition_penalty=1.1)
        got = jailbreak_run._sampled_decode(_cfg(sampled_decode=blk), 0)
        assert got["decode"]["repetition_penalty"] == 1.1


@pytest.mark.parametrize("over, trial, match", [
    ({"bogus": 1}, 0, "outside the allowed set"),
    ({"n": 2}, 0, "outside the allowed set"),
    ({"trial": 1}, 0, "trial"),
    ({"static_cap": 512}, 0, "static_cap"),
    ({"applies_to": ["controls"]}, 0, "static_families"),
    ({"top_k": True}, 0, "not a number"),
    ({"arm": ""}, 0, "arm"),
    ({"seed_base": "20260909"}, 0, "seed_base"),
    ({"source_kind": "vendor_recomendation"}, 0, "source_kind"),      # a typo refuses
])
def test_sampled_decode_validator_refusals(over, trial, match):
    with pytest.raises(SystemExit, match=match):
        jailbreak_run._sampled_decode(_cfg(sampled_decode=_sampled_block(**over)), trial)
    with pytest.raises(SystemExit, match="lacks required"):
        blk = _sampled_block()
        del blk["static_cap"]
        jailbreak_run._sampled_decode(_cfg(sampled_decode=blk), 0)
    with pytest.raises(SystemExit, match="static_cap"):     # cap drifted in attacks
        jailbreak_run._sampled_decode(_cfg(attacks={"static_max_new_tokens": 2048}), 0)


def _recorder():
    rec = {}

    def fake(defense, goals_arg, **kwargs):
        rec.update(defense=defense, goals=[g.goal_id for g in goals_arg], **kwargs)
        return {"n_goals_run": len(goals_arg), "n_goals_total": len(goals_arg),
                "n_target_forwards_total": 0, "per_goal": []}
    return rec, fake


def test_run_replayable_family_threads_decode_only_when_present(monkeypatch):
    import sisteer.eval.attacks as attacks_mod

    rec, fake = _recorder()
    monkeypatch.setattr(attacks_mod, "get_attack", lambda name: fake)
    cfg = {"attacks": {"static_max_new_tokens": 4096, "pasttense": {"seed": 5}}}
    jailbreak_run._run_replayable_family("pasttense", "DEF", _goals(2), cfg, "/tmp/o", trial=1)
    assert "decode" not in rec and "seed_base" not in rec           # byte-identical call
    rec.clear()
    jailbreak_run._run_replayable_family("pasttense", "DEF", _goals(2), cfg, "/tmp/o", trial=1,
                                         decode=DECODE, seed_base=SEED_BASE)
    assert rec["decode"] == DECODE and rec["seed_base"] == SEED_BASE
    assert rec["trial"] == 1 and rec["max_new_tokens"] == 4096 and rec["seed"] == 5
    with pytest.raises(SystemExit, match="fixed scope"):           # scope: the four only
        jailbreak_run._run_replayable_family("pap", "DEF", _goals(2), cfg, "/tmp/o",
                                             decode=DECODE, seed_base=SEED_BASE)
    with pytest.raises(SystemExit, match="fixed scope"):
        jailbreak_run._run_replayable_family("amplegcg_plus", "DEF", _goals(2), cfg, "/tmp/o",
                                             decode=DECODE, seed_base=SEED_BASE)
    with pytest.raises(SystemExit, match="seed_base"):
        jailbreak_run._run_replayable_family("pasttense", "DEF", _goals(2), cfg, "/tmp/o",
                                             decode=DECODE)


def _served_args(tmp_path, **over):
    base = dict(attack="pasttense", defense="none", shard=0, nshards=1, xstest=False,
                confirmatory=True, trial=0, private_root=str(tmp_path / "state"),
                # the vendor-preset decode is the DEFAULT; this
                # world's synthetic model has no vendor row, so its greedy runs OPT IN explicitly
                greedy=True,
                target_engine="served", target_proxy_url="http://127.0.0.1:7/v1",
                target_routing_table=str(tmp_path / "rt.json"), positions=None, scale=1.0,
                goal_ids_file=None, expect_pair_n_streams=None, target_max_model_len=4864)
    base.update(over)
    return SimpleNamespace(**base)


def _wire_cmd_generate(monkeypatch, chat, goals):
    import sisteer.models as models_mod

    def boom(*a, **k):
        raise AssertionError("no HF load on the served path")
    monkeypatch.setattr(models_mod, "load_model_and_tokenizer", boom)
    target = st.ServedChatTarget(chat, virtual_model="petri__base", cell_id="none",
                                 route_info={"virtual_model": "petri__base",
                                             "intervention_hash": "aa" * 8})
    target.evidence = {"steering_bites": {"applicable": False, "reason": "unhooked"}}
    monkeypatch.setattr(st, "build_served_target", lambda **kw: target)
    monkeypatch.setattr(jailbreak_run, "_valid_defense_ids", lambda cfg: {"none"})
    monkeypatch.setattr(jailbreak_run, "_eval_goals", lambda cfg, *, confirmatory: goals)
    monkeypatch.setattr(jailbreak_run, "_resolve_attacker_pool", lambda args, cfg: None)
    return target


def test_cmd_generate_static_served_sampled_writes_rows_and_manifest(monkeypatch, tmp_path):
    chat = _StubChat(decide=lambda p: ("txt", "length" if "past" in p["messages"][-1]["content"]
                                       else "stop", 4096))
    target = _wire_cmd_generate(monkeypatch, chat, _goals(2))
    cfg = _cfg(private_run="jb_sampled_t0")
    jailbreak_run.cmd_generate(_served_args(tmp_path), cfg)
    out = tmp_path / "state" / "jb_sampled_t0" / "pasttense" / "none" / "shard00"
    rows = _rows(out / "responses.jsonl")
    block = {"do_sample": True, **QWEN, "cap": 4096, "arm": ARM, "seed_base": SEED_BASE,
             "trial": 0}
    assert rows and all(r["decode"] == block and "at_cap" in r for r in rows)
    assert all(p["temperature"] == 0.7 and p["seed"] == r["sampling_seed"]
               and p["max_tokens"] == 4096 for p, r in zip(chat.payloads, rows))
    doc = json.loads((out / "served_target.json").read_text())
    assert doc["decode"] == block and doc["manifest_stage"] == "final"
    assert doc["n_sampled_requests"] == len(rows) == target.n_sampled_requests
    assert doc["window_clamp"]["n_target_forwards"] == len(rows)
    # trial mismatch refuses before anything is written
    with pytest.raises(SystemExit, match="trial"):
        jailbreak_run.cmd_generate(_served_args(tmp_path, trial=1),
                                   _cfg(private_run="jb_other"))
    assert not (tmp_path / "state" / "jb_other").exists()


def test_cmd_generate_static_served_without_sampled_decode_is_greedy(monkeypatch, tmp_path):
    chat = _StubChat()
    _wire_cmd_generate(monkeypatch, chat, _goals(1))
    cfg = _cfg(private_run="jb_greedy")
    del cfg["sampled_decode"]
    jailbreak_run.cmd_generate(_served_args(tmp_path), cfg)
    out = tmp_path / "state" / "jb_greedy" / "pasttense" / "none" / "shard00"
    rows = _rows(out / "responses.jsonl")
    # the served row's key set: the static schema plus the two engine provenance keys every served
    # row carries (``engine`` + ``serving_engine``; the pin without them, 2026-09-10 -> 2026-09-17,
    # was asserting the defect: an untagged ServedChatTarget wrote HF-shaped rows)
    assert rows and all(set(r) == {"goal_id", "sample_idx", "trial", "variant_key",
                                   "variant_name", "family", "variant", "attack_prompt",
                                   "response", "engine", st.SERVING_ENGINE_FIELD} for r in rows)
    assert all(p["temperature"] == 0.0 and "seed" not in p for p in chat.payloads)
    doc = json.loads((out / "served_target.json").read_text())
    assert "decode" not in doc and "n_sampled_requests" not in doc
    assert doc["manifest_stage"] == "final"


def test_cmd_generate_static_served_rows_carry_engine_provenance(monkeypatch, tmp_path):
    """End to end through the REAL dispatch (2026-09-17): a served sampled-static store's rows
    carry ``engine`` (``SERVED_ENGINE``) and ``serving_engine`` beside the sampled schema, and the
    manifest names the same serving engine the rows do; the greedy served run stamps both too. Before
    this the served static path handed ``run_static`` an untagged ``ServedChatTarget`` and every
    served static store came out with neither key — indistinguishable from an HF store to every
    reader and to the resume guard."""
    chat = _StubChat()
    target = _wire_cmd_generate(monkeypatch, chat, _goals(2))
    cfg = _cfg(private_run="jb_prov_sampled")
    jailbreak_run.cmd_generate(_served_args(tmp_path), cfg)
    out = tmp_path / "state" / "jb_prov_sampled" / "pasttense" / "none" / "shard00"
    rows = _rows(out / "responses.jsonl")
    assert len(rows) == 4 and all("decode" in r and "sampling_seed" in r for r in rows)
    assert all(r["engine"] == bs.SERVED_ENGINE for r in rows), [sorted(r) for r in rows]
    assert all(r[st.SERVING_ENGINE_FIELD] == target.serving_engine == st.SERVING_ENGINE_LENS for r in rows)
    doc = json.loads((out / "served_target.json").read_text())
    assert doc[st.SERVING_ENGINE_FIELD] == rows[0][st.SERVING_ENGINE_FIELD]
    # the greedy served run (explicit --greedy, no block): the same two provenance keys, no sampler keys
    greedy = _cfg(private_run="jb_prov_greedy")
    del greedy["sampled_decode"]
    jailbreak_run.cmd_generate(_served_args(tmp_path), greedy)
    grows = _rows(tmp_path / "state" / "jb_prov_greedy" / "pasttense" / "none" / "shard00" / "responses.jsonl")
    assert grows and all(r["engine"] == bs.SERVED_ENGINE and r[st.SERVING_ENGINE_FIELD] == st.SERVING_ENGINE_LENS
                         and "decode" not in r for r in grows)


def test_cmd_generate_sampled_decode_refuses_the_hf_target(monkeypatch, tmp_path):
    import sisteer.models as models_mod

    monkeypatch.setattr(models_mod, "load_model_and_tokenizer", lambda *a, **k: (None, None))
    monkeypatch.setattr(jailbreak_run, "_valid_defense_ids", lambda cfg: {"none"})
    monkeypatch.setattr(jailbreak_run, "_eval_goals", lambda cfg, *, confirmatory: _goals(1))
    monkeypatch.setattr(jailbreak_run, "_resolve_attacker_pool", lambda args, cfg: None)
    from sisteer.eval import jailbreak as jb_mod
    monkeypatch.setattr(jb_mod, "build_defense", lambda *a, **k: _GreedyOnly())
    with pytest.raises(SystemExit, match="target-engine served"):
        jailbreak_run.cmd_generate(_served_args(tmp_path, target_engine="hf"), _cfg())


# --------------------------------------------------------------------------- #
# 5. at_cap_static                                                              #
# --------------------------------------------------------------------------- #

def _write_store(root, run, attack, cell, shards):
    """shards: list of row lists -> shardNN/responses.jsonl (+ a done progress record per goal)."""
    for i, rows in enumerate(shards):
        d = root / run / attack / cell / f"shard{i:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        goals = {}
        for r in rows:
            goals[r["goal_id"]] = goals.get(r["goal_id"], 0) + 1
        (d / "progress.jsonl").write_text("".join(
            json.dumps({"goal_id": g, "category": "harmful", "family": attack, "n_variants": n,
                        "n_target_forwards": n, "outcome": "done", "capped": False}) + "\n"
            for g, n in goals.items()))


def _srow(gid, idx, fr, **over):
    block = {"do_sample": True, **QWEN, "cap": 4096, "arm": ARM, "seed_base": SEED_BASE, "trial": 0}
    r = {"goal_id": gid, "sample_idx": idx, "trial": 0, "variant_key": f"{idx}:0",
         "variant_name": "v", "family": "pasttense", "variant": f"{gid}/{idx}",
         "attack_prompt": f"{gid}/{idx}", "response": f"resp {gid} {idx}", "decode": block,
         "sampling_seed": sc.static_sampling_seed(SEED_BASE, 0, "pasttense", gid, idx),
         "finish_reason": fr, "completion_tokens": 4096 if fr == "length" else 10,
         "prompt_tokens": 5, "max_tokens": 4096, "at_cap": fr == "length"}
    r.update(over)
    return r


def test_at_cap_static_counts_across_shards(tmp_path, capsys):
    acs = _load_script("at_cap_static")
    _write_store(tmp_path, "run", "pasttense", "none", [
        [_srow("g0", 0, "stop"), _srow("g0", 1, "length")],
        [_srow("g1", 0, "length"), _srow("g1", 1, "stop"), _srow("g2", 0, "stop")],
    ])
    argv = ["--private-root", str(tmp_path), "--private-run", "run", "--attack", "pasttense",
            "--defense", "none"]
    assert acs.main(argv) == 0
    out = capsys.readouterr()
    assert out.out.strip() == "2" and "rows=5" in out.err
    assert acs.main(argv + ["--json"]) == 0
    doc = json.loads(capsys.readouterr().out.strip())
    assert doc == {"private_run": "run", "attack": "pasttense", "defense": "none",
                   "n_shards": 2, "n_rows": 5, "n_at_cap": 2}
    # zero at cap is a legitimate 0 with exit 0
    _write_store(tmp_path, "run0", "pasttense", "none", [[_srow("g0", 0, "stop")]])
    assert acs.main(["--private-root", str(tmp_path), "--private-run", "run0",
                     "--attack", "pasttense", "--defense", "none"]) == 0
    assert capsys.readouterr().out.strip() == "0"


def test_at_cap_static_refuses_a_greedy_or_broken_store(tmp_path, capsys):
    acs = _load_script("at_cap_static")

    def run(run_name):
        return acs.main(["--private-root", str(tmp_path), "--private-run", run_name,
                         "--attack", "pasttense", "--defense", "none"])

    greedy = {k: v for k, v in _srow("g0", 0, "stop").items()
              if k not in ("decode", "sampling_seed", "finish_reason", "completion_tokens",
                           "prompt_tokens", "max_tokens", "at_cap")}
    _write_store(tmp_path, "greedy", "pasttense", "none", [[greedy]])
    assert run("greedy") == 2 and "GREEDY" in capsys.readouterr().err
    assert run("missing") == 2 and "no shard" in capsys.readouterr().err
    _write_store(tmp_path, "empty", "pasttense", "none", [[]])
    assert run("empty") == 2 and "ZERO rows" in capsys.readouterr().err
    _write_store(tmp_path, "incons", "pasttense", "none", [[_srow("g0", 0, "length", at_cap=False)]])
    assert run("incons") == 2 and "inconsistent" in capsys.readouterr().err
    _write_store(tmp_path, "torn", "pasttense", "none", [[_srow("g0", 0, "stop")]])
    with (tmp_path / "torn" / "pasttense" / "none" / "shard00" / "responses.jsonl").open("a") as f:
        f.write('{"goal_id": "g1", "sample_idx"')
    assert run("torn") == 2 and "torn" in capsys.readouterr().err
    nofr = _srow("g0", 0, "stop")
    del nofr["finish_reason"]
    _write_store(tmp_path, "nofr", "pasttense", "none", [[nofr]])
    assert run("nofr") == 2 and "finish_reason" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# 6. escalate_static                                                            #
# --------------------------------------------------------------------------- #

BASE_RUN = "run_sampled_t0"
ESC_RUN = "run_sampled_t0_cap8192"


def _base_store(root, *, prefix="g"):
    """A complete sampled base at 4096 over two shards: g0/1 hit the cap on
    sample 1, g2 terminated, g3/0 hit the cap (five rows, three at cap)."""
    _write_store(root, BASE_RUN, "pasttense", "none", [
        [_srow("g0", 0, "stop"), _srow("g0", 1, "length"),
         _srow("g1", 0, "stop"), _srow("g1", 1, "length")],
        [_srow("g2", 0, "stop"), _srow("g3", 0, "length"), _srow("g3", 1, "stop")],
    ])


def _esc_config(tmp_path, **over):
    cfg = _cfg(private_run=ESC_RUN, attacks={"static_max_new_tokens": 8192,
                                              "target_max_new_tokens": 512},
               sampled_decode=_sampled_block(static_cap=8192, escalated_from={
                   "kind": "static", "cap": 4096, "private_run": BASE_RUN}))
    cfg.update(over)
    p = tmp_path / "esc.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _whitespace_tokenize(text):
    return text.split()


def _wire_escalate(monkeypatch, es, chat):
    target = st.ServedChatTarget(chat, virtual_model="petri__base", cell_id="none",
                                 route_info={"virtual_model": "petri__base"})
    target.evidence = {"steering_bites": {"applicable": False, "reason": "unhooked"}}
    built = {}

    def fake_build(**kw):
        built.update(kw)
        return target
    monkeypatch.setattr(st, "build_served_target", fake_build)
    monkeypatch.setattr(es, "load_tokenizer_for", lambda cfg: _whitespace_tokenize)
    monkeypatch.setattr(jailbreak_run, "_valid_defense_ids", lambda cfg: {"none"})
    return target, built


def _esc_argv(tmp_path, cfg_path, **over):
    args = {"--config": str(cfg_path), "--attack": "pasttense", "--defense": "none",
            "--private-root": str(tmp_path), "--proxy-url": "http://127.0.0.1:7/v1",
            "--routing-table": str(tmp_path / "rt.json"), "--max-model-len": "10240",
            "--concurrency": "3"}
    args.update(over)
    return [x for kv in args.items() for x in kv]


def _redraw(payload):
    """The re-draw oracle by item: g0/1 -> identical text, terminated below the base cap
    (boundary landing); g1/1 -> prefix extension, still terminating above the base cap;
    g3/0 -> a fork that is STILL at cap at 8192."""
    prompt = payload["messages"][-1]["content"]
    if prompt == "g0/1":
        return "resp g0 1", "stop", 900
    if prompt == "g1/1":
        return "resp g1 1 and then more", "stop", 5000
    if prompt == "g3/0":
        return "resp g3 X forked", "length", 8192
    raise AssertionError(f"unexpected re-draw of {prompt!r} (a terminated row was re-drawn)")


def test_escalate_static_copies_terminated_rows_and_redraws_at_cap_rows(monkeypatch, tmp_path, capsys):
    es = _load_script("escalate_static")
    _base_store(tmp_path)
    chat = _StubChat(decide=_redraw)
    target, built = _wire_escalate(monkeypatch, es, chat)
    cfg_path = _esc_config(tmp_path)
    assert es.main(_esc_argv(tmp_path, cfg_path)) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert built == {"proxy_url": "http://127.0.0.1:7/v1", "routing_table": str(tmp_path / "rt.json"),
                     "cell_id": "none", "controls": True, "max_model_len": 10240}
    shard = tmp_path / ESC_RUN / "pasttense" / "none" / "shard00"
    assert sorted(p.name for p in shard.iterdir()) == ["escalation.json", "progress.jsonl",
                                                       "responses.jsonl", "served_target.json"]
    assert not (tmp_path / ESC_RUN / "pasttense" / "none" / es.STAGING_NAME).exists()
    rows = _rows(shard / "responses.jsonl")
    base = _rows(tmp_path / BASE_RUN / "pasttense" / "none" / "shard00" / "responses.jsonl") + \
        _rows(tmp_path / BASE_RUN / "pasttense" / "none" / "shard01" / "responses.jsonl")
    assert [(r["goal_id"], r["sample_idx"]) for r in rows] == \
        [(r["goal_id"], r["sample_idx"]) for r in base]                # base order, a JOIN
    by = {(r["goal_id"], r["sample_idx"]): r for r in rows}
    # copied rows: byte-identical to the base + the three provenance keys
    for key in [("g0", 0), ("g1", 0), ("g2", 0), ("g3", 1)]:
        r = by[key]
        b = next(x for x in base if (x["goal_id"], x["sample_idx"]) == key)
        assert r["regenerated"] is False and r["source_cap"] == 4096 and r["escalated_from_cap"] == 4096
        assert {k: v for k, v in r.items()
                if k not in ("regenerated", "source_cap", "escalated_from_cap")} == b
        assert "prefix_extension" not in r and "boundary_landing" not in r
    # re-drawn rows: same seed, new cap, the divergence + boundary fields
    block_to = {"do_sample": True, **QWEN, "cap": 8192, "arm": ARM, "seed_base": SEED_BASE,
                "trial": 0}
    for key in [("g0", 1), ("g1", 1), ("g3", 0)]:
        r = by[key]
        assert r["regenerated"] is True and r["source_cap"] == 8192 and r["escalated_from_cap"] == 4096
        assert r["decode"] == block_to and r["max_tokens"] == 8192
        assert r["sampling_seed"] == sc.static_sampling_seed(SEED_BASE, 0, "pasttense", *key)
        assert r["source_finish_reason"] == "length" and r["source_completion_tokens"] == 4096
    ident, ext, fork = by[("g0", 1)], by[("g1", 1)], by[("g3", 0)]
    assert ident["response"] == "resp g0 1" and ident["identical"] and ident["prefix_extension"]
    assert ident["first_divergence_char"] is None and ident["first_divergence_token"] is None
    assert ident["boundary_landing"] is True and ident["at_cap"] is False   # 900 < 4096
    assert ext["prefix_extension"] and not ext["identical"] and ext["boundary_landing"] is False
    assert ext["first_divergence_char"] is None and ext["at_cap"] is False
    assert fork["prefix_extension"] is False and fork["identical"] is False
    assert fork["first_divergence_char"] == len("resp g3 ") and fork["first_divergence_token"] == 2
    assert fork["at_cap"] is True and fork["finish_reason"] == "length"
    assert fork["boundary_landing"] is False
    # the requests: exactly the three at-cap items, at 8192, with the base seeds + preset
    sent = {p["messages"][-1]["content"]: p for p in chat.payloads}
    assert set(sent) == {"g0/1", "g1/1", "g3/0"}
    assert all(p["max_tokens"] == 8192 and p["temperature"] == 0.7 and p["top_k"] == 20
               for p in sent.values())
    assert sent["g3/0"]["seed"] == by[("g3", 0)]["sampling_seed"]
    # escalation.json
    esc = json.loads((shard / "escalation.json").read_text())
    for k, v in {"attack": "pasttense", "defense": "none", "cap_from": 4096, "cap_to": 8192,
                 "n_rows": 7, "n_at_cap_before": 3, "n_regenerated": 3, "n_at_cap_after": 1,
                 "n_copied": 4, "seed_base": SEED_BASE, "trial": 0, "n_prefix_extension": 2,
                 "n_fork": 1, "n_identical": 1, "n_boundary_landing": 1}.items():
        assert esc[k] == v, k
    assert summary["n_regenerated"] == 3 and summary["n_at_cap_after"] == 1
    # progress: one terminal record per goal, run_static's shape + the rung's counts
    prog = _rows(shard / "progress.jsonl")
    assert [p["goal_id"] for p in prog] == ["g0", "g1", "g2", "g3"]
    assert all(p["outcome"] == "done" and p["capped"] is False for p in prog)
    assert [(p["n_variants"], p["n_regenerated"], p["n_copied"], p["n_target_forwards"])
            for p in prog] == [(2, 1, 1, 1), (2, 1, 1, 1), (1, 0, 1, 0), (2, 1, 1, 1)]
    # served_target.json carries the rung's decode block and the sampled request count
    doc = json.loads((shard / "served_target.json").read_text())
    assert doc["decode"] == block_to and doc["n_sampled_requests"] == 3
    assert doc["manifest_stage"] == "final"
    # the base run was never touched
    assert _rows(tmp_path / BASE_RUN / "pasttense" / "none" / "shard00" / "responses.jsonl") == base[:4]


def test_escalate_static_is_idempotent(monkeypatch, tmp_path, capsys):
    es = _load_script("escalate_static")
    _base_store(tmp_path)
    chat = _StubChat(decide=_redraw)
    _wire_escalate(monkeypatch, es, chat)
    cfg_path = _esc_config(tmp_path)
    assert es.main(_esc_argv(tmp_path, cfg_path)) == 0
    first = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    n_requests = len(chat.payloads)
    shard = tmp_path / ESC_RUN / "pasttense" / "none" / "shard00"
    before = (shard / "responses.jsonl").read_bytes()
    assert es.main(_esc_argv(tmp_path, cfg_path)) == 0
    again = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert len(chat.payloads) == n_requests                         # no server contact
    assert (shard / "responses.jsonl").read_bytes() == before
    assert again == first
    # a PARTIAL rung root is refused, never overwritten
    (shard / "escalation.json").unlink()
    with pytest.raises(SystemExit) as ei:
        es.main(_esc_argv(tmp_path, cfg_path))
    assert ei.value.code == 2 and "not a complete rung root" in capsys.readouterr().err


def test_escalate_static_refuses_incomplete_or_greedy_base(monkeypatch, tmp_path, capsys):
    es = _load_script("escalate_static")
    chat = _StubChat(decide=_redraw)
    _wire_escalate(monkeypatch, es, chat)
    cfg_path = _esc_config(tmp_path)

    def expect_refusal(match):
        with pytest.raises(SystemExit) as ei:
            es.main(_esc_argv(tmp_path, cfg_path))
        err = capsys.readouterr().err
        assert ei.value.code == 2 and match in err, err
        assert not (tmp_path / ESC_RUN).exists() and chat.payloads == []

    expect_refusal("no shard")                                        # no base at all
    _base_store(tmp_path)
    prog = tmp_path / BASE_RUN / "pasttense" / "none" / "shard01" / "progress.jsonl"
    lines = prog.read_text().splitlines()
    prog.write_text("\n".join(lines[:-1]) + "\n")                     # drop g3's record
    expect_refusal("INCOMPLETE")
    prog.write_text("\n".join(lines) + "\n")
    resp = tmp_path / BASE_RUN / "pasttense" / "none" / "shard00" / "responses.jsonl"
    rows = _rows(resp)
    greedy = [{k: v for k, v in r.items() if k not in ("finish_reason", "at_cap")} for r in rows]
    resp.write_text("".join(json.dumps(r) + "\n" for r in greedy))
    expect_refusal("finish_reason")                                   # a greedy base
    other = [dict(r, decode={**r["decode"], "temperature": 0.9}) for r in rows]
    resp.write_text("".join(json.dumps(r) + "\n" for r in other))
    expect_refusal("not this rung's base block")                      # another preset
    bad_seed = [dict(r, sampling_seed=r["sampling_seed"] + 1) for r in rows]
    resp.write_text("".join(json.dumps(r) + "\n" for r in bad_seed))
    expect_refusal("seeds do not reproduce")
    resp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    # config refusals
    with pytest.raises(SystemExit) as ei:
        es.main(_esc_argv(tmp_path, _esc_config(tmp_path, attacks={"static_max_new_tokens": 4096},
                                                sampled_decode=_sampled_block(
                                                    static_cap=4096, escalated_from={
                                                        "kind": "static", "cap": 4096,
                                                        "private_run": BASE_RUN}))))
    assert ei.value.code == 2 and "must exceed" in capsys.readouterr().err
    with pytest.raises(SystemExit) as ei:
        es.main(_esc_argv(tmp_path, _esc_config(tmp_path, sampled_decode=_sampled_block(static_cap=8192))))
    assert ei.value.code == 2 and "escalated_from is missing" in capsys.readouterr().err
    with pytest.raises(SystemExit) as ei:
        es.main(_esc_argv(tmp_path, _esc_config(tmp_path, private_run=BASE_RUN)))
    assert ei.value.code == 2 and "NEVER written in place" in capsys.readouterr().err
    with pytest.raises(SystemExit) as ei:
        es.main(_esc_argv(tmp_path, cfg_path, **{"--attack": "pap"}))
    assert ei.value.code == 2 and "not one of the sampled" in capsys.readouterr().err


def test_escalate_static_failed_redraw_writes_nothing(monkeypatch, tmp_path):
    es = _load_script("escalate_static")
    _base_store(tmp_path)

    def flaky(payload):
        if payload["messages"][-1]["content"] == "g1/1":
            raise RuntimeError("upstream exploded")
        return _redraw(payload)
    chat = _StubChat(decide=flaky)
    _wire_escalate(monkeypatch, es, chat)
    with pytest.raises(RuntimeError, match="upstream exploded"):
        es.main(_esc_argv(tmp_path, _esc_config(tmp_path)))
    assert not (tmp_path / ESC_RUN / "pasttense" / "none" / "shard00").exists()
    assert not (tmp_path / ESC_RUN / "pasttense" / "none" / es.STAGING_NAME).exists()
    # a re-draw that comes back without finish_reason is refused the same way
    chat2 = _StubChat(decide=lambda p: ("t", None, 5))
    _wire_escalate(monkeypatch, es, chat2)
    with pytest.raises(RuntimeError, match="finish_reason"):
        es.main(_esc_argv(tmp_path, _esc_config(tmp_path)))
    assert not (tmp_path / ESC_RUN / "pasttense" / "none" / "shard00").exists()


def test_divergence_fields_identical_prefix_fork():
    es = _load_script("escalate_static")
    tok = _whitespace_tokenize
    ident = es.divergence_fields("a b c", "a b c", tok)
    assert ident == {"prefix_extension": True, "identical": True,
                     "first_divergence_char": None, "first_divergence_token": None}
    ext = es.divergence_fields("a b c", "a b c d e", tok)
    assert ext == {"prefix_extension": True, "identical": False,
                   "first_divergence_char": None, "first_divergence_token": None}
    fork = es.divergence_fields("a b c d", "a b X d", tok)
    assert fork == {"prefix_extension": False, "identical": False,
                    "first_divergence_char": 4, "first_divergence_token": 2}
    shorter = es.divergence_fields("a b c", "a b", tok)          # new is a proper prefix
    assert shorter["prefix_extension"] is False and shorter["first_divergence_char"] == 3
    assert shorter["first_divergence_token"] == 2
    # byte semantics: a multi-byte prefix
    assert es.divergence_fields("héllo", "héllo wörld", tok)["prefix_extension"]
    with pytest.raises(SystemExit):                               # a fork needs the tokenizer
        es.divergence_fields("a b", "a c", None)


# --------------------------------------------------------------------------- #
# --items: the decision tool's disposition drives the static rung              #
# --------------------------------------------------------------------------- #

ITEMS_HEADER_KEY = "__ladder_items__"


def _static_disposition(path, rows, *, cap=4096, run=BASE_RUN, **header_over):
    """rows = {(goal_id, sample_idx): excluded_as | None}."""
    header = {"schema": 2, "kind": "static", "cap": cap, "private_run": run, "attack": "pasttense",
              "defense": "none", "trial": 0, "n_at_cap": len(rows),
              "n_escalate": sum(1 for v in rows.values() if v is None), "decision": "escalate",
              "rule": "item_filtered_v1", "classifier": {"version": "item_filtered_v1"},
              "tool_sha256": "t", "rep_ratio_source_sha256": "s"}
    header.update(header_over)
    with Path(path).open("w") as f:
        f.write(json.dumps({ITEMS_HEADER_KEY: header}) + "\n")
        for (gid, idx), ex in rows.items():
            f.write(json.dumps({"goal_id": gid, "sample_idx": idx, "trial": 0,
                                "class": "cap_coherent" if ex in (None, "boundary") else ex,
                                "rep_ratio": 0.0, "loop_by": None, "degenerate_reason": None,
                                "boundary": ex == "boundary", "sibling": None,
                                "escalate": ex is None, "excluded_as": ex}) + "\n")
    return Path(path)


def test_escalate_static_with_items_redraws_only_the_escalation_set(monkeypatch, tmp_path, capsys):
    es = _load_script("escalate_static")
    _base_store(tmp_path)
    chat = _StubChat(decide=_redraw)
    _wire_escalate(monkeypatch, es, chat)
    items = _static_disposition(tmp_path / "items.jsonl",
                                {("g0", 1): None, ("g1", 1): "boundary", ("g3", 0): "loop"})
    assert es.main(_esc_argv(tmp_path, _esc_config(tmp_path), **{"--items": str(items)})) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    shard = tmp_path / ESC_RUN / "pasttense" / "none" / "shard00"
    rows = _rows(shard / "responses.jsonl")
    by = {(r["goal_id"], r["sample_idx"]): r for r in rows}
    base = _rows(tmp_path / BASE_RUN / "pasttense" / "none" / "shard00" / "responses.jsonl") + \
        _rows(tmp_path / BASE_RUN / "pasttense" / "none" / "shard01" / "responses.jsonl")
    assert [(r["goal_id"], r["sample_idx"]) for r in rows] == [(r["goal_id"], r["sample_idx"]) for r in base]
    # only g0/1 was re-drawn
    assert [p["messages"][-1]["content"] for p in chat.payloads] == ["g0/1"]
    assert by[("g0", 1)]["regenerated"] is True and es.EXCLUDED_KEY not in by[("g0", 1)]
    # the excluded at-cap rows: copied forward STILL at cap, stamped, otherwise the base row
    for key, why in ((("g1", 1), "boundary"), (("g3", 0), "loop")):
        r = by[key]
        b = next(x for x in base if (x["goal_id"], x["sample_idx"]) == key)
        assert r[es.EXCLUDED_KEY] == why and r["regenerated"] is False and r["at_cap"] is True
        assert r["source_cap"] == 4096 and r["finish_reason"] == "length"
        assert {k: v for k, v in r.items()
                if k not in ("regenerated", "source_cap", "escalated_from_cap", es.EXCLUDED_KEY)} == b
        assert "prefix_extension" not in r
    # terminated rows carry no stamp
    assert all(es.EXCLUDED_KEY not in by[k] for k in [("g0", 0), ("g1", 0), ("g2", 0), ("g3", 1)])
    esc = json.loads((shard / "escalation.json").read_text())
    for k, v in {"n_rows": 7, "n_at_cap_before": 3, "n_regenerated": 1, "n_at_cap_after": 2, "n_copied": 6,
                 "n_at_cap_after_regenerated": 0, "n_at_cap_after_excluded": 2, "n_escalation_excluded": 2,
                 "n_escalation_excluded_by_class": {"loop": 1, "repetitive": 0, "boundary": 1},
                 "item_filter": True, "items_file": str(items),
                 "n_boundary_landing": 1, "n_identical": 1}.items():
        assert esc[k] == v, k
    assert esc["items_file_sha256"] == hashlib.sha256(items.read_bytes()).hexdigest()
    assert summary["n_regenerated"] == 1 and summary["n_at_cap_after"] == 2
    # the store's own count agrees with n_at_cap_after (the stamped rows ARE at cap)
    acs = _load_script("at_cap_static")
    assert acs.scan_static_store(shard.parent)["n_at_cap"] == 2
    # progress: the re-draw count per goal
    prog = _rows(shard / "progress.jsonl")
    assert [(p["n_variants"], p["n_regenerated"], p["n_copied"]) for p in prog] == \
        [(2, 1, 1), (2, 0, 2), (1, 0, 1), (2, 0, 2)]


def test_escalate_static_without_items_still_redraws_every_at_cap_row(monkeypatch, tmp_path, capsys):
    """The running jobs' behaviour, pinned: no --items -> every at-cap row re-draws,
    escalation.json says item_filter false."""
    es = _load_script("escalate_static")
    _base_store(tmp_path)
    chat = _StubChat(decide=_redraw)
    _wire_escalate(monkeypatch, es, chat)
    assert es.main(_esc_argv(tmp_path, _esc_config(tmp_path))) == 0
    capsys.readouterr()
    esc = json.loads((tmp_path / ESC_RUN / "pasttense" / "none" / "shard00" / "escalation.json").read_text())
    assert esc["n_regenerated"] == 3 and esc["item_filter"] is False and esc["items_file"] is None
    assert esc["n_escalation_excluded"] == 0 and esc["n_at_cap_after_excluded"] == 0
    assert esc["n_at_cap_after"] == esc["n_at_cap_after_regenerated"] == 1
    assert sorted(p["messages"][-1]["content"] for p in chat.payloads) == ["g0/1", "g1/1", "g3/0"]
    rows = _rows(tmp_path / ESC_RUN / "pasttense" / "none" / "shard00" / "responses.jsonl")
    assert all(es.EXCLUDED_KEY not in r for r in rows)


def test_escalate_static_items_refusals(monkeypatch, tmp_path, capsys):
    es = _load_script("escalate_static")
    _base_store(tmp_path)
    chat = _StubChat(decide=_redraw)
    _wire_escalate(monkeypatch, es, chat)
    cfg_path = _esc_config(tmp_path)

    def run(items):
        with pytest.raises(SystemExit) as ei:
            es.main(_esc_argv(tmp_path, cfg_path, **{"--items": str(items)}))
        assert ei.value.code == 2
        return capsys.readouterr().err

    full = {("g0", 1): None, ("g1", 1): "boundary", ("g3", 0): "loop"}
    # the header describes another store (cap / run / trial)
    assert "describes" in run(_static_disposition(tmp_path / "a.jsonl", full, cap=8192))
    assert "describes" in run(_static_disposition(tmp_path / "b.jsonl", full, run="other_run"))
    assert "describes" in run(_static_disposition(tmp_path / "c.jsonl", full, trial=1))
    # an at-cap row without a disposition / a row naming a terminated item
    assert "without a disposition" in run(_static_disposition(tmp_path / "d.jsonl",
                                                              {("g0", 1): None, ("g1", 1): "boundary"}))
    assert "not at cap in the base" in run(_static_disposition(tmp_path / "e.jsonl", {**full, ("g2", 0): None}))
    # a controls file / a torn file / an absent file
    assert "describes" in run(_static_disposition(tmp_path / "f.jsonl", full, kind="controls"))
    torn = tmp_path / "g.jsonl"
    torn.write_text(_static_disposition(tmp_path / "h.jsonl", full).read_text() + '{"goal_id": ')
    assert "torn" in run(torn)
    assert "absent" in run(tmp_path / "nope.jsonl")
    # a decision made under another rule can never drive a rung
    assert "not 'item_filtered_v1'" in run(_static_disposition(tmp_path / "i.jsonl", full, rule="at_cap_v0"))
    assert chat.payloads == []
    assert not (tmp_path / ESC_RUN / "pasttense" / "none" / "shard00").exists()


# ---------------------------------------------------------------------------- #
# the 8192 -> 16384 climb: a RUNG-SHAPED source                                 #
# ---------------------------------------------------------------------------- #

ESC16_RUN = "run_sampled_t0_cap16384"


def _esc16_config(tmp_path, **over):
    cfg = _cfg(private_run=ESC16_RUN, attacks={"static_max_new_tokens": 16384,
                                                "target_max_new_tokens": 512},
               sampled_decode=_sampled_block(static_cap=16384, escalated_from={
                   "kind": "static", "cap": 8192, "private_run": ESC_RUN}))
    cfg.update(over)
    p = tmp_path / "esc16.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _redraw16(payload):
    prompt = payload["messages"][-1]["content"]
    if prompt == "g3/0" and payload["max_tokens"] == 16384:
        return "resp g3 X forked and now it finishes", "stop", 12000
    raise AssertionError(f"unexpected re-draw of {prompt!r} at {payload['max_tokens']} (only g3/0 climbs)")


def _rung8192_root(monkeypatch, tmp_path, es, capsys):
    """The real 8192 rung over the base store, with g1/1 excluded as a loop: the resulting root holds copied
    rows at the 4096 block (g1/1 still AT CAP), re-drawn rows at the 8192 block (g3/0 still at cap)."""
    _base_store(tmp_path)
    chat = _StubChat(decide=_redraw)
    _wire_escalate(monkeypatch, es, chat)
    items = _static_disposition(tmp_path / "items8192.jsonl", {("g0", 1): None, ("g1", 1): "loop", ("g3", 0): None})
    assert es.main(_esc_argv(tmp_path, _esc_config(tmp_path), **{"--items": str(items)})) == 0
    capsys.readouterr()
    rows = _rows(tmp_path / ESC_RUN / "pasttense" / "none" / "shard00" / "responses.jsonl")
    by = {(r["goal_id"], r["sample_idx"]): r for r in rows}
    assert by[("g1", 1)]["at_cap"] and by[("g1", 1)]["regenerated"] is False and by[("g1", 1)]["decode"]["cap"] == 4096
    assert by[("g3", 0)]["at_cap"] and by[("g3", 0)]["regenerated"] is True and by[("g3", 0)]["decode"]["cap"] == 8192
    return rows


def test_escalate_static_climbs_from_a_rung_shaped_source_keyed_on_each_rows_regenerated_flag(monkeypatch, tmp_path, capsys):
    es = _load_script("escalate_static")
    src_rows = _rung8192_root(monkeypatch, tmp_path, es, capsys)
    chat = _StubChat(decide=_redraw16)
    _wire_escalate(monkeypatch, es, chat)
    # the 8192 root's at-cap rows are g1/1 (a copied loop) and g3/0 (re-drawn, still at cap): the decision at 8192
    # excludes the loop again and escalates g3/0
    items = _static_disposition(tmp_path / "items16384.jsonl", {("g1", 1): "loop", ("g3", 0): None}, cap=8192, run=ESC_RUN)
    assert es.main(_esc_argv(tmp_path, _esc16_config(tmp_path), **{"--items": str(items), "--max-model-len": "18432"})) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    shard = tmp_path / ESC16_RUN / "pasttense" / "none" / "shard00"
    rows = _rows(shard / "responses.jsonl")
    assert [(r["goal_id"], r["sample_idx"]) for r in rows] == [(r["goal_id"], r["sample_idx"]) for r in src_rows]
    by = {(r["goal_id"], r["sample_idx"]): r for r in rows}
    # rows copied THROUGH two rungs keep the 4096 block and source_cap 4096; escalated_from_cap is this rung's source
    for key in [("g0", 0), ("g1", 0), ("g2", 0), ("g3", 1)]:
        r = by[key]
        assert r["regenerated"] is False and r["source_cap"] == 4096 and r["escalated_from_cap"] == 8192
        assert r["decode"]["cap"] == 4096 and r["at_cap"] is False
    # a row RE-DRAWN at 8192 and terminated there is copied with the 8192 block and source_cap 8192 (its 8192-rung
    # divergence fields ride along: they describe the text this row carries)
    g01 = by[("g0", 1)]
    assert g01["regenerated"] is False and g01["source_cap"] == 8192 and g01["decode"]["cap"] == 8192
    assert g01["boundary_landing"] is True and g01["identical"] is True and g01["at_cap"] is False
    # the loop copied forward at cap AGAIN: stamped, still the 4096 block it was produced under
    g11 = by[("g1", 1)]
    assert g11[es.EXCLUDED_KEY] == "loop" and g11["at_cap"] is True and g11["regenerated"] is False
    assert g11["source_cap"] == 4096 and g11["decode"]["cap"] == 4096
    # the one climb: g3/0 re-drawn at 16384 with its seed, source facts from the 8192 draw
    g30 = by[("g3", 0)]
    block16 = {"do_sample": True, **QWEN, "cap": 16384, "arm": ARM, "seed_base": SEED_BASE, "trial": 0}
    assert g30["regenerated"] is True and g30["source_cap"] == 16384 and g30["escalated_from_cap"] == 8192
    assert g30["decode"] == block16 and g30["max_tokens"] == 16384 and g30["at_cap"] is False
    assert g30["source_finish_reason"] == "length" and g30["source_completion_tokens"] == 8192
    assert g30["boundary_landing"] is False   # 12000 >= 8192
    assert g30["prefix_extension"] is True and g30["response"].startswith("resp g3 X forked")
    assert [p["messages"][-1]["content"] for p in chat.payloads] == ["g3/0"] and chat.payloads[0]["max_tokens"] == 16384
    assert chat.payloads[0]["seed"] == g30["sampling_seed"] == sc.static_sampling_seed(SEED_BASE, 0, "pasttense", "g3", 0)
    # three caps of text in one root, each row saying which produced it
    assert sorted({r["decode"]["cap"] for r in rows}) == [4096, 8192, 16384]
    esc = json.loads((shard / "escalation.json").read_text())
    for k, v in {"cap_from": 8192, "cap_to": 16384, "n_rows": 7, "n_at_cap_before": 2, "n_regenerated": 1,
                 "n_at_cap_after": 1, "n_copied": 6, "n_escalation_excluded": 1, "n_at_cap_after_excluded": 1,
                 "n_at_cap_after_regenerated": 0, "base_private_run": ESC_RUN, "private_run": ESC16_RUN,
                 "max_model_len": 18432, "item_filter": True}.items():
        assert esc[k] == v, (k, esc[k])
    assert esc["n_escalation_excluded_by_class"] == {"loop": 1, "repetitive": 0, "boundary": 0}
    assert summary["cap_to"] == 16384 and json.loads((shard / "served_target.json").read_text())["decode"] == block16
    # the 8192 source was never touched
    assert _rows(tmp_path / ESC_RUN / "pasttense" / "none" / "shard00" / "responses.jsonl") == src_rows


def test_escalate_static_refuses_a_source_row_that_contradicts_its_regenerated_flag(monkeypatch, tmp_path, capsys):
    es = _load_script("escalate_static")
    _rung8192_root(monkeypatch, tmp_path, es, capsys)
    src = tmp_path / ESC_RUN / "pasttense" / "none" / "shard00" / "responses.jsonl"
    pristine = src.read_bytes()
    cfg_path = _esc16_config(tmp_path)
    _wire_escalate(monkeypatch, es, _StubChat(decide=_redraw16))

    def refused(mutate, needle):
        rows = _rows(src)
        for r in rows:
            mutate(r)
        src.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with pytest.raises(SystemExit) as ei:
            es.main(_esc_argv(tmp_path, cfg_path, **{"--max-model-len": "18432"}))
        assert ei.value.code == 2
        err = capsys.readouterr().err
        assert needle in err, err
        src.write_bytes(pristine)

    # a copied row (regenerated false) whose block is the 8192 one: contradicts its flag
    def m1(r):
        if (r["goal_id"], r["sample_idx"]) == ("g1", 0):
            r["decode"] = dict(r["decode"], cap=8192)
    refused(m1, "contradicts its regenerated flag")
    # a re-drawn row (regenerated true) claiming source_cap 4096
    def m2(r):
        if (r["goal_id"], r["sample_idx"]) == ("g3", 0):
            r["source_cap"] = 4096
    refused(m2, "re-drawn at another cap")
    # a copied row with a source_cap that is not a ladder rung below the source cap
    def m3(r):
        if (r["goal_id"], r["sample_idx"]) == ("g2", 0):
            r["source_cap"] = 8192
    refused(m3, "not a static-ladder rung below the source cap")
    # a config that skips a rung (4096 -> 16384) refuses before reading any row
    skip = _esc16_config(tmp_path, sampled_decode=_sampled_block(static_cap=16384, escalated_from={
        "kind": "static", "cap": 4096, "private_run": BASE_RUN}))
    with pytest.raises(SystemExit) as ei:
        es.main(_esc_argv(tmp_path, skip))
    assert ei.value.code == 2 and "skips a rung" in capsys.readouterr().err
    # nothing was written for the 16384 rung
    assert not (tmp_path / ESC16_RUN).exists()
