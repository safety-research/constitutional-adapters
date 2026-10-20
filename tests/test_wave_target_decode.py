"""The vendor preset reaches the multi-turn TARGET replies and best-of-N (the sdfbase-llama8b wave,
2026-09-10), behind explicit switches, with the sampler RECORDED where the draws land.

Scale (2026-09-10): the static families and the judged controls were the only legs the sampled
machinery reached; BoN was always sampled at the harness's T=1.0 (no top_p/top_k, nothing stored),
multi-turn target replies were greedy (target_temperature 0.0), Petri exposes no sampler. So:

  * ``ServedChatTarget.chat(decode=...)`` puts EXACTLY the preset's keys + the per-call seed on the
    body (SAMPLER_KEYS only) and counts the request in ``n_sampled_requests``; absent, the body is
    byte-identical to before.
  * The PyRIT target adapter and ActorAttack thread ``target_decode`` with a per-(goal, turn) seed
    (``multiturn_seed``) and stamp ``target_decode`` / ``served_window`` / ``rerun_of`` on every
    progress record; an errored goal's record carries the typed error's own record.
  * BoN rows and its manifest carry ``decode`` (the sampler actually sent), ``bon_decode_preset`` and
    ``decode_policy_id`` under every switch value — the sampler is never implicit. Since the rule of
    2026-09-17 the DEFAULT is the served model's VENDOR PRESET (the registry's ``<model>|bon|off|vendor_preset``
    row, resolved per model — ``model=`` names it); the grandfathered T=1 row is a by-name opt-in
    (``harness_default``) for stores drawn under it.
  * ``jailbreak_run --multiturn-target-decode`` defaults to ``auto``, which follows the config-level
    mode ``attacks.target_decode`` -- ``vendor_preset`` by default since 2026-09-16 (a served target
    decodes at the vendor preset, --model-key naming the row or the config's pins resolving it;
    tests/test_multiturn_vendor_default.py), ``generation_config`` / ``greedy`` when stated; before
    2026-09-16 ``auto`` was vendor only with a served target AND --model-key, else greedy. An explicit
    ``vendor`` without served / --model-key refuses; the preset comes from
    configs/sdfbase_sampled_decode.yaml HELD to the config's model pins.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from sisteer.capeval.serving.proxy import RouteEntry
from sisteer.eval.attacks import bon_served as bs
from sisteer.eval.attacks import pyrit_adapters as pa
from sisteer.eval.attacks import served_target as st

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))   # the served-BoN tests' fixture shapes (tests/ is not a package)

PRESET = {"temperature": 0.6, "top_p": 0.9}
LLAMA = ("NousResearch/Meta-Llama-3.1-8B-Instruct", "d10aef7999a2b5ba950ab3974312feeedbfe0b77")


class _StubChat:
    def __init__(self):
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        return {"choices": [{"message": {"content": "a reply"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2}}


def test_chat_carries_exactly_the_preset_keys_and_the_seed_and_counts_it(tmp_path):
    chat = _StubChat()
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x", timing_log=tmp_path / "t.jsonl")
    t.chat([{"role": "user", "content": "hi"}], max_new_tokens=512, do_sample=True,
           temperature=0.6, sampling_seed=1234, decode=PRESET)
    p = chat.payloads[0]
    assert p["temperature"] == 0.6 and p["top_p"] == 0.9 and p["seed"] == 1234 and p["max_tokens"] == 512
    assert "top_k" not in p and "presence_penalty" not in p            # exactly the preset's keys
    assert t.n_sampled_requests == 1 and t.n_target_forwards == 1
    # greedy: byte-identical to before (no seed, temperature 0, no sampler keys)
    t.chat([{"role": "user", "content": "hi"}], max_new_tokens=512)
    q = chat.payloads[1]
    assert q["temperature"] == 0.0 and "seed" not in q and "top_p" not in q and t.n_sampled_requests == 1
    # a preset on a greedy call, or without a seed, refuses
    with pytest.raises(st.ServedTargetError, match="do_sample=True"):
        t.chat([{"role": "user", "content": "hi"}], decode=PRESET, sampling_seed=1)
    with pytest.raises(st.ServedTargetError, match="sampling_seed"):
        t.chat([{"role": "user", "content": "hi"}], decode=PRESET, do_sample=True, temperature=0.6)


def test_multiturn_seed_is_the_static_rule_extended_by_the_turn():
    import hashlib
    s = pa.multiturn_seed(20260909, 1, "crescendo", "g7", 3)
    want = int.from_bytes(hashlib.sha256(b"20260909|1|crescendo|g7|3").digest()[:4], "big")
    assert s == want
    assert pa.multiturn_seed(20260909, 1, "crescendo", "g7", 4) != s          # the turn moves it
    assert pa.multiturn_seed(20260909, 1, "pair", "g7", 3) != s               # the attack moves it
    with pytest.raises(ValueError, match="seed_base"):
        pa.multiturn_seed(None, 0, "pair", "g", 0)


class _Defense:
    """A served-shaped defense: records every chat() call's kwargs."""

    def __init__(self):
        self.calls = []
        self.n_target_forwards = 0

    def chat(self, messages, *, max_new_tokens=512, do_sample=False, temperature=1.0,
             sampling_seed=None, decode=None):
        self.n_target_forwards += 1
        self.calls.append({"do_sample": do_sample, "temperature": temperature, "seed": sampling_seed,
                           "decode": decode, "n": len(messages)})
        return "ok"


def test_pyrit_target_threads_the_preset_seeded_per_turn_or_unseeded_when_declared(tmp_path):
    """At de78ab46 the constructor demanded a seed_fn under every preset and killed all four crescendo / PAIR
    target legs (`ValueError: a target_decode preset needs a seed_fn`) although `--multiturn-target-seed none`
    (multi-turn replies carry NO seed; the server's generation config governs) passes seed_fn None on
    purpose. Now: seed_fn None under a preset is the UNSEEDED draw — one turn through a served-shaped defense carries the
    preset and NO sampling_seed; a callable seed_fn seeds per turn as before; a non-callable, non-None seed_fn is the one
    caller bug still refused; the greedy adapter is untouched."""
    import asyncio
    from pyrit.models import Message
    asyncio.run(pa.init_pyrit_memory_async(tmp_path / "memory.db"))   # a PromptTarget needs central memory

    async def one_turn(tgt):
        out = await tgt._send_prompt_to_target_async(
            normalized_conversation=[Message.from_prompt(prompt="hi", role="user")])
        return out[0].get_value()

    # seeded (the pre-ruling per_turn policy): the request carries the preset + the turn's seed
    d = _Defense()
    tgt = pa.SteeredPromptTarget(defense=d, target_decode=PRESET, seed_fn=lambda turn: 100 + turn)
    assert tgt._target_decode == PRESET and asyncio.run(one_turn(tgt)) == "ok"
    assert d.calls[-1]["decode"] == PRESET and d.calls[-1]["seed"] == 100 and d.calls[-1]["do_sample"] is True
    assert d.calls[-1]["temperature"] == PRESET["temperature"]
    # UNSEEDED (--multiturn-target-seed none): admitted, and the request carries the preset with NO sampling_seed
    d2 = _Defense()
    tgt2 = pa.SteeredPromptTarget(defense=d2, target_decode=PRESET)          # seed_fn None
    assert tgt2._target_decode == PRESET and tgt2._seed_fn is None
    assert asyncio.run(one_turn(tgt2)) == "ok"
    assert d2.calls[-1]["decode"] == PRESET and d2.calls[-1]["seed"] is None and d2.calls[-1]["do_sample"] is True
    assert d2.calls[-1]["temperature"] == PRESET["temperature"]
    # the one caller bug still refused: a seed_fn that is neither None nor callable
    with pytest.raises(ValueError, match="callable"):
        pa.SteeredPromptTarget(defense=_Defense(), target_decode=PRESET, seed_fn=7)
    # greedy adapter: untouched defaults, no decode, no seed on the request
    d3 = _Defense()
    g = pa.SteeredPromptTarget(defense=d3)
    assert g._target_decode is None and g._do_sample is False and g._temperature == 0.0
    assert asyncio.run(one_turn(g)) == "ok" and d3.calls[-1]["decode"] is None and d3.calls[-1]["seed"] is None


def test_actorattack_records_target_decode_served_window_and_rerun_of(tmp_path):
    from sisteer.eval.attacks.multiturn import run_actorattack
    from sisteer.eval.jailbreak_data import Goal

    d = _Defense()
    goals = [Goal("g0", "benign objective", "illegal", "harmbench", "rev")]
    plan = '{"actors": [{"name": "a1", "queries": ["q0", "q1"]}]}'
    out = run_actorattack(
        d, goals, attacker_fn=lambda chat: plan, scorer_fn=lambda p: "1",
        out_dir=tmp_path / "aa", resume=False, actor_num=1, discovery_retries=0, max_new_tokens=64,
        target_decode=PRESET, seed_base=20260909, trial=1, served_window=8704,
        rerun_of="/store/of/the/original")
    rec = out["per_goal"][0]
    assert rec["target_decode"] == PRESET and rec["served_window"] == 8704
    assert rec["rerun_of"] == "/store/of/the/original"
    assert d.calls and all(c["do_sample"] and c["temperature"] == 0.6 and c["decode"] == PRESET for c in d.calls)
    turns = [c["seed"] for c in d.calls]
    assert turns == [pa.multiturn_seed(20260909, 1, "actorattack", "g0", i) for i in range(len(turns))]
    row = json.loads((tmp_path / "aa" / "progress.jsonl").read_text().splitlines()[0])
    assert row["target_decode"] == PRESET and row["served_window"] == 8704
    # greedy: none of the three keys appears (byte-identical records)
    d2 = _Defense()
    out2 = run_actorattack(d2, goals, attacker_fn=lambda chat: plan, scorer_fn=lambda p: "1",
                           out_dir=tmp_path / "aa2", resume=False, actor_num=1, discovery_retries=0,
                           max_new_tokens=64)
    rec2 = out2["per_goal"][0]
    assert not ({"target_decode", "served_window", "rerun_of"} & set(rec2))
    assert all(not c["do_sample"] and c["decode"] is None for c in d2.calls)


def _bon_world(tmp_path):
    import test_bon_served as tb  # the fixture shapes of the served-BoN tests
    backend = tb.RecordingBackend()
    proxy, routes = tb._proxy(tmp_path, backend)
    defense = bs.ServedDefense.over_proxy(proxy, virtual_model="petri__base", cell_id="none")
    return tb, backend, routes, defense


def test_bon_rows_and_manifest_record_the_sampler_under_both_switch_values(tmp_path):
    tb, backend, routes, defense = _bon_world(tmp_path)
    goals = tb._goals(2)
    jsonl = bs.bon_generate_served(defense, goals, out_dir=tmp_path / "vendor", routes=routes, n_max=3,
                                   max_concurrency=4, decode=PRESET, decode_label="vendor")
    rows = tb._read(jsonl)
    assert len(rows) == 6 and all(r["decode"] == PRESET and r["bon_decode_preset"] == "vendor" for r in rows)
    man = json.loads((tmp_path / "vendor" / bs.BON_SERVED_MANIFEST).read_text())
    assert man["decode"] == PRESET and man["bon_decode_preset"] == "vendor"
    # every request carried exactly the preset's keys + the unit's seed
    sent = [r for r in backend.payloads if r.get("max_tokens")]
    assert sent
    for r in sent:
        assert r["temperature"] == 0.6 and r["top_p"] == 0.9 and "seed" in r and "top_k" not in r
    # THE DEFAULT (2026-09-17): the served model's vendor preset from the registry, RECORDED with its policy id
    (tmp_path / "h").mkdir()
    _, backend2, routes2, defense2 = _bon_world(tmp_path / "h")
    jsonl2 = bs.bon_generate_served(defense2, goals, out_dir=tmp_path / "vendor_preset", routes=routes2, n_max=2,
                                    max_concurrency=4, model="gemma4e4b")
    rows2 = tb._read(jsonl2)
    gemma = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
    assert all(r["decode"] == gemma and r["bon_decode_preset"] == bs.BON_VENDOR_PRESET
               and r["decode_policy_id"] == "gemma4e4b_it|bon|off|vendor_preset@v1" for r in rows2)
    for r in [r for r in backend2.payloads if r.get("max_tokens")]:
        assert {k: r[k] for k in gemma} == gemma and "seed" in r and "presence_penalty" not in r
    # the grandfathered row BY NAME: temperature 1.0, nothing else — the sampler every earlier store was drawn under
    (tmp_path / "g").mkdir()
    _, _backend3, routes3, defense3 = _bon_world(tmp_path / "g")
    jsonl3 = bs.bon_generate_served(defense3, goals, out_dir=tmp_path / "harness", routes=routes3, n_max=2,
                                    max_concurrency=4, decode_label=bs.BON_HARNESS_DEFAULT)
    rows3 = tb._read(jsonl3)
    assert all(r["decode"] == {"temperature": 1.0} and r["bon_decode_preset"] == "harness_default" for r in rows3)
    # no model and no block: refused — best-of-N has no model-independent default any more
    with pytest.raises(bs.BonServedError, match="no model was named"):
        bs.bon_generate_served(defense3, goals, out_dir=tmp_path / "nomodel", routes=routes3, n_max=1, max_concurrency=1)
    with pytest.raises(bs.BonServedError, match="temperature"):
        bs.bon_generate_served(defense, goals, out_dir=tmp_path / "bad", routes=routes, n_max=1,
                               max_concurrency=1, decode={"top_p": 0.9})


def test_vendor_preset_for_reads_the_file_and_holds_it_to_the_config_pins():
    import jailbreak_run as jr
    got = jr.vendor_preset_for("llama8b", {"model": LLAMA[0], "model_revision": LLAMA[1]})
    assert got["decode"] == PRESET and got["seed_base"] == 20260909 and got["source_kind"] == "generation_config"
    with pytest.raises(SystemExit, match="was read from"):
        jr.vendor_preset_for("llama8b", {"model": LLAMA[0], "model_revision": "0000"})
    with pytest.raises(SystemExit, match="no vendor preset"):
        jr.vendor_preset_for("no_such_key", {"model": LLAMA[0], "model_revision": LLAMA[1]})
    assert jr.MULTITURN_TARGET_DECODE_MODES == ("auto", "vendor", "greedy")


def test_generate_parser_has_the_switches_with_auto_default():
    import argparse

    import jailbreak_run as jr
    ap = argparse.ArgumentParser()
    jr._add_target_engine_args(ap)
    ns = ap.parse_args([])
    assert ns.multiturn_target_decode == "auto" and ns.model_key is None and ns.rerun_of is None
    ns = ap.parse_args(["--multiturn-target-decode", "greedy", "--model-key", "llama8b", "--rerun-of", "/x"])
    assert (ns.multiturn_target_decode, ns.model_key, ns.rerun_of) == ("greedy", "llama8b", "/x")
    with pytest.raises(SystemExit):
        ap.parse_args(["--multiturn-target-decode", "sometimes"])


def test_route_entry_import_is_the_proxy_grammar():
    # a guard that the served route grammar this test relies on did not move under us
    r = RouteEntry(virtual_model="petri__base", cell_id="none", cell_class="base",
                   serving_identity="x", intervention_hash="aa" * 8)
    assert r.cell_id == "none"
