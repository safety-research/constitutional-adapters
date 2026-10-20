"""The SAMPLED multi-turn target's decode block (2026-09-10, the GLM errored-goal re-run).

The multi-turn adapters default to a greedy target (target_temperature 0.0). A sampled multi-turn
arm needs (a) the vendor preset's extra keys (top_p / top_k / repetition_penalty) ON the request
beside temperature, and (b) every response / timing row to RECORD the decode it ran under -- the
BoN lesson: records with no temperature were later misread as greedy. CPU only, no server.
"""

from __future__ import annotations

import json

import pytest

from sisteer.eval.attacks import multiturn
from sisteer.eval.attacks import served_target as st

VENDOR = {"top_p": 0.6, "top_k": 2, "repetition_penalty": 1.1}


class _StubChat:
    def __init__(self):
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(dict(payload))
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                "petri_intervention_hash": "cc" * 8}


def test_normalize_decode_validates_and_refuses_unknown_keys():
    assert st.normalize_decode(None) is None
    assert st.normalize_decode({}) is None
    assert st.normalize_decode({"top_p": None, "top_k": None}) is None
    assert st.normalize_decode(VENDOR) == {"top_p": 0.6, "top_k": 2, "repetition_penalty": 1.1}
    with pytest.raises(st.ServedTargetError):
        st.normalize_decode({"topp": 0.6})
    with pytest.raises(st.ServedTargetError):
        st.normalize_decode({"top_p": 1.5})
    # top_k 0 is the wheel's DISABLE value (contracts.PROTOCOL_TOP_K) and -1 its legacy alias: both pass, sent as given
    # (until 2026-09-16 this validator refused 0 -- the two validators were inverted; tests/test_multiturn_vendor_default.py)
    assert st.normalize_decode({"top_k": 0}) == {"top_k": 0} and st.normalize_decode({"top_k": -1}) == {"top_k": -1}
    with pytest.raises(st.ServedTargetError):
        st.normalize_decode({"top_k": -2})
    with pytest.raises(st.ServedTargetError):
        st.normalize_decode({"repetition_penalty": 0.0})


def test_greedy_call_never_carries_the_extras_and_records_greedy(tmp_path):
    chat = _StubChat()
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x",
                            timing_log=tmp_path / "t.jsonl", decode=VENDOR)
    t.chat([{"role": "user", "content": "hi"}], max_new_tokens=8)
    p = chat.payloads[0]
    assert p["temperature"] == 0.0
    assert not (set(p) & set(st.DECODE_EXTRA_KEYS)), "a greedy call must not send sampling extras"
    assert t.last_decode == {"do_sample": False, "temperature": 0.0, "temperature_source": "request", "seed": None,
                             "top_p": None, "top_k": None, "repetition_penalty": None}
    row = json.loads((tmp_path / "t.jsonl").read_text().splitlines()[0])
    assert row["decode"] == t.last_decode


def test_sampled_call_sends_the_preset_and_every_row_records_it(tmp_path):
    chat = _StubChat()
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x",
                            timing_log=tmp_path / "t.jsonl", decode=VENDOR)
    t.chat([{"role": "user", "content": "hi"}], max_new_tokens=8, do_sample=True, temperature=0.8)
    p = chat.payloads[0]
    assert p["temperature"] == 0.8 and p["top_p"] == 0.6 and p["top_k"] == 2 and p["repetition_penalty"] == 1.1
    assert t.last_decode == {"do_sample": True, "temperature": 0.8, "temperature_source": "request", "seed": None,
                             "top_p": 0.6, "top_k": 2, "repetition_penalty": 1.1}
    row = json.loads((tmp_path / "t.jsonl").read_text().splitlines()[0])
    assert row["decode"] == t.last_decode
    man = json.loads(st.write_target_manifest(tmp_path, t, stage="final").read_text())
    assert man["target_decode"] == {"top_p": 0.6, "top_k": 2, "repetition_penalty": 1.1}
    assert man["last_request_decode"] == t.last_decode


def test_default_target_without_a_decode_block_is_byte_identical(tmp_path):
    chat = _StubChat()
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x")
    t.chat([{"role": "user", "content": "hi"}], max_new_tokens=8, do_sample=True, temperature=1.0)
    p = chat.payloads[0]
    assert set(p) == {"model", "messages", "max_tokens", "temperature"}
    assert t.decode is None and t.last_decode["temperature"] == 1.0 and t.last_decode["top_k"] is None


class _Goal:
    def __init__(self, gid, text="objective", category="c"):
        self.goal_id, self.text, self.category = gid, text, category


class _FakeDefense:
    """Records every chat() kwargs set and exposes a served-style last_decode."""

    def __init__(self):
        self.calls = []
        self.n_target_forwards = 0
        self.last_decode = None

    def chat(self, messages, **kw):
        self.calls.append(dict(kw))
        self.n_target_forwards += 1
        t = kw.get("temperature", 0.0) if kw.get("do_sample") else 0.0
        self.last_decode = {"do_sample": bool(kw.get("do_sample")), "temperature": t,
                            "temperature_source": ("server_generation_config" if kw.get("do_sample") and t is None
                                                   else "request"),
                            "seed": None, "top_p": None, "top_k": None, "repetition_penalty": None}
        return "reply"


def _stub_attack_single(chat_fn, attacker_fn, scorer_fn, objective, *, on_resp=None, **kw):
    reply = chat_fn([{"role": "user", "content": "q1"}])
    if on_resp is not None:
        on_resp(reply)
    return {"succeeded": False, "best_objective": 0.0, "actors_used": 1}


@pytest.mark.parametrize("temp,expect_kw", [(0.0, {"max_new_tokens": 512}),
                                            (0.8, {"max_new_tokens": 512, "do_sample": True, "temperature": 0.8}),
                                            # None = SAMPLE, no temperature: the server's generation config governs
                                            (None, {"max_new_tokens": 512, "do_sample": True, "temperature": None})])
def test_actorattack_threads_target_temperature_and_records_decode(monkeypatch, tmp_path, temp, expect_kw):
    monkeypatch.setattr(multiturn, "_attack_single", _stub_attack_single)
    d = _FakeDefense()
    multiturn.run_actorattack(d, [_Goal("g1")], attacker_fn=lambda m: "{}", scorer_fn=lambda p: "no",
                              out_dir=tmp_path, resume=False, target_temperature=temp)
    assert d.calls == [expect_kw], "the greedy default must leave the chat kwargs byte-identical; T>0 samples"
    rows = [json.loads(l) for l in (tmp_path / "responses.jsonl").read_text().splitlines() if l.strip()]
    assert len(rows) == 1 and rows[0]["response"] == "reply"
    assert rows[0]["decode"]["do_sample"] is (temp is None or temp > 0) and rows[0]["decode"]["temperature"] == temp
    prog = json.loads((tmp_path / "progress.jsonl").read_text().splitlines()[0])
    assert prog["outcome"] == "failure" and prog["n_target_forwards"] == 1


def test_actorattack_hf_style_defense_without_last_decode_writes_plain_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(multiturn, "_attack_single", _stub_attack_single)

    class _HF(_FakeDefense):
        def chat(self, messages, **kw):
            self.calls.append(dict(kw)); self.n_target_forwards += 1
            return "reply"
    d = _HF(); del d.last_decode
    multiturn.run_actorattack(d, [_Goal("g1")], attacker_fn=lambda m: "{}", scorer_fn=lambda p: "no",
                              out_dir=tmp_path, resume=False)
    rows = [json.loads(l) for l in (tmp_path / "responses.jsonl").read_text().splitlines() if l.strip()]
    assert set(rows[0]) == {"goal_id", "order", "response"}


# --- the 4,096-cap multi-turn re-run (2026-09-11): a decode NOTE on every block, and the client timeout ---

def test_decode_note_rides_every_decode_block_only_when_set(tmp_path):
    """A model that ships no generation_config.json runs a sampled arm on vLLM's defaults; the rows'
    null top_p/top_k must then carry WHY (the note) -- and a target built without a note must not
    grow a key (every earlier record is byte-stable)."""
    chat = _StubChat()
    note = "qwen35_4b_ships_no_generation_config.json;vllm_defaults_govern_top_p_top_k"
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x",
                            timing_log=tmp_path / "t.jsonl", decode_note=note)
    t.chat([{"role": "user", "content": "hi"}], max_new_tokens=4096, do_sample=True, temperature=1.0)
    assert t.last_decode["note"] == note and t.last_decode["top_p"] is None and t.last_decode["top_k"] is None
    assert chat.payloads[0]["max_tokens"] == 4096 and "note" not in chat.payloads[0]  # never sent to the server
    row = json.loads((tmp_path / "t.jsonl").read_text().splitlines()[0])
    assert row["decode"]["note"] == note and row["max_tokens"] == 4096
    t2 = st.ServedChatTarget(_StubChat(), virtual_model="petri__x", cell_id="x", timing_log=tmp_path / "u.jsonl")
    t2.chat([{"role": "user", "content": "hi"}], max_new_tokens=8)
    assert "note" not in t2.last_decode and t2.decode_note is None
    assert st.ServedChatTarget(_StubChat(), virtual_model="v", cell_id="c", decode_note="").decode_note is None


class _StubRoute:
    # `none`: the one cell id that needs no TrainTest->harness registry lookup on main's variant path
    cell_id = "none"; virtual_model = "petri__none"; serving_identity = {"lens": "stacked"}
    si_prefix_hash = None; intervention_hash = "cc" * 8


class _StubClient:
    made = []

    def __init__(self, url, *, identity, **kw):
        self.base_url = url; self.identity = identity; self.kw = dict(kw)
        self.timeout_s = float(kw.get("timeout_s", 600.0))
        self.served_models = ["petri__none"]; self.proxy_identity = {"mode": "identity"}
        _StubClient.made.append(self)

    def load(self, require_virtual_models=()):
        assert list(require_virtual_models) == ["petri__none"]

    def chat_completions(self, payload):
        return _StubChat()(payload)

    def tokenize(self, payload):
        return {"count": 1}


def _stub_build(monkeypatch):
    monkeypatch.setattr(st.proxy_mod, "load_routing_table", lambda p: {"routes": [_StubRoute()], "base_model": "m"})
    monkeypatch.setattr(st, "route_for", lambda cell_id, routes: routes[0])
    monkeypatch.setattr(st, "ProxyClientEngine", _StubClient)
    _StubClient.made.clear()


def test_build_served_target_threads_the_client_timeout_and_records_it(monkeypatch, tmp_path):
    """The 600 s default killed four tasks on a compute node under 4,096-token replies; a run raises it
    DELIBERATELY and the value is a recorded fact (route_info -> served_target.json), never a guess."""
    _stub_build(monkeypatch)
    t = st.build_served_target(proxy_url="http://p/v1", routing_table=tmp_path / "rt.json", cell_id="none",
                               controls=False, timeout_s=3600, decode_note="n1")
    assert _StubClient.made[-1].kw == {"timeout_s": 3600.0}
    assert t.route_info["client_timeout_s"] == 3600.0 and t.decode_note == "n1"
    man = json.loads(st.write_target_manifest(tmp_path, t).read_text())
    assert man["client_timeout_s"] == 3600.0 and man["target_decode_note"] == "n1"
    # default: no timeout kwarg reaches the engine (its own 600 s stands) and the record says so
    t0 = st.build_served_target(proxy_url="http://p/v1", routing_table=tmp_path / "rt.json", cell_id="none", controls=False)
    assert "timeout_s" not in _StubClient.made[-1].kw and t0.route_info["client_timeout_s"] == 600.0
    assert t0.decode_note is None
    with pytest.raises(st.ServedTargetError):
        st.build_served_target(proxy_url="http://p/v1", routing_table=tmp_path / "rt.json", cell_id="none",
                               controls=False, timeout_s=0)



# --------------------------------------------------------------------------- #
# THE TARGET DECODE MODES: the server's generation config (the 2026-09-11 ->     #
# 2026-09-16 default, still `attacks.target_decode: generation_config`), greedy,  #
# explicit -- and the vendor preset, THE CAMPAIGN DEFAULT since 2026-09-16        #
# (tests/test_multiturn_vendor_default.py carries the default's own tests)        #
# --------------------------------------------------------------------------- #

from sisteer.eval.attacks import target_decode as td  # noqa: E402


def test_target_decode_helpers_name_the_modes():
    assert td.MODES == ("generation_config", "greedy", "explicit", "vendor_preset")
    assert td.CONFIG_MODES == ("vendor_preset", "generation_config", "greedy")
    assert td.DEFAULT_CONFIG_MODE == "vendor_preset" == td.VENDOR_PRESET == td.SOURCE_VENDOR_PRESET
    # the wire-level classification of a call's temperature is unchanged (a vendor preset is EXPLICIT on the wire)
    assert (td.decode_mode(None), td.decode_mode(0.0), td.decode_mode(0.7)) == ("generation_config", "greedy", "explicit")
    assert td.do_sample_for(None) is True and td.do_sample_for(0.0) is False and td.do_sample_for(0.7) is True
    assert td.temperature_source(None, True) == "server_generation_config"
    assert td.temperature_source(0.7, True) == "request" and td.temperature_source(0.0, False) == "request"
    assert td.temperature_source(None, False) == "request"          # a greedy call always sends 0.0


def test_generation_config_call_carries_no_sampling_key_and_records_the_source(tmp_path):
    """temperature=None + do_sample: the request has NO temperature / top_p / top_k / repetition_penalty
    key, so vLLM fills every one from the served model's generation config (its shipped
    generation_config.json, or the standup's --override-generation-config); the row says so."""
    chat = _StubChat()
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x", timing_log=tmp_path / "t.jsonl",
                            decode={"top_p": 0.6, "top_k": 2})   # a decode block rides only an EXPLICIT temperature
    t.chat([{"role": "user", "content": "hi"}], max_new_tokens=4096, do_sample=True, temperature=None)
    req = chat.payloads[0]
    assert set(req) == {"model", "messages", "max_tokens"}, req     # nothing about sampling travels
    assert t.last_decode == {"do_sample": True, "temperature": None, "temperature_source": "server_generation_config",
                             "seed": None, "top_p": None, "top_k": None, "repetition_penalty": None}
    row = json.loads((tmp_path / "t.jsonl").read_text().splitlines()[0])
    assert row["decode"] == t.last_decode and row["max_tokens"] == 4096
    # a greedy call on the same target is unchanged: temperature 0.0 on the wire, source "request"
    t.chat([{"role": "user", "content": "hi"}], max_new_tokens=8)
    assert chat.payloads[1]["temperature"] == 0.0 and t.last_decode["temperature_source"] == "request"
    # an explicit temperature still sends the block
    t.chat([{"role": "user", "content": "hi"}], max_new_tokens=8, do_sample=True, temperature=0.7)
    assert (chat.payloads[2]["temperature"], chat.payloads[2]["top_p"], chat.payloads[2]["top_k"]) == (0.7, 0.6, 2)


def test_driver_wire_level_decode_resolution_for_the_stated_modes():
    """jailbreak_run's wire-level temperature for the modes that HAVE one: an explicit
    attacks.target_decode: generation_config -> None (no sampling key; the 2026-09-11 -> 2026-09-16
    default, byte-identical to what it sent then); greedy / --target-greedy -> 0.0; --target-temperature
    wins over both; the config's attacks.target_temperature fills in when no flag is given; the extras need
    an EXPLICIT temperature (a generation_config run sends no sampling key at all, a vendor-preset run
    sends exactly the preset's keys, so an extra would be a record of something never sent). The DEFAULT
    (no flag, no config key) is the vendor preset since 2026-09-16 -- its resolution has no wire-level
    temperature here and is pinned in tests/test_multiturn_vendor_default.py."""
    import argparse
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    import jailbreak_run as jr
    p = argparse.ArgumentParser()
    jr._add_target_engine_args(p)
    ns = p.parse_args([])
    assert ns.target_temperature is None and ns.target_greedy is False
    gen = {"attacks": {"target_decode": "generation_config"}}
    assert jr._target_decode_mode(gen, ns) == "generation_config" and jr._target_temperature(gen, ns) is None
    assert jr._target_decode_mode({"attacks": {"target_decode": "greedy"}}, ns) == "greedy"
    assert jr._target_temperature({"attacks": {"target_decode": "greedy"}}, ns) == 0.0
    assert jr._target_temperature({"attacks": {"target_temperature": 0.5}}, ns) == 0.5
    with pytest.raises(SystemExit, match="target_decode"):
        jr._target_decode_mode({"attacks": {"target_decode": "vendor"}}, ns)      # the config word is vendor_preset
    assert jr._target_temperature({}, p.parse_args(["--target-greedy"])) == 0.0
    assert jr._target_temperature({}, p.parse_args(["--target-greedy", "--target-temperature", "0.7"])) == 0.7
    assert jr._target_temperature({"attacks": {"target_decode": "greedy"}}, p.parse_args(["--target-temperature", "1.0"])) == 1.0
    with pytest.raises(SystemExit, match="explicit --target-temperature"):
        jr._target_decode({}, p.parse_args(["--target-top-p", "0.9"]))            # the vendor-preset default
    with pytest.raises(SystemExit, match="explicit --target-temperature"):
        jr._target_decode(gen, p.parse_args(["--target-top-p", "0.9"]))           # the explicit generation config
    with pytest.raises(SystemExit, match="greedy call never"):
        jr._target_decode({}, p.parse_args(["--target-greedy", "--target-top-p", "0.9"]))
    assert jr._target_decode({}, p.parse_args(["--target-temperature", "0.7", "--target-top-p", "0.9"])) == {"top_p": 0.9}
    # the manifest block names the mode and where the temperature comes from
    rec = jr._target_decode_record(gen, ns)
    assert rec == {"mode": "generation_config", "temperature": None, "do_sample": True,
                   "temperature_source": "server_generation_config"}
    rec_g = jr._target_decode_record({}, p.parse_args(["--target-greedy"]))
    assert rec_g == {"mode": "greedy", "temperature": 0.0, "do_sample": False, "temperature_source": "request"}
    rec_e = jr._target_decode_record({}, p.parse_args(["--target-temperature", "0.8", "--target-top-k", "64",
                                                        "--target-decode-note", "shipped@ee0ef602"]))
    assert rec_e == {"mode": "explicit", "temperature": 0.8, "do_sample": True, "temperature_source": "request",
                     "top_k": 64, "note": "shipped@ee0ef602"}


def test_pyrit_target_samples_with_no_temperature_under_the_default(tmp_path):
    """The PyRIT target's do_sample follows target_decode.do_sample_for: None SAMPLES with temperature
    None reaching defense.chat (the server's generation config), 0.0 is greedy, t > 0 samples at t."""
    import asyncio

    from pyrit.models import Message

    from sisteer.eval.attacks import pyrit_adapters as pa

    async def go():
        await pa.init_pyrit_memory_async(tmp_path / "m.db")
        for temp, want in ((None, True), (0.0, False), (0.7, True)):
            d = _FakeDefense()
            tgt = pa.SteeredPromptTarget(defense=d, do_sample=td.do_sample_for(temp), temperature=temp)
            assert tgt._do_sample is want and tgt._temperature == temp
            await tgt.send_prompt_async(message=Message.from_prompt(prompt="hi", role="user"))
            assert d.calls[-1]["do_sample"] is want and d.calls[-1]["temperature"] == temp
    asyncio.run(go())
