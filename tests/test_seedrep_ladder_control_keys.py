"""Every control-plane Dict NAME the code builds is ``seedrep-<window>-<wave_uuid8>-<purpose>`` and every KEY
inside carries the full six-axis tuple ``(window, wave_uuid, leg, view, stage_or_cap, shard)`` — so a base
attempt and its rung attempts never collide, and a dead wave's state is unreachable by construction. Known-bad
inputs (a key short one axis, a malformed wave uuid, an off-vocabulary stage, a separator inside an axis) are
REFUSED, never coerced. Synthetic windows / uuids throughout."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _seedrep_testlib as T  # noqa: E402

L = T.load_ladder()


def test_every_dict_name_carries_W7_and_the_wave_uuid8_for_every_purpose():
    wave8 = T.WAVE.replace("-", "")[:8]
    for purpose in L.PURPOSES:
        name = L.dict_name(T.W7, T.WAVE, purpose)
        assert name == f"seedrep-{T.W7}-{wave8}-{purpose}"
        assert name.split("-")[1] == T.W7 and name.split("-")[2] == wave8
    assert set(L.PURPOSES) == {"accepted", "claims", "results", "heartbeats", "sentinels", "stop"}
    # two waves of identical plan content never share a Dict
    assert L.dict_name(T.W7, T.WAVE, "accepted") != L.dict_name(T.W7, T.WAVE2, "accepted")
    # known-bad: a malformed wave uuid, an unknown purpose, a W7 that is not seven hex chars
    with pytest.raises(L.LadderRefusal, match="uuid4"):
        L.dict_name(T.W7, "not-a-uuid", "accepted")
    with pytest.raises(L.LadderRefusal, match="purpose"):
        L.dict_name(T.W7, T.WAVE, "bogus")
    with pytest.raises(L.LadderRefusal, match="seven"):
        L.dict_name("0c192511", T.WAVE, "accepted")


def _axes(key_string: str):
    k, extra = L.parse_key(key_string)
    return (k.W7, k.wave_uuid, k.leg, k.view, k.stage, k.shard), extra


def test_every_key_the_code_builds_carries_all_six_axes():
    spec = T.static_spec(L, shards=("0",))
    shared = L.SharedVolume()
    kv = T.kvs(L)
    c = T.Container(L, shared, spec, "0", "a1")
    rec = c.enqueue_row(T.static_row("g0", 1, seed=7))
    # (1) the acceptance key: the producer's six axes, stage `base`
    L.accept_attempt(kv["accepted"], c.ref.key, "a1")
    (acc_key, _) = kv["accepted"].ops[-1][1], None
    assert _axes(acc_key) == ((T.W7, T.WAVE, "pasttense", "s1", "base", "0"), ())
    # (2) a claim key: the QUEUE's six axes (stage = the cap, shard = `queue`) + arm_s + item_key
    assert L.claim_item(kv["claims"], spec, rec["arm_s"], rec["item_key"], attempt_id="r1", call_id="fc-1")
    axes, extra = _axes(kv["claims"].ops[-1][1])
    assert axes == (T.W7, T.WAVE, "pasttense", "s1", "8192", "queue") and extra == ("none__s1", rec["item_key"])
    # (3) the elected-result key is the same shape
    rvol = shared.snapshot("r1")
    path, doc = L.write_materialised(rvol, spec, queue_record=rec, attempt_id="r1", call_id="fc-1",
                                     record={"text": "t", "finish_reason": "stop", "completion_tokens": 1, "prompt_tokens": 1,
                                             "max_tokens": 8192, "sampling_seed": 7}, text="t", request_id="rq", completion_id="cm")
    assert L.elect_result(kv["results"], spec, doc, path)
    axes, extra = _axes(kv["results"].ops[-1][1])
    assert axes == (T.W7, T.WAVE, "pasttense", "s1", "8192", "queue") and extra == ("none__s1", rec["item_key"])
    # (4) the records / inventory / markers carry the producer's key string; the closure the queue's
    c.complete()
    c.finish(kv["accepted"])
    assert _axes(c.producer.inventory["key"])[0] == (T.W7, T.WAVE, "pasttense", "s1", "base", "0")
    status = L.reconcile_queue(shared.read_only_view(), kv["accepted"], spec)
    assert _axes(status["closed"]["key"])[0] == (T.W7, T.WAVE, "pasttense", "s1", "8192", "queue")
    # every Dict key ever written carries six axes
    for kv_ in kv.values():
        for op, key, _ in kv_.ops:
            k, _ = L.parse_key(key)
            assert (k.W7, k.wave_uuid, k.leg, k.view) == (T.W7, T.WAVE, "pasttense", "s1") and k.stage in L.STAGES


@pytest.mark.parametrize("bad, match", [
    ("0c19251|11111111-2222-4333-8444-555555555555|pasttense|s1|base", "5 component"),        # one axis short
    ("0c19251|11111111-2222-4333-8444-555555555555|pasttense|s1|4096|0", "stage"),          # an off-vocabulary stage
    ("0c19251|wave|pasttense|s1|base|0", "uuid4"),                                          # a malformed wave uuid
    ("0c1925|11111111-2222-4333-8444-555555555555|pasttense|s1|base|0", "seven"),           # a six-char W7
    ("0c19251|11111111-2222-4333-8444-555555555555||s1|base|0", "leg"),                     # an empty axis
])
def test_a_key_missing_or_malforming_an_axis_is_refused(bad, match):
    with pytest.raises(L.LadderRefusal, match=match):
        L.parse_key(bad)


def test_an_axis_or_extra_component_carrying_the_separator_is_refused():
    with pytest.raises(L.LadderRefusal):
        L.SixAxisKey(T.W7, T.WAVE, "pasttense", "s1", "base", "0|1")
    k = L.SixAxisKey(T.W7, T.WAVE, "pasttense", "s1", "8192", "queue")
    with pytest.raises(L.LadderRefusal, match="separator"):
        L.key_str(k, "none__s1", "static__a|b")
    with pytest.raises(L.LadderRefusal):
        L.key_str(k, "")


def test_base_and_rung_attempts_of_one_shard_are_distinct_keys_in_one_dict():
    base = L.SixAxisKey(T.W7, T.WAVE, "pasttense", "s1", "base", "0")
    rung = base.with_stage("8192")
    assert L.key_str(base) != L.key_str(rung)
    kv = L.MemoryKV()
    assert L.accept_attempt(kv, base, "a-base") and L.accept_attempt(kv, rung, "a-rung")
    assert L.accepted_attempt(kv, base) == "a-base" and L.accepted_attempt(kv, rung) == "a-rung"
    # a second wave's identical key never reaches the first wave's acceptance
    other = L.SixAxisKey(T.W7, T.WAVE2, "pasttense", "s1", "base", "0")
    assert L.accepted_attempt(kv, other) is None and L.accept_attempt(kv, other, "b-base")


def test_item_key_is_one_derivation_path_safe_and_deterministic():
    assert L.item_key("static", "pasttense", "g0", 1) == "static__pasttense__g0__1"
    hashed = L.item_key("static", "pasttense", "goal/with/slashes", 1)
    assert hashed.startswith("static__h") and "/" not in hashed and "|" not in hashed
    assert hashed == L.item_key("static", "pasttense", "goal/with/slashes", 1)
    assert L.item_key("misalign", "leaking_explicit-none", 3) == "misalign__leaking_explicit-none__3"
    with pytest.raises(L.LadderRefusal, match="kind"):
        L.item_key("bon", "g0", 1)
    assert L.arm_s("none", 1) == "none__s1" and L.arm_s("em-bad@none", 2) == "em-bad@none__s2"
    with pytest.raises(L.LadderRefusal):
        L.arm_s("none", 0)
