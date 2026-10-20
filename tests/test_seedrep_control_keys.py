"""Every Dict name and every key the wave builds carries the six axes
(W7, wave_uuid, leg, view|arditi, stage_or_cap, shard) — a base attempt and its rung attempts never collide, a dead
wave's state is unreachable by construction. Known-bad: an empty axis, a slash inside an axis, an unknown purpose, a
short W7 — each REFUSES."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "modal"))

import seedrep_launch_plan as lp  # noqa: E402

W7, WAVE = "abcdef0", "01234567-89ab-4cde-8f01-23456789abcd"


def test_dict_names_carry_w7_and_wave_for_every_purpose():
    names = {p: lp.dict_name(W7, WAVE, p) for p in lp.DICT_PURPOSES}
    assert names == {"accepted": "seedrep-abcdef0-01234567-accepted", "claims": "seedrep-abcdef0-01234567-claims",
                     "results": "seedrep-abcdef0-01234567-results", "heartbeats": "seedrep-abcdef0-01234567-heartbeats",
                     "sentinels": "seedrep-abcdef0-01234567-sentinels", "stop": "seedrep-abcdef0-01234567-stop",
                     "reattempts": "seedrep-abcdef0-01234567-reattempts"}                 # the durable per-standup re-attempt ledger
    assert lp.dict_name(W7, "ffffffff" + WAVE[8:], "accepted") != names["accepted"], "another wave, another Dict"
    with pytest.raises(ValueError, match="not one of"):
        lp.dict_name(W7, WAVE, "leases")
    with pytest.raises(ValueError, match="W7"):
        lp.dict_name("abc", WAVE, "accepted")


def test_control_key_carries_six_axes_and_round_trips():
    k = lp.control_key(W7, WAVE, "pasttense", "s1", "base", 0)
    assert k == f"abcdef0/{WAVE}/pasttense/s1/base/0"
    assert lp.parse_control_key(k) == {"W7": W7, "wave_uuid": WAVE, "leg": "pasttense", "view": "s1", "stage": "base", "shard": "0", "extra": []}
    rung = lp.control_key(W7, WAVE, "pasttense", "s1", 8192, 0)
    assert rung != k and lp.parse_control_key(rung)["stage"] == "8192"
    petri = lp.control_key(W7, WAVE, "petri", "arditi", lp.STAGE_PETRI_RERUN, 0)
    assert lp.parse_control_key(petri)["stage"] == "petri-rerun"
    claim = lp.control_key(W7, WAVE, "pasttense", "s1", 8192, 0, "goal-17__3")
    assert lp.parse_control_key(claim)["extra"] == ["goal-17__3"]
    assert lp.control_key("1111111", WAVE, "pasttense", "s1", "base", 0) != k, "another W, another key"
    assert lp.control_key(W7, "ffffffff-ffff-4fff-8fff-ffffffffffff", "pasttense", "s1", "base", 0) != k, "another wave, another key"
    # the accepted Dict is keyed in the writer's spelling (the rung consumer reads it): the same six axes joined by `|`
    assert lp.accepted_key(W7, WAVE, "pasttense", "s1", "base", 0) == f"abcdef0|{WAVE}|pasttense|s1|base|0"
    with pytest.raises(ValueError, match="dashed uuid4"):
        lp.dict_name(W7, "0123456789abcdef0123456789abcdef", "accepted")


@pytest.mark.parametrize("bad", [("", "s1", "base", 0), ("pasttense", None, "base", 0), ("pasttense", "s1", "", 0),
                                 ("past/tense", "s1", "base", 0), ("pasttense", "s1", "base", "0/1")])
def test_control_key_refuses_a_missing_or_slashed_axis(bad):
    with pytest.raises(ValueError, match="non-empty slash-free"):
        lp.control_key(W7, WAVE, *bad)
    with pytest.raises(ValueError):
        lp.parse_control_key("a/b/c")


def test_every_plan_row_and_queue_producer_carries_its_six_axes():
    inputs = lp.load_inputs(REPO)
    plan = lp.build_plan(W="3" * 40, wave_uuid=WAVE, reqs_sha256=inputs["reqs_sha256"], roster=inputs["roster"], pins=inputs["pins"],
                         arditi_pins=inputs["arditi_pins"], arms=inputs["arms"], generated_at="2026-09-18T00:00:00Z")
    keys = []
    for r in plan["standups"] + plan["rungs"]:
        k = lp.parse_control_key(r["control_key"])
        assert k == {"W7": "3333333", "wave_uuid": WAVE, "leg": r["leg"], "view": r["engine"], "stage": r["stage"], "shard": str(r["shard"]), "extra": []}
        keys.append(r["control_key"])
    assert len(set(keys)) == len(keys) == 130        # 74 base + 56 executable rungs (the 4 GSM8K rungs included)
    for q in plan["queues"]:
        for p in q["producers"]:
            k = lp.parse_control_key(p["control_key"])
            assert k["leg"] == q["leg"] and k["view"] == q["engine"] and k["stage"] == q["feeder_stage"]
    assert set(plan["dicts"]) == set(lp.DICT_PURPOSES) and all(v.startswith("seedrep-3333333-01234567-") for v in plan["dicts"].values())
    assert plan["ladder_root"] == f"/vol/seedrep/3333333/{WAVE}/ladder"
