"""The steer-is-live positive control's verdict helper (scripts/sdfbase/full22/positive_control_verdict.py), driven on a
RECORDED result: sdfbase-llama8b job 95642 (2026-09-12) — every check PASS except `determinism`, which flipped on a
half of the probes on EVERY leg (steered 4/8, anchor 4/8, twin 4/8) under the merged serving tree. Default: FAIL (as before).
With the lane's opt-in F22_PC_TOLERATE_LENS_NONDETERMINISM=1: PASS_WITH_LENS_NONDETERMINISM. Any failing steer-is-live check,
or every probe flipping on a leg, still refuses under the opt-in; the anchor-only rule is unchanged."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HELPER = REPO / "scripts" / "sdfbase" / "full22" / "positive_control_verdict.py"
FIXTURE = REPO / "tests" / "fixtures" / "positive_control_lens_nondeterminism_llama8b_95642.json"
sys.path.insert(0, str(HELPER.parent))
import positive_control_verdict as pcv  # noqa: E402


def _rec():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _det(rec):
    return next(c for c in rec["checks"] if c["check"] == "determinism")


def test_the_recorded_llama8b_result_refuses_by_default_and_passes_with_warning_under_the_opt_in():
    rec = _rec()
    assert _det(rec)["status"] == "FAIL" and all(c["status"] == "PASS" for c in rec["checks"] if c["check"] != "determinism")
    assert pcv.verdict(rec, tolerate_lens=False) == "FAIL determinism"
    v = pcv.verdict(rec, tolerate_lens=True)
    assert v.startswith("PASS_WITH_LENS_NONDETERMINISM") and "steer_live_checks=PASS" in v
    for leg in ("steered", "anchor", "twin"):
        assert f"{leg}_nondeterministic=" in v


def test_a_failing_steer_live_check_or_every_probe_flipping_still_refuses_under_the_opt_in():
    rec = _rec()
    bites = copy.deepcopy(rec)
    next(c for c in bites["checks"] if c["check"] == "steering_bites")["status"] = "FAIL"
    assert pcv.verdict(bites, tolerate_lens=True) == "FAIL determinism,steering_bites"
    every = copy.deepcopy(rec)
    d = _det(every)
    d["nondeterministic_prompts"]["steered"] = [f"p{i}" for i in range(int(d["n_prompts"]))]   # ALL probes flipped on the lens leg
    assert pcv.verdict(every, tolerate_lens=True) == "FAIL determinism"


def test_the_anchor_only_rule_is_unchanged_and_the_cli_maps_verdicts_to_exit_codes(tmp_path):
    rec = _rec()
    d = _det(rec)
    d["nondeterministic_prompts"] = {"steered": [], "twin": [], "anchor": ["p1", "p2"]}
    assert pcv.verdict(rec, tolerate_lens=False).startswith("PASS_WITH_ANCHOR_NONDETERMINISM anchor_nondeterministic=2/")
    p = tmp_path / "pc.json"
    p.write_text(json.dumps(_rec()), encoding="utf-8")
    off = subprocess.run([sys.executable, str(HELPER), str(p)], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert off.returncode == 1 and off.stdout.strip() == "FAIL determinism"
    on = subprocess.run([sys.executable, str(HELPER), str(p)], capture_output=True, text=True,
                        env={"PATH": "/usr/bin:/bin", "F22_PC_TOLERATE_LENS_NONDETERMINISM": "1"})
    assert on.returncode == 0 and on.stdout.startswith("PASS_WITH_LENS_NONDETERMINISM")


def _nf(rec):
    return next(c for c in rec["checks"] if c["check"] == "noise_floor")


def test_d179_anchor_noise_that_also_reaches_the_noise_floor_is_the_same_recorded_warning():
    # The shape of 2026-09-15 llama8b lens legs 518075 / 518076 / 518077: `determinism` failed on the ANCHOR only
    # (steered [], twin []), `noise_floor` differed on a MINORITY of probes (2 of 8), every steer-is-live check PASS.
    # common.sh's inline fallback has admitted this since 2026-09-14; the helper, which takes precedence, refused it.
    rec = _rec()
    _det(rec)["nondeterministic_prompts"] = {"steered": [], "twin": [], "anchor": ["p1", "p2"]}
    nf = _nf(rec)
    nf["status"] = "FAIL"; nf["n_prompts"] = 8; nf["n_anchor_eq_twin"] = 6; nf["differing_prompts"] = ["q1", "q2"]
    v = pcv.verdict(rec, tolerate_lens=False)
    assert v.startswith("PASS_WITH_ANCHOR_NONDETERMINISM anchor_nondeterministic=2/")
    assert "noise_floor_anchor_vs_twin_differing=2/8" in v and "D179" in v
    # a MAJORITY of differing probes says the routes differ, not that the anchor is noisy: refused
    maj = copy.deepcopy(rec); _nf(maj)["differing_prompts"] = ["q1", "q2", "q3", "q4", "q5"]
    assert pcv.verdict(maj, tolerate_lens=False).startswith("FAIL") and "noise_floor" in pcv.verdict(maj, tolerate_lens=False)
    # a twin that does not repeat byte-identically is not anchor noise: refused
    tw = copy.deepcopy(rec); _det(tw)["nondeterministic_prompts"]["twin"] = ["p1"]
    assert pcv.verdict(tw, tolerate_lens=False).startswith("FAIL")
    # a third failing check is never covered by the warning
    th = copy.deepcopy(rec); next(c for c in th["checks"] if c["check"] == "steering_bites")["status"] = "FAIL"
    v3 = pcv.verdict(th, tolerate_lens=False)
    assert v3.startswith("FAIL") and "steering_bites" in v3
    # zero differing probes is not this rule's case either (noise_floor would not be failing) — the guard is 0 < k <= n/2
    z = copy.deepcopy(rec); _nf(z)["differing_prompts"] = []
    assert pcv.verdict(z, tolerate_lens=False).startswith("FAIL")
