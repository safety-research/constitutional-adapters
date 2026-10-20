"""Tests for ``scripts/sdfbase/derived_verdict.py`` — the pre-registered
equivalence tests on synthetic per-item / per-doc records.

Every record here is INVENTED (ids ``p0001``…, ``d001``…); nothing is real
data. The generators plant a per-item latent structure — a values effect v_i
and a register effect c_i — so m_ctl ≈ c, m_mix ≈ v + c, and the object
either carries v alone (the decomposition's prediction) or v + c (retains
the register), which is what P1/P2 must tell apart.
"""
from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dv = _load_module("derived_verdict", REPO / "scripts/sdfbase/derived_verdict.py")

FAST_ITEMS = 400   # bootstrap draws in tests (the plan's 10k is the default)
FAST_DOCS = 300


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------


def _row(i: str, source: str, coverage: str, margin: float, n_tok: int = 40):
    """One probe_score record with the producer's RECORD_KEYS; margin_mean
    is the PRIMARY statistic, the rest is filled consistently."""
    lp_v = -3.0 * n_tok
    lp_c = lp_v + margin * n_tok
    return {"id": i, "source": source, "coverage": coverage,
            "section_id": f"s{int(i[1:]) % 7}",
            "lp_sum_consistent": lp_c, "lp_sum_violating": lp_v,
            "n_tok_consistent": n_tok, "n_tok_violating": n_tok,
            "margin_mean": margin, "margin_sum": lp_c - lp_v,
            "tie": abs(margin) < 1e-3}


def make_probe_rows(n: int = 400, *, mu_v: float = 0.035, mu_c: float = 0.05,
                    sd_latent: float = 0.03, sd_noise: float = 0.01,
                    obj_keeps_register: bool = False, seed: int = 0,
                    out_frac: float = 0.10):
    """Three aligned probe files (obj, mix, ctl). m_ctl = c + e; m_mix = v +
    c + e; m_obj = v + e (or v + c + e when ``obj_keeps_register``). About
    ``out_frac`` of the items are coverage "out" (must be excluded)."""
    rng = random.Random(seed)
    obj, mix, ctl = [], [], []
    for k in range(n):
        i = f"p{k + 1:04d}"
        src = "advice" if k % 2 == 0 else "story"
        cov = "out" if rng.random() < out_frac else "in"
        v = rng.gauss(mu_v, sd_latent)
        c = rng.gauss(mu_c, sd_latent)
        m_ctl = c + rng.gauss(0, sd_noise)
        m_mix = v + c + rng.gauss(0, sd_noise)
        m_obj = (v + c if obj_keeps_register else v) + rng.gauss(0, sd_noise)
        obj.append(_row(i, src, cov, m_obj))
        mix.append(_row(i, src, cov, m_mix))
        ctl.append(_row(i, src, cov, m_ctl))
    return obj, mix, ctl


def make_ce_rows(n_docs: int = 20, *, gain_obj: float = 0.0,
                 gain_mix: float = 0.05, gain_ctl: float = 0.05,
                 sd_gain: float = 0.005, seed: int = 0,
                 noop_jitter: float = 0.0):
    """Three aligned ce_score per-doc files on the same docs. Each doc's
    no-op leg is shared (up to ``noop_jitter`` relative), the active legs
    carry the requested per-arm gains with per-doc noise."""
    rng = random.Random(seed)
    files = {"obj": [], "mix": [], "ctl": []}
    gains = {"obj": gain_obj, "mix": gain_mix, "ctl": gain_ctl}
    for k in range(n_docs):
        i = f"d{k + 1:03d}"
        ntok = rng.randint(1500, 4500)
        noop = ntok * rng.uniform(2.6, 3.4)
        for arm, g in gains.items():
            gk = g + rng.gauss(0, sd_gain)
            noop_k = noop * (1 + rng.uniform(-noop_jitter, noop_jitter))
            files[arm].append({"doc_id": i, "family": "ctl_heldout",
                               "source_tag": "advice" if k % 2 else "story",
                               "n_target_tokens": ntok,
                               "nll_active_sum": noop_k * (1 - gk),
                               "nll_noop_sum": noop_k})
    return files["obj"], files["mix"], files["ctl"]


def _calibration(rows_are: str, verdict: str, *, degenerate: bool = False):
    per = {
        "advice": {"verdict": verdict,
                   "sentence": f"stratum advice: planted sentence A ({verdict})"},
        "story": {"verdict": verdict,
                  "sentence": f"stratum story: planted sentence B ({verdict})"},
    }
    return {"recovery": {"verdict": verdict, "per_stratum": per,
                         "p1_p3_rows_are": rows_are,
                         "degenerate_parents_identical": degenerate},
            "arithmetic_residual_inconsistency": {
                "8": {"rms_inconsistency": 0.11, "rms_teacher_deviation": 1.9,
                      "ratio": 0.058}}}


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows),
                    encoding="utf-8")


# ---------------------------------------------------------------------------
# Constants are the plan's numbers
# ---------------------------------------------------------------------------


def test_constants_pinned_to_plan_2_6():
    assert dv.P1_TOST_MARGIN_NATS == 0.015
    assert dv.P1_TOST_CI_LEVEL == 0.90
    assert dv.P1_OWN_MARGIN_CI_LEVEL == 0.95
    assert dv.P2_REMNANT_MAX == 0.25
    assert dv.P2_CI_LEVEL == 0.90
    assert dv.P3_GAIN_MARGIN == 0.02
    assert dv.P3_PARENT_SHARE_MAX == 0.25
    assert dv.P3_CI_LEVEL == 0.90
    assert dv.N_BOOT_ITEMS == 10_000
    assert dv.N_BOOT_DOCS == 1_000
    assert dv.BOOT_SEED == 0


def test_verdict_vocabulary_is_exactly_plan_4_4():
    assert set(dv.VERDICTS) == {
        "confirmed", "falsified",
        "not recovered under this trainer + budget", "descriptive-only"}
    assert len(dv.VERDICTS) == 4
    assert dv.BANNED_WORDS == ("inexpressible",)
    assert set(dv.STATUSES) == {"pass", "fail", "inconclusive", "undefined"}


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def test_paired_items_keeps_only_in_coverage_and_pairs_by_id():
    obj, mix, ctl = make_probe_rows(400, seed=1)
    items = dv.paired_items(obj, mix, ctl)
    n_in = sum(1 for r in obj if r["coverage"] == "in")
    assert 300 < n_in < 400          # ~10% were marked out
    assert len(items) == n_in
    assert all(it["id"] == i for it, i in zip(
        items, sorted(r["id"] for r in obj if r["coverage"] == "in")))
    by_id = {r["id"]: r for r in obj}
    for it in items[:5]:
        assert it["m_obj"] == by_id[it["id"]]["margin_mean"]
        assert it["d1"] == pytest.approx(it["m_obj"] - (it["m_mix"] - it["m_ctl"]))


def test_paired_items_refuses_id_mismatch():
    obj, mix, ctl = make_probe_rows(50, seed=2)
    with pytest.raises(dv.InputError, match="id sets differ"):
        dv.paired_items(obj, mix[:-1], ctl)
    ctl2 = [dict(r) for r in ctl]
    ctl2[0]["id"] = "p9999"
    with pytest.raises(dv.InputError, match="id sets differ"):
        dv.paired_items(obj, mix, ctl2)


def test_paired_items_refuses_duplicate_ids_and_coverage_disagreement():
    obj, mix, ctl = make_probe_rows(50, seed=3)
    with pytest.raises(dv.InputError, match="duplicate"):
        dv.paired_items(obj + [obj[0]], mix + [mix[0]], ctl + [ctl[0]])
    mix2 = [dict(r) for r in mix]
    mix2[3]["coverage"] = "out" if mix2[3]["coverage"] == "in" else "in"
    with pytest.raises(dv.InputError, match="coverage disagrees"):
        dv.paired_items(obj, mix2, ctl)


def test_paired_docs_refuses_id_mismatch_and_noop_disagreement():
    o, m, c = make_ce_rows(20, seed=4)
    with pytest.raises(dv.InputError, match="doc_id sets differ"):
        dv.paired_docs(o, m, c[1:])
    c2 = [dict(r) for r in c]
    c2[5]["nll_noop_sum"] *= 1.05          # 5% off — not the same bare model
    with pytest.raises(dv.InputError, match="nll_noop_sum disagrees"):
        dv.paired_docs(o, m, c2)
    # within tolerance is fine
    o3, m3, c3 = make_ce_rows(20, seed=4, noop_jitter=2e-4)
    assert len(dv.paired_docs(o3, m3, c3)) == 20
    # and the tolerance is a knob, recorded by P3
    with pytest.raises(dv.InputError, match="nll_noop_sum disagrees"):
        dv.paired_docs(o3, m3, c3, noop_rtol=1e-6)
    m4 = [dict(r) for r in m]
    m4[2]["n_target_tokens"] += 1
    with pytest.raises(dv.InputError, match="n_target_tokens disagrees"):
        dv.paired_docs(o, m4, c)


# ---------------------------------------------------------------------------
# P1
# ---------------------------------------------------------------------------


def test_p1_passes_when_object_carries_the_subtraction():
    items = dv.paired_items(*make_probe_rows(400, seed=10))
    p1 = dv.p1_tost(items, n_boot=FAST_ITEMS)
    assert p1["status"] == "pass"
    assert p1["tost_within_margin"] and p1["own_margin_excludes_zero"]
    assert abs(p1["mean_d1"]) < 0.005
    assert -0.015 < p1["d1_ci"][0] < p1["d1_ci"][1] < 0.015
    assert p1["m_obj_ci"][0] > 0
    # the lane's estimand is reported alongside, with its own CI
    assert p1["estimand_mean"] == pytest.approx(0.035, abs=0.01)
    assert p1["estimand_ci"][0] < p1["estimand_mean"] < p1["estimand_ci"][1]
    assert p1["margin_nats"] == dv.P1_TOST_MARGIN_NATS
    assert p1["n_items"] == len(items)


def test_p1_fails_when_object_retains_the_register_margin():
    items = dv.paired_items(*make_probe_rows(
        400, seed=11, obj_keeps_register=True))
    p1 = dv.p1_tost(items, n_boot=FAST_ITEMS)
    assert p1["status"] == "fail"
    assert p1["mean_d1"] == pytest.approx(0.05, abs=0.01)   # D1 ≈ m_ctl
    assert p1["d1_ci"][0] >= 0.015


def test_p1_inconclusive_with_few_noisy_items():
    obj, mix, ctl = make_probe_rows(8, seed=12, sd_noise=0.05, out_frac=0.0)
    items = dv.paired_items(obj, mix, ctl)
    p1 = dv.p1_tost(items, n_boot=FAST_ITEMS)
    assert p1["status"] == "inconclusive"
    assert p1["d1_ci"][1] - p1["d1_ci"][0] > 0.03


def test_p1_inconclusive_when_object_has_no_margin_of_its_own():
    # v ≈ 0 and c ≈ 0: D1 is equivalent to 0 but m_obj's CI includes 0
    items = dv.paired_items(*make_probe_rows(
        400, seed=13, mu_v=0.0, mu_c=0.0, sd_latent=0.005, sd_noise=0.005))
    p1 = dv.p1_tost(items, n_boot=FAST_ITEMS)
    assert p1["tost_within_margin"] is True
    assert p1["own_margin_excludes_zero"] is False
    assert p1["status"] == "inconclusive"


def test_p1_bootstrap_is_deterministic_in_the_seed():
    items = dv.paired_items(*make_probe_rows(200, seed=14))
    a = dv.p1_tost(items, n_boot=200, seed=7)
    b = dv.p1_tost(items, n_boot=200, seed=7)
    c = dv.p1_tost(items, n_boot=200, seed=8)
    assert a["d1_ci"] == b["d1_ci"] and a["m_obj_ci"] == b["m_obj_ci"]
    assert a["d1_ci"] != c["d1_ci"]


# ---------------------------------------------------------------------------
# P2
# ---------------------------------------------------------------------------


def test_p2_passes_when_remnant_is_near_zero():
    items = dv.paired_items(*make_probe_rows(400, seed=20))
    p2 = dv.p2_remnant(items, n_boot=FAST_ITEMS)
    assert p2["status"] == "pass"
    assert abs(p2["remnant"]) < 0.1
    assert p2["remnant_ci"][1] <= 0.25
    assert p2["remnant_max"] == dv.P2_REMNANT_MAX


def test_p2_fails_when_object_retains_the_register():
    items = dv.paired_items(*make_probe_rows(
        400, seed=21, obj_keeps_register=True))
    p2 = dv.p2_remnant(items, n_boot=FAST_ITEMS)
    assert p2["status"] == "fail"
    assert p2["remnant"] == pytest.approx(1.0, abs=0.15)
    assert p2["remnant_ci"][0] > 0.25


def test_p2_undefined_when_control_margin_is_not_resolved():
    items = dv.paired_items(*make_probe_rows(400, seed=22, mu_c=0.0))
    p2 = dv.p2_remnant(items, n_boot=FAST_ITEMS)
    assert p2["status"] == "undefined"
    assert "control margin not resolved" in p2["reason"]
    assert p2["remnant"] is None and p2["remnant_ci"] is None
    assert p2["m_ctl_ci"][0] <= 0.0


# ---------------------------------------------------------------------------
# P3
# ---------------------------------------------------------------------------


def test_p3_passes_when_object_loses_the_gain_parents_keep():
    o, m, c = make_ce_rows(20, gain_obj=0.0, gain_mix=0.05, gain_ctl=0.05,
                           seed=30)
    p3 = dv.p3_ce_gain(o, m, c, n_boot=FAST_DOCS)
    assert p3["status"] == "pass"
    assert abs(p3["gain"]["obj"]) < 0.01
    assert p3["gain"]["mix"] == pytest.approx(0.05, abs=0.01)
    assert p3["gain"]["ctl"] == pytest.approx(0.05, abs=0.01)
    assert p3["share_cap"] == pytest.approx(0.25 * min(p3["gain"]["mix"],
                                                       p3["gain"]["ctl"]))
    assert p3["gain_ci"]["obj"][1] <= p3["share_cap"]
    assert p3["n_docs"] == 20
    assert p3["power_caveat"] and "n_docs = 20 < 30" in p3["power_caveat"]
    # sum/sum: gain equals 1 − Σactive/Σnoop by hand
    tot_a = sum(r["nll_active_sum"] for r in o)
    tot_n = sum(r["nll_noop_sum"] for r in o)
    assert p3["gain"]["obj"] == pytest.approx(1 - tot_a / tot_n)


def test_p3_fails_when_object_keeps_the_gain():
    o, m, c = make_ce_rows(20, gain_obj=0.05, gain_mix=0.05, gain_ctl=0.05,
                           seed=31)
    p3 = dv.p3_ce_gain(o, m, c, n_boot=FAST_DOCS)
    assert p3["status"] == "fail"
    assert p3["gain_ci"]["obj"][0] >= 0.02


def test_p3_share_clause_binds_when_parents_gain_is_tiny():
    # |gain_obj| = 0.01 is inside ±0.02, but the parents only gain 0.012, so
    # the ≤ 25%·min(parent) = 0.003 clause is what rules — and it fails.
    o, m, c = make_ce_rows(40, gain_obj=0.010, gain_mix=0.012, gain_ctl=0.013,
                           sd_gain=0.001, seed=32)
    p3 = dv.p3_ce_gain(o, m, c, n_boot=FAST_DOCS)
    assert p3["within_margin"] is True
    assert p3["under_share_cap"] is False
    assert p3["status"] == "fail"
    assert p3["power_caveat"] is None          # 40 docs ≥ 30
    # a CI that straddles the cap (noisy docs, deterministic seed) is
    # inconclusive, not fail — the clause is a margin, not a point test
    o2, m2, c2 = make_ce_rows(20, gain_obj=0.003, gain_mix=0.012,
                              gain_ctl=0.013, sd_gain=0.02, seed=34)
    p3b = dv.p3_ce_gain(o2, m2, c2, n_boot=FAST_DOCS)
    assert p3b["within_margin"] is True
    assert p3b["gain_ci"]["obj"][0] <= p3b["share_cap"] < p3b["gain_ci"]["obj"][1]
    assert p3b["status"] == "inconclusive"


def test_p3_undefined_when_parents_show_no_gain():
    o, m, c = make_ce_rows(20, gain_obj=0.0, gain_mix=-0.01, gain_ctl=0.0,
                           seed=34)
    p3 = dv.p3_ce_gain(o, m, c, n_boot=FAST_DOCS)
    assert p3["status"] == "undefined"
    assert "no control-corpus gain" in p3["reason"]


def test_p3_refuses_noop_disagreement_end_to_end():
    o, m, c = make_ce_rows(20, seed=35)
    c[0] = dict(c[0], nll_noop_sum=c[0]["nll_noop_sum"] * 0.9)
    with pytest.raises(dv.InputError, match="nll_noop_sum disagrees"):
        dv.p3_ce_gain(o, m, c, n_boot=50)


# ---------------------------------------------------------------------------
# Row verdict
# ---------------------------------------------------------------------------


def _preds(*, p1="pass", p2="pass", p3="pass"):
    def mk(name, st):
        return {"prediction": name, "status": st, "reason": f"{name} {st}"}
    return mk("P1", p1), mk("P2", p2), mk("P3", p3)


def test_row_confirmed_iff_every_applicable_prediction_passes():
    row = dv.row_verdict(*_preds())
    assert row["verdict"] == "confirmed"
    assert row["basis"] == ["P1", "P2", "P3"]
    assert row["roles"] == {"P1": "confirmatory", "P2": "confirmatory",
                            "P3": "confirmatory"}
    # an undefined leg is not applicable and does not block confirmation
    row = dv.row_verdict(*_preds(p2="undefined"))
    assert row["verdict"] == "confirmed" and row["basis"] == ["P1", "P3"]
    assert "P2" in row["reason"]
    # no P3 (probe-only lane) is still judged from P1/P2
    p1, p2, _ = _preds()
    row = dv.row_verdict(p1, p2, None)
    assert row["verdict"] == "confirmed" and row["basis"] == ["P1", "P2"]
    assert row["roles"]["P3"] == "absent"


def test_row_falsified_iff_any_applicable_prediction_fails():
    for which in ("p1", "p2", "p3"):
        row = dv.row_verdict(*_preds(**{which: "fail"}))
        assert row["verdict"] == "falsified"
        assert which.upper() in row["reason"]
    # a fail beats an inconclusive elsewhere
    row = dv.row_verdict(*_preds(p1="inconclusive", p3="fail"))
    assert row["verdict"] == "falsified"


def test_row_inconclusive_leg_yields_descriptive_only_with_reason():
    row = dv.row_verdict(*_preds(p1="inconclusive"))
    assert row["verdict"] == "descriptive-only"
    assert "inconclusive" in row["reason"] and "P1" in row["reason"]


def test_row_probe_descriptive_is_judged_from_p3_alone():
    # gemma: P1/P2 descriptive whatever they say; the row follows P3
    row = dv.row_verdict(*_preds(p1="fail", p2="fail", p3="pass"),
                         probe_descriptive=True)
    assert row["verdict"] == "confirmed"
    assert row["basis"] == ["P3"]
    assert row["roles"] == {"P1": "descriptive", "P2": "descriptive",
                            "P3": "confirmatory"}
    assert "D21" in row["reason"]
    row = dv.row_verdict(*_preds(p1="pass", p2="pass", p3="fail"),
                         probe_descriptive=True)
    assert row["verdict"] == "falsified"
    # no P3 on a probe-descriptive lane: nothing confirmatory → descriptive-only
    p1, p2, _ = _preds()
    row = dv.row_verdict(p1, p2, None, probe_descriptive=True)
    assert row["verdict"] == "descriptive-only"
    assert row["basis"] == []


def test_row_calibration_not_recovered_carries_sentences_verbatim():
    cal = _calibration("descriptive", "not recovered under this trainer + budget")
    row = dv.row_verdict(*_preds(), calibration=cal)     # even all-pass
    assert row["verdict"] == "not recovered under this trainer + budget"
    assert row["roles"] == {"P1": "descriptive", "P2": "descriptive",
                            "P3": "descriptive"}
    for s in ("stratum advice: planted sentence A", "stratum story: planted sentence B"):
        assert s in row["reason"]
        assert any(s in v for v in row["recovery_gate"]["per_stratum_sentences"].values())
    assert row["recovery_gate"]["per_stratum"] == cal["recovery"]["per_stratum"]


def test_row_calibration_recovered_is_confirmatory():
    cal = _calibration("confirmatory", "recovered")
    row = dv.row_verdict(*_preds(), calibration=cal)
    assert row["verdict"] == "confirmed"
    assert row["recovery_gate"]["verdict"] == "recovered"
    row = dv.row_verdict(*_preds(p1="fail"), calibration=cal)
    assert row["verdict"] == "falsified"


def test_row_calibration_unresolvable_is_descriptive_only():
    cal = _calibration("descriptive", "unresolvable", degenerate=True)
    row = dv.row_verdict(*_preds(), calibration=cal)
    assert row["verdict"] == "descriptive-only"
    assert "unresolvable" in row["reason"]
    assert row["recovery_gate"]["degenerate_parents_identical"] is True


def test_row_calibration_malformed_refuses():
    with pytest.raises(dv.InputError, match="recovery"):
        dv.row_verdict(*_preds(), calibration={"something": 1})
    with pytest.raises(dv.InputError, match="p1_p3_rows_are"):
        dv.row_verdict(*_preds(), calibration={
            "recovery": {"verdict": "recovered", "p1_p3_rows_are": "maybe"}})


def test_row_verdicts_stay_inside_the_vocabulary():
    seen = set()
    cals = [None, _calibration("descriptive", "not recovered under this trainer + budget"),
            _calibration("descriptive", "unresolvable", degenerate=True),
            _calibration("confirmatory", "recovered")]
    for st1 in dv.STATUSES:
        for st3 in dv.STATUSES + ("absent",):
            for pd in (False, True):
                for cal in cals:
                    p1, p2, p3 = _preds(p1=st1, p3=("pass" if st3 == "absent" else st3))
                    row = dv.row_verdict(p1, p2, None if st3 == "absent" else p3,
                                         probe_descriptive=pd, calibration=cal)
                    assert row["verdict"] in dv.VERDICTS
                    seen.add(row["verdict"])
    assert seen == set(dv.VERDICTS)


# ---------------------------------------------------------------------------
# evaluate(): the whole lane, diagnostics, banned words
# ---------------------------------------------------------------------------


def test_evaluate_confirmed_lane_with_diagnostics():
    obj, mix, ctl = make_probe_rows(400, seed=40)
    o, m, c = make_ce_rows(20, seed=40)
    res = dv.evaluate(obj, mix, ctl, obj_docs=o, mix_docs=m, ctl_docs=c,
                      label="V qwen 100k", n_boot_items=FAST_ITEMS,
                      n_boot_docs=FAST_DOCS,
                      calibration=_calibration("confirmatory", "recovered"))
    assert res["row"]["verdict"] == "confirmed"
    assert res["schema"] == dv.SCHEMA
    assert res["constants"]["P1_TOST_MARGIN_NATS"] == 0.015
    assert res["n_items_in_coverage"] == res["p1"]["n_items"] == res["p2"]["n_items"]
    d = res["diagnostics"]
    assert d["unregistered"] is True
    # m_obj tracks the per-item subtraction when the decomposition holds
    assert d["pearson_r_m_obj_vs_subtraction"] > 0.5
    assert d["mean_abs_m_obj_minus_m_mix"] == pytest.approx(0.05, abs=0.02)
    assert d["arithmetic_residual_inconsistency"]["8"]["ratio"] == 0.058
    # summary lines: one per prediction + the row
    lines = dv.summary_lines(res)
    assert len(lines) == 4
    assert lines[0].startswith("P1 ") and lines[-1].startswith("ROW V qwen 100k: confirmed")


def test_evaluate_falsified_lane_and_probe_only():
    obj, mix, ctl = make_probe_rows(400, seed=41, obj_keeps_register=True)
    res = dv.evaluate(obj, mix, ctl, n_boot_items=FAST_ITEMS)
    assert res["p3"] is None
    assert res["row"]["verdict"] == "falsified"
    assert res["row"]["basis"] == ["P1", "P2"]
    assert res["diagnostics"]["mean_abs_m_obj_minus_m_mix"] < 0.02


def test_evaluate_requires_all_three_ce_files_or_none():
    obj, mix, ctl = make_probe_rows(100, seed=42)
    o, m, c = make_ce_rows(10, seed=42)
    with pytest.raises(dv.InputError, match="all three ce files"):
        dv.evaluate(obj, mix, ctl, obj_docs=o, mix_docs=m, n_boot_items=50)


def test_banned_word_in_any_output_text_raises():
    obj, mix, ctl = make_probe_rows(100, seed=43)
    with pytest.raises(dv.BannedWordError):
        dv.evaluate(obj, mix, ctl, label="P_resid is Inexpressible here",
                    n_boot_items=50)
    with pytest.raises(dv.BannedWordError):
        dv.assert_no_banned_words('{"x": "INEXPRESSIBLE"}')
    dv.assert_no_banned_words('{"x": "not recovered under this trainer + budget"}')


# ---------------------------------------------------------------------------
# CLI end to end
# ---------------------------------------------------------------------------


PAIRS_SHA = "c" * 64   # the (synthetic) sha of the control held-out pool file


def _ce_json(path: Path, rows, *, pool: str = dv.P3_POOL,
             pairs_sha: str = PAIRS_SHA, scale: float = 1.0,
             cell: str = "x", object_sha: str = "d" * 64) -> Path:
    """The derived app's COMBINED ce_score file shape (analysis contract:
    aggregate + provenance + the per-doc records under ``docs``)."""
    n = sum(r["n_target_tokens"] for r in rows)
    a = sum(r["nll_active_sum"] for r in rows)
    b = sum(r["nll_noop_sum"] for r in rows)
    doc = {"ce_active": a / n, "ce_noop": b / n, "ratio": a / b,
           "gain": 1 - a / b, "n_docs": len(rows), "n_tokens": n,
           "pool": pool, "pairs_sha256": pairs_sha, "object_sha256": object_sha,
           "scale": scale, "cell": cell, "model": "fake/model",
           "revision": "r0", "identity_kind": "derived", "docs": rows}
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return path


def _write_manifest(pools: Path, doc_ids, *, pool_sha: str = PAIRS_SHA,
                    stem: str = dv.P3_POOL_MANIFEST_STEM) -> Path:
    """The REGISTERED pool manifest (build_ctl_heldout's shape: cell,
    pool_sha256, doc_ids_in_order, n_docs)."""
    pools.mkdir(parents=True, exist_ok=True)
    p = pools / f"pairs_{stem}.manifest.json"
    p.write_text(json.dumps({"cell": stem, "lane": "control_heldout",
                             "pool_sha256": pool_sha,
                             "doc_ids_in_order": list(doc_ids),
                             "n_docs": len(doc_ids)}, indent=1))
    return p


def _write_lane(tmp_path: Path, *, seed: int, keep_register: bool = False,
                ce: bool = True):
    obj, mix, ctl = make_probe_rows(400, seed=seed,
                                    obj_keeps_register=keep_register)
    paths = {}
    for k, rows in (("object", obj), ("mix", mix), ("ctl", ctl)):
        p = tmp_path / f"{k}_probe.jsonl"
        _write_jsonl(p, rows)
        paths[f"--{k}-probe"] = str(p)
    if ce:
        o, m, c = make_ce_rows(20, seed=seed)
        for k, rows in (("object", o), ("mix", m), ("ctl", c)):
            p = tmp_path / f"{k}__control-heldout__scale1.json"
            _ce_json(p, rows, cell=k, object_sha=(k[0] * 64))
            paths[f"--{k}-ce"] = str(p)
        _write_manifest(tmp_path / "pools", [r["doc_id"] for r in o])
        paths["--pools-dir"] = str(tmp_path / "pools")
    return paths


def _argv(paths: dict, out: Path, *extra):
    argv = []
    for k, v in paths.items():
        argv += [k, v]
    argv += ["--n-boot-items", str(FAST_ITEMS), "--n-boot-docs", str(FAST_DOCS),
             "--out", str(out), *extra]
    return argv


def test_cli_end_to_end_writes_verdict_json(tmp_path, capsys):
    paths = _write_lane(tmp_path, seed=50)
    out = tmp_path / "verdict.json"
    rc = dv.main(_argv(paths, out, "--label", "LoRA-sub qwen 100k", "--seed", "3"))
    assert rc == 0
    res = json.loads(out.read_text())
    assert res["label"] == "LoRA-sub qwen 100k"
    assert res["row"]["verdict"] == "confirmed"
    assert res["run"] == {"n_boot_items": FAST_ITEMS, "n_boot_docs": FAST_DOCS,
                          "seed": 3, "bootstrap": res["run"]["bootstrap"]}
    for k in ("object_probe", "mix_probe", "ctl_probe", "object_ce", "mix_ce",
              "ctl_ce"):
        assert len(res["inputs"][k]["sha256"]) == 64
    assert res["inputs"]["object_probe"]["n_rows"] == 400
    # the P3 inputs were PLACED on the registered pool: the combined files'
    # pool / pairs sha / object sha recorded, the manifest they were
    # validated against named
    for k in ("object_ce", "mix_ce", "ctl_ce"):
        assert res["inputs"][k]["pool"] == "control-heldout"
        assert res["inputs"][k]["pairs_sha256"] == PAIRS_SHA
        assert res["inputs"][k]["n_rows"] == 20
    assert res["inputs"]["object_ce"]["object_sha256"] == "o" * 64
    assert res["inputs"]["pool_manifest"]["stem"] == "constv2ctl-heldout"
    assert res["inputs"]["pool_manifest"]["n_docs"] == 20
    assert res["p3_pool"]["pool"] == "control-heldout"
    assert res["p3_pool"]["pairs_sha256"] == PAIRS_SHA
    assert res["p3_pool"]["object_scale"] == 1.0
    assert res["p3_pool"]["cells"] == {"object_ce": "object", "mix_ce": "mix",
                                       "ctl_ce": "ctl"}
    printed = capsys.readouterr().out
    assert "P1 " in printed and "P2 " in printed and "P3 " in printed
    assert "ROW LoRA-sub qwen 100k: confirmed" in printed
    assert "inexpressible" not in json.dumps(res).lower()


def test_cli_falsified_is_still_exit_zero(tmp_path):
    paths = _write_lane(tmp_path, seed=51, keep_register=True, ce=False)
    out = tmp_path / "v.json"
    assert dv.main(_argv(paths, out)) == 0
    res = json.loads(out.read_text())
    assert res["row"]["verdict"] == "falsified" and res["p3"] is None
    assert res["p3_pool"] is None and "pool_manifest" not in res["inputs"]


def test_cli_p3_inputs_are_placed_on_the_registered_pool(tmp_path, capsys):
    """The three P3 inputs are the derived app's combined ce_score files
    and must agree on pool == control-heldout and one pairs_sha256,
    validated against the REGISTERED pool manifest (never seeded from a
    score file). Every violation is exit 2, naming the problem; a bare
    per-doc JSONL is refused outright."""
    paths = _write_lane(tmp_path, seed=55)
    out = tmp_path / "v.json"
    o, m, c = make_ce_rows(20, seed=55)
    assert dv.main(_argv(paths, out)) == 0  # the valid lane

    def refused(argv, needle):
        assert dv.main(argv) == 2
        err = capsys.readouterr().err
        assert needle in err, err

    # (a) a file scored on another pool
    bad = dict(paths, **{"--object-ce": str(_ce_json(
        tmp_path / "o_wrong_pool.json", o, pool="constv2ctl-100k"))})
    refused(_argv(bad, out), "defined on 'control-heldout'")
    # (b) the three files disagree on pairs_sha256
    bad = dict(paths, **{"--mix-ce": str(_ce_json(
        tmp_path / "m_other_sha.json", m, pairs_sha="e" * 64))})
    refused(_argv(bad, out), "different pairs_sha256")
    # (c) all three agree but on a sha the registered manifest does not carry
    other = tmp_path / "other"
    other.mkdir()
    alt = {}
    for k, rows in (("object", o), ("mix", m), ("ctl", c)):
        alt[f"--{k}-ce"] = str(_ce_json(other / f"{k}.json", rows,
                                        pairs_sha="e" * 64))
    refused(_argv(dict(paths, **alt), out), "registered manifest")
    # (d) the registered manifest is absent -> PENDING, never seeded
    refused(_argv(dict(paths, **{"--pools-dir": str(tmp_path / "nopools")}),
                  out), "PENDING")
    # (e) a malformed manifest
    badpools = tmp_path / "badpools"
    badpools.mkdir()
    (badpools / f"pairs_{dv.P3_POOL_MANIFEST_STEM}.manifest.json").write_text(
        json.dumps({"cell": "x"}))
    refused(_argv(dict(paths, **{"--pools-dir": str(badpools)}), out),
            "malformed")
    # (f) a file covering 19 of the 20 registered docs is refused, not averaged
    bad = dict(paths, **{"--ctl-ce": str(_ce_json(tmp_path / "c_19.json",
                                                   c[:-1]))})
    refused(_argv(bad, out), "doc set differs")
    # (g) a parent scored at 2x is not the lane's estimand
    bad = dict(paths, **{"--mix-ce": str(_ce_json(tmp_path / "m_x2.json", m,
                                                   scale=2.0))})
    refused(_argv(bad, out), "scale 1.0")
    # (h) a bare per-doc JSONL carries no pool identity
    jl = tmp_path / "o.jsonl"
    _write_jsonl(jl, o)
    refused(_argv(dict(paths, **{"--object-ce": str(jl)}), out),
            "combined JSON")
    assert not out.exists() or json.loads(out.read_text())["p3_pool"]["pool"] \
        == "control-heldout"
    # the object itself MAY be a 2x row (recorded, not refused)
    ok = dict(paths, **{"--object-ce": str(_ce_json(tmp_path / "o_x2.json", o,
                                                     scale=2.0))})
    out2 = tmp_path / "v2.json"
    assert dv.main(_argv(ok, out2)) == 0
    assert json.loads(out2.read_text())["p3_pool"]["object_scale"] == 2.0


def test_cli_probe_descriptive_and_calibration(tmp_path):
    paths = _write_lane(tmp_path, seed=52)
    out = tmp_path / "v.json"
    assert dv.main(_argv(paths, out, "--probe-descriptive")) == 0
    res = json.loads(out.read_text())
    assert res["row"]["roles"]["P1"] == "descriptive"
    assert res["row"]["basis"] == ["P3"]
    assert res["row"]["verdict"] == "confirmed"
    cal = tmp_path / "calibration.json"
    cal.write_text(json.dumps(_calibration(
        "descriptive", "not recovered under this trainer + budget")))
    out2 = tmp_path / "v2.json"
    assert dv.main(_argv(paths, out2, "--calibration", str(cal))) == 0
    res2 = json.loads(out2.read_text())
    assert res2["row"]["verdict"] == "not recovered under this trainer + budget"
    assert "planted sentence A" in res2["row"]["reason"]
    assert res2["diagnostics"]["arithmetic_residual_inconsistency"]["8"]["ratio"] == 0.058


def test_cli_input_problems_exit_2(tmp_path, capsys):
    paths = _write_lane(tmp_path, seed=53)
    out = tmp_path / "v.json"
    # partial ce set
    partial = {k: v for k, v in paths.items() if k != "--ctl-ce"}
    assert dv.main(_argv(partial, out)) == 2
    assert "given together" in capsys.readouterr().err
    # id mismatch: drop a row from the mix probe file
    mix_rows = [json.loads(l) for l in Path(paths["--mix-probe"]).read_text().splitlines()]
    _write_jsonl(Path(paths["--mix-probe"]), mix_rows[:-1])
    assert dv.main(_argv(paths, out)) == 2
    assert "id sets differ" in capsys.readouterr().err
    assert not out.exists()
    # a missing ce file
    badce = dict(paths, **{"--object-ce": str(tmp_path / "nope.json")})
    assert dv.main(_argv(badce, out)) == 2
    # missing file
    bad = dict(paths, **{"--object-probe": str(tmp_path / "nope.jsonl")})
    assert dv.main(_argv(bad, out)) == 2
    # a banned word in the label is not an input problem: it raises
    sub = tmp_path / "b"
    sub.mkdir()
    paths = _write_lane(sub, seed=54)
    with pytest.raises(dv.BannedWordError):
        dv.main(_argv(paths, out, "--label", "inexpressible"))
    assert not out.exists()
