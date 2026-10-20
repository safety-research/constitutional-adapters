"""scripts/sdfbase/recon_clip.py — the reconnaissance's percentile -> clip
rule and the per-trial clip-activation rate.

CPU-only, pure python. Synthetic bench records, plus a fixture cut from the
REAL metrics.jsonl row schema of a CE-arm cell (invented numbers).
"""
from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "recon_clip", REPO / "scripts" / "sdfbase" / "recon_clip.py")
rc = importlib.util.module_from_spec(_spec)
sys.modules["recon_clip"] = rc
_spec.loader.exec_module(rc)

FIXTURE = REPO / "tests/fixtures/sdfbase/metrics_lora_r4_ce_synthetic.jsonl"
ARM = "psr_ce_all"


# --------------------------------------------------------------------------
# synthetic bench records
# --------------------------------------------------------------------------

def _record(*, arm=ARM, status="ok", steps=150, lr=4.5e-4, clip=1e6,
            warmup=0.05, norms=None, losses=None, steps_run=None, error="",
            stderr_tail="", seed=0, drop_series=False):
    """A tune_arm.py bench record of the shape the harness writes."""
    rng = random.Random(seed)
    n = steps if steps_run is None else steps_run
    if norms is None:
        norms = [math.exp(rng.gauss(math.log(0.3), 0.3)) for _ in range(n)]
    if losses is None:
        losses = [2.5 - 0.8 * i / max(n, 1) + rng.gauss(0, 0.02)
                  for i in range(n)]
    rec = {"status": status, "error": error, "arm": arm, "lr": lr,
           "max_grad_norm": clip, "warmup_frac": warmup,
           "steps_requested": steps, "steps_run": n, "steps_completed": n,
           "si_label": "constv2-mix-full", "effective_batch": 64,
           "pool_sha256": "0" * 16, "model": "m"}
    if stderr_tail:
        rec["stderr_tail"] = stderr_tail
    if not drop_series:
        rec["grad_norm_preclip"] = {"steps": list(range(1, n + 1)),
                                    "values": list(norms)}
        rec["loss"] = {"steps": list(range(1, n + 1)), "values": list(losses)}
    return rec


# --------------------------------------------------------------------------
# quantiles + the ceiling
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 7, 50, 145])
@pytest.mark.parametrize("q", [0.0, 0.5, 0.9, 0.95, 0.99, 1.0])
def test_quantile_matches_numpy_linear(n, q):
    rng = random.Random(n * 7 + int(q * 100))
    xs = [rng.lognormvariate(0, 1) for _ in range(n)]
    assert math.isclose(rc.quantile(xs, q), float(np.quantile(xs, q)),
                        rel_tol=1e-12, abs_tol=1e-12)


def test_quantile_refusals():
    with pytest.raises(ValueError):
        rc.quantile([], 0.5)
    with pytest.raises(ValueError):
        rc.quantile([1.0], 1.5)


def test_quantile_summary_shape_and_empty():
    s = rc.quantile_summary([3.0, 1.0, 2.0])
    assert set(s) == {"n", "min", "p50", "p90", "p95", "p99", "max"}
    assert s["n"] == 3 and s["min"] == 1.0 and s["max"] == 3.0
    assert s["p50"] == 2.0
    e = rc.quantile_summary([])
    assert e["n"] == 0 and all(e[k] is None for k in e if k != "n")


@pytest.mark.parametrize("x, want", [
    (0.85, 0.85), (0.851, 0.86), (0.8501, 0.86), (2.049, 2.1), (2.0, 2.0),
    (123.4, 130.0), (0.09501, 0.096), (0.27, 0.27), (0.2701, 0.28),
    (1e6, 1e6), (9.99, 10.0), (0.001234, 0.0013), (7.0, 7.0),
])
def test_ceil2sf(x, want):
    assert rc.ceil2sf(x) == want


def test_ceil2sf_refuses_nonpositive_and_nonfinite():
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            rc.ceil2sf(bad)


# --------------------------------------------------------------------------
# series from metrics-shaped rows
# --------------------------------------------------------------------------

def test_series_from_rows_sorts_skips_and_keeps_nan():
    rows = [{"step": 3, "a/grad_norm": 0.3}, {"step": 1, "a/grad_norm": 0.1},
            {"step": 2, "probe/x": 1.0}, {"step": 2, "a/grad_norm": float("nan")},
            {"step": 4, "a/grad_norm": True}, {"a/grad_norm": 9.0},
            {"step": 5, "a/grad_norm": "0.5"}]
    steps, vals = rc.series_from_rows(rows, "a/grad_norm")
    assert steps == [1, 2, 3]
    assert vals[0] == 0.1 and math.isnan(vals[1]) and vals[2] == 0.3


def test_load_metrics_rows_tolerates_only_a_torn_last_line(tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text('{"step": 1, "x": 1}\n\n{"step": 2, "x": 2}\n{"step": 3, "x')
    assert [r["step"] for r in rc.load_metrics_rows(p)] == [1, 2]
    p.write_text('{"step": 1, "x": 1}\n{"step": 2, "x\n{"step": 3, "x": 3}\n')
    with pytest.raises(ValueError, match="corrupt"):
        rc.load_metrics_rows(p)


# --------------------------------------------------------------------------
# eligibility
# --------------------------------------------------------------------------

def test_classify_stable_ok_record():
    v = rc.classify_run(_record())
    assert v["class"] == "stable" and v["eligible"]
    assert v["reasons"] == []
    w = v["windows"]
    assert w["grad_norm_early"]["steps"][0] == rc.BENCH_WARMUP_STEPS + 1
    assert w["grad_norm_late"]["steps"][1] == 150
    # the REGISTERED step-index windows (the 150-step rule):
    # grad norm steps 6-50 vs 101-150; loss the first 25 post-warmup steps
    # (6-30) vs the last 25 (126-150) — never fractions of the run
    assert w["grad_norm_early"]["steps"] == [6, 50]
    assert w["grad_norm_late"]["steps"] == [101, 150]
    assert w["loss_early"]["steps"] == [6, 30]
    assert w["loss_late"]["steps"] == [126, 150]
    assert w["recon_steps"] == rc.RECON_STEPS == 150


def test_stability_windows_are_the_registered_step_indices():
    assert rc.RECON_STEPS == 150
    assert rc.stability_windows() == {
        "grad_norm_early": (6, 50), "grad_norm_late": (101, 150),
        "loss_early": (6, 30), "loss_late": (126, 150)}
    # the windows are step indices anchored to the budget, not fractions
    # of whatever a record happens to hold — and the certified interval
    # they lie in is the clip quantile's whole domain
    v = rc.classify_run(_record(steps=150))
    assert v["class"] == "stable"
    assert v["windows"]["grad_norm_late"]["steps"] == [101, 150]
    assert v["windows"]["clip_interval"] == [6, 150] == list(rc.clip_interval())
    # a longer run is NOT classified on the same windows and read stable
    # (its tail past step 150 never met the windows yet set the clip) — it is
    # over_budget, with the certified span in the reason
    v = rc.classify_run(_record(steps=200))
    assert v["class"] == "over_budget" and not v["eligible"]
    assert "certify steps 6-150" in v["reasons"][0]


def test_classify_status_diverged_and_failed_signatures():
    assert rc.classify_run(_record(status="diverged", error="nan"))["class"] \
        == "diverged"
    f = rc.classify_run(_record(status="failed", error="CUDA error: device",
                                stderr_tail="RuntimeError"))
    assert f["class"] == "failed" and not f["eligible"]
    d = rc.classify_run(_record(status="failed",
                                stderr_tail="...\nTrainingDiverged: non-finite"))
    assert d["class"] == "diverged"
    u = rc.classify_run(_record(status="weird"))
    assert u["class"] == "failed" and "unknown" in u["reasons"][0]


def test_classify_incomplete_when_steps_short_of_the_request_or_the_budget():
    v = rc.classify_run(_record(steps=150, steps_run=120))
    assert v["class"] == "incomplete" and "steps_run=120" in v["reasons"][0]
    # the reconnaissance budget is required INDEPENDENTLY of what the
    # record requested: an 11-step smoke record that ran every step it
    # asked for is still not a reconnaissance run and never sets the clip
    v = rc.classify_run(_record(steps=11, steps_run=11))
    assert v["class"] == "incomplete" and not v["eligible"]
    assert "reconnaissance budget" in v["reasons"][0]
    assert "steps_run=11" in v["reasons"][0]
    v = rc.classify_run(_record(steps=8, steps_run=8))
    assert v["class"] == "incomplete" and "reconnaissance budget" in v["reasons"][0]
    # a 150-step record whose series is torn inside a window is incomplete
    rec = _record()
    rec["grad_norm_preclip"]["steps"] = [s for s in range(1, 151) if s != 120]
    rec["grad_norm_preclip"]["values"] = rec["grad_norm_preclip"]["values"][:149]
    v = rc.classify_run(rec)
    assert v["class"] == "incomplete" and "120" in v["reasons"][0]


def test_registered_windows_catch_a_blowup_the_fractional_windows_missed():
    """At 150 steps the fractional windows compared steps
    6-53 / 103-150 instead of the registered 6-50 / 101-150 — a change of
    ELIGIBILITY, not reporting. Norms 30x the baseline over steps 101-125
    put the registered late median at 15.5x early (diverged); the
    fractional late window (103-150) holds 23 inflated and 25 baseline
    values, so its median is the baseline and the run read as stable —
    and the clip was then derived FROM it."""
    base, hot = 0.3, 9.0
    norms = [hot if 101 <= s <= 125 else base for s in range(1, 151)]
    losses = [2.5 - 0.01 * i for i in range(150)]
    import statistics
    assert statistics.median(norms[102:150]) == base       # fractional: stable
    assert statistics.median(norms[100:150]) > 10 * base   # registered: diverged
    v = rc.classify_run(_record(norms=norms, losses=losses))
    assert v["class"] == "diverged" and not v["eligible"]
    assert "blow-up" in v["reasons"][0]
    assert v["windows"]["grad_norm_late"]["steps"] == [101, 150]
    # ...and it never enters the clip
    with pytest.raises(ValueError, match="no STABLE recon run"):
        rc.clip_from_records({"hot": _record(norms=norms, losses=losses)},
                             arm=ARM)


def test_classify_blowups_are_diverged_not_stable():
    # (iii-a): late grad-norm median > 10x early -> diverged, loss fine
    norms = [0.3] * 100 + [4.0] * 50
    losses = [2.5 - 0.01 * i for i in range(150)]
    v = rc.classify_run(_record(norms=norms, losses=losses))
    assert v["class"] == "diverged" and "blow-up" in v["reasons"][0]
    # (iii-b): loss rising over the run -> diverged, norms flat
    v = rc.classify_run(_record(norms=[0.3] * 150,
                                losses=[2.0 + 0.01 * i for i in range(150)]))
    assert v["class"] == "diverged" and "loss not decreasing" in v["reasons"][0]
    # a nan in the loss series -> diverged
    losses = [2.5 - 0.01 * i for i in range(150)]
    losses[80] = float("nan")
    v = rc.classify_run(_record(losses=losses))
    assert v["class"] == "diverged" and "non-finite" in v["reasons"][0]
    # a 9x late/early ratio with a falling loss is still stable
    norms = [0.3] * 100 + [2.6] * 50
    v = rc.classify_run(_record(norms=norms,
                                losses=[2.5 - 0.01 * i for i in range(150)]))
    assert v["class"] == "stable"


def test_classify_refuses_missing_record_and_seriesless_ok():
    with pytest.raises(ValueError, match="MISSING record is not an outcome"):
        rc.classify_run(None)
    with pytest.raises(ValueError, match="no grad_norm_preclip"):
        rc.classify_run(_record(drop_series=True))


# --------------------------------------------------------------------------
# the clip
# --------------------------------------------------------------------------

def test_clip_is_ceil2sf_of_the_max_stable_p95_and_diverged_runs_are_diagnostic():
    a = _record(seed=1)
    b = _record(seed=2, norms=[0.5] * 150)          # p95 = 0.5 exactly
    # a DIVERGED run with 100x norms: must not touch the clip
    div = _record(status="diverged", error="nan", seed=3,
                  norms=[50.0] * 150)
    res = rc.clip_from_records({"r1": a, "r2": b, "r5": div}, arm=ARM)
    assert res["eligible_runs"] == ["r1", "r2"]
    assert set(res["ineligible_runs"]) == {"r5"}
    assert res["ineligible_runs"]["r5"]["class"] == "diverged"
    p95_a = rc.quantile(a["grad_norm_preclip"]["values"][rc.BENCH_WARMUP_STEPS:],
                        0.95)
    want_raw = max(p95_a, 0.5)
    assert res["clip_raw"] == want_raw
    assert res["clip"] == rc.ceil2sf(want_raw)
    assert res["argmax_run"] == ("r1" if p95_a > 0.5 else "r2")
    assert res["runs"]["r5"]["diagnostic_only"] is True
    assert res["runs"]["r5"]["quantiles_after_warmup"]["p50"] == 50.0
    assert "p_rule" not in res["runs"]["r5"]
    # the pooled summary is over the STABLE post-warmup steps only
    assert res["pooled"]["n"] == 2 * (150 - rc.BENCH_WARMUP_STEPS)
    assert res["pooled"]["max"] < 50.0
    assert set(res["pooled"]) == {"n", "min", "p50", "p90", "p95", "p99", "max"}
    assert res["percentile"] == 0.95 and res["rule"] == rc.CLIP_RULE


def test_clip_refuses_without_a_stable_run_or_with_a_missing_record():
    div = _record(status="diverged", error="nan", norms=[50.0] * 150)
    with pytest.raises(ValueError, match="no STABLE recon run"):
        rc.clip_from_records({"r5": div}, arm=ARM)
    with pytest.raises(ValueError, match="MISSING record"):
        rc.clip_from_records({"r1": _record(), "r2": None}, arm=ARM)
    with pytest.raises(ValueError, match="no bench records"):
        rc.clip_from_records({}, arm=ARM)


def test_clip_refuses_a_foreign_arm_and_a_binding_recon_clip():
    with pytest.raises(ValueError, match="one clip per arm"):
        rc.clip_from_records({"r1": _record(arm="lora_r4_ce")}, arm=ARM)
    # a stable run whose OWN clip bound: not the unclipped distribution
    bound = _record(clip=0.25, norms=[0.3] * 150)
    with pytest.raises(ValueError, match="bound on"):
        rc.clip_from_records({"r1": bound}, arm=ARM)


def test_clip_percentile_is_a_parameter_and_deterministic():
    recs = {"r1": _record(seed=11), "r2": _record(seed=12)}
    a = rc.clip_from_records(recs, arm=ARM)
    b = rc.clip_from_records(dict(reversed(list(recs.items()))), arm=ARM)
    assert a == b
    p99 = rc.clip_from_records(recs, arm=ARM, percentile=0.99)
    assert p99["clip_raw"] >= a["clip_raw"]


# --------------------------------------------------------------------------
# the certified interval: exactly the budget, nothing outside it
# --------------------------------------------------------------------------

def _over_budget_record(n=200, tail_norm=1000.0, **kw):
    """The over-budget counterexample: the bench CLI's default 200 steps, norms 0.3
    through the budget and ``tail_norm`` past it, a loss that falls through
    the budget then climbs sharply (finite throughout), the non-binding 1e6
    clip, status ok, every requested step run."""
    norms = [0.3 if s <= rc.RECON_STEPS else tail_norm for s in range(1, n + 1)]
    losses = [2.5 - 0.005 * s if s <= rc.RECON_STEPS
              else 1.75 + 0.2 * (s - rc.RECON_STEPS) for s in range(1, n + 1)]
    return _record(steps=n, norms=norms, losses=losses, **kw)


def test_a_record_past_the_budget_is_over_budget_never_stable():
    """The registered-window classifier certifies steps 6-150
    only, while the clip's p95 ran over EVERY post-warmup step — so a
    200-step record (the bench CLI's default) with norms 0.3 through step
    150 and 1000 past it read STABLE and set clip = 1000, a tail nobody
    examined. Fail-closed: only a record of exactly RECON_STEPS can set the
    clip; a longer one is the named verdict over_budget, diagnostic only."""
    rec = _over_budget_record()
    assert rec["steps_requested"] == rec["steps_run"] == 200
    v = rc.classify_run(rec)
    assert v["class"] == "over_budget" and not v["eligible"]
    assert "over_budget" in rc.RUN_CLASSES
    assert "steps_run=200" in v["reasons"][0]
    assert "budget 150" in v["reasons"][0] and "151-200" in v["reasons"][0]
    # alone it sets nothing
    with pytest.raises(ValueError, match="no STABLE recon run"):
        rc.clip_from_records({"long": rec}, arm=ARM)
    # beside a stable budget-length run it is diagnostic only: the clip is
    # the stable run's p95 and 1000 appears nowhere but the diagnostic table
    ok = _record(norms=[0.3] * 150)
    res = rc.clip_from_records({"long": rec, "ok": ok}, arm=ARM)
    assert res["eligible_runs"] == ["ok"]
    assert res["ineligible_runs"]["long"]["class"] == "over_budget"
    assert res["clip"] == 0.3 and res["clip_raw"] == 0.3
    assert res["runs"]["long"]["diagnostic_only"] is True
    assert res["runs"]["long"]["quantiles_after_warmup"]["p95"] == 1000.0
    assert "p_rule" not in res["runs"]["long"]
    assert res["pooled"]["max"] == 0.3 and res["pooled"]["n"] == 145


def test_only_exactly_the_budget_is_eligible():
    assert rc.classify_run(_record(steps=150))["class"] == "stable"
    v149 = rc.classify_run(_record(steps=149))
    assert v149["class"] == "incomplete"
    assert "reconnaissance budget" in v149["reasons"][0]
    v151 = rc.classify_run(_record(steps=151))
    assert v151["class"] == "over_budget" and "steps_run=151" in v151["reasons"][0]
    # a 200-step record whose tail happens to be benign is refused too: the
    # rule is on the step count, not on what the unexamined tail holds
    v200 = rc.classify_run(_record(steps=200))
    assert v200["class"] == "over_budget" and not v200["eligible"]


def test_a_budget_record_yields_its_p95_over_the_certified_interval():
    rec = _record(seed=5)
    res = rc.clip_from_records({"r": rec}, arm=ARM)
    vals = rec["grad_norm_preclip"]["values"]
    cert = vals[rc.BENCH_WARMUP_STEPS:rc.RECON_STEPS]      # steps 6..150
    assert len(cert) == 145
    assert res["clip_interval"] == [6, 150]
    assert res["runs"]["r"]["p_rule"] == rc.quantile(cert, 0.95) == res["clip_raw"]
    assert res["runs"]["r"]["p_rule_interval"] == [6, 150]
    assert res["runs"]["r"]["p_rule_n"] == 145
    assert res["pooled"]["n"] == 145
    assert res["clip"] == rc.ceil2sf(res["clip_raw"])


def test_clip_interval_values_is_bounded_structurally_not_by_the_refusal():
    """The bound must hold whatever series reaches the quantile, not because
    a refusal upstream happened to keep long records out."""
    assert rc.clip_interval() == (6, 150)
    steps = list(range(1, 201))
    vals = [0.3 if s <= 150 else 1000.0 for s in steps]
    sel = rc.clip_interval_values(steps, vals)
    assert len(sel) == 145 and max(sel) == 0.3
    # selected by STEP INDEX, never by position
    sel2 = rc.clip_interval_values(list(reversed(steps)), list(reversed(vals)))
    assert sorted(sel2) == sorted(sel)
    # warm-up dropped at the low end as well
    assert rc.clip_interval_values([1, 5, 6, 150, 151], [9, 9, 1, 2, 9]) == [1.0, 2.0]


def test_a_series_past_steps_run_is_an_inconsistent_record_and_refused():
    """A record that CLAIMS the budget while its series runs past it is not
    a budget record with a benign label: it is refused, never read as
    stable through the claim."""
    rec = _over_budget_record()
    rec["steps_requested"] = rec["steps_run"] = rec["steps_completed"] = 150
    with pytest.raises(ValueError, match="steps_run=150"):
        rc.classify_run(rec)
    with pytest.raises(ValueError, match="steps_run=150"):
        rc.clip_from_records({"r": rec}, arm=ARM)


def test_a_budget_record_torn_between_the_windows_is_incomplete():
    """The certified interval is one thing: the norm series must be complete
    over all of 6-150 (not only inside the two norm windows) for the
    quantile over it to be the run's distribution."""
    rec = _record()
    rec["grad_norm_preclip"]["steps"] = [s for s in range(1, 151) if s != 75]
    rec["grad_norm_preclip"]["values"] = rec["grad_norm_preclip"]["values"][:149]
    v = rc.classify_run(rec)
    assert v["class"] == "incomplete" and "75" in v["reasons"][0]
    with pytest.raises(ValueError, match="no STABLE recon run"):
        rc.clip_from_records({"torn": rec}, arm=ARM)


def test_cli_marks_an_over_budget_record_diagnostic_only(tmp_path, capsys):
    (tmp_path / "s150.json").write_text(json.dumps(_record(norms=[0.3] * 150)))
    (tmp_path / "s200.json").write_text(json.dumps(_over_budget_record()))
    out = tmp_path / "clip.json"
    code = rc.main([str(tmp_path / "s150.json"), str(tmp_path / "s200.json"),
                    "--arm", ARM, "--out", str(out)])
    assert code == 0
    text = capsys.readouterr().out
    assert "OVER_BUDGET (diagnostic only)" in text
    assert f"CLIP {ARM} = 0.3" in text
    res = json.loads(out.read_text())
    assert res["eligible_runs"] == ["s150"]
    assert res["ineligible_runs"]["s200"]["class"] == "over_budget"
    assert res["clip_interval"] == [6, 150]
    # the over-budget record alone: refused at the CLI, no clip printed
    with pytest.raises(SystemExit, match="REFUSED"):
        rc.main([str(tmp_path / "s200.json"), "--arm", ARM])


# --------------------------------------------------------------------------
# the fixture in the REAL metrics.jsonl shape
# --------------------------------------------------------------------------

def test_fixture_has_the_real_row_schema():
    rows = rc.load_metrics_rows(FIXTURE)
    arm = "lora_r4_ce"
    step_rows = [r for r in rows if f"{arm}/grad_norm" in r]
    assert len(step_rows) == 72
    for r in step_rows:
        assert set(r) == {"step", f"{arm}/ce_loss", f"{arm}/ce", f"{arm}/lr",
                          f"{arm}/loss_tokens", f"{arm}/real_tokens",
                          f"{arm}/padded_tokens", f"{arm}/sec_per_step",
                          "epoch", f"{arm}/grad_norm"}
    assert rows[0] == {"step": 0, f"{arm}/step0_loss": 2.9137,
                       f"{arm}/noop_floor": 2.9137}
    assert any("probe/heldout_loss" in r for r in rows)


def test_clip_activation_rate_on_the_fixture_is_the_exact_share():
    rows = rc.load_metrics_rows(FIXTURE)
    key = rc.gradnorm_key("lora_r4_ce")
    assert key == "lora_r4_ce/grad_norm"
    vals = [r[key] for r in rows if key in r]
    for clip in (0.2, 0.4135, 1.0, 5.0, 100.0):
        rate, n = rc.clip_activation_rate(rows, key, clip)
        assert n == 72
        assert rate == sum(v > clip for v in vals) / 72
    assert rc.clip_activation_rate(rows, key, 100.0) == (0.0, 72)
    assert rc.clip_activation_rate(rows, "nope/grad_norm", 1.0) == (None, 0)
    # the file-level helper reads the same rows
    assert rc.clip_activation_from_metrics(FIXTURE, arm="lora_r4_ce",
                                           clip=1.0) == \
        rc.clip_activation_rate(rows, key, 1.0)
    # the lockstep arms' key is the GradNormBuffer drain's
    assert rc.gradnorm_key("psr_kl_all") == "train/grad_norm_preclip"


def test_bench_record_built_from_the_fixture_rows_yields_a_clip():
    """The clip helper over a record whose series came from REAL-shaped
    metrics rows (what bench does with its own recs): eligibility from the
    series, p95 over steps > 5, the ceiling."""
    rows = rc.load_metrics_rows(FIXTURE)
    arm = "lora_r4_ce"
    steps, norms = rc.series_from_rows(rows, rc.gradnorm_key(arm))
    _, losses = rc.series_from_rows(rows, rc.loss_key(arm))
    assert len(steps) == 72
    # the 72-step fixture is SHORT of the reconnaissance budget: a real
    # record of that length is incomplete for clip-setting, whatever it
    # requested — the registered verdict, never a clip
    short = _record(arm=arm, steps=72, norms=norms, losses=losses)
    assert rc.classify_run(short)["class"] == "incomplete"
    with pytest.raises(ValueError, match="no STABLE recon run"):
        rc.clip_from_records({"fixture": short}, arm=arm)
    # the same real-shaped norms tiled out to the budget feed the rule
    norms150 = (norms * 3)[:rc.RECON_STEPS]
    rec = _record(arm=arm, steps=rc.RECON_STEPS, norms=norms150)
    res = rc.clip_from_records({"fixture": rec}, arm=arm)
    post = norms150[rc.BENCH_WARMUP_STEPS:]
    assert res["eligible_runs"] == ["fixture"]
    assert res["clip_raw"] == rc.quantile(post, 0.95)
    assert res["clip"] == rc.ceil2sf(res["clip_raw"])
    assert res["pooled"]["n"] == len(post) == 145
    assert res["pooled"]["p50"] == rc.quantile(post, 0.5)
    # the fixture's spikes sit above the clip, the body below: a spike guard
    rate, n = rc.clip_activation_rate(rows, rc.gradnorm_key(arm), res["clip"])
    assert 0.0 < rate <= 0.06 and n == 72


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_prints_the_clip_and_writes_the_record(tmp_path, capsys):
    for i, rec in enumerate([_record(seed=1), _record(seed=2),
                             _record(status="diverged", error="nan", seed=3,
                                     norms=[50.0] * 150)]):
        (tmp_path / f"recon_{i + 1}.json").write_text(json.dumps(rec))
    out = tmp_path / "clip.json"
    code = rc.main([str(tmp_path / f"recon_{i}.json") for i in (1, 2, 3)]
                   + ["--arm", ARM, "--out", str(out)])
    assert code == 0
    text = capsys.readouterr().out
    res = json.loads(out.read_text())
    assert f"CLIP {ARM} = {res['clip']:g}" in text
    assert "DIVERGED (diagnostic only)" in text
    assert res["eligible_runs"] == ["recon_1", "recon_2"]
    with pytest.raises(SystemExit, match="REFUSED"):
        rc.main([str(tmp_path / "recon_3.json"), "--arm", ARM])
    with pytest.raises(SystemExit, match="no such record"):
        rc.main([str(tmp_path / "recon_9.json"), "--arm", ARM])


def test_cli_keeps_same_named_records_under_two_model_directories(tmp_path):
    """The bench's default filename encodes the SETTINGS, not the
    model (the model is the parent directory), so a base and an instruct
    reconnaissance run at identical settings share a basename. Keyed by
    stem, the later input silently replaced the earlier one and the clip
    came out as whichever survived — input order decided which evidence
    vanished. Both must count, whatever the order, and the clip is the
    max; the same file given twice is a duplicate identity and is refused
    before anything is computed."""
    name = "lr4.5e-4_w0.05.json"
    base, inst = tmp_path / "llama8b-base", tmp_path / "llama8b-it"
    base.mkdir()
    inst.mkdir()
    (base / name).write_text(json.dumps(_record(norms=[2.0] * 150, seed=1)))
    (inst / name).write_text(json.dumps(_record(norms=[0.3] * 150, seed=2)))
    out = tmp_path / "clip.json"
    for order in ((base, inst), (inst, base)):
        code = rc.main([str(d / name) for d in order]
                       + ["--arm", ARM, "--out", str(out)])
        assert code == 0
        res = json.loads(out.read_text())
        assert res["clip"] == 2.0, (order, res["clip"])
        assert sorted(res["eligible_runs"]) == [
            f"llama8b-base/{Path(name).stem}", f"llama8b-it/{Path(name).stem}"]
        assert res["argmax_run"] == f"llama8b-base/{Path(name).stem}"
        assert len(res["runs"]) == 2
        # every run id names the record it was read from
        assert set(res["record_paths"]) == set(res["runs"])
        assert {Path(p).resolve() for p in res["record_paths"].values()} == {
            (base / name).resolve(), (inst / name).resolve()}
    # one directory keeps the plain stems (the legacy run id)
    single = tmp_path / "single"
    single.mkdir()
    (single / "recon_1.json").write_text(json.dumps(_record(seed=1)))
    (single / "recon_2.json").write_text(json.dumps(_record(seed=2)))
    rc.main([str(single / "recon_1.json"), str(single / "recon_2.json"),
             "--arm", ARM, "--out", str(out)])
    assert json.loads(out.read_text())["eligible_runs"] == ["recon_1", "recon_2"]
    # the same record supplied twice is refused loudly, before computing
    with pytest.raises(SystemExit, match="duplicate"):
        rc.main([str(base / name), str(base / name), "--arm", ARM])
    with pytest.raises(SystemExit, match="duplicate"):
        rc.main([str(base / name), str(tmp_path / "llama8b-base" / ".." /
                                         "llama8b-base" / name),
                 "--arm", ARM])


def test_clip_activation_from_metrics_reads_only_the_summarys_attempt(tmp_path):
    """The reader half: a trial's metrics.jsonl is opened in APPEND
    mode across re-runs, so rows of an earlier attempt sit before the
    current trajectory's. With the summary's attempt stamp the reader
    selects exactly that segment; a file whose rows carry none of the
    stamp is not this summary's record and is refused, and a summary
    without a stamp (a design analysed before the column) reads every row
    only over a file that carries none — over stamped rows the pairing is
    ambiguous and refused (the next test)."""
    key = rc.gradnorm_key(ARM)
    rows = ([{"step": i, key: 0.1, rc.METRICS_ATTEMPT_KEY: "old"}
             for i in range(1, 11)]
            + [{"step": i, key: (1.0 if i % 2 else 0.1),
                rc.METRICS_ATTEMPT_KEY: "new"} for i in range(1, 11)])
    p = tmp_path / "metrics.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert rc.clip_activation_from_metrics(p, arm=ARM, clip=0.5,
                                           attempt="new") == (0.5, 10)
    assert rc.clip_activation_from_metrics(p, arm=ARM, clip=0.5,
                                           attempt="old") == (0.0, 10)
    # no stamp asked for over a file whose rows carry stamps: NOT the
    # whole-file blend (0.25 over 20) — an unstamped summary cannot be the
    # record of stamped rows, so the pairing is refused
    with pytest.raises(ValueError, match="carries no attempt stamp"):
        rc.clip_activation_from_metrics(p, arm=ARM, clip=0.5)
    with pytest.raises(ValueError, match="attempt"):
        rc.clip_activation_from_metrics(p, arm=ARM, clip=0.5, attempt="zzz")
    # an empty file is no evidence, stamp or not
    (tmp_path / "empty.jsonl").write_text("")
    assert rc.clip_activation_from_metrics(tmp_path / "empty.jsonl", arm=ARM,
                                           clip=0.5, attempt="new") == (None, 0)


def test_unstamped_summary_over_stamped_rows_is_refused(tmp_path):
    """The legacy-to-stamped seam: an UNSTAMPED summary describes
    a whole file only when that file carries no stamp at all. The old
    reader assumed that; upgrade-and-retry breaks it: a pre-stamp trial
    re-run under the stamped writer after a fingerprint change, interrupted
    after it appended stamped rows but before it published its summary,
    leaves the old unstamped terminal summary in place over a file whose
    tail is a stamped retry's partial trajectory. Read whole-file, the
    predecessor's 40% activation blended with the retry's non-clipping
    rows read 20% and crossed from ineligible to promotable. The pairing
    is refused with the file named and the stamps listed; a wholly
    unstamped file still reads whole (the legacy path, unchanged); a
    stamped summary still selects only its own rows over the same mixed
    file; an empty file stays no evidence."""
    key = rc.gradnorm_key(ARM)
    legacy = [{"step": i, key: (1.0 if i <= 4 else 0.1)} for i in range(1, 11)]
    retry = [{"step": i, key: 0.1, rc.METRICS_ATTEMPT_KEY: "retry"}
             for i in range(1, 11)]

    def write(name, rows):
        q = tmp_path / name
        q.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return q

    # 1. a wholly unstamped file under an unstamped summary: every row
    p_old = write("legacy.jsonl", legacy)
    assert rc.rows_of_attempt(legacy, None, path=p_old) == legacy
    assert rc.clip_activation_from_metrics(p_old, arm=ARM, clip=0.5) == (0.4, 10)
    # 2. the same predecessor with a stamped retry appended: whole-file
    #    would read 0.2 (promotable); the reader refuses instead
    p_mix = write("mixed.jsonl", legacy + retry)
    assert rc.clip_activation_rate(legacy + retry, key, 0.5) == (0.2, 20)
    with pytest.raises(ValueError, match="carries no attempt stamp") as ei:
        rc.clip_activation_from_metrics(p_mix, arm=ARM, clip=0.5)
    msg = str(ei.value)
    assert str(p_mix) in msg and "'retry'" in msg and "10 of 20" in msg
    with pytest.raises(ValueError, match="carries no attempt stamp"):
        rc.rows_of_attempt(legacy + retry, None)
    # 3. the retry's own (stamped) summary reads only its rows — the
    #    predecessor's unstamped rows are not its record either
    assert rc.clip_activation_from_metrics(p_mix, arm=ARM, clip=0.5,
                                           attempt="retry") == (0.0, 10)
    assert rc.rows_of_attempt(legacy + retry, "retry") == retry
    # 4. a wholly stamped file under an unstamped summary is refused too
    p_new = write("stamped.jsonl", retry)
    with pytest.raises(ValueError, match="carries no attempt stamp"):
        rc.clip_activation_from_metrics(p_new, arm=ARM, clip=0.5)
    # 5. an empty file is still no evidence for an unstamped summary
    assert rc.clip_activation_from_metrics(write("empty.jsonl", []), arm=ARM,
                                           clip=0.5) == (None, 0)
    assert rc.rows_of_attempt([], None) == []


# --------------------------------------------------------------------------
# the tail, the collapse and the pre-training floor — from the REAL
# wave-1 reconnaissance series of the llama8b_base lane
# --------------------------------------------------------------------------

REAL = REPO / "tests/fixtures/sdfbase/recon_llama8b_wave1_series.json"


def _real_records(arm=None) -> dict:
    """The six wave-1 bench records rebuilt from their real series (the
    fixture holds exactly the fields the classifier reads)."""
    data = json.loads(REAL.read_text())["runs"]
    out = {}
    for rid, r in data.items():
        if arm is not None and r["arm"] != arm:
            continue
        out[rid] = {"status": r["status"], "error": "", "arm": r["arm"],
                    "lr": r["lr"], "max_grad_norm": r["max_grad_norm"],
                    "warmup_frac": r["warmup_frac"],
                    "steps_requested": r["steps_requested"],
                    "steps_run": r["steps_run"], "steps_completed": r["steps_run"],
                    "si_label": r["pool"], "effective_batch": r["effective_batch"],
                    "pool_sha256": "0" * 16, "model": r["model"],
                    "grad_norm_preclip": r["grad_norm_preclip"], "loss": r["loss"]}
    return out


def test_real_recon_5_is_heavy_tail_not_stable():
    """The PSR box-top run (lr 1.4e-2) has certified-interval pre-clip
    norms p50 3.46 / p95 7,391 / max 31,260 — a spiking, heavy-tailed run at
    a learning rate that is too high — and the median-based windows called it
    STABLE (late p50 1.33 is far BELOW early p50 1,629; the loss fell from 11
    to 7, finite). The rule then took its p95 and emitted clip 7,400. A run
    whose p95 sits more than TAIL_RATIO_MAX above its own median is the
    class heavy_tail, ineligible, diagnostic only — and every other thing
    wrong with it is a named reason too: the early window is an explosion
    followed by a collapse, and the run ends three times worse than it
    began."""
    rec = _real_records("psr_ce_all")["recon_5"]
    gn = rec["grad_norm_preclip"]
    cert = rc.clip_interval_values(gn["steps"], gn["values"])
    p50, p95 = rc.quantile(cert, 0.5), rc.quantile(cert, 0.95)
    assert 3.4 < p50 < 3.5 and 7300 < p95 < 7400 and p95 / p50 > 2000
    assert "heavy_tail" in rc.RUN_CLASSES
    v = rc.classify_run(rec)
    assert v["class"] == "heavy_tail" and not v["eligible"]
    text = "\n".join(v["reasons"])
    assert "heavy tail" in text and "p95" in text
    assert f"bound {rc.TAIL_RATIO_MAX:g}" in text
    assert "grad-norm collapse" in text
    assert "loss above pre-training" in text
    # the REGISTERED rule alone passed it — which is why the tail check exists
    assert not any("blow-up" in r or "loss not decreasing" in r for r in v["reasons"])
    tail = v["windows"]["tail"]
    assert math.isclose(tail["ratio"], p95 / p50) and tail["ratio_max"] == rc.TAIL_RATIO_MAX
    assert tail["interval"] == [6, 150]
    assert v["windows"]["grad_norm_early"]["p50"] > 1000 > v["windows"]["grad_norm_late"]["p50"]
    assert v["windows"]["loss_pretraining"]["median"] < 2.5      # 2.38: untrained
    assert v["windows"]["loss_late"]["median"] > 7.0                # ends 3x worse


def test_clip_over_the_real_psr_records_is_the_anchors_not_7400():
    """The three real PSR records together: the two anchors (lr 4.5e-4, p95
    0.52 / 0.50) set the clip and recon_5 is diagnostic only — clip 0.53, the
    provisional PSR clip, never 7,400."""
    res = rc.clip_from_records(_real_records("psr_ce_all"), arm="psr_ce_all")
    assert res["eligible_runs"] == ["recon_1", "recon_3"]
    assert res["ineligible_runs"]["recon_5"]["class"] == "heavy_tail"
    p95s = {r: res["runs"][r]["p_rule"] for r in ("recon_1", "recon_3")}
    assert 0.50 < p95s["recon_3"] < p95s["recon_1"] < 0.53
    assert res["clip_raw"] == p95s["recon_1"] and res["argmax_run"] == "recon_1"
    assert res["clip"] == rc.ceil2sf(res["clip_raw"]) == 0.53
    r5 = res["runs"]["recon_5"]
    assert r5["diagnostic_only"] is True and "p_rule" not in r5
    assert r5["quantiles_after_warmup"]["p95"] > 7000       # what became 7,400
    assert res["pooled"]["n"] == 2 * 145 and res["pooled"]["max"] < 2.0


def test_real_anchors_stay_stable_and_real_recon_6_stays_diverged():
    """The four anchors (p95/p50 1.2-2.0) are stable with no reason filed;
    the LoRA box top stays diverged by the REGISTERED loss rule (its first
    reason) with its 183x tail named after it; the LoRA clip stays 0.15."""
    allr = _real_records()
    for rid in ("recon_1", "recon_2", "recon_3", "recon_4"):
        v = rc.classify_run(allr[rid])
        assert v["class"] == "stable" and v["eligible"] and v["reasons"] == [], rid
        assert v["windows"]["tail"]["ratio"] < 2.1, rid
    v6 = rc.classify_run(allr["recon_6"])
    assert v6["class"] == "diverged" and not v6["eligible"]
    assert "loss not decreasing" in v6["reasons"][0]
    assert any("heavy tail" in r for r in v6["reasons"])
    assert v6["windows"]["tail"]["ratio"] > 100
    lora = rc.clip_from_records(_real_records("lora_r4_ce"), arm="lora_r4_ce")
    assert lora["eligible_runs"] == ["recon_2", "recon_4"]
    assert lora["ineligible_runs"]["recon_6"]["class"] == "diverged"
    assert lora["clip"] == 0.15 and lora["argmax_run"] == "recon_2"


def _spiky(ratio: float, base: float = 0.3) -> list[float]:
    """150 norms: the base everywhere but 8 spikes spread evenly over the
    certified interval, so the window medians stay at the base and ONLY the
    p95 moves. Over 145 certified values p95 (linear interpolation, position
    136.8) = base + 0.8 (spike - base), so spike = base + (base*ratio - base)
    / 0.8 gives p95 / p50 == ratio."""
    spike = base + (base * ratio - base) / 0.8
    spike_steps = set(range(10, 150, 18))            # 10, 28, ..., 136: 8 steps
    assert len(spike_steps) == 8
    return [spike if s in spike_steps else base for s in range(1, 151)]


def test_tail_bound_is_the_decade_a_9x_tail_is_stable_an_11x_tail_is_heavy_tail():
    """TAIL_RATIO_MAX is the same decade the blow-up rule uses. Every healthy
    run measured sits at or under 3.3 (the four anchors 1.2-2.0; the
    published GLM / nemotron / qwen cells 1.3-2.0 over steps 6-150, <= 3.3
    whole-run); the module's own registered 9x late/early step is 8.7; the
    two blown-up runs are 183 and 2,134."""
    assert rc.TAIL_RATIO_MAX == 10.0 == rc.BLOWUP_NORM_RATIO
    v9 = rc.classify_run(_record(norms=_spiky(9.0)))
    assert v9["class"] == "stable" and v9["reasons"] == []
    assert math.isclose(v9["windows"]["tail"]["ratio"], 9.0)
    v11 = rc.classify_run(_record(norms=_spiky(11.0)))
    assert v11["class"] == "heavy_tail" and not v11["eligible"]
    assert len(v11["reasons"]) == 1 and "heavy tail" in v11["reasons"][0]
    assert math.isclose(v11["windows"]["tail"]["ratio"], 11.0)
    # its window medians are exactly the base: the registered rule sees nothing
    assert v11["windows"]["grad_norm_early"]["p50"] == 0.3
    assert v11["windows"]["grad_norm_late"]["p50"] == 0.3
    # alone it sets nothing; beside a stable run it is diagnostic only
    with pytest.raises(ValueError, match="heavy-tail"):
        rc.clip_from_records({"s": _record(norms=_spiky(11.0))}, arm=ARM)
    res = rc.clip_from_records({"s": _record(norms=_spiky(11.0)),
                                "ok": _record(norms=[0.3] * 150)}, arm=ARM)
    assert res["eligible_runs"] == ["ok"] and res["clip"] == 0.3
    assert res["ineligible_runs"]["s"]["class"] == "heavy_tail"
    assert res["runs"]["s"]["diagnostic_only"] is True and "p_rule" not in res["runs"]["s"]


def test_a_collapse_after_an_explosion_is_diverged_the_window_bound_is_symmetric():
    """recon_5's early window is an explosion (p50 1,629) and its late window
    the collapse after it (p50 1.33): early / late 1,224, and the registered
    rule — which only bounds late / early — read that as calm. The bound
    holds in both directions at the same decade. A smooth 100x decay across
    the run trips it on its own (early p50 ~19x late) while its p95 / p50
    (~7) stays under the tail bound; a 9x fall, like a 9x rise, is stable."""
    decay = [3.0 * 0.97 ** (s - 6) if s >= 6 else 3.0 for s in range(1, 151)]
    v = rc.classify_run(_record(norms=decay))
    assert v["class"] == "diverged" and not v["eligible"]
    assert len(v["reasons"]) == 1 and "grad-norm collapse" in v["reasons"][0]
    e, l = v["windows"]["grad_norm_early"]["p50"], v["windows"]["grad_norm_late"]["p50"]
    assert e > rc.BLOWUP_NORM_RATIO * l
    assert v["windows"]["tail"]["ratio"] < rc.TAIL_RATIO_MAX
    # mirror image: the same series rising is the registered blow-up
    assert rc.classify_run(_record(norms=list(reversed(decay))))["class"] == "diverged"
    # a 9x fall is stable, exactly as the registered 9x rise already was
    down9 = [2.6] * 50 + [0.3] * 100
    v9 = rc.classify_run(_record(norms=down9))
    assert v9["class"] == "stable" and v9["reasons"] == []
    assert v9["windows"]["grad_norm_early"]["p50"] == 2.6
    assert v9["windows"]["grad_norm_late"]["p50"] == 0.3


def test_a_loss_falling_from_a_blown_up_level_is_diverged_not_decreasing():
    """recon_5's loss went 2.38 (steps 1-5) -> 11.2 (early window) -> 7.07
    (late window): 'decreasing' by the registered rule, measured from a
    blown-up level, ending 3x worse than it began. A stable run's late
    median must not exceed its own PRE-TRAINING loss — the median over the
    bench's warm-up steps, where the lr is still ramping from zero. No
    constant, no cross-run reference; the same shape ending below the
    pre-training loss is stable; a series without those steps is
    incomplete, fail-closed."""
    pre = [2.4] * 5
    blown = pre + [11.0 - 4.0 * (s - 6) / 144 for s in range(6, 151)]
    v = rc.classify_run(_record(norms=[0.3] * 150, losses=blown))
    assert v["class"] == "diverged" and not v["eligible"]
    assert len(v["reasons"]) == 1
    assert "loss above pre-training" in v["reasons"][0]
    assert "steps 1-5" in v["reasons"][0]
    assert v["windows"]["loss_pretraining"] == {"steps": [1, 5], "median": 2.4}
    assert v["windows"]["loss_late"]["median"] < v["windows"]["loss_early"]["median"]
    fine = pre + [2.3 - 0.3 * (s - 6) / 144 for s in range(6, 151)]
    assert rc.classify_run(_record(norms=[0.3] * 150, losses=fine))["class"] == "stable"
    rec = _record(norms=[0.3] * 150, losses=fine)
    rec["loss"]["steps"] = rec["loss"]["steps"][5:]
    rec["loss"]["values"] = rec["loss"]["values"][5:]
    v = rc.classify_run(rec)
    assert v["class"] == "incomplete" and "[1, 2, 3, 4, 5]" in v["reasons"][0]


def test_the_registered_verdict_is_filed_first_and_every_extension_is_named():
    """Precedence: the registered rule's verdict stands (diverged first),
    then the tail (heavy_tail), then the two extended divergence criteria.
    Whichever names the class, every criterion that fired is a reason."""
    norms = [0.3] * 100 + [40.0] * 50                    # late 133x early; p95/p50 133
    v = rc.classify_run(_record(norms=norms))
    assert v["class"] == "diverged" and "blow-up" in v["reasons"][0]
    assert any("heavy tail" in r for r in v["reasons"])
    # tail + pre-training floor, registered rule passing: heavy_tail, both named
    pre = [2.4] * 5
    blown = pre + [11.0 - 4.0 * (s - 6) / 144 for s in range(6, 151)]
    v = rc.classify_run(_record(norms=_spiky(11.0), losses=blown))
    assert v["class"] == "heavy_tail"
    assert [("heavy tail" in r, "loss above pre-training" in r) for r in v["reasons"]] \
        == [(True, False), (False, True)]


def test_cli_files_the_real_psr_top_as_heavy_tail_and_prints_the_anchors_clip(tmp_path, capsys):
    recs = _real_records("psr_ce_all")
    for rid, rec in recs.items():
        (tmp_path / f"psr_ce_all__{rid}.json").write_text(json.dumps(rec))
    out = tmp_path / "clip.json"
    code = rc.main([str(tmp_path / f"psr_ce_all__{rid}.json") for rid in sorted(recs)]
                   + ["--arm", "psr_ce_all", "--out", str(out)])
    assert code == 0
    text = capsys.readouterr().out
    assert "HEAVY_TAIL (diagnostic only)" in text
    assert "CLIP psr_ce_all = 0.53" in text and "--clip-const 0.53" in text
    assert "7400" not in text and "7.4e+03" not in text
    res = json.loads(out.read_text())
    assert res["eligible_runs"] == ["psr_ce_all__recon_1", "psr_ce_all__recon_3"]
    assert res["ineligible_runs"]["psr_ce_all__recon_5"]["class"] == "heavy_tail"
    assert res["clip"] == 0.53


def test_run_classes_and_the_rule_name_heavy_tail():
    assert rc.RUN_CLASSES == ("stable", "diverged", "heavy_tail", "failed",
                              "incomplete", "over_budget")
    assert "heavy-tail" in rc.CLIP_RULE
