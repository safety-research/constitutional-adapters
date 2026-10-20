"""tune_arm.py analyze's clip-activation column + selection constraint, and
`emit-trials --confirm-from` — the one-row confirmation design at the fitted
argmin.

CPU-only. Synthetic trial directories (summary.json + metrics.jsonl in the
trainer's row schema) under the LR-only design; the real analyze, the real
emitter, the real trial-identity check, and one real (tiny-model) trial run
of a confirmation-stamped design.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from test_tune_arm import (  # noqa: E402  — ONE tune_arm module instance
    TINY, _analyze, _identified_payload, _trial_args, _write_cfg, _write_pool,
    _write_summaries, tune_arm)
from test_tune_arm_lr_only_design import (  # noqa: E402
    CLIP, LLAMA_BASE_CFG, _emit_ns, _lr_only_payload)

rc = tune_arm.recon_clip
guard = tune_arm.hpo_selection_guard
ARM = "psr_ce_all"
THR = rc.CLIP_ACTIVATION_MAX


def _write_metrics(trial_dir: Path, arm: str, clip: float, rate: float,
                   n: int = 40) -> None:
    """A metrics.jsonl in the trainer's row schema whose pre-clip norms
    exceed ``clip`` on exactly round(rate * n) of n optimizer steps."""
    k = int(round(rate * n))
    rows = [{"step": 0, f"{arm}/step0_loss": 3.0, f"{arm}/noop_floor": 3.0}]
    for s in range(1, n + 1):
        g = clip * 1.5 if s <= k else clip * 0.5
        rows.append({"step": s, f"{arm}/ce_loss": 2.0, f"{arm}/ce": 2.0,
                     f"{arm}/lr": 1e-4, f"{arm}/loss_tokens": 100,
                     f"{arm}/real_tokens": 100, f"{arm}/padded_tokens": 110,
                     f"{arm}/sec_per_step": 0.5, "epoch": 0,
                     f"{arm}/grad_norm": g})
    rows.append({"step": n, "probe/heldout_loss": 2.1})
    (trial_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")


RATES = {0: 0.05, 1: 0.30, 2: 0.0, 3: 0.50, 4: 0.10, 5: 0.025, 7: 0.25}
# t6 carries NO metrics.jsonl (rate unknown)


def _world(tmp_path, *, mutate=None, rates=RATES, payload=None):
    payload = payload or _lr_only_payload()
    _write_summaries(tmp_path, payload, mutate=mutate)
    for t in payload["trials"]:
        if t["index"] in rates:
            _write_metrics(tmp_path / f"r1_t{t['index']:02d}", ARM,
                           t["max_grad_norm"], rates[t["index"]])
    return payload


# --------------------------------------------------------------------------
# the clip-activation column and constraint
# --------------------------------------------------------------------------

def test_rows_carry_the_rate_from_their_own_metrics(tmp_path, capsys):
    _world(tmp_path)
    art = _analyze(tmp_path)
    by_id = {guard.trial_id(r): r for r in art["rows"]}
    for i, rate in RATES.items():
        r = by_id[f"r1_t{i:02d}"]
        assert r["clip_activation_rate"] == pytest.approx(rate)
        assert r["clip_n_steps"] == 40
    assert by_id["r1_t06"]["clip_activation_rate"] is None
    assert by_id["r1_t06"]["clip_n_steps"] == 0
    ca = art["clip_activation"]
    assert ca["threshold"] == THR == 0.25 and ca["target"] == 0.10
    assert ca["n_trials"] == 8 and ca["n_with_metrics"] == 7
    assert ca["over_threshold"] == ["r1_t01", "r1_t03", "r1_t07"]  # >= 0.25
    assert ca["all_known_over_threshold"] is False
    assert "best_raw_promotable_row" not in ca
    assert ca["per_trial"]["r1_t03"] == {"rate": 0.5, "n_steps": 40}
    assert ca["per_trial"]["r1_t06"] == {"rate": None, "n_steps": 0}
    # the fold table prints every trial with its rate
    out = capsys.readouterr().out
    assert "clip_act=0.500" in out and "clip_act=n/a" in out
    assert f"threshold {THR:.2f}" in out
    assert "not promotable" in out
    # the guard's briefs carry the rate where known
    g = art["selection_guard"]
    bt = g["spread"]["best_trial"]
    assert bt["clip_activation_rate"] == RATES.get(int(bt["trial"][-2:]),
                                                   None) or \
        "clip_activation_rate" not in bt
    assert g["clip_activation"]["over_threshold"] == ca["over_threshold"]
    assert g["text"][-1].startswith("CLIP ACTIVATION: threshold 0.25")


def test_interior_winner_reads_its_regime_off_the_nearest_trials(tmp_path,
                                                                 capsys):
    _world(tmp_path)
    art = _analyze(tmp_path)
    w = art["winner"]
    assert w["selected_from"] == "fitted_argmin"
    ca = w["clip_activation"]
    assert ca["k"] == 3 and len(ca["nearest"]) == 3
    # the nearest trials of a 2-level design are same-level neighbours
    assert all(n["warmup_frac"] == w["warmup_frac"] for n in ca["nearest"])
    dists = [n["log_box_distance"] for n in ca["nearest"]]
    assert dists == sorted(dists) and all(d < 1.0 for d in dists)
    ids = {n["trial"] for n in ca["nearest"]}
    known = [n["rate"] for n in ca["nearest"] if n["rate"] is not None]
    assert ca["rates_known"] == len(known)
    assert ca["flagged"] == any(r >= THR for r in known)
    assert ca["promotable"] == (None if not known else not ca["flagged"])
    assert art["selection_guard"]["clip_activation"]["winner"] == ca
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines()
                if "winner clip activation" in ln)
    assert all(t in line for t in ids)
    assert ("FLAGGED" in line) == ca["flagged"]
    # the guard stamp's text names the same verdict
    assert art["selection_guard"]["text"][-1].endswith(
        "FLAGGED" if ca["flagged"] else
        "promotable" if ca["promotable"] else "unknown")


def test_edge_fall_through_skips_over_threshold_trials(tmp_path):
    payload = _lr_only_payload()
    lo, hi = tune_arm.LLAMA8B_CE_LR_BOX[ARM]

    def mutate(idx, s):
        t = s["trial"]
        w = 0.0 if t["warmup_frac"] == 0.05 else 1.0
        s["best_heldout_loss"] = math.exp(
            2.0 * (math.log(t["lr"]) - math.log(hi * 10)) ** 2 - 0.5 * w)

    # make the unconstrained best raw trial clip on 60% of its steps
    _write_summaries(tmp_path, payload, mutate=mutate)
    ys = {t["index"]: json.loads((tmp_path / f"r1_t{t['index']:02d}/summary.json")
                                 .read_text())["best_heldout_loss"]
          for t in payload["trials"]}
    best_idx = min(ys, key=ys.get)
    rates = {i: 0.02 for i in ys}
    rates[best_idx] = 0.60
    for t in payload["trials"]:
        _write_metrics(tmp_path / f"r1_t{t['index']:02d}", ARM,
                       t["max_grad_norm"], rates[t["index"]])
    art = _analyze(tmp_path)
    assert art["selection"]["on_edge"] and art["winner"] is None
    assert guard.trial_id(art["best_raw_trial"]) == f"r1_t{best_idx:02d}"
    ca = art["clip_activation"]
    assert ca["over_threshold"] == [f"r1_t{best_idx:02d}"]
    runner = sorted((y, i) for i, y in ys.items() if i != best_idx)[0][1]
    assert ca["best_raw_promotable"] == f"r1_t{runner:02d}"
    g = art["selection_guard"]
    assert g["selected_from"] == "best_raw_trial"
    assert g["selected_trial"] == f"r1_t{runner:02d}"      # not the clipper
    assert g["fall_through"]["reason"] == "fitted_argmin_on_edge"
    assert g["fall_through"]["clip_activation_excluded"] == [f"r1_t{best_idx:02d}"]
    assert g["spread"]["best_trial"]["trial"] == f"r1_t{best_idx:02d}"
    assert g["spread"]["best_trial"]["clip_activation_rate"] == 0.6
    assert tune_arm.CLIP_ACTIVATION_ALARM not in {a["code"] for a in g["alarms"]}


def test_every_trial_over_threshold_alarms_the_guard(tmp_path):
    _world(tmp_path, rates={i: 0.9 for i in range(8)})
    art = _analyze(tmp_path)
    ca = art["clip_activation"]
    assert ca["all_known_over_threshold"] is True
    assert ca["best_raw_promotable"] is None
    g = art["selection_guard"]
    codes = {a["code"] for a in g["alarms"]}
    assert tune_arm.CLIP_ACTIVATION_ALARM in codes == {"clip_activation_over_threshold"} or \
        "clip_activation_over_threshold" in codes
    assert g["alarmed"] is True
    assert any("normaliser" in a["text"] for a in g["alarms"])
    assert any(ln.startswith("ALARM clip_activation_over_threshold")
               for ln in g["text"])
    assert g["text"][-1].startswith("CLIP ACTIVATION")
    assert art["winner"]["clip_activation"]["flagged"] is True
    assert art["winner"]["clip_activation"]["promotable"] is False
    # the fold generator refuses this guard until acknowledged
    from test_sdfbase_configs import msc
    with pytest.raises(SystemExit, match="UNACKNOWLEDGED"):
        msc._validate_fold({"llama8b_base": {ARM: {
            "lr": 1e-4, "max_grad_norm": CLIP, "warmup_frac": 0.15,
            "selection_guard": g}}}, "--fold-hpo")


def test_without_metrics_the_selection_is_the_pre_column_one(tmp_path):
    """No metrics.jsonl anywhere: rates None, nothing excluded, the fall-
    through pick IS the unconstrained best raw — every earlier design's
    behaviour, byte for byte on the selection."""
    payload = _lr_only_payload()
    lo, hi = tune_arm.LLAMA8B_CE_LR_BOX[ARM]

    def mutate(idx, s):
        t = s["trial"]
        s["best_heldout_loss"] = math.exp(
            2.0 * (math.log(t["lr"]) - math.log(hi * 10)) ** 2)

    _write_summaries(tmp_path, payload, mutate=mutate)
    art = _analyze(tmp_path)
    assert all(r["clip_activation_rate"] is None for r in art["rows"])
    ca = art["clip_activation"]
    assert ca["n_with_metrics"] == 0 and ca["over_threshold"] == []
    assert ca["best_raw_promotable"] == guard.trial_id(art["best_raw_trial"])
    g = art["selection_guard"]
    assert g["selected_trial"] == guard.trial_id(art["best_raw_trial"])
    assert "clip_activation_excluded" not in g["fall_through"]
    assert "clip_activation_rate" not in g["spread"]["best_trial"]
    assert not any(a["code"] == tune_arm.CLIP_ACTIVATION_ALARM
                   for a in g["alarms"])
    # interior case: the winner's regime is unknown, not flagged
    _write_summaries(tmp_path, payload)
    art = _analyze(tmp_path)
    w = art["winner"]["clip_activation"]
    assert w["rates_known"] == 0 and w["promotable"] is None
    assert w["flagged"] is False


def test_log_box_distance_is_normalised_and_penalises_a_level_mismatch():
    box = {"lr": [1e-5, 1e-2], "max_grad_norm": [CLIP, CLIP]}
    pt = {"lr": 1e-4, "max_grad_norm": CLIP, "warmup_frac": 0.15}
    same = {"lr": 1e-3, "max_grad_norm": CLIP, "warmup_frac": 0.15}
    other = {"lr": 1e-4, "max_grad_norm": CLIP, "warmup_frac": 0.05}
    d_same = tune_arm._log_box_distance(same, pt, box, {"max_grad_norm": CLIP})
    assert d_same == pytest.approx(1.0 / 3.0)          # one decade of three
    assert tune_arm._log_box_distance(other, pt, box, {"max_grad_norm": CLIP}) \
        == pytest.approx(1.0)
    # a searched clip axis contributes; a fixed one spans nothing
    box2 = {"lr": [1e-5, 1e-2], "max_grad_norm": [0.1, 10.0]}
    far = {"lr": 1e-4, "max_grad_norm": 10.0, "warmup_frac": 0.15}
    pt2 = {"lr": 1e-4, "max_grad_norm": 0.1, "warmup_frac": 0.15}
    assert tune_arm._log_box_distance(far, pt2, box2, {}) == pytest.approx(1.0)
    assert tune_arm._log_box_distance(far, pt2, box2, {"max_grad_norm": 1}) == 0.0


# --------------------------------------------------------------------------
# emit-trials --confirm-from
# --------------------------------------------------------------------------

def _interior_fit(tmp_path):
    _world(tmp_path)
    art = _analyze(tmp_path)
    assert art["winner"] is not None
    return tmp_path / "r1_fit.json", art


def test_confirm_from_emits_one_row_at_the_winner_verbatim(tmp_path, capsys):
    fit_path, art = _interior_fit(tmp_path)
    out = tmp_path / "r9.json"
    tune_arm.emit_trials(_emit_ns(tmp_path, confirm_from=str(fit_path),
                                  out=str(out), round=None))
    p = json.loads(out.read_text())
    w = art["winner"]
    assert p["round"] == tune_arm.CONFIRMATION_ROUND == 9
    assert len(p["trials"]) == 1
    t = p["trials"][0]
    assert t["index"] == 0
    # VERBATIM: the float bits, not a rounding
    assert t["lr"] == w["lr"] and t["max_grad_norm"] == w["max_grad_norm"]
    assert t["warmup_frac"] == w["warmup_frac"]
    assert set(t) == {"index", "lr", "warmup_frac", "max_grad_norm"}
    assert p["box"] == art["box"]
    assert p["warmup_levels"] == [0.05, 0.15]
    assert p["axes"] == ["lr"]                      # the LR-only design
    assert p["constant_axes"] == {"max_grad_norm": CLIP}
    assert p["pinned_axes"] == {} and p["grids"] == {}
    assert p["sobol_m"] == 0 and p["sobol_d"] == 0 and p["design_shift"] == {}
    assert p["arm"] == ARM and p["position"] == "all"
    assert p["si"] == "si_constitution"             # the artefact's data cell
    cfg = tune_arm._load_cfg(str(LLAMA_BASE_CFG))
    assert p["model"] == cfg["model"]
    assert p["model_revision"] == cfg["model_revision"]
    assert p["config_sha"] == tune_arm._config_file_sha(str(LLAMA_BASE_CFG))
    c = p["confirmation_of"]
    assert c["fit"] == str(fit_path)
    assert c["fit_sha256"] == hashlib.sha256(fit_path.read_bytes()).hexdigest()
    assert c["rounds"] == [1] and c["selected_from"] == "fitted_argmin"
    assert c["argmin"] == {k: w[k] for k in ("lr", "max_grad_norm", "warmup_frac")}
    assert c["fitted_min_log_y"] == art["selection"]["fitted_min_log_y"]
    assert c["sigma_log_y"] == art["fit"]["sigma_log_y"]
    assert c["guard_alarmed"] is False and c["n_rows"] == 8
    assert c["clip_activation"] == w["clip_activation"]
    assert tune_arm.is_confirmation_design(p)
    assert not tune_arm.is_confirmation_design(_lr_only_payload())
    # run_trial's identity check accepts it against the same config
    tune_arm._validate_trials_identity(p, cfg, str(LLAMA_BASE_CFG))
    out_txt = capsys.readouterr().out
    assert "wrote 1 CONFIRMATION trial" in out_txt and "round 9" in out_txt


def test_confirm_from_refusals(tmp_path):
    fit_path, art = _interior_fit(tmp_path)

    def ns(**o):
        base = dict(confirm_from=str(fit_path), out=str(tmp_path / "x.json"),
                    round=None)
        base.update(o)
        return _emit_ns(tmp_path, **base)

    with pytest.raises(SystemExit, match="takes no design flags"):
        tune_arm.emit_trials(ns(n=8))
    with pytest.raises(SystemExit, match="takes no design flags"):
        tune_arm.emit_trials(ns(clip_const=0.5))
    with pytest.raises(SystemExit, match="takes no design flags"):
        tune_arm.emit_trials(ns(box_from=str(fit_path)))
    with pytest.raises(SystemExit, match="contradicts the artefact's data cell"):
        tune_arm.emit_trials(ns(si="constv2-mix-10m"))
    with pytest.raises(SystemExit, match="pooled search rounds"):
        tune_arm.emit_trials(ns(round=1))
    with pytest.raises(SystemExit, match="not --arm"):
        tune_arm.emit_trials(ns(arm="lora_r4_ce"))
    # another model's config: the search trained the artefact's model
    gemma = REPO / "configs/sdfbase_gemma4e4b_base_fits.yaml"
    with pytest.raises(SystemExit, match="must run the model the search ran"):
        tune_arm.emit_trials(ns(config=str(gemma)))


def test_confirm_from_refuses_an_edge_or_non_pd_artefact(tmp_path):
    payload = _lr_only_payload()
    lo, hi = tune_arm.LLAMA8B_CE_LR_BOX[ARM]

    def edge(idx, s):
        t = s["trial"]
        s["best_heldout_loss"] = math.exp(
            2.0 * (math.log(t["lr"]) - math.log(hi * 10)) ** 2)

    _write_summaries(tmp_path, payload, mutate=edge)
    art = _analyze(tmp_path)
    assert art["selection"]["on_edge"]
    with pytest.raises(SystemExit, match="box edge"):
        tune_arm.emit_trials(_emit_ns(tmp_path, confirm_from=str(tmp_path / "r1_fit.json"),
                                      out=str(tmp_path / "x.json"), round=None))

    def saddle(idx, s):   # negative curvature in lr: no minimum
        t = s["trial"]
        s["best_heldout_loss"] = math.exp(
            -2.0 * (math.log(t["lr"]) - math.log(3e-4)) ** 2)

    d2 = tmp_path / "nonpd"
    d2.mkdir()
    _write_summaries(d2, payload, mutate=saddle)
    with pytest.raises(SystemExit, match="not positive definite"):
        _analyze(d2)
    with pytest.raises(SystemExit, match="no trustworthy interior minimum"):
        tune_arm.emit_trials(_emit_ns(tmp_path, confirm_from=str(d2 / "r1_fit.json"),
                                      out=str(tmp_path / "x.json"), round=None))


def test_analyze_never_pools_a_confirmation_trial(tmp_path):
    fit_path, art = _interior_fit(tmp_path)
    out = tmp_path / "r9.json"
    tune_arm.emit_trials(_emit_ns(tmp_path, confirm_from=str(fit_path),
                                  out=str(out), round=None))
    p = json.loads(out.read_text())
    d = tmp_path / "r9_t00"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({
        "trial": p["trials"][0], "round": 9, "status": "ok", "arm": ARM,
        "model": p["model"], "position": "all", "si": "si_constitution",
        "box": p["box"], "warmup_levels": p["warmup_levels"],
        "best_heldout_loss": 1.0, "floor_ok": True,
        "confirmation_of": p["confirmation_of"]}))
    # --rounds 1 globs r1_t* only: the artefact is unchanged — the pooled
    # rows exactly (inputs), the fitted winner to 1e-9 (CPU
    # torch.linalg.lstsq is not bitwise deterministic call to call: the
    # same design matrix returned lr 0.00029999999999999905 in one run and
    # 0.0003000000000000006 in the next, so an == on the fitted point
    # flapped on an untouched tree)
    again = _analyze(tmp_path)
    assert again["rows"] == art["rows"]
    w0, w1 = art["winner"], again["winner"]
    assert set(w0) == set(w1)
    for k in ("lr", "max_grad_norm"):
        assert math.isclose(w0[k], w1[k], rel_tol=1e-9, abs_tol=0.0), k
    assert w1["warmup_frac"] == w0["warmup_frac"]
    assert w1["selected_from"] == w0["selected_from"] == "fitted_argmin"
    assert w1["rounds_pooled"] == w0["rounds_pooled"] == [1]
    # listing its round refuses rather than re-fitting around the point
    with pytest.raises(SystemExit, match="CONFIRMATION trial"):
        _analyze(tmp_path, rounds="1,9")
    with pytest.raises(SystemExit, match="CONFIRMATION trial"):
        _analyze(tmp_path, rounds="9")


def test_emit_args_defaults_match_the_cli_parser():
    """The Namespace the driver / tests build carries exactly the parser's
    defaults (so an in-process emit cannot drift from the CLI)."""
    import argparse
    ap = argparse.ArgumentParser()
    # re-create the parser the way main() does, by parsing a minimal argv
    seen = {}

    def fake_emit(a):
        seen.update(vars(a))

    orig_argv = sys.argv
    try:
        sys.argv = ["tune_arm.py", "emit-trials", "--out", "o", "--arm", ARM,
                    "--config", "c"]
        orig = tune_arm.emit_trials
        tune_arm.emit_trials = fake_emit
        try:
            tune_arm.main()
        finally:
            tune_arm.emit_trials = orig
    finally:
        sys.argv = orig_argv
    del seen["fn"], seen["cmd"]
    want = {**tune_arm.EMIT_TRIALS_DEFAULTS, "out": "o", "arm": ARM,
            "config": "c"}
    assert seen == want
    assert seen["round"] is None and seen["confirm_from"] is None
    ns = tune_arm.emit_args(out="o", arm=ARM, config="c")
    assert vars(ns) == want
    with pytest.raises(TypeError, match="unknown"):
        tune_arm.emit_args(nope=1)
    del ap


def test_seed_designs_still_emit_at_round_one_by_default(tmp_path):
    out = tmp_path / "r1.json"
    tune_arm.emit_trials(_emit_ns(tmp_path, out=str(out), round=None,
                                  clip_const=CLIP))
    assert json.loads(out.read_text())["round"] == 1


# --------------------------------------------------------------------------
# a confirmation-stamped design through the REAL trial path (tiny model)
# --------------------------------------------------------------------------

def test_run_trial_echoes_the_confirmation_stamp_into_its_summary(
        tmp_path, tiny_llama):
    cfg = {"model": TINY, "dtype": "float32", "layer": 1, "max_len": 96,
           "train_kwargs": {"default": {"holdout_frac": 0.5,
                                        "holdout_salt": "two_si_holdout"}},
           "wandb_project": "test"}
    _write_cfg(tmp_path, cfg)
    _write_pool(tmp_path, n=16)
    payload = _identified_payload(tmp_path, arm="lora_r1_all", n=2)
    # cut to the one-row round-9 shape a confirmation design has
    payload["round"] = tune_arm.CONFIRMATION_ROUND
    payload["trials"] = [payload["trials"][0]]
    payload["confirmation_of"] = {"fit": "r1_fit.json", "fit_sha256": "0" * 64,
                                  "rounds": [1], "selected_from": "fitted_argmin"}
    (tmp_path / "trials.json").write_text(json.dumps(payload))
    tune_arm.run_trial(_trial_args(tmp_path))
    out = tmp_path / "r9_t00"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "ok" and summary["round"] == 9
    assert summary["confirmation_of"] == payload["confirmation_of"]
    assert tune_arm.is_confirmation_design(summary)
    # a search summary carries no such key
    payload["round"] = 1
    del payload["confirmation_of"]
    (tmp_path / "trials.json").write_text(json.dumps(payload))
    tune_arm.run_trial(_trial_args(tmp_path))
    s1 = json.loads((tmp_path / "r1_t00/summary.json").read_text())
    assert "confirmation_of" not in s1


# --------------------------------------------------------------------------
# The clip-activation rate is the CURRENT attempt's, not the file's
# --------------------------------------------------------------------------

def _run_attempt(tmp_path, *, lr: float, clip: float, epochs: int) -> dict:
    """One real ``run_trial`` at index 0 into ``tmp_path`` with the trial's
    lr / clip rewritten (a different fit fingerprint, so a second call
    re-runs into the SAME r1_t00 directory instead of skipping)."""
    payload = json.loads((tmp_path / "trials.json").read_text())
    payload["trials"] = [payload["trials"][0]]
    payload["trials"][0]["lr"] = lr
    payload["trials"][0]["max_grad_norm"] = clip
    (tmp_path / "trials.json").write_text(json.dumps(payload))
    tune_arm.run_trial(_trial_args(tmp_path, epochs=epochs))
    return json.loads((tmp_path / "r1_t00/summary.json").read_text())


def test_clip_activation_rate_is_over_the_current_attempt_only(
        tmp_path, tiny_llama):
    """With the REAL writer and reader: run_trial re-runs
    fingerprint-mismatched (or torn) work in the same directory and opens
    metrics.jsonl in append mode, so an earlier attempt's pre-clip norms
    sit before the current trajectory's. The rate the summary describes
    is the current attempt's; computed over the whole file it was a
    weighted blend of two different runs. Attempt A trains at a real lr
    (norms fall over the run); attempt B, the CURRENT one, at a frozen lr
    (norms stay at init level) under a clip set midway between A's first
    and last norm — so A's late rows sit below B's clip and dilute B's
    reading."""
    arm = "lora_r1_all"
    key = rc.gradnorm_key(arm)
    cfg = {"model": TINY, "dtype": "float32", "layer": 1, "max_len": 96,
           "train_kwargs": {"default": {"holdout_frac": 0.5,
                                        "holdout_salt": "two_si_holdout"}},
           "wandb_project": "test"}
    _write_cfg(tmp_path, cfg)
    _write_pool(tmp_path, n=32)
    _identified_payload(tmp_path, arm=arm, n=2)
    mpath = tmp_path / "r1_t00/metrics.jsonl"

    # attempt A: a real trajectory, non-binding clip
    s_a = _run_attempt(tmp_path, lr=1e-3, clip=1e6, epochs=6)
    assert s_a["status"] == "ok"
    rows_a = rc.load_metrics_rows(mpath)
    norms_a = [float(r[key]) for r in rows_a if key in r]
    assert len(norms_a) >= 3 and norms_a[0] > norms_a[-1], norms_a
    n_a = len(rows_a)
    clip_b = (norms_a[0] + norms_a[-1]) / 2.0

    # attempt B, the CURRENT one: different work, same directory
    s_b = _run_attempt(tmp_path, lr=1e-9, clip=clip_b, epochs=3)
    assert s_b["status"] == "ok"
    assert s_b["fit_fingerprint"] != s_a["fit_fingerprint"]
    rows_all = rc.load_metrics_rows(mpath)
    assert len(rows_all) > n_a, "the re-run did not append"
    rows_b = rows_all[n_a:]
    norms_b = [float(r[key]) for r in rows_b if key in r]
    assert norms_b, "attempt B logged no pre-clip norm"
    true_rate = sum(1 for v in norms_b if v > clip_b) / len(norms_b)
    diluted = (sum(1 for v in norms_a + norms_b if v > clip_b)
               / (len(norms_a) + len(norms_b)))
    assert diluted != true_rate, (norms_a, norms_b, clip_b)

    # the summary binds its own segment, and the reader honours it
    assert s_b[rc.METRICS_ATTEMPT_KEY]
    assert s_b[rc.METRICS_ATTEMPT_KEY] != s_a[rc.METRICS_ATTEMPT_KEY]
    assert all(r.get(rc.METRICS_ATTEMPT_KEY) == s_b[rc.METRICS_ATTEMPT_KEY]
               for r in rows_b)
    assert all(r.get(rc.METRICS_ATTEMPT_KEY) == s_a[rc.METRICS_ATTEMPT_KEY]
               for r in rows_a)
    rate, n_steps = tune_arm._trial_clip_activation(tmp_path / "r1_t00", s_b)
    assert n_steps == len(norms_b)
    assert rate == true_rate and rate != diluted
    # and analyze's fold block files the promotion verdict from THAT rate
    row = {"lr": 1e-9, "max_grad_norm": clip_b, "warmup_frac": 0.05,
           "round": 1, "index": 0, "y": 1.0,
           "clip_activation_rate": rate, "clip_n_steps": n_steps}
    blk = tune_arm.clip_activation_summary([row], y_key="y")
    assert blk["per_trial"][guard.trial_id(row)]["rate"] == true_rate
    assert (guard.trial_id(row) in blk["over_threshold"]) == (true_rate >= THR)


def test_unstamped_predecessor_summary_refuses_a_stamped_retrys_rows(tmp_path):
    """Through the trial reader: a trial whose terminal summary
    predates the attempt stamp is re-run (fingerprint change) under the
    stamped writer and interrupted after it appended rows but before it
    published its own summary. The old unstamped summary is still the one
    on disk, now over a file whose tail is a stamped partial trajectory.
    The reproduction: the predecessor clipped on 40% of its
    steps (ineligible at THR 25%); read whole-file with the retry's
    non-clipping rows appended, it read 20% — promotable. The reader must
    refuse the ambiguous pairing, naming the trial directory, rather than
    select either segment; the same summary over the unstamped file still
    reads 40% (legacy path unchanged); the retry's own stamped summary,
    once published, reads only its rows."""
    d = tmp_path / "r1_t00"
    d.mkdir()
    clip = 0.1
    key = rc.gradnorm_key(ARM)
    _write_metrics(d, ARM, clip, rate=0.40, n=40)
    legacy_summary = {"arm": ARM, "trial": {"max_grad_norm": clip}}
    assert rc.METRICS_ATTEMPT_KEY not in legacy_summary
    # the pre-stamp reading over a wholly unstamped file is unchanged
    assert tune_arm._trial_clip_activation(d, legacy_summary) == (0.40, 40)
    assert 0.40 >= THR
    # the interrupted stamped retry appended 40 non-clipping rows and no
    # summary; whole-file the blend reads 0.20 < THR
    with (d / "metrics.jsonl").open("a") as fh:
        for s in range(1, 41):
            fh.write(json.dumps({"step": s, key: clip * 0.5,
                                 rc.METRICS_ATTEMPT_KEY: "retry-1"}) + "\n")
    blend = rc.clip_activation_rate(rc.load_metrics_rows(d / "metrics.jsonl"),
                                    key, clip)
    assert blend == (0.20, 80) and blend[0] < THR
    with pytest.raises(ValueError, match="r1_t00") as ei:
        tune_arm._trial_clip_activation(d, legacy_summary)
    msg = str(ei.value)
    assert str(d / "metrics.jsonl") in msg and "attempt" in msg
    assert "'retry-1'" in msg
    # the retry's own summary reads only its own rows
    assert tune_arm._trial_clip_activation(
        d, {**legacy_summary, rc.METRICS_ATTEMPT_KEY: "retry-1"}) == (0.0, 40)
