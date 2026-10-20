"""scripts/sdfbase/continuation_drill.py — the §3.2.4 continuation drill,
CPU-only against a scripted fake client (the real client is the only place
modal is imported; the successor machinery it exercises is imported from
scripts/modal/fits_watch.py, which needs the `modal` stub).

Pinned:
- resume_proof parsing (CE / psr(mse) / lockstep line shapes; step 0 and
  a missing line are NOT a resume; loader refusals recorded);
- three_way_compare on synthetic torch blobs + curves: PASS when A'' == A
  bitwise and A' differs slightly (the floor), FAIL when A'' drifts past
  2x the floor, FAIL on a shape/key mismatch or a missing curve;
- run_drill end to end with a fake that walks running -> timeout,
  running -> cancel -> failed, successor -> done with RESUMING lines, twins
  -> COMPLETE: exactly one successor per ended item, cap honored, overall
  PASS; and the FAIL path where no checkpoint lands (the successor then
  restarts from zero);
- every drill is one run (run id in the volume roots and
  the compare dirs; a pre-existing compare dir refuses before any spawn),
  artifacts are materialized from the CURRENT marker with every file's
  sha256 verified (a mismatch fails, nothing stale is compared), and the
  three-way PASS requires the same published file set.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import pytest
import torch

from test_watch_successor import make_fake_modal  # noqa: E402 — same stub

REPO = Path(__file__).resolve().parents[1]
DRILL_PATH = REPO / "scripts" / "sdfbase" / "continuation_drill.py"


@pytest.fixture(scope="module")
def drill():
    """The drill module with a fake `modal` installed for the duration
    (fits_watch imports modal at module top when run_drill first
    imports it)."""
    fake = make_fake_modal()
    prev = sys.modules.get("modal")
    sys.modules["modal"] = fake
    sys.modules.pop("fits_watch", None)
    spec = importlib.util.spec_from_file_location("_continuation_drill_test",
                                                  DRILL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._fits_watch()  # bind fits_watch to THIS stub
    try:
        yield mod
    finally:
        sys.modules.pop("fits_watch", None)
        if prev is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prev


# --------------------------------------------------------------------------- #
# resume_proof                                                                  #
# --------------------------------------------------------------------------- #


def test_resume_proof_positive_ce_trainer(drill):
    lines = [
        "[fitckpt] loaded checkpoint /vol/out/drill/timeout/ckpt/c__psr_ce_all/ckpt.pt",
        "[fitckpt] train_object_ce RESUMING from checkpoint: epoch 1, "
        "micro-batch 40/125 in epoch, optimizer step 23",
        "[fitckpt] train_object_ce checkpoint written: epoch 1, "
        "micro-batch 80/125 in epoch, optimizer step 28",
        "[fitckpt] volume commit after checkpoint write (c__psr_ce_all)",
    ]
    p = drill.resume_proof(lines)
    assert p["resumed"] is True
    assert (p["epoch"], p["batches_done"], p["n_batches"], p["step"]) == \
        (1, 40, 125, 23)
    assert p["trainer"] == "train_object_ce"
    assert p["loaded_checkpoint"].endswith("ckpt.pt")
    assert p["checkpoint_written"] == [{"trainer": "train_object_ce",
                                        "epoch": 1, "batches_done": 80,
                                        "n_batches": 125, "step": 28}]
    assert p["refusals"] == [] and p["restart_from_zero"] is False
    # a str body works the same as a list of lines
    assert drill.resume_proof("\n".join(lines))["resumed"] is True


def test_resume_proof_other_trainer_line_shapes(drill):
    psr = ("[fitckpt] train_psr(mse) RESUMING from checkpoint: epoch 0, "
           "micro-batch 12/50 in epoch, optimizer step 7 — skipping to the "
           "checkpointed position")
    p = drill.resume_proof([psr])
    assert p["resumed"] and p["step"] == 7 and p["trainer"] == "train_psr(mse)"
    lock = ("[fitckpt] train_lockstep RESUMING from checkpoint: epoch 2, "
            "batch 3/10 in epoch, optimizer steps so far 12 — skipping to "
            "the checkpointed position")
    p = drill.resume_proof([lock])
    assert p["resumed"] and p["step"] == 12 and p["batches_done"] == 3


def test_resume_proof_negative(drill):
    p = drill.resume_proof(["[fitckpt] train_object_ce checkpoint written: "
                            "epoch 0, micro-batch 5/125 in epoch, optimizer "
                            "step 5"])
    assert p["resumed"] is False and p["restart_from_zero"] is True
    assert "RESUMING" in p["reason"]
    assert len(p["checkpoint_written"]) == 1
    p = drill.resume_proof([])
    assert p["resumed"] is False and p["restart_from_zero"] is True
    # a resume at optimizer step 0 is not a resume
    p = drill.resume_proof(["[fitckpt] train_object_ce RESUMING from "
                            "checkpoint: epoch 0, micro-batch 0/125 in epoch, "
                            "optimizer step 0"])
    assert p["resumed"] is False and p["restart_from_zero"] is False
    assert "step 0" in p["reason"]
    # loader refusals are recorded
    ref = ("[fitckpt] /vol/x/ckpt.pt was written for a DIFFERENT fit identity "
           "(checkpoint_key 'a' != expected 'b') — IGNORING it and starting "
           "this fit fresh")
    p = drill.resume_proof([ref])
    assert p["refusals"] == [ref] and p["resumed"] is False
    # a refusal of the latest slot followed by a resume from the previous
    # slot is still a resume
    p = drill.resume_proof([ref, "[fitckpt] train_object_ce RESUMING from "
                                 "checkpoint: epoch 1, micro-batch 2/9 in "
                                 "epoch, optimizer step 3"])
    assert p["resumed"] is True and len(p["refusals"]) == 1


# --------------------------------------------------------------------------- #
# three_way_compare                                                             #
# --------------------------------------------------------------------------- #


def _blob(seed=0, offset=0.0, drop_key=False):
    g = torch.Generator().manual_seed(seed)
    b = {"direction": torch.randn(4, 8, generator=g) + offset,
         "nested": {"gate_w": torch.randn(8, generator=g) + offset,
                    "gate_b": torch.tensor([0.5]) + offset,
                    "count": torch.tensor([3, 4])},  # int leaf: ignored
         "meta": {"arm": "psr_ce_all", "layers": [1, 2, 3]}}
    if drop_key:
        del b["nested"]["gate_b"]
    return b


def _write(d: Path, blob, curve, *, complete_files=("object.pt",
                                                     "train_budget_meta.json",
                                                     "metrics.jsonl")):
    d.mkdir(parents=True, exist_ok=True)
    torch.save(blob, d / "object.pt")
    (d / "heldout_curve.json").write_text(json.dumps({"heldout_curve": curve}))
    (d / "COMPLETE").write_text(json.dumps(
        {"object_file": "object.pt", "files": {n: "00" for n in complete_files}}))


def test_three_way_compare_pass_bitwise_resume(drill, tmp_path):
    a, a2, a3 = (tmp_path / n for n in ("a", "a2", "a3"))
    _write(a, _blob(), [1.0, 0.9, 0.8])
    _write(a2, _blob(offset=1e-6), [1.0, 0.9000001, 0.8])   # the floor
    _write(a3, _blob(), [1.0, 0.9, 0.8])                    # == A bitwise
    r = drill.three_way_compare(a, a2, a3, arm="psr_ce_all")
    assert r["pass"] is True
    assert r["params"]["bitwise_a3_a"] is True
    assert r["params"]["bitwise_a2_a"] is False
    assert r["params"]["dist_a3_a"] == 0.0
    assert 0 < r["params"]["dist_a2_a"] < 1e-4
    assert r["params"]["n_leaves"] == 3  # the int leaf is not a parameter
    assert r["params"]["n_elements"] == 4 * 8 + 8 + 1
    assert r["curve"]["bitwise_a3_a"] is True
    assert r["curve"]["maxabs_a3_a"] == 0.0
    assert r["curve"]["pass"] is True
    assert r["complete_structure"]["match"] is True
    assert r["complete_structure"]["reason"] is None
    assert r["rule"] == {"params": drill.COMPARE_RULE, "curve": drill.CURVE_RULE}
    assert r["curve"]["min_absdiff_a3_a"] == 0.0 and r["curve"]["n_common_a3_a"] == 3


def test_three_way_compare_fails_on_a_different_published_file_set(drill,
                                                                   tmp_path):
    """Bitwise-equal parameters and curve, but the resumed cell published
    a different file set: not the same publication -> FAIL, reason named."""
    a, a2, a3 = (tmp_path / n for n in ("a", "a2", "a3"))
    _write(a, _blob(), [1.0, 0.9])
    _write(a2, _blob(offset=1e-6), [1.0, 0.9])
    _write(a3, _blob(), [1.0, 0.9],
           complete_files=("object.pt", "train_budget_meta.json"))
    r = drill.three_way_compare(a, a2, a3)
    assert r["params"]["pass"] and r["curve"]["pass"]
    assert r["complete_structure"]["match"] is False
    assert "metrics.jsonl" in r["complete_structure"]["reason"]
    assert r["pass"] is False
    # a missing marker is not a match either
    (a3 / "COMPLETE").unlink()
    r = drill.three_way_compare(a, a2, a3)
    assert r["complete_structure"]["match"] is None and r["pass"] is False


def test_three_way_compare_pass_within_floor_not_bitwise(drill, tmp_path):
    """A'' inside 2x the twin floor passes without bitwise equality."""
    a, a2, a3 = (tmp_path / n for n in ("a", "a2", "a3"))
    _write(a, _blob(), [1.0, 0.9])
    _write(a2, _blob(offset=1e-4), [1.0, 0.9001])
    _write(a3, _blob(offset=1.5e-4), [1.0, 0.90015])
    r = drill.three_way_compare(a, a2, a3)
    assert r["pass"] is True
    assert r["params"]["bitwise_a3_a"] is False
    assert r["params"]["dist_a3_a"] <= 2 * r["params"]["dist_a2_a"]


def test_three_way_compare_fail_on_drift(drill, tmp_path):
    a, a2, a3 = (tmp_path / n for n in ("a", "a2", "a3"))
    _write(a, _blob(), [1.0, 0.9, 0.8])
    _write(a2, _blob(offset=1e-6), [1.0, 0.9000001, 0.8])
    _write(a3, _blob(offset=1e-3), [1.0, 0.95, 0.85])  # min off by 0.05
    r = drill.three_way_compare(a, a2, a3)
    assert r["pass"] is False
    assert r["params"]["pass"] is False and r["curve"]["pass"] is False
    assert r["curve"]["min_absdiff_a3_a"] == pytest.approx(0.05)
    # a best-value drift alone fails the compare even with bitwise params
    _write(a3, _blob(), [1.0, 0.95, 0.85])
    r = drill.three_way_compare(a, a2, a3)
    assert r["params"]["pass"] is True and r["params"]["bitwise_a3_a"] is True
    assert r["curve"]["pass"] is False and r["pass"] is False
    # a drift in the TAIL only (same minimum) is reported, not gated
    _write(a3, _blob(), [1.0, 0.95, 0.8])
    r = drill.three_way_compare(a, a2, a3)
    assert r["curve"]["maxabs_a3_a"] == pytest.approx(0.05)
    assert r["curve"]["min_absdiff_a3_a"] == 0.0 and r["curve"]["pass"] is True
    assert r["pass"] is True


def test_three_way_compare_structural_mismatches_are_infinite(drill, tmp_path):
    a, a2, a3 = (tmp_path / n for n in ("a", "a2", "a3"))
    _write(a, _blob(), [1.0, 0.9])
    _write(a2, _blob(offset=1e-6), [1.0, 0.9])
    _write(a3, _blob(drop_key=True), [1.0, 0.9, 0.8])  # key set + length
    r = drill.three_way_compare(a, a2, a3)
    assert r["params"]["dist_a3_a"] == float("inf")
    # the longer curve's extra epoch is coverage (reported), its lower
    # minimum is the failure
    assert r["curve"]["only_a3"] == 1 and r["curve"]["n_common_a3_a"] == 2
    assert r["curve"]["min_absdiff_a3_a"] == pytest.approx(0.1)
    assert r["curve"]["pass"] is False
    assert r["pass"] is False
    # a missing object / curve is also infinite, never a silent pass
    empty = tmp_path / "empty"
    empty.mkdir()
    r = drill.three_way_compare(a, a2, empty)
    assert r["params"]["object_found"]["a3"] is False
    assert r["curve"]["curve_found"]["a3"] is False
    assert r["curve"]["min_absdiff_a3_a"] == float("inf")
    assert r["curve"]["maxabs_a3_a"] == float("inf")
    assert r["pass"] is False


def test_curve_loader_accepts_a_fit_report(drill, tmp_path):
    d = tmp_path / "cell"
    d.mkdir()
    (d / "report_x_all.json").write_text(json.dumps(
        {"arms": {"psr_ce_all": {"heldout_curve": [2.0, 1.5]},
                  "lora_r4_ce": {"heldout_curve": [9.0]}}}))
    assert drill.load_heldout_curve(d, "psr_ce_all") == {0: 2.0, 1: 1.5}
    assert drill.load_heldout_curve(d, "lora_r4_ce") == {0: 9.0}
    assert drill.load_heldout_curve(tmp_path) is None


def _real_layout(d: Path, arm: str, curve: list, *, tail_only_from: int | None = None,
                 sidecar: bool = True) -> None:
    """The materialized file set of a REAL fit cell: no
    heldout_curve.json; train_budget_meta.json with fit_record.heldout_curve
    (the FULL curve, index 0 = the pre-training value); metrics.jsonl rows
    with the ARM-PREFIXED key and an ``epoch`` field (a resumed cell's
    metrics carry only the successor's rows: ``tail_only_from``)."""
    d.mkdir(parents=True, exist_ok=True)
    if sidecar:
        (d / "train_budget_meta.json").write_text(json.dumps(
            {"cell_id": f"x__{arm}", "arm": arm,
             "fit_record": {"best_epoch": 2, "heldout_curve": curve}}))
    rows = [{"step": 0, f"{arm}/step0_loss": 2.5, f"{arm}/noop_floor": 2.5},
            {"step": 0, f"{arm}/heldout_noop_ce": curve[0]},
            {"step": 0, f"{arm}/heldout_loss_init": curve[0]}]
    for e, v in enumerate(curve[1:]):
        if tail_only_from is not None and e < tail_only_from:
            continue
        rows.append({"step": e + 1, "epoch": e, f"{arm}/ce": v + 0.1,
                     f"{arm}/lr": 1e-3})
        rows.append({"step": e + 1, "epoch": e, f"{arm}/heldout_loss": v})
    rows.append({"step": 5, "norms/A_total": 1.0, "probe/heldout_loss": 9.0})
    if tail_only_from is not None:
        rows = [r for r in rows if "heldout_loss_init" not in json.dumps(r)]
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_curve_loader_reads_the_sidecar_and_the_arm_prefixed_metrics(drill, tmp_path):
    """The curve leg once found NO curve for any item — the loader knew
    neither the fits sidecar's ``fit_record.heldout_curve`` nor the
    ``<arm>/heldout_loss`` metrics key. Both now load, aligned (index 0 =
    the pre-training value, e+1 = after epoch e); the sidecar wins because
    it is the FULL curve even for a resumed cell (metrics hold only the
    successor's rows); the arm is inferred when unique, refused when not."""
    curve = [2.3, 2.2, 2.1, 2.15]
    full = tmp_path / "twin"
    _real_layout(full, "lora_r4_ce", curve)
    expect = {0: 2.3, 1: 2.2, 2: 2.1, 3: 2.15}
    assert drill.load_heldout_curve(full, "lora_r4_ce") == expect
    assert drill.load_heldout_curve(full, None) == expect       # the sidecar needs no arm
    # metrics only (no sidecar): the prefixed rows + the init row
    met = tmp_path / "metrics_only"
    _real_layout(met, "lora_r4_ce", curve, sidecar=False)
    assert drill.load_heldout_curve(met, "lora_r4_ce") == expect
    assert drill.load_heldout_curve(met, None) == expect        # unique prefix inferred
    assert drill.load_heldout_curve(met, "psr_ce_all") is None  # another arm: nothing
    assert drill._curve_from_metrics(met / "metrics.jsonl", "lora_r4_ce") == expect
    # a resumed cell: the sidecar carries the full curve, the metrics only
    # the tail — the loader returns the full one
    res = tmp_path / "resumed"
    _real_layout(res, "lora_r4_ce", curve, tail_only_from=2)
    assert drill.load_heldout_curve(res, "lora_r4_ce") == expect
    assert drill._curve_from_metrics(res / "metrics.jsonl", "lora_r4_ce") == {3: 2.15}
    # two arms in one metrics file, no arm given: ambiguous → None
    amb = tmp_path / "amb"
    amb.mkdir()
    (amb / "metrics.jsonl").write_text(
        json.dumps({"epoch": 0, "a/heldout_loss": 1.0}) + "\n"
        + json.dumps({"epoch": 0, "b/heldout_loss": 2.0}) + "\n")
    assert drill.load_heldout_curve(amb, None) is None
    assert drill.load_heldout_curve(amb, "b") == {1: 2.0}
    # the drill's own heldout_curve.json keeps precedence (list or dict form)
    (full / "heldout_curve.json").write_text(json.dumps({"heldout_curve": {"0": 9.0, "1": 8.0}}))
    assert drill.load_heldout_curve(full, "lora_r4_ce") == {0: 9.0, 1: 8.0}
    # the regression itself: the real layout under three_way_compare
    # yields curves for every item (curve_found all True)
    a, a2, a3 = (tmp_path / n for n in ("A", "A2", "A3"))
    for d_, c in ((a, curve), (a2, [v + 1e-4 for v in curve]), (a3, curve)):
        _write(d_, _blob(), [0.0])
        (d_ / "heldout_curve.json").unlink()
        _real_layout(d_, "lora_r4_ce", c, tail_only_from=(2 if d_ is a3 else None))
    r = drill.three_way_compare(a, a2, a3, arm="lora_r4_ce")
    assert r["curve"]["curve_found"] == {"a": True, "a2": True, "a3": True}
    assert r["curve"]["len_a"] == r["curve"]["len_a3"] == 4
    assert r["curve"]["pass"] is True and r["curve"]["bitwise_a3_a"] is True


def test_curve_rule_gates_the_best_value_not_the_chaotic_tail(drill, tmp_path):
    """The measured numbers, synthetically. Twins A / A' differ by
    7.1e-3 at the LAST epoch (the post-minimum overfit tail) and 6.8e-4 at
    their minima; a resumed run whose tail is 3.8e-2 off but whose minimum
    is within 2.9e-4 PASSES (max-abs 5x the floor is reported); a resumed
    run whose best value drifts by 5e-3 FAILS; with two bitwise twins the
    1e-3 absolute tolerance is the floor (5e-4 passes, 2e-3 fails)."""
    base = [2.3474, 2.3400, 2.3200, 2.3100, 2.3057, 2.3060, 2.3300, 2.4459]
    twin = list(base)
    twin[4] = 2.3057 - 6.8e-4        # the twins' minima differ by 6.8e-4
    twin[-1] = base[-1] - 7.1e-3     # ... and their tails by 7.1e-3
    a, a2, a3 = (tmp_path / n for n in ("a", "a2", "a3"))
    _write(a, _blob(), base)
    _write(a2, _blob(offset=1e-6), twin)
    healthy = list(base)
    healthy[4] = 2.3057 - 2.9e-4
    healthy[-1] = base[-1] - 3.8e-2  # a 5x-the-floor tail: chaos, not drift
    _write(a3, _blob(offset=1.5e-6), healthy)
    r = drill.three_way_compare(a, a2, a3)
    cu = r["curve"]
    assert cu["maxabs_a2_a"] == pytest.approx(7.1e-3)
    assert cu["maxabs_a3_a"] == pytest.approx(3.8e-2)      # > 2x the floor: reported
    assert cu["min_absdiff_a2_a"] == pytest.approx(6.8e-4)
    assert cu["min_absdiff_a3_a"] == pytest.approx(2.9e-4)
    assert cu["argmin_a"] == cu["argmin_a2"] == cu["argmin_a3"] == 4
    assert cu["min_tol"] == drill.CURVE_MIN_TOL == 1e-3
    assert cu["pass"] is True and r["pass"] is True
    # the best value drifts by 5e-3 (the whole basin sits higher): FAIL
    # (2 x 6.8e-4 = 1.36e-3 bound)
    drifted = list(base)
    for i in (3, 4, 5):
        drifted[i] += 5e-3
    _write(a3, _blob(offset=1.5e-6), drifted)
    r = drill.three_way_compare(a, a2, a3)
    assert r["curve"]["min_absdiff_a3_a"] == pytest.approx(5e-3)
    assert r["curve"]["argmin_a3"] == 4
    assert r["curve"]["pass"] is False and r["pass"] is False
    # a single epoch bumped out of the basin moves the argmin, not the
    # best value: the gate reads the value (a near-tie's index is noise)
    bumped = list(base)
    bumped[4] = 2.3057 + 5e-3
    _write(a3, _blob(offset=1.5e-6), bumped)
    r = drill.three_way_compare(a, a2, a3)
    assert r["curve"]["argmin_a3"] == 5
    assert r["curve"]["min_absdiff_a3_a"] == pytest.approx(3e-4)
    assert r["curve"]["pass"] is True
    # bitwise twins: the absolute tolerance is the floor
    _write(a2, _blob(offset=1e-6), base)
    ok = list(base)
    ok[4] = 2.3057 - 5e-4
    _write(a3, _blob(offset=1.5e-6), ok)
    r = drill.three_way_compare(a, a2, a3)
    assert r["curve"]["min_absdiff_a2_a"] == 0.0 and r["curve"]["pass"] is True
    bad = list(base)
    bad[4] = 2.3057 - 2e-3
    _write(a3, _blob(offset=1.5e-6), bad)
    assert drill.three_way_compare(a, a2, a3)["curve"]["pass"] is False
    # the rule strings say what is gated
    assert "min(A'')" in drill.CURVE_RULE and "diagnostic" in drill.CURVE_RULE
    assert drill.COMPARE_RULE == "dist(A'', A) <= max(2 * dist(A', A), 1e-9)"


def test_cli_refuses_without_torch_and_compare_only_runs_step_f(drill, tmp_path,
                                                                monkeypatch, capsys):
    """The drill was once invoked with an interpreter lacking torch and
    died at its final compare after the GPU steps; main() now refuses at
    START (before any Modal client exists). ``--compare-only <run dir>``
    re-runs step (f) alone on a materialized run and writes that verdict;
    without it --app / --config-sha are required."""
    real_find = drill.importlib.util.find_spec
    monkeypatch.setattr(drill.importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "torch" else real_find(name, *a, **k))
    monkeypatch.setattr(drill, "RealModalClient",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("modal touched")))
    rc = drill.main(["--app", "x", "--config-sha", "y"])
    err = capsys.readouterr().err
    assert rc == 2 and "torch is not importable" in err and ".venv/bin/python" in err
    assert drill.require_torch() is not None
    monkeypatch.setattr(drill.importlib.util, "find_spec", real_find)
    assert drill.require_torch() is None
    rc = drill.main(["--config-sha", "y"])
    assert rc == 2 and "--app and --config-sha are required" in capsys.readouterr().err
    # compare-only on a materialized run dir
    run = tmp_path / "run"
    for n in drill.ITEM_NAMES:
        _write(run / n, _blob(offset=(1e-6 if n == "twin_a2" else 0.0)),
               [1.0, 0.9, 0.8 + (1e-4 if n == "cancel" else 0.0)])
    out = tmp_path / "verdict.json"
    rc = drill.main(["--compare-only", str(run), "--arm", "psr_ce_all", "--out", str(out)])
    stdout = capsys.readouterr().out
    assert rc == 0
    v = json.loads(out.read_text())
    assert v["compare_only"] is True and v["pass"] is True and v["verdict"] == "PASS"
    assert set(v["compares"]) == set(drill.INTERRUPTED) and v["arm"] == "psr_ce_all"
    assert v["compares"]["cancel"]["curve"]["min_absdiff_a3_a"] == pytest.approx(1e-4)
    assert "compare timeout: params PASS" in stdout and "diagnostic" in stdout
    # a failing compare → exit 1
    _write(run / "timeout", _blob(offset=0.5), [1.0, 0.9, 0.3])
    assert drill.main(["--compare-only", str(run), "--out", str(out)]) == 1
    assert json.loads(out.read_text())["verdict"] == "FAIL"
    # a run dir missing an item refuses by name
    with pytest.raises(ValueError, match="missing materialized item dir"):
        drill.compare_only(tmp_path / "nope", arm=None, out=out)


# --------------------------------------------------------------------------- #
# run_drill with a scripted fake client                                         #
# --------------------------------------------------------------------------- #

CKPT_AFTER_S = 60.0       # the first checkpoint lands 60 s after spawn
TIMEOUT_S = 600.0         # the drill app's real function timeout
FINISH_S = 300.0          # successors and twins finish in 300 s
RESUME_LINES = [
    "[fitckpt] loaded checkpoint /vol/out/drill/X/ckpt/c/ckpt.pt",
    "[fitckpt] train_object_ce RESUMING from checkpoint: epoch 0, "
    "micro-batch 25/125 in epoch, optimizer step 5",
    "[fitckpt] train_object_ce checkpoint written: epoch 0, "
    "micro-batch 50/125 in epoch, optimizer step 10",
]
FRESH_LINES = [
    "[fitckpt] train_object_ce checkpoint written: epoch 0, "
    "micro-batch 25/125 in epoch, optimizer step 5",
]


def _blob_bytes(blob) -> bytes:
    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue()


class FakeClient:
    """Walks the drill's states on a fake clock (sleep advances it):
    original `timeout` item: running -> timeout at 600 s; original
    `cancel` item: running until cancelled -> failed; successors (a second
    spawn of the same tuple) and twins: done + COMPLETE after 300 s. The
    checkpoint lands 60 s after spawn unless ckpt_lands is False for that
    item name, in which case its successor restarts from zero.

    The volume serves each COMPLETE cell's files (``blobs[name]``: filename
    -> bytes; the marker lists their sha256s), so the drill materializes
    them exactly as it would off the real volume. ``corrupt[(name, file)]``
    substitutes other bytes for a fetch (a sha mismatch)."""

    def __init__(self, *, no_ckpt_for=(), twins_outlast_timeout=False):
        self.t = 1_700_000_000.0
        # the real contradiction: an uninterrupted twin spawned on the
        # DRILL app is the same fit as the timeout item, so it hits the
        # drill timeout too; only a call tagged `long` (spawned through a
        # LongTwinClient) runs to completion
        self.twins_outlast_timeout = twins_outlast_timeout
        self.calls: dict = {}
        self.n_by_item: dict = {}
        self.spawn_log: list = []
        self.cancelled: list = []
        self.no_ckpt_for = set(no_ckpt_for)
        self.blobs: dict = {
            name: {"object.pt": _blob_bytes(
                       _blob(offset=1e-6 if name == "twin_a2" else 0.0)),
                   "metrics.jsonl": b'{"step": 1}\n'}
            for name in ("timeout", "cancel", "twin_a", "twin_a2")}
        self.corrupt: dict = {}

    # -- clock
    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s

    # -- deploy
    def function_id(self):
        return "fu-drill-1"

    def deployment(self):
        return {"app_id": "ap-drill", "app_version": 1,
                "deploy_commit": "abc123def456"}

    # -- calls
    @staticmethod
    def _name(item):
        return item[5].rsplit("/", 1)[-1]

    def spawn(self, item):
        n = self.n_by_item.get(item, 0) + 1
        self.n_by_item[item] = n
        name = self._name(item)
        cid = f"fc-{name}-{n}"
        self.calls[cid] = {"item": item, "name": name,
                           "kind": "orig" if n == 1 else "succ",
                           "spawn_t": self.t, "cancel_t": None}
        self.spawn_log.append((name, n, cid))
        return cid

    def _elapsed(self, c):
        return self.t - c["spawn_t"]

    def state(self, cid):
        c = self.calls[cid]
        el = self._elapsed(c)
        if c["cancel_t"] is not None:
            return "failed"
        if c["kind"] == "orig" and c["name"] == "timeout":
            return "timeout" if el >= TIMEOUT_S else "running"
        if c["kind"] == "orig" and c["name"] == "cancel":
            return "running"  # would run on; the drill cancels it
        if (c["kind"] == "orig" and c["name"] in ("twin_a", "twin_a2")
                and self.twins_outlast_timeout and not c.get("long")):
            return "timeout" if el >= TIMEOUT_S else "running"
        return "done" if el >= FINISH_S else "running"

    def cancel(self, cid):
        self.calls[cid]["cancel_t"] = self.t
        self.cancelled.append(cid)

    def result(self, cid):
        c = self.calls[cid]
        if self.state(cid) != "done":
            return None
        if c["kind"] == "succ":
            lines = FRESH_LINES if c["name"] in self.no_ckpt_for else RESUME_LINES
        else:
            lines = FRESH_LINES
        return {"status": "ok", "rc": 0, "fitckpt_lines": list(lines)}

    def logs(self, cid):
        r = self.result(cid)
        return "\n".join(r["fitckpt_lines"]) if r else ""

    # -- volume
    def _ckpt_calls(self, name):
        return [c for c in self.calls.values()
                if c["name"] == name and self._elapsed(c) >= CKPT_AFTER_S
                and c["cancel_t"] is None]

    def ckpt_mtime(self, vol_path):
        name = vol_path.split("/ckpt/", 1)[0].rsplit("/", 1)[-1]
        if name in self.no_ckpt_for:
            return None
        cs = self._ckpt_calls(name)
        return max(c["spawn_t"] + CKPT_AFTER_S for c in cs) if cs else None

    def _complete(self, name):
        return any(c["name"] == name and self.state(cid) == "done"
                   for cid, c in self.calls.items())

    def read_json(self, vol_path):
        if vol_path.endswith("/COMPLETE"):
            name = vol_path.split("/cells_all/", 1)[0].rsplit("/", 1)[-1]
            if self._complete(name):
                return {"object_file": "object.pt",
                        "files": {f: hashlib.sha256(b).hexdigest()
                                  for f, b in self.blobs[name].items()}}
            return None
        if "/reports/" in vol_path:
            name = vol_path.split("/reports/", 1)[0].rsplit("/", 1)[-1]
            if self._complete(name):
                return {"arms": {"psr_ce_all": {"heldout_curve": [1.0, 0.9]}}}
        return None

    def read_bytes(self, vol_path):
        if vol_path.endswith("/COMPLETE"):
            m = self.read_json(vol_path)
            return json.dumps(m).encode() if m else None
        if "/cells_all/" not in vol_path:
            return None
        name = vol_path.split("/cells_all/", 1)[0].rsplit("/", 1)[-1]
        fname = vol_path.rsplit("/", 1)[-1]
        if not self._complete(name):
            return None
        if (name, fname) in self.corrupt:
            return self.corrupt[(name, fname)]
        return self.blobs[name].get(fname)


class LongTwinClient:
    """The second client: the SAME volume and clock (delegates to the shared
    FakeClient), a different deployed function (the long-timeout app), and
    calls it spawns are tagged `long` so they run to completion."""

    def __init__(self, base: FakeClient):
        self.base = base
        self.spawned: list = []

    def function_id(self):
        return "fu-drill-long-1"

    def deployment(self):
        return {"app_id": "ap-drill-long", "app_version": 1,
                "deploy_commit": "abc123def456"}

    def spawn(self, item):
        cid = self.base.spawn(item)
        self.base.calls[cid]["long"] = True
        self.spawned.append(cid)
        return cid

    def __getattr__(self, name):
        return getattr(self.base, name)


RUN = "20260902T181500Z"


def _items(drill, tmp_path, run_id=RUN):
    return drill.build_items(model="gemma4e4b_base", si="constv2-mix-1k",
                             arm="psr_ce_all", config_sha="abcdef0123456789",
                             out_root="out/drill", checkpoint_every=5,
                             compare_root=tmp_path / "cmp", run_id=run_id)


def _run(drill, client, items, **over):
    kw = dict(items=items, out_root="out/drill", poll_s=30, max_wait_s=5400,
              ckpt_every=5, function_timeout_s=600, run_id=RUN,
              app="sdfbase-fits-drill", function="fit_h200_std",
              log=lambda *a, **k: None)
    kw.update(over)
    return drill.run_drill(client, **kw)


def test_build_items_shape_and_distinct_roots(drill, tmp_path):
    """One run id namespaces the volume roots AND the compare dirs."""
    items = _items(drill, tmp_path)
    assert set(items) == set(drill.ITEM_NAMES)
    for name, spec in items.items():
        it = spec["item"]
        assert it == ("gemma4e4b_base", "constv2-mix-1k", "psr_ce_all",
                      "abcdef0123456789", False, f"out/drill/{RUN}/{name}", 5)
        drill.check_item_shape(it)  # round-trips through successor_item
        assert spec["dir"].endswith(f"{RUN}/{name}")
        assert not Path(spec["dir"]).exists()  # nothing is created up front
    assert drill.ckpt_path(items["timeout"]["item"]) == \
        f"out/drill/{RUN}/timeout/ckpt/constv2-mix-1k__psr_ce_all/ckpt.pt"
    assert drill.complete_path(items["twin_a"]["item"]) == \
        f"out/drill/{RUN}/twin_a/cells_all/constv2-mix-1k__psr_ce_all/COMPLETE"
    with pytest.raises(ValueError):
        drill.check_item_shape(("gemma4e4b_base", "si", "arm"))
    with pytest.raises(ValueError, match="run_id"):
        _items(drill, tmp_path, run_id="not/a/token")
    assert drill.default_run_id(1_700_000_000.0) == "20231114T221320Z"
    # the manifest row the drill hands the production successor path
    # carries the row's own spawn pin (fits_watch.row_pin)
    cell = drill.manifest_cell(items["timeout"]["item"], app="a", function="f",
                               function_id="fu-1", call_id="c", spawn_time="t")
    assert {k: cell[k] for k in ("app", "function", "function_id", "config_sha",
                                 "overwrite", "checkpoint_every")} == {
        "app": "a", "function": "f", "function_id": "fu-1",
        "config_sha": "abcdef0123456789", "overwrite": False,
        "checkpoint_every": 5}


def test_run_drill_full_pass(drill, tmp_path):
    items = _items(drill, tmp_path)
    client = FakeClient()
    lines = []
    v = _run(drill, client, items, app="sdfbase-fits-drill",
             function="fit_h200_std",
             log=lambda *a, **k: lines.append(" ".join(map(str, a))))
    assert v["verdict"] == "PASS" and v["pass"] is True, v["failed_steps"]
    assert v["run_id"] == RUN
    steps = v["steps"]
    assert {k: s["verdict"] for k, s in steps.items()} == {
        "a_deploy": "PASS", "b_checkpoint_landed": "PASS", "c_cancel": "PASS",
        "c_real_timeout": "PASS", "d_successor_timeout": "PASS",
        "d_successor_cancel": "PASS", "e_resume_timeout": "PASS",
        "e_resume_cancel": "PASS", "f_twins_complete": "PASS",
        "f_materialized": "PASS",
        "f_compare_timeout": "PASS", "f_compare_cancel": "PASS"}
    # materialized from the CURRENT marker, every file sha-verified, into
    # the run's own compare dirs
    for name in drill.ITEM_NAMES:
        m = steps["f_materialized"][name]
        assert m["materialized"] is True and m["verified"] is True
        assert m["fetched"] == ["metrics.jsonl", "object.pt"]
        assert m["sha_mismatch"] == [] and m["missing"] == []
        d = Path(items[name]["dir"])
        assert d.parent.name == RUN
        assert (d / "object.pt").read_bytes() == client.blobs[name]["object.pt"]
        assert (d / "COMPLETE").is_file() and (d / "heldout_curve.json").is_file()
    assert steps["f_compare_timeout"]["complete_structure"]["match"] is True
    assert steps["d_successor_timeout"]["marker_state"] == "absent"
    # exactly one successor per ended item; twins spawned once each
    by_name = {}
    for name, n, _cid in client.spawn_log:
        by_name[name] = max(by_name.get(name, 0), n)
    assert by_name == {"timeout": 2, "cancel": 2, "twin_a": 1, "twin_a2": 1}
    assert len(client.spawn_log) == 6
    # the successor rode the production path: same tuple, same app version
    for n in ("timeout", "cancel"):
        d = steps[f"d_successor_{n}"]
        assert d["same_app_version"] is True
        assert d["successor"]["function_id"] == "fu-drill-1"
        assert d["successor"]["predecessor_call_id"] == f"fc-{n}-1"
        assert d["successor"]["call_id"] == f"fc-{n}-2"
        assert d["n_successors"] == 1 and d["cap_honored"] is True
        assert "cap reached (1/1)" in d["cap_reason"]
        assert d["cell_status"].startswith("successor #1 spawned")
        e = steps[f"e_resume_{n}"]
        assert e["resume_proof"]["resumed"] is True
        assert e["resume_proof"]["step"] == 5
        assert e["complete_marker"] is True
    assert steps["d_successor_timeout"]["predecessor_state"] == "timeout"
    assert steps["d_successor_cancel"]["predecessor_state"] == "failed"
    assert steps["c_real_timeout"]["observed_state"] == "timeout"
    assert steps["c_real_timeout"]["wall_since_spawn_s"] >= 600
    assert steps["b_checkpoint_landed"]["ckpt_landed"] is True
    assert steps["b_checkpoint_landed"]["seconds_after_spawn"] == 60.0
    assert client.cancelled == ["fc-cancel-1"]
    assert steps["c_cancel"]["observed_state"] == "failed"
    assert steps["f_compare_timeout"]["params"]["bitwise_a3_a"] is True
    assert steps["f_compare_timeout"]["params"]["bitwise_a2_a"] is False
    assert v["items"]["timeout"]["successor_call_id"] == "fc-timeout-2"
    assert v["elapsed_s"] < 5400
    assert any("VERDICT PASS" in ln for ln in lines)
    out = drill.write_verdict(v, tmp_path / "verdict.json")
    assert json.loads(out.read_text())["verdict"] == "PASS"
    assert not (tmp_path / "verdict.json.tmp").exists()


def test_run_drill_twins_need_the_long_timeout_app(drill, tmp_path):
    """Found live (2026-09-03): the twins are the same fit as the timeout
    item, so on the drill app they hit the drill timeout too and step (f)
    can never pass. With a twin client (a second deploy of the same commit
    at the production timeout) the twins complete, the interrupted items
    and their successors still ride the drill app, and the verdict records
    both pins."""
    # without --twin-app: twins time out, f_twins_complete fails
    items = _items(drill, tmp_path)
    base = FakeClient(twins_outlast_timeout=True)
    v = _run(drill, base, items)
    assert v["pass"] is False
    f = v["steps"]["f_twins_complete"]
    assert f["verdict"] == "FAIL"
    assert f["twin_a"]["state"] == "timeout" and f["twin_a"]["complete"] is False
    assert v["steps"]["a_deploy"]["twins_on_long_timeout_app"] is False
    assert v["twin_app"] == "sdfbase-fits-drill"  # defaulted to the drill app
    # with --twin-app: PASS; twins via the long client, everything else via
    # the drill client
    items = _items(drill, tmp_path, run_id="20260902T190000Z")
    base = FakeClient(twins_outlast_timeout=True)
    long = LongTwinClient(base)
    v = _run(drill, base, items, run_id="20260902T190000Z",
             twin_client=long, twin_app="sdfbase-fits-drill-long")
    assert v["pass"] is True, v["failed_steps"]
    assert sorted(long.spawned) == ["fc-twin_a-1", "fc-twin_a2-1"]
    assert len(base.spawn_log) == 6          # 2 twins + 2 items + 2 successors
    a = v["steps"]["a_deploy"]
    assert a["function_id"] == "fu-drill-1"
    assert a["twin_function_id"] == "fu-drill-long-1"
    assert a["twin_app"] == "sdfbase-fits-drill-long"
    assert a["twins_on_long_timeout_app"] is True
    assert "sdfbase-fits-drill-long" in a["operator_step_twins"]
    assert v["twin_app"] == "sdfbase-fits-drill-long"
    assert v["twin_function_id"] == "fu-drill-long-1"
    assert v["params"]["twin_app"] == "sdfbase-fits-drill-long"
    # successors still pinned to the DRILL function (the row's own pin)
    for n in ("timeout", "cancel"):
        assert v["steps"][f"d_successor_{n}"]["successor"]["function_id"] == "fu-drill-1"
    assert v["steps"]["f_twins_complete"]["twin_a"]["state"] == "done"


def test_run_drill_refuses_a_pre_existing_compare_dir(drill, tmp_path):
    """A drill never compares against a previous run's artifacts: an
    existing compare dir refuses BEFORE anything is spawned."""
    items = _items(drill, tmp_path)
    Path(items["twin_a"]["dir"]).mkdir(parents=True)
    client = FakeClient()
    with pytest.raises(ValueError, match="already exist"):
        _run(drill, client, items)
    assert client.spawn_log == []
    # ... and _materialize itself refuses an existing dir even if reached
    ev = drill._materialize(client, items["twin_a"]["item"],
                            Path(items["twin_a"]["dir"]))
    assert ev["materialized"] is False and "already exists" in ev["reason"]


def test_run_drill_fails_when_a_fetched_file_mismatches_the_marker(drill,
                                                                  tmp_path):
    """Materialization verifies every fetched file's sha256 against the
    CURRENT marker: a mismatch writes nothing and fails the run (the compare
    then sees no object — never a stale or torn artifact)."""
    items = _items(drill, tmp_path)
    client = FakeClient()
    client.corrupt[("timeout", "object.pt")] = b"not the published bytes"
    v = _run(drill, client, items)
    assert v["pass"] is False
    m = v["steps"]["f_materialized"]
    assert m["verdict"] == "FAIL"
    assert m["timeout"]["materialized"] is False
    assert len(m["timeout"]["sha_mismatch"]) == 1
    assert m["timeout"]["sha_mismatch"][0].startswith("object.pt:")
    assert not Path(items["timeout"]["dir"]).exists()  # nothing written
    assert m["cancel"]["materialized"] is True
    c = v["steps"]["f_compare_timeout"]
    assert c["verdict"] == "FAIL" and c["params"]["object_found"]["a3"] is False
    assert v["steps"]["f_compare_cancel"]["verdict"] == "PASS"
    assert set(v["failed_steps"]) == {"f_materialized", "f_compare_timeout"}


def test_run_drill_fails_when_the_resumed_cell_publishes_a_different_file_set(
        drill, tmp_path):
    items = _items(drill, tmp_path)
    client = FakeClient()
    client.blobs["cancel"]["extra_debug.json"] = b"{}"
    v = _run(drill, client, items)
    assert v["pass"] is False
    assert v["steps"]["f_materialized"]["verdict"] == "PASS"  # all verified
    c = v["steps"]["f_compare_cancel"]
    assert c["verdict"] == "FAIL"
    assert c["params"]["pass"] and c["curve"]["pass"]
    assert c["complete_structure"]["match"] is False
    assert "extra_debug.json" in c["complete_structure"]["reason"]
    assert v["steps"]["f_compare_timeout"]["verdict"] == "PASS"
    assert v["failed_steps"] == ["f_compare_cancel"]


def test_run_drill_fails_when_no_checkpoint_lands(drill, tmp_path):
    items = _items(drill, tmp_path)
    client = FakeClient(no_ckpt_for={"timeout"})
    v = _run(drill, client, items)
    assert v["verdict"] == "FAIL" and v["pass"] is False
    b = v["steps"]["b_checkpoint_landed"]
    assert b["verdict"] == "FAIL" and b["ckpt_landed"] is False
    assert b["call_state"] == "timeout"  # waited until the call ended
    # the timeout still fired, the successor was still spawned (a
    # continuation event is a continuation event) — and its lines prove
    # a restart from zero, which is the second FAIL
    assert v["steps"]["c_real_timeout"]["verdict"] == "PASS"
    assert v["steps"]["d_successor_timeout"]["verdict"] == "PASS"
    e = v["steps"]["e_resume_timeout"]
    assert e["verdict"] == "FAIL"
    assert e["resume_proof"]["restart_from_zero"] is True
    # the untouched cancel lane still passes end to end
    assert v["steps"]["c_cancel"]["verdict"] == "PASS"
    assert v["steps"]["e_resume_cancel"]["verdict"] == "PASS"
    assert set(v["failed_steps"]) == {"b_checkpoint_landed", "e_resume_timeout"}


def test_run_drill_done_before_timeout_is_a_fail_not_a_successor(drill, tmp_path):
    """A 1k cell that FINISHES before the function timeout gives no
    continuation event: step c fails with the fix spelled out and no
    successor is spawned for it (a returned call is never respawned)."""
    items = _items(drill, tmp_path)
    client = FakeClient()
    orig_state = client.state

    def state(cid):
        c = client.calls[cid]
        if c["kind"] == "orig" and c["name"] == "timeout":
            return "done" if client._elapsed(c) >= FINISH_S else "running"
        return orig_state(cid)

    client.state = state
    v = _run(drill, client, items)
    assert v["pass"] is False
    c = v["steps"]["c_real_timeout"]
    assert c["verdict"] == "FAIL" and c["observed_state"] == "done"
    assert "finished before the function timeout" in c["reason"]
    d = v["steps"]["d_successor_timeout"]
    assert d["verdict"] == "FAIL" and d["can_respawn"] is False
    # the finished cell published COMPLETE: the marker verdict (read the way
    # the watcher reads it) is what refuses — a COMPLETE cell is never
    # respawned, whatever its call state
    assert d["marker_state"] == "complete"
    assert "COMPLETE" in d["can_respawn_reason"]
    assert v["steps"]["e_resume_timeout"]["verdict"] == "FAIL"
    assert max(n for name, n, _ in client.spawn_log if name == "timeout") == 1
    # a returned call WITHOUT a marker is the other refusal: not a
    # continuation event
    cell = drill.manifest_cell(items["timeout"]["item"], app="a", function="f",
                               function_id="fu", call_id="c", spawn_time="t")
    ok, why = drill._fits_watch().can_respawn(cell, "done", max_successors=1,
                                              marker="absent")
    assert not ok and "not a continuation event" in why


def test_run_drill_refuses_a_redeployed_app(drill, tmp_path):
    """The version pin is live in the drill too: if the deployed function
    id the drill sees differs from the one recorded at spawn, no successor
    is spawned and the step says why."""
    items = _items(drill, tmp_path)
    client = FakeClient()
    ids = iter(["fu-drill-1"] + ["fu-drill-2"] * 50)
    client.function_id = lambda: next(ids)  # first read pins, later reads differ
    v = _run(drill, client, items)
    d = v["steps"]["d_successor_timeout"]
    assert d["verdict"] == "FAIL" and d["same_app_version"] is False
    assert "redeployed" in d["same_app_version_reason"]
    assert v["pass"] is False
    assert all(n == 1 for name, n, _ in client.spawn_log)


def test_run_drill_rejects_bad_items(drill, tmp_path):
    items = _items(drill, tmp_path)
    del items["cancel"]
    with pytest.raises(ValueError, match="missing"):
        _run(drill, FakeClient(), items, poll_s=1, max_wait_s=10)
    items = _items(drill, tmp_path)
    items["cancel"]["item"] = items["timeout"]["item"]  # same out_root
    with pytest.raises(ValueError, match="distinct out_roots"):
        _run(drill, FakeClient(), items, poll_s=1, max_wait_s=10)
    # the app/function names ARE the rows' spawn pin: never blank
    with pytest.raises(ValueError, match="spawn pin"):
        _run(drill, FakeClient(), _items(drill, tmp_path), function="")


def test_cli_parser_defaults(drill):
    ns = drill.build_parser().parse_args(
        ["--app", "sdfbase-fits-drill", "--config-sha", "abcdef0123456789"])
    assert ns.function == "fit_h200_std"
    assert ns.model == "gemma4e4b_base" and ns.si == "constv2-mix-1k"
    assert ns.arm == "psr_ce_all" and ns.out_root == "out/drill"
    assert ns.checkpoint_every == 5 and ns.function_timeout == 600.0
    assert ns.poll == 30.0 and ns.max_wait == 5400.0
    assert ns.run_id is None  # main() mints the UTC-time default
    assert ns.twin_app is None  # main() then defaults it to --app
    assert drill.build_parser().parse_args(
        ["--app", "d", "--config-sha", "x", "--twin-app", "d-long"]
    ).twin_app == "d-long"
    assert ns.out == "$STAGE/sdfbase/drill_verdict.json"  # main() expands it through the site
    assert ns.compare_only is None
    # --app is validated in main() (not by argparse) so --compare-only can
    # run without a Modal app at all
    ns = drill.build_parser().parse_args(["--config-sha", "x"])
    assert ns.app is None
    ns = drill.build_parser().parse_args(["--compare-only", "/tmp/x"])
    assert ns.compare_only == "/tmp/x" and ns.app is None and ns.config_sha is None


def test_real_client_is_the_only_modal_import(drill):
    src = DRILL_PATH.read_text()
    head = src.split("class RealModalClient", 1)[0]
    assert "import modal" not in head.replace("import fits_watch", "")
    body = src.split("class RealModalClient", 1)[1].split("\nclass ", 1)[0]
    assert "import modal" in body
    # the drill imports the production successor functions, never copies them
    for fn in ("can_respawn", "same_app_version", "spawn_successor",
               "successor_item"):
        assert f"fw.{fn}(" in src or f".{fn}(" in src, fn
    assert "def can_respawn" not in src and "def spawn_successor" not in src


def test_run_drill_refuses_a_moved_deployment(drill, tmp_path):
    """The deployed function id does NOT change on redeploy
    (sdfbase-fits v10 -> v11 kept both fit ids), so the drill pins the
    deployment version too — the id stays, the app's production version
    moves between spawn and the successor decision, and no successor is
    spawned."""
    items = _items(drill, tmp_path)
    client = FakeClient()
    deps = iter([{"app_id": "ap-drill", "app_version": 1, "deploy_commit": "a"}]
                + [{"app_id": "ap-drill", "app_version": 2,
                    "deploy_commit": "b"}] * 50)
    client.deployment = lambda: next(deps)  # first read pins, later reads differ
    v = _run(drill, client, items)
    d = v["steps"]["d_successor_timeout"]
    assert d["verdict"] == "FAIL" and d["same_app_version"] is False
    assert "redeployed" in d["same_app_version_reason"]
    assert "v1" in d["same_app_version_reason"]
    assert "v2" in d["same_app_version_reason"]
    assert v["steps"]["a_deploy"]["deployment"]["app_version"] == 1
    assert v["pass"] is False
    assert all(n == 1 for name, n, _ in client.spawn_log)


def test_drill_manifest_cell_carries_the_deployment_pin(drill, tmp_path):
    items = _items(drill, tmp_path)
    cell = drill.manifest_cell(items["timeout"]["item"], app="a", function="f",
                               function_id="fu-1", call_id="c", spawn_time="t",
                               deployment={"app_id": "ap-1", "app_version": 4,
                                           "deploy_commit": "abc"})
    assert (cell["app_id"], cell["app_version"], cell["deploy_commit"]) == (
        "ap-1", 4, "abc")
    bare = drill.manifest_cell(items["timeout"]["item"], app="a", function="f",
                               function_id="fu-1", call_id="c", spawn_time="t")
    assert bare["app_version"] is None
    # a client without a deployment reader yields None (a refusal downstream)
    assert drill._client_deployment(object()) is None
