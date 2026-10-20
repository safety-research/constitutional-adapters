"""derived_admission.py: the stager's predicate for a derived object —
same-seed parents identified by sha AND fingerprint, gate liveness NOT FULLY
DEAD on any layer (since 2026-09-18 the recovery verdict and the published
Qwen derived object's own min over layers, ``DEFAULT_FLOOR`` /
``PUBLISHED_REFERENCE``, are recorded ALARMS, no longer refusal grounds; the
published object passes its own reference with no alarm and the earlier 0.933 —
layer 10 read in string-key file order — is pinned as the misread it was),
report.json agreeing with derived_meta, every declared sweep part complete, the
selection interior or the alarm stated; scale_matched for the subtraction —
on synthetic trees: recovered / not recovered, liveness at / above / below the
reference / fully dead, a parent-seed mismatch, a missing sweep part, an edge
without and with the extension. Two non-synthetic inputs:
``tests/fixtures/seedrep/published_qwen_derived_gate_liveness.json`` (the
published object's 32 per-layer live fractions) and
``tests/fixtures/seedrep/distilled_eb64_s1_gate_liveness.json`` (the real
eb64-s1 distilled object's, min 0.2127 at layer 12, none dead)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SEEDREP = REPO / "scripts" / "sdfbase" / "seedrep"
PUBLISHED_LIVENESS = REPO / "tests" / "fixtures" / "seedrep" / \
    "published_qwen_derived_gate_liveness.json"
#: the real qwen35_4b_base_eb64_s1 distilled object's 32 live fractions (min 0.2127 at
#: layer 12, none dead) — the object admitted on 2026-09-18.
REAL_DISTILLED_LIVENESS = REPO / "tests" / "fixtures" / "seedrep" / \
    "distilled_eb64_s1_gate_liveness.json"
MISREAD_FLOOR = 0.933   # the constant before 2026-09-17: layer 10 in string-key file order
if str(SEEDREP) not in sys.path:
    sys.path.insert(0, str(SEEDREP))

import derived_admission as da  # noqa: E402
import derived_sweep as ds  # noqa: E402

KEY = "qwen35_4b_base_eb64_s1"
MIX, CTL = "constv2-mix-full", "constv2ctl2-full"
MATCH = {"model_id": "Qwen/Qwen3.5-4B-Base", "model_revision": "1001bb4d",
         "target_layers": list(range(32)), "positions": "all",
         "response_header_len": None, "template_hash": "a4aee8afcf2e0711"}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _parent(bank: Path, key: str, cell: str, *, seed: int, fp: str, **override) -> tuple[Path, dict]:
    d = bank / key / "cells_all" / cell
    d.mkdir(parents=True)
    obj = cell.encode() * 64
    (d / "object.pt").write_bytes(obj)
    meta = {"cell_id": cell, "object_sha256": _sha(obj), "seed": seed, **MATCH, **override}
    (d / "train_budget_meta.json").write_text(json.dumps(meta))
    (d / "COMPLETE").write_text(json.dumps({"cell_id": cell, "object_sha256": _sha(obj),
                                            "fit_fingerprint": fp}))
    block = {"cell_id": cell, "model_key": key, "object_sha256": _sha(obj), "fit_fingerprint": fp}
    return d, block


def _part(resid: Path, m: float, score: float, *, complete=True):
    k = ds.format_mult(m)
    p = ds.part_path(resid, m)
    p.parent.mkdir(parents=True, exist_ok=True)
    per = {k: {"trainer": {"lr_mult": m}, "identity": {"trainer": {"lr_mult": m}},
               "score_worst_stratum_attainable_ed": score}}
    p.write_text(json.dumps({"sweep": {"lr_mults": [m], "complete": complete, "per_mult": per,
                                       "table": {k: {"score_worst_stratum_attainable_ed": score}},
                                       "budget": {"epochs_cap": 60, "patience": 5}}}))
    return per


def _merged(resid: Path, ladder, scores):
    per, table = {}, {}
    for m in ladder:
        per.update(_part(resid, m, scores[ds.format_mult(m)]))
        table[ds.format_mult(m)] = {"score_worst_stratum_attainable_ed": scores[ds.format_mult(m)]}
    sel = max(ladder, key=lambda m: scores[ds.format_mult(m)])
    (resid / "calibration_controls.json").write_text(json.dumps({
        "sweep": {"lr_mults": [float(m) for m in ladder], "selected_lr_mult": float(sel),
                  "selection": {"verdict": "interior", "reason": "test"},
                  "per_mult": per, "table": table, "complete": True}}))
    return float(sel)


def _distilled(tmp_path, *, seed_pos=1, seed_neg=1, verdict="recovered", liveness_min=0.95,
               liveness=None, ladder=(0.3, 1.0, 3.0), extensions=(), scores=None,
               drop_part=None, key=KEY, assess=True):
    bank = tmp_path / "bank"
    pos_dir, pos = _parent(bank, key, f"{MIX}__psr_ce_all", seed=seed_pos, fp="a" * 64)
    neg_dir, neg = _parent(bank, key, f"{CTL}__psr_ce_all", seed=seed_neg, fp="b" * 64)
    resid = bank / "derived" / "psr_residual" / key / f"{MIX}__psr_ce_all_minus_ctl_distilled"
    resid.mkdir(parents=True)
    scores = scores or {"0.3": 0.5, "1": 0.9, "3": 0.7, "0.1": 0.2, "10": 0.1}
    ds.declare(resid, list(ladder))
    for e in extensions:
        ds.ensure_extension(resid, e)
    full = sorted(set(ladder) | set(extensions))
    sel = _merged(resid, full, scores)
    if assess:
        ds.assess(resid)
    if drop_part is not None:
        ds.part_path(resid, drop_part).unlink()
    obj = b"distilled" * 32
    (resid / "object.pt").write_bytes(obj)
    if liveness is None:
        liveness = {str(i): 1.0 for i in range(32)}
        liveness["9"] = liveness_min
    (resid / "gate_liveness.json").write_text(json.dumps(liveness))
    (resid / "report.json").write_text(json.dumps({"recovery": {"verdict": verdict}}))
    (resid / "derived_meta.json").write_text(json.dumps({
        "kind": "psr", "derived_id": f"{MIX}__psr_ce_all_minus_ctl_distilled", "model_key": key,
        "parent_pos": pos, "parent_neg": neg, "recovery_verdict": verdict,
        "trainer": {"lr_mult": sel, "seed": seed_pos}, "object_sha256": _sha(obj)}))
    return resid, pos_dir, neg_dir


def _subtraction(tmp_path, *, scale_matched=True, seed_neg=1):
    bank = tmp_path / "bank"
    pos_dir, pos = _parent(bank, KEY, f"{MIX}__lora_r4_ce", seed=1, fp="c" * 64)
    neg_dir, neg = _parent(bank, KEY, f"{CTL}__lora_r4_ce", seed=seed_neg, fp="d" * 64)
    d = bank / "derived" / "lora_subtraction" / KEY / f"{MIX}__lora_r4_ce_minus_ctl"
    d.mkdir(parents=True)
    obj = b"lora-diff" * 16
    (d / "object.pt").write_bytes(obj)
    (d / "derived_meta.json").write_text(json.dumps({
        "kind": "lora_subtraction", "derived_id": f"{MIX}__lora_r4_ce_minus_ctl", "model_key": KEY,
        "parent_pos": pos, "parent_neg": neg, "scale_matched": scale_matched,
        "object_sha256": _sha(obj)}))
    return d, pos_dir, neg_dir


def test_recovered_interior_distilled_object_is_admitted(tmp_path):
    resid, p, n = _distilled(tmp_path)
    doc = da.admission(resid, pos_dir=p, neg_dir=n, key=KEY, seed=1)
    assert doc["admitted"], doc["reasons"]
    assert doc["reasons"] == [] and doc["alarms"] == []
    assert doc["recovery_verdict"] == "recovered" and doc["gate_liveness_min"] == 0.95
    assert doc["selected_lr_mult"] == 1.0 and doc["parts_complete"] is True
    assert doc["parents"]["pos"]["fit_fingerprint"] == "a" * 64 and doc["seed"] == 1
    assert doc["sweep"]["ladder"] == [0.3, 1.0, 3.0] and doc["assess"]["on_edge"] is False


def test_not_recovered_is_admitted_with_the_verdict_recorded_as_an_alarm(tmp_path):
    """Since 2026-09-18 the recovery verdict is recorded, not a refusal ground."""
    resid, p, n = _distilled(tmp_path, verdict="not recovered under this trainer + budget")
    doc = da.admission(resid, pos_dir=p, neg_dir=n)
    assert doc["admitted"], doc["reasons"]
    assert doc["reasons"] == []
    assert doc["recovery_verdict"] == "not recovered under this trainer + budget"
    assert len(doc["alarms"]) == 1 and "recovery_verdict 'not recovered under this trainer + budget'" in doc["alarms"][0]
    assert "D69" in doc["alarms"][0]
    assert doc["admission_rule"] == da.ADMISSION_RULE and doc["superseded_rule"] == da.SUPERSEDED_RULE


def test_report_verdict_must_still_agree_with_the_meta_verdict(tmp_path):
    """The report.json cross-check stays a refusal ground."""
    resid, p, n = _distilled(tmp_path, verdict="recovered")
    (resid / "report.json").write_text(json.dumps({"recovery": {"verdict": "not recovered"}}))
    doc = da.admission(resid, pos_dir=p, neg_dir=n)
    assert not doc["admitted"]
    assert any("report.json recovery.verdict 'not recovered' != derived_meta" in r for r in doc["reasons"])


def _published_liveness() -> dict:
    gl = json.loads(PUBLISHED_LIVENESS.read_text())
    assert set(gl) == {str(i) for i in range(32)} and all(isinstance(v, float) for v in gl.values())
    return gl


def test_default_floor_is_the_published_objects_min_over_layers():
    """The pin: DEFAULT_FLOOR is EXACTLY the published object's min over all 32
    layers (layer 9), the value its own gate_liveness_detail.json records as
    min_live_frac — not the first sub-1.0 entry in string-key file order (layer
    10, 0.9334, the misread 0.933 came from), and not rounded (0.8135 rounds up
    and would refuse the object)."""
    gl = _published_liveness()
    true_min = min(float(v) for v in gl.values())
    assert da.DEFAULT_FLOOR == 0.8134719699019494 == true_min
    assert da.liveness_min(gl) == (true_min, "9")
    assert gl["10"] == pytest.approx(0.9334, abs=5e-5) and gl["10"] != true_min
    first_sub_one = next(v for v in gl.values() if v < 1.0)      # file order: "0","1","10",…
    assert first_sub_one == gl["10"] and first_sub_one != true_min
    assert round(true_min, 4) == 0.8135 and true_min < 0.8135    # why the constant is not rounded
    assert sum(v < MISREAD_FLOOR for v in gl.values()) == 4      # L9, L26, L29, L30


def test_published_object_passes_its_own_reference_with_no_alarm_and_the_misread_is_now_an_alarm(tmp_path):
    gl = _published_liveness()
    resid, p, n = _distilled(tmp_path, liveness=gl)
    doc = da.admission(resid, pos_dir=p, neg_dir=n, key=KEY, seed=1)      # default reference
    assert doc["admitted"], doc["reasons"]
    assert doc["alarms"] == [] and doc["layers_below_reference"] == [] and doc["dead_layers"] == []
    assert doc["floor"] == doc["liveness_reference"] == da.DEFAULT_FLOOR == da.PUBLISHED_REFERENCE
    assert doc["gate_liveness_min"] == da.DEFAULT_FLOOR and doc["gate_liveness_min_layer"] == "9"
    assert doc["n_layers"] == 32 and doc["dead_floor"] == 0.0
    # Under the misread 0.933 the published object used to REFUSE itself on 4 layers;
    # the same reference is now an alarm naming those 4 layers.
    under_misread = da.admission(resid, pos_dir=p, neg_dir=n, key=KEY, seed=1, floor=MISREAD_FLOOR)
    assert under_misread["admitted"] and under_misread["reasons"] == []
    assert under_misread["layers_below_reference"] == ["9", "26", "29", "30"]
    assert any("min over layers 0.8135 (layer 9) < published reference 0.933 on 4 of 32 layers" in a
               for a in under_misread["alarms"])


def _real_distilled_liveness() -> dict:
    gl = json.loads(REAL_DISTILLED_LIVENESS.read_text())
    assert set(gl) == {str(i) for i in range(32)}
    return gl


def test_the_real_distilled_object_is_admitted_under_d69_with_both_alarms(tmp_path):
    """The object the ruling was about: gate liveness min 0.2127 at layer 12, 26 of
    32 layers below the published reference, ZERO layers dead; recovery verdict
    'not recovered under this trainer + budget'. Admitted with exactly two alarms."""
    gl = _real_distilled_liveness()
    resid, p, n = _distilled(tmp_path, liveness=gl,
                             verdict="not recovered under this trainer + budget")
    doc = da.admission(resid, pos_dir=p, neg_dir=n, key=KEY, seed=1)
    assert doc["admitted"], doc["reasons"]
    assert doc["reasons"] == [] and len(doc["alarms"]) == 2
    assert doc["gate_liveness_min"] == pytest.approx(0.21267065384712444)
    assert doc["gate_liveness_min_layer"] == "12"
    assert doc["n_layers_below_reference"] == 26 and doc["dead_layers"] == []
    assert doc["recovery_verdict"] == "not recovered under this trainer + budget"
    assert all("D69" in a for a in doc["alarms"])


def test_a_fully_dead_layer_is_the_one_liveness_refusal(tmp_path):
    gl = {str(i): 1.0 for i in range(32)}
    gl["7"], gl["20"] = 0.0, 0.0
    resid, p, n = _distilled(tmp_path, liveness=gl)
    doc = da.admission(resid, pos_dir=p, neg_dir=n)
    assert not doc["admitted"]
    assert doc["dead_layers"] == ["7", "20"]
    assert any("fully dead (live fraction <= 0.0) at layer(s) 7, 20" in r for r in doc["reasons"])
    # a raised dead floor refuses more; the published object's reference never refuses
    strict = da.admission(resid, pos_dir=p, neg_dir=n, dead_floor=0.5)
    assert not strict["admitted"] and strict["dead_layers"] == ["7", "20"]
    alive = {str(i): 1.0 for i in range(32)}
    alive["7"] = 0.05
    resid2, p2, n2 = _distilled(tmp_path / "b", liveness=alive)
    doc2 = da.admission(resid2, pos_dir=p2, neg_dir=n2)
    assert doc2["admitted"] and doc2["layers_below_reference"] == ["7"]
    doc3 = da.admission(resid2, pos_dir=p2, neg_dir=n2, dead_floor=0.05)
    assert not doc3["admitted"] and doc3["dead_layers"] == ["7"]


def test_liveness_min_is_over_every_layer_whatever_the_key_order():
    """A string-key-ordered mapping ("0","1","10",…,"19","2",…,"9") whose lowest
    value sits at a LATER layer than the first sub-1.0 entry: the min is that
    later value, never the first entry / first sub-1.0 entry read in file order."""
    string_order = sorted((str(i) for i in range(32)), key=str)
    assert string_order[:3] == ["0", "1", "10"] and string_order[-1] == "9"
    gl = {k: 1.0 for k in string_order}
    gl["10"], gl["25"], gl["9"] = 0.93, 0.7, 0.5
    assert next(v for v in gl.values() if v < 1.0) == 0.93                # the misread's read
    assert da.liveness_min(gl) == (0.5, "9")
    assert da.liveness_min(dict(reversed(list(gl.items())))) == (0.5, "9")
    assert da.liveness_min({str(i): v for i, v in enumerate([1.0, 0.5, 0.5])}) == (0.5, "1")
    assert da.liveness_min({"3": 0.2, "1": 0.9}) == (0.2, "3")           # int keys, sparse
    with pytest.raises(ValueError):
        da.liveness_min({})
    with pytest.raises((TypeError, ValueError)):
        da.liveness_min({"0": "live"})


@pytest.mark.parametrize("live,admitted,below", [
    (0.95, True, False), (0.90, True, False),      # 0.90 was refused under the misread floor
    (da.DEFAULT_FLOOR, True, False),               # the published object's own min: equality, no alarm
    (0.80, True, True), (0.5, True, True),         # below the reference: ADMITTED with an alarm
    (0.2127, True, True),                          # the real distilled object's min
    (0.0, False, True)])                           # fully dead: the one refusal
def test_gate_liveness_reference_and_dead_floor(tmp_path, live, admitted, below):
    resid, p, n = _distilled(tmp_path, liveness_min=live)
    doc = da.admission(resid, pos_dir=p, neg_dir=n)                        # default reference / dead floor
    assert doc["admitted"] is admitted, doc["reasons"]
    assert doc["gate_liveness_min"] == live and doc["gate_liveness_min_layer"] == "9"
    assert doc["floor"] == doc["liveness_reference"] == da.DEFAULT_FLOOR == 0.8134719699019494
    assert doc["layers_below_reference"] == (["9"] if below else [])
    if not admitted:
        assert doc["dead_layers"] == ["9"]
        assert any("fully dead (live fraction <= 0.0) at layer(s) 9" in r for r in doc["reasons"])
        assert not any("published reference" in a for a in doc["alarms"])   # dead trumps the alarm
    elif below:
        assert doc["reasons"] == [] and doc["dead_layers"] == []
        assert any(f"(layer 9) < published reference {da.DEFAULT_FLOOR} on 1 of 32 layers" in a
                   for a in doc["alarms"])
    else:
        assert doc["alarms"] == []


def test_gate_liveness_min_taken_over_all_layers_in_a_string_ordered_file(tmp_path):
    string_order = sorted((str(i) for i in range(32)), key=str)
    gl = {k: 1.0 for k in string_order}
    gl["10"], gl["9"] = 0.95, 0.5                  # first sub-1.0 in file order passes; layer 9 fails
    resid, p, n = _distilled(tmp_path, liveness=gl)
    assert json.loads((resid / "gate_liveness.json").read_text()) == gl    # order preserved on disk
    doc = da.admission(resid, pos_dir=p, neg_dir=n)
    # below the reference is an alarm, not a refusal — but the MIN is still layer 9's 0.5,
    # never the first sub-1.0 entry in file order.
    assert doc["admitted"] and doc["reasons"] == []
    assert doc["gate_liveness_min"] == 0.5 and doc["gate_liveness_min_layer"] == "9"
    assert doc["layers_below_reference"] == ["9"] and any("(layer 9)" in a for a in doc["alarms"])


def test_parent_seed_mismatch_is_not_admitted(tmp_path):
    resid, p, n = _distilled(tmp_path, seed_pos=1, seed_neg=2)
    doc = da.admission(resid, pos_dir=p, neg_dir=n, seed=1)
    assert not doc["admitted"] and doc["seed"] is None
    assert any("seeds differ" in r for r in doc["reasons"])
    assert any("neg parent seed 2 != expected 1" in r for r in doc["reasons"])


def test_missing_sweep_part_is_not_admitted(tmp_path):
    resid, p, n = _distilled(tmp_path, drop_part=3.0)
    doc = da.admission(resid, pos_dir=p, neg_dir=n)
    assert not doc["admitted"] and doc["parts_complete"] is False
    assert any("parts incomplete for lr_mult ['3']" in r for r in doc["reasons"])


def test_missing_declaration_or_assessment_is_not_admitted(tmp_path):
    resid, p, n = _distilled(tmp_path, assess=False)
    doc = da.admission(resid, pos_dir=p, neg_dir=n)
    assert not doc["admitted"] and any("no assess.json" in r for r in doc["reasons"])
    (resid / "sweep.json").unlink()
    doc = da.admission(resid, pos_dir=p, neg_dir=n)
    assert any("no sweep.json" in r for r in doc["reasons"])


def test_edge_selection_without_extension_refused_with_extension_admitted_with_alarm(tmp_path):
    scores = {"0.3": 0.9, "1": 0.5, "3": 0.4, "0.1": 0.95}
    resid, p, n = _distilled(tmp_path, scores=scores)          # 0.3 wins: bottom edge, no extension
    doc = da.admission(resid, pos_dir=p, neg_dir=n)
    assert not doc["admitted"]
    assert any("edge of [0.3, 1.0, 3.0] and the D7 extension rung (0.1) was never run" in r
               for r in doc["reasons"])
    resid2, p2, n2 = _distilled(tmp_path / "ext", scores=scores, extensions=(0.1,))   # 0.1 wins: still edge
    doc = da.admission(resid2, pos_dir=p2, neg_dir=n2)
    assert doc["admitted"], doc["reasons"]
    assert len(doc["alarms"]) == 1 and "persists after the one D7 extension" in doc["alarms"][0]
    assert doc["selected_lr_mult"] == 0.1 and doc["sweep"]["extensions"] == [0.1]


def test_parent_identity_by_sha_and_fingerprint(tmp_path):
    resid, p, n = _distilled(tmp_path)
    # a re-published parent: same cell id, different bytes
    (p / "object.pt").write_bytes(b"refit")
    meta = json.loads((p / "train_budget_meta.json").read_text())
    meta["object_sha256"] = _sha(b"refit")
    (p / "train_budget_meta.json").write_text(json.dumps(meta))
    marker = json.loads((p / "COMPLETE").read_text())
    marker["object_sha256"] = _sha(b"refit")
    (p / "COMPLETE").write_text(json.dumps(marker))
    doc = da.admission(resid, pos_dir=p, neg_dir=n)
    assert not doc["admitted"]
    assert any("pos parent object_sha256" in r and "different publication" in r for r in doc["reasons"])
    # a fingerprint drift alone is caught too
    resid, p, n = _distilled(tmp_path / "fp")
    marker = json.loads((p / "COMPLETE").read_text())
    marker["fit_fingerprint"] = "e" * 64
    (p / "COMPLETE").write_text(json.dumps(marker))
    doc = da.admission(resid, pos_dir=p, neg_dir=n)
    assert any("pos parent fit_fingerprint" in r for r in doc["reasons"])
    # the distilled object at a multiplier other than the assessed one
    resid, p, n = _distilled(tmp_path / "mult")
    m = json.loads((resid / "derived_meta.json").read_text())
    m["trainer"]["lr_mult"] = 3.0
    (resid / "derived_meta.json").write_text(json.dumps(m))
    doc = da.admission(resid, pos_dir=p, neg_dir=n)
    assert any("trainer.lr_mult 3.0 != assessed selection 1" in r for r in doc["reasons"])


def test_parents_must_match_on_the_shared_tuple_and_key(tmp_path):
    from sisteer.eval.train_budget import DERIVED_PARENT_MATCH_FIELDS
    assert tuple(da.PARENT_MATCH_FIELDS) == tuple(DERIVED_PARENT_MATCH_FIELDS)
    resid, p, n = _distilled(tmp_path)
    meta = json.loads((n / "train_budget_meta.json").read_text())
    meta["template_hash"] = "0000000000000000"
    (n / "train_budget_meta.json").write_text(json.dumps(meta))
    doc = da.admission(resid, pos_dir=p, neg_dir=n)
    assert any("disagree on template_hash" in r for r in doc["reasons"])
    doc = da.admission(resid, pos_dir=p, neg_dir=n, key="qwen35_4b_base_eb64_s2")
    assert any("!= expected key" in r for r in doc["reasons"])
    assert any("is not under …/qwen35_4b_base_eb64_s2/cells_all/" in r for r in doc["reasons"])


def test_subtraction_admission(tmp_path):
    d, p, n = _subtraction(tmp_path)
    doc = da.admission(d, pos_dir=p, neg_dir=n, key=KEY, seed=1)
    assert doc["admitted"] and doc["scale_matched"] is True and doc["floor"] is None
    assert doc["recovery_verdict"] is None and doc["sweep"] is None
    d, p, n = _subtraction(tmp_path / "unmatched", scale_matched=False)
    doc = da.admission(d, pos_dir=p, neg_dir=n)
    assert not doc["admitted"] and any("scale_matched False" in r for r in doc["reasons"])
    d, p, n = _subtraction(tmp_path / "seeds", seed_neg=3)
    assert not da.admission(d, pos_dir=p, neg_dir=n)["admitted"]


def test_cli_writes_admission_json_with_exit_codes(tmp_path):
    resid, p, n = _distilled(tmp_path)
    cmd = [sys.executable, str(SEEDREP / "derived_admission.py"), str(resid),
           "--parents", str(p), str(n), "--floor", repr(da.DEFAULT_FLOOR), "--key", KEY, "--seed", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0 and "ADMITTED" in r.stdout, r.stdout + r.stderr
    doc = json.loads((resid / "ADMISSION.json").read_text())
    assert doc["admitted"] and doc["checker"].endswith("derived_admission.py")
    # below the published reference: ADMITTED, exit 0, the alarm printed
    resid2, p2, n2 = _distilled(tmp_path / "low", liveness_min=0.5)
    r = subprocess.run([sys.executable, str(SEEDREP / "derived_admission.py"), str(resid2),
                        "--parents", str(p2), str(n2)], capture_output=True, text=True)
    assert r.returncode == 0 and "ADMITTED" in r.stdout and "NOT ADMITTED" not in r.stdout
    assert "! ALARM:" in r.stdout and "< published reference" in r.stdout
    low = json.loads((resid2 / "ADMISSION.json").read_text())
    assert low["admitted"] is True and low["layers_below_reference"] == ["9"]
    # a fully dead layer: NOT ADMITTED, exit 4; a raised --dead-floor refuses the 0.5 layer too
    resid3, p3, n3 = _distilled(tmp_path / "dead", liveness_min=0.0)
    r = subprocess.run([sys.executable, str(SEEDREP / "derived_admission.py"), str(resid3),
                        "--parents", str(p3), str(n3)], capture_output=True, text=True)
    assert r.returncode == 4 and "NOT ADMITTED" in r.stdout and "fully dead" in r.stdout
    assert json.loads((resid3 / "ADMISSION.json").read_text())["admitted"] is False
    assert (resid3 / "object.pt").exists()             # preserved, never deleted
    r = subprocess.run([sys.executable, str(SEEDREP / "derived_admission.py"), str(resid2),
                        "--parents", str(p2), str(n2), "--dead-floor", "0.5"],
                       capture_output=True, text=True)
    assert r.returncode == 4 and "fully dead (live fraction <= 0.5) at layer(s) 9" in r.stdout
    r = subprocess.run([sys.executable, str(SEEDREP / "derived_admission.py"), str(tmp_path / "nope"),
                        "--parents", str(p), str(n)], capture_output=True, text=True)
    assert r.returncode == 3


# --- the composed paired write (difference_relu since 2026-09-19) ------------

def _paired_write(tmp_path, *, scale_matched=True, liveness_min=0.31, dead_layer=False, no_liveness=False, seed_neg=1,
                  embedded_tokens=(1000, 1000), embedded_scale_matched=None, blob_kind="psr_paired_write", tamper_parent=False):
    """A REAL torch blob (the admission is bound to the bytes): kind + embedded provenance naming the parents
    (cell_id, object_sha256, data_scale.official_tokens) exactly as compose-v writes them; the sidecar mirrors it unless a test
    forges one side. `embedded_scale_matched` None = consistent with the embedded budgets."""
    import torch
    bank = tmp_path / "bank"
    pos_dir, pos = _parent(bank, KEY, f"{MIX}__psr_ce_all", seed=1, fp="e" * 64)
    neg_dir, neg = _parent(bank, KEY, f"{CTL}__psr_ce_all", seed=seed_neg, fp="f" * 64)
    full = {"budget_meta_sha256": "1" * 64, **MATCH}
    pt, nt = embedded_tokens
    pos_b = {**pos, **full, "data_scale": {"si": MIX, "official_tokens": pt}}
    neg_b = {**neg, **full, "data_scale": {"si": CTL, "official_tokens": nt}}
    derived = pt == nt
    prov = {"kind": blob_kind, "derived_id": f"{MIX}__psr_ce_all_minus_ctl", "model_key": KEY,
            "scale_matched": derived if embedded_scale_matched is None else embedded_scale_matched,
            "parent_pos": pos_b, "parent_neg": neg_b}
    z = torch.zeros(4)
    blob = {"kind": blob_kind, "layers": [0, 1], "provenance": prov,
            "pos": {"directions": {"0": z, "1": z}, "gate_w": {"0": z, "1": z}, "gate_b": {"0": z, "1": z}},
            "neg": {"directions": {"0": z, "1": z}, "gate_w": {"0": z, "1": z}, "gate_b": {"0": z, "1": z}}}
    d = bank / "derived" / "psr_paired_write" / KEY / f"{MIX}__psr_ce_all_minus_ctl"
    d.mkdir(parents=True)
    torch.save(blob, d / "object.pt")
    (d / "derived_meta.json").write_text(json.dumps({
        "kind": "psr_paired_write", "derived_id": f"{MIX}__psr_ce_all_minus_ctl", "model_key": KEY,
        "parent_pos": pos_b, "parent_neg": neg_b, "scale_matched": scale_matched,
        "object_sha256": _sha((d / "object.pt").read_bytes()),
        "derivation": "u_pos(z) - u_neg(z) on the SHARED pre-write layer output z"}))
    if not no_liveness:
        gl = {str(i): 0.9 for i in range(32)}
        gl["6"] = 0.0 if dead_layer else liveness_min
        (d / "gate_liveness.json").write_text(json.dumps(gl))
    if tamper_parent:   # the bank's parent bytes are not the object the derivation consumed (sidecar says otherwise)
        (pos_dir / "object.pt").write_bytes(b"not-the-parent" * 32)
    return d, pos_dir, neg_dir


def test_paired_write_is_admitted_with_the_floor_alarm_and_recovery_not_applicable(tmp_path):
    d, p, n = _paired_write(tmp_path)
    doc = da.admission(d, pos_dir=p, neg_dir=n, key=KEY, seed=1)
    assert doc["admitted"], doc["reasons"]
    assert doc["derived_kind"] == "psr_paired_write" and doc["scale_matched"] is True
    # condition 2: the reference alarm travels on the record
    assert any("< published reference" in a for a in doc["alarms"])
    assert doc["gate_liveness_min"] == 0.31 and doc["gate_liveness_min_layer"] == "6"
    assert doc["n_layers_below_reference"] == 1 and doc["dead_layers"] == []
    assert doc["liveness_reference"] == da.DEFAULT_FLOOR and doc["dead_floor"] == da.DEFAULT_DEAD_FLOOR
    # condition 1: recovery is stated NOT APPLICABLE with its reason — never omitted, never "passed"
    assert doc["recovery_verdict"] == da.RECOVERY_NOT_APPLICABLE == "not_applicable"
    assert doc["recovery_applicability"] == {"applies": False, "reason": da.RECOVERY_NOT_APPLICABLE_REASON}
    assert "no distillation trainer" in doc["recovery_applicability"]["reason"]
    assert doc["admission_rule"] == da.PAIRED_WRITE_RULE and doc["superseded_rule"] is None
    assert doc["sweep"] is None and doc["assess"] is None
    # bound to the bytes: the shared validator ran clean, the blob's kind selected the branch, scale matching was DERIVED
    b = doc["bytes_bound"]
    assert b["shared_validator_problems"] == 0 and b["blob_kind"] == "psr_paired_write"
    assert b["embedded_official_tokens"] == {"pos": 1000, "neg": 1000} and b["scale_matched_derived"] is True
    assert doc["parents"]["pos"]["object_bytes_verified"] is True and doc["parents"]["neg"]["object_bytes_verified"] is True


def test_forgery_1_sidecar_scale_matched_true_over_embedded_budgets_that_differ_is_refused(tmp_path):
    """Reproduction 1: a composed paired object whose EMBEDDED provenance says scale_matched False (parent
    budgets 100 / 200) must not become ADMITTED by flipping only the sidecar flag to True."""
    d, p, n = _paired_write(tmp_path, scale_matched=True, embedded_tokens=(100, 200), embedded_scale_matched=False)
    doc = da.admission(d, pos_dir=p, neg_dir=n, key=KEY, seed=1)
    assert not doc["admitted"]
    assert any("sidecar claims True but the embedded parent budgets are 100 vs 200" in r for r in doc["reasons"]), doc["reasons"]
    assert any("scale_matched False by the bound evidence" in r for r in doc["reasons"])
    assert doc["scale_matched"] is False and doc["bytes_bound"]["scale_matched_derived"] is False
    # and an embedded flag that contradicts its own embedded budgets is named too
    d2, p2, n2 = _paired_write(tmp_path / "flag", scale_matched=True, embedded_tokens=(100, 200), embedded_scale_matched=True)
    doc2 = da.admission(d2, pos_dir=p2, neg_dir=n2)
    assert not doc2["admitted"] and any("embedded flag True contradicts the embedded budgets" in r for r in doc2["reasons"])


def test_forgery_2_a_plain_psr_object_relabelled_psr_paired_write_in_its_sidecar_is_refused(tmp_path):
    """Reproduction 2: a correctly hashed `psr` object whose sidecar says psr_paired_write must not be
    ADMITTED with recovery not_applicable and no report / sweep / assessment — the branch is selected from the bytes."""
    d, p, n = _paired_write(tmp_path, blob_kind="psr")
    doc = da.admission(d, pos_dir=p, neg_dir=n, key=KEY, seed=1)
    assert not doc["admitted"]
    assert any("artifact kind 'psr' != sidecar kind 'psr_paired_write'" in r for r in doc["reasons"]), doc["reasons"]
    assert any("the object IS a 'psr', the sidecar says 'psr_paired_write'" in r for r in doc["reasons"])
    assert doc["bytes_bound"]["blob_kind"] == "psr"


def test_a_parent_whose_bank_bytes_are_not_the_recorded_object_is_refused(tmp_path):
    d, p, n = _paired_write(tmp_path, tamper_parent=True)
    doc = da.admission(d, pos_dir=p, neg_dir=n, key=KEY, seed=1)
    assert not doc["admitted"] and any("pos parent object.pt hashes to" in r for r in doc["reasons"])
    assert doc["parents"]["pos"]["object_bytes_verified"] is False


def test_paired_write_records_recovery_differently_from_a_distilled_object(tmp_path):
    dist, p, n = _distilled(tmp_path / "d")
    doc = da.admission(dist, pos_dir=p, neg_dir=n)
    assert doc["recovery_applicability"] == {"applies": True, "reason": None} and doc["recovery_verdict"] == "recovered"
    sub, p2, n2 = _subtraction(tmp_path / "s")
    assert da.admission(sub, pos_dir=p2, neg_dir=n2)["recovery_applicability"] is None


def test_paired_write_fully_dead_layer_is_the_one_liveness_refusal(tmp_path):
    d, p, n = _paired_write(tmp_path, dead_layer=True)
    doc = da.admission(d, pos_dir=p, neg_dir=n, key=KEY, seed=1)
    assert not doc["admitted"] and any("fully dead" in r for r in doc["reasons"]) and doc["dead_layers"] == ["6"]


def test_paired_write_missing_liveness_or_scale_match_refuses(tmp_path):
    d, p, n = _paired_write(tmp_path / "nolive", no_liveness=True)
    doc = da.admission(d, pos_dir=p, neg_dir=n)
    assert not doc["admitted"] and any("gate_liveness.json absent" in r for r in doc["reasons"])
    # a sidecar that UNDERclaims (False over equal embedded budgets) is refused too: any unbound claim refuses, in either direction
    d, p, n = _paired_write(tmp_path / "unmatched", scale_matched=False)
    doc = da.admission(d, pos_dir=p, neg_dir=n)
    assert not doc["admitted"] and any("sidecar claims False but the embedded parent budgets are 1000 vs 1000" in r for r in doc["reasons"])
    assert doc["bytes_bound"]["scale_matched_derived"] is True
    d, p, n = _paired_write(tmp_path / "seeds", seed_neg=2)
    assert not da.admission(d, pos_dir=p, neg_dir=n)["admitted"]


def test_paired_write_cli_admits_with_the_alarm_printed(tmp_path):
    d, p, n = _paired_write(tmp_path)
    out = tmp_path / "ADMISSION.json"
    r = subprocess.run([sys.executable, str(SEEDREP / "derived_admission.py"), str(d), "--parents", str(p), str(n),
                        "--key", KEY, "--seed", "1", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0 and "ADMITTED" in r.stdout and "NOT ADMITTED" not in r.stdout, r.stdout + r.stderr
    assert "! ALARM:" in r.stdout and "< published reference" in r.stdout
    doc = json.loads(out.read_text())
    assert doc["admitted"] and doc["recovery_verdict"] == "not_applicable"


def test_an_unknown_derived_kind_still_refuses_by_name(tmp_path):
    d, p, n = _paired_write(tmp_path)
    meta = json.loads((d / "derived_meta.json").read_text()); meta["kind"] = "psr_something_else"
    (d / "derived_meta.json").write_text(json.dumps(meta))
    with pytest.raises(da.InputError):
        da.admission(d, pos_dir=p, neg_dir=n)
