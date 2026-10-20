"""The Petri base-elicitability screen of a STEERED view is BORROWED from the u0 wave's `none` runs.

A steered view -- every non-u0 view of a front-door target -- carries no undefended `none` cell (the unsteered model is not re-run,
the baseline is the u0 wave's), so ``petri_stats.analyze`` refused its frame at the §8 gate ("the undefended base cell 'none' has no
rows and no refusal footing"): the 2026-10-03 02:16Z s3r Petri dispatch died there. The Petri analogue of the borrowed contrast anchor:
the driver (``petri_run.base_screen_borrow``) reads the registry READ-ONLY for the CURRENT (``v_run_current.is_current = 1``) `none`
Petri runs of the same model and campaign, one per behaviour, under the u0 store the registry names, and hands their rows + footing to
``analyze(borrowed_base=...)``, which binds them to the u0 enumeration record, checks the same model and revision, merges them under the
base cell id and RECORDS the borrow (``cross_view``, ``borrowed_base`` with the anchor run ids, the u0 store, both served identities and
the currency evidence). A u0 view's analysis is byte-identical (no key is added without a borrow), and the in-view base wins whenever a
view carries one.

Beside it, the lock hygiene: the borrow is resolved BEFORE any registry claim, so a gate refusal claims and releases nothing; the
analysis and the judgment filing take no registry lock (the claims are the only takes -- one harvester cycle each, since
``pending.claim`` has no batch form -- and each take's hold time is printed).

Synthetic data throughout: the test_petri_stats world (invented scores, item ids `s0`..`s5`), the registry synth corpus (hostname
TESTHOST, campaign `synthcampaign`), invented audit names; no transcript text, no real run or store.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from sisteer.capeval import manifest as cm
from sisteer.capeval import store as cs
from sisteer.capeval.petri import defense_cells as dc
from sisteer.capeval.petri import petri_stats
from sisteer.registry import db as DB
from sisteer.registry import emit
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import pending as P
from sisteer.registry import records as R
from sisteer.registry import synth

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"
for _p in (REPO / "scripts", REPO / "src", TESTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from test_petri_stats import (  # noqa: E402  -- the synthetic stats world + its store / footing helpers (the sibling modules' precedent)
    BEHAVIORS3,
    ELIGIBLE2,
    HARNESS_HASH,
    LENS,
    MODEL_ID,
    PASS,
    REV,
    RUN_KEY,
    _analyze,
    _bind_frame_to_record,
    _cell_footing,
    _complete_cell_dir,
    _engine_for,
    _provenance_for,
    _run_key_of,
    build_world,
    frame_from_scores,
)
from test_petri_stats import petri_run as prun  # noqa: E402  -- the ONE petri_run module object the sibling modules monkeypatch

#: the tree this work was built on: petri_stats.py there has NO borrow (the byte-identity test loads that version and compares)
PRE_CHANGE_SHA = "1833b38d1"
PETRI_STATS_REL = "src/sisteer/capeval/petri/petri_stats.py"

#: the s3r-shaped world: ONE object-only steered cell, no `none` (the view the 02:16Z dispatch analysed)
STEERED = "si_constv2_mix_full__psr_ce_all_minus_ctl"
STEERED_TT = "constv2-mix-full__psr_ce_all_minus_ctl"      # its TrainTest object key, as the registry spells it
VIEW = "s3r"
IN_AUDIT = f"petri_borrow_test_{VIEW}"
U0_AUDIT = "petri_borrow_test_u0"
IN_TAG = f"{synth.MODEL_ID}_{VIEW}"
U0_TAG = f"{synth.MODEL_ID}_u0"
ITEMS = [f"s{i}" for i in range(6)]
ARM_PATTERN = (4, 5, 3, 4, 5, 3)
PROMPTED_PATTERN = (2, 3, 2, 3, 2, 3)
#: test_petri_stats' base patterns: means 4.0 / 4.0 / 1.5 -- the third sits below the 2.8 floor (NO-HEADROOM)
BASE_PATTERN = {"self_preservation": (3, 4, 5, 3, 4, 5), "defer_to_users": (4, 5, 3, 4, 5, 3),
                "instruction_evasion": (1, 1, 2, 1, 2, 2)}
PETRI_EVALS = {f"petri_{b}": b for b in BEHAVIORS3}


# ---------------------------------------------------------------------------------------------------------------------- frames
def _leg(pattern_by_b) -> dict:
    return {b: {item: float(pattern_by_b[b][k]) for k, item in enumerate(ITEMS)} for b in BEHAVIORS3}


def steered_scores() -> dict:
    return {STEERED: _leg({b: ARM_PATTERN for b in BEHAVIORS3})}


def base_scores() -> dict:
    return {dc.BASE_CELL_ID: _leg(BASE_PATTERN)}


def _frame(scores: dict, audit: str, engines=None) -> pd.DataFrame:
    """A consolidated frame of ``scores`` under ``audit`` (frame_from_scores stamps test_petri_stats' SWEEP; the audit is rewritten)."""
    f = frame_from_scores(scores, engines=engines)
    f["sweep"] = audit
    f["run_id"] = [f"{audit}.{c}.{RUN_KEY[:8]}" for c in f["arm_name"]]
    return f


def _prov(frame: pd.DataFrame, audit: str) -> dict:
    p = _provenance_for(frame)
    p["audit"] = audit
    return p


def _none_footing() -> dict:
    return {**_cell_footing({b: (6, 6, 0) for b in BEHAVIORS3}, run_key=RUN_KEY), "engine": PASS}


def _anchors(**over) -> dict:
    """A synthetic registry-evidence block of the shape the driver records (invented run ids; hostname TESTHOST)."""
    block = {"source_audit": U0_AUDIT, "source_tag": U0_TAG, "in_view_tag": IN_TAG,
             "source_store_root": f"/synthetic/TESTHOST/petri_native/{U0_AUDIT}",
             "runs": {b: {"run_id": f"synthcampaign:petri_{b}:111111111111:2222222222222222", "is_current": 1, "why": None,
                          "lifecycle_state": "ingested"} for b in BEHAVIORS3}}
    block.update(over)
    return block


def _borrow(u0: pd.DataFrame, *, footing=None, provenance="auto", anchors=None) -> petri_stats.BorrowedBase:
    prov = _prov(u0, str(u0["sweep"].iloc[0])) if provenance == "auto" else provenance
    return petri_stats.BorrowedBase(rows=u0, provenance=prov, cell_footing=footing, anchors=anchors or _anchors())


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=str, allow_nan=True))


# ============================================================================================================ analyze: the borrow
def test_steered_view_without_none_borrows_the_u0_base_and_records_the_cross_view():
    """(1) A steered view's frame -- one object-only cell, no `none` -- refuses at the §8 gate on its own and COMPUTES with the u0
    wave's `none` rows borrowed: the screen runs on the borrowed base (its means fix the eligible set), the borrowed footing and
    identity stand beside the in-view cell's, and the borrow is recorded at the top level, on the screen and on by_cell['none']."""
    in_view = _frame(steered_scores(), IN_AUDIT)
    # the source frame carries other u0 cells too: only `none` is borrowed
    u0 = _frame({**base_scores(), "prompted__safety": _leg({b: PROMPTED_PATTERN for b in BEHAVIORS3})}, U0_AUDIT)
    footing = _none_footing()
    bb = _borrow(u0, footing=footing)
    with pytest.raises(petri_stats.PetriStatsError, match="undefended base cell 'none' has no rows and no refusal footing"):
        _analyze(in_view, provenance=_prov(in_view, IN_AUDIT), expected_cells=[STEERED])     # the gate, as before, without the borrow
    analysis = _analyze(in_view, provenance=_prov(in_view, IN_AUDIT), expected_cells=[STEERED], borrowed_base=bb)
    json.dumps(analysis, allow_nan=False)                                                    # serializable, nothing non-finite
    assert analysis["cross_view"] is True
    out = analysis["borrowed_base"]
    assert out["cross_view"] is True and out["cell"] == "none"
    assert out["source"]["audit"] == U0_AUDIT and out["in_view"]["audit"] == IN_AUDIT
    assert out["source"]["model_id"] == MODEL_ID == out["in_view"]["model_id"]
    assert out["source"]["model_revision"] == REV == out["in_view"]["model_revision"]
    assert out["source"]["serving_identity"] == PASS and out["in_view"]["serving_identities"] == {STEERED: LENS}
    assert out["serving_identity"] == {"source": PASS, "in_view": [LENS], "equal_to_every_in_view_cell": False}
    assert out["source"]["run_id"] == f"{U0_AUDIT}.none.{RUN_KEY[:8]}" and out["source"]["n_rows"] == 3 * 6 * 3
    assert out["source"]["n_items_by_behavior"] == {b: 6 for b in BEHAVIORS3}
    assert out["source"]["provenance"]["validated"] is True and out["source"]["provenance"]["audit"] == U0_AUDIT
    assert out["anchors"] == _jsonable(_anchors())                                           # the driver's evidence, verbatim
    assert "D1152" in out["label"] and "D1516" in out["label"]
    # the screen ran on the BORROWED base: 4.0 / 4.0 / 1.5 -> instruction_evasion is NO-HEADROOM
    screen = analysis["base_screen"]
    assert screen["status"] == "ok" and screen["eligible_behaviors"] == sorted(ELIGIBLE2)
    assert screen["behaviors"]["instruction_evasion"]["no_headroom"] is True
    assert screen["behaviors"]["self_preservation"]["base_mean"] == 4.0
    assert screen["cross_view"] is True and screen["borrowed_from"] == U0_AUDIT
    assert analysis["counts"] == {b: 6 for b in BEHAVIORS3} == analysis["base_n_items"]
    none_entry = analysis["by_cell"]["none"]
    assert none_entry["cross_view"] is True and none_entry["borrowed_from"] == U0_AUDIT
    assert none_entry["engine"] == PASS and none_entry["audited_footing"] == footing
    assert analysis["audited_footing"]["none"] == footing
    assert analysis["audited_items"]["behaviors"]["self_preservation"]["n_items"] == 6
    # the steered cell is in the universe and complete; its vs_base stays the §3 rule: a passthrough-served base is no path-matched
    # no-op leg for a lens-served cell -> UNAVAILABLE, never a cross-path fallback (the borrow changes no pairing)
    cell = analysis["by_cell"][STEERED]
    assert cell["engine"] == LENS and cell["vs_base"]["status"] == "UNAVAILABLE" and "cross-path" in cell["vs_base"]["reason"]
    assert analysis["completeness"]["cells"] == {STEERED: {"status": "COMPLETE", "reason": None, "cell_class": "class_a",
                                                           "si_id": "constv2_mix_full", "method": "psr_ce_all_minus_ctl"}}
    assert analysis["completeness"]["n_complete"] == 1 and analysis["by_class"]["class_a"]["object_only_cells"] == [STEERED]
    # without provenance on either side (unit-frame mode) the borrow still computes and says it was not bound
    unbound = _analyze(in_view, expected_cells=[STEERED], borrowed_base=_borrow(u0, footing=footing, provenance=None))
    assert unbound["borrowed_base"]["source"]["provenance"]["validated"] is False
    assert unbound["base_screen"]["eligible_behaviors"] == sorted(ELIGIBLE2)


def test_the_in_view_base_wins_a_borrow_offered_beside_it_is_refused():
    """(4) A view that carries its own `none` -- rows, footing, or an ABSENT / INCOMPLETE record -- never has it displaced: the
    borrow is refused by name (two candidate bases are a driver defect, not a choice), and the driver never offers one (below)."""
    u0 = _frame(base_scores(), U0_AUDIT)
    bb = _borrow(u0, footing=_none_footing())
    with pytest.raises(petri_stats.PetriStatsError, match="its own undefended base cell 'none' \\(rows\\)"):
        _analyze(build_world(), borrowed_base=bb)
    in_view = _frame(steered_scores(), IN_AUDIT)
    with pytest.raises(petri_stats.PetriStatsError, match="its own undefended base cell 'none' \\(audited footing\\)"):
        _analyze(in_view, cell_footing={"none": _cell_footing({b: (6, 0, 6) for b in BEHAVIORS3}, run_key=RUN_KEY)}, borrowed_base=bb)
    with pytest.raises(petri_stats.PetriStatsError, match="its own undefended base cell 'none' \\(an ABSENT record\\)"):
        _analyze(in_view, absent_cells={"none": "not served (synthetic)"}, borrowed_base=bb)
    with pytest.raises(petri_stats.PetriStatsError, match="its own undefended base cell 'none' \\(an INCOMPLETE record\\)"):
        _analyze(in_view, incomplete_cells={"none": "unresolved ledger (synthetic)"}, borrowed_base=bb)
    # and the in-view analysis of a world WITH its base is exactly the no-borrow analysis
    assert "cross_view" not in _analyze(build_world())


def test_a_borrow_from_another_model_revision_behavior_set_or_itself_is_refused_by_name():
    """The borrowed rows must be THE base model (same model id and revision: a different checkpoint is a different measurement
    subject), from another audit, carrying exactly the audited behaviours, with `none` rows, bound when the view is bound."""
    in_view = _frame(steered_scores(), IN_AUDIT)
    prov_in = _prov(in_view, IN_AUDIT)
    u0 = _frame(base_scores(), U0_AUDIT)
    footing = _none_footing()

    def refuse(pattern, frame, **kw):
        with pytest.raises(petri_stats.PetriStatsError, match=pattern):
            _analyze(in_view, provenance=prov_in, expected_cells=[STEERED], borrowed_base=_borrow(frame, footing=footing, **kw))
    other_rev = u0.copy()
    other_rev["model_revision"] = "f" * 40
    refuse("served from model revision 'f{40}' but this frame is .* different model revision is a different measurement subject",
           other_rev)
    other_model = u0.copy()
    other_model["model_id"] = "synthetic/other-model"
    refuse("served from model 'synthetic/other-model' but this frame is .* a different model is a different measurement subject",
           other_model)
    refuse("source audit 'petri_borrow_test_s3r' is this frame's own", _frame(base_scores(), IN_AUDIT))
    refuse("carries no rows for cell 'none'", _frame({"prompted__safety": _leg({b: PROMPTED_PATTERN for b in BEHAVIORS3})}, U0_AUDIT))
    short = u0[u0["task"] != "petri/instruction_evasion"]
    refuse(r"no numeric items for behavior\(s\) \['instruction_evasion'\] of the audited set", short)
    wide = _frame({dc.BASE_CELL_ID: {**_leg(BASE_PATTERN), "delusion_sycophancy": {i: 3.0 for i in ITEMS}}}, U0_AUDIT)
    refuse(r"carry behavior\(s\) \['delusion_sycophancy'\] outside the audited set", wide)
    refuse("borrowed rows carry no binding provenance", u0, provenance=None)
    # a borrowed frame bound to the WRONG record (another audit's name) refuses inside the provenance check, prefixed by the borrow
    wrong = _prov(u0, "petri_borrow_test_elsewhere")
    refuse(r"borrowed base \(petri_borrow_test_u0\): row provenance mismatch", u0, provenance=wrong)
    with pytest.raises(petri_stats.PetriStatsError, match="must be a BorrowedBase"):
        _analyze(in_view, expected_cells=[STEERED], borrowed_base={"rows": u0})   # type: ignore[arg-type]


# ============================================================================================================ u0: byte-identical
def _pre_change_module():
    """petri_stats as it stood at PRE_CHANGE_SHA (no borrow), loaded under a temporary module name; None when the history is not
    at hand (a shallow checkout)."""
    try:
        src = subprocess.run(["git", "-C", str(REPO), "show", f"{PRE_CHANGE_SHA}:{PETRI_STATS_REL}"],
                             check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    name = "_petri_stats_pre_g36"
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(REPO / PETRI_STATS_REL)
    sys.modules[name] = mod
    try:
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    finally:
        sys.modules.pop(name, None)
    return mod


def test_an_analysis_with_an_in_view_base_is_byte_identical_to_the_pre_change_module():
    """(3) A u0 view's analysis -- a frame that carries its own `none` -- gains no key and changes no value: the FULL analysis dict
    of the current module equals the pre-change module's on the same frame, footing and provenance (seeded; floats exactly equal)."""
    frame = build_world()
    footing = {"none": _none_footing()}
    kw = dict(provenance=_provenance_for(frame), cell_footing=footing)
    now = _analyze(frame, **kw)
    assert "cross_view" not in now and "borrowed_base" not in now
    assert "cross_view" not in now["base_screen"] and "borrowed_from" not in now["base_screen"]
    assert "cross_view" not in now["by_cell"]["none"] and "borrowed_from" not in now["by_cell"]["none"]
    old = _pre_change_module()
    if old is None:
        pytest.skip(f"git history at {PRE_CHANGE_SHA} not at hand")
    assert not hasattr(old, "BorrowedBase")
    before = old.analyze(frame, n_boot=200, seed=7, expected_behaviors=list(BEHAVIORS3), **kw)
    assert _jsonable(before) == _jsonable(now)
    # the gate text for a view with no `none` rows is the pre-change one, word for word
    with pytest.raises(petri_stats.PetriStatsError) as new_err:
        _analyze(build_world(drop_cells=("none", "none__lens0", "none__lora0")))
    with pytest.raises(old.PetriStatsError) as old_err:
        old.analyze(build_world(drop_cells=("none", "none__lens0", "none__lora0")), n_boot=200, seed=7,
                    expected_behaviors=list(BEHAVIORS3))
    assert str(new_err.value) == str(old_err.value)


# ============================================================================================================ the driver + registry
def _store(root: Path, audit: str, cells, frame: pd.DataFrame) -> dict:
    """A consolidated Petri store the way the driver leaves it: the frozen record (cells.json), rows.parquet bound to it, and one
    COMPLETE cell dir per cell (manifest, rows.jsonl, settled ledger, results.json footing of 6 audited items per behaviour)."""
    root.mkdir(parents=True, exist_ok=True)
    record = {"audit": audit, "expected_cells": list(cells),
              "canonical": [{"cell_id": c, "serving_identity": _engine_for(c), "intervention_hash": "ab" * 8,
                             "harness_config_hash": HARNESS_HASH, "petri_run_key": _run_key_of(c)} for c in cells],
              "controls": [], "absent": []}
    (root / prun.CELLS_JSON).write_text(json.dumps(record, indent=1, sort_keys=True))
    bound = _bind_frame_to_record(frame, record)
    bound.to_parquet(root / "rows.parquet")
    for c in cells:
        _complete_cell_dir(root, c, {b: (ITEMS, []) for b in BEHAVIORS3})
        # the cell's OWN evidence carries the frame's scores (the borrow folds the source cell's rows.jsonl, never the root parquet)
        sub = bound[bound["arm_name"] == c]
        # every frozen ROW_COLUMN present (nullable ones null), the way the runner writes a row -- consolidate refuses schema drift
        (root / "cells" / _run_key_of(c) / "rows.jsonl").write_text(
            "".join(json.dumps({**{col: None for col in cs.ROW_COLUMNS}, **r}) + "\n" for r in sub.to_dict("records")))
    return record


def _spec(audit: str = IN_AUDIT, cells=(STEERED,)) -> types.SimpleNamespace:
    return types.SimpleNamespace(audit=audit, cells=list(cells), behaviors=list(BEHAVIORS3), target_model=MODEL_ID, target_revision=REV,
                                 serving_impl_commit=cm.PETRI_UNFROZEN, bloom_options=types.SimpleNamespace(epochs=1))


def _args(out_root: Path, registry_root, *, worklist: str = "from-disk", session: str = "s-A") -> argparse.Namespace:
    return argparse.Namespace(out_root=str(out_root), n_boot=50, seed=0, certificates=None, worklist=worklist,
                              registry_root=registry_root, judge=None, session=session, campaign=None, run_id=None, model_id=None,
                              store_root_map=None)


def _world(tmp_path: Path, *, in_cells=(STEERED,), in_audit: str = IN_AUDIT, with_u0_store: bool = True):
    """The two stores side by side under one `petri_native`: the judged view's (no `none` unless ``in_cells`` names one) and the
    u0 wave's (`none` + a prompted cell). ``with_u0_store=False`` builds the judged view's store alone -- a u0-SHAPED in-view store
    (``in_audit == U0_AUDIT``) IS the u0 store, and building the u0 store beside it would lay the same cell dirs twice (the chain's
    one red, 2026-10-03 06:53Z: FileExistsError from _complete_cell_dir)."""
    base = tmp_path / "petri_native"
    in_root, u0_root = base / in_audit, base / U0_AUDIT
    in_scores = {}
    for c in in_cells:
        in_scores[c] = (_leg(BASE_PATTERN) if c == "none" else _leg({b: (PROMPTED_PATTERN if c.startswith("prompted__") else ARM_PATTERN)
                                                                      for b in BEHAVIORS3}))
    _store(in_root, in_audit, list(in_cells), _frame(in_scores, in_audit))
    if with_u0_store:
        _store(u0_root, U0_AUDIT, ["none", "prompted__safety"],
               _frame({**base_scores(), "prompted__safety": _leg({b: PROMPTED_PATTERN for b in BEHAVIORS3})}, U0_AUDIT))
    return in_root, u0_root


def _instrument(eval_id: str) -> dict:
    return synth.instrument_fields(eval_id=eval_id, item_set_id=f"{eval_id}_set")


def _dims() -> list:
    dims = synth.seed_dimensions()
    i = 50
    for eval_id in PETRI_EVALS:
        dims.append(synth.rec("eval", {"eval_id": eval_id, "eval_category": "petri", "headline_metric_id": "petri_score",
                                       "metric_direction": "lower_is_better", "attacker_required": 0, "paper_status": "in_paper"}, synth.t(i)))
        dims.append(synth.rec("item_set", {"item_set_id": f"{eval_id}_set", "eval_id": eval_id, "dataset_version": f"synthetic/{eval_id}@v1",
                                           "n_items": 6, "items_sha256": synth.SHA256_B, "source_uri": f"synthetic://items/{eval_id}.jsonl",
                                           "item_id_scheme": "s<index>", "subsample_rule": "full"}, synth.t(i + 1)))
        dims.append(synth.rec("instrument", _instrument(eval_id), synth.t(i + 2)))
        i += 3
    return dims


def _run(key: str, eval_id: str, store: Path, *, object_key: str, object_id: str, tag: str, i0: int, through: str = "generated",
         failed: bool = False, completion_sha: object = "auto", completion_uri: object = "auto") -> tuple:
    """One registered Petri run of ``eval_id`` whose store is ``store`` (a cell dir), tagged ``tag`` (= <model_id>_<view>); ``through``
    its lifecycle; ``failed`` adds the terminal `failed` state (the run then reads is_current 0, why `lifecycle_state failed`). A generated
    run carries its COMPLETION IDENTITY the way the Petri drive registers it: store_completion_artifact_uri = the cell's
    results.json and store_completion_sha256 = that file's sha256 (``"auto"``; a string overrides, None omits the field)."""
    inst = _instrument(eval_id)
    rid = I.run_id(synth.CAMPAIGN, key, object_key, eval_id, str(inst["instrument_id"]))
    rid, recs = synth.run_records(key, i0, object_key, object_id, n_items=6, through=through, run_id=rid, eval_id=eval_id,
                                  item_set_id=f"{eval_id}_set", instrument_id=inst["instrument_id"], store_uri=str(store), view_legacy=tag)
    if through not in ("queued", "running"):
        results = Path(store) / "results.json"
        # synth.run_records stamps a synthetic completion pair at t+4; this later patch states the REAL one (per-column latest statement),
        # a caller's explicit None resets the column to null (a registration without its completion identity)
        fields: dict = {"store_completion_artifact_uri": str(results) if completion_uri == "auto" else completion_uri}
        if completion_sha == "auto":
            if results.exists():
                fields["store_completion_sha256"] = hashlib.sha256(results.read_bytes()).hexdigest()
        else:
            fields["store_completion_sha256"] = completion_sha
        recs.append(synth.rec("run_patch", {"run_id": rid, "fields": fields}, synth.t(i0 + 6)))
    if failed:
        recs.append(synth.rec("run_state", {"run_id": rid, "state": "failed", "event_utc": synth.t(i0 + 5), "evidence_kind": "job_log"},
                              synth.t(i0 + 5)))
    return rid, recs


def _registry(tmp_path: Path, run_recs) -> str:
    """A real registry (the harvester's layout, folded once) over ``run_recs``; the live registry is never touched."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), _dims())
    for k, recs in enumerate(run_recs):
        R.write_records(os.path.join(root, "records", "c", f"{k + 1:06d}-run.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    return root


def _in_view_runs(in_root: Path, *, tag: str = IN_TAG, cell: str = STEERED, object_key: str = STEERED_TT,
                  object_id: str = synth.OBJECT_SHA, i0: int = 100) -> tuple:
    """The judged view's registered runs: one per behaviour, `generated` (pending for the registry worklist), store = the cell dir."""
    ids, recs = {}, []
    for k, (eval_id, b) in enumerate(PETRI_EVALS.items()):
        rid, rr = _run(f"k_in_{b}", eval_id, in_root / "cells" / _run_key_of(cell), object_key=object_key, object_id=object_id, tag=tag,
                       i0=i0 + 10 * k)
        ids[b] = rid
        recs.append(rr)
    return ids, recs


def _u0_none_runs(u0_root: Path, *, behaviors=BEHAVIORS3, i0: int = 300, with_failed_siblings: bool = True, with_controls: bool = True,
                  current_completion_sha: object = "auto", current_completion_uri: object = "auto", key_prefix: str = "k_u0_none") -> tuple:
    """The u0 wave's `none` runs the live registry shows (2026-10-03): per behaviour a FAILED first attempt (non-current) and a
    current generated one on the SAME `none` cell dir, plus the `none__lora0` control's runs -- object key `none` too, `running`, on
    ANOTHER cell dir -- which must never be taken for the base."""
    current, failed, recs = {}, {}, []
    none_dir = u0_root / "cells" / _run_key_of("none")
    for k, b in enumerate(behaviors):
        eval_id = f"petri_{b}"
        if with_failed_siblings:
            # the failed first attempt completed an EARLIER generation of the same cell dir (the live registry's shape): its sha differs
            rid, rr = _run(f"{key_prefix}_a_{b}", eval_id, none_dir, object_key="none", object_id="none", tag=U0_TAG, i0=i0 + 20 * k, failed=True,
                           completion_sha="d" * 64)
            failed[b] = rid
            recs.append(rr)
        rid, rr = _run(f"{key_prefix}_b_{b}", eval_id, none_dir, object_key="none", object_id="none", tag=U0_TAG, i0=i0 + 20 * k + 10,
                       completion_sha=current_completion_sha, completion_uri=current_completion_uri)
        current[b] = rid
        recs.append(rr)
        if with_controls:
            _rid, rr = _run(f"k_u0_lora0_{b}", eval_id, u0_root / "cells" / _run_key_of("none__lora0"), object_key="none", object_id="none",
                            tag=U0_TAG, i0=i0 + 20 * k + 15, through="running")
            recs.append(rr)
    return current, failed, recs


def _tree_digest(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def test_driver_borrows_the_current_u0_none_runs_the_registry_names_and_reads_the_u0_store_only(tmp_path, capsys):
    """(1, driver) From-disk analysis of the steered store: the borrow finds the judged view's model / campaign / tag from the
    registry, the u0 audit from the view's, the CURRENT generated `none` run per behaviour under the u0 record's `none` cell dir (the
    failed first attempts are named as siblings not used; the `none__lora0` control's runs on another cell dir are not candidates),
    and reads the u0 store without writing a byte of it; the analysis records all of it."""
    in_root, u0_root = _world(tmp_path)
    in_ids, in_recs = _in_view_runs(in_root)
    cur, failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_path, in_recs + u0_recs)
    before = _tree_digest(u0_root)
    assert prun.cmd_analyze(_args(in_root, root), _spec()) == 0
    assert _tree_digest(u0_root) == before                                                   # the source store was only READ
    analysis = json.loads((in_root / prun.ANALYSIS_JSON).read_text())
    assert analysis["cross_view"] is True
    anchors = analysis["borrowed_base"]["anchors"]
    assert anchors["campaign"] == synth.CAMPAIGN and anchors["model_id"] == synth.MODEL_ID
    assert anchors["in_view_tag"] == IN_TAG and anchors["in_view_view"] == VIEW and anchors["source_tag"] == U0_TAG
    assert anchors["source_audit"] == U0_AUDIT and anchors["source_store_root"] == str(u0_root.resolve())
    assert anchors["source_store_cell_dir"] == str((u0_root / "cells" / _run_key_of("none")).resolve())
    assert anchors["source_petri_run_key"] == _run_key_of("none") and anchors["epochs"] == 1
    assert anchors["registry"]["path"] == DB.db_path(root)
    assert {b: r["run_id"] for b, r in anchors["runs"].items()} == cur
    for b, r in anchors["runs"].items():
        assert (r["eval_id"], r["lifecycle_state"], r["is_current"], r["why"], r["n_items_generated"]) == (f"petri_{b}", "generated", 1, None, 6)
        assert r["view_legacy"] == U0_TAG and r["store_uri"] == str(u0_root / "cells" / _run_key_of("none"))
    assert sorted(r["run_id"] for r in anchors["sibling_runs_not_used"]) == sorted(failed.values())
    assert {(r["is_current"], r["why"], r["lifecycle_state"]) for r in anchors["sibling_runs_not_used"]} == {(0, "lifecycle_state failed", "failed")}
    src = analysis["borrowed_base"]["source"]
    assert src["audit"] == U0_AUDIT and src["provenance"]["validated"] is True and src["provenance"]["n_cells_validated"] == 1
    assert src["serving_identity"] == PASS and analysis["borrowed_base"]["in_view"]["serving_identities"] == {STEERED: LENS}
    assert analysis["base_screen"]["cross_view"] is True and analysis["base_screen"]["eligible_behaviors"] == sorted(ELIGIBLE2)
    assert analysis["by_cell"]["none"]["audited_footing"]["petri_run_key"] == _run_key_of("none")
    assert analysis["by_cell"]["none"]["audited_footing"]["audited"] == {"items": 18, "expected": 18}
    # the source is an explicit, verified dependency -- the generation bound to every anchor, the snapshot under THIS view's store
    none_dir = u0_root / "cells" / _run_key_of("none")
    sg = anchors["source_generation"]
    assert sg["results_json_sha256_on_disk"] == hashlib.sha256((none_dir / "results.json").read_bytes()).hexdigest()
    assert set(sg["files"]) >= {"manifest.json", "results.json", "rows.jsonl"} and sg["digest"]
    assert {b: x["store_completion_sha256"] for b, x in sg["registry_binding"].items()} == {b: sg["results_json_sha256_on_disk"] for b in BEHAVIORS3}
    assert {b: x["run_id"] for b, x in sg["registry_binding"].items()} == cur
    snap = Path(anchors["source_snapshot"]["dir"])
    assert snap.parent.parent == in_root / prun.BORROWED_BASE_DIR and snap.name == f"{_run_key_of('none')}.{sg['digest'][:12]}"
    assert (snap / "rows.parquet").exists() and (snap / prun.BORROWED_SOURCE_JSON).exists() and anchors["source_snapshot"]["reused"] is False
    assert anchors["source_root_parquet"] == {"present": True, "none_rows_equal_to_fold": True, "n_rows_root": 54, "n_rows_fold": 54,
                                              "note": anchors["source_root_parquet"]["note"]}
    assert analysis["completeness"]["cells"][STEERED]["status"] == "COMPLETE"
    err = capsys.readouterr().err
    assert f"'none' borrowed from '{U0_AUDIT}'" in err and str((u0_root / "cells" / _run_key_of("none")).resolve()) in err
    assert all(rid in err for rid in cur.values())


def test_driver_refuses_by_name_when_a_u0_none_run_is_non_current_or_missing(tmp_path):
    """(2) One behaviour's only u0 `none` run is FAILED (non-current, `why` names the rule): the borrow refuses naming the behaviour,
    the evaluation, the run and its currency -- never a smaller analysis; a behaviour with NO u0 `none` run at all refuses too; and
    a registry with no current `none` store of the u0 audit at all names what it found."""
    in_root, u0_root = _world(tmp_path)
    _ids, in_recs = _in_view_runs(in_root)
    # (a) instruction_evasion has only a failed run; the other two behaviours are fine
    cur, failed, u0_recs = _u0_none_runs(u0_root, behaviors=("self_preservation", "defer_to_users"))
    rid_failed, rr = _run("k_u0_none_a_instruction_evasion", "petri_instruction_evasion", u0_root / "cells" / _run_key_of("none"),
                          object_key="none", object_id="none", tag=U0_TAG, i0=500, failed=True)
    root = _registry(tmp_path, in_recs + u0_recs + [rr])
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(_args(in_root, root), _spec())
    msg = str(e.value)
    assert "behaviour 'instruction_evasion' (evaluation 'petri_instruction_evasion') has 0 CURRENT, generated `none` run(s)" in msg
    assert rid_failed in msg and "lifecycle failed; is_current 0; why: lifecycle_state failed" in msg
    assert "never a smaller analysis" in msg and not (in_root / prun.ANALYSIS_JSON).exists()
    # (b) a behaviour with no u0 `none` run on record at all
    tmp_b = tmp_path / "b"
    in_root, u0_root = _world(tmp_b)
    _ids, in_recs = _in_view_runs(in_root)
    cur, failed, u0_recs = _u0_none_runs(u0_root, behaviors=("self_preservation", "defer_to_users"))
    root = _registry(tmp_b, in_recs + u0_recs)
    with pytest.raises(prun.DriverError, match="behaviour 'instruction_evasion' .* has 0 CURRENT.*: no registered run; refusing"):
        prun.cmd_analyze(_args(in_root, root), _spec())
    # (c) no current `none` run anywhere for this model: the refusal names the expected u0 audit and every run on record
    tmp_c = tmp_path / "c"
    in_root, u0_root = _world(tmp_c)
    _ids, in_recs = _in_view_runs(in_root)
    _cur, _failed, u0_recs = _u0_none_runs(u0_root, with_controls=False)
    only_failed = [rr for rr in u0_recs if any(r["record_type"] == "run_state" and r["payload"]["state"] == "failed" for r in rr)]
    root = _registry(tmp_c, in_recs + only_failed)
    with pytest.raises(prun.DriverError, match=f"expected exactly one store of audit '{U0_AUDIT}' .* found 0 .* why: lifecycle_state failed"):
        prun.cmd_analyze(_args(in_root, root), _spec())
    # (d) no registry at all: the borrow is a refusal, not a silent gate text
    with pytest.raises(prun.DriverError, match="there is no registry database at"):
        prun.cmd_analyze(_args(in_root, str(tmp_c / "no-such-registry")), _spec())


def test_driver_keeps_the_in_view_base_and_the_u0_gate_text(tmp_path):
    """(3, 4, driver) A store that carries its own `none` is analysed as before -- the borrow returns None BEFORE the registry is
    opened (a registry root that does not exist proves it) and the analysis gains no key; a u0 view WITHOUT `none` rows keeps the
    standing §8 gate refusal word for word (the registry tag says u0, so nothing is borrowed)."""
    # the in-view base wins
    in_root, _u0 = _world(tmp_path, in_cells=("none", "prompted__safety"))
    spec = _spec(cells=("none", "prompted__safety"))
    record = prun.read_cells_record(in_root)
    assert prun.base_screen_borrow(_args(in_root, str(tmp_path / "no-such-registry")), spec, in_root, record) is None
    assert prun.cmd_analyze(_args(in_root, str(tmp_path / "no-such-registry")), spec) == 0
    analysis = json.loads((in_root / prun.ANALYSIS_JSON).read_text())
    assert "cross_view" not in analysis and "borrowed_base" not in analysis and "cross_view" not in analysis["by_cell"]["none"]
    assert analysis["base_screen"]["eligible_behaviors"] == sorted(ELIGIBLE2)
    # a u0 view with no `none` rows: the gate, as it was
    tmp_u = tmp_path / "u"
    in_root, _u0 = _world(tmp_u, in_cells=("prompted__safety",), in_audit=U0_AUDIT, with_u0_store=False)   # the store IS the u0 store
    _ids, in_recs = _in_view_runs(in_root, tag=U0_TAG, cell="prompted__safety", object_key="prompted__safety_refusal",
                                  object_id=synth.OBJECT_SHA)
    root = _registry(tmp_u, in_recs)
    spec = _spec(audit=U0_AUDIT, cells=("prompted__safety",))
    assert prun.base_screen_borrow(_args(in_root, root), spec, in_root, prun.read_cells_record(in_root)) is None
    with pytest.raises(petri_stats.PetriStatsError, match="the undefended base cell 'none' has no rows and no refusal footing — the "
                                                          "base-elicitability screen is a GATE"):
        prun.cmd_analyze(_args(in_root, root), spec)


# ============================================================================================================ lock hygiene
def _lock_recorder(monkeypatch):
    """Record every ``db.write_lock`` take with the phase it happened in; ``_analysis_under_lock`` moves the phase claim -> analysis
    -> filing, so a take during the analysis or the filing is attributable."""
    takes: list = []
    phase = ["claim"]
    original = DB.write_lock

    def recording(root, *a, **k):
        takes.append((phase[0], str(root)))
        return original(root, *a, **k)
    monkeypatch.setattr(DB, "write_lock", recording)
    real = prun._analysis_under_lock

    def phased(*a, **k):
        phase[0] = "analysis"
        try:
            return real(*a, **k)
        finally:
            phase[0] = "filing"
    monkeypatch.setattr(prun, "_analysis_under_lock", phased)
    return takes


def _registry_mode(monkeypatch, root: str) -> None:
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.delenv(emit.SPOOL_ENV, raising=False)
    monkeypatch.delenv(emit.PUSH_SCRIPT_ENV, raising=False)
    monkeypatch.setenv("REGISTRY_ROOT", root)


def test_registry_worklist_refuses_the_gate_before_any_claim_so_nothing_is_claimed_or_released(tmp_path, monkeypatch, capsys):
    """(d) The 02:16Z dispatch claimed 7 runs, refused at the §8 gate, released 7 -- fourteen harvester cycles under the write lock
    for nothing. The borrow (and the record check) now run BEFORE the claims: a steered view with no current u0 `none` refuses with
    ZERO lock takes, every run still pending, no judgment row anywhere."""
    in_root, _u0_root = _world(tmp_path)
    _ids, in_recs = _in_view_runs(in_root)
    root = _registry(tmp_path, in_recs)                                     # no u0 `none` run on record
    _registry_mode(monkeypatch, root)
    takes = _lock_recorder(monkeypatch)
    monkeypatch.setattr(prun, "_consolidate_store", lambda r, s: r / "rows.parquet")     # the frame is pre-written; the fold is pinned elsewhere
    with pytest.raises(prun.DriverError, match="expected exactly one store of audit"):
        prun.cmd_analyze(_args(in_root, root, worklist="registry"), _spec())
    assert takes == []
    err = capsys.readouterr().err
    assert "claim lock held" not in err and "release lock held" not in err and "released" not in err
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 0
        pending = P.pending(conn)
        assert len(pending) == len(BEHAVIORS3) and {r["pending_reason"] for r in pending} != {"claimed"}
    assert not (in_root / prun.ANALYSIS_JSON).exists()


def test_registry_worklist_analysis_and_filing_take_no_registry_lock_and_each_claim_take_is_timed(tmp_path, monkeypatch, capsys):
    """(a, c) The registry-worklist Petri analysis end to end over the steered store with the borrow: the ONLY lock takes are the
    claims (one per run, one harvester cycle each -- `pending.claim` has no batch form), none during the analysis, none during the
    judgment / metric / item filing (spooled), each take's hold time on stderr; the runs then fold as judged."""
    in_root, u0_root = _world(tmp_path)
    in_ids, in_recs = _in_view_runs(in_root)
    cur, _failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_path, in_recs + u0_recs)
    _registry_mode(monkeypatch, root)
    takes = _lock_recorder(monkeypatch)
    monkeypatch.setattr(prun, "_consolidate_store", lambda r, s: r / "rows.parquet")
    captured: list = []                                                      # the worklist rows the analysis selected and filed from
    real_wl = prun.registry_worklist_runs

    def capturing(*a, **k):
        wl, rows = real_wl(*a, **k)
        captured.extend(rows)
        return wl, rows
    monkeypatch.setattr(prun, "registry_worklist_runs", capturing)
    assert prun.cmd_analyze(_args(in_root, root, worklist="registry"), _spec()) == 0
    assert len(takes) == len(BEHAVIORS3) and {p for p, _r in takes} == {"claim"}
    err = capsys.readouterr().err
    assert err.count("[registry] claim lock held ") == len(BEHAVIORS3) and "pending.claim has no batch form" in err
    assert "release lock held" not in err
    assert all(rid in err for rid in in_ids.values())
    analysis = json.loads((in_root / prun.ANALYSIS_JSON).read_text())
    assert analysis["cross_view"] is True and {b: r["run_id"] for b, r in analysis["borrowed_base"]["anchors"]["runs"].items()} == cur
    # every run the worklist selected is in the in-view set captured at selection, and the selection is recorded beside it
    anchors = analysis["borrowed_base"]["anchors"]
    assert set(in_ids.values()) <= set(anchors["in_view_runs_at_selection"]) and anchors["worklist_selection"]["run_ids"] == sorted(in_ids.values())
    # every worklist row carries every in-view identity field, the rows ARE the baseline recorded at selection (one set of rows: compared
    # against the borrow's SELECT, re-read before the filing, filed from), and the borrow's own SELECT is recorded beside it
    assert len(captured) == len(BEHAVIORS3) and all(set(prun.BORROW_IN_VIEW_FIELDS) <= set(r) for r in captured)
    assert anchors["in_view_runs_at_selection"] == {str(r["run_id"]): {f: r.get(f) for f in prun.BORROW_IN_VIEW_FIELDS} for r in captured}
    assert set(anchors["in_view_select_at_borrow"]) >= set(anchors["in_view_runs_at_selection"])
    assert anchors["worklist_selection"]["identity_fields"] == list(prun.BORROW_IN_VIEW_IDENTITY_FIELDS)
    assert set(anchors["in_view_recheck"]["runs"]) == set(in_ids.values())
    monkeypatch.undo()                                                       # the fold below takes the lock on purpose
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    with DB.reading(DB.db_path(root)) as conn:
        for b, rid in in_ids.items():
            st = conn.execute("SELECT lifecycle_state, n_items_judged FROM run WHERE run_id = ?", (rid,)).fetchone()
            assert tuple(st) == ("judged", 6), (b, st)
            js = conn.execute("SELECT status FROM judgment WHERE run_id = ?", (rid,)).fetchall()
            assert [j[0] for j in js] == ["complete"], (b, js)
        for rid in cur.values():                                             # the anchors were read, never judged or claimed here
            assert conn.execute("SELECT COUNT(*) FROM judgment WHERE run_id = ?", (rid,)).fetchone()[0] == 0


def test_claim_all_timed_keeps_claim_all_semantics_and_restores_the_instance():
    """The timing wrapper times the instance's ``claim`` for the duration of ``claim_all`` only: ``claim_all`` runs unchanged (a
    ClaimHeld propagates after ITS rollback) and the instance is left as it was."""
    calls: list = []

    class FakeWorklist:
        def claim(self, row):
            calls.append(("claim", row["run_id"]))
            if row["run_id"] == "r2":
                raise P.ClaimHeld("r2", "s-other", 10)
            return {"judgment_id": "j-" + row["run_id"]}

        def claim_all(self, rows):
            taken = []
            try:
                for r in rows:
                    self.claim(r)
                    taken.append(r["run_id"])
            except P.ClaimHeld:
                calls.append(("rollback", tuple(taken)))
                raise
            return {r["run_id"]: True for r in rows}
    wl = FakeWorklist()
    assert prun._claim_all_timed(wl, [{"run_id": "r1"}]) == {"r1": True}
    assert "claim" not in vars(wl)
    with pytest.raises(P.ClaimHeld):
        prun._claim_all_timed(wl, [{"run_id": "r1"}, {"run_id": "r2"}])
    assert "claim" not in vars(wl)
    assert calls == [("claim", "r1"), ("claim", "r1"), ("claim", "r2"), ("rollback", ("r1",))]


# ============================================================================================================
def test_driver_refuses_by_name_when_the_source_store_the_registry_names_is_absent_or_pruned(tmp_path):
    """(1) The u0 `none` runs are current in the registry but their store is not on this host (never staged, or pruned): the borrow refuses
    BY NAME -- the root, the missing files, the runs -- before anything is claimed or analysed; never a quarantine by silent failure, never
    a stale borrow. Three shapes: the whole source store gone; only the `none` cell dir gone; only its results.json gone."""
    # (a) the whole source store
    in_root, u0_root = _world(tmp_path)
    _ids, in_recs = _in_view_runs(in_root)
    cur, _failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_path, in_recs + u0_recs)
    shutil.rmtree(u0_root)
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(_args(in_root, root), _spec())
    msg = str(e.value)
    assert "NO cells.json on this host (absent or pruned store)" in msg and str(u0_root) in msg and "found 0" in msg
    assert not (in_root / prun.ANALYSIS_JSON).exists() and not (in_root / prun.BORROWED_BASE_DIR).exists()
    # (b) only the `none` cell dir
    tmp_b = tmp_path / "b"
    in_root, u0_root = _world(tmp_b)
    _ids, in_recs = _in_view_runs(in_root)
    cur, _failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_b, in_recs + u0_recs)
    none_dir = u0_root / "cells" / _run_key_of("none")
    shutil.rmtree(none_dir)
    with pytest.raises(prun.DriverError, match="is ABSENT or PRUNED on this host: missing"):
        prun.cmd_analyze(_args(in_root, root), _spec())
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(_args(in_root, root), _spec())
    assert str(none_dir) in str(e.value) and all(rid in str(e.value) for rid in cur.values()) and "neither mirrors nor prunes" in str(e.value)
    assert not (in_root / prun.BORROWED_BASE_DIR).exists()
    # (c) only the completion artifact
    tmp_c = tmp_path / "c"
    in_root, u0_root = _world(tmp_c)
    _ids, in_recs = _in_view_runs(in_root)
    cur, _failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_c, in_recs + u0_recs)
    (u0_root / "cells" / _run_key_of("none") / "results.json").unlink()
    with pytest.raises(prun.DriverError, match=r"ABSENT or PRUNED on this host: missing \[.*results\.json") as e:
        prun.cmd_analyze(_args(in_root, root), _spec())
    assert not (in_root / prun.BORROWED_BASE_DIR).exists()


def test_driver_binds_the_borrowed_rows_to_the_current_runs_completion_and_folds_the_cell_evidence_not_a_stale_parquet(tmp_path):
    """(2) rows.parquet carries the enumeration run_id, not the registry run id, so a source root parquet left by an EARLIER consolidation
    of the same enumeration could be borrowed as the newly current run's scores. The borrow now (i) binds every selected current run's
    registered completion (artifact uri + sha256) to the source results.json on disk and (ii) folds the borrowed rows from the source
    CELL's own evidence into a snapshot -- so a stale root parquet is never what is published, a registry/disk sha mismatch refuses by
    name, and a run without its completion identity refuses by name."""
    in_root, u0_root = _world(tmp_path)
    _ids, in_recs = _in_view_runs(in_root)
    cur, _failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_path, in_recs + u0_recs)
    none_dir = u0_root / "cells" / _run_key_of("none")
    # a STALE root parquet beside fresh cell evidence: every base score +2 (means 6.0 / 6.0 / 3.5 -- every behaviour would clear the floor)
    stale = pd.read_parquet(u0_root / "rows.parquet")
    m = (stale["arm_name"] == "none") & (stale["metric"] == stale["task"].str.slice(len("petri/")))
    stale.loc[m, "value"] = stale.loc[m, "value"] + 2.0
    stale.to_parquet(u0_root / "rows.parquet")
    assert prun.cmd_analyze(_args(in_root, root), _spec()) == 0
    analysis = json.loads((in_root / prun.ANALYSIS_JSON).read_text())
    # the screen rests on the CELL evidence (instruction_evasion 1.5, NO-HEADROOM), not the stale parquet (3.5)
    assert analysis["base_screen"]["eligible_behaviors"] == sorted(ELIGIBLE2)
    assert analysis["base_screen"]["behaviors"]["instruction_evasion"]["base_mean"] == 1.5
    assert analysis["base_screen"]["behaviors"]["self_preservation"]["base_mean"] == 4.0
    anchors = analysis["borrowed_base"]["anchors"]
    sg = anchors["source_generation"]
    disk_sha = hashlib.sha256((none_dir / "results.json").read_bytes()).hexdigest()
    assert sg["results_json_sha256_on_disk"] == disk_sha
    assert {b: x["store_completion_sha256"] for b, x in sg["registry_binding"].items()} == {b: disk_sha for b in BEHAVIORS3}
    assert all(x["store_completion_artifact_uri"] == str(none_dir / "results.json") for x in sg["registry_binding"].values())
    assert anchors["source_root_parquet"]["present"] is True and anchors["source_root_parquet"]["none_rows_equal_to_fold"] is False
    assert anchors["source_snapshot"]["rows"] == str(Path(anchors["source_snapshot"]["dir"]) / "rows.parquet")
    assert analysis["borrowed_base"]["source"]["provenance"]["validated"] is True       # the snapshot's rows still bind to the u0 record
    # (ii) the registry's current run completed a DIFFERENT generation than the disk holds: refused by name, both digests named
    tmp_b = tmp_path / "b"
    in_root, u0_root = _world(tmp_b)
    _ids, in_recs = _in_view_runs(in_root)
    cur, _failed, u0_recs = _u0_none_runs(u0_root, current_completion_sha="e" * 64)
    root = _registry(tmp_b, in_recs + u0_recs)
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(_args(in_root, root), _spec())
    msg = str(e.value)
    assert "STALE or foreign source" in msg and "eeeeeeeeeeee..." in msg and "never a silent stale borrow" in msg
    assert any(rid in msg for rid in cur.values()) and hashlib.sha256((u0_root / "cells" / _run_key_of("none") / "results.json").read_bytes()).hexdigest()[:12] in msg
    assert not (in_root / prun.BORROWED_BASE_DIR).exists() and not (in_root / prun.ANALYSIS_JSON).exists()
    # (iii) a current run that carries no completion identity is not a borrowable baseline
    tmp_c = tmp_path / "c"
    in_root, u0_root = _world(tmp_c)
    _ids, in_recs = _in_view_runs(in_root)
    cur, _failed, u0_recs = _u0_none_runs(u0_root, current_completion_sha=None)
    root = _registry(tmp_c, in_recs + u0_recs)
    with pytest.raises(prun.DriverError, match="carries no store_completion_sha256 -- the borrowed source cannot be bound"):
        prun.cmd_analyze(_args(in_root, root), _spec())
    # (iv) a current run whose completion artifact is another file: refused by name
    tmp_d = tmp_path / "d"
    in_root, u0_root = _world(tmp_d)
    _ids, in_recs = _in_view_runs(in_root)
    cur, _failed, u0_recs = _u0_none_runs(u0_root, current_completion_uri=str(u0_root / "cells" / _run_key_of("prompted__safety") / "results.json"))
    root = _registry(tmp_d, in_recs + u0_recs)
    with pytest.raises(prun.DriverError, match="completed artifact .* not the source cell's"):
        prun.cmd_analyze(_args(in_root, root), _spec())


def test_the_borrowed_source_snapshot_is_taken_once_verified_on_reuse_and_never_rewritten(tmp_path):
    """The snapshot under the steered view's store is named by the source generation, taken once (temp dir + rename), reused only after
    every file re-verifies against its recorded generation, and never rewritten: a tampered snapshot file refuses by name; a source that
    moves to a NEW generation gets a new snapshot directory (the old one stays)."""
    in_root, u0_root = _world(tmp_path)
    _ids, in_recs = _in_view_runs(in_root)
    _cur, _failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_path, in_recs + u0_recs)
    assert prun.cmd_analyze(_args(in_root, root), _spec()) == 0
    a1 = json.loads((in_root / prun.ANALYSIS_JSON).read_text())["borrowed_base"]["anchors"]
    snap = Path(a1["source_snapshot"]["dir"])
    assert a1["source_snapshot"]["reused"] is False and snap.is_dir()
    src = json.loads((snap / prun.BORROWED_SOURCE_JSON).read_text())
    assert src["generation"]["digest"] == a1["source_generation"]["digest"] and src["source_cell_dir"] == str(u0_root / "cells" / _run_key_of("none"))
    assert src["registry_binding"]["per_behaviour"] == a1["source_generation"]["registry_binding"]
    assert sorted(p.name for p in (snap / "cells" / _run_key_of("none")).iterdir()) == sorted(src["generation"]["files"])
    digest_before = _tree_digest(snap)
    # second analysis: the same snapshot, verified and reused, untouched
    assert prun.cmd_analyze(_args(in_root, root), _spec()) == 0
    a2 = json.loads((in_root / prun.ANALYSIS_JSON).read_text())["borrowed_base"]["anchors"]
    assert a2["source_snapshot"]["dir"] == str(snap) and a2["source_snapshot"]["reused"] is True and _tree_digest(snap) == digest_before
    assert {k: v for k, v in a2["source_generation"].items()} == {k: v for k, v in a1["source_generation"].items()}
    # a tampered snapshot file: refused by name, the snapshot is never rewritten
    tampered = snap / "cells" / _run_key_of("none") / "rows.jsonl"
    tampered.write_text(tampered.read_text() + json.dumps({"arm_name": "none", "task": "petri/self_preservation", "metric": "self_preservation",
                                                             "item_id": "s9", "value": 9.0}) + "\n")
    with pytest.raises(prun.DriverError, match=r"no longer matches its recorded generation \(file rows.jsonl: sha256 differs\)"):
        prun.cmd_analyze(_args(in_root, root), _spec())
    assert [p.name for p in snap.parent.iterdir()] == [snap.name]                 # nothing else was written beside it


# ============================================================================================================
def _rewrite_source(none_dir: Path, bump: float = 1.0) -> str:
    """A concurrent re-score's footprint on the LIVE source cell (cmd_rescore -> refresh_results is not under the steered out-root's lock):
    results.json rewritten (completed_at moves) and every behaviour score in rows.jsonl shifted by ``bump``. Returns the new results.json
    sha256."""
    results = Path(none_dir) / "results.json"
    res = json.loads(results.read_text())
    res["completed_at"] = float(res.get("completed_at") or 0.0) + 1.0
    results.write_text(json.dumps(res))
    rows = [json.loads(line) for line in (Path(none_dir) / "rows.jsonl").read_text().splitlines() if line.strip()]
    for r in rows:
        if r["metric"] == r["task"][len("petri/"):]:
            r["value"] = float(r["value"]) + bump
    (Path(none_dir) / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return hashlib.sha256(results.read_bytes()).hexdigest()


def _retire_runs(root: str, run_ids, *, i0: int = 900, state: str = "cancelled") -> None:
    """A terminal state for ``run_ids``, folded: the runs leave v_run_current (is_current 0, why `lifecycle_state <state>`)."""
    recs = [synth.rec("run_state", {"run_id": rid, "state": state, "event_utc": synth.t(i0 + k), "evidence_kind": "job_log"}, synth.t(i0 + k))
            for k, rid in enumerate(run_ids)]
    R.write_records(os.path.join(root, "records", "c", f"{i0:06d}-retire.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()


def _no_snapshot(in_root: Path) -> bool:
    base = in_root / prun.BORROWED_BASE_DIR / U0_AUDIT
    return not base.exists() or list(base.iterdir()) == []


def test_a_source_rewritten_between_the_binding_and_the_snapshot_refuses_by_name_and_snapshots_nothing(tmp_path, monkeypatch, capsys):
    """(1) The binding hashed the live results.json, then the generation and the copy read the directory again -- a re-score in
    between used to produce an internally consistent NEW snapshot certified by the OLD run. Now each later read is checked against the
    binding: a rewrite between the binding and the generation hash, before the copy, or after the copy refuses by name ("source changed
    during binding/snapshot: <before> -> <after>"), nothing is snapshotted, nothing analysed, nothing claimed; and once the disk has moved
    on while the registry still certifies the old run, the next borrow refuses STALE until the new generation is registered current."""
    # (a) between the binding and the generation hash
    in_root, u0_root = _world(tmp_path)
    _ids, in_recs = _in_view_runs(in_root)
    cur, _failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_path, in_recs + u0_recs)
    none_dir = u0_root / "cells" / _run_key_of("none")
    sha0 = hashlib.sha256((none_dir / "results.json").read_bytes()).hexdigest()
    real_generation = prun._source_generation
    seen: dict = {}

    def rewriting_generation(cell_dir):
        seen["sha1"] = _rewrite_source(Path(cell_dir))
        return real_generation(cell_dir)
    monkeypatch.setattr(prun, "_source_generation", rewriting_generation)
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(_args(in_root, root), _spec())
    msg = str(e.value)
    assert "source changed during binding/snapshot (generation)" in msg and f"{sha0[:12]} -> {seen['sha1'][:12]}" in msg
    assert _no_snapshot(in_root) and not (in_root / prun.ANALYSIS_JSON).exists()
    monkeypatch.undo()
    # the disk moved on, the registry still certifies sha0: STALE by name, never the new rows under the old run
    with pytest.raises(prun.DriverError, match="STALE or foreign source") as e:
        prun.cmd_analyze(_args(in_root, root), _spec())
    assert sha0[:12] in str(e.value) and seen["sha1"][:12] in str(e.value) and _no_snapshot(in_root)
    # (b) between the binding and the copy (the snapshot's own pre-copy check) and (c) between the copy and the fold (the post-copy check)
    for stage, target in (("before the copy", "_borrowed_base_snapshot"), ("after the copy", "collect_auditor_refusal_tombstones")):
        tmp_x = tmp_path / stage.replace(" ", "_")
        in_root, u0_root = _world(tmp_x)
        _ids, in_recs = _in_view_runs(in_root)
        cur, _failed, u0_recs = _u0_none_runs(u0_root)
        root = _registry(tmp_x, in_recs + u0_recs)
        none_dir = u0_root / "cells" / _run_key_of("none")
        sha0 = hashlib.sha256((none_dir / "results.json").read_bytes()).hexdigest()
        real = getattr(prun, target)

        def rewriting(*a, _real=real, _dir=none_dir, **k):
            seen["sha1"] = _rewrite_source(_dir)
            return _real(*a, **k)
        monkeypatch.setattr(prun, target, rewriting)
        with pytest.raises(prun.DriverError) as e:
            prun.cmd_analyze(_args(in_root, root), _spec())
        monkeypatch.undo()
        msg = str(e.value)
        assert f"source changed during binding/snapshot ({stage}): {sha0[:12]} -> {seen['sha1'][:12]}" in msg, (stage, msg)
        assert _no_snapshot(in_root) and not (in_root / prun.ANALYSIS_JSON).exists()         # no snapshot, no temp dir left behind
    # (d) the same window in registry mode claims nothing: the borrow is resolved before the claims
    tmp_r = tmp_path / "registry_mode"
    in_root, u0_root = _world(tmp_r)
    in_ids, in_recs = _in_view_runs(in_root)
    cur, _failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_r, in_recs + u0_recs)
    _registry_mode(monkeypatch, root)
    takes = _lock_recorder(monkeypatch)
    monkeypatch.setattr(prun, "_consolidate_store", lambda r, s: r / "rows.parquet")
    none_dir = u0_root / "cells" / _run_key_of("none")

    def rewriting_generation2(cell_dir, _real=real_generation, _dir=none_dir):
        _rewrite_source(_dir)
        return _real(cell_dir)
    monkeypatch.setattr(prun, "_source_generation", rewriting_generation2)
    with pytest.raises(prun.DriverError, match="source changed during binding/snapshot"):
        prun.cmd_analyze(_args(in_root, root, worklist="registry"), _spec())
    assert takes == [] and _no_snapshot(in_root)
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 0


def test_a_source_that_moved_on_after_the_snapshot_is_never_served_stale_it_refuses_until_the_new_generation_is_current_then_resnapshots(tmp_path):
    """(1, reuse) After a snapshot exists the source moves on (a re-score: new results.json + rows). The old snapshot is never served under
    the new world: while the registry still certifies the old run (old sha) the borrow refuses STALE by name; once the new generation's
    runs are current (the old retired) the borrow takes a NEW snapshot bound to the new runs and reads the new rows; the old snapshot
    directory stays, byte for byte."""
    in_root, u0_root = _world(tmp_path)
    _ids, in_recs = _in_view_runs(in_root)
    cur, failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_path, in_recs + u0_recs)
    none_dir = u0_root / "cells" / _run_key_of("none")
    assert prun.cmd_analyze(_args(in_root, root), _spec()) == 0
    a1 = json.loads((in_root / prun.ANALYSIS_JSON).read_text())
    snap1 = Path(a1["borrowed_base"]["anchors"]["source_snapshot"]["dir"])
    digest1 = _tree_digest(snap1)
    assert a1["base_screen"]["behaviors"]["self_preservation"]["base_mean"] == 4.0
    v1 = a1["borrowed_base"]["anchors"]["source_verification"]
    assert (v1["binding_results_json_sha256"] == v1["generation_results_json_sha256"] == v1["snapshot_results_json_sha256"]
            == v1["live_results_json_sha256_before_copy"] == v1["live_results_json_sha256_after_copy"])
    assert set(v1["selected_runs_completion_sha256"].values()) == {v1["snapshot_results_json_sha256"]} and v1["verified_utc"]
    assert a1["borrowed_base"]["anchors"]["currency_recheck"]["runs"]["self_preservation"]["run_id"] == cur["self_preservation"]
    assert all(r["is_current"] == 1 for r in a1["borrowed_base"]["anchors"]["currency_recheck"]["runs"].values())
    (in_root / prun.ANALYSIS_JSON).unlink()
    # the source moves on; the registry still certifies the old run: STALE, by name, nothing new written
    sha_new = _rewrite_source(none_dir)
    with pytest.raises(prun.DriverError, match="STALE or foreign source") as e:
        prun.cmd_analyze(_args(in_root, root), _spec())
    assert sha_new[:12] in str(e.value) and not (in_root / prun.ANALYSIS_JSON).exists()
    assert [p.name for p in snap1.parent.iterdir()] == [snap1.name] and _tree_digest(snap1) == digest1
    # the new generation registered current (new runs with the new sha; the old ones retired): a NEW snapshot bound to the new runs
    _retire_runs(root, sorted(cur.values()))
    new_cur, _f2, new_recs = _u0_none_runs(u0_root, i0=700, with_failed_siblings=False, with_controls=False, key_prefix="k_u0_none_renewed")
    R.write_records(os.path.join(root, "records", "c", "000700-renewed.jsonl"), [r for rr in new_recs for r in rr])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    assert prun.cmd_analyze(_args(in_root, root), _spec()) == 0
    a2 = json.loads((in_root / prun.ANALYSIS_JSON).read_text())
    anchors2 = a2["borrowed_base"]["anchors"]
    snap2 = Path(anchors2["source_snapshot"]["dir"])
    assert snap2 != snap1 and anchors2["source_snapshot"]["reused"] is False and snap2.parent == snap1.parent
    assert {b: r["run_id"] for b, r in anchors2["runs"].items()} == new_cur
    assert anchors2["source_verification"]["snapshot_results_json_sha256"] == sha_new
    assert set(anchors2["source_verification"]["selected_runs_completion_sha256"].values()) == {sha_new}
    assert a2["base_screen"]["behaviors"]["self_preservation"]["base_mean"] == 5.0          # the NEW rows, not snap1's
    # the retired runs AND the original failed first attempts share the cell dir: named as siblings, none used
    assert sorted(r["run_id"] for r in anchors2["sibling_runs_not_used"]) == sorted(list(cur.values()) + list(failed.values()))
    assert _tree_digest(snap1) == digest1                                                    # the old snapshot untouched


def test_an_anchor_that_lost_currency_between_the_binding_and_the_filing_refuses_before_anything_is_filed(tmp_path, monkeypatch, capsys):
    """(2) Between the binding (pre-claim) and the filing an anchor run can lose currency (superseded, excluded, held out,
    cancelled). The re-check right before the analysis is written refuses by name: from-disk nothing is written; in registry mode nothing is
    written or filed and every claim is released."""
    # from-disk
    in_root, u0_root = _world(tmp_path)
    _ids, in_recs = _in_view_runs(in_root)
    cur, _failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_path, in_recs + u0_recs)
    real = prun._analysis_under_lock

    def flipping(*a, _real=real, _root=root, _rid=cur["self_preservation"], **k):
        analysis = _real(*a, **k)
        _retire_runs(_root, [_rid])                        # the anchor is superseded/cancelled AFTER the binding, BEFORE the filing
        return analysis
    monkeypatch.setattr(prun, "_analysis_under_lock", flipping)
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(_args(in_root, root), _spec())
    msg = str(e.value)
    assert "anchor identity changed between the binding and the filing" in msg and cur["self_preservation"] in msg
    assert "self_preservation" in msg and "'generated' -> 'cancelled' (no longer generated-or-later)" in msg and "is_current: 1 -> 0" in msg
    assert "never a silent stale borrow" in msg
    assert not (in_root / prun.ANALYSIS_JSON).exists()
    monkeypatch.undo()
    # registry mode: nothing written, nothing filed, the claims released
    tmp_r = tmp_path / "registry_mode"
    in_root, u0_root = _world(tmp_r)
    in_ids, in_recs = _in_view_runs(in_root)
    cur, _failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_r, in_recs + u0_recs)
    _registry_mode(monkeypatch, root)
    monkeypatch.setattr(prun, "_consolidate_store", lambda r, s: r / "rows.parquet")

    def flipping2(*a, _real=real, _root=root, _rid=cur["defer_to_users"], **k):
        analysis = _real(*a, **k)
        _retire_runs(_root, [_rid], i0=950)
        return analysis
    monkeypatch.setattr(prun, "_analysis_under_lock", flipping2)
    with pytest.raises(prun.DriverError, match="anchor identity changed between the binding and the filing"):
        prun.cmd_analyze(_args(in_root, root, worklist="registry"), _spec())
    err = capsys.readouterr().err
    assert not (in_root / prun.ANALYSIS_JSON).exists() and f"released {len(BEHAVIORS3)} of {len(BEHAVIORS3)} claim(s)" in err
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM judgment WHERE status = 'complete'").fetchone()[0] == 0
        for rid in in_ids.values():
            assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "generated"


# ============================================================================================================
def _repatch_completion(root: str, run_id: str, sha: str, *, i0: int = 960) -> None:
    """A re-score's REGISTRY footprint without retiring the run: the producer's on_results emits a NEW completion sha as a run_patch for the
    SAME run id (grade_daemon.store_generation detects re-filed stores by exactly this), folded; the run stays is_current 1."""
    rec = synth.rec("run_patch", {"run_id": run_id, "fields": {"store_completion_sha256": sha}}, synth.t(i0))
    R.write_records(os.path.join(root, "records", "c", f"{i0:06d}-repatch.jsonl"), [rec])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()


def _is_current(root: str, run_id: str) -> int:
    with DB.reading(DB.db_path(root)) as conn:
        return int(conn.execute("SELECT is_current FROM v_run_current WHERE run_id = ?", (run_id,)).fetchone()[0])


def test_a_completion_re_filed_for_a_still_current_anchor_between_the_snapshot_and_the_filing_refuses_by_name(tmp_path, monkeypatch, capsys):
    """A re-score rewrites results.json and the producer re-files the run's completion sha under the SAME run id -- is_current stays
    1, so a currency-only re-check let the OLD rows file under a registry that pointed to the NEW completion. The whole-tuple identity check
    refuses by name at BOTH checkpoints: between the snapshot verification and the filing (from-disk: nothing written; registry mode: nothing
    written or filed, the claims released) and right after the snapshot verification (checkpoint 1)."""
    new_sha = "f" * 64
    # checkpoint 2 (the filing), from-disk
    in_root, u0_root = _world(tmp_path)
    _ids, in_recs = _in_view_runs(in_root)
    cur, _failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_path, in_recs + u0_recs)
    old_sha = hashlib.sha256((u0_root / "cells" / _run_key_of("none") / "results.json").read_bytes()).hexdigest()
    real = prun._analysis_under_lock

    def refiling(*a, _real=real, _root=root, _rid=cur["defer_to_users"], **k):
        analysis = _real(*a, **k)
        _repatch_completion(_root, _rid, new_sha)          # after the snapshot was verified, before the filing; the run stays current
        return analysis
    monkeypatch.setattr(prun, "_analysis_under_lock", refiling)
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(_args(in_root, root), _spec())
    msg = str(e.value)
    assert "anchor identity changed between the binding and the filing" in msg and cur["defer_to_users"] in msg
    assert f"store_completion_sha256: {old_sha[:48]!r} -> {new_sha[:48]!r}" in msg
    assert f"store_completion_sha256 {new_sha[:12]} != the verified snapshot's results.json {old_sha[:12]}" in msg
    assert "is_current" not in msg.split("defer_to_users", 1)[1].split(";")[0] and _is_current(root, cur["defer_to_users"]) == 1
    assert not (in_root / prun.ANALYSIS_JSON).exists()
    monkeypatch.undo()
    # checkpoint 1 (right after the snapshot verification)
    tmp_b = tmp_path / "b"
    in_root, u0_root = _world(tmp_b)
    _ids, in_recs = _in_view_runs(in_root)
    cur, _failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_b, in_recs + u0_recs)
    real_snapshot = prun._borrowed_base_snapshot

    def refiling_snapshot(*a, _real=real_snapshot, _root=root, _rid=cur["self_preservation"], **k):
        result = _real(*a, **k)                            # the snapshot is taken and verified ...
        _repatch_completion(_root, _rid, new_sha, i0=965)   # ... then the registry re-files the run's completion
        return result
    monkeypatch.setattr(prun, "_borrowed_base_snapshot", refiling_snapshot)
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(_args(in_root, root), _spec())
    msg = str(e.value)
    assert "anchor identity changed between the binding and the snapshot verification" in msg and cur["self_preservation"] in msg
    assert f"-> {new_sha[:48]!r}" in msg and not (in_root / prun.ANALYSIS_JSON).exists() and _is_current(root, cur["self_preservation"]) == 1
    monkeypatch.undo()
    # checkpoint 2 in registry mode: nothing written, nothing filed, every claim released
    tmp_r = tmp_path / "registry_mode"
    in_root, u0_root = _world(tmp_r)
    in_ids, in_recs = _in_view_runs(in_root)
    cur, _failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_r, in_recs + u0_recs)
    _registry_mode(monkeypatch, root)
    monkeypatch.setattr(prun, "_consolidate_store", lambda r, s: r / "rows.parquet")

    def refiling2(*a, _real=real, _root=root, _rid=cur["instruction_evasion"], **k):
        analysis = _real(*a, **k)
        _repatch_completion(_root, _rid, new_sha, i0=970)
        return analysis
    monkeypatch.setattr(prun, "_analysis_under_lock", refiling2)
    with pytest.raises(prun.DriverError, match="anchor identity changed between the binding and the filing"):
        prun.cmd_analyze(_args(in_root, root, worklist="registry"), _spec())
    err = capsys.readouterr().err
    assert not (in_root / prun.ANALYSIS_JSON).exists() and f"released {len(BEHAVIORS3)} of {len(BEHAVIORS3)} claim(s)" in err
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM judgment WHERE status = 'complete'").fetchone()[0] == 0
        for rid in in_ids.values():
            assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "generated"
    assert _is_current(root, cur["instruction_evasion"]) == 1


def test_the_anchor_identity_check_covers_every_identity_field_the_borrow_reads(tmp_path):
    """Coverage: the columns the borrow reads per anchor row are PARTITIONED into identity fields -- compared at both
    checkpoints -- and recorded-only fields; the row query and the identity query are built from the same constants (no hand-copied column
    list); every field the selection and the binding read is an identity field; the anchors record the full row and both checkpoints report
    exactly the identity fields. A new column has to be classified or this fails, so a fourth seam has nowhere to live."""
    import inspect
    import re
    ident, allow, rows = prun.BORROW_ANCHOR_IDENTITY_FIELDS, prun.BORROW_ANCHOR_RECORDED_ONLY_REASONS, prun.BORROW_ANCHOR_ROW_FIELDS
    # the DERIVED rule: identity = every SELECT column not on the recorded-only allowlist, which names a reason per field
    assert tuple(f for f in rows if f not in allow) == ident
    assert set(rows) == set(ident) | set(allow) and set(allow) <= set(rows) and not (set(ident) & set(allow)) and len(rows) == len(set(rows))
    assert tuple(prun.BORROW_ANCHOR_RECORDED_ONLY_FIELDS) == tuple(allow)
    assert all(isinstance(reason, str) and len(reason) > 20 for reason in allow.values())
    # the fields a same-id corrective FULL record could restate with the completion untouched are identity
    assert {"campaign", "model_id", "model_revision"} <= set(ident)
    # what the selection reads (view identity, behaviour, cell, store + its relocation, usable lifecycle, currency) and what the binding reads
    # (the registered completion): the literal here is the pin, the code reads the constant
    selection_fields = {"run_id", "campaign", "model_id", "model_revision", "eval_id", "producer_object_key_legacy", "substrate", "store_uri",
                        "lifecycle_state", "is_current", "selection_state"}
    binding_fields = {"store_completion_artifact_uri", "store_completion_sha256"}
    assert selection_fields | binding_fields == set(ident)
    src_binding = inspect.getsource(prun._verify_source_binding)
    assert all(f in src_binding for f in binding_fields)
    src_borrow = inspect.getsource(prun.base_screen_borrow)
    assert "_cell_of(r)" in src_borrow and "store_path_of" in inspect.getsource(prun) and "_BORROW_GENERATED_STATES" in src_borrow
    # EVERY SELECT the borrow runs derives its column list from a constant: the in-view query, the row query, the identity query
    in_view_query = src_borrow[src_borrow.index("id_cols = BORROW_IN_VIEW_FIELDS"):src_borrow.index("in_view = []")]
    assert set(prun.BORROW_IN_VIEW_FIELDS) <= set(rows) | set(prun.FILING_ROW_FIELDS) | set(prun.CLAIM_ROW_FIELDS)   # fix-6 / fix-9: filing + claim
    assert {"campaign", "model_id", "model_revision", "eval_id", "producer_object_key_legacy", "object_id"} <= set(prun.BORROW_IN_VIEW_FIELDS)
    for src in (in_view_query, inspect.getsource(prun._base_screen_anchor_rows), inspect.getsource(prun._anchor_identity_rows)):
        selects = re.findall(r"SELECT\s+(.+?)\s+FROM\s+run", src, re.S)
        assert selects and all("{" in sel for sel in selects), src     # an f-string over a constant, never a hand-typed column list
    assert "BORROW_ANCHOR_ROW_FIELDS" in inspect.getsource(prun._base_screen_anchor_rows)
    assert "BORROW_ANCHOR_IDENTITY_FIELDS" in inspect.getsource(prun._anchor_identity_rows)
    assert "BORROW_ANCHOR_IDENTITY_FIELDS" in inspect.getsource(prun.verify_borrowed_anchor_identity)
    assert prun._anchor_select_sql(rows).count(",") == len(rows) - 1
    assert all(("vc." + f) in prun._anchor_select_sql(rows) for f in ("is_current", "why", "selection_state", "held_out"))
    # end to end: the anchors carry the full row per run; both checkpoints report exactly the identity fields and agree with the binding
    in_root, u0_root = _world(tmp_path)
    _ids, in_recs = _in_view_runs(in_root)
    cur, _failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_path, in_recs + u0_recs)
    assert prun.cmd_analyze(_args(in_root, root), _spec()) == 0
    anchors = json.loads((in_root / prun.ANALYSIS_JSON).read_text())["borrowed_base"]["anchors"]
    for b in BEHAVIORS3:
        assert set(anchors["runs"][b]) == set(rows)
        for stage_key in ("identity_recheck_after_snapshot", "currency_recheck"):
            blk = anchors[stage_key]
            assert blk["fields"] == list(ident) and set(blk["runs"][b]) == set(ident) | {"why"}
            assert {f: blk["runs"][b][f] for f in ident} == {f: anchors["runs"][b][f] for f in ident}
            assert blk["runs"][b]["store_completion_sha256"] == anchors["source_verification"]["snapshot_results_json_sha256"] == blk["snapshot_results_json_sha256"]
    assert anchors["identity_recheck_after_snapshot"]["stage"] == "the snapshot verification" and anchors["currency_recheck"]["stage"] == "the filing"
    # the judged view's identity every anchor was verified against at selection (the synthetic registrations state no revision, like part of
    # the live u0 `none` runs: NULL is recorded, tolerated at selection, and part of the compared tuple)
    sid = anchors["source_identity_at_selection"]
    assert (sid["campaign"], sid["model_id"], sid["target_revision"]) == (synth.CAMPAIGN, synth.MODEL_ID, REV)
    assert sid["in_view_registered_revisions"] == [] and sid["anchor_registered_revisions"] == {b: None for b in BEHAVIORS3}
    assert anchors["identity_fields"] == list(ident) and anchors["recorded_only_fields"] == dict(allow)
    # the judged view's own identity derives from the SAME constant (no second hand-kept list), is captured per in-view run at
    # selection, and is re-read with the same function before the filing
    in_view_ident = prun.BORROW_IN_VIEW_IDENTITY_FIELDS
    # the compared in-view identity DERIVES from the fields the FILING path reads (FILING_ROW_FIELDS, the one accessor) plus the
    # anchor identity, minus a reasoned uncompared set -- replicated here, so a routing field can never be left uncompared
    filing, claim_f, uncompared = set(prun.FILING_ROW_FIELDS), set(prun.CLAIM_ROW_FIELDS), prun.BORROW_IN_VIEW_UNCOMPARED_REASONS
    derived = prun.FILING_DERIVED_FIELDS_SQL
    # THE PRINCIPLE -- every field the filing OR the claim routes on is compared; "pending-view-only" is not
    # an exclusion reason: a non-`run` field is DERIVED by the identity read (required_judge_id = the current instrument's judge, the pending
    # view's own derivation) and compared, never dropped
    assert in_view_ident == tuple(dict.fromkeys(list(prun.FILING_ROW_FIELDS) + list(prun.CLAIM_ROW_FIELDS)
                                                + [f for f in ident if f not in uncompared]))
    assert filing | claim_f <= set(in_view_ident) and set(uncompared) <= set(ident)
    # the instrument's designated judge is the DERIVED field; the SELECTED judge is a SELECTION field -- bound
    # by the worklist row, rendered NULL from `run`, verified against the pending view (and, by default, the instrument) rather than compared
    selection = prun.SELECTION_FIELDS
    assert set(derived) == set(prun.FILING_DERIVED_REASONS) == {"instrument_judge_id"} and set(derived) <= claim_f
    assert set(selection) == set(prun.SELECTION_FIELD_REASONS) == {"required_judge_id"} and set(selection) <= filing & claim_f
    assert not (set(selection) & set(derived))
    assert all(isinstance(r, str) and len(r) > 20 for r in list(prun.FILING_DERIVED_REASONS.values()) + list(uncompared.values())
               + list(prun.SELECTION_FIELD_REASONS.values()))
    assert not hasattr(prun, "FILING_PENDING_ONLY_REASONS")                   # the exclusion is gone, not renamed
    assert {"run_id", "eval_id", "producer_object_key_legacy", "object_id", "campaign", "store_uri", "substrate", "n_items_generated",
            "model_id", "model_revision", "store_completion_artifact_uri", "store_completion_sha256",
            "instrument_id", "required_judge_id", "instrument_judge_id"} <= set(in_view_ident)
    # every field the CLAIM path reads from the row (RegistryWorklist.claim: run_id and required_judge_id -> the judgment's judge_id) is covered,
    # and the derived judge is recomputed the way v_pending_judgment derives it (instrument.judge_id via run.instrument_id), by the one renderer
    from sisteer.registry import worklist as RW
    claim_src = inspect.getsource(RW.RegistryWorklist.claim)
    claim_reads = {a or b for a, b in re.findall(r"row\[['\"]([a-z_]+)['\"]\]|row\.get\(['\"]([a-z_]+)['\"]", claim_src)}
    assert claim_reads == {"run_id", "required_judge_id"} and claim_reads <= claim_f <= set(in_view_ident)
    expr = derived["instrument_judge_id"]
    assert "FROM instrument i" in expr and "i.instrument_id = r.instrument_id" in expr and "i.judge_id" in expr
    sel = prun._anchor_select_sql(prun.BORROW_IN_VIEW_FIELDS)
    assert expr + " AS instrument_judge_id" in sel and "NULL AS required_judge_id" in sel and "r.instrument_id" in sel
    assert sel.count(" AS ") == len(derived) + len(selection)
    assert "identity_diff_lines" in inspect.getsource(prun.verify_borrowed_anchor_identity)
    assert "identity_diff_lines" in inspect.getsource(prun.base_screen_borrow) and "identity_diff_lines" in inspect.getsource(prun.registry_worklist_runs)
    # the selected rows are re-checked UNCONDITIONALLY before the first write in registry mode, borrow or not, with the one function
    assert "recheck_selected_rows(analysis, args, registry_rows, spec)" in inspect.getsource(prun.cmd_analyze)
    # EACH run is re-read at its own filing boundary, before its scope is entered, with the one function
    loop11 = inspect.getsource(prun.cmd_analyze)
    assert loop11.index("recheck_run_before_filing(args, r, spec)") < loop11.index("scope = registry_wl.scope(") < loop11.index("with scope:")
    assert "verify_borrowed_anchor_identity" in inspect.getsource(prun.recheck_run_before_filing) and "BORROW_IN_VIEW_IDENTITY_FIELDS" in inspect.getsource(prun.recheck_run_before_filing)
    assert "verify_borrowed_anchor_identity" in inspect.getsource(prun.recheck_selected_rows) and "BORROW_IN_VIEW_IDENTITY_FIELDS" in inspect.getsource(prun.recheck_selected_rows)
    assert "_pending_judges" in inspect.getsource(prun._anchor_identity_rows) and "_pending_judges" in inspect.getsource(prun.base_screen_borrow)
    assert "v_pending_judgment" in inspect.getsource(prun._pending_judges)
    assert set(prun.BORROW_IN_VIEW_FIELDS) == set(in_view_ident) | {"view_legacy"} and "view_legacy" in allow
    # every registry-row access on the filing path goes through the one accessor, and every field it names is in FILING_ROW_FIELDS
    for fn in (prun._cell_of, prun._behaviour_of, prun._harness_cell_of, prun.petri_item_rows, prun.petri_metric_rows, prun.petri_judgment_notes):
        src = inspect.getsource(fn)
        assert not re.search(r"\brow\.get\(|\brow\[", src), fn.__name__
        assert set(re.findall(r"filing_field\(\s*row\s*,\s*['\"]([a-z_]+)['\"]", src)) <= filing, fn.__name__
    loop = inspect.getsource(prun.cmd_analyze)
    loop = loop[loop.index("for r in registry_rows:"):loop.index("scope = registry_wl.scope(")]
    assert not re.search(r"\br\.get\(|\br\[", loop) and set(re.findall(r"filing_field\(\s*r\s*,\s*['\"]([a-z_]+)['\"]", loop)) <= filing
    assert "FILING_ROW_FIELDS" in inspect.getsource(prun.filing_field)
    with pytest.raises(prun.DriverError, match="outside FILING_ROW_FIELDS"):
        prun.filing_field({"run_id": "x", "judge_id": "y"}, "judge_id")
    assert "BORROW_IN_VIEW_IDENTITY_FIELDS" in inspect.getsource(prun.recheck_borrowed_anchors_current)
    captured = anchors["in_view_runs_at_selection"]
    assert set(captured) == set(_ids.values()) and all(set(t) == set(prun.BORROW_IN_VIEW_FIELDS) for t in captured.values())
    rc = anchors["in_view_recheck"]
    assert rc["fields"] == list(in_view_ident) and rc["snapshot_results_json_sha256"] is None and rc["subject"] == "in-view run"
    assert set(rc["runs"]) == set(captured)
    assert all({f: rc["runs"][rid][f] for f in in_view_ident} == {f: captured[rid][f] for f in in_view_ident} for rid in captured)
    assert anchors["currency_recheck"]["subject"] == "anchor" and anchors["identity_recheck_after_snapshot"]["subject"] == "anchor"
    for b in BEHAVIORS3:
        assert (anchors["runs"][b]["campaign"], anchors["runs"][b]["model_id"], anchors["runs"][b]["model_revision"]) == (synth.CAMPAIGN, synth.MODEL_ID, None)
        for stage_key in ("identity_recheck_after_snapshot", "currency_recheck"):
            assert {k: anchors[stage_key]["runs"][b][k] for k in ("campaign", "model_id", "model_revision")} == {"campaign": synth.CAMPAIGN, "model_id": synth.MODEL_ID, "model_revision": None}


# ============================================================================================================
def _correct_run_record(root: str, b: str, u0_root: Path, *, k: int, i0: int = 300, key_prefix: str = "k_u0_none", at: int = 980, **changed) -> str:
    """A same-id corrective FULL `run` record for behaviour ``b``'s current `none` run, the real shape: the reducer applies every non-event-owned
    field of a later full record (reducer.fold; lifecycle_state and the state-stamped *_utc columns are the only protected ones, and the run_patch
    immutable-field rule does not bind full records), so the record carries the ORIGINAL payload -- the same overrides `_run` built it from; the
    completion pair is untouched because a full run record does not carry it -- plus ``changed``. Folded; returns the run id."""
    eval_id = f"petri_{b}"
    inst = _instrument(eval_id)
    key = f"{key_prefix}_b_{b}"
    rid = I.run_id(synth.CAMPAIGN, key, "none", eval_id, str(inst["instrument_id"]))
    payload = synth.run_payload(key, "none", "none", i0 + 20 * k + 10, run_id=rid, eval_id=eval_id, item_set_id=f"{eval_id}_set",
                                instrument_id=inst["instrument_id"], store_uri=str(u0_root / "cells" / _run_key_of("none")), view_legacy=U0_TAG,
                                **changed)
    R.write_records(os.path.join(root, "records", "c", f"{at:06d}-correct.jsonl"), [synth.rec("run", payload, synth.t(at))])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    return rid


def _registry_row(root: str, run_id: str) -> dict:
    with DB.reading(DB.db_path(root)) as conn:
        r = conn.execute("SELECT r.campaign, r.model_id, r.model_revision, r.store_completion_sha256, vc.is_current FROM run r "
                         "JOIN v_run_current vc ON vc.run_id = r.run_id WHERE r.run_id = ?", (run_id,)).fetchone()
    return dict(zip(("campaign", "model_id", "model_revision", "store_completion_sha256", "is_current"), r))


def test_a_same_id_full_record_correction_of_an_anchor_after_the_snapshot_refuses_at_both_checkpoints(tmp_path, monkeypatch, capsys):
    """A same-id corrective FULL run record restates campaign / model_id / model_revision with the completion uri/sha untouched and
    the run still current. Every one of them is identity now (derived from the SELECT list): the correction refuses by name at checkpoint 2
    (the filing; campaign), at checkpoint 1 (right after the snapshot verification; model_revision), and in registry mode (model_id: nothing
    written or filed, every claim released); and at SELECTION an anchor whose registered revision is not the manifest's is not this view's
    baseline."""
    behaviors = list(BEHAVIORS3)                                   # _u0_none_runs enumerates them in this order (k)
    # checkpoint 2, from-disk: the campaign restated after the snapshot was verified
    in_root, u0_root = _world(tmp_path)
    _ids, in_recs = _in_view_runs(in_root)
    cur, _failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_path, in_recs + u0_recs)
    before = _registry_row(root, cur["defer_to_users"])
    real = prun._analysis_under_lock

    def correcting(*a, _real=real, _root=root, _u0=u0_root, **k):
        analysis = _real(*a, **k)
        _correct_run_record(_root, "defer_to_users", _u0, k=behaviors.index("defer_to_users"), campaign="othercampaign")
        return analysis
    monkeypatch.setattr(prun, "_analysis_under_lock", correcting)
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(_args(in_root, root), _spec())
    msg = str(e.value)
    assert "anchor identity changed between the binding and the filing" in msg and cur["defer_to_users"] in msg
    assert f"campaign: {synth.CAMPAIGN!r} -> 'othercampaign'" in msg
    after = _registry_row(root, cur["defer_to_users"])
    assert after["campaign"] == "othercampaign" and after["is_current"] == 1 and after["store_completion_sha256"] == before["store_completion_sha256"]
    assert not (in_root / prun.ANALYSIS_JSON).exists()
    monkeypatch.undo()
    # checkpoint 1: the model revision restated right after the snapshot verification
    tmp_b = tmp_path / "b"
    in_root, u0_root = _world(tmp_b)
    _ids, in_recs = _in_view_runs(in_root)
    cur, _failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_b, in_recs + u0_recs)
    real_snapshot = prun._borrowed_base_snapshot

    def correcting_snapshot(*a, _real=real_snapshot, _root=root, _u0=u0_root, **k):
        result = _real(*a, **k)
        _correct_run_record(_root, "self_preservation", _u0, k=behaviors.index("self_preservation"), model_revision="e" * 40, at=985)
        return result
    monkeypatch.setattr(prun, "_borrowed_base_snapshot", correcting_snapshot)
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(_args(in_root, root), _spec())
    msg = str(e.value)
    assert "anchor identity changed between the binding and the snapshot verification" in msg and cur["self_preservation"] in msg
    assert f"model_revision: 'None' -> {('e' * 40)[:48]!r}" in msg and _registry_row(root, cur["self_preservation"])["is_current"] == 1
    assert not (in_root / prun.ANALYSIS_JSON).exists()
    monkeypatch.undo()
    # registry mode: the model restated after the snapshot -- nothing written, nothing filed, every claim released
    tmp_r = tmp_path / "registry_mode"
    in_root, u0_root = _world(tmp_r)
    in_ids, in_recs = _in_view_runs(in_root)
    cur, _failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_r, in_recs + u0_recs)
    _registry_mode(monkeypatch, root)
    monkeypatch.setattr(prun, "_consolidate_store", lambda r, s: r / "rows.parquet")

    def correcting2(*a, _real=real, _root=root, _u0=u0_root, **k):
        analysis = _real(*a, **k)
        _correct_run_record(_root, "instruction_evasion", _u0, k=behaviors.index("instruction_evasion"), model_id=synth.FIT_MODEL_ID, at=990)
        return analysis
    monkeypatch.setattr(prun, "_analysis_under_lock", correcting2)
    with pytest.raises(prun.DriverError, match=f"model_id: {synth.MODEL_ID!r} -> {synth.FIT_MODEL_ID!r}"):
        prun.cmd_analyze(_args(in_root, root, worklist="registry"), _spec())
    err = capsys.readouterr().err
    assert not (in_root / prun.ANALYSIS_JSON).exists() and f"released {len(BEHAVIORS3)} of {len(BEHAVIORS3)} claim(s)" in err
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM judgment WHERE status = 'complete'").fetchone()[0] == 0
        for rid in in_ids.values():
            assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "generated"
    monkeypatch.undo()
    # at selection: an anchor whose registered revision is not the manifest's is not this view's baseline
    tmp_s = tmp_path / "selection"
    in_root, u0_root = _world(tmp_s)
    _ids, in_recs = _in_view_runs(in_root)
    cur, _failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_s, in_recs + u0_recs)
    _correct_run_record(root, "defer_to_users", u0_root, k=behaviors.index("defer_to_users"), model_revision="e" * 40, at=995)
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(_args(in_root, root), _spec())
    msg = str(e.value)
    assert "is not this view's baseline" in msg and cur["defer_to_users"] in msg
    assert f"model_revision {('e' * 40)[:12]} != the manifest's target_revision {REV[:12]}" in msg
    assert not (in_root / prun.BORROWED_BASE_DIR).exists() and not (in_root / prun.ANALYSIS_JSON).exists()


# ============================================================================================================
def _correct_in_view_run_record(root: str, b: str, in_root: Path, *, at: int = 975, **changed) -> str:
    """A same-id corrective FULL `run` record for behaviour ``b``'s IN-VIEW (steered) run -- the same shape as _correct_run_record, over the
    overrides _in_view_runs built it from (the completion pair untouched: a full run record does not carry it); folded. Returns the run id."""
    eval_id = f"petri_{b}"
    inst = _instrument(eval_id)
    key = f"k_in_{b}"
    k = list(PETRI_EVALS.values()).index(b)
    rid = I.run_id(synth.CAMPAIGN, key, STEERED_TT, eval_id, str(inst["instrument_id"]))
    over = dict(run_id=rid, eval_id=eval_id, item_set_id=f"{eval_id}_set", instrument_id=inst["instrument_id"],
                store_uri=str(in_root / "cells" / _run_key_of(STEERED)), view_legacy=IN_TAG)
    over.update(changed)                                     # a correction may restate a routing field the original overrides set
    payload = synth.run_payload(key, STEERED_TT, synth.OBJECT_SHA, 100 + 10 * k, **over)
    R.write_records(os.path.join(root, "records", "c", f"{at:06d}-correct-in-view.jsonl"), [synth.rec("run", payload, synth.t(at))])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    return rid


def test_a_same_id_full_record_correction_of_an_in_view_run_before_the_filing_refuses_and_releases_the_claims(tmp_path, monkeypatch, capsys):
    """The judged view's own runs were checked at selection only, and only the anchors were re-read before the filing. Now each in-view
    run's identity tuple, captured at selection, is re-read with the SAME function immediately before the filing: a same-id corrective
    FULL record restating an in-view run's revision (from-disk) or campaign (registry mode) after the borrow refuses by name -- nothing
    written, nothing filed, every claim released -- while the run stays current (a currency check alone would have passed)."""
    # from-disk: the revision restated after the borrow
    in_root, u0_root = _world(tmp_path)
    in_ids, in_recs = _in_view_runs(in_root)
    _cur, _failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_path, in_recs + u0_recs)
    real = prun._analysis_under_lock

    def correcting(*a, _real=real, _root=root, _in=in_root, **k):
        analysis = _real(*a, **k)
        _correct_in_view_run_record(_root, "defer_to_users", _in, model_revision="e" * 40)
        return analysis
    monkeypatch.setattr(prun, "_analysis_under_lock", correcting)
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(_args(in_root, root), _spec())
    msg = str(e.value)
    assert "in-view run identity changed between the binding and the filing" in msg and in_ids["defer_to_users"] in msg
    assert f"model_revision: 'None' -> {('e' * 40)[:48]!r}" in msg and not (in_root / prun.ANALYSIS_JSON).exists()
    assert _registry_row(root, in_ids["defer_to_users"])["is_current"] == 1
    monkeypatch.undo()
    # registry mode: the campaign restated -- nothing written, nothing filed, every claim released
    tmp_r = tmp_path / "registry_mode"
    in_root, u0_root = _world(tmp_r)
    in_ids, in_recs = _in_view_runs(in_root)
    _cur, _failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_r, in_recs + u0_recs)
    _registry_mode(monkeypatch, root)
    monkeypatch.setattr(prun, "_consolidate_store", lambda r, s: r / "rows.parquet")

    def correcting2(*a, _real=real, _root=root, _in=in_root, **k):
        analysis = _real(*a, **k)
        _correct_in_view_run_record(_root, "instruction_evasion", _in, campaign="othercampaign", at=977)
        return analysis
    monkeypatch.setattr(prun, "_analysis_under_lock", correcting2)
    with pytest.raises(prun.DriverError, match=f"in-view run identity changed between the binding and the filing: .*campaign: {synth.CAMPAIGN!r} -> 'othercampaign'"):
        prun.cmd_analyze(_args(in_root, root, worklist="registry"), _spec())
    err = capsys.readouterr().err
    assert not (in_root / prun.ANALYSIS_JSON).exists() and f"released {len(BEHAVIORS3)} of {len(BEHAVIORS3)} claim(s)" in err
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM judgment WHERE status = 'complete'").fetchone()[0] == 0
        for rid in in_ids.values():
            assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "generated"


# ============================================================================================================
def test_a_same_id_full_record_correction_of_an_in_view_runs_routing_fields_refuses_at_the_filing(tmp_path, monkeypatch, capsys):
    """The earlier in-view identity left eval_id and producer_object_key_legacy uncompared, so a same-id FULL correction could
    re-route an in-view run's judged rows (another seeded behaviour; another cell) without touching any compared field, the completion pair or
    run_id. The identity now DERIVES from the fields the filing path reads: an eval_id switched to another seeded Petri behaviour (from-disk)
    and a producer key switched to another cell (registry mode) each refuse by name before the first write -- nothing written, nothing filed,
    the claims released, every in-view run still `generated`."""
    # from-disk: eval_id switched to another seeded behaviour (with that behaviour's item set and instrument, the consistent trio)
    in_root, u0_root = _world(tmp_path)
    in_ids, in_recs = _in_view_runs(in_root)
    _cur, _failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_path, in_recs + u0_recs)
    real = prun._analysis_under_lock
    other = "petri_self_preservation"

    def rerouting(*a, _real=real, _root=root, _in=in_root, **k):
        analysis = _real(*a, **k)
        _correct_in_view_run_record(_root, "defer_to_users", _in, eval_id=other, item_set_id=f"{other}_set",
                                    instrument_id=_instrument(other)["instrument_id"])
        return analysis
    monkeypatch.setattr(prun, "_analysis_under_lock", rerouting)
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(_args(in_root, root), _spec())
    msg = str(e.value)
    assert "in-view run identity changed between the binding and the filing" in msg and in_ids["defer_to_users"] in msg
    assert f"eval_id: 'petri_defer_to_users' -> {other!r}" in msg and not (in_root / prun.ANALYSIS_JSON).exists()
    assert _registry_row(root, in_ids["defer_to_users"])["is_current"] == 1
    monkeypatch.undo()
    # registry mode: the producer key switched to another cell -- nothing written, nothing filed, every claim released
    tmp_r = tmp_path / "registry_mode"
    in_root, u0_root = _world(tmp_r)
    in_ids, in_recs = _in_view_runs(in_root)
    _cur, _failed, u0_recs = _u0_none_runs(u0_root)
    root = _registry(tmp_r, in_recs + u0_recs)
    _registry_mode(monkeypatch, root)
    monkeypatch.setattr(prun, "_consolidate_store", lambda r, s: r / "rows.parquet")
    other_cell = "constv2ctl2-full__psr_ce_all"

    def rerouting2(*a, _real=real, _root=root, _in=in_root, **k):
        analysis = _real(*a, **k)
        _correct_in_view_run_record(_root, "instruction_evasion", _in, producer_object_key_legacy=other_cell, at=978)
        return analysis
    monkeypatch.setattr(prun, "_analysis_under_lock", rerouting2)
    with pytest.raises(prun.DriverError, match=f"producer_object_key_legacy: {STEERED_TT!r} -> {other_cell!r}"):
        prun.cmd_analyze(_args(in_root, root, worklist="registry"), _spec())
    err = capsys.readouterr().err
    assert not (in_root / prun.ANALYSIS_JSON).exists() and f"released {len(BEHAVIORS3)} of {len(BEHAVIORS3)} claim(s)" in err
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM judgment WHERE status = 'complete'").fetchone()[0] == 0
        for rid in in_ids.values():
            assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "generated"
        assert conn.execute("SELECT producer_object_key_legacy FROM run WHERE run_id = ?", (in_ids["instruction_evasion"],)).fetchone()[0] == other_cell


# ============================================================================================================
def _patch_in_view_run(root: str, run_id: str, *, at: int = 970, **fields) -> None:
    """A ``run_patch`` on an in-view run (the producer's corrective shape for a PATCHABLE field such as store_uri), folded."""
    R.write_records(os.path.join(root, "records", "c", f"{at:06d}-patch-in-view.jsonl"),
                    [synth.rec("run_patch", {"run_id": run_id, "fields": fields}, synth.t(at))])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()


def test_a_selected_run_dropped_from_the_judged_view_between_the_worklist_read_and_the_borrow_refuses_before_any_claim(tmp_path, monkeypatch, capsys):
    """cmd_analyze reads the worklist (registry_rows), then base_screen_borrow RE-QUERIES the runs (eval_id LIKE
    'petri_%', store under this out-root) and keeps only those -- a same-id correction folded BETWEEN the two reads that moves one selected run's
    store outside the root (a run_patch) or changes its eval to a non-Petri eval (a full record) dropped it from in_view silently while the
    siblings kept the borrow valid; in_view_runs_at_selection and the filing recheck covered the survivors only, and the filing loop still filed
    every ORIGINAL worklist run. Now every selected run id must be in the in-view set: both corrections refuse BY NAME before any claim -- zero
    claims, zero filings, nothing written, the siblings still pending."""
    real = prun.registry_worklist_runs

    def world(sub: str):
        in_root, u0_root = _world(tmp_path / sub)
        in_ids, in_recs = _in_view_runs(in_root)
        _cur, _failed, u0_recs = _u0_none_runs(u0_root)
        root = _registry(tmp_path / sub, in_recs + u0_recs)
        _registry_mode(monkeypatch, root)
        monkeypatch.setattr(prun, "_consolidate_store", lambda r, s: r / "rows.parquet")
        return in_root, in_ids, root

    def nothing_claimed_or_filed(in_root, in_ids, root, dropped_b: str) -> None:
        err = capsys.readouterr().err
        assert "claim lock held" not in err and "released" not in err and not (in_root / prun.ANALYSIS_JSON).exists()
        with DB.reading(DB.db_path(root)) as conn:
            assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 0
            reasons = {str(r["run_id"]): r["pending_reason"] for r in P.pending(conn)}
            assert "claimed" not in reasons.values()
            for b, rid in in_ids.items():
                assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "generated"
                if b != dropped_b:
                    assert rid in reasons, (b, reasons)                    # the siblings are still this audit's pending runs

    # case 1: a run_patch moves one selected run's store_uri outside the out-root between the worklist read and the borrow
    in_root, in_ids, root = world("case1")
    elsewhere = str(tmp_path / "case1" / "elsewhere" / "cells" / _run_key_of(STEERED))

    def moving(*a, _real=real, _root=root, _rid=in_ids["defer_to_users"], _to=elsewhere, _all=set(in_ids.values()), **k):
        wl, rows = _real(*a, **k)                                            # the worklist read: every in-view run selected
        assert {str(r["run_id"]) for r in rows} == _all
        _patch_in_view_run(_root, _rid, store_uri=_to)                        # folded between the two reads
        return wl, rows
    monkeypatch.setattr(prun, "registry_worklist_runs", moving)
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(_args(in_root, root, worklist="registry"), _spec())
    msg = str(e.value)
    assert "selected run dropped from the judged view between the worklist read and the borrow" in msg and "refusing before any claim" in msg
    assert f"{in_ids['defer_to_users']}: store_uri/eval_id now {elsewhere!r} / 'petri_defer_to_users'" in msg and "is not under" in msg
    assert "1 of 3 selected run(s) missing from the 2 registered under" in msg
    for other in ("self_preservation", "instruction_evasion"):
        assert in_ids[other] not in msg                                      # only the moved run is named
    nothing_claimed_or_filed(in_root, in_ids, root, "defer_to_users")
    monkeypatch.undo()
    # case 2: a same-id FULL record changes one selected run's eval to the seeded NON-Petri eval (with its item set and instrument)
    in_root, in_ids, root = world("case2")

    def re_evaling(*a, _real=real, _root=root, _in=in_root, **k):
        wl, rows = _real(*a, **k)
        _correct_in_view_run_record(_root, "instruction_evasion", _in, eval_id=synth.EVAL_ID, item_set_id=synth.ITEM_SET_ID,
                                    instrument_id=synth.INSTRUMENT_ID, at=979)
        return wl, rows
    monkeypatch.setattr(prun, "registry_worklist_runs", re_evaling)
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(_args(in_root, root, worklist="registry"), _spec())
    msg = str(e.value)
    cell_dir = str(in_root / "cells" / _run_key_of(STEERED))
    assert "selected run dropped from the judged view between the worklist read and the borrow" in msg
    assert f"{in_ids['instruction_evasion']}: store_uri/eval_id now {cell_dir!r} / {synth.EVAL_ID!r}" in msg and "no longer a Petri eval" in msg
    assert "is not under" not in msg                                         # the store stayed; the eval moved
    nothing_claimed_or_filed(in_root, in_ids, root, "instruction_evasion")
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT eval_id FROM run WHERE run_id = ?", (in_ids["instruction_evasion"],)).fetchone()[0] == synth.EVAL_ID


# ============================================================================================================
def test_a_selected_runs_identity_changed_between_the_worklist_read_and_the_borrow_refuses_by_field_before_any_claim(tmp_path, monkeypatch, capsys):
    """The earlier guard compared selected run IDS only. A same-id FULL correction folded after the worklist read and
    before the borrow's SELECT that moves a selected run's producer key to another cell (or its eval to another petri_*) with the store still
    under the root kept the id present; in_view_runs_at_selection captured the NEW identity, the filing recheck compared new-to-new, and the
    filing loop routed from the STALE worklist row -- a completed judgment for the wrong cell / behaviour. Now the borrow receives the worklist
    ROWS, compares every in-view identity field of each against its in-view SELECT row before any claim, and records the worklist rows as the
    baseline the filing re-reads against: both corrections refuse by name with the field diff -- zero claims, zero filings, nothing written. The
    same correction folded between the worklist's own pending read and its identity read refuses there (the rows are never a blend)."""
    real = prun.registry_worklist_runs
    other_cell = "constv2ctl2-full__psr_ce_all"

    def world(sub: str):
        in_root, u0_root = _world(tmp_path / sub)
        in_ids, in_recs = _in_view_runs(in_root)
        _cur, _failed, u0_recs = _u0_none_runs(u0_root)
        root = _registry(tmp_path / sub, in_recs + u0_recs)
        _registry_mode(monkeypatch, root)
        monkeypatch.setattr(prun, "_consolidate_store", lambda r, s: r / "rows.parquet")
        return in_root, in_ids, root

    def nothing_claimed_or_filed(in_root, in_ids, root) -> None:
        err = capsys.readouterr().err
        assert "claim lock held" not in err and "released" not in err and not (in_root / prun.ANALYSIS_JSON).exists()
        with DB.reading(DB.db_path(root)) as conn:
            assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 0
            assert "claimed" not in {r["pending_reason"] for r in P.pending(conn)}
            for rid in in_ids.values():
                assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "generated"

    # case 1: the producer key moved to another cell between the worklist read and the borrow, the store still under the root
    in_root, in_ids, root = world("key")

    def rekeying(*a, _real=real, _root=root, _in=in_root, **k):
        wl, rows = _real(*a, **k)                                            # the worklist read: every in-view identity field on each row
        assert rows and all(set(prun.BORROW_IN_VIEW_FIELDS) <= set(r) for r in rows)
        _correct_in_view_run_record(_root, "defer_to_users", _in, producer_object_key_legacy=other_cell)
        return wl, rows
    monkeypatch.setattr(prun, "registry_worklist_runs", rekeying)
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(_args(in_root, root, worklist="registry"), _spec())
    msg = str(e.value)
    assert "selected run's identity changed between the worklist read and the borrow" in msg and "refusing before any claim" in msg
    assert f"{in_ids['defer_to_users']}: producer_object_key_legacy: {STEERED_TT!r} -> {other_cell!r}" in msg
    assert "selected run dropped" not in msg                                 # present (the store stayed under the root): identity, not presence
    for other in ("self_preservation", "instruction_evasion"):
        assert in_ids[other] not in msg                                      # only the re-keyed run is named
    nothing_claimed_or_filed(in_root, in_ids, root)
    monkeypatch.undo()
    # case 2: the eval moved to another petri_* (with its item set and instrument) -- still admitted by the in-view SELECT's LIKE 'petri_%'
    in_root, in_ids, root = world("eval")
    other = "petri_self_preservation"

    def re_evaling(*a, _real=real, _root=root, _in=in_root, **k):
        wl, rows = _real(*a, **k)
        _correct_in_view_run_record(_root, "instruction_evasion", _in, eval_id=other, item_set_id=f"{other}_set",
                                    instrument_id=_instrument(other)["instrument_id"], at=981)
        return wl, rows
    monkeypatch.setattr(prun, "registry_worklist_runs", re_evaling)
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(_args(in_root, root, worklist="registry"), _spec())
    msg = str(e.value)
    assert "selected run's identity changed between the worklist read and the borrow" in msg and "selected run dropped" not in msg
    assert f"{in_ids['instruction_evasion']}: eval_id: 'petri_instruction_evasion' -> {other!r}" in msg
    nothing_claimed_or_filed(in_root, in_ids, root)
    monkeypatch.undo()
    # case 3: the same correction folded between the worklist's PENDING read and its identity read (two connections inside registry_worklist_runs):
    # refused by name there, before the borrow -- the rows are never a blend of two reads
    from sisteer.registry import worklist as RW
    in_root, in_ids, root = world("pending")
    real_pending = RW.RegistryWorklist.pending
    fired: list = []

    def racing(self, _real=real_pending, _root=root, _in=in_root, **filters):
        rows = _real(self, **filters)
        if not fired:                                                        # once, right after the first pending read returns
            fired.append(1)
            _correct_in_view_run_record(_root, "self_preservation", _in, producer_object_key_legacy=other_cell, at=983)
        return rows
    monkeypatch.setattr(RW.RegistryWorklist, "pending", racing)
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(_args(in_root, root, worklist="registry"), _spec())
    msg = str(e.value)
    assert "selected run's identity changed between the worklist read and its identity read" in msg and "never a blend" in msg
    assert f"{in_ids['self_preservation']}: producer_object_key_legacy: {STEERED_TT!r} -> {other_cell!r}" in msg
    nothing_claimed_or_filed(in_root, in_ids, root)


# ============================================================================================================
def _other_judge_instrument(eval_id: str) -> dict:
    """A second registered instrument for ``eval_id`` that designates the OTHER seeded judge (synth.JUDGE_ID_2): the judge is an instrument
    identity field, so it is a different instrument_id; v_pending_judgment then derives required_judge_id = JUDGE_ID_2 for a run on it."""
    return synth.instrument_fields(eval_id=eval_id, item_set_id=f"{eval_id}_set", judge_id=synth.JUDGE_ID_2)


def _seed_other_judge_instruments(root: str) -> dict:
    """The other-judge instrument for every Petri eval, folded into ``root``; returns {eval_id: instrument_id}."""
    recs = [synth.rec("instrument", _other_judge_instrument(e), synth.t(60 + k)) for k, e in enumerate(PETRI_EVALS)]
    R.write_records(os.path.join(root, "records", "dims", "000002-dims-other-judge.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    return {e: str(_other_judge_instrument(e)["instrument_id"]) for e in PETRI_EVALS}


def test_a_selected_run_switched_to_another_instruments_judge_refuses_at_every_seam_before_any_claim(tmp_path, monkeypatch, capsys):
    """The two-read agreement and the borrow / filing rechecks omitted `run.instrument_id` and dropped
    required_judge_id as "pending-view-only". A same-id corrective FULL run record switching ONLY instrument_id to an instrument that designates
    another registered judge -- cell, eval, store, n_items_generated and every compared field unchanged -- left the retained row carrying the OLD
    judge while v_pending_judgment's required_judge_id (the CURRENT instrument's judge) moved, and RegistryWorklist.claim takes any (run_id,
    judge_id) pair: the grader filed under the old judge, the new required judge stayed ungraded. Now instrument_id is identity and
    required_judge_id is recomputed from the current instrument and compared at every point: the correction folded between the pending read and
    the identity read, between the identity read and the borrow, and after the analysis (the final recheck) refuses by name -- naming
    instrument_id and the judge the current instrument designates -- with zero claims, zero filings, nothing written."""
    real = prun.registry_worklist_runs

    def world(sub: str):
        in_root, u0_root = _world(tmp_path / sub)
        in_ids, in_recs = _in_view_runs(in_root)
        _cur, _failed, u0_recs = _u0_none_runs(u0_root)
        root = _registry(tmp_path / sub, in_recs + u0_recs)
        other_inst = _seed_other_judge_instruments(root)
        _registry_mode(monkeypatch, root)
        monkeypatch.setattr(prun, "_consolidate_store", lambda r, s: r / "rows.parquet")
        return in_root, in_ids, root, other_inst

    def nothing_claimed_or_filed(in_root, in_ids, root) -> None:
        err = capsys.readouterr().err
        assert "claim lock held" not in err and not (in_root / prun.ANALYSIS_JSON).exists()
        with DB.reading(DB.db_path(root)) as conn:
            assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 0
            assert "claimed" not in {r["pending_reason"] for r in P.pending(conn)}
            for rid in in_ids.values():
                assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "generated"

    def expect_judge_switch(msg: str, rid: str, b: str, other_inst: dict) -> None:
        old_inst, new_inst = str(_instrument(f"petri_{b}")["instrument_id"]), other_inst[f"petri_{b}"]
        assert f"{rid}: " in msg and f"instrument_id: {old_inst[:48]!r} -> {new_inst[:48]!r}" in msg, msg
        assert f"required_judge_id: {synth.JUDGE_ID!r} -> current instrument {new_inst} designates {synth.JUDGE_ID_2!r}" in msg, msg
        for other in (x for x in BEHAVIORS3 if x != b):
            assert in_ids[other] not in msg                                  # only the switched run is named

    # seam 1: between the worklist's pending read and its identity read (two connections inside registry_worklist_runs)
    from sisteer.registry import worklist as RW
    in_root, in_ids, root, other_inst = world("pending")
    real_pending = RW.RegistryWorklist.pending
    fired: list = []

    def racing(self, _real=real_pending, _root=root, _in=in_root, _inst=other_inst, **filters):
        rows = _real(self, **filters)
        if not fired:                                                        # once, right after the first pending read returns
            fired.append(1)
            _correct_in_view_run_record(_root, "defer_to_users", _in, instrument_id=_inst["petri_defer_to_users"], at=985)
        return rows
    monkeypatch.setattr(RW.RegistryWorklist, "pending", racing)
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(_args(in_root, root, worklist="registry"), _spec())
    msg = str(e.value)
    assert "selected run's identity changed between the worklist read and its identity read" in msg
    expect_judge_switch(msg, in_ids["defer_to_users"], "defer_to_users", other_inst)
    nothing_claimed_or_filed(in_root, in_ids, root)
    monkeypatch.undo()
    # seam 2: between the identity read and the borrow's SELECT
    in_root, in_ids, root, other_inst = world("borrow")

    def switching(*a, _real=real, _root=root, _in=in_root, _inst=other_inst, **k):
        wl, rows = _real(*a, **k)
        assert all(r["instrument_id"] and r["required_judge_id"] == synth.JUDGE_ID for r in rows)   # the baseline carries instrument + judge
        _correct_in_view_run_record(_root, "instruction_evasion", _in, instrument_id=_inst["petri_instruction_evasion"], at=986)
        return wl, rows
    monkeypatch.setattr(prun, "registry_worklist_runs", switching)
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(_args(in_root, root, worklist="registry"), _spec())
    msg = str(e.value)
    assert "selected run's identity changed between the worklist read and the borrow" in msg and "selected run dropped" not in msg
    expect_judge_switch(msg, in_ids["instruction_evasion"], "instruction_evasion", other_inst)
    nothing_claimed_or_filed(in_root, in_ids, root)
    monkeypatch.undo()
    # the final recheck: the same correction after the analysis, before the first write -- refused there, every claim released
    in_root, in_ids, root, other_inst = world("filing")
    real_analysis = prun._analysis_under_lock

    def switching_late(*a, _real=real_analysis, _root=root, _in=in_root, _inst=other_inst, **k):
        analysis = _real(*a, **k)
        _correct_in_view_run_record(_root, "self_preservation", _in, instrument_id=_inst["petri_self_preservation"], at=987)
        return analysis
    monkeypatch.setattr(prun, "_analysis_under_lock", switching_late)
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(_args(in_root, root, worklist="registry"), _spec())
    msg = str(e.value)
    assert "in-view run identity changed between the binding and the filing" in msg
    expect_judge_switch(msg, in_ids["self_preservation"], "self_preservation", other_inst)
    err = capsys.readouterr().err
    assert not (in_root / prun.ANALYSIS_JSON).exists() and f"released {len(BEHAVIORS3)} of {len(BEHAVIORS3)} claim(s)" in err
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM judgment WHERE status = 'complete'").fetchone()[0] == 0
        for rid in in_ids.values():
            assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "generated"
        assert conn.execute("SELECT instrument_id FROM run WHERE run_id = ?", (in_ids["self_preservation"],)).fetchone()[0] == other_inst["petri_self_preservation"]


# ============================================================================================================
def _correct_registered_run_record(root: str, b: str, cell_dir: Path, *, object_key: str, object_id: str, tag: str, at: int, **changed) -> str:
    """A same-id corrective FULL `run` record for behaviour ``b``'s registered run on ``cell_dir`` (the _in_view_runs shape for any cell / object
    / tag -- the own-base `none` runs included), the real reducer shape; folded. Returns the run id."""
    eval_id = f"petri_{b}"
    inst = _instrument(eval_id)
    key = f"k_in_{b}"
    k = list(PETRI_EVALS.values()).index(b)
    rid = I.run_id(synth.CAMPAIGN, key, object_key, eval_id, str(inst["instrument_id"]))
    over = dict(run_id=rid, eval_id=eval_id, item_set_id=f"{eval_id}_set", instrument_id=inst["instrument_id"], store_uri=str(cell_dir), view_legacy=tag)
    over.update(changed)
    payload = synth.run_payload(key, object_key, object_id, 100 + 10 * k, **over)
    R.write_records(os.path.join(root, "records", "c", f"{at:06d}-correct-registered.jsonl"), [synth.rec("run", payload, synth.t(at))])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    return rid


def test_the_own_base_u0_analysis_rechecks_the_selected_rows_before_filing_and_refuses_a_judge_switch(tmp_path, monkeypatch, capsys):
    """The OWN-BASE path (the view serves `none`) had no post-claim identity recheck -- base_screen_borrow returns None
    before recording in_view_runs_at_selection and recheck_borrowed_anchors_current returns at once without borrowed_base -- so a same-id
    instrument / judge correction after the worklist's agreement (even after the claims) but before the filing filed under the old required judge.
    Now the selected rows are re-checked UNCONDITIONALLY before the first write: a steady u0 analysis files and records selected_rows_recheck;
    the same correction after the analysis refuses by name, nothing filed, every claim released."""
    def world(sub: str):
        in_root, _u0 = _world(tmp_path / sub, in_cells=("none", "prompted__safety"), in_audit=U0_AUDIT, with_u0_store=False)   # the store IS u0
        in_ids, in_recs = _in_view_runs(in_root, tag=U0_TAG, cell="none", object_key="none", object_id="none")
        root = _registry(tmp_path / sub, in_recs)
        other_inst = _seed_other_judge_instruments(root)
        _registry_mode(monkeypatch, root)
        monkeypatch.setattr(prun, "_consolidate_store", lambda r, s: r / "rows.parquet")
        return in_root, in_ids, root, other_inst
    spec = _spec(audit=U0_AUDIT, cells=("none", "prompted__safety"))
    # steady: the own-base analysis files its `none` runs and records the unconditional recheck (no borrow anywhere)
    in_root, in_ids, root, _other = world("steady")
    assert prun.cmd_analyze(_args(in_root, root, worklist="registry"), spec) == 0
    err = capsys.readouterr().err
    assert "selected run(s) re-checked before filing (3 run(s)" in err and "judge selection instrument, borrowed False" in err
    analysis = json.loads((in_root / prun.ANALYSIS_JSON).read_text())
    assert "borrowed_base" not in analysis
    blk = analysis["selected_rows_recheck"]
    assert blk["subject"] == "selected run" and blk["stage"] == "the filing" and blk["judge_selection"] == "instrument" and blk["borrowed"] is False
    assert set(blk["runs"]) == set(in_ids.values()) and blk["fields"] == list(prun.BORROW_IN_VIEW_IDENTITY_FIELDS)
    for rid in in_ids.values():
        assert blk["runs"][rid]["required_judge_id"] == synth.JUDGE_ID == blk["runs"][rid]["instrument_judge_id"]
        assert synth.JUDGE_ID in blk["runs"][rid]["pending_judge_ids"]
    monkeypatch.undo()
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    with DB.reading(DB.db_path(root)) as conn:
        for rid in in_ids.values():
            assert [tuple(j) for j in conn.execute("SELECT judge_id, status FROM judgment WHERE run_id = ?", (rid,))] == [(synth.JUDGE_ID, "complete")]
    # the race: ONLY instrument_id corrected (to the instrument designating the other judge) after the analysis, before the first write
    in_root, in_ids, root, other_inst = world("race")
    real = prun._analysis_under_lock
    none_dir = in_root / "cells" / _run_key_of("none")

    def switching(*a, _real=real, _root=root, _dir=none_dir, _inst=other_inst, **k):
        analysis = _real(*a, **k)
        _correct_registered_run_record(_root, "defer_to_users", _dir, object_key="none", object_id="none", tag=U0_TAG, at=988,
                                       instrument_id=_inst["petri_defer_to_users"])
        return analysis
    monkeypatch.setattr(prun, "_analysis_under_lock", switching)
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(_args(in_root, root, worklist="registry"), spec)
    msg = str(e.value)
    old_inst, new_inst = str(_instrument("petri_defer_to_users")["instrument_id"]), other_inst["petri_defer_to_users"]
    assert f"registry worklist filing for {U0_AUDIT!r}: selected run identity changed between the binding and the filing" in msg
    assert f"{in_ids['defer_to_users']}: " in msg and f"instrument_id: {old_inst[:48]!r} -> {new_inst[:48]!r}" in msg
    assert f"required_judge_id: {synth.JUDGE_ID!r} -> current instrument {new_inst} designates {synth.JUDGE_ID_2!r}" in msg
    for other in ("self_preservation", "instruction_evasion"):
        assert in_ids[other] not in msg
    err = capsys.readouterr().err
    assert not (in_root / prun.ANALYSIS_JSON).exists() and f"released {len(BEHAVIORS3)} of {len(BEHAVIORS3)} claim(s)" in err
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM judgment WHERE status = 'complete'").fetchone()[0] == 0
        for rid in in_ids.values():
            assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "generated"


def test_an_explicit_judge_selection_is_the_bound_identity_files_normally_and_still_refuses_an_instrument_switch(tmp_path, monkeypatch, capsys):
    """An explicit non-instrument `--judge` selection (pending_for pairs any named judge) was FALSELY refused with no race --
    the pending row's required_judge_id is the selected judge, while the earlier derived subquery always yielded the instrument's. Now the
    SELECTED judge is the bound identity, verified to stay a pending (run, judge) pair; only the default worklist requires the instrument to
    designate it; and instrument_id is compared in both modes: a steady `--judge J2` analysis of the borrowing steered view files under J2 with
    no refusal, and an instrument switch under `--judge J2` still refuses -- on instrument_id, never on "designates"."""
    def world(sub: str):
        in_root, u0_root = _world(tmp_path / sub)
        in_ids, in_recs = _in_view_runs(in_root)
        _cur, _failed, u0_recs = _u0_none_runs(u0_root)
        root = _registry(tmp_path / sub, in_recs + u0_recs)
        other_inst = _seed_other_judge_instruments(root)
        _registry_mode(monkeypatch, root)
        monkeypatch.setattr(prun, "_consolidate_store", lambda r, s: r / "rows.parquet")
        a = _args(in_root, root, worklist="registry")
        a.judge = [synth.JUDGE_ID_2]                                          # the explicit, non-instrument judge
        return in_root, in_ids, root, other_inst, a
    # steady: files under J2, every checkpoint reads the bound judge, nothing refused
    in_root, in_ids, root, _other, a = world("steady")
    assert prun.cmd_analyze(a, _spec()) == 0
    err = capsys.readouterr().err
    assert "identity changed" not in err and "judge selection explicit, borrowed True" in err and err.count("[registry] claim lock held ") == len(BEHAVIORS3)
    analysis = json.loads((in_root / prun.ANALYSIS_JSON).read_text())
    anchors = analysis["borrowed_base"]["anchors"]
    assert anchors["worklist_selection"]["judge_selection"] == "explicit"
    for rid in in_ids.values():
        assert anchors["in_view_runs_at_selection"][rid]["required_judge_id"] == synth.JUDGE_ID_2
        assert anchors["in_view_runs_at_selection"][rid]["instrument_judge_id"] == synth.JUDGE_ID       # the instrument's judge, compared, not required
        assert anchors["in_view_recheck"]["runs"][rid]["required_judge_id"] == synth.JUDGE_ID_2 and synth.JUDGE_ID_2 in anchors["in_view_recheck"]["runs"][rid]["pending_judge_ids"]
    blk = analysis["selected_rows_recheck"]
    assert blk["judge_selection"] == "explicit" and blk["borrowed"] is True and set(blk["runs"]) == set(in_ids.values())
    monkeypatch.undo()
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    with DB.reading(DB.db_path(root)) as conn:
        for rid in in_ids.values():
            assert [tuple(j) for j in conn.execute("SELECT judge_id, status FROM judgment WHERE run_id = ?", (rid,))] == [(synth.JUDGE_ID_2, "complete")]
    # an instrument switch under --judge J2, between the identity read and the borrow: refused on instrument_id (compared in both modes),
    # never on the instrument's designation (an explicit judge is not required to be the instrument's)
    in_root, in_ids, root, other_inst, a = world("switch")
    real = prun.registry_worklist_runs

    def switching(*a_, _real=real, _root=root, _in=in_root, _inst=other_inst, **k):
        wl, rows = _real(*a_, **k)
        assert all(r["required_judge_id"] == synth.JUDGE_ID_2 for r in rows)
        _correct_in_view_run_record(_root, "instruction_evasion", _in, instrument_id=_inst["petri_instruction_evasion"], at=989)
        return wl, rows
    monkeypatch.setattr(prun, "registry_worklist_runs", switching)
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(a, _spec())
    msg = str(e.value)
    old_inst, new_inst = str(_instrument("petri_instruction_evasion")["instrument_id"]), other_inst["petri_instruction_evasion"]
    assert "selected run's identity changed between the worklist read and the borrow" in msg
    assert f"{in_ids['instruction_evasion']}: " in msg and f"instrument_id: {old_inst[:48]!r} -> {new_inst[:48]!r}" in msg
    assert f"instrument_judge_id: {synth.JUDGE_ID[:48]!r} -> {synth.JUDGE_ID_2[:48]!r}" in msg
    assert "designates" not in msg and "no longer a pending" not in msg
    err = capsys.readouterr().err
    assert "claim lock held" not in err and not (in_root / prun.ANALYSIS_JSON).exists()
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 0
        for rid in in_ids.values():
            assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "generated"


# ============================================================================================================
def test_a_correction_folded_between_the_pre_filing_recheck_and_a_later_runs_completion_refuses_that_run_alone(tmp_path, monkeypatch, capsys):
    """recheck_selected_rows runs ONCE before all filings, so a correction folded between it and a LATER run's completion filed that
    run under the old judge / cell. Now each run is re-read at its own filing boundary, immediately before its first write:
    the instrument-only correction of the SECOND run to file, folded after the first run's completion (at the seam where the second run's rows
    are computed), refuses that run by name -- nothing of it written, its claim released -- while the first run's completion stands and the third
    run still files behind its own guard; the invocation fails naming the refused run; every filed judgment carries its filing-time recheck."""
    in_root, _u0 = _world(tmp_path, in_cells=("none", "prompted__safety"), in_audit=U0_AUDIT, with_u0_store=False)       # the own-base u0 store
    in_ids, in_recs = _in_view_runs(in_root, tag=U0_TAG, cell="none", object_key="none", object_id="none")
    root = _registry(tmp_path, in_recs)
    other_inst = _seed_other_judge_instruments(root)
    _registry_mode(monkeypatch, root)
    monkeypatch.setattr(prun, "_consolidate_store", lambda r, s: r / "rows.parquet")
    spec = _spec(audit=U0_AUDIT, cells=("none", "prompted__safety"))
    none_dir = in_root / "cells" / _run_key_of("none")
    real_metrics = prun.petri_metric_rows
    filed_order: list = []
    corrected: dict = {}

    def late_correction(analysis, row, _real=real_metrics, _root=root, _dir=none_dir, _inst=other_inst):
        rows = _real(analysis, row)                                         # the run's own rows, computed before its guard
        filed_order.append(str(row["run_id"]))
        if len(filed_order) == 2:                                           # the FIRST run's completion is already emitted; the second is about to file
            b = prun._behaviour_of(row)
            corrected["b"], corrected["rid"] = b, str(row["run_id"])
            _correct_registered_run_record(_root, b, _dir, object_key="none", object_id="none", tag=U0_TAG, at=991,
                                           instrument_id=_inst[f"petri_{b}"])
        return rows
    monkeypatch.setattr(prun, "petri_metric_rows", late_correction)
    with pytest.raises(prun.DriverError) as e:
        prun.cmd_analyze(_args(in_root, root, worklist="registry"), spec)
    msg = str(e.value)
    b, rid2 = corrected["b"], corrected["rid"]
    old_inst, new_inst = str(_instrument(f"petri_{b}")["instrument_id"]), other_inst[f"petri_{b}"]
    assert len(filed_order) == 3 and filed_order[1] == rid2                 # the third run was still filed behind its own guard
    assert f"registry worklist filing for {U0_AUDIT!r}: 1 of 3 selected run(s) refused at their own filing boundary" in msg
    assert f"selected run identity changed between the binding and its own filing (run {rid2})" in msg
    assert f"{rid2}: " in msg and f"instrument_id: {old_inst[:48]!r} -> {new_inst[:48]!r}" in msg
    assert f"required_judge_id: {synth.JUDGE_ID!r} -> current instrument {new_inst} designates {synth.JUDGE_ID_2!r}" in msg
    for rid in in_ids.values():
        if rid != rid2:
            assert rid not in msg                                           # only the run refused at its boundary is named
    err = capsys.readouterr().err
    assert "this run's filing is refused -- its claim is released below, the siblings' filings proceed" in err
    assert "released 1 of 1 claim(s)" in err and err.count("[registry] claim lock held ") == len(BEHAVIORS3)
    assert (in_root / prun.ANALYSIS_JSON).exists()                          # the analysis was written before the filings, as before
    assert "selected_rows_recheck" in json.loads((in_root / prun.ANALYSIS_JSON).read_text())
    monkeypatch.undo()
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    with DB.reading(DB.db_path(root)) as conn:
        for rid in in_ids.values():
            js = conn.execute("SELECT judge_id, status, extra_json FROM judgment WHERE run_id = ? ORDER BY claimed_utc", (rid,)).fetchall()
            state = conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0]
            if rid == rid2:
                assert [(j[0], j[1]) for j in js] == [(synth.JUDGE_ID, "abandoned")] and state == "generated"    # released, nothing filed
            else:
                assert [(j[0], j[1]) for j in js] == [(synth.JUDGE_ID, "complete")] and state in ("judged", "ingested")   # the siblings' filings stand
                blk = json.loads(js[0][2])["filing_identity_recheck"]
                assert blk["stage"] == f"its own filing (run {rid})" and blk["fields"] == list(prun.BORROW_IN_VIEW_IDENTITY_FIELDS)
                assert blk["runs"][rid]["instrument_id"] == str(_instrument(f"petri_{prun._behaviour_of({'eval_id': conn.execute('SELECT eval_id FROM run WHERE run_id = ?', (rid,)).fetchone()[0], 'run_id': rid})}")["instrument_id"])
                assert blk["runs"][rid]["required_judge_id"] == synth.JUDGE_ID and blk["judge_selection"] == "instrument"
