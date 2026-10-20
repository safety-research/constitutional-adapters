"""math500-steer: per-LEG store pins, the SERVED-STEER stamp class selected by the leg's own
record, the external arm's fit-provenance status, the row's ARM. Synthetic stores only (``math500_synthetic_store``); the
no-switch (llama) class has no synthetic family on this tree (the builder's docstring) and is exercised on its selection
functions with hand-built records — its production reading is the real llama stores'."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import math500_synthetic_store as S
import pytest

from sisteer.capeval.serving import proxy as PX
from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]
import cluster_site as _cs  # noqa: E402

# The printers take each cluster's QoS and log directory from the site config: one synthetic two-cluster site for the module.
import tempfile  # noqa: E402
_SITE_CFG = _cs.write_site(Path(tempfile.mkdtemp(prefix="m5t-pins-site-")))


@pytest.fixture(autouse=True)
def _cluster_site(monkeypatch):
    monkeypatch.setenv("SISTEER_CONFIG", str(_SITE_CFG))



def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load("tinycap_math500_thinking", REPO / "scripts" / "tinycap_math500_thinking.py")
V = _load("tinycap_gsm8k_validators", REPO / "scripts" / "tinycap_gsm8k_validators.py")
CELLS = _load("m5t_cells_for_pins_test", REPO / "scripts" / "sdfbase" / "math500" / "cells.py")

B_FAST = 400
GEMMA, QWEN = "gemma4e4b", "qwen35_4b"
PRIMARY = S.MODE_PRIMARY
NONE, PROMPTED, OBJECT = S.CELL_BARE, S.CELL_PROMPTED, S.CELL_OBJECT
ALPHASTEER = "alphasteer"
T1 = S.FULL_SHA
T2 = "2" * 40
T3 = "3" * 40
STEER_PIN = "ab" * 32           # a SYNTHETIC 64-hex object digest for the external arm (the admission record's pin stands in the real matrix)


def _analyze(specs, rows, **kw):
    kw.setdefault("waves", ["published", "ext1", "steer"])
    return M.analyze(specs, S.SHA7, cells=rows, sampling_rows=S.SAMPLING_ROWS, render_facts=S.RENDER_FACTS, math500_ids=S.IDS,
                     n_resamples=B_FAST, read_eval_logs=False, analysis_sha=T1, repo_root=REPO, **kw)


def _small(model, *cells):
    rows = [r for r in S.design_rows() if r["model_key"] == model and r["sampling_mode"] == PRIMARY and r["cell"] in cells]
    assert len(rows) == len(cells)
    return rows


def _pins(rows, tree):
    return {M.row_leg_key(r): tree for r in rows}


# --------------------------------------------------------------------------- #
# per-leg store pins                                                           #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def two_tree_store(tmp_path_factory):
    """The published gemma three (tree T1, root pub/) and the extension gemma constitution row (tree T2, root ext1/) — two trees, two
    wave-isolated roots, no declared tree set."""
    root = tmp_path_factory.mktemp("two_tree")
    pub = _small(GEMMA, NONE, PROMPTED, OBJECT)
    ext = [r for r in S.ext_rows() if r["model_key"] == GEMMA and r["cell"] == S.CELL_PROMPTED_CONST]
    S.build_all(root / "pub", rows=pub)
    S.build_all(root / "ext1", rows=ext, tree_sha=T2)
    return root, pub, ext


def test_store_pins_admit_each_leg_by_its_own_tree_and_the_cross_tree_contrast_computes(two_tree_store):
    root, pub, ext = two_tree_store
    pins = {**_pins(pub, T1), **_pins(ext, T2)}
    rep = _analyze([str(root / "pub"), str(root / "ext1")], pub + ext, store_pins=pins)
    assert rep["gates"]["failed"] == [] and rep["unusable_legs"] == {}
    comp = next(c for c in rep["comparisons"] if c["name"] == f"{GEMMA}/{PRIMARY}")
    assert comp["estimands"] is not None and "prompted_constitution" in comp["leg_keys"]
    block = rep[M.STORE_PINS_KEY]
    assert block["pins"] == dict(sorted(pins.items())) and sorted(block["trees"]) == [T2, T1] or sorted(block["trees"]) == sorted({T1, T2})
    for r in pub:
        leg = rep["legs"][M.row_leg_key(r)]
        assert leg["producer"]["admitted_pin"] == T1 and leg["producer"]["admission"] == "per-leg pins" and leg["producer"]["matches_pin"]
    leg = rep["legs"][M.row_leg_key(ext[0])]
    assert leg["producer"]["admitted_pin"] == T2 and leg["producer"]["tree_sha"] == T2 and leg["producer"]["pin"] == T2
    assert rep["gen_roots"] and M.TREE_SET_KEY not in rep
    # a PINNED gen root may name any pinned tree of the doc (the run's --sha is the record label)
    rep2 = _analyze([f"{root / 'pub'}={S.SHA7}", f"{root / 'ext1'}={T2}"], pub + ext, store_pins=pins)
    assert rep2["gates"]["failed"] == [] and rep2["unusable_legs"] == {}


def test_store_pins_refuse_another_tree_a_missing_entry_and_a_disagreeing_pinned_root_by_name(two_tree_store):
    root, pub, ext = two_tree_store
    wrong = {**_pins(pub, T1), **_pins(ext, T3)}           # the extension leg pinned to a tree that never produced it
    rep = _analyze([str(root / "pub"), str(root / "ext1")], pub + ext, store_pins=wrong)
    k = M.row_leg_key(ext[0])
    assert "population" in rep["gates"]["failed"]
    # the wrong tree is refused BY NAME before any estimand: the root-pin check names the store's tree against the row's admitted pin
    # (the admitted pin IS the root pin under per-leg admission), the pins check names the doc — either wording names both trees
    assert k in rep["unusable_legs"] and T2[:8] in rep["unusable_legs"][k] and T3[:8] in rep["unusable_legs"][k]
    assert "does not carry the wave pin" in rep["unusable_legs"][k] or "is not the admitted pin" in rep["unusable_legs"][k]
    assert rep["legs"][k]["producer"]["admitted_pin"] == T3 and rep["legs"][k]["producer"]["matches_pin"] is False
    missing = _pins(pub, T1)                               # no entry for the extension leg at all
    rep = _analyze([str(root / "pub"), str(root / "ext1")], pub + ext, store_pins=missing)
    assert k in rep["unusable_legs"] and "no admitted pin" in rep["unusable_legs"][k]
    # a pinned gen root whose tree disagrees with the leg's admitted pin refuses by name (two spellings of the leg's tree)
    rep = _analyze([str(root / "pub"), f"{root / 'ext1'}={T3}"], pub + ext, store_pins={**_pins(pub, T1), **_pins(ext, T2), "x/y/z": T3})
    assert k in rep["unusable_legs"] and "gen root's pin" in rep["unusable_legs"][k]


def test_store_pins_loader_refuses_off_contract_inputs_and_pins_never_coexist_with_a_declared_tree_set(tmp_path, two_tree_store):
    root, pub, ext = two_tree_store
    assert M.load_store_pins(None) is None
    p = tmp_path / "pins.json"
    p.write_text(json.dumps({"_meta": {"x": 1}, f"{GEMMA}/none/{PRIMARY}": T1[:7]}), encoding="utf-8")
    assert M.load_store_pins(p) == {f"{GEMMA}/none/{PRIMARY}": T1[:7]}
    # two spellings of ONE tree merge to the longer; two different trees refuse by name
    assert M.load_store_pins([{f"{GEMMA}/none/{PRIMARY}": T1[:7]}, {f"{GEMMA}/none/{PRIMARY}": T1}]) == {f"{GEMMA}/none/{PRIMARY}": T1}
    with pytest.raises(M.AnalysisError, match="pinned twice"):
        M.load_store_pins([{f"{GEMMA}/none/{PRIMARY}": T1}, {f"{GEMMA}/none/{PRIMARY}": T2}])
    with pytest.raises(M.AnalysisError, match="not a tree sha"):
        M.load_store_pins({f"{GEMMA}/none/{PRIMARY}": "xyz"})
    with pytest.raises(M.AnalysisError, match="not a leg key"):
        M.load_store_pins({"gemma4e4b/none": T1})
    with pytest.raises(M.AnalysisError, match="not a JSON object"):
        (tmp_path / "bad.json").write_text("[]", encoding="utf-8"); M.load_store_pins(tmp_path / "bad.json")
    assert M.admitted_pin_problem("a/b/c", T1, {"a/b/c": T1[:7]}) == (T1[:7], None)
    assert "no admitted pin" in M.admitted_pin_problem("a/b/c", T1, {})[1]
    assert "is not the admitted pin" in M.admitted_pin_problem("a/b/c", T2, {"a/b/c": T1})[1]
    with pytest.raises(M.AnalysisError, match="two admissions"):
        _analyze([str(root / "pub"), str(root / "ext1")], pub + ext, store_pins=_pins(pub, T1),
                 tree_set={"also": [T2], "allow": ["*"], "why": "x"})


# --------------------------------------------------------------------------- #
# the served-steer stamp class on an external arm                              #
# --------------------------------------------------------------------------- #


def _steer_row():
    """The gemma AlphaSteer stub as a LAUNCH row with an admission pin — the launcher's shape (cells.py rows 33-40)."""
    stub = [r for r in S.steer_rows(100, models=[GEMMA], status=S.STATUS_LAUNCH) if r["cell"] == ALPHASTEER][0]
    template = S.matrix_row(GEMMA, OBJECT, PRIMARY)
    return {**stub, "object_sha256": STEER_PIN, "expected_stamps": list(template["expected_stamps"]),
            "twin_expected_stamps": list(template["twin_expected_stamps"]), "deferred_reason": None, "held_reason": None,
            "staged_id": ALPHASTEER, "view": "it"}


def _logprob_bites(cell, *, n_conclusive=5, n_min=3):
    rec = S.steering_bites_record(cell, n_differing=1)
    stamps = dict(zip(C.EXPECTED_STAMP_TRIPLE_FIELDS, V.served_steer_stamps(C.THINKING_ON)))
    rec["stamps_by_leg"] = {"steered": [dict(stamps) for _ in range(rec["n_probes"])], "scale0": [dict(stamps) for _ in range(rec["n_probes"])]}
    rec[C.OOS_EVIDENCE_BITES_LIVENESS_GATE] = C.LENS_LIVENESS_LOGPROB
    rec["object_kind"] = ALPHASTEER
    rec[C.OOS_EVIDENCE_BITES_TEXT_DIVERGENCE] = {"verified": True, "n_probes": rec["n_probes"], "n_differing": 1, "error": None,
                                                "role": "reported beside the logprob statistic"}
    rec[C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS] = {
        "applicable": True, "verified": True, "kind": ALPHASTEER, "twin": C.TWIN_CELL_ID, "n_probes": rec["n_probes"],
        "n_conclusive": n_conclusive, "min_conclusive_probes": n_min, "min_common_tokens": 2, "top_logprobs": 20, "max_tokens": 1,
        "diag": {"n_items": rec["n_probes"], "steered_divergence_median": 0.32, "noise_ceiling": 0.0, "min_ratio": 2.0, "abs_floor": 0.001, "bar": 0.001},
        "noise_floor": {"measured": 0.0, "how": "the scale-0 twin against itself", "min_ratio": 2.0, "abs_floor": 0.001, "bar": 0.001, "binding": "abs_floor"},
        "divergence": {"steered_vs_scale0_median": 0.32, "clears_bar": True},
        "not_a_dose_axis": C.LOGPROB_LIVENESS_NOT_A_DOSE_AXIS}
    return rec


def _served_steer_edits(row, *, n_conclusive=5, fit_kind=M.FIT_KIND_EXPECTED_EXTERNAL):
    """The edits that turn the builder's fitted (always / cell-stable / document_fit) leg into the served-steer route's record of an
    EXTERNAL object: the request-unique / never stamps, the logprob-ruled bites, (B′) attested `never`, the class-B mirror's fit_kind."""
    want = dict(zip(C.EXPECTED_STAMP_TRIPLE_FIELDS, V.served_steer_stamps(C.THINKING_ON)))

    def summary_edit(summary):
        for f, v in want.items():
            summary[f] = v

    def ledger_edit(rows):
        for r in rows:
            for f, v in want.items():
                if f in r:
                    r[f] = v
            if C.LEDGER_KEY_SALT_REGIME in r:
                r[C.LEDGER_KEY_SALT_REGIME] = want["salt_regime"]

    def manifest_edit(manifest):
        manifest[M.MANIFEST_KEY_TRANSFER_PROVENANCE]["fit_kind"] = fit_kind

    def evidence_edit(ev):
        ev[C.OOS_EVIDENCE_BITES_BLOCK] = _logprob_bites(ALPHASTEER, n_conclusive=n_conclusive)
        ev[M.EVIDENCE_LENS_POSITION] = C.lens_position_attested(
            S.standup_endpoint(GEMMA, ALPHASTEER, policy=PX.RESUMED_PREFILL_NEVER, pin=STEER_PIN), cell=ALPHASTEER,
            expected_sha256=STEER_PIN, model_key=GEMMA)

    return {"summary_edit": summary_edit, "ledger_edit": ledger_edit, "manifest_edit": manifest_edit, "evidence_edit": evidence_edit}


def _build_steer(root, row, **edits):
    pub = _small(GEMMA, NONE, PROMPTED, OBJECT)
    S.build_all(root / "pub", rows=pub)
    S.build_all(root / "steer", rows=[row], tree_sha=T2, overrides={(GEMMA, ALPHASTEER, PRIMARY): {**_served_steer_edits(row), **edits}})
    return pub


def test_served_steer_class_is_selected_by_the_legs_own_liveness_gate_and_passes_a2_a3_with_the_external_fit_kind(tmp_path):
    row = _steer_row()
    pub = _build_steer(tmp_path, row)
    pins = {**_pins(pub, T1), M.row_leg_key(row): T2}
    rep = _analyze([str(tmp_path / "pub"), str(tmp_path / "steer")], pub + [row], store_pins=pins)
    k = M.row_leg_key(row)
    assert k == f"{GEMMA}/{ALPHASTEER}/{PRIMARY}"           # an external arm is keyed on the main view whatever its bank view
    leg = rep["legs"][k]
    assert leg.get("refused") is None and leg.get("missing") is None, (leg.get("refused"), leg.get("missing"))
    assert leg["stamp_class"]["class"] == V.STAMP_CLASS_SERVED_STEER
    assert leg["stamp_class"]["stamps"] == list(V.served_steer_stamps(C.THINKING_ON)) == [PX.SALT_REGIME_REQUEST_UNIQUE, PX.RESUMED_PREFILL_NEVER, C.RENDER_REGIME_THINKING_ON]
    assert leg["stamps"]["passed"] and leg["certification"]["passed"], (leg["stamps"]["failures"], leg["certification"]["failures"])
    assert leg["certification"]["stamp_class"] == V.STAMP_CLASS_SERVED_STEER and leg["certification"]["cert_class"] == C.CERT_CLASS_SWITCH
    assert leg["identity"]["object_artifact_sha256"] == STEER_PIN and leg["producer"]["admitted_pin"] == T2
    for r in pub:      # the matrix-row class on every published leg, byte for byte
        assert rep["legs"][M.row_leg_key(r)]["stamp_class"]["class"] == V.STAMP_CLASS_ROW
    assert rep["gates"]["certification"]["status"] == M.GATE_PASS
    # the synthetic builder writes NO logprob-liveness probe records (the real served-steer stores carry them and pass A2 whole —
    # the production read): the ONLY identity lines on this leg may name that fixture gap, by probe kind, and nothing else
    ident = [f for f in rep["gates"]["identity"]["detail"].split("; ") if f.startswith(f"{k}:")] if rep["gates"]["identity"]["status"] != M.GATE_PASS else []
    assert ident and all("logprob_bites" in f and "probe shortfall" in f for f in ident), ident
    assert leg["request_evidence"]["failures"] and all("logprob_bites" in f for f in leg["request_evidence"]["failures"])
    assert rep["gates"]["failed"] == ["identity"] and rep["unusable_legs"] == {}
    comp = next(c for c in rep["comparisons"] if c["name"] == f"{GEMMA}/{PRIMARY}")
    assert comp["estimands"] is not None and ALPHASTEER in comp["leg_keys"]
    assert M.PRW.paper_arm_of(ALPHASTEER) == "alphasteer_rebake"      # the export's declared key for the cell


@pytest.mark.parametrize("case", ["inconclusive_logprob", "document_fit_on_external", "text_gate_stamps_mismatch", "class_selector_on_non_fitted"])
def test_served_steer_known_bad_inputs_still_refuse_by_name(tmp_path, case):
    row = _steer_row()
    if case == "inconclusive_logprob":
        pub = _build_steer(tmp_path, row, **{"evidence_edit": _served_steer_edits(row, n_conclusive=1)["evidence_edit"]})
    elif case == "document_fit_on_external":
        pub = _build_steer(tmp_path, row, **{"manifest_edit": _served_steer_edits(row, fit_kind=M.FIT_KIND_EXPECTED)["manifest_edit"]})
    elif case == "text_gate_stamps_mismatch":
        # the request-unique / never stamps WITHOUT the logprob selector: the row's class holds and the stamps refuse by field
        def evidence_edit(ev):
            ev[M.EVIDENCE_LENS_POSITION] = C.lens_position_attested(
                S.standup_endpoint(GEMMA, ALPHASTEER, policy=PX.RESUMED_PREFILL_NEVER, pin=STEER_PIN), cell=ALPHASTEER,
                expected_sha256=STEER_PIN, model_key=GEMMA)
        pub = _build_steer(tmp_path, row, evidence_edit=evidence_edit)
    else:
        pub = _small(GEMMA, NONE, PROMPTED, OBJECT)
        S.build_all(tmp_path / "pub", rows=pub)
        S.build_all(tmp_path / "steer", rows=[row], tree_sha=T2, overrides={(GEMMA, ALPHASTEER, PRIMARY): _served_steer_edits(row)})
        row = {**row, "fitted": False, "route_kind": "prompted"}       # the selector on a NON-fitted row is another producer's record
    pins = {**_pins(pub, T1), M.row_leg_key(row): T2}
    rep = _analyze([str(tmp_path / "pub"), str(tmp_path / "steer")], pub + [row], store_pins=pins)
    k = M.row_leg_key(row)
    leg = rep["legs"][k]
    if case == "inconclusive_logprob":
        assert rep["gates"]["certification"]["status"] == M.GATE_FAIL
        assert any("n_conclusive 1 < min_conclusive_probes 3" in f for f in leg["certification"]["failures"])
    elif case == "document_fit_on_external":
        assert leg["refused"] and "fit_kind 'document_fit' != 'verified'" in leg["refused"] and "external object arm" in leg["refused"]
    elif case == "text_gate_stamps_mismatch":
        assert leg["stamp_class"]["class"] == V.STAMP_CLASS_ROW
        assert rep["gates"]["identity"]["status"] == M.GATE_FAIL
        assert any("salt_regime" in f and "request-unique" in f for f in leg["stamps"]["failures"])
    else:
        assert leg["refused"] is not None and ("non-fitted" in leg["refused"] or "fitted=" in leg["refused"] or "fitted" in leg["refused"])


def test_a_lora_arm_carrying_the_mirror_word_refuses_by_name(tmp_path):
    rows = _small(GEMMA, NONE, PROMPTED, OBJECT)

    def manifest_edit(manifest):
        manifest[M.MANIFEST_KEY_TRANSFER_PROVENANCE]["fit_kind"] = M.FIT_KIND_EXPECTED_EXTERNAL

    S.build_all(tmp_path / "pub", rows=rows, overrides={(GEMMA, OBJECT, PRIMARY): {"manifest_edit": manifest_edit}})
    rep = _analyze(str(tmp_path / "pub"), rows, waves=None, store_pins=None)
    leg = rep["legs"][M.leg_key(GEMMA, OBJECT, PRIMARY)]
    assert leg["refused"] and "fit_kind 'verified' != 'document_fit'" in leg["refused"] and "LoRA arm" in leg["refused"]


# --------------------------------------------------------------------------- #
# the row's arm, the fit kind by arm, the primary row by arm                    #
# --------------------------------------------------------------------------- #


def test_the_row_arm_the_fit_kind_and_the_primary_row_are_keyed_by_the_arm_and_refuse_off_contract():
    assert M.row_arm({"cell": NONE}) == C.THINKING_ON == M.row_arm({"cell": NONE, "thinking_mode": "on"})
    assert M.row_arm({"thinking_mode": C.THINKING_NOT_AVAILABLE}) == C.THINKING_NOT_AVAILABLE
    with pytest.raises(M.AnalysisError, match="thinking_mode 'off' is not one of"):
        M.row_arm({"index": 3, "model_key": GEMMA, "cell": NONE, "thinking_mode": "off"})
    assert M.fit_kind_expected(OBJECT) == M.FIT_KIND_EXPECTED == "document_fit"
    assert M.fit_kind_expected(S.CELL_LORA_MIX) == M.fit_kind_expected(S.CELL_LORA_CTL2) == "document_fit"
    assert M.fit_kind_expected(ALPHASTEER) == M.fit_kind_expected("arditi") == M.FIT_KIND_EXPECTED_EXTERNAL == "verified"
    with pytest.raises(M.AnalysisError, match="not a registered arm"):
        M.fit_kind_expected("not_an_arm")
    assert M.PRIMARY_MODE_BY_ARM == {C.THINKING_ON: M.MODE_PRIMARY, C.THINKING_NOT_AVAILABLE: M.MODE_CARD}
    assert M.primary_mode_for({"thinking_mode": C.THINKING_NOT_AVAILABLE}) == "card" == CELLS.SAMPLING_CARD
    assert M.primary_mode_for({}) == M.MODE_PRIMARY == CELLS.SAMPLING_THINKING_CARD
    assert M.row_is_primary({"sampling_mode": "card", "thinking_mode": C.THINKING_NOT_AVAILABLE})
    assert not M.row_is_primary({"sampling_mode": "card"}) and M.row_is_primary({"sampling_mode": M.MODE_PRIMARY})
    # the launcher's stamp triples follow the arm; its render-regime mirror is the contracts' table on the two MATH-500 arms
    assert CELLS.RENDER_REGIME_BY_ARM == {arm: C.RENDER_REGIME_BY_ARM[arm] for arm in (C.THINKING_ON, C.THINKING_NOT_AVAILABLE)}
    assert CELLS.stamps_for(True, "on") == CELLS.STAMPS_FITTED and CELLS.stamps_for(False, "on") == CELLS.STAMPS_BASE
    assert CELLS.stamps_for(True, C.THINKING_NOT_AVAILABLE) == ("cell-stable", "always", C.RENDER_REGIME_UNSPECIFIED)
    assert CELLS.stamps_for(False, C.THINKING_NOT_AVAILABLE) == ("cell-stable", "not-consulted", C.RENDER_REGIME_UNSPECIFIED)
    assert CELLS.twin_stamps_for(C.THINKING_NOT_AVAILABLE) == ("cell-stable", "always", C.RENDER_REGIME_UNSPECIFIED)
    with pytest.raises(ValueError):
        CELLS.stamps_for(True, "off")
    for r in CELLS.M5T_CELLS:
        want_render = C.RENDER_REGIME_BY_ARM[r["thinking_mode"]]
        assert r["expected_stamps"][2] == want_render, r["index"]
        if r["twin_expected_stamps"] is not None:
            assert r["twin_expected_stamps"][2] == want_render
    llama = [r for r in CELLS.M5T_CELLS if r["model_key"] == "llama8b"]
    assert llama and all(r["thinking_mode"] == C.THINKING_NOT_AVAILABLE and r["sampling_mode"] == "card" for r in llama)


def test_the_no_switch_arm_selects_the_no_switch_class_and_refuses_a_mislabel_by_name():
    """The arm-selection surfaces with hand-built records (no synthetic no-switch family exists on this tree)."""
    fact_switch = dict(S.RENDER_FACTS["gemma4"])
    fact_none = {**fact_switch, "thinking_switch": False, "split_mode": C.THINK_SPLIT_NONE}
    row_on = {**S.matrix_row(GEMMA, OBJECT, PRIMARY)}
    row_ns = {**row_on, "thinking_mode": C.THINKING_NOT_AVAILABLE}
    # certification: the class from the arm x the fact; a mislabel refuses by name before any predicate is read
    rec = M.certification_check({}, {}, row=row_ns, fact=fact_switch, family="gemma4", object_pin=STEER_PIN, arm=C.THINKING_NOT_AVAILABLE)
    assert rec["cert_class"] == C.CERT_CLASS_SWITCH and any("no-switch class is for a switch-less family only" in f for f in rec["failures"])
    rec = M.certification_check({}, {}, row=row_on, fact=fact_none, family="llama", object_pin=STEER_PIN, arm=C.THINKING_ON)
    assert any("there is no thinking-ON class for it" in f for f in rec["failures"])
    rec = M.certification_check({}, {}, row=row_ns, fact=fact_none, family="llama", object_pin=STEER_PIN, arm=C.THINKING_NOT_AVAILABLE)
    assert rec["cert_class"] == C.CERT_CLASS_NO_SWITCH and rec["arm"] == C.THINKING_NOT_AVAILABLE
    assert any("no render_probe block on a no-switch cell" in f for f in rec["failures"])            # (A″) required, never assumed
    # the ledger's thinking flag at the arm
    fails_on = M.ledger_confirmation.__doc__ is not None
    assert fails_on
    # request evidence: under the no-switch arm a task call that CARRIED the switch refuses by name
    rows = [{"kind": M.REQUEST_EVIDENCE_KIND_TASK, "task": M.TASK, "vmodel": "v", "enable_thinking_present": True, "enable_thinking_value": True,
             "max_tokens": M.CAP, "sample": "x/0", "request_id": "r0", "body_sha256": "0" * 64, "version": 2}]
    assert M.validators_module()._probe_arm({"enable_thinking_present": False}, arm=C.THINKING_NOT_AVAILABLE) == C.THINKING_NOT_AVAILABLE
    assert M.validators_module()._probe_arm({"enable_thinking_present": True, "enable_thinking_value": True}, arm=C.THINKING_NOT_AVAILABLE) == C.THINKING_ON
    assert rows[0]["enable_thinking_present"] is True


def test_the_validators_module_is_loaded_once_and_derives_the_served_steer_triple_through_the_contract():
    """`validators_module()` is idempotent (ONE load per process) and the served-steer triple it exposes is DERIVED through the
    contract rule, never spelled. (Which copy of the reader the validators bound to is a property of the test session's import
    order — the full suite registers the reader under its script name from several modules — so no cross-copy identity is asserted.)"""
    v = M.validators_module()
    assert v is M.validators_module() and v.__name__ == "tinycap_gsm8k_validators"
    assert callable(v.leg_expected_stamps) and callable(v.served_steer_problems) and callable(v._probe_arm) and callable(v._no_switch_manifest_problem)
    assert v.M.__name__ == "tinycap_math500_thinking" and v.M.TASK == M.TASK and v.M.FIT_KIND_EXPECTED_EXTERNAL == M.FIT_KIND_EXPECTED_EXTERNAL
    # the served-steer triple is DERIVED through the contract rule, never spelled here
    assert tuple(V.served_steer_stamps(C.THINKING_ON)) == tuple(C.expected_stamp_regimes(lens=True, arm=C.THINKING_ON, policy=PX.RESUMED_PREFILL_NEVER))
    assert tuple(V.served_steer_stamps(C.THINKING_NOT_AVAILABLE))[2] == C.RENDER_REGIME_UNSPECIFIED


# --------------------------------------------------------------------------- #
# the lens serving route BY CONSTRUCTION (both launchers) and the #138 guard     #
# --------------------------------------------------------------------------- #

import os
import re
import shutil
import subprocess
import tempfile

G8K = _load("g8k_cells_for_pins_test", REPO / "scripts" / "sdfbase" / "gsm8k" / "g8k_cells.py")
SP = _load("steer_store_pin_for_pins_test", REPO / "scripts" / "sdfbase" / "full22" / "steer_store_pin.py")
SHA_ANY = "e" * 40
ARDITI = "arditi"
#: the all-positions Arditi cell (arditi-allpos-cap): object kind vector_all, natively servable, NEVER run — no store anywhere
ARDITI_ALLPOS, VECTOR_ALL = "arditi_allpos", "vector_all"
#: the two engines' serving identities as the drivers file them (pinned to manifest.serving_identity below)
LENS_IDENTITY, NATIVE_IDENTITY = "proxy_vllm+lens+interleaved", "proxy_vllm+lora+interleaved"
#: the three all-positions legs (allpos-store-pins, 2026-09-17): admitted at their launch tree in store_pins_arditi_allpos.json (GSM8K, both
#: rungs) and store_pins_default.json (MATH-500, keyed on the `it` panel view); their engine entries are the stamp script's to derive from the
#: PULLED stores — the ONLY admitted steer legs that may lack one, until it runs (the strict-xfail gate below announces the stamp)
ALLPOS_LEGS_ENGINE_PENDING = frozenset({"qwen35_4b/it/arditi_allpos@8192", "qwen35_4b/it/arditi_allpos@16384", "qwen35_4b/arditi_allpos/thinking_card@it"})
#: the engine EVERY all-positions LAUNCH RECORD names (standup_jobs.jsonl `engine` of jobs 544615 / 544730 / 544616): the independent check on
#: what the stamp derives — a stamp that reads `lens` for one of these legs has read the wrong store
ALLPOS_LAUNCH_ENGINE = "lora_native"
M5T_ROOT_REL, G8K_ROOT_REL = "exp/math500_thinking_on_full22", "exp/gsm8k_paired200_paper_objects"


def _steer_store(root: Path, engine: str | None, *, top: bool = True, cell: bool = True, cell_id: str = ARDITI) -> Path:
    """A SYNTHETIC steer store at ``root`` in the shape BOTH engines' drivers file (never a real store's bytes): the top-level
    ``evidence.json`` ``serving_identity`` and one ``cells/<key>/manifest.json`` ``route`` (+ the cell's evidence.json). The lens
    record carries NO ``serving_engine`` stamp (the historical record); the native one stamps ``lora_native`` beside its ``+lora+``
    identity. ``engine None`` = a root holding only ladder bookkeeping — a job that never served, which is NOT a store on record.
    ``cell_id`` names the steer cell the store is of (its cell key and virtual model); the reader never reads it, the pin follows
    the engine markers alone."""
    root.mkdir(parents=True, exist_ok=True)
    if engine is None:
        (root / "ladder_state.json").write_text("{}")
        return root
    ident = LENS_IDENTITY if engine == "lens" else NATIVE_IDENTITY
    route = {"virtual_model": f"petri__{cell_id}", "serving_identity": ident, "lens_served_positions": ["all"], "reduction_mode": "stacked"}
    if engine == "native":
        route["serving_engine"] = "lora_native"
    if cell:
        c = root / "cells" / f"{cell_id}__controls__t0__a0__11111111__22222222"
        c.mkdir(parents=True, exist_ok=True)
        (c / "manifest.json").write_text(json.dumps({"cell_id": cell_id, "route": route}))
        (c / "evidence.json").write_text(json.dumps({"serving_identity": ident}))
    if top:
        (root / "evidence.json").write_text(json.dumps({"serving_identity": ident, "engine": "vllm"}))
    return root


def _pins_record(path: Path, engines: dict[str, str] | None = None, *, legs: dict[str, str] | None = None) -> Path:
    """A SYNTHETIC store-pins record (the analysis' shape: `_meta` + leg -> tree entries) with an `_meta.serving_engine_by_leg` block
    naming `engines` (leg key -> lens | lora_native) the way stamp_store_pins_engines.py writes it; `engines` None = no block at all
    (a record that knows no engine). Placeholder trees; never a real record's bytes."""
    doc = {"_meta": {"schema": "test_store_pins/2", "note": "synthetic"}, **(legs or {})}
    if engines is not None:
        doc["_meta"]["serving_engine_by_leg"] = {k: {"engine": e, "producer_tree_sha": "1" * 40, "pulled_store": "pull_x/" + k, "readings": [], "stamped": "2026-01-01"}
                                                 for k, e in engines.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1))
    return path


def _m5t_row(key, cell):
    rows = [r for r in CELLS.M5T_CELLS if r["model_key"] == key and r["cell"] == cell and r["wave"] == "steer"]
    assert len(rows) == 1, (key, cell)
    return rows[0]


def _g8k_row(key, cell, matrix=None):
    rows = [r for r in G8K.matrix_rows(G8K.resolve_matrix(matrix)) if r["model_key"] == key and r["cell"] == cell and r.get("object_source") == G8K.OBJECT_SOURCE_EXTERNAL]
    assert len(rows) == 1, (key, cell, matrix)
    return rows[0]


def test_the_pin_follows_the_store_over_synthetic_stores_in_both_launchers(tmp_path):
    """The pin-follows-the-store rule, rule by rule on synthetic rows + stores, the PINS source silenced by a synthetic
    record that knows no engine (the DISK source alone): (a) an AlphaSteer row pins the lens BY KIND, with or without a wave root, and never
    reads a store (a native store planted at its root changes nothing); (b) an Arditi row with a LENS store on record pins the lens
    `store:lens(disk)`; (c) with NO store (an absent root, or a root holding only ladder bookkeeping) it is unpinned `unpinned:no-store`;
    (d) with a NATIVE store it is unpinned `store:native(disk)`; (e) a store carrying both engines' stamps REFUSES by name; a reading with
    no wave root is `unpinned:store-unread` (never "no store"); every non-external row is `unpinned:not-external`. GSM8K reads the CAP
    root AND the rung sibling root together. row_sh emits no serving-path literal any more."""
    for mod in (CELLS, G8K):
        assert mod.SERVING_PATH_LENS == "lens" == SP.SERVING_PATH_LENS
    wave = tmp_path / "wave"
    blind = [_pins_record(tmp_path / "pins_blind.json")]                       # a record with no engine block: the pins know nothing
    for mod, alpha, arditi in ((CELLS, _m5t_row(GEMMA, ALPHASTEER), _m5t_row(GEMMA, ARDITI)),
                               (G8K, _g8k_row(GEMMA, ALPHASTEER), _g8k_row(GEMMA, ARDITI)),
                               (G8K, _g8k_row("llama8b", ALPHASTEER, "llama"), _g8k_row("llama8b", ARDITI, "llama"))):
        P = dict(store_pins=blind)
        # (a) by kind, store or no store
        assert mod.serving_path_pin(alpha, **P) == ("lens", "kind:alphasteer") == mod.serving_path_pin(alpha, wave_root=wave, **P) == mod.serving_path_pin(alpha)
        _steer_store(wave / alpha["out_root"], "native")
        assert mod.serving_path_pin(alpha, wave_root=wave, **P) == ("lens", "kind:alphasteer") and mod.serving_path_for(alpha) == "lens"
        # (c) no store: absent root, then a root with only ladder bookkeeping
        assert mod.serving_path_pin(arditi, wave_root=wave, **P) == (None, "unpinned:no-store") and mod.serving_path_for(arditi, wave_root=wave, **P) is None
        _steer_store(wave / arditi["out_root"], None)
        assert mod.serving_path_pin(arditi, wave_root=wave, **P) == (None, "unpinned:no-store")
        # no wave root: unread, never "no store"; the path alone reads None (the plan's kind-only reading)
        assert mod.serving_path_pin(arditi, **P) == (None, "unpinned:store-unread") and mod.serving_path_for(arditi, **P) is None
        # (b) a lens store
        _steer_store(wave / arditi["out_root"], "lens")
        assert mod.serving_path_pin(arditi, wave_root=wave, **P) == ("lens", "store:lens(disk)") and mod.serving_path_for(arditi, wave_root=wave, **P) == "lens"
        # (d) a native store
        shutil.rmtree(wave / arditi["out_root"]); _steer_store(wave / arditi["out_root"], "native")
        assert mod.serving_path_pin(arditi, wave_root=wave, **P) == (None, "store:native(disk)")
        # (e) a mixed store: the top-level evidence says lens, the cell route says native
        (wave / arditi["out_root"] / "evidence.json").write_text(json.dumps({"serving_identity": LENS_IDENTITY}))
        with pytest.raises(SystemExit, match="BOTH engines' stamps") as exc:
            mod.serving_path_pin(arditi, wave_root=wave, **P)
        assert "evidence.json:serving_identity='proxy_vllm+lens+interleaved' -> lens" in str(exc.value) and "-> native" in str(exc.value) and arditi["out_root"] in str(exc.value)
        shutil.rmtree(wave / arditi["out_root"])
        # every other row: the pin has no rule for it
        others = [r for r in (CELLS.M5T_CELLS if mod is CELLS else G8K.matrix_rows()) if r.get("object_source") != mod.OBJECT_SOURCE_EXTERNAL]
        assert others and all(mod.serving_path_pin(r, wave_root=wave) == (None, "unpinned:not-external") for r in others)
        assert mod.serving_path_pin({"object_source": "itdoc", "cell": "x"}) == (None, "unpinned:not-external")
        # a serving-path literal no longer rides the row: the store, not the row, is the source
        assert "SERVING_PATH" not in mod.row_sh(arditi) and "SERVING_PATH" not in mod.row_sh(alpha)
    # GSM8K: the cap root and the rung sibling root are ONE store on record — either alone pins; a disagreement refuses
    g = _g8k_row(GEMMA, ARDITI); P = dict(store_pins=blind)
    assert G8K.store_roots_rel(g) == [g["out_root"], g["rung_out_root"]] and g["rung_out_root"].endswith("__rung16384")
    _steer_store(wave / g["rung_out_root"], "lens")
    assert G8K.serving_path_pin(g, wave_root=wave, **P) == ("lens", "store:lens(disk)")          # rung store alone
    _steer_store(wave / g["out_root"], "lens")
    assert G8K.serving_path_pin(g, wave_root=wave, **P) == ("lens", "store:lens(disk)")          # both, agreeing
    shutil.rmtree(wave / g["rung_out_root"]); _steer_store(wave / g["rung_out_root"], "native")
    with pytest.raises(SystemExit, match="BOTH engines' stamps"):
        G8K.serving_path_pin(g, wave_root=wave, **P)
    assert CELLS.store_roots_rel(_m5t_row(GEMMA, ARDITI)) == [_m5t_row(GEMMA, ARDITI)["out_root"]]
    # a steer cell the rule does not know refuses by name (external_row_problem refuses it earlier in production)
    with pytest.raises(SystemExit, match="not a steer cell"):
        CELLS.serving_path_pin({"object_source": "external", "cell": "psr_thing", "out_root": "x"}, wave_root=wave, **P)


def test_the_pins_source_makes_a_store_on_record_travel_with_the_launch_tree(tmp_path):
    """A store on record is the launching cluster's DISK **and** the repo's store-pins
    record's engine for the row's admitted leg(s) — a row whose canonical store lives on the OTHER cluster still follows it. Pins-only
    lens (no disk store) -> lens `store:lens(pins)`; disk + pins agreeing -> `store:lens(disk+pins)`; disk native vs pins lens ->
    REFUSED by name naming both sources; pins native alone -> unpinned `store:native(pins)`, and a lens export over it is what the
    shell refuses; a row the pins do not know and the disk does not hold -> `unpinned:no-store`; the pins speak with NO wave root
    too (the grading host's plan: a record-pinned Arditi row reads lens there). A malformed engine entry refuses by name; a record that is
    not a file refuses by name. Both launchers, the GSM8K row's TWO leg keys (either cap's entry pins the row)."""
    wave = tmp_path / "wave"
    for mod, arditi in ((CELLS, _m5t_row(GEMMA, ARDITI)), (G8K, _g8k_row(GEMMA, ARDITI)), (G8K, _g8k_row("llama8b", ARDITI, "llama"))):
        keys = mod.leg_keys(arditi)
        lens_pins = [_pins_record(tmp_path / "pins_lens.json", {keys[0]: "lens"})]
        native_pins = [_pins_record(tmp_path / "pins_native.json", {keys[0]: "lora_native"})]
        other_pins = [_pins_record(tmp_path / "pins_other.json", {"other/main/arditi@8192": "lens", "other/arditi/thinking_card": "lens"})]
        # pins-only lens: the disk holds nothing here (the store is on the other cluster), the record knows the leg
        assert mod.serving_path_pin(arditi, wave_root=wave, store_pins=lens_pins) == ("lens", "store:lens(pins)")
        assert mod.serving_path_pin(arditi, store_pins=lens_pins) == ("lens", "store:lens(pins)")                   # no wave root: the pins still speak
        assert mod.serving_path_for(arditi, store_pins=lens_pins) == "lens"
        # the unknown row: neither source knows it
        assert mod.serving_path_pin(arditi, wave_root=wave, store_pins=other_pins) == (None, "unpinned:no-store")
        assert mod.serving_path_pin(arditi, store_pins=other_pins) == (None, "unpinned:store-unread")
        # disk + pins agreeing
        _steer_store(wave / arditi["out_root"], "lens")
        assert mod.serving_path_pin(arditi, wave_root=wave, store_pins=lens_pins) == ("lens", "store:lens(disk+pins)")
        assert mod.serving_path_pin(arditi, wave_root=wave, store_pins=other_pins) == ("lens", "store:lens(disk)")
        # disk native vs pins lens: refused, both sources named
        shutil.rmtree(wave / arditi["out_root"]); _steer_store(wave / arditi["out_root"], "native")
        with pytest.raises(SystemExit, match="BOTH engines' stamps") as exc:
            mod.serving_path_pin(arditi, wave_root=wave, store_pins=lens_pins)
        assert "disk: ['native']" in str(exc.value) and "pins: ['lens']" in str(exc.value) and "pins_lens.json" in str(exc.value) and "serving_engine_by_leg" in str(exc.value)
        assert mod.serving_path_pin(arditi, wave_root=wave, store_pins=native_pins) == (None, "store:native(disk+pins)")
        shutil.rmtree(wave / arditi["out_root"])
        assert mod.serving_path_pin(arditi, wave_root=wave, store_pins=native_pins) == (None, "store:native(pins)")
        # disk lens vs pins native: refused the other way round too
        _steer_store(wave / arditi["out_root"], "lens")
        with pytest.raises(SystemExit, match="BOTH engines' stamps"):
            mod.serving_path_pin(arditi, wave_root=wave, store_pins=native_pins)
        shutil.rmtree(wave / arditi["out_root"])
        # two records read together (the GSM8K launcher reads three): the second may name the row
        assert mod.serving_path_pin(arditi, wave_root=wave, store_pins=other_pins + lens_pins) == ("lens", "store:lens(pins)")
        # malformed: an entry without an engine, an unknown engine, a record that is not a file
        bad = _pins_record(tmp_path / "pins_bad.json"); doc = json.loads(bad.read_text()); doc["_meta"]["serving_engine_by_leg"] = {keys[0]: {"producer_tree_sha": "1" * 40}}; bad.write_text(json.dumps(doc))
        with pytest.raises(SystemExit, match="carries no 'engine'"):
            mod.serving_path_pin(arditi, wave_root=wave, store_pins=[bad])
        with pytest.raises(SystemExit, match="is not one of"):
            mod.serving_path_pin(arditi, wave_root=wave, store_pins=[_pins_record(tmp_path / "pins_typo.json", {keys[0]: "typo"})])
        with pytest.raises(SystemExit, match="is not a file"):
            mod.serving_path_pin(arditi, wave_root=wave, store_pins=[tmp_path / "absent.json"])
        # an AlphaSteer row never consults the pins (a native pins entry for it changes nothing)
        alpha = _m5t_row(GEMMA, ALPHASTEER) if mod is CELLS else _g8k_row(arditi["model_key"], ALPHASTEER, None if arditi["model_key"] == GEMMA else "llama")
        assert mod.serving_path_pin(alpha, wave_root=wave, store_pins=[_pins_record(tmp_path / "pins_a.json", {mod.leg_keys(alpha)[0]: "lora_native"})]) == ("lens", "kind:alphasteer")
    # the GSM8K row's rung-cap leg alone pins the row too (any cap)
    g = _g8k_row(GEMMA, ARDITI)
    assert G8K.leg_keys(g) == [f"{GEMMA}/main/arditi@8192", f"{GEMMA}/main/arditi@16384"]
    assert G8K.serving_path_pin(g, wave_root=wave, store_pins=[_pins_record(tmp_path / "pins_rung.json", {G8K.leg_keys(g)[1]: "lens"})]) == ("lens", "store:lens(pins)")


def _pulled_store(root: Path, tree_sha: str) -> Path:
    """A SYNTHETIC pulled lens store of one admitted leg, as the stamping script reads it: the cell manifest's `producer.tree_sha` names
    the leg's pinned tree, the route and the evidence carry the lens identity."""
    c = root / "cells" / "arditi__controls__t0__a0__11111111__22222222"
    c.mkdir(parents=True, exist_ok=True)
    (c / "manifest.json").write_text(json.dumps({"cell_id": ARDITI, "route": {"virtual_model": "petri__arditi", "serving_identity": LENS_IDENTITY},
                                                 "producer": {"driver": "served_capability_run", "tree_sha": tree_sha}}))
    (c / "evidence.json").write_text(json.dumps({"serving_identity": LENS_IDENTITY}))
    (root / "evidence.json").write_text(json.dumps({"serving_identity": LENS_IDENTITY, "engine": "vllm"}))
    return root


def test_an_admitted_leg_whose_engine_the_record_does_not_carry_refuses_by_name_via_the_stampers_own_allow_missing_output(tmp_path):
    """A leg ADMITTED in a store-pins record (a leg -> tree entry: a KNOWN store) but with no
    `serving_engine_by_leg` entry is reachable through `stamp_store_pins_engines.py --allow-missing`, which keeps the admission and drops
    the engine. Read as `unpinned:no-store`, with the disk absent, it would relaunch a known store natively. It REFUSES by leg name
    (`refused:record-engine-unknown`) — tested through the stamper's ACTUAL output, not a hand-built record: the real MATH-500 record
    re-stamped over a synthetic pull that holds every steer leg's store at its pinned tree except llama8b/arditi/card (allowed missing) ->
    row 40 refuses (the CLI too), the stamped rows still read `store:lens(pins)`, a lens disk store beside the unknown record does not
    rescue it; the GSM8K llama record likewise with the row's `@8192` leg missing while its `@16384` leg is stamped — the unknown leg
    refuses, never the other leg's engine. A leg NOT admitted anywhere stays `unpinned:no-store`."""
    stamper = REPO / "scripts" / "sdfbase" / "full22" / "stamp_store_pins_engines.py"
    empty = tmp_path / "empty-wave"
    # MATH-500
    rec = tmp_path / "m5t_pins.json"; shutil.copy(CELLS.STORE_PINS_PATHS[0], rec)
    doc = json.loads(rec.read_text()); pull = tmp_path / "pull_m5t"
    ext = [r for r in CELLS.M5T_CELLS if r.get("object_source") == CELLS.OBJECT_SOURCE_EXTERNAL]
    # the record ADMITS a leg by carrying its leg -> tree entry; a row it does not know is NOT admitted. Every external row is admitted now:
    # the all-positions row 53 joined the record at its launch tree (allpos-store-pins, 2026-09-17), so no external row is left unadmitted
    admitted = [r for r in ext if CELLS.leg_keys(r)[0] in doc]
    never_run = [r for r in ext if CELLS.leg_keys(r)[0] not in doc]
    assert never_run == [] and [r["cell"] for r in admitted].count(ARDITI_ALLPOS) == 1, [(r["job_id"], CELLS.leg_keys(r)) for r in never_run]
    allpos = next(r for r in admitted if r["cell"] == ARDITI_ALLPOS)
    missing = CELLS.leg_keys(_m5t_row("llama8b", ARDITI))[0]
    assert missing == "llama8b/arditi/card" and missing in doc
    for r in admitted:
        key = CELLS.leg_keys(r)[0]
        if key != missing:
            _pulled_store(pull / r["out_root"], doc[key])
    out = subprocess.run([sys.executable, str(stamper), "--launcher", "math500", "--pins", str(rec), "--pulled-root", str(pull), "--allow-missing", missing],
                         capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0 and "allowed missing: no engine recorded" in out.stdout and f"wrote {rec}" in out.stdout, out.stdout + out.stderr
    new = json.loads(rec.read_text())
    assert missing in new and missing not in new["_meta"]["serving_engine_by_leg"] and len(new["_meta"]["serving_engine_by_leg"]) == len(admitted) - 1
    assert set(new["_meta"]["serving_engine_by_leg"]) == {CELLS.leg_keys(r)[0] for r in admitted} - {missing}    # exactly the legs it was given stores for; it invents nothing
    with pytest.raises(SystemExit, match="refused:record-engine-unknown") as exc:
        CELLS.serving_path_pin(_m5t_row("llama8b", ARDITI), wave_root=empty, store_pins=[rec])
    assert missing in str(exc.value) and "ADMITTED" in str(exc.value) and "never relaunched on a guessed engine" in str(exc.value)
    with pytest.raises(SystemExit, match="refused:record-engine-unknown"):
        CELLS.serving_path_pin(_m5t_row("llama8b", ARDITI), store_pins=[rec])                                     # no wave root: the same
    assert CELLS.serving_path_pin(_m5t_row(GEMMA, ARDITI), wave_root=empty, store_pins=[rec]) == ("lens", "store:lens(pins)")
    # THE ALL-POSITIONS ROW is admitted like any other: over the re-stamped synthetic record (its synthetic pull above was a LENS store) it
    # reads `store:lens(pins)` — the record's engine, not the kind's — with or without a wave root; over a record that does not know it (the
    # GSM8K llama record) it is `unpinned:no-store` with the disk empty and `store-unread` with no wave root. Nothing is invented for it.
    assert SP.steer_cell_kind(allpos["cell"]) == VECTOR_ALL and allpos["out_root"] and not (empty / allpos["out_root"]).exists()
    assert CELLS.serving_path_pin(allpos, wave_root=empty, store_pins=[rec]) == ("lens", "store:lens(pins)") == CELLS.serving_path_pin(allpos, store_pins=[rec])
    assert CELLS.serving_path_pin(allpos, wave_root=empty, store_pins=[G8K.STORE_PINS_PATHS[1]]) == (None, "unpinned:no-store")
    assert CELLS.serving_path_for(allpos, wave_root=empty, store_pins=[G8K.STORE_PINS_PATHS[1]]) is None
    assert CELLS.serving_path_pin(allpos, store_pins=[G8K.STORE_PINS_PATHS[1]]) == (None, "unpinned:store-unread")
    _steer_store(empty / _m5t_row("llama8b", ARDITI)["out_root"], "lens")                                        # a lens disk store does not rescue it
    with pytest.raises(SystemExit, match="refused:record-engine-unknown"):
        CELLS.serving_path_pin(_m5t_row("llama8b", ARDITI), wave_root=empty, store_pins=[rec])
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "sdfbase" / "math500" / "cells.py"), "row", "--index", str(_m5t_row("llama8b", ARDITI)["index"]),
                        "--serving-path", "--wave-root", str(tmp_path / "empty2"), "--store-pins", str(rec)], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 1 and "refused:record-engine-unknown" in r.stderr and r.stdout == ""
    # a stamping run over a pull that lacks a leg WITHOUT --allow-missing refuses (the record is never left half-stamped silently)
    rec2 = tmp_path / "m5t_pins2.json"; shutil.copy(CELLS.STORE_PINS_PATHS[0], rec2)
    out = subprocess.run([sys.executable, str(stamper), "--launcher", "math500", "--pins", str(rec2), "--pulled-root", str(pull)], capture_output=True, text=True, cwd=REPO)
    assert out.returncode != 0 and "no pulled store" in out.stdout + out.stderr and json.loads(rec2.read_text()) == json.loads(CELLS.STORE_PINS_PATHS[0].read_text())
    # GSM8K (the llama record): the @8192 leg of row 13 allowed missing, its @16384 leg stamped
    grec = tmp_path / "g8k_llama_pins.json"; shutil.copy(G8K.STORE_PINS_PATHS[1], grec)
    gdoc = json.loads(grec.read_text()); gpull = tmp_path / "pull_g8k"
    row13 = _g8k_row("llama8b", ARDITI, "llama"); gmissing = G8K.leg_keys(row13)[0]
    assert gmissing == "llama8b/main/arditi@8192" and gmissing in gdoc and G8K.leg_keys(row13)[1] in gdoc
    for r in G8K.matrix_rows(G8K.resolve_matrix("llama")):
        if r.get("object_source") != G8K.OBJECT_SOURCE_EXTERNAL:
            continue
        for key, rel in zip(G8K.leg_keys(r), G8K.store_roots_rel(r)):
            if key in gdoc and key != gmissing:
                _pulled_store(gpull / rel, gdoc[key])
    out = subprocess.run([sys.executable, str(stamper), "--launcher", "gsm8k", "--matrix", "llama", "--pins", str(grec), "--pulled-root", str(gpull), "--allow-missing", gmissing],
                         capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0 and "allowed missing" in out.stdout, out.stdout + out.stderr
    gnew = json.loads(grec.read_text())
    assert gmissing not in gnew["_meta"]["serving_engine_by_leg"] and gnew["_meta"]["serving_engine_by_leg"][G8K.leg_keys(row13)[1]]["engine"] == "lens"
    with pytest.raises(SystemExit, match="refused:record-engine-unknown") as exc:
        G8K.serving_path_pin(row13, wave_root=empty, store_pins=[grec])
    assert gmissing in str(exc.value)
    assert G8K.serving_path_pin(_g8k_row("llama8b", ALPHASTEER, "llama"), wave_root=empty, store_pins=[grec]) == ("lens", "kind:alphasteer")
    # a leg admitted nowhere is still simply unknown
    assert G8K.serving_path_pin(_g8k_row(GEMMA, ARDITI), wave_root=empty, store_pins=[grec]) == (None, "unpinned:no-store")
    # ...and the GSM8K arditi_allpos row (its own matrix, two legs) is admitted by EXACTLY ONE record on this tree — its own,
    # store_pins_arditi_allpos.json, registered last — and by no other, the re-stamped llama record included: over the llama record alone it
    # reads `unpinned:no-store` (that record does not know it) and `store-unread` with no wave root — never a pin invented from another
    # matrix's record
    g_new = [r for r in G8K.matrix_rows(G8K.resolve_matrix("arditi_allpos")) if r["cell"] == ARDITI_ALLPOS]
    assert len(g_new) == 1 and g_new[0].get("object_source") == G8K.OBJECT_SOURCE_EXTERNAL and len(G8K.leg_keys(g_new[0])) == 2, g_new
    g_new = g_new[0]
    holders = {k: [p.name for p in list(G8K.STORE_PINS_PATHS) + [grec] if k in json.loads(p.read_text())] for k in G8K.leg_keys(g_new)}
    assert holders == {k: ["store_pins_arditi_allpos.json"] for k in G8K.leg_keys(g_new)}, holders
    assert G8K.serving_path_pin(g_new, wave_root=empty, store_pins=[grec]) == (None, "unpinned:no-store")
    assert G8K.serving_path_for(g_new, wave_root=empty, store_pins=[grec]) is None and G8K.serving_path_pin(g_new, store_pins=[grec]) == (None, "unpinned:store-unread")


def _readings_engines(readings) -> set:
    """The engines a stamped entry's readings name — an identity's path segment, or a `serving_engine` stamp — through the pin's own readers."""
    return {SP.stamp_engine(value, where=field) if field.endswith(SP.SERVED_ENGINE_FIELD) else SP.identity_engine(value, where=field)
            for _f, field, value in readings}


def test_the_repo_store_pins_records_carry_every_steer_legs_engine_derived_from_its_pulled_store():
    """The stamped records on THIS tree (stamp_store_pins_engines.py over the grading host pulls): every admitted steer leg of every external row
    of the four GSM8K matrices and the MATH-500 matrix has an engine entry — the ONE exception, pinned by name, being the three all-positions
    legs while their engine stamp is pending (allpos-store-pins, 2026-09-17: admitted at their launch tree, refusing by name meanwhile; the
    strict-xfail gate below announces the stamp); its engine is the one the leg's LAUNCH RECORD names (lens for every store on record before
    the all-positions wave; lora_native for the all-positions legs, whose standup records say so — the independent check on the stamp's
    derivation); its `producer_tree_sha` is the leg's own pin (the pulled store was the admitted one); its readings all name that engine;
    its provenance path is relative (a pull's name, never a workstation path); the schema is /2 with the source note; the leg -> tree entries
    still load through the analysis' reader (which skips `_meta`) — the admission is untouched. And the REAL `arditi` rows read
    `store:lens(pins)` with no disk store, on both launchers: the relayed prediction that llama8b/arditi goes native is wrong today."""
    for mod, pins_paths, rows in ((G8K, G8K.STORE_PINS_PATHS, [r for name in ("default", "llama", "arditi_allpos") for r in G8K.matrix_rows(G8K.resolve_matrix(name))]),
                                  (CELLS, CELLS.STORE_PINS_PATHS, list(CELLS.M5T_CELLS))):
        ext = [r for r in rows if r.get("object_source") == mod.OBJECT_SOURCE_EXTERNAL]
        assert ext
        docs = {p: json.loads(p.read_text()) for p in pins_paths}
        stamped = {p: d["_meta"].get("serving_engine_by_leg") for p, d in docs.items()}
        for r in ext:
            for key in mod.leg_keys(r):
                holders = [p for p, d in docs.items() if key in d]
                assert len(holders) <= 1, (key, holders)
                if not holders:
                    continue                                        # a leg the pins do not admit (no rung store): no engine either
                p = holders[0]; entry = (stamped[p] or {}).get(key)
                if entry is None:
                    assert key in ALLPOS_LEGS_ENGINE_PENDING, (key, str(p), "admitted without an engine entry — stamp it or set the leg aside")
                    continue                                        # admitted, stamp pending: the launchers refuse the leg by name meanwhile
                want = ALLPOS_LAUNCH_ENGINE if key in ALLPOS_LEGS_ENGINE_PENDING else "lens"
                assert entry["engine"] == want and entry["producer_tree_sha"] == docs[p][key], (key, entry)
                assert not entry["pulled_store"].startswith("/") and entry["pulled_store"].endswith(mod.store_roots_rel(r)[mod.leg_keys(r).index(key)]), entry["pulled_store"]
                assert entry["readings"] and _readings_engines(entry["readings"]) == {SP.ENGINE_LENS if want == "lens" else SP.ENGINE_NATIVE}, (key, entry["readings"])
                assert docs[p]["_meta"]["schema"].endswith("/2") and "stamp_store_pins_engines.py" in docs[p]["_meta"]["serving_engine_source"]
        for p, d in docs.items():
            block = stamped[p] or {}
            assert set(block) <= {k for k in d if k != "_meta"}, (p, set(block) - set(d))
            loaded = M.load_store_pins(p)                          # the analysis' reader: _meta skipped, every leg -> tree entry intact
            assert loaded == {k: v for k, v in d.items() if k != "_meta"}
        arditi_rows = [r for r in ext if r["cell"] == ARDITI]
        assert arditi_rows and all(mod.serving_path_pin(r, wave_root="/nonexistent/wave-root") == ("lens", "store:lens(pins)") for r in arditi_rows)
        assert all(mod.serving_path_pin(r) == ("lens", "store:lens(pins)") for r in arditi_rows)             # the grading host plan's reading too
    itdoc = json.loads(G8K.STORE_PINS_PATHS[2].read_text())
    assert "serving_engine_by_leg" not in itdoc["_meta"] and not any("arditi" in k or "alphasteer" in k for k in itdoc)   # no steer leg: untouched
    # the all-positions record is the FOURTH GSM8K record and admits exactly the row's two legs; the MATH-500 record admits the row-53 leg on
    # its `it` key view; the three are ONE launch tree (the launch records' sha), the same on both launchers
    allpos, m5t = json.loads(G8K.STORE_PINS_PATHS[3].read_text()), json.loads(CELLS.STORE_PINS_PATHS[0].read_text())
    g_keys = G8K.leg_keys(next(r for r in G8K.matrix_rows(G8K.resolve_matrix("arditi_allpos")) if r["cell"] == ARDITI_ALLPOS))
    m_key = CELLS.leg_keys(next(r for r in CELLS.M5T_CELLS if r["cell"] == ARDITI_ALLPOS))[0]
    assert G8K.STORE_PINS_PATHS[3].name == "store_pins_arditi_allpos.json" and set(allpos) - {"_meta"} == set(g_keys)
    assert set(g_keys) | {m_key} == ALLPOS_LEGS_ENGINE_PENDING and m_key.endswith("@it")
    assert len({allpos[k] for k in g_keys} | {m5t[m_key]}) == 1 and re.fullmatch(r"[0-9a-f]{40}", m5t[m_key])


def test_the_all_positions_legs_carry_the_engine_their_launch_records_name_once_the_stamp_has_run():
    """After the stamp: every all-positions leg's entry exists, its engine is the launch record's (`lora_native` in the standup records of
    jobs 544615 / 544730 / 544616), its producer tree is the leg's pin, its readings all name the native engine, its provenance is a pull's
    relative path, and the all-positions GSM8K record's schema has moved to /2."""
    g, m = json.loads(G8K.STORE_PINS_PATHS[3].read_text()), json.loads(CELLS.STORE_PINS_PATHS[0].read_text())
    for doc, keys in ((g, sorted(k for k in ALLPOS_LEGS_ENGINE_PENDING if not k.endswith("@it"))),
                      (m, sorted(k for k in ALLPOS_LEGS_ENGINE_PENDING if k.endswith("@it")))):
        assert keys and all(k in doc for k in keys), keys
        block = doc["_meta"].get("serving_engine_by_leg") or {}
        for k in keys:
            assert k in block, f"{k}: admitted, engine not stamped yet"
            entry = block[k]
            assert entry["engine"] == ALLPOS_LAUNCH_ENGINE and entry["producer_tree_sha"] == doc[k], (k, entry)
            assert entry["readings"] and _readings_engines(entry["readings"]) == {SP.ENGINE_NATIVE} and not entry["pulled_store"].startswith("/"), (k, entry)
    assert g["_meta"]["schema"] == "g8k_store_pins/2"


def test_the_store_reader_reads_every_marker_and_refuses_a_store_that_names_no_engine_or_cannot_be_read(tmp_path):
    lens = _steer_store(tmp_path / "lens", "lens")
    s = SP.read_store_engines(lens)
    assert s.engines == ("lens",) and [(f, field, e) for f, field, _v, e in s.readings] == [
        ("evidence.json", "serving_identity", "lens"),
        ("cells/arditi__controls__t0__a0__11111111__22222222/manifest.json", "route.serving_identity", "lens"),
        ("cells/arditi__controls__t0__a0__11111111__22222222/evidence.json", "serving_identity", "lens")]
    native = _steer_store(tmp_path / "native", "native")
    s = SP.read_store_engines(native)
    assert s.engines == ("native",) and ("cells/arditi__controls__t0__a0__11111111__22222222/manifest.json", "route.serving_engine", "lora_native", "native") in s.readings
    # a partial store (the cell manifest written, no top-level evidence yet) still names its engine — a died-mid-run leg follows its engine
    assert SP.read_store_engines(_steer_store(tmp_path / "partial", "lens", top=False)).engines == ("lens",)
    assert SP.read_store_engines(_steer_store(tmp_path / "toponly", "native", cell=False)).engines == ("native",)
    # not a store: absent, or bookkeeping only
    assert SP.read_store_engines(tmp_path / "absent") is None and SP.read_store_engines(_steer_store(tmp_path / "ladder", None)) is None
    # a store on record that names NO engine is refused by the pin (never "absent"); unreadable JSON, an unknown identity path and an
    # unknown engine stamp refuse by name
    blank = tmp_path / "blank"; blank.mkdir(); (blank / "evidence.json").write_text(json.dumps({"engine": "vllm"}))
    assert SP.read_store_engines(blank).engines == ()
    with pytest.raises(SystemExit, match="names NO engine"):
        SP.pin_for("static_r", [blank])
    bad = tmp_path / "bad"; bad.mkdir(); (bad / "evidence.json").write_text("{not json")
    with pytest.raises(SystemExit, match="cannot be read as JSON"):
        SP.read_store_engines(bad)
    odd = tmp_path / "odd"; odd.mkdir(); (odd / "evidence.json").write_text(json.dumps({"serving_identity": "proxy_vllm+magic+interleaved"}))
    with pytest.raises(SystemExit, match="names path 'magic'"):
        SP.read_store_engines(odd)
    typo = _steer_store(tmp_path / "typo", "native"); m = typo / "cells" / "arditi__controls__t0__a0__11111111__22222222" / "manifest.json"
    m.write_text(json.dumps({"route": {"serving_identity": NATIVE_IDENTITY, "serving_engine": "typo"}}))
    with pytest.raises(SystemExit, match="serving_engine 'typo' is not one of"):
        SP.read_store_engines(typo)
    # a passthrough identity names no engine (an unsteered leg); the pin over it alone is "names NO engine"
    assert SP.identity_engine("proxy_vllm+passthrough+interleaved", where="x") is None
    assert SP.pin_for("alphasteer", None) == ("lens", "kind:alphasteer", ()) and SP.pin_for("static_r", None) == (None, "unpinned:store-unread", ())
    assert SP.pin_for("static_r", [lens, tmp_path / "absent"])[:2] == ("lens", "store:lens(disk)") and SP.pin_for("static_r", [native])[:2] == (None, "store:native(disk)")


def test_the_all_positions_arditi_cell_is_a_vector_all_kind_whose_pin_follows_a_store_that_does_not_exist_yet(tmp_path):
    """arditi-allpos-cap: ``arditi_allpos`` (the all-positions Arditi direction, object kind ``vector_all``) is a steer cell of the
    store-following pin, natively servable like ``static_r`` — and it has NEVER been run: no store on either cluster, no pins entry, and
    nothing is invented for it. So rule (c) is its production reading today: an empty wave root (an absent row root, or one holding only
    ladder bookkeeping) with the pins silent -> ``unpinned:no-store``, after which the launcher's kind-aware default selects
    ``lora_native``; with NO wave root -> ``unpinned:store-unread``, never "no store". Rules (b), (d), (e) then hold over SYNTHETIC
    stores planted under a vector_all row's root exactly as they do for static_r — a lens store pins the lens ``store:lens(disk)``, a
    native-stamped store reads ``store:native(disk)``, a mixed-engine store (or two roots / two sources disagreeing) refuses BY NAME —
    and the PINS source speaks for the row alone. KNOWN-BAD twins: a kind in NEITHER tuple refuses by name before any store is read (a
    lens store at its root changes nothing); a near-miss cell id is not the cell; the static_r reading is unchanged."""
    assert SP.steer_cell_kind(ARDITI_ALLPOS) == VECTOR_ALL == SP.KIND_VECTOR_ALL == SP.STEER_CELL_KINDS[ARDITI_ALLPOS]
    assert VECTOR_ALL in SP.NATIVELY_SERVABLE_KINDS and VECTOR_ALL not in SP.LENS_ONLY_KINDS
    # the partition: every steer kind in exactly one tuple, the servable ones exactly static_r + vector_all, lens-only alphasteer alone
    assert set(SP.LENS_ONLY_KINDS) | set(SP.NATIVELY_SERVABLE_KINDS) == set(SP.STEER_CELL_KINDS.values())
    assert not (set(SP.LENS_ONLY_KINDS) & set(SP.NATIVELY_SERVABLE_KINDS))
    assert SP.NATIVELY_SERVABLE_KINDS == ("static_r", "vector_all") and SP.LENS_ONLY_KINDS == ("alphasteer",)
    # (c) NO store: an absent row root under an empty wave root; a root holding only ladder bookkeeping; no roots at all — the pins silent
    wave = tmp_path / "wave"; wave.mkdir()
    assert SP.pin_for(VECTOR_ALL, [wave / "row_root"], ()) == (None, "unpinned:no-store", ())
    assert SP.pin_for(VECTOR_ALL, [_steer_store(wave / "ladder_only", None, cell_id=ARDITI_ALLPOS)], ()) == (None, "unpinned:no-store", ())
    assert SP.pin_for(VECTOR_ALL, [], ()) == (None, "unpinned:no-store", ())
    # no wave root: the disk is unread — never "no store"
    assert SP.pin_for(VECTOR_ALL, None, ()) == (None, "unpinned:store-unread", ())
    # (b) a synthetic LENS store under the row's root pins the lens; the store read is returned
    lens = _steer_store(wave / "lens_row", "lens", cell_id=ARDITI_ALLPOS)
    path, reason, stores = SP.pin_for(VECTOR_ALL, [lens], ())
    assert (path, reason) == ("lens", "store:lens(disk)") and [s.root for s in stores] == [str(lens)] and stores[0].engines == ("lens",)
    assert f"cells/{ARDITI_ALLPOS}__controls__t0__a0__11111111__22222222/manifest.json" in [f for f, *_ in stores[0].readings]
    assert SP.pin_for(VECTOR_ALL, [lens, wave / "absent"], ())[:2] == ("lens", "store:lens(disk)")       # an absent sibling root is silent
    # (d) a synthetic NATIVE-stamped store: unpinned, the store's engine named — the launcher then serves natively as the store did
    native = _steer_store(wave / "native_row", "native", cell_id=ARDITI_ALLPOS)
    path, reason, stores = SP.pin_for(VECTOR_ALL, [native], ())
    assert (path, reason) == (None, "store:native(disk)") and stores[0].engines == ("native",)
    # (e) a mixed-engine store refuses BY NAME (the kind and both readings in the message); so do two roots that disagree
    mixed = _steer_store(wave / "mixed_row", "native", cell_id=ARDITI_ALLPOS)
    (mixed / "evidence.json").write_text(json.dumps({"serving_identity": LENS_IDENTITY}))
    with pytest.raises(SystemExit, match="a vector_all row's store on record carries BOTH engines' stamps") as exc:
        SP.pin_for(VECTOR_ALL, [mixed], ())
    assert "evidence.json:serving_identity='proxy_vllm+lens+interleaved' -> lens" in str(exc.value) and "-> native" in str(exc.value)
    assert str(mixed) in str(exc.value)
    with pytest.raises(SystemExit, match="BOTH engines' stamps"):
        SP.pin_for(VECTOR_ALL, [lens, native], ())
    # the PINS source alone speaks for a vector_all row too (its canonical store on the other cluster); sources agree or refuse
    leg = "synthetic-leg-" + ARDITI_ALLPOS
    lens_pins = SP.read_pins_engines([_pins_record(tmp_path / "pins_lens.json", {leg: "lens"})], [leg])
    native_pins = SP.read_pins_engines([_pins_record(tmp_path / "pins_native.json", {leg: "lora_native"})], [leg])
    assert SP.pin_for(VECTOR_ALL, [wave / "absent"], lens_pins)[:2] == ("lens", "store:lens(pins)") == SP.pin_for(VECTOR_ALL, None, lens_pins)[:2]
    assert SP.pin_for(VECTOR_ALL, [wave / "absent"], native_pins)[:2] == (None, "store:native(pins)")
    assert SP.pin_for(VECTOR_ALL, [lens], lens_pins)[:2] == ("lens", "store:lens(disk+pins)")
    with pytest.raises(SystemExit, match=r"BOTH engines' stamps \(disk: \['native'\]; pins: \['lens'\]\)"):
        SP.pin_for(VECTOR_ALL, [native], lens_pins)
    # a vector_all store that names NO engine is refused, never "absent"
    blank = wave / "blank_row"; blank.mkdir(); (blank / "evidence.json").write_text(json.dumps({"engine": "vllm"}))
    with pytest.raises(SystemExit, match="a vector_all row's store is on record but names NO engine"):
        SP.pin_for(VECTOR_ALL, [blank], ())
    # KNOWN-BAD: a kind in NEITHER tuple refuses by name BEFORE any source is read — a lens store at its root, lens pins, change nothing
    for bad_kind in ("psr", "lora_all", "vector", "diffmean", ""):
        with pytest.raises(SystemExit, match=f"kind {bad_kind!r} is neither lens-only") as exc:
            SP.pin_for(bad_kind, [lens], lens_pins)
        assert "store:" not in str(exc.value) and "['alphasteer']" in str(exc.value) and "['static_r', 'vector_all']" in str(exc.value)
    # a near-miss id is not the cell
    for near in ("arditi_all", "arditi-allpos", "ARDITI_ALLPOS", "vector_all"):
        with pytest.raises(SystemExit, match="not a steer cell"):
            SP.steer_cell_kind(near)
    # the static_r and alphasteer readings are UNCHANGED by the new kind
    assert SP.pin_for("static_r", [lens], ())[:2] == ("lens", "store:lens(disk)") and SP.pin_for("static_r", [wave / "absent"], ()) == (None, "unpinned:no-store", ())
    assert SP.pin_for("static_r", [native], ())[:2] == (None, "store:native(disk)") and SP.pin_for("static_r", None, ()) == (None, "unpinned:store-unread", ())
    assert SP.pin_for("alphasteer", [native], native_pins) == ("lens", "kind:alphasteer", ())
    assert SP.LAUNCHER_REASONS == ("kind:alphasteer", "store:lens", "unpinned:no-store", "store:native")   # no new reason: the rule is the same


#: the ONE source line each kind table is declared on (anchored regexes: the harness below rewrites a whole line, whatever its value —
#: so the pending line can be emptied or deleted upstream without the harness losing its anchor)
LINE_SERVABLE = r"^NATIVELY_SERVABLE_KINDS: tuple\[str, \.\.\.\] = .*$"
LINE_LENS_ONLY = r"^LENS_ONLY_KINDS: tuple\[str, \.\.\.\] = .*$"
LINE_PENDING = r"^PENDING_ADAPTER_REGISTRATIONS: Mapping\[str, str\] = .*$"


def _exec_mutated_store_pin(replacements: dict[str, str]) -> dict:
    """Execute steer_store_pin's SOURCE with each ``line regex -> new line`` applied (every regex must match exactly one line) as a
    fresh, throwaway module — the module's import-time table checks under a KNOWN-BAD table, without touching the imported module. A
    real module object registered in ``sys.modules`` for the duration: the module's frozen dataclasses resolve their (future-annotation)
    string annotations through ``sys.modules[cls.__module__]``, so a bare namespace cannot execute the source."""
    path = REPO / "scripts" / "sdfbase" / "full22" / "steer_store_pin.py"
    src = path.read_text()
    for pattern, new_line in replacements.items():
        src, n = re.subn(pattern, lambda _m: new_line, src, flags=re.M)
        assert n == 1, (n, pattern)
    name = "steer_store_pin_mutated_for_pins_test"
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    mod.__file__ = str(path)
    sys.modules[name] = mod
    try:
        exec(compile(src, str(path), "exec"), mod.__dict__)
    finally:
        sys.modules.pop(name, None)
    return mod.__dict__


def test_the_pending_adapter_registration_is_declared_in_the_module_and_a_known_bad_table_refuses_at_import():
    """The gap between the mirror and registry_adapter.ARM_OBJECT_KINDS is DECLARED (SP.PENDING_ADAPTER_REGISTRATIONS), consistent with
    the mirror, and checked at import: the module's own source re-executed with a KNOWN-BAD table — a steer kind in neither tuple, in
    both tuples, a pending cell the mirror does not have, a pending cell under another kind — refuses by name before any row could be
    resolved; the unmutated source executes and reproduces the imported tables (the harness itself is not the thing failing)."""
    # the ONLY registration ever declared pending is arditi_allpos, as vector_all — the mirror's kind; whether it is STILL pending on this
    # tree is the two-state question _assert_mirror_agrees / _assert_no_pending_registration_has_landed answer against the live adapter
    assert set(SP.PENDING_ADAPTER_REGISTRATIONS) <= {ARDITI_ALLPOS}, dict(SP.PENDING_ADAPTER_REGISTRATIONS)
    assert all(SP.STEER_CELL_KINDS[c] == k == VECTOR_ALL for c, k in SP.PENDING_ADAPTER_REGISTRATIONS.items())
    ns = _exec_mutated_store_pin({})
    assert ns["STEER_CELL_KINDS"] == SP.STEER_CELL_KINDS and ns["PENDING_ADAPTER_REGISTRATIONS"] == SP.PENDING_ADAPTER_REGISTRATIONS
    assert ns["LENS_ONLY_KINDS"] == SP.LENS_ONLY_KINDS and ns["NATIVELY_SERVABLE_KINDS"] == SP.NATIVELY_SERVABLE_KINDS
    servable, lens_only, pending = LINE_SERVABLE, LINE_LENS_ONLY, LINE_PENDING
    # a steer kind in NEITHER tuple (vector_all dropped from the servable set)
    with pytest.raises(SystemExit, match=r"do not partition .*in neither \['vector_all'\]"):
        _exec_mutated_store_pin({servable: 'NATIVELY_SERVABLE_KINDS: tuple[str, ...] = (KIND_STATIC_R,)'})
    # a steer kind in BOTH tuples
    with pytest.raises(SystemExit, match=r"in both LENS_ONLY_KINDS and NATIVELY_SERVABLE_KINDS \['vector_all'\]"):
        _exec_mutated_store_pin({lens_only: 'LENS_ONLY_KINDS: tuple[str, ...] = (KIND_ALPHASTEER, KIND_VECTOR_ALL)'})
    # a pending cell the mirror does not have; a pending cell under another kind than the mirror's
    with pytest.raises(SystemExit, match=r"PENDING_ADAPTER_REGISTRATIONS \{'arditi_other': 'vector_all'\} does not match STEER_CELL_KINDS"):
        _exec_mutated_store_pin({pending: 'PENDING_ADAPTER_REGISTRATIONS: Mapping[str, str] = {"arditi_other": KIND_VECTOR_ALL}'})
    with pytest.raises(SystemExit, match=r"PENDING_ADAPTER_REGISTRATIONS \{'arditi_allpos': 'static_r'\} does not match STEER_CELL_KINDS"):
        _exec_mutated_store_pin({pending: 'PENDING_ADAPTER_REGISTRATIONS: Mapping[str, str] = {"arditi_allpos": KIND_STATIC_R}'})
    # every refusal is the module's own by-name form
    with pytest.raises(SystemExit, match=r"^REFUSED \(steer store pin\): "):
        _exec_mutated_store_pin({servable: 'NATIVELY_SERVABLE_KINDS: tuple[str, ...] = ()'})


def _assert_mirror_agrees() -> tuple[dict, dict, dict]:
    """THE MIRROR in either of its two states, read LIVE off the modules (so a monkeypatched adapter or table is what it judges): for
    every steer cell registry_adapter.ARM_OBJECT_KINDS KNOWS the two tables agree on the kind — a landed pending cell included; every
    steer cell the adapter does NOT know is declared in PENDING_ADAPTER_REGISTRATIONS with the mirror's kind. Returns (known, unknown,
    landed). It never fails on a landed registration whose kinds agree — asking for that line's deletion is
    :func:`_assert_no_pending_registration_has_landed`'s named job, so an upstream registration merges green here."""
    from sisteer.capeval.petri import registry_adapter as ra
    adapter = ra.ARM_OBJECT_KINDS
    known = {c: k for c, k in SP.STEER_CELL_KINDS.items() if c in adapter}
    unknown = {c: k for c, k in SP.STEER_CELL_KINDS.items() if c not in adapter}
    landed = {c: k for c, k in SP.PENDING_ADAPTER_REGISTRATIONS.items() if c in adapter}
    disagree = {c: (k, adapter[c]) for c, k in known.items() if adapter[c] != k}
    assert known and not disagree, (f"steer_store_pin.STEER_CELL_KINDS disagrees with registry_adapter.ARM_OBJECT_KINDS on {sorted(disagree)} "
                                    f"(mirror, adapter): {disagree} — the mirror is corrected to the adapter, never the reverse")
    undeclared = {c: k for c, k in unknown.items() if SP.PENDING_ADAPTER_REGISTRATIONS.get(c) != k}
    assert not undeclared, (f"steer cell(s) {sorted(undeclared)} are in STEER_CELL_KINDS but registry_adapter.ARM_OBJECT_KINDS does not know them "
                            f"and PENDING_ADAPTER_REGISTRATIONS does not declare them with the mirror's kind ({undeclared}): drift, or an undeclared gap")
    return known, unknown, landed


def _assert_no_pending_registration_has_landed() -> None:
    """The stale-line check, by name: a pending registration the adapter now KNOWS has served its purpose (the kinds agree —
    _assert_mirror_agrees ruled on that first); the message names the cell, the constant, the file and the one edit to make."""
    from sisteer.capeval.petri import registry_adapter as ra
    landed = {c: ra.ARM_OBJECT_KINDS[c] for c in SP.PENDING_ADAPTER_REGISTRATIONS if c in ra.ARM_OBJECT_KINDS}
    assert not landed, (f"registration(s) {sorted(landed)} have LANDED: registry_adapter.ARM_OBJECT_KINDS now maps {landed} and the mirror is "
                        f"compared against it for them. Delete the entr{'y' if len(landed) == 1 else 'ies'} for {sorted(landed)} from "
                        "PENDING_ADAPTER_REGISTRATIONS in scripts/sdfbase/full22/steer_store_pin.py — nothing else needs to change")


def test_no_declared_pending_registration_has_landed_in_the_adapter():
    """The named stale-line check on THIS tree: green while the adapter does not know the pending cell; the day it does (the upstream
    registration), red with the one-line deletion to make — while the mirror test itself stays green, the kinds agreeing."""
    _assert_no_pending_registration_has_landed()


def test_the_mirror_holds_in_the_registered_state_and_names_the_stale_pending_line(monkeypatch):
    """BOTH states of the two-state mirror, each SIMULATED rather than assumed of the tree — the tree has been in each of them in
    turn, so neither may be the test's premise. (1) REGISTERED with a stale pending line: the mirror passes with nothing edited,
    arditi_allpos is a KNOWN cell, and the stale-line check fails with the message a reader acts on (the cell, LANDED, Delete, the
    constant, the file, the adapter's kind). (2) After the cleanup, the pending set emptied: every check passes — this is the tree's
    real state since the registration landed. (3) PENDING, the tree's earlier state, simulated by removing the id from the adapter:
    the cell is unknown and the pending entry declares it. KNOWN-BAD twins: the adapter registering the cell under ANOTHER kind is a
    named disagreement; a steer cell the adapter does not know and the pending set does not declare is named as undeclared; a pending
    entry under a kind other than the mirror's is named the same way (and refused at import by the module itself, see above)."""
    from sisteer.capeval.petri import registry_adapter as ra
    original, original_pending = dict(ra.ARM_OBJECT_KINDS), dict(SP.PENDING_ADAPTER_REGISTRATIONS)
    # the tree today: the registration has LANDED and the pending line is gone. Both facts are asserted, because each is the
    # premise of one of the states below — and each was the opposite a commit ago.
    assert original.get(ARDITI_ALLPOS) == VECTOR_ALL and original_pending == {}
    # (3) the PENDING state, simulated: the adapter does not know the cell and the pending entry declares it with the mirror's kind
    monkeypatch.setattr(ra, "ARM_OBJECT_KINDS", {k: v for k, v in original.items() if k != ARDITI_ALLPOS})
    monkeypatch.setattr(SP, "PENDING_ADAPTER_REGISTRATIONS", {ARDITI_ALLPOS: VECTOR_ALL})
    known, unknown, landed = _assert_mirror_agrees()
    assert unknown == {ARDITI_ALLPOS: VECTOR_ALL} and not landed and ARDITI_ALLPOS not in known
    _assert_no_pending_registration_has_landed()          # nothing has landed yet, so nothing to clean
    # (1) the REGISTERED state, with the pending line not yet removed
    monkeypatch.setattr(ra, "ARM_OBJECT_KINDS", {**original, ARDITI_ALLPOS: VECTOR_ALL})
    known, unknown, landed = _assert_mirror_agrees()
    assert known[ARDITI_ALLPOS] == VECTOR_ALL and unknown == {} and landed == {ARDITI_ALLPOS: VECTOR_ALL} and set(known) == set(SP.STEER_CELL_KINDS)
    with pytest.raises(AssertionError) as exc:
        _assert_no_pending_registration_has_landed()
    msg = str(exc.value)
    assert all(s in msg for s in ("['arditi_allpos']", "LANDED", "Delete the entry", "PENDING_ADAPTER_REGISTRATIONS",
                                  "scripts/sdfbase/full22/steer_store_pin.py", "{'arditi_allpos': 'vector_all'}", "nothing else needs to change")), msg
    # the state AFTER the cleanup: the pending set emptied, the adapter knowing every steer cell — both checks pass, nothing else to edit
    monkeypatch.setattr(SP, "PENDING_ADAPTER_REGISTRATIONS", {})
    known, unknown, landed = _assert_mirror_agrees()
    assert set(known) == set(SP.STEER_CELL_KINDS) and not unknown and not landed
    _assert_no_pending_registration_has_landed()
    # KNOWN-BAD: registered under ANOTHER kind — a disagreement, named with both kinds (mirror, adapter)
    monkeypatch.setattr(SP, "PENDING_ADAPTER_REGISTRATIONS", {ARDITI_ALLPOS: VECTOR_ALL})
    monkeypatch.setattr(ra, "ARM_OBJECT_KINDS", {**original, ARDITI_ALLPOS: "static_r"})
    with pytest.raises(AssertionError, match=r"disagrees with registry_adapter\.ARM_OBJECT_KINDS on \['arditi_allpos'\].*'arditi_allpos': \('vector_all', 'static_r'\)"):
        _assert_mirror_agrees()
    # KNOWN-BAD: a steer cell nobody declares (the adapter does not know it, the pending set does not name it)
    monkeypatch.setattr(ra, "ARM_OBJECT_KINDS", {k: v for k, v in original.items() if k != "arditi_other"})
    monkeypatch.setattr(SP, "PENDING_ADAPTER_REGISTRATIONS", {})
    monkeypatch.setattr(SP, "STEER_CELL_KINDS", {**SP.STEER_CELL_KINDS, "arditi_other": VECTOR_ALL})
    with pytest.raises(AssertionError, match=r"steer cell\(s\) \['arditi_other'\] are in STEER_CELL_KINDS .* does not declare them"):
        _assert_mirror_agrees()
    # KNOWN-BAD: pending under a kind the mirror does not give the cell. Its premise is the PENDING world — the wrong-kind
    # check only applies to a cell the adapter does not know — so the adapter is patched back to not knowing it.
    monkeypatch.setattr(SP, "STEER_CELL_KINDS", {c: k for c, k in SP.STEER_CELL_KINDS.items() if c != "arditi_other"})
    monkeypatch.setattr(ra, "ARM_OBJECT_KINDS", {k: v for k, v in original.items() if k != ARDITI_ALLPOS})
    monkeypatch.setattr(SP, "PENDING_ADAPTER_REGISTRATIONS", {ARDITI_ALLPOS: "static_r"})
    with pytest.raises(AssertionError, match=r"\['arditi_allpos'\] are in STEER_CELL_KINDS .*does not declare them with the mirror's kind \(\{'arditi_allpos': 'vector_all'\}\)"):
        _assert_mirror_agrees()
    # everything restored: today's REAL state reads back — the registration has landed and the pending line is gone, so the
    # cell is KNOWN, nothing is unknown, and nothing is left pending. (This assertion used to read the other way round; it is
    # the tree's state that moved, not the mirror's rule, which is why the rule itself needed no edit across the landing.)
    monkeypatch.undo()
    monkeypatch.setenv("SISTEER_CONFIG", str(_SITE_CFG))   # undo() dropped the module's synthetic site too
    known, unknown, landed = _assert_mirror_agrees()
    assert known[ARDITI_ALLPOS] == VECTOR_ALL and unknown == {} and landed == {}
    _assert_no_pending_registration_has_landed()


def test_the_store_pin_mirrors_are_pinned_to_the_library_and_agree_with_the_math500_readers_engine():
    """steer_store_pin is stdlib-only (the node's python3 before any venv), so its kind table, lens-only kinds and engine markers are
    MIRRORS — checked here against registry_adapter / defense_cells / manifest / contracts, never trusted. Its engine reading of a route
    agrees with the MATH-500 module's `certification_path_of` (the validators' route-keyed `native_engine` selector)."""
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import defense_cells as dc, registry_adapter as ra
    # THE MIRROR, in whichever of its two states this tree is in (the arditi_allpos registration lands upstream separately): every steer
    # cell the adapter KNOWS agrees on the kind; every cell it does NOT know is a declared pending registration with the mirror's kind.
    # A landed registration passes here (the kinds agree) and is asked for by name in test_no_declared_pending_registration_has_landed.
    known, unknown, landed = _assert_mirror_agrees()
    assert set(known) | set(unknown) == set(SP.STEER_CELL_KINDS) and {"alphasteer", "arditi"} <= set(known)
    assert set(unknown) | set(landed) == set(SP.PENDING_ADAPTER_REGISTRATIONS)          # every pending entry is still unknown, or has landed
    assert SP.STEER_CELL_KINDS == {"alphasteer": "alphasteer", "arditi": "static_r", "arditi_allpos": "vector_all"}
    # the launchers schedule a SUBSET of the mirror's cells, each with the mirror's kind (the arditi_allpos row lands in each launcher
    # separately — this holds before and after); the only cells the mirror knows that NO launcher schedules are pending registrations
    for mod in (G8K, CELLS):
        assert set(mod.STEER_CELLS) <= set(SP.STEER_CELL_KINDS), (mod.__name__, mod.STEER_CELLS)
        assert {c: SP.steer_cell_kind(c) for c in mod.STEER_CELLS} == {c: SP.STEER_CELL_KINDS[c] for c in mod.STEER_CELLS}
    unscheduled = set(SP.STEER_CELL_KINDS) - set(G8K.STEER_CELLS) - set(CELLS.STEER_CELLS)
    assert unscheduled <= set(SP.PENDING_ADAPTER_REGISTRATIONS), unscheduled
    # the lens-only / natively-servable split is the library's disposition of each kind, and the servable kinds are residual-slot kinds
    for kind in SP.STEER_CELL_KINDS.values():
        assert (dc.native_serving_disposition(kind, None)[0] == "skip") == (kind in SP.LENS_ONLY_KINDS) == (kind not in SP.NATIVELY_SERVABLE_KINDS), kind
    assert dc.native_serving_disposition("static_r", None) == ("serve", None)            # Arditi IS natively servable (native-psr-serving)
    assert dc.native_serving_disposition("vector_all", None) == ("serve", None)          # ...and so is the all-positions direction's kind
    assert set(SP.NATIVELY_SERVABLE_KINDS) <= set(dc.RESIDUAL_SLOT_KINDS) <= set(dc.NATIVE_LORA_SERVABLE_KINDS)
    assert ra.ARM_OBJECT_KINDS["vector_all"] == "vector_all" == SP.KIND_VECTOR_ALL      # the KIND string itself is the adapter's
    assert SP.IDENTITY_PATHS == cm.SERVING_PATHS["proxy_vllm"] and SP.SERVED_ENGINE_FIELD == C.SERVED_ENGINE_FIELD
    assert SP.SERVING_ENGINES == C.SERVING_ENGINES and SP.SERVING_ENGINE_LORA_NATIVE == C.SERVING_ENGINE_LORA_NATIVE and SP.SERVING_ENGINE_LENS == C.SERVING_ENGINE_LENS
    assert LENS_IDENTITY == cm.serving_identity("proxy_vllm", "lens", "interleaved") and NATIVE_IDENTITY == cm.serving_identity("proxy_vllm", "lora", "interleaved")
    assert cm.parse_serving_identity(LENS_IDENTITY).path == SP.IDENTITY_PATH_LENS == M.STEERED_PATH_LENS
    assert cm.parse_serving_identity(NATIVE_IDENTITY).path == SP.IDENTITY_PATH_LORA == M.STEERED_PATH_LORA
    assert SP.SERVING_PATH_NATIVE == ra.DEFAULT_STANDUP_SERVING_PATH == "lora_native" and SP.SERVING_PATH_LENS == "lens"
    assert SP.LAUNCHER_REASONS == ("kind:alphasteer", "store:lens", "unpinned:no-store", "store:native")
    assert SP.store_reason("lens", ["pins", "disk"]) == "store:lens(disk+pins)" and SP.store_reason("native", ["pins"]) == "store:native(pins)"
    # the leg keys the pins are consulted under are the analysis readers' (the mirrors, pinned)
    from sisteer.capeval.tinycap import contracts as _C  # noqa: F401 — the analysis modules are loaded above (M, V)
    for r in G8K.matrix_rows():
        assert G8K.leg_keys(r) == [V.leg_key(r["model_key"], r["view"], r["cell"], cap) for cap in G8K.RUNGS], r["job_id"]
    for r in CELLS.M5T_CELLS:
        assert CELLS.leg_keys(r) == [M.row_leg_key(r)], r["job_id"]
    assert CELLS.LEG_KEY_VIEW_SEP == M.LEG_KEY_VIEW_SEP == "@"
    assert (SP.PINS_META_KEY, SP.PINS_ENGINE_BLOCK, SP.PINS_ENGINE_FIELD) == ("_meta", "serving_engine_by_leg", "engine")
    # the reader's engine == certification_path_of's path on the two routes
    lens_route = {"serving_identity": LENS_IDENTITY}
    native_route = {"serving_identity": NATIVE_IDENTITY, C.SERVED_ENGINE_FIELD: C.SERVING_ENGINE_LORA_NATIVE}
    assert M.certification_path_of(lens_route) == ("lens", []) and SP.identity_engine(LENS_IDENTITY, where="x") == "lens"
    assert M.certification_path_of(native_route) == ("lora", []) and SP.identity_engine(NATIVE_IDENTITY, where="x") == "native" == SP.stamp_engine("lora_native", where="x")
    assert V.STAMP_CLASS_NATIVE == "native_engine"


def _plan(mod_path, *args):
    r = subprocess.run([sys.executable, str(mod_path), "plan", "--sha", SHA_ANY, *args], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    return [l for l in r.stdout.splitlines() if "sbatch" in l]


def test_the_planned_line_carries_the_lens_route_only_for_a_submission_pinned_lens_throughout(tmp_path, monkeypatch):
    """The grading host plan cannot read a cluster DISK store; it reads kinds and the repo's PINS records. Its F22_SERVING_PATH=lens prefix rides
    only a submission whose EVERY row pins the lens — by kind (AlphaSteer) or by the record (an Arditi row with a lens engine in the pins:
    every one today). An Arditi row the record does not know is unpinned on the grading host (`store-unread`) and drops the prefix, which would
    otherwise override that row as EXPLICIT lens on the cluster."""
    m5t = REPO / "scripts" / "sdfbase" / "math500" / "cells.py"
    alpha = _plan(m5t, "--cluster", "primary", "--wave", "steer", "--rows", "33,35,37,39", "--single-array", "--max-concurrent", "2")
    assert len(alpha) == 1 and alpha[0].startswith("F22_SERVING_PATH=lens sbatch -J m5t-steer ") and "--array=33,35,37,39%2" in alpha[0]
    with_arditi = _plan(m5t, "--cluster", "primary", "--wave", "steer", "--rows", "33,35,37,40", "--single-array", "--max-concurrent", "2")
    assert len(with_arditi) == 1 and with_arditi[0].startswith("F22_SERVING_PATH=lens sbatch -J m5t-steer ") and "--array=33,35,37,40%2" in with_arditi[0]   # row 40: lens by the record
    ext1 = _plan(m5t, "--cluster", "primary", "--wave", "ext1", "--single-array", "--max-concurrent", "2")
    assert len(ext1) == 1 and ext1[0].startswith("sbatch -J m5t-ext1 ") and "F22_SERVING_PATH" not in ext1[0]
    g8k = REPO / "scripts" / "sdfbase" / "gsm8k" / "g8k_cells.py"
    g_steer = _plan(g8k, "--cluster", "cluster_a", "--wave", "steer", "--max-concurrent", "2")            # AlphaSteer by kind + Arditi by the record
    assert g_steer and all(l.startswith("F22_SERVING_PATH=lens sbatch -J g8k-") for l in g_steer)
    g_ext = _plan(g8k, "--cluster", "cluster_a", "--wave", "ext", "--max-concurrent", "2")
    assert g_ext and all(l.startswith("sbatch -J g8k-") and "F22_SERVING_PATH" not in l for l in g_ext)
    # with a record that knows no engine the Arditi rows are unread on the grading host and the prefix goes (in-process: the plan's own reading)
    blind = (_pins_record(tmp_path / "blind.json"),)
    monkeypatch.setattr(CELLS, "STORE_PINS_PATHS", blind); monkeypatch.setattr(G8K, "STORE_PINS_PATHS", blind)
    lines = CELLS.plan(SHA_ANY, "primary", max_concurrent=2, rows="33,35,37,40", wave="steer", single_array=True).lines
    assert len(lines) == 1 and lines[0].startswith("sbatch -J m5t-steer ") and "F22_SERVING_PATH" not in lines[0]
    lines = CELLS.plan(SHA_ANY, "primary", max_concurrent=2, rows="33,35,37,39", wave="steer", single_array=True).lines
    assert len(lines) == 1 and lines[0].startswith("F22_SERVING_PATH=lens sbatch -J m5t-steer ")
    g_lines = G8K.plan(SHA_ANY, "cluster_a", max_concurrent=2, wave=G8K.WAVE_STEER).lines
    assert g_lines and all(l.startswith("sbatch -J g8k-") and "F22_SERVING_PATH" not in l for l in g_lines)
    g_lines = G8K.plan(SHA_ANY, "cluster_a", max_concurrent=2, wave=G8K.WAVE_STEER, rows="47,49,51").lines
    assert g_lines and all(l.startswith("F22_SERVING_PATH=lens sbatch -J g8k-") for l in g_lines)


def test_the_cli_prints_the_pin_and_its_reason_and_requires_the_wave_root_on_an_external_row(tmp_path):
    m5t, g8k = REPO / "scripts" / "sdfbase" / "math500" / "cells.py", REPO / "scripts" / "sdfbase" / "gsm8k" / "g8k_cells.py"

    def run(mod, *args, check=True):
        r = subprocess.run([sys.executable, str(mod), "row", *args], capture_output=True, text=True, cwd=REPO)
        if check:
            assert r.returncode == 0, r.stdout + r.stderr
        return r
    wave = tmp_path / "wave"
    blind = str(_pins_record(tmp_path / "blind.json"))
    for mod, alpha_args, arditi_args, base_args, arditi_row in (
            (m5t, ("--key", GEMMA, "--cell", ALPHASTEER, "--sampling", PRIMARY), ("--key", GEMMA, "--cell", ARDITI, "--sampling", PRIMARY),
             ("--key", GEMMA, "--cell", NONE, "--sampling", PRIMARY), _m5t_row(GEMMA, ARDITI)),
            (g8k, ("--key", GEMMA, "--view", "main", "--cell", ALPHASTEER), ("--key", GEMMA, "--view", "main", "--cell", ARDITI),
             ("--key", GEMMA, "--view", "main", "--cell", NONE), _g8k_row(GEMMA, ARDITI))):
        r = run(mod, *arditi_args, "--serving-path", check=False)                       # an external row: the wave root is REQUIRED
        assert r.returncode == 1 and "FOLLOWS ITS STORE" in r.stderr and "--wave-root" in r.stderr and r.stdout == "", (r.stdout, r.stderr)
        # the repo's records (the default): the real row's admitted leg is a lens store — pinned by the record with no disk store
        assert run(mod, *arditi_args, "--serving-path", "--wave-root", str(wave)).stdout == "lens\nstore:lens(pins)\n"
        # a record that knows nothing (--store-pins, the test seam): the disk alone
        B = ("--store-pins", blind)
        assert run(mod, *arditi_args, "--serving-path", "--wave-root", str(wave), *B).stdout == "\nunpinned:no-store\n"
        assert run(mod, *alpha_args, "--serving-path", "--wave-root", str(wave), *B).stdout == "lens\nkind:alphasteer\n"
        assert run(mod, *base_args, "--serving-path").stdout == "\nunpinned:not-external\n"     # not external: no wave root needed
        _steer_store(wave / arditi_row["out_root"], "lens")
        assert run(mod, *arditi_args, "--serving-path", "--wave-root", str(wave), *B).stdout == "lens\nstore:lens(disk)\n"
        assert run(mod, *arditi_args, "--serving-path", "--wave-root", str(wave)).stdout == "lens\nstore:lens(disk+pins)\n"
        shutil.rmtree(wave / arditi_row["out_root"]); _steer_store(wave / arditi_row["out_root"], "native")
        assert run(mod, *arditi_args, "--serving-path", "--wave-root", str(wave), *B).stdout == "\nstore:native(disk)\n"
        r = run(mod, *arditi_args, "--serving-path", "--wave-root", str(wave), check=False)          # the real record says lens: refused
        assert r.returncode == 1 and "BOTH engines' stamps" in r.stderr and "disk: ['native']" in r.stderr and "pins: ['lens']" in r.stderr and r.stdout == ""
        (wave / arditi_row["out_root"] / "evidence.json").write_text(json.dumps({"serving_identity": LENS_IDENTITY}))
        r = run(mod, *arditi_args, "--serving-path", "--wave-root", str(wave), *B, check=False)
        assert r.returncode == 1 and "BOTH engines' stamps" in r.stderr and r.stdout == ""
        shutil.rmtree(wave / arditi_row["out_root"])
    assert SP.pin_lines(None, "unpinned:no-store") == "\nunpinned:no-store\n" and SP.pin_lines("lens", "store:lens(pins)") == "lens\nstore:lens(pins)\n"


def _m5t_dry(env, *args):
    e = {**os.environ, "M5T_DRY_RUN": "1", "M5T_DRY_S": "24", "M5T_REPO_DIR": str(REPO), "M5T_DRY_SHARED": tempfile.mkdtemp(), **env}
    e.pop("F22_SERVING_PATH", None) if "F22_SERVING_PATH" not in env else None
    return subprocess.run(["bash", str(REPO / "scripts" / "sdfbase" / "math500" / "math500_thinking.sbatch"), *args], capture_output=True, text=True, env=e, cwd=REPO)


def _g8k_dry(env, *args):
    e = {**os.environ, "G8K_DRY_RUN": "1", "G8K_DRY_S": "20", "G8K_REPO_DIR": str(REPO), "G8K_DRY_SHARED": tempfile.mkdtemp(), **env}
    e.pop("F22_SERVING_PATH", None) if "F22_SERVING_PATH" not in env else None
    return subprocess.run(["bash", str(REPO / "scripts" / "sdfbase" / "gsm8k" / "gsm8k_thinking.sbatch"), *args], capture_output=True, text=True, env=e, cwd=REPO)


def test_the_dry_run_resolves_the_serving_path_from_the_row_before_init_and_refuses_a_conflicting_shell_by_name(tmp_path):
    r = _m5t_dry({}, GEMMA, ALPHASTEER, PRIMARY, SHA_ANY, "1")
    assert r.returncode == 0 and "SERVING_PATH: path=lens pinned_by_row=lens reason=kind:alphasteer" in r.stdout and "serving path: lens — pinned by the ROW (kind:alphasteer" in r.stdout, r.stdout[-2000:] + r.stderr[-800:]
    r = _m5t_dry({}, GEMMA, NONE, PRIMARY, SHA_ANY, "0")
    # there is no "launcher's default" any more — an unpinned row's path is the kind-aware AUTO over the declared cells
    assert r.returncode == 0 and "SERVING_PATH: path=lora_native pinned_by_row=none reason=unpinned:not-external" in r.stdout and "the kind-aware default selects it at f22_init" in r.stdout, r.stdout[-1500:]
    r = _m5t_dry({"F22_SERVING_PATH": "lens"}, GEMMA, NONE, PRIMARY, SHA_ANY, "0")           # a submitter's lens shell stands on a row that pins none
    assert r.returncode == 0 and "SERVING_PATH: path=lens pinned_by_row=none reason=unpinned:not-external" in r.stdout
    r = _m5t_dry({"F22_SERVING_PATH": "lora_native"}, GEMMA, ALPHASTEER, PRIMARY, SHA_ANY, "1")
    assert r.returncode != 0 and "the row pins serving path 'lens' (kind:alphasteer" in r.stdout + r.stderr and "refused by name" in r.stdout + r.stderr
    r = _g8k_dry({}, GEMMA, "main", ALPHASTEER, "8192", SHA_ANY, "1", "0")
    assert r.returncode == 0 and "SERVING_PATH: path=lens pinned_by_row=lens reason=kind:alphasteer" in r.stdout, r.stdout[-1500:] + r.stderr[-500:]
    r = _g8k_dry({}, GEMMA, "main", NONE, "8192", SHA_ANY, "0", "0")
    assert r.returncode == 0 and "SERVING_PATH: path=lora_native pinned_by_row=none reason=unpinned:not-external" in r.stdout, r.stdout[-1500:]
    r = _g8k_dry({"F22_SERVING_PATH": "lora_native"}, GEMMA, "main", ALPHASTEER, "8192", SHA_ANY, "1", "0")
    assert r.returncode != 0 and "refused by name" in r.stdout + r.stderr
    # THE ARDITI ROW FOLLOWS ITS STORE — both launchers. With the repo's records (the launcher's default) the real row is pinned lens by the
    # PINS source even with no disk store under the dry run's shared root; the disk-only readings below silence the pins with a synthetic
    # record through the test seam (M5T_STORE_PINS / G8K_STORE_PINS), then the kept refusal of a conflicting submitter export and the
    # refusal of a lens export over a native store
    shared = tmp_path / "shared"
    m5t_root = shared / M5T_ROOT_REL / _m5t_row(GEMMA, ARDITI)["out_root"]
    g8k_root = shared / G8K_ROOT_REL / _g8k_row(GEMMA, ARDITI)["out_root"]
    blind = str(_pins_record(tmp_path / "blind.json"))
    r = _m5t_dry({"M5T_DRY_SHARED": str(shared)}, GEMMA, ARDITI, PRIMARY, SHA_ANY, "1")
    assert r.returncode == 0 and "SERVING_PATH: path=lens pinned_by_row=lens reason=store:lens(pins) selection=explicit" in r.stdout and "serving path: lens — pinned by the ROW (store:lens(pins)" in r.stdout and "pins checkout)" in r.stdout, r.stdout[-1500:] + r.stderr[-500:]
    r = _g8k_dry({"G8K_DRY_SHARED": str(shared)}, GEMMA, "main", ARDITI, "8192", SHA_ANY, "1", "0")
    assert r.returncode == 0 and "SERVING_PATH: path=lens pinned_by_row=lens reason=store:lens(pins) selection=explicit" in r.stdout and "pins checkout)" in r.stdout, r.stdout[-1500:] + r.stderr[-500:]
    m5t_env, g8k_env = {"M5T_DRY_SHARED": str(shared), "M5T_STORE_PINS": blind}, {"G8K_DRY_SHARED": str(shared), "G8K_STORE_PINS": blind}
    # (c) no store: unpinned, and the kind-aware AUTO selects native (Arditi serves on the residual slot)
    r = _m5t_dry(m5t_env, GEMMA, ARDITI, PRIMARY, SHA_ANY, "1")
    assert r.returncode == 0 and "SERVING_PATH: path=lora_native pinned_by_row=none reason=unpinned:no-store selection=auto" in r.stdout, r.stdout[-1500:] + r.stderr[-500:]
    assert "serving path: not pinned by the row (unpinned:no-store" in r.stdout and f"pins {blind})" in r.stdout and "[serving-path] AUTO-selected lora_native: 1 of 1 steered cells natively servable" in r.stdout
    r = _g8k_dry(g8k_env, GEMMA, "main", ARDITI, "8192", SHA_ANY, "1", "0")
    assert r.returncode == 0 and "SERVING_PATH: path=lora_native pinned_by_row=none reason=unpinned:no-store selection=auto" in r.stdout, r.stdout[-1500:] + r.stderr[-500:]
    assert "[serving-path] AUTO-selected lora_native: 1 of 1 steered cells natively servable" in r.stdout
    # (b) a lens store on record: the lens, EXPLICIT; the GSM8K rung leg reads the cap root too
    _steer_store(m5t_root, "lens"); _steer_store(g8k_root, "lens")
    r = _m5t_dry(m5t_env, GEMMA, ARDITI, PRIMARY, SHA_ANY, "1")
    assert r.returncode == 0 and "SERVING_PATH: path=lens pinned_by_row=lens reason=store:lens(disk) selection=explicit" in r.stdout and "serving path: lens — pinned by the ROW (store:lens(disk)" in r.stdout, r.stdout[-1500:]
    r = _g8k_dry(g8k_env, GEMMA, "main", ARDITI, "8192", SHA_ANY, "1", "0")
    assert r.returncode == 0 and "SERVING_PATH: path=lens pinned_by_row=lens reason=store:lens(disk) selection=explicit" in r.stdout, r.stdout[-1500:]
    r = _g8k_dry(g8k_env, GEMMA, "main", ARDITI, "16384", SHA_ANY, "1", "0")
    assert r.returncode == 0 and "SERVING_PATH: path=lens pinned_by_row=lens reason=store:lens(disk) selection=explicit" in r.stdout, r.stdout[-1500:]
    r = _g8k_dry({"G8K_DRY_SHARED": str(shared)}, GEMMA, "main", ARDITI, "8192", SHA_ANY, "1", "0")                # disk + the repo's record
    assert r.returncode == 0 and "SERVING_PATH: path=lens pinned_by_row=lens reason=store:lens(disk+pins) selection=explicit" in r.stdout, r.stdout[-1500:]
    r = _m5t_dry({**m5t_env, "F22_SERVING_PATH": "lora_native"}, GEMMA, ARDITI, PRIMARY, SHA_ANY, "1")           # the kept refusal
    assert r.returncode != 0 and "the row pins serving path 'lens' (store:lens(disk)" in r.stdout + r.stderr and "refused by name" in r.stdout + r.stderr
    # (d) a native store on record: unpinned, AUTO native; a submitter's lens export over it refuses (it would write a mixed store); and
    # against the repo's record (lens) the native disk store is a mixed record — refused
    shutil.rmtree(m5t_root); _steer_store(m5t_root, "native"); shutil.rmtree(g8k_root); _steer_store(g8k_root, "native")
    r = _m5t_dry(m5t_env, GEMMA, ARDITI, PRIMARY, SHA_ANY, "1")
    assert r.returncode == 0 and "SERVING_PATH: path=lora_native pinned_by_row=none reason=store:native(disk) selection=auto" in r.stdout, r.stdout[-1500:]
    r = _g8k_dry(g8k_env, GEMMA, "main", ARDITI, "8192", SHA_ANY, "1", "0")
    assert r.returncode == 0 and "SERVING_PATH: path=lora_native pinned_by_row=none reason=store:native(disk) selection=auto" in r.stdout, r.stdout[-1500:]
    for r in (_m5t_dry({**m5t_env, "F22_SERVING_PATH": "lens"}, GEMMA, ARDITI, PRIMARY, SHA_ANY, "1"),
              _g8k_dry({**g8k_env, "F22_SERVING_PATH": "lens"}, GEMMA, "main", ARDITI, "8192", SHA_ANY, "1", "0")):
        assert r.returncode != 0 and "NATIVE-stamped (store:native(disk)" in r.stdout + r.stderr and "refused by name" in r.stdout + r.stderr and "SERVING_PATH:" not in r.stdout
    for r in (_m5t_dry({"M5T_DRY_SHARED": str(shared)}, GEMMA, ARDITI, PRIMARY, SHA_ANY, "1"), _g8k_dry({"G8K_DRY_SHARED": str(shared)}, GEMMA, "main", ARDITI, "8192", SHA_ANY, "1", "0")):
        assert r.returncode != 0 and "BOTH engines' stamps" in r.stdout + r.stderr and "pins: ['lens']" in r.stdout + r.stderr and "SERVING_PATH:" not in r.stdout, r.stdout[-800:] + r.stderr[-800:]
    # (e) a mixed store dies HERE, before any GPU minute, in both launchers
    (m5t_root / "evidence.json").write_text(json.dumps({"serving_identity": LENS_IDENTITY})); (g8k_root / "evidence.json").write_text(json.dumps({"serving_identity": LENS_IDENTITY}))
    for r in (_m5t_dry(m5t_env, GEMMA, ARDITI, PRIMARY, SHA_ANY, "1"), _g8k_dry(g8k_env, GEMMA, "main", ARDITI, "8192", SHA_ANY, "1", "0")):
        assert r.returncode != 0 and "BOTH engines' stamps" in r.stdout + r.stderr and "SERVING_PATH:" not in r.stdout, r.stdout[-800:] + r.stderr[-800:]


def _served_cells_slice(cells_served, all_cells, engine="lens:stacked"):
    """f22_standup's SERVED_CELLS block (between the wave test's two markers), RUN under set -u with or without a roster."""
    src = (REPO / "scripts" / "sdfbase" / "full22" / "common.sh").read_text().splitlines()
    i = next(k for k, l in enumerate(src) if l.startswith("  # SERVED_CELLS (native-serving-default): ALL_CELLS minus"))
    j = next(k for k, l in enumerate(src) if k > i and l.startswith("  local adj_json;"))
    out = Path(tempfile.mkdtemp())
    (out / "endpoint.json").write_text(json.dumps({"cells_served": str(out / "cells_served.json")} if cells_served is not None else {}))
    if cells_served is not None:
        (out / "cells_served.json").write_text(json.dumps(cells_served))
    roster = ("ALL_CELLS=(" + " ".join(all_cells) + ")\n") if all_cells is not None else ""
    script = ("set -u\nf22_log(){ echo \"LOG $*\"; }\n" f"SPY={sys.executable}\nREPO_DIR={REPO}\nout={out}\nTARGET_ENGINE={engine}\n" + roster
              + "served(){\n  local ep=\"$out/endpoint.json\"\n" + "\n".join(src[i:j]) + "\n  echo \"SERVED=${SERVED_CELLS[*]}\"\n  return 0\n}\nserved; echo \"rc=$?\"\n")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return int(r.stdout.strip().splitlines()[-1].split("rc=")[1]), r.stdout + r.stderr


def test_a_single_cell_launch_with_an_empty_roster_and_a_healthy_standup_passes_traffic_and_a_record_lacking_cells_requested_refuses_by_name():
    """The #138 served-cells guard for a SINGLE-CELL job (the MATH-500 / GSM8K launchers set no campaign roster): a healthy standup whose
    own record names its cells_requested PASSES traffic with exactly those cells served; the native engine's skip of the object cell
    leaves the twin alone served; a record lacking cells_requested, or no record at all, REFUSES BY NAME — never an empty served set."""
    rc, out = _served_cells_slice({"cells_requested": ["none", ALPHASTEER]}, None)                # the lens standup's own record
    assert rc == 0 and f"SERVED=none {ALPHASTEER}" in out and "the standup's own cells_requested (none alphasteer)" in out, out
    rc, out = _served_cells_slice({"cells_requested": ["none", ALPHASTEER], "cells_skipped": [{"cell_id": ALPHASTEER, "kind": ALPHASTEER, "reason": "no PEFT form"}]},
                                  None, engine="lora_native")
    assert rc == 0 and "SERVED=none\n" in out and "SKIPPED BY NAME" in out, out                    # the native engine skipped the object cell
    rc, out = _served_cells_slice({"cells_skipped": []}, None)                                   # KNOWN-BAD: a record naming no cells
    assert rc == 1 and "REFUSING TRAFFIC" in out and "no roster (ALL_CELLS is empty" in out and "names no cells_requested" in out, out
    rc, out = _served_cells_slice(None, None)                                                    # KNOWN-BAD: no record at all
    assert rc == 1 and "REFUSING TRAFFIC" in out and "cannot be derived" in out, out


def test_the_whole_view_served_cells_slice_stays_byte_identical_under_the_campaign_roster():
    """The campaign's whole-view path (ALL_CELLS set by f22_bank_view) reads exactly as before the single-cell derivation was added:
    the roster minus the skipped cells, nothing said about a roster source; every skipped cell refuses the whole view as before."""
    rc, out = _served_cells_slice({"cells_skipped": []}, ["none", "a", "b"])
    assert rc == 0 and "SERVED=none a b" in out and "single-cell job" not in out and "cells_requested" not in out, out
    rc, out = _served_cells_slice({"cells_requested": ["x"], "cells_skipped": []}, ["none", "a"])   # a record's cells_requested never overrides a roster
    assert rc == 0 and "SERVED=none a" in out and "cells_requested" not in out, out
    rc, out = _served_cells_slice({"cells_skipped": [{"cell_id": "psr", "kind": "psr", "reason": "r"}]}, ["psr"], engine="lora_native")
    assert rc == 1 and "REFUSING TRAFFIC: the standup serves none of psr" in out, out
    # the slice the wave test's harness pins is the one read here: the same two markers bound it
    src = (REPO / "scripts" / "sdfbase" / "full22" / "common.sh").read_text()
    assert src.count("  # SERVED_CELLS (native-serving-default): ALL_CELLS minus") == 1 and src.count("  local adj_json;") == 1


def test_the_dry_run_tools_start_each_rows_dry_run_without_the_enclosing_jobs_serving_path():
    """The prod dry run (dry_run_prod.sbatch) resolves its own f22_init default and EXPORTS it before the tool spawns every row's
    dry run; a steer row must still pass (its own pin selects the lens), never read the parent's default as a submitter conflict —
    job 507051 refused row 33 exactly that way. Both tools, one row each, under an exported default and under an exported lens."""
    m5t = REPO / "scripts" / "sdfbase" / "math500" / "m5t_dry_run.py"
    for exported in ("lora_native", "lens"):
        env = {**os.environ, "F22_SERVING_PATH": exported, "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}"}
        r = subprocess.run([sys.executable, str(m5t), "--stage", "pre", "--sha", SHA_ANY, "--only", GEMMA, ALPHASTEER, PRIMARY],
                           capture_output=True, text=True, cwd=REPO, env=env)
        assert r.returncode == 0 and "VERDICT: PASS" in r.stdout and "SERVING_PATH: path=lens pinned_by_row=lens" in r.stdout, (exported, r.stdout[-1500:], r.stderr[-800:])
        # launch-cells-wiring: the tool's plan carries the declared cells and the selection mode (the row's pin, recorded EXPLICIT); the
        # pin's REASON rides the line since arditi-native-pin
        assert "SERVING_PATH: path=lens pinned_by_row=lens reason=kind:alphasteer selection=explicit" in r.stdout and "LAUNCH_CELLS: source=F22_LAUNCH_CELLS cells=none alphasteer" in r.stdout, r.stdout[-1500:]
        r = subprocess.run([sys.executable, str(m5t), "--stage", "pre", "--sha", SHA_ANY, "--only", GEMMA, NONE, PRIMARY],
                           capture_output=True, text=True, cwd=REPO, env=env)
        assert r.returncode == 0 and "SERVING_PATH: path=lora_native pinned_by_row=none reason=unpinned:not-external" in r.stdout, (exported, r.stdout[-1500:])
        # an Arditi row under the tool's fresh scratch shared root has no DISK store, but the repo's store-pins record knows its admitted
        # leg (a lens store): pinned lens by the record, EXPLICIT — the tool passes, whatever the enclosing job exported
        r = subprocess.run([sys.executable, str(m5t), "--stage", "pre", "--sha", SHA_ANY, "--only", GEMMA, ARDITI, PRIMARY],
                           capture_output=True, text=True, cwd=REPO, env=env)
        assert r.returncode == 0 and "VERDICT: PASS" in r.stdout and "SERVING_PATH: path=lens pinned_by_row=lens reason=store:lens(pins) selection=explicit" in r.stdout, (exported, r.stdout[-1500:], r.stderr[-800:])
    g8k = REPO / "scripts" / "sdfbase" / "gsm8k" / "g8k_dry_run.py"
    src = g8k.read_text()
    assert 'env.pop("F22_SERVING_PATH", None)' in src and 'env.pop("F22_SERVING_PATH", None)' in m5t.read_text()


# --- the kind-aware serving default: the single-cell launchers DECLARE their cell source from the ROW ------------------------------
# f22_init's contract: every caller declares how the serving path is selected — F22_SERVING_PATH (explicit, wins) >
# F22_LAUNCH_CELLS (a Bash array of HARNESS ids) > F22_LAUNCH_MANIFEST > deferred (F22_WAVE_ROOT) > REFUSED naming the caller. Earlier
# both launchers DEFERRED (they own F22_WAVE_ROOT) and the standup bound the path from the row's overlay manifest; now
# m5t_select_serving_path / g8k_select_serving_path declare F22_LAUNCH_CELLS = exactly the cells the standup serves (`--cell <row cell>`,
# `none` implicit -> cells_requested [none, <cell>]) in harness spelling, and keep the row's lens pin, which AUTO never overrides.

LORA = "constv2-mix-full__lora_r4_ce"                      # a natively servable kind on a launchable (gemma4e4b, thinking_card) ext1 row
LORA_HARNESS = "si_constv2_mix_full__lora_r4_ce"
COMMON_SH = REPO / "scripts" / "sdfbase" / "full22" / "common.sh"
M5T_SH = REPO / "scripts" / "sdfbase" / "math500" / "m5t_common.sh"
SELECT_REPORT = ('declare -p F22_LAUNCH_CELLS 2>/dev/null || echo "F22_LAUNCH_CELLS undeclared"; echo "PATH=${F22_SERVING_PATH:-unset}"; '
                 'echo "MODE=${F22_SERVING_PATH_MODE:-unset}"; echo "SELECTION=${F22_SERVING_PATH_SELECTION_JSON:-}"')


def _sh_funcs(path, *names):
    """The named shell functions' source, in file order (each ends at the first line that is exactly `}`)."""
    sh = path.read_text()
    out = []
    for name in names:
        start = sh.index(f"{name}(){{")
        out.append(sh[start:sh.index("\n}\n", start) + 3])
    return "\n".join(out)


def _m5t_select(body, **env):
    """What m5t_init runs BEFORE f22_init fixes the path — m5t_select_serving_path (the launcher's declaration from the row) — then
    common.sh's own f22_select_serving_path, under set -u with the real cells.py and selector on this tree (f22_init itself needs a
    cluster checkout; the selection functions are sliced the way tests/test_sdfbase_full22_wave.py slices them)."""
    script = ("set -u\nf22_log(){ echo \"LOG $*\"; }\nf22_fatal(){ echo \"FATAL $*\"; exit 1; }\n"
              + _sh_funcs(COMMON_SH, "f22_bind_engine", "f22_select_auto", "f22_select_serving_path")
              + "\n" + _sh_funcs(M5T_SH, "m5t_py", "m5t_select_serving_path")
              + f"\nREPO_DIR={REPO}\nM5T_DIR=scripts/sdfbase/math500\nF22_SELECT_PY={sys.executable}\nREDUCTION_MODE=stacked\n" + body)
    base = {k: v for k, v in os.environ.items()
            if k not in ("F22_SERVING_PATH", "F22_LAUNCH_CELLS", "F22_LAUNCH_MANIFEST", "F22_SERVING_PATH_DEFER", "F22_WAVE_ROOT", "CPY", "SLURM_ARRAY_TASK_ID")}
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={**base, **env}, cwd=REPO)


def _selection(out):
    line = [l for l in out.splitlines() if l.startswith("SELECTION=")][0]
    return json.loads(line[len("SELECTION="):])


def test_launch_cells_are_none_plus_the_rows_harness_cell_in_both_launchers():
    """cells.py / g8k_cells.py launch_cells_for: exactly the cells the standup's `--cell <row cell>` (`none` implicit) makes
    cells_requested read, in HARNESS spelling (the selector's grammar): `none` alone on the base row, `none` + the row's harness cell
    otherwise — on GSM8K the STANDUP spelling (the `@thinking` variant on a declared row). The CLI prints them one per line."""
    for r in CELLS.M5T_CELLS:
        assert CELLS.launch_cells_for(r) == (["none"] if r["harness_cell"] == "none" else ["none", r["harness_cell"]]), r["job_id"]
    steer = [r for r in CELLS.M5T_CELLS if r["cell"] == ALPHASTEER and r["model_key"] == GEMMA]
    assert len(steer) == 1 and CELLS.launch_cells_for(steer[0]) == ["none", "alphasteer"]
    assert CELLS.launch_cells_for(CELLS.row_for(GEMMA, LORA, PRIMARY)) == ["none", LORA_HARNESS]
    assert CELLS.launch_cells_for(CELLS.row_for(GEMMA, NONE, PRIMARY)) == ["none"]
    declared = 0
    for name in G8K.MATRICES:
        for r in G8K.matrix_rows(G8K.resolve_matrix(name)):
            cells = G8K.launch_cells_for(r)
            own = ["none"] if r["standup_harness_cell"] == "none" else ["none", r["standup_harness_cell"]]
            # a matrix that declares `standup_companions` (qwen_seedrep, qwen-base-seed-replicates O12a) registers each
            # companion's harness cell ONCE after the row's own, so every standup of the view composes one LoRA kernel
            # configuration; the driver still measures the row's cell alone (g8k_cells.launch_cells_for / row_companions).
            # Every matrix without companions reads [] here and keeps the exact two-element form.
            companions = [h for _cell, h, _sid, _pin in G8K.row_companions(r) if h not in own]
            assert cells == own + companions, (name, r["job_id"])
            if r["declared"]:
                declared += 1; assert cells[1].endswith("@thinking"), (name, r["job_id"], cells)
    assert declared >= 1
    r = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/math500/cells.py"), "row", "--key", GEMMA, "--cell", ALPHASTEER, "--sampling", PRIMARY, "--launch-cells"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0 and r.stdout == "none\nalphasteer\n", r.stdout + r.stderr
    r = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/math500/cells.py"), "row", "--key", GEMMA, "--cell", NONE, "--sampling", PRIMARY, "--launch-cells"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0 and r.stdout == "none\n", r.stdout + r.stderr
    r = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/gsm8k/g8k_cells.py"), "row", "--key", GEMMA, "--view", "main", "--cell", ALPHASTEER, "--launch-cells"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0 and r.stdout == "none\nalphasteer\n", r.stdout + r.stderr


def test_m5t_init_declares_f22_launch_cells_from_the_row_and_the_row_pin_wins_over_auto(tmp_path):
    """(a) a steer row declares F22_LAUNCH_CELLS = its cell + none AND pins the lens: f22_select_serving_path records EXPLICIT lens,
    nothing classified — the kind-aware AUTO never runs over a pinned row; (b) a LoRA row declares its cell and leaves the path to AUTO,
    which selects lora_native over exactly the declared cells (cells_source F22_LAUNCH_CELLS, the record the standup carries); the base
    row declares `none` alone and AUTO files the passthrough-only default. The ARRAY form declares the same from the array index. Since
    arditi-native-pin the declaration reads the row's STORE under F22_WAVE_ROOT (exported by m5t_init first; refused by name when unset):
    an Arditi row is unpinned with no store (AUTO native), pinned lens by a lens store (EXPLICIT), unpinned by a native store, and a
    submitter's lens export over a native store refuses like a conflicting export over a pin."""
    wave = tmp_path / "wave"; wave.mkdir()
    W = {"F22_WAVE_ROOT": str(wave)}
    r = _m5t_select(f"m5t_select_serving_path {GEMMA} {ALPHASTEER} {PRIMARY}\nf22_select_serving_path\n" + SELECT_REPORT, **W)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'declare -a F22_LAUNCH_CELLS=([0]="none" [1]="alphasteer")' in r.stdout and "PATH=lens" in r.stdout and "MODE=explicit" in r.stdout, r.stdout
    assert "launch cells declared: none alphasteer (F22_LAUNCH_CELLS)" in r.stdout and "[serving-path] EXPLICIT lens (F22_SERVING_PATH set by caller)" in r.stdout
    assert f"serving path: lens — pinned by the ROW (kind:alphasteer; wave root {wave}; pins checkout)" in r.stdout
    sel = _selection(r.stdout)
    assert sel["mode"] == "explicit" and sel["selected"] == "lens" and sel["cells_by_disposition"] is None and "cells_requested" not in sel
    r = _m5t_select(f"m5t_select_serving_path {GEMMA} {LORA} {PRIMARY}\nf22_select_serving_path\n" + SELECT_REPORT, **W)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f'declare -a F22_LAUNCH_CELLS=([0]="none" [1]="{LORA_HARNESS}")' in r.stdout and "PATH=lora_native" in r.stdout and "MODE=auto" in r.stdout, r.stdout
    assert "not pinned by the row (unpinned:not-external" in r.stdout and "the kind-aware default selects it at f22_init" in r.stdout and "[serving-path] AUTO-selected lora_native: 1 of 1 steered cells natively servable" in r.stdout
    sel = _selection(r.stdout)
    assert sel == {**sel, "mode": "auto", "selected": "lora_native", "cells_source": "F22_LAUNCH_CELLS", "spec_source": "kind_only",
                   "cells_requested": ["none", LORA_HARNESS], "cells_by_disposition": {"passthrough": ["none"], "serve": [LORA_HARNESS], "skip": []}}
    r = _m5t_select(f"m5t_select_serving_path {GEMMA} {NONE} {PRIMARY}\nf22_select_serving_path\n" + SELECT_REPORT, **W)
    assert r.returncode == 0 and 'declare -a F22_LAUNCH_CELLS=([0]="none")' in r.stdout and "PATH=lora_native" in r.stdout and "MODE=auto" in r.stdout, r.stdout + r.stderr
    sel = _selection(r.stdout)
    assert sel["cells_requested"] == ["none"] and sel["reason"].startswith("passthrough-only launch")
    # the pin wins over AUTO in BOTH directions: a submitter's explicit export on an unpinned row is EXPLICIT (never a conflict, never
    # classified); a submitter's export AGAINST a pinned row is refused by name before the selector is reached
    r = _m5t_select(f"m5t_select_serving_path {GEMMA} {LORA} {PRIMARY}\nf22_select_serving_path\n" + SELECT_REPORT, F22_SERVING_PATH="lens", **W)
    assert r.returncode == 0 and "PATH=lens" in r.stdout and "MODE=explicit" in r.stdout and "which wins as EXPLICIT" in r.stdout, r.stdout + r.stderr
    r = _m5t_select(f"m5t_select_serving_path {GEMMA} {ALPHASTEER} {PRIMARY}\nf22_select_serving_path\n" + SELECT_REPORT, F22_SERVING_PATH="lora_native", **W)
    assert r.returncode == 1 and "FATAL the row pins serving path 'lens' (kind:alphasteer" in r.stdout and "refused by name" in r.stdout and not [l for l in r.stdout.splitlines() if l.startswith("PATH=")], r.stdout + r.stderr
    # the ARRAY form: the same declaration from the array index (the production submission's grammar)
    steer = [x for x in CELLS.M5T_CELLS if x["cell"] == ALPHASTEER and x["model_key"] == GEMMA][0]
    r = _m5t_select("m5t_select_serving_path ARRAY\nf22_select_serving_path\n" + SELECT_REPORT, SLURM_ARRAY_TASK_ID=str(steer["index"]), **W)
    assert r.returncode == 0 and 'declare -a F22_LAUNCH_CELLS=([0]="none" [1]="alphasteer")' in r.stdout and "PATH=lens" in r.stdout and "MODE=explicit" in r.stdout, r.stdout + r.stderr
    # THE ARDITI ROW FOLLOWS ITS STORE: no wave root -> refused by name before any row is read
    r = _m5t_select(f"m5t_select_serving_path {GEMMA} {ARDITI} {PRIMARY}\nf22_select_serving_path\n" + SELECT_REPORT)
    assert r.returncode == 1 and "FATAL m5t_select_serving_path: F22_WAVE_ROOT is unset" in r.stdout and "FOLLOWS ITS STORE" in r.stdout, r.stdout + r.stderr
    arditi = _m5t_row(GEMMA, ARDITI); root = wave / arditi["out_root"]
    # the repo's record (the default PINS source) knows the real row's leg as a lens store: pinned lens with no disk store, EXPLICIT
    r = _m5t_select(f"m5t_select_serving_path {GEMMA} {ARDITI} {PRIMARY}\nf22_select_serving_path\n" + SELECT_REPORT, **W)
    assert r.returncode == 0 and "PATH=lens" in r.stdout and "MODE=explicit" in r.stdout and f"serving path: lens — pinned by the ROW (store:lens(pins); wave root {wave}; pins checkout)" in r.stdout, r.stdout + r.stderr
    # the disk alone, the pins silenced by a synthetic record through the test seam (M5T_STORE_PINS: honoured only in a local dry run, logged)
    blind = str(_pins_record(tmp_path / "blind.json")); WB = {**W, "M5T_STORE_PINS": blind, "M5T_DRY_RUN": "1"}
    # the seam REPLACES the admitted record, so a Slurm job that inherits it — or any caller outside an explicit
    # local dry run — refuses by name before any row is read: MATH-500 row 40 (llama Arditi: a lens store on record, disk absent here) never
    # reaches lora_native; the same job without the seam reads the record
    for env in ({"SLURM_JOB_ID": "123456", "M5T_DRY_RUN": "1"}, {"SLURM_JOB_ID": "123456"}, {}):
        r = _m5t_select("m5t_select_serving_path llama8b arditi card\nf22_select_serving_path\n" + SELECT_REPORT, M5T_STORE_PINS=blind, **W, **env)
        assert r.returncode == 1 and "FATAL M5T_STORE_PINS is set" in r.stdout and "refused by name" in r.stdout, (env, r.stdout + r.stderr)
        assert not [l for l in r.stdout.splitlines() if l.startswith("PATH=")] and "lora_native" not in r.stdout and "AUTO-selected" not in r.stdout, (env, r.stdout)
    r = _m5t_select("m5t_select_serving_path llama8b arditi card\nf22_select_serving_path\n" + SELECT_REPORT, SLURM_JOB_ID="123456", **W)
    assert r.returncode == 0 and "PATH=lens" in r.stdout and "MODE=explicit" in r.stdout and "(store:lens(pins)" in r.stdout, r.stdout + r.stderr
    r = _m5t_select(f"m5t_select_serving_path {GEMMA} {ARDITI} {PRIMARY}\nf22_select_serving_path\n" + SELECT_REPORT, **WB)        # (c) no store
    assert r.returncode == 0 and 'declare -a F22_LAUNCH_CELLS=([0]="none" [1]="arditi")' in r.stdout and "PATH=lora_native" in r.stdout and "MODE=auto" in r.stdout, r.stdout + r.stderr
    assert f"serving path: not pinned by the row (unpinned:no-store; wave root {wave}; pins {blind})" in r.stdout and "[serving-path] AUTO-selected lora_native: 1 of 1 steered cells natively servable; skipped by name: none" in r.stdout
    assert _selection(r.stdout)["cells_by_disposition"] == {"passthrough": ["none"], "serve": ["arditi"], "skip": []}
    _steer_store(root, "lens")                                                                                                       # (b) a lens store
    r = _m5t_select(f"m5t_select_serving_path {GEMMA} {ARDITI} {PRIMARY}\nf22_select_serving_path\n" + SELECT_REPORT, **WB)
    assert r.returncode == 0 and "PATH=lens" in r.stdout and "MODE=explicit" in r.stdout and f"serving path: lens — pinned by the ROW (store:lens(disk); wave root {wave}" in r.stdout, r.stdout + r.stderr
    assert _selection(r.stdout)["cells_by_disposition"] is None
    r = _m5t_select(f"m5t_select_serving_path {GEMMA} {ARDITI} {PRIMARY}\nf22_select_serving_path\n" + SELECT_REPORT, **W)         # disk + the record
    assert r.returncode == 0 and "PATH=lens" in r.stdout and "(store:lens(disk+pins)" in r.stdout, r.stdout + r.stderr
    r = _m5t_select(f"m5t_select_serving_path {GEMMA} {ARDITI} {PRIMARY}\nf22_select_serving_path\n" + SELECT_REPORT, F22_SERVING_PATH="lora_native", **WB)
    assert r.returncode == 1 and "FATAL the row pins serving path 'lens' (store:lens(disk)" in r.stdout and "refused by name" in r.stdout, r.stdout + r.stderr
    r = _m5t_select("m5t_select_serving_path ARRAY\nf22_select_serving_path\n" + SELECT_REPORT, SLURM_ARRAY_TASK_ID=str(arditi["index"]), **WB)
    assert r.returncode == 0 and "PATH=lens" in r.stdout and "MODE=explicit" in r.stdout and "(store:lens(disk)" in r.stdout, r.stdout + r.stderr
    shutil.rmtree(root); _steer_store(root, "native")                                                                                  # (d) a native store
    r = _m5t_select(f"m5t_select_serving_path {GEMMA} {ARDITI} {PRIMARY}\nf22_select_serving_path\n" + SELECT_REPORT, **WB)
    assert r.returncode == 0 and "PATH=lora_native" in r.stdout and "MODE=auto" in r.stdout and f"not pinned by the row (store:native(disk); wave root {wave}" in r.stdout, r.stdout + r.stderr
    r = _m5t_select(f"m5t_select_serving_path {GEMMA} {ARDITI} {PRIMARY}\nf22_select_serving_path\n" + SELECT_REPORT, F22_SERVING_PATH="lora_native", **WB)
    assert r.returncode == 0 and "PATH=lora_native" in r.stdout and "MODE=explicit" in r.stdout, r.stdout + r.stderr                # agreeing export: fine
    r = _m5t_select(f"m5t_select_serving_path {GEMMA} {ARDITI} {PRIMARY}\nf22_select_serving_path\n" + SELECT_REPORT, F22_SERVING_PATH="lens", **WB)
    assert r.returncode == 1 and "FATAL the row's steer store on record is NATIVE-stamped (store:native(disk)" in r.stdout and "refused by name" in r.stdout and not [l for l in r.stdout.splitlines() if l.startswith("PATH=")], r.stdout + r.stderr
    r = _m5t_select(f"m5t_select_serving_path {GEMMA} {ARDITI} {PRIMARY}\nf22_select_serving_path\n" + SELECT_REPORT, **W)         # native disk vs the record's lens
    assert r.returncode == 1 and "BOTH engines' stamps" in r.stdout + r.stderr and "pins: ['lens']" in r.stdout + r.stderr, r.stdout + r.stderr
    (root / "evidence.json").write_text(json.dumps({"serving_identity": LENS_IDENTITY}))                                                # (e) mixed
    r = _m5t_select(f"m5t_select_serving_path {GEMMA} {ARDITI} {PRIMARY}\nf22_select_serving_path\n" + SELECT_REPORT, **WB)
    assert r.returncode == 1 and "BOTH engines' stamps" in r.stdout + r.stderr and "FATAL cells.py:" in r.stdout and "refused the pin" in r.stdout, r.stdout + r.stderr


def test_known_bad_an_init_with_no_declaration_and_no_manifest_is_refused_by_name_and_a_rowless_init_still_defers():
    """The KNOWN-BAD (the refusal itself, through the launcher's slice): no m5t_select_serving_path, no manifest, no wave root ->
    f22_select_serving_path refuses naming the caller and every way to declare. The launcher's row-less callers (the smoke, the Nemotron
    probe, the prod dry run) declare nothing on purpose and DEFER through the wave root they own — pinned in m5t_init's text: the
    declaration runs before the root is exported, both before f22_init."""
    r = _m5t_select("f22_select_serving_path\n" + SELECT_REPORT)
    assert r.returncode == 1 and "FATAL serving-path selection: bash declared no cell source" in r.stdout and not [l for l in r.stdout.splitlines() if l.startswith("PATH=")], r.stdout + r.stderr
    for needle in ("F22_LAUNCH_CELLS=(<harness ids>; empty = this job stands up no target)", "F22_LAUNCH_MANIFEST=<the launch's manifest>",
                   "F22_SERVING_PATH_DEFER=1", "F22_WAVE_ROOT", "set F22_SERVING_PATH explicitly"):
        assert needle in r.stdout, needle
    r = _m5t_select("export F22_WAVE_ROOT=/dev/null/owned-root\nf22_select_serving_path\n" + SELECT_REPORT)
    assert r.returncode == 0 and "MODE=deferred" in r.stdout and "F22_LAUNCH_CELLS undeclared" in r.stdout and "DEFERRED to the standup: this job owns its wave root" in r.stdout, r.stdout + r.stderr
    src = M5T_SH.read_text()
    init = src[src.index("m5t_init(){"):src.index("m5t_row(){")]
    # arditi-native-pin: the wave root is exported BEFORE the declaration, which reads the row's store under it (--wave-root)
    assert init.index("export F22_WAVE_ROOT=") < init.index('m5t_select_serving_path "$@"') < init.index('f22_init "$sha"')
    assert '[ $# -ge 1 ] && m5t_select_serving_path "$@"' in init
    select = src[src.index("m5t_select_serving_path(){"):src.index("m5t_init(){")]
    assert select.index('root="${F22_WAVE_ROOT:-}"') < select.index('--serving-path --wave-root "$root" ${pins_args[@]+"${pins_args[@]}"})') < select.index("--launch-cells)") < select.index("F22_LAUNCH_CELLS=()") < select.index('export F22_SERVING_PATH="$pin"')
    assert select.count("--wave-root") == 2 and "M5T_SERVING_PATH_REASON=" in select and "reason=%s" in src[src.index("m5t_print_plan(){"):]
    for name in ("smoke_thinking.sbatch", "nemo_backend_probe.sbatch", "dry_run_prod.sbatch"):
        text = (REPO / "scripts" / "sdfbase" / "math500" / name).read_text()
        assert 'm5t_init "$WAVE_SHA"\n' in text and "F22_LAUNCH_CELLS" not in text, name


def test_the_dry_run_declares_the_launch_cells_and_prints_the_selected_path_not_a_retyped_default():
    """The dry-run branch runs the SAME f22_select_serving_path over the same declaration, so its SERVING_PATH: line is the path
    f22_init fixes (selection=explicit on the row's pin, =auto otherwise) and LAUNCH_CELLS: the declared cells — both launchers."""
    r = _m5t_dry({}, GEMMA, ALPHASTEER, PRIMARY, SHA_ANY, "1")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-800:]
    assert "SERVING_PATH: path=lens pinned_by_row=lens reason=kind:alphasteer selection=explicit" in r.stdout and "LAUNCH_CELLS: source=F22_LAUNCH_CELLS cells=none alphasteer\n" in r.stdout
    assert "[serving-path] EXPLICIT lens" in r.stdout and "STANDUP: " in r.stdout and " --cell alphasteer " in r.stdout
    r = _m5t_dry({}, GEMMA, LORA, PRIMARY, SHA_ANY, "1")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-800:]
    assert "SERVING_PATH: path=lora_native pinned_by_row=none reason=unpinned:not-external selection=auto" in r.stdout and f"LAUNCH_CELLS: source=F22_LAUNCH_CELLS cells=none {LORA_HARNESS}\n" in r.stdout
    assert "[serving-path] AUTO-selected lora_native: 1 of 1 steered cells natively servable" in r.stdout
    r = _m5t_dry({}, GEMMA, NONE, PRIMARY, SHA_ANY, "0")
    assert r.returncode == 0 and "SERVING_PATH: path=lora_native pinned_by_row=none reason=unpinned:not-external selection=auto" in r.stdout and "LAUNCH_CELLS: source=F22_LAUNCH_CELLS cells=none\n" in r.stdout, r.stdout[-1500:]
    assert "AUTO-selected lora_native: passthrough-only launch" in r.stdout
    r = _m5t_dry({"F22_SERVING_PATH": "lens"}, GEMMA, NONE, PRIMARY, SHA_ANY, "0")
    assert r.returncode == 0 and "SERVING_PATH: path=lens pinned_by_row=none reason=unpinned:not-external selection=explicit" in r.stdout and "LAUNCH_CELLS: source=F22_LAUNCH_CELLS cells=none\n" in r.stdout, r.stdout[-1500:]
    r = _m5t_dry({}, GEMMA, ARDITI, PRIMARY, SHA_ANY, "1")                    # the Arditi row under a fresh stub root: lens BY THE RECORD (no disk store)
    assert r.returncode == 0 and "SERVING_PATH: path=lens pinned_by_row=lens reason=store:lens(pins) selection=explicit" in r.stdout and "LAUNCH_CELLS: source=F22_LAUNCH_CELLS cells=none arditi\n" in r.stdout, r.stdout[-1500:] + r.stderr[-500:]
    assert "STANDUP: " in r.stdout and " --cell arditi " in r.stdout
    r = _g8k_dry({}, GEMMA, "main", ALPHASTEER, "8192", SHA_ANY, "1", "0")
    assert r.returncode == 0 and "SERVING_PATH: path=lens pinned_by_row=lens reason=kind:alphasteer selection=explicit" in r.stdout and "LAUNCH_CELLS: source=F22_LAUNCH_CELLS cells=none alphasteer\n" in r.stdout, r.stdout[-1500:] + r.stderr[-500:]
    r = _g8k_dry({}, GEMMA, "main", NONE, "8192", SHA_ANY, "0", "0")
    assert r.returncode == 0 and "SERVING_PATH: path=lora_native pinned_by_row=none reason=unpinned:not-external selection=auto" in r.stdout and "LAUNCH_CELLS: source=F22_LAUNCH_CELLS cells=none\n" in r.stdout, r.stdout[-1500:]
    r = _g8k_dry({}, GEMMA, "main", ARDITI, "8192", SHA_ANY, "1", "0")
    assert r.returncode == 0 and "SERVING_PATH: path=lens pinned_by_row=lens reason=store:lens(pins) selection=explicit" in r.stdout and "LAUNCH_CELLS: source=F22_LAUNCH_CELLS cells=none arditi\n" in r.stdout, r.stdout[-1500:] + r.stderr[-500:]
    # the dry-run branches set F22_WAVE_ROOT from their stub root before the declaration reads the store under it
    for rel in ("math500/math500_thinking.sbatch", "gsm8k/gsm8k_thinking.sbatch"):
        text = (REPO / "scripts" / "sdfbase" / rel).read_text()
        dry = text[text.index("_DRY_RUN:-0}\" = 1 ]; then"):text.index("  exit 0\nfi\n")]
        assert dry.index('F22_WAVE_ROOT="$WAVE_ROOT"') < dry.index("_select_serving_path \"${ROW_ARGS[@]}\""), rel
    # the dry-run branches run the selector on python3 as f22_init does — never on $CPY, a placeholder path in the launcher tests
    for rel in ("math500/math500_thinking.sbatch", "gsm8k/gsm8k_thinking.sbatch"):
        text = (REPO / "scripts" / "sdfbase" / rel).read_text()
        dry = text[text.index("_DRY_RUN:-0}\" = 1 ]; then"):text.index("  exit 0\nfi\n")]
        assert dry.count("f22_select_serving_path\n") == 1 and "F22_SELECT_PY=" not in dry and dry.index("_select_serving_path \"${ROW_ARGS[@]}\"") < dry.index("\n  f22_select_serving_path\n") < dry.index("_row \"${ROW_ARGS[@]}\""), rel
