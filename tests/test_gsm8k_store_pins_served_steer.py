"""math-eight-arms-extension — the GSM8K analysis admits the evidence that EXISTS, never a widened gate (the E3 follow-up
brief's item 6 and the P5 multi-tree read), over the SYNTHETIC stores of ``tests/gsm8k_synthetic_store.py``:

* ``--store-pins`` (the IFEval battery's per-row admission): every leg admitted by ITS OWN producer tree; a leg whose tree is
  not its admitted pin refuses by name, a leg without an entry refuses by name, the loader refuses an off-contract doc, and
  the report files the doc and every leg's admitted pin; the overlap record binds to the models' ``none`` legs' pin;
* the SERVED-STEER stamp class, selected by the leg's OWN ``steering_bites.liveness_gate == logprob`` (never by cell name):
  the (request-unique, never, <render>) triple derived through the contract rule, a conclusive logprob verdict and a
  lens-on-bite record that is a token bite or the class-E logprob admission — tested against KNOWN-BAD inputs: a served-steer
  leg whose ``n_conclusive`` is below the record's minimum, and a text-class leg wearing request-unique stamps, both refused
  by name;
* the matrix-UNION read (repeatable ``--matrix``): a later matrix's LAUNCH row replaces an earlier HELD row of the same cell;
  any other duplicate refuses by name.

Every store here is synthetic (placeholder shas, invented ids); nothing is real recorded data."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from sisteer.capeval.serving import proxy as PX
from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
import gsm8k_synthetic_store as G  # noqa: E402


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load("tinycap_gsm8k_paired200", REPO / "scripts" / "tinycap_gsm8k_paired200.py")
V = A.V
M = A.M

T1, T2, T3 = G.FULL_SHA, "8" * 40, "9" * 40
PIN = G.SHA7
GEMMA, QWEN = "gemma4e4b", "qwen35_4b"
NONE, CONST_PROMPT = "none", "prompted__si_constitution"
B_FAST = 50


def analyze(gen_root, sha=PIN, *, matrix, **kw):
    kw.setdefault("canonical_ids", G.IDS)
    kw.setdefault("sampling_rows", G.SAMPLING_ROWS)
    kw.setdefault("render_facts", G.RENDER_FACTS)
    kw.setdefault("pins", G.pins_doc(G.matrix_rows()))
    kw.setdefault("n_resamples", B_FAST)
    kw.setdefault("read_eval_logs", False)
    kw.setdefault("analysis_sha", G.FULL_SHA)
    return A.analyze(gen_root, sha, matrix=matrix, **kw)


def _small(rows, *triples):
    keep = set(triples)
    return [r for r in rows if (r["model_key"], r["view"], r["cell"]) in keep]


@pytest.fixture(scope="module")
def two_tree_store(tmp_path_factory):
    """Two wave-isolated pulls: gemma's published ``main`` none at T1 under one root; gemma's extension prompted-constitution
    cell at T2 under another — the smallest cross-tree contrast."""
    root = tmp_path_factory.mktemp("g8k_pins")
    rows = G.eight_arm_matrix(with_steer=False)
    published = _small(rows, (GEMMA, "main", NONE))
    ext = _small(rows, (GEMMA, "main", CONST_PROMPT))
    a, b = root / "pull_published", root / "pull_ext"
    G.build_rows(a, published, producer_fields=True)
    G.build_rows(b, ext, producer_fields=True, tree_sha=T2)
    return {"root": root, "a": a, "b": b, "rows": published + ext}


def _key(model, view, cell, cap=G.CAP_BASE):
    return V.leg_key(model, view, cell, cap)


# --------------------------------------------------------------------------- #
# --store-pins                                                                 #
# --------------------------------------------------------------------------- #


def test_store_pins_admit_each_leg_by_its_own_tree_and_the_cross_tree_contrast_computes(two_tree_store):
    s = two_tree_store
    pins = {_key(GEMMA, "main", NONE): T1, _key(GEMMA, "main", CONST_PROMPT): T2}
    rep = analyze([str(s["a"]), str(s["b"])], matrix=s["rows"], overlap=None, waves=["published", G.WAVE_EXT], store_pins=pins)
    none_leg, ext_leg = rep["legs"][_key(GEMMA, "main", NONE)], rep["legs"][_key(GEMMA, "main", CONST_PROMPT)]
    assert A.usable(none_leg) and A.usable(ext_leg), (none_leg.get("refused"), ext_leg.get("refused"))
    assert none_leg["producer"]["admitted_pin"] == T1 and ext_leg["producer"]["admitted_pin"] == T2
    assert none_leg["producer"]["admission"] == "per-row pins" and none_leg["producer"]["matches_pin"] and ext_leg["producer"]["matches_pin"]
    assert rep["gates"]["A0_completeness"]["status"] == A.GATE_PASS and rep["unusable_legs"] == {}
    assert rep[A.STORE_PINS_KEY]["pins"] == pins and set(rep[A.STORE_PINS_KEY]["trees"]) == {T1, T2}
    assert rep[A.STORE_PINS_KEY]["legs"][_key(GEMMA, "main", CONST_PROMPT)] == {"admitted_pin": T2, "tree_sha": T2, "usable": True}
    c = next(c for c in rep["contrasts"] if c["cell"] == CONST_PROMPT)
    assert c["computable"], c.get("reason")
    assert c["compatibility"]["cross_tree"] and c["compatibility"]["declared"] and c["compatibility"]["rule"] == A.COMPAT_RULE_DECLARED
    assert rep["gates"]["A7_population"]["status"] == A.GATE_PASS
    # A10 without a record still refuses (the record is an operator input); its binding pin under pins is the none legs' tree
    assert rep["gates"]["A10_overlap"]["status"] == A.GATE_FAIL and "no overlap record" in rep["gates"]["A10_overlap"]["detail"]
    assert A.overlap_pin_under_store_pins([GEMMA], pins) == T1
    assert rep["gen_roots"] == [{"path": str(s["a"]), "pin": PIN, "pinned": False}, {"path": str(s["b"]), "pin": PIN, "pinned": False}]


def test_store_pins_refuse_another_tree_and_a_missing_entry_by_name(two_tree_store):
    s = two_tree_store
    wrong = {_key(GEMMA, "main", NONE): T1, _key(GEMMA, "main", CONST_PROMPT): T3}
    rep = analyze([str(s["a"]), str(s["b"])], matrix=s["rows"], overlap=None, waves=["published", G.WAVE_EXT], store_pins=wrong)
    leg = rep["legs"][_key(GEMMA, "main", CONST_PROMPT)]
    # the row's pin replaces the root's, so the producer check refuses the store with the wave-pin sentence naming the admitted tree
    assert not A.usable(leg) and ("is not the admitted pin" in leg["refused"] or "does not carry the wave pin" in leg["refused"])
    assert T3[:8] in leg["refused"] and leg["admitted_pin"] == T3
    assert rep["gates"]["A0_completeness"]["status"] == A.GATE_FAIL and _key(GEMMA, "main", CONST_PROMPT) in rep["unusable_legs"]
    missing = {_key(GEMMA, "main", NONE): T1}
    rep2 = analyze([str(s["a"]), str(s["b"])], matrix=s["rows"], overlap=None, waves=["published", G.WAVE_EXT], store_pins=missing)
    leg2 = rep2["legs"][_key(GEMMA, "main", CONST_PROMPT)]
    assert not A.usable(leg2) and "no admitted pin" in leg2["refused"]
    # without a pins doc the published one-pin rule stands: the leg is a store from another tree
    rep3 = analyze([str(s["a"]), str(s["b"])], matrix=s["rows"], overlap=None, waves=["published", G.WAVE_EXT])
    assert "does not carry the wave pin" in rep3["legs"][_key(GEMMA, "main", CONST_PROMPT)]["refused"]


def test_store_pins_loader_and_overlap_binding_refuse_off_contract_inputs(tmp_path):
    with pytest.raises(A.AnalysisError, match="not a tree sha"):
        A.load_store_pins({"a/b/c@8192": "zz"})
    with pytest.raises(A.AnalysisError, match="not a leg key"):
        A.load_store_pins({"abc@8192": T1})
    with pytest.raises(A.AnalysisError, match="pinned twice with different trees"):
        A.load_store_pins([{"a/b/c@8192": T1}, {"a/b/c@8192": T2}])
    # two spellings of ONE tree are one pin; documentation keys ride along unread; several files merge
    p1, p2 = tmp_path / "p1.json", tmp_path / "p2.json"
    p1.write_text(json.dumps({"_meta": {"note": "x"}, "a/b/c@8192": T1[:7].upper()}))
    p2.write_text(json.dumps({"a/b/c@8192": T1, "a/b/d@8192": T2}))
    assert A.load_store_pins([p1, p2]) == {"a/b/c@8192": T1, "a/b/d@8192": T2}
    assert A.load_store_pins(None) is None
    assert A.overlap_pin_under_store_pins([GEMMA, QWEN], {_key(GEMMA, "main", NONE): T1, _key(QWEN, "it", NONE): T1}) == T1
    with pytest.raises(A.AnalysisError, match="different trees"):
        A.overlap_pin_under_store_pins([GEMMA, QWEN], {_key(GEMMA, "main", NONE): T1, _key(QWEN, "main", NONE): T2})
    with pytest.raises(A.AnalysisError, match="names no"):
        A.overlap_pin_under_store_pins([GEMMA], {_key(QWEN, "main", NONE): T1})
    with pytest.raises(A.AnalysisError, match="two admissions"):
        analyze(str(tmp_path), matrix=G.matrix_rows(), store_pins={"a/b/c@8192": T1}, tree_set={"also": [T2], "allow": ["*"], "why": "x"},
                repo_root=tmp_path)


# --------------------------------------------------------------------------- #
# the served-steer stamp class                                                 #
# --------------------------------------------------------------------------- #


def _logprob_bites(cell: str, *, n_conclusive: int = 5, n_min: int = 3) -> dict:
    """A steering-bites block whose LOGPROB statistic ruled (the served-steer route's record): the text count reported beside."""
    rec = G.S.steering_bites_record(cell, n_differing=1)
    stamps = dict(zip(C.EXPECTED_STAMP_TRIPLE_FIELDS, V.served_steer_stamps(C.THINKING_ON)))
    rec["stamps_by_leg"] = {"steered": [dict(stamps) for _ in range(rec["n_probes"])], "scale0": [dict(stamps) for _ in range(rec["n_probes"])]}
    rec[C.OOS_EVIDENCE_BITES_LIVENESS_GATE] = C.LENS_LIVENESS_LOGPROB
    rec["object_kind"] = "alphasteer"
    rec[C.OOS_EVIDENCE_BITES_TEXT_DIVERGENCE] = {"verified": True, "n_probes": rec["n_probes"], "n_differing": 1, "error": None,
                                                "role": "reported beside the logprob statistic"}
    rec[C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS] = {
        "applicable": True, "verified": True, "kind": "alphasteer", "twin": C.TWIN_CELL_ID, "n_probes": rec["n_probes"],
        "n_conclusive": n_conclusive, "min_conclusive_probes": n_min, "min_common_tokens": 2, "top_logprobs": 20, "max_tokens": 1,
        "diag": {"n_items": rec["n_probes"], "steered_divergence_median": 0.32, "noise_ceiling": 0.0, "min_ratio": 2.0, "abs_floor": 0.001, "bar": 0.001},
        "noise_floor": {"measured": 0.0, "how": "the scale-0 twin against itself", "min_ratio": 2.0, "abs_floor": 0.001, "bar": 0.001, "binding": "abs_floor"},
        "divergence": {"steered_vs_scale0_median": 0.32, "clears_bar": True},
        "not_a_dose_axis": C.LOGPROB_LIVENESS_NOT_A_DOSE_AXIS}
    return rec


def _served_steer_endpoint(row):
    ep = G.standup_endpoint(row)
    spell = C.cell_spellings(str(row["cell"]), declared=False)
    ep[C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS] = {spell.harness: PX.RESUMED_PREFILL_NEVER}
    return ep


def _served_steer_edits(row, *, n_conclusive: int = 5):
    """The edits that turn the fixture's fitted (always / cell-stable) leg into the served-steer route's record."""
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
        # the fixture stamps the all-position tuple; a steer row runs the steer tuple (contracts.g8k_steer_tuple_name)
        manifest["oos_experiment"] = str(row["tuple_name"])

    def evidence_edit(ev):
        ev[C.OOS_EVIDENCE_BITES_BLOCK] = _logprob_bites(str(row["cell"]), n_conclusive=n_conclusive)
        ev[G.S.EVIDENCE_LENS_POSITION] = C.lens_position_attested(_served_steer_endpoint(row), cell=str(row["cell"]),
                                                                 expected_sha256=str(row["object_sha256"]), model_key=str(row["model_key"]),
                                                                 arm=C.THINKING_ON)

    return {"summary_edit": summary_edit, "ledger_edit": ledger_edit, "evidence_edit": evidence_edit, "manifest_edit": manifest_edit}


@pytest.fixture(scope="module")
def steer_rows():
    rows = G.eight_arm_matrix(with_steer=True)
    steer = [r for r in rows if r.get("wave") == G.WAVE_STEER and r["model_key"] == GEMMA and r["cell"] == "alphasteer"]
    assert steer, "the fixture's steer stub for gemma alphasteer"
    row = {**steer[0], "status": G.STATUS_LAUNCH, "object_sha256": G.object_pin(GEMMA, "main", "alphasteer"), "held_reason": None,
           "deferred_reason": None, "staged_id": G.staged_id_of("alphasteer", "main"), "tuple_name": C.g8k_steer_tuple_name(GEMMA, G.CAP_BASE)}
    return _small(rows, (GEMMA, "main", NONE)) + [row]


def _pins_for(rows):
    return {_key(r["model_key"], r["view"], r["cell"]): T1 for r in rows}


def test_served_steer_class_is_selected_by_the_legs_own_liveness_gate_and_passes_a2_a3(tmp_path, steer_rows):
    row = steer_rows[-1]
    root = tmp_path / "pull"
    G.build_rows(root, steer_rows[:-1], producer_fields=True)
    G.build_store(root, row, correct=set(range(190)), producer_fields=True, **_served_steer_edits(row))
    rep = analyze(str(root), matrix=steer_rows, overlap=None, waves=["published", G.WAVE_STEER], store_pins=_pins_for(steer_rows))
    leg = rep["legs"][_key(GEMMA, "main", "alphasteer")]
    assert A.usable(leg), leg.get("refused")
    assert leg["stamp_class"]["class"] == V.STAMP_CLASS_SERVED_STEER and leg["stamp_class"]["stamps"] == list(V.served_steer_stamps(C.THINKING_ON))
    assert leg["stamp_class"]["selected_by"] == V.STAMP_CLASS_SERVED_STEER_SELECTOR
    assert leg["stamps"]["passed"], leg["stamps"]["failures"]
    assert leg["certification"]["passed"] if "passed" in leg["certification"] else not leg["certification"].get("failures"), leg["certification"]
    a2, a3 = rep["gates"]["A2_identity"], rep["gates"]["A3_certification"]
    assert a3["status"] == A.GATE_PASS, a3
    # A2 on this leg: the STAMP triple passes under the class; the fixture writes no `logprob_bites` probe records (the real
    # served-steer stores carry them and pass A2 whole), so the only A2 lines this synthetic leg may draw name that shortfall
    a2_lines = [f for f in (a2.get("detail") or "").split("; ") if f.startswith(_key(GEMMA, "main", "alphasteer"))]
    assert all("logprob_bites" in f for f in a2_lines) and not any("stamps:" in f for f in a2_lines), a2_lines
    # the none leg keeps the matrix row's class
    assert rep["legs"][_key(GEMMA, "main", NONE)]["stamp_class"]["class"] == V.STAMP_CLASS_ROW


def test_served_steer_known_bad_inputs_still_refuse_by_name(tmp_path, steer_rows):
    row = steer_rows[-1]
    # (a) a served-steer leg whose logprob verdict is NOT conclusive (n_conclusive below the record's own minimum)
    root = tmp_path / "pull_inconclusive"
    G.build_rows(root, steer_rows[:-1], producer_fields=True)
    G.build_store(root, row, correct=set(range(190)), producer_fields=True, **_served_steer_edits(row, n_conclusive=2))
    rep = analyze(str(root), matrix=steer_rows, overlap=None, waves=["published", G.WAVE_STEER], store_pins=_pins_for(steer_rows))
    leg = rep["legs"][_key(GEMMA, "main", "alphasteer")]
    text = json.dumps(leg.get("certification")) + str(leg.get("refused"))
    assert "n_conclusive 2 < min_conclusive_probes 3" in text, text[:600]
    assert rep["gates"]["A3_certification"]["status"] == A.GATE_FAIL
    # (b) a TEXT-class leg (no logprob gate on its record) wearing the served-steer stamps: held to the ROW's class, refused by name
    root2 = tmp_path / "pull_text_class"
    G.build_rows(root2, steer_rows[:-1], producer_fields=True)
    want = dict(zip(C.EXPECTED_STAMP_TRIPLE_FIELDS, V.served_steer_stamps(C.THINKING_ON)))

    def summary_edit(summary):
        for f, v in want.items():
            summary[f] = v

    def manifest_edit(manifest):
        manifest["oos_experiment"] = str(row["tuple_name"])

    G.build_store(root2, row, correct=set(range(190)), producer_fields=True, summary_edit=summary_edit, manifest_edit=manifest_edit)
    rep2 = analyze(str(root2), matrix=steer_rows, overlap=None, waves=["published", G.WAVE_STEER], store_pins=_pins_for(steer_rows))
    leg2 = rep2["legs"][_key(GEMMA, "main", "alphasteer")]
    assert leg2["stamp_class"]["class"] == V.STAMP_CLASS_ROW
    assert not leg2["stamps"]["passed"] and any("'request-unique' != expected 'cell-stable'" in f for f in leg2["stamps"]["failures"]), leg2["stamps"]
    assert rep2["gates"]["A2_identity"]["status"] == A.GATE_FAIL
    # the selector on a non-fitted row is another producer's record: refused by name
    out = V.leg_expected_stamps({"fitted": False, "declared": False, "expected_stamps": ["cell-stable", "not_consulted", "thinking_on"]},
                                {C.OOS_EVIDENCE_BITES_BLOCK: {C.OOS_EVIDENCE_BITES_LIVENESS_GATE: C.LENS_LIVENESS_LOGPROB}}, arm=C.THINKING_ON)
    assert out["problem"] and "non-fitted" in out["problem"]
    # the class's own requirements, on records alone
    ok = {C.OOS_EVIDENCE_BITES_BLOCK: {C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS: {"n_conclusive": 5, "min_conclusive_probes": 3}},
          M.EVIDENCE_LENS_ON_BITE: {"differs_from_base": False, C.LENS_ON_BITE_LIVENESS_EVIDENCE_KEY: C.LENS_ON_BITE_EVIDENCE_LOGPROB}}
    assert V.served_steer_problems(ok) == []
    bad_bite = {**ok, M.EVIDENCE_LENS_ON_BITE: {"differs_from_base": False}}
    assert any("neither a token bite nor the logprob-admitted record" in p for p in V.served_steer_problems(bad_bite))
    assert V.served_steer_stamps(C.THINKING_NOT_AVAILABLE) == tuple(C.expected_stamp_regimes(lens=True, arm=C.THINKING_NOT_AVAILABLE,
                                                                                               policy=PX.RESUMED_PREFILL_NEVER))


# --------------------------------------------------------------------------- #
# the matrix-union read                                                        #
# --------------------------------------------------------------------------- #


def test_matrix_union_replaces_a_held_row_with_a_later_launch_row_and_refuses_other_duplicates(tmp_path):
    rows = G.matrix_rows()
    held = next(r for r in rows if r["status"] == G.STATUS_HELD)
    key = (held["model_key"], held["view"], held["cell"])
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    launched = {**held, "status": G.STATUS_LAUNCH, "held_reason": None}
    b.write_text(json.dumps({"schema": "g8k_matrix/2", "wave_root_rel": "exp/x", "rows": [launched]}), encoding="utf-8")
    union, prov = A.load_matrix_union([a, b])
    assert len(union) == len(rows)
    got = next(r for r in union if (r["model_key"], r["view"], r["cell"]) == key)
    assert got["status"] == G.STATUS_LAUNCH
    assert [x["cell_key"] for x in prov["replaced"]] == [V.cell_key(*key)] and prov["replaced"][0]["launched_in"] == str(b)
    assert [d["n_rows"] for d in prov["documents"]] == [len(rows), 1] and prov["documents"][1]["schema"] == "g8k_matrix/2"
    mx = V.load_matrix(union)                    # the union validates row by row like any in-memory matrix
    assert mx["counts"]["rows"] == len(rows)
    # a LAUNCH row named twice, or a hold after evidence, refuses by name
    c = tmp_path / "c.json"
    c.write_text(json.dumps({"rows": [next(r for r in rows if r["status"] == G.STATUS_LAUNCH)]}), encoding="utf-8")
    with pytest.raises(A.AnalysisError, match="only a LAUNCH row replacing a HELD row is admitted"):
        A.load_matrix_union([a, c])
    d = tmp_path / "d.json"
    d.write_text(json.dumps({"rows": [held]}), encoding="utf-8")
    with pytest.raises(A.AnalysisError, match="only a LAUNCH row replacing a HELD row is admitted"):
        A.load_matrix_union([b, d])
    with pytest.raises(A.AnalysisError, match="no such file"):
        A.load_matrix_union([tmp_path / "missing.json"])


# --------------------------------------------------------------------------- #
# the committed per-matrix records: the all-positions arm's                     #
# --------------------------------------------------------------------------- #


def test_the_arditi_allpos_record_admits_both_rung_legs_at_their_launch_tree_and_refuses_the_arditi_twin():
    """allpos-store-pins (2026-09-17): `configs/sdfbase/gsm8k/store_pins_arditi_allpos.json` is the all-positions matrix's admitted
    record — read TOGETHER with store_pins_default.json in the matrix-union read (`--matrix matrix.json --matrix matrix_arditi_allpos.json
    --store-pins store_pins_default.json --store-pins store_pins_arditi_allpos.json`, the it-doc shape). Through the analysis' own loader:
    the two files merge with no key in common; the record's keys are EXACTLY the row's two leg keys as the launcher derives them
    (`g8k_cells.leg_keys`: the `@8192` leg and its `@16384` rung), both at ONE 40-hex tree, the `_meta` naming that tree, the two jobs and
    the standup records; each leg's producer tree at that pin is admitted. KNOWN-BAD twins, each refused by name: the `arditi` spelling on
    the same row (a different intervention — NOT admitted here, and its published pin is another tree), the main-view spelling, a rung the
    ladder does not have, and the `arditi` row's own tree offered as this leg's producer. The launcher names the record last in
    STORE_PINS_PATHS, and the stamp script's per-matrix derivation (`store_pins_<matrix_name_of>.json`) lands on this file."""
    G8K = _load("g8k_cells_for_g8k_pins_test", REPO / "scripts" / "sdfbase" / "gsm8k" / "g8k_cells.py")
    default, allpos = REPO / "configs/sdfbase/gsm8k/store_pins_default.json", REPO / "configs/sdfbase/gsm8k/store_pins_arditi_allpos.json"
    row = G8K.matrix_rows(G8K.resolve_matrix("arditi_allpos"))[0]
    keys = G8K.leg_keys(row)
    assert keys == ["qwen35_4b/it/arditi_allpos@8192", "qwen35_4b/it/arditi_allpos@16384"] and row["cell"] == "arditi_allpos"
    doc = json.loads(allpos.read_text(encoding="utf-8"))
    assert doc["_meta"]["schema"].startswith("g8k_store_pins/") and set(doc) - {"_meta"} == set(keys)
    tree = doc[keys[0]]
    assert doc[keys[1]] == tree and re.fullmatch(r"[0-9a-f]{40}", tree), tree
    assert doc["_meta"]["trees"] == {"allpos": tree} and doc["_meta"]["source_by_leg"] == {k: "allpos" for k in keys}
    src = doc["_meta"]["sources_by_tree"]["allpos"]
    assert tree in src and "544615" in src and "544730" in src and "standup_jobs.jsonl" in src
    assert "arditi_allpos" in doc["_meta"]["label"] and "arditi_allpos" in doc["_meta"]["note"] and "store_pins_default.json" in doc["_meta"]["note"]
    # the merged admission (the matrix-union read's two records), through the analysis' loader
    only_default = A.load_store_pins(default)
    pins = A.load_store_pins([default, allpos])
    assert set(only_default).isdisjoint(keys) and pins == {**only_default, keys[0]: tree, keys[1]: tree}
    for k in keys:
        assert A.admitted_pin_problem(k, tree, pins) == (tree, None)
    # KNOWN-BAD twins
    for bad in ("qwen35_4b/it/arditi@8192", "qwen35_4b/main/arditi_allpos@8192", "qwen35_4b/it/arditi_allpos@4096"):
        pin, why = A.admitted_pin_problem(bad, tree, pins)
        assert pin is None and "no admitted pin" in why, (bad, why)
    arditi_tree = only_default["qwen35_4b/main/arditi@8192"]                    # the published `arditi` row's pin: another tree
    assert arditi_tree != tree and only_default["qwen35_4b/main/arditi@16384"] != tree
    pin, why = A.admitted_pin_problem(keys[0], arditi_tree, pins)
    assert pin == tree and "is not the admitted pin" in why and arditi_tree[:8] in why and tree[:8] in why
    pin, why = A.admitted_pin_problem(keys[1], "0" * 40, pins)
    assert pin == tree and "is not the admitted pin" in why
    # the launcher's roster and the stamp script's derivation both land on this file
    assert G8K.STORE_PINS_PATHS[-1] == allpos and [p.name for p in G8K.STORE_PINS_PATHS].count(allpos.name) == 1
    assert G8K.matrix_name_of(G8K.resolve_matrix("arditi_allpos")) == "arditi_allpos"
    assert REPO / "configs" / "sdfbase" / "gsm8k" / f"store_pins_{G8K.matrix_name_of(G8K.resolve_matrix('arditi_allpos'))}.json" == allpos
