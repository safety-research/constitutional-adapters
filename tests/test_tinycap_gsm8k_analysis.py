"""``scripts/tinycap_gsm8k_paired200.py`` / ``scripts/tinycap_gsm8k_validators.py`` /
``scripts/gsm8k_overlap_check.py`` over the SYNTHETIC stores of ``tests/gsm8k_synthetic_store.py``
(written with this tree's real writers): the 38-leg design set with the four held rows present in
the matrix but never legs, an escalated cell with its rung, a triggered cell with no eligible item,
the within-view contrasts under both readings and both axes, the W / X fixture that flips a CI side,
the verdict rule, the noise-floor pair, the three-way gate-mapping test (canonical-id drift / mixed
producer / a summary-consistent altered score), the accepted-attempt ledger criterion by
``attempt_id``, G-READER-RUNG's checks on the synthetic rung store, every gate failing on a
constructed store, the refused cells by name, the fail-closed ``main()`` boundary (exit 3 / exit 4),
the two-venv shape, the figures and the sidecar, and the overlap check against a hand-computed 2x2
and a 199-pair refusal."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import gsm8k_synthetic_store as G
import pytest

from sisteer.capeval.tiers import cap_ladder as CL
from sisteer import site as SITE
from sisteer.capeval.tiers import inspect_rung as IR
from sisteer.capeval.tinycap import completion
from sisteer.capeval.serving import serving_attestation as SA
from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load("tinycap_gsm8k_paired200", REPO / "scripts" / "tinycap_gsm8k_paired200.py")
O = _load("gsm8k_overlap_check", REPO / "scripts" / "gsm8k_overlap_check.py")
V = A.V
M = A.M

B_FAST = 400
GEMMA, QWEN, NEMO, GLM = "gemma4e4b", "qwen35_4b", "nemotron3nano4b", "glm46vflash"
NONE, PROMPTED = G.CELL_NONE, G.CELL_PROMPTED
CONST_PSR, CONST_LORA = "constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce"
CTL_PSR, CTL_LORA = "constv2ctl2-full__psr_ce_all", "constv2ctl2-full__lora_r4_ce"
SI12_PSR, SI12_LORA = "safety_refusal__psr_kl_all", "safety_refusal__lora_r4"
ESC = (QWEN, "main", CONST_LORA)
NO_ELIGIBLE = (GLM, "main", CTL_PSR)
BELOW = (NEMO, "main", NONE)
HAS_INSPECT = importlib.util.find_spec("inspect_ai") is not None
ROWS = G.matrix_rows()
VIEWS_RETIRED = (GEMMA, f"{GEMMA}_it", QWEN, f"{QWEN}_it", NEMO, GLM)


def row(key, view, cell):
    return G.row_for(ROWS, key, view, cell)


def rows_of(*triples):
    return [row(*t) for t in triples]


def analyze(gen_root, *, matrix=None, **kw):
    kw.setdefault("matrix", matrix if matrix is not None else ROWS)
    kw.setdefault("canonical_ids", G.IDS)
    kw.setdefault("sampling_rows", G.SAMPLING_ROWS)
    kw.setdefault("render_facts", G.RENDER_FACTS)
    kw.setdefault("pins", G.pins_doc(ROWS))
    kw.setdefault("n_resamples", B_FAST)
    kw.setdefault("read_eval_logs", False)
    kw.setdefault("analysis_sha", G.FULL_SHA)
    return A.analyze(gen_root, G.SHA7, **kw)


def leg_of(report, key, view, cell, cap=G.CAP_BASE):
    return report["legs"][V.leg_key(key, view, cell, cap)]


def contrast_of(report, key, view, cell):
    return next(c for c in report["contrasts"] if c["model_key"] == key and c["view"] == view and c["cell"] == cell)


def small_matrix(*triples, held=()):
    """A matrix reduced to the named launch triples (+ the named held rows), every other row dropped."""
    keep = set(triples) | set(held)
    return [r for r in ROWS if (r["model_key"], r["view"], r["cell"]) in keep]


def build_overlap(root):
    retired = root / "retired"
    for view in VIEWS_RETIRED:
        G.build_retired_store(retired, view, correct=set(range(180)), extraction_failed={3, 7})
    for key in G.MODEL_KEYS:
        G.build_overlap_stock_store(root, key, correct=set(range(178)) | {181, 182})
    return O.run(retired_root=retired, stock_root=root / "overlap", models=list(G.MODEL_KEYS), canonical_ids=G.IDS, ids_path=None,
                 pairing_rows=G.pairing_rows(), pairing_source="synthetic rows", repo_root=REPO, store_pin=G.SHA7, n_resamples=B_FAST, seed=0, ci=0.95, eval_logs=False)


@pytest.fixture(scope="module")
def full_store(tmp_path_factory):
    """All 38 launch legs (gemma's two header-located si12 rows are
    servable), an escalated cell with its rung, a triggered cell with no
    eligible item, a below-threshold cell, the .eval logs (inspect venv only)
    and the overlap record."""
    root = tmp_path_factory.mktemp("g8k_full")
    dirs = G.build_all(root, rows=ROWS, skip=[ESC, NO_ELIGIBLE, BELOW])
    assert len(dirs) == 35
    esc = G.build_escalated_cell(root, row(*ESC), at_cap=range(190, 199), loops={190, 191}, capped_scored_correct={198},
                                 rung_still_at_cap={192, 193}, rung_boundary={194})
    assert esc["record"]["exit"] == CL.ESCALATE and esc["rung_cell_dir"] is not None
    G.build_store(root, row(*NO_ELIGIBLE), correct=G.CORRECT_BY_CELL[CTL_PSR] - set(range(190, 198)), unscored=range(190, 198), loops=range(190, 198))
    G.build_store(root, row(*BELOW), correct=G.CORRECT_NONE - set(range(198, 200)), unscored=range(198, 200))   # 2 of 200 = 1.0 %: at the line
    if HAS_INSPECT:
        for cd in [*dirs.values(), esc["source_cell_dir"], esc["rung_cell_dir"]]:
            G.write_inspect_log(cd, name="2026-01-01T00-00-00")
        for k in (NO_ELIGIBLE, BELOW):
            G.write_inspect_log(next((root / row(*k)["out_root"] / "cells").glob("*")), name="2026-01-01T00-00-00")
    overlap = build_overlap(root)
    return SimpleNamespace(root=root, esc=esc, overlap=overlap)


@pytest.fixture(scope="module")
def full_report(full_store):
    return analyze(full_store.root, overlap=full_store.overlap, read_eval_logs=HAS_INSPECT)


# --------------------------------------------------------------------------- #
# the reader contract and the matrix                                           #
# --------------------------------------------------------------------------- #


def test_reader_constants_are_the_producers_own_by_import():
    """A-P1.1: every contract item a producer exports is taken from it by IMPORT (the same object, or an equal
    value) — the adapter's rule / threshold / report line / exits / record fields, the contract's subset schema
    / fields / validator / digest / cap_rung key, the standard row task — and the fixture spells none of them
    itself. The reader's former spellings (a U+2212 minus in the rule, an `eligible` list) are gone."""
    assert V.TASK == C.GSM8K_STANDARD_TASK == C.GSM8K_PAIRED_TASK == IR.GSM8K_TASK == "inspect/gsm8k_paired200"
    assert V.N_ITEMS == 200 and V.CAP_BASE == 8192 and V.CAP_RUNG == CL.CAP_CEILING == IR.GSM8K_CEILING == 16384
    # the adapter (inspect_rung) is the ladder record's author: rule, threshold, report line, exits, field names
    assert V.GSM8K_TRIGGER_RULE is IR.GSM8K_TRIGGER_RULE and G.TRIGGER_RULE is IR.GSM8K_TRIGGER_RULE
    assert "\u2212" not in V.GSM8K_TRIGGER_RULE and " - degenerate - boundary" in V.GSM8K_TRIGGER_RULE
    assert V.GSM8K_TRIGGER_THRESHOLD == IR.GSM8K_TRIGGER_THRESHOLD == G.TRIGGER_THRESHOLD == 0.01
    assert V.GSM8K_TRIGGER_RULES_ADMITTED is IR.GSM8K_TRIGGER_RULES_ADMITTED and V.GSM8K_TRIGGER_RULE_3PCT_RETIRED is IR.GSM8K_TRIGGER_RULE_3PCT_RETIRED
    assert V.GSM8K_TRIGGER_PROVENANCE is IR.GSM8K_TRIGGER_PROVENANCE and A.TRIGGER_PCT == V.trigger_pct_label() == "1 %"
    assert V.GSM8K_REPORT_LINE == IR.GSM8K_REPORT_LINE == G.REPORT_LINE == CL.AT_CAP_SHARE_THRESHOLD == 0.01
    assert V.EXIT_TRIGGERED_NO_ELIGIBLE == IR.EXIT_TRIGGERED_NO_ELIGIBLE == G.EXIT_TRIGGERED_NO_ELIGIBLE
    assert V.LADDER_EXITS == tuple(IR.EXITS) and set(V.EXITS_TRIGGERED_TERMINAL) | set(V.EXITS_NOT_TRIGGERED) <= set(IR.TERMINAL_EXITS)
    assert V.EXITS_NOT_TRIGGERED == tuple(IR.NOT_TRIGGERED_EXITS) and V.EXITS_TRIGGERED_TERMINAL == tuple(IR.TRIGGERED_TERMINAL_EXITS)
    assert V.EXITS_NAMING_A_RUNG == (IR.EXIT_ESCALATE, CL.EXIT_TRANSPORT_REFUSED) and IR.EXIT_ESCALATE == CL.ESCALATE
    assert V.LADDER_RECORD_FIELDS == tuple(IR.RECORD_INSTRUMENT_FIELDS) and "eligible_ids" in V.LADDER_RECORD_FIELDS and "eligible" not in V.LADDER_RECORD_FIELDS
    assert (V.LADDER_ELIGIBLE_IDS, V.LADDER_NON_TERMINATING_IDS, V.LADDER_BOUNDARY_IDS) == (IR.RECORD_FIELD_ELIGIBLE_IDS, IR.RECORD_FIELD_NON_TERMINATING_IDS, IR.RECORD_FIELD_BOUNDARY_IDS)
    assert V.LADDER_ELIGIBLE_IDS == "eligible_ids" and V.LADDER_RECORD_SCHEMA == CL.RECORD_SCHEMA and V.LADDER_RECORD_NAME == CL.RECORD_NAME
    assert V.C_STOP_AT_CAP == IR.STOP_REASON_AT_CAP == "max_tokens"
    # this instrument's rule is NOT Evals' truncated-share string
    assert V.GSM8K_TRIGGER_RULE != CL.TRIGGER_RULE and "0.01" in V.GSM8K_TRIGGER_RULE
    # the contract owns the subset file and the rung identity
    assert V.RUNG_SUBSET_SCHEMA == C.ITEM_SUBSET_SCHEMA == "rung_subset/1" and V.RUNG_SUBSET_FIELDS == tuple(C.ITEM_SUBSET_FIELDS)
    assert V.validate_item_subset is C.validate_item_subset and V.subset_sha16 is C.item_subset_sha16
    assert V.IDENTITY_KEY_CAP_RUNG == C.IDENTITY_KEY_CAP_RUNG == "cap_rung" and V.CAP_RUNG_FIELDS == tuple(C.CAP_RUNG_FIELDS)
    assert V.MANIFEST_KEY_ITEM_SUBSET == C.MANIFEST_KEY_ITEM_SUBSET and V.ITEM_SUBSET_BLOCK_FIELDS == tuple(C.ITEM_SUBSET_BLOCK_FIELDS)
    # the class stamp triples are the contract's derivations (never the plan's spelling)
    assert V.STAMPS_BASE == ("cell-stable", "not-consulted", "thinking_on") and V.STAMPS_FITTED == ("cell-stable", "always", "thinking_on")
    assert V.STAMPS_DECLARED == ("request-unique", "declared-request-unique", "thinking_on") and V.STAMPS_DECLARED[1] == C.DECLARED_SALT_POLICY_LABEL
    assert V.STAMPS_DECLARED == tuple(C.expected_stamp_regimes(lens=True, arm=C.THINKING_ON, policy=C.DECLARED_SALT_POLICY_LABEL))
    assert (V.STAMPS_BASE, V.STAMPS_FITTED, V.STAMPS_DECLARED) == (tuple(C.G8K_STAMPS_BASE), tuple(C.G8K_STAMPS_FITTED), tuple(C.G8K_STAMPS_DECLARED))
    assert V.expected_stamps_for(fitted=True, declared=True) == tuple(C.g8k_expected_stamps(fitted=True, declared=True)) == V.STAMPS_DECLARED
    assert V.expected_stamps_for(fitted=False, declared=False) == V.STAMPS_BASE and V.expected_stamps_for(fitted=True, declared=False) == V.STAMPS_FITTED
    # the tuple names are the contract's (contracts.g8k_tuple_name / G8K_EXPERIMENT_NAMES): the rung's derived from the row's
    assert V.expected_tuple_name({"model_key": QWEN, "declared": True}, 16384) == C.g8k_tuple_name(QWEN, 16384, declared=True) in C.G8K_EXPERIMENT_NAMES
    with pytest.raises(A.AnalysisError, match="authorised"):
        V.tuple_name_for_cap("gsm8k_paired200_nomodel_8192", 16384)
    # the attempt-scoped fields and any further name the contracts export flip their pin on by themselves
    assert V.LEDGER_ATTEMPT_ID_FIELD == G.LEDGER_ATTEMPT_ID_FIELD == "attempt_id" and V.MANIFEST_KEY_ACCEPTED_ATTEMPT == G.MANIFEST_KEY_ACCEPTED_ATTEMPT
    assert {V.RESUME_ATTEMPT_STARTED_FIELD, V.RESUME_ATTEMPT_ENDED_FIELD} == {"started", "ended"} <= set(C.RESUME_ATTEMPT_SCOPED_FIELDS)
    for ours, theirs in (("LEDGER_ATTEMPT_ID_FIELD", "LEDGER_ATTEMPT_ID_FIELD"), ("MANIFEST_KEY_ACCEPTED_ATTEMPT", "MANIFEST_KEY_ACCEPTED_ATTEMPT"),
                         ("STAMPS_DECLARED", "STAMPS_DECLARED"), ("STAMPS_DECLARED", "GSM8K_STAMPS_DECLARED"), ("STAMPS_DECLARED", "DECLARED_STAMP_TRIPLE")):
        if hasattr(C, theirs):
            assert tuple(getattr(V, ours)) == tuple(getattr(C, theirs)), (ours, theirs)
    # the layout literals the reader keeps are lane B's (g8k_cells), pinned when the launcher module is importable
    g8k_cells = _optional_g8k_cells()
    if g8k_cells is not None:
        assert V.RUNG_SUBSET_NAME == g8k_cells.RUNG_SUBSET_FILENAME and V.LADDER_RECORD_NAME == g8k_cells.CAP_LADDER_FILENAME
        assert V.RUNG_SUBSET_REL.startswith(g8k_cells.RUNG_DIR_REL + "/") and V.CAP_BASE == g8k_cells.CAP and V.CAP_RUNG == g8k_cells.RUNG_CAP
        # the row's `object_source` spellings — the external bank and the it-doc re-fit's admission record — are the generator's
        assert V.OBJECT_SOURCE_EXTERNAL == g8k_cells.OBJECT_SOURCE_EXTERNAL == "external" and V.OBJECT_SOURCE_ITDOC == g8k_cells.OBJECT_SOURCE_ITDOC == "itdoc"
        assert V.OBJECT_SOURCES == tuple(g8k_cells.OBJECT_SOURCES)
    assert V.tuple_name_for_cap("gsm8k_paired200_qwen35_4b_declared_8192", 16384) == "gsm8k_paired200_qwen35_4b_declared_16384"
    with pytest.raises(A.AnalysisError, match="8192"):
        V.tuple_name_for_cap("gsm8k_paired200_qwen35_4b", 16384)


def _optional_g8k_cells():
    p = REPO / "scripts" / "sdfbase" / "gsm8k" / "g8k_cells.py"
    if not p.is_file():
        return None
    try:
        return _load("g8k_cells", p)
    except Exception:       # noqa: BLE001 — lane B's launcher module is edited concurrently; an unimportable copy is not this test's finding
        return None


def test_matrix_loads_with_the_plans_counts_and_refuses_defects(tmp_path):
    mx = V.load_matrix(ROWS)
    # every launch row is servable — no expected refusal in the design (38 servable, 32 computable contrasts)
    assert mx["counts"] == {"rows": 42, "held": 4, "launched": 38, "fitted_launched": 28, "in_support": 10, "declared": 4, "views": 6,
                            "contrasts_designed": 32, "expected_refusals": 0}
    assert A.load_expected_refusals(mx, None) == {}
    # the mapping form (a file / mapping keyed by the 8,192 leg) resolves beside the rows; a leg outside the launch set refuses
    doc = G.expected_refusals_doc(ROWS)
    assert set(doc) == {V.leg_key(GEMMA, "main", SI12_PSR, 8192), V.leg_key(GEMMA, "main", SI12_LORA, 8192)}
    assert A.load_expected_refusals(mx, doc) == doc
    with pytest.raises(A.AnalysisError, match="not launch rows"):
        A.load_expected_refusals(mx, {V.leg_key(GEMMA, "it", CONST_PSR, 8192): "x"})
    # lane B's row shape: a boolean flag with the reason beside it resolves to the reason as the code
    flagged = [dict(r) for r in ROWS]
    r = next(x for x in flagged if (x["model_key"], x["view"], x["cell"]) == (GEMMA, "main", SI12_PSR))
    r.update(expected_refusal=True, expected_refusal_reason="header-located object on a header-invariant family")
    assert A.load_expected_refusals(V.load_matrix(flagged), None) == {V.leg_key(GEMMA, "main", SI12_PSR, 8192): "header-located object on a header-invariant family"}
    assert V.load_matrix(flagged)["counts"]["expected_refusals"] == 1
    assert {(r["model_key"], r["view"], r["cell"]) for r in mx["held"]} == {(GEMMA, "it", CONST_PSR), (GEMMA, "it", CONST_LORA), (QWEN, "it", CONST_PSR), (QWEN, "it", CONST_LORA)}
    assert all(r["held_reason"] for r in mx["held"])
    assert all(r["tag"] == (r["model_key"] if r["view"] == "main" else f"{r['model_key']}_{r['view']}") for r in mx["rows"])
    assert all(r["rung_out_root"] == r["out_root"] + "__rung16384" and r["rung_subset"].startswith("rung16384/") for r in mx["rows"])
    # the file form: the digest is the file's bytes
    p = G.write_matrix(tmp_path / "matrix.json", ROWS)
    mx2 = V.load_matrix(p)
    assert mx2["path"] == str(p) and len(mx2["sha256"]) == 64 and mx2["counts"] == mx["counts"]
    # defects refuse by name
    assert V.matrix_pins_problem(mx["rows"], G.pins_doc(ROWS)) is None
    bad_pins = G.pins_doc(ROWS)
    bad_pins[QWEN][G.staged_id_of(CONST_PSR, "main")]["object_sha256"] = "0" * 64
    assert "pin file" in V.matrix_pins_problem(mx["rows"], bad_pins)
    for edit, why in (
            (lambda r: r.update(status="paused"), "status"),
            (lambda r: r.update(status="held", held_reason=None), "held_reason"),
            (lambda r: r.update(tag="qwen35_4b_main"), "tag"),
            (lambda r: r.update(sampling_mode="general"), "sampling_mode"),
            (lambda r: r.update(fitted=True), "fitted"),
            (lambda r: r.update(rung_out_root=r["out_root"] + "__rung8192"), "rung_out_root"),
            (lambda r: r.update(expected_stamps=["cell-stable", "always", "thinking_on"]), "expected_stamps"),
    ):
        rows = [dict(r) for r in ROWS]
        edit(next(r for r in rows if (r["model_key"], r["view"], r["cell"]) == (QWEN, "main", NONE)))
        with pytest.raises(A.AnalysisError, match=why):
            V.load_matrix(rows)
    rows = [dict(r) for r in ROWS]
    r = next(x for x in rows if (x["model_key"], x["view"], x["cell"]) == (QWEN, "main", SI12_PSR))
    r["served_positions_expected"] = "all"          # a declared cell served all-position: contradicts the class
    with pytest.raises(A.AnalysisError, match="declared"):
        V.load_matrix(rows)
    rows = [dict(r) for r in ROWS] + [dict(row(QWEN, "main", NONE))]
    with pytest.raises(A.AnalysisError, match="twice"):
        V.load_matrix(rows)


# --------------------------------------------------------------------------- #
# the full synthetic run                                                       #
# --------------------------------------------------------------------------- #


def test_full_synthetic_run_is_a_production_report(full_report):
    rep = full_report
    assert rep["unusable_legs"] == {}
    assert rep["gates"]["failed"] == [] and rep["gates"]["pending"] == [], rep["gates"]["failed"]
    assert A.production_refusals(rep) == []
    assert rep["verdict"]["word"] == "filed"
    # 38 servable rows, 32 computable contrasts, no refusal expected or filed
    assert rep["design"]["n_contrasts_designed"] == 32 and rep["design"]["n_contrasts_computable"] == 32
    assert rep["design"]["n_launch_rows"] == 38 and rep["design"]["n_held_rows"] == 4 and rep["design"]["n_escalated_cells"] == 1
    assert rep["design"]["n_expected_refusals"] == 0 and rep["design"]["n_servable"] == 38 and rep["expected_refusals"] == {}
    assert rep["refused"] == [] and all(c["expected_refusal"] is None for c in rep["contrasts"])
    for cell in (SI12_PSR, SI12_LORA):
        assert leg_of(rep, GEMMA, "main", cell)["refused"] is None and contrast_of(rep, GEMMA, "main", cell)["computable"]
    assert len(rep["held"]) == 4 and all(h["reason"] for h in rep["held"])
    # held rows are never legs and never missing: no leg key of theirs exists, and the completeness gate counts 38
    for h in rep["held"]:
        assert V.leg_key(h["model_key"], h["view"], h["cell"], 8192) not in rep["legs"]
    assert rep["gates"]["A0_completeness"]["n_launch_rows"] == 38 and rep["gates"]["A0_completeness"]["n_refused_expected"] == 0
    assert rep["gates"]["A0_completeness"]["n_usable_base"] == 38 and rep["gates"]["A0_completeness"]["n_held"] == 4
    # 38 base legs + 1 rung leg are read
    assert len(rep["legs"]) == 39 and sum(1 for k in rep["legs"] if k.endswith("@16384")) == 1
    # the top items of a production report carry no FAILED gate and every merged word filed
    assert not any("FAILED" in t for t in rep["top_items"]) and any(t.endswith(A.WORD_LOWER) for t in rep["top_items"])
    # the it view contributes prompted - none only
    it = [c for c in rep["contrasts"] if c["view"] == "it"]
    assert {c["cell"] for c in it} == {PROMPTED} and len(it) == 2
    assert rep["matrix"]["counts"]["launched"] == 38 and rep["canonical_ids"]["n"] == 200
    # every gate is named A0..A10
    assert set(A.GATE_NAMES) == {k for k in rep["gates"] if k[0] == "A" and "_" in k}
    assert rep["gates"]["A10_overlap"]["status"] == A.GATE_PASS and set(rep["gates"]["A10_overlap"]["models"]) == set(G.MODEL_KEYS)


def test_the_contrast_words_under_both_readings_and_both_axes(full_report):
    rep = full_report
    # a constitution object 20 items lower one-way on a model whose none has no at-cap draw (glm; nemotron's none is the
    # below-threshold cell with two at cap, exactly at the 1 % line): resolved LOWER under every axis, both readings agree
    c = contrast_of(rep, GLM, "main", CONST_PSR)
    for reading in A.READINGS:
        v = c["readings"][reading]["verdict"]
        assert v["word"] == A.WORD_LOWER and v["agree_W_X"] and v["W"]["word"] == v["X"]["word"] == v["stock"]["word"] == A.WORD_LOWER
        est = c["readings"][reading]["estimands"]
        assert est["n_excluded_X"] == 0
        for axis in A.AXES:
            for b in A.BRANCHES:
                st = est[axis][b]["stats"][A.CONTRAST_VS_NONE]
                assert st["point"] == pytest.approx(-0.10) and st["ci_high"] < 0
        assert c["readings"][reading]["mcnemar"][A.AXIS_STOCK][A.BRANCH_PRIMARY]["p_value"] < 0.001
    # a control object: a two-way discordance whose CI covers zero -> not resolved
    c = contrast_of(rep, GLM, "main", CTL_LORA)
    v = c["readings"][A.READING_MERGED]["verdict"]
    assert v["word"] == A.WORD_NOT_RESOLVED and "(a)" in v["W"]["reason"]
    # an un-escalated cell's two readings are the same object
    assert c["escalated"] is False
    assert rep["cells"][V.cell_key(GLM, "main", CTL_LORA)]["readings"][A.READING_MERGED]["same_as_r8192"] is True
    # nemotron's none carries two unscored at-cap draws: X excludes exactly those two from every nemotron contrast
    n = contrast_of(rep, NEMO, "main", CONST_PSR)
    assert n["readings"][A.READING_MERGED]["estimands"]["n_excluded_X"] == 2
    assert n["readings"][A.READING_MERGED]["estimands"][A.AXIS_X][A.BRANCH_PRIMARY]["n_items"] == 198
    assert c["readings"][A.READING_MERGED]["estimands"][A.AXIS_STOCK][A.BRANCH_PRIMARY]["stats"] == c["readings"][A.READING_R8192]["estimands"][A.AXIS_STOCK][A.BRANCH_PRIMARY]["stats"]
    # the words map onto this experiment's vocabulary
    assert {w for words in rep["verdict"]["words"].values() for w in words.values()} <= {A.WORD_LOWER, A.WORD_HIGHER, A.WORD_NOT_RESOLVED}
    # the summary table carries both readings side by side
    s = next(x for x in rep["contrast_summary"] if x["name"] == c["name"])
    assert set(s) >= {A.READING_R8192, A.READING_MERGED} and s[A.READING_MERGED]["point"][A.AXIS_STOCK][A.BRANCH_PRIMARY] == pytest.approx(-0.01)


def test_the_noise_floor_pair_is_filed_per_model_and_never_pooled(full_report):
    rep = full_report
    assert {n["model_key"] for n in rep["noise_floor"]} == {GEMMA, QWEN}
    for n in rep["noise_floor"]:
        assert n["computable"] and n["kind"] == "noise_floor"
        st = n["readings"][A.READING_MERGED]["estimands"][A.AXIS_STOCK][A.BRANCH_PRIMARY]["stats"][A.CONTRAST_NOISE]
        assert st["point"] == 0.0 and st["ci_low"] <= 0.0 <= st["ci_high"]
        assert n["readings"][A.READING_MERGED]["verdict"]["word"] == A.WORD_NOT_RESOLVED
    # the pair never enters a contrast: every contrast is cell - none within ONE view
    assert all(c["none_key"].split("/")[1] == c["view"] for c in rep["contrasts"])


def test_the_wx_fixture_a_scored_correct_ceiling_at_cap_item_flips_a_ci_side(tmp_path):
    """Six case-(c) items (3.0 % — over the 1 % trigger since 2026-09-12; the rung is
    NOT built here: this test is about the W / X axes, the ladder gate has its own):
    at the cap AFTER the think block, scored, and CORRECT in the cell — items
    ``none`` also answers correctly, below its cap. The STOCK axis reads parity
    (170 vs 170); W scores the six wrong, a one-way six-item deficit whose CI
    excludes zero ('lower'); X excludes them from both legs, parity again (the CI
    covers zero). W and X disagree -> the word is withheld."""
    rows = small_matrix((QWEN, "main", NONE), (QWEN, "main", CONST_LORA))
    G.build_store(tmp_path, row(QWEN, "main", NONE), correct=set(range(170)))
    G.build_store(tmp_path, row(QWEN, "main", CONST_LORA), correct=set(range(170)), capped_scored=range(164, 170))
    rep = analyze(tmp_path, matrix=rows, overlap=None)
    c = contrast_of(rep, QWEN, "main", CONST_LORA)
    r = c["readings"][A.READING_MERGED]
    assert not c["escalated"] and rep["cells"][V.cell_key(QWEN, "main", CONST_LORA)]["ladder"]["triggered"] is True   # no rung store: the merged reading is the base's
    stock, w, x = (r["estimands"][a][A.BRANCH_PRIMARY]["stats"][A.CONTRAST_VS_NONE] for a in (A.AXIS_STOCK, A.AXIS_W, A.AXIS_X))
    # the two legs are identical apart from the six: the stock and X intervals sit exactly at zero (they cover it), W's excludes it
    assert stock["point"] == 0.0 and stock["ci_low"] <= 0 <= stock["ci_high"] and not stock["excludes_zero"]
    assert w["point"] == pytest.approx(-0.03) and w["ci_high"] < 0 and w["excludes_zero"] and w["point_sign"] == -1
    assert x["point"] == 0.0 and x["ci_low"] <= 0 <= x["ci_high"] and not x["excludes_zero"] and r["estimands"]["n_excluded_X"] == 6
    assert x["point_sign"] == 0 and M._interval_side(w["ci_low"], w["ci_high"]) == -1 and M._interval_side(x["ci_low"], x["ci_high"]) == 0
    assert r["verdict"]["W"]["word"] == A.WORD_LOWER and r["verdict"]["X"]["word"] == A.WORD_NOT_RESOLVED
    assert r["verdict"]["word"] == A.WORD_NOT_RESOLVED and not r["verdict"]["agree_W_X"] and "X:" in r["verdict"]["reason"]
    # the residual mask is the stop reason regardless of `scored`: six scored items carry it
    items = rep["cells"][V.cell_key(QWEN, "main", CONST_LORA)]["readings"][A.READING_MERGED]["items"]
    assert sum(1 for it in items.values() if it["at_cap"] and it["scored"] and it["value"] == 1.0) == 6
    assert rep["cells"][V.cell_key(QWEN, "main", CONST_LORA)]["readings"][A.READING_MERGED]["n_at_cap_residual"] == 6
    # over the trigger on the record and on the reader's own share (3.0 % > 1 %; the record escalates, no rung was built)
    at_cap = rep["legs"][V.leg_key(QWEN, "main", CONST_LORA, 8192)]["at_cap"]
    assert at_cap["share"] == 0.03 and at_cap["trigger_exceeded"] is True and at_cap["line_1pct_exceeded"] is True
    assert rep["cells"][V.cell_key(QWEN, "main", CONST_LORA)]["ladder"]["exit"] == CL.ESCALATE


def test_the_verdict_rule_one_item_bound_case_stays_unresolved(tmp_path):
    rows = small_matrix((GLM, "main", NONE), (GLM, "main", CONST_PSR))
    G.build_store(tmp_path, row(GLM, "main", NONE), correct=set(range(170)), no_think={40, 41})
    G.build_store(tmp_path, row(GLM, "main", CONST_PSR), correct=set(range(169)), no_think={40, 41})
    rep = analyze(tmp_path, matrix=rows, overlap=None)
    v = contrast_of(rep, GLM, "main", CONST_PSR)["readings"][A.READING_MERGED]["verdict"]
    assert v["word"] == A.WORD_NOT_RESOLVED
    assert v["W"]["conditions"]["d_bound_excludes_zero_same_side"] is True and v["W"]["conditions"]["a_primary_ci_excludes_zero"] is False


# --------------------------------------------------------------------------- #
# the three-way gate-mapping test                                              #
# --------------------------------------------------------------------------- #


def test_gate_mapping_canonical_drift_mixed_producer_and_a_summary_consistent_altered_score(tmp_path):
    rows = small_matrix((QWEN, "main", NONE), (QWEN, "main", CONST_PSR), (QWEN, "main", CTL_PSR))
    G.build_all(tmp_path / "gen", rows=rows)
    # (1) canonical-id drift: one id swapped in the supplied canonical set -> A7 FAILS naming the population; A0 passes (200 unique ids)
    drift = [i for i in G.IDS if i != G.IDS[7]] + ["gsm8k_ffffffff"]
    rep = analyze(tmp_path / "gen", matrix=rows, canonical_ids=drift, overlap=None)
    assert rep["gates"]["A7_population"]["status"] == A.GATE_FAIL and "canonical" in rep["gates"]["A7_population"]["detail"]
    assert rep["gates"]["A0_completeness"]["status"] == A.GATE_PASS and rep["unusable_legs"] == {}
    assert "A7_population" in rep["gates"]["failed"] and A.production_refusals(rep) and rep["verdict"]["word"] is None
    # (2) a mixed-producer pair: the object cell from another tree carrying the same 7-char pin -> its contrast does not run, A7 FAILS
    other = G.SHA7 + "f" * 33
    G.build_all(tmp_path / "mixed", rows=rows, overrides={(QWEN, "main", CONST_PSR): {"tree_sha": other}})
    rep = analyze(tmp_path / "mixed", matrix=rows, overlap=None)
    c = contrast_of(rep, QWEN, "main", CONST_PSR)
    assert not c["computable"] and "producer" in c["reason"] and other in c["reason"]
    assert rep["gates"]["A7_population"]["status"] == A.GATE_FAIL and "producer" in rep["gates"]["A7_population"]["detail"]
    assert contrast_of(rep, QWEN, "main", CTL_PSR)["computable"] and rep["unusable_legs"] == {}
    assert any("did not run" in r for r in A.production_refusals(rep))
    # (3) an outcome score altered CONSISTENTLY with its summary (and rows) but not with the .eval log -> the re-derivation refuses the leg: A0 FAILS
    pytest.importorskip("inspect_ai")
    dirs = G.build_all(tmp_path / "eval", rows=rows)
    for k, cd in dirs.items():
        G.write_inspect_log(cd, name="2026-01-01T00-00-00",
                            value_edit=(lambda item_id, v: (1.0 - v) if item_id == G.IDS[3] else v) if k.endswith(CONST_PSR) else None)
    rep = analyze(tmp_path / "eval", matrix=rows, overlap=None, read_eval_logs=True)
    leg = leg_of(rep, QWEN, "main", CONST_PSR)
    assert leg["refused"] and "disagrees" in leg["refused"] and G.IDS[3] in leg["refused"]
    assert rep["gates"]["A0_completeness"]["status"] == A.GATE_FAIL and leg["leg"] in rep["unusable_legs"]
    assert leg_of(rep, QWEN, "main", CTL_PSR)["eval_check"]["agrees"] is True
    # each of the three FAILS is on its named gate, not on the others
    assert rep["gates"]["A7_population"]["status"] == A.GATE_PASS


# --------------------------------------------------------------------------- #
# A1: the accepted attempt by attempt_id                                       #
# --------------------------------------------------------------------------- #


def test_the_accepted_attempt_ledger_criterion_reads_by_attempt_id(tmp_path):
    rows = small_matrix((NEMO, "main", NONE), (NEMO, "main", CONST_LORA))
    # a1 died after 150 items with one rejected request; a2 issued all 200 and is accepted; a3 finalised (probes only)
    attempts = [G.AttemptSpec("a1", completed=False, n_items=150, rejected_items=[10]), G.AttemptSpec("a2", accepted=True),
                G.AttemptSpec("a3", kind="finalisation")]
    G.build_all(tmp_path / "gen", rows=rows, overrides={(NEMO, "main", CONST_LORA): {"attempts": attempts}})
    rep = analyze(tmp_path / "gen", matrix=rows, overlap=None)
    led = leg_of(rep, NEMO, "main", CONST_LORA)["ledger"]
    assert led["passed"], led["failures"]
    assert led["accepted_attempt"] == "a2" and led["n_calls"] == 200 and led["n_rejected"] == 0 and led["n_stray_rows"] == 0
    assert led["attempts"]["a1"] == {**led["attempts"]["a1"], "n_task_rows": 150, "n_rejected": 1, "n_forwarded": 149, "kind": "generation", "accepted": False}
    assert led["attempts"]["a3"]["n_task_rows"] == 0 and led["attempts"]["a3"]["kind"] == "finalisation" and led["attempts"]["a3"]["n_probe_rows"] > 0
    assert led["n_task_rows_all_attempts"] == 350
    re_ = leg_of(rep, NEMO, "main", CONST_LORA)["request_evidence"]
    assert re_["passed"], re_["failures"]
    assert re_["per_attempt"]["a1"]["n_task_records"] == 150 and re_["per_attempt"]["a2"]["n_task_records"] == 200 and re_["n_stray_records"] == 0
    assert rep["gates"]["A1_ledger"]["status"] == A.GATE_PASS and rep["gates"]["A2_identity"]["status"] == A.GATE_PASS
    # the pre-contract whole-ledger reading would have refused a2 for a1's rejection: the rejection is ATTRIBUTED, not read as the clean attempt's
    assert leg_of(rep, NEMO, "main", CONST_LORA)["summary"]["n_http_rejected_retried"] == 0

    # a stray row: an attempt the manifest does not record -> A1 FAILS by name
    def stray(lrows):
        lrows[-1]["attempt_id"] = "zz"

    G.build_all(tmp_path / "stray", rows=rows, overrides={(NEMO, "main", CONST_LORA): {"ledger_edit": stray}})
    rep = analyze(tmp_path / "stray", matrix=rows, overlap=None)
    g = rep["gates"]["A1_ledger"]
    assert g["status"] == A.GATE_FAIL and "stray" in g["detail"] and "zz" in g["detail"]
    # a manifest naming no accepted attempt refuses the leg at collection (the producer contract)
    G.build_all(tmp_path / "noacc", rows=rows, overrides={(NEMO, "main", CONST_LORA): {"manifest_edit": lambda m: m.pop(V.MANIFEST_KEY_ACCEPTED_ATTEMPT)}})
    rep = analyze(tmp_path / "noacc", matrix=rows, overlap=None)
    assert V.MANIFEST_KEY_ACCEPTED_ATTEMPT in leg_of(rep, NEMO, "main", CONST_LORA)["refused"]
    # a rejected call inside the ACCEPTED attempt is censoring: refused
    G.build_all(tmp_path / "rej", rows=rows, overrides={(NEMO, "main", CONST_LORA): {"attempts": [G.AttemptSpec("a1", accepted=True, rejected_items=[5])]}})
    rep = analyze(tmp_path / "rej", matrix=rows, overlap=None)
    assert "censored" in rep["gates"]["A1_ledger"]["detail"] or "rejected" in rep["gates"]["A1_ledger"]["detail"]
    # the ledger's max_tokens_requested must be the leg's cap
    def wrong_cap(lrows):
        for r in lrows:
            if r["task"] == V.TASK:
                r["max_tokens_requested"] = 4096
    G.build_all(tmp_path / "cap", rows=rows, overrides={(NEMO, "main", CONST_LORA): {"ledger_edit": wrong_cap}})
    assert "max_tokens_requested" in analyze(tmp_path / "cap", matrix=rows, overlap=None)["gates"]["A1_ledger"]["detail"]


# --------------------------------------------------------------------------- #
# the declared class (G-READER-DECL's checks) and the rung (G-READER-RUNG's)   #
# --------------------------------------------------------------------------- #


def test_the_declared_leg_class_checks(full_report, tmp_path):
    leg = leg_of(full_report, QWEN, "main", SI12_PSR)
    assert leg["refused"] is None and leg["declared"] and leg["fitted"]
    ident = leg["identity"]
    assert ident["virtual_model"].endswith("@thinking") and ident[C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT] == ["thinking"]
    assert leg["stamps"]["passed"] and leg["stamps"]["expected"] == dict(zip(C.EXPECTED_STAMP_TRIPLE_FIELDS, V.STAMPS_DECLARED))
    cert = leg["certification"]
    assert cert["passed"], cert["failures"]
    assert cert["route_lens_served_positions"] == ["response"] and cert["declared"] and cert["route_lens_serve_declarations"]
    assert leg["request_evidence"]["passed"] and leg["request_evidence"]["declared"]
    # an undeclared cell of the same view carries neither
    base = leg_of(full_report, QWEN, "main", CONST_PSR)
    assert base["identity"][C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT] is None and not base["certification"]["route_lens_serve_declarations"]
    assert base["certification"]["route_lens_served_positions"] == ["all"]
    # a declared leg whose task records carry no declaration FAILS A2 by name
    rows = small_matrix((QWEN, "main", NONE), (QWEN, "main", SI12_PSR))

    def undeclare(recs):
        for r in recs:
            if r["kind"] == "task":
                r["petri_thinking_on_present"], r["petri_thinking_on_value"] = False, None

    G.build_all(tmp_path / "gen", rows=rows, overrides={(QWEN, "main", SI12_PSR): {"request_evidence_edit": undeclare}})
    rep = analyze(tmp_path / "gen", matrix=rows, overlap=None)
    assert rep["gates"]["A2_identity"]["status"] == A.GATE_FAIL and "DECLARED" in rep["gates"]["A2_identity"]["detail"]
    # ... and a declaration on an UNDECLARED cell fails too
    def declare(recs):
        recs[-1]["petri_thinking_on_present"], recs[-1]["petri_thinking_on_value"] = True, True

    G.build_all(tmp_path / "gen2", rows=rows, overrides={(QWEN, "main", NONE): {"request_evidence_edit": declare}})
    rep = analyze(tmp_path / "gen2", matrix=rows, overlap=None)
    assert rep["gates"]["A2_identity"]["status"] == A.GATE_FAIL and "UNDECLARED" in rep["gates"]["A2_identity"]["detail"]


def test_g_reader_rung_checks_on_the_synthetic_rung_store(full_store, full_report):
    rep = full_report
    ck = V.cell_key(*ESC)
    cell = rep["cells"][ck]
    lad = cell["ladder"]
    assert lad["exit"] == CL.ESCALATE and lad["escalated"] and lad["triggered"] and lad["n_at_cap"] == 9 and lad["n_non_terminating"] == 2
    assert lad["n_eligible"] == lad["n_subset"] == 7 and lad["record_problem"] is None and lad["subset_problem"] is None
    rung = rep["legs"][cell["rung_leg"]]
    assert rung["refused"] is None and rung["rung"] and rung["cap"] == 16384 and rung["n_expected"] == 7
    assert rung["attempted_ids"] == lad["subset_ids"] == sorted(full_store.esc["subset"])
    assert rung["identity"][V.IDENTITY_KEY_CAP_RUNG]["source_identity_hash"] == rep["legs"][cell["base_leg"]]["identity"]["harness_config_hash"]
    assert rung["identity"][V.IDENTITY_KEY_CAP_RUNG]["subset_sha16"] == V.subset_sha16(lad["subset_ids"]) == lad["subset_sha16"]
    # the rung's hashed identity IS the contract's derivation from the planner's subset file; the record was read at the planner's path
    assert rung["identity"][V.IDENTITY_KEY_CAP_RUNG] == C.cap_rung_identity(full_store.esc["subset_doc"])
    assert Path(lad["record_path"]) == Path(full_store.esc["source_out_root"]) / CL.RECORD_NAME and lad["record_present"]
    assert rung["ledger"]["passed"] and rung["ledger"]["n_calls"] == 7 and rung["ledger"]["n_rejected"] == 0
    assert rung["identity"][C.MANIFEST_KEY_AUTHORISED_EXPERIMENT] == G.tuple_name_of(QWEN, declared=False, cap=16384)
    if HAS_INSPECT:
        assert rung["eval_check"]["available"] and rung["eval_check"]["agrees"] and rung["eval_check"]["per_item"]["n_compared"] == 7
    merged = cell["readings"][A.READING_MERGED]
    # residual at cap in the MERGED reading = the 2 loops (never redrawn, still the 8,192 draw) + the 2 redraws still at the ceiling
    assert merged["escalated"] and merged["n"] == 200 and merged["n_from_rung"] == 7 and merged["n_at_cap_residual"] == 4
    assert cell["readings"][A.READING_R8192]["n_at_cap_residual"] == 9
    assert merged["redraw_outcomes"] == {"still_at_cap": 2, "boundary_landing": 1, "extended": 4}
    for i in lad["subset_ids"]:
        assert merged["items"][i]["source"] == A.SOURCE_RUNG and cell["readings"][A.READING_R8192]["items"][i]["source"] == A.SOURCE_R8192
    assert cell["redraws"][G.IDS[194]]["outcome"] == "boundary_landing" and cell["redraws"][G.IDS[192]]["still_at_cap"]
    cost = next(t for t in rep["truncation_cost"] if t["cell_key"] == ck)
    assert cost["n_subset"] == 7 and cost["residual_at_ceiling"] == 2 and cost["n_boundary_landing"] == 1 and cost["n_extended"] == 4
    assert rep["gates"]["A5_at_cap"]["status"] == A.GATE_PASS and rep["gates"]["A6_ladder"]["status"] == A.GATE_PASS
    c = contrast_of(rep, *ESC)
    assert c["escalated"] and c["readings"][A.READING_MERGED]["estimands"][A.AXIS_STOCK][A.BRANCH_PRIMARY]["stats"] != c["readings"][A.READING_R8192]["estimands"][A.AXIS_STOCK][A.BRANCH_PRIMARY]["stats"]
    # the triggered cell with no eligible item: a terminal record, no rung, its reading the 8,192 one
    ck2 = V.cell_key(*NO_ELIGIBLE)
    lad2 = rep["cells"][ck2]["ladder"]
    assert lad2["exit"] == V.EXIT_TRIGGERED_NO_ELIGIBLE and lad2["triggered"] and lad2["n_eligible"] == 0 and lad2["n_non_terminating"] == 8
    assert rep["cells"][ck2]["rung_leg"] is None and rep["cells"][ck2]["readings"][A.READING_MERGED]["same_as_r8192"]
    assert V.leg_key(*NO_ELIGIBLE, 8192) in rep["truncation"]["terminal_no_rung"]
    # the below-threshold cell: 2 at cap = 1.0 % does not trigger (strict >); AT the 1 % line, not over it
    lad3 = rep["cells"][V.cell_key(*BELOW)]["ladder"]
    assert lad3["exit"] == CL.EXIT_BELOW_THRESHOLD and not lad3["triggered"] and not lad3["over_1pct"]   # 2 of 200: AT the line, not over it
    assert V.leg_key(*BELOW, 8192) not in rep["truncation"]["over_1pct"]
    assert set(rep["truncation"]["over_1pct"]) == set(rep["truncation"]["triggered"])                   # the line IS the trigger now


def test_rung_defects_refuse_by_name(tmp_path):
    rows = small_matrix((QWEN, "main", NONE), (QWEN, "main", CONST_LORA))
    G.build_store(tmp_path / "gen", row(QWEN, "main", NONE), correct=G.CORRECT_NONE)
    esc = G.build_escalated_cell(tmp_path / "gen", row(QWEN, "main", CONST_LORA), at_cap=range(190, 199), loops={190, 191})
    rep = analyze(tmp_path / "gen", matrix=rows, overlap=None)
    assert rep["gates"]["A6_ladder"]["status"] == A.GATE_PASS and rep["cells"][V.cell_key(QWEN, "main", CONST_LORA)]["rung_leg"]
    # the escalated cell's rung store removed -> A6 FAILS naming it
    shutil.rmtree(Path(esc["rung_cell_dir"]).parents[1])
    rep = analyze(tmp_path / "gen", matrix=rows, overlap=None)
    assert rep["gates"]["A6_ladder"]["status"] == A.GATE_FAIL and "escalated" in rep["gates"]["A6_ladder"]["detail"]
    # the subset file bound to another source hash -> A5 FAILS, no rung read
    G.build_store(tmp_path / "src", row(QWEN, "main", NONE), correct=G.CORRECT_NONE)
    esc = G.build_escalated_cell(tmp_path / "src", row(QWEN, "main", CONST_LORA), at_cap=range(190, 199), loops={190, 191})
    doc = json.loads(Path(esc["subset_path"]).read_text())
    doc["source_identity_hash"] = "deadbeef" + "0" * 8
    Path(esc["subset_path"]).write_text(json.dumps(doc))
    rep = analyze(tmp_path / "src", matrix=rows, overlap=None)
    assert rep["gates"]["A5_at_cap"]["status"] == A.GATE_FAIL and "source_identity_hash" in rep["gates"]["A5_at_cap"]["detail"]
    # the rung store over one id too many (a subset that is not the record's eligible set) -> the rung leg refuses: its
    # cap_rung digest is over the wide set while the planner's subset file names the eligible seven
    G.build_store(tmp_path / "wide", row(QWEN, "main", NONE), correct=G.CORRECT_NONE)
    esc = G.build_escalated_cell(tmp_path / "wide", row(QWEN, "main", CONST_LORA), at_cap=range(190, 199), loops={190, 191}, write_rung=False)
    wide = sorted(esc["subset"] + [G.IDS[0]])
    manifest = completion.read_store_manifest(esc["source_cell_dir"])
    G.build_store(tmp_path / "wide", row(QWEN, "main", CONST_LORA), correct=set(range(3)), cap=16384, ids=wide, rung_source=manifest, ladder=False)
    rep = analyze(tmp_path / "wide", matrix=rows, overlap=None)
    rung = rep["legs"][V.leg_key(QWEN, "main", CONST_LORA, 16384)]
    assert rung["refused"] and "cap_rung" in rung["refused"] and "subset" in rung["refused"]
    assert rep["gates"]["A0_completeness"]["status"] == A.GATE_FAIL
    # ... and a rung store whose cap_rung digest matches the subset file but whose draws exceed it refuses on its population
    G.build_store(tmp_path / "wide2", row(QWEN, "main", NONE), correct=G.CORRECT_NONE)
    esc = G.build_escalated_cell(tmp_path / "wide2", row(QWEN, "main", CONST_LORA), at_cap=range(190, 199), loops={190, 191}, write_rung=False)
    manifest = completion.read_store_manifest(esc["source_cell_dir"])
    G.build_store(tmp_path / "wide2", row(QWEN, "main", CONST_LORA), correct=set(range(3)), cap=16384, ids=sorted(esc["subset"] + [G.IDS[0]]),
                  rung_source=manifest, subset_doc=esc["subset_doc"], subset_path=esc["subset_path"], ladder=False)
    rung = analyze(tmp_path / "wide2", matrix=rows, overlap=None)["legs"][V.leg_key(QWEN, "main", CONST_LORA, 16384)]
    assert rung["refused"] and "draws" in rung["refused"]
    # a record carrying Evals' rule string instead of this instrument's -> A5 FAILS
    G.build_store(tmp_path / "rule", row(QWEN, "main", NONE), correct=G.CORRECT_NONE)
    G.build_store(tmp_path / "rule", row(QWEN, "main", CONST_LORA), correct=G.CORRECT_BY_CELL[CONST_LORA], ladder_edit=lambda r: r.update(rule=CL.TRIGGER_RULE))
    rep = analyze(tmp_path / "rule", matrix=rows, overlap=None)
    assert rep["gates"]["A5_at_cap"]["status"] == A.GATE_FAIL and "rule" in rep["gates"]["A5_at_cap"]["detail"]
    # the transport-refused terminal record on a triggered cell: A6 passes with no rung; a rung store beside it fails
    G.build_store(tmp_path / "tr", row(QWEN, "main", NONE), correct=G.CORRECT_NONE)
    G.build_store(tmp_path / "tr", row(QWEN, "main", CONST_LORA), correct=G.CORRECT_BY_CELL[CONST_LORA] - set(range(190, 199)),
                  unscored=range(190, 199), transport_ok=False)
    rep = analyze(tmp_path / "tr", matrix=rows, overlap=None)
    lad = rep["cells"][V.cell_key(QWEN, "main", CONST_LORA)]["ladder"]
    assert lad["exit"] == CL.EXIT_TRANSPORT_REFUSED and lad["triggered"] and rep["gates"]["A6_ladder"]["status"] == A.GATE_PASS
    assert rep["gates"]["A5_at_cap"]["status"] == A.GATE_PASS and lad["transport"]["ok"] is False


# --------------------------------------------------------------------------- #
# every gate fails on a constructed store; refusals by name; the boundary       #
# --------------------------------------------------------------------------- #


def test_gates_each_fail_on_a_constructed_store(tmp_path):
    rows = small_matrix((GLM, "main", NONE), (GLM, "main", CONST_PSR))
    common = {"no_think": {40, 41}}

    def build(name, **over):
        G.build_all(tmp_path / name, rows=rows, overrides={(GLM, "main", CONST_PSR): over}, **common)
        return analyze(tmp_path / name, matrix=rows, overlap=None)

    # A3: the lens-position record's digest is not the pin
    def bad_digest(ev):
        ev["lens_position_attested"]["object_digest"]["scalar"] = "0" * 64
    rep = build("a3", evidence_edit=bad_digest)
    assert rep["gates"]["A3_certification"]["status"] == A.GATE_FAIL and "object_digest" in rep["gates"]["A3_certification"]["detail"]
    # A4: the scoring path — another unterminated policy
    rep = build("a4", summary_edit=lambda s: s.update(unterminated_policy="incorrect"), header_edit=lambda h: h.update(unterminated_policy="incorrect"))
    assert rep["gates"]["A4_scoring_path"]["status"] == A.GATE_FAIL and "unterminated_policy" in rep["gates"]["A4_scoring_path"]["detail"]
    # A5: no ladder record on the cell
    rep = build("a5", ladder=False)
    assert rep["gates"]["A5_at_cap"]["status"] == A.GATE_FAIL and "cap_ladder.json" in rep["gates"]["A5_at_cap"]["detail"]
    # A2 (identity at collection -> the leg refuses -> A0): another cap in the hashed identity
    rep = build("a2", manifest_edit=lambda m: m[C.MANIFEST_KEY_TINYCAP][C.IDENTITY_KEY_MAX_TOKENS_BY_TASK].update({V.TASK: 4096}))
    assert "max_tokens_by_task" in leg_of(rep, GLM, "main", CONST_PSR)["refused"] and rep["gates"]["A0_completeness"]["status"] == A.GATE_FAIL
    # A2 (per usable leg): a stamp that is not the class's
    rep = build("a2b", summary_edit=lambda s: s.update(resumed_policy="never"))
    assert rep["gates"]["A2_identity"]["status"] == A.GATE_FAIL and "stamps" in rep["gates"]["A2_identity"]["detail"]
    # A0: a missing leg that is NOT an expected refusal
    G.build_store(tmp_path / "a0", row(GLM, "main", NONE), correct=G.CORRECT_NONE, **common)
    rep = analyze(tmp_path / "a0", matrix=rows, overlap=None)
    assert rep["gates"]["A0_completeness"]["status"] == A.GATE_FAIL and V.leg_key(GLM, "main", CONST_PSR, 8192) in rep["unusable_legs"]
    assert not contrast_of(rep, GLM, "main", CONST_PSR)["computable"]
    # A7 pending without canonical ids (an operator input, never waived)
    G.build_all(tmp_path / "a7", rows=rows, **common)
    rep = analyze(tmp_path / "a7", matrix=rows, overlap=None, canonical_ids=None)
    assert rep["gates"]["A7_population"]["status"] == A.GATE_PENDING and rep["gates"]["pending_unwaivable"]
    assert analyze(tmp_path / "a7", matrix=rows, overlap=None, canonical_ids=None, allow_pending_gates=True)["verdict"]["word"] is None
    # a diagnostic store and a partial store refuse by name
    rep = build("diag", manifest_edit=lambda m: m[C.MANIFEST_KEY_TINYCAP].update({C.IDENTITY_KEY_DIAGNOSTIC: "smoke"}))
    assert "diagnostic" in leg_of(rep, GLM, "main", CONST_PSR)["refused"]
    # a wrong store pin refuses every leg before any estimand
    G.build_all(tmp_path / "pin", rows=rows, **common)
    rep = A.analyze(tmp_path / "pin", "abcdef0", matrix=rows, canonical_ids=G.IDS, sampling_rows=G.SAMPLING_ROWS, render_facts=G.RENDER_FACTS,
                    pins=G.pins_doc(ROWS), n_resamples=B_FAST, read_eval_logs=False, analysis_sha=G.FULL_SHA)
    assert len(rep["unusable_legs"]) == 2 and all("abcdef0" in why for why in rep["unusable_legs"].values()) and "estimands" not in rep


def test_a10_overlap_gate(full_report, full_store):
    rep = full_report
    # the overlap block per model: 200 pairs, a verdict, the ids digest bound to the canonical set
    for key in G.MODEL_KEYS:
        per = rep["gates"]["A10_overlap"]["models"][key]
        assert per["complete"] and per["n_pairs"] == 200 and per["verdict"] in ("continuation", "new_instrument")
    # without the record the gate FAILS; with a record over another id set it FAILS by name
    assert analyze(full_store.root, overlap=None)["gates"]["A10_overlap"]["status"] == A.GATE_FAIL
    other = dict(full_store.overlap)
    other["ids_sha256"] = "0" * 64
    g = A.gate_overlap(other, list(G.MODEL_KEYS), G.IDS, store_pin=G.SHA7)
    assert g["status"] == A.GATE_FAIL and "ids_sha256" in g["detail"]
    assert rep["gates"]["A10_overlap"]["store_pin"] == G.SHA7 == rep["gates"]["A10_overlap"]["report_store_pin"]
    assert all(per["producer_tree_sha"] == G.FULL_SHA and per["stock_store_pin"] == G.SHA7 for per in rep["gates"]["A10_overlap"]["models"].values())
    assert any("HELD" in f for f in rep["findings"]) and not any("refused by name" in f for f in rep["findings"])


# --------------------------------------------------------------------------- #
# a cell the DRIVER certified unservable: a named outcome only when            #
# every independent check holds — otherwise a missing store                    #
# --------------------------------------------------------------------------- #

NEMO_UNCLOSED_ATTEMPTS = [  # gsm8k-paired200 row 27 at 8f5f1495 (job 483711_27), verbatim in shape
    {"attempt": 1, "prompt_id": "arithmetic", "seed": 0, "max_tokens": 512, "finish_reason": "stop", "reasoning_chars": 0, "content_chars": 1103, "class": "unclosed_block"},
    {"attempt": 2, "prompt_id": "reasoning", "seed": 1, "max_tokens": 512, "finish_reason": "stop", "reasoning_chars": 0, "content_chars": 900, "class": "unclosed_block"},
    {"attempt": 3, "prompt_id": "reasoning", "seed": 2, "max_tokens": 512, "finish_reason": "stop", "reasoning_chars": 0, "content_chars": 113, "class": "unclosed_block"},
]
T_A1_START, T_EVIDENCE, T_A1_END, T_A2_START = "2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z", "2026-01-01T00:10:00Z", "2026-01-01T01:00:00Z"
#: the driver's one run as the manifest records it before its probes: started, never completed, nothing accepted
REFUSED_A1 = [("a1", False, T_A1_START)]


def _certified_refusal_evidence(family: str, *, header_ids=None, completion=None, attempts=None, certifying_class="unclosed_block") -> dict:
    """Predicate (A)'s REFUSAL record through the real `contracts.render_certified` (the driver writes `e.evidence`
    before aborting), the certifying draw's class beside it, and the recorded ON draws. `header_ids` / `completion`
    override the live-measured ids / the ON completion to build a NON-behavioural refusal."""
    fact = C.FAMILY_RENDER_FACTS[family]
    ids = header_ids if header_ids is not None else {"off": list(fact.off_ids), "on": list(fact.on_ids)}
    on = completion if completion is not None else {"content": "17 plus 26 is 43. 43.", "reasoning": None, "finish_reason": "stop"}
    with pytest.raises(C.RenderCertificationError) as e:
        C.render_certified(family=family, header_ids_by_arm=ids, measured_live=True,
                           probe_regimes={C.THINKING_ON: C.RENDER_REGIME_THINKING_ON, C.THINKING_OFF: C.RENDER_REGIME_THINKING_OFF},
                           on_completion=on)
    record = {**dict(e.value.evidence), **({"certifying_draw_class": certifying_class} if certifying_class is not None else {})}
    assert record["render_certified"] is False and record["certification_problems"]
    return {"render_certified": record,
            "render_probe": {"off": {"content": "43", "reasoning": None, "finish_reason": "stop", "render_regime": "thinking_off"},
                             "on": {**on, "render_regime": "thinking_on"},
                             "on_attempts": NEMO_UNCLOSED_ATTEMPTS if attempts is None else attempts}}


def _ladder_state(attempt_id="a1", *, result="DRIVER_FAILED", started=T_A1_START, ended=T_A1_END, sha=G.FULL_SHA, accepted=None) -> dict:
    return {"schema": "g8k_ladder_state/1", "accepted_generation_attempt": accepted,
            "attempts": [{"attempt_id": attempt_id, "kind": "generation", "started": started, "ended": ended, "rc": 2, "result": result,
                          "sha": sha, "job_id": "483711", "node": "TESTHOST"}]}


def _make_unservable(gen_root, r, *, evidence: dict | None, written_at: str = T_EVIDENCE, ladder_state: dict | None = "default",
                     attempts=REFUSED_A1, accepted=None, tree_sha=None, bites_edit=None):
    """Turn a built store into the driver-abort shape: the manifest stays but records the runs as the driver does
    before its probes (`attempts` = [(id, completed, started)], nothing accepted unless said), the summary marker
    goes, evidence.json is the certification refusal on THIS cell's route (or absent), the wrapper's ladder_state
    beside it. (The synthetic builder insists on one accepted run, so the refused shape is written here.)"""
    root = gen_root / r["out_root"]
    cd = next((root / "cells").glob("*"))
    completion.task_summary_path(cd, A.TASK).unlink()
    manifest = json.loads((cd / "manifest.json").read_text(encoding="utf-8"))
    sha = tree_sha or manifest[C.MANIFEST_KEY_PRODUCER]["tree_sha"]
    manifest[C.MANIFEST_KEY_RESUME_ATTEMPTS] = [
        {"ts": started, "started": started, "tree_sha": sha, "attempt": i + 1, "completed": bool(done), "attempt_id": aid,
         "kind": "generation" if done else None, "generation_calls": 200 if done else None, "ended": None, "terminal": None,
         "finalised_from_attempt": None}
        for i, (aid, done, started) in enumerate(attempts)]
    manifest[V.MANIFEST_KEY_ACCEPTED_ATTEMPT] = accepted
    (cd / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    for f in (cd / "evidence.json", root / "evidence.json", root / V.LADDER_STATE_FILENAME):
        if f.exists():
            f.unlink()
    if evidence is not None:
        spell = C.cell_spellings(r["cell"], declared=bool(r["declared"]))
        fitted = bool(r["fitted"])
        # the steering-bites control the driver ran BEFORE the render probe, exactly as it records it on a lens leg —
        # the ON-render declared probe on a DECLARED row, the OFF probe on an undeclared one
        bites = ((G.steering_bites_record_declared(r["cell"]) if r["declared"] else G.S.steering_bites_record(r["cell"])) if fitted
                 else {C.OOS_EVIDENCE_BITES_APPLICABLE: False, "reason": "no steering selector on this route"})
        if bites_edit:
            bites_edit(bites)
        base = {"driver": "served_capability_run_v2", "cell_id": r["cell"], "harness_cell_id": spell.harness, "virtual_model": spell.vmodel,
                "engine": "vllm", "mask": "generated" if fitted else None, "written_at": written_at, C.OOS_EVIDENCE_BITES_BLOCK: bites}
        (cd / "evidence.json").write_text(json.dumps({**base, **evidence}, indent=1), encoding="utf-8")
    if ladder_state is not None:
        (root / V.LADDER_STATE_FILENAME).write_text(json.dumps(_ladder_state() if ladder_state == "default" else ladder_state), encoding="utf-8")
    return cd


def _refused_wave(tmp_path, *, cell=CONST_PSR, evidence=None, attempts=REFUSED_A1, accepted=None, tree_sha=None, written_at=T_EVIDENCE,
                  ladder_state="default", manifest_edit=None, bites_edit=None, with_overlap=False, extra_rows=()):
    """The small wave: nemotron none + the refused cell + one normally built contrast (nemotron lora)."""
    triples = [(NEMO, "main", NONE), (NEMO, "main", CONST_LORA), (NEMO, "main", cell), *extra_rows]
    rows = small_matrix(*dict.fromkeys(triples))
    G.build_all(tmp_path / "gen", rows=rows, overrides={(NEMO, "main", cell): ({"manifest_edit": manifest_edit} if manifest_edit else {})})
    _make_unservable(tmp_path / "gen", row(NEMO, "main", cell),
                     evidence=_certified_refusal_evidence("nemotron_h") if evidence is None else evidence, written_at=written_at,
                     ladder_state=ladder_state, attempts=attempts, accepted=accepted, tree_sha=tree_sha, bites_edit=bites_edit)
    overlap = build_overlap(tmp_path / "ov") if with_overlap else None
    return rows, analyze(tmp_path / "gen", matrix=rows, overlap=overlap)


def test_a_driver_certified_unservable_cell_is_a_named_refused_outcome_and_a0_passes(tmp_path):
    """(a) manifest + no summary + evidence `render_certified: false` with three `unclosed_block` draws, the manifest's
    one run incomplete and unaccepted, the wrapper's ladder_state naming that run DRIVER_FAILED at the pin, the evidence
    written inside it -> `refused_certified`: A0 PASSES, the contrast is not computable and named (excluded from
    `_not_run`), the report is a production result with the cell filed by name; (c) an `--expected-refusals` leg with
    no store still files as `refused_expected`; (d) the refusal list and the findings carry the attempt classes and
    the NEUTRAL, route-naming observation."""
    expected_triple = G.expected_refusal_triples(ROWS)[0]
    expected = {k: v for k, v in G.expected_refusals_doc(ROWS).items() if k == V.leg_key(*expected_triple, 8192)}
    rows = small_matrix((NEMO, "main", NONE), (NEMO, "main", CONST_PSR), (NEMO, "main", CONST_LORA), (GEMMA, "main", NONE), expected_triple)
    G.build_all(tmp_path / "gen", rows=rows, skip=[expected_triple])
    cd = _make_unservable(tmp_path / "gen", row(NEMO, "main", CONST_PSR), evidence=_certified_refusal_evidence("nemotron_h"))
    assert completion.read_store_manifest(cd) is not None and completion.read_task_summary(cd, A.TASK) is None
    rep = analyze(tmp_path / "gen", matrix=rows, overlap=build_overlap(tmp_path / "ov"), expected_refusals=expected)
    leg = leg_of(rep, NEMO, "main", CONST_PSR)
    assert leg["certified_unservable"] is True and leg["missing"] is None
    assert leg["refused"].startswith(A.REFUSAL_CERTIFIED_UNSERVABLE + ": ") and "think shape" in leg["refused"]
    cert = leg["render_certification"]
    assert cert["admitted"] is True and cert["not_admitted_because"] is None and cert["refused_attempt"] == "a1"
    assert "evidence" not in cert and cert["evidence_written_at"] == T_EVIDENCE
    assert cert["attempt_classes"] == ["unclosed_block"] * 3 and cert["certifying_draw_class"] == "unclosed_block" and cert["n_unclosed"] == 3
    assert [a["content_chars"] for a in cert["on_attempts"]] == [1103, 900, 113]
    assert cert["observation"] == ("all 3 ON probe draws in this standup stopped without closing the thought block, on the route of the "
                                   f"cell's object ({CONST_PSR}); certification refused")
    assert "suppress" not in cert["observation"] and cert["fitted"] is True
    assert V.steering_bites_problems(json.loads((cd / "evidence.json").read_text())[C.OOS_EVIDENCE_BITES_BLOCK]) == []
    # A0: PASS, the leg a NAMED outcome beside the expected refusal, nothing unusable
    a0 = rep["gates"]["A0_completeness"]
    assert a0["status"] == A.GATE_PASS and rep["unusable_legs"] == {}
    assert a0["legs"][leg["leg"]]["status"] == A.LEG_STATUS_REFUSED_CERTIFIED == "refused_certified"
    assert a0["legs"][leg["leg"]]["attempt_classes"] == ["unclosed_block"] * 3 and a0["legs"][leg["leg"]]["expected_code"] is None
    assert a0["n_refused_certified"] == 1 and a0["n_refused_expected"] == 1 and a0["n_usable_base"] == 3
    # (c) the expected refusal is exactly as before
    exp_leg = V.leg_key(*expected_triple, 8192)
    assert a0["legs"][exp_leg]["status"] == "refused_expected" and leg_of(rep, *expected_triple)["missing"]
    assert {(x["leg"], x["code"], x["status"]) for x in rep["refused"]} == {
        (leg["leg"], A.REFUSAL_CERTIFIED_UNSERVABLE, "refused_certified"), (exp_leg, G.EXPECTED_REFUSAL_CODE, "refused_expected")}
    # the contrast: not computable, named, excluded from "did not run"; the verdict proceeds on the other cells
    c = contrast_of(rep, NEMO, "main", CONST_PSR)
    assert not c["computable"] and c["certified_refusal"] == A.REFUSAL_CERTIFIED_UNSERVABLE and c["expected_refusal"] is None
    assert A.REFUSAL_CERTIFIED_UNSERVABLE in c["reason"]
    assert A._not_run(rep["contrasts"]) == [] and A.production_refusals(rep) == [] and rep["verdict"]["word"] == "filed"
    assert rep["design"]["n_certified_unservable"] == 1 and rep["design"]["n_contrasts_computable"] == 1
    assert contrast_of(rep, NEMO, "main", CONST_LORA)["computable"] and "estimands" in rep
    assert next(x for x in rep["contrast_summary"] if x["cell"] == CONST_PSR)["certified_refusal"] == A.REFUSAL_CERTIFIED_UNSERVABLE
    # (d) the refusal record and the readout carry the classes and the neutral observation
    rec = next(x for x in rep["refused"] if x["status"] == "refused_certified")
    assert rec["attempt_classes"] == ["unclosed_block"] * 3 and rec["certifying_draw_class"] == "unclosed_block"
    assert rec["observation"].startswith("all 3 ON probe draws in this standup stopped without closing the thought block")
    assert "predicate (A)" in rec["reason"] and "think shape" in rec["reason"]
    line = next(f for f in rep["findings"] if f.startswith(leg["leg"] + ": refused by name"))
    assert f"({A.REFUSAL_CERTIFIED_UNSERVABLE})" in line and "['unclosed_block', 'unclosed_block', 'unclosed_block']" in line
    assert "certifying draw 'unclosed_block'" in line and "stopped without closing the thought block" in line and "suppress" not in line
    assert line in rep["top_items"] and not any("no estimand" in t for t in rep["top_items"])
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        A.print_table(rep)
    assert f"REFUSED by name: {leg['leg']} ({A.REFUSAL_CERTIFIED_UNSERVABLE}) — ON draws ['unclosed_block'" in buf.getvalue()
    # the class literal and the field name are the driver's own
    srun = _load("served_capability_run", REPO / "scripts" / "served_capability_run.py")
    assert A.PROBE_CLASS_UNCLOSED_BLOCK == srun.PROBE_CLASS_UNCLOSED_BLOCK and A.REFUSAL_CERTIFIED_UNSERVABLE == C.REC_THINKING_UNSERVABLE
    assert V.BEHAVIOURAL_REFUSAL_CLASSES == (srun.PROBE_CLASS_UNCLOSED_BLOCK,) and V.CERTIFYING_DRAW_CLASS_FIELD == srun.CERTIFYING_DRAW_CLASS_FIELD
    # without a ladder_state beside the store the manifest + evidence binding alone admits it
    _, rep2 = _refused_wave(tmp_path / "noladder", ladder_state=None)
    assert leg_of(rep2, NEMO, "main", CONST_PSR)["certified_unservable"] is True and rep2["gates"]["A0_completeness"]["status"] == A.GATE_PASS


def test_a_certified_refusal_on_the_bare_route_says_so(tmp_path):
    """The neutral wording names the BARE route when the refused cell is `none`; the view's contrasts then have no
    base and do not run (correctly: a refused base leg blocks its view, it is not a named contrast outcome)."""
    triples = [(NEMO, "main", NONE), (NEMO, "main", CONST_LORA)]
    rows = small_matrix(*triples)
    G.build_all(tmp_path / "gen", rows=rows)
    _make_unservable(tmp_path / "gen", row(NEMO, "main", NONE), evidence=_certified_refusal_evidence("nemotron_h"))
    rep = analyze(tmp_path / "gen", matrix=rows, overlap=None)
    leg = leg_of(rep, NEMO, "main", NONE)
    assert leg["certified_unservable"] is True and leg["render_certification"]["fitted"] is False
    assert leg["render_certification"]["observation"] == ("all 3 ON probe draws in this standup stopped without closing the thought block, on "
                                                          "the BARE route (no object); certification refused")
    assert rep["gates"]["A0_completeness"]["legs"][leg["leg"]]["status"] == "refused_certified"
    c = contrast_of(rep, NEMO, "main", CONST_LORA)
    assert not c["computable"] and c["certified_refusal"] is None and c in A._not_run(rep["contrasts"])
    assert any("did not run" in r for r in A.production_refusals(rep))


def test_a_certified_refusal_is_not_admitted_when_the_producer_or_the_binding_fails(tmp_path):
    """Review P1 reproductions — each stays `missing` (BLOCKING, A0 FAILS) with the certification named and the reason
    it is not admitted: a wrong producer sha; a newer incomplete attempt beside an older refusal's evidence; a
    ladder_state whose latest run is not the manifest's, or did not end DRIVER_FAILED, or ran at another sha; a last
    run that completed (torn); a rung leg."""
    def check(label, rep, needle):
        leg = leg_of(rep, NEMO, "main", CONST_PSR)
        assert leg["certified_unservable"] is False and leg["refused"] is None, label
        assert leg["missing"] and "filed no summary" in leg["missing"] and A.REFUSAL_CERTIFIED_UNSERVABLE in leg["missing"], label
        assert "NOT admitted as a certified refusal" in leg["missing"] and needle in leg["missing"], (label, leg["missing"])
        assert leg["render_certification"]["admitted"] is False and needle in leg["render_certification"]["not_admitted_because"], label
        a0 = rep["gates"]["A0_completeness"]
        assert a0["status"] == A.GATE_FAIL and leg["leg"] in rep["unusable_legs"] and a0["n_refused_certified"] == 0, label
        c = contrast_of(rep, NEMO, "main", CONST_PSR)
        assert not c["computable"] and c["certified_refusal"] is None and c in A._not_run(rep["contrasts"]), label
        assert rep["verdict"]["word"] is None and any("did not run" in r for r in A.production_refusals(rep)), label

    # P1: the wrong producer sha (every attempt's tree_sha moves with it, as a store from another tree would carry)
    def other_tree(m):
        m[C.MANIFEST_KEY_PRODUCER]["tree_sha"] = "f" * 40
    _, rep = _refused_wave(tmp_path / "sha", manifest_edit=other_tree, tree_sha="f" * 40, ladder_state=_ladder_state(sha="f" * 40))
    check("wrong producer sha", rep, "does not carry the wave pin")
    # P1: a newer incomplete attempt (a resume that died before rewriting the evidence) beside the older refusal's evidence
    _, rep = _refused_wave(tmp_path / "newer", attempts=[("a1", False, T_A1_START), ("a2", False, T_A2_START)],
                           written_at=T_EVIDENCE, ladder_state=_ladder_state("a2", started=T_A2_START, ended=None))
    check("newer attempt", rep, "predates the last recorded attempt 'a2'")
    # the wrapper's ledger disagrees with the manifest: a run the driver never recorded
    _, rep = _refused_wave(tmp_path / "ledger", ladder_state=_ladder_state("a2", started=T_A2_START, ended=None))
    check("ladder_state newer", rep, "latest attempt 'a2' != the manifest's latest 'a1'")
    _, rep = _refused_wave(tmp_path / "result", ladder_state=_ladder_state(result="COMPLETE"))
    check("ladder_state result", rep, "ended 'COMPLETE', not 'DRIVER_FAILED'")
    _, rep = _refused_wave(tmp_path / "lsha", ladder_state=_ladder_state(sha="e" * 40))
    check("ladder_state sha", rep, "not the wave pin")
    _, rep = _refused_wave(tmp_path / "late", written_at="2026-01-01T00:20:00Z")
    check("evidence after the end", rep, "after the wrapper recorded attempt 'a1' ending")
    # torn: the last run COMPLETED (a completed run writes a summary) / an accepted attempt named
    _, rep = _refused_wave(tmp_path / "torn", attempts=[("a1", True, T_A1_START)], accepted="a1",
                           ladder_state=_ladder_state(result="COMPLETE", accepted="a1"))
    check("completed last run", rep, "COMPLETED")
    # the manifest's model pin / object pin still hold on the refusal path
    _, rep = _refused_wave(tmp_path / "pin", manifest_edit=lambda m: m.__setitem__("model_revision", "deadbeef"))
    check("model pin", rep, "model_revision")
    _, rep = _refused_wave(tmp_path / "obj", manifest_edit=lambda m: m[C.MANIFEST_KEY_PROVENANCE].__setitem__("artifact_sha256", "0" * 64))
    check("object pin", rep, "another object was served")


def test_a_certified_refusal_is_not_admitted_unless_the_non_behavioural_predicates_hold(tmp_path):
    """Review P2 reproductions — `render_certified: false` alone is not enough: wrong live header ids with a valid
    think-shape completion (the SERVING PATH failed) blocks; so do a regime stamp that is not its arm's, a draw of
    another class among the attempts, a certifying draw outside the behavioural set, no recorded draws, and evidence
    naming another cell's route."""
    def check(label, rep, needle):
        leg = leg_of(rep, NEMO, "main", CONST_PSR)
        assert leg["certified_unservable"] is False and leg["missing"] and "NOT admitted" in leg["missing"] and needle in leg["missing"], (label, leg["missing"])
        assert rep["gates"]["A0_completeness"]["status"] == A.GATE_FAIL and leg["leg"] in rep["unusable_legs"], label

    fact = C.FAMILY_RENDER_FACTS["nemotron_h"]
    wrong_ids = {"off": list(fact.off_ids), "on": list(fact.off_ids)}                         # the ON header measured as OFF's
    shaped = {"content": "Okay, 17 + 26 = 43.\n</think>\n43", "reasoning": None, "finish_reason": "stop"}
    _, rep = _refused_wave(tmp_path / "hdr", evidence=_certified_refusal_evidence("nemotron_h", header_ids=wrong_ids, completion=shaped))
    check("wrong header ids + think_shape", rep, "the SERVING PATH failed certification")
    bad_regime = _certified_refusal_evidence("nemotron_h")
    bad_regime["render_probe"]["on"]["render_regime"] = "thinking_off"
    _, rep = _refused_wave(tmp_path / "regime", evidence=bad_regime)
    check("regime stamp", rep, "render_regime stamps")
    mixed = _certified_refusal_evidence("nemotron_h", attempts=[*NEMO_UNCLOSED_ATTEMPTS[:2], {**NEMO_UNCLOSED_ATTEMPTS[2], "class": "no_thinking_emitted"}])
    _, rep = _refused_wave(tmp_path / "mixed", evidence=mixed)
    check("a draw of another class", rep, "outside the behavioural set")
    _, rep = _refused_wave(tmp_path / "cdc", evidence=_certified_refusal_evidence("nemotron_h", certifying_class="no_think_shape"))
    check("certifying class", rep, "certifying_draw_class 'no_think_shape'")
    _, rep = _refused_wave(tmp_path / "nodraws", evidence=_certified_refusal_evidence("nemotron_h", attempts=[]))
    check("no draws", rep, "records no ON draw")
    other_cell = _certified_refusal_evidence("nemotron_h")
    other_cell["virtual_model"] = C.cell_spellings(CONST_LORA, declared=False).vmodel
    _, rep = _refused_wave(tmp_path / "route", evidence=other_cell)
    check("another cell's route", rep, "virtual_model")
    # a FITTED run whose steering-bites control FAILED (recorded before the render probe) is an
    # independent serving failure the behavioural refusal must not hide — not admitted, the bites failure named
    def failed_bites(b):
        b[C.OOS_EVIDENCE_BITES_VERIFIED] = False
        b[C.OOS_EVIDENCE_BITES_ERROR] = "steering_bites: arm(s) ['psr_ce_all'] produce NO difference between the serving scale and scale 0"
        b["stamp_error"] = "probe leg 'steered' stamp triple is not the expected row (resumed_policy 'never' != 'always')"
        b[C.OOS_EVIDENCE_BITES_N_DIFFERING] = 0
    _, rep = _refused_wave(tmp_path / "bites", bites_edit=failed_bites)
    check("failed bites control", rep, "the steering-bites control the producer ran on this fitted route FAILED")
    leg = leg_of(rep, NEMO, "main", CONST_PSR)
    why = leg["render_certification"]["not_admitted_because"]
    assert "not verified" in why and "carries error" in why and "carries stamp_error" in why and "n_differing 0 < 1" in why
    assert rep["gates"]["A3_certification"]["status"] == A.GATE_PASS      # A3 reads usable legs only; A0 is what blocks
    # ... a verified control with no differing probe, or the wrong twin, blocks the same way
    _, rep = _refused_wave(tmp_path / "nodiff", bites_edit=lambda b: b.__setitem__(C.OOS_EVIDENCE_BITES_N_DIFFERING, 0))
    check("no differing probe", rep, "n_differing 0 < 1")
    _, rep = _refused_wave(tmp_path / "twin", bites_edit=lambda b: b.__setitem__(C.OOS_EVIDENCE_BITES_TWIN, "prompted__safety__lens0"))
    check("wrong twin", rep, "twin 'prompted__safety__lens0' != 'none__lens0'")
    _, rep = _refused_wave(tmp_path / "counts", bites_edit=lambda b: b.__setitem__(C.OOS_EVIDENCE_BITES_N_DIFFERING, 99))
    check("inconsistent counts", rep, "the counts do not support each other")
    # the helper is the ordinary certification's reading too: the same record passes both
    assert V.steering_bites_problems(G.S.steering_bites_record(CONST_PSR)) == []
    # evidence written before the certifying-class field (8f5f1495's predecessor) is still admitted: the class is absent
    _, rep = _refused_wave(tmp_path / "nocdc", evidence=_certified_refusal_evidence("nemotron_h", certifying_class=None))
    leg = leg_of(rep, NEMO, "main", CONST_PSR)
    assert leg["certified_unservable"] is True and leg["render_certification"]["certifying_draw_class"] is None


def test_a_summary_less_store_without_a_certified_refusal_stays_missing_and_a0_fails(tmp_path):
    """(b) manifest + no summary + NO evidence (or evidence that does not say `render_certified: false`) -> `missing`,
    A0 FAILS, the contrast "did not run" — exactly as before."""
    rows = small_matrix((NEMO, "main", NONE), (NEMO, "main", CONST_PSR))
    for label, evidence in (("none", None),
                            ("certified_true", {"render_certified": {"render_certified": True, "certification_problems": []},
                                                "render_probe": {"on_attempts": [{"class": "think_shape"}]}}),
                            ("no_flag", {"render_certified": {}, "render_probe": {"on_attempts": NEMO_UNCLOSED_ATTEMPTS}})):
        gen = tmp_path / label
        G.build_all(gen, rows=rows)
        _make_unservable(gen, row(NEMO, "main", CONST_PSR), evidence=evidence)
        rep = analyze(gen, matrix=rows, overlap=None)
        leg = leg_of(rep, NEMO, "main", CONST_PSR)
        assert leg["missing"] and "filed no summary" in leg["missing"] and leg["refused"] is None and leg["certified_unservable"] is False, label
        assert "NOT admitted" not in leg["missing"] and "render_certification" not in leg, label
        a0 = rep["gates"]["A0_completeness"]
        assert a0["status"] == A.GATE_FAIL and leg["leg"] in rep["unusable_legs"] and a0["n_refused_certified"] == 0, label
        assert a0["legs"][leg["leg"]]["status"] == "missing" and rep["refused"] == [], label
        c = contrast_of(rep, NEMO, "main", CONST_PSR)
        assert not c["computable"] and c["certified_refusal"] is None and c in A._not_run(rep["contrasts"]), label
        assert any("did not run" in r for r in A.production_refusals(rep)) and rep["verdict"]["word"] is None, label
    # a malformed evidence.json certifies nothing either
    gen = tmp_path / "malformed"
    G.build_all(gen, rows=rows)
    cd = _make_unservable(gen, row(NEMO, "main", CONST_PSR), evidence=None)
    (cd / "evidence.json").write_text("{not json", encoding="utf-8")
    rep = analyze(gen, matrix=rows, overlap=None)
    assert leg_of(rep, NEMO, "main", CONST_PSR)["missing"] and rep["gates"]["A0_completeness"]["status"] == A.GATE_FAIL


def test_main_refuses_a_non_production_report_with_exit_3_and_a_differing_match_with_exit_4(tmp_path, monkeypatch):
    # C1 (2026-09-12): a PSR cell is out of every figure by ruling, so a render needs a non-PSR arm to draw — the LoRA cell
    # keeps this test about the exit codes (the PSR cell stays in the report and the tables; the figures never see it)
    rows = small_matrix((NEMO, "main", NONE), (NEMO, "main", CONST_PSR), (NEMO, "main", CONST_LORA))
    G.build_all(tmp_path / "gen", rows=rows)
    matrix = G.write_matrix(tmp_path / "matrix.json", rows)
    ids = G.write_json(tmp_path / "ids.json", G.IDS)
    pins = G.write_json(tmp_path / "pins.json", G.pins_doc(ROWS))
    srows = G.write_json(tmp_path / "rows.json", G.SAMPLING_ROWS)
    facts = G.write_json(tmp_path / "facts.json", G.RENDER_FACTS)
    overlap = G.write_json(tmp_path / "overlap.json", build_overlap(tmp_path / "ov"))
    common = ["--gen-root", str(tmp_path / "gen"), "--sha", G.SHA7, "--matrix", str(matrix), "--ids", str(ids), "--pins", str(pins),
              "--sampling-rows", str(srows), "--render-facts", str(facts), "--bootstrap", str(B_FAST), "--no-eval-logs", "--wandb-mode", "disabled",
              "--no-mirror", "--analysis-sha", G.FULL_SHA, "--repo-root", str(REPO)]
    monkeypatch.setitem(sys.modules, "wandb", None)      # importing wandb must not be needed under --wandb-mode disabled
    # no overlap record -> A10 fails -> exit 3, the report written, no figures
    rc = A.main([*common, "--out", str(tmp_path / "r0.json"), "--figs-dir", str(tmp_path / "figs0")])
    assert rc == A.EXIT_NOT_PRODUCTION and (tmp_path / "r0.json").is_file() and not (tmp_path / "figs0").exists()
    assert A.gate_report(tmp_path / "r0.json") == A.EXIT_NOT_PRODUCTION
    # a production run writes the report, the figures and the sidecars
    pytest.importorskip("matplotlib")
    rc = A.main([*common, "--overlap", str(overlap), "--out", str(tmp_path / "r1.json"), "--figs-dir", str(tmp_path / "figs1")])
    assert rc == 0 and A.gate_report(tmp_path / "r1.json") == 0
    r1 = json.loads((tmp_path / "r1.json").read_text())
    assert set(r1["figures"]["written"]) == set(A.figures_for(r1)) and (tmp_path / "figs1" / A.NUMBERS_FILENAME).is_file()
    # C1: the publication gate runs on a published-shape render too — PASS, the PSR cell recorded as excluded by ruling, and
    # absent from every plotted record; the report and the tables JSON still carry it
    assert r1[A.PUBLICATION_GATE_KEY]["status"] == "PASS" and r1[A.PUBLICATION_GATE_KEY]["excluded_arms"] == [CONST_PSR]
    assert r1["figures"][A.FIGURES_EXCLUDED_ARMS_KEY] == [CONST_PSR]
    numbers1 = json.loads((tmp_path / "figs1" / A.NUMBERS_FILENAME).read_text())
    assert not [r for recs in numbers1["figures"].values() for r in recs if CONST_PSR in r["name"] or CONST_PSR in r["source"]]
    assert any(c["cell"] == CONST_PSR for c in r1["controls"]) and V.cell_key(NEMO, "main", CONST_PSR) in r1["cells"]
    # the match: identical -> 0 and the A9 gate compared; a report whose numbers differ -> exit 4 before any figure
    rc = A.main([*common, "--overlap", str(overlap), "--out", str(tmp_path / "r2.json"), "--no-figs", "--require-report-match", str(tmp_path / "r1.json")])
    assert rc == 0 and json.loads((tmp_path / "r2.json").read_text())["gates"]["A9_report_match"]["compared"] is True
    assert A.diff_reports(tmp_path / "r1.json", tmp_path / "r2.json") == 0
    r1["bootstrap"]["B"] = B_FAST + 1
    (tmp_path / "r1b.json").write_text(json.dumps(r1))
    rc = A.main([*common, "--overlap", str(overlap), "--out", str(tmp_path / "r3.json"), "--figs-dir", str(tmp_path / "figs3"), "--require-report-match", str(tmp_path / "r1b.json")])
    assert rc == A.EXIT_REPORTS_DIFFER and not (tmp_path / "figs3").exists()
    r3 = json.loads((tmp_path / "r3.json").read_text())
    assert r3["gates"]["A9_report_match"]["status"] == A.GATE_FAIL
    # A-P2.1: the saved mismatch report reads as what it is — the aggregate, the verdict and the top items re-derived from the
    # gate RECORDS after A9 was filed, and --gate-report refuses it (it used to print "all passed" over gates.failed == [])
    assert r3["gates"]["failed"] == ["A9_report_match"] and r3["verdict"]["word"] is None and "A9_report_match" in r3["verdict"]["reason"]
    assert r3["top_items"][0].startswith("A9_report_match: FAILED") and A.gate_report(tmp_path / "r3.json") == A.EXIT_NOT_PRODUCTION
    assert any("A9_report_match" in r for r in A.production_refusals(r3))
    # a report whose aggregate was never refreshed (the pre-fix shape) is still refused, from the records, and named as torn
    torn = json.loads((tmp_path / "r3.json").read_text())
    torn["gates"]["failed"] = []
    refusals = A.production_refusals(torn)
    assert any("A9_report_match" in r for r in refusals) and any("torn" in r for r in refusals)
    assert A.diff_reports(tmp_path / "r1b.json", tmp_path / "r2.json") == A.EXIT_REPORTS_DIFFER


# --------------------------------------------------------------------------- #
# the two-venv shape, the figures, W&B and the mirror                          #
# --------------------------------------------------------------------------- #


def test_two_venv_shape_the_log_reading_run_and_the_figure_run_agree_on_every_numeric_leaf(full_store, full_report):
    rep_f = analyze(full_store.root, overlap=full_store.overlap, read_eval_logs=False)
    d = A.numeric_leaf_diff(full_report, rep_f)
    assert d["equal"], (d["differing"][:5], d["missing"][:5], d["added"][:5])
    assert d["shared_numeric"] > 5000
    if HAS_INSPECT:
        base = full_report["legs"][full_report["cells"][V.cell_key(*ESC)]["base_leg"]]
        assert base["eval_check"]["available"] and base["eval_check"]["ladder_derivation"]["complete"]
        assert base["at_cap"]["split_source"] == A.SPLIT_SOURCE_EVAL_LOG and base["at_cap"]["n_loop"] == 2
        assert rep_f["legs"][base["leg"]]["at_cap"]["split_source"] == A.SPLIT_SOURCE_STORE_ROWS
    assert set(A.REPORT_DIFF_SKIP) == {"eval_check", "figures", "mirrored", "wandb_url", "analysis_code", "report_match"}
    runner = SITE.expand_path("$GRADE/tinycap/gsm8k/analysis_gsm8k.sh")
    if runner.is_file():
        text = runner.read_text()
        for needle in ("tinycap-laneB/bin/python", "tiny-capability-lmeval/bin/python", "--gate-report", "--require-report-match", "--no-eval-logs",
                       "--diff-reports", "cmp -s", "DONE", "tinycap_gsm8k_paired200.py", "gsm8k_overlap_check.py", "gsm8k_paired200_selection_ids"):
            assert needle in text, needle


@pytest.fixture(scope="module")
def full_figures(full_report, tmp_path_factory):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    figs_dir = tmp_path_factory.mktemp("g8k_figs")
    return figs_dir, A.make_figures(full_report, figs_dir)


def test_figures_render_with_short_titles_tall_axes_and_a_resolvable_sidecar(full_report, full_figures):
    _figs_dir, figs = full_figures
    stems = set(A.figures_for(full_report))
    assert stems == set(figs["written"]) and {f"final_g8k_contrasts_{m}" for m in G.MODEL_KEYS} <= stems
    assert {"final_g8k_accuracy_by_cell", "final_g8k_atcap_share", "final_g8k_truncation_cost", "final_g8k_atcap_classes", "final_g8k_overlap",
            "final_g8k_ledger", "final_g8k_completion_tokens"} <= stems and len(stems) == 11
    assert set(A.figure_captions(full_report)) == stems
    for paths in figs["written"].values():
        assert [Path(p).suffix for p in paths] == [".png", ".pdf"] and all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in paths)
    numbers = json.loads(Path(figs["numbers"]).read_text())
    n_records = 0
    for stem, recs in numbers["figures"].items():
        for rec in recs:
            n_records += 1
            if rec["transform"] is None:
                assert M.T._same_leaf(rec["value"], A.report_path(full_report, rec["source"])), (stem, rec["name"])
    assert n_records > 200
    import matplotlib.pyplot as plt
    for stem, fn in A.figures_for(full_report).items():
        fig, _nums = fn(full_report)
        fig.canvas.draw()
        title = [t for t in fig.texts if t.get_fontweight() == "bold"]
        assert len(title) == 1 and title[0].get_text().count("\n") <= 1, stem
        assert all(len(line) <= 90 for line in title[0].get_text().split("\n")), (stem, title[0].get_text())
        assert A.axes_height_fraction(fig) >= 0.80, (stem, A.axes_height_fraction(fig))
        plt.close(fig)
    with pytest.raises(A.AnalysisError, match="title"):
        A.check_title("a\nb\nc")


def test_wandb_run_name_figure_keys_summary_and_the_mirror(full_report, full_figures, tmp_path, monkeypatch):
    calls: dict = {}

    class _Summary:
        @staticmethod
        def update(d):
            calls["summary"] = dict(d)

    class _Run:
        url = "https://wandb.test/run"
        summary = _Summary()

        def log_artifact(self, art):
            calls["artifact"] = art.name

        def finish(self):
            calls["finished"] = True

    class _Artifact:
        def __init__(self, name, type):
            self.name = name

        def add_file(self, path, name=None):
            calls.setdefault("artifact_files", []).append((Path(path).name, name))

    fake = SimpleNamespace(init=lambda **kw: calls.setdefault("init", kw) and _Run(), Image=lambda p: p, Artifact=_Artifact,
                           log=lambda d: calls.setdefault("logged", []).append(d))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("SISTEER_WANDB_PROJECT_TINYCAP", "example-tinycap")        # the purpose's project, as a site names it
    png = tmp_path / "figs" / "final_g8k_overlap.png"
    png.parent.mkdir()
    png.write_bytes(b"png")
    (tmp_path / "figs" / f"final_g8k_contrasts_{GLM}.png").write_bytes(b"stale")     # a stray file beside the manifest's: never uploaded
    url = A.log_wandb(full_report, {"written": {"final_g8k_overlap": [str(png)]}}, mode="offline")
    assert url == "https://wandb.test/run" and calls["finished"]
    assert calls["init"]["name"] == f"gsm8k-paired200-{G.SHA7}" and calls["init"]["project"] == "example-tinycap"
    assert A.WANDB_PURPOSE == "tinycap"
    assert calls["artifact"] == f"figures-gsm8k-paired200-{G.SHA7}" and calls["artifact_files"] == [(png.name, png.name)]
    assert calls["init"]["config"][f"figures_{A.FIGURES_EXCLUDED_KEY}"] == []
    assert calls["logged"] == [{"g8k/final_g8k_overlap": str(png)}]
    s = calls["summary"]
    assert s[f"{GLM}/main/{CONST_PSR}/merged/stock/primary/point"] == pytest.approx(-0.10)
    assert s[f"{GLM}/main/{CONST_PSR}/r8192/W/score_as_is/point"] == pytest.approx(-0.10)
    # nemotron's none: 170 correct of 198 scored (its two at-cap draws are unscored and were not among the correct items)
    assert s[f"{NEMO}/main/{CONST_PSR}/merged/stock/primary/point"] == pytest.approx(0.75 - 170 / 198)
    assert s["verdict"] == full_report["verdict"]["verdict"] and s["gates_failed"] == 0
    assert A.log_wandb(full_report, {}, mode="disabled") is None
    figs_dir, figs = full_figures
    mirrored = A.mirror_figures(figs_dir, tmp_path / "mirror", manifest=figs)
    want = sorted(Path(p).name for ps in figs["written"].values() for p in ps) + sorted([Path(figs["numbers"]).name, Path(figs["tables"]).name])
    assert sorted(Path(p).name for p in mirrored) == sorted(want)


def test_figure_exclude_family_on_the_cli_records_the_exclusion_and_moves_no_numeric_leaf(tmp_path, monkeypatch):
    """``--figure-exclude-family GLM`` end to end: the run's report carries ``figures.excluded_families``, matches the no-flag
    run under ``--require-report-match`` (A9 PASSED), every data block is equal, only the GLM forest is missing from the
    figure set, and an unknown family is a refusal by name."""
    pytest.importorskip("matplotlib")
    # C1 (2026-09-12): the PSR cells are out of every figure by ruling, so each family carries a LoRA cell for the figures to draw
    rows = small_matrix((NEMO, "main", NONE), (NEMO, "main", CONST_PSR), (NEMO, "main", CONST_LORA),
                        (GLM, "main", NONE), (GLM, "main", CTL_PSR), (GLM, "main", CTL_LORA))
    G.build_all(tmp_path / "gen", rows=rows)
    matrix = G.write_matrix(tmp_path / "matrix.json", rows)
    ids = G.write_json(tmp_path / "ids.json", G.IDS)
    pins = G.write_json(tmp_path / "pins.json", G.pins_doc(ROWS))
    srows = G.write_json(tmp_path / "rows.json", G.SAMPLING_ROWS)
    facts = G.write_json(tmp_path / "facts.json", G.RENDER_FACTS)
    overlap = G.write_json(tmp_path / "overlap.json", build_overlap(tmp_path / "ov"))
    common = ["--gen-root", str(tmp_path / "gen"), "--sha", G.SHA7, "--matrix", str(matrix), "--ids", str(ids), "--pins", str(pins),
              "--sampling-rows", str(srows), "--render-facts", str(facts), "--overlap", str(overlap), "--bootstrap", str(B_FAST), "--no-eval-logs",
              "--wandb-mode", "disabled", "--no-mirror", "--analysis-sha", G.FULL_SHA, "--repo-root", str(REPO)]
    monkeypatch.setitem(sys.modules, "wandb", None)
    out_all, out_no = tmp_path / "all.json", tmp_path / "noglm.json"
    assert A.main([*common, "--out", str(out_all), "--figs-dir", str(tmp_path / "figs_all")]) == 0
    assert A.main([*common, "--out", str(out_no), "--figs-dir", str(tmp_path / "figs_noglm"), "--figure-exclude-family", GLM,
                   "--require-report-match", str(out_all)]) == 0
    r_all, r_no = (json.loads(f.read_text()) for f in (out_all, out_no))
    assert A.FIGURES_EXCLUDED_KEY not in r_all["figures"] and r_no["figures"][A.FIGURES_EXCLUDED_KEY] == [GLM] and r_no["figures"][A.FAMILIES_SHOWN_KEY] == [NEMO]
    # C1: both renders record the PSR cells as excluded by ruling and pass the gate; the family flag is the only difference
    assert r_all["figures"][A.FIGURES_EXCLUDED_ARMS_KEY] == r_no["figures"][A.FIGURES_EXCLUDED_ARMS_KEY] == sorted([CONST_PSR, CTL_PSR])
    assert r_all[A.PUBLICATION_GATE_KEY]["status"] == r_no[A.PUBLICATION_GATE_KEY]["status"] == "PASS"
    assert r_no["report_match"]["equal"] is True and r_no["gates"]["A9_report_match"]["status"] == A.GATE_PASS and r_no["gates"]["A9_report_match"]["compared"]
    assert A.numeric_leaf_diff(r_all, r_no)["equal"] and A.diff_reports(out_all, out_no) == 0 and A.gate_report(out_no) == 0
    for key in ("design", "cells", "legs", "contrasts", "contrast_summary", "controls", "truncation", "truncation_cost", "noise_floor", "held", "refused", "verdict"):
        assert r_all[key] == r_no[key], key
    assert {k: v for k, v in r_all["gates"].items() if k != "A9_report_match"} == {k: v for k, v in r_no["gates"].items() if k != "A9_report_match"}
    assert GLM in r_no["design"]["models"] and any(c["model_key"] == GLM for c in r_no["contrasts"])
    assert set(r_all["figures"]["written"]) - set(r_no["figures"]["written"]) == {f"final_g8k_contrasts_{GLM}"}
    numbers = json.loads(Path(r_no["figures"]["numbers"]).read_text())
    assert numbers[A.FIGURES_EXCLUDED_KEY] == [GLM] and not [r for recs in numbers["figures"].values() for r in recs if GLM in r["name"]]
    assert (tmp_path / "figs_all" / f"final_g8k_contrasts_{GLM}.png").is_file() and not (tmp_path / "figs_noglm" / f"final_g8k_contrasts_{GLM}.png").exists()
    with pytest.raises(A.AnalysisError, match="not a model key"):
        A.main([*common, "--out", str(tmp_path / "x.json"), "--figs-dir", str(tmp_path / "figs_x"), "--figure-exclude-family", "nope"])


def test_a_figures_dir_holds_one_exclusion_set_and_the_wandb_artifact_is_the_manifest(full_report, tmp_path, monkeypatch):
    """Review of 2026-09-11: a re-used --figs-dir kept the previous render's GLM forest on disk and a directory upload would have
    shipped it under a config declaring GLM excluded. A no-flag render followed by an exclusion render into the same directory
    (and the reverse) refuses by name; the same set re-renders in place (the runner's E then E-publish); stale figure files with
    no sidecar refuse too; the mirror is held to the same rule; and the W&B artifact is exactly the manifest's files."""
    pytest.importorskip("matplotlib")
    d = tmp_path / "d"
    A.make_figures(full_report, d)
    assert (d / f"final_g8k_contrasts_{GLM}.png").is_file()
    with pytest.raises(A.AnalysisError, match=A.FIGURES_EXCLUDED_KEY):
        A.make_figures(full_report, d, exclude_families=[GLM])
    assert (d / f"final_g8k_contrasts_{GLM}.png").is_file() and A.FIGURES_EXCLUDED_KEY not in json.loads((d / A.NUMBERS_FILENAME).read_text())
    d2 = tmp_path / "d2"
    figs = A.make_figures(full_report, d2, exclude_families=[GLM])
    with pytest.raises(A.AnalysisError, match=A.FIGURES_EXCLUDED_KEY):
        A.make_figures(full_report, d2)
    figs2 = A.make_figures(full_report, d2, exclude_families=[GLM])
    assert figs2["written"].keys() == figs["written"].keys() and not (d2 / f"final_g8k_contrasts_{GLM}.png").exists()
    d3 = tmp_path / "d3"
    A.make_figures(full_report, d3)
    (d3 / A.NUMBERS_FILENAME).unlink()
    with pytest.raises(A.AnalysisError, match="would not write"):
        A.make_figures(full_report, d3, exclude_families=[GLM])
    # the mirror: a directory holding the no-flag sidecar refuses an exclusion render's files, and the other way round
    # (C1: every render also carries the PSR ruling's arms, so the guard is called with the render's own `arms` as main() does)
    arms = figs[A.FIGURES_EXCLUDED_ARMS_KEY]
    assert arms and all(A.PG.is_psr_cell(c) for c in arms)
    with pytest.raises(A.AnalysisError, match=A.FIGURES_EXCLUDED_KEY):
        A.refuse_mixed_render(d, [GLM], list(figs["written"]), arms=arms)
    with pytest.raises(A.AnalysisError, match=A.FIGURES_EXCLUDED_KEY):
        A.refuse_mixed_render(d2, [], list(figs["written"]) + [f"final_g8k_contrasts_{GLM}"], arms=arms)
    A.refuse_mixed_render(d2, [GLM], list(figs["written"]), arms=arms)
    with pytest.raises(A.AnalysisError, match=A.FIGURES_EXCLUDED_ARMS_KEY):
        A.refuse_mixed_render(d2, [GLM], list(figs["written"]))          # a render that forgot the ruling's arms is a mixed render too
    A.refuse_mixed_render(tmp_path / "absent", [GLM], [], arms=arms)
    # the W&B artifact is the manifest's files: a stray forest copied into the directory is not uploaded
    shutil.copy(d / f"final_g8k_contrasts_{GLM}.png", d2 / f"final_g8k_contrasts_{GLM}.png")
    calls: dict = {}

    class _Artifact:
        def __init__(self, name, type):
            self.name = name

        def add_file(self, path, name=None):
            calls.setdefault("files", []).append(Path(path).name)

    run = SimpleNamespace(url="u", summary=SimpleNamespace(update=lambda d: None), log_artifact=lambda art: calls.setdefault("art", art.name), finish=lambda: None)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kw: calls.setdefault("init", kw) and run, Image=lambda p: p, Artifact=_Artifact, log=lambda d: None))
    A.log_wandb(full_report, figs, mode="offline")
    want = sorted(Path(p).name for ps in figs["written"].values() for p in ps) + sorted([A.NUMBERS_FILENAME, A.TABLES_FILENAME])
    assert sorted(calls["files"]) == sorted(want) and f"final_g8k_contrasts_{GLM}.png" not in calls["files"]
    assert calls["init"]["config"][f"figures_{A.FIGURES_EXCLUDED_KEY}"] == [GLM]


# --------------------------------------------------------------------------- #
# the overlap check                                                            #
# --------------------------------------------------------------------------- #


def test_overlap_check_against_a_hand_computed_2x2_and_a_199_pair_refusal(tmp_path, capsys):
    retired = tmp_path / "retired"
    # retired: items 0..179 correct (acc 0.90); stock: 0..177 + {181, 182} correct (acc 0.90); b = stock-only {181, 182}, c = retired-only {178, 179}
    G.build_retired_store(retired, QWEN, correct=set(range(180)), extraction_failed={3})
    G.build_retired_store(retired, f"{QWEN}_it", correct=set(range(179)))
    G.build_overlap_stock_store(tmp_path, QWEN, correct=set(range(178)) | {181, 182})
    rep = O.run(retired_root=retired, stock_root=tmp_path / "overlap", models=[QWEN], canonical_ids=G.IDS, ids_path=None, pairing_rows=G.pairing_rows(),
                pairing_source="synthetic", repo_root=REPO, store_pin=G.SHA7, n_resamples=2000, seed=0, ci=0.95, eval_logs=False)
    b = rep["models"][QWEN]
    assert b["usable"] and b["verdict"] == "continuation", b["reason"]
    cmp_ = b["comparison"]
    assert cmp_["n_pairs"] == 200 and cmp_["complete"] and cmp_["acc_retired"] == 0.90 and cmp_["acc_stock"] == 0.90 and cmp_["delta"] == 0.0
    assert cmp_["n_stock_correct_retired_wrong"] == 2 and cmp_["n_stock_wrong_retired_correct"] == 2 and cmp_["disagreement_b_plus_c"] == 4
    assert cmp_["n_both_correct"] == 178 and cmp_["n_both_wrong"] == 18
    from sisteer.eval.realign_stats import mcnemar
    ref = mcnemar({i: 1 if i in set(range(178)) | {181, 182} else 0 for i in range(200)}, {i: 1 if i < 180 else 0 for i in range(200)})
    assert cmp_["mcnemar"]["p_value"] == ref["p_value"] and cmp_["mcnemar"]["p_value"] >= 0.99 and cmp_["mcnemar"]["n_discordant"] == 4
    assert cmp_["ci_low"] > -0.05 and cmp_["ci_high"] < 0.05 and cmp_["ci_inside_tolerance"] and "power" in cmp_["power_note"]
    assert b["retired"]["n_extraction_failed"] == 1 and b["retired"]["gen"] == {"batch_size": 8, "do_sample": False, "max_new_tokens": 512}
    assert b["retired_id_match"]["n_matched_exact"] == 200 and b["retired_id_match"]["n_content_mismatch"] == 0
    assert b["it_view_retired"]["agreement_with_main"]["n_disagree"] == 1 and b["it_view_retired"]["agreement_with_main"]["n_common"] == 200
    assert rep["pairing"]["n_canonical_in_selection"] == 200 and rep["n_canonical"] == 200
    # the pairing classes: one retired id at another position (suffix match only), one with different bytes (a finding)
    reordered = list(G.RETIRED_IDS)
    reordered[5], reordered[6] = f"g0006-{reordered[5].split('-')[1]}", f"g0005-{reordered[6].split('-')[1]}"
    reordered[9] = "g0009-00000000"
    cls = O.classify_retired_ids(reordered, O.derive_pairing(G.pairing_rows()))
    assert cls["n_matched_exact"] == 197 and cls["n_matched_content_reordered"] == 2 and cls["n_content_mismatch"] == 1 and cls["content_mismatch_ids"] == ["g0009-00000000"]
    # 199 retired acc rows -> the model's block is not usable; main() exits 3 and files the reason
    G.build_retired_store(tmp_path / "r199", QWEN, correct=set(range(180)), ids=G.RETIRED_IDS[:199])
    rep = O.run(retired_root=tmp_path / "r199", stock_root=tmp_path / "overlap", models=[QWEN], canonical_ids=G.IDS, ids_path=None,
                pairing_rows=G.pairing_rows(), pairing_source="synthetic", repo_root=REPO, store_pin=G.SHA7, n_resamples=200, seed=0, ci=0.95, eval_logs=False)
    assert not rep["models"][QWEN]["usable"] and "199" in rep["models"][QWEN]["reason"] and rep["models"][QWEN]["verdict"] is None
    # a stock row whose producer paired_id disagrees with the derivation refuses the stock leg
    def bad_pair(rows_out):
        ex = json.loads(rows_out[0]["extra_json"])
        ex["paired_id"] = "g9999-deadbeef"
        rows_out[0]["extra_json"] = json.dumps(ex)
    G.build_overlap_stock_store(tmp_path / "bp", QWEN, correct=set(range(178)), rows_edit=bad_pair)
    rep = O.run(retired_root=retired, stock_root=tmp_path / "bp" / "overlap", models=[QWEN], canonical_ids=G.IDS, ids_path=None,
                pairing_rows=G.pairing_rows(), pairing_source="synthetic", repo_root=REPO, store_pin=G.SHA7, n_resamples=200, seed=0, ci=0.95, eval_logs=False)
    assert not rep["models"][QWEN]["usable"] and "paired_id" in rep["models"][QWEN]["reason"]
    # the retired identity must be OFF greedy 512: another gen refuses the retired leg
    G.build_retired_store(tmp_path / "gen", QWEN, correct=set(range(180)), gen={"do_sample": True, "max_new_tokens": 512})
    rep = O.run(retired_root=tmp_path / "gen", stock_root=tmp_path / "overlap", models=[QWEN], canonical_ids=G.IDS, ids_path=None,
                pairing_rows=G.pairing_rows(), pairing_source="synthetic", repo_root=REPO, store_pin=G.SHA7, n_resamples=200, seed=0, ci=0.95, eval_logs=False)
    assert not rep["models"][QWEN]["usable"] and "do_sample" in rep["models"][QWEN]["reason"]
    # the CLI: a rows file, the ids file, exit 0 on the complete case and 3 on the 199 case
    ids = G.write_json(tmp_path / "ids.json", G.IDS)
    prows = G.write_json(tmp_path / "prows.json", G.pairing_rows())
    rc = O.main(["--retired-root", str(retired), "--stock-root", str(tmp_path / "overlap"), "--ids", str(ids), "--pairing", str(prows), "--models", QWEN,
                 "--bootstrap", "200", "--repo-root", str(REPO), "--sha", G.SHA7, "--out", str(tmp_path / "ov.json")])
    assert rc == 0 and json.loads((tmp_path / "ov.json").read_text())["models"][QWEN]["verdict"] == "continuation"
    rc = O.main(["--retired-root", str(tmp_path / "r199"), "--stock-root", str(tmp_path / "overlap"), "--ids", str(ids), "--pairing", str(prows), "--models", QWEN,
                 "--bootstrap", "200", "--repo-root", str(REPO), "--sha", G.SHA7, "--out", str(tmp_path / "ov199.json")])
    assert rc == O.EXIT_REFUSED and "199" in capsys.readouterr().err
    # the derivation refuses rows that are not the retired selection (an id that is not _content_id over its own question)
    rows_bad = G.pairing_rows()
    rows_bad[0]["item_id"] = "g0000-00000000"
    with pytest.raises(A.AnalysisError, match="_content_id"):
        O.derive_pairing(rows_bad)


def test_overlap_eval_logs_hash_every_sample_input_both_ways(tmp_path):
    pytest.importorskip("inspect_ai")
    retired = tmp_path / "retired"
    G.build_retired_store(retired, NEMO, correct=set(range(180)))
    G.build_overlap_stock_store(tmp_path, NEMO, correct=set(range(178)))
    cd = next((tmp_path / "overlap" / NEMO / "none" / "greedy_off512" / "cells").glob("*"))
    G.write_inspect_log(cd, name="2026-01-01T00-00-00")
    rep = O.run(retired_root=retired, stock_root=tmp_path / "overlap", models=[NEMO], canonical_ids=G.IDS, ids_path=None, pairing_rows=G.pairing_rows(),
                pairing_source="synthetic", repo_root=REPO, store_pin=G.SHA7, n_resamples=200, seed=0, ci=0.95, eval_logs=True)
    tc = rep["models"][NEMO]["stock"]["text_check"]
    assert tc["run"] and tc["available"] and tc["n_inputs"] == 200 and tc["n_inspect_id_mismatch"] == 0 and tc["n_retired_suffix_mismatch"] == 0
    # a log whose sample inputs are other text: the byte check refuses the stock leg
    G.build_overlap_stock_store(tmp_path / "bad", NEMO, correct=set(range(178)))
    cd = next((tmp_path / "bad" / "overlap" / NEMO / "none" / "greedy_off512" / "cells").glob("*"))
    G.write_inspect_log(cd, name="2026-01-01T00-00-00", inputs={i: q + "\n" for i, q in zip(G.IDS, G.QUESTIONS)})
    rep = O.run(retired_root=retired, stock_root=tmp_path / "bad" / "overlap", models=[NEMO], canonical_ids=G.IDS, ids_path=None, pairing_rows=G.pairing_rows(),
                pairing_source="synthetic", repo_root=REPO, store_pin=G.SHA7, n_resamples=200, seed=0, ci=0.95, eval_logs=True)
    assert not rep["models"][NEMO]["usable"] and "hash" in rep["models"][NEMO]["reason"]


# --------------------------------------------------------------------------- #
# the synthetic builder writes through the producer's own helpers              #
# --------------------------------------------------------------------------- #


def test_the_fixtures_records_are_the_producers_own(full_store):
    root = full_store.root
    cd = next((root / row(QWEN, "main", CONST_PSR)["out_root"] / "cells").glob("*"))
    outcomes = completion.read_task_outcomes(cd, V.TASK)
    admitted = {frozenset(fs) for fs in C.TASK_OUTCOME_ITEM_FIELD_SETS}
    assert set(outcomes) == set(C.TASK_OUTCOMES_HEADER_FIELDS) and all(frozenset(it) in admitted for it in outcomes["items"])
    completion.verify_outcomes_against_summary(outcomes, completion.read_task_summary(cd, V.TASK))
    manifest = completion.read_store_manifest(cd)
    assert set(manifest[C.MANIFEST_KEY_PRODUCER]) == set(C.PRODUCER_BLOCK_FIELDS)
    ledger_rows = [json.loads(line) for line in (root / row(QWEN, "main", CONST_PSR)["out_root"] / C.LEDGER_FILENAME).read_text().splitlines()]
    assert all(set(C.LEDGER_ROW_FIELDS) <= set(r) and r["attempt_id"] == "a1" for r in ledger_rows)
    recs = [json.loads(line) for line in (root / row(QWEN, "main", CONST_PSR)["out_root"] / C.REQUEST_EVIDENCE_FILENAME).read_text().splitlines()]
    assert all(set(C.REQUEST_EVIDENCE_FIELDS) <= set(r) and r["attempt_id"] == "a1" for r in recs)
    assert C.request_evidence_reconciliation(recs, ledger_rows, filed=[(V.TASK, i) for i in G.IDS]) is None
    out_root = root / row(QWEN, "main", CONST_PSR)["out_root"]
    ladder = json.loads((out_root / CL.RECORD_NAME).read_text())          # the planner's location: <out_root>/cap_ladder.json
    assert ladder["schema"] == CL.RECORD_SCHEMA and ladder["rule"] == V.GSM8K_TRIGGER_RULE and ladder["exit"] == CL.EXIT_CLEAN
    assert set(IR.RECORD_INSTRUMENT_FIELDS) <= set(ladder) and not (cd / CL.RECORD_NAME).exists()
    # the wrapper's RESULT line and the driver's gen_stats copy of the summary are beside the store, as on the cluster
    assert "result=COMPLETE" in (out_root / G.RESULT_NAME).read_text() and set(json.loads((cd / "gen_stats.json").read_text())) == {V.TASK}
    # the synthetic ids are the two hashes of one text, as the two instruments derive them
    assert all(G.IDS[i] == C.gsm8k_sample_id(G.QUESTIONS[i]) and G.RETIRED_IDS[i] == O.retired_id(i, G.QUESTIONS[i]) for i in range(200))


def test_ladder_fixtures_are_the_adapters_own_and_the_old_reader_spelling_is_refused(full_store):
    """A-P1.1: the ladder record on disk IS `inspect_rung.classify_store`'s (re-run, equal), in the adapter's spellings
    (`eligible_ids`, an ASCII-hyphen rule), and the subset file IS its `rung_subset/1` body, validated by the contract
    (no `subset_sha16` key, no `gsm8k_rung_subset/1` schema); a record in the reader's FORMER spelling — the shape the
    old fixture hand-wrote — is refused by name, so the reader can no longer agree with itself alone."""
    esc = full_store.esc
    root = Path(esc["source_out_root"])
    record = json.loads((root / CL.RECORD_NAME).read_text())
    again, subset = G.classify(root)
    assert record == json.loads(json.dumps(again)) and subset is not None
    assert set(IR.RECORD_INSTRUMENT_FIELDS) <= set(record) and record["exit"] == CL.ESCALATE and record["next_cap"] == 16384
    assert sorted(record["eligible_ids"]) == sorted(esc["subset"]) and record["n_eligible"] == 7 and record["n_non_terminating"] == 2
    assert "eligible" not in record and "−" not in record["rule"] and record["rule"] == IR.GSM8K_TRIGGER_RULE
    on_disk = json.loads(Path(esc["subset_path"]).read_text())
    assert on_disk == C.validate_item_subset(on_disk, where="test") and on_disk == subset
    assert set(on_disk) == set(C.ITEM_SUBSET_FIELDS) and on_disk["schema"] == C.ITEM_SUBSET_SCHEMA and "subset_sha16" not in on_disk
    assert on_disk["task"] == C.GSM8K_STANDARD_TASK and on_disk["source_out_root"] == str(root) and on_disk["transport_check"] == record["transport"]
    manifest = completion.read_store_manifest(esc["source_cell_dir"])
    at_cap = [i for i in G.IDS[190:199]]
    assert V.ladder_record_problem(record, n=200, at_cap_ids=at_cap, source_manifest=manifest) is None
    assert V.rung_subset_problem(on_disk, record=record, source_manifest=manifest, source_out_root=row(*ESC)["out_root"]) is None
    # the rung manifest: cap_rung and item_subset are the contract's derivations from THIS subset file
    rung_manifest = completion.read_store_manifest(esc["rung_cell_dir"])
    assert rung_manifest[C.MANIFEST_KEY_TINYCAP][C.IDENTITY_KEY_CAP_RUNG] == C.cap_rung_identity(on_disk)
    assert rung_manifest[C.MANIFEST_KEY_ITEM_SUBSET] == C.item_subset_block(on_disk, path=str(esc["subset_path"]))
    assert V.item_subset_block_problem(rung_manifest, subset=on_disk, source_manifest=manifest, source_out_root=row(*ESC)["out_root"]) is None
    # the reader's former spellings are REFUSED: `eligible` instead of `eligible_ids`; the U+2212 rule; the extra subset key
    old = {**{k: v for k, v in record.items() if k != "eligible_ids"}, "eligible": record["eligible_ids"]}
    assert "eligible_ids" in (V.ladder_record_problem(old, n=200, at_cap_ids=at_cap) or "")
    minus = {**record, "rule": record["rule"].replace(" - degenerate - boundary", " − degenerate − boundary")}
    assert "rule" in (V.ladder_record_problem(minus, n=200, at_cap_ids=at_cap) or "")
    extra = {**on_disk, "subset_sha16": C.item_subset_sha16(on_disk["ids"])}
    assert "unknown ['subset_sha16']" in (V.rung_subset_problem(extra, record=record, source_manifest=manifest) or "")
    old_schema = {**on_disk, "schema": "gsm8k_rung_subset/1"}
    assert "schema" in (V.rung_subset_problem(old_schema, record=record, source_manifest=manifest) or "")
    # the adapter's transport_refused record names the rung it could not carry (next_cap 16,384 + a refusing verdict): accepted
    refused = {**record, "exit": CL.EXIT_TRANSPORT_REFUSED, "escalate": False, "transport": {**record["transport"], "ok": False}}
    assert V.ladder_record_problem(refused, n=200, at_cap_ids=at_cap) is None
    assert "next_cap" in (V.ladder_record_problem({**refused, "next_cap": None}, n=200, at_cap_ids=at_cap) or "")
    # a record over ANOTHER store (its source identity hash) is refused by name
    assert "identity_hash" in (V.ladder_record_problem(record, n=200, at_cap_ids=at_cap, source_manifest={"harness_config_hash": "0" * 16}) or "")
    # both writers' locations are read; two records that disagree refuse
    cd = Path(esc["source_cell_dir"])
    path, doc, problem = V.read_ladder_record(root, cd)
    assert path == root / CL.RECORD_NAME and doc == record and problem is None
    CL.write_cell_record(cd, {**record, "n_at_cap": 8})
    try:
        _path, _doc, problem = V.read_ladder_record(root, cd)
        assert problem and "disagree" in problem
    finally:
        (cd / CL.RECORD_NAME).unlink()


def test_rung_source_root_is_matched_relocation_aware(tmp_path):
    """A-P1.2: the planner records the CLUSTER's absolute source root in the subset file (and the driver copies it into
    the rung manifest's item_subset block); the reader holds a LOCAL pull under another root. The match is on the
    trailing layout components (controls/<TAG>/<slug>/<mode>) beside the source identity hash — an absolute cluster
    source with a different local pull root passes; a tail naming another store refuses by name."""
    rows = small_matrix((QWEN, "main", NONE), (QWEN, "main", CONST_LORA))
    cluster_root = "/shared/someone/exp/gsm8k_paired200_wave"
    src_row = row(QWEN, "main", CONST_LORA)
    cluster_source = f"{cluster_root}/{src_row['out_root']}"
    assert V.same_store_tail(cluster_source, src_row["out_root"]) and V.same_store_tail(cluster_source + "/", src_row["out_root"] + "/")
    assert V.same_store_tail(src_row["out_root"], src_row["out_root"]) and not V.same_store_tail(cluster_root, src_row["out_root"])
    assert not V.same_store_tail(f"{cluster_root}/controls/qwen35_4b/constv2ctl2-full--psr_ce_all/thinking_card", src_row["out_root"])
    assert not V.same_store_tail(f"{cluster_root}/controls/qwen35_4b_it/{src_row['cell_slug']}/thinking_card", src_row["out_root"])
    assert not V.same_store_tail("", src_row["out_root"]) and not V.same_store_tail(cluster_source, "")
    for name, source in (("cluster", cluster_source), ("other_cell", f"{cluster_root}/controls/qwen35_4b/constv2ctl2-full--psr_ce_all/thinking_card")):
        gen = tmp_path / name
        G.build_store(gen, row(QWEN, "main", NONE), correct=G.CORRECT_NONE)
        esc = G.build_escalated_cell(gen, src_row, at_cap=range(190, 199), loops={190, 191}, write_rung=False)
        relocated = {**esc["subset_doc"], "source_out_root": source}
        subset_path = G.write_rung_subset(gen, src_row, relocated)
        manifest = completion.read_store_manifest(esc["source_cell_dir"])
        G.build_store(gen, src_row, correct={0, 2, 4}, cap=16384, ids=esc["subset"], rung_source=manifest, subset_doc=relocated,
                      subset_path=subset_path, ladder=False)
        rep = analyze(gen, matrix=rows, overlap=None)
        lad = rep["cells"][V.cell_key(QWEN, "main", CONST_LORA)]["ladder"]
        if name == "cluster":
            assert lad["subset_problem"] is None and rep["gates"]["A5_at_cap"]["status"] == A.GATE_PASS, lad["subset_problem"]
            rung = rep["legs"][V.leg_key(QWEN, "main", CONST_LORA, 16384)]
            assert rung["refused"] is None and rep["gates"]["A6_ladder"]["status"] == A.GATE_PASS and rep["unusable_legs"] == {}
            assert json.loads(Path(subset_path).read_text())["source_out_root"] == cluster_source
            assert completion.read_store_manifest(rep["legs"][rung["leg"]]["cell_dir"])[C.MANIFEST_KEY_ITEM_SUBSET]["source_out_root"] == cluster_source
        else:
            assert lad["subset_problem"] and "source_out_root" in lad["subset_problem"] and rep["gates"]["A5_at_cap"]["status"] == A.GATE_FAIL
            assert rep["cells"][V.cell_key(QWEN, "main", CONST_LORA)]["rung_leg"] is None
    # the rung manifest's item_subset block is held to the SAME rule: a block naming another store refuses the rung leg
    gen = tmp_path / "block"
    G.build_store(gen, row(QWEN, "main", NONE), correct=G.CORRECT_NONE)
    esc = G.build_escalated_cell(gen, src_row, at_cap=range(190, 199), loops={190, 191})
    rung_cd = Path(esc["rung_cell_dir"])
    m = json.loads((rung_cd / "manifest.json").read_text())
    m[C.MANIFEST_KEY_ITEM_SUBSET]["source_out_root"] = f"{cluster_root}/controls/qwen35_4b/none/thinking_card"
    (rung_cd / "manifest.json").write_text(json.dumps(m, indent=1))
    rung = analyze(gen, matrix=rows, overlap=None)["legs"][V.leg_key(QWEN, "main", CONST_LORA, 16384)]
    assert rung["refused"] and C.MANIFEST_KEY_ITEM_SUBSET in rung["refused"] and "source_out_root" in rung["refused"]
    m[C.MANIFEST_KEY_ITEM_SUBSET]["source_out_root"] = esc["subset_doc"]["source_out_root"]
    m[C.MANIFEST_KEY_ITEM_SUBSET]["subset_sha16"] = "0" * 16
    (rung_cd / "manifest.json").write_text(json.dumps(m, indent=1))
    rung = analyze(gen, matrix=rows, overlap=None)["legs"][V.leg_key(QWEN, "main", CONST_LORA, 16384)]
    assert rung["refused"] and "subset_sha16" in rung["refused"]
    del m[C.MANIFEST_KEY_ITEM_SUBSET]
    (rung_cd / "manifest.json").write_text(json.dumps(m, indent=1))
    rung = analyze(gen, matrix=rows, overlap=None)["legs"][V.leg_key(QWEN, "main", CONST_LORA, 16384)]
    assert rung["refused"] and "carries no item_subset" in rung["refused"]


def test_per_item_log_check_fails_a_count_preserving_swap_of_outcome_fields(tmp_path):
    """A-P1.3: two SCORED items — one at the cap (case (c)), one below it — swap their `at_cap` / `stop_reason` /
    `completion_tokens` in the outcome record (and rows.jsonl), every marginal preserved: the parent's per-item
    score check still agrees; the outcome-field comparison against the retained .eval log fails BOTH items by
    name on all three fields, and the leg is refused."""
    pytest.importorskip("inspect_ai")
    rows = small_matrix((QWEN, "main", NONE), (QWEN, "main", CONST_LORA))
    G.write_inspect_log(G.build_store(tmp_path, row(QWEN, "main", NONE), correct=G.CORRECT_NONE), name="2026-01-01T00-00-00")
    cd = G.build_store(tmp_path, row(QWEN, "main", CONST_LORA), correct=set(range(150)) | {197}, capped_scored={197})
    G.write_inspect_log(cd, name="2026-01-01T00-00-00")            # the log of the store AS GENERATED
    a, b = G.IDS[197], G.IDS[3]                                      # a: at cap, scored, correct; b: below the cap, scored, correct
    out_path = cd / C.TASK_OUTCOMES_DIR / C.task_outcomes_filename(V.TASK)
    outcomes = json.loads(out_path.read_text())
    by_id = {it["item_id"]: it for it in outcomes["items"]}
    assert by_id[a]["at_cap"] and not by_id[b]["at_cap"] and by_id[a]["scored"] and by_id[b]["scored"]
    for f in ("at_cap", "stop_reason", "completion_tokens"):
        by_id[a][f], by_id[b][f] = by_id[b][f], by_id[a][f]
    out_path.write_text(json.dumps(outcomes))
    lines = [json.loads(line) for line in (cd / "rows.jsonl").read_text().splitlines()]
    for r in lines:
        if r["item_id"] in (a, b):
            ex = json.loads(r["extra_json"])
            ex.update(stop_reason=by_id[r["item_id"]]["stop_reason"], completion_tokens=by_id[r["item_id"]]["completion_tokens"])
            r["extra_json"] = json.dumps(ex)
    (cd / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    # the store still agrees with itself: summary == outcomes, rows == outcomes (the swap is count-preserving)
    completion.verify_outcomes_against_summary(json.loads(out_path.read_text()), completion.read_task_summary(cd, V.TASK))
    header = {k: v for k, v in outcomes.items() if k != "items"}
    ec = V.eval_cross_check(cd, header, outcomes["items"], max_tokens=8192)
    assert ec["available"] and ec["agrees"] is False and ec["count_detail"] is None
    per = ec["per_item"]
    assert per["mismatches"] == [] and per["n_mismatched"] == 0             # the parent's score comparison sees nothing
    assert per["n_field_mismatched"] == 6 and {(m["item_id"], m["field"]) for m in per["field_mismatches"]} == {(i, f) for i in (a, b) for f in ("at_cap", "stop_reason", "completion_tokens")}
    assert per["fields_compared"] == list(V.OUTCOME_FIELDS_FROM_LOG)
    assert all(f in ec["detail"] for f in ("at_cap", "stop_reason", "completion_tokens")) and a in ec["detail"] and b in ec["detail"]
    rep = analyze(tmp_path, matrix=rows, overlap=None, read_eval_logs=True)
    leg = leg_of(rep, QWEN, "main", CONST_LORA)
    assert leg["refused"] and "disagrees" in leg["refused"] and "at_cap" in leg["refused"] and rep["gates"]["A0_completeness"]["status"] == A.GATE_FAIL
    assert leg_of(rep, QWEN, "main", NONE)["eval_check"]["agrees"] is True
    # the unswapped store: every field agrees on every draw, at-cap draws included
    G.build_store(tmp_path / "ok", row(QWEN, "main", CONST_LORA), correct=set(range(150)) | {197}, capped_scored={197}, unscored={195, 196})
    cd2 = next((tmp_path / "ok" / row(QWEN, "main", CONST_LORA)["out_root"] / "cells").glob("*"))
    G.write_inspect_log(cd2, name="2026-01-01T00-00-00")
    outcomes2 = completion.read_task_outcomes(cd2, V.TASK)
    ec2 = V.eval_cross_check(cd2, {k: v for k, v in outcomes2.items() if k != "items"}, outcomes2["items"], max_tokens=8192)
    assert ec2["agrees"] and ec2["per_item"]["n_field_mismatched"] == 0 and ec2["per_item"]["n_compared"] == 200


def test_overlap_eval_logs_fail_closed_on_a_missing_or_unreadable_log(tmp_path, capsys):
    """A-P1.4: `--eval-logs` asks for the byte check on the stock side; a leg whose retained .eval is missing (or
    unreadable) can not make it and is UNUSABLE — never `available: True` with `n_inputs: 0` — and main() exits 3."""
    retired = tmp_path / "retired"
    G.build_retired_store(retired, NEMO, correct=set(range(180)))
    G.build_overlap_stock_store(tmp_path, NEMO, correct=set(range(178)))
    rep = O.run(retired_root=retired, stock_root=tmp_path / "overlap", models=[NEMO], canonical_ids=G.IDS, ids_path=None, pairing_rows=G.pairing_rows(),
                pairing_source="synthetic", repo_root=REPO, store_pin=G.SHA7, n_resamples=200, seed=0, ci=0.95, eval_logs=True)
    b = rep["models"][NEMO]
    assert not b["usable"] and b["verdict"] is None and "did not run" in b["reason"] and "--eval-logs" in b["reason"]
    tc = b["stock"]["text_check"]
    assert tc["run"] and tc["available"] is False and tc["n_inputs"] == 0 and tc["n_required"] == 200 and "validated" not in tc
    # the same stores WITHOUT --eval-logs are usable (the byte check was not asked for)
    rep_off = O.run(retired_root=retired, stock_root=tmp_path / "overlap", models=[NEMO], canonical_ids=G.IDS, ids_path=None, pairing_rows=G.pairing_rows(),
                    pairing_source="synthetic", repo_root=REPO, store_pin=G.SHA7, n_resamples=200, seed=0, ci=0.95, eval_logs=False)
    assert rep_off["models"][NEMO]["usable"] and rep_off["models"][NEMO]["stock"]["text_check"] == {"run": False}
    # an UNREADABLE log (present, not an Inspect log) is refused the same way, naming the producer's selection rule
    cd = next((tmp_path / "overlap" / NEMO / "none" / "greedy_off512" / "cells").glob("*"))
    log_dir = C.inspect_log_dir(cd, V.TASK, 0)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "2026-01-01T00-00-00_gsm8k_paired200_torn.eval").write_bytes(b"not an eval log")
    rep = O.run(retired_root=retired, stock_root=tmp_path / "overlap", models=[NEMO], canonical_ids=G.IDS, ids_path=None, pairing_rows=G.pairing_rows(),
                pairing_source="synthetic", repo_root=REPO, store_pin=G.SHA7, n_resamples=200, seed=0, ci=0.95, eval_logs=True)
    b = rep["models"][NEMO]
    assert not b["usable"] and "did not run" in b["reason"] and b["stock"]["text_check"]["log_present"] is True
    # the CLI exits 3 and files the reason
    ids = G.write_json(tmp_path / "ids.json", G.IDS)
    prows = G.write_json(tmp_path / "prows.json", G.pairing_rows())
    rc = O.main(["--retired-root", str(retired), "--stock-root", str(tmp_path / "overlap"), "--ids", str(ids), "--pairing", str(prows), "--models", NEMO,
                 "--bootstrap", "200", "--repo-root", str(REPO), "--sha", G.SHA7, "--eval-logs", "--out", str(tmp_path / "ov.json")])
    assert rc == O.EXIT_REFUSED and "did not run" in capsys.readouterr().err
    assert json.loads((tmp_path / "ov.json").read_text())["models"][NEMO]["usable"] is False


def test_overlap_eval_logs_require_every_attempted_input(tmp_path):
    """A-P1.4 (the incomplete case): a log that qualifies but yields fewer than the 200 attempted inputs is not a byte check."""
    pytest.importorskip("inspect_ai")
    retired = tmp_path / "retired"
    G.build_retired_store(retired, NEMO, correct=set(range(180)))
    G.build_overlap_stock_store(tmp_path, NEMO, correct=set(range(178)))
    cd = next((tmp_path / "overlap" / NEMO / "none" / "greedy_off512" / "cells").glob("*"))
    G.write_inspect_log(cd, name="2026-01-01T00-00-00")
    # sabotage the reader's input extraction only: monkeypatching is not needed — a log over 199 samples does not QUALIFY
    # under the producer's rule (expected_n 200), so it is the "did not run" case; the 200-sample log validates fully
    rep = O.run(retired_root=retired, stock_root=tmp_path / "overlap", models=[NEMO], canonical_ids=G.IDS, ids_path=None, pairing_rows=G.pairing_rows(),
                pairing_source="synthetic", repo_root=REPO, store_pin=G.SHA7, n_resamples=200, seed=0, ci=0.95, eval_logs=True)
    tc = rep["models"][NEMO]["stock"]["text_check"]
    assert rep["models"][NEMO]["usable"] and tc["validated"] is True and tc["n_inputs"] == 200 and tc["n_inspect_id_mismatch"] == 0
    import shutil
    shutil.rmtree(C.inspect_log_dir(cd, V.TASK, 0))
    G.write_inspect_log(cd, name="2026-01-01T00-00-00", n_samples=199)
    rep = O.run(retired_root=retired, stock_root=tmp_path / "overlap", models=[NEMO], canonical_ids=G.IDS, ids_path=None, pairing_rows=G.pairing_rows(),
                pairing_source="synthetic", repo_root=REPO, store_pin=G.SHA7, n_resamples=200, seed=0, ci=0.95, eval_logs=True)
    b = rep["models"][NEMO]
    assert not b["usable"] and "did not run" in b["reason"] and b["stock"]["text_check"]["n_inputs"] == 0


OTHER_TREE = "1234567812345678123456781234567812345678"     # a synthetic second producer tree, self-consistent in its own manifest


def _overlap_run(retired, stock_root, key, *, store_pin=G.SHA7, eval_logs=False):
    return O.run(retired_root=retired, stock_root=stock_root, models=[key], canonical_ids=G.IDS, ids_path=None, pairing_rows=G.pairing_rows(),
                 pairing_source="synthetic", repo_root=REPO, store_pin=store_pin, n_resamples=200, seed=0, ci=0.95, eval_logs=eval_logs)


def test_overlap_check_holds_the_stock_producer_to_the_store_pin_not_to_its_own_sha(tmp_path, capsys):
    """Guard 1 (2026-09-11): the overlap check used to hand `producer_problem` the manifest's OWN tree sha as the
    pin — self-referential, so a store from another tree passed. It now takes the wave pin (`--sha`, run E's) and a
    manifest whose tree does not carry it is refused with BOTH hashes named, before any outcome or log is read."""
    retired = tmp_path / "retired"
    G.build_retired_store(retired, QWEN, correct=set(range(180)))
    # a store whose manifest is self-consistent (producer and every attempt on OTHER_TREE) but is not the pin's
    G.build_overlap_stock_store(tmp_path / "other", QWEN, correct=set(range(178)), tree_sha=OTHER_TREE)
    rep = _overlap_run(retired, tmp_path / "other" / "overlap", QWEN)
    b = rep["models"][QWEN]
    assert not b["usable"] and b["verdict"] is None and b["comparison"] is None
    assert b["reason"].startswith("producer:") and OTHER_TREE in b["reason"] and G.SHA7 in b["reason"] and "another tree" in b["reason"]
    assert b["stock"]["producer_tree_sha"] == OTHER_TREE and b["stock"]["store_pin"] == G.SHA7 and rep["store_pin"] == G.SHA7
    # the same store IS usable under its own tree's pin: the pin decides, the manifest never vouches for itself
    rep = _overlap_run(retired, tmp_path / "other" / "overlap", QWEN, store_pin=OTHER_TREE[:7])
    assert rep["models"][QWEN]["usable"] and rep["models"][QWEN]["stock"]["producer_tree_sha"] == OTHER_TREE and rep["store_pin"] == OTHER_TREE[:7]
    # and the pin's own store passes under the pin (the full 40-hex form accepted too), and refuses under another
    G.build_overlap_stock_store(tmp_path, QWEN, correct=set(range(178)))
    assert _overlap_run(retired, tmp_path / "overlap", QWEN)["models"][QWEN]["usable"]
    assert _overlap_run(retired, tmp_path / "overlap", QWEN, store_pin=G.FULL_SHA)["models"][QWEN]["usable"]
    assert OTHER_TREE[:7] in _overlap_run(retired, tmp_path / "overlap", QWEN, store_pin=OTHER_TREE[:7])["models"][QWEN]["reason"]
    # a malformed pin is refused before anything is read; the CLI requires --sha (exit 2) and files the refusal (exit 3)
    with pytest.raises(A.AnalysisError, match="store pin"):
        _overlap_run(retired, tmp_path / "overlap", QWEN, store_pin="not-a-sha")
    ids = G.write_json(tmp_path / "ids.json", G.IDS)
    prows = G.write_json(tmp_path / "prows.json", G.pairing_rows())
    common = ["--retired-root", str(retired), "--ids", str(ids), "--pairing", str(prows), "--models", QWEN, "--bootstrap", "200", "--repo-root", str(REPO)]
    with pytest.raises(SystemExit) as e:
        O.main([*common, "--stock-root", str(tmp_path / "other" / "overlap"), "--out", str(tmp_path / "nosha.json")])
    assert e.value.code == 2 and not (tmp_path / "nosha.json").exists()
    rc = O.main([*common, "--stock-root", str(tmp_path / "other" / "overlap"), "--sha", G.SHA7, "--out", str(tmp_path / "ov_other.json")])
    err = capsys.readouterr().err
    assert rc == O.EXIT_REFUSED and OTHER_TREE in err and G.SHA7 in err
    assert json.loads((tmp_path / "ov_other.json").read_text())["models"][QWEN]["usable"] is False
    rc = O.main([*common, "--stock-root", str(tmp_path / "overlap"), "--sha", G.SHA7, "--out", str(tmp_path / "ov_pin.json")])
    assert rc == 0 and json.loads((tmp_path / "ov_pin.json").read_text())["store_pin"] == G.SHA7


def test_eval_cross_check_refuses_two_qualifying_logs_and_binds_the_one_to_the_accepted_attempt(tmp_path):
    """Guard 2 (2026-09-11): a log dir holding TWO logs that both qualify under the producer's rule is two
    generations of one leg; the reader refuses, naming every candidate with its mtime, instead of taking the newest
    (the producer's `reusable_log` keeps its newest-wins rule — it is the driver's resume rule, unchanged). The one
    qualifying log is also bound to the accepted attempt's [started, ended]: a log created outside it is refused."""
    pytest.importorskip("inspect_ai")
    import os
    from sisteer.capeval.tiers import inspect_served as isv
    retired = tmp_path / "retired"
    G.build_retired_store(retired, NEMO, correct=set(range(180)))
    G.build_overlap_stock_store(tmp_path, NEMO, correct=set(range(178)))
    cd = next((tmp_path / "overlap" / NEMO / "none" / "greedy_off512" / "cells").glob("*"))
    manifest = completion.read_store_manifest(cd)
    outcomes = completion.read_task_outcomes(cd, V.TASK)
    header = {k: v for k, v in outcomes.items() if k != "items"}
    d = C.inspect_log_dir(cd, V.TASK, 0)
    t_old, t_new = 1_800_000_000.0, 1_800_003_600.0
    older = G.write_inspect_log(cd, name="2026-01-01T00-00-00", mtime=t_old)                 # the record's own generation
    newer = G.write_inspect_log(cd, name="2026-01-01T00-00-05", mtime=t_new)                 # a second complete generation of the same leg
    ids = [str(it["item_id"]) for it in outcomes["items"]]
    # the producer's rule is unchanged: both qualify, and its selector still takes the NEWEST by mtime
    assert V.qualifying_logs(d, expected_n=200, expected_ids=ids, expected_max_tokens=512) == sorted([older, newer])
    assert Path(isv.reusable_log(d, expected_n=200, expected_ids=ids, expected_max_tokens=512).location).name == newer.name
    os.utime(newer, (t_old - 60, t_old - 60))
    assert Path(isv.reusable_log(d, expected_n=200, expected_ids=ids, expected_max_tokens=512).location).name == older.name
    os.utime(newer, (t_new, t_new))
    # the reader refuses: NOT RUN, both candidates named with their mtimes, no log re-derived
    ec = V.eval_cross_check(cd, header, outcomes["items"], max_tokens=512, accepted_attempt=V.accepted_attempt(manifest))
    assert ec["available"] is False and ec["agrees"] is None and ec["log_present"] is True and ec["selected_logs"] == []
    assert ec["reason"].startswith("2 retained .eval logs") and "refusing to take the newest" in ec["reason"]
    assert str(older) in ec["reason"] and str(newer) in ec["reason"] and "2026-01-01T00-00-05" in ec["reason"]
    assert "mtime 2027-01-15T08:00:00Z" in ec["reason"] and "mtime 2027-01-15T09:00:00Z" in ec["reason"]
    rep = _overlap_run(retired, tmp_path / "overlap", NEMO, eval_logs=True)
    b = rep["models"][NEMO]
    assert not b["usable"] and "did not run" in b["reason"] and older.name in b["reason"] and newer.name in b["reason"]
    assert b["stock"]["text_check"]["available"] is False and b["stock"]["text_check"]["n_inputs"] == 0
    # one qualifying log (the other removed): the leg passes, the log is named and bound to the accepted attempt's window
    older.unlink()
    ec = V.eval_cross_check(cd, header, outcomes["items"], max_tokens=512, accepted_attempt=V.accepted_attempt(manifest))
    assert ec["available"] and ec["agrees"] and ec["selected_logs"] == [newer.name]
    assert ec["attempt_binding"] == {"attempt_id": "a1", "started": "2026-01-01T00:00:00Z", "ended": "2026-01-01T00:30:00Z",
                                     "logs": [{"log": newer.name, "created": "2026-01-01T00:00:00Z"}]}
    rep = _overlap_run(retired, tmp_path / "overlap", NEMO, eval_logs=True)
    assert rep["models"][NEMO]["usable"] and rep["models"][NEMO]["stock"]["text_check"]["validated"] is True
    assert rep["models"][NEMO]["stock"]["text_check"]["attempt_binding"]["attempt_id"] == "a1"
    # a retained log that does not qualify under the producer's rule is not a candidate: the strict selector agrees with the
    # producer on WHICH files qualify (a torn file and a 199-sample log do not) and differs only in how many may — one
    (d / "2026-01-01T00-00-01_gsm8k_paired200_torn.eval").write_bytes(b"not an eval log")
    short = G.write_inspect_log(cd, name="2026-01-01T00-00-02", n_samples=199, mtime=t_new + 60)
    assert V.qualifying_logs(d, expected_n=200, expected_ids=ids, expected_max_tokens=512) == [newer]
    assert Path(isv.reusable_log(d, expected_n=200, expected_ids=ids, expected_max_tokens=512).location).name == newer.name
    assert V.eval_cross_check(cd, header, outcomes["items"], max_tokens=512, accepted_attempt=V.accepted_attempt(manifest))["available"]
    short.unlink()
    # the window binding: the one qualifying log created OUTSIDE the accepted attempt's window is another run's evidence
    newer.unlink()
    outside = G.write_inspect_log(cd, name="2026-01-01T00-00-07", created="2026-01-01T12:15:18+00:00")
    ec = V.eval_cross_check(cd, header, outcomes["items"], max_tokens=512, accepted_attempt=V.accepted_attempt(manifest))
    assert ec["available"] is False and ec["reason"].startswith(outside.name) and "outside the accepted generation attempt 'a1'" in ec["reason"]
    assert "created 2026-01-01T12:15:18Z" in ec["reason"] and "[2026-01-01T00:00:00Z, 2026-01-01T00:30:00Z]" in ec["reason"]
    rep = _overlap_run(retired, tmp_path / "overlap", NEMO, eval_logs=True)
    assert not rep["models"][NEMO]["usable"] and "outside the accepted generation attempt" in rep["models"][NEMO]["reason"]
    # no binding asked for (the reader's former call): the same log re-derives as before
    ec = V.eval_cross_check(cd, header, outcomes["items"], max_tokens=512)
    assert ec["available"] and ec["agrees"] and ec["attempt_binding"] is None
    # an accepted attempt that records no window cannot bind the log: a refusal, never a pass
    entry = dict(V.accepted_attempt(manifest)); entry.pop("ended")
    ec = V.eval_cross_check(cd, header, outcomes["items"], max_tokens=512, accepted_attempt=entry)
    assert ec["available"] is False and "records no complete [started, ended] window" in ec["reason"]
    # a log whose created stamp cannot be placed (naive, or absent) is refused too — checked on the helper: Inspect's writer
    # normalises a naive `created` to +00:00, so no real log carries one and the branch is the reader's defence
    for created in ("2026-01-01T00:00:00", None, ""):
        fake = SimpleNamespace(location=str(outside), eval=SimpleNamespace(created=created))
        assert "no placeable creation time" in V.log_outside_attempt(fake, V.accepted_attempt(manifest))
    assert V.log_outside_attempt(SimpleNamespace(location=str(outside), eval=SimpleNamespace(created="2026-01-01T00:10:00Z")),
                                 V.accepted_attempt(manifest)) is None
    outside.unlink()
    # the run-E path carries the same binding on every leg (the fixture's logs are created inside their accepted attempt)
    rows = small_matrix((NEMO, "main", NONE))
    G.write_inspect_log(G.build_store(tmp_path / "e", row(NEMO, "main", NONE), correct=G.CORRECT_NONE), name="2026-01-01T00-00-00")
    leg = leg_of(analyze(tmp_path / "e", matrix=rows, overlap=None, read_eval_logs=True), NEMO, "main", NONE)
    assert leg["eval_check"]["agrees"] is True and leg["eval_check"]["attempt_binding"]["attempt_id"] == "a1"
    assert leg["eval_check"]["attempt_binding"]["logs"][0]["created"] == "2026-01-01T00:00:00Z"


def test_top_items_surface_the_failed_gates_from_the_records(tmp_path):
    """The simplify pass's flag: `top_items` filtered findings on the substring "FAILED", which no finding carries;
    it now reads the gate RECORDS — every FAILED gate first with its detail, then the unwaivable pendings, the
    contrasts without an estimand, the triggered cells, the merged words."""
    rows = small_matrix((GLM, "main", NONE), (GLM, "main", CONST_PSR))
    G.build_store(tmp_path, row(GLM, "main", NONE), correct=G.CORRECT_NONE, no_think={40, 41})
    rep = analyze(tmp_path, matrix=rows, overlap=None)
    assert rep["gates"]["failed"] == ["A0_completeness", "A10_overlap"]
    assert rep["top_items"][0].startswith("A0_completeness: FAILED — ") and rep["top_items"][1].startswith("A10_overlap: FAILED — ")
    assert any("no estimand" in t for t in rep["top_items"]) and rep["top_items"] == A.top_items_of(rep)
    rep2 = analyze(tmp_path, matrix=rows, overlap=None, canonical_ids=None)
    assert any(t.startswith("A7_population: PENDING (not waivable)") for t in rep2["top_items"])


def test_the_frozen_matrix_pins_d20s_counts():
    """Gemma's two header-located main cells are SERVABLE thinking-ON, so the frozen matrix names NO expected refusal —
    38 servable of 38 launched, 32 computable contrasts of 32 designed — and every declared row carries the contract's
    stamp triple (the reader refuses the plan's spelling)."""
    path = REPO / "configs" / "sdfbase" / "gsm8k" / "matrix.json"
    doc = json.loads(path.read_text())
    mx = V.load_matrix(path)
    # the LAUNCHED shape of the PUBLISHED wave is fixed (38 / 4 declared): since math-eight-arms-extension E2 the document appends
    # further waves (`ext`, `steer` — rows carrying a `wave` key, under wave-isolated roots; the reader admits them, its external-row
    # rules pinned in test_eight_arms_gsm8k_analysis / test_sdfbase_gsm8k_sbatch), so the published numbers are read off the
    # published slice; held = the four it-view constitution objects + the admitted prompted cells (none over today's roster).
    # the INSTRUCT-DIRECT wave is a THIRD matrix document — configs/sdfbase/gsm8k/matrix_itdoc.json (schema 2, its own
    # wave root, every row wave `itdoc`) — never rows of THIS document: the holds stay here byte for byte and no row of this file
    # carries `object_source: itdoc` (test_eight_arms_gsm8k_analysis pins the it-doc document's shape and the reader's admission of it)
    pub = [r for r in doc["rows"] if r.get("wave") is None]
    pmx = V.load_matrix([dict(r) for r in pub])
    held = [r for r in pub if r["status"] == "held"]
    d8 = [r for r in held if r["view"] == "it" and r["cell"] in ("constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce")]
    admitted = [r for r in held if r not in d8]
    assert len(d8) == 4 and all(r["route_kind"] == "prompted" and r["held_reason"].startswith("D28 unscheduled") and r["index"] >= 42 for r in admitted)
    assert pmx["counts"]["held"] == 4 + len(admitted) and pmx["counts"]["rows"] == 38 + pmx["counts"]["held"] == 42 and pmx["counts"]["launched"] == 38 and pmx["counts"]["declared"] == 4
    assert pmx["counts"]["expected_refusals"] == 0 and pmx["counts"]["contrasts_designed"] == 32
    assert mx["counts"]["held"] == pmx["counts"]["held"] and mx["counts"]["declared"] == 4 and mx["counts"]["expected_refusals"] == 0
    assert mx["counts"]["launched"] == 38 + sum(1 for r in doc["rows"] if r.get("wave") is not None and r["status"] == "launch")
    assert {r.get("wave") for r in doc["rows"]} >= {None, "ext", "steer"} and all(r["status"] == "launch" for r in doc["rows"] if r.get("wave"))
    assert not any(r.get("object_source") == V.OBJECT_SOURCE_ITDOC or r.get("wave") == "itdoc" for r in doc["rows"]), "the it-doc wave is matrix_itdoc.json's"
    assert A.load_expected_refusals(mx, None) == {}
    counts = doc["counts"]
    assert counts["launched_by_wave"]["published"] == 38 and counts["expected_refusals"] == 0 and counts.get("expected_refusal_rows", []) == []
    assert counts["servable"] == counts["launched"] == mx["counts"]["launched"] and counts["contrasts_computable"] == counts["contrasts_designed"]
    for r in doc["rows"]:
        assert not r.get("expected_refusal") and r.get("expected_refusal_reason") in (None, "")
        if r["status"] == "launch" and r["declared"]:
            assert tuple(r["expected_stamps"]) == V.STAMPS_DECLARED
    assert {(r["model_key"], r["cell"]) for r in mx["launch"] if r["declared"]} == {(GEMMA, SI12_PSR), (GEMMA, SI12_LORA), (QWEN, SI12_PSR), (QWEN, SI12_LORA)}


_REAL_STORE = (SITE.expand_path("$GRADE/tinycap/math500/pull_928a944/controls/nemotron3nano4b")
               / "constv2-mix-full--lora_r4_ce_minus_ctl" / "thinking_card")


def test_synthetic_store_layout_matches_the_real_producer_store(full_store):
    """The fixture's layout against a REAL pulled MATH-500 store of the same producer (present on the analysis host
    only; skipped elsewhere): the cell-dir name shape, the file layout, the manifest / outcome / row / summary /
    evidence / ledger / request-evidence field sets. Fixture-only extras are named, never silently allowed."""
    if not _REAL_STORE.is_dir():
        pytest.skip(f"no real producer store at {_REAL_STORE}")
    real_cd = next(p for p in (_REAL_STORE / "cells").iterdir() if p.is_dir())
    real_task = next(iter(json.loads((real_cd / "gen_stats.json").read_text())))
    root = full_store.root / row(QWEN, "main", CONST_PSR)["out_root"]
    cd = next((root / "cells").glob("*"))
    import re
    shape = re.compile(r"^[a-z0-9_.-]+__controls__t0__a0__[0-9a-f]{8}__[0-9a-f]{8}$")
    assert shape.match(real_cd.name) and shape.match(cd.name)

    def layout(cell_dir, task):
        t = task.replace("/", "__")
        files = {str(p.relative_to(cell_dir)).replace(t, "<task>") for p in cell_dir.rglob("*") if p.is_file()}
        return {re.sub(r"[^/]+\.eval$", "<log>.eval", f) for f in files}

    real_layout, synth_layout = layout(real_cd, real_task), layout(cd, V.TASK)
    fixture_only = {f"private/{G.CHANNELS_NAME}"}
    # the figure venv carries no inspect_ai, so the fixture writes no .eval there: that one entry is expected absent in that venv only
    log_entry = {f for f in real_layout if f.endswith("<log>.eval")}
    assert len(log_entry) == 1 and next(iter(log_entry)).startswith(f"{C.INSPECT_LOGS_SUBDIR}/<task>__s0/")
    want = real_layout if HAS_INSPECT else real_layout - log_entry
    assert synth_layout - fixture_only == want, (sorted(synth_layout - want), sorted(want - synth_layout))
    real_root_files = {p.name for p in _REAL_STORE.iterdir() if p.is_file()}
    synth_root_files = {p.name for p in root.iterdir() if p.is_file()}
    root_fixture_only = {"endpoint.json", CL.RECORD_NAME}          # the standup's endpoint record and the planner's record: written after the pull the real store came from
    assert synth_root_files - root_fixture_only <= real_root_files and {C.LEDGER_FILENAME, C.REQUEST_EVIDENCE_FILENAME, "evidence.json", G.RESULT_NAME} <= synth_root_files
    rm, sm = json.loads((real_cd / "manifest.json").read_text()), json.loads((cd / "manifest.json").read_text())
    reader_keys = {"cell_id", "harness_cell_id", "virtual_model", "intervention_hash", "harness_config_hash", "model_id", "model_revision",
                   C.MANIFEST_KEY_PRODUCER, C.MANIFEST_KEY_PROVENANCE, C.MANIFEST_KEY_TINYCAP, "resolved_config", C.MANIFEST_KEY_ROUTE, C.MANIFEST_KEY_THINKING,
                   C.MANIFEST_KEY_GEOMETRY, C.MANIFEST_KEY_SAMPLING, C.MANIFEST_KEY_USAGE_LEDGER, C.MANIFEST_KEY_REQUEST_EVIDENCE, "proxy_upstream_timeout_s",
                   "binding", C.MANIFEST_KEY_RENDER_FACT_EVIDENCE, C.MANIFEST_KEY_RESUME_ATTEMPTS, C.LENS_SERVED_POSITIONS_KEY, "header_located", "support",
                   C.MANIFEST_KEY_TRANSFER_PROVENANCE, "ledger_thinking_refusals", "ledger_url"}
    assert reader_keys <= set(rm) and reader_keys <= set(sm)
    assert set(rm[C.MANIFEST_KEY_ROUTE]) == set(sm[C.MANIFEST_KEY_ROUTE]) and set(rm["binding"]) == set(sm["binding"])
    assert set(rm[C.MANIFEST_KEY_TINYCAP]["served"]) == set(sm[C.MANIFEST_KEY_TINYCAP]["served"]) and set(rm[C.MANIFEST_KEY_THINKING]) == set(sm[C.MANIFEST_KEY_THINKING])
    assert set(rm[C.MANIFEST_KEY_TRANSFER_PROVENANCE]) >= set(sm[C.MANIFEST_KEY_TRANSFER_PROVENANCE]) >= {"family", "header_located", "served_header_ids", "served_positions", "served_render"}
    assert set(rm[C.MANIFEST_KEY_RESUME_ATTEMPTS][0]) <= set(sm[C.MANIFEST_KEY_RESUME_ATTEMPTS][0])      # lane H's entries add attempt_id / kind
    ro, so = json.loads((real_cd / "outcomes" / C.task_outcomes_filename(real_task)).read_text()), completion.read_task_outcomes(cd, V.TASK)
    assert set(ro) == set(so) and set(ro["items"][0]) == set(so["items"][0])
    rr, sr = json.loads((real_cd / "rows.jsonl").read_text().splitlines()[0]), json.loads((cd / "rows.jsonl").read_text().splitlines()[0])
    assert set(json.loads(rr["extra_json"])) == set(json.loads(sr["extra_json"])) and {"task", "item_id", "metric", "value", "extra_json", "harness_config_hash", "intervention_hash"} <= set(sr)
    rs, ss = json.loads((real_cd / "summary" / C.task_outcomes_filename(real_task)).read_text()), completion.read_task_summary(cd, V.TASK)
    assert set(A._CONTROLS_SUMMARY_FIELDS) <= set(rs) and set(A._CONTROLS_SUMMARY_FIELDS) <= set(ss)
    re_, se = json.loads((real_cd / "evidence.json").read_text()), json.loads((cd / "evidence.json").read_text())
    assert {G.S.EVIDENCE_RENDER_CERTIFIED, G.S.EVIDENCE_LENS_POSITION, G.S.EVIDENCE_LENS_ON_BITE, C.OOS_EVIDENCE_BITES_BLOCK, "cell_id", "virtual_model"} <= set(re_) & set(se)
    rl = json.loads((_REAL_STORE / C.LEDGER_FILENAME).read_text().splitlines()[0])
    sl = json.loads((root / C.LEDGER_FILENAME).read_text().splitlines()[0])
    assert set(rl) == set(C.LEDGER_ROW_FIELDS) and set(sl) == set(C.LEDGER_ROW_FIELDS) | {"attempt_id"}
    rq = json.loads((_REAL_STORE / C.REQUEST_EVIDENCE_FILENAME).read_text().splitlines()[0])
    sq = json.loads((root / C.REQUEST_EVIDENCE_FILENAME).read_text().splitlines()[0])
    # ifeval-canonical-battery (P29): a record is held to the field set of the version it was WRITTEN under —
    # the real store is version 2, the synthetic one is written by the current (version 3) producer
    assert rq["request_evidence_version"] == 2 and set(rq) == set(C.REQUEST_EVIDENCE_FIELDS_V2)
    assert set(sq) == set(C.REQUEST_EVIDENCE_FIELDS) | {"attempt_id"}
    assert "result=COMPLETE" in (_REAL_STORE / G.RESULT_NAME).read_text() and "result=COMPLETE" in (root / G.RESULT_NAME).read_text()


# --------------------------------------------------------------------------- #
# the served-positions RECORD GAP (served-positions-null-diagnosis.md §4.2): a DECLARED cell's manifest carries the   #
# producer's null (the bare-key lookup at 8f5f1495); the fact is resolved from the standup record beside the job, or  #
# the leg refuses by name — never accepted as null                                                                     #
# --------------------------------------------------------------------------- #

DECL = (QWEN, "main", SI12_PSR)
DECLARED_KEY = C.cell_spellings(SI12_PSR, declared=True).harness
BARE_KEY = C.cell_spellings(SI12_PSR, declared=False).harness


def _declared_wave(root, **over):
    rows = small_matrix((QWEN, "main", NONE), DECL)
    G.build_all(root, rows=rows, overrides={DECL: over})
    return rows, analyze(root, matrix=rows, overlap=None)


def test_record_gap_i_a_declared_store_with_the_producers_null_resolves_from_the_standup_record(full_report, tmp_path):
    """(i) the four declared legs of the full synthetic wave carry the producer's NULL in all three manifest fields
    and PASS A0 / A3 (and A2's probe inventory) with the record gap filed on the leg, in the completeness gate, in the
    provenance and in the report's `record_gaps` block naming the producer cause; an undeclared leg carries none."""
    rep = full_report
    declared = sorted(k for k, leg in rep["legs"].items() if leg["declared"] and leg["fitted"])
    assert len(declared) == 4
    for k in declared:
        leg = rep["legs"][k]
        assert leg["refused"] is None and leg["identity"][C.MANIFEST_KEY_TRANSFER_PROVENANCE]["served_positions"] is None
        gaps = leg[V.RECORD_GAPS_KEY]
        assert [g["field"] for g in gaps] == list(V.RECORD_GAP_FIELDS) == ["transfer_provenance.served_positions", "route.lens_served_positions",
                                                                           "lens_served_positions"]
        assert all(g["manifest"] is None and g["resolved"] == ["response"] and g["cause"] == V.RECORD_GAP_CAUSE_BARE_KEY_LOOKUP for g in gaps)
        assert all(g["resolved_from"].endswith(f"/{V.STANDUP_TASK_SUBDIR}/{V.STANDUP_ROUTING_TABLE}[{C.LENS_SERVED_POSITIONS_KEY}]['si_safety__"
                                               + leg["cell"].split("__", 1)[1] + "@thinking']") for g in gaps)
        cert = leg["certification"]
        assert cert["passed"], cert["failures"]
        assert cert["route_lens_served_positions"] == ["response"] and cert["route_lens_served_positions_source"] == "standup_record"
        assert cert["manifest_lens_served_positions_source"] == "standup_record" and cert["standup_record"]["resolved"] == ["response"]
        assert cert[V.EVIDENCE_THINKING_HEADER_BY_ARM]["header_certified"] is True
        assert cert[M.EVIDENCE_LENS_POSITION] is None and cert[M.EVIDENCE_LENS_ON_BITE] is None
        assert leg["request_evidence"]["passed"], leg["request_evidence"]["failures"]
        assert leg["standup_record"]["problem"] is None and leg["standup_record"]["job_id"] and leg["standup_record"]["jobdir"].endswith(leg["standup_record"]["job_id"])
        assert rep["gates"]["A0_completeness"]["legs"][k][V.RECORD_GAPS_KEY] == 3
        prov = rep["provenance"]["cells"][k]
        assert prov["served_positions"] == ["response"] and prov["served_positions_source"] == "standup_record" and prov[V.RECORD_GAPS_KEY] == gaps
    block = rep[V.RECORD_GAPS_KEY]
    assert block["n_legs"] == 4 and block["n_gaps"] == 12 and sorted(block["legs"]) == declared
    assert block["by_cause"] == {V.RECORD_GAP_CAUSE_BARE_KEY_LOOKUP: 12} and block["by_field"] == {f: 4 for f in V.RECORD_GAP_FIELDS}
    assert "8f5f1495" in V.RECORD_GAP_CAUSE_BARE_KEY_LOOKUP and "served_capability_run.py:4007-4008" in V.RECORD_GAP_CAUSE_BARE_KEY_LOOKUP
    gap_findings = [f for f in rep["findings"] if ": RECORD GAP — " in f]
    assert len(gap_findings) == 4 and all(V.RECORD_GAP_CAUSE_BARE_KEY_LOOKUP in f and "null in the manifest" in f for f in gap_findings)
    assert sum(1 for t in rep["top_items"] if ": RECORD GAP — " in t) == 4 and not any("FAILED" in t for t in rep["top_items"])
    assert A.production_refusals(rep) == []
    # an undeclared fitted leg: no gap, the manifest states its positions, no standup record was read
    base = leg_of(rep, QWEN, "main", CONST_PSR)
    assert V.RECORD_GAPS_KEY not in base and "standup_record" not in base
    assert base["certification"]["route_lens_served_positions"] == ["all"] and base["certification"]["route_lens_served_positions_source"] == "manifest"
    assert base["certification"]["standup_record"] is None and rep["provenance"]["cells"][base["leg"]][V.RECORD_GAPS_KEY] == []
    # predicate (B) on the invariant-header family (gemma) and the moving-header family (qwen): both certified; the
    # invariance verdict is the FACT's, never read as a pass
    for key, verified in ((GEMMA, True), (QWEN, False)):
        h = leg_of(rep, key, "main", SI12_PSR)["certification"][V.EVIDENCE_THINKING_HEADER_BY_ARM]
        assert h["verified"] is verified and h["invariant"] is verified and h["overridden_refusal_present"] is (not verified)
        assert h["header_certified"] is True and h["header_attested_agrees"] is True and h["certification_problems"] == []
    # the §4.3-fixed producer writes the value: the same pass, NO gap, the manifest the source
    rows = small_matrix((QWEN, "main", NONE), DECL)
    G.build_all(tmp_path / "fixed", rows=rows, overrides={DECL: {"producer_records_served_positions": True}})
    rep2 = analyze(tmp_path / "fixed", matrix=rows, overlap=None)
    leg = leg_of(rep2, *DECL)
    assert leg["refused"] is None and V.RECORD_GAPS_KEY not in leg and leg["certification"]["passed"], leg["certification"]["failures"]
    assert leg["identity"][C.MANIFEST_KEY_TRANSFER_PROVENANCE]["served_positions"] == ["response"]
    assert leg["certification"]["route_lens_served_positions_source"] == "manifest" and rep2[V.RECORD_GAPS_KEY]["n_legs"] == 0
    assert leg["standup_record"]["problem"] is None                      # the record is read either way; it bound


def test_record_gap_ii_the_same_store_without_the_standup_record_refuses_by_name(tmp_path):
    """(ii) the declared store with the producer's null and NO pulled standup record refuses `STANDUP_RECORD_MISSING`
    (A0 FAILS, the leg unusable); so does a record directory lacking a file, and a manifest naming no routing table."""
    rows, rep = _declared_wave(tmp_path / "none", standup_record=False)
    leg = leg_of(rep, *DECL)
    assert leg["refused"] and V.STANDUP_RECORD_MISSING in leg["refused"] and "None != ['response']" in leg["refused"] and "DECLARED" in leg["refused"]
    assert leg["standup_record"]["problem"].startswith(V.STANDUP_RECORD_MISSING) and leg["standup_record"]["resolved"] is None
    assert rep["gates"]["A0_completeness"]["status"] == A.GATE_FAIL and V.leg_key(*DECL, 8192) in rep["unusable_legs"]
    assert rep[V.RECORD_GAPS_KEY]["n_legs"] == 0 and not contrast_of(rep, *DECL)["computable"]
    assert V.STANDUP_RECORD_MISSING in rep["unusable_legs"][V.leg_key(*DECL, 8192)] and A.production_refusals(rep)
    # the directory is there but a file is not
    rows, rep = _declared_wave(tmp_path / "partial")
    (tmp_path / "partial" / V.STANDUP_DIR_PARENT / G.standup_jobdir(row(*DECL), 8192) / V.STANDUP_TASK_SUBDIR / V.STANDUP_REGISTRY).unlink()
    rep = analyze(tmp_path / "partial", matrix=rows, overlap=None)
    leg = leg_of(rep, *DECL)
    assert leg["refused"] and V.STANDUP_RECORD_MISSING in leg["refused"] and "lacks" in leg["refused"] and V.STANDUP_REGISTRY in leg["refused"]
    # the manifest names no routing table / another layout: the record cannot be located
    rows, rep = _declared_wave(tmp_path / "noroute", manifest_edit=lambda m: m.pop("routing_table"))
    assert V.STANDUP_RECORD_MISSING in leg_of(rep, *DECL)["refused"] and "names no routing_table" in leg_of(rep, *DECL)["refused"]
    rows, rep = _declared_wave(tmp_path / "layout", manifest_edit=lambda m: m.update(routing_table="/elsewhere/routing_table.json"))
    assert V.STANDUP_RECORD_MISSING in leg_of(rep, *DECL)["refused"] and "not a standup record path" in leg_of(rep, *DECL)["refused"]


def test_record_gap_iii_a_routing_table_naming_the_bare_key_too_or_all_refuses_and_so_does_every_broken_binding(tmp_path):
    """(iii) a routing table naming the bare key beside the declared one is not a declared standup's shape; `["all"]`
    under the declared key is not the declared mode; and each of the cross-checks (the routes entry's hash and vmodel,
    the registry's positions / declaration / digest / expect hash, the attestation's and the standup's job id, the tree
    sha, the manifest's declared-class facts) refuses `STANDUP_RECORD_MISMATCH` by name."""
    def bare_too(docs):
        docs[G.STANDUP_ROUTING_TABLE][C.LENS_SERVED_POSITIONS_KEY][BARE_KEY] = ["response"]
    rows, rep = _declared_wave(tmp_path / "bare", standup_record_edit=bare_too)
    leg = leg_of(rep, *DECL)
    assert leg["refused"] and V.STANDUP_RECORD_MISMATCH in leg["refused"] and "BARE key" in leg["refused"] and BARE_KEY in leg["refused"]
    assert rep["gates"]["A0_completeness"]["status"] == A.GATE_FAIL and rep[V.RECORD_GAPS_KEY]["n_legs"] == 0

    def all_mode(docs):
        docs[G.STANDUP_ROUTING_TABLE][C.LENS_SERVED_POSITIONS_KEY][DECLARED_KEY] = ["all"]
    rows, rep = _declared_wave(tmp_path / "all", standup_record_edit=all_mode)
    leg = leg_of(rep, *DECL)
    assert leg["refused"] and V.STANDUP_RECORD_MISMATCH in leg["refused"] and "['all'] != ['response']" in leg["refused"]

    def other_cell(docs):
        docs[G.STANDUP_ROUTING_TABLE][C.LENS_SERVED_POSITIONS_KEY] = {"si_other__psr_kl_all@thinking": ["response"]}
    rows, rep = _declared_wave(tmp_path / "other", standup_record_edit=other_cell)
    assert "not the declared key" in leg_of(rep, *DECL)["refused"]
    cases = {
        "routes_hash": (lambda d: d[G.STANDUP_ROUTING_TABLE]["routes"][1].update(intervention_hash="0" * 16), "intervention_hash"),
        "routes_vmodel": (lambda d: d[G.STANDUP_ROUTING_TABLE]["routes"][1].update(virtual_model="petri__other"), "virtual_model"),
        "routes_twice": (lambda d: d[G.STANDUP_ROUTING_TABLE]["routes"].append(dict(d[G.STANDUP_ROUTING_TABLE]["routes"][1])), "2 time(s)"),
        "registry_positions": (lambda d: d[G.STANDUP_REGISTRY]["entries"][0]["resolved_spec"].update(positions="all"), "resolved_spec.positions"),
        "registry_declaration": (lambda d: d[G.STANDUP_REGISTRY]["entries"][0]["resolved_spec"].update({C.SERVE_OUT_OF_SUPPORT_KEY: []}), C.SERVE_OUT_OF_SUPPORT_KEY),
        "registry_digest": (lambda d: d[G.STANDUP_REGISTRY]["entries"][0].update(artifact_sha16="f" * 16), "artifact_sha16"),
        "registry_expect_hash": (lambda d: d[G.STANDUP_REGISTRY]["entries"][0].update(expect_hash="0" * 16), "expect_hash"),
        "attestation_job": (lambda d: d[G.STANDUP_ATTESTATION].update(slurm_job_id="1"), "slurm_job_id"),
        "standup_job": (lambda d: d[G.STANDUP_JOBS][0].update(job_id="1"), "job_id"),
        "tree": (lambda d: d[G.STANDUP_JOBS][0].update(sha="1" * 40), "tree_sha"),
    }
    for name, (edit, word) in cases.items():
        rows, rep = _declared_wave(tmp_path / name, standup_record_edit=edit)
        leg = leg_of(rep, *DECL)
        assert leg["refused"] and V.STANDUP_RECORD_MISMATCH in leg["refused"] and word in leg["refused"], (name, leg["refused"])
    # the manifest's own declared-class facts are part of the binding
    manifest_cases = {
        "header_located": (lambda m: m.update(header_located=False), "manifest header_located"),
        "tp_header_located": (lambda m: m[C.MANIFEST_KEY_TRANSFER_PROVENANCE].update(header_located=False), f"{C.MANIFEST_KEY_TRANSFER_PROVENANCE}.header_located"),
        "declaration": (lambda m: m[C.MANIFEST_KEY_ROUTE][C.ROUTE_KEY_SERVE_DECLARATIONS].pop(DECLARED_KEY), C.ROUTE_KEY_SERVE_DECLARATIONS),
        "header_ids": (lambda m: m[C.MANIFEST_KEY_TRANSFER_PROVENANCE]["served_header_ids"].pop("on"), "served_header_ids"),
    }
    for name, (edit, word) in manifest_cases.items():
        rows, rep = _declared_wave(tmp_path / f"m_{name}", manifest_edit=edit)
        leg = leg_of(rep, *DECL)
        assert leg["refused"] and word in leg["refused"], (name, leg["refused"])
    # the job dir's name must end in the job id (the binding's anchor): a renamed record directory refuses. The record
    # renamed is the DECLARED leg's own (selected by its job id) — serving/ holds every leg's record (the bare leg's too
    # since the launch attestation rides with every standup), so the first directory listed is not necessarily it
    rows, rep = _declared_wave(tmp_path / "renamed")
    d = tmp_path / "renamed" / V.STANDUP_DIR_PARENT
    bound = leg_of(rep, *DECL)["standup_record"]
    assert bound["resolved"] and bound["jobdir"], bound
    src = d / bound["jobdir"]
    assert src.is_dir(), (src, sorted(q.name for q in d.iterdir()))
    src.rename(d / (src.name + "_x"))
    rep = analyze(tmp_path / "renamed", matrix=rows, overlap=None)
    assert V.STANDUP_RECORD_MISSING in leg_of(rep, *DECL)["refused"]              # the manifest's path no longer resolves


def test_record_gap_iv_an_undeclared_store_with_a_null_still_refuses(tmp_path):
    """(iv) an UNDECLARED fitted store whose manifest carries a null served-positions fact refuses exactly as before —
    the standup record is a DECLARED cell's resolution and is never consulted on an undeclared row."""
    rows = small_matrix((QWEN, "main", NONE), (QWEN, "main", CONST_PSR))

    def null_it(m):
        m[C.MANIFEST_KEY_TRANSFER_PROVENANCE]["served_positions"] = None
        m[C.MANIFEST_KEY_ROUTE][C.LENS_SERVED_POSITIONS_KEY] = None
        m[C.LENS_SERVED_POSITIONS_KEY] = None
    G.build_all(tmp_path, rows=rows, overrides={(QWEN, "main", CONST_PSR): {"manifest_edit": null_it}})
    rep = analyze(tmp_path, matrix=rows, overlap=None)
    leg = leg_of(rep, QWEN, "main", CONST_PSR)
    assert leg["refused"] == f"{C.MANIFEST_KEY_TRANSFER_PROVENANCE}.served_positions None != ['all']"
    assert "standup_record" not in leg and rep[V.RECORD_GAPS_KEY]["n_legs"] == 0 and rep["gates"]["A0_completeness"]["status"] == A.GATE_FAIL
    # no DECLARED standup record (the registry the served-positions fact is read from) is written for an undeclared row; the
    # LAUNCH record every leg carries since C2 (attestation + server log) is a different record and lives beside it
    assert not list((tmp_path / V.STANDUP_DIR_PARENT).rglob(V.STANDUP_REGISTRY.split("/")[-1]))
    assert list((tmp_path / V.STANDUP_DIR_PARENT).rglob(A.M.LAUNCH_LOG_FILE))
    # and a declared row whose OTHER two fields are null while transfer_provenance states the value: A3 resolves them
    # from the record too (the gate is class-aware, never a fall-through)
    rows2 = small_matrix((QWEN, "main", NONE), DECL)

    def route_null(m):
        m[C.MANIFEST_KEY_ROUTE][C.LENS_SERVED_POSITIONS_KEY] = None
        m[C.LENS_SERVED_POSITIONS_KEY] = None
    G.build_all(tmp_path / "route", rows=rows2, overrides={DECL: {"producer_records_served_positions": True, "manifest_edit": route_null}})
    leg = leg_of(analyze(tmp_path / "route", matrix=rows2, overlap=None), *DECL)
    assert leg["refused"] is None and [g["field"] for g in leg[V.RECORD_GAPS_KEY]] == ["route.lens_served_positions", "lens_served_positions"]
    assert leg["certification"]["passed"] and leg["certification"]["route_lens_served_positions_source"] == "standup_record"


def test_a3_on_a_declared_row_requires_predicate_b_and_refuses_the_undeclared_cells_records(tmp_path):
    """The class-aware certification: a DECLARED row is held to predicate (B)'s record and the ON bites, and REFUSES a
    (B′) / (C) block (the driver never writes one there — its presence is another producer's record); `verified` is
    held to the family fact (qwen's header moves: a `verified: true` there is not the driver's record)."""
    def a3(name, **over):
        _rows, rep = _declared_wave(tmp_path / name, **over)
        return rep["gates"]["A3_certification"]

    g = a3("lp", evidence_edit=lambda ev: ev.update({M.EVIDENCE_LENS_POSITION: {"served_positions": ["response"], "header_located": True}}))
    assert g["status"] == A.GATE_FAIL and f"{M.EVIDENCE_LENS_POSITION} block on a DECLARED" in g["detail"]
    g = a3("bite", evidence_edit=lambda ev: ev.update({M.EVIDENCE_LENS_ON_BITE: {"applicable": True, "verified": True}}))
    assert g["status"] == A.GATE_FAIL and f"{M.EVIDENCE_LENS_ON_BITE} block on a DECLARED" in g["detail"]
    g = a3("nohdr", evidence_edit=lambda ev: ev.pop(G.EVIDENCE_THINKING_HEADER_BY_ARM))
    assert g["status"] == A.GATE_FAIL and "predicate (B) was not recorded" in g["detail"]

    def uncertified(ev):
        ev[G.EVIDENCE_THINKING_HEADER_BY_ARM].update(header_certified=False, certification_problems=["synthetic: header mismatch"])
    g = a3("uncert", evidence_edit=uncertified)
    assert g["status"] == A.GATE_FAIL and "header_certified is False" in g["detail"] and "certification_problems" in g["detail"]

    def faked_verified(ev):
        ev[G.EVIDENCE_THINKING_HEADER_BY_ARM].update(verified=True, invariant=True, header_moves=False)
        ev[G.EVIDENCE_THINKING_HEADER_BY_ARM].pop("overridden_refusal")
    g = a3("verified", evidence_edit=faked_verified)
    assert g["status"] == A.GATE_FAIL and "MOVING-header family" in g["detail"] and "header_moves" in g["detail"]
    g = a3("fallback", evidence_edit=lambda ev: ev[G.EVIDENCE_THINKING_HEADER_BY_ARM].update(source="local_tokenizer", fallback_reason="synthetic"))
    assert g["status"] == A.GATE_FAIL and "fallback_reason" in g["detail"]
    g = a3("onearm", evidence_edit=lambda ev: ev[G.EVIDENCE_THINKING_HEADER_BY_ARM].update(measured_arms=["on"]))
    assert g["status"] == A.GATE_FAIL and "measured_arms" in g["detail"]

    def other_ids(ev):
        ev[G.EVIDENCE_THINKING_HEADER_BY_ARM]["attested_thinking_header"]["on_ids"] = [1, 2, 3]
    g = a3("attested", evidence_edit=other_ids)
    assert g["status"] == A.GATE_FAIL and "attested_thinking_header" in g["detail"]
    # the bites probe: OFF-rendered, or a steered stamp that is not the declared triple, or the undeclared body
    g = a3("offprobe", evidence_edit=lambda ev: ev[C.OOS_EVIDENCE_BITES_BLOCK].update({C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM: C.THINKING_OFF}))
    assert g["status"] == A.GATE_FAIL and C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM in g["detail"]

    def wrong_stamp(ev):
        ev[C.OOS_EVIDENCE_BITES_BLOCK]["stamps_by_leg"]["steered"][2]["resumed_policy"] = "always"
    g = a3("stamp", evidence_edit=wrong_stamp)
    assert g["status"] == A.GATE_FAIL and "stamp triple" in g["detail"] and "declared" in g["detail"]
    g = a3("body", evidence_edit=lambda ev: ev[C.OOS_EVIDENCE_BITES_BLOCK].update(probe_extra_body=C.chat_template_kwargs(C.THINKING_ON)))
    assert g["status"] == A.GATE_FAIL and "probe_extra_body" in g["detail"]
    # an undeclared row is unchanged: its (B′) / (C) records are REQUIRED, predicate (B)'s block is not read
    rows = small_matrix((QWEN, "main", NONE), (QWEN, "main", CONST_PSR))
    G.build_all(tmp_path / "undecl", rows=rows, overrides={(QWEN, "main", CONST_PSR): {"evidence_edit": lambda ev: ev.pop(M.EVIDENCE_LENS_POSITION)}})
    g = analyze(tmp_path / "undecl", matrix=rows, overlap=None)["gates"]["A3_certification"]
    assert g["status"] == A.GATE_FAIL and "predicate (B′) was not recorded" in g["detail"]


def test_the_declared_probe_inventory_and_the_fixtures_declared_shape_are_the_producers():
    """The reader's declared-class constants are the driver's spellings (pinned to the fixture's and the MATH module's),
    and the declared probe inventory is what the four real declared cells record at 8f5f1495 (counted on the pulled
    request evidence): bites thinking-ON on the route and the twin, render OFF + ON, NO lens-ON bite."""
    assert V.EVIDENCE_THINKING_HEADER_BY_ARM == G.EVIDENCE_THINKING_HEADER_BY_ARM == "thinking_header_by_arm"
    assert M.RENDER_SOURCE_SERVED == G.HEADER_SOURCE_SERVED == "served_tokenize"
    assert (V.STANDUP_ROUTING_TABLE, V.STANDUP_JOBS, V.STANDUP_ATTESTATION, V.STANDUP_REGISTRY) == (G.STANDUP_ROUTING_TABLE, G.STANDUP_JOBS,
                                                                                                   G.STANDUP_ATTESTATION, G.STANDUP_REGISTRY)
    assert (V.STANDUP_DIR_PARENT, V.STANDUP_TASK_SUBDIR) == (G.STANDUP_DIR_PARENT, G.STANDUP_TASK_SUBDIR) == ("serving", "gsm8k")
    assert V.RECORD_GAP_FIELDS == (f"{C.MANIFEST_KEY_TRANSFER_PROVENANCE}.served_positions", f"{C.MANIFEST_KEY_ROUTE}.{C.LENS_SERVED_POSITIONS_KEY}",
                                   C.LENS_SERVED_POSITIONS_KEY)
    assert V.THINKING_HEADER_CERTIFIED_FLAGS == (C.OOS_EVIDENCE_HEADER_CERTIFIED, "header_attested_agrees", C.OOS_EVIDENCE_PROMPT_OPENED_THINK_CERTIFIED)
    # the registry's content identity is the PRODUCER's function over the file's bytes (vllm_server._attestation_bindings)
    from sisteer.capeval import manifest as CM
    assert V.registry_index_sha16 is CM.sha256_hex16 and V.ATTESTATION_KEY_REGISTRY_INDEX_SHA16 == "registry_index_sha16"
    server_src = (REPO / "src" / "sisteer" / "capeval" / "serving" / "vllm_server.py").read_text()
    assert 'registry_index_sha = cm.sha256_hex16(index.read_bytes())' in server_src and '"registry_index_sha16": registry_index_sha' in server_src
    docs = G.standup_record_docs(row(*DECL), G.RENDER_FACTS["qwen3_5"], cap=8192, tree_sha=G.FULL_SHA, intervention_hash="0" * 16, max_model_len=11008)
    assert docs[G.STANDUP_ATTESTATION]["registry_index_sha16"] == CM.sha256_hex16(G.standup_doc_bytes(G.STANDUP_REGISTRY, docs[G.STANDUP_REGISTRY]))
    fact = G.RENDER_FACTS["qwen3_5"]
    ev = {C.OOS_EVIDENCE_BITES_BLOCK: G.steering_bites_record_declared(SI12_PSR), G.EVIDENCE_THINKING_HEADER_BY_ARM: G.thinking_header_by_arm_record(fact)}
    vm = C.cell_spellings(SI12_PSR, declared=True).vmodel
    inv, problem = V.probe_inventory_per_run(ev, route_vmodel=vm, fitted=True, declared=True)
    assert problem is None
    assert inv == {(vm, "off", "render"): 1, (vm, "on", "render"): 1, (vm, "on", "bites"): 5, (C.twin_vmodel(), "on", "bites"): 5}
    # the undeclared inventory is the MATH module's, unchanged
    ev_u = {C.OOS_EVIDENCE_BITES_BLOCK: G.S.steering_bites_record(CONST_PSR),
            M.EVIDENCE_LENS_ON_BITE: {"lens_vmodel": "petri__x", "base_vmodel": "petri__base"}}
    assert V.probe_inventory_per_run(ev_u, route_vmodel="petri__x", fitted=True, declared=False) == M.probe_inventory_per_run(ev_u, route_vmodel="petri__x", fitted=True)
    # the fixture's declared evidence carries no (B′)/(C) block and its bites probe is the declared one
    assert G.steering_bites_record_declared(SI12_PSR)[C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM] == "on"
    assert G.steering_bites_record_declared(SI12_PSR)["stamps_by_leg"]["steered"][0] == dict(zip(C.EXPECTED_STAMP_TRIPLE_FIELDS, V.STAMPS_DECLARED))
    assert G.standup_endpoint(row(*DECL))[C.LENS_SERVED_POSITIONS_KEY] == {DECLARED_KEY: ["response"]}
    assert G.standup_endpoint(row(QWEN, "main", CONST_PSR))[C.LENS_SERVED_POSITIONS_KEY] == {C.cell_spellings(CONST_PSR, declared=False).harness: ["all"]}
    with pytest.raises(AssertionError):
        G.lens_position_record(row(*DECL))


_REAL_G8K_PULL = SITE.expand_path("$GRADE/tinycap/gsm8k/pull_8f5f149")


def test_the_four_real_declared_manifests_resolve_from_their_pulled_standup_records():
    """On the analysis host only (skipped elsewhere): the four real declared stores of the 8f5f1495 wave carry the
    producer's null and resolve `["response"]` from their pulled standup records with three gaps each; their real
    evidence passes the class-aware certification and their declared probe inventory derives."""
    matrix = REPO / "configs" / "sdfbase" / "gsm8k" / "matrix.json"
    if not (_REAL_G8K_PULL / "controls").is_dir() or not (_REAL_G8K_PULL / V.STANDUP_DIR_PARENT).is_dir():
        pytest.skip(f"no pulled wave with standup records at {_REAL_G8K_PULL}")
    facts = M.load_render_facts(None)
    n = 0
    for r in V.load_matrix(matrix)["launch"]:
        if not (r["declared"] and r["fitted"]):
            continue
        root = _REAL_G8K_PULL / r["out_root"]
        cells = list((root / "cells").glob("*")) if (root / "cells").is_dir() else []
        if len(cells) != 1:
            pytest.skip(f"no single store under {root}")
        manifest = completion.read_store_manifest(cells[0])
        assert manifest[C.MANIFEST_KEY_TRANSFER_PROVENANCE]["served_positions"] is None and manifest[C.LENS_SERVED_POSITIONS_KEY] is None
        rec = V.declared_served_positions_record(manifest, row=r, gen_root=_REAL_G8K_PULL, fact=facts[r["family"]])
        assert rec["problem"] is None and rec["resolved"] == ["response"] and len(rec["gaps"]) == 3, rec["problem"]
        assert rec["jobdir"].endswith(rec["job_id"]) and rec["job_id"].isdigit()
        att = json.loads((Path(rec["standup_dir"]) / V.STANDUP_ATTESTATION).read_text())
        assert rec["registry_index_sha16"] == att["registry_index_sha16"] == V.registry_index_sha16((Path(rec["standup_dir"]) / V.STANDUP_REGISTRY).read_bytes())
        evidence = json.loads((cells[0] / "evidence.json").read_text())
        cert = V.certification_check(evidence, manifest, row=r, fact=facts[r["family"]], standup_record=rec)
        assert cert["passed"], cert["failures"]
        assert cert["route_lens_served_positions"] == ["response"] and cert["route_lens_served_positions_source"] == "standup_record"
        inv, problem = V.probe_inventory_per_run(evidence, route_vmodel=manifest[C.MANIFEST_KEY_ROUTE]["virtual_model"], fitted=True, declared=True)
        assert problem is None and sum(v for (_vm, arm, kind), v in inv.items() if kind == "bites" and arm == "on") == 10
        n += 1
    assert n == 4


def test_a10_refuses_a_complete_overlap_record_from_another_wave(full_store, tmp_path):
    """Review of 337efb7a (HIGH): A10 passed a reused overlap record from another wave over the same 200 ids. The
    CONSUMER now binds the record to the analysis's store pin — a COMPLETE record built by the real checker over stock
    stores of ANOTHER pin fails by name; so does the same-pin record with one stock producer tree edited in memory, a
    record whose stock block was read under another pin, and a record from before the provenance fields."""
    other_full = "abcdef0123456789abcdef0123456789abcdef01"
    root = tmp_path / "otherwave"
    retired = root / "retired"
    for view in VIEWS_RETIRED:
        G.build_retired_store(retired, view, correct=set(range(180)), extraction_failed={3, 7})
    for key in G.MODEL_KEYS:
        G.build_overlap_stock_store(root, key, correct=set(range(178)) | {181, 182}, sha=other_full[:7], tree_sha=other_full)
    other = O.run(retired_root=retired, stock_root=root / "overlap", models=list(G.MODEL_KEYS), canonical_ids=G.IDS, ids_path=None,
                  pairing_rows=G.pairing_rows(), pairing_source="synthetic rows", repo_root=REPO, store_pin=other_full[:7], n_resamples=B_FAST,
                  seed=0, ci=0.95, eval_logs=False)
    assert other["store_pin"] == other_full[:7] and all(b["usable"] and b["comparison"]["complete"] for b in other["models"].values())
    assert all(b["stock"]["producer_tree_sha"] == other_full for b in other["models"].values())
    # the checker's own verdict is COMPLETE; the consumer, analysing pin G.SHA7, refuses it by the pin
    rep = analyze(full_store.root, overlap=other)
    g = rep["gates"]["A10_overlap"]
    assert g["status"] == A.GATE_FAIL and f"store_pin {other_full[:7]!r} is not this analysis's store pin {G.SHA7!r}" in g["detail"]
    assert all(f"{key}: the overlap record's stock producer tree {other_full} does not carry this analysis's store pin" in g["detail"] for key in G.MODEL_KEYS)
    assert "A10_overlap" in rep["gates"]["failed"] and A.production_refusals(rep)
    # the same-pin record with ONE stock producer tree rewritten in memory (the reviewer's probe) fails on that model
    mutated = json.loads(json.dumps(full_store.overlap))
    mutated["models"][QWEN]["stock"]["producer_tree_sha"] = other_full
    g = A.gate_overlap(mutated, list(G.MODEL_KEYS), G.IDS, store_pin=G.SHA7)
    assert g["status"] == A.GATE_FAIL and f"{QWEN}: the overlap record's stock producer tree {other_full}" in g["detail"] and GEMMA + ":" not in g["detail"]
    # the record's pin rewritten alone fails; a stock block read under another pin fails
    mutated = json.loads(json.dumps(full_store.overlap)); mutated["store_pin"] = other_full[:7]
    assert "not this analysis's store pin" in A.gate_overlap(mutated, list(G.MODEL_KEYS), G.IDS, store_pin=G.SHA7)["detail"]
    mutated = json.loads(json.dumps(full_store.overlap)); mutated["models"][GEMMA]["stock"]["store_pin"] = other_full[:7]
    assert f"{GEMMA}: the overlap record's stock block was read under store_pin" in A.gate_overlap(mutated, list(G.MODEL_KEYS), G.IDS, store_pin=G.SHA7)["detail"]
    # a record from before the provenance fields (no store_pin / no producer_tree_sha) is refused, never passed on ids alone
    legacy = json.loads(json.dumps(full_store.overlap)); legacy.pop("store_pin")
    for b in legacy["models"].values():
        b["stock"].pop("producer_tree_sha"); b["stock"].pop("store_pin")
    g = A.gate_overlap(legacy, list(G.MODEL_KEYS), G.IDS, store_pin=G.SHA7)
    assert g["status"] == A.GATE_FAIL and "no usable store_pin" in g["detail"] and "no 40-hex producer_tree_sha" in g["detail"]
    # the two spellings of ONE pin (the runner's 7 and the full commit) are the same pin
    assert A.gate_overlap(full_store.overlap, list(G.MODEL_KEYS), G.IDS, store_pin=G.FULL_SHA)["status"] == A.GATE_PASS
    assert A._same_pin(G.SHA7, G.FULL_SHA) and not A._same_pin(G.SHA7, other_full[:7])


def test_the_registry_is_bound_to_the_attestation_by_its_bytes_before_it_is_read(tmp_path):
    """Review of 337efb7a (HIGH): the resolver compared copied registry fields and ignored the attestation's
    `registry_index_sha16`. Now the registry bytes are hashed with the producer's function and held to the attested
    digest BEFORE parsing: a registry substituted beside a STALE attestation — keeping every checked identifier, changing
    only the thinking-ON header tuple — refuses; an attestation without the digest refuses; a consistently re-sealed
    registry whose header tuple is not the family fact's refuses on the tuple; the sealed fixture resolves."""
    rows, rep = _declared_wave(tmp_path / "ok")
    leg = leg_of(rep, *DECL)
    assert leg["refused"] is None and len(leg[V.RECORD_GAPS_KEY]) == 3
    d = tmp_path / "ok" / V.STANDUP_DIR_PARENT / G.standup_jobdir(row(*DECL), 8192) / V.STANDUP_TASK_SUBDIR
    att = json.loads((d / V.STANDUP_ATTESTATION).read_text())
    assert att["registry_index_sha16"] == V.registry_index_sha16((d / V.STANDUP_REGISTRY).read_bytes())
    # a substituted registry beside the STALE attestation: the checked identifiers untouched, the ON header tuple changed
    reg = json.loads((d / V.STANDUP_REGISTRY).read_text())
    reg["assistant_header_ids_thinking_on"] = [1, 2, 3]
    (d / V.STANDUP_REGISTRY).write_text(json.dumps(reg, indent=1), encoding="utf-8")
    rep = analyze(tmp_path / "ok", matrix=rows, overlap=None)
    leg = leg_of(rep, *DECL)
    assert leg["refused"] and V.STANDUP_RECORD_MISMATCH in leg["refused"] and "registry_index_sha16" in leg["refused"] and "not the registry the workers installed" in leg["refused"]
    assert rep["gates"]["A0_completeness"]["status"] == A.GATE_FAIL and rep[V.RECORD_GAPS_KEY]["n_legs"] == 0
    # even a byte-level re-serialization of the SAME content (a re-indented copy) is not the attested bytes
    rows, rep = _declared_wave(tmp_path / "reser")
    d2 = tmp_path / "reser" / V.STANDUP_DIR_PARENT / G.standup_jobdir(row(*DECL), 8192) / V.STANDUP_TASK_SUBDIR
    (d2 / V.STANDUP_REGISTRY).write_text(json.dumps(json.loads((d2 / V.STANDUP_REGISTRY).read_text()), indent=2), encoding="utf-8")
    assert "registry_index_sha16" in leg_of(analyze(tmp_path / "reser", matrix=rows, overlap=None), *DECL)["refused"]
    # an attestation that records no digest cannot bind the registry: refused by name
    rows, rep = _declared_wave(tmp_path / "nodigest")
    d3 = tmp_path / "nodigest" / V.STANDUP_DIR_PARENT / G.standup_jobdir(row(*DECL), 8192) / V.STANDUP_TASK_SUBDIR
    att = json.loads((d3 / V.STANDUP_ATTESTATION).read_text()); att.pop("registry_index_sha16")
    (d3 / V.STANDUP_ATTESTATION).write_text(json.dumps(att, indent=1), encoding="utf-8")
    assert "records no registry_index_sha16" in leg_of(analyze(tmp_path / "nodigest", matrix=rows, overlap=None), *DECL)["refused"]
    # a CONSISTENT record (re-sealed after the edit) whose header tuple is not the family fact's refuses on the tuple
    def other_header(docs):
        docs[G.STANDUP_REGISTRY]["assistant_header_ids_thinking_on"] = [1, 2, 3]
    rows, rep = _declared_wave(tmp_path / "hdr", standup_record_edit=other_header)
    refused = leg_of(rep, *DECL)["refused"]
    assert refused and "assistant_header_ids_thinking_on" in refused and "registry_index_sha16" not in refused
    # ... and an attestation naming another job's registry_dir refuses
    def other_dir(docs):
        docs[G.STANDUP_ATTESTATION]["registry_dir"] = docs[G.STANDUP_ATTESTATION]["registry_dir"].replace("_8192_", "_16384_")
    rows, rep = _declared_wave(tmp_path / "dir", standup_record_edit=other_dir)
    assert "registry_dir" in leg_of(rep, *DECL)["refused"]


def test_overlap_check_holds_the_outcomes_record_to_the_producer_before_reading_a_value(tmp_path, full_store):
    """Review round 2 (HIGH): the checker held the MANIFEST to the pin and read the outcomes' values without asking
    which tree wrote them. A pinned manifest beside an off-pin outcomes record (`producer_tree_sha` another tree) is
    now refused by the checker WITHOUT --eval-logs, naming both trees, and the consumer (A10) refuses the resulting
    record; a null outcomes producer refuses too; the pin's own store still passes and records both trees."""
    retired = tmp_path / "retired"
    G.build_retired_store(retired, QWEN, correct=set(range(180)))
    G.build_overlap_stock_store(tmp_path / "offpin", QWEN, correct=set(range(178)), outcomes_tree_sha=OTHER_TREE)
    rep = _overlap_run(retired, tmp_path / "offpin" / "overlap", QWEN)
    b = rep["models"][QWEN]
    assert not b["usable"] and b["verdict"] is None and b["comparison"] is None
    assert b["reason"].startswith("outcomes producer:") and OTHER_TREE in b["reason"] and G.FULL_SHA in b["reason"]
    assert b["stock"]["producer_tree_sha"] == G.FULL_SHA and b["stock"]["outcomes_producer_tree_sha"] == OTHER_TREE and b["stock"]["available"] is False
    # the consumer: the record is not usable for that model — A10 FAILS by name
    g = A.gate_overlap(rep, [QWEN], G.IDS, store_pin=G.SHA7)
    assert g["status"] == A.GATE_FAIL and "outcomes producer:" in g["detail"] and OTHER_TREE in g["detail"]
    # a null outcomes producer (a record from a producer that never wrote one) refuses the same way
    G.build_overlap_stock_store(tmp_path / "nullpin", QWEN, correct=set(range(178)), header_edit=lambda h: h.update(producer_tree_sha=None))
    b = _overlap_run(retired, tmp_path / "nullpin" / "overlap", QWEN)["models"][QWEN]
    assert not b["usable"] and b["reason"].startswith("outcomes producer:") and "None" in b["reason"]
    # the pin's own store: usable, both trees recorded and equal
    G.build_overlap_stock_store(tmp_path / "pin", QWEN, correct=set(range(178)))
    b = _overlap_run(retired, tmp_path / "pin" / "overlap", QWEN)["models"][QWEN]
    assert b["usable"] and b["stock"]["producer_tree_sha"] == b["stock"]["outcomes_producer_tree_sha"] == G.FULL_SHA
    # the full synthetic wave's record carries the outcomes producer on every model
    assert all(m["stock"]["outcomes_producer_tree_sha"] == G.FULL_SHA for m in full_store.overlap["models"].values())


def test_a_declared_certified_refusal_needs_the_on_render_steering_proof(tmp_path):
    """Review round 2 (MEDIUM): the summary-less certified-refusal admission ran only the generic bites checks on a
    DECLARED row, so an OFF-rendered probe record or one without the declared steered stamps was admitted as a
    behavioural outcome. A declared refused store with the VALID ON-render proof (the declared probe body, the
    declared triple on every steered reply) is admitted and A0 PASSES; the same store with the probe rendered OFF, or
    with the steered stamps absent, or with a steered stamp not the declared triple, is NOT admitted — the leg stays
    missing naming the failed control and A0 FAILS. The undeclared row's admission (row 27's shape) is unchanged."""
    rows = small_matrix((QWEN, "main", NONE), DECL)

    def wave(name, bites_edit=None):
        G.build_all(tmp_path / name, rows=rows)
        _make_unservable(tmp_path / name, row(*DECL), evidence=_certified_refusal_evidence("qwen3_5"), bites_edit=bites_edit)
        rep = analyze(tmp_path / name, matrix=rows, overlap=None)
        return rep, leg_of(rep, *DECL)

    rep, leg = wave("ok")
    assert leg["certified_unservable"] is True and leg["missing"] is None and leg["render_certification"]["admitted"] is True
    assert rep["gates"]["A0_completeness"]["status"] == A.GATE_PASS and rep["gates"]["A0_completeness"]["legs"][leg["leg"]]["status"] == "refused_certified"
    assert leg["standup_record"]["problem"] is None and leg[V.RECORD_GAPS_KEY]           # the declared identity path ran, the record resolved
    for name, edit, word in (
            ("off", lambda b: b.update({C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM: C.THINKING_OFF, "probe_extra_body": C.chat_template_kwargs(C.THINKING_OFF)}),
             C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM),
            ("nostamps", lambda b: b.update(stamps_by_leg={"steered": [], "scale0": []}), "stamps_by_leg.steered"),
            ("stamp", lambda b: b["stamps_by_leg"]["steered"][1].update(resumed_policy="always"), "stamp triple"),
            ("body", lambda b: b.update(probe_extra_body=C.chat_template_kwargs(C.THINKING_ON)), "probe_extra_body")):
        rep, leg = wave(name, bites_edit=edit)
        assert leg["certified_unservable"] is False and leg["missing"] and "steering-bites control" in leg["missing"] and word in leg["missing"], (name, leg["missing"])
        assert leg["render_certification"]["admitted"] is False and rep["gates"]["A0_completeness"]["status"] == A.GATE_FAIL
        assert V.leg_key(*DECL, 8192) in rep["unusable_legs"]
    # the UNDECLARED refused row (row 27's shape): admitted exactly as before, with its OFF-rendered bites record
    _, rep = _refused_wave(tmp_path / "undeclared")
    leg = leg_of(rep, NEMO, "main", CONST_PSR)
    assert leg["certified_unservable"] is True and rep["gates"]["A0_completeness"]["status"] == A.GATE_PASS
    assert json.loads((Path(leg["cell_dir"]) / "evidence.json").read_text())[C.OOS_EVIDENCE_BITES_BLOCK][C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM] == C.THINKING_OFF


def test_a10_requires_the_outcomes_producer_on_every_stock_block(full_store):
    """Review round 3 (HIGH): A10 never read the stock blocks' `outcomes_producer_tree_sha`, so a same-pin record from a
    checker that never asked which tree wrote the values (usable: true, complete, the pinned manifest's tree advertised)
    passed. Now every stock block must carry a 40-hex outcomes producer EQUAL to its manifest producer: the real field
    passes; changed to another tree, nulled, or removed — each with `usable: true` and a complete comparison untouched —
    refuses by name, the missing/null case instructing regeneration."""
    other_full = "abcdef0123456789abcdef0123456789abcdef01"
    good = A.gate_overlap(full_store.overlap, list(G.MODEL_KEYS), G.IDS, store_pin=G.SHA7)
    assert good["status"] == A.GATE_PASS and all(m["outcomes_producer_tree_sha"] == m["producer_tree_sha"] == G.FULL_SHA for m in good["models"].values())
    for name, mutate, word in (
            ("changed", lambda s: s.update(outcomes_producer_tree_sha=other_full), f"outcomes producer tree {other_full} != its manifest producer tree {G.FULL_SHA}"),
            ("null", lambda s: s.update(outcomes_producer_tree_sha=None), "no 40-hex outcomes_producer_tree_sha (None)"),
            ("absent", lambda s: s.pop("outcomes_producer_tree_sha"), "no 40-hex outcomes_producer_tree_sha (None)"),
            ("short", lambda s: s.update(outcomes_producer_tree_sha=G.SHA7), f"no 40-hex outcomes_producer_tree_sha ({G.SHA7!r})")):
        mutated = json.loads(json.dumps(full_store.overlap))
        mutate(mutated["models"][QWEN]["stock"])
        assert mutated["models"][QWEN]["usable"] is True and mutated["models"][QWEN]["comparison"]["complete"] is True
        g = A.gate_overlap(mutated, list(G.MODEL_KEYS), G.IDS, store_pin=G.SHA7)
        assert g["status"] == A.GATE_FAIL and f"{QWEN}: the overlap record's " in g["detail"] and word in g["detail"], (name, g["detail"])
        assert GEMMA + ":" not in g["detail"] and "A10" not in g["detail"]
        if name in ("null", "absent"):
            assert "regenerate the overlap record" in g["detail"]
    # the consumer end to end: the mutated record fed to analyze() fails A10 and the report is not a production result
    mutated = json.loads(json.dumps(full_store.overlap))
    mutated["models"][GEMMA]["stock"]["outcomes_producer_tree_sha"] = None
    rep = analyze(full_store.root, overlap=mutated)
    assert rep["gates"]["A10_overlap"]["status"] == A.GATE_FAIL and "A10_overlap" in rep["gates"]["failed"] and A.production_refusals(rep)


# --------------------------------------------------------------------------- #
# llama-paper-rows — the NO-SWITCH arm `not_available` through the REAL reader  #
# a Llama store written by the real no-switch producers passes every gate;      #
# each refusal site flips when ONE fact is wrong; the trigger provenance is     #
# pinned; the figure exclusion and the four-family assembly                     #
# --------------------------------------------------------------------------- #

LLAMA = G.LLAMA_KEY
NA = C.THINKING_NOT_AVAILABLE
LLAMA_ROWS = G.matrix_rows_llama()
LLAMA_DOC = G.matrix_doc_llama(LLAMA_ROWS)
L_NONE, L_NONE_IT = (LLAMA, "main", NONE), (LLAMA, "it", NONE)
L_FITTED, L_FITTED_IT = (LLAMA, "main", CONST_LORA), (LLAMA, "it", CONST_LORA)         # the six-arm design: no PSR cell on the Llama rows
L_CONST = (LLAMA, "main", G.CELL_PROMPTED_CONST)
L_PROMPTED = (LLAMA, "it", PROMPTED)
#: the Llama row-provenance fixture: the `main` view's stores are built with `--bank-view main` (the four fields on every row), the
#: `it` view's without (a run before / without the flag — legal, the fields absent)
LLAMA_BANK_VIEW_BY_VIEW = {"main": "main", "it": None}
#: the published wave's synthetic pin when it is a PEER of the Llama run (another wave, another pin)
PEER_PIN = "1234567"
REAL_LLAMA_MATRIX = REPO / "configs" / "sdfbase" / "gsm8k" / "matrix_llama.json"


def _as_first_publication(report):
    """A report AS THE PUBLISHED WAVE'S FIRST ANALYSIS FILED IT (2026-09-10 to 2026-09-12): its ladder ran the retired 3 % exception,
    so its truncation block, design and cells say so. The fixture's stores are classified at the instrument's 1 % (the published
    wave was re-classified under it); only the filed report's trigger FACTS are rewritten here — the mixed-trigger assembly path
    (a column that ran another trigger) needs exactly such a peer."""
    r = json.loads(json.dumps(report, default=str))
    t3 = IR.GSM8K_TRIGGER_THRESHOLD_3PCT_RETIRED
    tr = r["truncation"]
    tr.update(trigger_threshold=t3, trigger_thresholds_seen=[t3], rule=IR.GSM8K_TRIGGER_RULE_3PCT_RETIRED, trigger_provenance=A.TRIGGER_PROVENANCE_3PCT,
              table_header=f"at-cap share per cell at the {A.CAP_BASE:,} cap — {A.TRIGGER_PROVENANCE_3PCT}")
    for c in tr["cells"]:
        c["trigger_threshold"] = t3
    r["design"]["trigger_threshold"] = t3
    r["design"]["trigger_thresholds_by_model"] = {m: [t3] for m in r["design"]["trigger_thresholds_by_model"]}
    return r


def lrow(key, view, cell):
    return G.row_for(LLAMA_ROWS, key, view, cell)


def llama_rows_of(*triples):
    return [lrow(*t) for t in triples]


def analyze_llama(gen_root, *, matrix=None, **kw):
    kw.setdefault("render_facts", G.RENDER_FACTS_ALL)
    kw.setdefault("pins", G.pins_doc(LLAMA_ROWS))
    return analyze(gen_root, matrix=(matrix if matrix is not None else LLAMA_DOC), **kw)


def build_overlap_not_applicable(root, models=(LLAMA,)):
    """The overlap check's record for a no-switch model THROUGH the real checker (P12): no store is read, the block is the
    explicit `not_applicable` record."""
    return O.run(retired_root=root / "retired", stock_root=root / "overlap", models=list(models), canonical_ids=G.IDS, ids_path=None,
                 pairing_rows=G.pairing_rows(), pairing_source="synthetic rows", repo_root=REPO, store_pin=G.SHA7, n_resamples=B_FAST, seed=0,
                 ci=0.95, eval_logs=False)


def build_llama(root, rows, **over_common):
    """Every launch row of `rows` as a Llama store (the `main` view with the row provenance, the `it` view without), the .eval logs
    in the inspect venv; `over_common` extends every leg's kwargs."""
    overrides = {(r["model_key"], r["view"], r["cell"]): {"bank_view": LLAMA_BANK_VIEW_BY_VIEW[r["view"]], **over_common} for r in rows}
    dirs = G.build_all(root, rows=rows, overrides=overrides)
    if HAS_INSPECT:
        for cd in dirs.values():
            G.write_inspect_log(cd, name="2026-01-01T00-00-00")
    return dirs


@pytest.fixture(scope="module")
def llama_store(tmp_path_factory):
    root = tmp_path_factory.mktemp("g8k_llama")
    dirs = build_llama(root, LLAMA_ROWS)
    assert len(dirs) == len(LLAMA_ROWS)
    return SimpleNamespace(root=root, overlap=build_overlap_not_applicable(root))


@pytest.fixture(scope="module")
def llama_report(llama_store):
    return analyze_llama(llama_store.root, overlap=llama_store.overlap, read_eval_logs=HAS_INSPECT)


def test_the_no_switch_reader_constants_are_the_producers_own_by_import():
    assert V.THINKING_ARM_NOT_AVAILABLE == NA == "not_available" and V.THINKING_ARM_ON == C.THINKING_ON and V.THINKING_ARMS == (C.THINKING_ON, NA)
    assert V.SAMPLING_MODE_BY_ARM == {C.THINKING_ON: C.SAMPLING_THINKING_CARD, NA: C.SAMPLING_CARD} and V.SAMPLING_MODE_BY_ARM[NA] == "card"
    assert V.THINKING_LABEL_BY_ARM[NA] == "thinking: not available" and V.thinking_label(NA) == A.NA_LABEL
    with pytest.raises(A.AnalysisError, match="not one of"):
        V.thinking_label("off")
    # the no-switch stamp triples are the contract's derivations under the `unspecified` render the proxy stamps when no switch rides
    assert V.STAMPS_BASE_NOT_AVAILABLE == tuple(C.G8K_STAMPS_BASE_NOT_AVAILABLE) == ("cell-stable", "not-consulted", C.RENDER_REGIME_UNSPECIFIED)
    assert V.STAMPS_FITTED_NOT_AVAILABLE == tuple(C.G8K_STAMPS_FITTED_NOT_AVAILABLE) == ("cell-stable", "always", C.RENDER_REGIME_UNSPECIFIED)
    assert V.expected_stamps_for(fitted=True, declared=False, arm=NA) == V.STAMPS_FITTED_NOT_AVAILABLE
    assert V.expected_stamps_for(fitted=False, declared=False, arm=NA) == V.STAMPS_BASE_NOT_AVAILABLE
    assert V.expected_stamps_for(fitted=True, declared=False) == V.STAMPS_FITTED            # the default arm is byte-identical
    with pytest.raises(ValueError, match="declared"):
        V.expected_stamps_for(fitted=True, declared=True, arm=NA)
    # the overlap check's status literal is the gate's; the class literals are the contract's
    assert O.STATUS_NOT_APPLICABLE == A.OVERLAP_STATUS_NOT_APPLICABLE == "not_applicable" and O.SCHEMA == A.OVERLAP_SCHEMA == "gsm8k_overlap/1"
    assert C.FAMILY_BY_MODEL_KEY[LLAMA] in C.NO_THINKING_CHANNEL_FAMILIES and O.is_no_switch_model(LLAMA) and not O.is_no_switch_model(GEMMA)
    g8k_cells = _optional_g8k_cells()
    if g8k_cells is not None:
        assert (V.MATRIX_SCHEMA_1, V.MATRIX_SCHEMA_2) == (g8k_cells.MATRIX_SCHEMA, g8k_cells.MATRIX_SCHEMA_2)
        assert g8k_cells.THINKING_MODE_BY_FAMILY[C.FAMILY_BY_MODEL_KEY[LLAMA]] == NA and g8k_cells.SAMPLING_MODE_BY_MODEL[LLAMA] == C.SAMPLING_CARD
    # the synthetic Llama matrix is the real document's shape (when the file is in the tree): the same cells per view, the same arm /
    # mode / stamps / roots / class per (view, cell); the row count is DERIVED, never pinned
    if REAL_LLAMA_MATRIX.is_file():
        real = V.load_matrix(REAL_LLAMA_MATRIX)
        # E2: the appended steer wave's rows may be HELD by a `holds:` entry (a staged record the driver's mirror refuses) — none today
        # (E2-16); the published twelve are always launched, so launched == rows − held whatever the steer rows' status
        held = [r for r in real["rows"] if r["status"] == "held"]
        assert real["schema"] == V.MATRIX_SCHEMA_2 and real["wave_root_rel"] == G.LLAMA_WAVE_ROOT_REL and real["counts"]["held"] == len(held) == 0
        assert all(r.get("wave") == "steer" for r in held) and all(r["status"] == "launch" for r in real["rows"] if r.get("wave") is None)
        assert real["counts"]["launched"] == real["counts"]["rows"] - len(held) == len(real["rows"]) - len(held) and real["counts"]["declared"] == 0
        # the PUBLISHED wave (rows without a `wave` key) is the six-arm shape the synthetic matrix mirrors; E2's appended steering rows
        # (wave `steer`, main view, external objects) are the launcher's own and have their own reader tests
        published = [r for r in real["rows"] if r.get("wave") is None]
        assert len(published) == len(LLAMA_ROWS) and all(r.get("wave") == "steer" for r in real["rows"] if r.get("wave") is not None)
        by_key = {(r["view"], r["cell"]): r for r in published}
        synth = {(r["view"], r["cell"]): r for r in LLAMA_ROWS}
        assert set(by_key) == set(synth), (sorted(set(by_key) ^ set(synth)))
        for k, r in by_key.items():
            s_ = synth[k]
            for f in ("model_key", "tag", "fitted", "declared", "route_kind", "thinking_mode", "sampling_mode", "expected_stamps", "out_root",
                      "rung_out_root", "family", "tuple_name", "served_positions_expected", "staged_id"):
                assert r[f] == s_[f] if f != "staged_id" else (r[f] is None) == (s_[f] is None), (k, f, r[f], s_[f])
            assert r["tuple_names"] in (None, []) and r["thinking_mode"] == NA and r["sampling_mode"] == C.SAMPLING_CARD


def test_a_llama_store_by_the_real_producers_passes_the_reader_end_to_end(llama_report, llama_store):
    rep = llama_report
    n = len(LLAMA_ROWS)
    assert rep["unusable_legs"] == {} and rep["gates"]["failed"] == [] and rep["gates"]["pending"] == [], rep["gates"]["failed"]
    assert A.production_refusals(rep) == [] and rep["verdict"]["word"] == "filed"
    assert rep["design"]["n_launch_rows"] == n and rep["design"]["n_contrasts_designed"] == rep["design"]["n_contrasts_computable"] == n - 2
    # the arm per model, its label, the hold-out, the schema-2 document's facts
    assert rep["design"]["thinking_mode_by_model"] == {LLAMA: NA} and rep["design"]["no_switch_models"] == [LLAMA]
    assert rep["design"]["thinking_label_by_model"] == {LLAMA: "thinking: not available"}
    assert rep["design"]["held_out_of_pooled_means"] == {LLAMA: A.HOLD_OUT_REASON} and "pooled cross-family mean" in A.HOLD_OUT_REASON
    assert rep["design"]["matrix_schema"] == V.MATRIX_SCHEMA_2 and rep["design"]["wave_root_rel"] == G.LLAMA_WAVE_ROOT_REL
    # every Llama leg is labelled "thinking: not available", on the leg, the controls table and the provenance rows
    for leg in rep["legs"].values():
        assert leg["thinking_mode"] == NA and leg["thinking_label"] == "thinking: not available" and leg["row_thinking"] == NA
        assert leg["sampling_mode"] == C.SAMPLING_CARD and leg["render_fact"]["split_mode"] == C.THINK_SPLIT_NONE and leg["render_fact"]["thinking_switch"] is False
        assert leg["summary"]["thinking"] == NA and leg["summary"]["think_split_mode"] == C.THINK_SPLIT_NONE and leg["summary"]["reasoning_parser"] is None
        assert leg["ledger"]["thinking"] == [None] and leg["request_evidence"]["cert_class"] == C.CERT_CLASS_NO_SWITCH
        assert leg["certification"]["cert_class"] == C.CERT_CLASS_NO_SWITCH and leg["certification"]["arm"] == NA
        assert leg["thinking"]["by_construction"] is True and leg["thinking"]["n_stray_tag_draws"] == 0
        assert leg["identity"][C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING] is None and leg["identity"][C.MANIFEST_KEY_AUTHORISED_EXPERIMENT] is None
    for row_ in rep["controls"]:
        assert row_["thinking_mode"] == NA and row_["thinking_label"] == "thinking: not available"
    # the row provenance: the four fields when the run emitted them (the `main` view, `--bank-view main`), None when it did not
    for lk, prov in rep["provenance"]["cells"].items():
        leg = rep["legs"][lk]
        rp = prov["row_provenance"]
        assert prov["thinking_mode"] == NA and set(rp) == {"present", *C.ROW_PROVENANCE_FIELDS}
        if leg["view"] == "main":
            r = lrow(LLAMA, "main", leg["cell"])
            assert rp["present"] is True and rp["view"] == "main" and rp["route_kind"] == r["route_kind"] == C.route_kind_of(leg["cell"])
            assert rp["object_artifact_sha256"] == (r["object_sha256"] if r["fitted"] else None) and rp["producer_tree_sha"] == G.FULL_SHA
        else:
            assert rp["present"] is False and all(rp[f] is None for f in C.ROW_PROVENANCE_FIELDS)
    # A10: the checker's not_applicable block, admitted for the no-switch family alone
    a10 = rep["gates"]["A10_overlap"]
    assert a10["status"] == A.GATE_PASS and a10["not_applicable"] == [LLAMA] and a10["models"][LLAMA]["status"] == "not_applicable"
    assert a10["models"][LLAMA]["pin"] == G.SHA7 and a10["models"][LLAMA]["family"] == "llama"
    block = llama_store.overlap["models"][LLAMA]
    assert block["schema"] == "gsm8k_overlap/1" and block["status"] == "not_applicable" and block["model_key"] == LLAMA and block["family"] == "llama"
    assert block["pin"] == G.SHA7 and isinstance(block["why"], str) and "no thinking-OFF arm" in block["why"] and block["comparison"] is None
    assert any("overlap llama8b" in f for f in rep["findings"])
    # the stamp triples the legs wore are the `unspecified` render's
    for lk, leg in rep["legs"].items():
        want = V.STAMPS_FITTED_NOT_AVAILABLE if leg["fitted"] else V.STAMPS_BASE_NOT_AVAILABLE
        assert tuple(leg["stamps"]["observed"].values()) == want, lk
    # the fixture's records ARE the producers' own: the no-switch item shape, the ledger's null, the wire's absent switch, the
    # producer's reconciliation rule under the no-switch arm, the (C″) record verbatim
    root = llama_store.root
    r = lrow(*L_FITTED)
    cd = next((root / r["out_root"] / "cells").glob("*"))
    outcomes = completion.read_task_outcomes(cd, V.TASK)
    assert all(frozenset(it) == frozenset(C.TASK_OUTCOME_ITEM_FIELDS_NO_SWITCH) for it in outcomes["items"]) and "stray_close_tag" in outcomes["items"][0]
    ledger_rows = [json.loads(line) for line in (root / r["out_root"] / C.LEDGER_FILENAME).read_text().splitlines()]
    recs = [json.loads(line) for line in (root / r["out_root"] / C.REQUEST_EVIDENCE_FILENAME).read_text().splitlines()]
    assert ledger_rows and all(x["thinking"] is None for x in ledger_rows)
    assert recs and all(x["enable_thinking_present"] is False and x["enable_thinking_value"] is None and x["petri_thinking_on_present"] is False for x in recs)
    assert C.request_evidence_no_switch_absent(recs) is None
    # the shared gate's spelling of the NO-SWITCH class since the IFEval battery landed: `expected_enable_thinking=None`
    assert C.request_evidence_reconciliation(recs, ledger_rows, filed=[(V.TASK, i) for i in G.IDS], expected_enable_thinking=None) is None
    evidence = json.loads((cd / "evidence.json").read_text())
    assert evidence[C.LENS_ON_BITE_KEY] == C.no_switch_lens_on_bite_record() and evidence["render_certified"]["cert_class"] == C.CERT_CLASS_NO_SWITCH
    assert evidence["steering_bites"][C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM] == NA and evidence["steering_bites"]["probe_extra_body"] == {}
    manifest = completion.read_store_manifest(cd)
    assert manifest[C.MANIFEST_KEY_THINKING]["enable_thinking"] is None and manifest[C.MANIFEST_KEY_THINKING]["render"] == C.THINKING_RENDER_NATIVE
    ident = manifest[C.MANIFEST_KEY_TINYCAP]
    assert ident["thinking"]["mode"] == NA and ident[C.IDENTITY_KEY_THINKING_BY_TASK] == {V.TASK: NA}
    for absent in (C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING, C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT, C.IDENTITY_KEY_THINKING_HEADER_IDS):
        assert absent not in ident, absent
    store_rows = [json.loads(line) for line in (cd / "rows.jsonl").read_text().splitlines()]
    extras = [json.loads(x["extra_json"]) for x in store_rows if x["item_id"] != C.AGG_ITEM_ID and x["metric"] == C.METRIC_ACC]
    assert extras and all(e[C.ROW_KEY_THINKING] == NA and set(C.ROW_PROVENANCE_FIELDS) <= set(e) and set(C.ROW_LIFT_FIELDS) <= set(e) for e in extras)
    # the certified-refusal / expected-refusal machinery is untouched: nothing refused, nothing held
    assert rep["refused"] == [] and rep["held"] == []


def _llama_pair(tmp_path, name, **over):
    """A two-row Llama design (none + the constitution PSR object, `main`) with `over` on the fitted leg; the reader over it."""
    rows = llama_rows_of(L_NONE, L_FITTED)
    root = tmp_path / name
    G.build_all(root, rows=rows, overrides={L_NONE: {"bank_view": "main"}, L_FITTED: {"bank_view": "main", **over}})
    return analyze_llama(root, matrix=rows, overlap=build_overlap_not_applicable(root))


def test_each_no_switch_refusal_site_flips_when_one_fact_is_wrong(tmp_path):
    leg_key = V.leg_key(*L_FITTED, 8192)

    def refused(rep):
        return leg_of(rep, *L_FITTED)["refused"] or ""

    # A1: a ledger row saying a switch was sent (`thinking: on` — or `false`, which would record a switch that was never there)
    def ledger_on(lrows):
        for x in lrows:
            if x["task"] == V.TASK:
                x["thinking"] = "on"
    rep = _llama_pair(tmp_path, "a1", ledger_edit=ledger_on)
    assert rep["gates"]["A1_ledger"]["status"] == A.GATE_FAIL and "thinking: None" in rep["gates"]["A1_ledger"]["detail"]

    def ledger_false(lrows):
        for x in lrows:
            if x["task"] == V.TASK:
                x["thinking"] = False
    rep = _llama_pair(tmp_path, "a1b", ledger_edit=ledger_false)
    assert rep["gates"]["A1_ledger"]["status"] == A.GATE_FAIL and "P17" in rep["gates"]["A1_ledger"]["detail"]

    # A2: a request-evidence record whose body CARRIED enable_thinking
    def switched(recs):
        for x in recs:
            if x["kind"] == C.REQUEST_EVIDENCE_KIND_TASK:
                x["enable_thinking_present"], x["enable_thinking_value"] = True, False
                break
    rep = _llama_pair(tmp_path, "a2", request_evidence_edit=switched)
    g = rep["gates"]["A2_identity"]
    assert g["status"] == A.GATE_FAIL and "CARRIED" in g["detail"] and "no-switch rule refuses" in g["detail"]

    # A3: the EXPLICIT lens_on_bite inapplicability record is REQUIRED — an absent block fails, an ON-bite block fails
    def drop_bite(ev):
        ev.pop(C.LENS_ON_BITE_KEY)
    rep = _llama_pair(tmp_path, "a3", evidence_edit=drop_bite)
    assert rep["gates"]["A3_certification"]["status"] == A.GATE_FAIL and "REQUIRED" in rep["gates"]["A3_certification"]["detail"]

    def on_bite(ev):
        ev[C.LENS_ON_BITE_KEY] = {"applicable": True, "verified": True, "reasoning_present": True}
    rep = _llama_pair(tmp_path, "a3b", evidence_edit=on_bite)
    assert rep["gates"]["A3_certification"]["status"] == A.GATE_FAIL and "ON-bite block" in rep["gates"]["A3_certification"]["detail"]

    # A3: the bites control under a switch (an OFF-rendered probe), the render probe under an arm, a switch-class render record
    def off_bites(ev):
        ev["steering_bites"][C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM] = C.THINKING_OFF
    rep = _llama_pair(tmp_path, "a3c", evidence_edit=off_bites)
    assert rep["gates"]["A3_certification"]["status"] == A.GATE_FAIL and "probe_render_arm" in rep["gates"]["A3_certification"]["detail"]

    def on_probe(ev):
        ev["render_probe"][C.THINKING_ON] = ev["render_probe"].pop(NA)
    rep = _llama_pair(tmp_path, "a3d", evidence_edit=on_probe)
    assert rep["gates"]["A3_certification"]["status"] == A.GATE_FAIL and "ONE probe" in rep["gates"]["A3_certification"]["detail"]

    def switch_class(ev):
        ev["render_certified"] = G.S.render_certified_record("llama", G.RENDER_FACTS_ALL["llama"])
    rep = _llama_pair(tmp_path, "a3e", evidence_edit=switch_class)
    assert rep["gates"]["A3_certification"]["status"] == A.GATE_FAIL and "cert_class" in rep["gates"]["A3_certification"]["detail"]

    # A2 (per leg): a stamp of the thinking-ON render on a no-switch leg
    rep = _llama_pair(tmp_path, "a2b", summary_edit=lambda s: s.update(render_regime=C.RENDER_REGIME_THINKING_ON))
    assert rep["gates"]["A2_identity"]["status"] == A.GATE_FAIL and "stamps" in rep["gates"]["A2_identity"]["detail"]

    # identity (at collection -> A0): a manifest recording a switch, a present header-ids key, an out-of-support key, an OFF summary,
    # a transfer provenance served under the thinking-ON render
    rep = _llama_pair(tmp_path, "id1", manifest_edit=lambda m: m[C.MANIFEST_KEY_THINKING].update(enable_thinking=False))
    assert "records null" in refused(rep) and rep["gates"]["A0_completeness"]["status"] == A.GATE_FAIL
    rep = _llama_pair(tmp_path, "id2", manifest_edit=lambda m: m[C.MANIFEST_KEY_TINYCAP].update({C.IDENTITY_KEY_THINKING_HEADER_IDS: {"off": [1], "on": [1]}}))
    assert C.IDENTITY_KEY_THINKING_HEADER_IDS in refused(rep) and "present iff" in refused(rep)
    rep = _llama_pair(tmp_path, "id3", manifest_edit=lambda m: m[C.MANIFEST_KEY_TINYCAP].update({C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING: True}))
    assert C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING in refused(rep) and "in-support" in refused(rep)
    rep = _llama_pair(tmp_path, "id4", manifest_edit=lambda m: m[C.MANIFEST_KEY_TINYCAP]["thinking"].update(mode=C.THINKING_OFF))
    assert "identity thinking.mode" in refused(rep)
    rep = _llama_pair(tmp_path, "id5", manifest_edit=lambda m: m[C.MANIFEST_KEY_TRANSFER_PROVENANCE].update(served_render=C.RENDER_REGIME_THINKING_ON))
    assert "served_render" in refused(rep) and C.RENDER_REGIME_UNSPECIFIED in refused(rep)
    rep = _llama_pair(tmp_path, "id6", manifest_edit=lambda m: m[C.MANIFEST_KEY_RENDER_FACT_EVIDENCE].update(thinking_switch=True))
    assert "thinking_switch" in refused(rep)
    rep = _llama_pair(tmp_path, "id7", summary_edit=lambda s: s.update(thinking=C.THINKING_OFF), header_edit=lambda h: h.update(thinking=C.THINKING_OFF))
    assert "thinking" in refused(rep)

    # the per-row arm and the row provenance (I2): a row wearing another arm; a provenance field disagreeing with the matrix row
    def row_off(rows_out):
        extra = json.loads(rows_out[0]["extra_json"])
        extra[C.ROW_KEY_THINKING] = C.THINKING_OFF
        rows_out[0]["extra_json"] = json.dumps(extra)
    rep = _llama_pair(tmp_path, "row1", rows_edit=row_off)
    assert f"per-row {C.ROW_KEY_THINKING}" in refused(rep) and "off" in refused(rep)

    def row_view(rows_out):
        for x in rows_out:
            extra = json.loads(x["extra_json"])
            extra["view"] = "it"
            x["extra_json"] = json.dumps(extra)
    rep = _llama_pair(tmp_path, "row2", rows_edit=row_view)
    assert "row provenance disagrees" in refused(rep) and "view" in refused(rep)

    def row_partial(rows_out):
        extra = json.loads(rows_out[0]["extra_json"])
        for f in C.ROW_PROVENANCE_FIELDS:
            extra.pop(f)
        rows_out[0]["extra_json"] = json.dumps(extra)
    rep = _llama_pair(tmp_path, "row3", rows_edit=row_partial)
    assert "present together" in refused(rep)

    # a switch-arm store under a no-switch row (the mislabel): the identity refuses before any gate
    rows = llama_rows_of(L_NONE, L_FITTED)
    root = tmp_path / "mislabel"
    G.build_store(root, lrow(*L_NONE), correct=G.CORRECT_NONE, bank_view="main")
    G.build_store(root, lrow(*L_FITTED), correct=G.CORRECT_BY_CELL[CONST_PSR], arm=C.THINKING_ON, sampling_mode=G.MODE_CARD)
    rep = analyze_llama(root, matrix=rows, overlap=build_overlap_not_applicable(root))
    assert leg_key in rep["unusable_legs"] and "thinking" in rep["unusable_legs"][leg_key] and NA in rep["unusable_legs"][leg_key]


def test_the_none_splits_stray_tag_rule_fails_the_primary_leg_closed_and_the_sensitivity_masks(tmp_path):
    """P19 / P21: a literal `</think>` in a completion (the `none` split's stray_close_tag), a literal `<think>`, and the provider's
    `client_lift_fired` record are STRAY-TAG anomalies on a no-thinking-channel leg: A4 fails the leg closed, each draw named; the
    sensitivity reading masks them and reports the count. The thinking record files the same count by source."""
    rep = _llama_pair(tmp_path, "stray", stray_close={5}, stray={9}, lifted={7})
    g = rep["gates"]["A4_scoring_path"]
    key = V.leg_key(*L_FITTED, 8192)
    assert g["status"] == A.GATE_FAIL and "PRIMARY fails closed" in g["detail"] and "stray_close_tag" in g["detail"] and "client_lift_fired" in g["detail"]
    assert "stray_open_tag" in g["detail"]
    rec = g["legs"][key]["stray_tag"]
    assert rec["n_stray"] == 3 and sorted(rec["stray_ids"]) == sorted(G.IDS[i] for i in (5, 7, 9)) and rec["primary"]["fail_closed"] is True
    assert rec["sensitivity"]["n_masked"] == 3 and rec["sensitivity"]["n"] == 197 and rec["sensitivity"]["n_scored"] == 197
    assert rec["sensitivity"]["acc_stock_masked"] != rec["sensitivity"]["acc_stock"] and rec["rule"] == M.NO_REASONING_RULE[C.THINK_SPLIT_NONE]
    leg = leg_of(rep, *L_FITTED)
    assert leg["thinking"]["n_stray_tag_draws"] == 3 and leg["thinking"]["stray_tag_draws_by_source"] == {M.PRESENT_BY_RECORD: 2, M.PRESENT_BY_CHANNEL: 1}
    assert leg["refused"] is None                       # the leg is READ (the gate, not the collection, refuses it)
    # the stray flags were DERIVED from the content by the real `none` split, never typed: the record and the rows agree
    items = leg["items"]
    assert items[G.IDS[5]]["stray_close_tag"] is True and items[G.IDS[9]]["stray_open_tag"] is True and items[G.IDS[7]]["stray_close_tag"] is False
    assert all(not it["think_block_present"] and not it["think_block_unterminated"] for it in items.values())
    assert leg["rows_lifted"][G.IDS[7]]["client_lift_fired"] is True and leg["rows_lifted"][G.IDS[5]]["client_lift_fired"] is False
    # the production boundary refuses the report; the clean pair is a production result
    assert A.production_refusals(rep) and "A4_scoring_path" in A.production_refusals(rep)[0]
    clean = _llama_pair(tmp_path, "clean")
    assert clean["gates"]["A4_scoring_path"]["status"] == A.GATE_PASS and clean["gates"]["A4_scoring_path"]["legs"][key]["stray_tag"]["n_stray"] == 0
    assert A.production_refusals(clean) == []


def test_a10_admits_not_applicable_for_the_no_switch_family_only(llama_store, full_store):
    ov = llama_store.overlap
    # the checker never writes a not_applicable block for a switch family, and never a real block for a no-switch one
    with pytest.raises(A.AnalysisError, match="thinking switch"):
        O.not_applicable_block(GEMMA, store_pin=G.SHA7)
    # a not_applicable block on gemma FAILS A10 by name; on llama8b it passes; a REAL block on llama8b fails
    gemma_na = {**ov, "models": {GEMMA: {**ov["models"][LLAMA], "model_key": GEMMA, "family": "gemma4"}}}
    g = A.gate_overlap(gemma_na, [GEMMA], G.IDS, store_pin=G.SHA7)
    assert g["status"] == A.GATE_FAIL and "HAS a thinking switch" in g["detail"] and g["not_applicable"] == []
    g = A.gate_overlap(ov, [LLAMA], G.IDS, store_pin=G.SHA7)
    assert g["status"] == A.GATE_PASS and g["not_applicable"] == [LLAMA] and g["models"][LLAMA]["status"] == "not_applicable"
    real_on_llama = {**ov, "models": {LLAMA: {**full_store.overlap["models"][GEMMA], "model_key": LLAMA}}}
    g = A.gate_overlap(real_on_llama, [LLAMA], G.IDS, store_pin=G.SHA7)
    assert g["status"] == A.GATE_FAIL and "REAL overlap block on the no-switch family" in g["detail"]
    # the block's own facts are held: another pin, a missing why, another family
    for edit, why in ((lambda b: b.update(pin="abcdef0"), "store pin"), (lambda b: b.update(why=""), "no `why`"), (lambda b: b.update(family="gemma4"), "family")):
        bad = {**ov, "models": {LLAMA: dict(ov["models"][LLAMA])}}
        edit(bad["models"][LLAMA])
        g = A.gate_overlap(bad, [LLAMA], G.IDS, store_pin=G.SHA7)
        assert g["status"] == A.GATE_FAIL and why in g["detail"], why
    # a record with no block for the no-switch model names what it needs; the published record's four models are untouched
    g = A.gate_overlap({**ov, "models": {}}, [LLAMA], G.IDS, store_pin=G.SHA7)
    assert g["status"] == A.GATE_FAIL and "not_applicable block" in g["detail"]
    g = A.gate_overlap(full_store.overlap, list(G.MODEL_KEYS), G.IDS, store_pin=G.SHA7)
    assert g["status"] == A.GATE_PASS and g["not_applicable"] == [] and all(per["status"] == "checked" for per in g["models"].values())


def test_the_schema_aware_matrix_loader_refuses_the_no_switch_defects_by_name():
    # a schema-1 DOCUMENT carrying a schema-2 field: `thinking_mode` on a row, or `wave_root_rel`
    rows = [dict(r) for r in ROWS]
    rows[0]["thinking_mode"] = C.THINKING_ON
    with pytest.raises(A.AnalysisError, match="schema-2 field"):
        V.load_matrix({"schema": V.MATRIX_SCHEMA_1, "rows": rows})
    with pytest.raises(A.AnalysisError, match="wave_root_rel"):
        V.load_matrix({"schema": V.MATRIX_SCHEMA_1, "wave_root_rel": "x", "rows": [dict(r) for r in ROWS]})
    with pytest.raises(A.AnalysisError, match="not one of"):
        V.load_matrix({"schema": "g8k_matrix/9", "rows": [dict(r) for r in ROWS]})
    # a document naming no schema reads as schema 1 (the published matrix's rows load as today); a bare list is schema-less
    assert V.load_matrix({"rows": [dict(r) for r in ROWS]})["counts"] == V.load_matrix(ROWS)["counts"]
    assert V.load_matrix(ROWS)["schema"] is None and V.load_matrix({"rows": [dict(r) for r in ROWS]})["schema"] == V.MATRIX_SCHEMA_1
    assert all(r["thinking_mode"] == C.THINKING_ON for r in V.load_matrix(ROWS)["rows"])           # the schema-1 row's arm made explicit
    # a schema-2 document: every row carries the arm; the wave root is named
    mx = V.load_matrix(LLAMA_DOC)
    assert mx["schema"] == V.MATRIX_SCHEMA_2 and mx["wave_root_rel"] == G.LLAMA_WAVE_ROOT_REL and all(r["thinking_mode"] == NA for r in mx["rows"])
    assert mx["counts"]["launched"] == len(LLAMA_ROWS) and mx["counts"]["declared"] == 0 and V.thinking_mode_by_model(mx["launch"]) == {LLAMA: NA}
    with pytest.raises(A.AnalysisError, match="wave_root_rel"):
        V.load_matrix({"schema": V.MATRIX_SCHEMA_2, "rows": [dict(r) for r in LLAMA_ROWS]})
    bare = [dict(r) for r in LLAMA_ROWS]
    bare[0].pop("thinking_mode")
    with pytest.raises(A.AnalysisError, match="every row"):
        V.load_matrix({"schema": V.MATRIX_SCHEMA_2, "wave_root_rel": G.LLAMA_WAVE_ROOT_REL, "rows": bare})
    # the row rules per arm
    for edit, why in (
            (lambda r: r.update(thinking_mode=C.THINKING_OFF), "not one of"),
            (lambda r: r.update(sampling_mode=C.SAMPLING_THINKING_CARD), "sampling_mode"),
            (lambda r: r.update(declared=True, served_positions_expected="response"), "cannot be declared"),
            (lambda r: r.update(tuple_name=f"gsm8k_paired200_{LLAMA}_8192"), "tuple_name"),
            (lambda r: r.update(tuple_names=[f"gsm8k_paired200_{LLAMA}_8192"]), "tuple_names"),
            (lambda r: r.update(expected_stamps=["cell-stable", "always", "thinking_on"]), "expected_stamps"),
            (lambda r: r.update(out_root=r["out_root"].replace("/card", "/thinking_card")), "out_root"),
            (lambda r: r.update(family="gemma4"), "only there"),
    ):
        rows = [dict(r) for r in LLAMA_ROWS]
        edit(next(r for r in rows if (r["model_key"], r["view"], r["cell"]) == L_FITTED))
        with pytest.raises(A.AnalysisError, match=why):
            V.load_matrix(rows)
    # the no-switch arm on a switch family's row refuses; `on` on the published rows is byte-identical
    rows = [dict(r) for r in ROWS]
    next(r for r in rows if (r["model_key"], r["view"], r["cell"]) == (QWEN, "main", NONE))["thinking_mode"] = NA
    with pytest.raises(A.AnalysisError, match="only there"):
        V.load_matrix(rows)
    # a model whose rows run under two arms refuses
    with pytest.raises(A.AnalysisError, match="two thinking arms"):
        V.thinking_mode_by_model([{"model_key": "m", "status": "launch", "thinking_mode": NA}, {"model_key": "m", "status": "launch"}])


def _edit_row_extra(rows, item_id, fn):
    for r in rows:
        if r["item_id"] == item_id:
            ex = json.loads(r["extra_json"]); fn(ex); r["extra_json"] = json.dumps(ex)


def test_no_switch_raw_row_evidence_is_read_independently_of_the_flags_and_a_torn_record_refuses(tmp_path):
    """P2 (review): the reader inspected only the provider's lift fields, so a scored row whose raw content carried `</think>`
    while the outcome flag read false passed A4. Now the raw row content is read on its own: a literal tag the flags missed
    is a named stray-tag anomaly (A4 fails closed, the sensitivity reading masks it); a row flag disagreeing with the outcome
    record's, or a raw_content_sha256 that is not the row text's, is a TORN record and refuses the leg by name."""
    key = V.leg_key(*L_FITTED, 8192)
    target = G.IDS[11]

    def literal_only(rows):            # the content carries the tag; every flag stays false; the digest is the text's
        def fn(ex):
            ex["content"] = (ex.get("content") or "") + "</think>"
            ex["raw_content_sha256"] = hashlib.sha256(ex["content"].encode("utf-8")).hexdigest()
        _edit_row_extra(rows, target, fn)
    rep = _llama_pair(tmp_path, "literal", rows_edit=literal_only)
    leg = leg_of(rep, *L_FITTED)
    assert leg["refused"] is None and leg["rows_raw_tags"] == {target: ["</think"]}
    g = rep["gates"]["A4_scoring_path"]
    rec = g["legs"][key]["stray_tag"]
    assert g["status"] == A.GATE_FAIL and "raw row content carries think-tag literal(s)" in g["detail"]
    assert rec["stray_ids"] == [target] and rec["n_stray"] == 1 and rec["sensitivity"]["n_masked"] == 1
    assert leg["items"][target]["stray_close_tag"] is False and leg["rows_lifted"][target]["client_lift_fired"] is False

    def flag_disagrees(rows):          # the row says stray, the outcome record says not: torn
        _edit_row_extra(rows, target, lambda ex: ex.__setitem__("stray_close_tag", True))
    rep = _llama_pair(tmp_path, "torn_flag", rows_edit=flag_disagrees)
    assert "disagrees with the outcome record" in (leg_of(rep, *L_FITTED)["refused"] or "") and "stray_close_tag=True" in leg_of(rep, *L_FITTED)["refused"]

    def digest_wrong(rows):
        _edit_row_extra(rows, target, lambda ex: ex.__setitem__("raw_content_sha256", "ab" * 32))
    rep = _llama_pair(tmp_path, "torn_sha", rows_edit=digest_wrong)
    assert "is not the sha256 of the row's text" in (leg_of(rep, *L_FITTED)["refused"] or "")

    def digest_missing(rows):
        _edit_row_extra(rows, target, lambda ex: ex.pop("raw_content_sha256"))
    rep = _llama_pair(tmp_path, "no_sha", rows_edit=digest_missing)
    assert "carries no raw_content_sha256" in (leg_of(rep, *L_FITTED)["refused"] or "")
    # the reproduced case: content with the tag, the row flag TRUE, the outcome flag false, no lift — refused (torn), never PASS
    def reproduced(rows):
        def fn(ex):
            ex["content"] = (ex.get("content") or "") + "</think>"
            ex["stray_close_tag"] = True
            ex["raw_content_sha256"] = hashlib.sha256(ex["content"].encode("utf-8")).hexdigest()
        _edit_row_extra(rows, target, fn)
    rep = _llama_pair(tmp_path, "reproduced", rows_edit=reproduced)
    assert "disagrees with the outcome record" in (leg_of(rep, *L_FITTED)["refused"] or "") and A.production_refusals(rep)
    # the clean pair still reads clean, and every row carries the digest of its own text
    clean = _llama_pair(tmp_path, "clean2")
    assert clean["gates"]["A4_scoring_path"]["status"] == A.GATE_PASS and leg_of(clean, *L_FITTED)["rows_raw_tags"] == {}


def test_the_engine_resolved_sampler_record_beside_the_store_is_read_per_leg_as_a_finding_never_a_gate(tmp_path):
    """The launcher's recorder writes `<out root>/resolved_sampler.json`
    after the driver; the analysis reads it per usable leg (`V.resolved_sampler_record`), files a report block and a finding line
    on anything but AGREES — and never moves a gate. A store from before the recorder is NOT RECORDED and says so; a record whose
    client reference or cap is not this row's is a binding problem, by name; a silent log is UNREADABLE, never agreement. The
    record is the REAL recorder's, run over a synthetic vLLM log in the recorder's own line shapes."""
    if str(REPO / "tests") not in sys.path:
        sys.path.insert(0, str(REPO / "tests"))
    import test_record_resolved_sampler as TR
    rows = llama_rows_of(L_NONE, L_FITTED)
    root = tmp_path / "rs"
    dirs = G.build_all(root, rows=rows, overrides={L_NONE: {"bank_view": "main"}, L_FITTED: {"bank_view": "main"}})
    overlap = build_overlap_not_applicable(root)
    rep0 = analyze_llama(root, matrix=rows, overlap=overlap)
    fitted_leg, none_leg = V.leg_key(*L_FITTED, 8192), V.leg_key(*L_NONE, 8192)

    def gates(rep):
        return {g: v.get("status") for g, v in rep["gates"].items() if isinstance(v, dict)}
    # NO leg carries the recorder's record (stores from before the recorder rode the launcher): the report carries NO block —
    # a report over such stores keeps the published shape byte for byte (the frozen-reproduction rule); the
    # block appears once a leg is recorded (below)
    assert A.RESOLVED_SAMPLER_KEY == "resolved_sampler" and V.RESOLVED_SAMPLER_FILENAME == "resolved_sampler.json"
    assert A.RESOLVED_SAMPLER_KEY not in rep0 and A.resolved_sampler_block(rep0["legs"]) is None
    assert not any("RESOLVED SAMPLER" in f for f in rep0["findings"])
    out_root = Path(dirs[f"{L_FITTED[0]}/{L_FITTED[1]}/{L_FITTED[2]}"]).parents[1]
    sampling = next(r for r in rows if (r["model_key"], r["view"], r["cell"]) == L_FITTED)["sampling"]
    assert {k: sampling[k] for k in ("temperature", "top_p")} == {"temperature": 0.6, "top_p": 0.9}

    def record(lines, *, expect=("temperature=0.6", "top_p=0.9"), client_cap="8192"):
        log = tmp_path / "vllm_server.log"
        log.write_text("\n".join(lines) + "\n")
        argv = ["--vllm-log", str(log), "--out", str(out_root / V.RESOLVED_SAMPLER_FILENAME), "--client-cap", client_cap]
        for c in ("48", "512", "2048"):
            argv += ["--probe-cap", c]
        for e in expect:
            argv += ["--expect", e]
        assert TR.rrs.main(argv) == 0                                            # recorded, never propagated
        return analyze_llama(root, matrix=rows, overlap=overlap)
    agree = ([TR._STARTED, TR._ATTEST_INFO] + [TR._sp(i, cap=48, **TR._GREEDY) for i in range(3)]
             + [TR._sp(100 + i, cap=8192, t=0.6, tp=0.9, tk=0, pp=0.0) for i in range(5)])
    rep = record(agree)
    rs = rep["legs"][fitted_leg]["resolved_sampler"]
    assert rs["present"] and rs["verdict"] == "AGREES" and rs["problem"] is None and rs["n_client_requests"] == 5 and rs["n_probe_requests"] == 3
    assert rs["reference"] == {"temperature": 0.6, "top_p": 0.9} and rs["client_caps"] == ["8192"] and rs["divergences"] == [] and rs["path"].endswith("/resolved_sampler.json")
    assert rep[A.RESOLVED_SAMPLER_KEY]["n_recorded"] == 1 and rep[A.RESOLVED_SAMPLER_KEY]["verdicts"] == {"AGREES": 1} and rep[A.RESOLVED_SAMPLER_KEY]["legs_not_agreeing"] == []
    assert not rep["legs"][none_leg]["resolved_sampler"]["present"] and not any("RESOLVED SAMPLER" in f for f in rep["findings"])
    assert gates(rep) == gates(rep0) and rep["verdict"] == rep0["verdict"]
    # DIVERGED (the client class resolved temperature 1.0 — vLLM's neutral default, the "never reached the engine" signature): a
    # finding line naming the leg, the record's why, the divergence — the gates and the verdict untouched
    div = [TR._STARTED, TR._ATTEST_INFO] + [TR._sp(100 + i, cap=8192, t=1.0, tp=0.9, tk=0, pp=0.0) for i in range(5)]
    rep = record(div)
    rs = rep["legs"][fitted_leg]["resolved_sampler"]
    assert rs["verdict"] == "DIVERGED" and rs["problem"] is None and rs["divergences"][0]["key"] == "temperature" and rs["divergences"][0]["equals_vllm_neutral_default"] is True
    assert rep[A.RESOLVED_SAMPLER_KEY]["legs_not_agreeing"] == [fitted_leg] and rep[A.RESOLVED_SAMPLER_KEY]["verdicts"] == {"DIVERGED": 1}
    assert [f for f in rep["findings"] if "RESOLVED SAMPLER" in f] == [f"{fitted_leg}: RESOLVED SAMPLER DIVERGED — {rs['why']}"]
    assert gates(rep) == gates(rep0) and rep["verdict"] == rep0["verdict"]
    # a record run against ANOTHER client reference attests nothing about this leg: a binding problem by name (the finding carries it)
    rep = record(agree, expect=("temperature=0.7", "top_p=0.9"))
    rs = rep["legs"][fitted_leg]["resolved_sampler"]
    assert rs["verdict"] == "DIVERGED" and "is not this row's sampling row" in rs["problem"] and rep[A.RESOLVED_SAMPLER_KEY]["n_problems"] == 1
    assert any(f.startswith(f"{fitted_leg}: RESOLVED SAMPLER DIVERGED — the recorder's client reference") for f in rep["findings"])
    # a client cap that is not this leg's: the rung's requests land unclassified (the recorder says so), and this reader names the cap
    rep = record(agree, client_cap="4096")
    rs = rep["legs"][fitted_leg]["resolved_sampler"]
    assert rs["verdict"] == "UNCLASSIFIED_REQUESTS" and "do not name this leg's cap 8192" in rs["problem"] and rs["n_unclassified_requests"] == 5
    assert gates(rep) == gates(rep0)
    # silence is never agreement: a log with neither requests nor the flag's attestation is UNREADABLE — a finding
    rep = record([TR._STARTED])
    assert rep["legs"][fitted_leg]["resolved_sampler"]["verdict"] == "UNREADABLE" and any(f"{fitted_leg}: RESOLVED SAMPLER UNREADABLE — " in f for f in rep["findings"])
    # a torn record is a problem, not a crash; the reader's vocabulary is the recorder's; the file name is the launcher's literal
    (out_root / V.RESOLVED_SAMPLER_FILENAME).write_text("{not json")
    rep = analyze_llama(root, matrix=rows, overlap=overlap)
    rs = rep["legs"][fitted_leg]["resolved_sampler"]
    assert rs["present"] and rs["verdict"] is None and "unreadable" in rs["problem"] and fitted_leg in rep[A.RESOLVED_SAMPLER_KEY]["legs_not_agreeing"]
    assert set(V.RESOLVED_SAMPLER_VERDICTS) == {"AGREES", "DIVERGED", "UNREADABLE", "UNCOMPARED", "UNCLASSIFIED_REQUESTS", "FLAG_SET_NO_REQUESTS_YET", "REFUSED"}
    assert V.RESOLVED_SAMPLER_KEYS == TR.rrs.SAMPLER_KEYS and V.RESOLVED_SAMPLER_AGREES == "AGREES"
    cells = _optional_g8k_cells()
    if cells is not None:
        assert cells.RESOLVED_SAMPLER_FILENAME == V.RESOLVED_SAMPLER_FILENAME and cells.SAMPLING_PARAM_KEYS == V.RESOLVED_SAMPLER_KEYS


def test_the_report_states_the_harmonised_trigger_and_its_provenance(full_report):
    rep = full_report
    tr = rep["truncation"]
    assert tr["trigger_threshold"] == tr["line_1pct"] == 0.01 and tr["rule"] == IR.GSM8K_TRIGGER_RULE
    assert tr["trigger_provenance"] == IR.GSM8K_TRIGGER_PROVENANCE
    assert "2026-09-12" in tr["trigger_provenance"] and "3 % exception of 2026-09-10" in tr["trigger_provenance"]
    assert set(tr["rules_admitted_on_rungs"]) == {V.RUNG_PLANNED_UNDER_CURRENT, V.RUNG_PLANNED_UNDER_RETIRED_3PCT}
    assert tr["rules_admitted_on_rungs"][V.RUNG_PLANNED_UNDER_RETIRED_3PCT] == {"rule": IR.GSM8K_TRIGGER_RULE_3PCT_RETIRED, "threshold": 0.03}
    assert tr["rules_admitted_on_rungs"][V.RUNG_PLANNED_UNDER_CURRENT] == {"rule": IR.GSM8K_TRIGGER_RULE, "threshold": 0.01}
    # the fixture's one rung was planned under the current rule; none under the retired one; un-escalated cells carry no label
    assert tr["rungs_planned_under"] == {V.RUNG_PLANNED_UNDER_CURRENT: [V.leg_key(*ESC, 8192)], V.RUNG_PLANNED_UNDER_RETIRED_3PCT: []}
    esc_cell = next(c for c in tr["cells"] if c["leg"] == V.leg_key(*ESC, 8192))
    assert esc_cell["rung_planned_under"] == V.RUNG_PLANNED_UNDER_CURRENT
    assert all(c["rung_planned_under"] is None for c in tr["cells"] if not c["escalated"])
    # the findings and the top items spell the trigger from the constant, never a re-typed "3 %"
    assert A.TRIGGER_PCT == "1 %" and A.TRIGGER_FINDING_MARK == "> the 1 % trigger"
    assert any(f.startswith(V.leg_key(*ESC, 8192)) and A.TRIGGER_FINDING_MARK in f for f in rep["findings"])
    assert not any("3 % trigger" in f for f in rep["findings"])
    assert any(A.TRIGGER_FINDING_MARK in t for t in A.top_items_of(rep))


def test_a_rung_planned_under_the_retired_three_percent_pair_is_admitted_on_its_evidence_and_reported(tmp_path):
    """The four published rungs were planned under the 3 % rule of 2026-09-10: their subset files carry that pair and
    their manifests hash that rule string into the store identity. The reader admits the pair THERE (the eligible set
    does not depend on the threshold; a 3 % trigger implies a 1 % trigger, so the rung is the one the current rule
    plans), reports which pair each rung was planned under, and still holds a cap_ladder.json RECORD to the current
    pair — records are re-filed by the planner, evidence is never re-written."""
    rows = small_matrix((QWEN, "main", NONE), (QWEN, "main", CONST_LORA))
    gen = tmp_path / "gen"
    G.build_store(gen, row(QWEN, "main", NONE), correct=set(range(170)))
    esc = G.build_escalated_cell(gen, row(QWEN, "main", CONST_LORA), at_cap=range(190, 199), loops={190, 191}, write_rung=False)
    retired = {**esc["subset_doc"], "rule": IR.GSM8K_TRIGGER_RULE_3PCT_RETIRED, "threshold": IR.GSM8K_TRIGGER_THRESHOLD_3PCT_RETIRED}
    C.validate_item_subset(retired)
    subset_path = G.write_rung_subset(gen, row(QWEN, "main", CONST_LORA), retired)          # the planner's file, under the retired pair
    manifest = completion.read_store_manifest(esc["source_cell_dir"])
    subset = esc["subset"]
    G.build_store(gen, row(QWEN, "main", CONST_LORA), correct={j for j in range(len(subset)) if j % 2 == 0}, unscored=set(), cap=G.CAP_RUNG,
                  ids=subset, rung_source=manifest, subset_doc=retired, subset_path=subset_path,
                  completion_tokens={j: 9000 for j in range(len(subset))}, ladder=False)
    rep = analyze(gen, matrix=rows, overlap=None)
    ck = V.cell_key(QWEN, "main", CONST_LORA)
    lad = rep["cells"][ck]["ladder"]
    assert lad["exit"] == CL.ESCALATE and lad["subset_problem"] is None and lad["rung_planned_under"] == V.RUNG_PLANNED_UNDER_RETIRED_3PCT
    assert rep["cells"][ck]["rung_leg"] is not None and rep["cells"][ck]["readings"][A.READING_MERGED]["escalated"]
    assert rep["gates"]["A6_ladder"]["status"] == A.GATE_PASS and rep["gates"]["A2_identity"]["status"] == A.GATE_PASS
    assert rep["truncation"]["rungs_planned_under"] == {V.RUNG_PLANNED_UNDER_CURRENT: [],
                                                        V.RUNG_PLANNED_UNDER_RETIRED_3PCT: [V.leg_key(QWEN, "main", CONST_LORA, 8192)]}
    # the record was filed by the adapter under the CURRENT rule: the reader holds a record to that pair alone
    record = esc["record"]
    assert record["rule"] == IR.GSM8K_TRIGGER_RULE and record["threshold"] == 0.01
    at_cap_ids = [G.IDS[j] for j in range(190, 199)]
    assert V.ladder_record_problem(record, n=200, at_cap_ids=at_cap_ids, source_manifest=manifest) is None
    stale = {**record, "rule": IR.GSM8K_TRIGGER_RULE_3PCT_RETIRED, "threshold": 0.03}
    assert "rule is not this instrument's" in (V.ladder_record_problem(stale, n=200, at_cap_ids=at_cap_ids, source_manifest=manifest) or "")
    # the validators one by one: the retired pair passes on the subset file and on the rung identity; a mismatched pair or a
    # foreign rule does not
    out_root = str(row(QWEN, "main", CONST_LORA)["out_root"])
    assert V.rung_subset_problem(retired, record=record, source_manifest=manifest, source_out_root=out_root) is None
    mixed = {**retired, "threshold": 0.01}
    assert "not the threshold its own rule states" in (V.rung_subset_problem(mixed, record=record, source_manifest=manifest) or "")
    foreign = {**retired, "rule": CL.TRIGGER_RULE}
    assert "not this instrument's" in (V.rung_subset_problem(foreign, record=record, source_manifest=manifest) or "")
    key = C.cap_rung_identity(retired)
    assert key["rule"] == IR.GSM8K_TRIGGER_RULE_3PCT_RETIRED and V.cap_rung_problem(key, source_manifest=manifest, subset=retired) is None
    assert "not this instrument's rule string" in (V.cap_rung_problem({**key, "rule": CL.TRIGGER_RULE}, source_manifest=manifest) or "")
    assert V.rung_planned_under(IR.GSM8K_TRIGGER_RULE) == V.RUNG_PLANNED_UNDER_CURRENT
    assert V.rung_planned_under(IR.GSM8K_TRIGGER_RULE_3PCT_RETIRED) == V.RUNG_PLANNED_UNDER_RETIRED_3PCT and V.rung_planned_under(CL.TRIGGER_RULE) is None

# --- superseded when the eight-arm design landed: this tree's own presentation layer — the four-family /
# peer-report assembly, `--figure-include-arm`, the trigger footers and the per-row trigger provenance — gave way to the
# MATH-500 module's shared selectors (`--figure-exclude-arm CELL`, `--figure-modes`, the eight-arm figures with a `--peer-report
# REPORT[=PIN]` column) and their tests in tests/test_eight_arms_gsm8k_analysis.py; the twelve tests of that layer were
# retired here, reproducible at the record's analysis pin. The Llama DATA path (the no-switch arm on every row, the schema-2
# matrix, the card sampling row, the row provenance, the noise floors) keeps its tests below.


# --- the certification predicates dispatch on the store's own serving path -----------


def _native_route(cell: str) -> dict:
    """The route block the driver writes on a NATIVE fitted leg: the `+lora+` identity, the engine stamped beside the
    `not_applicable` mode, no declaration, and `lens_served_positions` NULL (a native record has no served-positions map;
    `_standup_served_positions` writes the null through — the engine's fact, `header_located = False`)."""
    return {"virtual_model": C.cell_spellings(cell, declared=False).vmodel, "serving_identity": "proxy_vllm+lora+interleaved",
            C.SERVED_ENGINE_FIELD: C.SERVING_ENGINE_LORA_NATIVE, C.REDUCTION_MODE_FIELD: C.REDUCTION_MODE_NOT_APPLICABLE,
            C.ROUTE_KEY_SERVE_DECLARATIONS: {}, C.LENS_SERVED_POSITIONS_KEY: None}


def _producer_row(r: dict, *, arm: str = V.THINKING_ARM_ON) -> dict:
    """The row as the matrix LOADER files it (`row["expected_stamps"] = expected_stamps_for(fitted, declared, arm)` on EVERY
    row, engine-unaware): the certification reads the class triple off every leg's row, so a fixture row must carry it
    exactly as the producer does — a native leg's triple is unused by its reading, but present."""
    return {**r, "expected_stamps": list(V.expected_stamps_for(fitted=bool(r["fitted"]), declared=bool(r["declared"]), arm=arm))}


def _native_bites(block: dict, *, arm: str) -> dict:
    """The fixture's bites block re-stamped as the NATIVE driver files it: the ENGINE's zero cell `none__lora0`, the
    non-lens probe body (`render_fragment(lens=False, …)`), every reply's `resumed_policy` the proxy's non-lens
    `not-consulted` (no selector on a lora leg) under the probe's own render regime."""
    stamps = {**block["stamps_by_leg"]["steered"][0], "resumed_policy": "not-consulted"}
    return {**block, C.OOS_EVIDENCE_BITES_TWIN: C.TWIN_CELL_ID_LORA,
            "probe_extra_body": C.render_fragment(lens=False, arm=arm, header_located=False),
            "stamps_by_leg": {"steered": [dict(stamps) for _ in range(G.S.N_BITES_PROMPTS)],
                              "scale0": [dict(stamps) for _ in range(G.S.N_BITES_PROMPTS)]}}


def _native_fitted_inputs(r, *, no_switch: bool = False):
    """One NATIVELY served fitted leg's evidence through the real producer helpers, as the driver files it: (A)/(A″)
    through contracts, (B′) the driver's OWN inapplicability record (`served_capability_run._native_lens_position_record`),
    (C) through `contracts.lens_on_bite_verdict` (engine-neutral) or (C″) the explicit no-switch record, the bites against
    the engine's zero cell, the no-switch header measured with `lens=False` (`bs._is_lens(route)` on a lora route)."""
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    import served_capability_run as srun
    fact = G.RENDER_FACTS_ALL[r["family"]]
    cell = str(r["cell"])
    if no_switch:
        ev = {G.S.EVIDENCE_RENDER_CERTIFIED: G.render_certified_record_no_switch(fact["family"], fact),
              G.S.EVIDENCE_RENDER_PROBE: G.render_probe_record_no_switch(),
              G.EVIDENCE_THINKING_HEADER_BY_ARM: G.no_switch_header_record(fact, lens=False),
              C.OOS_EVIDENCE_BITES_BLOCK: _native_bites(G.steering_bites_record_no_switch(cell), arm=C.THINKING_NOT_AVAILABLE),
              G.S.EVIDENCE_LENS_POSITION: srun._native_lens_position_record(),
              G.S.EVIDENCE_LENS_ON_BITE: C.no_switch_lens_on_bite_record()}
    else:
        ev = {G.S.EVIDENCE_RENDER_CERTIFIED: G.S.render_certified_record(fact["family"], fact),
              G.S.EVIDENCE_RENDER_PROBE: G.S.render_probe_record(fact),
              C.OOS_EVIDENCE_BITES_BLOCK: _native_bites(G.S.steering_bites_record(cell), arm=C.THINKING_OFF),
              G.S.EVIDENCE_LENS_POSITION: srun._native_lens_position_record(),
              G.S.EVIDENCE_LENS_ON_BITE: G.lens_on_bite_record(r, fact)}
    return ev, {C.MANIFEST_KEY_ROUTE: _native_route(cell), C.LENS_SERVED_POSITIONS_KEY: None}, fact


def test_the_gsm8k_certification_validates_every_present_engine_stamp():
    """At the GSM8K instrument (it reads `certification_path_of` off its MATH-500 module): a `+lens+` fitted leg
    stamped `serving_engine: "typo"` or a present null certified before; it fails by name now; the absent historical
    stamp and an explicit `lens` pass as before; the native stamp beside the lens identity is torn; a `+lora+` leg
    with a null stamp cannot be bound to an engine."""
    unset = object()
    r = _producer_row(row(QWEN, "main", CONST_LORA))
    _ev, manifest, fact = _native_fitted_inputs(r)
    lens_r = {**r, "expected_stamps": list(V.expected_stamps_for(fitted=True, declared=False))}
    lens_ev = {G.S.EVIDENCE_RENDER_CERTIFIED: G.S.render_certified_record(fact["family"], fact),
               G.S.EVIDENCE_RENDER_PROBE: G.S.render_probe_record(fact), C.OOS_EVIDENCE_BITES_BLOCK: G.S.steering_bites_record(CONST_LORA),
               M.EVIDENCE_LENS_POSITION: G.lens_position_record(r), M.EVIDENCE_LENS_ON_BITE: G.lens_on_bite_record(r, fact)}
    lens_route = {"virtual_model": C.cell_spellings(CONST_LORA, declared=False).vmodel, C.ROUTE_KEY_SERVE_DECLARATIONS: {},
                  C.LENS_SERVED_POSITIONS_KEY: ["all"], "serving_identity": "proxy_vllm+lens+interleaved", C.REDUCTION_MODE_FIELD: "per_row"}

    def cert(engine):
        route = dict(lens_route)
        if engine is not unset:
            route[C.SERVED_ENGINE_FIELD] = engine
        return V.certification_check(lens_ev, {C.MANIFEST_KEY_ROUTE: route, C.LENS_SERVED_POSITIONS_KEY: ["all"]}, row=lens_r, fact=fact)
    for ok in (unset, C.SERVING_ENGINE_LENS):
        c = cert(ok)
        assert c["failures"] == [] and c["passed"], (ok, c["failures"])
    for bad in ("typo", None, "hf"):
        c = cert(bad)
        assert not c["passed"] and [f for f in c["failures"] if f"route.{C.SERVED_ENGINE_FIELD} {bad!r} is not one of" in f], (bad, c["failures"])
    c = cert(C.SERVING_ENGINE_LORA_NATIVE)
    assert not c["passed"] and [f for f in c["failures"] if "a torn record" in f]
    # the native leg with its stamp NULLED (present, not absent): an unknown engine, by name — and the lens predicates
    # then refuse the native record in their own words too
    route = manifest[C.MANIFEST_KEY_ROUTE]
    d = " ".join(V.certification_check(_ev, {**manifest, C.MANIFEST_KEY_ROUTE: {**route, C.SERVED_ENGINE_FIELD: None}}, row=r, fact=fact)["failures"])
    assert f"route.{C.SERVED_ENGINE_FIELD} None is not one of" in d and "cannot be bound to an engine" in d and "another engine's record" in d
    d = " ".join(V.certification_check(_ev, {**manifest, C.MANIFEST_KEY_ROUTE: {k: v for k, v in route.items() if k != C.SERVED_ENGINE_FIELD}}, row=r, fact=fact)["failures"])
    assert "names the native path but" in d and "is absent" in d and "cannot be bound to an engine" in d


def test_the_gsm8k_certification_accepts_a_native_fitted_leg_and_refuses_the_lens_facts_by_name():
    """The GSM8K reader required the LENS (B′) fields, the object-digest map, the policy and the stamp triple on EVERY
    fitted arm, so a correct native LoRA store was REJECTED. It now dispatches on the route's serving path (the
    MATH-500 reader's one rule): a native leg's producer-built evidence passes under both certification classes; every
    lens-only fact on a native record, a torn identity/engine pairing, a header-located row served natively and the
    native record on a lens-served job refuse BY NAME; the lens reading is untouched."""
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    import served_capability_run as srun
    for key in (QWEN, GEMMA, NEMO, GLM):                       # the SWITCH class, an undeclared lora cell
        r = _producer_row(row(key, "main", CONST_LORA))
        ev, manifest, fact = _native_fitted_inputs(r)
        cert = V.certification_check(ev, manifest, row=r, fact=fact)
        assert cert["failures"] == [], (key, cert["failures"])
        assert cert["passed"] and cert[M.EVIDENCE_LENS_POSITION]["status"] == C.NOT_APPLICABLE
        assert cert["route_lens_served_positions"] is None and cert[M.EVIDENCE_BITES][C.OOS_EVIDENCE_BITES_TWIN] == C.TWIN_CELL_ID_LORA
    lr = _producer_row(G.row_for(LLAMA_ROWS, LLAMA, "main", CONST_LORA), arm=C.THINKING_NOT_AVAILABLE)     # the NO-SWITCH class: (C″) is the same explicit record on both engines
    ev, manifest, fact = _native_fitted_inputs(lr, no_switch=True)
    cert = V.certification_check(ev, manifest, row=lr, fact=fact, arm=C.THINKING_NOT_AVAILABLE)
    assert cert["failures"] == [], cert["failures"]
    assert cert["passed"] and cert["cert_class"] == C.CERT_CLASS_NO_SWITCH

    r = _producer_row(row(QWEN, "main", CONST_LORA))
    ev, manifest, fact = _native_fitted_inputs(r)
    route = manifest[C.MANIFEST_KEY_ROUTE]

    def fails(ev_=None, man=None, *, needle, r_=None):
        d = " ".join(V.certification_check(ev_ if ev_ is not None else ev, man if man is not None else manifest,
                                           row=r_ if r_ is not None else r, fact=fact)["failures"])
        assert needle in d, (needle, d)
        return d

    fails({**ev, M.EVIDENCE_LENS_POSITION: G.lens_position_record(r)}, needle="another engine's")
    fails({k: v for k, v in ev.items() if k != M.EVIDENCE_LENS_POSITION}, needle="never an absent block")
    fails({**ev, M.EVIDENCE_LENS_POSITION: {"verified": False, "error": "OOS_DECLARATION_MISMATCH: x"}}, needle="FAILED as recorded")
    fails(man={**manifest, C.MANIFEST_KEY_ROUTE: {**route, C.LENS_SERVED_POSITIONS_KEY: ["all"]}, C.LENS_SERVED_POSITIONS_KEY: ["all"]},
          needle="a served-positions map is a lens fact")
    fails(r_={**r, "served_positions_expected": "response"}, needle="a header-located row has no native standup")
    fails({**ev, C.OOS_EVIDENCE_BITES_BLOCK: {**ev[C.OOS_EVIDENCE_BITES_BLOCK], C.OOS_EVIDENCE_BITES_TWIN: C.TWIN_CELL_ID}},
          needle="the ENGINE's zero cell")
    torn = {**manifest, C.MANIFEST_KEY_ROUTE: {k: v for k, v in route.items() if k != C.SERVED_ENGINE_FIELD}}
    d = fails(man=torn, needle="cannot be bound to an engine")
    assert "another engine's record" in d
    # a DECLARED (header-located) row can have no native store
    dr = {**row(QWEN, "main", SI12_LORA), "expected_stamps": list(V.expected_stamps_for(fitted=True, declared=True))}
    dev = {**ev, G.EVIDENCE_THINKING_HEADER_BY_ARM: G.thinking_header_by_arm_record(fact)}
    fails(dev, man={**manifest, C.MANIFEST_KEY_ROUTE: {**route, "virtual_model": C.cell_spellings(SI12_LORA, declared=True).vmodel}},
          r_=dr, needle="undeclared all-position cells only")
    # the LENS reading: the fixture's lens leg passes with and without its identity stated (the real route block carries
    # it), and the native inapplicability record on a lens-served job refuses by name
    lens_r = {**r, "expected_stamps": list(V.expected_stamps_for(fitted=True, declared=False))}
    lens_ev = {G.S.EVIDENCE_RENDER_CERTIFIED: G.S.render_certified_record(fact["family"], fact),
               G.S.EVIDENCE_RENDER_PROBE: G.S.render_probe_record(fact), C.OOS_EVIDENCE_BITES_BLOCK: G.S.steering_bites_record(CONST_LORA),
               M.EVIDENCE_LENS_POSITION: G.lens_position_record(r), M.EVIDENCE_LENS_ON_BITE: G.lens_on_bite_record(r, fact)}
    lens_route = {"virtual_model": C.cell_spellings(CONST_LORA, declared=False).vmodel, C.ROUTE_KEY_SERVE_DECLARATIONS: {},
                  C.LENS_SERVED_POSITIONS_KEY: ["all"]}
    for extra in ({}, {"serving_identity": "proxy_vllm+lens+interleaved", C.REDUCTION_MODE_FIELD: "per_row"}):
        lm = {C.MANIFEST_KEY_ROUTE: {**lens_route, **extra}, C.LENS_SERVED_POSITIONS_KEY: ["all"]}
        cert = V.certification_check(lens_ev, lm, row=lens_r, fact=fact)
        assert cert["failures"] == [], cert["failures"]
    d = " ".join(V.certification_check({**lens_ev, M.EVIDENCE_LENS_POSITION: srun._native_lens_position_record()}, lm,
                                       row=lens_r, fact=fact)["failures"])
    assert "on a lens-served job" in d and "another engine's record" in d


def test_a_natively_served_leg_is_held_to_the_engines_own_stamp_triple_never_the_lens_classes():
    """Served-steer stamps x native serving by default, the seam the textual merge did not show: the matrix loader files the
    LENS class triple (`cell-stable`, `always`, render) on EVERY fitted row, engine-unaware, and the served-steer class (the lens
    launcher's never-resume triple) is selected from `steering_bites.liveness_gate == logprob`, which the producer writes on
    EITHER engine. A natively served leg's responses are stamped by the proxy's non-lens rule — `cell-stable`, `not-consulted`,
    the arm's render (contracts.expected_stamp_regimes lens=False) — so, handed the store's route, the selector holds it to THAT
    triple (class `native_engine`) whatever its liveness statistic, on the switch and the no-switch arm, and the A2 summary-stamp
    check then passes on the engine's stamps and fails on the row's. The same evidence on a lens route, on an identity-less route
    (every lens store on record) or with no route reads exactly as the served-steer rule writes it."""
    r = _producer_row(row(QWEN, "main", CONST_LORA))
    ev, manifest, fact = _native_fitted_inputs(r)
    route = manifest[C.MANIFEST_KEY_ROUTE]
    engine_on = list(C.expected_stamp_regimes(lens=False, arm=C.THINKING_ON, policy=C.POLICY_NOT_CONSULTED))
    assert engine_on[:2] == ["cell-stable", "not-consulted"] and engine_on != list(r["expected_stamps"])
    for ev_ in (ev, {**ev, C.OOS_EVIDENCE_BITES_BLOCK: {**ev[C.OOS_EVIDENCE_BITES_BLOCK], C.OOS_EVIDENCE_BITES_LIVENESS_GATE: C.LENS_LIVENESS_LOGPROB}}):
        native = V.leg_expected_stamps(r, ev_, arm=V.THINKING_ARM_ON, route=route)
        assert native["class"] == V.STAMP_CLASS_NATIVE and native["problem"] is None and native["stamps"] == engine_on, native
        assert native["selected_by"] == V.STAMP_CLASS_NATIVE_SELECTOR
    # the no-switch arm: the engine's triple at the `unspecified` render
    lr = _producer_row(G.row_for(LLAMA_ROWS, LLAMA, "main", CONST_LORA), arm=C.THINKING_NOT_AVAILABLE)
    lev, lman, _ = _native_fitted_inputs(lr, no_switch=True)
    na = V.leg_expected_stamps(lr, lev, arm=C.THINKING_NOT_AVAILABLE, route=lman[C.MANIFEST_KEY_ROUTE])
    assert na["class"] == V.STAMP_CLASS_NATIVE
    assert na["stamps"] == list(C.expected_stamp_regimes(lens=False, arm=C.THINKING_NOT_AVAILABLE, policy=C.POLICY_NOT_CONSULTED))
    assert na["stamps"][2] == C.RENDER_REGIME_UNSPECIFIED and na["stamps"] != list(lr["expected_stamps"])
    # the A2 boundary (tinycap_gsm8k_paired200: `M.stamp_check(summary, stamp_class["stamps"])`): a native summary stamped as the
    # proxy stamps a non-lens leg passes against the engine's triple and FAILS against the row's lens triple
    def native_summary(stamps):
        return {**{f: v for f, v in zip(C.EXPECTED_STAMP_TRIPLE_FIELDS, stamps)}, **{C.STAMP_REGIME_COUNT_FIELD[f]: 1 for f in C.EXPECTED_STAMP_TRIPLE_FIELDS}}
    ok = M.stamp_check(native_summary(engine_on), native["stamps"])
    assert ok["failures"] == [], ok
    bad = M.stamp_check(native_summary(engine_on), list(r["expected_stamps"]))
    assert bad["failures"] and any("resumed_policy" in f and "not-consulted" in f and "always" in f for f in bad["failures"]), bad
    # the lens readings are untouched: the selector's original shapes
    ev_lp = {**ev, C.OOS_EVIDENCE_BITES_BLOCK: {**ev[C.OOS_EVIDENCE_BITES_BLOCK], C.OOS_EVIDENCE_BITES_LIVENESS_GATE: C.LENS_LIVENESS_LOGPROB}}
    lens_route = {"virtual_model": C.cell_spellings(CONST_LORA, declared=False).vmodel, C.ROUTE_KEY_SERVE_DECLARATIONS: {},
                  C.LENS_SERVED_POSITIONS_KEY: ["all"], "serving_identity": "proxy_vllm+lens+interleaved"}
    lens = V.leg_expected_stamps(r, ev_lp, arm=V.THINKING_ARM_ON, route=lens_route)
    assert lens["class"] == V.STAMP_CLASS_SERVED_STEER and lens["stamps"] == list(V.served_steer_stamps(V.THINKING_ARM_ON))
    assert V.leg_expected_stamps(r, ev_lp, arm=V.THINKING_ARM_ON)["class"] == V.STAMP_CLASS_SERVED_STEER
    assert V.leg_expected_stamps(r, ev_lp, arm=V.THINKING_ARM_ON, route={"virtual_model": lens_route["virtual_model"]})["class"] == V.STAMP_CLASS_SERVED_STEER
    assert V.leg_expected_stamps(r, ev, arm=V.THINKING_ARM_ON, route=lens_route)["class"] == V.STAMP_CLASS_ROW
    # a torn native route (engine stamped, identity lens) gets NO native exemption here; certification_check names it
    torn = {**route, "serving_identity": "proxy_vllm+lens+interleaved"}
    assert V.leg_expected_stamps(r, ev_lp, arm=V.THINKING_ARM_ON, route=torn)["class"] != V.STAMP_CLASS_NATIVE


def test_the_all_positions_arditi_arm_is_admissible_on_the_native_route_by_the_same_route_keyed_selector():
    """arditi-allpos-cap: the stamp-class selector (`leg_expected_stamps`) and the certification path (`certification_path_of`) are
    keyed on the store's ROUTE — nothing in the readers keys on the literal cell id — so a SYNTHETIC `arditi_allpos` row served
    natively (the `+lora+` identity stamped `serving_engine: lora_native`) is held to class `native_engine` with the engine's own
    triple exactly as the `arditi` row is, whatever its liveness statistic; on a lens route under the logprob gate it would be
    served-steer; with no route, the row's class. The route block is written by hand here: the cell GRAMMAR on this tree does not
    admit `arditi_allpos` (`contracts.cell_spellings` refuses it — its Class B registration lands in a separate change), which is
    the blocked serving support this test names rather than works around."""
    # the grammar ADMITS the id since the Class B registration landed (it refused before, which is what made the route block
    # below hand-written); the spellings of a bare Class B id are the id itself on both sides
    spell = C.cell_spellings("arditi_allpos", declared=False)
    assert spell.harness == "arditi_allpos" and "arditi_allpos" in C.launchable_class_b_ids()
    assert "arditi_allpos" in M.ARM_BY_CELL and M.ARM_BY_CELL["arditi_allpos"].external
    # a SYNTHETIC row of the new arm, built off an existing fitted/undeclared synthetic row (the matrix carries no such row —
    # the arm's own matrix is a new file of the launcher's, and no store exists on either cluster)
    r = _producer_row({**row(QWEN, "main", CONST_LORA), "cell": "arditi_allpos", "cell_slug": "arditi_allpos",
                       "harness_cell": "arditi_allpos", "view": "it", "object_source": "external",
                       "selection_outcome": "user_override_over_budget_2026-09-16"})
    assert r["fitted"] and not r["declared"]
    route = {"virtual_model": "petri__arditi_allpos", "serving_identity": "proxy_vllm+lora+interleaved",
             C.SERVED_ENGINE_FIELD: C.SERVING_ENGINE_LORA_NATIVE, C.REDUCTION_MODE_FIELD: C.REDUCTION_MODE_NOT_APPLICABLE,
             C.ROUTE_KEY_SERVE_DECLARATIONS: {}, C.LENS_SERVED_POSITIONS_KEY: None}
    assert M.certification_path_of(route) == (M.STEERED_PATH_LORA, [])
    assert V.route_class_of("arditi_allpos") == M.ROUTE_STEERED and M.steered_path_of(route["serving_identity"]) == "lora"
    engine_on = list(C.expected_stamp_regimes(lens=False, arm=C.THINKING_ON, policy=C.POLICY_NOT_CONSULTED))
    for gate in (C.LENS_LIVENESS_TEXT, C.LENS_LIVENESS_LOGPROB):
        ev = {C.OOS_EVIDENCE_BITES_BLOCK: {C.OOS_EVIDENCE_BITES_LIVENESS_GATE: gate, "object_kind": "vector_all"}}
        sel = V.leg_expected_stamps(r, ev, arm=V.THINKING_ARM_ON, route=route)
        assert sel["class"] == V.STAMP_CLASS_NATIVE == "native_engine" and sel["problem"] is None and sel["stamps"] == engine_on, sel
        assert sel["selected_by"] == V.STAMP_CLASS_NATIVE_SELECTOR
    ev_lp = {C.OOS_EVIDENCE_BITES_BLOCK: {C.OOS_EVIDENCE_BITES_LIVENESS_GATE: C.LENS_LIVENESS_LOGPROB, "object_kind": "vector_all"}}
    lens_route = {**{k: v for k, v in route.items() if k != C.SERVED_ENGINE_FIELD}, "serving_identity": "proxy_vllm+lens+interleaved"}
    assert V.leg_expected_stamps(r, ev_lp, arm=V.THINKING_ARM_ON, route=lens_route)["class"] == V.STAMP_CLASS_SERVED_STEER
    assert V.leg_expected_stamps(r, ev_lp, arm=V.THINKING_ARM_ON)["class"] == V.STAMP_CLASS_SERVED_STEER
    # THE DEPENDENCY THAT LANDED: the MATH-500 reader's logprob-liveness rule admits only `contracts.LOGPROB_LIVENESS_KINDS`, which
    # now CARRIES `vector_all` — so a logprob-gated leg of this arm is scoped for the logprob statistic instead of silently taking
    # the greedy-TEXT one, which is exactly where a correct constant add at a small norm reads as a no-op. Membership, not equality:
    # the tuple legitimately grows. The refusal is still pinned, on a kind that really is outside it.
    assert "vector_all" in C.LOGPROB_LIVENESS_KINDS and "psr" not in C.LOGPROB_LIVENESS_KINDS
    assert M._logprob_liveness_problems({"object_kind": "vector_all", C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS: None})[0] \
        != f"steering bites: liveness_gate logprob on object_kind 'vector_all', not one of {list(C.LOGPROB_LIVENESS_KINDS)}"
    problems = M._logprob_liveness_problems({"object_kind": "psr", C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS: None})
    assert problems[0] == f"steering bites: liveness_gate logprob on object_kind 'psr', not one of {list(C.LOGPROB_LIVENESS_KINDS)}"


def test_the_gsm8k_cell_record_carries_the_rows_declared_selection_and_fit_provenance_blocks_verbatim():
    """The GSM8K reader files the matrix row's DECLARED `selection` and `fit_provenance` blocks on the cell record beside its
    `selection_outcome` (the paper-rows export reads them through `_pr_stamp`); a row declaring none leaves them null."""
    import inspect
    src = inspect.getsource(A.collect_cell) if hasattr(A, "collect_cell") else Path(A.__file__).read_text(encoding="utf-8")
    assert 'cell["selection"] = row.get("selection")' in src and 'cell["fit_provenance"] = row.get("fit_provenance")' in src
    sel = {"outcome": "user_override_over_budget_2026-09-16", "rule_returns": "ar_c0p2", "rule_reading_pp": 5.04, "override_reading_pp": 10.26,
           "budget_pp": 10, "ladder_cap": 2048, "single_trial": True, "dose": {"value": 0.3, "unit": "raw_coefficient"},
           "alpha_equiv": None, "replicate_note": "single trial"}
    cell = {"cell_key": "k", "cell": "arditi_allpos", "fitted": True, "selection_outcome": sel["outcome"], "selection": sel}
    assert A._pr_selection(cell, {}, selection_outcome=sel["outcome"])[0] == sel
    assert A._pr_selection({**cell, "selection": None}, {}, selection_outcome=sel["outcome"]) == (None, None)
    with pytest.raises(A.AnalysisError, match="ladder_cap"):
        A._pr_selection({**cell, "selection": {k: v for k, v in sel.items() if k != "ladder_cap"}}, {}, selection_outcome=sel["outcome"])
    assert A._pr_serving_commit(cell, {}) == (None, None) and A._pr_fit_provenance(cell, {}) == (None, None)
    with pytest.raises(A.AnalysisError, match="fit_provenance is missing"):
        A._pr_fit_provenance({**cell, "fit_provenance": {"fit_layer": 13}}, {})


def test_a_natively_served_static_r_row_reads_class_native_engine_because_the_selector_is_route_keyed():
    """A never-run Arditi row takes the NATIVE path (the pin follows the store), and the stamp-class selector is keyed on the
    store's ROUTE, never on the cell's kind, so an Arditi (kind `static_r`) external row served natively — the `+lora+`
    identity stamped `serving_engine: lora_native` — is held to class `native_engine` (the engine's own triple) whatever its
    liveness statistic, exactly as a LoRA row is; on the lens route under the logprob gate it stays the served-steer class
    (every Arditi store on record today), and with no route the row's class. Pinned on the launcher's REAL external rows
    (the frozen matrices: the switch-arm Arditi rows and the no-switch Llama one) so the validators need no change when the
    first native Arditi store lands; the validators themselves are unchanged by this test."""
    g8k_cells = _optional_g8k_cells()
    if g8k_cells is None:
        pytest.skip("the GSM8K launcher's matrix reader is not importable on this tree")
    arditi = [r for r in g8k_cells.matrix_rows() if r.get("object_source") == g8k_cells.OBJECT_SOURCE_EXTERNAL and r["cell"] == "arditi"]
    assert len(arditi) == 3 and all(r["fitted"] and not r["declared"] and g8k_cells.row_thinking(r) == "on" for r in arditi), [r["job_id"] for r in arditi]
    engine_on = list(C.expected_stamp_regimes(lens=False, arm=C.THINKING_ON, policy=C.POLICY_NOT_CONSULTED))
    for r0 in arditi:
        r = _producer_row(r0)
        ev, manifest, _fact = _native_fitted_inputs(r)
        route = manifest[C.MANIFEST_KEY_ROUTE]
        assert route["serving_identity"] == "proxy_vllm+lora+interleaved" and route[C.SERVED_ENGINE_FIELD] == C.SERVING_ENGINE_LORA_NATIVE
        assert M.certification_path_of(route) == (M.STEERED_PATH_LORA, [])
        ev_lp = {**ev, C.OOS_EVIDENCE_BITES_BLOCK: {**ev[C.OOS_EVIDENCE_BITES_BLOCK], C.OOS_EVIDENCE_BITES_LIVENESS_GATE: C.LENS_LIVENESS_LOGPROB}}
        for ev_ in (ev, ev_lp):
            sel = V.leg_expected_stamps(r, ev_, arm=V.THINKING_ARM_ON, route=route)
            assert sel["class"] == V.STAMP_CLASS_NATIVE == "native_engine" and sel["problem"] is None and sel["stamps"] == engine_on, (r["job_id"], sel)
            assert sel["selected_by"] == V.STAMP_CLASS_NATIVE_SELECTOR and sel["stamps"] != list(r["expected_stamps"])
        # the lens readings on the same row are #137's: served-steer under the logprob gate (the stores on record), the row's class without it
        lens_route = {k: v for k, v in route.items() if k != C.SERVED_ENGINE_FIELD}
        lens_route["serving_identity"] = "proxy_vllm+lens+interleaved"
        assert V.leg_expected_stamps(r, ev_lp, arm=V.THINKING_ARM_ON, route=lens_route)["class"] == V.STAMP_CLASS_SERVED_STEER
        assert V.leg_expected_stamps(r, ev_lp, arm=V.THINKING_ARM_ON)["class"] == V.STAMP_CLASS_SERVED_STEER
        assert V.leg_expected_stamps(r, ev, arm=V.THINKING_ARM_ON, route=lens_route)["class"] == V.STAMP_CLASS_ROW
    # the no-switch Llama Arditi row (matrix_llama row 13): the engine's triple at the `unspecified` render
    llama = [r for r in g8k_cells.matrix_rows(g8k_cells.resolve_matrix("llama"))
             if r.get("object_source") == g8k_cells.OBJECT_SOURCE_EXTERNAL and r["cell"] == "arditi"]
    assert len(llama) == 1 and g8k_cells.row_thinking(llama[0]) == "not_available"
    lr = _producer_row(llama[0], arm=C.THINKING_NOT_AVAILABLE)
    lev, lman, _ = _native_fitted_inputs(lr, no_switch=True)
    na = V.leg_expected_stamps(lr, lev, arm=C.THINKING_NOT_AVAILABLE, route=lman[C.MANIFEST_KEY_ROUTE])
    assert na["class"] == V.STAMP_CLASS_NATIVE and na["stamps"] == list(C.expected_stamp_regimes(lens=False, arm=C.THINKING_NOT_AVAILABLE, policy=C.POLICY_NOT_CONSULTED))
    assert na["stamps"][2] == C.RENDER_REGIME_UNSPECIFIED and na["stamps"] != list(lr["expected_stamps"])


# --------------------------------------------------------------------------- #
# native-positions-resolve: the NATIVE record gap — `transfer_provenance.served_positions` null on every natively served
# fitted card (the driver reads the standup's LENS map only, which a native standup writes empty; the block's header fields
# were then derived from the null, fail closed), resolved from the native standup record beside the job                #
# --------------------------------------------------------------------------- #

NATIVE_TRIPLE = (QWEN, "main", CONST_LORA)
NATIVE_FIXTURE = REPO / "tests" / "fixtures" / "gsm8k_native_card"
TP, ROUTE = C.MANIFEST_KEY_TRANSFER_PROVENANCE, C.MANIFEST_KEY_ROUTE


def _native_gap_store(root, triple=NATIVE_TRIPLE, *, rows=None, served_positions=None, position_mode="all", record=True,
                      served_positions_map=None, record_edit=None, manifest_edit=None):
    """One NATIVELY served undeclared fitted store carrying the producer's record gap — a store written through the real
    writers, re-stamped as the native driver files it (`G.nativize_manifest`: the `+lora+` route with the engine stamp, the
    engine's `header_located: false`, null served-positions maps, the transfer-provenance block through the real helper with
    the driver's null so its header fields derive fail-closed exactly as the real card's) — with the native standup record
    beside its job (`G.native_standup_record_docs`, the real record's shape). `served_positions` simulates the fixed producer;
    `position_mode` is the routes[] attestation (None: an unattested route); `served_positions_map` the engine-neutral map;
    `record=False` leaves no pulled record. Returns (row, cell_dir, manifest, arm, fact)."""
    r = G.row_for(rows, *triple) if rows is not None else row(*triple)
    arm = str(r.get("thinking_mode") or C.THINKING_ON)
    fact = G.RENDER_FACTS_ALL[r["family"]]

    def edit(m):
        G.nativize_manifest(m, r, fact, arm=arm, served_positions=served_positions)
        if manifest_edit:
            manifest_edit(m)
    cd = G.build_store(root, r, correct=range(120), manifest_edit=edit, launch_record=False, ladder=False)
    manifest = json.loads((cd / "manifest.json").read_text())
    if record:
        docs = G.native_standup_record_docs(r, fact, cap=G.CAP_BASE, tree_sha=G.FULL_SHA, intervention_hash=manifest["intervention_hash"],
                                            max_model_len=manifest[C.MANIFEST_KEY_TINYCAP]["served"]["max_model_len"],
                                            position_mode=position_mode, served_positions_map=served_positions_map)
        if record_edit:
            record_edit(docs)
        G.write_standup_record(root, G.standup_jobdir(r, G.CAP_BASE), docs)
    return r, cd, manifest, arm, fact


def _native_identity(r, cd, manifest, arm, fact, *, standup_record):
    summary = completion.read_task_summary(cd, G.TASK)
    return V.identity_problem(manifest, summary, row=r, cap=G.CAP_BASE, model_pin=G.S.model_pin(str(r["tag"])),
                              sampling_row=G.SAMPLING_ROWS[str(r["sampling_mode"])][str(r["model_key"])], fact=fact,
                              standup_record=standup_record, arm=arm)


def test_the_native_record_constants_are_shared_with_the_fixture_builder():
    assert V.STANDUP_SERVING_ATTESTATION == G.STANDUP_SERVING_ATTESTATION == "serving_attestation.json"
    # launch-attestation-native: the launch attestation's NATIVE record is the same file, by import from the engine's own module,
    # and the two launch records are keyed by exactly the contracts' two engines
    assert M.LAUNCH_NATIVE_ATTESTATION_FILE == SA.SERVING_ATTESTATION_FILE == G.S.STANDUP_SERVING_ATTESTATION == G.STANDUP_SERVING_ATTESTATION
    assert M.LAUNCH_ATTESTATION_FILE == V.STANDUP_ATTESTATION == G.STANDUP_ATTESTATION == "steering_attestation.json"
    assert M.LAUNCH_RECORD_FILE_BY_ENGINE == {C.SERVING_ENGINE_LENS: M.LAUNCH_ATTESTATION_FILE, C.SERVING_ENGINE_LORA_NATIVE: M.LAUNCH_NATIVE_ATTESTATION_FILE}
    assert tuple(M.LAUNCH_RECORD_FILE_BY_ENGINE) == C.SERVING_ENGINES and A.LAUNCH_TASK_SUBDIR == V.STANDUP_TASK_SUBDIR == "gsm8k"
    assert V.NATIVE_STANDUP_RECORD_FILES == (V.STANDUP_ROUTING_TABLE, V.STANDUP_JOBS, V.STANDUP_SERVING_ATTESTATION)
    assert V.SERVED_POSITIONS_KEY == "served_positions" and V.ROUTE_POSITION_MODE_KEY == "petri_position_mode"
    assert V.NATIVE_RECORD_GAP_FIELDS == ("transfer_provenance.served_positions", "transfer_provenance.header_located",
                                          "transfer_provenance.header_check_class")
    assert V.NATIVE_RECORD_GAP_FIELDS[0] == V.RECORD_GAP_FIELDS[0]
    assert "served_capability_run.py:5505,5591-5601,6323" in V.RECORD_GAP_CAUSE_NATIVE_LENS_MAP_ONLY and "@b367a35d" in V.RECORD_GAP_CAUSE_NATIVE_LENS_MAP_ONLY
    assert V.RECORD_GAP_CAUSE_NATIVE_LENS_MAP_ONLY != V.RECORD_GAP_CAUSE_BARE_KEY_LOOKUP
    # the contract functions the record re-derives through: the fail-closed reading of the null and the reading of `all`
    assert C.header_located(None) is True and C.header_located(["all"]) is False
    assert C.header_check_class(fit_response_header_len=None, served_positions=None) == C.HEADER_CHECK_DOCUMENT_FIT_SERVED_HEADER_LOCATED
    assert C.header_check_class(fit_response_header_len=None, served_positions=["all"]) == C.HEADER_CHECK_DOCUMENT_FIT_INAPPLICABLE


def test_native_gap_i_a_natively_served_fitted_card_with_the_producers_null_resolves_from_its_standup_record(tmp_path):
    """(i) the natively served undeclared fitted card — the SWITCH class (qwen, thinking-ON) and the NO-SWITCH class (llama,
    the real cards' arm) — carries the producer's null with the block's two header fields derived fail-closed from it, beside
    the engine's own `header_located: false`; the identity check REFUSED it before. Now the native standup record beside the
    job resolves `["all"]` off the routes[] attestation (`petri_position_mode`), cross-checked to the manifest and the job, the
    block's header fields are RE-DERIVED through the contract, the leg is admitted WITH the record only, and the three gaps
    are filed by field with the producer cause; the native certification passes and reports the record. The ENGINE-NEUTRAL map
    (the producer fix's record) is read FIRST when present; the FIXED producer's stated value needs no record at all."""
    for name, triple, rows in (("qwen_on", NATIVE_TRIPLE, None), ("llama_no_switch", (LLAMA, "main", CONST_LORA), LLAMA_ROWS)):
        root = tmp_path / name
        r, cd, manifest, arm, fact = _native_gap_store(root, triple, rows=rows)
        tp = manifest[TP]
        assert tp["served_positions"] is None and tp["header_located"] is True and tp["header_check_class"] == C.HEADER_CHECK_DOCUMENT_FIT_SERVED_HEADER_LOCATED
        assert manifest["header_located"] is False and manifest[C.LENS_SERVED_POSITIONS_KEY] is None and manifest[ROUTE][C.LENS_SERVED_POSITIONS_KEY] is None
        assert manifest[ROUTE][C.SERVED_ENGINE_FIELD] == C.SERVING_ENGINE_LORA_NATIVE and M.certification_path_of(manifest[ROUTE]) == (M.STEERED_PATH_LORA, [])
        assert V.native_served_positions_gap(manifest) is True
        rec = V.native_served_positions_record(manifest, row=r, gen_root=root)
        assert rec["problem"] is None, (name, rec["problem"])
        assert rec["resolved"] == ["all"] and rec["source"] == V.NATIVE_RESOLVED_FROM_ROUTES and rec["engine"] == C.SERVING_ENGINE_LORA_NATIVE
        assert rec["header_located"] is False and rec["header_check_class"] == C.HEADER_CHECK_DOCUMENT_FIT_INAPPLICABLE
        assert rec["resolved_from"].endswith(f"/{V.STANDUP_TASK_SUBDIR}/{V.STANDUP_ROUTING_TABLE}.routes[{r['harness_cell']!r}].{V.ROUTE_POSITION_MODE_KEY}")
        assert rec["jobdir"].endswith(rec["job_id"]) and rec["job_id"].isdigit() and Path(rec["standup_dir"]).is_dir()
        gaps = rec["gaps"]
        assert [g["field"] for g in gaps] == list(V.NATIVE_RECORD_GAP_FIELDS)
        assert [(g["manifest"], g["resolved"]) for g in gaps] == [(None, ["all"]), (True, False),
                                                                  (C.HEADER_CHECK_DOCUMENT_FIT_SERVED_HEADER_LOCATED, C.HEADER_CHECK_DOCUMENT_FIT_INAPPLICABLE)]
        assert all(g["cause"] == V.RECORD_GAP_CAUSE_NATIVE_LENS_MAP_ONLY and g["source"] == V.NATIVE_RESOLVED_FROM_ROUTES
                   and g["resolved_from"] == rec["resolved_from"] for g in gaps)
        assert [g.get("derivation") for g in gaps] == [None, "contracts.header_located(<the resolved positions>)",
                                                        "contracts.header_check_class(<the resolved positions>)"]
        # admitted WITH the record only: the same null with none read refuses by name, naming both record keys
        assert _native_identity(r, cd, manifest, arm, fact, standup_record=rec) is None
        problem = _native_identity(r, cd, manifest, arm, fact, standup_record=None)
        assert problem and "natively served UNDECLARED cell" in problem and "none was read" in problem, problem
        assert V.SERVED_POSITIONS_KEY in problem and V.ROUTE_POSITION_MODE_KEY in problem and "None != ['all']" in problem
        # the identity check does not trust a record blindly: a record resolving another mode, or one whose manifest pairing is
        # not the fail-closed derivation, refuses in its own words even when the record says `problem: None`
        forged = {**rec, "resolved": ["response"]}
        assert "resolves ['response'] != ['all']" in _native_identity(r, cd, manifest, arm, fact, standup_record=forged)
        bad_pair = json.loads(json.dumps(manifest))
        bad_pair[TP]["header_located"] = False
        assert "not the contract's fail-closed derivation" in _native_identity(r, cd, bad_pair, arm, fact, standup_record=rec)
        # the native certification passes on the native evidence and carries the record beside its verdict
        pr = _producer_row(r, arm=arm)
        ev, _man, _fact = _native_fitted_inputs(pr, no_switch=(arm == C.THINKING_NOT_AVAILABLE))
        cert = V.certification_check(ev, manifest, row=pr, fact=fact, standup_record=rec, arm=arm)
        assert cert["failures"] == [] and cert["passed"], (name, cert["failures"])
        assert cert["standup_record"] == {k: rec[k] for k in ("resolved", "resolved_from", "source", "problem", "jobdir", "job_id")}
        assert cert["route_lens_served_positions"] is None and cert["route_lens_served_positions_source"] == "manifest"
    harness = row(*NATIVE_TRIPLE)["harness_cell"]
    # the ENGINE-NEUTRAL map alone (no routes[] attestation at all): resolved from it, by name
    root = tmp_path / "neutral"
    r, cd, manifest, arm, fact = _native_gap_store(root, position_mode=None, served_positions_map={harness: "all"})
    rec = V.native_served_positions_record(manifest, row=r, gen_root=root)
    assert rec["problem"] is None and rec["resolved"] == ["all"] and rec["source"] == V.NATIVE_RESOLVED_FROM_NEUTRAL_MAP
    assert rec["resolved_from"].endswith(f"{V.STANDUP_ROUTING_TABLE}[{V.SERVED_POSITIONS_KEY}][{harness!r}]")
    assert all(g["source"] == V.NATIVE_RESOLVED_FROM_NEUTRAL_MAP for g in rec["gaps"]) and len(rec["gaps"]) == 3
    assert _native_identity(r, cd, manifest, arm, fact, standup_record=rec) is None
    # both records present and agreeing: the neutral map is the source
    root = tmp_path / "both"
    r, cd, manifest, arm, fact = _native_gap_store(root, position_mode="all", served_positions_map={harness: ["all"]})
    rec = V.native_served_positions_record(manifest, row=r, gen_root=root)
    assert rec["problem"] is None and rec["resolved"] == ["all"] and rec["source"] == V.NATIVE_RESOLVED_FROM_NEUTRAL_MAP
    # the FIXED producer: the stated value is held to the row as before and needs no record; the record still binds, no gap
    root = tmp_path / "fixed"
    r, cd, manifest, arm, fact = _native_gap_store(root, served_positions=["all"])
    tp = manifest[TP]
    assert tp["served_positions"] == ["all"] and tp["header_located"] is False and tp["header_check_class"] == C.HEADER_CHECK_DOCUMENT_FIT_INAPPLICABLE
    assert V.native_served_positions_gap(manifest) is False
    assert _native_identity(r, cd, manifest, arm, fact, standup_record=None) is None
    rec = V.native_served_positions_record(manifest, row=r, gen_root=root)
    assert rec["problem"] is None and rec["resolved"] == ["all"] and rec["gaps"] == []
    # ... and a fixed producer whose stated value disagrees with the record, or whose derivations do not follow, refuses
    for edit, word in ((lambda m: m[TP].update(served_positions=["response"]), "disagrees with the standup record"),
                       (lambda m: m[TP].update(header_located=True), "!= contracts.header_located"),
                       (lambda m: m[TP].update(header_check_class=C.HEADER_CHECK_FIT_HEADER_RECORDED), "!= contracts.header_check_class")):
        m2 = json.loads(json.dumps(manifest))
        edit(m2)
        assert word in V.native_served_positions_record(m2, row=r, gen_root=root)["problem"], word


def test_a_native_leg_whose_lens_named_stamp_is_the_neutral_maps_value_is_admitted_and_held_to_it(tmp_path):
    """The FIXED producer's record: the driver's standup lookup reads the routing table's ENGINE-NEUTRAL `served_positions`
    map first (`served_capability_run._standup_served_positions`) and writes the value into the LENS-named field on every
    engine (`served_capability_run.py` 6315-6316) — so a natively served fitted leg carries `route.lens_served_positions =
    ['all']` (and the top-level copy) beside `transfer_provenance.served_positions = ['all']`, no null anywhere (the pulled
    qwen35_4b `arditi_allpos` legs). The reader read the stamp as a lens fact and refused (A3), and — resolving the native
    standup record on the producer's NULL only — never opened the routing table for it. Now: the stamped leg resolves the
    record too (`native_served_positions_stamped`), the record admits the stamp iff it IS the neutral map's value for the
    cell, and the certification holds the stamp to that record; a differing stamp, a table with no neutral entry (a routes[]
    attestation admits no stamp), and no record at all refuse by name; the lens legs' reading is untouched."""
    harness = row(*NATIVE_TRIPLE)["harness_cell"]

    def stamp(value):
        def edit(m):
            m[ROUTE][C.LENS_SERVED_POSITIONS_KEY] = list(value)
            m[C.LENS_SERVED_POSITIONS_KEY] = list(value)
        return edit

    root = tmp_path / "admitted"
    r, cd, manifest, arm, fact = _native_gap_store(root, served_positions=["all"], position_mode="all",
                                                   served_positions_map={harness: ["all"]}, manifest_edit=stamp(["all"]))
    assert manifest[ROUTE][C.LENS_SERVED_POSITIONS_KEY] == ["all"] == manifest[C.LENS_SERVED_POSITIONS_KEY] == manifest[TP]["served_positions"]
    assert V.native_served_positions_gap(manifest) is False and V.native_served_positions_stamped(manifest) is True
    rec = V.native_served_positions_record(manifest, row=r, gen_root=root)
    assert rec["problem"] is None, rec["problem"]
    assert rec["resolved"] == ["all"] and rec["source"] == V.NATIVE_RESOLVED_FROM_NEUTRAL_MAP and rec["gaps"] == []
    assert _native_identity(r, cd, manifest, arm, fact, standup_record=rec) is None
    pr = _producer_row(r, arm=arm)
    ev, _man, _fact = _native_fitted_inputs(pr)
    cert = V.certification_check(ev, manifest, row=pr, fact=fact, standup_record=rec, arm=arm)
    assert cert["failures"] == [] and cert["passed"], cert["failures"]
    assert cert["route_lens_served_positions"] == ["all"] and cert["standup_record"]["source"] == V.NATIVE_RESOLVED_FROM_NEUTRAL_MAP
    # the stamp is never admitted on its own: no record read -> refused, naming the record it needs
    d = " ".join(V.certification_check(ev, manifest, row=pr, fact=fact, standup_record=None, arm=arm)["failures"])
    assert "a served-positions map is a lens fact" in d and "no native standup record was read" in d, d
    # (a) native, stamp ['prefill'] against the neutral map's ['all']: the record refuses by the two values, and so does the
    # certification through it
    root = tmp_path / "differs"
    r, cd, manifest, arm, fact = _native_gap_store(root, served_positions=["all"], position_mode="all",
                                                   served_positions_map={harness: ["all"]}, manifest_edit=stamp(["prefill"]))
    rec = V.native_served_positions_record(manifest, row=r, gen_root=root)
    assert rec["problem"] and "the stamp ['prefill'] != routing_table.json[served_positions] ['all']" in rec["problem"], rec["problem"]
    assert "a served-positions map is a lens fact" in rec["problem"]
    d = " ".join(V.certification_check(ev, manifest, row=pr, fact=fact, standup_record=rec, arm=arm)["failures"])
    assert "route.lens_served_positions = ['prefill'] on a natively served cell" in d and "does not resolve it" in d and "['prefill'] !=" in d, d
    # (b) native, the stamp ['all'] but the routing table carries NO neutral entry (the routes[] attestation alone resolves
    # the positions for the identity check; it admits no stamp)
    root = tmp_path / "noneutral"
    r, cd, manifest, arm, fact = _native_gap_store(root, served_positions=["all"], position_mode="all",
                                                   served_positions_map=None, manifest_edit=stamp(["all"]))
    rec = V.native_served_positions_record(manifest, row=r, gen_root=root)
    assert rec["problem"] and "carries no served_positions entry for the cell" in rec["problem"] and "admits no stamp" in rec["problem"], rec["problem"]
    d = " ".join(V.certification_check(ev, manifest, row=pr, fact=fact, standup_record=rec, arm=arm)["failures"])
    assert "a served-positions map is a lens fact" in d and "carries no served_positions entry" in d, d
    # ... and a record that resolved from the routes[] attestation (no problem of its own) still admits no stamp
    routes_only = {**rec, "problem": None, "resolved": ["all"], "source": V.NATIVE_RESOLVED_FROM_ROUTES}
    d = " ".join(V.certification_check(ev, manifest, row=pr, fact=fact, standup_record=routes_only, arm=arm)["failures"])
    assert "not the served_positions map" in d and "only that map admits a stamp" in d, d
    # the producer's NULL beside the neutral map reads exactly as before (the gap path: resolved, gaps filed, null stamp kept)
    root = tmp_path / "gap"
    r, cd, manifest, arm, fact = _native_gap_store(root, served_positions_map={harness: ["all"]})
    assert V.native_served_positions_stamped(manifest) is False and V.native_served_positions_gap(manifest) is True
    rec = V.native_served_positions_record(manifest, row=r, gen_root=root)
    assert rec["problem"] is None and len(rec["gaps"]) == 3
    cert = V.certification_check(ev, manifest, row=pr, fact=fact, standup_record=rec, arm=arm)
    assert cert["passed"] and cert["route_lens_served_positions"] is None
    # a LENS leg is not this case: the predicate is false on the lens route, and its rule reads byte for byte as before
    lens_r = {**pr, "expected_stamps": list(V.expected_stamps_for(fitted=True, declared=False))}
    lens_man = {C.MANIFEST_KEY_ROUTE: {"virtual_model": C.cell_spellings(CONST_LORA, declared=False).vmodel, C.ROUTE_KEY_SERVE_DECLARATIONS: {},
                                       C.LENS_SERVED_POSITIONS_KEY: ["all"], "serving_identity": "proxy_vllm+lens+interleaved",
                                       C.REDUCTION_MODE_FIELD: "per_row"}, C.LENS_SERVED_POSITIONS_KEY: ["all"]}
    assert V.native_served_positions_stamped(lens_man) is False
    lens_ev = {G.S.EVIDENCE_RENDER_CERTIFIED: G.S.render_certified_record(fact["family"], fact),
               G.S.EVIDENCE_RENDER_PROBE: G.S.render_probe_record(fact), C.OOS_EVIDENCE_BITES_BLOCK: G.S.steering_bites_record(CONST_LORA),
               M.EVIDENCE_LENS_POSITION: G.lens_position_record(pr), M.EVIDENCE_LENS_ON_BITE: G.lens_on_bite_record(pr, fact)}
    cert = V.certification_check(lens_ev, lens_man, row=lens_r, fact=fact)
    assert cert["failures"] == [] and cert["route_lens_served_positions"] == ["all"]
    off = {**lens_man, C.MANIFEST_KEY_ROUTE: {**lens_man[C.MANIFEST_KEY_ROUTE], C.LENS_SERVED_POSITIONS_KEY: ["response"]}}
    d = " ".join(V.certification_check(lens_ev, off, row=lens_r, fact=fact)["failures"])
    assert "route.lens_served_positions = ['response'] != ['all']" in d, d


def test_native_gap_ii_the_same_card_without_the_standup_record_refuses_by_name(tmp_path):
    """(ii) the native card with the producer's null and NO pulled standup record refuses `STANDUP_RECORD_MISSING` (the
    identity check names the record it needs); so does a record directory lacking the native attestation (a pull that
    brought the lens trio only), and a manifest naming no routing table or another layout."""
    root = tmp_path / "none"
    r, cd, manifest, arm, fact = _native_gap_store(root, record=False)
    rec = V.native_served_positions_record(manifest, row=r, gen_root=root)
    assert rec["problem"].startswith(V.STANDUP_RECORD_MISSING) and rec["resolved"] is None and rec["gaps"] == []
    assert V.STANDUP_SERVING_ATTESTATION in rec["problem"] and V.STANDUP_REGISTRY not in rec["problem"]       # the NATIVE file set is named
    problem = _native_identity(r, cd, manifest, arm, fact, standup_record=rec)
    assert problem and V.STANDUP_RECORD_MISSING in problem and "None != ['all']" in problem and "natively served UNDECLARED" in problem
    assert "does not resolve it" in problem
    root = tmp_path / "partial"
    r, cd, manifest, arm, fact = _native_gap_store(root)
    (root / V.STANDUP_DIR_PARENT / G.standup_jobdir(r, G.CAP_BASE) / V.STANDUP_TASK_SUBDIR / V.STANDUP_SERVING_ATTESTATION).unlink()
    rec = V.native_served_positions_record(manifest, row=r, gen_root=root)
    assert V.STANDUP_RECORD_MISSING in rec["problem"] and "lacks" in rec["problem"] and V.STANDUP_SERVING_ATTESTATION in rec["problem"]
    root = tmp_path / "noroute"
    r, cd, manifest, arm, fact = _native_gap_store(root, manifest_edit=lambda m: m.pop("routing_table"))
    assert "names no routing_table" in V.native_served_positions_record(manifest, row=r, gen_root=root)["problem"]
    root = tmp_path / "layout"
    r, cd, manifest, arm, fact = _native_gap_store(root, manifest_edit=lambda m: m.update(routing_table="/elsewhere/routing_table.json"))
    assert "not a standup record path" in V.native_served_positions_record(manifest, row=r, gen_root=root)["problem"]


def test_native_gap_iii_a_record_attesting_another_mode_none_or_a_broken_binding_refuses(tmp_path):
    """(iii) a record whose attested mode is `response` refuses twice by name (not the row's `all`; header-located on an
    undeclared row — declared iff header-located, on the RESOLVED fact); so does the neutral map saying so, the two records
    disagreeing, a native-vocabulary mode the contract cannot classify (`prefill_all` reads located, fail closed), a route
    attesting NO mode under either key (`NATIVE_POSITION_MODE_UNATTESTED` — never a default), and every broken binding: the
    attestation's kind / job, the standup's job / tree / engine, the routes entry's hash / vmodel / identity / multiplicity,
    a lens map naming the cell, and the manifest's own native facts (a located top-level header, a stated map, a block
    pairing that is not the fail-closed derivation from the null). A torn native route is not this gap at all: it refuses
    on the header rule exactly as before."""
    harness = row(*NATIVE_TRIPLE)["harness_cell"]
    modes = {
        "response": (dict(position_mode="response"), ("resolves ['response'] != ['all']", "header_located=True", V.STANDUP_RECORD_MISMATCH)),
        "neutral_response": (dict(position_mode=None, served_positions_map={harness: "response"}), ("resolves ['response'] != ['all']", "header_located=True")),
        "disagree": (dict(position_mode="response", served_positions_map={harness: "all"}), ("disagree", V.ROUTE_POSITION_MODE_KEY)),
        "unattested": (dict(position_mode=None), (V.NATIVE_POSITION_MODE_UNATTESTED, "attests no position mode", "never defaulted")),
        "native_vocabulary": (dict(position_mode="prefill_all"), (f"{V.NATIVE_POSITION_MODE_UNCLASSIFIED}: 'prefill_all'", "resolves ['prefill_all'] != ['all']")),
        "native_vocabulary_neutral": (dict(position_mode=None, served_positions_map={harness: ["prompt_last"]}), (f"{V.NATIVE_POSITION_MODE_UNCLASSIFIED}: 'prompt_last'",)),
        "bad_neutral_value": (dict(position_mode=None, served_positions_map={harness: 7}), ("is not a served-positions value", V.NATIVE_POSITION_MODE_UNATTESTED)),
    }
    for name, (kw, words) in modes.items():
        root = tmp_path / name
        r, cd, manifest, arm, fact = _native_gap_store(root, **kw)
        rec = V.native_served_positions_record(manifest, row=r, gen_root=root)
        assert rec["problem"] and rec["resolved"] is None and rec["gaps"] == [], (name, rec)
        for w in words:
            assert w in rec["problem"], (name, w, rec["problem"])
        problem = _native_identity(r, cd, manifest, arm, fact, standup_record=rec)
        assert problem and "does not resolve it" in problem and words[0] in problem, (name, problem)
    RT, SA, SJ = G.STANDUP_ROUTING_TABLE, G.STANDUP_SERVING_ATTESTATION, G.STANDUP_JOBS
    bindings = {
        "attestation_kind": (lambda d: d[SA].update(kind="lens"), "kind"),
        "attestation_job": (lambda d: d[SA].update(slurm_job_id="1"), "slurm_job_id"),
        "standup_job": (lambda d: d[SJ][0].update(job_id="1"), "job_id"),
        "tree": (lambda d: d[SJ][0].update(sha="1" * 40), "tree_sha"),
        "standup_engine": (lambda d: d[SJ][0].update(engine="lens"), ".engine"),
        "routes_hash": (lambda d: d[RT]["routes"][1].update(intervention_hash="0" * 16), "intervention_hash"),
        "routes_vmodel": (lambda d: d[RT]["routes"][1].update(virtual_model="petri__other"), "virtual_model"),
        "routes_identity": (lambda d: d[RT]["routes"][1].update(serving_identity="proxy_vllm+lens+interleaved"), "serving_identity"),
        "routes_twice": (lambda d: d[RT]["routes"].append(dict(d[RT]["routes"][1])), "2 time(s)"),
        "lens_map_names_cell": (lambda d: d[RT][C.LENS_SERVED_POSITIONS_KEY].update({harness: ["all"]}), "lens map naming"),
    }
    for name, (edit, word) in bindings.items():
        root = tmp_path / f"b_{name}"
        r, cd, manifest, arm, fact = _native_gap_store(root, record_edit=edit)
        rec = V.native_served_positions_record(manifest, row=r, gen_root=root)
        assert rec["problem"] and V.STANDUP_RECORD_MISMATCH in rec["problem"] and word in rec["problem"] and rec["resolved"] is None, (name, rec["problem"])
        assert "does not resolve it" in _native_identity(r, cd, manifest, arm, fact, standup_record=rec)
    manifest_cases = {
        "top_header_located": (lambda m: m.update(header_located=True), "manifest header_located"),
        "route_map_stated": (lambda m: m[ROUTE].update({C.LENS_SERVED_POSITIONS_KEY: ["all"]}), "a served-positions map is a lens fact"),
        "top_map_stated": (lambda m: m.update({C.LENS_SERVED_POSITIONS_KEY: ["all"]}), "a served-positions map is a lens fact"),
        "tp_header_located_false": (lambda m: m[TP].update(header_located=False), "not the producer's fail-closed derivation"),
        "tp_class_tampered": (lambda m: m[TP].update(header_check_class=C.HEADER_CHECK_DOCUMENT_FIT_INAPPLICABLE), "not the producer's fail-closed derivation"),
    }
    for name, (edit, word) in manifest_cases.items():
        root = tmp_path / f"m_{name}"
        r, cd, manifest, arm, fact = _native_gap_store(root, manifest_edit=edit)
        rec = V.native_served_positions_record(manifest, row=r, gen_root=root)
        assert rec["problem"] and word in rec["problem"] and rec["resolved"] is None, (name, rec["problem"])
        assert _native_identity(r, cd, manifest, arm, fact, standup_record=rec), name
    # the identity check's OWN guard against an unclassifiable resolved mode: a row that EXPECTS the restricted native mode
    # (so the row match cannot refuse first) with a record resolving it refuses UNCLASSIFIED by name, never through the
    # contract's fail-closed header reading; `all` on the same row reads as before
    root = tmp_path / "unclassified_row"
    r, cd, manifest, arm, fact = _native_gap_store(root)
    restricted_row = {**r, "served_positions_expected": "prefill_all"}
    forged = {"resolved": ["prefill_all"], "problem": None, "source": V.NATIVE_RESOLVED_FROM_ROUTES}
    problem = _native_identity(restricted_row, cd, manifest, arm, fact, standup_record=forged)
    assert problem and f"{V.NATIVE_POSITION_MODE_UNCLASSIFIED}: the standup record resolves ['prefill_all']" in problem and "fail closed" in problem
    assert "read header-located on an undeclared cell" not in problem
    assert V.NATIVE_POSITION_MODE_UNCLASSIFIED == C.NATIVE_POSITION_MODE_UNCLASSIFIED == "NATIVE_POSITION_MODE_UNCLASSIFIED"
    # a TORN native route (the `+lora+` identity with no engine stamp) is not the gap: the header rule refuses as before
    root = tmp_path / "torn"
    r, cd, manifest, arm, fact = _native_gap_store(root, manifest_edit=lambda m: m[ROUTE].pop(C.SERVED_ENGINE_FIELD))
    assert V.native_served_positions_gap(manifest) is False
    rec = V.native_served_positions_record(manifest, row=r, gen_root=root)
    assert "not a natively served one" in rec["problem"] and "cannot be bound to an engine" in rec["problem"]
    assert _native_identity(r, cd, manifest, arm, fact, standup_record=None) == f"{TP}.header_located True != False (declared iff header-located)"


def test_native_gap_iv_a_lens_route_with_the_same_null_still_refuses_exactly_as_before(tmp_path):
    """(iv) an UNDECLARED LENS store whose manifest carries the null refuses byte-for-byte as before — the native record is
    a NATIVE route's resolution and is never consulted on a lens route; a lens manifest wearing the native gap's block shape
    (the header located beside the null) refuses on the header rule, as before, whatever record is offered."""
    rows = small_matrix((QWEN, "main", NONE), (QWEN, "main", CONST_PSR))

    def null_it(m):
        m[TP]["served_positions"] = None
        m[ROUTE][C.LENS_SERVED_POSITIONS_KEY] = None
        m[C.LENS_SERVED_POSITIONS_KEY] = None
    G.build_all(tmp_path, rows=rows, overrides={(QWEN, "main", CONST_PSR): {"manifest_edit": null_it}})
    rep = analyze(tmp_path, matrix=rows, overlap=None)
    leg = leg_of(rep, QWEN, "main", CONST_PSR)
    assert leg["refused"] == f"{TP}.served_positions None != ['all']" and "standup_record" not in leg
    assert rep[V.RECORD_GAPS_KEY]["n_legs"] == 0 and rep["gates"]["A0_completeness"]["status"] == A.GATE_FAIL
    r = row(QWEN, "main", CONST_PSR)
    cd, manifest, why = V.read_store(tmp_path / r["out_root"])
    assert manifest is not None and V.native_served_positions_gap(manifest) is False
    fact = G.RENDER_FACTS_ALL[r["family"]]
    manifest[TP].update(header_located=True, header_check_class=C.HEADER_CHECK_DOCUMENT_FIT_SERVED_HEADER_LOCATED)
    assert V.native_served_positions_gap(manifest) is False
    for offered in (None, {"resolved": ["all"], "problem": None, "source": V.NATIVE_RESOLVED_FROM_ROUTES}):
        assert _native_identity(r, cd, manifest, C.THINKING_ON, fact, standup_record=offered) == f"{TP}.header_located True != False (declared iff header-located)"
    assert "not a natively served one" in V.native_served_positions_record(manifest, row=r, gen_root=tmp_path)["problem"]


def test_native_gap_v_the_declared_path_is_untouched(tmp_path):
    """(v) the DECLARED cell's resolution reads exactly as before (its three gaps, its cause, its lens record), the declared
    manifest is not the native gap, and the native resolver refuses a declared row as a caller bug."""
    rows, rep = _declared_wave(tmp_path)
    leg = leg_of(rep, *DECL)
    assert leg["refused"] is None and [g["field"] for g in leg[V.RECORD_GAPS_KEY]] == list(V.RECORD_GAP_FIELDS)
    assert all(g["cause"] == V.RECORD_GAP_CAUSE_BARE_KEY_LOOKUP and g["resolved"] == ["response"] for g in leg[V.RECORD_GAPS_KEY])
    assert leg["standup_record"]["problem"] is None and leg["standup_record"]["source"] is None
    assert leg["certification"]["standup_record"]["source"] is None and leg["certification"]["route_lens_served_positions_source"] == "standup_record"
    gap_findings = [f for f in rep["findings"] if ": RECORD GAP — " in f]
    assert len(gap_findings) == 1 and "null in the manifest" in gap_findings[0] and "re-derived" not in gap_findings[0]
    assert gap_findings[0].endswith(f"({V.RECORD_GAP_CAUSE_BARE_KEY_LOOKUP})")
    r = row(*DECL)
    cd, manifest, _why = V.read_store(tmp_path / r["out_root"])
    assert manifest is not None and V.native_served_positions_gap(manifest) is False
    with pytest.raises(V.AnalysisError, match="not an undeclared fitted row"):
        V.native_served_positions_record(manifest, row=r, gen_root=tmp_path)
    with pytest.raises(V.AnalysisError, match="not a declared fitted row"):
        V.declared_served_positions_record(manifest, row=row(*NATIVE_TRIPLE), gen_root=tmp_path)


def _native_wave(root, *, fixed: bool, **over):
    """The bare leg and ONE natively served fitted leg written by the builder AS THE NATIVE DRIVER FILES IT
    (`serving_engine=lora_native`: the `+lora+` route with the engine stamp, the engine's stamp triple, the bites control
    against the engine's zero cell, predicate (B′)'s inapplicability record, the native standup record beside the job);
    `fixed` simulates the FIXED producer (the stated `["all"]`, the engine-neutral map in the record) — else the record
    gap (the null, the routes[] attestation only). Returns (rows, report)."""
    rows = small_matrix((QWEN, "main", NONE), NATIVE_TRIPLE)
    G.build_all(root, rows=rows, overrides={NATIVE_TRIPLE: {"serving_engine": C.SERVING_ENGINE_LORA_NATIVE,
                                                            "producer_records_served_positions": fixed, **over}})
    return rows, analyze(root, matrix=rows, overlap=None)


def test_native_gap_end_to_end_a_synthetic_native_all_positions_card_passes_the_reader_with_and_without_the_gap(tmp_path):
    """Through `analyze`, on a native store written by the builder as the native driver files it: (a) the card with the
    producer's null is ADMITTED at A0 (the identity check consumes the record the leg collector resolved through the ONE
    shared predicate), passes the native certification, the engine's stamp triple and the request-evidence inventory
    against the engine's zero cell — every gate but the overlap record this small wave supplies none of — with its
    `standup_record` and the three gaps filed on the leg, in the completeness gate, in the provenance row and in the
    report's `record_gaps` block under the native cause, the findings line naming the null AND the two re-derived fields;
    (b) the FIXED producer's card (the stated `["all"]`, the engine-neutral map in its record) passes the same gates with
    NO gap and no record read; (c) with the record removed the gap card refuses `STANDUP_RECORD_MISSING` and A0 fails; (d)
    a declared native row is a builder-refused matrix defect."""
    k = V.leg_key(*NATIVE_TRIPLE, G.CAP_BASE)
    every_gate_but_overlap = [g for g in A.GATE_NAMES if g != "A10_overlap"]
    # (a) the record gap, resolved
    rows, rep = _native_wave(tmp_path / "gap", fixed=False)
    leg = rep["legs"][k]
    assert leg["refused"] is None and leg["missing"] is None and leg["identity_validated"] is True
    assert leg["identity"][TP]["served_positions"] is None and leg["identity"][TP]["header_located"] is True
    assert leg["arm_contract"]["steered_path"] == M.STEERED_PATH_LORA
    assert leg["standup_record"]["resolved"] == ["all"] and leg["standup_record"]["source"] == V.NATIVE_RESOLVED_FROM_ROUTES
    assert leg["standup_record"]["problem"] is None and leg["standup_record"]["jobdir"].endswith(leg["standup_record"]["job_id"])
    cert, stamps, req = leg["certification"], leg["stamps"], leg["request_evidence"]
    assert cert["passed"] and cert[M.EVIDENCE_LENS_POSITION]["status"] == C.NOT_APPLICABLE and cert["standup_record"]["resolved"] == ["all"], cert["failures"]
    assert cert["route_lens_served_positions"] is None and cert[M.EVIDENCE_BITES][C.OOS_EVIDENCE_BITES_TWIN] == C.TWIN_CELL_ID_LORA
    assert stamps["passed"] and stamps["expected"]["resumed_policy"] == C.POLICY_NOT_CONSULTED, stamps
    assert req["passed"], req["failures"]
    gaps = leg[V.RECORD_GAPS_KEY]
    assert [g["field"] for g in gaps] == list(V.NATIVE_RECORD_GAP_FIELDS) and all(g["cause"] == V.RECORD_GAP_CAUSE_NATIVE_LENS_MAP_ONLY for g in gaps)
    assert all(rep["gates"][g]["status"] == A.GATE_PASS for g in every_gate_but_overlap), {g: rep["gates"][g]["status"] for g in A.GATE_NAMES}
    assert rep["unusable_legs"] == {} and rep["gates"]["A0_completeness"]["legs"][k][V.RECORD_GAPS_KEY] == 3
    # the `none` card (no fitted object) rides the same native wave: never asked for served positions, no record, no gap,
    # and its passthrough route is not the native gap — so it never meets NATIVE_POSITION_MODE_UNATTESTED
    none_leg = rep["legs"][V.leg_key(QWEN, "main", NONE, G.CAP_BASE)]
    assert none_leg["refused"] is None and "standup_record" not in none_leg and V.RECORD_GAPS_KEY not in none_leg
    _cd0, none_manifest, _why0 = V.read_store(tmp_path / "gap" / row(QWEN, "main", NONE)["out_root"])
    assert none_manifest is not None and V.native_served_positions_gap(none_manifest) is False and not none_manifest.get(TP)
    assert C.is_lens_cell(NONE) is False
    assert rep["provenance"]["cells"][k][V.RECORD_GAPS_KEY] == gaps
    block = rep[V.RECORD_GAPS_KEY]
    assert block["n_legs"] == 1 and block["n_gaps"] == 3 and block["legs"] == {k: gaps}
    assert block["by_cause"] == {V.RECORD_GAP_CAUSE_NATIVE_LENS_MAP_ONLY: 3} and block["by_field"] == {f: 1 for f in V.NATIVE_RECORD_GAP_FIELDS}
    assert "NATIVELY served" in block["note"] and V.SERVED_POSITIONS_KEY in block["note"] and V.ROUTE_POSITION_MODE_KEY in block["note"]
    findings = [f for f in rep["findings"] if ": RECORD GAP — " in f]
    assert len(findings) == 1 and findings[0].startswith(f"{k}: RECORD GAP — {TP}.served_positions null in the manifest; ")
    assert f"{TP}.header_located True -> False" in findings[0] and "re-derived from the null" in findings[0]
    assert f"{C.HEADER_CHECK_DOCUMENT_FIT_SERVED_HEADER_LOCATED!r} -> {C.HEADER_CHECK_DOCUMENT_FIT_INAPPLICABLE!r}" in findings[0]
    assert findings[0].endswith(f"({V.RECORD_GAP_CAUSE_NATIVE_LENS_MAP_ONLY})") and leg["standup_record"]["resolved_from"] in findings[0]
    assert A.record_gap_text(gaps) in findings[0]
    # the record on disk is the real record's shape: the routes[] attestation, the empty lens map, no neutral map yet
    table = json.loads((Path(leg["standup_record"]["standup_dir"]) / V.STANDUP_ROUTING_TABLE).read_text())
    assert table[C.LENS_SERVED_POSITIONS_KEY] == {} and V.SERVED_POSITIONS_KEY not in table
    # (b) the FIXED producer: the stated value, the neutral map beside the attestation, no gap, no record read
    rows, rep2 = _native_wave(tmp_path / "fixed", fixed=True)
    leg2 = rep2["legs"][k]
    assert leg2["refused"] is None and leg2["identity_validated"] is True and "standup_record" not in leg2 and V.RECORD_GAPS_KEY not in leg2
    assert leg2["identity"][TP]["served_positions"] == ["all"] and leg2["identity"][TP]["header_located"] is False
    assert leg2["identity"][TP]["header_check_class"] == C.HEADER_CHECK_DOCUMENT_FIT_INAPPLICABLE
    assert leg2["certification"]["passed"] and leg2["stamps"]["passed"] and leg2["request_evidence"]["passed"]
    assert all(rep2["gates"][g]["status"] == A.GATE_PASS for g in every_gate_but_overlap) and rep2[V.RECORD_GAPS_KEY]["n_legs"] == 0
    cd2, manifest2, _ = V.read_store(tmp_path / "fixed" / row(*NATIVE_TRIPLE)["out_root"])
    table2 = json.loads((Path(leg2["gen_root"]) / V.STANDUP_DIR_PARENT / G.standup_jobdir(row(*NATIVE_TRIPLE), G.CAP_BASE)
                         / V.STANDUP_TASK_SUBDIR / V.STANDUP_ROUTING_TABLE).read_text())
    assert table2[V.SERVED_POSITIONS_KEY] == {row(*NATIVE_TRIPLE)["harness_cell"]: ["all"]} and table2[C.LENS_SERVED_POSITIONS_KEY] == {}
    rec2 = V.native_served_positions_record(manifest2, row=row(*NATIVE_TRIPLE), gen_root=Path(leg2["gen_root"]))
    assert rec2["problem"] is None and rec2["resolved"] == ["all"] and rec2["source"] == V.NATIVE_RESOLVED_FROM_NEUTRAL_MAP and rec2["gaps"] == []
    # (c) the gap card with no pulled record: refused by name, the gate fails, nothing filed
    shutil.rmtree(tmp_path / "gap" / V.STANDUP_DIR_PARENT)
    rep3 = analyze(tmp_path / "gap", matrix=rows, overlap=None)
    leg3 = rep3["legs"][k]
    assert leg3["refused"] and V.STANDUP_RECORD_MISSING in leg3["refused"] and "natively served UNDECLARED" in leg3["refused"]
    assert leg3["standup_record"]["problem"].startswith(V.STANDUP_RECORD_MISSING) and V.RECORD_GAPS_KEY not in leg3
    assert rep3["gates"]["A0_completeness"]["status"] == A.GATE_FAIL and k in rep3["unusable_legs"] and rep3[V.RECORD_GAPS_KEY]["n_legs"] == 0
    # (d) a declared native row is a matrix defect the builder refuses (the reader refuses it too: certification_check)
    with pytest.raises(ValueError, match="DECLARED natively served row"):
        G.build_store(tmp_path / "decl", row(*DECL), correct=range(10), serving_engine=C.SERVING_ENGINE_LORA_NATIVE)
    with pytest.raises(ValueError, match="serving_engine"):
        G.build_store(tmp_path / "eng", row(*NATIVE_TRIPLE), correct=range(10), serving_engine="hf")



def test_a_native_wave_legs_launch_is_attested_through_the_native_record_by_the_builder_and_the_reader(tmp_path):
    """launch-attestation-native, through `analyze` on the native wave the builder writes AS THE NATIVE DRIVER FILES IT: the
    natively served fitted leg's standup dir carries the STRICT `serving_attestation.json` beside `vllm_server.log` (the
    native echo) and NO `steering_attestation.json`; the leg collector's launch attestation selects the native record off the
    leg's `route.serving_engine`, reads it strictly and attests — `engine: lora_native`, bound by job id and port, the native
    flags projected — while the `none` leg beside it attests through the lens record. Known-bad: the same wave with no launch
    record written reads `absent` naming the NATIVE record (never the lens one), and a native leg whose standup dir ALSO
    carries a lens record refuses as torn."""
    k = V.leg_key(*NATIVE_TRIPLE, G.CAP_BASE)
    rows, rep = _native_wave(tmp_path / "wave", fixed=True)
    leg, none_leg = rep["legs"][k], rep["legs"][V.leg_key(QWEN, "main", NONE, G.CAP_BASE)]
    assert leg["refused"] is None and rep["unusable_legs"] == {}
    launch = leg["launch"]
    assert launch["status"] == M.LAUNCH_ATTESTED and launch["problems"] == [], launch["problems"]
    assert launch["engine"] == C.SERVING_ENGINE_LORA_NATIVE and launch["record_file"] == M.LAUNCH_NATIVE_ATTESTATION_FILE
    assert launch["job_id"] == G.standup_job_id(row(*NATIVE_TRIPLE), G.CAP_BASE) and launch["geometry_reconciled"] is True
    d = Path(launch["standup_dir"])
    assert sorted(p.name for p in d.iterdir()) == sorted([V.STANDUP_ROUTING_TABLE, V.STANDUP_JOBS, V.STANDUP_SERVING_ATTESTATION, M.LAUNCH_LOG_FILE])
    record = SA.read_serving_attestation(d / V.STANDUP_SERVING_ATTESTATION)
    assert record["slurm_job_id"] == launch["job_id"] and record["argv_sha256"] == launch["argv_sha256"] and json.loads(launch["excluded"]["port"]["value"]) == record["port"]
    assert json.loads(launch["settings"]["enable_lora"]) is True and json.loads(launch["settings"]["max_lora_rank"]) == 1 and "enforce_eager" not in launch["settings"]
    # O106: the cell-set plumbing sits under `excluded` with the ruling's reason, the kernel configuration under `settings`
    assert json.loads(launch["excluded"]["max_loras"]["value"]) == 2 and "O106" in launch["excluded"]["max_loras"]["why"] and "max_loras" not in launch["settings"]
    modules = json.loads(launch["excluded"]["lora_modules"]["value"])
    assert [m["LoRAModulePath"]["name"] for m in modules] == [C.TWIN_CELL_ID_LORA, row(*NATIVE_TRIPLE)["harness_cell"]]
    assert [m["LoRAModulePath"]["path"] for m in modules] == [record["adapters"][n]["dir"] for n in (C.TWIN_CELL_ID_LORA, row(*NATIVE_TRIPLE)["harness_cell"])]
    assert json.loads(launch["excluded"]["worker_extension_cls"]["value"]) == G.S.NATIVE_WORKER_EXTENSION
    # the log's echo is the real log's shape: NOT a literal (the LoRAModulePath reprs), read by the strict echo reader only
    text = (d / M.LAUNCH_LOG_FILE).read_text()
    assert "LoRAModulePath(name=" in text and M.launch_echo(text) is not None
    assert none_leg["launch"]["status"] == M.LAUNCH_ATTESTED and none_leg["launch"]["engine"] == C.SERVING_ENGINE_LENS
    assert none_leg["launch"]["record_file"] == M.LAUNCH_ATTESTATION_FILE and (Path(none_leg["launch"]["standup_dir"]) / M.LAUNCH_ATTESTATION_FILE).is_file()
    # known-bad (i): no launch record written for the native leg — absent, the NATIVE record named as missing with the remedy
    rows, rep2 = _native_wave(tmp_path / "nolaunch", fixed=True, launch_record=False, standup_record=False)
    launch2 = rep2["legs"][k]["launch"]
    assert launch2["status"] == M.LAUNCH_ABSENT and launch2["engine"] == C.SERVING_ENGINE_LORA_NATIVE and launch2["record_file"] == "serving_attestation.json"
    assert len(launch2["problems"]) == 1 and "no standup record" in launch2["problems"][0] and "pull the job's serving/" in launch2["problems"][0]
    # known-bad (ii): a lens record ALSO beside the native one — torn, refused by name although the native record is strict-valid
    (d / V.STANDUP_ATTESTATION).write_text(json.dumps({"slurm_job_id": launch["job_id"], "port": record["port"]}))
    rep3 = analyze(tmp_path / "wave", matrix=rows, overlap=None)
    launch3 = rep3["legs"][k]["launch"]
    assert launch3["status"] == M.LAUNCH_ABSENT and len(launch3["problems"]) == 1
    assert "the lens engine's launch record beside a lora_native leg's standup" in launch3["problems"][0] and "torn / mixed" in launch3["problems"][0]

def test_the_native_card_fixture_mirrors_the_real_record_shape_and_resolves(tmp_path):
    """The synthetic fixture under tests/fixtures/gsm8k_native_card mirrors — key for key — a real natively served card's
    manifest fields and its standup record (`routing_table.json` v2 with the empty lens map and the routes[] attestation,
    `standup_jobs.jsonl`, `serving_attestation.json`): the resolver reads it as the real record, the builder's documents
    carry the same key sets, and the fixture flipped to another attested mode refuses."""
    manifest = json.loads((NATIVE_FIXTURE / "manifest_excerpt.json").read_text())
    r = json.loads((NATIVE_FIXTURE / "row.json").read_text())
    jobdir = "g8k_llama8b_constv2-mix-full--psr_ce_all_8192_500001"
    shutil.copytree(NATIVE_FIXTURE / V.STANDUP_DIR_PARENT, tmp_path / V.STANDUP_DIR_PARENT)
    d = tmp_path / V.STANDUP_DIR_PARENT / jobdir / V.STANDUP_TASK_SUBDIR
    assert sorted(p.name for p in d.iterdir()) == sorted(V.NATIVE_STANDUP_RECORD_FILES)
    table = json.loads((d / V.STANDUP_ROUTING_TABLE).read_text())
    assert table["version"] == 2 and table[C.LENS_SERVED_POSITIONS_KEY] == {} and V.SERVED_POSITIONS_KEY not in table
    assert table["routes"][1]["cell_id"] == r["harness_cell"] and table["routes"][1][V.ROUTE_POSITION_MODE_KEY] == "all"
    assert V.ROUTE_POSITION_MODE_KEY not in table["routes"][0] and V.ROUTE_POSITION_MODE_KEY not in table["routes"][2]
    assert manifest["header_located"] is False and manifest[TP]["served_positions"] is None and manifest[TP]["header_located"] is True
    assert V.native_served_positions_gap(manifest) is True
    rec = V.native_served_positions_record(manifest, row=r, gen_root=tmp_path)
    assert rec["problem"] is None, rec["problem"]
    assert rec["resolved"] == ["all"] and rec["source"] == V.NATIVE_RESOLVED_FROM_ROUTES and rec["job_id"] == "500001" and rec["jobdir"] == jobdir
    assert [(g["field"], g["manifest"], g["resolved"]) for g in rec["gaps"]] == [
        (V.RECORD_GAP_FIELD_TRANSFER, None, ["all"]), (V.RECORD_GAP_FIELD_TRANSFER_HEADER_LOCATED, True, False),
        (V.RECORD_GAP_FIELD_TRANSFER_HEADER_CHECK_CLASS, C.HEADER_CHECK_DOCUMENT_FIT_SERVED_HEADER_LOCATED, C.HEADER_CHECK_DOCUMENT_FIT_INAPPLICABLE)]
    # the builder writes the same shapes (the fixture is the real record's key set; the builder is what the tests build with)
    docs = G.native_standup_record_docs(row(*NATIVE_TRIPLE), G.RENDER_FACTS_ALL["qwen3_5"], cap=G.CAP_BASE, tree_sha=G.FULL_SHA,
                                        intervention_hash="0" * 16, max_model_len=G.MAX_MODEL_LEN_BASE)
    assert set(docs[G.STANDUP_ROUTING_TABLE]) == set(table)
    assert [set(x) for x in docs[G.STANDUP_ROUTING_TABLE]["routes"]] == [set(x) for x in table["routes"]]
    assert set(docs[G.STANDUP_ROUTING_TABLE]["adapters"][C.TWIN_CELL_ID_LORA]) == set(table["adapters"][C.TWIN_CELL_ID_LORA])
    att = json.loads((d / V.STANDUP_SERVING_ATTESTATION).read_text())
    assert set(docs[G.STANDUP_SERVING_ATTESTATION]) == set(att) and att["kind"] == C.SERVING_ENGINE_LORA_NATIVE
    # launch-attestation-native: the fixture's record and the builder's are STRICT-VALID (the launch attestation reads a native
    # leg's record through the strict reader) — exactly the schema-2 key set, admitted by the reader; a trimmed record refuses
    assert set(att) == set(SA.REQUIRED_TOP_KEYS_V2) and SA.read_serving_attestation(d / V.STANDUP_SERVING_ATTESTATION)["slurm_job_id"] == "500001"
    (tmp_path / "builder_att.json").write_text(json.dumps(docs[G.STANDUP_SERVING_ATTESTATION]))
    assert SA.read_serving_attestation(tmp_path / "builder_att.json")["id_name_map"] == docs[G.STANDUP_SERVING_ATTESTATION]["id_name_map"]
    trimmed = {k: att[k] for k in ("schema_version", "kind", "slurm_job_id", "port", "argv_sha256", "id_name_map", "adapters", "expected")}
    (tmp_path / "trimmed_att.json").write_text(json.dumps(trimmed))
    with pytest.raises(SA.ServingAttestationError, match="lacks required keys"):
        SA.read_serving_attestation(tmp_path / "trimmed_att.json")
    jobs = [json.loads(line) for line in (d / V.STANDUP_JOBS).read_text().splitlines() if line.strip()]
    bound = {"job_id", "sha", "engine", "serving_path", "node", "key", "view", "max_num_seqs", "max_model_len"}      # the keys the reader binds, and their neighbours
    assert len(jobs) == 1 and bound <= set(jobs[0]) and bound <= set(docs[G.STANDUP_JOBS][0]) and jobs[0]["engine"] == C.SERVING_ENGINE_LORA_NATIVE
    # known-bad on the fixture: the attestation flipped to a header-located mode, and the engine-neutral map disagreeing
    table["routes"][1][V.ROUTE_POSITION_MODE_KEY] = "response"
    (d / V.STANDUP_ROUTING_TABLE).write_text(json.dumps(table))
    problem = V.native_served_positions_record(manifest, row=r, gen_root=tmp_path)["problem"]
    assert "resolves ['response'] != ['all']" in problem and "header_located=True" in problem
    table["routes"][1][V.ROUTE_POSITION_MODE_KEY] = "all"
    table[V.SERVED_POSITIONS_KEY] = {r["harness_cell"]: "response"}
    (d / V.STANDUP_ROUTING_TABLE).write_text(json.dumps(table))
    assert "disagree" in V.native_served_positions_record(manifest, row=r, gen_root=tmp_path)["problem"]
