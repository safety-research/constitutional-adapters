"""The MATH-500 analysis' eight-arm extension (math-eight-arms-extension plan P5 / §4.5 / §4.8; lane B B2–B13).

Everything runs on SYNTHETIC stores (`tests/math500_synthetic_store.py`: placeholder shas, invented ids, `TESTHOST`) —
never a recorded leg. The C4 fixture is the 24 published rows (`complete`, tree T1 under one pull root) + the nine
extension rows (`launch`, wave `ext1`, tree T2 under a second, wave-isolated pull root pinned to T2) + the six
steering stubs (`deferred`), analysed with `--waves published,ext1` under a declared tree set.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

import math500_synthetic_store as S
import pytest

from sisteer.capeval.serving import serving_attestation as SA
from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("tinycap_math500_thinking", REPO / "scripts" / "tinycap_math500_thinking.py")
M = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = M
_SPEC.loader.exec_module(M)
PG = M.PG

#: THE DECLARED PANEL ROSTER's kinds — what a published design lays out, NEVER `M.ARM_KINDS` (the live registry). Registering a
#: further arm (arditi-allpos-cap's `arditi_allpos`, which has its own matrix and its own export) must not widen a panel whose
#: matrix does not carry it, so these assertions are held to the roster; an analysis whose rows DO carry a further arm covers it
#: too (M.design_arm_roster), and a test of such a matrix must say so explicitly rather than reach for the registry.
PANEL_KINDS: list[str] = [a.kind for a in M.PUBLISHED_ARMS]

B_FAST = 400
GEMMA, QWEN, NEMO, GLM = "gemma4e4b", "qwen35_4b", "nemotron3nano4b", "glm46vflash"
PRIMARY = S.MODE_PRIMARY
BARE, PROMPTED, OBJECT = S.CELL_BARE, S.CELL_PROMPTED, S.CELL_OBJECT
CONST, CTL2, MIX = S.CELL_PROMPTED_CONST, S.CELL_LORA_CTL2, S.CELL_LORA_MIX
T1 = S.FULL_SHA                     # the published tree (the store pin S.SHA7 is its 7-char prefix)
T2 = "a" * 40                       # the extension tree: the ext1 legs' producer
T3 = "b" * 40                       # a tree nobody declared
PROBE_FILES = ("scripts/served_capability_run.py", "scripts/sdfbase/math500/cells.py")
WHY = "the nine W1 legs were produced at the extension's tree; the 24 published legs stay at the published tree (reused, never re-run)"
WAVES = ["published", "ext1"]


def _fake_git(monkeypatch, *, known, diffs):
    monkeypatch.setattr(M, "_git_output", S.fake_git_for(known, diffs, diff_args=M.GIT_DIFF_NAMES_ARGS))


def _tree_set(monkeypatch, tmp_path, also=(T2,)):
    _fake_git(monkeypatch, known={T1, T2}, diffs={(T1, t): list(PROBE_FILES) for t in also})
    ts = M.verify_tree_set(S.SHA7, list(also), list(PROBE_FILES), repo_root=tmp_path, reason=WHY)
    assert ts["problems"] == []
    return ts


def _build_c4(root: Path, *, skip=(), ext_overrides=None, pub_overrides=None, steer_status=S.STATUS_DEFERRED):
    """The C4 stores: the published rows under root/pub (tree T1), the `ext1` rows under root/ext1 (tree T2, wave-isolated
    `controls_ext1/` roots). Returns (matrix rows, gen-root specs)."""
    pub, ext = S.design_rows(), S.ext_rows()
    S.build_all(root / "pub", rows=pub, overrides=pub_overrides)
    S.build_all(root / "ext1", rows=[r for r in ext if (r["model_key"], r["cell"]) not in set(skip)], tree_sha=T2, overrides=ext_overrides)
    rows = pub + ext + S.steer_rows(len(pub) + len(ext), status=steer_status)
    return rows, [f"{root / 'pub'}={S.SHA7}", f"{root / 'ext1'}={T2}"]


def _analyze(specs, rows, ts, **kw):
    kw.setdefault("waves", WAVES)
    return M.analyze(specs, S.SHA7, cells=rows, sampling_rows=S.SAMPLING_ROWS, render_facts=S.RENDER_FACTS, math500_ids=S.IDS,
                     n_resamples=B_FAST, read_eval_logs=False, analysis_sha=T1, tree_set=ts, repo_root=REPO, **kw)


def comparison(report, model_key, mode=PRIMARY):
    return next(c for c in report["comparisons"] if c["model_key"] == model_key and c["mode"] == mode)


@pytest.fixture(scope="module")
def c4(tmp_path_factory):
    root = tmp_path_factory.mktemp("c4")
    rows, specs = _build_c4(root)
    return {"root": root, "rows": rows, "specs": specs}


@pytest.fixture(scope="module")
def c4_report(c4):
    mp = pytest.MonkeyPatch()
    try:
        ts = _tree_set(mp, c4["root"])
        return _analyze(c4["specs"], c4["rows"], ts)
    finally:
        mp.undo()


# --------------------------------------------------------------------------- #
# the registry, the statuses, the roots                                        #
# --------------------------------------------------------------------------- #


def test_the_arm_registry_keeps_the_published_names_and_orders_the_eight_arms():
    # the published eight in their published order; the all-positions Arditi arm (arditi-allpos-cap) is APPENDED after them
    assert [a.cell for a in M.PUBLISHED_ARMS] == [BARE, PROMPTED, CONST, CTL2, MIX, OBJECT, "alphasteer", "arditi"]
    assert [a.cell for a in M.ARMS] == [*(a.cell for a in M.PUBLISHED_ARMS), "arditi_allpos"] and M.ARMS[:8] == M.PUBLISHED_ARMS
    allpos = M.ARM_BY_CELL["arditi_allpos"]
    assert allpos.kind == M.KIND_ARDITI_ALLPOS == "arditi_allpos" and allpos.external and allpos.view_keyed
    assert not any(a.view_keyed for a in M.PUBLISHED_ARMS) and M.fit_kind_expected("arditi_allpos") == M.FIT_KIND_EXPECTED_EXTERNAL
    assert M.KINDS == ("bare", "prompted", "object") and M.DESIGN_CELLS == (BARE, PROMPTED, OBJECT)
    assert M.contrasts_for(list(M.KINDS)) == dict(M.CONTRASTS) and list(M.contrasts_for(list(M.KINDS))) == ["object_minus_bare", "prompted_minus_bare"]
    wide = M.contrasts_for(["bare", "object", "lora_mix", "prompted_constitution", "prompted"])
    assert list(wide) == ["prompted_minus_bare", "prompted_constitution_minus_bare", "lora_mix_minus_bare", "object_minus_bare"]
    assert wide["lora_mix_minus_bare"] == {"lora_mix": +1, "bare": -1}
    with pytest.raises(M.AnalysisError, match="bare"):
        M.contrasts_for(["object", "prompted"])
    assert M.ROW_STATUSES == ("complete", "launch", "held", "deferred") and M.EVIDENCE_STATUSES == ("complete", "launch")
    assert M.PROMPTED_SI_PREFIX_SHA16 == S.SI_PREFIX_SHA16          # the fixture writes what the analysis pins
    for a in M.ARMS:
        assert M.KIND_COLOR[a.kind] and M.KIND_LABEL[a.kind] == a.label
    assert len({M.KIND_COLOR[a.kind] for a in M.ARMS}) == 9, "a fixed hue per arm, never shared"
    assert M.KIND_COLOR["arditi_allpos"] == M.CAT_NINTH_BROWN not in set(M.CAT8.values()) and M.KIND_COLOR["arditi"] == M.CAT8["red"]


def test_the_synthetic_matrix_rows_are_the_launchers_rows():
    """Merge seam: the synthetic store's design rows and extension rows are the LAUNCHER's
    own rows (`scripts/sdfbase/math500/cells.py`) key for key — the published 24 with the extension keys the producer
    stamps (`status: launch`; published-ness is the wave) and rows 24-32 (wave `ext1`) at their matrix indices — so the C4
    fixture cannot drift from the matrix the extension legs launch from. The steering stubs (33-38) have no producer row yet."""
    real = S.M5T_CELLS
    pub = [r for r in real if r["wave"] == S.WAVE_PUBLISHED]
    ext = [r for r in real if r["wave"] == S.WAVE_EXT1]
    steer = [r for r in real if r["wave"] == S.WAVE_STEER]
    assert len(pub) == 24 and [r["index"] for r in ext] == list(range(24, 33))
    assert {r["status"] for r in pub + ext} == {S.STATUS_LAUNCH}
    assert S.cells_matrix() == pub and S.design_rows() == pub
    assert S.ext_rows() == ext
    assert all(r["geometry_L"] == (S.MAX_MODEL_LEN_LONGSI if r["cell"] == CONST else S.MAX_MODEL_LEN) for r in ext)
    # the steering rung is REAL on the launcher now (its admission record landed): eight rows, `launch` on the three switch
    # models and `held` on the no-switch one, every one pinned to the record's external object. The FIXTURE keeps its
    # own DESIGN-ONLY stubs below — the analysis must read a `deferred` row whatever the launcher's current rung is
    # (the GSM8K instrument's steering rows are still deferred), and this file's reader tests are what pin that.
    assert {r["wave"] for r in steer} == {S.WAVE_STEER} and {r["cell"] for r in steer} == set(S.STEER_CELLS)
    assert all(r["object_source"] == "external" and len(r["object_sha256"]) == 64 for r in steer)
    assert {r["status"] for r in steer} <= {S.STATUS_LAUNCH, S.STATUS_HELD}
    matrix = S.eight_arm_matrix()
    assert matrix[:33] == real[:33] and [r["index"] for r in matrix] == list(range(39))
    assert [r["status"] for r in matrix[33:]] == [S.STATUS_DEFERRED] * 6


def test_load_cells_admits_the_design_statuses_and_wave_isolated_roots_and_refuses_by_name():
    rows = M.load_cells(S.eight_arm_matrix())
    assert len(rows) == 39 and [M.row_status(r) for r in rows].count("deferred") == 6
    assert rows[24]["out_root"] == f"controls_ext1/{GEMMA}/prompted--si_constitution/{PRIMARY}" and rows[0]["out_root"].startswith("controls/")
    # the published rows stay byte-identical: no status / wave key is written back
    assert M.load_cells(S.cells_matrix()) == S.cells_matrix()
    assert M.row_status(S.cells_matrix()[0]) == "launch" and M.row_wave(S.cells_matrix()[0]) == "published"
    bad = dict(S.eight_arm_matrix()[24], out_root=f"elsewhere/{GEMMA}/prompted--si_constitution/{PRIMARY}")
    with pytest.raises(M.AnalysisError, match="wave-isolated"):
        M.load_cells([bad])
    with pytest.raises(M.AnalysisError, match="status 'odd'"):
        M.load_cells([dict(S.eight_arm_matrix()[24], status="odd")])
    with pytest.raises(M.AnalysisError, match="not a registered arm"):
        M.load_cells([dict(S.eight_arm_matrix()[24], cell="constv2-mix-full__psr_ce_all")])
    # `arditi` IS a cell this tree's grammar admits (defense_cells.SERVED_ONLY_CLASS_B_IDS — served from the arditi
    # bank, no trainer here), so an EVIDENCE row for it loads WITH its own external pin and refuses only for the
    # reason every external arm refuses: no 64-hex object sha256 on the row.
    stub = next(r for r in S.eight_arm_matrix() if r["cell"] == "arditi")
    assert M.load_cells([stub])[0]["cell"] == "arditi"
    assert C.cell_spellings("arditi", declared=False).harness == "arditi"
    assert M.load_cells([dict(stub, status="launch", expected_stamps=["a", "b", "c"], object_sha256="e" * 64)])[0]["cell"] == "arditi"
    # an alphasteer EVIDENCE row is a legal lens cell (its pin rides the row: an external object, P8)
    astub = dict(next(r for r in S.eight_arm_matrix() if r["cell"] == "alphasteer"), status="launch", expected_stamps=["a", "b", "c"], object_sha256="f" * 64)
    assert M.load_cells([astub])[0]["cell"] == "alphasteer"
    assert M.required_rows(M.load_cells(S.eight_arm_matrix()), ["published"]) == M.load_cells(S.eight_arm_matrix())[:24]
    assert len(M.required_rows(M.load_cells(S.eight_arm_matrix()), None)) == 33


def test_gen_root_specs_and_the_admitted_pins():
    assert M.parse_gen_root_spec("pull_a=abcdef1") == (Path("pull_a"), "abcdef1")
    assert M.parse_gen_root_spec("pull_a") == (Path("pull_a"), None)
    assert M.parse_gen_root_spec("dir=name/x") == (Path("dir=name/x"), None)             # not a hex pin: a path
    assert M.parse_gen_root_spec(f"p={T2}") == (Path("p"), T2)
    roots = M.GenRoots([("pub", None), ("ext", T2)], primary_pin=S.SHA7, admitted=[T2])
    assert roots.declared and roots.primary == Path("pub") and roots.record() == [{"path": "pub", "pin": S.SHA7, "pinned": False},
                                                                                  {"path": "ext", "pin": T2, "pinned": True}]
    assert not M.GenRoots.single("pub", S.SHA7).declared and M.GenRoots([("pub", S.SHA7)], primary_pin=S.SHA7).declared
    with pytest.raises(M.AnalysisError, match="unadmitted producer tree"):
        M.GenRoots([("ext", T3)], primary_pin=S.SHA7, admitted=[T2])
    with pytest.raises(M.AnalysisError, match="named twice"):
        M.GenRoots([("pub", None), ("pub", T2)], primary_pin=S.SHA7, admitted=[T2])
    with pytest.raises(M.AnalysisError, match="at least one"):
        M.GenRoots([], primary_pin=S.SHA7)


def test_the_reviewed_allowlist_file_yields_globs_with_evidence_and_refuses_without(tmp_path):
    md = tmp_path / "tree_set_review_55e3cbd_1234567.md"
    md.write_text("# review\n\nProducer paths:\n- `scripts/served_capability_run.py` — the client-timeout fix; no served identity field moves\n"
                  "- `scripts/sdfbase/**` — launcher only; the served identity is the driver's\n\n| glob | evidence |\n|---|---|\n"
                  "| `tests/**` | tests never run on the cluster |\n\nprose without a glob\n", encoding="utf-8")
    globs, evidence = M.load_tree_diff_allow_file(md)
    assert globs == ["scripts/served_capability_run.py", "scripts/sdfbase/**", "tests/**"]
    assert evidence["tests/**"] == "tests never run on the cluster" and evidence["scripts/sdfbase/**"].startswith("launcher only")
    (tmp_path / "bad.md").write_text("- `scripts/x.py` —\n", encoding="utf-8")
    with pytest.raises(M.AnalysisError, match="carries no evidence"):
        M.load_tree_diff_allow_file(tmp_path / "bad.md")
    (tmp_path / "dup.md").write_text("- `a/**` — one\n- `a/**` — two\n", encoding="utf-8")
    with pytest.raises(M.AnalysisError, match="named twice"):
        M.load_tree_diff_allow_file(tmp_path / "dup.md")
    (tmp_path / "none.md").write_text("no globs here\n", encoding="utf-8")
    with pytest.raises(M.AnalysisError, match="no allowlisted glob"):
        M.load_tree_diff_allow_file(tmp_path / "none.md")
    with pytest.raises(M.AnalysisError, match="no such file"):
        M.load_tree_diff_allow_file(tmp_path / "missing.md")
    # the block files the file and the evidence beside the globs (B7)
    ts = M.verify_tree_set(S.SHA7, [], globs, repo_root=tmp_path, reason=None, allow_file=str(md), allow_evidence=evidence)
    assert ts["allow"] == globs and ts["allow_file"] == str(md) and ts["allow_evidence"] == evidence


def test_object_pins_are_keyed_by_model_and_arm_and_read_under_the_rows_staged_ids():
    rows = M.load_cells(S.eight_arm_matrix())
    pins = M.load_object_pins(REPO / "configs" / "sdfbase_full22_pins.json", rows)
    for key in S.EXT_MODELS:
        three = {pins[(key, OBJECT)], pins[(key, CTL2)], pins[(key, MIX)]}
        assert len(three) == 3, f"{key}: three DISTINCT fitted objects"
        assert pins[(key, OBJECT)] == S.object_pin(key) and pins[(key, MIX)] == S.staged_object_pin(key, f"base__{MIX}")
    assert (GLM, OBJECT) in pins and (GLM, MIX) not in pins           # glm has no extension row
    assert M.staged_id_of_row({"fitted": True, "cell": OBJECT}) == f"derived__{OBJECT}"
    assert M.staged_id_of_row({"fitted": True, "cell": MIX}) == f"base__{MIX}"
    assert M.staged_id_of_row({"fitted": True, "cell": MIX, "staged_id": "custom__id"}) == "custom__id"
    assert M.staged_id_of_row({"fitted": False, "cell": BARE}) is None
    legacy = M.load_object_pins(REPO / "configs" / "sdfbase_full22_pins.json")
    assert legacy[QWEN] == S.object_pin(QWEN) and M.normalise_object_pins(legacy)[(QWEN, OBJECT)] == S.object_pin(QWEN)
    with pytest.raises(M.AnalysisError, match="!= the pin file's"):
        M.load_object_pins(REPO / "configs" / "sdfbase_full22_pins.json", [dict(rows[26], object_sha256="0" * 64)])
    with pytest.raises(M.AnalysisError, match="external-object arm needs the row's own 64-hex object_sha256"):
        M.load_object_pins(REPO / "configs" / "sdfbase_full22_pins.json",
                           [dict(next(r for r in rows if r["cell"] == "alphasteer"), status="launch")])


# --------------------------------------------------------------------------- #
# the C4 fixture: gates, identity per (model, arm), compatibility, design, pooled #
# --------------------------------------------------------------------------- #


def test_the_c4_fixture_passes_every_gate_with_three_distinct_objects_per_model(c4_report):
    rep = c4_report
    assert rep["gates"]["failed"] == [] and rep["gates"]["pending"] == [] and rep["verdict"]["word"] is not None
    assert M.production_refusals(rep) == [] and rep["unusable_legs"] == {}
    assert len(rep["legs"]) == 33 and rep["design"]["n_required"] == 33 and rep["design"]["n_deferred"] == 6 and rep["design"]["n_held"] == 0
    assert rep["design"]["waves"] == WAVES and rep["provenance"]["waves"] == WAVES and rep["design"]["waves_in_matrix"] == ["ext1", "published", "steer"]
    assert [g["pinned"] for g in rep["gen_roots"]] == [True, True] and rep["provenance"]["gen_roots"] == rep["gen_roots"]
    # identity per (model, arm): every fitted leg validated against ITS row's object — three distinct shas per model
    for key in S.EXT_MODELS:
        shas = {rep["legs"][M.leg_key(key, c, PRIMARY)]["identity"]["object_artifact_sha256"] for c in (OBJECT, CTL2, MIX)}
        assert len(shas) == 3 and all(rep["legs"][M.leg_key(key, c, PRIMARY)]["identity_validated"] for c in (OBJECT, CTL2, MIX, CONST))
        assert rep["legs"][M.leg_key(key, CONST, PRIMARY)]["identity"]["system_sha"] == S.SI_PREFIX_SHA16["si_constitution"]
        assert rep["legs"][M.leg_key(key, CONST, PRIMARY)]["identity"]["served"]["max_model_len"] == S.MAX_MODEL_LEN_LONGSI
        # every ext leg comes from the second root, at T2, under the root's pin; the published legs at T1 under the store pin
        ext = rep["legs"][M.leg_key(key, MIX, PRIMARY)]["producer"]
        assert ext["tree_sha"] == T2 and ext["pin"] == T2 and ext["root_pinned"] and ext["root"].endswith("ext1") and ext["matches_pin"]
        pub = rep["legs"][M.leg_key(key, BARE, PRIMARY)]["producer"]
        assert pub["tree_sha"] == T1 and pub["pin"] == S.SHA7 and pub["root"].endswith("pub")
    # the comparisons: every arm PRESENT gets its contrast; the two deferred arms are named absent, never refused
    c = comparison(rep, GEMMA)
    assert c["usable"] and set(c["leg_keys"]) == {"bare", "prompted", "prompted_constitution", "lora_ctl2", "lora_mix", "object"}
    assert c["arms_absent"] == ["alphasteer", "arditi"]
    assert list(c["verdicts"]) == ["prompted_minus_bare", "prompted_constitution_minus_bare", "lora_ctl2_minus_bare", "lora_mix_minus_bare", "object_minus_bare"]
    assert c["compatibility"]["cross_tree"] is True and c["compatibility"]["projection_equal"] and c["compatibility"]["arm_contracts_valid"]
    assert c["compatibility"]["legs"]["lora_mix"]["tree"] == T2 and c["compatibility"]["legs"]["bare"]["tree"] == T1
    glm = comparison(rep, GLM)
    assert glm["usable"] and set(glm["leg_keys"]) == set(M.KINDS) and list(glm["verdicts"]) == ["object_minus_bare", "prompted_minus_bare"]
    assert glm["arms_absent"] == ["prompted_constitution", "lora_ctl2", "lora_mix", "alphasteer", "arditi"]
    # the deferred stubs are statuses in design.status_by_row, never collected
    for key in S.EXT_MODELS:
        for cell in ("alphasteer", "arditi"):
            k = M.leg_key(key, cell, PRIMARY)
            assert k not in rep["legs"] and rep["design"]["status_by_row"][k]["status"] == "deferred"
            assert rep["design"]["status_by_row"][k]["reason"].startswith("deferred: the AlphaSteer / Arditi rung")
    assert rep["design"]["arms_by_model"][GEMMA] == {"bare": "required", "prompted": "required", "prompted_constitution": "required",
                                                     "lora_ctl2": "required", "lora_mix": "required", "object": "required",
                                                     "alphasteer": "deferred", "arditi": "deferred"}
    # the design is the MATRIX's arms: this fixture's matrix carries the published eight, so registering a ninth arm in the
    # module does not add a slot, a status or a pooled record here (panel_slots reads these cells back)
    assert rep["design"]["arms"] == [a.cell for a in M.PUBLISHED_ARMS] and "arditi_allpos" not in rep["design"]["arm_kinds"]
    assert rep["design"]["arms_by_model"][GLM]["lora_mix"] == "absent"
    assert any("DEFERRED" in f for f in rep["findings"])
    # provenance rows carry the identity keys, the producer's pin and root
    row = rep["provenance"]["cells"][M.leg_key(QWEN, MIX, PRIMARY)]
    assert row["object_artifact_sha256"] == S.staged_object_pin(QWEN, f"base__{MIX}") and row["route_kind"] == "lens" and row["view"] == "main"
    assert row["producer_tree_sha"] == T2 and row["pin"] == T2 and row["thinking"] == "on" and row["sampling_mode"] == PRIMARY
    assert row["sampling_ref"] and row["wave"] == "ext1" and row["arm"] == "lora_mix" and row["chat_template_hash"]
    assert rep["provenance"]["trigger"].startswith("MATH-500: one pass") and rep["provenance"]["sampling"][NEMO]


def test_pooled_by_arm_is_a_report_field_with_defined_estimand_and_unavailable_where_a_panel_leg_is_not_there(c4_report):
    ea = c4_report["eightarm"]
    assert ea["panel_models"] == [GEMMA, QWEN, NEMO] and list(ea["held_out"]) == ["llama8b"] and ea["not_in_panel"] == [GLM]
    assert ea["n_available"] == 6 and ea["arms"] == PANEL_KINDS
    for kind in ("bare", "prompted", "prompted_constitution", "lora_ctl2", "lora_mix", "object"):
        rec = ea["pooled_by_arm"][kind]
        assert rec["status"] == "available" and rec["weights"] == {GEMMA: 1 / 3, QWEN: 1 / 3, NEMO: 1 / 3} and rec["n_items"] == 500
        p = rec["primary"]
        assert set(p["by_model"]) == set(ea["panel_models"]) and abs(p["point"] - sum(p["by_model"].values()) / 3) < 1e-12
        assert p["ci_low"] <= p["point"] <= p["ci_high"] and p["n_replicates_used"] + p["n_degenerate_replicates"] == B_FAST
    for kind in ("alphasteer", "arditi"):
        rec = ea["pooled_by_arm"][kind]
        assert rec["status"] == "unavailable" and all(f"{m}/{rec['cell']}: deferred" in rec["reason"] for m in ea["panel_models"])
    # arditi-allpos-cap: the ninth arm has no row in this fixture's matrix, so it is not in this panel at all — no pooled
    # record, no slot, no caption word (asserted in full by the registered-but-unanalysed test below)
    assert "arditi_allpos" not in ea["pooled_by_arm"] and ea["arms"] == [a.kind for a in M.PUBLISHED_ARMS]
    # the fixture's arms: lora_ctl2 (302/500 on every model) reads above bare (300/500); lora_mix (290) below
    assert ea["pooled_by_arm"]["lora_ctl2"]["primary"]["minus_bare"]["point"] > 0 > ea["pooled_by_arm"]["lora_mix"]["primary"]["minus_bare"]["point"]
    assert ea["pooled_by_arm"]["bare"]["primary"]["minus_bare"]["point"] == 0.0
    assert "ONE item-resampling index vector per replicate" in ea["estimand"] and ea["bootstrap"]["B"] == B_FAST


def test_pooled_bootstrap_shares_one_index_vector_per_replicate_across_the_panel(c4_report):
    """The arm's legs set to bare's own: with SHARED draws every replicate's difference is exactly zero; independent draws
    per model would scatter it."""
    bare = {m: c4_report["legs"][M.leg_key(m, BARE, PRIMARY)] for m in M.POOLED_PANEL_MODELS}
    out = M.pooled_arm_bootstrap(bare, bare, n_resamples=B_FAST, seed=0, ci=0.95)
    for branch in M.BRANCHES:
        d = out[branch]["minus_bare"]
        assert d["point"] == 0.0 and d["ci_low"] == 0.0 == d["ci_high"] and d["replicate_signs"]["zero"] == out[branch]["n_replicates_used"]
    again = M.pooled_arm_bootstrap(bare, bare, n_resamples=B_FAST, seed=0, ci=0.95)
    assert again == out
    with pytest.raises(M.AnalysisError, match="same models"):
        M.pooled_arm_bootstrap({GEMMA: bare[GEMMA]}, bare, n_resamples=10, seed=0, ci=0.95)


def test_pooled_means_are_invariant_to_the_sensitivity_rows_and_a_held_panel_model_is_unavailable(c4, c4_report, monkeypatch):
    ts = _tree_set(monkeypatch, c4["root"])
    full = c4_report                                    # the module-scoped run over EVERY row (same specs, rows and tree set)
    primary_only = [r for r in c4["rows"] if r["sampling_mode"] == PRIMARY]
    slim = _analyze(c4["specs"], primary_only, ts)
    assert json.dumps(slim["eightarm"], sort_keys=True) == json.dumps(full["eightarm"], sort_keys=True)
    assert len(slim["legs"]) < len(full["legs"])
    # a HELD panel-model leg: reported in design.status_by_row, never collected, the arm's pooled mean UNAVAILABLE by name
    held = [dict(r, status="held", held_reason="held: the object is being re-fitted") if (r["model_key"], r["cell"]) == (NEMO, MIX) else r
            for r in c4["rows"]]
    rep = _analyze(c4["specs"], held, ts)
    k = M.leg_key(NEMO, MIX, PRIMARY)
    assert k not in rep["legs"] and rep["design"]["status_by_row"][k] == {"status": "held", "wave": "ext1", "reason": "held: the object is being re-fitted",
                                                                          "model_key": NEMO, "cell": MIX, "arm": "lora_mix", "mode": PRIMARY}
    pooled = rep["eightarm"]["pooled_by_arm"]["lora_mix"]
    assert pooled["status"] == "unavailable" and f"{NEMO}/{MIX}: held" in pooled["reason"]
    assert rep["eightarm"]["pooled_by_arm"]["lora_ctl2"]["status"] == "available" and rep["eightarm"]["n_available"] == 5
    assert rep["gates"]["failed"] == [] and comparison(rep, NEMO)["arms_absent"] == ["lora_mix", "alphasteer", "arditi"]
    assert any("HELD" in f and k in f for f in rep["findings"])


def test_one_launched_w1_leg_absent_fails_the_population_gate_by_name(tmp_path, monkeypatch):
    rows, specs = _build_c4(tmp_path, skip=[(GEMMA, CONST)])
    ts = _tree_set(monkeypatch, tmp_path)
    rep = _analyze(specs, rows, ts)
    k = M.leg_key(GEMMA, CONST, PRIMARY)
    assert rep["legs"][k]["missing"].startswith(f"no out-root {rows[24]['out_root']!r} under any gen root")
    assert set(rep["unusable_legs"]) == {k} and rep["gates"]["population"]["status"] == M.GATE_FAIL and k in rep["gates"]["population"]["detail"]
    assert rep["verdict"]["word"] is None and any(k in r for r in M.production_refusals(rep))
    assert comparison(rep, GEMMA)["usable"] is False and comparison(rep, QWEN)["usable"] is True
    assert rep["eightarm"]["pooled_by_arm"]["prompted_constitution"]["status"] == "unavailable"
    # the same matrix WITHOUT that wave in --waves: the nine ext1 rows are `not_analysed`, nothing is missing
    rep = _analyze(specs, rows, ts, waves=["published"])
    assert rep["gates"]["failed"] == [] and len(rep["legs"]) == 24 and rep["design"]["n_not_analysed"] == 9
    assert rep["design"]["status_by_row"][k]["status"] == "not_analysed" and "ext1" in rep["design"]["status_by_row"][k]["reason"]
    with pytest.raises(M.AnalysisError, match="no launch row of the analysed waves"):
        _analyze(specs, rows, ts, waves=["nowhere"])


def test_a_swapped_object_a_wrong_prefix_a_wrong_window_and_a_wrong_route_refuse_by_name(tmp_path, monkeypatch):
    """F11b / P5 (i′): every fitted arm is held to ITS OWN object; a prompted leg to its SI digest; every leg to its row's
    served window and route class."""
    ext = {r["cell"]: r for r in S.ext_rows(models=[GEMMA])}
    swapped = {(GEMMA, CTL2, PRIMARY): {"object_sha": S.staged_object_pin(GEMMA, f"base__{MIX}")},
               (GEMMA, MIX, PRIMARY): {"object_sha": S.staged_object_pin(GEMMA, f"base__{CTL2}")}}
    rows, specs = _build_c4(tmp_path, ext_overrides=swapped)
    ts = _tree_set(monkeypatch, tmp_path)
    rep = _analyze(specs, rows, ts)
    why = rep["legs"][M.leg_key(GEMMA, CTL2, PRIMARY)]["refused"]
    assert "another object was served" in why and CTL2 in why and S.staged_object_pin(GEMMA, f"base__{CTL2}") in why
    assert "another object was served" in rep["legs"][M.leg_key(GEMMA, MIX, PRIMARY)]["refused"]
    assert rep["legs"][M.leg_key(QWEN, MIX, PRIMARY)]["refused"] is None and rep["gates"]["population"]["status"] == M.GATE_FAIL

    def edit(manifest_edit):
        root = tmp_path / f"e{len(list(tmp_path.iterdir()))}"
        S.build_store(root, model_key=GEMMA, cell=CONST, row=ext[CONST], tree_sha=T2, correct=set(range(290)), manifest_edit=manifest_edit)
        rows2 = S.design_rows() + [ext[CONST]]
        rep2 = M.analyze([f"{tmp_path / 'pub'}={S.SHA7}", f"{root}={T2}"], S.SHA7, cells=rows2, sampling_rows=S.SAMPLING_ROWS,
                         render_facts=S.RENDER_FACTS, math500_ids=S.IDS, n_resamples=B_FAST, read_eval_logs=False, analysis_sha=T1,
                         tree_set=ts, repo_root=REPO, waves=WAVES)
        return rep2["legs"][M.leg_key(GEMMA, CONST, PRIMARY)]["refused"]

    def wrong_prefix(m):
        m["resolved_config"]["system_sha"] = S.SI_PREFIX_SHA16["safety_refusal"]
    assert "another system prompt was served" in edit(wrong_prefix)

    def wrong_si(m):
        m[C.MANIFEST_KEY_PROVENANCE]["si_id"] = "safety_refusal"
    assert "provenance.si_id 'safety_refusal' != the prompted cell's SI 'si_constitution'" in edit(wrong_si)

    def wrong_window(m):
        m[C.MANIFEST_KEY_TINYCAP]["served"]["max_model_len"] = S.MAX_MODEL_LEN
    assert f"served max_model_len {S.MAX_MODEL_LEN} != the row's geometry L {S.MAX_MODEL_LEN_LONGSI}" in edit(wrong_window)
    # the COMPLETE geometry is the row's contract — S and the client width too, never the window alone. The case this
    # guards (S / concurrency 33 -> 1 on a constitution leg, the echo S = 1) used to pass the arm contract and read
    # `server_settings_equal: true`; now every leg is held to the S its model budget buys at its window
    s_const = S.served_seqs(GEMMA, S.MAX_MODEL_LEN_LONGSI)
    assert s_const == 589_824 // S.MAX_MODEL_LEN_LONGSI == 24 and S.served_seqs(GEMMA, S.MAX_MODEL_LEN) == 33

    def served_s_one(m):
        m[C.MANIFEST_KEY_TINYCAP]["served"]["max_num_seqs"] = m[C.MANIFEST_KEY_TINYCAP]["served"]["concurrency"] = 1
    assert f"served max_num_seqs 1 != the standup's resolved S {s_const} (geometry.max_num_seqs)" in edit(served_s_one)

    def client_width(m):
        m[C.MANIFEST_KEY_TINYCAP]["served"]["concurrency"] = 1
    assert f"served concurrency 1 != the served max_num_seqs {s_const} — the client width is the served concurrency" in edit(client_width)

    def top_level_width(m):
        m[C.MANIFEST_KEY_CONCURRENCY] = 4
    assert f"manifest {C.MANIFEST_KEY_CONCURRENCY} 4 != the served max_num_seqs {s_const}" in edit(top_level_width)

    def geometry_s_one(m):                      # the record itself edited to S = 1, consistently — refused against the budget's rule
        g = m[C.MANIFEST_KEY_GEOMETRY]
        g["max_num_seqs"], g["batched_tokens"] = 1, S.MAX_MODEL_LEN_LONGSI
        m[C.MANIFEST_KEY_TINYCAP]["served"]["max_num_seqs"] = m[C.MANIFEST_KEY_TINYCAP]["served"]["concurrency"] = m[C.MANIFEST_KEY_CONCURRENCY] = 1
    assert f"geometry.max_num_seqs 1 != the budget's S {s_const} at L {S.MAX_MODEL_LEN_LONGSI:,} (budget 589,824 // L" in edit(geometry_s_one)

    def batched_off(m):
        m[C.MANIFEST_KEY_GEOMETRY]["batched_tokens"] += 1
    assert "geometry.batched_tokens" in edit(batched_off) and "!= max_num_seqs" in edit(batched_off)

    def seqs_override(m):
        m[C.MANIFEST_KEY_GEOMETRY]["seqs_source"] = "override"
    assert "geometry.seqs_source 'override' != 'model_budget'" in edit(seqs_override)

    def not_identity(m):
        m[C.MANIFEST_KEY_GEOMETRY]["identity"] = False
    assert "geometry.identity False: the standup did not attest max_num_seqs as the budget's S" in edit(not_identity)

    def other_entry(m):
        m[C.MANIFEST_KEY_GEOMETRY]["eval"] = "capability:gen_long_math"
    assert "geometry.eval 'capability:gen_long_math' != the row's geometry entry 'capability:gen_long_math_longsi'" in edit(other_entry)

    def other_model(m):
        m[C.MANIFEST_KEY_GEOMETRY]["model_key"] = QWEN
    assert f"geometry.model_key '{QWEN}' != the row's model '{GEMMA}'" in edit(other_model)

    def no_geometry(m):
        del m[C.MANIFEST_KEY_GEOMETRY]
    assert "manifest carries no `geometry` block" in edit(no_geometry)
    # an untouched leg's record passes the whole contract (the c4 legs are at their rows' windows and the budget's S)
    assert edit(lambda m: None) is None

    def lens_route(m):
        m[C.MANIFEST_KEY_ROUTE]["serving_identity"] = "proxy_vllm+lens+interleaved"
    assert "is a steered route on the 'prompted' arm" in edit(lens_route)

    def object_on_prompted(m):
        m[C.MANIFEST_KEY_PROVENANCE]["artifact_sha256"] = "c" * 64
    assert "an object was served" in edit(object_on_prompted)


def test_compatibility_refuses_a_moved_shared_field_a_moved_server_setting_and_an_unadmitted_tree(tmp_path, monkeypatch):
    """P5 (i) / (i″) / (iii): a LoRA leg vs `none` is COMPATIBLE across trees when the shared projection is equal; a moved
    chat template, a moved enable flag or a moved reasoning parser refuses naming the field; a leg on a tree nobody
    declared refuses before any estimand; the same leg under two roots is a filesystem union and refuses."""
    def moved_template(m):
        m[C.MANIFEST_KEY_PROVENANCE]["serving"]["chat_template_hash"] = "ffffffffffffffff"
    rows, specs = _build_c4(tmp_path / "tpl", ext_overrides={(QWEN, MIX, PRIMARY): {"manifest_edit": moved_template}})
    ts = _tree_set(monkeypatch, tmp_path)
    rep = _analyze(specs, rows, ts)
    c = comparison(rep, QWEN)
    assert c["usable"] is False and "shared-field projection differs — chat_template_hash" in c["reason"] and "lora_mix" in c["reason"]
    assert c["compatibility"]["projection_equal"] is False and c["compatibility"]["cross_tree"] is True
    assert rep["legs"][M.leg_key(QWEN, MIX, PRIMARY)]["refused"] is None       # the leg is fine on its own; the CONTRAST is not
    assert comparison(rep, GEMMA)["usable"] is True
    # B3 (code review 2026-09-12): the pooled mean over an arm whose contrast was REFUSED on one panel model is UNAVAILABLE,
    # naming that model and the reason, and carries no point / CI — it used to pool the incompatible leg as `available`
    pooled = rep["eightarm"]["pooled_by_arm"]
    assert pooled["lora_mix"]["status"] == "unavailable"
    assert f"{QWEN}/{MIX}: incompatible with {QWEN}/none" in pooled["lora_mix"]["reason"] and "chat_template_hash" in pooled["lora_mix"]["reason"]
    assert not any(k in pooled["lora_mix"] for k in ("primary", "score_as_is", "n_items", "weights"))
    assert pooled["lora_mix"]["pairs"][QWEN] == {"compatible": False, "reason": pooled["lora_mix"]["pairs"][QWEN]["reason"], "cross_tree": True}
    assert pooled["lora_mix"]["pairs"][GEMMA]["compatible"] and pooled["lora_mix"]["pairs"][NEMO]["compatible"]
    # every other arm still pools (the moved template touched one leg); the pair records say why each is admitted
    assert pooled["lora_ctl2"]["status"] == "available" and pooled["prompted_constitution"]["status"] == "available"
    assert all(p["compatible"] and p["cross_tree"] for p in pooled["lora_ctl2"]["pairs"].values())
    assert pooled["bare"]["status"] == "available" and all(p["cross_tree"] is False for p in pooled["bare"]["pairs"].values())
    assert "declared tree set" in pooled["lora_mix"]["pair_rule"] and rep["eightarm"]["n_available"] == 5
    assert rep["gates"]["population"]["status"] == M.GATE_FAIL and QWEN in rep["gates"]["population"]["detail"]

    def moved_flag(m):
        m[C.MANIFEST_KEY_THINKING]["enable_thinking"] = False
    rows, specs = _build_c4(tmp_path / "flag", ext_overrides={(NEMO, CTL2, PRIMARY): {"manifest_edit": moved_flag}})
    rep = _analyze(specs, rows, _tree_set(monkeypatch, tmp_path))
    assert "shared-field projection differs — enable_thinking" in comparison(rep, NEMO)["reason"]

    def moved_parser(m):
        m[C.MANIFEST_KEY_TINYCAP]["served"]["reasoning_parser"] = "other"
    rows, specs = _build_c4(tmp_path / "parser", ext_overrides={(GEMMA, CTL2, PRIMARY): {"manifest_edit": moved_parser}})
    rep = _analyze(specs, rows, _tree_set(monkeypatch, tmp_path))
    # the served parser is also the family's render fact: the LEG refuses first (its own contract), by name
    why = rep["legs"][M.leg_key(GEMMA, CTL2, PRIMARY)]["refused"] or comparison(rep, GEMMA)["reason"]
    assert "reasoning_parser" in why or "parser" in why
    # (iii) a tree nobody declared under the pinned root refuses the leg before any estimand
    rows, specs = _build_c4(tmp_path / "third", ext_overrides={(GEMMA, MIX, PRIMARY): {"tree_sha": T3}})
    rep = _analyze(specs, rows, _tree_set(monkeypatch, tmp_path))
    why = rep["legs"][M.leg_key(GEMMA, MIX, PRIMARY)]["refused"]
    assert T3 in why and "does not carry the wave pin" in why and T2 in why
    # a duplicate logical leg: the same row's out root under BOTH roots
    rows, specs = _build_c4(tmp_path / "dup")
    S.build_store(tmp_path / "dup" / "pub", model_key=GEMMA, cell=MIX, row=rows[26], tree_sha=T1, correct=set(range(290)))
    rep = _analyze(specs, rows, _tree_set(monkeypatch, tmp_path))
    why = rep["legs"][M.leg_key(GEMMA, MIX, PRIMARY)]["refused"]
    assert "exists under 2 gen roots" in why and "filesystem union" in why


def test_launch_attestation_is_read_from_the_standups_own_log_and_every_launch_flag_is_compared(c4_report, tmp_path, monkeypatch):
    """C2 (code review 2026-09-12): the server projection used to compare five recorded fields, so a changed precision or
    server default across admitted trees read `server_settings_equal: true`. The LAUNCH is now read from the standup's own
    record (the API server's `non-default args` echo in `vllm_server.log`, bound to the leg by the workers' attestation) and
    every echoed flag is compared except the named exclusions: a changed dtype and a changed generation-config default each
    refuse the contrast by the flag's name; a differing `--enable-log-requests` and a differing geometry do not; a leg WITHOUT
    the record refuses under a declared tree set with the remedy, and the record never claims equality it did not read."""
    rep = c4_report
    c = comparison(rep, GEMMA)
    launch = rep["legs"][M.leg_key(GEMMA, MIX, PRIMARY)]["launch"]
    assert launch["status"] == M.LAUNCH_ATTESTED and launch["problems"] == [] and launch["job_id"] and launch["standup_dir"].endswith("/math500")
    assert json.loads(launch["settings"]["dtype"]) == S.LAUNCH_DTYPE and json.loads(launch["settings"]["enable_prefix_caching"]) is True
    assert set(launch["excluded"]) >= {"port", "max_model_len", "max_num_seqs", "chat_template"} and "dtype" not in launch["excluded"]
    assert launch["generation_config_override"]["present"] is True and "top_k" in launch["generation_config_override"]["params"]
    assert c["compatibility"]["server_settings_equal"] is True and c["compatibility"]["recorded_settings_equal"] is True
    assert all(c["compatibility"]["launch_attested"].values()) and c["compatibility"]["server_settings_note"] is None
    proj = M.server_projection(rep["legs"][M.leg_key(GEMMA, MIX, PRIMARY)])
    assert proj["launch.status"] == M.LAUNCH_ATTESTED and "launch.dtype" in proj and "launch.port" not in proj and "launch.max_model_len" not in proj
    assert proj["requested_revision"] and proj["thinking_header_ids"] and proj["reasoning_parser_thinking"] == proj["reasoning_parser"]

    def analyse(name, overrides):
        rows, specs = _build_c4(tmp_path / name, ext_overrides=overrides)
        return _analyze(specs, rows, _tree_set(monkeypatch, tmp_path))

    def set_dtype(args):
        args["dtype"] = "float16"
    r = analyse("dtype", {(QWEN, MIX, PRIMARY): {"launch_edit": set_dtype}})
    c = comparison(r, QWEN)
    assert c["usable"] is False and "normalised server settings differ — launch.dtype" in c["reason"] and "float16" in c["reason"]
    assert c["compatibility"]["server_settings_equal"] is False and c["compatibility"]["recorded_settings_equal"] is True
    assert r["eightarm"]["pooled_by_arm"]["lora_mix"]["status"] == "unavailable" and "launch.dtype" in r["eightarm"]["pooled_by_arm"]["lora_mix"]["reason"]

    # the SERVER'S DEFAULT SAMPLING — the `--override-generation-config` echo and the engine's override WARNING — is compared by
    # EFFECT: nemotron's requests send temperature and top_p on every request, so a server default on exactly those keys reaches
    # no request -> ADMITTED, the literal difference REPORTED beside the equality
    def set_override(args):
        args["override_generation_config"] = {"temperature": 0.6, "top_p": 0.9}
    r = analyse("gen", {(NEMO, CTL2, PRIMARY): {"launch_edit": set_override}})
    c = comparison(r, NEMO)
    assert c["usable"] is True and c["compatibility"]["server_settings_equal"] is True
    assert c["compatibility"]["server_settings_literal_equal"] is False
    assert "launch.override_generation_config" in c["compatibility"]["server_settings_literal_difference"]
    assert c["compatibility"]["effective_decode_difference"] is None
    assert c["compatibility"]["effective_decode_by_leg"]["lora_ctl2"]["temperature"] == "1.0 (client)"
    assert c["compatibility"]["effective_decode_by_leg"]["lora_ctl2"]["top_k"] == "0.0 (server_default)"
    assert r["eightarm"]["pooled_by_arm"]["lora_ctl2"]["status"] == "available"
    assert M.d48_notes(r)["server_settings_literal_differs"][f"{NEMO}/{PRIMARY}"]["difference"].startswith("launch.override_generation_config")
    # ... but a server default on a key nemotron's requests do NOT send (top_k) reaches every request: refused by the KEY's name
    def set_override_topk(args):
        args["override_generation_config"] = {"top_k": 20}
    r = analyse("gen_topk", {(NEMO, CTL2, PRIMARY): {"launch_edit": set_override_topk}})
    c = comparison(r, NEMO)
    assert c["usable"] is False and "effective decode differs" in c["reason"] and "top_k" in c["reason"] and "20.0 (server_default)" in c["reason"]
    assert c["compatibility"]["server_settings_equal"] is False and c["compatibility"]["server_settings_literal_equal"] is False
    assert r["eightarm"]["pooled_by_arm"]["lora_ctl2"]["status"] == "unavailable" and "top_k" in r["eightarm"]["pooled_by_arm"]["lora_ctl2"]["reason"]
    # ... and an untransmitted key whose default moves by a hair (repetition_penalty 1.1) refuses too
    def set_override_rep(args):
        args["override_generation_config"] = {"temperature": 1.0, "top_p": 0.95, "repetition_penalty": 1.1}
    r = analyse("gen_rep", {(NEMO, MIX, PRIMARY): {"launch_edit": set_override_rep}})
    c = comparison(r, NEMO)
    assert c["usable"] is False and "repetition_penalty" in c["reason"] and "1.1 (server_default)" in c["reason"]
    # the generation defaults as the ENGINE reports them (the override WARNING line) changed on one tree on keys gemma's
    # requests SEND (temperature / top_k / top_p): no request is reached -> admitted, literal difference reported
    orig = S.GENERATION_CONFIG_BY_KEY
    monkeypatch.setattr(S, "GENERATION_CONFIG_BY_KEY", {**orig, GEMMA: {"temperature": 0.5, "top_k": 1, "top_p": 0.5}})
    rows, specs = _build_c4(tmp_path / "gc", pub_overrides=None)          # the published legs built under the ORIGINAL table below
    monkeypatch.setattr(S, "GENERATION_CONFIG_BY_KEY", orig)
    S.build_all(tmp_path / "gc" / "pub", rows=S.design_rows())            # rebuild the published legs with the original defaults
    r = _analyze(specs, rows, _tree_set(monkeypatch, tmp_path))
    c = comparison(r, GEMMA)
    assert c["usable"] is True and c["compatibility"]["server_settings_equal"] is True and c["compatibility"]["server_settings_literal_equal"] is False
    assert "launch.generation_config_override" in c["compatibility"]["server_settings_literal_difference"]
    # ... the same WARNING naming a key gemma's requests do NOT send (min_p): the default reaches every request -> refused
    monkeypatch.setattr(S, "GENERATION_CONFIG_BY_KEY", {**orig, GEMMA: {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "min_p": 0.2}})
    rows, specs = _build_c4(tmp_path / "gc_minp", pub_overrides=None)
    monkeypatch.setattr(S, "GENERATION_CONFIG_BY_KEY", orig)
    S.build_all(tmp_path / "gc_minp" / "pub", rows=S.design_rows())
    r = _analyze(specs, rows, _tree_set(monkeypatch, tmp_path))
    c = comparison(r, GEMMA)
    assert c["usable"] is False and "effective decode differs" in c["reason"] and "min_p" in c["reason"] and "0.2 (server_default)" in c["reason"]
    # THE EXTENSION CASE (verbatim shape): the published qwen standups ran with NO server default (the snapshot ships no
    # generation_config.json: no WARNING, no echo flag), the extension standups with the wave tree's pinned vendor row as
    # `--override-generation-config` (echoed AND stated by the engine); qwen's requests send temperature / top_p / top_k /
    # min_p / presence_penalty, so only repetition_penalty is the server's — 1.0 (neutral) == 1.0 (the pin): ADMITTED, the
    # literal fork REPORTED on the comparison and on the pooled pairs, the per-leg override dicts on record
    vendor = {"min_p": 0.0, "repetition_penalty": 1.0, "temperature": 0.7, "top_k": 20, "top_p": 0.8}
    def pin_qwen(args):
        args["override_generation_config"] = dict(vendor)
    monkeypatch.setattr(S, "GENERATION_CONFIG_BY_KEY", {**orig, QWEN: None})
    S.build_all(tmp_path / "w1q" / "pub", rows=S.design_rows())
    monkeypatch.setattr(S, "GENERATION_CONFIG_BY_KEY", {**orig, QWEN: dict(vendor)})
    S.build_all(tmp_path / "w1q" / "ext1", rows=S.ext_rows(), tree_sha=T2,
                overrides={(QWEN, cell, PRIMARY): {"launch_edit": pin_qwen} for cell in (CONST, CTL2, MIX)})
    monkeypatch.setattr(S, "GENERATION_CONFIG_BY_KEY", orig)
    rows = S.design_rows() + S.ext_rows() + S.steer_rows(33)
    r = _analyze([f"{tmp_path / 'w1q' / 'pub'}={S.SHA7}", f"{tmp_path / 'w1q' / 'ext1'}={T2}"], rows, _tree_set(monkeypatch, tmp_path))
    c = comparison(r, QWEN)
    assert c["usable"] is True and c["compatibility"]["server_settings_equal"] is True and c["compatibility"]["server_settings_literal_equal"] is False
    assert c["compatibility"]["generation_config_override_by_leg"]["bare"] is None
    assert json.loads(c["compatibility"]["generation_config_override_by_leg"]["lora_ctl2"]) == {k: float(v) for k, v in vendor.items()}
    eff = c["compatibility"]["effective_decode_by_leg"]
    assert eff["bare"]["repetition_penalty"] == "1.0 (server_default)" == eff["lora_ctl2"]["repetition_penalty"]
    assert eff["bare"]["top_k"] == "20.0 (client)" == eff["lora_ctl2"]["top_k"]
    assert all(r["eightarm"]["pooled_by_arm"][k]["status"] == "available" for k in ("prompted_constitution", "lora_ctl2", "lora_mix"))
    assert M.production_refusals(r) == []
    notes = M.d48_notes(r)
    assert set(notes["server_settings_literal_differs"]) == {f"{QWEN}/{PRIMARY}"} and notes["masked_primary"]["n_masked"] == 0
    # the two INTENTIONAL differences never refuse: request logging, and the row's geometry — the constitution leg's store
    # IS at its larger window and the budget's smaller S there (the c4 fixture serves every leg at its row's geometry), so
    # the qwen contrast pools legs at 17,664 x 40 beside 24,064 x 29 with `server_settings_equal: true`
    def log_requests(args):
        args["enable_log_requests"] = True
    r = analyse("intentional", {(QWEN, MIX, PRIMARY): {"launch_edit": log_requests}})
    c = comparison(r, QWEN)
    assert c["usable"] is True and c["compatibility"]["server_settings_equal"] is True
    assert "enable_log_requests" in r["legs"][M.leg_key(QWEN, MIX, PRIMARY)]["launch"]["excluded"]
    ctl2, mix = r["legs"][M.leg_key(QWEN, CONST, PRIMARY)], r["legs"][M.leg_key(QWEN, MIX, PRIMARY)]
    assert (ctl2["identity"]["served"]["max_model_len"], ctl2["identity"]["served"]["max_num_seqs"]) == (S.MAX_MODEL_LEN_LONGSI, S.served_seqs(QWEN, S.MAX_MODEL_LEN_LONGSI))
    assert (mix["identity"]["served"]["max_model_len"], mix["identity"]["served"]["max_num_seqs"]) == (S.MAX_MODEL_LEN, S.served_seqs(QWEN, S.MAX_MODEL_LEN))
    assert ctl2["launch"]["geometry_reconciled"] is True and mix["launch"]["geometry_reconciled"] is True
    # the echo's GEOMETRY is reconciled with the geometry the manifest records as served — an echo naming another S (S = 1 in
    # the log) or another window is not this leg's launch and attests nothing; the contrast refuses by name and the pooled arm
    # is unavailable
    def echo_s_one(args):
        args["max_num_seqs"], args["max_num_batched_tokens"] = 1, args["max_model_len"]
    def echo_other_window(args):
        args["max_model_len"], args["max_num_batched_tokens"] = 24064, 24064 * args["max_num_seqs"]
    r = analyse("echo", {(QWEN, CTL2, PRIMARY): {"launch_edit": echo_s_one}, (NEMO, MIX, PRIMARY): {"launch_edit": echo_other_window}})
    leg = r["legs"][M.leg_key(QWEN, CTL2, PRIMARY)]
    assert leg["refused"] is None and leg["launch"]["status"] == M.LAUNCH_ABSENT and leg["launch"]["geometry_reconciled"] is False
    assert any("the echo's max_num_seqs 1 != the manifest's served max_num_seqs" in p for p in leg["launch"]["problems"]), leg["launch"]["problems"]
    c = comparison(r, QWEN)
    assert c["usable"] is False and "launch settings cannot be compared" in c["reason"] and "max_num_seqs 1" in c["reason"]
    assert c["compatibility"]["server_settings_equal"] is None and c["compatibility"]["launch_attested"]["lora_ctl2"] is False
    assert r["eightarm"]["pooled_by_arm"]["lora_ctl2"]["status"] == "unavailable"
    leg = r["legs"][M.leg_key(NEMO, MIX, PRIMARY)]
    assert leg["launch"]["status"] == M.LAUNCH_ABSENT and any("the echo's max_model_len 24064 != the manifest's served max_model_len" in p for p in leg["launch"]["problems"])
    # the finding's reproduction end to end: the RECORD served at S = 1 (served block, client width, echo) on a constitution leg
    # is refused by the ARM CONTRACT (the leg unusable), not admitted with a pooled result
    def record_s_one(m):
        m[C.MANIFEST_KEY_TINYCAP]["served"]["max_num_seqs"] = m[C.MANIFEST_KEY_TINYCAP]["served"]["concurrency"] = m[C.MANIFEST_KEY_CONCURRENCY] = 1
    r = analyse("s1", {(GEMMA, CTL2, PRIMARY): {"manifest_edit": record_s_one, "launch_edit": echo_s_one}})
    leg = r["legs"][M.leg_key(GEMMA, CTL2, PRIMARY)]
    assert leg["refused"] is not None and "served max_num_seqs 1 != the standup's resolved S" in leg["refused"]
    assert comparison(r, GEMMA)["usable"] is False and r["eightarm"]["pooled_by_arm"]["lora_ctl2"]["status"] == "unavailable"
    # a leg WITHOUT the standup record: refused under a declaration with the remedy; the record's verdict is None, never True
    r = analyse("missing", {(NEMO, MIX, PRIMARY): {"launch_record": False}})
    c = comparison(r, NEMO)
    leg = r["legs"][M.leg_key(NEMO, MIX, PRIMARY)]
    assert leg["refused"] is None and leg["launch"]["status"] == M.LAUNCH_ABSENT and any("no standup record" in p for p in leg["launch"]["problems"])
    assert c["usable"] is False and "launch settings cannot be compared" in c["reason"] and "lora_mix: no launch attestation" in c["reason"]
    assert "pull the job's serving/" in c["reason"] and c["compatibility"]["server_settings_equal"] is None
    assert c["compatibility"]["launch_attested"]["lora_mix"] is False and c["compatibility"]["launch_attested"]["bare"] is True
    assert c["compatibility"]["recorded_settings_equal"] is True and "unattested" in c["compatibility"]["server_settings_note"]
    assert r["eightarm"]["pooled_by_arm"]["lora_mix"]["status"] == "unavailable" and "no launch attestation" in r["eightarm"]["pooled_by_arm"]["lora_mix"]["reason"]
    # a log whose echo is another launch's (the port disagrees with the attestation) is not this leg's record
    def other_port(args):
        args["port"] = 1
    r = analyse("port", {(GEMMA, CTL2, PRIMARY): {"launch_edit": other_port}})
    leg = r["legs"][M.leg_key(GEMMA, CTL2, PRIMARY)]
    assert leg["launch"]["status"] == M.LAUNCH_ABSENT and any("another launch" in p for p in leg["launch"]["problems"])
    assert comparison(r, GEMMA)["usable"] is False


def test_the_published_shape_keeps_the_one_sha_rule_and_records_no_launch_verdict_it_did_not_read(tmp_path):
    """Without a declared tree set the one-sha rule stands and nothing refuses on a missing standup record — the published
    pull carries none — but the compatibility record says so: `server_settings_equal` None, `launch_attested` false."""
    rows = S.cells_matrix()
    S.build_all(tmp_path / "pub", rows=rows, launch_record=False)
    rep = M.analyze(tmp_path / "pub", S.SHA7, cells=rows, sampling_rows=S.SAMPLING_ROWS, render_facts=S.RENDER_FACTS, math500_ids=S.IDS,
                    n_resamples=B_FAST, read_eval_logs=False, analysis_sha=T1, repo_root=REPO)
    c = comparison(rep, GEMMA)
    assert c["usable"] is True and c["compatibility"]["declared"] is False and c["compatibility"]["server_settings_equal"] is None
    assert not any(c["compatibility"]["launch_attested"].values()) and c["compatibility"]["recorded_settings_equal"] is True
    assert rep["legs"][M.leg_key(GEMMA, BARE, PRIMARY)]["launch"]["status"] == M.LAUNCH_ABSENT
    assert M.production_refusals(rep) == [] and "design" not in rep


# --------------------------------------------------------------------------- #
# presentation selectors, the eight-arm figures, the publication gate          #
# --------------------------------------------------------------------------- #


def _entries(figs, stem):
    return json.loads(Path(figs["numbers"]).read_text())["figures"][stem]


def test_selectors_pick_the_rows_shown_recompute_nothing_and_are_validated(c4_report, tmp_path):
    rep = c4_report
    plain = M.make_figures(rep, tmp_path / "plain", exclude_families=[GLM])
    paper = M.make_figures(rep, tmp_path / "paper", exclude_families=[GLM], modes=[PRIMARY])
    assert set(plain["written"]) == set(paper["written"]) == set(M.FIGURES) | set(M.EIGHTARM_STEMS)
    # 33 legs in the report; shown = 33 - 6 GLM legs - 6 qwen sensitivity legs - 3 nemotron sensitivity legs = 18
    assert "selection" not in plain and paper["selection"]["n_legs_report"] == 33 and paper["selection"]["n_legs_shown"] == 18
    assert paper["modes"] == [PRIMARY] and paper["excluded_families"] == [GLM] and "excluded_arms" not in paper
    side = json.loads(Path(paper["numbers"]).read_text())
    assert side["modes"] == [PRIMARY] and side["selection"]["n_legs_shown"] == 18 and json.loads(Path(plain["numbers"]).read_text()).get("modes") is None
    # no sensitivity-row number reaches a figure under the mode selector; the plain render still draws them
    assert not any("general" in e["name"] or "battery" in e["name"] for stem in paper["written"] for e in _entries(paper, stem))
    assert any("general" in e["name"] for e in _entries(plain, "final_m5t_sampling_sensitivity"))
    # the tables JSON is the FULL report's under an arm / mode selector (only the family exclusion filters rows, as published)
    t_plain, t_paper = json.loads(Path(plain["tables"]).read_text()), json.loads(Path(paper["tables"]).read_text())
    for k in ("modes", "selection"):
        t_paper.pop(k, None)
    assert t_plain == t_paper
    # the report itself is untouched: pooled means identical before and after any render
    assert rep["eightarm"] == c4_report["eightarm"] and "figures" not in rep
    # --figure-exclude-arm: validated against the report's matrix; the bare cell never; a deferred arm may be left out
    with pytest.raises(M.AnalysisError, match="not a cell of this report's matrix"):
        M.make_figures(rep, tmp_path / "bad1", exclude_arms=["constv2-mix-full__psr_ce_all"])
    with pytest.raises(M.AnalysisError, match="cannot be left out"):
        M.make_figures(rep, tmp_path / "bad2", exclude_arms=[BARE])
    with pytest.raises(M.AnalysisError, match="not a sampling mode"):
        M.make_figures(rep, tmp_path / "bad3", modes=["card"])
    seven = M.make_figures(rep, tmp_path / "seven", exclude_families=[GLM], exclude_arms=["arditi", MIX], modes=[PRIMARY])
    assert seven["excluded_arms"] == sorted([MIX, "arditi"]) and seven["selection"]["n_legs_shown"] == 15   # 18 minus the three LoRA-mix legs
    gone = [PG._segment_pattern(t) for t in (MIX, "lora_mix", "arditi")]       # whole segments: the `_minus_ctl` sibling stays
    for stem in seven["written"]:
        assert not any(p.search(e["name"]) or p.search(json.dumps(e["source"])) for e in _entries(seven, stem) for p in gone), stem
    assert any(PG._segment_pattern(OBJECT).search(e["name"]) for e in _entries(seven, "final_m5t_atcap_thinksplit"))
    assert M.publication_gate_record(rep, seven, tmp_path / "seven")["status"] == PG.GATE_PASS
    # a directory holds ONE selection
    with pytest.raises(M.AnalysisError, match="excluded_arms"):
        M.make_figures(rep, tmp_path / "seven", exclude_families=[GLM], modes=[PRIMARY])
    with pytest.raises(M.AnalysisError, match="modes"):
        M.make_figures(rep, tmp_path / "plain", exclude_families=[GLM], modes=[PRIMARY])
    # EVERY figure's title carries both counts under a selection (plan P5); the family exclusion alone keeps the published titles (B11)
    import matplotlib
    matplotlib.use("Agg")
    view = {**rep, "figures": {"excluded_families": [GLM], "modes": [PRIMARY]}}
    for stem, fn in M.figures_for(view).items():
        fig, _ = fn(view)
        title = fig.texts[0].get_text()
        assert "33 legs in report · 18 shown" in title, (stem, title)
        assert len(title.splitlines()) <= M.TITLE_MAX_LINES and all(len(line) <= M.TITLE_MAX_CHARS for line in title.splitlines()), (stem, title)
        fig, _ = fn({**rep, "figures": {"excluded_families": [GLM]}})
        assert "legs in report" not in fig.texts[0].get_text(), stem


def test_a_mode_selection_without_the_primary_row_skips_the_primary_only_figures_and_passes_the_gate(c4_report, tmp_path):
    """The defect this guards: `--figure-modes battery` validated, but the accuracy figure and the two eight-arm
    figures still drew MODE_PRIMARY, the sampling-sensitivity builder fell back onto a model's first PRESENT mode, and the
    contrasts figure drew the `<mode> - <primary>` deltas — 126 excluded-mode records from the accuracy figure alone, so
    the gate refused a VALID selection. Now the primary-only figures are SKIPPED with the reason, a model without a shown
    row is left out of the sensitivity panels (never handed another row), the deltas need both rows shown, and the gate
    passes; the aggregates and the plain render are untouched."""
    rep = c4_report
    battery = M.make_figures(rep, tmp_path / "battery", exclude_families=[GLM], modes=[M.MODE_BATTERY])
    side = json.loads(Path(battery["numbers"]).read_text())
    assert set(battery["written"]) == (set(M.FIGURES) | set(M.EIGHTARM_STEMS)) - set(M.PRIMARY_ONLY_FIGURES)
    assert set(side["skipped"]) == set(M.PRIMARY_ONLY_FIGURES) == {"final_m5t_accuracy_by_cell", M.EIGHTARM_STEM_ACCURACY, M.EIGHTARM_STEM_CONTRASTS}
    assert all(f"draws the primary sampling row ({PRIMARY}) only" in why for why in side["skipped"].values())
    assert side["modes"] == [M.MODE_BATTERY] and battery["selection"]["n_legs_shown"] < battery["selection"]["n_legs_report"] == 33
    # no record of an excluded mode reaches any written figure: the gate PASSES the selection it used to refuse
    gone = [PG._segment_pattern(m) for m in (PRIMARY, M.MODE_GENERAL)]
    for stem in battery["written"]:
        assert not any(p.search(e["name"]) or p.search(json.dumps(e["source"])) for e in _entries(battery, stem) for p in gone), stem
    assert M.publication_gate_record(rep, battery, tmp_path / "battery")["status"] == PG.GATE_PASS
    # the sensitivity figure draws the battery row of every model that HAS one (qwen, nemotron); gemma has none and is omitted
    sens = _entries(battery, "final_m5t_sampling_sensitivity")
    assert sens and all(f"/{M.MODE_BATTERY}]" in e["name"] for e in sens)
    assert {e["name"].split("[")[1].split("/")[0] for e in sens} == {QWEN, NEMO}
    view = {**rep, "figures": {"excluded_families": [GLM], "modes": [M.MODE_BATTERY]}}
    assert M._sensitivity_models(view) == [QWEN, NEMO] and M._shown_modes_of(view, GEMMA) == []
    assert M.figure_stems_for(rep, view=view) == [s_ for s_ in M.figures_for(rep) if s_ not in M.PRIMARY_ONLY_FIGURES]
    assert M.figure_skip_reason("final_m5t_accuracy_by_cell", view) and M.figure_skip_reason("final_m5t_accuracy_by_cell", rep) is None
    # the contrasts figure keeps the battery comparisons and drops every `<mode> - <primary>` delta
    rows = M._contrast_rows(view)
    assert rows and all(r["kind"] == "contrast" and f"[{M.MODE_BATTERY}]" in r["label"] for r in rows)
    assert any(r["kind"] == "delta" for r in M._contrast_rows({**rep, "figures": {"excluded_families": [GLM]}}))
    # the report and the plain render are untouched; a selection WITH the primary writes the full set as before
    assert rep["eightarm"] == c4_report["eightarm"] and "figures" not in rep
    paper = M.make_figures(rep, tmp_path / "paper", exclude_families=[GLM], modes=[PRIMARY])
    assert set(paper["written"]) == set(M.FIGURES) | set(M.EIGHTARM_STEMS) and json.loads(Path(paper["numbers"]).read_text())["skipped"] == {}
    # the CLI end to end: the selection renders, gates and exits 0
    for m in (M.MODE_GENERAL,):
        both = M.make_figures(rep, tmp_path / "general", exclude_families=[GLM], modes=[m])
        assert set(json.loads(Path(both["numbers"]).read_text())["skipped"]) == set(M.PRIMARY_ONLY_FIGURES)
        assert M.publication_gate_record(rep, both, tmp_path / "general")["status"] == PG.GATE_PASS


def test_eight_arm_figures_draw_eight_fixed_slots_hold_llama_out_and_read_the_pooled_field(c4_report, tmp_path):
    figs = M.make_figures(c4_report, tmp_path / "f", exclude_families=[GLM], modes=[PRIMARY])
    acc = _entries(figs, M.EIGHTARM_STEM_ACCURACY)
    names = {e["name"] for e in acc}
    for key in S.EXT_MODELS:
        for kind in ("bare", "prompted", "prompted_constitution", "lora_ctl2", "lora_mix", "object"):
            assert f"eightarm.acc.point[{key}/{kind}]" in names
        assert f"eightarm.acc.point[{key}/alphasteer]" not in names and f"eightarm.acc.point[{key}/arditi]" not in names
    assert not any(f"[{GLM}/" in n for n in names)
    assert "eightarm.pooled.point[lora_mix]" in names and "eightarm.pooled.point[arditi]" not in names
    pooled = next(e for e in acc if e["name"] == "eightarm.pooled.point[lora_ctl2]")
    assert pooled["source"] == "eightarm.pooled_by_arm.lora_ctl2.primary.point" and pooled["value"] == c4_report["eightarm"]["pooled_by_arm"]["lora_ctl2"]["primary"]["point"]
    con = _entries(figs, M.EIGHTARM_STEM_CONTRASTS)
    assert any(e["source"] == f"estimands.{GEMMA}.{PRIMARY}.primary.stats.lora_mix_minus_bare.point" for e in con)
    assert any(e["source"] == "eightarm.pooled_by_arm.lora_mix.primary.minus_bare.point" for e in con)
    caps = json.loads(Path(figs["numbers"]).read_text())["captions"]
    assert "deferred" in caps[M.EIGHTARM_STEM_ACCURACY] and "held out of the pooled group" in caps[M.EIGHTARM_STEM_ACCURACY]
    # the published shape draws exactly the published ten: no eight-arm figure, no eight-arm caption
    pub_rows = S.cells_matrix()
    root = tmp_path / "pub"
    S.build_all(root, rows=pub_rows)
    pub = M.analyze(root, S.SHA7, cells=pub_rows, sampling_rows=S.SAMPLING_ROWS, render_facts=S.RENDER_FACTS, math500_ids=S.IDS,
                    n_resamples=B_FAST, read_eval_logs=False, analysis_sha=T1)
    assert "design" not in pub and "eightarm" not in pub and "gen_roots" not in pub and pub["cells"] == pub_rows
    assert set(M.figures_for(pub)) == set(M.FIGURES) and set(M.figure_captions(pub)) == set(M.FIGURES)
    assert set(M.figures_for(c4_report)) == set(M.FIGURES) | set(M.EIGHTARM_STEMS)
    assert all("compatibility" in c and c["compatibility"]["declared"] is False and "cross_tree" in c["compatibility"] for c in pub["comparisons"])


def _fig_title(fig):
    """The title `_title` wrote (a figure-level `fig.text`, not a suptitle): the first line of the topmost text."""
    return "\n".join(t.get_text() for t in fig.texts)


def _slot_x_positions(fig):
    """The bar-center x positions a rendered eight-arm accuracy figure drew (every Rectangle patch on its axes)."""
    ax = fig.axes[0]
    return sorted(round(p.get_x() + p.get_width() / 2, 6) for p in ax.patches)


def test_a_registered_arm_in_neither_the_roster_nor_a_row_is_nowhere_and_a_roster_arm_with_no_row_reads_absent(c4_report):
    """THE DESIGN IS THE DECLARED ROSTER PLUS THE ARMS THE ROWS CARRY — two invariants, both general (they hold for the tenth arm):

    * registering an arm must not widen a panel: `arditi_allpos` is registered (external, its own hue and label) and is in
      neither the roster nor any analysed row here, so it appears in no design list, no slot map, no pooled record, no figure
      slot and no caption numeral;
    * a ROSTER arm that no row carries keeps its slot and reads `absent` — `absent` is a statement about the panel, not an
      artefact of the registry (GLM carries no LoRA-mix row in this fixture and still has the slot).
    """
    rep = c4_report
    assert "arditi_allpos" in M.ARM_BY_CELL and "arditi_allpos" not in M.PANEL_ROSTER_CELLS
    analysed = {r["cell"] for r in rep["cells"]}
    assert "arditi_allpos" not in analysed
    assert rep["design"]["arms"] == [a.cell for a in M.design_arm_roster(analysed)] == [a.cell for a in M.PUBLISHED_ARMS]
    assert rep["design"]["arms_by_model"][GLM]["lora_mix"] == "absent", "a roster arm with no row keeps its slot"
    assert all("arditi_allpos" not in slots for slots in rep["design"]["arms_by_model"].values())
    assert "arditi_allpos" not in rep["eightarm"]["pooled_by_arm"] and "arditi_allpos" not in rep["eightarm"]["arms"]
    assert all("arditi_allpos" not in (c.get("arms_absent") or []) for c in rep["comparisons"])
    assert [a.cell for a in M.panel_slots(rep)] == [a.cell for a in M.PUBLISHED_ARMS]
    cap = M.figure_captions(rep)[M.EIGHTARM_STEM_ACCURACY]
    assert cap.startswith("The eight-arm panel (plan P5): per model, eight FIXED arm slots") and "all-positions" not in cap
    # and the derivation itself: the roster in panel order, plus a carried arm, whatever order the cells arrive in; the roster
    # is DECLARED cell by cell, so a further registered arm cannot widen it
    assert M.PANEL_ROSTER_CELLS == (BARE, PROMPTED, CONST, CTL2, MIX, OBJECT, "alphasteer", "arditi")
    assert [a.cell for a in M.design_arm_roster(set())] == list(M.PANEL_ROSTER_CELLS)
    assert [a.cell for a in M.design_arm_roster({OBJECT, "arditi_allpos"})] == [*M.PANEL_ROSTER_CELLS, "arditi_allpos"]
    assert [a.cell for a in M.design_arm_roster({"made_up_cell"})] == list(M.PANEL_ROSTER_CELLS)
    assert [a.cell for a in M.arms_of_cells({OBJECT, BARE, "arditi_allpos"})] == [BARE, OBJECT, "arditi_allpos"]
    assert M.arms_of_cells({"made_up_cell"}) == [] and [a.cell for a in M.design_arms({})] == [a.cell for a in M.ARMS]


def test_figure_geometry_derives_from_the_arms_the_design_covers_not_from_the_registry(c4_report):
    """arditi-allpos-cap: a further registered arm must not shift the bars of a panel that does not carry it. A figure lays out
    the arms the report's DESIGN covers (`panel_slots`): this fixture's eight-arm design lays out eight at the pitch offsets the
    eight-arm formula gives — byte for byte the pre-change geometry — while a design that does cover nine lays out nine, and
    each panel's caption counts ITS OWN arms rather than the registry's length."""
    plt = M.T._mpl()[0]
    eight_cells = [a.cell for a in M.PUBLISHED_ARMS]
    assert [a.cell for a in M.panel_slots(c4_report)] == eight_cells
    nine = {**c4_report, "design": {**c4_report["design"], "arms": [a.cell for a in M.ARMS]}}
    assert [a.cell for a in M.panel_slots(nine)] == [a.cell for a in M.ARMS] and len(M.ARMS) == 9
    # an excluded arm keeps its (empty) slot: the geometry is the design's, the legend the drawn set's
    excluded = {**c4_report, "figures": {**(c4_report.get("figures") or {}), M.FIGURES_EXCLUDED_ARMS_KEY: ["arditi"]}}
    assert [a.cell for a in M.panel_slots(excluded)] == eight_cells
    fig8, numbers8 = M.fig_eightarm_accuracy(c4_report)
    fig9, numbers9 = M.fig_eightarm_accuracy(nine)
    try:
        pitch = 0.105
        models = M._shown_models(c4_report, list(c4_report["design"]["models"]))
        groups = len(models) + 1
        want8 = sorted(round(g + (s - (8 - 1) / 2) * pitch, 6) for g in range(groups) for s in range(8))
        want9 = sorted(round(g + (s - (9 - 1) / 2) * pitch, 6) for g in range(groups) for s in range(9))
        assert _slot_x_positions(fig8) == want8, "an eight-arm design lays out the eight-arm geometry exactly"
        assert _slot_x_positions(fig9) == want9 and want8 != want9
        assert [n["name"] for n in numbers8] == [n["name"] for n in numbers9]      # the ninth slot is an empty stub: no number drawn
        # the title's tail is ELIDED to the title-length rule on this fixture, so the count is read where it survives — the
        # caption — and the title is held to not CLAIMING the ninth arm on an eight-arm panel
        assert " of 9 arms" not in _fig_title(fig8) and "Eight arms" in _fig_title(fig8)
        cap8, cap9 = M.figure_captions(c4_report)[M.EIGHTARM_STEM_ACCURACY], M.figure_captions(nine)[M.EIGHTARM_STEM_ACCURACY]
        assert cap8.startswith("The eight-arm panel (plan P5): per model, eight FIXED arm slots in panel order (none, prompted safety, "
                               "prompted constitution, LoRA ctl2, LoRA mix, LoRA mix−ctl, AlphaSteer, Arditi) — ")
        assert cap9.startswith("The nine-arm panel (plan P5): per model, nine FIXED arm slots in panel order (none, prompted safety, "
                               "prompted constitution, LoRA ctl2, LoRA mix, LoRA mix−ctl, AlphaSteer, Arditi, Arditi all-positions) — ")
        assert cap8.split(" — ", 1)[1] == cap9.split(" — ", 1)[1]      # only the head counts the panel; the rest is unchanged
        # the forest lays out one row per arm the design covers (non-bare): nine-arm design -> one more per model group
        f8, _ = M.fig_eightarm_contrasts(c4_report)
        f9, _ = M.fig_eightarm_contrasts(nine)
        try:
            assert len(f9.axes[0].get_yticks()) - len(f8.axes[0].get_yticks()) == len(models) + 1
        finally:
            plt.close(f8)
            plt.close(f9)
    finally:
        plt.close(fig8)
        plt.close(fig9)


def test_the_publication_gate_refuses_a_planted_psr_record_and_passes_the_paper_view(c4_report, tmp_path):
    figs = M.make_figures(c4_report, tmp_path / "ok", exclude_families=[GLM], modes=[PRIMARY])
    rec = M.publication_gate_record(c4_report, figs, tmp_path / "ok")
    assert rec["status"] == PG.GATE_PASS and set(rec["figures"]) == set(figs["written"]) and rec["excluded_families"] == [GLM]
    assert rec["excluded_modes"] == ["general", "battery"]
    side = Path(figs["numbers"])
    doc = json.loads(side.read_text())
    doc["figures"][M.EIGHTARM_STEM_ACCURACY].append({"name": f"eightarm.acc.point[{QWEN}/constv2-mix-full__psr_ce_all]",
                                                      "source": "estimands.x", "transform": None, "value": 0.5})
    side.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(PG.PublicationRefused) as e:
        M.publication_gate_record(c4_report, figs, tmp_path / "ok")
    assert len(e.value.violations) == 1 and "constv2-mix-full__psr_ce_all" in e.value.violations[0] and M.EIGHTARM_STEM_ACCURACY in e.value.violations[0]
    # a stub render that wrote nothing has nothing to publish
    assert M.publication_gate_record(c4_report, {}, tmp_path / "nowhere")["status"] == PG.GATE_PASS


def test_the_cli_runs_the_c4_analysis_through_every_new_flag_and_gates_publication(c4, tmp_path, monkeypatch, capsys):
    _fake_git(monkeypatch, known={T1, T2}, diffs={(T1, T2): list(PROBE_FILES)})
    calls = []
    monkeypatch.setattr(M, "log_wandb", lambda *a, **k: calls.append("wandb"))
    cells = S.write_cells_json(tmp_path / "in" / "cells.json", c4["rows"])
    rows = S.write_sampling_rows(tmp_path / "in" / "rows.json")
    facts = S.write_render_facts(tmp_path / "in" / "facts.json")
    ids = tmp_path / "in" / "ids.json"
    ids.write_text(json.dumps(S.IDS))
    review = tmp_path / "in" / "tree_set_review.md"
    review.write_text("".join(f"- `{g}` — the extension's launcher / driver change; no shared served-identity field moves\n" for g in PROBE_FILES), encoding="utf-8")
    out = tmp_path / "report.json"
    args = ["--gen-root", c4["specs"][0], "--gen-root", c4["specs"][1], "--sha", S.SHA7, "--also-sha", T2, "--tree-diff-allow-file", str(review),
            "--tree-set-reason", WHY, "--waves", "published,ext1", "--cells", str(cells), "--sampling-rows", str(rows), "--render-facts", str(facts),
            "--math500-ids", str(ids), "--analysis-sha", T1, "--bootstrap", str(B_FAST), "--wandb-mode", "disabled", "--no-mirror", "--no-eval-logs",
            "--out", str(out), "--figs-dir", str(tmp_path / "figs"),
            "--figure-exclude-family", GLM, "--figure-modes", PRIMARY]
    rc = M.main(args)
    assert rc == 0 and calls == ["wandb"]
    rep = json.loads(out.read_text())
    assert rep["publication_gate"]["status"] == PG.GATE_PASS and rep["tree_set"]["allow_file"] == str(review)
    assert rep["tree_set"]["allow"] == list(PROBE_FILES) and set(rep["tree_set"]["allow_evidence"]) == set(PROBE_FILES)
    assert rep["design"]["waves"] == WAVES and rep["gen_roots"][1]["pin"] == T2 and rep["figures"]["modes"] == [PRIMARY]
    assert set(rep["figures"]["written"]) == set(M.FIGURES) | set(M.EIGHTARM_STEMS) and M.main(["--gate-report", str(out)]) == 0
    table = capsys.readouterr().out
    assert "producer trees: primary" in table
    # an unknown arm on the CLI is an error, not a silent no-op
    with pytest.raises(M.AnalysisError, match="not a cell of this report's matrix"):
        M.main([*args, "--figure-exclude-arm", "constv2-mix-full__psr_ce_all", "--figs-dir", str(tmp_path / "figs2")])
    # a planted PSR record in a render is refused before the mirror / W&B pass with exit 6 and the record in the report
    real = M.make_figures

    def planted(report, figs_dir, **kw):
        figs = real(report, figs_dir, **kw)
        side = Path(figs["numbers"])
        doc = json.loads(side.read_text())
        doc["figures"][M.EIGHTARM_STEM_CONTRASTS].append({"name": "x[safety_refusal__psr_kl_all]", "source": "y", "transform": None, "value": 1})
        side.write_text(json.dumps(doc), encoding="utf-8")
        return figs
    monkeypatch.setattr(M, "make_figures", planted)
    calls.clear()
    rc = M.main([*args[:-4], "--figs-dir", str(tmp_path / "figs3"), "--figure-exclude-family", GLM, "--figure-modes", PRIMARY])
    assert rc == PG.EXIT_PUBLICATION_REFUSED == 6 and calls == []
    rep = json.loads(out.read_text())
    assert rep["publication_gate"]["status"] == PG.GATE_REFUSED and "safety_refusal__psr_kl_all" in rep["publication_gate"]["violations"][0]
    assert "PUBLICATION REFUSED" in capsys.readouterr().err


def test_the_launch_attestation_records_the_serving_config_the_leg_actually_served(c4_report, tmp_path, monkeypatch):
    """The AlphaSteer rungs were selected on ANOTHER tree's serving config, so a leg run from
    current main is a SECOND INSTRUMENT and the fork must be visible per leg rather than inferred from a wave sha.
    Every leg's launch attestation therefore carries `serving_config_sha256` — the digest of the config the standup
    actually served, from the standup's own record. It is RECORDED, never compared: each row may legitimately serve
    its own overlay (an external-bank row does), so it never enters the cross-arm projection and never refuses a
    contrast. A leg produced before the record existed reads `absent` with a null digest — honest, never a refusal."""
    rep = c4_report
    for key in (GEMMA, QWEN, NEMO):
        leg = rep["legs"][M.leg_key(key, MIX, PRIMARY)]
        launch = leg["launch"]
        assert launch["status"] == M.LAUNCH_ATTESTED and launch["serving_config_record"] == M.LAUNCH_SERVING_CONFIG_READ
        assert C.is_sha256_hex(launch["serving_config_sha256"]) and launch["serving_config"].endswith(".math500.yaml")
        # RECORDED, not compared: the digest is in the attestation and in no projected (cross-arm) field
        proj = M.server_projection(leg)
        assert "serving_config_sha256" not in proj and not any("serving_config" in k for k in proj)
        assert "serving_config_sha256" not in (launch["settings"] or {})
    # two legs of one model serving DIFFERENT configs stay comparable (the steering rows' own overlays are exactly this)
    assert comparison(rep, GEMMA)["compatibility"]["server_settings_equal"] is True

    def analyse(name, overrides):
        rows, specs = _build_c4(tmp_path / name, ext_overrides=overrides)
        return _analyze(specs, rows, _tree_set(monkeypatch, tmp_path))

    r = analyse("othercfg", {(QWEN, MIX, PRIMARY): {"serving_config_sha256": "a" * 64}})
    leg = r["legs"][M.leg_key(QWEN, MIX, PRIMARY)]
    assert leg["launch"]["serving_config_sha256"] == "a" * 64
    assert leg["launch"]["serving_config_sha256"] != r["legs"][M.leg_key(QWEN, CTL2, PRIMARY)]["launch"]["serving_config_sha256"]
    c = comparison(r, QWEN)
    assert c["usable"] is True and c["compatibility"]["server_settings_equal"] is True      # recorded, never compared
    # a leg with NO record (the shape every published leg has): absent, null, still attested
    r = analyse("norecord", {(NEMO, CTL2, PRIMARY): {"serving_config_record": False}})
    leg = r["legs"][M.leg_key(NEMO, CTL2, PRIMARY)]
    assert leg["launch"]["status"] == M.LAUNCH_ATTESTED and leg["launch"]["problems"] == []
    assert leg["launch"]["serving_config_sha256"] is None
    assert leg["launch"]["serving_config_record"] == M.LAUNCH_SERVING_CONFIG_ABSENT
    assert comparison(r, NEMO)["usable"] is True
    # a MALFORMED record says so by name, and still does not refuse the leg
    d = Path(leg["launch"]["standup_dir"])
    (d / S.STANDUP_SERVING_CONFIG).write_text('{"serving_config_sha256": "nope"}')
    rec = M.serving_config_record(d)
    assert rec["serving_config_sha256"] is None and rec["serving_config_record"].startswith(M.LAUNCH_SERVING_CONFIG_UNREADABLE)
    # the producer's own record is the ONE writer: the standup writes it before the engine starts, the pull carries it
    standup = (REPO / "scripts/sdfbase/mt_serving_up.py").read_text()
    assert f'SERVING_CONFIG_RECORD_FILE = "{S.STANDUP_SERVING_CONFIG}"' in standup
    assert '"serving_config_sha256": _sha256(cfg_path)' in standup
    assert f'$SERVING_TASK/{S.STANDUP_SERVING_CONFIG}"' in (REPO / "scripts/sdfbase/math500/pull_wave.sh").read_text()


# --------------------------------------------------------------------------- #
# masking scope, refuse-unless-understood, and presentation                   #
# --------------------------------------------------------------------------- #


def _first_title_lines(fig):
    return [t for t in fig.texts if t.get_fontweight() == "bold"][0].get_text().split("\n")


def test_h1_a_masked_primary_draw_is_scoped_to_its_contrast_and_to_its_models_pair_inside_the_pooled_arm(c4_report, tmp_path, monkeypatch):
    """The first cut unioned the mask over the WHOLE comparison: one nemotron ctl2 draw moved all six nemotron arms,
    all five contrasts and every leg of the pooled ctl2 arm, and the pooled record lacked the with-reading. Now: ONE
    `eos_in_thoughts` draw on the nemotron extension ctl2 leg (classes handed in as run F takes them) leaves the ctl2 − bare
    contrast and the ctl2 leg's own accuracy; every other contrast, leg, transition and model is BYTE-EQUAL to the clean
    fixture; inside the pooled ctl2 arm only nemotron's pair excludes it (gemma's and qwen's pairs and every other pooled arm
    are the clean run's) and the with-reading is filed beside; every estimand figure carries the masking line."""
    j, leg = 270, (NEMO, CTL2, PRIMARY)
    item, key = S.IDS[j], M.leg_key(NEMO, CTL2, PRIMARY)
    rows, specs = _build_c4(tmp_path / "m", ext_overrides={leg: {"unscored": {j}, "not_at_cap": {j}}})
    classes = {key: {item: {"class": "eos_in_thoughts", "why": "test", "tag_literals": []}}}
    rep = _analyze(specs, rows, _tree_set(monkeypatch, tmp_path), anomaly_classes=classes)
    assert M.production_refusals(rep) == [] and rep["gates"]["split_anomalies"]["primary_masked"][key]["n_masked"] == 1
    clean = c4_report
    c, c0 = comparison(rep, NEMO), comparison(clean, NEMO)
    ctl2_contrast = M.contrast_name("lora_ctl2")
    assert set(c["masked_contrasts"]) == {ctl2_contrast} and c["masked_contrasts"][ctl2_contrast]["n_items"] == S.N - 1
    assert c["n_masked"] == 1 and c["mask_rule"] == M.MASK_SCOPE_RULE
    assert rep["masked_primary"]["by_contrast"] == {f"{NEMO}/{PRIMARY}": [ctl2_contrast]} and rep["masked_primary"]["rule"] == M.MASK_SCOPE_RULE
    for b in M.BRANCHES:
        e, e0 = c["estimands"][b], c0["estimands"][b]
        assert e["n_items"] == S.N and e["n_items_by_contrast"] == {**{k: S.N for k in e["stats"]}, ctl2_contrast: S.N - 1}
        for name in e["stats"]:
            if name != ctl2_contrast:
                assert e["stats"][name] == e0["stats"][name], (b, name)            # byte-equal to the unmasked run
        for kind in e["acc_by_leg"]:
            if kind != "lora_ctl2":
                assert e["acc_by_leg"][kind] == e0["acc_by_leg"][kind], (b, kind)
        assert e["acc_by_leg"]["lora_ctl2"]["n_scored"] + e["acc_by_leg"]["lora_ctl2"]["n_censored"] == S.N - 1
        assert c["estimands_with_anomalies"][b]["n_items"] == S.N and "n_items_by_contrast" not in c["estimands_with_anomalies"][b]
    for name in c["transitions"]:
        if name != ctl2_contrast:
            assert c["transitions"][name] == c0["transitions"][name] and c["mcnemar"][name] == c0["mcnemar"][name], name
    for m in (GEMMA, QWEN, GLM):
        assert json.dumps(comparison(rep, m)["estimands"], sort_keys=True) == json.dumps(comparison(clean, m)["estimands"], sort_keys=True), m
        assert "n_masked" not in comparison(rep, m)
    # the pooled arm: nemotron's pair alone excludes the item; the common resample stays over 500; the with-reading beside
    pa, pb = rep["eightarm"]["pooled_by_arm"], clean["eightarm"]["pooled_by_arm"]
    ctl2 = pa["lora_ctl2"]
    assert ctl2["n_items"] == S.N and ctl2["n_items_by_model"] == {GEMMA: S.N, QWEN: S.N, NEMO: S.N - 1}
    assert ctl2["n_masked"] == 1 and ctl2["n_masked_by_model"] == {GEMMA: 0, QWEN: 0, NEMO: 1} and ctl2["masked_ids_by_model"] == {NEMO: [item]}
    assert ctl2["masked_items"] == {key: [item]} and ctl2["mask_rule"] == M.MASK_SCOPE_RULE
    assert set(ctl2["estimands_with_anomalies"]) == set(M.BRANCHES) | {"n_items"} and ctl2["estimands_with_anomalies"]["n_items"] == S.N
    for b in M.BRANCHES:
        assert ctl2[b]["by_model"][GEMMA] == pb["lora_ctl2"][b]["by_model"][GEMMA] and ctl2[b]["by_model"][QWEN] == pb["lora_ctl2"][b]["by_model"][QWEN]
        assert ctl2[b]["by_model"][NEMO] == c["estimands"][b]["acc_by_leg"]["lora_ctl2"]["point"]
        assert ctl2[b]["bare"]["by_model"][NEMO] == c["masked_contrasts"][ctl2_contrast]["acc_by_leg"][b]["bare"]["point"]
        assert ctl2[b]["bare"]["by_model"][GEMMA] == pb["lora_ctl2"][b]["bare"]["by_model"][GEMMA]
    for kind, rec in pa.items():
        if rec["status"] != "available" or kind == "lora_ctl2":
            continue
        assert {b: rec[b] for b in M.BRANCHES} == {b: pb[kind][b] for b in M.BRANCHES}, kind
        assert rec["n_masked"] == 0 and rec["n_items_by_model"] == {m: S.N for m in M.POOLED_PANEL_MODELS} and "estimands_with_anomalies" not in rec
    assert rep["eightarm"]["population_rule"] == M.MASK_SCOPE_RULE
    with pytest.raises(M.AnalysisError, match="not in the panel's common set"):
        M.pooled_arm_bootstrap({m: rep["legs"][M.leg_key(m, CTL2, PRIMARY)] for m in M.POOLED_PANEL_MODELS},
                               {m: rep["legs"][M.leg_key(m, BARE, PRIMARY)] for m in M.POOLED_PANEL_MODELS},
                               n_resamples=10, seed=0, ci=0.95, exclude_by_model={NEMO: ["not_an_item"]})
    # the notes, the footnote and every estimand figure carry the scoped masking disclosure
    notes = M.d48_notes(rep)
    assert notes["masked_primary"]["by_contrast"] == rep["masked_primary"]["by_contrast"] and notes["masked_primary"]["scope_rule"] == M.MASK_SCOPE_RULE
    assert "removed from that arm's contrast only (H1)" in M._d48_footnote(rep)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    view = {**rep, "figures": {"excluded_families": [GLM], "modes": [PRIMARY]}}
    for stem in ("final_m5t_accuracy_by_cell", "final_m5t_contrasts", "final_m5t_sampling_sensitivity", "final_m5t_item_transitions", *M.EIGHTARM_STEMS):
        fig, _ = M.figures_for(view)[stem](view)
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        assert any(t.startswith("masked (D48): 1 eos_in_thoughts draw(s) on nemotron3nano4b lora_ctl2") for t in texts), stem
        plt.close(fig)
    figs = M.make_figures(rep, tmp_path / "figs", exclude_families=[GLM], modes=[PRIMARY])
    caps = json.loads(Path(figs["numbers"]).read_text())["captions"]
    assert all("Masked (D48, scoped per review H1)" in caps[stem] for stem in ("final_m5t_accuracy_by_cell", "final_m5t_contrasts", *M.EIGHTARM_STEMS))
    tables = json.loads(Path(figs["tables"]).read_text())
    assert tables["notes"]["masked_primary"]["scope_rule"] == M.MASK_SCOPE_RULE
    # second reader M-c: the masked leg's unscored count in the notes is its SCOPED n_censored (population minus scored items),
    # never the raw summary's — which still counts the masked draw — while its at-cap count is the raw draw's (the mask is non-cap)
    rec = tables["notes"]["conditional_vs_score_as_is"]["by_comparison"][f"{NEMO}/{PRIMARY}"][ctl2_contrast]
    ctl_row = next(row for row in rep["controls"] if row["leg"] == key)
    scoped = c["estimands"][M.BRANCH_PRIMARY]["acc_by_leg"]["lora_ctl2"]
    assert rec["scoped_to_masked_contrast"] is True and rec["n_items"] == S.N - 1
    assert rec["arm_n_scored"] == scoped["n_scored"] and rec["arm_n_unscored"] == scoped["n_censored"] and rec["arm_n_scored"] + rec["arm_n_unscored"] == S.N - 1
    assert rec["arm_n_unscored_raw_draw"] == ctl_row["summary"]["n_unscored"] == rec["arm_n_unscored"] + 1
    assert rec["arm_n_at_cap"] == ctl_row["summary"]["n_at_cap"] and rec["arm_n_draws"] == S.N
    bare_in = c["masked_contrasts"][ctl2_contrast]["acc_by_leg"][M.BRANCH_PRIMARY]["bare"]
    assert rec["bare_n_scored"] == bare_in["n_scored"] and rec["bare_n_unscored"] == bare_in["n_censored"]
    for other in (n for n in c["estimands"][M.BRANCH_PRIMARY]["stats"] if n != ctl2_contrast):
        o = tables["notes"]["conditional_vs_score_as_is"]["by_comparison"][f"{NEMO}/{PRIMARY}"][other]
        assert o["scoped_to_masked_contrast"] is False and o["n_items"] == S.N and o["arm_n_unscored"] == o["arm_n_unscored_raw_draw"], other
    texts = [t.get_text() for ax in M.figures_for(view)["final_m5t_contrasts"](view)[0].axes for t in ax.texts]
    first, second = M._denominator_lines(M.contrast_denominators(rep, rep["comparisons"].index(c), ctl2_contrast))
    assert first in texts and second in texts
    assert first.startswith(f"arm {scoped['n_scored']} · bare {bare_in['n_scored']} scored of {S.N - 1}; unscored {scoped['n_censored']} · {bare_in['n_censored']};")
    assert second.startswith(f"at cap {ctl_row['summary']['n_at_cap']} · ") and "Δ at-cap" in second and "shift" in second
    plt.close("all")
    # second reader L: the six RAW-DRAW panels carry the estimand-only statement (the mask does not reach them), the six
    # estimand figures the masking line — all twelve generated stems disclose; the clean fixture's raw-draw panels say nothing
    for stem, fn in M.figures_for(view).items():
        fig, _ = fn(view)
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        if stem in M.RAW_DRAW_PANELS:
            assert any(t.startswith("D48 mask (1 eos_in_thoughts draw(s) on nemotron3nano4b lora_ctl2) applies only to the named accuracy / contrast estimands; "
                                    "this panel retains the raw draw") for t in texts), stem
            assert not any(t.startswith("masked (D48)") for t in texts), stem
        else:
            assert any(t.startswith("masked (D48): 1 eos_in_thoughts draw(s) on nemotron3nano4b lora_ctl2") for t in texts), stem
        plt.close(fig)
    assert set(M.RAW_DRAW_PANELS) | set(M.FIGURES_NEED_ESTIMANDS) == set(M.FIGURE_STEMS) and len(M.figures_for(view)) == 12
    for stem in M.RAW_DRAW_PANELS:
        assert "applies only to the named accuracy / contrast estimands" in caps[stem] and "retains the raw draw" in caps[stem] and item in caps[stem], stem
    clean_view = {**clean, "figures": {"excluded_families": [GLM], "modes": [PRIMARY]}}
    for stem in M.RAW_DRAW_PANELS:
        fig, _ = M.figures_for(clean_view)[stem](clean_view)
        assert not any("D48" in t.get_text() for ax in fig.axes for t in ax.texts), stem
        plt.close(fig)
        assert "retains the raw draw" not in M.figure_captions(clean_view)[stem], stem
    # a render that EXCLUDES the masked draw's family has nothing to disclose on the raw-draw panels
    hidden = {**rep, "figures": {"excluded_families": [GLM, NEMO], "modes": [PRIMARY]}}
    assert M._raw_draw_footnote(hidden) == "" and M._raw_draw_caption(hidden) == ""


def test_h2_a_generation_config_key_outside_the_understood_set_refuses_and_the_output_limit_is_read_by_provenance(c4_report, tmp_path, monkeypatch):
    """Review H2 (the reviewer's case: a server-wide `max_new_tokens` on a real qwen leg passed as `server_settings_equal
    true`) as CORRECTED by the second reader's M-a. vLLM 0.25.1 composes the output limit by PROVENANCE: the flag's
    `--override-generation-config max_new_tokens` is `override_max_tokens`, a hard minimum — a CLIP, refused below the client
    cap; the model file's limit under `--generation-config auto` reaches the WARNING as `max_tokens` but is only the FALLBACK
    `get_max_tokens` takes when a request omits its cap, which MATH-500's requests never do — so a WARNING-only limit is
    REPORTED and the contrast admitted (revision 2 refused it: the false refusal this test used to codify); under a custom
    generation-config PATH the merged limit IS the clip; the two records disagreeing refuses; a clip above the client cap is
    understood and admitted; any other key in either record (seed, stop, logit_bias, …) refuses by name."""
    def analyse(name, overrides):
        rows, specs = _build_c4(tmp_path / name, ext_overrides=overrides)
        return _analyze(specs, rows, _tree_set(monkeypatch, tmp_path))

    vendor = {"min_p": 0.0, "repetition_penalty": 1.0, "temperature": 0.7, "top_k": 20, "top_p": 0.8}
    none_line = f"{M.CAP} (client cap {M.CAP}; server default none)"           # every production qwen leg's exact line (r2 = r3)
    # (1) the case, verbatim: the extension's qwen pin plus a FLAG max_new_tokens below the client cap — a clip, refused
    def pin_with_limit(args):
        args["override_generation_config"] = {**vendor, "max_new_tokens": 4096}
    r = analyse("limit", {(QWEN, MIX, PRIMARY): {"launch_edit": pin_with_limit}})
    c = comparison(r, QWEN)
    assert c["usable"] is False and "effective decode differs" in c["reason"] and "max_tokens" in c["reason"] and "4096" in c["reason"]
    assert c["compatibility"]["server_settings_equal"] is False and c["compatibility"]["server_settings_literal_equal"] is False
    lim = c["compatibility"]["effective_output_limit_by_leg"]
    assert lim["lora_mix"] == (f"4096 (client cap {M.CAP}; server clip 4096 from --override-generation-config max_new_tokens (a clip under every "
                               "generation-config source); clipped by the server default)")
    assert lim["bare"] == none_line and c["compatibility"]["output_limit_rule"] == M.OUTPUT_LIMIT_RULE
    assert "max_new_tokens" in c["compatibility"]["override_keys_by_leg"]["lora_mix"] and c["compatibility"]["override_key_problem"] is None
    assert r["eightarm"]["pooled_by_arm"]["lora_mix"]["status"] == "unavailable" and "max_tokens" in r["eightarm"]["pooled_by_arm"]["lora_mix"]["reason"]
    rec = M.effective_output_limit(r["legs"][M.leg_key(QWEN, MIX, PRIMARY)])
    assert rec == {"client_cap": M.CAP, "generation_config_source": "auto", "warning_recorded": True, "flag_recorded": True,
                   "state": M.OUTPUT_LIMIT_STATE_AUTO_FLAG_CLIP, "server_clip_max_tokens": 4096,
                   "clip_source": "--override-generation-config max_new_tokens (a clip under every generation-config source)",
                   "server_fallback_max_tokens": None, "fallback_governs": False, "effective": 4096, "clipped": True, "problem": None,
                   "rule": M.OUTPUT_LIMIT_RULE}
    # (2) the same limit stated by the ENGINE ALONE (the model's generation_config.json under auto, renamed max_tokens in
    # the WARNING) on gemma's extension legs is a FALLBACK the requests' own cap replaces: reported, not a clip, ADMITTED
    orig = S.GENERATION_CONFIG_BY_KEY
    monkeypatch.setattr(S, "GENERATION_CONFIG_BY_KEY", {**orig, GEMMA: {**orig[GEMMA], "max_tokens": 8192}})
    rows, specs = _build_c4(tmp_path / "warn", pub_overrides=None)
    monkeypatch.setattr(S, "GENERATION_CONFIG_BY_KEY", orig)
    S.build_all(tmp_path / "warn" / "pub", rows=S.design_rows())
    r = _analyze(specs, rows, _tree_set(monkeypatch, tmp_path))
    c = comparison(r, GEMMA)
    assert c["usable"] is True and c["compatibility"]["server_settings_equal"] is True and c["compatibility"]["server_settings_literal_equal"] is False
    lim = c["compatibility"]["effective_output_limit_by_leg"]
    assert lim["lora_mix"] == (f"{M.CAP} (client cap {M.CAP}; server fallback 8192 (generation_config.json under --generation-config auto: "
                               "replaced by the request's cap, not a clip))")
    assert lim["bare"] == none_line
    rec = M.effective_output_limit(r["legs"][M.leg_key(GEMMA, MIX, PRIMARY)])
    assert rec["server_fallback_max_tokens"] == 8192 and rec["server_clip_max_tokens"] is None and rec["clip_source"] is None
    assert rec["effective"] == M.CAP and rec["clipped"] is False and rec["fallback_governs"] is False and rec["problem"] is None
    assert all(r["eightarm"]["pooled_by_arm"][k]["status"] == "available" for k in ("prompted_constitution", "lora_ctl2", "lora_mix"))
    assert M.production_refusals(r) == []
    # (3) the same WARNING under a CUSTOM generation-config PATH (echoed on every gemma primary leg, so the path itself is no
    # literal difference): the merged max_tokens is override_max_tokens — a CLIP on the extension legs, refused by provenance
    def custom_path(args):
        args["generation_config"] = "/opt/generation-config"
    monkeypatch.setattr(S, "GENERATION_CONFIG_BY_KEY", {**orig, GEMMA: {**orig[GEMMA], "max_tokens": 8192}})
    rows, specs = _build_c4(tmp_path / "path", ext_overrides={(GEMMA, cell, PRIMARY): {"launch_edit": custom_path} for cell in (CONST, CTL2, MIX)})
    monkeypatch.setattr(S, "GENERATION_CONFIG_BY_KEY", orig)
    S.build_all(tmp_path / "path" / "pub", rows=S.design_rows(), overrides={(GEMMA, cell, PRIMARY): {"launch_edit": custom_path} for cell in (BARE, PROMPTED, OBJECT)})
    r = _analyze(specs, rows, _tree_set(monkeypatch, tmp_path))
    c = comparison(r, GEMMA)
    assert c["usable"] is False and "max_tokens" in c["reason"] and "8192" in c["reason"] and c["compatibility"]["server_settings_equal"] is False
    rec = M.effective_output_limit(r["legs"][M.leg_key(GEMMA, CTL2, PRIMARY)])
    assert rec["generation_config_source"] == "/opt/generation-config" and rec["server_clip_max_tokens"] == 8192 and rec["server_fallback_max_tokens"] is None
    assert rec["effective"] == 8192 and rec["clipped"] is True and "custom path" in rec["clip_source"] and rec["problem"] is None
    bare_rec = M.effective_output_limit(r["legs"][M.leg_key(GEMMA, BARE, PRIMARY)])
    assert bare_rec["generation_config_source"] == "/opt/generation-config" and bare_rec["server_clip_max_tokens"] is None and bare_rec["effective"] == M.CAP
    assert all(r["eightarm"]["pooled_by_arm"][k]["status"] == "unavailable" for k in ("prompted_constitution", "lora_ctl2", "lora_mix"))
    # (4) the two records DISAGREE (the flag names 4096, the WARNING — the merged dict with the flag applied last — 8192): refused, not read
    monkeypatch.setattr(S, "GENERATION_CONFIG_BY_KEY", {**orig, QWEN: {**orig[QWEN], "max_tokens": 8192}})
    rows, specs = _build_c4(tmp_path / "disagree", ext_overrides={(QWEN, MIX, PRIMARY): {"launch_edit": pin_with_limit}})
    monkeypatch.setattr(S, "GENERATION_CONFIG_BY_KEY", orig)
    S.build_all(tmp_path / "disagree" / "pub", rows=S.design_rows())
    r = _analyze(specs, rows, _tree_set(monkeypatch, tmp_path))
    c = comparison(r, QWEN)
    assert c["usable"] is False and "disagree" in c["reason"] and "4096" in c["reason"] and "8192" in c["reason"]
    rec = M.effective_output_limit(r["legs"][M.leg_key(QWEN, MIX, PRIMARY)])
    assert rec["problem"] and "disagree" in rec["problem"] and "PROBLEM" in c["compatibility"]["effective_output_limit_by_leg"]["lora_mix"]
    assert rec["state"] == M.OUTPUT_LIMIT_STATE_DISAGREE and rec["effective"] is None       # M1: an unreadable record is not read as a number
    assert c["compatibility"]["effective_output_limit_by_leg"]["lora_mix"].startswith("UNREADABLE (client cap ")
    # (5) a clip ABOVE the client cap on the extension side changes nothing served: understood, compared, admitted (the literal fork reported)
    def pin_loose(args):
        args["override_generation_config"] = {**vendor, "max_new_tokens": 2 * M.CAP}
    r = analyse("loose", {(QWEN, cell, PRIMARY): {"launch_edit": pin_loose} for cell in (CONST, CTL2, MIX)})
    c = comparison(r, QWEN)
    assert c["usable"] is True and c["compatibility"]["server_settings_equal"] is True and c["compatibility"]["server_settings_literal_equal"] is False
    assert c["compatibility"]["effective_output_limit_by_leg"]["lora_mix"] == (
        f"{M.CAP} (client cap {M.CAP}; server clip {2 * M.CAP} from --override-generation-config max_new_tokens (a clip under every generation-config source))")
    assert r["eightarm"]["pooled_by_arm"]["lora_mix"]["status"] == "available" and M.production_refusals(r) == []
    # the fallback governs only a leg that records NO client cap (a request that omits its max_tokens takes the file's)
    leg = json.loads(json.dumps(r["legs"][M.leg_key(QWEN, BARE, PRIMARY)]))
    leg["identity"][C.IDENTITY_KEY_MAX_TOKENS_BY_TASK] = {}
    leg["launch"]["generation_config_override"]["server_default"] = json.dumps({"temperature": 0.7, "max_tokens": 8192}, sort_keys=True)
    rec = M.effective_output_limit(leg)
    assert rec["client_cap"] is None and rec["fallback_governs"] is True and rec["effective"] == 8192 and rec["clipped"] is False
    assert "the fallback governs (no client cap recorded)" in M._output_limit_text(rec)
    # REFUSE-UNLESS-UNDERSTOOD: a key the effect rule does not understand refuses by name, whatever the client sends
    for key, value in (("seed", 0), ("stop", ["</s>"]), ("logit_bias", {"1": -100.0})):
        def edit(args, key=key, value=value):
            args["override_generation_config"] = {"temperature": 0.6, "top_p": 0.95, key: value}
        r = analyse(f"unknown_{key}", {(NEMO, CTL2, PRIMARY): {"launch_edit": edit}})
        c = comparison(r, NEMO)
        assert c["usable"] is False and "does not understand" in c["reason"] and repr(key) in c["reason"] and "refused, not waived" in c["reason"], key
        assert c["compatibility"]["server_settings_equal"] is False and c["compatibility"]["override_key_problem"] and key in c["compatibility"]["override_keys_by_leg"]["lora_ctl2"]
        assert r["eightarm"]["pooled_by_arm"]["lora_ctl2"]["status"] == "unavailable"
    assert set(M.OVERRIDE_UNDERSTOOD_KEYS) == set(C.SERVER_DEFAULTABLE_SAMPLING_KEYS) | {"max_new_tokens", "max_tokens"}
    rule = comparison(c4_report, QWEN)["compatibility"]["server_settings_rule"]
    assert "REFUSED UNLESS UNDERSTOOD" in rule and "by PROVENANCE" in rule and "never a clip" in rule
    assert all(v == none_line for v in comparison(c4_report, QWEN)["compatibility"]["effective_output_limit_by_leg"].values())


def test_m1_the_output_limit_states_are_exhaustive_and_fail_closed_on_vllm_with_a_warning_and_on_an_unattested_custom_path(c4_report, tmp_path, monkeypatch):
    """Second reader round 2, M1. vLLM 0.25.1 `get_diff_sampling_param`: `config = {} if src == "vllm" else
    self.try_get_generation_config(); config.update(self.override_generation_config)` and the WARNING is printed only when
    `diff_sampling_param and src != "vllm"` — so under `--generation-config vllm` NO file is loaded and NO WARNING can exist.
    Revision 3 accepted a WARNING-only record under `vllm` as a fallback and read a custom path with neither a WARNING nor a
    flag (`merged is None`) as no limit. Now the states are exhaustive (`OUTPUT_LIMIT_STATES`) and both records REFUSE:
    `vllm` + a WARNING is CONTRADICTORY; a custom path with no attested limit is UNKNOWN. The positive states — `vllm` + flag
    (a clip), `vllm` alone (no limit), a custom path whose WARNING names no limit (attested no limit), a custom path with a
    flag and no WARNING (the flag, applied last, is the clip) — read as they should; every production-shaped leg is
    `auto/no model-file limit` with the byte-identical line."""
    def analyse(name, *, ext_overrides=None, pub_overrides=None, gen_config=...):
        orig = S.GENERATION_CONFIG_BY_KEY
        if gen_config is not ...:
            monkeypatch.setattr(S, "GENERATION_CONFIG_BY_KEY", {**orig, GEMMA: gen_config})
        try:
            rows, specs = _build_c4(tmp_path / name, ext_overrides=ext_overrides, pub_overrides=pub_overrides)
        finally:
            monkeypatch.setattr(S, "GENERATION_CONFIG_BY_KEY", orig)
        return _analyze(specs, rows, _tree_set(monkeypatch, tmp_path))

    def every_gemma(edit):
        return ({(GEMMA, cell, PRIMARY): {"launch_edit": edit} for cell in (CONST, CTL2, MIX)},
                {(GEMMA, cell, PRIMARY): {"launch_edit": edit} for cell in (BARE, PROMPTED, OBJECT)})

    def vllm_source(args):
        args["generation_config"] = "vllm"

    def custom_path(args):
        args["generation_config"] = "/opt/generation-config"

    def vllm_with_flag(args):
        args["generation_config"] = "vllm"
        args["override_generation_config"] = {"temperature": 1.0, "max_new_tokens": 4096}

    def path_with_flag(args):
        args["generation_config"] = "/opt/generation-config"
        args["override_generation_config"] = {"temperature": 1.0, "max_new_tokens": 4096}

    none_line = f"{M.CAP} (client cap {M.CAP}; server default none)"
    assert len(set(M.OUTPUT_LIMIT_STATES)) == 11 and set(M.OUTPUT_LIMIT_REFUSED_STATES) <= set(M.OUTPUT_LIMIT_STATES)
    # the production shape: auto (no echo), the WARNING and the flag name no limit -> "auto/no model-file limit", the same line as r2 / r3
    rec = M.effective_output_limit(c4_report["legs"][M.leg_key(QWEN, MIX, PRIMARY)])
    assert rec["state"] == M.OUTPUT_LIMIT_STATE_AUTO_NO_LIMIT and rec["generation_config_source"] == "auto" and rec["problem"] is None
    assert M._output_limit_text(rec) == none_line and rec["warning_recorded"] is True
    # (1) REFUSED — `vllm` echoed on every gemma primary leg while the store still records the engine WARNING (gemma's snapshot ships
    # a generation_config.json): the engine cannot have printed it under vllm -> CONTRADICTORY, not a fallback (revision 3 admitted it)
    ext, pub = every_gemma(vllm_source)
    r = analyse("vllm_warn", ext_overrides=ext, pub_overrides=pub)
    c = comparison(r, GEMMA)
    assert c["usable"] is False and "contradict" in c["reason"] and "refused, not read" in c["reason"] and c["compatibility"]["server_settings_equal"] is False
    for cell in (BARE, MIX):
        rec = M.effective_output_limit(r["legs"][M.leg_key(GEMMA, cell, PRIMARY)])
        assert rec["state"] == M.OUTPUT_LIMIT_STATE_VLLM_CONTRADICTORY and rec["generation_config_source"] == "vllm"
        assert rec["warning_recorded"] is True and rec["problem"] and "contradict" in rec["problem"] and rec["effective"] is None
        assert rec["server_fallback_max_tokens"] is None and rec["server_clip_max_tokens"] is None and rec["fallback_governs"] is False
    assert c["compatibility"]["effective_output_limit_by_leg"]["lora_mix"].startswith("UNREADABLE (client cap ") and "PROBLEM" in c["compatibility"]["effective_output_limit_by_leg"]["lora_mix"]
    assert all(r["eightarm"]["pooled_by_arm"][k]["status"] == "unavailable" for k in ("prompted_constitution", "lora_ctl2", "lora_mix"))
    assert GEMMA in " ".join(M.production_refusals(r))
    # (2) REFUSED — a custom generation-config PATH on every gemma leg with NO WARNING recorded (the snapshot ships no file the
    # engine reports) and no flag: the file's limit is unread -> UNKNOWN (revision 3 read `merged is None` as no limit)
    ext, pub = every_gemma(custom_path)
    r = analyse("path_unknown", ext_overrides=ext, pub_overrides=pub, gen_config=None)
    c = comparison(r, GEMMA)
    assert c["usable"] is False and "UNKNOWN" in c["reason"] and "refused, not read" in c["reason"]
    rec = M.effective_output_limit(r["legs"][M.leg_key(GEMMA, CTL2, PRIMARY)])
    assert rec["state"] == M.OUTPUT_LIMIT_STATE_PATH_UNKNOWN and rec["generation_config_source"] == "/opt/generation-config"
    assert rec["warning_recorded"] is False and rec["flag_recorded"] is False and rec["effective"] is None and "UNKNOWN" in rec["problem"]
    assert c["compatibility"]["effective_output_limit_by_leg"]["lora_ctl2"].startswith("UNREADABLE (client cap ")
    assert all(r["eightarm"]["pooled_by_arm"][k]["status"] == "unavailable" for k in ("prompted_constitution", "lora_ctl2", "lora_mix"))
    # (3) `vllm` with NO WARNING recorded and no flag: no limit at all — admitted, the byte-identical line
    ext, pub = every_gemma(vllm_source)
    r = analyse("vllm_plain", ext_overrides=ext, pub_overrides=pub, gen_config=None)
    c = comparison(r, GEMMA)
    assert c["usable"] is True and c["compatibility"]["server_settings_equal"] is True
    rec = M.effective_output_limit(r["legs"][M.leg_key(GEMMA, MIX, PRIMARY)])
    assert rec["state"] == M.OUTPUT_LIMIT_STATE_VLLM_NO_LIMIT and rec["problem"] is None and rec["effective"] == M.CAP
    assert c["compatibility"]["effective_output_limit_by_leg"]["lora_mix"] == none_line
    # (4) `vllm` with NO WARNING and a flag max_new_tokens below the cap on the extension legs: the flag is the clip under every source ->
    # a served cap that differs from the published legs' -> refused as a clip (the same reason as the auto case above)
    r = analyse("vllm_flag", ext_overrides={(GEMMA, cell, PRIMARY): {"launch_edit": vllm_with_flag} for cell in (CONST, CTL2, MIX)},
                pub_overrides={(GEMMA, cell, PRIMARY): {"launch_edit": vllm_source} for cell in (BARE, PROMPTED, OBJECT)}, gen_config=None)
    c = comparison(r, GEMMA)
    assert c["usable"] is False and "effective decode differs" in c["reason"] and "4096" in c["reason"]
    rec = M.effective_output_limit(r["legs"][M.leg_key(GEMMA, MIX, PRIMARY)])
    assert rec["state"] == M.OUTPUT_LIMIT_STATE_VLLM_FLAG_CLIP and rec["server_clip_max_tokens"] == 4096 and rec["clipped"] is True and rec["problem"] is None
    assert rec["warning_recorded"] is False and rec["flag_recorded"] is True
    # (5) a custom PATH whose WARNING (the merged dict) names NO limit, on every gemma leg: attested no limit — admitted
    ext, pub = every_gemma(custom_path)
    r = analyse("path_attested", ext_overrides=ext, pub_overrides=pub)              # gemma's default WARNING carries no max_tokens
    c = comparison(r, GEMMA)
    assert c["usable"] is True and c["compatibility"]["effective_output_limit_by_leg"]["lora_mix"] == none_line
    rec = M.effective_output_limit(r["legs"][M.leg_key(GEMMA, MIX, PRIMARY)])
    assert rec["state"] == M.OUTPUT_LIMIT_STATE_PATH_ATTESTED_NO_LIMIT and rec["warning_recorded"] is True and rec["problem"] is None
    # (6) a custom PATH with a flag naming the limit and NO WARNING on the extension legs: the flag, applied last over the file, IS the merged
    # clip (known) -> differs from the published legs -> refused as a clip; the state names the missing WARNING
    r = analyse("path_flag", ext_overrides={(GEMMA, cell, PRIMARY): {"launch_edit": path_with_flag} for cell in (CONST, CTL2, MIX)},
                pub_overrides={(GEMMA, cell, PRIMARY): {"launch_edit": custom_path} for cell in (BARE, PROMPTED, OBJECT)}, gen_config=None)
    c = comparison(r, GEMMA)
    assert c["usable"] is False and "UNKNOWN" in c["reason"]          # its published legs (path, no record at all) refuse first
    rec = M.effective_output_limit(r["legs"][M.leg_key(GEMMA, MIX, PRIMARY)])
    assert rec["state"] == M.OUTPUT_LIMIT_STATE_PATH_FLAG_CLIP and rec["server_clip_max_tokens"] == 4096 and rec["problem"] is None
    assert "no override WARNING recorded" in rec["clip_source"]
    bare_rec = M.effective_output_limit(r["legs"][M.leg_key(GEMMA, BARE, PRIMARY)])
    assert bare_rec["state"] == M.OUTPUT_LIMIT_STATE_PATH_UNKNOWN and bare_rec["problem"]            # its published legs: path, no record at all
    # the rule names every fail-closed case
    assert "CONTRADICTORY" in M.OUTPUT_LIMIT_RULE and "UNKNOWN" in M.OUTPUT_LIMIT_RULE and "EXHAUSTIVE" in M.OUTPUT_LIMIT_RULE
    assert c4_report["comparisons"][0]["compatibility"]["output_limit_rule"] == M.OUTPUT_LIMIT_RULE


def test_mb_m2_m3_every_contrast_renders_both_differentials_with_their_directions_the_shift_decomposition_and_the_pooled_triples(c4_report, tmp_path):
    """Second reader M-b, corrected by round 2 (M2 / M3). M-b surfaced the denominators; M2 found the direction derived
    from AT-CAP counts, which are not censoring (a scored draw can finish at the cap): the censoring differential is now the
    UNSCORED-share difference `arm_unscored_share_minus_bare_unscored_share` (n_censored / n_items per leg over the
    contrast's own population) with its direction, the at-cap quantity is renamed neutrally (`Δ at-cap`, "more often at
    cap") and rendered BESIDE it, and the branch shift is the exact decomposition — score-as-is = conditional × scored share
    per leg, shift = bare conditional × bare unscored share − arm conditional × arm unscored share — checked against the
    filed points on every contrast. M3: a pooled row prints the per-model arm/bare scored triples beside the equal-weight
    means. Tested on the RENDER, not on field existence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rep = c4_report
    view = {**rep, "figures": {"excluded_families": [GLM], "modes": [PRIMARY]}}
    notes = M.conditional_vs_score_as_is_notes(rep)
    assert notes["censoring_rule"] == M.CENSORING_DIFFERENTIAL_RULE and notes["at_cap_rule"] == M.AT_CAP_DIFFERENTIAL_RULE
    assert "unscored share" in M.CENSORING_DIFFERENTIAL_RULE and "bare conditional × bare unscored share − arm conditional × arm unscored share" in M.CENSORING_DIFFERENTIAL_RULE
    assert "NOT a censoring measure" in M.AT_CAP_DIFFERENTIAL_RULE
    for word in (M.AT_CAP_DIRECTION_BARE, M.AT_CAP_DIRECTION_ARM, M.AT_CAP_DIRECTION_EQUAL):
        assert "censor" not in word                                    # the at-cap label is NEUTRAL
    assert M.CENSORING_DIFFERENTIAL_FIELD == "arm_unscored_share_minus_bare_unscored_share" and M.AT_CAP_DIFFERENTIAL_FIELD == "arm_at_cap_share_minus_bare_at_cap_share"
    ctl = {row["leg"]: row for row in rep["controls"]}
    cens_dirs = {M.CENSORING_DIRECTION_BARE, M.CENSORING_DIRECTION_ARM, M.CENSORING_DIRECTION_EQUAL}
    cap_dirs = {M.AT_CAP_DIRECTION_BARE, M.AT_CAP_DIRECTION_ARM, M.AT_CAP_DIRECTION_EQUAL}
    n_checked = 0
    for j, c in enumerate(rep["comparisons"]):
        if not c["primary"] or c["estimands"] is None:
            continue
        for name, signs in M.comp_contrasts(c).items():
            kind = M.plus_kind(signs)
            rec = notes["by_comparison"][f"{c['model_key']}/{c['mode']}"][name]
            sa, sb = ctl[c["leg_keys"][kind]]["summary"], ctl[c["leg_keys"]["bare"]]["summary"]
            acc = c["estimands"][M.BRANCH_PRIMARY]["acc_by_leg"]
            a, b = acc[kind], acc["bare"]
            assert rec["arm_n_scored"] == a["n_scored"] and rec["bare_n_scored"] == b["n_scored"] and rec["arm_n_correct"] == a["n_correct"]
            assert rec["arm_n_unscored"] == a["n_censored"] and rec["bare_n_unscored"] == b["n_censored"] and rec["arm_n_scored"] + rec["arm_n_unscored"] == rec["n_items"] == S.N
            # M2: the CENSORING differential from the UNSCORED counts over the contrast's population
            cens = a["n_censored"] / S.N - b["n_censored"] / S.N
            assert rec["arm_unscored_share"] == a["n_censored"] / S.N and rec["bare_unscored_share"] == b["n_censored"] / S.N
            assert rec[M.CENSORING_DIFFERENTIAL_FIELD] == cens and rec["censoring_differential_pp"] == 100.0 * cens
            assert rec["censoring_direction"] == (M.CENSORING_DIRECTION_EQUAL if cens == 0 else M.CENSORING_DIRECTION_ARM if cens > 0 else M.CENSORING_DIRECTION_BARE)
            # the at-cap differential, neutral, beside it
            cap = sa["n_at_cap"] / S.N - sb["n_at_cap"] / S.N
            assert rec["arm_n_at_cap"] == sa["n_at_cap"] and rec["arm_at_cap_share"] == sa["n_at_cap"] / S.N and rec["arm_n_draws"] == rec["bare_n_draws"] == S.N
            assert rec[M.AT_CAP_DIFFERENTIAL_FIELD] == cap and rec["at_cap_differential_pp"] == 100.0 * cap
            assert rec["at_cap_direction"] == (M.AT_CAP_DIRECTION_EQUAL if cap == 0 else M.AT_CAP_DIRECTION_ARM if cap > 0 else M.AT_CAP_DIRECTION_BARE)
            assert rec["censoring_direction"] in cens_dirs and rec["at_cap_direction"] in cap_dirs and rec["scoped_to_masked_contrast"] is False
            assert rec["directions_agree"] == (cens_dirs and (rec["censoring_direction"], rec["at_cap_direction"]) in {
                (M.CENSORING_DIRECTION_BARE, M.AT_CAP_DIRECTION_BARE), (M.CENSORING_DIRECTION_ARM, M.AT_CAP_DIRECTION_ARM),
                (M.CENSORING_DIRECTION_EQUAL, M.AT_CAP_DIRECTION_EQUAL)})
            # the exact decomposition reproduces the filed shift, and each filed point is the plain difference of the legs' accuracies
            p, q = c["estimands"][M.BRANCH_PRIMARY]["stats"][name]["point"], c["estimands"][M.BRANCH_SCORE_AS_IS]["stats"][name]["point"]
            a_cond, b_cond = a["n_correct"] / a["n_scored"], b["n_correct"] / b["n_scored"]
            assert rec["shift_pp"] == pytest.approx(100.0 * (q - p)) and rec["shift_identity_holds"] is True
            assert rec["shift_decomposition_pp"] == pytest.approx(100.0 * (b_cond * b["n_censored"] / S.N - a_cond * a["n_censored"] / S.N))
            assert rec["shift_bare_term_pp"] == pytest.approx(100.0 * b_cond * b["n_censored"] / S.N) and rec["arm_conditional"] == a_cond
            assert rec["shift_text"].startswith(f"{rec['shift_pp']:+.2f} pp = {b_cond:.3f} × ")
            paths = M.contrast_denominators(rep, j, name)["paths"]
            assert paths["arm_n_at_cap"] == f"controls.{rep['controls'].index(ctl[c['leg_keys'][kind]])}.summary.n_at_cap"
            assert paths["n_items"] == f"comparisons.{j}.estimands.{M.BRANCH_PRIMARY}.n_items" and paths["arm_n_correct"].endswith(f".{kind}.n_correct")
            n_checked += 1
    assert n_checked >= 5 * 3 + 2
    # the pooled arms: per-model denominators (the same scoped populations the pooled pair reads), the scored TRIPLES (M3), the
    # equal-weight means of BOTH differentials with their directions, and the pooled identity (mean shift == the pooled record's)
    for kind, rec in notes["pooled"].items():
        per = [notes["by_comparison"][f"{m}/{PRIMARY}"][M.contrast_name(kind)] for m in M.POOLED_PANEL_MODELS]
        assert set(rec["by_model"]) == set(M.POOLED_PANEL_MODELS) and rec["censoring_direction"] in cens_dirs and rec["at_cap_direction"] in cap_dirs
        assert rec["by_model"] == {m: {k: v for k, v in p.items() if k in rec["by_model"][m]} for m, p in zip(M.POOLED_PANEL_MODELS, per)}
        assert rec["scored_by_model"] == {m: {"arm_n_scored": p["arm_n_scored"], "bare_n_scored": p["bare_n_scored"], "n_items": p["n_items"]}
                                          for m, p in zip(M.POOLED_PANEL_MODELS, per)}
        for f in (M.CENSORING_DIFFERENTIAL_FIELD, M.AT_CAP_DIFFERENTIAL_FIELD, "arm_at_cap_share", "arm_unscored_share", "shift_pp", "shift_decomposition_pp"):
            assert rec[f] == pytest.approx(sum(p[f] for p in per) / 3), f
        assert rec["censoring_differential_pp"] == pytest.approx(100.0 * rec[M.CENSORING_DIFFERENTIAL_FIELD])
        assert rec["at_cap_differential_pp"] == pytest.approx(100.0 * rec[M.AT_CAP_DIFFERENTIAL_FIELD])
        pooled = rep["eightarm"]["pooled_by_arm"][kind]
        assert rec["pooled_shift_pp"] == pytest.approx(100.0 * (pooled[M.BRANCH_SCORE_AS_IS]["minus_bare"]["point"] - pooled[M.BRANCH_PRIMARY]["minus_bare"]["point"]))
        assert rec["shift_identity_holds"] is True
    assert set(notes["pooled"]) == {"prompted", "prompted_constitution", "lora_ctl2", "lora_mix", "object"}
    # RENDERED: under every contrast row of both forests TWO lines — the censoring line and the at-cap line, each with its direction
    fig, nums = M.fig_contrasts(view)
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    rows = [r for r in M._contrast_rows(view) if r["kind"] == "contrast"]
    assert len(rows) == 15                                   # three shown families × five arms on the primary row
    for r in rows:
        den = M.contrast_denominators(rep, r["comp"], r["contrast"])
        first, second = M._denominator_lines(den)
        assert first in texts and second in texts, (r["label"], first, second)
        assert first.startswith(f"arm {den['arm_n_scored']} · bare {den['bare_n_scored']} scored of {S.N}; unscored {den['arm_n_unscored']} · {den['bare_n_unscored']}; Δ censoring ")
        assert any(first.endswith(f"pp ({d})") for d in cens_dirs), first
        assert second.startswith(f"at cap {den['arm_n_at_cap']} · {den['bare_n_at_cap']} of {S.N} draws; Δ at-cap ")
        assert any(f"pp ({d})" in second for d in cap_dirs) and second.endswith(f"shift {den['shift_pp']:+.2f} pp"), second
        assert "censored" not in second                     # the at-cap line never says "censored"
        for f in ("arm_n_scored", "bare_n_scored", "arm_n_unscored", "bare_n_unscored", "arm_n_correct", "n_items", "arm_n_at_cap", "bare_n_at_cap",
                  "arm_unscored_share", "bare_unscored_share", M.CENSORING_DIFFERENTIAL_FIELD, "arm_at_cap_share", "bare_at_cap_share",
                  M.AT_CAP_DIFFERENTIAL_FIELD, "shift_pp", "shift_decomposition_pp"):
            assert any(n["name"] == f"{r['label']}.{f}" and n["value"] == den[f] for n in nums), (r["label"], f)
    plt.close(fig)
    fig, nums = M.EIGHTARM_FIGURES[M.EIGHTARM_STEM_CONTRASTS](view)
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    for m in M.POOLED_PANEL_MODELS:
        j = M._primary_comparison_index(rep, m, PRIMARY)
        for kind in ("prompted", "prompted_constitution", "lora_ctl2", "lora_mix", "object"):
            den = M.contrast_denominators(rep, j, M.contrast_name(kind))
            first, second = M._denominator_lines(den)
            assert first in texts and second in texts, (m, kind)
            for f in (M.CENSORING_DIFFERENTIAL_FIELD, M.AT_CAP_DIFFERENTIAL_FIELD):
                assert any(n["name"] == f"eightarm.contrast.{m}: {kind}.{f}" and n["value"] == den[f] for n in nums), (m, kind, f)
    # M3: every pooled row carries THREE lines — the per-model scored triples, then both means with the per-model values
    for kind in ("prompted", "prompted_constitution", "lora_ctl2", "lora_mix", "object"):
        pden = M.pooled_denominators(rep, kind)
        first, second, third = M._pooled_denominator_lines(pden)
        assert first in texts and second in texts and third in texts, (kind, first, second, third)
        assert first.startswith("scored arm/bare: ") and first.endswith(f" of {S.N} each")
        for m in M.POOLED_PANEL_MODELS:
            t = pden["scored_by_model"][m]
            assert f"{M._short_model(m)} {t['arm_n_scored']}/{t['bare_n_scored']}" in first, (kind, m)
        assert second.startswith("Δ censoring mean ") and " · per model " in second and second.count("/") == 2
        assert third.startswith("Δ at-cap mean ") and " · per model " in third and third.count("/") == 2 and f"shift mean {pden['shift_pp']:+.2f} pp" in third
        assert "censored" not in third
        label = f"pooled ({'/'.join(M.POOLED_PANEL_MODELS)}): {kind}"
        assert any(n["name"] == f"eightarm.contrast.{label}.{M.AT_CAP_DIFFERENTIAL_FIELD}" and n["value"] == pden[M.AT_CAP_DIFFERENTIAL_FIELD]
                   and len(n["source"]) == 12 for n in nums), kind
        assert any(n["name"] == f"eightarm.contrast.{label}.{M.CENSORING_DIFFERENTIAL_FIELD}" and n["value"] == pden[M.CENSORING_DIFFERENTIAL_FIELD]
                   and len(n["source"]) == 9 for n in nums), kind
        for m in M.POOLED_PANEL_MODELS:
            assert any(n["name"] == f"eightarm.contrast.{label}.{m}.arm_n_scored" and n["value"] == pden["scored_by_model"][m]["arm_n_scored"] for n in nums)
    plt.close(fig)
    assert M._short_model("gemma4e4b") == "gemma" and M._short_model("qwen35_4b") == "qwen" and M._short_model("nemotron3nano4b") == "nemotron"
    assert M.pooled_denominators(rep, "alphasteer") is None and M.pooled_denominators(rep, "bare") is None
    # the sidecar and the tables through the render; the captions carry both rules and the two-line reading
    figs = M.make_figures(rep, tmp_path / "den", exclude_families=[GLM], modes=[PRIMARY])
    side = json.loads(Path(figs["numbers"]).read_text())
    for stem in ("final_m5t_contrasts", M.EIGHTARM_STEM_CONTRASTS):
        names = {n["name"] for n in side["figures"][stem]}
        assert sum(1 for n in names if n.endswith(f".{M.CENSORING_DIFFERENTIAL_FIELD}")) >= 15, stem
        assert sum(1 for n in names if n.endswith(f".{M.AT_CAP_DIFFERENTIAL_FIELD}")) >= 15, stem
        assert sum(1 for n in names if n.endswith(".shift_decomposition_pp")) >= 15, stem
        cap = side["captions"][stem]
        assert M.CENSORING_DIFFERENTIAL_RULE in cap and M.AT_CAP_DIFFERENTIAL_RULE in cap and "Under each contrast row, two lines" in cap, stem
    tables = json.loads(Path(figs["tables"]).read_text())
    assert tables["notes"]["conditional_vs_score_as_is"] == json.loads(json.dumps(notes))
    # a contrast whose branches DISAGREE in sign is named in the caption with BOTH differentials and directions and the shift written out;
    # doctoring only the at-cap counts moves the at-cap line and its direction and leaves the censoring line alone
    doctored = json.loads(json.dumps(rep))
    c = comparison(doctored, QWEN)
    q = c["estimands"][M.BRANCH_SCORE_AS_IS]["stats"][M.contrast_name("prompted")]
    p = c["estimands"][M.BRANCH_PRIMARY]["stats"][M.contrast_name("prompted")]
    q["point"], q["point_sign"] = -p["point"], -int(p["point_sign"])
    ia = doctored["controls"].index(next(row for row in doctored["controls"] if row["leg"] == c["leg_keys"]["prompted"]))
    ib = doctored["controls"].index(next(row for row in doctored["controls"] if row["leg"] == c["leg_keys"]["bare"]))
    doctored["controls"][ia]["summary"]["n_at_cap"], doctored["controls"][ib]["summary"]["n_at_cap"] = 18, 78
    cap = M._headline_caption({**doctored, "figures": {"excluded_families": [GLM]}})
    assert f"{QWEN} prompted (" in cap and f"at cap 18 · 78 of {S.N} draws; Δ at-cap -12.0 pp — {M.AT_CAP_DIRECTION_BARE}; shift " in cap
    assert f"Δ censoring +0.0 pp — {M.CENSORING_DIRECTION_EQUAL}; at cap 18" in cap          # the unscored counts were not doctored
    doctored["controls"][ia]["summary"]["n_at_cap"], doctored["controls"][ib]["summary"]["n_at_cap"] = 78, 18
    assert f"Δ at-cap +12.0 pp — {M.AT_CAP_DIRECTION_ARM}; shift " in M._headline_caption({**doctored, "figures": {"excluded_families": [GLM]}})
    assert M._censoring_direction(0.0) == M.CENSORING_DIRECTION_EQUAL and M._censoring_direction(None) is None
    assert M._at_cap_direction(-0.1) == M.AT_CAP_DIRECTION_BARE and M._at_cap_direction(0.1) == M.AT_CAP_DIRECTION_ARM and M._at_cap_direction(None) is None
    # a leg without an at-cap count renders a readable at-cap line; the censoring line is unaffected
    doctored["controls"][ia]["summary"]["n_at_cap"] = None
    den = M.contrast_denominators(doctored, doctored["comparisons"].index(c), M.contrast_name("prompted"))
    assert den[M.AT_CAP_DIFFERENTIAL_FIELD] is None and den["at_cap_direction"] is None and den["directions_agree"] is None
    first, second = M._denominator_lines(den)
    assert "Δ at-cap n/a" in second and first.endswith(f"Δ censoring +0.0 pp ({M.CENSORING_DIRECTION_EQUAL})")
    # a record whose scored + unscored is not the contrast's population is refused, never read as a share
    doctored["comparisons"][doctored["comparisons"].index(c)]["estimands"][M.BRANCH_PRIMARY]["acc_by_leg"]["bare"]["n_censored"] += 1
    with pytest.raises(ValueError, match="not the contrast's population"):
        M.contrast_denominators(doctored, doctored["comparisons"].index(c), M.contrast_name("prompted"))


def test_m2_at_cap_and_unscored_disagree_and_each_line_carries_its_own_direction(tmp_path, monkeypatch):
    """Second reader round 2, M2 — the conflation the revision-3 regression codified: it derived the censoring direction from
    at-cap counts, and at-cap is not censored (a scored draw can finish at the cap). A store where the ARM has more at-cap
    draws (30 draws that closed their think block and ran to the cap in the answer — scored — plus 10 unscored) while the
    BARE anchor has more UNSCORED draws (20, all at cap): the censoring differential is −2.0 pp (the bare anchor more
    censored) while the at-cap differential is +4.0 pp (the arm more often at cap). Revision 3 would have rendered 'arm more
    censored' — reversed, the qwen LoRA mix case. Both lines render, each with its own direction, the decomposition reproduces
    the filed points, and the pooled arm's mean lines carry both directions with the per-model values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = {(QWEN, MIX, PRIMARY): {"unscored": set(range(10)), "capped_scored": set(range(100, 130))}}
    pub = {(QWEN, BARE, PRIMARY): {"unscored": set(range(20))}}
    rows, specs = _build_c4(tmp_path / "cens", ext_overrides=ext, pub_overrides=pub)
    r = _analyze(specs, rows, _tree_set(monkeypatch, tmp_path))
    c = comparison(r, QWEN)
    assert c["usable"] is True and M.production_refusals(r) == []
    j = r["comparisons"].index(c)
    name = M.contrast_name("lora_mix")
    den = M.contrast_denominators(r, j, name)
    assert (den["arm_n_unscored"], den["bare_n_unscored"], den["arm_n_at_cap"], den["bare_n_at_cap"]) == (10, 20, 40, 20)
    assert den["arm_n_scored"] == S.N - 10 and den["bare_n_scored"] == S.N - 20 and den["n_items"] == S.N
    assert den["censoring_differential_pp"] == pytest.approx(-2.0) and den["censoring_direction"] == M.CENSORING_DIRECTION_BARE
    assert den["at_cap_differential_pp"] == pytest.approx(4.0) and den["at_cap_direction"] == M.AT_CAP_DIRECTION_ARM
    assert den["directions_agree"] is False and den["shift_identity_holds"] is True
    acc = c["estimands"][M.BRANCH_PRIMARY]["acc_by_leg"]
    a_cond, b_cond = acc["lora_mix"]["n_correct"] / (S.N - 10), acc["bare"]["n_correct"] / (S.N - 20)
    assert den["shift_decomposition_pp"] == pytest.approx(100.0 * (b_cond * 20 / S.N - a_cond * 10 / S.N)) == pytest.approx(den["shift_pp"])
    first, second = M._denominator_lines(den)
    assert first == f"arm {S.N - 10} · bare {S.N - 20} scored of {S.N}; unscored 10 · 20; Δ censoring -2.0 pp ({M.CENSORING_DIRECTION_BARE})"
    assert second == f"at cap 40 · 20 of {S.N} draws; Δ at-cap +4.0 pp ({M.AT_CAP_DIRECTION_ARM}); shift {den['shift_pp']:+.2f} pp"
    # both forests render both lines; the notes and the sidecar carry both fields with their directions
    view = {**r, "figures": {"excluded_families": [GLM], "modes": [PRIMARY]}}
    for fn in (M.fig_contrasts, M.EIGHTARM_FIGURES[M.EIGHTARM_STEM_CONTRASTS]):
        fig, nums = fn(view)
        texts = {t.get_text() for ax in fig.axes for t in ax.texts}
        plt.close(fig)
        assert first in texts and second in texts
        mine = [n for n in nums if n["name"].startswith(f"{QWEN}: lora_mix") or n["name"].startswith(f"eightarm.contrast.{QWEN}: lora_mix")]
        assert any(n["name"].endswith(f".{M.CENSORING_DIFFERENTIAL_FIELD}") and n["value"] == pytest.approx(-0.02) for n in mine)
        assert any(n["name"].endswith(f".{M.AT_CAP_DIFFERENTIAL_FIELD}") and n["value"] == pytest.approx(0.04) for n in mine)
    rec = M.conditional_vs_score_as_is_notes(r)["by_comparison"][f"{QWEN}/{PRIMARY}"][name]
    assert rec["censoring_direction"] == M.CENSORING_DIRECTION_BARE and rec["at_cap_direction"] == M.AT_CAP_DIRECTION_ARM and rec["directions_agree"] is False
    # the pooled LoRA mix arm: the two means point OPPOSITE ways (gemma and nemotron contribute 0 to both), each line says which
    pden = M.pooled_denominators(r, "lora_mix")
    assert pden["censoring_differential_pp"] == pytest.approx(-2.0 / 3) and pden["censoring_direction"] == M.CENSORING_DIRECTION_BARE
    assert pden["at_cap_differential_pp"] == pytest.approx(4.0 / 3) and pden["at_cap_direction"] == M.AT_CAP_DIRECTION_ARM
    assert pden["directions_agree"] is False and pden["shift_identity_holds"] is True
    p1, p2, p3 = M._pooled_denominator_lines(pden)
    assert f"qwen {S.N - 10}/{S.N - 20}" in p1 and p1.endswith(f"of {S.N} each")
    assert p2.startswith(f"Δ censoring mean -0.7 pp ({M.CENSORING_DIRECTION_BARE}) · per model +0.0 / -2.0 / +0.0")
    assert p3.startswith(f"Δ at-cap mean +1.3 pp ({M.AT_CAP_DIRECTION_ARM}) · per model +0.0 / +4.0 / +0.0")
    # the other contrasts of the comparison share the censored anchor: every one reads the bare anchor as the more censored leg
    for other in M.comp_contrasts(c):
        if other != name:
            d = M.contrast_denominators(r, j, other)
            assert d["censoring_direction"] == M.CENSORING_DIRECTION_BARE and d["bare_n_unscored"] == 20 and d["shift_identity_holds"] is True



def test_closing_pass_one_scoring_definition_everywhere_and_a_scored_at_cap_draw_keeps_its_score(c4_report, tmp_path, monkeypatch):
    """The closing review: revisions 1–4 defined the primary estimand's omitted draws by their AT-CAP status
    ("an at-cap or unterminated draw is unscored") — false: a draw that closed its think block and ran to the cap in the
    answer is scored (production qwen LoRA mix: 49 of its 83 at-cap draws are scored, 3 of them correct). ONE definition
    now (`SCORING_DEFINITION`), rendered identically in the eight-arm block's `headline_reading`, in `HEADLINE_RULE`, in the
    cell-accuracy / contrasts / eight-arm accuracy / eight-arm contrasts captions of both the numbers sidecar and the tables
    JSON, and in the tables' conditional-vs-score-as-is rule; no false variant survives in any rendered export, in the
    module's source or in the record. And the definition is what the code does: an arm with 30 scored-at-cap draws (all
    correct by the store's default) keeps every one of them in n_scored and in n_correct."""
    definition = M.SCORING_DEFINITION
    assert definition == ("Primary (conditional) accuracy omits UNSCORED draws (those the checker could not score — capped inside an "
                          "unfinished think block or otherwise unterminated); scored-at-cap responses retain their scores. Score-as-is "
                          "counts only the unscored draws as wrong")
    false_variants = ("at-cap or unterminated", "at cap or unterminated", "unscored (at cap", "unscored (at-cap",
                      "at-cap draw is unscored", "many at-cap draws is read")

    def clean(text: str) -> bool:
        low = text.lower()
        return not any(v in low for v in false_variants)
    rep = c4_report
    view = {**rep, "figures": {"excluded_families": [GLM], "modes": [PRIMARY]}}
    assert definition in rep["eightarm"]["headline_reading"] and clean(rep["eightarm"]["headline_reading"])
    assert definition in M.HEADLINE_RULE and clean(M.HEADLINE_RULE) and M.HEADLINE_RULE.count(definition) == 1
    four = ("final_m5t_accuracy_by_cell", "final_m5t_contrasts", M.EIGHTARM_STEM_ACCURACY, M.EIGHTARM_STEM_CONTRASTS)
    caps = M.figure_captions(view)
    for stem in four:
        assert caps[stem].count(definition) == 1, stem          # stated once per caption, never twice
    assert all(clean(t) for t in caps.values())
    figs = M.make_figures(rep, tmp_path / "definition", exclude_families=[GLM], modes=[PRIMARY])
    numbers = json.loads(Path(figs["numbers"]).read_text())
    tables = json.loads(Path(figs["tables"]).read_text())
    for stem in four:
        assert definition in numbers["captions"][stem] and definition in tables["captions"][stem], stem
    assert tables["notes"]["conditional_vs_score_as_is"]["rule"] == M.HEADLINE_RULE and definition in tables["notes"]["conditional_vs_score_as_is"]["rule"]
    assert clean(json.dumps(numbers)) and clean(json.dumps(tables)) and clean(json.dumps(rep))
    assert clean(Path(M.__file__).read_text())
    # the definition is what the code does: 30 scored-at-cap draws, every one correct by default, stay scored AND correct
    ext = {(QWEN, MIX, PRIMARY): {"unscored": set(range(10)), "capped_scored": set(range(100, 130))}}
    rows, specs = _build_c4(tmp_path / "keeps", ext_overrides=ext)
    r = _analyze(specs, rows, _tree_set(monkeypatch, tmp_path))
    c = comparison(r, QWEN)
    assert c["usable"] is True and M.production_refusals(r) == []
    default_correct = set(S.default_kwargs(QWEN, MIX, PRIMARY)["correct"])
    assert set(range(100, 130)) <= default_correct
    acc = c["estimands"][M.BRANCH_PRIMARY]["acc_by_leg"]["lora_mix"]
    assert acc["n_scored"] == S.N - 10 and acc["n_censored"] == 10 and acc["n_correct"] == len(default_correct - set(range(10)))
    den = M.contrast_denominators(r, r["comparisons"].index(c), M.contrast_name("lora_mix"))
    assert (den["arm_n_at_cap"], den["arm_n_unscored"], den["arm_n_scored"]) == (40, 10, S.N - 10)
    assert r["eightarm"]["headline_reading"] == rep["eightarm"]["headline_reading"]


def test_closing_pass_the_server_settings_rule_places_the_model_file_fallback_under_auto_only(c4_report):
    """The closing review: the legacy `compatibility.server_settings_rule` string still said a model-file
    fallback exists under auto / vllm; under --generation-config vllm no model file is loaded (revision 4's M1, the
    authoritative `output_limit_rule`) — the fallback exists under auto ONLY, and the two rule strings now agree."""
    seen = 0
    for c in c4_report["comparisons"]:
        if not c.get("compatibility"):
            continue
        seen += 1
        rule, limit_rule = c["compatibility"]["server_settings_rule"], c["compatibility"]["output_limit_rule"]
        assert "a model-file FALLBACK under auto only" in rule and "vllm loads no model file" in rule, rule
        assert "auto / vllm" not in rule and "never a clip" in rule
        assert limit_rule == M.OUTPUT_LIMIT_RULE and "under --generation-config vllm no file is loaded" in limit_rule
    assert seen >= 1


def test_l6_m3_m4_two_word_titles_arm_agnostic_captions_and_the_geometry_and_headline_disclosures(c4_report, tmp_path, monkeypatch):
    """Review L6 (ten legacy titles were sentence-length; captions described three cells; the single-mode 'sensitivity'
    figure compared nothing), M3 (served geometry disclosed wherever it differs within a contrast) and M4 (the headline is
    conditional accuracy on scored items, the score-as-is sensitivity beside it; a pooled arm files no directional word,
    and says why)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rep = c4_report
    view = {**rep, "figures": {"excluded_families": [GLM], "modes": [PRIMARY]}}
    for stem, fn in M.figures_for(view).items():
        fig, _ = fn(view)
        assert M.figure_title_problem(fig) is None, stem
        lines = _first_title_lines(fig)
        assert len(lines) == 2 and len(lines[0].split()) <= M.TITLE_MAX_WORDS and lines[0] == lines[0].strip(), (stem, lines)
        plt.close(fig)
    fig, ax = plt.subplots()
    M._title(fig, "three word title\nsubtitle")
    assert "has 3 words" in M.figure_title_problem(fig)
    plt.close(fig)
    # the render enforces the rule: a builder that titles a figure with a sentence is refused before anything is written
    original = M.FIGURES["final_m5t_ledger"]

    def sentence_titled(report):
        fig, nums = original(report)
        [t for t in fig.texts if t.get_fontweight() == "bold"][0].set_text("Usage-ledger confirmation per cell\nsubtitle")
        return fig, nums
    monkeypatch.setitem(M.FIGURES, "final_m5t_ledger", sentence_titled)
    with pytest.raises(M.AnalysisError, match="final_m5t_ledger: title line 'Usage-ledger confirmation per cell' has 4 words"):
        M.make_figures(rep, tmp_path / "refused", exclude_families=[GLM], modes=[PRIMARY])
    monkeypatch.setitem(M.FIGURES, "final_m5t_ledger", original)
    plt.close("all")
    # the single-mode render: "Primary row" and a caption that says it compares no sampling rows; the plain render: "Sampling rows"
    fig, _ = M.FIGURES["final_m5t_sampling_sensitivity"](view)
    assert _first_title_lines(fig)[0] == "Primary row" and "no sampling-row comparison" in _first_title_lines(fig)[1]
    plt.close(fig)
    fig, _ = M.FIGURES["final_m5t_sampling_sensitivity"]({**rep, "figures": {"excluded_families": [GLM]}})
    assert _first_title_lines(fig)[0] == "Sampling rows"
    plt.close(fig)
    figs = M.make_figures(rep, tmp_path / "paper", exclude_families=[GLM], modes=[PRIMARY])
    caps = json.loads(Path(figs["numbers"]).read_text())["captions"]
    assert caps["final_m5t_sampling_sensitivity"].startswith("One sampling row shown (thinking_card)")
    plain = M.make_figures(rep, tmp_path / "plain", exclude_families=[GLM])
    assert json.loads(Path(plain["numbers"]).read_text())["captions"]["final_m5t_sampling_sensitivity"].startswith("Conditional accuracy (scored items) per cell under every sampling row")
    assert not any("three cells" in t or "the prompted and the object cell" in t or "object - bare and prompted - bare" in t for t in caps.values())
    assert "each arm present" in caps["final_m5t_accuracy_by_cell"] and "each arm present" in caps["final_m5t_item_transitions"]
    # M3: the c4 fixture serves the constitution row at the larger window with the budget's smaller S, so every panel model's
    # contrast spans two geometries — disclosed in the notes, the footnote of every estimand figure, the captions and the tables
    notes = M.d48_notes(rep)
    geo = notes["served_geometry"]
    for m in S.EXT_MODELS:
        rec = geo[f"{m}/{PRIMARY}"]
        assert rec["differs"] is True and rec["n_distinct"] == 2 and rec["same_window_differs_in_concurrency"] == [] and rec["rule"] == M.GEOMETRY_RULE
        assert rec["by_kind"]["prompted_constitution"]["max_model_len"] == S.MAX_MODEL_LEN_LONGSI and rec["by_kind"]["bare"]["max_model_len"] == S.MAX_MODEL_LEN
        assert rec["by_kind"]["bare"]["tree"] == T1 and rec["by_kind"]["lora_mix"]["tree"] == T2
    assert geo[f"{GLM}/{PRIMARY}"]["differs"] is False
    line = M._geometry_footnote(view)
    assert line.startswith("served L×S differs within contrast:") and f"{S.MAX_MODEL_LEN_LONGSI}×" in line and GLM not in line and "no byte claim" in line
    for stem in ("final_m5t_accuracy_by_cell", "final_m5t_contrasts", "final_m5t_sampling_sensitivity", "final_m5t_item_transitions", *M.EIGHTARM_STEMS):
        fig, _ = M.figures_for(view)[stem](view)
        assert any(t.get_text() == line for ax in fig.axes for t in ax.texts), stem
        plt.close(fig)
        assert M.GEOMETRY_RULE in caps[stem], stem
        if stem != "final_m5t_item_transitions":          # transitions are counts, not an accuracy
            assert "conditional accuracy" in caps[stem].lower(), stem
    # a same-window concurrency difference (the qwen case) is named as such
    doctored = json.loads(json.dumps(rep))
    doctored["legs"][M.leg_key(QWEN, MIX, PRIMARY)]["identity"]["served"]["max_num_seqs"] = 20
    rec = M.served_geometry_notes(doctored)[f"{QWEN}/{PRIMARY}"]
    assert rec["same_window_differs_in_concurrency"] == [S.MAX_MODEL_LEN]
    assert "two legs at the SAME window differ in concurrency" in M._disclosure_caption({**doctored, "figures": {"excluded_families": [GLM]}})
    # M4: the headline rule and the pooled word rule on the contrast captions; the notes carry both branches per contrast
    assert M.HEADLINE_RULE in caps["final_m5t_contrasts"] and M.HEADLINE_RULE in caps[M.EIGHTARM_STEM_CONTRASTS]
    assert M.POOLED_WORD_RULE in caps[M.EIGHTARM_STEM_CONTRASTS] and "no pooled verdict word" in _first_title_lines(M.EIGHTARM_FIGURES[M.EIGHTARM_STEM_CONTRASTS](view)[0])[1]
    plt.close("all")
    cvs = notes["conditional_vs_score_as_is"]
    rec = cvs["by_comparison"][f"{QWEN}/{PRIMARY}"][M.contrast_name("lora_mix")]
    assert set(rec) >= {"arm", "conditional_pp", "conditional_ci_pp", "score_as_is_pp", "score_as_is_ci_pp", "signs_agree", "n_items",
                        "arm_n_scored", "bare_n_scored", "arm_n_unscored", "arm_n_at_cap", "bare_n_unscored", "bare_n_at_cap", "word"}
    assert rec["arm"] == "lora_mix" and rec["n_items"] == S.N and rec["word"] == comparison(rep, QWEN)["verdicts"][M.contrast_name("lora_mix")]["word"]
    assert cvs["pooled"]["lora_mix"]["word"] is None and cvs["pooled_word_rule"] == M.POOLED_WORD_RULE and cvs["rule"] == M.HEADLINE_RULE
    ea = rep["eightarm"]
    assert ea["word_rule"] == M.POOLED_WORD_RULE and ea["population_rule"] == M.MASK_SCOPE_RULE
    assert all(isinstance(ea["pooled_by_arm"][k]["branches_agree_in_sign"], bool) for k in ("prompted", "lora_ctl2", "lora_mix", "object"))
    assert "branches_agree_in_sign" not in ea["pooled_by_arm"]["bare"]
    assert any("conditional on scored items (primary branch) · score-as-is" in f and "no pooled verdict word" in f for f in rep["findings"])
    tables = json.loads(Path(figs["tables"]).read_text())
    assert tables["notes"]["served_geometry"] == geo and tables["notes"]["conditional_vs_score_as_is"]["rule"] == M.HEADLINE_RULE
    # the y / x labels say what the number is
    fig, _ = M.EIGHTARM_FIGURES[M.EIGHTARM_STEM_ACCURACY](view)
    assert "conditional accuracy (scored items" in fig.axes[0].get_ylabel()
    plt.close(fig)
    fig, _ = M.EIGHTARM_FIGURES[M.EIGHTARM_STEM_CONTRASTS](view)
    assert fig.axes[0].get_xlabel().startswith("conditional-accuracy difference vs bare") and "hollow: score-as-is" in fig.axes[0].get_xlabel()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# the client-lock DISCLOSURE on every comparison                              #
# --------------------------------------------------------------------------- #

LOCK_A = ("# synthetic lock A\n"
          "inspect-ai==0.3.260 \\\n    --hash=sha256:aa\n    # via -r requirements.in\n"
          "pyyaml==6.0.3 \\\n    --hash=sha256:bb\n    # via inspect-ai\n")
LOCK_B = LOCK_A + "langdetect==1.0.9 \\\n    --hash=sha256:cc\n    # via -r requirements.in\n"          # a package the client never imports joins
LOCK_C = LOCK_A.replace("inspect-ai==0.3.260", "inspect-ai==0.3.261")                                    # an IMPORTED package moves version


def _synthetic_walk():
    from sisteer.capeval.tinycap import client_lock as CL
    walk = CL.ImportWalk(entry_points=("scripts/served_capability_run.py",), roots=("src", "scripts"))
    walk.modules = ["scripts/served_capability_run.py", "src/sisteer/capeval/tiers/inspect_served.py"]
    walk.third_party = {"inspect_ai": ["src/sisteer/capeval/tiers/inspect_served.py"], "yaml": ["scripts/served_capability_run.py"]}
    return walk


def _lock_stamped(sha: str):
    def edit(m):
        m.setdefault("frameworks", {})["client_lock_sha256"] = sha
    return edit


def _build_with_locks(root: Path, *, ext_lock_text: str, pub_lock_text: str):
    from sisteer.capeval.tinycap import client_lock as CL
    pub_sha, ext_sha = CL.lock_sha256(pub_lock_text), CL.lock_sha256(ext_lock_text)
    pub_over = {(r["model_key"], r["cell"], r["sampling_mode"]): {"manifest_edit": _lock_stamped(pub_sha)} for r in S.design_rows()}
    ext_over = {(r["model_key"], r["cell"], r["sampling_mode"]): {"manifest_edit": _lock_stamped(ext_sha)} for r in S.ext_rows()}
    rows, specs = _build_c4(root, ext_overrides=ext_over, pub_overrides=pub_over)
    return rows, specs, pub_sha, ext_sha


def _patch_lock_texts(monkeypatch, texts_by_tree):
    from sisteer.capeval.tinycap import client_lock as CL
    monkeypatch.setattr(M, "_client_lock_text", lambda tree, repo_root: (texts_by_tree[tree], None) if tree in texts_by_tree
                        else (None, f"git show {tree[:12]}:{CL.LOCK_REL} failed (synthetic: not in this clone)"))
    monkeypatch.setattr(M, "_client_lock_walk", lambda repo_root: _synthetic_walk())
    M._CLIENT_LOCK_TEXT_CACHE.clear()


def test_the_client_lock_disclosure_names_the_moved_digest_and_reads_one_instrument_when_no_imported_package_moved(tmp_path, monkeypatch):
    """The steering legs run under a client lock whose digest moved when distributions the MATH-500 client never imports
    were added. The comparison record DISCLOSES that: each leg's recorded digest, `sha256_equal`, and — through the SAME
    imported-package rule the preflight applies (client_lock.compare_locks over `git show` of each leg's producer tree,
    digest held to the recorded one) — `one_instrument: true` with the added distribution named. Nothing is refused: the
    shared projection never carried the digest, so `usable` and every gate stand. An IMPORTED package that moved version
    reads `one_instrument: false` with the package named — still a disclosure, never a refusal here (the launch-time
    preflight is the gate). An unreadable tree or a custody mismatch is UNDETERMINED by name, never read as equal."""
    rows, specs, pub_sha, ext_sha = _build_with_locks(tmp_path / "moved", ext_lock_text=LOCK_B, pub_lock_text=LOCK_A)
    _patch_lock_texts(monkeypatch, {T1: LOCK_A, T2: LOCK_B})
    rep = _analyze(specs, rows, _tree_set(monkeypatch, tmp_path))
    c = comparison(rep, GEMMA)
    assert c["usable"] is True and rep["gates"]["failed"] == []
    d = c["compatibility"]["client_lock"]
    assert d["by_leg"]["bare"] == {"sha256": pub_sha, "tree": T1} and d["by_leg"]["lora_mix"] == {"sha256": ext_sha, "tree": T2}
    assert d["distinct_sha256"] == [pub_sha, ext_sha] and d["sha256_equal"] is False and d["undetermined"] == []       # bare (published) first
    assert d["one_instrument"] is True and "no distribution the MATH-500 client imports changed version" in d["why"]
    (pair,) = d["pairs"].values()
    assert pair["one_instrument"] is True and pair["changed"] == [] and pair["n_compared"] == 2 and pair["problems"] == []
    assert pair["imported_added"] == [] and pair["outside_imported_set"]["added"] == ["langdetect"] and "ONE instrument" in pair["disclosure"]
    assert d["rule"] == M.CLIENT_LOCK_DISCLOSURE_RULE and "disclosed, never a refusal" in d["rule"]
    # a same-lock comparison: equal digests, one instrument, no pair to compare
    same = comparison(rep, GLM)["compatibility"]["client_lock"]           # GLM has no ext leg: every leg at T1 / LOCK_A
    assert same["sha256_equal"] is True and same["one_instrument"] is True and same["pairs"] == {} and same["why"] == "every leg records the same client lock"
    # the pooled arms are untouched by the disclosure (B3 only reads the comparison policy)
    assert rep["eightarm"]["pooled_by_arm"]["lora_mix"]["status"] == "available"
    # an IMPORTED distribution moved version between the two locks: named, one_instrument False, still not a refusal here
    rows, specs, _, _ = _build_with_locks(tmp_path / "bumped", ext_lock_text=LOCK_C, pub_lock_text=LOCK_A)
    _patch_lock_texts(monkeypatch, {T1: LOCK_A, T2: LOCK_C})
    rep = _analyze(specs, rows, _tree_set(monkeypatch, tmp_path))
    d = comparison(rep, GEMMA)["compatibility"]["client_lock"]
    assert d["one_instrument"] is False and "two instruments" in d["why"]
    (pair,) = d["pairs"].values()
    assert pair["one_instrument"] is False and pair["changed"] == ["inspect-ai"] and pair["problems"] == ["client_lock.inspect-ai: published 0.3.260 != wave 0.3.261 (imported by ['src/sisteer/capeval/tiers/inspect_served.py'])"]
    assert comparison(rep, GEMMA)["usable"] is True                      # a disclosure: the analysis does not refuse on it
    # an unreadable producer tree: UNDETERMINED by name, `one_instrument` None, never equal
    _patch_lock_texts(monkeypatch, {T1: LOCK_A})
    rep = _analyze(specs, rows, _tree_set(monkeypatch, tmp_path))
    d = comparison(rep, GEMMA)["compatibility"]["client_lock"]
    assert d["one_instrument"] is None and d["sha256_equal"] is False and any("cannot be read here" in u for u in d["undetermined"])
    (pair,) = d["pairs"].values()
    assert pair["one_instrument"] is None and "unreadable" in pair["why"]
    # a custody mismatch: the leg's recorded digest is not its tree's lock digest — UNDETERMINED by name
    _patch_lock_texts(monkeypatch, {T1: LOCK_A, T2: LOCK_B})               # T2's tree reads LOCK_B while the legs recorded LOCK_C's digest
    rep = _analyze(specs, rows, _tree_set(monkeypatch, tmp_path))
    d = comparison(rep, GEMMA)["compatibility"]["client_lock"]
    assert d["one_instrument"] is None and any("chain of custody" in u for u in d["undetermined"])
    # a store that predates the field (no digest at all) is UNDETERMINED by name, never equal
    rows, specs = _build_c4(tmp_path / "nolock")
    _patch_lock_texts(monkeypatch, {T1: LOCK_A, T2: LOCK_A})
    rep = _analyze(specs, rows, _tree_set(monkeypatch, tmp_path))
    d = comparison(rep, GEMMA)["compatibility"]["client_lock"]
    assert d["distinct_sha256"] == [] and d["sha256_equal"] is False and d["one_instrument"] is None
    assert all("records no client_lock_sha256" in u for u in d["undetermined"]) and len(d["undetermined"]) == len(comparison(rep, GEMMA)["leg_keys"])


# --------------------------------------------------------------------------- #
# E3: the it-doc row family — the `it` VIEW keyed apart from the main view      #
# --------------------------------------------------------------------------- #

IT = S.VIEW_IT
PIN_FILE = REPO / "configs" / "sdfbase_full22_pins.json"


def _c4_with_itdoc(root: Path, *, launch=((QWEN, CTL2),)):
    """The C4 stores plus the six it-doc rows (held, except ``launch``), the launched it-doc leg(s) built under their own
    wave root at T2 (`controls_itdoc/`, the row's own pin). Returns (matrix rows, gen-root specs)."""
    rows, specs = _build_c4(root)
    itdoc = S.itdoc_rows(len(rows), launch=launch)
    S.build_all(root / "itdoc", rows=itdoc, tree_sha=T2)
    return rows + itdoc, specs + [f"{root / 'itdoc'}={T2}"]


@pytest.fixture(scope="module")
def c4_itdoc(tmp_path_factory):
    root = tmp_path_factory.mktemp("c4_itdoc")
    rows, specs = _c4_with_itdoc(root)
    mp = pytest.MonkeyPatch()
    try:
        ts = _tree_set(mp, root)
        report = _analyze(specs, rows, ts, waves=[*WAVES, S.WAVE_ITDOC])
    finally:
        mp.undo()
    return {"root": root, "rows": rows, "specs": specs, "report": report}


def test_e3_load_cells_keys_an_it_view_row_apart_from_its_main_view_twin_and_refuses_a_twin_within_the_view():
    rows = S.eight_arm_matrix() + S.itdoc_rows(39)
    loaded = M.load_cells(rows)
    assert len(loaded) == 45
    it = [r for r in loaded if M.row_view(r) == IT]
    assert len(it) == 6 and all(M.row_status(r) == "held" and M.row_wave(r) == S.WAVE_ITDOC and M.is_itdoc_row(r) for r in it)
    main_ctl2 = next(r for r in loaded if r["model_key"] == QWEN and r["cell"] == CTL2 and M.row_view(r) == M.VIEW_MAIN)
    it_ctl2 = next(r for r in it if r["model_key"] == QWEN and r["cell"] == CTL2)
    # the two rows share (model, cell, mode) — the collision the view resolves — and are two legs with two pin keys
    assert (main_ctl2["cell"], main_ctl2["sampling_mode"]) == (it_ctl2["cell"], it_ctl2["sampling_mode"])
    assert M.row_leg_key(main_ctl2) == f"{QWEN}/{CTL2}/{PRIMARY}" and M.row_leg_key(it_ctl2) == f"{QWEN}/{CTL2}/{PRIMARY}@{IT}"
    assert M.pin_key(main_ctl2) == (QWEN, CTL2) and M.pin_key(it_ctl2) == (QWEN, CTL2, IT)
    assert it_ctl2["out_root"] == f"controls_itdoc/{QWEN}/{S.cell_slug(CTL2)}/{PRIMARY}" and main_ctl2["out_root"].startswith("controls_ext1/")
    # the same (model, cell, mode) named twice WITHIN a view refuses by the view-aware name
    with pytest.raises(M.AnalysisError, match=re.escape(f"names {QWEN}/{CTL2}/{PRIMARY}@{IT} twice")):
        M.load_cells(rows + [dict(it_ctl2)])
    with pytest.raises(M.AnalysisError, match=re.escape(f"names {QWEN}/{CTL2}/{PRIMARY} twice")):
        M.load_cells(rows + [dict(main_ctl2)])
    # the main-view rows load byte-identically with the it rows beside them
    assert M.load_cells(S.eight_arm_matrix()) == [r for r in loaded if M.row_view(r) == M.VIEW_MAIN]
    # the held it rows add nothing to the required set at any wave selection; `--waves ... itdoc` admits a launched one
    assert M.required_rows(loaded, [*WAVES, S.WAVE_ITDOC]) == M.required_rows(loaded, WAVES)
    launched = M.load_cells(S.eight_arm_matrix() + S.itdoc_rows(39, launch=[(QWEN, CTL2)]))
    req = M.required_rows(launched, [*WAVES, S.WAVE_ITDOC])
    assert [M.row_leg_key(r) for r in req if M.row_view(r) == IT] == [f"{QWEN}/{CTL2}/{PRIMARY}@{IT}"]
    assert M.required_rows(launched, WAVES) == M.required_rows(loaded, WAVES)      # not analysed unless its wave is named
    assert M.eight_arm_mode(launched, None, M.GenRoots.single("x", S.SHA7))        # a matrix carrying the it-doc wave is eight-arm


def test_e3_an_it_doc_row_is_pinned_by_its_own_sha_never_the_full22_file_and_refuses_without_one():
    rows = M.load_cells(S.eight_arm_matrix() + S.itdoc_rows(39, launch=[(QWEN, CTL2)]))
    pins = M.load_object_pins(PIN_FILE, rows)
    assert pins[(QWEN, CTL2, IT)] == S.itdoc_pin(QWEN, CTL2) and pins[(QWEN, CTL2)] == S.staged_object_pin(QWEN, f"base__{CTL2}")
    assert pins[(QWEN, CTL2, IT)] != pins[(QWEN, CTL2)]
    held = [r for r in rows if M.row_view(r) == IT and M.row_status(r) == "held"]
    assert len(held) == 5 and not any(M.pin_key(r) in pins for r in held)          # a hold is never pinned (never collected)
    launched = next(r for r in rows if M.row_view(r) == IT and M.row_status(r) == "launch")
    with pytest.raises(M.AnalysisError, match=r"@it: an it-doc row \(object_source 'itdoc', lane 'itdoc_eb64'\) is pinned by the row's own 64-hex"):
        M.load_object_pins(PIN_FILE, [dict(launched, object_sha256=None)])
    with pytest.raises(M.AnalysisError, match="never the full22 pin file"):
        M.load_object_pins(PIN_FILE, [dict(launched, object_sha256="deadbeef")])
    # the Context admits the three-tuple keying beside the published ones
    ctx_pins = M.normalise_object_pins(pins)
    assert ctx_pins[(QWEN, CTL2, IT)] == S.itdoc_pin(QWEN, CTL2) and ctx_pins[(QWEN, OBJECT)] == S.object_pin(QWEN)


def test_e3_the_design_block_files_the_held_it_view_rows_apart_and_leaves_every_main_view_status_untouched(c4_report):
    base = M.load_cells(S.eight_arm_matrix())
    rows = M.load_cells(S.eight_arm_matrix() + S.itdoc_rows(39))
    req0, req = M.required_rows(base, WAVES), M.required_rows(rows, WAVES)
    assert [M.row_leg_key(r) for r in req] == [M.row_leg_key(r) for r in req0]
    d0, d = M.design_block(base, req0, WAVES), M.design_block(rows, req, WAVES)
    assert d["arms_by_model"] == d0["arms_by_model"] == c4_report["design"]["arms_by_model"]
    main_keys = set(d0["status_by_row"])
    assert {k: d["status_by_row"][k] for k in main_keys} == d0["status_by_row"]           # every main entry byte-identical
    it_keys = set(d["status_by_row"]) - main_keys
    assert it_keys == {M.leg_key(m, c, PRIMARY, IT) for m in S.ITDOC_MODELS for c in S.ITDOC_CELLS}
    for k in it_keys:
        rec = d["status_by_row"][k]
        assert (rec["status"], rec["view"], rec["wave"], rec["reason"]) == ("held", IT, S.WAVE_ITDOC, S.ITDOC_HELD_REASON)
        assert rec["object_source"] == "itdoc" and rec["lane"] == S.ITDOC_LANE
        assert rec["qualification"] == S.ITDOC_QUALIFICATION_BY_MODEL[rec["model_key"]]
        assert rec["qualification_pattern"] == S.ITDOC_QUALIFICATION_PATTERN
    assert d["n_held"] == d0["n_held"] + 6 and d["n_required"] == d0["n_required"] and d["required_legs"] == d0["required_legs"]
    assert d0["views_in_matrix"] == ["main"] and d["views_in_matrix"] == [IT, "main"] and d0["arms_by_model_by_view"] == {}
    by_view = d["arms_by_model_by_view"][IT]
    assert set(by_view) == set(S.ITDOC_MODELS) and set(by_view[QWEN]) == {"lora_ctl2", "lora_mix", "object"}
    assert by_view[QWEN]["lora_ctl2"] == {"leg": f"{QWEN}/{CTL2}/{PRIMARY}@{IT}", "status": "held", "reason": S.ITDOC_HELD_REASON,
                                          "wave": S.WAVE_ITDOC, "object_source": "itdoc", "lane": S.ITDOC_LANE,
                                          "qualification": S.ITDOC_QUALIFICATION_BY_MODEL[QWEN],
                                          "qualification_pattern": S.ITDOC_QUALIFICATION_PATTERN}
    # every other design leaf is the one without the it rows
    # the it rows ADD their wave to the matrix's wave inventory (the reader lists the waves present) — every other leaf is unchanged
    assert d["waves_in_matrix"] == sorted(set(d0["waves_in_matrix"]) | {S.WAVE_ITDOC}) and S.WAVE_ITDOC not in d0["waves_in_matrix"]
    skip = ("status_by_row", "n_held", "n_rows", "views_in_matrix", "arms_by_model_by_view", "waves_in_matrix")
    same = {k: v for k, v in d.items() if k not in skip}
    assert same == {k: v for k, v in d0.items() if k not in skip}
    # the comparisons over the required rows are unchanged (every it row held: no it group)
    assert M.comparisons_from(req, eight_arm=True) == M.comparisons_from(req0, eight_arm=True)
    # an it group with no main-view bare row to anchor on refuses by name; outside eight-arm mode an it group refuses
    it_launch = [r for r in M.load_cells(S.itdoc_rows(0, launch=[(QWEN, CTL2)])) if M.row_status(r) == "launch"]
    with pytest.raises(M.AnalysisError, match=re.escape(f"{QWEN}/{PRIMARY}@{IT}: the 'it' view names ['lora_ctl2'] and no bare cell")):
        M.comparisons_from(it_launch, eight_arm=True)
    with pytest.raises(M.AnalysisError, match="eight-arm mode only"):
        M.comparisons_from(it_launch, eight_arm=False)


def test_e3_one_launched_it_doc_leg_is_read_against_the_main_bare_anchor_and_carries_its_qualification_on_every_reading(c4_itdoc, c4_report, tmp_path):
    rep = c4_itdoc["report"]
    it_key, main_key, anchor = f"{QWEN}/{CTL2}/{PRIMARY}@{IT}", f"{QWEN}/{CTL2}/{PRIMARY}", f"{QWEN}/{BARE}/{PRIMARY}"
    qual, pattern = S.ITDOC_QUALIFICATION_BY_MODEL[QWEN], S.ITDOC_QUALIFICATION_PATTERN
    assert M.production_refusals(rep) == [] and rep["verdict"]["word"] == "filed" and rep["design"]["waves"] == [*WAVES, S.WAVE_ITDOC]
    # the two ctl2 legs are read apart, each against its OWN pin; the it leg carries the carriage, the main leg does not
    it_leg, main_leg = rep["legs"][it_key], rep["legs"][main_key]
    assert it_leg["missing"] is None and it_leg["refused"] is None and it_leg["view"] == IT and main_leg["view"] == "main"
    assert it_leg["identity"]["object_artifact_sha256"] == S.itdoc_pin(QWEN, CTL2) != main_leg["identity"]["object_artifact_sha256"]
    assert it_leg["identity_validated"] and (it_leg["object_source"], it_leg["lane"]) == ("itdoc", S.ITDOC_LANE)
    assert it_leg["qualification"] == qual and it_leg["qualification_pattern"] == pattern
    assert not any(f in main_leg for f in M.ITDOC_CARRIAGE_FIELDS)
    # the it comparison: anchored on the MAIN view's bare leg, recorded with the rationale, never silently
    c = next(c for c in rep["comparisons"] if c["name"] == f"{QWEN}/{PRIMARY}@{IT}")
    assert c["view"] == IT and c["primary"] and c["usable"] and c["estimands"] is not None
    assert c["leg_keys"] == {"bare": anchor, "lora_ctl2": it_key} and c["arms_absent"] == [k for k in PANEL_KINDS if k not in ("bare", "lora_ctl2")]
    assert c["bare_anchor"] == {"view": "main", "leg": anchor, "rationale": M.BARE_ANCHOR_RATIONALE}
    assert c["itdoc_arms"] == {"lora_ctl2": {"leg": it_key, "object_source": "itdoc", "lane": S.ITDOC_LANE, "qualification": qual,
                                             "qualification_pattern": pattern}}
    word = c["verdicts"]["lora_ctl2_minus_bare"]["word"]
    assert list(c["verdicts"]) == ["lora_ctl2_minus_bare"] and word in (*M.WORDS, M.WORD_NOT_RESOLVED)      # a one-leg synthetic store may not resolve at the cap
    assert c["compatibility"]["cross_tree"] and c["compatibility"]["declared"]        # T2 arm vs the published T1 anchor, admitted
    # the main-view comparison, estimands, words and pooled group are the c4 run's, number for number
    main_c, c4_c = comparison(rep, QWEN), comparison(c4_report, QWEN)
    assert "view" not in main_c and "bare_anchor" not in main_c and main_c["leg_keys"] == c4_c["leg_keys"]
    assert main_c["estimands"] == c4_c["estimands"] and main_c["verdicts"] == c4_c["verdicts"]
    assert rep["estimands"][QWEN][PRIMARY] == c4_report["estimands"][QWEN][PRIMARY]
    assert rep["estimands"][QWEN][f"{PRIMARY}@{IT}"] == c["estimands"]
    assert rep["verdict"]["words"] == c4_report["verdict"]["words"] and rep["verdict"]["verdict"] == c4_report["verdict"]["verdict"]
    assert rep["verdict"]["view_words"] == {IT: {QWEN: {"lora_ctl2_minus_bare": word}}} and "view_words" not in c4_report["verdict"]
    assert rep["eightarm"]["pooled_by_arm"]["lora_ctl2"] == c4_report["eightarm"]["pooled_by_arm"]["lora_ctl2"]
    # the it contrast is its own reading (its correct set differs from the main ctl2 leg's), and no mode delta pairs it
    it_pt = c["estimands"][M.BRANCH_PRIMARY]["stats"]["lora_ctl2_minus_bare"]["point"]
    assert it_pt != main_c["estimands"][M.BRANCH_PRIMARY]["stats"]["lora_ctl2_minus_bare"]["point"]
    assert not any("@" in d["name"] for d in rep["mode_delta_records"]) and set(rep["mode_deltas"][QWEN]) == set(c4_report["mode_deltas"][QWEN])
    # the design block: the launched it row required, the five others held apart; the main slots as without them
    d = rep["design"]
    assert d["status_by_row"][it_key]["status"] == "required" and d["arms_by_model"] == c4_report["design"]["arms_by_model"]
    assert d["arms_by_model_by_view"][IT][QWEN]["lora_ctl2"]["status"] == "required"
    assert sum(1 for r in d["status_by_row"].values() if r.get("view") == IT and r["status"] == "held") == 5
    assert it_key in d["required_legs"] and {k: v for k, v in d["status_by_row"].items() if "@" not in k} == c4_report["design"]["status_by_row"]
    # the records: the controls row, the provenance row and the truncation cell carry the view and the carriage; main rows do not
    ctl = {r["leg"]: r for r in rep["controls"]}
    assert ctl[it_key]["view"] == IT and ctl[it_key]["qualification"] == qual and ctl[it_key]["lane"] == S.ITDOC_LANE
    assert "view" not in ctl[main_key] and "qualification" not in ctl[main_key]
    prov = rep["provenance"]["cells"]
    assert prov[it_key]["view"] == IT and prov[it_key]["qualification"] == qual and prov[it_key]["qualification_pattern"] == pattern
    assert prov[it_key]["lane"] == S.ITDOC_LANE and prov[it_key]["object_artifact_sha256"] == S.itdoc_pin(QWEN, CTL2)
    assert prov[main_key]["view"] == "main" and "qualification" not in prov[main_key]
    cells = {x["leg"]: x for x in rep["truncation"]["cells"]}
    assert cells[it_key]["view"] == IT and cells[it_key]["primary"] and "view" not in cells[main_key]
    # the findings name the anchor rule and carry the qualification verbatim
    f = next(f for f in rep["findings"] if f.startswith(f"{QWEN}/{PRIMARY}@{IT}:"))
    assert anchor in f and M.BARE_ANCHOR_RATIONALE in f and qual in f and pattern in f and f"lora_ctl2_minus_bare {word}" in f
    # the captions: every figure whose shown legs include the it-doc leg ends with the qualification verbatim; no other does
    caps = M.figure_captions(rep)
    sentence = f" it-doc re-fit ({QWEN} {IT}: lora_ctl2): {qual} {pattern}"
    assert set(caps) == set(M.FIGURES) | set(M.EIGHTARM_STEMS)
    assert all(caps[stem].endswith(sentence) for stem in M.ITDOC_CAPTION_FIGURES)
    assert not any("it-doc" in caps[stem] for stem in set(caps) - set(M.ITDOC_CAPTION_FIGURES))
    assert not any("it-doc" in v for v in M.figure_captions(c4_report).values())
    assert M.itdoc_caption(c4_report) == "" and M.itdoc_legs_shown(rep) == {(QWEN, IT): [ctl[it_key]]}
    # a render that leaves qwen out shows no it-doc leg and carries no sentence
    assert M.itdoc_caption({**rep, "figures": {M.FIGURES_EXCLUDED_KEY: [QWEN]}}) == ""
    # the figures render; the contrasts figure labels the it row by its view beside the main row; the eight-arm figures draw the
    # main view only; the publication gate passes
    figs = M.make_figures(rep, tmp_path / "f")
    assert figs["skipped"] == {} and set(figs["written"]) == set(M.FIGURES) | set(M.EIGHTARM_STEMS)
    con = _entries(figs, "final_m5t_contrasts")
    names = {e["name"] for e in con}
    assert any(n.startswith(f"{QWEN} [{IT}]: lora_ctl2_minus_bare.") for n in names) and any(n.startswith(f"{QWEN}: lora_ctl2_minus_bare.") for n in names)
    assert any(e["source"] == f"estimands.{QWEN}.{PRIMARY}@{IT}.primary.stats.lora_ctl2_minus_bare.point" for e in con)
    assert not any(f"[{IT}]" in e["name"] for stem in M.EIGHTARM_STEMS for e in _entries(figs, stem))
    assert not any(f"[{IT}]" in e["name"] for e in _entries(figs, "final_m5t_item_transitions"))
    assert any(e["name"] == f"at_cap.share[{it_key}]" for e in _entries(figs, "final_m5t_atcap_share"))
    assert json.loads(Path(figs["numbers"]).read_text())["captions"]["final_m5t_contrasts"].endswith(sentence)
    M.publication_gate_record(rep, figs, tmp_path / "f")


@pytest.fixture
def raw_launcher_cells():
    """The launcher's matrix module (``scripts/sdfbase/math500/cells.py``) executed FRESH under a throwaway name, so a test can
    read ``M5T_CELLS`` raw — including a row ``load_cells`` refuses. Registered in ``sys.modules`` for the exec (the launcher
    defines a dataclass, and under ``from __future__ import annotations`` dataclasses resolves its string annotations through
    ``sys.modules[cls.__module__].__dict__``, which is None for an unregistered module) and REMOVED on teardown, so the
    registration cannot outlive the test that asked for it and cannot reach a neighbour in the same process."""
    spec = importlib.util.spec_from_file_location("m5t_cells_raw_for_test", REPO / "scripts" / "sdfbase" / "math500" / "cells.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop(spec.name, None)


def test_e3_the_launchers_cells_py_admits_all_53_rows_and_files_rows_47_to_52_as_held_it_view_rows(raw_launcher_cells):
    """The launcher's 53 main / it-view rows, unchanged — plus the all-positions row (index 53), now READ like any other.

    That row is an EVIDENCE (`launch`) row of `arditi_allpos`. Until the cell's Class B registration landed, the tree's GRAMMAR
    refused the id and `load_cells` refused the whole live matrix by name — the reader could not read this arm's legs at all, and
    that refusal was pinned here as a tripwire rather than worked around. The registration landed, so the tripwire has stopped
    firing and the row joins the set: `load_cells` now reads all 54 rows, and the assertions below are the POSITIVE form of what
    the refusal used to stand in for. The published 53 are untouched by its arrival, which is the property worth keeping.
    """
    src = REPO / "scripts" / "sdfbase" / "math500" / "cells.py"
    raw = list(raw_launcher_cells.M5T_CELLS)
    allpos = [r for r in raw if r["cell"] == "arditi_allpos"]
    assert len(allpos) == 1 and allpos[0]["index"] == 53 and allpos[0]["status"] == "launch" and allpos[0]["view"] == IT
    # the grammar admits the id, so the LIVE matrix loads — all 54 rows, the published 53 plus this one
    assert C.cell_spellings("arditi_allpos", declared=False).harness == "arditi_allpos"
    live = M.load_cells(src)
    assert len(live) == 54 and [r["cell"] for r in live if r["index"] == 53] == ["arditi_allpos"]
    # its leg key is the `it`-view one (the arm is `view_keyed`), and it is NOT the prefill-only arm's key
    assert M.row_leg_key(allpos[0]) == f"{QWEN}/arditi_allpos/{PRIMARY}@{IT}" and M.pin_key(allpos[0]) == (QWEN, "arditi_allpos", IT)
    rows = M.load_cells([r for r in raw if r["cell"] != "arditi_allpos"])
    assert len(rows) == 53
    it = [r for r in rows if M.key_view(r) != M.VIEW_MAIN]
    assert [r["index"] for r in it] == list(range(47, 53))
    # the PUBLISHED external arms' rows (the `steer` wave) name their object bank's view (`it`, `ar_...`) as an IDENTITY key and
    # stay keyed main: one AlphaSteer / Arditi slot per model, no main-view twin — the view keys a row apart on the LoRA arms,
    # and on the all-positions arm, whose row genuinely sits on the `it` paper view (asserted above)
    ext = [r for r in rows if M.ARM_BY_CELL[r["cell"]].external]
    assert ext and any(M.row_view(r) != M.VIEW_MAIN for r in ext)
    assert all(M.key_view(r) == M.VIEW_MAIN and "@" not in M.row_leg_key(r) and len(M.pin_key(r)) == 2 for r in ext)
    assert all(r["view"] == IT and M.row_wave(r) == S.WAVE_ITDOC and M.row_status(r) == "held" and r["held_reason"] for r in it)
    assert all(M.is_itdoc_row(r) and r["lane"] == "itdoc_eb64" and r["qualification"] and r["qualification_pattern"] for r in it)
    assert [(r["model_key"], r["cell"]) for r in it] == [(m, c) for m in (GEMMA, QWEN) for c in (CTL2, MIX, OBJECT)]
    main_keys = {M.row_leg_key(r) for r in rows if M.key_view(r) == M.VIEW_MAIN}
    for r in it:        # each shares its triple with a main-view row and is keyed apart by its view
        assert M.leg_key(r["model_key"], r["cell"], r["sampling_mode"]) in main_keys and M.row_leg_key(r).endswith(f"@{IT}")
        assert r["out_root"] == f"controls_itdoc/{r['model_key']}/{S.cell_slug(r['cell'])}/{PRIMARY}"
    # the synthetic it-doc rows carry exactly the launcher's key set
    assert set(S.itdoc_rows(47)[0]) == set(rows[47])
    # the design block over the launcher's matrix: rows 47-52 held apart, every main-view entry as without them
    waves = [S.WAVE_PUBLISHED, S.WAVE_EXT1]
    req = M.required_rows(rows, waves)
    d = M.design_block(rows, req, waves)
    d0 = M.design_block([r for r in rows if M.key_view(r) == M.VIEW_MAIN], req, waves)
    assert d["arms_by_model"] == d0["arms_by_model"] and {k: v for k, v in d["status_by_row"].items() if "@" not in k} == d0["status_by_row"]
    assert set(d["arms_by_model_by_view"][IT]) == {GEMMA, QWEN} and d["n_held"] == d0["n_held"] + 6
    for m in (GEMMA, QWEN):
        for kind in ("lora_ctl2", "lora_mix", "object"):
            rec = d["arms_by_model_by_view"][IT][m][kind]
            assert rec["status"] == "held" and rec["lane"] == "itdoc_eb64" and rec["reason"] and rec["qualification"] and rec["qualification_pattern"]
    # a held it-doc row is never pinned, whatever sha it carries (held rows are not evidence)
    assert not any(len(k) == 3 for k in M.load_object_pins(PIN_FILE, rows))
    assert M.comparisons_from(req, eight_arm=True) == M.comparisons_from([r for r in req if M.key_view(r) == M.VIEW_MAIN], eight_arm=True)
    # the live steer rows (external arms, non-main bank views) sit in their model's MAIN group and keep their panel slot
    steer_waves = [S.WAVE_PUBLISHED, S.WAVE_EXT1, "steer", "llama"]        # llama's steer rows need llama's bare row (its own wave)
    req_steer = M.required_rows(rows, steer_waves)
    comps = M.comparisons_from(req_steer, eight_arm=True)
    assert not any("@" in c["name"] for c in comps)
    gemma = next(c for c in comps if c["model_key"] == GEMMA and c["mode"] == PRIMARY)
    assert gemma["leg_keys"].get("alphasteer") == f"{GEMMA}/alphasteer/{PRIMARY}" and gemma["leg_keys"].get("arditi") == f"{GEMMA}/arditi/{PRIMARY}"
    d_steer = M.design_block(rows, req_steer, steer_waves)
    assert d_steer["arms_by_model"][GEMMA]["alphasteer"] == "required" and d_steer["arms_by_model"][GEMMA]["arditi"] == "required"


# --------------------------------------------------------------------------- #
# launch-attestation-native: the launch record is selected by the leg's ENGINE, never by which file exists                #
# --------------------------------------------------------------------------- #
#: a natively served Llama PSR leg's geometry (the real cards': 10,752 x 44) and its served cell's harness name
L_NATIVE, S_NATIVE = 10752, 44
NATIVE_CELL = "si_constv2_mix_full__psr_ce_all"
_UNSET = object()


def _native_launch_leg(root, *, job="500476", engine=C.SERVING_ENGINE_LORA_NATIVE, top_engine=_UNSET, manifest_job=None, log_requests=False,
                       args_edit=None, record_edit=None, max_model_len=L_NATIVE, max_num_seqs=S_NATIVE):
    """A natively served leg's standup record under ``root`` — the STRICT native serving attestation and the server log
    carrying the native engine's echo (``LoRAModulePath`` reprs, ``enable_lora``, no ``enforce_eager``), NO lens record —
    and a manifest naming it (``route.serving_engine`` ``engine``; ``_UNSET`` omits the key; ``top_engine`` the top-level
    stamp; ``manifest_job`` the ``row_constants.slurm_job_id``). ``args_edit`` mutates the echoed args, ``record_edit`` the
    written attestation (re-serialised as is: the strict reader is what refuses it). Returns ``(manifest, standup dir, args)``."""
    jobdir = f"m5t_llama8b_consmix-psr_primary_{job}"
    port = S.launch_port(job)
    specs = S.native_adapter_specs("llama8b", NATIVE_CELL, intervention_hash="ed91ee5a98cd01de", source_kind="psr")
    _hf, rev = S.model_pin("llama8b")
    args = S.launch_echo_args("llama8b", revision=rev, port=port, max_model_len=max_model_len, max_num_seqs=max_num_seqs, reasoning_parser=None,
                              log_requests=log_requests, jobdir=jobdir, native=True, lora_modules=S.native_lora_modules(specs))
    if args_edit:
        args_edit(args)
    d = S.write_launch_record(root, jobdir, job=job, port=port, args=args, generation_config=S.GENERATION_CONFIG_BY_KEY["llama8b"], tree_sha="1" * 40,
                              model_key="llama8b", max_model_len=max_model_len, max_num_seqs=max_num_seqs, log_requests=log_requests, native=True, adapters=specs)
    if record_edit:
        att = json.loads((d / S.STANDUP_SERVING_ATTESTATION).read_text())
        record_edit(att)
        (d / S.STANDUP_SERVING_ATTESTATION).write_text(json.dumps(att, indent=1), encoding="utf-8")
    manifest = {"routing_table": S.cluster_routing_table_path(jobdir), C.MANIFEST_KEY_ROUTE: {"serving_identity": "proxy_vllm+lora+interleaved"},
                C.MANIFEST_KEY_TINYCAP: {"served": {"max_model_len": max_model_len, "max_num_seqs": max_num_seqs}}}
    if engine is not _UNSET:
        manifest[C.MANIFEST_KEY_ROUTE][C.SERVED_ENGINE_FIELD] = engine
    if top_engine is not _UNSET:
        manifest[C.SERVED_ENGINE_FIELD] = top_engine
    if manifest_job is not None:
        manifest["row_constants"] = {"slurm_job_id": manifest_job}
    return manifest, d, args


def _lens_launch_leg(root, *, job="480123", engine=_UNSET, model_key="gemma4e4b", reasoning_parser=_UNSET):
    """A lens leg's standup record (the workers' attestation + the lens echo) and its manifest — the historical record
    (no engine stamp) unless ``engine`` is given; the reasoning parser the family's (none for the no-switch Llama)."""
    jobdir = S.standup_jobdir(model_key, "none", "primary", job)
    port = S.launch_port(job)
    _hf, rev = S.model_pin(model_key)
    L, seqs = S.MAX_MODEL_LEN, S.served_seqs(model_key, S.MAX_MODEL_LEN)
    parser = (None if model_key == "llama8b" else "gemma") if reasoning_parser is _UNSET else reasoning_parser
    args = S.launch_echo_args(model_key, revision=rev, port=port, max_model_len=L, max_num_seqs=seqs, reasoning_parser=parser, jobdir=jobdir)
    d = S.write_launch_record(root, jobdir, job=job, port=port, args=args, generation_config=S.GENERATION_CONFIG_BY_KEY[model_key], tree_sha="1" * 40,
                              model_key=model_key, max_model_len=L, max_num_seqs=seqs)
    manifest = {"routing_table": S.cluster_routing_table_path(jobdir), C.MANIFEST_KEY_ROUTE: {"serving_identity": "proxy_vllm+passthrough+interleaved"},
                C.MANIFEST_KEY_TINYCAP: {"served": {"max_model_len": L, "max_num_seqs": seqs}}}
    if engine is not _UNSET:
        manifest[C.MANIFEST_KEY_ROUTE][C.SERVED_ENGINE_FIELD] = engine
    return manifest, d, args


def test_the_launch_record_is_selected_by_the_legs_engine_and_a_native_leg_attests_through_the_strict_record(tmp_path):
    """launch-attestation-native: `launch_attestation` read ONLY the lens workers' attestation (`steering_attestation.json`),
    so every natively served leg — whose standup writes `serving_attestation.json` INSTEAD, by design, beside the same
    `vllm_server.log` — read `absent` with "no workers' attestation" and the GSM8K contrast refused it (the primary
    cluster's three Llama PSR cards). Now the record is selected by the manifest's `route.serving_engine`: `lora_native` ->
    the native serving attestation read through the STRICT reader, bound by job id and port, the native echo projected
    (`enable_lora`, the LoRA config, `lora_modules` as `{"LoRAModulePath": {...}}` — the one non-literal the echo carries —
    and NO `enforce_eager`, graphs on); the record says which file and engine attested. A lens leg (an absent stamp — the
    historical record — or an explicit `lens`) reads exactly as before."""
    assert M.LAUNCH_RECORD_FILE_BY_ENGINE == {C.SERVING_ENGINE_LENS: "steering_attestation.json", C.SERVING_ENGINE_LORA_NATIVE: "serving_attestation.json"}
    assert set(M.LAUNCH_RECORD_FILE_BY_ENGINE) == set(C.SERVING_ENGINES)
    assert M.LAUNCH_NATIVE_ATTESTATION_FILE == SA.SERVING_ATTESTATION_FILE == S.STANDUP_SERVING_ATTESTATION and M.LAUNCH_ATTESTATION_FILE == S.STANDUP_ATTESTATION
    assert "steering_attestation.json | serving_attestation.json" in M.LAUNCH_REMEDY and "pull_wave.sh carries all three names" in M.LAUNCH_REMEDY
    root = tmp_path / "native"
    manifest, d, args = _native_launch_leg(root, manifest_job="500476")
    # a native standup writes ONE engine's record: the strict serving attestation, never the lens file
    assert sorted(p.name for p in d.iterdir()) == sorted([S.STANDUP_SERVING_ATTESTATION, S.STANDUP_JOBS, S.STANDUP_ROUTING_TABLE, S.LAUNCH_LOG])
    record = json.loads((d / S.STANDUP_SERVING_ATTESTATION).read_text())
    assert set(record) == set(SA.REQUIRED_TOP_KEYS_V2) and SA.read_serving_attestation(d / S.STANDUP_SERVING_ATTESTATION)["kind"] == C.SERVING_ENGINE_LORA_NATIVE
    out = M.launch_attestation(manifest, gen_root=root)
    assert out["status"] == M.LAUNCH_ATTESTED and out["problems"] == [], out["problems"]
    assert out["engine"] == C.SERVING_ENGINE_LORA_NATIVE and out["record_file"] == M.LAUNCH_NATIVE_ATTESTATION_FILE
    assert out["job_id"] == "500476" == record["slurm_job_id"] and out["argv_sha256"] == record["argv_sha256"] and len(out["argv_sha256"]) == 64
    assert out["geometry_reconciled"] is True and out["generation_config_override"]["present"] is True and out["log_requests_attested"] is False
    assert out["serving_config_record"] == M.LAUNCH_SERVING_CONFIG_ABSENT
    settings, excluded = out["settings"], out["excluded"]
    # the LoRA KERNEL configuration is compared (O106: it applies on every request, the zero adapter included) ...
    assert json.loads(settings["enable_lora"]) is True and json.loads(settings["max_lora_rank"]) == 1 and json.loads(settings["lora_target_modules"]) == ["o_proj"]
    assert "enforce_eager" not in settings and "enforce_eager" not in excluded        # graphs on: the native launch never echoes the flag
    # ... and the CELL-SET plumbing is recorded under `excluded` with O106's reason, never compared on a same-engine contrast
    for key in ("lora_modules", "max_loras", "max_cpu_loras"):
        assert key not in settings and excluded[key]["why"] == M.LAUNCH_CELL_SET_PLUMBING_WHY and "O106" in excluded[key]["why"], key
    assert json.loads(excluded["max_loras"]["value"]) == 2 == json.loads(excluded["max_cpu_loras"]["value"])
    modules = json.loads(excluded["lora_modules"]["value"])
    assert [m["LoRAModulePath"]["name"] for m in modules] == [C.TWIN_CELL_ID_LORA, NATIVE_CELL] and all(set(m) == {"LoRAModulePath"} for m in modules)
    assert set(modules[0]["LoRAModulePath"]) == {"name", "path", "base_model_name", "is_3d_lora_weight"}
    assert modules[1]["LoRAModulePath"]["path"].endswith(f"/{NATIVE_CELL}") and modules[0]["LoRAModulePath"]["base_model_name"] is None
    assert json.loads(excluded["worker_extension_cls"]["value"]) == S.NATIVE_WORKER_EXTENSION and json.loads(excluded["port"]["value"]) == S.launch_port("500476")
    assert "route.serving_engine" in out["rule"] and "never by which file exists" in out["rule"] and "read strictly" in out["rule"]
    proj = M.server_projection({"launch": out, "identity": {}})
    assert proj["launch.status"] == M.LAUNCH_ATTESTED and "launch.enable_lora" in proj and "launch.max_lora_rank" in proj
    assert "launch.lora_modules" not in proj and "launch.max_loras" not in proj and "launch.port" not in proj
    # the SAME leg with the top-level stamp agreeing, and with a manifest that names no job: attested, the record's job read
    agreeing = json.loads(json.dumps(manifest)); agreeing[C.SERVED_ENGINE_FIELD] = C.SERVING_ENGINE_LORA_NATIVE
    assert M.launch_attestation(agreeing, gen_root=root)["status"] == M.LAUNCH_ATTESTED
    del manifest["row_constants"]
    out = M.launch_attestation(manifest, gen_root=root)
    assert out["status"] == M.LAUNCH_ATTESTED and out["job_id"] == "500476"
    # the LENS leg, unchanged: the workers' attestation selected by an absent stamp (every published store) or an explicit `lens`
    for engine in (_UNSET, C.SERVING_ENGINE_LENS):
        lroot = tmp_path / f"lens_{engine if engine is not _UNSET else 'absent'}"
        lmanifest, ld, largs = _lens_launch_leg(lroot, engine=engine)
        assert sorted(p.name for p in ld.iterdir()) == sorted([S.STANDUP_ATTESTATION, S.STANDUP_JOBS, S.STANDUP_ROUTING_TABLE, S.LAUNCH_LOG])
        lout = M.launch_attestation(lmanifest, gen_root=lroot)
        assert lout["status"] == M.LAUNCH_ATTESTED and lout["problems"] == [], lout["problems"]
        assert lout["engine"] == C.SERVING_ENGINE_LENS and lout["record_file"] == M.LAUNCH_ATTESTATION_FILE == "steering_attestation.json"
        assert json.loads(lout["settings"]["enforce_eager"]) is True and "enable_lora" not in lout["settings"] and "lora_modules" not in lout["excluded"]
        assert json.loads(lout["excluded"]["worker_extension_cls"]["value"]) == S.LENS_WORKER_EXTENSION
    # the two engines' echoes differ by construction — the first flag a native-vs-lens projection names is the eager flag
    diff = M._first_difference({"lens": M.server_projection({"launch": lout, "identity": {}}, override=False),
                                "native": M.server_projection({"launch": M.launch_attestation(manifest, gen_root=root), "identity": {}}, override=False)})
    assert diff is not None and diff.startswith("launch.enforce_eager: ") and "'lens': 'true'" in diff and "'native': None" in diff


def test_a_native_leg_refuses_by_name_on_the_wrong_record_a_torn_dir_a_broken_binding_and_a_non_strict_record(tmp_path):
    """Known-bad, each refused by NAME with `status: absent` and never a lenient read: (a) a native manifest over a standup
    dir carrying only the LENS record — the native record named as missing (with the remedy) AND the lens record named as
    the other engine's (a torn dir); (b) both records beside one log, whichever engine the manifest states; (c) a LENS
    manifest over a native-only dir (the mirror of the real gap); (d) a record whose `slurm_job_id` disagrees with the
    manifest's or the standup dir's, an echo whose `port` or geometry is another launch's; (e) a record the STRICT reader
    refuses — a lens `kind`, an eager engine, an inferred name binding, a missing schema key, an unknown schema version —
    named through the reader's own words."""
    # (a) only the lens file, under the lens NAME, with the native bytes inside: the engine selects, the file name does not
    root = tmp_path / "lens_only"
    manifest, d, _ = _native_launch_leg(root)
    (d / S.STANDUP_SERVING_ATTESTATION).rename(d / S.STANDUP_ATTESTATION)
    out = M.launch_attestation(manifest, gen_root=root)
    assert out["status"] == M.LAUNCH_ABSENT and out["engine"] == C.SERVING_ENGINE_LORA_NATIVE and out["record_file"] == "serving_attestation.json"
    assert out["settings"] is None and out["job_id"] is None
    missing = [p for p in out["problems"] if p.startswith(f"{d / 'serving_attestation.json'}: no serving attestation (serving_attestation.json, the lora_native engine's launch record)")]
    torn = [p for p in out["problems"] if p.startswith(f"{d / 'steering_attestation.json'}: the lens engine's launch record beside a lora_native leg's standup")]
    assert len(missing) == 1 and "pull the job's serving/" in missing[0] and "serving_attestation.json" in missing[0]
    assert len(torn) == 1 and "torn / mixed standup dir" in torn[0] and "route.serving_engine reads 'lora_native'" in torn[0] and len(out["problems"]) == 2
    # (b) BOTH records beside one log: refused by name even though the selected record is strict-valid — under either manifest
    root = tmp_path / "both"
    manifest, d, _ = _native_launch_leg(root)
    (d / S.STANDUP_ATTESTATION).write_text(json.dumps({"slurm_job_id": "500476", "port": S.launch_port("500476"), "argv_sha256": "0" * 16}))
    out = M.launch_attestation(manifest, gen_root=root)
    assert out["status"] == M.LAUNCH_ABSENT and len(out["problems"]) == 1 and "the lens engine's launch record beside a lora_native leg's standup" in out["problems"][0]
    lens_manifest = json.loads(json.dumps(manifest))
    del lens_manifest[C.MANIFEST_KEY_ROUTE][C.SERVED_ENGINE_FIELD]
    out = M.launch_attestation(lens_manifest, gen_root=root)
    assert out["status"] == M.LAUNCH_ABSENT and out["engine"] == C.SERVING_ENGINE_LENS and out["record_file"] == "steering_attestation.json"
    assert len(out["problems"]) == 1 and out["problems"][0].startswith(f"{d / 'serving_attestation.json'}: the lora_native engine's launch record beside a lens leg's standup")
    # (c) a LENS manifest over the native-only dir (the real cards' standups, read with the OLD assumption): no workers'
    # attestation, and the native record named as the other engine's — never read as the lens record
    root = tmp_path / "lens_over_native"
    manifest, d, _ = _native_launch_leg(root)
    lens_manifest = json.loads(json.dumps(manifest))
    del lens_manifest[C.MANIFEST_KEY_ROUTE][C.SERVED_ENGINE_FIELD]
    out = M.launch_attestation(lens_manifest, gen_root=root)
    assert out["status"] == M.LAUNCH_ABSENT and len(out["problems"]) == 2
    assert any(p.startswith(f"{d / 'steering_attestation.json'}: no workers' attestation (steering_attestation.json, the lens engine's launch record)") for p in out["problems"])
    assert any("the lora_native engine's launch record beside a lens leg's standup" in p for p in out["problems"])
    # (d) the binding: the manifest's job, the record's job vs the dir, the echo's port, the echo's geometry
    root = tmp_path / "manifest_job"
    manifest, d, _ = _native_launch_leg(root, manifest_job="1")
    out = M.launch_attestation(manifest, gen_root=root)
    assert out["status"] == M.LAUNCH_ABSENT and out["problems"] == ["the manifest's row_constants.slurm_job_id '1' != the attestation's '500476'"]
    root = tmp_path / "record_job"
    manifest, d, _ = _native_launch_leg(root, record_edit=lambda a: a.update(slurm_job_id="1"))
    out = M.launch_attestation(manifest, gen_root=root)
    assert out["status"] == M.LAUNCH_ABSENT and any("slurm_job_id '1' != the standup directory's job '500476' — the attestation is another job's" in p for p in out["problems"])
    root = tmp_path / "port"
    manifest, d, _ = _native_launch_leg(root, args_edit=lambda a: a.update(port=1))
    out = M.launch_attestation(manifest, gen_root=root)
    assert out["status"] == M.LAUNCH_ABSENT and any(f"the echo's port 1 != the attestation's {S.launch_port('500476')} — the log is another launch's" in p for p in out["problems"])
    root = tmp_path / "geometry"
    manifest, d, _ = _native_launch_leg(root, args_edit=lambda a: a.update(max_num_seqs=1, max_num_batched_tokens=L_NATIVE))
    out = M.launch_attestation(manifest, gen_root=root)
    assert out["status"] == M.LAUNCH_ABSENT and out["geometry_reconciled"] is False
    assert any(f"the echo's max_num_seqs 1 != the manifest's served max_num_seqs {S_NATIVE}" in p for p in out["problems"])
    # (e) the STRICT reader's refusals, each by name — a lenient reader would have read every one of these as a record
    cases = [("kind", lambda a: a.update(kind=C.SERVING_ENGINE_LENS), "kind='lens'"),
             ("eager", lambda a: a["engine"].update(enforce_eager=True), "enforce_eager=True"),
             ("inferred", lambda a: a["name_binding"].update(asserted=False), "name_binding.asserted is not true"),
             ("missing", lambda a: a.pop("worker_adapters"), "lacks required keys ['worker_adapters']"),
             ("schema", lambda a: a.update(schema_version=3), "schema_version=3"),
             ("idmap", lambda a: a["id_name_map"].update({"1": NATIVE_CELL, "2": C.TWIN_CELL_ID_LORA}), "id_name_map"),
             ("notjson", None, "is not valid JSON")]
    for name, edit, word in cases:
        root = tmp_path / f"strict_{name}"
        manifest, d, _ = _native_launch_leg(root, record_edit=edit)
        if edit is None:
            (d / S.STANDUP_SERVING_ATTESTATION).write_text("{not json", encoding="utf-8")
        out = M.launch_attestation(manifest, gen_root=root)
        assert out["status"] == M.LAUNCH_ABSENT and out["job_id"] is None and out["settings"] is None, name
        assert len(out["problems"]) == 1 and out["problems"][0].startswith(f"{d / 'serving_attestation.json'}: not a strict lora_native serving attestation — ") and word in out["problems"][0], (name, out["problems"])


def test_the_engine_stamp_is_read_fail_closed_a_present_unknown_or_torn_stamp_selects_no_record(tmp_path):
    """The selection reads the STORED-RECORD contract: an absent `route.serving_engine` is the lens engine (every published
    store), `lens` and `lora_native` name theirs; a PRESENT value outside the two (a null, a typo, an unknown engine) or a
    top-level stamp disagreeing with the route's selects NO record — `engine` / `record_file` None, the problem named —
    rather than falling to whichever file exists."""
    assert M.launch_engine_of({}) == (C.SERVING_ENGINE_LENS, None)
    assert M.launch_engine_of({C.MANIFEST_KEY_ROUTE: {C.SERVED_ENGINE_FIELD: C.SERVING_ENGINE_LENS}}) == (C.SERVING_ENGINE_LENS, None)
    assert M.launch_engine_of({C.MANIFEST_KEY_ROUTE: {C.SERVED_ENGINE_FIELD: C.SERVING_ENGINE_LORA_NATIVE}}) == (C.SERVING_ENGINE_LORA_NATIVE, None)
    assert M.launch_engine_of({C.MANIFEST_KEY_ROUTE: {}, C.SERVED_ENGINE_FIELD: C.SERVING_ENGINE_LENS}) == (C.SERVING_ENGINE_LENS, None)
    root = tmp_path / "native"
    manifest, d, _ = _native_launch_leg(root)
    assert M.launch_attestation(manifest, gen_root=root)["status"] == M.LAUNCH_ATTESTED
    for bad in ("typo", None, "hf", "LORA_NATIVE"):
        m2 = json.loads(json.dumps(manifest))
        m2[C.MANIFEST_KEY_ROUTE][C.SERVED_ENGINE_FIELD] = bad
        engine, problem = M.launch_engine_of(m2)
        assert engine is None and f"route.serving_engine {bad!r} is not one of ['lens', 'lora_native']" in problem and "cannot be selected by engine" in problem
        out = M.launch_attestation(m2, gen_root=root)
        assert out["status"] == M.LAUNCH_ABSENT and out["engine"] is None and out["record_file"] is None and out["problems"] == [problem], bad
    # a torn record: the top-level stamp disagreeing with the route's (either way round)
    m2 = json.loads(json.dumps(manifest)); m2[C.SERVED_ENGINE_FIELD] = C.SERVING_ENGINE_LENS
    engine, problem = M.launch_engine_of(m2)
    assert engine is None and "top-level serving_engine 'lens' disagrees with route.serving_engine 'lora_native'" in problem and "torn" in problem
    assert M.launch_attestation(m2, gen_root=root)["problems"] == [problem]
    m3 = {C.MANIFEST_KEY_ROUTE: {}, C.SERVED_ENGINE_FIELD: C.SERVING_ENGINE_LORA_NATIVE}
    engine, problem = M.launch_engine_of(m3)
    assert engine is None and "top-level serving_engine 'lora_native' disagrees with route.serving_engine absent (read as 'lens')" in problem
    # an unreadable engine AND an unlocatable standup: both named, nothing read
    out = M.launch_attestation({C.MANIFEST_KEY_ROUTE: {C.SERVED_ENGINE_FIELD: "typo"}}, gen_root=root)
    assert out["status"] == M.LAUNCH_ABSENT and len(out["problems"]) == 2 and "cannot be selected by engine" in out["problems"][0] and "names no routing_table path" in out["problems"][1]


def test_the_startup_echo_is_read_strictly_the_native_constructor_admitted_and_every_other_non_literal_named(tmp_path):
    """The native engine's `non-default args:` echo is NOT a Python literal — vLLM echoes its `--lora-modules` as
    `LoRAModulePath(name=..., path=..., base_model_name=None, is_3d_lora_weight=False)` dataclass reprs (read off a real
    lora-native standup's log) — so `ast.literal_eval` fails on it and the old reader reported "no startup echo", which was
    false. The reading now admits exactly that constructor (keyword-only, literal keywords) as `{"LoRAModulePath": {...}}`
    and refuses EVERY other non-literal by name — another constructor, a positional argument, a call nested in a keyword,
    an empty call, a starred keyword, a name, an attribute — naming the echo key and the offending source; a body that is
    no dict, a non-string key and an unclosed expression are named too, and only a log with NO marker reads as "no startup
    echo". A lens echo is a plain literal and reads exactly as `ast.literal_eval` reads it."""
    root = tmp_path / "native"
    manifest, d, args = _native_launch_leg(root)
    log = d / S.LAUNCH_LOG
    text = log.read_text()
    line = next(l for l in text.splitlines() if M.LAUNCH_ECHO_MARKER in l)
    head, body = line.split(M.LAUNCH_ECHO_MARKER, 1)
    assert "LoRAModulePath(name='none__lora0', path='" in body and "base_model_name=None, is_3d_lora_weight=False)" in body
    with pytest.raises(ValueError):
        ast.literal_eval(body)                                      # the real log's shape: not a literal
    echo, problem = M.launch_echo_reading(text)
    assert problem is None and M.launch_echo(text) == echo and list(echo) == list(args)
    assert echo["lora_modules"] == [{"LoRAModulePath": {"name": m.name, "path": m.path, "base_model_name": None, "is_3d_lora_weight": False}} for m in args["lora_modules"]]
    assert {k: v for k, v in echo.items() if k != "lora_modules"} == {k: v for k, v in args.items() if k != "lora_modules"}
    rest = "\n".join(l for l in text.splitlines() if M.LAUNCH_ECHO_MARKER not in l)

    def with_body(b):
        return f"{head}{M.LAUNCH_ECHO_MARKER}{b}\n{rest}\n"
    port = args["port"]
    bad_values = [("{'lora_modules': [Foo(name='a', path='b')]}", "lora_modules", "Foo(name='a', path='b')"),
                  ("{'lora_modules': [LoRAModulePath('a', path='b')]}", "lora_modules", "LoRAModulePath('a', path='b')"),
                  ("{'lora_modules': [LoRAModulePath(name=LoRAModulePath(name='a', path='b'), path='c')]}", "lora_modules", "LoRAModulePath(name='a', path='b')"),
                  ("{'lora_modules': [LoRAModulePath()]}", "lora_modules", "LoRAModulePath()"),
                  ("{'lora_modules': [LoRAModulePath(name='a', **kw)]}", "lora_modules", "LoRAModulePath(name='a', **kw)"),
                  ("{'chat_template': PosixPath('/t.jinja')}", "chat_template", "PosixPath('/t.jinja')"),
                  ("{'middleware': [auth.Middleware]}", "middleware", "auth.Middleware"),
                  ("{'dtype': bfloat16}", "dtype", "bfloat16"),
                  ("{'x': {'a': 1, **more}}", "x", "{'a': 1, **more}"),
                  ("{'n': 1 + 2j}", "n", "1 + 2j")]
    for b, key, word in bad_values:
        echo, problem = M.launch_echo_reading(with_body(b))
        assert echo is None and problem.startswith(f"the echo's {key!r} carries a non-literal value the reader does not understand: {word}"), (b, problem)
        assert "the admitted constructors are ['LoRAModulePath'] (keyword-only, literal keywords)" in problem
    for b, word in (("[1, 2]", "is not a dict repr (List)"), ("{1: 2}", "carries a non-string key 1"), ("{'k': 1", "is not a Python expression")):
        echo, problem = M.launch_echo_reading(with_body(b))
        assert echo is None and word in problem, (b, problem)
    assert M.launch_echo_reading(rest) == (None, M.LAUNCH_ECHO_NO_MARKER) and M.LAUNCH_ECHO_NO_MARKER.startswith("no `non-default args:` startup echo")
    # end to end: a leg whose log carries an unknown constructor reads `absent` naming the LOG, the key and the source — never
    # the old "no startup echo" (the marker IS there), and never a partial projection
    log.write_text(with_body("{'lora_modules': [Foo(name='a', path='b')], 'port': %d, 'max_model_len': %d, 'max_num_seqs': %d}" % (port, L_NATIVE, S_NATIVE)))
    out = M.launch_attestation(manifest, gen_root=root)
    assert out["status"] == M.LAUNCH_ABSENT and out["settings"] is None and out["job_id"] == "500476"
    assert out["problems"] == [f"{log}: the echo's 'lora_modules' carries a non-literal value the reader does not understand: Foo(name='a', path='b') — "
                               "the admitted constructors are ['LoRAModulePath'] (keyword-only, literal keywords)"]
    log.write_text(rest + "\n")
    out = M.launch_attestation(manifest, gen_root=root)
    assert out["status"] == M.LAUNCH_ABSENT and out["problems"] == [f"{log}: {M.LAUNCH_ECHO_NO_MARKER}"]
    # the lens echo: a literal, read as literal_eval reads it (a lens log is byte-identical in behaviour)
    lroot = tmp_path / "lens"
    lmanifest, ld, largs = _lens_launch_leg(lroot)
    ltext = (ld / S.LAUNCH_LOG).read_text()
    lbody = next(l for l in ltext.splitlines() if M.LAUNCH_ECHO_MARKER in l).split(M.LAUNCH_ECHO_MARKER, 1)[1]
    assert M.launch_echo(ltext) == ast.literal_eval(lbody) == largs and M.launch_echo_reading(ltext)[1] is None
    # the admitted set is pinned: exactly the one constructor the real native echo carries
    assert M.LAUNCH_ECHO_CONSTRUCTORS == ("LoRAModulePath",)


def test_o106_a_cross_engine_contrast_is_not_admissible_whatever_its_flags_and_the_same_engine_rules_hold(tmp_path):
    """O106: (A) two legs attested through DIFFERENT engines are never comparable — the launch-settings step refuses by name
    BEFORE any flag equality could pass, naming the engine pair, the reason, the remedy and the ruling, and listing EVERY differing
    flag with both values; known-bad twin: a lens-vs-native pair whose EVERY other flag is equal is refused all the same. (B) On a
    SAME-engine contrast the native engine's cell-set plumbing (`lora_modules`, `max_loras`, `max_cpu_loras`) is excluded with O106's
    reason and the LoRA kernel configuration (`enable_lora`, `max_lora_rank`, `lora_target_modules`) and the CUDA-graph
    surface are compared: native-vs-native differing only in the plumbing reads equal; differing in `max_lora_rank`,
    `enable_lora` or `cudagraph_capture_sizes` refuses naming the key. Lens-vs-lens is unchanged."""
    lroot, nroot = tmp_path / "lens", tmp_path / "native"
    lmanifest, _ld, _ = _lens_launch_leg(lroot, model_key="llama8b")               # the SAME model on the lens engine: only the engine differs
    nmanifest, _nd, nargs = _native_launch_leg(nroot)
    lens = {"launch": M.launch_attestation(lmanifest, gen_root=lroot), "identity": {}, "producer": {"matches_pin": True}}
    native = {"launch": M.launch_attestation(nmanifest, gen_root=nroot), "identity": {}, "producer": {"matches_pin": True}}
    assert lens["launch"]["status"] == native["launch"]["status"] == M.LAUNCH_ATTESTED
    labels = {"anchor": "a", "steered": "b"}
    assert M.launch_engines_of({"a": lens, "b": native}, labels) == {"anchor": "lens", "steered": "lora_native"} and M.cross_engine({"a": lens, "b": native}, labels)
    # (A) the cross-engine refusal: the pair, the reason, the remedy, the ruling, every differing flag with both values
    why = M.server_settings_difference({"a": lens, "b": native}, labels)
    head = f"cross-engine contrast: anchor (lens) vs steered (lora_native) is not admissible — {M.CROSS_ENGINE_WHY} — O106; differing launch flags ("
    assert why.startswith(head), why
    assert "numerics=not_measured" in why and "FULL_AND_PIECEWISE vs eager" in why and "none__lora0" in why and M.CROSS_ENGINE_RULING == "O106"
    listed = why[len(head):].split("): ", 1)[1].split("; ")
    assert why[len(head):].startswith(f"{len(listed)}): ")
    keys = [d.split(": ", 1)[0] for d in listed]
    assert set(keys) == {"launch.enforce_eager", "launch.enable_lora", "launch.max_lora_rank", "launch.lora_target_modules"}, keys
    assert "launch.enforce_eager: {'anchor': 'true', 'steered': None}" in listed and "launch.enable_lora: {'anchor': None, 'steered': 'true'}" in listed
    assert not any(k in keys for k in ("launch.lora_modules", "launch.max_loras", "launch.max_cpu_loras", "launch.port", "launch.worker_extension_cls"))
    proj = {lab: M.server_projection({"a": lens, "b": native}[k], override=False) for lab, k in labels.items()}
    assert keys == [f for f in dict.fromkeys(list(proj["anchor"]) + list(proj["steered"])) if proj["anchor"].get(f) != proj["steered"].get(f)]
    problem = M.compatibility_problem({"a": lens, "b": native}, labels)
    assert problem == why
    verdict = M.server_settings_verdict({"a": lens, "b": native}, labels)
    assert verdict["server_settings_equal"] is False and verdict["cross_engine"] is True and verdict["cross_engine_refusal"] == why
    assert verdict["engine_by_leg"] == {"anchor": "lens", "steered": "lora_native"} and "O106" in verdict["server_settings_rule"]
    # known-bad twin: a native leg whose echo is EDITED to read exactly like the lens anchor's (every compared flag equal) is
    # STILL not admissible — equality never admits a cross-engine pair; the list then reads `none`
    lens_args = dict(_lens_launch_leg(tmp_path / "lens_args", model_key="llama8b")[2])
    tport = S.launch_port("500476")
    tmanifest, td, _ = _native_launch_leg(tmp_path / "twin2", args_edit=lambda a: (a.clear(), a.update({**lens_args, "port": tport, "max_model_len": L_NATIVE,
                                                                                                         "max_num_seqs": S_NATIVE, "max_num_batched_tokens": L_NATIVE * S_NATIVE})))
    twin = {"launch": M.launch_attestation(tmanifest, gen_root=tmp_path / "twin2"), "identity": {}, "producer": {"matches_pin": True}}
    assert twin["launch"]["status"] == M.LAUNCH_ATTESTED, twin["launch"]["problems"]
    assert M._first_difference({"a": M.server_projection(lens, override=False), "b": M.server_projection(twin, override=False)}) is None
    why = M.server_settings_difference({"a": lens, "b": twin}, labels)
    assert why is not None and why.startswith("cross-engine contrast: anchor (lens) vs steered (lora_native) is not admissible — ") and "O106" in why
    assert why.endswith(f"differing launch flags (0): {M.CROSS_ENGINE_EQUAL_FLAGS}") and "equal flags never make a cross-engine contrast admissible" in why
    assert M.compatibility_problem({"a": lens, "b": twin}, labels) == why and M.server_settings_verdict({"a": lens, "b": twin}, labels)["server_settings_equal"] is False
    # (B) same engine: differing ONLY in the cell-set plumbing reads EQUAL, the three recorded under `excluded` with the reason
    other_specs = S.native_adapter_specs("llama8b", "si_other__lora_r1", intervention_hash="cd" * 8, source_kind="lora")
    other_specs["si_third__lora_r1"] = {**other_specs["si_other__lora_r1"], "lora_int_id": 3, "dir": other_specs["si_other__lora_r1"]["dir"] + "_3"}

    def three_adapters(a):
        a["lora_modules"] = [S.LoRAModulePath(name=n, path=sp["dir"]) for n, sp in other_specs.items()]
        a["max_loras"] = a["max_cpu_loras"] = 3
    pmanifest, pd, _ = _native_launch_leg(tmp_path / "plumbing", args_edit=three_adapters)
    plumbing = {"launch": M.launch_attestation(pmanifest, gen_root=tmp_path / "plumbing"), "identity": {}, "producer": {"matches_pin": True}}
    assert plumbing["launch"]["status"] == M.LAUNCH_ATTESTED, plumbing["launch"]["problems"]
    assert json.loads(plumbing["launch"]["excluded"]["max_loras"]["value"]) == 3 != json.loads(native["launch"]["excluded"]["max_loras"]["value"])
    assert json.loads(plumbing["launch"]["excluded"]["lora_modules"]["value"]) != json.loads(native["launch"]["excluded"]["lora_modules"]["value"])
    for key in ("lora_modules", "max_loras", "max_cpu_loras"):
        assert plumbing["launch"]["excluded"][key]["why"] == M.LAUNCH_CELL_SET_PLUMBING_WHY == M.LAUNCH_EXCLUDED_KEYS[key] and "O106" in M.LAUNCH_EXCLUDED_KEYS[key]
    assert M.server_settings_difference({"a": native, "b": plumbing}, labels) is None and M.compatibility_problem({"a": native, "b": plumbing}, labels) is None
    v = M.server_settings_verdict({"a": native, "b": plumbing}, labels)
    assert v["server_settings_equal"] is True and v["cross_engine"] is False and v["cross_engine_refusal"] is None
    # ... and the kernel configuration / graph surface are COMPARED: each refuses naming the key, the same-engine message shape
    for name, edit, key in (("rank", lambda a: a.update(max_lora_rank=8), "launch.max_lora_rank"),
                            ("enable", lambda a: a.update(enable_lora=False), "launch.enable_lora"),
                            ("targets", lambda a: a.update(lora_target_modules=["o_proj", "down_proj"]), "launch.lora_target_modules"),
                            ("capture", lambda a: a.update(cudagraph_capture_sizes=[1, 2, 4, 8, 16, 24, 32, 40, 44]), "launch.cudagraph_capture_sizes"),
                            ("eager", lambda a: a.update(enforce_eager=True), "launch.enforce_eager")):
        omanifest, _od, _ = _native_launch_leg(tmp_path / f"kernel_{name}", args_edit=edit)
        other = {"launch": M.launch_attestation(omanifest, gen_root=tmp_path / f"kernel_{name}"), "identity": {}, "producer": {"matches_pin": True}}
        assert other["launch"]["status"] == M.LAUNCH_ATTESTED, (name, other["launch"]["problems"])
        why = M.server_settings_difference({"a": native, "b": other}, labels)
        assert why is not None and why.startswith(f"{M.SERVER_SETTINGS_DIFFER} — {key}: ") and "cross-engine" not in why, (name, why)
        assert key not in M.LAUNCH_EXCLUDED_KEYS and M.compatibility_problem({"a": native, "b": other}, labels) == why
    assert {"enable_lora", "max_lora_rank", "lora_target_modules", "enforce_eager", "cudagraph_mode", "cudagraph_capture_sizes"}.isdisjoint(M.LAUNCH_EXCLUDED_KEYS)
    assert {"lora_modules", "max_loras", "max_cpu_loras", "worker_extension_cls"} <= set(M.LAUNCH_EXCLUDED_KEYS)
    # lens-vs-lens: unchanged — the first differing field, or None
    l2manifest, _l2d, _ = _lens_launch_leg(tmp_path / "lens2", job="480124", model_key="llama8b")
    lens2 = {"launch": M.launch_attestation(l2manifest, gen_root=tmp_path / "lens2"), "identity": {}, "producer": {"matches_pin": True}}
    assert lens2["launch"]["status"] == M.LAUNCH_ATTESTED and M.server_settings_difference({"a": lens, "b": lens2}, labels) is None
    assert M.server_settings_difference({"a": native, "b": native}, labels) is None
    omanifest, _od, _ = _native_launch_leg(tmp_path / "dtype", args_edit=lambda a: a.update(dtype="float16"))
    other = {"launch": M.launch_attestation(omanifest, gen_root=tmp_path / "dtype"), "identity": {}}
    assert M.server_settings_difference({"a": native, "b": other}, {"x": "a", "y": "b"}) == f"{M.SERVER_SETTINGS_DIFFER} — launch.dtype: {{'x': '\"bfloat16\"', 'y': '\"float16\"'}}"


# --------------------------------------------------------------------------- #
# the DECLARED expected refusal of a comparison (the O106 cross-engine ruling: no native `none` anchor is run)          #
# --------------------------------------------------------------------------- #


def _refused_view_comparison(rep, *, arm_kind, it_key, main_bare, declared="O106", refusal=True, code_matches=True):
    """A `qwen35_4b/thinking_card@it` comparison record as `run_comparison` files a cross-engine refusal (the set-aside
    production run's shape: leg_keys {bare: the MAIN bare anchor, <arm>: the it-view leg}, usable False, estimands None, the
    reason IS the compatibility record's cross_engine_refusal), with the design's declaration and the derived fields as
    `comparisons_from` + `run_comparison` would stamp them."""
    why = (f"cross-engine contrast: bare (lens) vs {arm_kind} (lora_native) is not admissible — {M.CROSS_ENGINE_WHY} — {M.CROSS_ENGINE_RULING}; "
           "differing launch flags (1): launch.enable_lora: {'bare': None, '" + arm_kind + "': 'true'}")
    rec = {"name": f"{QWEN}/{PRIMARY}@{IT}", "model_key": QWEN, "mode": PRIMARY, "view": IT, "primary": True,
           "leg_keys": {"bare": main_bare, arm_kind: it_key}, "arms_absent": [], "itdoc_arms": {},
           "bare_anchor": {"view": M.VIEW_MAIN, "leg": main_bare, "rationale": M.BARE_ANCHOR_RATIONALE},
           "usable": not refusal, "reason": (why if refusal else None), "estimands": None,
           "compatibility": {"cross_engine": bool(refusal), "cross_engine_refusal": (why if refusal else None)}}
    if declared is not None:
        rec[M.ROW_EXPECTED_REFUSAL_KEY] = declared
    if refusal:
        code = M.comparison_refusal_code(rec)
        if code is not None and code_matches:
            rec[M.REFUSAL_CODE_KEY] = code
        if declared is not None and code == declared and code_matches:
            rec[M.REFUSED_BY_DESIGN_KEY] = True
    return rec


def test_the_launchers_cells_py_declares_row_53s_expected_refusal_with_the_readers_own_ruling_constant(raw_launcher_cells):
    """The all-positions row's comparison against the lens `none` anchor is O106-inadmissible and no native anchor is run (the
    ruling of 2026-09-17): the DESIGN declares the refusal on row 53 alone — `expected_refusal` = the launcher's constant,
    pinned equal to the reader's CROSS_ENGINE_RULING (never a new string) — and the reader's row field and code roster agree."""
    raw = list(raw_launcher_cells.M5T_CELLS)
    assert raw_launcher_cells.EXPECTED_REFUSAL_CROSS_ENGINE == M.CROSS_ENGINE_RULING == "O106"
    assert M.COMPARISON_REFUSAL_CODES == (M.CROSS_ENGINE_RULING,) and M.ROW_EXPECTED_REFUSAL_KEY == "expected_refusal"
    declaring = [r for r in raw if r.get(M.ROW_EXPECTED_REFUSAL_KEY)]
    assert [(r["index"], r["cell"], r[M.ROW_EXPECTED_REFUSAL_KEY]) for r in declaring] == [(53, "arditi_allpos", M.CROSS_ENGINE_RULING)]
    assert "expected_refusal" in raw_launcher_cells.ALLPOS_KEYS
    # the reader reads it off the row and stamps the it-view comparison — only that comparison, only when the REQUIRED rows are
    # exactly the declared arm (the it-doc arms of the same view are held rows: analyze feeds comparisons_from the required set)
    live = M.load_cells(REPO / "scripts" / "sdfbase" / "math500" / "cells.py")
    required = [r for r in live if M.row_status(r) == "launch" and (M.key_view(r) == M.VIEW_MAIN or r["cell"] == "arditi_allpos")]
    comps = M.comparisons_from(required, eight_arm=True)
    it_c = next(c for c in comps if c["name"] == f"{QWEN}/{PRIMARY}@{IT}")
    assert it_c[M.ROW_EXPECTED_REFUSAL_KEY] == M.CROSS_ENGINE_RULING and M.EXPECTED_REFUSAL_PROBLEM_KEY not in it_c
    assert it_c["leg_keys"] == {"bare": f"{QWEN}/{BARE}/{PRIMARY}", "arditi_allpos": f"{QWEN}/arditi_allpos/{PRIMARY}@{IT}"}
    assert all(M.ROW_EXPECTED_REFUSAL_KEY not in c and M.EXPECTED_REFUSAL_PROBLEM_KEY not in c for c in comps if c is not it_c)
    # a group where SOME arms declare (the held it-doc arms read beside the allpos row) is a named PROBLEM, never a partial admission
    with_itdoc = required + [r for r in live if M.key_view(r) == IT and r["cell"] != "arditi_allpos"]
    mixed = next(c for c in M.comparisons_from(with_itdoc, eight_arm=True) if c["name"] == f"{QWEN}/{PRIMARY}@{IT}")
    assert mixed.get(M.ROW_EXPECTED_REFUSAL_KEY) is None and "every arm must declare the same code" in mixed[M.EXPECTED_REFUSAL_PROBLEM_KEY]
    # the group rule directly: one code on every arm; none; a second code; a code the reader does not derive
    rows_by = {"bare": {}, "a": {"expected_refusal": "O106"}, "b": {"expected_refusal": "O106"}}
    assert M.expected_refusal_of_group(rows_by) == ("O106", None)
    assert M.expected_refusal_of_group({"bare": {}, "a": {}}) == (None, None)
    code, problem = M.expected_refusal_of_group({"bare": {}, "a": {"expected_refusal": "O106"}, "b": {"expected_refusal": "O999"}})
    assert code is None and "every arm must declare the same code" in problem
    code, problem = M.expected_refusal_of_group({"bare": {}, "a": {"expected_refusal": "O999"}})
    assert code is None and "is not a refusal code this reader derives" in problem


def test_a_comparison_refused_exactly_as_declared_passes_the_population_gate_and_the_report_is_a_production_result(c4_report):
    """The set-aside production run: the population gate filed the O106 cross-engine refusal of `qwen35_4b/thinking_card@it` as a
    FAILURE and the export wrote no paper rows. With the design declaring the refusal: the gate PASSES and records it, the
    comparison stays filed refused with the O106 text (never computed), `comparisons_not_run` / `production_refusals` no longer
    call it "did not run", and the arm's leg reaches the paper rows with the arm's own numbers and the refusal named. Twins: a
    refusal with a different / no code under the declaration FAILS naming both; a declared comparison that COMPUTES is refused
    ("expected a refusal, got a contrast"); an undeclared refusal FAILS with today's message byte for byte; a partial declaration
    fails by name."""
    rep = json.loads(json.dumps(c4_report))
    main_c = comparison(rep, QWEN)
    main_bare, main_obj = main_c["leg_keys"]["bare"], main_c["leg_keys"]["object"]
    it_key = f"{QWEN}/{OBJECT}/{PRIMARY}@{IT}"
    # the it-view leg: the main object leg re-keyed on the `it` view (its own items, summary and provenance travel with it)
    rep["legs"][it_key] = {**json.loads(json.dumps(rep["legs"][main_obj])), "leg": it_key, "view": IT}
    legs, ids = rep["legs"], S.IDS
    by_design = _refused_view_comparison(rep, arm_kind="object", it_key=it_key, main_bare=main_bare)
    assert by_design[M.REFUSAL_CODE_KEY] == M.CROSS_ENGINE_RULING and by_design[M.REFUSED_BY_DESIGN_KEY] is True
    comps = list(rep["comparisons"]) + [by_design]
    g = M.population_gate(legs, comps, ids)
    assert g["status"] == M.GATE_PASS and g["expected_refusals"] == {by_design["name"]: M.CROSS_ENGINE_RULING}, g
    assert M.comparisons_not_run(comps) == []
    rep2 = {**rep, "comparisons": comps, "gates": {**rep["gates"], "population": g}}
    assert M.production_refusals(rep2) == []
    # the comparison is NOT computed: its refusal text stands, its estimands stay None
    assert by_design["usable"] is False and by_design["estimands"] is None and f"— {M.CROSS_ENGINE_RULING};" in by_design["reason"]
    # the paper rows: the arm's row carries the ARM's own numbers (n_correct from its items, the exact binomial interval) and names
    # the refusal; the main bare leg is exported ONCE, from the main comparison (the view comparison's anchor role adds no row)
    rows, skipped = M.paper_rows_of_report(rep2, report_path="synthetic/report.json")
    it_rows = [r for r in rows if r["cell"] == OBJECT and r["view"] == IT]
    assert len(it_rows) == 1
    row = it_rows[0]
    leg = legs[it_key]
    scored = [rec for rec in leg["items"].values() if rec["scored"]]
    nc = sum(1 for rec in scored if float(rec["value"]) == 1.0)
    assert (row["n_paired"], row["n_scored"], row["n_correct"]) == (len(leg["items"]), len(scored), nc)
    assert row["point_conditional"] == nc / len(scored) and row["point_score_as_is"] == nc / len(leg["items"])
    assert row["ci_method"] == M.PRW.CI_EXACT and row["ci_conditional_low"] <= row["point_conditional"] <= row["ci_conditional_high"]
    assert f"refused by design ({M.CROSS_ENGINE_RULING})" in row["source"] and "the arm's numbers are reported as the arm's numbers" in row["source"]
    assert (row["n_at_cap"], row["at_cap_rate"]) == (leg["summary"]["n_at_cap"], leg["summary"]["at_cap_rate"])
    assert sum(1 for r in rows if r["cell"] == BARE and r["model_key"] == QWEN) == 1
    main_rows = [r for r in rows if r["cell"] == OBJECT and r["view"] == M.VIEW_MAIN and r["model_key"] == QWEN]
    assert len(main_rows) == 1 and main_rows[0]["ci_method"] == M.PRW.CI_BOOTSTRAP
    # ... and the arm mapping itself: the anchor role of the view comparison is skipped, the main bare leg maps to the main comparison
    arms = M._pr_leg_arms(rep2)
    assert arms[main_bare][1]["name"] == main_c["name"] and arms[it_key] == ("object", by_design)
    # twin (1): the declaration stands but the refusal has ANOTHER reason — no named code — still FAILS, naming both
    other = _refused_view_comparison(rep, arm_kind="object", it_key=it_key, main_bare=main_bare)
    other["reason"] = "the legs' normalised server settings differ — launch.dtype: {'bare': 'bfloat16', 'object': 'float16'}"
    other["compatibility"] = {"cross_engine": False, "cross_engine_refusal": None}
    other.pop(M.REFUSAL_CODE_KEY, None); other.pop(M.REFUSED_BY_DESIGN_KEY, None)
    assert M.comparison_refusal_code(other) is None
    g = M.population_gate(legs, list(rep["comparisons"]) + [other], ids)
    assert g["status"] == M.GATE_FAIL and "declared an expected refusal (O106) but refused with no named code" in g["detail"], g["detail"]
    assert M.comparisons_not_run(list(rep["comparisons"]) + [other]) == [other["name"]]
    # twin (2): a declared comparison that COMPUTES — "expected a refusal, got a contrast"
    computed = {**json.loads(json.dumps(main_c)), "name": f"{QWEN}/{PRIMARY}@{IT}", "view": IT, M.ROW_EXPECTED_REFUSAL_KEY: M.CROSS_ENGINE_RULING}
    g = M.population_gate(legs, list(rep["comparisons"]) + [computed], ids)
    assert g["status"] == M.GATE_FAIL and "expected a refusal (O106), got a contrast" in g["detail"], g["detail"]
    # twin (3): the UNDECLARED cross-engine refusal — the set-aside run — fails with today's message, byte for byte
    undeclared = _refused_view_comparison(rep, arm_kind="object", it_key=it_key, main_bare=main_bare, declared=None)
    assert M.REFUSED_BY_DESIGN_KEY not in undeclared and undeclared[M.REFUSAL_CODE_KEY] == M.CROSS_ENGINE_RULING
    g = M.population_gate(legs, list(rep["comparisons"]) + [undeclared], ids)
    assert g["status"] == M.GATE_FAIL and g["detail"] == f"{undeclared['name']}: {undeclared['reason']}"
    assert M.comparisons_not_run(list(rep["comparisons"]) + [undeclared]) == [undeclared["name"]]
    rep3 = {**rep, "comparisons": list(rep["comparisons"]) + [undeclared], "gates": {**rep["gates"], "population": g}}
    assert any(r == f"comparison(s) did not run: [{undeclared['name']!r}]" for r in M.production_refusals(rep3)), M.production_refusals(rep3)
    # twin (4): a partial declaration is a problem the gate fails by name
    partial = {**undeclared, M.EXPECTED_REFUSAL_PROBLEM_KEY: "expected_refusal declared on ['object'] as ['O106'] while the comparison's arms are ['lora_ctl2', 'object'] — a comparison refuses as one, so every arm must declare the same code"}
    g = M.population_gate(legs, list(rep["comparisons"]) + [partial], ids)
    assert g["status"] == M.GATE_FAIL and "every arm must declare the same code" in g["detail"]
    # a comparison refused on a MISSING leg has no code, whatever the design declares (no compatibility was read)
    assert M.comparison_refusal_code({"usable": False, "reason": "object: missing", "compatibility": None}) is None
    # the code is derived from the record's own facts: the text alone, without the compatibility bit, derives nothing
    assert M.comparison_refusal_code({"usable": False, "reason": undeclared["reason"], "compatibility": {"cross_engine": False}}) is None
# the capture-size carve: a `cudagraph_capture_sizes` difference between same-engine native legs is a function of S      #
# --------------------------------------------------------------------------- #

#: the two REAL native launch echoes of the pulled qwen35_4b `arditi_allpos` GSM8K pair (jobs 544615 @8192 and 544730 @16384;
#: both kernel-ceiling lowerings, S 69->48 and 39->27): the geometry and the capture list each echoed, number for number
REAL_CAPTURE_BASE = dict(max_model_len=10752, max_num_seqs=48, cudagraph_capture_sizes=[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96])
REAL_CAPTURE_RUNG = dict(max_model_len=18944, max_num_seqs=27, cudagraph_capture_sizes=[1, 2, 4, 8, 16, 24, 27, 32, 40, 48])


def _native_leg_at(root, *, job, geometry, capture_sizes=None, launch_edit=None):
    """A natively served leg whose launch echoes ``geometry``'s window and width and the capture list ``capture_sizes`` (the
    geometry's own by default), attested through the real reader — the shape `_rung_projection_problem` and the contrast
    comparison read. ``launch_edit`` mutates the ATTESTED launch dict (a reader-side record edit)."""
    L, S_ = int(geometry["max_model_len"]), int(geometry["max_num_seqs"])
    sizes = list(geometry["cudagraph_capture_sizes"] if capture_sizes is None else capture_sizes)

    def edit(a):
        a.update(max_num_batched_tokens=L * S_, cudagraph_capture_sizes=sizes)
    manifest, _d, _args = _native_launch_leg(root, job=job, args_edit=edit, max_model_len=L, max_num_seqs=S_)
    leg = {"launch": M.launch_attestation(manifest, gen_root=root), "identity": {}, "producer": {"matches_pin": True}, "leg": root.name}
    assert leg["launch"]["status"] == M.LAUNCH_ATTESTED, leg["launch"]["problems"]
    if launch_edit:
        launch_edit(leg["launch"])
    return leg


def test_a_capture_size_difference_between_same_engine_native_legs_is_admitted_only_as_each_sides_derivation_for_its_own_s(tmp_path):
    """The native composer hands vLLM `cudagraph_capture_sizes_for(max_num_seqs)` — the default sizes for S plus S itself — so two
    kernel-ceiling lowerings to different S echo different lists BY CONSTRUCTION (the pulled qwen35_4b `arditi_allpos` GSM8K
    pair: S 48 at the 8,192 base, 27 at the 16,384 rung); the same-engine comparison read the list as a setting and refused the
    rung/base pairing (A7). Now the difference is admitted ONLY IF each side's list is the composer's derivation for ITS OWN
    attested S (`M.capture_sizes_by_derivation`, through `vllm_server.cudagraph_capture_sizes_for` — derived, never trusted).
    Known-bad twins: the same S with different lists; a list that is not the derivation for its S; a leg whose S is not
    attested; cross-engine, refused as today with the key among the differing flags. Every other field is byte-identical."""
    from sisteer.capeval.serving import vllm_server as VS
    # the derivation reproduces BOTH real echoes, and it is the composer's own function the reader imports
    assert list(VS.cudagraph_capture_sizes_for(48)) == REAL_CAPTURE_BASE["cudagraph_capture_sizes"]
    assert list(VS.cudagraph_capture_sizes_for(27)) == REAL_CAPTURE_RUNG["cudagraph_capture_sizes"]
    assert M.LAUNCH_CAPTURE_SIZES_KEY == "cudagraph_capture_sizes" and "cudagraph_capture_sizes_for" in M.CAPTURE_SIZES_RULE
    base = _native_leg_at(tmp_path / "base", job="544615", geometry=REAL_CAPTURE_BASE)
    rung = _native_leg_at(tmp_path / "rung", job="544730", geometry=REAL_CAPTURE_RUNG)
    assert M.attested_launch_seqs(base) == 48 and M.attested_launch_seqs(rung) == 27
    labels = {"base": "a", "rung": "b"}
    legs = {"a": base, "b": rung}
    # the literal projections differ on exactly this key ...
    literal = {lab: M.server_projection(legs[k]) for lab, k in labels.items()}
    assert M._first_difference(literal).startswith("launch.cudagraph_capture_sizes: ")
    assert M._all_differences(literal) == [M._first_difference(literal)]
    # ... and the carve drops it, because each side's list IS its own S's derivation
    carved, note = M.capture_sizes_by_derivation({"base": base, "rung": rung}, literal)
    assert note is None and "launch.cudagraph_capture_sizes" not in carved["base"] and M._first_difference(carved) is None
    assert M.server_settings_difference(legs, labels) is None and M.compatibility_problem(legs, labels) is None
    v = M.server_settings_verdict(legs, labels)
    assert v["server_settings_equal"] is True and v["cross_engine"] is False
    # the literal reading stays beside it: the record says the lists differ and that the comparison admitted them by derivation
    assert v["server_settings_literal_equal"] is False and v["server_settings_literal_difference"].startswith("launch.cudagraph_capture_sizes: ")
    assert "cudagraph_capture_sizes_for" in v["server_settings_rule"]
    # the GSM8K rung/base pairing — where A7 fired — reads through the same carve
    import importlib.util
    spec = importlib.util.spec_from_file_location("tinycap_gsm8k_paired200_capture_test", REPO / "scripts" / "tinycap_gsm8k_paired200.py")
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    assert A._rung_projection_problem(base, rung) is None
    # twin (1): the SAME S, different lists — the rung's list on the base's width is not the derivation for 48
    same_s = _native_leg_at(tmp_path / "same_s", job="544616", geometry=REAL_CAPTURE_BASE, capture_sizes=REAL_CAPTURE_RUNG["cudagraph_capture_sizes"])
    why = M.server_settings_difference({"a": base, "b": same_s}, labels)
    assert why is not None and why.startswith(f"{M.SERVER_SETTINGS_DIFFER} — launch.cudagraph_capture_sizes: ") and "cross-engine" not in why, why
    assert "rung's list" in why and "is not the composer's derivation for its attested max_num_seqs 48" in why and "a real setting change" in why
    rung_why = A._rung_projection_problem(base, same_s)
    assert rung_why is not None and "is not the composer's derivation for its attested max_num_seqs 48" in rung_why, rung_why
    # twin (2): a list that is not the derivation for ITS S — the rung's width 27 with vLLM's default sizes and NO 27 (the list
    # that stops short of S, the 2026-09-16 incident's shape), and the base's width 48 with a list that stops at 48
    wrong = _native_leg_at(tmp_path / "wrong", job="544731", geometry=REAL_CAPTURE_RUNG, capture_sizes=[1, 2, 4, 8, 16, 24, 32, 40, 48])
    why = M.server_settings_difference({"a": base, "b": wrong}, labels)
    assert why is not None and "rung's list" in why and "for its attested max_num_seqs 27" in why and "[1, 2, 4, 8, 16, 24, 27, 32, 40, 48]" in why, why
    short = _native_leg_at(tmp_path / "short", job="544732", geometry=REAL_CAPTURE_BASE, capture_sizes=[1, 2, 4, 8, 16, 24, 32, 40, 48])
    why = M.server_settings_difference({"a": short, "b": rung}, labels)
    assert why is not None and "base's list" in why and "is not the composer's derivation for its attested max_num_seqs 48" in why, why
    assert M.server_settings_verdict({"a": short, "b": rung}, labels)["server_settings_equal"] is False
    # two EQUAL lists never reach the carve — there is no difference to admit, and the comparison reads them equal exactly as
    # before (whether one leg's list covers its own S is the standup's readiness coverage check, not the pairwise comparison's)
    equal_lists = _native_leg_at(tmp_path / "equal", job="544735", geometry=REAL_CAPTURE_RUNG, capture_sizes=REAL_CAPTURE_BASE["cudagraph_capture_sizes"])
    assert M.server_settings_difference({"a": base, "b": equal_lists}, labels) is None
    # twin (3): S not attested — the echo's max_num_seqs missing from the record, or the launch not geometry-reconciled
    def drop_s(launch):
        launch["excluded"].pop("max_num_seqs")
    no_s = _native_leg_at(tmp_path / "no_s", job="544733", geometry=REAL_CAPTURE_RUNG, launch_edit=drop_s)
    assert M.attested_launch_seqs(no_s) is None
    why = M.server_settings_difference({"a": base, "b": no_s}, labels)
    assert why is not None and "rung's attested max_num_seqs cannot be read" in why and "the difference stands" in why, why
    unreconciled = {**rung, "launch": {**rung["launch"], "geometry_reconciled": False}}
    assert M.attested_launch_seqs(unreconciled) is None
    # twin (4): cross-engine — O106 refuses as today, the capture key listed among the differing flags (the lens echo has none)
    lmanifest, _ld, _ = _lens_launch_leg(tmp_path / "lens", model_key="llama8b")
    lens = {"launch": M.launch_attestation(lmanifest, gen_root=tmp_path / "lens"), "identity": {}, "producer": {"matches_pin": True}, "leg": "lens"}
    assert lens["launch"]["status"] == M.LAUNCH_ATTESTED
    why = M.server_settings_difference({"a": lens, "b": base}, labels)
    assert why.startswith("cross-engine contrast: base (lens) vs rung (lora_native) is not admissible — ") and "O106" in why
    assert "launch.cudagraph_capture_sizes: {'base': None, 'rung': '[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96]'}" in why
    cross = {"base": lens, "rung": base}
    carved, note = M.capture_sizes_by_derivation(cross, {lab: M.server_projection(l, override=False) for lab, l in cross.items()})
    assert note is None and "launch.cudagraph_capture_sizes" in carved["rung"]        # untouched: O106 owns the refusal
    # a non-native pair differing on the key (the lens engine composes none): a real setting change, the key kept
    lens_with = {**lens, "launch": {**lens["launch"], "settings": {**lens["launch"]["settings"], "cudagraph_capture_sizes": "[1, 2, 4]"}}}
    pair = {"x": lens, "y": lens_with}
    carved, note = M.capture_sizes_by_derivation(pair, {lab: M.server_projection(l) for lab, l in pair.items()})
    assert note is not None and "composes no capture list" in note and "launch.cudagraph_capture_sizes" in carved["y"]
    # every other field is byte-identical: a dtype difference beside admitted capture lists names dtype exactly as before
    dtype = _native_leg_at(tmp_path / "dtype", job="544734", geometry=REAL_CAPTURE_RUNG, launch_edit=lambda l: l["settings"].update(dtype=json.dumps("float16")))
    assert M.server_settings_difference({"a": base, "b": dtype}, labels) == f"{M.SERVER_SETTINGS_DIFFER} — launch.dtype: {{'base': '\"bfloat16\"', 'rung': '\"float16\"'}}"
    # ... and with no difference on the key the projections come back untouched
    same, note = M.capture_sizes_by_derivation({"a": base, "b": base}, {"a": M.server_projection(base), "b": M.server_projection(base)})
    assert note is None and same == {"a": M.server_projection(base), "b": M.server_projection(base)}
