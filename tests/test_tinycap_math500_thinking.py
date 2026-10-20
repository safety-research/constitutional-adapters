"""``scripts/tinycap_math500_thinking.py``:
the MATH-500 thinking-ON analysis over the 24 SYNTHETIC stores of
``tests/math500_synthetic_store.py`` (written with this tree's real writers):
the per-model three-cell contrasts under both scoring branches with paired
bootstrap CIs and censoring bounds, the verdict rule (a word only when
both branches' CIs exclude zero on one side, the signs agree and the bound is
robust — a one-item edge stays unresolved), the sensitivity modes never
pooled, every gate's refusal by name (population, ledger incl. the
``error_code`` refusals, request evidence, identity incl. the sampling-row rule
and the ``diagnostic`` flag, certification (A)/(B′)/(C), at-cap, split
anomalies, the scoring path), the fail-closed ``main()`` boundary
(exit 3 / exit 4), the two-venv shape, the figures and the sidecar; and the two
readouts filed as RESULTS: the at-cap share per cell against the 1 %
line and the declared 3 % surfacing threshold with the truncated-reasoning vs
degenerate-loop split, and the no-reasoning-emitted share per split mode."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import math500_synthetic_store as S
import pytest

from sisteer.capeval.tiers import pairing as pr
from sisteer import site as SITE
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap.completion import CompletionError

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("tinycap_math500_thinking",
                                               REPO / "scripts" / "tinycap_math500_thinking.py")
M = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(M)

#: fewer replicates where the exact CI is not the point
B_FAST = 400
QWEN, GLM, NEMO, GEMMA = "qwen35_4b", "glm46vflash", "nemotron3nano4b", "gemma4e4b"
BARE, PROMPTED, OBJECT = S.CELL_BARE, S.CELL_PROMPTED, S.CELL_OBJECT
PRIMARY, GENERAL, BATTERY = S.MODE_PRIMARY, S.MODE_GENERAL, S.MODE_BATTERY


def analyze(gen_root, *, rows=None, **kw):
    kw.setdefault("cells", rows if rows is not None else S.cells_matrix())
    kw.setdefault("sampling_rows", S.SAMPLING_ROWS)
    kw.setdefault("render_facts", S.RENDER_FACTS)
    kw.setdefault("math500_ids", S.IDS)
    kw.setdefault("n_resamples", B_FAST)
    kw.setdefault("read_eval_logs", False)
    kw.setdefault("analysis_sha", S.FULL_SHA)
    return M.analyze(gen_root, S.SHA7, **kw)


def leg_of(report, model_key, cell, mode=PRIMARY):
    return report["legs"][M.leg_key(model_key, cell, mode)]


def rows_for(model_key, *modes):
    return [r for r in S.cells_matrix() if r["model_key"] == model_key and r["sampling_mode"] in modes]


def primary_rows(model_key):
    return rows_for(model_key, PRIMARY)


def comparison(report, model_key, mode=PRIMARY):
    return next(c for c in report["comparisons"] if c["model_key"] == model_key and c["mode"] == mode)


@pytest.fixture(scope="module")
def full_store(tmp_path_factory):
    root = tmp_path_factory.mktemp("m5t_full")
    S.build_all(root)
    return root


@pytest.fixture(scope="module")
def full_report(full_store):
    return analyze(full_store)


# --------------------------------------------------------------------------- #
# the reader contract is pinned EQUAL between the builder and the script       #
# --------------------------------------------------------------------------- #


def test_reader_constants_pin_the_builder():
    for name in ("REQUEST_EVIDENCE_FILENAME", "EVIDENCE_RENDER_CERTIFIED", "EVIDENCE_RENDER_PROBE", "EVIDENCE_LENS_POSITION",
                 "EVIDENCE_LENS_ON_BITE", "MANIFEST_KEY_TRANSFER_PROVENANCE", "MANIFEST_KEY_OOS_EXPERIMENT",
                 "MANIFEST_KEY_RESUME_ATTEMPTS", "MANIFEST_KEY_BINDING", "ROUTE_KEY_LENS_SERVED_POSITIONS",
                 "RESUME_ATTEMPT_FIELDS", "IDENTITY_KEY_VERSION", "IDENTITY_KEY_THINKING_HEADER_IDS", "CELL_OBJECT",
                 "CELL_BARE", "CELL_PROMPTED", "MODE_PRIMARY", "MODE_GENERAL", "MODE_BATTERY", "TASK", "CAP"):
        assert getattr(M, name) == getattr(S, name), name
    # the synthetic render-fact table is the CONTRACTS' tuples under a certified status (the producer helpers
    # compare against those tuples), and the analysis' default table IS the contracts'
    for fam, fact in S.RENDER_FACTS.items():
        real = C.FAMILY_RENDER_FACTS[fam]
        assert (fact["off_ids"], fact["on_ids"], fact["split_mode"]) == (list(real.off_ids), list(real.on_ids), real.split_mode)
        assert fact["evidence_status"] == C.RENDER_EVIDENCE_CERTIFIED
    assert set(M.load_render_facts(None)) >= set(S.RENDER_FACTS)
    assert M.N_ITEMS == S.N
    # the DESIGNED primary set the premise verdict is held to is the cells module's model list x the three cells
    assert tuple(M.DESIGN_MODEL_KEYS) == tuple(S.MODEL_KEYS) and M.N_PRIMARY_CELLS_DESIGN == 12 == len(S.MODEL_KEYS) * len(S.CELLS)
    assert M.cell_slug(OBJECT) == S.cell_slug(OBJECT) == "constv2-mix-full--lora_r4_ce_minus_ctl"
    assert M.MIRROR_DIR_DEFAULT.endswith("math500-thinking-on-full22")
    assert M.WANDB_RUN_PREFIX == "m5t-" and M.WANDB_PURPOSE == "tinycap"      # the project: wandb_project("tinycap"), the site's


# --------------------------------------------------------------------------- #
# the generalised paired bootstrap == pairing's 2x2 on a four-leg input         #
# --------------------------------------------------------------------------- #


def _four_legs(n=200):
    ids = [f"g_{i:04x}" for i in range(n)]

    def leg(correct, unscored=()):
        return {i: ((0, None) if j in unscored else (1, 1.0 if j in correct else 0.0)) for j, i in enumerate(ids)}

    return {("none", C.THINKING_OFF): leg(set(range(120))),
            ("psr", C.THINKING_OFF): leg(set(range(107)) | set(range(150, 163))),
            ("none", C.THINKING_ON): leg(set(range(126)), unscored={196, 197}),
            ("psr", C.THINKING_ON): leg(set(range(143)), unscored={198, 199})}


def test_generalised_bootstrap_equals_pairings_2x2_on_four_legs():
    """The script's ``paired_bootstrap_conditional`` (any legs, any signed
    contrasts) reproduces ``pairing.paired_bootstrap_conditional_2x2`` BIT FOR
    BIT on the four-leg design with the same contrasts — same index draws
    (a pure function of (seed, n, B)), same exact-rational arithmetic, same
    replicate signs — so pairing.py stays untouched and the new call is the
    proven primitive."""
    legs4 = _four_legs()
    want = pr.paired_bootstrap_conditional_2x2(legs4, n_resamples=2000, seed=0)
    legs = {pr.leg_label(k): v for k, v in legs4.items()}
    contrasts = {c: {pr.leg_label(k): s for k, s in signs.items()} for c, signs in pr.CONTRAST_SIGNS.items()}
    got = M.paired_bootstrap_conditional(legs, contrasts, n_resamples=2000, seed=0)
    for top in ("B", "seed", "level", "n_items", "n_scored_by_leg", "n_censored_by_leg", "n_correct_by_leg",
                "n_degenerate_replicates", "n_replicates_used", "acc_by_leg", "complete_case"):
        assert got[top] == want[top], top
    assert set(got["stats"]) == set(want["stats"])
    for c in want["stats"]:
        assert got["stats"][c] == want["stats"][c], c


def test_generalised_bootstrap_refusals():
    ids = [f"i{i}" for i in range(10)]
    a = {i: (1, 1.0 if j < 6 else 0.0) for j, i in enumerate(ids)}
    b = {i: (1, 1.0 if j < 4 else 0.0) for j, i in enumerate(ids)}
    contrasts = {"d": {"b": +1, "a": -1}}
    with pytest.raises(M.AnalysisError, match="ATTEMPTED id sets differ"):
        M.paired_bootstrap_conditional({"a": a, "b": {k: v for k, v in b.items() if k != "i3"}}, contrasts,
                                       n_resamples=10)
    with pytest.raises(M.AnalysisError, match="names a leg"):
        M.paired_bootstrap_conditional({"a": a}, contrasts, n_resamples=10)
    with pytest.raises(M.AnalysisError, match="mask"):
        M.paired_bootstrap_conditional({"a": a, "b": {**b, "i0": (2, 1.0)}}, contrasts, n_resamples=10)
    with pytest.raises(M.AnalysisError, match="masked-out item carries no value"):
        M.paired_bootstrap_conditional({"a": a, "b": {**b, "i0": (0, 1.0)}}, contrasts, n_resamples=10)
    with pytest.raises(M.AnalysisError, match="scored no item"):
        M.paired_bootstrap_conditional({"a": a, "b": {i: (0, None) for i in ids}}, contrasts, n_resamples=10)
    with pytest.raises(M.AnalysisError, match="needs >= 1 resamples"):
        M.paired_bootstrap_conditional({"a": a, "b": b}, contrasts, n_resamples=0)
    out = M.paired_bootstrap_conditional({"a": a, "b": b}, contrasts, n_resamples=50)
    assert out["stats"]["d"]["point"] == -0.2 and out["stats"]["d"]["point_sign"] == -1
    assert out["stats"]["d"]["bound_low"] == out["stats"]["d"]["bound_high"] == -0.2


# --------------------------------------------------------------------------- #
# the synthetic 24-leg run: every comparison, both branches, the words          #
# --------------------------------------------------------------------------- #


@pytest.fixture
def raw_launcher_cells():
    """The launcher's matrix module (``scripts/sdfbase/math500/cells.py``) executed FRESH under a throwaway name, so a test
    can read ``M5T_CELLS`` raw — including a row ``load_cells`` refuses. Registered in ``sys.modules`` for the exec (the
    launcher's dataclass resolves its future-annotation strings through ``sys.modules[cls.__module__]``) and REMOVED on
    teardown, so the registration cannot outlive the test that asked for it."""
    spec = importlib.util.spec_from_file_location("m5t_cells_rows_for_builder_test", REPO / "scripts" / "sdfbase" / "math500" / "cells.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop(spec.name, None)


def test_the_builder_writes_exactly_the_matrix_legs_and_nothing_else(full_store, tmp_path, raw_launcher_cells):
    """The synthetic store builder and the analysis both READ lane B's matrix
    (``scripts/sdfbase/math500/cells.py``) and rows (``sampling_rows.yaml``): the
    built legs are exactly the PUBLISHED wave's 24 (the tree's matrix also carries
    the eight-arm extension's rows, wave ``ext1``, which the builder writes only
    under :func:`math500_synthetic_store.ext_rows`), a triple the matrix does not
    name cannot be built, and the analysis' loader reads the same rows the builder
    wrote from."""
    # the launcher's matrix also carries the BLOCKED all-positions row (index 53): an EVIDENCE row of `arditi_allpos`, an id this
    # tree's cell grammar does not admit (its Class B registration lands in a separate change), so `load_cells` refuses the live
    # file by name — pinned in tests/test_eight_arms_math500_analysis.py. This test's subject is the PUBLISHED wave, so it reads
    # the matrix without that one row rather than asserting around the refusal.
    whole = M.load_cells([r for r in raw_launcher_cells.M5T_CELLS if r["cell"] != "arditi_allpos"])
    matrix = M.required_rows(whole, [M.WAVE_PUBLISHED])
    assert len(whole) > len(matrix) == 24 and matrix == S.cells_matrix()
    # the tree's matrix carries more waves than the published one, and a HELD row too (the rung's
    # no-switch rows wait for their family's certified render fact); both are design statuses the analysis reads
    assert {M.row_wave(r) for r in whole} > {M.WAVE_PUBLISHED}
    assert {M.row_status(r) for r in whole} <= set(M.ROW_STATUSES) and M.STATUS_LAUNCH in {M.row_status(r) for r in whole}
    assert {M.row_status(r) for r in matrix} == {M.STATUS_LAUNCH}
    built = sorted(str(p.relative_to(full_store)).split("/cells/")[0] for p in full_store.glob("controls/*/*/*/cells/*"))
    assert built == sorted(r["out_root"] for r in matrix)
    assert sorted(S.build_all(tmp_path / "again", rows=matrix[:3])) == sorted(
        M.leg_key(r["model_key"], r["cell"], r["sampling_mode"]) for r in matrix[:3])
    with pytest.raises(KeyError, match="gemma4e4b"):
        S.build_store(tmp_path / "x", model_key=GEMMA, cell=BARE, mode=BATTERY, correct=set())   # not in the matrix
    assert S.SAMPLING_ROWS == M.load_sampling_rows(REPO / "configs" / "sdfbase" / "math500" / "sampling_rows.yaml")


def test_full_synthetic_run_is_a_production_report(full_report):
    rep = full_report
    assert rep["unusable_legs"] == {}
    assert len(rep["legs"]) == 24 and len(rep["cells"]) == 24
    names = sorted((c["model_key"], c["mode"]) for c in rep["comparisons"])
    assert names == sorted([(GEMMA, PRIMARY), (QWEN, PRIMARY), (GLM, PRIMARY), (NEMO, PRIMARY),
                            (QWEN, GENERAL), (QWEN, BATTERY), (GLM, BATTERY), (NEMO, BATTERY)])
    assert all(c["usable"] and c["estimands"] is not None for c in rep["comparisons"])
    assert rep["gates"]["failed"] == [] and rep["gates"]["pending"] == []
    assert M.production_refusals(rep) == []
    assert rep["verdict"]["word"] is not None
    # the estimands block is keyed model -> mode -> branch, both branches
    for m in (GEMMA, QWEN, GLM, NEMO):
        est = rep["estimands"][m][PRIMARY]
        assert set(est) == set(M.BRANCHES)
        for b in M.BRANCHES:
            assert set(est[b]["stats"]) == {M.CONTRAST_OBJECT, M.CONTRAST_PROMPTED}
            assert est[b]["B"] == B_FAST and est[b]["seed"] == 0 and est[b]["n_items"] == 500
    # every store carries the same sha; the analysis code sha is a separate field
    assert rep["sha"] == S.SHA7 and rep["analysis_code_sha"] == S.FULL_SHA
    assert rep["analysis_code_sha"] != rep["sha"]
    # the at-cap RESULT block: the 12 primary cells first, the 12 sensitivity rows flagged, no at-cap draw in
    # the fixture, so the premise holds and there is no headline
    tr = rep["truncation"]
    assert tr["matrix_complete"] and tr["n_primary_cells_in_matrix"] == tr["n_primary_cells_read"] == 12 == tr["n_primary_cells_design"]
    assert [c["leg"] for c in tr["cells"] if c["primary"]] == \
        [M.leg_key(r["model_key"], r["cell"], r["sampling_mode"]) for r in rep["cells"] if r["sampling_mode"] == PRIMARY]
    assert sum(c["sensitivity"] for c in tr["cells"]) == 12 and all(c["sensitivity"] is not c["primary"] for c in tr["cells"])
    assert all(c["share"] == 0.0 and c["n_loop"] == c["n_truncated_reasoning"] == c["n_text_unavailable"] == 0 for c in tr["cells"])
    assert tr["premise_failed"] is False and tr["headline"] is None and tr["primary_over_1pct"] == [] == tr["primary_legs_not_read"]
    assert tr["primary_legs_absent_from_matrix"] == []           # the premise HOLDS only on the complete designed 12-cell read
    assert (tr["premise_rate"], tr["line_1pct"], tr["surfacing_threshold"], tr["cap"]) == (0.015, 0.01, 0.03, M.CAP)
    assert tr["loop_rule"] == M.LOOP_RULE_TEXT and "rows.jsonl" in tr["loop_rule_scope"]
    # the GLM fixture's two tag-free items are a counted share, the implicit-open families 0 by construction
    assert leg_of(rep, GLM, BARE)["thinking"]["n_no_reasoning_emitted"] == 2
    assert leg_of(rep, QWEN, BARE)["thinking"]["by_construction"] and leg_of(rep, NEMO, BARE)["thinking"]["by_construction"]
    assert leg_of(rep, GEMMA, BARE)["thinking"] == {**leg_of(rep, GEMMA, BARE)["thinking"], "n_no_reasoning_emitted": 0,
                                                     "by_construction": False, "split_mode": "record_only"}


def test_the_primary_contrasts_and_words(full_report):
    rep = full_report
    for m in (GEMMA, QWEN, GLM, NEMO):
        est = rep["estimands"][m][PRIMARY]
        so = est[M.BRANCH_PRIMARY]["stats"][M.CONTRAST_OBJECT]
        sp = est[M.BRANCH_PRIMARY]["stats"][M.CONTRAST_PROMPTED]
        assert so["point"] == pytest.approx(-0.08) and so["excludes_zero"] and so["point_sign"] == -1
        assert so["bound_low"] == so["bound_high"] == pytest.approx(-0.08)          # no censoring: a point bound
        assert sp["point"] == pytest.approx(-0.004) and not sp["excludes_zero"]
        # no censoring -> the two branches are identical
        assert est[M.BRANCH_SCORE_AS_IS]["stats"][M.CONTRAST_OBJECT]["point"] == so["point"]
        v = comparison(rep, m)["verdicts"]
        assert v[M.CONTRAST_OBJECT]["word"] == M.WORD_LOWER
        assert v[M.CONTRAST_PROMPTED]["word"] == M.WORD_NOT_RESOLVED
        assert "(a)" in v[M.CONTRAST_PROMPTED]["reason"]
        assert rep["verdict"]["words"][m][M.CONTRAST_OBJECT] == M.WORD_LOWER
        # the statement carries both branches, the McNemar sensitivity and the replicate signs
        st = v[M.CONTRAST_OBJECT]["statement"]
        assert "McNemar" in st and "replicate signs" in st and M.CONTRAST_OBJECT in st
    # a sensitivity comparison is filed under its own mode with the same estimand shape, no word
    sens = comparison(rep, QWEN, BATTERY)
    assert sens["primary"] is False and sens["estimands"][M.BRANCH_PRIMARY]["stats"][M.CONTRAST_OBJECT]["point"] \
        == pytest.approx(-0.08)
    assert sens["verdicts"][M.CONTRAST_OBJECT]["word"] is None
    assert QWEN in rep["verdict"]["words"] and BATTERY not in rep["verdict"]["words"][QWEN]


def test_mode_deltas_are_paired_per_cell_and_never_pooled(full_report):
    rep = full_report
    d = rep["mode_deltas"]
    assert set(d) == {QWEN, GLM, NEMO} and set(d[QWEN]) == {GENERAL, BATTERY}
    est = d[QWEN][GENERAL][M.BRANCH_PRIMARY]
    # general shifts every cell by +10 items -> +2.0 pp per cell, paired over the 500 ids
    for kind in (M.KIND_BARE, M.KIND_PROMPTED, M.KIND_OBJECT):
        s = est["stats"][M.mode_delta_contrast(kind, GENERAL)]
        assert s["point"] == pytest.approx(0.02) and s["excludes_zero"]
        assert s["legs"] == {f"{GENERAL}/{kind}": 1, f"{PRIMARY}/{kind}": -1}
    est_b = d[NEMO][BATTERY][M.BRANCH_PRIMARY]
    assert est_b["stats"][M.mode_delta_contrast(M.KIND_BARE, BATTERY)]["point"] == pytest.approx(0.01)
    # the primary estimand is IDENTICAL to a run over the primary rows alone: nothing was pooled
    assert rep["estimands"][QWEN][PRIMARY] == analyze(
        Path(rep["gen_root"]), rows=primary_rows(QWEN))["estimands"][QWEN][PRIMARY]


def test_a_sensitivity_store_swapped_into_the_primary_slot_refuses_by_identity(tmp_path):
    """A battery leg copied onto the primary out root is refused BY
    IDENTITY (its sampling block names the battery mode and the battery row),
    never silently computed into the primary comparison."""
    S.build_all(tmp_path, rows=rows_for(QWEN, PRIMARY, BATTERY))
    primary_root = S.out_root_of(tmp_path, QWEN, OBJECT, PRIMARY)
    battery_root = S.out_root_of(tmp_path, QWEN, OBJECT, BATTERY)
    shutil.rmtree(primary_root)
    shutil.copytree(battery_root, primary_root)
    rep = analyze(tmp_path, rows=rows_for(QWEN, PRIMARY, BATTERY))
    leg = leg_of(rep, QWEN, OBJECT, PRIMARY)
    assert leg["refused"] and "sampling" in leg["refused"] and BATTERY in leg["refused"]
    assert not comparison(rep, QWEN, PRIMARY)["usable"]
    assert comparison(rep, QWEN, BATTERY)["usable"]           # its own slot still reads its own store
    assert QWEN not in rep.get("estimands", {}) or PRIMARY not in rep["estimands"][QWEN]
    assert any("unusable" in r for r in M.production_refusals(rep))


# --------------------------------------------------------------------------- #
# the verdict rule and the censoring bound                                    #
# --------------------------------------------------------------------------- #


def _stat(point, lo, hi, blo=None, bhi=None):
    blo = point if blo is None else blo
    bhi = point if bhi is None else bhi
    return {"point": point, "ci_low": lo, "ci_high": hi, "bound_low": blo, "bound_high": bhi,
            "excludes_zero": bool(lo > 0 or hi < 0), "point_sign": (point > 0) - (point < 0),
            "sign_identified": bool(blo > 0 or bhi < 0), "replicate_signs": {"negative": 0, "zero": 0, "positive": 0},
            "bound_ci_low": blo, "bound_ci_high": bhi}


def _est(p, q):
    return {M.BRANCH_PRIMARY: {"stats": {"c": p}, "n_replicates_used": 1, "level": 0.95},
            M.BRANCH_SCORE_AS_IS: {"stats": {"c": q}, "n_replicates_used": 1, "level": 0.95}}


def test_verdict_rule_unit():
    ok = M.verdict_for(_est(_stat(-0.08, -0.10, -0.06, -0.09, -0.07), _stat(-0.085, -0.11, -0.06)), "c")
    assert ok["word"] == M.WORD_LOWER and ok["side"] == -1 and all(ok["conditions"].values())
    up = M.verdict_for(_est(_stat(0.08, 0.06, 0.10), _stat(0.085, 0.06, 0.11)), "c")
    assert up["word"] == M.WORD_HIGHER
    # (a) the primary CI covers zero
    a = M.verdict_for(_est(_stat(-0.01, -0.03, 0.01), _stat(-0.02, -0.03, -0.01)), "c")
    assert a["word"] == M.WORD_NOT_RESOLVED and "(a)" in a["reason"]
    # (b) score-as-is covers zero
    b = M.verdict_for(_est(_stat(-0.03, -0.05, -0.01), _stat(-0.01, -0.03, 0.01)), "c")
    assert b["word"] == M.WORD_NOT_RESOLVED and "(b)" in b["reason"]
    # (b) both exclude zero but on opposite sides
    b2 = M.verdict_for(_est(_stat(-0.03, -0.05, -0.01), _stat(0.03, 0.01, 0.05)), "c")
    assert b2["word"] == M.WORD_NOT_RESOLVED and "(b)" in b2["reason"] and "(c)" in b2["reason"]
    # (d) the censoring bound covers zero while both CIs exclude it
    d = M.verdict_for(_est(_stat(-0.03, -0.05, -0.01, -0.04, 0.005), _stat(-0.03, -0.05, -0.01)), "c")
    assert d["word"] == M.WORD_NOT_RESOLVED and "(d)" in d["reason"]


def test_one_item_edge_passes_the_bound_but_stays_unresolved(tmp_path):
    """Object correct = bare minus ONE item — the censoring bound is a
    point off zero (sign 'identified' on the sample), but the bootstrap CI
    covers zero; the word is withheld."""
    S.build_all(tmp_path, rows=primary_rows(GEMMA),
                overrides={(GEMMA, OBJECT, PRIMARY): {"correct": S.BARE_CORRECT - {7}}})
    rep = analyze(tmp_path, rows=primary_rows(GEMMA))
    so = rep["estimands"][GEMMA][PRIMARY][M.BRANCH_PRIMARY]["stats"][M.CONTRAST_OBJECT]
    assert so["point"] == pytest.approx(-0.002) and so["sign_identified"] and not so["excludes_zero"]
    v = comparison(rep, GEMMA)["verdicts"][M.CONTRAST_OBJECT]
    assert v["word"] == M.WORD_NOT_RESOLVED and "(a)" in v["reason"] and v["conditions"]["d_bound_excludes_zero_same_side"]


def test_censoring_bounds_bracket_both_branch_points_and_the_notes_read_the_masks(tmp_path):
    S.build_all(tmp_path, rows=primary_rows(QWEN), overrides={(QWEN, OBJECT, PRIMARY): {"unscored": S.LAST4 | {0, 1}}})
    rep = analyze(tmp_path, rows=primary_rows(QWEN))
    est = rep["estimands"][QWEN][PRIMARY]
    p, q = est[M.BRANCH_PRIMARY]["stats"][M.CONTRAST_OBJECT], est[M.BRANCH_SCORE_AS_IS]["stats"][M.CONTRAST_OBJECT]
    assert p["bound_low"] <= min(p["point"], q["point"]) <= max(p["point"], q["point"]) <= p["bound_high"]
    assert p["bound_low"] < p["bound_high"]
    assert est[M.BRANCH_PRIMARY]["n_censored_by_leg"][M.KIND_OBJECT] == 6
    assert est[M.BRANCH_SCORE_AS_IS]["n_censored_by_leg"][M.KIND_OBJECT] == 0
    # the exact test's one-sided note is mask-dependent (the parent's discipline)
    mc = comparison(rep, QWEN)["mcnemar"][M.CONTRAST_OBJECT]
    assert mc[M.BRANCH_PRIMARY]["fully_scored"] is False and mc[M.BRANCH_SCORE_AS_IS]["fully_scored"] is True
    assert mc[M.BRANCH_PRIMARY]["one_sided_discordance"] and "registered conditional bootstrap" in mc[M.BRANCH_PRIMARY]["note"]
    assert "fully scored" in mc[M.BRANCH_SCORE_AS_IS]["note"]
    assert "replicate_signs" in mc[M.BRANCH_PRIMARY]
    # censored items are reported in the controls and the cap gate still passes (6 / 500 at cap = 1.2 %)
    assert rep["gates"]["cap"]["status"] == M.GATE_PASS
    assert rep["gates"]["cap"]["legs"][M.leg_key(QWEN, OBJECT, PRIMARY)]["n_case_b_unterminated_at_cap"] == 6


# --------------------------------------------------------------------------- #
# gate 0: population / completeness                                            #
# --------------------------------------------------------------------------- #


def test_population_gate_refuses_499_ids_a_duplicate_and_a_rows_count_mismatch(tmp_path):
    rows = primary_rows(GEMMA)
    # (i) 499 unique attempted ids on one leg
    S.build_all(tmp_path / "a", rows=rows, overrides={(GEMMA, OBJECT, PRIMARY): {"ids": S.IDS[:499]}})
    rep = analyze(tmp_path / "a", rows=rows)
    leg = leg_of(rep, GEMMA, OBJECT)
    assert leg["refused"] and "499" in leg["refused"]
    assert rep["gates"]["population"]["status"] == M.GATE_FAIL
    # (ii) a duplicated id written into the record after the fact
    S.build_all(tmp_path / "b", rows=rows)
    cd = Path(leg_of(analyze(tmp_path / "b", rows=rows), GEMMA, BARE)["cell_dir"])
    p = cd / C.TASK_OUTCOMES_DIR / C.task_outcomes_filename(S.TASK)
    doc = json.loads(p.read_text())
    doc["items"][1]["item_id"] = doc["items"][0]["item_id"]
    p.write_text(json.dumps(doc))
    leg = leg_of(analyze(tmp_path / "b", rows=rows), GEMMA, BARE)
    assert leg["refused"] and "duplicate" in leg["refused"]
    # (iii) rows.jsonl != n_scored
    S.build_all(tmp_path / "c", rows=rows, overrides={(GEMMA, PROMPTED, PRIMARY): {"rows_edit": lambda rs: rs.pop()}})
    leg = leg_of(analyze(tmp_path / "c", rows=rows), GEMMA, PROMPTED)
    assert leg["refused"] and "rows.jsonl" in leg["refused"]
    # (iv) the attempted ids are not the canonical MATH-500 set
    S.build_all(tmp_path / "d", rows=rows)
    rep = analyze(tmp_path / "d", rows=rows, math500_ids=[*S.IDS[:499], "math_deadbeef"])
    assert rep["gates"]["population"]["status"] == M.GATE_FAIL and "canonical" in rep["gates"]["population"]["detail"]
    # (v) no canonical set supplied -> PENDING, never waivable
    rep = analyze(tmp_path / "d", rows=rows, math500_ids=None, allow_pending_gates=True)
    assert rep["gates"]["population"]["status"] == M.GATE_PENDING
    assert rep["gates"]["pending_unwaivable"] and rep["verdict"]["word"] is None
    assert M.production_refusals(rep)


def test_a_missing_leg_is_recorded_and_the_other_models_still_run(tmp_path):
    S.build_all(tmp_path, rows=primary_rows(GEMMA) + primary_rows(QWEN))
    shutil.rmtree(S.out_root_of(tmp_path, QWEN, PROMPTED, PRIMARY))
    rep = analyze(tmp_path, rows=primary_rows(GEMMA) + primary_rows(QWEN))
    assert leg_of(rep, QWEN, PROMPTED)["missing"]
    assert not comparison(rep, QWEN)["usable"] and comparison(rep, GEMMA)["usable"]
    assert GEMMA in rep["estimands"] and QWEN not in rep["estimands"]
    assert rep["verdict"]["word"] is None and M.production_refusals(rep)


def test_a_partial_store_and_a_diagnostic_store_are_refused_by_name(tmp_path):
    rows = primary_rows(GEMMA)
    S.build_all(tmp_path / "p", rows=rows,
                overrides={(GEMMA, BARE, PRIMARY): {"ids": S.IDS[:20], "limits": {"inspect_limit": 20}, "n_expected_full": 500}})
    leg = leg_of(analyze(tmp_path / "p", rows=rows), GEMMA, BARE)
    assert leg["refused"] and "PARTIAL" in leg["refused"]

    def diag(m):
        m[C.MANIFEST_KEY_TINYCAP]["diagnostic"] = True

    S.build_all(tmp_path / "d", rows=rows, overrides={(GEMMA, BARE, PRIMARY): {"manifest_edit": diag}})
    leg = leg_of(analyze(tmp_path / "d", rows=rows), GEMMA, BARE)
    assert leg["refused"] and "diagnostic" in leg["refused"]


# --------------------------------------------------------------------------- #
# gate 1: the ledger confirmation                                              #
# --------------------------------------------------------------------------- #


def _one_leg_report(tmp_path, name, model, cell, mode=PRIMARY, *, overrides, **analyze_kw):
    """Build the (model, mode) comparison's three legs with ``overrides`` on ONE
    of them, then analyze — the shape every negative gate case shares."""
    rows = rows_for(model, mode)
    S.build_all(tmp_path / name, rows=rows, overrides={(model, cell, mode): overrides})
    return analyze(tmp_path / name, rows=rows, **analyze_kw)


def _ledger_case(tmp_path, name, edit, model=GEMMA, cell=OBJECT):
    rep = _one_leg_report(tmp_path, name, model, cell, overrides={"ledger_edit": edit})
    return rep, rep["gates"]["ledger"]


def _task_row(rs, k=0):
    """The k-th TASK row of a ledger (under identity version 2 the probes' labelled
    rows come first, in the driver's order)."""
    return [r for r in rs if r["task"] == S.TASK][k]


def test_ledger_gate_refuses_by_name(tmp_path):
    def undeclared(rs):
        _task_row(rs, 3)["error_code"] = "thinking_on_undeclared"

    rep, g = _ledger_case(tmp_path, "code", undeclared)
    assert g["status"] == M.GATE_FAIL and "thinking_on_undeclared" in g["detail"]
    assert g["legs"][M.leg_key(GEMMA, OBJECT, PRIMARY)]["error_code"] == {"thinking_on_undeclared": 1}
    assert M.production_refusals(rep)

    def off(rs):
        _task_row(rs)["thinking"] = False

    assert "thinking" in _ledger_case(tmp_path, "off", off)[1]["detail"]

    def cap(rs):
        _task_row(rs)["max_tokens_requested"] = 4096

    assert "max_tokens_requested" in _ledger_case(tmp_path, "cap", cap)[1]["detail"]

    def rejected(rs):
        _task_row(rs).update(rejected=True, status=504, error_class="server_error", finish_reason=None)

    d = _ledger_case(tmp_path, "rej", rejected)[1]["detail"]
    assert "rejected" in d and "censor" in d

    def short(rs):
        rs.remove(_task_row(rs))

    assert "499" in _ledger_case(tmp_path, "short", short)[1]["detail"]

    def dup_sample(rs):
        _task_row(rs, 1)["sample"] = _task_row(rs)["sample"]

    assert "unique" in _ledger_case(tmp_path, "dup", dup_sample)[1]["detail"]

    def length(rs):
        _task_row(rs)["finish_reason"] = "length"

    assert "n_at_cap" in _ledger_case(tmp_path, "len", length)[1]["detail"]

    def grader(rs):
        rs.append({**_task_row(rs), "sample": "grader/0", "vmodel": "petri__base", "thinking": False,
                   "max_tokens_requested": 4096})

    rep, _g = _ledger_case(tmp_path, "grader", grader)
    assert rep["gates"]["scoring_path"]["status"] == M.GATE_FAIL and "grader" in rep["gates"]["scoring_path"]["detail"]
    # the probes travel THROUGH the ledger under identity version 2 (FC4 (2)): split off the task rules (500 calls,
    # thinking-ON, the cap), counted by kind, and held to "no probe was rejected or errored"
    led = leg_of(_one_leg_report(tmp_path, "probes", GEMMA, OBJECT, overrides={}), GEMMA, OBJECT)["ledger"]
    n_probe = len(S.probe_records_one_attempt(OBJECT, fitted=True))
    assert led["passed"] and led["n_calls"] == 500 and led["n_probe_rows"] == n_probe and led["n_other_task_rows"] == 0
    assert led["probe_rows_by_kind"] == {C.probe_task_label(C.PROBE_KIND_BITES): 2 * S.N_BITES_PROMPTS,
                                         C.probe_task_label(C.PROBE_KIND_RENDER): 2, C.probe_task_label(C.PROBE_KIND_LENS_ON_BITE): 2}

    def probe_rejected(rs):
        rs[0].update(rejected=True, status=400, error_class=C.LEDGER_ERROR_CLASS_BAD_REQUEST_OTHER)

    d = _ledger_case(tmp_path, "prej", probe_rejected)[1]["detail"]
    assert "probe row" in d and C.probe_task_label(C.PROBE_KIND_BITES) in d


def test_ledger_gate_missing_file_upstream_mismatch_and_resume_rows(tmp_path):
    rows = primary_rows(GEMMA)
    S.build_all(tmp_path / "nofile", rows=rows, overrides={(GEMMA, BARE, PRIMARY): {"ledger": False}})
    g = analyze(tmp_path / "nofile", rows=rows)["gates"]["ledger"]
    assert g["status"] == M.GATE_FAIL and "no usage_ledger.jsonl" in g["detail"]

    def upstream(m):
        m[S.MANIFEST_KEY_BINDING]["ledger_upstream"] = "http://127.0.0.1:1"

    S.build_all(tmp_path / "up", rows=rows, overrides={(GEMMA, BARE, PRIMARY): {"manifest_edit": upstream}})
    g = analyze(tmp_path / "up", rows=rows)["gates"]["ledger"]
    assert g["status"] == M.GATE_FAIL and "ledger_upstream" in g["detail"]
    # same-cap re-sends with ONE recorded run are the HARNESS's (the client retry ladder) — they PASS with a readout;
    # a recorded resume's re-sends are binned into the run that made them and counted apart
    S.build_all(tmp_path / "r1", rows=rows, overrides={(GEMMA, BARE, PRIMARY): {"resume_rows": 3, "resumes": 0}})
    rep = analyze(tmp_path / "r1", rows=rows)
    led = rep["gates"]["ledger"]["legs"][M.leg_key(GEMMA, BARE, PRIMARY)]
    assert rep["gates"]["ledger"]["status"] == M.GATE_PASS and led["n_resume_rows"] == 3 and led["resume_attempts"] == 1
    assert (led["redraws"]["n_redraw_calls"], led["redraws"]["n_items_redrawn"], led["redraws"]["n_resume_calls"],
            led["redraws"]["attributed"]) == (3, 3, 0, True)
    S.build_all(tmp_path / "r2", rows=rows, overrides={(GEMMA, BARE, PRIMARY): {"resume_rows": 3}})
    rep = analyze(tmp_path / "r2", rows=rows)
    assert rep["gates"]["ledger"]["status"] == M.GATE_PASS
    led = rep["gates"]["ledger"]["legs"][M.leg_key(GEMMA, BARE, PRIMARY)]
    assert led["n_resume_rows"] == 3 and led["resume_attempts"] == 2
    assert (led["redraws"]["n_redraw_calls"], led["redraws"]["n_items_redrawn"], led["redraws"]["n_resume_calls"],
            led["redraws"]["max_calls_per_item"], led["redraws"]["attributed"]) == (0, 0, 3, 2, True)
    assert rep["provenance"]["cells"][M.leg_key(GEMMA, BARE, PRIMARY)]["n_resume_rows"] == 3
    # a third call of a resumed sample inside the resume's run is the harness's re-send there: within the ladder, attributed
    def thrice(rs):
        rs.append({**_task_row(rs), "ts": "2026-01-01T02:00:00Z", "request_id": "f" * 32})

    S.build_all(tmp_path / "r3", rows=rows, overrides={(GEMMA, BARE, PRIMARY): {"resume_rows": 1, "ledger_edit": thrice}})
    led = analyze(tmp_path / "r3", rows=rows)["legs"][M.leg_key(GEMMA, BARE, PRIMARY)]["ledger"]
    assert (led["redraws"]["n_redraw_calls"], led["redraws"]["n_resume_calls"], led["redraws"]["max_calls_per_item"]) == (1, 1, 3)
    # a run record with no readable ts cannot place a re-send in its run: refused by name while re-sends exist
    def no_ts(m):
        m[M.MANIFEST_KEY_RESUME_ATTEMPTS][1]["ts"] = "yesterday"

    S.build_all(tmp_path / "r4", rows=rows, overrides={(GEMMA, BARE, PRIMARY): {"resume_rows": 2, "manifest_edit": no_ts}})
    g = analyze(tmp_path / "r4", rows=rows)["gates"]["ledger"]
    assert g["status"] == M.GATE_FAIL and "no readable ts" in g["detail"]


# --------------------------------------------------------------------------- #
# gate 2: request evidence and identity                                       #
# --------------------------------------------------------------------------- #


def _first_task_record(recs):
    return next(i for i, r in enumerate(recs) if r["kind"] == C.REQUEST_EVIDENCE_KIND_TASK)


def _req_case(tmp_path, name, **kw):
    rep = _one_leg_report(tmp_path, name, NEMO, OBJECT, overrides=kw)
    return rep, rep["gates"]["identity"]


def test_request_evidence_gate_refuses_a_missing_record_a_declaration_and_a_missing_file(tmp_path):
    _rep, g = _req_case(tmp_path, "nofile", request_evidence=False)
    assert g["status"] == M.GATE_FAIL and "request_evidence.jsonl" in g["detail"]

    def declared(recs):
        recs[5]["petri_thinking_on_present"] = True
        recs[5]["petri_thinking_on_value"] = True

    _rep, g = _req_case(tmp_path, "decl", request_evidence_edit=declared)
    assert g["status"] == M.GATE_FAIL and "petri_thinking_on" in g["detail"]

    def missing(recs):
        recs.pop(_first_task_record(recs) + 7)

    _rep, g = _req_case(tmp_path, "miss", request_evidence_edit=missing)
    assert g["status"] == M.GATE_FAIL and "no request-evidence record" in g["detail"]

    def version(recs):
        recs[0]["request_evidence_version"] = 1          # the legacy fragment-digest record

    _rep, g = _req_case(tmp_path, "ver", request_evidence_edit=version)
    assert g["status"] == M.GATE_FAIL and "request_evidence_version" in g["detail"]

    def not_on(recs):
        recs[_first_task_record(recs)]["enable_thinking_value"] = False

    _rep, g = _req_case(tmp_path, "off", request_evidence_edit=not_on)
    assert g["status"] == M.GATE_FAIL and "enable_thinking" in g["detail"]
    # the producer's records pass: 500 task records plus every probe record ONE run of the driver writes
    ok = _req_case(tmp_path, "ok")[1]
    assert ok["status"] == M.GATE_PASS
    n_probes = len(S.probe_records_one_attempt(OBJECT, fitted=True))
    assert n_probes == 2 * S.N_BITES_PROMPTS + 2 + 2
    assert ok["legs"][M.leg_key(NEMO, OBJECT, PRIMARY)]["request_evidence"]["n_records"] == 500 + n_probes


def _identity_case(tmp_path, name, edit, model=QWEN, cell=OBJECT, mode=PRIMARY, key="manifest_edit"):
    rep = _one_leg_report(tmp_path, name, model, cell, mode, overrides={key: edit})
    return rep, leg_of(rep, model, cell, mode)


def test_identity_refusals_name_the_field(tmp_path):
    def version(m):
        m[C.MANIFEST_KEY_TINYCAP][S.IDENTITY_KEY_VERSION] = 1

    assert "identity_version" in _identity_case(tmp_path, "v1", version)[1]["refused"]

    def cap(m):
        m[C.MANIFEST_KEY_TINYCAP][C.IDENTITY_KEY_MAX_TOKENS_BY_TASK] = {S.TASK: 4096}

    assert "max_tokens_by_task" in _identity_case(tmp_path, "cap", cap)[1]["refused"]

    def declared(m):
        m[C.MANIFEST_KEY_TINYCAP][C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT] = ["thinking"]

    assert "serve_out_of_support" in _identity_case(tmp_path, "decl", declared)[1]["refused"]

    def oos_on_bare(m):
        m[C.MANIFEST_KEY_TINYCAP][C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING] = True

    assert "out_of_support_thinking" in _identity_case(tmp_path, "oosb", oos_on_bare, cell=BARE)[1]["refused"]

    def oos_off_object(m):
        del m[C.MANIFEST_KEY_TINYCAP][C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING]

    assert "out_of_support_thinking" in _identity_case(tmp_path, "ooso", oos_off_object)[1]["refused"]

    def rev(m):
        m["model_revision"] = "0" * 40

    assert "model_revision" in _identity_case(tmp_path, "rev", rev)[1]["refused"]

    def digest(m):
        m[C.MANIFEST_KEY_PROVENANCE]["artifact_sha256"] = "f" * 64

    assert "pin" in _identity_case(tmp_path, "pin", digest)[1]["refused"]

    def provenance_class(m):
        m[S.MANIFEST_KEY_TRANSFER_PROVENANCE]["header_check_class"] = "document_fit_served_header_located"

    assert "header_check_class" in _identity_case(tmp_path, "cls", provenance_class)[1]["refused"]

    def provenance_absent(m):
        del m[S.MANIFEST_KEY_TRANSFER_PROVENANCE]

    assert "transfer_provenance" in _identity_case(tmp_path, "tpa", provenance_absent)[1]["refused"]

    def tuple_name(m):
        m[S.MANIFEST_KEY_OOS_EXPERIMENT] = "fst_exp1_qwen4b_gsm8k"

    assert "oos_experiment" in _identity_case(tmp_path, "tup", tuple_name)[1]["refused"]

    def cell_id(m):
        m["cell_id"] = BARE

    assert "cell_id" in _identity_case(tmp_path, "cid", cell_id)[1]["refused"]

    def retry(m):
        m[C.MANIFEST_KEY_TINYCAP][C.IDENTITY_KEY_MATH500_RETRY_MAX_TOKENS] = 8192

    assert "math500_retry_max_tokens" in _identity_case(tmp_path, "retry", retry)[1]["refused"]


def test_sampling_row_rule_three_tables_exact_keys(tmp_path):
    """The manifest's sampling block equals ITS mode's literal row —
    a key the row does not name refuses, a key it names but the block lacks
    refuses, a differing value refuses, and the summary's three numbers must
    agree; the battery and general rows are held to their own tables."""
    def extra_key(m):
        m[C.MANIFEST_KEY_SAMPLING]["repetition_penalty"] = 1.1        # primary qwen names none

    r = _identity_case(tmp_path, "extra", extra_key)[1]["refused"]
    assert "repetition_penalty" in r and "not name" in r

    def missing_key(m):
        del m[C.MANIFEST_KEY_SAMPLING]["top_k"]

    assert "top_k" in _identity_case(tmp_path, "miss", missing_key)[1]["refused"]

    def value(m):
        m[C.MANIFEST_KEY_SAMPLING]["temperature"] = 0.7

    assert "temperature" in _identity_case(tmp_path, "val", value)[1]["refused"]

    def summary_t(s):
        s["temperature"] = 0.7

    assert "summary" in _identity_case(tmp_path, "sum", summary_t, key="summary_edit")[1]["refused"]

    def mode(m):
        m[C.MANIFEST_KEY_SAMPLING]["mode"] = BATTERY

    assert BATTERY in _identity_case(tmp_path, "mode", mode)[1]["refused"]
    # the battery row is held to the battery table: the GLM battery row's repetition_penalty is REQUIRED there
    def drop_rep(m):
        del m[C.MANIFEST_KEY_SAMPLING]["repetition_penalty"]

    r = _identity_case(tmp_path, "glmb", drop_rep, model=GLM, mode=BATTERY)[1]["refused"]
    assert "repetition_penalty" in r
    # the sampling_ref on every row must be the block's digest
    def bad_ref(rows):
        e = json.loads(rows[0]["extra_json"])
        e["sampling_ref"] = "000000000000"
        rows[0]["extra_json"] = json.dumps(e)

    rep, _leg = _identity_case(tmp_path, "ref", bad_ref, key="rows_edit")
    assert rep["gates"]["identity"]["status"] == M.GATE_FAIL and "sampling_ref" in rep["gates"]["identity"]["detail"]
    # the identity hash of the tinycap block must carry the same numbers as the sampling block
    def identity_t(m):
        m[C.MANIFEST_KEY_TINYCAP]["temperature"] = 0.7

    assert "identity" in _identity_case(tmp_path, "idt", identity_t)[1]["refused"]


def test_sampling_rows_loader_json_and_yaml_and_contracts_cross_check(tmp_path):
    p = S.write_sampling_rows(tmp_path / "rows.json")
    got = M.load_sampling_rows(p)
    # the registry's modes: the three MATH-500 modes + the ONE `card` row the GSM8K Llama rows run (llama-paper-rows P6; one-way)
    assert got[PRIMARY][QWEN]["temperature"] == 0.6 and set(got) == {PRIMARY, GENERAL, BATTERY, C.SAMPLING_CARD}
    yaml = pytest.importorskip("yaml")
    (tmp_path / "rows.yaml").write_text(yaml.safe_dump({"modes": S.SAMPLING_ROWS}))
    assert M.load_sampling_rows(tmp_path / "rows.yaml") == got
    # the PINNED run config (lane B's rows: {model: {mode: row}} shape) reads to the same table the builder wrote from
    real = M.load_sampling_rows(REPO / "configs" / "sdfbase" / "math500" / "sampling_rows.yaml")
    assert real == got == S.SAMPLING_ROWS
    assert set(real) == {PRIMARY, GENERAL, BATTERY, C.SAMPLING_CARD} and set(real[BATTERY]) == {QWEN, GLM, NEMO}
    assert set(real[C.SAMPLING_CARD]) == {"llama8b"} and real[C.SAMPLING_CARD]["llama8b"]["temperature"] == C.SAMPLING_BY_MODEL_KEY["llama8b"].temperature
    (tmp_path / "undeclared.yaml").write_text(yaml.safe_dump({"modes": [PRIMARY], "rows": {QWEN: {BATTERY: real[BATTERY][QWEN]}}}))
    with pytest.raises(M.AnalysisError, match="not declared"):
        M.load_sampling_rows(tmp_path / "undeclared.yaml")
    # a mode this tree's contracts KNOW is cross-checked: `card` with the wrong numbers refuses
    bad = {**S.SAMPLING_ROWS, C.SAMPLING_CARD: {QWEN: {"temperature": 0.1, "top_p": 0.95, "top_k": 20}}}
    with pytest.raises(M.AnalysisError, match="contracts.sampling_for"):
        M.load_sampling_rows(bad)
    row = C.sampling_for(QWEN, C.SAMPLING_CARD)
    ok = {**S.SAMPLING_ROWS, C.SAMPLING_CARD: {QWEN: {"temperature": row.temperature, "top_p": row.top_p,
                                                     "top_k": row.top_k}}}
    assert M.load_sampling_rows(ok)[C.SAMPLING_CARD][QWEN]["top_k"] == 20


def test_stamp_triple_must_equal_the_matrix_row(tmp_path):
    def policy(s):
        s["resumed_policy"] = "conditional"

    rep, _leg = _identity_case(tmp_path, "pol", policy, key="summary_edit")
    assert rep["gates"]["identity"]["status"] == M.GATE_FAIL and "resumed_policy" in rep["gates"]["identity"]["detail"]

    def render(s):
        s["render_regime"] = C.RENDER_REGIME_THINKING_OFF

    rep, _ = _identity_case(tmp_path, "ren", render, key="summary_edit")
    assert "render_regime" in rep["gates"]["identity"]["detail"]

    def mixed(s):
        s["n_render_regimes"] = 2

    rep, _ = _identity_case(tmp_path, "mix", mixed, key="summary_edit")
    assert "MIXED" in rep["gates"]["identity"]["detail"]

    def absent(s):
        s["salt_regime"] = None

    rep, _ = _identity_case(tmp_path, "abs", absent, key="summary_edit")
    assert rep["gates"]["identity"]["status"] == M.GATE_FAIL and "salt_regime" in rep["gates"]["identity"]["detail"]

    def vocab(s):
        s["salt_regime"] = "shared"

    rep, _ = _identity_case(tmp_path, "voc", vocab, key="summary_edit")
    assert "off-contract" in rep["gates"]["identity"]["detail"]


# --------------------------------------------------------------------------- #
# gate 3: certification (A) / (B′) / (C), the render facts                     #
# --------------------------------------------------------------------------- #


def _cert_case(tmp_path, name, edit, model=NEMO, cell=OBJECT, **kw):
    rep = _one_leg_report(tmp_path, name, model, cell, overrides={"evidence_edit": edit}, **kw)
    return rep, rep["gates"]["certification"]


def test_certification_gate_refusals(tmp_path):
    """Every field of the producer's (A) / (B′) / (C) records, mutated on a
    store built THROUGH the producer helpers, refuses NAMING the field."""
    def not_certified(ev):
        ev[S.EVIDENCE_RENDER_CERTIFIED]["render_certified"] = False

    g = _cert_case(tmp_path, "a", not_certified)[1]
    assert g["status"] == M.GATE_FAIL and "render_certified.render_certified" in g["detail"]

    def ids(ev):
        ev[S.EVIDENCE_RENDER_CERTIFIED]["measured_header_ids_by_arm"]["on"] = [10, 1503, 19464, 1010, 12, 13]

    assert "measured_header_ids_by_arm" in _cert_case(tmp_path, "ids", ids)[1]["detail"]

    def source(ev):
        ev[S.EVIDENCE_RENDER_CERTIFIED]["render_source"] = "local_tokenizer"

    assert "served_tokenize" in _cert_case(tmp_path, "src", source)[1]["detail"]

    def regime(ev):
        ev[S.EVIDENCE_RENDER_CERTIFIED]["render_regime_certified"] = None

    d = _cert_case(tmp_path, "reg", regime)[1]["detail"]
    assert "render_regime_certified" in d and "never run" in d

    def no_position(ev):
        del ev[S.EVIDENCE_LENS_POSITION]

    assert "lens_position_attested" in _cert_case(tmp_path, "nop", no_position)[1]["detail"]

    def header_located(ev):
        ev[S.EVIDENCE_LENS_POSITION]["served_positions"] = ["response"]

    d = _cert_case(tmp_path, "hl", header_located)[1]["detail"]
    assert "header-located" in d or "header_located" in d

    def mixed(ev):
        ev[S.EVIDENCE_LENS_POSITION]["served_positions"] = ["all", "response"]

    assert _cert_case(tmp_path, "mixed", mixed)[1]["status"] == M.GATE_FAIL

    def own_key(ev):
        ev[S.EVIDENCE_LENS_POSITION]["own_key"] = S.harness_id_of(OBJECT) + "@thinking"

    assert "own_key" in _cert_case(tmp_path, "own", own_key)[1]["detail"]

    def digest(ev):
        ev[S.EVIDENCE_LENS_POSITION]["object_digest"]["scalar"] = "f" * 64

    assert "object_digest.scalar" in _cert_case(tmp_path, "dig", digest)[1]["detail"]

    def by_cell(ev):
        ev[S.EVIDENCE_LENS_POSITION]["object_digest"]["by_cell"] = {S.harness_id_of(BARE): S.object_pin(NEMO)}

    assert "object_digest.by_cell" in _cert_case(tmp_path, "byc", by_cell)[1]["detail"]

    def policy(ev):
        ev[S.EVIDENCE_LENS_POSITION]["attested_policy"] = "conditional"

    assert "attested_policy" in _cert_case(tmp_path, "pol", policy)[1]["detail"]

    def stamps(ev):
        ev[S.EVIDENCE_LENS_POSITION]["expected_stamps"] = ["request-unique", "always", "thinking_on"]

    assert "expected_stamps" in _cert_case(tmp_path, "stm", stamps)[1]["detail"]

    def lp_failed(ev):
        ev[S.EVIDENCE_LENS_POSITION] = {"verified": False, "error": "OOS_DECLARATION_UNATTESTED: no map"}

    d = _cert_case(tmp_path, "lpf", lp_failed)[1]["detail"]
    assert "lens_position_attested" in d and "OOS_DECLARATION_UNATTESTED" in d

    def no_bite(ev):
        ev[S.EVIDENCE_LENS_ON_BITE]["reasoning_present"] = False

    assert "lens_on_bite.reasoning_present" in _cert_case(tmp_path, "bite", no_bite)[1]["detail"]

    def bite_same(ev):
        ev[S.EVIDENCE_LENS_ON_BITE]["differs_from_base"] = False

    assert "differs_from_base" in _cert_case(tmp_path, "same", bite_same)[1]["detail"]

    def bite_failed(ev):
        ev[S.EVIDENCE_LENS_ON_BITE] = {"applicable": True, "verified": False, "error": "predicate (C) lens-ON bite failed"}

    assert "predicate (C)" in _cert_case(tmp_path, "bitef", bite_failed)[1]["detail"]

    def bite_vmodel(ev):
        ev[S.EVIDENCE_LENS_ON_BITE]["lens_vmodel"] = S.vmodel_of(BARE)

    assert "lens_vmodel" in _cert_case(tmp_path, "bitev", bite_vmodel)[1]["detail"]

    def bite_absent(ev):
        del ev[S.EVIDENCE_LENS_ON_BITE]

    assert "lens_on_bite" in _cert_case(tmp_path, "biteabs", bite_absent)[1]["detail"]

    def bites(ev):
        ev["steering_bites"]["verified"] = False

    assert "steering bites" in _cert_case(tmp_path, "sb", bites)[1]["detail"]

    def probe_arm(ev):
        ev["steering_bites"]["probe_render_arm"] = C.THINKING_ON

    assert "probe_render_arm" in _cert_case(tmp_path, "arm", probe_arm)[1]["detail"]

    def twin(ev):
        ev["steering_bites"]["twin"] = "none"

    assert "twin" in _cert_case(tmp_path, "twin", twin)[1]["detail"]

    def n_probes(ev):
        del ev["steering_bites"]["n_probes"]

    assert "n_probes" in _cert_case(tmp_path, "np", n_probes)[1]["detail"]
    # a bare cell must NOT carry the lens blocks, and its bites block reads applicable: false
    def lens_on_bare(ev):
        ev[S.EVIDENCE_LENS_POSITION] = {"served_positions": ["all"]}

    assert "lens_position_attested" in _cert_case(tmp_path, "bare", lens_on_bare, cell=BARE)[1]["detail"]

    def bites_on_bare(ev):
        ev["steering_bites"] = {"applicable": True, "verified": True}

    assert "non-lens" in _cert_case(tmp_path, "sbbare", bites_on_bare, cell=BARE)[1]["detail"]


def test_route_lens_served_positions_is_the_selected_cells_list(tmp_path):
    """The driver copies the SELECTED cell's served modes (a list) into
    ``route.lens_served_positions`` — the reader reads that list: the expected
    list passes, another mode fails, null on a fitted cell fails, a list on a
    non-lens cell fails, an ABSENT key is the one waivable pending stamp, and
    the manifest's top-level copy must agree."""
    rows = primary_rows(GLM)

    def wrong(m):
        m[C.MANIFEST_KEY_ROUTE][S.ROUTE_KEY_LENS_SERVED_POSITIONS] = ["last"]
        m[C.LENS_SERVED_POSITIONS_KEY] = ["last"]

    S.build_all(tmp_path / "w", rows=rows, overrides={(GLM, OBJECT, PRIMARY): {"manifest_edit": wrong}})
    g = analyze(tmp_path / "w", rows=rows)["gates"]["certification"]
    assert g["status"] == M.GATE_FAIL and "route.lens_served_positions" in g["detail"] and "'last'" in g["detail"]

    def null(m):
        m[C.MANIFEST_KEY_ROUTE][S.ROUTE_KEY_LENS_SERVED_POSITIONS] = None
        m[C.LENS_SERVED_POSITIONS_KEY] = None

    S.build_all(tmp_path / "n", rows=rows, overrides={(GLM, OBJECT, PRIMARY): {"manifest_edit": null}})
    g = analyze(tmp_path / "n", rows=rows)["gates"]["certification"]
    assert g["status"] == M.GATE_FAIL and "route.lens_served_positions = None" in g["detail"]

    def on_bare(m):
        m[C.MANIFEST_KEY_ROUTE][S.ROUTE_KEY_LENS_SERVED_POSITIONS] = ["all"]
        m[C.LENS_SERVED_POSITIONS_KEY] = ["all"]

    S.build_all(tmp_path / "b", rows=rows, overrides={(GLM, BARE, PRIMARY): {"manifest_edit": on_bare}})
    g = analyze(tmp_path / "b", rows=rows)["gates"]["certification"]
    assert g["status"] == M.GATE_FAIL and "non-lens cell" in g["detail"]

    def absent(m):
        del m[C.MANIFEST_KEY_ROUTE][S.ROUTE_KEY_LENS_SERVED_POSITIONS]
        del m[C.LENS_SERVED_POSITIONS_KEY]

    S.build_all(tmp_path / "a", rows=rows, overrides={(GLM, OBJECT, PRIMARY): {"manifest_edit": absent}})
    rep = analyze(tmp_path / "a", rows=rows)
    g = rep["gates"]["certification"]
    assert g["status"] == M.GATE_PENDING and g["pending_stamps"] == [M.STAMP_ROUTE_SERVED_POSITIONS]
    assert rep["verdict"]["word"] is None and M.production_refusals(rep)
    waived = analyze(tmp_path / "a", rows=rows, allow_pending_gates=True)
    assert waived["verdict"]["word"] is not None and waived["verdict"]["gates_pending"] == ["certification"]

    def disagree(m):
        m[C.LENS_SERVED_POSITIONS_KEY] = ["last"]

    S.build_all(tmp_path / "d", rows=rows, overrides={(GLM, OBJECT, PRIMARY): {"manifest_edit": disagree}})
    g = analyze(tmp_path / "d", rows=rows)["gates"]["certification"]
    assert g["status"] == M.GATE_FAIL and "!= route.lens_served_positions" in g["detail"]


def test_a_declared_cell_a_candidate_fact_and_a_split_mode_mismatch_refuse(tmp_path):
    rows = primary_rows(GLM)

    def declared(m):
        m[C.MANIFEST_KEY_ROUTE][C.ROUTE_KEY_SERVE_DECLARATIONS] = {
            S.harness_id_of(OBJECT) + "@thinking": {"serve_out_of_support": ["thinking"]}}

    S.build_all(tmp_path / "decl", rows=rows, overrides={(GLM, OBJECT, PRIMARY): {"manifest_edit": declared}})
    g = analyze(tmp_path / "decl", rows=rows)["gates"]["certification"]
    assert g["status"] == M.GATE_FAIL and "lens_serve_declarations" in g["detail"]
    # every family present must be CERTIFIED: a candidate fact refuses by family name
    S.build_all(tmp_path / "cand", rows=rows)
    facts = {k: dict(v) for k, v in S.RENDER_FACTS.items()}
    facts["glm4v"]["evidence_status"] = "candidate"
    rep = analyze(tmp_path / "cand", rows=rows, render_facts=facts)
    g = rep["gates"]["certification"]
    assert g["status"] == M.GATE_FAIL and "glm4v" in g["detail"] and "candidate" in g["detail"]
    assert M.production_refusals(rep)
    # a family absent from the fact table refuses the leg by name
    with pytest.raises(M.AnalysisError, match="render fact"):
        analyze(tmp_path / "cand", rows=rows, render_facts={k: v for k, v in S.RENDER_FACTS.items() if k != "glm4v"})

    # the summary's think_split_mode must be the family fact's (the FAMILY decides the mode)
    def split(s):
        s["think_split_mode"] = C.THINK_SPLIT_IMPLICIT_OPEN

    S.build_all(tmp_path / "split", rows=rows, overrides={(GLM, BARE, PRIMARY): {"summary_edit": split}})
    g = analyze(tmp_path / "split", rows=rows)["gates"]["scoring_path"]
    assert g["status"] == M.GATE_FAIL and "think_split_mode" in g["detail"]

    def parser(s):
        s["reasoning_parser"] = "gemma4"

    S.build_all(tmp_path / "parser", rows=rows, overrides={(GLM, BARE, PRIMARY): {"summary_edit": parser}})
    assert "reasoning_parser" in analyze(tmp_path / "parser", rows=rows)["gates"]["scoring_path"]["detail"]


# --------------------------------------------------------------------------- #
# gates 4-6: at-cap, split anomalies, the scoring path                         #
# --------------------------------------------------------------------------- #


def test_cap_gate_is_advisory_at_the_ceiling_and_a_re_cost_below_it(tmp_path, monkeypatch):
    """At the ceiling (the run's cap) a 3 % cell is REPORTED — the gate passes with an advisory naming it and the
    verdict is not withheld on its account; below the ceiling the parent's re-cost refusal stands."""
    assert M.CAP == M.AT_CAP_CEILING == C.MATH500_MAX_TOKENS_DEFAULT
    rows = primary_rows(QWEN)
    S.build_all(tmp_path / "hi", rows=rows, overrides={(QWEN, OBJECT, PRIMARY): {"capped_scored": set(range(15))}})
    rep = analyze(tmp_path / "hi", rows=rows)
    g = rep["gates"]["cap"]
    assert g["status"] == M.GATE_PASS and g["binding"] is False and g["ceiling"] == M.CAP and g["detail"] is None
    leg = g["legs"][M.leg_key(QWEN, OBJECT, PRIMARY)]
    assert leg["n_case_c_answer_at_cap"] == 15 and leg["status"] == M.GATE_PASS and "3.0" in leg["advisory"] and "no rung above" in leg["advisory"]
    assert g["advisories"] == [leg["advisory"]] and "cap" not in rep["gates"]["failed"]
    assert not any("cap" in r for r in M.production_refusals(rep) if "at_cap" in r)
    # below the ceiling (a smaller cap than the rule's ceiling) the refusal is the parent's, verbatim
    monkeypatch.setattr(M, "AT_CAP_CEILING", M.CAP * 2)
    rep_b = analyze(tmp_path / "hi", rows=rows)
    gb = rep_b["gates"]["cap"]
    assert gb["status"] == M.GATE_FAIL and gb["binding"] is True and "re-cost" in gb["detail"] and "3.0" in gb["detail"]
    assert M.production_refusals(rep_b) and rep_b["verdict"]["word"] is None
    monkeypatch.setattr(M, "AT_CAP_CEILING", M.CAP)
    S.build_all(tmp_path / "ok", rows=rows, overrides={(QWEN, OBJECT, PRIMARY): {"capped_scored": set(range(10))}})
    ok = analyze(tmp_path / "ok", rows=rows)["gates"]["cap"]
    assert ok["status"] == M.GATE_PASS and ok["advisories"] == []


def test_split_anomaly_gate_per_family_mode(tmp_path):
    # (i) an unterminated block NOT at the cap
    rows = primary_rows(QWEN)
    S.build_all(tmp_path / "a", rows=rows, overrides={(QWEN, BARE, PRIMARY): {"unscored": {3}, "not_at_cap": {3}}})
    g = analyze(tmp_path / "a", rows=rows)["gates"]["split_anomalies"]
    assert g["status"] == M.GATE_FAIL and "n_unterminated_not_at_cap" in g["detail"]

    # (ii) a SCORED unterminated row (the Nemotron trap: chain-of-thought read as the answer)
    def scored_unterminated(h):
        h["items"][5]["think_block_unterminated"] = True
        h["n_think_block_unterminated"] = 1

    def summary_too(s):
        s["n_think_block_unterminated"] = 1

    rows = primary_rows(NEMO)
    S.build_all(tmp_path / "b", rows=rows,
                overrides={(NEMO, OBJECT, PRIMARY): {"header_edit": scored_unterminated, "summary_edit": summary_too}})
    rep = analyze(tmp_path / "b", rows=rows)
    g = rep["gates"]["split_anomalies"]
    assert g["status"] == M.GATE_FAIL and "scored" in g["detail"] and "unterminated" in g["detail"]
    assert M.production_refusals(rep)
    # (iii) an implicit-open family with a block-free item: every completion must carry the block
    S.build_all(tmp_path / "c", rows=rows, overrides={(NEMO, BARE, PRIMARY): {"no_think": {9}}})
    g = analyze(tmp_path / "c", rows=rows)["gates"]["split_anomalies"]
    assert g["status"] == M.GATE_FAIL and "n_think_block_present" in g["detail"]
    # ... while GLM (explicit_tags) reports its present RATE as a finding, not a failure
    rows = primary_rows(GLM)
    S.build_all(tmp_path / "d", rows=rows, overrides={(GLM, BARE, PRIMARY): {"no_think": set(range(50))}})
    rep = analyze(tmp_path / "d", rows=rows)
    assert rep["gates"]["split_anomalies"]["status"] == M.GATE_PASS
    assert rep["gates"]["split_anomalies"]["counts"][M.leg_key(GLM, BARE, PRIMARY)]["think_present_rate"] == 0.9
    assert any("think block present" in f for f in rep["findings"])
    # (iv) a stray open tag is reported and the rule re-run without it; the word does not move
    rows = primary_rows(QWEN)
    S.build_all(tmp_path / "e", rows=rows, overrides={(QWEN, OBJECT, PRIMARY): {"stray": {11, 12}}})
    g = analyze(tmp_path / "e", rows=rows)["gates"]["split_anomalies"]
    assert g["status"] == M.GATE_PASS and g["masked_out"][M.leg_key(QWEN, OBJECT, PRIMARY)] == [S.IDS[11], S.IDS[12]]
    assert g["word_moves"] is False


# --------------------------------------------------------------------------- #
# unterminated-not-at-cap draws sub-classed from the log; primary legs fail    #
# closed, sensitivity legs mask-and-report with a REAL tri-state word check    #
# --------------------------------------------------------------------------- #

LIFTED_RAW = ("Let me work this out. The system instruction says 'Your response should begin with {lit}...'. "
              "Odd, but the user wants ANSWER on its own line. So the value is 7.</think>\n\nWe compute directly.\n\nANSWER: 7")
LIFTED_TEXTS = ("Let me work this out. The system instruction says 'Your response should begin with \n\nWe compute directly.\n\nANSWER: 7",
                "...'. Odd, but the user wants ANSWER on its own line. So the value is 7.")
EOS_RAW = "Let me analyze the text provided by the system instruction again. The text says: `<|system|>` ... Then `"


def _split_fixture(tmp_path, *, rows, leg, lifted, eos, literals=("<thinking>", "<thinking>", "<think>"), **overrides):
    """A store whose ``leg`` (model, cell, mode) carries ``lifted`` + ``eos``
    unterminated-not-at-cap items, and — in the inspect venv — a real ``.eval``
    log in inspect's lifted shape for them: a lifted draw's raw body closes its
    block and answers while the sample's output carries the pre-literal thoughts
    and the answer as text and the removed span as a reasoning part; an EOS draw's
    raw body never closes (finish_reason stop). Returns whether the log exists."""
    anomalies = set(lifted) | set(eos)
    over = {leg: {"unscored": anomalies, "not_at_cap": anomalies, **overrides}}
    dirs = S.build_all(tmp_path, rows=rows, overrides=over)
    has_inspect = importlib.util.find_spec("inspect_ai") is not None
    if has_inspect:
        texts, raw = {}, {}
        for j, lit in zip(sorted(lifted), literals):
            texts[S.IDS[j]] = LIFTED_TEXTS
            raw[S.IDS[j]] = (LIFTED_RAW.format(lit=lit), "stop")
        for j in sorted(eos):
            texts[S.IDS[j]] = (EOS_RAW, None)
            raw[S.IDS[j]] = (EOS_RAW, "stop")
        S.write_inspect_log(dirs[M.leg_key(*leg)], name="pass", texts=texts, raw_bodies=raw)
    return has_inspect


def test_classify_unterminated_anomaly_is_the_lift_grammar_over_the_raw_body():
    lifted = M.classify_unterminated_anomaly(LIFTED_RAW.format(lit="<thinking>"), finish_reason="stop", completion=LIFTED_TEXTS[0])
    assert lifted["class"] == M.ANOMALY_LIFTED_CLOSE == "lifted_close" and lifted["tag_literals"] == ["<thinking>"]
    assert lifted["raw_has_close"] is True and lifted["completion_has_close"] is False and "`<thinking>`" in lifted["why"]
    assert lifted["close_in_code_span"] is False and lifted["answer_marker_after_close"] is True
    # the class is a reading of the text: the why names what was observed, never that the model "answered" or is excused
    assert "removed by inspect_ai's client-side lift" in lifted["why"] and "ANSWER: line follows" in lifted["why"]
    assert "answered" not in lifted["why"] and "model behaviour" not in lifted["why"]
    # every literal inspect_ai's `<think([^>]*)>` would pair, counted as written
    multi = M.classify_unterminated_anomaly("a <thinking_process> b <think> c</think> d\nANSWER: 4", finish_reason="stop", completion="a  d\nANSWER: 4")
    assert multi["tag_literals"] == ["<thinking_process>", "<think>"] and multi["class"] == "lifted_close"
    # the ambiguous cases are `other`, named: (i) the last raw close sits inside a fenced example, (ii) a
    # close with no answer marker after it (the model quoted a close and reasoned on to a mid-sentence stop)
    fenced = ("I recall the format:\n```\n<thinking>steps</think>\n```\nSo I keep reasoning about the sum and then `",)[0]
    r = M.classify_unterminated_anomaly(fenced, finish_reason="stop", completion="I recall the format:\n```\n\n```\nSo I keep reasoning about the sum and then `")
    assert r["class"] == "other" and r["close_in_code_span"] is True and "fenced / inline-code span" in r["why"]
    inline = "the template is `<thinking>x</think>` I think; continuing the derivation, 3 + 4 gives"
    r = M.classify_unterminated_anomaly(inline, finish_reason="stop", completion="the template is `` I think; continuing the derivation, 3 + 4 gives")
    assert r["class"] == "other" and r["close_in_code_span"] is True
    quoted = "It said 'begin with <thinking>' ... so I reason: 2 + 2 = 4, hence</think> wait, no, I meant to keep going and the"
    r = M.classify_unterminated_anomaly(quoted, finish_reason="stop", completion="It said 'begin with  wait, no, I meant to keep going and the")
    assert r["class"] == "other" and r["answer_marker_after_close"] is False and "no answer marker" in r["why"] and "pending stronger evidence" in r["why"]
    # a fence CLOSED before the genuine close does not shadow it
    ok = "format: ```<thinking>x</think>``` fine. Real reasoning here.</think>\nThe answer.\n\nANSWER: 9"
    r = M.classify_unterminated_anomaly(ok, finish_reason="stop", completion="format:  fine. Real reasoning here.</think>\nThe answer.\n\nANSWER: 9")
    assert r["class"] == "other" and "both" in r["why"]        # here the completion still has a close: not the lift's doing
    r = M.classify_unterminated_anomaly(ok, finish_reason="stop", completion="format:  fine. Real reasoning here.\nThe answer.\n\nANSWER: 9")
    assert r["class"] == "lifted_close" and r["close_in_code_span"] is False and r["answer_marker_after_close"] is True
    eos = M.classify_unterminated_anomaly(EOS_RAW, finish_reason="stop", completion=EOS_RAW)
    assert eos["class"] == M.ANOMALY_EOS_IN_THOUGHTS == "eos_in_thoughts" and eos["tag_literals"] == [] and eos["raw_has_close"] is False
    # named `other`: no raw body; a close on both sides; no close but not an EOS stop
    assert M.classify_unterminated_anomaly(None, finish_reason="stop", completion="x")["class"] == "other"
    both = M.classify_unterminated_anomaly("t</think>a", finish_reason="stop", completion="t</think>a")
    assert both["class"] == "other" and "both" in both["why"]
    assert M.classify_unterminated_anomaly("t", finish_reason="length", completion="t")["class"] == "other"
    kinds = M.anomaly_kinds({"a": {"think_block_unterminated": True, "at_cap": False, "scored": False, "stray_open_tag": True},
                             "b": {"think_block_unterminated": False, "at_cap": False, "scored": False, "stray_open_tag": False},
                             "c": {"think_block_unterminated": False, "at_cap": False, "scored": True, "stray_open_tag": True},
                             "d": {"think_block_unterminated": True, "at_cap": True, "scored": False, "stray_open_tag": False}})
    assert kinds == {"a": "unterminated_not_at_cap", "b": "unscored_other", "c": "stray_open_tag"}


def test_sensitivity_leg_anomalies_are_masked_and_reported_with_a_real_word_check(tmp_path):
    """(a) A SENSITIVITY leg (qwen general, object) with 3 lifted_close + 1
    eos_in_thoughts draws, its comparison usable and the verdict rule's word
    unmoved without them -> PASSED, the classes reported per leg (from the log in
    the inspect venv; `unavailable` in the figure venv), the items still
    unscored, `word_moves` False from a comparison that was actually re-run.
    The anomalies sit on items the PRIMARY object leg gets wrong, so the
    general-minus-primary delta is concordant on them and its word cannot move
    (the delta-driven movement is its own test below)."""
    rows = rows_for(QWEN, PRIMARY, GENERAL)
    leg = (QWEN, OBJECT, GENERAL)
    A = (270, 271, 272, 273)
    assert not set(A) & S.CORRECT_BY_CELL[S.CELL_OBJECT]
    has_inspect = _split_fixture(tmp_path, rows=rows, leg=leg, lifted=set(A[:3]), eos={A[3]})
    rep = analyze(tmp_path, rows=rows, read_eval_logs=True)
    key = M.leg_key(*leg)
    g = rep["gates"]["split_anomalies"]
    assert g["status"] == M.GATE_PASS and g["word_moves"] is False and g["pending"] == []
    assert g["masked_out"] == {key: [S.IDS[j] for j in A]}
    w = g["without"][f"{QWEN}/{GENERAL}"]
    assert w["skipped"] is None and w["word_filed"] is False and w["masked_out"] == g["masked_out"][key]
    # the RULE's word on a sensitivity comparison (which files none): object lower on both sides, prompted unresolved
    assert w["word_before"] == w["word_after"] == {M.CONTRAST_OBJECT: M.WORD_LOWER, M.CONTRAST_PROMPTED: M.WORD_NOT_RESOLVED}
    assert "4 anomalous items reported; the word does not move without them (2 comparison(s) re-run, mode deltas included)" == g["detail"]
    # the sensitivity-minus-primary delta consumes the leg too and is re-run under the same mask (P2-1)
    wd = g["without"][f"{QWEN}/{GENERAL}-minus-{PRIMARY}"]
    assert wd["kind"] == "mode_delta" and wd["skipped"] is None and wd["word_filed"] is False and wd["masked_out"] == g["masked_out"][key]
    assert set(wd["word_before"]) == {M.mode_delta_contrast(k, GENERAL) for k in M.KINDS} and wd["word_before"] == wd["word_after"]
    assert set(g["without"]) == {f"{QWEN}/{GENERAL}", f"{QWEN}/{GENERAL}-minus-{PRIMARY}"} and w["kind"] == "comparison"
    c = g["counts"][key]
    assert (c["primary"], c["n_unterminated_not_at_cap"], c["anomaly_kinds"]) == \
        (False, 4, {S.IDS[j]: "unterminated_not_at_cap" for j in A})
    u = rep["legs"][key]["unscored"]
    assert (u["n_unscored"], u["n_unterminated_not_at_cap"], u["n_unterminated_at_cap"], u["n_unscored_other"]) == (4, 4, 0, 0)
    assert u["unterminated_not_at_cap_ids"] == g["masked_out"][key]
    assert next(r for r in rep["controls"] if r["leg"] == key)["unscored"] == u
    finding = next(f for f in rep["findings"] if f.startswith(key) and "SENSITIVITY leg" in f)
    if has_inspect:
        assert (c["class_source"], c["n_lifted_close"], c["n_eos_in_thoughts"], c["n_class_other"], c["n_class_unavailable"]) == \
            ("eval_log", 3, 1, 0, 0)
        assert c["tag_literals"] == {"<thinking>": 2, "<think>": 1}
        assert {i: x["class"] for i, x in c["classes"].items()} == {**{S.IDS[j]: "lifted_close" for j in A[:3]}, S.IDS[A[3]]: "eos_in_thoughts"}
        assert u["by_class"] == {"lifted_close": 3, "eos_in_thoughts": 1, "other": 0, "unavailable": 0} and u["class_source"] == "eval_log"
        assert "3 lifted_close" in finding and "`<thinking>` x2, `<think>` x1" in finding and "1 eos_in_thoughts" in finding
        assert rep["legs"][key]["eval_check"]["anomaly_classes"][S.IDS[A[0]]]["class"] == "lifted_close"
    else:
        assert (c["class_source"], c["n_class_unavailable"], c["n_lifted_close"], c["classes"], c["tag_literals"]) == ("unavailable", 4, 0, {}, {})
        assert u["by_class"] == {"lifted_close": 0, "eos_in_thoughts": 0, "other": 0, "unavailable": 4}
        assert "classes unavailable in this run" in finding
    assert "the items stay unscored under their class" in finding and M.production_refusals(rep) == []


def test_a_masked_sensitivity_leg_whose_word_moves_fails_the_gate_by_comparison_name(tmp_path):
    """(b) The knife edge: the object leg is correct on 8 items the bare leg
    misses and UNSCORED on 4 the bare leg gets — under score-as-is those 4 count
    against the object cell and the CI covers zero (the rule's word: not
    resolved); masked from both legs the same 8-item lead resolves "higher". The
    word moves -> FAILED, the comparison named."""
    rows = rows_for(QWEN, PRIMARY, GENERAL)
    leg = (QWEN, OBJECT, GENERAL)
    bare_correct = S.default_kwargs(QWEN, S.CELL_BARE, GENERAL)["correct"]
    anomalies = {300, 301, 302, 303}
    assert anomalies <= bare_correct
    object_correct = (bare_correct - anomalies) | set(range(310, 318))
    assert not (set(range(310, 318)) & bare_correct)
    _split_fixture(tmp_path, rows=rows, leg=leg, lifted={300, 301, 302}, eos={303}, correct=object_correct)
    rep = analyze(tmp_path, rows=rows, read_eval_logs=True)
    g = rep["gates"]["split_anomalies"]
    assert g["status"] == M.GATE_FAIL and g["word_moves"] is True
    w = g["without"][f"{QWEN}/{GENERAL}"]
    assert w["word_before"][M.CONTRAST_OBJECT] == M.WORD_NOT_RESOLVED and w["word_after"][M.CONTRAST_OBJECT] == M.WORD_HIGHER
    assert f"{QWEN}/{GENERAL} (comparison): 4 anomalous items change the word" in g["detail"]
    assert "split_anomalies" in rep["gates"]["failed"] and any("gate(s) FAILED" in r for r in M.production_refusals(rep))
    # the anomalies are items the primary object leg gets wrong too, so the delta is concordant on them: unmoved, not named
    d = g["without"][f"{QWEN}/{GENERAL}-minus-{PRIMARY}"]
    assert d["word_before"] == d["word_after"] and "(mode_delta)" not in g["detail"]


def test_a_mode_delta_whose_word_alone_moves_fails_the_gate_naming_the_delta(tmp_path):
    """The three-cell general word stands ("lower" before and
    after) while the object general-minus-primary DELTA's rule moves — under
    score-as-is the 4 anomalies (primary-object correct, general-object
    unscored) hold its CI over zero; masked from both legs the 9-item lead
    resolves "higher". The masking pass consumes the delta too -> FAILED, the
    delta named; the comparison's own entry is unmoved."""
    rows = rows_for(QWEN, PRIMARY, GENERAL)
    leg = (QWEN, OBJECT, GENERAL)
    primary_object = S.CORRECT_BY_CELL[S.CELL_OBJECT]                 # {0..259}
    anomalies = {250, 251, 252, 253}
    assert anomalies <= primary_object
    general_object = (primary_object - anomalies) | set(range(260, 269))   # 9 general-only items vs the primary row
    _split_fixture(tmp_path, rows=rows, leg=leg, lifted={250, 251, 252}, eos={253}, correct=general_object)
    rep = analyze(tmp_path, rows=rows, read_eval_logs=True)
    g = rep["gates"]["split_anomalies"]
    comp, delta = g["without"][f"{QWEN}/{GENERAL}"], g["without"][f"{QWEN}/{GENERAL}-minus-{PRIMARY}"]
    assert comp["word_before"] == comp["word_after"] and comp["word_before"][M.CONTRAST_OBJECT] == M.WORD_LOWER
    c_obj = M.mode_delta_contrast(M.KIND_OBJECT, GENERAL)
    assert delta["kind"] == "mode_delta" and delta["word_before"][c_obj] == M.WORD_NOT_RESOLVED and delta["word_after"][c_obj] == M.WORD_HIGHER
    assert {k: v for k, v in delta["word_before"].items() if k != c_obj} == {k: v for k, v in delta["word_after"].items() if k != c_obj}
    assert g["status"] == M.GATE_FAIL and g["word_moves"] is True
    assert f"{QWEN}/{GENERAL}-minus-{PRIMARY} (mode_delta): 4 anomalous items change the word" in g["detail"]
    assert f"{QWEN}/{GENERAL} (comparison)" not in g["detail"]


def test_d48_a_primary_leg_eos_in_thoughts_draw_is_masked_and_reported_when_no_word_moves(tmp_path):
    """ONE `eos_in_thoughts` draw on a PRIMARY leg (nemotron object) whose
    consuming comparison, re-run without it, moves no verdict word -> the gate PASSES, the draw is masked from the FILED
    contrast (n_items 499; the WITH reading kept beside as `estimands_with_anomalies`), the item (id, tokens, finish
    reason) and `n_masked` ride the gate, the comparison and the tables' notes; the run WITHOUT eval logs admits it only
    from the matched report's classes (`class_source: matched_report`) and fails closed without them."""
    pytest.importorskip("inspect_ai")
    rows = primary_rows(NEMO)
    leg = (NEMO, OBJECT, PRIMARY)
    j = 270                                           # an item the object leg gets wrong: masking it cannot move a word
    assert j not in S.CORRECT_BY_CELL[S.CELL_OBJECT]
    _split_fixture(tmp_path, rows=rows, leg=leg, lifted=set(), eos={j})
    rep = analyze(tmp_path, rows=rows, read_eval_logs=True)
    key = M.leg_key(*leg)
    g = rep["gates"]["split_anomalies"]
    assert g["status"] == M.GATE_PASS and g["word_moves"] is False and M.production_refusals(rep) == []
    pm = g["primary_masked"][key]
    assert pm["n_masked"] == 1 and pm["class_source"] == "eval_log" and pm["records_rerun"] == [f"{NEMO}/{PRIMARY}"]
    assert pm["items"] == [{"item_id": S.IDS[j], "class": "eos_in_thoughts", "completion_tokens": pm["items"][0]["completion_tokens"],
                            "finish_reason": "stop", "scored": False, "at_cap": False}] and pm["items"][0]["completion_tokens"] < M.CAP
    assert g["n_masked_by_leg"] == {key: 1} and "1 on primary leg(s) masked from the filed contrast(s) under D48" in g["detail"]
    assert g["counts"][key]["primary"] is True and g["counts"][key]["n_eos_in_thoughts"] == 1
    c = comparison(rep, NEMO)
    assert c["n_masked"] == 1 and c["masked_items"] == {M.KIND_OBJECT: [S.IDS[j]]}
    # review H1: the mask is SCOPED to the contrast that reads the leg — the record's population stays 500, the object − bare
    # contrast alone is filed over 499 (its own two-leg estimand, the pair's bare over 499 kept in `masked_contrasts`), the
    # object leg's own accuracy is over its 499 items, and the prompted − bare contrast, the bare and prompted legs and
    # their transitions are BYTE-EQUAL to the reading with the item (`estimands_with_anomalies`)
    for b in M.BRANCHES:
        e, e0 = c["estimands"][b], c["estimands_with_anomalies"][b]
        assert e["n_items"] == S.N == e0["n_items"] and "n_items_by_contrast" not in e0
        assert e["n_items_by_contrast"] == {M.CONTRAST_OBJECT: S.N - 1, M.CONTRAST_PROMPTED: S.N}
        assert e["stats"][M.CONTRAST_PROMPTED] == e0["stats"][M.CONTRAST_PROMPTED] and e["complete_case"][M.CONTRAST_PROMPTED] == e0["complete_case"][M.CONTRAST_PROMPTED]
        assert e["acc_by_leg"]["bare"] == e0["acc_by_leg"]["bare"] and e["acc_by_leg"]["prompted"] == e0["acc_by_leg"]["prompted"]
        assert e["stats"][M.CONTRAST_OBJECT] != e0["stats"][M.CONTRAST_OBJECT]
        assert e["acc_by_leg"]["object"]["n_scored"] + e["acc_by_leg"]["object"]["n_censored"] == S.N - 1
        assert e["acc_by_leg"]["bare"]["n_scored"] + e["acc_by_leg"]["bare"]["n_censored"] == S.N
    mc = c["masked_contrasts"]
    assert set(mc) == {M.CONTRAST_OBJECT} and mc[M.CONTRAST_OBJECT]["n_items"] == S.N - 1 and mc[M.CONTRAST_OBJECT]["masked_items"] == [S.IDS[j]]
    assert mc[M.CONTRAST_OBJECT]["legs"] == {"object": key, "bare": M.leg_key(NEMO, BARE, PRIMARY)}
    bare_in = mc[M.CONTRAST_OBJECT]["acc_by_leg"][M.BRANCH_PRIMARY]["bare"]
    assert bare_in["n_scored"] + bare_in["n_censored"] == S.N - 1 and bare_in["n_correct"] == c["estimands"][M.BRANCH_PRIMARY]["acc_by_leg"]["bare"]["n_correct"] - 1
    assert c["mask_rule"] == M.MASK_SCOPE_RULE
    w = g["without"][f"{NEMO}/{PRIMARY}"]
    assert w["word_before"] == w["word_after"] == {k: v["word"] for k, v in c["verdicts"].items()}
    assert rep["masked_primary"] == {"by_leg": {key: [S.IDS[j]]}, "n_masked": 1, "by_contrast": {f"{NEMO}/{PRIMARY}": [M.CONTRAST_OBJECT]},
                                     "rule": M.MASK_SCOPE_RULE}
    assert any("masked from every record that consumes the leg" in f and f.startswith(key) for f in rep["findings"])
    notes = M.d48_notes(rep)
    assert notes["masked_primary"] == {"by_leg": {key: {"n_masked": 1, "items": [S.IDS[j]]}}, "by_arm": {"object": 1}, "n_masked": 1,
                                       "by_contrast": {f"{NEMO}/{PRIMARY}": [M.CONTRAST_OBJECT]}, "scope_rule": M.MASK_SCOPE_RULE}
    assert "masked (D48): 1 eos_in_thoughts draw(s)" in M._d48_footnote(rep)
    # the run WITHOUT the log: the classes come from the report it must match, named as such; the numbers are the same
    matched = M.matched_anomaly_classes(rep)
    assert set(matched) == {key} and matched[key][S.IDS[j]]["class"] == "eos_in_thoughts"
    rep_f = analyze(tmp_path, rows=rows, read_eval_logs=False, anomaly_classes=matched)
    gf = rep_f["gates"]["split_anomalies"]
    assert gf["status"] == M.GATE_PASS and gf["primary_masked"][key]["class_source"] == "matched_report"
    assert rep_f["legs"][key]["unscored"]["class_source"] == "matched_report" and rep_f["legs"][key]["unscored"]["by_class"]["eos_in_thoughts"] == 1
    assert M.numeric_leaf_diff(rep, rep_f)["equal"] is True and M.production_refusals(rep_f) == []
    # ... and WITHOUT them it fails closed (an unclassified draw is never admitted)
    rep_u = analyze(tmp_path, rows=rows, read_eval_logs=False)
    gu = rep_u["gates"]["split_anomalies"]
    assert gu["status"] == M.GATE_FAIL and "classes unavailable" in gu["detail"] and "fail-closed" in gu["detail"] and gu["primary_masked"] == {}
    assert comparison(rep_u, NEMO)["estimands"][M.BRANCH_PRIMARY]["n_items"] == S.N and "n_masked" not in comparison(rep_u, NEMO)


def test_d48_a_primary_leg_eos_in_thoughts_draw_that_moves_the_word_or_a_lifted_close_keeps_failing_closed(tmp_path):
    """The two branches that stay closed: (a) the knife edge of the sensitivity test, on the PRIMARY leg — four
    eos_in_thoughts draws the bare leg gets right hold the score-as-is CI over zero; masked, the 8-item lead resolves
    "higher" -> the word moves -> FAILED, naming the comparison, nothing masked from the filed contrast; (b) ONE
    `lifted_close` draw on a primary leg -> FAILED on the class, whatever the word does."""
    pytest.importorskip("inspect_ai")
    rows = primary_rows(QWEN)
    leg = (QWEN, OBJECT, PRIMARY)
    bare_correct = S.default_kwargs(QWEN, S.CELL_BARE, PRIMARY)["correct"]          # the primary bare row: items 0..299
    anomalies = {290, 291, 292, 293}
    assert anomalies <= bare_correct and not (set(range(310, 318)) & bare_correct)
    _split_fixture(tmp_path / "moves", rows=rows, leg=leg, lifted=set(), eos=anomalies, correct=(bare_correct - anomalies) | set(range(310, 318)))
    rep = analyze(tmp_path / "moves", rows=rows, read_eval_logs=True)
    g = rep["gates"]["split_anomalies"]
    assert g["status"] == M.GATE_FAIL and g["word_moves"] is True and g["primary_masked"] == {}
    assert f"{M.leg_key(*leg)}: n_unterminated_not_at_cap 4 != 0 on a PRIMARY leg" in g["detail"] and "the word moves on" in g["detail"]
    assert "D48 admits an eos_in_thoughts draw only when no verdict word moves" in g["detail"]
    c = comparison(rep, QWEN)
    assert c["estimands"][M.BRANCH_PRIMARY]["n_items"] == S.N and "estimands_with_anomalies" not in c and "masked_primary" not in rep
    assert M.production_refusals(rep)
    _split_fixture(tmp_path / "lifted", rows=rows, leg=leg, lifted={270}, eos=set())
    rep = analyze(tmp_path / "lifted", rows=rows, read_eval_logs=True)
    g = rep["gates"]["split_anomalies"]
    assert g["status"] == M.GATE_FAIL and g["primary_masked"] == {} and "1 lifted_close" in g["detail"] and "fail-closed on either class (D180)" in g["detail"]


def test_the_installed_parser_a_quoted_fenced_example_before_a_mid_sentence_stop_is_not_lifted_close():
    """Through inspect_ai's own lift: a reasoning-only body quotes a fenced
    ``<thinking>…</think>`` example, reasons on, and stops mid-sentence. The
    lift removes the quoted block, the producer's split reads the remainder as
    unterminated (correctly), and the classifier must NOT file `lifted_close` —
    the removed close sat inside a fence and no answer follows it. A
    real-shaped body (a quoted literal, a genuine close, an ANSWER line)
    through the same parser IS `lifted_close`."""
    pytest.importorskip("inspect_ai")
    from inspect_ai.model._reasoning import parse_content_with_reasoning

    from sisteer.capeval.tiers.think_split import split_think
    sol = ("Let me recall the expected shape:\n```\n<thinking>\nreason here\n</think>\n```\nOK. Now the actual problem: the sum "
           "of the roots is 3 and the product is 2, so the quadratic is x^2 - 3x + 2 and I should check whether `")
    lifted_text, capsule = parse_content_with_reasoning(sol)
    assert capsule is not None and "</think>" not in lifted_text and "reason here" in capsule.reasoning
    split = split_think(lifted_text, mode="implicit_open")
    assert split.unterminated is True                                # the producer's record: unterminated, not at cap
    rec = M.classify_unterminated_anomaly(sol, finish_reason="stop", completion=lifted_text)
    assert rec["class"] == "other" and rec["class"] != M.ANOMALY_LIFTED_CLOSE
    assert rec["close_in_code_span"] is True and rec["raw_has_close"] is True and rec["completion_has_close"] is False
    # the same shape without the fence: still not lifted_close — nothing answer-like follows the quoted close
    unfenced = sol.replace("```\n", "")
    lifted_text2, _ = parse_content_with_reasoning(unfenced)
    rec2 = M.classify_unterminated_anomaly(unfenced, finish_reason="stop", completion=lifted_text2)
    assert rec2["class"] == "other" and rec2["answer_marker_after_close"] is False and "no answer marker" in rec2["why"]
    # the positive control, the real draws' shape
    real = LIFTED_RAW.format(lit="<thinking>")
    lifted_real, capsule_real = parse_content_with_reasoning(real)
    assert capsule_real is not None and "</think>" not in lifted_real and split_think(lifted_real, mode="implicit_open").unterminated
    rec3 = M.classify_unterminated_anomaly(real, finish_reason="stop", completion=lifted_real)
    assert rec3["class"] == "lifted_close" and rec3["answer_marker_after_close"] is True and rec3["close_in_code_span"] is False


def test_a_masked_leg_whose_comparison_did_not_run_leaves_the_gate_pending_never_passed(tmp_path):
    """(c) The general bare and prompted legs are absent, so the general
    comparison is unusable: nothing was re-run -> `word_moves` "unusable",
    the skipped comparison recorded by name, PENDING (not waivable), never PASSED."""
    rows = rows_for(QWEN, PRIMARY, GENERAL)
    leg = (QWEN, OBJECT, GENERAL)
    built = [r for r in rows if r["sampling_mode"] == PRIMARY or r["cell"] == OBJECT]
    _split_fixture(tmp_path, rows=built, leg=leg, lifted={21, 22, 23}, eos={24})
    rep = analyze(tmp_path, rows=rows, read_eval_logs=True)
    g = rep["gates"]["split_anomalies"]
    assert g["status"] == M.GATE_PENDING and g["word_moves"] == M.WORD_MOVES_UNUSABLE == "unusable"
    w = g["without"][f"{QWEN}/{GENERAL}"]
    assert w["skipped"] and "no out-root" in w["skipped"] and w["word_before"] is None and w["word_after"] is None
    assert len(g["pending"]) == 2 and g["pending"][0].startswith(f"{QWEN}/{GENERAL}: 4 anomalous item(s) masked but the comparison did not run")
    assert g["pending"][1].startswith(f"{QWEN}/{GENERAL}-minus-{PRIMARY}: 4 anomalous item(s) masked but the mode_delta did not run")
    assert g["detail"] == "; ".join(g["pending"])
    assert g["without"][f"{QWEN}/{GENERAL}-minus-{PRIMARY}"]["skipped"] and g["without"][f"{QWEN}/{GENERAL}-minus-{PRIMARY}"]["kind"] == "mode_delta"
    assert "split_anomalies" in rep["gates"]["pending"] and "split_anomalies" not in rep["gates"]["failed"]
    assert any(it["gate"] == "split_anomalies" and it["waivable"] is False for it in rep["gates"]["pending_unwaivable"])
    assert any("PENDING and not waivable" in r and "split_anomalies" in r for r in M.production_refusals(rep))
    # the counts and the sensitivity finding are still filed for the masked leg
    assert g["counts"][M.leg_key(*leg)]["n_unterminated_not_at_cap"] == 4
    assert any(f.startswith(M.leg_key(*leg)) and "SENSITIVITY leg" in f for f in rep["findings"])


def test_a_primary_leg_with_one_lifted_close_draw_fails_closed(tmp_path):
    """(d) One lifted_close draw on a PRIMARY leg -> FAILED, whatever the
    class: the primary rule is fail-closed on either sub-class."""
    rows = primary_rows(QWEN)
    leg = (QWEN, OBJECT, PRIMARY)
    has_inspect = _split_fixture(tmp_path, rows=rows, leg=leg, lifted={7}, eos=set())
    rep = analyze(tmp_path, rows=rows, read_eval_logs=True)
    g = rep["gates"]["split_anomalies"]
    assert g["status"] == M.GATE_FAIL and "split_anomalies" in rep["gates"]["failed"]
    key = M.leg_key(*leg)
    assert f"{key}: n_unterminated_not_at_cap 1 != 0 on a PRIMARY leg" in g["detail"] and "fail-closed on either class" in g["detail"]
    c = g["counts"][key]
    if has_inspect:
        assert (c["n_lifted_close"], c["n_eos_in_thoughts"]) == (1, 0) and "1 lifted_close" in g["detail"]
    else:
        assert c["n_class_unavailable"] == 1 and "classes unavailable" in g["detail"]
    # the word check still ran on the primary comparison (a real False, not vacuity) and the draw stays unscored
    assert g["word_moves"] is False and g["without"][f"{QWEN}/{PRIMARY}"]["skipped"] is None
    assert rep["legs"][key]["unscored"]["n_unterminated_not_at_cap"] == 1


def test_the_class_leaves_are_the_only_declared_difference_between_run_e_and_run_f(tmp_path):
    """(e) The class leaves are E-only by design and declared exactly: E and F
    agree on every scored numeric leaf (the anomaly COUNT, the masked ids, the
    re-run words included); the raw comparison's moved leaves are all covered by
    the declaration and nothing else is."""
    rows = rows_for(QWEN, PRIMARY, GENERAL)
    leg = (QWEN, OBJECT, GENERAL)
    has_inspect = _split_fixture(tmp_path, rows=rows, leg=leg, lifted={270, 271, 272}, eos={273})
    e, f = analyze(tmp_path, rows=rows, read_eval_logs=True), analyze(tmp_path, rows=rows, read_eval_logs=False)
    key = M.leg_key(*leg)
    for rep in (e, f):
        assert rep["gates"]["split_anomalies"]["status"] == M.GATE_PASS and rep["legs"][key]["unscored"]["n_unterminated_not_at_cap"] == 4
    assert f["gates"]["split_anomalies"]["counts"][key]["class_source"] == "unavailable"
    d = M.numeric_leaf_diff(e, f)
    assert d["equal"] and d["shared_numeric"] > 1000, d
    raw = M.T.numeric_leaf_diff(e, f)
    moved = [p_ for p_, _a, _b in raw["differing"]] + [p_ for p_, _a, _b in raw["type_changed"]] + raw["missing"] + raw["added"]
    if has_inspect:
        assert moved and all(M._anomaly_class_leaf_skipped(p_) for p_ in moved), moved
        assert {p_[-1] for p_ in moved if p_[0] == "gates"} <= {"n_lifted_close", "n_eos_in_thoughts", "n_class_other", "n_class_unavailable",
                                                                "<thinking>", "<think>"}
    else:
        assert moved == []
    # the declaration, exactly
    assert M.REPORT_DIFF_SKIP_ANOMALY_CLASS_FIELDS == ("n_lifted_close", "n_eos_in_thoughts", "n_class_other", "n_class_unavailable",
                                                        "classes", "tag_literals", "class_source")
    assert M.REPORT_DIFF_SKIP_UNSCORED_FIELDS == ("by_class", "class_source")
    assert M._split_leaf_skipped(("gates", "split_anomalies", "counts", key, "n_lifted_close"))
    assert M._split_leaf_skipped(("gates", "split_anomalies", "counts", key, "tag_literals", "<thinking>"))
    assert M._split_leaf_skipped(("legs", key, "unscored", "by_class", "lifted_close"))
    assert M._split_leaf_skipped(("controls", 3, "unscored", "class_source"))
    assert not M._split_leaf_skipped(("gates", "split_anomalies", "counts", key, "n_unterminated_not_at_cap"))
    assert not M._split_leaf_skipped(("legs", key, "unscored", "n_unterminated_not_at_cap"))
    assert not M._split_leaf_skipped(("legs", key, "unscored", "unterminated_not_at_cap_ids"))
    assert not M._split_leaf_skipped(("gates", "split_anomalies", "masked_out", key))
    assert not M._split_leaf_skipped(("truncation", "cells", 0, "n_lifted_close"))     # a same-named leaf elsewhere is not declared
    # the two-venv shape test's negative pins still hold with the new predicate in front
    assert not M._split_leaf_skipped(("legs", key, "at_cap", "n_at_cap"))


def test_scoring_path_gate(tmp_path):
    rows = primary_rows(GEMMA)

    def policy(s):
        s["unterminated_policy"] = C.UNTERMINATED_INCORRECT

    S.build_all(tmp_path / "a", rows=rows, overrides={(GEMMA, BARE, PRIMARY): {"summary_edit": policy}})
    g = analyze(tmp_path / "a", rows=rows)["gates"]["scoring_path"]
    assert g["status"] == M.GATE_FAIL and "unterminated_policy" in g["detail"]

    def grader(s):
        s["grader_max_tokens"] = 4096

    S.build_all(tmp_path / "b", rows=rows, overrides={(GEMMA, BARE, PRIMARY): {"summary_edit": grader}})
    g = analyze(tmp_path / "b", rows=rows)["gates"]["scoring_path"]
    assert g["status"] == M.GATE_FAIL and "grader_max_tokens" in g["detail"]
    # gemma: an unscored item is a reasoning-only cap-bound output, counted as case (b) with the channel flag
    S.build_all(tmp_path / "c", rows=rows, overrides={(GEMMA, OBJECT, PRIMARY): {"unscored": S.LAST4}})
    rep = analyze(tmp_path / "c", rows=rows)
    ctl = next(r for r in rep["controls"] if r["leg"] == M.leg_key(GEMMA, OBJECT, PRIMARY))
    assert ctl["summary"]["n_content_empty_with_reasoning"] == 4
    assert rep["gates"]["cap"]["legs"][M.leg_key(GEMMA, OBJECT, PRIMARY)]["n_case_b_unterminated_at_cap"] == 4
    assert rep["gates"]["scoring_path"]["status"] == M.GATE_PASS


# --------------------------------------------------------------------------- #
# the report's tables: controls, provenance, transitions                        #
# --------------------------------------------------------------------------- #


def test_controls_provenance_and_transitions(full_report):
    rep = full_report
    ctl = {r["leg"]: r for r in rep["controls"]}
    assert len(ctl) == 24
    row = ctl[M.leg_key(QWEN, OBJECT, PRIMARY)]
    assert row["summary"]["acc"] == pytest.approx(0.52) and row["summary"]["n_at_cap"] == 0
    assert row["summary"]["mean_completion_tokens"] > 0 and row["summary"]["p99_completion_tokens"] > 0
    assert row["summary"]["n_think_block_present"] == 500 and row["summary"]["n_think_block_unterminated"] == 0
    assert row["ledger"]["decode_tok_s_mean"] == pytest.approx(S.DECODE_TOK_S, rel=1e-3)
    assert row["ledger"]["decode_tok_s_aggregate"] == pytest.approx(S.DECODE_TOK_S, rel=1e-3)
    assert row["ledger"]["n_calls"] == 500 and row["ledger"]["error_code"] == {}
    # the at-cap readout and the no-reasoning readout ride the controls row, pinned key for key
    assert row["at_cap"] == {"rate": 0.0, "share": 0.0, "n_at_cap": 0, "n_with_stop_reason": 500, "line_1pct": 0.01,
                             "line_1pct_exceeded": False, "surfacing_threshold": 0.03, "at_or_above_surfacing_threshold": False,
                             "n_case_b_unterminated_at_cap": 0, "n_case_c_answer_at_cap": 0,
                             "split_source": "store_rows", "split_source_reason": "the .eval log was not read; the store rows' split stands",
                             "n_loop": 0, "n_truncated_reasoning": 0, "n_text_unavailable": 0,
                             "store_rows_split": {"text_source": "rows.jsonl", "n_at_cap": 0, "n_classified": 0, "n_loop": 0,
                                                  "n_truncated_reasoning": 0, "n_text_unavailable": 0, "tail_classes": {}}}
    assert row["thinking"] == {"split_mode": "implicit_open", "n_on_rows": 500, "n_finished": 500, "n_no_reasoning_emitted": 0,
                               "share_no_reasoning": 0.0, "n_no_reasoning_at_cap": 0, "by_construction": True,
                               "n_present_by_record": 500, "n_present_by_channel": 0,
                               "rule": M.NO_REASONING_RULE["implicit_open"]}
    prov = rep["provenance"]
    assert prov["store_pin"] == S.SHA7 and prov["analysis_code_sha"] == S.FULL_SHA
    cell = prov["cells"][M.leg_key(QWEN, OBJECT, PRIMARY)]
    assert cell["producer_tree_sha"] == S.FULL_SHA and cell["object_artifact_sha256"] == S.object_pin(QWEN)
    assert cell["tuple_name"] == S.tuple_name(QWEN, PRIMARY) == "math500_thinking_on_full22_qwen35_4b"
    assert cell["job_id"] == S.matrix_row(QWEN, OBJECT, PRIMARY)["job_id"] == "m5t-qwen35_4b-object"
    assert cell["harness_config_hash"] and cell["model_revision"] == S.model_pin(QWEN)[1]
    assert ctl[M.leg_key(QWEN, BARE, PRIMARY)]["provenance"]["object_artifact_sha256"] is None
    tr = comparison(rep, QWEN)["transitions"]
    assert tr[M.CONTRAST_OBJECT]["counts"] == {"C->C": 260, "C->I": 40, "I->C": 0, "I->I": 200, "->U": 0}
    assert tr[M.CONTRAST_PROMPTED]["counts"]["C->I"] == 20 and tr[M.CONTRAST_PROMPTED]["counts"]["I->C"] == 18
    mc = comparison(rep, QWEN)["mcnemar"][M.CONTRAST_OBJECT]
    assert mc[M.BRANCH_PRIMARY]["C->I"] == 40 and mc[M.BRANCH_PRIMARY]["one_sided_discordance"]
    assert mc["registered_contrast"] == M.CONTRAST_OBJECT


# --------------------------------------------------------------------------- #
# the cells-matrix loader                                                      #
# --------------------------------------------------------------------------- #


def test_cells_loader_json_python_and_refusals(tmp_path):
    p = S.write_cells_json(tmp_path / "cells.json")
    rows = M.load_cells(p)
    assert len(rows) == 24 and sum(r["fitted"] for r in rows) == 8
    assert sum(r["sampling_mode"] == BATTERY for r in rows) == 9 and sum(r["sampling_mode"] == GENERAL for r in rows) == 3
    py = tmp_path / "cells.py"
    py.write_text("from collections import namedtuple\nR = namedtuple('R', " + repr(list(rows[0])) +
                  ")\nM5T_CELLS = [R(**r) for r in " + repr(rows) + "]\n")
    assert M.load_cells(py) == rows
    bad = [dict(rows[0])]
    del bad[0]["expected_stamps"]
    with pytest.raises(M.AnalysisError, match="expected_stamps"):
        M.load_cells(bad)
    off = [{**rows[0], "out_root": "controls/somewhere/else"}]
    with pytest.raises(M.AnalysisError, match="out_root"):
        M.load_cells(off)
    dup = [rows[0], rows[0]]
    with pytest.raises(M.AnalysisError, match="twice"):
        M.load_cells(dup)
    # an absolute cluster path is accepted when its tail is the analysis' own layout
    absolute = [{**rows[0], "out_root": "/shared/USER/exp/m5t/" + rows[0]["out_root"]}]
    assert M.load_cells(absolute)[0]["out_root"] == rows[0]["out_root"]


def test_family_and_pins_resolve_from_the_matrix_and_the_checked_in_files():
    row = next(r for r in S.cells_matrix() if r["model_key"] == GLM)
    assert M.family_of(row) == "glm4v"
    assert M.family_of({**row, "family": None, "model_key": QWEN}) == "qwen3_5"    # contracts' registry row
    with pytest.raises(M.AnalysisError, match="family"):
        M.family_of({**row, "family": None})                                          # glm is not registered here
    pins = M.load_object_pins(REPO / "configs" / "sdfbase_full22_pins.json")
    assert pins[QWEN].startswith("e24b4aaa") and pins[GEMMA].startswith("be93b469")
    assert M.model_pin(REPO, NEMO) == ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f")


# --------------------------------------------------------------------------- #
# main(): the fail-closed boundary, exit 3 / exit 4, the two-venv shape          #
# --------------------------------------------------------------------------- #


def _main_args(gen_root, tmp_path, *extra, rows=None):
    cells = S.write_cells_json(tmp_path / "in" / "cells.json", primary_rows(GEMMA) if rows is None else rows)
    rows = S.write_sampling_rows(tmp_path / "in" / "rows.json")
    facts = S.write_render_facts(tmp_path / "in" / "facts.json")
    ids = tmp_path / "in" / "ids.json"
    ids.write_text(json.dumps(S.IDS))
    return ["--gen-root", str(gen_root), "--sha", S.SHA7, "--cells", str(cells), "--sampling-rows", str(rows),
            "--render-facts", str(facts), "--math500-ids", str(ids), "--analysis-sha", S.FULL_SHA,
            "--bootstrap", str(B_FAST), "--wandb-mode", "disabled", "--no-mirror", "--no-eval-logs", *extra]


def test_main_writes_a_production_report_without_importing_wandb(tmp_path, capsys):
    S.build_all(tmp_path / "gen", rows=primary_rows(GEMMA))
    sys.modules.pop("wandb", None)
    out = tmp_path / "report.json"
    rc = M.main(_main_args(tmp_path / "gen", tmp_path, "--no-figs", "--out", str(out)))
    assert rc == 0 and "wandb" not in sys.modules
    rep = json.loads(out.read_text())
    assert rep["wandb_url"] is None and rep["verdict"]["word"] is not None
    assert M.production_refusals(rep) == []
    assert "figures" not in rep
    # --gate-report on it passes; a perturbed copy differs (exit 4); the same report matches
    assert M.main(["--gate-report", str(out)]) == 0
    other = tmp_path / "other.json"
    doc = json.loads(out.read_text())
    doc["estimands"][GEMMA][PRIMARY][M.BRANCH_PRIMARY]["stats"][M.CONTRAST_OBJECT]["point"] += 0.002
    other.write_text(json.dumps(doc))
    assert M.main(["--diff-reports", str(out), str(other)]) == M.EXIT_REPORTS_DIFFER == 4
    assert M.main(["--diff-reports", str(out), str(out)]) == 0
    rc = M.main(_main_args(tmp_path / "gen", tmp_path, "--no-figs", "--out", str(tmp_path / "second.json"),
                           "--require-report-match", str(other)))
    assert rc == 4
    assert json.loads((tmp_path / "second.json").read_text())["report_match"]["equal"] is False
    rc = M.main(_main_args(tmp_path / "gen", tmp_path, "--no-figs", "--out", str(tmp_path / "third.json"),
                           "--require-report-match", str(out)))
    assert rc == 0 and json.loads((tmp_path / "third.json").read_text())["report_match"]["equal"] is True


def test_main_refuses_a_non_production_report_with_exit_3_before_any_side_effect(tmp_path, monkeypatch):
    S.build_all(tmp_path / "gen", rows=primary_rows(GEMMA),
                overrides={(GEMMA, OBJECT, PRIMARY): {"capped_scored": set(range(20))}})
    # the 4 % cell must REFUSE here: below the ceiling the cap gate binds (at the ceiling it is advisory)
    monkeypatch.setattr(M, "AT_CAP_CEILING", M.CAP * 2)
    calls = []
    monkeypatch.setattr(M, "make_figures", lambda *a, **k: calls.append("figs") or {})
    monkeypatch.setattr(M, "mirror_figures", lambda *a, **k: calls.append("mirror") or [])
    monkeypatch.setattr(M, "log_wandb", lambda *a, **k: calls.append("wandb"))
    out = tmp_path / "report.json"
    rc = M.main(_main_args(tmp_path / "gen", tmp_path, "--out", str(out)))
    assert rc == M.EXIT_NOT_PRODUCTION == 3 and calls == []
    rep = json.loads(out.read_text())
    assert "cap" in rep["gates"]["failed"] and rep["verdict"]["word"] is None
    assert M.main(["--gate-report", str(out)]) == 3


def test_two_venv_shape_the_log_reading_run_and_the_figure_run_agree_on_every_numeric_leaf(tmp_path):
    """The runner's run E (inspect venv, eval logs read) and run F (figure venv,
    ``--no-eval-logs``) must agree on every numeric leaf outside the skip list;
    with no ``.eval`` log on disk the record stands alone in E and the two
    reports are numerically identical."""
    S.build_all(tmp_path, rows=primary_rows(GEMMA))
    e = analyze(tmp_path, rows=primary_rows(GEMMA), read_eval_logs=True)
    f = analyze(tmp_path, rows=primary_rows(GEMMA), read_eval_logs=False)
    ec = leg_of(e, GEMMA, BARE)["eval_check"]
    assert ec["available"] is False and ec["log_present"] is False
    assert leg_of(f, GEMMA, BARE)["eval_check"]["reason"] == "log reading disabled"
    d = M.numeric_leaf_diff(e, f)
    assert d["equal"] and d["shared_numeric"] > 1000
    # the skip lists: the parent's component list unchanged (nothing new), plus exactly the four by-design split leaves
    assert set(M.REPORT_DIFF_SKIP) == {"eval_check", "figures", "mirrored", "wandb_url", "analysis_code", "report_match"}
    assert M.REPORT_DIFF_SKIP_SPLIT_FIELDS == ("split_source", "n_loop", "n_truncated_reasoning", "n_text_unavailable")
    # ... and the nested predicate covers exactly the leg record, its controls mirror and the truncation cells — never the
    # store-rows split that rides alongside, never a same-named leaf elsewhere
    leg = M.leg_key(GEMMA, BARE, PRIMARY)
    assert M._split_leaf_skipped(("legs", leg, "at_cap", "n_loop")) and M._split_leaf_skipped(("controls", 0, "at_cap", "split_source"))
    assert M._split_leaf_skipped(("truncation", "cells", 3, "n_text_unavailable"))
    assert not M._split_leaf_skipped(("legs", leg, "at_cap", "store_rows_split", "n_loop"))
    assert not M._split_leaf_skipped(("legs", leg, "at_cap", "n_at_cap")) and not M._split_leaf_skipped(("truncation", "n_loop"))
    assert not M._split_leaf_skipped(("legs", leg, "eval_check", "at_cap_tails", "n_loop"))   # the parent's own skip covers it
    # the ONE re-draw leaf that differs by design is the string naming the attribution source; every numeric leaf of
    # the ledger-derived readout stays in the match, and the log cross-check's numbers ride under the parent's `eval_check`
    assert M.REPORT_DIFF_SKIP_REDRAW_FIELDS == ("attribution_source",)
    assert M._split_leaf_skipped(("legs", leg, "ledger", "redraws", "attribution_source"))
    assert M._split_leaf_skipped(("gates", "ledger", "legs", leg, "redraws", "attribution_source"))
    assert not M._split_leaf_skipped(("legs", leg, "ledger", "redraws", "n_redraw_calls"))
    assert not M._split_leaf_skipped(("legs", leg, "ledger", "attribution_source"))
    assert "eval_check" in M.REPORT_DIFF_SKIP and leg_of(e, GEMMA, BARE)["eval_check"]["redraws"] is None
    # the runner: both venvs, the gate between the passes, the match before publication, the sidecar cmp
    runner = SITE.expand_path("$GRADE/tinycap/math500/analysis_math500.sh")
    if not runner.is_file():
        pytest.skip(f"runner not installed at {runner}")
    text = runner.read_text()
    assert "tinycap-laneB/bin/python" in text and "tiny-capability-lmeval/bin/python" in text
    assert "--gate-report" in text and "--require-report-match" in text and "--no-eval-logs" in text
    assert "--diff-reports" in text and "cmp -s" in text and "DONE" in text
    assert "tinycap_math500_thinking.py" in text


# --------------------------------------------------------------------------- #
# figures, the numbers sidecar, W&B and the mirror                              #
# --------------------------------------------------------------------------- #


def _bbox(artist, renderer):
    return artist.get_window_extent(renderer=renderer)


@pytest.fixture(scope="module")
def full_figures(full_report, tmp_path_factory):
    """The 24-leg report's seven figures, PNG + PDF + sidecars, rendered once: ``(figs_dir, manifest)``."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    figs_dir = tmp_path_factory.mktemp("m5t_figs")
    return figs_dir, M.make_figures(full_report, figs_dir)


def test_figures_render_with_resolvable_sidecar_short_titles_and_tall_axes(full_report, full_figures):
    _figs_dir, figs = full_figures
    assert set(figs["written"]) == set(M.FIGURES) and figs["skipped"] == {}
    # the two readouts' figures are part of the set, each with a caption
    assert {"final_m5t_atcap_share", "final_m5t_atcap_loopsplit", "final_m5t_no_reasoning_share"} <= set(M.FIGURES)
    assert set(M.figure_captions(full_report)) == set(M.FIGURES)
    for stem, paths in figs["written"].items():
        assert [Path(p).suffix for p in paths] == [".png", ".pdf"]
        assert all(Path(p).is_file() and Path(p).stat().st_size > 0 and Path(p).is_absolute() for p in paths)
    numbers = json.loads(Path(figs["numbers"]).read_text())
    assert set(numbers["figures"]) == set(M.FIGURES) and numbers["sha"] == S.SHA7
    n_records = 0
    for stem, recs in numbers["figures"].items():
        for rec in recs:
            n_records += 1
            paths = [rec["source"]] if isinstance(rec["source"], str) else rec["source"]
            leaves = [M.report_path(full_report, p) for p in paths]
            if rec["transform"] is None:
                assert len(leaves) == 1 and M._same_leaf(rec["value"], leaves[0]), (stem, rec["name"])
    assert n_records > 100
    tables = json.loads(Path(figs["tables"]).read_text())
    assert set(tables) >= {"effects", "acc_by_leg", "controls", "verdict", "gates", "mode_deltas", "captions", "sha"}
    # every figure: a title of at most two lines, no subtitle prose, axes >= 80 % of the figure height
    for stem, fn in M.FIGURES.items():
        fig, _nums = fn(full_report)
        fig.canvas.draw()
        title = [t for t in fig.texts if t.get_fontweight() == "bold"]
        assert len(title) == 1 and title[0].get_text().count("\n") <= 1, stem
        assert all(len(line) <= 90 for line in title[0].get_text().split("\n")), stem
        assert M.axes_height_fraction(fig) >= 0.80, (stem, M.axes_height_fraction(fig))
        r = fig.canvas.get_renderer()
        for ax in fig.axes:
            for t in ax.texts:
                assert _bbox(t, r).x1 <= fig.bbox.x1 + 0.5 and _bbox(t, r).x0 >= -0.5, (stem, t.get_text())
        import matplotlib.pyplot as plt
        plt.close(fig)
    with pytest.raises(M.AnalysisError, match="title"):
        M.check_title("a\nb\nc")


def test_figure_exclude_family_leaves_the_family_out_of_figures_and_tables_and_the_report_untouched(full_report, full_figures, tmp_path):
    """GLM stays out of every figure and headline table, its legs in the report's data. With
    ``exclude_families=[GLM]`` all ten stems are drawn without a GLM row, model or comparison, no sidecar record names a GLM
    leg, every record still resolves in the FULL report (the view keeps the report's indices), the tables' per-model / per-leg
    blocks drop GLM while the gates and the verdict stay the wave's, the exclusion is recorded in the manifest and both
    sidecars, the at-cap share title counts the cells shown, the report is not mutated, and an unknown family or every family
    refuses by name."""
    pytest.importorskip("matplotlib")
    import copy
    before = copy.deepcopy(full_report)
    _all_dir, all_figs = full_figures
    figs = M.make_figures(full_report, tmp_path / "noglm", exclude_families=[GLM])
    assert full_report == before and "figures" not in full_report
    # the exclusion (and the families shown) are recorded ONLY under an exclusion: the no-flag outputs keep the pre-flag shape byte for byte
    assert figs[M.FIGURES_EXCLUDED_KEY] == [GLM] and set(figs[M.FAMILIES_SHOWN_KEY]) == set(S.MODEL_KEYS) - {GLM}
    assert M.FIGURES_EXCLUDED_KEY not in all_figs and M.FAMILIES_SHOWN_KEY not in all_figs
    assert set(figs["written"]) == set(M.FIGURES) and figs["skipped"] == {}
    numbers, all_numbers = (json.loads(Path(f["numbers"]).read_text()) for f in (figs, all_figs))
    assert numbers[M.FIGURES_EXCLUDED_KEY] == [GLM] and set(numbers["figures"]) == set(M.FIGURES)
    assert M.FIGURES_EXCLUDED_KEY not in all_numbers and M.FAMILIES_SHOWN_KEY not in all_numbers

    def glm_records(doc):
        return [(st, r["name"]) for st, recs in doc["figures"].items() for r in recs if GLM in r["name"] or GLM in json.dumps(r["source"])]

    assert glm_records(all_numbers) and not glm_records(numbers)
    n_records = 0
    for stem, recs in numbers["figures"].items():
        for rec in recs:
            n_records += 1
            if rec["transform"] is None:
                assert M._same_leaf(rec["value"], M.report_path(full_report, rec["source"])), (stem, rec["name"])
    assert n_records > 100
    tables, all_tables = (json.loads(Path(f["tables"]).read_text()) for f in (figs, all_figs))
    assert tables[M.FIGURES_EXCLUDED_KEY] == [GLM] and M.FIGURES_EXCLUDED_KEY not in all_tables and M.FAMILIES_SHOWN_KEY not in all_tables
    for key in ("effects", "acc_by_leg", "mode_deltas"):
        assert GLM in all_tables[key] and {k: v for k, v in all_tables[key].items() if k != GLM} == tables[key], key
    assert [r for r in all_tables["controls"] if r["model_key"] != GLM] == tables["controls"] and any(r["model_key"] == GLM for r in all_tables["controls"])
    assert {k for k in all_tables["provenance"]["cells"] if k.startswith(f"{GLM}/")} and not {k for k in tables["provenance"]["cells"] if k.startswith(f"{GLM}/")}
    assert tables["gates"] == all_tables["gates"] == full_report["gates"] and tables["verdict"] == all_tables["verdict"] == full_report["verdict"]
    # every figure drawn from the view keeps the title, axes and bounds rules and names no GLM tick or label
    import matplotlib.pyplot as plt
    view = {**full_report, "figures": {M.FIGURES_EXCLUDED_KEY: [GLM]}}
    assert M.figure_excluded_families(view) == {GLM} and M.figure_excluded_families(full_report) == frozenset()
    for stem, fn in M.FIGURES.items():
        fig, nums = fn(view)
        fig.canvas.draw()
        lines = _title_lines(fig)
        assert len(lines) <= 2 and all(len(line) <= M.TITLE_MAX_CHARS for line in lines), (stem, lines)
        assert M.axes_height_fraction(fig) >= M.AXES_MIN_FRACTION and _texts_in_bounds(fig), stem
        assert nums and not [r for r in nums if GLM in r["name"]], stem
        for ax in fig.axes:
            assert not [t.get_text() for t in [*ax.get_xticklabels(), *ax.get_yticklabels(), *ax.texts] if GLM in t.get_text()], stem
        rec = {r["name"]: r for r in nums}
        tr = full_report["truncation"]
        if stem == "final_m5t_atcap_share":
            # a shown-cells aggregate is LABELLED, with the wave's value beside it, in the title and in the sidecar (never a silent substitution)
            shown = [c for c in tr["cells"] if c["model_key"] != GLM and c["primary"]]
            top = max(shown, key=lambda c: c["share"] or 0.0)
            assert lines[1].startswith("3 families shown (wave: premise") and f"max {100 * (top['share'] or 0.0):.1f} %" in lines[1] and not lines[1].endswith("…"), lines[1]
            assert rec["truncation.n_primary_at_or_above_threshold_report"]["value"] == len(tr["primary_at_or_above_threshold"])
            assert rec["truncation.n_primary_at_or_above_threshold_shown"]["value"] == sum(1 for c in shown if c["at_or_above_surfacing_threshold"])
            assert rec["truncation.n_primary_over_1pct_report"]["value"] == len(tr["primary_over_1pct"])
            assert rec["truncation.max_primary_share_shown"]["value"] == top["share"] and rec["truncation.max_primary_share_report"]["value"] == tr["max_primary_share"]
            assert "shown" in rec["truncation.max_primary_share_shown"]["transform"] and rec["truncation.max_primary_share_report"]["transform"] is None
            assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == [m for m in dict.fromkeys(c["model_key"] for c in tr["cells"]) if m != GLM]
        if stem == "final_m5t_atcap_loopsplit":
            legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
            all_rows = [r for r in full_report["controls"] if r.get("at_cap") and r.get("thinking")]
            shown_rows = [r for r in all_rows if r["model_key"] != GLM]
            for f, word in (("n_loop", "degenerate loop"), ("n_truncated_reasoning", "truncated reasoning"), ("n_text_unavailable", "text unavailable")):
                s_v, r_v = sum(r["at_cap"][f] for r in shown_rows), sum(r["at_cap"][f] for r in all_rows)
                assert any(t.startswith(word) and f"({s_v:,} draws over the 3 families shown; wave {r_v:,})" in t for t in legend), (f, legend)
                assert rec[f"at_cap.total_{f}_shown"]["value"] == s_v and rec[f"at_cap.total_{f}_report"]["value"] == r_v
            assert not any(r["name"].endswith(("_shown", "_report")) for r in json.loads(Path(all_figs["numbers"]).read_text())["figures"][stem])
        if stem == "final_m5t_no_reasoning_share":
            assert lines[1].startswith("3 families shown: ") and "; wave max " in lines[1], lines[1]
            assert "thinking.max_share_no_reasoning_report" in rec and rec["thinking.max_share_no_reasoning_report"]["transform"]
        plt.close(fig)
    # the reading over the shown cells: labelled, the wave's premise word and count first; the no-exclusion line is the as-run one
    tr = full_report["truncation"]
    assert M._share_reading(tr) == M._share_reading(tr, shown=None)
    shown_line = M._share_reading(tr, shown=[c for c in tr["cells"] if c["model_key"] != GLM])
    wave = {True: f"premise failed, {len(tr['primary_at_or_above_threshold'])} >= 3 %", False: f"premise holds, {len(tr['primary_over_1pct'])} over 1 %",
            None: "premise undetermined"}[tr["premise_failed"]]
    assert shown_line.startswith(f"3 families shown (wave: {wave}") and len(shown_line) <= M.TITLE_MAX_CHARS
    # (3) a valid exclusion set leaving ONE primary-only model: every figure still renders (the sensitivity figure falls back to the visible set)
    figs1 = M.make_figures(full_report, tmp_path / "gemma_only", exclude_families=[GLM, NEMO, QWEN])
    assert set(figs1["written"]) == set(M.FIGURES) and figs1[M.FAMILIES_SHOWN_KEY] == [GEMMA]
    fig, _n = M.fig_sampling_sensitivity({**full_report, "figures": {M.FIGURES_EXCLUDED_KEY: [GLM, NEMO, QWEN]}})
    assert len(fig.axes) == 1 and [t.get_text() for t in fig.axes[0].texts][:1] == [GEMMA]
    plt.close(fig)
    with pytest.raises(M.AnalysisError, match="leaves no model"):
        M.make_figures(full_report, tmp_path / "bad", exclude_families=[GLM, NEMO, QWEN, GEMMA])
    with pytest.raises(M.AnalysisError, match="not a model key"):
        M.make_figures(full_report, tmp_path / "bad", exclude_families=["nope"])
    with pytest.raises(M.AnalysisError, match="leaves no model"):
        M.make_figures(full_report, tmp_path / "bad", exclude_families=list(S.MODEL_KEYS))


def test_figure_exclude_family_on_the_cli_records_the_exclusion_and_moves_no_numeric_leaf(tmp_path):
    """``--figure-exclude-family GLM`` end to end on a gemma + GLM primary matrix: the run's report carries
    ``figures.excluded_families``, matches the no-flag run under ``--require-report-match``, every data block is equal, all ten
    figures are written without a GLM record, and an unknown family is a refusal by name."""
    pytest.importorskip("matplotlib")
    rows = primary_rows(GEMMA) + primary_rows(GLM)
    S.build_all(tmp_path / "gen", rows=rows)
    sys.modules.pop("wandb", None)
    out_all, out_no = tmp_path / "all.json", tmp_path / "noglm.json"
    assert M.main(_main_args(tmp_path / "gen", tmp_path, "--out", str(out_all), "--figs-dir", str(tmp_path / "figs_all"), rows=rows)) == 0
    assert M.main(_main_args(tmp_path / "gen", tmp_path, "--out", str(out_no), "--figs-dir", str(tmp_path / "figs_noglm"), "--figure-exclude-family", GLM,
                             "--require-report-match", str(out_all), rows=rows)) == 0
    r_all, r_no = (json.loads(f.read_text()) for f in (out_all, out_no))
    assert M.FIGURES_EXCLUDED_KEY not in r_all["figures"] and r_no["figures"][M.FIGURES_EXCLUDED_KEY] == [GLM] and r_no["figures"][M.FAMILIES_SHOWN_KEY] == [GEMMA]
    assert r_no["report_match"]["equal"] is True and M.numeric_leaf_diff(r_all, r_no)["equal"]
    assert M.main(["--diff-reports", str(out_all), str(out_no)]) == 0 and M.main(["--gate-report", str(out_no)]) == 0
    for key in ("cells", "legs", "comparisons", "controls", "estimands", "mode_deltas", "truncation", "gates", "verdict", "provenance"):
        assert r_all[key] == r_no[key], key
    assert GLM in r_no["estimands"] and any(c["model_key"] == GLM for c in r_no["comparisons"])
    assert set(r_all["figures"]["written"]) == set(r_no["figures"]["written"]) == set(M.FIGURES)
    numbers = json.loads(Path(r_no["figures"]["numbers"]).read_text())
    assert numbers[M.FIGURES_EXCLUDED_KEY] == [GLM] and not [r for recs in numbers["figures"].values() for r in recs if GLM in r["name"]]
    assert [r for recs in json.loads(Path(r_all["figures"]["numbers"]).read_text())["figures"].values() for r in recs if GLM in r["name"]]
    with pytest.raises(M.AnalysisError, match="not a model key"):
        M.main(_main_args(tmp_path / "gen", tmp_path, "--out", str(tmp_path / "x.json"), "--figs-dir", str(tmp_path / "figs_x"),
                          "--figure-exclude-family", "nope", rows=rows))


def test_a_figures_dir_holds_one_exclusion_set_and_the_wandb_artifact_is_the_manifest(full_report, tmp_path, monkeypatch):
    """Review of 2026-09-11: a re-used --figs-dir keeps the previous render's files on disk and a directory upload would ship them
    under a config declaring a family excluded. A no-flag render followed by an exclusion render into the same directory (and the
    reverse) refuses by name; the same set re-renders in place (the runner's E then E-publish); stale figure files with no sidecar
    refuse too; the mirror is held to the same rule; the W&B artifact is exactly the manifest's files."""
    pytest.importorskip("matplotlib")
    d = tmp_path / "d"
    M.make_figures(full_report, d)
    with pytest.raises(M.AnalysisError, match=M.FIGURES_EXCLUDED_KEY):
        M.make_figures(full_report, d, exclude_families=[GLM])
    d2 = tmp_path / "d2"
    figs = M.make_figures(full_report, d2, exclude_families=[GLM])
    with pytest.raises(M.AnalysisError, match=M.FIGURES_EXCLUDED_KEY):
        M.make_figures(full_report, d2)
    assert M.make_figures(full_report, d2, exclude_families=[GLM])["written"].keys() == figs["written"].keys()
    d3 = tmp_path / "d3"
    M.make_figures(full_report, d3)
    (d3 / M.NUMBERS_FILENAME).unlink()
    (d3 / "final_m5t_stray.png").write_bytes(b"stale")
    with pytest.raises(M.AnalysisError, match="would not write"):
        M.make_figures(full_report, d3, exclude_families=[GLM])
    with pytest.raises(M.AnalysisError, match=M.FIGURES_EXCLUDED_KEY):
        M.refuse_mixed_render(d, [GLM], list(figs["written"]))
    M.refuse_mixed_render(d2, [GLM], list(figs["written"]))
    M.refuse_mixed_render(tmp_path / "absent", [GLM], [])
    (d2 / "final_m5t_stray.png").write_bytes(b"stale")       # a stray file beside the manifest's: never uploaded
    calls = _fake_wandb(monkeypatch)
    M.log_wandb(full_report, figs, mode="offline")
    want = sorted(Path(p).name for ps in figs["written"].values() for p in ps) + sorted([M.NUMBERS_FILENAME, M.TABLES_FILENAME])
    assert sorted(calls["artifact_files"]) == sorted(want) and "final_m5t_stray.png" not in calls["artifact_files"]
    assert calls["init"]["config"][f"figures_{M.FIGURES_EXCLUDED_KEY}"] == [GLM]


def test_an_excluded_family_supplying_the_maximum_keeps_the_waves_value_labelled(tmp_path):
    """The premise's max at-cap share sits on a GLM cell; excluding GLM must not silently substitute the shown max — the title
    labels it as over the families shown with the wave's verdict beside it, and the sidecar carries both values."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    rows = primary_rows(GEMMA) + primary_rows(GLM)
    S.build_all(tmp_path / "gen", rows=rows, overrides={(GLM, OBJECT, PRIMARY): {"capped_scored": set(range(40))}})
    rep = analyze(tmp_path / "gen", rows=rows)
    tr = rep["truncation"]
    assert tr["max_primary_share_leg"].startswith(f"{GLM}/") and tr["max_primary_share"] >= 0.08
    figs = M.make_figures(rep, tmp_path / "noglm", exclude_families=[GLM])
    rec = {r["name"]: r for r in json.loads(Path(figs["numbers"]).read_text())["figures"]["final_m5t_atcap_share"]}
    assert rec["truncation.max_primary_share_report"]["value"] == tr["max_primary_share"]
    assert rec["truncation.max_primary_share_shown"]["value"] == max(c["share"] for c in tr["cells"] if c["model_key"] == GEMMA and c["primary"]) < tr["max_primary_share"]
    import matplotlib.pyplot as plt
    fig, _n = M.fig_atcap_share({**rep, "figures": {M.FIGURES_EXCLUDED_KEY: [GLM]}})
    line = _title_lines(fig)[1]
    plt.close(fig)
    assert line.startswith("1 family shown (wave: premise failed, ") and f"max {100 * rec['truncation.max_primary_share_shown']['value']:.1f} %" in line, line
    assert GLM not in line and len(line) <= M.TITLE_MAX_CHARS and not line.endswith("…")


def test_shown_only_describes_the_shown_set_labelled_and_carries_no_waves_value(full_report, tmp_path):
    """Review round 2 on the paper view (2026-09-13): under ``shown_only`` the three set-describing figures (at-cap share, loop
    split, no-reasoning share) carry the SHOWN set's descriptor only, labelled as shown, and the numbers sidecar has no
    report-wide aggregate over rows the render does not draw; the switch is recorded in the manifest and both sidecars; the
    report is untouched; the directory holds one setting of it; without a selector it refuses."""
    pytest.importorskip("matplotlib")
    import copy
    import matplotlib.pyplot as plt
    before = copy.deepcopy(full_report)
    figs = M.make_figures(full_report, tmp_path / "shown", exclude_families=[GLM], modes=[PRIMARY], shown_only=True)
    assert full_report == before and "figures" not in full_report
    assert figs[M.FIGURES_SHOWN_ONLY_KEY] is True and figs[M.FIGURES_SELECTION_KEY][M.FIGURES_SHOWN_ONLY_KEY] is True
    numbers, tables = (json.loads(Path(figs[k]).read_text()) for k in ("numbers", "tables"))
    assert numbers[M.FIGURES_SHOWN_ONLY_KEY] is True and tables[M.FIGURES_SHOWN_ONLY_KEY] is True
    recs = [(st, r) for st, rs in numbers["figures"].items() for r in rs]
    assert not [(st, r["name"]) for st, r in recs if r["name"].endswith("_report")]
    assert not [(st, r["name"]) for st, r in recs if "wave" in (r.get("transform") or "")]
    for st, r in recs:
        if r["transform"] is None:
            assert M._same_leaf(r["value"], M.report_path(full_report, r["source"])), (st, r["name"])
    assert "cells SHOWN only" in numbers["captions"]["final_m5t_atcap_share"] and "verdict on the premise" not in numbers["captions"]["final_m5t_atcap_share"]
    view = {**full_report, "figures": {M.FIGURES_EXCLUDED_KEY: [GLM], M.FIGURES_MODES_KEY: [PRIMARY], M.FIGURES_SHOWN_ONLY_KEY: True}}
    assert M.figure_shown_only(view) and not M.figure_shown_only(full_report)
    tr = full_report["truncation"]
    prim = [c for c in tr["cells"] if c["model_key"] != GLM and c["mode"] == PRIMARY and c["primary"]]
    n_thr, n_over = sum(1 for c in prim if c["at_or_above_surfacing_threshold"]), sum(1 for c in prim if c["line_1pct_exceeded"])
    top = max(c["share"] or 0.0 for c in prim)
    fig, nums = M.fig_atcap_share(view)
    fig.canvas.draw()
    lines = _title_lines(fig)
    plt.close(fig)
    assert lines[1] == f"shown: {n_thr} of {len(prim)} primary cells >= 3 %, {n_over} over 1 %, max {100 * top:.1f} % · {M._selection_line(view)}", lines[1]
    assert "wave" not in lines[1] and len(lines[1]) <= M.TITLE_MAX_CHARS
    rec = {r["name"]: r for r in nums}
    assert rec["truncation.n_primary_at_or_above_threshold_shown"]["value"] == n_thr and rec["truncation.max_primary_share_shown"]["value"] == top
    assert "truncation.n_primary_at_or_above_threshold_report" not in rec and "truncation.max_primary_share_report" not in rec
    assert "shown" in rec["truncation.n_primary_at_or_above_threshold_shown"]["transform"]
    rows = [r for r in full_report["controls"] if r.get("at_cap") and r.get("thinking") and r["model_key"] != GLM and r["mode"] == PRIMARY]
    fig, nums = M.fig_atcap_loopsplit(view)
    legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    plt.close(fig)
    for f, word in (("n_loop", "degenerate loop"), ("n_truncated_reasoning", "truncated reasoning"), ("n_text_unavailable", "text unavailable")):
        s_v = sum(r["at_cap"][f] for r in rows)
        assert any(t.startswith(word) and t.endswith(f"({s_v:,} draws over the {len(rows)} legs shown)") for t in legend), (f, legend)
    assert not any("wave" in t for t in legend)
    rec = {r["name"]: r for r in nums}
    assert rec["at_cap.total_n_loop_shown"]["value"] == sum(r["at_cap"]["n_loop"] for r in rows) and "at_cap.total_n_loop_report" not in rec
    fig, nums = M.fig_no_reasoning_share(view)
    lines = _title_lines(fig)
    plt.close(fig)
    assert lines[1].startswith("shown: ") and "wave" not in lines[1] and "thinking.max_share_no_reasoning_report" not in {r["name"] for r in nums}
    # one setting per directory, either way round; and no selector means no shown set
    with pytest.raises(M.AnalysisError, match=M.FIGURES_SHOWN_ONLY_KEY):
        M.make_figures(full_report, tmp_path / "shown", exclude_families=[GLM], modes=[PRIMARY])
    plain = M.make_figures(full_report, tmp_path / "plain", exclude_families=[GLM], modes=[PRIMARY])
    with pytest.raises(M.AnalysisError, match=M.FIGURES_SHOWN_ONLY_KEY):
        M.make_figures(full_report, tmp_path / "plain", exclude_families=[GLM], modes=[PRIMARY], shown_only=True)
    with pytest.raises(M.AnalysisError, match="needs a selector"):
        M.make_figures(full_report, tmp_path / "none", shown_only=True)
    assert M.FIGURES_SHOWN_ONLY_KEY not in plain


def test_shown_only_false_equals_shown_only_absent_on_the_synthetic_report(full_report, full_figures, tmp_path):
    """What this proves, exactly: two renders by the CURRENT code — ``shown_only`` absent vs ``shown_only=False`` — write
    byte-equal sidecars on the synthetic report, the key appears nowhere, and the labelled-exclusion path keeps the wave's
    value beside the shown one. It does NOT compare against a pre-change render; that is
    :func:`test_default_render_is_byte_identical_to_the_preserved_pre_flag_render`."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    _all_dir, all_figs = full_figures
    for doc in (all_figs, json.loads(Path(all_figs["numbers"]).read_text()), json.loads(Path(all_figs["tables"]).read_text())):
        assert M.FIGURES_SHOWN_ONLY_KEY not in doc
    a = M.make_figures(full_report, tmp_path / "a", exclude_families=[GLM])
    b = M.make_figures(full_report, tmp_path / "b", exclude_families=[GLM], shown_only=False)
    for key in ("numbers", "tables"):
        assert Path(a[key]).read_bytes() == Path(b[key]).read_bytes(), key
    numbers = json.loads(Path(a["numbers"]).read_text())
    assert M.FIGURES_SHOWN_ONLY_KEY not in a and M.FIGURES_SHOWN_ONLY_KEY not in numbers and "verdict on the premise" in numbers["captions"]["final_m5t_atcap_share"]
    rec = {r["name"]: r for r in numbers["figures"]["final_m5t_atcap_share"]}
    assert rec["truncation.n_primary_at_or_above_threshold_report"]["transform"] == "length of the list (the wave's value)"
    view = {**full_report, "figures": {M.FIGURES_EXCLUDED_KEY: [GLM]}}
    fig, _n = M.fig_atcap_share(view)
    line = _title_lines(fig)[1]
    plt.close(fig)
    assert line.startswith("3 families shown (wave: premise")
    fig, _n = M.fig_atcap_loopsplit(view)
    legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    plt.close(fig)
    assert all("; wave " in t for t in legend)


#: the preserved PRE-FLAG render of the production report — a directory holding ``RENDER.json`` (``report``: the production
#: report's path; ``figure_code_sha``: the renderer commit that drew it, the parent of the shown_only commit) and ``figs/``
#: (the ten PNG + PDF pairs and the two JSON sidecars). Machine-local by nature (the production report is not in the
#: repository), so the comparison test SKIPS when the variable is unset or the directory absent, and says so.
PREFLAG_RENDER_ENV = "SISTEER_M5T_PREFLAG_RENDER"
#: the first publication's preserved figure set (the production analysis run's own render, an EARLIER renderer) — value
#: identity name-for-name, not bytes: the module added sidecar entries, captions and a `notes` block between that run and
#: the flag's parent, so its bytes are not reproducible by the current default render and the test does not claim they are
R1_RENDER_ENV = "SISTEER_M5T_R1_RENDER"


def test_default_render_is_byte_identical_to_the_preserved_pre_flag_render(tmp_path):
    """Review round 3 on the paper view (M4): the default render (no selector, ``shown_only`` absent) of the PRODUCTION report by
    the current code is compared with a PRESERVED render of the same report by the pre-flag renderer (the parent of the
    shown_only commit), named by :data:`PREFLAG_RENDER_ENV`. Compared, exactly: (1) the two JSON sidecars, byte for byte;
    (2) every PNG, by sha256 of the bytes (matplotlib's Agg PNG output is deterministic for the same code, data and font
    stack — two renders here agree byte for byte); the PDFs are NOT compared — each carries its creation timestamp, so their
    bytes differ by construction (checked only for presence). (3) When :data:`R1_RENDER_ENV` names the first publication's
    preserved set, every entry of ITS numbers sidecar is found in the default render's with the same source, value and
    transform (name for name; the default render carries more entries, never fewer or moved ones)."""
    pytest.importorskip("matplotlib")
    root = os.environ.get(PREFLAG_RENDER_ENV)
    if not root or not (Path(root).expanduser() / "RENDER.json").is_file():
        pytest.skip(f"{PREFLAG_RENDER_ENV} does not name a preserved pre-flag render (RENDER.json + figs/); the production report is machine-local")
    root = Path(root).expanduser()
    stamp = json.loads((root / "RENDER.json").read_text(encoding="utf-8"))
    report_path = Path(stamp["report"])
    if not report_path.is_file():
        pytest.skip(f"the preserved render's report {report_path} is not on this machine")
    import hashlib
    ref_dir = root / "figs"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == stamp["report_sha256"], "the production report moved under the preserved render"
    figs = M.make_figures(report, tmp_path / "default")
    here = tmp_path / "default"
    assert figs["skipped"] == {} and M.FIGURES_SHOWN_ONLY_KEY not in figs
    for name in (M.NUMBERS_FILENAME, M.TABLES_FILENAME):
        assert (here / name).read_bytes() == (ref_dir / name).read_bytes(), f"{name}: the default render's sidecar differs from the pre-flag render {stamp['figure_code_sha'][:7]}"
    pngs = sorted(p.name for p in ref_dir.glob("*.png"))
    assert pngs and set(pngs) == {Path(p).name for paths in figs["written"].values() for p in paths if p.endswith(".png")}
    for name in pngs:
        assert hashlib.sha256((here / name).read_bytes()).hexdigest() == hashlib.sha256((ref_dir / name).read_bytes()).hexdigest(), \
            f"{name}: PNG bytes differ from the pre-flag render {stamp['figure_code_sha'][:7]}"
    assert all((here / n.replace(".png", ".pdf")).is_file() and (ref_dir / n.replace(".png", ".pdf")).is_file() for n in pngs)
    r1 = os.environ.get(R1_RENDER_ENV)
    if not r1 or not (Path(r1).expanduser() / M.NUMBERS_FILENAME).is_file():
        return
    r1_doc = json.loads((Path(r1).expanduser() / M.NUMBERS_FILENAME).read_text(encoding="utf-8"))
    here_doc = json.loads((here / M.NUMBERS_FILENAME).read_text(encoding="utf-8"))
    here_by = {(stem, e["name"]): e for stem, es in here_doc["figures"].items() for e in es}
    missing = [(stem, e["name"]) for stem, es in r1_doc["figures"].items() for e in es if (stem, e["name"]) not in here_by]
    assert not missing, f"entries of the first publication's sidecar absent from the default render: {missing[:5]}"
    moved = [(stem, e["name"]) for stem, es in r1_doc["figures"].items() for e in es
             if (e["source"], e.get("transform")) != (here_by[(stem, e["name"])]["source"], here_by[(stem, e["name"])].get("transform"))
             or not M._same_leaf(e["value"], here_by[(stem, e["name"])]["value"])]
    assert not moved, f"entries whose source, transform or value differ from the first publication's: {moved[:5]}"
    assert r1_doc["sha"] == here_doc["sha"] == report["sha"] and r1_doc["analysis_code_sha"] == here_doc["analysis_code_sha"]


def test_shown_only_scopes_legend_captions_and_tables_to_the_shown_rows_and_refuses_a_leak(full_report, full_figures, tmp_path, monkeypatch):
    """Review round 3 on the paper view (2026-09-14): under ``shown_only`` no sidecar, legend or caption names a row the render
    leaves out. Pinned here on the synthetic 24-leg report with GLM excluded and the primary row alone shown: the at-cap
    legend has no lighter-shade entry (no sensitivity cell is drawn) — and has one when a sensitivity mode IS shown; the
    captions describe the rows drawn (no "(general, battery)", no excluded family among the no-reasoning classes, "primary
    rows" alone in the contrasts caption) and the tables carry the same captions; the sampling-row figure is skipped with its
    reason; the tables JSON holds the shown rows only — controls, effects / acc_by_leg modes, no mode delta, the verdict's
    words / statements / joined string, every gate's per-leg block, the cap advisories, provenance.cells, the notes; the
    selection rule says so; :func:`shown_only_leaks` is empty on both sidecars and the publication gate's own predicate
    finds no excluded family or hidden mode outside the selection record; a scoping that leaks makes the render REFUSE; and
    the default render is untouched (its legend, captions and rule text are the published ones)."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    figs = M.make_figures(full_report, tmp_path / "shown", exclude_families=[GLM], modes=[PRIMARY], shown_only=True)
    assert "final_m5t_sampling_sensitivity" not in figs["written"] and "two rows" in figs["skipped"]["final_m5t_sampling_sensitivity"]
    assert figs[M.FIGURES_SELECTION_KEY]["rule"] == M.SELECTION_RULE_SHOWN_ONLY
    numbers, tables = (json.loads(Path(figs[k]).read_text()) for k in ("numbers", "tables"))
    assert numbers[M.FIGURES_SELECTION_KEY]["rule"] == tables[M.FIGURES_SELECTION_KEY]["rule"] == M.SELECTION_RULE_SHOWN_ONLY
    view = {**full_report, "figures": {M.FIGURES_EXCLUDED_KEY: [GLM], M.FIGURES_MODES_KEY: [PRIMARY], M.FIGURES_SHOWN_ONLY_KEY: True}}
    shown_legs = {r["leg"] for r in full_report["controls"] if r["model_key"] != GLM and r["mode"] == PRIMARY}
    shown_models = [m for m in (GEMMA, QWEN, NEMO) if any(r["model_key"] == m for r in full_report["controls"])]
    # the legend describes what is drawn
    fig, _n = M.fig_atcap_share(view)
    legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    plt.close(fig)
    assert not any("sensitivity" in t for t in legend), legend
    two_rows = {**full_report, "figures": {M.FIGURES_EXCLUDED_KEY: [GLM], M.FIGURES_MODES_KEY: [PRIMARY, BATTERY], M.FIGURES_SHOWN_ONLY_KEY: True}}
    fig, _n = M.fig_atcap_share(two_rows)
    legend2 = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    plt.close(fig)
    assert "sensitivity row (lighter)" in legend2
    # the captions describe the rows drawn; the tables carry the same captions
    cap = numbers["captions"]
    assert set(cap) == set(figs["written"]) and tables["captions"] == cap
    assert "the primary cells in the cell hue." in cap["final_m5t_atcap_share"] and "(general, battery)" not in cap["final_m5t_atcap_share"]
    assert cap["final_m5t_contrasts"].startswith("arm − bare per model (primary rows), as") and "sensitivity row" not in cap["final_m5t_contrasts"]
    assert "record_only (gemma4e4b)" in cap["final_m5t_no_reasoning_share"] and "explicit_tags" not in cap["final_m5t_no_reasoning_share"]
    assert f"implicit_open families ({QWEN}, {NEMO})" in cap["final_m5t_no_reasoning_share"]
    cap2 = M.figure_captions(two_rows)
    assert f"the sensitivity rows ({BATTERY}) in a lighter shade" in cap2["final_m5t_atcap_share"] and "general" not in cap2["final_m5t_atcap_share"]
    assert cap2["final_m5t_contrasts"].startswith("arm − bare per model (primary rows first, then each sensitivity row")
    assert M._sensitivity_caption(view).endswith("as the accuracy figure does.") and "(general, battery)" not in M._sensitivity_caption(view)
    # the tables JSON holds the shown rows only
    assert {r["leg"] for r in tables["controls"]} == shown_legs
    for block in ("effects", "acc_by_leg"):
        assert set(tables[block]) == set(shown_models) and all(set(by) == {PRIMARY} for by in tables[block].values()), block
    assert tables["mode_deltas"] == {}
    assert set(tables["verdict"]["words"]) == set(tables["verdict"]["statements"]) == set(shown_models)
    assert GLM not in tables["verdict"]["verdict"] and all(m in tables["verdict"]["verdict"] for m in shown_models)
    assert tables["verdict"]["word"] == full_report["verdict"]["word"]
    for gate in ("ledger", "identity", "certification", "cap", "scoring_path"):
        assert set(tables["gates"][gate]["legs"]) == shown_legs, gate
        assert tables["gates"][gate]["status"] == full_report["gates"][gate]["status"]
    assert set(tables["gates"]["split_anomalies"]["counts"]) == shown_legs
    assert all(a.split(":")[0] in shown_legs for a in tables["gates"]["cap"]["advisories"])
    assert set(tables["provenance"]["cells"]) == shown_legs
    assert all(k.split("/")[0] in shown_models and k.split("/")[1] == PRIMARY for k in tables["notes"]["served_geometry"])
    assert all(k.split("/")[0] in shown_models and k.split("/")[1] == PRIMARY for k in tables["notes"]["conditional_vs_score_as_is"]["by_comparison"])
    assert tables[M.FIGURES_EXCLUDED_KEY] == [GLM]                    # the selection record names what was left out, by design
    # no hidden row anywhere outside the selection record — by this module's check and by the publication gate's own predicate
    assert M.shown_only_leaks(numbers, view) == [] and M.shown_only_leaks(tables, view) == []
    hidden_modes = [m for m in (GENERAL, BATTERY) if any(r["mode"] == m for r in full_report["controls"])]
    for doc in (numbers, tables):
        # the gate's predicate reads the mode NAMES as whole segments, so it runs over the row-keyed and figure-describing parts;
        # the report's free-text provenance (trigger / sampling / note) may say "battery" for the cluster instrument, a common
        # noun, and is left to the identifier-level check above
        body = {k: v for k, v in doc.items() if k not in M.SELECTION_RECORD_KEYS}
        if isinstance(body.get("provenance"), dict):
            body["provenance"] = {"cells": body["provenance"].get("cells")}
        assert M.PG.entry_violations(body, excluded_families=[GLM], excluded_modes=hidden_modes) == []
    # a scoping that leaks makes the render refuse before either sidecar is written
    monkeypatch.setattr(M, "scope_tables_to_shown", lambda tables, view: dict(tables))
    with pytest.raises(M.AnalysisError, match="names a row the render does not draw"):
        M.make_figures(full_report, tmp_path / "leaky", exclude_families=[GLM], modes=[PRIMARY], shown_only=True)
    assert not (tmp_path / "leaky" / M.TABLES_FILENAME).exists() and not (tmp_path / "leaky" / M.NUMBERS_FILENAME).exists()
    monkeypatch.undo()
    # the default render is untouched
    _all_dir, all_figs = full_figures
    all_numbers = json.loads(Path(all_figs["numbers"]).read_text())
    assert "the sensitivity rows (general, battery) in a lighter shade" in all_numbers["captions"]["final_m5t_atcap_share"]
    assert "explicit_tags (glm)" in all_numbers["captions"]["final_m5t_no_reasoning_share"]
    assert all_numbers["captions"]["final_m5t_contrasts"].startswith("arm − bare per model (primary rows first, then each sensitivity row")
    fig, _n = M.fig_atcap_share(full_report)
    legend_all = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    plt.close(fig)
    assert "sensitivity row (lighter)" in legend_all
    plain = M.make_figures(full_report, tmp_path / "plain", exclude_families=[GLM], modes=[PRIMARY])
    assert plain[M.FIGURES_SELECTION_KEY]["rule"] == M.SELECTION_RULE and "final_m5t_sampling_sensitivity" in plain["written"]
    plain_tables = json.loads(Path(plain["tables"]).read_text())
    assert {r["leg"] for r in plain_tables["controls"]} > shown_legs and GLM not in plain_tables["effects"]


def _fake_wandb(monkeypatch):
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
            calls.setdefault("artifact_files", []).append(Path(path).name)

    def _init(**kw):
        calls["init"] = kw
        return _Run()

    fake = SimpleNamespace(init=_init, Image=lambda p: p, Artifact=_Artifact,
                           log=lambda d: calls.setdefault("logged", []).append(d))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return calls


def test_wandb_run_name_figure_keys_summary_and_the_mirror(full_report, full_figures, tmp_path, monkeypatch):
    calls = _fake_wandb(monkeypatch)
    monkeypatch.setenv("SISTEER_WANDB_PROJECT_TINYCAP", "example-tinycap")        # the purpose's project, as a site names it
    png = tmp_path / "figs" / "final_m5t_contrasts.png"
    png.parent.mkdir()
    png.write_bytes(b"png")
    url = M.log_wandb(full_report, {"written": {"final_m5t_contrasts": [str(png)]}}, mode="offline")
    assert url == "https://wandb.test/run" and calls["finished"]
    assert calls["init"]["name"] == f"m5t-{S.SHA7}" and calls["init"]["project"] == "example-tinycap"
    assert calls["artifact"] == f"figures-m5t-{S.SHA7}" and calls["artifact_files"] == [png.name]
    assert calls["init"]["config"][f"figures_{M.FIGURES_EXCLUDED_KEY}"] == []
    assert calls["logged"] == [{"m5t/final_m5t_contrasts": str(png)}]
    s = calls["summary"]
    assert s[f"{QWEN}/{PRIMARY}/{M.BRANCH_PRIMARY}/{M.CONTRAST_OBJECT}/point"] == pytest.approx(-0.08)
    assert s[f"controls/{QWEN}/{OBJECT}/{PRIMARY}/acc"] == pytest.approx(0.52)
    assert s["verdict"] == full_report["verdict"]["verdict"]
    # the two readouts' numbers per cell and the premise verdict
    assert s[f"controls/{QWEN}/{OBJECT}/{PRIMARY}/at_cap_share"] == 0.0 and s[f"controls/{QWEN}/{OBJECT}/{PRIMARY}/at_cap_n_loop"] == 0
    assert s[f"controls/{GLM}/{BARE}/{PRIMARY}/thinking_share_no_reasoning"] == pytest.approx(0.004)
    assert s["truncation/premise_failed"] is False and s["truncation/n_primary_at_or_above_threshold"] == 0
    assert s["truncation/max_primary_share"] == 0.0 and s["truncation/n_primary_over_1pct"] == 0
    assert M.log_wandb(full_report, {}, mode="disabled") is None
    # the mirror copies exactly the manifest's files, byte-identical, and refuses a missing entry
    figs_dir, figs = full_figures
    mirrored = M.mirror_figures(figs_dir, tmp_path / "mirror", manifest=figs)
    want = sorted(Path(p).name for ps in figs["written"].values() for p in ps) + \
        sorted([Path(figs["numbers"]).name, Path(figs["tables"]).name])
    assert sorted(Path(p).name for p in mirrored) == sorted(want)
    bad = {**figs, "written": {**figs["written"], "ghost": [str(figs_dir / "ghost.png")]}}
    with pytest.raises(M.AnalysisError, match="ghost"):
        M.mirror_figures(figs_dir, tmp_path / "mirror2", manifest=bad)


# --------------------------------------------------------------------------- #
# the reader reads the PRODUCER's records — built through its helpers          #
# --------------------------------------------------------------------------- #


def _producer_certification_inputs(model_key=QWEN, cell=OBJECT):
    """The real producer helpers' outputs for one fitted leg, as the driver files them."""
    fact = S.RENDER_FACTS[S.FAMILY_BY_KEY[model_key]]
    evidence = {S.EVIDENCE_RENDER_CERTIFIED: S.render_certified_record(fact["family"], fact),
                S.EVIDENCE_RENDER_PROBE: S.render_probe_record(fact),
                C.OOS_EVIDENCE_BITES_BLOCK: S.steering_bites_record(cell),
                S.EVIDENCE_LENS_POSITION: S.lens_position_record(model_key, cell),
                S.EVIDENCE_LENS_ON_BITE: S.lens_on_bite_record(cell, fact)}
    manifest = {C.MANIFEST_KEY_ROUTE: {"virtual_model": S.vmodel_of(cell), C.ROUTE_KEY_SERVE_DECLARATIONS: {},
                                       C.LENS_SERVED_POSITIONS_KEY: ["all"]},
                C.LENS_SERVED_POSITIONS_KEY: ["all"]}
    row = S.matrix_row(model_key, cell, PRIMARY)
    return evidence, manifest, row, fact


def test_certification_reader_accepts_the_real_producer_helpers_outputs():
    """``render_certified()``, ``lens_position_attested()`` over a
    producer-shaped standup record, ``lens_on_bite_verdict()``, the driver's
    ``steering_bites`` block and ``route.lens_served_positions`` as the SELECTED
    cell's LIST pass the certification reader with no failure and nothing pending."""
    for model_key in (QWEN, GEMMA, GLM, NEMO):
        evidence, manifest, row, fact = _producer_certification_inputs(model_key)
        rec = M.certification_check(evidence, manifest, row=row, fact=fact, family=fact["family"],
                                    object_pin=S.object_pin(model_key))
        assert rec["failures"] == [], (model_key, rec["failures"])
        assert rec["passed"] is True and rec["pending_stamps"] == []
    # the producer's FAILURE shapes are refused by name: RenderCertificationError.evidence, the (B′) / (C) error records
    evidence, manifest, row, fact = _producer_certification_inputs()
    with pytest.raises(C.RenderCertificationError) as e:
        C.render_certified(family=fact["family"], header_ids_by_arm={"off": [1], "on": [2]}, measured_live=False)
    bad = {**evidence, S.EVIDENCE_RENDER_CERTIFIED: dict(e.value.evidence)}
    d = " ".join(M.certification_check(bad, manifest, row=row, fact=fact, family=fact["family"],
                                       object_pin=S.object_pin(QWEN))["failures"])
    assert "render_certified" in d and "render_source" in d and "measured_header_ids_by_arm" in d
    bad = {**evidence, S.EVIDENCE_LENS_POSITION: {"verified": False, "error": "OOS_DECLARATION_MISMATCH: x"}}
    d = " ".join(M.certification_check(bad, manifest, row=row, fact=fact, family=fact["family"],
                                       object_pin=S.object_pin(QWEN))["failures"])
    assert "lens_position_attested" in d and "OOS_DECLARATION_MISMATCH" in d
    bad = {**evidence, S.EVIDENCE_LENS_ON_BITE: {"applicable": True, "verified": False, "error": "predicate (C) lens-ON bite failed"}}
    d = " ".join(M.certification_check(bad, manifest, row=row, fact=fact, family=fact["family"],
                                       object_pin=S.object_pin(QWEN))["failures"])
    assert "lens_on_bite" in d and "predicate (C)" in d


LENS_SERVING_IDENTITY, LORA_SERVING_IDENTITY = "proxy_vllm+lens+interleaved", "proxy_vllm+lora+interleaved"
#: the stamp triple every reply on a NATIVE leg wears (the proxy's non-lens triple: no selector, so the resumed policy
#: is `not-consulted`; the salt regime cell-stable) — `test_served_capability_native.NATIVE_TRIPLE`'s spellings
NATIVE_STAMPS = {"salt_regime": "cell-stable", "resumed_policy": "not-consulted", "render_regime": C.RENDER_REGIME_THINKING_OFF}


def _native_producer_certification_inputs(model_key=QWEN, cell=OBJECT):
    """The real producer helpers' outputs for one NATIVELY served fitted leg, as the driver files them: (A) through
    ``contracts.render_certified``; (B′) the driver's OWN inapplicability record
    (``served_capability_run._native_lens_position_record`` — a native standup has no served-positions map, no resumed-
    prefill policy, no lens salt); (C) through ``contracts.lens_on_bite_verdict`` (engine-neutral: the ON bite runs against
    the base route on both engines); the bites block against the ENGINE's zero cell ``none__lora0`` with the non-lens probe
    body and stamp triple; the route block stamped ``+lora+`` / ``serving_engine lora_native`` / ``reduction_mode
    not_applicable`` with ``lens_served_positions`` NULL — the driver's ``_standup_served_positions`` finds no map on a
    native record and writes the null through (the engine's fact, `header_located = False`)."""
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    import served_capability_run as srun
    fact = S.RENDER_FACTS[S.FAMILY_BY_KEY[model_key]]
    bites = {**S.steering_bites_record(cell), C.OOS_EVIDENCE_BITES_TWIN: C.TWIN_CELL_ID_LORA,
             "probe_extra_body": C.render_fragment(lens=False, arm=C.THINKING_OFF, header_located=False),
             "stamps_by_leg": {"steered": [dict(NATIVE_STAMPS) for _ in range(S.N_BITES_PROMPTS)],
                               "scale0": [dict(NATIVE_STAMPS) for _ in range(S.N_BITES_PROMPTS)]}}
    evidence = {S.EVIDENCE_RENDER_CERTIFIED: S.render_certified_record(fact["family"], fact),
                S.EVIDENCE_RENDER_PROBE: S.render_probe_record(fact),
                C.OOS_EVIDENCE_BITES_BLOCK: bites,
                S.EVIDENCE_LENS_POSITION: srun._native_lens_position_record(),
                S.EVIDENCE_LENS_ON_BITE: S.lens_on_bite_record(cell, fact)}
    route = {"virtual_model": S.vmodel_of(cell), "serving_identity": LORA_SERVING_IDENTITY,
             C.SERVED_ENGINE_FIELD: C.SERVING_ENGINE_LORA_NATIVE, C.REDUCTION_MODE_FIELD: C.REDUCTION_MODE_NOT_APPLICABLE,
             C.ROUTE_KEY_SERVE_DECLARATIONS: {}, C.LENS_SERVED_POSITIONS_KEY: None}
    manifest = {C.MANIFEST_KEY_ROUTE: route, C.LENS_SERVED_POSITIONS_KEY: None}
    row = S.matrix_row(model_key, cell, PRIMARY)
    return evidence, manifest, row, fact


def _native_routing_table_under(gen_root, *, served_positions_map, model_key=QWEN, cell=OBJECT):
    """A pulled NATIVE standup record's routing table under ``<gen_root>/serving/<jobdir>/math500/`` — ``lens_served_positions``
    EMPTY (a native standup states no lens mode) and the ENGINE-NEUTRAL ``served_positions`` map as given — plus the manifest
    fields that locate it (the cluster's absolute ``routing_table`` path, of which the reader takes the trailing
    ``serving/<jobdir>/math500/routing_table.json``). Returns the manifest fields."""
    import json as _json
    job = S.standup_job_id(model_key, cell, PRIMARY, tree_sha=S.FULL_SHA)
    jobdir = S.standup_jobdir(model_key, cell, PRIMARY, job)
    d = Path(gen_root) / S.STANDUP_DIR_PARENT / jobdir / S.STANDUP_TASK_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    hf, rev = S.model_pin(model_key)
    table = {"base_model": hf, "base_revision": rev, "serving_mode": "interleaved", "version": 2, C.LENS_SERVED_POSITIONS_KEY: {}}
    if served_positions_map is not None:
        table[C.SERVED_POSITIONS_KEY] = dict(served_positions_map)
    (d / S.STANDUP_ROUTING_TABLE).write_text(_json.dumps(table, indent=1), encoding="utf-8")
    return {"routing_table": S.cluster_routing_table_path(jobdir)}


def test_a_native_legs_lens_named_stamp_is_held_to_the_routing_tables_neutral_map(tmp_path):
    """The FIXED producer writes the routing table's ENGINE-NEUTRAL `served_positions` value into the LENS-named field on a
    native leg too (`served_capability_run._standup_served_positions` -> the manifest write at 6315-6316), so a natively served
    fitted leg carries `route.lens_served_positions = ['all']` (the pulled qwen35_4b `arditi_allpos` MATH-500 leg) and the
    reader refused it as a lens fact. The stamp is admitted ONLY as the pulled routing table's `served_positions[<harness>]`
    under the leg's wave root (`certification_check(..., gen_root=)`): a differing value, a table with no neutral entry, no
    table in the pull, and no wave root at all refuse by name — and the null stamp, the torn native route and every lens
    reading are byte-identical."""
    harness = S.harness_id_of(OBJECT)
    evidence, manifest, row, fact = _native_producer_certification_inputs()
    route = manifest[C.MANIFEST_KEY_ROUTE]

    def stamped(value, locate):
        return {**manifest, **locate, C.MANIFEST_KEY_ROUTE: {**route, C.LENS_SERVED_POSITIONS_KEY: list(value)}, C.LENS_SERVED_POSITIONS_KEY: list(value)}

    def cert(man, **kw):
        return M.certification_check(evidence, man, row=row, fact=fact, family=fact["family"], object_pin=S.object_pin(QWEN), **kw)

    root = tmp_path / "admitted"
    locate = _native_routing_table_under(root, served_positions_map={harness: ["all"]})
    rec = cert(stamped(["all"], locate), gen_root=root)
    assert rec["failures"] == [] and rec["passed"] and rec["route_lens_served_positions"] == ["all"], rec["failures"]
    assert M.native_served_positions_stamp_problem(stamped(["all"], locate), served=["all"], harness=harness, gen_root=root) is None
    # the null stamp (the record-gap producer) still passes, with or without a wave root
    assert cert(manifest, gen_root=root)["failures"] == [] and cert(manifest)["failures"] == []
    # no wave root: the stamp cannot be held to anything and refuses as before, in the lens-fact words
    d = " ".join(cert(stamped(["all"], locate))["failures"])
    assert "a served-positions map is a lens fact" in d and "no wave root was given" in d, d
    # (a) native ['prefill'] against the neutral map's ['all']
    d = " ".join(cert(stamped(["prefill"], locate), gen_root=root)["failures"])
    assert "route.lens_served_positions = ['prefill'] on a natively served cell" in d and "the stamp ['prefill'] !=" in d and "['all']" in d, d
    # (b) the stamp with NO neutral entry for the cell (the routes[] attestation admits no stamp)
    root = tmp_path / "noneutral"
    locate = _native_routing_table_under(root, served_positions_map=None)
    d = " ".join(cert(stamped(["all"], locate), gen_root=root)["failures"])
    assert f"carries no served_positions[{harness!r}] entry" in d and "never the routes[] attestation" in d, d
    root = tmp_path / "othercell"
    locate = _native_routing_table_under(root, served_positions_map={"some_other_cell": ["all"]})
    assert f"carries no served_positions[{harness!r}] entry" in " ".join(cert(stamped(["all"], locate), gen_root=root)["failures"])
    # no routing table in the pull, and a manifest that names none
    root = tmp_path / "notpulled"
    locate = _native_routing_table_under(root, served_positions_map={harness: ["all"]})
    (Path(root) / S.STANDUP_DIR_PARENT / locate["routing_table"].split("/")[-3] / S.STANDUP_TASK_SUBDIR / S.STANDUP_ROUTING_TABLE).unlink()
    assert "is not in the pull" in " ".join(cert(stamped(["all"], locate), gen_root=root)["failures"])
    assert "names no routing_table" in " ".join(cert(stamped(["all"], {}), gen_root=root)["failures"])
    # a value that is not a served-positions value at all
    assert "is not a served-positions value" in " ".join(cert(stamped([7], locate), gen_root=root)["failures"])
    # the LENS reading with a wave root given is byte-identical: the helper is never consulted on a lens route
    lens_ev, lens_man, lens_row, lens_fact = _producer_certification_inputs()
    a = M.certification_check(lens_ev, lens_man, row=lens_row, fact=lens_fact, family=lens_fact["family"], object_pin=S.object_pin(QWEN))
    b = M.certification_check(lens_ev, lens_man, row=lens_row, fact=lens_fact, family=lens_fact["family"], object_pin=S.object_pin(QWEN), gen_root=root)
    assert a == b and a["failures"] == []


def test_certification_reader_accepts_a_natively_served_fitted_legs_producer_records():
    """The reader used to run the LENS predicate (B′) on EVERY fitted arm — the
    lens fields, the object-digest map, the policy and the stamp triple, and the lens served-positions rule, so a correct
    native LoRA store (``lens_position_attested`` ``not_applicable`` / ``verified`` null, ``lens_served_positions`` null)
    was REJECTED. Now the predicates dispatch on the route's own serving path: a native leg's evidence, built through the
    producer's helpers, passes with nothing pending; each lens-only fact smuggled onto the native record, the native record
    on a lens route, a torn identity/engine pairing, and a header-located row served natively refuse BY NAME; the lens
    reading of the producer helpers' outputs (the test above) is untouched, with or without the identity stated."""
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    import served_capability_run as srun
    for model_key in (QWEN, GEMMA, GLM, NEMO):
        evidence, manifest, row, fact = _native_producer_certification_inputs(model_key)
        rec = M.certification_check(evidence, manifest, row=row, fact=fact, family=fact["family"], object_pin=S.object_pin(model_key))
        assert rec["failures"] == [], (model_key, rec["failures"])
        assert rec["passed"] is True and rec["pending_stamps"] == []
        assert rec[S.EVIDENCE_LENS_POSITION]["status"] == C.NOT_APPLICABLE and rec["route_lens_served_positions"] is None
        assert rec[M.EVIDENCE_BITES][C.OOS_EVIDENCE_BITES_TWIN] == C.TWIN_CELL_ID_LORA
    evidence, manifest, row, fact = _native_producer_certification_inputs()
    route = manifest[C.MANIFEST_KEY_ROUTE]

    def fails(ev=None, man=None, *, needle, r=None):
        d = " ".join(M.certification_check(ev if ev is not None else evidence, man if man is not None else manifest,
                                           row=r if r is not None else row, fact=fact, family=fact["family"],
                                           object_pin=S.object_pin(QWEN))["failures"])
        assert needle in d, (needle, d)
        return d

    # the LENS (B′) success record on the native route is another engine's record; a failed lens attestation and an absent
    # block refuse too (the driver records the inapplicability by name, never an absent block)
    fails({**evidence, S.EVIDENCE_LENS_POSITION: S.lens_position_record(QWEN, OBJECT)}, needle="another engine's")
    fails({**evidence, S.EVIDENCE_LENS_POSITION: {"verified": False, "error": "OOS_DECLARATION_MISMATCH: x"}},
          needle="FAILED as recorded")
    fails({k: v for k, v in evidence.items() if k != S.EVIDENCE_LENS_POSITION}, needle="never an absent block")
    fails({**evidence, S.EVIDENCE_LENS_POSITION: {**srun._native_lens_position_record(), "verified": True}},
          needle="nothing is attested on this engine")
    # a served-positions map on the native record is a lens fact; a header-located row has no native standup
    fails(man={**manifest, C.MANIFEST_KEY_ROUTE: {**route, C.LENS_SERVED_POSITIONS_KEY: ["all"]}, C.LENS_SERVED_POSITIONS_KEY: ["all"]},
          needle="a served-positions map is a lens fact")
    fails(r={**row, "served_positions_expected": "response"}, needle="a header-located row has no native standup")
    # the twin is the ENGINE's zero cell (the lens twin on a native route refuses, as before)
    fails({**evidence, C.OOS_EVIDENCE_BITES_BLOCK: {**evidence[C.OOS_EVIDENCE_BITES_BLOCK], C.OOS_EVIDENCE_BITES_TWIN: C.TWIN_CELL_ID}},
          needle="the ENGINE's zero cell")
    # a `+lora+` identity WITHOUT the engine stamp is torn: named, and read as the lens's — where the native record refuses too
    torn = {**manifest, C.MANIFEST_KEY_ROUTE: {k: v for k, v in route.items() if k != C.SERVED_ENGINE_FIELD}}
    d = fails(man=torn, needle="cannot be bound to an engine")
    assert "another engine's record" in d
    # the native engine stamp beside a LENS identity, and an identity carrying both markers, are torn too
    fails(man={**manifest, C.MANIFEST_KEY_ROUTE: {**route, "serving_identity": LENS_SERVING_IDENTITY}}, needle="a torn record")
    both = M.certification_path_of({"serving_identity": "proxy_vllm+lens+lora+interleaved"})
    assert both[0] is None and both[1][0].startswith("route.serving_identity") and "two steering paths" in both[1][0]
    # the LENS reading: the producer helpers' outputs pass with the identity stated (the real route block's shape) exactly as
    # without it, and the native inapplicability record on a lens route refuses by name
    lens_ev, lens_man, lens_row, lens_fact = _producer_certification_inputs()
    stated = {**lens_man, C.MANIFEST_KEY_ROUTE: {**lens_man[C.MANIFEST_KEY_ROUTE], "serving_identity": LENS_SERVING_IDENTITY,
                                                 C.REDUCTION_MODE_FIELD: "per_row"}}
    for man in (lens_man, stated):
        rec = M.certification_check(lens_ev, man, row=lens_row, fact=lens_fact, family=lens_fact["family"], object_pin=S.object_pin(QWEN))
        assert rec["failures"] == [] and rec["passed"] is True
    d = " ".join(M.certification_check({**lens_ev, S.EVIDENCE_LENS_POSITION: srun._native_lens_position_record()}, stated,
                                       row=lens_row, fact=lens_fact, family=lens_fact["family"], object_pin=S.object_pin(QWEN))["failures"])
    assert "on a lens-served job" in d and "another engine's record" in d
    assert (M.certification_path_of(lens_man[C.MANIFEST_KEY_ROUTE]), M.certification_path_of(stated[C.MANIFEST_KEY_ROUTE]),
            M.certification_path_of(route)) == ((None, []), (M.STEERED_PATH_LENS, []), (M.STEERED_PATH_LORA, []))


PASSTHROUGH_SERVING_IDENTITY = "proxy_vllm+passthrough+interleaved"
_UNSET = object()


@pytest.mark.parametrize("identity,engine,verdict", [
    # a PRESENT stamp outside contracts.SERVING_ENGINES is torn whatever the identity ('typo' / a present
    # null beside a +lens+ identity certified as lens)
    (LENS_SERVING_IDENTITY, "typo", "unknown"), (LENS_SERVING_IDENTITY, None, "unknown"), (LENS_SERVING_IDENTITY, "hf", "unknown"),
    (LORA_SERVING_IDENTITY, "typo", "unknown"), (PASSTHROUGH_SERVING_IDENTITY, "typo", "unknown"),
    # the native path binds to exactly the native stamp (a present null is caught by the rule above, as any unknown value)
    (LORA_SERVING_IDENTITY, C.SERVING_ENGINE_LORA_NATIVE, "lora"),
    (LORA_SERVING_IDENTITY, None, "unknown"), (LORA_SERVING_IDENTITY, _UNSET, "unbound"), (LORA_SERVING_IDENTITY, C.SERVING_ENGINE_LENS, "unbound"),
    # the lens path admits the ABSENT historical stamp or an explicit `lens`; the native stamp beside it is torn
    (LENS_SERVING_IDENTITY, _UNSET, "lens"), (LENS_SERVING_IDENTITY, C.SERVING_ENGINE_LENS, "lens"),
    (LENS_SERVING_IDENTITY, C.SERVING_ENGINE_LORA_NATIVE, "lens-torn"),
    # a passthrough identity is EITHER engine's unsteered leg (the native driver stamps lora_native on its `none` route)
    (PASSTHROUGH_SERVING_IDENTITY, _UNSET, "none"), (PASSTHROUGH_SERVING_IDENTITY, C.SERVING_ENGINE_LENS, "none"),
    (PASSTHROUGH_SERVING_IDENTITY, C.SERVING_ENGINE_LORA_NATIVE, "none"),
    # no identity at all: the historical shape; the native stamp on it is torn (the native writers stamp the identity)
    (None, _UNSET, "none"), (None, C.SERVING_ENGINE_LORA_NATIVE, "noid-torn"),
])
def test_certification_path_validates_every_present_engine_stamp(identity, engine, verdict):
    """`certification_path_of` rejected only the exact native stamp beside a
    non-lora identity, so any OTHER present value — a typo, a present null, `hf` — beside a `+lens+` identity read as the
    lens path with no problem named, and an otherwise-valid store certified. Every PRESENT stamp is now held to
    contracts.SERVING_ENGINES; a lens identity admits the absent historical stamp or an explicit `lens` only; and a
    passthrough identity beside the native stamp is the native engine's own unsteered leg, not a torn record."""
    route = {"virtual_model": "petri__x", "serving_identity": identity}
    if engine is not _UNSET:
        route[C.SERVED_ENGINE_FIELD] = engine
    path, problems = M.certification_path_of(route)
    if verdict == "unknown":
        assert path is None and len(problems) == 1 and f"route.{C.SERVED_ENGINE_FIELD} {engine!r} is not one of {list(C.SERVING_ENGINES)}" in problems[0]
        assert "cannot be bound to an engine" in problems[0]
    elif verdict == "unbound":
        assert path is None and len(problems) == 1 and "names the native path but" in problems[0] and "cannot be bound to an engine" in problems[0]
    elif verdict == "lens-torn":
        assert path == M.STEERED_PATH_LENS and len(problems) == 1 and "a torn record" in problems[0]
    elif verdict == "noid-torn":
        assert path is None and len(problems) == 1 and "naming no serving identity" in problems[0] and "a torn record" in problems[0]
    else:
        assert (path, problems) == ({"lora": M.STEERED_PATH_LORA, "lens": M.STEERED_PATH_LENS, "none": None}[verdict], [])


def test_certification_refuses_a_lens_leg_whose_engine_stamp_is_not_a_known_engine():
    """The same rule end to end on the real lens producer helpers' outputs: a `+lens+` fitted leg stamped
    `serving_engine: "typo"` (or a present null) passed the whole MATH-500 certification before; it fails by name now,
    while the same leg with the stamp absent or `lens` passes exactly as before."""
    lens_ev, lens_man, lens_row, lens_fact = _producer_certification_inputs()
    base_route = {**lens_man[C.MANIFEST_KEY_ROUTE], "serving_identity": LENS_SERVING_IDENTITY, C.REDUCTION_MODE_FIELD: "per_row"}

    def cert(engine):
        route = dict(base_route)
        if engine is not _UNSET:
            route[C.SERVED_ENGINE_FIELD] = engine
        return M.certification_check(lens_ev, {**lens_man, C.MANIFEST_KEY_ROUTE: route}, row=lens_row, fact=lens_fact,
                                     family=lens_fact["family"], object_pin=S.object_pin(QWEN))
    for ok in (_UNSET, C.SERVING_ENGINE_LENS):
        rec = cert(ok)
        assert rec["failures"] == [] and rec["passed"] is True, (ok, rec["failures"])
    for bad in ("typo", None, "hf"):
        rec = cert(bad)
        assert rec["passed"] is False, bad
        assert [f for f in rec["failures"] if f"route.{C.SERVED_ENGINE_FIELD} {bad!r} is not one of" in f], (bad, rec["failures"])
    rec = cert(C.SERVING_ENGINE_LORA_NATIVE)
    assert rec["passed"] is False and [f for f in rec["failures"] if "a torn record" in f]


def _logprob_gated_bites(cell=OBJECT, *, n_text_differing=0, verified=True, **over):
    """The driver's ``steering_bites`` block on a lens leg ruled by the LOGPROB liveness statistic — built by running
    the driver's OWN gate over a scripted client, never a copy of its shape: text bites with ``n_text_differing``
    differing prompts, first-token logprobs 0.5 nats apart against a bit-stable twin (``verified`` True) or
    bit-identical to it (``verified`` False), then edited with ``over``."""
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    import served_capability_run as srun
    from test_lens_liveness_logprob import Chat, _evidence
    chat = Chat(steered_text=("s" if n_text_differing else "w"), twin_text="w", delta=(0.5 if verified else 0.0))
    ev, _err = _evidence(chat, cell_id="alphasteer", arm="alphasteer")
    b = dict(ev[C.OOS_EVIDENCE_BITES_BLOCK])
    b[C.OOS_EVIDENCE_BITES_TWIN] = C.TWIN_CELL_ID
    b[C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM] = C.THINKING_OFF        # the fixture leg renders OFF, as the reader expects
    b["arm"] = cell.split("__")[-1]
    b.update(over)
    return b


def test_certification_accepts_a_logprob_gated_leg_with_inert_text_and_refuses_its_malformations():
    """A leg whose ``steering_bites.liveness_gate`` is ``logprob`` passes with
    ``n_differing`` 0 (the text count is REPORTED, not required) when the ruling block is verified, carries the MEASURED
    floor and the divergence clearing the bar, and the not-a-dose-axis sentence verbatim. Each of those facts missing or
    wrong is refused BY NAME; a text-gated leg keeps the historical ``n_differing >= 1`` rule byte for byte."""
    evidence, manifest, row, fact = _producer_certification_inputs(GEMMA)
    live = _logprob_gated_bites(n_text_differing=0)
    assert live[C.OOS_EVIDENCE_BITES_LIVENESS_GATE] == "logprob" and live["n_differing"] == 0 and "error" not in live
    ok = {**evidence, C.OOS_EVIDENCE_BITES_BLOCK: live}
    rec = M.certification_check(ok, manifest, row=row, fact=fact, family=fact["family"], object_pin=S.object_pin(GEMMA))
    assert rec["failures"] == [] and rec["passed"] is True

    def fails(block, needle):
        d = " ".join(M.certification_check({**evidence, C.OOS_EVIDENCE_BITES_BLOCK: block}, manifest, row=row, fact=fact,
                                           family=fact["family"], object_pin=S.object_pin(GEMMA))["failures"])
        assert needle in d, (needle, d)
        return d

    # the ruling statistic dead: the driver's own failed record (bit-identical logprobs) is refused
    dead = _logprob_gated_bites(n_text_differing=5, verified=False)
    assert dead["verified"] is False and "error" in dead
    fails(dead, "not verified")
    # the sentence missing / altered; the floor not the measured one; the divergence under the bar; the text observation gone
    lp = dict(live[C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS])
    fails({**live, C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS: {**lp, "not_a_dose_axis": "a dose axis"}}, "not-a-dose-axis sentence")
    fails({**live, C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS: {**lp, "noise_floor": {**lp["noise_floor"], "measured": 0.25}}},
          "noise_floor.measured")
    fails({**live, C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS: {**lp, "noise_floor": {**lp["noise_floor"], "binding": "constant"}}},
          "names neither bound")
    fails({**live, C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS: {**lp, "diag": {**lp["diag"], "bar": 9.0}}}, "does not clear the bar")
    fails({**live, C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS: {**lp, "n_conclusive": 2}}, "n_conclusive")
    fails({k: v for k, v in live.items() if k != C.OOS_EVIDENCE_BITES_TEXT_DIVERGENCE}, "text_divergence")
    fails({k: v for k, v in live.items() if k != C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS}, "no logprob_liveness block")
    fails({**live, "object_kind": "psr"}, "not one of")
    fails({**live, C.OOS_EVIDENCE_BITES_LIVENESS_GATE: "rho"}, "liveness_gate")
    # the text gate is untouched: a text-gated block with n_differing 0 still fails the historical rule
    text_gated = S.steering_bites_record(OBJECT, n_differing=0)
    fails(text_gated, "n_differing 0 < 1")
    text_gated2 = {**S.steering_bites_record(OBJECT, n_differing=0), C.OOS_EVIDENCE_BITES_LIVENESS_GATE: "text"}
    fails(text_gated2, "n_differing 0 < 1")


def test_the_probe_inventory_counts_the_logprob_liveness_probes_when_the_block_is_recorded(tmp_path):
    """The driver's per-run inventory reads the logprob probes off the block: one per prompt on the route,
    TWO per prompt on the twin (the scale-0 leg and its repeat), thinking-OFF; without the block nothing is added.
    End to end through the request-evidence gate: a store whose evidence carries the block AND whose attempts carry the
    probes reconciles; the block without the attempts is a shortfall; the attempts without the block are stray."""
    route_vm, twin_vm = S.vmodel_of(OBJECT), C.twin_vmodel()
    base_ev, *_ = _producer_certification_inputs(GEMMA)                 # the producer-shaped leg: bites + lens-ON bite
    inv0, p0 = M.probe_inventory_per_run(base_ev, route_vmodel=route_vm, fitted=True)
    assert p0 is None and not any(k[2] == C.PROBE_KIND_LOGPROB_BITES for k in inv0)
    lp_ev = {**base_ev, C.OOS_EVIDENCE_BITES_BLOCK: _logprob_gated_bites()}
    inv, p = M.probe_inventory_per_run(lp_ev, route_vmodel=route_vm, fitted=True)
    assert p is None
    assert inv == {**inv0, (route_vm, C.THINKING_OFF, C.PROBE_KIND_LOGPROB_BITES): 5, (twin_vm, C.THINKING_OFF, C.PROBE_KIND_LOGPROB_BITES): 10}
    bad = {**base_ev, C.OOS_EVIDENCE_BITES_BLOCK: {**_logprob_gated_bites(), C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS: {"n_probes": 0}}}
    _inv, p = M.probe_inventory_per_run(bad, route_vmodel=route_vm, fitted=True)
    assert p is not None and "logprob_liveness.n_probes" in p
    # the no-switch class takes the same count through contracts.no_switch_probe_inventory
    inv_ns, p_ns = M.probe_inventory_per_run(lp_ev, route_vmodel=route_vm, fitted=True, cert_class=C.CERT_CLASS_NO_SWITCH)
    assert p_ns is None and inv_ns == C.no_switch_probe_inventory(route_vm, fitted=True, n_probes=5, n_logprob_probes=5)

    def with_block(ev):
        ev[C.OOS_EVIDENCE_BITES_BLOCK] = _logprob_gated_bites()

    def with_probes(attempts):
        for target, legs in ((route_vm, 1), (twin_vm, 2)):
            n = 0
            for i in range(5):
                for _ in range(legs):
                    attempts.append(S.probe_attempt(target, kind=C.PROBE_KIND_LOGPROB_BITES, n=n, arm=C.THINKING_OFF, lens=True,
                                                    prompt=f"probe prompt {i}", max_tokens=C.LOGPROB_BITES_MAX_TOKENS))
                    n += 1

    _rep, g, rec = _req_gate(tmp_path, "lpboth", evidence_edit=with_block, attempts_edit=with_probes)
    assert g["status"] != M.GATE_FAIL or "logprob_bites" not in g["detail"], g["detail"]
    assert rec["probe_inventory"][f"{route_vm}/{C.THINKING_OFF}/{C.PROBE_KIND_LOGPROB_BITES}"] == {"expected_per_run": 5, "records": 5}
    assert rec["probe_inventory"][f"{twin_vm}/{C.THINKING_OFF}/{C.PROBE_KIND_LOGPROB_BITES}"] == {"expected_per_run": 10, "records": 10}
    _rep, g, _rec = _req_gate(tmp_path, "lpblockonly", evidence_edit=with_block)
    assert g["status"] == M.GATE_FAIL and "shortfall" in g["detail"] and "logprob_bites" in g["detail"]
    _rep, g, _rec = _req_gate(tmp_path, "lpprobesonly", attempts_edit=with_probes)
    assert g["status"] == M.GATE_FAIL and "never sends" in g["detail"] and "logprob_bites" in g["detail"]


def test_reader_record_field_lists_pin_the_producer_helpers():
    """The reader's field lists for the three helper records equal the keys the
    helpers actually return; the request-evidence fields are the contracts'."""
    evidence, _m, _row, _fact = _producer_certification_inputs()
    # the core keys the reader REQUIRES, then the additive keys — key for key with the producer
    assert tuple(evidence[S.EVIDENCE_RENDER_CERTIFIED]) == M.RENDER_CERTIFIED_FIELDS + M.RENDER_CERTIFIED_D105_FIELDS
    assert tuple(evidence[S.EVIDENCE_LENS_POSITION]) == M.LENS_POSITION_ATTESTED_FIELDS
    assert tuple(evidence[S.EVIDENCE_LENS_ON_BITE]) == M.LENS_ON_BITE_FIELDS + M.LENS_ON_BITE_DRIVER_FIELDS
    assert M.REQUEST_EVIDENCE_FIELDS == C.REQUEST_EVIDENCE_FIELDS and "kind" in M.REQUEST_EVIDENCE_FIELDS
    assert M.REQUEST_EVIDENCE_VERSION == C.REQUEST_EVIDENCE_VERSION
    assert M.RENDER_SOURCE_SERVED == evidence[S.EVIDENCE_RENDER_CERTIFIED]["render_source"]
    assert M.EVIDENCE_LENS_ON_BITE == C.LENS_ON_BITE_KEY and M.ROUTE_KEY_LENS_SERVED_POSITIONS == C.LENS_SERVED_POSITIONS_KEY
    assert M.MANIFEST_KEY_TRANSFER_PROVENANCE == C.MANIFEST_KEY_TRANSFER_PROVENANCE
    assert M.MANIFEST_KEY_OOS_EXPERIMENT == C.MANIFEST_KEY_AUTHORISED_EXPERIMENT
    assert M.MANIFEST_KEY_RESUME_ATTEMPTS == C.MANIFEST_KEY_RESUME_ATTEMPTS
    assert M.HEADER_CHECK_CLASS_EXPECTED == C.HEADER_CHECK_DOCUMENT_FIT_INAPPLICABLE
    assert M.RENDER_FACT_CERTIFIED == C.RENDER_EVIDENCE_CERTIFIED


def test_sampling_ref_is_the_contracts_digest_not_default_json(tmp_path):
    """A row stamped by the producer (``contracts.sampling_ref``: compact
    separators) passes; the reader's expected digest IS the contracts' digest,
    and the default-whitespace digest the reader used to compute is a different
    string that must be refused."""
    import hashlib
    row = S.SAMPLING_ROWS[PRIMARY][QWEN]
    block = S.sampling_block_of(QWEN, PRIMARY, row)
    want = C.sampling_ref(block)
    loose = hashlib.sha256(json.dumps(dict(block), sort_keys=True).encode()).hexdigest()[:12]
    assert loose != want
    manifest = {C.MANIFEST_KEY_SAMPLING: block}
    rows = [{"_extra": {C.SAMPLING_REF_FIELD: want}} for _ in range(3)]
    rec = M.sampling_ref_check(rows, manifest)
    assert rec["status"] == M.GATE_PASS and rec["expected"] == want
    rec = M.sampling_ref_check([{"_extra": {C.SAMPLING_REF_FIELD: loose}}], manifest)
    assert rec["status"] == M.GATE_FAIL and "sampling_ref" in rec["detail"]


# --------------------------------------------------------------------------- #
# the wave pin is CHECKED — per leg, per outcomes record, per resume          #
# attempt, and across a comparison's legs — before any estimand               #
# --------------------------------------------------------------------------- #


def test_wave_pin_refuses_wrong_pin_torn_outcomes_and_mixed_producers(tmp_path):
    rows = primary_rows(QWEN)
    S.build_all(tmp_path / "gen", rows=rows)
    common = {"cells": rows, "sampling_rows": S.SAMPLING_ROWS, "render_facts": S.RENDER_FACTS, "math500_ids": S.IDS,
              "n_resamples": B_FAST, "read_eval_logs": False, "analysis_sha": S.FULL_SHA}
    # a WRONG pin refuses every leg before any estimand, naming the pin and the producer sha
    rep = M.analyze(tmp_path / "gen", "abcdef0", **common)
    assert len(rep["unusable_legs"]) == 3
    assert all("abcdef0" in why and S.FULL_SHA in why and "producer" in why for why in rep["unusable_legs"].values())
    assert all(c["estimands"] is None for c in rep["comparisons"]) and "estimands" not in rep
    assert rep["gates"]["population"]["status"] == M.GATE_FAIL and M.production_refusals(rep)
    # the full 40-hex pin is accepted like the 7-char one; a non-hex / too-short pin is an operator error
    assert M.analyze(tmp_path / "gen", S.FULL_SHA, **common)["unusable_legs"] == {}
    for bad in ("xyz", "12345", "ABCDEF0", S.FULL_SHA + "0"):
        with pytest.raises(M.AnalysisError, match="pin"):
            M.analyze(tmp_path / "gen", bad, **common)
    # the outcomes record's own producer sha must equal the manifest's
    S.build_all(tmp_path / "torn", rows=rows, overrides={(QWEN, OBJECT, PRIMARY): {"outcomes_tree_sha": "1" * 40}})
    why = leg_of(analyze(tmp_path / "torn", rows=rows), QWEN, OBJECT)["refused"]
    assert "producer_tree_sha" in why and "1" * 40 in why and S.FULL_SHA in why
    # a manifest without the producer block, or from another driver, is refused by name
    def no_producer(m):
        del m[C.MANIFEST_KEY_PRODUCER]

    S.build_all(tmp_path / "nop", rows=rows, overrides={(QWEN, BARE, PRIMARY): {"manifest_edit": no_producer}})
    assert "producer" in leg_of(analyze(tmp_path / "nop", rows=rows), QWEN, BARE)["refused"]

    def other_driver(m):
        m[C.MANIFEST_KEY_PRODUCER]["driver"] = "tinycap_sandbox"

    S.build_all(tmp_path / "drv", rows=rows, overrides={(QWEN, BARE, PRIMARY): {"manifest_edit": other_driver}})
    assert "driver" in leg_of(analyze(tmp_path / "drv", rows=rows), QWEN, BARE)["refused"]
    # every recorded run of the store must be from the pinned tree: a resume from another tree is a mixed-producer LEG
    mixed_runs = [{"ts": "t1", "tree_sha": "2" * 40, "attempt": 1, "completed": True},
                  {"ts": "t2", "tree_sha": S.FULL_SHA, "attempt": 2, "completed": True}]
    S.build_all(tmp_path / "mleg", rows=rows, overrides={(QWEN, OBJECT, PRIMARY): {"resume_attempts": mixed_runs}})
    why = leg_of(analyze(tmp_path / "mleg", rows=rows), QWEN, OBJECT)["refused"]
    assert "resume_attempts" in why and "2" * 40 in why
    # ... and the record itself is required, one entry per run, numbered from 1
    S.build_all(tmp_path / "nores", rows=rows, overrides={(QWEN, OBJECT, PRIMARY): {"resume_attempts": []}})
    assert "resume_attempts" in leg_of(analyze(tmp_path / "nores", rows=rows), QWEN, OBJECT)["refused"]
    S.build_all(tmp_path / "num", rows=rows,
                overrides={(QWEN, OBJECT, PRIMARY): {"resume_attempts": [{"ts": "t", "tree_sha": S.FULL_SHA, "attempt": 2,
                                                                          "completed": True}]}})
    assert "attempt" in leg_of(analyze(tmp_path / "num", rows=rows), QWEN, OBJECT)["refused"]
    # three legs that EACH match the 7-char pin but come from two trees: the comparison is mixed-producer — no estimand,
    # the population gate FAILS naming it, no word
    other = S.SHA7 + "f" * 33
    S.build_all(tmp_path / "mixed", rows=rows, overrides={(QWEN, OBJECT, PRIMARY): {"tree_sha": other}})
    rep = analyze(tmp_path / "mixed", rows=rows)
    assert rep["unusable_legs"] == {}
    c = comparison(rep, QWEN)
    assert c["usable"] is False and c["estimands"] is None and "producer" in c["reason"] and other in c["reason"]
    assert rep["gates"]["population"]["status"] == M.GATE_FAIL and "producer" in rep["gates"]["population"]["detail"]
    assert rep["verdict"]["word"] is None and M.production_refusals(rep)
    assert leg_of(rep, QWEN, OBJECT)["producer"]["tree_sha"] == other and leg_of(rep, QWEN, OBJECT)["producer"]["pin"] == S.SHA7
    # a sensitivity row from another tree pairs with NO primary leg: its own comparison and its mode delta both stand down
    rows2 = rows_for(QWEN, PRIMARY, GENERAL)
    S.build_all(tmp_path / "mixed2", rows=rows2, overrides={(QWEN, BARE, GENERAL): {"tree_sha": other}})
    rep = analyze(tmp_path / "mixed2", rows=rows2)
    assert comparison(rep, QWEN, GENERAL)["usable"] is False and comparison(rep, QWEN, PRIMARY)["usable"] is True
    delta = next(r for r in rep["mode_delta_records"] if r["mode"] == GENERAL)
    assert delta["usable"] is False and "producer" in delta["reason"]
    # the provenance table carries the pin and every leg's checked producer record
    assert rep["provenance"]["store_pin"] == S.SHA7
    assert rep["provenance"]["cells"][M.leg_key(QWEN, BARE, GENERAL)]["producer_tree_sha"] == other


# --------------------------------------------------------------------------- #
# request evidence is reconciled against EVERY attempted call — task          #
# calls per (task, sample) against the ledger's attempts, probes against the  #
# driver's per-run inventory times the recorded runs                          #
# --------------------------------------------------------------------------- #


def _req_gate(tmp_path, name, model=NEMO, cell=OBJECT, **kw):
    rep = _one_leg_report(tmp_path, name, model, cell, overrides=kw)
    return rep, rep["gates"]["identity"], leg_of(rep, model, cell)["request_evidence"]


def test_request_evidence_probes_are_reconciled_not_ignored(tmp_path):
    """Under identity version 2 the probes travel THROUGH the ledger, labelled
    ``probe/<kind>`` with sample ``<vmodel>/<n>`` (FC4 (2)), and the evidence is
    reconciled against the ledger's attempts BY REQUEST ID through the producer's
    own rule (``contracts.request_evidence_reconciliation``): deleting every probe
    record (or one) leaves ledger attempts with NO record; a probe PAIR to a route
    / render / kind the driver never sends is off the driver's inventory; a probe
    record re-labelled as a task call disagrees with its ledger row; a pair whose
    sample is not ``<vmodel>/<n>`` is not the wrapper's; the inventory still
    needs the bites block's ``n_probes``; an untouched store reaches every slot
    exactly once per completed run."""
    n_probe = len(S.probe_records_one_attempt(OBJECT, fitted=True))

    def no_probes(recs):
        recs[:] = [r for r in recs if r["kind"] != C.REQUEST_EVIDENCE_KIND_PROBE]

    _rep, g, rec = _req_gate(tmp_path, "noprobes", request_evidence_edit=no_probes)
    assert g["status"] == M.GATE_FAIL and "probe/" in g["detail"] and "NO request-evidence record" in g["detail"]
    assert rec["n_probe_records"] == 0 and rec["n_ledger_rows"] == 500 + n_probe

    def one_probe_gone(recs):
        recs.pop(next(i for i, r in enumerate(recs) if r["kind"] == "probe" and r["vmodel"] == C.twin_vmodel()))

    _rep, g, _rec = _req_gate(tmp_path, "oneprobe", request_evidence_edit=one_probe_gone)
    assert g["status"] == M.GATE_FAIL and C.twin_vmodel() in g["detail"] and "shortfall" in g["detail"]
    # a bare cell has its two render probes too
    _rep, g, _rec = _req_gate(tmp_path, "barenp", cell=BARE, request_evidence_edit=no_probes)
    assert g["status"] == M.GATE_FAIL and C.probe_task_label(C.PROBE_KIND_RENDER) in g["detail"]

    def stray_pair(attempts):       # a record AND its ledger row: the identity holds, the inventory refuses
        attempts.append(S.probe_attempt("petri__somebody_else", kind=C.PROBE_KIND_RENDER, n=0, arm=C.THINKING_OFF, lens=False))

    _rep, g, _rec = _req_gate(tmp_path, "stray", attempts_edit=stray_pair)
    assert g["status"] == M.GATE_FAIL and "petri__somebody_else" in g["detail"] and "never sends" in g["detail"]

    def probe_as_task(recs):        # the record says a task call, the ledger row says probe/bites
        recs[0]["task"], recs[0]["sample"] = S.TASK, "math_00000000/0"

    _rep, g, _rec = _req_gate(tmp_path, "psample", request_evidence_edit=probe_as_task)
    assert g["status"] == M.GATE_FAIL and "ledger labels the attempt" in g["detail"] and "probe/" in g["detail"]

    def bad_label(attempts):        # a consistent pair whose sample the wrapper never writes
        a = S.probe_attempt(S.vmodel_of(OBJECT), kind=C.PROBE_KIND_RENDER, n=0, arm=C.THINKING_OFF, lens=True)
        a.record["sample"] = a.row["sample"] = "math_00000000/0"
        attempts.append(a)

    _rep, g, _rec = _req_gate(tmp_path, "blabel", attempts_edit=bad_label)
    assert g["status"] == M.GATE_FAIL and "<vmodel>/<n>" in g["detail"]
    # the probe inventory needs the bites block's n_probes: an unreadable count refuses the reconciliation by name
    def no_n_probes(ev):
        del ev["steering_bites"]["n_probes"]

    _rep, g, _rec = _req_gate(tmp_path, "nonp", evidence_edit=no_n_probes)
    assert g["status"] == M.GATE_FAIL and "n_probes" in g["detail"]
    # an untouched store: every slot exactly at its per-run count, nothing unattributed, one completed run
    _rep, g, rec = _req_gate(tmp_path, "ok")
    assert g["status"] == M.GATE_PASS and rec["passed"] and rec["n_probe_records_unattributed"] == 0
    assert rec["n_runs"] == rec["n_completed_runs"] == 1 and rec["n_incomplete_runs"] == 0 and rec["n_probe_records"] == n_probe
    vm, k, base = S.vmodel_of(OBJECT), S.N_BITES_PROMPTS, S.vmodel_of(BARE)
    render, bites, bite = C.PROBE_KIND_RENDER, C.PROBE_KIND_BITES, C.PROBE_KIND_LENS_ON_BITE
    assert rec["probe_inventory"] == {
        f"{vm}/{C.THINKING_OFF}/{render}": {"expected_per_run": 1, "records": 1},
        f"{vm}/{C.THINKING_ON}/{render}": {"expected_per_run": 1, "records": 1},
        f"{vm}/{C.THINKING_OFF}/{bites}": {"expected_per_run": k, "records": k},
        f"{C.twin_vmodel()}/{C.THINKING_OFF}/{bites}": {"expected_per_run": k, "records": k},
        f"{vm}/{C.THINKING_ON}/{bite}": {"expected_per_run": 1, "records": 1},
        f"{base}/{C.THINKING_ON}/{bite}": {"expected_per_run": 1, "records": 1}}
    _rep, _g, rec = _req_gate(tmp_path, "okbare", cell=BARE)
    assert rec["passed"] and rec["probe_inventory"] == {f"{base}/{C.THINKING_OFF}/{render}": {"expected_per_run": 1, "records": 1},
                                                        f"{base}/{C.THINKING_ON}/{render}": {"expected_per_run": 1, "records": 1}}


def test_request_evidence_task_records_are_reconciled_per_attempt_against_the_ledger(tmp_path):
    """One record per ATTEMPTED call, bound to its ledger row by request id:
    a real resume (a second attempt — a record and a ledger row with a fresh
    id — per re-sent sample, a second run of probes, two recorded runs)
    PASSES; a run whose preflight refused is on record
    ``completed: false`` with its probes in both files (attributable, a
    finding); a store whose LAST run never completed is refused by name; probe
    attempts beyond the recorded runs are counted, a record WITHOUT its attempt
    is refused; deleting records leaves ledger attempts with NO record; a
    duplicated record is one attempt with two records; a record whose labels or
    digest disagree with its row, a wire cap other than the store's, a ledger from
    before the identity columns, and a missing ledger are each named."""
    rows = primary_rows(GEMMA)
    n_probe = len(S.probe_records_one_attempt(OBJECT, fitted=True))
    S.build_all(tmp_path / "resume", rows=rows, overrides={(GEMMA, OBJECT, PRIMARY): {"resume_rows": 4}})
    rep = analyze(tmp_path / "resume", rows=rows)
    assert rep["gates"]["identity"]["status"] == M.GATE_PASS and rep["gates"]["ledger"]["status"] == M.GATE_PASS
    leg = leg_of(rep, GEMMA, OBJECT)
    rec = leg["request_evidence"]
    assert rec["n_task_records"] == 504 and rec["n_task_attempts_ledger"] == 504 and rec["n_resent_samples"] == 4
    assert rec["n_probe_records"] == 2 * n_probe and rec["n_ledger_rows"] == 504 + 2 * n_probe
    assert rec["n_runs"] == rec["n_completed_runs"] == 2 and rec["n_probe_records_unattributed"] == 0
    assert leg["ledger"]["n_probe_rows"] == 2 * n_probe and leg["producer"]["n_incomplete_runs"] == 0
    assert M.production_refusals(rep) == []
    # a run whose preflight refused: on record (completed false) with its probes in BOTH files — attributable, a finding
    S.build_all(tmp_path / "preflight", rows=rows,
                overrides={(GEMMA, OBJECT, PRIMARY): {"resumes": 2, "resume_rows": 4, "incomplete_runs": [1]}})
    rep = analyze(tmp_path / "preflight", rows=rows)
    rec = leg_of(rep, GEMMA, OBJECT)["request_evidence"]
    assert rec["passed"] and rec["n_runs"] == 3 and rec["n_completed_runs"] == 2 and rec["n_incomplete_runs"] == 1
    assert rec["n_probe_records"] == 3 * n_probe and rec["n_probe_records_unattributed"] == 0
    assert any("did not complete" in f for f in rep["findings"]) and M.production_refusals(rep) == []
    # ... but a store whose LAST run never completed is not a passed run: refused by name
    S.build_all(tmp_path / "lastbad", rows=rows, overrides={(GEMMA, OBJECT, PRIMARY): {"resumes": 1, "incomplete_runs": [1]}})
    why = leg_of(analyze(tmp_path / "lastbad", rows=rows), GEMMA, OBJECT)["refused"]
    assert why and "did not complete" in why and "resume_attempts" in why
    S.build_all(tmp_path / "nobool", rows=rows, overrides={(GEMMA, OBJECT, PRIMARY): {
        "resume_attempts": [{"ts": "t", "tree_sha": S.FULL_SHA, "attempt": 1, "completed": "yes"}]}})
    assert "completed" in leg_of(analyze(tmp_path / "nobool", rows=rows), GEMMA, OBJECT)["refused"]
    # a probe ATTEMPT (record + row) beyond what the recorded runs account for: counted, a finding, not a failure ...
    def extra_pair(attempts):
        attempts.append(S.probe_attempt(S.vmodel_of(OBJECT), kind=C.PROBE_KIND_RENDER, n=5, arm=C.THINKING_ON, lens=True,
                                        ts="2026-01-01T05:00:00Z"))

    S.build_all(tmp_path / "extra_pair", rows=rows, overrides={(GEMMA, OBJECT, PRIMARY): {"attempts_edit": extra_pair}})
    rep = analyze(tmp_path / "extra_pair", rows=rows)
    rec = leg_of(rep, GEMMA, OBJECT)["request_evidence"]
    assert rec["passed"] and rec["n_probe_records_unattributed"] == 1 and any("beyond what the recorded runs" in f for f in rep["findings"])
    # ... while a probe RECORD without its ledger attempt is a record for a call the ledger never received
    S.build_all(tmp_path / "orphan", rows=rows, overrides={(GEMMA, OBJECT, PRIMARY): {
        "request_evidence_edit": lambda recs: recs.extend(S.probe_records_one_attempt(OBJECT, fitted=True, ts="2026-01-01T05:00:00Z"))}})
    g = analyze(tmp_path / "orphan", rows=rows)["gates"]["identity"]
    assert g["status"] == M.GATE_FAIL and "the ledger never received" in g["detail"]

    def drop_second(recs):
        seen, out = set(), []
        for r in recs:
            if r["kind"] == "task":
                if r["sample"] in seen:
                    continue
                seen.add(r["sample"])
            out.append(r)
        recs[:] = out

    S.build_all(tmp_path / "short", rows=rows,
                overrides={(GEMMA, OBJECT, PRIMARY): {"resume_rows": 4, "request_evidence_edit": drop_second}})
    g = analyze(tmp_path / "short", rows=rows)["gates"]["identity"]
    assert g["status"] == M.GATE_FAIL and "4 attempted call(s) in the ledger have NO request-evidence record" in g["detail"]

    def extra_record(recs):
        recs.append(dict(recs[_first_task_record(recs)]))

    S.build_all(tmp_path / "extra", rows=rows, overrides={(GEMMA, OBJECT, PRIMARY): {"request_evidence_edit": extra_record}})
    g = analyze(tmp_path / "extra", rows=rows)["gates"]["identity"]
    assert g["status"] == M.GATE_FAIL and "two request-evidence records" in g["detail"]

    def other_task(recs):
        recs[_first_task_record(recs)]["task"] = "inspect/gsm8k"

    S.build_all(tmp_path / "otask", rows=rows, overrides={(GEMMA, OBJECT, PRIMARY): {"request_evidence_edit": other_task}})
    d = analyze(tmp_path / "otask", rows=rows)["gates"]["identity"]["detail"]
    assert "inspect/gsm8k" in d and "ledger labels the attempt" in d

    def other_vmodel(recs):
        recs[_first_task_record(recs)]["vmodel"] = S.vmodel_of(BARE)

    S.build_all(tmp_path / "ovm", rows=rows, overrides={(GEMMA, OBJECT, PRIMARY): {"request_evidence_edit": other_vmodel}})
    assert "vmodel" in analyze(tmp_path / "ovm", rows=rows)["gates"]["identity"]["detail"]

    def no_kind(recs):
        del recs[0]["kind"]

    S.build_all(tmp_path / "nokind", rows=rows, overrides={(GEMMA, OBJECT, PRIMARY): {"request_evidence_edit": no_kind}})
    assert "kind" in analyze(tmp_path / "nokind", rows=rows)["gates"]["identity"]["detail"]

    def bad_digest(recs):
        recs[0]["body_sha256"] = "nope"

    S.build_all(tmp_path / "digest", rows=rows, overrides={(GEMMA, OBJECT, PRIMARY): {"request_evidence_edit": bad_digest}})
    assert "body_sha256" in analyze(tmp_path / "digest", rows=rows)["gates"]["identity"]["detail"]

    def other_digest(recs):         # a 64-hex digest that is not the ledger's: the two records disagree on what was sent
        recs[_first_task_record(recs)]["body_sha256"] = "0" * 64

    S.build_all(tmp_path / "odigest", rows=rows, overrides={(GEMMA, OBJECT, PRIMARY): {"request_evidence_edit": other_digest}})
    assert "disagree on what was sent" in analyze(tmp_path / "odigest", rows=rows)["gates"]["identity"]["detail"]

    def wrong_cap(recs):
        recs[_first_task_record(recs)]["max_tokens"] = 4096

    S.build_all(tmp_path / "cap", rows=rows, overrides={(GEMMA, OBJECT, PRIMARY): {"request_evidence_edit": wrong_cap}})
    assert "max_tokens" in analyze(tmp_path / "cap", rows=rows)["gates"]["identity"]["detail"]
    # a ledger from before the identity columns cannot be reconciled
    def bare_rows(rs):
        for r in rs:
            r["request_id"], r["body_sha256"] = None, None

    S.build_all(tmp_path / "bare", rows=rows, overrides={(GEMMA, OBJECT, PRIMARY): {"ledger_edit": bare_rows}})
    d = analyze(tmp_path / "bare", rows=rows)["gates"]["identity"]["detail"]
    assert "request_id" in d and "cannot be reconciled" in d
    # no ledger: the attempts cannot be reconciled — the evidence check refuses by name, beside the ledger gate
    S.build_all(tmp_path / "noledger", rows=rows, overrides={(GEMMA, OBJECT, PRIMARY): {"ledger": False}})
    rep = analyze(tmp_path / "noledger", rows=rows)
    assert rep["gates"]["identity"]["status"] == M.GATE_FAIL and "cannot be reconciled" in rep["gates"]["identity"]["detail"]
    # the manifest's request_evidence path must name the file the reader reads
    def other_path(m):
        m[C.MANIFEST_KEY_REQUEST_EVIDENCE] = "/elsewhere/other.jsonl"

    S.build_all(tmp_path / "path", rows=rows, overrides={(GEMMA, OBJECT, PRIMARY): {"manifest_edit": other_path}})
    assert "other.jsonl" in analyze(tmp_path / "path", rows=rows)["gates"]["identity"]["detail"]


# --------------------------------------------------------------------------- #
# the outcome record is the PRODUCER's — `outcome_record()` with the          #
# identity-v2 `sampling_ref` stamp, through the real `finish_task()`, read by #
# the analysis                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(not S.STAMPED_OUTCOME_ITEMS_ADMITTED, strict=True, raises=CompletionError,
                   reason="pre-harness-fix tree: `outcome_record()` stamps `sampling_ref` on every item of an "
                          "identity-v2 run, no admitted contracts item shape carries it "
                          f"({S.OUTCOME_ITEM_FIELD_SETS_NAME}), and `finish_task()`'s exact validator refuses the "
                          "producer's own record after generation (D57 (1)); the harness fix admits the v2 shape "
                          "and this test then MUST pass")
def test_a_stamped_v2_outcome_record_completes_and_is_read(tmp_path):
    """The REAL ``outcome_record()`` with a non-null digest, through the
    REAL ``finish_task()`` (summary, outcomes file, done marker), read by the
    analysis: every item carries the manifest block's digest and the identity
    gate holds it to that digest — a record stamped with ANOTHER block's digest
    is a named failure. The old fixture assembled its items by hand without the
    stamp, so the writer / contracts mismatch that kills every v2 run after
    generation never reached a test."""
    rows = primary_rows(QWEN)
    S.build_all(tmp_path / "gen", rows=rows, outcome_sampling_ref=S.OUTCOME_STAMP_ALWAYS)
    rep = analyze(tmp_path / "gen", rows=rows)
    assert rep["unusable_legs"] == {}
    want = S.expected_sampling_ref(QWEN, PRIMARY)
    for cell in (BARE, PROMPTED, OBJECT):
        leg = leg_of(rep, QWEN, cell)
        assert leg["sampling_ref"]["expected"] == want and leg["sampling_ref"]["status"] == M.GATE_PASS
        assert leg["sampling_ref"]["n_items_stamped"] == 500
        assert all(it[C.SAMPLING_REF_FIELD] == want for it in leg["items"].values())
        # the record on disk IS the producer's shape, item for item
        cd = Path(leg["cell_dir"])
        doc = json.loads((cd / C.TASK_OUTCOMES_DIR / C.task_outcomes_filename(M.TASK)).read_text())
        assert all(C.SAMPLING_REF_FIELD in it for it in doc["items"])
    assert rep["gates"]["identity"]["status"] == M.GATE_PASS and M.production_refusals(rep) == []
    assert rep["verdict"]["word"] is not None
    # the stamped items are READ: a record stamped with another block's digest fails the identity gate by name
    S.build_all(tmp_path / "wrong", rows=rows, overrides={(QWEN, OBJECT, PRIMARY): {"outcome_sampling_ref": "0" * 12}})
    rep = analyze(tmp_path / "wrong", rows=rows)
    g = rep["gates"]["identity"]
    assert g["status"] == M.GATE_FAIL and "outcome item" in g["detail"] and "0" * 12 in g["detail"] and want in g["detail"]
    assert M.production_refusals(rep) and rep["verdict"]["word"] is None


def test_the_fixtures_outcome_record_is_the_producers_own():
    """The synthetic store's record is built THROUGH ``inspect_served.outcome_record``
    (never assembled by hand): the module names the seam it keys on, and the
    default stamp follows what THIS tree's writer admits."""
    assert S.OUTCOME_ITEM_FIELD_SETS and all(C.SAMPLING_REF_FIELD not in fs or fs[-1] == C.SAMPLING_REF_FIELD
                                             for fs in S.OUTCOME_ITEM_FIELD_SETS)
    assert S.STAMPED_OUTCOME_ITEMS_ADMITTED == any(C.SAMPLING_REF_FIELD in fs for fs in S.OUTCOME_ITEM_FIELD_SETS)
    assert S.default_outcome_stamp(QWEN, PRIMARY) == (S.expected_sampling_ref(QWEN, PRIMARY)
                                                      if S.STAMPED_OUTCOME_ITEMS_ADMITTED else None)
    assert len(S.expected_sampling_ref(QWEN, PRIMARY)) == 12


# --------------------------------------------------------------------------- #
# the .eval re-derivation reads the log the PRODUCER would select             #
# --------------------------------------------------------------------------- #


def test_the_eval_rederivation_selects_the_completed_log_beside_an_abandoned_one(tmp_path):
    """An interrupted evaluation leaves a cancelled ``.eval`` beside the
    resume's successful complete log. The producer selects the completed log of
    the plan's population (``inspect_served.reusable_log``: status success, the
    attempted count, no errored sample, the attempted ids as a multiset); the
    inherited reader read EVERY matching log, ``results_from_log`` raised on the
    cancelled one, the cross-check went unavailable with ``log_present`` true and
    the scoring-path gate turned a legitimate resume into an unwaivable pending.
    The analysis now selects with the producer's rule before the per-item
    cross-check; with no qualifying log among present ones it fails closed
    (pending, never waivable) naming what disqualified each candidate; a
    selected log that disagrees with the record still refuses the leg. Both
    logs are written through the real Inspect log writer."""
    pytest.importorskip("inspect_ai")
    rows = primary_rows(QWEN)
    dirs = S.build_all(tmp_path / "gen", rows=rows, overrides={(QWEN, OBJECT, PRIMARY): {"resume_rows": 3}})
    cd = dirs[f"{QWEN}/{OBJECT}/{PRIMARY}"]
    t0 = 1_700_000_000
    abandoned = S.write_inspect_log(cd, name="2026-01-01T00-00-00", n_samples=120, cancelled=True, mtime=t0)
    good = S.write_inspect_log(cd, name="2026-01-01T01-00-00", mtime=t0 + 3600)
    for cell in (BARE, PROMPTED):
        S.write_inspect_log(dirs[f"{QWEN}/{cell}/{PRIMARY}"], name="2026-01-01T00-00-00")
    rep = analyze(tmp_path / "gen", rows=rows, read_eval_logs=True)
    leg = leg_of(rep, QWEN, OBJECT)
    ec = leg["eval_check"]
    assert leg["refused"] is None, leg["refused"]
    assert ec["available"] is True and ec["agrees"] is True and ec["log_present"] is True
    assert ec["selected_logs"] == [good.name]
    assert ec["derived"]["n_attempted_samples"] == 500 and ec["derived"]["n_unscored_samples"] == 0
    assert ec["per_item"]["n_compared"] == 500 and ec["per_item"]["n_mismatched"] == 0
    assert ec["text"]["n_completions"] == 500 and ec["text"]["n_empty"] == 0
    for cell in (BARE, PROMPTED):
        assert leg_of(rep, QWEN, cell)["eval_check"]["available"] is True
    assert rep["gates"]["scoring_path"]["status"] == M.GATE_PASS and rep["gates"]["pending"] == []
    assert M.production_refusals(rep) == [] and rep["verdict"]["word"] is not None
    # the run that reads the logs and the run that does not agree on every numeric leaf (the two-venv shape)
    assert M.numeric_leaf_diff(rep, analyze(tmp_path / "gen", rows=rows, read_eval_logs=False))["equal"]
    # only the abandoned log survives: no qualifying log -> fail closed, the cancelled log named with its status
    good.unlink()
    rep = analyze(tmp_path / "gen", rows=rows, read_eval_logs=True)
    ec = leg_of(rep, QWEN, OBJECT)["eval_check"]
    assert ec["available"] is False and ec["log_present"] is True and ec["selected_logs"] == []
    assert abandoned.name in ec["reason"] and "cancelled" in ec["reason"] and "reusable_log" in ec["reason"]
    g = rep["gates"]["scoring_path"]
    assert g["status"] == M.GATE_PENDING and any(abandoned.name in p for p in g["pending"])
    assert rep["gates"]["pending_unwaivable"] and rep["verdict"]["word"] is None
    assert any("not waivable" in r for r in M.production_refusals(rep))
    assert analyze(tmp_path / "gen", rows=rows, read_eval_logs=True, allow_pending_gates=True)["verdict"]["word"] is None
    # a successful log over the WRONG population (499 draws) beside it is not this task's evidence either
    short = S.write_inspect_log(cd, name="2026-01-01T02-00-00", n_samples=499, mtime=t0 + 7200)
    ec = leg_of(analyze(tmp_path / "gen", rows=rows, read_eval_logs=True), QWEN, OBJECT)["eval_check"]
    assert ec["available"] is False and short.name in ec["reason"] and "499" in ec["reason"] and "500" in ec["reason"]
    short.unlink()
    # ... nor a complete successful log over ANOTHER selection (one id substituted)
    items = completion.read_task_outcomes(cd, M.TASK)["items"]
    swapped = [{**it, "item_id": "math_ffffffff"} if i == 7 else it for i, it in enumerate(items)]
    other = S.write_inspect_log(cd, name="2026-01-01T03-00-00", items=swapped, mtime=t0 + 9000)
    ec = leg_of(analyze(tmp_path / "gen", rows=rows, read_eval_logs=True), QWEN, OBJECT)["eval_check"]
    assert ec["available"] is False and other.name in ec["reason"] and "selection" in ec["reason"]
    other.unlink()
    # the SELECTED log disagrees with the record item by item: the check ran and the leg is refused
    S.write_inspect_log(cd, name="2026-01-01T04-00-00", mtime=t0 + 12000,
                        value_edit=lambda item_id, v: (1.0 - v) if item_id == S.IDS[0] else v)
    rep = analyze(tmp_path / "gen", rows=rows, read_eval_logs=True)
    leg = leg_of(rep, QWEN, OBJECT)
    assert leg["eval_check"]["available"] is True and leg["eval_check"]["agrees"] is False
    assert leg["refused"] and "disagrees" in leg["refused"] and S.IDS[0] in leg["refused"]
    assert M.production_refusals(rep)


# --------------------------------------------------------------------------- #
# the selection holds every retained log to the cap                            #
# --------------------------------------------------------------------------- #


def test_a_retained_log_at_another_cap_or_recording_none_is_not_this_experiments_evidence(tmp_path):
    """The analysis' selection called ``reusable_log`` and
    ``retained_log_reason`` WITHOUT ``expected_max_tokens``, so a retained or
    copied-in ``.eval`` with the right 500 ids and a success status but generated
    at 4,096 (or recording no cap) was SELECTED here while the producer's
    ``run_one_task`` (``expected_max_tokens=plan.max_tokens``) rejects it —
    a false "independent cross-check agrees". The fixture masked it: its logs
    carried a bare plan with no ``config.max_tokens``. Now the fixture writes the
    plan Inspect records (the eval-level cap and sampling row) and the analysis
    hands both selector calls the cap the leg's identity was validated to, so a
    wrong-cap or capless log alone is NOT selected: the check is not run, the
    scoring-path gate files an unwaivable pending naming the file and both caps
    (or the missing cap), the verdict withholds its word."""
    pytest.importorskip("inspect_ai")
    from inspect_ai.log import read_eval_log

    from sisteer.capeval.tiers import inspect_served as isv

    rows = primary_rows(QWEN)
    dirs = S.build_all(tmp_path / "gen", rows=rows)
    cd = dirs[f"{QWEN}/{OBJECT}/{PRIMARY}"]
    for cell in (BARE, PROMPTED):
        S.write_inspect_log(dirs[f"{QWEN}/{cell}/{PRIMARY}"], name="2026-01-01T00-00-00")
    # the fixture's default log carries the producer's plan: the store's own cap and its sampling row
    good = S.write_inspect_log(cd, name="2026-01-01T00-00-00")
    head = read_eval_log(str(good), header_only=True)
    assert head.plan.config.max_tokens == S.CAP == M.CAP and isv.log_max_tokens(head) == S.CAP
    summary = completion.read_task_summary(cd, M.TASK)
    assert head.plan.config.temperature == summary["temperature"] and head.plan.config.top_p == summary["top_p"]
    assert head.plan.config.top_k == summary["top_k"] and head.plan.steps and head.plan.steps[0].solver == "generate"
    good.unlink()
    # a complete, successful log of the right 500 ids GENERATED AT 4,096: another instrument's evidence
    wrong = S.write_inspect_log(cd, name="2026-01-01T01-00-00", max_tokens=4096)
    assert isv.log_max_tokens(read_eval_log(str(wrong), header_only=True)) == 4096
    rep = analyze(tmp_path / "gen", rows=rows, read_eval_logs=True)
    leg = leg_of(rep, QWEN, OBJECT)
    ec = leg["eval_check"]
    assert leg["refused"] is None, leg["refused"]
    assert ec["available"] is False and ec["log_present"] is True and ec["selected_logs"] == []
    assert wrong.name in ec["reason"] and "completion cap 4096" in ec["reason"] and str(M.CAP) in ec["reason"]
    assert "reusable_log" in ec["reason"] and f"generated at completion cap {M.CAP}" in ec["reason"]
    g = rep["gates"]["scoring_path"]
    assert g["status"] == M.GATE_PENDING and any(wrong.name in p and "4096" in p for p in g["pending"])
    assert rep["gates"]["pending_unwaivable"] and rep["verdict"]["word"] is None
    assert any("not waivable" in r for r in M.production_refusals(rep))
    assert analyze(tmp_path / "gen", rows=rows, read_eval_logs=True, allow_pending_gates=True)["verdict"]["word"] is None
    for cell in (BARE, PROMPTED):
        assert leg_of(rep, QWEN, cell)["eval_check"]["available"] is True
    wrong.unlink()
    # ... and one RECORDING NO CAP cannot be held to the plan's: refused by name, never read as a match
    capless = S.write_inspect_log(cd, name="2026-01-01T02-00-00", max_tokens=None)
    assert isv.log_max_tokens(read_eval_log(str(capless), header_only=True)) is None
    rep = analyze(tmp_path / "gen", rows=rows, read_eval_logs=True)
    ec = leg_of(rep, QWEN, OBJECT)["eval_check"]
    assert ec["available"] is False and ec["log_present"] is True and ec["selected_logs"] == []
    assert capless.name in ec["reason"] and "records no completion cap" in ec["reason"] and str(M.CAP) in ec["reason"]
    assert rep["gates"]["scoring_path"]["status"] == M.GATE_PENDING and rep["gates"]["pending_unwaivable"]
    assert rep["verdict"]["word"] is None and any("not waivable" in r for r in M.production_refusals(rep))


def test_an_older_valid_log_is_selected_over_newer_logs_at_another_cap_or_none(tmp_path):
    """With a newer wrong-cap log beside an older valid 8,192 log, the unheld
    selection took the newer one (mtime order) — and when THAT log disagreed
    with the record, the leg was refused for a disagreement the producer's own
    evidence does not show (a needless block). Held to the cap, the selection
    passes the wrong-cap and capless newcomers over and re-derives the valid
    log: the cross-check is available, agrees, the gate passes and the verdict
    speaks. Remove the valid log and the same two newcomers are named as what
    disqualified every candidate."""
    pytest.importorskip("inspect_ai")
    rows = primary_rows(QWEN)
    dirs = S.build_all(tmp_path / "gen", rows=rows)
    cd = dirs[f"{QWEN}/{OBJECT}/{PRIMARY}"]
    for cell in (BARE, PROMPTED):
        S.write_inspect_log(dirs[f"{QWEN}/{cell}/{PRIMARY}"], name="2026-01-01T00-00-00")
    t0 = 1_700_000_000
    def flip(item_id, v):                     # a log that DISAGREES with the record on one item
        return (1.0 - v) if item_id == S.IDS[0] else v
    valid = S.write_inspect_log(cd, name="2026-01-01T00-00-00", mtime=t0)
    wrong = S.write_inspect_log(cd, name="2026-01-01T01-00-00", max_tokens=4096, value_edit=flip, mtime=t0 + 3600)
    capless = S.write_inspect_log(cd, name="2026-01-01T02-00-00", max_tokens=None, value_edit=flip, mtime=t0 + 7200)
    assert min(cd.glob(f"{C.INSPECT_LOGS_SUBDIR}/**/*.eval"), key=lambda q: q.stat().st_mtime) == valid
    rep = analyze(tmp_path / "gen", rows=rows, read_eval_logs=True)
    leg = leg_of(rep, QWEN, OBJECT)
    ec = leg["eval_check"]
    assert leg["refused"] is None, leg["refused"]
    assert ec["available"] is True and ec["agrees"] is True and ec["log_present"] is True
    assert ec["selected_logs"] == [valid.name]
    assert ec["per_item"]["n_compared"] == 500 and ec["per_item"]["n_mismatched"] == 0
    assert rep["gates"]["scoring_path"]["status"] == M.GATE_PASS and rep["gates"]["pending"] == []
    assert M.production_refusals(rep) == [] and rep["verdict"]["word"] is not None
    assert M.numeric_leaf_diff(rep, analyze(tmp_path / "gen", rows=rows, read_eval_logs=False))["equal"]
    # without the valid log, the two newcomers are all that is left: none qualifies, each named with its cap defect
    valid.unlink()
    rep = analyze(tmp_path / "gen", rows=rows, read_eval_logs=True)
    ec = leg_of(rep, QWEN, OBJECT)["eval_check"]
    assert ec["available"] is False and ec["selected_logs"] == []
    assert wrong.name in ec["reason"] and "completion cap 4096" in ec["reason"]
    assert capless.name in ec["reason"] and "records no completion cap" in ec["reason"]
    assert rep["gates"]["scoring_path"]["status"] == M.GATE_PENDING and rep["gates"]["pending_unwaivable"]


def test_the_selector_is_handed_the_cap_the_legs_identity_was_validated_to(tmp_path, monkeypatch):
    """The seam itself: both producer selectors receive ``expected_max_tokens``
    equal to the leg's identity ``max_tokens_by_task[TASK]`` — the value
    ``identity_problem`` held to :data:`CAP` before any log was read — on every
    call, and ``eval_cross_check`` cannot be called without it (keyword-only, no
    default), so no future caller can drop the predicate the way the original
    did."""
    pytest.importorskip("inspect_ai")
    import inspect as _inspect

    from sisteer.capeval.tiers import inspect_served as isv

    param = _inspect.signature(M.eval_cross_check).parameters["max_tokens"]
    assert param.kind is _inspect.Parameter.KEYWORD_ONLY and param.default is _inspect.Parameter.empty
    seen: list[tuple[str, object]] = []
    real_reusable, real_reason = isv.reusable_log, isv.retained_log_reason

    def _reusable(d, *, expected_n, expected_ids=None, expected_max_tokens="UNSET"):
        seen.append(("reusable_log", expected_max_tokens))
        return real_reusable(d, expected_n=expected_n, expected_ids=expected_ids, expected_max_tokens=expected_max_tokens)

    def _reason(d, *, expected_n, expected_ids=None, expected_max_tokens="UNSET"):
        seen.append(("retained_log_reason", expected_max_tokens))
        return real_reason(d, expected_n=expected_n, expected_ids=expected_ids, expected_max_tokens=expected_max_tokens)

    monkeypatch.setattr(isv, "reusable_log", _reusable)
    monkeypatch.setattr(isv, "retained_log_reason", _reason)
    rows = primary_rows(QWEN)
    dirs = S.build_all(tmp_path / "gen", rows=rows)
    for cell in (BARE, PROMPTED):
        S.write_inspect_log(dirs[f"{QWEN}/{cell}/{PRIMARY}"], name="2026-01-01T00-00-00")
    S.write_inspect_log(dirs[f"{QWEN}/{OBJECT}/{PRIMARY}"], name="2026-01-01T00-00-00", max_tokens=4096)
    rep = analyze(tmp_path / "gen", rows=rows, read_eval_logs=True)
    identity_cap = leg_of(rep, QWEN, OBJECT)["identity"][C.IDENTITY_KEY_MAX_TOKENS_BY_TASK][M.TASK]
    assert identity_cap == M.CAP == 16384
    assert {name for name, _ in seen} == {"reusable_log", "retained_log_reason"}
    assert seen and all(cap == identity_cap for _, cap in seen), seen
    assert leg_of(rep, QWEN, OBJECT)["eval_check"]["available"] is False


# --------------------------------------------------------------------------- #
# a refused REQUIRED mode delta reaches the publication boundary               #
# --------------------------------------------------------------------------- #


def test_a_refused_required_mode_delta_reaches_the_publication_boundary(tmp_path, monkeypatch):
    """The WHOLE ``general`` sensitivity triple from another producer tree that
    matches the short pin. Every leg passes the pin; the triple is consistent
    within itself (its own comparison runs — no word, by design); the primary
    triple runs; and the per-cell paired delta to the primary cannot pair two
    trees, so ``run_mode_delta`` refuses. Before: the population gate and the
    refusals examined only the within-mode comparisons, so the gate PASSED, the
    refusals were empty and a verdict word was filed while the sensitivity
    estimand silently went missing. Now every REQUIRED mode-delta record is part
    of production completeness: the gate FAILS naming it, the refusals name it,
    no word is filed, and ``main()`` exits 3 before any side effect. (The
    existing test changes ONE sensitivity leg's sha, which the sensitivity
    comparison itself catches earlier.)"""
    rows = rows_for(QWEN, PRIMARY, GENERAL)
    other = S.SHA7 + "e" * 33
    S.build_all(tmp_path / "gen", rows=rows,
                overrides={(QWEN, cell, GENERAL): {"tree_sha": other} for cell in (BARE, PROMPTED, OBJECT)})
    rep = analyze(tmp_path / "gen", rows=rows)
    assert rep["unusable_legs"] == {}
    assert comparison(rep, QWEN, PRIMARY)["usable"] and comparison(rep, QWEN, GENERAL)["usable"]
    assert comparison(rep, QWEN, GENERAL)["estimands"] is not None          # consistent WITHIN the triple
    delta = next(r for r in rep["mode_delta_records"] if r["mode"] == GENERAL)
    assert delta["usable"] is False and delta["estimands"] is None
    assert "producer" in delta["reason"] and other in delta["reason"]
    assert GENERAL not in rep["mode_deltas"].get(QWEN, {})
    g = rep["gates"]["population"]
    assert g["status"] == M.GATE_FAIL and delta["name"] in g["detail"] and "producer" in g["detail"]
    assert rep["gates"]["failed"] == ["population"]
    refusals = M.production_refusals(rep)
    assert any(delta["name"] in r and "mode delta" in r for r in refusals), refusals
    assert rep["verdict"]["word"] is None and delta["name"] in rep["verdict"]["reason"]
    assert any(delta["name"] in f and "no estimand" in f for f in rep["findings"])
    assert any(delta["name"] in t for t in rep["top_items"])
    # main(): exit 3 before any figure, mirror or W&B call; --gate-report on the written report refuses too
    calls = []
    monkeypatch.setattr(M, "make_figures", lambda *a, **k: calls.append("figs") or {})
    monkeypatch.setattr(M, "mirror_figures", lambda *a, **k: calls.append("mirror") or [])
    monkeypatch.setattr(M, "log_wandb", lambda *a, **k: calls.append("wandb"))
    cells = S.write_cells_json(tmp_path / "in" / "cells.json", rows)
    rows_path = S.write_sampling_rows(tmp_path / "in" / "rows.json")
    facts = S.write_render_facts(tmp_path / "in" / "facts.json")
    ids = tmp_path / "in" / "ids.json"
    ids.write_text(json.dumps(S.IDS))
    out = tmp_path / "report.json"
    rc = M.main(["--gen-root", str(tmp_path / "gen"), "--sha", S.SHA7, "--cells", str(cells), "--sampling-rows", str(rows_path),
                 "--render-facts", str(facts), "--math500-ids", str(ids), "--analysis-sha", S.FULL_SHA, "--bootstrap", str(B_FAST),
                 "--wandb-mode", "disabled", "--no-mirror", "--no-eval-logs", "--out", str(out)])
    assert rc == M.EXIT_NOT_PRODUCTION == 3 and calls == []
    assert M.main(["--gate-report", str(out)]) == 3
    # the control: the same rows under ONE tree are a production report whose delta ran ...
    S.build_all(tmp_path / "one", rows=rows)
    rep2 = analyze(tmp_path / "one", rows=rows)
    assert M.production_refusals(rep2) == [] and rep2["mode_deltas"][QWEN][GENERAL] and rep2["verdict"]["word"] is not None
    # ... and the boundary reads the delta RECORDS themselves, not only the gate that folds them
    rep2["mode_delta_records"][0]["estimands"] = None
    assert any("mode delta" in r for r in M.production_refusals(rep2))


# --------------------------------------------------------------------------- #
# the at-cap readout as a RESULT; the no-reasoning readout                     #
# --------------------------------------------------------------------------- #

LOOP_BLOCK = "1 = 3/3? no, "                 # a 13-character block a degenerating draw repeats to the cap
AT_CAP_KEYS = {"rate", "share", "n_at_cap", "n_with_stop_reason", "line_1pct", "line_1pct_exceeded", "surfacing_threshold",
               "at_or_above_surfacing_threshold", "n_case_b_unterminated_at_cap", "n_case_c_answer_at_cap", "split_source",
               "split_source_reason", "n_loop", "n_truncated_reasoning", "n_text_unavailable", "store_rows_split"}
SPLIT_KEYS = {"text_source", "n_at_cap", "n_classified", "n_loop", "n_truncated_reasoning", "n_text_unavailable", "tail_classes"}
THINKING_KEYS = {"split_mode", "n_on_rows", "n_finished", "n_no_reasoning_emitted", "share_no_reasoning",
                 "n_no_reasoning_at_cap", "by_construction", "n_present_by_record", "n_present_by_channel", "rule"}
TRUNCATION_KEYS = {"premise_rate", "premise_source", "line_1pct", "surfacing_threshold", "cap", "n_primary_cells_design",
                   "n_primary_cells_in_matrix", "n_primary_cells_read", "primary_legs_not_read", "primary_legs_absent_from_matrix",
                   "matrix_complete", "primary_over_1pct", "primary_at_or_above_threshold", "max_primary_share",
                   "max_primary_share_leg", "premise_failed", "headline", "loop_rule", "loop_rule_scope", "cells",
                   # the re-draws aggregated beside the at-cap shares, the caveat stated once
                   "cells_redrawn", "n_cells_redrawn", "n_items_redrawn_total", "n_redraw_calls_total", "n_items_sample_error_total",
                   "max_calls_per_item", "redraw_mechanism", "redraw_ladder", "redraw_selection_caveat"}
#: the per-leg re-draw readout — the ledger's classification of every call beyond one per (sample, recorded run)
REDRAW_KEYS = {"n_redraw_calls", "n_items_redrawn", "share_items_redrawn", "max_calls_per_item", "n_items_sample_error", "attributed",
               "n_resume_calls", "n_unexplained_calls", "n_items_unexplained", "n_items_body_changed", "n_over_ladder",
               "calls_per_item_hist", "n_runs_recorded", "n_runs_unreadable_ts", "final_draw_rule", "n_final_draw_file_last_differs",
               "ladder", "n_items_sample_error_note", "mechanism", "selection_caveat", "attribution_source"}
#: the log cross-check's record under eval_check.redraws.ledger_check (the log-reading run only)
LEDGER_CHECK_KEYS = {"agrees", "run", "n_items_compared", "n_mismatched", "mismatches", "n_failed_attempts_by_class",
                     "n_failed_attempts_exact", "n_failed_attempts_bounded", "unverifiable_items", "final_draw_selection",
                     "n_items", "n_items_with_failed_attempts", "n_failed_attempts", "n_model_retries", "n_generate_calls",
                     "n_items_errored", "n_at_cap"}


def _class_counts(**named):
    """A `n_failed_attempts_by_class` block with the named classes' counts and 0 elsewhere."""
    return {cls: named.get(cls, 0) for cls in M.FAILED_ATTEMPT_CLASSES}
CLASS_KEYS = {"loop", "rule", "period", "repeats", "cover", "zlib_ratio"}


def _derivation(n=400):
    """A long derivation with no repeated block: every step carries its own numbers."""
    return " ".join(f"step {i}: let x_{i} = {i * i + 3 * i + 1}, so f(x_{i}) = {(i * 7919) % 1009} and the sum is "
                    f"{(i * 104729) % 65537}." for i in range(n))


def test_classify_at_cap_tail_flags_exact_repeats_only():
    # the repeated-block rule: a looping tail after some real reasoning, reported at its TRUE period
    loop = M.classify_at_cap_tail(_derivation(20) + LOOP_BLOCK * 200)
    assert set(loop) == CLASS_KEYS
    assert loop["loop"] and loop["rule"] == M.LOOP_RULE_REPEATED_BLOCK and loop["period"] == len(LOOP_BLOCK)
    assert loop["repeats"] >= M.LOOP_MIN_REPEATS and loop["cover"] >= M.LOOP_MIN_COVER and loop["zlib_ratio"] < 0.15
    # the smoke's real loop, in shape: a 41-character CJK block repeated to the cap
    smoke_like = "1 = 3/3？不，1 = 3/3？不，1 = 3/3？不，1 = 3/3？不对，"
    real = M.classify_at_cap_tail("先算一下。" * 20 + smoke_like * 40)
    assert real["loop"] and real["rule"] == M.LOOP_RULE_REPEATED_BLOCK and real["period"] == len(smoke_like) and real["repeats"] >= 25
    # a super-block whose sub-blocks differ (the smoke's real loop alternated "no" and "wrong"): the period is the super-block's
    super_block = LOOP_BLOCK * 3 + "1 = 3/3? wrong, "
    assert M.classify_at_cap_tail(super_block * 40)["period"] == len(super_block)
    # the cover bound is a bound: a 100-char block (prose, no inner period) over 60 % of the tail fires, over 53 % it does not
    prose = _derivation(60)
    block, prefix = prose[100:200], prose[-700:]
    at_bound = M.classify_at_cap_tail(prefix[-600:] + block * 9)
    assert at_bound["loop"] and at_bound["rule"] == M.LOOP_RULE_REPEATED_BLOCK and at_bound["period"] == 100
    assert at_bound["repeats"] == 9 and 0.6 <= at_bound["cover"] < 0.62
    under = M.classify_at_cap_tail(prefix + block * 8)
    assert not under["loop"] and under["rule"] == M.LOOP_RULE_NONE and under["period"] == 100 and under["repeats"] == 8
    # the repeat bound: a 300-char block five times fires; three whole repeats of a 400-char block do not (the tail holds
    # fewer than four of them — with a 1,500-character tail no period above 375 can fire)
    big = prose[200:500]
    assert M.classify_at_cap_tail(big * 5)["loop"] and M.classify_at_cap_tail(big * 5)["period"] == 300
    biggest = prose[300:700]
    three = M.classify_at_cap_tail(biggest * 3)
    assert not three["loop"] and three["rule"] == M.LOOP_RULE_NONE, three
    # compression NEVER decides: tails that compress like a loop but progress are not loops — a drifting counter, an
    # English enumeration and its CJK twin (each compresses to ~0.1 of its bytes; none has a repeated block) ...
    drifting = "".join("-" * 60 + f"{i:03d}" for i in range(60))
    z = M.classify_at_cap_tail(drifting)
    assert not z["loop"] and z["rule"] == M.LOOP_RULE_NONE and z["zlib_ratio"] < 0.15
    enumeration = "\n".join(f"For n = {i:2d}, n mod 7 = {i % 7}, so this candidate is "
                            + ("divisible by seven." if i % 7 == 0 else "not divisible by seven.") for i in range(1, 51))
    en = M.classify_at_cap_tail(enumeration)
    assert not en["loop"] and en["rule"] == M.LOOP_RULE_NONE and en["zlib_ratio"] < 0.15, en
    cjk = "\n".join(f"当 n = {i} 时，n 除以 7 的余数是 {i % 7}，所以这个候选" + ("能被七整除。" if i % 7 == 0 else "不能被七整除。")
                    for i in range(1, 51))
    zh = M.classify_at_cap_tail(cjk)
    assert not zh["loop"] and zh["rule"] == M.LOOP_RULE_NONE and zh["zlib_ratio"] < 0.15, zh
    # ... and a genuine long numerical expansion: the digits of sqrt(2) as a long-division working (no period, low
    # compression — an honest computation that reaches the cap is truncated reasoning)
    digits = str(math.isqrt(2 * 10 ** 1600))
    expansion = "1." + digits[1:] + "\n" + "\n".join(f"bring down 00, remainder {digits[i:i + 6]}, next digit {digits[i + 6]}"
                                                    for i in range(0, 600, 7))
    ex = M.classify_at_cap_tail(expansion)
    assert not ex["loop"] and ex["rule"] == M.LOOP_RULE_NONE, ex
    # NEGATIVES — a rule that flags any of these is wrong: a long unique derivation ...
    deriv = M.classify_at_cap_tail(_derivation())
    assert not deriv["loop"] and deriv["rule"] == M.LOOP_RULE_NONE and deriv["zlib_ratio"] > 0.15
    assert deriv["period"] is None or deriv["cover"] < M.LOOP_MIN_COVER
    # the record's rule vocabulary is exactly the two deciding outcomes; zlib is a recorded diagnostic, never a rule
    assert {M.LOOP_RULE_REPEATED_BLOCK, M.LOOP_RULE_NONE} == {"repeated_block", "none"} and not hasattr(M, "LOOP_RULE_ZLIB")
    # ... a structured enumeration whose blocks differ (the shape a naive line-repeat rule mislabelled as loops) ...
    options = "\n".join(f"Option {k}: take n = {k}; then n^2 + n = {k * k + k}, the partial sum is {k * (k + 1) // 2} and the "
                        f"remainder mod 7 is {(k * k + k) % 7}, which {'does' if k % 2 else 'does not'} match." for k in range(1, 9))
    opt = M.classify_at_cap_tail(_derivation(10) + "\n" + options)
    assert not opt["loop"] and opt["rule"] == M.LOOP_RULE_NONE, opt
    # ... a numbered list, and a LaTeX-heavy tail with the same macros on every line
    numbered = "\n".join(f"{k}. The {['first', 'second', 'third', 'fourth'][k % 4]} case gives {k * 37 % 101} after {k} steps."
                         for k in range(1, 40))
    assert not M.classify_at_cap_tail(numbered)["loop"]
    latex = "\n".join(f"$$\\frac{{{k}}}{{{k + 1}}} + \\sqrt{{{k * k}}} = \\frac{{{2 * k + 1}}}{{{k + 1}}}$$" for k in range(1, 60))
    assert not M.classify_at_cap_tail(latex)["loop"]
    # degenerate inputs: nothing to classify is not a loop, and the record says why (no block repeats even twice)
    assert M.classify_at_cap_tail("") == {"loop": False, "rule": "none", "period": None, "repeats": None, "cover": None, "zlib_ratio": 1.0}
    assert M.classify_at_cap_tail(None)["loop"] is False and M.classify_at_cap_tail("short")["period"] is None
    # the rule text names every constant it is built from, and says compression is a diagnostic that never decides
    for tok in (str(M.LOOP_TAIL_CHARS), str(M.LOOP_PERIOD_MIN), str(M.LOOP_PERIOD_MAX), str(M.LOOP_MIN_REPEATS), "60%",
                "diagnostic", "never decides"):
        assert tok in M.LOOP_RULE_TEXT, tok


def test_at_cap_share_against_the_1pct_line_and_the_3pct_surfacing_threshold(tmp_path):
    """The share is the summary's own at-cap rate (n_at_cap over
    n_stop_reason), filed per cell against the 1 % line; a PRIMARY cell at
    exactly 3 % fails the premise (the threshold is >=) and the headline leads
    with the loop / truncated-reasoning split; a cell at exactly 1 % is not over
    the line (the harness rule is > 1 %); a sensitivity row over the line is
    reported and flagged but never decides the premise. The parent's 2 % cap
    gate still fails a 3 % cell — the readout is filed, the verdict is not."""
    rows = primary_rows(QWEN)
    texts = {j: (f"reasoning{C.THINK_CLOSE if hasattr(C, 'THINK_CLOSE') else '</think>'}{LOOP_BLOCK * 200}", None) for j in range(6)}
    texts.update({j: (f"reasoning</think>{_derivation()}", None) for j in range(6, 15)})
    S.build_all(tmp_path / "thr", rows=rows, overrides={(QWEN, OBJECT, PRIMARY): {"capped_scored": set(range(15)), "texts": texts},
                                                        (QWEN, BARE, PRIMARY): {"capped_scored": set(range(4))}})
    rep = analyze(tmp_path / "thr", rows=rows)
    obj, bare = leg_of(rep, QWEN, OBJECT)["at_cap"], leg_of(rep, QWEN, BARE)["at_cap"]
    assert set(obj) == AT_CAP_KEYS
    assert obj["share"] == 0.03 == obj["rate"] and obj["n_at_cap"] == 15 and obj["n_with_stop_reason"] == 500
    assert obj["line_1pct_exceeded"] is True and obj["at_or_above_surfacing_threshold"] is True
    assert (obj["n_loop"], obj["n_truncated_reasoning"], obj["n_text_unavailable"]) == (6, 9, 0)
    assert obj["n_case_c_answer_at_cap"] == 15 and obj["n_case_b_unterminated_at_cap"] == 0
    # no log read: the split the record reports IS the store rows', and the store-rows split rides alongside, identical
    assert obj["split_source"] == "store_rows" and "not read" in obj["split_source_reason"]
    assert set(obj["store_rows_split"]) == SPLIT_KEYS and obj["store_rows_split"]["text_source"] == "rows.jsonl"
    assert {k: obj["store_rows_split"][k] for k in ("n_loop", "n_truncated_reasoning", "n_text_unavailable", "n_classified")} == \
        {"n_loop": 6, "n_truncated_reasoning": 9, "n_text_unavailable": 0, "n_classified": 15}
    classes = obj["store_rows_split"]["tail_classes"]
    assert set(classes) == set(S.IDS[:15]) and all(classes[S.IDS[j]]["loop"] is (j < 6) for j in range(15))
    assert all(len(c["text_sha256"]) == 64 and c["text_source"] == "rows.jsonl" and set(c) == CLASS_KEYS | {"text_source", "text_sha256"}
               for c in classes.values())
    assert bare["share"] == 0.008 and bare["line_1pct_exceeded"] is False and bare["at_or_above_surfacing_threshold"] is False
    assert (bare["n_loop"], bare["n_truncated_reasoning"], bare["n_text_unavailable"]) == (0, 4, 0)
    tr = rep["truncation"]
    assert set(tr) == TRUNCATION_KEYS
    assert tr["premise_failed"] is True and tr["primary_at_or_above_threshold"] == [M.leg_key(QWEN, OBJECT, PRIMARY)]
    assert tr["primary_over_1pct"] == [M.leg_key(QWEN, OBJECT, PRIMARY)] and tr["max_primary_share"] == 0.03
    assert tr["max_primary_share_leg"] == M.leg_key(QWEN, OBJECT, PRIMARY)
    assert "did not hold for MATH-500" in tr["headline"] and "GSM8K" in tr["headline"] and f"{M.CAP:,}" in tr["headline"]
    assert "loop 6 / truncated reasoning 9" in tr["headline"] and tr["headline"].endswith("split leads.")
    assert rep["top_items"][0] == tr["headline"] and tr["headline"] in rep["findings"]
    assert any(f.startswith(M.leg_key(QWEN, OBJECT, PRIMARY)) and "> the 1 % line" in f for f in rep["findings"])
    assert not any("> the 1 % line" in f and f.startswith(M.leg_key(QWEN, BARE, PRIMARY)) for f in rep["findings"])
    assert tr["n_primary_cells_in_matrix"] == tr["n_primary_cells_read"] == 3 and tr["matrix_complete"] is False
    # a reduced matrix: the nine designed primary cells it does not name are listed as not read (the premise still FAILS
    # here — one cell at the threshold is enough on its own)
    absent = [M.leg_key(m, c, PRIMARY) for m in (GEMMA, GLM, NEMO) for c in (BARE, PROMPTED, OBJECT)]
    assert sorted(tr["primary_legs_not_read"]) == sorted(absent) == sorted(tr["primary_legs_absent_from_matrix"])
    assert {c["leg"] for c in tr["cells"]} == set(rep["legs"])
    assert all(c["primary"] and not c["sensitivity"] for c in tr["cells"])
    # the parent's re-cost gate (2 %) is not this readout: at the ceiling it is ADVISORY — the 3 % cell is named, the
    # gate passes, and the verdict is not withheld on its account (no rung above)
    assert "cap" not in rep["gates"]["failed"] and rep["gates"]["cap"]["binding"] is False
    assert any(M.leg_key(QWEN, OBJECT, PRIMARY) in a and "3.0" in a for a in rep["gates"]["cap"]["advisories"])
    # exactly ON the 1 % line is not over it; a sensitivity row over it is flagged, reported, and never decides the premise
    rows2 = rows_for(QWEN, PRIMARY, GENERAL)
    S.build_all(tmp_path / "on", rows=rows2, overrides={(QWEN, BARE, PRIMARY): {"capped_scored": set(range(5))},
                                                        (QWEN, BARE, GENERAL): {"capped_scored": set(range(10))}})
    rep2 = analyze(tmp_path / "on", rows=rows2)
    on = leg_of(rep2, QWEN, BARE)["at_cap"]
    assert on["share"] == 0.01 and on["line_1pct_exceeded"] is False and on["at_or_above_surfacing_threshold"] is False
    tr2 = rep2["truncation"]
    # below the threshold on a REDUCED matrix the premise is UNDETERMINED, never holding: nine designed cells were not read
    assert tr2["premise_failed"] is None and tr2["headline"] is None and tr2["primary_over_1pct"] == []
    assert sorted(tr2["primary_legs_not_read"]) == sorted(absent) and tr2["matrix_complete"] is False
    gen = next(c for c in tr2["cells"] if c["leg"] == M.leg_key(QWEN, BARE, GENERAL))
    assert gen["sensitivity"] is True and gen["share"] == 0.02 and gen["line_1pct_exceeded"] is True
    assert any(f.startswith(gen["leg"]) and "> the 1 % line" in f and "[sensitivity row]" in f for f in rep2["findings"])
    assert [c["primary"] for c in tr2["cells"]] == [True] * 3 + [False] * 3
    assert M.production_refusals(rep2) == [] and not any(t.startswith("The ruling") for t in rep2["top_items"])


def test_no_reasoning_share_per_split_mode(tmp_path):
    """A FINISHED draw whose record carries no think block is the model
    declining to think — counted per cell as a share of the ON draws, with the
    family's rule stated: record_only (gemma, the server parser's empty
    reasoning channel), explicit_tags (glm, no tag); implicit_open (qwen,
    nemotron) 0 by construction. An at-cap draw without reasoning is counted
    apart, never folded into the finished share."""
    rows = primary_rows(GEMMA) + primary_rows(GLM) + primary_rows(NEMO)
    S.build_all(tmp_path, rows=rows, overrides={(GEMMA, BARE, PRIMARY): {"no_think": {1, 2, 3, 7}, "capped_scored": {7}},
                                                (GLM, BARE, PRIMARY): {"no_think": set(range(10))}})
    rep = analyze(tmp_path, rows=rows)
    assert M.production_refusals(rep) == []
    g = leg_of(rep, GEMMA, BARE)["thinking"]
    assert g == {"split_mode": "record_only", "n_on_rows": 500, "n_finished": 499, "n_no_reasoning_emitted": 3,
                 "share_no_reasoning": 0.006, "n_no_reasoning_at_cap": 1, "by_construction": False,
                 "n_present_by_record": 496, "n_present_by_channel": 0,
                 "rule": M.NO_REASONING_RULE["record_only"]}
    glm = leg_of(rep, GLM, BARE)["thinking"]
    assert (glm["split_mode"], glm["n_no_reasoning_emitted"], glm["share_no_reasoning"], glm["by_construction"]) == \
        ("explicit_tags", 10, 0.02, False)
    # the builder's default reasoning channel under explicit_tags is None: the 10 tag-free draws are the genuine no-reasoning
    # case, none called present by the channel (the client-lift fallback is the test below); the split-anomaly gate's line
    # states what its count IS (a tag in the text) and that the channel is empty on the rest
    assert (glm["n_present_by_record"], glm["n_present_by_channel"]) == (490, 0)
    assert any(f.startswith(M.leg_key(GLM, BARE, PRIMARY)) and "think block present in the TEXT on 490 of 500 completions (98.0 %)" in f
               and "the reasoning channel is empty on the other 10 (the D115 finding)" in f for f in rep["findings"])
    nemo = leg_of(rep, NEMO, BARE)["thinking"]
    assert (nemo["split_mode"], nemo["n_no_reasoning_emitted"], nemo["share_no_reasoning"], nemo["by_construction"]) == \
        ("implicit_open", 0, 0.0, True)
    assert "by construction" in nemo["rule"] and "reasoning channel" in g["rule"] and "<think>" in glm["rule"]
    # every nonzero share is a finding in the cell's name; the block's cells carry the share too
    assert any(f.startswith(M.leg_key(GEMMA, BARE, PRIMARY)) and "3 of 500" in f and "no reasoning" in f for f in rep["findings"])
    assert any(f.startswith(M.leg_key(GLM, BARE, PRIMARY)) and "10 of 500" in f for f in rep["findings"])
    assert not any("no reasoning" in f and f.startswith(M.leg_key(NEMO, BARE, PRIMARY)) for f in rep["findings"])
    cell = next(c for c in rep["truncation"]["cells"] if c["leg"] == M.leg_key(GLM, BARE, PRIMARY))
    assert (cell["share_no_reasoning"], cell["n_no_reasoning_emitted"], cell["no_reasoning_by_construction"]) == (0.02, 10, False)
    # under implicit_open a record with think_block_present false is one the scorer cannot have filed: the readout counts
    # it (by_construction stays a statement about the mode) and the split-anomaly gate FAILS the leg by name
    S.build_all(tmp_path / "imp", rows=primary_rows(NEMO), overrides={(NEMO, BARE, PRIMARY): {"no_think": {4}}})
    rep2 = analyze(tmp_path / "imp", rows=primary_rows(NEMO))
    th = leg_of(rep2, NEMO, BARE)["thinking"]
    assert th["n_no_reasoning_emitted"] == 1 and th["by_construction"] is True
    assert "split_anomalies" in rep2["gates"]["failed"] and "implicit_open" in rep2["gates"]["split_anomalies"]["detail"]
    with pytest.raises(M.AnalysisError, match="split mode"):
        M.thinking_record({}, split_mode="not_a_mode", rows_reasoning={})


def test_explicit_tags_reads_the_client_lifted_reasoning_channel_beside_the_record_flag(tmp_path):
    """The GLM readout artefact (2026-09-11): inspect_ai 0.3.260 lifts a COMPLETE
    ``<think>…</think>`` block out of the text into a ``ContentReasoning`` part
    before the scorer's tag test, so under ``explicit_tags`` a finished draw that
    emitted its block has ``think_block_present`` FALSE on the record while its
    row's ``extra_json.reasoning`` carries the block (969 of 1,000 GLM draws in
    production read as "no reasoning"). The readout therefore reads the
    row's channel beside the record's flag, and says which source made each
    call: (1) record false + non-empty channel -> present, by channel; (2) record
    false + EMPTY channel (None or "") + finished -> the genuine no-reasoning
    case, still counted; (3) an at-cap UNTERMINATED draw (the open tag is in the
    text, record true) -> the record's call, unchanged; an at-cap case-(c) draw
    with a lifted block is present by channel, not "no reasoning at cap"."""
    rows = primary_rows(GLM)
    lifted = {j: (f"answer {j}", f"lifted reasoning {j}") for j in range(10)}                 # (1)
    empty = {10: ("answer 10", None), 11: ("answer 11", None), 12: ("answer 12", "")}        # (2)
    case_c = {13: ("closed its block, then ran to the cap in the answer", "lifted reasoning 13")}
    S.build_all(tmp_path, rows=rows, overrides={(GLM, BARE, PRIMARY): {
        "no_think": set(range(14)), "unscored": {20, 21, 22, 23}, "capped_scored": {13},     # (3): 20-23 are case (b)
        "texts": {**lifted, **empty, **case_c}}})
    rep = analyze(tmp_path, rows=rows)
    assert M.production_refusals(rep) == []
    leg = leg_of(rep, GLM, BARE)
    assert leg["thinking"] == {"split_mode": "explicit_tags", "n_on_rows": 500, "n_finished": 495, "n_no_reasoning_emitted": 3,
                               "share_no_reasoning": 0.006, "n_no_reasoning_at_cap": 0, "by_construction": False,
                               "n_present_by_record": 486, "n_present_by_channel": 11,
                               "rule": M.NO_REASONING_RULE["explicit_tags"]}
    # the record's own counts are untouched: the producer's flag is false on all 14, the summary and the split-anomaly gate
    # still read 486 present, the at-cap readout still sees 4 case-(b) and 1 case-(c) draw
    assert leg["summary"]["n_think_block_present"] == 486 and sum(1 for r in leg["items"].values() if r["think_block_present"]) == 486
    assert rep["gates"]["split_anomalies"]["counts"][leg["leg"]]["n_think_block_present"] == 486
    assert (leg["at_cap"]["n_case_b_unterminated_at_cap"], leg["at_cap"]["n_case_c_answer_at_cap"], leg["at_cap"]["n_at_cap"]) == (4, 1, 5)
    # every present call has exactly one source, and the counts partition the 500 draws
    th = leg["thinking"]
    assert th["n_present_by_record"] + th["n_present_by_channel"] + th["n_no_reasoning_emitted"] + th["n_no_reasoning_at_cap"] == 500
    # the pure function, draw by draw: record true wins whatever the channel; channel only under explicit_tags; "" is empty
    rec_true, rec_false = {"think_block_present": True, "at_cap": True}, {"think_block_present": False, "at_cap": False}
    assert M.reasoning_source(rec_true, reasoning=None, split_mode="explicit_tags") == M.PRESENT_BY_RECORD == "record"
    assert M.reasoning_source(rec_true, reasoning="r", split_mode="explicit_tags") == "record"
    assert M.reasoning_source(rec_false, reasoning="r", split_mode="explicit_tags") == M.PRESENT_BY_CHANNEL == "channel"
    assert M.reasoning_source(rec_false, reasoning="", split_mode="explicit_tags") is None
    assert M.reasoning_source(rec_false, reasoning=None, split_mode="explicit_tags") is None
    assert M.reasoning_source(rec_false, reasoning="r", split_mode="record_only") is None
    assert M.reasoning_source(rec_false, reasoning="r", split_mode="implicit_open") is None
    # the findings: the 3 genuine draws are the no-reasoning line; the 11 lifted
    # ones are a readout fact naming both sources
    fs = [f for f in rep["findings"] if f.startswith(leg["leg"])]
    assert any("3 of 500 thinking-ON draws finished with no reasoning emitted (0.6 %)" in f for f in fs)
    assert any("reasoning present on 497 of 500 thinking-ON draws — 11 by the reasoning channel" in f and "+ 486 by the record" in f
               and "would have read 11 draw(s) as no reasoning" in f for f in fs)
    cell = next(c for c in rep["truncation"]["cells"] if c["leg"] == leg["leg"])
    assert (cell["n_present_by_record"], cell["n_present_by_channel"], cell["n_no_reasoning_emitted"]) == (486, 11, 3)
    # the split-anomaly gate's line keeps the producer's count (486: a tag in the TEXT) and says what it is, so it does not
    # contradict the reasoning finding above; the old "think block present on N" wording is gone from every finding
    gate_lines = [f for f in fs if "think block present in the TEXT" in f]
    assert len(gate_lines) == 1 and "think block present in the TEXT on 486 of 500 completions (97.2 %)" in gate_lines[0]
    assert "inspect_ai lifts a closed block out of the text client-side" in gate_lines[0]
    assert "the reasoning channel carries 11 more (see the reasoning finding); 3 carry neither (the D115 finding)" in gate_lines[0]
    assert gate_lines[0].endswith("a tag-free completion is scored as the answer (a fact about the render, not a failure)")
    assert not any("think block present on " in f for f in rep["findings"])
    assert rep["gates"]["split_anomalies"]["status"] == M.GATE_PASS
    # the rule names both sources and the client parser, and still names the tag (the artifact's caption reads it)
    assert "<think>" in th["rule"] and "reasoning channel" in th["rule"] and "inspect_ai 0.3.260" in th["rule"]
    assert "n_present_by_channel" in M.figure_captions(rep)["final_m5t_no_reasoning_share"]


def test_the_channel_fallback_is_explicit_tags_only_the_other_split_modes_read_the_record_verbatim(tmp_path):
    """(4) A parser-separated cell (gemma, ``record_only``) and a prompt-opened
    one (nemotron, ``implicit_open``) are the record's call byte for byte, even
    on a record-false draw whose row carries a non-empty reasoning channel: under
    record_only the producer's flag already read the channel (the server parser
    owns the split — a false flag with a channel is the producer's contradiction
    to surface, not ours to paper over), under implicit_open a false flag is a
    record the scorer cannot have filed (the gate fails it). Both readouts equal
    the ones the old flag-only rule filed, with by_channel 0."""
    rows = primary_rows(GEMMA) + primary_rows(NEMO)
    # the builder files reasoning "reasoning" on every gemma row, so a no_think gemma item IS flag false + non-empty channel
    S.build_all(tmp_path, rows=rows, overrides={(GEMMA, BARE, PRIMARY): {"no_think": {1, 2, 3}},
                                                (NEMO, BARE, PRIMARY): {"no_think": {4, 5}, "texts": {4: ("r</think>7", "lifted"),
                                                                                                      5: ("r</think>7", "lifted")}}})
    rep = analyze(tmp_path, rows=rows)
    g = leg_of(rep, GEMMA, BARE)
    assert all(json.loads(r["extra_json"])["reasoning"] == "reasoning" for r in _acc_rows(g["cell_dir"]) if json.loads(r["extra_json"])["think_block_present"] is False)
    assert g["thinking"] == {"split_mode": "record_only", "n_on_rows": 500, "n_finished": 500, "n_no_reasoning_emitted": 3,
                             "share_no_reasoning": 0.006, "n_no_reasoning_at_cap": 0, "by_construction": False,
                             "n_present_by_record": 497, "n_present_by_channel": 0, "rule": M.NO_REASONING_RULE["record_only"]}
    n = leg_of(rep, NEMO, BARE)["thinking"]
    assert n == {"split_mode": "implicit_open", "n_on_rows": 500, "n_finished": 500, "n_no_reasoning_emitted": 2,
                 "share_no_reasoning": 0.004, "n_no_reasoning_at_cap": 0, "by_construction": True,
                 "n_present_by_record": 498, "n_present_by_channel": 0, "rule": M.NO_REASONING_RULE["implicit_open"]}
    assert "split_anomalies" in rep["gates"]["failed"] and "implicit_open" in rep["gates"]["split_anomalies"]["detail"]
    assert not any("by the reasoning channel" in f for f in rep["findings"])
    # the gate's explicit_tags line is explicit_tags-only: a record_only cell files no "think block present" line, and an
    # implicit_open cell's flag-false records are the unchanged gate FAILURE, word for word
    assert not any("think block present" in f for f in rep["findings"])
    assert (f"{M.leg_key(NEMO, BARE, PRIMARY)}: n_think_block_present 498 != 500 under implicit_open — the prompt opens the block on "
            "every completion") in rep["gates"]["split_anomalies"]["detail"]


def _acc_rows(cell_dir):
    return [r for r in (json.loads(line) for line in (Path(cell_dir) / "rows.jsonl").read_text().splitlines() if line)
            if r["item_id"] != C.AGG_ITEM_ID and r["metric"] == C.METRIC_ACC]


def test_the_channel_fallback_is_venv_independent_run_e_and_run_f_agree_on_every_numeric_leaf(tmp_path):
    """(5) The fallback reads rows.jsonl, which both venvs read, so the log-reading
    run E and the figure run F file the same readout and still agree on
    every numeric leaf. In the inspect venv a real ``.eval`` log in inspect's
    lifted shape (a ``ContentReasoning`` part before the text) rides beside the
    store and the per-item cross-check agrees with the record; in the figure
    venv no log is written and E's record stands alone — the match holds in
    both."""
    rows = primary_rows(GLM)
    lifted = {j: (f"answer {j}", f"lifted reasoning {j}") for j in range(20)}
    dirs = S.build_all(tmp_path, rows=rows, overrides={(GLM, BARE, PRIMARY): {"no_think": set(range(20)), "texts": lifted}})
    has_inspect = importlib.util.find_spec("inspect_ai") is not None
    if has_inspect:
        S.write_inspect_log(dirs[M.leg_key(GLM, BARE, PRIMARY)], name="pass", texts={S.IDS[j]: t for j, t in lifted.items()})
    e = analyze(tmp_path, rows=rows, read_eval_logs=True)
    f = analyze(tmp_path, rows=rows, read_eval_logs=False)
    for rep in (e, f):
        assert M.production_refusals(rep) == []
        th = leg_of(rep, GLM, BARE)["thinking"]
        assert (th["n_no_reasoning_emitted"], th["share_no_reasoning"], th["n_present_by_channel"], th["n_present_by_record"]) == (0, 0.0, 20, 480)
        assert not any("finished with no reasoning emitted" in f_ and f_.startswith(M.leg_key(GLM, BARE, PRIMARY)) for f_ in rep["findings"])
        assert any("20 by the reasoning channel" in f_ for f_ in rep["findings"])
        assert any("think block present in the TEXT on 480 of 500 completions (96.0 %)" in f_
                   and "the reasoning channel carries the other 20 (see the reasoning finding)" in f_ for f_ in rep["findings"])
    ec = leg_of(e, GLM, BARE)["eval_check"]
    assert ec["available"] is has_inspect and (ec["agrees"] is True if has_inspect else ec["log_present"] is False)
    d = M.numeric_leaf_diff(e, f)
    assert d["equal"] and d["shared_numeric"] > 1000, d


def test_the_readouts_are_absent_on_a_refused_or_missing_leg_and_the_premise_is_then_undetermined(tmp_path):
    """The readouts are RESULTS: a refused or missing leg carries neither key
    (absent, not zero), the block lists no cell for it and names it as not
    read; the premise is then None unless another primary cell fails it alone."""
    rows = primary_rows(GEMMA)
    S.build_all(tmp_path / "r", rows=rows,
                overrides={(GEMMA, BARE, PRIMARY): {"ids": S.IDS[:20], "limits": {"inspect_limit": 20}, "n_expected_full": 500}})
    rep = analyze(tmp_path / "r", rows=rows)
    leg = leg_of(rep, GEMMA, BARE)
    assert leg["refused"] and "at_cap" not in leg and "thinking" not in leg
    ctl = next(r for r in rep["controls"] if r["leg"] == leg["leg"])
    assert ctl["at_cap"] is None and ctl["thinking"] is None
    tr = rep["truncation"]
    assert {c["leg"] for c in tr["cells"]} == {M.leg_key(GEMMA, PROMPTED, PRIMARY), M.leg_key(GEMMA, OBJECT, PRIMARY)}
    # not read = the refused leg AND the nine designed primary cells this reduced matrix never named (listed separately too)
    absent = [M.leg_key(m, c, PRIMARY) for m in (QWEN, GLM, NEMO) for c in (BARE, PROMPTED, OBJECT)]
    assert sorted(tr["primary_legs_not_read"]) == sorted(absent + [leg["leg"]]) and sorted(tr["primary_legs_absent_from_matrix"]) == sorted(absent)
    assert tr["premise_failed"] is None and tr["headline"] is None and tr["matrix_complete"] is False
    assert (tr["n_primary_cells_read"], tr["n_primary_cells_in_matrix"]) == (2, 3) and tr["max_primary_share"] == 0.0
    assert not any("The ruling" in t for t in rep["top_items"])
    # a MISSING leg likewise — and a cell at the threshold among the others fails the premise on its own
    S.build_all(tmp_path / "m", rows=rows, overrides={(GEMMA, OBJECT, PRIMARY): {"capped_scored": set(range(15))}})
    shutil.rmtree(S.out_root_of(tmp_path / "m", GEMMA, BARE, PRIMARY))
    rep2 = analyze(tmp_path / "m", rows=rows)
    assert leg_of(rep2, GEMMA, BARE)["missing"] and "at_cap" not in leg_of(rep2, GEMMA, BARE)
    tr2 = rep2["truncation"]
    assert tr2["premise_failed"] is True and sorted(tr2["primary_legs_not_read"]) == sorted(absent + [M.leg_key(GEMMA, BARE, PRIMARY)])
    assert tr2["primary_at_or_above_threshold"] == [M.leg_key(GEMMA, OBJECT, PRIMARY)] and tr2["headline"]
    # the premise HOLDS only on the complete designed set: all 12 primary cells read, every one below the threshold; a
    # designed cell missing from the matrix leaves it undetermined even when every cell read is at 0 %
    full = [r for r in S.cells_matrix() if r["sampling_mode"] == PRIMARY]
    S.build_all(tmp_path / "full", rows=full)
    tr3 = analyze(tmp_path / "full", rows=full)["truncation"]
    assert tr3["premise_failed"] is False and tr3["matrix_complete"] and tr3["primary_legs_not_read"] == [] == tr3["primary_legs_absent_from_matrix"]
    nine = [r for r in full if r["model_key"] != NEMO]          # a matrix carries whole triples: drop one model's
    tr4 = analyze(tmp_path / "full", rows=nine)["truncation"]
    nemo_legs = [M.leg_key(NEMO, c, PRIMARY) for c in (BARE, PROMPTED, OBJECT)]
    assert tr4["premise_failed"] is None and tr4["primary_legs_not_read"] == nemo_legs == tr4["primary_legs_absent_from_matrix"]
    assert tr4["n_primary_cells_read"] == 9 and tr4["max_primary_share"] == 0.0 and tr4["matrix_complete"] is False


def test_torn_at_cap_or_think_present_counts_refuse_by_name(tmp_path):
    """The share is read off the summary's counts and the split off the per-item
    flags: a record where the two disagree is torn and the leg is refused, never
    a share from one source read against a split from another."""
    rows = primary_rows(QWEN)

    def flip_at_cap(h):
        h["items"][0]["at_cap"] = True

    def drop_present(h):
        h["items"][3]["think_block_present"] = False

    def fake_rate(s):
        s["at_cap_rate"] = 0.5

    S.build_all(tmp_path / "a", rows=rows, overrides={(QWEN, BARE, PRIMARY): {"header_edit": flip_at_cap}})
    leg = leg_of(analyze(tmp_path / "a", rows=rows), QWEN, BARE)
    assert leg["refused"] and "n_at_cap" in leg["refused"] and "1 outcome items with at_cap" in leg["refused"]
    S.build_all(tmp_path / "b", rows=rows, overrides={(QWEN, BARE, PRIMARY): {"header_edit": drop_present}})
    leg = leg_of(analyze(tmp_path / "b", rows=rows), QWEN, BARE)
    assert leg["refused"] and "n_think_block_present" in leg["refused"] and "499 outcome items" in leg["refused"]
    S.build_all(tmp_path / "c", rows=rows, overrides={(QWEN, BARE, PRIMARY): {"summary_edit": fake_rate}})
    leg = leg_of(analyze(tmp_path / "c", rows=rows), QWEN, BARE)
    assert leg["refused"] and "at_cap_rate 0.5" in leg["refused"] and "n_at_cap / n_stop_reason" in leg["refused"]
    # a leg refused on any of these carries neither readout: the keys are absent, not zero
    assert "at_cap" not in leg and "thinking" not in leg


def test_the_eval_log_classifies_every_at_cap_draw_and_the_two_venvs_still_agree(tmp_path):
    """rows.jsonl carries both channels of every SCORED draw only, so the
    store-rows split cannot see an unscored case-(b) draw (the cap inside the
    think block) — those are ``n_text_unavailable``, never folded. The retained
    ``.eval`` log carries every draw: the log-reading run classifies all of them
    (``eval_check.at_cap_tails``) and REPORTS that split (``split_source``
    eval_log, ``n_text_unavailable`` 0), holding a scored draw's log text equal
    to its row text by digest; the run without logs reports the store rows'
    split (``split_source`` store_rows) with the unavailable count. The
    store-rows split rides alongside in both, identical; the two runs differ on
    exactly the by-design leaves, which the nested skip removes and nothing
    else — every other leaf is still held equal."""
    pytest.importorskip("inspect_ai")
    rows = primary_rows(GEMMA)
    loop, deriv = LOOP_BLOCK * 200, _derivation()
    # record_only: an unscored case-(b) draw is reasoning only (content ""); a capped scored case-(c) draw has both channels
    texts = {0: ("", loop), 1: ("", deriv), 2: ("", loop), 5: (deriv[:900], "some reasoning"), 6: (loop, "r")}
    dirs = S.build_all(tmp_path, rows=rows, overrides={(GEMMA, BARE, PRIMARY): {"unscored": {0, 1, 2}, "capped_scored": {5, 6},
                                                                                "texts": texts}})
    cell_dir = dirs[M.leg_key(GEMMA, BARE, PRIMARY)]
    log_texts = {S.IDS[j]: t for j, t in texts.items()}
    S.write_inspect_log(cell_dir, name="pass", texts=log_texts)
    e, f = analyze(tmp_path, rows=rows, read_eval_logs=True), analyze(tmp_path, rows=rows)
    leg = leg_of(e, GEMMA, BARE)
    assert leg["refused"] is None and leg["eval_check"]["available"] and leg["eval_check"]["agrees"]
    tails = leg["eval_check"]["at_cap_tails"]
    assert {k: tails[k] for k in tails if k != "tail_classes"} == {"text_source": "eval_log", "n_at_cap": 5, "n_classified": 5,
                                                                  "n_loop": 3, "n_truncated_reasoning": 2, "n_text_unavailable": 0}
    assert {i: c["loop"] for i, c in tails["tail_classes"].items()} == {S.IDS[0]: True, S.IDS[1]: False, S.IDS[2]: True,
                                                                       S.IDS[5]: False, S.IDS[6]: True}
    # the log-reading run REPORTS the log's split: every at-cap draw classified, none text-unavailable
    ac = leg["at_cap"]
    assert ac["split_source"] == "eval_log" and "every at-cap draw's text" in ac["split_source_reason"]
    assert (ac["n_loop"], ac["n_truncated_reasoning"], ac["n_text_unavailable"]) == (3, 2, 0)
    assert ac["n_case_b_unterminated_at_cap"] == 3 and ac["n_case_c_answer_at_cap"] == 2 and ac["share"] == 0.01
    # ... with the store-rows split alongside: the two scored draws classified, the three case-(b) draws unavailable
    rows_split = ac["store_rows_split"]
    assert (rows_split["n_loop"], rows_split["n_truncated_reasoning"], rows_split["n_text_unavailable"], rows_split["n_classified"]) == (1, 1, 3, 2)
    assert rows_split["tail_classes"][S.IDS[0]] == {"text_source": None, "text_sha256": None, "loop": None, "rule": None, "period": None,
                                                    "repeats": None, "cover": None, "zlib_ratio": None}
    assert rows_split["tail_classes"][S.IDS[6]]["text_sha256"] == tails["tail_classes"][S.IDS[6]]["text_sha256"]
    cell = next(c for c in e["truncation"]["cells"] if c["leg"] == leg["leg"])
    assert cell["split_source"] == "eval_log" and (cell["n_loop"], cell["n_text_unavailable"]) == (3, 0)
    assert "1 cell(s) split from the eval logs" in e["truncation"]["loop_rule_scope"] and "2 cell(s) split from the store rows" in e["truncation"]["loop_rule_scope"]
    assert not any(f_.startswith(leg["leg"]) and "1 % line" in f_ for f_ in e["findings"])     # exactly 1 % is not over the line
    # the run without logs reports the store rows' split, the unavailable count included, and the same store-rows block
    fc = leg_of(f, GEMMA, BARE)["at_cap"]
    assert leg_of(f, GEMMA, BARE)["eval_check"]["at_cap_tails"] is None and fc["split_source"] == "store_rows"
    assert (fc["n_loop"], fc["n_truncated_reasoning"], fc["n_text_unavailable"]) == (1, 1, 3) and fc["store_rows_split"] == rows_split
    assert {k: v for k, v in fc.items() if k not in M.REPORT_DIFF_SKIP_SPLIT_FIELDS + ("split_source_reason",)} == \
        {k: v for k, v in ac.items() if k not in M.REPORT_DIFF_SKIP_SPLIT_FIELDS + ("split_source_reason",)}
    # the two runs differ on EXACTLY the by-design leaves (the parent's raw comparison names them, every one covered by the
    # nested skip) and on nothing else (the analysis' comparison is equal); a moved store-rows leaf is still caught
    raw = M.T.numeric_leaf_diff(e, f)
    moved = [p for p, _a, _b in raw["differing"]] + [p for p, _a, _b in raw["type_changed"]] + raw["missing"] + raw["added"]
    assert moved and all(M._split_leaf_skipped(p) for p in moved), moved
    assert {p[-1] for p in moved} == {"n_loop", "n_truncated_reasoning", "n_text_unavailable"}
    d = M.numeric_leaf_diff(e, f)
    assert d["equal"] and d["shared_numeric"] > 1000, d
    perturbed = json.loads(json.dumps(f, default=str))
    perturbed["legs"][leg["leg"]]["at_cap"]["store_rows_split"]["n_loop"] += 1
    assert not M.numeric_leaf_diff(e, perturbed)["equal"]
    # a log whose text for a SCORED at-cap draw differs from the row's is one completion on record twice, differently
    S.write_inspect_log(cell_dir, name="pass", texts={**log_texts, S.IDS[5]: (deriv[:900] + " altered", "some reasoning")})
    torn = leg_of(analyze(tmp_path, rows=rows, read_eval_logs=True), GEMMA, BARE)
    assert torn["refused"] and S.IDS[5] in torn["refused"] and "rows.jsonl" in torn["refused"] and "at_cap" not in torn
    # a log read that lacks some at-cap draw's text leaves the store rows' split standing, and says so (the builder always
    # writes an output per sample, so the incomplete-log case is exercised on the chooser directly)
    partial = dict(fc)
    M.choose_split(partial, {**tails, "n_classified": 4, "n_text_unavailable": 1})
    assert partial["split_source"] == "store_rows" and "carried no text for 1 of 5" in partial["split_source_reason"]
    assert (partial["n_loop"], partial["n_truncated_reasoning"], partial["n_text_unavailable"]) == (1, 1, 3)
    M.choose_split(partial, None)
    assert partial["split_source"] == "store_rows"


def test_the_readout_schema_pins_the_new_keys(full_report):
    leg = leg_of(full_report, GEMMA, BARE)
    assert set(leg["at_cap"]) == AT_CAP_KEYS and set(leg["thinking"]) == THINKING_KEYS
    tr = full_report["truncation"]
    assert set(tr) == TRUNCATION_KEYS
    assert set(tr["cells"][0]) == {"leg", "model_key", "cell", "kind", "mode", "primary", "sensitivity", *M._TRUNCATION_CELL_AT_CAP_FIELDS,
                                   "n_on_rows", "n_no_reasoning_emitted", "share_no_reasoning", "no_reasoning_by_construction",
                                   "n_present_by_record", "n_present_by_channel",
                                   *M._TRUNCATION_CELL_REDRAW_FIELDS, "redraws_attributed"}
    assert set(M._TRUNCATION_CELL_AT_CAP_FIELDS) < AT_CAP_KEYS
    # the re-draw readout rides the ledger record (and its controls mirror), pinned key for key; the fixture's stores
    # have no re-send, so every count is 0 and the classification is attributed
    rd = leg["ledger"]["redraws"]
    assert set(rd) == REDRAW_KEYS and set(M._TRUNCATION_CELL_REDRAW_FIELDS) < REDRAW_KEYS
    assert (rd["n_redraw_calls"], rd["n_items_redrawn"], rd["share_items_redrawn"], rd["max_calls_per_item"],
            rd["n_items_sample_error"], rd["attributed"]) == (0, 0, 0.0, 1, 0, True)
    assert (rd["n_items_body_changed"], rd["n_final_draw_file_last_differs"], rd["final_draw_rule"]) == (0, 0, M.FINAL_DRAW_RULE_TEXT)
    assert rd["calls_per_item_hist"] == {"1": 500} and rd["ladder"]["calls_per_sample_run"] == M.LADDER_CALLS_PER_SAMPLE_RUN
    assert tr["cells"][0]["redraws_attributed"] is True and tr["cells_redrawn"] == [] and tr["n_items_redrawn_total"] == 0
    assert tr["redraw_selection_caveat"] == M.REDRAW_SELECTION_CAVEAT and tr["redraw_ladder"] == M.LADDER_TEXT
    assert set(leg["at_cap"]["store_rows_split"]) == SPLIT_KEYS and leg["at_cap"]["split_source"] == "store_rows"
    assert all(c["split_source"] == "store_rows" for c in tr["cells"]) and "24 cell(s) split from the store rows" in tr["loop_rule_scope"]
    assert leg["eval_check"]["at_cap_tails"] is None                 # log reading disabled in this fixture
    assert (tr["cap"], tr["n_primary_cells_design"]) == (M.CAP, 12) and M.CAP == C.MATH500_MAX_TOKENS_DEFAULT
    assert (M.AT_CAP_LINE_1PCT, M.AT_CAP_SURFACING_THRESHOLD, M.AT_CAP_PREMISE_RATE) == (0.01, 0.03, 0.015)
    assert tr["primary_legs_absent_from_matrix"] == [] and tr["matrix_complete"] is True
    # the controls table carries both readouts per leg, so the figures, W&B and the tables JSON read one place
    row = next(r for r in full_report["controls"] if r["leg"] == leg["leg"])
    assert row["at_cap"] == leg["at_cap"] and row["thinking"] == leg["thinking"]


def _title_lines(fig):
    title = [t for t in fig.texts if t.get_fontweight() == "bold"]
    assert len(title) == 1
    return title[0].get_text().split("\n")


def _texts_in_bounds(fig):
    r = fig.canvas.get_renderer()
    return all(t.get_window_extent(renderer=r).x1 <= fig.bbox.x1 + 0.5 and t.get_window_extent(renderer=r).x0 >= -0.5
               for ax in fig.axes for t in ax.texts)


READOUT_FIGURES = ("final_m5t_atcap_share", "final_m5t_atcap_loopsplit", "final_m5t_no_reasoning_share")


def _check_readout_figures(rep):
    """Every readout figure renders under the title rule (<= 2 lines of <= 90 characters), the axes rule and the
    text-bounds rule, whatever the counts; returns the three figures' title lines by stem."""
    import matplotlib.pyplot as plt
    lines = {}
    for stem in READOUT_FIGURES:
        fig, nums = M.FIGURES[stem](rep)
        fig.canvas.draw()
        lines[stem] = _title_lines(fig)
        assert len(lines[stem]) <= 2 and all(len(line) <= M.TITLE_MAX_CHARS for line in lines[stem]), (stem, lines[stem])
        assert M.axes_height_fraction(fig) >= M.AXES_MIN_FRACTION and _texts_in_bounds(fig), stem
        assert nums, stem
        plt.close(fig)
    return lines


def test_the_readout_figures_render_whatever_the_counts(tmp_path):
    """Regression: the loop-split figure's second title line once carried the three totals, so a realistic matrix (5 loop /
    5 truncated / 10 text-unavailable per leg x 24 legs) pushed it past the 90-character rule and the exception out of
    ``make_figures()`` lost every later figure, the sidecar, the mirror, W&B and the report JSON. The second lines are now
    fixed sentences or bounded readings (the totals travel in the legend), and the three figures render on: large mixed
    counts; maximum-width counts (thousands per class, every draw at cap); a leg with no at-cap draw; a leg whose every
    at-cap draw is text-unavailable (case b only)."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    loops = {j: (f"reasoning</think>{LOOP_BLOCK * 200}", None) for j in range(5)}
    derivs = {j: (f"reasoning</think>{_derivation(20)}", None) for j in range(5, 10)}
    # (1) the reported case: 5 loop / 5 truncated / 10 unavailable on every leg but two — one with no at-cap draw at all,
    #     one whose 20 at-cap draws are all unscored (text unavailable)
    S.build_all(tmp_path / "mixed", capped_scored=set(range(10)), unscored=set(range(10, 20)), texts={**loops, **derivs},
                overrides={(GEMMA, BARE, PRIMARY): {"capped_scored": set(), "unscored": set(), "texts": {}},
                           (GEMMA, PROMPTED, PRIMARY): {"capped_scored": set(), "unscored": set(range(20)), "texts": {}}})
    rep = analyze(tmp_path / "mixed")
    cells = rep["truncation"]["cells"]
    assert len(cells) == 24 and (sum(c["n_loop"] for c in cells), sum(c["n_truncated_reasoning"] for c in cells),
                                 sum(c["n_text_unavailable"] for c in cells)) == (110, 110, 240)
    none_at_cap = leg_of(rep, GEMMA, BARE)["at_cap"]
    all_unavailable = leg_of(rep, GEMMA, PROMPTED)["at_cap"]
    assert (none_at_cap["n_at_cap"], none_at_cap["n_loop"], none_at_cap["n_text_unavailable"]) == (0, 0, 0)
    assert (all_unavailable["n_at_cap"], all_unavailable["n_loop"], all_unavailable["n_truncated_reasoning"],
            all_unavailable["n_text_unavailable"]) == (20, 0, 0, 20)
    figs = M.make_figures(rep, tmp_path / "mixed" / "figs")
    assert set(figs["written"]) == set(M.FIGURES) and figs["skipped"] == {} and Path(figs["numbers"]).is_file()
    lines = _check_readout_figures(rep)
    assert lines["final_m5t_atcap_loopsplit"][1] == "totals in the legend; grey = unscored draws whose text the store rows do not carry"
    assert lines["final_m5t_atcap_share"][1].startswith("premise failed: 11 primary cell(s) >= 3 %; max 4.0 %")   # gemma bare has none
    # the totals are on the legend, not the title
    import matplotlib.pyplot as plt
    fig, _n = M.fig_atcap_loopsplit(rep)
    legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any("110 draws" in t and "loop" in t for t in legend) and any("240 draws" in t for t in legend)
    # ... and the legend's title names the split's source — no log was read here, so the store rows'
    assert fig.axes[0].get_legend().get_title().get_text() == "store rows only; unscored draws' text not carried"
    assert "store rows only" in M.figure_captions(rep)["final_m5t_atcap_loopsplit"]
    plt.close(fig)
    # a report whose cells were split from the eval logs says so in the legend title, the second line and the caption
    logged = json.loads(json.dumps(rep, default=str))
    for row in logged["controls"]:
        if row.get("at_cap"):
            row["at_cap"]["split_source"] = "eval_log"
    for c in logged["truncation"]["cells"]:
        c["split_source"] = "eval_log"
    fig, _n = M.fig_atcap_loopsplit(logged)
    assert fig.axes[0].get_legend().get_title().get_text() == "classified from the eval logs"
    assert _title_lines(fig)[1] == "totals in the legend; every at-cap draw classified from the eval logs (case b included)"
    assert "classified from the eval logs" in M.figure_captions(logged)["final_m5t_atcap_loopsplit"]
    plt.close(fig)
    # (2) maximum-width counts: every draw of every leg at the cap — half unscored (text unavailable), half scored and
    #     classified (thousands per class), one GLM sensitivity leg with every finished draw tag-free (a 100 % no-reasoning
    #     share and the longest label), and the premise failed on all 12 primary cells at 100 %
    rows = [r for r in S.cells_matrix() if r["sampling_mode"] == PRIMARY or (r["model_key"] == GLM and r["sampling_mode"] == BATTERY)]
    wide_texts = {**{j: (f"reasoning</think>{LOOP_BLOCK * 200}", None) for j in range(250, 255)},
                  **{j: (f"reasoning</think>{_derivation(20)}", None) for j in range(255, 500)}}
    S.build_all(tmp_path / "wide", rows=rows, capped_scored=set(range(250, 500)), unscored=set(range(250)), texts=wide_texts,
                overrides={(GLM, PROMPTED, BATTERY): {"capped_scored": set(), "unscored": set(), "texts": {}, "no_think": set(range(500))}})
    rep2 = analyze(tmp_path / "wide", rows=rows)
    cells2 = rep2["truncation"]["cells"]
    assert sum(c["n_truncated_reasoning"] for c in cells2) == 245 * 14 and sum(c["n_text_unavailable"] for c in cells2) == 250 * 14
    assert rep2["truncation"]["premise_failed"] is True and len(rep2["truncation"]["primary_at_or_above_threshold"]) == 12
    assert leg_of(rep2, GLM, PROMPTED, BATTERY)["thinking"]["share_no_reasoning"] == 1.0
    figs2 = M.make_figures(rep2, tmp_path / "wide" / "figs")
    assert set(figs2["written"]) == set(M.FIGURES)
    lines2 = _check_readout_figures(rep2)
    assert lines2["final_m5t_atcap_share"][1].startswith("premise failed: 12 primary cell(s) >= 3 %; max 100.0 %")
    assert lines2["final_m5t_no_reasoning_share"][1].startswith("max 100.0 % in glm46vflash prompted [battery]")
    # the bounded-line helper itself: a short line is untouched, an over-long one is cut to the rule, never raised on
    assert M._fit_title_line("short") == "short"
    cut = M._fit_title_line("x" * 200)
    assert len(cut) == M.TITLE_MAX_CHARS and cut.endswith("…") and M.check_title(cut) == cut


# --------------------------------------------------------------------------- #
# the multi-tree allowance — declared producer trees, verified on the          #
# repository before any leg is read; every comparison still on ONE tree        #
# --------------------------------------------------------------------------- #

PROBE_FILES = ("scripts/served_capability_run.py", "tests/test_served_render_probe_prefix.py")
OTHER_TREE = "a" * 40           # a second producer tree that does NOT carry the pin's prefix
THIRD_TREE = "b" * 40           # a tree nobody declared
OUTSIDE_FILE = "src/sisteer/capeval/tiers/think_split.py"
WHY = "three rows were produced after the driver's render-probe fix; the other rows at the primary tree"


def _fake_git(monkeypatch, *, known, diffs):
    """A stand-in for ``M._git_output``: ``rev-parse --verify --quiet <ref>^{commit}``
    resolves a known sha (or a unique prefix of one), the diff enumeration
    (``M.GIT_DIFF_NAMES_ARGS`` A B --) returns the configured file list
    NUL-terminated; anything else is unexpected."""
    calls = []

    def fake(repo_root, *args):
        calls.append(args)
        if args[:3] == ("rev-parse", "--verify", "--quiet"):
            ref = args[3].removesuffix("^{commit}")
            hits = [sha for sha in known if sha.startswith(ref)]
            if len(hits) != 1:
                raise subprocess.CalledProcessError(1, ["git", *args])
            return hits[0] + "\n"
        if args[:len(M.GIT_DIFF_NAMES_ARGS)] == M.GIT_DIFF_NAMES_ARGS and args[-1] == "--":
            return "".join(f + "\0" for f in diffs[(args[-3], args[-2])])
        raise AssertionError(f"unexpected git call {args}")

    monkeypatch.setattr(M, "_git_output", fake)
    return calls


def test_multi_tree_allowance_admits_declared_trees_and_keeps_every_comparison_on_one_tree(tmp_path, monkeypatch, capsys):
    real_git = M._git_output
    rows = primary_rows(GEMMA) + primary_rows(QWEN)
    gemma_other = {(GEMMA, c, PRIMARY): {"tree_sha": OTHER_TREE} for c in (BARE, PROMPTED, OBJECT)}
    S.build_all(tmp_path / "gen", rows=rows, overrides=gemma_other)
    # with only --sha, the second tree's legs are refused as today, and NOTHING of the allowance appears: no block, no
    # per-leg field, no table line, no W&B field — the one-tree path is the reviewed one
    rep = analyze(tmp_path / "gen", rows=rows)
    assert set(rep["unusable_legs"]) == {M.leg_key(GEMMA, c, PRIMARY) for c in (BARE, PROMPTED, OBJECT)}
    assert all(OTHER_TREE in why and S.SHA7 in why and "declared" not in why for why in rep["unusable_legs"].values())
    assert "tree_set" not in rep and not any("producer trees" in f for f in rep["findings"])
    assert "producer_tree" not in leg_of(rep, QWEN, BARE) and "declared_trees" not in leg_of(rep, QWEN, BARE)["producer"]
    M.print_table(rep)
    assert "producer trees" not in capsys.readouterr().out
    wb = _fake_wandb(monkeypatch)
    M.log_wandb(rep, {}, mode="offline")
    assert "tree_set/note" not in wb["summary"]
    # the declared set, verified through a fake git that reports exactly the allowlisted paths
    calls = _fake_git(monkeypatch, known={S.FULL_SHA, OTHER_TREE}, diffs={(S.FULL_SHA, OTHER_TREE): list(PROBE_FILES)})
    ts = M.verify_tree_set(S.SHA7, [OTHER_TREE], PROBE_FILES, repo_root=tmp_path, reason=WHY)
    assert ts["problems"] == [] and ts["primary"] == S.FULL_SHA and ts["trees"] == [S.FULL_SHA, OTHER_TREE] and ts["declared"]
    assert ts["diff_files"] == {OTHER_TREE: sorted(PROBE_FILES)} and ts["allow"] == list(PROBE_FILES) and ts["why"] == WHY
    assert (*M.GIT_DIFF_NAMES_ARGS, S.FULL_SHA, OTHER_TREE, "--") in calls
    assert M.GIT_DIFF_NAMES_ARGS == ("diff", "--no-renames", "--name-only", "-z", "--ignore-submodules=none")
    rep = analyze(tmp_path / "gen", rows=rows, tree_set=ts)
    assert rep["unusable_legs"] == {}
    assert all(c["usable"] and c["estimands"] is not None for c in rep["comparisons"])
    assert M.production_refusals(rep) == [] and rep["verdict"]["word"] is not None
    assert leg_of(rep, GEMMA, OBJECT)["producer_tree"] == OTHER_TREE and leg_of(rep, QWEN, OBJECT)["producer_tree"] == S.FULL_SHA
    assert leg_of(rep, GEMMA, OBJECT)["producer"]["declared_trees"] == [OTHER_TREE]
    assert leg_of(rep, GEMMA, OBJECT)["producer"]["matches_pin"] and leg_of(rep, GEMMA, OBJECT)["producer"]["pin"] == S.SHA7
    filed = rep["tree_set"]
    assert filed["legs"] == {**{M.leg_key(GEMMA, c, PRIMARY): OTHER_TREE for c in (BARE, PROMPTED, OBJECT)},
                             **{M.leg_key(QWEN, c, PRIMARY): S.FULL_SHA for c in (BARE, PROMPTED, OBJECT)}}
    assert filed["why"] == WHY and filed["diff_files"] == {OTHER_TREE: sorted(PROBE_FILES)} and filed["allow"] == list(PROBE_FILES)
    assert any(OTHER_TREE in f and WHY in f for f in rep["findings"])
    note = M.tree_set_note(filed)
    assert OTHER_TREE in note and S.FULL_SHA in note and PROBE_FILES[0] in note and WHY in note
    # the two-venv match reads the block as metadata: a report differing ONLY in tree_set is numerically equal
    other = json.loads(json.dumps(rep))
    other["tree_set"]["why"] = "another reason"
    other["tree_set"]["diff_files"] = {}
    assert M.numeric_leaf_diff(rep, other)["equal"]
    assert M._metadata_skipped(("tree_set",)) and not M._metadata_skipped(("legs", "tree_set"))
    # a comparison MIXING the two DECLARED trees: under a declaration the
    # one-tree rule per comparison gives way to the compatibility relations — the shared-field projection equal, every
    # arm's own contract held, every tree admitted — and the comparison RUNS with its cross-tree record filed; the
    # negative cases (a shared field moved; an unadmitted tree) are pinned in tests/test_eight_arms_math500_analysis.py
    S.build_all(tmp_path / "mixed", rows=rows, overrides={(GEMMA, OBJECT, PRIMARY): {"tree_sha": OTHER_TREE}})
    rep = analyze(tmp_path / "mixed", rows=rows, tree_set=ts)
    assert rep["unusable_legs"] == {}
    c = comparison(rep, GEMMA)
    assert c["usable"] is True and c["estimands"] is not None
    assert c["compatibility"]["cross_tree"] is True and c["compatibility"]["declared"] is True and c["compatibility"]["projection_equal"] is True
    assert c["compatibility"]["legs"]["object"]["tree"] == OTHER_TREE and c["compatibility"]["legs"]["bare"]["tree"] == S.FULL_SHA
    assert comparison(rep, QWEN)["usable"] is True and comparison(rep, QWEN)["compatibility"]["cross_tree"] is False
    assert rep["gates"]["population"]["status"] == M.GATE_PASS and rep["verdict"]["word"] is not None and not M.production_refusals(rep)
    # a leg from a tree nobody declared is refused by name, the declared trees listed
    S.build_all(tmp_path / "third", rows=rows, overrides={**gemma_other, (QWEN, BARE, PRIMARY): {"tree_sha": THIRD_TREE}})
    rep = analyze(tmp_path / "third", rows=rows, tree_set=ts)
    why = leg_of(rep, QWEN, BARE)["refused"]
    assert THIRD_TREE in why and OTHER_TREE in why and "declared" in why and leg_of(rep, QWEN, BARE)["producer_tree"] is None
    # analyze() verifies the DECLARATION itself, from its fields, and never trusts a caller's `problems: []`: an
    # incomplete declaration, a verified block whose `also` list was altered afterwards, and one for another pin are
    # all refused before any cell is loaded
    read = []
    monkeypatch.setattr(M, "collect_leg", lambda *a, **k: read.append(1))
    with pytest.raises(M.AnalysisError, match="tree_set") as e:
        analyze(tmp_path / "gen", rows=rows, tree_set={"primary_pin": S.SHA7, "also": [THIRD_TREE]})
    assert THIRD_TREE in str(e.value) and "not a commit" in str(e.value) and "--tree-set-reason" in str(e.value)
    with pytest.raises(M.AnalysisError, match="tree_set") as e:
        analyze(tmp_path / "gen", rows=rows, tree_set={**ts, "also": [OTHER_TREE, THIRD_TREE], "problems": []})
    assert THIRD_TREE in str(e.value)
    with pytest.raises(M.AnalysisError, match="tree_set") as e:
        analyze(tmp_path / "gen", rows=rows, tree_set={**ts, "allow": [], "problems": []})
    assert "allowlist []" in str(e.value)
    with pytest.raises(M.AnalysisError, match="primary pin"):
        analyze(tmp_path / "gen", rows=rows, tree_set={**ts, "primary_pin": "abcdef0"})
    # ... and with the REAL git against a directory that is no repository, the same declaration is refused too
    monkeypatch.setattr(M, "_git_output", real_git)
    with pytest.raises(M.AnalysisError, match="tree_set") as e:
        analyze(tmp_path / "gen", rows=rows, repo_root=tmp_path / "nowhere",
                tree_set={"primary_pin": S.SHA7, "also": ["f" * 40], "allow": list(PROBE_FILES), "why": WHY})
    assert "not a commit of the repository" in str(e.value)
    assert read == []


def test_tree_set_refusals_at_the_startup_boundary_before_any_leg_is_read(tmp_path, monkeypatch, capsys):
    rows = primary_rows(GEMMA)
    S.build_all(tmp_path / "gen", rows=rows, overrides={(GEMMA, c, PRIMARY): {"tree_sha": OTHER_TREE} for c in (BARE, PROMPTED, OBJECT)})
    calls = []
    real_collect, real_log_wandb = M.collect_leg, M.log_wandb
    monkeypatch.setattr(M, "make_figures", lambda *a, **k: calls.append("figs") or {})
    monkeypatch.setattr(M, "mirror_figures", lambda *a, **k: calls.append("mirror") or [])
    monkeypatch.setattr(M, "log_wandb", lambda *a, **k: calls.append("wandb"))
    monkeypatch.setattr(M, "collect_leg", lambda *a, **k: calls.append("leg") or real_collect(*a, **k))
    allow = [x for f in PROBE_FILES for x in ("--tree-diff-allow", f)]
    declared = ["--also-sha", OTHER_TREE, *allow, "--tree-set-reason", WHY]
    out = tmp_path / "report.json"

    def run(*extra, git_diff):
        _fake_git(monkeypatch, known={S.FULL_SHA, OTHER_TREE}, diffs={(S.FULL_SHA, OTHER_TREE): git_diff})
        calls.clear()
        return M.main(_main_args(tmp_path / "gen", tmp_path, "--out", str(out), *extra))

    def problems():
        return json.loads(out.read_text())["tree_set"]["problems"]

    # a diff outside the allowlist: refused by name (the path, the tree), exit 3, no leg read, no side effect
    rc = run(*declared, git_diff=[*PROBE_FILES, OUTSIDE_FILE])
    assert rc == M.EXIT_NOT_PRODUCTION == 3 and calls == []
    rep = json.loads(out.read_text())
    assert "legs" not in rep and "comparisons" not in rep and rep["verdict"]["word"] is None and rep["sha"] == S.SHA7
    [problem] = rep["tree_set"]["problems"]
    assert problem.endswith(str([OUTSIDE_FILE])) and OTHER_TREE in problem and S.FULL_SHA in problem and "allowlist" in problem
    assert rep["tree_set"]["diff_files"] == {OTHER_TREE: sorted([*PROBE_FILES, OUTSIDE_FILE])}
    assert [r for r in M.production_refusals(rep) if r.startswith("tree_set: ")] and M.main(["--gate-report", str(out)]) == 3
    assert "tree_set" in capsys.readouterr().err
    # an unknown declared sha; a malformed one; the primary itself declared again; a tree declared twice
    rc = run("--also-sha", THIRD_TREE, *allow, "--tree-set-reason", WHY, git_diff=list(PROBE_FILES))
    assert rc == 3 and calls == [] and any(THIRD_TREE in p and "not a commit" in p for p in problems())
    rc = run("--also-sha", "abc", *allow, "--tree-set-reason", WHY, git_diff=list(PROBE_FILES))
    assert rc == 3 and calls == [] and any("'abc'" in p and "40-hex" in p for p in problems())
    rc = run("--also-sha", S.FULL_SHA, *allow, "--tree-set-reason", WHY, git_diff=list(PROBE_FILES))
    assert rc == 3 and calls == [] and any("primary tree itself" in p for p in problems())
    rc = run("--also-sha", OTHER_TREE, *declared, git_diff=list(PROBE_FILES))
    assert rc == 3 and calls == [] and any("twice" in p for p in problems())
    # no reason: an argparse error before anything runs
    with pytest.raises(SystemExit) as e:
        run("--also-sha", OTHER_TREE, *allow, git_diff=list(PROBE_FILES))
    assert e.value.code == 2 and calls == []
    # an empty allowlist admits only an identical tree
    rc = run("--also-sha", OTHER_TREE, "--tree-set-reason", WHY, git_diff=list(PROBE_FILES))
    assert rc == 3 and calls == [] and any("allowlist []" in p for p in problems())
    # the control: the allowlisted diff — the legs are read, the report is a production result carrying the block
    rc = run(*declared, git_diff=list(PROBE_FILES))
    assert rc == 0 and calls == ["leg", "leg", "leg", "figs", "wandb"]
    rep = json.loads(out.read_text())
    ts = rep["tree_set"]
    assert ts["problems"] == [] and ts["primary"] == S.FULL_SHA and ts["also"] == [OTHER_TREE] and ts["why"] == WHY
    assert ts["diff_files"] == {OTHER_TREE: sorted(PROBE_FILES)} and ts["allow"] == list(PROBE_FILES) and ts["trees"] == [S.FULL_SHA, OTHER_TREE]
    assert ts["legs"] == {M.leg_key(GEMMA, c, PRIMARY): OTHER_TREE for c in (BARE, PROMPTED, OBJECT)}
    assert M.production_refusals(rep) == [] and rep["verdict"]["word"] is not None and M.main(["--gate-report", str(out)]) == 0
    table = capsys.readouterr().out
    assert "producer trees: primary" in table and OTHER_TREE in table and WHY in table
    # the W&B summary carries the same one-line note (the mock from the wandb test)
    wb = _fake_wandb(monkeypatch)
    real_log_wandb(rep, {}, mode="offline")
    assert wb["summary"]["tree_set/note"] == M.tree_set_note(ts) and OTHER_TREE in wb["summary"]["tree_set/note"]
    # the glob form of the allowlist admits the same diff
    rc = run("--also-sha", OTHER_TREE, "--tree-diff-allow", "scripts/served_capability_run.py",
             "--tree-diff-allow", "tests/test_served_render_probe_*.py", "--tree-set-reason", WHY, git_diff=list(PROBE_FILES))
    assert rc == 0


def test_verify_tree_set_reads_the_repository_itself(tmp_path):
    """The default git seam against a real repository: commits differing in exactly
    the probe file and its test are admitted, one that also touches a scoring
    file is refused by name, and an unknown sha (declared or primary) is refused."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "GIT_CONFIG_GLOBAL": str(tmp_path / "no-config"),
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}

    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env).stdout.strip()

    def commit(msg, **files):
        for f, text in files.items():
            (repo / f).parent.mkdir(parents=True, exist_ok=True)
            (repo / f).write_text(text)
        git("add", "-A")
        git("commit", "-q", "--no-verify", "--no-gpg-sign", "-m", msg)
        return git("rev-parse", "HEAD")

    git("init", "-q")
    a = commit("one", **{f: "v1\n" for f in (*PROBE_FILES, OUTSIDE_FILE)})
    b = commit("two", **{f: "v2\n" for f in PROBE_FILES})
    c = commit("three", **{OUTSIDE_FILE: "v2\n"})
    ts = M.verify_tree_set(a[:7], [b], PROBE_FILES, repo_root=repo, reason=WHY)
    assert ts["problems"] == [] and ts["primary"] == a and ts["diff_files"] == {b: sorted(PROBE_FILES)} and ts["trees"] == [a, b]
    ts = M.verify_tree_set(a[:7], [c], PROBE_FILES, repo_root=repo, reason=WHY)
    [p] = ts["problems"]
    assert p.endswith(str([OUTSIDE_FILE])) and c in p and ts["diff_files"] == {c: sorted([*PROBE_FILES, OUTSIDE_FILE])}
    ts = M.verify_tree_set(a[:7], [b], ["scripts/*"], repo_root=repo, reason=WHY)          # the test file is outside the glob
    [p] = ts["problems"]
    assert p.endswith(str([PROBE_FILES[1]]))
    ts = M.verify_tree_set(a[:7], ["f" * 40], PROBE_FILES, repo_root=repo, reason=WHY)
    assert ts["primary"] == a and any("f" * 40 in p and "not a commit" in p for p in ts["problems"])
    ts = M.verify_tree_set("0" * 7, [b], PROBE_FILES, repo_root=repo, reason=WHY)
    assert ts["primary"] is None and any("0" * 7 in p and "not a commit" in p for p in ts["problems"])
    ts = M.verify_tree_set(a[:7], [b], PROBE_FILES, repo_root=repo, reason="  ")
    assert any("--tree-set-reason" in p for p in ts["problems"])
    # rename detection would HIDE a removed path: an unallowlisted file renamed INTO an allowlisted destination shows up
    # in git's own `--name-only` as the destination alone; the verifier enumerates with renames OFF and refuses by name
    hidden, dest = "NOTES.md", "tests/test_served_render_probe_extra.py"
    d = commit("four", **{hidden: "notes that must not vanish\n"})
    git("mv", hidden, dest)
    e = commit("five")
    assert git("diff", "--name-only", d, e) == dest                                        # the rename, as git reports it
    ts = M.verify_tree_set(d[:7], [e], ["tests/test_served_render_probe_*.py"], repo_root=repo, reason=WHY)
    [p] = ts["problems"]
    assert p.endswith(str([hidden])) and ts["diff_files"] == {e: [hidden, dest]}
    # without a declared tree nothing is asked of git — a directory that is no repository is fine
    ts = M.verify_tree_set(S.SHA7, repo_root=tmp_path / "nowhere")
    assert ts["problems"] == [] and ts["declared"] is False and ts["primary"] is None


# --------------------------------------------------------------------------- #
# the harness's same-cap re-sends, its retry ladder, the .eval log             #
# cross-check of the ledger, and the per-item outcome-field check              #
# --------------------------------------------------------------------------- #

#: the wave's real qwen object cell at producer tree 928a944d: 507 ledger calls over 500 samples,
#: 7 re-sent once at the cap. Row 5 was RE-RUN on 55e3cbdd, so the live `pull_<sha7>` holds the re-run (500 calls,
#: no re-send) and this store lives on the cluster renamed aside as the `*_928a944d-censored` twin (its
#: draws were re-sent under the SDK's 600-s timeout), excluded by `pull_wave.sh`. It is pulled ALONE, under the live mode
#: name so the analysis's cell resolution is unchanged, into a twins root; absent -> the test SKIPS naming the command.
REAL_TWINS_ROOT = SITE.expand_path("$GRADE/tinycap/math500/pull_928a944_twins")
REAL_QWEN_OBJECT = REAL_TWINS_ROOT / "controls" / "qwen35_4b" / "constv2-mix-full--lora_r4_ce_minus_ctl" / "thinking_card"
REAL_QWEN_OBJECT_PIN = "928a944"
#: the pull command's shape (the skip message prints it): the same rsync flags and excludes as `pull_wave.sh`, the source the
#: wave root on the cluster — `HOST:<wave root>` comes from the environment (`M5T_PULL_SRC`, the launcher's convention: no
#: host name and no home path live in the repository) — and the retired twin's directory name as its tail
REAL_QWEN_OBJECT_PULL_CMD = (
    f"mkdir -p {REAL_QWEN_OBJECT} && rsync -az --prune-empty-dirs --exclude='inspect_evals_cache' --exclude='serving' "
    "\"${M5T_PULL_SRC:?HOST:<wave root>}\"/controls/qwen35_4b/constv2-mix-full--lora_r4_ce_minus_ctl/"
    f"thinking_card_928a944d-censored/ {REAL_QWEN_OBJECT}/")
SEVEN_REDRAWS = {j: 1 for j in range(7)}


def test_the_retry_ladder_is_read_from_its_sources_never_typed():
    """The bound on same-cap re-sends per sample per recorded run is the product
    of three retry layers, each read from its owner: the openai SDK's default
    request retries, contracts' Inspect model-call retries, and the harness's
    ``retry_on_error`` literal read off ``inspect_served.run_one_task``'s source."""
    import inspect as pyinspect
    import re

    import openai

    from sisteer.capeval.tiers import inspect_served as isv
    assert M.SDK_MAX_RETRIES == openai.DEFAULT_MAX_RETRIES == 2
    assert M.INSPECT_MAX_RETRIES == C.INSPECT_MAX_RETRIES == 2
    literal = re.findall(r"retry_on_error\s*=\s*(\d+)\s*,", pyinspect.getsource(isv.run_one_task))
    assert literal == ["2"] and M.HARNESS_RETRY_ON_ERROR == 2
    assert M.LADDER_CALLS_PER_ATTEMPT == 9 and M.LADDER_CALLS_PER_SAMPLE_RUN == 27
    assert "= 27 calls" in M.LADDER_TEXT and "one attempt at most 9" in M.LADDER_TEXT
    assert "600 s" in M.REDRAW_MECHANISM_TEXT and "FINAL draws" in M.REDRAW_SELECTION_CAVEAT
    assert "body_sha256" in M.REDRAW_MECHANISM_TEXT and "latest request-start" in M.FINAL_DRAW_RULE_TEXT
    # the failed-attempt classes: the SDK's timeout error name and the provider's refusal prefix, read from their owners
    assert M.SDK_TIMEOUT_ERROR_NAME == openai.APITimeoutError.__name__ == "APITimeoutError"
    assert M.PROVIDER_REFUSAL_MARKER == "tinycap provider:" and pyinspect.getsource(isv).count(M.PROVIDER_REFUSAL_MARKER) >= 2
    src = pyinspect.getsource(isv)
    assert all(m in src for m in M.PROVIDER_PRE_WIRE_MARKERS) and all(m in src for m in M.PROVIDER_POST_RESPONSE_MARKERS)
    assert "on {self.route.virtual_model!r}" in src                 # on_response's `where` — the post-response shape
    assert M.classify_failed_attempt(S.FAILED_ATTEMPT_TIMEOUT_MESSAGE) == M.FAILED_ATTEMPT_TIMEOUT_LADDER
    assert M.classify_failed_attempt("Request timed out.") == M.FAILED_ATTEMPT_TIMEOUT_LADDER
    # the neither-channel refusal and a stamp refusal are raised AFTER a response: bounded, never exactly one call
    assert M.classify_failed_attempt(S.FAILED_ATTEMPT_REFUSAL_MESSAGE) == M.FAILED_ATTEMPT_PROVIDER_REFUSAL_POST_RESPONSE
    assert M.classify_failed_attempt("tinycap provider: inspect/math item 'math_1' sample 0 on 'petri__base': stamp model mismatch") == \
        M.FAILED_ATTEMPT_PROVIDER_REFUSAL_POST_RESPONSE
    assert M.classify_failed_attempt(S.FAILED_ATTEMPT_PRE_WIRE_MESSAGE) == M.FAILED_ATTEMPT_PROVIDER_REFUSAL_PRE_WIRE
    assert M.classify_failed_attempt("tinycap provider: extra_body sets stream — the proxy 400s it") == M.FAILED_ATTEMPT_PROVIDER_REFUSAL_PRE_WIRE
    # the unmapped-task refusal is raised on both sides of the wire: a provider refusal this analysis cannot place
    assert M.classify_failed_attempt("tinycap provider: request for task 'x' (item 'i' sample 0) has no arm in this run's per-task "
                                     "thinking map") == M.FAILED_ATTEMPT_PROVIDER_REFUSAL_UNPLACED
    assert all(M.classify_failed_attempt(m) == M.FAILED_ATTEMPT_UNVERIFIABLE for m in ("RuntimeError: boom", "", None))
    assert M.failed_attempt_bounds(M.FAILED_ATTEMPT_TIMEOUT_LADDER) == (9, 9)
    assert M.failed_attempt_bounds(M.FAILED_ATTEMPT_PROVIDER_REFUSAL_PRE_WIRE) == (0, 0)
    assert M.failed_attempt_bounds(M.FAILED_ATTEMPT_PROVIDER_REFUSAL_POST_RESPONSE) == (1, 9)
    assert M.failed_attempt_bounds(M.FAILED_ATTEMPT_PROVIDER_REFUSAL_UNPLACED) == (0, 9) == M.failed_attempt_bounds(M.FAILED_ATTEMPT_UNVERIFIABLE)
    # the retained tail can contradict the ladder's exact count (a last model call that did not exhaust the SDK's retries)
    # but never establish a bounded one
    ladder = SimpleNamespace(message=S.FAILED_ATTEMPT_TIMEOUT_MESSAGE, events=[SimpleNamespace(event="model", retries=2, error="Request timed out.")])
    assert M.failed_attempt_detail(ladder) == {"class": M.FAILED_ATTEMPT_TIMEOUT_LADDER, "low": 9, "high": 9, "tail_retries": 2, "note": None}
    short = SimpleNamespace(message=S.FAILED_ATTEMPT_TIMEOUT_MESSAGE, events=[SimpleNamespace(event="model", retries=1, error="Request timed out.")])
    d = M.failed_attempt_detail(short)
    assert (d["class"], d["low"], d["high"], d["tail_retries"]) == (M.FAILED_ATTEMPT_UNVERIFIABLE, 0, 9, 1) and "not established" in d["note"]
    refusal = SimpleNamespace(message=S.FAILED_ATTEMPT_REFUSAL_MESSAGE, events=[SimpleNamespace(event="model", retries=1, error="tinycap provider: x")])
    assert M.failed_attempt_detail(refusal) == {"class": M.FAILED_ATTEMPT_PROVIDER_REFUSAL_POST_RESPONSE, "low": 1, "high": 9,
                                                 "tail_retries": 1, "note": None}
    assert M.failed_attempt_detail(SimpleNamespace(message=S.FAILED_ATTEMPT_TIMEOUT_MESSAGE, events=[]))["low"] == 9
    # the outcome-defining fields the per-item log check compares, lane A's list
    assert M.OUTCOME_FIELDS_FROM_LOG == ("stop_reason", "at_cap", "completion_tokens", "prompt_tokens", "think_block_present",
                                         "think_block_unterminated", "stray_open_tag", "content_empty_with_reasoning")


def _redraw_readout(rd):
    return {k: rd[k] for k in ("n_redraw_calls", "n_items_redrawn", "share_items_redrawn", "max_calls_per_item",
                               "n_items_sample_error", "attributed", "n_resume_calls", "n_unexplained_calls",
                               "n_items_unexplained", "n_over_ladder", "n_runs_recorded")}


def test_ledger_gate_attributes_same_cap_resends_and_fails_unexplained_duplicates(tmp_path):
    """A store with seven same-cap re-sends and ONE recorded run passes the gate
    with the exact readout (the real qwen object cell's shape); a duplicate at
    another cap or a rejected duplicate is UNEXPLAINED and fails; more calls in
    one run than the ladder can make fails naming the bound; exactly the bound
    passes, and a count past one attempt's ladder implies a failed sample
    attempt (the ledger's lower bound)."""
    rows = primary_rows(QWEN)
    key = M.leg_key(QWEN, OBJECT, PRIMARY)
    S.build_all(tmp_path / "ok", rows=rows, overrides={(QWEN, OBJECT, PRIMARY): {"redraws": SEVEN_REDRAWS}})
    rep = analyze(tmp_path / "ok", rows=rows)
    assert rep["gates"]["ledger"]["status"] == M.GATE_PASS and rep["gates"]["identity"]["status"] == M.GATE_PASS
    led = leg_of(rep, QWEN, OBJECT)["ledger"]
    assert led["passed"] and led["n_calls"] == 507 and led["n_unique_samples"] == 500 and led["n_resume_rows"] == 7
    rd = led["redraws"]
    assert set(rd) == REDRAW_KEYS
    assert _redraw_readout(rd) == {"n_redraw_calls": 7, "n_items_redrawn": 7, "share_items_redrawn": 0.014, "max_calls_per_item": 2,
                                   "n_items_sample_error": 0, "attributed": True, "n_resume_calls": 0, "n_unexplained_calls": 0,
                                   "n_items_unexplained": 0, "n_over_ladder": 0, "n_runs_recorded": 1}
    assert rd["calls_per_item_hist"] == {"1": 493, "2": 7} and rd["attribution_source"].startswith("usage ledger (the classification rule alone")
    assert rd["ladder"] == {"sdk_max_retries": 2, "inspect_max_retries": 2, "retry_on_error": 2, "calls_per_attempt": 9,
                            "calls_per_sample_run": 27, "text": M.LADDER_TEXT}
    # the readout rides the truncation cell beside the at-cap share and the controls mirror; the finding names it; the
    # figure carries the re-drawn share as its second series with a resolvable sidecar record
    cell = next(c for c in rep["truncation"]["cells"] if c["leg"] == key)
    assert (cell["n_redraw_calls"], cell["n_items_redrawn"], cell["share_items_redrawn"], cell["max_calls_per_item"],
            cell["n_items_sample_error"], cell["redraws_attributed"]) == (7, 7, 0.014, 2, 0, True)
    assert rep["truncation"]["cells_redrawn"] == [key] and rep["truncation"]["n_items_redrawn_total"] == 7
    assert rep["truncation"]["n_redraw_calls_total"] == 7 and rep["truncation"]["max_calls_per_item"] == 2
    assert any(f.startswith(key) and "7 same-cap re-send(s) over 7 item(s) (1.4 % of items; max 2 calls on one item" in f
               for f in rep["findings"])
    assert next(r for r in rep["controls"] if r["leg"] == key)["ledger"]["redraws"] == rd
    assert M.production_refusals(rep) == []
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _check_readout_figures(rep)
    fig, nums = M.FIGURES["final_m5t_atcap_share"](rep)
    plt.close(fig)
    rec = next(n for n in nums if n["name"] == f"redraws.share_items_redrawn[{key}]")
    assert rec["value"] == 0.014 and M.report_path(rep, rec["source"]) == 0.014
    assert M.REDRAW_SELECTION_CAVEAT in M.figure_captions(rep)["final_m5t_atcap_share"]
    fig, nums = M.FIGURES["final_m5t_ledger"](rep)
    plt.close(fig)
    assert next(n for n in nums if n["name"] == f"ledger.redraws.n_redraw_calls[{key}]")["value"] == 7

    def other_cap(rs):
        dup = [r for r in rs if r["task"] == S.TASK and r["sample"] == C.sample_header_value(S.IDS[0], 0)]
        assert len(dup) == 2
        dup[0]["max_tokens_requested"] = 4096

    g = _one_leg_report(tmp_path, "cap", QWEN, OBJECT, overrides={"redraws": {0: 1}, "ledger_edit": other_cap})["gates"]["ledger"]
    assert g["status"] == M.GATE_FAIL and "no recorded resume or same-cap re-send accounts for" in g["detail"]
    assert "max_tokens_requested" in g["detail"]
    rd = g["legs"][key]["redraws"]
    assert (rd["n_unexplained_calls"], rd["n_items_unexplained"], rd["n_redraw_calls"], rd["n_items_redrawn"], rd["attributed"]) == (1, 1, 0, 0, False)

    def rejected_dup(rs):
        dup = [r for r in rs if r["task"] == S.TASK and r["sample"] == C.sample_header_value(S.IDS[0], 0)]
        dup[0].update(rejected=True, status=504, error_class="server_error", finish_reason=None)

    g = _one_leg_report(tmp_path, "rej", QWEN, OBJECT, overrides={"redraws": {0: 1}, "ledger_edit": rejected_dup})["gates"]["ledger"]
    assert g["status"] == M.GATE_FAIL and "censored" in g["detail"] and "no recorded resume or same-cap re-send" in g["detail"]
    assert g["legs"][key]["redraws"]["n_unexplained_calls"] == 1 and g["legs"][key]["redraws"]["attributed"] is False

    bound = M.LADDER_CALLS_PER_SAMPLE_RUN
    g = _one_leg_report(tmp_path, "over", QWEN, OBJECT, overrides={"redraws": {3: bound}})["gates"]["ledger"]
    assert g["status"] == M.GATE_FAIL and "more calls in one recorded run than the retry ladder can make" in g["detail"]
    assert f"= {bound} calls" in g["detail"] and f"('{C.sample_header_value(S.IDS[3], 0)}', {bound + 1})" in g["detail"]
    rd = g["legs"][key]["redraws"]
    assert (rd["n_over_ladder"], rd["max_calls_per_item"], rd["attributed"], rd["n_items_sample_error"]) == (1, bound + 1, False, 1)
    assert rd["n_redraw_calls"] == 0 and rd["n_unexplained_calls"] == bound
    rep = _one_leg_report(tmp_path, "edge", QWEN, OBJECT, overrides={"redraws": {3: bound - 1, 4: M.LADDER_CALLS_PER_ATTEMPT}})
    rd = leg_of(rep, QWEN, OBJECT)["ledger"]["redraws"]
    assert rep["gates"]["ledger"]["status"] == M.GATE_PASS and rd["attributed"]
    assert (rd["max_calls_per_item"], rd["n_items_sample_error"], rd["n_items_redrawn"], rd["n_redraw_calls"]) == (bound, 2, 2, bound - 1 + M.LADDER_CALLS_PER_ATTEMPT)
    assert rd["calls_per_item_hist"] == {"1": 498, str(M.LADDER_CALLS_PER_ATTEMPT + 1): 1, str(bound): 1}


def test_the_log_reading_run_cross_checks_the_ledger_against_the_logs_model_events(tmp_path):
    """Run E holds the ledger's calls per sample to the .eval log's record —
    ``sum(1 + retries)`` over the final attempt's model events plus one
    attempt's ladder per failed attempt — and REFUSES the leg on disagreement;
    the cross-check's numbers live under ``eval_check`` and the numeric readout
    is identical to run F's, so the two-venv match still holds with re-sends."""
    pytest.importorskip("inspect_ai")
    rows = primary_rows(QWEN)
    key = M.leg_key(QWEN, OBJECT, PRIMARY)
    dirs = S.build_all(tmp_path, rows=rows, overrides={(QWEN, OBJECT, PRIMARY): {"redraws": {**SEVEN_REDRAWS, 7: M.LADDER_CALLS_PER_ATTEMPT}}})
    cell_dir = dirs[key]
    # the log Inspect writes for this ledger: seven samples with one model-call retry each, one sample whose first
    # attempt failed (its nine calls dropped from the log) and whose second attempt succeeded on the first call
    S.write_inspect_log(cell_dir, name="pass", model_retries={S.IDS[j]: [1] for j in range(7)}, failed_attempts={S.IDS[7]: 1})
    e = analyze(tmp_path, rows=rows, read_eval_logs=True)
    leg = leg_of(e, QWEN, OBJECT)
    assert leg["refused"] is None and leg["eval_check"]["available"] and leg["eval_check"]["agrees"]
    rd = leg["ledger"]["redraws"]
    assert (rd["n_redraw_calls"], rd["n_items_redrawn"], rd["n_items_sample_error"], rd["max_calls_per_item"], rd["attributed"]) == \
        (7 + M.LADDER_CALLS_PER_ATTEMPT, 8, 1, M.LADDER_CALLS_PER_ATTEMPT + 1, True)
    assert rd["attribution_source"].startswith("usage ledger, cross-checked per sample against the .eval log")
    assert "log_check" not in rd                       # the comparison's numbers moved under eval_check
    lc = leg["eval_check"]["redraws"]
    assert (lc["n_items"], lc["n_model_retries"], lc["n_failed_attempts"], lc["n_items_with_failed_attempts"], lc["n_generate_calls"],
            lc["n_items_errored"]) == (500, 7, 1, 1, 500, 0)
    assert set(lc["per_item"]) == set(S.IDS[:8]) and lc["per_item"][S.IDS[7]] == {
        "final_calls": 1, "model_events": 1, "model_retries": 0, "failed_attempts": 1,
        "failed_attempt_classes": [M.FAILED_ATTEMPT_TIMEOUT_LADDER], "failed_attempt_bounds": [[9, 9]], "failed_attempt_notes": [],
        "errored": False}
    assert set(lc["ledger_check"]) == LEDGER_CHECK_KEYS
    assert lc["ledger_check"]["agrees"] and lc["ledger_check"]["n_items_compared"] == 500 and lc["ledger_check"]["run"] == 1
    assert lc["ledger_check"]["n_mismatched"] == 0 and lc["ledger_check"]["n_model_retries"] == 7
    assert lc["ledger_check"]["n_failed_attempts_by_class"] == _class_counts(timeout_ladder=1)
    assert (lc["ledger_check"]["n_failed_attempts_exact"], lc["ledger_check"]["n_failed_attempts_bounded"],
            lc["ledger_check"]["unverifiable_items"]) == (1, 0, [])
    assert lc["per_item"][S.IDS[7]]["failed_attempt_bounds"] == [[9, 9]] and lc["per_item"][S.IDS[7]]["failed_attempt_notes"] == []
    assert lc["ledger_check"]["final_draw_selection"] == {"rule": M.FINAL_DRAW_RULE_TEXT, "n_latest_start_disagreements": 0,
                                                         "n_file_last_disagreements": 0}
    assert e["gates"]["ledger"]["status"] == M.GATE_PASS, e["gates"]["ledger"]["detail"]
    # run F: the ledger rule alone, the same numeric readout, another attribution source; every numeric leaf matches
    f = analyze(tmp_path, rows=rows)
    fl = leg_of(f, QWEN, OBJECT)
    assert fl["eval_check"]["redraws"] is None and fl["ledger"]["redraws"]["attribution_source"].startswith("usage ledger (the classification rule alone")
    assert {k: v for k, v in fl["ledger"]["redraws"].items() if k != "attribution_source"} == \
        {k: v for k, v in rd.items() if k != "attribution_source"}
    d = M.numeric_leaf_diff(e, f)
    assert d["equal"] and d["shared_numeric"] > 1000, d
    assert M.T.numeric_leaf_diff(e, f)["equal"]         # even before the nested skip: the log's numbers live under eval_check
    # a log whose model events do not account for a ledger call REFUSES the leg, naming the item
    S.write_inspect_log(cell_dir, name="pass", model_retries={S.IDS[j]: [1] for j in range(6)}, failed_attempts={S.IDS[7]: 1})
    bad = leg_of(analyze(tmp_path, rows=rows, read_eval_logs=True), QWEN, OBJECT)
    assert bad["refused"] and "disagree with the .eval log's model events for 1 item(s) in recorded run 1" in bad["refused"]
    assert S.IDS[6] in bad["refused"] and "a re-send the log cannot account for" in bad["refused"]
    mm = bad["eval_check"]["redraws"]["ledger_check"]["mismatches"]
    assert bad["eval_check"]["redraws"]["ledger_check"]["agrees"] is False and mm == [
        {"item_id": S.IDS[6], "ledger_calls": 2, "log_calls": 1, "final_calls": 1, "failed_attempt_classes": [],
         "why": "the ledger's calls are not the log's final-attempt calls plus what each failed attempt's class implies"}]
    assert "at_cap" not in bad                          # refused before any readout
    # a failed attempt the log records but the ledger never saw refuses too
    S.write_inspect_log(cell_dir, name="pass", model_retries={S.IDS[j]: [1] for j in range(7)}, failed_attempts={S.IDS[7]: 1, S.IDS[9]: 1})
    bad = leg_of(analyze(tmp_path, rows=rows, read_eval_logs=True), QWEN, OBJECT)
    assert bad["refused"] and S.IDS[9] in bad["refused"] and "1 item(s)" in bad["refused"]
    # several model events in one attempt (a call exhausted its retries, the next call succeeded) are summed
    dirs2 = S.build_all(tmp_path / "ev", rows=rows, overrides={(QWEN, OBJECT, PRIMARY): {"redraws": {0: 3}}})
    S.write_inspect_log(dirs2[key], name="pass", model_retries={S.IDS[0]: [2, 0]})
    leg2 = leg_of(analyze(tmp_path / "ev", rows=rows, read_eval_logs=True), QWEN, OBJECT)
    assert leg2["refused"] is None
    assert leg2["eval_check"]["redraws"]["per_item"] == {S.IDS[0]: {"final_calls": 4, "model_events": 2, "model_retries": 2,
                                                                   "failed_attempts": 0, "failed_attempt_classes": [],
                                                                   "failed_attempt_bounds": [], "failed_attempt_notes": [], "errored": False}}
    assert leg2["ledger"]["redraws"]["n_redraw_calls"] == 3 and leg2["ledger"]["redraws"]["n_items_sample_error"] == 0


def test_the_per_item_log_check_fails_a_count_preserving_outcome_swap(tmp_path):
    """Swapping ``at_cap`` / ``stop_reason`` / ``completion_tokens`` between a
    capped and a finished scored item keeps every count and every score, so the
    parent's score-only check passes the swapped record; the outcome-field check
    fails it on both items, field by field, and the leg is refused."""
    pytest.importorskip("inspect_ai")
    from inspect_ai.log import read_eval_log
    rows = primary_rows(QWEN)
    key = M.leg_key(QWEN, OBJECT, PRIMARY)
    # the capped draw's text is on record twice (the row and the log): the same text in both, as the producer files it
    texts = {5: ("the answer is 7", "a derivation")}
    dirs = S.build_all(tmp_path, rows=rows, overrides={(QWEN, OBJECT, PRIMARY): {"capped_scored": {5}, "texts": texts}})
    cell_dir = dirs[key]
    log_path = S.write_inspect_log(cell_dir, name="pass", texts={S.IDS[5]: texts[5]})
    before = leg_of(analyze(tmp_path, rows=rows, read_eval_logs=True), QWEN, OBJECT)
    assert before["refused"] is None and before["eval_check"]["per_item"]["fields_compared"] == list(M.OUTCOME_FIELDS_FROM_LOG)
    assert before["eval_check"]["per_item"]["n_field_mismatched"] == 0
    p = cell_dir / C.TASK_OUTCOMES_DIR / C.task_outcomes_filename(S.TASK)
    doc = json.loads(p.read_text())
    by = {it["item_id"]: it for it in doc["items"]}
    a, b = by[S.IDS[5]], by[S.IDS[6]]
    assert a["at_cap"] and not b["at_cap"] and a["scored"] and b["scored"] and a["completion_tokens"] != b["completion_tokens"]
    for f in ("at_cap", "stop_reason", "completion_tokens"):
        a[f], b[f] = b[f], a[f]
    p.write_text(json.dumps(doc))
    results = M.isv.results_from_log(read_eval_log(str(log_path)), sample_idx=0, primary_scorer=M.isv.PRIMARY_SCORER[M.TASK_NAME])
    assert M.T.per_item_log_check(results, doc["items"])["agrees"]          # the parent's check cannot see the swap
    per = M.per_item_outcome_check(results, doc["items"])
    assert not per["agrees"] and per["n_field_mismatched"] == 6 and per["n_mismatched"] == 0
    assert {(m["item_id"], m["field"]) for m in per["field_mismatches"]} == \
        {(i, f) for i in (S.IDS[5], S.IDS[6]) for f in ("at_cap", "stop_reason", "completion_tokens")}
    detail = M.per_item_outcome_detail(per)
    assert "6 outcome-defining field(s) differ" in detail and f"{S.IDS[5]}/0.at_cap: log True != record False" in detail
    leg = leg_of(analyze(tmp_path, rows=rows, read_eval_logs=True), QWEN, OBJECT)
    assert leg["refused"] and "outcome-defining field(s) differ" in leg["refused"] and f"{S.IDS[5]}/0.at_cap" in leg["refused"]
    assert leg["eval_check"]["agrees"] is False and leg["eval_check"]["per_item"]["n_field_mismatched"] == 6


def test_the_real_qwen_object_cell_passes_the_ledger_gate_with_seven_redraws():
    """The wave's real qwen object cell: 507 calls over 500 samples, seven
    re-sent once at the cap, every call HTTP 200, one recorded run — the gate
    passes with ``n_items_redrawn == 7``; in the log-reading venv the .eval
    log's seven model-call retries account for exactly those calls and the
    outcome-field comparison agrees on all 500 draws."""
    if not REAL_QWEN_OBJECT.is_dir():
        pytest.skip(f"the 928a944d qwen object twin is not pulled at {REAL_QWEN_OBJECT}; pull it (read-only) with: "
                    f"{REAL_QWEN_OBJECT_PULL_CMD}")
    cd, why = M.T._cell_dir(REAL_QWEN_OBJECT)
    assert cd is not None, why
    manifest = completion.read_store_manifest(cd)
    # the pin: this is the 928a944d store (the live pull's row 5 is the 55e3cbdd re-run: 500 calls, no re-send)
    assert str(manifest["producer"]["tree_sha"]).startswith(REAL_QWEN_OBJECT_PIN), manifest["producer"]["tree_sha"]
    summary = completion.read_task_summary(cd, M.TASK)
    outcomes = completion.read_task_outcomes(cd, M.TASK)
    header = {k: v for k, v in outcomes.items() if k != "items"}
    ledger_path = REAL_QWEN_OBJECT / C.LEDGER_FILENAME
    ledger_rows = list(M._iter_jsonl(ledger_path))
    route, tinycap = manifest[C.MANIFEST_KEY_ROUTE], manifest[C.MANIFEST_KEY_TINYCAP]
    kw = {"rows": ledger_rows, "header": header, "summary": summary, "attempted_ids": sorted(it["item_id"] for it in outcomes["items"]),
          "vmodel": str(route["virtual_model"]), "max_model_len": tinycap["served"].get("max_model_len"), "manifest": manifest}
    led = M.ledger_confirmation(ledger_path, **kw)
    assert led["passed"], led["failures"]
    assert (led["n_calls"], led["n_unique_samples"], led["n_rejected"], led["n_resume_rows"], led["resume_attempts"]) == (507, 500, 0, 7, 1)
    rd = led["redraws"]
    assert (rd["n_items_redrawn"], rd["n_redraw_calls"], rd["max_calls_per_item"], rd["n_items_sample_error"], rd["attributed"],
            rd["n_unexplained_calls"], rd["n_resume_calls"], rd["share_items_redrawn"]) == (7, 7, 2, 0, True, 0, 0, 0.014)
    assert rd["calls_per_item_hist"] == {"1": 493, "2": 7}
    assert rd["n_items_body_changed"] == 0 and rd["n_final_draw_file_last_differs"] == 0 and led["n_length"] == summary["n_at_cap"]
    if importlib.util.find_spec("inspect_ai") is None:
        return
    cap = int(tinycap[C.IDENTITY_KEY_MAX_TOKENS_BY_TASK][M.TASK])
    ec = M.eval_cross_check(cd, header, outcomes["items"], max_tokens=cap)
    assert ec["available"] and ec["agrees"], ec.get("detail")
    assert ec["per_item"]["n_field_mismatched"] == 0 and ec["per_item"]["n_compared"] == 500
    assert (ec["redraws"]["n_items"], ec["redraws"]["n_model_retries"], ec["redraws"]["n_failed_attempts"], ec["redraws"]["n_generate_calls"]) == (500, 7, 0, 500)
    assert len(ec["redraws"]["per_item"]) == 7 and all(v == {"final_calls": 2, "model_events": 1, "model_retries": 1, "failed_attempts": 0,
                                                            "failed_attempt_classes": [], "failed_attempt_bounds": [],
                                                            "failed_attempt_notes": [], "errored": False}
                                                      for v in ec["redraws"]["per_item"].values())
    assert ec["redraws"]["n_at_cap"] == summary["n_at_cap"] == 179
    led2 = M.ledger_confirmation(ledger_path, **kw, log_attribution=ec["redraws"])
    lc = led2["redraws"]["log_check"]
    assert led2["passed"] and lc["agrees"] and lc["n_items_compared"] == 500 and set(lc) == LEDGER_CHECK_KEYS
    # the final-draw selection on the real cell: the latest-start row agrees with the log's at-cap bit on every sample (here
    # the file's last row does too — qwen's abandoned requests finished before their replacements)
    assert lc["final_draw_selection"] == {"rule": M.FINAL_DRAW_RULE_TEXT, "n_latest_start_disagreements": 0, "n_file_last_disagreements": 0}
    assert lc["n_failed_attempts_by_class"] == _class_counts() and (lc["n_failed_attempts_exact"], lc["n_failed_attempts_bounded"]) == (0, 0)
    resent = Counter(r["sample"] for r in ledger_rows if r["task"] == M.TASK)
    assert set(ec["redraws"]["per_item"]) == {M._sample_item(s) for s, n in resent.items() if n > 1}


def _phantom_attempts(attempts, item_id):
    """The same-cap re-send attempts of ``item_id`` in the builder's joint inventory (at the cap; the final draw of a
    default item is not)."""
    sample = C.sample_header_value(item_id, 0)
    return [a for a in attempts if a.row["task"] == S.TASK and a.row["sample"] == sample and a.row["finish_reason"] == "length"]


def test_a_resend_with_another_body_is_unexplained_even_when_the_log_counts_match(tmp_path):
    """A re-send is bound to the body. A duplicate carrying another
    ``body_sha256`` than the sample's first call — recorded consistently in BOTH
    inventories, so the request-evidence reconciliation and the log's call
    counts still agree — is a changed prompt / seed / sampling under the same
    sample label: UNEXPLAINED, the gate fails by name. A recorded resume that
    re-sends the sample with another body fails too."""
    rows = primary_rows(QWEN)
    key = M.leg_key(QWEN, OBJECT, PRIMARY)
    other = "e" * 64

    def rebody(attempts):
        (a,) = _phantom_attempts(attempts, S.IDS[0])
        a.record["body_sha256"] = other
        a.row["body_sha256"] = other

    dirs = S.build_all(tmp_path / "b", rows=rows, overrides={(QWEN, OBJECT, PRIMARY): {"redraws": {0: 1}, "attempts_edit": rebody}})
    rep = analyze(tmp_path / "b", rows=rows)
    assert rep["gates"]["identity"]["status"] == M.GATE_PASS            # both inventories agree with each other
    g = rep["gates"]["ledger"]
    assert g["status"] == M.GATE_FAIL and "body_sha256 is not the sample's first call's" in g["detail"]
    assert "another experiment, not a re-send" in g["detail"] and C.sample_header_value(S.IDS[0], 0) in g["detail"]
    rd = g["legs"][key]["redraws"]
    assert (rd["n_items_body_changed"], rd["n_unexplained_calls"], rd["n_items_unexplained"], rd["n_redraw_calls"],
            rd["attributed"]) == (1, 1, 1, 0, False)
    if importlib.util.find_spec("inspect_ai") is not None:
        # the log's counts match (one model-call retry), so only the body binding catches it
        S.write_inspect_log(dirs[key], name="pass", model_retries={S.IDS[0]: [1]})
        e = analyze(tmp_path / "b", rows=rows, read_eval_logs=True)
        leg = leg_of(e, QWEN, OBJECT)
        assert leg["refused"] is None and leg["eval_check"]["redraws"]["ledger_check"]["agrees"]
        assert e["gates"]["ledger"]["status"] == M.GATE_FAIL and "body_sha256" in e["gates"]["ledger"]["detail"]
    # the same body on every re-send is what the real cells show; the gate passes them (the "ok" case of the suite)
    S.build_all(tmp_path / "ok", rows=rows, overrides={(QWEN, OBJECT, PRIMARY): {"redraws": {0: 1}}})
    assert analyze(tmp_path / "ok", rows=rows)["gates"]["ledger"]["status"] == M.GATE_PASS

    def rebody_resume(attempts):
        sample = C.sample_header_value(S.IDS[0], 0)
        late = [a for a in attempts if a.row["task"] == S.TASK and a.row["sample"] == sample and a.row["ts"] != S.TS0]
        assert len(late) == 1
        late[0].record["body_sha256"] = other
        late[0].row["body_sha256"] = other

    S.build_all(tmp_path / "r", rows=rows, overrides={(QWEN, OBJECT, PRIMARY): {"resume_rows": 1, "attempts_edit": rebody_resume}})
    g = analyze(tmp_path / "r", rows=rows)["gates"]["ledger"]
    assert g["status"] == M.GATE_FAIL and "a recorded resume re-sent with another body_sha256" in g["detail"]
    assert g["legs"][key]["redraws"]["n_items_body_changed"] == 1 and g["legs"][key]["redraws"]["n_resume_calls"] == 1


def test_the_final_draw_is_the_latest_request_start_not_the_files_last_row(tmp_path):
    """The ledger appends a row when the upstream reply completes and
    stamps ``ts`` at the request start, so an abandoned at-cap request that
    finishes after its replacement lands LAST in the file. Three such samples
    on items that finished normally: the file's last rows say ``length``, the
    latest-start rows say ``stop``; the at-cap count follows the replacement
    (n_length == the summary's n_at_cap), the gate passes and reports how many
    samples the file order would have got wrong. In the log-reading run the
    selected row's finish is held to the log's at-cap bit per sample: a final
    row that disagrees refuses the leg by name."""
    rows = primary_rows(QWEN)
    key = M.leg_key(QWEN, OBJECT, PRIMARY)
    three = {0: 1, 1: 1, 2: 1}
    dirs = S.build_all(tmp_path, rows=rows, overrides={(QWEN, OBJECT, PRIMARY): {"redraws": three, "redraw_late_completion": True}})
    cell_dir = dirs[key]
    ledger = [json.loads(l) for l in (cell_dir.parent.parent / C.LEDGER_FILENAME).read_text().splitlines() if l.strip()]
    for j in range(3):
        mine = [r for r in ledger if r["task"] == S.TASK and r["sample"] == C.sample_header_value(S.IDS[j], 0)]
        assert [r["finish_reason"] for r in mine] == ["stop", "length"]        # file order: the abandoned request last ...
        assert mine[0]["ts"] > mine[1]["ts"]                                     # ... though it STARTED first
    rep = analyze(tmp_path, rows=rows)
    led = leg_of(rep, QWEN, OBJECT)["ledger"]
    assert rep["gates"]["ledger"]["status"] == M.GATE_PASS and led["passed"]
    assert led["n_length"] == led["summary_n_at_cap"] == leg_of(rep, QWEN, OBJECT)["summary"]["n_at_cap"]
    rd = led["redraws"]
    assert (rd["n_redraw_calls"], rd["n_items_redrawn"], rd["n_final_draw_file_last_differs"], rd["attributed"]) == (3, 3, 3, True)
    # the file's last rows would have read 3 abandoned at-cap draws as the samples' finish: the very miscount the rule fixes
    n_file_last_length = sum(1 for j in range(500)
                             if [r for r in ledger if r["task"] == S.TASK and r["sample"] == C.sample_header_value(S.IDS[j], 0)][-1]["finish_reason"] == "length")
    assert n_file_last_length == led["n_length"] + 3
    if importlib.util.find_spec("inspect_ai") is None:
        return
    S.write_inspect_log(cell_dir, name="pass", model_retries={S.IDS[j]: [1] for j in range(3)})
    e = analyze(tmp_path, rows=rows, read_eval_logs=True)
    leg = leg_of(e, QWEN, OBJECT)
    assert leg["refused"] is None and e["gates"]["ledger"]["status"] == M.GATE_PASS
    lc = leg["eval_check"]["redraws"]["ledger_check"]
    assert lc["agrees"] and lc["final_draw_selection"] == {"rule": M.FINAL_DRAW_RULE_TEXT, "n_latest_start_disagreements": 0,
                                                          "n_file_last_disagreements": 3}
    assert leg["eval_check"]["redraws"]["n_at_cap"] == leg["summary"]["n_at_cap"]
    f = analyze(tmp_path, rows=rows)
    assert M.numeric_leaf_diff(e, f)["equal"] and M.T.numeric_leaf_diff(e, f)["equal"]
    # a selected final row whose finish reason is not the log's at-cap bit for its sample refuses the leg by name
    p = cell_dir.parent.parent / C.LEDGER_FILENAME
    edited = []
    for r in ledger:
        if r["task"] == S.TASK and r["sample"] == C.sample_header_value(S.IDS[3], 0):
            r = {**r, "finish_reason": "length"}
        edited.append(r)
    p.write_text("\n".join(json.dumps(r) for r in edited) + "\n")
    bad = leg_of(analyze(tmp_path, rows=rows, read_eval_logs=True), QWEN, OBJECT)
    assert bad["refused"] and S.IDS[3] in bad["refused"] and "1 item(s)" in bad["refused"]
    assert bad["eval_check"]["redraws"]["ledger_check"]["mismatches"] == [
        {"item_id": S.IDS[3], "ledger_final_finish_reason": "length", "log_at_cap": False,
         "why": "the selected final row's finish reason is not the log's at-cap bit for the sample"}]
    assert bad["eval_check"]["redraws"]["ledger_check"]["final_draw_selection"]["n_latest_start_disagreements"] == 1


def test_failed_attempt_classes_decide_the_calls_a_failed_attempt_made(tmp_path):
    """``LADDER_CALLS_PER_ATTEMPT`` is an upper bound, and only the exhausted
    timeout ladder is exact (nine, because the error type is the ladder's own).
    A pre-wire provider refusal made exactly 0 calls; a post-response one (the
    neither-channel refusal) is BOUNDED in [1, 9] — the SDK's retries can
    precede it — and any other failure in [0, 9]; bounded items are named, the
    maximum never substituted. An exhausted ladder (10 calls), a recovered
    neither-channel failure (2), an unverifiable failure inside its bound and a
    pre-wire refusal (1) all reconcile; a pre-wire refusal over 2 calls, a
    refusal over 11 and an unverifiable failure over 11 refuse."""
    pytest.importorskip("inspect_ai")
    rows = primary_rows(QWEN)
    key = M.leg_key(QWEN, OBJECT, PRIMARY)
    dirs = S.build_all(tmp_path, rows=rows, overrides={(QWEN, OBJECT, PRIMARY): {"redraws": {0: 1, 1: M.LADDER_CALLS_PER_ATTEMPT, 2: 4}}})
    cell_dir = dirs[key]
    S.write_inspect_log(cell_dir, name="pass", failed_attempts={S.IDS[0]: [S.FAILED_ATTEMPT_REFUSAL_MESSAGE], S.IDS[1]: 1,
                                                                S.IDS[2]: ["RuntimeError: boom"], S.IDS[3]: [S.FAILED_ATTEMPT_PRE_WIRE_MESSAGE]})
    e = analyze(tmp_path, rows=rows, read_eval_logs=True)
    leg = leg_of(e, QWEN, OBJECT)
    assert leg["refused"] is None and e["gates"]["ledger"]["status"] == M.GATE_PASS
    lc = leg["eval_check"]["redraws"]["ledger_check"]
    assert lc["agrees"] and lc["n_failed_attempts_by_class"] == _class_counts(timeout_ladder=1, provider_refusal_post_response=1,
                                                                             unverifiable=1, provider_refusal_pre_wire=1)
    assert (lc["n_failed_attempts_exact"], lc["n_failed_attempts_bounded"]) == (2, 2)
    assert lc["unverifiable_items"] == [S.IDS[0], S.IDS[2]] and lc["n_items_with_failed_attempts"] == 4
    per = leg["eval_check"]["redraws"]["per_item"]
    assert per[S.IDS[0]]["failed_attempt_classes"] == [M.FAILED_ATTEMPT_PROVIDER_REFUSAL_POST_RESPONSE] and per[S.IDS[0]]["failed_attempt_bounds"] == [[1, 9]]
    assert per[S.IDS[1]]["failed_attempt_classes"] == [M.FAILED_ATTEMPT_TIMEOUT_LADDER] and per[S.IDS[1]]["failed_attempt_bounds"] == [[9, 9]]
    assert per[S.IDS[2]]["failed_attempt_classes"] == [M.FAILED_ATTEMPT_UNVERIFIABLE] and per[S.IDS[2]]["failed_attempt_bounds"] == [[0, 9]]
    assert per[S.IDS[3]]["failed_attempt_classes"] == [M.FAILED_ATTEMPT_PROVIDER_REFUSAL_PRE_WIRE] and per[S.IDS[3]]["failed_attempt_bounds"] == [[0, 0]]
    # the ledger alone sees a failed attempt only past one attempt's ladder: the refused-after-one-call item is invisible
    rd = leg["ledger"]["redraws"]
    assert (rd["n_items_sample_error"], rd["n_items_redrawn"], rd["n_redraw_calls"]) == (1, 3, 1 + M.LADDER_CALLS_PER_ATTEMPT + 4)
    assert "invisible" in rd["n_items_sample_error_note"]
    # run F reads the same numeric readout; the two reports still match on every numeric leaf
    f = analyze(tmp_path, rows=rows)
    assert M.numeric_leaf_diff(e, f)["equal"] and M.T.numeric_leaf_diff(e, f)["equal"]
    # a pre-wire refusal is exactly no call: over a sample the ledger shows two calls for it is a mismatch (1 expected)
    S.write_inspect_log(cell_dir, name="pass", failed_attempts={S.IDS[0]: [S.FAILED_ATTEMPT_PRE_WIRE_MESSAGE], S.IDS[1]: 1,
                                                                S.IDS[2]: ["RuntimeError: boom"]})
    bad = leg_of(analyze(tmp_path, rows=rows, read_eval_logs=True), QWEN, OBJECT)
    assert bad["refused"] and S.IDS[0] in bad["refused"]
    assert bad["eval_check"]["redraws"]["ledger_check"]["mismatches"] == [
        {"item_id": S.IDS[0], "ledger_calls": 2, "log_calls": 1, "final_calls": 1,
         "failed_attempt_classes": [M.FAILED_ATTEMPT_PROVIDER_REFUSAL_PRE_WIRE],
         "why": "the ledger's calls are not the log's final-attempt calls plus what each failed attempt's class implies"}]
    # a post-response refusal the log records over a sample the ledger shows ELEVEN calls for is outside its bound [2, 10]
    dirs2 = S.build_all(tmp_path / "p", rows=rows, overrides={(QWEN, OBJECT, PRIMARY): {"redraws": {1: M.LADDER_CALLS_PER_ATTEMPT + 1}}})
    S.write_inspect_log(dirs2[key], name="pass", failed_attempts={S.IDS[1]: [S.FAILED_ATTEMPT_REFUSAL_MESSAGE]})
    bad = leg_of(analyze(tmp_path / "p", rows=rows, read_eval_logs=True), QWEN, OBJECT)
    assert bad["refused"] and S.IDS[1] in bad["refused"]
    (m,) = bad["eval_check"]["redraws"]["ledger_check"]["mismatches"]
    assert m["log_calls"] == [2, 1 + M.LADDER_CALLS_PER_ATTEMPT] and m["ledger_calls"] == 2 + M.LADDER_CALLS_PER_ATTEMPT
    assert m["why"].endswith("(a bounded failed attempt: a bound, not a count)")
    # an unverifiable failure outside its bound [1, 10] refuses too, and says the comparison was a bound
    dirs3 = S.build_all(tmp_path / "u", rows=rows, overrides={(QWEN, OBJECT, PRIMARY): {"redraws": {2: M.LADDER_CALLS_PER_ATTEMPT + 1}}})
    S.write_inspect_log(dirs3[key], name="pass", failed_attempts={S.IDS[2]: ["RuntimeError: boom"]})
    bad = leg_of(analyze(tmp_path / "u", rows=rows, read_eval_logs=True), QWEN, OBJECT)
    assert bad["refused"] and S.IDS[2] in bad["refused"]
    (m,) = bad["eval_check"]["redraws"]["ledger_check"]["mismatches"]
    assert m["log_calls"] == [1, 1 + M.LADDER_CALLS_PER_ATTEMPT] and m["ledger_calls"] == 2 + M.LADDER_CALLS_PER_ATTEMPT
    assert bad["eval_check"]["redraws"]["ledger_check"]["unverifiable_items"] == [S.IDS[2]]


def test_a_timed_out_call_a_retried_empty_200_and_the_refusal_then_a_recovered_re_run_reconcile(tmp_path):
    """A repro through the provider: a request times out at the SDK, the SDK's
    retry returns an EMPTY HTTP-200 completion (an identical body), the provider
    raises the neither-channel refusal AFTER ``super().generate()`` — two calls
    behind one ``tinycap provider:`` error — and the sample's re-run succeeds on
    its first call: three ledger rows for the sample. Read as "a refusal is
    exactly one call" the log would expect two and run E would falsely REFUSE a
    legitimate recovered sample while run F accepted it. The refusal is a bound:
    the gate PASSES, the attempt is bounded [1, 9] and its item named, run E
    agrees, and the two runs' numeric leaves are equal."""
    pytest.importorskip("inspect_ai")
    rows = primary_rows(QWEN)
    key = M.leg_key(QWEN, OBJECT, PRIMARY)
    dirs = S.build_all(tmp_path, rows=rows, overrides={(QWEN, OBJECT, PRIMARY): {"redraws": {0: 2}}})
    ledger = [json.loads(l) for l in (dirs[key].parent.parent / C.LEDGER_FILENAME).read_text().splitlines() if l.strip()]
    mine = [r for r in ledger if r["task"] == S.TASK and r["sample"] == C.sample_header_value(S.IDS[0], 0)]
    assert len(mine) == 3 and len({r["body_sha256"] for r in mine}) == 1 and all(r["status"] == 200 for r in mine)
    S.write_inspect_log(dirs[key], name="pass", failed_attempts={S.IDS[0]: [S.FAILED_ATTEMPT_REFUSAL_MESSAGE]})
    e = analyze(tmp_path, rows=rows, read_eval_logs=True)
    leg = leg_of(e, QWEN, OBJECT)
    assert leg["refused"] is None and e["gates"]["ledger"]["status"] == M.GATE_PASS and M.production_refusals(e) == []
    lc = leg["eval_check"]["redraws"]["ledger_check"]
    assert lc["agrees"] and lc["n_mismatched"] == 0 and lc["unverifiable_items"] == [S.IDS[0]]
    assert lc["n_failed_attempts_by_class"] == _class_counts(provider_refusal_post_response=1) and lc["n_failed_attempts_bounded"] == 1
    per = leg["eval_check"]["redraws"]["per_item"][S.IDS[0]]
    assert per["final_calls"] == 1 and per["failed_attempt_bounds"] == [[1, M.LADDER_CALLS_PER_ATTEMPT]]
    assert 3 != per["final_calls"] + 1                    # the exact-one reading would have refused this sample
    rd = leg["ledger"]["redraws"]
    assert (rd["n_redraw_calls"], rd["n_items_redrawn"], rd["attributed"], rd["n_items_sample_error"]) == (2, 1, True, 0)
    f = analyze(tmp_path, rows=rows)
    assert leg_of(f, QWEN, OBJECT)["ledger"]["redraws"]["n_redraw_calls"] == 2 and f["gates"]["ledger"]["status"] == M.GATE_PASS
    assert M.numeric_leaf_diff(e, f)["equal"] and M.T.numeric_leaf_diff(e, f)["equal"]


def test_e3_main_view_leg_keys_are_unchanged_and_an_it_view_key_round_trips_through_every_parser():
    """E3: the it-doc rows share (model, cell, mode) with the main-view rows of their cells; a non-main view's key appends
    ``@<view>`` to the mode segment, the main spelling is byte-identical, and every ``split("/")`` reader still finds the model
    at [0] and the cell at [1]."""
    assert M.leg_key(QWEN, BARE, PRIMARY) == f"{QWEN}/none/{PRIMARY}" == M.leg_key(QWEN, BARE, PRIMARY, M.VIEW_MAIN)
    assert (M.VIEW_MAIN, M.VIEW_IT, M.OBJECT_SOURCE_ITDOC, M.LEG_KEY_VIEW_SEP) == ("main", "it", "itdoc", "@")
    k = M.leg_key(QWEN, OBJECT, PRIMARY, M.VIEW_IT)
    assert k == f"{QWEN}/{OBJECT}/{PRIMARY}@it" and k != M.leg_key(QWEN, OBJECT, PRIMARY)
    assert k.split("/")[0] == QWEN and k.split("/")[1] == OBJECT and k.count("/") == 2
    assert M._kind_of_leg_key(k) == "object" == M._kind_of_leg_key(M.leg_key(QWEN, OBJECT, PRIMARY))
    assert M.leg_key_view(k) == "it" and M.leg_key_view(M.leg_key(QWEN, OBJECT, PRIMARY)) == "main"
    assert M.leg_key_view(f"{QWEN}/{PRIMARY}@it") == "it" and M.leg_key_view(f"{QWEN}/{PRIMARY}") == "main"      # a comparison name
    assert M._leg_words(k) == f"{QWEN} object [it]" and M._leg_words(M.leg_key(QWEN, OBJECT, PRIMARY)) == f"{QWEN} object"
    assert M._comparison_words(f"{QWEN}/{PRIMARY}@it") == f"{QWEN} [it]" and M._comparison_words(f"{QWEN}/{PRIMARY}") == QWEN
    assert M._leg_short({"model_key": QWEN, "kind": "object", "mode": PRIMARY}) == f"{QWEN} object"
    assert M._leg_short({"model_key": QWEN, "kind": "object", "mode": GENERAL, "view": "it"}) == f"{QWEN} object [it] [{GENERAL}]"
    row = {"model_key": QWEN, "cell": OBJECT, "sampling_mode": PRIMARY}
    assert M.row_view(row) == "main" and M.row_leg_key(row) == M.leg_key(QWEN, OBJECT, PRIMARY) and M.pin_key(row) == (QWEN, OBJECT)
    assert M.row_leg_key({**row, "view": "it"}) == k and M.pin_key({**row, "view": "it"}) == (QWEN, OBJECT, "it")
    # an EXTERNAL arm's `view` is its object bank's directory (an identity key), never a second row of the cell: keyed main
    ext = {"model_key": QWEN, "cell": "alphasteer", "sampling_mode": PRIMARY, "view": "it"}
    assert M.key_view(ext) == "main" and M.row_view(ext) == "it" and M.row_leg_key(ext) == f"{QWEN}/alphasteer/{PRIMARY}"
    assert M.pin_key(ext) == (QWEN, "alphasteer") and M.key_view({**ext, "cell": "arditi", "view": "ar_r0p05"}) == "main"
    assert M.row_leg_key({**ext, "cell": "arditi", "view": "ar_r0p05"}) == f"{QWEN}/arditi/{PRIMARY}"       # byte-identical to before
    # arditi-allpos-cap: the all-positions Arditi row is an `it`-VIEW row (its bank view rides apart as `bank_view`), keyed on the ARM
    # (`Arm.view_keyed`), never by a blanket external -> main rule: its key carries `@it`; the two published external arms' keys do not
    allpos = {"model_key": QWEN, "cell": "arditi_allpos", "sampling_mode": PRIMARY, "view": "it", "bank_view": "ar_c0p3"}
    assert M.key_view(allpos) == "it" and M.row_leg_key(allpos) == f"{QWEN}/arditi_allpos/{PRIMARY}@it" and M.pin_key(allpos) == (QWEN, "arditi_allpos", "it")
    assert M.leg_key_view(M.row_leg_key(allpos)) == "it" and M._kind_of_leg_key(M.row_leg_key(allpos)) == "arditi_allpos"
    assert M.key_view({**allpos, "cell": "arditi"}) == "main" and M.key_view({**allpos, "cell": "alphasteer"}) == "main"
    assert M.key_view({**row, "view": "it"}) == "it"
    assert not M.is_itdoc_row(row) and M.is_itdoc_row({**row, "object_source": "itdoc"})
    assert M.itdoc_carriage({**row, "object_source": "itdoc", "lane": "l"}) == {"object_source": "itdoc", "lane": "l", "qualification": None,
                                                                              "qualification_pattern": None}
    assert M.qualification_text({"qualification": "x", "qualification_pattern": "y"}) == "x y"
    assert M.qualification_text({"qualification": None, "qualification_pattern": "y"}) == "y" and M.qualification_text({}) is None
    assert M.comp_view({"mode": PRIMARY}) == "main" and M.comp_slot({"mode": PRIMARY}) == PRIMARY
    assert M.comp_view({"mode": PRIMARY, "view": "it"}) == "it" and M.comp_slot({"mode": PRIMARY, "view": "it"}) == f"{PRIMARY}@it"
    # the object pins admit the (model, cell, view) keying beside the published two
    pins = M.normalise_object_pins({QWEN: "a" * 64, (QWEN, OBJECT, "it"): "b" * 64, (GEMMA, OBJECT): "c" * 64})
    assert pins == {(QWEN, OBJECT): "a" * 64, (QWEN, OBJECT, "it"): "b" * 64, (GEMMA, OBJECT): "c" * 64}
    with pytest.raises(M.AnalysisError, match="neither a model key nor a"):
        M.normalise_object_pins({(QWEN,): "a" * 64})


def test_key_shown_reads_an_it_view_key_through_the_same_selectors_as_row_shown(full_report):
    """A non-main view's key appends ``@<view>`` to the mode segment, and :func:`_key_shown` reads it through the same
    three selectors (family, arm, mode) as :func:`_row_shown` — the view is not a selector — so a leg id or a comparison
    name spelt with a view is shown exactly when its main spelling is, hidden exactly when its main spelling is. The
    default render's filter (families only) is untouched."""
    view = {**full_report, "figures": {M.FIGURES_EXCLUDED_KEY: [GLM], M.FIGURES_MODES_KEY: [PRIMARY], M.FIGURES_SHOWN_ONLY_KEY: True}}
    leg, it_leg = M.leg_key(QWEN, OBJECT, PRIMARY), M.leg_key(QWEN, OBJECT, PRIMARY, M.VIEW_IT)
    assert it_leg == f"{leg}@it"
    assert M._key_shown(view, it_leg) == M._key_shown(view, leg) == M._row_shown(view, {"model_key": QWEN, "cell": OBJECT, "mode": PRIMARY}) is True
    assert M._key_shown(view, f"{QWEN}/{PRIMARY}@it") == M._key_shown(view, f"{QWEN}/{PRIMARY}") is True
    # a mode the render does not draw hides both spellings; an excluded family hides both spellings
    assert M._key_shown(view, M.leg_key(QWEN, OBJECT, GENERAL, M.VIEW_IT)) == M._key_shown(view, M.leg_key(QWEN, OBJECT, GENERAL)) is False
    assert M._key_shown(view, f"{QWEN}/{GENERAL}@it") == M._key_shown(view, f"{QWEN}/{GENERAL}") is False
    assert M._key_shown(view, M.leg_key(GLM, OBJECT, PRIMARY, M.VIEW_IT)) is False and M._key_shown(view, f"{GLM}/{PRIMARY}@it") is False
    # the default render (shown_only absent): the families-only filter, as before — a view-spelt key of a shown family passes
    default = M._shown_only_keys_filter(full_report)
    assert default(it_leg) and default(f"{QWEN}/{GENERAL}@it") and default(leg)
