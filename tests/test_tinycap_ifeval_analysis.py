"""The IFEval canonical battery — the analysis over synthetic stores built through the REAL producers.

Pinned: the full launch matrix (30 cells) + a promoted chain (2048 -> 4096 -> 8192 over a draw-level subset) analyzes to a
PRODUCTION report — the three shares per cell (at-cap = truncated + loop), the four metrics with CIs under both readings,
``cell_minus_none`` contrasts under both readings with the verdict rule (both prompt-level CIs, merged), the noise floor for
the three two-view models (none for Nemotron, stated), ``merged - r2048`` for the promoted cell, gates I0–I8 PASS and I9
PENDING (waivable), the held rows counted ``held`` never ``missing``; each gate fails on a constructed defect (a torn ledger
count, an altered presence_penalty on the wire, a v2 evidence record, a wrong seed, a missing rung store, a rung over another
draw set, a missing draw in a pass-1 store, a foreign default_chat_template_kwargs, a wrong stamp); the ledger reconciliation
(ledger-at-cap == recorded-at-cap per leg; a torn record refuses); I3 reads the served S from ``geometry.kv_shrink``; figures
render with <= 2-line titles, tall axes, a resolvable numbers sidecar and the trigger provenance in every caption;
``--figure-exclude-family`` drops a family; the results skeleton renders every double-brace placeholder; exit codes 3 / 4 at
the CLI boundary; the two-venv numeric-leaf match.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.capeval.tiers import cap_ladder as CL
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import ifeval_contracts as IC

import ifeval_synthetic_store as G
import tinycap_ifeval_ladder_plan as LP

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load("tinycap_ifeval_battery", "scripts/tinycap_ifeval_analysis_under_test.py") if False else _load("tinycap_ifeval_battery", "scripts/tinycap_ifeval_battery.py")
V = A.V
ROWS = G.matrix_rows()
LAUNCH = G.launch_rows(ROWS)
ESC = ("gemma4e4b", "main", "constv2-mix-full__lora_r4_ce")
BELOW = ("qwen35_4b", "main", "constv2ctl2-full__lora_r4_ce")
B_FAST = 200
HAS_INSPECT = importlib.util.find_spec("inspect_ai") is not None


def row(*t):
    return G.row_for(ROWS, *t)


def analyze(root, *, matrix=None, **kw):
    kw.setdefault("matrix", matrix if matrix is not None else ROWS)
    kw.setdefault("canonical_ids", G.IDS)
    kw.setdefault("n_resamples", B_FAST)
    kw.setdefault("read_eval_logs", False)
    kw.setdefault("analysis_sha", G.FULL_SHA)
    return A.analyze(root, G.SHA7, **kw)


def small_matrix(*triples):
    keep = set(triples)
    rows = [dict(r) for r in ROWS if (r["model_key"], r["view"], r["cell"]) in keep]
    for i, r in enumerate(rows):
        r["index"] = i
    return rows


SMALL = small_matrix(("gemma4e4b", "it", "none"), ("gemma4e4b", "it", "prompted__safety_refusal"), ("gemma4e4b", "it", "constv2-mix-full__lora_r4_ce"),
                     ("gemma4e4b", "main", "none"), ESC, ("gemma4e4b", "main", "constv2ctl2-full__lora_r4_ce"),
                     ("qwen35_4b", "main", "none"), BELOW, ("llama8b", "main", "none"), ("llama8b", "it", "none"))


def build_small(root, *, skip=()):
    for r in SMALL:
        if r["status"] == IC.STATUS_HELD or (r["model_key"], r["view"], r["cell"]) in set(skip) | {ESC}:
            continue
        G.build_store(root, r, **({"at_cap": {(i, 0) for i in range(5)}} if (r["model_key"], r["view"], r["cell"]) == BELOW else {}))
    if ESC not in set(skip):
        return G.build_escalated_cell(root, row(*ESC), at_cap={(i, t) for i in range(12) for t in (0, 1)}, loops={(0, 0), (0, 1)},
                                      rung_still_at_cap={(i, t) for i in range(1, 8) for t in (0, 1)}, rung_boundary={(9, 0)},
                                      rung8192=True, rung8192_still_at_cap={(1, 0)})
    return None


@pytest.fixture(scope="module")
def small_store(tmp_path_factory):
    root = tmp_path_factory.mktemp("ife_small")
    esc = build_small(root)
    return SimpleNamespace(root=root, esc=esc)


@pytest.fixture(scope="module")
def small_report(small_store):
    return analyze(small_store.root, matrix=SMALL)


@pytest.fixture(scope="module")
def full_store(tmp_path_factory):
    """All 30 launch legs + the promoted chain; the two held rows and the eight deferred slots are never built."""
    root = tmp_path_factory.mktemp("ife_full")
    G.build_all(root, rows=ROWS, skip=[ESC])
    esc = G.build_escalated_cell(root, row(*ESC), at_cap={(i, t) for i in range(12) for t in (0, 1)}, loops={(0, 0), (0, 1)},
                                 rung_still_at_cap={(i, t) for i in range(1, 8) for t in (0, 1)}, rung_boundary={(9, 0)}, rung8192=True,
                                 rung8192_still_at_cap={(1, 0)})
    return SimpleNamespace(root=root, esc=esc)


# --------------------------------------------------------------------------- the production result
def test_full_synthetic_wave_is_a_production_report_with_held_rows_counted_held(full_store):
    report = analyze(full_store.root)
    # this module reads NO .eval log (the CPU suite has no inspect_ai): since revision 3 every gate but the two log-derived
    # ones is clear and the report is NOT production — I4 / I5 PENDING, never waivable; the log-read production path is pinned in
    # test_ifeval_analysis_review_r3.py
    assert A.non_log_gates_clear(report), A.production_refusals(report)
    assert report["verdict"]["production"] is False and report["gates"]["failed"] == []
    assert report["gates"]["pending"] == ["I4_scoring_path", "I5_at_cap", "I9_report_match"] and report["verdict"]["gates_pending_log_derived"] == list(A.LOG_DERIVED_GATES)
    assert report["gates"]["I0_completeness"]["counts"] == {"usable": 30, "held": 2, "deferred": 8}
    assert len(report["held"]) == 2 and all("P41" in h["reason"] for h in report["held"])
    assert len(report["deferred"]) == 8 and all(d["record"].startswith("configs/sdfbase/ifeval/deferred/") for d in report["deferred"])
    assert report["contrast_summary"]["n_designed"] == 23 and report["contrast_summary"]["n_filed"] == 23
    assert report["verdict"]["n_usable_cells"] == 30 and report["verdict"]["n_promoted_cells"] == 1
    assert {n["model_key"]: n["status"] for n in report["noise_floor"]} == {"gemma4e4b": "filed", "qwen35_4b": "filed", "nemotron3nano4b": "no floor (one view)", "llama8b": "filed"}
    assert len(report["legs"]) == 30 + 2 and report["unusable_legs"] == {}
    # the cross-eval guard keys, per cell: the object sha (a stated null on bare / prompted), the view, the store pin
    for key, c in report["cells"].items():
        # `pin` = the pass-1 store's producer tree (per-row admission, V24); `label` = the record's; a slot carries the label only
        assert set(("object_sha256", "view", "pin", "pins_by_cap", "label")) <= set(c) and c["label"] == G.SHA7, key
        assert c["pin"] == (G.FULL_SHA if c.get("usable") else G.SHA7), key
        r = row(*key.split("/"))
        assert c["object_sha256"] == r["object_sha256"] and c["view"] == r["view"]
    assert report["design"]["n_deferred_rows"] == 8
    assert report["design"]["trigger_provenance"] == "1 % raw at-cap of draws, per the IFEval ruling"
    assert "whitespace" in report["design"]["loop_rule_disclosure"].lower()
    assert report["cost"]["total_gpu_h_actual"] > 0


def test_the_three_shares_the_four_metrics_and_the_two_readings(small_report, small_store):
    r = small_report
    c = r["cells"]["/".join(ESC)]
    s = c["shares"]
    assert s["n_draws"] == 1082 and s["n_at_cap"] == 24 and s["exit"] == CL.ESCALATE
    assert s["n_at_cap"] == s["n_truncated"] + s["n_loop"]                                 # at-cap = truncated (incl. boundary) + loop (integers)
    assert s["share_at_cap"] == 24 / 1082                                                      # the analysis's own exact ratio
    assert (s["share_truncated"], s["share_loop"]) == (round(22 / 1082, 6), round(2 / 1082, 6))   # the record's, each its own rounding
    assert s["n_truncated"] == 22 and s["n_loop"] == 2 and s["trigger_provenance"] == IC.IFEVAL_TRIGGER_PROVENANCE
    assert c["ladder"]["promoted"] == [4096, 8192] and c["redraw_outcomes"] == {"4096": {"boundary_landing": 1, "extended": 7, "still_at_cap": 14}, "8192": {"extended": 13, "still_at_cap": 1}}
    # the residual: 1 still at cap at 8,192 + the 2 loops never re-drawn = 3 of 1,082 -> THE decision at the top is below the
    # threshold (the 8,192 store's own exit is `ceiling`: 1 of its 14 draws), both on record (C1)
    assert c["ladder"]["residual"] == {"n_at_cap": 3, "share_at_cap": 3 / 1082, "final_cap": 8192, "final_exit": CL.EXIT_BELOW_THRESHOLD,
                                       "final_store_exit": CL.EXIT_CEILING}
    assert c["ladder"]["exits"]["4096"]["reading"] == "merged" and c["ladder"]["exits"]["4096"]["n_draws"] == 1082 and c["ladder"]["exits"]["4096"]["n_at_cap"] == 16
    assert c["ladder"]["exits"]["4096"]["store"] == {"exit": CL.ESCALATE, "next_cap": 8192, "n_draws": 22, "n_at_cap": 14, "share_at_cap": round(14 / 22, 6),
                                                     "n_truncated": 14, "n_loop": 0, "n_boundary": 0}
    assert c["ladder"]["decision_problems"] == [] and s["decision"]["reading"] == "r2048" and s["store_exit"] == s["exit"] == CL.ESCALATE
    for reading in A.READINGS:
        for m in A.METRICS:
            leg = c["metrics"][reading][m]
            assert 0 <= leg["ci_low"] <= leg["point"] <= leg["ci_high"] <= 1 and leg["n_scored"] > 0
    # the merged reading moves the metric where the rung scored what the truncated draws could not
    assert c["metrics"]["merged"]["prompt_strict"]["point"] != c["metrics"]["r2048"]["prompt_strict"]["point"]
    tc = next(t for t in r["truncation_cost"] if t["cell_key"] == "/".join(ESC))
    assert set(tc["metrics"]) == set(A.METRICS) and tc["promoted"] == [4096, 8192]
    below = r["cells"]["/".join(BELOW)]
    assert below["shares"]["n_at_cap"] == 5 and below["shares"]["exit"] == CL.EXIT_BELOW_THRESHOLD and below["ladder"]["promoted"] == []
    # every contrast is filed under BOTH readings with the verdict rule on the merged prompt-level pair
    for con in r["contrasts"]:
        assert con["status"] == "filed" and set(con["readings"]) == {"r2048", "merged"}
        assert set(con["readings"]["merged"]) == set(A.METRICS) and con["verdict"]["word"] in (A.WORD_LOWER, A.WORD_HIGHER, A.WORD_NOT_RESOLVED)
    esc_con = next(con for con in r["contrasts"] if con["cell_key"] == "/".join(ESC))
    assert esc_con["verdict"]["word"] == A.WORD_LOWER          # the constitution PSR rate 0.60 vs none 0.72 resolves
    # the noise floor for gemma (both views), none for the one-view qwen small matrix here
    assert {n["model_key"]: n["status"] for n in r["noise_floor"]}["gemma4e4b"] == "filed"
    assert r["gates"]["I1_ledger"]["reconciliation"][V.leg_key(*ESC, 2048)]["ledger_at_cap"] == 24 == r["gates"]["I1_ledger"]["reconciliation"][V.leg_key(*ESC, 2048)]["recorded_at_cap"]
    assert r["gates"]["I3_certification"]["served"][V.leg_key(*ESC, 2048)]["served_seqs"] == 233
    # C5: the reported decode rate is the AGGREGATE over the generation wall (~S x the per-sequence rate); the per-request
    # sum / sum reading rides beside it, labelled
    assert c["decode_tok_s_per_request"] == pytest.approx(G.DECODE_TOK_S, rel=1e-6) and c["decode_tok_s_aggregate"] > 50 * c["decode_tok_s_per_request"]
    assert r["cost"]["legs"][V.leg_key(*ESC, 2048)]["generation_span"]["elapsed_s"] > 0 and "decode_tok_s_mean" not in c


# --------------------------------------------------------------------------- each gate fails on a constructed defect
def _one_cell_matrix(triple):
    return small_matrix(("gemma4e4b", "main", "none"), triple) if triple != ("gemma4e4b", "main", "none") else small_matrix(triple)


def _defect_report(tmp_path, defect: str):
    key = ("gemma4e4b", "main", "constv2ctl2-full__lora_r4_ce")
    m = _one_cell_matrix(key)
    G.build_store(tmp_path, row("gemma4e4b", "main", "none"))
    kw: dict = {}
    if defect == "ledger_torn":
        def ledger_edit(rows):
            rows[0]["finish_reason"] = "length"
        kw["ledger_edit"] = ledger_edit
    elif defect == "presence_penalty_altered":
        def body_edit(body):
            body["presence_penalty"] = 1.0
        kw["body_edit"] = body_edit
    elif defect == "evidence_v2":
        def re_edit(recs):
            for r in recs:
                r["request_evidence_version"] = 2
                for f in IC.REQUEST_EVIDENCE_V3_FIELDS:
                    r.pop(f, None)
        kw["request_evidence_edit"] = re_edit
    elif defect == "wrong_seed":
        def body_edit(body):
            body["seed"] = 7
        kw["body_edit"] = body_edit
    elif defect == "foreign_template_pin":
        # the pin AS SERVED lives in the driver's evidence (`serve_header.template_kwargs`), not on the endpoint (V20)
        def evidence_edit(ev):
            ev["serve_header"]["template_kwargs"] = {"enable_thinking": True}
        kw["evidence_edit"] = evidence_edit
    elif defect == "wrong_stamp":
        def summary_edit(s):
            s["render_regime"] = "thinking_on"
        kw["summary_edit"] = summary_edit
    elif defect == "row_identity":
        def rows_edit(rows):
            for r in rows:
                if r["item_id"] != C.AGG_ITEM_ID:
                    extra = json.loads(r["extra_json"])
                    extra["view"] = "it"
                    r["extra_json"] = json.dumps(extra)
        kw["rows_edit"] = rows_edit
    elif defect == "missing_draw":
        def header_edit(h):
            h["items"] = h["items"][:-1]
            h["n_attempted_samples"] -= 1
        kw["header_edit"] = header_edit
    elif defect == "record_rule":
        pass
    G.build_store(tmp_path, row(*key), **kw)
    return analyze(tmp_path, matrix=m)


@pytest.mark.parametrize("defect, gate, match", [
    ("ledger_torn", "I1_ledger", "finish_reason == length"),
    ("presence_penalty_altered", "I2_identity", "presence_penalty"),
    ("evidence_v2", "I2_identity", "not recorded"),
    ("wrong_seed", "I2_identity", "seed on the wire"),
    ("foreign_template_pin", "I3_certification", "template_kwargs"),
    ("wrong_stamp", "I2_identity", "stamps"),
    ("row_identity", "I2_identity", "view"),
])
def test_each_gate_fails_on_its_constructed_defect(tmp_path, defect, gate, match):
    report = _defect_report(tmp_path, defect)
    assert report["gates"][gate]["status"] == A.GATE_FAIL, report["gates"][gate]
    assert match in (report["gates"][gate].get("detail") or "")
    assert report["verdict"]["production"] is False and A.production_refusals(report)


def test_a_llama_store_reads_not_available_and_unspecified_and_a_switched_one_is_refused_by_name(tmp_path):
    """The frozen matrix's llama rows carry `thinking: off` and `expected_stamps` `thinking_off` (the MODE and the OFF
    row — the bytes do not move), while a Llama store's producer hashes and stamps the family-resolved arm
    `not_available` and its proxy stamps `render_regime: unspecified` (no `enable_thinking` on the wire). The gates
    read the FAMILY-resolved expectation, so the real llama store passes and a store that claims the switch
    (`thinking_off`, `thinking_by_task off`, a row stamped `off`) is refused by name."""
    llama = ("llama8b", "main", "none")
    m = small_matrix(("gemma4e4b", "main", "none"), llama)
    G.build_store(tmp_path, row("gemma4e4b", "main", "none"))
    cd = G.build_store(tmp_path, row(*llama))
    summary = completion.read_task_summary(cd, IC.IFEVAL_TASK)
    assert summary["thinking"] == C.THINKING_NOT_AVAILABLE and summary["render_regime"] == C.RENDER_REGIME_UNSPECIFIED
    report = analyze(tmp_path, matrix=m)
    assert A.non_log_gates_clear(report), A.production_refusals(report)            # no .eval log read: not production (r3 S1)
    leg = report["legs"][V.leg_key(*llama, 2048)]
    assert leg["status"] == "usable" and leg["stamps"] is None and leg["identity"] is None and leg["rows_check"]["passed"]
    assert leg["request_evidence"]["passed"] and leg["scoring_path"]["failures"] == [] and leg["scoring_path"]["thinking"] == C.THINKING_NOT_AVAILABLE
    manifest = json.loads((cd / "manifest.json").read_text())
    assert manifest["resolved_config"]["tinycap"][C.IDENTITY_KEY_THINKING_BY_TASK] == {IC.IFEVAL_TASK: C.THINKING_NOT_AVAILABLE}
    assert manifest[C.MANIFEST_KEY_THINKING]["enable_thinking"] is None and manifest[C.MANIFEST_KEY_THINKING]["mode"] == C.THINKING_OFF
    # the wire carried no switch: every evidence record reads enable_thinking_present False
    recs = V.iter_jsonl(cd.parents[1] / C.REQUEST_EVIDENCE_FILENAME)
    assert recs and all(r["enable_thinking_present"] is False for r in recs)
    assert report["design"]["thinking_arm_by_family"] == {"gemma4": C.THINKING_OFF, "llama": C.THINKING_NOT_AVAILABLE}
    # a llama store stamped as if it had a switch: the row's frozen `thinking_off` is NOT what the gate expects
    def summary_edit(s):
        s["render_regime"] = C.RENDER_REGIME_THINKING_OFF
        s["thinking"] = C.THINKING_OFF
    def header_edit(h):
        h["thinking"] = C.THINKING_OFF          # consistent with the summary: finish_task verifies the two agree
    def identity_edit(t):
        t[C.IDENTITY_KEY_THINKING_BY_TASK] = {IC.IFEVAL_TASK: C.THINKING_OFF}
    def rows_edit(rows):
        for r in rows:
            if r["item_id"] != C.AGG_ITEM_ID:
                extra = json.loads(r["extra_json"])
                extra["thinking"] = C.THINKING_OFF
                r["extra_json"] = json.dumps(extra)
    def body_edit(body):
        body[C.CHAT_TEMPLATE_KWARGS_KEY] = {C.ENABLE_THINKING_KEY: False}      # the switch ON THE WIRE for a no-switch family
    bad = tmp_path / "switched"
    G.build_store(bad, row("gemma4e4b", "main", "none"))
    G.build_store(bad, row(*llama), summary_edit=summary_edit, header_edit=header_edit, identity_edit=identity_edit, rows_edit=rows_edit,
                  body_edit=body_edit)
    rep2 = analyze(bad, matrix=m)
    leg2 = rep2["legs"][V.leg_key(*llama, 2048)]
    assert rep2["verdict"]["production"] is False
    assert leg2["stamps"] and "unspecified" in leg2["stamps"] and "thinking_off" in leg2["stamps"]
    assert leg2["identity"] and "not_available" in leg2["identity"] and "'off'" in leg2["identity"]
    assert not leg2["rows_check"]["passed"] and any("thinking" in f for f in leg2["rows_check"]["failures"])
    assert any("not_available" in f for f in leg2["scoring_path"]["failures"])
    assert not leg2["request_evidence"]["passed"] and any("no-switch arm" in f for f in leg2["request_evidence"]["failures"])
    # a gemma store is held to the switch arm as before: a record WITHOUT the switch is refused
    def body_edit_g(body):
        body.pop(C.CHAT_TEMPLATE_KWARGS_KEY, None)
    g = tmp_path / "gemma_noswitch"
    G.build_store(g, row("gemma4e4b", "main", "none"), body_edit=body_edit_g)
    rep3 = analyze(g, matrix=small_matrix(("gemma4e4b", "main", "none")))
    leg3 = rep3["legs"][V.leg_key("gemma4e4b", "main", "none", 2048)]
    assert not leg3["request_evidence"]["passed"] and any("is not False (thinking OFF)" in f for f in leg3["request_evidence"]["failures"])


def test_a_missing_draw_in_a_pass1_store_fails_population_and_completeness_counts_it_missing_never_held(tmp_path):
    m = _one_cell_matrix(("gemma4e4b", "main", "constv2ctl2-full__lora_r4_ce"))
    G.build_store(tmp_path, row("gemma4e4b", "main", "none"))
    # a store whose marker declares the expected 1,082 draws but files 1,081 — finish_task writes PARTIAL; the leg is unusable
    G.build_store(tmp_path, row("gemma4e4b", "main", "constv2ctl2-full__lora_r4_ce"), drop_draws={(G.IDS[-1], 1)}, n_expected_draws=1082, ladder=False)
    report = analyze(tmp_path, matrix=m)
    assert report["gates"]["I0_completeness"]["status"] == A.GATE_FAIL and "PARTIAL" in report["gates"]["I0_completeness"]["detail"]
    assert report["gates"]["I0_completeness"]["counts"].get("missing") == 1 and "held" not in report["gates"]["I0_completeness"]["counts"]
    # the held rows of the FULL matrix are counted held, never missing
    report2 = analyze(tmp_path, matrix=small_matrix(("gemma4e4b", "main", "none"), ("gemma4e4b", "it", "constv2-mix-full__lora_r4_ce")))
    assert report2["gates"]["I0_completeness"]["counts"] == {"usable": 1, "held": 1} and report2["gates"]["I0_completeness"]["status"] == A.GATE_PASS


def test_a_promoted_cell_without_its_rung_store_fails_i0_and_a_rung_over_another_draw_set_is_refused_by_name(tmp_path):
    m = _one_cell_matrix(ESC)
    G.build_store(tmp_path, row("gemma4e4b", "main", "none"))
    esc = G.build_escalated_cell(tmp_path, row(*ESC), at_cap={(i, t) for i in range(12) for t in (0, 1)})
    # remove the rung store: promoted, no rung leg
    import shutil
    shutil.rmtree(esc["rung_cell_dir"].parents[1])
    report = analyze(tmp_path, matrix=m)
    assert report["gates"]["I0_completeness"]["status"] == A.GATE_FAIL and "promoted at 4096" in report["gates"]["I0_completeness"]["detail"]
    # a rung store over one draw fewer than its subset (the same subset block) is refused by the draw validator, by name
    draws = IC.draws_of_subset(esc["subset"])
    G.build_store(tmp_path, row(*ESC), at_cap=(), cap=4096, draws=draws[:-1], rung_source=completion.read_store_manifest(esc["source_cell_dir"]),
                  subset_doc=esc["subset"], subset_path=str(esc["subset_path"]), ladder=False, n_expected_draws=len(draws) - 1)
    report = analyze(tmp_path, matrix=m)
    rung_leg = report["legs"][V.leg_key(*ESC, 4096)]
    assert rung_leg["status"] != "usable" and "not exactly its draw subset" in (rung_leg["refused"] or "")
    assert report["gates"]["I0_completeness"]["status"] == A.GATE_FAIL


def test_i3_reads_the_served_s_from_the_kv_shrink_and_the_ledger_reconciliation_names_the_torn_leg(tmp_path):
    m = small_matrix(("qwen35_4b", "main", "none"))
    G.build_store(tmp_path, row("qwen35_4b", "main", "none"), kv_shrink=G.kv_shrink(142, 101, G.MAX_MODEL_LEN_BY_CAP[2048]))
    report = analyze(tmp_path, matrix=m)
    leg = report["legs"][V.leg_key("qwen35_4b", "main", "none", 2048)]
    # the served S is the KV-derived one (`served_geometry.geometry.max_num_seqs`), the record held through the producer's own rule
    assert leg["certification"]["served_seqs"] == 101 and leg["certification"]["configured_seqs"] == 142
    assert leg["certification"]["kv_shrink"]["configured_max_num_seqs"] == 142 and leg["certification"]["seqs_source"] == "kv_derived"
    assert report["gates"]["I3_certification"]["status"] == A.GATE_PASS
    assert leg["request_evidence"]["n_presence_penalty_present"] == 1082 and leg["request_evidence"]["presence_penalty_expected"] is True
    assert report["gates"]["I1_ledger"]["reconciliation"][leg["key"]]["client_timeout_s"] == 7200.0


def test_i5_holds_the_integer_at_cap_identity_not_a_tolerance_on_rounded_shares(tmp_path):
    """C3: 4 at-cap draws of 1,082 = 2 truncated + 2 loops. The producer rounds each share to six decimals on its own:
    4 / 1,082 -> 0.003697 while 0.001848 + 0.001848 = 0.003696 — the rounded shares differ by 1e-6 + epsilon, which the
    old tolerance refused. The record is CORRECT: the integer identity holds and every share is its own count's rounding."""
    from sisteer.capeval.tiers import inspect_rung as IR
    m = small_matrix(("gemma4e4b", "main", "none"))
    cd = G.build_store(tmp_path, row("gemma4e4b", "main", "none"), at_cap={(0, 0), (0, 1), (1, 0), (1, 1)}, loops={(0, 0), (0, 1)})
    rec = json.loads((cd.parents[1] / IC.CAP_LADDER_FILENAME).read_text())
    assert (rec["n_at_cap"], rec["n_truncated"], rec["n_loop"]) == (4, 2, 2)
    assert rec["share_at_cap"] == 0.003697 and rec["share_truncated"] == rec["share_loop"] == 0.001848
    assert abs(rec["share_at_cap"] - (rec["share_truncated"] + rec["share_loop"])) > 1e-6       # the reproduction of the refusal
    assert V.ladder_record_problem(rec, items=completion.read_task_outcomes(cd, IC.IFEVAL_TASK)["items"], cap=2048,
                                   source_manifest=completion.read_store_manifest(cd)) is None
    report = analyze(tmp_path, matrix=m)
    # the integer identity half PASSES on the leg (no failure named); the gate itself reads PENDING because no .eval log was read
    # here — the text half did not run and PENDING is never a PASS
    g5 = report["gates"]["I5_at_cap"]
    assert g5["status"] == A.GATE_PENDING and g5["failures"] == [] and g5["legs"][leg_key := V.leg_key("gemma4e4b", "main", "none", 2048)] == A.GATE_PASS, g5
    assert report["legs"][leg_key]["ladder_problem"] is None and g5["texts_reclassified"] is False
    assert report["cells"]["gemma4e4b/main/none"]["shares"]["exit"] == CL.EXIT_BELOW_THRESHOLD
    # the identity is on the counts: a record whose n_loop is torn by one is refused by the integer check, by name
    torn = {**rec, "n_loop": 3, "share_loop": round(3 / 1082, IR.SHARE_DECIMALS)}
    why = V.ladder_record_problem(torn, items=completion.read_task_outcomes(cd, IC.IFEVAL_TASK)["items"], cap=2048,
                                  source_manifest=completion.read_store_manifest(cd))
    assert why is not None and "n_loop" in why
    # and a share that is not the producer's rounding of its own count is refused, by name
    off = {**rec, "share_at_cap": 0.0037}
    why = V.ladder_record_problem(off, items=completion.read_task_outcomes(cd, IC.IFEVAL_TASK)["items"], cap=2048,
                                  source_manifest=completion.read_store_manifest(cd))
    assert why is not None and "share_at_cap" in why and f"{IR.SHARE_DECIMALS}" in why


def test_a_promoted_wave_analyzes_after_relocation_without_rewriting_its_provenance(tmp_path):
    """C2: every recorded root is absolute and names where the planner / driver ran (a volume path, an earlier pull); the
    documented workflow pulls the wave to a LOCAL root. The chain's source binding is therefore compared as the WAVE-RELATIVE
    layout (``ifeval_paths().out_root``) + the identity hash — so the same wave, moved wholesale with every recorded path
    untouched, still analyzes to the same production report."""
    import shutil
    m = _one_cell_matrix(ESC)
    built = tmp_path / "pull_a"
    G.build_store(built, row("gemma4e4b", "main", "none"))
    G.build_escalated_cell(built, row(*ESC), at_cap={(i, t) for i in range(12) for t in (0, 1)}, loops={(0, 0), (0, 1)},
                           rung_still_at_cap={(i, t) for i in range(1, 8) for t in (0, 1)}, rung_boundary={(9, 0)},
                           rung8192=True, rung8192_still_at_cap={(1, 0)})
    before = analyze(built, matrix=m)
    assert A.non_log_gates_clear(before), A.production_refusals(before)            # no .eval log read: not production (r3 S1)
    moved = tmp_path / "elsewhere" / "pull_b"
    moved.parent.mkdir()
    shutil.move(str(built), str(moved))
    # the provenance still names the ORIGINAL root: nothing was rewritten
    subset = json.loads((moved / IC.ifeval_paths(*ESC, 4096).subset).read_text())
    assert subset["source_out_root"].startswith(str(built)) and not Path(subset["source_out_root"]).exists()
    after = analyze(moved, matrix=m)
    assert A.non_log_gates_clear(after), A.production_refusals(after)
    cell = after["cells"]["/".join(ESC)]
    assert cell["ladder"]["promoted"] == [4096, 8192] and all(h["subset_problem"] is None for h in cell["chain"])
    assert after["legs"][V.leg_key(*ESC, 4096)]["status"] == "usable" and after["legs"][V.leg_key(*ESC, 8192)]["status"] == "usable"
    # the two reports agree on every result leaf (only gen_root / out_root strings moved)
    assert A.report_match(before, after)["equal"]
    # the tail match is by the wave-relative layout: a volume-style recorded root matches, another cell's does not, and an
    # absolute tail (the old caller's mistake) is refused by name rather than silently never matching
    layout = IC.ifeval_paths(*ESC, 2048).out_root
    assert V.same_store_tail(f"/vol/tinycap/ifeval_canonical/{G.SHA7}/{layout}", layout)
    assert V.same_store_tail(f"/vol/tinycap/ifeval_canonical/{G.SHA7}/{layout}/", layout)
    assert not V.same_store_tail(f"/vol/tinycap/ifeval_canonical/{G.SHA7}/{IC.ifeval_paths('gemma4e4b', 'main', 'none', 2048).out_root}", layout)
    assert not V.same_store_tail(f"/vol/tinycap/ifeval_canonical/{G.SHA7}/{layout}__rung4096", layout)
    with pytest.raises(ValueError, match="wave-relative"):
        V.same_store_tail(f"/vol/{layout}", f"/local/{layout}")


def test_i1_and_i2_scope_the_ledger_and_the_evidence_to_the_accepted_attempt_so_a_retry_passes(tmp_path):
    """C4: the driver appends request evidence and ledger rows across attempts and certifies the ACCEPTED one; a reader that
    counted every historical task record against the draw count refused a successful retry after an interrupted attempt.
    Both gates now scope through the contract helpers (``records_of_attempts`` / ``rows_of_attempts`` off the manifest's
    ``accepted_generation_attempt``) and report every other attempt's rows beside the accepted attempt's."""
    key = ("gemma4e4b", "main", "constv2ctl2-full__lora_r4_ce")
    m = _one_cell_matrix(key)
    G.build_store(tmp_path, row("gemma4e4b", "main", "none"))
    cd = G.build_store(tmp_path, row(*key), abandoned_attempt=37)
    manifest = completion.read_store_manifest(cd)
    assert manifest[C.MANIFEST_KEY_ACCEPTED_GENERATION_ATTEMPT] == G.ATTEMPT_ID_RETRY == "a2"
    assert [a["attempt_id"] for a in manifest[C.MANIFEST_KEY_RESUME_ATTEMPTS]] == ["a1", "a2"]
    root = cd.parents[1]
    recs = V.iter_jsonl(root / C.REQUEST_EVIDENCE_FILENAME)
    assert len(recs) == 1082 + 37 and sum(1 for r in recs if r["attempt_id"] == "a1") == 37        # the history stays on disk
    report = analyze(tmp_path, matrix=m)
    leg = report["legs"][V.leg_key(*key, 2048)]
    assert report["gates"]["I1_ledger"]["status"] == A.GATE_PASS, report["gates"]["I1_ledger"]["detail"]
    assert report["gates"]["I2_identity"]["status"] == A.GATE_PASS, report["gates"]["I2_identity"]["detail"]
    assert A.non_log_gates_clear(report), A.production_refusals(report)            # no .eval log read: not production (r3 S1)
    ev, led = leg["request_evidence"], leg["ledger"]
    assert ev["accepted_attempt"] == "a2" and ev["n_task_records"] == 1082 and ev["n_task_records_all_attempts"] == 1119
    assert ev["records_by_attempt"] == {"a1": 37, "a2": 1082}
    assert led["accepted_attempt"] == "a2" and led["n_forwarded"] == 1082 and led["rows_by_attempt"] == {"a1": 37, "a2": 1082}
    assert led["n_task_rows_all_attempts"] == 1119 and report["gates"]["I1_ledger"]["reconciliation"][leg["key"]]["n_forwarded"] == 1082
    # a manifest naming no accepted attempt is refused by BOTH gates, by name — the evidence cannot be scoped
    m1 = small_matrix(("gemma4e4b", "main", "none"))
    G.build_store(tmp_path / "noacc", row("gemma4e4b", "main", "none"), manifest_edit=lambda mf: mf.pop(C.MANIFEST_KEY_ACCEPTED_GENERATION_ATTEMPT))
    r1 = analyze(tmp_path / "noacc", matrix=m1)
    for gate in ("I1_ledger", "I2_identity"):
        assert r1["gates"][gate]["status"] == A.GATE_FAIL and C.MANIFEST_KEY_ACCEPTED_GENERATION_ATTEMPT in r1["gates"][gate]["detail"], gate
    # an accepted attempt the manifest never recorded is refused by name
    G.build_store(tmp_path / "unrec", row("gemma4e4b", "main", "none"), manifest_edit=lambda mf: mf.__setitem__(C.MANIFEST_KEY_ACCEPTED_GENERATION_ATTEMPT, "a9"))
    r2 = analyze(tmp_path / "unrec", matrix=m1)
    assert r2["gates"]["I2_identity"]["status"] == A.GATE_FAIL and "not a recorded attempt" in r2["gates"]["I2_identity"]["detail"]
    # a task record without its attempt stamp is refused by the contract helper, by name (never counted around)
    G.build_store(tmp_path / "nostamp", row("gemma4e4b", "main", "none"), request_evidence_edit=lambda rs: rs[0].pop(C.REQUEST_EVIDENCE_ATTEMPT_FIELD))
    r3 = analyze(tmp_path / "nostamp", matrix=m1)
    assert r3["gates"]["I2_identity"]["status"] == A.GATE_FAIL and C.REQUEST_EVIDENCE_ATTEMPT_FIELD in r3["gates"]["I2_identity"]["detail"]


def test_c1_a_correctly_completed_ladder_whose_rung_store_still_escalates_on_its_own_population_is_a_production_result(tmp_path):
    """C1, from the planner through the analysis: 24 at-cap draws of 1,082 promote; the 4,096 rung leaves 3 of its 24 redraws
    at cap (12.5 % of the RUNG STORE — the adapter's own exit is `escalate` to 8,192 — but 3 of 1,082 = 0.28 % of the MERGED
    reading). THE decision is the merged one: the planner files it INSIDE the rung record, plans no 8,192 rung, and the
    analysis follows the same field — completeness, I5 and I6 pass, the chain ends at 4,096 with the residual reported.
    Before the fix the analysis followed the persisted rung-only `exit`, demanded an 8,192 subset / store the planner never
    created, and a correctly completed ladder failed I0."""
    m = _one_cell_matrix(ESC)
    G.build_store(tmp_path, row("gemma4e4b", "main", "none"))
    # pass 1 through the REAL planner: the record + the 4,096 subset + the rung state
    G.build_store(tmp_path, row(*ESC), at_cap={(i, t) for i in range(12) for t in (0, 1)}, ladder=False)
    planned = LP.plan_all([row(*ESC)], wave_root=tmp_path, sha=G.FULL_SHA, from_cap=2048, classify=G.classify, served_seqs_of=lambda k: 64)
    assert planned[0]["exit"] == CL.ESCALATE and planned[0]["next_cap"] == 4096 and planned[0]["n_draws"] == 24
    subset_path = Path(planned[0]["subset_path"])
    subset = json.loads(subset_path.read_text())
    src_cd = completion.read_store_manifest(Path(planned[0]["out_root"]) / "cells" / next(p.name for p in (Path(planned[0]["out_root"]) / "cells").iterdir()))
    # the rung store: exactly the subset's 24 draws, 3 still at cap
    G.build_store(tmp_path, row(*ESC), cap=4096, draws=IC.draws_of_subset(subset), rung_source=src_cd, subset_doc=subset, subset_path=str(subset_path),
                  at_cap={(1, 0), (2, 0), (3, 0)}, completion_tokens={(G.IDS.index(i), t): 3000 for i, t in IC.draws_of_subset(subset)}, ladder=False)
    planned2 = LP.plan_all([row(*ESC)], wave_root=tmp_path, sha=G.FULL_SHA, from_cap=4096, classify=G.classify, served_seqs_of=lambda k: 64)
    p2 = planned2[0]
    assert p2["store_exit"] == CL.ESCALATE and p2["exit"] == CL.EXIT_BELOW_THRESHOLD and p2["line"] is None
    assert p2["decision"]["n_at_cap"] == 3 and p2["decision"]["n_draws"] == 1082 and p2["decision"]["n_rung"] == 24
    assert not (tmp_path / IC.ifeval_paths(*ESC, 8192).subset).exists() and not (tmp_path / IC.ifeval_paths(*ESC, 8192).ladder_state).exists()
    # the analysis follows THE decision: every non-log gate clear (no .eval log read here — r3 S1), the chain ending at 4,096
    report = analyze(tmp_path, matrix=m)
    assert A.non_log_gates_clear(report), A.production_refusals(report)
    c = report["cells"]["/".join(ESC)]
    assert c["ladder"]["promoted"] == [4096] and c["ladder"]["residual"]["final_cap"] == 4096
    assert c["ladder"]["residual"] == {"n_at_cap": 3, "share_at_cap": 3 / 1082, "final_cap": 4096, "final_exit": CL.EXIT_BELOW_THRESHOLD,
                                       "final_store_exit": CL.ESCALATE}
    assert c["ladder"]["exits"]["4096"]["exit"] == CL.EXIT_BELOW_THRESHOLD and c["ladder"]["exits"]["4096"]["store"]["exit"] == CL.ESCALATE
    # I5's identity half passes on every leg (no failure named); the gate reads PENDING with no .eval log read
    assert report["gates"]["I0_completeness"]["status"] == A.GATE_PASS and report["gates"]["I5_at_cap"]["status"] == A.GATE_PENDING
    assert report["gates"]["I5_at_cap"]["failures"] == [] and set(report["gates"]["I5_at_cap"]["legs"].values()) == {A.GATE_PASS}
    assert report["gates"]["I6_ladder"]["status"] == A.GATE_PASS and report["legs"][V.leg_key(*ESC, 4096)]["status"] == "usable"
    assert c["redraw_outcomes"] == {"4096": {"extended": 21, "still_at_cap": 3}}
    assert "(below_threshold; store escalate)" in A._ladder_table(report)
    # a record WITHOUT the decision block (a pre-C1 planner's) is refused by name — never read through the store's own exit
    rung_rec_path = tmp_path / IC.ifeval_paths(*ESC, 4096).cap_ladder
    rec = json.loads(rung_rec_path.read_text())
    rec.pop(IC.LADDER_DECISION_KEY)
    rung_rec_path.write_text(json.dumps(rec))
    report2 = analyze(tmp_path, matrix=m)
    assert report2["gates"]["I5_at_cap"]["status"] == A.GATE_FAIL and "no 'decision' block" in report2["gates"]["I5_at_cap"]["detail"]
    assert report2["gates"]["I6_ladder"]["status"] == A.GATE_FAIL and report2["verdict"]["production"] is False
    # a decision whose merged count disagrees with the analysis's own merge is refused by name (I6)
    rec3 = json.loads(rung_rec_path.read_text())
    rec3[IC.LADDER_DECISION_KEY] = {**IC.ladder_decision({**rec3, IC.LADDER_DECISION_KEY: p2["decision"]}), "n_at_cap": 4, "share_at_cap": round(4 / 1082, 6)}
    rung_rec_path.write_text(json.dumps(rec3))
    report3 = analyze(tmp_path, matrix=m)
    assert report3["gates"]["I6_ladder"]["status"] == A.GATE_FAIL and "the chain merges to 3 of 1082" in report3["gates"]["I6_ladder"]["detail"]


# --------------------------------------------------------------------------- figures, sidecars, the record, the CLI boundary
def test_figures_render_with_short_titles_tall_axes_resolvable_sidecars_and_the_provenance_in_every_caption(small_report, tmp_path):
    figs = A.make_figures(small_report, tmp_path / "figs")
    stems = set(figs["written"])
    # the two two-view models of the small matrix (gemma, llama) each get a noise-floor figure; qwen runs one view here
    assert {"final_ifeval_contrasts_gemma4e4b", "final_ifeval_contrasts_qwen35_4b", "final_ifeval_contrasts_llama8b", "final_ifeval_metrics_by_cell",
            "final_ifeval_shares", "final_ifeval_exits", "final_ifeval_ladder", "final_ifeval_truncation_cost", "final_ifeval_completion_tokens",
            "final_ifeval_ledger", "final_ifeval_cost", "final_ifeval_noise_floor_gemma4e4b", "final_ifeval_noise_floor_llama8b"} == stems
    for paths in figs["written"].values():
        assert {Path(p).suffix for p in paths} == {".png", ".pdf"} and all(Path(p).stat().st_size > 1000 for p in paths)
    nums = json.loads(Path(figs["numbers"]).read_text())
    assert nums["trigger_provenance"] == IC.IFEVAL_TRIGGER_PROVENANCE
    assert all("1 % raw at-cap of draws" in cap for cap in nums["captions"].values())
    for stem, entries in nums["figures"].items():
        for e in entries:
            assert A.T.report_path(small_report, e["source"]) == e["value"] or e["value"] is None, (stem, e)
    tables = json.loads(Path(figs["tables"]).read_text())
    assert set(tables) >= {"contrasts", "noise_floor", "truncation", "cells", "gates", "verdict", "held", "cost", "captions"}
    # --figure-exclude-family drops that family's contrast figure and its cells from the fixed figures
    rep2 = analyze(Path(small_report["gen_root"]), matrix=SMALL, exclude_families=["llama"])
    assert "final_ifeval_contrasts_llama8b" not in A.figures_for(rep2) and "final_ifeval_contrasts_gemma4e4b" in A.figures_for(rep2)
    assert "final_ifeval_noise_floor_llama8b" not in A.figures_for(rep2) and "final_ifeval_noise_floor_gemma4e4b" in A.figures_for(rep2)


def test_the_results_skeleton_renders_every_placeholder(small_report):
    md = A.render_results_md(small_report)
    # the fixture read no .eval log, so the render says so (r3 S1): every placeholder filled, the verdict word honest
    assert "{{" not in md and "NOT a production result" in md and G.SHA7 in md
    assert "1 % raw at-cap of draws, per the IFEval ruling" in md and "whitespace" in md.lower()
    assert "| gemma4e4b/main/constv2-mix-full__lora_r4_ce |" in md and "lower than none" in md
    skeleton = A.RESULTS_TEMPLATE.read_text()
    assert skeleton.count("{{") >= 20 and "at-cap, truncated and loop shares" in skeleton


def test_main_refuses_a_bare_no_eval_logs_pass_and_the_comparison_only_pass_certifies_i9_without_production(tmp_path, small_store, small_report):
    """Earlier this test ran a `--no-eval-logs` pass to a PRODUCTION report with figures and a rendered record —
    the bypass itself. Now: `--no-eval-logs` outside `--compare-only` is refused at the command line (exit 2, nothing written);
    `--compare-only` needs a reference and may name nothing it is not allowed to publish; the comparison-only pass exits 0 on equal
    numeric leaves with a report that claims no I4 / I5 pass and no production, 4 on a differing leaf, and 3 on any refusal beyond
    the unread logs (a torn ledger) BEFORE matching. The production path with the logs read is in test_ifeval_analysis_review_r3.py."""
    root = small_store.root
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps(G.IDS))
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps({"schema": IC.MATRIX_SCHEMA, "rows": SMALL}))
    common = ["--gen-root", str(root), "--sha", G.SHA7, "--matrix", str(matrix), "--ids", str(ids), "--bootstrap", str(B_FAST),
              "--wandb-mode", "disabled", "--no-mirror", "--analysis-sha", G.FULL_SHA]
    out1 = tmp_path / "r1.json"
    # (a) the bypass is gone: a production invocation with the logs unread is refused before any analysis, nothing written
    with pytest.raises(SystemExit) as e:
        A.main([*common, "--no-eval-logs", "--out", str(out1), "--figs-dir", str(tmp_path / "figs1"), "--results-md", str(tmp_path / "results.md")])
    assert e.value.code == 2 and not out1.exists() and not (tmp_path / "figs1").exists() and not (tmp_path / "results.md").exists()
    # (b) --compare-only needs a reference and publishes nothing: a record, a figures dir or a W&B name is refused by name
    ref = tmp_path / "reference.json"
    ref.write_text(json.dumps(small_report, default=str))
    for bad in (["--compare-only", "--no-eval-logs"],
                ["--compare-only", "--require-report-match", str(ref), "--results-md", str(tmp_path / "x.md")],
                ["--compare-only", "--require-report-match", str(ref), "--figs-dir", str(tmp_path / "figsx")],
                ["--compare-only", "--require-report-match", str(ref), "--wandb-name", "x"]):
        with pytest.raises(SystemExit) as e:
            A.main([*common, *bad, "--out", str(tmp_path / "never.json")])
        assert e.value.code == 2 and not (tmp_path / "never.json").exists(), bad
    # (c) the comparison-only pass: I9 PASSES on equal numeric leaves; the report claims no I4 / I5 pass and no production; nothing published
    out2 = tmp_path / "r2.json"
    rc = A.main([*common, "--compare-only", "--no-eval-logs", "--require-report-match", str(ref), "--out", str(out2)])
    r2 = json.loads(out2.read_text())
    assert rc == 0 and r2["gates"]["I9_report_match"]["status"] == A.GATE_PASS and r2["report_match"]["equal"]
    assert r2["verdict"]["production"] is False and r2["verdict"]["comparison_only"] is True and r2["verdict"]["eval_logs_validated"] is False
    assert r2["gates"]["I4_scoring_path"]["status"] == r2["gates"]["I5_at_cap"]["status"] == A.GATE_PENDING
    assert r2["verdict"]["gates_pending_log_derived"] == list(A.LOG_DERIVED_GATES)
    assert r2["comparison_only"]["mode"] == A.COMPARISON_ONLY_MODE and r2["comparison_only"]["eval_logs_read"] is False
    assert r2["comparison_only"]["reference_production"] is False                    # the reference here read no log either
    assert "figures" not in r2 and "mirrored" not in r2 and "wandb_url" not in r2 and not (out2.parent / "figs").exists()
    assert A.main(["--gate-report", str(out2)]) == A.EXIT_NOT_PRODUCTION == 3       # a comparison-only report is never production
    # (d) a differing reference -> exit 4, the report written with I9 FAILED
    other = json.loads(ref.read_text())
    other["cells"]["/".join(ESC)]["shares"]["n_at_cap"] = 999
    (tmp_path / "other.json").write_text(json.dumps(other))
    out3 = tmp_path / "r3.json"
    rc = A.main([*common, "--compare-only", "--no-eval-logs", "--require-report-match", str(tmp_path / "other.json"), "--out", str(out3)])
    assert rc == A.EXIT_REPORTS_DIFFER == 4 and json.loads(out3.read_text())["gates"]["I9_report_match"]["status"] == A.GATE_FAIL
    # (e) a refusal beyond the unread logs (a torn ledger) -> exit 3 BEFORE the match, nothing published
    torn = tmp_path / "torn"
    G.build_store(torn, row("gemma4e4b", "main", "none"), ledger_edit=lambda rows: rows[0].__setitem__("finish_reason", "length"))
    m1 = tmp_path / "m1.json"
    m1.write_text(json.dumps({"schema": IC.MATRIX_SCHEMA, "rows": small_matrix(("gemma4e4b", "main", "none"))}))
    rc = A.main(["--gen-root", str(torn), "--sha", G.SHA7, "--matrix", str(m1), "--ids", str(ids), "--bootstrap", "50", "--compare-only", "--no-eval-logs",
                 "--require-report-match", str(ref), "--wandb-mode", "disabled", "--no-mirror", "--out", str(tmp_path / "r4.json")])
    r4 = json.loads((tmp_path / "r4.json").read_text())
    assert rc == A.EXIT_NOT_PRODUCTION == 3 and r4["gates"]["I1_ledger"]["status"] == A.GATE_FAIL and "report_match" not in r4
    assert A.main(["--gate-report", str(tmp_path / "r4.json")]) == 3
    assert A.main(["--diff-reports", str(ref), str(out2)]) == 0


@pytest.mark.skipif(not HAS_INSPECT, reason="needs inspect_ai for the .eval cross-check")
def test_the_eval_log_pass_reads_both_sample_logs_and_agrees_with_the_outcome_record(tmp_path):
    m = small_matrix(("gemma4e4b", "main", "none"))
    cd = G.build_store(tmp_path, row("gemma4e4b", "main", "none"), at_cap={(3, 1)})
    G.write_inspect_logs(cd)
    report = analyze(tmp_path, matrix=m, read_eval_logs=True)
    leg = report["legs"][V.leg_key("gemma4e4b", "main", "none", 2048)]
    assert leg["eval_check"]["passed"] is True and leg["eval_check"]["n_log_samples"] == 1082
    assert report["gates"]["I4_scoring_path"]["status"] == A.GATE_PASS and report["gates"]["I4_scoring_path"]["eval_logs_read"] is True
    # an altered score in the log disagrees with the outcome record by name
    G.write_inspect_log(cd, name="altered", sample_idx=0, value_edit=lambda iid, v: {**v, "prompt_level_strict": not v["prompt_level_strict"]})
    report = analyze(tmp_path, matrix=m, read_eval_logs=True)
    assert report["gates"]["I4_scoring_path"]["status"] == A.GATE_FAIL


def test_the_readers_constants_are_the_producers_own_by_import():
    from sisteer.capeval.tiers import inspect_rung as IR
    assert V.TRIGGER_RULE is IR.IFEVAL_TRIGGER_RULE and V.TRIGGER_THRESHOLD == IR.IFEVAL_INSTRUMENT.threshold == 0.01
    assert V.TRIGGER_PROVENANCE == IC.IFEVAL_TRIGGER_PROVENANCE and V.LEDGER_FINISH_AT_CAP == CL.LEDGER_FINISH_AT_CAP == "length"
    assert A.GATE_NAMES == ("I0_completeness", "I1_ledger", "I2_identity", "I3_certification", "I4_scoring_path", "I5_at_cap", "I6_ladder",
                            "I7_population", "I8_two_reading", "I9_report_match")
    assert A.READINGS == ("r2048", "merged") and A.METRICS == ("prompt_strict", "prompt_loose", "inst_strict", "inst_loose")
    assert V.load_matrix(REPO / IC.IFEVAL_MATRIX_REL)["counts"] == {"rows": 40, "launched": 30, "held": 2, "deferred": 8, "fitted_launched": 13, "contrasts_designed": 23}
    with pytest.raises(A.AnalysisError, match="expected_stamps"):
        V.load_matrix([{**ROWS[0], "expected_stamps": ["x", "y", "z"]}])



def test_collect_leg_reads_the_modal_launchers_real_store_shape(tmp_path):
    """The wave shakedown (2026-09-13) and the rung wave (V20 / V36 / V41): the Modal launcher writes NO endpoint.json — the standup's
    endpoint record sits INSIDE `served_geometry.json` (`endpoint`) beside the served `geometry` block; the thinking pin AS SERVED is
    the driver's `evidence.json` `serve_header.template_kwargs`; the `*_mean_by_trial` summary fields are LISTS indexed by trial. The
    fixture writes exactly that shape (it once wrote an endpoint.json with `default_chat_template_kwargs` and `geometry.eval`, keys the
    producer never writes — the reader that read them refused every real store), and the reader takes it; a store with only the old
    cluster-lane endpoint.json is refused by name."""
    r = row("qwen35_4b", "main", "none")
    cd = G.build_store(tmp_path, r)
    root = cd.parent.parent
    assert not (root / "endpoint.json").exists()
    served = json.loads((root / "served_geometry.json").read_text(encoding="utf-8"))
    assert set(served) >= {"geometry", "endpoint", "standup_argv"} and served["geometry"]["eval"] == "capability:ifeval_2048"
    assert served["geometry"]["window_source"] == "table" and served["endpoint"]["geometry"]["eval"] is None      # the launcher's two spellings
    assert "default_chat_template_kwargs" not in served["endpoint"]
    ev = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
    assert ev["serve_header"]["template_kwargs"] == {"enable_thinking": False}
    sp = completion.task_summary_path(cd, IC.IFEVAL_TASK)
    summary = json.loads(sp.read_text(encoding="utf-8"))
    keys = [k for k in summary if k.endswith("_mean_by_trial")]
    assert keys and all(isinstance(summary[k], dict) for k in keys)
    for k in keys:
        summary[k] = [summary[k].get(str(t)) for t in range(IC.IFEVAL_SAMPLES_PER_ITEM)]      # the producer's list
    sp.write_text(json.dumps(summary), encoding="utf-8")
    assert V.by_trial_value([0.1, 0.2], 1) == 0.2 and V.by_trial_value({"1": 0.2}, 1) == 0.2
    assert V.by_trial_value([0.1], 1) is None and V.by_trial_value(None, 0) is None
    read_logs = importlib.util.find_spec("inspect_ai") is not None
    if read_logs:
        G.write_inspect_logs(cd)                       # the REAL .eval logs the cross-check reads against the list-shaped means
    ctx = A.Context(canonical_ids=None, pins=None, expected_refusals={}, read_eval_logs=read_logs)
    leg = A.collect_leg(tmp_path, r, cap=G.CAP1, ctx=ctx)
    assert leg["status"] == "usable", leg.get("refused")
    cert = leg["certification"]
    assert cert["passed"], cert["failures"]
    assert cert["geometry_eval"] == "capability:ifeval_2048" and cert["template_kwargs"] == {"enable_thinking": False} and cert["thinking_switch"] is True
    assert cert["served_seqs"] == served["geometry"]["max_num_seqs"] and cert["window_source"] == "table" and cert["seqs_source"] == "model_budget"
    assert cert["manifest_geometry_source"]["window_source"] == "override"          # reconciled, never the contract
    assert leg["endpoint"] == served["endpoint"] and leg["recorder"]["verdict"] == "AGREES" and leg["result"]["result"] == "COMPLETE"
    if read_logs:
        assert leg["eval_check"]["passed"], leg["eval_check"]["failures"]
    # the old cluster-lane shape (an endpoint.json alone) is not the producer's: refused by name
    (root / "served_geometry.json").rename(root / "served_geometry.json.bak")
    (root / "endpoint.json").write_text(json.dumps(served["endpoint"]), encoding="utf-8")
    leg2 = A.collect_leg(tmp_path, r, cap=G.CAP1, ctx=A.Context(canonical_ids=None, pins=None, expected_refusals={}, read_eval_logs=False))
    assert leg2["status"] == "usable" and not leg2["certification"]["passed"] and "no served_geometry.json" in leg2["certification"]["failures"][0]
