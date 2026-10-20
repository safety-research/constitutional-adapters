"""ifeval-canonical-battery — the ANALYSIS revision after the opening review (decisions-analysis N5; nine findings, all accepted).

Synthetic stores only (``tests/ifeval_synthetic_store.py``: placeholder shas, invented items, ``TESTHOST``), written through the
REAL producers and read by the REAL analysis. Pinned, one block per finding:

* H1 — the served object's FIT CONDITION is read from the manifest's ``fit_provenance`` (fit ``model_id`` / ``revision`` vs
  ``served_model_id`` / ``served_revision`` and the producer's ``transfer`` flag), NEVER from the bank-view name: an ``it``-view
  LoRA whose manifest files a fit on the served weights reads "no transfer"; a ``main``-view LoRA fit on other weights reads
  "a transfer"; an ``it``-view manifest filing base weights still reads "a transfer"; a ``transfer`` flag disagreeing with the
  identity fails I3 by name; the exits caption and the record's served-object paragraph name the condition from the manifest.
* M2 — the pass-1 attestation is SCOPED to pass 1; the merged reading is attested separately per pair: a pair whose two views'
  ladders re-drew different draws keeps its pass-1 ``noise_floor`` label and is labelled the variability of the whole adaptive
  protocol under the merged reading, with the count of final draws differing in cap; a pair with no promotion is trivially
  identical and says so; the table and caption carry both scopes.
* M3 — the final readout counts the at-cap responses that pass prompt strict as scored and states the protocol (loops never
  re-drawn); the captions carry the observed net change with its CI and no recoverability claim.
* M4 — the stock-score cross-check refuses a positive-to-positive move (1 vs 2 followed instructions) on any of the five fields.
* M5 — I5 re-derives the loop rule from the retained texts: a loop draw moved into the eligible list with consistent counts is
  refused by name; a tampered ``text_sha256`` is refused; the clean store re-classifies every at-cap draw and verifies every hash;
  without the logs the gate is PENDING and says so (``texts_reclassified`` false; since revision 3 never PASSED).
* M6 — the thinking-OFF pin AS SERVED is generated per family from the legs (gemma: a switch, the OFF pin composed; llama: no
  switch, nothing composed); the KV-shrink legs are listed by name; the record label is distinguished from the admitted trees.
* L7 — the numbers sidecar carries every plotted endpoint (CI bounds, r2048 marks, quantile band ends, floor bands, estimates),
  each resolving in the report; the W&B images carry their captions.
* L8 — the final table carries the stock per-trial stderr, the mean / p99 completion tokens and the decode throughput with their
  reading labels; the subset figures name what they leave out in a footnote and their captions.
* L9 / the record — the prose filler fills every slot from the report, writes the revision block and the conditional-inference
  limitation, and states the served-object finding from the manifests.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from sisteer.capeval.tiers import cap_ladder as CL
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import ifeval_contracts as IC

import ifeval_synthetic_store as G

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


A = _load("tinycap_ifeval_battery", "scripts/tinycap_ifeval_battery.py")
P = _load("tinycap_ifeval_record_prose", "scripts/tinycap_ifeval_record_prose.py")
V = A.V
ROWS = G.matrix_rows()
B_FAST = 150
HAS_INSPECT = importlib.util.find_spec("inspect_ai") is not None
LLAMA, NONE, CONST, LORA = "llama8b", IC.CELL_NONE, IC.CELL_PROMPTED_CONSTITUTION, "constv2-mix-full__lora_r4_ce"
NONE_G = ("gemma4e4b", "main", NONE)


def row(*t):
    return G.row_for(ROWS, *t)


def _rows(*triples):
    out = [dict(r) for r in ROWS if (r["model_key"], r["view"], r["cell"]) in set(triples)]
    for i, r in enumerate(out):
        r["index"] = i
    return out


def analyze(root, rows, **kw):
    kw.setdefault("canonical_ids", G.IDS)
    kw.setdefault("n_resamples", B_FAST)
    kw.setdefault("read_eval_logs", False)
    kw.setdefault("analysis_sha", G.FULL_SHA)
    return A.analyze(root, kw.pop("sha", G.SHA7), matrix=rows, **kw)


def _ceiling_cell(root, r, *, shift: int = 0, **common):
    """A cell whose merged residual stays above 1 % at 8,192 (many pass-1 loops never re-drawn): 30 at cap, 20 loops, 10 eligible ->
    5 still at cap at 4,096 -> 1 still at cap at 8,192; residual 21 / 1,082 -> a CEILING exit. ``shift`` moves the item indices so
    two views of one cell re-draw DIFFERENT draws."""
    at_cap = {(i + shift, t) for i in range(15) for t in (0, 1)}
    loops = {(i + shift, t) for i in range(10) for t in (0, 1)}
    return G.build_escalated_cell(root, r, at_cap=at_cap, loops=loops, rung_still_at_cap={(i + shift, 0) for i in range(10, 15)},
                                  rung8192=True, rung8192_still_at_cap={(10 + shift, 0)}, **common)


@pytest.fixture(scope="module")
def llama_store(tmp_path_factory):
    """llama on both views: `none` anchors, the LoRA on `main` (a base-weight fit: a transfer) and on `it` (fit on the served weights)
    — the `it` LoRA at the ceiling, the `main` LoRA below threshold; plus the prompted-constitution pair PROMOTED on both views over
    DIFFERENT draws (the merged attestation's case)."""
    root = tmp_path_factory.mktemp("ife_r2")
    for v in ("main", "it"):
        G.build_store(root, row(LLAMA, v, NONE))
    G.build_store(root, row(LLAMA, "main", LORA), at_cap={(i, 0) for i in range(4)}, loops={(0, 0)})
    _ceiling_cell(root, row(LLAMA, "it", LORA))
    _ceiling_cell(root, row(LLAMA, "main", CONST), shift=0)
    _ceiling_cell(root, row(LLAMA, "it", CONST), shift=3)
    return root


LLAMA_ROWS = _rows((LLAMA, "main", NONE), (LLAMA, "it", NONE), (LLAMA, "main", LORA), (LLAMA, "it", LORA), (LLAMA, "main", CONST), (LLAMA, "it", CONST))


@pytest.fixture(scope="module")
def llama_report(llama_store):
    """The fixture reads NO .eval log (the CPU suite has no inspect_ai), so since revision 3 it is NOT a production
    report: I4 / I5 are PENDING and never waivable. Every other gate is clear — that is what these tests need; the production
    path with the logs read is pinned in ``test_ifeval_analysis_review_r3.py``."""
    rep = analyze(llama_store, LLAMA_ROWS)
    assert A.non_log_gates_clear(rep), A.production_refusals(rep)
    assert rep["verdict"]["production"] is False and rep["verdict"]["gates_pending_log_derived"] == list(A.LOG_DERIVED_GATES)
    return rep


# --------------------------------------------------------------------------- H1: the fit condition from the manifest, never the view
def test_the_fit_condition_is_read_from_the_manifest_and_the_it_view_lora_is_a_fit_on_the_served_weights(llama_report, llama_store):
    rep = llama_report
    it, main = rep["cells"][V.cell_key(LLAMA, "it", LORA)]["fit_provenance"], rep["cells"][V.cell_key(LLAMA, "main", LORA)]["fit_provenance"]
    served_hf, served_rev = IC.IFEVAL_MODEL_PINS[LLAMA]
    assert it["recorded"] and it["model_id"] == it["served_model_id"] == served_hf and it["revision"] == it["served_revision"] == served_rev
    assert it["transfer"] is False and it["chat_template_bypassed"] is True and it["fit_kind"] == "document"
    assert it["fit_weights"] == "served_checkpoint" and it["condition"] == V.FIT_CONDITION_SAME_WEIGHTS and it["problem"] is None
    assert main["model_id"] == G.BASE_FIT_PINS[LLAMA][0] != main["served_model_id"] == served_hf and main["transfer"] is True
    assert main["fit_weights"] == "other_checkpoint" and main["condition"] == V.FIT_CONDITION_TRANSFER
    none = rep["cells"][V.cell_key(LLAMA, "it", NONE)]["fit_provenance"]
    assert none["condition"] == V.FIT_CONDITION_NO_OBJECT and none["model_id"] is None and none["recorded"]
    assert "never inferred from the bank-view name" in it["rule"]
    # the legs carry it too, and the record's A.7 table names fit vs served per cell
    assert rep["legs"][V.leg_key(LLAMA, "it", LORA, 8192)]["fit_provenance"]["condition"] == V.FIT_CONDITION_SAME_WEIGHTS
    table = A._fit_table(rep)
    assert f"| {V.cell_key(LLAMA, 'it', LORA)} |" in table and "| False | True |" in table and V.FIT_CONDITION_SAME_WEIGHTS in table
    assert f"`{G.BASE_FIT_PINS[LLAMA][0]}`" in table and V.FIT_CONDITION_TRANSFER in table
    # the exits caption states the ceiling cell's condition FROM THE MANIFEST, with the observed net change and the passing at-cap count
    cap = A._exits_caption(rep, "final_ifeval_exits")
    fr = rep["final_readouts"][V.cell_key(LLAMA, "it", LORA)]
    assert fr["exit"] == CL.EXIT_CEILING and "document-fit LoRA on the served weights themselves" in cap and "transfer = False" in cap
    assert "base-fit" not in cap and "transfer objects" not in cap and "not a reason to widen" not in cap
    assert f"{fr['n_at_cap_pass_prompt_strict']} of {fr['n_at_cap']} at-cap responses pass prompt strict" in cap and "was not measured" in cap
    assert "observed net change of the ladder (merged - r2048, prompt strict)" in cap


def test_a_view_name_never_decides_the_condition_and_a_disagreeing_transfer_flag_fails_i3_by_name(tmp_path):
    rows = _rows((LLAMA, "it", NONE), (LLAMA, "it", LORA))
    G.build_store(tmp_path / "base_on_it", row(LLAMA, "it", NONE))
    base_hf, base_rev = G.BASE_FIT_PINS[LLAMA]

    def base_weights(m):
        m["fit_provenance"].update(model_id=base_hf, revision=base_rev, transfer=True)
    G.build_store(tmp_path / "base_on_it", row(LLAMA, "it", LORA), manifest_edit=base_weights)
    rep = analyze(tmp_path / "base_on_it", rows)
    fp = rep["cells"][V.cell_key(LLAMA, "it", LORA)]["fit_provenance"]
    assert fp["condition"] == V.FIT_CONDITION_TRANSFER and fp["model_id"] == base_hf and fp["problem"] is None     # the view says `it`; the manifest decides
    assert rep["gates"]["I3_certification"]["status"] == A.GATE_PASS
    # the producer's flag disagreeing with the fit-vs-served identity is a contract violation, named on I3
    G.build_store(tmp_path / "flag", row(LLAMA, "it", NONE))
    G.build_store(tmp_path / "flag", row(LLAMA, "it", LORA), manifest_edit=lambda m: m["fit_provenance"].update(transfer=True))
    rep2 = analyze(tmp_path / "flag", rows)
    fp2 = rep2["cells"][V.cell_key(LLAMA, "it", LORA)]["fit_provenance"]
    assert fp2["condition"] == V.FIT_CONDITION_SAME_WEIGHTS and "transfer flag True disagrees" in fp2["problem"]
    assert rep2["gates"]["I3_certification"]["status"] == A.GATE_FAIL and "fit provenance" in rep2["gates"]["I3_certification"]["detail"]
    # a fitted cell with no fit_provenance block at all: the condition is "not recorded" and I3 refuses (what was fit on is not on record)
    G.build_store(tmp_path / "none", row(LLAMA, "it", NONE))
    G.build_store(tmp_path / "none", row(LLAMA, "it", LORA), manifest_edit=lambda m: m.pop("fit_provenance"))
    rep3 = analyze(tmp_path / "none", rows)
    assert rep3["cells"][V.cell_key(LLAMA, "it", LORA)]["fit_provenance"]["condition"] == V.FIT_CONDITION_UNRECORDED
    assert rep3["gates"]["I3_certification"]["status"] == A.GATE_FAIL
    # a bare cell without the block is fine (nothing was fit)
    G.build_store(tmp_path / "bare", row(LLAMA, "it", NONE), manifest_edit=lambda m: m.pop("fit_provenance"))
    rep4 = analyze(tmp_path / "bare", _rows((LLAMA, "it", NONE)))
    assert rep4["gates"]["I3_certification"]["status"] == A.GATE_PASS and rep4["cells"][V.cell_key(LLAMA, "it", NONE)]["fit_provenance"]["recorded"] is False


# --------------------------------------------------------------------------- M2: the two scopes of the noise-floor attestation
def test_the_pass1_attestation_is_scoped_and_the_merged_reading_of_a_pair_that_redrew_differently_is_the_protocols_variability(llama_report):
    rep = llama_report
    none = A.noise_floor_pair(rep, LLAMA, NONE)
    const = A.noise_floor_pair(rep, LLAMA, CONST)
    assert none["label"] == const["label"] == A.LABEL_FLOOR                     # the pass-1 attestation, unchanged
    assert none["label_scope"] == const["label_scope"] == none["attestation"]["scope"] == A.ATTESTATION_SCOPE_PASS1 and "2048" in A.ATTESTATION_SCOPE_PASS1
    ma_none, ma_const = none["merged_attestation"], const["merged_attestation"]
    # no promotion on either view: the merged reading IS pass 1 — byte-identical, trivially, and said so
    assert ma_none["trivially_identical"] and ma_none["identical"] and ma_none["label"] == A.MERGED_LABEL_BYTE_IDENTICAL == none["merged_label"]
    assert ma_none["n_final_cap_differs"] == 0 and ma_none["n_final_body_identical"] == 1082 == ma_none["n_draws"] and "no promotion on either" in none["merged_label_text"]
    # both views promoted to [4096, 8192] over DIFFERENT draws (the fixture shifts the `it` view's at-cap items by 3): the pass-1 label
    # stays a floor (same promotions, same 2,048 request bodies) while the merged reading is the whole adaptive protocol's variability
    assert ma_const["promotions"] == {"main": [4096, 8192], "it": [4096, 8192]}
    # the chains are RECORDED beside the pass-1 attestation and never decide its label (the unequal-chain case is in r3's tests)
    lp = const["attestation"]["ladder_promotions"]
    assert lp["main"] == lp["it"] == [4096, 8192] and lp["equal"] is True and lp["deciding"] is False
    assert "ladder_promotions" not in {c["name"] for c in const["attestation"]["checks"]}
    assert not ma_const["identical"] and ma_const["label"] == A.MERGED_LABEL_PROTOCOL == const["merged_label"]
    assert ma_const["n_final_cap_differs"] > 0 and ma_const["n_final_cap_differs"] == ma_const["n_differing_draws"]
    assert all(d["main"]["cap"] != d["it"]["cap"] for d in ma_const["differing_draws"]) and ma_const["n_final_body_identical"] == 1082 - ma_const["n_final_cap_differs"]
    assert "re-drew differently" in const["merged_label_text"] and "variability of the whole adaptive protocol" in const["merged_label_text"]
    assert "never changed to force equality" in ma_const["rule"]
    # the table carries both scopes; the caption states the merged scope per pair; the contrast caption says the none band is untouched
    table = A._noise_table(rep)
    assert "label (pass-1 attestation: 2,048 request bodies) | merged reading |" in table
    assert f"| {LLAMA} | {NONE}(it) - {NONE}(main) | {A.LABEL_FLOOR} | = pass 1 (no promotion on either view); 1082 of 1082 final request bodies identical |" in table
    assert f"| {LLAMA} | {CONST}(it) - {CONST}(main) | {A.LABEL_FLOOR} | **variability of the whole adaptive protocol, not a byte-identical-request floor**: {ma_const['n_final_cap_differs']} of 1082 final draws differ in cap" in table
    cap = A._noise_floor_caption(rep, f"{A.NOISE_FLOOR_STEM}_{LLAMA}", LLAMA)
    assert "on pass 1 (2,048 request bodies)" in cap and "the merged reading is the pass-1 reading (no promotion on either view)" in cap
    assert f"{ma_const['n_final_cap_differs']} of 1082 final draws differ in cap" in cap and "two PASS-ONE stores" in cap
    ccap = A._contrast_caption(rep, f"final_ifeval_contrasts_{LLAMA}", LLAMA)
    assert "does not touch this band" in ccap
    # the merged reading's numbers are still filed for both pairs (the pair is not dropped, only labelled)
    assert const["readings"][A.READING_MERGED]["prompt_strict"]["n_items"] == 541


# --------------------------------------------------------------------------- M3: the loop finding stated operationally
def test_the_final_readout_counts_at_cap_responses_that_pass_and_the_captions_claim_only_the_observed_net_change(llama_report):
    rep = llama_report
    k = V.cell_key(LLAMA, "it", LORA)
    fr = rep["final_readouts"][k]
    c = rep["cells"][k]
    assert fr["loops_redrawn"] is False and "never re-drawn" in fr["protocol"] and "was not measured" in fr["protocol"]
    assert 0 <= fr["n_loop_pass_prompt_strict"] <= fr["n_at_cap_pass_prompt_strict"] <= fr["n_at_cap"] and fr["n_at_cap_pass_prompt_loose"] >= fr["n_at_cap_pass_prompt_strict"]
    # the fixture's at-cap draws fail one instruction each, so the loops (at cap) that still pass are those whose remaining instruction count passes
    assert fr["n_at_cap"] == 21 and fr["n_loop"] == 20 and fr["n_truncated"] == 1
    tcap = A._truncation_caption(rep, "final_ifeval_truncation_cost")
    assert "observed net change" in tcap and "loops were never re-drawn" in tcap and "not a bound" in tcap and "bought back" not in tcap
    ecap = A._exits_caption(rep, "final_ifeval_exits")
    assert "cannot finish" not in ecap and "widen" not in ecap and A._net_change(rep, k) in ecap
    assert A._net_change(rep, V.cell_key(LLAMA, "main", LORA)) == "not promoted"
    table = A._final_table(rep)
    assert "| at-cap passing prompt strict |" in table and f"| {fr['n_at_cap_pass_prompt_strict']} of {fr['n_at_cap']} |" in table


# --------------------------------------------------------------------------- M4: exact integer equality on the five stock-score fields
@pytest.mark.skipif(not HAS_INSPECT, reason="the .eval cross-check needs inspect_ai")
def test_a_positive_to_positive_stock_score_move_is_refused_by_the_cross_check(tmp_path):
    m = _rows(NONE_G)
    cd = G.build_store(tmp_path, row(*NONE_G), at_cap={(3, 1)})
    G.write_inspect_logs(cd)
    rep = analyze(tmp_path, m, read_eval_logs=True)
    assert rep["gates"]["I4_scoring_path"]["status"] == A.GATE_PASS
    # 1 -> 2 followed instructions: an integer inequality with the SAME truthiness — the r1 reader let it pass
    moved = []

    def bump(iid, v):
        if 1 <= int(v["inst_level_strict"]) and len(moved) < 3:
            moved.append(iid)
            return {**v, "inst_level_strict": int(v["inst_level_strict"]) + 1}
        return v
    G.write_inspect_log(cd, name="bumped", sample_idx=0, value_edit=bump)
    assert moved
    rep2 = analyze(tmp_path, m, read_eval_logs=True)
    g = rep2["gates"]["I4_scoring_path"]
    assert g["status"] == A.GATE_FAIL and "inst_level_strict" in g["detail"] and "!= the outcome item's" in g["detail"]
    # and the stock aggregate metrics of each trial's log are read for comparability (labelled, never deciding)
    leg = rep["legs"][V.leg_key(*NONE_G, 2048)]
    stock = leg["eval_check"]["stock_metrics_by_trial"]
    assert set(stock) == {"0", "1"} and all(isinstance(stock[t]["prompt_strict_stderr"], float) and stock[t]["prompt_strict_stderr"] > 0 for t in stock)
    assert set(stock["0"]) == set(V.STOCK_METRIC_NAMES) and "never decide" in leg["eval_check"]["stock_metrics_note"]
    assert A._stock_cell(rep, V.cell_key(*NONE_G)).startswith("t0: ") and "±" in A._stock_cell(rep, V.cell_key(*NONE_G))
    assert A._stock_cell(rep2, V.cell_key(*NONE_G)) != "logs not read"
    # a log without a results block reads "not recorded", never zero
    assert V.stock_metrics_of_log(types.SimpleNamespace(results=None)) is None


# --------------------------------------------------------------------------- M5: the loop rule re-derived from the retained texts
def _consistent_move(rec, key):
    """Move ONE loop draw into the eligible list with every count and share kept consistent (the mutation the r1 gate accepted)."""
    rec["loop_draws"] = [d for d in rec["loop_draws"] if (d[0], d[1]) != key]
    rec["eligible_draws"] = sorted(rec["eligible_draws"] + [list(key)])
    rec["n_loop"] -= 1
    rec["n_truncated"] += 1
    rec["n_eligible"] = rec["n_truncated"]
    rec["n_non_terminating"] = rec["n_loop"]
    for share, count in (("share_truncated", "n_truncated"), ("share_loop", "n_loop"), ("share_eligible", "n_eligible")):
        rec[share] = round(rec[count] / rec["n_draws"], 6)
    rec["eligible_ids"] = sorted({i for i, _ in rec["eligible_draws"]})
    rec["non_terminating_ids"] = sorted({i for i, _ in rec["loop_draws"]})
    for f in rec["item_facts"]:
        if (f["item_id"], f["sample_idx"]) == key:
            f["kind"], f["eligible"] = "truncated", True
    if rec.get("decision"):
        rec["decision"]["eligible_draws"] = rec["eligible_draws"]
        rec["decision"]["n_eligible"] = rec["n_eligible"]


@pytest.mark.skipif(not HAS_INSPECT, reason="re-deriving the loop rule needs the retained .eval texts (inspect_ai)")
def test_i5_rederives_the_loop_rule_from_the_retained_texts_and_refuses_a_consistent_list_move_and_a_tampered_hash(tmp_path):
    m = _rows(NONE_G)
    cd = G.build_store(tmp_path, row(*NONE_G), at_cap={(i, 0) for i in range(5)}, loops={(0, 0), (1, 0)})
    G.write_inspect_logs(cd)
    rep = analyze(tmp_path, m, read_eval_logs=True)
    g = rep["gates"]["I5_at_cap"]
    leg = rep["legs"][V.leg_key(*NONE_G, 2048)]
    assert g["status"] == A.GATE_PASS and g["texts_reclassified"] is True and g["n_at_cap_reclassified"] == 5
    at = leg["at_cap_text"]
    assert at["passed"] is True and at["n_at_cap"] == 5 == at["n_text_sha256_verified"] and at["n_text_sha256_mismatch"] == 0
    assert at["n_loop_derived"] == 2 == at["n_loop_recorded"] and at["n_eligible_derived"] == 3 == at["n_eligible_recorded"] and at["rule"] == CL.DEGENERATE_RULE
    assert g["reclassification"][V.leg_key(*NONE_G, 2048)]["n_loop_derived"] == 2
    # (a) a loop draw moved into the eligible list with EVERY count kept consistent: the record is internally consistent (the r1 check
    # passes) — the re-derivation refuses it by name
    rec_path = tmp_path / IC.ifeval_paths(*NONE_G, 2048).cap_ladder
    rec = json.loads(rec_path.read_text())
    key = (G.IDS[0], 0)
    _consistent_move(rec, key)
    rec_path.write_text(json.dumps(rec))
    rep2 = analyze(tmp_path, m, read_eval_logs=True)
    leg2 = rep2["legs"][V.leg_key(*NONE_G, 2048)]
    assert leg2["ladder_problem"] is None                                   # internally consistent: the r1 gate would have passed
    assert rep2["gates"]["I5_at_cap"]["status"] == A.GATE_FAIL and "re-derived from the retained texts" in rep2["gates"]["I5_at_cap"]["detail"]
    assert leg2["at_cap_text"]["n_loop_derived"] == 2 and leg2["at_cap_text"]["n_loop_recorded"] == 1
    assert any(str(key[0]) in f for f in leg2["at_cap_text"]["failures"])
    # without the logs the gate cannot re-derive: it is PENDING, never PASSED (the earlier revision's test here asserted PASSED, which
    # let a no-log pass certify the moved list), the pending is not waivable and the report is not production
    rep2b = analyze(tmp_path, m, read_eval_logs=False)
    assert rep2b["gates"]["I5_at_cap"]["status"] == A.GATE_PENDING and rep2b["gates"]["I5_at_cap"]["texts_reclassified"] is False
    assert "not re-derived" in rep2b["legs"][V.leg_key(*NONE_G, 2048)]["at_cap_text"]["note"]
    assert rep2b["verdict"]["production"] is False and "I5_at_cap" in rep2b["verdict"]["gates_pending_log_derived"]
    # (b) a tampered text hash on one at-cap draw (the store rebuilt clean first; the builder re-files the record)
    G.build_store(tmp_path, row(*NONE_G), at_cap={(i, 0) for i in range(5)}, loops={(0, 0), (1, 0)})
    G.write_inspect_logs(cd)
    rec = json.loads(rec_path.read_text())
    for f in rec["item_facts"]:
        if (f["item_id"], f["sample_idx"]) == (G.IDS[2], 0):
            f["text_sha256"] = "0" * 64
    rec_path.write_text(json.dumps(rec))
    rep3 = analyze(tmp_path, m, read_eval_logs=True)
    at3 = rep3["legs"][V.leg_key(*NONE_G, 2048)]["at_cap_text"]
    assert rep3["gates"]["I5_at_cap"]["status"] == A.GATE_FAIL and at3["n_text_sha256_mismatch"] == 1 and "sha256" in rep3["gates"]["I5_at_cap"]["detail"]
    # the two-venv comparison leaves the log-derived block out (a venv without inspect_ai reads neither it nor `eval_check`)
    cmp = A.comparable_report(rep3)
    assert "at_cap_text" not in cmp["legs"][V.leg_key(*NONE_G, 2048)] and "texts_reclassified" not in cmp["gates"]["I5_at_cap"]


# --------------------------------------------------------------------------- M6: instrument facts generated from the legs
def test_the_thinking_pin_as_served_the_shrink_legs_and_the_label_note_are_generated_from_the_admitted_legs(tmp_path):
    G.build_store(tmp_path, row(*NONE_G))
    G.build_store(tmp_path, row(LLAMA, "main", NONE))
    G.build_store(tmp_path, row("nemotron3nano4b", "main", NONE), kv_shrink=G.kv_shrink(256, 240, G.MAX_MODEL_LEN_BY_CAP[2048]))
    rep = analyze(tmp_path, _rows(NONE_G, (LLAMA, "main", NONE), ("nemotron3nano4b", "main", NONE)))
    pins = rep["design"]["thinking_pin_as_served"]
    gem, lla = pins[C.FAMILY_BY_MODEL_KEY["gemma4e4b"]], pins[C.FAMILY_BY_MODEL_KEY[LLAMA]]
    assert gem["switch"] is True and gem["template_kwargs"] == {"enable_thinking": False} and gem["consistent"] and gem["models"] == ["gemma4e4b"]
    assert "switch on the template" in gem["as_served"] and '"enable_thinking": false' in gem["as_served"]
    assert lla["switch"] is False and lla["template_kwargs"] == {} and "no switch: nothing composed" in lla["as_served"]
    shrink = rep["design"]["kv_shrink_legs"]
    assert [s["key"] for s in shrink] == [V.leg_key("nemotron3nano4b", "main", NONE, 2048)] and shrink[0]["configured_seqs"] == 256 and shrink[0]["served_seqs"] == 240
    md = A._instrument_facts_md(rep)
    assert f"| {C.FAMILY_BY_MODEL_KEY['gemma4e4b']} |" in md and "KV-derived served S" in md and "configured 256 -> served 240" in md
    # the label note: legacy one-pin admission vs per-row pins (the label names the run, not every store's tree)
    assert "ONE store pin" in rep["provenance"]["label_note"]
    ctx = A.Context(canonical_ids=None, pins=None, expected_refusals={}, store_pins={"x@2048": "a" * 40})
    note = A.label_note("e43f81f", {"2048": ["a" * 40, "b" * 40], "8192": ["e43f81f2" + "0" * 32]}, ctx)
    assert "NOT the producer tree of every store" in note and "3 producer trees" in note
    # a family whose legs disagree on the pin is filed inconsistent, every variant listed
    G.build_store(tmp_path / "mixed", row(*NONE_G))
    G.build_store(tmp_path / "mixed", row("gemma4e4b", "it", NONE), evidence_edit=lambda ev: ev["serve_header"].update(template_kwargs={}))
    rep2 = analyze(tmp_path / "mixed", _rows(NONE_G, ("gemma4e4b", "it", NONE)))
    g2 = rep2["design"]["thinking_pin_as_served"][C.FAMILY_BY_MODEL_KEY["gemma4e4b"]]
    assert g2["consistent"] is False and len(g2["variants"]) == 2 and "INCONSISTENT" in g2["as_served"]
    assert rep2["gates"]["I3_certification"]["status"] == A.GATE_FAIL                 # the missing pin on a switch family is I3's refusal, as before
    md2 = A.render_results_md(rep)
    assert "### A.7 Fit provenance" in md2 and "### A.8 Instrument facts as served" in md2 and V.FIT_CONDITION_RULE in md2


# --------------------------------------------------------------------------- L7: every plotted endpoint in the sidecar; captions on the W&B images
def test_every_plotted_endpoint_is_in_the_sidecar_with_a_resolvable_source_and_the_wandb_images_carry_captions(llama_report, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    rep = llama_report
    figs = A.make_figures(rep, tmp_path / "figs")
    nums = json.loads(Path(figs["numbers"]).read_text())
    names = {stem: {e["name"] for e in entries} for stem, entries in nums["figures"].items()}
    for stem, entries in nums["figures"].items():
        for e in entries:
            assert A.T.report_path(rep, e["source"]) == e["value"] or e["value"] is None, (stem, e)
    k_it = V.cell_key(LLAMA, "it", LORA)
    assert {f"{k_it}/prompt_strict/merged/ci_low", f"{k_it}/prompt_strict/merged/ci_high", f"{k_it}/prompt_strict/r2048/point", f"{k_it}/prompt_strict/r2048/ci_low"} <= names[f"final_ifeval_contrasts_{LLAMA}"]
    assert {f"{LLAMA}/floor_band/prompt_strict/ci_low", f"{LLAMA}/floor_band/inst_strict/ci_high"} <= names[f"final_ifeval_contrasts_{LLAMA}"]
    assert {f"{k_it}/prompt_loose/merged/ci_low", f"{k_it}/inst_loose/merged/ci_high"} <= names["final_ifeval_metrics_by_cell"]
    assert {f"{k_it}/completion_tokens/p50", f"{k_it}/completion_tokens/p99", f"{k_it}/completion_tokens/mean"} <= names["final_ifeval_completion_tokens"]
    assert {f"{k_it}/share_truncated", f"{k_it}/share_loop"} <= names["final_ifeval_shares"] and {f"{k_it}/final/share_loop", f"{k_it}/final/share_truncated"} <= names["final_ifeval_exits"]
    assert {f"{k_it}/prompt_strict/merged_minus_r2048/ci_low", f"{k_it}/inst_strict/merged_minus_r2048/ci_high"} <= names["final_ifeval_truncation_cost"]
    assert f"{k_it}/residual/n_at_cap" in names["final_ifeval_ladder"]
    assert {f"{k_it}@2048/forwarded", f"{k_it}@2048/recorded_at_cap"} <= names["final_ifeval_ledger"] and f"{k_it}@2048/gpu_h_central_estimate" in names["final_ifeval_cost"]
    assert set(figs["captions"]) == set(figs["written"]) == set(nums["captions"]) and figs["captions"] == nums["captions"]
    # the W&B images carry the captions (a stub wandb module records every Image and log call)
    calls: dict[str, list] = {"images": [], "logged": []}

    class Image:
        def __init__(self, path, caption=None):
            calls["images"].append((path, caption))

    class Run:
        url = "stub://run"
        summary: dict = {}

        def log_artifact(self, art):
            pass

        def finish(self):
            pass

    class Artifact:
        def __init__(self, name, type):
            pass

        def add_dir(self, d):
            pass
    stub = types.SimpleNamespace(Image=Image, Artifact=Artifact, init=lambda **kw: Run(), log=lambda d: calls["logged"].append(d))
    monkeypatch.setitem(sys.modules, "wandb", stub)
    assert A.log_wandb(rep, figs, mode="offline") == "stub://run"
    assert len(calls["images"]) == len(figs["written"]) and all(cap and cap == figs["captions"][Path(p).stem] for p, cap in calls["images"])
    # the trigger provenance in every caption still; the exits caption's condition from the manifest
    assert all("1 % raw at-cap of draws" in c for c in figs["captions"].values()) and "transfer = False" in figs["captions"]["final_ifeval_exits"]


# --------------------------------------------------------------------------- L8: the diagnostics with their labels; visible subset exclusions
def test_the_final_table_carries_labelled_token_and_throughput_columns_and_subset_figures_name_their_exclusions(tmp_path):
    pytest.importorskip("matplotlib")
    held, pending = ("gemma4e4b", "it", LORA), ("gemma4e4b", "it", "alphasteer")
    rows = _rows(("gemma4e4b", "it", NONE), held, pending, NONE_G, ("gemma4e4b", "main", LORA))
    G.build_store(tmp_path, row("gemma4e4b", "it", NONE))
    G.build_store(tmp_path, row(*NONE_G))
    _ceiling_cell(tmp_path, row("gemma4e4b", "main", LORA))
    rep = analyze(tmp_path, rows)
    assert A.non_log_gates_clear(rep), A.production_refusals(rep)                   # no .eval log read here: not production (r3 S1)
    table = A._final_table(rep)
    header = table.splitlines()[0]
    assert "mean / p99 completion tokens (pass-1 leg)" in header and "decode tok/s aggregate (pass-1 leg)" in header and "stock prompt strict ± stderr per trial (pass 1)" in header
    c = rep["cells"][V.cell_key(*NONE_G)]
    assert f"| {c['completion_tokens']['mean']:.0f} / {c['completion_tokens']['p99']} | {c['decode_tok_s_aggregate']:.0f} | logs not read |" in table
    md = A.render_results_md(rep)
    assert "The completion-token and throughput columns are the PASS-1 leg's" in md and "never" in md and "deciding" in md
    # the subset figures: a footnote names the slots not measured and the measured cells the rule leaves out; the captions say the same
    import matplotlib.pyplot as plt
    shown_promoted = [k for k, cc in rep["cells"].items() if cc.get("usable") and cc["ladder"]["promoted"]]
    assert shown_promoted == [V.cell_key("gemma4e4b", "main", LORA)]
    for fn, why in ((A.fig_ladder, "promoted cells only"), (A.fig_truncation_cost, "promoted cells only")):
        fig, _ = fn(rep)
        foot = [t.get_text() for t in fig.texts if t.get_text().startswith("not shown:")]
        assert foot and "1 HELD + 1 PENDING slot(s) not measured" in foot[0] and f"2 measured cell(s) not shown ({why})" in foot[0], (fn.__name__, foot)
        assert A.axes_height_fraction(fig) >= A.AXES_MIN_FRACTION
        plt.close(fig)
    for fn in (A.fig_completion_tokens, A.fig_ledger, A.fig_cost):
        fig, _ = fn(rep)
        foot = [t.get_text() for t in fig.texts if t.get_text().startswith("not shown:")]
        assert foot and "1 HELD + 1 PENDING slot(s) not measured" in foot[0] and "measured cell(s) not shown" not in foot[0], (fn.__name__, foot)
        plt.close(fig)
    caps = A.figure_captions(rep)
    for stem in ("final_ifeval_ladder", "final_ifeval_truncation_cost", "final_ifeval_completion_tokens", "final_ifeval_ledger", "final_ifeval_cost"):
        assert "Not shown: 1 HELD + 1 PENDING slots are not measured and not drawn" in caps[stem], stem
    assert "2 measured cells are not drawn (promoted cells only)" in caps["final_ifeval_ladder"]
    ns = A._not_shown(rep, shown_promoted)
    assert ns == {"held": [V.cell_key(*held)], "pending": [V.cell_key(*pending)],
                  "usable_not_shown": [V.cell_key("gemma4e4b", "it", NONE), V.cell_key(*NONE_G)]}


# --------------------------------------------------------------------------- L9 / the record: the prose filler over a rendered record
def test_the_prose_filler_fills_every_slot_writes_the_revision_block_and_states_the_served_object_finding_from_the_manifests(llama_report, tmp_path):
    rep = llama_report
    rendered = A.render_results_md(rep)
    assert rendered.count("{…}") >= 10
    md = P.fill(rep, rendered, G.FULL_SHA)
    assert md.count("{…}") == 0 and "{{" not in md
    assert "## Revision 2 after review" in md and "## Appendix A" in md and md.index("## Revision 2") < md.index("## Appendix A")
    assert "## Revision 3 after the iterating review" in md and md.index("## Revision 2") < md.index("## Revision 3") < md.index("## Appendix A")
    assert all(item.split(" — ")[0] in md for item in ("H1 —", "M2 —", "M3 —", "M4 —", "M5 —", "M6 —", "L7 —", "L8 —", "L9 —", "S1 (HIGH) —", "S2 —", "S3 —", "S4 —"))
    assert "8. Conditional inference:" in md and "unadjusted for multiplicity" in md and "not a demonstrated equivalence" in md
    # the served-object finding, per cell, from the MANIFESTS — the it-view LoRA is a fit on the served weights, the main a transfer
    k_it, k_main = V.cell_key(LLAMA, "it", LORA), V.cell_key(LLAMA, "main", LORA)
    served_hf = IC.IFEVAL_MODEL_PINS[LLAMA][0]
    assert f"`{k_it}` is an INSTRUCT-weight document-fit LoRA served on {served_hf}: `fit_provenance.model_id = {served_hf}` = `served_model_id`, `transfer = false`, `chat_template_bypassed = true`" in md
    assert f"`{k_main}` is a BASE-weight document-fit LoRA served on {served_hf}: `fit_provenance.model_id = {G.BASE_FIT_PINS[LLAMA][0]}` vs `served_model_id = {served_hf}`, `transfer = true`" in md
    # the reading is GENERATED by fit condition from the exit + loop fields, never the r2 binary sentence
    assert "the base-fit transfers on `main` do not" not in P.served_object_paragraph(rep)
    assert f"Reading for {LLAMA}, by fit condition, from each cell's final exit and loop fields" in md
    # the r1 sentences are gone from the HEADLINE and the results sections (the revision block and the incidents list QUOTE them as history)
    body = md[:md.index("## Revision 2 after review")]
    for section in ("## Headline", "## 4. Results", "## 9. Limitations"):
        assert section in body
    head = body[body.index("## Headline"):body.index("## 1. What was asked")]
    assert "base-fit objects served on the instruct target" not in head and "transfer objects (they loop)" not in head and "cannot finish" not in head
    results = body[body.index("## 4. Results"):body.index("## 5. Gates")]
    assert "loops the re-draw cannot finish" not in results and "bought back" not in results and "observed net change" in results.lower()
    assert "Loops (degenerate at-cap texts by the whitespace 4-gram rule) are never re-drawn by the protocol" in md
    # the as-run facts generated from the legs: the pin AS SERVED per family, the label row, the shrink sentence, the slot claim
    assert "thinking-OFF pin AS SERVED (generated from the admitted legs' records; Appendix A.8)" in md
    assert "| Llama-3.1-8B-Instruct |" in md and "no switch: nothing composed (template_kwargs {}) (read off" in md
    assert "| gemma-4-E4B-it |" in md and "no admitted leg of this model in the report" in md          # no gemma leg in this fixture: stated, not typed
    assert "| record label — the record label `" in md and "is the ONE store pin every admitted leg's producer tree carries" in md and "no leg served a KV-derived S" in md
    assert "The held and pending rows appear as slots, never as measured, in every PER-CELL table" in md and "the SUBSET figures" in md
    assert "Two scopes (revision 2)" in md and "variability of the whole adaptive protocol" in md
    # nothing machine-specific in the record
    assert str(Path.home()) not in md
    # the CLI round trip writes the filled record and reports zero slots
    out = tmp_path / "results.md"
    out.write_text(rendered, encoding="utf-8")
    rep_path = tmp_path / "r.json"
    rep_path.write_text(json.dumps(rep, default=str), encoding="utf-8")
    assert P.main([str(rep_path), str(out), G.FULL_SHA]) == 0 and out.read_text(encoding="utf-8").count("{…}") == 0
    # the §5 I9 note is written from the second venv's match report when given (never typed)
    i9 = {"report_match": {"shared_numeric": 123, "differing": [], "type_changed": [], "missing": [], "added": [], "equal": True},
          "gates": {"I9_report_match": {"status": A.GATE_PASS}}}
    md_i9 = P.fill(rep, rendered, G.FULL_SHA, i9=i9)
    assert "## 5. Gates\n\nI9 (the two-venv numeric-leaf match) reads PENDING in the table below" in md_i9 and "**PASSED**: 123 shared numeric leaves equal, 0 differing, 0 missing, 0 added" in md_i9
    assert "I9 (the two-venv numeric-leaf match) reads PENDING" not in md          # without the match file the note is absent, not invented
    # a filler pointed at a record whose slot was already consumed refuses by name (never a silent second fill)
    with pytest.raises(SystemExit, match="expected exactly once"):
        P.fill(rep, out.read_text(encoding="utf-8"), G.FULL_SHA)
